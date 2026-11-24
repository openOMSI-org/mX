//! The bindless heap: argument buffers of `MTLResourceID`s (textures, top-level acceleration
//! structures, samplers) and of GPU addresses (buffer views), slot allocation with deferred
//! reuse, and the residency list the encoders hand to `useResources`.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2_metal::*;

use crate::{Error, Mt, Po, Result};

/// One argument buffer and its slots.
struct Slots {
    buffer: Mt<Po<dyn MTLBuffer>>,
    capacity: u32,
    next: u32,
    free: Vec<u32>,
    /// Released slots and the submission after whose completion they may be reused.
    deferred: Vec<(u64, u32)>,
}

impl Slots {
    fn new(device: &ProtocolObject<dyn MTLDevice>, capacity: u32, label: &str) -> Result<Slots> {
        let bytes = capacity as usize * 8;
        let buffer = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| Error::OutOfMemory(label.into()))?;
        buffer.setLabel(Some(&crate::ns(label)));
        // (empty slots read as a null resource: undefined to sample, but never a stale one)
        // SAFETY: the buffer is `bytes` long and nothing uses it yet.
        unsafe { std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, bytes) };
        Ok(Slots {
            buffer: Mt(buffer),
            capacity,
            next: 0,
            free: Vec::new(),
            deferred: Vec::new(),
        })
    }

    fn alloc(&mut self, what: &'static str) -> Result<u32> {
        if let Some(s) = self.free.pop() {
            return Ok(s);
        }
        if self.next == self.capacity {
            return Err(Error::HeapFull(what));
        }
        self.next += 1;
        Ok(self.next - 1)
    }

    fn write(&self, slot: u32, id: MTLResourceID) {
        // SAFETY: `MTLResourceID` is `#[repr(C)]` around one `u64`, the layout an argument
        // buffer of `texture2d`/`sampler` entries has on Tier 2 (Metal 3) devices.
        let raw: u64 = unsafe { std::mem::transmute(id) };
        self.write_raw(slot, raw);
    }

    fn write_raw(&self, slot: u32, raw: u64) {
        debug_assert!(slot < self.capacity);
        // SAFETY: the slot is inside the buffer; the GPU reads only slots in use, and this one
        // is not (fresh or past its deferred release).
        unsafe { self.buffer.contents().as_ptr().cast::<u64>().add(slot as usize).write(raw) };
    }

    fn reclaim(&mut self, completed: u64) {
        let free = &mut self.free;
        self.deferred.retain(|&(after, slot)| {
            let done = after <= completed;
            if done {
                free.push(slot);
            }
            !done
        });
    }
}

/// A list of resources for `useResources`.
type ResourceList = Vec<NonNull<ProtocolObject<dyn MTLResource>>>;

/// A buffer in the heap: the buffer, how many views it has there, how many of them writable.
type HeapBuffer = (Mt<Po<dyn MTLBuffer>>, u32, u32);

/// A buffer view in the heap.
struct BufferView {
    buffer: Mt<Po<dyn MTLBuffer>>,
    /// Where the view starts in the buffer.
    offset: u64,
    /// The buffer's identity (its `Arc` address).
    owner: usize,
    writable: bool,
}

/// The heap's argument buffers: textures, samplers, buffers.
pub(crate) struct Tables([Mt<Po<dyn MTLBuffer>>; 3]);

impl Tables {
    pub fn get(&self, table: mx_shader::msl::HeapTable) -> &ProtocolObject<dyn MTLBuffer> {
        use mx_shader::msl::HeapTable as H;
        &self.0[match table {
            H::Textures => 0,
            H::Samplers => 1,
            H::Buffers => 2,
        }]
    }
}

pub(crate) struct Heap {
    textures: Slots,
    samplers: Slots,
    /// The samplers by heap slot, for pipelines that bind them directly.
    sampler_objects: HashMap<u32, Mt<Po<dyn MTLSamplerState>>>,
    /// GPU addresses (`device T*`, naga's layout for a `binding_array` of buffers).
    buffers: Slots,
    buffer_views: HashMap<u32, BufferView>,
    /// The heap's views of each buffer.
    buffer_views_of: HashMap<usize, Vec<u32>>,
    /// Buffers released but maybe still read by the GPU, kept alive until then.
    retired_buffers: Vec<(u64, Mt<Po<dyn MTLBuffer>>)>,
    /// Buffers in the heap, by owner, with how many views and how many writable ones each has
    /// (a buffer is declared once whatever its views).
    heap_buffers: HashMap<usize, HeapBuffer>,
    /// Buffers copied into, declared until the submission that wrote them is done.
    recent_buffers: Vec<(u64, Mt<Po<dyn MTLBuffer>>)>,
    /// The textures in the heap, dense, for `useResources`.
    resident: Vec<Mt<Po<dyn MTLTexture>>>,
    /// Per texture slot: its place in `resident` (`u32::MAX`: none).
    dense_of: Vec<u32>,
    /// Per place in `resident`: its slot.
    slot_of: Vec<u32>,
    /// Textures released but maybe still read by the GPU, kept alive until then.
    retired: Vec<(u64, Mt<Po<dyn MTLTexture>>)>,
    /// Bumped whenever what encoders must declare changes, so they call `useResources` again.
    pub generation: u64,
    /// macOS 15+: every heap texture is in this residency set, which the queue keeps
    /// resident; encoders then declare (`useResources`) only the textures the GPU may be
    /// writing, which Metal's hazard tracking needs to know about. Without it (macOS 13 and
    /// 14) every encoder declares the whole heap, which costs more with every texture.
    set: Option<Mt<Po<dyn MTLResidencySet>>>,
    set_dirty: bool,
    /// Per slot: the texture the view belongs to (its `Arc` address) and whether passes may
    /// write it (render targets, storage textures, textures made outside mX).
    owner: HashMap<u32, (usize, bool)>,
    /// The heap's views of each texture.
    views_of: HashMap<usize, Vec<u32>>,
    /// Views passes may write: declared by every encoder.
    hot: HashMap<u32, Mt<Po<dyn MTLTexture>>>,
    /// Views of textures with `STORAGE_BINDING`, which shaders may write through the heap:
    /// declared for writing by every encoder.
    storage: HashMap<u32, Mt<Po<dyn MTLTexture>>>,
    /// Top-level acceleration structures in the texture table, declared (with the bottom-level
    /// structures of their last build) by every encoder.
    tlases: HashMap<u32, Arc<crate::rt::TlasShared>>,
    /// Acceleration structures released but maybe still read by the GPU, kept alive until then.
    retired_tlases: Vec<(u64, Arc<crate::rt::TlasShared>)>,
    /// Views of textures copied into, declared until the submission that wrote them is done.
    recent: Vec<(u64, Mt<Po<dyn MTLTexture>>)>,
    /// Resources (by owner) written by a submission not yet done: a view put in the heap after
    /// the write (`create_buffer_init`, `write_texture`, then `create_view`) is declared too.
    writes: HashMap<usize, u64>,
}

impl Heap {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, textures: u32, samplers: u32, buffers: u32) -> Result<Heap> {
        Ok(Heap {
            textures: Slots::new(device, textures, "mX texture heap")?,
            samplers: Slots::new(device, samplers, "mX sampler heap")?,
            sampler_objects: HashMap::new(),
            buffers: Slots::new(device, buffers, "mX buffer heap")?,
            buffer_views: HashMap::new(),
            buffer_views_of: HashMap::new(),
            retired_buffers: Vec::new(),
            heap_buffers: HashMap::new(),
            recent_buffers: Vec::new(),
            resident: Vec::new(),
            dense_of: vec![u32::MAX; textures as usize],
            slot_of: Vec::new(),
            retired: Vec::new(),
            generation: 1,
            set: residency_set(device),
            set_dirty: false,
            owner: HashMap::new(),
            views_of: HashMap::new(),
            hot: HashMap::new(),
            storage: HashMap::new(),
            tlases: HashMap::new(),
            retired_tlases: Vec::new(),
            recent: Vec::new(),
            writes: HashMap::new(),
        })
    }

    /// Whether the heap is kept resident by a residency set (macOS 15+).
    pub fn residency_set(&self) -> Option<&ProtocolObject<dyn MTLResidencySet>> {
        self.set.as_deref().map(|s| &**s)
    }

    /// Put view `tex` of texture `owner` (its `Arc` address) in the heap. `hot`: passes
    /// may write the texture; `storage`: shaders may write it through the heap.
    pub fn add_texture(&mut self, tex: &Po<dyn MTLTexture>, owner: usize, hot: bool, storage: bool) -> Result<u32> {
        let slot = self.textures.alloc("textures")?;
        self.textures.write(slot, tex.gpuResourceID());
        self.dense_of[slot as usize] = self.resident.len() as u32;
        self.slot_of.push(slot);
        self.resident.push(Mt(tex.clone()));
        self.owner.insert(slot, (owner, hot));
        self.views_of.entry(owner).or_default().push(slot);
        if storage {
            self.storage.insert(slot, Mt(tex.clone()));
        } else if hot {
            self.hot.insert(slot, Mt(tex.clone()));
        } else if let Some(&until) = self.writes.get(&owner) {
            self.recent.push((until, Mt(tex.clone())));
        }
        if let Some(set) = &self.set {
            set.addAllocation(ProtocolObject::from_ref(&**tex));
            self.set_dirty = true;
        }
        self.generation += 1;
        Ok(slot)
    }

    /// Texture `owner` was written by a copy (or zeroed) in submission `until`: encoders
    /// declare its views until that submission is done.
    pub fn written(&mut self, owner: usize, until: u64) {
        if self.set.is_none() {
            return;
        }
        let w = self.writes.entry(owner).or_insert(until);
        *w = (*w).max(until);
        if let Some((buffer, _, writable)) = self.heap_buffers.get(&owner) {
            // (writable buffers are declared by every encoder anyway)
            if *writable == 0 {
                self.recent_buffers.push((until, buffer.clone()));
                self.generation += 1;
            }
            return;
        }
        let Some(slots) = self.views_of.get(&owner) else { return };
        for &slot in slots {
            if self.owner.get(&slot).is_some_and(|(_, hot)| !hot) {
                let dense = self.dense_of[slot as usize] as usize;
                self.recent.push((until, self.resident[dense].clone()));
            }
        }
        self.generation += 1;
    }

    /// Changes to the residency set take effect for command buffers committed after this.
    pub fn commit_residency(&mut self) {
        if let (Some(set), true) = (&self.set, self.set_dirty) {
            set.commit();
            self.set_dirty = false;
        }
    }

    /// Take a texture out of the heap. Its slot is reused once submission `after` is done.
    pub fn remove_texture(&mut self, slot: u32, after: u64) {
        let dense = std::mem::replace(&mut self.dense_of[slot as usize], u32::MAX) as usize;
        let tex = self.resident.swap_remove(dense);
        self.slot_of.swap_remove(dense);
        if let Some(&moved) = self.slot_of.get(dense) {
            self.dense_of[moved as usize] = dense as u32;
        }
        if let Some((owner, _)) = self.owner.remove(&slot)
            && let Some(v) = self.views_of.get_mut(&owner)
        {
            v.retain(|s| *s != slot);
            if v.is_empty() {
                self.views_of.remove(&owner);
            }
        }
        self.hot.remove(&slot);
        self.storage.remove(&slot);
        self.retired.push((after, tex));
        self.textures.deferred.push((after, slot));
        self.generation += 1;
    }

    /// Put a top-level acceleration structure in the texture table (`Tlas::index`).
    pub fn add_tlas(&mut self, tlas: &Arc<crate::rt::TlasShared>) -> Result<u32> {
        let slot = self.textures.alloc("textures")?;
        self.textures.write(slot, tlas.raw.gpuResourceID());
        self.tlases.insert(slot, tlas.clone());
        self.generation += 1;
        Ok(slot)
    }

    /// Take a top-level acceleration structure out of the heap. Its slot is reused once
    /// submission `after` is done.
    pub fn remove_tlas(&mut self, slot: u32, after: u64) {
        if let Some(t) = self.tlases.remove(&slot) {
            self.retired_tlases.push((after, t));
        }
        self.textures.deferred.push((after, slot));
        self.generation += 1;
    }

    /// The top-level acceleration structure at heap `slot` (for a pipeline that binds it
    /// directly).
    pub fn tlas_at(&self, slot: u32) -> Option<&Arc<crate::rt::TlasShared>> {
        self.tlases.get(&slot)
    }

    pub fn add_sampler(&mut self, s: &Po<dyn MTLSamplerState>) -> Result<u32> {
        let slot = self.samplers.alloc("samplers")?;
        self.samplers.write(slot, s.gpuResourceID());
        self.sampler_objects.insert(slot, Mt(s.clone()));
        Ok(slot)
    }

    /// The texture view at heap `slot` (for a pipeline that binds it directly).
    pub fn texture_at(&self, slot: u32) -> Option<&Po<dyn MTLTexture>> {
        let dense = *self.dense_of.get(slot as usize)?;
        self.resident.get(dense as usize).map(|t| &t.0)
    }

    /// The buffer view at heap `slot`: its buffer and where the view starts (the bound tier
    /// binds it directly).
    pub fn buffer_at(&self, slot: u32) -> Option<(&Po<dyn MTLBuffer>, u64)> {
        self.buffer_views.get(&slot).map(|v| (&v.buffer.0, v.offset))
    }

    /// The sampler at heap `slot`.
    pub fn sampler_at(&self, slot: u32) -> Option<&Po<dyn MTLSamplerState>> {
        self.sampler_objects.get(&slot).map(|s| &s.0)
    }

    pub fn remove_sampler(&mut self, slot: u32, after: u64) {
        self.sampler_objects.remove(&slot);
        self.samplers.deferred.push((after, slot));
    }

    /// Put `size` bytes of `buffer` (identity `owner`) from `offset` in the heap.
    /// `writable`: shaders may write them, so every encoder declares the buffer for writing.
    pub fn add_buffer(&mut self, buffer: &Po<dyn MTLBuffer>, owner: usize, offset: u64, writable: bool) -> Result<u32> {
        let slot = self.buffers.alloc("buffers")?;
        self.buffers.write_raw(slot, buffer.gpuAddress() + offset);
        self.buffer_views.insert(
            slot,
            BufferView {
                buffer: Mt(buffer.clone()),
                offset,
                owner,
                writable,
            },
        );
        self.buffer_views_of.entry(owner).or_default().push(slot);
        let entry = self.heap_buffers.entry(owner).or_insert_with(|| (Mt(buffer.clone()), 0, 0));
        entry.1 += 1;
        entry.2 += writable as u32;
        if entry.1 == 1
            && !writable
            && let Some(&until) = self.writes.get(&owner)
        {
            self.recent_buffers.push((until, Mt(buffer.clone())));
        }
        if entry.1 == 1
            && let Some(set) = &self.set
        {
            set.addAllocation(ProtocolObject::from_ref(&**buffer));
            self.set_dirty = true;
        }
        self.generation += 1;
        Ok(slot)
    }

    /// Take a buffer view out of the heap. Its slot is reused once submission `after` is done.
    pub fn remove_buffer(&mut self, slot: u32, after: u64) {
        let Some(view) = self.buffer_views.remove(&slot) else { return };
        if let Some(v) = self.buffer_views_of.get_mut(&view.owner) {
            v.retain(|s| *s != slot);
            if v.is_empty() {
                self.buffer_views_of.remove(&view.owner);
            }
        }
        if let Some(entry) = self.heap_buffers.get_mut(&view.owner) {
            entry.1 -= 1;
            entry.2 -= view.writable as u32;
            if entry.1 == 0 {
                self.heap_buffers.remove(&view.owner);
                // (out of the residency set once the GPU is done with it, in `reclaim`)
                self.retired_buffers.push((after, view.buffer));
            }
        }
        self.buffers.deferred.push((after, slot));
        self.generation += 1;
    }

    pub fn buffers_in_use(&self) -> usize {
        self.buffer_views.len()
    }

    /// Everything released before submission `completed` finished is free again.
    pub fn reclaim(&mut self, completed: u64) {
        self.textures.reclaim(completed);
        self.samplers.reclaim(completed);
        self.buffers.reclaim(completed);
        let set = &self.set;
        let mut removed = false;
        let heap_buffers = &self.heap_buffers;
        self.retired_buffers.retain(|(after, buffer)| {
            let keep = *after > completed;
            // (a buffer that went out and came back in keeps its place in the set)
            let back = heap_buffers.values().any(|(b, _, _)| std::ptr::eq(&***b, &***buffer));
            if !keep
                && !back
                && let Some(set) = set
            {
                set.removeAllocation(ProtocolObject::from_ref(&***buffer));
                removed = true;
            }
            keep
        });
        let before = self.recent_buffers.len();
        self.recent_buffers.retain(|(until, _)| *until > completed);
        if self.recent_buffers.len() != before {
            self.generation += 1;
        }
        self.retired.retain(|(after, tex)| {
            let keep = *after > completed;
            if !keep && let Some(set) = set {
                set.removeAllocation(ProtocolObject::from_ref(&***tex));
                removed = true;
            }
            keep
        });
        self.set_dirty |= removed;
        self.retired_tlases.retain(|(after, _)| *after > completed);
        self.writes.retain(|_, until| *until > completed);
        let before = self.recent.len();
        self.recent.retain(|(until, _)| *until > completed);
        if self.recent.len() != before {
            self.generation += 1;
        }
    }

    /// The three argument buffers, for binding where each pipeline wants them.
    pub fn tables(&self) -> Option<Tables> {
        Some(Tables([self.textures.buffer.clone(), self.samplers.buffer.clone(), self.buffers.buffer.clone()]))
    }

    /// Declare the heap's textures to a render encoder: with a residency set, those passes
    /// may write (and the views of `written`, textures this command buffer copied into);
    /// without, all of them.
    pub fn use_on_render(&self, enc: &ProtocolObject<dyn MTLRenderCommandEncoder>, written: &[usize]) {
        let stages = MTLRenderStages::Vertex | MTLRenderStages::Fragment;
        let (read, write) = self.resource_lists(written);
        for (list, usage) in [(read, MTLResourceUsage::Read), (write, MTLResourceUsage::Read | MTLResourceUsage::Write)] {
            if let Some(ptr) = NonNull::new(list.as_ptr().cast_mut()) {
                // SAFETY: `list` holds valid resources kept alive by the heap (the caller
                // holds the heap lock).
                unsafe { enc.useResources_count_usage_stages(ptr, list.len(), usage, stages) };
            }
        }
    }

    pub fn use_on_compute(&self, enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, written: &[usize]) {
        let (read, write) = self.resource_lists(written);
        for (list, usage) in [(read, MTLResourceUsage::Read), (write, MTLResourceUsage::Read | MTLResourceUsage::Write)] {
            if let Some(ptr) = NonNull::new(list.as_ptr().cast_mut()) {
                // SAFETY: as in `use_on_render`.
                unsafe { enc.useResources_count_usage(ptr, list.len(), usage) };
            }
        }
    }

    /// What an encoder declares: resources it reads, and buffers shaders may write.
    fn resource_lists(&self, written: &[usize]) -> (ResourceList, ResourceList) {
        let buf = |b: &Mt<Po<dyn MTLBuffer>>| NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&*b.0));
        let tex = |t: &Mt<Po<dyn MTLTexture>>| NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&*t.0));
        let accel = |a: &Po<dyn MTLAccelerationStructure>| NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&**a));
        let mut read = self.resource_list(written);
        let mut write: ResourceList = self.storage.values().map(tex).collect();
        for t in self.tlases.values() {
            read.push(accel(&t.raw));
            read.extend(t.built_with.lock().iter().map(|b| accel(&b.0.raw)));
        }
        for (owner, (b, _, writable)) in &self.heap_buffers {
            if *writable > 0 {
                write.push(buf(b));
            } else if self.set.is_none() || written.contains(owner) {
                read.push(buf(b));
            }
        }
        if self.set.is_some() {
            read.extend(self.recent_buffers.iter().map(|(_, b)| buf(b)));
        }
        (read, write)
    }

    fn resource_list(&self, written: &[usize]) -> Vec<NonNull<ProtocolObject<dyn MTLResource>>> {
        let ptr = |t: &Mt<Po<dyn MTLTexture>>| NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&*t.0));
        if self.set.is_none() {
            // (storage textures are declared for writing, in `resource_lists`)
            return self.resident.iter().zip(&self.slot_of).filter(|(_, s)| !self.storage.contains_key(s)).map(|(t, _)| ptr(t)).collect();
        }
        let mut list: Vec<_> = self.hot.values().map(ptr).chain(self.recent.iter().map(|(_, t)| ptr(t))).collect();
        for owner in written {
            for &slot in self.views_of.get(owner).into_iter().flatten().filter(|s| !self.storage.contains_key(s)) {
                list.push(ptr(&self.resident[self.dense_of[slot as usize] as usize]));
            }
        }
        list
    }

    pub fn textures_in_use(&self) -> usize {
        self.resident.len()
    }
}

/// A residency set for the heap where Metal has them (macOS 15+), unless
/// `MX_NO_RESIDENCY_SET` is set.
fn residency_set(device: &ProtocolObject<dyn MTLDevice>) -> Option<Mt<Po<dyn MTLResidencySet>>> {
    if std::env::var_os("MX_NO_RESIDENCY_SET").is_some() || !device.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
        return None;
    }
    let d = MTLResidencySetDescriptor::new();
    d.setLabel(Some(&crate::ns("mX heap")));
    match device.newResidencySetWithDescriptor_error(&d) {
        Ok(set) => Some(Mt(set)),
        Err(e) => {
            log::warn!("mX: no residency set ({}); the heap is declared in every pass", e.localizedDescription());
            None
        }
    }
}

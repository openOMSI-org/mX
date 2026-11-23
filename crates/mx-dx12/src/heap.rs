//! Descriptor heaps: the shader-visible heaps (bindless slots, the sampler index buffer of the
//! heap group, the ring root tables are copied into) and the CPU-only pools views and bind
//! groups keep their descriptors in.
//!
//! Shader-visible CBV/SRV/UAV heap:
//!
//! | descriptors | use |
//! |-------------|-----|
//! | `0 .. T` | bindless texture slots (`TextureView::index`) |
//! | `T` | the heap group's sampler index buffer (the identity: entry `i` holds `i`) |
//! | `T + 1 .. T + 1 + B` | bindless buffer slots (`BufferView::index`) as raw SRVs |
//! | `T + 1 + B .. T + 1 + 2B` | the same slots as raw UAVs (null for views made read-only) |
//! | `T + 1 + 2B ..` | blocks of [`BLOCK`] descriptors for root tables, one encoder at a time |

use std::collections::HashMap;

use windows::Win32::Graphics::Direct3D12::*;

use crate::track::Track;
use crate::{Buffer, Error, Result, Texture, hr_err};

/// Descriptors per root-table block.
pub const BLOCK: u32 = 1024;
/// Root-table blocks in the shader-visible heap.
pub const BLOCKS: u32 = 128;
/// D3D12's largest shader-visible CBV/SRV/UAV heap on resource binding tiers 1 and 2.
pub const MAX_VIEW_HEAP: u32 = 1_000_000;

/// One allocator of slots with deferred reuse.
struct Slots {
    capacity: u32,
    next: u32,
    free: Vec<u32>,
    /// Released slots and the submission after whose completion they may be reused.
    deferred: Vec<(u64, u32)>,
    in_use: usize,
}

impl Slots {
    fn new(capacity: u32) -> Slots {
        Slots {
            capacity,
            next: 0,
            free: Vec::new(),
            deferred: Vec::new(),
            in_use: 0,
        }
    }

    fn alloc(&mut self, what: &'static str) -> Result<u32> {
        let s = match self.free.pop() {
            Some(s) => s,
            None if self.next == self.capacity => return Err(Error::HeapFull(what)),
            None => {
                self.next += 1;
                self.next - 1
            }
        };
        self.in_use += 1;
        Ok(s)
    }

    fn release(&mut self, slot: u32, after: u64) {
        self.in_use -= 1;
        self.deferred.push((after, slot));
    }

    /// Slots released before `completed` are free again; they are also added to `freed`.
    fn reclaim(&mut self, completed: u64, freed: &mut Vec<u32>) {
        let free = &mut self.free;
        self.deferred.retain(|&(after, slot)| {
            let done = after <= completed;
            if done {
                free.push(slot);
                freed.push(slot);
            }
            !done
        });
    }
}

/// The two shader-visible heaps. Immutable after creation; descriptors are written into them
/// through their CPU handles (slots no submission can be reading).
pub(crate) struct Views {
    pub view_heap: ID3D12DescriptorHeap,
    pub view_cpu: usize,
    pub view_gpu: u64,
    pub view_inc: u32,
    pub sampler_heap: ID3D12DescriptorHeap,
    pub sampler_cpu: usize,
    pub sampler_gpu: u64,
    pub sampler_inc: u32,
    /// Bindless texture slots (`T`).
    pub textures: u32,
    /// Bindless buffer slots (`B`).
    pub buffers: u32,
}

impl Views {
    pub fn new(device: &ID3D12Device, textures: u32, buffers: u32) -> Result<Views> {
        let heap = |ty, n: u32, what: &str| -> Result<ID3D12DescriptorHeap> {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: ty,
                NumDescriptors: n,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: a plain creation from a filled description.
            let h: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| hr_err(what, e))?;
            crate::set_name(&h, what);
            Ok(h)
        };
        let view_heap = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, textures + 1 + 2 * buffers + BLOCK * BLOCKS, "mX view heap")?;
        let sampler_heap = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, mx_shader::hlsl::SAMPLER_HEAP_SIZE, "mX sampler heap")?;
        // SAFETY: plain getters on live objects.
        unsafe {
            Ok(Views {
                view_cpu: view_heap.GetCPUDescriptorHandleForHeapStart().ptr,
                view_gpu: view_heap.GetGPUDescriptorHandleForHeapStart().ptr,
                view_inc: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                sampler_cpu: sampler_heap.GetCPUDescriptorHandleForHeapStart().ptr,
                sampler_gpu: sampler_heap.GetGPUDescriptorHandleForHeapStart().ptr,
                sampler_inc: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER),
                view_heap,
                sampler_heap,
                textures,
                buffers,
            })
        }
    }

    pub fn view_cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.view_cpu + index as usize * self.view_inc as usize,
        }
    }

    pub fn view_gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.view_gpu + index as u64 * self.view_inc as u64,
        }
    }

    pub fn sampler_cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.sampler_cpu + index as usize * self.sampler_inc as usize,
        }
    }

    /// The heap group's sampler index buffer descriptor.
    pub fn sampler_index_slot(&self) -> u32 {
        self.textures
    }

    /// The raw SRV of buffer slot `slot`.
    pub fn buffer_srv(&self, slot: u32) -> u32 {
        self.textures + 1 + slot
    }

    /// Null descriptors into texture slots `tex` and buffer slots `buf` (both their raw SRV and
    /// UAV). A slot no view holds kept whatever was there: nothing, on a new heap, or the view of
    /// a texture or buffer since destroyed. A shader that reached one through a stale or wrong
    /// index read freed memory, and the GPU faulted - Direct3D 12 removed the device (some
    /// drivers, some machines). A null descriptor reads zeros.
    pub fn write_nulls(&self, device: &ID3D12Device, tex: impl IntoIterator<Item = u32>, buf: impl IntoIterator<Item = u32>) {
        let tex2d = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 1,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: 1,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: null descriptors (no resource) into slots no submission reads: a new heap, or
        // slots whose release the queue has passed.
        unsafe {
            for s in tex {
                device.CreateShaderResourceView(None, Some(&tex2d), self.view_cpu(s));
            }
            for s in buf {
                device.CreateShaderResourceView(None, Some(&srv), self.view_cpu(self.buffer_srv(s)));
                device.CreateUnorderedAccessView(None, None, Some(&uav), self.view_cpu(self.buffer_uav(s)));
            }
        }
    }

    /// The raw UAV of buffer slot `slot`.
    pub fn buffer_uav(&self, slot: u32) -> u32 {
        self.textures + 1 + self.buffers + slot
    }

    /// The first descriptor of root-table block `b`.
    pub fn block_start(&self, b: u32) -> u32 {
        self.textures + 1 + 2 * self.buffers + b * BLOCK
    }
}

/// Slot bookkeeping of the bindless heap and the root-table blocks.
pub(crate) struct Heap {
    textures: Slots,
    samplers: Slots,
    buffers: Slots,
    /// Textures whose heap view was released but which a submission may still read.
    retired: Vec<(u64, Texture)>,
    /// Buffers likewise.
    retired_buffers: Vec<(u64, Buffer)>,
    /// The buffers with views in the heap (by resource), with how many views and how many
    /// writable ones: a command list keeps them in the state shaders reach them in (see
    /// `Heap::buffer_states`).
    heap_buffers: HashMap<usize, (Track, u32, u32)>,
    /// Bumped whenever `heap_buffers` changes.
    pub buffer_generation: u64,
    free_blocks: Vec<u32>,
}

impl Heap {
    pub fn new(textures: u32, samplers: u32, buffers: u32) -> Heap {
        Heap {
            textures: Slots::new(textures),
            samplers: Slots::new(samplers),
            buffers: Slots::new(buffers),
            retired: Vec::new(),
            retired_buffers: Vec::new(),
            heap_buffers: HashMap::new(),
            buffer_generation: 1,
            free_blocks: (0..BLOCKS).rev().collect(),
        }
    }

    /// A slot for a view of `buffer`. `writable`: shaders may write it through the heap.
    pub fn alloc_buffer(&mut self, buffer: &Buffer, writable: bool) -> Result<u32> {
        let slot = self.buffers.alloc("buffers")?;
        let e = self.heap_buffers.entry(buffer.key()).or_insert_with(|| (buffer.0.track.clone(), 0, 0));
        e.1 += 1;
        e.2 += writable as u32;
        self.buffer_generation += 1;
        Ok(slot)
    }

    /// Release a buffer slot (reused after submission `after` is done) and keep the buffer
    /// alive until then.
    pub fn release_buffer(&mut self, slot: u32, after: u64, buffer: Buffer, writable: bool) {
        self.buffers.release(slot, after);
        if let Some(e) = self.heap_buffers.get_mut(&buffer.key()) {
            e.1 -= 1;
            e.2 -= writable as u32;
            if e.1 == 0 {
                self.heap_buffers.remove(&buffer.key());
            }
        }
        self.buffer_generation += 1;
        self.retired_buffers.push((after, buffer));
    }

    /// The state each buffer with heap views is in while shaders may reach it through the
    /// heap: `UNORDERED_ACCESS` with a writable view, else readable by every shader stage.
    pub fn buffer_states(&self) -> impl Iterator<Item = (&Track, D3D12_RESOURCE_STATES)> {
        self.heap_buffers.values().map(|(track, _, writable)| {
            let state = if *writable > 0 {
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            };
            (track, state)
        })
    }

    pub fn buffers_in_use(&self) -> usize {
        self.buffers.in_use
    }

    pub fn alloc_texture(&mut self) -> Result<u32> {
        self.textures.alloc("textures")
    }

    /// Release a texture slot (reused after submission `after` is done) and keep the texture
    /// alive until then.
    pub fn release_texture(&mut self, slot: u32, after: u64, texture: Texture) {
        self.textures.release(slot, after);
        self.retired.push((after, texture));
    }

    pub fn alloc_sampler(&mut self) -> Result<u32> {
        self.samplers.alloc("samplers")
    }

    pub fn release_sampler(&mut self, slot: u32, after: u64) {
        self.samplers.release(slot, after);
    }

    pub fn alloc_block(&mut self) -> Option<u32> {
        self.free_blocks.pop()
    }

    /// Blocks whose command lists finished (or were never submitted).
    pub fn free_blocks(&mut self, blocks: impl IntoIterator<Item = u32>) {
        self.free_blocks.extend(blocks);
    }

    /// Everything released before submission `completed` finished is free again. Returns the
    /// textures to drop outside the lock (buffers are dropped here: dropping one takes no lock)
    /// and the texture and buffer slots freed, whose descriptors still point at what they held.
    pub fn reclaim(&mut self, completed: u64) -> (Vec<Texture>, Vec<u32>, Vec<u32>) {
        let (mut tex_slots, mut buf_slots, mut sampler_slots) = (Vec::new(), Vec::new(), Vec::new());
        self.textures.reclaim(completed, &mut tex_slots);
        self.samplers.reclaim(completed, &mut sampler_slots);
        self.buffers.reclaim(completed, &mut buf_slots);
        self.retired_buffers.retain(|(after, _)| *after > completed);
        let mut done = Vec::new();
        self.retired.retain_mut(|(after, t)| {
            if *after <= completed {
                done.push(t.clone());
                false
            } else {
                true
            }
        });
        (done, tex_slots, buf_slots)
    }

    pub fn textures_in_use(&self) -> usize {
        self.textures.in_use
    }
}

/// A CPU-only descriptor (in a [`CpuPool`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cpu(pub usize);

impl Cpu {
    pub fn handle(self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.0 }
    }
}

/// Non-shader-visible descriptors of one type, in chunks that are never freed.
pub(crate) struct CpuPool {
    ty: D3D12_DESCRIPTOR_HEAP_TYPE,
    inc: usize,
    chunks: Vec<ID3D12DescriptorHeap>,
    free: Vec<usize>,
}

const CHUNK: u32 = 256;

impl CpuPool {
    pub fn new(device: &ID3D12Device, ty: D3D12_DESCRIPTOR_HEAP_TYPE) -> CpuPool {
        CpuPool {
            ty,
            // SAFETY: a plain getter.
            inc: unsafe { device.GetDescriptorHandleIncrementSize(ty) } as usize,
            chunks: Vec::new(),
            free: Vec::new(),
        }
    }

    pub fn alloc(&mut self, device: &ID3D12Device) -> Result<Cpu> {
        if let Some(p) = self.free.pop() {
            return Ok(Cpu(p));
        }
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: self.ty,
            NumDescriptors: CHUNK,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        // SAFETY: a plain creation from a filled description.
        let h: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| hr_err("CPU descriptor heap", e))?;
        // SAFETY: a plain getter.
        let start = unsafe { h.GetCPUDescriptorHandleForHeapStart() }.ptr;
        self.chunks.push(h);
        self.free.extend((1..CHUNK as usize).rev().map(|i| start + i * self.inc));
        Ok(Cpu(start))
    }

    /// Return a descriptor. CPU descriptors are copied (into tables) or read (render targets)
    /// when a command is recorded, so they may be reused at once.
    pub fn free(&mut self, d: Cpu) {
        self.free.push(d.0);
    }
}

/// The CPU-only pools of a device.
pub(crate) struct CpuPools {
    pub rtv: CpuPool,
    pub dsv: CpuPool,
    pub view: CpuPool,
}

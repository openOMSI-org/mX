//! Buffers, textures, views, samplers, bind groups and queue writes.

use std::ops::Range;
use std::sync::Arc;

use mx_types as t;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use crate::conv::{self, SHADER_READ};
use crate::device::{Recording, Shared};
use crate::heap::Cpu;
use crate::track::{self, Track};
use crate::{Device, Error, Queue, Result, hr_err};

// ---------------------------------------------------------------------------------------------
// Raw creation helpers

fn heap_props(ty: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: ty,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    }
}

pub(crate) fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    }
}

/// A committed buffer.
pub(crate) fn raw_buffer(device: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES, label: &str) -> Result<ID3D12Resource> {
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: a plain creation from filled descriptions.
    unsafe { device.CreateCommittedResource(&heap_props(heap), D3D12_HEAP_FLAG_NONE, &buffer_desc(size.max(4), flags), state, None, &mut r) }.map_err(|e| hr_err(label, e))?;
    let r = r.ok_or_else(|| Error::OutOfMemory(label.into()))?;
    crate::set_name(&r, label);
    Ok(r)
}

/// An upload-heap buffer holding `data`.
pub(crate) fn upload_buffer(device: &ID3D12Device, data: &[u8], label: &str) -> Result<ID3D12Resource> {
    let r = raw_buffer(device, data.len() as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, label)?;
    write_mapped(&r, 0, data)?;
    Ok(r)
}

/// Copy `data` into a CPU-visible buffer at `offset`.
pub(crate) fn write_mapped(r: &ID3D12Resource, offset: usize, data: &[u8]) -> Result<()> {
    let mut p = std::ptr::null_mut();
    let none = D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: an upload buffer at least `offset + data.len()` long, not read by the GPU now.
    unsafe {
        r.Map(0, Some(&none), Some(&mut p)).map_err(|e| hr_err("mapping a buffer", e))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast::<u8>().add(offset), data.len());
        let written = D3D12_RANGE {
            Begin: offset,
            End: offset + data.len(),
        };
        r.Unmap(0, Some(&written));
    }
    Ok(())
}

/// An SRV of `n` `uint`s as a `StructuredBuffer<uint>` (naga's sampler index buffers).
pub(crate) fn structured_u32_srv(n: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: 0,
                NumElements: n,
                StructureByteStride: 4,
                Flags: D3D12_BUFFER_SRV_FLAG_NONE,
            },
        },
    }
}

// ---------------------------------------------------------------------------------------------
// Buffers

pub(crate) struct BufferInner {
    pub raw: ID3D12Resource,
    pub size: u64,
    pub usage: t::BufferUsages,
    pub va: u64,
    pub track: Track,
    /// The CPU address of a mappable buffer, mapped once on first use (`mapped_ptr`).
    mapped: std::sync::OnceLock<usize>,
    shared: Arc<Shared>,
}

#[derive(Clone)]
pub struct Buffer(pub(crate) Arc<BufferInner>);

pub struct BufferInitDescriptor<'a> {
    pub label: Option<&'a str>,
    pub contents: &'a [u8],
    pub usage: t::BufferUsages,
}

impl Buffer {
    pub fn size(&self) -> u64 {
        self.0.size
    }

    pub fn usage(&self) -> t::BufferUsages {
        self.0.usage
    }

    fn map<R>(&self, range: Range<u64>, write: bool, f: impl FnOnce(*mut u8, usize) -> R) -> Result<R> {
        let (flag, what) = if write {
            (t::BufferUsages::MAP_WRITE, "map_write")
        } else {
            (t::BufferUsages::MAP_READ, "map_read")
        };
        assert!(self.0.usage.contains(flag), "{what} on a buffer without {flag:?}");
        assert!(
            range.start <= range.end && range.end <= self.0.size,
            "{what} range {range:?} outside the buffer ({} bytes)",
            self.0.size
        );
        self.0.shared.wait_idle()?;
        let mut p = std::ptr::null_mut();
        let read = if write {
            D3D12_RANGE { Begin: 0, End: 0 }
        } else {
            D3D12_RANGE {
                Begin: range.start as usize,
                End: range.end as usize,
            }
        };
        // SAFETY: a CPU-visible buffer the GPU is done with; the slice is inside it.
        unsafe {
            self.0.raw.Map(0, Some(&read), Some(&mut p)).map_err(|e| hr_err("mapping a buffer", e))?;
            let r = f(p.cast::<u8>().add(range.start as usize), (range.end - range.start) as usize);
            let written = if write {
                D3D12_RANGE {
                    Begin: range.start as usize,
                    End: range.end as usize,
                }
            } else {
                D3D12_RANGE { Begin: 0, End: 0 }
            };
            self.0.raw.Unmap(0, Some(&written));
            Ok(r)
        }
    }

    /// Read the buffer's contents. Waits until the GPU has finished all submitted work, so
    /// the bytes are those the last submission left. Needs `MAP_READ`.
    pub fn map_read<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        // SAFETY: the mapped range, read-only, while the GPU is idle.
        self.map(range, false, |p, n| f(unsafe { std::slice::from_raw_parts(p, n) }))
    }

    /// Read the buffer's contents now, without waiting: the caller knows the submissions
    /// that wrote it are done (`Device::is_done`), as after wgpu's `map_async`. Needs `MAP_READ`.
    pub fn map_read_now<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        // SAFETY: the mapped range, read-only, while the GPU is idle.
        self.map(range, false, |p, n| f(unsafe { std::slice::from_raw_parts(p, n) }))
    }

    /// Write the buffer's contents from the CPU. Waits until the GPU has finished all
    /// submitted work first. Needs `MAP_WRITE`.
    pub fn map_write<R>(&self, range: Range<u64>, f: impl FnOnce(&mut [u8]) -> R) -> Result<R> {
        // SAFETY: the mapped range, which nothing else uses while the GPU is idle.
        self.map(range, true, |p, n| f(unsafe { std::slice::from_raw_parts_mut(p, n) }))
    }

    /// The CPU address of a `MAP_READ` / `MAP_WRITE` buffer's bytes, for mapping without
    /// waiting for the whole GPU (as wgpu's `map_async` does). The buffer stays mapped (D3D12
    /// allows it for upload and readback heaps).
    ///
    /// # Safety
    /// The caller reads or writes through it only while no submitted GPU work uses the
    /// buffer (see `Device::is_done`).
    pub unsafe fn mapped_ptr(&self) -> std::ptr::NonNull<u8> {
        assert!(
            self.0.usage.intersects(t::BufferUsages::MAP_READ | t::BufferUsages::MAP_WRITE),
            "mapped_ptr on a buffer without MAP_READ or MAP_WRITE"
        );
        let p = *self.0.mapped.get_or_init(|| {
            let mut p = std::ptr::null_mut();
            // SAFETY: a CPU-visible buffer; a null range: the whole buffer may be read.
            unsafe { self.0.raw.Map(0, None, Some(&mut p)) }.expect("mapping a buffer");
            p as usize
        });
        std::ptr::NonNull::new(p as *mut u8).expect("Map returned null")
    }

    /// The `ID3D12Resource`, for interop. Between submissions it is in `COMMON` (upload
    /// buffers `GENERIC_READ`, readback buffers `COPY_DEST`).
    pub fn raw(&self) -> &ID3D12Resource {
        &self.0.raw
    }

    /// The identity the heap and the tracker know this buffer by.
    pub(crate) fn key(&self) -> usize {
        windows::core::Interface::as_raw(&self.0.raw) as usize
    }

    /// A range of the buffer in the bindless heap ([`BufferView::index`]): a raw SRV, and a raw
    /// UAV when `writable`. Needs `STORAGE` (and no `MAP_*`: mappable buffers live in heaps
    /// shaders cannot write).
    pub fn create_view(&self, desc: &t::BufferViewDescriptor<'_>) -> Result<BufferView> {
        let shared = &self.0.shared;
        let size = desc.resolve(self.size(), self.usage(), &shared.limits);
        assert!(!(desc.writable && self.0.track.fixed), "a writable buffer view needs a buffer without MAP_READ / MAP_WRITE");
        let slot = shared.heap.lock().alloc_buffer(self, desc.writable)?;
        let (first, n) = (desc.offset / 4, (size / 4) as u32);
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: first,
                    NumElements: n,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: first,
                    NumElements: n,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        let v = &shared.views;
        // SAFETY: descriptors of a live buffer into the slot just allocated, which no
        // submission reads (fresh, or past its deferred release). A read-only view's UAV is a
        // null descriptor, so a stray write through `mx_rw_buffers` goes nowhere.
        unsafe {
            shared.device.CreateShaderResourceView(&self.0.raw, Some(&srv), v.view_cpu(v.buffer_srv(slot)));
            let target = if desc.writable { Some(&self.0.raw) } else { None };
            shared.device.CreateUnorderedAccessView(target, None, Some(&uav), v.view_cpu(v.buffer_uav(slot)));
        }
        Ok(BufferView(Arc::new(BufferViewInner {
            buffer: self.clone(),
            offset: desc.offset,
            size,
            writable: desc.writable,
            slot,
        })))
    }
}

pub(crate) struct BufferViewInner {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: u64,
    pub writable: bool,
    slot: u32,
}

impl Drop for BufferViewInner {
    fn drop(&mut self) {
        let shared = &self.buffer.0.shared;
        shared.heap.lock().release_buffer(self.slot, shared.release_after(), self.buffer.clone(), self.writable);
    }
}

/// A range of a buffer in the bindless heap.
#[derive(Clone)]
pub struct BufferView(pub(crate) Arc<BufferViewInner>);

impl BufferView {
    /// The view's index in the heap: `mx_buffers[i]`, `mx_rw_buffers[i]` or a shader's own
    /// buffer view in WGSL.
    pub fn index(&self) -> u32 {
        self.0.slot
    }

    pub fn buffer(&self) -> &Buffer {
        &self.0.buffer
    }

    pub fn offset(&self) -> u64 {
        self.0.offset
    }

    pub fn size(&self) -> u64 {
        self.0.size
    }

    pub fn is_writable(&self) -> bool {
        self.0.writable
    }
}

/// A buffer as `create_buffer` makes it: upload heap for `MAP_WRITE`, readback heap for
/// `MAP_READ`, default heap otherwise; uniform buffers rounded up to 256 bytes (CBVs cover
/// whole 256-byte blocks, as wgpu-hal).
pub(crate) fn create_buffer(shared: &Arc<Shared>, desc: &t::BufferDescriptor<'_>) -> Result<Buffer> {
    let u = desc.usage;
    let mut alloc = desc.size.max(4).next_multiple_of(4);
    if u.contains(t::BufferUsages::UNIFORM) {
        alloc = alloc.next_multiple_of(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT as u64);
    }
    let (heap, state, fixed) = if u.contains(t::BufferUsages::MAP_READ) {
        (D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, true)
    } else if u.contains(t::BufferUsages::MAP_WRITE) {
        (D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, true)
    } else {
        (D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, false)
    };
    let flags = if u.contains(t::BufferUsages::STORAGE) && !fixed {
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_FLAG_NONE
    };
    let raw = raw_buffer(&shared.device, alloc, heap, flags, state, desc.label.unwrap_or("buffer"))?;
    // SAFETY: a plain getter.
    let va = unsafe { raw.GetGPUVirtualAddress() };
    Ok(Buffer(Arc::new(BufferInner {
        track: Track {
            raw: raw.clone(),
            home: state,
            fixed,
            mips: 1,
            layers: 1,
            planes: 1,
            is_texture: false,
        },
        raw,
        size: desc.size,
        usage: u,
        va,
        mapped: std::sync::OnceLock::new(),
        shared: shared.clone(),
    })))
}

// ---------------------------------------------------------------------------------------------
// Textures

pub(crate) struct TextureInner {
    pub raw: ID3D12Resource,
    pub size: t::Extent3d,
    pub format: t::TextureFormat,
    pub dimension: t::TextureDimension,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub usage: t::TextureUsages,
    pub track: Track,
    pub shared: Arc<Shared>,
}

#[derive(Clone)]
pub struct Texture(pub(crate) Arc<TextureInner>);

/// A texture's resource in its home. With enhanced barriers it is created with a layout
/// (`CreateCommittedResource3`) rather than a legacy state: the tracker records only enhanced
/// barriers then, and the debug layer refuses those on a texture resting in a legacy state
/// other than `COMMON` (id 1350, #3).
fn create_texture_resource(shared: &Shared, d: &D3D12_RESOURCE_DESC, home: D3D12_RESOURCE_STATES, raw: &mut Option<ID3D12Resource>) -> windows::core::Result<()> {
    let heap = heap_props(D3D12_HEAP_TYPE_DEFAULT);
    if shared.enhanced_barriers {
        let device: ID3D12Device10 = shared.device.cast()?;
        let d1 = D3D12_RESOURCE_DESC1 {
            Dimension: d.Dimension,
            Alignment: d.Alignment,
            Width: d.Width,
            Height: d.Height,
            DepthOrArraySize: d.DepthOrArraySize,
            MipLevels: d.MipLevels,
            Format: d.Format,
            SampleDesc: d.SampleDesc,
            Layout: d.Layout,
            Flags: d.Flags,
            SamplerFeedbackMipRegion: D3D12_MIP_REGION::default(),
        };
        let layout = crate::track::initial_layout(home);
        // SAFETY: a plain creation from filled descriptions; no protected session, no castable formats.
        return unsafe { device.CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &d1, layout, None, None::<&ID3D12ProtectedResourceSession>, None, raw) };
    }
    // SAFETY: a plain creation from filled descriptions.
    unsafe { shared.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, d, home, None, raw) }
}

/// The state a texture rests in: readable through the heap if it may be in it.
pub(crate) fn home_state(usage: t::TextureUsages) -> D3D12_RESOURCE_STATES {
    if usage.contains(t::TextureUsages::TEXTURE_BINDING) {
        SHADER_READ
    } else {
        D3D12_RESOURCE_STATE_COMMON
    }
}

impl Texture {
    pub(crate) fn from_raw_parts(raw: ID3D12Resource, desc: &t::TextureDescriptor<'_>, home: D3D12_RESOURCE_STATES, shared: Arc<Shared>) -> Texture {
        let layers = if desc.dimension == t::TextureDimension::D3 { 1 } else { desc.size.depth_or_array_layers };
        Texture(Arc::new(TextureInner {
            track: Track {
                raw: raw.clone(),
                home,
                fixed: false,
                mips: desc.mip_level_count,
                layers,
                planes: if desc.format.has_stencil() { 2 } else { 1 },
                is_texture: true,
            },
            raw,
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            shared,
        }))
    }

    pub fn size(&self) -> t::Extent3d {
        self.0.size
    }
    pub fn width(&self) -> u32 {
        self.0.size.width
    }
    pub fn height(&self) -> u32 {
        self.0.size.height
    }
    pub fn format(&self) -> t::TextureFormat {
        self.0.format
    }
    pub fn dimension(&self) -> t::TextureDimension {
        self.0.dimension
    }
    pub fn mip_level_count(&self) -> u32 {
        self.0.mip_level_count
    }
    pub fn sample_count(&self) -> u32 {
        self.0.sample_count
    }
    pub fn usage(&self) -> t::TextureUsages {
        self.0.usage
    }

    /// The state the texture is in whenever no command list of mX's runs (see the crate docs):
    /// work of others on mX's queue finds it there and leaves it there.
    pub fn rest_state(&self) -> D3D12_RESOURCE_STATES {
        self.0.track.home
    }

    /// Array layers (1 for a 3D texture).
    pub(crate) fn layers(&self) -> u32 {
        self.0.track.layers
    }

    /// The `ID3D12Resource`, for interop. Between submissions it is in
    /// `PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE` if it has `TEXTURE_BINDING` usage,
    /// else in `COMMON`.
    pub fn raw(&self) -> &ID3D12Resource {
        &self.0.raw
    }

    /// A view. With `TEXTURE_BINDING` usage it gets a bindless index ([`TextureView::index`]).
    pub fn create_view(&self, desc: &t::TextureViewDescriptor<'_>) -> Result<TextureView> {
        let tex = &self.0;
        let shared = &tex.shared;
        let format = desc.format.unwrap_or(tex.format);
        let layers = self.layers();
        let dimension = desc.dimension.unwrap_or_else(|| conv::default_view_dimension(tex.dimension, layers));
        let mip_count = desc.mip_level_count.unwrap_or(tex.mip_level_count - desc.base_mip_level);
        let layer_count = desc.array_layer_count.unwrap_or(layers - desc.base_array_layer);
        let v = ViewParams {
            dimension,
            multisampled: tex.sample_count > 1,
            base_mip: desc.base_mip_level,
            mips: mip_count,
            base_layer: desc.base_array_layer,
            layers: layer_count,
            plane: conv::plane(desc.aspect),
        };
        let device = &shared.device;
        let srv_format = conv::srv_format(format, desc.aspect);
        let mut view = ViewInner {
            texture: self.clone(),
            format,
            dimension,
            aspect: desc.aspect,
            base_mip_level: desc.base_mip_level,
            mip_level_count: mip_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: layer_count,
            slot: None,
            srv: None,
            uav: None,
            rtv: None,
            dsv: None,
        };
        let mut cpu = shared.cpu.lock();
        if tex.usage.contains(t::TextureUsages::TEXTURE_BINDING)
            && let Some(f) = srv_format
        {
            let d = v.srv(f);
            let slot = shared.heap.lock().alloc_texture()?;
            let staging = cpu.view.alloc(device)?;
            // SAFETY: descriptors of a live texture written into a fresh heap slot and a CPU
            // descriptor nobody else holds.
            unsafe {
                device.CreateShaderResourceView(&tex.raw, Some(&d), shared.views.view_cpu(slot));
                device.CreateShaderResourceView(&tex.raw, Some(&d), staging.handle());
            }
            view.slot = Some(slot);
            view.srv = Some(staging);
        }
        if tex.usage.contains(t::TextureUsages::STORAGE_BINDING)
            && let Some(f) = srv_format
        {
            let d = v.uav(f);
            let c = cpu.view.alloc(device)?;
            // SAFETY: as above.
            unsafe { device.CreateUnorderedAccessView(&tex.raw, None, Some(&d), c.handle()) };
            view.uav = Some(c);
        }
        if tex.usage.contains(t::TextureUsages::RENDER_ATTACHMENT) {
            if format.has_depth() {
                let c = cpu.dsv.alloc(device)?;
                // SAFETY: as above.
                unsafe { device.CreateDepthStencilView(&tex.raw, Some(&v.dsv(conv::format(format))), c.handle()) };
                view.dsv = Some(c);
            } else {
                let c = cpu.rtv.alloc(device)?;
                // SAFETY: as above.
                unsafe { device.CreateRenderTargetView(&tex.raw, Some(&v.rtv(conv::format(format))), c.handle()) };
                view.rtv = Some(c);
            }
        }
        drop(cpu);
        Ok(TextureView(Arc::new(view)))
    }
}

/// The subresources and shape of a view.
struct ViewParams {
    dimension: t::TextureViewDimension,
    multisampled: bool,
    base_mip: u32,
    mips: u32,
    base_layer: u32,
    layers: u32,
    plane: u32,
}

impl ViewParams {
    /// wgpu-hal's SRV description.
    fn srv(&self, format: DXGI_FORMAT) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
        use t::TextureViewDimension as D;
        let mut d = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_SRV_DIMENSION_UNKNOWN,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: Default::default(),
        };
        match self.dimension {
            D::D1 => {
                d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                d.Anonymous.Texture1D = D3D12_TEX1D_SRV {
                    MostDetailedMip: self.base_mip,
                    MipLevels: self.mips,
                    ResourceMinLODClamp: 0.0,
                };
            }
            D::D2 if self.multisampled && self.base_layer == 0 => {
                d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
                d.Anonymous.Texture2DMS = D3D12_TEX2DMS_SRV { UnusedField_NothingToDefine: 0 };
            }
            D::D2 if self.base_layer == 0 => {
                d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                d.Anonymous.Texture2D = D3D12_TEX2D_SRV {
                    MostDetailedMip: self.base_mip,
                    MipLevels: self.mips,
                    PlaneSlice: self.plane,
                    ResourceMinLODClamp: 0.0,
                };
            }
            D::D2 | D::D2Array if self.multisampled => {
                d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                d.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV {
                    FirstArraySlice: self.base_layer,
                    ArraySize: self.layers,
                };
            }
            D::D2 | D::D2Array => {
                d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: self.base_mip,
                    MipLevels: self.mips,
                    FirstArraySlice: self.base_layer,
                    ArraySize: self.layers,
                    PlaneSlice: self.plane,
                    ResourceMinLODClamp: 0.0,
                };
            }
            D::D3 => {
                d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                d.Anonymous.Texture3D = D3D12_TEX3D_SRV {
                    MostDetailedMip: self.base_mip,
                    MipLevels: self.mips,
                    ResourceMinLODClamp: 0.0,
                };
            }
            D::Cube if self.base_layer == 0 => {
                d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                d.Anonymous.TextureCube = D3D12_TEXCUBE_SRV {
                    MostDetailedMip: self.base_mip,
                    MipLevels: self.mips,
                    ResourceMinLODClamp: 0.0,
                };
            }
            D::Cube | D::CubeArray => {
                d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
                d.Anonymous.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV {
                    MostDetailedMip: self.base_mip,
                    MipLevels: self.mips,
                    First2DArrayFace: self.base_layer,
                    NumCubes: self.layers / 6,
                    ResourceMinLODClamp: 0.0,
                };
            }
        }
        d
    }

    fn uav(&self, format: DXGI_FORMAT) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
        use t::TextureViewDimension as D;
        let mut d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_UNKNOWN,
            Anonymous: Default::default(),
        };
        match self.dimension {
            D::D1 => {
                d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                d.Anonymous.Texture1D = D3D12_TEX1D_UAV { MipSlice: self.base_mip };
            }
            D::D2 if self.base_layer == 0 => {
                d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                d.Anonymous.Texture2D = D3D12_TEX2D_UAV {
                    MipSlice: self.base_mip,
                    PlaneSlice: self.plane,
                };
            }
            D::D2 | D::D2Array | D::Cube | D::CubeArray => {
                d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                    MipSlice: self.base_mip,
                    FirstArraySlice: self.base_layer,
                    ArraySize: self.layers,
                    PlaneSlice: self.plane,
                };
            }
            D::D3 => {
                d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                d.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                    MipSlice: self.base_mip,
                    FirstWSlice: 0,
                    WSize: u32::MAX,
                };
            }
        }
        d
    }

    fn rtv(&self, format: DXGI_FORMAT) -> D3D12_RENDER_TARGET_VIEW_DESC {
        use t::TextureViewDimension as D;
        let mut d = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_UNKNOWN,
            Anonymous: Default::default(),
        };
        match self.dimension {
            D::D1 => {
                d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1D;
                d.Anonymous.Texture1D = D3D12_TEX1D_RTV { MipSlice: self.base_mip };
            }
            D::D2 if self.multisampled && self.base_layer == 0 => {
                d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
                d.Anonymous.Texture2DMS = D3D12_TEX2DMS_RTV { UnusedField_NothingToDefine: 0 };
            }
            D::D2 if self.base_layer == 0 => {
                d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                d.Anonymous.Texture2D = D3D12_TEX2D_RTV {
                    MipSlice: self.base_mip,
                    PlaneSlice: 0,
                };
            }
            D::D2 | D::D2Array | D::Cube | D::CubeArray if self.multisampled => {
                d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
                d.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_RTV {
                    FirstArraySlice: self.base_layer,
                    ArraySize: self.layers,
                };
            }
            D::D2 | D::D2Array | D::Cube | D::CubeArray => {
                d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
                d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_RTV {
                    MipSlice: self.base_mip,
                    FirstArraySlice: self.base_layer,
                    ArraySize: self.layers,
                    PlaneSlice: 0,
                };
            }
            D::D3 => {
                d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
                d.Anonymous.Texture3D = D3D12_TEX3D_RTV {
                    MipSlice: self.base_mip,
                    FirstWSlice: self.base_layer,
                    WSize: self.layers,
                };
            }
        }
        d
    }

    fn dsv(&self, format: DXGI_FORMAT) -> D3D12_DEPTH_STENCIL_VIEW_DESC {
        let mut d = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_DSV_DIMENSION_UNKNOWN,
            Flags: D3D12_DSV_FLAG_NONE,
            Anonymous: Default::default(),
        };
        let array = self.base_layer != 0 || self.layers > 1 || matches!(self.dimension, t::TextureViewDimension::D2Array);
        match (self.multisampled, array) {
            (true, false) => {
                d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
                d.Anonymous.Texture2DMS = D3D12_TEX2DMS_DSV { UnusedField_NothingToDefine: 0 };
            }
            (true, true) => {
                d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
                d.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_DSV {
                    FirstArraySlice: self.base_layer,
                    ArraySize: self.layers,
                };
            }
            (false, false) => {
                d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                d.Anonymous.Texture2D = D3D12_TEX2D_DSV { MipSlice: self.base_mip };
            }
            (false, true) => {
                d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_DSV {
                    MipSlice: self.base_mip,
                    FirstArraySlice: self.base_layer,
                    ArraySize: self.layers,
                };
            }
        }
        d
    }
}

pub(crate) struct ViewInner {
    pub texture: Texture,
    pub format: t::TextureFormat,
    pub dimension: t::TextureViewDimension,
    pub aspect: t::TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
    slot: Option<u32>,
    pub srv: Option<Cpu>,
    pub uav: Option<Cpu>,
    pub rtv: Option<Cpu>,
    pub dsv: Option<Cpu>,
}

impl ViewInner {
    /// The view's subresources: its mips and layers, the planes of its aspect.
    pub fn range(&self) -> track::Range {
        let planes = match (self.texture.0.track.planes, self.aspect) {
            (1, _) => 0..1,
            (_, t::TextureAspect::DepthOnly) => 0..1,
            (_, t::TextureAspect::StencilOnly) => 1..2,
            (n, t::TextureAspect::All) => 0..n,
        };
        let layers = if self.texture.0.dimension == t::TextureDimension::D3 {
            0..1
        } else {
            self.base_array_layer..self.base_array_layer + self.array_layer_count
        };
        track::Range {
            mips: self.base_mip_level..self.base_mip_level + self.mip_level_count,
            layers,
            planes,
        }
    }
}

impl Drop for ViewInner {
    fn drop(&mut self) {
        let shared = &self.texture.0.shared;
        if let Some(slot) = self.slot {
            shared.heap.lock().release_texture(slot, shared.release_after(), self.texture.clone());
        }
        let mut cpu = shared.cpu.lock();
        for d in [self.srv, self.uav].into_iter().flatten() {
            cpu.view.free(d);
        }
        if let Some(d) = self.rtv {
            cpu.rtv.free(d);
        }
        if let Some(d) = self.dsv {
            cpu.dsv.free(d);
        }
    }
}

#[derive(Clone)]
pub struct TextureView(pub(crate) Arc<ViewInner>);

impl TextureView {
    /// The view's index in the bindless heap: `mx_tex2d[i]` (or the array of its dimension)
    /// in WGSL. Panics for views without one (no `TEXTURE_BINDING`, or a depth-stencil view
    /// of both aspects); see [`TextureView::try_index`].
    pub fn index(&self) -> u32 {
        self.try_index()
            .expect("this texture view is not in the bindless heap (its texture lacks TEXTURE_BINDING usage, or it shows both depth and stencil)")
    }

    pub fn try_index(&self) -> Option<u32> {
        self.0.slot
    }

    pub fn texture(&self) -> &Texture {
        &self.0.texture
    }

    pub fn format(&self) -> t::TextureFormat {
        self.0.format
    }

    pub fn dimension(&self) -> t::TextureViewDimension {
        self.0.dimension
    }

    pub fn mip_levels(&self) -> Range<u32> {
        self.0.base_mip_level..self.0.base_mip_level + self.0.mip_level_count
    }

    pub fn array_layers(&self) -> Range<u32> {
        self.0.base_array_layer..self.0.base_array_layer + self.0.array_layer_count
    }
}

// ---------------------------------------------------------------------------------------------
// Samplers

pub(crate) struct SamplerInner {
    slot: u32,
    compare: bool,
    shared: Arc<Shared>,
}

impl Drop for SamplerInner {
    fn drop(&mut self) {
        self.shared.heap.lock().release_sampler(self.slot, self.shared.release_after());
    }
}

#[derive(Clone)]
pub struct Sampler(pub(crate) Arc<SamplerInner>);

impl Sampler {
    /// The sampler's index in the heap: `mx_samplers[i]`, or `mx_samplers_cmp[i]` for a
    /// comparison sampler.
    pub fn index(&self) -> u32 {
        self.0.slot
    }

    pub fn is_comparison(&self) -> bool {
        self.0.compare
    }
}

/// wgpu-hal's sampler description.
pub(crate) fn sampler_desc(desc: &t::SamplerDescriptor<'_>) -> D3D12_SAMPLER_DESC {
    D3D12_SAMPLER_DESC {
        Filter: conv::filter(desc),
        AddressU: conv::address(desc.address_mode_u),
        AddressV: conv::address(desc.address_mode_v),
        AddressW: conv::address(desc.address_mode_w),
        MipLODBias: 0.0,
        MaxAnisotropy: desc.anisotropy_clamp as u32,
        ComparisonFunc: conv::compare(desc.compare.unwrap_or(t::CompareFunction::Never)),
        BorderColor: conv::border(desc.border_color),
        MinLOD: desc.lod_min_clamp,
        MaxLOD: desc.lod_max_clamp,
    }
}

// ---------------------------------------------------------------------------------------------
// Bind groups (root bindings)

#[derive(Clone, Copy)]
pub struct BufferBinding<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    /// `None`: to the end of the buffer.
    pub size: Option<u64>,
}

#[derive(Clone, Copy)]
pub enum BindingResource<'a> {
    Buffer(BufferBinding<'a>),
    TextureView(&'a TextureView),
    Sampler(&'a Sampler),
    AccelerationStructure(&'a crate::Tlas),
}

#[derive(Clone, Copy)]
pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

/// Resources for one group of root bindings. There is no layout: the pipeline in use says
/// which bindings it reads, and drawing with one of them missing is an error.
pub struct BindGroupDescriptor<'a> {
    pub label: Option<&'a str>,
    pub entries: &'a [BindGroupEntry<'a>],
}

pub(crate) enum Bound {
    Buffer {
        buffer: Buffer,
        /// The entry's offset and size (a uniform buffer's view is made anew at another
        /// offset, see `RenderPass::set_buffer_offset`).
        offset: u64,
        size: u64,
        cbv: Option<Cpu>,
        srv: Option<Cpu>,
        uav: Option<Cpu>,
    },
    Texture(TextureView),
    /// (kept so its heap slot stays this sampler's while the group's index buffer names it)
    Sampler {
        _keep: Sampler,
    },
    Tlas(Arc<crate::rt::TlasShared>, Cpu),
}

pub(crate) struct BindGroupInner {
    pub entries: Vec<(u32, Bound)>,
    /// The `StructuredBuffer<uint>` of the group's samplers' heap indices, by binding.
    pub sampler_indices: Option<(ID3D12Resource, Cpu)>,
    shared: Arc<Shared>,
}

impl Drop for BindGroupInner {
    fn drop(&mut self) {
        let mut cpu = self.shared.cpu.lock();
        for (_, b) in &self.entries {
            match b {
                Bound::Buffer { cbv, srv, uav, .. } => {
                    for d in [cbv, srv, uav].into_iter().flatten() {
                        cpu.view.free(*d);
                    }
                }
                Bound::Tlas(_, d) => cpu.view.free(*d),
                _ => {}
            }
        }
        if let Some((_, d)) = &self.sampler_indices {
            cpu.view.free(*d);
        }
    }
}

#[derive(Clone)]
pub struct BindGroup(pub(crate) Arc<BindGroupInner>);

impl BindGroup {
    pub(crate) fn get(&self, binding: u32) -> Option<&Bound> {
        self.0.entries.iter().find(|(b, _)| *b == binding).map(|(_, r)| r)
    }
}

// ---------------------------------------------------------------------------------------------
// Creation

impl Device {
    pub fn create_buffer(&self, desc: &t::BufferDescriptor<'_>) -> Result<Buffer> {
        create_buffer(&self.shared, desc)
    }

    /// A buffer filled with `contents` (the fill runs before the next submission).
    pub fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Result<Buffer> {
        let buffer = self.create_buffer(&t::BufferDescriptor {
            label: desc.label,
            size: desc.contents.len() as u64,
            usage: desc.usage
                | if desc.usage.contains(t::BufferUsages::MAP_WRITE) {
                    t::BufferUsages::empty()
                } else {
                    t::BufferUsages::COPY_DST
                },
        })?;
        if desc.usage.contains(t::BufferUsages::MAP_WRITE) {
            buffer.map_write(0..buffer.size(), |b| b.copy_from_slice(desc.contents))?;
        } else {
            Queue { shared: self.shared.clone() }.write_buffer(&buffer, 0, desc.contents)?;
        }
        Ok(buffer)
    }

    pub fn create_texture(&self, desc: &t::TextureDescriptor<'_>) -> Result<Texture> {
        let u = desc.usage;
        let f = desc.format;
        let mut flags = D3D12_RESOURCE_FLAG_NONE;
        if u.contains(t::TextureUsages::RENDER_ATTACHMENT) {
            if f.has_depth() {
                flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                if !u.contains(t::TextureUsages::TEXTURE_BINDING) {
                    flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
                }
            } else {
                flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            }
        }
        if u.contains(t::TextureUsages::STORAGE_BINDING) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }
        let d = D3D12_RESOURCE_DESC {
            Dimension: match desc.dimension {
                t::TextureDimension::D1 => D3D12_RESOURCE_DIMENSION_TEXTURE1D,
                t::TextureDimension::D2 => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                t::TextureDimension::D3 => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            },
            Alignment: 0,
            Width: desc.size.width as u64,
            Height: desc.size.height,
            DepthOrArraySize: desc.size.depth_or_array_layers as u16,
            MipLevels: desc.mip_level_count as u16,
            Format: conv::resource_format(f, u, desc.view_formats.iter().any(|v| *v != f)),
            SampleDesc: DXGI_SAMPLE_DESC { Count: desc.sample_count, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
        };
        let home = home_state(u);
        let label = desc.label.unwrap_or("texture");
        let mut raw: Option<ID3D12Resource> = None;
        create_texture_resource(&self.shared, &d, home, &mut raw).map_err(|e| hr_err(label, e))?;
        let raw = raw.ok_or_else(|| Error::OutOfMemory(label.into()))?;
        crate::set_name(&raw, label);
        Ok(Texture::from_raw_parts(raw, desc, home, self.shared.clone()))
    }

    /// Wrap a texture made outside mX (an XR swapchain image, a DLSS output).
    ///
    /// # Safety
    /// `desc` must describe `raw` (size, format, mips, samples, usage), and between submissions
    /// that use it, `raw` must be in `PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE` if
    /// `desc.usage` has `TEXTURE_BINDING`, else in `COMMON` (mX leaves it there too).
    pub unsafe fn texture_from_raw(&self, raw: ID3D12Resource, desc: &t::TextureDescriptor<'_>) -> Texture {
        Texture::from_raw_parts(raw, desc, home_state(desc.usage), self.shared.clone())
    }

    pub fn create_sampler(&self, desc: &t::SamplerDescriptor<'_>) -> Result<Sampler> {
        let slot = self.shared.heap.lock().alloc_sampler()?;
        // SAFETY: a fresh slot of the sampler heap.
        unsafe { self.shared.device.CreateSampler(&sampler_desc(desc), self.shared.views.sampler_cpu(slot)) };
        Ok(Sampler(Arc::new(SamplerInner {
            slot,
            compare: desc.compare.is_some(),
            shared: self.shared.clone(),
        })))
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup {
        let shared = &self.shared;
        let device = &shared.device;
        let mut cpu = shared.cpu.lock();
        let mut alloc = || cpu.view.alloc(device).expect("out of memory for descriptors");
        let mut samplers: Vec<(u32, u32)> = Vec::new();
        let entries = desc
            .entries
            .iter()
            .map(|e| {
                let r = match e.resource {
                    BindingResource::Buffer(b) => {
                        let buf = &b.buffer.0;
                        let size = b.size.unwrap_or(buf.size - b.offset);
                        assert!(b.offset + size <= buf.size, "bind group entry {}: range past the end of the buffer", e.binding);
                        let cbv = buf.usage.contains(t::BufferUsages::UNIFORM).then(|| {
                            assert!(
                                b.offset.is_multiple_of(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT as u64),
                                "bind group entry {}: uniform buffer offsets must be multiples of 256",
                                e.binding
                            );
                            let d = alloc();
                            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                                BufferLocation: buf.va + b.offset,
                                SizeInBytes: size.max(1).next_multiple_of(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT as u64) as u32,
                            };
                            // SAFETY: a view of a live buffer into a CPU descriptor of ours.
                            unsafe { device.CreateConstantBufferView(Some(&desc), d.handle()) };
                            d
                        });
                        let (srv, uav) = if buf.usage.contains(t::BufferUsages::STORAGE) {
                            assert!(b.offset.is_multiple_of(4), "bind group entry {}: storage buffer offsets must be multiples of 4", e.binding);
                            let (first, n) = (b.offset / 4, size.div_ceil(4) as u32);
                            let s = alloc();
                            let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                                Format: DXGI_FORMAT_R32_TYPELESS,
                                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                    Buffer: D3D12_BUFFER_SRV {
                                        FirstElement: first,
                                        NumElements: n,
                                        StructureByteStride: 0,
                                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                                    },
                                },
                            };
                            let u = alloc();
                            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                                Format: DXGI_FORMAT_R32_TYPELESS,
                                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                    Buffer: D3D12_BUFFER_UAV {
                                        FirstElement: first,
                                        NumElements: n,
                                        StructureByteStride: 0,
                                        CounterOffsetInBytes: 0,
                                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                                    },
                                },
                            };
                            // SAFETY: as above (the buffer allows UAVs: it has STORAGE usage).
                            unsafe {
                                device.CreateShaderResourceView(&buf.raw, Some(&srv), s.handle());
                                device.CreateUnorderedAccessView(&buf.raw, None, Some(&uav), u.handle());
                            }
                            (Some(s), Some(u))
                        } else {
                            (None, None)
                        };
                        Bound::Buffer {
                            buffer: b.buffer.clone(),
                            offset: b.offset,
                            size,
                            cbv,
                            srv,
                            uav,
                        }
                    }
                    BindingResource::TextureView(v) => Bound::Texture(v.clone()),
                    BindingResource::Sampler(s) => {
                        samplers.push((e.binding, s.index()));
                        Bound::Sampler { _keep: s.clone() }
                    }
                    BindingResource::AccelerationStructure(a) => {
                        let d = alloc();
                        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: DXGI_FORMAT_UNKNOWN,
                            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: a.inner.va },
                            },
                        };
                        // SAFETY: an acceleration structure SRV takes no resource, only its address.
                        unsafe { device.CreateShaderResourceView(None, Some(&srv), d.handle()) };
                        Bound::Tlas(a.inner.clone(), d)
                    }
                };
                (e.binding, r)
            })
            .collect();
        let sampler_indices = if samplers.is_empty() {
            None
        } else {
            let n = samplers.iter().map(|s| s.0).max().unwrap() + 1;
            let mut v = vec![0u32; n as usize];
            for (b, s) in samplers {
                v[b as usize] = s;
            }
            let buf = upload_buffer(device, crate::device::bytes_of_u32(&v), desc.label.unwrap_or("mX sampler indices")).expect("out of memory for a sampler index buffer");
            let d = cpu.view.alloc(device).expect("out of memory for descriptors");
            // SAFETY: as above.
            unsafe { device.CreateShaderResourceView(&buf, Some(&structured_u32_srv(n)), d.handle()) };
            Some((buf, d))
        };
        drop(cpu);
        BindGroup(Arc::new(BindGroupInner {
            entries,
            sampler_indices,
            shared: shared.clone(),
        }))
    }
}

// ---------------------------------------------------------------------------------------------
// Copies and queue writes

/// A buffer footprint for a texture copy.
pub(crate) struct Footprint {
    pub offset: u64,
    pub bytes_per_row: u32,
    /// Rows of blocks (texel rows for uncompressed formats) per layer.
    pub rows_per_image: u32,
}

/// A copy location in a texture.
pub(crate) fn texture_location(tex: &Texture, sub: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: the texture outlives the copy call (the caller holds it).
        pResource: unsafe { crate::borrow(&tex.0.raw) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
    }
}

/// The copy extent in whole blocks (wgpu's physical size).
pub(crate) fn physical(format: t::TextureFormat, size: t::Extent3d) -> t::Extent3d {
    let (bw, bh) = format.block_dimensions();
    t::Extent3d {
        width: size.width.next_multiple_of(bw),
        height: size.height.next_multiple_of(bh),
        depth_or_array_layers: size.depth_or_array_layers,
    }
}

/// One buffer ↔ texture copy, a layer at a time for array textures. The texels of layer `i`
/// start at `fp.offset + i × bytes_per_row × rows_per_image` in `raw`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_buffer_texture(rec: &mut Recording, raw: &ID3D12Resource, fp: Footprint, tex: &Texture, at: &t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d, to_texture: bool) {
    let format = tex.format();
    let size = physical(format, size);
    let is_3d = tex.dimension() == t::TextureDimension::D3;
    let (layers, depth) = if is_3d { (1, size.depth_or_array_layers) } else { (size.depth_or_array_layers, 1) };
    let plane = conv::plane(at.aspect);
    assert!(
        fp.bytes_per_row.is_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT),
        "texture copies need bytes_per_row to be a multiple of {}",
        D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
    );
    let range = track::Range {
        mips: at.mip_level..at.mip_level + 1,
        layers: if is_3d { 0..1 } else { at.origin.z..at.origin.z + layers },
        planes: plane..plane + 1,
    };
    rec.track
        .need(&tex.0.track, Some(range), if to_texture { D3D12_RESOURCE_STATE_COPY_DEST } else { D3D12_RESOURCE_STATE_COPY_SOURCE });
    rec.barriers();
    let bytes_per_image = fp.bytes_per_row as u64 * fp.rows_per_image as u64;
    for l in 0..layers {
        let layer = if is_3d { 0 } else { at.origin.z + l };
        let sub = tex.0.track.subresource(at.mip_level, layer, plane);
        let buf = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: the resource outlives the copy call.
            pResource: unsafe { crate::borrow(raw) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: fp.offset + l as u64 * bytes_per_image,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: conv::copy_format(format, at.aspect),
                        Width: size.width,
                        Height: size.height,
                        Depth: depth,
                        RowPitch: fp.bytes_per_row,
                    },
                },
            },
        };
        let texl = texture_location(tex, sub);
        let z = if is_3d { at.origin.z } else { 0 };
        // SAFETY: regions inside the buffer and the texture (the caller's contract; the debug
        // layer checks).
        unsafe {
            if to_texture {
                rec.list().CopyTextureRegion(&texl, at.origin.x, at.origin.y, z, &buf, None);
            } else {
                let b = D3D12_BOX {
                    left: at.origin.x,
                    top: at.origin.y,
                    front: z,
                    right: at.origin.x + size.width,
                    bottom: at.origin.y + size.height,
                    back: z + depth,
                };
                rec.list().CopyTextureRegion(&buf, 0, 0, 0, &texl, Some(&b));
            }
        }
    }
    rec.track.restore_textures();
}

impl Queue {
    /// Copy `data` into `buffer` at `offset` before the next submission's commands run.
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        assert!(offset + data.len() as u64 <= buffer.size(), "write_buffer past the end of the buffer");
        let shared = self.shared.clone();
        self.shared.with_pending(|rec| {
            let (staging, at) = shared.stage(rec, data)?;
            rec.track.need(&buffer.0.track, None, D3D12_RESOURCE_STATE_COPY_DEST);
            rec.barriers();
            // SAFETY: both ranges are inside their buffers.
            unsafe { rec.list().CopyBufferRegion(&buffer.0.raw, offset, &staging, at, data.len() as u64) };
            Ok(())
        })
    }

    /// Copy texels from `data` into a texture before the next submission's commands run.
    pub fn write_texture(&self, dst: t::TexelCopyTextureInfo<'_, Texture>, data: &[u8], layout: t::TexelCopyBufferLayout, size: t::Extent3d) -> Result<()> {
        let tex = dst.texture;
        let format = tex.format();
        let phys = physical(format, size);
        let (_, bh) = format.block_dimensions();
        let row_bytes = format.bytes_per_row(phys.width).expect("copies of depth-stencil textures go through one aspect") as usize;
        let src_bpr = layout.bytes_per_row.map_or(row_bytes, |b| b as usize);
        let rows = (phys.height / bh) as usize;
        let src_rows = layout.rows_per_image.map_or(rows, |r| (r / bh.max(1)).max(1) as usize);
        // D3D12 wants rows 256-aligned and layers 512-aligned: repack
        let bpr = row_bytes.next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize);
        let rows_per_image = if bpr.is_multiple_of(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as usize) {
            rows
        } else {
            rows.next_multiple_of(2)
        };
        let images = phys.depth_or_array_layers as usize;
        let mut staging = vec![0u8; (bpr * rows_per_image * images).max(4)];
        for i in 0..images {
            for r in 0..rows {
                let s = layout.offset as usize + (i * src_rows + r) * src_bpr;
                let d = (i * rows_per_image + r) * bpr;
                staging[d..d + row_bytes].copy_from_slice(&data[s..s + row_bytes]);
            }
        }
        let shared = self.shared.clone();
        self.shared.with_pending(|rec| {
            let (raw, at) = shared.stage(rec, &staging)?;
            let fp = Footprint {
                offset: at,
                bytes_per_row: bpr as u32,
                rows_per_image: rows_per_image as u32,
            };
            copy_buffer_texture(rec, &raw, fp, tex, &dst, size, true);
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------------------------
// GPU-driven draws

/// The count and index buffer of the last `encode_indirect_draws` of a batch.
#[derive(Clone)]
pub(crate) struct IndirectSource {
    pub count: Buffer,
    pub count_offset: u64,
    pub index: Option<(Buffer, u64, t::IndexFormat)>,
}

pub(crate) struct IndirectDrawsInner {
    pub max_draws: u32,
    pub indexed: bool,
    /// The arguments laid out for the pipelines' command signatures.
    pub laid_out: Buffer,
    pub source: parking_lot::Mutex<Option<IndirectSource>>,
}

/// A batch of draws whose arguments and count the GPU writes (see `t::IndirectDrawsDescriptor`).
#[derive(Clone)]
pub struct IndirectDraws(pub(crate) Arc<IndirectDrawsInner>);

impl IndirectDraws {
    pub fn max_draws(&self) -> u32 {
        self.0.max_draws
    }
}

//! Buffers, textures, views, samplers and bind groups.

use std::ops::Range;
use std::sync::Arc;

use mx_types as t;
use objc2_foundation::NSRange;
use objc2_metal::*;

use crate::device::Shared;
use crate::{Device, Error, Mt, Po, Queue, Result, conv};

// ---------------------------------------------------------------------------------------------
// Buffers

pub(crate) struct BufferInner {
    pub raw: Mt<Po<dyn MTLBuffer>>,
    pub size: u64,
    pub usage: t::BufferUsages,
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

    /// Read the buffer's contents. Waits until the GPU has finished all submitted work, so
    /// the bytes are those the last submission left. Needs `MAP_READ`.
    pub fn map_read<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        assert!(self.0.usage.contains(t::BufferUsages::MAP_READ), "map_read on a buffer without MAP_READ");
        assert!(
            range.start <= range.end && range.end <= self.0.size,
            "map_read range {range:?} outside the buffer ({} bytes)",
            self.0.size
        );
        self.0.shared.wait_idle()?;
        // SAFETY: shared storage, in range, and the GPU is idle.
        let bytes = unsafe { std::slice::from_raw_parts(self.0.raw.contents().as_ptr().cast::<u8>().add(range.start as usize), (range.end - range.start) as usize) };
        Ok(f(bytes))
    }

    /// Read the buffer's contents now, without waiting: the caller knows the submissions
    /// that wrote it are done (`Device::is_done`), as after wgpu's `map_async`. Needs `MAP_READ`.
    pub fn map_read_now<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        assert!(self.0.usage.contains(t::BufferUsages::MAP_READ), "map_read on a buffer without MAP_READ");
        assert!(
            range.start <= range.end && range.end <= self.0.size,
            "map_read range {range:?} outside the buffer ({} bytes)",
            self.0.size
        );
        // SAFETY: shared storage, in range, and the GPU is idle.
        let bytes = unsafe { std::slice::from_raw_parts(self.0.raw.contents().as_ptr().cast::<u8>().add(range.start as usize), (range.end - range.start) as usize) };
        Ok(f(bytes))
    }

    /// Write the buffer's contents from the CPU. Waits until the GPU has finished all
    /// submitted work first. Needs `MAP_WRITE`.
    pub fn map_write<R>(&self, range: Range<u64>, f: impl FnOnce(&mut [u8]) -> R) -> Result<R> {
        assert!(self.0.usage.contains(t::BufferUsages::MAP_WRITE), "map_write on a buffer without MAP_WRITE");
        assert!(
            range.start <= range.end && range.end <= self.0.size,
            "map_write range {range:?} outside the buffer ({} bytes)",
            self.0.size
        );
        self.0.shared.wait_idle()?;
        // SAFETY: shared storage, in range, the GPU is idle, and `&self` hands out no other slice
        // (a second `map_write` waits here before taking one).
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.0.raw.contents().as_ptr().cast::<u8>().add(range.start as usize), (range.end - range.start) as usize) };
        Ok(f(bytes))
    }

    /// The CPU address of a `MAP_READ` / `MAP_WRITE` buffer's bytes, for mapping without
    /// waiting for the whole GPU (as wgpu's `map_async` does).
    ///
    /// # Safety
    /// The caller reads or writes through it only while no submitted GPU work uses the
    /// buffer (see `Device::is_done`).
    pub unsafe fn mapped_ptr(&self) -> std::ptr::NonNull<u8> {
        assert!(
            self.0.usage.intersects(t::BufferUsages::MAP_READ | t::BufferUsages::MAP_WRITE),
            "mapped_ptr on a buffer without MAP_READ or MAP_WRITE"
        );
        self.0.raw.contents().cast()
    }

    /// The `MTLBuffer`, for interop.
    pub fn raw(&self) -> &objc2::runtime::ProtocolObject<dyn MTLBuffer> {
        &self.0.raw
    }

    /// A range of the buffer in the bindless heap ([`BufferView::index`]). Needs `STORAGE`.
    pub fn create_view(&self, desc: &t::BufferViewDescriptor<'_>) -> Result<BufferView> {
        let size = desc.resolve(self.size(), self.usage(), &self.0.shared.limits);
        let slot = self.0.shared.heap.lock().add_buffer(&self.0.raw, self.owner_id(), desc.offset, desc.writable)?;
        Ok(BufferView(Arc::new(BufferViewInner {
            buffer: self.clone(),
            offset: desc.offset,
            size,
            writable: desc.writable,
            slot,
        })))
    }

    /// The identity the heap knows this buffer by.
    pub(crate) fn owner_id(&self) -> usize {
        Arc::as_ptr(&self.0) as *const u8 as usize
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
        shared.heap.lock().remove_buffer(self.slot, shared.release_after());
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

// ---------------------------------------------------------------------------------------------
// Textures

pub(crate) struct TextureInner {
    pub raw: Mt<Po<dyn MTLTexture>>,
    pub size: t::Extent3d,
    pub format: t::TextureFormat,
    pub dimension: t::TextureDimension,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub usage: t::TextureUsages,
    /// What nothing has written yet of a texture made by mX (zeroed at the next submission).
    pub unwritten: parking_lot::Mutex<Option<t::Unwritten>>,
    /// Made outside mX (a drawable, `texture_from_raw`): written by others at any time.
    external: bool,
    shared: Arc<Shared>,
}

#[derive(Clone)]
pub struct Texture(pub(crate) Arc<TextureInner>);

impl Texture {
    pub(crate) fn from_raw_parts(raw: Mt<Po<dyn MTLTexture>>, desc: &t::TextureDescriptor<'_>, shared: Arc<Shared>) -> Texture {
        Texture(Arc::new(TextureInner {
            raw,
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            unwritten: parking_lot::Mutex::new(None),
            external: true,
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

    /// A view. With `TEXTURE_BINDING` usage, or as a storage texture (`STORAGE_BINDING`, one mip
    /// of a 2D view), it gets a bindless index ([`TextureView::index`]).
    pub fn create_view(&self, desc: &t::TextureViewDescriptor<'_>) -> Result<TextureView> {
        let tex = &self.0;
        let format = desc.format.unwrap_or(tex.format);
        let layers = if tex.dimension == t::TextureDimension::D3 { 1 } else { tex.size.depth_or_array_layers };
        let dimension = desc.dimension.unwrap_or_else(|| conv::default_view_dimension(tex.dimension, layers));
        let mip_count = desc.mip_level_count.unwrap_or(tex.mip_level_count - desc.base_mip_level);
        let layer_count = desc.array_layer_count.unwrap_or(layers - desc.base_array_layer);
        let pixel = conv::aspect_format(format, desc.aspect);
        let ty = conv::texture_type(dimension, tex.sample_count);
        let whole =
            pixel == tex.raw.pixelFormat() && ty == tex.raw.textureType() && desc.base_mip_level == 0 && mip_count == tex.mip_level_count && desc.base_array_layer == 0 && layer_count == layers;
        let raw = if whole {
            tex.raw.clone()
        } else {
            let levels = NSRange::new(desc.base_mip_level as usize, mip_count as usize);
            let slices = NSRange::new(desc.base_array_layer as usize, layer_count as usize);
            // SAFETY: the ranges are inside the texture (checked by Metal's validation layer in
            // debug); the format is one of the texture's view formats.
            let v = unsafe { tex.raw.newTextureViewWithPixelFormat_textureType_levels_slices(pixel, ty, levels, slices) };
            Mt(v.ok_or_else(|| Error::OutOfMemory("texture view".into()))?)
        };
        if let Some(l) = desc.label {
            raw.setLabel(Some(&crate::ns(l)));
        }
        let sampled = tex.usage.contains(t::TextureUsages::TEXTURE_BINDING) && !(format.has_stencil() && desc.aspect == t::TextureAspect::All);
        let storage = t::is_heap_storage_view(tex.usage, format, dimension, mip_count);
        let slot = if sampled || storage {
            let hot = tex.external || tex.usage.intersects(t::TextureUsages::RENDER_ATTACHMENT | t::TextureUsages::STORAGE_BINDING);
            Some(tex.shared.heap.lock().add_texture(&raw, self.owner_id(), hot, storage)?)
        } else {
            None
        };
        Ok(TextureView(Arc::new(ViewInner {
            raw,
            texture: self.clone(),
            format,
            dimension,
            base_mip_level: desc.base_mip_level,
            mip_level_count: mip_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: layer_count,
            slot,
        })))
    }

    /// The `MTLTexture`, for interop.
    pub fn raw(&self) -> &objc2::runtime::ProtocolObject<dyn MTLTexture> {
        &self.0.raw
    }

    /// The identity the heap knows this texture's views by.
    pub(crate) fn owner_id(&self) -> usize {
        Arc::as_ptr(&self.0) as *const u8 as usize
    }
}

pub(crate) struct ViewInner {
    pub raw: Mt<Po<dyn MTLTexture>>,
    pub texture: Texture,
    pub format: t::TextureFormat,
    pub dimension: t::TextureViewDimension,
    pub base_mip_level: u32,
    pub mip_level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
    slot: Option<u32>,
}

impl Drop for ViewInner {
    fn drop(&mut self) {
        if let Some(slot) = self.slot {
            let shared = &self.texture.0.shared;
            shared.heap.lock().remove_texture(slot, shared.release_after());
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
    pub raw: Mt<Po<dyn MTLSamplerState>>,
    slot: u32,
    compare: bool,
    shared: Arc<Shared>,
}

impl Drop for SamplerInner {
    fn drop(&mut self) {
        self.shared.heap.lock().remove_sampler(self.slot, self.shared.release_after());
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
    Buffer { buffer: Buffer, offset: u64, size: u64 },
    Texture(TextureView),
    Sampler(Sampler),
    Tlas(Arc<crate::rt::TlasShared>),
}

#[derive(Clone)]
pub struct BindGroup(pub(crate) Arc<Vec<(u32, Bound)>>);

impl BindGroup {
    pub(crate) fn get(&self, binding: u32) -> Option<&Bound> {
        self.0.iter().find(|(b, _)| *b == binding).map(|(_, r)| r)
    }
}

// ---------------------------------------------------------------------------------------------
// Creation

impl Device {
    pub fn create_buffer(&self, desc: &t::BufferDescriptor<'_>) -> Result<Buffer> {
        let mappable = desc.usage.intersects(t::BufferUsages::MAP_READ | t::BufferUsages::MAP_WRITE);
        let options = if mappable {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModePrivate
        };
        let len = desc.size.max(4) as usize;
        let raw = self
            .shared
            .device
            .newBufferWithLength_options(len, options)
            .ok_or_else(|| Error::OutOfMemory(desc.label.unwrap_or("buffer").into()))?;
        if let Some(l) = desc.label {
            raw.setLabel(Some(&crate::ns(l)));
        }
        Ok(Buffer(Arc::new(BufferInner {
            raw: Mt(raw),
            size: desc.size,
            usage: desc.usage,
            shared: self.shared.clone(),
        })))
    }

    /// A buffer filled with `contents` (the fill runs before the next submission).
    pub fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Result<Buffer> {
        let buffer = self.create_buffer(&t::BufferDescriptor {
            label: desc.label,
            size: desc.contents.len() as u64,
            usage: desc.usage | t::BufferUsages::COPY_DST,
        })?;
        if desc.usage.contains(t::BufferUsages::MAP_WRITE) {
            buffer.map_write(0..buffer.size(), |b| b.copy_from_slice(desc.contents))?;
        } else {
            Queue { shared: self.shared.clone() }.write_buffer(&buffer, 0, desc.contents)?;
        }
        Ok(buffer)
    }

    pub fn create_texture(&self, desc: &t::TextureDescriptor<'_>) -> Result<Texture> {
        if desc.format.is_compressed() && !self.shared.bc {
            return Err(Error::Unsupported(format!("{:?}", desc.format)));
        }
        let d = MTLTextureDescriptor::new();
        let layers = desc.size.depth_or_array_layers;
        let ty = match desc.dimension {
            t::TextureDimension::D1 => MTLTextureType::Type1D,
            t::TextureDimension::D2 if desc.sample_count > 1 => MTLTextureType::Type2DMultisample,
            // (a 6-layer texture may be viewed as a cube: Metal allows that from a 2D array)
            t::TextureDimension::D2 if layers > 1 => MTLTextureType::Type2DArray,
            t::TextureDimension::D2 => MTLTextureType::Type2D,
            t::TextureDimension::D3 => MTLTextureType::Type3D,
        };
        // SAFETY: plain property setters on a fresh descriptor; sizes are checked by Metal.
        unsafe {
            d.setTextureType(ty);
            d.setPixelFormat(conv::pixel_format(desc.format));
            d.setWidth(desc.size.width as usize);
            d.setHeight(desc.size.height as usize);
            if desc.dimension == t::TextureDimension::D3 {
                d.setDepth(layers as usize);
            } else {
                d.setArrayLength(layers as usize);
            }
            d.setMipmapLevelCount(desc.mip_level_count as usize);
            d.setSampleCount(desc.sample_count as usize);
        }
        let mut usage = conv::texture_usage(desc.usage, desc.format, desc.view_formats.iter().any(|f| *f != desc.format));
        if desc.format.has_depth() || desc.format.has_stencil() {
            // (depth and stencil are zeroed by a render pass that clears them)
            usage |= MTLTextureUsage::RenderTarget;
        }
        d.setUsage(usage);
        d.setStorageMode(MTLStorageMode::Private);
        let raw = self
            .shared
            .device
            .newTextureWithDescriptor(&d)
            .ok_or_else(|| Error::OutOfMemory(desc.label.unwrap_or("texture").into()))?;
        if let Some(l) = desc.label {
            raw.setLabel(Some(&crate::ns(l)));
        }
        let inner = Arc::new(TextureInner {
            raw: Mt(raw),
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            unwritten: parking_lot::Mutex::new(Some(t::Unwritten::new(desc))),
            external: false,
            shared: self.shared.clone(),
        });
        self.shared.fresh.lock().push(Arc::downgrade(&inner));
        Ok(Texture(inner))
    }

    /// Wrap a texture made outside mX (MetalFX output, a capture, an XR image).
    ///
    /// # Safety
    /// `desc` must describe `raw` (size, format, mips, samples, usage).
    pub unsafe fn texture_from_raw(&self, raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLTexture>>, desc: &t::TextureDescriptor<'_>) -> Texture {
        Texture::from_raw_parts(Mt(raw), desc, self.shared.clone())
    }

    pub fn create_sampler(&self, desc: &t::SamplerDescriptor<'_>) -> Result<Sampler> {
        let d = MTLSamplerDescriptor::new();
        d.setSAddressMode(conv::address(desc.address_mode_u));
        d.setTAddressMode(conv::address(desc.address_mode_v));
        d.setRAddressMode(conv::address(desc.address_mode_w));
        d.setMagFilter(conv::filter(desc.mag_filter));
        d.setMinFilter(conv::filter(desc.min_filter));
        d.setMipFilter(conv::mip_filter(desc.mipmap_filter));
        d.setLodMinClamp(desc.lod_min_clamp);
        d.setLodMaxClamp(desc.lod_max_clamp);
        d.setMaxAnisotropy(desc.anisotropy_clamp.max(1) as usize);
        if let Some(c) = desc.compare {
            d.setCompareFunction(conv::compare(c));
        }
        if let Some(b) = desc.border_color {
            d.setBorderColor(conv::border(b));
        }
        d.setSupportArgumentBuffers(true);
        if let Some(l) = desc.label {
            d.setLabel(Some(&crate::ns(l)));
        }
        let raw = self.shared.device.newSamplerStateWithDescriptor(&d).ok_or_else(|| Error::OutOfMemory("sampler".into()))?;
        let slot = self.shared.heap.lock().add_sampler(&raw)?;
        Ok(Sampler(Arc::new(SamplerInner {
            raw: Mt(raw),
            slot,
            compare: desc.compare.is_some(),
            shared: self.shared.clone(),
        })))
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup {
        let entries = desc
            .entries
            .iter()
            .map(|e| {
                let r = match e.resource {
                    BindingResource::Buffer(b) => {
                        let size = b.size.unwrap_or(b.buffer.size() - b.offset);
                        assert!(b.offset + size <= b.buffer.size(), "bind group entry {}: range past the end of the buffer", e.binding);
                        Bound::Buffer {
                            buffer: b.buffer.clone(),
                            offset: b.offset,
                            size,
                        }
                    }
                    BindingResource::TextureView(v) => Bound::Texture(v.clone()),
                    BindingResource::Sampler(s) => Bound::Sampler(s.clone()),
                    BindingResource::AccelerationStructure(a) => Bound::Tlas(a.inner.clone()),
                };
                (e.binding, r)
            })
            .collect();
        BindGroup(Arc::new(entries))
    }
}

// ---------------------------------------------------------------------------------------------
// Queue writes

impl Queue {
    /// Copy `data` into `buffer` at `offset` before the next submission's commands run.
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        assert!(offset + data.len() as u64 <= buffer.size(), "write_buffer past the end of the buffer");
        self.shared.heap.lock().written(buffer.owner_id(), self.shared.release_after());
        self.shared.with_pending(|p| {
            let (staging, at) = self.shared.stage(p, data)?;
            // SAFETY: both ranges are in their buffers.
            unsafe {
                p.blit
                    .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staging, at, &buffer.0.raw, offset as usize, data.len())
            };
            Ok(())
        })
    }

    /// Copy texels from `data` into a texture before the next submission's commands run.
    pub fn write_texture(&self, dst: t::TexelCopyTextureInfo<'_, Texture>, data: &[u8], layout: t::TexelCopyBufferLayout, size: t::Extent3d) -> Result<()> {
        let tex = dst.texture;
        if let Some(u) = tex.0.unwritten.lock().as_mut() {
            u.written(tex.format(), tex.dimension(), tex.size(), dst.mip_level, dst.origin, size);
        }
        self.shared.heap.lock().written(tex.owner_id(), self.shared.release_after());
        // (the staging buffer starts at `layout.offset` of `data`)
        self.shared.with_pending(|p| {
            let (staging, at) = self.shared.stage(p, &data[layout.offset as usize..])?;
            copy_buffer_texture(&p.blit, &staging, at as u64, tex, &dst, layout, size, true);
            Ok(())
        })
    }
}

/// `size` at `origin` of mip `level`, cut to the level's own size. A copy of a compressed
/// format covers whole blocks (WebGPU's physical size: 4×4 into a 1×1 mip); Metal takes the
/// level's real size there, as wgpu-hal passes it.
pub(crate) fn clamp_to_mip(tex: &Texture, level: u32, origin: t::Origin3d, size: t::Extent3d) -> t::Extent3d {
    let at = |full: u32| (full >> level).max(1);
    let depth = if tex.dimension() == t::TextureDimension::D3 {
        at(tex.size().depth_or_array_layers).saturating_sub(origin.z).min(size.depth_or_array_layers)
    } else {
        size.depth_or_array_layers
    };
    t::Extent3d {
        width: at(tex.width()).saturating_sub(origin.x).min(size.width),
        height: at(tex.height()).saturating_sub(origin.y).min(size.height),
        depth_or_array_layers: depth,
    }
}

/// One buffer ↔ texture copy, a slice at a time for array textures. The texels start at
/// `buffer_offset` in `buffer` (`layout.offset` is not applied again).
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_buffer_texture(
    blit: &objc2::runtime::ProtocolObject<dyn MTLBlitCommandEncoder>,
    buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buffer_offset: u64,
    tex: &Texture,
    at: &t::TexelCopyTextureInfo<'_, Texture>,
    layout: t::TexelCopyBufferLayout,
    size: t::Extent3d,
    to_texture: bool,
) {
    let format = tex.format();
    let (_, bh) = format.block_dimensions();
    let bytes_per_row = layout
        .bytes_per_row
        .unwrap_or_else(|| format.bytes_per_row(size.width).expect("copies of depth-stencil textures need bytes_per_row"));
    let rows = layout.rows_per_image.unwrap_or(size.height.div_ceil(bh));
    let bytes_per_image = bytes_per_row as usize * rows as usize;
    let is_3d = tex.dimension() == t::TextureDimension::D3;
    let (slices, depth) = if is_3d { (1, size.depth_or_array_layers) } else { (size.depth_or_array_layers, 1) };
    // (the buffer's layout follows `size`; only the region Metal is given is cut to the level)
    let real = clamp_to_mip(tex, at.mip_level, at.origin, size);
    let options = match (format.has_stencil(), at.aspect) {
        (true, t::TextureAspect::DepthOnly) => MTLBlitOption::DepthFromDepthStencil,
        (true, t::TextureAspect::StencilOnly) => MTLBlitOption::StencilFromDepthStencil,
        _ => MTLBlitOption::None,
    };
    for s in 0..slices {
        let offset = buffer_offset as usize + s as usize * bytes_per_image;
        let origin = t::Origin3d {
            z: if is_3d { at.origin.z } else { 0 },
            ..at.origin
        };
        let slice = if is_3d { 0 } else { at.origin.z + s };
        let msize = conv::size(t::Extent3d {
            width: real.width,
            height: real.height,
            depth_or_array_layers: if is_3d { real.depth_or_array_layers } else { depth },
        });
        // SAFETY: the regions are inside the texture and the buffer (the caller's contract,
        // checked by Metal's validation layer in debug).
        unsafe {
            if to_texture {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                    buffer,
                    offset,
                    bytes_per_row as usize,
                    bytes_per_image,
                    msize,
                    &tex.0.raw,
                    slice as usize,
                    at.mip_level as usize,
                    conv::origin(origin),
                    options,
                );
            } else {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                    &tex.0.raw,
                    slice as usize,
                    at.mip_level as usize,
                    conv::origin(origin),
                    msize,
                    buffer,
                    offset,
                    bytes_per_row as usize,
                    bytes_per_image,
                    options,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GPU-driven draws

pub(crate) struct IndirectDrawsInner {
    pub max_draws: u32,
    pub indexed: bool,
    /// Draws encoded by `encode_indirect_draws`, executed by `execute_indirect_draws`; it
    /// inherits the pipeline, the buffers and the other state from the render encoder.
    pub icb: Mt<Po<dyn MTLIndirectCommandBuffer>>,
    /// The argument buffer the encoding kernel reaches `icb` through (its resource ID).
    pub container: Mt<Po<dyn MTLBuffer>>,
    /// The index buffer of the last encoding, read through `icb`.
    pub index: parking_lot::Mutex<Option<Buffer>>,
}

/// A batch of draws whose arguments and count the GPU writes (see `t::IndirectDrawsDescriptor`).
#[derive(Clone)]
pub struct IndirectDraws(pub(crate) Arc<IndirectDrawsInner>);

impl IndirectDraws {
    pub fn max_draws(&self) -> u32 {
        self.0.max_draws
    }
}

/// The kernels that encode GPU-driven draws (MSL: indirect command buffers are written from
/// kernels, which WGSL has no words for).
pub(crate) const INDIRECT_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct DrawArgs { uint vertex_count; uint instance_count; uint first_vertex; uint first_instance; };
struct DrawIndexedArgs { uint index_count; uint instance_count; uint first_index; int base_vertex; uint first_instance; };
struct Icb { command_buffer cmds [[id(0)]]; };
struct Params { uint max_draws; uint primitive; uint index_offset; uint index_u32; };

kernel void mx_encode_draws(device const Icb& icb [[buffer(0)]], device const DrawArgs* args [[buffer(1)]],
                            device const uint* count [[buffer(2)]], constant Params& p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= p.max_draws) { return; }
    render_command cmd(icb.cmds, i);
    if (i >= *count) { cmd.reset(); return; }
    DrawArgs a = args[i];
    cmd.draw_primitives(primitive_type(p.primitive), a.first_vertex, a.vertex_count, a.instance_count, a.first_instance);
}

kernel void mx_encode_indexed_draws(device const Icb& icb [[buffer(0)]], device const DrawIndexedArgs* args [[buffer(1)]],
                                    device const uint* count [[buffer(2)]], constant Params& p [[buffer(3)]],
                                    device const uchar* indices [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.max_draws) { return; }
    render_command cmd(icb.cmds, i);
    if (i >= *count) { cmd.reset(); return; }
    DrawIndexedArgs a = args[i];
    device const uchar* base = indices + p.index_offset;
    if (p.index_u32 != 0) {
        cmd.draw_indexed_primitives(primitive_type(p.primitive), a.index_count, (device const uint*)base + a.first_index,
                                    a.instance_count, uint(a.base_vertex), a.first_instance);
    } else {
        cmd.draw_indexed_primitives(primitive_type(p.primitive), a.index_count, (device const ushort*)base + a.first_index,
                                    a.instance_count, uint(a.base_vertex), a.first_instance);
    }
}
"#;

impl Device {
    /// A batch of GPU-driven draws: an indirect command buffer of `max_draws` commands.
    pub fn create_indirect_draws(&self, desc: &t::IndirectDrawsDescriptor<'_>) -> Result<IndirectDraws> {
        if self.shared.bound {
            return Err(Error::Unsupported("GPU-driven draws on a Metal 2 GPU (the bound tier)".into()));
        }
        let d = MTLIndirectCommandBufferDescriptor::new();
        d.setCommandTypes(if desc.indexed { MTLIndirectCommandType::DrawIndexed } else { MTLIndirectCommandType::Draw });
        d.setInheritPipelineState(true);
        d.setInheritBuffers(true);
        d.setInheritDepthStencilState(true);
        d.setInheritDepthBias(true);
        d.setInheritDepthClipMode(true);
        d.setInheritCullMode(true);
        d.setInheritFrontFacingWinding(true);
        d.setInheritTriangleFillMode(true);
        let n = desc.max_draws.max(1) as usize;
        // SAFETY: a filled descriptor; the count is positive.
        let icb = unsafe {
            self.shared
                .device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(&d, n, MTLResourceOptions::StorageModePrivate)
        }
        .ok_or_else(|| Error::OutOfMemory(desc.label.unwrap_or("indirect draws").into()))?;
        if let Some(l) = desc.label {
            icb.setLabel(Some(&crate::ns(l)));
        }
        let container = self
            .shared
            .device
            .newBufferWithLength_options(8, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| Error::OutOfMemory("indirect draws".into()))?;
        // SAFETY: `MTLResourceID` is one `u64`, the layout of a `command_buffer` in a Tier 2
        // argument buffer; the buffer is 8 bytes and unused yet.
        unsafe { container.contents().as_ptr().cast::<MTLResourceID>().write(icb.gpuResourceID()) };
        Ok(IndirectDraws(Arc::new(IndirectDrawsInner {
            max_draws: desc.max_draws,
            indexed: desc.indexed,
            icb: Mt(icb),
            container: Mt(container),
            index: parking_lot::Mutex::new(None),
        })))
    }

    pub(crate) fn indirect_kernels(&self) -> crate::device::IndirectKernels {
        let mut k = self.shared.indirect.lock();
        k.get_or_insert_with(|| {
            let lib = self
                .shared
                .device
                .newLibraryWithSource_options_error(&crate::ns(INDIRECT_KERNELS), None)
                .unwrap_or_else(|e| panic!("mX's indirect draw kernels: {}", e.localizedDescription()));
            let make = |name: &str| {
                let f = lib.newFunctionWithName(&crate::ns(name)).expect("an indirect draw kernel");
                Mt(self
                    .shared
                    .device
                    .newComputePipelineStateWithFunction_error(&f)
                    .unwrap_or_else(|e| panic!("mX's indirect draw kernels: {}", e.localizedDescription())))
            };
            [make("mx_encode_draws"), make("mx_encode_indexed_draws")]
        })
        .clone()
    }
}

//! Buffers, textures, views, samplers, bind groups and queue writes.

use std::ops::Range;
use std::sync::Arc;

use glow::HasContext;
use mx_types as t;

use crate::conv;
use crate::device::{Garbage, Shared};
use crate::heap::{BufferSlot, TextureSlot};
use crate::{Device, Error, Queue, Result, gl_err};

// ---------------------------------------------------------------------------------------------
// Buffers

pub(crate) struct BufferInner {
    pub raw: glow::Buffer,
    pub size: u64,
    pub usage: t::BufferUsages,
    pub shared: Arc<Shared>,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        self.shared.retire(Garbage::Buffer(self.raw));
    }
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
        let gl = self.0.shared.ctx.lock();
        let len = (range.end - range.start) as usize;
        if len == 0 {
            return Ok(f(&[]));
        }
        // SAFETY: a mapping of a live buffer inside its size, read and unmapped before anything
        // else uses the context.
        unsafe {
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(self.0.raw));
            let p = gl.map_buffer_range(glow::COPY_READ_BUFFER, range.start as i32, len as i32, glow::MAP_READ_BIT);
            if p.is_null() {
                return Err(Error::Device("glMapBufferRange failed".into()));
            }
            let r = f(std::slice::from_raw_parts(p, len));
            gl.unmap_buffer(glow::COPY_READ_BUFFER);
            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            Ok(r)
        }
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
        let gl = self.0.shared.ctx.lock();
        let len = (range.end - range.start) as usize;
        if len == 0 {
            return Ok(f(&[]));
        }
        // SAFETY: a mapping of a live buffer inside its size, read and unmapped before anything
        // else uses the context.
        unsafe {
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(self.0.raw));
            let p = gl.map_buffer_range(glow::COPY_READ_BUFFER, range.start as i32, len as i32, glow::MAP_READ_BIT);
            if p.is_null() {
                return Err(Error::Device("glMapBufferRange failed".into()));
            }
            let r = f(std::slice::from_raw_parts(p, len));
            gl.unmap_buffer(glow::COPY_READ_BUFFER);
            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            Ok(r)
        }
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
        let mut bytes = vec![0u8; (range.end - range.start) as usize];
        let r = f(&mut bytes);
        let gl = self.0.shared.ctx.lock();
        // SAFETY: a range inside a live buffer.
        unsafe {
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(self.0.raw));
            gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, range.start as i32, &bytes);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        }
        Ok(r)
    }

    /// Not on GL: GL maps buffers only for the length of a call (persistent mappings need GL
    /// 4.4, which ES never has). Use `map_read` / `map_write`.
    ///
    /// # Safety
    /// Never returns.
    pub unsafe fn mapped_ptr(&self) -> std::ptr::NonNull<u8> {
        panic!("mapped_ptr is not available on OpenGL; use map_read / map_write")
    }

    /// The GL buffer, for interop (use it with `Device::with_gl`).
    pub fn raw(&self) -> glow::Buffer {
        self.0.raw
    }

    /// A range of the buffer in the heap ([`BufferView::index`]). Needs `STORAGE`, and a device
    /// with storage buffers (GL 4.3, ES 3.1).
    pub fn create_view(&self, desc: &t::BufferViewDescriptor<'_>) -> Result<BufferView> {
        let shared = &self.0.shared;
        // (without storage buffers - OpenGL 3.3, ES 3.0 - a view is writable nowhere and read as
        // a uniform block: `bound::views_as_uniforms`)
        if desc.writable && !shared.caps.compute {
            return Err(Error::Unsupported("writable buffer views (OpenGL 4.3 / OpenGL ES 3.1)".into()));
        }
        let size = desc.resolve(self.size(), self.usage(), &shared.limits);
        let slot = shared.heap.lock().add_buffer(BufferSlot {
            raw: self.0.raw,
            offset: desc.offset,
            size,
        })?;
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
        shared.heap.lock().remove_buffer(self.slot, shared.release_after());
    }
}

/// A range of a buffer in the heap.
#[derive(Clone)]
pub struct BufferView(pub(crate) Arc<BufferViewInner>);

impl BufferView {
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
    pub raw: glow::Texture,
    pub target: u32,
    /// A multisampled render target on a context without multisampled textures (ES 3.0): its
    /// samples live here, and `raw` is a name without storage.
    pub rb: Option<glow::Renderbuffer>,
    pub size: t::Extent3d,
    pub format: t::TextureFormat,
    pub dimension: t::TextureDimension,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub usage: t::TextureUsages,
    /// Made outside mX (`texture_from_raw`): not deleted.
    pub external: bool,
    pub shared: Arc<Shared>,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        if !self.external {
            self.shared.retire(Garbage::Texture(self.raw));
            if let Some(rb) = self.rb {
                self.shared.retire(Garbage::Renderbuffer(rb));
            }
        }
    }
}

#[derive(Clone)]
pub struct Texture(pub(crate) Arc<TextureInner>);

impl Texture {
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

    pub(crate) fn layers(&self) -> u32 {
        t::Unwritten::layer_count(self.0.dimension, self.0.size)
    }

    /// The GL texture, for interop.
    pub fn raw(&self) -> glow::Texture {
        self.0.raw
    }

    /// A view. With `TEXTURE_BINDING` usage, or as a storage texture (`STORAGE_BINDING`, one mip
    /// of a 2D view), it gets a heap index ([`TextureView::index`]).
    ///
    /// GL samples a texture whole: a view chooses mips (set when it is bound) but no other
    /// format and no single layer of an array to sample as a 2D texture (those need GL 4.3's
    /// texture views, which ES lacks). Render passes take any single layer and mip.
    pub fn create_view(&self, desc: &t::TextureViewDescriptor<'_>) -> Result<TextureView> {
        let tex = &self.0;
        let layers = self.layers();
        let dimension = desc.dimension.unwrap_or(match (tex.dimension, layers) {
            (t::TextureDimension::D3, _) => t::TextureViewDimension::D3,
            (_, 1) => t::TextureViewDimension::D2,
            (_, 6) if tex.target == glow::TEXTURE_CUBE_MAP => t::TextureViewDimension::Cube,
            _ => t::TextureViewDimension::D2Array,
        });
        let mip_count = desc.mip_level_count.unwrap_or(tex.mip_level_count - desc.base_mip_level);
        let layer_count = desc.array_layer_count.unwrap_or(layers - desc.base_array_layer);
        let format = desc.format.unwrap_or(tex.format);
        let sampled = tex.usage.contains(t::TextureUsages::TEXTURE_BINDING) && !(format.has_stencil() && desc.aspect == t::TextureAspect::All);
        let storage = t::is_heap_storage_view(tex.usage, format, dimension, mip_count);
        let slot = if sampled || storage {
            Some(tex.shared.heap.lock().add_texture(TextureSlot {
                raw: tex.raw,
                target: tex.target,
                base_mip: desc.base_mip_level,
                mip_count,
                depth_stencil_mode: depth_stencil_mode(format, desc.aspect),
                image: storage.then(|| (crate::conv::format(format).internal, desc.base_array_layer)),
            })?)
        } else {
            None
        };
        Ok(TextureView(Arc::new(ViewInner {
            texture: self.clone(),
            format,
            dimension,
            aspect: desc.aspect,
            base_mip_level: desc.base_mip_level,
            mip_level_count: mip_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: layer_count,
            slot,
        })))
    }
}

/// What a view of a depth-stencil texture samples.
fn depth_stencil_mode(format: t::TextureFormat, aspect: t::TextureAspect) -> Option<u32> {
    (format.has_depth() && format.has_stencil()).then(|| if aspect == t::TextureAspect::StencilOnly { glow::STENCIL_INDEX } else { glow::DEPTH_COMPONENT })
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
}

impl ViewInner {
    pub fn binding(&self) -> TextureSlot {
        TextureSlot {
            raw: self.texture.0.raw,
            target: self.texture.0.target,
            base_mip: self.base_mip_level,
            mip_count: self.mip_level_count,
            depth_stencil_mode: depth_stencil_mode(self.format, self.aspect),
            image: None,
        }
    }
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
    pub raw: glow::Sampler,
    slot: u32,
    compare: bool,
    shared: Arc<Shared>,
}

impl Drop for SamplerInner {
    fn drop(&mut self) {
        self.shared.heap.lock().remove_sampler(self.slot, self.shared.release_after());
        self.shared.retire(Garbage::Sampler(self.raw));
    }
}

#[derive(Clone)]
pub struct Sampler(pub(crate) Arc<SamplerInner>);

impl Sampler {
    pub fn index(&self) -> u32 {
        self.0.slot
    }
    pub fn is_comparison(&self) -> bool {
        self.0.compare
    }
}

// ---------------------------------------------------------------------------------------------
// Bind groups (root bindings)

pub struct BufferBinding<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: Option<u64>,
}

pub enum BindingResource<'a> {
    Buffer(BufferBinding<'a>),
    TextureView(&'a TextureView),
    Sampler(&'a Sampler),
    AccelerationStructure(&'a crate::Tlas),
}

pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

pub struct BindGroupDescriptor<'a> {
    pub label: Option<&'a str>,
    pub entries: &'a [BindGroupEntry<'a>],
}

#[derive(Clone)]
pub(crate) enum Bound {
    Buffer { buffer: Buffer, offset: u64, size: u64 },
    Texture(TextureView),
    Sampler(Sampler),
}

#[derive(Clone)]
pub struct BindGroup(pub(crate) Arc<Vec<(u32, Bound)>>);

impl BindGroup {
    pub(crate) fn get(&self, binding: u32) -> Option<&Bound> {
        self.0.iter().find(|(b, _)| *b == binding).map(|(_, r)| r)
    }
}

// ---------------------------------------------------------------------------------------------
// GPU-driven batches: not on this tier

#[derive(Clone)]
pub struct IndirectDraws(());

impl IndirectDraws {
    pub fn max_draws(&self) -> u32 {
        0
    }
}

// ---------------------------------------------------------------------------------------------
// Creation

impl Device {
    pub fn create_buffer(&self, desc: &t::BufferDescriptor<'_>) -> Result<Buffer> {
        let shared = &self.shared;
        let gl = shared.ctx.lock();
        let usage = if desc.usage.intersects(t::BufferUsages::MAP_READ) {
            glow::STREAM_READ
        } else if desc.usage.intersects(t::BufferUsages::MAP_WRITE) {
            glow::STREAM_DRAW
        } else {
            glow::DYNAMIC_DRAW
        };
        // SAFETY: plain creation; a new buffer reads as zeros (wgpu's guarantee).
        let raw = unsafe {
            let raw = gl.create_buffer().map_err(|e| gl_err("buffer", e))?;
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(raw));
            gl.buffer_data_u8_slice(glow::COPY_WRITE_BUFFER, &vec![0u8; desc.size.max(4) as usize], usage);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            if let Some(l) = desc.label
                && gl.supports_debug()
            {
                gl.object_label(glow::BUFFER, raw.0.get(), Some(l));
            }
            raw
        };
        Ok(Buffer(Arc::new(BufferInner {
            raw,
            size: desc.size,
            usage: desc.usage,
            shared: shared.clone(),
        })))
    }

    pub fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Result<Buffer> {
        let b = self.create_buffer(&t::BufferDescriptor {
            label: desc.label,
            size: desc.contents.len() as u64,
            usage: desc.usage | t::BufferUsages::COPY_DST,
        })?;
        Queue { shared: self.shared.clone() }.write_buffer(&b, 0, desc.contents)?;
        Ok(b)
    }

    pub fn create_texture(&self, desc: &t::TextureDescriptor<'_>) -> Result<Texture> {
        let shared = &self.shared;
        if desc.format.is_compressed() && !shared.caps.bc {
            return Err(Error::Unsupported(format!("{:?} (BC compression)", desc.format)));
        }
        let layers = t::Unwritten::layer_count(desc.dimension, desc.size);
        // (a 6-layer 2D texture that may be sampled is made a cube map, as wgpu-hal decides
        // from its views; mX decides at creation)
        let cube = desc.view_formats.is_empty() && layers == 6 && desc.size.width == desc.size.height && desc.usage.contains(t::TextureUsages::TEXTURE_BINDING);
        let target = conv::texture_target(desc.dimension, layers, desc.sample_count, cube);
        // (ES 3.0 has multisampled renderbuffers only: enough for a render target that is
        // resolved, as MSAA is; ES 3.0 drivers - ANGLE on older chips - drew it black before)
        let as_renderbuffer = target == glow::TEXTURE_2D_MULTISAMPLE && !shared.caps.multisample_textures;
        if as_renderbuffer
            && desc
                .usage
                .intersects(t::TextureUsages::TEXTURE_BINDING | t::TextureUsages::STORAGE_BINDING | t::TextureUsages::COPY_SRC | t::TextureUsages::COPY_DST)
        {
            return Err(Error::Unsupported("multisampled textures that are sampled or copied (GL 3.2 / OpenGL ES 3.1)".into()));
        }
        let f = conv::format(desc.format);
        let gl = shared.ctx.lock();
        let mut rb = None;
        let (w, h) = (desc.size.width as i32, desc.size.height as i32);
        // SAFETY: plain creation and allocation of the texture's storage.
        let raw = unsafe {
            let raw = gl.create_texture().map_err(|e| gl_err("texture", e))?;
            gl.bind_texture(target, Some(raw));
            let levels = desc.mip_level_count as i32;
            match target {
                glow::TEXTURE_2D_MULTISAMPLE if as_renderbuffer => {
                    let r = gl.create_renderbuffer().map_err(|e| gl_err("renderbuffer", e))?;
                    gl.bind_renderbuffer(glow::RENDERBUFFER, Some(r));
                    gl.renderbuffer_storage_multisample(glow::RENDERBUFFER, desc.sample_count as i32, f.internal, w, h);
                    gl.bind_renderbuffer(glow::RENDERBUFFER, None);
                    rb = Some(r);
                }
                glow::TEXTURE_2D_MULTISAMPLE => gl.tex_storage_2d_multisample(target, desc.sample_count as i32, f.internal, w, h, true),
                glow::TEXTURE_2D | glow::TEXTURE_CUBE_MAP => gl.tex_storage_2d(target, levels, f.internal, w, h),
                glow::TEXTURE_2D_ARRAY => gl.tex_storage_3d(target, levels, f.internal, w, h, layers as i32),
                _ => gl.tex_storage_3d(target, levels, f.internal, w, h, desc.size.depth_or_array_layers as i32),
            }
            if target != glow::TEXTURE_2D_MULTISAMPLE {
                gl.tex_parameter_i32(target, glow::TEXTURE_MAX_LEVEL, levels - 1);
                // (integer and depth formats sample only with nearest filtering in GL's
                // defaults; mX's samplers set the filters)
                gl.tex_parameter_i32(target, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(target, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            }
            gl.bind_texture(target, None);
            if let Some(l) = desc.label
                && gl.supports_debug()
            {
                gl.object_label(glow::TEXTURE, raw.0.get(), Some(l));
            }
            raw
        };
        let tex = Texture(Arc::new(TextureInner {
            raw,
            target,
            rb,
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            external: false,
            shared: shared.clone(),
        }));
        crate::command::zero_texture(&gl, shared, &tex);
        Ok(tex)
    }

    /// A texture GL objects made outside mX (an XR swapchain image): kept, never deleted.
    ///
    /// # Safety
    /// `raw` is a texture of the device's context (or one shared with it) made as `desc` says.
    pub unsafe fn texture_from_raw(&self, raw: glow::Texture, desc: &t::TextureDescriptor<'_>) -> Texture {
        let layers = t::Unwritten::layer_count(desc.dimension, desc.size);
        Texture(Arc::new(TextureInner {
            raw,
            target: conv::texture_target(desc.dimension, layers, desc.sample_count, false),
            rb: None,
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            external: true,
            shared: self.shared.clone(),
        }))
    }

    pub fn create_sampler(&self, desc: &t::SamplerDescriptor<'_>) -> Result<Sampler> {
        let shared = &self.shared;
        let gl = shared.ctx.lock();
        // SAFETY: plain creation and parameters.
        let raw = unsafe {
            let s = gl.create_sampler().map_err(|e| gl_err("sampler", e))?;
            gl.sampler_parameter_i32(s, glow::TEXTURE_WRAP_S, conv::address(desc.address_mode_u) as i32);
            gl.sampler_parameter_i32(s, glow::TEXTURE_WRAP_T, conv::address(desc.address_mode_v) as i32);
            gl.sampler_parameter_i32(s, glow::TEXTURE_WRAP_R, conv::address(desc.address_mode_w) as i32);
            gl.sampler_parameter_i32(s, glow::TEXTURE_MAG_FILTER, conv::filter(desc.mag_filter) as i32);
            gl.sampler_parameter_i32(s, glow::TEXTURE_MIN_FILTER, conv::min_filter(desc.min_filter, desc.mipmap_filter) as i32);
            gl.sampler_parameter_f32(s, glow::TEXTURE_MIN_LOD, desc.lod_min_clamp);
            gl.sampler_parameter_f32(s, glow::TEXTURE_MAX_LOD, desc.lod_max_clamp);
            if desc.anisotropy_clamp > 1 {
                gl.sampler_parameter_f32(s, glow::TEXTURE_MAX_ANISOTROPY_EXT, desc.anisotropy_clamp as f32);
            }
            if let Some(c) = desc.compare {
                gl.sampler_parameter_i32(s, glow::TEXTURE_COMPARE_MODE, glow::COMPARE_REF_TO_TEXTURE as i32);
                gl.sampler_parameter_i32(s, glow::TEXTURE_COMPARE_FUNC, conv::compare(c) as i32);
            }
            if let Some(b) = desc.border_color {
                let c = match b {
                    t::SamplerBorderColor::TransparentBlack => [0.0, 0.0, 0.0, 0.0],
                    t::SamplerBorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
                    t::SamplerBorderColor::OpaqueWhite => [1.0; 4],
                };
                gl.sampler_parameter_f32_slice(s, glow::TEXTURE_BORDER_COLOR, &c);
            }
            s
        };
        drop(gl);
        let slot = shared.heap.lock().add_sampler(raw)?;
        Ok(Sampler(Arc::new(SamplerInner {
            raw,
            slot,
            compare: desc.compare.is_some(),
            shared: shared.clone(),
        })))
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup {
        let entries = desc
            .entries
            .iter()
            .map(|e| {
                let r = match &e.resource {
                    BindingResource::Buffer(b) => Bound::Buffer {
                        buffer: b.buffer.clone(),
                        offset: b.offset,
                        size: b.size.unwrap_or(b.buffer.size() - b.offset),
                    },
                    BindingResource::TextureView(v) => Bound::Texture((*v).clone()),
                    BindingResource::Sampler(s) => Bound::Sampler((*s).clone()),
                    BindingResource::AccelerationStructure(_) => panic!("acceleration structures need a device with bindless (Metal 3, Vulkan, Direct3D 12)"),
                };
                (e.binding, r)
            })
            .collect();
        BindGroup(Arc::new(entries))
    }

    /// Not on this tier (see `Adapter::supports_indirect_count`).
    pub fn create_indirect_draws(&self, _desc: &t::IndirectDrawsDescriptor<'_>) -> Result<IndirectDraws> {
        Err(Error::Unsupported("GPU-driven draw batches".into()))
    }
}

// ---------------------------------------------------------------------------------------------
// Queue writes: run at once on the context, so after every submission before and before every
// submission after, as wgpu promises.

impl Queue {
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        assert!(offset + data.len() as u64 <= buffer.size(), "write_buffer past the end of the buffer");
        let gl = self.shared.ctx.lock();
        // SAFETY: a range inside a live buffer.
        unsafe {
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer.0.raw));
            gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset as i32, data);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        }
        Ok(())
    }

    pub fn write_texture(&self, dst: t::TexelCopyTextureInfo<'_, Texture>, data: &[u8], layout: t::TexelCopyBufferLayout, size: t::Extent3d) -> Result<()> {
        let gl = self.shared.ctx.lock();
        crate::command::upload(
            &gl,
            dst.texture,
            dst.mip_level,
            dst.origin,
            size,
            layout,
            crate::command::Source::Bytes(&data[layout.offset as usize..]),
        );
        Ok(())
    }
}

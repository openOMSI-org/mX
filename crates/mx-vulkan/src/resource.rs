//! Buffers, textures, views, samplers and bind groups.

use std::ops::Range;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use mx_types as t;

use crate::device::{Garbage, Keep, Shared};
use crate::{Device, Error, Queue, Result, conv, vk_err};

// ---------------------------------------------------------------------------------------------
// Buffers

pub(crate) struct BufferInner {
    pub raw: vk::Buffer,
    alloc: Option<Allocation>,
    pub size: u64,
    pub usage: t::BufferUsages,
    pub shared: Arc<Shared>,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        if let Some(a) = self.alloc.take() {
            self.shared.retire(Garbage::Buffer(self.raw, a));
        }
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

    fn mapped(&self, range: &Range<u64>, what: &str) -> *mut u8 {
        assert!(
            range.start <= range.end && range.end <= self.0.size,
            "{what} range {range:?} outside the buffer ({} bytes)",
            self.0.size
        );
        let base = self.0.alloc.as_ref().and_then(|a| a.mapped_ptr()).expect("mappable buffers are mapped").as_ptr().cast::<u8>();
        // SAFETY: inside the mapping (checked above).
        unsafe { base.add(range.start as usize) }
    }

    /// Read the buffer's contents. Waits until the GPU has finished all submitted work, so
    /// the bytes are those the last submission left. Needs `MAP_READ`.
    pub fn map_read<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        assert!(self.0.usage.contains(t::BufferUsages::MAP_READ), "map_read on a buffer without MAP_READ");
        let p = self.mapped(&range, "map_read");
        self.0.shared.wait_idle()?;
        // SAFETY: host-visible, coherent, in range, and the GPU is idle.
        let bytes = unsafe { std::slice::from_raw_parts(p, (range.end - range.start) as usize) };
        Ok(f(bytes))
    }

    /// Read the buffer's contents now, without waiting: the caller knows the submissions
    /// that wrote it are done (`Device::is_done`), as after wgpu's `map_async`. Needs `MAP_READ`.
    pub fn map_read_now<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        assert!(self.0.usage.contains(t::BufferUsages::MAP_READ), "map_read on a buffer without MAP_READ");
        let p = self.mapped(&range, "map_read");
        // SAFETY: host-visible, coherent, in range, and the GPU is idle.
        let bytes = unsafe { std::slice::from_raw_parts(p, (range.end - range.start) as usize) };
        Ok(f(bytes))
    }

    /// Write the buffer's contents from the CPU. Waits until the GPU has finished all
    /// submitted work first. Needs `MAP_WRITE`.
    pub fn map_write<R>(&self, range: Range<u64>, f: impl FnOnce(&mut [u8]) -> R) -> Result<R> {
        assert!(self.0.usage.contains(t::BufferUsages::MAP_WRITE), "map_write on a buffer without MAP_WRITE");
        let p = self.mapped(&range, "map_write");
        self.0.shared.wait_idle()?;
        // SAFETY: as in `map_read`; `&self` hands out no other slice (a second `map_write`
        // waits here before taking one).
        let bytes = unsafe { std::slice::from_raw_parts_mut(p, (range.end - range.start) as usize) };
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
        self.0.alloc.as_ref().and_then(|a| a.mapped_ptr()).expect("mappable buffers are mapped").cast()
    }

    /// The `VkBuffer`, for interop.
    pub fn raw(&self) -> vk::Buffer {
        self.0.raw
    }

    /// A range of the buffer in the bindless heap ([`BufferView::index`]). Needs `STORAGE`.
    pub fn create_view(&self, desc: &t::BufferViewDescriptor<'_>) -> Result<BufferView> {
        let shared = &self.0.shared;
        let size = desc.resolve(self.size(), self.usage(), &shared.limits);
        if size > shared.caps.max_storage_buffer_range {
            return Err(Error::Unsupported(format!(
                "a buffer view of {size} bytes (the device's maxStorageBufferRange is {})",
                shared.caps.max_storage_buffer_range
            )));
        }
        let slot = shared.heap.lock().add_buffer(&shared.raw, self.0.raw, desc.offset, size)?;
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
    pub raw: vk::Image,
    alloc: Option<Allocation>,
    /// Made by mX (else wrapped: a swapchain or external image, not destroyed here).
    owned: bool,
    pub size: t::Extent3d,
    pub format: t::TextureFormat,
    pub dimension: t::TextureDimension,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub usage: t::TextureUsages,
    /// What nothing has written yet of a texture made by mX (zeroed at the next submission).
    pub unwritten: parking_lot::Mutex<Option<t::Unwritten>>,
    pub shared: Arc<Shared>,
    /// What a wrapped image needs alive (its swapchain).
    _keep: Option<Keep>,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        if self.owned
            && let Some(a) = self.alloc.take()
        {
            self.shared.retire(Garbage::Image(self.raw, a));
        }
    }
}

#[derive(Clone)]
pub struct Texture(pub(crate) Arc<TextureInner>);

impl Texture {
    pub(crate) fn wrap(raw: vk::Image, desc: &t::TextureDescriptor<'_>, shared: Arc<Shared>, keep: Option<Keep>) -> Texture {
        Texture(Arc::new(TextureInner {
            raw,
            alloc: None,
            owned: false,
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            unwritten: parking_lot::Mutex::new(None),
            shared,
            _keep: keep,
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

    /// The image's memory and its offset in it (None for a wrapped image), for interop
    /// (Streamline names both). Every image of mX's rests in `GENERAL` between commands.
    pub fn raw_memory(&self) -> Option<(vk::DeviceMemory, u64)> {
        // SAFETY: the memory outlives the texture's use; only its handle is read.
        self.0.alloc.as_ref().map(|a| (unsafe { a.memory() }, a.offset()))
    }

    pub(crate) fn layers(&self) -> u32 {
        if self.0.dimension == t::TextureDimension::D3 { 1 } else { self.0.size.depth_or_array_layers }
    }

    /// A view. With `TEXTURE_BINDING` usage, or as a storage texture (`STORAGE_BINDING`, one mip
    /// of a 2D view), it gets a bindless index ([`TextureView::index`]).
    pub fn create_view(&self, desc: &t::TextureViewDescriptor<'_>) -> Result<TextureView> {
        let tex = &self.0;
        let shared = &tex.shared;
        let format = desc.format.unwrap_or(tex.format);
        let layers = self.layers();
        let dimension = desc.dimension.unwrap_or_else(|| conv::default_view_dimension(tex.dimension, layers));
        let mip_count = desc.mip_level_count.unwrap_or(tex.mip_level_count - desc.base_mip_level);
        let layer_count = desc.array_layer_count.unwrap_or(layers - desc.base_array_layer);
        let aspect = conv::aspects(format, desc.aspect);
        let info = vk::ImageViewCreateInfo::default()
            .image(tex.raw)
            .view_type(conv::view_type(dimension))
            .format(conv::format(format, shared.caps.d24s8))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: desc.base_mip_level,
                level_count: mip_count,
                base_array_layer: desc.base_array_layer,
                layer_count,
            });
        // (a view in a format that cannot be a storage image - the sRGB view of an `Rgba8Unorm`
        // storage texture, sampled only - leaves `STORAGE` out of its usage, which it would
        // inherit from the image: VUID-VkImageViewCreateInfo-usage-02275)
        let mut view_usage = vk::ImageViewUsageCreateInfo::default()
            .usage((conv::image_usage(tex.usage, tex.format) | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST) & !vk::ImageUsageFlags::STORAGE);
        let info = if tex.owned && tex.usage.contains(t::TextureUsages::STORAGE_BINDING) && !t::is_storage_view_format(format) {
            info.push_next(&mut view_usage)
        } else {
            info
        };
        // SAFETY: the ranges are inside the image (the caller's contract, checked by the
        // validation layer); the format is one of its view formats.
        let raw = unsafe { shared.raw.create_image_view(&info, None) }.map_err(|e| vk_err("texture view", e))?;
        if let Some(l) = desc.label {
            shared.name(raw, l);
        }
        let sampled = tex.usage.contains(t::TextureUsages::TEXTURE_BINDING) && !(format.has_stencil() && desc.aspect == t::TextureAspect::All);
        let storage = t::is_heap_storage_view(tex.usage, format, dimension, mip_count);
        let slot = if sampled || storage {
            let bindings = if sampled {
                crate::heap::bindings_for(dimension, format, desc.aspect, tex.sample_count)
            } else {
                0
            };
            match shared.heap.lock().add_texture(&shared.raw, raw, bindings, storage) {
                Ok(s) => Some(s),
                Err(e) => {
                    // SAFETY: just made, unused.
                    unsafe { shared.raw.destroy_image_view(raw, None) };
                    return Err(e);
                }
            }
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

    /// The `VkImage`, for interop. Images are kept in `GENERAL` layout.
    pub fn raw(&self) -> vk::Image {
        self.0.raw
    }
}

pub(crate) struct ViewInner {
    pub raw: vk::ImageView,
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
        let shared = &self.texture.0.shared;
        if let Some(slot) = self.slot {
            shared.heap.lock().remove_texture(slot, shared.release_after());
        }
        shared.retire(Garbage::View(self.raw));
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

    /// The `VkImageView`, for interop.
    pub fn raw(&self) -> vk::ImageView {
        self.0.raw
    }

    /// The size of the view's first mip level.
    pub(crate) fn extent(&self) -> t::Extent3d {
        let tex = &self.0.texture;
        tex.size().mip_level_size(self.0.base_mip_level, tex.dimension())
    }
}

// ---------------------------------------------------------------------------------------------
// Samplers

pub(crate) struct SamplerInner {
    pub raw: vk::Sampler,
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
    /// The sampler's index in the heap: `mx_samplers[i]`, or `mx_samplers_cmp[i]` for a
    /// comparison sampler.
    pub fn index(&self) -> u32 {
        self.0.slot
    }

    pub fn is_comparison(&self) -> bool {
        self.0.compare
    }

    /// The `VkSampler`, for interop.
    pub fn raw(&self) -> vk::Sampler {
        self.0.raw
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
        let shared = &self.shared;
        let location = if desc.usage.contains(t::BufferUsages::MAP_READ) {
            MemoryLocation::GpuToCpu
        } else if desc.usage.contains(t::BufferUsages::MAP_WRITE) {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        };
        let name = desc.label.unwrap_or("buffer");
        let (raw, mut alloc) = shared.create_raw_buffer(desc.size.max(4), conv::buffer_usage(desc.usage, shared.caps.ray_query), location, name)?;
        if let Some(l) = desc.label {
            shared.name(raw, l);
        }
        // a new buffer reads as zeros (wgpu's guarantee; the allocator reuses memory)
        match alloc.mapped_slice_mut() {
            Some(bytes) => bytes.fill(0),
            None => shared.with_pending(|cmd, _| {
                // SAFETY: the whole of a new buffer, which has TRANSFER_DST.
                unsafe { shared.raw.cmd_fill_buffer(cmd, raw, 0, vk::WHOLE_SIZE, 0) }
            }),
        }
        Ok(Buffer(Arc::new(BufferInner {
            raw,
            alloc: Some(alloc),
            size: desc.size,
            usage: desc.usage,
            shared: shared.clone(),
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
            self.queue().write_buffer(&buffer, 0, desc.contents)?;
        }
        Ok(buffer)
    }

    pub fn create_texture(&self, desc: &t::TextureDescriptor<'_>) -> Result<Texture> {
        let shared = &self.shared;
        if desc.format.is_compressed() && !shared.caps.bc {
            return Err(Error::Unsupported(format!("{:?}", desc.format)));
        }
        let layers = desc.size.depth_or_array_layers;
        let (ty, depth, array) = match desc.dimension {
            t::TextureDimension::D1 => (vk::ImageType::TYPE_1D, 1, layers),
            t::TextureDimension::D2 => (vk::ImageType::TYPE_2D, 1, layers),
            t::TextureDimension::D3 => (vk::ImageType::TYPE_3D, layers, 1),
        };
        let mut flags = vk::ImageCreateFlags::empty();
        // (as wgpu-hal: a square 2D texture of 6n layers may be viewed as cubes)
        if desc.dimension == t::TextureDimension::D2 && layers >= 6 && layers.is_multiple_of(6) && desc.size.width == desc.size.height && desc.sample_count == 1 {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        if desc.view_formats.iter().any(|f| *f != desc.format) {
            flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        let format = conv::format(desc.format, shared.caps.d24s8);
        let usage = conv::image_usage(desc.usage, desc.format) | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(ty)
            .format(format)
            .extent(vk::Extent3D {
                width: desc.size.width,
                height: desc.size.height,
                depth,
            })
            .mip_levels(desc.mip_level_count)
            .array_layers(array)
            .samples(conv::samples(desc.sample_count))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let name = desc.label.unwrap_or("texture");
        // SAFETY: plain creation and binding of a new image.
        let (raw, alloc) = unsafe {
            let raw = shared.raw.create_image(&info, None).map_err(|e| vk_err(name, e))?;
            let req = shared.raw.get_image_memory_requirements(raw);
            let alloc = shared.allocator.lock().allocate(&AllocationCreateDesc {
                name,
                requirements: req,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
            let alloc = match alloc {
                Ok(a) => a,
                Err(e) => {
                    shared.raw.destroy_image(raw, None);
                    return Err(crate::alloc_err(name, e));
                }
            };
            if let Err(e) = shared.raw.bind_image_memory(raw, alloc.memory(), alloc.offset()) {
                shared.raw.destroy_image(raw, None);
                let _ = shared.allocator.lock().free(alloc);
                return Err(vk_err(name, e));
            }
            (raw, alloc)
        };
        if let Some(l) = desc.label {
            shared.name(raw, l);
        }
        // (to GENERAL, where every image lives, and zeroed at the next submission)
        let inner = Arc::new(TextureInner {
            raw,
            alloc: Some(alloc),
            owned: true,
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            unwritten: parking_lot::Mutex::new(Some(t::Unwritten::new(desc))),
            shared: shared.clone(),
            _keep: None,
        });
        shared.fresh.lock().push(Arc::downgrade(&inner));
        Ok(Texture(inner))
    }

    /// Wrap an image made outside mX (an XR swapchain image, an upscaler's output).
    ///
    /// # Safety
    /// `desc` must describe `raw` (size, format, mips, samples, usage), the image must be in
    /// `GENERAL` layout whenever mX uses it, and it must outlive the texture.
    pub unsafe fn texture_from_raw(&self, raw: vk::Image, desc: &t::TextureDescriptor<'_>) -> Texture {
        Texture::wrap(raw, desc, self.shared.clone(), None)
    }

    pub fn create_sampler(&self, desc: &t::SamplerDescriptor<'_>) -> Result<Sampler> {
        let shared = &self.shared;
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(conv::filter(desc.mag_filter))
            .min_filter(conv::filter(desc.min_filter))
            .mipmap_mode(conv::mip_filter(desc.mipmap_filter))
            .address_mode_u(conv::address(desc.address_mode_u))
            .address_mode_v(conv::address(desc.address_mode_v))
            .address_mode_w(conv::address(desc.address_mode_w))
            .min_lod(desc.lod_min_clamp)
            .max_lod(desc.lod_max_clamp);
        if let Some(c) = desc.compare {
            info = info.compare_enable(true).compare_op(conv::compare(c));
        }
        if desc.anisotropy_clamp > 1 && shared.caps.anisotropy {
            info = info.anisotropy_enable(true).max_anisotropy(desc.anisotropy_clamp as f32);
        }
        if let Some(b) = desc.border_color {
            info = info.border_color(conv::border(b));
        }
        // SAFETY: plain creation.
        let raw = unsafe { shared.raw.create_sampler(&info, None) }.map_err(|e| vk_err("sampler", e))?;
        if let Some(l) = desc.label {
            shared.name(raw, l);
        }
        let slot = match shared.heap.lock().add_sampler(&shared.raw, raw, desc.compare.is_some()) {
            Ok(s) => s,
            Err(e) => {
                // SAFETY: just made, unused.
                unsafe { shared.raw.destroy_sampler(raw, None) };
                return Err(e);
            }
        };
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
        let shared = &self.shared;
        shared.with_pending(|cmd, rec| {
            let (staging, at) = shared.stage(rec, data)?;
            let region = [vk::BufferCopy {
                src_offset: at,
                dst_offset: offset,
                size: data.len() as u64,
            }];
            // SAFETY: both ranges are in their buffers.
            unsafe { shared.raw.cmd_copy_buffer(cmd, staging, buffer.0.raw, &region) };
            rec.keep.push(Box::new(buffer.clone()));
            Ok(())
        })
    }

    /// Copy texels from `data` into a texture before the next submission's commands run.
    pub fn write_texture(&self, dst: t::TexelCopyTextureInfo<'_, Texture>, data: &[u8], layout: t::TexelCopyBufferLayout, size: t::Extent3d) -> Result<()> {
        let shared = &self.shared;
        let tex = dst.texture;
        if let Some(u) = tex.0.unwritten.lock().as_mut() {
            u.written(tex.format(), tex.dimension(), tex.size(), dst.mip_level, dst.origin, size);
        }
        // (the staging buffer starts at `layout.offset` of `data`)
        shared.with_pending(|cmd, rec| {
            let (staging, at) = shared.stage(rec, &data[layout.offset as usize..])?;
            let region = [buffer_image_region(&dst, layout, at, size)];
            // SAFETY: the region is inside the texture and the staging buffer (the caller's
            // contract, checked by the validation layer).
            unsafe { shared.raw.cmd_copy_buffer_to_image(cmd, staging, dst.texture.0.raw, vk::ImageLayout::GENERAL, &region) };
            rec.keep.push(Box::new(dst.texture.clone()));
            Ok(())
        })
    }
}

/// `size` at `origin` of mip `level`, cut to the level's own size: a compressed format's
/// copy covers whole blocks (4×4 into a 1×1 mip), Vulkan wants the level's real extent
/// there (as wgpu-hal passes it).
pub(crate) fn clamp_to_mip(tex: &Texture, level: u32, origin: t::Origin3d, size: t::Extent3d) -> t::Extent3d {
    let at = |full: u32| (full >> level).max(1);
    let full = tex.size();
    let depth = if tex.dimension() == t::TextureDimension::D3 {
        at(full.depth_or_array_layers).saturating_sub(origin.z).min(size.depth_or_array_layers)
    } else {
        size.depth_or_array_layers
    };
    t::Extent3d {
        width: at(full.width).saturating_sub(origin.x).min(size.width),
        height: at(full.height).saturating_sub(origin.y).min(size.height),
        depth_or_array_layers: depth,
    }
}

/// One buffer ↔ texture copy region. The texels start at `buffer_offset` in the buffer
/// (`layout.offset` is not applied again).
pub(crate) fn buffer_image_region(at: &t::TexelCopyTextureInfo<'_, Texture>, layout: t::TexelCopyBufferLayout, buffer_offset: u64, size: t::Extent3d) -> vk::BufferImageCopy {
    let tex = at.texture;
    let format = tex.format();
    let (bw, bh) = format.block_dimensions();
    let block = format.block_copy_size().unwrap_or(match at.aspect {
        t::TextureAspect::StencilOnly => 1,
        _ => 4,
    });
    let block = if format.has_stencil() && at.aspect == t::TextureAspect::StencilOnly { 1 } else { block };
    let is_3d = tex.dimension() == t::TextureDimension::D3;
    let mut aspect = conv::aspects(format, at.aspect);
    if aspect == vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL {
        // (a copy takes one aspect; `All` of a depth-stencil texture means its depth)
        aspect = vk::ImageAspectFlags::DEPTH;
    }
    vk::BufferImageCopy {
        buffer_offset,
        buffer_row_length: layout.bytes_per_row.map_or(0, |b| b / block * bw),
        buffer_image_height: layout.rows_per_image.map_or(0, |r| r * bh),
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: at.mip_level,
            base_array_layer: if is_3d { 0 } else { at.origin.z },
            layer_count: if is_3d { 1 } else { size.depth_or_array_layers },
        },
        image_offset: conv::origin(at.origin, is_3d),
        image_extent: conv::extent(clamp_to_mip(tex, at.mip_level, at.origin, size), is_3d),
    }
}

// ---------------------------------------------------------------------------------------------
// GPU-driven draws

/// Where a batch of GPU-driven draws reads its arguments (see `encode_indirect_draws`).
#[derive(Clone)]
pub(crate) struct IndirectSource {
    pub args: Buffer,
    pub args_offset: u64,
    pub count: Buffer,
    pub count_offset: u64,
    pub index: Option<(Buffer, u64, t::IndexFormat)>,
}

pub(crate) struct IndirectDrawsInner {
    pub max_draws: u32,
    pub indexed: bool,
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

impl Device {
    /// A batch of GPU-driven draws. Needs `drawIndirectCount` (`Adapter::supports_indirect_count`).
    pub fn create_indirect_draws(&self, desc: &t::IndirectDrawsDescriptor<'_>) -> Result<IndirectDraws> {
        if !self.shared.caps.draw_indirect_count {
            return Err(Error::Unsupported("GPU-driven draw counts (drawIndirectCount)".into()));
        }
        Ok(IndirectDraws(Arc::new(IndirectDrawsInner {
            max_draws: desc.max_draws,
            indexed: desc.indexed,
            source: parking_lot::Mutex::new(None),
        })))
    }
}

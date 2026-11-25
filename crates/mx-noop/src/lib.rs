//! mX's no-op backend: a device that takes every call and draws nothing, as wgpu's `noop`
//! backend did. A dedicated server opens it (it draws no picture but runs the renderer's
//! bookkeeping), and so do tests that need a device but no GPU.
//!
//! Objects are made and checked as on the other backends: shaders are parsed and validated,
//! heap indices are handed out and taken back, buffers the CPU maps (`MAP_READ`, `MAP_WRITE`)
//! keep their bytes and the commands that copy into them run at `submit`. Everything else is
//! dropped: textures hold no texels, draws and dispatches do nothing. Every submission is
//! done as soon as it is submitted.

use std::ops::{Index, IndexMut, Range};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use mx_types as t;
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no device (the no-op backend always has one)")]
    NoDevice,
    #[error("out of GPU memory creating {0}")]
    OutOfMemory(String),
    #[error("{0} is not supported on this device")]
    Unsupported(String),
    #[error("the bindless heap is full: {0}")]
    HeapFull(&'static str),
    #[error(transparent)]
    Shader(#[from] mx_shader::ShaderError),
    #[error("{label}: the shader did not compile: {message}")]
    Compile { label: String, message: String },
    #[error("{label}: the pipeline was not made: {message}")]
    Pipeline { label: String, message: String },
    #[error("the GPU reported an error: {0}")]
    Device(String),
    /// Never returned: kept so that every backend's errors have the same shape.
    #[error("the GPU was lost: {0}")]
    DeviceLost(String),
    #[error("surface: {0}")]
    Surface(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Objects compare and hash by identity (the same object, not equal contents).
macro_rules! identity {
    ($($t:ty),*) => {$(
        impl PartialEq for $t {
            fn eq(&self, o: &Self) -> bool {
                Arc::ptr_eq(&self.0, &o.0)
            }
        }
        impl Eq for $t {}
        impl std::hash::Hash for $t {
            fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
                (Arc::as_ptr(&self.0) as *const u8 as usize).hash(h)
            }
        }
    )*};
}

// ---------------------------------------------------------------------------------------------
// The heap: indices handed out and taken back, as on the other backends.

struct Slots {
    next: u32,
    free: Vec<u32>,
    cap: u32,
    in_use: usize,
}

impl Slots {
    fn new(cap: u32) -> Slots {
        Slots {
            next: 0,
            free: Vec::new(),
            cap,
            in_use: 0,
        }
    }
    fn take(&mut self, what: &'static str) -> Result<u32> {
        let i = match self.free.pop() {
            Some(i) => i,
            None if self.next < self.cap => {
                self.next += 1;
                self.next - 1
            }
            None => return Err(Error::HeapFull(what)),
        };
        self.in_use += 1;
        Ok(i)
    }
    fn give(&mut self, i: u32) {
        self.in_use -= 1;
        self.free.push(i);
    }
}

struct Shared {
    textures: Mutex<Slots>,
    samplers: Mutex<Slots>,
    buffers: Mutex<Slots>,
    submitted: AtomicU64,
    limits: t::Limits,
}

// ---------------------------------------------------------------------------------------------
// Instance, adapter, device, queue

#[derive(Clone, Default)]
pub struct Instance(());

impl Instance {
    pub fn new() -> Instance {
        Instance(())
    }

    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        vec![Adapter(())]
    }

    pub fn default_adapter(&self) -> Result<Adapter> {
        Ok(Adapter(()))
    }

    /// A surface whose frames are textures of the device's that nothing shows.
    pub fn create_surface<'w>(&self, window: impl HasWindowHandle + HasDisplayHandle + Send + Sync + 'w) -> Result<Surface<'w>> {
        let _ = window;
        Ok(Surface {
            state: Mutex::new(None),
            _lifetime: std::marker::PhantomData,
        })
    }

    /// # Safety
    /// Nothing is done with the handles.
    pub unsafe fn create_surface_unsafe(&self, _display: raw_window_handle::RawDisplayHandle, _window: raw_window_handle::RawWindowHandle) -> Result<Surface<'static>> {
        Ok(Surface {
            state: Mutex::new(None),
            _lifetime: std::marker::PhantomData,
        })
    }
}

#[derive(Clone)]
pub struct Adapter(());

impl Adapter {
    pub fn info(&self) -> t::AdapterInfo {
        t::AdapterInfo {
            name: "mX no-op".into(),
            backend: t::Backend::Noop,
            device_type: t::DeviceType::Cpu,
            vendor: 0,
            device: 0,
            driver: String::new(),
        }
    }
    pub fn limits(&self) -> t::Limits {
        t::Limits::default()
    }
    pub fn memory_info(&self) -> Option<t::AdapterMemory> {
        None
    }
    pub fn supports_msaa(&self, _format: t::TextureFormat, samples: u32) -> bool {
        matches!(samples, 1 | 2 | 4 | 8)
    }
    pub fn texture_units(&self) -> (u32, u32, u32) {
        (u32::MAX, u32::MAX, u32::MAX)
    }
    pub fn supports_base_vertex(&self) -> bool {
        true
    }
    pub fn supports_bc(&self) -> bool {
        true
    }
    pub fn supports_ray_query(&self) -> bool {
        false
    }
    pub fn supports_timestamps(&self) -> bool {
        false
    }
    /// Whether this GPU runs on the bound tier: the heap bound slot by slot at each draw,
    /// heap accesses indexed by immediates or constants only.
    pub fn is_bound_tier(&self) -> bool {
        false
    }
    pub fn supports_indirect_count(&self) -> bool {
        true
    }
    pub fn supports_compute(&self) -> bool {
        true
    }
    pub fn storage_buffers_per_stage(&self) -> (u32, u32) {
        (u32::MAX, u32::MAX)
    }

    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Result<(Device, Queue)> {
        let mut limits = t::Limits::default().capped(desc.limits.as_ref());
        limits.max_bindless_textures = desc.bindless_textures;
        limits.max_bindless_samplers = desc.bindless_samplers;
        limits.max_bindless_buffers = desc.bindless_buffers;
        let shared = Arc::new(Shared {
            textures: Mutex::new(Slots::new(desc.bindless_textures)),
            samplers: Mutex::new(Slots::new(desc.bindless_samplers)),
            buffers: Mutex::new(Slots::new(desc.bindless_buffers)),
            submitted: AtomicU64::new(0),
            limits,
        });
        Ok((
            Device {
                shared: shared.clone(),
                info: self.info(),
            },
            Queue { shared },
        ))
    }
}

#[derive(Clone, Debug)]
pub struct DeviceDescriptor<'a> {
    pub label: Option<&'a str>,
    pub bindless_textures: u32,
    pub bindless_samplers: u32,
    pub bindless_buffers: u32,
    /// Limits the device keeps to, where lower than the adapter's (a weaker chip's, to
    /// test on); None: the adapter's.
    pub limits: Option<t::Limits>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionIndex(pub u64);

#[derive(Clone)]
pub struct Device {
    shared: Arc<Shared>,
    info: t::AdapterInfo,
}

impl PartialEq for Device {
    fn eq(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &o.shared)
    }
}
impl Eq for Device {}
impl std::hash::Hash for Device {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        (Arc::as_ptr(&self.shared) as *const u8 as usize).hash(h)
    }
}

impl Device {
    pub fn info(&self) -> &t::AdapterInfo {
        &self.info
    }
    pub fn limits(&self) -> t::Limits {
        self.shared.limits
    }
    pub fn poll(&self) -> Result<()> {
        Ok(())
    }
    pub fn wait(&self, _index: SubmissionIndex) -> Result<()> {
        Ok(())
    }
    pub fn wait_idle(&self) -> Result<()> {
        Ok(())
    }
    pub fn is_done(&self, _index: SubmissionIndex) -> bool {
        true
    }
    pub fn supports_bc(&self) -> bool {
        true
    }
    pub fn supports_timestamps(&self) -> bool {
        false
    }
    /// Whether the device is known lost (a call on it said so; Direct3D 12 tells it at any time).
    pub fn is_lost(&self) -> bool {
        false
    }

    pub fn bindless_textures_in_use(&self) -> usize {
        self.shared.textures.lock().in_use
    }
    pub fn bindless_buffers_in_use(&self) -> usize {
        self.shared.buffers.lock().in_use
    }

    pub fn create_buffer(&self, desc: &t::BufferDescriptor<'_>) -> Result<Buffer> {
        // (only what the CPU reads or writes keeps its bytes: a server's meshes would otherwise
        // all be held in memory a second time)
        let bytes = desc.usage.intersects(t::BufferUsages::MAP_READ | t::BufferUsages::MAP_WRITE).then(|| vec![0u8; desc.size as usize]);
        Ok(Buffer(Arc::new(BufferInner {
            size: desc.size,
            usage: desc.usage,
            bytes: Mutex::new(bytes),
        })))
    }

    pub fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Result<Buffer> {
        let b = self.create_buffer(&t::BufferDescriptor {
            label: desc.label,
            size: desc.contents.len() as u64,
            usage: desc.usage,
        })?;
        b.write(0, desc.contents);
        Ok(b)
    }

    pub fn create_texture(&self, desc: &t::TextureDescriptor<'_>) -> Result<Texture> {
        let max = self.shared.limits.max_texture_dimension_2d;
        if desc.size.width == 0 || desc.size.height == 0 || desc.size.width > max || desc.size.height > max {
            return Err(Error::Device(format!("a texture of {}x{}", desc.size.width, desc.size.height)));
        }
        Ok(Texture(Arc::new(TextureInner {
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            shared: self.shared.clone(),
        })))
    }

    /// # Safety
    /// Nothing: there is no raw object.
    pub unsafe fn texture_from_raw(&self, desc: &t::TextureDescriptor<'_>) -> Texture {
        Texture(Arc::new(TextureInner {
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            shared: self.shared.clone(),
        }))
    }

    pub fn create_sampler(&self, desc: &t::SamplerDescriptor<'_>) -> Result<Sampler> {
        let index = self.shared.samplers.lock().take("samplers")?;
        Ok(Sampler(Arc::new(SamplerInner {
            index,
            comparison: desc.compare.is_some(),
            shared: self.shared.clone(),
        })))
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup {
        let _ = desc;
        BindGroup(Arc::new(()))
    }

    pub fn create_indirect_draws(&self, desc: &t::IndirectDrawsDescriptor<'_>) -> Result<IndirectDraws> {
        Ok(IndirectDraws(desc.max_draws))
    }

    pub fn create_query_set(&self, desc: &t::QuerySetDescriptor<'_>) -> Result<QuerySet> {
        Ok(QuerySet(Arc::new((desc.count, desc.ty))))
    }

    pub fn create_blas(&self, _desc: &t::CreateBlasDescriptor<'_>, _sizes: &[t::BlasTriangleGeometrySizeDescriptor]) -> Result<Blas> {
        Err(Error::Unsupported("ray tracing".into()))
    }

    pub fn create_tlas(&self, _desc: &t::CreateTlasDescriptor<'_>) -> Result<Tlas> {
        Err(Error::Unsupported("ray tracing".into()))
    }

    pub fn create_shader_module(&self, desc: &ShaderModuleDescriptor<'_>) -> Result<ShaderModule> {
        let module = mx_shader::Module::from_wgsl(desc.label.unwrap_or("shader"), desc.source)?;
        Ok(ShaderModule(Arc::new(module)))
    }

    pub fn precompile(&self, _module: &ShaderModule) -> Result<()> {
        Ok(())
    }

    pub fn create_render_pipeline(&self, desc: &t::RenderPipelineDescriptor<'_, ShaderModule>) -> Result<RenderPipeline> {
        use mx_shader::naga::ShaderStage;
        let m = &desc.vertex.module.0;
        m.entry_point(ShaderStage::Vertex, desc.vertex.entry_point)?;
        m.specialize(ShaderStage::Vertex, entry_name(m, ShaderStage::Vertex, desc.vertex.entry_point)?, desc.vertex.constants)?;
        if let Some(f) = &desc.fragment {
            let m = &f.module.0;
            m.entry_point(ShaderStage::Fragment, f.entry_point)?;
            m.specialize(ShaderStage::Fragment, entry_name(m, ShaderStage::Fragment, f.entry_point)?, f.constants)?;
        }
        Ok(RenderPipeline(()))
    }

    pub fn create_compute_pipeline(&self, desc: &t::ComputePipelineDescriptor<'_, ShaderModule>) -> Result<ComputePipeline> {
        use mx_shader::naga::ShaderStage;
        let m = &desc.module.0;
        m.specialize(ShaderStage::Compute, entry_name(m, ShaderStage::Compute, desc.entry_point)?, desc.constants)?;
        Ok(ComputePipeline(()))
    }

    pub fn create_command_encoder(&self, _label: Option<&str>) -> CommandEncoder {
        CommandEncoder { work: Vec::new() }
    }

    pub fn create_render_bundle_encoder(&self, _desc: &t::RenderBundleEncoderDescriptor<'_>) -> RenderBundleEncoder {
        RenderBundleEncoder(())
    }
}

/// The name of the entry point a pipeline stage names (or the module's only one of the stage).
fn entry_name<'m>(m: &'m mx_shader::Module, stage: mx_shader::naga::ShaderStage, name: Option<&str>) -> Result<&'m str> {
    Ok(m.entry_point(stage, name)?.name.as_str())
}

#[derive(Clone)]
pub struct Queue {
    shared: Arc<Shared>,
}

impl Queue {
    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) -> SubmissionIndex {
        for b in buffers {
            for w in b.work {
                w.run();
            }
        }
        SubmissionIndex(self.shared.submitted.fetch_add(1, Ordering::AcqRel) + 1)
    }
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<()> {
        buffer.write(offset, data);
        Ok(())
    }
    pub fn write_texture(&self, _dst: t::TexelCopyTextureInfo<'_, Texture>, _data: &[u8], _layout: t::TexelCopyBufferLayout, _size: t::Extent3d) -> Result<()> {
        Ok(())
    }
    pub fn get_timestamp_period(&self) -> f32 {
        1.0
    }
}

// ---------------------------------------------------------------------------------------------
// Resources

struct BufferInner {
    size: u64,
    usage: t::BufferUsages,
    bytes: Mutex<Option<Vec<u8>>>,
}

#[derive(Clone)]
pub struct Buffer(Arc<BufferInner>);

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
    fn write(&self, offset: u64, data: &[u8]) {
        if let Some(b) = self.0.bytes.lock().as_mut() {
            let at = (offset as usize).min(b.len());
            let n = data.len().min(b.len() - at);
            b[at..at + n].copy_from_slice(&data[..n]);
        }
    }
    fn read_into(&self, offset: u64, out: &mut [u8]) {
        match self.0.bytes.lock().as_ref() {
            Some(b) => {
                let at = (offset as usize).min(b.len());
                let n = out.len().min(b.len() - at);
                out[..n].copy_from_slice(&b[at..at + n]);
                out[n..].fill(0);
            }
            None => out.fill(0),
        }
    }
    fn mapped<R>(&self, range: Range<u64>, what: t::BufferUsages, f: impl FnOnce(&mut [u8]) -> R) -> Result<R> {
        if !self.0.usage.contains(what) {
            return Err(Error::Device(format!("a buffer mapped without {what:?}")));
        }
        let mut g = self.0.bytes.lock();
        let b = g.as_mut().expect("a mappable buffer keeps its bytes");
        let r = (range.start as usize).min(b.len())..(range.end as usize).min(b.len());
        Ok(f(&mut b[r]))
    }
    pub fn map_read<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        self.mapped(range, t::BufferUsages::MAP_READ, |b| f(b))
    }
    pub fn map_read_now<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        self.map_read(range, f)
    }
    pub fn map_write<R>(&self, range: Range<u64>, f: impl FnOnce(&mut [u8]) -> R) -> Result<R> {
        self.mapped(range, t::BufferUsages::MAP_WRITE, f)
    }
    /// The buffer's bytes (its `Vec` is never reallocated, so the pointer stays valid).
    ///
    /// # Safety
    /// The buffer is mappable; the caller does not read or write it while a submission copies
    /// into it.
    pub unsafe fn mapped_ptr(&self) -> std::ptr::NonNull<u8> {
        let mut g = self.0.bytes.lock();
        let b = g.as_mut().expect("mX: mapped_ptr of a buffer made without MAP_READ or MAP_WRITE");
        std::ptr::NonNull::new(b.as_mut_ptr()).unwrap_or(std::ptr::NonNull::dangling())
    }
    pub fn create_view(&self, desc: &t::BufferViewDescriptor<'_>) -> Result<BufferView> {
        let size = desc.size.unwrap_or(self.0.size.saturating_sub(desc.offset));
        if desc.offset + size > self.0.size {
            return Err(Error::Device("a buffer view past the end of its buffer".into()));
        }
        // (no device here: the buffer's heap is found through a view's own slots)
        Ok(BufferView(Arc::new(BufferViewInner {
            buffer: self.clone(),
            offset: desc.offset,
            size,
            writable: desc.writable,
            index: BUFFER_INDEX.fetch_add(1, Ordering::Relaxed) as u32,
        })))
    }
}

/// Buffer views' indices (no device is at hand where they are made; never full).
static BUFFER_INDEX: AtomicU64 = AtomicU64::new(0);

struct BufferViewInner {
    buffer: Buffer,
    offset: u64,
    size: u64,
    writable: bool,
    index: u32,
}

#[derive(Clone)]
pub struct BufferView(Arc<BufferViewInner>);

impl BufferView {
    pub fn index(&self) -> u32 {
        self.0.index
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

struct TextureInner {
    size: t::Extent3d,
    format: t::TextureFormat,
    dimension: t::TextureDimension,
    mip_level_count: u32,
    sample_count: u32,
    usage: t::TextureUsages,
    shared: Arc<Shared>,
}

#[derive(Clone)]
pub struct Texture(Arc<TextureInner>);

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
    pub fn create_view(&self, desc: &t::TextureViewDescriptor<'_>) -> Result<TextureView> {
        let tex = &self.0;
        let layers = if tex.dimension == t::TextureDimension::D3 { 1 } else { tex.size.depth_or_array_layers };
        let dimension = desc.dimension.unwrap_or(match (tex.dimension, layers) {
            (t::TextureDimension::D1, _) => t::TextureViewDimension::D1,
            (t::TextureDimension::D3, _) => t::TextureViewDimension::D3,
            (_, 1) => t::TextureViewDimension::D2,
            _ => t::TextureViewDimension::D2Array,
        });
        let mips = desc.mip_level_count.unwrap_or(tex.mip_level_count.saturating_sub(desc.base_mip_level));
        let count = desc.array_layer_count.unwrap_or(layers.saturating_sub(desc.base_array_layer));
        let format = desc.format.unwrap_or(tex.format);
        let index = if tex.usage.contains(t::TextureUsages::TEXTURE_BINDING) || t::is_heap_storage_view(tex.usage, format, dimension, mips) {
            Some(tex.shared.textures.lock().take("texture views")?)
        } else {
            None
        };
        Ok(TextureView(Arc::new(ViewInner {
            texture: self.clone(),
            format: desc.format.unwrap_or(tex.format),
            dimension,
            mips: desc.base_mip_level..desc.base_mip_level + mips,
            layers: desc.base_array_layer..desc.base_array_layer + count,
            index,
        })))
    }
}

struct ViewInner {
    texture: Texture,
    format: t::TextureFormat,
    dimension: t::TextureViewDimension,
    mips: Range<u32>,
    layers: Range<u32>,
    index: Option<u32>,
}

impl Drop for ViewInner {
    fn drop(&mut self) {
        if let Some(i) = self.index {
            self.texture.0.shared.textures.lock().give(i);
        }
    }
}

#[derive(Clone)]
pub struct TextureView(Arc<ViewInner>);

impl TextureView {
    pub fn index(&self) -> u32 {
        self.0.index.expect("mX: the index of a texture view made without TEXTURE_BINDING")
    }
    pub fn try_index(&self) -> Option<u32> {
        self.0.index
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
        self.0.mips.clone()
    }
    pub fn array_layers(&self) -> Range<u32> {
        self.0.layers.clone()
    }
}

struct SamplerInner {
    index: u32,
    comparison: bool,
    shared: Arc<Shared>,
}

impl Drop for SamplerInner {
    fn drop(&mut self) {
        self.shared.samplers.lock().give(self.index);
    }
}

#[derive(Clone)]
pub struct Sampler(Arc<SamplerInner>);

impl Sampler {
    pub fn index(&self) -> u32 {
        self.0.index
    }
    pub fn is_comparison(&self) -> bool {
        self.0.comparison
    }
}

pub struct BufferBinding<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: Option<u64>,
}

pub enum BindingResource<'a> {
    Buffer(BufferBinding<'a>),
    TextureView(&'a TextureView),
    Sampler(&'a Sampler),
    AccelerationStructure(&'a Tlas),
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
pub struct BindGroup(Arc<()>);

#[derive(Clone)]
pub struct IndirectDraws(u32);

impl IndirectDraws {
    pub fn max_draws(&self) -> u32 {
        self.0
    }
}

#[derive(Clone)]
pub struct QuerySet(Arc<(u32, t::QueryType)>);

impl QuerySet {
    pub fn count(&self) -> u32 {
        self.0.0
    }
    pub fn ty(&self) -> t::QueryType {
        self.0.1
    }
}

#[derive(Clone)]
pub struct Blas(Arc<()>);

pub struct Tlas {
    instances: Vec<Option<t::TlasInstance<Blas>>>,
}

impl Tlas {
    pub fn max_instances(&self) -> u32 {
        self.instances.len() as u32
    }

    /// (never called: this backend makes no acceleration structures)
    pub fn index(&self) -> u32 {
        unreachable!("no acceleration structures on this backend")
    }
}

impl Index<usize> for Tlas {
    type Output = Option<t::TlasInstance<Blas>>;
    fn index(&self, i: usize) -> &Self::Output {
        &self.instances[i]
    }
}

impl IndexMut<usize> for Tlas {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.instances[i]
    }
}

identity!(Buffer, Texture, TextureView, Sampler, BindGroup, Blas);

// ---------------------------------------------------------------------------------------------
// Shaders and pipelines

pub struct ShaderModuleDescriptor<'a> {
    pub label: Option<&'a str>,
    pub source: &'a str,
}

#[derive(Clone)]
pub struct ShaderModule(Arc<mx_shader::Module>);

#[derive(Clone)]
pub struct RenderPipeline(());

#[derive(Clone)]
pub struct ComputePipeline(());

// ---------------------------------------------------------------------------------------------
// Commands

/// What a command buffer does to the bytes the CPU sees, at `submit`.
enum Work {
    Copy { src: Buffer, src_offset: u64, dst: Buffer, dst_offset: u64, size: u64 },
    Clear { buffer: Buffer, offset: u64, size: u64 },
}

impl Work {
    fn run(self) {
        match self {
            Work::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                let mut bytes = vec![0u8; size as usize];
                src.read_into(src_offset, &mut bytes);
                dst.write(dst_offset, &bytes);
            }
            Work::Clear { buffer, offset, size } => buffer.write(offset, &vec![0u8; size as usize]),
        }
    }
}

pub struct CommandBuffer {
    work: Vec<Work>,
}

pub struct CommandEncoder {
    work: Vec<Work>,
}

impl CommandEncoder {
    pub fn finish(self) -> CommandBuffer {
        CommandBuffer { work: self.work }
    }
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: Option<u64>) {
        let size = size.unwrap_or(buffer.size().saturating_sub(offset));
        self.work.push(Work::Clear { buffer: buffer.clone(), offset, size });
    }
    pub fn clear_texture(&mut self, _texture: &Texture, _range: &t::ImageSubresourceRange) {}
    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64) {
        self.work.push(Work::Copy {
            src: src.clone(),
            src_offset,
            dst: dst.clone(),
            dst_offset,
            size,
        });
    }
    pub fn copy_buffer_to_texture(&mut self, _src: t::TexelCopyBufferInfo<'_, Buffer>, _dst: t::TexelCopyTextureInfo<'_, Texture>, _size: t::Extent3d) {}
    pub fn copy_texture_to_buffer(&mut self, _src: t::TexelCopyTextureInfo<'_, Texture>, _dst: t::TexelCopyBufferInfo<'_, Buffer>, _size: t::Extent3d) {}
    pub fn copy_texture_to_texture(&mut self, _src: t::TexelCopyTextureInfo<'_, Texture>, _dst: t::TexelCopyTextureInfo<'_, Texture>, _size: t::Extent3d) {}
    pub fn generate_mipmaps(&mut self, _texture: &Texture) {}
    pub fn resolve_query_set(&mut self, set: &QuerySet, range: Range<u32>, dst: &Buffer, dst_offset: u64) {
        let _ = set;
        let size = (range.end - range.start) as u64 * t::QUERY_SIZE as u64;
        self.work.push(Work::Clear {
            buffer: dst.clone(),
            offset: dst_offset,
            size,
        });
    }
    pub fn write_timestamp(&mut self, _set: &QuerySet, _index: u32) {}
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indirect_draws(
        &mut self,
        _draws: &IndirectDraws,
        _pipeline: &RenderPipeline,
        _args: &Buffer,
        _args_offset: u64,
        _count: &Buffer,
        _count_offset: u64,
        _index: Option<(&Buffer, u64, t::IndexFormat)>,
    ) {
    }
    pub fn build_acceleration_structures<'a>(&mut self, _blas: impl IntoIterator<Item = &'a t::BlasBuildEntry<'a, Blas, Buffer>>, _tlas: impl IntoIterator<Item = &'a Tlas>) {}
    pub fn push_debug_group(&mut self, _label: &str) {}
    pub fn pop_debug_group(&mut self) {}
    pub fn insert_debug_marker(&mut self, _label: &str) {}
    pub fn begin_render_pass(&mut self, _desc: &t::RenderPassDescriptor<'_, TextureView, QuerySet>) -> RenderPass<'_> {
        RenderPass(std::marker::PhantomData)
    }
    pub fn begin_compute_pass(&mut self, _desc: &t::ComputePassDescriptor<'_, QuerySet>) -> ComputePass<'_> {
        ComputePass(std::marker::PhantomData)
    }
}

pub struct RenderPass<'e>(std::marker::PhantomData<&'e mut CommandEncoder>);

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, _pipeline: &RenderPipeline) {}
    pub fn set_bind_group(&mut self, _group: u32, _bind_group: &BindGroup) {}
    pub fn set_buffer_offset(&mut self, _group: u32, _binding: u32, _offset: u64) {}
    pub fn set_immediates(&mut self, _offset: u32, _data: &[u8]) {}
    pub fn set_vertex_buffer(&mut self, _slot: u32, _buffer: &Buffer, _offset: u64) {}
    pub fn set_index_buffer(&mut self, _buffer: &Buffer, _offset: u64, _format: t::IndexFormat) {}
    pub fn set_viewport(&mut self, _x: f32, _y: f32, _w: f32, _h: f32, _min_depth: f32, _max_depth: f32) {}
    pub fn set_scissor_rect(&mut self, _x: u32, _y: u32, _w: u32, _h: u32) {}
    pub fn set_stencil_reference(&mut self, _reference: u32) {}
    pub fn set_blend_constant(&mut self, _c: t::Color) {}
    pub fn draw(&mut self, _vertices: Range<u32>, _instances: Range<u32>) {}
    pub fn draw_indexed(&mut self, _indices: Range<u32>, _base_vertex: i32, _instances: Range<u32>) {}
    pub fn draw_indirect(&mut self, _args: &Buffer, _offset: u64) {}
    pub fn draw_indexed_indirect(&mut self, _args: &Buffer, _offset: u64) {}
    pub fn multi_draw_indirect(&mut self, _args: &Buffer, _offset: u64, _count: u32) {}
    pub fn multi_draw_indexed_indirect(&mut self, _args: &Buffer, _offset: u64, _count: u32) {}
    pub fn execute_indirect_draws(&mut self, _draws: &IndirectDraws) {}
    pub fn begin_occlusion_query(&mut self, _index: u32) {}
    pub fn end_occlusion_query(&mut self) {}
    pub fn execute_bundles<'b>(&mut self, bundles: impl IntoIterator<Item = &'b RenderBundle>) {
        bundles.into_iter().for_each(drop);
    }
    pub fn push_debug_group(&mut self, _label: &str) {}
    pub fn pop_debug_group(&mut self) {}
    pub fn insert_debug_marker(&mut self, _label: &str) {}
}

pub struct ComputePass<'e>(std::marker::PhantomData<&'e mut CommandEncoder>);

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, _pipeline: &ComputePipeline) {}
    pub fn set_bind_group(&mut self, _group: u32, _bind_group: &BindGroup) {}
    pub fn set_immediates(&mut self, _offset: u32, _data: &[u8]) {}
    pub fn dispatch_workgroups(&mut self, _x: u32, _y: u32, _z: u32) {}
    pub fn dispatch_workgroups_indirect(&mut self, _args: &Buffer, _offset: u64) {}
    pub fn push_debug_group(&mut self, _label: &str) {}
    pub fn pop_debug_group(&mut self) {}
    pub fn insert_debug_marker(&mut self, _label: &str) {}
}

#[derive(Clone)]
pub struct RenderBundle(());

pub struct RenderBundleEncoder(());

impl RenderBundleEncoder {
    pub fn finish(self, _label: Option<&str>) -> RenderBundle {
        RenderBundle(())
    }
    pub fn set_pipeline(&mut self, _p: &RenderPipeline) {}
    pub fn set_bind_group(&mut self, _group: u32, _g: &BindGroup) {}
    pub fn set_buffer_offset(&mut self, _group: u32, _binding: u32, _offset: u64) {}
    pub fn set_immediates(&mut self, _offset: u32, _data: &[u8]) {}
    pub fn set_vertex_buffer(&mut self, _slot: u32, _b: &Buffer, _offset: u64) {}
    pub fn set_index_buffer(&mut self, _b: &Buffer, _offset: u64, _format: t::IndexFormat) {}
    pub fn draw(&mut self, _vertices: Range<u32>, _instances: Range<u32>) {}
    pub fn draw_indexed(&mut self, _indices: Range<u32>, _base_vertex: i32, _instances: Range<u32>) {}
    pub fn draw_indirect(&mut self, _args: &Buffer, _offset: u64) {}
    pub fn draw_indexed_indirect(&mut self, _args: &Buffer, _offset: u64) {}
}

// ---------------------------------------------------------------------------------------------
// Surfaces

/// A window's surface: each frame is a fresh texture of the configured size that nothing shows.
pub struct Surface<'w> {
    state: Mutex<Option<(Device, t::SurfaceConfiguration)>>,
    _lifetime: std::marker::PhantomData<&'w ()>,
}

impl Surface<'_> {
    pub fn get_capabilities(&self, _adapter: &Adapter) -> t::SurfaceCapabilities {
        t::SurfaceCapabilities {
            formats: vec![
                t::TextureFormat::Bgra8UnormSrgb,
                t::TextureFormat::Bgra8Unorm,
                t::TextureFormat::Rgba8UnormSrgb,
                t::TextureFormat::Rgba8Unorm,
            ],
            present_modes: vec![t::PresentMode::Fifo, t::PresentMode::Immediate],
            alpha_modes: vec![t::CompositeAlphaMode::Opaque],
            usages: t::TextureUsages::RENDER_ATTACHMENT | t::TextureUsages::COPY_SRC | t::TextureUsages::COPY_DST,
        }
    }

    pub fn configure(&self, device: &Device, config: &t::SurfaceConfiguration) {
        *self.state.lock() = Some((device.clone(), config.clone()));
    }

    pub fn get_current_texture(&self) -> t::CurrentSurfaceTexture<SurfaceTexture> {
        let g = self.state.lock();
        let (device, c) = g.as_ref().expect("get_current_texture on a surface that was never configured");
        let made = device.create_texture(&t::TextureDescriptor {
            label: Some("surface"),
            size: t::Extent3d {
                width: c.width.max(1),
                height: c.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: t::TextureDimension::D2,
            format: c.format,
            usage: c.usage,
            view_formats: &[],
        });
        match made {
            Ok(texture) => t::CurrentSurfaceTexture::Success(SurfaceTexture { texture }),
            Err(_) => t::CurrentSurfaceTexture::Outdated,
        }
    }
}

/// A frame of a surface. Draw into `texture`, submit, then `present`.
pub struct SurfaceTexture {
    pub texture: Texture,
}

impl SurfaceTexture {
    pub fn present(self) {}
}

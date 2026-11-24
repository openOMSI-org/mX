//! mX's objects over the backends (see the crate's documentation, Backends).

use std::ops::{Index, IndexMut, Range};
use std::sync::Arc;

use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::{Error, Pick, Result, conv, mismatch, t};

def_enum!(
    #[derive(Clone)]
    Adapter
);
def_enum!(
    #[derive(Clone, PartialEq, Eq, Hash)]
    Device
);
def_enum!(
    #[derive(Clone)]
    Queue
);
def_enum!(
    #[derive(Clone, PartialEq, Eq, Hash)]
    Buffer
);
def_enum!(
    #[derive(Clone)]
    BufferView
);
def_enum!(
    #[derive(Clone, PartialEq, Eq, Hash)]
    Texture
);
def_enum!(
    #[derive(Clone, PartialEq, Eq, Hash)]
    TextureView
);
def_enum!(
    #[derive(Clone, PartialEq, Eq, Hash)]
    Sampler
);
def_enum!(
    #[derive(Clone)]
    QuerySet
);
def_enum!(
    #[derive(Clone, PartialEq, Eq, Hash)]
    BindGroup
);
def_enum!(
    #[derive(Clone)]
    ShaderModule
);
def_enum!(
    #[derive(Clone)]
    RenderPipeline
);
def_enum!(
    #[derive(Clone)]
    ComputePipeline
);
def_enum!(
    #[derive(Clone)]
    RenderBundle
);
def_enum!(
    #[derive(Clone)]
    IndirectDraws
);
def_enum!(
    #[derive(Clone, PartialEq, Eq, Hash)]
    Blas
);
def_enum!(CommandEncoder);
def_enum!(CommandBuffer);
def_enum!(RenderBundleEncoder);

/// A position in the queue's order of submissions (the first is 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionIndex(pub u64);

// ---------------------------------------------------------------------------------------------
// Instance and adapters

/// The backends' entry points that loaded.
pub(crate) struct Opened {
    #[cfg(target_vendor = "apple")]
    metal: Option<mx_metal::Instance>,
    #[cfg(windows)]
    dx12: Option<mx_dx12::Instance>,
    #[cfg(feature = "vulkan")]
    vulkan: Option<mx_vulkan::Instance>,
    #[cfg(feature = "gl")]
    gl: Option<mx_gl::Instance>,
    #[cfg(windows)]
    dx11: Option<mx_dx11::Instance>,
    #[cfg(feature = "noop")]
    noop: Option<mx_noop::Instance>,
}

/// The entry point: the backends of this build that the program asked for.
#[derive(Clone)]
pub struct Instance {
    pub(crate) backends: Arc<Opened>,
}

impl Default for Instance {
    fn default() -> Self {
        Self::new()
    }
}

impl Instance {
    /// Every backend of this build, or those `MX_BACKENDS` names.
    pub fn new() -> Instance {
        Self::with_backends(t::Backends::from_env().unwrap_or(t::Backends::ALL))
    }

    /// The backends in `which` that this build has (and whose libraries load).
    pub fn with_backends(which: t::Backends) -> Instance {
        let _ = which;
        Instance {
            backends: Arc::new(Opened {
                #[cfg(target_vendor = "apple")]
                metal: which.contains(t::Backends::METAL).then(mx_metal::Instance::new),
                #[cfg(windows)]
                dx12: which.contains(t::Backends::DX12).then(mx_dx12::Instance::new),
                #[cfg(feature = "vulkan")]
                vulkan: which.contains(t::Backends::VULKAN).then(mx_vulkan::Instance::new),
                #[cfg(feature = "gl")]
                gl: which.contains(t::Backends::GL).then(mx_gl::Instance::new),
                #[cfg(windows)]
                dx11: which.contains(t::Backends::DX11).then(mx_dx11::Instance::new),
                #[cfg(feature = "noop")]
                noop: which.contains(t::Backends::NOOP).then(mx_noop::Instance::new),
            }),
        }
    }

    /// The backends this build has: what a graphics API setting can offer.
    pub fn compiled_backends() -> t::Backends {
        let mut b = t::Backends::empty();
        if cfg!(target_vendor = "apple") {
            b |= t::Backends::METAL;
        }
        if cfg!(windows) {
            b |= t::Backends::DX12;
        }
        if cfg!(feature = "vulkan") {
            b |= t::Backends::VULKAN;
        }
        if cfg!(feature = "gl") {
            b |= t::Backends::GL;
        }
        if cfg!(windows) {
            b |= t::Backends::DX11;
        }
        if cfg!(feature = "noop") {
            b |= t::Backends::NOOP;
        }
        b
    }

    /// Every adapter of every backend opened: Metal, Direct3D 12, Vulkan, then OpenGL, each
    /// backend's in its own order (discrete GPUs first).
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let mut v: Vec<Adapter> = Vec::new();
        let b = &self.backends;
        let _ = b;
        #[cfg(target_vendor = "apple")]
        if let Some(i) = &b.metal {
            v.extend(i.enumerate_adapters().into_iter().map(Adapter::Metal));
        }
        #[cfg(windows)]
        if let Some(i) = &b.dx12 {
            v.extend(i.enumerate_adapters().into_iter().map(Adapter::Dx12));
        }
        #[cfg(feature = "vulkan")]
        if let Some(i) = &b.vulkan {
            v.extend(i.enumerate_adapters().into_iter().map(Adapter::Vulkan));
        }
        #[cfg(feature = "gl")]
        if let Some(i) = &b.gl {
            v.extend(i.enumerate_adapters().into_iter().map(Adapter::Gl));
        }
        #[cfg(windows)]
        if let Some(i) = &b.dx11 {
            v.extend(i.enumerate_adapters().into_iter().map(Adapter::Dx11));
        }
        #[cfg(feature = "noop")]
        if let Some(i) = &b.noop {
            v.extend(i.enumerate_adapters().into_iter().map(Adapter::Noop));
        }
        v
    }

    /// The first adapter of the first backend that has one (see `enumerate_adapters`).
    pub fn default_adapter(&self) -> Result<Adapter> {
        self.enumerate_adapters().into_iter().next().ok_or(Error::NoDevice)
    }

    /// The first adapter of `backend`.
    pub fn adapter_for(&self, backend: t::Backend) -> Result<Adapter> {
        self.enumerate_adapters().into_iter().find(|a| a.backend() == backend).ok_or(Error::NoDevice)
    }

    /// A surface for a window (a winit `Window`, `Arc<Window>` or `&Window`). It takes on a
    /// backend when it is configured with a device (on Apple platforms, call on the main
    /// thread).
    pub fn create_surface<'w>(&self, window: impl HasWindowHandle + HasDisplayHandle + Send + Sync + 'w) -> Result<Surface<'w>> {
        let raw_window = window.window_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        let raw_display = window.display_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        Ok(Surface {
            instance: self.clone(),
            display: raw_display,
            window: raw_window,
            inner: Mutex::new(None),
            _window: Some(Box::new(window)),
        })
    }

    /// # Safety
    /// The handles are valid for as long as the surface is used.
    pub unsafe fn create_surface_unsafe(&self, display: RawDisplayHandle, window: RawWindowHandle) -> Result<Surface<'static>> {
        Ok(Surface {
            instance: self.clone(),
            display,
            window,
            inner: Mutex::new(None),
            _window: None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DeviceDescriptor<'a> {
    pub label: Option<&'a str>,
    /// Texture views the heap holds at once.
    pub bindless_textures: u32,
    /// Samplers the heap holds at once.
    pub bindless_samplers: u32,
    /// Buffer views the heap holds at once.
    pub bindless_buffers: u32,
    /// Vulkan only: device extensions to turn on besides mX's own (DLSS's). Ignored elsewhere.
    pub vulkan_extensions: &'a [&'static std::ffi::CStr],
    /// Limits the device keeps to, where lower than the adapter's (a weaker chip's, to
    /// test on); None: the adapter's.
    pub limits: Option<t::Limits>,
}

impl Default for DeviceDescriptor<'_> {
    fn default() -> Self {
        DeviceDescriptor {
            limits: None,
            label: None,
            bindless_textures: 1 << 16,
            bindless_samplers: 2048,
            bindless_buffers: 1 << 15,
            vulkan_extensions: &[],
        }
    }
}

macro_rules! device_desc {
    ($k:ident, $d:expr) => {
        $k::DeviceDescriptor {
            label: $d.label,
            bindless_textures: $d.bindless_textures,
            bindless_samplers: $d.bindless_samplers,
            bindless_buffers: $d.bindless_buffers,
            limits: $d.limits,
        }
    };
}

#[cfg(feature = "vulkan")]
macro_rules! vulkan_device_desc {
    ($d:expr) => {
        mx_vulkan::DeviceDescriptor {
            label: $d.label,
            bindless_textures: $d.bindless_textures,
            bindless_samplers: $d.bindless_samplers,
            bindless_buffers: $d.bindless_buffers,
            limits: $d.limits,
            extensions: $d.vulkan_extensions,
        }
    };
}

impl Adapter {
    pub fn info(&self) -> t::AdapterInfo {
        on!(Adapter, self, a => a.info())
    }
    pub fn limits(&self) -> t::Limits {
        on!(Adapter, self, a => a.limits())
    }
    pub fn supports_bc(&self) -> bool {
        on!(Adapter, self, a => a.supports_bc())
    }
    pub fn supports_ray_query(&self) -> bool {
        on!(Adapter, self, a => a.supports_ray_query())
    }
    pub fn supports_timestamps(&self) -> bool {
        on!(Adapter, self, a => a.supports_timestamps())
    }
    /// Whether the adapter runs on mX's bound tier (OpenGL, Metal 2): heap accesses must be
    /// indexed by immediates or constants; no GPU-driven batches or ray queries.
    pub fn is_bound_tier(&self) -> bool {
        on!(Adapter, self, a => a.is_bound_tier())
    }
    pub fn supports_indirect_count(&self) -> bool {
        on!(Adapter, self, a => a.supports_indirect_count())
    }
    pub fn supports_compute(&self) -> bool {
        on!(Adapter, self, a => a.supports_compute())
    }
    pub fn storage_buffers_per_stage(&self) -> (u32, u32) {
        on!(Adapter, self, a => a.storage_buffers_per_stage())
    }
    /// Indexed draws with a base vertex (not below OpenGL 3.2 / ES 3.2).
    pub fn supports_base_vertex(&self) -> bool {
        on!(Adapter, self, a => a.supports_base_vertex())
    }
    /// Textures a pipeline may read through root bindings: (vertex, fragment, together).
    pub fn texture_units(&self) -> (u32, u32, u32) {
        on!(Adapter, self, a => a.texture_units())
    }
    /// The GPU's memory, where the system tells (not on OpenGL).
    pub fn memory_info(&self) -> Option<t::AdapterMemory> {
        on!(Adapter, self, a => a.memory_info())
    }
    /// Whether `format` renders and resolves with `samples` samples per pixel.
    pub fn supports_msaa(&self, format: t::TextureFormat, samples: u32) -> bool {
        on!(Adapter, self, a => a.supports_msaa(format, samples))
    }

    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Result<(Device, Queue)> {
        Ok(match self {
            #[cfg(target_vendor = "apple")]
            Adapter::Metal(a) => {
                let (d, q) = a.request_device(&device_desc!(mx_metal, desc))?;
                (Device::Metal(d), Queue::Metal(q))
            }
            #[cfg(windows)]
            Adapter::Dx12(a) => {
                let (d, q) = a.request_device(&device_desc!(mx_dx12, desc))?;
                (Device::Dx12(d), Queue::Dx12(q))
            }
            #[cfg(feature = "vulkan")]
            Adapter::Vulkan(a) => {
                let (d, q) = a.request_device(&vulkan_device_desc!(desc))?;
                (Device::Vulkan(d), Queue::Vulkan(q))
            }
            #[cfg(feature = "gl")]
            Adapter::Gl(a) => {
                let (d, q) = a.request_device(&device_desc!(mx_gl, desc))?;
                (Device::Gl(d), Queue::Gl(q))
            }
            #[cfg(windows)]
            Adapter::Dx11(a) => {
                let (d, q) = a.request_device(&device_desc!(mx_dx11, desc))?;
                (Device::Dx11(d), Queue::Dx11(q))
            }
            #[cfg(feature = "noop")]
            Adapter::Noop(a) => {
                let (d, q) = a.request_device(&device_desc!(mx_noop, desc))?;
                (Device::Noop(d), Queue::Noop(q))
            }
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Device and queue

pub struct BufferInitDescriptor<'a> {
    pub label: Option<&'a str>,
    pub contents: &'a [u8],
    pub usage: t::BufferUsages,
}

pub struct ShaderModuleDescriptor<'a> {
    pub label: Option<&'a str>,
    /// WGSL. The heap prelude (`mx_tex2d`, `mx_samplers`, ...) is added by mX.
    pub source: &'a str,
}

#[derive(Clone, Copy)]
pub struct BufferBinding<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: Option<u64>,
}

#[derive(Clone, Copy)]
pub enum BindingResource<'a> {
    Buffer(BufferBinding<'a>),
    TextureView(&'a TextureView),
    Sampler(&'a Sampler),
    AccelerationStructure(&'a Tlas),
}

#[derive(Clone, Copy)]
pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

#[derive(Clone, Copy)]
pub struct BindGroupDescriptor<'a> {
    pub label: Option<&'a str>,
    pub entries: &'a [BindGroupEntry<'a>],
}

/// A bind group for backend `$k` from `desc`, made by `$d` (the backend's device). Tlases are
/// locked for the call (see `Tlas`).
macro_rules! bind_group {
    ($k:ident, $v:ident, $d:expr, $desc:expr) => {{
        let guards: Vec<_> = $desc
            .entries
            .iter()
            .filter_map(|e| match &e.resource {
                BindingResource::AccelerationStructure(t) => Some(t.lock_for(t::Backend::$v)),
                _ => None,
            })
            .collect();
        let mut tlases = guards.iter().map(|g| match &**g {
            TlasRaw::$v(t) => t,
            #[allow(unreachable_patterns)]
            _ => mismatch(),
        });
        let entries: Vec<$k::BindGroupEntry<'_>> = $desc
            .entries
            .iter()
            .map(|e| $k::BindGroupEntry {
                binding: e.binding,
                resource: match &e.resource {
                    BindingResource::Buffer(b) => $k::BindingResource::Buffer($k::BufferBinding {
                        buffer: b.buffer.pick(),
                        offset: b.offset,
                        size: b.size,
                    }),
                    BindingResource::TextureView(v) => $k::BindingResource::TextureView(v.pick()),
                    BindingResource::Sampler(s) => $k::BindingResource::Sampler(s.pick()),
                    BindingResource::AccelerationStructure(_) => $k::BindingResource::AccelerationStructure(tlases.next().expect("a tlas per entry")),
                },
            })
            .collect();
        $d.create_bind_group(&$k::BindGroupDescriptor {
            label: $desc.label,
            entries: &entries,
        })
    }};
}

impl Device {
    pub fn info(&self) -> &t::AdapterInfo {
        on!(Device, self, d => d.info())
    }
    pub fn limits(&self) -> t::Limits {
        on!(Device, self, d => d.limits())
    }
    pub fn poll(&self) -> Result<()> {
        on!(Device, self, d => d.poll().map_err(Error::from))
    }
    pub fn wait(&self, index: SubmissionIndex) -> Result<()> {
        on!(Device, self, d => d.wait(si(index)).map_err(Error::from))
    }
    pub fn wait_idle(&self) -> Result<()> {
        on!(Device, self, d => d.wait_idle().map_err(Error::from))
    }
    pub fn is_done(&self, index: SubmissionIndex) -> bool {
        on!(Device, self, d => d.is_done(si(index)))
    }
    /// Whether the device is lost for good (Direct3D 12's removed device; elsewhere false):
    /// a renderer that failed on it is not built again on it.
    pub fn is_lost(&self) -> bool {
        on!(Device, self, d => d.is_lost())
    }
    pub fn bindless_textures_in_use(&self) -> usize {
        on!(Device, self, d => d.bindless_textures_in_use())
    }
    /// BC texture compression.
    pub fn supports_bc(&self) -> bool {
        on!(Device, self, d => d.supports_bc())
    }
    /// Timestamp queries.
    pub fn supports_timestamps(&self) -> bool {
        on!(Device, self, d => d.supports_timestamps())
    }
    pub fn bindless_buffers_in_use(&self) -> usize {
        on!(Device, self, d => d.bindless_buffers_in_use())
    }
    pub fn create_buffer(&self, desc: &t::BufferDescriptor<'_>) -> Result<Buffer> {
        // (over the device's limit is no shortage of memory: refused as unsupported, on every
        // backend alike)
        let max = self.limits().max_buffer_size;
        if desc.size > max {
            return Err(Error::Unsupported(format!("a buffer of {} bytes (the device's limit is {max})", desc.size)));
        }
        wrap_res!(Device, Buffer, self, d => d.create_buffer(desc))
    }
    pub fn create_texture(&self, desc: &t::TextureDescriptor<'_>) -> Result<Texture> {
        let l = self.limits();
        let side = if desc.dimension == t::TextureDimension::D3 {
            l.max_texture_dimension_3d
        } else {
            l.max_texture_dimension_2d
        };
        if desc.size.width > side || desc.size.height > side {
            return Err(Error::Unsupported(format!("a texture of {}x{} (the device's limit is {side})", desc.size.width, desc.size.height)));
        }
        wrap_res!(Device, Texture, self, d => d.create_texture(desc))
    }
    pub fn create_sampler(&self, desc: &t::SamplerDescriptor<'_>) -> Result<Sampler> {
        wrap_res!(Device, Sampler, self, d => d.create_sampler(desc))
    }
    pub fn create_query_set(&self, desc: &t::QuerySetDescriptor<'_>) -> Result<QuerySet> {
        wrap_res!(Device, QuerySet, self, d => d.create_query_set(desc))
    }
    pub fn create_indirect_draws(&self, desc: &t::IndirectDrawsDescriptor<'_>) -> Result<IndirectDraws> {
        wrap_res!(Device, IndirectDraws, self, d => d.create_indirect_draws(desc))
    }
    pub fn create_blas(&self, desc: &t::CreateBlasDescriptor<'_>, sizes: &[t::BlasTriangleGeometrySizeDescriptor]) -> Result<Blas> {
        wrap_res!(Device, Blas, self, d => d.create_blas(desc, sizes))
    }
    pub fn create_tlas(&self, desc: &t::CreateTlasDescriptor<'_>) -> Result<Tlas> {
        let raw = wrap_res!(Device, TlasRaw, self, d => d.create_tlas(desc))?;
        let index = match &raw {
            #[cfg(target_vendor = "apple")]
            TlasRaw::Metal(t) => t.index(),
            #[cfg(windows)]
            TlasRaw::Dx12(t) => t.index(),
            #[cfg(feature = "vulkan")]
            TlasRaw::Vulkan(t) => t.index(),
            #[cfg(feature = "gl")]
            TlasRaw::Gl(t) => t.index(),
            #[cfg(windows)]
            TlasRaw::Dx11(t) => t.index(),
            #[cfg(feature = "noop")]
            TlasRaw::Noop(t) => t.index(),
        };
        Ok(Tlas {
            raw: Mutex::new(raw),
            index,
            instances: vec![None; desc.max_instances as usize],
        })
    }
    pub fn create_command_encoder(&self, label: Option<&str>) -> CommandEncoder {
        wrap!(Device, CommandEncoder, self, d => d.create_command_encoder(label))
    }
    pub fn create_render_bundle_encoder(&self, desc: &t::RenderBundleEncoderDescriptor<'_>) -> RenderBundleEncoder {
        wrap!(Device, RenderBundleEncoder, self, d => d.create_render_bundle_encoder(desc))
    }
    pub fn create_render_pipeline(&self, desc: &crate::RenderPipelineDescriptor<'_>) -> Result<RenderPipeline> {
        wrap_res!(Device, RenderPipeline, self, d => d.create_render_pipeline(&conv::render_pipeline(desc)))
    }
    pub fn create_compute_pipeline(&self, desc: &crate::ComputePipelineDescriptor<'_>) -> Result<ComputePipeline> {
        wrap_res!(Device, ComputePipeline, self, d => d.create_compute_pipeline(&conv::compute_pipeline(desc)))
    }
    pub fn precompile(&self, module: &ShaderModule) -> Result<()> {
        on!(Device, self, d => d.precompile(module.pick()).map_err(Error::from))
    }

    pub fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Result<Buffer> {
        macro_rules! go {
            ($k:ident, $d:expr) => {
                $d.create_buffer_init(&$k::BufferInitDescriptor {
                    label: desc.label,
                    contents: desc.contents,
                    usage: desc.usage,
                })
            };
        }
        Ok(match self {
            #[cfg(target_vendor = "apple")]
            Device::Metal(d) => Buffer::Metal(go!(mx_metal, d)?),
            #[cfg(windows)]
            Device::Dx12(d) => Buffer::Dx12(go!(mx_dx12, d)?),
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => Buffer::Vulkan(go!(mx_vulkan, d)?),
            #[cfg(feature = "gl")]
            Device::Gl(d) => Buffer::Gl(go!(mx_gl, d)?),
            #[cfg(windows)]
            Device::Dx11(d) => Buffer::Dx11(go!(mx_dx11, d)?),
            #[cfg(feature = "noop")]
            Device::Noop(d) => Buffer::Noop(go!(mx_noop, d)?),
        })
    }

    pub fn create_shader_module(&self, desc: &ShaderModuleDescriptor<'_>) -> Result<ShaderModule> {
        macro_rules! go {
            ($k:ident, $d:expr) => {
                $d.create_shader_module(&$k::ShaderModuleDescriptor {
                    label: desc.label,
                    source: desc.source,
                })
            };
        }
        Ok(match self {
            #[cfg(target_vendor = "apple")]
            Device::Metal(d) => ShaderModule::Metal(go!(mx_metal, d)?),
            #[cfg(windows)]
            Device::Dx12(d) => ShaderModule::Dx12(go!(mx_dx12, d)?),
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => ShaderModule::Vulkan(go!(mx_vulkan, d)?),
            #[cfg(feature = "gl")]
            Device::Gl(d) => ShaderModule::Gl(go!(mx_gl, d)?),
            #[cfg(windows)]
            Device::Dx11(d) => ShaderModule::Dx11(go!(mx_dx11, d)?),
            #[cfg(feature = "noop")]
            Device::Noop(d) => ShaderModule::Noop(go!(mx_noop, d)?),
        })
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup {
        match self {
            #[cfg(target_vendor = "apple")]
            Device::Metal(d) => BindGroup::Metal(bind_group!(mx_metal, Metal, d, desc)),
            #[cfg(windows)]
            Device::Dx12(d) => BindGroup::Dx12(bind_group!(mx_dx12, Dx12, d, desc)),
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => BindGroup::Vulkan(bind_group!(mx_vulkan, Vulkan, d, desc)),
            #[cfg(feature = "gl")]
            Device::Gl(d) => BindGroup::Gl(bind_group!(mx_gl, Gl, d, desc)),
            #[cfg(windows)]
            Device::Dx11(d) => BindGroup::Dx11(bind_group!(mx_dx11, Dx11, d, desc)),
            #[cfg(feature = "noop")]
            Device::Noop(d) => BindGroup::Noop(bind_group!(mx_noop, Noop, d, desc)),
        }
    }
}

/// Submission indices pass between mX and the backends as their `u64`.
trait ConvertIndex<T> {
    fn convert(i: SubmissionIndex) -> T;
}

/// The backend's submission index for `i` (the backend inferred from the call).
fn si<T>(i: SubmissionIndex) -> T
where
    SubmissionIndex: ConvertIndex<T>,
{
    <SubmissionIndex as ConvertIndex<T>>::convert(i)
}

macro_rules! index_conv {
    ($($(#[$c:meta])* $k:ident;)*) => {$(
        $(#[$c])*
        impl ConvertIndex<$k::SubmissionIndex> for SubmissionIndex {
            fn convert(i: SubmissionIndex) -> $k::SubmissionIndex {
                $k::SubmissionIndex(i.0)
            }
        }
        $(#[$c])*
        impl From<$k::SubmissionIndex> for SubmissionIndex {
            fn from(i: $k::SubmissionIndex) -> SubmissionIndex {
                SubmissionIndex(i.0)
            }
        }
    )*};
}

index_conv! {
    #[cfg(target_vendor = "apple")] mx_metal;
    #[cfg(windows)] mx_dx12;
    #[cfg(feature = "vulkan")] mx_vulkan;
    #[cfg(feature = "gl")] mx_gl;
    #[cfg(windows)] mx_dx11;
    #[cfg(feature = "noop")] mx_noop;
}

impl Queue {
    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) -> SubmissionIndex {
        let buffers = buffers.into_iter();
        match self {
            #[cfg(target_vendor = "apple")]
            Queue::Metal(q) => q
                .submit(buffers.map(|b| match b {
                    CommandBuffer::Metal(b) => b,
                    #[allow(unreachable_patterns)]
                    _ => mismatch(),
                }))
                .into(),
            #[cfg(windows)]
            Queue::Dx12(q) => q
                .submit(buffers.map(|b| match b {
                    CommandBuffer::Dx12(b) => b,
                    #[allow(unreachable_patterns)]
                    _ => mismatch(),
                }))
                .into(),
            #[cfg(feature = "vulkan")]
            Queue::Vulkan(q) => q
                .submit(buffers.map(|b| match b {
                    CommandBuffer::Vulkan(b) => b,
                    #[allow(unreachable_patterns)]
                    _ => mismatch(),
                }))
                .into(),
            #[cfg(feature = "gl")]
            Queue::Gl(q) => q
                .submit(buffers.map(|b| match b {
                    CommandBuffer::Gl(b) => b,
                    #[allow(unreachable_patterns)]
                    _ => mismatch(),
                }))
                .into(),
            #[cfg(windows)]
            Queue::Dx11(q) => q
                .submit(buffers.map(|b| match b {
                    CommandBuffer::Dx11(b) => b,
                    #[allow(unreachable_patterns)]
                    _ => mismatch(),
                }))
                .into(),
            #[cfg(feature = "noop")]
            Queue::Noop(q) => q
                .submit(buffers.map(|b| match b {
                    CommandBuffer::Noop(b) => b,
                    #[allow(unreachable_patterns)]
                    _ => mismatch(),
                }))
                .into(),
        }
    }
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<()> {
        on!(Queue, self, q => q.write_buffer(buffer.pick(), offset, data).map_err(Error::from))
    }
    pub fn write_texture(&self, dst: crate::TexelCopyTextureInfo<'_>, data: &[u8], layout: t::TexelCopyBufferLayout, size: t::Extent3d) -> Result<()> {
        on!(Queue, self, q => q.write_texture(conv::texture_info(&dst), data, layout, size).map_err(Error::from))
    }
    pub fn get_timestamp_period(&self) -> f32 {
        on!(Queue, self, q => q.get_timestamp_period())
    }
}

// ---------------------------------------------------------------------------------------------
// Resources

impl Buffer {
    pub fn size(&self) -> u64 {
        on!(Buffer, self, b => b.size())
    }
    pub fn usage(&self) -> t::BufferUsages {
        on!(Buffer, self, b => b.usage())
    }
    pub fn map_read<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        on!(Buffer, self, b => b.map_read(range, f).map_err(Error::from))
    }
    /// Read without waiting: the caller knows the submissions that wrote the buffer are done
    /// (`Device::is_done`), as after wgpu's `map_async`. Needs `MAP_READ`.
    pub fn map_read_now<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        on!(Buffer, self, b => b.map_read_now(range, f).map_err(Error::from))
    }
    pub fn map_write<R>(&self, range: Range<u64>, f: impl FnOnce(&mut [u8]) -> R) -> Result<R> {
        on!(Buffer, self, b => b.map_write(range, f).map_err(Error::from))
    }
    /// # Safety
    /// As the backend's: the caller keeps off the GPU's work (not on OpenGL).
    pub unsafe fn mapped_ptr(&self) -> std::ptr::NonNull<u8> {
        // SAFETY: the caller's.
        on!(Buffer, self, b => unsafe { b.mapped_ptr() })
    }
    pub fn create_view(&self, desc: &t::BufferViewDescriptor<'_>) -> Result<BufferView> {
        wrap_res!(Buffer, BufferView, self, b => b.create_view(desc))
    }
    /// The whole buffer as a root binding.
    pub fn as_entire_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding { buffer: self, offset: 0, size: None })
    }
}

impl BufferView {
    pub fn index(&self) -> u32 {
        on!(BufferView, self, v => v.index())
    }
    pub fn buffer(&self) -> Buffer {
        wrap!(BufferView, Buffer, self, v => v.buffer().clone())
    }
    pub fn offset(&self) -> u64 {
        on!(BufferView, self, v => v.offset())
    }
    pub fn size(&self) -> u64 {
        on!(BufferView, self, v => v.size())
    }
    pub fn is_writable(&self) -> bool {
        on!(BufferView, self, v => v.is_writable())
    }
}

impl Texture {
    pub fn size(&self) -> t::Extent3d {
        on!(Texture, self, x => x.size())
    }
    pub fn width(&self) -> u32 {
        on!(Texture, self, x => x.width())
    }
    pub fn height(&self) -> u32 {
        on!(Texture, self, x => x.height())
    }
    pub fn format(&self) -> t::TextureFormat {
        on!(Texture, self, x => x.format())
    }
    pub fn dimension(&self) -> t::TextureDimension {
        on!(Texture, self, x => x.dimension())
    }
    pub fn mip_level_count(&self) -> u32 {
        on!(Texture, self, x => x.mip_level_count())
    }
    pub fn sample_count(&self) -> u32 {
        on!(Texture, self, x => x.sample_count())
    }
    pub fn usage(&self) -> t::TextureUsages {
        on!(Texture, self, x => x.usage())
    }
    pub fn create_view(&self, desc: &t::TextureViewDescriptor<'_>) -> Result<TextureView> {
        wrap_res!(Texture, TextureView, self, x => x.create_view(desc))
    }
    /// Mip 0 of the whole texture, at the origin, for copies.
    pub fn as_image_copy(&self) -> crate::TexelCopyTextureInfo<'_> {
        t::TexelCopyTextureInfo {
            texture: self,
            mip_level: 0,
            origin: t::Origin3d::ZERO,
            aspect: t::TextureAspect::All,
        }
    }
}

impl TextureView {
    pub fn index(&self) -> u32 {
        on!(TextureView, self, v => v.index())
    }
    pub fn try_index(&self) -> Option<u32> {
        on!(TextureView, self, v => v.try_index())
    }
    pub fn texture(&self) -> Texture {
        wrap!(TextureView, Texture, self, v => v.texture().clone())
    }
    pub fn format(&self) -> t::TextureFormat {
        on!(TextureView, self, v => v.format())
    }
    pub fn dimension(&self) -> t::TextureViewDimension {
        on!(TextureView, self, v => v.dimension())
    }
    pub fn mip_levels(&self) -> Range<u32> {
        on!(TextureView, self, v => v.mip_levels())
    }
    pub fn array_layers(&self) -> Range<u32> {
        on!(TextureView, self, v => v.array_layers())
    }
}

impl Sampler {
    pub fn index(&self) -> u32 {
        on!(Sampler, self, s => s.index())
    }
    pub fn is_comparison(&self) -> bool {
        on!(Sampler, self, s => s.is_comparison())
    }
}

impl QuerySet {
    pub fn count(&self) -> u32 {
        on!(QuerySet, self, q => q.count())
    }
    pub fn ty(&self) -> t::QueryType {
        on!(QuerySet, self, q => q.ty())
    }
}

impl IndirectDraws {
    pub fn max_draws(&self) -> u32 {
        on!(IndirectDraws, self, d => d.max_draws())
    }
}

// ---------------------------------------------------------------------------------------------
// Acceleration structures

/// The backend's top-level structure.
pub(crate) enum TlasRaw {
    #[cfg(target_vendor = "apple")]
    Metal(mx_metal::Tlas),
    #[cfg(windows)]
    Dx12(mx_dx12::Tlas),
    #[cfg(feature = "vulkan")]
    Vulkan(mx_vulkan::Tlas),
    #[cfg(feature = "gl")]
    Gl(mx_gl::Tlas),
    #[cfg(windows)]
    Dx11(mx_dx11::Tlas),
    #[cfg(feature = "noop")]
    Noop(mx_noop::Tlas),
}

/// A top-level acceleration structure. Instances are set by index, as in wgpu
/// (`tlas[i] = Some(instance)`), and handed to the backend when it is built.
pub struct Tlas {
    pub(crate) raw: Mutex<TlasRaw>,
    index: u32,
    instances: Vec<Option<crate::TlasInstance>>,
}

impl Tlas {
    pub fn max_instances(&self) -> u32 {
        self.instances.len() as u32
    }

    /// The structure's index in the bindless heap, for ray queries through
    /// `@group(7) @binding(11) var mx_tlas: binding_array<acceleration_structure>;`
    /// (`mx_tlas[i]`). It shares the index space of texture views; it is valid until the
    /// structure is dropped, and reads what its last build made.
    pub fn index(&self) -> u32 {
        self.index
    }

    pub(crate) fn lock_for(&self, backend: t::Backend) -> parking_lot::MutexGuard<'_, TlasRaw> {
        let g = self.raw.lock();
        let _ = backend;
        g
    }
}

impl Index<usize> for Tlas {
    type Output = Option<crate::TlasInstance>;
    fn index(&self, i: usize) -> &Self::Output {
        &self.instances[i]
    }
}

impl IndexMut<usize> for Tlas {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.instances[i]
    }
}

// ---------------------------------------------------------------------------------------------
// Commands

/// Each backend's command buffers out of a list of mX's (all of one backend).
impl CommandEncoder {
    pub fn finish(self) -> CommandBuffer {
        wrap!(CommandEncoder, CommandBuffer, self, e => e.finish())
    }
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: Option<u64>) {
        on!(CommandEncoder, self, e => e.clear_buffer(buffer.pick(), offset, size))
    }
    pub fn clear_texture(&mut self, texture: &Texture, range: &t::ImageSubresourceRange) {
        on!(CommandEncoder, self, e => e.clear_texture(texture.pick(), range))
    }
    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64) {
        on!(CommandEncoder, self, e => e.copy_buffer_to_buffer(src.pick(), src_offset, dst.pick(), dst_offset, size))
    }
    pub fn copy_buffer_to_texture(&mut self, src: crate::TexelCopyBufferInfo<'_>, dst: crate::TexelCopyTextureInfo<'_>, size: t::Extent3d) {
        on!(CommandEncoder, self, e => e.copy_buffer_to_texture(conv::buffer_info(&src), conv::texture_info(&dst), size))
    }
    pub fn copy_texture_to_buffer(&mut self, src: crate::TexelCopyTextureInfo<'_>, dst: crate::TexelCopyBufferInfo<'_>, size: t::Extent3d) {
        on!(CommandEncoder, self, e => e.copy_texture_to_buffer(conv::texture_info(&src), conv::buffer_info(&dst), size))
    }
    pub fn copy_texture_to_texture(&mut self, src: crate::TexelCopyTextureInfo<'_>, dst: crate::TexelCopyTextureInfo<'_>, size: t::Extent3d) {
        on!(CommandEncoder, self, e => e.copy_texture_to_texture(conv::texture_info(&src), conv::texture_info(&dst), size))
    }
    pub fn generate_mipmaps(&mut self, texture: &Texture) {
        on!(CommandEncoder, self, e => e.generate_mipmaps(texture.pick()))
    }
    pub fn resolve_query_set(&mut self, set: &QuerySet, range: Range<u32>, dst: &Buffer, dst_offset: u64) {
        on!(CommandEncoder, self, e => e.resolve_query_set(set.pick(), range, dst.pick(), dst_offset))
    }
    pub fn write_timestamp(&mut self, set: &QuerySet, index: u32) {
        on!(CommandEncoder, self, e => e.write_timestamp(set.pick(), index))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indirect_draws(
        &mut self,
        draws: &IndirectDraws,
        pipeline: &RenderPipeline,
        args: &Buffer,
        args_offset: u64,
        count: &Buffer,
        count_offset: u64,
        index: Option<(&Buffer, u64, t::IndexFormat)>,
    ) {
        on!(CommandEncoder, self, e => e.encode_indirect_draws(draws.pick(), pipeline.pick(), args.pick(), args_offset, count.pick(), count_offset, conv::index(index)))
    }
    pub fn build_acceleration_structures<'a>(&mut self, blas: impl IntoIterator<Item = &'a crate::BlasBuildEntry<'a>>, tlas: impl IntoIterator<Item = &'a Tlas>) {
        let blas: Vec<&crate::BlasBuildEntry<'a>> = blas.into_iter().collect();
        let tlas: Vec<&Tlas> = tlas.into_iter().collect();
        macro_rules! go {
            ($v:ident, $k:ident, $e:expr) => {{
                let entries: Vec<_> = blas.iter().map(|b| conv::blas_entry(b)).collect();
                let mut guards: Vec<_> = tlas.iter().map(|t| (t, t.raw.lock())).collect();
                for (t, g) in &mut guards {
                    #[allow(irrefutable_let_patterns)]
                    let TlasRaw::$v(raw) = &mut **g else { mismatch() };
                    for (i, inst) in t.instances.iter().enumerate() {
                        raw[i] = inst.as_ref().map(|inst| t::TlasInstance {
                            blas: Pick::<$k::Blas>::pick(&inst.blas).clone(),
                            transform: inst.transform,
                            custom_data: inst.custom_data,
                            mask: inst.mask,
                        });
                    }
                }
                let raws: Vec<_> = guards
                    .iter()
                    .map(|(_, g)| match &**g {
                        TlasRaw::$v(raw) => raw,
                        #[allow(unreachable_patterns)]
                        _ => mismatch(),
                    })
                    .collect();
                $e.build_acceleration_structures(entries.iter(), raws.into_iter());
            }};
        }
        match self {
            #[cfg(target_vendor = "apple")]
            CommandEncoder::Metal(e) => go!(Metal, mx_metal, e),
            #[cfg(windows)]
            CommandEncoder::Dx12(e) => go!(Dx12, mx_dx12, e),
            #[cfg(feature = "vulkan")]
            CommandEncoder::Vulkan(e) => go!(Vulkan, mx_vulkan, e),
            #[cfg(feature = "gl")]
            CommandEncoder::Gl(e) => go!(Gl, mx_gl, e),
            #[cfg(windows)]
            CommandEncoder::Dx11(e) => go!(Dx11, mx_dx11, e),
            #[cfg(feature = "noop")]
            CommandEncoder::Noop(e) => go!(Noop, mx_noop, e),
        }
    }
    pub fn push_debug_group(&mut self, label: &str) {
        on!(CommandEncoder, self, e => e.push_debug_group(label))
    }
    pub fn pop_debug_group(&mut self) {
        on!(CommandEncoder, self, e => e.pop_debug_group())
    }
    pub fn insert_debug_marker(&mut self, label: &str) {
        on!(CommandEncoder, self, e => e.insert_debug_marker(label))
    }
    pub fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<'_>) -> RenderPass<'_> {
        wrap!(CommandEncoder, RenderPass, self, e => {
            let colors = conv::colors(desc.color_attachments);
            e.begin_render_pass(&conv::render_pass(desc, &colors))
        })
    }
    pub fn begin_compute_pass(&mut self, desc: &crate::ComputePassDescriptor<'_>) -> ComputePass<'_> {
        wrap!(CommandEncoder, ComputePass, self, e => e.begin_compute_pass(&conv::compute_pass(desc)))
    }
}

/// A render pass. Ends when dropped.
pub enum RenderPass<'e> {
    #[cfg(target_vendor = "apple")]
    Metal(mx_metal::RenderPass<'e>),
    #[cfg(windows)]
    Dx12(mx_dx12::RenderPass<'e>),
    #[cfg(feature = "vulkan")]
    Vulkan(mx_vulkan::RenderPass<'e>),
    #[cfg(feature = "gl")]
    Gl(mx_gl::RenderPass<'e>),
    #[cfg(windows)]
    Dx11(mx_dx11::RenderPass<'e>),
    #[cfg(feature = "noop")]
    Noop(mx_noop::RenderPass<'e>),
}

/// A compute pass. Ends when dropped.
pub enum ComputePass<'e> {
    #[cfg(target_vendor = "apple")]
    Metal(mx_metal::ComputePass<'e>),
    #[cfg(windows)]
    Dx12(mx_dx12::ComputePass<'e>),
    #[cfg(feature = "vulkan")]
    Vulkan(mx_vulkan::ComputePass<'e>),
    #[cfg(feature = "gl")]
    Gl(mx_gl::ComputePass<'e>),
    #[cfg(windows)]
    Dx11(mx_dx11::ComputePass<'e>),
    #[cfg(feature = "noop")]
    Noop(mx_noop::ComputePass<'e>),
}

/// The drawing methods render passes and render bundle encoders share.
macro_rules! drawing {
    ($ty:ident) => {
        pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
            on!($ty, self, p => p.set_pipeline(pipeline.pick()))
        }
        pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
            on!($ty, self, p => p.set_bind_group(group, bind_group.pick()))
        }
        /// Read the uniform buffer at (`group`, `binding`) of the bind group set at `group`
        /// from `offset` bytes past its own offset, for the draws that follow (until that group
        /// is set again): a draw's constants in one buffer of many, without a bind group each
        /// (as wgpu's dynamic offsets or Direct3D 12's root constant buffer views). `offset` is
        /// a multiple of `min_uniform_buffer_offset_alignment`.
        pub fn set_buffer_offset(&mut self, group: u32, binding: u32, offset: u64) {
            on!($ty, self, p => p.set_buffer_offset(group, binding, offset))
        }
        pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
            on!($ty, self, p => p.set_immediates(offset, data))
        }
        pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64) {
            on!($ty, self, p => p.set_vertex_buffer(slot, buffer.pick(), offset))
        }
        pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64, format: t::IndexFormat) {
            on!($ty, self, p => p.set_index_buffer(buffer.pick(), offset, format))
        }
        pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
            on!($ty, self, p => p.draw(vertices, instances))
        }
        pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
            on!($ty, self, p => p.draw_indexed(indices, base_vertex, instances))
        }
        pub fn draw_indirect(&mut self, args: &Buffer, offset: u64) {
            on!($ty, self, p => p.draw_indirect(args.pick(), offset))
        }
        pub fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64) {
            on!($ty, self, p => p.draw_indexed_indirect(args.pick(), offset))
        }
    };
}

impl RenderPass<'_> {
    drawing!(RenderPass);

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        on!(RenderPass, self, p => p.set_viewport(x, y, w, h, min_depth, max_depth))
    }
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        on!(RenderPass, self, p => p.set_scissor_rect(x, y, w, h))
    }
    pub fn set_stencil_reference(&mut self, reference: u32) {
        on!(RenderPass, self, p => p.set_stencil_reference(reference))
    }
    pub fn set_blend_constant(&mut self, c: t::Color) {
        on!(RenderPass, self, p => p.set_blend_constant(c))
    }
    pub fn multi_draw_indirect(&mut self, args: &Buffer, offset: u64, count: u32) {
        on!(RenderPass, self, p => p.multi_draw_indirect(args.pick(), offset, count))
    }
    pub fn multi_draw_indexed_indirect(&mut self, args: &Buffer, offset: u64, count: u32) {
        on!(RenderPass, self, p => p.multi_draw_indexed_indirect(args.pick(), offset, count))
    }
    pub fn execute_indirect_draws(&mut self, draws: &IndirectDraws) {
        on!(RenderPass, self, p => p.execute_indirect_draws(draws.pick()))
    }
    pub fn begin_occlusion_query(&mut self, index: u32) {
        on!(RenderPass, self, p => p.begin_occlusion_query(index))
    }
    pub fn end_occlusion_query(&mut self) {
        on!(RenderPass, self, p => p.end_occlusion_query())
    }
    pub fn execute_bundles<'b>(&mut self, bundles: impl IntoIterator<Item = &'b RenderBundle>) {
        let bundles: Vec<&RenderBundle> = bundles.into_iter().collect();
        on!(RenderPass, self, p => p.execute_bundles(bundles.iter().map(|b| b.pick())))
    }
    pub fn push_debug_group(&mut self, label: &str) {
        on!(RenderPass, self, p => p.push_debug_group(label))
    }
    pub fn pop_debug_group(&mut self) {
        on!(RenderPass, self, p => p.pop_debug_group())
    }
    pub fn insert_debug_marker(&mut self, label: &str) {
        on!(RenderPass, self, p => p.insert_debug_marker(label))
    }
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        on!(ComputePass, self, p => p.set_pipeline(pipeline.pick()))
    }
    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        on!(ComputePass, self, p => p.set_bind_group(group, bind_group.pick()))
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        on!(ComputePass, self, p => p.set_immediates(offset, data))
    }
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        on!(ComputePass, self, p => p.dispatch_workgroups(x, y, z))
    }
    pub fn dispatch_workgroups_indirect(&mut self, args: &Buffer, offset: u64) {
        on!(ComputePass, self, p => p.dispatch_workgroups_indirect(args.pick(), offset))
    }
    pub fn push_debug_group(&mut self, label: &str) {
        on!(ComputePass, self, p => p.push_debug_group(label))
    }
    pub fn pop_debug_group(&mut self) {
        on!(ComputePass, self, p => p.pop_debug_group())
    }
    pub fn insert_debug_marker(&mut self, label: &str) {
        on!(ComputePass, self, p => p.insert_debug_marker(label))
    }
}

/// What render passes and render bundle encoders both do, for code that records into either
/// (as `wgpu::util::RenderEncoder`).
pub trait RenderEncoder {
    fn set_pipeline(&mut self, pipeline: &RenderPipeline);
    fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup);
    fn set_buffer_offset(&mut self, group: u32, binding: u32, offset: u64);
    fn set_immediates(&mut self, offset: u32, data: &[u8]);
    fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64);
    fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64, format: t::IndexFormat);
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);
    fn draw_indirect(&mut self, args: &Buffer, offset: u64);
    fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64);
}

macro_rules! render_encoder {
    ($ty:ty) => {
        impl RenderEncoder for $ty {
            fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
                Self::set_pipeline(self, pipeline)
            }
            fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
                Self::set_bind_group(self, group, bind_group)
            }
            fn set_buffer_offset(&mut self, group: u32, binding: u32, offset: u64) {
                Self::set_buffer_offset(self, group, binding, offset)
            }
            fn set_immediates(&mut self, offset: u32, data: &[u8]) {
                Self::set_immediates(self, offset, data)
            }
            fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64) {
                Self::set_vertex_buffer(self, slot, buffer, offset)
            }
            fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64, format: t::IndexFormat) {
                Self::set_index_buffer(self, buffer, offset, format)
            }
            fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
                Self::draw(self, vertices, instances)
            }
            fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
                Self::draw_indexed(self, indices, base_vertex, instances)
            }
            fn draw_indirect(&mut self, args: &Buffer, offset: u64) {
                Self::draw_indirect(self, args, offset)
            }
            fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64) {
                Self::draw_indexed_indirect(self, args, offset)
            }
        }
    };
}

render_encoder!(RenderPass<'_>);
render_encoder!(RenderBundleEncoder);

impl RenderBundleEncoder {
    drawing!(RenderBundleEncoder);

    pub fn finish(self, label: Option<&str>) -> RenderBundle {
        wrap!(RenderBundleEncoder, RenderBundle, self, e => e.finish(label))
    }
}

// ---------------------------------------------------------------------------------------------
// Surfaces

pub(crate) enum SurfaceRaw {
    #[cfg(target_vendor = "apple")]
    Metal(mx_metal::Surface<'static>),
    #[cfg(windows)]
    Dx12(mx_dx12::Surface<'static>),
    #[cfg(feature = "vulkan")]
    Vulkan(mx_vulkan::Surface<'static>),
    #[cfg(feature = "gl")]
    Gl(mx_gl::Surface<'static>),
    #[cfg(windows)]
    Dx11(mx_dx11::Surface<'static>),
    #[cfg(feature = "noop")]
    Noop(mx_noop::Surface<'static>),
}

impl SurfaceRaw {
    fn backend(&self) -> t::Backend {
        match self {
            #[cfg(target_vendor = "apple")]
            SurfaceRaw::Metal(_) => t::Backend::Metal,
            #[cfg(windows)]
            SurfaceRaw::Dx12(_) => t::Backend::Dx12,
            #[cfg(feature = "vulkan")]
            SurfaceRaw::Vulkan(_) => t::Backend::Vulkan,
            #[cfg(feature = "gl")]
            SurfaceRaw::Gl(_) => t::Backend::Gl,
            #[cfg(windows)]
            SurfaceRaw::Dx11(_) => t::Backend::Dx11,
            #[cfg(feature = "noop")]
            SurfaceRaw::Noop(_) => t::Backend::Noop,
        }
    }
}

/// Anything a surface keeps alive (its window).
trait Keep {}
impl<T> Keep for T {}

/// Where a window's frames go. It takes on the backend of the device it is configured with;
/// configured with a device of another backend (the player changed the graphics API), it lets
/// go of the old backend's swapchain first.
pub struct Surface<'w> {
    instance: Instance,
    display: RawDisplayHandle,
    window: RawWindowHandle,
    inner: Mutex<Option<SurfaceRaw>>,
    _window: Option<Box<dyn Keep + Send + Sync + 'w>>,
}

// SAFETY: the raw handles are only passed to the backends, which take them from any thread
// (Metal's layer is made on the main thread: the caller's contract, as for every backend).
unsafe impl Send for Surface<'_> {}
unsafe impl Sync for Surface<'_> {}

impl Surface<'_> {
    /// The backend's surface for `backend`, made (and any other backend's dropped) if needed.
    fn raw_for(&self, backend: t::Backend) -> Result<parking_lot::MappedMutexGuard<'_, SurfaceRaw>> {
        let mut g = self.inner.lock();
        if g.as_ref().is_none_or(|s| s.backend() != backend) {
            *g = None;
            let b = &self.instance.backends;
            let _ = b;
            // SAFETY: the handles are valid while the surface lives (the caller's contract, or
            // the window the surface owns).
            let made = unsafe {
                match backend {
                    #[cfg(target_vendor = "apple")]
                    t::Backend::Metal => b.metal.as_ref().ok_or(Error::NoDevice)?.create_surface_unsafe(self.display, self.window).map(SurfaceRaw::Metal)?,
                    #[cfg(windows)]
                    t::Backend::Dx12 => b.dx12.as_ref().ok_or(Error::NoDevice)?.create_surface_unsafe(self.display, self.window).map(SurfaceRaw::Dx12)?,
                    #[cfg(feature = "vulkan")]
                    t::Backend::Vulkan => b.vulkan.as_ref().ok_or(Error::NoDevice)?.create_surface_unsafe(self.display, self.window).map(SurfaceRaw::Vulkan)?,
                    #[cfg(feature = "gl")]
                    t::Backend::Gl => b.gl.as_ref().ok_or(Error::NoDevice)?.create_surface_unsafe(self.display, self.window).map(SurfaceRaw::Gl)?,
                    #[cfg(windows)]
                    t::Backend::Dx11 => b.dx11.as_ref().ok_or(Error::NoDevice)?.create_surface_unsafe(self.display, self.window).map(SurfaceRaw::Dx11)?,
                    #[cfg(feature = "noop")]
                    t::Backend::Noop => b.noop.as_ref().ok_or(Error::NoDevice)?.create_surface_unsafe(self.display, self.window).map(SurfaceRaw::Noop)?,
                    #[allow(unreachable_patterns)]
                    other => return Err(Error::Unsupported(format!("{other:?} in this build"))),
                }
            };
            *g = Some(made);
        }
        Ok(parking_lot::MutexGuard::map(g, |s| s.as_mut().expect("made above")))
    }

    /// Whether `adapter` can present to this surface: on Vulkan its queue is asked (a second
    /// graphics card may not reach the window's screen); elsewhere any adapter of the
    /// surface's backend can, and one of another backend cannot.
    pub fn is_supported_by(&self, adapter: &Adapter) -> bool {
        {
            let Ok(raw) = self.raw_for(adapter.backend()) else { return false };
            // (without Vulkan the lock is only taken to see that the backend has a surface)
            #[cfg(not(feature = "vulkan"))]
            drop(raw);
            #[cfg(feature = "vulkan")]
            if let (SurfaceRaw::Vulkan(s), Adapter::Vulkan(a)) = (&*raw, adapter) {
                return s.is_supported_by(a);
            }
        }
        // (with the surface's lock let go: `get_capabilities` takes it again)
        !self.get_capabilities(adapter).formats.is_empty()
    }

    pub fn get_capabilities(&self, adapter: &Adapter) -> t::SurfaceCapabilities {
        let Ok(raw) = self.raw_for(adapter.backend()) else { return Default::default() };
        match (&*raw, adapter) {
            #[cfg(target_vendor = "apple")]
            (SurfaceRaw::Metal(s), Adapter::Metal(a)) => s.get_capabilities(a),
            #[cfg(windows)]
            (SurfaceRaw::Dx12(s), Adapter::Dx12(a)) => s.get_capabilities(a),
            #[cfg(feature = "vulkan")]
            (SurfaceRaw::Vulkan(s), Adapter::Vulkan(a)) => s.get_capabilities(a),
            #[cfg(feature = "gl")]
            (SurfaceRaw::Gl(s), Adapter::Gl(a)) => s.get_capabilities(a),
            #[cfg(windows)]
            (SurfaceRaw::Dx11(s), Adapter::Dx11(a)) => s.get_capabilities(a),
            #[cfg(feature = "noop")]
            (SurfaceRaw::Noop(s), Adapter::Noop(a)) => s.get_capabilities(a),
            #[allow(unreachable_patterns)]
            _ => mismatch(),
        }
    }

    pub fn configure(&self, device: &Device, config: &t::SurfaceConfiguration) {
        let raw = self.raw_for(device.backend()).unwrap_or_else(|e| panic!("mX: no surface for {:?}: {e}", device.backend()));
        match (&*raw, device) {
            #[cfg(target_vendor = "apple")]
            (SurfaceRaw::Metal(s), Device::Metal(d)) => s.configure(d, config),
            #[cfg(windows)]
            (SurfaceRaw::Dx12(s), Device::Dx12(d)) => s.configure(d, config),
            #[cfg(feature = "vulkan")]
            (SurfaceRaw::Vulkan(s), Device::Vulkan(d)) => s.configure(d, config),
            #[cfg(feature = "gl")]
            (SurfaceRaw::Gl(s), Device::Gl(d)) => s.configure(d, config),
            #[cfg(windows)]
            (SurfaceRaw::Dx11(s), Device::Dx11(d)) => s.configure(d, config),
            #[cfg(feature = "noop")]
            (SurfaceRaw::Noop(s), Device::Noop(d)) => s.configure(d, config),
            #[allow(unreachable_patterns)]
            _ => mismatch(),
        }
    }

    /// The configured Metal surface, for interop (MetalFX frame generation presents through
    /// its layer); None before `configure` or on another backend.
    #[cfg(target_vendor = "apple")]
    pub fn with_metal<R>(&self, f: impl FnOnce(&mx_metal::Surface<'static>) -> R) -> Option<R> {
        match self.inner.lock().as_ref() {
            Some(SurfaceRaw::Metal(s)) => Some(f(s)),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// The Direct3D 12 surface, for interop (a vendor's frame generation makes the swapchain:
    /// `mx_dx12::Surface::set_swapchain_maker`); None before `configure` or on another backend.
    #[cfg(windows)]
    pub fn with_dx12<R>(&self, f: impl FnOnce(&mx_dx12::Surface<'static>) -> R) -> Option<R> {
        match self.inner.lock().as_ref() {
            Some(SurfaceRaw::Dx12(s)) => Some(f(s)),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// The Vulkan surface, for interop (`mx_vulkan::Surface::set_swapchain_maker`); None before
    /// `configure` or on another backend.
    #[cfg(feature = "vulkan")]
    pub fn with_vulkan<R>(&self, f: impl FnOnce(&mx_vulkan::Surface<'static>) -> R) -> Option<R> {
        match self.inner.lock().as_ref() {
            Some(SurfaceRaw::Vulkan(s)) => Some(f(s)),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    pub fn get_current_texture(&self) -> t::CurrentSurfaceTexture<SurfaceTexture> {
        let g = self.inner.lock();
        let raw = g.as_ref().expect("get_current_texture on a surface that was never configured");
        fn lift<S>(c: t::CurrentSurfaceTexture<S>, f: impl FnOnce(S) -> SurfaceTexture) -> t::CurrentSurfaceTexture<SurfaceTexture> {
            use t::CurrentSurfaceTexture as C;
            match c {
                C::Success(s) => C::Success(f(s)),
                C::Suboptimal(s) => C::Suboptimal(f(s)),
                C::Timeout => C::Timeout,
                C::Occluded => C::Occluded,
                C::Outdated => C::Outdated,
                C::Lost => C::Lost,
            }
        }
        match raw {
            #[cfg(target_vendor = "apple")]
            SurfaceRaw::Metal(s) => lift(s.get_current_texture(), |st| SurfaceTexture {
                texture: Texture::Metal(st.texture.clone()),
                raw: SurfaceTextureRaw::Metal(st),
            }),
            #[cfg(windows)]
            SurfaceRaw::Dx12(s) => lift(s.get_current_texture(), |st| SurfaceTexture {
                texture: Texture::Dx12(st.texture.clone()),
                raw: SurfaceTextureRaw::Dx12(st),
            }),
            #[cfg(feature = "vulkan")]
            SurfaceRaw::Vulkan(s) => lift(s.get_current_texture(), |st| SurfaceTexture {
                texture: Texture::Vulkan(st.texture.clone()),
                raw: SurfaceTextureRaw::Vulkan(st),
            }),
            #[cfg(feature = "gl")]
            SurfaceRaw::Gl(s) => lift(s.get_current_texture(), |st| SurfaceTexture {
                texture: Texture::Gl(st.texture.clone()),
                raw: SurfaceTextureRaw::Gl(st),
            }),
            #[cfg(windows)]
            SurfaceRaw::Dx11(s) => lift(s.get_current_texture(), |st| SurfaceTexture {
                texture: Texture::Dx11(st.texture.clone()),
                raw: SurfaceTextureRaw::Dx11(st),
            }),
            #[cfg(feature = "noop")]
            SurfaceRaw::Noop(s) => lift(s.get_current_texture(), |st| SurfaceTexture {
                texture: Texture::Noop(st.texture.clone()),
                raw: SurfaceTextureRaw::Noop(st),
            }),
        }
    }
}

enum SurfaceTextureRaw {
    #[cfg(target_vendor = "apple")]
    Metal(mx_metal::SurfaceTexture),
    #[cfg(windows)]
    Dx12(mx_dx12::SurfaceTexture),
    #[cfg(feature = "vulkan")]
    Vulkan(mx_vulkan::SurfaceTexture),
    #[cfg(feature = "gl")]
    Gl(mx_gl::SurfaceTexture),
    #[cfg(windows)]
    Dx11(mx_dx11::SurfaceTexture),
    #[cfg(feature = "noop")]
    Noop(mx_noop::SurfaceTexture),
}

pub struct SurfaceTexture {
    pub texture: Texture,
    raw: SurfaceTextureRaw,
}

impl SurfaceTexture {
    /// Show the frame (after all work submitted so far).
    pub fn present(self) {
        on!(SurfaceTextureRaw, self.raw, s => s.present())
    }
}

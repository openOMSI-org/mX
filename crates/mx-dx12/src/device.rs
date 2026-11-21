//! Instance, adapters, the device and its queue, submissions and the fence.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use mx_shader::hlsl::ShaderModel;
use mx_types as t;
use parking_lot::Mutex;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

use crate::compiler::Compiler;
use crate::heap::{BLOCK, BLOCKS, CpuPool, CpuPools, Heap, MAX_VIEW_HEAP, Views};
use crate::track::Tracker;
use crate::{Buffer, CommandBuffer, Error, Result, hr_err};

/// Whether to turn on the D3D12 debug layer: `MX_DX12_VALIDATION=1` (or `0` to force it off);
/// on by default in debug builds, as wgpu's `InstanceFlags::from_build_config`.
fn validation() -> bool {
    match std::env::var("MX_DX12_VALIDATION") {
        Ok(v) => v != "0",
        Err(_) => cfg!(debug_assertions),
    }
}

/// The entry point: a DXGI factory and the shader compiler.
pub struct Instance {
    pub(crate) factory: Option<IDXGIFactory4>,
    compiler: std::result::Result<Arc<Compiler>, String>,
    pub(crate) allow_tearing: bool,
}

impl Default for Instance {
    fn default() -> Self {
        Self::new()
    }
}

impl Instance {
    pub fn new() -> Instance {
        let validate = validation();
        if validate {
            let mut debug: Option<ID3D12Debug> = None;
            // SAFETY: a plain query; without the Graphics Tools feature it fails and is skipped.
            if unsafe { D3D12GetDebugInterface(&mut debug) }.is_ok()
                && let Some(d) = debug
            {
                // SAFETY: before any device is created, as required.
                unsafe { d.EnableDebugLayer() };
            }
        }
        // SAFETY: plain factory creation (the debug flag needs the DXGI debug layer: retry
        // without it).
        let factory = unsafe {
            let flags = if validate { DXGI_CREATE_FACTORY_DEBUG } else { DXGI_CREATE_FACTORY_FLAGS(0) };
            CreateDXGIFactory2::<IDXGIFactory4>(flags).or_else(|_| CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)))
        };
        let factory = match factory {
            Ok(f) => Some(f),
            Err(e) => {
                log::error!("mX: no DXGI 1.4 factory: {e}");
                None
            }
        };
        let mut allow_tearing = false;
        if let Some(f5) = factory.as_ref().and_then(|f| f.cast::<IDXGIFactory5>().ok()) {
            let mut v = windows::core::BOOL(0);
            // SAFETY: a valid out-pointer of the size given.
            let r = unsafe { f5.CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, (&raw mut v).cast(), std::mem::size_of_val(&v) as u32) };
            allow_tearing = r.is_ok() && v.as_bool();
        }
        Instance {
            factory,
            compiler: Compiler::load().map(Arc::new),
            allow_tearing,
        }
    }

    /// The adapters mX can run on (feature level 11.0, resource binding tier 2, a shader model
    /// the compiler can target), high performance first.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let Some(factory) = &self.factory else { return Vec::new() };
        let compiler = match &self.compiler {
            Ok(c) => c.clone(),
            Err(e) => {
                log::error!("mX: no shader compiler: {e}");
                return Vec::new();
            }
        };
        let mut raw = Vec::new();
        if let Ok(f6) = factory.cast::<IDXGIFactory6>() {
            for i in 0.. {
                // SAFETY: plain enumeration.
                match unsafe { f6.EnumAdapterByGpuPreference::<IDXGIAdapter1>(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) } {
                    Ok(a) => raw.push(a),
                    Err(_) => break,
                }
            }
        } else {
            for i in 0.. {
                // SAFETY: plain enumeration.
                match unsafe { factory.EnumAdapters1(i) } {
                    Ok(a) => raw.push(a),
                    Err(_) => break,
                }
            }
        }
        raw.into_iter().filter_map(|a| Adapter::expose(a, &compiler)).collect()
    }

    /// A discrete GPU if there is one, else an integrated one, else whatever there is.
    pub fn default_adapter(&self) -> Result<Adapter> {
        let mut all = self.enumerate_adapters();
        let rank = |a: &Adapter| match a.0.info.device_type {
            t::DeviceType::DiscreteGpu => 0,
            t::DeviceType::IntegratedGpu => 1,
            _ => 2,
        };
        all.sort_by_key(rank);
        all.into_iter().next().ok_or(Error::NoDevice)
    }
}

pub(crate) struct Caps {
    pub shader_model: ShaderModel,
    pub ray_query: bool,
    /// Enhanced barriers (`ID3D12GraphicsCommandList7::Barrier`), unless
    /// `MX_DX12_LEGACY_BARRIERS` is set.
    pub enhanced_barriers: bool,
}

pub(crate) struct AdapterInner {
    pub raw: IDXGIAdapter1,
    pub device: ID3D12Device,
    pub compiler: Arc<Compiler>,
    pub info: t::AdapterInfo,
    pub caps: Caps,
}

#[derive(Clone)]
pub struct Adapter(pub(crate) Arc<AdapterInner>);

fn feature<T: Default>(device: &ID3D12Device, f: D3D12_FEATURE, mut data: T) -> Option<T> {
    // SAFETY: `data` is the structure `f` asks for, with its size.
    unsafe { device.CheckFeatureSupport(f, (&raw mut data).cast(), std::mem::size_of::<T>() as u32) }.ok().map(|()| data)
}

/// The highest shader model the device takes (wgpu-hal's probe, highest first).
fn device_shader_model(device: &ID3D12Device) -> ShaderModel {
    use ShaderModel as S;
    let models = [
        (D3D_SHADER_MODEL_6_9, S::V6_9),
        (D3D_SHADER_MODEL_6_8, S::V6_8),
        (D3D_SHADER_MODEL_6_7, S::V6_7),
        (D3D_SHADER_MODEL_6_6, S::V6_6),
        (D3D_SHADER_MODEL_6_5, S::V6_5),
        (D3D_SHADER_MODEL_6_4, S::V6_4),
        (D3D_SHADER_MODEL_6_3, S::V6_3),
        (D3D_SHADER_MODEL_6_2, S::V6_2),
        (D3D_SHADER_MODEL_6_1, S::V6_1),
        (D3D_SHADER_MODEL_6_0, S::V6_0),
    ];
    for (raw, _) in models {
        if let Some(d) = feature(device, D3D12_FEATURE_SHADER_MODEL, D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: raw }) {
            return models.iter().find(|(r, _)| *r == d.HighestShaderModel).map_or(S::V5_1, |(_, s)| *s);
        }
    }
    S::V5_1
}

impl Adapter {
    fn expose(raw: IDXGIAdapter1, compiler: &Arc<Compiler>) -> Option<Adapter> {
        // SAFETY: plain getters and device creation (a device is a singleton per adapter).
        let desc = unsafe { raw.GetDesc1() }.ok()?;
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: as above.
        if let Err(e) = unsafe { D3D12CreateDevice(&raw, D3D_FEATURE_LEVEL_11_0, &mut device) } {
            log::info!("mX: skipping an adapter without Direct3D 12: {e}");
            return None;
        }
        let device = device?;
        let options = feature(&device, D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS::default())?;
        if options.ResourceBindingTier.0 < D3D12_RESOURCE_BINDING_TIER_2.0 {
            log::info!("mX: skipping an adapter of resource binding tier 1 (no bindless heap)");
            return None;
        }
        let arch = feature(&device, D3D12_FEATURE_ARCHITECTURE, D3D12_FEATURE_DATA_ARCHITECTURE::default()).unwrap_or_default();
        let device_max = device_shader_model(&device);
        // As wgpu-hal: the lower of the device's and DXC's shader models; SM 5.1 devices are
        // left out with DXC; FXC compiles SM 5.1.
        let shader_model = match compiler.max_shader_model() {
            Some(dxc) => {
                let sm = if device_max < dxc { device_max } else { dxc };
                if sm < ShaderModel::V6_0 {
                    log::info!("mX: skipping an adapter without shader model 6 (DXC needs it)");
                    return None;
                }
                sm
            }
            None => ShaderModel::V5_1,
        };
        let options5 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS5, D3D12_FEATURE_DATA_D3D12_OPTIONS5::default()).unwrap_or_default();
        let ray_query = options5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0 && shader_model >= ShaderModel::V6_5;
        let options12 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS12, D3D12_FEATURE_DATA_D3D12_OPTIONS12::default()).unwrap_or_default();
        let enhanced_barriers = options12.EnhancedBarriersSupported.as_bool() && std::env::var_os("MX_DX12_LEGACY_BARRIERS").is_none();
        let name_len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
        let device_type = if software {
            t::DeviceType::Cpu
        } else if arch.UMA.as_bool() {
            t::DeviceType::IntegratedGpu
        } else {
            t::DeviceType::DiscreteGpu
        };
        let info = t::AdapterInfo {
            name: String::from_utf16_lossy(&desc.Description[..name_len]),
            backend: t::Backend::Dx12,
            device_type,
            vendor: desc.VendorId,
            device: desc.DeviceId,
            driver: format!("{} shader model {}", if compiler.is_dxc() { "DXC" } else { "FXC" }, shader_model.to_str()),
        };
        Some(Adapter(Arc::new(AdapterInner {
            raw,
            device,
            compiler: compiler.clone(),
            info,
            caps: Caps {
                shader_model,
                ray_query,
                enhanced_barriers,
            },
        })))
    }

    pub fn info(&self) -> t::AdapterInfo {
        self.0.info.clone()
    }

    pub fn limits(&self) -> t::Limits {
        t::Limits {
            max_texture_dimension_2d: D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
            max_texture_dimension_3d: D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
            max_texture_array_layers: D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION,
            max_bindless_textures: (MAX_VIEW_HEAP - 1 - BLOCK * BLOCKS).min(500_000),
            max_bindless_samplers: mx_shader::hlsl::SAMPLER_HEAP_SIZE,
            // (two descriptors per buffer view; `request_device` fits both kinds in the heap)
            max_bindless_buffers: 200_000,
            // (a raw view starts on 16 bytes; 256 keeps every backend's rule)
            buffer_view_offset_alignment: 256,
            // (the root signature's 64 DWORDs: 3 special constants, 2 heap tables, 7 group
            // tables, 32 for immediates)
            max_immediate_size: 128,
            max_vertex_buffers: 8,
            max_root_buffers: 13,
            max_color_attachments: 8,
            max_buffer_size: 1 << 31,
            max_storage_buffer_binding_size: 1 << 31,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
        }
    }

    /// The GPU's memory: its dedicated video memory and the budget DXGI gives the program.
    pub fn memory_info(&self) -> Option<t::AdapterMemory> {
        use windows::Win32::Graphics::Dxgi::{DXGI_MEMORY_SEGMENT_GROUP_LOCAL, IDXGIAdapter3};
        use windows::core::Interface;
        // SAFETY: plain queries of a live adapter.
        unsafe {
            let desc = self.0.raw.GetDesc1().ok()?;
            let budget = self.0.raw.cast::<IDXGIAdapter3>().ok().and_then(|a| {
                let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).ok().map(|()| info.Budget)
            });
            Some(t::AdapterMemory {
                dedicated_bytes: desc.DedicatedVideoMemory as u64,
                budget_bytes: budget,
            })
        }
    }

    /// Whether `format` renders and resolves with `samples` samples per pixel.
    pub fn supports_msaa(&self, format: t::TextureFormat, samples: u32) -> bool {
        if samples <= 1 {
            return true;
        }
        let data = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: crate::conv::format(format),
            SampleCount: samples,
            ..Default::default()
        };
        feature(&self.0.device, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, data).is_some_and(|d| d.NumQualityLevels > 0)
    }

    /// Textures a pipeline may read through root bindings (on OpenGL, through anything: texture
    /// units): (vertex stage, fragment stage, the pipeline together).
    pub fn texture_units(&self) -> (u32, u32, u32) {
        (128, 128, 256)
    }

    /// Indexed draws with a base vertex (not below OpenGL 3.2 / ES 3.2).
    pub fn supports_base_vertex(&self) -> bool {
        true
    }

    /// Every Direct3D 12 device samples BC formats.
    pub fn supports_bc(&self) -> bool {
        true
    }

    /// Compute shaders: every bindless device has them.
    pub fn supports_compute(&self) -> bool {
        true
    }

    /// How many storage buffers the vertex and fragment stages may read as root bindings
    /// (any number through the heap).
    pub fn storage_buffers_per_stage(&self) -> (u32, u32) {
        (64, 64)
    }

    /// Whether this GPU runs on the bound tier: the heap bound slot by slot at each draw,
    /// heap accesses indexed by immediates or constants only.
    pub fn is_bound_tier(&self) -> bool {
        false
    }
    /// GPU-driven draw counts (`Device::create_indirect_draws`): `ExecuteIndirect` takes a
    /// count buffer on every Direct3D 12 device.
    pub fn supports_indirect_count(&self) -> bool {
        true
    }

    /// DXR tier 1.1 (inline ray queries) and shader model 6.5.
    pub fn supports_ray_query(&self) -> bool {
        self.0.caps.ray_query
    }

    /// Direct queues always take timestamps.
    pub fn supports_timestamps(&self) -> bool {
        true
    }

    /// The `IDXGIAdapter1`, for interop.
    pub fn raw(&self) -> &IDXGIAdapter1 {
        &self.0.raw
    }

    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Result<(Device, Queue)> {
        let a = &self.0;
        let limits = self.limits();
        let textures = desc.bindless_textures.min(limits.max_bindless_textures).max(1);
        let samplers = desc.bindless_samplers.min(limits.max_bindless_samplers).max(1);
        let room = (MAX_VIEW_HEAP - 1 - BLOCK * BLOCKS).saturating_sub(textures) / 2;
        let buffers = desc.bindless_buffers.min(limits.max_bindless_buffers).min(room).max(1);
        let device = a.device.clone();
        let qdesc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        // SAFETY: plain creations.
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&qdesc) }.map_err(|e| hr_err("command queue", e))?;
        if let Some(l) = desc.label {
            crate::set_name(&queue, l);
        }
        // SAFETY: as above.
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.map_err(|e| hr_err("fence", e))?;
        let views = Views::new(&device, textures, buffers)?;
        // the heap group's sampler index buffer: sampler heap slot i at index i
        let identity: Vec<u32> = (0..mx_shader::hlsl::SAMPLER_HEAP_SIZE).collect();
        let identity_buffer = crate::resource::upload_buffer(&device, bytes_of_u32(&identity), "mX heap sampler indices")?;
        // SAFETY: a structured-buffer SRV of the buffer just made, into its own slot.
        unsafe {
            device.CreateShaderResourceView(
                &identity_buffer,
                Some(&crate::resource::structured_u32_srv(identity.len() as u32)),
                views.view_cpu(views.sampler_index_slot()),
            )
        };
        let zero_buffer = crate::resource::raw_buffer(&device, ZERO_BUFFER_SIZE, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, "mX zeros")?;
        // SAFETY: a plain getter.
        let frequency = unsafe { queue.GetTimestampFrequency() }.unwrap_or(0);
        let mut cpu = CpuPools {
            rtv: CpuPool::new(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV),
            dsv: CpuPool::new(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV),
            view: CpuPool::new(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
        };
        // a null render target for the holes of a pass's colour attachments (as wgpu-hal)
        let null_rtv = cpu.rtv.alloc(&device)?;
        let null_desc = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_RTV { MipSlice: 0, PlaneSlice: 0 },
            },
        };
        // every sampler slot starts as a valid (default) sampler, so the 2048-wide sampler
        // ranges of the root signatures only ever cover initialised descriptors
        let default_sampler = crate::resource::sampler_desc(&t::SamplerDescriptor::default());
        // SAFETY: a null view into our own CPU descriptor; samplers into the fresh heap.
        unsafe {
            device.CreateRenderTargetView(None, Some(&null_desc), null_rtv.handle());
            for i in 0..mx_shader::hlsl::SAMPLER_HEAP_SIZE {
                device.CreateSampler(&default_sampler, views.sampler_cpu(i));
            }
        }
        let shared = Arc::new(Shared {
            device,
            queue,
            fence,
            views,
            heap: Mutex::new(Heap::new(textures, samplers, buffers)),
            cpu: Mutex::new(cpu),
            last_submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            in_flight: Mutex::new(VecDeque::new()),
            pending: Mutex::new(None),
            ring: Mutex::new(Ring::new()),
            lists: Mutex::new(Vec::new()),
            bundle_lists: Mutex::new(Vec::new()),
            compiler: a.compiler.clone(),
            shader_model: a.caps.shader_model,
            enhanced_barriers: a.caps.enhanced_barriers,
            info: a.info.clone(),
            limits: t::Limits {
                max_bindless_textures: textures,
                max_bindless_samplers: samplers,
                max_bindless_buffers: buffers,
                ..limits
            }
            .capped(desc.limits.as_ref()),
            ray_query: a.caps.ray_query,
            timestamp_period: if frequency > 0 { (1e9 / frequency as f64) as f32 } else { 1.0 },
            zero_buffer,
            null_rtv,
            _identity_buffer: identity_buffer,
            shader_cache: Mutex::new(HashMap::new()),
            mips: Mutex::new(crate::command::MipTools::default()),
            indirect: Mutex::new(None),
        });
        Ok((Device { shared: shared.clone() }, Queue { shared }))
    }
}

pub(crate) const ZERO_BUFFER_SIZE: u64 = 256 << 10;

pub(crate) fn bytes_of_u32(v: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and u8 no alignment needs.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

#[derive(Clone, Debug)]
pub struct DeviceDescriptor<'a> {
    pub label: Option<&'a str>,
    /// Texture views the heap holds at once.
    pub bindless_textures: u32,
    /// Samplers the heap holds at once (at most 2048 on Direct3D 12).
    pub bindless_samplers: u32,
    /// Buffer views the heap holds at once.
    pub bindless_buffers: u32,
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
        }
    }
}

/// A position in the queue's order of submissions (the first is 1). It is the value the
/// device's fence reaches when the submission is done.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionIndex(pub u64);

pub(crate) type ListPair = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Compiled shaders by (HLSL, entry point, profile).
pub(crate) type ShaderCache = HashMap<(String, String, String), Arc<Vec<u8>>>;

/// A command list being recorded, with what it uses.
pub(crate) struct Recording {
    lists: Option<ListPair>,
    pub track: Tracker,
    /// Objects the GPU reads while the list runs.
    pub keep: Vec<Box<dyn Any + Send + Sync>>,
    blocks: Vec<u32>,
    block_used: u32,
    scratch: Vec<(Buffer, u64)>,
    /// A storage write happened since the last UAV barrier.
    pub uav_dirty: bool,
    /// The heap's `buffer_generation` whose buffer states this list last applied (0: none).
    pub heap_buffers_applied: u64,
    pub closed: bool,
    pub shared: Arc<Shared>,
}

impl Recording {
    pub fn new(shared: &Arc<Shared>) -> Recording {
        let pair = shared.lists.lock().pop();
        let (alloc, list) = match pair {
            Some((alloc, list)) => {
                // SAFETY: the pair's last list finished on the GPU (it came back through
                // `retire`) or was never submitted.
                unsafe {
                    alloc.Reset().expect("ID3D12CommandAllocator::Reset failed");
                    list.Reset(&alloc, None).expect("ID3D12GraphicsCommandList::Reset failed");
                }
                (alloc, list)
            }
            None => {
                // SAFETY: plain creations.
                unsafe {
                    let alloc: ID3D12CommandAllocator = shared.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("out of memory for a command allocator");
                    let list: ID3D12GraphicsCommandList = shared
                        .device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                        .expect("out of memory for a command list");
                    (alloc, list)
                }
            }
        };
        // SAFETY: the device's two shader-visible heaps, set once per list.
        unsafe { list.SetDescriptorHeaps(&[Some(shared.views.view_heap.clone()), Some(shared.views.sampler_heap.clone())]) };
        Recording {
            lists: Some((alloc, list)),
            track: Tracker::new(shared.enhanced_barriers),
            keep: Vec::new(),
            blocks: Vec::new(),
            block_used: BLOCK,
            scratch: Vec::new(),
            uav_dirty: false,
            heap_buffers_applied: 0,
            closed: false,
            shared: shared.clone(),
        }
    }

    pub fn list(&self) -> &ID3D12GraphicsCommandList {
        &self.lists.as_ref().expect("a finished recording").1
    }

    pub fn keep<T: Any + Send + Sync>(&mut self, t: T) {
        self.keep.push(Box::new(t));
    }

    /// Record the barriers asked for so far.
    pub fn barriers(&mut self) {
        let list = self.lists.as_ref().expect("a finished recording").1.clone();
        self.track.flush(&list);
    }

    /// `n` consecutive descriptors of the shader-visible heap for a root table, valid until
    /// the list is done. Returns the index of the first.
    pub fn table(&mut self, n: u32) -> u32 {
        assert!(n <= BLOCK, "a root group of {n} descriptors; mX takes {BLOCK}");
        if self.block_used + n > BLOCK {
            let mut b = self.shared.heap.lock().alloc_block();
            if b.is_none() {
                // every block is in flight: wait for the GPU to give some back
                let _ = self.shared.wait_idle();
                b = self.shared.heap.lock().alloc_block();
            }
            let b = b.expect("mX: the root descriptor tables of unsubmitted command encoders use the whole ring; submit some first");
            self.blocks.push(b);
            self.block_used = 0;
        }
        let first = self.shared.views.block_start(*self.blocks.last().unwrap()) + self.block_used;
        self.block_used += n;
        first
    }

    /// `size` bytes of a GPU scratch buffer (for rewritten indirect arguments), 4-aligned.
    pub fn scratch(&mut self, size: u64) -> (Buffer, u64) {
        const CHUNK: u64 = 64 << 10;
        let size = size.next_multiple_of(4);
        if let Some((b, used)) = self.scratch.last_mut()
            && *used + size <= b.size()
        {
            let at = *used;
            *used += size;
            return (b.clone(), at);
        }
        let b = crate::resource::create_buffer(
            &self.shared,
            &t::BufferDescriptor {
                label: Some("mX indirect arguments"),
                size: size.max(CHUNK),
                usage: t::BufferUsages::INDIRECT | t::BufferUsages::COPY_DST,
            },
        )
        .expect("out of memory for indirect argument scratch");
        self.scratch.push((b.clone(), size));
        (b, 0)
    }

    /// Put everything back in its home state and close the list.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.track.restore_all();
        self.barriers();
        // SAFETY: a list in the recording state.
        if let Err(e) = unsafe { self.list().Close() } {
            log::error!("mX: closing a command list failed: {e}");
        }
        self.closed = true;
    }

    fn take(&mut self) -> Retired {
        Retired {
            lists: self.lists.take(),
            keep: std::mem::take(&mut self.keep),
            blocks: std::mem::take(&mut self.blocks),
            track: std::mem::take(&mut self.track),
            scratch: std::mem::take(&mut self.scratch),
        }
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        if self.lists.is_some() {
            // (never submitted: nothing ran, so everything can be reused now)
            self.close();
            let r = self.take();
            r.retire(&self.shared);
        }
    }
}

/// What a submission holds until the GPU is done with it.
pub(crate) struct Retired {
    lists: Option<ListPair>,
    keep: Vec<Box<dyn Any + Send + Sync>>,
    blocks: Vec<u32>,
    track: Tracker,
    scratch: Vec<(Buffer, u64)>,
}

impl Retired {
    fn retire(self, shared: &Shared) {
        if let Some(pair) = self.lists {
            shared.lists.lock().push(pair);
        }
        shared.heap.lock().free_blocks(self.blocks);
        drop((self.keep, self.track, self.scratch));
    }
}

/// Staging memory for queue writes: 4 MiB upload-heap blocks, mapped once, written one after
/// the other and reused once the submission that read them is done (a committed resource
/// per write cost an allocation of the driver's each, every frame).
pub(crate) struct Ring {
    free: Vec<Block>,
    current: Option<(Block, u64)>,
    used: Vec<Block>,
    in_flight: VecDeque<(u64, Vec<Block>)>,
}

/// An upload buffer and its CPU address (mapped for its whole life, which D3D12 allows).
#[derive(Clone)]
pub(crate) struct Block {
    pub raw: ID3D12Resource,
    ptr: usize,
}

impl Ring {
    pub const BLOCK: u64 = 4 << 20;
    const KEEP: usize = 8;

    fn new() -> Ring {
        Ring {
            free: Vec::new(),
            current: None,
            used: Vec::new(),
            in_flight: VecDeque::new(),
        }
    }

    fn retire(&mut self, index: u64) {
        if let Some((b, _)) = self.current.take() {
            self.used.push(b);
        }
        if !self.used.is_empty() {
            self.in_flight.push_back((index, std::mem::take(&mut self.used)));
        }
    }

    fn reclaim(&mut self, completed: u64) {
        while self.in_flight.front().is_some_and(|(i, _)| *i <= completed) {
            for b in self.in_flight.pop_front().unwrap().1 {
                if self.free.len() < Self::KEEP {
                    self.free.push(b);
                }
            }
        }
    }
}

pub(crate) struct Shared {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    pub views: Views,
    pub heap: Mutex<Heap>,
    pub cpu: Mutex<CpuPools>,
    pub last_submitted: AtomicU64,
    pub completed: AtomicU64,
    in_flight: Mutex<VecDeque<(u64, Retired)>>,
    pub pending: Mutex<Option<Recording>>,
    /// Staging memory of the queue's writes (used under the `pending` lock).
    ring: Mutex<Ring>,
    lists: Mutex<Vec<ListPair>>,
    /// Bundle allocators and lists of dropped render bundles, for the next ones.
    pub bundle_lists: Mutex<Vec<ListPair>>,
    pub compiler: Arc<Compiler>,
    /// Barriers are recorded as enhanced barriers (see `track`).
    pub enhanced_barriers: bool,
    pub shader_model: ShaderModel,
    pub info: t::AdapterInfo,
    pub limits: t::Limits,
    pub ray_query: bool,
    pub timestamp_period: f32,
    pub zero_buffer: ID3D12Resource,
    pub null_rtv: crate::heap::Cpu,
    _identity_buffer: ID3D12Resource,
    /// Compiled shaders by (HLSL, entry point, profile), as wgpu-hal's cache.
    pub shader_cache: Mutex<ShaderCache>,
    pub mips: Mutex<crate::command::MipTools>,
    /// The compute pipeline that lays out GPU-driven draw arguments for `ExecuteIndirect`.
    pub indirect: Mutex<Option<crate::ComputePipeline>>,
}

impl Shared {
    /// The submission that will be the next one: anything released now may be read by
    /// command lists recorded before, which are submitted at the earliest then.
    pub fn release_after(&self) -> u64 {
        self.last_submitted.load(Ordering::Acquire) + 1
    }

    /// Run `f` on the queue-writes recording (created on first use).
    pub fn with_pending<R>(self: &Arc<Self>, f: impl FnOnce(&mut Recording) -> R) -> R {
        let mut p = self.pending.lock();
        let rec = p.get_or_insert_with(|| Recording::new(self));
        f(rec)
    }

    /// Copy `data` into staging memory the next submission reads: an upload buffer and the
    /// offset of the bytes in it (512-aligned, as texture copies need). Call it inside
    /// `with_pending`, so the bytes go out with the writes that read them; big writes get a
    /// buffer of their own (kept by `rec`).
    pub fn stage(&self, rec: &mut Recording, data: &[u8]) -> Result<(ID3D12Resource, u64)> {
        if data.len() as u64 > Ring::BLOCK / 4 {
            let raw = crate::resource::upload_buffer(&self.device, data, "mX staging")?;
            rec.keep(raw.clone());
            return Ok((raw, 0));
        }
        let mut ring = self.ring.lock();
        let len = (data.len() as u64).max(4);
        let align = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64;
        let fits = ring.current.as_ref().is_some_and(|(_, at)| at.next_multiple_of(align) + len <= Ring::BLOCK);
        if !fits {
            if let Some((full, _)) = ring.current.take() {
                ring.used.push(full);
            }
            let block = match ring.free.pop() {
                Some(b) => b,
                None => {
                    let raw = crate::resource::raw_buffer(
                        &self.device,
                        Ring::BLOCK,
                        D3D12_HEAP_TYPE_UPLOAD,
                        D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                        "mX staging",
                    )?;
                    let mut p = std::ptr::null_mut();
                    let none = D3D12_RANGE { Begin: 0, End: 0 };
                    // SAFETY: an upload buffer, mapped for good (it is never read on the CPU).
                    unsafe { raw.Map(0, Some(&none), Some(&mut p)) }.map_err(|e| hr_err("mapping staging memory", e))?;
                    Block { raw, ptr: p as usize }
                }
            };
            ring.current = Some((block, 0));
        }
        let (block, at) = ring.current.as_mut().expect("a current block");
        let offset = at.next_multiple_of(align);
        // SAFETY: `offset + data.len()` is inside the mapped block (checked above), whose
        // bytes no submitted work reads (blocks come back only once their submission is done).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), (block.ptr as *mut u8).add(offset as usize), data.len()) };
        *at = offset + len;
        Ok((block.raw.clone(), offset))
    }

    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) -> SubmissionIndex {
        let mut in_flight = self.in_flight.lock();
        let index = self.last_submitted.load(Ordering::Acquire) + 1;
        let mut retired = Vec::new();
        let mut pending = self.pending.lock();
        if let Some(mut p) = pending.take() {
            p.close();
            retired.push(p.take());
            // (the ring's blocks the writes read wait for this submission)
            self.ring.lock().retire(index);
        }
        drop(pending);
        for mut b in buffers {
            b.rec.close();
            retired.push(b.rec.take());
        }
        let lists: Vec<Option<ID3D12CommandList>> = retired.iter().filter_map(|r| r.lists.as_ref()).map(|(_, l)| l.cast::<ID3D12CommandList>().ok()).collect();
        // SAFETY: closed lists whose resources the retired records keep alive.
        unsafe {
            if !lists.is_empty() {
                self.queue.ExecuteCommandLists(&lists);
            }
            if let Err(e) = self.queue.Signal(&self.fence, index) {
                log::error!("mX: ID3D12CommandQueue::Signal failed: {e}");
            }
        }
        for r in retired {
            in_flight.push_back((index, r));
        }
        self.last_submitted.store(index, Ordering::Release);
        SubmissionIndex(index)
    }

    /// Note finished submissions and free what they held. With `wait`, block until all
    /// submissions up to `wait` are done.
    pub fn poll(&self, wait: Option<u64>) -> Result<()> {
        // SAFETY: plain fence calls; a null event makes SetEventOnCompletion block until the
        // fence reaches the value.
        let done = unsafe {
            if let Some(target) = wait
                && self.fence.GetCompletedValue() < target
            {
                self.fence.SetEventOnCompletion(target, HANDLE::default()).map_err(|e| hr_err("waiting for the GPU", e))?;
            }
            self.fence.GetCompletedValue()
        };
        if done == u64::MAX {
            // SAFETY: a plain getter.
            let reason = unsafe { self.device.GetDeviceRemovedReason() };
            return Err(Error::DeviceLost(format!("device removed: {reason:?}")));
        }
        let finished: Vec<Retired> = {
            let mut in_flight = self.in_flight.lock();
            let mut v = Vec::new();
            while in_flight.front().is_some_and(|(i, _)| *i <= done) {
                v.push(in_flight.pop_front().unwrap().1);
            }
            v
        };
        self.completed.fetch_max(done, Ordering::AcqRel);
        for r in finished {
            r.retire(self);
        }
        self.ring.lock().reclaim(done);
        let textures = self.heap.lock().reclaim(done);
        drop(textures);
        Ok(())
    }

    pub fn wait_idle(&self) -> Result<()> {
        self.poll(Some(self.last_submitted.load(Ordering::Acquire)))
    }
}

impl Drop for Shared {
    /// The device's own objects (the staging ring, the command lists and their allocators,
    /// what the last submissions keep) go only once the queue is done with them: the debug
    /// layer reported a resource released while the queue still used it when a program
    /// ended right after its last frame (#4).
    fn drop(&mut self) {
        let _ = self.wait_idle();
    }
}

/// The GPU. Cheap to clone; every clone is the same device.
#[derive(Clone)]
pub struct Device {
    pub(crate) shared: Arc<Shared>,
}

impl Device {
    pub fn info(&self) -> &t::AdapterInfo {
        &self.shared.info
    }

    pub fn limits(&self) -> t::Limits {
        self.shared.limits
    }

    /// Note finished work and free what it held, without waiting.
    pub fn poll(&self) -> Result<()> {
        self.shared.poll(None)
    }

    /// Wait until `index` (and everything before it) has finished on the GPU.
    pub fn wait(&self, index: SubmissionIndex) -> Result<()> {
        self.shared.poll(Some(index.0))
    }

    /// Whether `index` has finished on the GPU (as of the last `poll` or `wait`).
    pub fn is_done(&self, index: SubmissionIndex) -> bool {
        self.shared.completed.load(Ordering::Acquire) >= index.0
    }

    pub fn wait_idle(&self) -> Result<()> {
        self.shared.wait_idle()
    }

    /// BC texture compression (as the adapter's `supports_bc`).
    pub fn supports_bc(&self) -> bool {
        true
    }

    /// Timestamp queries (as the adapter's `supports_timestamps`).
    pub fn supports_timestamps(&self) -> bool {
        true
    }

    /// Texture views in the heap right now.
    pub fn bindless_textures_in_use(&self) -> usize {
        self.shared.heap.lock().textures_in_use()
    }

    /// Buffer views in the heap right now.
    pub fn bindless_buffers_in_use(&self) -> usize {
        self.shared.heap.lock().buffers_in_use()
    }

    /// The `ID3D12Device`, for interop (DLSS, FSR 3, OpenXR).
    pub fn raw(&self) -> &ID3D12Device {
        &self.shared.device
    }

    /// The shader model shaders are compiled for.
    pub fn shader_model(&self) -> ShaderModel {
        self.shared.shader_model
    }
}

/// Where command lists go: the device's direct queue.
#[derive(Clone)]
pub struct Queue {
    pub(crate) shared: Arc<Shared>,
}

impl Queue {
    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) -> SubmissionIndex {
        self.shared.submit(buffers)
    }

    /// The `ID3D12CommandQueue`, for interop. Work submitted to it directly is ordered with
    /// mX's but not counted in its submission indices.
    pub fn raw(&self) -> &ID3D12CommandQueue {
        &self.shared.queue
    }

    /// Nanoseconds per timestamp tick (from the queue's timestamp frequency).
    pub fn get_timestamp_period(&self) -> f32 {
        self.shared.timestamp_period
    }
}

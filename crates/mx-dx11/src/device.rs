//! Instance, adapter, device and queue; submissions and their event queries.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use mx_types as t;
use parking_lot::Mutex;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

use crate::heap::Heap;
use crate::{Error, Result, hr};

/// The entry point: a DXGI factory.
#[derive(Clone)]
pub struct Instance {
    pub(crate) factory: Option<IDXGIFactory2>,
    pub(crate) allow_tearing: bool,
}

// SAFETY: DXGI factories are free-threaded.
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl Default for Instance {
    fn default() -> Self {
        Self::new()
    }
}

impl Instance {
    pub fn new() -> Instance {
        // SAFETY: plain creation.
        let factory: Option<IDXGIFactory2> = unsafe { CreateDXGIFactory1() }.ok();
        let allow_tearing = factory.as_ref().and_then(|f| f.cast::<IDXGIFactory5>().ok()).is_some_and(|f5| {
            let mut allow = windows::core::BOOL(0);
            // SAFETY: a plain query into a BOOL.
            unsafe {
                f5.CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                    (&mut allow as *mut windows::core::BOOL).cast(),
                    std::mem::size_of::<windows::core::BOOL>() as u32,
                )
            }
            .is_ok()
                && allow.as_bool()
        });
        Instance { factory, allow_tearing }
    }

    /// Every adapter that makes a feature level 11.0 device; software ones (WARP) last.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let Some(factory) = &self.factory else { return Vec::new() };
        let mut found = Vec::new();
        let mut software = Vec::new();
        for i in 0.. {
            // SAFETY: enumeration until DXGI_ERROR_NOT_FOUND.
            let Ok(a) = (unsafe { factory.EnumAdapters1(i) }) else { break };
            // SAFETY: a plain query.
            let Ok(desc) = (unsafe { a.GetDesc1() }) else { continue };
            match make_device(Some(&a.cast().expect("IDXGIAdapter1 is an IDXGIAdapter"))) {
                Ok((device, context, level)) => {
                    let adapter = Adapter(Arc::new(AdapterInner {
                        raw: a,
                        desc,
                        level,
                        probe: Mutex::new(Some((device, context))),
                    }));
                    if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                        software.push(adapter);
                    } else {
                        found.push(adapter);
                    }
                }
                Err(e) => log::info!("mX: {} makes no Direct3D 11 device mX can use: {e}", name(&desc)),
            }
        }
        found.extend(software);
        found
    }

    pub fn default_adapter(&self) -> Result<Adapter> {
        self.enumerate_adapters().into_iter().next().ok_or(Error::NoDevice)
    }
}

fn name(desc: &DXGI_ADAPTER_DESC1) -> String {
    let n = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
    String::from_utf16_lossy(&desc.Description[..n])
}

/// A feature level 11.0 (or 11.1) device on `adapter` and its immediate context, which must be
/// an `ID3D11DeviceContext1` (constant buffer offsets).
fn make_device(adapter: Option<&IDXGIAdapter>) -> Result<(ID3D11Device, ID3D11DeviceContext1, D3D_FEATURE_LEVEL)> {
    let levels = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0];
    let mut flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    if std::env::var_os("MX_DX11_DEBUG").is_some_and(|v| v == "1") {
        flags |= D3D11_CREATE_DEVICE_DEBUG;
    }
    let mut device = None;
    let mut context = None;
    let mut level = D3D_FEATURE_LEVEL::default();
    // SAFETY: out-parameters of a plain creation.
    unsafe {
        D3D11CreateDevice(
            adapter,
            if adapter.is_some() { D3D_DRIVER_TYPE_UNKNOWN } else { D3D_DRIVER_TYPE_HARDWARE },
            HMODULE::default(),
            flags,
            Some(&levels),
            D3D11_SDK_VERSION,
            Some(&mut device),
            Some(&mut level),
            Some(&mut context),
        )
    }
    .map_err(|e| hr("D3D11CreateDevice", e))?;
    let device = device.ok_or(Error::NoDevice)?;
    let context: ID3D11DeviceContext = context.ok_or(Error::NoDevice)?;
    let context = context
        .cast::<ID3D11DeviceContext1>()
        .map_err(|_| Error::Unsupported("ID3D11DeviceContext1 (Windows 8 or newer)".into()))?;
    Ok((device, context, level))
}

pub(crate) struct AdapterInner {
    pub raw: IDXGIAdapter1,
    pub desc: DXGI_ADAPTER_DESC1,
    pub level: D3D_FEATURE_LEVEL,
    /// The device made to check the adapter, handed to the first `request_device`.
    probe: Mutex<Option<(ID3D11Device, ID3D11DeviceContext1)>>,
}

// SAFETY: DXGI adapters and D3D11 devices are free-threaded; the context is only used under
// the device's lock once handed over.
unsafe impl Send for AdapterInner {}
unsafe impl Sync for AdapterInner {}

#[derive(Clone)]
pub struct Adapter(pub(crate) Arc<AdapterInner>);

impl Adapter {
    pub fn info(&self) -> t::AdapterInfo {
        let d = &self.0.desc;
        let device_type = if d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            t::DeviceType::Cpu
        } else if d.DedicatedVideoMemory < 512 << 20 {
            t::DeviceType::IntegratedGpu
        } else {
            t::DeviceType::DiscreteGpu
        };
        t::AdapterInfo {
            name: name(d),
            backend: t::Backend::Dx11,
            device_type,
            vendor: d.VendorId,
            device: d.DeviceId,
            driver: format!("feature level {:x}", self.0.level.0),
        }
    }

    pub fn limits(&self) -> t::Limits {
        t::Limits {
            max_texture_dimension_2d: 16384,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 2048,
            // (the heap is on the CPU: any size)
            max_bindless_textures: 500_000,
            max_bindless_samplers: 4096,
            max_bindless_buffers: 500_000,
            buffer_view_offset_alignment: 256,
            // (a constant buffer of 4096 vectors at most: the immediates' is `b13`)
            max_immediate_size: 256,
            max_vertex_buffers: 16,
            max_root_buffers: mx_shader::hlsl11::MAX_CBVS,
            max_color_attachments: 8,
            max_buffer_size: 1 << 31,
            max_storage_buffer_binding_size: 1 << 31,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
        }
    }

    pub fn memory_info(&self) -> Option<t::AdapterMemory> {
        let d = &self.0.desc;
        let dedicated = if d.DedicatedVideoMemory > 0 { d.DedicatedVideoMemory as u64 } else { d.SharedSystemMemory as u64 };
        let budget = self.0.raw.cast::<IDXGIAdapter3>().ok().and_then(|a| {
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            // SAFETY: a plain query into `info`.
            unsafe { a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.ok().map(|_| info.Budget)
        });
        Some(t::AdapterMemory {
            dedicated_bytes: dedicated,
            budget_bytes: budget,
        })
    }

    /// Whether `format` renders and resolves with `samples` samples per pixel.
    pub fn supports_msaa(&self, format: t::TextureFormat, samples: u32) -> bool {
        if samples <= 1 {
            return true;
        }
        let Some((device, _)) = self.probe() else { return false };
        // SAFETY: a plain query.
        unsafe { device.CheckMultisampleQualityLevels(crate::conv::format(format), samples) }.is_ok_and(|levels| levels > 0)
    }

    fn probe(&self) -> Option<(ID3D11Device, ID3D11DeviceContext1)> {
        if let Some(p) = self.0.probe.lock().clone() {
            return Some(p);
        }
        make_device(Some(&self.0.raw.cast().ok()?)).ok().map(|(d, c, _)| (d, c))
    }

    /// Textures a pipeline may read through root bindings: 128 shader resource views a stage.
    pub fn texture_units(&self) -> (u32, u32, u32) {
        (mx_shader::hlsl11::MAX_SRVS, mx_shader::hlsl11::MAX_SRVS, 2 * mx_shader::hlsl11::MAX_SRVS)
    }

    pub fn supports_base_vertex(&self) -> bool {
        true
    }

    pub fn supports_bc(&self) -> bool {
        true
    }

    /// Never: ray queries need bindless and acceleration structures.
    pub fn supports_ray_query(&self) -> bool {
        false
    }

    pub fn supports_timestamps(&self) -> bool {
        true
    }

    /// Whether this GPU runs on the bound tier: always on Direct3D 11.
    pub fn is_bound_tier(&self) -> bool {
        true
    }

    /// Never on this tier: GPU-driven batches need indirect command buffers or draw counts.
    pub fn supports_indirect_count(&self) -> bool {
        false
    }

    pub fn supports_compute(&self) -> bool {
        true
    }

    /// Storage buffers in the vertex and fragment stages: shader resource views (read) or UAVs
    /// (written, fragment stage only, 8 on feature level 11.0).
    pub fn storage_buffers_per_stage(&self) -> (u32, u32) {
        (mx_shader::hlsl11::MAX_SRVS, mx_shader::hlsl11::MAX_SRVS)
    }

    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Result<(Device, Queue)> {
        let (device, context) = match self.0.probe.lock().take() {
            Some(p) => p,
            None => {
                let (d, c, _) = make_device(Some(&self.0.raw.cast().map_err(|e| hr("IDXGIAdapter", e))?))?;
                (d, c)
            }
        };
        let limits = self.limits();
        let limits = t::Limits {
            max_bindless_textures: desc.bindless_textures.min(limits.max_bindless_textures),
            max_bindless_samplers: desc.bindless_samplers.min(limits.max_bindless_samplers),
            max_bindless_buffers: desc.bindless_buffers.min(limits.max_bindless_buffers),
            ..limits
        }
        .capped(desc.limits.as_ref());
        let annotation = context.cast::<ID3DUserDefinedAnnotation>().ok();
        let constants = |size: u32| -> Result<ID3D11Buffer> {
            let d = D3D11_BUFFER_DESC {
                ByteWidth: size,
                Usage: D3D11_USAGE_DYNAMIC,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                MiscFlags: 0,
                StructureByteStride: 0,
            };
            let mut b = None;
            // SAFETY: plain creation.
            unsafe { device.CreateBuffer(&d, None, Some(&mut b)) }.map_err(|e| hr("constant buffer", e))?;
            b.ok_or_else(|| Error::OutOfMemory("constant buffer".into()))
        };
        let immediates = constants(256)?;
        let special = constants(16)?;
        let shared = Arc::new(Shared {
            heap: Mutex::new(Heap::new(limits.max_bindless_textures, limits.max_bindless_samplers, limits.max_bindless_buffers)),
            last_submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            events: Mutex::new(VecDeque::new()),
            info: self.info(),
            limits,
            ctx: Mutex::new(Ctx {
                context,
                annotation,
                immediates,
                special,
                disjoint: None,
                timestamp_frequency: 1_000_000_000,
            }),
            device,
        });
        Ok((Device { shared: shared.clone() }, Queue { shared }))
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

/// A position in the queue's order of submissions (the first is 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionIndex(pub u64);

/// The immediate context and what lives with it (used only under its lock).
pub(crate) struct Ctx {
    pub context: ID3D11DeviceContext1,
    pub annotation: Option<ID3DUserDefinedAnnotation>,
    /// The immediates' constant buffer (`b13`), written before each draw that changed them.
    pub immediates: ID3D11Buffer,
    /// naga's special constants (`b12`): the first vertex and instance of each draw.
    pub special: ID3D11Buffer,
    /// The disjoint query timestamps of the current submission are measured under.
    pub disjoint: Option<ID3D11Query>,
    /// Ticks a second of the last disjoint query that ended.
    pub timestamp_frequency: u64,
}

pub(crate) struct Shared {
    pub device: ID3D11Device,
    pub ctx: Mutex<Ctx>,
    pub heap: Mutex<Heap>,
    pub last_submitted: AtomicU64,
    pub completed: AtomicU64,
    /// The event query after each submission, oldest first.
    pub events: Mutex<VecDeque<(u64, ID3D11Query)>>,
    pub info: t::AdapterInfo,
    pub limits: t::Limits,
}

// SAFETY: the device is free-threaded; the context and the queries are used only under the
// `ctx` lock.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    /// The submission that will be the next one: anything released now may be read by
    /// command buffers recorded before, which are submitted at the earliest then.
    pub fn release_after(&self) -> u64 {
        self.last_submitted.load(Ordering::Acquire) + 1
    }

    /// Note finished submissions (waiting up to `wait_for`'s completion), free what they held.
    pub fn poll(&self, wait_for: Option<u64>) -> Result<()> {
        let ctx = self.ctx.lock();
        let mut events = self.events.lock();
        let mut completed = self.completed.load(Ordering::Acquire);
        while let Some((index, query)) = events.front().cloned() {
            let must = wait_for.is_some_and(|w| index <= w);
            let mut done = windows::core::BOOL(0);
            loop {
                // SAFETY: an event query of this device, ended; the result is a BOOL.
                let r = unsafe { crate::query::get_data(&ctx.context, &query, (&mut done as *mut windows::core::BOOL).cast(), std::mem::size_of::<windows::core::BOOL>() as u32) };
                if r.is_err() {
                    return Err(hr("waiting for the GPU", r.into()));
                }
                if done.as_bool() || !must {
                    break;
                }
                std::thread::yield_now();
            }
            if !done.as_bool() {
                break;
            }
            events.pop_front();
            completed = index;
        }
        if events.is_empty() {
            completed = self.last_submitted.load(Ordering::Acquire);
        }
        drop(events);
        drop(ctx);
        self.completed.store(completed, Ordering::Release);
        self.heap.lock().reclaim(completed);
        // SAFETY: a plain query.
        if let Err(e) = unsafe { self.device.GetDeviceRemovedReason() } {
            return Err(Error::DeviceLost(e.to_string()));
        }
        Ok(())
    }

    pub fn wait_idle(&self) -> Result<()> {
        self.poll(Some(self.last_submitted.load(Ordering::Acquire)))
    }
}

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

    pub fn wait_idle(&self) -> Result<()> {
        self.shared.wait_idle()
    }

    /// Whether `index` has finished on the GPU (as of the last `poll` or `wait`).
    pub fn is_done(&self, index: SubmissionIndex) -> bool {
        self.shared.completed.load(Ordering::Acquire) >= index.0
    }

    pub fn supports_bc(&self) -> bool {
        true
    }

    pub fn supports_timestamps(&self) -> bool {
        true
    }

    /// Whether the device is known lost (a call on it said so; Direct3D 12 tells it at any time).
    pub fn is_lost(&self) -> bool {
        false
    }

    pub fn bindless_textures_in_use(&self) -> usize {
        self.shared.heap.lock().textures_in_use()
    }

    pub fn bindless_buffers_in_use(&self) -> usize {
        self.shared.heap.lock().buffers_in_use()
    }

    /// The `ID3D11Device`, for interop.
    pub fn raw(&self) -> &ID3D11Device {
        &self.shared.device
    }

    /// Run `f` with the immediate context, mX's own work held off meanwhile (interop).
    pub fn with_context<R>(&self, f: impl FnOnce(&ID3D11DeviceContext1) -> R) -> R {
        let ctx = self.shared.ctx.lock();
        f(&ctx.context)
    }
}

/// Where command buffers go.
#[derive(Clone)]
pub struct Queue {
    pub(crate) shared: Arc<Shared>,
}

impl Queue {
    /// Replay the command buffers on the immediate context, in order, then an event query.
    pub fn submit(&self, buffers: impl IntoIterator<Item = crate::CommandBuffer>) -> SubmissionIndex {
        let s = &self.shared;
        let mut ctx = s.ctx.lock();
        for b in buffers {
            crate::command::execute(&mut ctx, s, b);
        }
        crate::query::end_disjoint(&mut ctx);
        let index = s.last_submitted.fetch_add(1, Ordering::AcqRel) + 1;
        let desc = D3D11_QUERY_DESC {
            Query: D3D11_QUERY_EVENT,
            MiscFlags: 0,
        };
        let mut q = None;
        // SAFETY: plain creation; the query ends on the immediate context.
        match unsafe { s.device.CreateQuery(&desc, Some(&mut q)) } {
            Ok(()) if q.is_some() => {
                let q = q.expect("checked");
                unsafe {
                    ctx.context.End(&q);
                    ctx.context.Flush();
                }
                s.events.lock().push_back((index, q));
            }
            _ => {
                log::error!("mX: no event query for submission {index}; it is waited for at once");
                // SAFETY: a plain flush.
                unsafe { ctx.context.Flush() };
            }
        }
        SubmissionIndex(index)
    }

    /// Nanoseconds per timestamp tick: timestamps are converted to nanoseconds when resolved.
    pub fn get_timestamp_period(&self) -> f32 {
        1.0
    }
}

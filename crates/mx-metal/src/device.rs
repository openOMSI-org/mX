//! Instance, adapters, the device and its queue.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use mx_types as t;
use objc2_metal::*;
use parking_lot::Mutex;

use crate::heap::Heap;
use crate::{CommandBuffer, Error, Mt, Po, Result};

/// The entry point. On Apple platforms it only lists Metal devices.
#[derive(Default)]
pub struct Instance {}

impl Instance {
    pub fn new() -> Instance {
        Instance {}
    }

    /// Every Metal 3 device, the system's default first.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let mut devices: Vec<Po<dyn MTLDevice>> = Vec::new();
        if let Some(d) = MTLCreateSystemDefaultDevice() {
            devices.push(d);
        }
        #[cfg(target_os = "macos")]
        for d in MTLCopyAllDevices().iter() {
            if !devices.iter().any(|x| x.registryID() == d.registryID()) {
                devices.push(d);
            }
        }
        // (Metal 3 first; Metal 2 GPUs after them, on the bound tier, unless MX_METAL2=0)
        let metal2 = std::env::var_os("MX_METAL2").is_none_or(|v| v != "0");
        let (mut three, two): (Vec<_>, Vec<_>) = devices.into_iter().partition(|d| d.supportsFamily(MTLGPUFamily::Metal3));
        if metal2 {
            three.extend(two.into_iter().filter(|d| d.supportsFamily(MTLGPUFamily::Mac2) || d.supportsFamily(MTLGPUFamily::Apple4)));
        }
        three.into_iter().map(|d| Adapter { raw: Mt(d) }).collect()
    }

    /// The system's default Metal device (Metal 3 where there is one).
    pub fn default_adapter(&self) -> Result<Adapter> {
        self.enumerate_adapters().into_iter().next().ok_or(Error::NoDevice)
    }
}

#[derive(Clone)]
pub struct Adapter {
    raw: Mt<Po<dyn MTLDevice>>,
}

impl Adapter {
    /// A Metal 2 GPU: mX runs it on the bound tier (heap accesses indexed by immediates or
    /// constants only; no GPU-driven batches, no ray queries). `MX_METAL_BOUND=1` runs a Metal 3
    /// GPU so too, to test the tier.
    pub fn is_bound_tier(&self) -> bool {
        !self.raw.supportsFamily(MTLGPUFamily::Metal3) || std::env::var_os("MX_METAL_BOUND").is_some_and(|v| v == "1")
    }

    pub fn info(&self) -> t::AdapterInfo {
        let d = &self.raw;
        let device_type = if d.hasUnifiedMemory() || d.isLowPower() {
            t::DeviceType::IntegratedGpu
        } else {
            t::DeviceType::DiscreteGpu
        };
        t::AdapterInfo {
            name: d.name().to_string(),
            backend: t::Backend::Metal,
            device_type,
            vendor: 0,
            device: 0,
            driver: String::new(),
        }
    }

    pub fn limits(&self) -> t::Limits {
        let d = &self.raw;
        // (Apple GPU family 7+, i.e. Apple silicon and every Metal 3 Mac, take 16384²)
        t::Limits {
            max_texture_dimension_2d: 16384,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 2048,
            // (the bound tier keeps the heap on the CPU: any size)
            max_bindless_textures: if self.is_bound_tier() || d.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2 {
                500_000
            } else {
                0
            },
            max_bindless_samplers: 2048,
            max_bindless_buffers: if self.is_bound_tier() || d.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2 {
                500_000
            } else {
                0
            },
            // (a GPU address needs only 4-byte alignment; 256 keeps every backend's rule)
            buffer_view_offset_alignment: 256,
            max_immediate_size: 4096,
            max_vertex_buffers: mx_shader::msl::MAX_VERTEX_BUFFERS,
            max_root_buffers: mx_shader::msl::MAX_ROOT_BUFFERS,
            max_color_attachments: 8,
            max_buffer_size: d.maxBufferLength() as u64,
            max_storage_buffer_binding_size: d.maxBufferLength() as u64,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
        }
    }

    /// The GPU's memory: the working set the system recommends (on Apple silicon, the share of
    /// unified memory the GPU may use).
    pub fn memory_info(&self) -> Option<t::AdapterMemory> {
        let set = self.raw.recommendedMaxWorkingSetSize();
        Some(t::AdapterMemory {
            dedicated_bytes: set,
            budget_bytes: Some(set),
        })
    }

    /// Whether `format` renders and resolves with `samples` samples per pixel.
    pub fn supports_msaa(&self, format: t::TextureFormat, samples: u32) -> bool {
        let _ = format;
        samples <= 1 || self.raw.supportsTextureSampleCount(samples as usize)
    }

    /// Textures a pipeline may read through root bindings (on OpenGL, through anything: texture
    /// units): (vertex stage, fragment stage, the pipeline together).
    pub fn texture_units(&self) -> (u32, u32, u32) {
        (mx_shader::msl::MAX_ROOT_TEXTURES, mx_shader::msl::MAX_ROOT_TEXTURES, 2 * mx_shader::msl::MAX_ROOT_TEXTURES)
    }

    /// Indexed draws with a base vertex (not below OpenGL 3.2 / ES 3.2).
    pub fn supports_base_vertex(&self) -> bool {
        true
    }

    pub fn supports_bc(&self) -> bool {
        self.raw.supportsBCTextureCompression()
    }

    pub fn supports_ray_query(&self) -> bool {
        !self.is_bound_tier() && self.raw.supportsRaytracing()
    }

    /// Compute shaders: every bindless device has them.
    pub fn supports_compute(&self) -> bool {
        true
    }

    /// How many storage buffers the vertex and fragment stages may read as root bindings
    /// (any number through the heap).
    pub fn storage_buffers_per_stage(&self) -> (u32, u32) {
        (31, 31)
    }

    /// GPU-driven draw counts (`Device::create_indirect_draws`): indirect command buffers,
    /// on every Metal 3 device (their kernels read the heap through argument buffers).
    pub fn supports_indirect_count(&self) -> bool {
        !self.is_bound_tier()
    }

    pub fn supports_timestamps(&self) -> bool {
        crate::query::timestamp_counter_set(&self.raw).is_some()
    }

    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Result<(Device, Queue)> {
        let limits = self.limits();
        let textures = desc.bindless_textures.min(limits.max_bindless_textures);
        let samplers = desc.bindless_samplers.min(limits.max_bindless_samplers);
        let buffers = desc.bindless_buffers.min(limits.max_bindless_buffers).max(1);
        if textures == 0 {
            return Err(Error::Unsupported("Tier 2 argument buffers (bindless)".into()));
        }
        let queue = self.raw.newCommandQueue().ok_or_else(|| Error::OutOfMemory("command queue".into()))?;
        if let Some(l) = desc.label {
            queue.setLabel(Some(&crate::ns(l)));
        }
        let heap = Heap::new(&self.raw, textures, samplers, buffers)?;
        if let Some(set) = heap.residency_set() {
            // (the heap's textures stay resident for everything this queue runs)
            queue.addResidencySet(set);
        }
        let shared = Arc::new(Shared {
            device: self.raw.clone(),
            queue: Mt(queue),
            heap: Mutex::new(heap),
            last_submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            in_flight: Mutex::new(VecDeque::new()),
            pending: Mutex::new(None),
            ring: Mutex::new(Ring::new()),
            fresh: Mutex::new(Vec::new()),
            zeros: Mutex::new(None),
            indirect: Mutex::new(None),
            info: self.info(),
            limits: t::Limits {
                max_bindless_textures: textures,
                max_bindless_samplers: samplers,
                max_bindless_buffers: buffers,
                ..limits
            }
            .capped(desc.limits.as_ref()),
            bc: self.supports_bc(),
            timestamp_period: crate::query::measure_timestamp_period(&self.raw),
            bound: self.is_bound_tier(),
        });
        if shared.bound {
            log::info!("mX: {} has no Metal 3; running on the bound tier", shared.info.name);
        }
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

/// Copies queued by `Queue::write_*`, run before the next submission's commands.
/// Staging memory for queue writes: 4 MiB blocks of shared memory written one after the
/// other and reused once the submission that read them is done (a new `MTLBuffer` per
/// write cost a system call and a page-zeroing each, every frame).
pub(crate) struct Ring {
    free: Vec<Mt<Po<dyn MTLBuffer>>>,
    /// The block being filled, and how far.
    current: Option<(Mt<Po<dyn MTLBuffer>>, usize)>,
    /// Blocks the pending writes read (the current one too, once written to).
    used: Vec<Mt<Po<dyn MTLBuffer>>>,
    in_flight: VecDeque<(u64, Vec<Block>)>,
}

type Block = Mt<Po<dyn MTLBuffer>>;

impl Ring {
    pub const BLOCK: usize = 4 << 20;
    /// Blocks kept for reuse; more are released.
    const KEEP: usize = 8;

    pub fn new() -> Ring {
        Ring {
            free: Vec::new(),
            current: None,
            used: Vec::new(),
            in_flight: VecDeque::new(),
        }
    }

    /// The pending writes go out as submission `index`: their blocks wait for it.
    fn retire(&mut self, index: u64) {
        if let Some((b, _)) = self.current.take() {
            self.used.push(b);
        }
        if !self.used.is_empty() {
            self.in_flight.push_back((index, std::mem::take(&mut self.used)));
        }
    }

    /// Submissions up to `completed` are done: their blocks are free again.
    fn reclaim(&mut self, completed: u64) {
        while let Some((i, _)) = self.in_flight.front()
            && *i <= completed
        {
            let (_, blocks) = self.in_flight.pop_front().unwrap();
            for b in blocks {
                if self.free.len() < Self::KEEP {
                    self.free.push(b);
                }
            }
        }
    }
}

pub(crate) struct PendingWrites {
    pub cmd: Mt<Po<dyn MTLCommandBuffer>>,
    pub blit: Mt<Po<dyn MTLBlitCommandEncoder>>,
    pub staging: Vec<Mt<Po<dyn MTLBuffer>>>,
    pub ended: bool,
}

impl Drop for PendingWrites {
    fn drop(&mut self) {
        // (writes never submitted: Metal requires an encoder to be ended before release)
        if !self.ended {
            self.blit.endEncoding();
        }
    }
}

/// Submitted command buffers not yet seen complete, with their submission index.
type InFlight = VecDeque<(u64, Mt<Po<dyn MTLCommandBuffer>>)>;

/// The kernels that encode GPU-driven draws: (draws, indexed draws).
pub(crate) type IndirectKernels = [Mt<Po<dyn MTLComputePipelineState>>; 2];

pub(crate) struct Shared {
    pub device: Mt<Po<dyn MTLDevice>>,
    pub queue: Mt<Po<dyn MTLCommandQueue>>,
    pub heap: Mutex<Heap>,
    pub last_submitted: AtomicU64,
    pub completed: AtomicU64,
    pub in_flight: Mutex<InFlight>,
    pub pending: Mutex<Option<PendingWrites>>,
    /// Staging memory of the queue's writes (used under the `pending` lock).
    pub ring: Mutex<Ring>,
    /// Textures made since the last submission (see `zero_new_textures`).
    pub fresh: Mutex<Vec<std::sync::Weak<crate::resource::TextureInner>>>,
    pub zeros: Mutex<Option<Mt<Po<dyn MTLBuffer>>>>,
    /// The kernels that encode GPU-driven draws into indirect command buffers: (draws,
    /// indexed draws), made on first use.
    pub indirect: Mutex<Option<IndirectKernels>>,
    pub info: t::AdapterInfo,
    pub limits: t::Limits,
    pub bc: bool,
    pub timestamp_period: f32,
    /// A Metal 2 GPU (no Metal 3): the heap is bound slot by slot at each draw (`bound tier`).
    pub bound: bool,
}

impl Shared {
    /// The submission that will be the next one: anything released now may be read by
    /// command buffers recorded before, which are submitted at the earliest then.
    pub fn release_after(&self) -> u64 {
        self.last_submitted.load(Ordering::Acquire) + 1
    }

    pub fn new_command_buffer(&self) -> Mt<Po<dyn MTLCommandBuffer>> {
        Mt(self.queue.commandBuffer().expect("Metal returned no command buffer"))
    }

    /// Run `f` on the pending-writes blit encoder (created on first use).
    pub fn with_pending<R>(&self, f: impl FnOnce(&mut PendingWrites) -> R) -> R {
        let mut p = self.pending.lock();
        let p = p.get_or_insert_with(|| {
            let cmd = self.new_command_buffer();
            cmd.setLabel(Some(&crate::ns("mX queue writes")));
            let blit = Mt(cmd.blitCommandEncoder().expect("Metal returned no blit encoder"));
            PendingWrites {
                cmd,
                blit,
                staging: Vec::new(),
                ended: false,
            }
        });
        f(p)
    }

    /// Copy `data` into staging memory the next submission reads: a buffer and the offset
    /// of the bytes in it (256-aligned, enough for any texel or block). Call it inside
    /// `with_pending`, so the bytes go out with the writes that read them.
    pub fn stage(&self, p: &mut PendingWrites, data: &[u8]) -> Result<(Mt<Po<dyn MTLBuffer>>, usize)> {
        if data.len() > Ring::BLOCK / 4 {
            // (a big write gets a buffer of its own, released with the command buffer)
            let b = self.staging_buffer(data)?;
            p.staging.push(b.clone());
            return Ok((b, 0));
        }
        let mut ring = self.ring.lock();
        let len = data.len().max(4);
        let fits = ring.current.as_ref().is_some_and(|(_, at)| at.next_multiple_of(256) + len <= Ring::BLOCK);
        if !fits {
            if let Some((full, _)) = ring.current.take() {
                ring.used.push(full);
            }
            let block = match ring.free.pop() {
                Some(b) => b,
                None => Mt(self
                    .device
                    .newBufferWithLength_options(Ring::BLOCK, MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined)
                    .ok_or_else(|| Error::OutOfMemory("staging memory".into()))?),
            };
            block.setLabel(Some(&crate::ns("mX staging")));
            ring.current = Some((block, 0));
        }
        let (block, at) = ring.current.as_mut().expect("a current block");
        let offset = at.next_multiple_of(256);
        // SAFETY: `offset + data.len()` is inside the block (checked above), whose bytes no
        // submitted work reads (blocks come back only once their submission is done).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), block.contents().as_ptr().cast::<u8>().add(offset), data.len()) };
        *at = offset + len;
        Ok((block.clone(), offset))
    }

    pub fn staging_buffer(&self, data: &[u8]) -> Result<Mt<Po<dyn MTLBuffer>>> {
        let len = data.len().max(4);
        let b = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined)
            .ok_or_else(|| Error::OutOfMemory("staging buffer".into()))?;
        // SAFETY: the new buffer is `len >= data.len()` bytes and not yet used by the GPU.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents().as_ptr().cast::<u8>(), data.len()) };
        Ok(Mt(b))
    }

    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) -> SubmissionIndex {
        let mut in_flight = self.in_flight.lock();
        let index = self.last_submitted.load(Ordering::Acquire) + 1;
        let mut last = None;
        // (the residency set's changes before anything that reads the heap)
        self.heap.lock().commit_residency();
        // (new textures are zeroed before the queue's writes land in them)
        if let Some(z) = self.zero_new_textures(index) {
            z.commit();
            last = Some(z);
        }
        let mut pending = self.pending.lock();
        if let Some(mut p) = pending.take() {
            p.blit.endEncoding();
            p.ended = true;
            p.cmd.commit();
            // (the staging buffers live as long as the command buffer: Metal retains them;
            // the ring's blocks wait for this submission)
            last = Some(p.cmd.clone());
            self.ring.lock().retire(index);
        }
        drop(pending);
        for b in buffers {
            if !b.written.is_empty() {
                let mut heap = self.heap.lock();
                for &owner in &b.written {
                    heap.written(owner, index);
                }
            }
            b.raw.commit();
            last = Some(b.raw);
        }
        let last = last.unwrap_or_else(|| {
            // an empty submission still gets an index the caller can wait for
            let cmd = self.new_command_buffer();
            cmd.commit();
            cmd
        });
        in_flight.push_back((index, last));
        self.last_submitted.store(index, Ordering::Release);
        SubmissionIndex(index)
    }

    /// Note finished submissions and free what they held. With `wait`, block until all
    /// submissions up to `wait` are done.
    pub fn poll(&self, wait: Option<u64>) -> Result<()> {
        let mut error = None;
        {
            let mut in_flight = self.in_flight.lock();
            if let Some(target) = wait
                && let Some((_, cmd)) = in_flight.iter().rev().find(|(i, _)| *i <= target)
            {
                // (the queue runs in order: when the last one up to `target` is done, all are)
                cmd.waitUntilCompleted();
            }
            while let Some((i, cmd)) = in_flight.front() {
                match cmd.status() {
                    MTLCommandBufferStatus::Completed => {}
                    MTLCommandBufferStatus::Error => {
                        error = Some(match cmd.error() {
                            // (MTLCommandBufferErrorDeviceRemoved: an eGPU unplugged, a GPU reset)
                            Some(e) if e.code() == 11 => Error::DeviceLost(e.localizedDescription().to_string()),
                            Some(e) => Error::Device(e.localizedDescription().to_string()),
                            None => Error::Device("a command buffer failed".into()),
                        });
                    }
                    _ => break,
                }
                self.completed.store(*i, Ordering::Release);
                in_flight.pop_front();
            }
        }
        self.heap.lock().reclaim(self.completed.load(Ordering::Acquire));
        self.ring.lock().reclaim(self.completed.load(Ordering::Acquire));
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub fn wait_idle(&self) -> Result<()> {
        self.poll(Some(self.last_submitted.load(Ordering::Acquire)))
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

    pub fn wait_idle(&self) -> Result<()> {
        self.shared.wait_idle()
    }

    /// Whether `index` has finished on the GPU (as of the last `poll` or `wait`).
    pub fn is_done(&self, index: SubmissionIndex) -> bool {
        self.shared.completed.load(std::sync::atomic::Ordering::Acquire) >= index.0
    }

    /// BC texture compression (as the adapter's `supports_bc`).
    pub fn supports_bc(&self) -> bool {
        self.shared.bc
    }

    /// Timestamp queries (as the adapter's `supports_timestamps`).
    pub fn supports_timestamps(&self) -> bool {
        crate::query::timestamp_counter_set(&self.shared.device).is_some()
    }

    /// Whether the device is known lost (a call on it said so; Direct3D 12 tells it at any time).
    pub fn is_lost(&self) -> bool {
        false
    }

    /// Texture views in the heap right now.
    pub fn bindless_textures_in_use(&self) -> usize {
        self.shared.heap.lock().textures_in_use()
    }

    /// Buffer views in the heap right now.
    pub fn bindless_buffers_in_use(&self) -> usize {
        self.shared.heap.lock().buffers_in_use()
    }

    /// The `MTLDevice`, for interop (MetalFX, capture).
    pub fn raw(&self) -> &objc2::runtime::ProtocolObject<dyn MTLDevice> {
        &self.shared.device
    }
}

/// Where command buffers go.
#[derive(Clone)]
pub struct Queue {
    pub(crate) shared: Arc<Shared>,
}

impl Queue {
    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) -> SubmissionIndex {
        self.shared.submit(buffers)
    }

    /// The `MTLCommandQueue`, for interop.
    pub fn raw(&self) -> &objc2::runtime::ProtocolObject<dyn MTLCommandQueue> {
        &self.shared.queue
    }
}

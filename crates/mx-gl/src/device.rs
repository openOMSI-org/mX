//! Instance, adapter, device and queue; submissions and their fences.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use glow::HasContext;
use mx_types as t;
use parking_lot::Mutex;

use crate::caps::Caps;
use crate::context::{Context, Library};
use crate::heap::Heap;
use crate::{Error, Result};

/// The entry point: EGL, loaded at run time.
#[derive(Clone)]
pub struct Instance {
    pub(crate) lib: Option<Arc<Library>>,
}

impl Default for Instance {
    fn default() -> Self {
        Self::new()
    }
}

impl Instance {
    pub fn new() -> Instance {
        Instance { lib: Library::load() }
    }

    /// The default display's GL device, if EGL is there and gives OpenGL ES 3.0 or OpenGL 3.3.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let Some(lib) = &self.lib else { return Vec::new() };
        match Context::new(lib.clone(), khronos_egl::DEFAULT_DISPLAY) {
            Ok(ctx) => {
                let caps = {
                    let gl = ctx.lock();
                    Caps::read(&gl, ctx.api)
                };
                if caps.version < (3, 0) || (ctx.api == crate::context::Api::Desktop && caps.version < (3, 3)) {
                    log::info!("mX: {} offers GL {:?}, below what mX needs", caps.name, caps.version);
                    return Vec::new();
                }
                vec![Adapter(Arc::new(AdapterInner { ctx: Arc::new(ctx), caps }))]
            }
            Err(e) => {
                log::info!("mX: no GL context: {e}");
                Vec::new()
            }
        }
    }

    pub fn default_adapter(&self) -> Result<Adapter> {
        self.enumerate_adapters().into_iter().next().ok_or(Error::NoDevice)
    }
}

pub(crate) struct AdapterInner {
    pub ctx: Arc<Context>,
    pub caps: Caps,
}

#[derive(Clone)]
pub struct Adapter(pub(crate) Arc<AdapterInner>);

impl Adapter {
    pub fn info(&self) -> t::AdapterInfo {
        let c = &self.0.caps;
        let lower = c.name.to_lowercase();
        let device_type = if lower.contains("llvmpipe") || lower.contains("softpipe") || lower.contains("swiftshader") || lower.contains("basic render driver") || lower.contains("warp") {
            t::DeviceType::Cpu
        } else if lower.contains("intel") || lower.contains("mali") || lower.contains("adreno") || lower.contains("powervr") {
            t::DeviceType::IntegratedGpu
        } else {
            t::DeviceType::Other
        };
        t::AdapterInfo {
            name: c.name.clone(),
            backend: t::Backend::Gl,
            device_type,
            vendor: 0,
            device: 0,
            driver: format!("{} ({})", c.driver, c.vendor),
        }
    }

    pub fn limits(&self) -> t::Limits {
        self.0.caps.limits
    }

    /// Not told by GL.
    pub fn memory_info(&self) -> Option<t::AdapterMemory> {
        None
    }

    /// Whether `format` renders and resolves with `samples` samples per pixel (`GL_MAX_SAMPLES`).
    pub fn supports_msaa(&self, format: t::TextureFormat, samples: u32) -> bool {
        let _ = format;
        samples <= 1 || samples <= self.0.caps.max_samples
    }

    /// Textures a pipeline may read through root bindings (on OpenGL, through anything: texture
    /// units): (vertex stage, fragment stage, the pipeline together).
    pub fn texture_units(&self) -> (u32, u32, u32) {
        (self.0.caps.vertex_texture_units, self.0.caps.fragment_texture_units, self.0.caps.combined_texture_units)
    }

    /// Indexed draws with a base vertex (not below OpenGL 3.2 / ES 3.2).
    pub fn supports_base_vertex(&self) -> bool {
        self.0.caps.base_vertex
    }

    pub fn supports_bc(&self) -> bool {
        self.0.caps.bc
    }

    /// Never: ray queries need bindless and acceleration structures.
    pub fn supports_ray_query(&self) -> bool {
        false
    }

    pub fn supports_timestamps(&self) -> bool {
        self.0.caps.timestamps
    }

    /// Whether this GPU runs on the bound tier: the heap bound slot by slot at each draw,
    /// heap accesses indexed by immediates or constants only.
    pub fn is_bound_tier(&self) -> bool {
        true
    }
    /// Never on this tier: GPU-driven batches need indirect command buffers or draw counts.
    pub fn supports_indirect_count(&self) -> bool {
        false
    }

    /// Compute shaders (GL 4.3, ES 3.1).
    pub fn supports_compute(&self) -> bool {
        self.0.caps.compute
    }

    /// How many storage buffers the vertex and fragment stages may read (0 below GL 4.3 / ES
    /// 3.1, and on drivers that offer them to compute only).
    pub fn storage_buffers_per_stage(&self) -> (u32, u32) {
        (self.0.caps.vertex_storage_blocks, self.0.caps.fragment_storage_blocks)
    }

    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Result<(Device, Queue)> {
        let a = &self.0;
        let caps = a.caps.clone();
        let (vao, fbo_draw, fbo_read) = {
            let gl = a.ctx.lock();
            // SAFETY: plain creations on the current context.
            unsafe {
                let vao = gl.create_vertex_array().map_err(|e| crate::gl_err("vertex array", e))?;
                let draw = gl.create_framebuffer().map_err(|e| crate::gl_err("framebuffer", e))?;
                let read = gl.create_framebuffer().map_err(|e| crate::gl_err("framebuffer", e))?;
                gl.bind_vertex_array(Some(vao));
                if a.ctx.api == crate::context::Api::Desktop {
                    // (as WebGPU: sRGB targets encode, cube maps sample across faces)
                    gl.enable(glow::FRAMEBUFFER_SRGB);
                    gl.enable(glow::TEXTURE_CUBE_MAP_SEAMLESS);
                    gl.enable(glow::PROGRAM_POINT_SIZE);
                }
                (vao, draw, read)
            }
        };
        let limits = t::Limits {
            max_bindless_textures: desc.bindless_textures.min(caps.limits.max_bindless_textures),
            max_bindless_samplers: desc.bindless_samplers.min(caps.limits.max_bindless_samplers),
            max_bindless_buffers: desc.bindless_buffers.min(caps.limits.max_bindless_buffers),
            ..caps.limits
        }
        .capped(desc.limits.as_ref());
        let shared = Arc::new(Shared {
            ctx: a.ctx.clone(),
            heap: Mutex::new(Heap::new(limits.max_bindless_textures, limits.max_bindless_samplers, limits.max_bindless_buffers)),
            last_submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            fences: Mutex::new(VecDeque::new()),
            garbage: Mutex::new(Vec::new()),
            info: Adapter(a.clone()).info(),
            limits,
            caps,
            _vao: vao,
            fbo_draw,
            fbo_read,
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

/// A GL object deleted once the GPU no longer uses it.
pub(crate) enum Garbage {
    Buffer(glow::Buffer),
    Texture(glow::Texture),
    Renderbuffer(glow::Renderbuffer),
    Sampler(glow::Sampler),
    Program(glow::Program),
    Query(glow::Query),
}

pub(crate) struct Shared {
    pub ctx: Arc<Context>,
    pub caps: Caps,
    pub heap: Mutex<Heap>,
    pub last_submitted: AtomicU64,
    pub completed: AtomicU64,
    /// The fence after each submission, oldest first.
    pub fences: Mutex<VecDeque<(u64, glow::Fence)>>,
    pub garbage: Mutex<Vec<(u64, Garbage)>>,
    pub info: t::AdapterInfo,
    pub limits: t::Limits,
    /// The one vertex array object every draw uses (bound once; its attributes are set per
    /// draw).
    pub _vao: glow::VertexArray,
    /// Framebuffers whose attachments passes, copies and resolves set.
    pub fbo_draw: glow::Framebuffer,
    pub fbo_read: glow::Framebuffer,
}

// SAFETY: GL objects are plain names, used only under the context's lock.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    /// The submission that will be the next one: anything released now may be read by
    /// command buffers recorded before, which are submitted at the earliest then.
    pub fn release_after(&self) -> u64 {
        self.last_submitted.load(Ordering::Acquire) + 1
    }

    pub fn retire(&self, g: Garbage) {
        self.garbage.lock().push((self.release_after(), g));
    }

    /// Note finished submissions (waiting up to `wait_for`'s completion), free what they held.
    pub fn poll(&self, wait_for: Option<u64>) -> Result<()> {
        let gl = self.ctx.lock();
        let mut fences = self.fences.lock();
        let mut completed = self.completed.load(Ordering::Acquire);
        while let Some(&(index, fence)) = fences.front() {
            let must = wait_for.is_some_and(|w| index <= w);
            // SAFETY: a fence this device made, not deleted yet.
            let status = unsafe {
                if must {
                    loop {
                        let s = gl.client_wait_sync(fence, glow::SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000);
                        if s != glow::TIMEOUT_EXPIRED {
                            break s;
                        }
                    }
                } else {
                    gl.client_wait_sync(fence, 0, 0)
                }
            };
            match status {
                glow::ALREADY_SIGNALED | glow::CONDITION_SATISFIED => {
                    // SAFETY: signalled, no longer waited on.
                    unsafe { gl.delete_sync(fence) };
                    fences.pop_front();
                    completed = index;
                }
                glow::WAIT_FAILED => return Err(Error::Device("glClientWaitSync failed".into())),
                _ => break,
            }
        }
        if fences.is_empty() {
            completed = self.last_submitted.load(Ordering::Acquire);
        }
        drop(fences);
        self.completed.store(completed, Ordering::Release);
        self.heap.lock().reclaim(completed);
        let done: Vec<Garbage> = {
            let mut g = self.garbage.lock();
            let (done, keep): (Vec<_>, Vec<_>) = g.drain(..).partition(|(after, _)| *after <= completed);
            *g = keep;
            done.into_iter().map(|(_, x)| x).collect()
        };
        // SAFETY: objects no submission uses any more.
        unsafe {
            for g in done {
                match g {
                    Garbage::Buffer(b) => gl.delete_buffer(b),
                    Garbage::Texture(t) => gl.delete_texture(t),
                    Garbage::Renderbuffer(r) => gl.delete_renderbuffer(r),
                    Garbage::Sampler(s) => gl.delete_sampler(s),
                    Garbage::Program(p) => gl.delete_program(p),
                    Garbage::Query(q) => gl.delete_query(q),
                }
            }
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

    /// BC texture compression (as the adapter's `supports_bc`).
    pub fn supports_bc(&self) -> bool {
        self.shared.caps.bc
    }

    /// Timestamp queries (as the adapter's `supports_timestamps`).
    pub fn supports_timestamps(&self) -> bool {
        self.shared.caps.timestamps
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

    /// Run `f` with the GL context current, for interop.
    pub fn with_gl<R>(&self, f: impl FnOnce(&glow::Context) -> R) -> R {
        let gl = self.shared.ctx.lock();
        f(&gl)
    }
}

/// Where command buffers go.
#[derive(Clone)]
pub struct Queue {
    pub(crate) shared: Arc<Shared>,
}

impl Queue {
    /// Replay the command buffers on the context, in order, then fence.
    pub fn submit(&self, buffers: impl IntoIterator<Item = crate::CommandBuffer>) -> SubmissionIndex {
        let s = &self.shared;
        let gl = s.ctx.lock();
        for b in buffers {
            crate::command::execute(&gl, s, b);
        }
        let index = s.last_submitted.fetch_add(1, Ordering::AcqRel) + 1;
        // SAFETY: a fence on the current context.
        match unsafe { gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) } {
            Ok(f) => s.fences.lock().push_back((index, f)),
            Err(e) => log::error!("mX: glFenceSync failed ({e}); submission {index} is waited for at once"),
        }
        // SAFETY: as above.
        unsafe { gl.flush() };
        SubmissionIndex(index)
    }

    /// Nanoseconds per timestamp tick: GL timestamps are in nanoseconds.
    pub fn get_timestamp_period(&self) -> f32 {
        1.0
    }
}

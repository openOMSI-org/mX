//! Window surfaces: a DXGI flip-model swapchain on a Win32 window.

use std::marker::PhantomData;
use std::sync::Arc;

use mx_types as t;
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, WAIT_TIMEOUT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::WaitForSingleObject;
use windows::core::Interface;

use crate::device::Shared;
use crate::{Adapter, Device, Error, Instance, Result, Texture, conv};

trait Window: HasWindowHandle + HasDisplayHandle {}
impl<W: HasWindowHandle + HasDisplayHandle> Window for W {}

/// A Win32 handle used from any thread (window handles and waitable objects are).
#[derive(Clone, Copy)]
struct Handle(HANDLE);
// SAFETY: kernel handles may be waited on from any thread.
unsafe impl Send for Handle {}
// SAFETY: as above.
unsafe impl Sync for Handle {}

#[derive(Clone, Copy)]
struct Hwnd(HWND);
// SAFETY: the window handle is only passed to DXGI, which takes it from any thread.
unsafe impl Send for Hwnd {}
// SAFETY: as above.
unsafe impl Sync for Hwnd {}

/// What a swapchain maker is given: a vendor's frame generation (DLSS-G, FSR 3, XeSS-FG)
/// makes the window's swapchain itself, from these, and hands back its own.
pub struct SwapchainRequest<'a> {
    pub factory: &'a IDXGIFactory4,
    pub queue: &'a ID3D12CommandQueue,
    pub hwnd: HWND,
    pub desc: &'a DXGI_SWAP_CHAIN_DESC1,
}

/// Makes a surface's swapchain in place of `IDXGIFactory2::CreateSwapChainForHwnd`
/// (`Surface::set_swapchain_maker`).
pub type SwapchainMaker = Arc<dyn Fn(&SwapchainRequest<'_>) -> windows::core::Result<IDXGISwapChain1> + Send + Sync>;

struct Swap {
    raw: IDXGISwapChain3,
    /// (None: the swapchain gives none, as some vendors' do)
    waitable: Option<Handle>,
    flags: DXGI_SWAP_CHAIN_FLAG,
    shared: Arc<Shared>,
    config: t::SurfaceConfiguration,
}

impl Drop for Swap {
    fn drop(&mut self) {
        // (the queue must be done with the buffers before the swapchain goes)
        let _ = self.shared.wait_idle();
        // SAFETY: the swapchain's own waitable object, closed once.
        if let Some(w) = self.waitable {
            unsafe {
                let _ = CloseHandle(w.0);
            }
        }
    }
}

/// Where a window's frames go.
pub struct Surface<'w> {
    factory: IDXGIFactory4,
    hwnd: Hwnd,
    allow_tearing: bool,
    swap: Mutex<Option<Swap>>,
    maker: Mutex<Option<SwapchainMaker>>,
    _window: Option<Box<dyn Window + Send + Sync + 'w>>,
    _lifetime: PhantomData<&'w ()>,
}

impl Instance {
    /// A surface for a window (a winit `Window`, `Arc<Window>` or `&Window`).
    pub fn create_surface<'w>(&self, window: impl HasWindowHandle + HasDisplayHandle + Send + Sync + 'w) -> Result<Surface<'w>> {
        let raw_window = window.window_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        let raw_display = window.display_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        // SAFETY: the handles are valid while `window` lives, and the surface keeps it.
        let s = unsafe { self.create_surface_unsafe(raw_display, raw_window)? };
        Ok(Surface {
            factory: s.factory.clone(),
            hwnd: s.hwnd,
            allow_tearing: s.allow_tearing,
            swap: Mutex::new(None),
            maker: Mutex::new(None),
            _window: Some(Box::new(window)),
            _lifetime: PhantomData,
        })
    }

    /// A surface from raw handles.
    ///
    /// # Safety
    /// The window must outlive the surface.
    pub unsafe fn create_surface_unsafe(&self, _display: RawDisplayHandle, window: RawWindowHandle) -> Result<Surface<'static>> {
        let factory = self.factory.clone().ok_or_else(|| Error::Surface("no DXGI factory".into()))?;
        match window {
            RawWindowHandle::Win32(h) => Ok(Surface {
                factory,
                hwnd: Hwnd(HWND(h.hwnd.get() as *mut _)),
                allow_tearing: self.allow_tearing,
                swap: Mutex::new(None),
                maker: Mutex::new(None),
                _window: None,
                _lifetime: PhantomData,
            }),
            other => Err(Error::Surface(format!("not a window Direct3D 12 can draw into: {other:?}"))),
        }
    }
}

fn sync(mode: t::PresentMode, tearing: bool) -> (u32, DXGI_PRESENT) {
    match mode {
        t::PresentMode::Immediate | t::PresentMode::AutoNoVsync if tearing => (0, DXGI_PRESENT_ALLOW_TEARING),
        t::PresentMode::Immediate | t::PresentMode::AutoNoVsync | t::PresentMode::Mailbox => (0, DXGI_PRESENT(0)),
        t::PresentMode::Fifo | t::PresentMode::FifoRelaxed | t::PresentMode::AutoVsync => (1, DXGI_PRESENT(0)),
    }
}

impl Surface<'_> {
    pub fn get_capabilities(&self, _adapter: &Adapter) -> t::SurfaceCapabilities {
        let mut present_modes = vec![t::PresentMode::Fifo, t::PresentMode::AutoVsync, t::PresentMode::Mailbox, t::PresentMode::AutoNoVsync];
        if self.allow_tearing {
            present_modes.push(t::PresentMode::Immediate);
        }
        t::SurfaceCapabilities {
            formats: vec![
                t::TextureFormat::Bgra8UnormSrgb,
                t::TextureFormat::Bgra8Unorm,
                t::TextureFormat::Rgba8UnormSrgb,
                t::TextureFormat::Rgba8Unorm,
                t::TextureFormat::Rgb10a2Unorm,
                t::TextureFormat::Rgba16Float,
            ],
            present_modes,
            alpha_modes: vec![t::CompositeAlphaMode::Opaque],
            // (not TEXTURE_BINDING: a swapchain buffer rests in PRESENT, not where the heap
            // could read it)
            usages: t::TextureUsages::RENDER_ATTACHMENT | t::TextureUsages::COPY_SRC | t::TextureUsages::COPY_DST,
        }
    }

    pub fn configure(&self, device: &Device, config: &t::SurfaceConfiguration) {
        let mut guard = self.swap.lock();
        let shared = &device.shared;
        let buffers = (config.desired_maximum_frame_latency + 1).clamp(2, 16);
        let (w, h) = (config.width.max(1), config.height.max(1));
        let format = conv::format_nosrgb(config.format);
        if let Some(s) = guard.as_mut()
            && Arc::ptr_eq(&s.shared, shared)
        {
            shared.wait_idle().expect("the GPU failed before a swapchain resize");
            // SAFETY: the GPU is idle and the frames handed out were presented (dropped).
            unsafe { s.raw.ResizeBuffers(buffers, w, h, format, s.flags) }
                .unwrap_or_else(|e| crate::dred::fail("IDXGISwapChain::ResizeBuffers failed (are textures of the old frames still held?)", e));
            // SAFETY: a plain setter.
            let _ = unsafe { s.raw.SetMaximumFrameLatency(config.desired_maximum_frame_latency.max(1)) };
            s.config = config.clone();
            return;
        }
        *guard = None;
        let mut flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        if self.allow_tearing {
            flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        }
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: w,
            Height: h,
            Format: format,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: buffers,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_IGNORE,
            Flags: flags.0 as u32,
        };
        let maker = self.maker.lock().clone();
        // SAFETY: a live window (the surface's contract) and the device's direct queue.
        let raw = unsafe {
            let sc1 = match maker {
                Some(make) => make(&SwapchainRequest {
                    factory: &self.factory,
                    queue: &shared.queue,
                    hwnd: self.hwnd.0,
                    desc: &desc,
                })
                .expect("the surface's swapchain maker failed"),
                None => self
                    .factory
                    .CreateSwapChainForHwnd(&shared.queue, self.hwnd.0, &desc, None, None)
                    .expect("IDXGIFactory2::CreateSwapChainForHwnd failed"),
            };
            let _ = self.factory.MakeWindowAssociation(self.hwnd.0, DXGI_MWA_NO_ALT_ENTER);
            sc1.cast::<IDXGISwapChain3>().expect("no IDXGISwapChain3")
        };
        // SAFETY: plain setters / getters on the new swapchain.
        let waitable = unsafe {
            let _ = raw.SetMaximumFrameLatency(config.desired_maximum_frame_latency.max(1));
            raw.GetFrameLatencyWaitableObject()
        };
        let waitable = (!waitable.is_invalid() && !waitable.0.is_null()).then_some(Handle(waitable));
        // (a vendor's swapchain may be made with other flags than asked: DLSS-G drops the latency
        // object; `ResizeBuffers` must be given the swapchain's own)
        // SAFETY: a plain getter.
        let flags = unsafe { raw.GetDesc1() }.map_or(flags, |d| DXGI_SWAP_CHAIN_FLAG(d.Flags as i32));
        *guard = Some(Swap {
            raw,
            waitable,
            flags,
            shared: shared.clone(),
            config: config.clone(),
        });
    }

    pub fn get_current_texture(&self) -> t::CurrentSurfaceTexture<SurfaceTexture> {
        let guard = self.swap.lock();
        let Some(s) = guard.as_ref() else {
            panic!("get_current_texture on a surface that was never configured");
        };
        // SAFETY: the swapchain's waitable object.
        if let Some(w) = s.waitable
            && unsafe { WaitForSingleObject(w.0, 1000) } == WAIT_TIMEOUT
        {
            return t::CurrentSurfaceTexture::Timeout;
        }
        // SAFETY: plain getters on a live swapchain.
        let raw: ID3D12Resource = match unsafe { s.raw.GetBuffer(s.raw.GetCurrentBackBufferIndex()) } {
            Ok(r) => r,
            Err(_) => return t::CurrentSurfaceTexture::Lost,
        };
        let c = &s.config;
        let desc = t::TextureDescriptor {
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
            usage: c.usage - t::TextureUsages::TEXTURE_BINDING,
            view_formats: &c.view_formats,
        };
        let texture = Texture::from_raw_parts(raw, &desc, D3D12_RESOURCE_STATE_PRESENT, s.shared.clone());
        let (interval, flags) = sync(c.present_mode, self.allow_tearing);
        t::CurrentSurfaceTexture::Success(SurfaceTexture {
            texture,
            swap: s.raw.clone(),
            interval,
            flags,
        })
    }

    /// Have `maker` make the swapchain from the next `configure` on (None: DXGI's own again).
    /// The current swapchain is let go of, once the GPU is done with it: configure the surface
    /// again before the next frame.
    pub fn set_swapchain_maker(&self, maker: Option<SwapchainMaker>) {
        *self.maker.lock() = maker;
        *self.swap.lock() = None;
    }

    /// The configuration of the current swapchain.
    pub fn config(&self) -> Option<t::SurfaceConfiguration> {
        self.swap.lock().as_ref().map(|s| s.config.clone())
    }

    /// The `IDXGISwapChain3`, once configured, for interop.
    pub fn raw(&self) -> Option<IDXGISwapChain3> {
        self.swap.lock().as_ref().map(|s| s.raw.clone())
    }
}

/// A frame of a surface. Draw into `texture`, submit, then `present`.
pub struct SurfaceTexture {
    pub texture: Texture,
    swap: IDXGISwapChain3,
    interval: u32,
    flags: DXGI_PRESENT,
}

impl SurfaceTexture {
    /// Show the frame once the work submitted so far is done (the queue orders it).
    pub fn present(self) {
        let SurfaceTexture { texture, swap, interval, flags } = self;
        drop(texture);
        // SAFETY: the swapchain is alive; the buffer is back in PRESENT (its home state).
        let hr = unsafe { swap.Present(interval, flags) };
        if hr.is_err() {
            log::warn!("mX: IDXGISwapChain::Present: {hr:?}");
        }
    }
}

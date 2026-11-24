//! The EGL display and the device's one GL context, behind a lock.
//!
//! Each operation takes the lock, which makes the context current on the calling thread with
//! the device's 1×1 pbuffer (or a window's surface, to present), and releases it after: mX's
//! objects are `Send + Sync`, GL's context is not.

use std::ffi::c_void;
use std::sync::Arc;

use khronos_egl as egl;
use parking_lot::{Mutex, MutexGuard};

use crate::{Error, Result};

pub(crate) type Egl = egl::DynamicInstance<egl::EGL1_4>;

/// The EGL library, loaded once.
pub(crate) struct Library {
    pub egl: Egl,
}

// SAFETY: EGL entry points are thread-safe; the library is only read.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

/// The names the EGL library goes by on each platform. On Windows this is ANGLE's (shipped
/// next to the game), over Direct3D 11.
fn library_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["libEGL.dll"]
    } else if cfg!(target_vendor = "apple") {
        &["libEGL.dylib"]
    } else if cfg!(target_os = "android") {
        &["libEGL.so"]
    } else {
        &["libEGL.so.1", "libEGL.so"]
    }
}

impl Library {
    pub fn load() -> Option<Arc<Library>> {
        // (`MX_EGL_LIBRARY`: a path to try first, such as ANGLE's libEGL in a test's folder)
        let chosen = std::env::var("MX_EGL_LIBRARY").ok();
        for name in chosen.iter().map(String::as_str).chain(library_names().iter().copied()) {
            // SAFETY: loading the system's (or ANGLE's) EGL; its entry points are what the
            // EGL specification says they are.
            match unsafe { Egl::load_required_from_filename(name) } {
                Ok(egl) => return Some(Arc::new(Library { egl })),
                Err(e) => log::debug!("mX: no EGL as {name}: {e}"),
            }
        }
        None
    }
}

const EGL_PLATFORM_ANGLE_ANGLE: egl::Enum = 0x3202;
const EGL_PLATFORM_ANGLE_TYPE_ANGLE: egl::Attrib = 0x3203;
const EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE: egl::Attrib = 0x3208;
const EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE: egl::Attrib = 0x3209;
const EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE: egl::Attrib = 0x322A;
const EGL_PLATFORM_ANGLE_DEVICE_TYPE_D3D_WARP_ANGLE: egl::Attrib = 0x320B;

/// `MX_ANGLE_DEVICE=warp|hardware` (Windows, ANGLE's EGL): ANGLE's Direct3D 11 renderer on
/// that device - WARP, Microsoft's software renderer, for tests on machines without a GPU.
fn angle_device() -> Option<egl::Attrib> {
    if !cfg!(windows) {
        return None;
    }
    match std::env::var("MX_ANGLE_DEVICE").ok()?.to_ascii_lowercase().as_str() {
        "warp" => Some(EGL_PLATFORM_ANGLE_DEVICE_TYPE_D3D_WARP_ANGLE),
        "hardware" => Some(EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE),
        _ => None,
    }
}

/// Which GL a context runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Api {
    Es,
    Desktop,
}

pub(crate) struct Inner {
    pub gl: glow::Context,
    pub surface: Option<egl::Surface>,
}

/// One device's EGL display, config and context.
pub(crate) struct Context {
    pub lib: Arc<Library>,
    pub display: egl::Display,
    pub config: egl::Config,
    pub context: egl::Context,
    pub pbuffer: egl::Surface,
    pub api: Api,
    /// `OES_draw_buffers_indexed`'s entry points, for ES 3.0 and 3.1 (see [`IndexedOes`]).
    pub indexed_oes: Option<IndexedOes>,
    pub inner: Mutex<Inner>,
}

/// Blending and colour masks per draw buffer under `OES_draw_buffers_indexed`'s names. glow
/// loads only the ES 3.2 names (`glEnablei`), which ANGLE's ES 3.0 refuses although it has the
/// extension.
#[derive(Clone, Copy)]
pub(crate) struct IndexedOes {
    pub enable: unsafe extern "system" fn(u32, u32),
    pub disable: unsafe extern "system" fn(u32, u32),
    pub blend_func_separate: unsafe extern "system" fn(u32, u32, u32, u32, u32),
    pub blend_equation_separate: unsafe extern "system" fn(u32, u32, u32),
    pub color_mask: unsafe extern "system" fn(u32, u8, u8, u8, u8),
}

impl IndexedOes {
    fn load(e: &Egl) -> Option<IndexedOes> {
        let f = |name: &str| e.get_proc_address(name);
        // SAFETY: the extension's entry points have these signatures (OES_draw_buffers_indexed).
        unsafe {
            Some(IndexedOes {
                enable: std::mem::transmute::<extern "system" fn(), unsafe extern "system" fn(u32, u32)>(f("glEnableiOES")?),
                disable: std::mem::transmute::<extern "system" fn(), unsafe extern "system" fn(u32, u32)>(f("glDisableiOES")?),
                blend_func_separate: std::mem::transmute::<extern "system" fn(), unsafe extern "system" fn(u32, u32, u32, u32, u32)>(f("glBlendFuncSeparateiOES")?),
                blend_equation_separate: std::mem::transmute::<extern "system" fn(), unsafe extern "system" fn(u32, u32, u32)>(f("glBlendEquationSeparateiOES")?),
                color_mask: std::mem::transmute::<extern "system" fn(), unsafe extern "system" fn(u32, u8, u8, u8, u8)>(f("glColorMaskiOES")?),
            })
        }
    }
}

// SAFETY: the context is only used under `inner`'s lock, made current by the locker.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

/// The context, current on this thread until dropped.
pub(crate) struct Current<'a> {
    ctx: &'a Context,
    guard: MutexGuard<'a, Inner>,
}

impl std::ops::Deref for Current<'_> {
    type Target = glow::Context;
    fn deref(&self) -> &glow::Context {
        &self.guard.gl
    }
}

impl Current<'_> {
    /// Draw to `surface` (a window's) until dropped or changed again; `None`: the pbuffer.
    pub fn set_surface(&mut self, surface: Option<egl::Surface>) -> Result<()> {
        let s = surface.unwrap_or(self.ctx.pbuffer);
        self.ctx
            .lib
            .egl
            .make_current(self.ctx.display, Some(s), Some(s), Some(self.ctx.context))
            .map_err(|e| Error::Device(format!("eglMakeCurrent: {e}")))?;
        self.guard.surface = surface;
        Ok(())
    }

    pub fn ctx(&self) -> &Context {
        self.ctx
    }
}

impl Drop for Current<'_> {
    fn drop(&mut self) {
        // (another thread may take the context next)
        let _ = self.ctx.lib.egl.make_current(self.ctx.display, None, None, None);
    }
}

impl Context {
    /// Take the context and make it current on this thread.
    pub fn lock(&self) -> Current<'_> {
        let guard = self.inner.lock();
        let s = guard.surface.unwrap_or(self.pbuffer);
        if let Err(e) = self.lib.egl.make_current(self.display, Some(s), Some(s), Some(self.context)) {
            log::error!("mX: eglMakeCurrent failed: {e}");
        }
        Current { ctx: self, guard }
    }

    /// The default display's context: OpenGL ES 3.2 down to 3.0, or desktop OpenGL 4.6 down to
    /// 3.3 core where the driver has no ES (`MX_GL_DESKTOP` asks for desktop first).
    pub fn new(lib: Arc<Library>, native_display: *mut c_void) -> Result<Context> {
        let e = &lib.egl;
        // SAFETY: the native display handle comes from the window system (or is the default).
        let display = match angle_device() {
            // SAFETY: ANGLE's platform display over the default device, as its docs make it.
            Some(device) => {
                let e15 = e.upcast::<egl::EGL1_5>().ok_or(Error::NoDevice)?;
                unsafe {
                    e15.get_platform_display(
                        EGL_PLATFORM_ANGLE_ANGLE,
                        native_display,
                        &[
                            EGL_PLATFORM_ANGLE_TYPE_ANGLE,
                            EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE,
                            EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE,
                            device,
                            egl::ATTRIB_NONE,
                        ],
                    )
                }
                .map_err(|_| Error::NoDevice)?
            }
            // SAFETY: the native display handle comes from the window system (or is the default).
            None => unsafe { e.get_display(native_display) }.ok_or(Error::NoDevice)?,
        };
        e.initialize(display).map_err(|_| Error::NoDevice)?;
        let desktop_first = std::env::var_os("MX_GL_DESKTOP").is_some();
        let apis = if desktop_first { [Api::Desktop, Api::Es] } else { [Api::Es, Api::Desktop] };
        for api in apis {
            if let Some(ctx) = Self::with_api(&lib, display, api) {
                return Ok(ctx);
            }
        }
        Err(Error::NoDevice)
    }

    fn with_api(lib: &Arc<Library>, display: egl::Display, api: Api) -> Option<Context> {
        let e = &lib.egl;
        let (bind, renderable) = match api {
            Api::Es => (egl::OPENGL_ES_API, egl::OPENGL_ES3_BIT),
            Api::Desktop => (egl::OPENGL_API, egl::OPENGL_BIT),
        };
        e.bind_api(bind).ok()?;
        let attribs = [
            egl::SURFACE_TYPE,
            egl::PBUFFER_BIT | egl::WINDOW_BIT,
            egl::RENDERABLE_TYPE,
            renderable,
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::ALPHA_SIZE,
            8,
            egl::NONE,
        ];
        let config = e.choose_first_config(display, &attribs).ok()??;
        let versions: &[(i32, i32)] = match api {
            Api::Es => &[(3, 2), (3, 1), (3, 0)],
            Api::Desktop => &[(4, 6), (4, 5), (4, 3), (4, 1), (3, 3)],
        };
        const CONTEXT_MAJOR_VERSION: i32 = 0x3098;
        const CONTEXT_MINOR_VERSION: i32 = 0x30FB;
        const CONTEXT_OPENGL_PROFILE_MASK: i32 = 0x30FD;
        const CONTEXT_OPENGL_CORE_PROFILE_BIT: i32 = 0x1;
        let context = versions.iter().find_map(|&(major, minor)| {
            let mut a = vec![CONTEXT_MAJOR_VERSION, major, CONTEXT_MINOR_VERSION, minor];
            if api == Api::Desktop {
                a.extend([CONTEXT_OPENGL_PROFILE_MASK, CONTEXT_OPENGL_CORE_PROFILE_BIT]);
            }
            a.push(egl::NONE);
            e.create_context(display, config, None, &a).ok()
        })?;
        let pbuffer = e.create_pbuffer_surface(display, config, &[egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE]).ok()?;
        e.make_current(display, Some(pbuffer), Some(pbuffer), Some(context)).ok()?;
        // SAFETY: the context is current; the loader returns GL's entry points by name.
        let gl = unsafe { glow::Context::from_loader_function(|name| e.get_proc_address(name).map_or(std::ptr::null(), |p| p as *const c_void)) };
        let _ = e.make_current(display, None, None, None);
        Some(Context {
            lib: lib.clone(),
            display,
            config,
            context,
            pbuffer,
            api,
            indexed_oes: if api == Api::Es { IndexedOes::load(e) } else { None },
            inner: Mutex::new(Inner { gl, surface: None }),
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let e = &self.lib.egl;
        let _ = e.make_current(self.display, None, None, None);
        let _ = e.destroy_surface(self.display, self.pbuffer);
        let _ = e.destroy_context(self.display, self.context);
    }
}

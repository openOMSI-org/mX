//! mX: openOMSI's bindless graphics library.
//!
//! The API follows wgpu's names and shapes where the concept is the same. What differs:
//!
//! - **No layouts.** Pipelines have no pipeline or bind group layout; what a shader reads is
//!   reflected from it.
//! - **Bindless heap.** Every texture view made with `TEXTURE_BINDING`, every sampler and every
//!   buffer view ([`Buffer::create_view`]) has an index ([`TextureView::index`],
//!   [`Sampler::index`], [`BufferView::index`]). Shaders read them as `mx_tex2d[i]`,
//!   `mx_samplers[j]`, `mx_buffers[k]` or their own typed buffer views (see [`PRELUDE`] and
//!   `mx_shader`), usually with the indices passed as immediates
//!   ([`RenderPass::set_immediates`], WGSL `var<immediate>`). A draw needs no bind group.
//! - **Root bind groups** (WGSL groups `0..=6`) stay for code being ported from wgpu; group 7
//!   is the heap.
//!
//! # Backends
//!
//! One build holds every backend of its platform, and the game picks one when it opens a
//! device, as a graphics API setting: Metal on Apple platforms; Direct3D 12, Vulkan and OpenGL
//! (through ANGLE) on Windows; Vulkan and OpenGL on Linux and Android. (The `vulkan` and `gl`
//! features, on by default, add those two; on Apple platforms Vulkan runs on MoltenVK.) Each
//! object is the backend's own behind an enum, and every call is one `match` to the backend
//! that made it: no dynamic dispatch, no validation layer. [`Instance::with_backends`] (or
//! `MX_BACKENDS=vulkan,gl`) chooses which backends are opened; [`Adapter::backend`] tells which
//! one an adapter is. (The no-op backend - a device that draws nothing, for dedicated servers
//! and tests - is opened only when asked for: `Backends::NOOP`, `MX_BACKENDS=noop`.)
//! `as_metal()`, `as_vulkan()`, `as_dx12()` and `as_gl()` reach the
//! backend's object for interop (MetalFX, DLSS, FSR 3, OpenXR).

/// The shader side (WGSL with the heap prelude, the bound tier's lowering, GLSL), for tools
/// and tests that check shaders as mX compiles them.
pub use mx_shader;
pub use mx_shader::{HEAP_GROUP, MAX_ROOT_GROUP, PRELUDE};
pub(crate) use mx_types as t;
pub use mx_types::*;

#[cfg(windows)]
pub use mx_dx11;
#[cfg(windows)]
pub use mx_dx12;
#[cfg(feature = "gl")]
pub use mx_gl;
#[cfg(target_vendor = "apple")]
pub use mx_metal;
#[cfg(feature = "noop")]
pub use mx_noop;
#[cfg(feature = "vulkan")]
pub use mx_vulkan;

/// Match an mX object to its backend's: `on!(Type, value, x => expr)`, the same expression in
/// every arm (typed per backend; see [`Pick`]).
macro_rules! on {
    ($ty:ident, $v:expr, $x:ident => $e:expr) => {
        match $v {
            #[cfg(target_vendor = "apple")]
            $ty::Metal($x) => $e,
            #[cfg(windows)]
            $ty::Dx12($x) => $e,
            #[cfg(feature = "vulkan")]
            $ty::Vulkan($x) => $e,
            #[cfg(feature = "gl")]
            $ty::Gl($x) => $e,
            #[cfg(windows)]
            $ty::Dx11($x) => $e,
            #[cfg(feature = "noop")]
            $ty::Noop($x) => $e,
        }
    };
}

/// As [`on!`], wrapping the result in `Out`'s variant of the same backend.
macro_rules! wrap {
    ($ty:ident, $out:ident, $v:expr, $x:ident => $e:expr) => {
        match $v {
            #[cfg(target_vendor = "apple")]
            $ty::Metal($x) => $out::Metal($e),
            #[cfg(windows)]
            $ty::Dx12($x) => $out::Dx12($e),
            #[cfg(feature = "vulkan")]
            $ty::Vulkan($x) => $out::Vulkan($e),
            #[cfg(feature = "gl")]
            $ty::Gl($x) => $out::Gl($e),
            #[cfg(windows)]
            $ty::Dx11($x) => $out::Dx11($e),
            #[cfg(feature = "noop")]
            $ty::Noop($x) => $out::Noop($e),
        }
    };
}

/// As [`wrap!`] for a backend `Result`: `Ok(Out::Variant(..))` or the error as mX's.
macro_rules! wrap_res {
    ($ty:ident, $out:ident, $v:expr, $x:ident => $e:expr) => {
        match $v {
            #[cfg(target_vendor = "apple")]
            $ty::Metal($x) => $e.map($out::Metal).map_err(Error::from),
            #[cfg(windows)]
            $ty::Dx12($x) => $e.map($out::Dx12).map_err(Error::from),
            #[cfg(feature = "vulkan")]
            $ty::Vulkan($x) => $e.map($out::Vulkan).map_err(Error::from),
            #[cfg(feature = "gl")]
            $ty::Gl($x) => $e.map($out::Gl).map_err(Error::from),
            #[cfg(windows)]
            $ty::Dx11($x) => $e.map($out::Dx11).map_err(Error::from),
            #[cfg(feature = "noop")]
            $ty::Noop($x) => $e.map($out::Noop).map_err(Error::from),
        }
    };
}

/// An enum of one type over the backends, with [`Pick`] and `as_*` accessors.
macro_rules! def_enum {
    ($(#[$a:meta])* $name:ident) => {
        $(#[$a])*
        // (the backends' objects inline: a box would cost an indirection on every call)
        #[allow(clippy::large_enum_variant)]
        pub enum $name {
            #[cfg(target_vendor = "apple")]
            Metal(mx_metal::$name),
            #[cfg(windows)]
            Dx12(mx_dx12::$name),
            #[cfg(feature = "vulkan")]
            Vulkan(mx_vulkan::$name),
            #[cfg(feature = "gl")]
            Gl(mx_gl::$name),
            #[cfg(windows)]
            Dx11(mx_dx11::$name),
            #[cfg(feature = "noop")]
            Noop(mx_noop::$name),
        }

        // (names the object and its backend: the backends' objects have no `Debug` of their own)
        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!("mx::", stringify!($name), "({:?})"), self.backend())
            }
        }

        #[allow(unreachable_patterns)]
        impl $name {
            /// The backend that made it.
            pub fn backend(&self) -> $crate::Backend {
                match self {
                    #[cfg(target_vendor = "apple")]
                    Self::Metal(_) => $crate::Backend::Metal,
                    #[cfg(windows)]
                    Self::Dx12(_) => $crate::Backend::Dx12,
                    #[cfg(feature = "vulkan")]
                    Self::Vulkan(_) => $crate::Backend::Vulkan,
                    #[cfg(feature = "gl")]
                    Self::Gl(_) => $crate::Backend::Gl,
                    #[cfg(windows)]
                    Self::Dx11(_) => $crate::Backend::Dx11,
                    #[cfg(feature = "noop")]
                    Self::Noop(_) => $crate::Backend::Noop,
                }
            }
            #[cfg(target_vendor = "apple")]
            pub fn as_metal(&self) -> Option<&mx_metal::$name> {
                match self {
                    Self::Metal(x) => Some(x),
                    _ => None,
                }
            }
            #[cfg(windows)]
            pub fn as_dx12(&self) -> Option<&mx_dx12::$name> {
                match self {
                    Self::Dx12(x) => Some(x),
                    _ => None,
                }
            }
            #[cfg(feature = "vulkan")]
            pub fn as_vulkan(&self) -> Option<&mx_vulkan::$name> {
                match self {
                    Self::Vulkan(x) => Some(x),
                    _ => None,
                }
            }
            #[cfg(feature = "gl")]
            pub fn as_gl(&self) -> Option<&mx_gl::$name> {
                match self {
                    Self::Gl(x) => Some(x),
                    _ => None,
                }
            }
            #[cfg(windows)]
            pub fn as_dx11(&self) -> Option<&mx_dx11::$name> {
                match self {
                    Self::Dx11(x) => Some(x),
                    _ => None,
                }
            }
            #[cfg(feature = "noop")]
            pub fn as_noop(&self) -> Option<&mx_noop::$name> {
                match self {
                    Self::Noop(x) => Some(x),
                    _ => None,
                }
            }
        }

        #[cfg(target_vendor = "apple")]
        #[allow(unreachable_patterns)]
        impl Pick<mx_metal::$name> for $name {
            fn pick(&self) -> &mx_metal::$name {
                match self {
                    Self::Metal(x) => x,
                    _ => mismatch(),
                }
            }
        }
        #[cfg(windows)]
        #[allow(unreachable_patterns)]
        impl Pick<mx_dx12::$name> for $name {
            fn pick(&self) -> &mx_dx12::$name {
                match self {
                    Self::Dx12(x) => x,
                    _ => mismatch(),
                }
            }
        }
        #[cfg(feature = "vulkan")]
        #[allow(unreachable_patterns)]
        impl Pick<mx_vulkan::$name> for $name {
            fn pick(&self) -> &mx_vulkan::$name {
                match self {
                    Self::Vulkan(x) => x,
                    _ => mismatch(),
                }
            }
        }
        #[cfg(feature = "gl")]
        #[allow(unreachable_patterns)]
        impl Pick<mx_gl::$name> for $name {
            fn pick(&self) -> &mx_gl::$name {
                match self {
                    Self::Gl(x) => x,
                    _ => mismatch(),
                }
            }
        }
        #[cfg(windows)]
        #[allow(unreachable_patterns)]
        impl Pick<mx_dx11::$name> for $name {
            fn pick(&self) -> &mx_dx11::$name {
                match self {
                    Self::Dx11(x) => x,
                    _ => mismatch(),
                }
            }
        }
        #[cfg(feature = "noop")]
        #[allow(unreachable_patterns)]
        impl Pick<mx_noop::$name> for $name {
            fn pick(&self) -> &mx_noop::$name {
                match self {
                    Self::Noop(x) => x,
                    _ => mismatch(),
                }
            }
        }
    };
}

mod api;
mod conv;

pub use api::*;

/// The backend's object inside an mX object, for the backend the surrounding call goes to.
/// Objects of two backends used together (a Vulkan texture in a GL pass) are a bug and panic.
pub trait Pick<B> {
    fn pick(&self) -> &B;
}

#[cold]
#[track_caller]
pub(crate) fn mismatch() -> ! {
    panic!("mX: objects of two different backends used together (each device's objects work only with that device)")
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no GPU mX can use with the backends asked for")]
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
    /// The GPU is gone (removed, reset, hung): nothing made on this device works any more;
    /// the device has to be made again.
    #[error("the GPU was lost: {0}")]
    DeviceLost(String),
    #[error("surface: {0}")]
    Surface(String),
}

pub type Result<T> = std::result::Result<T, Error>;

macro_rules! from_backend_error {
    ($($(#[$c:meta])* $k:ident;)*) => {$(
        $(#[$c])*
        impl From<$k::Error> for Error {
            fn from(e: $k::Error) -> Error {
                use $k::Error as E;
                match e {
                    E::NoDevice => Error::NoDevice,
                    E::OutOfMemory(s) => Error::OutOfMemory(s),
                    E::Unsupported(s) => Error::Unsupported(s),
                    E::HeapFull(s) => Error::HeapFull(s),
                    E::Shader(s) => Error::Shader(s),
                    E::Compile { label, message } => Error::Compile { label, message },
                    E::Pipeline { label, message } => Error::Pipeline { label, message },
                    E::Device(s) => Error::Device(s),
                    E::DeviceLost(s) => Error::DeviceLost(s),
                    E::Surface(s) => Error::Surface(s),
                }
            }
        }
    )*};
}

from_backend_error! {
    #[cfg(target_vendor = "apple")] mx_metal;
    #[cfg(windows)] mx_dx12;
    #[cfg(feature = "vulkan")] mx_vulkan;
    #[cfg(feature = "gl")] mx_gl;
    #[cfg(windows)] mx_dx11;
    #[cfg(feature = "noop")] mx_noop;
}

/// What the backends know of a lost GPU: Direct3D 12's DRED report (the reason, where each
/// unfinished command list stood, a page fault and the resources there) or Vulkan's
/// `VK_EXT_device_fault` report. `None` while no device is lost, or where the backend keeps
/// no such report. The first loss is also written to the log by the backend itself.
pub fn lost_report() -> Option<String> {
    #[cfg(windows)]
    if let Some(r) = mx_dx12::removed_report() {
        return Some(r);
    }
    #[cfg(feature = "vulkan")]
    if let Some(r) = mx_vulkan::fault_report() {
        return Some(r);
    }
    None
}

pub type RenderPipelineDescriptor<'a> = mx_types::RenderPipelineDescriptor<'a, ShaderModule>;
pub type ComputePipelineDescriptor<'a> = mx_types::ComputePipelineDescriptor<'a, ShaderModule>;
pub type VertexState<'a> = mx_types::VertexState<'a, ShaderModule>;
pub type FragmentState<'a> = mx_types::FragmentState<'a, ShaderModule>;
pub type RenderPassDescriptor<'a> = mx_types::RenderPassDescriptor<'a, TextureView, QuerySet>;
pub type ComputePassDescriptor<'a> = mx_types::ComputePassDescriptor<'a, QuerySet>;
pub type PassTimestampWrites<'a> = mx_types::PassTimestampWrites<'a, QuerySet>;
pub type RenderPassColorAttachment<'a> = mx_types::RenderPassColorAttachment<'a, TextureView>;
pub type RenderPassDepthStencilAttachment<'a> = mx_types::RenderPassDepthStencilAttachment<'a, TextureView>;
pub type TexelCopyTextureInfo<'a> = mx_types::TexelCopyTextureInfo<'a, Texture>;
pub type TexelCopyBufferInfo<'a> = mx_types::TexelCopyBufferInfo<'a, Buffer>;
pub type BlasBuildEntry<'a> = mx_types::BlasBuildEntry<'a, Blas, Buffer>;
pub type BlasTriangleGeometry<'a> = mx_types::BlasTriangleGeometry<'a, Buffer>;
pub type TlasInstance = mx_types::TlasInstance<Blas>;

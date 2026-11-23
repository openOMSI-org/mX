//! mX on Direct3D 12.
//!
//! The bindless heap is the start of one shader-visible CBV/SRV/UAV descriptor heap (a texture
//! view's index is its descriptor's place there) and one shader-visible sampler heap (2048
//! samplers); every root signature has one descriptor table over each, with an SRV range per
//! typed array of the prelude, all starting at descriptor 0. Root groups 0..=6 are one
//! descriptor table each, filled per draw from a ring in the same heap; immediates are root
//! constants. Registers are fixed by `(group, binding)`: see `mx_shader::hlsl`.
//!
//! Resource states: every resource has a home state (`ALL_SHADER_RESOURCE` for textures with
//! `TEXTURE_BINDING`, so the heap can read them at any time; `COMMON` otherwise) and is in it
//! between operations. A command list moves what a pass, copy or dispatch uses out of the home
//! state and back (buffers return at the end of the list), so command lists can run in any
//! order. See `docs/ARCHITECTURE.md`.
//!
//! This backend has been compile-checked only (`x86_64-pc-windows-msvc`); it has never run.

#![cfg(windows)]

mod bundle;
mod command;
mod compiler;
mod conv;
mod device;
mod dred;
pub use dred::removed_report;
mod heap;
mod pipeline;
mod query;
mod resource;
mod rt;
mod surface;
mod track;

pub use bundle::{RenderBundle, RenderBundleEncoder};
pub use command::{CommandBuffer, CommandEncoder, ComputePass, RenderPass};
pub use device::{Adapter, Device, DeviceDescriptor, Instance, Queue, SubmissionIndex};
pub use pipeline::{ComputePipeline, RenderPipeline, ShaderModule, ShaderModuleDescriptor};
pub use query::QuerySet;
pub use resource::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferInitDescriptor, BufferView, IndirectDraws, Sampler, Texture, TextureView};
pub use rt::{Blas, Tlas};
pub use surface::{Surface, SurfaceTexture, SwapchainMaker, SwapchainRequest};
/// The `windows` crate this backend's native handles come from.
pub use windows;

use windows::Win32::Graphics::Direct3D12 as d3d12;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no Direct3D 12 device mX can use (it needs feature level 11.0 and resource binding tier 2)")]
    NoDevice,
    #[error("out of GPU memory creating {0}")]
    OutOfMemory(String),
    #[error("{0} is not supported on this device")]
    Unsupported(String),
    #[error("the bindless heap is full: {0}")]
    HeapFull(&'static str),
    #[error(transparent)]
    Shader(#[from] mx_shader::ShaderError),
    #[error("{label}: Direct3D 12 could not compile the shader: {message}")]
    Compile { label: String, message: String },
    #[error("{label}: Direct3D 12 could not build the pipeline: {message}")]
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

/// A Direct3D error from creating `what`.
pub(crate) fn hr_err(what: &str, e: windows::core::Error) -> Error {
    use windows::Win32::Foundation::E_OUTOFMEMORY;
    use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET};
    let code = e.code();
    if code == E_OUTOFMEMORY {
        Error::OutOfMemory(what.into())
    } else if code == DXGI_ERROR_DEVICE_REMOVED || code == DXGI_ERROR_DEVICE_RESET {
        crate::dred::log_once(what);
        Error::DeviceLost(format!("{what}: {e}"))
    } else {
        Error::Device(format!("{what}: {e}"))
    }
}

/// A NUL-terminated UTF-16 string.
pub(crate) fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Name a D3D12 object for debuggers (PIX, the debug layer).
pub(crate) fn set_name(obj: &impl windows::core::Interface, label: &str) {
    if let Ok(o) = obj.cast::<d3d12::ID3D12Object>() {
        let w = wide(label);
        // SAFETY: a NUL-terminated string that outlives the call.
        let _ = unsafe { o.SetName(windows::core::PCWSTR(w.as_ptr())) };
    }
}

/// A COM pointer reused without a reference count of its own, for the `ManuallyDrop<Option<T>>`
/// fields of D3D12 descriptions (barriers, copy locations, pipeline descriptions).
///
/// # Safety
/// `t` must outlive every use of the result, and the result must never be dropped.
pub(crate) unsafe fn borrow<T: windows::core::Interface>(t: &T) -> std::mem::ManuallyDrop<Option<T>> {
    // SAFETY: `Option<T>` of a COM interface is a nullable pointer, as `T` is a non-null one;
    // the copy is never released (ManuallyDrop), so the reference count stays the owner's.
    unsafe { std::mem::transmute_copy(t) }
}

/// Objects compare and hash by identity (the same object, not equal contents), as wgpu's do.
macro_rules! identity {
    ($($t:ty => |$x:ident| $e:expr;)*) => {$(
        impl PartialEq for $t {
            fn eq(&self, o: &Self) -> bool {
                let a = { let $x = self; $e };
                let b = { let $x = o; $e };
                a == b
            }
        }
        impl Eq for $t {}
        impl std::hash::Hash for $t {
            fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
                let $x = self;
                ($e).hash(h)
            }
        }
    )*};
}

use std::sync::Arc;

identity! {
    Buffer => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    Texture => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    TextureView => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    Sampler => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    BindGroup => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    Blas => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    Device => |x| Arc::as_ptr(&x.shared) as *const u8 as usize;
}

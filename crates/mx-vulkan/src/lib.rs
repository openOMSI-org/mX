//! mX on Vulkan (1.3, or 1.2 with dynamic rendering and synchronization2; MoltenVK works).
//!
//! The bindless heap is one descriptor set (set 0) of `UPDATE_AFTER_BIND | PARTIALLY_BOUND`
//! runtime arrays, one binding per typed array of the prelude; root groups 0..=6 are sets
//! 1..=7, allocated per draw from the pipeline's reflected bindings; immediates are push
//! constants. Every image lives in `GENERAL` layout (swapchain images go to `PRESENT_SRC` to
//! be shown), so the heap's descriptors are valid whatever a pass does, and passes and copies
//! are ordered by one memory barrier between them. See `docs/ARCHITECTURE.md`.

mod bundle;
mod command;
mod conv;
mod device;
mod fault;
pub use fault::fault_report;
mod heap;
mod pipeline;
mod query;
mod resource;
mod rt;
mod surface;
mod zero;

pub use ash;
pub use bundle::{RenderBundle, RenderBundleEncoder};
pub use command::{CommandBuffer, CommandEncoder, ComputePass, RenderPass};
pub use device::{Adapter, Device, DeviceDescriptor, Instance, Queue, SubmissionIndex};
pub use pipeline::{ComputePipeline, RenderPipeline, ShaderModule, ShaderModuleDescriptor};
pub use query::QuerySet;
pub use resource::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferInitDescriptor, BufferView, IndirectDraws, Sampler, Texture, TextureView};
pub use rt::{Blas, Tlas};
pub use surface::{Surface, SurfaceTexture, SwapchainFns, SwapchainMaker};

use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no Vulkan device mX can use (it needs Vulkan 1.2 with descriptor indexing, timeline semaphores, dynamic rendering and synchronization2)")]
    NoDevice,
    #[error("out of GPU memory creating {0}")]
    OutOfMemory(String),
    #[error("{0} is not supported on this device")]
    Unsupported(String),
    #[error("the bindless heap is full: {0}")]
    HeapFull(&'static str),
    #[error(transparent)]
    Shader(#[from] mx_shader::ShaderError),
    #[error("{label}: Vulkan could not compile the shader: {message}")]
    Compile { label: String, message: String },
    #[error("{label}: Vulkan could not build the pipeline: {message}")]
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

/// A Vulkan error from creating `what`.
pub(crate) fn vk_err(what: &str, e: vk::Result) -> Error {
    match e {
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => Error::OutOfMemory(what.into()),
        vk::Result::ERROR_DEVICE_LOST => {
            crate::fault::log_once(what);
            Error::DeviceLost(what.into())
        }
        other => Error::Device(format!("{what}: {other}")),
    }
}

pub(crate) fn alloc_err(what: &str, e: gpu_allocator::AllocationError) -> Error {
    match e {
        gpu_allocator::AllocationError::OutOfMemory => Error::OutOfMemory(what.into()),
        other => Error::Device(format!("{what}: {other}")),
    }
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

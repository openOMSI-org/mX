# The backend contract

`mx` wraps them: each mX type is an enum over the backends of the build, a call matches to the
backend that made the object (see ARCHITECTURE, Crates and backend selection). Every backend crate (`mx-metal`, `mx-vulkan`, `mx-dx12`, `mx-gl`, `mx-noop`) exports these items with
these signatures. `mx` dispatches to them; the conformance tests in `crates/mx/tests` run
against each. Shared plain data (`t::...`) comes from `mx-types`; WGSL handling from
`mx-shader`. `mx-metal` is the reference implementation: when in doubt, do what it does.

All GPU objects are cheap `Clone` handles (an `Arc` inside) and `Send + Sync`.
Errors: `pub enum Error` (a `thiserror` enum; at least `NoDevice`, `OutOfMemory(String)`,
`Unsupported(String)`, `HeapFull(&'static str)`, `Shader(mx_shader::ShaderError)`,
`Compile { label, message }`, `Pipeline { label, message }`, `Device(String)`,
`DeviceLost(String)` (the GPU removed, reset or hung: from polls and waits), `Surface(String)`), and `pub type Result<T> = std::result::Result<T, Error>`.
Misuse that wgpu reports as a validation error (wrong usage, out-of-range copy, draw without
pipeline or with a missing root binding) panics with a clear message.

## Instance, adapter, device, queue

```rust
pub struct Instance;
impl Instance {
    pub fn new() -> Instance;                                   // also Default
    pub fn enumerate_adapters(&self) -> Vec<Adapter>;          // only adapters that can run mX
    pub fn default_adapter(&self) -> Result<Adapter>;          // discrete first, then integrated
    pub fn create_surface<'w>(&self, window: impl HasWindowHandle + HasDisplayHandle + Send + Sync + 'w) -> Result<Surface<'w>>;
    pub unsafe fn create_surface_unsafe(&self, display: RawDisplayHandle, window: RawWindowHandle) -> Result<Surface<'static>>;
}
pub struct Adapter;   // Clone
impl Adapter {
    pub fn info(&self) -> t::AdapterInfo;
    pub fn limits(&self) -> t::Limits;
    pub fn supports_bc(&self) -> bool;
    pub fn supports_ray_query(&self) -> bool;
    pub fn supports_timestamps(&self) -> bool;
    pub fn supports_indirect_count(&self) -> bool;               // GPU-driven batches (create_indirect_draws)
    pub fn supports_compute(&self) -> bool;                      // false on GL 3.3 / ES 3.0
    pub fn storage_buffers_per_stage(&self) -> (u32, u32);       // (vertex, fragment); 0 on GL below 4.3 / ES 3.1
    pub fn texture_units(&self) -> (u32, u32, u32);              // root textures: (vertex, fragment, pipeline)
    pub fn memory_info(&self) -> Option<t::AdapterMemory>;       // dedicated bytes, budget; None on GL
    pub fn supports_msaa(&self, format: t::TextureFormat, samples: u32) -> bool;
    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Result<(Device, Queue)>;
}
pub struct DeviceDescriptor<'a> { pub label: Option<&'a str>, pub bindless_textures: u32, pub bindless_samplers: u32, pub bindless_buffers: u32 }
// Default: 1 << 16 textures, 2048 samplers, 1 << 15 buffer views
pub struct SubmissionIndex(pub u64);   // Copy, Ord; first submission is 1

impl Device {   // Clone
    pub fn info(&self) -> &t::AdapterInfo;
    pub fn limits(&self) -> t::Limits;
    pub fn poll(&self) -> Result<()>;                          // non-blocking: reclaim finished work
    pub fn wait(&self, index: SubmissionIndex) -> Result<()>;
    pub fn wait_idle(&self) -> Result<()>;
    pub fn is_done(&self, index: SubmissionIndex) -> bool;     // as of the last poll / wait
    pub fn bindless_textures_in_use(&self) -> usize;
    pub fn bindless_buffers_in_use(&self) -> usize;
    pub fn create_buffer(&self, desc: &t::BufferDescriptor<'_>) -> Result<Buffer>;
    pub fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Result<Buffer>;
    pub fn create_texture(&self, desc: &t::TextureDescriptor<'_>) -> Result<Texture>;
    pub fn create_sampler(&self, desc: &t::SamplerDescriptor<'_>) -> Result<Sampler>;
    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup;
    pub fn create_shader_module(&self, desc: &ShaderModuleDescriptor<'_>) -> Result<ShaderModule>;
    pub fn precompile(&self, module: &ShaderModule) -> Result<()>;   // every entry point now, overrides at their defaults
    pub fn create_render_pipeline(&self, desc: &t::RenderPipelineDescriptor<'_, ShaderModule>) -> Result<RenderPipeline>;
    pub fn create_compute_pipeline(&self, desc: &t::ComputePipelineDescriptor<'_, ShaderModule>) -> Result<ComputePipeline>;
    pub fn create_command_encoder(&self, label: Option<&str>) -> CommandEncoder;
    pub fn create_render_bundle_encoder(&self, desc: &t::RenderBundleEncoderDescriptor<'_>) -> RenderBundleEncoder;
    pub fn create_query_set(&self, desc: &t::QuerySetDescriptor<'_>) -> Result<QuerySet>;
    pub fn create_blas(&self, desc: &t::CreateBlasDescriptor<'_>, sizes: &[t::BlasTriangleGeometrySizeDescriptor]) -> Result<Blas>;
    pub fn create_tlas(&self, desc: &t::CreateTlasDescriptor<'_>) -> Result<Tlas>;
    pub fn create_indirect_draws(&self, desc: &t::IndirectDrawsDescriptor<'_>) -> Result<IndirectDraws>;
}
impl Queue {    // Clone
    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) -> SubmissionIndex;
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<()>;
    pub fn write_texture(&self, dst: t::TexelCopyTextureInfo<'_, Texture>, data: &[u8], layout: t::TexelCopyBufferLayout, size: t::Extent3d) -> Result<()>;
    pub fn get_timestamp_period(&self) -> f32;                 // nanoseconds per timestamp tick
}
```

`write_*` take effect before the commands of the next `submit` (they are staged and copied
on the GPU). An empty `submit([])` still returns an index that can be waited for.

## Resources

```rust
pub struct BufferInitDescriptor<'a> { pub label: Option<&'a str>, pub contents: &'a [u8], pub usage: t::BufferUsages }
impl Buffer {
    pub fn size(&self) -> u64;  pub fn usage(&self) -> t::BufferUsages;
    pub fn map_read<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R>;      // waits for the GPU to be idle
    pub fn map_write<R>(&self, range: Range<u64>, f: impl FnOnce(&mut [u8]) -> R) -> Result<R>; // likewise
    pub fn map_read_now<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R>;  // no wait: the caller checked Device::is_done
    pub unsafe fn mapped_ptr(&self) -> NonNull<u8>;                                             // MAP_* buffers; caller keeps off the GPU's work
    pub fn create_view(&self, desc: &t::BufferViewDescriptor<'_>) -> Result<BufferView>;        // STORAGE buffers; checked by `desc.resolve`
}
impl BufferView {
    pub fn index(&self) -> u32;            // bindless index: mx_buffers[i], mx_rw_buffers[i], a shader's own views
    pub fn buffer(&self) -> &Buffer; offset(); size(); is_writable();
}
impl Texture {
    pub fn size(&self) -> t::Extent3d; width(); height(); format(); dimension(); mip_level_count(); sample_count(); usage();
    pub fn create_view(&self, desc: &t::TextureViewDescriptor<'_>) -> Result<TextureView>;
}
impl TextureView {
    pub fn index(&self) -> u32;            // bindless index; panics if none
    pub fn try_index(&self) -> Option<u32>;
    pub fn texture(&self) -> &Texture; format(); dimension(); mip_levels() -> Range<u32>; array_layers() -> Range<u32>;
}
impl Sampler { pub fn index(&self) -> u32; pub fn is_comparison(&self) -> bool; }
impl QuerySet { pub fn count(&self) -> u32; pub fn ty(&self) -> t::QueryType; }
impl Blas {}
impl Tlas {    // instances are set by index, as in wgpu: tlas[i] = Some(instance)
    pub fn max_instances(&self) -> u32;
    pub fn index(&self) -> u32;            // bindless index (`mx_tlas[i]`, heap binding 11)
}
impl Index<usize> for Tlas { type Output = Option<t::TlasInstance<Blas>>; }
impl IndexMut<usize> for Tlas {}       // (needs &mut Tlas: Tlas is not Clone)

pub struct BufferBinding<'a> { pub buffer: &'a Buffer, pub offset: u64, pub size: Option<u64> }
pub enum BindingResource<'a> { Buffer(BufferBinding<'a>), TextureView(&'a TextureView), Sampler(&'a Sampler), AccelerationStructure(&'a Tlas) }
pub struct BindGroupEntry<'a> { pub binding: u32, pub resource: BindingResource<'a> }
pub struct BindGroupDescriptor<'a> { pub label: Option<&'a str>, pub entries: &'a [BindGroupEntry<'a>] }
pub struct ShaderModuleDescriptor<'a> { pub label: Option<&'a str>, pub source: &'a str }   // WGSL
```

A view gets a heap index iff its texture has `TEXTURE_BINDING` and it is not a
both-aspects view of a depth-stencil texture. A sampler always gets one, and so does a buffer
view. Buffer views have their own index space. A backend makes every buffer view reachable
by every pass without per-draw work from the caller: resident on Metal (residency set, or
`useResources` of the heap's buffers), in the descriptor array on Vulkan, in the state shaders
reach it in on Direct3D 12 (see the architecture's synchronisation section); writable views are
also declared for writing where the API tracks hazards. Freed indices are
reused only after the next submission completes (see `heap_slots_are_reused_only_after_the_gpu_is_done`).

## Commands

```rust
impl CommandEncoder {
    pub fn finish(self) -> CommandBuffer;
    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64);
    pub fn copy_buffer_to_texture(&mut self, src: t::TexelCopyBufferInfo<'_, Buffer>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d);
    pub fn copy_texture_to_buffer(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyBufferInfo<'_, Buffer>, size: t::Extent3d);
    pub fn copy_texture_to_texture(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d);
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: Option<u64>);
    pub fn generate_mipmaps(&mut self, texture: &Texture);
    pub fn resolve_query_set(&mut self, set: &QuerySet, range: Range<u32>, dst: &Buffer, dst_offset: u64);
    pub fn write_timestamp(&mut self, set: &QuerySet, index: u32);
    pub fn build_acceleration_structures<'a>(&mut self, blas: impl IntoIterator<Item = &'a t::BlasBuildEntry<'a, Blas, Buffer>>, tlas: impl IntoIterator<Item = &'a Tlas>);
    // outside passes; `count` draws (a u32 in `count`) of `args`, at most `max_draws`; `args` needs STORAGE | INDIRECT
    pub fn encode_indirect_draws(&mut self, draws: &IndirectDraws, pipeline: &RenderPipeline, args: &Buffer, args_offset: u64, count: &Buffer, count_offset: u64, index: Option<(&Buffer, u64, t::IndexFormat)>);
    pub fn begin_render_pass(&mut self, desc: &t::RenderPassDescriptor<'_, TextureView, QuerySet>) -> RenderPass<'_>;
    pub fn begin_compute_pass(&mut self, desc: &t::ComputePassDescriptor<'_, QuerySet>) -> ComputePass<'_>;
    pub fn push_debug_group(&mut self, label: &str); pub fn pop_debug_group(&mut self); pub fn insert_debug_marker(&mut self, label: &str);
}
impl RenderPass<'_> {  // ends on drop
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline);
    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup);   // groups 0..=6
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]);
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64);
    pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64, format: t::IndexFormat);
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32);
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32);
    pub fn set_stencil_reference(&mut self, reference: u32);
    pub fn set_blend_constant(&mut self, c: t::Color);
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);
    pub fn draw_indirect(&mut self, args: &Buffer, offset: u64);                     // t::DrawIndirectArgs
    pub fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64);             // t::DrawIndexedIndirectArgs
    pub fn multi_draw_indirect(&mut self, args: &Buffer, offset: u64, count: u32);
    pub fn multi_draw_indexed_indirect(&mut self, args: &Buffer, offset: u64, count: u32);
    pub fn execute_indirect_draws(&mut self, draws: &IndirectDraws);                 // the last encode_indirect_draws of the batch
    pub fn begin_occlusion_query(&mut self, index: u32); pub fn end_occlusion_query(&mut self);
    pub fn execute_bundles<'b>(&mut self, bundles: impl IntoIterator<Item = &'b RenderBundle>);
    pub fn push_debug_group(&mut self, label: &str); pub fn pop_debug_group(&mut self); pub fn insert_debug_marker(&mut self, label: &str);
}
impl ComputePass<'_> {  // ends on drop
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline);
    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup);
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]);
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32);
    pub fn dispatch_workgroups_indirect(&mut self, args: &Buffer, offset: u64);     // t::DispatchIndirectArgs
}
```

Render bundles are recorded on the CPU and replayed into the pass: they have the render
pass's drawing methods (`set_pipeline`, `set_bind_group`, `set_immediates`,
`set_vertex_buffer`, `set_index_buffer`, `draw*`), `RenderBundleEncoder::finish(self,
label: Option<&str>) -> RenderBundle`. State set in a bundle does not leak into the pass
(after `execute_bundles` the pass has no pipeline and no bindings, as in WebGPU).

Timestamps: `begin_render_pass` / `begin_compute_pass` honour `timestamp_writes`;
`resolve_query_set` writes `u64`s (`t::QUERY_SIZE` bytes each) at a
`t::QUERY_RESOLVE_BUFFER_ALIGNMENT`-aligned offset; ticks × `get_timestamp_period()` = ns.

## Surfaces

```rust
impl Surface<'_> {
    pub fn get_capabilities(&self, adapter: &Adapter) -> t::SurfaceCapabilities;
    pub fn configure(&self, device: &Device, config: &t::SurfaceConfiguration);
    pub fn get_current_texture(&self) -> t::CurrentSurfaceTexture<SurfaceTexture>;
}
pub struct SurfaceTexture { pub texture: Texture }
impl SurfaceTexture { pub fn present(self); }  // presents after all work submitted so far
```

## Interop (native handles)

Each backend exposes its native objects so DLSS, FSR 3, MetalFX and OpenXR can work on
mX's device: `Device::raw()`, `Queue::raw()`, `Buffer::raw()`, `Texture::raw()`,
`CommandEncoder::raw()` (the native command buffer / list), plus
`unsafe fn Device::texture_from_raw(&self, raw, desc: &t::TextureDescriptor<'_>) -> Texture`
for textures created outside mX (XR swapchain images). The raw types are the backend's own
(`ash::vk::*`, `windows::Win32::Graphics::Direct3D12::*`, `objc2-metal` protocol objects).

## Shader slot layout

Each backend maps WGSL `(group, binding)` itself (naga's per-backend binding maps). The heap
is always group 7 with the bindings in `mx_shader::heap` (and buffer views at 16..=63); root
groups are 0..=6.
- Metal: `mx_shader::msl`. Immediates at buffer 0, sizes at 1, then each heap binding the
  entry point uses (argument buffers of `MTLResourceID`s for textures and samplers, of GPU
  addresses for buffer views) and its root buffers, assigned per entry point; vertex buffers at
  `30 - index`. A pipeline binds its stages' heap tables when it is set.
- Vulkan: `mx_shader::spirv`. Set 0 = heap (UPDATE_AFTER_BIND | PARTIALLY_BOUND
  bindings: the texture and sampler arrays, and binding 8, the storage buffers every buffer
  view of a shader aliases), sets 1..=7 = root groups 0..=6 (one descriptor set
  layout per group, built from the pipeline's reflected roots), immediates = push constants.
- Direct3D 12: `mx_shader::hlsl`. The heap as descriptor ranges in one root table
  (textures: an SRV range per heap binding, all starting at heap offset 0, registers
  `t0, space{8 + binding}`, as long as the device's heap: naga's HLSL writer cannot declare
  an unbounded array), samplers through naga's sampler heap (`s0` / `s2048`, `space14`) and
  per-group sampler index buffers (`t{group}, space7`; the heap group's is the identity, a
  root group's indexed by binding), root group `g` as one descriptor table (registers
  `b|t|u{binding}, space{g}`), immediates as root constants (`b0, space15`), naga's special
  constants at `b1, space15`. Buffer views: raw SRVs and raw UAVs after the texture slots,
  one range per view a pipeline's stages use (`t0` or `u0`, `space{16 + binding}`), all over
  the same descriptors.

Compile with wgpu-hal's settings for the same API so output matches: naga bounds-check
policies as wgpu-hal sets them for a checked module on that API (`Module::bounds_check_policies()`
is Metal's, and Vulkan's without robust access; a module that uses the buffer heap leaves
buffer accesses unchecked), zero-initialised workgroup memory, bounded loops.

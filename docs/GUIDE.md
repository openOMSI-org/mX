# Using mX

This guide covers the parts of mX that differ from wgpu. Names and descriptor shapes follow
wgpu wherever the concept is the same, so `TextureDescriptor`, `RenderPassDescriptor`,
`queue.write_buffer` and the rest work as you would expect from wgpu.

## Device

```rust
let instance = mx::Instance::new();
let adapter = instance.default_adapter()?;            // a discrete GPU first, then an integrated one
let (device, queue) = adapter.request_device(&mx::DeviceDescriptor {
    label: Some("game"),
    bindless_textures: 1 << 16,                       // the defaults
    bindless_samplers: 2048,
    bindless_buffers: 1 << 15,
})?;
```

`bindless_textures`, `bindless_samplers` and `bindless_buffers` size the heap. mX lowers them to what the device
supports; `device.limits()` reports the sizes it got.

### Choosing the graphics API

A build has every backend of its platform. To offer them as a setting:

```rust
let available = mx::Instance::compiled_backends();             // e.g. DX12 | VULKAN | GL on Windows
let instance = mx::Instance::new();
let adapter = instance.adapter_for(mx::Backend::Dx12)          // the player's choice
    .or_else(|_| instance.default_adapter())?;                 // or the first that works
let (device, queue) = adapter.request_device(&Default::default())?;
surface.configure(&device, &config);                           // the surface follows the device's backend
```

Changing the setting means dropping the device and opening one on the new backend; the same
surface is configured with it. The renderer's code stays the same: every type is the same mX
type whatever the backend. Pick downlevel paths from `Adapter::supports_compute`,
`storage_buffers_per_stage` and `supports_indirect_count` (OpenGL has less), not from the
backend's name.

`Instance::enumerate_adapters` lists only adapters that can run mX. A Metal device needs
Metal 3 and Tier 2 argument buffers; a Vulkan device needs descriptor indexing, dynamic
rendering and synchronization2.

## The heap

Every texture view created from a texture with `TextureUsages::TEXTURE_BINDING` receives an
index, and so does every sampler (and every storage view and top-level acceleration
structure, below):

```rust
let view = texture.create_view(&Default::default())?;
let sampler = device.create_sampler(&mx::SamplerDescriptor::default())?;
let (t, s) = (view.index(), sampler.index());
```

Shaders read the heap through arrays that mX declares for them, in group 7:

| Binding | Array | Type |
|---------|-------|------|
| 0 | `mx_tex2d` | `binding_array<texture_2d<f32>>` |
| 1 | `mx_tex2d_array` | `binding_array<texture_2d_array<f32>>` |
| 2 | `mx_texcube` | `binding_array<texture_cube<f32>>` |
| 3 | `mx_tex3d` | `binding_array<texture_3d<f32>>` |
| 4 | `mx_texdepth` | `binding_array<texture_depth_2d>` |
| 5 | `mx_tex2d_u32` | `binding_array<texture_2d<u32>>` |
| 6 | `mx_texdepth_ms` | `binding_array<texture_depth_multisampled_2d>` |
| 7 | `mx_samplers` | `binding_array<sampler>` |
| 8 | `mx_samplers_cmp` | `binding_array<sampler_comparison>` |
| 9, 10 | `mx_buffers`, `mx_rw_buffers` | buffer views, below |

Bindings 11 to 63 are for arrays a shader declares itself: acceleration structures (11),
storage textures (12 to 15) and typed buffer views (16 to 63), below.

A texture index is valid in the array that matches the view's dimension and sample type: a 2D
view is `mx_tex2d[i]`, a cube view `mx_texcube[i]`, a 2D depth view both `mx_tex2d[i]` and
`mx_texdepth[i]`, a multisampled depth view `mx_texdepth_ms[i]` (read with
`textureLoad(mx_texdepth_ms[i], p, sample)`). Samplers have their own index space, and comparison samplers are read
through `mx_samplers_cmp`.

When a view or sampler is dropped, its index returns to the heap only after every submission
that could still read it has finished on the GPU. Holding on to an index after dropping the
view is a bug in the caller; mX does not detect it.

## Buffers in the heap

A buffer with `BufferUsages::STORAGE` can put ranges of itself in the heap as buffer views,
which have indices of their own:

```rust
let lights = device.create_buffer(&mx::BufferDescriptor { label: Some("lights"), size, usage: mx::BufferUsages::STORAGE | mx::BufferUsages::COPY_DST })?;
let view = lights.create_view(&mx::BufferViewDescriptor::default())?;     // the whole buffer, read-only
let out = results.create_view(&mx::BufferViewDescriptor { writable: true, ..Default::default() })?;
pass.set_immediates(0, bytemuck::cast_slice(&[view.index(), out.index()]));
```

Shaders read every buffer view through `mx_buffers[i].w` (an `array<u32>`) and write writable
ones through `mx_rw_buffers[i].w`. For typed access, a shader declares its own views of the
same heap in group 7, at bindings 16 to 63:

```wgsl
struct Light { position: vec4<f32>, colour: vec4<f32> }
struct Lights { l: array<Light> }
@group(7) @binding(16) var<storage, read> lights: binding_array<Lights>;

struct Draw { lights: u32, count: u32 }
var<immediate> draw: Draw;

fn shade(p: vec3<f32>) -> vec3<f32> {
    var c = vec3<f32>(0.0);
    for (var i = 0u; i < draw.count; i++) { c += light(lights[draw.lights].l[i], p); }
    return c;
}
```

Every typed view is the same array of buffers seen with another element type, so one index
works in all of them. What the bindless model asks of the renderer:

- A view's offset is a multiple of `limits().buffer_view_offset_alignment`, its size a
  multiple of 4.
- Shaders write only views made `writable`. On Direct3D 12 a write through a read-only view
  goes nowhere; elsewhere it is undefined.
- Buffer accesses in a shader that uses the buffer heap are not bounds-checked, like heap
  indices: keep counts in the data or the immediates. `arrayLength` of a heap buffer is
  refused when the shader is created.
- A pass sees what earlier passes and copies wrote to a heap buffer, as with bind groups. Two
  draws of one pass that write and read the same heap buffer need the same care as in any
  other API.

## Storage textures in the heap

A view of a texture made with `TextureUsages::STORAGE_BINDING` that shows one mip of a 2D
texture gets an index too (the same one, if the texture also has `TEXTURE_BINDING`). Shaders
write it through a view of the storage textures they declare at bindings 12 to 15, of any
format:

```wgsl
@group(7) @binding(12) var out_hdr: binding_array<texture_storage_2d<rgba16float, write>>;
@group(7) @binding(13) var out_ldr: binding_array<texture_storage_2d<rgba8unorm, write>>;
struct Pass { hdr: u32, ldr: u32 }
var<immediate> pass_: Pass;

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    textureStore(out_hdr[pass_.hdr], vec2<i32>(id.xy), vec4<f32>(1.0));
}
```

Every such array is the same array of storage textures, seen with another format: the format a
shader declares must be the view's. Storage textures are write-only (`write` access). On
OpenGL they need OpenGL 4.2 or ES 3.1, on Vulkan
`descriptorBindingStorageImageUpdateAfterBind`; a pipeline that uses them fails elsewhere.

## Acceleration structures in the heap

A top-level acceleration structure has an index in the heap, `tlas.index()`, from the index
space of texture views. A shader that runs ray queries declares the array at binding 11 (it is
not in the prelude, which would ask every shader for ray queries):

```wgsl
enable wgpu_ray_query;
@group(7) @binding(11) var mx_tlas: binding_array<acceleration_structure>;

rayQueryInitialize(&rq, mx_tlas[draw.tlas], desc);
```

The index reads what the structure's last build made. It is available where
`adapter.supports_ray_query()` is (not on the bound tier).

## Immediates

Small per-draw data, typically heap indices and a few constants, goes in immediates
(push constants on Vulkan, root constants on Direct3D 12):

```wgsl
struct Draw { albedo: u32, normal: u32, smp: u32, tint: f32 }
var<immediate> draw: Draw;
```

```rust
pass.set_immediates(0, bytemuck::bytes_of(&draw));
```

`device.limits().max_immediate_size` gives the size available: 4096 bytes on Metal, the
device's push constant limit on Vulkan (at least 128 bytes).

## Bind groups

Groups 0 to 6 work like wgpu's bind groups, without layouts. A pipeline reflects what its
shaders use, so a bind group is just its entries:

```rust
let group = device.create_bind_group(&mx::BindGroupDescriptor {
    label: Some("frame"),
    entries: &[mx::BindGroupEntry {
        binding: 0,
        resource: mx::BindingResource::Buffer(mx::BufferBinding { buffer: &uniforms, offset: 0, size: None }),
    }],
});
pass.set_bind_group(0, &group);
```

Use them for shaders that have not moved to the heap yet, and for uniform buffers read at
per-draw offsets (`set_buffer_offset`). Textures and samplers can be bound here too,
which is how wgpu code runs unchanged on `wgpu-mx`.

## Pipelines and shaders

```rust
let module = device.create_shader_module(&mx::ShaderModuleDescriptor { label: Some("mesh"), source: WGSL })?;
let pipeline = device.create_render_pipeline(&mx::RenderPipelineDescriptor {
    label: Some("mesh"),
    vertex: mx::VertexState { module: &module, entry_point: None, constants: &[], buffers: &[layout] },
    primitive: Default::default(),
    depth_stencil: None,
    multisample: Default::default(),
    fragment: Some(mx::FragmentState { module: &module, entry_point: None, constants: &[], targets: &[Some(format.into())] }),
})?;
```

There is no `layout` field. `constants` sets WGSL `override` values. `device.precompile(&module)`
compiles every entry point at load time, so shader errors and the driver's compile time show
up there instead of at the first draw.

mX appends its prelude after your source, so line numbers in error messages are your own.
Shader errors come back as `Error::Shader` with naga's message.

## Synchronisation

You do not write barriers. mX keeps wgpu's rules:

- `queue.write_buffer` and `queue.write_texture` take effect before the commands of the next
  `submit`.
- A pass sees everything that earlier passes and copies wrote, including writes to textures
  it reads through the heap.
- A new texture or buffer reads as zeros until something writes it.

`device.poll()` reclaims finished work without blocking; `device.wait(index)` blocks until a
submission is done. Mapping is synchronous: `buffer.map_read` and `buffer.map_write` wait for
the GPU to be idle. `wgpu-mx` implements wgpu's asynchronous `map_async` on top of
`Buffer::mapped_ptr` and `Device::is_done`.

## Ray queries

When `adapter.supports_ray_query()` is true, WGSL ray queries work in any shader:

```rust
let blas = device.create_blas(&Default::default(), &[size])?;
let mut tlas = device.create_tlas(&mx::CreateTlasDescriptor { label: None, max_instances: 1024, flags, update_mode })?;
tlas[0] = Some(mx::TlasInstance::new(blas.clone(), transform, custom_data, 0xff));
encoder.build_acceleration_structures([&entry], [&tlas]);
```

A structure created with `AccelerationStructureFlags::ALLOW_UPDATE` and
`AccelerationStructureUpdateMode::PreferUpdate` is refitted instead of rebuilt after its
first build. For a top-level structure that holds when it has the same number of instances of
the same bottom-level structures as in its previous build.

## Native handles

`Device::raw()`, `Queue::raw()`, `Buffer::raw()`, `Texture::raw()` and `CommandEncoder::raw()`
return the backend's own objects (`objc2-metal` protocol objects, `ash::vk` handles or
`windows` Direct3D 12 interfaces). `Device::texture_from_raw` wraps a texture made elsewhere,
such as an OpenXR swapchain image or an upscaler's output. openOMSI runs MetalFX this way.

## Moving from wgpu

The quickest route is `wgpu-mx`: replace the `wgpu` dependency with it and keep the code. From
there, a renderer can move pass by pass to the heap. Bind a pass's textures through the heap
instead of its bind groups, check that the frame is unchanged, and continue with the next
pass.

## GPU-driven draws

When a compute pass decides what to draw (culling, LOD selection), it writes draw arguments
and a count into buffers, and an `IndirectDraws` batch draws them without the CPU reading
anything back:

```rust
let batch = device.create_indirect_draws(&mx::IndirectDrawsDescriptor { label: Some("scene"), max_draws: 65536, indexed: true })?;

// each frame, after the culling pass has written `args` (DrawIndexedIndirectArgs) and `count` (u32)
encoder.encode_indirect_draws(&batch, &pipeline, &args, 0, &count, 0, Some((&indices, 0, mx::IndexFormat::Uint32)));
let mut pass = encoder.begin_render_pass(&desc);
pass.set_pipeline(&pipeline);
pass.set_immediates(0, bytemuck::bytes_of(&frame_indices));
pass.execute_indirect_draws(&batch);
```

`encode_indirect_draws` goes outside passes, after the arguments are written: Metal and
Direct3D 12 do work there (see ARCHITECTURE). Each draw finds its own data through
`@builtin(instance_index)` (its `first_instance`) in heap buffers. `args` needs
`STORAGE | INDIRECT` usage, `count` `INDIRECT`. `Adapter::supports_indirect_count` says whether
the device has it (every Metal 3 and Direct3D 12 device; Vulkan devices with
`drawIndirectCount`).

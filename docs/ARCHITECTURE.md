# Architecture

mX exists so that openOMSI owns the whole path from its renderer to the driver. It replaced
wgpu under openOMSI's renderer without changing the picture, and it changes how resources
reach shaders: through one bindless heap instead of bind groups.

## Goals

1. The same picture as wgpu. Tests render the same scenes with both and compare bytes.
2. Bindless. Shaders index one global heap of textures and samplers; a draw is a pipeline,
   some buffers and a few bytes of immediates.
3. Thin. One layer between the API and Metal, Vulkan or Direct3D 12, and the native handles
   are public for interop with MetalFX, DLSS, FSR 3 and OpenXR.
4. Familiar. Names and descriptor shapes follow wgpu where the concept is the same.

WGSL stays the shader language and naga compiles it. naga is a standalone compiler from the
wgpu project; mX uses it and nothing else from wgpu at run time. A second WGSL compiler would
be years of work and a second source of shading differences.

## Crates and backend selection

One build holds every backend of its platform, and the program picks one at run time, as a
game's graphics API setting does:

| Target | Backends in the build |
|--------|------------------------|
| Apple platforms | `mx-metal` (Metal 3: macOS 13+, iOS 16+); `mx-vulkan` (MoltenVK) and `mx-gl` with the default features |
| Windows | `mx-dx12` (feature level 11.0, resource binding tier 2), `mx-vulkan`, `mx-gl` (ANGLE over Direct3D 11) |
| Linux, Android and others | `mx-vulkan` (Vulkan 1.3, or 1.2 with dynamic rendering and synchronization2), `mx-gl` |
| Every target | `mx-noop` with the default features: a device that takes every call and draws nothing, opened only when asked for (`Backends::NOOP`, `MX_BACKENDS=noop`) |

`mx::Device` and every other object is an enum of the backends' own objects, and a call is one
`match` on it to the backend that made the object; arguments that are mX objects are picked
for that backend (`mx::Pick`), and mixing two backends' objects panics. There is no trait
object, no validation layer and no copy of descriptors beyond what a call takes. With one
backend in the build (`default-features = false` on an Apple platform) the `match` has one arm
and compiles away.

`Instance::with_backends` (or `MX_BACKENDS=metal,vulkan,dx12,gl` for `Instance::new`) chooses
which backends are opened; `Instance::adapter_for(Backend)` returns a backend's adapter for a
setting; a surface takes on the backend of the device it is configured with, and lets go of
another backend's swapchain when the setting changes. `as_metal()`, `as_vulkan()`, `as_dx12()`
and `as_gl()` reach the backend's objects for interop.

Interop (MetalFX, DLSS, FSR 3, OpenXR) needs the state a texture is in when work of another
library runs between mX's submissions on mX's queue. It is fixed: on Metal there is none; on
Vulkan every image rests in `GENERAL` between mX's commands; on Direct3D 12 every texture rests
in its home state (`mx_dx12::Texture::rest_state`). Native handles: `raw()` on devices,
queues, textures and views; `mx_vulkan::Texture::raw_memory`; `mx_vulkan::Queue::with_raw`
(mX's own submissions held off); `texture_from_raw` on Metal and Direct3D 12;
`Surface::with_metal` (the layer). A Vulkan device can be made with more extensions
(`DeviceDescriptor::vulkan_extensions`, DLSS's). A lost device is `Error::DeviceLost` from the
polls and waits (Vulkan `VK_ERROR_DEVICE_LOST`, Direct3D 12 device removal, Metal's
`MTLCommandBufferErrorDeviceRemoved`).

Every backend implements the same items with the same signatures
([BACKEND_API.md](BACKEND_API.md)), and the conformance tests hold them to the same behaviour.

`mx-types` holds the plain data the backends share. `mx-shader` turns WGSL into each backend's
language with naga and reflects what an entry point uses. `wgpu-mx` implements wgpu 29's API on
top of `mx`.

## Binding model

There are no bind group layouts or pipeline layouts. A backend reflects what an entry point
uses from the naga module and places each WGSL `(group, binding)` where its API wants it.

- Group 7 is the heap, declared by the prelude that mX appends to every module
  ([GUIDE.md](GUIDE.md) lists the arrays). It holds texture views, samplers, buffer views and
  top-level acceleration structures; a shader may add its own arrays of acceleration
  structures (binding 11), its own views of the storage textures (bindings 12 to 15) and its
  own typed views of the buffers (bindings 16 to 63).
- Groups 0 to 6 are root groups: uniform and storage buffers, storage textures, textures,
  samplers and acceleration structures, set with layout-free bind groups. Native mX code needs
  none of them; they remain for code ported from wgpu (`wgpu-mx`).
- `var<immediate>` holds per-draw bytes.
- `set_buffer_offset(group, binding, offset)` moves a root uniform buffer within its buffer
  for the following draws (Vulkan's and wgpu's dynamic offsets, without layouts): one table of
  per-material rows, bound once, read as a uniform buffer at each draw's row. Uniform reads stay
  in constant memory, which on Apple GPUs is faster than reading the rows from the heap.

The heap is group 7 rather than 0 so that openOMSI's shaders, which use groups 0 to 3, compile
unchanged.

A texture view gets one index and that index is valid in every typed array the view can be
read through, and in the storage-texture views when it is one (`STORAGE_BINDING`, one mip of a
2D view). Top-level acceleration structures take their indices from the same space. A buffer view likewise is one index, valid in `mx_buffers`, `mx_rw_buffers` and
every typed view: they are the same array of buffers seen with different element types. Freed
indices are reused only after every submission that could read them has completed.

A module that reads buffers through the heap is compiled with buffer accesses unchecked, like
heap indices in every bindless engine: on Metal a bounds check would need the size of every
heap buffer passed to every draw. `arrayLength` of a heap buffer is refused for the same
reason. Modules that do not use the buffer heap keep wgpu-hal's checks and draw what they drew
under wgpu.

### Metal

Three `MTLBuffer`s of 8-byte entries: `MTLResourceID`s of textures (and of top-level
acceleration structures, which share their indices), of samplers, and GPU addresses
(`gpuAddress + offset`) of buffer views. Every texture array, storage-texture view and
acceleration-structure array reads the first. Encoders declare storage textures in the heap
for writing, and the heap's acceleration structures with the bottom-level ones of their last
build, for reading (`useResources`). naga writes a `binding_array` of textures as
an argument buffer of resource IDs and a `binding_array` of buffers as an argument buffer of
`device T*`, so the same table serves every typed array of its kind.

Slots are assigned per entry point, counting only what it uses:

| Buffer slot | Contents |
|-------------|----------|
| 0 | immediates (`setVertexBytes`, `setFragmentBytes`) |
| 1 | sizes of runtime-sized root storage buffers, then of vertex buffers |
| 2 to 22 | the heap bindings the entry point uses, in binding order (each the table of its kind), then its root buffers in `(group, binding)` order |
| 23 to 30 | vertex buffers, slot `30 - index` |

A pipeline binds its stages' heap tables when it is set. A shader that reads `mx_tex2d`,
`mx_samplers` and two buffer views has 17 slots left for root buffers; the limit is 13 root
buffers, as in wgpu.

Root textures and samplers take texture and sampler slots from 0 in `(group, binding)` order.
Each entry point gets its own assignment and counts only the root bindings it uses, so a vertex
shader and a fragment shader from different modules work together.

**Direct heap textures.** A render entry point whose heap accesses are all textures and
samplers indexed by immediates or constants (a material's textures, chosen per draw) is
compiled without the argument buffers: `msl::write_direct` lowers those accesses to texture and
sampler slots after the root ones, with the bound tier's `bound::lower`, and at each draw that
changes the immediates the encoder binds the heap's textures at the indices the immediates
give (`setFragmentTexture`). On Apple GPUs this is what the bound wgpu renderer cost: reading
openOMSI's materials through the argument buffer took 0.55 ms more of a 4.5 ms main pass on an
M4. The API stays bindless; `MX_METAL_DIRECT=0` turns it off. Entry points that read buffer
views, or index the heap any other way, keep the argument buffers.

Vertex buffers are read by the vertex shader itself (naga's vertex pulling, as in wgpu-hal),
because Metal's fixed-function vertex fetch converts normalised formats one ulp differently.

### Vulkan

Descriptor set 0 is the heap: one `UPDATE_AFTER_BIND | PARTIALLY_BOUND` array per texture
and sampler binding of the prelude, binding 9, an array of storage buffers, binding 11, an
array of acceleration structures (devices with ray queries, which need
`descriptorBindingAccelerationStructureUpdateAfterBind` for it), and binding 12, an array of
storage images (`descriptorBindingStorageImageUpdateAfterBind`). Every buffer view a shader
declares (`mx_buffers`, `mx_rw_buffers`, its own) is a SPIR-V variable at set 0, binding 9, and
every storage-texture view one at binding 12: Vulkan lets variables of one descriptor type
share a binding. A view is written at its index into every binding its dimension and sample
type allow, so a 2D depth view goes into `mx_tex2d` and `mx_texdepth`, a multisampled depth
view into `mx_texdepth_ms`, a storage view among the storage images. Root group `g` is descriptor
set `g + 1`, with layouts built from what the pipeline's entry points use, sets allocated per
encoder and cached per bind group. Immediates are push constants.

Above `maxPerSetDescriptors` a layout is valid only where `vkGetDescriptorSetLayoutSupport`
accepts it, so mX halves the heap until the driver does. A device needs
`descriptorBindingStorageBufferUpdateAfterBind` and `shaderStorageBufferArrayNonUniformIndexing`
for the buffer heap, as it needs their sampled-image counterparts. MoltenVK answers by its
`maxPerSetDescriptors` of 1212 alone although its argument buffers handle much larger arrays,
so mX keeps the requested size there and says so in the log.

### Direct3D 12

One shader-visible CBV/SRV/UAV heap whose first `T` descriptors are the texture slots (SRVs of
texture views and of top-level acceleration structures), then the heap group's sampler index
buffer, then `B` raw SRVs and `B` raw UAVs of the buffer slots (a read-only view's UAV is a null
descriptor), then `T` UAVs of the texture slots (storage-texture views), then the ring of root
tables; and a shader-visible
sampler heap of 2048. Every root signature has a table over the texture slots
with one SRV range per typed array, all starting at descriptor 0 so the indices are shared.
The ranges are `T` long because naga's HLSL writer needs a size for a binding array. Samplers
go through naga's sampler heap and per-group sampler index buffers, the way wgpu-hal binds
them. Root group `g` is a descriptor table whose descriptors are copied per draw into a ring
in the same heap; immediates are root constants. Each buffer view a pipeline's stages use is
one more range of the heap table: `ByteAddressBuffer`s over the SRVs or `RWByteAddressBuffer`s
over the UAVs, `B` long, in a space of its own; so is each storage-texture view
(`RWTexture2D`s over the texture slots' UAVs) and the acceleration-structure array
(`RaytracingAccelerationStructure`s over the texture slots). A draw or dispatch whose pipeline
writes storage textures through the heap puts every texture with a storage view in the heap in
`UNORDERED_ACCESS` first; the next operation that does not returns them home.
`mx_shader::hlsl` has the register layout.

naga's HLSL writer did not support `binding_array`s of storage buffers, nor its validator
binding arrays of acceleration structures. mX uses naga 30.0.2 with those added (`vendor/naga`,
see its `MX_PATCHES.md`); shaders without them compile exactly as with the published crate.

## Shader compilation

Shaders are compiled the way wgpu-hal compiles them for the same API, so the same WGSL runs
the same code:

- naga's bounds-check policies as wgpu-hal sets them (`Restrict` on Metal),
- zeroed workgroup memory and bounded loops,
- on Metal, the newest MSL version the OS has (3.0 on macOS 13, 3.1 on 14, 3.2 from 15),
  `preserveInvariance`, and Metal's default fast math,
- on Direct3D 12, DXC with wgpu-hal's arguments, and FXC (shader model 5.1) when
  `dxcompiler.dll` is missing.

The parity tests found three differences from wgpu early on (bounds checks, the MSL version
and vertex fetch); all three are fixed and the tests guard against new ones.

## Synchronisation

mX keeps wgpu's guarantees: queue writes land before the next submission's commands, a pass
sees what earlier passes wrote, and no barriers are written by hand.

On Metal, hazards are tracked by Metal for the resources an encoder uses or declares. Textures
read through the heap are declared with `useResources`; see Residency below for which ones.

On Vulkan every image stays in `GENERAL` layout (swapchain images move to `PRESENT_SRC` to be
shown), so the heap's descriptors stay valid whatever a pass does. Which resources a pass reads
through the heap is not known to mX, so barriers are global memory barriers, but narrow ones:
every command (a render pass, a dispatch, a copy) says which pipeline stages it runs in and
which accesses it may make (`command::Op`), and the barrier before it waits only for the stages
that ran since the last barrier and makes visible only writes not yet visible to its stages
(`command::SyncState`). A render pass after a compute pass that wrote a buffer waits for
`COMPUTE_SHADER` writes in its vertex and fragment stages; a dispatch after one that wrote
nothing waits for its execution only; a copy after a copy waits for transfers. Pipelines know
whether their shaders write memory (storage buffers and textures, writable heap buffers), so a
pass whose pipelines write only attachments makes only attachment writes pending. Interop
through `CommandEncoder::raw` and acceleration-structure builds are treated as touching
everything.

On Metal, buffers in the heap are in the residency set (macOS 15 and later) or declared with
`useResources` by every encoder; buffers with a writable view are declared for writing by every
encoder, so Metal orders a pass's writes before the next pass's reads, and buffers written by a
copy are declared until that submission is done, as textures are.

On Vulkan the barriers above cover heap buffers too: a pass whose pipelines write heap buffers
leaves `SHADER_STORAGE_WRITE` pending for the next command.

On Direct3D 12 every resource has a home state: `PIXEL_SHADER_RESOURCE |
NON_PIXEL_SHADER_RESOURCE` for textures with `TEXTURE_BINDING`, so the heap can sample them in
any pass, and `COMMON` otherwise. A per-subresource tracker moves what a pass, copy, draw or
dispatch uses into the state it needs and returns textures home afterwards, so command lists
can be submitted in any order. A UAV barrier orders storage writes before the next draw or
dispatch. Before a draw or dispatch whose pipeline uses the buffer heap, a command list puts
every buffer with heap views in the state shaders reach it in (`UNORDERED_ACCESS` with a
writable view, both shader-resource states otherwise), again only when a buffer changed state
or the heap's buffers changed since the last time; buffers return to `COMMON` at the end of
the list as before.

Where the driver has enhanced barriers (`D3D12_OPTIONS12`), the tracker records each state
change as one (`ID3D12GraphicsCommandList7::Barrier`): buffer barriers carry synchronisation
and access only, texture barriers the layouts the states stand for, and the UAV barrier is a
global barrier of the shading stages. `MX_DX12_LEGACY_BARRIERS` keeps legacy transitions.

## GPU-driven draws

`IndirectDraws` is a batch of draws whose arguments and count the GPU writes, typically a
culling dispatch. `CommandEncoder::encode_indirect_draws` points it at the arguments and the
count outside passes; `RenderPass::execute_indirect_draws` draws it with the pass's pipeline
and bindings.

| Backend | Encode | Execute |
|---------|--------|---------|
| Metal | a kernel (MSL, compiled once per device) writes one indirect command buffer command per draw, and resets those past the count | `executeCommandsInBuffer`; the commands inherit the pipeline, the buffers and the other state, and every render pipeline supports indirect command buffers |
| Vulkan | nothing: the buffers are read in place | `vkCmdDraw[Indexed]IndirectCount` (`drawIndirectCount`) |
| Direct3D 12 | a dispatch lays the arguments out with naga's special constants in front, as the pipelines' command signatures read them | `ExecuteIndirect` with the count buffer |

### Zero initialisation

Textures are zeroed before the first submission after their creation, except for the mips
and layers that a queue write fills completely (`mx_types::Unwritten` tracks them). Render
targets and depth are cleared by a render pass, other textures by a copy from a buffer of
zeros. Vulkan fills new buffers with zeros. Metal's new buffers and Direct3D 12's committed
resources are zeroed by the system.

## Residency and staging

On macOS 15 and later the heap's views live in an `MTLResidencySet` that the queue keeps
resident. Encoders then declare with `useResources` only the textures that Metal's hazard
tracking must know about: render targets, storage textures and textures made outside mX at all
times; a texture written by a copy, a clear, a mipmap generation or zero initialisation until
the submission that wrote it has finished; and the textures the current command buffer copied
into. Recording 40 passes with 20000 textures in the heap takes about 0.5 ms this way and
112 ms when every encoder declares the whole heap, which is what happens on macOS 13 and 14
or with `MX_NO_RESIDENCY_SET` set.

Queue writes go through a staging ring on every backend: 4 MiB blocks of host-visible memory,
filled one after another and reused once the submission that read them is done. Writes larger
than 1 MiB get a buffer of their own.

## Upscalers and frame generation (`mx-upscale`)

`mx-upscale` drives the vendors' libraries on mX's objects through the backends' escape hatches
(`as_metal`, `as_dx12`, `as_vulkan`). No library is linked: MetalFX is opened by name, the
Windows DLLs are loaded from `Options::library_dir`. Upscalers record into command lists of
their own, submitted on mX's queue after the work submitted so far; the textures are handed
over in the states mX keeps them in between its commands (Direct3D 12: their home state;
Vulkan: `GENERAL`) and the vendors move them and back (XeSS, which moves nothing, gets its
barriers from `mx-upscale`). One Streamline session a process serves DLSS SR/RR and DLSS-G, with
frame tokens shared between them.

Frame generation comes in two kinds. MetalFX draws the window's picture into a texture of its
own and presents the made picture and then the drawn one, paced by a thread. DLSS-G, FSR 3 and
XeSS-FG make the window's swapchain: a surface takes a *swapchain maker*
(`mx_dx12::Surface::set_swapchain_maker`, `mx_vulkan::Surface::set_swapchain_maker`, reached
through `mx::Surface::with_dx12` / `with_vulkan`), used at its next `configure`; on Vulkan the
maker also hands back the functions to acquire and present with. mX's Vulkan queue lock is
reentrant, as AMD's swapchain submits from inside the present that holds it.

## Known platform behaviour

When several Metal devices in one process sample GPU counters at the same time, the M4
sometimes resolves both timestamps of a pass as 0. wgpu-hal samples the same way and has the
same problem. A game opens one device; the conformance tests take the GPU one at a time.

OpenGL is the bound tier: no bindless on the GPU, the heap lowered to bindings filled at each
draw ([BOUND_TIER.md](BOUND_TIER.md)).

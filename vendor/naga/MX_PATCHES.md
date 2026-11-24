# mX's changes to naga 30.0.2

This is naga 30.0.2 from crates.io with the changes below, used through `[patch.crates-io]` in
the workspace's `Cargo.toml`. Shaders that do not use these features compile exactly as with
the published crate.

## Binding arrays of storage buffers in the HLSL writer

naga validates `binding_array<T>` of storage buffers and writes it for SPIR-V and MSL, but its
HLSL writer hit `unreachable!()` on any access through one. mX's buffer heap needs it on
Direct3D 12.

- `back/hlsl/storage.rs`: `SubAccess::BindingArrayIndex` / `BindingArrayConstant` record which
  element of the array an access chain selects. `fill_access_chain` pushes one when it steps
  from a binding array to its element; `write_storage_address` skips them (they are not part of
  the byte address); the new `write_storage_var` writes the variable as `name[i]`, with
  `NonUniformResourceIndex` when the index is not uniform.
- Every `Load`, `Store` and `Interlocked*` on a storage buffer (`storage.rs`, `writer.rs`) writes
  its variable through `write_storage_var`.

`arrayLength` of an element is not supported; mX refuses it before naga sees it.

To be offered upstream (gfx-rs/wgpu).

## Samplers in their own registers below shader model 5.1

`back/hlsl/writer.rs`, `write_global_sampler`: naga writes every sampler as an element of a
2048-sampler heap indexed through a buffer, which shader model 5.0 (Direct3D 11) cannot
declare. Below 5.1 a sampler is written as `SamplerState name : register(sN);` at its binding's
register, as naga did before its sampler heaps. mX's Direct3D 11 backend (the bound tier)
needs it.

## Constant buffers below shader model 5.1

`back/hlsl/writer.rs`: the special constants (first vertex and instance) and the immediates
were written as `ConstantBuffer<T>`, which came with shader model 5.1. Below it they are written
as `cbuffer NameBuffer : register(bN) { T Name; }`, which reads the same.

## `textureLoad` from a depth texture in the GLSL writer

naga's GLSL writer refused every `textureLoad` from a depth texture (GLSL has no load from a
shadow sampler). openOMSI's screen-space clouds read the scene's depth so (`cloud_covered`),
and their pipelines failed on OpenGL.

- `back/glsl/writer.rs`, `depth_fetch_globals`: the depth textures (not multisampled) a module
  loads from and never samples. Such a texture is declared as a `sampler2D` (a depth texture
  without a comparison mode reads as one) instead of a `sampler2DShadow`.
- `write_image_load`: a load from one of those, or from a multisampled depth texture (already a
  `sampler2DMS`), is `texelFetch(..).x`, with the bounds-check policy of sampled images
  (`Restrict` writes its clamped level for depth textures too). A depth texture that is also
  sampled stays a shadow sampler, and a load from it is still an error.

## Binding arrays of acceleration structures

mX's heap holds top-level acceleration structures (`binding_array<acceleration_structure>`,
heap binding 11).

- `valid/analyzer.rs`: the capability check of an access into a binding array took every
  element that is neither an image nor a sampler for a buffer and reached `unreachable!()` on
  the `Handle` address space. An acceleration structure now asks for
  `ACCELERATION_STRUCTURE_BINDING_ARRAY`.
- `back/msl/writer.rs`: a binding array of acceleration structures was refused
  (`UnsupportedArrayOfType`). It is written as the other handle arrays are, an argument buffer of
  `instance_acceleration_structure` (MSL 2.4 and later), which Metal 3 fills with the
  structures' `gpuResourceID`s.

The SPIR-V and HLSL writers needed nothing: SPIR-V gets a runtime array of
`OpTypeAccelerationStructureKHR`, HLSL an array of `RaytracingAccelerationStructure` (sized,
as every heap array).

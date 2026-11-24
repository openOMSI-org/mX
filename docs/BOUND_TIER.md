# The bound tier

Backends for GPUs and drivers without working bindless: OpenGL (desktop 3.3 and 4.3, ES 3.0
and 3.1, with ANGLE over Direct3D 11 on Windows), Direct3D 11 and Metal 2. They run
the same WGSL and the same mX API as the bindless backends. This document is the design; the
roadmap places it as phase 5, done before the renderer port (phase 2) so that porting
openOMSI's renderer to mX's API leaves no player without a working game.

## What openOMSI needs from it

openOMSI on wgpu runs on GL today: through ANGLE on Windows when neither Direct3D 12 nor
Vulkan works, and on Linux drivers without Vulkan. On those devices it already picks downlevel
paths (`device_selection.rs`): lamps in textures instead of storage buffers when the fragment
stage has no storage, no ray tracing, no compute. The GL backend reports the same
capabilities, so those paths stay as they are:

| Capability | GL 3.3 / ES 3.0 | GL 4.3 / ES 3.1 |
|------------|-----------------|-----------------|
| Render pipelines, MSAA, instancing, occlusion queries | yes | yes |
| Storage buffers in vertex / fragment stages | no | as the driver reports (`MAX_*_SHADER_STORAGE_BLOCKS`) |
| Compute | no | yes |
| Indirect draws | GL 4.0 / ES 3.1 | yes |
| Timestamps | `GL_EXT_disjoint_timer_query` / GL 3.3 | same |
| Ray queries, GPU-driven batches | no | no |

## Binding model

There is no heap on the GPU. The heap stays where it is on the CPU (indices, deferred reuse,
the same API); shaders get fixed binding slots instead, filled at draw time.

### Lowering the heap (`mx_shader::bound`)

Before writing GLSL, mX rewrites the module:

1. Every access to a heap array, `mx_tex2d[i]`, `mx_texdepth_ms[i]`, `mx_samplers[j]`,
   `mx_buffers[k]`, a typed buffer view or a storage-texture view, whose index is a member of
   the immediates (`draw.tex`, as every
   bindless shader in mX's guide writes it), becomes a plain global of the element type at a
   slot of its own. The slot remembers where its index is: the byte offset of that member in
   the immediates.
2. At draw time the backend reads that offset from the immediates the pass set, looks the
   index up in the CPU heap and binds the view, sampler or buffer range to the slot.

Storage textures bind to image units on OpenGL (4.2 / ES 3.1 and later), to UAV slots on
Direct3D 11 and to texture slots on Metal 2; multisampled depth textures to texture units /
SRVs / texture slots. Acceleration structures are refused on this tier, in the heap as in root
groups (the tier has no ray queries).

A shader that indexes the heap with anything else (an index read from a buffer, computed, or
non-uniform) needs real bindless; creating its pipeline fails on this tier with an error that
names the access. Root groups (0 to 6) need no lowering: they map to GL binding points as wgpu
maps them.

Textures and samplers are separate in WGSL and combined in GLSL. naga's GLSL writer pairs
every texture with the samplers it is sampled with and reports the pairs; the backend gives
each pair a texture unit, and at draw time binds the pair's texture and sampler object to it.

### Immediates

GLSL has no push constants. As in wgpu-hal, the immediates block becomes a uniform block that
the backend fills with `glUniform*` calls before each draw that changed it.

## Commands

GL has one context per device and no command buffers. Command encoders record mX commands into
a list; `Queue::submit` replays the lists on the context, after the queue's pending writes. A
fence (`glFenceSync`) after each submission tells `Device::poll` when it is done, which drives
the heap's deferred reuse and the staging ring as on the other backends.

The context belongs to one thread at a time: a lock around it, made current for each operation
(creation, upload, submit), as wgpu-hal does.

## Platforms

| Platform | Context |
|----------|---------|
| Linux (X11, Wayland), Android | EGL (`khronos-egl`, loaded at run time) |
| Windows, ANGLE | EGL of the packaged `libEGL.dll` over Direct3D 11 |
| Windows, native GL | WGL (later) |

## Synchronisation and zeroing

GL orders commands itself; `glMemoryBarrier` follows passes and dispatches that wrote storage
buffers or images. New buffers and textures are zeroed as on the other backends (wgpu's
guarantee).

## Shading parity with wgpu

GLSL is written with naga's GLSL backend and the options wgpu-hal's GL backend uses: the GLSL
version of the context, `ADJUST_COORDINATE_SPACE`, bounds-check policies, zeroed workgroup
memory, and the same handling of the first vertex and first instance. The parity tests run on
GL as they run on Metal and Vulkan.

## Order of work

1. `mx_shader::bound` and `mx_shader::glsl`: the lowering and GLSL output, tested on every
   platform without a GPU.
2. `mx-gl`: EGL context, buffers, textures, samplers, the heap on the CPU, pipelines, render and
   compute passes, copies, queries, surfaces.
3. Device capabilities in `t::Limits` / `Adapter`, so a renderer picks its downlevel paths from
   mX as openOMSI picks them from wgpu.
4. The conformance tests and the parity tests on a GL device (Linux, or ANGLE on Windows).
5. WGL, then Direct3D 11 and Metal 2 on the same lowering.

## Status (2026-11-17)

Written, built for macOS, Linux and Windows, linted; not yet run on a GPU (this Mac has no EGL):

- `mx_shader::bound`: the lowering, with unit tests.
- `mx_shader::glsl`: GLSL output, its binding numbering and post-link bindings; unit tests
  compile the output with `glslangValidator`.
- `mx-gl` (`mx` with the `gl` feature): EGL context (ES 3.2 to 3.0, then desktop 4.6 to 3.3
  core), buffers, textures, views, samplers, the CPU heap, buffer views (GL 4.3 / ES 3.1),
  pipelines, render and compute passes, MSAA resolves, copies, queue writes, zeroing, render
  bundles, occlusion and timestamp queries, fences, window surfaces for X11, Windows (ANGLE)
  and Android.
- `Adapter::supports_compute` and `Adapter::storage_buffers_per_stage` on every backend, for a
  renderer's downlevel paths.

### Metal 2 (2026-11-18)

`mx-metal` runs a GPU without Metal 3 (an Intel or AMD Mac's, A11 to A13) on this tier: the
heap stays on the CPU and every heap access of a shader is lowered to a texture, sampler or
buffer slot (`msl::write_lowered`, the same `bound::lower`), bound at each draw from the
indices in the immediates. Metal binds those slots directly, so no argument buffer is needed;
GPU-driven batches and ray queries are refused (`Adapter::is_bound_tier`). Metal 3 GPUs use
the same direct binding for draws whose heap textures are chosen by immediates (it is faster
there too: see ARCHITECTURE.md). Tested on an M4 run as Metal 2 (`MX_METAL_BOUND=1`): the 28
conformance tests pass under Metal's validation (the GPU-driven one is skipped), and
openOMSI's renderer tests pass but those that need ray tracing.

### Direct3D 11 (2026-11-18)

`mx-dx11`, feature level 11.0 with `ID3D11DeviceContext1` (Windows 8 and newer): the heap on
the CPU (shader resource views, sampler states, buffer ranges), shaders lowered as for GL and
written as HLSL shader model 5.0 (`mx_shader::hlsl11`: registers per stage and class, the
immediates at `b13`, naga's first vertex and instance at `b12`), compiled with FXC. Two naga
patches make that possible (`vendor/naga/MX_PATCHES.md`): samplers as registers of their own
instead of naga's sampler heap, and `cbuffer` blocks instead of `ConstantBuffer<T>`. The
immediate context replays the command buffers; an event query after each submission drives the
deferred reuse. Uniform buffers that are also storage, vertex or index buffers get a second
resource for the constant buffer (Direct3D 11 binds constant buffers alone), refreshed before
draws that read them; copies between buffers and textures go through the CPU (Direct3D 11
has no such GPU copy). Compiled for `x86_64-pc-windows-gnu`; not run.

Not yet:

- Wayland windows (they need `wl_egl_window`), WGL (native GL on Windows without ANGLE).
- Running Direct3D 11 on Windows hardware.
- Known gaps against wgpu's GL backend: sample masks other than all samples (glow has no
  `glSampleMaski`); `Buffer::mapped_ptr` (GL maps only for one call: `map_read` / `map_write`
  instead); sampling one layer of an array texture as a 2D texture (needs texture views, which
  ES lacks); a non-zero base vertex below GL 3.2 / ES 3.2; reading back compressed textures.
Done the same day: one build holds every backend of the platform and the program picks one at
run time (ARCHITECTURE, Crates and backend selection), so a game offers Direct3D 12, Vulkan
and OpenGL as a setting, as openOMSI does on wgpu.

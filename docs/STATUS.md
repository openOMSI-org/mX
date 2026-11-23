# Status

What is implemented, and how far each part has been tested. Tested here means on an Apple M4
unless the row says otherwise.

## Backends

| Area | Metal | Vulkan | Direct3D 12 |
|------|-------|--------|-------------|
| Buffers, textures, views, samplers, the heap | tested | tested on MoltenVK | compiled |
| Buffer views in the heap (`mx_buffers`, typed views) | tested (2026-11-23, under Metal's API and shader validation) | tested on MoltenVK under the Khronos layer with synchronization validation, no findings (2026-11-23); passes on lavapipe in CI | conformance passes on WARP in CI |
| GPU-driven batches (`IndirectDraws`) | tested (2026-11-23) | passes on lavapipe in CI (MoltenVK has no draw counts: skipped there) | conformance passes on WARP in CI |
| Narrow barriers (Vulkan) / enhanced barriers (Direct3D 12) | | written, not yet run (unit-tested state machine) | compiled |
| Render and compute pipelines, root bind groups, immediates, copies | tested | tested on MoltenVK | compiled |
| MSAA resolve, render bundles, indirect and multi-draw indirect | tested | tested on MoltenVK | compiled |
| Timestamp and occlusion queries | tested | tested on MoltenVK | compiled |
| Surfaces and swapchains | tested | tested on MoltenVK (Win32, Xlib, XCB, Wayland and Android surfaces compiled) | compiled (DXGI flip model) |
| Zero initialisation, `clear_texture` | tested | tested on MoltenVK | compiled (`clear_texture`; committed resources start zeroed) |
| Workgroup memory | tested | tested on MoltenVK | compiled |
| Staging ring for queue writes | tested | tested on MoltenVK | compiled |
| Ray queries, refits | tested | compiled (MoltenVK has no ray queries) | compiled (DXR 1.1) |
| Native handles, `texture_from_raw` | tested (MetalFX in openOMSI) | compiled | compiled |
| Residency set (macOS 15+) | tested | | |

"Compiled" means built and linted in CI but never run on a GPU.

The no-op backend (`mx-noop`): written and compiled; openOMSI's dedicated server and its
renderer tests open it.

OpenGL / OpenGL ES (`mx-gl`, the bound tier): the 30 conformance tests pass on ANGLE over
Metal (OpenGL ES 3.0, the level ANGLE gives older chips; 2026-11-23), compute and the buffer
heap skipped there for want of ES 3.1, and in CI on Mesa's llvmpipe through EGL, as OpenGL ES
3.2 and as desktop OpenGL 4.5 core. Not yet run on ANGLE over Direct3D 11 or a GPU's own GL
driver. openOMSI's renderer is made on ANGLE (ES 3.0) and 48 of its 90 GPU tests pass there;
the others are not yet sorted into what OpenGL's paths leave out and what mX draws wrong. See [BOUND_TIER.md](BOUND_TIER.md#status-2026-11-17).

## Test suites

| Suite | Result |
|-------|--------|
| Conformance tests, 30 (`crates/mx/tests/conformance.rs`) | pass on Metal (2026-11-23, under Metal's API and shader validation; earlier with and without the residency set and with direct heap textures); pass on MoltenVK under the Khronos layer with synchronization validation (2026-11-23); pass in CI on lavapipe, WARP and llvmpipe's OpenGL |
| Pixel parity with wgpu, 7 scenes and 21 of openOMSI's full-screen passes | identical bytes on Metal and on MoltenVK |
| openOMSI's 23 WGSL modules of its mX renderer (`crates/mx/tests/openomsi`, openOMSI 940ab2f97) | compile through DXC (shader model 6.8) on Direct3D 12 (WARP, in CI), on lavapipe and MoltenVK under the Khronos layer with no findings, and on Metal (2026-11-23) |
| openOMSI's 17 OpenGL modules (`crates/mx/tests/openomsi_gl`, its `gl_backend()` variants) | every vertex and fragment entry point compiles as GLSL through the driver on ANGLE over Metal (OpenGL ES 3.0, 2026-11-23) and in CI on llvmpipe (ES 3.2, GL 4.5 core); the native set compiles on Metal 2 (`MX_METAL_BOUND=1`) |
| `wgpu-mx` through wgpu's API (`crates/wgpu-mx/tests`) | pass on Metal and MoltenVK |
| Khronos validation layer with synchronization validation (MoltenVK) | no synchronization findings; two known reports, below (2026-11-17, all 27 tests) |
| openOMSI's renderer tests on mX's native API (`omsi-render`, 2026-11-17) | 194 on the no-op device and 86 of 90 GPU tests on Metal pass; the 4 others fail the same way on wgpu (see openOMSI's `docs/MX_PORT.md`) |

## openOMSI on mX

openOMSI builds on `wgpu-mx` with two changes: its `wgpu` dependency points at `wgpu-mx`, and
MetalFX takes mX's native handles instead of wgpu-hal's. Grundorf runs clean under Metal's
validation layer with Enhanced graphics and with ray tracing. With Enhanced graphics it runs
at 104 fps against 82 fps for wgpu ([README](../README.md#performance) has the full numbers).

## Known reports from the Vulkan validation layer

- `VUID-vkCreateDescriptorSetLayout-support-09582` on MoltenVK only. MoltenVK rejects the heap
  layout because of its `maxPerSetDescriptors` of 1212, yet handles it. mX keeps the requested
  heap size on MoltenVK and shrinks it on drivers that actually reject the layout.
- `VUID-StandaloneSpirv-None-10684`. naga's SPIR-V output puts `ArrayStride` on arrays in
  function memory, for wgpu as well as for mX. Drivers accept it.

## Fixed while testing (2026-11-17)

- Metal: a heap buffer or texture written by `write_buffer` / `write_texture` (or
  `create_buffer_init`) before its first view was made was not declared to the encoders that
  read it with the residency set on, and was read before the copy landed (zeros in 4 runs of
  12). The heap now remembers writes by owner until their submission is done.
- Metal: every render pipeline was made with indirect command buffer support, which Metal
  refuses for some fragment shaders; the variant is now made when a GPU-driven batch first
  draws with the pipeline.
- A buffer or texture over the device's limit is `Error::Unsupported`, not out of memory, on
  every backend; `DeviceDescriptor::limits` caps a device's limits (a weaker chip's, to test
  on).

## Added (2026-11-17)

- `RenderPass::set_buffer_offset` / `RenderBundleEncoder::set_buffer_offset`: a root uniform
  buffer's offset per draw, on every backend (Metal: `set*BufferOffset`; Vulkan: dynamic
  uniform descriptors; Direct3D 12: a root CBV at the offset; GL: `glBindBufferRange`).
- Metal: direct heap textures (`msl::write_direct`, ARCHITECTURE.md). openOMSI's materials
  moved to the heap and its main pass runs as fast as with bound textures (Grundorf 1920x1080,
  M4, three runs each: 4.41-4.50 ms against 4.37-4.58 ms bound, 5.09-5.17 ms through the
  argument buffer).
- Surfaces take a swapchain maker on Direct3D 12 and Vulkan
  (`Surface::set_swapchain_maker`, `mx::Surface::with_dx12` / `with_vulkan`): the vendors'
  frame generation (DLSS-G, FSR 3, XeSS-FG) makes and presents the window's swapchain.

## mx-upscale (2026-11-18)

The vendors' upscalers and frame generators, on mX's devices, queues, textures and surfaces
(no wgpu-hal): one `Upscaler` and one `FrameGenerator` API over

| Vendor | Upscaling | Frame generation | State |
|--------|-----------|------------------|-------|
| Apple MetalFX | spatial, temporal | interpolation (macOS 26+) | temporal and interpolation run in openOMSI's GPU tests (M4); spatial written, not run |
| NVIDIA DLSS (Streamline 2.14.16) | SR, Ray Reconstruction | DLSS-G 2x-4x, with Reflex and PCL | compiled for Windows; not run |
| AMD FSR 3.1 (FidelityFX 2.4.0) | upscaling | frame generation (AMD's swapchain), D3D12 and Vulkan | compiled; not run |
| Intel XeSS 3.0.2 | SR, D3D12 and Vulkan | XeSS-FG with XeLL, D3D12 | compiled; not run |

The vendors' structures are transcribed from their public headers, sizes and offsets checked at
compile time against the headers' layout (not against the DLLs). Frame generation on Windows
makes the window's swapchain through the surface's maker. Open questions for the first run on
Windows: whether DLSS-G accepts mX's own command queue (Streamline's guide creates the queue
through its device proxy), and FSR 3 on Vulkan with one queue for all of AMD's roles
(submissions go through mX's lock, `submitFunc`).

## Bound tier: Metal 2 and Direct3D 11 (2026-11-18)

- Metal 2 GPUs run on the bound tier in `mx-metal` (`Adapter::is_bound_tier`); tested by running
  an M4 so (`MX_METAL_BOUND=1`): 28 conformance tests under Metal's validation, and openOMSI's
  renderer tests but those needing ray tracing.
- `mx-dx11`, Direct3D 11 (feature level 11.0, `ID3D11DeviceContext1`): compiled, not run.
  `Backends::DX11`, `MX_BACKENDS=dx11`. See BOUND_TIER.md.

## Shader cache on disk (2026-11-23)

Direct3D 12 (DXC or FXC) and Direct3D 11 (FXC) keep the bytecode they compile on disk,
keyed by the HLSL, entry point, profile, flags and the compiler's build
(`mx_shader::disk_cache`; `%LOCALAPPDATA%\mX\shaders`, `MX_SHADER_CACHE`). A second start
no longer compiles openOMSI's scene module again (some 15 s in DXC on WARP). The cache's
logic is unit-tested on every platform; CI on Windows runs openOMSI's shaders once into an
empty cache and once from it.

## Not done yet

- `examples/bench.rs` on Metal and MoltenVK.
- Running Vulkan on Linux and Android drivers.
- Running Direct3D 12 on Windows hardware, and compiling the generated HLSL with DXC.
- Comparing whole openOMSI frames (scene, sky, weather, ray tracing) with a wgpu build of the
  same source. The full-screen passes are compared; the 3D passes need real scenes.

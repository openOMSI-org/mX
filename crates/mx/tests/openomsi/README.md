# openOMSI's WGSL modules

The 23 shader modules of openOMSI's mX renderer (`omsi-render-mx`) as it assembles them for the
GPU, taken from openOMSI `940ab2f97` (2026-11-23): the list in `omsi-render-mx`'s
`shaders_validate_and_match_the_uniforms` test, each module written out as the test builds it.
The sky module holds the screen-space clouds, the scene module the glass and its mips, the
`*_basic` ones the phones' reduced precipitation.

`cargo test -p mx --test openomsi_shaders` compiles every entry point of every module for the
platform's GPU: DXC (or FXC) and Direct3D 12 on Windows - in CI on WARP, the first time the
game's HLSL meets DXC - Vulkan SPIR-V (under the Khronos layer in CI, lavapipe) and Metal.
`MX_OPENOMSI_SHADERS=<dir>` takes another set (`tools/openomsi_shaders.py` writes an older
one from an openOMSI checkout).

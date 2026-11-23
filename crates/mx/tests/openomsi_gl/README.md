# openOMSI's WGSL modules for OpenGL

The shader modules openOMSI's mX renderer (`omsi-render-mx`) makes on OpenGL, the bound tier,
as it assembles them there, taken from openOMSI `fc59155ad` (2026-11-23; its shaders as in
`940ab2f97`). On OpenGL (`gl_backend()`) openOMSI writes its modules differently from the
native set in `../openomsi`:

- the scene's per-draw arrays and its lamps are read from textures, not storage buffers
  (`ArrayPath::LightTextures`, what GLES 3.0 / GL 3.3 and ANGLE get; `scene_no_storage` is
  `ArrayPath::NoStorage`, without room for the lamps' textures);
- the precipitation's roof map is read as a float texture (`roof_as_float`), the tile masks
  through one sampler, no SSAO, ray tracing or tree fade in the scene;
- the `*_basic` modules are the reduced set of `basic_pipelines()` (a driver that failed the
  full one), on the no-storage path.

Left out because openOMSI makes none of their pipelines on OpenGL: SSAO, the lamps in the fog,
the volumetric fog, the puddles, the lamp sight, ray tracing and the temporal motion pass.

They were written by a test added to openOMSI for the purpose (not committed there): it set
`GL_BACKEND`, `ARRAY_PATH` and `BASIC_PIPELINES` as `Renderer::new` sets them on such a
device and wrote out `scene_shader_source(true)`, `sky_shader_source()`,
`corona_shader_source()`, `snow_shader_source()`, `rain_shader_source()`,
`fsr1_source_for(..)` and the plain `post`, `upscale`, `mip`, `xr_ui` and `overlay` files.

`cargo test -p mx --test openomsi_shaders` takes this set on a GL device: every vertex and
fragment entry point is written as GLSL and compiled by the driver (`Device::precompile`).
`MX_OPENOMSI_GL_SHADERS=<dir>` takes another set.

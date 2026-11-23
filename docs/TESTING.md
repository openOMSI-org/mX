# Testing

## Conformance tests

`crates/mx/tests/conformance.rs` renders and computes on the GPU and checks the pixels and
bytes it reads back. Every backend has to pass the same tests.

```sh
cargo test -p mx --test conformance                      # the platform's backend
MX_BACKENDS=vulkan cargo test -p mx --test conformance    # Vulkan (MoltenVK on macOS)
MX_BACKENDS=gl cargo test -p mx --test conformance        # OpenGL (EGL: Linux, ANGLE on Windows)
```

The tests take the GPU one at a time (see the Metal timestamp note in
[ARCHITECTURE.md](ARCHITECTURE.md#known-platform-behaviour)). Tests that need a feature the
device lacks, such as ray queries on MoltenVK, print `skipped` and pass.

A bug fix should come with a test here that fails without the fix.

## openOMSI's shaders

`crates/mx/tests/openomsi_shaders.rs` compiles every entry point of openOMSI's modules for the
device: the native set (`tests/openomsi`) on Metal, Vulkan, Direct3D 12 and the Metal 2 and
Direct3D 11 bound tiers, openOMSI's OpenGL variants (`tests/openomsi_gl`) on GL. On a Mac,
ANGLE's EGL from any Electron app gives OpenGL ES 3.0 over Metal:

```sh
cp "/Applications/Visual Studio Code.app/Contents/Frameworks/Electron Framework.framework/Versions/A/Libraries/"lib{EGL,GLESv2}.dylib /tmp/angle/
MX_EGL_LIBRARY=/tmp/angle/libEGL.dylib MX_BACKENDS=gl cargo test --release -p mx --test openomsi_shaders -- --nocapture
```

On Direct3D the compiled shaders are kept on disk between runs (`mx_shader::disk_cache`):
`MX_SHADER_CACHE=<folder>` puts them there, `MX_SHADER_CACHE=0` turns the cache off.

## Validation

Metal:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test -p mx
MX_NO_RESIDENCY_SET=1 MTL_DEBUG_LAYER=1 cargo test -p mx   # the macOS 13 and 14 path
```

Vulkan, with the Khronos layer (`brew install vulkan-validationlayers` on macOS):

```sh
export VK_LAYER_PATH=/opt/homebrew/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true
MX_BACKENDS=vulkan cargo test -p mx -- --test-threads=1
```

mX prints validation errors to stderr as `[mX Vulkan validation error]`. Two reports are known
and explained in [STATUS.md](STATUS.md#known-reports-from-the-vulkan-validation-layer).

## Parity with wgpu

`crates/mx-parity` draws the same scenes with wgpu (the revision openOMSI ships) and with mX,
and compares the bytes. A deliberate change, such as turning Metal's fast math off, makes the
shading scenes fail by one ulp, which shows the comparison is strict enough.

```sh
cargo test -p mx-parity
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib cargo test -p mx-parity --features vulkan
```

`openomsi_passes` also compares 21 of openOMSI's full-screen passes. It needs openOMSI's
shaders, which a script assembles from an openOMSI checkout:

```sh
python3 tools/openomsi_shaders.py <openOMSI checkout> <dir>
MX_OPENOMSI_SHADERS=<dir> cargo test -p mx-parity -p mx
```

With `MX_OPENOMSI_SHADERS` set, `crates/mx/tests/openomsi_shaders.rs` also compiles every
entry point of the 27 modules for each backend.

## wgpu-mx

`crates/wgpu-mx/tests/wgpu_api.rs` drives `wgpu-mx` through wgpu's own API, the way openOMSI
calls it: ray queries over two builds of one top-level structure, and `clear_texture`.

```sh
cargo test -p wgpu-mx
```

## Direct3D 12 from another host

```sh
rustup target add x86_64-pc-windows-msvc
cargo check -p mx-dx12 -p wgpu-mx --target x86_64-pc-windows-msvc
cargo clippy -p mx-dx12 --target x86_64-pc-windows-msvc
```

## Benchmarks

`crates/mx/tests/residency_cost.rs` measures the CPU time to record 40 render passes with
100 to 20000 textures in the heap. It is ignored by default:

```sh
cargo test -p mx --release --test residency_cost -- --ignored --nocapture
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | Runner | What it does |
|-----|--------|--------------|
| Format | Linux | `cargo fmt --check` |
| Linux (Vulkan) | Linux | clippy, the CPU-only tests, the conformance tests on Mesa's software Vulkan driver (lavapipe) with and without the Khronos validation layer, and `wgpu-mx`; the Android build is compiled |
| macOS (Metal, build) | macOS, Apple silicon | clippy and the CPU-only tests; the iOS build is compiled |
| Windows (Direct3D 12) | Windows | clippy, the CPU-only tests, and the conformance tests and `wgpu-mx` on WARP, the software Direct3D 12 driver |

The virtual GPU of GitHub's macOS runners lacks what mX needs, so the tests on Metal run on a
real Mac. lavapipe and WARP run on the processor: CI catches API and correctness errors but
says nothing about speed.

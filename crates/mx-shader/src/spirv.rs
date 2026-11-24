//! SPIR-V output and mX's Vulkan descriptor layout.
//!
//! | descriptor set | use |
//! |----------------|-----|
//! | 0              | the heap: binding `b` is `heap::*` (runtime arrays, UPDATE_AFTER_BIND, PARTIALLY_BOUND); every view of the buffer heap is binding `heap::BUFFERS`, one array of storage buffers; every view of the storage textures binding `heap::FIRST_STORAGE_TEXTURE`, one array of storage images; `heap::ACCELERATION_STRUCTURES` an array of acceleration structures |
//! | 1..=7          | root group `g` is set `g + 1`; bindings keep their WGSL numbers |
//!
//! `var<immediate>` is the push-constant block (offset 0, every stage of a pipeline).
//!
//! Shaders are written as wgpu-hal writes them for Vulkan: the SPIR-V version of the device's
//! Vulkan version, `LABEL_VARYINGS` and `FORCE_POINT_SIZE` (no coordinate-space adjustment:
//! the backend flips the viewport instead, as wgpu does), zeroed workgroup memory, bounded
//! loops, ray-query initialisation tracking and mX's bounds-check policies.

use crate::{HEAP_GROUP, Module, RootBinding, ShaderError, heap};
use naga::back::spv;

/// The descriptor set of the bindless heap.
pub const HEAP_SET: u32 = 0;

/// The descriptor binding of WGSL heap binding `binding`: buffer views all alias the one array
/// of storage buffers (Vulkan lets several variables share a binding of the same type).
/// Likewise every view of the storage textures is binding `heap::FIRST_STORAGE_TEXTURE`, one
/// array of storage images.
pub const fn heap_binding(binding: u32) -> u32 {
    if heap::is_buffer(binding) {
        heap::BUFFERS
    } else if heap::is_storage_texture(binding) {
        heap::FIRST_STORAGE_TEXTURE
    } else {
        binding
    }
}

/// The descriptor set of root group `group`.
pub const fn root_set(group: u32) -> u32 {
    group + 1
}

/// What the device offers, for the writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// SPIR-V version: (1, 5) for Vulkan 1.2, (1, 6) for 1.3 and later (as wgpu-hal).
    pub lang_version: (u8, u8),
    /// `VK_KHR_ray_query` is enabled: WGSL ray queries may be used.
    pub ray_query: bool,
    /// The device zeroes workgroup memory itself (`shaderZeroInitializeWorkgroupMemory`);
    /// else the shader does.
    pub native_zero_init: bool,
    /// Name varyings in the output (wgpu-hal does except on Qualcomm, whose compiler trips on it).
    pub label_varyings: bool,
    /// Emit debug names (`OpName`) for everything.
    pub debug: bool,
    /// The device enabled `robustBufferAccess2`: buffer accesses are left to it, unchecked.
    pub robust_buffer_access2: bool,
    /// The device enabled robust image access (`robustImageAccess2` where `VK_EXT_robustness2`
    /// is there, else `robustImageAccess`): texel loads are left to it, unchecked.
    pub robust_image_access: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            lang_version: (1, 5),
            ray_query: false,
            native_zero_init: false,
            label_varyings: true,
            debug: false,
            robust_buffer_access2: false,
            robust_image_access: false,
        }
    }
}

impl Options {
    /// The bounds checks wgpu-hal's Vulkan backend writes for a checked module: `Restrict`,
    /// except what the device's robustness features already cover.
    pub fn bounds_check_policies(&self, m: &Module) -> naga::proc::BoundsCheckPolicies {
        use naga::proc::BoundsCheckPolicy as P;
        let mut p = m.bounds_check_policies();
        if self.robust_buffer_access2 {
            p.buffer = P::Unchecked;
        }
        if self.robust_image_access {
            p.image_load = P::Unchecked;
        }
        p
    }
}

/// One entry point as SPIR-V.
#[derive(Clone, Debug)]
pub struct Output {
    pub words: Vec<u32>,
    /// The entry point's name in `words`.
    pub entry_point: String,
    /// Which heap bindings the entry point reads (bit per `heap::*` and buffer view).
    pub heap_bindings_used: u64,
    pub uses_immediates: bool,
    /// The root bindings this entry point uses, sorted by (group, binding).
    pub roots: Vec<RootBinding>,
    pub workgroup_size: [u32; 3],
}

/// The capabilities a Vulkan 1.2 device with mX's required features offers, plus the
/// optional ones (`Float16`, `Int64`, ...) the driver checks when the module is created.
fn capabilities(options: &Options) -> naga::FastHashSet<spv::Capability> {
    use spv::Capability as C;
    let mut caps: naga::FastHashSet<C> = [
        C::Shader,
        C::Matrix,
        C::Sampled1D,
        C::Image1D,
        C::ImageQuery,
        C::DerivativeControl,
        C::SampledCubeArray,
        C::ImageCubeArray,
        C::SampleRateShading,
        C::StorageImageExtendedFormats,
        C::StorageImageWriteWithoutFormat,
        C::ShaderNonUniform,
        C::RuntimeDescriptorArray,
        C::SampledImageArrayNonUniformIndexing,
        C::StorageBufferArrayNonUniformIndexing,
        C::ClipDistance,
        C::CullDistance,
        C::DrawParameters,
        C::MultiView,
        C::GroupNonUniform,
        C::GroupNonUniformVote,
        C::GroupNonUniformArithmetic,
        C::GroupNonUniformBallot,
        C::GroupNonUniformShuffle,
        C::GroupNonUniformShuffleRelative,
        C::GroupNonUniformQuad,
        C::Int8,
        C::Int16,
        C::Int64,
        C::Float16,
        C::Float64,
        C::StorageBuffer16BitAccess,
        C::UniformAndStorageBuffer16BitAccess,
        C::DotProduct,
        C::DotProductInputAll,
        C::DotProductInput4x8Bit,
        C::DotProductInput4x8BitPacked,
    ]
    .into_iter()
    .collect();
    if options.ray_query {
        caps.insert(C::RayQueryKHR);
    }
    caps
}

/// The binding map of a module: the heap in set 0, root group `g` in set `g + 1`.
pub fn binding_map(m: &Module) -> spv::BindingMap {
    let mut map = spv::BindingMap::new();
    // (the prelude's, the shader's buffer views, storage textures and acceleration structures)
    let declared = m
        .module
        .global_variables
        .iter()
        .filter_map(|(_, v)| v.binding)
        .filter(|b| b.group == HEAP_GROUP && b.binding <= heap::LAST_BUFFER_VIEW);
    for b in declared.map(|b| b.binding) {
        map.insert(
            naga::ResourceBinding { group: HEAP_GROUP, binding: b },
            spv::BindingInfo {
                descriptor_set: HEAP_SET,
                binding: heap_binding(b),
                binding_array_size: None,
            },
        );
    }
    for r in &m.roots {
        map.insert(
            naga::ResourceBinding { group: r.group, binding: r.binding },
            spv::BindingInfo {
                descriptor_set: root_set(r.group),
                binding: r.binding,
                binding_array_size: None,
            },
        );
    }
    map
}

const OP_CAPABILITY: u32 = 17;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_POINTER: u32 = 32;
const STORAGE_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_STORAGE_BUFFER: u32 = 12;
const CAPABILITY_RUNTIME_DESCRIPTOR_ARRAY: u32 = 5302;

/// naga writes the heap's unbounded `binding_array`s as runtime arrays of descriptors but does
/// not declare `RuntimeDescriptorArray`, which Vulkan requires for them (core in SPIR-V 1.5;
/// the device feature is `runtimeDescriptorArray`). Declare it where needed: for the texture
/// and sampler arrays (`UniformConstant`) and for the buffer heap (`StorageBuffer`).
fn declare_runtime_descriptor_arrays(words: &mut Vec<u32>) {
    let mut runtime_arrays = Vec::new();
    let mut needed = false;
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        if len == 0 || i + len > words.len() {
            return;
        }
        match op {
            OP_CAPABILITY if words[i + 1] == CAPABILITY_RUNTIME_DESCRIPTOR_ARRAY => return,
            OP_TYPE_RUNTIME_ARRAY => runtime_arrays.push(words[i + 1]),
            OP_TYPE_POINTER if matches!(words[i + 2], STORAGE_UNIFORM_CONSTANT | STORAGE_STORAGE_BUFFER) && runtime_arrays.contains(&words[i + 3]) => needed = true,
            _ => {}
        }
        i += len;
    }
    if needed {
        words.splice(5..5, [(2 << 16) | OP_CAPABILITY, CAPABILITY_RUNTIME_DESCRIPTOR_ARRAY]);
    }
}

/// Write one entry point of `m` as SPIR-V, with `constants` applied to its `override`s.
pub fn write(m: &Module, stage: naga::ShaderStage, entry: Option<&str>, constants: &[(&str, f64)], options: &Options) -> Result<Output, ShaderError> {
    let name = m.entry_point(stage, entry)?.name.clone();
    let (module, info) = m.specialize(stage, &name, constants)?;
    let backend_err = |message: String| ShaderError::Backend {
        label: m.label.clone(),
        target: "SPIR-V",
        message,
    };

    // (wgpu-hal starts from no flags: no ADJUST_COORDINATE_SPACE, no CLAMP_FRAG_DEPTH)
    let mut flags = spv::WriterFlags::FORCE_POINT_SIZE;
    flags.set(spv::WriterFlags::LABEL_VARYINGS, options.label_varyings);
    flags.set(spv::WriterFlags::DEBUG, options.debug);
    let spv_options = spv::Options {
        lang_version: options.lang_version,
        flags,
        fake_missing_bindings: false,
        binding_map: binding_map(m),
        capabilities: Some(capabilities(options)),
        bounds_check_policies: options.bounds_check_policies(m),
        zero_initialize_workgroup_memory: if options.native_zero_init {
            spv::ZeroInitializeWorkgroupMemoryMode::Native
        } else {
            spv::ZeroInitializeWorkgroupMemoryMode::Polyfill
        },
        force_loop_bounding: true,
        ray_query_initialization_tracking: true,
        // (f16 inputs and outputs go through f32, which every device takes)
        use_storage_input_output_16: false,
        debug_info: None,
        ..Default::default()
    };
    let pipeline = spv::PipelineOptions {
        shader_stage: stage,
        entry_point: name.clone(),
    };
    let mut words = spv::write_vec(&module, &info, &spv_options, Some(&pipeline)).map_err(|e| backend_err(e.to_string()))?;
    declare_runtime_descriptor_arrays(&mut words);

    let (ep_index, ep) = module
        .entry_points
        .iter()
        .enumerate()
        .find(|(_, e)| e.stage == stage && e.name == name)
        .ok_or_else(|| backend_err("entry point lost".into()))?;
    let ep_info = info.get_entry_point(ep_index);
    let heap_bindings_used = crate::heap_used(&module, |h| !ep_info[h].is_empty());
    let uses_immediates = module.global_variables.iter().any(|(h, v)| v.space == naga::AddressSpace::Immediate && !ep_info[h].is_empty());
    Ok(Output {
        words,
        roots: m.entry_point_roots(stage, &name)?,
        entry_point: name,
        heap_bindings_used,
        uses_immediates,
        workgroup_size: ep.workgroup_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RootKind;

    /// Run `spirv-val` on the words, where it is installed. Under Vulkan rules first; naga
    /// (as used by wgpu) still puts explicit layouts on types it shares between buffers and
    /// workgroup / function memory, which recent validators reject (VUID-10684) and drivers
    /// take: for that one finding, validate again under plain SPIR-V rules.
    fn validate(words: &[u32], env: &str) {
        let candidates = ["spirv-val", "/opt/homebrew/bin/spirv-val", "/usr/local/bin/spirv-val", "/usr/bin/spirv-val"];
        let Some(tool) = candidates.iter().find(|c| std::process::Command::new(c).arg("--version").output().is_ok()) else {
            eprintln!("spirv-val not found; skipping validation");
            return;
        };
        let dir = std::env::temp_dir().join(format!("mx-spirv-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shader.spv");
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        let run = |env: &str| {
            let out = std::process::Command::new(tool).args(["--target-env", env]).arg(&path).output().unwrap();
            (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
        };
        let (mut ok, mut msg) = run(env);
        if !ok && msg.contains("VUID-StandaloneSpirv-None-10684") {
            (ok, msg) = run(if env == "vulkan1.2" { "spv1.5" } else { "spv1.6" });
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(ok, "spirv-val: {msg}");
    }

    /// The (set, binding) decorations of the variable named `id_name` (by its `OpName`).
    fn decoration(words: &[u32], id_name: &str) -> (Option<u32>, Option<u32>) {
        let mut ids = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
            let ops = &words[i + 1..i + len];
            if op == 5 {
                let bytes: Vec<u8> = ops[1..].iter().flat_map(|w| w.to_le_bytes()).take_while(|b| *b != 0).collect();
                if bytes == id_name.as_bytes() {
                    ids.push(ops[0]);
                }
            }
            i += len.max(1);
        }
        let (mut s, mut b) = (None, None);
        let mut i = 5;
        while i < words.len() {
            let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
            let ops = &words[i + 1..i + len];
            if op == 71 && ids.contains(&ops[0]) {
                match ops[1] {
                    34 => s = Some(ops[2]),
                    33 => b = Some(ops[2]),
                    _ => {}
                }
            }
            i += len.max(1);
        }
        (s, b)
    }

    fn debug() -> Options {
        Options { debug: true, ..Default::default() }
    }

    #[test]
    fn heap_and_roots_land_in_their_sets() {
        let m = Module::from_wgsl(
            "t",
            "struct C { k: vec4<f32> }\n@group(0) @binding(0) var<uniform> c: C;\n@group(2) @binding(1) var<storage, read> xs: array<f32>;\n@group(2) @binding(4) var<storage, read> unused: array<f32>;\nstruct D { t: u32 }\nvar<immediate> d: D;\n@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return c.k * xs[0] * textureSample(mx_tex2d[d.t], mx_samplers[0], uv) * textureSampleCompare(mx_texdepth[1], mx_samplers_cmp[1], uv, 0.5); }",
        )
        .unwrap();
        let out = write(&m, naga::ShaderStage::Fragment, None, &[], &debug()).unwrap();
        validate(&out.words, "vulkan1.2");
        assert_eq!(decoration(&out.words, "mx_tex2d"), (Some(HEAP_SET), Some(heap::TEX_2D)));
        assert_eq!(decoration(&out.words, "mx_samplers_cmp"), (Some(HEAP_SET), Some(heap::SAMPLERS_CMP)));
        assert_eq!(decoration(&out.words, "c"), (Some(1), Some(0)));
        assert_eq!(decoration(&out.words, "xs"), (Some(3), Some(1)));
        assert_eq!(out.heap_bindings_used, 1 << heap::TEX_2D | 1 << heap::SAMPLERS | 1 << heap::TEX_DEPTH | 1 << heap::SAMPLERS_CMP);
        assert!(out.uses_immediates);
        let roots: Vec<_> = out.roots.iter().map(|r| (r.group, r.binding, r.kind)).collect();
        assert_eq!(roots, vec![(0, 0, RootKind::UniformBuffer), (2, 1, RootKind::StorageBuffer { read_only: true })]);
    }

    #[test]
    fn every_heap_array_validates() {
        let m = Module::from_wgsl(
            "t",
            "struct D { i: u32 }\nvar<immediate> d: D;\n@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                let s = mx_samplers[d.i];
                var c = textureSample(mx_tex2d[d.i], s, uv);
                c += textureSample(mx_tex2d_array[d.i], s, uv, 1);
                c += textureSample(mx_texcube[d.i], s, vec3<f32>(uv, 1.0));
                c += textureSample(mx_tex3d[d.i], s, vec3<f32>(uv, 0.5));
                c += vec4<f32>(textureSampleCompare(mx_texdepth[d.i], mx_samplers_cmp[d.i], uv, 0.5));
                c += vec4<f32>(textureLoad(mx_tex2d_u32[d.i], vec2<i32>(0), 0));
                c += vec4<f32>(textureLoad(mx_texdepth_ms[d.i], vec2<i32>(0), 1));
                c += vec4<f32>(f32(mx_buffers[d.i].w[1]));
                mx_rw_buffers[d.i].w[0] = 1u;
                return c;
            }",
        )
        .unwrap();
        for (v, env) in [((1, 5), "vulkan1.2"), ((1, 6), "vulkan1.3")] {
            let out = write(
                &m,
                naga::ShaderStage::Fragment,
                None,
                &[],
                &Options {
                    lang_version: v,
                    ..Default::default()
                },
            )
            .unwrap();
            validate(&out.words, env);
            assert_eq!(out.heap_bindings_used, (1 << heap::COUNT) - 1);
        }
    }

    #[test]
    fn storage_textures_and_acceleration_structures_in_the_heap() {
        let m = Module::from_wgsl(
            "t",
            "enable wgpu_ray_query;
            @group(7) @binding(11) var mx_tlas: binding_array<acceleration_structure>;
            @group(7) @binding(12) var out16: binding_array<texture_storage_2d<rgba16float, write>>;
            @group(7) @binding(15) var out8: binding_array<texture_storage_2d<rgba8unorm, write>>;
            struct D { a: u32, o: u32, p: u32 }
            var<immediate> d: D;
            @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                var rq: ray_query;
                rayQueryInitialize(&rq, mx_tlas[d.a], RayDesc(0u, 0xFFu, 0.0, 10.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0)));
                rayQueryProceed(&rq);
                let h = f32(rayQueryGetCommittedIntersection(&rq).instance_custom_data);
                textureStore(out16[d.o], vec2<i32>(id.xy), vec4<f32>(h));
                textureStore(out8[d.p], vec2<i32>(id.xy), vec4<f32>(0.5));
            }",
        )
        .unwrap();
        let out = write(
            &m,
            naga::ShaderStage::Compute,
            None,
            &[],
            &Options {
                lang_version: (1, 5),
                ray_query: true,
                debug: true,
                ..Default::default()
            },
        )
        .unwrap();
        validate(&out.words, "vulkan1.2");
        assert_eq!(decoration(&out.words, "mx_tlas"), (Some(HEAP_SET), Some(heap::ACCELERATION_STRUCTURES)));
        // (every view of the storage textures is the one array of storage images)
        assert_eq!(decoration(&out.words, "out16"), (Some(HEAP_SET), Some(heap::FIRST_STORAGE_TEXTURE)));
        assert_eq!(decoration(&out.words, "out8"), (Some(HEAP_SET), Some(heap::FIRST_STORAGE_TEXTURE)));
        assert_eq!(
            out.heap_bindings_used,
            1 << heap::ACCELERATION_STRUCTURES | 1 << heap::FIRST_STORAGE_TEXTURE | 1 << heap::LAST_STORAGE_TEXTURE
        );
    }

    #[test]
    fn overrides_and_compute() {
        let m = Module::from_wgsl(
            "t",
            "override K: u32 = 1u;\nvar<workgroup> tmp: array<u32, 64>;\n@group(0) @binding(0) var<storage, read_write> xs: array<u32>;\n@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) i: u32) { tmp[i] = xs[i] * K; workgroupBarrier(); xs[i] = tmp[63u - i] + arrayLength(&xs); for (var j = 0u; j < xs[0]; j++) { xs[1] += 1u; } }",
        )
        .unwrap();
        for native_zero_init in [false, true] {
            let out = write(
                &m,
                naga::ShaderStage::Compute,
                Some("main"),
                &[("K", 3.0)],
                &Options {
                    native_zero_init,
                    ..Default::default()
                },
            )
            .unwrap();
            validate(&out.words, "vulkan1.2");
            assert_eq!(out.workgroup_size, [64, 1, 1]);
            assert!(!out.uses_immediates);
            assert_eq!(out.roots.len(), 1);
        }
    }

    #[test]
    fn only_the_entry_points_roots_are_listed() {
        let m = Module::from_wgsl(
            "t",
            "@group(1) @binding(0) var img: texture_storage_2d<rgba8unorm, write>;\n@compute @workgroup_size(8, 8) fn cs(@builtin(global_invocation_id) id: vec3<u32>) { textureStore(img, id.xy, vec4<f32>(1.0)); }\n@vertex fn vs(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }",
        )
        .unwrap();
        let cs = write(&m, naga::ShaderStage::Compute, None, &[], &Options::default()).unwrap();
        validate(&cs.words, "vulkan1.2");
        assert_eq!(cs.roots.len(), 1);
        let vs = write(&m, naga::ShaderStage::Vertex, None, &[], &Options::default()).unwrap();
        validate(&vs.words, "vulkan1.2");
        assert!(vs.roots.is_empty());
    }

    #[test]
    fn ray_queries_need_the_capability() {
        let src = "enable wgpu_ray_query;\n@group(0) @binding(1) var acc: acceleration_structure;\n@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(1) fn main() { var rq: ray_query; rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 10.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0))); rayQueryProceed(&rq); out[0] = rayQueryGetCommittedIntersection(&rq).instance_custom_data; }";
        let m = Module::from_wgsl("t", src).unwrap();
        assert!(write(&m, naga::ShaderStage::Compute, None, &[], &Options::default()).is_err());
        let out = write(&m, naga::ShaderStage::Compute, None, &[], &Options { ray_query: true, ..debug() }).unwrap();
        validate(&out.words, "vulkan1.2");
        assert_eq!(decoration(&out.words, "acc"), (Some(1), Some(1)));
    }

    /// Every entry point of openOMSI's modules (`MX_OPENOMSI_SHADERS`, see
    /// `tools/openomsi_shaders.py`) as valid Vulkan 1.3 SPIR-V.
    #[test]
    fn openomsi_modules_are_valid_spirv() {
        let Ok(dir) = std::env::var("MX_OPENOMSI_SHADERS") else {
            eprintln!("skipped: set MX_OPENOMSI_SHADERS");
            return;
        };
        let options = Options {
            lang_version: (1, 6),
            ray_query: true,
            native_zero_init: true,
            robust_image_access: true,
            ..Default::default()
        };
        let mut n = 0;
        for e in std::fs::read_dir(&dir).unwrap() {
            let path = e.unwrap().path();
            if path.extension().is_none_or(|x| x != "wgsl") {
                continue;
            }
            let label = path.file_stem().unwrap().to_string_lossy().into_owned();
            let m = Module::from_wgsl(&label, &std::fs::read_to_string(&path).unwrap()).unwrap_or_else(|e| panic!("{e}"));
            for ep in &m.module.entry_points {
                let out = write(&m, ep.stage, Some(&ep.name), &[], &options).unwrap_or_else(|e| panic!("{label}::{}: {e}", ep.name));
                validate(&out.words, "vulkan1.3");
                n += 1;
            }
        }
        eprintln!("{n} entry points valid");
    }
}

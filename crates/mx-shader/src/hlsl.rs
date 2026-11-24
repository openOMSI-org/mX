//! HLSL output and mX's Direct3D 12 register layout.
//!
//! Registers are fixed by `(group, binding)`, so every entry point of every module agrees on
//! them and a root signature can be built from the union of what a pipeline's stages use:
//!
//! | space | registers | use |
//! |-------|-----------|-----|
//! | 0..=6 | `b{binding}` / `t{binding}` / `u{binding}` | root group `space`: uniform buffers (CBV), sampled textures, read-only storage and acceleration structures (SRV), storage buffers and textures (UAV) |
//! | 7     | `t{group}` | the sampler index buffer of group `group` (0..=7; 7 is the heap's) |
//! | 8..=14 | `t0` .. `t{heap size - 1}` | heap texture binding `space - 8` (`heap::TEX_2D` .. `heap::TEX_DEPTH_MS`) |
//! | 15    | `b0` / `b1` | immediates (root constants) / naga's special constants (first vertex, first instance, workgroup count) |
//! | 16 + `b` | `t0` .. / `u0` .. | view `b` of the buffer heap (`heap::BUFFERS`, `heap::RW_BUFFERS`, a shader's own): `ByteAddressBuffer`s when read-only, `RWByteAddressBuffer`s when written |
//! | 27    | `t0` .. `t{heap size - 1}` | `heap::ACCELERATION_STRUCTURES`: `RaytracingAccelerationStructure`s over the texture slots |
//! | 28..=31 | `u0` .. `u{heap size - 1}` | a shader's views of the storage textures (`heap::FIRST_STORAGE_TEXTURE + space - 28`): `RWTexture2D`s |
//! | 80    | `s0` / `s2048` | naga's sampler heaps (`nagaSamplerHeap`, `nagaComparisonSamplerHeap`), 2048 each |
//!
//! Buffers: every buffer view in the heap has a raw SRV and a raw UAV in the shader-visible
//! CBV/SRV/UAV heap, after the texture slots. Each view a shader declares gets its own space,
//! and the root signature points all of them at the same descriptors.
//!
//! Texture slots hold an SRV each: a texture view's, or a top-level acceleration structure's
//! (they share the index space). A view with `STORAGE_BINDING` also has a UAV at its index in
//! a second range of `heap size` descriptors, which every storage-texture view of a shader
//! reads.
//!
//! Samplers: naga's HLSL backend (as wgpu-hal uses it) reaches every sampler through one
//! shader-visible sampler heap and a per-group `StructuredBuffer<uint>` of heap indices. For a
//! root sampler the index buffer is indexed by its binding number; the heap's group (7) has an
//! identity index buffer, so `mx_samplers[i]` is sampler heap slot `i`.
//!
//! The heap's texture arrays are sized to the device's heap (naga's HLSL writer cannot
//! declare an unbounded array; wgpu-hal always passes a count too), so the HLSL of a module
//! depends on the heap size (`DeviceDescriptor::bindless_textures`) passed to [`write`].

use crate::{HEAP_GROUP, MAX_ROOT_GROUP, Module, RootBinding, RootKind, ShaderError, heap};
use naga::back::hlsl;

pub use hlsl::ShaderModel;

/// The space of the sampler index buffers (`t{group}`).
pub const SAMPLER_INDEX_SPACE: u32 = 7;
/// The space of heap binding 0; heap texture binding `b` is in `HEAP_SPACE + b`.
pub const HEAP_SPACE: u32 = 8;
/// The space of the sampler heaps.
pub const SAMPLER_HEAP_SPACE: u32 = 80;
/// The register of the comparison sampler heap in [`SAMPLER_HEAP_SPACE`].
pub const COMPARISON_SAMPLER_REGISTER: u32 = 2048;
/// How many samplers naga's sampler heap declarations hold (and so the sampler heap).
pub const SAMPLER_HEAP_SIZE: u32 = 2048;
/// The space of the immediates (`b0`) and the special constants (`b1`).
pub const IMMEDIATES_SPACE: u32 = 15;
pub const IMMEDIATES_REGISTER: u32 = 0;
pub const SPECIAL_CONSTANTS_REGISTER: u32 = 1;

/// Uniform buffers one entry point may use: resource binding tier 2 takes 14 CBVs per stage,
/// and the immediates and naga's special constants are two of them.
pub const MAX_UNIFORM_BUFFERS: u32 = 12;

/// The space of buffer view 0 of the heap; view `b` is in `BUFFER_SPACE + b`.
pub const BUFFER_SPACE: u32 = 16;

/// The space of heap texture binding `binding`, or of any other heap binding (buffer views,
/// acceleration structures, storage textures).
pub const fn heap_space(binding: u32) -> u32 {
    if heap::is_texture(binding) { HEAP_SPACE + binding } else { BUFFER_SPACE + binding }
}

/// How large the heap's arrays are declared (naga's HLSL writer needs a length).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapSizes {
    pub textures: u32,
    pub buffers: u32,
}

fn target(space: u32, register: u32) -> hlsl::BindTarget {
    hlsl::BindTarget {
        space: space as u8,
        register,
        binding_array_size: None,
        dynamic_storage_buffer_offsets_index: None,
        restrict_indexing: false,
    }
}

/// The options wgpu-hal's D3D12 backend writes HLSL with, with mX's register layout.
pub fn options(m: &Module, shader_model: ShaderModel, heap: HeapSizes) -> hlsl::Options {
    let mut binding_map = hlsl::BindingMap::default();
    for (_, var) in m.module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
        let bt = if rb.group == HEAP_GROUP {
            if heap::is_texture(rb.binding) || heap::is_storage_texture(rb.binding) || heap::is_acceleration_structure(rb.binding) {
                hlsl::BindTarget {
                    binding_array_size: Some(heap.textures),
                    ..target(heap_space(rb.binding), 0)
                }
            } else if heap::is_buffer(rb.binding) {
                hlsl::BindTarget {
                    binding_array_size: Some(heap.buffers),
                    ..target(heap_space(rb.binding), 0)
                }
            } else {
                // (the base index into the heap group's identity sampler index buffer)
                hlsl::BindTarget { space: u8::MAX, ..target(0, 0) }
            }
        } else {
            match m.module.types[var.ty].inner {
                // (naga does not use the space of samplers; the register indexes the group's
                // sampler index buffer, which mX lays out by binding)
                naga::TypeInner::Sampler { .. } => hlsl::BindTarget {
                    space: u8::MAX,
                    ..target(0, rb.binding)
                },
                _ => target(rb.group, rb.binding),
            }
        };
        binding_map.insert(rb, bt);
    }
    let sampler_buffer_binding_map = (0..=HEAP_GROUP).map(|g| (hlsl::SamplerIndexBufferKey { group: g }, target(SAMPLER_INDEX_SPACE, g))).collect();
    hlsl::Options {
        shader_model,
        binding_map,
        fake_missing_bindings: false,
        special_constants_binding: Some(target(IMMEDIATES_SPACE, SPECIAL_CONSTANTS_REGISTER)),
        immediates_target: Some(target(IMMEDIATES_SPACE, IMMEDIATES_REGISTER)),
        sampler_heap_target: hlsl::SamplerHeapBindTargets {
            standard_samplers: target(SAMPLER_HEAP_SPACE, 0),
            comparison_samplers: target(SAMPLER_HEAP_SPACE, COMPARISON_SAMPLER_REGISTER),
        },
        sampler_buffer_binding_map,
        // As wgpu-hal: zeroed workgroup memory, clamped indices (naga's HLSL writer has no
        // per-kind policies; buffers are bounds-checked by D3D12's robust access), bounded
        // loops and tracked ray query initialisation.
        zero_initialize_workgroup_memory: true,
        restrict_indexing: true,
        force_loop_bounding: true,
        ray_query_initialization_tracking: true,
        ..Default::default()
    }
}

/// One entry point in HLSL.
#[derive(Clone, Debug)]
pub struct Output {
    pub source: String,
    /// The function's name in `source` (naga renames names HLSL reserves).
    pub entry_point: String,
    /// The compile target, e.g. `vs_6_5`.
    pub profile: String,
    /// Which heap bindings the entry point reads (bit per `heap::*` and buffer view).
    pub heap_bindings_used: u64,
    pub uses_immediates: bool,
    /// The root bindings the entry point uses, sorted by (group, binding).
    pub roots: Vec<RootBinding>,
    pub workgroup_size: [u32; 3],
}

/// The HLSL stage prefix of a profile.
pub fn stage_prefix(stage: naga::ShaderStage) -> &'static str {
    match stage {
        naga::ShaderStage::Vertex => "vs",
        naga::ShaderStage::Fragment => "ps",
        naga::ShaderStage::Compute => "cs",
        naga::ShaderStage::Task => "as",
        naga::ShaderStage::Mesh => "ms",
        _ => "lib",
    }
}

/// The root bindings entry point `name` of `stage` uses (as wgpu counts them).
pub fn roots_for_entry_point(m: &Module, stage: naga::ShaderStage, name: &str) -> Result<Vec<RootBinding>, ShaderError> {
    m.entry_point_roots(stage, name)
}

/// Write one entry point of `m` as HLSL for `shader_model`, with `constants` applied to its
/// `override`s and the heap's arrays `heap` long. For a vertex entry point,
/// pass the fragment stage it is paired with (as wgpu-hal does): HLSL needs the vertex
/// outputs to match the fragment inputs.
pub fn write(
    m: &Module,
    stage: naga::ShaderStage,
    entry: Option<&str>,
    constants: &[(&str, f64)],
    shader_model: ShaderModel,
    heap: HeapSizes,
    fragment: Option<(&Module, &str)>,
) -> Result<Output, ShaderError> {
    let name = m.entry_point(stage, entry)?.name.clone();
    let roots = roots_for_entry_point(m, stage, &name)?;
    for r in &roots {
        debug_assert!(r.group <= MAX_ROOT_GROUP);
        if r.kind == RootKind::AccelerationStructure && shader_model < ShaderModel::V6_5 {
            return Err(ShaderError::Backend {
                label: m.label.clone(),
                target: "HLSL",
                message: "ray queries need shader model 6.5".into(),
            });
        }
    }
    let (module, info) = m.specialize(stage, &name, constants)?;
    let backend_err = |message: String| ShaderError::Backend {
        label: m.label.clone(),
        target: "HLSL",
        message,
    };
    let frag_ep = match fragment {
        Some((fm, fname)) => Some(hlsl::FragmentEntryPoint::new(&fm.module, fname).ok_or_else(|| backend_err(format!("no fragment entry point `{fname}` to pair with")))?),
        None => None,
    };
    let options = options(m, shader_model, heap);
    let pipeline = hlsl::PipelineOptions {
        entry_point: Some((stage, name.clone())),
    };
    let mut source = String::new();
    let mut writer = hlsl::Writer::new(&mut source, &options, &pipeline);
    let reflection = writer.write(&module, &info, frag_ep.as_ref()).map_err(|e| backend_err(e.to_string()))?;
    let entry_point = match reflection.entry_point_names.into_iter().next() {
        Some(Ok(n)) => n,
        Some(Err(e)) => return Err(backend_err(e.to_string())),
        None => return Err(backend_err("no entry point written".into())),
    };

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
        source,
        entry_point,
        profile: format!("{}_{}", stage_prefix(stage), shader_model.to_str()),
        heap_bindings_used,
        uses_immediates,
        roots,
        workgroup_size: ep.workgroup_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use naga::ShaderStage as S;

    const SM: ShaderModel = ShaderModel::V6_5;
    const HEAP: HeapSizes = HeapSizes { textures: 1 << 16, buffers: 1 << 14 };

    fn compact(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn buffer_views_are_indexed_byte_address_buffers() {
        let m = Module::from_wgsl(
            "t",
            "struct L { l: array<vec4<f32>> }\n@group(7) @binding(16) var<storage, read> lights: binding_array<L>;\nstruct D { a: u32, b: u32 }\nvar<immediate> d: D;\n@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { mx_rw_buffers[d.b].w[id.x] = u32(lights[d.a].l[id.x].x); atomicAdd(&counters[d.b].c, 1u); }\nstruct C { c: atomic<u32> }\n@group(7) @binding(18) var<storage, read_write> counters: binding_array<C>;",
        )
        .unwrap();
        let out = write(&m, S::Compute, None, &[], SM, HEAP, None).unwrap();
        let src = compact(&out.source);
        for want in [
            "ByteAddressBuffer lights[16384] : register(t0, space32)",
            "RWByteAddressBuffer mx_rw_buffers[16384] : register(u0, space26)",
            "RWByteAddressBuffer counters[16384] : register(u0, space34)",
            "lights[_e",
            "mx_rw_buffers[_e",
            "counters[_e",
        ] {
            assert!(src.contains(want), "missing `{want}` in:\n{}", out.source);
        }
    }

    #[test]
    fn heap_roots_and_immediates_land_in_their_registers() {
        let m = Module::from_wgsl(
            "t",
            "struct C { k: vec4<f32> }\n@group(0) @binding(0) var<uniform> c: C;\n@group(2) @binding(1) var<storage, read> xs: array<f32>;\n@group(2) @binding(3) var<storage, read_write> ys: array<f32>;\n@group(1) @binding(4) var smp: sampler;\n@group(1) @binding(2) var tex: texture_2d<f32>;\nstruct D { t: u32, s: u32 }\nvar<immediate> d: D;\n@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { ys[0] = 1.0; return c.k * xs[0] * textureSample(mx_tex2d[d.t], mx_samplers[d.s], uv) * textureSample(tex, smp, uv) * textureSampleCompare(mx_texdepth[0], mx_samplers_cmp[d.s], uv, 0.5); }",
        )
        .unwrap();
        let out = write(&m, S::Fragment, None, &[], SM, HEAP, None).unwrap();
        let src = compact(&out.source);
        for want in [
            "register(b0)",                   // group 0 binding 0
            "register(t1, space2)",           // read-only storage
            "register(u3, space2)",           // read-write storage
            "register(t2, space1)",           // root texture
            "register(t0, space8)",           // mx_tex2d
            "register(t0, space12)",          // mx_texdepth
            "register(s0, space80)",          // sampler heap
            "register(s2048, space80)",       // comparison sampler heap
            "register(t1, space7)",           // group 1's sampler indices
            "register(t7, space7)",           // the heap's sampler indices
            "register(b0, space15)",          // immediates
            "register(b1, space15)",          // special constants
            "nagaGroup1SamplerIndexArray[4]", // root sampler at binding 4
        ] {
            assert!(src.contains(want), "missing {want}:\n{}", out.source);
        }
        assert!(src.contains("Texture2D<float4> mx_tex2d[65536] : register(t0, space8)"), "{}", out.source);
        assert_eq!(out.heap_bindings_used, 1 << heap::TEX_2D | 1 << heap::TEX_DEPTH | 1 << heap::SAMPLERS | 1 << heap::SAMPLERS_CMP);
        assert!(out.uses_immediates);
        assert_eq!(out.profile, "ps_6_5");
        let roots: Vec<_> = out.roots.iter().map(|r| (r.group, r.binding)).collect();
        assert_eq!(roots, [(0, 0), (1, 2), (1, 4), (2, 1), (2, 3)]);
    }

    #[test]
    fn storage_textures_and_acceleration_structures_in_the_heap() {
        let m = Module::from_wgsl(
            "t",
            "enable wgpu_ray_query;\n@group(7) @binding(11) var mx_tlas: binding_array<acceleration_structure>;\n@group(7) @binding(12) var a: binding_array<texture_storage_2d<rgba16float, write>>;\n@group(7) @binding(13) var b: binding_array<texture_storage_2d<rgba8unorm, write>>;\nstruct D { t: u32, d: u32, o: u32 }\nvar<immediate> d: D;\n@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { var rq: ray_query; rayQueryInitialize(&rq, mx_tlas[d.t], RayDesc(0u, 0xFFu, 0.0, 10.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0))); rayQueryProceed(&rq); let z = textureLoad(mx_texdepth_ms[d.d], vec2<i32>(id.xy), 1); textureStore(a[d.o], vec2<i32>(id.xy), vec4<f32>(z)); textureStore(b[d.o], vec2<i32>(id.xy), vec4<f32>(f32(rayQueryGetCommittedIntersection(&rq).instance_custom_data))); }",
        )
        .unwrap();
        let out = write(&m, S::Compute, None, &[], SM, HEAP, None).unwrap();
        let src = compact(&out.source);
        for want in [
            "RaytracingAccelerationStructure mx_tlas[65536] : register(t0, space27)",
            "RWTexture2D<float4> a[65536] : register(u0, space28)",
            "RWTexture2D<unorm float4> b[65536] : register(u0, space29)",
            "Texture2DMS<float> mx_texdepth_ms[65536] : register(t0, space14)",
            "mx_texdepth_ms[_e",
        ] {
            assert!(src.contains(want), "missing `{want}` in:\n{}", out.source);
        }
        assert_eq!(out.heap_bindings_used, 1 << heap::TEX_DEPTH_MS | 1 << heap::ACCELERATION_STRUCTURES | 1 << 12 | 1 << 13);
    }

    #[test]
    fn vertex_outputs_follow_the_fragment_inputs() {
        let m = Module::from_wgsl(
            "t",
            "struct O { @builtin(position) p: vec4<f32>, @location(0) a: f32, @location(1) b: f32 }\n@vertex fn vs(@builtin(vertex_index) i: u32, @builtin(instance_index) k: u32) -> O { return O(vec4<f32>(f32(i)), 1.0, f32(k)); }\n@fragment fn fs(@location(1) b: f32) -> @location(0) vec4<f32> { return vec4<f32>(b); }",
        )
        .unwrap();
        let out = write(&m, S::Vertex, None, &[], SM, HEAP, Some((&m, "fs"))).unwrap();
        let src = compact(&out.source);
        assert!(src.contains("_NagaConstants.first_vertex"), "{}", out.source);
        assert!(src.contains("_NagaConstants.first_instance"), "{}", out.source);
        assert!(
            src.contains("struct VertexOutput_vs { float b : LOC1; float4 p : SV_Position; };"),
            "the unread output is dropped:\n{}",
            out.source
        );
        assert_eq!(out.profile, "vs_6_5");
    }

    #[test]
    fn compute_and_ray_queries() {
        let m = Module::from_wgsl(
            "t",
            "enable wgpu_ray_query;\n@group(0) @binding(1) var acc: acceleration_structure;\n@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(3) fn main(@builtin(num_workgroups) n: vec3<u32>) { var rq: ray_query; rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 10.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0))); rayQueryProceed(&rq); out[0] = rayQueryGetCommittedIntersection(&rq).instance_custom_data + n.x; }",
        )
        .unwrap();
        let out = write(&m, S::Compute, None, &[], SM, HEAP, None).unwrap();
        let src = compact(&out.source);
        assert!(src.contains("RaytracingAccelerationStructure acc : register(t1)"), "{}", out.source);
        assert!(src.contains("register(u0)"), "{}", out.source);
        assert_eq!(out.workgroup_size, [3, 1, 1]);
        assert!(write(&m, S::Compute, None, &[], ShaderModel::V6_0, HEAP, None).is_err());
    }

    #[test]
    fn overrides_are_applied() {
        let m = Module::from_wgsl("t", "override K: f32 = 1.0;\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(K); }").unwrap();
        let out = write(&m, S::Fragment, Some("fs"), &[("K", 0.25)], SM, HEAP, None).unwrap();
        assert!(out.source.contains("0.25"), "{}", out.source);
    }

    #[test]
    fn each_entry_point_keeps_only_its_roots() {
        let decls: String = (0..14).map(|i| format!("@group(0) @binding({i}) var<uniform> u{i}: vec4<f32>;\n")).collect();
        let src = format!("{decls}@vertex fn vs() -> @builtin(position) vec4<f32> {{ return u0 + u13; }}\n@fragment fn fs() -> @location(0) vec4<f32> {{ return u1; }}");
        let m = Module::from_wgsl("t", &src).unwrap();
        let vs: Vec<_> = roots_for_entry_point(&m, S::Vertex, "vs").unwrap().iter().map(|r| r.binding).collect();
        assert_eq!(vs, [0, 13]);
        let fs = write(&m, S::Fragment, None, &[], SM, HEAP, None).unwrap();
        assert_eq!(fs.roots.iter().map(|r| r.binding).collect::<Vec<_>>(), [1]);
    }

    #[test]
    fn a_shared_binding_keeps_each_stage_to_its_own_variable() {
        // as openOMSI's sky motion pass: a texture and a uniform buffer on @group(3) @binding(0)
        let src = "@group(3) @binding(0) var depth: texture_depth_2d;\n@group(3) @binding(0) var<uniform> motion: vec4<f32>;\n@vertex fn vs() -> @builtin(position) vec4<f32> { return motion; }\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(f32(textureDimensions(depth).x)); }\n@fragment fn fs_motion() -> @location(0) vec4<f32> { return motion; }";
        let m = Module::from_wgsl("t", src).unwrap();
        let vs = write(&m, S::Vertex, Some("vs"), &[], SM, HEAP, Some((&m, "fs_motion"))).unwrap();
        assert_eq!(vs.roots.iter().map(|r| r.kind).collect::<Vec<_>>(), [crate::RootKind::UniformBuffer]);
        let fs = write(&m, S::Fragment, Some("fs"), &[], SM, HEAP, None).unwrap();
        assert_eq!(fs.roots.iter().map(|r| r.kind).collect::<Vec<_>>(), [crate::RootKind::Texture]);
    }

    /// Every shader of `crates/mx/tests/conformance.rs` translates.
    #[test]
    fn the_conformance_shaders_translate() {
        const FULLSCREEN_VS: &str = "struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }\n@vertex fn vs(@builtin(vertex_index) i: u32) -> Out { let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)); return Out(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(p.x, 1.0 - p.y)); }\n";
        const STRIPES: &str = "struct Out { @builtin(position) pos: vec4<f32>, @location(0) @interpolate(flat) k: u32 }\n@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) k: u32) -> Out { let c = array<vec2<f32>, 6>(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)); let p = c[v % 6u]; return Out(vec4<f32>((f32(k) + p.x) / 2.0 - 1.0, p.y * 2.0 - 1.0, 0.0, 1.0), k); }\n@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return vec4<f32>(f32(i.k + 1u) / 4.0, 0.0, 0.0, 1.0); }";
        let render = [
            format!("{FULLSCREEN_VS}struct Draw {{ tex: u32, smp: u32 }}\nvar<immediate> draw: Draw;\n@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{ return textureSample(mx_tex2d[draw.tex], mx_samplers[draw.smp], i.uv); }}"),
            format!("{FULLSCREEN_VS}struct Tint {{ c: vec4<f32> }}\n@group(0) @binding(0) var<uniform> tint: Tint;\n@group(1) @binding(0) var tex: texture_2d<f32>;\n@group(1) @binding(1) var smp: sampler;\n@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{ return tint.c * textureSample(tex, smp, i.uv); }}"),
            "struct Out { @builtin(position) pos: vec4<f32>, @location(0) c: vec3<f32> }\n@vertex fn vs(@location(0) p: vec3<f32>, @location(1) c: vec3<f32>) -> Out { return Out(vec4<f32>(p, 1.0), c); }\n@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return vec4<f32>(i.c, 1.0); }".to_string(),
            format!("{FULLSCREEN_VS}override GREEN: f32 = 0.0;\n@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{ return vec4<f32>(0.0, GREEN, 0.0, 1.0); }}"),
            STRIPES.to_string(),
            "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> { let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0)); return vec4<f32>(p[i], 0.0, 1.0); }\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }".to_string(),
            // (the mip generator of the D3D12 backend)
            "struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }\n@vertex fn vs(@builtin(vertex_index) i: u32) -> Out { let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)); return Out(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(p.x, 1.0 - p.y)); }\n@group(0) @binding(0) var src: texture_2d<f32>;\n@group(0) @binding(1) var smp: sampler;\n@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return textureSampleLevel(src, smp, i.uv, 0.0); }".to_string(),
        ];
        for (k, src) in render.iter().enumerate() {
            let m = Module::from_wgsl("t", src).unwrap();
            let constants: &[(&str, f64)] = if k == 3 { &[("GREEN", 1.0)] } else { &[] };
            let fs = write(&m, S::Fragment, None, constants, SM, HEAP, None).unwrap_or_else(|e| panic!("shader {k}: {e}"));
            let vs = write(&m, S::Vertex, None, constants, SM, HEAP, Some((&m, &fs.entry_point))).unwrap_or_else(|e| panic!("shader {k}: {e}"));
            assert!(vs.source.contains("register(b1, space15)"), "shader {k}:\n{}", vs.source);
        }
        let compute = [
            "@group(0) @binding(0) var<storage, read> src: array<u32>;\n@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { if (id.x < arrayLength(&dst)) { dst[id.x] = src[id.x] * 2u + arrayLength(&src); } }",
            "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(4) fn main(@builtin(global_invocation_id) id: vec3<u32>) { out[id.x] = id.x + 1u; }",
            "enable wgpu_ray_query;\n@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@group(0) @binding(1) var acc: acceleration_structure;\n@compute @workgroup_size(3) fn main(@builtin(local_invocation_index) i: u32) { let x = array<f32, 3>(0.0, 10.0, 5.0)[i]; var rq: ray_query; rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 100.0, vec3<f32>(x, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0))); rayQueryProceed(&rq); let hit = rayQueryGetCommittedIntersection(&rq); if (hit.kind == RAY_QUERY_INTERSECTION_TRIANGLE) { out[i] = hit.instance_custom_data; } else { out[i] = 0xFFFFFFFFu; } }",
        ];
        for (k, src) in compute.iter().enumerate() {
            let m = Module::from_wgsl("t", src).unwrap();
            let cs = write(&m, S::Compute, None, &[], SM, HEAP, None).unwrap_or_else(|e| panic!("compute shader {k}: {e}"));
            assert!(cs.source.contains("register(u"), "compute shader {k}:\n{}", cs.source);
        }
    }

    /// openOMSI's real modules (`python3 tools/openomsi_shaders.py <openOMSI> <dir>`), when
    /// `MX_OPENOMSI_SHADERS=<dir>` is set: HLSL for every entry point. (Generation only: DXC
    /// compiles it on Windows.)
    #[test]
    fn openomsi_modules_translate() {
        let Ok(dir) = std::env::var("MX_OPENOMSI_SHADERS") else {
            eprintln!("skipped: set MX_OPENOMSI_SHADERS (see tools/openomsi_shaders.py)");
            return;
        };
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "wgsl"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no .wgsl files in {dir}");
        let mut failed = Vec::new();
        for path in &files {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let source = std::fs::read_to_string(path).unwrap();
            let m = match Module::from_wgsl(&name, &source) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("{name}: FAILED to parse\n{e}");
                    failed.push(name);
                    continue;
                }
            };
            let mut n = 0;
            let mut ok = true;
            for ep in &m.module.entry_points {
                if !matches!(ep.stage, S::Vertex | S::Fragment | S::Compute) {
                    continue;
                }
                n += 1;
                match write(&m, ep.stage, Some(&ep.name), &[], SM, HEAP, None) {
                    Ok(out) => {
                        // Resource binding tier 2 takes 14 CBVs per stage; immediates and
                        // naga's special constants take two
                        let cbvs = out.roots.iter().filter(|r| r.kind == crate::RootKind::UniformBuffer).count();
                        if cbvs > MAX_UNIFORM_BUFFERS as usize {
                            eprintln!("{name}: {:?} `{}` has {cbvs} uniform buffers; D3D12 takes {MAX_UNIFORM_BUFFERS}", ep.stage, ep.name);
                            ok = false;
                        }
                    }
                    Err(e) => {
                        eprintln!("{name}: {:?} `{}` FAILED\n{e}", ep.stage, ep.name);
                        ok = false;
                    }
                }
            }
            if ok {
                eprintln!("{name}: {n} entry points ok");
            } else {
                failed.push(name);
            }
        }
        assert!(failed.is_empty(), "{} of {} modules failed: {failed:?}", failed.len(), files.len());
    }
}

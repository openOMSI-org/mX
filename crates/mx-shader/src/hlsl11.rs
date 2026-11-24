//! HLSL shader model 5.0 for the Direct3D 11 backend: the bound tier (`docs/BOUND_TIER.md`).
//!
//! The heap is lowered to slots first ([`crate::bound`], as for GLSL). Direct3D 11 numbers its
//! registers per stage and per class, so each stage gets its own: `t` for textures and storage
//! buffers it only reads, `s` for samplers, `b` for uniform buffers, `u` for storage textures
//! and buffers it writes (in a fragment stage from `uav_base` on: the render targets come
//! first). Root bindings in `(group, binding)` order, then the stage's slots. The immediates
//! are a constant buffer at `b13`, naga's special constants (the first vertex and instance) at
//! `b12`.

use std::collections::BTreeMap;

use naga::back::hlsl;

use crate::bound::Slot;
use crate::bound::Stage;
use crate::{HEAP_GROUP, Module, ShaderError};

pub use hlsl::ShaderModel;

/// Where the immediates are: `cbuffer : register(b13)`.
pub const IMMEDIATES_REGISTER: u32 = 13;
/// naga's special constants (first vertex, first instance): `register(b12)`.
pub const SPECIAL_CONSTANTS_REGISTER: u32 = 12;
/// Constant buffers left for root bindings (below the special constants').
pub const MAX_CBVS: u32 = 12;
pub const MAX_SRVS: u32 = 128;
pub const MAX_SAMPLERS: u32 = 16;
/// UAV slots of shader model 5.0 (Direct3D 11.0; 11.1 has 64).
pub const MAX_UAVS: u32 = 8;

/// A Direct3D 11 register class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Class {
    /// `t`: a texture, or a storage buffer only read.
    Srv,
    /// `s`
    Sampler,
    /// `b`
    Cbv,
    /// `u`: a storage texture, or a storage buffer written.
    Uav,
}

/// One stage in HLSL, and where everything it reads is bound.
#[derive(Clone, Debug)]
pub struct Output {
    pub source: String,
    /// The function's name in `source`.
    pub entry_point: String,
    /// `vs_5_0`, `ps_5_0` or `cs_5_0`.
    pub profile: String,
    /// The root bindings the stage reads, and their registers.
    pub roots: Vec<(naga::ResourceBinding, Class, u8)>,
    /// The heap slots (indices from the immediates or constants), and their registers.
    pub slots: Vec<(Slot, Class, u8)>,
    pub uses_immediates: bool,
    pub workgroup_size: [u32; 3],
}

fn target(register: u32) -> hlsl::BindTarget {
    hlsl::BindTarget {
        space: 0,
        register,
        binding_array_size: None,
        dynamic_storage_buffer_offsets_index: None,
        restrict_indexing: false,
    }
}

/// The class of a global's register.
fn class_of(module: &naga::Module, var: &naga::GlobalVariable) -> Result<Class, String> {
    use naga::{AddressSpace as A, ImageClass, TypeInner as T};
    Ok(match (&module.types[var.ty].inner, var.space) {
        (T::Sampler { .. }, _) => Class::Sampler,
        (
            T::Image {
                class: ImageClass::Storage { .. }, ..
            },
            _,
        ) => Class::Uav,
        (T::Image { .. }, _) => Class::Srv,
        (T::AccelerationStructure { .. }, _) => return Err("ray queries need a device with bindless (Direct3D 12, Vulkan, Metal 3)".into()),
        (_, A::Uniform) => Class::Cbv,
        (_, A::Storage { access }) if access.contains(naga::StorageAccess::STORE) => Class::Uav,
        (_, A::Storage { .. }) => Class::Srv,
        (other, space) => return Err(format!("a {other:?} in {space:?} has no Direct3D 11 register")),
    })
}

/// Write one stage. `uav_base`: the first UAV register (a fragment stage's render target
/// count; 0 elsewhere). `fragment`: for a vertex stage, the fragment stage it is paired with
/// (HLSL needs the vertex outputs to match the fragment inputs).
pub fn write(m: &Module, s: &Stage, uav_base: u32, fragment: Option<(&Module, &str)>) -> Result<Output, ShaderError> {
    let backend_err = |message: String| ShaderError::Backend {
        label: m.label.clone(),
        target: "HLSL (shader model 5.0)",
        message,
    };
    // (the heap's arrays, no longer read once lowered, would be declared unsized: dropped)
    let mut module = s.lowered.module.clone();
    module.entry_points.retain(|e| e.stage == s.stage && e.name == s.entry);
    naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    let info = crate::validator().validate(&module).map_err(|e| backend_err(format!("{e:?}")))?;
    let module = &module;
    let info = &info;
    let (ep_index, ep) = module
        .entry_points
        .iter()
        .enumerate()
        .find(|(_, e)| e.stage == s.stage && e.name == s.entry)
        .ok_or_else(|| backend_err("entry point lost".into()))?;
    let ep_info = info.get_entry_point(ep_index);

    // registers: the roots in (group, binding) order, then the slots
    let mut next: BTreeMap<Class, u32> = BTreeMap::new();
    next.insert(Class::Uav, uav_base);
    let mut binding_map = hlsl::BindingMap::default();
    let mut used: Vec<(naga::ResourceBinding, Class)> = Vec::new();
    let mut heap_used: Vec<(naga::ResourceBinding, Class)> = Vec::new();
    for (h, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
        if ep_info[h].is_empty() {
            continue;
        }
        let class = class_of(module, var).map_err(backend_err)?;
        if rb.group == HEAP_GROUP {
            heap_used.push((rb, class));
        } else {
            used.push((rb, class));
        }
    }
    used.sort();
    heap_used.sort();
    let mut roots = Vec::new();
    let mut slots = Vec::new();
    for (list, is_slot) in [(&used, false), (&heap_used, true)] {
        for &(rb, class) in list {
            let n = next.entry(class).or_insert(0);
            let reg = *n;
            *n += 1;
            let max = match class {
                Class::Srv => MAX_SRVS,
                Class::Sampler => MAX_SAMPLERS,
                Class::Cbv => MAX_CBVS,
                Class::Uav => MAX_UAVS,
            };
            if reg >= max {
                return Err(backend_err(format!("more than {max} {class:?} registers in one stage")));
            }
            binding_map.insert(rb, target(reg));
            if is_slot {
                let slot = s
                    .lowered
                    .slots
                    .iter()
                    .find(|sl| sl.binding == rb.binding)
                    .ok_or_else(|| backend_err(format!("heap binding {} is not a lowered slot", rb.binding)))?;
                slots.push((*slot, class, reg as u8));
            } else {
                roots.push((rb, class, reg as u8));
            }
        }
    }
    // (globals the entry point does not read still need a register to be declared)
    for (_, var) in module.global_variables.iter() {
        if let Some(rb) = var.binding {
            binding_map.entry(rb).or_insert_with(|| target(0));
        }
    }

    let options = hlsl::Options {
        shader_model: ShaderModel::V5_0,
        binding_map,
        fake_missing_bindings: false,
        special_constants_binding: Some(target(SPECIAL_CONSTANTS_REGISTER)),
        immediates_target: Some(target(IMMEDIATES_REGISTER)),
        zero_initialize_workgroup_memory: true,
        restrict_indexing: true,
        force_loop_bounding: true,
        ..Default::default()
    };
    let frag_ep = match fragment {
        Some((fm, fname)) => Some(hlsl::FragmentEntryPoint::new(&fm.module, fname).ok_or_else(|| backend_err(format!("no fragment entry point `{fname}` to pair with")))?),
        None => None,
    };
    let pipeline = hlsl::PipelineOptions {
        entry_point: Some((s.stage, s.entry.clone())),
    };
    let mut source = String::new();
    let mut writer = hlsl::Writer::new(&mut source, &options, &pipeline);
    let reflection = writer.write(module, info, frag_ep.as_ref()).map_err(|e| backend_err(e.to_string()))?;
    let entry_point = match reflection.entry_point_names.into_iter().next() {
        Some(Ok(n)) => n,
        Some(Err(e)) => return Err(backend_err(e.to_string())),
        None => return Err(backend_err("no entry point written".into())),
    };
    let uses_immediates = module.global_variables.iter().any(|(h, v)| v.space == naga::AddressSpace::Immediate && !ep_info[h].is_empty());
    Ok(Output {
        source,
        entry_point,
        profile: format!("{}_5_0", crate::hlsl::stage_prefix(s.stage)),
        roots,
        slots,
        uses_immediates,
        workgroup_size: ep.workgroup_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap;
    use naga::ShaderStage as S;

    fn compact(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn heap_slots_and_roots_get_registers_of_their_own() {
        let m = Module::from_wgsl(
            "t",
            "struct C { k: vec4<f32> }\n@group(0) @binding(0) var<uniform> c: C;\n@group(1) @binding(0) var t0: texture_2d<f32>;\nstruct D { tex: u32, smp: u32 }\nvar<immediate> d: D;\n@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return c.k * textureSample(mx_tex2d[d.tex], mx_samplers[d.smp], uv) + textureLoad(t0, vec2<i32>(0), 0); }",
        )
        .unwrap();
        let s = crate::bound::stage(&m, S::Fragment, None, &[]).unwrap();
        let out = write(&m, &s, 1, None).unwrap();
        let src = compact(&out.source);
        assert!(src.contains("register(b0)"), "{src}");
        assert!(src.contains("register(t0)") && src.contains("register(t1)"), "{src}");
        assert!(src.contains("SamplerState") && src.contains("register(s0)"), "{src}");
        assert!(!src.contains("nagaSamplerHeap"), "{src}");
        assert!(!src.contains("ConstantBuffer<"), "shader model 5.0 has none: {src}");
        assert_eq!(out.profile, "ps_5_0");
        assert_eq!(out.slots.len(), 2);
    }

    #[test]
    fn storage_textures_are_uavs_and_multisampled_depth_an_srv() {
        let m = Module::from_wgsl(
            "t",
            "@group(7) @binding(13) var out: binding_array<texture_storage_2d<rgba8unorm, write>>;\nstruct D { d: u32, o: u32 }\nvar<immediate> d: D;\n@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { textureStore(out[d.o], vec2<i32>(id.xy), vec4<f32>(textureLoad(mx_texdepth_ms[d.d], vec2<i32>(id.xy), 1))); }",
        )
        .unwrap();
        let s = crate::bound::stage(&m, S::Compute, None, &[]).unwrap();
        let out = write(&m, &s, 0, None).unwrap();
        let src = compact(&out.source);
        assert!(src.contains("RWTexture2D<unorm float4>") && src.contains("register(u0)"), "{src}");
        assert!(src.contains("Texture2DMS<float>") && src.contains("register(t0)"), "{src}");
        let classes: Vec<_> = out.slots.iter().map(|(s, c, r)| (s.heap_binding, *c, *r)).collect();
        assert_eq!(classes, [(13, Class::Uav, 0), (heap::TEX_DEPTH_MS, Class::Srv, 0)]);
    }

    /// Immediates read as values (not heap indices) stay a constant buffer, at `b13`.
    #[test]
    fn immediates_are_a_cbuffer() {
        let m = Module::from_wgsl(
            "t",
            "struct D { k: vec4<f32> }\nvar<immediate> d: D;\n@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> { return d.k * f32(i); }",
        )
        .unwrap();
        let s = crate::bound::stage(&m, S::Vertex, None, &[]).unwrap();
        let out = write(&m, &s, 0, None).unwrap();
        let src = compact(&out.source);
        assert!(src.contains("register(b13) {"), "{src}");
        assert!(src.contains("register(b12) {"), "{src}");
        assert!(out.uses_immediates);
        assert_eq!(out.profile, "vs_5_0");
    }
}

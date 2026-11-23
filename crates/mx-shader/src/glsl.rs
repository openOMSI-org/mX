//! GLSL output for the bound tier (OpenGL and OpenGL ES, see `docs/BOUND_TIER.md`), written as
//! wgpu-hal's GL backend writes it.
//!
//! The heap is lowered to slots first ([`crate::bound`]). Bindings are numbered per resource
//! class (samplers, textures, storage images, uniform buffers, storage buffers) over every
//! stage of a pipeline: first the root bindings in `(group, binding)` order, then the slots of
//! the vertex stage, then those of the fragment stage. A texture's unit is its texture number,
//! and the sampler it is sampled with is bound to that unit (GLSL combines them).

use std::collections::BTreeMap;

use naga::back::glsl;

use crate::bound::Slot;
use crate::{HEAP_GROUP, Module, RootKind, ShaderError, heap};

pub use glsl::{FIRST_INSTANCE_BINDING, ImmediateItem, Version};

/// What a GL binding point holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Class {
    Sampler,
    Texture,
    Image,
    UniformBuffer,
    StorageBuffer,
}

/// The GL binding of every root binding and slot of a pipeline.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub bindings: BTreeMap<naga::ResourceBinding, (Class, u8)>,
}

pub use crate::bound::{Stage, stage, stage_with};

fn root_class(kind: RootKind) -> Class {
    match kind {
        RootKind::Sampler => Class::Sampler,
        RootKind::Texture => Class::Texture,
        RootKind::StorageTexture => Class::Image,
        RootKind::UniformBuffer => Class::UniformBuffer,
        RootKind::StorageBuffer { .. } => Class::StorageBuffer,
        // (no ray queries on this tier; pipelines with them fail before)
        RootKind::AccelerationStructure => Class::StorageBuffer,
    }
}

fn slot_class(slot: &Slot) -> Class {
    if heap::is_texture(slot.heap_binding) {
        Class::Texture
    } else if heap::is_sampler(slot.heap_binding) {
        Class::Sampler
    } else if slot.uniform {
        Class::UniformBuffer
    } else {
        Class::StorageBuffer
    }
}

impl Layout {
    /// Number the bindings of a pipeline's stages. Each stage's slots are its own: the slot
    /// bindings of two stages are told apart by the stage (see [`Layout::slot`]).
    pub fn for_pipeline(stages: &[&Stage]) -> Result<Layout, String> {
        let mut roots: BTreeMap<naga::ResourceBinding, Class> = BTreeMap::new();
        for s in stages {
            for r in &s.roots {
                if r.kind == RootKind::AccelerationStructure {
                    return Err("ray queries need a device with bindless (Metal 3, Vulkan, Direct3D 12)".into());
                }
                let rb = naga::ResourceBinding { group: r.group, binding: r.binding };
                if let Some(c) = roots.insert(rb, root_class(r.kind))
                    && c != root_class(r.kind)
                {
                    return Err(format!("@group({}) @binding({}) is a {c:?} in one stage and a {:?} in another", r.group, r.binding, root_class(r.kind)));
                }
            }
        }
        let mut next: BTreeMap<Class, u8> = BTreeMap::new();
        let mut take = |c: Class| -> Result<u8, String> {
            let n = next.entry(c).or_insert(0);
            let v = *n;
            *n = n.checked_add(1).ok_or_else(|| format!("more than 255 {c:?} bindings"))?;
            Ok(v)
        };
        let mut bindings = BTreeMap::new();
        for (rb, c) in roots {
            bindings.insert(rb, (c, take(c)?));
        }
        for (i, s) in stages.iter().enumerate() {
            for slot in &s.lowered.slots {
                let c = slot_class(slot);
                bindings.insert(stage_slot(i, slot.binding), (c, take(c)?));
            }
        }
        Ok(Layout { bindings })
    }

    /// The GL binding of slot `binding` of the `stage_index`-th stage.
    pub fn slot(&self, stage_index: usize, binding: u32) -> Option<(Class, u8)> {
        self.bindings.get(&stage_slot(stage_index, binding)).copied()
    }
}

/// The key of a stage's slot in [`Layout::bindings`]: slots live in the heap group, and each
/// stage gets its own range of it.
fn stage_slot(stage_index: usize, binding: u32) -> naga::ResourceBinding {
    naga::ResourceBinding {
        group: HEAP_GROUP,
        binding: binding + (stage_index as u32 + 1) * 0x1_0000,
    }
}

/// What the device offers, for the writer (as wgpu-hal's GL backend asks for it).
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub version: Version,
    /// `GL_EXT_texture_shadow_lod`.
    pub texture_shadow_lod: bool,
    /// `gl_BaseInstance` / `gl_BaseVertex` (GL 4.6, `GL_ARB_shader_draw_parameters`); without,
    /// the first instance is a uniform the backend sets.
    pub draw_parameters: bool,
}

/// One stage in GLSL.
#[derive(Clone, Debug)]
pub struct Output {
    pub source: String,
    /// The texture units and the samplers bound to them: (texture binding, sampler binding).
    pub texture_samplers: Vec<(u8, Option<u8>)>,
    /// Uniform names and the binding each must get after linking, where the GLSL version has
    /// no `layout(binding)` (GL 3.3, ES 3.0).
    pub named_bindings: Vec<(String, Class, u8)>,
    /// The immediates, as uniforms.
    pub immediates: Vec<ImmediateItem>,
    pub workgroup_size: [u32; 3],
}

/// Write one stage (the `stage_index`-th of the pipeline `layout` was made for).
pub fn write(m: &Module, s: &Stage, stage_index: usize, layout: &Layout, options: &Options) -> Result<Output, ShaderError> {
    let backend_err = |message: String| ShaderError::Backend {
        label: m.label.clone(),
        target: "GLSL",
        message,
    };
    let module = &s.lowered.module;
    let mut binding_map = glsl::BindingMap::default();
    for (h, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
        let found = if rb.group == HEAP_GROUP {
            layout.slot(stage_index, rb.binding)
        } else {
            layout.bindings.get(&rb).copied()
        };
        if let Some((_, b)) = found {
            binding_map.insert(rb, b);
        }
        let _ = h;
    }
    let mut flags = glsl::WriterFlags::ADJUST_COORDINATE_SPACE | glsl::WriterFlags::FORCE_POINT_SIZE;
    flags.set(glsl::WriterFlags::TEXTURE_SHADOW_LOD, options.texture_shadow_lod);
    flags.set(glsl::WriterFlags::DRAW_PARAMETERS, options.draw_parameters);
    let naga_options = glsl::Options {
        version: options.version,
        writer_flags: flags,
        binding_map,
        zero_initialize_workgroup_memory: true,
    };
    let pipeline = glsl::PipelineOptions {
        shader_stage: s.stage,
        entry_point: s.entry.clone(),
        multiview: None,
    };
    // As wgpu-hal: image loads checked where GL has the texture-levels query (desktop 4.3),
    // the rest left to GLSL.
    use naga::proc::BoundsCheckPolicy as P;
    let desktop_43 = matches!(options.version, Version::Desktop(v) if v >= 430);
    let policies = naga::proc::BoundsCheckPolicies {
        index: P::Unchecked,
        buffer: P::Unchecked,
        image_load: if desktop_43 { P::ReadZeroSkipWrite } else { P::Unchecked },
        binding_array: P::Unchecked,
    };
    let mut source = String::new();
    let mut writer = glsl::Writer::new(&mut source, module, &s.lowered.info, &naga_options, &pipeline, policies).map_err(|e| backend_err(e.to_string()))?;
    let reflection = writer.write().map_err(|e| backend_err(e.to_string()))?;
    drop(writer);

    let binding_of = |h: naga::Handle<naga::GlobalVariable>| module.global_variables[h].binding.and_then(|rb| naga_options.binding_map.get(&rb).copied());
    let mut texture_samplers = Vec::new();
    let mut named_bindings = Vec::new();
    for (name, mapping) in &reflection.texture_mapping {
        let Some(t) = binding_of(mapping.texture) else { continue };
        let class = match module.types[module.global_variables[mapping.texture].ty].inner {
            naga::TypeInner::Image {
                class: naga::ImageClass::Storage { .. },
                ..
            } => Class::Image,
            _ => Class::Texture,
        };
        if class == Class::Texture {
            texture_samplers.push((t, mapping.sampler.and_then(binding_of)));
        }
        named_bindings.push((name.clone(), class, t));
    }
    for (h, name) in &reflection.uniforms {
        let var = &module.global_variables[*h];
        let class = match var.space {
            naga::AddressSpace::Uniform => Class::UniformBuffer,
            naga::AddressSpace::Storage { .. } => Class::StorageBuffer,
            _ => continue,
        };
        if let Some(b) = binding_of(*h) {
            named_bindings.push((name.clone(), class, b));
        }
    }
    texture_samplers.sort_unstable();
    named_bindings.sort();
    let ep = module
        .entry_points
        .iter()
        .find(|e| e.stage == s.stage && e.name == s.entry)
        .ok_or_else(|| backend_err("entry point lost".into()))?;
    Ok(Output {
        source,
        texture_samplers,
        named_bindings,
        immediates: reflection.immediates_items,
        workgroup_size: ep.workgroup_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bound::Index;
    use naga::ShaderStage as S;

    /// Compile with `glslangValidator`, where it is installed.
    fn validate(source: &str, stage: S) {
        let candidates = ["glslangValidator", "/opt/homebrew/bin/glslangValidator", "/usr/local/bin/glslangValidator", "/usr/bin/glslangValidator"];
        let Some(tool) = candidates.iter().find(|c| std::process::Command::new(c).arg("--version").output().is_ok()) else {
            eprintln!("glslangValidator not found; skipping validation");
            return;
        };
        let ext = match stage {
            S::Vertex => "vert",
            S::Fragment => "frag",
            _ => "comp",
        };
        let dir = std::env::temp_dir().join(format!("mx-glsl-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("shader.{ext}"));
        std::fs::write(&path, source).unwrap();
        let out = std::process::Command::new(tool).arg(&path).output().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(out.status.success(), "glslangValidator:\n{}\n{source}", String::from_utf8_lossy(&out.stdout));
    }

    const OPTIONS: Options = Options {
        version: Version::Embedded { version: 300, is_webgl: false },
        texture_shadow_lod: false,
        draw_parameters: false,
    };

    #[test]
    fn heap_reads_by_immediates_become_slots() {
        let m = Module::from_wgsl(
            "t",
            "struct D { pad: u32, tex: u32, smp: u32 }\nvar<immediate> d: D;\nstruct Out { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> }\n@vertex fn vs() -> Out { return Out(vec4<f32>(0.0), vec2<f32>(0.0)); }\n@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return textureSample(mx_tex2d[d.tex], mx_samplers[d.smp], i.uv) + textureSample(mx_tex2d[d.tex], mx_samplers[d.smp], i.uv * 2.0); }",
        )
        .unwrap();
        let vs = stage(&m, S::Vertex, None, &[]).unwrap();
        let fs = stage(&m, S::Fragment, None, &[]).unwrap();
        let slots: Vec<_> = fs.lowered.slots.iter().map(|s| (s.heap_binding, s.index)).collect();
        // one slot per (array, index), however many reads
        assert_eq!(slots, [(heap::TEX_2D, Index::Immediate(4)), (heap::SAMPLERS, Index::Immediate(8))]);
        let layout = Layout::for_pipeline(&[&vs, &fs]).unwrap();
        let out = write(&m, &fs, 1, &layout, &OPTIONS).unwrap();
        validate(&out.source, S::Fragment);
        validate(&write(&m, &vs, 0, &layout, &OPTIONS).unwrap().source, S::Vertex);
        assert!(out.source.contains("uniform highp sampler2D"), "{}", out.source);
        assert!(!out.source.contains("mx_tex2d["), "{}", out.source);
        assert_eq!(out.texture_samplers, [(0, Some(0))]);
        assert!(!out.immediates.is_empty());
    }

    #[test]
    fn constant_indices_and_buffer_views_become_slots() {
        let m = Module::from_wgsl(
            "t",
            "struct L { l: array<vec4<f32>> }\n@group(7) @binding(16) var<storage, read> lights: binding_array<L>;\nstruct D { lights: u32 }\nvar<immediate> d: D;\n@fragment fn fs() -> @location(0) vec4<f32> { return lights[d.lights].l[0] + vec4<f32>(f32(mx_buffers[3].w[0])); }",
        )
        .unwrap();
        let fs = stage(&m, S::Fragment, None, &[]).unwrap();
        let slots: Vec<_> = fs.lowered.slots.iter().map(|s| (s.heap_binding, s.index)).collect();
        assert!(slots.contains(&(16, Index::Immediate(0))), "{slots:?}");
        assert!(slots.contains(&(heap::BUFFERS, Index::Constant(3))), "{slots:?}");
        let layout = Layout::for_pipeline(&[&fs]).unwrap();
        let out = write(
            &m,
            &fs,
            0,
            &layout,
            &Options {
                version: Version::Embedded { version: 310, is_webgl: false },
                ..OPTIONS
            },
        )
        .unwrap();
        validate(&out.source, S::Fragment);
        assert!(out.source.contains("buffer"), "{}", out.source);
    }

    /// Without storage buffers (GLSL ES 3.00) a read-only view of a fixed size - a frame's
    /// constants - is read as a uniform block; one with a runtime-sized array stays storage.
    #[test]
    fn read_only_views_of_a_fixed_size_become_uniform_blocks() {
        let m = Module::from_wgsl(
            "t",
            "struct C { k: vec4<f32>, m: mat4x4<f32> }\n@group(7) @binding(16) var<storage, read> cams: binding_array<C>;\nstruct D { pad: u32, cam: u32 }\nvar<immediate> d: D;\n@fragment fn fs() -> @location(0) vec4<f32> { return cams[d.cam].m * cams[d.cam].k; }",
        )
        .unwrap();
        let fs = crate::bound::stage_with(&m, S::Fragment, None, &[], true).unwrap();
        assert!(fs.lowered.slots.iter().all(|s| s.uniform && s.index == Index::Immediate(4)), "{:?}", fs.lowered.slots);
        let layout = Layout::for_pipeline(&[&fs]).unwrap();
        assert_eq!(layout.slot(0, fs.lowered.slots[0].binding).map(|s| s.0), Some(Class::UniformBuffer));
        let out = write(
            &m,
            &fs,
            0,
            &layout,
            &Options {
                version: Version::Embedded { version: 300, is_webgl: false },
                ..OPTIONS
            },
        )
        .unwrap();
        validate(&out.source, S::Fragment);
        assert!(!out.source.contains("buffer "), "{}", out.source);
        // (with storage buffers it stays one)
        let fs = crate::bound::stage_with(&m, S::Fragment, None, &[], false).unwrap();
        assert!(fs.lowered.slots.iter().all(|s| !s.uniform));
        // (a runtime-sized one cannot be a uniform block)
        let m = Module::from_wgsl(
            "t",
            "struct L { l: array<vec4<f32>> }\n@group(7) @binding(16) var<storage, read> lights: binding_array<L>;\nstruct D { lights: u32 }\nvar<immediate> d: D;\n@fragment fn fs() -> @location(0) vec4<f32> { return lights[d.lights].l[0]; }",
        )
        .unwrap();
        let fs = crate::bound::stage_with(&m, S::Fragment, None, &[], true).unwrap();
        assert!(fs.lowered.slots.iter().all(|s| !s.uniform));
    }

    /// A depth texture read with `textureLoad` and never sampled (openOMSI's clouds skip
    /// the pixels its scene covers so) is a `sampler2D` read with `texelFetch`, in GLSL ES
    /// 3.0 unchecked and in desktop GLSL 4.30 with the bounds checks; a multisampled one is
    /// read from its `sampler2DMS`. One also sampled with a comparison stays a shadow
    /// sampler, and its load is refused.
    #[test]
    fn depth_textures_only_loaded_from_are_fetched() {
        let m = Module::from_wgsl(
            "t",
            "@group(0) @binding(0) var d: texture_depth_2d;\n@group(0) @binding(1) var dm: texture_depth_multisampled_2d;\nfn covered(p: vec2<i32>) -> bool { return textureLoad(d, p, 0) <= 0.0; }\n@fragment fn fs(@builtin(position) c: vec4<f32>) -> @location(0) vec4<f32> { let p = vec2<i32>(c.xy); return vec4<f32>(select(0.0, 1.0, covered(p)) + textureLoad(dm, p, 1) + f32(textureDimensions(d).x)); }",
        )
        .unwrap();
        let fs = stage(&m, S::Fragment, None, &[]).unwrap();
        let layout = Layout::for_pipeline(&[&fs]).unwrap();
        for version in [Version::Embedded { version: 310, is_webgl: false }, Version::Desktop(430)] {
            let out = write(&m, &fs, 0, &layout, &Options { version, ..OPTIONS }).unwrap();
            validate(&out.source, S::Fragment);
            assert!(!out.source.contains("Shadow"), "{}", out.source);
            assert!(out.source.contains("sampler2D ") && out.source.contains("sampler2DMS "), "{}", out.source);
            assert_eq!(out.source.matches("texelFetch(").count(), 2, "{}", out.source);
        }
        // ES 3.0 has no multisampled textures: the plain one alone
        let m = Module::from_wgsl(
            "t",
            "@group(0) @binding(0) var d: texture_depth_2d;\n@fragment fn fs(@builtin(position) c: vec4<f32>) -> @location(0) vec4<f32> { return vec4<f32>(textureLoad(d, vec2<i32>(c.xy), 0)); }",
        )
        .unwrap();
        let fs = stage(&m, S::Fragment, None, &[]).unwrap();
        let layout = Layout::for_pipeline(&[&fs]).unwrap();
        let out = write(&m, &fs, 0, &layout, &OPTIONS).unwrap();
        validate(&out.source, S::Fragment);
        assert!(out.source.contains("uniform highp sampler2D ") && out.source.contains(").x"), "{}", out.source);
        let m = Module::from_wgsl(
            "t",
            "@group(0) @binding(0) var d: texture_depth_2d;\n@group(0) @binding(1) var s: sampler_comparison;\n@fragment fn fs(@builtin(position) c: vec4<f32>) -> @location(0) vec4<f32> { return vec4<f32>(textureLoad(d, vec2<i32>(c.xy), 0) + textureSampleCompareLevel(d, s, c.xy, 0.5)); }",
        )
        .unwrap();
        let fs = stage(&m, S::Fragment, None, &[]).unwrap();
        let layout = Layout::for_pipeline(&[&fs]).unwrap();
        let Err(e) = write(&m, &fs, 0, &layout, &OPTIONS) else {
            panic!("a load from a sampled depth texture")
        };
        assert!(e.to_string().contains("also sampled"), "{e}");
    }

    #[test]
    fn indices_that_need_bindless_are_refused() {
        let m = Module::from_wgsl(
            "t",
            "struct I { i: array<u32> }\n@group(0) @binding(0) var<storage, read> idx: I;\n@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return textureSample(mx_tex2d[idx.i[0]], mx_samplers[0], uv); }",
        )
        .unwrap();
        let e = stage(&m, S::Fragment, None, &[]).err().expect("an index from a buffer needs bindless");
        assert!(e.to_string().contains("immediates"), "{e}");
    }

    #[test]
    fn root_bindings_keep_wgpus_numbering() {
        let m = Module::from_wgsl(
            "t",
            "struct C { k: vec4<f32> }\n@group(0) @binding(0) var<uniform> c: C;\n@group(1) @binding(0) var t: texture_2d<f32>;\n@group(1) @binding(1) var s: sampler;\n@group(2) @binding(0) var<uniform> c2: C;\n@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return c.k * c2.k * textureSample(t, s, uv); }",
        )
        .unwrap();
        let fs = stage(&m, S::Fragment, None, &[]).unwrap();
        let layout = Layout::for_pipeline(&[&fs]).unwrap();
        let get = |g, b| layout.bindings[&naga::ResourceBinding { group: g, binding: b }];
        assert_eq!(get(0, 0), (Class::UniformBuffer, 0));
        assert_eq!(get(2, 0), (Class::UniformBuffer, 1));
        assert_eq!(get(1, 0), (Class::Texture, 0));
        assert_eq!(get(1, 1), (Class::Sampler, 0));
        // ES 3.0 has no layout(binding): the names to bind after linking
        let out = write(&m, &fs, 0, &layout, &OPTIONS).unwrap();
        validate(&out.source, S::Fragment);
        assert!(out.named_bindings.iter().any(|(_, c, b)| *c == Class::UniformBuffer && *b == 1), "{:?}", out.named_bindings);
    }
}

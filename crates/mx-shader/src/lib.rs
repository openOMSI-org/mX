//! WGSL through naga to MSL, SPIR-V and HLSL.
//!
//! Every shader gets the bindless heap: [`PRELUDE`] is appended to the source, so its names
//! (`mx_tex2d`, `mx_samplers`, `mx_buffers`, ...) are usable anywhere. WGSL group numbers are
//! virtual: the backend writers place each `(group, binding)` where their API wants it.
//!
//! Buffers are in the heap too. A buffer view's index reads it through `mx_buffers[i]` (words)
//! or through a typed view the shader declares itself in the heap group:
//!
//! ```wgsl
//! struct Lights { l: array<Light> }
//! @group(7) @binding(16) var<storage, read> lights: binding_array<Lights>;
//! ```
//!
//! Every typed view (bindings [`heap::FIRST_BUFFER_VIEW`]`..=`[`heap::LAST_BUFFER_VIEW`]) is
//! the same array of buffers seen with another element type, as `ByteAddressBuffer`s and
//! buffer device addresses are in other engines.

use std::fmt;

pub use naga;
use naga::valid::{Capabilities, ValidationFlags, Validator};

pub mod bound;
/// Compiled shaders kept on disk between runs (Direct3D's compilers).
pub mod disk_cache;
#[cfg(feature = "glsl")]
pub mod glsl;
#[cfg(feature = "hlsl")]
pub mod hlsl;
/// HLSL shader model 5.0, the Direct3D 11 backend's (the bound tier).
#[cfg(feature = "hlsl")]
pub mod hlsl11;
#[cfg(feature = "msl")]
pub mod msl;
#[cfg(feature = "spirv")]
pub mod spirv;

/// The WGSL group of the bindless heap. Groups below it are root (classic) bindings.
pub const HEAP_GROUP: u32 = 7;

/// The bindings of the heap, in [`HEAP_GROUP`].
pub mod heap {
    pub const TEX_2D: u32 = 0;
    pub const TEX_2D_ARRAY: u32 = 1;
    pub const TEX_CUBE: u32 = 2;
    pub const TEX_3D: u32 = 3;
    pub const TEX_DEPTH: u32 = 4;
    pub const TEX_2D_U32: u32 = 5;
    pub const SAMPLERS: u32 = 6;
    pub const SAMPLERS_CMP: u32 = 7;
    /// `mx_buffers`: every buffer view, read as words.
    pub const BUFFERS: u32 = 8;
    /// `mx_rw_buffers`: every buffer view, read and written as words.
    pub const RW_BUFFERS: u32 = 9;
    /// How many bindings the prelude declares.
    pub const COUNT: u32 = 10;
    /// The first binding a shader may give its own typed view of the buffer heap.
    pub const FIRST_BUFFER_VIEW: u32 = 16;
    /// The last one.
    pub const LAST_BUFFER_VIEW: u32 = 63;

    /// Whether a heap binding holds textures.
    pub const fn is_texture(binding: u32) -> bool {
        binding < SAMPLERS
    }

    /// Whether a heap binding holds samplers.
    pub const fn is_sampler(binding: u32) -> bool {
        binding == SAMPLERS || binding == SAMPLERS_CMP
    }

    /// Whether a heap binding is a view of the buffer heap.
    pub const fn is_buffer(binding: u32) -> bool {
        binding == BUFFERS || binding == RW_BUFFERS || (binding >= FIRST_BUFFER_VIEW && binding <= LAST_BUFFER_VIEW)
    }
}

/// Declarations every mX shader can use. Appended after the user's source, so the line
/// numbers in error messages are the user's own.
pub const PRELUDE: &str = "
// ---- mX bindless heap ----
@group(7) @binding(0) var mx_tex2d: binding_array<texture_2d<f32>>;
@group(7) @binding(1) var mx_tex2d_array: binding_array<texture_2d_array<f32>>;
@group(7) @binding(2) var mx_texcube: binding_array<texture_cube<f32>>;
@group(7) @binding(3) var mx_tex3d: binding_array<texture_3d<f32>>;
@group(7) @binding(4) var mx_texdepth: binding_array<texture_depth_2d>;
@group(7) @binding(5) var mx_tex2d_u32: binding_array<texture_2d<u32>>;
@group(7) @binding(6) var mx_samplers: binding_array<sampler>;
@group(7) @binding(7) var mx_samplers_cmp: binding_array<sampler_comparison>;
struct MxWords { w: array<u32> }
@group(7) @binding(8) var<storage, read> mx_buffers: binding_array<MxWords>;
@group(7) @binding(9) var<storage, read_write> mx_rw_buffers: binding_array<MxWords>;
";

/// The largest group a root binding may use.
pub const MAX_ROOT_GROUP: u32 = HEAP_GROUP - 1;

#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    #[error("{label}: WGSL error:\n{message}")]
    Parse { label: String, message: String },
    #[error("{label}: validation error:\n{message}")]
    Validate { label: String, message: String },
    #[error("{label}: {message}")]
    Layout { label: String, message: String },
    #[error("{label}: entry point `{name}` ({stage:?}) not found")]
    NoEntryPoint { label: String, name: String, stage: naga::ShaderStage },
    #[error("{label}: pipeline constants: {message}")]
    Constants { label: String, message: String },
    #[error("{label}: {target} output: {message}")]
    Backend { label: String, target: &'static str, message: String },
}

/// What a root binding is, as the shader declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RootKind {
    UniformBuffer,
    StorageBuffer {
        read_only: bool,
    },
    /// A sampled, depth or multisampled texture.
    Texture,
    StorageTexture,
    Sampler,
    AccelerationStructure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RootBinding {
    pub group: u32,
    pub binding: u32,
    pub kind: RootKind,
    /// A storage buffer ending in a runtime-sized array: its size is passed to the shader.
    pub runtime_sized: bool,
}

/// A parsed and validated shader.
#[derive(Clone)]
pub struct Module {
    pub label: String,
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
    /// The root bindings any entry point of the module uses, sorted by (group, binding).
    /// Two of them can share a (group, binding) when no entry point uses both, as WGSL
    /// allows; [`Module::entry_point_roots`] tells them apart.
    pub roots: Vec<RootBinding>,
    /// The views of the buffer heap the module declares (the prelude's two and the shader's
    /// own), sorted by binding.
    pub heap_buffers: Vec<HeapBuffer>,
    /// The variable behind each root binding.
    root_vars: Vec<(naga::Handle<naga::GlobalVariable>, RootBinding)>,
}

/// One view of the buffer heap: a `binding_array` of storage buffers in the heap group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeapBuffer {
    pub binding: u32,
    pub read_only: bool,
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Module").field("label", &self.label).field("roots", &self.roots).finish_non_exhaustive()
    }
}

pub(crate) fn validator() -> Validator {
    Validator::new(ValidationFlags::all(), Capabilities::all())
}

impl Module {
    /// Parse WGSL with the heap prelude and validate it.
    pub fn from_wgsl(label: &str, source: &str) -> Result<Module, ShaderError> {
        // The binding array extension goes on line 1 without a line break, so lines stay put.
        let full = format!("enable wgpu_binding_array; {source}\n{PRELUDE}");
        let module = naga::front::wgsl::parse_str(&full).map_err(|e| ShaderError::Parse {
            label: label.into(),
            message: e.emit_to_string(&full),
        })?;
        let info = validator().validate(&module).map_err(|e| ShaderError::Validate {
            label: label.into(),
            message: e.emit_to_string(&full),
        })?;
        let root_vars = reflect_roots(label, &module)?;
        let roots = root_vars.iter().map(|&(_, r)| r).collect();
        let heap_buffers = reflect_heap_buffers(label, &module)?;
        Ok(Module {
            label: label.into(),
            module,
            info,
            roots,
            heap_buffers,
            root_vars,
        })
    }

    /// The entry point of `stage` called `name`, or the only one of that stage.
    pub fn entry_point(&self, stage: naga::ShaderStage, name: Option<&str>) -> Result<&naga::EntryPoint, ShaderError> {
        let mut of_stage = self.module.entry_points.iter().filter(|e| e.stage == stage);
        let found = match name {
            Some(n) => of_stage.find(|e| e.name == n),
            None => {
                let first = of_stage.next();
                if of_stage.next().is_some() { None } else { first }
            }
        };
        found.ok_or_else(|| ShaderError::NoEntryPoint {
            label: self.label.clone(),
            name: name.unwrap_or("<the only one>").into(),
            stage,
        })
    }

    /// The heap bindings entry point `name` of `stage` uses, a bit per binding (`heap::*` and
    /// the shader's own buffer views).
    pub fn entry_point_heap(&self, stage: naga::ShaderStage, name: &str) -> Result<u64, ShaderError> {
        let index = self
            .module
            .entry_points
            .iter()
            .position(|e| e.stage == stage && e.name == name)
            .ok_or_else(|| ShaderError::NoEntryPoint {
                label: self.label.clone(),
                name: name.into(),
                stage,
            })?;
        let info = self.info.get_entry_point(index);
        Ok(heap_used(&self.module, |h| !info[h].is_empty()))
    }

    /// The bounds checks this module is written with. A module that reads buffers through the
    /// heap is a bindless one: as for heap textures, indices and offsets are the renderer's
    /// responsibility and buffer accesses are unchecked (the drivers' robust access still
    /// applies where the device has it). Other modules keep wgpu-hal's checks, so they draw
    /// what they drew under wgpu.
    pub fn bounds_check_policies(&self) -> naga::proc::BoundsCheckPolicies {
        let mut p = bounds_check_policies();
        if self.uses_heap_buffers() {
            p.buffer = naga::proc::BoundsCheckPolicy::Unchecked;
        }
        p
    }

    /// Whether any entry point reads or writes a buffer through the heap.
    pub fn uses_heap_buffers(&self) -> bool {
        (0..self.module.entry_points.len()).any(|i| {
            let info = self.info.get_entry_point(i);
            self.module
                .global_variables
                .iter()
                .any(|(h, v)| !info[h].is_empty() && v.binding.is_some_and(|b| b.group == HEAP_GROUP && heap::is_buffer(b.binding)))
        })
    }

    /// The module with `override`s resolved for one entry point, ready for a writer.
    /// The root bindings entry point `name` of `stage` uses, sorted by (group, binding). They
    /// are found by variable, not by number: openOMSI's scene module has a texture and a
    /// uniform buffer on one (group, binding), each used by other entry points.
    pub fn entry_point_roots(&self, stage: naga::ShaderStage, name: &str) -> Result<Vec<RootBinding>, ShaderError> {
        let index = self
            .module
            .entry_points
            .iter()
            .position(|e| e.stage == stage && e.name == name)
            .ok_or_else(|| ShaderError::NoEntryPoint {
                label: self.label.clone(),
                name: name.into(),
                stage,
            })?;
        let info = self.info.get_entry_point(index);
        Ok(self.root_vars.iter().filter(|&&(h, _)| !info[h].is_empty()).map(|&(_, r)| r).collect())
    }

    pub fn specialize(&self, stage: naga::ShaderStage, entry: &str, constants: &[(&str, f64)]) -> Result<(naga::Module, naga::valid::ModuleInfo), ShaderError> {
        let constants: naga::back::PipelineConstants = constants.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let (module, info) = naga::back::pipeline_constants::process_overrides(&self.module, &self.info, Some((stage, entry)), &constants).map_err(|e| ShaderError::Constants {
            label: self.label.clone(),
            message: e.to_string(),
        })?;
        Ok((module.into_owned(), info.into_owned()))
    }
}

fn reflect_roots(label: &str, module: &naga::Module) -> Result<Vec<(naga::Handle<naga::GlobalVariable>, RootBinding)>, ShaderError> {
    let layout_err = |message: String| ShaderError::Layout { label: label.into(), message };
    let mut roots = Vec::new();
    for (handle, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
        let inner = &module.types[var.ty].inner;
        if rb.group == HEAP_GROUP {
            if !matches!(inner, naga::TypeInner::BindingArray { .. }) {
                return Err(layout_err(format!(
                    "group {HEAP_GROUP} is the bindless heap; `{}` cannot be declared in it",
                    var.name.as_deref().unwrap_or("?")
                )));
            }
            continue;
        }
        if rb.group > MAX_ROOT_GROUP {
            return Err(layout_err(format!(
                "root bindings use groups 0..={MAX_ROOT_GROUP}; `{}` is in group {}",
                var.name.as_deref().unwrap_or("?"),
                rb.group
            )));
        }
        let kind = match (var.space, inner) {
            (naga::AddressSpace::Uniform, _) => RootKind::UniformBuffer,
            (naga::AddressSpace::Storage { access }, _) => RootKind::StorageBuffer {
                read_only: !access.contains(naga::StorageAccess::STORE),
            },
            (
                _,
                naga::TypeInner::Image {
                    class: naga::ImageClass::Storage { .. },
                    ..
                },
            ) => RootKind::StorageTexture,
            (_, naga::TypeInner::Image { .. }) => RootKind::Texture,
            (_, naga::TypeInner::Sampler { .. }) => RootKind::Sampler,
            (_, naga::TypeInner::AccelerationStructure { .. }) => RootKind::AccelerationStructure,
            (_, naga::TypeInner::BindingArray { .. }) => {
                return Err(layout_err(format!(
                    "`{}`: binding arrays belong to the heap; index `mx_tex2d` etc. instead",
                    var.name.as_deref().unwrap_or("?")
                )));
            }
            _ => return Err(layout_err(format!("`{}`: unsupported resource type", var.name.as_deref().unwrap_or("?")))),
        };
        let runtime_sized = matches!(kind, RootKind::StorageBuffer { .. }) && ends_in_runtime_array(module, var.ty);
        roots.push((
            handle,
            RootBinding {
                group: rb.group,
                binding: rb.binding,
                kind,
                runtime_sized,
            },
        ));
    }
    roots.sort_by_key(|(_, r)| (r.group, r.binding));
    Ok(roots)
}

/// The heap bindings of the globals `used` keeps, a bit per binding.
pub fn heap_used(module: &naga::Module, used: impl Fn(naga::Handle<naga::GlobalVariable>) -> bool) -> u64 {
    module
        .global_variables
        .iter()
        .filter(|&(h, _)| used(h))
        .filter_map(|(_, v)| v.binding)
        // (the bound tier's slot globals, `bound::slot_binding`, are not heap arrays)
        .filter(|b| b.group == HEAP_GROUP && b.binding <= heap::LAST_BUFFER_VIEW)
        .fold(0, |bits, b| bits | 1 << b.binding)
}

fn reflect_heap_buffers(label: &str, module: &naga::Module) -> Result<Vec<HeapBuffer>, ShaderError> {
    let layout_err = |message: String| ShaderError::Layout { label: label.into(), message };
    let mut views = Vec::new();
    for (_, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
        if rb.group != HEAP_GROUP {
            continue;
        }
        let name = var.name.as_deref().unwrap_or("?");
        let is_view = heap::is_buffer(rb.binding);
        let prelude = rb.binding < heap::COUNT;
        if !prelude && !is_view {
            return Err(layout_err(format!(
                "`{name}`: heap bindings {}..{} are reserved; give a buffer view a binding in {}..={}",
                heap::COUNT,
                heap::FIRST_BUFFER_VIEW,
                heap::FIRST_BUFFER_VIEW,
                heap::LAST_BUFFER_VIEW
            )));
        }
        if !is_view {
            continue;
        }
        let naga::AddressSpace::Storage { access } = var.space else {
            return Err(layout_err(format!("`{name}`: a view of the buffer heap is `var<storage, read>` or `var<storage, read_write>`")));
        };
        let naga::TypeInner::BindingArray { base, .. } = module.types[var.ty].inner else {
            return Err(layout_err(format!("`{name}`: a view of the buffer heap is a `binding_array`")));
        };
        if !matches!(module.types[base].inner, naga::TypeInner::Struct { .. }) {
            return Err(layout_err(format!("`{name}`: the elements of a buffer view are structs")));
        }
        if views.iter().any(|v: &HeapBuffer| v.binding == rb.binding) {
            return Err(layout_err(format!("`{name}`: heap binding {} is declared twice", rb.binding)));
        }
        views.push(HeapBuffer {
            binding: rb.binding,
            read_only: !access.contains(naga::StorageAccess::STORE),
        });
    }
    reject_heap_array_length(label, module)?;
    views.sort_by_key(|v| v.binding);
    Ok(views)
}

/// `arrayLength` of a heap buffer would need the length of every buffer in the heap passed to
/// every draw on Metal; a bindless shader keeps its counts in its own data instead.
fn reject_heap_array_length(label: &str, module: &naga::Module) -> Result<(), ShaderError> {
    let functions = module.functions.iter().map(|(_, f)| f).chain(module.entry_points.iter().map(|e| &e.function));
    for f in functions {
        for (_, expr) in f.expressions.iter() {
            let naga::Expression::ArrayLength(mut e) = *expr else { continue };
            loop {
                match f.expressions[e] {
                    naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => e = base,
                    naga::Expression::GlobalVariable(g) => {
                        let var = &module.global_variables[g];
                        if var.binding.is_some_and(|b| b.group == HEAP_GROUP && heap::is_buffer(b.binding)) {
                            return Err(ShaderError::Layout {
                                label: label.into(),
                                message: format!(
                                    "`arrayLength` of a heap buffer (`{}`): keep the count in the buffer or the immediates",
                                    var.name.as_deref().unwrap_or("?")
                                ),
                            });
                        }
                        break;
                    }
                    _ => break,
                }
            }
        }
    }
    Ok(())
}

fn ends_in_runtime_array(module: &naga::Module, ty: naga::Handle<naga::Type>) -> bool {
    match &module.types[ty].inner {
        naga::TypeInner::Array { size: naga::ArraySize::Dynamic, .. } => true,
        naga::TypeInner::Struct { members, .. } => members.last().is_some_and(|m| ends_in_runtime_array(module, m.ty)),
        _ => false,
    }
}

/// The bounds checks wgpu-hal applies to a module made with `create_shader_module` (checked,
/// as openOMSI makes all of its), so out-of-range reads give what they gave under wgpu:
/// `Restrict` clamps an index or a texel coordinate into range. This is wgpu-hal's choice on
/// Metal, and on Vulkan without robust buffer / image access; where a device has those,
/// wgpu-hal's Vulkan backend leaves the matching checks to the driver, and a backend must
/// do as wgpu-hal does on its API (see the backend's own module). The heap itself is
/// unchecked: an index is the renderer's responsibility, as in every bindless engine (and
/// wgpu-hal does not check binding arrays either).
pub fn bounds_check_policies() -> naga::proc::BoundsCheckPolicies {
    use naga::proc::BoundsCheckPolicy as P;
    naga::proc::BoundsCheckPolicies {
        index: P::Restrict,
        buffer: P::Restrict,
        image_load: P::Restrict,
        binding_array: P::Unchecked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelude_is_usable_and_lines_are_kept() {
        let m = Module::from_wgsl(
            "t",
            "struct D { t: u32, s: u32 }\nvar<immediate> d: D;\n@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return textureSample(mx_tex2d[d.t], mx_samplers[d.s], uv); }",
        )
        .unwrap();
        assert!(m.roots.is_empty());
        let err = Module::from_wgsl("t", "\n\nfn f() { let x: u32 = 1.5; }").unwrap_err().to_string();
        assert!(err.contains(":3:"), "{err}");
    }

    #[test]
    fn root_bindings_are_reflected() {
        let m = Module::from_wgsl(
            "t",
            "struct C { m: mat4x4<f32> }\n@group(0) @binding(0) var<uniform> cam: C;\n@group(1) @binding(2) var<storage, read> xs: array<f32>;\n@group(0) @binding(5) var t: texture_depth_2d;\n@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> { _ = textureDimensions(t); return cam.m * vec4<f32>(xs[i]); }",
        )
        .unwrap();
        let r: Vec<_> = m.roots.iter().map(|r| (r.group, r.binding, r.kind, r.runtime_sized)).collect();
        assert_eq!(
            r,
            vec![
                (0, 0, RootKind::UniformBuffer, false),
                (0, 5, RootKind::Texture, false),
                (1, 2, RootKind::StorageBuffer { read_only: true }, true)
            ]
        );
    }

    #[test]
    fn roots_sharing_a_binding_are_told_apart_by_variable() {
        // valid WGSL: no entry point uses both `depth` and `motion`
        let m = Module::from_wgsl(
            "t",
            "@group(3) @binding(0) var depth: texture_depth_2d;\n@group(3) @binding(0) var<uniform> motion: vec4<f32>;\n@vertex fn vs() -> @builtin(position) vec4<f32> { return motion; }\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(f32(textureDimensions(depth).x)); }",
        )
        .unwrap();
        let kinds = |stage, name| m.entry_point_roots(stage, name).unwrap().iter().map(|r| (r.group, r.binding, r.kind)).collect::<Vec<_>>();
        assert_eq!(kinds(naga::ShaderStage::Vertex, "vs"), [(3, 0, RootKind::UniformBuffer)]);
        assert_eq!(kinds(naga::ShaderStage::Fragment, "fs"), [(3, 0, RootKind::Texture)]);
    }

    #[test]
    fn buffer_views_are_reflected_and_checked() {
        let m = Module::from_wgsl(
            "t",
            "struct L { l: array<vec4<f32>> }\n@group(7) @binding(17) var<storage, read_write> out: binding_array<L>;\n@group(7) @binding(16) var<storage, read> lights: binding_array<L>;\nstruct D { a: u32 }\nvar<immediate> d: D;\n@compute @workgroup_size(1) fn main() { out[d.a].l[0] = lights[d.a].l[1] + vec4<f32>(f32(mx_buffers[d.a].w[2])); }",
        )
        .unwrap();
        let views: Vec<_> = m.heap_buffers.iter().map(|v| (v.binding, v.read_only)).collect();
        assert_eq!(views, [(heap::BUFFERS, true), (heap::RW_BUFFERS, false), (16, true), (17, false)]);
        assert!(m.uses_heap_buffers());
        assert_eq!(m.bounds_check_policies().buffer, naga::proc::BoundsCheckPolicy::Unchecked);
        let used = m.entry_point_heap(naga::ShaderStage::Compute, "main").unwrap();
        assert_eq!(used, 1 << heap::BUFFERS | 1 << 16 | 1 << 17);

        let plain = Module::from_wgsl("t", "@compute @workgroup_size(1) fn main() {}").unwrap();
        assert!(!plain.uses_heap_buffers());
        assert_eq!(plain.bounds_check_policies(), bounds_check_policies());

        let bad = [
            // reserved binding
            "struct L { l: array<u32> }\n@group(7) @binding(12) var<storage, read> x: binding_array<L>;",
            // not storage
            "@group(7) @binding(16) var x: binding_array<texture_2d<f32>>;",
            // elements that are not structs
            "@group(7) @binding(16) var<storage, read> x: binding_array<array<u32>>;",
            // arrayLength of a heap buffer
            "@compute @workgroup_size(1) fn main() { mx_rw_buffers[0].w[0] = arrayLength(&mx_buffers[1].w); }",
        ];
        for src in bad {
            assert!(matches!(Module::from_wgsl("t", src), Err(ShaderError::Layout { .. }) | Err(ShaderError::Validate { .. })), "{src}");
        }
    }

    #[test]
    fn the_heap_group_is_reserved() {
        let e = Module::from_wgsl("t", "@group(7) @binding(9) var<uniform> x: vec4<f32>;").unwrap_err();
        assert!(matches!(e, ShaderError::Layout { .. }), "{e}");
    }
}

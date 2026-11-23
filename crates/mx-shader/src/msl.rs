//! Metal Shading Language output and mX's Metal slot layout.
//!
//! | buffer slot | use |
//! |-------------|-----|
//! | 0           | immediates |
//! | 1           | sizes of runtime-sized root storage buffers, then of the vertex buffers |
//! | 2..=22      | the heap bindings the entry point uses, in binding order, each an argument buffer of `MTLResourceID`s (textures, samplers) or of GPU addresses (buffers); then its root buffers, in (group, binding) order |
//! | 23..=30     | vertex buffers, `30 - index` |
//!
//! Root textures and samplers take texture / sampler slots from 0 in (group, binding) order.
//! Each entry point gets its own assignment and counts only what it uses, so a shader that
//! reads `mx_tex2d`, `mx_samplers` and two buffer views has 17 slots left for root buffers.

use std::collections::BTreeMap;

use crate::{HEAP_GROUP, Module, RootKind, ShaderError, heap};
use naga::back::msl;

pub use msl::{AttributeMapping, VertexBufferMapping, VertexBufferStepMode};
pub use naga_types::VertexFormat;

pub const IMMEDIATES_SLOT: u32 = 0;
pub const SIZES_SLOT: u32 = 1;
/// The first slot of the heap bindings, then the root buffers.
pub const FIRST_HEAP_SLOT: u32 = 2;
/// The slots heap bindings and root buffers share.
pub const SHARED_SLOTS: u32 = 21;
pub const MAX_ROOT_BUFFERS: u32 = 13;
pub const MAX_VERTEX_BUFFERS: u32 = 8;
pub const MAX_ROOT_TEXTURES: u32 = 96;
pub const MAX_ROOT_SAMPLERS: u32 = 16;

pub const fn vertex_buffer_slot(index: u32) -> u32 {
    30 - index
}

/// Where a root binding lives in Metal's argument tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootSlot {
    Buffer {
        slot: u32,
        kind: RootKind,
    },
    Texture(u32),
    Sampler(u32),
    /// A top-level acceleration structure (ray queries), in a buffer slot.
    AccelerationStructure(u32),
}

/// What a heap binding's argument buffer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeapTable {
    Textures,
    Samplers,
    Buffers,
}

impl HeapTable {
    pub const fn of(binding: u32) -> HeapTable {
        if heap::is_texture(binding) {
            HeapTable::Textures
        } else if heap::is_sampler(binding) {
            HeapTable::Samplers
        } else {
            HeapTable::Buffers
        }
    }
}

/// The Metal slots of the heap bindings and root bindings of a module or an entry point.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    /// Heap binding -> buffer slot.
    pub heap: BTreeMap<u32, u32>,
    pub roots: BTreeMap<(u32, u32), RootSlot>,
}

impl Layout {
    /// Slots for every heap and root binding of the module.
    pub fn for_module(m: &Module) -> Result<Layout, ShaderError> {
        let heap = (0..m.module.entry_points.len()).fold(0, |bits, i| {
            let info = m.info.get_entry_point(i);
            bits | crate::heap_used(&m.module, |h| !info[h].is_empty())
        });
        Self::assign(m, heap, m.roots.iter())
    }

    /// Slots for what one entry point uses. As in wgpu, Metal's limits count what the entry
    /// point reads, not everything its module declares (openOMSI's motion vectors share the
    /// scene module, which declares more buffers than one stage takes).
    pub fn for_entry_point(m: &Module, stage: naga::ShaderStage, name: &str) -> Result<Layout, ShaderError> {
        Self::assign(m, m.entry_point_heap(stage, name)?, m.entry_point_roots(stage, name)?.iter())
    }

    /// The (slot, table) pairs a pipeline binds for its heap.
    pub fn heap_tables(&self) -> impl Iterator<Item = (u32, HeapTable)> + '_ {
        self.heap.iter().map(|(&b, &slot)| (slot, HeapTable::of(b)))
    }

    fn assign<'a>(m: &Module, heap_bits: u64, list: impl Iterator<Item = &'a crate::RootBinding>) -> Result<Layout, ShaderError> {
        let mut heap = BTreeMap::new();
        let mut next = FIRST_HEAP_SLOT;
        for b in 0..64 {
            if heap_bits & (1 << b) != 0 {
                heap.insert(b, next);
                next += 1;
            }
        }
        let heap_slots = next - FIRST_HEAP_SLOT;
        let (mut b, mut t, mut s) = (0, 0, 0);
        let mut roots = BTreeMap::new();
        for r in list {
            let slot = match r.kind {
                RootKind::UniformBuffer | RootKind::StorageBuffer { .. } => {
                    b += 1;
                    RootSlot::Buffer { slot: next + b - 1, kind: r.kind }
                }
                RootKind::Texture | RootKind::StorageTexture => {
                    t += 1;
                    RootSlot::Texture(t - 1)
                }
                RootKind::Sampler => {
                    s += 1;
                    RootSlot::Sampler(s - 1)
                }
                RootKind::AccelerationStructure => {
                    b += 1;
                    RootSlot::AccelerationStructure(next + b - 1)
                }
            };
            roots.insert((r.group, r.binding), slot);
        }
        let err = |message: String| ShaderError::Layout { label: m.label.clone(), message };
        let over = |n: u32, max: u32, what: &str| (n > max).then(|| err(format!("{n} root {what}; Metal takes {max}")));
        if let Some(e) = over(b, MAX_ROOT_BUFFERS, "buffers")
            .or(over(t, MAX_ROOT_TEXTURES, "textures"))
            .or(over(s, MAX_ROOT_SAMPLERS, "samplers"))
        {
            return Err(e);
        }
        if heap_slots + b > SHARED_SLOTS {
            return Err(err(format!("{heap_slots} heap bindings and {b} root buffers; Metal has {SHARED_SLOTS} buffer slots for both")));
        }
        Ok(Layout { heap, roots })
    }
}

/// One entry point in MSL.
#[derive(Clone, Debug)]
pub struct Output {
    pub source: String,
    /// The function's name in `source` (naga renames names MSL reserves).
    pub entry_point: String,
    /// Which heap bindings the entry point reads (bit per `heap::*` and buffer view).
    pub heap_bindings_used: u64,
    pub uses_immediates: bool,
    /// The root buffers whose byte sizes go to `SIZES_SLOT`, as consecutive `u32`s in this
    /// order (naga's: the order of declaration among the globals this entry point keeps). A
    /// heap buffer view in the list (group `HEAP_GROUP`) takes a `u32` that nothing reads.
    pub sized_buffers: Vec<(u32, u32)>,
    pub workgroup_size: [u32; 3],
    /// Bytes of each `var<workgroup>` the entry point uses, in threadgroup-memory index
    /// order: Metal needs each length set before a dispatch (as wgpu-hal sets it).
    pub workgroup_memory: Vec<u32>,
}

/// Write one entry point of `m` as MSL `version` (3.0 at least; wgpu-hal picks it by OS
/// version, see `mx_metal`), with `constants` applied to its `override`s. A vertex stage
/// fetches its attributes from `vertex_buffers` itself (naga's vertex pulling, as wgpu-hal
/// compiles every vertex shader): the conversions of normalised formats are then the
/// shader's arithmetic, not Metal's vertex fetch, which differs from it by an ulp.
pub fn write(
    m: &Module,
    layout: &Layout,
    stage: naga::ShaderStage,
    entry: Option<&str>,
    constants: &[(&str, f64)],
    version: (u8, u8),
    vertex_buffers: &[VertexBufferMapping],
) -> Result<Output, ShaderError> {
    let name = m.entry_point(stage, entry)?.name.clone();
    let (module, info) = m.specialize(stage, &name, constants)?;
    emit(m, layout, stage, &name, &module, &info, version, vertex_buffers)
}

/// A heap texture or sampler the direct form binds itself (`write_direct`): the slot, and the
/// texture or sampler slot it takes in Metal's argument tables.
#[derive(Clone, Copy, Debug)]
pub struct DirectSlot {
    pub slot: crate::bound::Slot,
    pub root: RootSlot,
}

/// One entry point in MSL with its heap accesses bound directly: where every heap texture,
/// sampler and buffer view it reads is indexed by an immediate (or a constant) - a material's
/// textures, a frame's buffers - each becomes a plain texture, sampler or buffer argument the
/// backend binds at draw time, from the immediates, instead of an index into the heap's
/// argument buffers. Apple's GPUs read a texture argument directly; through the argument
/// buffer each sample loads its descriptor first (the scene's main pass took an eighth
/// longer). None where the entry point needs real bindless (an index read from memory), or
/// reads buffer views and `buffers` is false: the bindless form (`write`) then.
#[allow(clippy::too_many_arguments)]
pub fn write_direct(
    m: &Module,
    stage: naga::ShaderStage,
    entry: Option<&str>,
    constants: &[(&str, f64)],
    version: (u8, u8),
    vertex_buffers: &[VertexBufferMapping],
    buffers: bool,
) -> Result<Option<Direct>, ShaderError> {
    write_lowered(m, stage, entry, constants, version, vertex_buffers, buffers).map(|r| r.ok())
}

/// An entry point written in the direct form: its source, layout and the heap slots it binds.
pub type Direct = (Output, Layout, Vec<DirectSlot>);

/// [`write_direct`], with why an entry point cannot be written so (Metal 2, where there is no
/// other way: the bound tier).
#[allow(clippy::too_many_arguments)]
pub fn write_lowered(
    m: &Module,
    stage: naga::ShaderStage,
    entry: Option<&str>,
    constants: &[(&str, f64)],
    version: (u8, u8),
    vertex_buffers: &[VertexBufferMapping],
    buffers: bool,
) -> Result<Result<Direct, String>, ShaderError> {
    let name = m.entry_point(stage, entry)?.name.clone();
    if m.entry_point_heap(stage, &name)? == 0 {
        return Ok(Err("it reads no heap".into()));
    }
    let (module, _) = m.specialize(stage, &name, constants)?;
    let mut lowered = match crate::bound::lower(m, module, stage, &name) {
        Ok(l) => l,
        Err(e) => return Ok(Err(e.to_string())),
    };
    // (a frame's constants read through a buffer view stay in Metal's constant address space,
    // where Apple's GPUs read uniforms fastest, as a root uniform buffer's do)
    crate::bound::views_as_uniforms(&mut lowered);
    if !buffers && lowered.slots.iter().any(|s| heap::is_buffer(s.heap_binding)) {
        return Ok(Err("it reads heap buffers".into()));
    }
    let mut layout = Layout::for_entry_point(m, stage, &name)?;
    layout.heap.clear();
    let mut textures = layout.roots.values().filter(|r| matches!(r, RootSlot::Texture(_))).count() as u32;
    let mut samplers = layout.roots.values().filter(|r| matches!(r, RootSlot::Sampler(_))).count() as u32;
    // (buffer slots after the root buffers'; the heap's own slots are not used)
    let mut next_buffer = layout
        .roots
        .values()
        .filter_map(|r| match r {
            RootSlot::Buffer { slot, .. } | RootSlot::AccelerationStructure(slot) => Some(*slot + 1),
            _ => None,
        })
        .max()
        .unwrap_or(FIRST_HEAP_SLOT);
    let mut direct = Vec::with_capacity(lowered.slots.len());
    for slot in &lowered.slots {
        let root = if heap::is_sampler(slot.heap_binding) {
            samplers += 1;
            RootSlot::Sampler(samplers - 1)
        } else if heap::is_buffer(slot.heap_binding) {
            next_buffer += 1;
            RootSlot::Buffer {
                slot: next_buffer - 1,
                kind: if slot.uniform { RootKind::UniformBuffer } else { RootKind::StorageBuffer { read_only: false } },
            }
        } else {
            textures += 1;
            RootSlot::Texture(textures - 1)
        };
        layout.roots.insert((HEAP_GROUP, slot.binding), root);
        direct.push(DirectSlot { slot: *slot, root });
    }
    if textures > MAX_ROOT_TEXTURES || samplers > MAX_ROOT_SAMPLERS {
        return Ok(Err(format!("{textures} textures and {samplers} samplers; Metal binds {MAX_ROOT_TEXTURES} and {MAX_ROOT_SAMPLERS}")));
    }
    if next_buffer > FIRST_HEAP_SLOT + SHARED_SLOTS {
        return Ok(Err(format!("{} buffers; Metal has {SHARED_SLOTS} buffer slots", next_buffer - FIRST_HEAP_SLOT)));
    }
    let out = emit(m, &layout, stage, &name, &lowered.module, &lowered.info, version, vertex_buffers)?;
    Ok(Ok((out, layout, direct)))
}

#[allow(clippy::too_many_arguments)]
fn emit(
    m: &Module,
    layout: &Layout,
    stage: naga::ShaderStage,
    name: &str,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    version: (u8, u8),
    vertex_buffers: &[VertexBufferMapping],
) -> Result<Output, ShaderError> {
    let name = name.to_string();
    let backend_err = |message: String| ShaderError::Backend {
        label: m.label.clone(),
        target: "MSL",
        message,
    };

    let mut resources = BTreeMap::new();
    for (&b, &slot) in &layout.heap {
        let target = msl::BindTarget {
            buffer: Some(slot as u8),
            // (a view of the buffer heap may be written)
            mutable: heap::is_buffer(b),
            ..Default::default()
        };
        resources.insert(naga::ResourceBinding { group: HEAP_GROUP, binding: b }, target);
    }
    for (&(group, binding), slot) in &layout.roots {
        let target = match *slot {
            RootSlot::Buffer { slot, .. } | RootSlot::AccelerationStructure(slot) => msl::BindTarget {
                buffer: Some(slot as u8),
                ..Default::default()
            },
            RootSlot::Texture(t) => msl::BindTarget {
                texture: Some(t as u8),
                mutable: true,
                ..Default::default()
            },
            RootSlot::Sampler(s) => msl::BindTarget {
                sampler: Some(msl::BindSamplerTarget::Resource(s as u8)),
                ..Default::default()
            },
        };
        resources.insert(naga::ResourceBinding { group, binding }, target);
    }
    let ep_resources = msl::EntryPointResources {
        resources,
        immediates_buffer: Some(IMMEDIATES_SLOT as u8),
        sizes_buffer: Some(SIZES_SLOT as u8),
    };

    let mut options = msl::Options {
        lang_version: version,
        bounds_check_policies: m.bounds_check_policies(),
        zero_initialize_workgroup_memory: true,
        force_loop_bounding: true,
        fake_missing_bindings: false,
        ..Default::default()
    };
    options.per_entry_point_map.insert(name.clone(), ep_resources);

    let pipeline = msl::PipelineOptions {
        entry_point: Some((stage, name.clone())),
        vertex_pulling_transform: stage == naga::ShaderStage::Vertex,
        vertex_buffer_mappings: vertex_buffers.to_vec(),
        ..Default::default()
    };
    let (source, tinfo) = msl::write_string(module, info, &options, &pipeline).map_err(|e| backend_err(e.to_string()))?;

    let (ep_index, ep) = module
        .entry_points
        .iter()
        .enumerate()
        .find(|(_, e)| e.stage == stage && e.name == name)
        .ok_or_else(|| backend_err("entry point lost".into()))?;
    let entry_point = match tinfo.entry_point_names.get(ep_index) {
        Some(Ok(n)) => n.clone(),
        Some(Err(e)) => return Err(backend_err(e.to_string())),
        // (only the asked entry point is written, so it is the only name)
        None => tinfo.entry_point_names.iter().flatten().next().cloned().ok_or_else(|| backend_err("no entry point written".into()))?,
    };

    let ep_info = info.get_entry_point(ep_index);
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).map_err(|e| backend_err(e.to_string()))?;
    let workgroup_memory = module
        .global_variables
        .iter()
        .filter(|(h, v)| v.space == naga::AddressSpace::WorkGroup && !ep_info[*h].is_empty())
        .map(|(_, v)| layouter[v.ty].size)
        .collect();
    let heap_bindings_used = crate::heap_used(module, |h| !ep_info[h].is_empty());
    let uses_immediates = module.global_variables.iter().any(|(h, v)| v.space == naga::AddressSpace::Immediate && !ep_info[h].is_empty());
    // (a heap buffer view takes one `u32` here too, which nothing reads: heap buffers are
    // unchecked and `arrayLength` of them is refused, see `Module::bounds_check_policies`)
    let sized_buffers = module
        .global_variables
        .iter()
        .filter(|(_, v)| module.types[v.ty].inner.needs_host_buffer_byte_size(&module.types))
        .filter_map(|(_, v)| v.binding.map(|b| (b.group, b.binding)))
        .collect();
    Ok(Output {
        source,
        entry_point,
        heap_bindings_used,
        uses_immediates,
        sized_buffers,
        workgroup_size: ep.workgroup_size,
        workgroup_memory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_and_roots_land_in_their_slots() {
        let m = Module::from_wgsl(
            "t",
            "struct C { k: vec4<f32> }\n@group(0) @binding(0) var<uniform> c: C;\n@group(2) @binding(1) var<storage, read> xs: array<f32>;\nstruct D { t: u32 }\nvar<immediate> d: D;\n@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return c.k * xs[0] * textureSample(mx_tex2d[d.t], mx_samplers[0], uv); }",
        )
        .unwrap();
        let layout = Layout::for_module(&m).unwrap();
        let out = write(&m, &layout, naga::ShaderStage::Fragment, None, &[], (3, 0), &[]).unwrap();
        // immediates 0, sizes 1, mx_tex2d 2, mx_samplers 3, then the roots
        for slot in 0..=5 {
            assert!(out.source.contains(&format!("[[buffer({slot})]]")), "slot {slot}: {}", out.source);
        }
        assert_eq!(layout.heap, BTreeMap::from([(heap::TEX_2D, 2), (heap::SAMPLERS, 3)]));
        assert_eq!(out.heap_bindings_used, 1 << heap::TEX_2D | 1 << heap::SAMPLERS);
        assert!(out.uses_immediates);
    }

    #[test]
    fn buffer_views_are_argument_buffers_of_addresses() {
        let m = Module::from_wgsl(
            "t",
            "struct L { l: array<vec4<f32>> }\n@group(7) @binding(16) var<storage, read> lights: binding_array<L>;\nstruct C { c: atomic<u32> }\n@group(7) @binding(18) var<storage, read_write> counters: binding_array<C>;\n@group(0) @binding(0) var<storage, read> xs: array<f32>;\nstruct D { a: u32 }\nvar<immediate> d: D;\n@compute @workgroup_size(1) fn main() { atomicAdd(&counters[d.a].c, u32(lights[d.a].l[0].x + xs[0])); }",
        )
        .unwrap();
        let layout = Layout::for_entry_point(&m, naga::ShaderStage::Compute, "main").unwrap();
        assert_eq!(layout.heap, BTreeMap::from([(16, 2), (18, 3)]));
        assert_eq!(layout.heap_tables().map(|(_, t)| t).collect::<Vec<_>>(), [HeapTable::Buffers, HeapTable::Buffers]);
        assert_eq!(
            layout.roots[&(0, 0)],
            RootSlot::Buffer {
                slot: 4,
                kind: RootKind::StorageBuffer { read_only: true }
            }
        );
        let out = write(&m, &layout, naga::ShaderStage::Compute, None, &[], (3, 0), &[]).unwrap();
        assert!(out.source.contains("NagaArgumentBufferWrapper<device L*>* lights [[buffer(2)]]"), "{}", out.source);
        assert!(out.source.contains("[[buffer(3)]]"), "{}", out.source);
        // in naga's order (declaration), heap views included: their entries are never read
        assert_eq!(out.sized_buffers[..2], [(HEAP_GROUP, 16), (0, 0)]);
        let fields = out.source.split("struct _mslBufferSizes {").nth(1).unwrap().split("};").next().unwrap().matches(';').count();
        assert_eq!(fields, out.sized_buffers.len(), "{}", out.source);
    }

    #[test]
    fn overrides_are_applied() {
        let m = Module::from_wgsl("t", "override K: f32 = 1.0;\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(K); }").unwrap();
        let layout = Layout::for_module(&m).unwrap();
        let out = write(&m, &layout, naga::ShaderStage::Fragment, Some("fs"), &[("K", 0.25)], (3, 0), &[]).unwrap();
        assert!(out.source.contains("0.25"), "{}", out.source);
    }
}
#[cfg(test)]
#[test]
fn sizes_follow_declaration_order() {
    let m = Module::from_wgsl(
        "t",
        "@group(0) @binding(3) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read> b: array<u32>;\n@group(0) @binding(0) var<storage, read> unused: array<u32>;\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(a[arrayLength(&a)-1u] + f32(b[2])); }\n@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(f32(unused[0])); }",
    )
    .unwrap();
    let l = Layout::for_module(&m).unwrap();
    let out = write(&m, &l, naga::ShaderStage::Fragment, None, &[], (3, 0), &[]).unwrap();
    assert_eq!(out.sized_buffers, vec![(0, 3), (0, 1)], "{}", out.source);
}

#[cfg(test)]
#[test]
fn ray_queries_get_a_buffer_slot() {
    let m = Module::from_wgsl(
        "t",
        "enable wgpu_ray_query;\n@group(0) @binding(1) var acc: acceleration_structure;\n@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(1) fn main() { var rq: ray_query; rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 10.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0))); rayQueryProceed(&rq); out[0] = rayQueryGetCommittedIntersection(&rq).instance_custom_data; }",
    )
    .unwrap();
    let l = Layout::for_module(&m).unwrap();
    assert_eq!(l.roots.get(&(0, 1)), Some(&RootSlot::AccelerationStructure(FIRST_HEAP_SLOT + 1)));
    let out = write(&m, &l, naga::ShaderStage::Compute, None, &[], (3, 0), &[]).unwrap();
    assert!(out.source.contains("instance_acceleration_structure"), "{}", out.source);
}

#[test]
fn each_entry_point_counts_only_the_roots_it_uses() {
    // 14 buffers in the module, 13 at most per entry point: as in wgpu, this compiles
    let decls: String = (0..14).map(|i| format!("@group(0) @binding({i}) var<uniform> u{i}: vec4<f32>;\n")).collect();
    let src = format!("{decls}@vertex fn vs() -> @builtin(position) vec4<f32> {{ return u0 + u13; }}\n@fragment fn fs() -> @location(0) vec4<f32> {{ return u1; }}");
    let m = Module::from_wgsl("t", &src).unwrap();
    assert!(Layout::for_module(&m).is_err());
    let vs = Layout::for_entry_point(&m, naga::ShaderStage::Vertex, "vs").unwrap();
    assert_eq!(vs.roots.keys().copied().collect::<Vec<_>>(), [(0, 0), (0, 13)]);
    assert_eq!(
        vs.roots[&(0, 13)],
        RootSlot::Buffer {
            slot: FIRST_HEAP_SLOT + 1,
            kind: RootKind::UniformBuffer
        }
    );
    let fs = Layout::for_entry_point(&m, naga::ShaderStage::Fragment, "fs").unwrap();
    assert_eq!(fs.roots.keys().copied().collect::<Vec<_>>(), [(0, 1)]);
}

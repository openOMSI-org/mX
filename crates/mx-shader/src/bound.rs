//! The heap without bindless: lowering heap accesses to fixed slots for the bound tier
//! (OpenGL, and later Direct3D 11 and Metal 2; see `docs/BOUND_TIER.md`).
//!
//! An access to a heap array whose index is a member of the immediates, `mx_tex2d[draw.tex]`,
//! becomes a plain global of the element type, a *slot*, and the slot records the byte offset
//! of that member. At draw time the backend reads the index there and binds what the heap holds
//! at it. A constant index (`mx_samplers[0]`) becomes a slot with that index. Any other index
//! needs bindless and is reported as an error.

use std::collections::HashMap;

use naga::{AddressSpace, Expression, Handle};

use crate::{HEAP_GROUP, Module, ShaderError, heap};

/// The binding of slot `n` in the heap group (above every heap binding a shader may declare).
pub const fn slot_binding(n: u32) -> u32 {
    heap::LAST_BUFFER_VIEW + 1 + n
}

/// Where a slot's heap index comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Index {
    /// The `u32` at this byte offset of the immediates.
    Immediate(u32),
    /// This index, written in the shader.
    Constant(u32),
}

/// One slot: a heap array element the shader reads at a fixed binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot {
    /// The heap binding it stands for (`heap::TEX_2D`, `heap::SAMPLERS`, a buffer view, ...).
    pub heap_binding: u32,
    pub index: Index,
    /// Its binding in the heap group ([`slot_binding`]).
    pub binding: u32,
    /// A read-only buffer view read as a uniform block (see [`views_as_uniforms`]).
    pub uniform: bool,
}

/// A module with its heap accesses lowered to slots.
pub struct Lowered {
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
    pub slots: Vec<Slot>,
}

/// Lower the heap accesses of one entry point (`module` and `info` already specialised for it,
/// see `Module::specialize`). `m` is the original module, for its label and immediates.
pub fn lower(m: &Module, mut module: naga::Module, stage: naga::ShaderStage, entry: &str) -> Result<Lowered, ShaderError> {
    let err = |message: String| ShaderError::Layout { label: m.label.clone(), message };
    let immediates = module.global_variables.iter().find(|(_, v)| v.space == AddressSpace::Immediate).map(|(h, _)| h);
    let heap_vars: HashMap<Handle<naga::GlobalVariable>, u32> = module
        .global_variables
        .iter()
        .filter_map(|(h, v)| v.binding.filter(|b| b.group == HEAP_GROUP && b.binding <= heap::LAST_BUFFER_VIEW).map(|b| (h, b.binding)))
        .collect();

    let mut slots: Vec<Slot> = Vec::new();
    let mut slot_vars: Vec<Handle<naga::GlobalVariable>> = Vec::new();
    let mut slot_of = |module: &mut naga::Module, heap_var: Handle<naga::GlobalVariable>, heap_binding: u32, index: Index| -> Handle<naga::GlobalVariable> {
        if let Some(i) = slots.iter().position(|s| s.heap_binding == heap_binding && s.index == index) {
            return slot_vars[i];
        }
        let n = slots.len() as u32;
        let var = &module.global_variables[heap_var];
        let naga::TypeInner::BindingArray { base, .. } = module.types[var.ty].inner else {
            unreachable!("heap globals are binding arrays")
        };
        let new = naga::GlobalVariable {
            name: Some(format!("{}_slot{n}", var.name.as_deref().unwrap_or("mx_heap"))),
            space: var.space,
            binding: Some(naga::ResourceBinding {
                group: HEAP_GROUP,
                binding: slot_binding(n),
            }),
            ty: base,
            init: None,
            memory_decorations: var.memory_decorations,
        };
        let h = module.global_variables.append(new, naga::Span::UNDEFINED);
        slots.push(Slot {
            heap_binding,
            index,
            binding: slot_binding(n),
            uniform: false,
        });
        slot_vars.push(h);
        h
    };

    // the functions to lower: every function and the entry point's
    let function_handles: Vec<Handle<naga::Function>> = module.functions.iter().map(|(h, _)| h).collect();
    let ep_index = module.entry_points.iter().position(|e| e.stage == stage && e.name == entry).ok_or_else(|| ShaderError::NoEntryPoint {
        label: m.label.clone(),
        name: entry.into(),
        stage,
    })?;
    for which in function_handles.into_iter().map(Some).chain(std::iter::once(None)) {
        let mut f = match which {
            Some(h) => std::mem::take(&mut module.functions[h]),
            None => std::mem::take(&mut module.entry_points[ep_index].function),
        };
        let mut replaced = Vec::new();
        let handles: Vec<Handle<Expression>> = f.expressions.iter().map(|(h, _)| h).collect();
        for h in handles {
            let (base, index) = match f.expressions[h] {
                Expression::Access { base, index } => (base, IndexExpr::Dynamic(index)),
                Expression::AccessIndex { base, index } => (base, IndexExpr::Constant(index)),
                _ => continue,
            };
            let Expression::GlobalVariable(g) = f.expressions[base] else { continue };
            let Some(&heap_binding) = heap_vars.get(&g) else { continue };
            let index = match index {
                IndexExpr::Constant(c) => Index::Constant(c),
                IndexExpr::Dynamic(e) => match constant(&f, e) {
                    Some(c) => Index::Constant(c),
                    None => immediate_offset(&f, &module, immediates, e).map(Index::Immediate).ok_or_else(|| {
                        err(format!(
                            "`{}[...]`: on this device (no bindless) a heap index must be a member of the immediates or a constant",
                            module.global_variables[g].name.as_deref().unwrap_or("heap")
                        ))
                    })?,
                },
            };
            let slot = slot_of(&mut module, g, heap_binding, index);
            f.expressions[h] = Expression::GlobalVariable(slot);
            replaced.push(h);
        }
        if !replaced.is_empty() {
            unemit(&mut f.body, &replaced, &f.expressions);
        }
        match which {
            Some(h) => module.functions[h] = f,
            None => module.entry_points[ep_index].function = f,
        }
    }

    let info = crate::validator().validate(&module).map_err(|e| ShaderError::Validate {
        label: m.label.clone(),
        message: format!("after lowering the heap: {}", e.as_inner()),
    })?;
    // what is still read from a heap array was reached some other way (a pointer passed to a
    // function, an array copied whole)
    let ep_info = info.get_entry_point(ep_index);
    if let Some((_, v)) = module.global_variables.iter().find(|(h, _)| heap_vars.contains_key(h) && !ep_info[*h].is_empty()) {
        return Err(err(format!(
            "`{}` is used in a way that needs bindless (index it directly with an immediate or a constant)",
            v.name.as_deref().unwrap_or("heap")
        )));
    }
    Ok(Lowered { module, info, slots })
}

enum IndexExpr {
    Dynamic(Handle<Expression>),
    Constant(u32),
}

/// The value of `e` when it is an integer literal.
fn constant(f: &naga::Function, e: Handle<Expression>) -> Option<u32> {
    match f.expressions[e] {
        Expression::Literal(naga::Literal::U32(v)) => Some(v),
        Expression::Literal(naga::Literal::I32(v)) => u32::try_from(v).ok(),
        Expression::Literal(naga::Literal::AbstractInt(v)) => u32::try_from(v).ok(),
        _ => None,
    }
}

/// The byte offset in the immediates of the `u32` member `e` loads, if that is what it is:
/// `draw.tex` (a load through a member access of the immediates), or a member of a loaded
/// immediates struct.
fn immediate_offset(f: &naga::Function, module: &naga::Module, immediates: Option<Handle<naga::GlobalVariable>>, e: Handle<Expression>) -> Option<u32> {
    let imm = immediates?;
    // the chain of member indices from the immediates global, innermost last
    let mut members = Vec::new();
    let mut cur = match f.expressions[e] {
        Expression::Load { pointer } => pointer,
        _ => e,
    };
    loop {
        match f.expressions[cur] {
            Expression::AccessIndex { base, index } => {
                members.push(index);
                cur = base;
            }
            Expression::Load { pointer } => cur = pointer,
            Expression::GlobalVariable(g) if g == imm => break,
            _ => return None,
        }
    }
    members.reverse();
    let mut ty = module.global_variables[imm].ty;
    let mut offset = 0;
    for i in members {
        match &module.types[ty].inner {
            naga::TypeInner::Struct { members, .. } => {
                let m = members.get(i as usize)?;
                offset += m.offset;
                ty = m.ty;
            }
            naga::TypeInner::Array { base, stride, .. } => {
                offset += i * stride;
                ty = *base;
            }
            naga::TypeInner::Vector { scalar, .. } => {
                offset += i * scalar.width as u32;
                ty = module.types.iter().find(|(_, t)| t.inner == naga::TypeInner::Scalar(*scalar)).map(|(h, _)| h)?;
            }
            _ => return None,
        }
    }
    matches!(
        module.types[ty].inner,
        naga::TypeInner::Scalar(naga::Scalar {
            kind: naga::ScalarKind::Uint | naga::ScalarKind::Sint,
            width: 4
        })
    )
    .then_some(offset)
}

/// Take `replaced` (now global variable expressions, which are never emitted) out of the
/// `Emit` ranges of `block` and its nested blocks.
fn unemit(block: &mut naga::Block, replaced: &[Handle<Expression>], exprs: &naga::Arena<Expression>) {
    let old = std::mem::take(block);
    for (stmt, span) in old.span_into_iter() {
        let stmt = match stmt {
            naga::Statement::Emit(range) => {
                let r = range.index_range();
                let mut start = r.start;
                for i in r.clone() {
                    if replaced.iter().any(|h| h.index() as u32 == i) {
                        if start < i {
                            block.push(naga::Statement::Emit(naga::Range::from_index_range(start..i, exprs)), span);
                        }
                        start = i + 1;
                    }
                }
                if start < r.end {
                    block.push(naga::Statement::Emit(naga::Range::from_index_range(start..r.end, exprs)), span);
                }
                continue;
            }
            naga::Statement::Block(mut b) => {
                unemit(&mut b, replaced, exprs);
                naga::Statement::Block(b)
            }
            naga::Statement::If { condition, mut accept, mut reject } => {
                unemit(&mut accept, replaced, exprs);
                unemit(&mut reject, replaced, exprs);
                naga::Statement::If { condition, accept, reject }
            }
            naga::Statement::Switch { selector, mut cases } => {
                for c in &mut cases {
                    unemit(&mut c.body, replaced, exprs);
                }
                naga::Statement::Switch { selector, cases }
            }
            naga::Statement::Loop { mut body, mut continuing, break_if } => {
                unemit(&mut body, replaced, exprs);
                unemit(&mut continuing, replaced, exprs);
                naga::Statement::Loop { body, continuing, break_if }
            }
            other => other,
        };
        block.push(stmt, span);
    }
}

/// One stage of the bound tier before its source (GLSL, HLSL 5.0): its lowered module and
/// what it reads.
/// The buffer slots of `lowered` read as uniform blocks instead of storage buffers: the
/// read-only views of a fixed size (no runtime-sized array) - a frame's constants, a camera -
/// for a stage without storage buffers. The backend binds the view's range as a uniform buffer
/// (its offset aligned for it: GL's buffer view alignment covers both). All or none: when a
/// view's type breaks the uniform layout rules (an array of scalars, say), they all stay
/// storage and the stage fails where storage buffers are missing, as before.
pub fn views_as_uniforms(lowered: &mut Lowered) {
    let module = &mut lowered.module;
    let mut changed = Vec::new();
    for (k, slot) in lowered.slots.iter().enumerate() {
        if !heap::is_buffer(slot.heap_binding) {
            continue;
        }
        let Some((h, v)) = module
            .global_variables
            .iter()
            .find(|(_, v)| v.binding.is_some_and(|b| b.group == HEAP_GROUP && b.binding == slot.binding))
        else {
            continue;
        };
        let read_only = matches!(v.space, AddressSpace::Storage { access } if !access.contains(naga::StorageAccess::STORE));
        if read_only && !runtime_sized(&module.types, v.ty) {
            changed.push((k, h, v.space));
        }
    }
    if changed.is_empty() {
        return;
    }
    for &(_, h, _) in &changed {
        module.global_variables[h].space = AddressSpace::Uniform;
    }
    match crate::validator().validate(module) {
        Ok(info) => {
            lowered.info = info;
            for &(k, ..) in &changed {
                lowered.slots[k].uniform = true;
            }
        }
        Err(_) => {
            for &(_, h, space) in &changed {
                module.global_variables[h].space = space;
            }
        }
    }
}

/// Whether type `ty` ends in a runtime-sized array.
fn runtime_sized(types: &naga::UniqueArena<naga::Type>, ty: Handle<naga::Type>) -> bool {
    match &types[ty].inner {
        naga::TypeInner::Array { size: naga::ArraySize::Dynamic, .. } => true,
        naga::TypeInner::Struct { members, .. } => members.last().is_some_and(|m| runtime_sized(types, m.ty)),
        _ => false,
    }
}

pub struct Stage {
    pub stage: naga::ShaderStage,
    pub entry: String,
    pub lowered: Lowered,
    pub roots: Vec<crate::RootBinding>,
}

/// Lower the heap of one entry point, with `constants` applied to its `override`s.
pub fn stage(m: &Module, stage: naga::ShaderStage, entry: Option<&str>, constants: &[(&str, f64)]) -> Result<Stage, ShaderError> {
    stage_with(m, stage, entry, constants, false)
}

/// [`stage`]; `uniform_views`: the stage has no storage buffers (OpenGL 3.3 and ES 3.0, most
/// ES 3.1 fragment stages), so its read-only buffer views are read as uniform blocks where they
/// can be ([`views_as_uniforms`]).
pub fn stage_with(m: &Module, stage: naga::ShaderStage, entry: Option<&str>, constants: &[(&str, f64)], uniform_views: bool) -> Result<Stage, ShaderError> {
    let name = m.entry_point(stage, entry)?.name.clone();
    let (module, _) = m.specialize(stage, &name, constants)?;
    let mut lowered = lower(m, module, stage, &name)?;
    if uniform_views {
        views_as_uniforms(&mut lowered);
    }
    Ok(Stage {
        stage,
        roots: m.entry_point_roots(stage, &name)?,
        entry: name,
        lowered,
    })
}

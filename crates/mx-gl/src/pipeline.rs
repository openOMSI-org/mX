//! Shader modules and pipelines: GL programs, with what each draw binds.

use std::sync::Arc;

use glow::HasContext;
use mx_shader::bound::Index;
use mx_shader::glsl::{self, Class};
use mx_shader::naga::ShaderStage;
use mx_shader::naga::back::glsl::{GlslScalarKind, GlslUniformType};
use mx_types as t;

use crate::conv;
use crate::device::{Garbage, Shared};
use crate::{Device, Error, Result};

pub struct ShaderModuleDescriptor<'a> {
    pub label: Option<&'a str>,
    /// WGSL. The heap prelude (`mx_tex2d`, `mx_samplers`, ...) is added by mX.
    pub source: &'a str,
}

pub(crate) struct ModuleInner {
    pub module: mx_shader::Module,
}

#[derive(Clone)]
pub struct ShaderModule(pub(crate) Arc<ModuleInner>);

/// What one heap slot of a pipeline binds: a heap entry (by the index the immediates or the
/// shader give) at a GL binding.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotBinding {
    pub class: Class,
    pub gl: u8,
    pub index: Index,
}

/// A root binding: WGSL `(group, binding)` at a GL binding.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RootBinding {
    pub group: u32,
    pub binding: u32,
    pub class: Class,
    pub gl: u8,
}

/// One immediate uniform: its location, type and place in the immediates.
pub(crate) struct ImmediateUniform {
    pub location: glow::UniformLocation,
    pub ty: GlslUniformType,
    pub offset: u32,
}

/// What every pipeline binds at draw time.
pub(crate) struct Bindings {
    pub roots: Vec<RootBinding>,
    pub slots: Vec<SlotBinding>,
    /// (texture unit, sampler binding) pairs: the sampler GLSL combines with each texture.
    pub texture_samplers: Vec<(u8, Option<u8>)>,
    pub immediates: Vec<ImmediateUniform>,
    pub first_instance: Option<glow::UniformLocation>,
    /// A stage writes storage buffers or images (a memory barrier follows).
    pub writes: bool,
}

/// A colour target's blend state in GL terms.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    pub blend: Option<([u32; 2], [u32; 2], [u32; 2])>,
    pub mask: t::ColorWrites,
}

pub(crate) struct VertexBuffer {
    pub stride: i32,
    pub instance: bool,
    pub attributes: Vec<(u32, conv::AttribDesc, i32)>,
}

pub(crate) struct RenderPipelineInner {
    pub program: glow::Program,
    pub bindings: Bindings,
    pub topology: u32,
    pub vertex_buffers: Vec<VertexBuffer>,
    pub primitive: t::PrimitiveState,
    pub depth_stencil: Option<t::DepthStencilState>,
    pub multisample: t::MultisampleState,
    pub targets: Vec<Target>,
    pub label: String,
    shared: Arc<Shared>,
}

impl Drop for RenderPipelineInner {
    fn drop(&mut self) {
        self.shared.retire(Garbage::Program(self.program));
    }
}

#[derive(Clone)]
pub struct RenderPipeline(pub(crate) Arc<RenderPipelineInner>);

pub(crate) struct ComputePipelineInner {
    pub program: glow::Program,
    pub bindings: Bindings,
    pub label: String,
    shared: Arc<Shared>,
}

impl Drop for ComputePipelineInner {
    fn drop(&mut self) {
        self.shared.retire(Garbage::Program(self.program));
    }
}

#[derive(Clone)]
pub struct ComputePipeline(pub(crate) Arc<ComputePipelineInner>);

impl Device {
    pub fn create_shader_module(&self, desc: &ShaderModuleDescriptor<'_>) -> Result<ShaderModule> {
        let module = mx_shader::Module::from_wgsl(desc.label.unwrap_or("shader"), desc.source)?;
        Ok(ShaderModule(Arc::new(ModuleInner { module })))
    }

    /// Lower the heap of every entry point now (GLSL is compiled with the pipeline, which
    /// numbers the bindings of its stages together).
    /// Each entry point lowered to the bound tier, written as this context's GLSL and compiled
    /// by the driver, on its own (a pipeline's stages are linked when it is made). Before, only
    /// the lowering was tried: a module the driver's GLSL compiler refused passed.
    pub fn precompile(&self, module: &ShaderModule) -> Result<()> {
        let options = self.options();
        for ep in &module.0.module.module.entry_points {
            let kind = match ep.stage {
                ShaderStage::Vertex => glow::VERTEX_SHADER,
                ShaderStage::Fragment => glow::FRAGMENT_SHADER,
                ShaderStage::Compute if self.shared.caps.compute => glow::COMPUTE_SHADER,
                _ => continue,
            };
            let stage = glsl::stage(&module.0.module, ep.stage, Some(&ep.name), &[])?;
            let layout = glsl::Layout::for_pipeline(&[&stage]).map_err(|message| Error::Compile {
                label: module.0.module.label.clone(),
                message,
            })?;
            let out = glsl::write(&module.0.module, &stage, 0, &layout, &options)?;
            let gl = self.shared.ctx.lock();
            // SAFETY: a shader made, compiled and deleted here, on the current context.
            unsafe {
                let sh = gl.create_shader(kind).map_err(|e| crate::gl_err("shader", e))?;
                gl.shader_source(sh, &out.source);
                gl.compile_shader(sh);
                let ok = gl.get_shader_compile_status(sh);
                let log = if ok { String::new() } else { gl.get_shader_info_log(sh) };
                gl.delete_shader(sh);
                if !ok {
                    log::debug!("{}: GLSL that failed:\n{}", ep.name, out.source);
                    return Err(Error::Compile {
                        label: module.0.module.label.clone(),
                        message: format!("{}: {}", ep.name, if log.trim().is_empty() { "the driver gave no log" } else { &log }),
                    });
                }
            }
        }
        Ok(())
    }

    fn options(&self) -> glsl::Options {
        let c = &self.shared.caps;
        glsl::Options {
            version: c.glsl,
            texture_shadow_lod: c.texture_shadow_lod,
            draw_parameters: c.draw_parameters,
        }
    }

    /// Compile and link the stages; return the program and what draws bind.
    fn program(&self, label: &str, stages: &[(&ShaderModule, glsl::Stage)]) -> Result<(glow::Program, Bindings)> {
        let pipe_err = |message: String| Error::Pipeline { label: label.into(), message };
        let refs: Vec<&glsl::Stage> = stages.iter().map(|(_, s)| s).collect();
        let layout = glsl::Layout::for_pipeline(&refs).map_err(pipe_err)?;
        let options = self.options();
        let mut outputs = Vec::new();
        for (i, (m, s)) in stages.iter().enumerate() {
            outputs.push(glsl::write(&m.0.module, s, i, &layout, &options)?);
        }
        let gl = self.shared.ctx.lock();
        // SAFETY: plain shader compilation and program linking on the current context.
        unsafe {
            let program = gl.create_program().map_err(|e| crate::gl_err("program", e))?;
            let mut shaders = Vec::new();
            for ((m, s), out) in stages.iter().zip(&outputs) {
                let kind = match s.stage {
                    ShaderStage::Vertex => glow::VERTEX_SHADER,
                    ShaderStage::Fragment => glow::FRAGMENT_SHADER,
                    _ => glow::COMPUTE_SHADER,
                };
                let sh = gl.create_shader(kind).map_err(|e| crate::gl_err("shader", e))?;
                gl.shader_source(sh, &out.source);
                gl.compile_shader(sh);
                if !gl.get_shader_compile_status(sh) {
                    let log = gl.get_shader_info_log(sh);
                    gl.delete_shader(sh);
                    gl.delete_program(program);
                    log::debug!("{label}: GLSL that failed:\n{}", out.source);
                    return Err(Error::Compile {
                        label: m.0.module.label.clone(),
                        message: if log.trim().is_empty() { "the driver gave no log".into() } else { log },
                    });
                }
                gl.attach_shader(program, sh);
                shaders.push(sh);
            }
            gl.link_program(program);
            for sh in shaders {
                gl.detach_shader(program, sh);
                gl.delete_shader(sh);
            }
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(pipe_err(log));
            }
            // bindings the GLSL could not declare (GL 3.3, ES 3.0)
            gl.use_program(Some(program));
            if !self.shared.caps.explicit_bindings {
                for out in &outputs {
                    for (name, class, b) in &out.named_bindings {
                        match class {
                            Class::Texture | Class::Image | Class::Sampler => {
                                if let Some(loc) = gl.get_uniform_location(program, name) {
                                    gl.uniform_1_i32(Some(&loc), *b as i32);
                                }
                            }
                            Class::UniformBuffer => {
                                if let Some(i) = gl.get_uniform_block_index(program, name) {
                                    gl.uniform_block_binding(program, i, *b as u32);
                                }
                            }
                            Class::StorageBuffer => {
                                if let Some(i) = gl.get_shader_storage_block_index(program, name) {
                                    gl.shader_storage_block_binding(program, i, *b as u32);
                                }
                            }
                        }
                    }
                }
            }
            let immediates = outputs
                .iter()
                .flat_map(|o| o.immediates.iter())
                .filter_map(|item| {
                    gl.get_uniform_location(program, &item.access_path).map(|location| ImmediateUniform {
                        location,
                        ty: item.ty,
                        offset: item.offset,
                    })
                })
                .collect();
            let first_instance = if self.shared.caps.draw_parameters {
                None
            } else {
                gl.get_uniform_location(program, glsl::FIRST_INSTANCE_BINDING)
            };
            gl.use_program(None);

            let roots = layout
                .bindings
                .iter()
                .filter(|(rb, _)| rb.group != mx_shader::HEAP_GROUP)
                .map(|(rb, &(class, b))| RootBinding {
                    group: rb.group,
                    binding: rb.binding,
                    class,
                    gl: b,
                })
                .collect();
            let mut slots = Vec::new();
            for (i, (_, s)) in stages.iter().enumerate() {
                for slot in &s.lowered.slots {
                    if let Some((class, b)) = layout.slot(i, slot.binding) {
                        slots.push(SlotBinding { class, gl: b, index: slot.index });
                    }
                }
            }
            let mut texture_samplers: Vec<(u8, Option<u8>)> = outputs.iter().flat_map(|o| o.texture_samplers.iter().copied()).collect();
            texture_samplers.sort_unstable();
            texture_samplers.dedup();
            let writes = stages.iter().any(|(m, s)| {
                s.roots
                    .iter()
                    .any(|r| matches!(r.kind, mx_shader::RootKind::StorageBuffer { read_only: false } | mx_shader::RootKind::StorageTexture))
                    || s.lowered
                        .slots
                        .iter()
                        .any(|slot| m.0.module.heap_buffers.iter().any(|v| v.binding == slot.heap_binding && !v.read_only))
            });
            Ok((
                program,
                Bindings {
                    roots,
                    slots,
                    texture_samplers,
                    immediates,
                    first_instance,
                    writes,
                },
            ))
        }
    }

    pub fn create_render_pipeline(&self, desc: &t::RenderPipelineDescriptor<'_, ShaderModule>) -> Result<RenderPipeline> {
        let label = desc.label.unwrap_or("render pipeline");
        let mut stages = vec![(
            desc.vertex.module,
            glsl::stage(&desc.vertex.module.0.module, ShaderStage::Vertex, desc.vertex.entry_point, desc.vertex.constants)?,
        )];
        if let Some(f) = &desc.fragment {
            stages.push((f.module, glsl::stage(&f.module.0.module, ShaderStage::Fragment, f.entry_point, f.constants)?));
        }
        let (program, bindings) = self.program(label, &stages)?;
        let vertex_buffers = desc
            .vertex
            .buffers
            .iter()
            .map(|b| VertexBuffer {
                stride: b.array_stride as i32,
                instance: b.step_mode == t::VertexStepMode::Instance,
                attributes: b.attributes.iter().map(|a| (a.shader_location, conv::vertex_format(a.format), a.offset as i32)).collect(),
            })
            .collect();
        let targets = desc
            .fragment
            .as_ref()
            .map_or(&[][..], |f| f.targets)
            .iter()
            .map(|ct| match ct {
                Some(ct) => Target {
                    blend: ct.blend.map(|b| {
                        (
                            [conv::blend_factor(b.color.src_factor), conv::blend_factor(b.color.dst_factor)],
                            [conv::blend_factor(b.alpha.src_factor), conv::blend_factor(b.alpha.dst_factor)],
                            [conv::blend_op(b.color.operation), conv::blend_op(b.alpha.operation)],
                        )
                    }),
                    mask: ct.write_mask,
                },
                None => Target {
                    blend: None,
                    mask: t::ColorWrites::empty(),
                },
            })
            .collect();
        Ok(RenderPipeline(Arc::new(RenderPipelineInner {
            program,
            bindings,
            topology: conv::topology(desc.primitive.topology),
            vertex_buffers,
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil,
            multisample: desc.multisample,
            targets,
            label: label.into(),
            shared: self.shared.clone(),
        })))
    }

    pub fn create_compute_pipeline(&self, desc: &t::ComputePipelineDescriptor<'_, ShaderModule>) -> Result<ComputePipeline> {
        let label = desc.label.unwrap_or("compute pipeline");
        if !self.shared.caps.compute {
            return Err(Error::Unsupported("compute shaders (OpenGL 4.3 / OpenGL ES 3.1)".into()));
        }
        let stage = glsl::stage(&desc.module.0.module, ShaderStage::Compute, desc.entry_point, desc.constants)?;
        let (program, bindings) = self.program(label, &[(desc.module, stage)])?;
        Ok(ComputePipeline(Arc::new(ComputePipelineInner {
            program,
            bindings,
            label: label.into(),
            shared: self.shared.clone(),
        })))
    }
}

/// Set one immediate uniform from the immediates' bytes.
///
/// # Safety
/// The program the location belongs to is in use.
pub(crate) unsafe fn set_immediate(gl: &glow::Context, u: &ImmediateUniform, bytes: &[u8]) {
    let o = u.offset as usize;
    let words = |n: usize| -> Vec<u32> {
        (0..n)
            .map(|i| bytes.get(o + i * 4..o + i * 4 + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect()
    };
    let floats = |n: usize| -> Vec<f32> { words(n).into_iter().map(f32::from_bits).collect() };
    let l = Some(&u.location);
    // SAFETY: the caller's.
    unsafe {
        match u.ty {
            GlslUniformType::Scalar(s) => match s.kind {
                GlslScalarKind::Float => gl.uniform_1_f32(l, floats(1)[0]),
                GlslScalarKind::Sint => gl.uniform_1_i32(l, words(1)[0] as i32),
                GlslScalarKind::Uint => gl.uniform_1_u32(l, words(1)[0]),
            },
            GlslUniformType::Vector { size, scalar } => {
                let n = size as usize;
                match scalar.kind {
                    GlslScalarKind::Float => {
                        let v = floats(n);
                        match n {
                            2 => gl.uniform_2_f32_slice(l, &v),
                            3 => gl.uniform_3_f32_slice(l, &v),
                            _ => gl.uniform_4_f32_slice(l, &v),
                        }
                    }
                    GlslScalarKind::Sint => {
                        let v: Vec<i32> = words(n).into_iter().map(|w| w as i32).collect();
                        match n {
                            2 => gl.uniform_2_i32_slice(l, &v),
                            3 => gl.uniform_3_i32_slice(l, &v),
                            _ => gl.uniform_4_i32_slice(l, &v),
                        }
                    }
                    GlslScalarKind::Uint => {
                        let v = words(n);
                        match n {
                            2 => gl.uniform_2_u32_slice(l, &v),
                            3 => gl.uniform_3_u32_slice(l, &v),
                            _ => gl.uniform_4_u32_slice(l, &v),
                        }
                    }
                }
            }
            GlslUniformType::Matrix { columns, rows, .. } => {
                // (columns of `rows` floats, each padded to the column alignment in memory)
                let (c, r) = (columns as usize, rows as usize);
                let stride = rows.alignment() as usize;
                let all = floats(c * stride);
                let v: Vec<f32> = (0..c).flat_map(|i| all[i * stride..i * stride + r].to_vec()).collect();
                match (c, r) {
                    (2, 2) => gl.uniform_matrix_2_f32_slice(l, false, &v),
                    (3, 3) => gl.uniform_matrix_3_f32_slice(l, false, &v),
                    (4, 4) => gl.uniform_matrix_4_f32_slice(l, false, &v),
                    (2, 3) => gl.uniform_matrix_2x3_f32_slice(l, false, &v),
                    (2, 4) => gl.uniform_matrix_2x4_f32_slice(l, false, &v),
                    (3, 2) => gl.uniform_matrix_3x2_f32_slice(l, false, &v),
                    (3, 4) => gl.uniform_matrix_3x4_f32_slice(l, false, &v),
                    (4, 2) => gl.uniform_matrix_4x2_f32_slice(l, false, &v),
                    _ => gl.uniform_matrix_4x3_f32_slice(l, false, &v),
                }
            }
        }
    }
}

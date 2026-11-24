//! Shader modules and pipelines: SPIR-V per entry point, pipeline layouts from the reflected
//! root bindings (set 0 the heap, set `g + 1` root group `g`), immediates as push constants.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::sync::Arc;

use ash::vk;
use mx_shader::naga::ShaderStage;
use mx_shader::{RootBinding, RootKind, spirv};
use mx_types as t;

use crate::device::{Garbage, Shared};
use crate::{Device, Error, Result, conv, vk_err};

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

/// The root bindings of one WGSL group, as its descriptor set layout has them.
pub(crate) struct Group {
    pub layout: vk::DescriptorSetLayout,
    pub bindings: Vec<(u32, RootKind, vk::DescriptorType)>,
}

pub(crate) struct PipelineInner {
    pub raw: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub bind_point: vk::PipelineBindPoint,
    /// Per WGSL root group (`None`: the pipeline reads nothing there).
    pub groups: Vec<Option<Group>>,
    pub immediate_size: u32,
    pub push_stages: vk::ShaderStageFlags,
    /// A stage may write memory (see `command::SyncState`).
    pub writes: bool,
    pub label: String,
    shared: Arc<Shared>,
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        self.shared.retire(Garbage::Pipeline(self.raw, self.layout));
    }
}

#[derive(Clone)]
pub struct RenderPipeline(pub(crate) Arc<PipelineInner>);

#[derive(Clone)]
pub struct ComputePipeline(pub(crate) Arc<PipelineInner>);

impl RenderPipeline {
    /// The `VkPipeline`, for interop.
    pub fn raw(&self) -> vk::Pipeline {
        self.0.raw
    }
}

impl ComputePipeline {
    /// The `VkPipeline`, for interop.
    pub fn raw(&self) -> vk::Pipeline {
        self.0.raw
    }
}

fn immediate_size(m: &mx_shader::Module) -> u32 {
    let module = &m.module;
    module
        .global_variables
        .iter()
        .filter(|(_, v)| v.space == mx_shader::naga::AddressSpace::Immediate)
        .map(|(_, v)| module.types[v.ty].inner.size(module.to_ctx()))
        .max()
        .unwrap_or(0)
}

fn descriptor_type(k: RootKind) -> vk::DescriptorType {
    match k {
        // (dynamic: a draw may read it at another offset, see `RenderPass::set_buffer_offset`)
        RootKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        RootKind::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
        RootKind::Texture => vk::DescriptorType::SAMPLED_IMAGE,
        RootKind::StorageTexture => vk::DescriptorType::STORAGE_IMAGE,
        RootKind::Sampler => vk::DescriptorType::SAMPLER,
        RootKind::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
    }
}

fn shader_stage(s: ShaderStage) -> vk::ShaderStageFlags {
    match s {
        ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
        ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        _ => vk::ShaderStageFlags::COMPUTE,
    }
}

/// One compiled entry point.
struct Stage {
    module: vk::ShaderModule,
    name: CString,
    stage: vk::ShaderStageFlags,
    roots: Vec<RootBinding>,
    /// The entry point may write memory: storage buffers or textures, or writable heap buffers.
    writes: bool,
}

impl Shared {
    /// The descriptor set layout of a root group with these bindings (cached).
    pub(crate) fn set_layout(&self, key: crate::device::SetLayoutKey) -> Result<vk::DescriptorSetLayout> {
        let mut cache = self.set_layouts.lock();
        if let Some(l) = cache.get(&key) {
            return Ok(*l);
        }
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = key
            .iter()
            .map(|&(b, ty, stages)| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(ty).descriptor_count(1).stage_flags(stages))
            .collect();
        // SAFETY: plain creation.
        let l = unsafe { self.raw.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| vk_err("descriptor set layout", e))?;
        cache.insert(key, l);
        Ok(l)
    }
}

impl Device {
    pub fn create_shader_module(&self, desc: &ShaderModuleDescriptor<'_>) -> Result<ShaderModule> {
        let module = mx_shader::Module::from_wgsl(desc.label.unwrap_or("shader"), desc.source)?;
        Ok(ShaderModule(Arc::new(ModuleInner { module })))
    }

    fn stage(&self, m: &ShaderModule, stage: ShaderStage, entry: Option<&str>, constants: &[(&str, f64)]) -> Result<(Stage, [u32; 3])> {
        let out = spirv::write(&m.0.module, stage, entry, constants, &self.shared.spirv)?;
        let storage_bits: u64 = (mx_shader::heap::FIRST_STORAGE_TEXTURE..=mx_shader::heap::LAST_STORAGE_TEXTURE).fold(0, |b, i| b | 1 << i);
        let uses_storage = out.heap_bindings_used & storage_bits != 0;
        let caps = &self.shared.caps;
        if uses_storage && !caps.heap_storage_images {
            return Err(Error::Unsupported(format!(
                "{}: storage textures in the bindless heap (descriptorBindingStorageImageUpdateAfterBind)",
                m.0.module.label
            )));
        }
        if out.heap_bindings_used & 1 << mx_shader::heap::ACCELERATION_STRUCTURES != 0 && !caps.heap_acceleration_structures {
            return Err(Error::Unsupported(format!(
                "{}: acceleration structures in the bindless heap (ray queries with descriptorBindingAccelerationStructureUpdateAfterBind)",
                m.0.module.label
            )));
        }
        let info = vk::ShaderModuleCreateInfo::default().code(&out.words);
        // SAFETY: valid SPIR-V words from naga.
        let module = unsafe { self.shared.raw.create_shader_module(&info, None) }.map_err(|e| Error::Compile {
            label: m.0.module.label.clone(),
            message: format!("vkCreateShaderModule: {e}"),
        })?;
        let name = CString::new(out.entry_point).expect("entry point names have no NUL");
        let writes = out.roots.iter().any(|r| matches!(r.kind, RootKind::StorageBuffer { read_only: false } | RootKind::StorageTexture))
            || m.0.module.heap_buffers.iter().any(|v| !v.read_only && out.heap_bindings_used & (1 << v.binding) != 0)
            || uses_storage;
        Ok((
            Stage {
                module,
                name,
                stage: shader_stage(stage),
                roots: out.roots,
                writes,
            },
            out.workgroup_size,
        ))
    }

    /// Compile every entry point of `module` now (with its overrides' defaults), so shader
    /// errors show up at load instead of at the first draw. (The driver compiles SPIR-V to
    /// machine code when a pipeline is made; this makes and checks the SPIR-V modules.)
    pub fn precompile(&self, module: &ShaderModule) -> Result<()> {
        for ep in &module.0.module.module.entry_points {
            if matches!(ep.stage, ShaderStage::Vertex | ShaderStage::Fragment | ShaderStage::Compute) {
                let (s, _) = self.stage(module, ep.stage, Some(&ep.name), &[])?;
                // SAFETY: not used by anything.
                unsafe { self.shared.raw.destroy_shader_module(s.module, None) };
            }
        }
        Ok(())
    }

    /// The pipeline layout of these stages: the heap, the root groups they read, the
    /// immediates.
    fn layout(&self, stages: &[Stage], immediate_size: u32) -> Result<(vk::PipelineLayout, Vec<Option<Group>>, vk::ShaderStageFlags)> {
        let shared = &self.shared;
        let all_stages = stages.iter().fold(vk::ShaderStageFlags::empty(), |a, s| a | s.stage);
        let mut roots: BTreeMap<(u32, u32), (RootKind, vk::ShaderStageFlags)> = BTreeMap::new();
        for s in stages {
            for r in &s.roots {
                roots.entry((r.group, r.binding)).or_insert((r.kind, vk::ShaderStageFlags::empty())).1 |= s.stage;
            }
        }
        let groups_used = roots.keys().map(|k| k.0 + 1).max().unwrap_or(0);
        let mut set_layouts = vec![shared.heap.lock().layout];
        let mut groups = Vec::new();
        for g in 0..groups_used {
            let bindings: Vec<(u32, RootKind, vk::DescriptorType)> = roots.iter().filter(|(k, _)| k.0 == g).map(|(k, v)| (k.1, v.0, descriptor_type(v.0))).collect();
            let key = roots.iter().filter(|(k, _)| k.0 == g).map(|(k, v)| (k.1, descriptor_type(v.0), v.1)).collect();
            let layout = shared.set_layout(key)?;
            set_layouts.push(layout);
            groups.push((!bindings.is_empty()).then_some(Group { layout, bindings }));
        }
        let push = [vk::PushConstantRange {
            stage_flags: all_stages,
            offset: 0,
            size: immediate_size.div_ceil(4) * 4,
        }];
        let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if immediate_size > 0 {
            info = info.push_constant_ranges(&push);
        }
        // SAFETY: plain creation from live set layouts.
        let layout = unsafe { shared.raw.create_pipeline_layout(&info, None) }.map_err(|e| vk_err("pipeline layout", e))?;
        Ok((layout, groups, all_stages))
    }

    pub fn create_render_pipeline(&self, desc: &t::RenderPipelineDescriptor<'_, ShaderModule>) -> Result<RenderPipeline> {
        let label = desc.label.unwrap_or("render pipeline");
        let shared = &self.shared;
        let mut stages = Vec::new();
        let r = (|| {
            stages.push(self.stage(desc.vertex.module, ShaderStage::Vertex, desc.vertex.entry_point, desc.vertex.constants)?.0);
            if let Some(fs) = &desc.fragment {
                stages.push(self.stage(fs.module, ShaderStage::Fragment, fs.entry_point, fs.constants)?.0);
            }
            let immediate_size = immediate_size(&desc.vertex.module.0.module).max(desc.fragment.as_ref().map_or(0, |f| immediate_size(&f.module.0.module)));
            let (layout, groups, push_stages) = self.layout(&stages, immediate_size)?;
            let writes = stages.iter().any(|s| s.writes);
            let raw = match self.graphics_pipeline(desc, &stages, layout) {
                Ok(p) => p,
                Err(e) => {
                    // SAFETY: just made, unused.
                    unsafe { shared.raw.destroy_pipeline_layout(layout, None) };
                    return Err(e);
                }
            };
            if let Some(l) = desc.label {
                shared.name(raw, l);
            }
            Ok(RenderPipeline(Arc::new(PipelineInner {
                raw,
                layout,
                bind_point: vk::PipelineBindPoint::GRAPHICS,
                groups,
                immediate_size,
                push_stages,
                writes,
                label: label.into(),
                shared: shared.clone(),
            })))
        })();
        for s in stages {
            // SAFETY: the pipeline (if made) no longer needs the modules.
            unsafe { shared.raw.destroy_shader_module(s.module, None) };
        }
        r
    }

    fn graphics_pipeline(&self, desc: &t::RenderPipelineDescriptor<'_, ShaderModule>, stages: &[Stage], layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
        let label = desc.label.unwrap_or("render pipeline");
        let shared = &self.shared;
        let stage_infos: Vec<vk::PipelineShaderStageCreateInfo> = stages
            .iter()
            .map(|s| vk::PipelineShaderStageCreateInfo::default().stage(s.stage).module(s.module).name(&s.name))
            .collect();

        let mut vb = Vec::new();
        let mut va = Vec::new();
        for (i, b) in desc.vertex.buffers.iter().enumerate() {
            vb.push(vk::VertexInputBindingDescription {
                binding: i as u32,
                stride: b.array_stride as u32,
                input_rate: match b.step_mode {
                    t::VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    t::VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                },
            });
            for a in b.attributes {
                va.push(vk::VertexInputAttributeDescription {
                    location: a.shader_location,
                    binding: i as u32,
                    format: conv::vertex_format(a.format),
                    offset: a.offset as u32,
                });
            }
        }
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&vb).vertex_attribute_descriptions(&va);
        let p = &desc.primitive;
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(conv::topology(p.topology))
            .primitive_restart_enable(p.strip_index_format.is_some());
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(conv::cull(p.cull_mode))
            .front_face(conv::front_face(p.front_face))
            .depth_clamp_enable(p.unclipped_depth)
            .line_width(1.0);
        let sample_mask = [desc.multisample.mask as u32, (desc.multisample.mask >> 32) as u32];
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(conv::samples(desc.multisample.count))
            .sample_mask(&sample_mask)
            .alpha_to_coverage_enable(desc.multisample.alpha_to_coverage_enabled);

        let mut depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let mut rendering = vk::PipelineRenderingCreateInfo::default();
        let (mut depth_format, mut stencil_format) = (vk::Format::UNDEFINED, vk::Format::UNDEFINED);
        if let Some(ds) = &desc.depth_stencil {
            let f = conv::format(ds.format, shared.caps.d24s8);
            if ds.format.has_depth() {
                depth_format = f;
            }
            if ds.format.has_stencil() {
                stencil_format = f;
            }
            // (as wgpu-hal: only what is enabled)
            if ds.depth_compare != t::CompareFunction::Always || ds.depth_write_enabled {
                depth_stencil = depth_stencil
                    .depth_test_enable(true)
                    .depth_write_enable(ds.depth_write_enabled)
                    .depth_compare_op(conv::compare(ds.depth_compare));
            }
            let s = &ds.stencil;
            if (s.front != t::StencilFaceState::IGNORE || s.back != t::StencilFaceState::IGNORE) && (s.read_mask != 0 || s.write_mask != 0) {
                let face = |f: &t::StencilFaceState| vk::StencilOpState {
                    fail_op: conv::stencil_op(f.fail_op),
                    pass_op: conv::stencil_op(f.pass_op),
                    depth_fail_op: conv::stencil_op(f.depth_fail_op),
                    compare_op: conv::compare(f.compare),
                    compare_mask: s.read_mask,
                    write_mask: s.write_mask,
                    reference: 0,
                };
                depth_stencil = depth_stencil.stencil_test_enable(true).front(face(&s.front)).back(face(&s.back));
            }
            let b = &ds.bias;
            if b.constant != 0 || b.slope_scale != 0.0 {
                raster = raster
                    .depth_bias_enable(true)
                    .depth_bias_constant_factor(b.constant as f32)
                    .depth_bias_clamp(b.clamp)
                    .depth_bias_slope_factor(b.slope_scale);
            }
        }
        let mut color_formats = Vec::new();
        let mut blends = Vec::new();
        if let Some(fs) = &desc.fragment {
            for target in fs.targets {
                match target {
                    Some(t) => {
                        color_formats.push(conv::format(t.format, shared.caps.d24s8));
                        let mut a = vk::PipelineColorBlendAttachmentState::default().color_write_mask(conv::write_mask(t.write_mask));
                        if let Some(b) = t.blend {
                            a = a
                                .blend_enable(true)
                                .src_color_blend_factor(conv::blend_factor(b.color.src_factor))
                                .dst_color_blend_factor(conv::blend_factor(b.color.dst_factor))
                                .color_blend_op(conv::blend_op(b.color.operation))
                                .src_alpha_blend_factor(conv::blend_factor(b.alpha.src_factor))
                                .dst_alpha_blend_factor(conv::blend_factor(b.alpha.dst_factor))
                                .alpha_blend_op(conv::blend_op(b.alpha.operation));
                        }
                        blends.push(a);
                    }
                    None => {
                        color_formats.push(vk::Format::UNDEFINED);
                        blends.push(vk::PipelineColorBlendAttachmentState::default());
                    }
                }
            }
        }
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
        rendering = rendering
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(stencil_format);
        let dynamic = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);

        let info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stage_infos)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering)];
        // SAFETY: every state struct above outlives the call; the stages and layout are live.
        unsafe { shared.raw.create_graphics_pipelines(vk::PipelineCache::null(), &info, None) }
            .map(|p| p[0])
            .map_err(|(_, e)| Error::Pipeline {
                label: label.into(),
                message: format!("vkCreateGraphicsPipelines: {e}"),
            })
    }

    pub fn create_compute_pipeline(&self, desc: &t::ComputePipelineDescriptor<'_, ShaderModule>) -> Result<ComputePipeline> {
        let label = desc.label.unwrap_or("compute pipeline");
        let shared = &self.shared;
        let (stage, _) = self.stage(desc.module, ShaderStage::Compute, desc.entry_point, desc.constants)?;
        let module = stage.module;
        let r = (|| {
            let immediate_size = immediate_size(&desc.module.0.module);
            let writes = stage.writes;
            let stages = [stage];
            let (layout, groups, push_stages) = self.layout(&stages, immediate_size)?;
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&stages[0].name))
                .layout(layout)];
            // SAFETY: as for render pipelines.
            let raw = match unsafe { shared.raw.create_compute_pipelines(vk::PipelineCache::null(), &info, None) } {
                Ok(p) => p[0],
                Err((_, e)) => {
                    // SAFETY: just made, unused.
                    unsafe { shared.raw.destroy_pipeline_layout(layout, None) };
                    return Err(Error::Pipeline {
                        label: label.into(),
                        message: format!("vkCreateComputePipelines: {e}"),
                    });
                }
            };
            if let Some(l) = desc.label {
                shared.name(raw, l);
            }
            Ok(ComputePipeline(Arc::new(PipelineInner {
                raw,
                layout,
                bind_point: vk::PipelineBindPoint::COMPUTE,
                groups,
                immediate_size,
                push_stages,
                writes,
                label: label.into(),
                shared: shared.clone(),
            })))
        })();
        // SAFETY: the pipeline (if made) no longer needs the module.
        unsafe { shared.raw.destroy_shader_module(module, None) };
        r
    }
}

//! Shader modules and pipelines: HLSL per entry point compiled by DXC (or FXC), root
//! signatures from the root bindings the pipeline's entry points use.
//!
//! Root parameters, in order:
//!
//! | index | parameter |
//! |-------|-----------|
//! | 0 | naga's special constants (`b1, space15`): first vertex, first instance / workgroup count |
//! | 1 | the heap: one SRV range per typed texture array (`t0, space8..=13`, all from descriptor 0) and the heap group's sampler index buffer (`t7, space7`, descriptor `T`) |
//! | 2 | the sampler heap: `s0` and `s2048` of `space14`, both from sampler 0 |
//! | 3 | immediates (`b0, space15`), if the pipeline has any |
//! | then | one descriptor table per root group the pipeline uses, in group order |

use std::collections::BTreeMap;
use std::sync::Arc;

use mx_shader::hlsl;
use mx_shader::naga::ShaderStage;
use mx_shader::{RootBinding, RootKind, heap};
use mx_types as t;
use windows::Win32::Graphics::Direct3D as d3d;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::{Device, Error, Result, conv, hr_err};

pub(crate) const SPECIAL_PARAM: u32 = 0;
pub(crate) const HEAP_PARAM: u32 = 1;
pub(crate) const SAMPLER_PARAM: u32 = 2;

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

/// One root group's descriptor table: the bindings in table order, and whether its sampler
/// index buffer follows them.
#[derive(Clone, Debug)]
pub(crate) struct GroupTable {
    pub param: u32,
    pub bindings: Vec<(u32, RootKind)>,
    pub samplers: bool,
}

impl GroupTable {
    pub fn len(&self) -> u32 {
        self.bindings.len() as u32 + self.samplers as u32
    }
}

/// What a pipeline binds, besides the heap.
pub(crate) struct Layout {
    pub signature: ID3D12RootSignature,
    pub groups: [Option<GroupTable>; 7],
    pub immediates: Option<(u32, u32)>,
    /// The pipeline reads buffers through the heap: a command list keeps the heap's buffers
    /// in the states shaders reach them in before its draws or dispatches.
    pub heap_buffers: bool,
    /// It may write them (a UAV barrier follows).
    pub heap_buffer_writes: bool,
}

pub(crate) struct RenderPipelineInner {
    pub pso: ID3D12PipelineState,
    pub layout: Layout,
    pub topology: d3d::D3D_PRIMITIVE_TOPOLOGY,
    pub strides: [u32; 8],
    pub draw: ID3D12CommandSignature,
    pub draw_indexed: ID3D12CommandSignature,
    pub label: String,
}

#[derive(Clone)]
pub struct RenderPipeline(pub(crate) Arc<RenderPipelineInner>);

pub(crate) struct ComputePipelineInner {
    pub pso: ID3D12PipelineState,
    pub layout: Layout,
    pub dispatch: ID3D12CommandSignature,
    pub label: String,
}

#[derive(Clone)]
pub struct ComputePipeline(pub(crate) Arc<ComputePipelineInner>);

impl RenderPipeline {
    /// The `ID3D12PipelineState`, for interop.
    pub fn raw(&self) -> &ID3D12PipelineState {
        &self.0.pso
    }
}

impl ComputePipeline {
    /// The `ID3D12PipelineState`, for interop.
    pub fn raw(&self) -> &ID3D12PipelineState {
        &self.0.pso
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

fn uses_f16(m: &mx_shader::Module) -> bool {
    use mx_shader::naga::{Scalar, TypeInner};
    m.module.types.iter().any(|(_, ty)| match ty.inner {
        TypeInner::Scalar(s) | TypeInner::Vector { scalar: s, .. } | TypeInner::Matrix { scalar: s, .. } => s == Scalar::F16,
        _ => false,
    })
}

/// One compiled entry point.
struct Stage {
    code: Arc<Vec<u8>>,
    roots: Vec<RootBinding>,
    /// The buffer views of the heap it uses: (binding, read-only).
    heap_buffers: Vec<(u32, bool)>,
}

fn bytecode(code: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: code.as_ptr().cast(),
        BytecodeLength: code.len(),
    }
}

impl Device {
    pub fn create_shader_module(&self, desc: &ShaderModuleDescriptor<'_>) -> Result<ShaderModule> {
        let module = mx_shader::Module::from_wgsl(desc.label.unwrap_or("shader"), desc.source)?;
        Ok(ShaderModule(Arc::new(ModuleInner { module })))
    }

    /// HLSL for one entry point, compiled (or found in the cache, as wgpu-hal keeps one).
    fn stage(&self, m: &ShaderModule, stage: ShaderStage, entry: Option<&str>, constants: &[(&str, f64)], fragment: Option<(&ShaderModule, &str)>) -> Result<Stage> {
        let sh = &self.shared;
        let label = &m.0.module.label;
        let heap = hlsl::HeapSizes {
            textures: sh.views.textures,
            buffers: sh.views.buffers,
        };
        let out = hlsl::write(&m.0.module, stage, entry, constants, sh.shader_model, heap, fragment.map(|(fm, n)| (&fm.0.module, n)))?;
        let heap_buffers =
            m.0.module
                .heap_buffers
                .iter()
                .filter(|v| out.heap_bindings_used & (1 << v.binding) != 0)
                .map(|v| (v.binding, v.read_only))
                .collect();
        log::trace!("{label} ({stage:?}) HLSL:\n{}", out.source);
        let key = (out.source, out.entry_point, out.profile);
        let cached = sh.shader_cache.lock().get(&key).cloned();
        let code = match cached {
            Some(c) => c,
            None => {
                let code = sh
                    .compiler
                    .compile_cached(sh.disk_cache.as_ref(), &key.0, label, &key.1, &key.2, uses_f16(&m.0.module))
                    .map_err(|message| Error::Compile { label: label.clone(), message })?;
                let code = Arc::new(code);
                let mut cache = sh.shader_cache.lock();
                if cache.len() > 256 {
                    cache.clear();
                }
                cache.insert(key, code.clone());
                code
            }
        };
        Ok(Stage { code, roots: out.roots, heap_buffers })
    }

    /// Compile every entry point of `module` now (with its overrides' defaults), so shader
    /// errors and the compiler's time show up at load instead of at the first draw. (Vertex
    /// entry points are compiled without a fragment stage to pair with; a pipeline that
    /// drops vertex outputs its fragment stage does not read compiles them again.)
    pub fn precompile(&self, module: &ShaderModule) -> Result<()> {
        for ep in &module.0.module.module.entry_points {
            if matches!(ep.stage, ShaderStage::Vertex | ShaderStage::Fragment | ShaderStage::Compute) {
                self.stage(module, ep.stage, Some(&ep.name), &[], None)?;
            }
        }
        Ok(())
    }

    pub fn create_render_pipeline(&self, desc: &t::RenderPipelineDescriptor<'_, ShaderModule>) -> Result<RenderPipeline> {
        let label = desc.label.unwrap_or("render pipeline");
        let pipe_err = |message: String| Error::Pipeline { label: label.into(), message };
        let fs = match &desc.fragment {
            Some(f) => {
                let name = f.module.0.module.entry_point(ShaderStage::Fragment, f.entry_point)?.name.clone();
                Some((self.stage(f.module, ShaderStage::Fragment, Some(&name), f.constants, None)?, name))
            }
            None => None,
        };
        let vs = self.stage(
            desc.vertex.module,
            ShaderStage::Vertex,
            desc.vertex.entry_point,
            desc.vertex.constants,
            desc.fragment.as_ref().zip(fs.as_ref()).map(|(f, (_, n))| (f.module, n.as_str())),
        )?;
        let mut roots = vs.roots.clone();
        let mut heap_buffers = vs.heap_buffers.clone();
        if let Some((f, _)) = &fs {
            roots.extend(f.roots.iter().copied());
            heap_buffers.extend(f.heap_buffers.iter().copied());
        }
        let imm = immediate_size(&desc.vertex.module.0.module).max(desc.fragment.as_ref().map_or(0, |f| immediate_size(&f.module.0.module)));
        let layout = self.layout(&roots, &heap_buffers, imm, true).map_err(pipe_err)?;

        if desc.vertex.buffers.len() > 8 {
            return Err(pipe_err(format!("{} vertex buffers; mX takes 8", desc.vertex.buffers.len())));
        }
        let mut strides = [0u32; 8];
        let mut elements = Vec::new();
        for (i, b) in desc.vertex.buffers.iter().enumerate() {
            strides[i] = b.array_stride as u32;
            let (class, rate) = match b.step_mode {
                t::VertexStepMode::Vertex => (D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0),
                t::VertexStepMode::Instance => (D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1),
            };
            for a in b.attributes {
                elements.push(D3D12_INPUT_ELEMENT_DESC {
                    // (naga's semantic for vertex inputs, as wgpu-hal)
                    SemanticName: windows::core::s!("LOC"),
                    SemanticIndex: a.shader_location,
                    Format: conv::vertex_format(a.format),
                    InputSlot: i as u32,
                    AlignedByteOffset: a.offset as u32,
                    InputSlotClass: class,
                    InstanceDataStepRate: rate,
                });
            }
        }
        let (topology_type, topology) = conv::topology(desc.primitive.topology);
        let bias = desc.depth_stencil.map(|d| d.bias).unwrap_or_default();
        let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
        let targets = desc.fragment.as_ref().map_or(&[][..], |f| f.targets);
        if targets.len() > 8 {
            return Err(pipe_err(format!("{} colour targets; Direct3D 12 takes 8", targets.len())));
        }
        for (f, ct) in rtv_formats.iter_mut().zip(targets) {
            if let Some(ct) = ct {
                *f = conv::format(ct.format);
            }
        }
        let d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: the signature outlives the creation call.
            pRootSignature: unsafe { crate::borrow(&layout.signature) },
            VS: bytecode(&vs.code),
            PS: fs.as_ref().map_or_else(D3D12_SHADER_BYTECODE::default, |(f, _)| bytecode(&f.code)),
            BlendState: D3D12_BLEND_DESC {
                AlphaToCoverageEnable: desc.multisample.alpha_to_coverage_enabled.into(),
                IndependentBlendEnable: true.into(),
                RenderTarget: conv::render_targets(targets),
            },
            SampleMask: desc.multisample.mask as u32,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: match desc.primitive.cull_mode {
                    None => D3D12_CULL_MODE_NONE,
                    Some(t::Face::Front) => D3D12_CULL_MODE_FRONT,
                    Some(t::Face::Back) => D3D12_CULL_MODE_BACK,
                },
                FrontCounterClockwise: (desc.primitive.front_face == t::FrontFace::Ccw).into(),
                DepthBias: bias.constant,
                DepthBiasClamp: bias.clamp,
                SlopeScaledDepthBias: bias.slope_scale,
                DepthClipEnable: (!desc.primitive.unclipped_depth).into(),
                MultisampleEnable: (desc.multisample.count > 1).into(),
                AntialiasedLineEnable: false.into(),
                ForcedSampleCount: 0,
                ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
            },
            DepthStencilState: desc.depth_stencil.as_ref().map(conv::depth_stencil).unwrap_or_default(),
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: if elements.is_empty() { std::ptr::null() } else { elements.as_ptr() },
                NumElements: elements.len() as u32,
            },
            IBStripCutValue: match desc.primitive.strip_index_format {
                Some(t::IndexFormat::Uint16) => D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF,
                Some(t::IndexFormat::Uint32) => D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF,
                None => D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            },
            PrimitiveTopologyType: topology_type,
            NumRenderTargets: targets.len() as u32,
            RTVFormats: rtv_formats,
            DSVFormat: desc.depth_stencil.as_ref().map_or(DXGI_FORMAT_UNKNOWN, |d| conv::format(d.format)),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: desc.multisample.count,
                Quality: 0,
            },
            ..Default::default()
        };
        // SAFETY: every pointer in the description outlives the call.
        let pso: ID3D12PipelineState = unsafe { self.shared.device.CreateGraphicsPipelineState(&d) }.map_err(|e| pipe_err(e.to_string()))?;
        crate::set_name(&pso, label);
        let sig = |stride: usize, ty| self.command_signature(&layout.signature, stride, ty);
        Ok(RenderPipeline(Arc::new(RenderPipelineInner {
            draw: sig(std::mem::size_of::<t::DrawIndirectArgs>(), D3D12_INDIRECT_ARGUMENT_TYPE_DRAW)?,
            draw_indexed: sig(std::mem::size_of::<t::DrawIndexedIndirectArgs>(), D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED)?,
            pso,
            layout,
            topology,
            strides,
            label: label.into(),
        })))
    }

    pub fn create_compute_pipeline(&self, desc: &t::ComputePipelineDescriptor<'_, ShaderModule>) -> Result<ComputePipeline> {
        let label = desc.label.unwrap_or("compute pipeline");
        let pipe_err = |message: String| Error::Pipeline { label: label.into(), message };
        let cs = self.stage(desc.module, ShaderStage::Compute, desc.entry_point, desc.constants, None)?;
        let layout = self.layout(&cs.roots, &cs.heap_buffers, immediate_size(&desc.module.0.module), false).map_err(pipe_err)?;
        let d = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: the signature outlives the creation call.
            pRootSignature: unsafe { crate::borrow(&layout.signature) },
            CS: bytecode(&cs.code),
            ..Default::default()
        };
        // SAFETY: every pointer in the description outlives the call.
        let pso: ID3D12PipelineState = unsafe { self.shared.device.CreateComputePipelineState(&d) }.map_err(|e| pipe_err(e.to_string()))?;
        crate::set_name(&pso, label);
        Ok(ComputePipeline(Arc::new(ComputePipelineInner {
            dispatch: self.command_signature(&layout.signature, std::mem::size_of::<t::DispatchIndirectArgs>(), D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH)?,
            pso,
            layout,
            label: label.into(),
        })))
    }

    /// The root signature for these root bindings (from every stage; the same binding twice is
    /// fine, the same binding as two kinds is not) and heap buffer views ((binding, read-only),
    /// from every stage).
    fn layout(&self, roots: &[RootBinding], heap_buffers: &[(u32, bool)], immediate_size: u32, graphics: bool) -> std::result::Result<Layout, String> {
        let mut by_group: BTreeMap<u32, BTreeMap<u32, RootKind>> = BTreeMap::new();
        for r in roots {
            let g = by_group.entry(r.group).or_default();
            match g.insert(r.binding, r.kind) {
                Some(k) if k != r.kind => {
                    return Err(format!("@group({}) @binding({}) is a {:?} in one stage and a {:?} in another", r.group, r.binding, k, r.kind));
                }
                _ => {}
            }
        }
        let uniforms = by_group.values().flat_map(|g| g.values()).filter(|k| **k == RootKind::UniformBuffer).count();
        if uniforms > hlsl::MAX_UNIFORM_BUFFERS as usize {
            return Err(format!("{uniforms} uniform buffers; Direct3D 12 takes {} per pipeline in mX", hlsl::MAX_UNIFORM_BUFFERS));
        }
        let t = self.shared.views.textures;
        let range = |ty, n, reg, space, offset| D3D12_DESCRIPTOR_RANGE {
            RangeType: ty,
            NumDescriptors: n,
            BaseShaderRegister: reg,
            RegisterSpace: space,
            OffsetInDescriptorsFromTableStart: offset,
        };
        // (all range arrays are made before the parameters point into them)
        // (every buffer view a stage declares is a range over the same descriptors: the raw
        // SRVs for read-only views, the raw UAVs for written ones)
        let mut views: Vec<(u32, bool)> = heap_buffers.to_vec();
        views.sort_unstable();
        views.dedup();
        let v = &self.shared.views;
        let heap_ranges: Vec<D3D12_DESCRIPTOR_RANGE> = (0..heap::SAMPLERS)
            .map(|b| range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, t, 0, hlsl::heap_space(b), 0))
            .chain(std::iter::once(range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, mx_shader::HEAP_GROUP, hlsl::SAMPLER_INDEX_SPACE, t)))
            .chain(views.iter().map(|&(b, read_only)| {
                if read_only {
                    range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, v.buffers, 0, hlsl::heap_space(b), v.buffer_srv(0))
                } else {
                    range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, v.buffers, 0, hlsl::heap_space(b), v.buffer_uav(0))
                }
            }))
            .collect();
        let n = hlsl::SAMPLER_HEAP_SIZE;
        let sampler_ranges = [
            range(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, n, 0, hlsl::SAMPLER_HEAP_SPACE, 0),
            range(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, n, hlsl::COMPARISON_SAMPLER_REGISTER, hlsl::SAMPLER_HEAP_SPACE, 0),
        ];
        let mut groups: [Option<GroupTable>; 7] = Default::default();
        let mut group_ranges: Vec<(u32, Vec<D3D12_DESCRIPTOR_RANGE>)> = Vec::new();
        let first_group_param = if immediate_size > 0 { 4 } else { 3 };
        for (&g, bindings) in &by_group {
            let mut rs = Vec::new();
            let mut table = GroupTable {
                param: first_group_param + group_ranges.len() as u32,
                bindings: Vec::new(),
                samplers: false,
            };
            for (&b, &k) in bindings {
                let ty = match k {
                    RootKind::Sampler => {
                        table.samplers = true;
                        continue;
                    }
                    RootKind::UniformBuffer => D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    RootKind::StorageBuffer { read_only: true } | RootKind::Texture | RootKind::AccelerationStructure => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    RootKind::StorageBuffer { read_only: false } | RootKind::StorageTexture => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                };
                rs.push(range(ty, 1, b, g, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND));
                table.bindings.push((b, k));
            }
            if table.samplers {
                rs.push(range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, g, hlsl::SAMPLER_INDEX_SPACE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND));
            }
            if rs.is_empty() {
                continue;
            }
            group_ranges.push((g, rs));
            groups[g as usize] = Some(table);
        }
        let table = |r: &[D3D12_DESCRIPTOR_RANGE]| D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: r.len() as u32,
                    pDescriptorRanges: r.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let constants = |reg, n| D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: reg,
                    RegisterSpace: hlsl::IMMEDIATES_SPACE,
                    Num32BitValues: n,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let mut params = vec![constants(hlsl::SPECIAL_CONSTANTS_REGISTER, 3), table(&heap_ranges), table(&sampler_ranges)];
        let immediates = (immediate_size > 0).then(|| {
            let dwords = immediate_size.div_ceil(4);
            params.push(constants(hlsl::IMMEDIATES_REGISTER, dwords));
            (3, dwords)
        });
        for (_, rs) in &group_ranges {
            params.push(table(rs));
        }
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: if graphics {
                D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            } else {
                D3D12_ROOT_SIGNATURE_FLAG_NONE
            },
        };
        let mut blob = None;
        let mut errors = None;
        // SAFETY: the description and every array it points to live across the call.
        let r = unsafe { D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, Some(&mut errors)) };
        if let Err(e) = r {
            let detail = errors
                // SAFETY: the error blob's own buffer.
                .map(|b: d3d::ID3DBlob| unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())).into_owned() })
                .unwrap_or_default();
            return Err(format!("root signature: {e} {detail}"));
        }
        let blob = blob.ok_or("root signature: no blob")?;
        // SAFETY: the serialized root signature's bytes.
        let signature: ID3D12RootSignature = unsafe {
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            self.shared.device.CreateRootSignature(0, bytes)
        }
        .map_err(|e| format!("root signature: {e}"))?;
        Ok(Layout {
            signature,
            groups,
            immediates,
            heap_buffers: !views.is_empty(),
            heap_buffer_writes: views.iter().any(|&(_, read_only)| !read_only),
        })
    }

    /// An indirect command signature that first sets the special constants (first vertex and
    /// instance, or the workgroup count), then draws / dispatches, as wgpu-hal's with
    /// `INDIRECT_BUILTIN_UPDATE`: arguments are 3 `u32`s then the API's arguments.
    fn command_signature(&self, root: &ID3D12RootSignature, args: usize, ty: D3D12_INDIRECT_ARGUMENT_TYPE) -> Result<ID3D12CommandSignature> {
        let descs = [
            D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                    Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                        RootParameterIndex: SPECIAL_PARAM,
                        DestOffsetIn32BitValues: 0,
                        Num32BitValuesToSet: 3,
                    },
                },
            },
            D3D12_INDIRECT_ARGUMENT_DESC { Type: ty, ..Default::default() },
        ];
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: (12 + args) as u32,
            NumArgumentDescs: descs.len() as u32,
            pArgumentDescs: descs.as_ptr(),
            NodeMask: 0,
        };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: the description outlives the call.
        unsafe { self.shared.device.CreateCommandSignature(&desc, root, &mut sig) }.map_err(|e| hr_err("command signature", e))?;
        sig.ok_or_else(|| Error::Device("no command signature".into()))
    }
}

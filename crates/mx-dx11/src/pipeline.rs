//! Shader modules and pipelines: WGSL, lowered to slots, to HLSL shader model 5.0, compiled
//! with FXC (`D3DCompile`), and the fixed-function states.

use std::sync::Arc;

use mx_shader::hlsl11::{self, Class};
use mx_shader::naga::ShaderStage;
use mx_types as t;
use windows::Win32::Graphics::Direct3D::Fxc::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::core::{PCSTR, s};

use crate::{Device, Error, Result, conv, hr};

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

/// What one stage reads, and where it is bound.
#[derive(Clone, Debug, Default)]
pub(crate) struct StageBindings {
    pub roots: Vec<(mx_shader::naga::ResourceBinding, Class, u8)>,
    pub slots: Vec<(mx_shader::bound::Slot, Class, u8)>,
    pub uses_immediates: bool,
}

pub(crate) struct RenderPipelineInner {
    pub vs: ID3D11VertexShader,
    pub ps: Option<ID3D11PixelShader>,
    pub layout: Option<ID3D11InputLayout>,
    pub rasterizer: ID3D11RasterizerState,
    pub blend: ID3D11BlendState,
    pub depth_stencil: ID3D11DepthStencilState,
    pub topology: D3D_PRIMITIVE_TOPOLOGY,
    pub sample_mask: u32,
    /// The stride of each vertex buffer slot.
    pub strides: Vec<u32>,
    pub vertex: StageBindings,
    pub fragment: StageBindings,
    /// The first UAV register of the fragment stage (its render targets come first).
    pub uav_base: u32,
}

// SAFETY: Direct3D 11 shaders and states are free-threaded.
unsafe impl Send for RenderPipelineInner {}
unsafe impl Sync for RenderPipelineInner {}

#[derive(Clone)]
pub struct RenderPipeline(pub(crate) Arc<RenderPipelineInner>);

pub(crate) struct ComputePipelineInner {
    pub cs: ID3D11ComputeShader,
    pub stage: StageBindings,
}

// SAFETY: as above.
unsafe impl Send for ComputePipelineInner {}
unsafe impl Sync for ComputePipelineInner {}

#[derive(Clone)]
pub struct ComputePipeline(pub(crate) Arc<ComputePipelineInner>);

/// Compile HLSL with FXC, or read what an earlier run compiled from the disk cache
/// (`mx_shader::disk_cache`: FXC took minutes over openOMSI's scene pipelines on WARP).
fn compile(label: &str, source: &str, entry: &str, profile: &str) -> Result<Vec<u8>> {
    static DISK: std::sync::OnceLock<Option<mx_shader::disk_cache::DiskCache>> = std::sync::OnceLock::new();
    let disk = DISK.get_or_init(mx_shader::disk_cache::DiskCache::from_env).as_ref();
    let key = mx_shader::disk_cache::Key {
        compiler: "d3dcompiler_47",
        source,
        entry,
        profile,
        flags: "O3 strict",
    };
    if let Some(code) = disk.and_then(|d| d.get(&key)) {
        return Ok(code);
    }
    let code = bytes(&compile_fxc(label, source, entry, profile)?).to_vec();
    if let Some(d) = disk {
        d.put(&key, &code);
    }
    Ok(code)
}

fn compile_fxc(label: &str, source: &str, entry: &str, profile: &str) -> Result<ID3DBlob> {
    let entry = std::ffi::CString::new(entry).expect("an entry point name");
    let profile = std::ffi::CString::new(profile).expect("a profile");
    let mut code = None;
    let mut errors = None;
    // (as wgpu-hal: optimised, IEEE strictness off, row-major matrices are naga's business)
    let flags = D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_ENABLE_STRICTNESS;
    // SAFETY: the source, entry point and profile live over the call.
    let r = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR(c"mx".as_ptr().cast()),
            None,
            None,
            PCSTR(entry.as_ptr().cast()),
            PCSTR(profile.as_ptr().cast()),
            flags,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    let message = || {
        errors
            .as_ref()
            // SAFETY: the blob holds the error text.
            .map(|b| unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())).into_owned() })
            .unwrap_or_default()
    };
    if r.is_err() {
        return Err(Error::Compile {
            label: label.into(),
            message: format!("{}\n{}", message(), r.err().map(|e| e.to_string()).unwrap_or_default()),
        });
    }
    code.ok_or_else(|| Error::Compile {
        label: label.into(),
        message: message(),
    })
}

fn bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own buffer, alive as long as the blob.
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) }
}

fn bindings(out: &hlsl11::Output) -> StageBindings {
    StageBindings {
        roots: out.roots.clone(),
        slots: out.slots.clone(),
        uses_immediates: out.uses_immediates,
    }
}

impl Device {
    pub fn create_shader_module(&self, desc: &ShaderModuleDescriptor<'_>) -> Result<ShaderModule> {
        let module = mx_shader::Module::from_wgsl(desc.label.unwrap_or("shader"), desc.source)?;
        Ok(ShaderModule(Arc::new(ModuleInner { module })))
    }

    /// Lower and write every entry point of `module` now (with its overrides' defaults), so
    /// shader errors show up at load instead of at the first draw.
    pub fn precompile(&self, module: &ShaderModule) -> Result<()> {
        for ep in &module.0.module.module.entry_points {
            if matches!(ep.stage, ShaderStage::Vertex | ShaderStage::Fragment | ShaderStage::Compute) {
                let s = mx_shader::bound::stage(&module.0.module, ep.stage, Some(&ep.name), &[])?;
                hlsl11::write(&module.0.module, &s, 0, None)?;
            }
        }
        Ok(())
    }

    pub fn create_render_pipeline(&self, desc: &t::RenderPipelineDescriptor<'_, ShaderModule>) -> Result<RenderPipeline> {
        let label = desc.label.unwrap_or("render pipeline");
        let device = &self.shared.device;
        let pipe_err = |message: String| Error::Pipeline { label: label.into(), message };
        let vm = &desc.vertex.module.0.module;
        let vstage = mx_shader::bound::stage(vm, ShaderStage::Vertex, desc.vertex.entry_point, desc.vertex.constants)?;
        let (frag_pair, fstage) = match &desc.fragment {
            Some(f) => {
                let s = mx_shader::bound::stage(&f.module.0.module, ShaderStage::Fragment, f.entry_point, f.constants)?;
                (Some((&f.module.0.module, s.entry.clone())), Some((f, s)))
            }
            None => (None, None),
        };
        let vout = hlsl11::write(vm, &vstage, 0, frag_pair.as_ref().map(|(m, n)| (*m, n.as_str())))?;
        let vblob = compile(label, &vout.source, &vout.entry_point, &vout.profile)?;
        let mut vs = None;
        // SAFETY: bytecode FXC made.
        unsafe { device.CreateVertexShader(&vblob, None, Some(&mut vs)) }.map_err(|e| hr("vertex shader", e))?;
        let vs = vs.ok_or_else(|| pipe_err("no vertex shader".into()))?;
        let targets = desc.fragment.as_ref().map_or(&[][..], |f| f.targets);
        let uav_base = targets.len() as u32;
        let (ps, fragment) = match &fstage {
            Some((f, s)) => {
                let out = hlsl11::write(&f.module.0.module, s, uav_base, None)?;
                let blob = compile(label, &out.source, &out.entry_point, &out.profile)?;
                let mut ps = None;
                // SAFETY: bytecode FXC made.
                unsafe { device.CreatePixelShader(&blob, None, Some(&mut ps)) }.map_err(|e| hr("pixel shader", e))?;
                (ps, bindings(&out))
            }
            None => (None, StageBindings::default()),
        };

        // the vertex buffers: one input element per attribute, "LOC<location>" as naga names them
        let mut elements = Vec::new();
        let mut strides = Vec::new();
        for (slot, vb) in desc.vertex.buffers.iter().enumerate() {
            strides.push(vb.array_stride as u32);
            let instance = vb.step_mode == t::VertexStepMode::Instance;
            for a in vb.attributes {
                elements.push(D3D11_INPUT_ELEMENT_DESC {
                    SemanticName: s!("LOC"),
                    SemanticIndex: a.shader_location,
                    Format: conv::vertex_format(a.format),
                    InputSlot: slot as u32,
                    AlignedByteOffset: a.offset as u32,
                    InputSlotClass: if instance { D3D11_INPUT_PER_INSTANCE_DATA } else { D3D11_INPUT_PER_VERTEX_DATA },
                    InstanceDataStepRate: instance as u32,
                });
            }
        }
        let layout = if elements.is_empty() {
            None
        } else {
            let mut l = None;
            // SAFETY: the elements and the vertex shader's bytecode.
            unsafe { device.CreateInputLayout(&elements, &vblob, Some(&mut l)) }.map_err(|e| pipe_err(format!("the vertex buffers do not match the vertex shader: {e}")))?;
            l
        };

        let p = &desc.primitive;
        let bias = desc.depth_stencil.map(|d| d.bias).unwrap_or_default();
        let rd = D3D11_RASTERIZER_DESC {
            FillMode: D3D11_FILL_SOLID,
            CullMode: match p.cull_mode {
                None => D3D11_CULL_NONE,
                Some(t::Face::Front) => D3D11_CULL_FRONT,
                Some(t::Face::Back) => D3D11_CULL_BACK,
            },
            FrontCounterClockwise: (p.front_face == t::FrontFace::Ccw).into(),
            DepthBias: bias.constant,
            DepthBiasClamp: bias.clamp,
            SlopeScaledDepthBias: bias.slope_scale,
            DepthClipEnable: (!p.unclipped_depth).into(),
            // (as WebGPU: every draw is clipped to the scissor, the whole target by default)
            ScissorEnable: true.into(),
            MultisampleEnable: (desc.multisample.count > 1).into(),
            AntialiasedLineEnable: false.into(),
        };
        let mut rasterizer = None;
        // SAFETY: plain creations of states.
        unsafe { device.CreateRasterizerState(&rd, Some(&mut rasterizer)) }.map_err(|e| hr("rasterizer state", e))?;

        let rts = conv::render_targets(targets);
        let mut bd = D3D11_BLEND_DESC {
            AlphaToCoverageEnable: desc.multisample.alpha_to_coverage_enabled.into(),
            IndependentBlendEnable: true.into(),
            RenderTarget: rts,
        };
        if targets.is_empty() {
            bd.RenderTarget[0].RenderTargetWriteMask = 0;
        }
        let mut blend = None;
        unsafe { device.CreateBlendState(&bd, Some(&mut blend)) }.map_err(|e| hr("blend state", e))?;

        let dsd = match &desc.depth_stencil {
            Some(ds) => conv::depth_stencil(ds),
            None => D3D11_DEPTH_STENCIL_DESC {
                DepthEnable: false.into(),
                DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D11_COMPARISON_ALWAYS,
                StencilEnable: false.into(),
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: D3D11_DEPTH_STENCILOP_DESC {
                    StencilFailOp: D3D11_STENCIL_OP_KEEP,
                    StencilDepthFailOp: D3D11_STENCIL_OP_KEEP,
                    StencilPassOp: D3D11_STENCIL_OP_KEEP,
                    StencilFunc: D3D11_COMPARISON_ALWAYS,
                },
                BackFace: D3D11_DEPTH_STENCILOP_DESC {
                    StencilFailOp: D3D11_STENCIL_OP_KEEP,
                    StencilDepthFailOp: D3D11_STENCIL_OP_KEEP,
                    StencilPassOp: D3D11_STENCIL_OP_KEEP,
                    StencilFunc: D3D11_COMPARISON_ALWAYS,
                },
            },
        };
        let mut depth_stencil = None;
        unsafe { device.CreateDepthStencilState(&dsd, Some(&mut depth_stencil)) }.map_err(|e| hr("depth-stencil state", e))?;

        Ok(RenderPipeline(Arc::new(RenderPipelineInner {
            vs,
            ps,
            layout,
            rasterizer: rasterizer.ok_or_else(|| pipe_err("no rasterizer state".into()))?,
            blend: blend.ok_or_else(|| pipe_err("no blend state".into()))?,
            depth_stencil: depth_stencil.ok_or_else(|| pipe_err("no depth-stencil state".into()))?,
            topology: conv::topology(p.topology),
            sample_mask: desc.multisample.mask as u32,
            strides,
            vertex: bindings(&vout),
            fragment,
            uav_base,
        })))
    }

    pub fn create_compute_pipeline(&self, desc: &t::ComputePipelineDescriptor<'_, ShaderModule>) -> Result<ComputePipeline> {
        let label = desc.label.unwrap_or("compute pipeline");
        let m = &desc.module.0.module;
        let s = mx_shader::bound::stage(m, ShaderStage::Compute, desc.entry_point, desc.constants)?;
        let out = hlsl11::write(m, &s, 0, None)?;
        let blob = compile(label, &out.source, &out.entry_point, &out.profile)?;
        let mut cs = None;
        // SAFETY: bytecode FXC made.
        unsafe { self.shared.device.CreateComputeShader(&blob, None, Some(&mut cs)) }.map_err(|e| hr("compute shader", e))?;
        Ok(ComputePipeline(Arc::new(ComputePipelineInner {
            cs: cs.ok_or_else(|| Error::Pipeline {
                label: label.into(),
                message: "no compute shader".into(),
            })?,
            stage: bindings(&out),
        })))
    }
}

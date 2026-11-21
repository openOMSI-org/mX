//! Command encoders and passes.

use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use mx_shader::{MAX_ROOT_GROUP, RootKind};
use mx_types as t;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use crate::conv::SHADER_READ;
use crate::device::{Recording, Shared};
use crate::pipeline::{HEAP_PARAM, Layout, SAMPLER_PARAM, SPECIAL_PARAM};
use crate::resource::{Bound, Footprint, copy_buffer_texture, physical, texture_location};
use crate::track::Track;
use crate::{BindGroup, Buffer, ComputePipeline, Device, QuerySet, RenderPipeline, Texture, TextureView, conv};

const GROUPS: usize = MAX_ROOT_GROUP as usize + 1;
const MAX_IMMEDIATES: usize = 256;

/// Recorded commands, ready for [`crate::Queue::submit`].
pub struct CommandBuffer {
    pub(crate) rec: Recording,
}

pub struct CommandEncoder {
    pub(crate) rec: Recording,
}

impl Device {
    pub fn create_command_encoder(&self, label: Option<&str>) -> CommandEncoder {
        let rec = Recording::new(&self.shared);
        if let Some(l) = label {
            crate::set_name(rec.list(), l);
        }
        CommandEncoder { rec }
    }
}

fn begin_event(list: &ID3D12GraphicsCommandList, label: &str) {
    let w = crate::wide(label);
    // SAFETY: a NUL-terminated UTF-16 string (PIX's "unicode" event, metadata 0, as wgpu).
    unsafe { list.BeginEvent(0, Some(w.as_ptr().cast()), (w.len() * 2) as u32) };
}

fn set_marker(list: &ID3D12GraphicsCommandList, label: &str) {
    let w = crate::wide(label);
    // SAFETY: as in `begin_event`.
    unsafe { list.SetMarker(0, Some(w.as_ptr().cast()), (w.len() * 2) as u32) };
}

impl CommandEncoder {
    /// The `ID3D12GraphicsCommandList` the encoder records into, for interop (DLSS / FSR
    /// evaluate here). The descriptor heaps set on it are mX's; resources mX knows are in
    /// their home states (see the crate docs) between mX's commands.
    pub fn raw(&self) -> &ID3D12GraphicsCommandList {
        self.rec.list()
    }

    /// Zero the mips and layers of `range` (colour to 0, depth to 0.0, stencil to 0): a
    /// render pass that clears where the texture can be a render target, else a copy from
    /// a new buffer (committed resources start zeroed).
    pub fn clear_texture(&mut self, texture: &Texture, range: &t::ImageSubresourceRange) {
        let dim = texture.dimension();
        let is_3d = dim == t::TextureDimension::D3;
        let (mips, layers) = range.resolve(texture.mip_level_count(), t::Unwritten::layer_count(dim, texture.size()));
        if mips.is_empty() || layers.is_empty() {
            return;
        }
        let format = texture.format();
        let ds = format.has_depth() || format.has_stencil();
        let depth = format.has_depth() && range.aspect != t::TextureAspect::StencilOnly;
        let stencil = format.has_stencil() && range.aspect != t::TextureAspect::DepthOnly;
        if texture.usage().contains(t::TextureUsages::RENDER_ATTACHMENT) {
            for level in mips {
                for layer in if is_3d { 0..1 } else { layers.clone() } {
                    let view = texture
                        .create_view(&t::TextureViewDescriptor {
                            label: Some("mX clear"),
                            dimension: Some(if is_3d {
                                t::TextureViewDimension::D3
                            } else if texture.size().depth_or_array_layers > 1 {
                                t::TextureViewDimension::D2Array
                            } else {
                                t::TextureViewDimension::D2
                            }),
                            base_mip_level: level,
                            mip_level_count: Some(1),
                            base_array_layer: layer,
                            array_layer_count: Some(1),
                            ..Default::default()
                        })
                        .expect("a view of one subresource to clear");
                    let color = [Some(t::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: t::Operations {
                            load: t::LoadOp::Clear(t::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: t::StoreOp::Store,
                        },
                    })];
                    let depth_stencil = ds.then(|| t::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: format.has_depth().then_some(if depth {
                            t::Operations {
                                load: t::LoadOp::Clear(0.0),
                                store: t::StoreOp::Store,
                            }
                        } else {
                            t::Operations {
                                load: t::LoadOp::Load,
                                store: t::StoreOp::Store,
                            }
                        }),
                        stencil_ops: format.has_stencil().then_some(if stencil {
                            t::Operations {
                                load: t::LoadOp::Clear(0),
                                store: t::StoreOp::Store,
                            }
                        } else {
                            t::Operations {
                                load: t::LoadOp::Load,
                                store: t::StoreOp::Store,
                            }
                        }),
                    });
                    drop(self.begin_render_pass(&t::RenderPassDescriptor {
                        label: Some("mX clear"),
                        color_attachments: if ds { &[] } else { &color },
                        depth_stencil_attachment: depth_stencil,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    }));
                }
            }
            return;
        }
        let (bw, bh) = format.block_dimensions();
        let block = format.block_copy_size().unwrap_or(4);
        let mut aspects = Vec::new();
        if format.has_depth() && format.has_stencil() {
            if depth {
                aspects.push(t::TextureAspect::DepthOnly);
            }
            if stencil {
                aspects.push(t::TextureAspect::StencilOnly);
            }
        } else {
            aspects.push(t::TextureAspect::All);
        }
        for level in mips {
            let mip = texture.size().mip_level_size(level, dim);
            let size = t::Extent3d {
                width: mip.width.div_ceil(bw) * bw,
                height: mip.height.div_ceil(bh) * bh,
                depth_or_array_layers: if is_3d { mip.depth_or_array_layers } else { layers.len() as u32 },
            };
            let bytes_per_row = (size.width / bw * block).next_multiple_of(t::COPY_BYTES_PER_ROW_ALIGNMENT);
            let rows = size.height / bh;
            let len = bytes_per_row as u64 * rows as u64 * size.depth_or_array_layers as u64;
            let zeros = crate::resource::create_buffer(
                &self.rec.shared,
                &t::BufferDescriptor {
                    label: Some("mX zeros"),
                    size: len,
                    usage: t::BufferUsages::COPY_SRC,
                },
            )
            .expect("memory for a buffer of zeros");
            for &aspect in &aspects {
                self.copy_buffer_to_texture(
                    t::TexelCopyBufferInfo {
                        buffer: &zeros,
                        layout: t::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(bytes_per_row),
                            rows_per_image: Some(rows),
                        },
                    },
                    t::TexelCopyTextureInfo {
                        texture,
                        mip_level: level,
                        origin: t::Origin3d {
                            x: 0,
                            y: 0,
                            z: if is_3d { 0 } else { layers.start },
                        },
                        aspect,
                    },
                    size,
                );
            }
            self.rec.keep(zeros);
        }
    }

    pub fn finish(mut self) -> CommandBuffer {
        self.rec.close();
        CommandBuffer { rec: self.rec }
    }

    /// Fill `size` bytes (to the end with `None`) from `offset` with zeros (copies from a
    /// zeroed buffer, as wgpu-hal).
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: Option<u64>) {
        let size = size.unwrap_or(buffer.size() - offset);
        assert!(offset + size <= buffer.size(), "clear_buffer past the end of the buffer");
        if size == 0 {
            return;
        }
        let rec = &mut self.rec;
        rec.track.need(&buffer.0.track, None, D3D12_RESOURCE_STATE_COPY_DEST);
        rec.barriers();
        let zero = rec.shared.zero_buffer.clone();
        let mut at = offset;
        while at < offset + size {
            let n = crate::device::ZERO_BUFFER_SIZE.min(offset + size - at);
            // SAFETY: both ranges inside their buffers.
            unsafe { rec.list().CopyBufferRegion(&buffer.0.raw, at, &zero, 0, n) };
            at += n;
        }
    }

    pub fn push_debug_group(&mut self, label: &str) {
        begin_event(self.rec.list(), label);
    }

    pub fn pop_debug_group(&mut self) {
        // SAFETY: closes the event pushed before.
        unsafe { self.rec.list().EndEvent() };
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        set_marker(self.rec.list(), label);
    }

    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64) {
        assert!(src_offset + size <= src.size() && dst_offset + size <= dst.size(), "copy_buffer_to_buffer out of range");
        let rec = &mut self.rec;
        rec.track.need(&src.0.track, None, D3D12_RESOURCE_STATE_COPY_SOURCE);
        rec.track.need(&dst.0.track, None, D3D12_RESOURCE_STATE_COPY_DEST);
        rec.barriers();
        // SAFETY: both ranges checked above.
        unsafe { rec.list().CopyBufferRegion(&dst.0.raw, dst_offset, &src.0.raw, src_offset, size) };
    }

    fn footprint(texture: &Texture, layout: t::TexelCopyBufferLayout, size: t::Extent3d) -> Footprint {
        let format = texture.format();
        let (_, bh) = format.block_dimensions();
        let phys = physical(format, size);
        let bytes_per_row = layout
            .bytes_per_row
            .unwrap_or_else(|| format.bytes_per_row(phys.width).expect("copies of depth-stencil textures need bytes_per_row"));
        Footprint {
            offset: layout.offset,
            bytes_per_row,
            rows_per_image: layout.rows_per_image.unwrap_or(phys.height) / bh,
        }
    }

    pub fn copy_buffer_to_texture(&mut self, src: t::TexelCopyBufferInfo<'_, Buffer>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d) {
        let fp = Self::footprint(dst.texture, src.layout, size);
        self.rec.track.need(&src.buffer.0.track, None, D3D12_RESOURCE_STATE_COPY_SOURCE);
        copy_buffer_texture(&mut self.rec, &src.buffer.0.raw, fp, dst.texture, &dst, size, true);
    }

    pub fn copy_texture_to_buffer(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyBufferInfo<'_, Buffer>, size: t::Extent3d) {
        let fp = Self::footprint(src.texture, dst.layout, size);
        self.rec.track.need(&dst.buffer.0.track, None, D3D12_RESOURCE_STATE_COPY_DEST);
        copy_buffer_texture(&mut self.rec, &dst.buffer.0.raw, fp, src.texture, &src, size, false);
    }

    pub fn copy_texture_to_texture(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d) {
        let size = physical(src.texture.format(), size);
        let is_3d = src.texture.dimension() == t::TextureDimension::D3;
        let (layers, depth) = if is_3d { (1, size.depth_or_array_layers) } else { (size.depth_or_array_layers, 1) };
        let range = |at: &t::TexelCopyTextureInfo<'_, Texture>| {
            let p = conv::plane(at.aspect);
            crate::track::Range {
                mips: at.mip_level..at.mip_level + 1,
                layers: if is_3d { 0..1 } else { at.origin.z..at.origin.z + layers },
                planes: p..p + 1,
            }
        };
        let rec = &mut self.rec;
        rec.track.need(&src.texture.0.track, Some(range(&src)), D3D12_RESOURCE_STATE_COPY_SOURCE);
        rec.track.need(&dst.texture.0.track, Some(range(&dst)), D3D12_RESOURCE_STATE_COPY_DEST);
        rec.barriers();
        for l in 0..layers {
            let (sl, dl) = if is_3d { (0, 0) } else { (src.origin.z + l, dst.origin.z + l) };
            let s = texture_location(src.texture, src.texture.0.track.subresource(src.mip_level, sl, conv::plane(src.aspect)));
            let d = texture_location(dst.texture, dst.texture.0.track.subresource(dst.mip_level, dl, conv::plane(dst.aspect)));
            let (sz, dz) = if is_3d { (src.origin.z, dst.origin.z) } else { (0, 0) };
            let b = D3D12_BOX {
                left: src.origin.x,
                top: src.origin.y,
                front: sz,
                right: src.origin.x + size.width,
                bottom: src.origin.y + size.height,
                back: sz + depth,
            };
            // SAFETY: regions inside both textures (the debug layer checks).
            unsafe { rec.list().CopyTextureRegion(&d, dst.origin.x, dst.origin.y, dz, &s, Some(&b)) };
        }
        rec.track.restore_textures();
    }

    /// Fill a texture's mip chain from level 0, each level a linear-filtered draw from the one
    /// above (a box filter for even sizes, as Metal's blit and Vulkan's linear blit). Needs a
    /// 2D (or array / cube) texture of a filterable, renderable format with `TEXTURE_BINDING`
    /// and `RENDER_ATTACHMENT` usage.
    pub fn generate_mipmaps(&mut self, texture: &Texture) {
        let u = texture.usage();
        assert!(
            u.contains(t::TextureUsages::TEXTURE_BINDING | t::TextureUsages::RENDER_ATTACHMENT),
            "generate_mipmaps on Direct3D 12 draws each level: the texture needs TEXTURE_BINDING and RENDER_ATTACHMENT"
        );
        assert_eq!(texture.dimension(), t::TextureDimension::D2, "generate_mipmaps: 2D textures only on Direct3D 12");
        let format = texture.format();
        let device = Device { shared: self.rec.shared.clone() };
        let (pipeline, sampler) = device.mip_tools(format);
        for layer in 0..texture.layers() {
            for level in 1..texture.mip_level_count() {
                let view = |mip| {
                    texture
                        .create_view(&t::TextureViewDescriptor {
                            label: Some("mX mip"),
                            dimension: Some(t::TextureViewDimension::D2),
                            base_mip_level: mip,
                            mip_level_count: Some(1),
                            base_array_layer: layer,
                            array_layer_count: Some(1),
                            ..Default::default()
                        })
                        .expect("out of descriptors for a mip view")
                };
                let (src, dst) = (view(level - 1), view(level));
                {
                    let mut rp = self.begin_render_pass(&t::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(t::RenderPassColorAttachment {
                            view: &dst,
                            resolve_target: None,
                            ops: t::Operations {
                                load: t::LoadOp::DontCare,
                                store: t::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rp.set_pipeline(&pipeline);
                    let imm = [src.index(), sampler];
                    rp.set_immediates(0, crate::device::bytes_of_u32(&imm));
                    rp.draw(0..3, 0..1);
                }
                self.rec.keep((src, dst));
            }
        }
    }

    pub fn begin_render_pass(&mut self, desc: &t::RenderPassDescriptor<'_, TextureView, QuerySet>) -> RenderPass<'_> {
        let rec = &mut self.rec;
        let mut rtvs = Vec::new();
        let mut extent = None;
        let mut resolves = Vec::new();
        let used = desc.color_attachments.iter().rposition(Option::is_some).map_or(0, |i| i + 1);
        for a in &desc.color_attachments[..used] {
            let Some(a) = a else {
                rtvs.push(rec.shared.null_rtv.handle());
                continue;
            };
            let v = &a.view.0;
            rec.track.need(&v.texture.0.track, Some(v.range()), D3D12_RESOURCE_STATE_RENDER_TARGET);
            rtvs.push(v.rtv.expect("a colour attachment needs RENDER_ATTACHMENT usage").handle());
            extent.get_or_insert(v.texture.size().mip_level_size(v.base_mip_level, v.texture.dimension()));
            if let Some(r) = a.resolve_target {
                resolves.push((a.view.clone(), r.clone()));
            }
            rec.keep(a.view.clone());
        }
        let mut dsv = None;
        if let Some(ds) = &desc.depth_stencil_attachment {
            let v = &ds.view.0;
            rec.track.need(&v.texture.0.track, Some(v.range()), D3D12_RESOURCE_STATE_DEPTH_WRITE);
            dsv = Some(v.dsv.expect("a depth attachment needs RENDER_ATTACHMENT usage").handle());
            extent.get_or_insert(v.texture.size().mip_level_size(v.base_mip_level, v.texture.dimension()));
            rec.keep(ds.view.clone());
        }
        rec.barriers();
        let list = rec.list().clone();
        // SAFETY: CPU descriptors of live views (read now, at record time).
        unsafe {
            list.OMSetRenderTargets(rtvs.len() as u32, if rtvs.is_empty() { None } else { Some(rtvs.as_ptr()) }, false, dsv.as_ref().map(|d| d as *const _));
            for (a, rtv) in desc.color_attachments.iter().zip(&rtvs) {
                if let Some(a) = a
                    && let t::LoadOp::Clear(c) = a.ops.load
                {
                    list.ClearRenderTargetView(*rtv, &[c.r as f32, c.g as f32, c.b as f32, c.a as f32], None);
                }
            }
            if let (Some(ds), Some(dsv)) = (&desc.depth_stencil_attachment, dsv) {
                let mut flags = D3D12_CLEAR_FLAGS(0);
                let (mut depth, mut stencil) = (1.0, 0);
                if let Some(t::Operations { load: t::LoadOp::Clear(d), .. }) = ds.depth_ops {
                    flags |= D3D12_CLEAR_FLAG_DEPTH;
                    depth = d;
                }
                if let Some(t::Operations { load: t::LoadOp::Clear(s), .. }) = ds.stencil_ops
                    && ds.view.texture().format().has_stencil()
                {
                    flags |= D3D12_CLEAR_FLAG_STENCIL;
                    stencil = s as u8;
                }
                if flags.0 != 0 {
                    list.ClearDepthStencilView(dsv, flags, depth, stencil, None);
                }
            }
            // WebGPU's pass defaults: the whole target, blend constant 0, stencil reference 0
            let e = extent.unwrap_or_default();
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: e.width as f32,
                Height: e.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: e.width as i32,
                bottom: e.height as i32,
            }]);
            list.OMSetBlendFactor(Some(&[0.0; 4]));
            list.OMSetStencilRef(0);
        }
        if let Some(l) = desc.label {
            begin_event(&list, l);
        }
        let mut end_timestamp = None;
        if let Some(ts) = &desc.timestamp_writes {
            if let Some(i) = ts.beginning_of_pass_write_index {
                ts.query_set.write(&list, i);
            }
            end_timestamp = ts.end_of_pass_write_index.map(|i| (ts.query_set.clone(), i));
            rec.keep(ts.query_set.clone());
        }
        if let Some(q) = desc.occlusion_query_set {
            rec.keep(q.clone());
        }
        RenderPass {
            enc: self,
            state: PassState::default(),
            pipeline: None,
            index: None,
            index_dirty: false,
            vertex: Default::default(),
            vertex_dirty: false,
            resolves,
            end_timestamp,
            occlusion: desc.occlusion_query_set.cloned(),
            occlusion_index: None,
            labelled: desc.label.is_some(),
        }
    }

    pub fn begin_compute_pass(&mut self, desc: &t::ComputePassDescriptor<'_, QuerySet>) -> ComputePass<'_> {
        let list = self.rec.list().clone();
        if let Some(l) = desc.label {
            begin_event(&list, l);
        }
        let mut end_timestamp = None;
        if let Some(ts) = &desc.timestamp_writes {
            if let Some(i) = ts.beginning_of_pass_write_index {
                ts.query_set.write(&list, i);
            }
            end_timestamp = ts.end_of_pass_write_index.map(|i| (ts.query_set.clone(), i));
            self.rec.keep(ts.query_set.clone());
        }
        ComputePass {
            enc: self,
            state: PassState::default(),
            pipeline: None,
            end_timestamp,
            labelled: desc.label.is_some(),
        }
    }
}

/// The mip generator of a device: one pipeline per format, one linear sampler.
#[derive(Default)]
pub(crate) struct MipTools {
    pipelines: HashMap<t::TextureFormat, RenderPipeline>,
    sampler: Option<u32>,
}

const MIP_SHADER: &str = "
struct M { src: u32, smp: u32 }
var<immediate> m: M;
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return Out(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(p.x, 1.0 - p.y));
}
@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return textureSampleLevel(mx_tex2d[m.src], mx_samplers[m.smp], i.uv, 0.0); }
";

impl Device {
    fn mip_tools(&self, format: t::TextureFormat) -> (RenderPipeline, u32) {
        let mut tools = self.shared.mips.lock();
        let sampler = *tools.sampler.get_or_insert_with(|| {
            let slot = self.shared.heap.lock().alloc_sampler().expect("the sampler heap is full");
            let d = t::SamplerDescriptor {
                mag_filter: t::FilterMode::Linear,
                min_filter: t::FilterMode::Linear,
                ..Default::default()
            };
            // SAFETY: a fresh slot of the sampler heap, never released.
            unsafe { self.shared.device.CreateSampler(&crate::resource::sampler_desc(&d), self.shared.views.sampler_cpu(slot)) };
            slot
        });
        if let Some(p) = tools.pipelines.get(&format) {
            return (p.clone(), sampler);
        }
        let m = self
            .create_shader_module(&crate::ShaderModuleDescriptor {
                label: Some("mX mipmaps"),
                source: MIP_SHADER,
            })
            .expect("the mip shader compiles");
        let p = self
            .create_render_pipeline(&t::RenderPipelineDescriptor {
                label: Some("mX mipmaps"),
                vertex: t::VertexState {
                    module: &m,
                    entry_point: None,
                    constants: &[],
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(t::FragmentState {
                    module: &m,
                    entry_point: None,
                    constants: &[],
                    targets: &[Some(format.into())],
                }),
            })
            .unwrap_or_else(|e| panic!("generate_mipmaps: {format:?} cannot be rendered to: {e}"));
        tools.pipelines.insert(format, p.clone());
        (p, sampler)
    }
}

/// Lays out GPU-driven draw arguments as the pipelines' command signatures read them: naga's
/// special constants (first vertex or base vertex, first instance, 0), then the API's
/// arguments (see `RenderPass::indirect`).
const INDIRECT_SHADER: &str = "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
struct P { max_draws: u32, indexed: u32 }
var<immediate> p: P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.max_draws) { return; }
    let n = select(4u, 5u, p.indexed != 0u);
    let s = i * n;
    let d = i * (n + 3u);
    dst[d] = src[s + n - 2u];
    dst[d + 1u] = src[s + n - 1u];
    dst[d + 2u] = 0u;
    for (var k = 0u; k < n; k++) { dst[d + 3u + k] = src[s + k]; }
}
";

impl Device {
    fn indirect_tool(&self) -> crate::ComputePipeline {
        let mut tool = self.shared.indirect.lock();
        tool.get_or_insert_with(|| {
            let m = self
                .create_shader_module(&crate::ShaderModuleDescriptor {
                    label: Some("mX indirect draws"),
                    source: INDIRECT_SHADER,
                })
                .expect("the indirect draw shader compiles");
            self.create_compute_pipeline(&t::ComputePipelineDescriptor {
                label: Some("mX indirect draws"),
                module: &m,
                entry_point: None,
                constants: &[],
            })
            .expect("the indirect draw pipeline is made")
        })
        .clone()
    }

    /// A batch of GPU-driven draws: a buffer its arguments are laid out into for
    /// `ExecuteIndirect`.
    pub fn create_indirect_draws(&self, desc: &t::IndirectDrawsDescriptor<'_>) -> crate::Result<crate::IndirectDraws> {
        let stride = if desc.indexed { 32 } else { 28 };
        let laid_out = self.create_buffer(&t::BufferDescriptor {
            label: desc.label,
            size: (desc.max_draws.max(1) as u64) * stride,
            usage: t::BufferUsages::STORAGE | t::BufferUsages::INDIRECT,
        })?;
        Ok(crate::IndirectDraws(std::sync::Arc::new(crate::resource::IndirectDrawsInner {
            max_draws: desc.max_draws,
            indexed: desc.indexed,
            laid_out,
            source: parking_lot::Mutex::new(None),
        })))
    }
}

impl CommandEncoder {
    /// Point a batch of GPU-driven draws at the arguments and the count the GPU wrote:
    /// `count` (a `u32` at `count_offset`) draws of `args` from `args_offset`, at most the
    /// batch's `max_draws`, drawn with `pipeline` and the index buffer `index` (indexed
    /// batches). Outside passes, after the arguments are written and before the pass that
    /// executes the batch: a dispatch lays them out for `ExecuteIndirect` here. `args` needs
    /// `STORAGE | INDIRECT`, `count` `INDIRECT`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indirect_draws(
        &mut self,
        draws: &crate::IndirectDraws,
        pipeline: &RenderPipeline,
        args: &Buffer,
        args_offset: u64,
        count: &Buffer,
        count_offset: u64,
        index: Option<(&Buffer, u64, t::IndexFormat)>,
    ) {
        let _ = pipeline;
        assert_eq!(draws.0.indexed, index.is_some(), "an indexed batch needs an index buffer, and only it");
        assert!(
            args.usage().contains(t::BufferUsages::STORAGE | t::BufferUsages::INDIRECT),
            "GPU-driven draw arguments need STORAGE | INDIRECT usage"
        );
        assert!(count.usage().contains(t::BufferUsages::INDIRECT), "a GPU-driven draw count needs INDIRECT usage");
        let device = Device { shared: self.rec.shared.clone() };
        let tool = device.indirect_tool();
        let n = if draws.0.indexed { 20 } else { 16 };
        let bg = device.create_bind_group(&crate::BindGroupDescriptor {
            label: None,
            entries: &[
                crate::BindGroupEntry {
                    binding: 0,
                    resource: crate::BindingResource::Buffer(crate::BufferBinding {
                        buffer: args,
                        offset: args_offset,
                        size: Some((draws.0.max_draws.max(1) as u64 * n).min(args.size() - args_offset)),
                    }),
                },
                crate::BindGroupEntry {
                    binding: 1,
                    resource: crate::BindingResource::Buffer(crate::BufferBinding {
                        buffer: &draws.0.laid_out,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        {
            let mut cp = self.begin_compute_pass(&Default::default());
            cp.set_pipeline(&tool);
            cp.set_bind_group(0, &bg);
            cp.set_immediates(0, bytemuck_u32(&[draws.0.max_draws, draws.0.indexed as u32]));
            cp.dispatch_workgroups(draws.0.max_draws.div_ceil(64).max(1), 1, 1);
        }
        *draws.0.source.lock() = Some(crate::resource::IndirectSource {
            count: count.clone(),
            count_offset,
            index: index.map(|(b, o, f)| (b.clone(), o, f)),
        });
    }
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and u8 has no alignment needs.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Bindings set on a pass and not yet handed to the command list.
pub(crate) struct PassState {
    pub(crate) groups: [Option<BindGroup>; GROUPS],
    /// Groups whose tables must be written again (bit per group).
    pub(crate) dirty: u8,
    pub(crate) immediates: Box<[u8; MAX_IMMEDIATES]>,
    pub(crate) immediates_dirty: bool,
    pub(crate) special: Option<[u32; 3]>,
    /// Offsets added to root uniform buffers' own (`set_buffer_offset`): (group, binding, bytes).
    pub(crate) offsets: Vec<(u32, u32, u64)>,
}

impl Default for PassState {
    fn default() -> Self {
        PassState {
            groups: Default::default(),
            dirty: !0,
            immediates: Box::new([0; MAX_IMMEDIATES]),
            immediates_dirty: true,
            special: None,
            offsets: Vec::new(),
        }
    }
}

/// Root-parameter setters of one pipeline kind.
pub(crate) trait Roots {
    fn table(&self, param: u32, gpu: D3D12_GPU_DESCRIPTOR_HANDLE);
    fn constants(&self, param: u32, data: &[u32]);
}

pub(crate) struct Graphics<'a>(pub(crate) &'a ID3D12GraphicsCommandList);
struct Compute<'a>(&'a ID3D12GraphicsCommandList);

// SAFETY (both impls): parameters of the root signature set on the list, data of the size
// the parameter declares.
impl Roots for Graphics<'_> {
    fn table(&self, param: u32, gpu: D3D12_GPU_DESCRIPTOR_HANDLE) {
        unsafe { self.0.SetGraphicsRootDescriptorTable(param, gpu) }
    }
    fn constants(&self, param: u32, data: &[u32]) {
        unsafe { self.0.SetGraphicsRoot32BitConstants(param, data.len() as u32, data.as_ptr().cast(), 0) }
    }
}

impl Roots for Compute<'_> {
    fn table(&self, param: u32, gpu: D3D12_GPU_DESCRIPTOR_HANDLE) {
        unsafe { self.0.SetComputeRootDescriptorTable(param, gpu) }
    }
    fn constants(&self, param: u32, data: &[u32]) {
        unsafe { self.0.SetComputeRoot32BitConstants(param, data.len() as u32, data.as_ptr().cast(), 0) }
    }
}

/// Where a draw's bindings are recorded: a command list ([`Recording`]) or a render bundle
/// (`crate::bundle`), which notes the states its resources need for the list it runs in.
pub(crate) trait Target {
    fn shared(&self) -> Arc<Shared>;
    /// `n` consecutive descriptors for a root table (see [`Recording::table`]).
    fn table(&mut self, n: u32) -> u32;
    fn need(&mut self, t: &Track, range: Option<crate::track::Range>, state: D3D12_RESOURCE_STATES);
    fn keep_boxed(&mut self, b: Box<dyn Any + Send + Sync>);
}

impl Target for Recording {
    fn shared(&self) -> Arc<Shared> {
        self.shared.clone()
    }
    fn table(&mut self, n: u32) -> u32 {
        Recording::table(self, n)
    }
    fn need(&mut self, t: &Track, range: Option<crate::track::Range>, state: D3D12_RESOURCE_STATES) {
        self.track.need(t, range, state);
    }
    fn keep_boxed(&mut self, b: Box<dyn Any + Send + Sync>) {
        self.keep.push(b);
    }
}

/// After a pipeline is set: the heap tables, and everything else to be set again.
pub(crate) fn bind_heap(shared: &Shared, r: &impl Roots, state: &mut PassState) {
    let v = &shared.views;
    r.table(HEAP_PARAM, v.view_gpu(0));
    r.table(SAMPLER_PARAM, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: v.sampler_gpu });
    state.dirty = !0;
    state.immediates_dirty = true;
    state.special = None;
}

/// Put the buffers with heap views in the states shaders reach them in (see
/// `Heap::buffer_states`), unless nothing changed since the last time in this list.
pub(crate) fn need_heap_buffers(rec: &mut Recording) {
    let shared = rec.shared.clone();
    let heap = shared.heap.lock();
    let settled = rec.track.buffers_settled();
    if settled && rec.heap_buffers_applied == heap.buffer_generation {
        return;
    }
    for (track, state) in heap.buffer_states() {
        rec.track.need(track, None, state);
    }
    rec.track.buffers_settled();
    rec.heap_buffers_applied = heap.buffer_generation;
}

/// Write the root tables the pipeline reads and move their resources to the states they are
/// read in. Returns whether a storage resource is bound (written).
pub(crate) fn bind_roots(rec: &mut impl Target, r: &impl Roots, layout: &Layout, state: &mut PassState, label: &str) -> bool {
    let mut writes = false;
    for (g, table) in layout.groups.iter().enumerate() {
        let Some(table) = table else { continue };
        if state.dirty & (1 << g) == 0 {
            continue;
        }
        let Some(group) = state.groups[g].clone() else {
            panic!("{label}: draw with nothing bound at @group({g})");
        };
        let first = rec.table(table.len());
        let shared = rec.shared();
        let (views, device) = (&shared.views, &shared.device);
        let mut at = first;
        for &(b, kind) in &table.bindings {
            let Some(bound) = group.get(b) else {
                panic!("{label}: draw with nothing bound at @group({g}) @binding({b})");
            };
            let mismatch = || panic!("{label}: the resource at @group({g}) @binding({b}) is not the kind the shader declares ({kind:?})");
            let extra = state.offsets.iter().find(|o| o.0 == g as u32 && o.1 == b).map(|o| o.2);
            let src = match (kind, bound) {
                (RootKind::UniformBuffer, Bound::Buffer { buffer, offset, size, .. }) if extra.is_some() => {
                    let extra = extra.unwrap_or(0);
                    rec.need(&buffer.0.track, None, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                    // (at `set_buffer_offset`'s offset: a view of its own, written into the ring)
                    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                        BufferLocation: buffer.0.va + offset + extra,
                        SizeInBytes: (*size).max(1).next_multiple_of(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT as u64) as u32,
                    };
                    // SAFETY: a view of a live buffer into this list's ring descriptors.
                    unsafe { device.CreateConstantBufferView(Some(&desc), views.view_cpu(at)) };
                    at += 1;
                    continue;
                }
                (RootKind::UniformBuffer, Bound::Buffer { buffer, cbv: Some(d), .. }) => {
                    rec.need(&buffer.0.track, None, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                    *d
                }
                (RootKind::StorageBuffer { read_only: true }, Bound::Buffer { buffer, srv: Some(d), .. }) => {
                    rec.need(&buffer.0.track, None, SHADER_READ);
                    *d
                }
                (RootKind::StorageBuffer { read_only: false }, Bound::Buffer { buffer, uav: Some(d), .. }) => {
                    rec.need(&buffer.0.track, None, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    writes = true;
                    *d
                }
                (RootKind::Texture, Bound::Texture(v)) => {
                    rec.need(&v.0.texture.0.track, Some(v.0.range()), SHADER_READ);
                    v.0.srv.unwrap_or_else(|| panic!("{label}: the texture at @group({g}) @binding({b}) has no TEXTURE_BINDING usage"))
                }
                (RootKind::StorageTexture, Bound::Texture(v)) => {
                    rec.need(&v.0.texture.0.track, Some(v.0.range()), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    writes = true;
                    v.0.uav.unwrap_or_else(|| panic!("{label}: the texture at @group({g}) @binding({b}) has no STORAGE_BINDING usage"))
                }
                (RootKind::AccelerationStructure, Bound::Tlas(tl, d)) => {
                    rec.keep_boxed(Box::new(tl.clone()));
                    *d
                }
                _ => mismatch(),
            };
            // SAFETY: a CPU descriptor of the bind group into this list's ring descriptors.
            unsafe { device.CopyDescriptorsSimple(1, views.view_cpu(at), src.handle(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
            at += 1;
        }
        if table.samplers {
            let (_, d) = group.0.sampler_indices.as_ref().unwrap_or_else(|| panic!("{label}: draw with no sampler bound in @group({g})"));
            // SAFETY: as above.
            unsafe { device.CopyDescriptorsSimple(1, views.view_cpu(at), d.handle(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
        }
        r.table(table.param, views.view_gpu(first));
        rec.keep_boxed(Box::new(group));
    }
    state.dirty = 0;
    writes
}

pub(crate) fn set_immediates(state: &mut PassState, offset: u32, data: &[u8]) {
    let o = offset as usize;
    assert!(o + data.len() <= MAX_IMMEDIATES, "set_immediates past {MAX_IMMEDIATES} bytes");
    state.immediates[o..o + data.len()].copy_from_slice(data);
    state.immediates_dirty = true;
}

pub(crate) fn flush_constants(r: &impl Roots, layout: &Layout, state: &mut PassState, special: [u32; 3]) {
    if state.immediates_dirty
        && let Some((param, dwords)) = layout.immediates
    {
        // (on the stack: the material's places change at nearly every draw)
        let mut words = [0u32; MAX_IMMEDIATES / 4];
        for (w, c) in words.iter_mut().zip(state.immediates[..dwords as usize * 4].as_chunks::<4>().0) {
            *w = u32::from_le_bytes(*c);
        }
        r.constants(param, &words[..dwords as usize]);
        state.immediates_dirty = false;
    }
    if state.special != Some(special) {
        r.constants(SPECIAL_PARAM, &special);
        state.special = Some(special);
    }
}

/// A render pass. Ends when dropped.
pub struct RenderPass<'e> {
    pub(crate) enc: &'e mut CommandEncoder,
    pub(crate) state: PassState,
    pipeline: Option<RenderPipeline>,
    index: Option<(Buffer, u64, t::IndexFormat)>,
    index_dirty: bool,
    vertex: [Option<(Buffer, u64)>; 8],
    vertex_dirty: bool,
    resolves: Vec<(TextureView, TextureView)>,
    end_timestamp: Option<(QuerySet, u32)>,
    occlusion: Option<QuerySet>,
    occlusion_index: Option<u32>,
    labelled: bool,
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        let rec = &mut self.enc.rec;
        let list = rec.list().clone();
        if let Some((q, i)) = &self.end_timestamp {
            q.write(&list, *i);
        }
        if !self.resolves.is_empty() {
            for (src, dst) in &self.resolves {
                rec.track.need(&src.0.texture.0.track, Some(src.0.range()), D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
                rec.track.need(&dst.0.texture.0.track, Some(dst.0.range()), D3D12_RESOURCE_STATE_RESOLVE_DEST);
            }
            rec.barriers();
            for (src, dst) in &self.resolves {
                let (s, d) = (&src.0, &dst.0);
                let ss = s.texture.0.track.subresource(s.base_mip_level, s.base_array_layer, 0);
                let ds = d.texture.0.track.subresource(d.base_mip_level, d.base_array_layer, 0);
                // SAFETY: a multisampled and a single-sampled view of the same format and size.
                unsafe { list.ResolveSubresource(&d.texture.0.raw, ds, &s.texture.0.raw, ss, conv::format(d.format)) };
                rec.keep(dst.clone());
            }
        }
        rec.track.restore_textures();
        rec.barriers();
        if self.labelled {
            // SAFETY: closes the pass's event.
            unsafe { list.EndEvent() };
        }
    }
}

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        let p = &pipeline.0;
        let rec = &mut self.enc.rec;
        let list = rec.list().clone();
        // SAFETY: a live root signature and pipeline (kept with the list).
        unsafe {
            list.SetGraphicsRootSignature(&p.layout.signature);
            list.SetPipelineState(&p.pso);
            list.IASetPrimitiveTopology(p.topology);
        }
        bind_heap(&rec.shared, &Graphics(&list), &mut self.state);
        rec.keep(pipeline.clone());
        self.pipeline = Some(pipeline.clone());
        self.vertex_dirty = true;
    }

    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        assert!(group <= MAX_ROOT_GROUP, "root bind groups are 0..={MAX_ROOT_GROUP}; group {} is the heap", mx_shader::HEAP_GROUP);
        self.state.groups[group as usize] = Some(bind_group.clone());
        self.state.offsets.retain(|o| o.0 != group);
        self.state.dirty |= 1 << group;
    }

    /// Read the buffer at (`group`, `binding`) of the bind group set at `group` from `offset`
    /// bytes past its own offset, for the draws that follow (until that group is set again):
    /// a draw's constants in one buffer of many. A multiple of 256; uniform buffers only on
    /// this backend.
    pub fn set_buffer_offset(&mut self, group: u32, binding: u32, offset: u64) {
        match self.state.offsets.iter_mut().find(|o| o.0 == group && o.1 == binding) {
            Some(o) => o.2 = offset,
            None => self.state.offsets.push((group, binding, offset)),
        }
        self.state.dirty |= 1 << group;
    }

    /// Set `data` at byte `offset` of the shader's `var<immediate>` block.
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        set_immediates(&mut self.state, offset, data);
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64) {
        assert!(slot < 8, "vertex buffer slot {slot}: mX takes 8");
        self.vertex[slot as usize] = Some((buffer.clone(), offset));
        self.vertex_dirty = true;
    }

    pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64, format: t::IndexFormat) {
        self.index = Some((buffer.clone(), offset, format));
        self.index_dirty = true;
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        let v = D3D12_VIEWPORT {
            TopLeftX: x,
            TopLeftY: y,
            Width: w,
            Height: h,
            MinDepth: min_depth,
            MaxDepth: max_depth,
        };
        // SAFETY: a plain state setter.
        unsafe { self.enc.rec.list().RSSetViewports(&[v]) };
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        let r = RECT {
            left: x as i32,
            top: y as i32,
            right: (x + w) as i32,
            bottom: (y + h) as i32,
        };
        // SAFETY: a plain state setter.
        unsafe { self.enc.rec.list().RSSetScissorRects(&[r]) };
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        // SAFETY: a plain state setter.
        unsafe { self.enc.rec.list().OMSetStencilRef(reference) };
    }

    pub fn set_blend_constant(&mut self, c: t::Color) {
        // SAFETY: a plain state setter.
        unsafe { self.enc.rec.list().OMSetBlendFactor(Some(&[c.r as f32, c.g as f32, c.b as f32, c.a as f32])) };
    }

    pub(crate) fn reset_state(&mut self) {
        self.pipeline = None;
        self.index = None;
        self.vertex = Default::default();
        self.state.groups = Default::default();
        self.state.offsets.clear();
        self.state.dirty = !0;
        self.state.immediates_dirty = true;
    }

    /// Bind everything the draw reads. `special`: the first vertex and first instance the
    /// shader's built-ins add (`None`: an indirect draw sets them).
    fn flush(&mut self, special: Option<[u32; 3]>, indexed: bool) -> RenderPipeline {
        let pipeline = self.pipeline.clone().expect("draw without a pipeline");
        let p = &pipeline.0;
        let rec = &mut self.enc.rec;
        let list = rec.list().clone();
        let r = Graphics(&list);
        for (b, _) in self.vertex.iter().flatten() {
            rec.track.need(&b.0.track, None, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        }
        if indexed {
            let (b, _, _) = self.index.as_ref().expect("draw_indexed without an index buffer");
            rec.track.need(&b.0.track, None, D3D12_RESOURCE_STATE_INDEX_BUFFER);
        }
        let writes = bind_roots(rec, &r, &p.layout, &mut self.state, &p.label);
        if p.layout.heap_buffers {
            need_heap_buffers(rec);
        }
        if rec.uav_dirty {
            rec.track.uav_barrier();
        }
        rec.uav_dirty = writes || p.layout.heap_buffer_writes;
        rec.barriers();
        if self.vertex_dirty {
            for (i, v) in self.vertex.iter().enumerate() {
                let Some((b, offset)) = v else { continue };
                let view = D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: b.0.va + offset,
                    SizeInBytes: (b.size() - offset) as u32,
                    StrideInBytes: p.strides[i],
                };
                // SAFETY: a view inside a live buffer (kept by the tracker).
                unsafe { list.IASetVertexBuffers(i as u32, Some(&[view])) };
            }
            self.vertex_dirty = false;
        }
        if self.index_dirty
            && let Some((b, offset, format)) = &self.index
        {
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: b.0.va + offset,
                SizeInBytes: (b.size() - offset) as u32,
                Format: conv::index_format(*format),
            };
            // SAFETY: as above.
            unsafe { list.IASetIndexBuffer(Some(&view)) };
            self.index_dirty = false;
        }
        match special {
            Some(s) => flush_constants(&r, &p.layout, &mut self.state, s),
            None => {
                // (the command signature sets them; they are unknown afterwards)
                let keep = self.state.special;
                flush_constants(&r, &p.layout, &mut self.state, keep.unwrap_or([0; 3]));
                self.state.special = None;
            }
        }
        pipeline
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.flush(Some([vertices.start, instances.start, 0]), false);
        if vertices.is_empty() || instances.is_empty() {
            return;
        }
        // SAFETY: a complete pipeline and bindings were set by `flush`.
        unsafe { self.enc.rec.list().DrawInstanced(vertices.len() as u32, instances.len() as u32, vertices.start, instances.start) };
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.flush(Some([base_vertex as u32, instances.start, 0]), true);
        if indices.is_empty() || instances.is_empty() {
            return;
        }
        // SAFETY: as in `draw`; the index range is the caller's to keep inside the buffer.
        unsafe {
            self.enc
                .rec
                .list()
                .DrawIndexedInstanced(indices.len() as u32, instances.len() as u32, indices.start, base_vertex, instances.start)
        };
    }

    /// Rewrite `count` indirect draws into the extended layout the command signatures read
    /// (first vertex / base vertex, first instance, 0, then the draw's arguments), as
    /// wgpu-core's indirect validation does on the GPU; here with copies.
    fn indirect(&mut self, args: &Buffer, offset: u64, count: u32, indexed: bool) {
        if count == 0 {
            return;
        }
        let n = if indexed { 20 } else { 16 };
        let ext = 12 + n;
        let rec = &mut self.enc.rec;
        let (scratch, at) = rec.scratch(ext * count as u64);
        rec.track.need(&args.0.track, None, D3D12_RESOURCE_STATE_COPY_SOURCE);
        rec.track.need(&scratch.0.track, None, D3D12_RESOURCE_STATE_COPY_DEST);
        rec.barriers();
        let list = rec.list().clone();
        for i in 0..count as u64 {
            let (src, dst) = (offset + i * n, at + i * ext);
            // (first vertex, first instance) or (base vertex, first instance)
            let special = if indexed { src + 12 } else { src + 8 };
            // SAFETY: ranges inside both buffers (the caller's arguments; checked by the debug layer).
            unsafe {
                list.CopyBufferRegion(&scratch.0.raw, dst, &args.0.raw, special, 8);
                list.CopyBufferRegion(&scratch.0.raw, dst + 12, &args.0.raw, src, n);
            }
        }
        rec.track.need(&scratch.0.track, None, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        let p = self.flush(None, indexed);
        let sig = if indexed { &p.0.draw_indexed } else { &p.0.draw };
        // SAFETY: a command signature of the pipeline's root signature and arguments it reads.
        unsafe { self.enc.rec.list().ExecuteIndirect(sig, count, &scratch.0.raw, at, None, 0) };
    }

    /// Draw a batch the GPU filled (see `CommandEncoder::encode_indirect_draws`), with the
    /// pipeline and bindings set now: `ExecuteIndirect` with the count buffer.
    pub fn execute_indirect_draws(&mut self, draws: &crate::IndirectDraws) {
        let src = draws.0.source.lock().clone().expect("execute_indirect_draws before encode_indirect_draws");
        if let Some((b, offset, format)) = &src.index {
            self.set_index_buffer(b, *offset, *format);
        }
        let indexed = draws.0.indexed;
        let rec = &mut self.enc.rec;
        rec.track.need(&draws.0.laid_out.0.track, None, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        rec.track.need(&src.count.0.track, None, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        rec.keep(draws.clone());
        rec.keep(src.count.clone());
        let p = self.flush(None, indexed);
        let sig = if indexed { &p.0.draw_indexed } else { &p.0.draw };
        // SAFETY: a command signature of the pipeline's root signature, the arguments laid out
        // for it, and a live count buffer.
        unsafe {
            self.enc
                .rec
                .list()
                .ExecuteIndirect(sig, draws.0.max_draws, &draws.0.laid_out.0.raw, 0, &src.count.0.raw, src.count_offset)
        };
    }

    pub fn draw_indirect(&mut self, args: &Buffer, offset: u64) {
        self.indirect(args, offset, 1, false);
    }

    pub fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64) {
        self.indirect(args, offset, 1, true);
    }

    pub fn multi_draw_indirect(&mut self, args: &Buffer, offset: u64, count: u32) {
        self.indirect(args, offset, count, false);
    }

    pub fn multi_draw_indexed_indirect(&mut self, args: &Buffer, offset: u64, count: u32) {
        self.indirect(args, offset, count, true);
    }

    /// Count the samples of the following draws into query `index` of the pass's occlusion
    /// query set.
    pub fn begin_occlusion_query(&mut self, index: u32) {
        let q = self.occlusion.as_ref().expect("begin_occlusion_query in a pass without an occlusion query set");
        q.begin_occlusion(self.enc.rec.list(), index);
        self.occlusion_index = Some(index);
    }

    pub fn end_occlusion_query(&mut self) {
        let q = self.occlusion.as_ref().expect("end_occlusion_query in a pass without an occlusion query set");
        let i = self.occlusion_index.take().expect("end_occlusion_query without begin_occlusion_query");
        q.end_occlusion(self.enc.rec.list(), i);
    }

    pub fn push_debug_group(&mut self, label: &str) {
        begin_event(self.enc.rec.list(), label);
    }

    pub fn pop_debug_group(&mut self) {
        // SAFETY: closes the event pushed before.
        unsafe { self.enc.rec.list().EndEvent() };
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        set_marker(self.enc.rec.list(), label);
    }

    /// The command list, for interop.
    pub fn raw(&self) -> &ID3D12GraphicsCommandList {
        self.enc.rec.list()
    }
}

/// A compute pass. Ends when dropped.
pub struct ComputePass<'e> {
    pub(crate) enc: &'e mut CommandEncoder,
    pub(crate) state: PassState,
    pipeline: Option<ComputePipeline>,
    end_timestamp: Option<(QuerySet, u32)>,
    labelled: bool,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        let rec = &mut self.enc.rec;
        let list = rec.list().clone();
        if let Some((q, i)) = &self.end_timestamp {
            q.write(&list, *i);
        }
        rec.track.restore_textures();
        rec.barriers();
        if self.labelled {
            // SAFETY: closes the pass's event.
            unsafe { list.EndEvent() };
        }
    }
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        let p = &pipeline.0;
        let rec = &mut self.enc.rec;
        let list = rec.list().clone();
        // SAFETY: a live root signature and pipeline (kept with the list).
        unsafe {
            list.SetComputeRootSignature(&p.layout.signature);
            list.SetPipelineState(&p.pso);
        }
        bind_heap(&rec.shared, &Compute(&list), &mut self.state);
        rec.keep(pipeline.clone());
        self.pipeline = Some(pipeline.clone());
    }

    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        assert!(group <= MAX_ROOT_GROUP, "root bind groups are 0..={MAX_ROOT_GROUP}");
        self.state.groups[group as usize] = Some(bind_group.clone());
        self.state.dirty |= 1 << group;
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        set_immediates(&mut self.state, offset, data);
    }

    /// Bind everything the dispatch reads; `groups`: the workgroup count the shader's
    /// `num_workgroups` reads (`None`: an indirect dispatch sets it).
    fn flush(&mut self, groups: Option<[u32; 3]>) -> ComputePipeline {
        let pipeline = self.pipeline.clone().expect("dispatch without a pipeline");
        let p = &pipeline.0;
        let rec = &mut self.enc.rec;
        let list = rec.list().clone();
        let r = Compute(&list);
        // (each dispatch is its own scope: storage textures of the last one go back home, so
        // the heap can read them)
        rec.track.begin_scope();
        self.state.dirty = !0;
        let writes = bind_roots(rec, &r, &p.layout, &mut self.state, &p.label);
        rec.track.end_scope();
        if p.layout.heap_buffers {
            need_heap_buffers(rec);
        }
        if rec.uav_dirty {
            rec.track.uav_barrier();
        }
        rec.uav_dirty = writes || p.layout.heap_buffer_writes;
        rec.barriers();
        match groups {
            Some(g) => flush_constants(&r, &p.layout, &mut self.state, g),
            None => {
                let keep = self.state.special;
                flush_constants(&r, &p.layout, &mut self.state, keep.unwrap_or([0; 3]));
                self.state.special = None;
            }
        }
        pipeline
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.flush(Some([x, y, z]));
        if x == 0 || y == 0 || z == 0 {
            return;
        }
        // SAFETY: a complete pipeline and bindings were set by `flush`.
        unsafe { self.enc.rec.list().Dispatch(x, y, z) };
    }

    /// The workgroup count is copied in front of itself for the command signature (it sets
    /// the special constants `num_workgroups` reads, then dispatches).
    pub fn dispatch_workgroups_indirect(&mut self, args: &Buffer, offset: u64) {
        let rec = &mut self.enc.rec;
        let (scratch, at) = rec.scratch(24);
        rec.track.need(&args.0.track, None, D3D12_RESOURCE_STATE_COPY_SOURCE);
        rec.track.need(&scratch.0.track, None, D3D12_RESOURCE_STATE_COPY_DEST);
        rec.barriers();
        // SAFETY: ranges inside both buffers.
        unsafe {
            rec.list().CopyBufferRegion(&scratch.0.raw, at, &args.0.raw, offset, 12);
            rec.list().CopyBufferRegion(&scratch.0.raw, at + 12, &args.0.raw, offset, 12);
        }
        rec.track.need(&scratch.0.track, None, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        let p = self.flush(None);
        // SAFETY: the pipeline's dispatch signature over 24 bytes of arguments.
        unsafe { self.enc.rec.list().ExecuteIndirect(&p.0.dispatch, 1, &scratch.0.raw, at, None, 0) };
    }

    pub fn push_debug_group(&mut self, label: &str) {
        begin_event(self.enc.rec.list(), label);
    }

    pub fn pop_debug_group(&mut self) {
        // SAFETY: closes the event pushed before.
        unsafe { self.enc.rec.list().EndEvent() };
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        set_marker(self.enc.rec.list(), label);
    }

    /// The command list, for interop.
    pub fn raw(&self) -> &ID3D12GraphicsCommandList {
        self.enc.rec.list()
    }
}

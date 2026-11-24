//! Command encoders record; `Queue::submit` replays on the immediate context (see `execute`).

use std::collections::HashMap;
use std::ops::Range;

use mx_shader::bound::Index;
use mx_shader::hlsl11::Class;
use mx_types as t;
use windows::Win32::Graphics::Direct3D11::*;
use windows::core::PCWSTR;

use crate::bundle::RenderBundle;
use crate::device::{Ctx, Shared};
use crate::pipeline::{ComputePipeline, RenderPipeline, StageBindings};
use crate::query::QuerySet;
use crate::resource::{BindGroup, Bound, Buffer, Texture, TextureView};
use crate::{Device, conv};

const GROUPS: usize = (mx_shader::MAX_ROOT_GROUP + 1) as usize;
const MAX_IMMEDIATES: usize = 256;
const MAX_VERTEX_BUFFERS: usize = 16;

#[derive(Clone)]
pub(crate) struct ColorAttachment {
    pub view: TextureView,
    pub resolve: Option<TextureView>,
    pub clear: Option<t::Color>,
    pub store: bool,
}

#[derive(Clone)]
pub(crate) struct DepthAttachment {
    pub view: TextureView,
    /// `Some(None)`: the aspect is loaded; `Some(Some(v))`: cleared to `v`; `None`: absent.
    pub depth: Option<Option<f32>>,
    pub depth_store: bool,
    pub stencil: Option<Option<u32>>,
    pub stencil_store: bool,
}

/// One recorded command.
#[derive(Clone)]
pub(crate) enum Cmd {
    ClearBuffer(Buffer, u64, u64),
    CopyBufferToBuffer(Buffer, u64, Buffer, u64, u64),
    CopyBufferToTexture(Buffer, t::TexelCopyBufferLayout, Texture, u32, t::Origin3d, t::Extent3d),
    CopyTextureToBuffer(Texture, u32, t::Origin3d, Buffer, t::TexelCopyBufferLayout, t::Extent3d),
    CopyTextureToTexture(Texture, u32, t::Origin3d, Texture, u32, t::Origin3d, t::Extent3d),
    ClearTexture(Texture, t::ImageSubresourceRange),
    GenerateMipmaps(Texture),
    BeginRenderPass {
        colors: Vec<Option<ColorAttachment>>,
        depth: Option<DepthAttachment>,
        timestamps: Option<(QuerySet, Option<u32>, Option<u32>)>,
        label: Option<String>,
    },
    EndRenderPass,
    BeginComputePass {
        timestamps: Option<(QuerySet, Option<u32>, Option<u32>)>,
        label: Option<String>,
    },
    EndComputePass,
    SetRenderPipeline(RenderPipeline),
    SetComputePipeline(ComputePipeline),
    SetBindGroup(u32, BindGroup),
    SetBufferOffset(u32, u32, u64),
    SetImmediates(u32, Vec<u8>),
    SetVertexBuffer(u32, Buffer, u64),
    SetIndexBuffer(Buffer, u64, t::IndexFormat),
    SetViewport(f32, f32, f32, f32, f32, f32),
    SetScissor(u32, u32, u32, u32),
    SetStencilReference(u32),
    SetBlendConstant(t::Color),
    Draw(Range<u32>, Range<u32>),
    DrawIndexed(Range<u32>, i32, Range<u32>),
    DrawIndirect(Buffer, u64, bool),
    Dispatch(u32, u32, u32),
    DispatchIndirect(Buffer, u64),
    BeginOcclusionQuery(QuerySet, u32),
    EndOcclusionQuery(QuerySet),
    WriteTimestamp(QuerySet, u32),
    ResolveQuerySet(QuerySet, Range<u32>, Buffer, u64),
    PushDebugGroup(String),
    PopDebugGroup,
    InsertDebugMarker(String),
    ExecuteBundle(RenderBundle),
}

pub struct CommandBuffer {
    pub(crate) cmds: Vec<Cmd>,
}

pub struct CommandEncoder {
    pub(crate) cmds: Vec<Cmd>,
}

impl Device {
    pub fn create_command_encoder(&self, label: Option<&str>) -> CommandEncoder {
        let mut cmds = Vec::new();
        if let Some(l) = label {
            cmds.push(Cmd::InsertDebugMarker(l.into()));
        }
        CommandEncoder { cmds }
    }
}

impl CommandEncoder {
    pub fn finish(self) -> CommandBuffer {
        CommandBuffer { cmds: self.cmds }
    }

    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: Option<u64>) {
        let size = size.unwrap_or(buffer.size() - offset);
        assert!(offset + size <= buffer.size(), "clear_buffer past the end of the buffer");
        if size > 0 {
            self.cmds.push(Cmd::ClearBuffer(buffer.clone(), offset, size));
        }
    }

    pub fn clear_texture(&mut self, texture: &Texture, range: &t::ImageSubresourceRange) {
        self.cmds.push(Cmd::ClearTexture(texture.clone(), *range));
    }

    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64) {
        assert!(src_offset + size <= src.size() && dst_offset + size <= dst.size(), "copy_buffer_to_buffer out of range");
        self.cmds.push(Cmd::CopyBufferToBuffer(src.clone(), src_offset, dst.clone(), dst_offset, size));
    }

    pub fn copy_buffer_to_texture(&mut self, src: t::TexelCopyBufferInfo<'_, Buffer>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d) {
        src.layout.assert_encoder_pitch();
        self.cmds
            .push(Cmd::CopyBufferToTexture(src.buffer.clone(), src.layout, dst.texture.clone(), dst.mip_level, dst.origin, size));
    }

    pub fn copy_texture_to_buffer(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyBufferInfo<'_, Buffer>, size: t::Extent3d) {
        dst.layout.assert_encoder_pitch();
        self.cmds
            .push(Cmd::CopyTextureToBuffer(src.texture.clone(), src.mip_level, src.origin, dst.buffer.clone(), dst.layout, size));
    }

    pub fn copy_texture_to_texture(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d) {
        self.cmds.push(Cmd::CopyTextureToTexture(
            src.texture.clone(),
            src.mip_level,
            src.origin,
            dst.texture.clone(),
            dst.mip_level,
            dst.origin,
            size,
        ));
    }

    pub fn generate_mipmaps(&mut self, texture: &Texture) {
        self.cmds.push(Cmd::GenerateMipmaps(texture.clone()));
    }

    pub fn resolve_query_set(&mut self, set: &QuerySet, range: Range<u32>, dst: &Buffer, dst_offset: u64) {
        assert!(range.end <= set.count(), "resolve_query_set range past the set");
        assert!(dst_offset + range.len() as u64 * 8 <= dst.size(), "resolve_query_set past the end of the buffer");
        self.cmds.push(Cmd::ResolveQuerySet(set.clone(), range, dst.clone(), dst_offset));
    }

    pub fn write_timestamp(&mut self, set: &QuerySet, index: u32) {
        self.cmds.push(Cmd::WriteTimestamp(set.clone(), index));
    }

    /// Not on this tier (see `Adapter::supports_indirect_count`).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indirect_draws(
        &mut self,
        _draws: &crate::IndirectDraws,
        _pipeline: &RenderPipeline,
        _args: &Buffer,
        _args_offset: u64,
        _count: &Buffer,
        _count_offset: u64,
        _index: Option<(&Buffer, u64, t::IndexFormat)>,
    ) {
        panic!("GPU-driven draw batches need a device with bindless (Metal 3, Vulkan, Direct3D 12)");
    }

    pub fn build_acceleration_structures<'a>(&mut self, _blas: impl IntoIterator<Item = &'a t::BlasBuildEntry<'a, crate::Blas, Buffer>>, _tlas: impl IntoIterator<Item = &'a crate::Tlas>) {
        panic!("acceleration structures need a device with bindless (Metal 3, Vulkan, Direct3D 12)");
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.cmds.push(Cmd::PushDebugGroup(label.into()));
    }
    pub fn pop_debug_group(&mut self) {
        self.cmds.push(Cmd::PopDebugGroup);
    }
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.cmds.push(Cmd::InsertDebugMarker(label.into()));
    }

    pub fn begin_render_pass(&mut self, desc: &t::RenderPassDescriptor<'_, TextureView, QuerySet>) -> RenderPass<'_> {
        let colors = desc
            .color_attachments
            .iter()
            .map(|c| {
                c.as_ref().map(|c| ColorAttachment {
                    view: c.view.clone(),
                    resolve: c.resolve_target.cloned(),
                    clear: match c.ops.load {
                        t::LoadOp::Clear(v) => Some(v),
                        // (`DontCare` clears too: the contents are undefined either way)
                        t::LoadOp::DontCare => Some(t::Color::TRANSPARENT),
                        t::LoadOp::Load => None,
                    },
                    store: c.ops.store == t::StoreOp::Store,
                })
            })
            .collect();
        let depth = desc.depth_stencil_attachment.as_ref().map(|d| DepthAttachment {
            view: d.view.clone(),
            depth: d.depth_ops.map(|o| match o.load {
                t::LoadOp::Clear(v) => Some(v),
                t::LoadOp::DontCare => Some(1.0),
                t::LoadOp::Load => None,
            }),
            depth_store: d.depth_ops.is_none_or(|o| o.store == t::StoreOp::Store),
            stencil: d.stencil_ops.map(|o| match o.load {
                t::LoadOp::Clear(v) => Some(v),
                t::LoadOp::DontCare => Some(0),
                t::LoadOp::Load => None,
            }),
            stencil_store: d.stencil_ops.is_none_or(|o| o.store == t::StoreOp::Store),
        });
        self.cmds.push(Cmd::BeginRenderPass {
            colors,
            depth,
            timestamps: desc
                .timestamp_writes
                .as_ref()
                .map(|w| (w.query_set.clone(), w.beginning_of_pass_write_index, w.end_of_pass_write_index)),
            label: desc.label.map(Into::into),
        });
        RenderPass {
            enc: self,
            occlusion: desc.occlusion_query_set.cloned(),
        }
    }

    pub fn begin_compute_pass(&mut self, desc: &t::ComputePassDescriptor<'_, QuerySet>) -> ComputePass<'_> {
        self.cmds.push(Cmd::BeginComputePass {
            timestamps: desc
                .timestamp_writes
                .as_ref()
                .map(|w| (w.query_set.clone(), w.beginning_of_pass_write_index, w.end_of_pass_write_index)),
            label: desc.label.map(Into::into),
        });
        ComputePass { enc: self }
    }
}

/// A render pass. Ends when dropped.
pub struct RenderPass<'e> {
    enc: &'e mut CommandEncoder,
    occlusion: Option<QuerySet>,
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        self.enc.cmds.push(Cmd::EndRenderPass);
    }
}

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.enc.cmds.push(Cmd::SetRenderPipeline(pipeline.clone()));
    }
    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        assert!(
            group <= mx_shader::MAX_ROOT_GROUP,
            "root bind groups are 0..={}; group {} is the heap",
            mx_shader::MAX_ROOT_GROUP,
            mx_shader::HEAP_GROUP
        );
        self.enc.cmds.push(Cmd::SetBindGroup(group, bind_group.clone()));
    }
    /// Read the uniform buffer at (`group`, `binding`) of the bind group set at `group` from
    /// `offset` bytes past its own offset, for the draws that follow (until that group is set
    /// again). A multiple of `min_uniform_buffer_offset_alignment`.
    pub fn set_buffer_offset(&mut self, group: u32, binding: u32, offset: u64) {
        self.enc.cmds.push(Cmd::SetBufferOffset(group, binding, offset));
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        assert!(offset as usize + data.len() <= MAX_IMMEDIATES, "set_immediates past {MAX_IMMEDIATES} bytes");
        self.enc.cmds.push(Cmd::SetImmediates(offset, data.to_vec()));
    }
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64) {
        self.enc.cmds.push(Cmd::SetVertexBuffer(slot, buffer.clone(), offset));
    }
    pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64, format: t::IndexFormat) {
        self.enc.cmds.push(Cmd::SetIndexBuffer(buffer.clone(), offset, format));
    }
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        self.enc.cmds.push(Cmd::SetViewport(x, y, w, h, min_depth, max_depth));
    }
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.enc.cmds.push(Cmd::SetScissor(x, y, w, h));
    }
    pub fn set_stencil_reference(&mut self, reference: u32) {
        self.enc.cmds.push(Cmd::SetStencilReference(reference));
    }
    pub fn set_blend_constant(&mut self, c: t::Color) {
        self.enc.cmds.push(Cmd::SetBlendConstant(c));
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.enc.cmds.push(Cmd::Draw(vertices, instances));
    }
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.enc.cmds.push(Cmd::DrawIndexed(indices, base_vertex, instances));
    }
    pub fn draw_indirect(&mut self, args: &Buffer, offset: u64) {
        self.enc.cmds.push(Cmd::DrawIndirect(args.clone(), offset, false));
    }
    pub fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64) {
        self.enc.cmds.push(Cmd::DrawIndirect(args.clone(), offset, true));
    }
    pub fn multi_draw_indirect(&mut self, args: &Buffer, offset: u64, count: u32) {
        for i in 0..count as u64 {
            self.draw_indirect(args, offset + i * std::mem::size_of::<t::DrawIndirectArgs>() as u64);
        }
    }
    pub fn multi_draw_indexed_indirect(&mut self, args: &Buffer, offset: u64, count: u32) {
        for i in 0..count as u64 {
            self.draw_indexed_indirect(args, offset + i * std::mem::size_of::<t::DrawIndexedIndirectArgs>() as u64);
        }
    }
    /// Not on this tier (see `Adapter::supports_indirect_count`).
    pub fn execute_indirect_draws(&mut self, _draws: &crate::IndirectDraws) {
        panic!("GPU-driven draw batches need a device with bindless (Metal 3, Vulkan, Direct3D 12)");
    }
    pub fn begin_occlusion_query(&mut self, index: u32) {
        let q = self.occlusion.clone().expect("begin_occlusion_query in a pass without an occlusion query set");
        self.enc.cmds.push(Cmd::BeginOcclusionQuery(q, index));
    }
    pub fn end_occlusion_query(&mut self) {
        let q = self.occlusion.clone().expect("end_occlusion_query in a pass without an occlusion query set");
        self.enc.cmds.push(Cmd::EndOcclusionQuery(q));
    }
    /// Replay bundles; afterwards the pass has no pipeline and no bindings, as in WebGPU.
    pub fn execute_bundles<'b>(&mut self, bundles: impl IntoIterator<Item = &'b RenderBundle>) {
        for b in bundles {
            self.enc.cmds.push(Cmd::ExecuteBundle(b.clone()));
        }
    }
    pub fn push_debug_group(&mut self, label: &str) {
        self.enc.push_debug_group(label);
    }
    pub fn pop_debug_group(&mut self) {
        self.enc.pop_debug_group();
    }
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.enc.insert_debug_marker(label);
    }
}

/// A compute pass. Ends when dropped.
pub struct ComputePass<'e> {
    enc: &'e mut CommandEncoder,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        self.enc.cmds.push(Cmd::EndComputePass);
    }
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.enc.cmds.push(Cmd::SetComputePipeline(pipeline.clone()));
    }
    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        assert!(group <= mx_shader::MAX_ROOT_GROUP, "root bind groups are 0..={}", mx_shader::MAX_ROOT_GROUP);
        self.enc.cmds.push(Cmd::SetBindGroup(group, bind_group.clone()));
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.enc.cmds.push(Cmd::SetImmediates(offset, data.to_vec()));
    }
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.enc.cmds.push(Cmd::Dispatch(x, y, z));
    }
    pub fn dispatch_workgroups_indirect(&mut self, args: &Buffer, offset: u64) {
        self.enc.cmds.push(Cmd::DispatchIndirect(args.clone(), offset));
    }
    pub fn push_debug_group(&mut self, label: &str) {
        self.enc.push_debug_group(label);
    }
    pub fn pop_debug_group(&mut self) {
        self.enc.pop_debug_group();
    }
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.enc.insert_debug_marker(label);
    }
}

// ---------------------------------------------------------------------------------------------
// Replay

#[derive(Clone)]
enum Pipeline {
    None,
    Render(RenderPipeline),
    Compute(ComputePipeline),
}

/// The pass being replayed.
struct Pass {
    colors: Vec<Option<ColorAttachment>>,
    depth: Option<DepthAttachment>,
    rtvs: Vec<Option<ID3D11RenderTargetView>>,
    dsv: Option<ID3D11DepthStencilView>,
    timestamps: Option<(QuerySet, Option<u32>)>,
    labelled: bool,
}

struct State {
    groups: [Option<BindGroup>; GROUPS],
    /// `set_buffer_offset`'s, by (group, binding), until the group is set again.
    offsets: HashMap<(u32, u32), u64>,
    immediates: [u8; MAX_IMMEDIATES],
    immediates_dirty: bool,
    pipeline: Pipeline,
    pipeline_dirty: bool,
    bindings_dirty: bool,
    vertex: [Option<(Buffer, u64)>; MAX_VERTEX_BUFFERS],
    vertex_dirty: bool,
    index: Option<(Buffer, u64, t::IndexFormat)>,
    stencil_reference: u32,
    blend_constant: [f32; 4],
    pass: Option<Pass>,
    compute_timestamps: Option<(QuerySet, Option<u32>)>,
    compute_labelled: bool,
}

impl State {
    fn new() -> State {
        State {
            groups: Default::default(),
            offsets: HashMap::new(),
            immediates: [0; MAX_IMMEDIATES],
            immediates_dirty: true,
            pipeline: Pipeline::None,
            pipeline_dirty: true,
            bindings_dirty: true,
            vertex: Default::default(),
            vertex_dirty: true,
            index: None,
            stencil_reference: 0,
            blend_constant: [0.0; 4],
            pass: None,
            compute_timestamps: None,
            compute_labelled: false,
        }
    }

    /// What a bundle leaves behind: no pipeline and no bindings, as in WebGPU.
    fn reset_bindings(&mut self) {
        self.groups = Default::default();
        self.offsets.clear();
        self.pipeline = Pipeline::None;
        self.vertex = Default::default();
        self.index = None;
        self.pipeline_dirty = true;
        self.bindings_dirty = true;
        self.vertex_dirty = true;
    }
}

pub(crate) fn execute(ctx: &mut Ctx, shared: &Shared, cb: CommandBuffer) {
    let mut st = State::new();
    for cmd in cb.cmds {
        run(ctx, shared, &mut st, cmd);
    }
    // (nothing stays bound past a command buffer)
    // SAFETY: plain state resets.
    unsafe { ctx.context.ClearState() };
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn run(ctx: &mut Ctx, shared: &Shared, st: &mut State, cmd: Cmd) {
    let c = ctx.context.clone();
    // SAFETY (for the calls below): the immediate context under its lock, with resources of
    // this device kept alive by the command.
    unsafe {
        match cmd {
            Cmd::ClearBuffer(b, offset, size) => {
                crate::resource::write_buffer(ctx, &b, offset, &vec![0u8; size as usize]);
            }
            Cmd::CopyBufferToBuffer(src, so, dst, d_o, size) => {
                let bx = D3D11_BOX {
                    left: so as u32,
                    top: 0,
                    front: 0,
                    right: (so + size) as u32,
                    bottom: 1,
                    back: 1,
                };
                let d = &dst.0;
                for r in [d.raw.as_ref(), d.cb.as_ref()].into_iter().flatten() {
                    c.CopySubresourceRegion(r, 0, d_o as u32, 0, 0, src.resource(), 0, Some(&bx));
                }
            }
            Cmd::CopyBufferToTexture(src, layout, dst, mip, origin, size) => {
                let len = (src.size() - layout.offset) as usize;
                let bytes = read_buffer(ctx, shared, &src, layout.offset, len);
                upload(ctx, &dst, mip, origin, size, t::TexelCopyBufferLayout { offset: 0, ..layout }, &bytes);
            }
            Cmd::CopyTextureToBuffer(src, mip, origin, dst, layout, size) => read_back(ctx, shared, &src, mip, origin, &dst, layout, size),
            Cmd::CopyTextureToTexture(src, smip, so, dst, dmip, d_o, size) => {
                let layered = src.dimension() != t::TextureDimension::D3;
                let layers = if layered { size.depth_or_array_layers } else { 1 };
                for l in 0..layers {
                    let bx = D3D11_BOX {
                        left: so.x,
                        top: so.y,
                        front: if layered { 0 } else { so.z },
                        right: so.x + size.width,
                        bottom: so.y + size.height,
                        back: if layered { 1 } else { so.z + size.depth_or_array_layers },
                    };
                    let (sl, dl, dz) = if layered { (so.z + l, d_o.z + l, 0) } else { (0, 0, d_o.z) };
                    c.CopySubresourceRegion(dst.raw(), dst.subresource(dmip, dl), d_o.x, d_o.y, dz, src.raw(), src.subresource(smip, sl), Some(&bx));
                }
            }
            Cmd::ClearTexture(tex, range) => {
                let (mips, layers) = range.resolve(tex.mip_level_count(), tex.layers());
                zero_texture(ctx, &tex, mips, layers);
            }
            Cmd::GenerateMipmaps(tex) => match tex.create_view(&Default::default()) {
                Ok(v) if v.0.srv.is_some() => c.GenerateMips(v.0.srv.as_ref().expect("checked")),
                _ => log::warn!("mX: generate_mipmaps on a texture that is not sampled and drawn into"),
            },
            Cmd::BeginRenderPass { colors, depth, timestamps, label } => {
                let labelled = label.as_deref().is_some_and(|l| {
                    begin_event(ctx, l);
                    true
                });
                unbind_reads(&c);
                let mut rtvs = Vec::with_capacity(colors.len());
                let mut extent = None;
                for a in &colors {
                    let Some(a) = a else {
                        rtvs.push(None);
                        continue;
                    };
                    let v = a.view.0.rtv(0).ok();
                    if let (Some(v), Some(color)) = (&v, a.clear) {
                        c.ClearRenderTargetView(v, &[color.r as f32, color.g as f32, color.b as f32, color.a as f32]);
                    }
                    let tex = a.view.texture();
                    extent.get_or_insert(tex.size().mip_level_size(a.view.0.base_mip_level, tex.dimension()));
                    rtvs.push(v);
                }
                let dsv = depth.as_ref().and_then(|d| {
                    let v = d.view.0.dsv(0).ok()?;
                    let mut flags = 0u32;
                    let (mut dv, mut sv) = (1.0f32, 0u8);
                    if let Some(Some(x)) = d.depth {
                        flags |= D3D11_CLEAR_DEPTH.0;
                        dv = x;
                    }
                    if let Some(Some(x)) = d.stencil {
                        flags |= D3D11_CLEAR_STENCIL.0;
                        sv = x as u8;
                    }
                    if flags != 0 {
                        c.ClearDepthStencilView(&v, flags, dv, sv);
                    }
                    let tex = d.view.texture();
                    extent.get_or_insert(tex.size().mip_level_size(d.view.0.base_mip_level, tex.dimension()));
                    Some(v)
                });
                c.OMSetRenderTargets(Some(&rtvs), dsv.as_ref());
                if let Some(e) = extent {
                    let vp = D3D11_VIEWPORT {
                        TopLeftX: 0.0,
                        TopLeftY: 0.0,
                        Width: e.width as f32,
                        Height: e.height as f32,
                        MinDepth: 0.0,
                        MaxDepth: 1.0,
                    };
                    c.RSSetViewports(Some(&[vp]));
                    c.RSSetScissorRects(Some(&[windows::Win32::Foundation::RECT {
                        left: 0,
                        top: 0,
                        right: e.width as i32,
                        bottom: e.height as i32,
                    }]));
                }
                let timestamps = timestamps.map(|(set, begin, end)| {
                    if let Some(i) = begin {
                        crate::query::begin_disjoint(ctx, shared);
                        ctx.context.End(&set.0.queries[i as usize]);
                    }
                    (set, end)
                });
                st.reset_bindings();
                st.pass = Some(Pass {
                    colors,
                    depth,
                    rtvs,
                    dsv,
                    timestamps,
                    labelled,
                });
            }
            Cmd::EndRenderPass => {
                let Some(pass) = st.pass.take() else { return };
                c.OMSetRenderTargets(None, None);
                // (what the pass does not store is the driver's to drop)
                for (a, v) in pass.colors.iter().zip(&pass.rtvs) {
                    if let (Some(a), Some(v)) = (a, v)
                        && !a.store
                        && a.resolve.is_none()
                    {
                        c.DiscardView(v);
                    }
                }
                if let (Some(d), Some(v)) = (&pass.depth, &pass.dsv)
                    && !d.depth_store
                    && (!d.view.format().has_stencil() || !d.stencil_store)
                {
                    c.DiscardView(v);
                }
                // MSAA resolves
                for a in pass.colors.iter().flatten() {
                    if let Some(r) = &a.resolve {
                        let (src, dst) = (a.view.texture(), r.texture());
                        c.ResolveSubresource(
                            dst.raw(),
                            dst.subresource(r.0.base_mip_level, r.0.base_array_layer),
                            src.raw(),
                            src.subresource(a.view.0.base_mip_level, a.view.0.base_array_layer),
                            conv::format(a.view.format()),
                        );
                    }
                }
                drop((pass.rtvs, pass.dsv));
                if let Some((set, Some(i))) = pass.timestamps {
                    crate::query::begin_disjoint(ctx, shared);
                    ctx.context.End(&set.0.queries[i as usize]);
                }
                if pass.labelled {
                    end_event(ctx);
                }
                st.reset_bindings();
            }
            Cmd::BeginComputePass { timestamps, label } => {
                st.compute_labelled = label.as_deref().is_some_and(|l| {
                    begin_event(ctx, l);
                    true
                });
                st.compute_timestamps = timestamps.map(|(set, begin, end)| {
                    if let Some(i) = begin {
                        crate::query::begin_disjoint(ctx, shared);
                        ctx.context.End(&set.0.queries[i as usize]);
                    }
                    (set, end)
                });
                st.reset_bindings();
            }
            Cmd::EndComputePass => {
                unbind_compute_writes(&c);
                if let Some((set, Some(i))) = st.compute_timestamps.take() {
                    crate::query::begin_disjoint(ctx, shared);
                    ctx.context.End(&set.0.queries[i as usize]);
                }
                if std::mem::take(&mut st.compute_labelled) {
                    end_event(ctx);
                }
                st.reset_bindings();
            }
            Cmd::SetRenderPipeline(p) => {
                st.pipeline = Pipeline::Render(p);
                st.pipeline_dirty = true;
                st.bindings_dirty = true;
                st.immediates_dirty = true;
                st.vertex_dirty = true;
            }
            Cmd::SetComputePipeline(p) => {
                st.pipeline = Pipeline::Compute(p);
                st.pipeline_dirty = true;
                st.bindings_dirty = true;
                st.immediates_dirty = true;
            }
            Cmd::SetBindGroup(g, bg) => {
                st.groups[g as usize] = Some(bg);
                st.offsets.retain(|(group, _), _| *group != g);
                st.bindings_dirty = true;
            }
            Cmd::SetBufferOffset(g, b, offset) => {
                st.offsets.insert((g, b), offset);
                st.bindings_dirty = true;
            }
            Cmd::SetImmediates(offset, data) => {
                st.immediates[offset as usize..offset as usize + data.len()].copy_from_slice(&data);
                st.immediates_dirty = true;
                // (heap slots are chosen by the immediates)
                st.bindings_dirty = true;
            }
            Cmd::SetVertexBuffer(slot, b, offset) => {
                st.vertex[slot as usize] = Some((b, offset));
                st.vertex_dirty = true;
            }
            Cmd::SetIndexBuffer(b, offset, format) => {
                c.IASetIndexBuffer(b.resource(), conv::index_format(format), offset as u32);
                st.index = Some((b, offset, format));
            }
            Cmd::SetViewport(x, y, w, h, min, max) => c.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: x,
                TopLeftY: y,
                Width: w,
                Height: h,
                MinDepth: min,
                MaxDepth: max,
            }])),
            Cmd::SetScissor(x, y, w, h) => c.RSSetScissorRects(Some(&[windows::Win32::Foundation::RECT {
                left: x as i32,
                top: y as i32,
                right: (x + w) as i32,
                bottom: (y + h) as i32,
            }])),
            Cmd::SetStencilReference(r) => {
                st.stencil_reference = r;
                st.pipeline_dirty = true;
            }
            Cmd::SetBlendConstant(k) => {
                st.blend_constant = [k.r as f32, k.g as f32, k.b as f32, k.a as f32];
                st.pipeline_dirty = true;
            }
            Cmd::Draw(v, i) => {
                if flush_render(ctx, shared, st, v.start as i32, i.start) {
                    ctx.context.DrawInstanced(v.len() as u32, i.len() as u32, v.start, i.start);
                }
            }
            Cmd::DrawIndexed(idx, base, i) => {
                if flush_render(ctx, shared, st, base, i.start) {
                    ctx.context.DrawIndexedInstanced(idx.len() as u32, i.len() as u32, idx.start, base, i.start);
                }
            }
            Cmd::DrawIndirect(args, offset, indexed) => {
                // (the first vertex and instance are in the arguments, which the CPU does not
                // see: shaders reading `vertex_index` count from 0 here)
                if flush_render(ctx, shared, st, 0, 0) {
                    let r = args.resource();
                    if indexed {
                        ctx.context.DrawIndexedInstancedIndirect(r, offset as u32);
                    } else {
                        ctx.context.DrawInstancedIndirect(r, offset as u32);
                    }
                }
            }
            Cmd::Dispatch(x, y, z) => {
                if flush_compute(ctx, shared, st) {
                    ctx.context.Dispatch(x, y, z);
                }
            }
            Cmd::DispatchIndirect(args, offset) => {
                if flush_compute(ctx, shared, st) {
                    ctx.context.DispatchIndirect(args.resource(), offset as u32);
                }
            }
            Cmd::BeginOcclusionQuery(set, i) => c.Begin(&set.0.queries[i as usize]),
            Cmd::EndOcclusionQuery(set) => {
                // (the query begun last: an occlusion pass runs one at a time)
                for q in &set.0.queries {
                    c.End(q);
                }
            }
            Cmd::WriteTimestamp(set, i) => {
                crate::query::begin_disjoint(ctx, shared);
                ctx.context.End(&set.0.queries[i as usize]);
            }
            Cmd::ResolveQuerySet(set, range, dst, offset) => {
                let values = crate::query::results(ctx, &set, range);
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                crate::resource::write_buffer(ctx, &dst, offset, &bytes);
            }
            Cmd::PushDebugGroup(l) => begin_event(ctx, &l),
            Cmd::PopDebugGroup => end_event(ctx),
            Cmd::InsertDebugMarker(l) => {
                if let Some(a) = &ctx.annotation {
                    let w = wide(&l);
                    a.SetMarker(PCWSTR(w.as_ptr()));
                }
            }
            Cmd::ExecuteBundle(b) => {
                st.reset_bindings();
                for cmd in b.0.cmds.iter().cloned() {
                    run(ctx, shared, st, cmd);
                }
                st.reset_bindings();
            }
        }
    }
}

fn begin_event(ctx: &Ctx, label: &str) {
    if let Some(a) = &ctx.annotation {
        let w = wide(label);
        // SAFETY: a nul-terminated string over the call.
        unsafe { a.BeginEvent(PCWSTR(w.as_ptr())) };
    }
}

fn end_event(ctx: &Ctx) {
    if let Some(a) = &ctx.annotation {
        // SAFETY: a plain call.
        unsafe { a.EndEvent() };
    }
}

/// No shader resource view stays bound where a pass is about to draw into its texture.
fn unbind_reads(c: &ID3D11DeviceContext1) {
    let none: [Option<ID3D11ShaderResourceView>; 128] = std::array::from_fn(|_| None);
    // SAFETY: plain state changes.
    unsafe {
        c.VSSetShaderResources(0, Some(&none));
        c.PSSetShaderResources(0, Some(&none));
        c.CSSetShaderResources(0, Some(&none));
    }
}

fn unbind_compute_writes(c: &ID3D11DeviceContext1) {
    let none: [Option<ID3D11UnorderedAccessView>; 8] = Default::default();
    // SAFETY: plain state changes.
    unsafe { c.CSSetUnorderedAccessViews(0, none.len() as u32, Some(none.as_ptr()), None) };
}

#[derive(Clone, Copy, PartialEq)]
enum StageKind {
    Vertex,
    Pixel,
    Compute,
}

/// What a stage binds: shader resource views, samplers, constant buffers (with their first
/// constant and count) and UAVs, by register.
#[derive(Default)]
struct Binds {
    srvs: Vec<(u8, ID3D11ShaderResourceView)>,
    samplers: Vec<(u8, ID3D11SamplerState)>,
    cbvs: Vec<(u8, ID3D11Buffer, u32, u32)>,
    uavs: Vec<(u8, ID3D11UnorderedAccessView)>,
}

/// A uniform buffer as a constant buffer: its `cb`, refreshed from its storage copy where
/// shaders may have written that.
fn constant_buffer(ctx: &Ctx, b: &Buffer, offset: u64, size: u64) -> Option<(ID3D11Buffer, u32, u32)> {
    let cb = b.0.cb.clone()?;
    if let (Some(raw), true) = (&b.0.raw, b.usage().contains(t::BufferUsages::STORAGE)) {
        // SAFETY: two buffers of the same size.
        unsafe { ctx.context.CopySubresourceRegion(&cb, 0, 0, 0, 0, raw, 0, None) };
    }
    // (in shader constants of 16 bytes; the window a multiple of 16 constants)
    let first = (offset / 16) as u32;
    let count = (size.div_ceil(16) as u32).next_multiple_of(16).min(4096);
    Some((cb, first, count))
}

fn gather(ctx: &Ctx, shared: &Shared, st: &State, b: &StageBindings) -> Binds {
    let mut out = Binds::default();
    for &(rb, class, reg) in &b.roots {
        let Some(r) = st.groups[rb.group as usize].as_ref().and_then(|g| g.get(rb.binding)) else {
            log::error!("mX: draw with nothing bound at @group({}) @binding({})", rb.group, rb.binding);
            continue;
        };
        let extra = st.offsets.get(&(rb.group, rb.binding)).copied().unwrap_or(0);
        match (class, r) {
            (Class::Cbv, Bound::Buffer { buffer, offset, size }) => {
                if let Some((cb, first, count)) = constant_buffer(ctx, buffer, offset + extra, *size) {
                    out.cbvs.push((reg, cb, first, count));
                }
            }
            (Class::Srv, Bound::Buffer { buffer, offset, size }) => {
                if let Ok(v) = buffer.range_views(offset + extra, *size)
                    && let Some(s) = v.srv
                {
                    out.srvs.push((reg, s));
                }
            }
            (Class::Uav, Bound::Buffer { buffer, offset, size }) => {
                if let Ok(v) = buffer.range_views(offset + extra, *size)
                    && let Some(u) = v.uav
                {
                    out.uavs.push((reg, u));
                }
            }
            (Class::Srv, Bound::Texture(v)) => {
                if let Some(s) = &v.0.srv {
                    out.srvs.push((reg, s.clone()));
                }
            }
            (Class::Uav, Bound::Texture(v)) => {
                if let Ok(u) = v.0.uav() {
                    out.uavs.push((reg, u));
                }
            }
            (Class::Sampler, Bound::Sampler(s)) => out.samplers.push((reg, s.0.raw.clone())),
            _ => log::error!("mX: the resource at @group({}) @binding({}) is not the kind the shader declares", rb.group, rb.binding),
        }
    }
    let heap = shared.heap.lock();
    for &(slot, class, reg) in &b.slots {
        let index = match slot.index {
            Index::Immediate(at) => {
                let at = at as usize;
                u32::from_le_bytes([st.immediates[at], st.immediates[at + 1], st.immediates[at + 2], st.immediates[at + 3]])
            }
            Index::Constant(i) => i,
        };
        let h = slot.heap_binding;
        if mx_shader::heap::is_sampler(h) {
            if let Some(s) = heap.sampler(index) {
                out.samplers.push((reg, s.clone()));
            }
        } else if mx_shader::heap::is_texture(h) {
            if let Some(s) = heap.texture(index).and_then(|t| t.srv.clone()) {
                out.srvs.push((reg, s));
            }
        } else if mx_shader::heap::is_storage_texture(h) {
            if let Some(u) = heap.texture(index).and_then(|t| t.uav.clone()) {
                out.uavs.push((reg, u));
            }
        } else if let Some(bs) = heap.buffer(index)
            && let Ok(v) = bs.buffer.range_views(bs.offset, bs.size)
        {
            match class {
                Class::Uav => out.uavs.extend(v.uav.map(|u| (reg, u))),
                _ => out.srvs.extend(v.srv.map(|s| (reg, s))),
            }
        }
    }
    out
}

/// Bind `binds` to `stage`; returns the UAVs a pixel stage needs (bound with the targets).
fn apply(c: &ID3D11DeviceContext1, stage: StageKind, binds: Binds) -> Vec<(u8, ID3D11UnorderedAccessView)> {
    // SAFETY: plain state changes with live objects.
    unsafe {
        for (reg, s) in binds.srvs {
            let v = [Some(s)];
            match stage {
                StageKind::Vertex => c.VSSetShaderResources(reg as u32, Some(&v)),
                StageKind::Pixel => c.PSSetShaderResources(reg as u32, Some(&v)),
                StageKind::Compute => c.CSSetShaderResources(reg as u32, Some(&v)),
            }
        }
        for (reg, s) in binds.samplers {
            let v = [Some(s)];
            match stage {
                StageKind::Vertex => c.VSSetSamplers(reg as u32, Some(&v)),
                StageKind::Pixel => c.PSSetSamplers(reg as u32, Some(&v)),
                StageKind::Compute => c.CSSetSamplers(reg as u32, Some(&v)),
            }
        }
        for (reg, b, first, count) in binds.cbvs {
            let v = [Some(b)];
            let (f, n) = ([first], [count]);
            match stage {
                StageKind::Vertex => c.VSSetConstantBuffers1(reg as u32, 1, Some(v.as_ptr()), Some(f.as_ptr()), Some(n.as_ptr())),
                StageKind::Pixel => c.PSSetConstantBuffers1(reg as u32, 1, Some(v.as_ptr()), Some(f.as_ptr()), Some(n.as_ptr())),
                StageKind::Compute => c.CSSetConstantBuffers1(reg as u32, 1, Some(v.as_ptr()), Some(f.as_ptr()), Some(n.as_ptr())),
            }
        }
        match stage {
            StageKind::Compute => {
                for (reg, u) in binds.uavs {
                    let v = [Some(u)];
                    c.CSSetUnorderedAccessViews(reg as u32, 1, Some(v.as_ptr()), None);
                }
                Vec::new()
            }
            StageKind::Pixel => binds.uavs,
            StageKind::Vertex => {
                if !binds.uavs.is_empty() {
                    log::error!("mX: a vertex shader writes storage, which Direct3D 11 does not allow");
                }
                Vec::new()
            }
        }
    }
}

/// Write the immediates' constant buffer, and bind it and the special constants to `stages`.
fn write_constants(ctx: &Ctx, st: &State, first_vertex: i32, first_instance: u32, compute: bool, immediates: bool) {
    let c = &ctx.context;
    // SAFETY: dynamic constant buffers of the device, mapped with DISCARD and written whole.
    unsafe {
        if st.immediates_dirty && immediates {
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            if c.Map(&ctx.immediates, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut m)).is_ok() {
                std::ptr::copy_nonoverlapping(st.immediates.as_ptr(), m.pData.cast::<u8>(), MAX_IMMEDIATES);
                c.Unmap(&ctx.immediates, 0);
            }
        }
        let im = [Some(ctx.immediates.clone())];
        let reg = mx_shader::hlsl11::IMMEDIATES_REGISTER;
        if compute {
            c.CSSetConstantBuffers(reg, Some(&im));
            return;
        }
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        if c.Map(&ctx.special, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut m)).is_ok() {
            let v: [u32; 4] = [first_vertex as u32, first_instance, 0, 0];
            std::ptr::copy_nonoverlapping(v.as_ptr().cast::<u8>(), m.pData.cast::<u8>(), 16);
            c.Unmap(&ctx.special, 0);
        }
        let sp = [Some(ctx.special.clone())];
        let sreg = mx_shader::hlsl11::SPECIAL_CONSTANTS_REGISTER;
        c.VSSetConstantBuffers(reg, Some(&im));
        c.PSSetConstantBuffers(reg, Some(&im));
        c.VSSetConstantBuffers(sreg, Some(&sp));
        c.PSSetConstantBuffers(sreg, Some(&sp));
    }
}

/// Set what changed before a draw; false: nothing to draw with (no render pipeline).
fn flush_render(ctx: &mut Ctx, shared: &Shared, st: &mut State, first_vertex: i32, first_instance: u32) -> bool {
    let Pipeline::Render(p) = st.pipeline.clone() else {
        log::error!("mX: draw without a render pipeline");
        return false;
    };
    let pi = &p.0;
    let c = ctx.context.clone();
    // SAFETY: plain state changes with live objects.
    unsafe {
        if st.pipeline_dirty {
            c.IASetInputLayout(pi.layout.as_ref());
            c.IASetPrimitiveTopology(pi.topology);
            c.VSSetShader(&pi.vs, None);
            c.PSSetShader(pi.ps.as_ref(), None);
            c.RSSetState(&pi.rasterizer);
            c.OMSetBlendState(&pi.blend, Some(&st.blend_constant), pi.sample_mask);
            c.OMSetDepthStencilState(&pi.depth_stencil, st.stencil_reference);
            st.pipeline_dirty = false;
        }
        if st.vertex_dirty {
            for (slot, stride) in pi.strides.iter().enumerate() {
                if let Some((b, offset)) = &st.vertex[slot] {
                    let v = [Some(b.resource().clone())];
                    let (s, o) = ([*stride], [*offset as u32]);
                    c.IASetVertexBuffers(slot as u32, 1, Some(v.as_ptr()), Some(s.as_ptr()), Some(o.as_ptr()));
                }
            }
            st.vertex_dirty = false;
        }
    }
    if st.bindings_dirty {
        let vb = gather(ctx, shared, st, &pi.vertex);
        let fb = gather(ctx, shared, st, &pi.fragment);
        apply(&c, StageKind::Vertex, vb);
        let uavs = apply(&c, StageKind::Pixel, fb);
        if !uavs.is_empty() {
            let base = pi.uav_base;
            let count = uavs.iter().map(|(r, _)| *r as u32 + 1).max().unwrap_or(base) - base;
            let mut list: Vec<Option<ID3D11UnorderedAccessView>> = vec![None; count as usize];
            for (r, u) in uavs {
                list[(r as u32 - base) as usize] = Some(u);
            }
            // SAFETY: the pass's targets again, the fragment stage's UAVs after them.
            if let Some(pass) = &st.pass {
                unsafe {
                    c.OMSetRenderTargetsAndUnorderedAccessViews(Some(&pass.rtvs), pass.dsv.as_ref(), base, count, Some(list.as_ptr()), None);
                }
            }
        }
        st.bindings_dirty = false;
    }
    write_constants(ctx, st, first_vertex, first_instance, false, pi.vertex.uses_immediates || pi.fragment.uses_immediates);
    st.immediates_dirty = false;
    true
}

fn flush_compute(ctx: &mut Ctx, shared: &Shared, st: &mut State) -> bool {
    let Pipeline::Compute(p) = st.pipeline.clone() else {
        log::error!("mX: dispatch without a compute pipeline");
        return false;
    };
    let c = ctx.context.clone();
    if st.pipeline_dirty {
        // SAFETY: a plain state change.
        unsafe { c.CSSetShader(&p.0.cs, None) };
        st.pipeline_dirty = false;
    }
    if st.bindings_dirty {
        let b = gather(ctx, shared, st, &p.0.stage);
        apply(&c, StageKind::Compute, b);
        st.bindings_dirty = false;
    }
    write_constants(ctx, st, 0, 0, true, p.0.stage.uses_immediates);
    st.immediates_dirty = false;
    true
}

// ---------------------------------------------------------------------------------------------
// Uploads, read-backs and zeroing

/// `len` bytes of `b` from `offset`, read by the CPU (through a staging copy for a buffer the
/// CPU cannot map; waits for the GPU).
fn read_buffer(ctx: &Ctx, shared: &Shared, b: &Buffer, offset: u64, len: usize) -> Vec<u8> {
    let c = &ctx.context;
    let mut out = vec![0u8; len];
    if len == 0 {
        return out;
    }
    // SAFETY: buffers of this device, mapped and unmapped under the context's lock.
    unsafe {
        let staging = if b.0.staging {
            b.resource().clone()
        } else {
            let d = D3D11_BUFFER_DESC {
                ByteWidth: (len as u32).next_multiple_of(16),
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
                StructureByteStride: 0,
            };
            let mut s = None;
            if shared.device.CreateBuffer(&d, None, Some(&mut s)).is_err() {
                return out;
            }
            let Some(s) = s else { return out };
            let bx = D3D11_BOX {
                left: offset as u32,
                top: 0,
                front: 0,
                right: offset as u32 + len as u32,
                bottom: 1,
                back: 1,
            };
            c.CopySubresourceRegion(&s, 0, 0, 0, 0, b.resource(), 0, Some(&bx));
            s
        };
        let at = if b.0.staging { offset as usize } else { 0 };
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        if c.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut m)).is_ok() {
            std::ptr::copy_nonoverlapping(m.pData.cast::<u8>().add(at), out.as_mut_ptr(), len);
            c.Unmap(&staging, 0);
        }
    }
    out
}

/// Bytes per row and rows per image (in block rows) of a copy of `size` with `layout`.
fn pitches(format: t::TextureFormat, size: t::Extent3d, layout: &t::TexelCopyBufferLayout) -> (u32, u32) {
    let (_, bh) = format.block_dimensions();
    let row = layout.bytes_per_row.unwrap_or_else(|| format.bytes_per_row(size.width).unwrap_or(size.width * 4));
    let rows = layout.rows_per_image.unwrap_or(size.height).div_ceil(bh);
    (row, rows)
}

/// Write `data` (laid out as `layout`, its offset already applied) into `size` texels of mip
/// `mip` of `tex` at `origin`.
pub(crate) fn upload(ctx: &Ctx, tex: &Texture, mip: u32, origin: t::Origin3d, size: t::Extent3d, layout: t::TexelCopyBufferLayout, data: &[u8]) {
    let (row, rows) = pitches(tex.format(), size, &layout);
    let image = row as usize * rows as usize;
    let layered = tex.dimension() != t::TextureDimension::D3;
    let layers = if layered { size.depth_or_array_layers } else { 1 };
    for l in 0..layers {
        let bx = D3D11_BOX {
            left: origin.x,
            top: origin.y,
            front: if layered { 0 } else { origin.z },
            right: origin.x + size.width,
            bottom: origin.y + size.height,
            back: if layered { 1 } else { origin.z + size.depth_or_array_layers },
        };
        let start = if layered { l as usize * image } else { 0 };
        if start >= data.len() {
            break;
        }
        let layer = if layered { origin.z + l } else { 0 };
        // SAFETY: the source holds the rows the box needs (the caller's layout).
        unsafe {
            ctx.context
                .UpdateSubresource(tex.raw(), tex.subresource(mip, layer), Some(&bx), data[start..].as_ptr().cast(), row, image as u32)
        };
    }
}

/// Copy `size` texels of mip `mip` of `tex` at `origin` into `dst` as `layout` lays them out
/// (through a staging texture; waits for the GPU).
#[allow(clippy::too_many_arguments)]
fn read_back(ctx: &Ctx, shared: &Shared, tex: &Texture, mip: u32, origin: t::Origin3d, dst: &Buffer, layout: t::TexelCopyBufferLayout, size: t::Extent3d) {
    let format = tex.format();
    let (row, rows) = pitches(format, size, &layout);
    let (bw, bh) = format.block_dimensions();
    let tight = format.bytes_per_row(size.width).unwrap_or(size.width * 4) as usize;
    let layered = tex.dimension() != t::TextureDimension::D3;
    let layers = if layered { size.depth_or_array_layers } else { 1 };
    let depth = if layered { 1 } else { size.depth_or_array_layers };
    let c = &ctx.context;
    let block_rows = size.height.div_ceil(bh);
    let mut out = vec![0u8; (row as usize * rows as usize * (layers * depth) as usize).max(1)];
    for l in 0..layers {
        // a staging texture of the copied region
        let d = D3D11_TEXTURE2D_DESC {
            Width: size.width.next_multiple_of(bw),
            Height: size.height.next_multiple_of(bh),
            MipLevels: 1,
            ArraySize: 1,
            Format: conv::copy_format(format, t::TextureAspect::All),
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
        };
        let mut s = None;
        // SAFETY: plain creation, copy, map and unmap under the context's lock.
        unsafe {
            if !layered || shared.device.CreateTexture2D(&d, None, Some(&mut s)).is_err() {
                // (3D read-backs: through a 3D staging texture)
                let d3 = D3D11_TEXTURE3D_DESC {
                    Width: size.width,
                    Height: size.height,
                    Depth: depth,
                    MipLevels: 1,
                    Format: conv::copy_format(format, t::TextureAspect::All),
                    Usage: D3D11_USAGE_STAGING,
                    BindFlags: 0,
                    CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                    MiscFlags: 0,
                };
                let mut s3 = None;
                if shared.device.CreateTexture3D(&d3, None, Some(&mut s3)).is_err() {
                    log::error!("mX: no staging texture for a read-back");
                    return;
                }
                let Some(s3) = s3 else { return };
                let bx = D3D11_BOX {
                    left: origin.x,
                    top: origin.y,
                    front: origin.z,
                    right: origin.x + size.width,
                    bottom: origin.y + size.height,
                    back: origin.z + depth,
                };
                c.CopySubresourceRegion(&s3, 0, 0, 0, 0, tex.raw(), tex.subresource(mip, 0), Some(&bx));
                let mut m = D3D11_MAPPED_SUBRESOURCE::default();
                if c.Map(&s3, 0, D3D11_MAP_READ, 0, Some(&mut m)).is_ok() {
                    for z in 0..depth as usize {
                        for y in 0..block_rows as usize {
                            let from = m.pData.cast::<u8>().add(z * m.DepthPitch as usize + y * m.RowPitch as usize);
                            let to = z * row as usize * rows as usize + y * row as usize;
                            std::ptr::copy_nonoverlapping(from, out.as_mut_ptr().add(to), tight);
                        }
                    }
                    c.Unmap(&s3, 0);
                }
                break;
            }
            let Some(s) = s else { return };
            let bx = D3D11_BOX {
                left: origin.x,
                top: origin.y,
                front: 0,
                right: origin.x + size.width,
                bottom: origin.y + size.height,
                back: 1,
            };
            c.CopySubresourceRegion(&s, 0, 0, 0, 0, tex.raw(), tex.subresource(mip, origin.z + l), Some(&bx));
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            if c.Map(&s, 0, D3D11_MAP_READ, 0, Some(&mut m)).is_ok() {
                for y in 0..block_rows as usize {
                    let from = m.pData.cast::<u8>().add(y * m.RowPitch as usize);
                    let to = l as usize * row as usize * rows as usize + y * row as usize;
                    std::ptr::copy_nonoverlapping(from, out.as_mut_ptr().add(to), tight);
                }
                c.Unmap(&s, 0);
            }
        }
    }
    // (the rows as laid out; the last image's padding rows are not written, as wgpu does)
    let len = ((layers * depth) as usize - 1) * row as usize * rows as usize + (block_rows as usize - 1) * row as usize + tight;
    let len = len.min(out.len()).min((dst.size() - layout.offset) as usize);
    crate::resource::write_buffer(ctx, dst, layout.offset, &out[..len]);
}

/// Zero mips `mips` of layers `layers` of `tex` (wgpu's guarantee for new textures, and
/// `clear_texture`): render targets and depth through clears, the rest through uploads.
pub(crate) fn zero_texture(ctx: &mut Ctx, tex: &Texture, mips: Range<u32>, layers: Range<u32>) {
    let format = tex.format();
    let usage = tex.usage();
    let c = ctx.context.clone();
    for mip in mips {
        let size = tex.size().mip_level_size(mip, tex.dimension());
        let slices = if tex.dimension() == t::TextureDimension::D3 {
            0..size.depth_or_array_layers
        } else {
            layers.clone()
        };
        if usage.contains(t::TextureUsages::RENDER_ATTACHMENT) {
            for s in slices {
                // SAFETY: plain clears of views of the texture.
                unsafe {
                    if format.has_depth() {
                        if let Ok(v) = crate::resource::dsv_of(tex, format, mip, s, 0) {
                            let flags = D3D11_CLEAR_DEPTH.0 | if format.has_stencil() { D3D11_CLEAR_STENCIL.0 } else { 0 };
                            c.ClearDepthStencilView(&v, flags, 0.0, 0);
                        }
                    } else if let Ok(v) = crate::resource::rtv_of(tex, format, mip, s) {
                        c.ClearRenderTargetView(&v, &[0.0; 4]);
                    }
                }
            }
            continue;
        }
        if tex.sample_count() > 1 || format.has_depth() && format.has_stencil() {
            // (neither takes uploads; nothing else can write them either)
            continue;
        }
        let Some(row) = format.bytes_per_row(size.width) else { continue };
        let (_, bh) = format.block_dimensions();
        let rows = size.height.div_ceil(bh);
        let depth = if tex.dimension() == t::TextureDimension::D3 { size.depth_or_array_layers } else { 1 };
        let zeros = vec![0u8; row as usize * rows as usize * depth as usize];
        let one = t::Extent3d { depth_or_array_layers: depth, ..size };
        let ls = if tex.dimension() == t::TextureDimension::D3 { 0..1 } else { layers.clone() };
        for l in ls {
            upload(
                ctx,
                tex,
                mip,
                t::Origin3d { x: 0, y: 0, z: l },
                one,
                t::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(size.height),
                },
                &zeros,
            );
        }
    }
}

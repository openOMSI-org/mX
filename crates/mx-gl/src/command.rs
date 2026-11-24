//! Command encoders record; `Queue::submit` replays on the context (see `execute`).

use std::ops::Range;

use glow::HasContext;
use mx_shader::bound::Index;
use mx_shader::glsl::Class;
use mx_types as t;

use crate::bundle::RenderBundle;
use crate::device::Shared;
use crate::pipeline::{Bindings, ComputePipeline, RenderPipeline};
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

enum Pipeline {
    None,
    Render(RenderPipeline),
    Compute(ComputePipeline),
}

/// The state of a replay: what is set and what must be applied before the next draw.
struct State {
    pipeline: Pipeline,
    groups: [Option<BindGroup>; GROUPS],
    immediates: [u8; MAX_IMMEDIATES],
    vertex: [Option<(Buffer, u64)>; MAX_VERTEX_BUFFERS],
    index: Option<(Buffer, u64, t::IndexFormat)>,
    bindings_dirty: bool,
    immediates_dirty: bool,
    /// Offsets added to root uniform buffers' own (`set_buffer_offset`): (group, binding, bytes).
    offsets: Vec<(u32, u32, u64)>,
    /// The attribute arrays enabled now.
    enabled_attribs: u32,
    /// Something wrote storage since the last `glMemoryBarrier`.
    storage_written: bool,
    /// The current pass's end-of-pass timestamp, and its colour attachments (for resolves).
    pass_end: Option<(QuerySet, u32)>,
    pass_colors: Vec<Option<ColorAttachment>>,
    pass_depth: Option<DepthAttachment>,
    /// The pass opened a debug group (its label), closed when it ends.
    pass_labelled: bool,
    target_size: (u32, u32),
}

impl State {
    fn new() -> State {
        State {
            pipeline: Pipeline::None,
            groups: Default::default(),
            immediates: [0; MAX_IMMEDIATES],
            vertex: Default::default(),
            index: None,
            bindings_dirty: true,
            immediates_dirty: true,
            offsets: Vec::new(),
            enabled_attribs: 0,
            storage_written: false,
            pass_end: None,
            pass_colors: Vec::new(),
            pass_depth: None,
            pass_labelled: false,
            target_size: (0, 0),
        }
    }
}

/// Replay one command buffer on the current context.
pub(crate) fn execute(gl: &glow::Context, shared: &Shared, cb: CommandBuffer) {
    let mut st = State::new();
    for cmd in cb.cmds {
        run(gl, shared, &mut st, cmd);
    }
    // SAFETY: plain state resets on the current context.
    unsafe {
        gl.use_program(None);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }
}

fn barrier_if_written(gl: &glow::Context, shared: &Shared, st: &mut State) {
    if st.storage_written && shared.caps.compute {
        // SAFETY: a barrier on the current context.
        unsafe { gl.memory_barrier(glow::ALL_BARRIER_BITS) };
        st.storage_written = false;
    }
}

fn run(gl: &glow::Context, shared: &Shared, st: &mut State, cmd: Cmd) {
    // SAFETY (the whole function): GL calls on the current context, with objects mX keeps
    // alive and ranges checked when the commands were recorded.
    unsafe {
        match cmd {
            Cmd::ClearBuffer(b, offset, size) => {
                barrier_if_written(gl, shared, st);
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(b.0.raw));
                gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset as i32, &vec![0u8; size as usize]);
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            }
            Cmd::CopyBufferToBuffer(src, so, dst, d_o, size) => {
                barrier_if_written(gl, shared, st);
                gl.bind_buffer(glow::COPY_READ_BUFFER, Some(src.0.raw));
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(dst.0.raw));
                gl.copy_buffer_sub_data(glow::COPY_READ_BUFFER, glow::COPY_WRITE_BUFFER, so as i32, d_o as i32, size as i32);
                gl.bind_buffer(glow::COPY_READ_BUFFER, None);
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            }
            Cmd::CopyBufferToTexture(src, layout, tex, mip, origin, size) => {
                barrier_if_written(gl, shared, st);
                gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(src.0.raw));
                upload(gl, &tex, mip, origin, size, layout, Source::Buffer(layout.offset));
                gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            }
            Cmd::CopyTextureToBuffer(tex, mip, origin, dst, layout, size) => {
                barrier_if_written(gl, shared, st);
                read_back(gl, shared, &tex, mip, origin, &dst, layout, size);
            }
            Cmd::CopyTextureToTexture(src, smip, so, dst, dmip, d_o, size) => {
                barrier_if_written(gl, shared, st);
                copy_texture(gl, shared, &src, smip, so, &dst, dmip, d_o, size);
            }
            Cmd::ClearTexture(tex, range) => {
                let (mips, layers) = range.resolve(tex.mip_level_count(), tex.layers());
                zero_range(gl, shared, &tex, mips, layers);
            }
            Cmd::GenerateMipmaps(tex) => {
                gl.bind_texture(tex.0.target, Some(tex.0.raw));
                gl.tex_parameter_i32(tex.0.target, glow::TEXTURE_BASE_LEVEL, 0);
                gl.tex_parameter_i32(tex.0.target, glow::TEXTURE_MAX_LEVEL, tex.mip_level_count() as i32 - 1);
                gl.generate_mipmap(tex.0.target);
                gl.bind_texture(tex.0.target, None);
            }
            Cmd::BeginRenderPass { colors, depth, timestamps, label } => {
                barrier_if_written(gl, shared, st);
                if let Some(l) = &label {
                    push_group(gl, l);
                }
                if let Some((q, begin, end)) = &timestamps {
                    if let Some(i) = begin {
                        q.timestamp(gl, *i);
                    }
                    st.pass_end = end.map(|e| (q.clone(), e));
                }
                begin_pass(gl, shared, st, &colors, depth.as_ref());
                st.pass_colors = colors;
                st.pass_depth = depth;
                st.pass_labelled = label.is_some();
            }
            Cmd::EndRenderPass => {
                end_pass(gl, shared, st);
                end_scope(gl, st);
            }
            Cmd::BeginComputePass { timestamps, label } => {
                if let Some(l) = &label {
                    push_group(gl, l);
                }
                if let Some((q, begin, end)) = timestamps {
                    if let Some(i) = begin {
                        q.timestamp(gl, i);
                    }
                    st.pass_end = end.map(|e| (q.clone(), e));
                }
                st.pass_colors.clear();
                st.pass_labelled = label.is_some();
            }
            Cmd::EndComputePass => {
                barrier_if_written(gl, shared, st);
                end_scope(gl, st);
            }
            Cmd::SetRenderPipeline(p) => {
                gl.use_program(Some(p.0.program));
                apply_render_state(gl, shared, &p);
                st.pipeline = Pipeline::Render(p);
                st.bindings_dirty = true;
                st.immediates_dirty = true;
            }
            Cmd::SetComputePipeline(p) => {
                gl.use_program(Some(p.0.program));
                st.pipeline = Pipeline::Compute(p);
                st.bindings_dirty = true;
                st.immediates_dirty = true;
            }
            Cmd::SetBindGroup(g, bg) => {
                st.groups[g as usize] = Some(bg);
                st.offsets.retain(|o| o.0 != g);
                st.bindings_dirty = true;
            }
            Cmd::SetBufferOffset(g, b, offset) => {
                match st.offsets.iter_mut().find(|o| o.0 == g && o.1 == b) {
                    Some(o) => o.2 = offset,
                    None => st.offsets.push((g, b, offset)),
                }
                st.bindings_dirty = true;
            }
            Cmd::SetImmediates(offset, data) => {
                let o = offset as usize;
                st.immediates[o..o + data.len()].copy_from_slice(&data);
                st.immediates_dirty = true;
                // (heap slots read their indices from the immediates)
                st.bindings_dirty = true;
            }
            Cmd::SetVertexBuffer(slot, b, offset) => st.vertex[slot as usize] = Some((b, offset)),
            Cmd::SetIndexBuffer(b, offset, f) => {
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(b.0.raw));
                st.index = Some((b, offset, f));
            }
            Cmd::SetViewport(x, y, w, h, near, far) => {
                gl.viewport(x as i32, y as i32, w as i32, h as i32);
                gl.depth_range_f32(near, far);
            }
            Cmd::SetScissor(x, y, w, h) => {
                gl.enable(glow::SCISSOR_TEST);
                gl.scissor(x as i32, y as i32, w as i32, h as i32);
            }
            Cmd::SetStencilReference(r) => {
                if let Pipeline::Render(p) = &st.pipeline
                    && let Some(ds) = &p.0.depth_stencil
                {
                    let s = &ds.stencil;
                    gl.stencil_func_separate(glow::FRONT, conv::compare(s.front.compare), r as i32, s.read_mask);
                    gl.stencil_func_separate(glow::BACK, conv::compare(s.back.compare), r as i32, s.read_mask);
                }
            }
            Cmd::SetBlendConstant(c) => gl.blend_color(c.r as f32, c.g as f32, c.b as f32, c.a as f32),
            Cmd::Draw(vertices, instances) => {
                let Pipeline::Render(p) = &st.pipeline else { panic!("draw without a pipeline") };
                let p = p.clone();
                prepare_draw(gl, shared, st, &p, instances.start);
                draw_arrays(gl, shared, p.0.topology, vertices.start, vertices.len() as u32, instances.len() as u32, instances.start);
            }
            Cmd::DrawIndexed(indices, base_vertex, instances) => {
                let Pipeline::Render(p) = &st.pipeline else { panic!("draw without a pipeline") };
                let p = p.clone();
                let (_, offset, f) = st.index.clone().expect("draw_indexed without an index buffer");
                let (ty, size) = conv::index_type(f);
                prepare_draw(gl, shared, st, &p, instances.start);
                let at = (offset + indices.start as u64 * size) as i32;
                if base_vertex != 0 {
                    assert!(shared.caps.base_vertex, "a base vertex needs OpenGL 3.2 / OpenGL ES 3.2");
                }
                draw_elements(gl, shared, p.0.topology, indices.len() as u32, ty, at, instances.len() as u32, base_vertex, instances.start);
            }
            Cmd::DrawIndirect(args, offset, indexed) => {
                let Pipeline::Render(p) = &st.pipeline else { panic!("draw without a pipeline") };
                let p = p.clone();
                // (without draw parameters the shader's instance index adds the first
                // instance from a uniform, which a GPU-side draw cannot set; and OpenGL ES's
                // indirect commands have no first instance at all, "reservedMustBeZero")
                if !shared.caps.indirect || !shared.caps.draw_parameters {
                    'emulated: {
                        // OpenGL ES 3.0 (ANGLE over older chips) has no indirect draws, and
                        // without draw parameters their first instance is lost: the
                        // arguments are read back and drawn directly (the GPU's earlier work on
                        // them waits; the draw is the same)
                        let len = if indexed { 20 } else { 16 };
                        let mut a = [0u32; 5];
                        gl.bind_buffer(glow::COPY_READ_BUFFER, Some(args.0.raw));
                        let ptr = gl.map_buffer_range(glow::COPY_READ_BUFFER, offset as i32, len, glow::MAP_READ_BIT);
                        if !ptr.is_null() {
                            std::ptr::copy_nonoverlapping(ptr, a.as_mut_ptr() as *mut u8, len as usize);
                            gl.unmap_buffer(glow::COPY_READ_BUFFER);
                        }
                        gl.bind_buffer(glow::COPY_READ_BUFFER, None);
                        if indexed {
                            let (count, instances, first, base, first_instance) = (a[0], a[1], a[2], a[3] as i32, a[4]);
                            let (_, ioffset, f) = st.index.clone().expect("draw_indexed_indirect without an index buffer");
                            let (ty, size) = conv::index_type(f);
                            if count == 0 || instances == 0 {
                                break 'emulated;
                            }
                            if base != 0 && !shared.caps.base_vertex {
                                log::warn!("mX: an indirect draw with a base vertex left out (OpenGL ES 3.0 has none)");
                                break 'emulated;
                            }
                            prepare_draw(gl, shared, st, &p, first_instance);
                            let at = (ioffset + first as u64 * size) as i32;
                            draw_elements(gl, shared, p.0.topology, count, ty, at, instances, base, first_instance);
                        } else {
                            let (count, instances, first, first_instance) = (a[0], a[1], a[2], a[3]);
                            if count == 0 || instances == 0 {
                                break 'emulated;
                            }
                            prepare_draw(gl, shared, st, &p, first_instance);
                            draw_arrays(gl, shared, p.0.topology, first, count, instances, first_instance);
                        }
                    }
                } else {
                    prepare_draw(gl, shared, st, &p, 0);
                    gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(args.0.raw));
                    if indexed {
                        let (_, _, f) = st.index.clone().expect("draw_indexed_indirect without an index buffer");
                        gl.draw_elements_indirect_offset(p.0.topology, conv::index_type(f).0, offset as i32);
                    } else {
                        gl.draw_arrays_indirect_offset(p.0.topology, offset as i32);
                    }
                    gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, None);
                }
            }
            Cmd::Dispatch(x, y, z) => {
                let Pipeline::Compute(p) = &st.pipeline else { panic!("dispatch without a pipeline") };
                let p = p.clone();
                barrier_if_written(gl, shared, st);
                bind(gl, shared, st, &p.0.bindings, &p.0.label, 0);
                if x > 0 && y > 0 && z > 0 {
                    gl.dispatch_compute(x, y, z);
                }
                st.storage_written |= p.0.bindings.writes;
            }
            Cmd::DispatchIndirect(args, offset) => {
                let Pipeline::Compute(p) = &st.pipeline else { panic!("dispatch without a pipeline") };
                let p = p.clone();
                barrier_if_written(gl, shared, st);
                bind(gl, shared, st, &p.0.bindings, &p.0.label, 0);
                gl.bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, Some(args.0.raw));
                gl.dispatch_compute_indirect(offset as i32);
                gl.bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, None);
                st.storage_written |= p.0.bindings.writes;
            }
            Cmd::BeginOcclusionQuery(q, i) => q.begin_occlusion(gl, i),
            Cmd::EndOcclusionQuery(q) => q.end_occlusion(gl),
            Cmd::WriteTimestamp(q, i) => q.timestamp(gl, i),
            Cmd::ResolveQuerySet(q, range, dst, offset) => {
                let values = q.results(gl, range);
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(dst.0.raw));
                gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset as i32, &bytes);
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            }
            Cmd::PushDebugGroup(l) => push_group(gl, &l),
            Cmd::PopDebugGroup => {
                if gl.supports_debug() {
                    gl.pop_debug_group();
                }
            }
            Cmd::InsertDebugMarker(l) => {
                if gl.supports_debug() {
                    gl.debug_message_insert(glow::DEBUG_SOURCE_APPLICATION, glow::DEBUG_TYPE_MARKER, 0, glow::DEBUG_SEVERITY_NOTIFICATION, &l);
                }
            }
            Cmd::ExecuteBundle(b) => {
                for c in b.0.cmds.iter().cloned() {
                    run(gl, shared, st, c);
                }
                // (a bundle's state does not leak into the pass)
                st.pipeline = Pipeline::None;
                st.groups = Default::default();
                st.offsets.clear();
                st.vertex = Default::default();
                st.index = None;
                st.bindings_dirty = true;
            }
        }
    }
}

/// The end of a pass: its end timestamp, its debug group, no pipeline.
fn end_scope(gl: &glow::Context, st: &mut State) {
    if let Some((q, i)) = st.pass_end.take() {
        q.timestamp(gl, i);
    }
    if std::mem::take(&mut st.pass_labelled) {
        // SAFETY: closes the group the pass opened.
        unsafe {
            if gl.supports_debug() {
                gl.pop_debug_group();
            }
        }
    }
    st.pipeline = Pipeline::None;
}

fn push_group(gl: &glow::Context, label: &str) {
    // SAFETY: a debug group on the current context.
    unsafe {
        if gl.supports_debug() {
            gl.push_debug_group(glow::DEBUG_SOURCE_APPLICATION, 0, label);
        }
    }
}

/// The fixed-function state of a render pipeline.
///
/// # Safety
/// The context is current.
unsafe fn apply_render_state(gl: &glow::Context, shared: &Shared, p: &RenderPipeline) {
    let p = &p.0;
    // SAFETY: the caller's.
    unsafe {
        match p.primitive.cull_mode {
            None => gl.disable(glow::CULL_FACE),
            Some(f) => {
                gl.enable(glow::CULL_FACE);
                gl.cull_face(if f == t::Face::Front { glow::FRONT } else { glow::BACK });
            }
        }
        gl.front_face(if p.primitive.front_face == t::FrontFace::Ccw { glow::CCW } else { glow::CW });
        if shared.ctx.api == crate::context::Api::Desktop {
            if p.primitive.unclipped_depth {
                gl.enable(glow::DEPTH_CLAMP);
            } else {
                gl.disable(glow::DEPTH_CLAMP);
            }
        }
        match &p.depth_stencil {
            Some(ds) => {
                gl.enable(glow::DEPTH_TEST);
                gl.depth_func(conv::compare(ds.depth_compare));
                gl.depth_mask(ds.depth_write_enabled);
                let b = ds.bias;
                if b.constant != 0 || b.slope_scale != 0.0 {
                    gl.enable(glow::POLYGON_OFFSET_FILL);
                    gl.polygon_offset(b.slope_scale, b.constant as f32);
                } else {
                    gl.disable(glow::POLYGON_OFFSET_FILL);
                }
                let s = &ds.stencil;
                let stencil_on = |f: &t::StencilFaceState| {
                    f.compare != t::CompareFunction::Always || f.fail_op != t::StencilOperation::Keep || f.depth_fail_op != t::StencilOperation::Keep || f.pass_op != t::StencilOperation::Keep
                };
                if ds.format.has_stencil() && (stencil_on(&s.front) || stencil_on(&s.back)) {
                    gl.enable(glow::STENCIL_TEST);
                    for (face, f) in [(glow::FRONT, &s.front), (glow::BACK, &s.back)] {
                        gl.stencil_func_separate(face, conv::compare(f.compare), 0, s.read_mask);
                        gl.stencil_op_separate(face, conv::stencil_op(f.fail_op), conv::stencil_op(f.depth_fail_op), conv::stencil_op(f.pass_op));
                        gl.stencil_mask_separate(face, s.write_mask);
                    }
                } else {
                    gl.disable(glow::STENCIL_TEST);
                }
            }
            None => {
                gl.disable(glow::DEPTH_TEST);
                gl.disable(glow::STENCIL_TEST);
                gl.disable(glow::POLYGON_OFFSET_FILL);
            }
        }
        if p.multisample.alpha_to_coverage_enabled {
            gl.enable(glow::SAMPLE_ALPHA_TO_COVERAGE);
        } else {
            gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
        }
        // (a sample mask other than all samples is not applied: glow has no glSampleMaski;
        // openOMSI never sets one)
        if p.multisample.count > 1 && p.multisample.mask as u32 != u32::MAX {
            log::warn!("{}: multisample masks are not applied on OpenGL", p.label);
        }
        for (i, target) in p.targets.iter().enumerate() {
            let i = i as u32;
            let m = target.mask;
            gl.color_mask_draw_buffer(
                i,
                m.contains(t::ColorWrites::RED),
                m.contains(t::ColorWrites::GREEN),
                m.contains(t::ColorWrites::BLUE),
                m.contains(t::ColorWrites::ALPHA),
            );
            match target.blend {
                Some((color, alpha, ops)) => {
                    gl.enable_draw_buffer(glow::BLEND, i);
                    gl.blend_func_separate_draw_buffer(i, color[0], color[1], alpha[0], alpha[1]);
                    gl.blend_equation_separate_draw_buffer(i, ops[0], ops[1]);
                }
                None => gl.disable_draw_buffer(glow::BLEND, i),
            }
        }
    }
}

/// Bind the root bindings, heap slots and immediates a pipeline reads.
///
/// # Safety
/// The context is current and the pipeline's program in use.
unsafe fn bind(gl: &glow::Context, shared: &Shared, st: &mut State, b: &Bindings, label: &str, first_instance: u32) {
    // SAFETY: the caller's.
    unsafe {
        if st.bindings_dirty {
            let mut samplers: [Option<glow::Sampler>; 256] = [None; 256];
            for r in &b.roots {
                let Some(group) = st.groups[r.group as usize].as_ref() else {
                    panic!("{label}: draw with nothing bound at @group({})", r.group);
                };
                let Some(res) = group.get(r.binding) else {
                    panic!("{label}: draw with nothing bound at @group({}) @binding({})", r.group, r.binding);
                };
                match (r.class, res) {
                    (Class::UniformBuffer, Bound::Buffer { buffer, offset, size }) => {
                        let extra = st.offsets.iter().find(|o| o.0 == r.group && o.1 == r.binding).map_or(0, |o| o.2);
                        gl.bind_buffer_range(glow::UNIFORM_BUFFER, r.gl as u32, Some(buffer.0.raw), (*offset + extra) as i32, *size as i32)
                    }
                    (Class::StorageBuffer, Bound::Buffer { buffer, offset, size }) => gl.bind_buffer_range(glow::SHADER_STORAGE_BUFFER, r.gl as u32, Some(buffer.0.raw), *offset as i32, *size as i32),
                    (Class::Texture, Bound::Texture(v)) => bind_texture(gl, r.gl, v.0.binding()),
                    (Class::Image, Bound::Texture(v)) => {
                        let f = conv::format(v.format());
                        let layered = v.0.array_layer_count > 1 || v.texture().dimension() == t::TextureDimension::D3;
                        gl.bind_image_texture(
                            r.gl as u32,
                            Some(v.texture().0.raw),
                            v.0.base_mip_level as i32,
                            layered,
                            v.0.base_array_layer as i32,
                            glow::READ_WRITE,
                            f.internal,
                        );
                    }
                    (Class::Sampler, Bound::Sampler(s)) => samplers[r.gl as usize] = Some(s.0.raw),
                    _ => panic!("{label}: the resource at @group({}) @binding({}) is not the kind the shader declares", r.group, r.binding),
                }
            }
            if !b.slots.is_empty() {
                let heap = shared.heap.lock();
                for s in &b.slots {
                    let index = match s.index {
                        Index::Constant(c) => c,
                        Index::Immediate(o) => {
                            let o = o as usize;
                            u32::from_le_bytes([st.immediates[o], st.immediates[o + 1], st.immediates[o + 2], st.immediates[o + 3]])
                        }
                    };
                    match s.class {
                        Class::Texture => match heap.texture(index) {
                            Some(t) => bind_texture(gl, s.gl, t),
                            None => log::warn!("{label}: heap texture {index} is empty"),
                        },
                        Class::Sampler => samplers[s.gl as usize] = heap.sampler(index),
                        Class::Image => match heap.texture(index) {
                            Some(crate::heap::TextureSlot {
                                raw,
                                base_mip,
                                image: Some((internal, layer)),
                                ..
                            }) => gl.bind_image_texture(s.gl as u32, Some(raw), base_mip as i32, false, layer as i32, glow::WRITE_ONLY, internal),
                            _ => log::warn!("{label}: heap index {index} is not a storage texture"),
                        },
                        Class::StorageBuffer => match heap.buffer(index) {
                            Some(buf) => gl.bind_buffer_range(glow::SHADER_STORAGE_BUFFER, s.gl as u32, Some(buf.raw), buf.offset as i32, buf.size as i32),
                            None => log::warn!("{label}: heap buffer {index} is empty"),
                        },
                        // (a read-only view read as a uniform block: `bound::views_as_uniforms`)
                        Class::UniformBuffer => match heap.buffer(index) {
                            Some(buf) => gl.bind_buffer_range(glow::UNIFORM_BUFFER, s.gl as u32, Some(buf.raw), buf.offset as i32, buf.size as i32),
                            None => log::warn!("{label}: heap buffer {index} is empty"),
                        },
                    }
                }
            }
            for &(unit, sampler) in &b.texture_samplers {
                gl.bind_sampler(unit as u32, sampler.and_then(|s| samplers[s as usize]));
            }
            st.bindings_dirty = false;
        }
        if st.immediates_dirty {
            for u in &b.immediates {
                crate::pipeline::set_immediate(gl, u, &st.immediates);
            }
            st.immediates_dirty = false;
        }
        if let Some(loc) = &b.first_instance {
            gl.uniform_1_u32(Some(loc), first_instance);
        }
    }
}

/// Bind a texture (view) to a unit: the texture whole, with the view's mips.
///
/// # Safety
/// The context is current.
unsafe fn bind_texture(gl: &glow::Context, unit: u8, t: crate::heap::TextureSlot) {
    // SAFETY: the caller's; the texture is alive (the heap or a bind group holds its view).
    unsafe {
        gl.active_texture(glow::TEXTURE0 + unit as u32);
        gl.bind_texture(t.target, Some(t.raw));
        if t.target != glow::TEXTURE_2D_MULTISAMPLE {
            gl.tex_parameter_i32(t.target, glow::TEXTURE_BASE_LEVEL, t.base_mip as i32);
            gl.tex_parameter_i32(t.target, glow::TEXTURE_MAX_LEVEL, (t.base_mip + t.mip_count) as i32 - 1);
        }
        if let Some(mode) = t.depth_stencil_mode {
            gl.tex_parameter_i32(t.target, glow::DEPTH_STENCIL_TEXTURE_MODE, mode as i32);
        }
    }
}

/// Everything a draw needs: bindings, immediates, vertex attributes.
///
/// # Safety
/// The context is current and the pipeline's program in use.
unsafe fn prepare_draw(gl: &glow::Context, shared: &Shared, st: &mut State, p: &RenderPipeline, first_instance: u32) {
    // SAFETY: the caller's.
    unsafe {
        bind(gl, shared, st, &p.0.bindings, &p.0.label, first_instance);
        let mut enabled = 0u32;
        for (i, vb) in p.0.vertex_buffers.iter().enumerate() {
            let Some((buffer, offset)) = &st.vertex[i] else {
                if !vb.attributes.is_empty() {
                    panic!("{}: draw without a vertex buffer in slot {i}", p.0.label);
                }
                continue;
            };
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer.0.raw));
            // (without `gl_BaseInstance`, instance-rate attributes start at the first instance
            // by offset, as wgpu-hal does)
            let base = *offset as i32 + if vb.instance && !shared.caps.draw_parameters { first_instance as i32 * vb.stride } else { 0 };
            for &(loc, a, at) in &vb.attributes {
                gl.enable_vertex_attrib_array(loc);
                enabled |= 1 << loc;
                if a.integer {
                    gl.vertex_attrib_pointer_i32(loc, a.size, a.ty, vb.stride, base + at);
                } else {
                    gl.vertex_attrib_pointer_f32(loc, a.size, a.ty, a.normalized, vb.stride, base + at);
                }
                gl.vertex_attrib_divisor(loc, vb.instance as u32);
            }
        }
        for loc in 0..32 {
            if st.enabled_attribs & (1 << loc) != 0 && enabled & (1 << loc) == 0 {
                gl.disable_vertex_attrib_array(loc);
            }
        }
        st.enabled_attribs = enabled;
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        st.storage_written |= p.0.bindings.writes;
    }
}

/// Attach a view to the bound draw framebuffer.
///
/// # Safety
/// The context is current with a framebuffer bound to `target`.
unsafe fn attach(gl: &glow::Context, target: u32, attachment: u32, v: &TextureView) {
    let tex = v.texture();
    // SAFETY: the caller's.
    unsafe {
        if let Some(rb) = tex.0.rb {
            gl.framebuffer_renderbuffer(target, attachment, glow::RENDERBUFFER, Some(rb));
            return;
        }
        match tex.0.target {
            glow::TEXTURE_2D | glow::TEXTURE_2D_MULTISAMPLE => gl.framebuffer_texture_2d(target, attachment, tex.0.target, Some(tex.0.raw), v.0.base_mip_level as i32),
            glow::TEXTURE_CUBE_MAP => gl.framebuffer_texture_2d(target, attachment, glow::TEXTURE_CUBE_MAP_POSITIVE_X + v.0.base_array_layer, Some(tex.0.raw), v.0.base_mip_level as i32),
            _ => gl.framebuffer_texture_layer(target, attachment, Some(tex.0.raw), v.0.base_mip_level as i32, v.0.base_array_layer as i32),
        }
    }
}

fn depth_attachment(f: t::TextureFormat) -> u32 {
    if f.has_stencil() { glow::DEPTH_STENCIL_ATTACHMENT } else { glow::DEPTH_ATTACHMENT }
}

fn begin_pass(gl: &glow::Context, shared: &Shared, st: &mut State, colors: &[Option<ColorAttachment>], depth: Option<&DepthAttachment>) {
    // SAFETY: GL calls on the current context with live views.
    unsafe {
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(shared.fbo_draw));
        let mut size = (0, 0);
        let mut draw_buffers = Vec::new();
        for (i, c) in colors.iter().enumerate() {
            let at = glow::COLOR_ATTACHMENT0 + i as u32;
            match c {
                Some(c) => {
                    attach(gl, glow::DRAW_FRAMEBUFFER, at, &c.view);
                    let s = c.view.texture().size().mip_level_size(c.view.0.base_mip_level, c.view.texture().dimension());
                    size = (s.width, s.height);
                    draw_buffers.push(at);
                }
                None => {
                    gl.framebuffer_texture_2d(glow::DRAW_FRAMEBUFFER, at, glow::TEXTURE_2D, None, 0);
                    draw_buffers.push(glow::NONE);
                }
            }
        }
        for i in colors.len()..shared.limits.max_color_attachments as usize {
            gl.framebuffer_texture_2d(glow::DRAW_FRAMEBUFFER, glow::COLOR_ATTACHMENT0 + i as u32, glow::TEXTURE_2D, None, 0);
        }
        gl.framebuffer_texture_2d(glow::DRAW_FRAMEBUFFER, glow::DEPTH_STENCIL_ATTACHMENT, glow::TEXTURE_2D, None, 0);
        if let Some(d) = depth {
            attach(gl, glow::DRAW_FRAMEBUFFER, depth_attachment(d.view.format()), &d.view);
            let s = d.view.texture().size().mip_level_size(d.view.0.base_mip_level, d.view.texture().dimension());
            size = (s.width, s.height);
        }
        gl.draw_buffers(&draw_buffers);
        st.target_size = size;
        gl.disable(glow::SCISSOR_TEST);
        gl.viewport(0, 0, size.0 as i32, size.1 as i32);
        gl.depth_range_f32(0.0, 1.0);
        // clears, with every write enabled whatever the last pipeline left
        for (i, c) in colors.iter().enumerate() {
            let Some(c) = c else { continue };
            let Some(v) = c.clear else { continue };
            gl.color_mask_draw_buffer(i as u32, true, true, true, true);
            match c.view.format().sample_type() {
                t::TextureSampleType::Uint => gl.clear_buffer_u32_slice(glow::COLOR, i as u32, &[v.r as u32, v.g as u32, v.b as u32, v.a as u32]),
                t::TextureSampleType::Sint => gl.clear_buffer_i32_slice(glow::COLOR, i as u32, &[v.r as i32, v.g as i32, v.b as i32, v.a as i32]),
                _ => gl.clear_buffer_f32_slice(glow::COLOR, i as u32, &[v.r as f32, v.g as f32, v.b as f32, v.a as f32]),
            }
        }
        if let Some(d) = depth {
            let depth_clear = d.depth.flatten();
            let stencil_clear = d.stencil.flatten();
            if depth_clear.is_some() {
                gl.depth_mask(true);
            }
            if stencil_clear.is_some() {
                gl.stencil_mask_separate(glow::FRONT_AND_BACK, !0);
            }
            match (depth_clear, stencil_clear) {
                (Some(z), Some(s)) => gl.clear_buffer_depth_stencil(glow::DEPTH_STENCIL, 0, z, s as i32),
                (Some(z), None) => gl.clear_buffer_f32_slice(glow::DEPTH, 0, &[z]),
                (None, Some(s)) => gl.clear_buffer_i32_slice(glow::STENCIL, 0, &[s as i32]),
                (None, None) => {}
            }
        }
    }
}

fn end_pass(gl: &glow::Context, shared: &Shared, st: &mut State) {
    // SAFETY: GL calls on the current context with live views.
    unsafe {
        // MSAA resolves: a blit per resolved attachment
        let colors = std::mem::take(&mut st.pass_colors);
        for c in colors.iter().flatten() {
            let Some(r) = &c.resolve else { continue };
            let (w, h) = (st.target_size.0 as i32, st.target_size.1 as i32);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(shared.fbo_read));
            attach(gl, glow::READ_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, &c.view);
            gl.read_buffer(glow::COLOR_ATTACHMENT0);
            let resolve_fbo = gl.create_framebuffer().ok();
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, resolve_fbo);
            attach(gl, glow::DRAW_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, r);
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            gl.disable(glow::SCISSOR_TEST);
            gl.blit_framebuffer(0, 0, w, h, 0, 0, w, h, glow::COLOR_BUFFER_BIT, glow::NEAREST);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(shared.fbo_draw));
            if let Some(f) = resolve_fbo {
                gl.delete_framebuffer(f);
            }
        }
        if shared.caps.invalidate {
            let mut discard: Vec<u32> = colors
                .iter()
                .enumerate()
                .filter(|(_, c)| c.as_ref().is_some_and(|c| !c.store))
                .map(|(i, _)| glow::COLOR_ATTACHMENT0 + i as u32)
                .collect();
            if let Some(d) = &st.pass_depth {
                match (d.depth.is_some() && !d.depth_store, d.stencil.is_some() && !d.stencil_store) {
                    (true, true) => discard.push(depth_attachment(d.view.format())),
                    (true, false) => discard.push(glow::DEPTH_ATTACHMENT),
                    (false, true) => discard.push(glow::STENCIL_ATTACHMENT),
                    (false, false) => {}
                }
            }
            if !discard.is_empty() {
                gl.invalidate_framebuffer(glow::DRAW_FRAMEBUFFER, &discard);
            }
        }
        st.pass_depth = None;
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
    }
    barrier_if_written(gl, shared, st);
}

// ---------------------------------------------------------------------------------------------
// Uploads, read-backs, copies, zeroing

/// Where upload bytes come from: memory, or the bound `PIXEL_UNPACK_BUFFER` at an offset.
pub(crate) enum Source<'a> {
    Bytes(&'a [u8]),
    Buffer(u64),
}

/// The GL image target of `layer` of a texture (cube faces are their own targets).
fn image_target(tex: &Texture, layer: u32) -> u32 {
    if tex.0.target == glow::TEXTURE_CUBE_MAP {
        glow::TEXTURE_CUBE_MAP_POSITIVE_X + layer
    } else {
        tex.0.target
    }
}

/// Copy texels into a texture: `size` texels at `origin` of `mip`, rows `layout.bytes_per_row`
/// apart and images `layout.rows_per_image` rows apart.
pub(crate) fn upload(gl: &glow::Context, tex: &Texture, mip: u32, origin: t::Origin3d, size: t::Extent3d, layout: t::TexelCopyBufferLayout, src: Source<'_>) {
    let f = tex.format();
    let desc = conv::format(f);
    let (bw, bh) = f.block_dimensions();
    let block = f.block_copy_size().unwrap_or(4);
    let row_bytes = layout.bytes_per_row.unwrap_or_else(|| size.width.div_ceil(bw) * block);
    let rows = layout.rows_per_image.unwrap_or(size.height.div_ceil(bh));
    let image_bytes = row_bytes as u64 * rows as u64;
    let target = tex.0.target;
    // SAFETY: the texture is alive; the region is inside it and the source holds the bytes
    // (both checked by the caller's contract, as on every backend).
    unsafe {
        gl.bind_texture(target, Some(tex.0.raw));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        if !f.is_compressed() {
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, (row_bytes / block * bw) as i32);
            gl.pixel_store_i32(glow::UNPACK_IMAGE_HEIGHT, rows as i32);
        }
        let layered = matches!(target, glow::TEXTURE_2D_ARRAY | glow::TEXTURE_3D | glow::TEXTURE_CUBE_MAP_ARRAY);
        let depth = size.depth_or_array_layers.max(1);
        let (x, y, mut w, mut h) = (origin.x as i32, origin.y as i32, size.width as i32, size.height as i32);
        if f.is_compressed() {
            // (WebGPU's size is whole blocks, past the edge of a mip smaller than a block;
            // GL takes the mip's real texels there and refuses more)
            w = w.min(((tex.width() >> mip).max(1) - origin.x) as i32);
            h = h.min(((tex.height() >> mip).max(1) - origin.y) as i32);
        }
        if layered || target != glow::TEXTURE_CUBE_MAP {
            let whole = image_bytes * if layered { depth as u64 } else { 1 };
            match (&src, f.is_compressed(), layered) {
                (Source::Bytes(b), false, false) => gl.tex_sub_image_2d(target, mip as i32, x, y, w, h, desc.external, desc.data_type, glow::PixelUnpackData::Slice(Some(b))),
                (Source::Buffer(o), false, false) => gl.tex_sub_image_2d(target, mip as i32, x, y, w, h, desc.external, desc.data_type, glow::PixelUnpackData::BufferOffset(*o as u32)),
                (Source::Bytes(b), false, true) => gl.tex_sub_image_3d(
                    target,
                    mip as i32,
                    x,
                    y,
                    origin.z as i32,
                    w,
                    h,
                    depth as i32,
                    desc.external,
                    desc.data_type,
                    glow::PixelUnpackData::Slice(Some(b)),
                ),
                (Source::Buffer(o), false, true) => gl.tex_sub_image_3d(
                    target,
                    mip as i32,
                    x,
                    y,
                    origin.z as i32,
                    w,
                    h,
                    depth as i32,
                    desc.external,
                    desc.data_type,
                    glow::PixelUnpackData::BufferOffset(*o as u32),
                ),
                (Source::Bytes(b), true, false) => gl.compressed_tex_sub_image_2d(target, mip as i32, x, y, w, h, desc.internal, glow::CompressedPixelUnpackData::Slice(&b[..whole as usize])),
                (Source::Buffer(o), true, false) => gl.compressed_tex_sub_image_2d(
                    target,
                    mip as i32,
                    x,
                    y,
                    w,
                    h,
                    desc.internal,
                    glow::CompressedPixelUnpackData::BufferRange(*o as u32..(*o + whole) as u32),
                ),
                (Source::Bytes(b), true, true) => gl.compressed_tex_sub_image_3d(
                    target,
                    mip as i32,
                    x,
                    y,
                    origin.z as i32,
                    w,
                    h,
                    depth as i32,
                    desc.internal,
                    glow::CompressedPixelUnpackData::Slice(&b[..whole as usize]),
                ),
                (Source::Buffer(o), true, true) => gl.compressed_tex_sub_image_3d(
                    target,
                    mip as i32,
                    x,
                    y,
                    origin.z as i32,
                    w,
                    h,
                    depth as i32,
                    desc.internal,
                    glow::CompressedPixelUnpackData::BufferRange(*o as u32..(*o + whole) as u32),
                ),
            }
        } else {
            // a cube map: one image per face
            for i in 0..depth {
                let face = image_target(tex, origin.z + i);
                let at = i as u64 * image_bytes;
                match (&src, f.is_compressed()) {
                    (Source::Bytes(b), false) => gl.tex_sub_image_2d(face, mip as i32, x, y, w, h, desc.external, desc.data_type, glow::PixelUnpackData::Slice(Some(&b[at as usize..]))),
                    (Source::Buffer(o), false) => gl.tex_sub_image_2d(face, mip as i32, x, y, w, h, desc.external, desc.data_type, glow::PixelUnpackData::BufferOffset((*o + at) as u32)),
                    (Source::Bytes(b), true) => gl.compressed_tex_sub_image_2d(
                        face,
                        mip as i32,
                        x,
                        y,
                        w,
                        h,
                        desc.internal,
                        glow::CompressedPixelUnpackData::Slice(&b[at as usize..(at + image_bytes) as usize]),
                    ),
                    (Source::Buffer(o), true) => gl.compressed_tex_sub_image_2d(
                        face,
                        mip as i32,
                        x,
                        y,
                        w,
                        h,
                        desc.internal,
                        glow::CompressedPixelUnpackData::BufferRange((*o + at) as u32..(*o + at + image_bytes) as u32),
                    ),
                }
            }
        }
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
        gl.pixel_store_i32(glow::UNPACK_IMAGE_HEIGHT, 0);
        gl.bind_texture(target, None);
    }
}

/// Read texels of one mip into a buffer, a layer at a time through the read framebuffer.
#[allow(clippy::too_many_arguments)]
fn read_back(gl: &glow::Context, shared: &Shared, tex: &Texture, mip: u32, origin: t::Origin3d, dst: &Buffer, layout: t::TexelCopyBufferLayout, size: t::Extent3d) {
    let f = tex.format();
    if f.is_compressed() {
        read_back_compressed(gl, shared, tex, mip, origin, dst, layout, size);
        return;
    }
    let desc = conv::format(f);
    let block = f.block_copy_size().unwrap_or(4);
    let row_bytes = layout.bytes_per_row.unwrap_or(size.width * block);
    let rows = layout.rows_per_image.unwrap_or(size.height);
    // SAFETY: GL calls on the current context with live objects; regions checked by the caller.
    unsafe {
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(shared.fbo_read));
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(dst.0.raw));
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
        gl.pixel_store_i32(glow::PACK_ROW_LENGTH, (row_bytes / block) as i32);
        let attachment = if f.has_depth() { depth_attachment(f) } else { glow::COLOR_ATTACHMENT0 };
        for layer in 0..size.depth_or_array_layers.max(1) {
            let z = origin.z + layer;
            match tex.0.target {
                glow::TEXTURE_2D => gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, attachment, glow::TEXTURE_2D, Some(tex.0.raw), mip as i32),
                glow::TEXTURE_CUBE_MAP => gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, attachment, image_target(tex, z), Some(tex.0.raw), mip as i32),
                _ => gl.framebuffer_texture_layer(glow::READ_FRAMEBUFFER, attachment, Some(tex.0.raw), mip as i32, z as i32),
            }
            if !f.has_depth() {
                gl.read_buffer(glow::COLOR_ATTACHMENT0);
            }
            let at = layout.offset + layer as u64 * row_bytes as u64 * rows as u64;
            gl.read_pixels(
                origin.x as i32,
                origin.y as i32,
                size.width as i32,
                size.height as i32,
                desc.external,
                desc.data_type,
                glow::PixelPackData::BufferOffset(at as u32),
            );
            gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, attachment, glow::TEXTURE_2D, None, 0);
        }
        gl.pixel_store_i32(glow::PACK_ROW_LENGTH, 0);
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
    }
}

/// A non-indexed draw from instance `first_instance`. With draw parameters the shader's
/// instance index adds `gl_BaseInstance`, which only a base-instance draw sets (GL 4.2);
/// without, `prepare_draw` has put the first instance in a uniform.
///
/// # Safety
/// The context is current, the draw's state bound.
#[allow(clippy::too_many_arguments)]
unsafe fn draw_arrays(gl: &glow::Context, shared: &Shared, topology: u32, first: u32, count: u32, instances: u32, first_instance: u32) {
    // SAFETY: the caller's.
    unsafe {
        if shared.caps.draw_parameters && first_instance != 0 {
            gl.draw_arrays_instanced_base_instance(topology, first as i32, count as i32, instances as i32, first_instance);
        } else {
            gl.draw_arrays_instanced(topology, first as i32, count as i32, instances as i32);
        }
    }
}

/// An indexed draw from `at` (bytes into the index buffer), as [`draw_arrays`].
///
/// # Safety
/// As [`draw_arrays`]; a non-zero `base_vertex` needs GL 3.2 / ES 3.2.
#[allow(clippy::too_many_arguments)]
unsafe fn draw_elements(gl: &glow::Context, shared: &Shared, topology: u32, count: u32, ty: u32, at: i32, instances: u32, base_vertex: i32, first_instance: u32) {
    // SAFETY: the caller's.
    unsafe {
        if shared.caps.draw_parameters && first_instance != 0 {
            gl.draw_elements_instanced_base_vertex_base_instance(topology, count as i32, ty, at, instances as i32, base_vertex, first_instance);
        } else if base_vertex != 0 {
            gl.draw_elements_instanced_base_vertex(topology, count as i32, ty, at, instances as i32, base_vertex);
        } else {
            gl.draw_elements_instanced(topology, count as i32, ty, at, instances as i32);
        }
    }
}

/// Read blocks of a compressed texture's mip into a buffer. GL has no framebuffer of a
/// compressed format (and ES no `glGetCompressedTexImage`): each layer's blocks are copied
/// with `glCopyImageSubData` (GL 4.3, ES 3.2) into an unsigned integer texture of the
/// block's size, one texel a block (`RG32UI` for 8 bytes, `RGBA32UI` for 16), read back
/// through the read framebuffer and written into the buffer row by row.
#[allow(clippy::too_many_arguments)]
fn read_back_compressed(gl: &glow::Context, shared: &Shared, tex: &Texture, mip: u32, origin: t::Origin3d, dst: &Buffer, layout: t::TexelCopyBufferLayout, size: t::Extent3d) {
    let f = tex.format();
    assert!(
        shared.caps.copy_image,
        "copy_texture_to_buffer of compressed textures needs OpenGL 4.3 or OpenGL ES 3.2 (glCopyImageSubData)"
    );
    let block = f.block_copy_size().unwrap_or(16);
    let (bw, bh) = f.block_dimensions();
    let (blocks_x, blocks_y) = (size.width.div_ceil(bw), size.height.div_ceil(bh));
    let row_bytes = layout.bytes_per_row.unwrap_or(blocks_x * block) as usize;
    let block_rows = layout.rows_per_image.map_or(blocks_y, |r| r.div_ceil(bh)) as usize;
    // (the copy covers the mip's real texels: WebGPU's copy size is whole blocks, past the
    // edge of a mip smaller than a block, which GL refuses)
    let mip_w = (tex.width() >> mip).max(1);
    let mip_h = (tex.size().height >> mip).max(1);
    let (w, h) = (size.width.min(mip_w - origin.x), size.height.min(mip_h - origin.y));
    let internal = if block == 8 { glow::RG32UI } else { glow::RGBA32UI };
    let per_block = (block / 4) as usize;
    // SAFETY: GL calls on the current context with live objects; a scratch texture made and
    // deleted here; regions checked by the caller.
    unsafe {
        let Ok(scratch) = gl.create_texture() else {
            log::error!("mX: no texture to read compressed blocks through");
            return;
        };
        gl.bind_texture(glow::TEXTURE_2D, Some(scratch));
        gl.tex_storage_2d(glow::TEXTURE_2D, 1, internal, blocks_x as i32, blocks_y as i32);
        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(shared.fbo_read));
        gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(scratch), 0);
        gl.read_buffer(glow::COLOR_ATTACHMENT0);
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 4);
        gl.pixel_store_i32(glow::PACK_ROW_LENGTH, 0);
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(dst.0.raw));
        let mut texels = vec![0u32; (blocks_x * blocks_y * 4) as usize];
        for layer in 0..size.depth_or_array_layers.max(1) {
            gl.copy_image_sub_data(
                tex.0.raw,
                tex.0.target,
                mip as i32,
                origin.x as i32,
                origin.y as i32,
                (origin.z + layer) as i32,
                scratch,
                glow::TEXTURE_2D,
                0,
                0,
                0,
                0,
                w as i32,
                h as i32,
                1,
            );
            // (RGBA_INTEGER / UNSIGNED_INT: the read every GL takes for an unsigned integer
            // framebuffer)
            gl.read_pixels(
                0,
                0,
                blocks_x as i32,
                blocks_y as i32,
                glow::RGBA_INTEGER,
                glow::UNSIGNED_INT,
                glow::PixelPackData::Slice(Some(bytemuck_u8(&mut texels))),
            );
            let base = layout.offset as usize + layer as usize * row_bytes * block_rows;
            for y in 0..blocks_y as usize {
                let mut row = Vec::with_capacity(blocks_x as usize * block as usize);
                for x in 0..blocks_x as usize {
                    let t = &texels[(y * blocks_x as usize + x) * 4..][..per_block];
                    for v in t {
                        row.extend_from_slice(&v.to_le_bytes());
                    }
                }
                gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, (base + y * row_bytes) as i32, &row);
            }
        }
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, None, 0);
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        gl.delete_texture(scratch);
    }
}

/// `u32`s as their bytes.
fn bytemuck_u8(v: &mut [u32]) -> &mut [u8] {
    // SAFETY: u32 has no padding and u8 no alignment needs; the length is the same bytes.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

#[allow(clippy::too_many_arguments)]
fn copy_texture(gl: &glow::Context, shared: &Shared, src: &Texture, smip: u32, so: t::Origin3d, dst: &Texture, dmip: u32, d_o: t::Origin3d, size: t::Extent3d) {
    // SAFETY: GL calls on the current context with live textures; regions checked by the caller.
    unsafe {
        if shared.caps.copy_image {
            gl.copy_image_sub_data(
                src.0.raw,
                src.0.target,
                smip as i32,
                so.x as i32,
                so.y as i32,
                so.z as i32,
                dst.0.raw,
                dst.0.target,
                dmip as i32,
                d_o.x as i32,
                d_o.y as i32,
                d_o.z as i32,
                size.width as i32,
                size.height as i32,
                size.depth_or_array_layers.max(1) as i32,
            );
            return;
        }
        // without glCopyImageSubData: a blit per layer
        let f = src.format();
        let (attachment, mask) = if f.has_depth() {
            (depth_attachment(f), glow::DEPTH_BUFFER_BIT | if f.has_stencil() { glow::STENCIL_BUFFER_BIT } else { 0 })
        } else {
            (glow::COLOR_ATTACHMENT0, glow::COLOR_BUFFER_BIT)
        };
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(shared.fbo_read));
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(shared.fbo_draw));
        for i in 0..size.depth_or_array_layers.max(1) {
            for (target, tex, mip, z) in [(glow::READ_FRAMEBUFFER, src, smip, so.z + i), (glow::DRAW_FRAMEBUFFER, dst, dmip, d_o.z + i)] {
                match tex.0.target {
                    glow::TEXTURE_2D => gl.framebuffer_texture_2d(target, attachment, glow::TEXTURE_2D, Some(tex.0.raw), mip as i32),
                    glow::TEXTURE_CUBE_MAP => gl.framebuffer_texture_2d(target, attachment, image_target(tex, z), Some(tex.0.raw), mip as i32),
                    _ => gl.framebuffer_texture_layer(target, attachment, Some(tex.0.raw), mip as i32, z as i32),
                }
            }
            if !f.has_depth() {
                gl.read_buffer(glow::COLOR_ATTACHMENT0);
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            }
            gl.disable(glow::SCISSOR_TEST);
            let (w, h) = (size.width as i32, size.height as i32);
            gl.blit_framebuffer(
                so.x as i32,
                so.y as i32,
                so.x as i32 + w,
                so.y as i32 + h,
                d_o.x as i32,
                d_o.y as i32,
                d_o.x as i32 + w,
                d_o.y as i32 + h,
                mask,
                glow::NEAREST,
            );
        }
        gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, attachment, glow::TEXTURE_2D, None, 0);
        gl.framebuffer_texture_2d(glow::DRAW_FRAMEBUFFER, attachment, glow::TEXTURE_2D, None, 0);
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
    }
}

/// Zero a new texture (wgpu's guarantee that it reads as zeros).
pub(crate) fn zero_texture(gl: &glow::Context, shared: &Shared, tex: &Texture) {
    zero_range(gl, shared, tex, 0..tex.mip_level_count(), 0..tex.layers());
}

fn zero_range(gl: &glow::Context, shared: &Shared, tex: &Texture, mips: Range<u32>, layers: Range<u32>) {
    let f = tex.format();
    // render targets, depth and multisampled textures: cleared through the draw framebuffer
    if f.has_depth() || tex.sample_count() > 1 {
        // SAFETY: GL calls on the current context with a live texture.
        unsafe {
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(shared.fbo_draw));
            let attachment = if f.has_depth() { depth_attachment(f) } else { glow::COLOR_ATTACHMENT0 };
            gl.disable(glow::SCISSOR_TEST);
            for mip in mips {
                for layer in layers.clone() {
                    match tex.0.target {
                        _ if tex.0.rb.is_some() => gl.framebuffer_renderbuffer(glow::DRAW_FRAMEBUFFER, attachment, glow::RENDERBUFFER, tex.0.rb),
                        glow::TEXTURE_2D | glow::TEXTURE_2D_MULTISAMPLE => gl.framebuffer_texture_2d(glow::DRAW_FRAMEBUFFER, attachment, tex.0.target, Some(tex.0.raw), mip as i32),
                        glow::TEXTURE_CUBE_MAP => gl.framebuffer_texture_2d(glow::DRAW_FRAMEBUFFER, attachment, image_target(tex, layer), Some(tex.0.raw), mip as i32),
                        _ => gl.framebuffer_texture_layer(glow::DRAW_FRAMEBUFFER, attachment, Some(tex.0.raw), mip as i32, layer as i32),
                    }
                    if f.has_depth() {
                        gl.depth_mask(true);
                        gl.stencil_mask_separate(glow::FRONT_AND_BACK, !0);
                        if f.has_stencil() {
                            gl.clear_buffer_depth_stencil(glow::DEPTH_STENCIL, 0, 0.0, 0);
                        } else {
                            gl.clear_buffer_f32_slice(glow::DEPTH, 0, &[0.0]);
                        }
                    } else {
                        gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                        gl.color_mask_draw_buffer(0, true, true, true, true);
                        gl.clear_buffer_f32_slice(glow::COLOR, 0, &[0.0; 4]);
                    }
                }
            }
            gl.framebuffer_texture_2d(glow::DRAW_FRAMEBUFFER, attachment, glow::TEXTURE_2D, None, 0);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
        }
        return;
    }
    for mip in mips {
        let s = tex.size().mip_level_size(mip, tex.dimension());
        let (bw, bh) = f.block_dimensions();
        let row = s.width.div_ceil(bw) * f.block_copy_size().unwrap_or(4);
        let rows = s.height.div_ceil(bh);
        let depth = if tex.dimension() == t::TextureDimension::D3 {
            s.depth_or_array_layers
        } else {
            layers.len() as u32
        };
        let zeros = vec![0u8; row as usize * rows as usize * depth as usize];
        let origin = t::Origin3d {
            x: 0,
            y: 0,
            z: if tex.dimension() == t::TextureDimension::D3 { 0 } else { layers.start },
        };
        let size = t::Extent3d {
            width: s.width,
            height: s.height,
            depth_or_array_layers: depth,
        };
        upload(
            gl,
            tex,
            mip,
            origin,
            size,
            t::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(rows),
            },
            Source::Bytes(&zeros),
        );
    }
}

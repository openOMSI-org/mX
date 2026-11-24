//! Command encoders and passes.

use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;

use mx_shader::MAX_ROOT_GROUP;
use mx_shader::msl::{self, RootSlot};
use mx_types as t;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::device::Shared;
use crate::pipeline::StageInfo;
use crate::resource::{Bound, copy_buffer_texture};
use crate::{BindGroup, Buffer, ComputePipeline, Device, Mt, Po, QuerySet, RenderPipeline, Texture, TextureView, conv};

const GROUPS: usize = MAX_ROOT_GROUP as usize + 1;
const MAX_IMMEDIATES: usize = 4096;

/// Recorded commands, ready for [`crate::Queue::submit`].
pub struct CommandBuffer {
    pub(crate) raw: Mt<Po<dyn MTLCommandBuffer>>,
    /// Textures copied into (their heap identity), for the heap's hazard declarations.
    pub(crate) written: Vec<usize>,
}

pub struct CommandEncoder {
    pub(crate) raw: Mt<Po<dyn MTLCommandBuffer>>,
    pub(crate) shared: Arc<Shared>,
    /// Occlusion query sets zeroed in this encoder.
    reset_queries: Vec<usize>,
    /// Textures copied into so far (see `Heap::written`): later passes declare them.
    written: Vec<usize>,
}

impl Device {
    pub fn create_command_encoder(&self, label: Option<&str>) -> CommandEncoder {
        let raw = self.shared.new_command_buffer();
        if let Some(l) = label {
            raw.setLabel(Some(&crate::ns(l)));
        }
        CommandEncoder {
            raw,
            shared: self.shared.clone(),
            reset_queries: Vec::new(),
            written: Vec::new(),
        }
    }
}

impl CommandEncoder {
    /// The `MTLCommandBuffer` the encoder records into, for interop (MetalFX encodes its
    /// scaler here). Passes must be ended before.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }

    /// Point a batch of GPU-driven draws at the arguments and the count the GPU wrote:
    /// `count` (a `u32` at `count_offset`) draws of `args` from `args_offset`, at most the
    /// batch's `max_draws`, drawn with `pipeline` and the index buffer `index` (indexed
    /// batches). Outside passes, after the arguments are written and before the pass that
    /// executes the batch: a kernel encodes the draws into the batch's indirect command buffer
    /// here (the ones past the count are reset).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indirect_draws(
        &mut self,
        draws: &crate::IndirectDraws,
        pipeline: &crate::RenderPipeline,
        args: &Buffer,
        args_offset: u64,
        count: &Buffer,
        count_offset: u64,
        index: Option<(&Buffer, u64, t::IndexFormat)>,
    ) {
        let d = &draws.0;
        assert_eq!(d.indexed, index.is_some(), "an indexed batch needs an index buffer, and only it");
        assert!(
            args.usage().contains(t::BufferUsages::INDIRECT) && count.usage().contains(t::BufferUsages::INDIRECT),
            "indirect draw arguments and counts need INDIRECT usage"
        );
        let device = crate::Device { shared: self.shared.clone() };
        let [draw, indexed] = device.indirect_kernels();
        let params: [u32; 4] = [
            d.max_draws,
            pipeline.0.primitive.0 as u32,
            index.map_or(0, |(_, o, _)| o as u32),
            index.is_some_and(|(_, _, f)| f == t::IndexFormat::Uint32) as u32,
        ];
        let enc = self.raw.computeCommandEncoder().expect("Metal returned no compute encoder");
        enc.setLabel(Some(&crate::ns("mX indirect draws")));
        enc.setComputePipelineState(if d.indexed { &indexed } else { &draw });
        // SAFETY: live buffers at the kernel's slots; the parameters are 16 bytes.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&d.container), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&args.0.raw), args_offset as usize, 1);
            enc.setBuffer_offset_atIndex(Some(&count.0.raw), count_offset as usize, 2);
            enc.setBytes_length_atIndex(NonNull::new(params.as_ptr().cast_mut().cast()).expect("not null"), 16, 3);
            if let Some((b, _, _)) = index {
                enc.setBuffer_offset_atIndex(Some(&b.0.raw), 0, 4);
            }
        }
        // (the kernel writes the commands through the argument buffer)
        enc.useResource_usage(ProtocolObject::from_ref(&**d.icb), MTLResourceUsage::Write);
        let width = draw.threadExecutionWidth().max(1);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: d.max_draws.max(1) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize { width, height: 1, depth: 1 },
        );
        enc.endEncoding();
        *d.index.lock() = index.map(|(b, _, _)| b.clone());
    }

    /// Fill `size` bytes (to the end with `None`) from `offset` with zeros.
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: Option<u64>) {
        let size = size.unwrap_or(buffer.size() - offset);
        assert!(offset + size <= buffer.size(), "clear_buffer past the end of the buffer");
        if size == 0 {
            return;
        }
        self.wrote_id(buffer.owner_id());
        self.blit(|b| b.fillBuffer_range_value(&buffer.0.raw, objc2_foundation::NSRange::new(offset as usize, size as usize), 0));
    }

    /// Zero the mips and layers of `range` (colour to 0, depth to 0.0, stencil to 0).
    pub fn clear_texture(&mut self, texture: &Texture, range: &t::ImageSubresourceRange) {
        let layers = t::Unwritten::layer_count(texture.dimension(), texture.size());
        let (mips, layers) = range.resolve(texture.mip_level_count(), layers);
        let aspects = crate::zero::Aspects::of(texture.format(), range.aspect);
        self.wrote(texture);
        for level in mips {
            crate::zero::zero(&self.raw, &self.shared, texture, level, layers.clone(), aspects);
        }
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.raw.pushDebugGroup(&crate::ns(label));
    }

    pub fn pop_debug_group(&mut self) {
        self.raw.popDebugGroup();
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        // (a command buffer has no signposts: an empty group stands in)
        self.raw.pushDebugGroup(&crate::ns(label));
        self.raw.popDebugGroup();
    }

    pub fn finish(self) -> CommandBuffer {
        CommandBuffer { raw: self.raw, written: self.written }
    }

    fn wrote(&mut self, t: &Texture) {
        self.wrote_id(t.owner_id());
    }

    /// A texture or buffer (by heap identity) was written by a blit.
    fn wrote_id(&mut self, id: usize) {
        if !self.written.contains(&id) {
            self.written.push(id);
        }
    }

    fn blit(&mut self, f: impl FnOnce(&ProtocolObject<dyn MTLBlitCommandEncoder>)) {
        let b = self.raw.blitCommandEncoder().expect("Metal returned no blit encoder");
        f(&b);
        b.endEncoding();
    }

    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64) {
        assert!(src_offset + size <= src.size() && dst_offset + size <= dst.size(), "copy_buffer_to_buffer out of range");
        self.wrote_id(dst.owner_id());
        self.blit(|b| {
            // SAFETY: both ranges checked above.
            unsafe { b.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&src.0.raw, src_offset as usize, &dst.0.raw, dst_offset as usize, size as usize) }
        });
    }

    pub fn copy_buffer_to_texture(&mut self, src: t::TexelCopyBufferInfo<'_, Buffer>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d) {
        src.layout.assert_encoder_pitch();
        self.wrote(dst.texture);
        self.blit(|b| copy_buffer_texture(b, &src.buffer.0.raw, src.layout.offset, dst.texture, &dst, src.layout, size, true));
    }

    pub fn copy_texture_to_buffer(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyBufferInfo<'_, Buffer>, size: t::Extent3d) {
        dst.layout.assert_encoder_pitch();
        self.wrote_id(dst.buffer.owner_id());
        self.blit(|b| copy_buffer_texture(b, &dst.buffer.0.raw, dst.layout.offset, src.texture, &src, dst.layout, size, false));
    }

    pub fn copy_texture_to_texture(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d) {
        self.wrote(dst.texture);
        let is_3d = src.texture.dimension() == t::TextureDimension::D3;
        // (cut to both levels' real sizes, as for buffer copies: see `clamp_to_mip`)
        let a = crate::resource::clamp_to_mip(src.texture, src.mip_level, src.origin, size);
        let b = crate::resource::clamp_to_mip(dst.texture, dst.mip_level, dst.origin, size);
        let size = t::Extent3d {
            width: a.width.min(b.width),
            height: a.height.min(b.height),
            depth_or_array_layers: a.depth_or_array_layers.min(b.depth_or_array_layers),
        };
        let (slices, depth) = if is_3d { (1, size.depth_or_array_layers) } else { (size.depth_or_array_layers, 1) };
        self.blit(|b| {
            for s in 0..slices {
                let (src_slice, dst_slice) = if is_3d { (0, 0) } else { (src.origin.z + s, dst.origin.z + s) };
                let so = t::Origin3d {
                    z: if is_3d { src.origin.z } else { 0 },
                    ..src.origin
                };
                let d_o = t::Origin3d {
                    z: if is_3d { dst.origin.z } else { 0 },
                    ..dst.origin
                };
                // SAFETY: regions inside both textures (Metal validates in debug).
                unsafe {
                    b.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &src.texture.0.raw,
                        src_slice as usize,
                        src.mip_level as usize,
                        conv::origin(so),
                        conv::size(t::Extent3d { depth_or_array_layers: depth, ..size }),
                        &dst.texture.0.raw,
                        dst_slice as usize,
                        dst.mip_level as usize,
                        conv::origin(d_o),
                    )
                };
            }
        });
    }

    /// Fill a texture's mip chain from level 0 (box filter, as Metal's blit does).
    pub fn generate_mipmaps(&mut self, texture: &Texture) {
        self.wrote(texture);
        self.blit(|b| b.generateMipmapsForTexture(&texture.0.raw));
    }

    pub fn begin_render_pass(&mut self, desc: &t::RenderPassDescriptor<'_, TextureView, QuerySet>) -> RenderPass<'_> {
        let d = MTLRenderPassDescriptor::new();
        if let Some(ts) = &desc.timestamp_writes {
            // SAFETY: attachment 0; sample indices inside the set (or "don't sample").
            unsafe {
                let a = d.sampleBufferAttachments().objectAtIndexedSubscript(0);
                a.setSampleBuffer(Some(ts.query_set.sample_buffer()));
                a.setStartOfVertexSampleIndex(ts.beginning_of_pass_write_index.map_or(MTLCounterDontSample, |i| i as usize));
                a.setEndOfVertexSampleIndex(MTLCounterDontSample);
                a.setStartOfFragmentSampleIndex(MTLCounterDontSample);
                a.setEndOfFragmentSampleIndex(ts.end_of_pass_write_index.map_or(MTLCounterDontSample, |i| i as usize));
            }
        }
        if let Some(q) = desc.occlusion_query_set {
            // (Metal adds to what is in the buffer: start every query of the set at zero, once
            // per encoder, before the first pass that counts into it)
            let key = std::sync::Arc::as_ptr(&q.0) as usize;
            if !self.reset_queries.contains(&key) {
                self.reset_queries.push(key);
                let vb = q.visibility_buffer();
                self.blit(|b| b.fillBuffer_range_value(vb, objc2_foundation::NSRange::new(0, vb.length()), 0));
            }
            d.setVisibilityResultBuffer(Some(q.visibility_buffer()));
        }
        for (i, a) in desc.color_attachments.iter().enumerate() {
            let Some(a) = a else { continue };
            // SAFETY: `i` below Metal's 8 colour attachments.
            let ca = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
            set_attachment_view(&ca, a.view);
            ca.setLoadAction(conv::load_action(&a.ops.load));
            if let t::LoadOp::Clear(c) = a.ops.load {
                ca.setClearColor(MTLClearColor {
                    red: c.r,
                    green: c.g,
                    blue: c.b,
                    alpha: c.a,
                });
            }
            if let Some(r) = a.resolve_target {
                ca.setResolveTexture(Some(&r.0.raw));
                ca.setResolveLevel(r.0.base_mip_level as usize);
                ca.setResolveSlice(r.0.base_array_layer as usize);
            }
            ca.setStoreAction(conv::store_action(a.ops.store, a.resolve_target.is_some()));
        }
        if let Some(ds) = &desc.depth_stencil_attachment {
            let format = ds.view.texture().format();
            if format.has_depth() {
                let da = d.depthAttachment();
                set_attachment_view(&da, ds.view);
                match &ds.depth_ops {
                    Some(ops) => {
                        da.setLoadAction(conv::load_action(&ops.load));
                        if let t::LoadOp::Clear(v) = ops.load {
                            da.setClearDepth(v as f64);
                        }
                        da.setStoreAction(conv::store_action(ops.store, false));
                    }
                    None => {
                        da.setLoadAction(MTLLoadAction::Load);
                        da.setStoreAction(MTLStoreAction::Store);
                    }
                }
            }
            if format.has_stencil() {
                let sa = d.stencilAttachment();
                set_attachment_view(&sa, ds.view);
                match &ds.stencil_ops {
                    Some(ops) => {
                        sa.setLoadAction(conv::load_action(&ops.load));
                        if let t::LoadOp::Clear(v) = ops.load {
                            sa.setClearStencil(v);
                        }
                        sa.setStoreAction(conv::store_action(ops.store, false));
                    }
                    None => {
                        sa.setLoadAction(MTLLoadAction::Load);
                        sa.setStoreAction(MTLStoreAction::Store);
                    }
                }
            }
        }
        let enc = self.raw.renderCommandEncoderWithDescriptor(&d).expect("Metal returned no render encoder");
        if let Some(l) = desc.label {
            enc.setLabel(Some(&crate::ns(l)));
        }
        // (the heap's argument buffers go where each pipeline's stages want them, see
        // `bind_stage`)
        RenderPass {
            enc: Mt(enc),
            shared: &self.shared,
            state: PassState {
                written: self.written.clone(),
                tables: self.shared.heap.lock().tables(),
                ..PassState::default()
            },
            pipeline: None,
            index: None,
        }
    }

    pub fn begin_compute_pass(&mut self, desc: &t::ComputePassDescriptor<'_, QuerySet>) -> ComputePass<'_> {
        let d = MTLComputePassDescriptor::new();
        if let Some(ts) = &desc.timestamp_writes {
            // SAFETY: attachment 0; sample indices inside the set (or "don't sample").
            unsafe {
                let a = d.sampleBufferAttachments().objectAtIndexedSubscript(0);
                a.setSampleBuffer(Some(ts.query_set.sample_buffer()));
                a.setStartOfEncoderSampleIndex(ts.beginning_of_pass_write_index.map_or(MTLCounterDontSample, |i| i as usize));
                a.setEndOfEncoderSampleIndex(ts.end_of_pass_write_index.map_or(MTLCounterDontSample, |i| i as usize));
            }
        }
        let enc = self.raw.computeCommandEncoderWithDescriptor(&d).expect("Metal returned no compute encoder");
        if let Some(l) = desc.label {
            enc.setLabel(Some(&crate::ns(l)));
        }
        ComputePass {
            enc: Mt(enc),
            shared: &self.shared,
            state: PassState {
                written: self.written.clone(),
                tables: self.shared.heap.lock().tables(),
                ..PassState::default()
            },
            pipeline: None,
        }
    }
}

fn set_attachment_view(a: &MTLRenderPassAttachmentDescriptor, v: &TextureView) {
    // (the view's own texture: Metal renders to a level / slice of it)
    a.setTexture(Some(&v.0.raw));
}

/// Bindings set on a pass and not yet handed to the encoder.
struct PassState {
    groups: [Option<BindGroup>; GROUPS],
    roots_dirty: bool,
    immediates: Box<[u8; MAX_IMMEDIATES]>,
    immediates_dirty: bool,
    heap_generation: u64,
    /// The encoder's copied-into textures and buffers when the pass began (see `Heap::written`).
    written: Vec<usize>,
    /// The heap's argument buffers, by `msl::HeapTable`.
    tables: Option<crate::heap::Tables>,
    /// The pipeline changed: bind its stages' heap tables.
    heap_dirty: bool,
    /// Bytes from each vertex buffer's offset to its end (0: none set), for the vertex
    /// shader's own fetch.
    vertex_sizes: [u64; msl::MAX_VERTEX_BUFFERS as usize],
    /// Offsets added to root buffers' own (`set_buffer_offset`): (group, binding, bytes).
    offsets: Vec<(u32, u32, u64)>,
}

impl PassState {
    /// What `set_buffer_offset` added to the buffer at (`group`, `binding`).
    fn extra_offset(&self, group: u32, binding: u32) -> u64 {
        self.offsets.iter().find(|o| o.0 == group && o.1 == binding).map_or(0, |o| o.2)
    }
}

impl Default for PassState {
    fn default() -> Self {
        PassState {
            offsets: Vec::new(),
            groups: Default::default(),
            roots_dirty: true,
            immediates: Box::new([0; MAX_IMMEDIATES]),
            immediates_dirty: true,
            heap_generation: 0,
            written: Vec::new(),
            tables: None,
            heap_dirty: true,
            vertex_sizes: [0; msl::MAX_VERTEX_BUFFERS as usize],
        }
    }
}

/// A function binding one stage's resources: (slot, buffer, offset) / (slot, texture) /
/// (slot, sampler) / (slot, bytes).
trait StageBinder {
    fn buffer(&self, slot: u32, b: &ProtocolObject<dyn MTLBuffer>, offset: u64);
    /// A new offset into the buffer already at `slot`.
    fn buffer_offset(&self, slot: u32, offset: u64);
    fn texture(&self, slot: u32, t: &ProtocolObject<dyn MTLTexture>);
    fn sampler(&self, slot: u32, s: &ProtocolObject<dyn MTLSamplerState>);
    fn bytes(&self, slot: u32, data: &[u8]);
    fn accel(&self, slot: u32, a: &ProtocolObject<dyn MTLAccelerationStructure>);
    /// Make resources reached only indirectly (the BLASes of a TLAS) resident.
    fn use_read(&self, list: &[NonNull<ProtocolObject<dyn MTLResource>>]);
}

struct VertexStage<'a>(&'a ProtocolObject<dyn MTLRenderCommandEncoder>);
struct FragmentStage<'a>(&'a ProtocolObject<dyn MTLRenderCommandEncoder>);
struct ComputeStage<'a>(&'a ProtocolObject<dyn MTLComputeCommandEncoder>);

fn bytes_ptr(data: &[u8]) -> NonNull<std::ffi::c_void> {
    NonNull::new(data.as_ptr().cast_mut().cast()).expect("slice pointer is never null")
}

// SAFETY (all of the below): live objects bound at slots inside Metal's tables, as laid out
// by `mx_shader::msl`.
impl StageBinder for VertexStage<'_> {
    fn buffer(&self, slot: u32, b: &ProtocolObject<dyn MTLBuffer>, offset: u64) {
        unsafe { self.0.setVertexBuffer_offset_atIndex(Some(b), offset as usize, slot as usize) }
    }
    fn buffer_offset(&self, slot: u32, offset: u64) {
        unsafe { self.0.setVertexBufferOffset_atIndex(offset as usize, slot as usize) }
    }
    fn texture(&self, slot: u32, t: &ProtocolObject<dyn MTLTexture>) {
        unsafe { self.0.setVertexTexture_atIndex(Some(t), slot as usize) }
    }
    fn sampler(&self, slot: u32, s: &ProtocolObject<dyn MTLSamplerState>) {
        unsafe { self.0.setVertexSamplerState_atIndex(Some(s), slot as usize) }
    }
    fn bytes(&self, slot: u32, data: &[u8]) {
        unsafe { self.0.setVertexBytes_length_atIndex(bytes_ptr(data), data.len(), slot as usize) }
    }
    fn accel(&self, slot: u32, a: &ProtocolObject<dyn MTLAccelerationStructure>) {
        unsafe { self.0.setVertexAccelerationStructure_atBufferIndex(Some(a), slot as usize) }
    }
    fn use_read(&self, list: &[NonNull<ProtocolObject<dyn MTLResource>>]) {
        if let Some(p) = NonNull::new(list.as_ptr().cast_mut()) {
            unsafe { self.0.useResources_count_usage_stages(p, list.len(), MTLResourceUsage::Read, MTLRenderStages::Vertex) }
        }
    }
}

impl StageBinder for FragmentStage<'_> {
    fn buffer(&self, slot: u32, b: &ProtocolObject<dyn MTLBuffer>, offset: u64) {
        unsafe { self.0.setFragmentBuffer_offset_atIndex(Some(b), offset as usize, slot as usize) }
    }
    fn buffer_offset(&self, slot: u32, offset: u64) {
        unsafe { self.0.setFragmentBufferOffset_atIndex(offset as usize, slot as usize) }
    }
    fn texture(&self, slot: u32, t: &ProtocolObject<dyn MTLTexture>) {
        unsafe { self.0.setFragmentTexture_atIndex(Some(t), slot as usize) }
    }
    fn sampler(&self, slot: u32, s: &ProtocolObject<dyn MTLSamplerState>) {
        unsafe { self.0.setFragmentSamplerState_atIndex(Some(s), slot as usize) }
    }
    fn bytes(&self, slot: u32, data: &[u8]) {
        unsafe { self.0.setFragmentBytes_length_atIndex(bytes_ptr(data), data.len(), slot as usize) }
    }
    fn accel(&self, slot: u32, a: &ProtocolObject<dyn MTLAccelerationStructure>) {
        unsafe { self.0.setFragmentAccelerationStructure_atBufferIndex(Some(a), slot as usize) }
    }
    fn use_read(&self, list: &[NonNull<ProtocolObject<dyn MTLResource>>]) {
        if let Some(p) = NonNull::new(list.as_ptr().cast_mut()) {
            unsafe { self.0.useResources_count_usage_stages(p, list.len(), MTLResourceUsage::Read, MTLRenderStages::Fragment) }
        }
    }
}

impl StageBinder for ComputeStage<'_> {
    fn buffer(&self, slot: u32, b: &ProtocolObject<dyn MTLBuffer>, offset: u64) {
        unsafe { self.0.setBuffer_offset_atIndex(Some(b), offset as usize, slot as usize) }
    }
    fn buffer_offset(&self, slot: u32, offset: u64) {
        unsafe { self.0.setBufferOffset_atIndex(offset as usize, slot as usize) }
    }
    fn texture(&self, slot: u32, t: &ProtocolObject<dyn MTLTexture>) {
        unsafe { self.0.setTexture_atIndex(Some(t), slot as usize) }
    }
    fn sampler(&self, slot: u32, s: &ProtocolObject<dyn MTLSamplerState>) {
        unsafe { self.0.setSamplerState_atIndex(Some(s), slot as usize) }
    }
    fn bytes(&self, slot: u32, data: &[u8]) {
        unsafe { self.0.setBytes_length_atIndex(bytes_ptr(data), data.len(), slot as usize) }
    }
    fn accel(&self, slot: u32, a: &ProtocolObject<dyn MTLAccelerationStructure>) {
        unsafe { self.0.setAccelerationStructure_atBufferIndex(Some(a), slot as usize) }
    }
    fn use_read(&self, list: &[NonNull<ProtocolObject<dyn MTLResource>>]) {
        if let Some(p) = NonNull::new(list.as_ptr().cast_mut()) {
            unsafe { self.0.useResources_count_usage(p, list.len(), MTLResourceUsage::Read) }
        }
    }
}

/// Bind the root resources and immediates one stage reads.
/// Bind the heap textures and samplers `stage` reads directly (`msl::write_direct`): each at the
/// index its immediates (or its constant) give. An index with nothing at it binds nothing.
fn bind_direct(b: &impl StageBinder, stage: &StageInfo, state: &PassState, heap: &crate::heap::Heap) {
    for d in &stage.direct {
        let index = match d.slot.index {
            mx_shader::bound::Index::Immediate(at) => {
                let at = at as usize;
                u32::from_le_bytes([state.immediates[at], state.immediates[at + 1], state.immediates[at + 2], state.immediates[at + 3]])
            }
            mx_shader::bound::Index::Constant(i) => i,
        };
        match d.root {
            RootSlot::Texture(slot) => {
                if let Some(t) = heap.texture_at(index) {
                    b.texture(slot, t);
                }
            }
            RootSlot::Sampler(slot) => {
                if let Some(s) = heap.sampler_at(index) {
                    b.sampler(slot, s);
                }
            }
            RootSlot::Buffer { slot, .. } => {
                if let Some((buffer, offset)) = heap.buffer_at(index) {
                    b.buffer(slot, buffer, offset);
                }
            }
            RootSlot::AccelerationStructure(slot) => {
                if let Some(tlas) = heap.tlas_at(index) {
                    b.accel(slot, &tlas.raw);
                    let built = tlas.built_with.lock().clone();
                    let list: Vec<NonNull<ProtocolObject<dyn MTLResource>>> = built.iter().map(|x| NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&*x.0.raw.0))).collect();
                    b.use_read(&list);
                }
            }
        }
    }
}

fn bind_stage(b: &impl StageBinder, stage: &StageInfo, state: &PassState, immediate_size: u32, pipeline_label: &dyn Fn() -> String) {
    if state.heap_dirty
        && let Some(tables) = &state.tables
    {
        for (slot, table) in stage.layout.heap_tables() {
            b.buffer(slot, tables.get(table), 0);
        }
    }
    if state.roots_dirty {
        let mut sizes: Vec<u32> = Vec::with_capacity(stage.sized_buffers.len());
        for (&(group, binding), slot) in &stage.layout.roots {
            // (heap textures bound directly: `bind_direct`)
            if group == mx_shader::HEAP_GROUP {
                continue;
            }
            let resource = state.groups[group as usize].as_ref().and_then(|g| g.get(binding));
            let Some(resource) = resource else {
                panic!("{}: draw with nothing bound at @group({group}) @binding({binding})", pipeline_label());
            };
            match (slot, resource) {
                (RootSlot::Buffer { slot, .. }, Bound::Buffer { buffer, offset, .. }) => b.buffer(*slot, &buffer.0.raw, *offset + state.extra_offset(group, binding)),
                (RootSlot::Texture(s), Bound::Texture(v)) => b.texture(*s, &v.0.raw),
                (RootSlot::Sampler(s), Bound::Sampler(smp)) => b.sampler(*s, &smp.0.raw),
                (RootSlot::AccelerationStructure(s), Bound::Tlas(tlas)) => {
                    b.accel(*s, &tlas.raw);
                    let built = tlas.built_with.lock().clone();
                    let list: Vec<NonNull<ProtocolObject<dyn MTLResource>>> = built.iter().map(|x| NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&*x.0.raw.0))).collect();
                    b.use_read(&list);
                }
                _ => panic!("{}: the resource at @group({group}) @binding({binding}) is not the kind the shader declares", pipeline_label()),
            }
        }
        for key in &stage.sized_buffers {
            if key.0 == mx_shader::HEAP_GROUP {
                // (a heap buffer view's entry, which nothing reads)
                sizes.push(0);
            } else if let Some(Bound::Buffer { size, .. }) = state.groups[key.0 as usize].as_ref().and_then(|g| g.get(key.1)) {
                sizes.push(*size as u32);
            }
        }
        sizes.extend(stage.vertex_buffers.iter().map(|&i| u32::try_from(state.vertex_sizes[i as usize]).unwrap_or(u32::MAX)));
        if !sizes.is_empty() {
            b.bytes(msl::SIZES_SLOT, bytemuck_u32(&sizes));
        }
    }
    if state.immediates_dirty && stage.uses_immediates && immediate_size > 0 {
        b.bytes(msl::IMMEDIATES_SLOT, &state.immediates[..immediate_size as usize]);
    }
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and u8 has no alignment needs.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// A render pass. Ends when dropped.
pub struct RenderPass<'e> {
    enc: Mt<Po<dyn MTLRenderCommandEncoder>>,
    shared: &'e Shared,
    state: PassState,
    pipeline: Option<RenderPipeline>,
    index: Option<(Buffer, u64, t::IndexFormat)>,
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        self.enc.endEncoding();
    }
}

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        let p = &pipeline.0;
        self.enc.setRenderPipelineState(&p.state);
        if let Some(ds) = &p.depth_stencil {
            self.enc.setDepthStencilState(Some(ds));
        }
        self.enc.setCullMode(p.cull);
        self.enc.setFrontFacingWinding(p.winding);
        self.enc.setDepthClipMode(if p.clamp_depth { MTLDepthClipMode::Clamp } else { MTLDepthClipMode::Clip });
        self.enc.setDepthBias_slopeScale_clamp(p.bias.constant as f32, p.bias.slope_scale, p.bias.clamp);
        self.pipeline = Some(pipeline.clone());
        self.state.roots_dirty = true;
        self.state.immediates_dirty = true;
        self.state.heap_dirty = true;
    }

    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        assert!(group <= MAX_ROOT_GROUP, "root bind groups are 0..={MAX_ROOT_GROUP}; group {} is the heap", mx_shader::HEAP_GROUP);
        self.state.groups[group as usize] = Some(bind_group.clone());
        self.state.offsets.retain(|o| o.0 != group);
        self.state.roots_dirty = true;
    }

    /// Read the buffer at (`group`, `binding`) of the bind group set at `group` from `offset`
    /// bytes past its own offset, for the draws that follow (until that group is set again):
    /// a draw's constants in one buffer of many, as a dynamic offset in wgpu or a root CBV in
    /// Direct3D 12. `offset` is a multiple of `min_uniform_buffer_offset_alignment` (of the
    /// storage one for a storage buffer).
    pub fn set_buffer_offset(&mut self, group: u32, binding: u32, offset: u64) {
        match self.state.offsets.iter_mut().find(|o| o.0 == group && o.1 == binding) {
            Some(o) => o.2 = offset,
            None => self.state.offsets.push((group, binding, offset)),
        }
        if self.state.roots_dirty {
            return;
        }
        // (bound already: only the offset changes, as Metal's own call does it)
        let Some(pipeline) = &self.pipeline else { return };
        let base = match self.state.groups[group as usize].as_ref().and_then(|g| g.get(binding)) {
            Some(Bound::Buffer { offset: base, .. }) => *base,
            _ => return,
        };
        let p = &pipeline.0;
        if let Some(RootSlot::Buffer { slot, .. }) = p.vertex.layout.roots.get(&(group, binding)) {
            VertexStage(&self.enc).buffer_offset(*slot, base + offset);
        }
        if let Some(f) = &p.fragment
            && let Some(RootSlot::Buffer { slot, .. }) = f.layout.roots.get(&(group, binding))
        {
            FragmentStage(&self.enc).buffer_offset(*slot, base + offset);
        }
    }

    /// Set `data` at byte `offset` of the shader's `var<immediate>` block.
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        let o = offset as usize;
        self.state.immediates[o..o + data.len()].copy_from_slice(data);
        self.state.immediates_dirty = true;
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64) {
        assert!(slot < msl::MAX_VERTEX_BUFFERS, "vertex buffer slot {slot}: mX takes {}", msl::MAX_VERTEX_BUFFERS);
        // SAFETY: live buffer at a vertex-buffer slot.
        unsafe { self.enc.setVertexBuffer_offset_atIndex(Some(&buffer.0.raw), offset as usize, msl::vertex_buffer_slot(slot) as usize) };
        // (the vertex shader is told the buffer's size: it bounds its own fetches by it)
        self.state.vertex_sizes[slot as usize] = buffer.size().saturating_sub(offset);
        self.state.roots_dirty = true;
    }

    pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64, format: t::IndexFormat) {
        self.index = Some((buffer.clone(), offset, format));
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        self.enc.setViewport(MTLViewport {
            originX: x as f64,
            originY: y as f64,
            width: w as f64,
            height: h as f64,
            znear: min_depth as f64,
            zfar: max_depth as f64,
        });
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.enc.setScissorRect(MTLScissorRect {
            x: x as usize,
            y: y as usize,
            width: w as usize,
            height: h as usize,
        });
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        self.enc.setStencilReferenceValue(reference);
    }

    pub fn set_blend_constant(&mut self, c: t::Color) {
        self.enc.setBlendColorRed_green_blue_alpha(c.r as f32, c.g as f32, c.b as f32, c.a as f32);
    }

    pub(crate) fn reset_state(&mut self) {
        self.state.offsets.clear();
        self.pipeline = None;
        self.index = None;
        self.state.groups = Default::default();
        self.state.vertex_sizes = Default::default();
        self.state.roots_dirty = true;
        self.state.immediates_dirty = true;
    }

    pub fn draw_indirect(&mut self, args: &Buffer, offset: u64) {
        let prim = self.flush().0.primitive;
        // SAFETY: `t::DrawIndirectArgs` is Metal's MTLDrawPrimitivesIndirectArguments.
        unsafe { self.enc.drawPrimitives_indirectBuffer_indirectBufferOffset(prim, &args.0.raw, offset as usize) };
    }

    /// Draw a batch the GPU filled (see `CommandEncoder::encode_indirect_draws`), with the
    /// pipeline and bindings set now: its indirect command buffer, which inherits them.
    pub fn execute_indirect_draws(&mut self, draws: &crate::IndirectDraws) {
        let d = &draws.0;
        let pipeline = self.flush().clone();
        // (the commands inherit the pipeline: its variant made for indirect command buffers,
        // for this batch only)
        let icb_state = match pipeline.icb_state(&self.shared.device) {
            Ok(s) => s.clone(),
            Err(e) => panic!("mX: this pipeline cannot draw GPU-driven batches on Metal: {e}"),
        };
        self.enc.setRenderPipelineState(&icb_state);
        let stages = MTLRenderStages::Vertex | MTLRenderStages::Fragment;
        self.enc.useResource_usage_stages(ProtocolObject::from_ref(&**d.icb), MTLResourceUsage::Read, stages);
        if let Some(b) = d.index.lock().as_ref() {
            // (the commands read the index buffer by address)
            self.enc.useResource_usage_stages(ProtocolObject::from_ref(&**b.0.raw), MTLResourceUsage::Read, MTLRenderStages::Vertex);
        }
        // SAFETY: an indirect command buffer of `max_draws` commands, encoded (or reset) by
        // `encode_indirect_draws`, executed with a pipeline that supports them.
        unsafe { self.enc.executeCommandsInBuffer_withRange(&d.icb, objc2_foundation::NSRange::new(0, d.max_draws as usize)) };
        self.enc.setRenderPipelineState(&pipeline.0.state);
    }

    pub fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64) {
        let prim = self.flush().0.primitive;
        let (buffer, ib_offset, format) = self.index.as_ref().expect("draw_indexed_indirect without an index buffer");
        // SAFETY: `t::DrawIndexedIndirectArgs` is Metal's MTLDrawIndexedPrimitivesIndirectArguments.
        unsafe {
            self.enc.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                prim,
                conv::index_type(*format),
                &buffer.0.raw,
                *ib_offset as usize,
                &args.0.raw,
                offset as usize,
            )
        };
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

    /// Count the samples of the following draws into query `index` of the pass's
    /// occlusion query set.
    pub fn begin_occlusion_query(&mut self, index: u32) {
        self.enc.setVisibilityResultMode_offset(MTLVisibilityResultMode::Counting, index as usize * 8);
    }

    pub fn end_occlusion_query(&mut self) {
        self.enc.setVisibilityResultMode_offset(MTLVisibilityResultMode::Disabled, 0);
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.enc.pushDebugGroup(&crate::ns(label));
    }

    pub fn pop_debug_group(&mut self) {
        self.enc.popDebugGroup();
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.enc.insertDebugSignpost(&crate::ns(label));
    }

    fn flush(&mut self) -> &RenderPipeline {
        let pipeline = self.pipeline.as_ref().expect("draw without a pipeline");
        let p = &pipeline.0;
        // textures reached through the heap must be resident in this encoder
        let heap_used = p.vertex.heap_used | p.fragment.as_ref().map_or(0, |f| f.heap_used);
        if heap_used != 0 {
            let heap = self.shared.heap.lock();
            if heap.generation != self.state.heap_generation {
                heap.use_on_render(&self.enc, &self.state.written);
                self.state.heap_generation = heap.generation;
            }
        }
        // (the pass's label only for a panic: asking Metal for it on every draw cost time)
        let enc = &self.enc;
        let label = || enc.label().map(|l| l.to_string()).unwrap_or_else(|| "render pass".into());
        bind_stage(&VertexStage(&self.enc), &p.vertex, &self.state, p.immediate_size, &label);
        if let Some(f) = &p.fragment {
            bind_stage(&FragmentStage(&self.enc), f, &self.state, p.immediate_size, &label);
        }
        // the heap textures and samplers the stages bind directly, by the immediates' indices
        let direct = !p.vertex.direct.is_empty() || p.fragment.as_ref().is_some_and(|f| !f.direct.is_empty());
        if direct && (self.state.roots_dirty || self.state.immediates_dirty || self.state.heap_dirty) {
            let heap = self.shared.heap.lock();
            bind_direct(&VertexStage(&self.enc), &p.vertex, &self.state, &heap);
            if let Some(f) = &p.fragment {
                bind_direct(&FragmentStage(&self.enc), f, &self.state, &heap);
            }
        }
        self.state.roots_dirty = false;
        self.state.immediates_dirty = false;
        self.state.heap_dirty = false;
        pipeline
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        let prim = self.flush().0.primitive;
        if vertices.is_empty() || instances.is_empty() {
            return;
        }
        // SAFETY: a complete pipeline and bindings were set by `flush`.
        unsafe {
            self.enc
                .drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(prim, vertices.start as usize, vertices.len(), instances.len(), instances.start as usize);
        }
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        let prim = self.flush().0.primitive;
        if indices.is_empty() || instances.is_empty() {
            return;
        }
        let (buffer, offset, format) = self.index.as_ref().expect("draw_indexed without an index buffer");
        // SAFETY: as in `draw`; the index range is the caller's to keep inside the buffer.
        unsafe {
            self.enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                prim,
                indices.len(),
                conv::index_type(*format),
                &buffer.0.raw,
                (offset + indices.start as u64 * format.size()) as usize,
                instances.len(),
                base_vertex as isize,
                instances.start as usize,
            );
        }
    }

    /// The `MTLRenderCommandEncoder`, for interop.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLRenderCommandEncoder> {
        &self.enc
    }
}

/// A compute pass. Ends when dropped.
pub struct ComputePass<'e> {
    enc: Mt<Po<dyn MTLComputeCommandEncoder>>,
    shared: &'e Shared,
    state: PassState,
    pipeline: Option<ComputePipeline>,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        self.enc.endEncoding();
    }
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.enc.setComputePipelineState(&pipeline.0.state);
        // (each `var<workgroup>`'s threadgroup memory, rounded up to 16 bytes as Metal asks
        // and wgpu-hal does)
        for (i, &size) in pipeline.0.stage.workgroup_memory.iter().enumerate() {
            // SAFETY: an index the compiled function declares, a length within the limit
            // (naga's validation checked the workgroup storage size).
            unsafe { self.enc.setThreadgroupMemoryLength_atIndex(size.next_multiple_of(16) as usize, i) };
        }
        self.pipeline = Some(pipeline.clone());
        self.state.roots_dirty = true;
        self.state.immediates_dirty = true;
        self.state.heap_dirty = true;
    }

    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        assert!(group <= MAX_ROOT_GROUP, "root bind groups are 0..={MAX_ROOT_GROUP}");
        self.state.groups[group as usize] = Some(bind_group.clone());
        self.state.roots_dirty = true;
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        let o = offset as usize;
        self.state.immediates[o..o + data.len()].copy_from_slice(data);
        self.state.immediates_dirty = true;
    }

    pub fn dispatch_workgroups_indirect(&mut self, args: &Buffer, offset: u64) {
        let wg = self.flush();
        // SAFETY: `t::DispatchIndirectArgs` is Metal's MTLDispatchThreadgroupsIndirectArguments.
        unsafe {
            self.enc
                .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(&args.0.raw, offset as usize, wg)
        };
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.enc.pushDebugGroup(&crate::ns(label));
    }

    pub fn pop_debug_group(&mut self) {
        self.enc.popDebugGroup();
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.enc.insertDebugSignpost(&crate::ns(label));
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        let wg = self.flush();
        if x == 0 || y == 0 || z == 0 {
            return;
        }
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: x as usize,
                height: y as usize,
                depth: z as usize,
            },
            wg,
        );
    }

    fn flush(&mut self) -> MTLSize {
        let pipeline = self.pipeline.clone().expect("dispatch without a pipeline");
        let p = &pipeline.0;
        if p.stage.heap_used != 0 {
            let heap = self.shared.heap.lock();
            if heap.generation != self.state.heap_generation {
                heap.use_on_compute(&self.enc, &self.state.written);
                self.state.heap_generation = heap.generation;
            }
        }
        bind_stage(&ComputeStage(&self.enc), &p.stage, &self.state, p.immediate_size, &|| "compute pass".to_string());
        if !p.stage.direct.is_empty() && (self.state.roots_dirty || self.state.immediates_dirty || self.state.heap_dirty) {
            let heap = self.shared.heap.lock();
            bind_direct(&ComputeStage(&self.enc), &p.stage, &self.state, &heap);
        }
        self.state.roots_dirty = false;
        self.state.immediates_dirty = false;
        self.state.heap_dirty = false;
        p.workgroup_size
    }
}

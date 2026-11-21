//! Render bundles: draws recorded once on the CPU and replayed into render passes.
//!
//! A bundle is recorded into a Direct3D 12 bundle (a command list of its own) when it is
//! finished, on the thread that finishes it; the pass that runs it then only puts the
//! bundle's resources in the states it needs and calls `ExecuteBundle`. Replaying the
//! commands into the pass's list instead cost the frame's thread about 1.4 ms in openOMSI,
//! where the bundles are recorded on helper threads to keep that thread free.
//!
//! What a bundle cannot hold (an indirect draw, which mX rewrites with copies first; a
//! storage write, which needs barriers between draws; two states of one resource) is
//! replayed as before. `MX_DX12_BUNDLES=0` replays every bundle, for comparison.

use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use mx_types as t;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::command::{Graphics, PassState, Target, bind_heap, bind_roots, flush_constants, need_heap_buffers};
use crate::conv;
use crate::device::{ListPair, Shared};
use crate::heap::BLOCK;
use crate::track::{Track, read_only};
use crate::{BindGroup, Buffer, Device, RenderPass, RenderPipeline};

#[derive(Clone)]
pub(crate) enum Cmd {
    Pipeline(RenderPipeline),
    BindGroup(u32, BindGroup),
    BufferOffset(u32, u32, u64),
    Immediates(u32, Box<[u8]>),
    VertexBuffer(u32, Buffer, u64),
    IndexBuffer(Buffer, u64, t::IndexFormat),
    Draw(Range<u32>, Range<u32>),
    DrawIndexed(Range<u32>, i32, Range<u32>),
    DrawIndirect(Buffer, u64),
    DrawIndexedIndirect(Buffer, u64),
}

/// Records a [`RenderBundle`]: the drawing half of a render pass.
pub struct RenderBundleEncoder {
    cmds: Vec<Cmd>,
    shared: Arc<Shared>,
    _desc: t::RenderBundleDepthStencil,
}

#[derive(Clone)]
pub struct RenderBundle(pub(crate) Arc<BundleInner>);

pub(crate) struct BundleInner {
    cmds: Vec<Cmd>,
    baked: Option<Baked>,
}

/// A bundle recorded into a Direct3D 12 bundle, with what the list that runs it must do first.
struct Baked {
    pair: Option<ListPair>,
    /// The states its resources must be in (read states of one resource merged).
    needs: Vec<Need>,
    /// It reads buffers through the heap (see `need_heap_buffers`).
    heap_buffers: bool,
    /// Its root tables' descriptor blocks, its own while it lives.
    blocks: Vec<u32>,
    shared: Arc<Shared>,
    /// Bind groups and the like the bundle's tables name (the commands keep the rest).
    _keep: Vec<Box<dyn Any + Send + Sync>>,
}

impl Drop for Baked {
    /// A list that ran the bundle keeps it until the GPU is done (`RenderPass::execute_bundles`),
    /// so the GPU no longer reads it here.
    fn drop(&mut self) {
        self.shared.heap.lock().free_blocks(self.blocks.drain(..));
        if let Some(pair) = self.pair.take() {
            self.shared.bundle_lists.lock().push(pair);
        }
    }
}

fn baking_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MX_DX12_BUNDLES").map_or(true, |v| v != "0"))
}

impl Device {
    pub fn create_render_bundle_encoder(&self, desc: &t::RenderBundleEncoderDescriptor<'_>) -> RenderBundleEncoder {
        let ds = desc.depth_stencil.unwrap_or(t::RenderBundleDepthStencil {
            format: t::TextureFormat::Depth32Float,
            depth_read_only: true,
            stencil_read_only: true,
        });
        RenderBundleEncoder {
            cmds: Vec::new(),
            shared: self.shared.clone(),
            _desc: ds,
        }
    }
}

impl RenderBundleEncoder {
    pub fn finish(self, _label: Option<&str>) -> RenderBundle {
        let baked = if baking_on() { bake(&self.shared, &self.cmds) } else { None };
        RenderBundle(Arc::new(BundleInner { cmds: self.cmds, baked }))
    }
    pub fn set_pipeline(&mut self, p: &RenderPipeline) {
        self.cmds.push(Cmd::Pipeline(p.clone()));
    }
    pub fn set_bind_group(&mut self, group: u32, g: &BindGroup) {
        self.cmds.push(Cmd::BindGroup(group, g.clone()));
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.cmds.push(Cmd::Immediates(offset, data.into()));
    }
    pub fn set_buffer_offset(&mut self, group: u32, binding: u32, offset: u64) {
        self.cmds.push(Cmd::BufferOffset(group, binding, offset));
    }
    pub fn set_vertex_buffer(&mut self, slot: u32, b: &Buffer, offset: u64) {
        self.cmds.push(Cmd::VertexBuffer(slot, b.clone(), offset));
    }
    pub fn set_index_buffer(&mut self, b: &Buffer, offset: u64, format: t::IndexFormat) {
        self.cmds.push(Cmd::IndexBuffer(b.clone(), offset, format));
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.cmds.push(Cmd::Draw(vertices, instances));
    }
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.cmds.push(Cmd::DrawIndexed(indices, base_vertex, instances));
    }
    pub fn draw_indirect(&mut self, args: &Buffer, offset: u64) {
        self.cmds.push(Cmd::DrawIndirect(args.clone(), offset));
    }
    pub fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64) {
        self.cmds.push(Cmd::DrawIndexedIndirect(args.clone(), offset));
    }
}

/// A resource's state need: its subresources (None: all), the state.
type Need = (Track, Option<crate::track::Range>, D3D12_RESOURCE_STATES);
/// A need's key: the resource and its subresources as (mips, layers, planes) bounds.
type NeedKey = (usize, Option<(u32, u32, u32, u32, u32, u32)>);

/// What a bundle being recorded binds: its descriptors, and the states it needs.
struct BundleRec {
    shared: Arc<Shared>,
    blocks: Vec<u32>,
    block_used: u32,
    needs: HashMap<NeedKey, Need>,
    keep: Vec<Box<dyn Any + Send + Sync>>,
    /// It cannot be a bundle (see the module's documentation).
    failed: bool,
}

impl Target for BundleRec {
    fn shared(&self) -> Arc<Shared> {
        self.shared.clone()
    }
    fn table(&mut self, n: u32) -> u32 {
        if self.block_used + n > BLOCK {
            match self.shared.heap.lock().alloc_block() {
                Some(b) => {
                    self.blocks.push(b);
                    self.block_used = 0;
                }
                None => {
                    // (no block free: replayed instead; this one's descriptors are written over
                    // harmlessly, the bundle never runs)
                    self.failed = true;
                    self.block_used = 0;
                }
            }
        }
        let b = *self.blocks.last().expect("a bundle's first block, taken by `bake`");
        let first = self.shared.views.block_start(b) + self.block_used;
        self.block_used += n;
        first
    }
    fn need(&mut self, t: &Track, range: Option<crate::track::Range>, state: D3D12_RESOURCE_STATES) {
        let k = (
            t.raw.as_raw() as usize,
            range.as_ref().map(|r| (r.mips.start, r.mips.end, r.layers.start, r.layers.end, r.planes.start, r.planes.end)),
        );
        match self.needs.get_mut(&k) {
            Some(e) if e.2 == state => {}
            Some(e) if read_only(e.2) && read_only(state) => e.2 |= state,
            Some(_) => self.failed = true,
            None => {
                self.needs.insert(k, (t.clone(), range, state));
            }
        }
    }
    fn keep_boxed(&mut self, b: Box<dyn Any + Send + Sync>) {
        self.keep.push(b);
    }
}

/// Record `cmds` into a Direct3D 12 bundle (None: replayed instead).
fn bake(shared: &Arc<Shared>, cmds: &[Cmd]) -> Option<Baked> {
    if cmds.iter().any(|c| matches!(c, Cmd::DrawIndirect(..) | Cmd::DrawIndexedIndirect(..))) {
        return None;
    }
    let pair = shared.bundle_lists.lock().pop();
    let (alloc, list) = match pair {
        Some((alloc, list)) => {
            // SAFETY: the pair of a dropped bundle, which no list runs any more.
            unsafe {
                alloc.Reset().ok()?;
                list.Reset(&alloc, None).ok()?;
            }
            (alloc, list)
        }
        None => {
            // SAFETY: plain creations.
            unsafe {
                let alloc: ID3D12CommandAllocator = shared.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE).ok()?;
                let list: ID3D12GraphicsCommandList = shared.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, &alloc, None).ok()?;
                (alloc, list)
            }
        }
    };
    // (its first block before anything is recorded: without one it is replayed)
    let first = shared.heap.lock().alloc_block();
    let Some(first) = first else {
        shared.bundle_lists.lock().push((alloc, list));
        return None;
    };
    let mut rec = BundleRec {
        shared: shared.clone(),
        blocks: vec![first],
        block_used: 0,
        needs: HashMap::new(),
        keep: Vec::new(),
        failed: false,
    };
    // SAFETY: the device's two shader-visible heaps, the ones every list sets.
    unsafe { list.SetDescriptorHeaps(&[Some(shared.views.view_heap.clone()), Some(shared.views.sampler_heap.clone())]) };
    let mut state = PassState::default();
    let mut pipeline: Option<RenderPipeline> = None;
    let mut vertex: [Option<(Buffer, u64)>; 8] = Default::default();
    let mut index: Option<(Buffer, u64, t::IndexFormat)> = None;
    let (mut vertex_dirty, mut index_dirty) = (true, true);
    let mut heap_buffers = false;
    let r = Graphics(&list);
    for c in cmds {
        let special = match c {
            Cmd::Pipeline(p) => {
                // SAFETY: a live root signature and pipeline (the bundle's commands keep them).
                unsafe {
                    list.SetGraphicsRootSignature(&p.0.layout.signature);
                    list.SetPipelineState(&p.0.pso);
                    list.IASetPrimitiveTopology(p.0.topology);
                }
                bind_heap(shared, &r, &mut state);
                pipeline = Some(p.clone());
                vertex_dirty = true;
                continue;
            }
            Cmd::BindGroup(g, bg) => {
                assert!(*g <= mx_shader::MAX_ROOT_GROUP, "root bind groups are 0..={}", mx_shader::MAX_ROOT_GROUP);
                state.groups[*g as usize] = Some(bg.clone());
                state.offsets.retain(|o| o.0 != *g);
                state.dirty |= 1 << g;
                continue;
            }
            Cmd::BufferOffset(g, b, o) => {
                match state.offsets.iter_mut().find(|x| x.0 == *g && x.1 == *b) {
                    Some(x) => x.2 = *o,
                    None => state.offsets.push((*g, *b, *o)),
                }
                state.dirty |= 1 << g;
                continue;
            }
            Cmd::Immediates(o, d) => {
                crate::command::set_immediates(&mut state, *o, d);
                continue;
            }
            Cmd::VertexBuffer(s, b, o) => {
                assert!(*s < 8, "vertex buffer slot {s}: mX takes 8");
                vertex[*s as usize] = Some((b.clone(), *o));
                vertex_dirty = true;
                continue;
            }
            Cmd::IndexBuffer(b, o, f) => {
                index = Some((b.clone(), *o, *f));
                index_dirty = true;
                continue;
            }
            Cmd::Draw(v, i) => ([v.start, i.start, 0], false),
            Cmd::DrawIndexed(_, bv, i) => ([*bv as u32, i.start, 0], true),
            Cmd::DrawIndirect(..) | Cmd::DrawIndexedIndirect(..) => unreachable!("left out above"),
        };
        let (special, indexed) = special;
        let p = &pipeline.as_ref().expect("draw without a pipeline").0;
        for (b, _) in vertex.iter().flatten() {
            rec.need(&b.0.track, None, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        }
        if indexed {
            let (b, _, _) = index.as_ref().expect("draw_indexed without an index buffer");
            rec.need(&b.0.track, None, D3D12_RESOURCE_STATE_INDEX_BUFFER);
        }
        if bind_roots(&mut rec, &r, &p.layout, &mut state, &p.label) || p.layout.heap_buffer_writes {
            rec.failed = true;
        }
        heap_buffers |= p.layout.heap_buffers;
        if rec.failed {
            break;
        }
        if vertex_dirty {
            for (i, v) in vertex.iter().enumerate() {
                let Some((b, offset)) = v else { continue };
                let view = D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: b.0.va + offset,
                    SizeInBytes: (b.size() - offset) as u32,
                    StrideInBytes: p.strides[i],
                };
                // SAFETY: a view inside a live buffer (the bundle's commands keep it).
                unsafe { list.IASetVertexBuffers(i as u32, Some(&[view])) };
            }
            vertex_dirty = false;
        }
        if index_dirty && let Some((b, offset, format)) = &index {
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: b.0.va + offset,
                SizeInBytes: (b.size() - offset) as u32,
                Format: conv::index_format(*format),
            };
            // SAFETY: as above.
            unsafe { list.IASetIndexBuffer(Some(&view)) };
            index_dirty = false;
        }
        flush_constants(&r, &p.layout, &mut state, special);
        // SAFETY: a complete pipeline and bindings were set above.
        unsafe {
            match c {
                Cmd::Draw(v, i) if !v.is_empty() && !i.is_empty() => list.DrawInstanced(v.len() as u32, i.len() as u32, v.start, i.start),
                Cmd::DrawIndexed(ix, bv, i) if !ix.is_empty() && !i.is_empty() => list.DrawIndexedInstanced(ix.len() as u32, i.len() as u32, ix.start, *bv, i.start),
                _ => {}
            }
        }
    }
    // SAFETY: closes the bundle recorded above.
    let closed = unsafe { list.Close() }.is_ok();
    let baked = Baked {
        pair: Some((alloc, list)),
        needs: rec.needs.into_values().collect(),
        heap_buffers,
        blocks: std::mem::take(&mut rec.blocks),
        shared: shared.clone(),
        _keep: std::mem::take(&mut rec.keep),
    };
    // (a failed one gives its blocks and list back as it drops)
    (closed && !rec.failed).then_some(baked)
}

impl RenderPass<'_> {
    /// Run bundles. Each starts with no pipeline or bindings, and the pass has none
    /// afterwards (as in WebGPU).
    pub fn execute_bundles<'b>(&mut self, bundles: impl IntoIterator<Item = &'b RenderBundle>) {
        for b in bundles {
            self.reset_state();
            if let Some(baked) = &b.0.baked {
                let rec = &mut self.enc.rec;
                for (track, range, state) in &baked.needs {
                    rec.track.need(track, range.clone(), *state);
                }
                if baked.heap_buffers {
                    need_heap_buffers(rec);
                }
                if rec.uav_dirty {
                    rec.track.uav_barrier();
                    rec.uav_dirty = false;
                }
                rec.barriers();
                let list = rec.list().clone();
                // SAFETY: a closed bundle whose resources are in the states it needs; the list
                // keeps it (and through it everything it names) until the GPU is done.
                unsafe { list.ExecuteBundle(&baked.pair.as_ref().expect("a baked bundle's list").1) };
                rec.keep(b.clone());
                // (what the bundle set stays set on the list: everything is set again)
                self.state.special = None;
                continue;
            }
            for c in b.0.cmds.iter() {
                match c {
                    Cmd::Pipeline(p) => self.set_pipeline(p),
                    Cmd::BindGroup(g, bg) => self.set_bind_group(*g, bg),
                    Cmd::BufferOffset(g, b, o) => self.set_buffer_offset(*g, *b, *o),
                    Cmd::Immediates(o, d) => self.set_immediates(*o, d),
                    Cmd::VertexBuffer(s, buf, o) => self.set_vertex_buffer(*s, buf, *o),
                    Cmd::IndexBuffer(buf, o, f) => self.set_index_buffer(buf, *o, *f),
                    Cmd::Draw(v, i) => self.draw(v.clone(), i.clone()),
                    Cmd::DrawIndexed(ix, bv, i) => self.draw_indexed(ix.clone(), *bv, i.clone()),
                    Cmd::DrawIndirect(buf, o) => self.draw_indirect(buf, *o),
                    Cmd::DrawIndexedIndirect(buf, o) => self.draw_indexed_indirect(buf, *o),
                }
            }
        }
        self.reset_state();
    }
}

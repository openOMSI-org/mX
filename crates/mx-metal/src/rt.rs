//! Acceleration structures for WGSL ray queries.

use std::ops::{Index, IndexMut};
use std::sync::Arc;

use mx_types as t;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::*;

use crate::device::Shared;
use crate::{Buffer, CommandEncoder, Device, Error, Mt, Po, Result};

pub(crate) struct BlasInner {
    pub raw: Mt<Po<dyn MTLAccelerationStructure>>,
    sizes: Vec<t::BlasTriangleGeometrySizeDescriptor>,
    flags: t::AccelerationStructureFlags,
    update_mode: t::AccelerationStructureUpdateMode,
    /// Built at least once (a refit needs a built structure).
    built: std::sync::atomic::AtomicBool,
}

/// Whether a structure made with `flags` and `mode` is refitted rather than rebuilt (wgpu's
/// `PreferUpdate` on a structure that allows updates).
fn refits(flags: t::AccelerationStructureFlags, mode: t::AccelerationStructureUpdateMode) -> bool {
    flags.contains(t::AccelerationStructureFlags::ALLOW_UPDATE) && mode == t::AccelerationStructureUpdateMode::PreferUpdate
}

/// A bottom-level acceleration structure: triangles of one mesh.
#[derive(Clone)]
pub struct Blas(pub(crate) Arc<BlasInner>);

impl std::fmt::Debug for Blas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Blas").field("geometries", &self.0.sizes.len()).finish()
    }
}

/// A top-level acceleration structure: instances of bottom-level ones. Set instances with
/// `tlas[i] = Some(TlasInstance::new(..))`, then build.
pub struct Tlas {
    pub(crate) inner: Arc<TlasShared>,
    instances: Vec<Option<t::TlasInstance<Blas>>>,
    flags: t::AccelerationStructureFlags,
    update_mode: t::AccelerationStructureUpdateMode,
    /// Instances in the last build (a refit needs the same count and structures).
    last_count: parking_lot::Mutex<Option<usize>>,
    /// Its place in the heap's texture table ([`Tlas::index`]).
    slot: u32,
    shared: Arc<Shared>,
}

impl Drop for Tlas {
    fn drop(&mut self) {
        let after = self.shared.release_after();
        self.shared.heap.lock().remove_tlas(self.slot, after);
    }
}

/// The part of a top-level structure bind groups hold on to.
pub(crate) struct TlasShared {
    pub raw: Mt<Po<dyn MTLAccelerationStructure>>,
    /// The structures the last build referenced (kept alive, and made resident wherever the
    /// top-level one is bound).
    pub built_with: parking_lot::Mutex<Arc<Vec<Blas>>>,
}

impl Tlas {
    pub fn max_instances(&self) -> u32 {
        self.instances.len() as u32
    }

    /// The structure's index in the bindless heap: `mx_tlas[i]` in WGSL, with
    /// `@group(7) @binding(11) var mx_tlas: binding_array<acceleration_structure>;`. It shares
    /// the index space of texture views.
    pub fn index(&self) -> u32 {
        self.slot
    }

    /// The `MTLAccelerationStructure`, for interop.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLAccelerationStructure> {
        &self.inner.raw
    }
}

impl Index<usize> for Tlas {
    type Output = Option<t::TlasInstance<Blas>>;
    fn index(&self, i: usize) -> &Self::Output {
        &self.instances[i]
    }
}

impl IndexMut<usize> for Tlas {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.instances[i]
    }
}

fn usage(flags: t::AccelerationStructureFlags) -> MTLAccelerationStructureUsage {
    let mut u = MTLAccelerationStructureUsage::None;
    if flags.contains(t::AccelerationStructureFlags::ALLOW_UPDATE) {
        u |= MTLAccelerationStructureUsage::Refit;
    }
    if flags.contains(t::AccelerationStructureFlags::PREFER_FAST_BUILD) {
        u |= MTLAccelerationStructureUsage::PreferFastBuild;
    }
    u
}

fn vertex_format(f: t::VertexFormat) -> MTLAttributeFormat {
    match f {
        t::VertexFormat::Float32x3 => MTLAttributeFormat::Float3,
        t::VertexFormat::Float32x2 => MTLAttributeFormat::Float2,
        t::VertexFormat::Float16x2 => MTLAttributeFormat::Half2,
        t::VertexFormat::Float16x4 => MTLAttributeFormat::Half4,
        t::VertexFormat::Float32x4 => MTLAttributeFormat::Float4,
        other => panic!("{other:?} is not a vertex format for acceleration structures"),
    }
}

/// The triangle geometry descriptors of a bottom-level structure. With `data`, they point
/// at the buffers to build from; without, they only carry the sizes.
fn primitive_descriptor(
    sizes: &[t::BlasTriangleGeometrySizeDescriptor],
    data: Option<&[t::BlasTriangleGeometry<'_, Buffer>]>,
    flags: t::AccelerationStructureFlags,
) -> Retained<MTLPrimitiveAccelerationStructureDescriptor> {
    let geoms: Vec<Retained<MTLAccelerationStructureGeometryDescriptor>> = sizes
        .iter()
        .enumerate()
        .map(|(i, size)| {
            let g = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
            g.setOpaque(size.flags.contains(t::AccelerationStructureGeometryFlags::OPAQUE));
            g.setVertexFormat(vertex_format(size.vertex_format));
            let triangles = size.index_count.unwrap_or(size.vertex_count) / 3;
            // SAFETY: plain setters; offsets point inside the buffers (checked at build).
            unsafe {
                g.setTriangleCount(triangles as usize);
                if let Some(f) = size.index_format {
                    g.setIndexType(crate::conv::index_type(f));
                }
                if let Some(d) = data.and_then(|d| d.get(i)) {
                    g.setVertexBuffer(Some(&d.vertex_buffer.0.raw));
                    g.setVertexStride(d.vertex_stride as usize);
                    g.setVertexBufferOffset((d.first_vertex as u64 * d.vertex_stride) as usize);
                    if let (Some(ib), Some(f)) = (d.index_buffer, size.index_format) {
                        g.setIndexBuffer(Some(&ib.0.raw));
                        g.setIndexBufferOffset((d.first_index.unwrap_or(0) as u64 * f.size()) as usize);
                    }
                } else {
                    g.setVertexStride(size.vertex_format.size() as usize);
                }
            }
            Retained::into_super(g)
        })
        .collect();
    let d = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
    d.setGeometryDescriptors(Some(&NSArray::from_retained_slice(&geoms)));
    d.setUsage(usage(flags));
    d
}

impl Device {
    pub fn create_blas(&self, desc: &t::CreateBlasDescriptor<'_>, sizes: &[t::BlasTriangleGeometrySizeDescriptor]) -> Result<Blas> {
        if self.shared.bound {
            return Err(Error::Unsupported("ray queries on a Metal 2 GPU (the bound tier)".into()));
        }
        if !self.shared.device.supportsRaytracing() {
            return Err(Error::Unsupported("ray tracing".into()));
        }
        let d = primitive_descriptor(sizes, None, desc.flags);
        let s = self.shared.device.accelerationStructureSizesWithDescriptor(&d);
        let raw = self
            .shared
            .device
            .newAccelerationStructureWithSize(s.accelerationStructureSize)
            .ok_or_else(|| Error::OutOfMemory("acceleration structure".into()))?;
        if let Some(l) = desc.label {
            raw.setLabel(Some(&crate::ns(l)));
        }
        Ok(Blas(Arc::new(BlasInner {
            raw: Mt(raw),
            sizes: sizes.to_vec(),
            flags: desc.flags,
            update_mode: desc.update_mode,
            built: std::sync::atomic::AtomicBool::new(false),
        })))
    }

    pub fn create_tlas(&self, desc: &t::CreateTlasDescriptor<'_>) -> Result<Tlas> {
        if self.shared.bound {
            return Err(Error::Unsupported("ray queries on a Metal 2 GPU (the bound tier)".into()));
        }
        if !self.shared.device.supportsRaytracing() {
            return Err(Error::Unsupported("ray tracing".into()));
        }
        let d = instance_descriptor(desc.max_instances, None, &[], desc.flags);
        let s = self.shared.device.accelerationStructureSizesWithDescriptor(&d);
        let raw = self
            .shared
            .device
            .newAccelerationStructureWithSize(s.accelerationStructureSize)
            .ok_or_else(|| Error::OutOfMemory("acceleration structure".into()))?;
        if let Some(l) = desc.label {
            raw.setLabel(Some(&crate::ns(l)));
        }
        let inner = Arc::new(TlasShared {
            raw: Mt(raw),
            built_with: parking_lot::Mutex::new(Arc::new(Vec::new())),
        });
        let slot = self.shared.heap.lock().add_tlas(&inner)?;
        Ok(Tlas {
            inner,
            slot,
            instances: vec![None; desc.max_instances as usize],
            flags: desc.flags,
            update_mode: desc.update_mode,
            last_count: parking_lot::Mutex::new(None),
            shared: self.shared.clone(),
        })
    }
}

fn instance_descriptor(count: u32, buffer: Option<&ProtocolObject<dyn MTLBuffer>>, blases: &[Blas], flags: t::AccelerationStructureFlags) -> Retained<MTLInstanceAccelerationStructureDescriptor> {
    let d = MTLInstanceAccelerationStructureDescriptor::descriptor();
    d.setInstanceCount(count as usize);
    d.setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::UserID);
    // SAFETY: the stride of the user-ID descriptor type set above.
    unsafe { d.setInstanceDescriptorStride(std::mem::size_of::<MTLAccelerationStructureUserIDInstanceDescriptor>()) };
    if let Some(b) = buffer {
        d.setInstanceDescriptorBuffer(Some(b));
    }
    let list: Vec<Po<dyn MTLAccelerationStructure>> = blases.iter().map(|b| b.0.raw.0.clone()).collect();
    d.setInstancedAccelerationStructures(Some(&NSArray::from_retained_slice(&list)));
    d.setUsage(usage(flags));
    d
}

/// Row-major 3x4 (WGSL, wgpu) to Metal's four packed columns, as wgpu-hal does.
fn packed(m: &[f32; 12]) -> MTLPackedFloat4x3 {
    let col = |c: usize| MTLPackedFloat3 { x: m[c], y: m[4 + c], z: m[8 + c] };
    MTLPackedFloat4x3 {
        columns: [col(0), col(1), col(2), col(3)],
    }
}

impl CommandEncoder {
    /// Build bottom-level structures from their geometry, then top-level ones from their
    /// instances. A structure made with `ALLOW_UPDATE` and `PreferUpdate` is refitted once it
    /// has been built (the same triangles moved; for a top-level one, the same number of
    /// instances of the same structures, transforms changed).
    pub fn build_acceleration_structures<'a>(&mut self, blas: impl IntoIterator<Item = &'a t::BlasBuildEntry<'a, Blas, Buffer>>, tlas: impl IntoIterator<Item = &'a Tlas>) {
        let device = &self.shared.device;
        let enc = self.raw.accelerationStructureCommandEncoder().expect("Metal returned no acceleration structure encoder");
        let scratch = |size: usize| {
            device
                .newBufferWithLength_options(size.max(16), MTLResourceOptions::StorageModePrivate)
                .expect("out of memory for an acceleration structure scratch buffer")
        };
        let mut keep: Vec<Mt<Po<dyn MTLBuffer>>> = Vec::new();
        for entry in blas {
            let b = &entry.blas.0;
            assert_eq!(entry.geometry.len(), b.sizes.len(), "a BLAS build needs one geometry per size it was created with");
            let d = primitive_descriptor(&b.sizes, Some(&entry.geometry), b.flags);
            let s = device.accelerationStructureSizesWithDescriptor(&d);
            let refit = refits(b.flags, b.update_mode) && b.built.load(std::sync::atomic::Ordering::Acquire);
            let sb = scratch(if refit { s.refitScratchBufferSize } else { s.buildScratchBufferSize });
            // SAFETY: descriptor, structure and scratch buffer sized for each other.
            unsafe {
                if refit {
                    enc.refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset(&b.raw, &d, None, Some(&sb), 0);
                } else {
                    enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(&b.raw, &d, &sb, 0);
                }
            }
            b.built.store(true, std::sync::atomic::Ordering::Release);
            keep.push(Mt(sb));
        }
        // (the instances of a top-level structure read the bottom-level ones built above:
        // Metal orders builds within one encoder)
        for t in tlas {
            let mut blases: Vec<Blas> = Vec::new();
            let mut descs: Vec<MTLAccelerationStructureUserIDInstanceDescriptor> = Vec::new();
            for inst in t.instances.iter().flatten() {
                let index = match blases.iter().position(|b| Arc::ptr_eq(&b.0, &inst.blas.0)) {
                    Some(i) => i,
                    None => {
                        blases.push(inst.blas.clone());
                        blases.len() - 1
                    }
                };
                descs.push(MTLAccelerationStructureUserIDInstanceDescriptor {
                    transformationMatrix: packed(&inst.transform),
                    options: MTLAccelerationStructureInstanceOptions::None,
                    mask: inst.mask as u32,
                    intersectionFunctionTableOffset: 0,
                    accelerationStructureIndex: index as u32,
                    userID: inst.custom_data & 0x00FF_FFFF,
                });
            }
            let bytes = std::mem::size_of_val(descs.as_slice());
            let ib = t
                .shared
                .staging_buffer(
                    // SAFETY: plain-old-data `#[repr(C)]` structs viewed as bytes.
                    unsafe { std::slice::from_raw_parts(descs.as_ptr().cast::<u8>(), bytes) },
                )
                .expect("out of memory for TLAS instances");
            let d = instance_descriptor(descs.len() as u32, Some(&ib), &blases, t.flags);
            let s = device.accelerationStructureSizesWithDescriptor(&d);
            let same = {
                let prev = t.inner.built_with.lock();
                *t.last_count.lock() == Some(descs.len()) && prev.len() == blases.len() && prev.iter().zip(&blases).all(|(a, b)| Arc::ptr_eq(&a.0, &b.0))
            };
            let refit = refits(t.flags, t.update_mode) && same;
            let sb = scratch(if refit { s.refitScratchBufferSize } else { s.buildScratchBufferSize });
            // SAFETY: as above; the instance buffer holds `descs.len()` descriptors.
            unsafe {
                if refit {
                    enc.refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset(&t.inner.raw, &d, None, Some(&sb), 0);
                } else {
                    enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(&t.inner.raw, &d, &sb, 0);
                }
            }
            *t.last_count.lock() = Some(descs.len());
            keep.push(ib);
            keep.push(Mt(sb));
            *t.inner.built_with.lock() = Arc::new(blases);
        }
        enc.endEncoding();
        drop(keep); // (Metal retains the buffers the encoder used)
    }
}

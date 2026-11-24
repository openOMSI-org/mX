//! Acceleration structures for WGSL ray queries (`VK_KHR_acceleration_structure`).

use std::ops::{Index, IndexMut};
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocation;
use mx_types as t;

use crate::device::{Garbage, Shared};
use crate::{Buffer, CommandEncoder, Device, Error, Result, vk_err};

/// An acceleration structure and the buffer it lives in.
pub(crate) struct Accel {
    pub raw: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    alloc: Option<Allocation>,
    /// The structure's device address (what TLAS instances refer to).
    pub address: u64,
    shared: Arc<Shared>,
}

impl Drop for Accel {
    fn drop(&mut self) {
        if let Some(a) = self.alloc.take() {
            self.shared.retire(Garbage::Accel(self.raw, self.buffer, a));
        }
    }
}

fn create_accel(shared: &Arc<Shared>, ty: vk::AccelerationStructureTypeKHR, size: u64, label: Option<&str>) -> Result<Accel> {
    let f = shared.fns.accel.as_ref().ok_or_else(|| Error::Unsupported("ray queries".into()))?;
    let (buffer, alloc) = shared.create_raw_buffer(
        size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        label.unwrap_or("acceleration structure"),
    )?;
    let info = vk::AccelerationStructureCreateInfoKHR::default().buffer(buffer).size(size).ty(ty);
    // SAFETY: the buffer is live and large enough.
    let raw = match unsafe { f.create_acceleration_structure(&info, None) } {
        Ok(r) => r,
        Err(e) => {
            shared.destroy(Garbage::Buffer(buffer, alloc));
            return Err(vk_err("acceleration structure", e));
        }
    };
    if let Some(l) = label {
        shared.name(raw, l);
    }
    // SAFETY: a live structure.
    let address = unsafe { f.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(raw)) };
    Ok(Accel {
        raw,
        buffer,
        alloc: Some(alloc),
        address,
        shared: shared.clone(),
    })
}

pub(crate) struct BlasInner {
    pub accel: Accel,
    sizes: Vec<t::BlasTriangleGeometrySizeDescriptor>,
    flags: t::AccelerationStructureFlags,
    update_mode: t::AccelerationStructureUpdateMode,
    /// Built at least once (an update needs a built structure).
    built: std::sync::atomic::AtomicBool,
}

/// Whether a structure made with `flags` and `mode` is updated in place rather than rebuilt
/// (wgpu's `PreferUpdate` on a structure that allows updates).
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

impl Blas {
    /// The `VkAccelerationStructureKHR`, for interop.
    pub fn raw(&self) -> vk::AccelerationStructureKHR {
        self.0.accel.raw
    }
}

/// A top-level acceleration structure: instances of bottom-level ones. Set instances with
/// `tlas[i] = Some(TlasInstance::new(..))`, then build.
pub struct Tlas {
    pub(crate) inner: Arc<TlasShared>,
    instances: Vec<Option<t::TlasInstance<Blas>>>,
    flags: t::AccelerationStructureFlags,
    update_mode: t::AccelerationStructureUpdateMode,
    /// Instances in the last build (an update needs the same count and structures).
    last_count: parking_lot::Mutex<Option<u32>>,
    /// Its place in the heap ([`Tlas::index`]).
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
    pub raw: vk::AccelerationStructureKHR,
    _accel: Accel,
    /// The structures the last build referenced (kept alive with it).
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

    /// The `VkAccelerationStructureKHR`, for interop.
    pub fn raw(&self) -> vk::AccelerationStructureKHR {
        self.inner.raw
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

fn build_flags(flags: t::AccelerationStructureFlags) -> vk::BuildAccelerationStructureFlagsKHR {
    use t::AccelerationStructureFlags as F;
    use vk::BuildAccelerationStructureFlagsKHR as V;
    let mut v = V::empty();
    for (f, b) in [
        (F::ALLOW_UPDATE, V::ALLOW_UPDATE),
        (F::ALLOW_COMPACTION, V::ALLOW_COMPACTION),
        (F::PREFER_FAST_TRACE, V::PREFER_FAST_TRACE),
        (F::PREFER_FAST_BUILD, V::PREFER_FAST_BUILD),
        (F::LOW_MEMORY, V::LOW_MEMORY),
    ] {
        if flags.contains(f) {
            v |= b;
        }
    }
    v
}

fn vertex_format(f: t::VertexFormat) -> vk::Format {
    match f {
        t::VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        t::VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        t::VertexFormat::Float16x2 => vk::Format::R16G16_SFLOAT,
        t::VertexFormat::Float16x4 => vk::Format::R16G16B16A16_SFLOAT,
        other => panic!("{other:?} is not a vertex format for acceleration structures"),
    }
}

/// The triangle geometries of a bottom-level structure (with `data`: pointing at the
/// buffers to build from) and their primitive counts.
fn triangles(shared: &Shared, sizes: &[t::BlasTriangleGeometrySizeDescriptor], data: Option<&[t::BlasTriangleGeometry<'_, Buffer>]>) -> (Vec<vk::AccelerationStructureGeometryKHR<'static>>, Vec<u32>) {
    let mut geoms = Vec::new();
    let mut counts = Vec::new();
    for (i, size) in sizes.iter().enumerate() {
        let mut tri = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vertex_format(size.vertex_format))
            .vertex_stride(size.vertex_format.size())
            .max_vertex(size.vertex_count.saturating_sub(1))
            .index_type(size.index_format.map_or(vk::IndexType::NONE_KHR, crate::conv::index_type));
        if let Some(d) = data.and_then(|d| d.get(i)) {
            tri = tri.vertex_stride(d.vertex_stride).vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: shared.buffer_address(d.vertex_buffer.0.raw) + d.first_vertex as u64 * d.vertex_stride,
            });
            if let (Some(ib), Some(f)) = (d.index_buffer, size.index_format) {
                tri = tri.index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: shared.buffer_address(ib.0.raw) + d.first_index.unwrap_or(0) as u64 * f.size(),
                });
            }
        }
        let mut flags = vk::GeometryFlagsKHR::empty();
        if size.flags.contains(t::AccelerationStructureGeometryFlags::OPAQUE) {
            flags |= vk::GeometryFlagsKHR::OPAQUE;
        }
        if size.flags.contains(t::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION) {
            flags |= vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION;
        }
        geoms.push(
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: tri })
                .flags(flags),
        );
        counts.push(size.index_count.unwrap_or(size.vertex_count) / 3);
    }
    (geoms, counts)
}

fn instance_geometry(address: u64) -> vk::AccelerationStructureGeometryKHR<'static> {
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR { device_address: address }),
        })
}

fn build_sizes(
    shared: &Shared,
    ty: vk::AccelerationStructureTypeKHR,
    flags: t::AccelerationStructureFlags,
    geoms: &[vk::AccelerationStructureGeometryKHR<'_>],
    counts: &[u32],
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let f = shared.fns.accel.as_ref().expect("ray queries enabled");
    let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(build_flags(flags))
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geoms);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: a plain query.
    unsafe { f.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &info, counts, &mut sizes) };
    sizes
}

impl Device {
    pub fn create_blas(&self, desc: &t::CreateBlasDescriptor<'_>, sizes: &[t::BlasTriangleGeometrySizeDescriptor]) -> Result<Blas> {
        let shared = &self.shared;
        if !shared.caps.ray_query {
            return Err(Error::Unsupported("ray queries".into()));
        }
        let (geoms, counts) = triangles(shared, sizes, None);
        let s = build_sizes(shared, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, desc.flags, &geoms, &counts);
        let accel = create_accel(shared, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, s.acceleration_structure_size, desc.label)?;
        Ok(Blas(Arc::new(BlasInner {
            accel,
            sizes: sizes.to_vec(),
            flags: desc.flags,
            update_mode: desc.update_mode,
            built: std::sync::atomic::AtomicBool::new(false),
        })))
    }

    pub fn create_tlas(&self, desc: &t::CreateTlasDescriptor<'_>) -> Result<Tlas> {
        let shared = &self.shared;
        if !shared.caps.ray_query {
            return Err(Error::Unsupported("ray queries".into()));
        }
        let geoms = [instance_geometry(0)];
        let s = build_sizes(shared, vk::AccelerationStructureTypeKHR::TOP_LEVEL, desc.flags, &geoms, &[desc.max_instances]);
        let accel = create_accel(shared, vk::AccelerationStructureTypeKHR::TOP_LEVEL, s.acceleration_structure_size, desc.label)?;
        let inner = Arc::new(TlasShared {
            raw: accel.raw,
            _accel: accel,
            built_with: parking_lot::Mutex::new(Arc::new(Vec::new())),
        });
        let slot = shared.heap.lock().add_tlas(&shared.raw, inner.raw)?;
        Ok(Tlas {
            inner,
            slot,
            shared: shared.clone(),
            instances: vec![None; desc.max_instances as usize],
            flags: desc.flags,
            update_mode: desc.update_mode,
            last_count: parking_lot::Mutex::new(None),
        })
    }
}

/// Vulkan's `VkAccelerationStructureInstanceKHR`, laid out as wgpu-hal writes it.
#[repr(C)]
#[derive(Clone, Copy)]
struct RawInstance {
    transform: [f32; 12],
    custom_data_and_mask: u32,
    shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

impl CommandEncoder {
    /// A scratch buffer of `size` bytes for a build, and its aligned device address.
    fn scratch(&mut self, size: u64) -> u64 {
        let align = self.shared.caps.scratch_alignment;
        let (buf, alloc) = self
            .shared
            .create_raw_buffer(
                size + align,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
                "acceleration structure scratch",
            )
            .expect("out of memory for an acceleration structure scratch buffer");
        let address = self.shared.buffer_address(buf).next_multiple_of(align);
        self.rec.temp.push(Garbage::Buffer(buf, alloc));
        address
    }

    /// Build bottom-level structures from their geometry, then top-level ones from their
    /// instances. A structure made with `ALLOW_UPDATE` and `PreferUpdate` is updated in place
    /// once it has been built (for a top-level one: the same number of instances of the same
    /// structures).
    pub fn build_acceleration_structures<'a>(&mut self, blas: impl IntoIterator<Item = &'a t::BlasBuildEntry<'a, Blas, Buffer>>, tlas: impl IntoIterator<Item = &'a Tlas>) {
        let shared = self.shared.clone();
        let f = shared.fns.accel.as_ref().expect("ray queries are not supported on this device");
        self.before(crate::command::Op::ANY);
        for entry in blas {
            let b = &entry.blas.0;
            assert_eq!(entry.geometry.len(), b.sizes.len(), "a BLAS build needs one geometry per size it was created with");
            let (geoms, counts) = triangles(&shared, &b.sizes, Some(&entry.geometry));
            let s = build_sizes(&shared, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, b.flags, &geoms, &counts);
            let update = refits(b.flags, b.update_mode) && b.built.load(std::sync::atomic::Ordering::Acquire);
            let scratch = self.scratch(if update { s.update_scratch_size } else { s.build_scratch_size });
            let info = [vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(build_flags(b.flags))
                .mode(if update {
                    vk::BuildAccelerationStructureModeKHR::UPDATE
                } else {
                    vk::BuildAccelerationStructureModeKHR::BUILD
                })
                .src_acceleration_structure(if update { b.accel.raw } else { vk::AccelerationStructureKHR::null() })
                .dst_acceleration_structure(b.accel.raw)
                .geometries(&geoms)
                .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch })];
            let ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = counts.iter().map(|&n| vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(n)).collect();
            // SAFETY: structure, geometry and scratch sized for each other.
            unsafe { f.cmd_build_acceleration_structures(self.raw, &info, &[&ranges]) };
            b.built.store(true, std::sync::atomic::Ordering::Release);
            self.keep(entry.blas);
            for g in &entry.geometry {
                self.keep(g.vertex_buffer);
                if let Some(ib) = g.index_buffer {
                    self.keep(ib);
                }
            }
        }
        // the instances read the bottom-level structures built above
        self.shared.full_barrier(self.raw);
        for t in tlas {
            let mut blases: Vec<Blas> = Vec::new();
            let mut raws: Vec<RawInstance> = Vec::new();
            for inst in t.instances.iter().flatten() {
                if !blases.iter().any(|b| Arc::ptr_eq(&b.0, &inst.blas.0)) {
                    blases.push(inst.blas.clone());
                }
                raws.push(RawInstance {
                    transform: inst.transform,
                    custom_data_and_mask: (inst.custom_data & 0x00FF_FFFF) | (u32::from(inst.mask) << 24),
                    shader_binding_table_record_offset_and_flags: 0,
                    acceleration_structure_reference: inst.blas.0.accel.address,
                });
            }
            // SAFETY: plain-old-data `#[repr(C)]` structs viewed as bytes.
            let bytes = unsafe { std::slice::from_raw_parts(raws.as_ptr().cast::<u8>(), std::mem::size_of_val(raws.as_slice())) };
            let (ib, alloc) = shared
                .staging_buffer(
                    bytes,
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .expect("out of memory for TLAS instances");
            let geoms = [instance_geometry(shared.buffer_address(ib))];
            let count = raws.len() as u32;
            let s = build_sizes(&shared, vk::AccelerationStructureTypeKHR::TOP_LEVEL, t.flags, &geoms, &[count]);
            let same = {
                let prev = t.inner.built_with.lock();
                *t.last_count.lock() == Some(count) && prev.len() == blases.len() && prev.iter().zip(&blases).all(|(a, b)| Arc::ptr_eq(&a.0, &b.0))
            };
            let update = refits(t.flags, t.update_mode) && same;
            let scratch = self.scratch(if update { s.update_scratch_size } else { s.build_scratch_size });
            let info = [vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(build_flags(t.flags))
                .mode(if update {
                    vk::BuildAccelerationStructureModeKHR::UPDATE
                } else {
                    vk::BuildAccelerationStructureModeKHR::BUILD
                })
                .src_acceleration_structure(if update { t.inner.raw } else { vk::AccelerationStructureKHR::null() })
                .dst_acceleration_structure(t.inner.raw)
                .geometries(&geoms)
                .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch })];
            let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)];
            // SAFETY: as above; the instance buffer holds `count` instances.
            unsafe { f.cmd_build_acceleration_structures(self.raw, &info, &[&ranges]) };
            self.rec.temp.push(Garbage::Buffer(ib, alloc));
            self.keep(&blases);
            self.rec.keep.push(Box::new(t.inner.clone()));
            *t.last_count.lock() = Some(count);
            *t.inner.built_with.lock() = Arc::new(blases);
        }
        self.after(crate::command::Op::ANY);
    }
}

//! Acceleration structures for WGSL ray queries (DXR 1.1 inline ray tracing).

use std::ops::{Index, IndexMut};
use std::sync::Arc;

use mx_types as t;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;
use windows::core::Interface;

use crate::device::Shared;
use crate::resource::raw_buffer;
use crate::{Buffer, CommandEncoder, Device, Error, Result, conv};

pub(crate) struct BlasInner {
    pub raw: ID3D12Resource,
    pub va: u64,
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
    /// The acceleration structure's buffer, for interop.
    pub fn raw(&self) -> &ID3D12Resource {
        &self.0.raw
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
    last_count: parking_lot::Mutex<Option<usize>>,
    /// Its texture slot in the heap ([`Tlas::index`]).
    slot: u32,
    shared: Arc<Shared>,
}

impl Drop for Tlas {
    fn drop(&mut self) {
        let after = self.shared.release_after();
        self.shared.heap.lock().release_tlas(self.slot, after, self.inner.clone());
    }
}

/// The part of a top-level structure bind groups hold on to.
pub(crate) struct TlasShared {
    pub raw: ID3D12Resource,
    pub va: u64,
    /// The structures the last build referenced (kept alive with it).
    pub built_with: parking_lot::Mutex<Arc<Vec<Blas>>>,
}

impl Tlas {
    pub fn max_instances(&self) -> u32 {
        self.instances.len() as u32
    }

    /// The structure's index in the bindless heap: `mx_tlas[i]` in WGSL, with
    /// `@group(7) @binding(11) var mx_tlas: binding_array<acceleration_structure>;`. It shares
    /// the index space of texture views (its SRV is in a texture slot).
    pub fn index(&self) -> u32 {
        self.slot
    }

    /// The acceleration structure's buffer, for interop.
    pub fn raw(&self) -> &ID3D12Resource {
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

/// Triangle geometry descriptions; with `data`, pointing at the buffers to build from.
fn geometry(sizes: &[t::BlasTriangleGeometrySizeDescriptor], data: Option<&[t::BlasTriangleGeometry<'_, Buffer>]>) -> Vec<D3D12_RAYTRACING_GEOMETRY_DESC> {
    sizes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let d = data.and_then(|d| d.get(i));
            let (vertex, stride, index) = match d {
                Some(d) => (
                    d.vertex_buffer.0.va + d.first_vertex as u64 * d.vertex_stride,
                    d.vertex_stride,
                    match (d.index_buffer, s.index_format) {
                        (Some(ib), Some(f)) => ib.0.va + d.first_index.unwrap_or(0) as u64 * f.size(),
                        _ => 0,
                    },
                ),
                None => (0, s.vertex_format.size(), 0),
            };
            D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: conv::as_geometry_flags(s.flags),
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: s.index_format.map_or(DXGI_FORMAT_UNKNOWN, conv::index_format),
                        VertexFormat: conv::vertex_format(s.vertex_format),
                        IndexCount: s.index_count.unwrap_or(0),
                        VertexCount: s.vertex_count,
                        IndexBuffer: index,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vertex,
                            StrideInBytes: stride,
                        },
                    },
                },
            }
        })
        .collect()
}

fn blas_inputs(geoms: &[D3D12_RAYTRACING_GEOMETRY_DESC], flags: t::AccelerationStructureFlags) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: conv::as_build_flags(flags),
        NumDescs: geoms.len() as u32,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: geoms.as_ptr() },
    }
}

fn tlas_inputs(count: u32, instances: u64, flags: t::AccelerationStructureFlags) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: conv::as_build_flags(flags),
        NumDescs: count,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { InstanceDescs: instances },
    }
}

impl Shared {
    fn device5(&self) -> Result<ID3D12Device5> {
        if !self.ray_query {
            return Err(Error::Unsupported("ray queries (DXR 1.1)".into()));
        }
        self.device.cast::<ID3D12Device5>().map_err(|_| Error::Unsupported("ray tracing (ID3D12Device5)".into()))
    }

    fn prebuild(&self, inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS) -> Result<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO> {
        let d5 = self.device5()?;
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: inputs whose pointers live across the call; a valid out-pointer.
        unsafe { d5.GetRaytracingAccelerationStructurePrebuildInfo(inputs, &mut info) };
        Ok(info)
    }

    /// A buffer for an acceleration structure (in its own resource state for good).
    fn as_buffer(&self, size: u64, label: &str) -> Result<(ID3D12Resource, u64)> {
        let raw = raw_buffer(
            &self.device,
            size.next_multiple_of(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT as u64),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            label,
        )?;
        // SAFETY: a plain getter.
        let va = unsafe { raw.GetGPUVirtualAddress() };
        Ok((raw, va))
    }
}

impl Device {
    pub fn create_blas(&self, desc: &t::CreateBlasDescriptor<'_>, sizes: &[t::BlasTriangleGeometrySizeDescriptor]) -> Result<Blas> {
        let geoms = geometry(sizes, None);
        let info = self.shared.prebuild(&blas_inputs(&geoms, desc.flags))?;
        let (raw, va) = self.shared.as_buffer(info.ResultDataMaxSizeInBytes, desc.label.unwrap_or("BLAS"))?;
        Ok(Blas(Arc::new(BlasInner {
            raw,
            va,
            sizes: sizes.to_vec(),
            flags: desc.flags,
            update_mode: desc.update_mode,
            built: std::sync::atomic::AtomicBool::new(false),
        })))
    }

    pub fn create_tlas(&self, desc: &t::CreateTlasDescriptor<'_>) -> Result<Tlas> {
        let info = self.shared.prebuild(&tlas_inputs(desc.max_instances, 0, desc.flags))?;
        let (raw, va) = self.shared.as_buffer(info.ResultDataMaxSizeInBytes, desc.label.unwrap_or("TLAS"))?;
        let slot = self.shared.heap.lock().alloc_texture()?;
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: va },
            },
        };
        // SAFETY: an acceleration structure's SRV (no resource: its address) into a fresh slot.
        unsafe { self.shared.device.CreateShaderResourceView(None, Some(&srv), self.shared.views.view_cpu(slot)) };
        Ok(Tlas {
            inner: Arc::new(TlasShared {
                raw,
                va,
                built_with: parking_lot::Mutex::new(Arc::new(Vec::new())),
            }),
            slot,
            shared: self.shared.clone(),
            instances: vec![None; desc.max_instances as usize],
            flags: desc.flags,
            update_mode: desc.update_mode,
            last_count: parking_lot::Mutex::new(None),
        })
    }
}

impl CommandEncoder {
    /// Build bottom-level structures from their geometry, then top-level ones from their
    /// instances. A structure made with `ALLOW_UPDATE` and `PreferUpdate` is updated in place
    /// once it has been built (for a top-level one: the same number of instances of the same
    /// structures).
    pub fn build_acceleration_structures<'a>(&mut self, blas: impl IntoIterator<Item = &'a t::BlasBuildEntry<'a, Blas, Buffer>>, tlas: impl IntoIterator<Item = &'a Tlas>) {
        let rec = &mut self.rec;
        let shared = rec.shared.clone();
        let list4: ID3D12GraphicsCommandList4 = rec.list().cast().expect("ray tracing needs ID3D12GraphicsCommandList4");
        let scratch = |size: u64| {
            raw_buffer(
                &shared.device,
                size.max(256),
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                "mX AS scratch",
            )
            .expect("out of memory for acceleration structure scratch")
        };
        let mut built_any = false;
        for entry in blas {
            let b = &entry.blas.0;
            assert_eq!(entry.geometry.len(), b.sizes.len(), "a BLAS build needs one geometry per size it was created with");
            for g in &entry.geometry {
                rec.track.need(&g.vertex_buffer.0.track, None, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if let Some(ib) = g.index_buffer {
                    rec.track.need(&ib.0.track, None, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                }
            }
            rec.barriers();
            let geoms = geometry(&b.sizes, Some(&entry.geometry));
            let mut inputs = blas_inputs(&geoms, b.flags);
            let info = shared.prebuild(&inputs).expect("ray tracing is supported (a BLAS exists)");
            let update = refits(b.flags, b.update_mode) && b.built.load(std::sync::atomic::Ordering::Acquire);
            if update {
                inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            }
            let sb = scratch(if update { info.UpdateScratchDataSizeInBytes } else { info.ScratchDataSizeInBytes });
            let d = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: b.va,
                Inputs: inputs,
                SourceAccelerationStructureData: if update { b.va } else { 0 },
                // SAFETY: a plain getter.
                ScratchAccelerationStructureData: unsafe { sb.GetGPUVirtualAddress() },
            };
            // SAFETY: destination, scratch and inputs sized by the prebuild info.
            unsafe { list4.BuildRaytracingAccelerationStructure(&d, None) };
            b.built.store(true, std::sync::atomic::Ordering::Release);
            rec.keep(sb);
            rec.keep(entry.blas.clone());
            built_any = true;
        }
        if built_any {
            // (the top-level builds read the bottom-level ones)
            rec.track.uav_barrier();
            rec.barriers();
        }
        for tl in tlas {
            let mut blases: Vec<Blas> = Vec::new();
            let mut descs: Vec<D3D12_RAYTRACING_INSTANCE_DESC> = Vec::new();
            for inst in tl.instances.iter().flatten() {
                if !blases.iter().any(|b| Arc::ptr_eq(&b.0, &inst.blas.0)) {
                    blases.push(inst.blas.clone());
                }
                // wgpu-hal's instance layout: the row-major 3×4 transform as it is, 24-bit
                // custom data and the 8-bit mask, no hit group or flags
                descs.push(D3D12_RAYTRACING_INSTANCE_DESC {
                    Transform: inst.transform,
                    _bitfield1: (inst.custom_data & 0x00FF_FFFF) | (u32::from(inst.mask) << 24),
                    _bitfield2: 0,
                    AccelerationStructure: inst.blas.0.va,
                });
            }
            // SAFETY: plain-old-data `#[repr(C)]` structs viewed as bytes.
            let bytes = unsafe { std::slice::from_raw_parts(descs.as_ptr().cast::<u8>(), std::mem::size_of_val(descs.as_slice())) };
            let ib = crate::resource::upload_buffer(&shared.device, if bytes.is_empty() { &[0; 64] } else { bytes }, "mX TLAS instances").expect("out of memory for TLAS instances");
            // SAFETY: a plain getter.
            let mut inputs = tlas_inputs(descs.len() as u32, unsafe { ib.GetGPUVirtualAddress() }, tl.flags);
            let info = shared.prebuild(&inputs).expect("ray tracing is supported (a TLAS exists)");
            let same = {
                let prev = tl.inner.built_with.lock();
                *tl.last_count.lock() == Some(descs.len()) && prev.len() == blases.len() && prev.iter().zip(&blases).all(|(a, b)| Arc::ptr_eq(&a.0, &b.0))
            };
            let update = refits(tl.flags, tl.update_mode) && same;
            if update {
                inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            }
            let sb = scratch(if update { info.UpdateScratchDataSizeInBytes } else { info.ScratchDataSizeInBytes });
            let d = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: tl.inner.va,
                Inputs: inputs,
                SourceAccelerationStructureData: if update { tl.inner.va } else { 0 },
                // SAFETY: a plain getter.
                ScratchAccelerationStructureData: unsafe { sb.GetGPUVirtualAddress() },
            };
            // SAFETY: as above; the instance buffer holds `descs.len()` 16-byte-aligned
            // descriptions (a committed resource starts 64 KiB aligned).
            unsafe { list4.BuildRaytracingAccelerationStructure(&d, None) };
            rec.keep((ib, sb));
            rec.keep(tl.inner.clone());
            *tl.last_count.lock() = Some(descs.len());
            *tl.inner.built_with.lock() = Arc::new(blases);
        }
        // (ray queries after this read the structures)
        rec.track.uav_barrier();
        rec.barriers();
    }
}

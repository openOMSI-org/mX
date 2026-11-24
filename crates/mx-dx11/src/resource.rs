//! Buffers, textures, views, samplers, bind groups and queue writes.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use mx_types as t;
use parking_lot::Mutex;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::device::{Ctx, Shared};
use crate::heap::BufferSlot;
use crate::{Device, Error, Queue, Result, conv, hr};

// ---------------------------------------------------------------------------------------------
// Buffers

/// Shader views of a range of a storage buffer, made on first use.
#[derive(Clone, Default)]
pub(crate) struct RangeViews {
    pub srv: Option<ID3D11ShaderResourceView>,
    pub uav: Option<ID3D11UnorderedAccessView>,
}

pub(crate) struct BufferInner {
    /// The buffer itself (`None`: a buffer that is only a uniform buffer, `cb` alone).
    pub raw: Option<ID3D11Buffer>,
    /// The constant buffer where `UNIFORM` usage comes with others (Direct3D 11 takes
    /// constant buffers alone): a copy of `raw` refreshed before each draw that reads it.
    pub cb: Option<ID3D11Buffer>,
    /// A staging buffer (`MAP_READ` / `MAP_WRITE`): mapped by the CPU, a copy source or
    /// destination only.
    pub staging: bool,
    pub size: u64,
    pub usage: t::BufferUsages,
    pub views: Mutex<HashMap<(u64, u64), RangeViews>>,
    pub shared: Arc<Shared>,
}

// SAFETY: Direct3D 11 resources and views are free-threaded.
unsafe impl Send for BufferInner {}
unsafe impl Sync for BufferInner {}

#[derive(Clone)]
pub struct Buffer(pub(crate) Arc<BufferInner>);

pub struct BufferInitDescriptor<'a> {
    pub label: Option<&'a str>,
    pub contents: &'a [u8],
    pub usage: t::BufferUsages,
}

impl Buffer {
    pub fn size(&self) -> u64 {
        self.0.size
    }

    pub fn usage(&self) -> t::BufferUsages {
        self.0.usage
    }

    /// The resource copies and the input assembler use.
    pub(crate) fn resource(&self) -> &ID3D11Buffer {
        self.0.raw.as_ref().or(self.0.cb.as_ref()).expect("a buffer has a resource")
    }

    /// Map the buffer (a staging one) for `mode`, run `f` on the bytes of `range`, unmap.
    fn map<R>(&self, range: Range<u64>, mode: D3D11_MAP, f: impl FnOnce(*mut u8, usize) -> R) -> Result<R> {
        assert!(self.0.staging, "only MAP_READ / MAP_WRITE buffers are mapped");
        let ctx = self.0.shared.ctx.lock();
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: a staging buffer of this device, mapped and unmapped under the context's lock.
        unsafe { ctx.context.Map(self.resource(), 0, mode, 0, Some(&mut m)) }.map_err(|e| hr("mapping a buffer", e))?;
        let len = (range.end - range.start) as usize;
        // SAFETY: the mapping covers the whole buffer.
        let r = f(unsafe { m.pData.cast::<u8>().add(range.start as usize) }, len);
        unsafe { ctx.context.Unmap(self.resource(), 0) };
        Ok(r)
    }

    /// Read the buffer's contents. Waits until the GPU has finished all submitted work, so
    /// the bytes are those the last submission left. Needs `MAP_READ`.
    pub fn map_read<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        assert!(self.0.usage.contains(t::BufferUsages::MAP_READ), "map_read on a buffer without MAP_READ");
        assert!(
            range.start <= range.end && range.end <= self.0.size,
            "map_read range {range:?} outside the buffer ({} bytes)",
            self.0.size
        );
        self.0.shared.wait_idle()?;
        self.map_read_now(range, f)
    }

    /// Read the buffer's contents now: the caller knows the submissions that wrote it are done
    /// (`Device::is_done`). Needs `MAP_READ`.
    pub fn map_read_now<R>(&self, range: Range<u64>, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        assert!(self.0.usage.contains(t::BufferUsages::MAP_READ), "map_read on a buffer without MAP_READ");
        if range.start == range.end {
            return Ok(f(&[]));
        }
        // SAFETY: the mapped bytes, valid until the unmap after `f`.
        self.map(range, D3D11_MAP_READ, |p, len| f(unsafe { std::slice::from_raw_parts(p, len) }))
    }

    /// Write the buffer's contents from the CPU. Waits until the GPU has finished all
    /// submitted work first. Needs `MAP_WRITE`.
    pub fn map_write<R>(&self, range: Range<u64>, f: impl FnOnce(&mut [u8]) -> R) -> Result<R> {
        assert!(self.0.usage.contains(t::BufferUsages::MAP_WRITE), "map_write on a buffer without MAP_WRITE");
        assert!(
            range.start <= range.end && range.end <= self.0.size,
            "map_write range {range:?} outside the buffer ({} bytes)",
            self.0.size
        );
        self.0.shared.wait_idle()?;
        if range.start == range.end {
            return Ok(f(&mut []));
        }
        // SAFETY: the mapped bytes, valid until the unmap after `f`.
        self.map(range, D3D11_MAP_READ_WRITE, |p, len| f(unsafe { std::slice::from_raw_parts_mut(p, len) }))
    }

    /// Not on Direct3D 11: buffers are mapped only for the length of a call. Use `map_read` /
    /// `map_write`.
    ///
    /// # Safety
    /// Never returns.
    pub unsafe fn mapped_ptr(&self) -> std::ptr::NonNull<u8> {
        panic!("mapped_ptr is not available on Direct3D 11; use map_read / map_write")
    }

    /// The `ID3D11Buffer`, for interop.
    pub fn raw(&self) -> &ID3D11Buffer {
        self.resource()
    }

    /// A range of the buffer in the heap ([`BufferView::index`]). Needs `STORAGE`.
    pub fn create_view(&self, desc: &t::BufferViewDescriptor<'_>) -> Result<BufferView> {
        let shared = &self.0.shared;
        assert!(self.0.usage.contains(t::BufferUsages::STORAGE), "create_view on a buffer without STORAGE usage");
        let size = desc.resolve(self.size(), self.usage(), &shared.limits);
        let slot = shared.heap.lock().add_buffer(BufferSlot {
            buffer: self.clone(),
            offset: desc.offset,
            size,
        })?;
        Ok(BufferView(Arc::new(BufferViewInner {
            buffer: self.clone(),
            offset: desc.offset,
            size,
            writable: desc.writable,
            slot,
        })))
    }

    /// The raw (byte address) views of `size` bytes from `offset`.
    pub(crate) fn range_views(&self, offset: u64, size: u64) -> Result<RangeViews> {
        let mut views = self.0.views.lock();
        if let Some(v) = views.get(&(offset, size)) {
            return Ok(v.clone());
        }
        let raw = self.0.raw.as_ref().ok_or_else(|| Error::Unsupported("a shader view of a uniform-only buffer".into()))?;
        let device = &self.0.shared.device;
        let first = (offset / 4) as u32;
        let count = (size / 4) as u32;
        let mut v = RangeViews::default();
        let srv = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D_SRV_DIMENSION_BUFFEREX,
            Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                BufferEx: D3D11_BUFFEREX_SRV {
                    FirstElement: first,
                    NumElements: count,
                    Flags: D3D11_BUFFEREX_SRV_FLAG_RAW.0 as u32,
                },
            },
        };
        // SAFETY: plain creations of views inside the buffer.
        unsafe { device.CreateShaderResourceView(raw, Some(&srv), Some(&mut v.srv)) }.map_err(|e| hr("buffer view", e))?;
        let uav = D3D11_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
            Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D11_BUFFER_UAV {
                    FirstElement: first,
                    NumElements: count,
                    Flags: D3D11_BUFFER_UAV_FLAG_RAW.0 as u32,
                },
            },
        };
        unsafe { device.CreateUnorderedAccessView(raw, Some(&uav), Some(&mut v.uav)) }.map_err(|e| hr("buffer view", e))?;
        views.insert((offset, size), v.clone());
        Ok(v)
    }
}

pub(crate) struct BufferViewInner {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: u64,
    pub writable: bool,
    slot: u32,
}

impl Drop for BufferViewInner {
    fn drop(&mut self) {
        let shared = &self.buffer.0.shared;
        shared.heap.lock().remove_buffer(self.slot, shared.release_after());
    }
}

/// A range of a buffer in the heap.
#[derive(Clone)]
pub struct BufferView(pub(crate) Arc<BufferViewInner>);

impl BufferView {
    pub fn index(&self) -> u32 {
        self.0.slot
    }
    pub fn buffer(&self) -> &Buffer {
        &self.0.buffer
    }
    pub fn offset(&self) -> u64 {
        self.0.offset
    }
    pub fn size(&self) -> u64 {
        self.0.size
    }
    pub fn is_writable(&self) -> bool {
        self.0.writable
    }
}

// ---------------------------------------------------------------------------------------------
// Textures

pub(crate) struct TextureInner {
    pub raw: ID3D11Resource,
    pub size: t::Extent3d,
    pub format: t::TextureFormat,
    pub dimension: t::TextureDimension,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub usage: t::TextureUsages,
    /// Made a cube texture (6 square layers that may be sampled).
    pub cube: bool,
    pub shared: Arc<Shared>,
}

// SAFETY: Direct3D 11 resources are free-threaded.
unsafe impl Send for TextureInner {}
unsafe impl Sync for TextureInner {}

#[derive(Clone)]
pub struct Texture(pub(crate) Arc<TextureInner>);

impl Texture {
    pub fn size(&self) -> t::Extent3d {
        self.0.size
    }
    pub fn width(&self) -> u32 {
        self.0.size.width
    }
    pub fn height(&self) -> u32 {
        self.0.size.height
    }
    pub fn format(&self) -> t::TextureFormat {
        self.0.format
    }
    pub fn dimension(&self) -> t::TextureDimension {
        self.0.dimension
    }
    pub fn mip_level_count(&self) -> u32 {
        self.0.mip_level_count
    }
    pub fn sample_count(&self) -> u32 {
        self.0.sample_count
    }
    pub fn usage(&self) -> t::TextureUsages {
        self.0.usage
    }

    pub(crate) fn layers(&self) -> u32 {
        t::Unwritten::layer_count(self.0.dimension, self.0.size)
    }

    /// The subresource of `mip` in `layer` (and `plane`, for stencil).
    pub(crate) fn subresource(&self, mip: u32, layer: u32) -> u32 {
        mip + layer * self.0.mip_level_count
    }

    /// The `ID3D11Resource`, for interop.
    pub fn raw(&self) -> &ID3D11Resource {
        &self.0.raw
    }

    /// A view. With `TEXTURE_BINDING` usage it gets a heap index ([`TextureView::index`]):
    /// its shader resource view; as a storage texture (`STORAGE_BINDING`, one mip of a 2D view)
    /// too, with its unordered access view.
    pub fn create_view(&self, desc: &t::TextureViewDescriptor<'_>) -> Result<TextureView> {
        let tex = &self.0;
        let layers = self.layers();
        let dimension = desc.dimension.unwrap_or(match (tex.dimension, layers) {
            (t::TextureDimension::D3, _) => t::TextureViewDimension::D3,
            (t::TextureDimension::D1, _) => t::TextureViewDimension::D1,
            (_, 1) => t::TextureViewDimension::D2,
            (_, 6) if tex.cube => t::TextureViewDimension::Cube,
            _ => t::TextureViewDimension::D2Array,
        });
        let mip_count = desc.mip_level_count.unwrap_or(tex.mip_level_count - desc.base_mip_level);
        let layer_count = desc.array_layer_count.unwrap_or(layers - desc.base_array_layer);
        let format = desc.format.unwrap_or(tex.format);
        let view = ViewInner {
            texture: self.clone(),
            format,
            dimension,
            aspect: desc.aspect,
            base_mip_level: desc.base_mip_level,
            mip_level_count: mip_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: layer_count,
            slot: None,
            srv: None,
            rtvs: Mutex::new(HashMap::new()),
            dsv: Mutex::new(HashMap::new()),
            uav: Mutex::new(None),
        };
        let srv = if tex.usage.contains(t::TextureUsages::TEXTURE_BINDING) {
            conv::srv_format(format, desc.aspect).map(|f| view.make_srv(f)).transpose()?
        } else {
            None
        };
        // (a storage texture of the heap: its UAV, bound to the slots of heap storage views)
        let uav = if t::is_heap_storage_view(tex.usage, format, dimension, mip_count) {
            Some(view.uav()?)
        } else {
            None
        };
        let slot = if srv.is_some() || uav.is_some() {
            Some(tex.shared.heap.lock().add_texture(crate::heap::TextureSlot { srv: srv.clone(), uav: uav.clone() })?)
        } else {
            None
        };
        Ok(TextureView(Arc::new(ViewInner {
            texture: view.texture.clone(),
            format: view.format,
            dimension: view.dimension,
            aspect: view.aspect,
            base_mip_level: view.base_mip_level,
            mip_level_count: view.mip_level_count,
            base_array_layer: view.base_array_layer,
            array_layer_count: view.array_layer_count,
            slot,
            srv,
            rtvs: Mutex::new(HashMap::new()),
            dsv: Mutex::new(HashMap::new()),
            uav: Mutex::new(uav),
        })))
    }
}

pub(crate) struct ViewInner {
    pub texture: Texture,
    pub format: t::TextureFormat,
    pub dimension: t::TextureViewDimension,
    pub aspect: t::TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
    slot: Option<u32>,
    pub srv: Option<ID3D11ShaderResourceView>,
    /// Render target views by (mip, layer), made on first use.
    rtvs: Mutex<HashMap<(u32, u32), ID3D11RenderTargetView>>,
    /// Depth-stencil views by read-only flags, made on first use.
    dsv: Mutex<HashMap<u32, ID3D11DepthStencilView>>,
    uav: Mutex<Option<ID3D11UnorderedAccessView>>,
}

// SAFETY: Direct3D 11 views are free-threaded.
unsafe impl Send for ViewInner {}
unsafe impl Sync for ViewInner {}

impl ViewInner {
    fn make_srv(&self, format: DXGI_FORMAT) -> Result<ID3D11ShaderResourceView> {
        use t::TextureViewDimension as V;
        let tex = &self.texture.0;
        let (mips, base_mip) = (self.mip_level_count, self.base_mip_level);
        let (first, layers) = (self.base_array_layer, self.array_layer_count);
        let ms = tex.sample_count > 1;
        let (dim, anon) = match self.dimension {
            V::D1 => (
                D3D_SRV_DIMENSION_TEXTURE1D,
                D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture1D: D3D11_TEX1D_SRV {
                        MostDetailedMip: base_mip,
                        MipLevels: mips,
                    },
                },
            ),
            V::D2 if ms => (
                D3D_SRV_DIMENSION_TEXTURE2DMS,
                D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DMS: D3D11_TEX2DMS_SRV::default(),
                },
            ),
            V::D2 if first == 0 && tex.size.depth_or_array_layers <= 1 => (
                D3D_SRV_DIMENSION_TEXTURE2D,
                D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_SRV {
                        MostDetailedMip: base_mip,
                        MipLevels: mips,
                    },
                },
            ),
            V::D2 | V::D2Array if ms => (
                D3D_SRV_DIMENSION_TEXTURE2DMSARRAY,
                D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DMSArray: D3D11_TEX2DMS_ARRAY_SRV {
                        FirstArraySlice: first,
                        ArraySize: layers,
                    },
                },
            ),
            V::D2 | V::D2Array => (
                D3D_SRV_DIMENSION_TEXTURE2DARRAY,
                D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D11_TEX2D_ARRAY_SRV {
                        MostDetailedMip: base_mip,
                        MipLevels: mips,
                        FirstArraySlice: first,
                        ArraySize: layers,
                    },
                },
            ),
            V::Cube => (
                D3D_SRV_DIMENSION_TEXTURECUBE,
                D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    TextureCube: D3D11_TEXCUBE_SRV {
                        MostDetailedMip: base_mip,
                        MipLevels: mips,
                    },
                },
            ),
            V::CubeArray => (
                D3D_SRV_DIMENSION_TEXTURECUBEARRAY,
                D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    TextureCubeArray: D3D11_TEXCUBE_ARRAY_SRV {
                        MostDetailedMip: base_mip,
                        MipLevels: mips,
                        First2DArrayFace: first,
                        NumCubes: layers / 6,
                    },
                },
            ),
            V::D3 => (
                D3D_SRV_DIMENSION_TEXTURE3D,
                D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture3D: D3D11_TEX3D_SRV {
                        MostDetailedMip: base_mip,
                        MipLevels: mips,
                    },
                },
            ),
        };
        let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: dim,
            Anonymous: anon,
        };
        let mut srv = None;
        // SAFETY: plain creation of a view inside the texture.
        unsafe { tex.shared.device.CreateShaderResourceView(&tex.raw, Some(&desc), Some(&mut srv)) }.map_err(|e| hr("texture view", e))?;
        srv.ok_or_else(|| Error::OutOfMemory("texture view".into()))
    }

    /// The render target view of mip `base_mip_level`, layer (or depth slice) `layer`.
    pub fn rtv(&self, layer: u32) -> Result<ID3D11RenderTargetView> {
        let key = (self.base_mip_level, self.base_array_layer + layer);
        if let Some(v) = self.rtvs.lock().get(&key) {
            return Ok(v.clone());
        }
        let v = rtv_of(&self.texture, self.format, key.0, key.1)?;
        self.rtvs.lock().insert(key, v.clone());
        Ok(v)
    }

    /// The depth-stencil view (`read_only`: `D3D11_DSV_READ_ONLY_*` flags).
    pub fn dsv(&self, read_only: u32) -> Result<ID3D11DepthStencilView> {
        if let Some(v) = self.dsv.lock().get(&read_only) {
            return Ok(v.clone());
        }
        let v = dsv_of(&self.texture, self.format, self.base_mip_level, self.base_array_layer, read_only)?;
        self.dsv.lock().insert(read_only, v.clone());
        Ok(v)
    }

    /// The unordered access view (a storage texture), of mip `base_mip_level`.
    pub fn uav(&self) -> Result<ID3D11UnorderedAccessView> {
        if let Some(v) = self.uav.lock().clone() {
            return Ok(v);
        }
        let tex = &self.texture.0;
        let (mip, first, count) = (self.base_mip_level, self.base_array_layer, self.array_layer_count);
        let (dim, anon) = match tex.dimension {
            t::TextureDimension::D3 => (
                D3D11_UAV_DIMENSION_TEXTURE3D,
                D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture3D: D3D11_TEX3D_UAV {
                        MipSlice: mip,
                        FirstWSlice: 0,
                        WSize: u32::MAX,
                    },
                },
            ),
            t::TextureDimension::D1 => (
                D3D11_UAV_DIMENSION_TEXTURE1D,
                D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture1D: D3D11_TEX1D_UAV { MipSlice: mip },
                },
            ),
            _ if self.dimension == t::TextureViewDimension::D2 => (
                D3D11_UAV_DIMENSION_TEXTURE2D,
                D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_UAV { MipSlice: mip },
                },
            ),
            _ => (
                D3D11_UAV_DIMENSION_TEXTURE2DARRAY,
                D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2DArray: D3D11_TEX2D_ARRAY_UAV {
                        MipSlice: mip,
                        FirstArraySlice: first,
                        ArraySize: count,
                    },
                },
            ),
        };
        let desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
            Format: conv::format(self.format),
            ViewDimension: dim,
            Anonymous: anon,
        };
        let mut v = None;
        // SAFETY: plain creation of a view inside the texture.
        unsafe { tex.shared.device.CreateUnorderedAccessView(&tex.raw, Some(&desc), Some(&mut v)) }.map_err(|e| hr("storage texture view", e))?;
        let v = v.ok_or_else(|| Error::OutOfMemory("storage texture view".into()))?;
        *self.uav.lock() = Some(v.clone());
        Ok(v)
    }
}

/// A render target view of `mip`, layer (or depth slice) `slice` of `tex`, read as `format`.
pub(crate) fn rtv_of(tex: &Texture, format: t::TextureFormat, mip: u32, slice: u32) -> Result<ID3D11RenderTargetView> {
    let t0 = &tex.0;
    let (dim, anon) = match t0.dimension {
        t::TextureDimension::D3 => (
            D3D11_RTV_DIMENSION_TEXTURE3D,
            D3D11_RENDER_TARGET_VIEW_DESC_0 {
                Texture3D: D3D11_TEX3D_RTV {
                    MipSlice: mip,
                    FirstWSlice: slice,
                    WSize: 1,
                },
            },
        ),
        _ if t0.sample_count > 1 => (
            D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY,
            D3D11_RENDER_TARGET_VIEW_DESC_0 {
                Texture2DMSArray: D3D11_TEX2DMS_ARRAY_RTV { FirstArraySlice: slice, ArraySize: 1 },
            },
        ),
        _ => (
            D3D11_RTV_DIMENSION_TEXTURE2DARRAY,
            D3D11_RENDER_TARGET_VIEW_DESC_0 {
                Texture2DArray: D3D11_TEX2D_ARRAY_RTV {
                    MipSlice: mip,
                    FirstArraySlice: slice,
                    ArraySize: 1,
                },
            },
        ),
    };
    let desc = D3D11_RENDER_TARGET_VIEW_DESC {
        Format: conv::format(format),
        ViewDimension: dim,
        Anonymous: anon,
    };
    let mut v = None;
    // SAFETY: plain creation of a view inside the texture.
    unsafe { t0.shared.device.CreateRenderTargetView(&t0.raw, Some(&desc), Some(&mut v)) }.map_err(|e| hr("render target view", e))?;
    v.ok_or_else(|| Error::OutOfMemory("render target view".into()))
}

/// A depth-stencil view of `mip`, layer `slice` of `tex` (`read_only`: `D3D11_DSV_READ_ONLY_*`).
pub(crate) fn dsv_of(tex: &Texture, format: t::TextureFormat, mip: u32, slice: u32, read_only: u32) -> Result<ID3D11DepthStencilView> {
    let t0 = &tex.0;
    let (dim, anon) = if t0.sample_count > 1 {
        (
            D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY,
            D3D11_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DMSArray: D3D11_TEX2DMS_ARRAY_DSV { FirstArraySlice: slice, ArraySize: 1 },
            },
        )
    } else {
        (
            D3D11_DSV_DIMENSION_TEXTURE2DARRAY,
            D3D11_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DArray: D3D11_TEX2D_ARRAY_DSV {
                    MipSlice: mip,
                    FirstArraySlice: slice,
                    ArraySize: 1,
                },
            },
        )
    };
    let desc = D3D11_DEPTH_STENCIL_VIEW_DESC {
        Format: conv::format(format),
        ViewDimension: dim,
        Flags: read_only,
        Anonymous: anon,
    };
    let mut v = None;
    // SAFETY: plain creation of a view inside the texture.
    unsafe { t0.shared.device.CreateDepthStencilView(&t0.raw, Some(&desc), Some(&mut v)) }.map_err(|e| hr("depth-stencil view", e))?;
    v.ok_or_else(|| Error::OutOfMemory("depth-stencil view".into()))
}

impl Drop for ViewInner {
    fn drop(&mut self) {
        if let Some(slot) = self.slot {
            let shared = &self.texture.0.shared;
            shared.heap.lock().remove_texture(slot, shared.release_after());
        }
    }
}

#[derive(Clone)]
pub struct TextureView(pub(crate) Arc<ViewInner>);

impl TextureView {
    pub fn index(&self) -> u32 {
        self.try_index()
            .expect("this texture view is not in the bindless heap (its texture lacks TEXTURE_BINDING usage, or it shows both depth and stencil)")
    }
    pub fn try_index(&self) -> Option<u32> {
        self.0.slot
    }
    pub fn texture(&self) -> &Texture {
        &self.0.texture
    }
    pub fn format(&self) -> t::TextureFormat {
        self.0.format
    }
    pub fn dimension(&self) -> t::TextureViewDimension {
        self.0.dimension
    }
    pub fn mip_levels(&self) -> Range<u32> {
        self.0.base_mip_level..self.0.base_mip_level + self.0.mip_level_count
    }
    pub fn array_layers(&self) -> Range<u32> {
        self.0.base_array_layer..self.0.base_array_layer + self.0.array_layer_count
    }
}

// ---------------------------------------------------------------------------------------------
// Samplers

pub(crate) struct SamplerInner {
    pub raw: ID3D11SamplerState,
    slot: u32,
    compare: bool,
    shared: Arc<Shared>,
}

// SAFETY: Direct3D 11 states are free-threaded.
unsafe impl Send for SamplerInner {}
unsafe impl Sync for SamplerInner {}

impl Drop for SamplerInner {
    fn drop(&mut self) {
        self.shared.heap.lock().remove_sampler(self.slot, self.shared.release_after());
    }
}

#[derive(Clone)]
pub struct Sampler(pub(crate) Arc<SamplerInner>);

impl Sampler {
    pub fn index(&self) -> u32 {
        self.0.slot
    }
    pub fn is_comparison(&self) -> bool {
        self.0.compare
    }
}

// ---------------------------------------------------------------------------------------------
// Bind groups (root bindings)

pub struct BufferBinding<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: Option<u64>,
}

pub enum BindingResource<'a> {
    Buffer(BufferBinding<'a>),
    TextureView(&'a TextureView),
    Sampler(&'a Sampler),
    AccelerationStructure(&'a crate::Tlas),
}

pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

pub struct BindGroupDescriptor<'a> {
    pub label: Option<&'a str>,
    pub entries: &'a [BindGroupEntry<'a>],
}

#[derive(Clone)]
pub(crate) enum Bound {
    Buffer { buffer: Buffer, offset: u64, size: u64 },
    Texture(TextureView),
    Sampler(Sampler),
}

#[derive(Clone)]
pub struct BindGroup(pub(crate) Arc<Vec<(u32, Bound)>>);

impl BindGroup {
    pub(crate) fn get(&self, binding: u32) -> Option<&Bound> {
        self.0.iter().find(|(b, _)| *b == binding).map(|(_, r)| r)
    }
}

// ---------------------------------------------------------------------------------------------
// GPU-driven batches: not on this tier

#[derive(Clone)]
pub struct IndirectDraws(());

impl IndirectDraws {
    pub fn max_draws(&self) -> u32 {
        0
    }
}

// ---------------------------------------------------------------------------------------------
// Creation

fn bind_flags(usage: t::TextureUsages, format: t::TextureFormat) -> u32 {
    let mut f = 0;
    if usage.contains(t::TextureUsages::TEXTURE_BINDING) {
        f |= D3D11_BIND_SHADER_RESOURCE.0;
    }
    if usage.contains(t::TextureUsages::RENDER_ATTACHMENT) {
        f |= if format.has_depth() { D3D11_BIND_DEPTH_STENCIL.0 } else { D3D11_BIND_RENDER_TARGET.0 };
    }
    if usage.contains(t::TextureUsages::STORAGE_BINDING) {
        f |= D3D11_BIND_UNORDERED_ACCESS.0;
    }
    f as u32
}

impl Device {
    pub fn create_buffer(&self, desc: &t::BufferDescriptor<'_>) -> Result<Buffer> {
        let shared = &self.shared;
        let u = desc.usage;
        let size = desc.size.max(4).next_multiple_of(16);
        let staging = u.intersects(t::BufferUsages::MAP_READ | t::BufferUsages::MAP_WRITE);
        let zeros = vec![0u8; size as usize];
        let init = D3D11_SUBRESOURCE_DATA {
            pSysMem: zeros.as_ptr().cast(),
            SysMemPitch: 0,
            SysMemSlicePitch: 0,
        };
        let make = |d: &D3D11_BUFFER_DESC, what: &str| -> Result<ID3D11Buffer> {
            let mut b = None;
            // SAFETY: plain creation, zero-filled (wgpu's guarantee: new buffers read as zeros).
            unsafe { shared.device.CreateBuffer(d, Some(&init), Some(&mut b)) }.map_err(|e| hr(what, e))?;
            b.ok_or_else(|| Error::OutOfMemory(what.into()))
        };
        let (raw, cb) = if staging {
            let d = D3D11_BUFFER_DESC {
                ByteWidth: size as u32,
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: (D3D11_CPU_ACCESS_READ.0 | D3D11_CPU_ACCESS_WRITE.0) as u32,
                MiscFlags: 0,
                StructureByteStride: 0,
            };
            (Some(make(&d, "staging buffer")?), None)
        } else {
            let mut bind = 0;
            let mut misc = 0;
            if u.contains(t::BufferUsages::VERTEX) {
                bind |= D3D11_BIND_VERTEX_BUFFER.0;
            }
            if u.contains(t::BufferUsages::INDEX) {
                bind |= D3D11_BIND_INDEX_BUFFER.0;
            }
            if u.contains(t::BufferUsages::STORAGE) {
                bind |= D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0;
                misc |= D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS.0;
            }
            if u.contains(t::BufferUsages::INDIRECT) {
                misc |= D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS.0;
                // (indirect arguments are written by shaders through a raw view)
                bind |= D3D11_BIND_UNORDERED_ACCESS.0;
                misc |= D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS.0;
            }
            let raw = if bind != 0 || misc != 0 || !u.contains(t::BufferUsages::UNIFORM) {
                let d = D3D11_BUFFER_DESC {
                    ByteWidth: size as u32,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: bind as u32,
                    CPUAccessFlags: 0,
                    MiscFlags: misc as u32,
                    StructureByteStride: 0,
                };
                Some(make(&d, "buffer")?)
            } else {
                None
            };
            let cb = if u.contains(t::BufferUsages::UNIFORM) {
                let d = D3D11_BUFFER_DESC {
                    ByteWidth: size as u32,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                    CPUAccessFlags: 0,
                    MiscFlags: 0,
                    StructureByteStride: 0,
                };
                Some(make(&d, "uniform buffer")?)
            } else {
                None
            };
            (raw, cb)
        };
        Ok(Buffer(Arc::new(BufferInner {
            raw,
            cb,
            staging,
            size: desc.size,
            usage: desc.usage,
            views: Mutex::new(HashMap::new()),
            shared: shared.clone(),
        })))
    }

    pub fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Result<Buffer> {
        let b = self.create_buffer(&t::BufferDescriptor {
            label: desc.label,
            size: desc.contents.len() as u64,
            usage: desc.usage | t::BufferUsages::COPY_DST,
        })?;
        Queue { shared: self.shared.clone() }.write_buffer(&b, 0, desc.contents)?;
        Ok(b)
    }

    pub fn create_texture(&self, desc: &t::TextureDescriptor<'_>) -> Result<Texture> {
        let shared = &self.shared;
        let layers = t::Unwritten::layer_count(desc.dimension, desc.size);
        let cube = layers == 6 && desc.size.width == desc.size.height && desc.usage.contains(t::TextureUsages::TEXTURE_BINDING) && desc.sample_count == 1;
        let format = conv::resource_format(desc.format, desc.usage, !desc.view_formats.is_empty());
        let bind = bind_flags(desc.usage, desc.format);
        let mut misc = if cube { D3D11_RESOURCE_MISC_TEXTURECUBE.0 as u32 } else { 0 };
        // (render targets that may be sampled: their mips can be made by the GPU)
        if desc.mip_level_count > 1 && bind & D3D11_BIND_RENDER_TARGET.0 as u32 != 0 && bind & D3D11_BIND_SHADER_RESOURCE.0 as u32 != 0 {
            misc |= D3D11_RESOURCE_MISC_GENERATE_MIPS.0 as u32;
        }
        let device = &shared.device;
        let raw: ID3D11Resource = match desc.dimension {
            t::TextureDimension::D1 => {
                let d = D3D11_TEXTURE1D_DESC {
                    Width: desc.size.width,
                    MipLevels: desc.mip_level_count,
                    ArraySize: layers,
                    Format: format,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: bind,
                    CPUAccessFlags: 0,
                    MiscFlags: misc,
                };
                let mut tex = None;
                // SAFETY: plain creation.
                unsafe { device.CreateTexture1D(&d, None, Some(&mut tex)) }.map_err(|e| hr("texture", e))?;
                windows::core::Interface::cast(&tex.ok_or_else(|| Error::OutOfMemory("texture".into()))?).map_err(|e| hr("texture", e))?
            }
            t::TextureDimension::D2 => {
                let d = D3D11_TEXTURE2D_DESC {
                    Width: desc.size.width,
                    Height: desc.size.height,
                    MipLevels: desc.mip_level_count,
                    ArraySize: layers,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: desc.sample_count, Quality: 0 },
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: bind,
                    CPUAccessFlags: 0,
                    MiscFlags: misc,
                };
                let mut tex = None;
                // SAFETY: plain creation.
                unsafe { device.CreateTexture2D(&d, None, Some(&mut tex)) }.map_err(|e| hr("texture", e))?;
                windows::core::Interface::cast(&tex.ok_or_else(|| Error::OutOfMemory("texture".into()))?).map_err(|e| hr("texture", e))?
            }
            t::TextureDimension::D3 => {
                let d = D3D11_TEXTURE3D_DESC {
                    Width: desc.size.width,
                    Height: desc.size.height,
                    Depth: desc.size.depth_or_array_layers,
                    MipLevels: desc.mip_level_count,
                    Format: format,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: bind,
                    CPUAccessFlags: 0,
                    MiscFlags: misc,
                };
                let mut tex = None;
                // SAFETY: plain creation.
                unsafe { device.CreateTexture3D(&d, None, Some(&mut tex)) }.map_err(|e| hr("texture", e))?;
                windows::core::Interface::cast(&tex.ok_or_else(|| Error::OutOfMemory("texture".into()))?).map_err(|e| hr("texture", e))?
            }
        };
        let tex = Texture(Arc::new(TextureInner {
            raw,
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            cube,
            shared: shared.clone(),
        }));
        // (wgpu's guarantee: a new texture reads as zeros)
        let mut ctx = shared.ctx.lock();
        crate::command::zero_texture(&mut ctx, &tex, 0..desc.mip_level_count, 0..layers);
        Ok(tex)
    }

    /// A texture made outside mX (an XR swapchain image), described by `desc`.
    ///
    /// # Safety
    /// `raw` is a texture of this device made as `desc` says.
    pub unsafe fn texture_from_raw(&self, raw: ID3D11Resource, desc: &t::TextureDescriptor<'_>) -> Texture {
        let layers = t::Unwritten::layer_count(desc.dimension, desc.size);
        Texture(Arc::new(TextureInner {
            raw,
            size: desc.size,
            format: desc.format,
            dimension: desc.dimension,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            usage: desc.usage,
            cube: layers == 6 && desc.size.width == desc.size.height && desc.usage.contains(t::TextureUsages::TEXTURE_BINDING),
            shared: self.shared.clone(),
        }))
    }

    pub fn create_sampler(&self, desc: &t::SamplerDescriptor<'_>) -> Result<Sampler> {
        let shared = &self.shared;
        let d = D3D11_SAMPLER_DESC {
            Filter: conv::filter(desc),
            AddressU: conv::address(desc.address_mode_u),
            AddressV: conv::address(desc.address_mode_v),
            AddressW: conv::address(desc.address_mode_w),
            MipLODBias: 0.0,
            MaxAnisotropy: desc.anisotropy_clamp.clamp(1, 16) as u32,
            ComparisonFunc: desc.compare.map_or(D3D11_COMPARISON_NEVER, conv::compare),
            BorderColor: conv::border(desc.border_color),
            MinLOD: desc.lod_min_clamp,
            MaxLOD: desc.lod_max_clamp,
        };
        let mut s = None;
        // SAFETY: plain creation.
        unsafe { shared.device.CreateSamplerState(&d, Some(&mut s)) }.map_err(|e| hr("sampler", e))?;
        let raw = s.ok_or_else(|| Error::OutOfMemory("sampler".into()))?;
        let slot = shared.heap.lock().add_sampler(raw.clone())?;
        Ok(Sampler(Arc::new(SamplerInner {
            raw,
            slot,
            compare: desc.compare.is_some(),
            shared: shared.clone(),
        })))
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup {
        let entries = desc
            .entries
            .iter()
            .map(|e| {
                let r = match &e.resource {
                    BindingResource::Buffer(b) => Bound::Buffer {
                        buffer: b.buffer.clone(),
                        offset: b.offset,
                        size: b.size.unwrap_or(b.buffer.size() - b.offset),
                    },
                    BindingResource::TextureView(v) => Bound::Texture((*v).clone()),
                    BindingResource::Sampler(s) => Bound::Sampler((*s).clone()),
                    BindingResource::AccelerationStructure(_) => panic!("acceleration structures need a device with bindless (Metal 3, Vulkan, Direct3D 12)"),
                };
                (e.binding, r)
            })
            .collect();
        BindGroup(Arc::new(entries))
    }

    /// Not on this tier (see `Adapter::supports_indirect_count`).
    pub fn create_indirect_draws(&self, _desc: &t::IndirectDrawsDescriptor<'_>) -> Result<IndirectDraws> {
        Err(Error::Unsupported("GPU-driven draw batches".into()))
    }
}

// ---------------------------------------------------------------------------------------------
// Queue writes: run at once on the immediate context, so after every submission before and
// before every submission after, as wgpu promises.

/// Write `data` at `offset` of `buffer` (both of its resources where it has two).
pub(crate) fn write_buffer(ctx: &Ctx, buffer: &Buffer, offset: u64, data: &[u8]) {
    let b = &buffer.0;
    if b.staging {
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: a staging buffer, mapped and written inside its size.
        unsafe {
            if ctx.context.Map(buffer.resource(), 0, D3D11_MAP_WRITE, 0, Some(&mut m)).is_ok() {
                std::ptr::copy_nonoverlapping(data.as_ptr(), m.pData.cast::<u8>().add(offset as usize), data.len());
                ctx.context.Unmap(buffer.resource(), 0);
            }
        }
        return;
    }
    let bx = D3D11_BOX {
        left: offset as u32,
        top: 0,
        front: 0,
        right: (offset + data.len() as u64) as u32,
        bottom: 1,
        back: 1,
    };
    for r in [b.raw.as_ref(), b.cb.as_ref()].into_iter().flatten() {
        // SAFETY: a range inside the buffer; constant buffers take partial updates through
        // `UpdateSubresource1` (Direct3D 11.1).
        unsafe { ctx.context.UpdateSubresource1(r, 0, Some(&bx), data.as_ptr().cast(), 0, 0, 0) };
    }
}

impl Queue {
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        assert!(offset + data.len() as u64 <= buffer.size(), "write_buffer past the end of the buffer");
        let ctx = self.shared.ctx.lock();
        write_buffer(&ctx, buffer, offset, data);
        Ok(())
    }

    pub fn write_texture(&self, dst: t::TexelCopyTextureInfo<'_, Texture>, data: &[u8], layout: t::TexelCopyBufferLayout, size: t::Extent3d) -> Result<()> {
        let ctx = self.shared.ctx.lock();
        crate::command::upload(&ctx, dst.texture, dst.mip_level, dst.origin, size, layout, &data[layout.offset as usize..]);
        Ok(())
    }
}

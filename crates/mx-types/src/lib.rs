//! Plain data shared by every mX backend.
//!
//! Names and shapes follow wgpu wherever the concept is the same, so code moving from wgpu to
//! mX changes `wgpu::` to `mx::` and little else. Descriptors that refer to GPU objects are
//! generic over the object type, so each backend fills them with its own objects.

#![forbid(unsafe_code)]

use bitflags::bitflags;

mod format;
pub use format::*;

/// The graphics API under a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    Metal,
    Vulkan,
    Dx12,
    /// OpenGL or OpenGL ES (the bound tier: no bindless on the GPU).
    Gl,
    /// Direct3D 11 (the bound tier).
    Dx11,
    /// No GPU: a device that takes every call and draws nothing (dedicated servers, tests).
    Noop,
}

bitflags! {
    /// A set of backends, to choose which an `Instance` opens.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Backends: u32 {
        const METAL = 1 << 0;
        const VULKAN = 1 << 1;
        const DX12 = 1 << 2;
        const GL = 1 << 3;
        /// The no-op device: never opened unless asked for by name (not in `ALL`).
        const NOOP = 1 << 4;
        /// Direct3D 11 (the bound tier).
        const DX11 = 1 << 5;
        /// The bindless backends.
        const PRIMARY = Self::METAL.bits() | Self::VULKAN.bits() | Self::DX12.bits();
        const ALL = Self::PRIMARY.bits() | Self::GL.bits() | Self::DX11.bits();
    }
}

impl Backends {
    /// The set `MX_BACKENDS` names (`metal`, `vulkan`, `dx12`, `gl`, `noop`, comma-separated), if set.
    pub fn from_env() -> Option<Backends> {
        let v = std::env::var("MX_BACKENDS").ok()?;
        Some(v.split(',').fold(Backends::empty(), |b, name| {
            b | match name.trim().to_ascii_lowercase().as_str() {
                "metal" | "mtl" => Backends::METAL,
                "vulkan" | "vk" => Backends::VULKAN,
                "dx11" | "d3d11" => Backends::DX11,
                "dx12" | "d3d12" => Backends::DX12,
                "gl" | "gles" | "opengl" => Backends::GL,
                "noop" | "none" => Backends::NOOP,
                _ => Backends::empty(),
            }
        }))
    }
}

impl From<Backend> for Backends {
    fn from(b: Backend) -> Backends {
        match b {
            Backend::Metal => Backends::METAL,
            Backend::Vulkan => Backends::VULKAN,
            Backend::Dx12 => Backends::DX12,
            Backend::Gl => Backends::GL,
            Backend::Noop => Backends::NOOP,
            Backend::Dx11 => Backends::DX11,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceType {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: Backend,
    pub device_type: DeviceType,
    /// The vendor's PCI id (0 where the platform does not report one).
    pub vendor: u32,
    pub device: u32,
    pub driver: String,
}

/// What the system says about a GPU's memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdapterMemory {
    /// The GPU's own memory (on unified-memory GPUs, what the system lets it use).
    pub dedicated_bytes: u64,
    /// What the driver lets the program allocate all together, where it says.
    pub budget_bytes: Option<u64>,
}

/// What a device can hold. The bindless capacities are fixed when the device is opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,
    /// Texture views that can be in the heap at once.
    pub max_bindless_textures: u32,
    /// Samplers that can be in the heap at once.
    pub max_bindless_samplers: u32,
    /// Buffer views that can be in the heap at once.
    pub max_bindless_buffers: u32,
    /// What a buffer view's offset is a multiple of.
    pub buffer_view_offset_alignment: u32,
    /// Bytes of `var<immediate>` data per draw.
    pub max_immediate_size: u32,
    pub max_vertex_buffers: u32,
    pub max_root_buffers: u32,
    pub max_color_attachments: u32,
    /// The largest buffer.
    pub max_buffer_size: u64,
    /// The largest range of a storage buffer one binding (or buffer view) reaches.
    pub max_storage_buffer_binding_size: u64,
    /// What root uniform and storage buffer binding offsets are multiples of.
    pub min_uniform_buffer_offset_alignment: u32,
    pub min_storage_buffer_offset_alignment: u32,
}

impl Limits {
    /// These limits, none above `cap`'s (what a program asks a device to keep to: a weaker
    /// chip's, to test on). Alignments are kept at the larger of the two.
    pub fn capped(self, cap: Option<&Limits>) -> Limits {
        let Some(c) = cap else { return self };
        Limits {
            max_texture_dimension_2d: self.max_texture_dimension_2d.min(c.max_texture_dimension_2d),
            max_texture_dimension_3d: self.max_texture_dimension_3d.min(c.max_texture_dimension_3d),
            max_texture_array_layers: self.max_texture_array_layers.min(c.max_texture_array_layers),
            max_bindless_textures: self.max_bindless_textures.min(c.max_bindless_textures),
            max_bindless_samplers: self.max_bindless_samplers.min(c.max_bindless_samplers),
            max_bindless_buffers: self.max_bindless_buffers.min(c.max_bindless_buffers),
            buffer_view_offset_alignment: self.buffer_view_offset_alignment.max(c.buffer_view_offset_alignment),
            max_immediate_size: self.max_immediate_size.min(c.max_immediate_size),
            max_vertex_buffers: self.max_vertex_buffers.min(c.max_vertex_buffers),
            max_root_buffers: self.max_root_buffers.min(c.max_root_buffers),
            max_color_attachments: self.max_color_attachments.min(c.max_color_attachments),
            max_buffer_size: self.max_buffer_size.min(c.max_buffer_size),
            max_storage_buffer_binding_size: self.max_storage_buffer_binding_size.min(c.max_storage_buffer_binding_size),
            min_uniform_buffer_offset_alignment: self.min_uniform_buffer_offset_alignment.max(c.min_uniform_buffer_offset_alignment),
            min_storage_buffer_offset_alignment: self.min_storage_buffer_offset_alignment.max(c.min_storage_buffer_offset_alignment),
        }
    }

    /// An older chip's (WebGPU's downlevel limits, as wgpu names them): 2048 texels a side,
    /// 256 array layers, 256 MB buffers.
    pub fn downlevel() -> Limits {
        Limits {
            max_texture_dimension_2d: 2048,
            max_texture_dimension_3d: 256,
            max_texture_array_layers: 256,
            max_buffer_size: 1 << 28,
            ..Limits::default()
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_texture_dimension_2d: 16384,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 2048,
            max_bindless_textures: 1 << 16,
            max_bindless_samplers: 2048,
            max_bindless_buffers: 1 << 15,
            buffer_view_offset_alignment: 256,
            max_immediate_size: 128,
            max_vertex_buffers: 8,
            max_root_buffers: 13,
            max_color_attachments: 8,
            max_buffer_size: 1 << 28,
            max_storage_buffer_binding_size: 1 << 27,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Buffers

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct BufferUsages: u32 {
        const MAP_READ = 1 << 0;
        const MAP_WRITE = 1 << 1;
        const COPY_SRC = 1 << 2;
        const COPY_DST = 1 << 3;
        const INDEX = 1 << 4;
        const VERTEX = 1 << 5;
        const UNIFORM = 1 << 6;
        const STORAGE = 1 << 7;
        const INDIRECT = 1 << 8;
        /// Destination of `CommandEncoder::resolve_query_set`.
        const QUERY_RESOLVE = 1 << 9;
        /// Vertex or index data of a bottom-level acceleration structure build.
        const BLAS_INPUT = 1 << 10;
    }
}

/// A range of a buffer in the bindless heap, read in WGSL through `mx_buffers[i]`,
/// `mx_rw_buffers[i]` or a typed view the shader declares (see `mx_shader`). The buffer
/// needs `STORAGE` usage.
#[derive(Clone, Debug, Default)]
pub struct BufferViewDescriptor<'a> {
    pub label: Option<&'a str>,
    /// A multiple of `Limits::buffer_view_offset_alignment`.
    pub offset: u64,
    /// Bytes from `offset`, a multiple of 4; `None`: to the end of the buffer.
    pub size: Option<u64>,
    /// Shaders may write the view (`mx_rw_buffers`, `var<storage, read_write>` views). Writing
    /// a view made without it is undefined, and on Direct3D 12 the write goes nowhere.
    pub writable: bool,
}

impl BufferViewDescriptor<'_> {
    /// Check the view against its buffer and return its size in bytes. Panics on misuse, as
    /// wgpu's validation would report it.
    pub fn resolve(&self, buffer_size: u64, usage: BufferUsages, limits: &Limits) -> u64 {
        assert!(usage.contains(BufferUsages::STORAGE), "a buffer view needs a buffer with STORAGE usage");
        let align = limits.buffer_view_offset_alignment.max(4) as u64;
        assert!(self.offset.is_multiple_of(align), "buffer view offset {} is not a multiple of {align}", self.offset);
        assert!(self.offset <= buffer_size, "buffer view offset {} past the end of the buffer ({buffer_size} bytes)", self.offset);
        let size = self.size.unwrap_or(buffer_size - self.offset);
        assert!(
            self.offset + size <= buffer_size,
            "buffer view {}..{} outside the buffer ({buffer_size} bytes)",
            self.offset,
            self.offset + size
        );
        assert!(size.is_multiple_of(4) && size > 0, "buffer view size {size} is not a positive multiple of 4");
        size
    }
}

#[derive(Clone, Debug)]
pub struct BufferDescriptor<'a> {
    pub label: Option<&'a str>,
    pub size: u64,
    pub usage: BufferUsages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

impl IndexFormat {
    pub const fn size(self) -> u64 {
        match self {
            IndexFormat::Uint16 => 2,
            IndexFormat::Uint32 => 4,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Textures

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct TextureUsages: u32 {
        const COPY_SRC = 1 << 0;
        const COPY_DST = 1 << 1;
        /// Sampled or loaded in shaders: views of it get a bindless index.
        const TEXTURE_BINDING = 1 << 2;
        const STORAGE_BINDING = 1 << 3;
        const RENDER_ATTACHMENT = 1 << 4;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TextureAspect {
    #[default]
    All,
    StencilOnly,
    DepthOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

impl Extent3d {
    /// The size of mip level `level`, never smaller than one texel.
    pub fn mip_level_size(&self, level: u32, dimension: TextureDimension) -> Extent3d {
        Extent3d {
            width: (self.width >> level).max(1),
            height: (self.height >> level).max(1),
            depth_or_array_layers: match dimension {
                TextureDimension::D3 => (self.depth_or_array_layers >> level).max(1),
                _ => self.depth_or_array_layers,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Origin3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Origin3d {
    pub const ZERO: Origin3d = Origin3d { x: 0, y: 0, z: 0 };
}

#[derive(Clone, Debug)]
pub struct TextureDescriptor<'a> {
    pub label: Option<&'a str>,
    pub size: Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsages,
    /// Other formats views of it may have (sRGB / linear pairs).
    pub view_formats: &'a [TextureFormat],
}

#[derive(Clone, Debug, Default)]
pub struct TextureViewDescriptor<'a> {
    pub label: Option<&'a str>,
    pub format: Option<TextureFormat>,
    pub dimension: Option<TextureViewDimension>,
    pub aspect: TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
}

/// How texel rows sit in a buffer for buffer ↔ texture copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TexelCopyBufferLayout {
    pub offset: u64,
    pub bytes_per_row: Option<u32>,
    pub rows_per_image: Option<u32>,
}

/// Mips and layers of a texture (`CommandEncoder::clear_texture`); a `None` count runs to
/// the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ImageSubresourceRange {
    pub aspect: TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
}

impl ImageSubresourceRange {
    /// The mips and layers this range takes of a texture with `mips` levels and `layers`
    /// layers (1 for a 3D texture: its slices belong to each mip).
    pub fn resolve(&self, mips: u32, layers: u32) -> (std::ops::Range<u32>, std::ops::Range<u32>) {
        let m = self.base_mip_level..self.mip_level_count.map_or(mips, |n| self.base_mip_level + n).min(mips);
        let l = self.base_array_layer..self.array_layer_count.map_or(layers, |n| self.base_array_layer + n).min(layers);
        (m, l)
    }
}

/// The subresources (mip × layer) of a new texture nothing has written yet: wgpu's
/// guarantee is that a texture reads as zeros until written, so mX zeroes what is left of
/// it before the first submission after its creation. A queue write that fills a whole
/// subresource takes it off (its zeroing would be overwritten at once).
#[derive(Clone, Debug)]
pub struct Unwritten {
    layers: u32,
    left: Vec<bool>,
}

impl Unwritten {
    /// Every subresource of a texture made from `desc`.
    pub fn new(desc: &TextureDescriptor<'_>) -> Unwritten {
        let layers = Self::layer_count(desc.dimension, desc.size);
        Unwritten {
            layers,
            left: vec![true; (desc.mip_level_count * layers) as usize],
        }
    }

    /// Layers in the sense of subresources: a 3D texture's slices are part of each mip.
    pub fn layer_count(dimension: TextureDimension, size: Extent3d) -> u32 {
        if dimension == TextureDimension::D3 { 1 } else { size.depth_or_array_layers.max(1) }
    }

    /// A write of `size` texels at `origin` of mip `level` (of a texture of `full` size,
    /// `dimension`, `format`): the subresources it covers whole are written.
    pub fn written(&mut self, format: TextureFormat, dimension: TextureDimension, full: Extent3d, level: u32, origin: Origin3d, size: Extent3d) {
        let mip = full.mip_level_size(level, dimension);
        let (bw, bh) = format.block_dimensions();
        // (a compressed mip smaller than a block is covered by one whole block)
        let round = |v: u32, b: u32| v.div_ceil(b) * b;
        let covers_plane = origin.x == 0 && origin.y == 0 && size.width >= round(mip.width, bw) && size.height >= round(mip.height, bh);
        if !covers_plane {
            return;
        }
        if dimension == TextureDimension::D3 {
            if origin.z == 0 && size.depth_or_array_layers >= mip.depth_or_array_layers {
                self.take(level, 0..1);
            }
        } else {
            self.take(level, origin.z..origin.z + size.depth_or_array_layers);
        }
    }

    /// Mark `layers` of mip `level` as written.
    pub fn take(&mut self, level: u32, layers: std::ops::Range<u32>) {
        for l in layers.start..layers.end.min(self.layers) {
            if let Some(b) = self.left.get_mut((level * self.layers + l) as usize) {
                *b = false;
            }
        }
    }

    /// What is still to be zeroed: per mip, runs of consecutive layers.
    pub fn runs(&self) -> Vec<(u32, std::ops::Range<u32>)> {
        let mut out = Vec::new();
        for (mip, row) in self.left.chunks(self.layers as usize).enumerate() {
            let mut l = 0;
            while l < self.layers {
                if row[l as usize] {
                    let start = l;
                    while l < self.layers && row[l as usize] {
                        l += 1;
                    }
                    out.push((mip as u32, start..l));
                } else {
                    l += 1;
                }
            }
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        !self.left.iter().any(|b| *b)
    }
}

#[cfg(test)]
mod unwritten_tests {
    use super::*;

    fn desc(size: Extent3d, mips: u32, dimension: TextureDimension, format: TextureFormat) -> TextureDescriptor<'static> {
        TextureDescriptor {
            label: None,
            size,
            mip_level_count: mips,
            sample_count: 1,
            dimension,
            format,
            usage: TextureUsages::COPY_DST,
            view_formats: &[],
        }
    }

    #[test]
    fn whole_writes_take_subresources_off() {
        let size = Extent3d {
            width: 16,
            height: 8,
            depth_or_array_layers: 3,
        };
        let d = desc(size, 2, TextureDimension::D2, TextureFormat::Rgba8Unorm);
        let mut u = Unwritten::new(&d);
        assert_eq!(u.runs(), vec![(0, 0..3), (1, 0..3)]);
        // a part of mip 0, layer 0: still to be zeroed
        u.written(
            d.format,
            d.dimension,
            size,
            0,
            Origin3d::ZERO,
            Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        assert_eq!(u.runs(), vec![(0, 0..3), (1, 0..3)]);
        // layer 1 of mip 0 whole, then all of mip 1
        u.written(
            d.format,
            d.dimension,
            size,
            0,
            Origin3d { x: 0, y: 0, z: 1 },
            Extent3d {
                width: 16,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        u.written(
            d.format,
            d.dimension,
            size,
            1,
            Origin3d::ZERO,
            Extent3d {
                width: 8,
                height: 4,
                depth_or_array_layers: 3,
            },
        );
        assert_eq!(u.runs(), vec![(0, 0..1), (0, 2..3)]);
        assert!(!u.is_empty());
    }

    #[test]
    fn compressed_and_3d() {
        let size = Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        };
        let d = desc(size, 4, TextureDimension::D2, TextureFormat::Bc1RgbaUnorm);
        let mut u = Unwritten::new(&d);
        // mips 2 (2×2) and 3 (1×1) take a whole 4×4 block
        for level in 0..4 {
            let m = size.mip_level_size(level, d.dimension);
            u.written(
                d.format,
                d.dimension,
                size,
                level,
                Origin3d::ZERO,
                Extent3d {
                    width: m.width.div_ceil(4) * 4,
                    height: m.height.div_ceil(4) * 4,
                    depth_or_array_layers: 1,
                },
            );
        }
        assert!(u.is_empty());
        let size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 4,
        };
        let d = desc(size, 1, TextureDimension::D3, TextureFormat::R8Unorm);
        let mut u = Unwritten::new(&d);
        u.written(
            d.format,
            d.dimension,
            size,
            0,
            Origin3d::ZERO,
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 2,
            },
        );
        assert_eq!(u.runs(), vec![(0, 0..1)]);
        u.written(
            d.format,
            d.dimension,
            size,
            0,
            Origin3d::ZERO,
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 4,
            },
        );
        assert!(u.is_empty());
    }
}

impl TexelCopyBufferLayout {
    /// Copies between a buffer and a texture in a command encoder take rows that start
    /// [`COPY_BYTES_PER_ROW_ALIGNMENT`] bytes apart, as in WebGPU (Direct3D 12 cannot copy
    /// others). Queue writes have no such rule. Panics on another pitch, on every backend.
    pub fn assert_encoder_pitch(&self) {
        if let Some(b) = self.bytes_per_row {
            assert!(
                b.is_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT),
                "texture copies need bytes_per_row to be a multiple of {COPY_BYTES_PER_ROW_ALIGNMENT}, not {b}"
            );
        }
    }
}

/// Row pitch of copies between buffers and textures. Metal does not need it, D3D12 does;
/// it is kept on every backend so code that works on one works on all.
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
pub const COPY_BUFFER_ALIGNMENT: u64 = 4;

// ---------------------------------------------------------------------------------------------
// Samplers

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AddressMode {
    #[default]
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToBorder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FilterMode {
    #[default]
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MipmapFilterMode {
    #[default]
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SamplerBorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplerDescriptor<'a> {
    pub label: Option<&'a str>,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: MipmapFilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    /// A comparison sampler: reached through `mx_samplers_cmp`.
    pub compare: Option<CompareFunction>,
    pub anisotropy_clamp: u16,
    pub border_color: Option<SamplerBorderColor>,
}

impl Default for SamplerDescriptor<'_> {
    fn default() -> Self {
        SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pipeline state

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexBufferLayout<'a> {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: &'a [VertexAttribute],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    #[default]
    TriangleList,
    TriangleStrip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FrontFace {
    #[default]
    Ccw,
    Cw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    Front,
    Back,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: FrontFace,
    pub cull_mode: Option<Face>,
    /// Depth clipping off: fragments beyond the far plane are clamped, not dropped.
    pub unclipped_depth: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    pub compare: CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

impl StencilFaceState {
    pub const IGNORE: StencilFaceState = StencilFaceState {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self::IGNORE
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct StencilState {
    pub front: StencilFaceState,
    pub back: StencilFaceState,
    pub read_mask: u32,
    pub write_mask: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DepthBiasState {
    pub constant: i32,
    pub slope_scale: f32,
    pub clamp: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
    pub stencil: StencilState,
    pub bias: DepthBiasState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl BlendComponent {
    pub const REPLACE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };
    pub const OVER: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub const REPLACE: Self = Self {
        color: BlendComponent::REPLACE,
        alpha: BlendComponent::REPLACE,
    };
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::OVER,
    };
    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        color: BlendComponent::OVER,
        alpha: BlendComponent::OVER,
    };
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ColorWrites: u32 {
        const RED = 1 << 0;
        const GREEN = 1 << 1;
        const BLUE = 1 << 2;
        const ALPHA = 1 << 3;
        const COLOR = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits();
        const ALL = Self::COLOR.bits() | Self::ALPHA.bits();
    }
}

impl Default for ColorWrites {
    fn default() -> Self {
        ColorWrites::ALL
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: Option<BlendState>,
    pub write_mask: ColorWrites,
}

impl From<TextureFormat> for ColorTargetState {
    fn from(format: TextureFormat) -> Self {
        ColorTargetState {
            format,
            blend: None,
            write_mask: ColorWrites::ALL,
        }
    }
}

/// The values of a shader's `override` declarations, by name.
pub type PipelineConstants<'a> = &'a [(&'a str, f64)];

#[derive(Clone, Debug)]
pub struct VertexState<'a, M> {
    pub module: &'a M,
    pub entry_point: Option<&'a str>,
    pub constants: PipelineConstants<'a>,
    pub buffers: &'a [VertexBufferLayout<'a>],
}

#[derive(Clone, Debug)]
pub struct FragmentState<'a, M> {
    pub module: &'a M,
    pub entry_point: Option<&'a str>,
    pub constants: PipelineConstants<'a>,
    pub targets: &'a [Option<ColorTargetState>],
}

/// A render pipeline. There is no layout: every pipeline sees the bindless heap (group 0),
/// the root bindings its shaders declare (group 1) and the immediates.
#[derive(Clone, Debug)]
pub struct RenderPipelineDescriptor<'a, M> {
    pub label: Option<&'a str>,
    pub vertex: VertexState<'a, M>,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub fragment: Option<FragmentState<'a, M>>,
}

#[derive(Clone, Debug)]
pub struct ComputePipelineDescriptor<'a, M> {
    pub label: Option<&'a str>,
    pub module: &'a M,
    pub entry_point: Option<&'a str>,
    pub constants: PipelineConstants<'a>,
}

// ---------------------------------------------------------------------------------------------
// Passes

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub const TRANSPARENT: Color = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
    pub const BLACK: Color = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const WHITE: Color = Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<V> {
    Clear(V),
    Load,
    /// The previous contents are not needed (tile memory is not filled from memory).
    DontCare,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreOp {
    Store,
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Operations<V> {
    pub load: LoadOp<V>,
    pub store: StoreOp,
}

#[derive(Clone, Debug)]
pub struct RenderPassColorAttachment<'a, T> {
    pub view: &'a T,
    pub resolve_target: Option<&'a T>,
    pub ops: Operations<Color>,
}

#[derive(Clone, Debug)]
pub struct RenderPassDepthStencilAttachment<'a, T> {
    pub view: &'a T,
    pub depth_ops: Option<Operations<f32>>,
    pub stencil_ops: Option<Operations<u32>>,
}

#[derive(Clone, Debug)]
pub struct RenderPassDescriptor<'a, T, Q> {
    pub label: Option<&'a str>,
    pub color_attachments: &'a [Option<RenderPassColorAttachment<'a, T>>],
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'a, T>>,
    pub timestamp_writes: Option<PassTimestampWrites<'a, Q>>,
    pub occlusion_query_set: Option<&'a Q>,
}

impl<T, Q> Default for RenderPassDescriptor<'_, T, Q> {
    fn default() -> Self {
        RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TexelCopyTextureInfo<'a, T> {
    pub texture: &'a T,
    pub mip_level: u32,
    pub origin: Origin3d,
    pub aspect: TextureAspect,
}

#[derive(Clone, Copy, Debug)]
pub struct TexelCopyBufferInfo<'a, B> {
    pub buffer: &'a B,
    pub layout: TexelCopyBufferLayout,
}

#[derive(Clone, Debug)]
pub struct ComputePassDescriptor<'a, Q> {
    pub label: Option<&'a str>,
    pub timestamp_writes: Option<PassTimestampWrites<'a, Q>>,
}

impl<Q> Default for ComputePassDescriptor<'_, Q> {
    fn default() -> Self {
        ComputePassDescriptor { label: None, timestamp_writes: None }
    }
}

// ---------------------------------------------------------------------------------------------
// Indirect arguments (the layouts the GPU reads; the same on every API)

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// A batch of draws whose arguments and count the GPU writes (GPU-driven rendering): made once,
/// filled by `CommandEncoder::encode_indirect_draws` outside passes, drawn by
/// `RenderPass::execute_indirect_draws`. Metal encodes it into an indirect command buffer,
/// Vulkan draws it with `vkCmdDraw*IndirectCount`, Direct3D 12 with `ExecuteIndirect`.
#[derive(Clone, Debug, Default)]
pub struct IndirectDrawsDescriptor<'a> {
    pub label: Option<&'a str>,
    /// The most draws the batch holds; the count buffer gives how many run.
    pub max_draws: u32,
    /// Its arguments are `DrawIndexedIndirectArgs` (else `DrawIndirectArgs`).
    pub indexed: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

macro_rules! as_bytes {
    ($($t:ty),*) => {$(
        impl $t {
            pub fn as_bytes(&self) -> &[u8] {
                bytemuck::bytes_of(self)
            }
        }
    )*};
}
as_bytes!(DrawIndirectArgs, DrawIndexedIndirectArgs, DispatchIndirectArgs);

// ---------------------------------------------------------------------------------------------
// Queries

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueryType {
    /// GPU time in ticks; `Queue::get_timestamp_period` gives nanoseconds per tick.
    Timestamp,
    /// Samples that passed the depth and stencil tests.
    Occlusion,
}

#[derive(Clone, Debug)]
pub struct QuerySetDescriptor<'a> {
    pub label: Option<&'a str>,
    pub ty: QueryType,
    pub count: u32,
}

/// Bytes one query result takes when resolved into a buffer.
pub const QUERY_SIZE: u32 = 8;
pub const QUERY_RESOLVE_BUFFER_ALIGNMENT: u64 = 256;

#[derive(Clone, Debug)]
pub struct PassTimestampWrites<'a, Q> {
    pub query_set: &'a Q,
    pub beginning_of_pass_write_index: Option<u32>,
    pub end_of_pass_write_index: Option<u32>,
}

// ---------------------------------------------------------------------------------------------
// Surfaces

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PresentMode {
    /// Vsync where the platform has it, else the closest.
    #[default]
    AutoVsync,
    /// No vsync where the platform has it, else the closest.
    AutoNoVsync,
    Fifo,
    FifoRelaxed,
    Immediate,
    Mailbox,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum CompositeAlphaMode {
    #[default]
    Auto,
    Opaque,
    PreMultiplied,
    PostMultiplied,
    Inherit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceConfiguration {
    pub usage: TextureUsages,
    pub format: TextureFormat,
    pub width: u32,
    pub height: u32,
    pub present_mode: PresentMode,
    /// Frames the CPU may run ahead of the display (2 is the usual).
    pub desired_maximum_frame_latency: u32,
    pub alpha_mode: CompositeAlphaMode,
    pub view_formats: Vec<TextureFormat>,
}

#[derive(Clone, Debug, Default)]
pub struct SurfaceCapabilities {
    /// The preferred format first.
    pub formats: Vec<TextureFormat>,
    pub present_modes: Vec<PresentMode>,
    pub alpha_modes: Vec<CompositeAlphaMode>,
    pub usages: TextureUsages,
}

/// What `Surface::get_current_texture` found.
pub enum CurrentSurfaceTexture<S> {
    Success(S),
    /// Usable, but the surface should be configured again (it changed size, say).
    Suboptimal(S),
    Timeout,
    /// The window is hidden or minimised: nothing to draw into.
    Occluded,
    /// Configure the surface again before the next frame.
    Outdated,
    /// Make a new surface.
    Lost,
}

// ---------------------------------------------------------------------------------------------
// Render bundles

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderBundleDepthStencil {
    pub format: TextureFormat,
    pub depth_read_only: bool,
    pub stencil_read_only: bool,
}

#[derive(Clone, Debug)]
pub struct RenderBundleEncoderDescriptor<'a> {
    pub label: Option<&'a str>,
    pub color_formats: &'a [Option<TextureFormat>],
    pub depth_stencil: Option<RenderBundleDepthStencil>,
    pub sample_count: u32,
}

// ---------------------------------------------------------------------------------------------
// Ray tracing (acceleration structures for WGSL ray queries)

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct AccelerationStructureFlags: u32 {
        const ALLOW_UPDATE = 1 << 0;
        const ALLOW_COMPACTION = 1 << 1;
        const PREFER_FAST_TRACE = 1 << 2;
        const PREFER_FAST_BUILD = 1 << 3;
        const LOW_MEMORY = 1 << 4;
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct AccelerationStructureGeometryFlags: u32 {
        const OPAQUE = 1 << 0;
        const NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AccelerationStructureUpdateMode {
    #[default]
    Build,
    PreferUpdate,
}

/// The shape of one triangle geometry of a bottom-level structure, fixed at creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlasTriangleGeometrySizeDescriptor {
    pub vertex_format: VertexFormat,
    pub vertex_count: u32,
    pub index_format: Option<IndexFormat>,
    pub index_count: Option<u32>,
    pub flags: AccelerationStructureGeometryFlags,
}

#[derive(Clone, Debug, Default)]
pub struct CreateBlasDescriptor<'a> {
    pub label: Option<&'a str>,
    pub flags: AccelerationStructureFlags,
    pub update_mode: AccelerationStructureUpdateMode,
}

#[derive(Clone, Debug)]
pub struct CreateTlasDescriptor<'a> {
    pub label: Option<&'a str>,
    pub max_instances: u32,
    pub flags: AccelerationStructureFlags,
    pub update_mode: AccelerationStructureUpdateMode,
}

/// The data of one triangle geometry for a build.
#[derive(Clone, Debug)]
pub struct BlasTriangleGeometry<'a, B> {
    pub size: &'a BlasTriangleGeometrySizeDescriptor,
    pub vertex_buffer: &'a B,
    pub first_vertex: u32,
    pub vertex_stride: u64,
    pub index_buffer: Option<&'a B>,
    pub first_index: Option<u32>,
}

/// One bottom-level build: `geometry` matches the sizes the structure was created with.
#[derive(Clone, Debug)]
pub struct BlasBuildEntry<'a, A, B> {
    pub blas: &'a A,
    pub geometry: Vec<BlasTriangleGeometry<'a, B>>,
}

/// An instance of a bottom-level structure in a top-level one.
#[derive(Clone, Debug)]
pub struct TlasInstance<A> {
    pub blas: A,
    /// Object to world, row-major 3×4.
    pub transform: [f32; 12],
    /// 24 bits, read back as `RayIntersection::instance_custom_data`.
    pub custom_data: u32,
    pub mask: u8,
}

impl<A> TlasInstance<A> {
    pub fn new(blas: A, transform: [f32; 12], custom_data: u32, mask: u8) -> Self {
        TlasInstance { blas, transform, custom_data, mask }
    }
}

/// Whether a texture view takes a place among the heap's storage textures
/// (`binding_array<texture_storage_2d<F, write>>`, heap bindings 12 to 15): one mip of a 2D
/// view of a texture made with `STORAGE_BINDING`, in a format shaders can write.
pub fn is_heap_storage_view(usage: TextureUsages, format: TextureFormat, dimension: TextureViewDimension, mip_level_count: u32) -> bool {
    usage.contains(TextureUsages::STORAGE_BINDING) && dimension == TextureViewDimension::D2 && mip_level_count == 1 && is_storage_view_format(format)
}

/// Whether a view in `format` can be a storage view at all. A texture made with
/// `STORAGE_BINDING` may still have views that are not: an sRGB view for sampling of an
/// `Rgba8Unorm` storage texture. Direct3D 12 removes the device on an sRGB, depth or
/// block-compressed UAV (`CreateUnorderedAccessView` returns nothing: an invalid one is
/// `DXGI_ERROR_INVALID_CALL`), and Vulkan wants such a view's usage without `STORAGE`.
pub fn is_storage_view_format(format: TextureFormat) -> bool {
    !format.has_depth() && !format.is_compressed() && !format.is_srgb()
}

#[cfg(test)]
mod storage_view_tests {
    use super::*;

    /// Ray tracing's texture atlas: an `Rgba8Unorm` storage texture with an sRGB view for
    /// sampling. The sRGB view must not be a storage view (Direct3D 12 removed the device on
    /// its UAV); the plain one is, and takes its heap slot.
    #[test]
    fn an_srgb_view_of_a_storage_texture_is_no_storage_view() {
        let usage = TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
        assert!(!is_storage_view_format(TextureFormat::Rgba8UnormSrgb));
        assert!(!is_heap_storage_view(usage, TextureFormat::Rgba8UnormSrgb, TextureViewDimension::D2, 1));
        assert!(is_storage_view_format(TextureFormat::Rgba8Unorm));
        assert!(is_heap_storage_view(usage, TextureFormat::Rgba8Unorm, TextureViewDimension::D2, 1));
        assert!(!is_storage_view_format(TextureFormat::Depth32Float));
        assert!(!is_storage_view_format(TextureFormat::Bc1RgbaUnorm));
    }
}

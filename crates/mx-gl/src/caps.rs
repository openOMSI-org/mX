//! What a GL context can do, read once when the adapter is made.

use glow::HasContext;
use mx_shader::glsl::Version;
use mx_types as t;

use crate::context::Api;

#[derive(Clone, Debug)]
pub(crate) struct Caps {
    /// (major, minor) of GL or GL ES.
    pub version: (u32, u32),
    pub glsl: Version,
    pub name: String,
    pub vendor: String,
    pub driver: String,
    /// Compute shaders (GL 4.3, ES 3.1).
    pub compute: bool,
    pub vertex_storage_blocks: u32,
    pub fragment_storage_blocks: u32,
    /// `glDrawElementsBaseVertex` (GL 3.2, ES 3.2).
    pub base_vertex: bool,
    /// `gl_BaseInstance` in shaders (GL 4.6 or `ARB_shader_draw_parameters`).
    pub draw_parameters: bool,
    /// `glCopyImageSubData` (GL 4.3, ES 3.2).
    pub copy_image: bool,
    /// Indirect draws (GL 4.0, ES 3.1).
    pub indirect: bool,
    pub texture_shadow_lod: bool,
    /// `GL_EXT_disjoint_timer_query` or desktop GL 3.3.
    pub timestamps: bool,
    /// S3TC / BPTC texture compression.
    pub bc: bool,
    /// `layout(binding = n)` in GLSL (GL 4.2, ES 3.1); else bindings are set after linking.
    pub explicit_bindings: bool,
    /// `glInvalidateFramebuffer` (GL 4.3, ES 3.0).
    pub invalidate: bool,
    pub vertex_texture_units: u32,
    pub fragment_texture_units: u32,
    pub combined_texture_units: u32,
    /// `GL_MAX_SAMPLES`.
    pub max_samples: u32,
    /// Multisampled textures, `glTexStorage2DMultisample` (GL 3.2, ES 3.1). Without them a
    /// multisampled render target is a renderbuffer (ES 3.0), which shaders cannot sample.
    pub multisample_textures: bool,
    pub limits: t::Limits,
}

impl Caps {
    pub fn read(gl: &glow::Context, api: Api) -> Caps {
        // SAFETY (all of the below): plain queries on a current context.
        unsafe {
            let v = gl.version();
            let version = (v.major, v.minor);
            let es = api == Api::Es;
            let at_least = |desktop: (u32, u32), gles: (u32, u32)| if es { version >= gles } else { version >= desktop };
            let ext = gl.supported_extensions();
            let has = |e: &str| ext.contains(e);
            let glsl = if es {
                let n = match version {
                    (3, 2..) => 320,
                    (3, 1) => 310,
                    _ => 300,
                };
                Version::Embedded { version: n, is_webgl: false }
            } else {
                let n = match version {
                    (4, 6..) => 460,
                    (4, 5) => 450,
                    (4, 4) => 440,
                    (4, 3) => 430,
                    (4, 2) => 420,
                    (4, 1) => 410,
                    (4, 0) => 400,
                    _ => 330,
                };
                Version::Desktop(n)
            };
            let compute = at_least((4, 3), (3, 1));
            let get = |p: u32| gl.get_parameter_i32(p).max(0) as u32;
            let (vertex_storage_blocks, fragment_storage_blocks) = if compute {
                (get(glow::MAX_VERTEX_SHADER_STORAGE_BLOCKS), get(glow::MAX_FRAGMENT_SHADER_STORAGE_BLOCKS))
            } else {
                (0, 0)
            };
            let max_2d = get(glow::MAX_TEXTURE_SIZE);
            let limits = t::Limits {
                max_texture_dimension_2d: max_2d,
                max_texture_dimension_3d: get(glow::MAX_3D_TEXTURE_SIZE),
                max_texture_array_layers: get(glow::MAX_ARRAY_TEXTURE_LAYERS),
                // (the heap is on the CPU: as large as wanted; what one draw reads is bounded
                // by the texture units)
                max_bindless_textures: 1 << 20,
                max_bindless_samplers: 1 << 16,
                max_bindless_buffers: if compute { 1 << 20 } else { 0 },
                buffer_view_offset_alignment: if compute { get(glow::SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT).max(4) } else { 256 },
                // (immediates are uniforms: wgpu-hal offers 256 bytes on GL)
                max_immediate_size: 256,
                max_vertex_buffers: get(glow::MAX_VERTEX_ATTRIBS).min(16),
                max_root_buffers: get(glow::MAX_COMBINED_UNIFORM_BLOCKS).min(13),
                max_color_attachments: get(glow::MAX_COLOR_ATTACHMENTS).min(8),
                max_buffer_size: 1 << 30,
                max_storage_buffer_binding_size: if compute { gl.get_parameter_i64(glow::MAX_SHADER_STORAGE_BLOCK_SIZE).max(0) as u64 } else { 0 },
                min_uniform_buffer_offset_alignment: 256,
                min_storage_buffer_offset_alignment: 256,
            };
            Caps {
                version,
                glsl,
                name: gl.get_parameter_string(glow::RENDERER),
                vendor: gl.get_parameter_string(glow::VENDOR),
                driver: gl.get_parameter_string(glow::VERSION),
                compute,
                vertex_storage_blocks,
                fragment_storage_blocks,
                base_vertex: at_least((3, 2), (3, 2)) || has("GL_EXT_draw_elements_base_vertex") || has("GL_OES_draw_elements_base_vertex"),
                draw_parameters: (!es && version >= (4, 6)) || has("GL_ARB_shader_draw_parameters"),
                copy_image: at_least((4, 3), (3, 2)) || has("GL_EXT_copy_image"),
                indirect: at_least((4, 0), (3, 1)),
                texture_shadow_lod: has("GL_EXT_texture_shadow_lod"),
                // (an ES driver may have the extension for elapsed time alone: ANGLE's counter
                // for timestamps has no bits, every timestamp read 0)
                timestamps: ((!es && version >= (3, 3)) || has("GL_EXT_disjoint_timer_query")) && timestamps_count(gl),
                bc: has("GL_EXT_texture_compression_s3tc") && (has("GL_EXT_texture_compression_bptc") || has("GL_ARB_texture_compression_bptc") || at_least((4, 2), (9, 9))),
                explicit_bindings: at_least((4, 2), (3, 1)),
                invalidate: at_least((4, 3), (3, 0)),
                max_samples: get(glow::MAX_SAMPLES),
                multisample_textures: at_least((3, 2), (3, 1)),
                vertex_texture_units: get(glow::MAX_VERTEX_TEXTURE_IMAGE_UNITS),
                fragment_texture_units: get(glow::MAX_TEXTURE_IMAGE_UNITS),
                combined_texture_units: get(glow::MAX_COMBINED_TEXTURE_IMAGE_UNITS),
                limits,
            }
        }
    }
}

/// Whether a timestamp query gives a time on this context: one is written and read back.
///
/// # Safety
/// The context is current.
unsafe fn timestamps_count(gl: &glow::Context) -> bool {
    // SAFETY: the caller's; a query made, used once and deleted here.
    unsafe {
        let Ok(q) = gl.create_query() else { return false };
        while gl.get_error() != glow::NO_ERROR {}
        gl.query_counter(q, glow::TIMESTAMP);
        gl.finish();
        let mut v = [0u8; 8];
        gl.get_query_parameter_u64_with_offset(q, glow::QUERY_RESULT, v.as_mut_ptr() as usize);
        let ok = gl.get_error() == glow::NO_ERROR && u64::from_le_bytes(v) > 0;
        gl.delete_query(q);
        ok
    }
}

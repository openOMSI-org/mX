//! Conformance tests: every backend must pass these with identical pixels.
//!
//! They run on the real GPU. Each renders off-screen, reads the pixels back and checks them.

use mx::*;

const W: u32 = 64;
const H: u32 = 64;

struct Gpu {
    device: Device,
    queue: Queue,
    _one_at_a_time: std::sync::MutexGuard<'static, ()>,
}

/// The tests take the GPU one at a time: with several Metal devices sampling GPU counters at
/// once (the test threads), the M4 sometimes returns timestamps of 0. A game has one device.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Whether the GPU lacks compute shaders or the buffer heap (OpenGL ES 3.0, OpenGL below 4.3:
/// ANGLE over older chips); the tests of them say so and pass.
fn lacks_compute() -> bool {
    let a = Instance::new().default_adapter().expect("a GPU mX can use");
    let lacks = !a.supports_compute() || a.limits().max_bindless_buffers == 0;
    if lacks {
        eprintln!("skipped: no compute shaders or storage buffers on this GPU");
    }
    lacks
}

fn gpu() -> Gpu {
    let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let adapter = Instance::new().default_adapter().expect("a GPU mX can use");
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("conformance"),
            ..Default::default()
        })
        .unwrap();
    Gpu { device, queue, _one_at_a_time: guard }
}

impl Gpu {
    fn target(&self, format: TextureFormat) -> (Texture, TextureView) {
        let t = self
            .device
            .create_texture(&TextureDescriptor {
                label: Some("target"),
                size: Extent3d {
                    width: W,
                    height: H,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
            .unwrap();
        let v = t.create_view(&Default::default()).unwrap();
        (t, v)
    }

    /// A 4×4 texture of one colour, in the heap.
    fn solid(&self, rgba: [u8; 4]) -> (Texture, TextureView) {
        let size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };
        let t = self
            .device
            .create_texture(&TextureDescriptor {
                label: Some("solid"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            })
            .unwrap();
        let data: Vec<u8> = rgba.iter().copied().cycle().take(4 * 4 * 4).collect();
        self.queue
            .write_texture(
                TexelCopyTextureInfo {
                    texture: &t,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &data,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16),
                    rows_per_image: None,
                },
                size,
            )
            .unwrap();
        let v = t.create_view(&Default::default()).unwrap();
        (t, v)
    }

    fn read(&self, mut enc: CommandEncoder, t: &Texture) -> Vec<[u8; 4]> {
        let bpr = W * 4;
        let rb = self
            .device
            .create_buffer(&BufferDescriptor {
                label: Some("readback"),
                size: (bpr * H) as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            })
            .unwrap();
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: t,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &rb,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(H),
                },
            },
            t.size(),
        );
        let i = self.queue.submit([enc.finish()]);
        self.device.wait(i).unwrap();
        rb.map_read(0..rb.size(), |b| b.as_chunks::<4>().0.to_vec()).unwrap()
    }

    fn shader(&self, src: &str) -> ShaderModule {
        self.device
            .create_shader_module(&ShaderModuleDescriptor { label: Some("test"), source: src })
            .unwrap_or_else(|e| panic!("{e}"))
    }

    fn pipeline(&self, m: &ShaderModule, format: TextureFormat, buffers: &[VertexBufferLayout<'_>], depth: Option<DepthStencilState>, constants: &[(&str, f64)]) -> RenderPipeline {
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("test"),
                vertex: VertexState {
                    module: m,
                    entry_point: None,
                    constants,
                    buffers,
                },
                primitive: PrimitiveState::default(),
                depth_stencil: depth,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: m,
                    entry_point: None,
                    constants,
                    targets: &[Some(format.into())],
                }),
            })
            .unwrap_or_else(|e| panic!("{e}"))
    }
}

fn pass<'a>(enc: &'a mut CommandEncoder, view: &'a TextureView, clear: Color) -> RenderPass<'a> {
    enc.begin_render_pass(&RenderPassDescriptor {
        label: Some("test pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(clear),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    })
}

const FULLSCREEN_VS: &str = "
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return Out(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(p.x, 1.0 - p.y));
}
";

#[test]
fn clear_and_read_back() {
    let g = gpu();
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    drop(pass(&mut enc, &v, Color { r: 1.0, g: 0.5, b: 0.0, a: 1.0 }));
    let px = g.read(enc, &t);
    assert!(px.iter().all(|p| *p == [255, 128, 0, 255]), "{:?}", px[0]);
}

#[test]
fn bindless_textures_chosen_by_immediates() {
    let g = gpu();
    let (_rt, red) = g.solid([255, 0, 0, 255]);
    let (_gt, green) = g.solid([0, 255, 0, 255]);
    let sampler = g.device.create_sampler(&SamplerDescriptor::default()).unwrap();
    assert_ne!(red.index(), green.index());
    let m = g.shader(&format!(
        "{FULLSCREEN_VS}
struct Draw {{ tex: u32, smp: u32 }}
var<immediate> draw: Draw;
@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{ return textureSample(mx_tex2d[draw.tex], mx_samplers[draw.smp], i.uv); }}"
    ));
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = pass(&mut enc, &v, Color::BLACK);
        rp.set_pipeline(&p);
        for (x, view) in [(0, &red), (W / 2, &green)] {
            rp.set_scissor_rect(x, 0, W / 2, H);
            rp.set_immediates(0, bytemuck::cast_slice(&[view.index(), sampler.index()]));
            rp.draw(0..3, 0..1);
        }
    }
    let px = g.read(enc, &t);
    assert_eq!(px[0], [255, 0, 0, 255]);
    assert_eq!(px[(W - 1) as usize], [0, 255, 0, 255]);
    assert_eq!(px[(W * H - 1) as usize], [0, 255, 0, 255]);
}

#[test]
fn root_bind_groups_as_in_wgpu() {
    let g = gpu();
    let m = g.shader(&format!(
        "{FULLSCREEN_VS}
struct Tint {{ c: vec4<f32> }}
@group(0) @binding(0) var<uniform> tint: Tint;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;
@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{ return tint.c * textureSample(tex, smp, i.uv); }}"
    ));
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    let tint = g
        .device
        .create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0.5f32, 1.0, 1.0, 1.0]),
            usage: BufferUsages::UNIFORM,
        })
        .unwrap();
    let (_wt, white) = g.solid([255, 255, 255, 255]);
    let sampler = g.device.create_sampler(&SamplerDescriptor::default()).unwrap();
    let g0 = g.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding { buffer: &tint, offset: 0, size: None }),
        }],
    });
    let g1 = g.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&white),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
        ],
    });
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = pass(&mut enc, &v, Color::BLACK);
        rp.set_pipeline(&p);
        rp.set_bind_group(0, &g0);
        rp.set_bind_group(1, &g1);
        rp.draw(0..3, 0..1);
    }
    let px = g.read(enc, &t);
    assert_eq!(px[100], [128, 255, 255, 255]);
}

#[test]
fn vertex_and_index_buffers_with_depth() {
    let g = gpu();
    let m = g.shader(
        "struct Out { @builtin(position) pos: vec4<f32>, @location(0) c: vec3<f32> }
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) c: vec3<f32>) -> Out { return Out(vec4<f32>(p, 1.0), c); }
@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return vec4<f32>(i.c, 1.0); }",
    );
    // two full-screen quads: blue near (z 0.2) drawn second-to-first behind red far (z 0.8)
    let v: [[f32; 6]; 8] = [
        [-1.0, -1.0, 0.2, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.2, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.2, 0.0, 0.0, 1.0],
        [-1.0, 1.0, 0.2, 0.0, 0.0, 1.0],
        [-1.0, -1.0, 0.8, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.8, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.8, 1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.8, 1.0, 0.0, 0.0],
    ];
    let idx: [u16; 6] = [0, 1, 2, 0, 2, 3];
    let vb = g
        .device
        .create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&v),
            usage: BufferUsages::VERTEX,
        })
        .unwrap();
    let ib = g
        .device
        .create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&idx),
            usage: BufferUsages::INDEX,
        })
        .unwrap();
    let attrs = vertex_attr_array![0 => Float32x3, 1 => Float32x3];
    let layout = VertexBufferLayout {
        array_stride: 24,
        step_mode: VertexStepMode::Vertex,
        attributes: &attrs,
    };
    let depth_state = DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
        stencil: Default::default(),
        bias: Default::default(),
    };
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[layout], Some(depth_state), &[]);
    let depth = g
        .device
        .create_texture(&TextureDescriptor {
            label: Some("depth"),
            size: Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .unwrap();
    let dv = depth.create_view(&Default::default()).unwrap();
    let (t, tv) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &tv,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &dv,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&p);
        rp.set_vertex_buffer(0, &vb, 0);
        rp.set_index_buffer(&ib, 0, IndexFormat::Uint16);
        rp.draw_indexed(0..6, 0, 0..1); // near, blue
        rp.draw_indexed(0..6, 4, 0..1); // far, red: hidden
    }
    let px = g.read(enc, &t);
    assert!(px.iter().all(|p| *p == [0, 0, 255, 255]), "{:?}", px[0]);
}

/// A pipeline with no fragment stage (depth only, as shadow maps and a depth prepass are
/// drawn) writes depth that a later pass tests against. OpenGL ES links no program without
/// a fragment shader: mX gives it an empty one (openOMSI's "shadow clear" failed there).
#[test]
fn depth_only_pipelines_draw_without_a_fragment_stage() {
    let g = gpu();
    let m = g.shader(
        "@vertex fn vs_left(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var q = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(-1.0, 1.0));
    return vec4<f32>(q[i], 0.2, 1.0);
}
@vertex fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.5, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }",
    );
    let depth_state = DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
        stencil: Default::default(),
        bias: Default::default(),
    };
    let depth_only = g
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("depth only"),
            vertex: VertexState {
                module: &m,
                entry_point: Some("vs_left"),
                constants: &[],
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(depth_state.clone()),
            multisample: MultisampleState::default(),
            fragment: None,
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let colour = g
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("colour"),
            vertex: VertexState {
                module: &m,
                entry_point: Some("vs_full"),
                constants: &[],
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(depth_state),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &m,
                entry_point: Some("fs"),
                constants: &[],
                targets: &[Some(TextureFormat::Rgba8Unorm.into())],
            }),
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let depth = g
        .device
        .create_texture(&TextureDescriptor {
            label: Some("depth"),
            size: Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .unwrap();
    let dv = depth.create_view(&Default::default()).unwrap();
    let (t, tv) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        // the depth-only pass: the left half at depth 0.2, no colour target at all
        let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
            label: Some("depth only"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &dv,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&depth_only);
        rp.draw(0..6, 0..1);
    }
    {
        // red over everything at 0.5: hidden where the depth-only pass drew
        let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
            label: Some("colour"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &tv,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &dv,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&colour);
        rp.draw(0..3, 0..1);
    }
    let px = g.read(enc, &t);
    let at = |x: u32, y: u32| px[(y * W + x) as usize];
    for y in [1, H / 2, H - 2] {
        assert_eq!(at(1, y), [0, 0, 0, 255], "left, row {y}: behind the depth-only quad");
        assert_eq!(at(W / 4, y), [0, 0, 0, 255], "left, row {y}");
        assert_eq!(at(W * 3 / 4, y), [255, 0, 0, 255], "right, row {y}");
        assert_eq!(at(W - 2, y), [255, 0, 0, 255], "right, row {y}");
    }
}

#[test]
fn pipeline_constants_reach_overrides() {
    let g = gpu();
    let m = g.shader(&format!(
        "{FULLSCREEN_VS}\noverride GREEN: f32 = 0.0;\n@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{ return vec4<f32>(0.0, GREEN, 0.0, 1.0); }}"
    ));
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[("GREEN", 1.0)]);
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = pass(&mut enc, &v, Color::BLACK);
        rp.set_pipeline(&p);
        rp.draw(0..3, 0..1);
    }
    assert_eq!(g.read(enc, &t)[0], [0, 255, 0, 255]);
}

#[test]
fn compute_with_runtime_sized_storage() {
    if lacks_compute() {
        return;
    }
    let g = gpu();
    let m = g.shader(
        "@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&dst)) { dst[id.x] = src[id.x] * 2u + arrayLength(&src); }
}",
    );
    let p = g
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            module: &m,
            entry_point: None,
            constants: &[],
        })
        .unwrap();
    let n = 100u32;
    let input: Vec<u32> = (0..n).collect();
    let src = g
        .device
        .create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input),
            usage: BufferUsages::STORAGE,
        })
        .unwrap();
    let dst = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: n as u64 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
        .unwrap();
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: n as u64 * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let bg = g.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding { buffer: &src, offset: 0, size: None }),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding { buffer: &dst, offset: 0, size: None }),
            },
        ],
    });
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&p);
        cp.set_bind_group(0, &bg);
        cp.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&dst, 0, &rb, 0, n as u64 * 4);
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    let out: Vec<u32> = rb.map_read(0..rb.size(), |b| bytemuck::cast_slice(b).to_vec()).unwrap();
    let want: Vec<u32> = (0..n).map(|x| x * 2 + n).collect();
    assert_eq!(out, want);
}

#[test]
fn heap_slots_are_reused_only_after_the_gpu_is_done() {
    let g = gpu();
    let (_t1, v1) = g.solid([1, 2, 3, 4]);
    let first = v1.index();
    // the slot is released but a submission may still read it: not handed out again yet
    drop(v1);
    let (_t2, v2) = g.solid([1, 2, 3, 4]);
    assert_ne!(v2.index(), first);
    let i = g.queue.submit([]);
    g.device.wait(i).unwrap();
    let (_t3, v3) = g.solid([1, 2, 3, 4]);
    assert_eq!(v3.index(), first, "a slot freed before a finished submission is reused");
}

#[test]
fn shader_errors_are_reported_not_panics() {
    let g = gpu();
    let e = g
        .device
        .create_shader_module(&ShaderModuleDescriptor {
            label: Some("bad"),
            source: "fn f() -> u32 { return 1.5; }",
        })
        .expect_err("an error");
    assert!(e.to_string().contains("bad"), "{e}");
}

// ---------------------------------------------------------------------------------------------
// Indirect draws, bundles, queries, ray queries

/// Quad `k` of four vertical stripes for instance `k`, coloured by `k`.
const STRIPES: &str = "
struct Out { @builtin(position) pos: vec4<f32>, @location(0) @interpolate(flat) k: u32 }
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) k: u32) -> Out {
    let c = array<vec2<f32>, 6>(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0));
    let p = c[v % 6u];
    return Out(vec4<f32>((f32(k) + p.x) / 2.0 - 1.0, p.y * 2.0 - 1.0, 0.0, 1.0), k);
}
@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return vec4<f32>(f32(i.k + 1u) / 4.0, 0.0, 0.0, 1.0); }
";

/// The red value of each of the four stripes (0 where nothing was drawn).
fn stripes(px: &[[u8; 4]]) -> [u8; 4] {
    let row = (H / 2 * W) as usize;
    std::array::from_fn(|k| px[row + k * (W as usize / 4) + 2][0])
}

fn buffer(g: &Gpu, data: &[u8], usage: BufferUsages) -> Buffer {
    g.device.create_buffer_init(&BufferInitDescriptor { label: None, contents: data, usage }).unwrap()
}

#[test]
fn indirect_draws_read_their_arguments_from_the_gpu() {
    let g = gpu();
    let m = g.shader(STRIPES);
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    let draws = [
        DrawIndirectArgs {
            vertex_count: 6,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        DrawIndirectArgs {
            vertex_count: 6,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 2,
        },
    ];
    let args = buffer(&g, bytemuck::cast_slice(&draws), BufferUsages::INDIRECT);
    let indexed = DrawIndexedIndirectArgs {
        index_count: 6,
        instance_count: 1,
        first_index: 0,
        base_vertex: 0,
        first_instance: 3,
    };
    let iargs = buffer(&g, indexed.as_bytes(), BufferUsages::INDIRECT);
    let indices = buffer(&g, bytemuck::cast_slice(&[0u16, 1, 2, 3, 4, 5]), BufferUsages::INDEX);
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = pass(&mut enc, &v, Color::BLACK);
        rp.set_pipeline(&p);
        rp.multi_draw_indirect(&args, 0, 2);
        rp.set_index_buffer(&indices, 0, IndexFormat::Uint16);
        rp.draw_indexed_indirect(&iargs, 0);
    }
    assert_eq!(stripes(&g.read(enc, &t)), [64, 0, 191, 255]);
}

#[test]
fn draws_read_one_uniform_buffer_at_their_own_offsets() {
    let g = gpu();
    let m = g.shader(&format!(
        "{FULLSCREEN_VS}
struct Tint {{ c: vec4<f32> }}
@group(0) @binding(0) var<uniform> tint: Tint;
@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{ return tint.c; }}"
    ));
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    // red at 0, green at 256, blue at 512 (one row of 256 bytes each)
    let mut rows = vec![0f32; 3 * 64];
    for (k, c) in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]].iter().enumerate() {
        rows[k * 64..k * 64 + 4].copy_from_slice(c);
    }
    let buf = g
        .device
        .create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&rows),
            usage: BufferUsages::UNIFORM,
        })
        .unwrap();
    let group = g.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &buf,
                offset: 0,
                size: Some(16),
            }),
        }],
    });
    // the right half drawn from a bundle: its offset is its own
    let mut be = g.device.create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
        label: None,
        color_formats: &[Some(TextureFormat::Rgba8Unorm)],
        depth_stencil: None,
        sample_count: 1,
    });
    be.set_pipeline(&p);
    be.set_bind_group(0, &group);
    be.set_buffer_offset(0, 0, 512);
    be.draw(0..3, 0..1);
    let bundle = be.finish(None);
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = pass(&mut enc, &v, Color::BLACK);
        rp.set_pipeline(&p);
        rp.set_bind_group(0, &group);
        rp.set_scissor_rect(0, 0, W / 2, H);
        rp.set_buffer_offset(0, 0, 256);
        rp.draw(0..3, 0..1);
        // (set again: back to the group's own offset)
        rp.set_scissor_rect(0, 0, W / 4, H);
        rp.set_bind_group(0, &group);
        rp.draw(0..3, 0..1);
        rp.set_scissor_rect(W / 2, 0, W / 2, H);
        rp.execute_bundles([&bundle]);
    }
    let px = g.read(enc, &t);
    assert_eq!(px[0], [255, 0, 0, 255], "the group's own offset");
    assert_eq!(px[(W / 2 - 1) as usize], [0, 255, 0, 255], "the pass's offset");
    assert_eq!(px[(W - 1) as usize], [0, 0, 255, 255], "the bundle's offset");
}

#[test]
fn render_bundles_replay_without_leaking_state() {
    let g = gpu();
    let m = g.shader(STRIPES);
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    let mut be = g.device.create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
        label: None,
        color_formats: &[Some(TextureFormat::Rgba8Unorm)],
        depth_stencil: None,
        sample_count: 1,
    });
    be.set_pipeline(&p);
    be.draw(0..6, 1..2);
    let bundle = be.finish(None);
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = pass(&mut enc, &v, Color::BLACK);
        rp.execute_bundles([&bundle, &bundle]);
        // the pass has no pipeline after the bundles: set one again
        rp.set_pipeline(&p);
        rp.draw(0..6, 3..4);
    }
    assert_eq!(stripes(&g.read(enc, &t)), [0, 128, 0, 255]);
}

#[test]
fn indirect_dispatch_and_clear_buffer() {
    if lacks_compute() {
        return;
    }
    let g = gpu();
    let m = g.shader(
        "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(4) fn main(@builtin(global_invocation_id) id: vec3<u32>) { out[id.x] = id.x + 1u; }",
    );
    let p = g
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            module: &m,
            entry_point: None,
            constants: &[],
        })
        .unwrap();
    let out = buffer(&g, &[0xAB; 64], BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST);
    let args = buffer(&g, DispatchIndirectArgs { x: 3, y: 1, z: 1 }.as_bytes(), BufferUsages::INDIRECT);
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let bg = g.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding { buffer: &out, offset: 0, size: None }),
        }],
    });
    let mut enc = g.device.create_command_encoder(None);
    enc.clear_buffer(&out, 48, None);
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&p);
        cp.set_bind_group(0, &bg);
        cp.dispatch_workgroups_indirect(&args, 0);
    }
    enc.clear_buffer(&out, 8, Some(8));
    enc.copy_buffer_to_buffer(&out, 0, &rb, 0, 64);
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    let got: Vec<u32> = rb.map_read(0..64, |b| bytemuck::cast_slice(b).to_vec()).unwrap();
    assert_eq!(got, [1, 2, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0]);
}

fn resolve(g: &Gpu, enc: &mut CommandEncoder, set: &QuerySet) -> Buffer {
    let n = set.count() as u64 * QUERY_SIZE as u64;
    let dst = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: n,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
        })
        .unwrap();
    enc.resolve_query_set(set, 0..set.count(), &dst, 0);
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: n,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    enc.copy_buffer_to_buffer(&dst, 0, &rb, 0, n);
    rb
}

#[test]
fn occlusion_queries_count_passing_samples() {
    let g = gpu();
    let m = g.shader(STRIPES);
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    let set = g
        .device
        .create_query_set(&QuerySetDescriptor {
            label: None,
            ty: QueryType::Occlusion,
            count: 2,
        })
        .unwrap();
    let (_t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &v,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: Some(&set),
        });
        rp.set_pipeline(&p);
        rp.begin_occlusion_query(0);
        rp.draw(0..6, 1..2);
        rp.end_occlusion_query();
        rp.begin_occlusion_query(1);
        rp.set_scissor_rect(0, 0, 1, 1); // stripe 1 does not cover the top-left pixel
        rp.draw(0..6, 1..2);
        rp.end_occlusion_query();
    }
    let rb = resolve(&g, &mut enc, &set);
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    let q: Vec<u64> = rb.map_read(0..16, |b| bytemuck::cast_slice(b).to_vec()).unwrap();
    assert!(q[0] > 0 && q[1] == 0, "{q:?}");
}

#[test]
fn timestamps_bracket_a_pass() {
    let adapter = Instance::new().default_adapter().unwrap();
    if !adapter.supports_timestamps() {
        eprintln!("skipped: no timestamp queries on this GPU");
        return;
    }
    let g = gpu();
    let m = g.shader(STRIPES);
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    let set = g
        .device
        .create_query_set(&QuerySetDescriptor {
            label: None,
            ty: QueryType::Timestamp,
            count: 2,
        })
        .unwrap();
    let (_t, v) = g.target(TextureFormat::Rgba8Unorm);
    // Apple's stage-boundary counters now and then give a pass a zero or an end before its
    // start (wgpu-hal samples the same way; WebGPU allows it). The mechanism has to work, so
    // one of a few tries must give a plausible interval.
    let mut seen = Vec::new();
    for _ in 0..5 {
        let mut enc = g.device.create_command_encoder(None);
        {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &v,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: Some(PassTimestampWrites {
                    query_set: &set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
                occlusion_query_set: None,
            });
            rp.set_pipeline(&p);
            rp.draw(0..6, 0..4);
        }
        let rb = resolve(&g, &mut enc, &set);
        let i = g.queue.submit([enc.finish()]);
        g.device.wait(i).unwrap();
        let q: Vec<u64> = rb.map_read(0..16, |b| bytemuck::cast_slice(b).to_vec()).unwrap();
        if q[0] > 0 && q[1] >= q[0] && ((q[1] - q[0]) as f64 * g.queue.get_timestamp_period() as f64) < 1e9 {
            return;
        }
        seen.push(q);
    }
    panic!("no plausible pass interval in 5 tries: {seen:?}");
}

#[test]
fn ray_queries_hit_instances_with_their_custom_data() {
    let adapter = Instance::new().default_adapter().unwrap();
    if !adapter.supports_ray_query() {
        eprintln!("skipped: no ray queries on this GPU");
        return;
    }
    let g = gpu();
    // one triangle around the origin in the z = 0 plane
    let verts: [f32; 9] = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0];
    let vb = buffer(&g, bytemuck::cast_slice(&verts), BufferUsages::BLAS_INPUT);
    let size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = g.device.create_blas(&Default::default(), &[size]).unwrap();
    let mut tlas = g
        .device
        .create_tlas(&CreateTlasDescriptor {
            label: None,
            max_instances: 2,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: Default::default(),
        })
        .unwrap();
    // instance 0 at x = 0, instance 1 moved to x = 10
    tlas[0] = Some(TlasInstance::new(blas.clone(), [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 7, 0xFF));
    tlas[1] = Some(TlasInstance::new(blas.clone(), [1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0x123456, 0xFF));
    let m = g.shader(
        "enable wgpu_ray_query;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var acc: acceleration_structure;
@compute @workgroup_size(3) fn main(@builtin(local_invocation_index) i: u32) {
    let x = array<f32, 3>(0.0, 10.0, 5.0)[i];
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 100.0, vec3<f32>(x, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0)));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_TRIANGLE) { out[i] = hit.instance_custom_data; } else { out[i] = 0xFFFFFFFFu; }
}",
    );
    let p = g
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            module: &m,
            entry_point: None,
            constants: &[],
        })
        .unwrap();
    let out = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 12,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
        .unwrap();
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 12,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let mut enc = g.device.create_command_encoder(None);
    let entry = BlasBuildEntry {
        blas: &blas,
        geometry: vec![BlasTriangleGeometry {
            size: &size,
            vertex_buffer: &vb,
            first_vertex: 0,
            vertex_stride: 12,
            index_buffer: None,
            first_index: None,
        }],
    };
    enc.build_acceleration_structures([&entry], [&tlas]);
    let bg = g.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding { buffer: &out, offset: 0, size: None }),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::AccelerationStructure(&tlas),
            },
        ],
    });
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&p);
        cp.set_bind_group(0, &bg);
        cp.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &rb, 0, 12);
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    let got: Vec<u32> = rb.map_read(0..12, |b| bytemuck::cast_slice(b).to_vec()).unwrap();
    assert_eq!(got, [7, 0x123456, u32::MAX]);
}

#[test]
fn msaa_resolves_partial_coverage_on_edges() {
    let g = gpu();
    let m = g.shader(
        "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0));
    return vec4<f32>(p[i], 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }",
    );
    let p = g
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            vertex: VertexState {
                module: &m,
                entry_point: None,
                constants: &[],
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState { count: 4, ..Default::default() },
            fragment: Some(FragmentState {
                module: &m,
                entry_point: None,
                constants: &[],
                targets: &[Some(TextureFormat::Rgba8Unorm.into())],
            }),
        })
        .unwrap();
    let ms = g
        .device
        .create_texture(&TextureDescriptor {
            label: Some("msaa"),
            size: Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 4,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .unwrap();
    let msv = ms.create_view(&Default::default()).unwrap();
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &msv,
                resolve_target: Some(&v),
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&p);
        rp.draw(0..3, 0..1);
    }
    let px = g.read(enc, &t);
    let at = |x: u32, y: u32| px[(y * W + x) as usize][0];
    assert_eq!(at(0, H - 1), 255, "inside");
    assert_eq!(at(W - 1, 0), 0, "outside");
    let edge = (0..W).map(|x| at(x, x)).filter(|r| *r > 0 && *r < 255).count();
    assert!(edge > W as usize / 2, "only {edge} partially covered pixels on the diagonal");
}

/// A compressed format's small mips are copied in whole blocks (4×4 into a 2×2 or 1×1
/// level, WebGPU's physical size): the block lands, and nothing else is touched (Metal
/// wants the level's real size, which mX passes, as wgpu-hal does).
#[test]
fn compressed_small_mips_take_whole_blocks() {
    let adapter = Instance::new().default_adapter().unwrap();
    if !adapter.supports_bc() {
        eprintln!("skipped: no BC formats on this GPU");
        return;
    }
    let g = gpu();
    let t = g
        .device
        .create_texture(&TextureDescriptor {
            label: Some("bc1"),
            size: Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 4,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bc1RgbaUnorm,
            usage: TextureUsages::COPY_DST | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .unwrap();
    // level 0: 2×2 blocks, then one block each for 4×4, 2×2 and 1×1
    for level in 0..4u32 {
        let blocks = if level == 0 { 2 } else { 1 };
        let data: Vec<u8> = (0..blocks * blocks * 8).map(|i| (i as u8).wrapping_mul(7).wrapping_add(level as u8 * 31)).collect();
        g.queue
            .write_texture(
                TexelCopyTextureInfo {
                    texture: &t,
                    mip_level: level,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &data,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(blocks * 8),
                    rows_per_image: None,
                },
                Extent3d {
                    width: blocks * 4,
                    height: blocks * 4,
                    depth_or_array_layers: 1,
                },
            )
            .unwrap();
    }
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 3 * 256,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let mut enc = g.device.create_command_encoder(None);
    for (k, level) in [1u32, 2, 3].into_iter().enumerate() {
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &t,
                mip_level: level,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &rb,
                layout: TexelCopyBufferLayout {
                    offset: k as u64 * 256,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
    }
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    let got: Vec<Vec<u8>> = rb.map_read(0..rb.size(), |b| (0..3).map(|k| b[k * 256..k * 256 + 8].to_vec()).collect()).unwrap();
    for (k, level) in [1u32, 2, 3].into_iter().enumerate() {
        let want: Vec<u8> = (0..8).map(|i| (i as u8).wrapping_mul(7).wrapping_add(level as u8 * 31)).collect();
        assert_eq!(got[k], want, "mip {level}");
    }
}

/// Read `w`×`h` texels of 4 bytes from mip `level`, layer `layer` of `t`.
fn read_texels(g: &Gpu, t: &Texture, level: u32, layer: u32, w: u32, h: u32) -> Vec<u8> {
    let row = (w * 4).div_ceil(256) * 256;
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: (row * h) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let mut enc = g.device.create_command_encoder(None);
    enc.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: t,
            mip_level: level,
            origin: Origin3d { x: 0, y: 0, z: layer },
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &rb,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    rb.map_read(0..rb.size(), |b| {
        (0..h as usize).flat_map(|y| b[y * row as usize..y * row as usize + (w * 4) as usize].to_vec()).collect()
    })
    .unwrap()
}

fn texture(g: &Gpu, size: u32, layers: u32, mips: u32, format: TextureFormat, usage: TextureUsages) -> Texture {
    g.device
        .create_texture(&TextureDescriptor {
            label: Some("zero-init"),
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
        .unwrap()
}

/// wgpu's guarantee: a texture or buffer reads as zeros until written, whatever the memory
/// held before. Memory is dirtied first (textures cleared to 0xAB and freed) so a backend
/// that hands out recycled memory as it is fails.
#[test]
fn new_textures_and_buffers_read_as_zeros() {
    let g = gpu();
    const S: u32 = 256;
    let rw = TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    for _ in 0..4 {
        let dirty: Vec<(Texture, TextureView)> = (0..16)
            .map(|_| {
                let t = texture(&g, S, 2, 2, TextureFormat::Rgba8Unorm, TextureUsages::RENDER_ATTACHMENT | rw);
                let v = t
                    .create_view(&TextureViewDescriptor {
                        dimension: Some(TextureViewDimension::D2),
                        mip_level_count: Some(1),
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                    .unwrap();
                (t, v)
            })
            .collect();
        let mut enc = g.device.create_command_encoder(None);
        for (_, v) in &dirty {
            drop(pass(&mut enc, v, Color { r: 0.67, g: 0.67, b: 0.67, a: 0.67 }));
        }
        let bufs: Vec<Buffer> = (0..8).map(|_| buffer(&g, &[0xAB; (S * S) as usize], BufferUsages::COPY_SRC | BufferUsages::STORAGE)).collect();
        let i = g.queue.submit([enc.finish()]);
        g.device.wait(i).unwrap();
        drop((dirty, bufs));
    }
    let zeros = |n: u32| vec![0u8; (n * n * 4) as usize];
    // nothing written
    let blank = texture(&g, S, 2, 2, TextureFormat::Rgba8Unorm, rw);
    let target = texture(&g, S, 2, 2, TextureFormat::Rgba8Unorm, TextureUsages::RENDER_ATTACHMENT | rw);
    // mip 0 of layer 0 written in part, layer 1 of mip 0 whole
    let part = texture(&g, S, 2, 2, TextureFormat::Rgba8Unorm, rw);
    let at = |layer| TexelCopyTextureInfo {
        texture: &part,
        mip_level: 0,
        origin: Origin3d { x: 0, y: 0, z: layer },
        aspect: TextureAspect::All,
    };
    let half = Extent3d {
        width: S / 2,
        height: S,
        depth_or_array_layers: 1,
    };
    g.queue
        .write_texture(
            at(0),
            &vec![7u8; (S / 2 * S * 4) as usize],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(S / 2 * 4),
                rows_per_image: None,
            },
            half,
        )
        .unwrap();
    g.queue
        .write_texture(
            at(1),
            &vec![9u8; (S * S * 4) as usize],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(S * 4),
                rows_per_image: None,
            },
            Extent3d {
                width: S,
                height: S,
                depth_or_array_layers: 1,
            },
        )
        .unwrap();
    let gpu_buffer = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: (S * S) as u64,
            usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
        })
        .unwrap();
    for t in [&blank, &target] {
        for (level, layer) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert!(
                read_texels(&g, t, level, layer, S >> level, S >> level) == zeros(S >> level),
                "mip {level} layer {layer} of a new texture is not zeros"
            );
        }
    }
    let p = read_texels(&g, &part, 0, 0, S, S);
    for (i, px) in p.chunks(4).enumerate() {
        let want = if (i as u32 % S) < S / 2 { [7; 4] } else { [0; 4] };
        assert_eq!(px, want, "texel {i} of the partly written layer");
    }
    assert!(read_texels(&g, &part, 0, 1, S, S).iter().all(|b| *b == 9));
    assert!(read_texels(&g, &part, 1, 0, S / 2, S / 2) == zeros(S / 2));
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: (S * S) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let mut enc = g.device.create_command_encoder(None);
    enc.copy_buffer_to_buffer(&gpu_buffer, 0, &rb, 0, (S * S) as u64);
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    assert!(rb.map_read(0..rb.size(), |b| b.iter().all(|x| *x == 0)).unwrap(), "a new buffer is not zeros");
}

/// `clear_texture` zeroes the mips and layers asked for and nothing else, for colour,
/// depth and compressed formats.
#[test]
fn clear_texture_zeroes_its_range() {
    let g = gpu();
    const S: u32 = 64;
    let rw = TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    for usage in [rw, rw | TextureUsages::RENDER_ATTACHMENT] {
        let t = texture(&g, S, 2, 2, TextureFormat::Rgba8Unorm, usage);
        for level in 0..2 {
            let n = S >> level;
            g.queue
                .write_texture(
                    TexelCopyTextureInfo {
                        texture: &t,
                        mip_level: level,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    &vec![5u8; (n * n * 4 * 2) as usize],
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(n * 4),
                        rows_per_image: Some(n),
                    },
                    Extent3d {
                        width: n,
                        height: n,
                        depth_or_array_layers: 2,
                    },
                )
                .unwrap();
        }
        let mut enc = g.device.create_command_encoder(None);
        enc.clear_texture(
            &t,
            &ImageSubresourceRange {
                base_mip_level: 1,
                base_array_layer: 1,
                ..Default::default()
            },
        );
        let i = g.queue.submit([enc.finish()]);
        g.device.wait(i).unwrap();
        for (level, layer, want) in [(0, 0, 5u8), (0, 1, 5), (1, 0, 5), (1, 1, 0)] {
            let n = S >> level;
            assert!(
                read_texels(&g, &t, level, layer, n, n).iter().all(|b| *b == want),
                "{usage:?}: mip {level} layer {layer} should be {want}"
            );
        }
    }
    // depth: cleared to 0.0
    let d = texture(&g, S, 1, 1, TextureFormat::Depth32Float, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC);
    let dv = d.create_view(&Default::default()).unwrap();
    let mut enc = g.device.create_command_encoder(None);
    drop(enc.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &dv,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(0.5),
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    }));
    enc.clear_texture(&d, &ImageSubresourceRange::default());
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    assert!(read_texels(&g, &d, 0, 0, S, S).iter().all(|b| *b == 0), "a cleared depth texture is not 0.0");
    // compressed
    if Instance::new().default_adapter().unwrap().supports_bc() {
        let c = texture(&g, 8, 1, 1, TextureFormat::Bc1RgbaUnorm, rw);
        g.queue
            .write_texture(
                TexelCopyTextureInfo {
                    texture: &c,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &[0x5A; 32],
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16),
                    rows_per_image: None,
                },
                Extent3d {
                    width: 8,
                    height: 8,
                    depth_or_array_layers: 1,
                },
            )
            .unwrap();
        let mut enc = g.device.create_command_encoder(None);
        enc.clear_texture(&c, &ImageSubresourceRange::default());
        let rb = g
            .device
            .create_buffer(&BufferDescriptor {
                label: None,
                size: 512,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            })
            .unwrap();
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &c,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &rb,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        let i = g.queue.submit([enc.finish()]);
        g.device.wait(i).unwrap();
        assert!(
            rb.map_read(0..512, |b| b[..16].iter().chain(&b[256..272]).all(|x| *x == 0)).unwrap(),
            "a cleared BC1 texture is not zeros"
        );
    }
}

/// `var<workgroup>` memory is shared by a workgroup's invocations (Metal needs its length
/// set for each dispatch): a reduction through it, two arrays of different sizes.
#[test]
fn workgroup_memory_is_shared() {
    if lacks_compute() {
        return;
    }
    let g = gpu();
    let m = g.shader(
        "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
var<workgroup> small: array<u32, 3>;
var<workgroup> sums: array<u32, 64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) w: vec3<u32>) {
    sums[i] = i + w.x * 1000u;
    if (i < 3u) { small[i] = 7u * (i + 1u); }
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s / 2u) {
        if (i < s) { sums[i] = sums[i] + sums[i + s]; }
        workgroupBarrier();
    }
    if (i == 0u) { out[w.x] = sums[0] + small[0] + small[1] + small[2]; }
}",
    );
    let p = g
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            module: &m,
            entry_point: None,
            constants: &[],
        })
        .unwrap();
    let out = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 8,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
        .unwrap();
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 8,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let bg = g.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding { buffer: &out, offset: 0, size: None }),
        }],
    });
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&p);
        cp.set_bind_group(0, &bg);
        cp.dispatch_workgroups(2, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &rb, 0, 8);
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    let got: Vec<u32> = rb.map_read(0..8, |b| b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()).unwrap();
    // 0 + 1 + .. + 63 = 2016, plus 64 × 1000 for the second workgroup, plus 7 + 14 + 21
    assert_eq!(got, [2016 + 42, 2016 + 64_000 + 42]);
}

/// Queue writes from staging memory that is reused: frames written back to back without
/// waiting, each read by the GPU right after its writes, see exactly their own bytes.
#[test]
fn queue_writes_of_many_frames_land_intact() {
    let g = gpu();
    const WRITES: usize = 600;
    const CHUNK: usize = 4096;
    const FRAMES: usize = 6;
    let size = (WRITES * CHUNK) as u64;
    let dst = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        })
        .unwrap();
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: size * FRAMES as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let byte = |frame: usize, w: usize| (frame * 31 + w * 7) as u8;
    let mut last = None;
    for frame in 0..FRAMES {
        for w in 0..WRITES {
            // (odd sizes: the next write's offset in the staging memory is not a multiple of 4)
            let n = CHUNK - (w % 3);
            g.queue.write_buffer(&dst, (w * CHUNK) as u64, &vec![byte(frame, w); n]).unwrap();
        }
        let mut enc = g.device.create_command_encoder(None);
        enc.copy_buffer_to_buffer(&dst, 0, &rb, size * frame as u64, size);
        last = Some(g.queue.submit([enc.finish()]));
        g.device.poll().unwrap();
    }
    g.device.wait(last.unwrap()).unwrap();
    rb.map_read(0..rb.size(), |b| {
        for frame in 0..FRAMES {
            for w in 0..WRITES {
                let at = frame * WRITES * CHUNK + w * CHUNK;
                let n = CHUNK - (w % 3);
                assert!(b[at..at + n].iter().all(|x| *x == byte(frame, w)), "frame {frame}, write {w}");
            }
        }
    })
    .unwrap();
}

/// Structures made with ALLOW_UPDATE and PreferUpdate are refitted after their first build:
/// moved triangles and moved instances are where the rays find them.
#[test]
fn acceleration_structures_refit() {
    let adapter = Instance::new().default_adapter().unwrap();
    if !adapter.supports_ray_query() {
        eprintln!("skipped: no ray queries on this GPU");
        return;
    }
    let g = gpu();
    let tri = |dx: f32| -> Vec<u8> { [-1.0 + dx, -1.0, 0.0, 1.0 + dx, -1.0, 0.0, dx, 1.0, 0.0].iter().flat_map(|f: &f32| f.to_le_bytes()).collect() };
    let vb = buffer(&g, &tri(0.0), BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST);
    let size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let update = (
        AccelerationStructureFlags::ALLOW_UPDATE | AccelerationStructureFlags::PREFER_FAST_BUILD,
        AccelerationStructureUpdateMode::PreferUpdate,
    );
    let blas = g
        .device
        .create_blas(
            &CreateBlasDescriptor {
                label: None,
                flags: update.0,
                update_mode: update.1,
            },
            &[size],
        )
        .unwrap();
    let mut tlas = g
        .device
        .create_tlas(&CreateTlasDescriptor {
            label: None,
            max_instances: 2,
            flags: update.0,
            update_mode: update.1,
        })
        .unwrap();
    let at = |x: f32| [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    tlas[0] = Some(TlasInstance::new(blas.clone(), at(0.0), 7, 0xFF));
    tlas[1] = Some(TlasInstance::new(blas.clone(), at(10.0), 9, 0xFF));
    let m = g.shader(
        "enable wgpu_ray_query;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var acc: acceleration_structure;
@compute @workgroup_size(4) fn main(@builtin(local_invocation_index) i: u32) {
    let x = array<f32, 4>(0.0, 10.0, 20.0, 23.0)[i];
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 100.0, vec3<f32>(x, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0)));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_TRIANGLE) { out[i] = hit.instance_custom_data; } else { out[i] = 0xFFFFFFFFu; }
}",
    );
    let p = g
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            module: &m,
            entry_point: None,
            constants: &[],
        })
        .unwrap();
    let out = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
        .unwrap();
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 16,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let run = |tlas: &Tlas| -> Vec<u32> {
        let mut enc = g.device.create_command_encoder(None);
        let entry = BlasBuildEntry {
            blas: &blas,
            geometry: vec![BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vb,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: None,
                first_index: None,
            }],
        };
        enc.build_acceleration_structures([&entry], [tlas]);
        let bg = g.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding { buffer: &out, offset: 0, size: None }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::AccelerationStructure(tlas),
                },
            ],
        });
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&p);
            cp.set_bind_group(0, &bg);
            cp.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &rb, 0, 16);
        let i = g.queue.submit([enc.finish()]);
        g.device.wait(i).unwrap();
        rb.map_read(0..16, |b| b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()).unwrap()
    };
    let miss = u32::MAX;
    assert_eq!(run(&tlas), [7, 9, miss, miss], "first build");
    // instance 1 moves to x = 20 (a top-level refit: same instances, new transform)
    tlas[1] = Some(TlasInstance::new(blas.clone(), at(20.0), 9, 0xFF));
    assert_eq!(run(&tlas), [7, miss, 9, miss], "top-level refit");
    // the triangle moves 3 m along x in the mesh itself (a bottom-level refit), seen through
    // both instances: at y = 0 it now spans x = 2.5..3.5 (+ 20 for instance 1)
    g.queue.write_buffer(&vb, 0, &tri(3.0)).unwrap();
    assert_eq!(run(&tlas), [miss, miss, miss, 9], "bottom-level refit");
}

/// Heap textures written on the GPU and read through the heap right after, in one command
/// buffer and across submissions: a copy into a texture, a render into a target, and many
/// frames of both. With a residency set (macOS 15+) only such textures are declared to the
/// passes: they must still see the writes.
#[test]
fn heap_reads_see_gpu_writes() {
    let g = gpu();
    let sampler = g.device.create_sampler(&SamplerDescriptor::default()).unwrap();
    let m = g.shader(&format!(
        "{FULLSCREEN_VS}
struct Draw {{ tex: u32, smp: u32 }}
var<immediate> draw: Draw;
@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{ return textureSample(mx_tex2d[draw.tex], mx_samplers[draw.smp], i.uv); }}"
    ));
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    let copied = g
        .device
        .create_texture(&TextureDescriptor {
            label: Some("copied into"),
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        })
        .unwrap();
    let copied_view = copied.create_view(&Default::default()).unwrap();
    let rendered = g
        .device
        .create_texture(&TextureDescriptor {
            label: Some("rendered"),
            size: Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .unwrap();
    let rendered_view = rendered.create_view(&Default::default()).unwrap();
    // (other textures in the heap, so the declared ones are a part of it)
    let _others: Vec<_> = (0..200).map(|i| g.solid([i as u8, 0, 0, 255])).collect();
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    for frame in 0..8u8 {
        let c = [frame * 30, 255 - frame * 30, 7, 255];
        // (rows 256 bytes apart: copies in an encoder need COPY_BYTES_PER_ROW_ALIGNMENT)
        let rows: Vec<u8> = (0..4).flat_map(|_| c.repeat(4).into_iter().chain(std::iter::repeat_n(0, 240))).collect();
        let src = buffer(&g, &rows, BufferUsages::COPY_SRC);
        let mut enc = g.device.create_command_encoder(None);
        // copy, then sample it in the same command buffer
        enc.copy_buffer_to_texture(
            TexelCopyBufferInfo {
                buffer: &src,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            TexelCopyTextureInfo {
                texture: &copied,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
        // render the copied colour into a target, then sample that target in the next pass
        {
            let mut rp = pass(&mut enc, &rendered_view, Color::BLACK);
            rp.set_pipeline(&p);
            rp.set_immediates(0, bytemuck::cast_slice(&[copied_view.index(), sampler.index()]));
            rp.draw(0..3, 0..1);
        }
        {
            let mut rp = pass(&mut enc, &v, Color::BLACK);
            rp.set_pipeline(&p);
            rp.set_immediates(0, bytemuck::cast_slice(&[rendered_view.index(), sampler.index()]));
            rp.draw(0..3, 0..1);
        }
        let px = g.read(enc, &t);
        assert!(px.iter().all(|p| *p == c), "frame {frame}: {:?}, want {c:?}", px[0]);
    }
}

fn heap_buffer(g: &Gpu, data: &[u8], usage: BufferUsages) -> Buffer {
    buffer(g, data, usage | BufferUsages::STORAGE)
}

#[test]
fn buffers_in_the_heap_need_no_bind_groups() {
    if lacks_compute() {
        return;
    }
    let g = gpu();
    // a typed read-only view at an offset, the words of a writable view, chosen by immediates
    let m = g.shader(
        "struct Item { a: u32, b: u32 }
struct Items { items: array<Item> }
@group(7) @binding(16) var<storage, read> items: binding_array<Items>;
struct Draw { src: u32, dst: u32, n: u32 }
var<immediate> draw: Draw;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < draw.n) {
        let it = items[draw.src].items[id.x];
        mx_rw_buffers[draw.dst].w[id.x] = it.a * 10u + it.b + mx_buffers[draw.src].w[0];
    }
}",
    );
    let p = g
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            module: &m,
            entry_point: None,
            constants: &[],
        })
        .unwrap();
    let n = 100u32;
    let align = g.device.limits().buffer_view_offset_alignment as usize;
    let mut words = vec![0u32; align / 4];
    words.extend((0..n).flat_map(|i| [i, i % 7]));
    let src = heap_buffer(&g, bytemuck::cast_slice(&words), BufferUsages::empty());
    let src_view = src
        .create_view(&BufferViewDescriptor {
            offset: align as u64,
            ..Default::default()
        })
        .unwrap();
    let dst = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: n as u64 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
        .unwrap();
    let dst_view = dst.create_view(&BufferViewDescriptor { writable: true, ..Default::default() }).unwrap();
    assert_ne!(src_view.index(), dst_view.index());
    assert_eq!(g.device.bindless_buffers_in_use(), 2);
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: n as u64 * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&p);
        cp.set_immediates(0, bytemuck::cast_slice(&[src_view.index(), dst_view.index(), n]));
        cp.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&dst, 0, &rb, 0, n as u64 * 4);
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    let out: Vec<u32> = rb.map_read(0..rb.size(), |b| bytemuck::cast_slice(b).to_vec()).unwrap();
    // (`mx_buffers[src].w[0]` is the view's first word: item 0's `a`, which is 0)
    let want: Vec<u32> = (0..n).map(|i| i * 10 + i % 7).collect();
    assert_eq!(out, want);
}

#[test]
fn fragments_read_heap_buffers_chosen_by_immediates() {
    if lacks_compute() {
        return;
    }
    let g = gpu();
    let red = heap_buffer(&g, bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 1.0]), BufferUsages::empty());
    let green = heap_buffer(&g, bytemuck::cast_slice(&[0.0f32, 1.0, 0.0, 1.0]), BufferUsages::empty());
    let (red, green) = (red.create_view(&Default::default()).unwrap(), green.create_view(&Default::default()).unwrap());
    let m = g.shader(&format!(
        "{FULLSCREEN_VS}
struct Colour {{ c: vec4<f32> }}
@group(7) @binding(20) var<storage, read> colours: binding_array<Colour>;
struct Draw {{ colour: u32 }}
var<immediate> draw: Draw;
@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{ return colours[draw.colour].c; }}"
    ));
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    let (t, v) = g.target(TextureFormat::Rgba8Unorm);
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut rp = pass(&mut enc, &v, Color::BLACK);
        rp.set_pipeline(&p);
        for (x, view) in [(0, &red), (W / 2, &green)] {
            rp.set_scissor_rect(x, 0, W / 2, H);
            rp.set_immediates(0, bytemuck::cast_slice(&[view.index()]));
            rp.draw(0..3, 0..1);
        }
    }
    let px = g.read(enc, &t);
    assert_eq!(px[0], [255, 0, 0, 255]);
    assert_eq!(px[(W - 1) as usize], [0, 255, 0, 255]);
}

#[test]
fn heap_buffers_see_writes_of_earlier_passes_and_copies() {
    if lacks_compute() {
        return;
    }
    let g = gpu();
    // pass 1 writes through the heap, a copy overwrites part of it, pass 2 reads both
    let fill = g.shader(
        "struct Draw { dst: u32 }
var<immediate> draw: Draw;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { mx_rw_buffers[draw.dst].w[id.x] = id.x + 1u; }",
    );
    let sum = g.shader(
        "struct Draw { src: u32, dst: u32 }
var<immediate> draw: Draw;
@compute @workgroup_size(1) fn main() {
    var s = 0u;
    for (var i = 0u; i < 64u; i++) { s += mx_buffers[draw.src].w[i]; }
    mx_rw_buffers[draw.dst].w[0] = s;
}",
    );
    let pipe = |m| {
        g.device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                module: m,
                entry_point: None,
                constants: &[],
            })
            .unwrap()
    };
    let (fill, sum) = (pipe(&fill), pipe(&sum));
    let data = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 256,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        })
        .unwrap();
    let out = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
        .unwrap();
    let rw = |b: &Buffer| b.create_view(&BufferViewDescriptor { writable: true, ..Default::default() }).unwrap();
    let (data_view, out_view) = (rw(&data), rw(&out));
    let twos = buffer(&g, bytemuck::cast_slice(&[2u32; 8]), BufferUsages::COPY_SRC);
    let rb = g
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })
        .unwrap();
    let mut enc = g.device.create_command_encoder(None);
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&fill);
        cp.set_immediates(0, bytemuck::cast_slice(&[data_view.index()]));
        cp.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&twos, 0, &data, 0, 32);
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&sum);
        cp.set_immediates(0, bytemuck::cast_slice(&[data_view.index(), out_view.index()]));
        cp.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &rb, 0, 4);
    let i = g.queue.submit([enc.finish()]);
    g.device.wait(i).unwrap();
    let got: u32 = rb.map_read(0..4, |b| bytemuck::cast_slice::<u8, u32>(b)[0]).unwrap();
    // 1..=64, with the first 8 replaced by 2
    assert_eq!(got, (9..=64).sum::<u32>() + 8 * 2);
}

#[test]
fn buffer_heap_slots_are_reused_only_after_the_gpu_is_done() {
    if lacks_compute() {
        return;
    }
    let g = gpu();
    let b = heap_buffer(&g, &[0; 16], BufferUsages::empty());
    let first = b.create_view(&Default::default()).unwrap();
    let slot = first.index();
    drop(first);
    let second = b.create_view(&Default::default()).unwrap();
    assert_ne!(second.index(), slot);
    let i = g.queue.submit([]);
    g.device.wait(i).unwrap();
    let third = b.create_view(&Default::default()).unwrap();
    assert_eq!(third.index(), slot, "a slot freed before a finished submission is reused");
}

#[test]
fn gpu_driven_draws_stop_at_the_gpus_count() {
    let g = gpu();
    if !Instance::new().default_adapter().unwrap().supports_indirect_count() {
        eprintln!("no GPU-driven draw counts on this device; skipped");
        return;
    }
    // instance i fills the i-th quarter of the target; 4 draws in the batch, the count says 3
    let m = g.shader(
        "struct Out { @builtin(position) pos: vec4<f32> }
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    let corner = vec2<f32>(f32((v + 1u) % 6u / 3u), f32(v % 2u));
    let x = (f32(i) + corner.x) * 0.5 - 1.0;
    return Out(vec4<f32>(x, corner.y * 2.0 - 1.0, 0.0, 1.0));
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    );
    let p = g.pipeline(&m, TextureFormat::Rgba8Unorm, &[], None, &[]);
    for indexed in [false, true] {
        let (t, v) = g.target(TextureFormat::Rgba8Unorm);
        let usage = BufferUsages::STORAGE | BufferUsages::INDIRECT;
        let (args, index) = if indexed {
            let a: Vec<DrawIndexedIndirectArgs> = (0..4)
                .map(|i| DrawIndexedIndirectArgs {
                    index_count: 6,
                    instance_count: 1,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: i,
                })
                .collect();
            (
                buffer(&g, bytemuck::cast_slice(&a), usage),
                Some(buffer(&g, bytemuck::cast_slice(&[0u16, 1, 2, 3, 4, 5]), BufferUsages::INDEX)),
            )
        } else {
            let a: Vec<DrawIndirectArgs> = (0..4)
                .map(|i| DrawIndirectArgs {
                    vertex_count: 6,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: i,
                })
                .collect();
            (buffer(&g, bytemuck::cast_slice(&a), usage), None)
        };
        let count = buffer(&g, bytemuck::cast_slice(&[3u32]), BufferUsages::INDIRECT);
        let draws = g
            .device
            .create_indirect_draws(&IndirectDrawsDescriptor {
                label: Some("batch"),
                max_draws: 4,
                indexed,
            })
            .unwrap();
        let mut enc = g.device.create_command_encoder(None);
        enc.encode_indirect_draws(&draws, &p, &args, 0, &count, 0, index.as_ref().map(|b| (b, 0, IndexFormat::Uint16)));
        {
            let mut rp = pass(&mut enc, &v, Color::BLACK);
            rp.set_pipeline(&p);
            rp.execute_indirect_draws(&draws);
        }
        let px = g.read(enc, &t);
        let row = (H / 2 * W) as usize;
        let quarter = |q: u32| px[row + (q * W / 4 + W / 8) as usize][0];
        assert_eq!([quarter(0), quarter(1), quarter(2), quarter(3)], [255, 255, 255, 0], "indexed: {indexed}");
    }
}

/// `Surface::is_supported_by` gives an answer: on every backend but Vulkan it asked for the
/// surface's capabilities while still holding the surface's lock, which `get_capabilities`
/// takes again, and the thread waited for itself (openOMSI hung at start on Direct3D 12).
/// The noop backend takes any window handle, so no window is needed.
#[test]
fn a_surface_answers_whether_an_adapter_can_present_to_it() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let instance = Instance::with_backends(Backends::NOOP);
        let adapter = instance.enumerate_adapters().into_iter().next().expect("the noop adapter");
        let window = raw_window_handle::Win32WindowHandle::new(std::num::NonZeroIsize::new(1).unwrap());
        let display = raw_window_handle::WindowsDisplayHandle::new();
        // SAFETY: the noop backend keeps nothing of the handles.
        let surface = unsafe { instance.create_surface_unsafe(display.into(), window.into()) }.expect("a noop surface");
        let _ = tx.send(surface.is_supported_by(&adapter));
    });
    let supported = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("is_supported_by did not return");
    assert!(supported, "the noop adapter can present to a noop surface");
}

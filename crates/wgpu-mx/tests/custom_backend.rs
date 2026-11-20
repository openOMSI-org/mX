//! mX behind a real `wgpu::Instance` (the `custom` feature): wgpu's own API, mX underneath.

use wgpu::util::DeviceExt;

fn now<F: std::future::Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop())) {
        std::task::Poll::Ready(v) => v,
        std::task::Poll::Pending => panic!("future not ready"),
    }
}

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn open() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu_mx::custom::instance();
    let adapter = now(instance.request_adapter(&Default::default())).expect("an mX adapter");
    assert!(adapter.get_info().driver.starts_with("mX"), "{:?}", adapter.get_info());
    let (device, queue) = now(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        ..Default::default()
    }))
    .unwrap();
    assert!(wgpu_mx::custom::is_mx(&device));
    (device, queue)
}

fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let v = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    v
}

/// A full-screen draw that samples a texture through a bind group and reads a colour from a
/// uniform buffer at a dynamic offset, then a copy back.
#[test]
fn render_with_bind_groups_and_dynamic_offsets() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (device, queue) = open();
    let size = wgpu::Extent3d {
        width: 64,
        height: 64,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let tex = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("texel"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &[0, 128, 0, 255],
    );
    // two 256-byte slots: red in the first, blue in the second
    let mut uniforms = vec![0u8; 512];
    uniforms[..16].copy_from_slice(&[1f32, 0.0, 0.0, 0.0].map(f32::to_le_bytes).concat());
    uniforms[256..272].copy_from_slice(&[0f32, 0.0, 1.0, 0.0].map(f32::to_le_bytes).concat());
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &uniforms,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
    let view = tex.create_view(&Default::default());
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &ubuf,
                    offset: 0,
                    size: wgpu::BufferSize::new(16),
                }),
            },
        ],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("draw"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var<uniform> tint: vec4<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return textureSample(t, s, vec2<f32>(0.5)) + tint; }"
                .into(),
        ),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("draw"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let tv = target.create_view(&Default::default());
    for (offset, want) in [(0u32, [255u8, 128, 0, 255]), (256, [0, 128, 255, 255])] {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &tv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&pipeline);
            rp.set_bind_group(0, &group, &[offset]);
            rp.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            size,
        );
        queue.submit([enc.finish()]);
        let px = read(&device, &rb);
        assert_eq!(&px[..4], &want, "dynamic offset {offset}");
        assert_eq!(&px[256 * 63 + 63 * 4..256 * 63 + 64 * 4], &want);
    }
}

/// What mX cannot create is reported as wgpu reports it, so the caller can fall back: an
/// error in the error scope for a resource, an `Err` for a device.
#[test]
fn failures_are_errors_not_panics() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (device, _queue) = open();
    // (over the device's limit a buffer is refused as wgpu refuses it, a validation error;
    // under it, a device without the memory gives out-of-memory: either is an error, not a panic)
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let invalid = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _huge = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("64 TiB"),
        size: 1 << 46,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let (invalid, oom) = (now(invalid.pop()), now(oom.pop()));
    assert!(
        matches!(invalid, Some(wgpu::Error::Validation { .. })) || matches!(oom, Some(wgpu::Error::OutOfMemory { .. })),
        "no error for a 64 TiB buffer: {invalid:?} / {oom:?}"
    );

    let adapter = now(wgpu_mx::custom::instance().request_adapter(&Default::default())).unwrap();
    let missing = wgpu::Features::all() - adapter.features();
    assert!(!missing.is_empty());
    assert!(
        now(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: missing,
            ..Default::default()
        }))
        .is_err()
    );
}

/// A compute pass, a buffer mapped at creation, and a shader error caught by an error scope.
#[test]
fn compute_mapping_and_error_scopes() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (device, queue) = open();
    let data = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    data.slice(..).get_mapped_range_mut().copy_from_slice(&[1u32, 2, 3, 4].map(u32::to_le_bytes).concat());
    data.unmap();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read_write> v: array<u32, 4>;\n@compute @workgroup_size(4) fn main(@builtin(local_invocation_index) i: u32) { v[i] = v[i] * 10u; }".into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: data.as_entire_binding(),
        }],
    });
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipeline);
        cp.set_bind_group(0, &group, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&data, 0, &rb, 0, 16);
    queue.submit([enc.finish()]);
    assert_eq!(read(&device, &rb), [10u32, 20, 30, 40].map(u32::to_le_bytes).concat());

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _bad = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bad"),
        source: wgpu::ShaderSource::Wgsl("fn broken( {".into()),
    });
    let err = now(scope.pop());
    assert!(matches!(err, Some(wgpu::Error::Validation { .. })), "{err:?}");
}

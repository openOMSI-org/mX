//! Instance, adapters, the device and its queue; submissions and deferred destruction.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString, c_void};
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ash::{ext, khr, vk};
use gpu_allocator::vulkan::{Allocation, Allocator, AllocatorCreateDesc};
use mx_types as t;
use parking_lot::Mutex;

use crate::heap::Heap;
use crate::{CommandBuffer, Error, Result, vk_err};

// ---------------------------------------------------------------------------------------------
// Instance

pub(crate) struct InstanceShared {
    pub entry: ash::Entry,
    pub raw: ash::Instance,
    pub surface: Option<khr::surface::Instance>,
    pub extensions: Vec<&'static CStr>,
    debug: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl InstanceShared {
    pub fn has(&self, ext: &CStr) -> bool {
        self.extensions.contains(&ext)
    }
}

impl Drop for InstanceShared {
    fn drop(&mut self) {
        // SAFETY: everything made from the instance holds an `Arc` to it, so this is last.
        unsafe {
            if let Some((d, m)) = self.debug.take() {
                d.destroy_debug_utils_messenger(m, None);
            }
            self.raw.destroy_instance(None);
        }
    }
}

/// The entry point: the Vulkan loader and an instance.
#[derive(Default)]
pub struct Instance {
    pub(crate) shared: Option<Arc<InstanceShared>>,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader passes valid callback data for the duration of the call.
    let msg = unsafe { data.as_ref().and_then(|d| d.message_as_c_str()).map(|m| m.to_string_lossy().into_owned()) }.unwrap_or_default();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("Vulkan: {msg}");
        eprintln!("[mX Vulkan validation error] {msg}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("Vulkan: {msg}");
    } else {
        log::debug!("Vulkan: {msg}");
    }
    vk::FALSE
}

/// The Vulkan loader: `MX_VULKAN_LIBRARY` if set, else the system's, else (Apple) the usual
/// Homebrew / SDK places and MoltenVK itself.
fn load_entry() -> Option<ash::Entry> {
    if let Ok(path) = std::env::var("MX_VULKAN_LIBRARY") {
        // SAFETY: loading a Vulkan loader the user named.
        return unsafe { ash::Entry::load_from(path) }.ok();
    }
    // SAFETY: loading the system's Vulkan loader.
    if let Ok(e) = unsafe { ash::Entry::load() } {
        return Some(e);
    }
    #[cfg(target_vendor = "apple")]
    for p in [
        "libvulkan.1.dylib",
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "libMoltenVK.dylib",
        "/opt/homebrew/lib/libMoltenVK.dylib",
        "/usr/local/lib/libMoltenVK.dylib",
    ] {
        // SAFETY: as above.
        if let Ok(e) = unsafe { ash::Entry::load_from(p) } {
            return Some(e);
        }
    }
    None
}

fn create_instance() -> Option<Arc<InstanceShared>> {
    let entry = load_entry()?;
    // SAFETY: plain queries and creation through a loaded entry.
    unsafe {
        let version = entry.try_enumerate_instance_version().ok()?.unwrap_or(vk::API_VERSION_1_0);
        if version < vk::API_VERSION_1_2 {
            log::info!("mX: the Vulkan loader is older than 1.2");
            return None;
        }
        let available: Vec<CString> = entry
            .enumerate_instance_extension_properties(None)
            .ok()?
            .iter()
            .filter_map(|p| p.extension_name_as_c_str().ok().map(CStr::to_owned))
            .collect();
        let wanted: [&'static CStr; 9] = [
            khr::surface::NAME,
            ext::metal_surface::NAME,
            khr::win32_surface::NAME,
            khr::xlib_surface::NAME,
            khr::xcb_surface::NAME,
            khr::wayland_surface::NAME,
            khr::android_surface::NAME,
            khr::portability_enumeration::NAME,
            ext::debug_utils::NAME,
        ];
        let extensions: Vec<&'static CStr> = wanted.into_iter().filter(|w| available.iter().any(|a| a.as_c_str() == *w)).collect();
        let ext_ptrs: Vec<*const std::ffi::c_char> = extensions.iter().map(|e| e.as_ptr()).collect();
        let api = version.min(vk::API_VERSION_1_3);
        let app = vk::ApplicationInfo::default().application_name(c"mX").engine_name(c"mX").api_version(api);
        let mut flags = vk::InstanceCreateFlags::empty();
        if extensions.contains(&khr::portability_enumeration::NAME) {
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let info = vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&ext_ptrs).flags(flags);
        let raw = match entry.create_instance(&info, None) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("mX: vkCreateInstance failed: {e}");
                return None;
            }
        };
        let surface = extensions.contains(&khr::surface::NAME).then(|| khr::surface::Instance::new(&entry, &raw));
        let debug = if extensions.contains(&ext::debug_utils::NAME) {
            let d = ext::debug_utils::Instance::new(&entry, &raw);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
                .pfn_user_callback(Some(debug_callback));
            d.create_debug_utils_messenger(&info, None).ok().map(|m| (d, m))
        } else {
            None
        };
        Some(Arc::new(InstanceShared {
            entry,
            raw,
            surface,
            extensions,
            debug,
        }))
    }
}

impl Instance {
    /// Load Vulkan and make an instance. Without a usable loader the instance lists no
    /// adapters.
    pub fn new() -> Instance {
        Instance { shared: create_instance() }
    }

    /// Every device that can run mX, discrete GPUs first.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let Some(inst) = &self.shared else { return Vec::new() };
        // SAFETY: a plain query.
        let phys = unsafe { inst.raw.enumerate_physical_devices() }.unwrap_or_default();
        let mut list: Vec<Adapter> = phys.into_iter().filter_map(|p| Adapter::probe(inst, p)).collect();
        list.sort_by_key(|a| match a.0.props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 4,
            _ => 3,
        });
        list
    }

    pub fn default_adapter(&self) -> Result<Adapter> {
        self.enumerate_adapters().into_iter().next().ok_or(Error::NoDevice)
    }

    /// The `VkInstance`, for interop (OpenXR).
    pub fn raw(&self) -> Option<&ash::Instance> {
        self.shared.as_ref().map(|s| &s.raw)
    }
}

// ---------------------------------------------------------------------------------------------
// Adapter

/// What a physical device offers mX.
#[derive(Clone)]
pub(crate) struct Caps {
    pub api_version: u32,
    /// Dynamic rendering and synchronization2 are core (1.3), not extensions.
    pub core13: bool,
    pub family: u32,
    pub ray_query: bool,
    /// Buffer device addresses (Vulkan 1.2 `bufferDeviceAddress`): on whenever there, as
    /// DLSS asks for them.
    pub device_address: bool,
    pub timestamps: bool,
    pub bc: bool,
    pub d24s8: bool,
    pub zero_init: bool,
    pub portability: bool,
    pub swapchain: bool,
    pub multi_draw_indirect: bool,
    /// `drawIndirectCount` (Vulkan 1.2): GPU-driven draw counts.
    pub draw_indirect_count: bool,
    pub precise_occlusion: bool,
    pub anisotropy: bool,
    pub robust_buffer_access: bool,
    pub robust_image_access: bool,
    pub robustness2: bool,
    pub robust_buffer_access2: bool,
    pub robust_image_access2: bool,
    pub image_robustness_ext: bool,
    /// `robustImageAccess` (1.3 / VK_EXT_image_robustness) itself.
    pub image_robustness_supported: bool,
    pub core: vk::PhysicalDeviceFeatures,
    pub f11_draw_parameters: bool,
    pub f11_storage16: bool,
    pub f12_int8: bool,
    pub f12_float16: bool,
    pub f12_ubo_std_layout: bool,
    pub max_storage_buffer_range: u64,
    pub update_unused_while_pending: bool,
    /// Storage images in the heap (`descriptorBindingStorageImageUpdateAfterBind`).
    pub heap_storage_images: bool,
    /// `shaderStorageImageArrayNonUniformIndexing`.
    pub storage_image_non_uniform: bool,
    /// Acceleration structures in the heap (`descriptorBindingAccelerationStructureUpdateAfterBind`).
    pub heap_acceleration_structures: bool,
    pub limits: t::Limits,
    pub timestamp_period: f32,
    pub scratch_alignment: u64,
}

pub(crate) struct AdapterInner {
    pub instance: Arc<InstanceShared>,
    pub phys: vk::PhysicalDevice,
    pub props: vk::PhysicalDeviceProperties,
    pub caps: Caps,
}

#[derive(Clone)]
pub struct Adapter(pub(crate) Arc<AdapterInner>);

fn has_ext(list: &[CString], name: &CStr) -> bool {
    list.iter().any(|e| e.as_c_str() == name)
}

impl Adapter {
    fn probe(inst: &Arc<InstanceShared>, phys: vk::PhysicalDevice) -> Option<Adapter> {
        let raw = &inst.raw;
        // SAFETY: plain queries of a physical device of this instance; the feature chains
        // point at locals that outlive the calls.
        unsafe {
            let props = raw.get_physical_device_properties(phys);
            let name = props.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if props.api_version < vk::API_VERSION_1_2 {
                log::info!(
                    "mX: {name} is Vulkan {}.{}, mX needs 1.2",
                    vk::api_version_major(props.api_version),
                    vk::api_version_minor(props.api_version)
                );
                return None;
            }
            let extensions: Vec<CString> = raw
                .enumerate_device_extension_properties(phys)
                .ok()?
                .iter()
                .filter_map(|p| p.extension_name_as_c_str().ok().map(CStr::to_owned))
                .collect();
            let core13 = props.api_version >= vk::API_VERSION_1_3;
            let ray_exts = has_ext(&extensions, khr::acceleration_structure::NAME) && has_ext(&extensions, khr::ray_query::NAME) && has_ext(&extensions, khr::deferred_host_operations::NAME);
            let portability = has_ext(&extensions, khr::portability_subset::NAME);
            let robustness2 = has_ext(&extensions, ext::robustness2::NAME);
            let image_robustness_ext = !core13 && has_ext(&extensions, ext::image_robustness::NAME);

            let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut dynr = vk::PhysicalDeviceDynamicRenderingFeatures::default();
            let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default();
            let mut imgrob = vk::PhysicalDeviceImageRobustnessFeatures::default();
            let mut rob2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
            let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
            let mut rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12);
            if core13 {
                f2 = f2.push_next(&mut f13);
            } else {
                f2 = f2.push_next(&mut dynr).push_next(&mut sync2);
                if image_robustness_ext {
                    f2 = f2.push_next(&mut imgrob);
                }
            }
            if robustness2 {
                f2 = f2.push_next(&mut rob2);
            }
            if ray_exts {
                f2 = f2.push_next(&mut accel).push_next(&mut rq);
            }
            raw.get_physical_device_features2(phys, &mut f2);
            let core = f2.features;

            let (dynamic_rendering, synchronization2, zero_init, robust_image_access_13) = if core13 {
                (f13.dynamic_rendering, f13.synchronization2, f13.shader_zero_initialize_workgroup_memory, f13.robust_image_access)
            } else {
                (dynr.dynamic_rendering, sync2.synchronization2, 0, imgrob.robust_image_access)
            };
            let missing: Vec<&str> = [
                (f12.descriptor_indexing, "descriptorIndexing"),
                (f12.runtime_descriptor_array, "runtimeDescriptorArray"),
                (f12.descriptor_binding_partially_bound, "descriptorBindingPartiallyBound"),
                (f12.descriptor_binding_sampled_image_update_after_bind, "descriptorBindingSampledImageUpdateAfterBind"),
                (f12.shader_sampled_image_array_non_uniform_indexing, "shaderSampledImageArrayNonUniformIndexing"),
                (f12.descriptor_binding_storage_buffer_update_after_bind, "descriptorBindingStorageBufferUpdateAfterBind"),
                (f12.shader_storage_buffer_array_non_uniform_indexing, "shaderStorageBufferArrayNonUniformIndexing"),
                (f12.timeline_semaphore, "timelineSemaphore"),
                (dynamic_rendering, "dynamicRendering"),
                (synchronization2, "synchronization2"),
            ]
            .into_iter()
            .filter(|(v, _)| *v == 0)
            .map(|(_, n)| n)
            .collect();
            if !missing.is_empty() || (!core13 && !(has_ext(&extensions, khr::dynamic_rendering::NAME) && has_ext(&extensions, khr::synchronization2::NAME))) {
                log::info!("mX: {name} lacks {missing:?}");
                return None;
            }
            let families = raw.get_physical_device_queue_family_properties(phys);
            let family = families.iter().position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))? as u32;

            let l = &props.limits;
            let mut di = vk::PhysicalDeviceDescriptorIndexingProperties::default();
            let mut asp = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut di);
            if ray_exts {
                p2 = p2.push_next(&mut asp);
            }
            raw.get_physical_device_properties2(phys, &mut p2);
            let samplers = (di.max_descriptor_set_update_after_bind_samplers.min(di.max_per_stage_descriptor_update_after_bind_samplers) / 2).min(4096);
            let resources = di.max_per_stage_update_after_bind_resources.saturating_sub(2 * samplers + 256);
            // (an eighth of the per-stage resources for buffer views, the rest for textures)
            let buffers = di
                .max_descriptor_set_update_after_bind_storage_buffers
                .min(di.max_per_stage_descriptor_update_after_bind_storage_buffers)
                .min(resources / 8)
                .min(500_000);
            let resources = resources - buffers;
            // (ray queries with the acceleration structures in the heap, `Tlas::index`: every
            // driver with ray queries updates them after binding)
            let ray_query =
                ray_exts && accel.acceleration_structure != 0 && accel.descriptor_binding_acceleration_structure_update_after_bind != 0 && rq.ray_query != 0 && f12.buffer_device_address != 0;
            let heap_storage_images = f12.descriptor_binding_storage_image_update_after_bind != 0;
            let heap_acceleration_structures = ray_query;
            // (the texture slots: a sampled image per texture binding, a storage image and an
            // acceleration structure each)
            let texture_bindings = mx_shader::heap::SAMPLERS + heap_storage_images as u32 + heap_acceleration_structures as u32;
            let mut textures = (di
                .max_descriptor_set_update_after_bind_sampled_images
                .min(di.max_per_stage_descriptor_update_after_bind_sampled_images)
                .min(resources)
                / texture_bindings)
                .min(500_000);
            if heap_storage_images {
                textures = textures.min(di.max_descriptor_set_update_after_bind_storage_images.min(di.max_per_stage_descriptor_update_after_bind_storage_images));
            }
            if heap_acceleration_structures {
                textures = textures.min(
                    asp.max_descriptor_set_update_after_bind_acceleration_structures
                        .min(asp.max_per_stage_descriptor_update_after_bind_acceleration_structures),
                );
            }
            let limits = t::Limits {
                max_texture_dimension_2d: l.max_image_dimension2_d,
                max_texture_dimension_3d: l.max_image_dimension3_d,
                max_texture_array_layers: l.max_image_array_layers,
                max_bindless_textures: textures,
                max_bindless_samplers: samplers,
                max_bindless_buffers: buffers,
                buffer_view_offset_alignment: (l.min_storage_buffer_offset_alignment as u32).max(4),
                max_immediate_size: l.max_push_constants_size,
                max_vertex_buffers: l.max_vertex_input_bindings.min(8),
                max_root_buffers: l.max_per_stage_descriptor_uniform_buffers.min(l.max_per_stage_descriptor_storage_buffers),
                max_color_attachments: l.max_color_attachments.min(8),
                max_buffer_size: 1 << 31,
                max_storage_buffer_binding_size: l.max_storage_buffer_range as u64,
                min_uniform_buffer_offset_alignment: 256,
                min_storage_buffer_offset_alignment: 256,
            };
            let fmt_ok = |f: vk::Format, need: vk::FormatFeatureFlags| raw.get_physical_device_format_properties(phys, f).optimal_tiling_features.contains(need);
            let caps = Caps {
                api_version: props.api_version,
                core13,
                family,
                ray_query,
                device_address: f12.buffer_device_address != 0,
                timestamps: l.timestamp_compute_and_graphics != 0 && families[family as usize].timestamp_valid_bits > 0,
                bc: core.texture_compression_bc != 0,
                d24s8: fmt_ok(vk::Format::D24_UNORM_S8_UINT, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
                // (MoltenVK offers it, but its SPIRV-Cross then zeroes with a loop over
                // `gl_WorkGroupSize`, which it never declares unless the shader reads that
                // built-in: the pipeline fails to compile. naga's own zeroing works there.)
                zero_init: zero_init != 0 && !portability,
                portability,
                swapchain: has_ext(&extensions, khr::swapchain::NAME),
                multi_draw_indirect: core.multi_draw_indirect != 0,
                draw_indirect_count: f12.draw_indirect_count != 0,
                precise_occlusion: core.occlusion_query_precise != 0,
                anisotropy: core.sampler_anisotropy != 0,
                // (wgpu-hal's choices, which decide its bounds checks)
                robust_buffer_access: core.robust_buffer_access != 0,
                robust_image_access: if robustness2 { rob2.robust_image_access2 != 0 } else { robust_image_access_13 != 0 },
                robustness2,
                robust_buffer_access2: robustness2 && rob2.robust_buffer_access2 != 0,
                robust_image_access2: robustness2 && rob2.robust_image_access2 != 0,
                image_robustness_ext,
                image_robustness_supported: robust_image_access_13 != 0,
                core,
                f11_draw_parameters: f11.shader_draw_parameters != 0,
                f11_storage16: f11.storage_buffer16_bit_access != 0,
                f12_int8: f12.shader_int8 != 0,
                f12_float16: f12.shader_float16 != 0,
                f12_ubo_std_layout: f12.uniform_buffer_standard_layout != 0,
                update_unused_while_pending: f12.descriptor_binding_update_unused_while_pending != 0,
                heap_storage_images,
                storage_image_non_uniform: f12.shader_storage_image_array_non_uniform_indexing != 0,
                heap_acceleration_structures,
                max_storage_buffer_range: l.max_storage_buffer_range as u64,
                limits,
                timestamp_period: l.timestamp_period,
                scratch_alignment: asp.min_acceleration_structure_scratch_offset_alignment.max(1) as u64,
            };
            Some(Adapter(Arc::new(AdapterInner {
                instance: inst.clone(),
                phys,
                props,
                caps,
            })))
        }
    }

    pub fn info(&self) -> t::AdapterInfo {
        let p = &self.0.props;
        t::AdapterInfo {
            name: p.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            backend: t::Backend::Vulkan,
            device_type: match p.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => t::DeviceType::DiscreteGpu,
                vk::PhysicalDeviceType::INTEGRATED_GPU => t::DeviceType::IntegratedGpu,
                vk::PhysicalDeviceType::VIRTUAL_GPU => t::DeviceType::VirtualGpu,
                vk::PhysicalDeviceType::CPU => t::DeviceType::Cpu,
                _ => t::DeviceType::Other,
            },
            vendor: p.vendor_id,
            device: p.device_id,
            driver: format!(
                "Vulkan {}.{}.{}, driver {:#x}",
                vk::api_version_major(p.api_version),
                vk::api_version_minor(p.api_version),
                vk::api_version_patch(p.api_version),
                p.driver_version
            ),
        }
    }

    pub fn limits(&self) -> t::Limits {
        self.0.caps.limits
    }

    /// The GPU's memory: its largest device-local heap (the budget is not asked for).
    pub fn memory_info(&self) -> Option<t::AdapterMemory> {
        // SAFETY: a plain query of a live physical device.
        let mem = unsafe { self.0.instance.raw.get_physical_device_memory_properties(self.0.phys) };
        let largest = mem.memory_heaps[..mem.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()?;
        Some(t::AdapterMemory {
            dedicated_bytes: largest,
            budget_bytes: None,
        })
    }

    /// Whether `format` renders and resolves with `samples` samples per pixel.
    pub fn supports_msaa(&self, format: t::TextureFormat, samples: u32) -> bool {
        if samples <= 1 {
            return true;
        }
        let l = &self.0.props.limits;
        let counts = if format.has_depth() {
            l.framebuffer_depth_sample_counts
        } else {
            l.framebuffer_color_sample_counts
        };
        counts.contains(vk::SampleCountFlags::from_raw(samples))
    }

    /// Textures a pipeline may read through root bindings (on OpenGL, through anything: texture
    /// units): (vertex stage, fragment stage, the pipeline together).
    pub fn texture_units(&self) -> (u32, u32, u32) {
        {
            let l = &self.0.props.limits;
            (
                l.max_per_stage_descriptor_sampled_images,
                l.max_per_stage_descriptor_sampled_images,
                l.max_descriptor_set_sampled_images,
            )
        }
    }

    /// Indexed draws with a base vertex (not below OpenGL 3.2 / ES 3.2).
    pub fn supports_base_vertex(&self) -> bool {
        true
    }

    pub fn supports_bc(&self) -> bool {
        self.0.caps.bc
    }

    pub fn supports_ray_query(&self) -> bool {
        self.0.caps.ray_query
    }

    /// Compute shaders: every bindless device has them.
    pub fn supports_compute(&self) -> bool {
        true
    }

    /// How many storage buffers the vertex and fragment stages may read as root bindings
    /// (any number through the heap).
    pub fn storage_buffers_per_stage(&self) -> (u32, u32) {
        (self.0.caps.limits.max_root_buffers, self.0.caps.limits.max_root_buffers)
    }

    /// Whether this GPU runs on the bound tier: the heap bound slot by slot at each draw,
    /// heap accesses indexed by immediates or constants only.
    pub fn is_bound_tier(&self) -> bool {
        false
    }
    /// GPU-driven draw counts (`Device::create_indirect_draws`).
    pub fn supports_indirect_count(&self) -> bool {
        self.0.caps.draw_indirect_count
    }

    pub fn supports_timestamps(&self) -> bool {
        self.0.caps.timestamps
    }

    /// The `VkPhysicalDevice`, for interop.
    pub fn raw(&self) -> vk::PhysicalDevice {
        self.0.phys
    }

    /// The `VkInstance` the adapter belongs to, for interop.
    pub fn raw_instance(&self) -> &ash::Instance {
        &self.0.instance.raw
    }

    /// The device extensions the adapter has.
    pub fn extensions(&self) -> Vec<std::ffi::CString> {
        // SAFETY: a query of the adapter's own physical device.
        let props = unsafe { self.0.instance.raw.enumerate_device_extension_properties(self.0.phys) }.unwrap_or_default();
        props.iter().filter_map(|p| p.extension_name_as_c_str().ok().map(CStr::to_owned)).collect()
    }

    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Result<(Device, Queue)> {
        let a = &self.0;
        let caps = &a.caps;
        let inst = &a.instance.raw;
        let textures = desc.bindless_textures.min(caps.limits.max_bindless_textures);
        let samplers = desc.bindless_samplers.min(caps.limits.max_bindless_samplers);
        let buffers = desc.bindless_buffers.min(caps.limits.max_bindless_buffers);
        if textures == 0 || samplers == 0 || buffers == 0 {
            return Err(Error::Unsupported("a bindless heap (descriptor indexing)".into()));
        }

        let mut exts: Vec<&CStr> = Vec::new();
        if caps.swapchain {
            exts.push(khr::swapchain::NAME);
        }
        if caps.portability {
            exts.push(khr::portability_subset::NAME);
        }
        if !caps.core13 {
            exts.push(khr::dynamic_rendering::NAME);
            exts.push(khr::synchronization2::NAME);
            if caps.image_robustness_ext {
                exts.push(ext::image_robustness::NAME);
            }
        }
        if caps.robustness2 {
            exts.push(ext::robustness2::NAME);
        }
        if caps.ray_query {
            exts.push(khr::acceleration_structure::NAME);
            exts.push(khr::ray_query::NAME);
            exts.push(khr::deferred_host_operations::NAME);
        }
        // (VK_EXT_device_fault where the driver has it: a lost device's report, `fault`)
        let mut fault = vk::PhysicalDeviceFaultFeaturesEXT::default();
        let fault_on = self.extensions().iter().any(|e| e.as_c_str() == ext::device_fault::NAME) && {
            let mut q = vk::PhysicalDeviceFeatures2::default().push_next(&mut fault);
            // SAFETY: a plain query into a local chain.
            unsafe { inst.get_physical_device_features2(a.phys, &mut q) };
            fault.device_fault == vk::TRUE
        };
        let mut fault = vk::PhysicalDeviceFaultFeaturesEXT::default().device_fault(fault_on);
        if fault_on {
            exts.push(ext::device_fault::NAME);
        }
        // (what the program asks for besides: DLSS's extensions)
        if !desc.extensions.is_empty() {
            let available = unsafe { inst.enumerate_device_extension_properties(a.phys) }.map_err(|e| vk_err("device extensions", e))?;
            for e in desc.extensions {
                if !available.iter().any(|a| a.extension_name_as_c_str().ok() == Some(*e)) {
                    return Err(Error::Unsupported(format!("the device extension {}", e.to_string_lossy())));
                }
                if !exts.contains(e) {
                    exts.push(e);
                }
            }
        }
        let ext_ptrs: Vec<*const std::ffi::c_char> = exts.iter().map(|e| e.as_ptr()).collect();

        // the core features mX uses, where the device has them
        let s = &caps.core;
        let core = vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(caps.robust_buffer_access)
            .full_draw_index_uint32(s.full_draw_index_uint32 != 0)
            .image_cube_array(s.image_cube_array != 0)
            .independent_blend(s.independent_blend != 0)
            .sample_rate_shading(s.sample_rate_shading != 0)
            .dual_src_blend(s.dual_src_blend != 0)
            .multi_draw_indirect(s.multi_draw_indirect != 0)
            .draw_indirect_first_instance(s.draw_indirect_first_instance != 0)
            .depth_clamp(s.depth_clamp != 0)
            .depth_bias_clamp(s.depth_bias_clamp != 0)
            .fill_mode_non_solid(s.fill_mode_non_solid != 0)
            .sampler_anisotropy(s.sampler_anisotropy != 0)
            .texture_compression_bc(s.texture_compression_bc != 0)
            .occlusion_query_precise(s.occlusion_query_precise != 0)
            .vertex_pipeline_stores_and_atomics(s.vertex_pipeline_stores_and_atomics != 0)
            .fragment_stores_and_atomics(s.fragment_stores_and_atomics != 0)
            .shader_storage_image_extended_formats(s.shader_storage_image_extended_formats != 0)
            .shader_storage_image_write_without_format(s.shader_storage_image_write_without_format != 0)
            .shader_clip_distance(s.shader_clip_distance != 0)
            .shader_cull_distance(s.shader_cull_distance != 0)
            .shader_float64(s.shader_float64 != 0)
            .shader_int64(s.shader_int64 != 0)
            .shader_int16(s.shader_int16 != 0);
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(caps.f11_draw_parameters)
            .storage_buffer16_bit_access(caps.f11_storage16);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(caps.update_unused_while_pending)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .descriptor_binding_storage_image_update_after_bind(caps.heap_storage_images)
            .shader_storage_image_array_non_uniform_indexing(caps.storage_image_non_uniform)
            .timeline_semaphore(true)
            .draw_indirect_count(caps.draw_indirect_count)
            .shader_int8(caps.f12_int8)
            .shader_float16(caps.f12_float16)
            .uniform_buffer_standard_layout(caps.f12_ubo_std_layout)
            .buffer_device_address(caps.device_address);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .robust_image_access(caps.robust_image_access && caps.image_robustness_supported)
            .shader_zero_initialize_workgroup_memory(caps.zero_init);
        let mut dynr = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut imgrob = vk::PhysicalDeviceImageRobustnessFeatures::default().robust_image_access(caps.robust_image_access && caps.image_robustness_supported);
        let mut rob2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default()
            .robust_buffer_access2(caps.robust_buffer_access2)
            .robust_image_access2(caps.robust_image_access2);
        let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true)
            .descriptor_binding_acceleration_structure_update_after_bind(caps.heap_acceleration_structures);
        let mut rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        // (MoltenVK: enable what the portability subset offers)
        let mut port = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default();
        if caps.portability {
            let mut q = vk::PhysicalDeviceFeatures2::default().push_next(&mut port);
            // SAFETY: a plain query into a local chain.
            unsafe { inst.get_physical_device_features2(a.phys, &mut q) };
        }

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(caps.family).queue_priorities(&priorities)];
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&ext_ptrs)
            .enabled_features(&core)
            .push_next(&mut f11)
            .push_next(&mut f12);
        if caps.core13 {
            info = info.push_next(&mut f13);
        } else {
            info = info.push_next(&mut dynr).push_next(&mut sync2);
            if caps.image_robustness_ext {
                info = info.push_next(&mut imgrob);
            }
        }
        if caps.robustness2 {
            info = info.push_next(&mut rob2);
        }
        if caps.ray_query {
            info = info.push_next(&mut accel).push_next(&mut rq);
        }
        if caps.portability {
            info = info.push_next(&mut port);
        }
        if fault_on {
            info = info.push_next(&mut fault);
        }
        // SAFETY: the create info and its chain point at locals that outlive the call.
        let raw = unsafe { inst.create_device(a.phys, &info, None) }.map_err(|e| vk_err("device", e))?;
        if fault_on {
            crate::fault::register(ext::device_fault::Device::new(inst, &raw));
        }
        // SAFETY: family / index 0 were created above.
        let queue = unsafe { raw.get_device_queue(caps.family, 0) };

        let fns = Fns {
            swapchain: caps.swapchain.then(|| khr::swapchain::Device::new(inst, &raw)),
            sync2: (!caps.core13).then(|| khr::synchronization2::Device::new(inst, &raw)),
            dynamic_rendering: (!caps.core13).then(|| khr::dynamic_rendering::Device::new(inst, &raw)),
            accel: caps.ray_query.then(|| khr::acceleration_structure::Device::new(inst, &raw)),
            debug: a.instance.has(ext::debug_utils::NAME).then(|| ext::debug_utils::Device::new(inst, &raw)),
        };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: inst.clone(),
            device: raw.clone(),
            physical_device: a.phys,
            debug_settings: Default::default(),
            buffer_device_address: caps.ray_query,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| crate::alloc_err("allocator", e))?;
        let mut sem_type = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        // SAFETY: a plain creation on the new device.
        let timeline = unsafe { raw.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut sem_type), None) }.map_err(|e| vk_err("timeline semaphore", e))?;
        let heap = Heap::new(
            &raw,
            textures,
            samplers,
            buffers,
            caps.update_unused_while_pending,
            caps.portability,
            caps.heap_storage_images,
            caps.heap_acceleration_structures,
        )?;
        let (textures, samplers, buffers) = heap.capacity();
        let vendor = a.props.vendor_id;
        let shared = Arc::new(Shared {
            instance: a.instance.clone(),
            phys: a.phys,
            raw,
            queue: parking_lot::ReentrantMutex::new(queue),
            caps: caps.clone(),
            fns,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            timeline,
            heap: Mutex::new(heap),
            last_submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            in_flight: Mutex::new(VecDeque::new()),
            pending: Mutex::new(None),
            ring: Mutex::new(Ring::new()),
            pending_waits: Mutex::new(Vec::new()),
            fresh: Mutex::new(Vec::new()),
            garbage: Mutex::new(Vec::new()),
            free_cmd_pools: Mutex::new(Vec::new()),
            free_desc_pools: Mutex::new(Vec::new()),
            set_layouts: Mutex::new(HashMap::new()),
            info: self.info(),
            limits: t::Limits {
                max_bindless_textures: textures,
                max_bindless_samplers: samplers,
                max_bindless_buffers: buffers,
                ..caps.limits
            }
            .capped(desc.limits.as_ref()),
            spirv: mx_shader::spirv::Options {
                lang_version: if caps.api_version >= vk::API_VERSION_1_3 { (1, 6) } else { (1, 5) },
                ray_query: caps.ray_query,
                native_zero_init: caps.zero_init,
                // (as wgpu-hal: not on Qualcomm)
                label_varyings: vendor != 0x5143,
                debug: false,
                robust_buffer_access2: caps.robust_buffer_access2,
                robust_image_access: caps.robust_image_access,
            },
        });
        if let Some(l) = desc.label {
            shared.name(queue, l);
        }
        let owner = Arc::new(Owner(shared.clone()));
        Ok((
            Device {
                shared: shared.clone(),
                _owner: owner.clone(),
            },
            Queue { shared, _owner: owner },
        ))
    }
}

#[derive(Clone, Debug)]
pub struct DeviceDescriptor<'a> {
    pub label: Option<&'a str>,
    /// Texture views the heap holds at once.
    pub bindless_textures: u32,
    /// Samplers the heap holds at once.
    pub bindless_samplers: u32,
    /// Buffer views the heap holds at once.
    pub bindless_buffers: u32,
    /// Device extensions to turn on besides mX's own (DLSS's); one the device lacks fails
    /// `request_device` with `Error::Unsupported`.
    pub extensions: &'a [&'static CStr],
    /// Limits the device keeps to, where lower than the adapter's (a weaker chip's, to
    /// test on); None: the adapter's.
    pub limits: Option<t::Limits>,
}

impl Default for DeviceDescriptor<'_> {
    fn default() -> Self {
        DeviceDescriptor {
            limits: None,
            label: None,
            bindless_textures: 1 << 16,
            bindless_samplers: 2048,
            bindless_buffers: 1 << 15,
            extensions: &[],
        }
    }
}

/// A position in the queue's order of submissions (the first is 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionIndex(pub u64);

// ---------------------------------------------------------------------------------------------
// The shared device

/// Extension function tables (only where the device is older than Vulkan 1.3, or optional).
pub(crate) struct Fns {
    pub swapchain: Option<khr::swapchain::Device>,
    pub sync2: Option<khr::synchronization2::Device>,
    pub dynamic_rendering: Option<khr::dynamic_rendering::Device>,
    pub accel: Option<khr::acceleration_structure::Device>,
    pub debug: Option<ext::debug_utils::Device>,
}

/// Something kept alive until the GPU is done with the submission that uses it.
pub(crate) type Keep = Box<dyn Any + Send + Sync>;

/// A Vulkan object to destroy once the GPU no longer uses it.
pub(crate) enum Garbage {
    Buffer(vk::Buffer, Allocation),
    Image(vk::Image, Allocation),
    View(vk::ImageView),
    Sampler(vk::Sampler),
    QueryPool(vk::QueryPool),
    Pipeline(vk::Pipeline, vk::PipelineLayout),
    /// A structure and the buffer it lives in.
    Accel(vk::AccelerationStructureKHR, vk::Buffer, Allocation),
}

/// What a command buffer holds until the GPU has run it.
#[derive(Default)]
pub(crate) struct Recording {
    pub cmd_pools: Vec<vk::CommandPool>,
    pub desc_pools: Vec<vk::DescriptorPool>,
    pub keep: Vec<Keep>,
    pub temp: Vec<Garbage>,
}

impl Recording {
    pub fn merge(&mut self, other: Recording) {
        self.cmd_pools.extend(other.cmd_pools);
        self.desc_pools.extend(other.desc_pools);
        self.keep.extend(other.keep);
        self.temp.extend(other.temp);
    }
}

/// Copies queued by `Queue::write_*` and layout changes of new images, run before the next
/// submission's commands.
pub(crate) struct Pending {
    pub cmd: vk::CommandBuffer,
    pub rec: Recording,
}

/// Staging memory for queue writes: 4 MiB host-visible blocks written one after the other
/// and reused once the submission that read them is done (a buffer and an allocation per
/// write cost several driver calls each, every frame).
pub(crate) struct Ring {
    free: Vec<(vk::Buffer, Allocation)>,
    /// The block being filled, and how far.
    current: Option<((vk::Buffer, Allocation), u64)>,
    /// Blocks the pending writes read.
    used: Vec<(vk::Buffer, Allocation)>,
    in_flight: VecDeque<(u64, Vec<(vk::Buffer, Allocation)>)>,
}

impl Ring {
    pub const BLOCK: u64 = 4 << 20;
    /// Blocks kept for reuse; more are released.
    const KEEP: usize = 8;

    fn new() -> Ring {
        Ring {
            free: Vec::new(),
            current: None,
            used: Vec::new(),
            in_flight: VecDeque::new(),
        }
    }

    fn retire(&mut self, index: u64) {
        if let Some((b, _)) = self.current.take() {
            self.used.push(b);
        }
        if !self.used.is_empty() {
            self.in_flight.push_back((index, std::mem::take(&mut self.used)));
        }
    }

    /// Blocks of submissions up to `completed` are free again; the ones over `KEEP` are
    /// returned for destruction.
    fn reclaim(&mut self, completed: u64) -> Vec<(vk::Buffer, Allocation)> {
        let mut extra = Vec::new();
        while self.in_flight.front().is_some_and(|(i, _)| *i <= completed) {
            for b in self.in_flight.pop_front().unwrap().1 {
                if self.free.len() < Self::KEEP {
                    self.free.push(b);
                } else {
                    extra.push(b);
                }
            }
        }
        extra
    }

    /// Every block, for the device's destruction.
    fn drain(&mut self) -> Vec<(vk::Buffer, Allocation)> {
        let mut all: Vec<_> = self.free.drain(..).chain(self.used.drain(..)).collect();
        all.extend(self.current.take().map(|(b, _)| b));
        all.extend(self.in_flight.drain(..).flat_map(|(_, v)| v));
        all
    }
}

/// Descriptor set layout of one root group, by its bindings.
pub(crate) type SetLayoutKey = Vec<(u32, vk::DescriptorType, vk::ShaderStageFlags)>;

pub(crate) struct Shared {
    pub instance: Arc<InstanceShared>,
    pub phys: vk::PhysicalDevice,
    pub raw: ash::Device,
    /// (reentrant: a vendor's swapchain submits to the queue from inside the present that holds
    /// it, through `Queue::with_raw`)
    pub queue: parking_lot::ReentrantMutex<vk::Queue>,
    pub caps: Caps,
    pub fns: Fns,
    pub allocator: ManuallyDrop<Mutex<Allocator>>,
    pub timeline: vk::Semaphore,
    pub heap: Mutex<Heap>,
    pub last_submitted: AtomicU64,
    pub completed: AtomicU64,
    in_flight: Mutex<VecDeque<(u64, Recording)>>,
    pending: Mutex<Option<Pending>>,
    /// Staging memory of the queue's writes (used under the `pending` lock).
    ring: Mutex<Ring>,
    /// Binary semaphores (swapchain acquires) the next submission waits for.
    pub pending_waits: Mutex<Vec<vk::Semaphore>>,
    /// Textures made since the last submission (see `zero_new_textures`).
    pub fresh: Mutex<Vec<std::sync::Weak<crate::resource::TextureInner>>>,
    garbage: Mutex<Vec<(u64, Garbage)>>,
    free_cmd_pools: Mutex<Vec<vk::CommandPool>>,
    free_desc_pools: Mutex<Vec<vk::DescriptorPool>>,
    pub set_layouts: Mutex<HashMap<SetLayoutKey, vk::DescriptorSetLayout>>,
    pub info: t::AdapterInfo,
    pub limits: t::Limits,
    pub spirv: mx_shader::spirv::Options,
}

// A completed pool can retain its largest command/descriptor allocation in the driver.
const IDLE_POOLS: usize = 16;

pub(crate) const ALL_READ_WRITE: vk::AccessFlags2 = vk::AccessFlags2::from_raw(vk::AccessFlags2::MEMORY_READ.as_raw() | vk::AccessFlags2::MEMORY_WRITE.as_raw());

impl Shared {
    /// The submission that will be the next one: anything released now may be read by
    /// command buffers recorded before, which are submitted at the earliest then.
    pub fn release_after(&self) -> u64 {
        self.last_submitted.load(Ordering::Acquire) + 1
    }

    /// Destroy `g` once the GPU is past everything submitted so far (and the next one).
    pub fn retire(&self, g: Garbage) {
        self.garbage.lock().push((self.release_after(), g));
    }

    pub fn name<H: vk::Handle>(&self, handle: H, label: &str) {
        if let Some(d) = &self.fns.debug
            && let Ok(name) = CString::new(label)
        {
            // SAFETY: a live handle of this device.
            let _ = unsafe { d.set_debug_utils_object_name(&vk::DebugUtilsObjectNameInfoEXT::default().object_handle(handle).object_name(&name)) };
        }
    }

    pub fn take_cmd_pool(&self) -> vk::CommandPool {
        if let Some(p) = self.free_cmd_pools.lock().pop() {
            return p;
        }
        let info = vk::CommandPoolCreateInfo::default().queue_family_index(self.caps.family).flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: a plain creation.
        unsafe { self.raw.create_command_pool(&info, None) }.expect("out of memory for a command pool")
    }

    pub fn take_desc_pool(&self) -> Option<vk::DescriptorPool> {
        self.free_desc_pools.lock().pop()
    }

    /// A new primary command buffer from a pool of its own, begun, with the barrier that
    /// orders it after everything submitted before.
    pub fn begin_cmd(&self) -> (vk::CommandPool, vk::CommandBuffer) {
        let pool = self.take_cmd_pool();
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool is ours alone until the command buffer completes.
        let cmd = unsafe { self.raw.allocate_command_buffers(&info) }.expect("out of memory for a command buffer")[0];
        // SAFETY: as above.
        unsafe {
            self.raw
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
        }
        .expect("vkBeginCommandBuffer");
        self.full_barrier(cmd);
        (pool, cmd)
    }

    /// End a command buffer: its writes become visible to later commands and to the host.
    pub fn end_cmd(&self, cmd: vk::CommandBuffer) {
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST)
            .dst_access_mask(ALL_READ_WRITE | vk::AccessFlags2::HOST_READ)];
        self.barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
        // SAFETY: recording, ours.
        unsafe { self.raw.end_command_buffer(cmd) }.expect("vkEndCommandBuffer");
    }

    pub fn barrier2(&self, cmd: vk::CommandBuffer, dep: &vk::DependencyInfo<'_>) {
        // SAFETY: a recording command buffer; the dependency's arrays outlive the call.
        unsafe {
            match &self.fns.sync2 {
                Some(f) => f.cmd_pipeline_barrier2(cmd, dep),
                None => self.raw.cmd_pipeline_barrier2(cmd, dep),
            }
        }
    }

    /// Everything before waits for everything before to be written: mX's one barrier
    /// between passes and copies.
    pub fn full_barrier(&self, cmd: vk::CommandBuffer) {
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(ALL_READ_WRITE)];
        self.barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
    }

    /// Copies after wait for copies before, and see what they wrote.
    pub fn transfer_barrier(&self, cmd: vk::CommandBuffer) {
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE)];
        self.barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
    }

    /// A layout transition of a whole image.
    pub fn transition(&self, cmd: vk::CommandBuffer, image: vk::Image, aspect: vk::ImageAspectFlags, from: vk::ImageLayout, to: vk::ImageLayout) {
        let b = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(ALL_READ_WRITE)
            .old_layout(from)
            .new_layout(to)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            })];
        self.barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b));
    }

    /// Run `f` on the pending-writes command buffer (begun on first use), after a barrier.
    pub fn with_pending<R>(&self, f: impl FnOnce(vk::CommandBuffer, &mut Recording) -> R) -> R {
        let mut p = self.pending.lock();
        let p = match p.as_mut() {
            Some(p) => {
                // (the writes before are copies: the next copy waits for them)
                self.transfer_barrier(p.cmd);
                p
            }
            None => {
                let (pool, cmd) = self.begin_cmd();
                self.name(cmd, "mX queue writes");
                p.insert(Pending {
                    cmd,
                    rec: Recording {
                        cmd_pools: vec![pool],
                        ..Default::default()
                    },
                })
            }
        };
        f(p.cmd, &mut p.rec)
    }

    /// Copy `data` into staging memory the next submission reads: a buffer and the offset
    /// of the bytes in it (256-aligned). Call it inside `with_pending`, so the bytes go out
    /// with the writes that read them; big writes get a buffer of their own (in `rec`).
    pub fn stage(&self, rec: &mut Recording, data: &[u8]) -> Result<(vk::Buffer, u64)> {
        if data.len() as u64 > Ring::BLOCK / 4 {
            let (b, a) = self.staging_buffer(data, vk::BufferUsageFlags::TRANSFER_SRC)?;
            rec.temp.push(Garbage::Buffer(b, a));
            return Ok((b, 0));
        }
        let mut ring = self.ring.lock();
        let len = (data.len() as u64).max(4);
        let fits = ring.current.as_ref().is_some_and(|(_, at)| at.next_multiple_of(256) + len <= Ring::BLOCK);
        if !fits {
            if let Some((full, _)) = ring.current.take() {
                ring.used.push(full);
            }
            let block = match ring.free.pop() {
                Some(b) => b,
                None => self.create_raw_buffer(Ring::BLOCK, vk::BufferUsageFlags::TRANSFER_SRC, gpu_allocator::MemoryLocation::CpuToGpu, "mX staging")?,
            };
            ring.current = Some((block, 0));
        }
        let ((buf, alloc), at) = ring.current.as_mut().expect("a current block");
        let offset = at.next_multiple_of(256);
        let base = alloc.mapped_ptr().expect("staging memory is mapped").as_ptr().cast::<u8>();
        // SAFETY: `offset + data.len()` is inside the mapped block (checked above), whose
        // bytes no submitted work reads (blocks come back only once their submission is done).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), base.add(offset as usize), data.len()) };
        *at = offset + len;
        Ok((*buf, offset))
    }

    /// A host-visible buffer holding `data`, for a copy (freed when the copy is done).
    pub fn staging_buffer(&self, data: &[u8], usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, Allocation)> {
        let (buf, alloc) = self.create_raw_buffer(data.len().max(4) as u64, usage, gpu_allocator::MemoryLocation::CpuToGpu, "mX staging")?;
        let dst = alloc.mapped_ptr().expect("staging memory is mapped").as_ptr().cast::<u8>();
        // SAFETY: the allocation is at least `data.len()` bytes, mapped and not yet in use.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
        Ok((buf, alloc))
    }

    pub fn create_raw_buffer(&self, size: u64, usage: vk::BufferUsageFlags, location: gpu_allocator::MemoryLocation, name: &str) -> Result<(vk::Buffer, Allocation)> {
        let info = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: plain creation, binding and (on failure) destruction of a new buffer.
        unsafe {
            let buf = self.raw.create_buffer(&info, None).map_err(|e| vk_err(name, e))?;
            let req = self.raw.get_buffer_memory_requirements(buf);
            let alloc = self.allocator.lock().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name,
                requirements: req,
                location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            });
            let alloc = match alloc {
                Ok(a) => a,
                Err(e) => {
                    self.raw.destroy_buffer(buf, None);
                    return Err(crate::alloc_err(name, e));
                }
            };
            if let Err(e) = self.raw.bind_buffer_memory(buf, alloc.memory(), alloc.offset()) {
                self.raw.destroy_buffer(buf, None);
                let _ = self.allocator.lock().free(alloc);
                return Err(vk_err(name, e));
            }
            Ok((buf, alloc))
        }
    }

    pub fn buffer_address(&self, buffer: vk::Buffer) -> u64 {
        // SAFETY: a live buffer made with SHADER_DEVICE_ADDRESS usage.
        unsafe { self.raw.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) }
    }

    /// Submit `buffers` after the pending writes. `signal` (a binary semaphore) is signalled
    /// with the submission.
    pub fn submit(&self, buffers: Vec<CommandBuffer>, signal: Option<vk::Semaphore>) -> SubmissionIndex {
        let queue = self.queue.lock();
        let index = self.last_submitted.load(Ordering::Acquire) + 1;
        let mut rec = Recording::default();
        // (batches: one queue submission each; see `CommandEncoder::split`)
        let mut batches = vec![Vec::new()];
        // (new textures to GENERAL and zeroed before the queue's writes land in them)
        if let Some((cmd, r)) = self.zero_new_textures() {
            batches[0].push(vk::CommandBufferSubmitInfo::default().command_buffer(cmd));
            rec.merge(r);
        }
        let mut pending = self.pending.lock();
        if let Some(p) = pending.take() {
            self.end_cmd(p.cmd);
            batches[0].push(vk::CommandBufferSubmitInfo::default().command_buffer(p.cmd));
            rec.merge(p.rec);
            // (the ring's blocks the writes read wait for this submission)
            self.ring.lock().retire(index);
        }
        drop(pending);
        for mut b in buffers {
            for (i, &c) in b.raws.iter().enumerate() {
                if i > 0 {
                    batches.push(Vec::new());
                }
                batches.last_mut().unwrap().push(vk::CommandBufferSubmitInfo::default().command_buffer(c));
            }
            rec.merge(std::mem::take(&mut b.rec));
        }
        let waits: Vec<vk::SemaphoreSubmitInfo> = self
            .pending_waits
            .lock()
            .drain(..)
            .map(|s| vk::SemaphoreSubmitInfo::default().semaphore(s).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS))
            .collect();
        let mut signals = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(index)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        if let Some(s) = signal {
            signals.push(vk::SemaphoreSubmitInfo::default().semaphore(s).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS));
        }
        let n = batches.len();
        for (i, cmds) in batches.iter().enumerate() {
            let mut submit = vk::SubmitInfo2::default().command_buffer_infos(cmds);
            if i == 0 {
                submit = submit.wait_semaphore_infos(&waits);
            }
            if i + 1 == n {
                submit = submit.signal_semaphore_infos(&signals);
            }
            // SAFETY: the command buffers are ended and owned by this submission; the queue
            // is locked.
            let r = unsafe {
                match &self.fns.sync2 {
                    Some(f) => f.queue_submit2(*queue, &[submit], vk::Fence::null()),
                    None => self.raw.queue_submit2(*queue, &[submit], vk::Fence::null()),
                }
            };
            if let Err(e) = r {
                log::error!("mX: vkQueueSubmit2 failed: {e}");
            }
        }
        self.in_flight.lock().push_back((index, rec));
        self.last_submitted.store(index, Ordering::Release);
        SubmissionIndex(index)
    }

    /// Note finished submissions and free what they held. With `wait`, block until all
    /// submissions up to `wait` are done.
    pub fn poll(&self, wait: Option<u64>) -> Result<()> {
        let last = self.last_submitted.load(Ordering::Acquire);
        if let Some(target) = wait.map(|w| w.min(last))
            && target > self.completed.load(Ordering::Acquire)
        {
            let sems = [self.timeline];
            let values = [target];
            // SAFETY: waiting on our own timeline semaphore.
            unsafe { self.raw.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values), u64::MAX) }.map_err(|e| vk_err("waiting for the GPU", e))?;
        }
        // SAFETY: a plain query.
        let value = unsafe { self.raw.get_semaphore_counter_value(self.timeline) }.map_err(|e| vk_err("polling the GPU", e))?;
        self.completed.fetch_max(value, Ordering::AcqRel);
        let done: Vec<Recording> = {
            let mut in_flight = self.in_flight.lock();
            let mut done = Vec::new();
            while in_flight.front().is_some_and(|(i, _)| *i <= value) {
                done.push(in_flight.pop_front().unwrap().1);
            }
            done
        };
        for rec in done {
            self.recycle(rec);
        }
        self.heap.lock().reclaim(value);
        let extra = self.ring.lock().reclaim(value);
        for (b, a) in extra {
            self.destroy(Garbage::Buffer(b, a));
        }
        let dead: Vec<Garbage> = {
            let mut g = self.garbage.lock();
            let mut dead = Vec::new();
            let mut i = 0;
            while i < g.len() {
                if g[i].0 <= value {
                    dead.push(g.swap_remove(i).1);
                } else {
                    i += 1;
                }
            }
            dead
        };
        for g in dead {
            self.destroy(g);
        }
        Ok(())
    }

    /// Give a finished command buffer's pools back and free what it held.
    pub(crate) fn recycle(&self, rec: Recording) {
        // SAFETY: the GPU is done with everything allocated from these pools.
        unsafe {
            for p in rec.cmd_pools {
                let mut idle = self.free_cmd_pools.lock();
                if idle.len() < IDLE_POOLS && self.raw.reset_command_pool(p, vk::CommandPoolResetFlags::empty()).is_ok() {
                    idle.push(p);
                } else {
                    drop(idle);
                    self.raw.destroy_command_pool(p, None);
                }
            }
            for p in rec.desc_pools {
                let mut idle = self.free_desc_pools.lock();
                if idle.len() < IDLE_POOLS && self.raw.reset_descriptor_pool(p, vk::DescriptorPoolResetFlags::empty()).is_ok() {
                    idle.push(p);
                } else {
                    drop(idle);
                    self.raw.destroy_descriptor_pool(p, None);
                }
            }
        }
        drop(rec.keep);
        for g in rec.temp {
            self.destroy(g);
        }
    }

    pub fn destroy(&self, g: Garbage) {
        // SAFETY: the GPU no longer uses the object (deferred until its submission is done).
        unsafe {
            match g {
                Garbage::Buffer(b, a) => {
                    self.raw.destroy_buffer(b, None);
                    let _ = self.allocator.lock().free(a);
                }
                Garbage::Image(i, a) => {
                    self.raw.destroy_image(i, None);
                    let _ = self.allocator.lock().free(a);
                }
                Garbage::View(v) => self.raw.destroy_image_view(v, None),
                Garbage::Sampler(s) => self.raw.destroy_sampler(s, None),
                Garbage::QueryPool(q) => self.raw.destroy_query_pool(q, None),
                Garbage::Pipeline(p, l) => {
                    self.raw.destroy_pipeline(p, None);
                    self.raw.destroy_pipeline_layout(l, None);
                }
                Garbage::Accel(s, b, a) => {
                    if let Some(f) = &self.fns.accel {
                        f.destroy_acceleration_structure(s, None);
                    }
                    self.raw.destroy_buffer(b, None);
                    let _ = self.allocator.lock().free(a);
                }
            }
        }
    }

    pub fn wait_idle(&self) -> Result<()> {
        self.poll(Some(self.last_submitted.load(Ordering::Acquire)))
    }

    /// Drop the pending writes without running them (the device is going away).
    fn discard_pending(&self) {
        if let Some(p) = self.pending.lock().take() {
            self.recycle(p.rec);
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: every object made from the device holds an `Arc` to it: nothing else uses
        // it now. Wait for the GPU, then destroy what is left.
        unsafe {
            let _ = self.raw.device_wait_idle();
            self.discard_pending();
            let blocks = self.ring.lock().drain();
            for (b, a) in blocks {
                self.destroy(Garbage::Buffer(b, a));
            }
            let recs: Vec<Recording> = self.in_flight.lock().drain(..).map(|(_, r)| r).collect();
            for r in recs {
                self.recycle(r);
            }
            let garbage: Vec<Garbage> = self.garbage.lock().drain(..).map(|(_, g)| g).collect();
            for g in garbage {
                self.destroy(g);
            }
            for p in self.free_cmd_pools.lock().drain(..) {
                self.raw.destroy_command_pool(p, None);
            }
            for p in self.free_desc_pools.lock().drain(..) {
                self.raw.destroy_descriptor_pool(p, None);
            }
            for (_, l) in self.set_layouts.lock().drain() {
                self.raw.destroy_descriptor_set_layout(l, None);
            }
            self.heap.lock().destroy(&self.raw);
            self.raw.destroy_semaphore(self.timeline, None);
            ManuallyDrop::drop(&mut self.allocator);
            self.raw.destroy_device(None);
        }
    }
}

/// Held by every `Device` and `Queue` clone: when the last goes, the device waits for the GPU
/// and lets go of what in-flight work kept alive (which in turn hold the device).
pub(crate) struct Owner(Arc<Shared>);

impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.0.wait_idle();
        self.0.discard_pending();
    }
}

// ---------------------------------------------------------------------------------------------
// Device and queue

/// The GPU. Cheap to clone; every clone is the same device.
#[derive(Clone)]
pub struct Device {
    pub(crate) shared: Arc<Shared>,
    pub(crate) _owner: Arc<Owner>,
}

impl Device {
    pub fn info(&self) -> &t::AdapterInfo {
        &self.shared.info
    }

    pub fn limits(&self) -> t::Limits {
        self.shared.limits
    }

    /// Buffer views in the heap right now.
    pub fn bindless_buffers_in_use(&self) -> usize {
        self.shared.heap.lock().buffers_in_use()
    }

    /// Note finished work and free what it held, without waiting.
    pub fn poll(&self) -> Result<()> {
        self.shared.poll(None)
    }

    /// Wait until `index` (and everything before it) has finished on the GPU.
    pub fn wait(&self, index: SubmissionIndex) -> Result<()> {
        self.shared.poll(Some(index.0))
    }

    /// Whether `index` has finished on the GPU (as of the last `poll` or `wait`).
    pub fn is_done(&self, index: SubmissionIndex) -> bool {
        self.shared.completed.load(Ordering::Acquire) >= index.0
    }

    pub fn wait_idle(&self) -> Result<()> {
        self.shared.wait_idle()
    }

    /// BC texture compression (as the adapter's `supports_bc`).
    pub fn supports_bc(&self) -> bool {
        self.shared.caps.bc
    }

    /// Timestamp queries (as the adapter's `supports_timestamps`).
    pub fn supports_timestamps(&self) -> bool {
        self.shared.caps.timestamps
    }

    /// Whether the device is known lost (a call on it said so; Direct3D 12 tells it at any time).
    pub fn is_lost(&self) -> bool {
        false
    }

    /// Texture views in the heap right now.
    pub fn bindless_textures_in_use(&self) -> usize {
        self.shared.heap.lock().textures_in_use()
    }

    /// The `VkDevice` (with its function table), for interop (DLSS, FSR, OpenXR).
    pub fn raw(&self) -> &ash::Device {
        &self.shared.raw
    }

    pub fn raw_instance(&self) -> &ash::Instance {
        &self.shared.instance.raw
    }

    pub fn raw_physical_device(&self) -> vk::PhysicalDevice {
        self.shared.phys
    }

    pub fn queue_family_index(&self) -> u32 {
        self.shared.caps.family
    }

    pub(crate) fn queue(&self) -> Queue {
        Queue {
            shared: self.shared.clone(),
            _owner: self._owner.clone(),
        }
    }
}

/// Where command buffers go.
#[derive(Clone)]
pub struct Queue {
    pub(crate) shared: Arc<Shared>,
    pub(crate) _owner: Arc<Owner>,
}

impl Queue {
    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) -> SubmissionIndex {
        self.shared.submit(buffers.into_iter().collect(), None)
    }

    /// The `VkQueue`, for interop. Submissions to it must not overlap mX's own (mX locks it
    /// while submitting): see [`Queue::with_raw`].
    pub fn raw(&self) -> vk::Queue {
        *self.shared.queue.lock()
    }

    /// `f` with the `VkQueue`, mX's own submissions held off meanwhile: where a program
    /// submits work of its own (DLSS, FSR 3) between mX's. Images rest in `GENERAL` between
    /// mX's commands; the work leaves them there.
    pub fn with_raw<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        let q = self.shared.queue.lock();
        f(*q)
    }

    /// The index of the queue in its family.
    pub fn queue_index(&self) -> u32 {
        0
    }
}

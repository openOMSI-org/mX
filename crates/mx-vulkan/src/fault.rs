//! `VK_EXT_device_fault`: what the driver says about a lost device.
//!
//! VK_ERROR_DEVICE_LOST says nothing of where or why; drivers with this extension (recent
//! NVIDIA, AMD, Arm, Qualcomm ones) keep a description and the faulting addresses, read once
//! after the loss and written to the log.

use std::fmt::Write as _;

use ash::{ext, vk};

static DEVICES: parking_lot::Mutex<Vec<ext::device_fault::Device>> = parking_lot::Mutex::new(Vec::new());

/// Remember a device made with the extension.
pub(crate) fn register(f: ext::device_fault::Device) {
    DEVICES.lock().push(f);
}

/// The driver's report of the first lost device that has one.
pub fn fault_report() -> Option<String> {
    let devices = DEVICES.lock();
    devices.iter().find_map(report)
}

fn report(f: &ext::device_fault::Device) -> Option<String> {
    let get = f.fp().get_device_fault_info_ext;
    let mut counts = vk::DeviceFaultCountsEXT::default();
    // SAFETY: the counts query (no info pointer), as the extension asks first.
    let r = unsafe { get(f.device(), &mut counts, std::ptr::null_mut()) };
    if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
        return None;
    }
    // (no fault recorded: the device is not lost, or the driver knows nothing of it)
    let none_yet = counts.address_info_count == 0 && counts.vendor_info_count == 0;
    let mut addresses = vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
    let mut vendors = vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
    // (the vendor's binary dump is left out: it is for the vendor's own tools)
    counts.vendor_binary_size = 0;
    let mut info = vk::DeviceFaultInfoEXT {
        p_address_infos: if addresses.is_empty() { std::ptr::null_mut() } else { addresses.as_mut_ptr() },
        p_vendor_infos: if vendors.is_empty() { std::ptr::null_mut() } else { vendors.as_mut_ptr() },
        ..Default::default()
    };
    // SAFETY: arrays of the counts just read, alive over the call.
    let r = unsafe { get(f.device(), &mut counts, &mut info) };
    if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
        return None;
    }
    let mut out = String::new();
    let desc = info.description_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
    if none_yet && desc.trim().is_empty() {
        return None;
    }
    let _ = write!(out, "the device was lost: {}", if desc.is_empty() { "no description" } else { desc.as_str() });
    for a in addresses.iter().take(counts.address_info_count as usize) {
        let _ = write!(out, "\n  {:?} at {:#x} (+/- {:#x})", a.address_type, a.reported_address, a.address_precision);
    }
    for v in vendors.iter().take(counts.vendor_info_count as usize) {
        let d = v.description_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
        let _ = write!(out, "\n  vendor fault {:#x} / {:#x}: {d}", v.vendor_fault_code, v.vendor_fault_data);
    }
    Some(out)
}

/// Log the report once per run, at the first sight of the loss.
pub(crate) fn log_once(at: &str) {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if DONE.swap(true, std::sync::atomic::Ordering::AcqRel) {
        return;
    }
    match fault_report() {
        Some(r) => log::error!("mX: {at}: {r}"),
        None => log::error!("mX: {at}: the device was lost (the driver gives no fault report: no VK_EXT_device_fault)"),
    }
}

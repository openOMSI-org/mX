//! Device Removed Extended Data: what the GPU was doing when Direct3D 12 removed the device.
//!
//! A removed device answers every call with DXGI_ERROR_DEVICE_REMOVED and nothing more; the
//! log then said only "device removed" (openOMSI #4157, an RTX 3060 at loading). DRED keeps,
//! for every command list on the GPU, the operations it got through (auto-breadcrumbs, with
//! the PIX event names around them), and on a page fault the address and the resources that
//! live there or were just freed. It is turned on before the first device is made and costs
//! a few per cent at most (Microsoft's figure for breadcrumbs); `MX_DX12_DRED=0` leaves it off.

use std::fmt::Write as _;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// Turn DRED's breadcrumbs, their contexts and the page-fault report on, before any device
/// is made (the settings are read when a device is created).
pub(crate) fn enable() {
    if std::env::var("MX_DX12_DRED").is_ok_and(|v| v == "0") {
        return;
    }
    let mut settings: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
    // SAFETY: a plain query of the runtime's DRED settings (Windows 10 1903 and later; it
    // fails and is skipped before).
    if unsafe { D3D12GetDebugInterface(&mut settings) }.is_err() {
        log::info!("mX: no DRED on this system (Windows 10 1903 or later has it)");
        return;
    }
    let Some(settings) = settings else { return };
    // SAFETY: before any device is created, as required.
    unsafe {
        settings.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        settings.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    }
    if let Ok(s1) = settings.cast::<ID3D12DeviceRemovedExtendedDataSettings1>() {
        // SAFETY: as above.
        unsafe { s1.SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON) };
    }
}

/// The report of a removed device: the reason, the command lists that had not finished and
/// where they stood, and the page fault. Empty when the device is not removed.
pub(crate) fn report(device: &ID3D12Device) -> String {
    // SAFETY: a plain getter.
    let reason = unsafe { device.GetDeviceRemovedReason() };
    let mut out = String::new();
    match reason {
        Ok(()) => return out,
        Err(e) => {
            let _ = write!(out, "the device was removed: {} ({:#010x})", reason_name(e.code().0), e.code().0 as u32);
        }
    }
    let Ok(dred) = device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
        out.push_str("; no DRED report (DRED is off or not supported)");
        return out;
    };
    // SAFETY: the runtime's own lists, valid while the device lives; read only.
    unsafe {
        match dred.GetAutoBreadcrumbsOutput1() {
            Ok(b) => breadcrumbs(&mut out, b.pHeadAutoBreadcrumbNode),
            Err(e) => {
                let _ = write!(out, "\n  no breadcrumbs: {e}");
            }
        }
        match dred.GetPageFaultAllocationOutput1() {
            Ok(p) if p.PageFaultVA != 0 => {
                let _ = write!(out, "\n  page fault at GPU address {:#x}", p.PageFaultVA);
                allocations(&mut out, "    living there", p.pHeadExistingAllocationNode);
                allocations(&mut out, "    freed there just before", p.pHeadRecentFreedAllocationNode);
            }
            Ok(_) => out.push_str("\n  no page fault"),
            Err(e) => {
                let _ = write!(out, "\n  no page-fault report: {e}");
            }
        }
    }
    out
}

/// The devices made (a device is free-threaded: Direct3D 12 lets any thread call it).
struct Registered(ID3D12Device);
// SAFETY: ID3D12Device is free-threaded.
unsafe impl Send for Registered {}
// SAFETY: as above.
unsafe impl Sync for Registered {}
static DEVICES: parking_lot::Mutex<Vec<Registered>> = parking_lot::Mutex::new(Vec::new());

/// Remember `device`, for the report of a removal found where no device is at hand (an
/// error code, a panic).
pub(crate) fn register(device: &ID3D12Device) {
    let mut d = DEVICES.lock();
    if !d.iter().any(|r| r.0.as_raw() == device.as_raw()) {
        d.push(Registered(device.clone()));
    }
}

/// The report of the first removed device made in this run, if one was removed.
pub fn removed_report() -> Option<String> {
    let devices = DEVICES.lock();
    devices.iter().map(|d| report(&d.0)).find(|r| !r.is_empty())
}

/// Log the report once per run (every call on a removed device fails, and the first sight
/// of it is the one worth keeping).
pub(crate) fn log_once(at: &str) {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if DONE.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    if let Some(r) = removed_report()
        && !DONE.swap(true, std::sync::atomic::Ordering::AcqRel)
    {
        log::error!("mX: {at}: {r}");
    }
}

/// A call on the device failed: with the device removed, the report goes to the log before
/// the panic (a removed device fails the next allocation, list reset or resize, which panicked
/// with only "device removed").
pub(crate) fn fail(what: &str, e: windows::core::Error) -> ! {
    log_once(what);
    panic!("mX: {what}: {e}")
}

fn reason_name(code: i32) -> &'static str {
    match code as u32 {
        0x887A0005 => "DXGI_ERROR_DEVICE_REMOVED",
        0x887A0006 => "DXGI_ERROR_DEVICE_HUNG (a command took too long or the GPU faulted)",
        0x887A0007 => "DXGI_ERROR_DEVICE_RESET (bad commands)",
        0x887A0020 => "DXGI_ERROR_DRIVER_INTERNAL_ERROR",
        0x887A0001 => "DXGI_ERROR_INVALID_CALL",
        0x8007000E => "E_OUTOFMEMORY",
        _ => "an error",
    }
}

/// How many operations of an unfinished list are shown before and after the last done.
const BEFORE: u32 = 4;
const AFTER: u32 = 6;

unsafe fn breadcrumbs(out: &mut String, mut node: *const D3D12_AUTO_BREADCRUMB_NODE1) {
    let mut lists = 0;
    let mut open = 0;
    while !node.is_null() && lists < 256 {
        // SAFETY: a node of the runtime's list (see `report`).
        let n = unsafe { &*node };
        lists += 1;
        let done = if n.pLastBreadcrumbValue.is_null() { 0 } else { unsafe { *n.pLastBreadcrumbValue } };
        if done < n.BreadcrumbCount && !n.pCommandHistory.is_null() {
            open += 1;
            if open <= 4 {
                let _ = write!(
                    out,
                    "\n  command list {} on queue {}: {done} of {} operations done",
                    unsafe { name(n.pCommandListDebugNameW) },
                    unsafe { name(n.pCommandQueueDebugNameW) },
                    n.BreadcrumbCount
                );
                let from = done.saturating_sub(BEFORE);
                let to = (done + AFTER).min(n.BreadcrumbCount);
                for i in from..to {
                    // SAFETY: an index below the history's count.
                    let op = unsafe { *n.pCommandHistory.add(i as usize) };
                    let mark = if i == done { "  <- not finished" } else { "" };
                    let _ = write!(out, "\n    [{i}] {}{}{mark}", op_name(op), unsafe { context(n, i) });
                }
                // (the pass it was in: the last event begun before it and not ended)
                if let Some(pass) = unsafe { open_event(n, done) } {
                    let _ = write!(out, "\n    in the pass \"{pass}\"");
                }
            }
        }
        node = n.pNext;
    }
    if open == 0 {
        let _ = write!(out, "\n  breadcrumbs: every one of the {lists} command lists had finished (a hang outside a list, or the present)");
    } else if open > 4 {
        let _ = write!(out, "\n  ... and {} more unfinished command lists", open - 4);
    }
}

/// The breadcrumb context (PIX event or marker text) at operation `i`, if any.
unsafe fn context(n: &D3D12_AUTO_BREADCRUMB_NODE1, i: u32) -> String {
    if n.pBreadcrumbContexts.is_null() {
        return String::new();
    }
    for k in 0..n.BreadcrumbContextsCount {
        // SAFETY: an index below the contexts' count.
        let c = unsafe { *n.pBreadcrumbContexts.add(k as usize) };
        if c.BreadcrumbIndex == i {
            return format!(" \"{}\"", unsafe { name(c.pContextString) });
        }
    }
    String::new()
}

/// The innermost event begun before operation `done` and not ended by then.
unsafe fn open_event(n: &D3D12_AUTO_BREADCRUMB_NODE1, done: u32) -> Option<String> {
    let mut stack: Vec<u32> = Vec::new();
    for i in 0..done.min(n.BreadcrumbCount) {
        // SAFETY: an index below the history's count.
        let op = unsafe { *n.pCommandHistory.add(i as usize) };
        if op == D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT {
            stack.push(i);
        } else if op == D3D12_AUTO_BREADCRUMB_OP_ENDEVENT {
            stack.pop();
        }
    }
    let i = *stack.last()?;
    let c = unsafe { context(n, i) };
    (!c.is_empty()).then(|| c.trim().trim_matches('"').to_string())
}

unsafe fn allocations(out: &mut String, what: &str, mut node: *const D3D12_DRED_ALLOCATION_NODE1) {
    let mut k = 0;
    while !node.is_null() && k < 16 {
        // SAFETY: a node of the runtime's list (see `report`).
        let n = unsafe { &*node };
        let _ = write!(out, "\n{what}: {} ({})", unsafe { name(n.ObjectNameW) }, allocation_type(n.AllocationType));
        node = n.pNext;
        k += 1;
    }
    if k == 0 {
        let _ = write!(out, "\n{what}: nothing");
    }
}

unsafe fn name(p: windows::core::PCWSTR) -> String {
    if p.is_null() {
        return "(unnamed)".into();
    }
    // SAFETY: a NUL-terminated string of the runtime's.
    unsafe { p.to_string() }.unwrap_or_else(|_| "(unreadable)".into())
}

fn allocation_type(t: D3D12_DRED_ALLOCATION_TYPE) -> &'static str {
    match t.0 {
        19 => "command queue",
        20 => "command allocator",
        21 => "pipeline state",
        22 => "command list",
        23 => "fence",
        24 => "descriptor heap",
        25 => "heap",
        27 => "query heap",
        28 => "command signature",
        29 => "pipeline library",
        34 => "resource",
        _ => "other",
    }
}

fn op_name(op: D3D12_AUTO_BREADCRUMB_OP) -> String {
    let s = match op.0 {
        0 => "SetMarker",
        1 => "BeginEvent",
        2 => "EndEvent",
        3 => "DrawInstanced",
        4 => "DrawIndexedInstanced",
        5 => "ExecuteIndirect",
        6 => "Dispatch",
        7 => "CopyBufferRegion",
        8 => "CopyTextureRegion",
        9 => "CopyResource",
        11 => "ResolveSubresource",
        12 => "ClearRenderTargetView",
        13 => "ClearUnorderedAccessView",
        14 => "ClearDepthStencilView",
        15 => "ResourceBarrier",
        16 => "ExecuteBundle",
        17 => "Present",
        18 => "ResolveQueryData",
        19 => "BeginSubmission",
        20 => "EndSubmission",
        25 => "ResolveSubresourceRegion",
        26 => "WriteBufferImmediate",
        31 => "BuildRaytracingAccelerationStructure",
        34 => "DispatchRays",
        39 => "SetPipelineState1",
        45 => "Barrier",
        46 => "BeginCommandList",
        _ => return format!("operation {}", op.0),
    };
    s.into()
}

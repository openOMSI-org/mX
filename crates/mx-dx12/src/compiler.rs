//! HLSL to DXIL or DXBC, as wgpu-hal compiles it: DXC (`dxcompiler.dll`, loaded at run time)
//! when it can be found, FXC (`d3dcompiler_47.dll`, shader model 5.1) otherwise, with the same
//! arguments.

use std::ffi::c_void;

use mx_shader::disk_cache::{DiskCache, Key};
use mx_shader::hlsl::ShaderModel;
use parking_lot::Mutex;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::Dxc::*;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::System::LibraryLoader::{GetModuleFileNameW, GetProcAddress, LoadLibraryW};
use windows::core::{GUID, HRESULT, Interface, PCSTR, PCWSTR, s, w};

type DxcCreateInstanceFn = unsafe extern "system" fn(clsid: *const GUID, iid: *const GUID, out: *mut *mut c_void) -> HRESULT;

type D3DCompileFn = unsafe extern "system" fn(
    src: *const c_void,
    len: usize,
    name: PCSTR,
    defines: *const c_void,
    include: *mut c_void,
    entry: PCSTR,
    target: PCSTR,
    flags1: u32,
    flags2: u32,
    code: *mut *mut c_void,
    errors: *mut *mut c_void,
) -> HRESULT;

/// `D3DCOMPILE_ENABLE_STRICTNESS`, `D3DCOMPILE_DEBUG`, `D3DCOMPILE_SKIP_OPTIMIZATION`.
const D3DCOMPILE_ENABLE_STRICTNESS: u32 = 1 << 11;
const D3DCOMPILE_DEBUG: u32 = 1;
const D3DCOMPILE_SKIP_OPTIMIZATION: u32 = 1 << 2;

/// Whether shaders are compiled with debug information and without optimisation, as wgpu-hal
/// does under `InstanceFlags::DEBUG`, which wgpu turns on in debug builds
/// (`InstanceFlags::from_build_config`): the same here, so a debug build of mX compiles what
/// a debug build of wgpu does. `MX_DX12_SHADER_DEBUG=0` / `1` overrides it.
pub(crate) fn debug_shaders() -> bool {
    match std::env::var("MX_DX12_SHADER_DEBUG") {
        Ok(v) => v != "0",
        Err(_) => cfg!(debug_assertions),
    }
}

pub(crate) enum Compiler {
    Dxc {
        compiler: Mutex<IDxcCompiler3>,
        /// The highest shader model this DXC knows (wgpu: DXC 1.x compiles SM 6.x).
        max_shader_model: ShaderModel,
        /// The compiler's version and build, for the disk cache's keys.
        identity: String,
        _lib: HMODULE,
    },
    Fxc {
        compile: D3DCompileFn,
        identity: String,
        _lib: HMODULE,
    },
}

/// The size of the DLL `lib` was loaded from: two builds of one version differ in it.
fn dll_size(lib: HMODULE) -> u64 {
    let mut path = [0u16; 1024];
    // SAFETY: a module handle of this process and a buffer of the length given.
    let n = unsafe { GetModuleFileNameW(Some(lib), &mut path) } as usize;
    if n == 0 || n >= path.len() {
        return 0;
    }
    std::fs::metadata(String::from_utf16_lossy(&path[..n])).map_or(0, |m| m.len())
}

// SAFETY: the DXC compiler object is only used under its mutex; FXC's D3DCompile is
// thread-safe; the module handles are never freed.
unsafe impl Send for Compiler {}
// SAFETY: as above.
unsafe impl Sync for Compiler {}

fn shader_model_of_dxc(major: u32, minor: u32) -> ShaderModel {
    use ShaderModel as S;
    if major > 1 {
        return S::V6_9;
    }
    match minor {
        0 => S::V6_0,
        1 => S::V6_1,
        2 => S::V6_2,
        3 => S::V6_3,
        4 => S::V6_4,
        5 => S::V6_5,
        6 => S::V6_6,
        7 => S::V6_7,
        8 => S::V6_8,
        _ => S::V6_9,
    }
}

impl Compiler {
    /// DXC if `dxcompiler.dll` loads, else FXC (as wgpu's `Dx12Compiler::Auto`);
    /// `MX_DX12_COMPILER=fxc` asks for FXC (tests of FXC's optimizer).
    pub fn load() -> Result<Compiler, String> {
        if std::env::var("MX_DX12_COMPILER").is_ok_and(|v| v.eq_ignore_ascii_case("fxc")) {
            return Self::fxc();
        }
        match Self::dxc() {
            Ok(c) => Ok(c),
            Err(e) => {
                log::info!("mX: {e}; compiling shaders with FXC (shader model 5.1)");
                Self::fxc()
            }
        }
    }

    fn dxc() -> Result<Compiler, String> {
        // SAFETY: loading a system / application DLL by name; the handle is kept for good.
        let lib = unsafe { LoadLibraryW(w!("dxcompiler.dll")) }.map_err(|e| format!("dxcompiler.dll not loaded ({e})"))?;
        // SAFETY: a symbol of the loaded module, with DXC's documented signature.
        let create: DxcCreateInstanceFn = unsafe {
            let f = GetProcAddress(lib, s!("DxcCreateInstance")).ok_or("dxcompiler.dll has no DxcCreateInstance")?;
            std::mem::transmute::<unsafe extern "system" fn() -> isize, DxcCreateInstanceFn>(f)
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid GUIDs and out-pointer.
        unsafe { create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut raw) }
            .ok()
            .map_err(|e| format!("DxcCreateInstance: {e}"))?;
        // SAFETY: DxcCreateInstance returned an owned IDxcCompiler3.
        let compiler = unsafe { IDxcCompiler3::from_raw(raw) };
        let (mut major, mut minor) = (1, 0);
        // (DXC 1.0 has no version interface: assume 1.0, as wgpu does)
        if let Ok(v) = compiler.cast::<IDxcVersionInfo>() {
            // SAFETY: two valid out-pointers.
            let _ = unsafe { v.GetVersion(&mut major, &mut minor) };
        }
        // (DXC signs its output with dxil.dll where it finds one: a cached unsigned blob must
        // not be used once it is there, nor the other way round)
        // SAFETY: as above; DXC loads the same DLL itself when it compiles.
        let signed = unsafe { LoadLibraryW(w!("dxil.dll")) }.is_ok();
        Ok(Compiler::Dxc {
            compiler: Mutex::new(compiler),
            max_shader_model: shader_model_of_dxc(major, minor),
            identity: format!("dxc {major}.{minor} {} dxil {signed}", dll_size(lib)),
            _lib: lib,
        })
    }

    fn fxc() -> Result<Compiler, String> {
        // SAFETY: as in `dxc`.
        let lib = unsafe { LoadLibraryW(w!("d3dcompiler_47.dll")) }.map_err(|e| format!("neither dxcompiler.dll nor d3dcompiler_47.dll could be loaded ({e})"))?;
        // SAFETY: a symbol of the loaded module, with D3DCompile's documented signature.
        let compile: D3DCompileFn = unsafe {
            let f = GetProcAddress(lib, s!("D3DCompile")).ok_or("d3dcompiler_47.dll has no D3DCompile")?;
            std::mem::transmute::<unsafe extern "system" fn() -> isize, D3DCompileFn>(f)
        };
        Ok(Compiler::Fxc {
            compile,
            identity: format!("fxc {}", dll_size(lib)),
            _lib: lib,
        })
    }

    /// The highest shader model the compiler takes (`None`: FXC, 5.1 only).
    pub fn max_shader_model(&self) -> Option<ShaderModel> {
        match self {
            Compiler::Dxc { max_shader_model, .. } => Some(*max_shader_model),
            Compiler::Fxc { .. } => None,
        }
    }

    pub fn is_dxc(&self) -> bool {
        matches!(self, Compiler::Dxc { .. })
    }

    fn identity(&self) -> &str {
        match self {
            Compiler::Dxc { identity, .. } | Compiler::Fxc { identity, .. } => identity,
        }
    }

    /// As [`Compiler::compile`], through the disk cache when there is one: a shader compiled
    /// in an earlier run (the same HLSL, entry point, profile, flags and compiler) is read
    /// back instead of compiled again.
    pub fn compile_cached(&self, disk: Option<&DiskCache>, source: &str, name: &str, entry: &str, profile: &str, f16: bool) -> Result<Vec<u8>, String> {
        let flags = format!("debug {} f16 {f16}", debug_shaders());
        let key = Key {
            compiler: self.identity(),
            source,
            entry,
            profile,
            flags: &flags,
        };
        if let Some(code) = disk.and_then(|d| d.get(&key)) {
            return Ok(code);
        }
        let code = self.compile(source, name, entry, profile, f16)?;
        if let Some(d) = disk {
            d.put(&key, &code);
        }
        Ok(code)
    }

    /// Compile `source`'s `entry` for `profile` (e.g. `ps_6_5`). `name` names the source in
    /// messages (the module's label, as wgpu passes it).
    pub fn compile(&self, source: &str, name: &str, entry: &str, profile: &str, f16: bool) -> Result<Vec<u8>, String> {
        match self {
            Compiler::Dxc { compiler, .. } => {
                let name = crate::wide(name);
                let entry = crate::wide(entry);
                let profile = crate::wide(profile);
                let mut args: Vec<PCWSTR> = vec![
                    PCWSTR(name.as_ptr()),
                    w!("-E"),
                    PCWSTR(entry.as_ptr()),
                    w!("-T"),
                    PCWSTR(profile.as_ptr()),
                    w!("-HV"),
                    w!("2018"),
                    w!("-no-warnings"),
                    DXC_ARG_ENABLE_STRICTNESS,
                ];
                if debug_shaders() {
                    args.push(DXC_ARG_DEBUG);
                    args.push(DXC_ARG_SKIP_OPTIMIZATIONS);
                }
                if f16 {
                    args.push(w!("-enable-16bit-types"));
                }
                let buffer = DxcBuffer {
                    Ptr: source.as_ptr().cast(),
                    Size: source.len(),
                    Encoding: DXC_CP_UTF8.0,
                };
                let compiler = compiler.lock();
                // SAFETY: the buffer and the arguments outlive the call.
                let result: IDxcResult = unsafe { compiler.Compile(&buffer, Some(&args), None) }.map_err(|e| format!("DXC: {e}"))?;
                let mut errors: Option<IDxcBlobUtf8> = None;
                // SAFETY: a valid out-pointer for the interface asked for.
                unsafe { result.GetOutput(DXC_OUT_ERRORS, std::ptr::null_mut(), &mut errors) }.map_err(|e| format!("DXC errors: {e}"))?;
                if let Some(errors) = errors {
                    // SAFETY: the blob's own string and length.
                    let text = unsafe {
                        let len = errors.GetStringLength();
                        if len == 0 {
                            String::new()
                        } else {
                            String::from_utf8_lossy(std::slice::from_raw_parts(errors.GetStringPointer().0, len)).into_owned()
                        }
                    };
                    if !text.is_empty() {
                        return Err(format!("DXC compile error: {text}"));
                    }
                }
                let mut object: Option<IDxcBlob> = None;
                // SAFETY: as above.
                unsafe { result.GetOutput(DXC_OUT_OBJECT, std::ptr::null_mut(), &mut object) }.map_err(|e| format!("DXC object: {e}"))?;
                let object = object.ok_or("DXC returned no object")?;
                // SAFETY: the blob's own buffer and size.
                Ok(unsafe { std::slice::from_raw_parts(object.GetBufferPointer().cast::<u8>(), object.GetBufferSize()) }.to_vec())
            }
            Compiler::Fxc { compile, .. } => {
                let cstr = |s: &str| std::ffi::CString::new(s).unwrap_or_default();
                let (name, entry, profile) = (cstr(name), cstr(entry), cstr(profile));
                let (mut code, mut errors) = (std::ptr::null_mut(), std::ptr::null_mut());
                // SAFETY: valid strings and out-pointers; the blobs returned are owned.
                let hr = unsafe {
                    compile(
                        source.as_ptr().cast(),
                        source.len(),
                        PCSTR(name.as_ptr().cast()),
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        PCSTR(entry.as_ptr().cast()),
                        PCSTR(profile.as_ptr().cast()),
                        if debug_shaders() {
                            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION
                        } else {
                            D3DCOMPILE_ENABLE_STRICTNESS
                        },
                        0,
                        &mut code,
                        &mut errors,
                    )
                };
                // SAFETY: owned blobs (or null) from D3DCompile.
                let (code, errors) = unsafe { (blob(code), blob(errors)) };
                match (hr.ok(), code) {
                    (Ok(()), Some(code)) => Ok(code),
                    (r, _) => Err(format!(
                        "FXC D3DCompile error ({}): {}",
                        r.err().map(|e| e.to_string()).unwrap_or_default(),
                        errors.map(|e| String::from_utf8_lossy(&e).into_owned()).unwrap_or_default()
                    )),
                }
            }
        }
    }
}

/// The bytes of an owned `ID3DBlob` pointer (released here).
///
/// # Safety
/// `p` is null or an owned `ID3DBlob`.
unsafe fn blob(p: *mut c_void) -> Option<Vec<u8>> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the caller's contract.
    let b = unsafe { ID3DBlob::from_raw(p) };
    // SAFETY: the blob's own buffer and size.
    Some(unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }.to_vec())
}

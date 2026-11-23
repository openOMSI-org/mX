//! openOMSI's own WGSL modules, as its renderer assembles them, compile through mX unchanged.
//!
//! The modules are in `tests/openomsi` (see its README); `MX_OPENOMSI_SHADERS=<dir>` takes
//! another set (`python3 tools/openomsi_shaders.py <openOMSI checkout> <dir>`).

use mx::*;

#[test]
fn every_openomsi_module_compiles_for_the_gpu() {
    let dir = std::env::var("MX_OPENOMSI_SHADERS").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/tests/openomsi").to_string());
    let adapter = Instance::new().default_adapter().expect("a GPU mX can use");
    eprintln!("on {:?}", adapter.info());
    if adapter.is_bound_tier() {
        // (these are the native renderer's modules - storage buffers, depth read by
        // textureLoad; on OpenGL openOMSI writes variants of its own, `gl_backend()`)
        eprintln!("skipped: the bound tier takes openOMSI's OpenGL variants, not these");
        return;
    }
    let (device, _queue) = adapter.request_device(&Default::default()).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "wgsl"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .wgsl files in {dir}");
    let mut failed = Vec::new();
    for path in &files {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        if name.starts_with("rt") && !adapter.supports_ray_query() {
            eprintln!("{name}: skipped, no ray queries");
            continue;
        }
        let source = std::fs::read_to_string(path).unwrap();
        let t = std::time::Instant::now();
        let r = device
            .create_shader_module(&ShaderModuleDescriptor { label: Some(&name), source: &source })
            .and_then(|m| device.precompile(&m));
        match r {
            Ok(()) => eprintln!("{name}: ok in {:.2} s", t.elapsed().as_secs_f32()),
            Err(e) => {
                eprintln!("{name}: FAILED\n{e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{} of {} modules failed: {failed:?}", failed.len(), files.len());
}

//! Compiled shaders kept on disk between runs, for the backends whose compiler is slow
//! (DXC and FXC on Direct3D: openOMSI's scene module took DXC some 15 s on WARP, and every
//! start of the game paid it again).
//!
//! An entry is the compiler's output for one key: the generated source, the entry point,
//! the profile, the flags and the compiler's identity. Its file is named by a 128-bit hash
//! of the key and holds the key's length and a second hash of it, so a file found under a
//! name is used only for the key that wrote it; a damaged or foreign file is ignored (and
//! written again). Writes go to a temporary file renamed into place, so two processes
//! filling the same cache never read half a file. The folder is kept under a size bound,
//! the oldest entries removed first.
//!
//! Where: `MX_SHADER_CACHE=<folder>`, or `MX_SHADER_CACHE=0` for none; otherwise the
//! platform's cache folder (`%LOCALAPPDATA%\mX\shaders`, `~/Library/Caches/mX/shaders`,
//! `$XDG_CACHE_HOME/mX/shaders` or `~/.cache/mX/shaders`).

use std::io::Write as _;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"MXSC";
/// Bumped when the file layout changes (old files are then ignored and replaced).
const FORMAT: u32 = 1;
const HEADER: usize = 4 + 4 + 8 + 8 + 8 + 8;
/// The folder's bound: past it the oldest entries go, down to three quarters of it.
const DEFAULT_MAX_BYTES: u64 = 512 << 20;

/// A folder of compiled shaders.
#[derive(Debug, Clone)]
pub struct DiskCache {
    dir: PathBuf,
    max_bytes: u64,
}

/// What an entry is the output for: everything the compiler's output depends on.
#[derive(Debug, Clone, Copy)]
pub struct Key<'a> {
    /// The compiler and its version, and anything else of it the output depends on.
    pub compiler: &'a str,
    pub source: &'a str,
    pub entry: &'a str,
    pub profile: &'a str,
    /// The compiler's flags, as text.
    pub flags: &'a str,
}

impl Key<'_> {
    fn parts(&self) -> [&[u8]; 5] {
        [self.compiler.as_bytes(), self.source.as_bytes(), self.entry.as_bytes(), self.profile.as_bytes(), self.flags.as_bytes()]
    }

    fn len(&self) -> u64 {
        self.parts().iter().map(|p| p.len() as u64 + 8).sum()
    }

    /// Hashes of the key with `seed` (each part length-prefixed, so parts cannot run into
    /// each other).
    fn hash(&self, seed: u64) -> u64 {
        let mut h = Fnv::new(seed);
        for p in self.parts() {
            h.write(&(p.len() as u64).to_le_bytes());
            h.write(p);
        }
        h.finish()
    }
}

/// FNV-1a, 64 bits: stable across Rust versions and platforms, unlike `std`'s hasher.
struct Fnv(u64);

impl Fnv {
    fn new(seed: u64) -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn finish(&self) -> u64 {
        // (a final mix: FNV's low bits are weak on their own)
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    let mut h = Fnv::new(7);
    h.write(bytes);
    h.finish()
}

impl DiskCache {
    /// The cache at `MX_SHADER_CACHE` or the platform's cache folder (`None`: turned off, or
    /// no folder could be made).
    pub fn from_env() -> Option<DiskCache> {
        let dir = match std::env::var_os("MX_SHADER_CACHE") {
            Some(v) if v == "0" || v.is_empty() => return None,
            Some(v) => PathBuf::from(v),
            None => default_dir()?,
        };
        DiskCache::open(&dir)
    }

    /// The cache in `dir` (made if missing), trimmed to its bound.
    pub fn open(dir: &Path) -> Option<DiskCache> {
        DiskCache::with_bound(dir, DEFAULT_MAX_BYTES)
    }

    /// As [`DiskCache::open`], bounded to `max_bytes`.
    pub fn with_bound(dir: &Path, max_bytes: u64) -> Option<DiskCache> {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::warn!("mX: no shader cache in {}: {e}", dir.display());
            return None;
        }
        let cache = DiskCache { dir: dir.to_path_buf(), max_bytes };
        cache.trim();
        Some(cache)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, key: &Key<'_>) -> PathBuf {
        self.dir.join(format!("{:016x}{:016x}.bin", key.hash(1), key.hash(2)))
    }

    /// The compiled code stored for `key`, if any (and whole).
    pub fn get(&self, key: &Key<'_>) -> Option<Vec<u8>> {
        let bytes = std::fs::read(self.path(key)).ok()?;
        let (head, code) = bytes.split_at_checked(HEADER)?;
        let u32_at = |i: usize| u32::from_le_bytes(head[i..i + 4].try_into().unwrap());
        let u64_at = |i: usize| u64::from_le_bytes(head[i..i + 8].try_into().unwrap());
        let whole = &head[0..4] == MAGIC && u32_at(4) == FORMAT && u64_at(8) == key.len() && u64_at(16) == key.hash(3) && u64_at(24) == code.len() as u64 && u64_at(32) == checksum(code);
        whole.then(|| code.to_vec())
    }

    /// Store `code` for `key`. Failures (a full disk, a read-only folder) are logged and
    /// otherwise ignored: the cache only saves time.
    pub fn put(&self, key: &Key<'_>, code: &[u8]) {
        let path = self.path(key);
        let tmp = self.dir.join(format!(
            "{}.{}.{:x}.tmp",
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("entry"),
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())
        ));
        let write = || -> std::io::Result<()> {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(MAGIC)?;
            f.write_all(&FORMAT.to_le_bytes())?;
            f.write_all(&key.len().to_le_bytes())?;
            f.write_all(&key.hash(3).to_le_bytes())?;
            f.write_all(&(code.len() as u64).to_le_bytes())?;
            f.write_all(&checksum(code).to_le_bytes())?;
            f.write_all(code)?;
            f.sync_all()?;
            drop(f);
            std::fs::rename(&tmp, &path)
        };
        if let Err(e) = write() {
            let _ = std::fs::remove_file(&tmp);
            log::warn!("mX: shader cache entry not written ({}): {e}", path.display());
        }
    }

    /// Remove the oldest entries (by modification time) while the folder holds more than
    /// its bound, down to three quarters of it; and temporary files a crashed process left
    /// over an hour ago.
    fn trim(&self) {
        let Ok(read) = std::fs::read_dir(&self.dir) else { return };
        let now = std::time::SystemTime::now();
        let mut entries = Vec::new();
        let mut total = 0u64;
        for e in read.flatten() {
            let path = e.path();
            let Ok(meta) = e.metadata() else { continue };
            let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            match path.extension().and_then(|x| x.to_str()) {
                Some("bin") => {
                    total += meta.len();
                    entries.push((modified, meta.len(), path));
                }
                Some("tmp") if now.duration_since(modified).is_ok_and(|d| d.as_secs() > 3600) => {
                    let _ = std::fs::remove_file(&path);
                }
                _ => {}
            }
        }
        if total <= self.max_bytes {
            return;
        }
        entries.sort();
        let target = self.max_bytes / 4 * 3;
        for (_, len, path) in entries {
            if total <= target {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total -= len;
            }
        }
    }
}

fn default_dir() -> Option<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        env("LOCALAPPDATA")?
    } else if cfg!(target_vendor = "apple") {
        env("HOME")?.join("Library/Caches")
    } else {
        env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|h| h.join(".cache")))?
    };
    Some(base.join("mX").join("shaders"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mx-disk-cache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    const KEY: Key<'static> = Key {
        compiler: "dxc 1.8",
        source: "float4 main() : SV_Target { return 1; }",
        entry: "main",
        profile: "ps_6_0",
        flags: "",
    };

    #[test]
    fn entries_come_back_for_their_key_only() {
        let d = dir("keys");
        let c = DiskCache::open(&d).unwrap();
        assert_eq!(c.get(&KEY), None);
        c.put(&KEY, b"DXIL code");
        assert_eq!(c.get(&KEY).as_deref(), Some(&b"DXIL code"[..]));
        // (another process opening the folder later finds it)
        assert_eq!(DiskCache::open(&d).unwrap().get(&KEY).as_deref(), Some(&b"DXIL code"[..]));
        for other in [
            Key { compiler: "dxc 1.9", ..KEY },
            Key {
                source: "float4 main() : SV_Target { return 2; }",
                ..KEY
            },
            Key { entry: "main2", ..KEY },
            Key { profile: "ps_6_5", ..KEY },
            Key { flags: "-Zi", ..KEY },
            // (the parts' boundaries count: "main" + "ps_6_0" is not "mainp" + "s_6_0")
            Key {
                entry: "mainp",
                profile: "s_6_0",
                ..KEY
            },
        ] {
            assert_eq!(c.get(&other), None, "{other:?}");
        }
        c.put(&KEY, b"newer");
        assert_eq!(c.get(&KEY).as_deref(), Some(&b"newer"[..]));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn damaged_or_foreign_files_are_not_used() {
        let d = dir("damaged");
        let c = DiskCache::open(&d).unwrap();
        c.put(&KEY, b"0123456789");
        let path = c.path(&KEY);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(c.get(&KEY), None, "a flipped bit");
        std::fs::write(&path, &bytes[..HEADER + 3]).unwrap();
        assert_eq!(c.get(&KEY), None, "cut short");
        std::fs::write(&path, b"MXSC").unwrap();
        assert_eq!(c.get(&KEY), None, "no header");
        // a file of another key under this one's name (a hash collision) is refused
        let other = Key { source: "other", ..KEY };
        c.put(&other, b"x");
        std::fs::copy(c.path(&other), &path).unwrap();
        assert_eq!(c.get(&KEY), None, "another key's entry");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_folder_is_kept_under_its_bound_oldest_first() {
        let d = dir("bound");
        let c = DiskCache::with_bound(&d, 1 << 20).unwrap();
        let sources: Vec<String> = (0..12).map(|i| format!("source {i}")).collect();
        let keys: Vec<Key<'_>> = sources.iter().map(|s| Key { source: s, ..KEY }).collect();
        for (i, k) in keys.iter().enumerate() {
            c.put(k, &vec![i as u8; 128 << 10]);
            // (distinct modification times, oldest first)
            let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000 + i as u64);
            std::fs::File::options().write(true).open(c.path(k)).unwrap().set_modified(t).unwrap();
        }
        let c = DiskCache::with_bound(&d, 1 << 20).unwrap();
        let kept: Vec<bool> = keys.iter().map(|k| c.get(k).is_some()).collect();
        let n = kept.iter().filter(|&&k| k).count();
        assert!((1..=6).contains(&n), "{kept:?}");
        assert!(kept[11] && !kept[0], "the newest kept, the oldest gone: {kept:?}");
        assert!(kept.windows(2).all(|w| w[0] <= w[1]), "removed oldest first: {kept:?}");
        let _ = std::fs::remove_dir_all(&d);
    }
}

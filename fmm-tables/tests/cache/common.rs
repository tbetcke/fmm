//! Helpers of the cache tests: per-test directories, the file of a key, header
//! patching, an independent FNV-1a, and a macro that runs a generic check for every
//! family in both precisions.

use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};

use nd_fmm_tables::cache::{CacheKey, CachedTable, HEADER_LEN, TableCache};

/// Degrees of the round-trip tests (brief T7: cached tables at p ≤ 4 in tests).
pub const DEGREES: [usize; 3] = [0, 1, 4];

/// A fresh subdirectory of `CARGO_TARGET_TMPDIR`, one per test, removed on drop.
pub struct TestDir(PathBuf);

impl TestDir {
    /// Creates `CARGO_TARGET_TMPDIR/nd-fmm-tables-cache/<name>`, emptied first.
    pub fn new(name: &str) -> Self {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("nd-fmm-tables-cache")
            .join(name);
        if path.exists() {
            fs::remove_dir_all(&path).unwrap();
        }
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// Returns the directory.
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// Returns a cache in the subdirectory `sub`.
    pub fn cache(&self, sub: &str) -> TableCache {
        TableCache::new(self.0.join(sub))
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Returns the path of the file of family `F` at degree `p` in `cache`.
pub fn file_of<F: CachedTable>(cache: &TableCache, p: usize) -> PathBuf {
    cache.path(&CacheKey::of::<F>(p))
}

/// Returns the bytes of the file of family `F` at degree `p` in `cache`.
pub fn bytes_of<F: CachedTable>(cache: &TableCache, p: usize) -> Vec<u8> {
    fs::read(file_of::<F>(cache, p)).unwrap()
}

/// Returns the names of the entries of `dir`, sorted.
pub fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// Overwrites the u32 at byte `offset` of `path` with `value`, little-endian.
pub fn patch_u32(path: &Path, offset: usize, value: u32) {
    let mut bytes = fs::read(path).unwrap();
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    fs::write(path, bytes).unwrap();
}

/// Reads the u32 at byte `offset` of `bytes`, little-endian.
pub fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// Reads the u64 at byte `offset` of `bytes`, little-endian.
pub fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

/// FNV-1a 64-bit, written out independently of the crate.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// Returns the payload of a cache file.
pub fn payload(bytes: &[u8]) -> &[u8] {
    &bytes[HEADER_LEN..]
}

/// Stores `table` in a fresh cache under `dir` and returns the file bytes. Two tables
/// with byte-identical files are bit-identical: the payload holds every value bit for
/// bit.
pub fn encoded<F: CachedTable>(dir: &TestDir, sub: &str, table: &F) -> Vec<u8> {
    let cache = dir.cache(sub);
    cache.store(table).unwrap();
    bytes_of::<F>(&cache, table.p())
}

/// Asserts that `a` and `b` are the same table bit for bit: equal as values
/// (`PartialEq`, which covers every field) and with byte-identical files.
pub fn assert_bit_identical<F: CachedTable + PartialEq + Debug>(dir: &TestDir, a: &F, b: &F) {
    assert_eq!(a, b);
    assert!(encoded(dir, "bits-a", a) == encoded(dir, "bits-b", b));
}

/// Runs `$check::<F>($args)` for every table family F in f64 and f32.
macro_rules! for_each_family {
    ($check:ident($($arg:expr),* $(,)?)) => {{
        use nd_fmm_tables::{L2lTables, M2lClasses, M2lTables, M2mTables, RotationTables};
        $check::<M2mTables<f64>>($($arg),*);
        $check::<M2mTables<f32>>($($arg),*);
        $check::<L2lTables<f64>>($($arg),*);
        $check::<L2lTables<f32>>($($arg),*);
        $check::<M2lTables<f64>>($($arg),*);
        $check::<M2lTables<f32>>($($arg),*);
        $check::<M2lClasses<f64>>($($arg),*);
        $check::<M2lClasses<f32>>($($arg),*);
        $check::<RotationTables<f64>>($($arg),*);
        $check::<RotationTables<f32>>($($arg),*);
    }};
}

pub(crate) use for_each_family;

//! Rejection of stale, mismatched and corrupt files, by `load` and by `load_or_build`,
//! which must rebuild, report `Rebuilt` with the right error and replace the file.

use std::fmt::Debug;
use std::fs;

use nd_fmm_math::CONVENTION_VERSION;
use nd_fmm_tables::cache::{
    CacheError, CacheOutcome, CachedTable, FORMAT_VERSION, HEADER_LEN, KeyField, MAGIC, TableCache,
};
use nd_fmm_tables::{L2lTables, M2lClasses, M2lTables, M2mTables};

use crate::common::{
    TestDir, bytes_of, file_of, fnv1a, for_each_family, patch_u32, u32_at, u64_at,
};

/// Degree of the rejection tests.
const P: usize = 1;

/// Checks that the file of `F` at degree `p` is rejected with an error that satisfies
/// `is_expected`, by `load` and by `load_or_build`; that `load_or_build` returns the cold
/// build and reports `Rebuilt`; and that the replaced file loads.
fn assert_rejected<F: CachedTable + PartialEq + Debug>(
    cache: &TableCache,
    p: usize,
    is_expected: impl Fn(&CacheError) -> bool,
) {
    let error = cache.load::<F>(p).unwrap_err();
    assert!(is_expected(&error), "load: unexpected {error:?}");
    let (table, outcome) = cache.load_or_build::<F>(p);
    match outcome {
        CacheOutcome::Rebuilt(error) => {
            assert!(is_expected(&error), "load_or_build: unexpected {error:?}");
        }
        other => panic!("expected Rebuilt, got {other:?}"),
    }
    // Error measure: exact equality with the cold build, before and after replacement.
    let cold = F::build(p);
    assert_eq!(table, cold);
    assert_eq!(cache.load::<F>(p).unwrap(), cold);
}

/// Stores the build of `F` at [`P`] in `cache` and returns the path of its file.
fn stored<F: CachedTable>(cache: &TableCache) -> std::path::PathBuf {
    cache.store(&F::build(P)).unwrap();
    file_of::<F>(cache, P)
}

fn check_stale_versions<F: CachedTable + PartialEq + Debug>(dir: &TestDir) {
    let cache = dir.cache("cache");
    let path = stored::<F>(&cache);
    patch_u32(&path, 12, CONVENTION_VERSION + 1);
    assert_rejected::<F>(&cache, P, |e| {
        matches!(e, &CacheError::ConventionVersion { found, expected }
            if found == CONVENTION_VERSION + 1 && expected == CONVENTION_VERSION)
    });
    patch_u32(&path, 8, FORMAT_VERSION + 1);
    assert_rejected::<F>(&cache, P, |e| {
        matches!(e, &CacheError::FormatVersion { found, expected }
            if found == FORMAT_VERSION + 1 && expected == FORMAT_VERSION)
    });
}

#[test]
fn stale_convention_or_format_version_is_rebuilt() {
    let dir = TestDir::new("stale-versions");
    for_each_family!(check_stale_versions(&dir));
}

#[test]
fn file_renamed_to_another_key_is_rebuilt() {
    let dir = TestDir::new("key-mismatch");
    let cache = dir.cache("cache");
    let rename = |from: std::path::PathBuf, to: std::path::PathBuf| fs::rename(from, to).unwrap();

    // Another kind: an L2L file under the M2M name, a dense M2L file under the class
    // form's name.
    rename(
        stored::<L2lTables<f64>>(&cache),
        file_of::<M2mTables<f64>>(&cache, P),
    );
    assert_rejected::<M2mTables<f64>>(&cache, P, |e| {
        matches!(
            e,
            CacheError::KeyMismatch {
                field: KeyField::Kind,
                found: 2,
                expected: 1
            }
        )
    });
    rename(
        stored::<M2lTables<f32>>(&cache),
        file_of::<M2lClasses<f32>>(&cache, P),
    );
    assert_rejected::<M2lClasses<f32>>(&cache, P, |e| {
        matches!(
            e,
            CacheError::KeyMismatch {
                field: KeyField::Kind,
                found: 3,
                expected: 4
            }
        )
    });

    // Another degree: p = 1 under the name of p = 2.
    rename(
        stored::<M2mTables<f64>>(&cache),
        file_of::<M2mTables<f64>>(&cache, 2),
    );
    assert_rejected::<M2mTables<f64>>(&cache, 2, |e| {
        matches!(
            e,
            CacheError::KeyMismatch {
                field: KeyField::P,
                found: 1,
                expected: 2
            }
        )
    });

    // Another precision: f32 under the name of f64.
    rename(
        stored::<L2lTables<f32>>(&cache),
        file_of::<L2lTables<f64>>(&cache, P),
    );
    assert_rejected::<L2lTables<f64>>(&cache, P, |e| {
        matches!(
            e,
            CacheError::KeyMismatch {
                field: KeyField::Precision,
                found: 32,
                expected: 64
            }
        )
    });
}

fn check_truncated<F: CachedTable + PartialEq + Debug>(dir: &TestDir) {
    let cache = dir.cache("cache");
    let path = stored::<F>(&cache);
    let bytes = fs::read(&path).unwrap();
    let payload_len = (bytes.len() - HEADER_LEN) as u64;
    // One byte short of the payload.
    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert_rejected::<F>(&cache, P, |e| {
        matches!(e, &CacheError::Length { found, expected }
            if found == payload_len - 1 && expected == payload_len)
    });
    // Shorter than the header.
    fs::write(&path, &bytes[..20]).unwrap();
    assert_rejected::<F>(
        &cache,
        P,
        |e| matches!(e, &CacheError::Length { found: 20, expected } if expected == HEADER_LEN as u64),
    );
    // Empty.
    fs::write(&path, []).unwrap();
    assert_rejected::<F>(&cache, P, |e| {
        matches!(e, CacheError::Length { found: 0, .. })
    });
}

#[test]
fn truncated_file_is_rebuilt() {
    let dir = TestDir::new("truncated");
    for_each_family!(check_truncated(&dir));
}

fn check_flipped_byte<F: CachedTable + PartialEq + Debug>(dir: &TestDir) {
    let cache = dir.cache("cache");
    let path = stored::<F>(&cache);
    let bytes = fs::read(&path).unwrap();
    // The first, a middle and the last payload byte; the last lies in the index tables
    // of the class form and the rotation tables, where the checksum must still win.
    for at in [HEADER_LEN, (HEADER_LEN + bytes.len()) / 2, bytes.len() - 1] {
        let mut corrupt = bytes.clone();
        corrupt[at] ^= 0x10;
        fs::write(&path, &corrupt).unwrap();
        let stored_checksum = u64_at(&bytes, 36);
        assert_rejected::<F>(&cache, P, |e| {
            matches!(e, &CacheError::Checksum { found, expected }
                if expected == stored_checksum && found == fnv1a(&corrupt[HEADER_LEN..]))
        });
    }
}

#[test]
fn flipped_payload_byte_is_rebuilt() {
    let dir = TestDir::new("checksum");
    for_each_family!(check_flipped_byte(&dir));
}

fn check_bad_magic<F: CachedTable + PartialEq + Debug>(dir: &TestDir) {
    let cache = dir.cache("cache");
    let path = stored::<F>(&cache);
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] = b'X';
    fs::write(&path, &bytes).unwrap();
    assert_rejected::<F>(&cache, P, |e| {
        matches!(e, CacheError::BadMagic { found, expected }
            if found[0] == b'X' && found[1..] == MAGIC[1..] && *expected == MAGIC)
    });
}

#[test]
fn bad_magic_is_rebuilt() {
    let dir = TestDir::new("magic");
    for_each_family!(check_bad_magic(&dir));
}

/// Replaces the payload of the file at `path` and fixes its length and checksum, as a
/// writer of another format could.
fn rewrite_payload(path: &std::path::Path, payload: &[u8]) {
    let mut bytes = fs::read(path).unwrap()[..HEADER_LEN].to_vec();
    bytes[28..36].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes[36..44].copy_from_slice(&fnv1a(payload).to_le_bytes());
    bytes.extend_from_slice(payload);
    fs::write(path, bytes).unwrap();
}

#[test]
fn consistent_header_with_wrong_payload_structure_is_rebuilt() {
    let dir = TestDir::new("structure");
    let cache = dir.cache("cache");

    // A payload shorter than the table needs, with a matching length and checksum.
    let path = stored::<M2mTables<f64>>(&cache);
    let full = bytes_of::<M2mTables<f64>>(&cache, P)[HEADER_LEN..].to_vec();
    let full_len = full.len() as u64;
    rewrite_payload(&path, &full[..full.len() / 2]);
    assert_rejected::<M2mTables<f64>>(&cache, P, |e| {
        matches!(e, &CacheError::Length { found, expected }
            if found == full_len / 2 && expected == full_len)
    });

    // Trailing bytes after the table.
    let mut longer = full.clone();
    longer.extend_from_slice(&[0; 8]);
    rewrite_payload(&path, &longer);
    assert_rejected::<M2mTables<f64>>(&cache, P, |e| {
        matches!(e, &CacheError::Length { found, expected }
            if found == full_len + 8 && expected == full_len)
    });

    // A class index out of range in the offsets of the class form: the first u32 of
    // the last (class, element) pair.
    let path = stored::<M2lClasses<f64>>(&cache);
    let mut payload = bytes_of::<M2lClasses<f64>>(&cache, P)[HEADER_LEN..].to_vec();
    let at = payload.len() - 8;
    assert!(u32_at(&payload, at) < 16);
    payload[at..at + 4].copy_from_slice(&16u32.to_le_bytes());
    rewrite_payload(&path, &payload);
    assert_rejected::<M2lClasses<f64>>(&cache, P, |e| {
        matches!(
            e,
            CacheError::InvalidPayload {
                found: 16,
                bound: 16,
                ..
            }
        )
    });
}

//! The files themselves: the documented header, determinism, atomic writes and
//! temporary files, an unwritable directory, the f32 files, and the file sizes.

use std::fs;
use std::thread;

use nd_fmm_math::CONVENTION_VERSION;
use nd_fmm_tables::cache::{
    CacheError, CacheKey, CacheOutcome, CachedTable, FORMAT_VERSION, HEADER_LEN, MAGIC, Precision,
    Stored, TableCache,
};
use nd_fmm_tables::{L2lTables, M2lClasses, M2lTables, M2mTables, RotationTables};

use crate::common::{
    TestDir, bytes_of, encoded, entries, file_of, fnv1a, for_each_family, payload, u32_at, u64_at,
};

fn check_header<F: CachedTable>(dir: &TestDir) {
    let cache = dir.cache("cache");
    let p = 2;
    cache.store(&F::build(p)).unwrap();
    let bytes = bytes_of::<F>(&cache, p);
    let key = CacheKey::of::<F>(p);
    // Error measure: exact equality with the layout of the module documentation.
    assert_eq!(bytes[0..8], MAGIC);
    assert_eq!(&MAGIC, b"NDFMMTAB");
    assert_eq!(u32_at(&bytes, 8), FORMAT_VERSION);
    assert_eq!(u32_at(&bytes, 12), CONVENTION_VERSION);
    assert_eq!(u32_at(&bytes, 16), key.kind.code());
    assert_eq!(u32_at(&bytes, 20), p as u32);
    assert_eq!(
        u32_at(&bytes, 24),
        8 * <F::Scalar as Stored>::PRECISION.bytes() as u32
    );
    assert_eq!(u64_at(&bytes, 28), (bytes.len() - HEADER_LEN) as u64);
    assert_eq!(u64_at(&bytes, 36), fnv1a(payload(&bytes)));
}

#[test]
fn header_follows_the_documented_layout() {
    let dir = TestDir::new("header");
    for_each_family!(check_header(&dir));
}

/// The payload of a matrix family: the matrices of `values`, each value
/// little-endian in precision `T`.
fn matrix_payload<T: Stored>(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &v in values {
        match T::PRECISION {
            Precision::F64 => bytes.extend_from_slice(&v.to_le_bytes()),
            Precision::F32 => bytes.extend_from_slice(&(v as f32).to_le_bytes()),
        }
    }
    bytes
}

#[test]
fn matrix_payloads_are_the_matrix_sets_in_both_precisions() {
    // The payload of M2M, L2L and dense M2L is `MatrixSet::as_slice` of the f64 build,
    // little-endian; in f32 each entry is rounded with `as f32`.
    //
    // Error measure: byte-for-byte equality.
    let dir = TestDir::new("matrix-payloads");
    let cache = dir.cache("cache");
    let p = 2;
    let m2m = M2mTables::<f64>::build(p);
    let l2l = L2lTables::<f64>::build(p);
    let m2l = M2lTables::<f64>::build(p);
    cache.store(&M2mTables::<f64>::build(p)).unwrap();
    cache.store(&M2mTables::<f32>::build(p)).unwrap();
    cache.store(&L2lTables::<f64>::build(p)).unwrap();
    cache.store(&L2lTables::<f32>::build(p)).unwrap();
    cache.store(&M2lTables::<f64>::build(p)).unwrap();
    cache.store(&M2lTables::<f32>::build(p)).unwrap();
    let check = |bytes: Vec<u8>, want: Vec<u8>| assert!(payload(&bytes) == want.as_slice());
    let m2m = m2m.matrices().as_slice();
    let l2l = l2l.matrices().as_slice();
    let m2l = m2l.matrices().as_slice();
    check(
        bytes_of::<M2mTables<f64>>(&cache, p),
        matrix_payload::<f64>(m2m),
    );
    check(
        bytes_of::<M2mTables<f32>>(&cache, p),
        matrix_payload::<f32>(m2m),
    );
    check(
        bytes_of::<L2lTables<f64>>(&cache, p),
        matrix_payload::<f64>(l2l),
    );
    check(
        bytes_of::<L2lTables<f32>>(&cache, p),
        matrix_payload::<f32>(l2l),
    );
    check(
        bytes_of::<M2lTables<f64>>(&cache, p),
        matrix_payload::<f64>(m2l),
    );
    check(
        bytes_of::<M2lTables<f32>>(&cache, p),
        matrix_payload::<f32>(m2l),
    );
}

#[test]
fn f32_files_equal_the_cast_of_the_f64_build() {
    // The class form and the rotation tables have `cast`; their f32 files equal the
    // files of the cast f64 build. The matrix families are checked value by value in
    // `matrix_payloads_are_the_matrix_sets_in_both_precisions`.
    //
    // Error measure: byte-for-byte equality of the files.
    let dir = TestDir::new("f32-cast");
    for p in [0, 1, 4] {
        let direct = encoded(&dir, "built", &M2lClasses::<f32>::build(p));
        let cast = encoded(&dir, "cast", &M2lClasses::<f64>::build(p).cast::<f32>());
        assert!(direct == cast, "class form, p = {p}");
        let direct = encoded(&dir, "built", &RotationTables::<f32>::build(p));
        let cast = encoded(&dir, "cast", &RotationTables::<f64>::build(p).cast::<f32>());
        assert!(direct == cast, "rotation tables, p = {p}");
    }
}

fn check_deterministic<F: CachedTable>(dir: &TestDir) {
    let cache = dir.cache("cache");
    let p = 2;
    cache.store(&F::build(p)).unwrap();
    let first = bytes_of::<F>(&cache, p);
    // A second build over the existing file, twice.
    let table = F::build(p);
    cache.store(&table).unwrap();
    assert!(bytes_of::<F>(&cache, p) == first);
    cache.store(&table).unwrap();
    assert!(bytes_of::<F>(&cache, p) == first);
}

#[test]
fn storing_twice_gives_byte_identical_files() {
    // Error measure: byte-for-byte equality of the files.
    let dir = TestDir::new("deterministic");
    for_each_family!(check_deterministic(&dir));
}

#[test]
fn concurrent_stores_of_one_key_both_succeed() {
    // Two writers of the same key at the same time, each with its own build; threads
    // of one process stand in for two processes (the temporary names differ in the
    // counter, as they would in the process id).
    //
    // Error measure: byte-for-byte equality with a sequential store.
    let dir = TestDir::new("concurrent");
    let p = 3;
    let reference = encoded(&dir, "reference", &RotationTables::<f64>::build(p));
    let cache = dir.cache("cache");
    for _ in 0..4 {
        thread::scope(|s| {
            let writers: Vec<_> = (0..2)
                .map(|_| s.spawn(|| cache.store(&RotationTables::<f64>::build(p))))
                .collect();
            for writer in writers {
                writer.join().unwrap().unwrap();
            }
        });
        assert!(bytes_of::<RotationTables<f64>>(&cache, p) == reference);
        assert_eq!(
            entries(cache.dir()),
            [CacheKey::of::<RotationTables<f64>>(p).file_name()]
        );
    }
}

#[test]
fn no_temporary_file_remains_after_a_store() {
    // Error measure: exact equality of the directory listing.
    let dir = TestDir::new("no-temporary");
    let cache = dir.cache("cache");
    cache.store(&M2mTables::<f64>::build(1)).unwrap();
    cache.store(&M2lClasses::<f32>::build(1)).unwrap();
    cache.store(&M2mTables::<f64>::build(1)).unwrap();
    let mut want = vec![
        CacheKey::of::<M2mTables<f64>>(1).file_name(),
        CacheKey::of::<M2lClasses<f32>>(1).file_name(),
    ];
    want.sort();
    assert_eq!(entries(cache.dir()), want);
}

#[test]
fn leftover_temporary_files_do_not_affect_loading() {
    // A crashed writer leaves a partial temporary file, headerless or with a valid
    // header and a short payload; some carry this process's id, so the writer must
    // skip their names.
    //
    // Error measure: exact equality with the cold build.
    let dir = TestDir::new("leftover");
    let cache = dir.cache("cache");
    fs::create_dir_all(cache.dir()).unwrap();
    let p = 1;
    let name = CacheKey::of::<M2mTables<f64>>(p).file_name();
    let complete = encoded(&dir, "complete", &M2mTables::<f64>::build(p));
    let mut leftovers = vec![format!(".{name}.99999999-0.tmp")];
    leftovers.extend((0..8).map(|k| format!(".{name}.{}-{k}.tmp", std::process::id())));
    for (k, leftover) in leftovers.iter().enumerate() {
        let partial = if k % 2 == 0 {
            vec![0; HEADER_LEN + 5]
        } else {
            complete[..complete.len() / 2].to_vec()
        };
        fs::write(cache.dir().join(leftover), partial).unwrap();
    }

    // Loading sees no file, and a cold `load_or_build` builds and stores.
    assert!(cache.load::<M2mTables<f64>>(p).unwrap_err().is_not_found());
    let (table, outcome) = cache.load_or_build::<M2mTables<f64>>(p);
    assert!(matches!(outcome, CacheOutcome::Built), "{outcome:?}");
    assert_eq!(table, M2mTables::<f64>::build(p));
    let (_, outcome) = cache.load_or_build::<M2mTables<f64>>(p);
    assert!(matches!(outcome, CacheOutcome::Loaded), "{outcome:?}");
    assert!(bytes_of::<M2mTables<f64>>(&cache, p) == complete);

    // The leftovers are untouched, and there is no new temporary file.
    let mut want = leftovers.clone();
    want.push(name);
    want.sort();
    assert_eq!(entries(cache.dir()), want);
}

#[test]
fn unwritable_directory_gives_not_stored_and_a_correct_table() {
    // The cache directory lies below a regular file, which fails even as root.
    //
    // Error measure: exact equality with the cold build.
    let dir = TestDir::new("unwritable");
    let blocker = dir.path().join("regular-file");
    fs::write(&blocker, b"not a directory").unwrap();
    let cache = TableCache::new(blocker.join("cache"));
    let p = 1;

    let error = cache.load::<M2lTables<f64>>(p).unwrap_err();
    assert!(matches!(error, CacheError::Io { .. }), "{error:?}");
    let error = cache.store(&M2lTables::<f64>::build(p)).unwrap_err();
    assert!(matches!(error, CacheError::Io { .. }), "{error:?}");

    let (table, outcome) = cache.load_or_build::<M2lTables<f64>>(p);
    assert!(
        matches!(outcome, CacheOutcome::NotStored(CacheError::Io { .. })),
        "{outcome:?}"
    );
    assert_eq!(table, M2lTables::<f64>::build(p));
    let (table, outcome) = cache.load_or_build::<RotationTables<f32>>(p);
    assert!(matches!(outcome, CacheOutcome::NotStored(_)), "{outcome:?}");
    assert_eq!(table, RotationTables::<f32>::build(p));

    // Nothing was written.
    assert_eq!(fs::read(&blocker).unwrap(), b"not a directory");
    assert_eq!(entries(dir.path()), ["regular-file"]);
}

#[test]
fn file_lengths_of_the_matrix_families() {
    // n = (p + 1)², B = (p + 1)(2p + 1)(2p + 3)/3.
    //
    // Error measure: exact equality of the file lengths.
    let dir = TestDir::new("lengths");
    let cache = dir.cache("cache");
    for p in [0, 1, 3] {
        let n = (p + 1) * (p + 1);
        let b = (p + 1) * (2 * p + 1) * (2 * p + 3) / 3;
        let len = |values: usize, bytes: usize, indices: usize| {
            (HEADER_LEN + values * bytes + 4 * indices) as u64
        };
        let file_len = |path| fs::metadata(path).unwrap().len();
        cache.store(&M2mTables::<f64>::build(p)).unwrap();
        cache.store(&M2lTables::<f32>::build(p)).unwrap();
        cache.store(&M2lClasses::<f64>::build(p)).unwrap();
        assert_eq!(
            file_len(file_of::<M2mTables<f64>>(&cache, p)),
            len(8 * n * n, 8, 0)
        );
        assert_eq!(
            file_len(file_of::<M2lTables<f32>>(&cache, p)),
            len(316 * n * n, 4, 0)
        );
        assert_eq!(
            file_len(file_of::<M2lClasses<f64>>(&cache, p)),
            len(16 * n * n + 96 * b, 8, 2 * 316)
        );
    }
}

fn print_size<F: CachedTable>(cache: &TableCache, p: usize) {
    let (_, outcome) = cache.load_or_build::<F>(p);
    assert!(matches!(outcome, CacheOutcome::Built), "{outcome:?}");
    let bytes = fs::metadata(file_of::<F>(cache, p)).unwrap().len();
    println!(
        "{:<32} {:>12} bytes  {:>9.3} MB",
        CacheKey::of::<F>(p).file_name(),
        bytes,
        bytes as f64 / 1e6
    );
}

#[test]
#[ignore = "builds every family at p = 8; run in release with --nocapture"]
fn file_sizes_at_p8() {
    // Reports the file size of every family at p = 8 (brief T7).
    let dir = TestDir::new("sizes-p8");
    let cache = dir.cache("cache");
    for_each_family!(print_size(&cache, 8));
}

//! The tuner of the device path (Phase 4 T12, C4.7; docs/phase4/T12-autotune.md, "Tests
//! that define done"; the `tune` module of `nd-fmm-exec`).
//!
//! Without MPI, on the tuning cache alone ([`TuningCache`]):
//! - the round trip: stored entries load back equal, every field;
//! - a file of another key, or with another CubeCL, convention, candidate-set or format
//!   version, is rejected as stale; a file with a flipped byte, cut short at any length,
//!   empty, or not a tuning-cache file is rejected as corrupt, never with a panic;
//! - two concurrent writers (eight threads, each storing its own entries forty times)
//!   leave a valid file, equal to one writer's entries, and no temporary file.
//!
//! With MPI, one test (this executable's one MPI initialisation), on the CPU runtime
//! (feature `cpu`): the scenarios of `tune_common` in f64 at p = 6 (all of them) and in f32
//! at p = 3 (without the stale files and the budget, which do not depend on the
//! precision), and the strategy at p = 12 in f64, which without `table_cache` keeps the static rule
//! (`Rotation`; the dense candidate would take seconds to build) and is not stored, and
//! with it is tuned. The test prints the backends it ran. Metal runs the same scenarios in
//! `tests/device_metal.rs` (ignored, by hand), CUDA in `tests/device_cuda.rs` (ignored, by
//! hand on locust; f32 p = 8 and f64 p = 6 in full, f32 p = 3, and the strategy at f64
//! p = 12), since this executable's one MPI test is the CPU runtime's.
//!
//! ```text
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release --test device_tune -- --nocapture
//! ```
#![cfg(feature = "gpu")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use nd_fmm_exec::fmm::OperatorKind;
use nd_fmm_exec::tune::{
    CANDIDATE_SET_VERSION, Candidate, Decision, DecisionReport, FORMAT_VERSION, GemmChoice, Source,
    Timing, TuningCache, TuningCacheError, TuningKey,
};
use nd_fmm_kernels::p2p::P2pLayout;
use nd_fmm_kernels::translate::{GemmLayout, Orientation};
use nd_fmm_kernels::{BackendKind, Precision};

#[cfg(feature = "cpu")]
mod tune_common;

/// A fresh, empty directory under `CARGO_TARGET_TMPDIR`.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("device_tune")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("the test directory is created");
    dir
}

fn key() -> TuningKey {
    TuningKey {
        backend: BackendKind::Metal,
        device: "Apple M3 Max".into(),
        compiler: "wgpu<msl>".into(),
        precision: Precision::F32,
        p: 8,
        gradients: true,
    }
}

/// Entries of every kind of decision and timing, `tag` making them distinct.
fn entries(tag: u64) -> Vec<DecisionReport> {
    let hand = GemmChoice::hand_written(GemmLayout::Cube {
        rows: 32,
        columns: 2,
        per_unit: 4,
    });
    let library = GemmChoice {
        orientation: Orientation::CoefficientMajor,
        ..GemmChoice::library(Precision::F32)
    };
    vec![
        DecisionReport {
            decision: Decision::Strategy,
            gradients: true,
            source: Source::Tuned,
            choice: Candidate::Dense(hand),
            level: Some(4),
            pairs: 584_136,
            cut: false,
            times: vec![
                (
                    Candidate::Dense(hand),
                    Timing::Measured(Duration::from_nanos(4_762_400 + tag)),
                ),
                (
                    Candidate::Rotation,
                    Timing::Measured(Duration::from_nanos(6_700_000)),
                ),
                (
                    Candidate::Dense(library),
                    Timing::Unregistered("the library does not take the level's shapes".into()),
                ),
            ],
            note: None,
        },
        DecisionReport {
            decision: Decision::Gemm {
                kind: OperatorKind::M2m,
                bucket: 4096,
            },
            gradients: false,
            source: Source::Tuned,
            choice: Candidate::Gemm(library),
            level: None,
            pairs: tag as usize,
            cut: true,
            times: vec![
                (Candidate::Gemm(hand), Timing::Failed("a launch".into())),
                (
                    Candidate::Gemm(library),
                    Timing::Measured(Duration::from_micros(80)),
                ),
                (
                    Candidate::Gemm(hand),
                    Timing::Skipped("past the tuning deadline".into()),
                ),
            ],
            note: None,
        },
        DecisionReport {
            decision: Decision::P2p { bucket: 64 },
            gradients: true,
            source: Source::Tuned,
            choice: Candidate::P2p(P2pLayout::Plane { planes: 2 }),
            level: Some(4),
            pairs: 1 << 30,
            cut: false,
            times: Vec::new(),
            note: None,
        },
    ]
}

/// Loaded entries are marked cached; otherwise equal.
fn as_loaded(entries: Vec<DecisionReport>) -> Vec<DecisionReport> {
    entries
        .into_iter()
        .map(|e| DecisionReport {
            source: Source::Cached,
            ..e
        })
        .collect()
}

#[test]
fn the_cache_round_trips() {
    let dir = fresh("round_trip");
    let cache = TuningCache::new(&dir);
    assert_eq!(cache.load(&key()), Err(TuningCacheError::Missing));
    cache.store(&key(), &entries(7)).unwrap();
    assert_eq!(cache.load(&key()).unwrap(), as_loaded(entries(7)));
    // Stored again, the file is replaced whole.
    cache.store(&key(), &entries(8)).unwrap();
    assert_eq!(cache.load(&key()).unwrap(), as_loaded(entries(8)));
    let files: Vec<_> = fs::read_dir(&dir).unwrap().collect();
    assert_eq!(files.len(), 1, "one file, no temporary left");
    println!(
        "tuning cache: {} round-trips every field of {} entries",
        key().file_name(),
        entries(0).len()
    );
}

#[test]
fn stale_and_corrupt_files_are_rejected() {
    let dir = fresh("rejected");
    let cache = TuningCache::new(&dir);
    cache.store(&key(), &entries(1)).unwrap();
    let path = cache.path(&key());
    let original = fs::read_to_string(&path).unwrap();
    // Another key under this key's name: each field of the key.
    for other in [
        TuningKey { p: 9, ..key() },
        TuningKey {
            precision: Precision::F64,
            ..key()
        },
        TuningKey {
            device: "Apple M4 Max".into(),
            ..key()
        },
        TuningKey {
            compiler: "wgpu<wgsl>".into(),
            ..key()
        },
        TuningKey {
            backend: BackendKind::Cuda,
            ..key()
        },
    ] {
        let other_dir = fresh("other");
        TuningCache::new(&other_dir)
            .store(&other, &entries(1))
            .unwrap();
        fs::copy(TuningCache::new(&other_dir).path(&other), &path).unwrap();
        match cache.load(&key()) {
            Err(TuningCacheError::Stale(_)) => {}
            got => panic!("{other:?}: {got:?}"),
        }
    }
    // Another version: the field rewritten, the checksum recomputed.
    let rewrite = |field: &str, value: &str| {
        let (magic, rest) = original.split_once('\n').unwrap();
        let (_, body) = rest.split_once('\n').unwrap();
        let body: String = body
            .lines()
            .map(|line| match line.split_once(' ') {
                Some((f, _)) if f == field => format!("{f} {value}\n"),
                _ => format!("{line}\n"),
            })
            .collect();
        format!("{magic}\nchecksum {:016x}\n{body}", fnv1a(body.as_bytes()))
    };
    for (field, value) in [
        ("cubecl", "0.10.0".to_owned()),
        ("kernels", "0.0.1".to_owned()),
        ("convention", "2".to_owned()),
        ("candidates", (CANDIDATE_SET_VERSION + 1).to_string()),
        ("format", (FORMAT_VERSION + 1).to_string()),
    ] {
        fs::write(&path, rewrite(field, &value)).unwrap();
        match cache.load(&key()) {
            Err(TuningCacheError::Stale(reason)) if reason.starts_with(field) => {}
            got => panic!("{field} {value}: {got:?}"),
        }
    }
    // A rewritten field with the old checksum: corrupt.
    let rewritten = rewrite("p", "9");
    let mut lines: Vec<&str> = rewritten.lines().collect();
    lines[1] = original.lines().nth(1).unwrap();
    fs::write(&path, lines.join("\n") + "\n").unwrap();
    assert!(matches!(
        cache.load(&key()),
        Err(TuningCacheError::Corrupt(_))
    ));
    // Every flipped byte, every truncation, and files that are not tuning-cache files.
    let bytes = original.as_bytes();
    let mut rejected = 0;
    for i in 0..bytes.len() {
        let mut flipped = bytes.to_vec();
        flipped[i] ^= 0x01;
        fs::write(&path, &flipped).unwrap();
        let got = cache.load(&key());
        assert!(got.is_err(), "byte {i} flipped: {got:?}");
        rejected += 1;
    }
    for len in 0..bytes.len() {
        fs::write(&path, &bytes[..len]).unwrap();
        match cache.load(&key()) {
            Err(TuningCacheError::Corrupt(_)) => rejected += 1,
            got => panic!("cut at {len} of {}: {got:?}", bytes.len()),
        }
    }
    for junk in [
        &b"NDFMMTAB"[..],
        &[0xff, 0xfe, 0x00][..],
        b"NDFMMTUN\nchecksum zz\n",
    ] {
        fs::write(&path, junk).unwrap();
        assert!(matches!(
            cache.load(&key()),
            Err(TuningCacheError::Corrupt(_))
        ));
    }
    fs::write(&path, &original).unwrap();
    assert!(cache.load(&key()).is_ok());
    println!(
        "tuning cache: other keys and versions stale; {rejected} flipped or cut files and junk \
         corrupt, none panicked"
    );
}

#[test]
fn concurrent_writers_leave_a_valid_file() {
    let dir = fresh("concurrent");
    let cache = Arc::new(TuningCache::new(&dir));
    let writers: Vec<_> = (0..8u64)
        .map(|w| {
            let cache = Arc::clone(&cache);
            std::thread::spawn(move || {
                for _ in 0..40 {
                    cache.store(&key(), &entries(w)).unwrap();
                    // A reader between the writes sees a whole file.
                    let read = cache.load(&key()).unwrap();
                    assert!((0..8).any(|t| read == as_loaded(entries(t))));
                }
            })
        })
        .collect();
    for writer in writers {
        writer.join().expect("a writer");
    }
    let read = cache.load(&key()).unwrap();
    assert!((0..8).any(|t| read == as_loaded(entries(t))));
    let names: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, vec![key().file_name()], "no temporary file left");
    println!("tuning cache: 8 concurrent writers × 40 stores leave a valid file");
}

/// FNV-1a, 64 bits: the checksum of a tuning-cache file's body.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The scenarios on the CPU runtime (module documentation).
#[cfg(feature = "cpu")]
#[test]
fn tuning_on_the_cpu_runtime() {
    use nd_fmm_exec::fmm::{Backend, FmmBuilder};
    use nd_fmm_exec::tables::M2lStrategy;
    use nd_fmm_exec::tune::CacheState;

    let universe = mpi::initialize().expect("this test owns MPI initialization");
    let comm = universe.world();
    let info = Backend::Cpu
        .probe()
        .expect("the CPU runtime is compiled in");
    eprintln!("on {info}");
    let root = fresh("scenarios");
    tune_common::check_tuning::<f64>(Backend::Cpu, 6, true, &root, &comm);
    tune_common::check_tuning::<f32>(Backend::Cpu, 3, false, &root, &comm);

    // p = 12 in f64: without `table_cache` the strategy keeps the static rule (Rotation)
    // and is not stored; with it, it is tuned.
    let points: Vec<[f64; 3]> = (0..4000)
        .map(|i| {
            let t = i as f64;
            [
                (0.37 * t).sin().abs(),
                (0.71 * t).cos().abs(),
                (0.13 * t).sin().abs(),
            ]
        })
        .collect();
    let dir = fresh("p12");
    let builder = FmmBuilder::<f64>::new(12)
        .backend(Backend::Cpu)
        .tuning_cache(&dir);
    let fmm = builder.build(&points, &points, &comm).unwrap();
    let report = fmm.device_report().unwrap().tuning.clone().unwrap();
    let strategy = report
        .decision(Decision::Strategy)
        .expect("a strategy decision");
    assert_eq!(strategy.source, Source::Static, "{report}");
    assert!(strategy.note.as_deref().unwrap().contains("table cache"));
    assert_eq!(fmm.strategy(), M2lStrategy::Rotation);
    drop(fmm);
    let tables = Path::new(env!("CARGO_TARGET_TMPDIR")).join("device_tune_tables");
    let fmm = builder
        .clone()
        .table_cache(&tables)
        .build(&points, &points, &comm)
        .unwrap();
    let report = fmm.device_report().unwrap().tuning.clone().unwrap();
    assert!(matches!(
        report.cache,
        CacheState::Loaded | CacheState::Missing
    ));
    let strategy = report.decision(Decision::Strategy).unwrap();
    assert_eq!(strategy.source, Source::Tuned, "{report}");
    eprintln!(
        "  cpu f64 p = 12: without a table cache the static rule (Rotation, not stored); with \
         one tuned: {} ({:.2} s)",
        strategy.choice,
        report.time.as_secs_f64()
    );
    eprintln!(
        "backends run: cpu (f32, f64); not run: metal (tests/device_metal.rs, by hand), cuda \
         (tests/device_cuda.rs, by hand on locust)"
    );
}

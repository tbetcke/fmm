//! The device path on Metal (Phase 4 T5, C4.1; T6, T7, T8, T9, T10): f32 only, ignored, run by hand on the
//! M3 Max outside the macOS sandbox (Metal has no adapter inside it):
//!
//! ```text
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features metal --release -- --ignored
//! ```
//!
//! Its own executable, because its one test initialises MPI (`tests/mpi_exec.rs` owns the
//! CPU-runtime repetition of every scenario). Every scenario runs at one thread through
//! `device_common::check_backend`: with every operator kind on the host fallback, bit for
//! bit against the host path for two charge vectors and a repeat, the transfers of each
//! evaluation against the formula of docs/design/device-path.md §4.1 and §7.2, nothing
//! re-uploaded in an evaluation, and every view on the device against the plan's; and
//! with the default placement (T10: every kind on the device, M2L under `Rotation` by the
//! rotation kernel), within 1e-5 of the
//! host output (relative L2), and so the multipoles and locals of every level, two
//! evaluations bit for bit, and the transfers of the formula with those kinds on the
//! device.
//! Error measures: exact equality, and the relative L2 difference from the host.
//!
//! Scenarios, f32 (Metal does no f64 arithmetic):
//! - the uniform cube, N = 2,000, sources equal to targets, p = 4: `Dense`, `Classes` and
//!   `Rotation` with gradients, and `Dense` without;
//! - the uniform cube, N = 20,000, p = 8, gradients: M2M, L2L and M2L with the GEMM of
//!   `DeviceGemm::Auto` (M2M and L2L on the library where it takes a level's shape, M2L
//!   hand-written), `DeviceGemm::Library` (M2L on the library too) and
//!   `DeviceGemm::HandWritten`, each level call's GEMM printed;
//! - an adaptive tree with W and X lists (a cloud and a dense blob), p = 3, eight points
//!   per leaf, gradients off and on;
//! - sources and targets disjoint by the parity of their level-2 cell, so that leaves
//!   hold sources only or targets only (empty target and source leaves), p = 4;
//! - coincident points: 40 positions with five copies each in a cloud, p = 4;
//! - f64 with Metal is refused with `SettingsError::PrecisionUnsupported` at build;
//! - `threads(4)` builds the pool of four threads for the host-fallback kinds
//!   (device-path.md §11): with every kind there the output still equals the host
//!   path's, and with the default placement the output of one thread;
//! - the tuner (T12): the scenarios of `tune_common` (static rule without a directory, a
//!   slowed candidate, the input-precision guard, the cache round trip, stale and
//!   corrupted files, every candidate's output within the FMM bounds, the budget) at p = 8
//!   in f32, where the library GEMM and its coefficient-major layout are candidates, and
//!   at p = 3 without the stale files and the budget.
//!
//! The test prints the device (`Backend::probe`) and the backends it ran.
#![cfg(feature = "metal")]

use mpi::Threading;
use mpi::traits::*;
use nd_fmm_exec::device::Gemm;
use nd_fmm_exec::fmm::{Backend, DeviceGemm, FmmBuilder, FmmError, OperatorKind, SettingsError};
use nd_fmm_exec::tables::M2lStrategy;

mod device_common;
mod tune_common;

use device_common::{Outcome, check_backend, output_bits};

/// SplitMix64, as in the other tests.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }
}

/// `n` points uniform in the unit cube.
fn cube(rng: &mut SplitMix64, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| core::array::from_fn(|_| rng.range(0.0, 1.0)))
        .collect()
}

/// `n` charges uniform in [−1, 1), in f32.
fn charges(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.range(-1.0, 1.0) as f32).collect()
}

/// One scenario: the host path at one thread, then Metal through `check_backend`.
fn scenario(
    name: &str,
    builder: FmmBuilder<f32>,
    sources: &[[f64; 3]],
    targets: &[[f64; 3]],
    charges: &[f32],
    comm: &mpi::topology::SimpleCommunicator,
) {
    let mut host = builder
        .clone()
        .build(sources, targets, comm)
        .unwrap_or_else(|error| panic!("{name}: the host FMM does not build: {error}"));
    let output = host.evaluate(charges).expect("the host FMM evaluates");
    let (outcome, difference) = check_backend(
        &builder,
        (sources, targets),
        charges,
        &mut host,
        &output,
        Backend::Metal,
        comm,
    );
    eprintln!(
        "rank {}: {name}: {}",
        comm.rank(),
        match outcome {
            Outcome::Ran => format!(
                "metal on the host fallback bit for bit ({} values), transfers as the \
                 formula; the default placement within {:.1e} (φ) and {:.1e} \
                 (∇φ) of the host, relative L2; per level multipoles {:.1e} (root, the \
                 global M2M: {:.1e}), locals {:.1e}",
                output_bits(&output).len(),
                difference.potential,
                difference.gradient,
                difference.multipoles,
                difference.root,
                difference.locals
            ),
            Outcome::OneRankOnly => "DeviceNeedsOneRank on every rank".to_owned(),
        }
    );
}

#[test]
#[ignore = "Metal: run by hand on the M3 Max, outside the macOS sandbox"]
fn metal_device_path() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();
    let info = Backend::Metal
        .probe()
        .unwrap_or_else(|reason| panic!("Metal: {reason} (run outside the macOS sandbox)"));
    eprintln!("rank {}: on {info}", comm.rank());

    let mut rng = SplitMix64(0x3e7a1);
    let points = cube(&mut rng, 2000);
    let q = charges(&mut rng, points.len());
    for (strategy, gradients) in [
        (M2lStrategy::Dense, true),
        (M2lStrategy::Dense, false),
        (M2lStrategy::Classes, true),
        (M2lStrategy::Rotation, true),
    ] {
        let builder = FmmBuilder::<f32>::new(4)
            .strategy(strategy)
            .gradients(gradients);
        scenario(
            &format!("uniform cube, {strategy:?}, gradients {gradients}"),
            builder,
            &points,
            &points,
            &q,
            &comm,
        );
    }

    // p = 8 (T8, T9): under `DeviceGemm::Auto` M2M and L2L take the library GEMM where it
    // accepts a level's shape and M2L runs the hand-written kernel; under
    // `DeviceGemm::Library` M2L takes the library too (the report says where); and the
    // hand-written kernel everywhere (`DeviceGemm::HandWritten`).
    let large = cube(&mut rng, 20_000);
    let q8 = charges(&mut rng, large.len());
    for gemm in [
        DeviceGemm::Auto,
        DeviceGemm::Library,
        DeviceGemm::HandWritten,
    ] {
        let builder = FmmBuilder::<f32>::new(8).gradients(true).device_gemm(gemm);
        scenario(
            &format!("uniform cube, N = 20,000, p = 8, {gemm:?} GEMM"),
            builder.clone(),
            &large,
            &large,
            &q8,
            &comm,
        );
        if comm.size() == 1 {
            let fmm = builder
                .backend(Backend::Metal)
                .build(&large, &large, &comm)
                .unwrap();
            let report = fmm.device_report().unwrap();
            for t in &report.translations {
                eprintln!(
                    "rank 0:   {} {:?} level {}: {} pairs, {}, {} chunk(s){}",
                    t.kind,
                    t.pass,
                    t.level,
                    t.pairs,
                    t.gemm,
                    t.chunks,
                    t.library_rejection
                        .as_deref()
                        .map_or(String::new(), |r| format!("; library rejected: {r}"))
                );
            }
            let m2l_library = report
                .translations_of(OperatorKind::M2l)
                .filter(|t| t.gemm == Gemm::Library)
                .count();
            match gemm {
                DeviceGemm::Library => assert!(m2l_library > 0, "{gemm:?}: no M2L on the library"),
                _ => assert_eq!(m2l_library, 0, "{gemm:?}: M2L on the library"),
            }
        }
    }

    // A cloud and a dense blob near a corner: leaves on several levels, W and X lists.
    let mut adaptive = cube(&mut rng, 200);
    adaptive
        .extend((0..200).map(|_| -> [f64; 3] { core::array::from_fn(|_| rng.range(0.02, 0.1)) }));
    let q = charges(&mut rng, adaptive.len());
    for gradients in [false, true] {
        let builder = FmmBuilder::<f32>::new(3)
            .gradients(gradients)
            .max_points_per_leaf(8);
        let fmm = builder.build(&adaptive, &adaptive, &comm).unwrap();
        let sizes = fmm.list_sizes();
        assert!(sizes.w > 0 && sizes.x > 0, "W and X lists: {sizes:?}");
        drop(fmm);
        scenario(
            &format!("adaptive tree with W and X, gradients {gradients}"),
            builder,
            &adaptive,
            &adaptive,
            &q,
            &comm,
        );
    }

    // Sources and targets by the parity of their level-2 cell.
    let mixed = cube(&mut rng, 1200);
    let parity = |x: &[f64; 3]| x.iter().map(|&c| (c * 4.0) as usize).sum::<usize>() % 2;
    let (sources, targets): (Vec<[f64; 3]>, Vec<[f64; 3]>) =
        mixed.into_iter().partition(|x| parity(x) == 0);
    let q = charges(&mut rng, sources.len());
    scenario(
        "sources-only next to targets-only leaves",
        FmmBuilder::<f32>::new(4)
            .gradients(true)
            .max_points_per_leaf(16),
        &sources,
        &targets,
        &q,
        &comm,
    );

    // Coincident points: 40 positions with five copies each, in a cloud.
    let mut coincident = cube(&mut rng, 600);
    for x in cube(&mut rng, 40) {
        coincident.extend(std::iter::repeat_n(x, 5));
    }
    let q = charges(&mut rng, coincident.len());
    scenario(
        "coincident points",
        FmmBuilder::<f32>::new(4)
            .gradients(true)
            .max_points_per_leaf(16),
        &coincident,
        &coincident,
        &q,
        &comm,
    );

    // f64 is refused at build, on every rank.
    let error = FmmBuilder::<f64>::new(4)
        .backend(Backend::Metal)
        .build(&points, &points, &comm)
        .err();
    assert_eq!(
        error,
        Some(FmmError::InvalidSettings(
            SettingsError::PrecisionUnsupported {
                backend: Backend::Metal
            }
        ))
    );
    eprintln!("rank {}: f64 with Metal: {}", comm.rank(), error.unwrap());

    // threads(4): the pool serves the host-fallback kinds.
    if comm.size() == 1 {
        let builder = FmmBuilder::<f32>::new(4).gradients(true);
        let q = charges(&mut rng, points.len());
        let mut host = builder.build(&points, &points, &comm).unwrap();
        let want = host.evaluate(&q).unwrap();
        let mut metal = builder
            .clone()
            .backend(Backend::Metal)
            .threads(4)
            .host_fallback(OperatorKind::ALL)
            .build(&points, &points, &comm)
            .unwrap();
        assert_eq!(metal.threading().threads, 4);
        assert_eq!(metal.operator().threads(), 4);
        assert_eq!(metal.device_report().unwrap().cpu_units, None);
        let got = metal.evaluate(&q).unwrap();
        assert_eq!(output_bits(&got), output_bits(&want), "metal at 4 threads");
        // With the default placement (T6, T7) the pool serves the host-fallback kinds,
        // and the output is that of one thread bit for bit.
        let device = |threads| {
            builder
                .clone()
                .backend(Backend::Metal)
                .threads(threads)
                .build(&points, &points, &comm)
                .unwrap()
                .evaluate(&q)
                .unwrap()
        };
        assert_eq!(
            output_bits(&device(4)),
            output_bits(&device(1)),
            "metal with the default placement at 4 threads"
        );
        eprintln!(
            "rank 0: metal with threads(4): a pool of 4 for the fallback, bit for bit; with \
             the default placement bit for bit one thread"
        );
    }
    // The tuner (T12), on one rank.
    if comm.size() == 1 {
        let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("device_metal_tune");
        tune_common::check_tuning::<f32>(Backend::Metal, 8, true, &root, &comm);
        tune_common::check_tuning::<f32>(Backend::Metal, 3, false, &root, &comm);
    }

    eprintln!(
        "rank {}: backends run: metal (f32); not run: cpu (tests/mpi_exec.rs with --features \
         cpu), cuda (type-checked, not run)",
        comm.rank()
    );
}

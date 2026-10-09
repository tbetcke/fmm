//! The device path on CUDA (Phase 4S T4, C4S.4): the scenarios of `tests/device_metal.rs`
//! in f32 and f64, ignored, run by hand on locust's H100 (docs/phase4s/README.md):
//!
//! ```text
//! tools/gh200/sync.sh
//! tools/gh200/remote.sh 'RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release -- --ignored'
//! ```
//!
//! Its own executable, because its one test initialises MPI. Every scenario runs at one
//! thread through `device_common::check_backend`, as on Metal: with every operator kind on
//! the host fallback, bit for bit against the host path for two charge vectors and a
//! repeat, the transfers of each evaluation against the formula of
//! docs/design/device-path.md §4.1 and §7.2, nothing re-uploaded in an evaluation, and
//! every view on the device against the plan's; and with the default placement (every
//! kind on the device), within the FMM bounds of the host output (relative L2 1e-5 in f32,
//! 1e-12 in f64), and so the multipoles and locals of every level, two evaluations and
//! two builds bit for bit, the transfers of the formula with those kinds on the device,
//! and a build with `device_timestamps(true)` bit for bit the default with five timing
//! windows and no sync of their own (CUDA times by events, F30).
//! Error measures: exact equality, and the relative L2 difference from the host.
//!
//! Scenarios, each in f32 and in f64 (CUDA does both, F27):
//! - the uniform cube, N = 2,000, sources equal to targets, p = 4: `Dense`, `Classes` and
//!   `Rotation` with gradients, and `Dense` without;
//! - the uniform cube, N = 20,000, p = 8, gradients, under `DeviceGemm::Auto`,
//!   `DeviceGemm::Library` and `DeviceGemm::HandWritten`, each level call's GEMM printed.
//!   Where Metal runs M2M and L2L (and under `Library` M2L) on the library GEMM, CUDA runs
//!   the hand-written kernel at every level call: the library's probe fails on the LLVM
//!   NVPTX path (device-path.md §18.1, F28). The test asserts that no level call runs the
//!   library, that every level call the static rule gives the library (f32, p = 8: M2M
//!   and L2L under `Auto`, every kind under `Library`) reports why it was rejected, and
//!   that no other level call does;
//! - an adaptive tree with W and X lists (a cloud and a dense blob), p = 3, eight points
//!   per leaf, gradients off and on;
//! - sources and targets disjoint by the parity of their level-2 cell, so that leaves
//!   hold sources only or targets only (empty target and source leaves), p = 4;
//! - coincident points: 40 positions with five copies each in a cloud, p = 4;
//! - the output pass (Phase 4S T9): in every scenario `check_backend` checks the output
//!   against the pass before T9 bit for bit, with the pass on the device (the default on
//!   CUDA) and on the host (`output_pass(Host)`); also the uniform cube at p = 6 with
//!   gradients, and `output_pass` `Auto` and `Device` on the device, `Host` on the host;
//! - f64 is accepted on CUDA (Metal refuses it with `PrecisionUnsupported`): the build
//!   succeeds and the device reports f64 arithmetic;
//! - `threads(4)` builds the pool of four threads for the host-fallback kinds
//!   (device-path.md §11): with every kind there the output still equals the host
//!   path's, and with the default placement the output of one thread;
//! - the tuner (Phase 4 T12): the scenarios of `tune_common` (static rule without a
//!   directory, a slowed candidate, the input-precision guard, the cache round trip, stale
//!   and corrupted files, every candidate's output within the FMM bounds, the budget) at
//!   f32 p = 8 and f64 p = 6 in full, and at f32 p = 3 without the stale files and the
//!   budget, as on Metal, and the strategy at f64 p = 12 of `tests/device_tune.rs` (the
//!   static rule without a table cache, `Dense` on CUDA since Phase 4S decision 9, tuned
//!   with one). On CUDA the library GEMM is never
//!   registered (its probe fails), so the hand-written layouts, the chunk budget and the
//!   P2P layouts are the candidates;
//! - the stage windows of `device_timestamps(true)` (device-path.md §8.3), the uniform cube
//!   at N = 10⁵, p = 8, gradients, in f32 and f64: after two warm-up evaluations, ten
//!   evaluations, each with five windows and one sync and the output of the first bit for
//!   bit, and per evaluation the host's wall time of `evaluate`, the sum of the five
//!   device spans and each span printed. Whether the windows overlap is read from the
//!   numbers (on Metal, T11, the spans added up to more than the evaluation's wall time),
//!   never asserted: timings are reported, not asserted;
//! - per-kind timings (Phase 4S T5): in every scenario `check_backend` also builds the
//!   default placement with `kind_timings(Synchronous)` and `kind_timings(Device)` (alone
//!   and inside the stage windows), each bit for bit the default with the syncs and
//!   windows its mode documents (`tests/kind_common`); and the call windows of
//!   `kind_timings(Device)` against the stage windows on the same cube in f32 and f64,
//!   printed (`tests/kind_common/windows.rs`).
//!
//! The test prints the device (`Backend::probe`) and the backends it ran.
#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::device::{Gemm, StageTiming};
use nd_fmm_exec::fmm::{
    Backend, DeviceGemm, DeviceStage, FmmBuilder, OperatorKind, OutputPass, Placement,
};
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_exec::tune::{CacheState, Decision, Source, library_applies};
use nd_fmm_kernels::Precision;

mod device_common;
mod kind_common;
#[path = "kind_common/windows.rs"]
mod kind_windows;
mod tune_common;

use device_common::{check_backend, output_bits};
use tune_common::Real;

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

/// `n` charges uniform in [−1, 1), in T.
fn charges<T: Real>(rng: &mut SplitMix64, n: usize) -> Vec<T> {
    (0..n).map(|_| T::from_f64(rng.range(-1.0, 1.0))).collect()
}

/// The precision of T.
fn precision<T>() -> Precision {
    if size_of::<T>() == 4 {
        Precision::F32
    } else {
        Precision::F64
    }
}

/// One scenario: the host path at one thread, then CUDA through `check_backend`.
fn scenario<T: Real>(
    name: &str,
    builder: FmmBuilder<T>,
    sources: &[[f64; 3]],
    targets: &[[f64; 3]],
    charges: &[T],
    comm: &SimpleCommunicator,
) {
    let name = format!("{} {name}", precision::<T>());
    let mut host = builder
        .clone()
        .build(sources, targets, comm)
        .unwrap_or_else(|error| panic!("{name}: the host FMM does not build: {error}"));
    let output = host.evaluate(charges).expect("the host FMM evaluates");
    let difference = check_backend(
        &builder,
        (sources, targets),
        charges,
        &mut host,
        &output,
        Backend::Cuda,
        comm,
    );
    let summary = format!(
        "cuda on the host fallback bit for bit ({} values), transfers as the \
         formula; the default placement within {:.1e} (φ) and {:.1e} \
         (∇φ) of the host, relative L2; per level multipoles {:.1e} (root, the \
         global M2M: {:.1e}), locals {:.1e}",
        output_bits(&output).len(),
        difference.potential,
        difference.gradient,
        difference.multipoles,
        difference.root,
        difference.locals
    );
    eprintln!("rank {}: {name}: {summary}", comm.rank());
}

/// The scenarios of the module documentation in T, up to the tuner.
fn scenarios<T: Real>(rng: &mut SplitMix64, comm: &SimpleCommunicator) {
    let precision = precision::<T>();
    let points = cube(rng, 2000);
    let q = charges::<T>(rng, points.len());
    for (strategy, gradients) in [
        (M2lStrategy::Dense, true),
        (M2lStrategy::Dense, false),
        (M2lStrategy::Classes, true),
        (M2lStrategy::Rotation, true),
    ] {
        let builder = FmmBuilder::<T>::new(4)
            .strategy(strategy)
            .gradients(gradients);
        scenario(
            &format!("uniform cube, {strategy:?}, gradients {gradients}"),
            builder,
            &points,
            &points,
            &q,
            comm,
        );
    }

    // The output pass (Phase 4S T9): on the device by default (CUDA does f64), on the host
    // with `output_pass(Host)`, bit for bit each other and the pass before T9
    // (`check_backend`, at p = 4 and 8 above and below), here also at p = 6 with
    // gradients.
    scenario(
        "uniform cube, p = 6, gradients",
        FmmBuilder::<T>::new(6).gradients(true),
        &points,
        &points,
        &q,
        comm,
    );
    if comm.size() == 1 {
        for (pass, want) in [
            (OutputPass::Auto, Placement::Device),
            (OutputPass::Device, Placement::Device),
            (OutputPass::Host, Placement::Host),
        ] {
            let fmm = FmmBuilder::<T>::new(4)
                .backend(Backend::Cuda)
                .output_pass(pass)
                .build(&points, &points, comm)
                .unwrap();
            assert_eq!(fmm.output_pass(), want, "output_pass({pass:?}) on CUDA");
        }
    }

    // p = 8: every GEMM setting runs the hand-written kernel on CUDA (F28); the level
    // calls the static rule gives the library report why the library was rejected.
    let large = cube(rng, 20_000);
    let q8 = charges::<T>(rng, large.len());
    for gemm in [
        DeviceGemm::Auto,
        DeviceGemm::Library,
        DeviceGemm::HandWritten,
    ] {
        let builder = FmmBuilder::<T>::new(8).gradients(true).device_gemm(gemm);
        scenario(
            &format!("uniform cube, N = 20,000, p = 8, {gemm:?} GEMM"),
            builder.clone(),
            &large,
            &large,
            &q8,
            comm,
        );
        if comm.size() == 1 {
            let fmm = builder
                .backend(Backend::Cuda)
                .build(&large, &large, comm)
                .unwrap();
            let report = fmm.device_report().unwrap();
            let n = (8 + 1) * (8 + 1);
            for t in &report.translations {
                eprintln!(
                    "rank 0:   {precision} {} {:?} level {}: {} pairs, {}, {} chunk(s){}",
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
                assert_ne!(
                    t.gemm,
                    Gemm::Library,
                    "{precision} {gemm:?}: {} level {} on the library on CUDA",
                    t.kind,
                    t.level
                );
                let library = library_applies(report.info.backend, precision, n)
                    && match gemm {
                        DeviceGemm::Auto => t.kind != OperatorKind::M2l,
                        DeviceGemm::Library => true,
                        DeviceGemm::HandWritten => false,
                    };
                assert_eq!(
                    t.library_rejection.is_some(),
                    library,
                    "{precision} {gemm:?}: {} level {}: the library {} the static rule's \
                     choice, rejection {:?}",
                    t.kind,
                    t.level,
                    if library { "is" } else { "is not" },
                    t.library_rejection
                );
            }
            eprintln!(
                "rank 0: {precision} {gemm:?}: every level call on the hand-written GEMM \
                 (the library's probe fails on CUDA, F28)"
            );
        }
    }

    // A cloud and a dense blob near a corner: leaves on several levels, W and X lists.
    let mut adaptive = cube(rng, 200);
    adaptive
        .extend((0..200).map(|_| -> [f64; 3] { core::array::from_fn(|_| rng.range(0.02, 0.1)) }));
    let q = charges::<T>(rng, adaptive.len());
    for gradients in [false, true] {
        let builder = FmmBuilder::<T>::new(3)
            .gradients(gradients)
            .max_points_per_leaf(8);
        let fmm = builder.build(&adaptive, &adaptive, comm).unwrap();
        let sizes = fmm.list_sizes();
        assert!(sizes.w > 0 && sizes.x > 0, "W and X lists: {sizes:?}");
        drop(fmm);
        scenario(
            &format!("adaptive tree with W and X, gradients {gradients}"),
            builder,
            &adaptive,
            &adaptive,
            &q,
            comm,
        );
    }

    // Sources and targets by the parity of their level-2 cell.
    let mixed = cube(rng, 1200);
    let parity = |x: &[f64; 3]| x.iter().map(|&c| (c * 4.0) as usize).sum::<usize>() % 2;
    let (sources, targets): (Vec<[f64; 3]>, Vec<[f64; 3]>) =
        mixed.into_iter().partition(|x| parity(x) == 0);
    let q = charges::<T>(rng, sources.len());
    scenario(
        "sources-only next to targets-only leaves",
        FmmBuilder::<T>::new(4)
            .gradients(true)
            .max_points_per_leaf(16),
        &sources,
        &targets,
        &q,
        comm,
    );

    // Coincident points: 40 positions with five copies each, in a cloud.
    let mut coincident = cube(rng, 600);
    for x in cube(rng, 40) {
        coincident.extend(std::iter::repeat_n(x, 5));
    }
    let q = charges::<T>(rng, coincident.len());
    scenario(
        "coincident points",
        FmmBuilder::<T>::new(4)
            .gradients(true)
            .max_points_per_leaf(16),
        &coincident,
        &coincident,
        &q,
        comm,
    );

    // threads(4): the pool serves the host-fallback kinds.
    if comm.size() == 1 {
        let builder = FmmBuilder::<T>::new(4).gradients(true);
        let q = charges::<T>(rng, points.len());
        let mut host = builder.build(&points, &points, comm).unwrap();
        let want = host.evaluate(&q).unwrap();
        let mut cuda = builder
            .clone()
            .backend(Backend::Cuda)
            .threads(4)
            .host_fallback(OperatorKind::ALL)
            .build(&points, &points, comm)
            .unwrap();
        assert_eq!(cuda.threading().threads, 4);
        assert_eq!(cuda.operator().threads(), 4);
        assert_eq!(cuda.device_report().unwrap().cpu_units, None);
        let got = cuda.evaluate(&q).unwrap();
        assert_eq!(
            output_bits(&got),
            output_bits(&want),
            "{precision}: cuda at 4 threads"
        );
        let device = |threads| {
            builder
                .clone()
                .backend(Backend::Cuda)
                .threads(threads)
                .build(&points, &points, comm)
                .unwrap()
                .evaluate(&q)
                .unwrap()
        };
        assert_eq!(
            output_bits(&device(4)),
            output_bits(&device(1)),
            "{precision}: cuda with the default placement at 4 threads"
        );
        eprintln!(
            "rank 0: {precision} cuda with threads(4): a pool of 4 for the fallback, bit for \
             bit; with the default placement bit for bit one thread"
        );
    }
}

/// The strategy at f64 p = 12, as `tests/device_tune.rs` checks it on the CPU runtime:
/// without `table_cache` it keeps the static rule and is not stored (the dense candidate
/// would take seconds to build; on CUDA the static rule is `Dense` at every p, Phase 4S
/// decision 9, so that build makes the dense tables once), and with a table cache it is
/// tuned. The tuning cache goes to `dir`, emptied first.
fn strategy_at_12(dir: &Path, comm: &SimpleCommunicator) {
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
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).expect("the test directory is created");
    let builder = FmmBuilder::<f64>::new(12)
        .backend(Backend::Cuda)
        .tuning_cache(dir);
    let fmm = builder.build(&points, &points, comm).unwrap();
    let report = fmm.device_report().unwrap().tuning.clone().unwrap();
    let strategy = report
        .decision(Decision::Strategy)
        .expect("a strategy decision");
    assert_eq!(strategy.source, Source::Static, "{report}");
    assert!(strategy.note.as_deref().unwrap().contains("table cache"));
    assert_eq!(fmm.strategy(), M2lStrategy::Dense, "Phase 4S decision 9");
    drop(fmm);
    let tables = Path::new(env!("CARGO_TARGET_TMPDIR")).join("device_cuda_tables");
    let fmm = builder
        .table_cache(&tables)
        .build(&points, &points, comm)
        .unwrap();
    let report = fmm.device_report().unwrap().tuning.clone().unwrap();
    assert!(matches!(
        report.cache,
        CacheState::Loaded | CacheState::Missing
    ));
    let strategy = report.decision(Decision::Strategy).unwrap();
    assert_eq!(strategy.source, Source::Tuned, "{report}");
    eprintln!(
        "  cuda f64 p = 12: without a table cache the static rule (Dense, not stored); \
         with one tuned: {} ({:.2} s)",
        strategy.choice,
        report.time.as_secs_f64()
    );
}

/// The stage windows of `device_timestamps(true)` in T (module documentation): asserts
/// the windows, the one sync and the bits; prints the times.
fn stage_windows<T: Real>(rng: &mut SplitMix64, comm: &SimpleCommunicator) {
    let precision = precision::<T>();
    let points = cube(rng, 100_000);
    let q = charges::<T>(rng, points.len());
    let builder = FmmBuilder::<T>::new(8)
        .gradients(true)
        .backend(Backend::Cuda);
    let first = builder
        .clone()
        .build(&points, &points, comm)
        .unwrap()
        .evaluate(&q)
        .unwrap();
    let mut fmm = builder
        .device_timestamps(true)
        .build(&points, &points, comm)
        .unwrap();
    assert_eq!(
        fmm.device_report().unwrap().stage_timing,
        StageTiming::DeviceTimestamps,
        "{precision}: the stage timing"
    );
    for _ in 0..2 {
        fmm.evaluate(&q).unwrap();
    }
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut overlapping = 0;
    let mut ratios = Vec::new();
    eprintln!(
        "rank 0: {precision} device_timestamps(true), uniform cube N = 100,000, p = 8 \
         ({:?}), in ms: wall of evaluate | sum of the spans | {}",
        fmm.strategy(),
        DeviceStage::ALL.map(|s| format!("{s:?}")).join(" | ")
    );
    const EVALUATIONS: usize = 10;
    for _ in 0..EVALUATIONS {
        let start = Instant::now();
        let output = fmm.evaluate(&q).unwrap();
        let wall = start.elapsed();
        assert_eq!(
            output_bits(&output),
            output_bits(&first),
            "{precision}: device_timestamps(true) changes the output"
        );
        let counters = fmm.device_counters().unwrap().evaluation;
        assert_eq!(
            (counters.windows, counters.syncs),
            (DeviceStage::ALL.len() as u64, 1),
            "{precision}: windows and syncs of an evaluation"
        );
        let device = output.timings.device.expect("device stage times");
        let sum = device.total();
        if sum > wall {
            overlapping += 1;
        }
        ratios.push(sum.as_secs_f64() / wall.as_secs_f64());
        eprintln!(
            "rank 0:   {:.3} | {:.3} | {}",
            ms(wall),
            ms(sum),
            DeviceStage::ALL
                .map(|s| format!("{:.3}", ms(device.get(s))))
                .join(" | ")
        );
    }
    ratios.sort_by(f64::total_cmp);
    eprintln!(
        "rank 0: {precision} stage windows: five windows and one sync per evaluation, the \
         output bit for bit the default's; the spans add up to {:.3}–{:.3} of the wall time \
         of evaluate (median {:.3}); in {overlapping} of {EVALUATIONS} evaluations more than \
         the wall time",
        ratios[0],
        ratios[EVALUATIONS - 1],
        ratios[EVALUATIONS / 2]
    );
}

#[test]
#[ignore = "CUDA: run by hand on locust"]
fn cuda_device_path() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();
    let info = Backend::Cuda
        .probe()
        .unwrap_or_else(|reason| panic!("CUDA: {reason} (run on locust)"));
    eprintln!("rank {}: on {info}", comm.rank());

    // f64 is accepted on CUDA, on every rank (Metal refuses it).
    assert!(info.supports(Precision::F64), "CUDA does f64 arithmetic");
    let mut rng = SplitMix64(0x3e7a1);
    let points = cube(&mut rng, 2000);
    let built = FmmBuilder::<f64>::new(4)
        .backend(Backend::Cuda)
        .build(&points, &points, &comm);
    match (comm.size(), built) {
        (1, Ok(fmm)) => {
            let report = fmm.device_report().expect("a device backend");
            assert!(report.info.supports(Precision::F64));
            eprintln!("rank 0: f64 with CUDA: accepted ({})", report.info);
        }
        (1, Err(error)) => panic!("f64 with CUDA is refused: {error}"),
        (_, result) => eprintln!(
            "rank {}: f64 with CUDA on {} ranks: {}",
            comm.rank(),
            comm.size(),
            result.err().map_or("built".to_owned(), |e| e.to_string())
        ),
    }

    scenarios::<f32>(&mut rng, &comm);
    scenarios::<f64>(&mut rng, &comm);

    if comm.size() == 1 {
        // The tuner (Phase 4 T12).
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("device_cuda_tune");
        tune_common::check_tuning::<f32>(Backend::Cuda, 8, true, &root, &comm);
        tune_common::check_tuning::<f32>(Backend::Cuda, 3, false, &root, &comm);
        tune_common::check_tuning::<f64>(Backend::Cuda, 6, true, &root, &comm);
        strategy_at_12(&root.join("p12"), &comm);
        // The stage windows.
        stage_windows::<f32>(&mut rng, &comm);
        stage_windows::<f64>(&mut rng, &comm);
        // The call windows of `KindTiming::Device` against them (Phase 4S T5).
        let points = cube(&mut rng, 100_000);
        let q32 = charges::<f32>(&mut rng, points.len());
        kind_windows::call_windows::<f32>(Backend::Cuda, (&points, &q32), 8, &comm);
        let q64 = charges::<f64>(&mut rng, points.len());
        kind_windows::call_windows::<f64>(Backend::Cuda, (&points, &q64), 8, &comm);
    }
    eprintln!("rank {}: {}", comm.rank(), kind_common::summary());

    eprintln!(
        "rank {}: backends run: cuda (f32, f64); not run: cpu (tests/mpi_exec.rs with \
         --features cpu), metal (tests/device_metal.rs, by hand on the M3 Max)",
        comm.rank()
    );
}

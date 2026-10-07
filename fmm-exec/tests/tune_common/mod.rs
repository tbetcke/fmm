//! The tuner of the device path (Phase 4 T12, C4.7) through `FmmBuilder::build` on one
//! backend, shared by `tests/device_tune.rs` (the CPU runtime), `tests/device_metal.rs`
//! (Metal, ignored) and `tests/device_cuda.rs` (CUDA, ignored; Phase 4S T4). The tests of
//! the brief, docs/phase4/T12-autotune.md, "Tests that define done", that need a device:
//!
//! - **the static rule without a directory**: no tuning, no file, every decision from the
//!   static rule, and the static strategy, GEMMs and P2P layout in the device report;
//! - **a slow candidate**: with a hook that adds a second to the static rule's candidate
//!   of every decision, every decision with two or more timed candidates chooses another;
//!   the choices are reported, and fixed across evaluations (two evaluations bit for bit,
//!   the report unchanged);
//! - **the input-precision guard**: a library strategy with TF32 inputs, offered to every
//!   GEMM and strategy decision by the hook and made the fastest by it, is never
//!   registered (its timing says why), never timed and never chosen;
//! - **the cache round trip**: a second and a third build from the directory take every
//!   tuned decision from the cache with the same choice, and all three outputs are equal
//!   bit for bit;
//! - **stale and corrupted files**, through a build: a file written for another key
//!   (another p), or with another CubeCL version or convention version (checksum
//!   recomputed), is rejected as stale and replaced by a new tuning; a file with a flipped
//!   byte or cut in half is rejected as corrupt, without a panic, and replaced;
//! - **every tuned choice within the FMM bounds**: with a hook that makes the k-th
//!   registered candidate of every decision the fastest, for every k, the output is
//!   within the FMM bounds of the host output of the same strategy (docs/phase4/README.md,
//!   "Accuracy measures": relative L2 1e-12 in f64, 1e-5 in f32, φ and ∇φ);
//! - **the budget**: with a budget of 0.3 s and a hook that sleeps 0.1 s per candidate, no
//!   candidate starts after the deadline, the cut decision keeps the candidates it timed
//!   (the static rule's first) and is marked, the decisions after it keep the static rule
//!   and are not stored; a second build with time tunes those and takes the others from
//!   the cache.
//!
//! The tests use the hook, never real speed differences (nothing about timing is
//! asserted but the deadline's order). The problem: N = 20,000 points uniform in the unit
//! cube, sources equal to targets, 64 points per leaf (a uniform tree to level 3), gradients
//! on; its level calls of ≥ 512 pairs are tuned (M2L on levels 2 and 3, M2M on level 2, L2L
//! on level 3).
//!
//! Error measures: exact equality (bit patterns, choices), and the relative L2 difference
//! from the host output.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use nd_fmm_exec::fmm::{Backend, Fmm, FmmBuilder, Output};
use nd_fmm_exec::fmm::{DeviceGemm, OperatorKind};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_exec::tune::{
    CacheState, Candidate, Decision, GemmChoice, GemmKind, InputPrecision, Source, Timing,
    TuningCache, TuningHook, TuningKey, TuningReport, gemm_candidates, p2p_candidates, static_gemm,
    static_p2p, static_strategy,
};
use nd_fmm_kernels::{DeviceInfo, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;

/// The float types of the scenarios.
pub trait Real: Stored + SimdScalar + Equivalence + Default + RealScalar {}

impl Real for f32 {}

impl Real for f64 {}

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

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }
}

/// The problem of the module documentation, and its charges in T.
struct Problem<T> {
    points: Vec<[f64; 3]>,
    charges: Vec<T>,
}

impl<T: Real> Problem<T> {
    fn new(n: usize) -> Self {
        let mut rng = SplitMix64(0x7e12_0001);
        let points = (0..n)
            .map(|_| core::array::from_fn(|_| rng.range(0.0, 1.0)))
            .collect();
        let charges = (0..n)
            .map(|_| <T as RealScalar>::from_f64(rng.range(-1.0, 1.0)))
            .collect();
        Self { points, charges }
    }
}

/// The precision of T.
fn precision<T>() -> Precision {
    if size_of::<T>() == 4 {
        Precision::F32
    } else {
        Precision::F64
    }
}

/// A fresh, empty directory `root/name`.
fn fresh(root: &Path, name: &str) -> PathBuf {
    let dir = root.join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("the test directory is created");
    dir
}

/// The tuning report of a device FMM.
fn tuning<T: Real>(fmm: &Fmm<'_, T>) -> TuningReport {
    fmm.device_report()
        .expect("a device backend")
        .tuning
        .clone()
        .expect("the build ended its tuning")
}

/// The static rule's candidate of `decision` on `info` (the first of its candidates).
fn static_candidate(
    info: &DeviceInfo,
    precision: Precision,
    p: usize,
    decision: Decision,
) -> Candidate {
    let n = (p + 1) * (p + 1);
    match decision {
        Decision::Strategy => match static_strategy(precision, p) {
            M2lStrategy::Rotation => Candidate::Rotation,
            _ => Candidate::Dense(static_gemm(
                info,
                precision,
                n,
                OperatorKind::M2l,
                DeviceGemm::Auto,
                None,
            )),
        },
        Decision::Gemm { kind, .. } => Candidate::Gemm(static_gemm(
            info,
            precision,
            n,
            kind,
            DeviceGemm::Auto,
            None,
        )),
        Decision::P2p { .. } => Candidate::P2p(static_p2p(info)),
    }
}

/// The candidates of `decision` in registration order (before the hook's offers).
fn candidates(
    info: &DeviceInfo,
    precision: Precision,
    p: usize,
    decision: Decision,
) -> Vec<Candidate> {
    let n = (p + 1) * (p + 1);
    match decision {
        Decision::Strategy => {
            let dense = gemm_candidates(info, precision, n, OperatorKind::M2l, false)
                .into_iter()
                .map(Candidate::Dense);
            if static_strategy(precision, p) == M2lStrategy::Rotation {
                std::iter::once(Candidate::Rotation).chain(dense).collect()
            } else {
                dense.chain([Candidate::Rotation]).collect()
            }
        }
        Decision::Gemm { kind, .. } => gemm_candidates(info, precision, n, kind, true)
            .into_iter()
            .map(Candidate::Gemm)
            .collect(),
        Decision::P2p { .. } => p2p_candidates(info)
            .into_iter()
            .map(Candidate::P2p)
            .collect(),
    }
}

/// The measured candidates of a decision.
fn measured(report: &nd_fmm_exec::tune::DecisionReport) -> Vec<Candidate> {
    report
        .times
        .iter()
        .filter(|(_, t)| matches!(t, Timing::Measured(_)))
        .map(|(c, _)| *c)
        .collect()
}

/// The host output of the same settings under `strategy`, memoised per strategy.
struct Hosts<'c, T: Real> {
    builder: FmmBuilder<T>,
    problem: &'c Problem<T>,
    comm: &'c SimpleCommunicator,
    outputs: Vec<(M2lStrategy, Output<T>)>,
}

impl<T: Real> Hosts<'_, T> {
    fn output(&mut self, strategy: M2lStrategy) -> &Output<T> {
        if !self.outputs.iter().any(|(s, _)| *s == strategy) {
            let mut host = self
                .builder
                .clone()
                .strategy(strategy)
                .build(&self.problem.points, &self.problem.points, self.comm)
                .expect("the host FMM builds");
            let output = host.evaluate(&self.problem.charges).expect("it evaluates");
            self.outputs.push((strategy, output));
        }
        &self.outputs.iter().find(|(s, _)| *s == strategy).unwrap().1
    }

    /// Asserts that `got` (of a device FMM with `strategy`) lies within the FMM bounds of
    /// the host output; returns the differences.
    fn check(&mut self, what: &str, strategy: M2lStrategy, got: &Output<T>) -> (f64, f64) {
        let bound = fmm_bound::<T>();
        let (potential, gradient) = relative_l2(got, self.output(strategy));
        assert!(
            potential <= bound && gradient <= bound,
            "{what}: {potential:e} (φ), {gradient:e} (∇φ) from the host (bound {bound:e})"
        );
        (potential, gradient)
    }
}

/// Runs the scenarios of the module documentation on `backend` in T at degree p, in
/// subdirectories of `root`; with `full` also the stale and corrupted files and the budget,
/// which test the cache and the deadline rather than the precision. Panics on a failed
/// check and prints what it did.
pub fn check_tuning<T: Real>(
    backend: Backend,
    p: usize,
    full: bool,
    root: &Path,
    comm: &SimpleCommunicator,
) {
    let precision = precision::<T>();
    let label = format!("{backend} {precision} p = {p}");
    let problem = Problem::<T>::new(20_000);
    let builder = FmmBuilder::<T>::new(p).gradients(true);
    let device = builder.clone().backend(backend);
    let mut hosts = Hosts {
        builder: builder.clone(),
        problem: &problem,
        comm,
        outputs: Vec::new(),
    };
    let build = |builder: &FmmBuilder<T>| {
        builder
            .build(&problem.points, &problem.points, comm)
            .unwrap_or_else(|error| panic!("{label}: the device FMM does not build: {error}"))
    };
    let info = backend.probe().expect("the backend opens");

    // The static rule without a directory.
    {
        let mut fmm = build(&device);
        let report = tuning(&fmm);
        assert_eq!(report.cache, CacheState::NoDirectory, "{label}");
        assert!(
            report.file.is_none() && report.stored.is_none(),
            "{label}: a file"
        );
        assert!(!report.tuned() && report.time == Duration::ZERO, "{label}");
        assert!(
            report.decisions.iter().all(|d| d.source == Source::Static
                && d.times.is_empty()
                && d.choice == static_candidate(&info, precision, p, d.decision)),
            "{label}: every decision from the static rule: {report}"
        );
        assert_eq!(fmm.strategy(), static_strategy(precision, p), "{label}");
        let device_report = fmm.device_report().unwrap();
        assert_eq!(device_report.p2p_layout, static_p2p(&info), "{label}");
        let n = (p + 1) * (p + 1);
        for t in &device_report.translations {
            let want = static_gemm(&info, precision, n, t.kind, DeviceGemm::Auto, None);
            assert_eq!(t.orientation, want.orientation, "{label}: {t:?}");
            assert_eq!(t.budget, want.budget, "{label}: {t:?}");
            if let GemmKind::HandWritten(layout) = want.gemm {
                assert_eq!(
                    t.gemm,
                    nd_fmm_exec::device::Gemm::HandWritten(layout),
                    "{label}"
                );
            }
        }
        let got = fmm.evaluate(&problem.charges).unwrap();
        hosts.check(&format!("{label}, static rule"), fmm.strategy(), &got);
        eprintln!(
            "  {label}: without a directory: {} decisions by the static rule, no file, no \
             tuning; strategy {:?}",
            report.decisions.len(),
            fmm.strategy()
        );
    }

    // A slow candidate: the static rule's candidate of every decision made slow.
    {
        let dir = fresh(root, "slow");
        let slow_info = info.clone();
        let hook = TuningHook::new().adjust(move |decision, candidate, time| {
            if *candidate == static_candidate(&slow_info, precision, p, *decision) {
                time + Duration::from_secs(1)
            } else {
                time
            }
        });
        let mut fmm = build(&device.clone().tuning_cache(&dir).tuning_hook(hook));
        let report = tuning(&fmm);
        assert_eq!(report.cache, CacheState::Missing, "{label}");
        assert_eq!(report.stored, Some(Ok(())), "{label}: {report}");
        let mut changed = 0;
        for d in report
            .decisions
            .iter()
            .filter(|d| d.source == Source::Tuned)
        {
            let first = static_candidate(&info, precision, p, d.decision);
            if measured(d).len() >= 2 {
                assert_ne!(
                    d.choice, first,
                    "{label}: {} chose the slowed candidate",
                    d.decision
                );
                changed += 1;
            }
        }
        assert!(changed >= 2, "{label}: too few tuned decisions: {report}");
        let first = fmm.evaluate(&problem.charges).unwrap();
        let second = fmm.evaluate(&problem.charges).unwrap();
        assert_eq!(
            output_bits(&first),
            output_bits(&second),
            "{label}: two evaluations"
        );
        assert_eq!(
            tuning(&fmm),
            report,
            "{label}: the report after evaluations"
        );
        let (phi, grad) = hosts.check(&format!("{label}, slow hook"), fmm.strategy(), &first);
        eprintln!(
            "  {label}: the slowed static candidate was not chosen in {changed} decisions; \
             two evaluations bit for bit; {phi:.1e} (φ), {grad:.1e} (∇φ) from the host\n{report}"
        );
    }

    // The input-precision guard: a TF32 library offered and made the fastest.
    {
        let dir = fresh(root, "guard");
        let tf32 = GemmChoice {
            gemm: GemmKind::Library {
                inputs: InputPrecision::Tf32,
            },
            ..GemmChoice::library(Precision::F32)
        };
        let is_tf32 = move |c: &Candidate| c.gemm() == Some(tf32);
        let hook = TuningHook::new()
            .offer(move |decision| match decision {
                Decision::Strategy => vec![Candidate::Dense(tf32)],
                Decision::Gemm { .. } => vec![Candidate::Gemm(tf32)],
                Decision::P2p { .. } => Vec::new(),
            })
            .adjust(move |_, candidate, time| {
                if is_tf32(candidate) {
                    Duration::ZERO
                } else {
                    time
                }
            });
        let fmm = build(&device.clone().tuning_cache(&dir).tuning_hook(hook));
        let report = tuning(&fmm);
        let mut offered = 0;
        for d in report
            .decisions
            .iter()
            .filter(|d| d.source == Source::Tuned)
        {
            assert!(!is_tf32(&d.choice), "{label}: {} chose TF32", d.decision);
            for (c, t) in d.times.iter().filter(|(c, _)| is_tf32(c)) {
                offered += 1;
                match t {
                    Timing::Unregistered(reason) if reason.contains("input-precision guard") => {}
                    other => panic!("{label}: {} registered {c}: {other}", d.decision),
                }
            }
        }
        assert!(
            offered >= 2,
            "{label}: the hook's candidate was not offered: {report}"
        );
        let stored = TuningCache::new(&dir)
            .load(&report.key)
            .expect("the file loads");
        assert!(
            stored.iter().all(|e| !is_tf32(&e.choice)),
            "{label}: TF32 stored"
        );
        eprintln!(
            "  {label}: a TF32 library offered to {offered} decisions and made the fastest by \
             the hook: never registered (input-precision guard), never chosen"
        );
    }

    // The cache round trip: two more builds from the directory.
    let round_trip = fresh(root, "round_trip");
    {
        let mut tuned = build(&device.clone().tuning_cache(&round_trip));
        let first = tuning(&tuned);
        assert!(
            first.tuned() && first.stored == Some(Ok(())),
            "{label}: {first}"
        );
        let want = tuned.evaluate(&problem.charges).unwrap();
        hosts.check(&format!("{label}, tuned"), tuned.strategy(), &want);
        for again in 0..2 {
            let mut fmm = build(&device.clone().tuning_cache(&round_trip));
            let report = tuning(&fmm);
            assert_eq!(report.cache, CacheState::Loaded, "{label}");
            assert!(
                !report.tuned() && report.stored.is_none(),
                "{label}: {report}"
            );
            for d in &first.decisions {
                let got = report.decision(d.decision).expect("the same decisions");
                assert_eq!(got.choice, d.choice, "{label}: {}", d.decision);
                let source = if d.source == Source::Tuned {
                    Source::Cached
                } else {
                    d.source
                };
                assert_eq!(got.source, source, "{label}: {}", d.decision);
            }
            assert_eq!(fmm.strategy(), tuned.strategy());
            let got = fmm.evaluate(&problem.charges).unwrap();
            assert_eq!(
                output_bits(&got),
                output_bits(&want),
                "{label}: build {} from the cache",
                again + 2
            );
        }
        eprintln!(
            "  {label}: tuned once ({:.2} s), then two builds from the cache: the same choices, \
             the outputs bit for bit",
            first.time.as_secs_f64()
        );
    }

    // Stale and corrupted files, through a build.
    if full {
        let key = tuning(&build(&device.clone().tuning_cache(&round_trip))).key;
        let file = TuningCache::new(&round_trip).path(&key);
        let original = fs::read_to_string(&file).expect("the round-trip file");
        let rewrite = |field: &str, value: &str| -> String {
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
        // Another key: the file of p + 1 under this key's name.
        let other = TuningKey {
            p: key.p + 1,
            ..key.clone()
        };
        let entries = TuningCache::new(&round_trip).load(&key).unwrap();
        let other_dir = fresh(root, "other_key");
        TuningCache::new(&other_dir)
            .store(&other, &entries)
            .unwrap();
        let other_text = fs::read(TuningCache::new(&other_dir).path(&other)).unwrap();
        let mut flipped = original.clone().into_bytes();
        let middle = flipped.len() / 2;
        flipped[middle] ^= 0x01;
        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            ("another key (p + 1)", other_text, "stale: p"),
            (
                "another CubeCL version",
                rewrite("cubecl", "0.10.0").into_bytes(),
                "stale: cubecl",
            ),
            (
                "another convention version",
                rewrite("convention", "999").into_bytes(),
                "stale: convention",
            ),
            ("a flipped byte", flipped, "corrupt"),
            (
                "a file cut in half",
                original.as_bytes()[..original.len() / 2].to_vec(),
                "corrupt",
            ),
            ("an empty file", Vec::new(), "corrupt"),
        ];
        for (what, bytes, reason) in cases {
            let dir = fresh(root, "stale");
            fs::write(TuningCache::new(&dir).path(&key), &bytes).unwrap();
            let mut fmm = build(&device.clone().tuning_cache(&dir));
            let report = tuning(&fmm);
            match &report.cache {
                CacheState::Rejected(why) if why.starts_with(reason) => {}
                other => panic!("{label}: {what}: {other:?}, expected `{reason}`"),
            }
            assert!(
                report.tuned() && report.stored == Some(Ok(())),
                "{label}: {what}"
            );
            assert!(
                TuningCache::new(&dir).load(&key).is_ok(),
                "{label}: {what}: the replacement loads"
            );
            let got = fmm.evaluate(&problem.charges).unwrap();
            hosts.check(&format!("{label}, {what}"), fmm.strategy(), &got);
            eprintln!(
                "  {label}: {what}: rejected ({}), re-tuned and replaced",
                match &report.cache {
                    CacheState::Rejected(why) => why.as_str(),
                    _ => unreachable!(),
                }
            );
        }
    }

    // Every tuned choice within the FMM bounds: the k-th candidate of every decision.
    {
        let most = [Decision::Strategy]
            .into_iter()
            .chain(
                [OperatorKind::M2m, OperatorKind::L2l, OperatorKind::M2l]
                    .map(|kind| Decision::Gemm { kind, bucket: 1 }),
            )
            .chain([Decision::P2p { bucket: 1 }])
            .map(|d| candidates(&info, precision, p, d).len())
            .max()
            .unwrap();
        let mut chosen: Vec<Candidate> = Vec::new();
        for k in 0..most {
            let dir = fresh(root, "kth");
            let kth_info = info.clone();
            let hook = TuningHook::new().adjust(move |decision, candidate, time| {
                let all = candidates(&kth_info, precision, p, *decision);
                match all.iter().position(|c| c == candidate) {
                    Some(i) if i == k.min(all.len() - 1) => Duration::ZERO,
                    _ => time + Duration::from_secs(1),
                }
            });
            let mut fmm = build(&device.clone().tuning_cache(&dir).tuning_hook(hook));
            let report = tuning(&fmm);
            for d in report
                .decisions
                .iter()
                .filter(|d| d.source == Source::Tuned)
            {
                let all = candidates(&info, precision, p, d.decision);
                let want = all[k.min(all.len() - 1)];
                if measured(d).contains(&want) {
                    assert_eq!(d.choice, want, "{label}: candidate {k} of {}", d.decision);
                }
                if !chosen.contains(&d.choice) {
                    chosen.push(d.choice);
                }
            }
            let got = fmm.evaluate(&problem.charges).unwrap();
            let (phi, grad) = hosts.check(&format!("{label}, candidate {k}"), fmm.strategy(), &got);
            eprintln!(
                "  {label}: candidate {k} of every decision chosen: strategy {:?}, {phi:.1e} (φ), \
                 {grad:.1e} (∇φ) from the host",
                fmm.strategy()
            );
        }
        eprintln!(
            "  {label}: {} distinct choices, each within the FMM bounds of the host",
            chosen.len()
        );
    }

    // The budget: 0.3 s, 0.1 s per candidate.
    if full {
        let dir = fresh(root, "budget");
        let budget = Duration::from_millis(300);
        let hook = TuningHook::new().adjust(|_, _, time| {
            std::thread::sleep(Duration::from_millis(100));
            time
        });
        let fmm = build(
            &device
                .clone()
                .tuning_cache(&dir)
                .tuning_budget(budget)
                .tuning_hook(hook),
        );
        let report = tuning(&fmm);
        assert!(
            report.deadline_hit,
            "{label}: the deadline was not reached: {report}"
        );
        assert!(
            report.last_start < budget,
            "{label}: a candidate started late: {report}"
        );
        let tuned: Vec<_> = report
            .decisions
            .iter()
            .filter(|d| d.source == Source::Tuned)
            .collect();
        assert!(!tuned.is_empty(), "{label}: nothing tuned");
        for d in &tuned {
            let first = d
                .times
                .iter()
                .find(|(_, t)| !matches!(t, Timing::Unregistered(_)));
            assert!(
                matches!(first, Some((_, Timing::Measured(_)))),
                "{label}: {}: the static rule's candidate first",
                d.decision
            );
        }
        let cut = tuned.iter().filter(|d| d.cut).count();
        let late: Vec<_> = report
            .decisions
            .iter()
            .filter(|d| {
                d.source == Source::Static
                    && d.note.as_deref().is_some_and(|n| n.contains("deadline"))
            })
            .collect();
        assert!(cut + late.len() >= 1, "{label}: nothing cut: {report}");
        let stored = TuningCache::new(&dir).load(&report.key).unwrap();
        assert_eq!(
            stored.len(),
            tuned.len(),
            "{label}: only tuned decisions stored"
        );
        // A second build with time: the late decisions are tuned now, the others cached.
        let second = tuning(&build(&device.clone().tuning_cache(&dir)));
        for d in &late {
            assert_eq!(
                second.decision(d.decision).map(|e| e.source),
                Some(Source::Tuned),
                "{label}: {} tuned by the second build",
                d.decision
            );
        }
        for d in &tuned {
            assert_eq!(
                second.decision(d.decision).map(|e| (e.source, e.choice)),
                Some((Source::Cached, d.choice)),
                "{label}: {} from the cache",
                d.decision
            );
        }
        eprintln!(
            "  {label}: a budget of 0.3 s with 0.1 s per candidate: {} tuned ({cut} cut), {} \
             left to the static rule, the last candidate started at {:.3} s; a second build \
             tuned those and took {} from the cache",
            tuned.len(),
            late.len(),
            report.last_start.as_secs_f64(),
            tuned.len()
        );
    }
}

/// The values as f64 bit patterns: potentials, then gradients (as `device_common`).
fn output_bits<T: RealScalar>(output: &Output<T>) -> Vec<u64> {
    let mut values: Vec<u64> = output
        .potential
        .iter()
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect();
    if let Some(gradient) = &output.gradient {
        values.extend(
            gradient
                .as_flattened()
                .iter()
                .map(|&v| RealScalar::to_f64(v).to_bits()),
        );
    }
    values
}

/// The relative L2 differences of `got` from `want`, φ and ∇φ (as `device_common`).
fn relative_l2<T: RealScalar>(got: &Output<T>, want: &Output<T>) -> (f64, f64) {
    let difference = |a: &[T], b: &[T]| -> f64 {
        let (mut d2, mut r2) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (RealScalar::to_f64(x), RealScalar::to_f64(y));
            d2 += (x - y) * (x - y);
            r2 += y * y;
        }
        if r2 > 0.0 {
            (d2 / r2).sqrt()
        } else {
            d2.sqrt()
        }
    };
    let potential = difference(&got.potential, &want.potential);
    let gradient = match (&got.gradient, &want.gradient) {
        (Some(a), Some(b)) => difference(a.as_flattened(), b.as_flattened()),
        _ => 0.0,
    };
    (potential, gradient)
}

/// The FMM bound of the device output against the host output (docs/phase4/README.md,
/// "Accuracy measures"): 1e-12 in f64, 1e-5 in f32, relative L2.
fn fmm_bound<T>() -> f64 {
    if size_of::<T>() == 4 { 1e-5 } else { 1e-12 }
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

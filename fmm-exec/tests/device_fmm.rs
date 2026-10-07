//! The C4.8 gate (Phase 4 T11, docs/phase4/README.md, "Exit gate"): the device FMM with
//! every operator kind on the device against the host FMM of the same settings, on the
//! uniform cube and the Plummer sphere of design §7, "Recommendation for Phase 4".
//!
//! Problems: N = 10⁵ points uniform in the cube [−1, 1)³ and in the Plummer sphere of
//! scale a = 0.1 (truncated at 10 a), drawn by the rules of
//! `nd_fmm_validate::fmm_accuracy::Distribution` (this crate cannot depend on
//! nd-fmm-validate), seed `0xc33`, sources equal to targets, `max_level` 16 and 64
//! points per leaf (the cube's tree is then uniform on level 4), gradients on, eight
//! charge vectors uniform in [−1, 1) (seeds `0xc33 + 1 + k`) and 1,000 targets sampled
//! from the points' generator: `fmm_accuracy::Config::c33` of the cube and of the Plummer
//! sphere, draw for draw. The default strategy (`Dense` at p ≤ 8, `Rotation` above), the
//! default device layouts, one thread.
//!
//! Workload points (README, "Workloads"), one per (backend, precision, p) compiled in:
//! - Metal (feature `metal`, by hand outside the macOS sandbox): f32 at p = 3 and 8;
//! - the CPU runtime (feature `cpu`): f64 at p = 8, 12 and 18, each at N = 10⁵, since
//!   none took more than the README's ten minutes on the M3 Max ([`F64_POINTS`]; the
//!   test prints N per row);
//! - CUDA (feature `cuda`, by hand on locust; Phase 4S T4, C4S.4): f32 at p = 3 and 8 and
//!   f64 at p = 8, 12 and 18, and the cube alone at N = 10⁶ ([`LARGE`]) in f32 and f64 at
//!   p = 8 (the H100 has 96 GB, and the direct sum at 1,000 targets stays cheap).
//!
//! The optional third distribution, the Gaussian clusters (five clusters of width 0.02),
//! runs at f32 p = 8 on Metal and at p = 8 in f32 and f64 on CUDA, reported and checked
//! like the others.
//!
//! Error measures (docs/phase3/README.md, "Error measures"): per charge vector, the
//! relative L2 and max errors of φ and of ∇φ (the Euclidean norm per target) at the
//! sampled targets against `direct_sum` in f64 over all sources divided by 4π, then the
//! root mean square over the eight vectors; f32 runs use the charges rounded to f32 and
//! the oracle on the rounded charges. The device-vs-host difference is the relative L2
//! difference of the outputs over all targets, φ and ∇φ, the largest over the vectors.
//!
//! Checks, per workload point and distribution:
//! - the device output within the FMM bounds of the host output: relative L2 1e-12 (f64)
//!   or 1e-5 (f32), φ and ∇φ;
//! - each of the four errors (φ L2, φ max, ∇φ L2, ∇φ max) of the device within 0.1%
//!   (f64) or 5% (f32) of the host's;
//! - two evaluations of the first charges bit-identical;
//! - every kind on the device, and the transfers of each evaluation the design's
//!   minimum (device-path.md §4.1, §8.2): the charges up, the output down, one sync;
//! - the C3.3 gate on the device path (docs/phase3/README.md, "Exit gate"): in f64, the
//!   φ L2 error of the Plummer sphere (and the clusters, where run) at most twice that of
//!   the uniform cube at the same p, N and backend; f32 is reported. The C3.2 gate on the
//!   device path is in `tests/accuracy.rs`.
//!
//! **The tuning budget at the C3.2 size** (Phase 4 T12): the cube at N = 10⁵ and p = 8 (f64
//! on the CPU runtime, f32 on Metal, f32 and f64 on CUDA) is built once more with a fresh tuning cache and the
//! default budget of 10 s; no candidate may start after the deadline
//! (`TuningReport::last_start` below the budget), the output of the tuned choices must lie
//! within the FMM bounds of the host output (one charge vector), and the tuning time and
//! every decision are printed.
//!
//! Every point is measured and printed (a Markdown table, with the host and device
//! evaluation times at one thread, reported and never asserted) before the checks are
//! asserted; the test prints the backends it ran. If a check fails, isolate the
//! operator kind with `FmmBuilder::host_fallback` and report the breakdown
//! (README, "Accuracy measures").
//!
//! Its own executable, because it initialises MPI; ignored, because it needs release mode:
//!
//! ```text
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release --test device_fmm -- --ignored --nocapture
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features metal --release --test device_fmm -- --ignored --nocapture
//! tools/gh200/remote.sh 'RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release --test device_fmm -- --ignored --nocapture'
//! ```
#![cfg(feature = "gpu")]

use std::time::{Duration, Instant};

use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::device::{Counters, DataKind, Traffic};
use nd_fmm_exec::fmm::{Backend, Fmm, FmmBuilder, OperatorKind, Output, Placement};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_math::RealScalar;
use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_tables::cache::Stored;

/// The number of points of a full-size run.
const N: usize = 100_000;

/// The number of sampled targets.
const SAMPLED: usize = 1000;

/// The number of charge vectors.
const CHARGE_VECTORS: u64 = 8;

/// The seed of the points and the sample; charge vector k uses `SEED + 1 + k`.
const SEED: u64 = 0xc33;

/// The f32 points on Metal: (p, N).
const F32_POINTS: [(usize, usize); 2] = [(3, N), (8, N)];

/// The f64 points on the CPU runtime: (p, N), each at N = 10⁵: on the M3 Max the slowest
/// point, the Plummer sphere at p = 18 with host and device, took 2.6 minutes, within the
/// README's ten.
const F64_POINTS: [(usize, usize); 3] = [(8, N), (12, N), (18, N)];

/// The cube's extra size on CUDA, at p = 8 in f32 and f64.
const LARGE: usize = 1_000_000;

/// SplitMix64, as `nd_fmm_validate::SplitMix64`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform in 0..n (n ≤ 2³², with a negligible bias).
    fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }
}

/// A point at distance `r` from the origin in a uniform direction (`points::sphere`):
/// rejection in the unit ball, norms below 10⁻³ rejected too.
fn at_radius(rng: &mut SplitMix64, r: f64) -> [f64; 3] {
    let v = loop {
        let v: [f64; 3] = core::array::from_fn(|_| rng.range(-1.0, 1.0));
        let norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if (1e-6..=1.0).contains(&norm_sq) {
            break v;
        }
    };
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|vi| r * (vi / norm))
}

/// A standard normal number (`points::gaussian_clusters`): Box–Muller, cosine branch.
fn normal(rng: &mut SplitMix64) -> f64 {
    let radius = (-2.0 * (1.0 - rng.uniform()).ln()).sqrt();
    radius * (2.0 * std::f64::consts::PI * rng.uniform()).cos()
}

/// The distributions, by name: the rules and parameters of
/// `nd_fmm_validate::fmm_accuracy::Distribution`.
fn distribution(name: &str, rng: &mut SplitMix64, n: usize) -> Vec<[f64; 3]> {
    match name {
        "cube" => (0..n)
            .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
            .collect(),
        "plummer" => {
            let mass = 1000.0 / 101.0f64.powf(1.5);
            (0..n)
                .map(|_| {
                    let u = mass * rng.uniform();
                    let r = if u > 0.0 {
                        0.1 / (u.powf(-2.0 / 3.0) - 1.0).sqrt()
                    } else {
                        0.0
                    };
                    at_radius(rng, r)
                })
                .collect()
        }
        "clusters" => {
            let centres = [
                [-0.5, -0.5, -0.5],
                [0.5, -0.4, 0.3],
                [-0.3, 0.6, 0.2],
                [0.4, 0.5, -0.6],
                [0.6, -0.3, 0.45],
            ];
            let k = centres.len();
            let mut points = Vec::with_capacity(n);
            for (j, c) in centres.iter().enumerate() {
                for _ in 0..n / k + usize::from(j < n % k) {
                    let z = loop {
                        let z: [f64; 3] = core::array::from_fn(|_| normal(rng));
                        if z[0] * z[0] + z[1] * z[1] + z[2] * z[2] <= 16.0 {
                            break z;
                        }
                    };
                    points.push(core::array::from_fn(|i| c[i] + 0.02 * z[i]));
                }
            }
            points
        }
        _ => unreachable!("{name}"),
    }
}

/// The points, the sampled targets, the charge vectors and the oracle of one
/// distribution at one N, as `fmm_accuracy::Problem::new` and `Oracle::new`.
struct Problem {
    name: &'static str,
    points: Vec<[f64; 3]>,
    sample: Vec<usize>,
    charges: Vec<Vec<f64>>,
}

impl Problem {
    fn new(name: &'static str, n: usize) -> Self {
        let mut rng = SplitMix64(SEED);
        let points = distribution(name, &mut rng, n);
        let mut order: Vec<usize> = (0..n).collect();
        for i in 0..SAMPLED {
            let j = i + rng.below(n - i);
            order.swap(i, j);
        }
        order.truncate(SAMPLED);
        let charges = (0..CHARGE_VECTORS)
            .map(|k| {
                let mut rng = SplitMix64(SEED + 1 + k);
                (0..n).map(|_| rng.range(-1.0, 1.0)).collect()
            })
            .collect();
        Self {
            name,
            points,
            sample: order,
            charges,
        }
    }

    /// The exact φ and ∇φ at the sampled targets for each of `charges`, divided by 4π.
    fn oracle(&self, charges: &[Vec<f64>]) -> Vec<(Vec<f64>, Vec<[f64; 3]>)> {
        let targets: Vec<[f64; 3]> = self.sample.iter().map(|&i| self.points[i]).collect();
        let scale = 4.0 * std::f64::consts::PI;
        charges
            .iter()
            .map(|q| {
                let mut phi = vec![0.0; targets.len()];
                let mut grad = vec![[0.0; 3]; targets.len()];
                direct_sum(&self.points, q, &targets, &mut phi, Some(&mut grad));
                (
                    phi.iter().map(|v| v / scale).collect(),
                    grad.iter().map(|g| g.map(|gk| gk / scale)).collect(),
                )
            })
            .collect()
    }
}

/// The four errors of one run: the root mean squares over the charge vectors of the
/// relative L2 and max errors of φ and of ∇φ.
#[derive(Clone, Copy, Debug, Default)]
struct Errors {
    potential_l2: f64,
    potential_max: f64,
    gradient_l2: f64,
    gradient_max: f64,
}

impl Errors {
    const NAMES: [&'static str; 4] = ["φ L2", "φ max", "∇φ L2", "∇φ max"];

    fn values(&self) -> [f64; 4] {
        [
            self.potential_l2,
            self.potential_max,
            self.gradient_l2,
            self.gradient_max,
        ]
    }
}

/// The relative L2 and max errors of one output at the sampled targets.
fn errors<T: RealScalar>(
    output: &Output<T>,
    sample: &[usize],
    (phi, grad): &(Vec<f64>, Vec<[f64; 3]>),
) -> [f64; 4] {
    let gradient = output.gradient.as_ref().expect("built with gradients");
    let (mut e2, mut r2, mut emax, mut rmax) = ([0.0f64; 2], [0.0f64; 2], [0.0f64; 2], [0.0f64; 2]);
    for (r, &i) in sample.iter().enumerate() {
        let dp = (RealScalar::to_f64(output.potential[i]) - phi[r]).abs();
        let dg: [f64; 3] =
            core::array::from_fn(|k| RealScalar::to_f64(gradient[i][k]) - grad[r][k]);
        let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        for (c, (e, x)) in [(dp, phi[r].abs()), (norm(dg), norm(grad[r]))]
            .into_iter()
            .enumerate()
        {
            e2[c] += e * e;
            r2[c] += x * x;
            emax[c] = emax[c].max(e);
            rmax[c] = rmax[c].max(x);
        }
    }
    [
        (e2[0] / r2[0]).sqrt(),
        emax[0] / rmax[0],
        (e2[1] / r2[1]).sqrt(),
        emax[1] / rmax[1],
    ]
}

/// The relative L2 difference of `got` from `want` over all targets, φ and ∇φ.
fn difference<T: RealScalar>(got: &Output<T>, want: &Output<T>) -> (f64, f64) {
    let relative = |a: &[T], b: &[T]| {
        let (mut d2, mut r2) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (RealScalar::to_f64(x), RealScalar::to_f64(y));
            d2 += (x - y) * (x - y);
            r2 += y * y;
        }
        (d2 / r2).sqrt()
    };
    let gradient = |o: &Output<T>| {
        o.gradient
            .as_ref()
            .expect("gradients")
            .as_flattened()
            .to_vec()
    };
    (
        relative(&got.potential, &want.potential),
        relative(&gradient(got), &gradient(want)),
    )
}

/// The bit patterns of an output.
fn bits<T: RealScalar>(output: &Output<T>) -> Vec<u64> {
    let gradient = output.gradient.as_ref().expect("gradients");
    output
        .potential
        .iter()
        .chain(gradient.as_flattened())
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect()
}

/// One row of the report.
struct Row {
    backend: Backend,
    precision: &'static str,
    distribution: &'static str,
    p: usize,
    n: usize,
    strategy: String,
    host: Errors,
    device: Errors,
    /// The largest relative L2 difference of the device output from the host's, φ and ∇φ.
    difference: (f64, f64),
    /// Transfers, launches and syncs of the device's last evaluation.
    counters: Counters,
    /// Mean evaluation times, host (one thread) and device.
    times: (Duration, Duration),
    wall: Duration,
}

/// Runs one workload point of one distribution on `backend` in `T` (module
/// documentation) and returns its row; pushes every failed check onto `failures`.
fn run<T: Stored + SimdScalar + Equivalence + Default + RealScalar>(
    backend: Backend,
    problem: &Problem,
    p: usize,
    comm: &SimpleCommunicator,
    failures: &mut Vec<String>,
) -> Row {
    let start = Instant::now();
    let f64_run = size_of::<T>() == 8;
    let precision = if f64_run { "f64" } else { "f32" };
    let charges: Vec<Vec<T>> = problem
        .charges
        .iter()
        .map(|q| q.iter().map(|&v| T::from_f64(v)).collect())
        .collect();
    let rounded: Vec<Vec<f64>> = charges
        .iter()
        .map(|q| q.iter().map(|&v| RealScalar::to_f64(v)).collect())
        .collect();
    let oracle = problem.oracle(&rounded);
    let builder = FmmBuilder::<T>::new(p).gradients(true);
    let mut host = builder
        .build(&problem.points, &problem.points, comm)
        .expect("the host FMM builds");
    let mut device: Fmm<'_, T> = builder
        .clone()
        .backend(backend)
        .build(&problem.points, &problem.points, comm)
        .unwrap_or_else(|error| panic!("{backend}: the device FMM does not build: {error}"));
    let what = format!("{backend}, {precision}, {}, p = {p}", problem.name);
    for kind in OperatorKind::ALL {
        assert_eq!(device.placement(kind), Placement::Device, "{what}: {kind}");
    }
    let strategy = device
        .device_report()
        .expect("a device backend")
        .strategy_name();

    let (mut host_errors, mut device_errors) = (Vec::new(), Vec::new());
    let mut largest = (0.0f64, 0.0f64);
    let (mut host_time, mut device_time) = (Duration::ZERO, Duration::ZERO);
    let mut first = None;
    for (q, exact) in charges.iter().zip(&oracle) {
        let t = Instant::now();
        let h = host.evaluate(q).expect("the host FMM evaluates");
        host_time += t.elapsed();
        let t = Instant::now();
        let d = device.evaluate(q).expect("the device FMM evaluates");
        device_time += t.elapsed();
        host_errors.push(errors(&h, &problem.sample, exact));
        device_errors.push(errors(&d, &problem.sample, exact));
        let (dp, dg) = difference(&d, &h);
        largest = (largest.0.max(dp), largest.1.max(dg));
        first.get_or_insert_with(|| bits(&d));
    }
    // Two evaluations of the first charges, bit for bit.
    let again = device
        .evaluate(&charges[0])
        .expect("the device FMM evaluates");
    if Some(bits(&again)) != first {
        failures.push(format!(
            "{what}: two evaluations of one charge vector differ"
        ));
    }
    // The design's minimum per evaluation: the charges up, the output down, one sync.
    let counters = device.device_counters().expect("a device backend");
    let s = size_of::<T>() as u64;
    let n = problem.points.len() as u64;
    for data in DataKind::ALL {
        let want = match data {
            DataKind::Charges => Traffic {
                uploads: 1,
                upload_bytes: n * s,
                ..Traffic::default()
            },
            DataKind::Output => Traffic {
                downloads: 1,
                download_bytes: 4 * n * s,
                ..Traffic::default()
            },
            _ => Traffic::default(),
        };
        if counters.evaluation_traffic.get(data) != want {
            failures.push(format!(
                "{what}: {data} moved {:?} in an evaluation, not {want:?}",
                counters.evaluation_traffic.get(data)
            ));
        }
    }
    if counters.evaluation.syncs != 1 {
        failures.push(format!(
            "{what}: {} syncs in an evaluation",
            counters.evaluation.syncs
        ));
    }

    let rms = |each: &[[f64; 4]]| -> Errors {
        let m =
            |c: usize| (each.iter().map(|e| e[c] * e[c]).sum::<f64>() / each.len() as f64).sqrt();
        Errors {
            potential_l2: m(0),
            potential_max: m(1),
            gradient_l2: m(2),
            gradient_max: m(3),
        }
    };
    let (host_errors, device_errors) = (rms(&host_errors), rms(&device_errors));
    let (bound, ratio_bound) = if f64_run { (1e-12, 1e-3) } else { (1e-5, 5e-2) };
    if largest.0 > bound || largest.1 > bound {
        failures.push(format!(
            "{what}: the device output differs from the host's by φ {:e}, ∇φ {:e} \
             (bound {bound:e})",
            largest.0, largest.1
        ));
    }
    for ((name, h), d) in Errors::NAMES
        .iter()
        .zip(host_errors.values())
        .zip(device_errors.values())
    {
        if (d / h - 1.0).abs() > ratio_bound {
            failures.push(format!(
                "{what}: the device's {name} error {d:e} is {} times the host's {h:e}",
                d / h
            ));
        }
    }
    let evaluations = charges.len() as u32;
    Row {
        backend,
        precision,
        distribution: problem.name,
        p,
        n: problem.points.len(),
        strategy,
        host: host_errors,
        device: device_errors,
        difference: largest,
        counters: counters.evaluation,
        times: (host_time / evaluations, device_time / evaluations),
        wall: start.elapsed(),
    }
}

/// The tuning budget at the C3.2 size (module documentation) on `backend` in T at degree p:
/// pushes a failed check onto `failures` and returns the report's lines.
fn tuning_budget<T: Stored + SimdScalar + Equivalence + Default + RealScalar>(
    backend: Backend,
    problem: &Problem,
    p: usize,
    comm: &SimpleCommunicator,
    failures: &mut Vec<String>,
) -> String {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("device_fmm_tuning_{backend}_{}", size_of::<T>()));
    let _ = std::fs::remove_dir_all(&dir);
    let builder = FmmBuilder::<T>::new(p).gradients(true);
    let charges: Vec<T> = problem.charges[0].iter().map(|&v| T::from_f64(v)).collect();
    let start = Instant::now();
    let mut device = builder
        .clone()
        .backend(backend)
        .tuning_cache(&dir)
        .build(&problem.points, &problem.points, comm)
        .unwrap_or_else(|error| panic!("{backend}: the tuned FMM does not build: {error}"));
    let build = start.elapsed();
    let report = device
        .device_report()
        .and_then(|r| r.tuning.clone())
        .expect("a device build reports its tuning");
    let what = format!("{backend}, cube, p = {p}, tuned");
    if report.last_start >= report.budget {
        failures.push(format!(
            "{what}: a candidate started at {:?}, after the budget {:?}",
            report.last_start, report.budget
        ));
    }
    let mut host = builder
        .clone()
        .strategy(device.strategy())
        .build(&problem.points, &problem.points, comm)
        .expect("the host FMM builds");
    let (d, h) = (
        device.evaluate(&charges).expect("it evaluates"),
        host.evaluate(&charges).expect("it evaluates"),
    );
    let (dp, dg) = difference(&d, &h);
    let bound = if size_of::<T>() == 8 { 1e-12 } else { 1e-5 };
    if dp > bound || dg > bound {
        failures.push(format!(
            "{what}: the output differs from the host's by φ {dp:e}, ∇φ {dg:e}"
        ));
    }
    format!(
        "{what}: build {:.1} s, of which tuning {:.2} s (budget {:.0} s, last candidate started \
         at {:.2} s); output within {dp:.1e} (φ), {dg:.1e} (∇φ) of the host\n{report}",
        build.as_secs_f64(),
        report.time.as_secs_f64(),
        report.budget.as_secs_f64(),
        report.last_start.as_secs_f64()
    )
}

/// The C3.3 gate on the device path (module documentation): in f64 the φ L2 error of each
/// clustered distribution at most twice the cube's at the same backend, p and N.
fn c33_gate(rows: &[Row], failures: &mut Vec<String>) -> Vec<String> {
    let mut lines = Vec::new();
    for r in rows.iter().filter(|r| r.distribution != "cube") {
        let Some(cube) = rows.iter().find(|c| {
            c.distribution == "cube"
                && (c.backend, c.precision, c.p, c.n) == (r.backend, r.precision, r.p, r.n)
        }) else {
            continue;
        };
        let ratio = r.device.potential_l2 / cube.device.potential_l2;
        let host_ratio = r.host.potential_l2 / cube.host.potential_l2;
        let gated = r.precision == "f64";
        lines.push(format!(
            "| {} | {} | {} | {} | {:.3e} | {:.3e} | {ratio:.2} | {host_ratio:.2} | {} |",
            r.backend,
            r.precision,
            r.distribution,
            r.p,
            r.device.potential_l2,
            cube.device.potential_l2,
            match (gated, ratio <= 2.0) {
                (true, true) => "yes",
                (true, false) => "no",
                (false, _) => "(f32, reported)",
            }
        ));
        if gated && ratio > 2.0 {
            failures.push(format!(
                "{} {} p = {}: the device's φ L2 error is {ratio} times the cube's (C3.3)",
                r.backend, r.distribution, r.p
            ));
        }
    }
    lines
}

#[test]
#[ignore = "release mode: N = 10^5, eight charge vectors per point, host and device"]
fn device_fmm_gate() {
    let universe = mpi::initialize().expect("this test owns MPI initialization");
    let comm = universe.world();
    assert_eq!(comm.size(), 1, "the gate runs on one rank");

    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut ran = Vec::new();
    // The CPU runtime in f64; each problem is drawn once per N.
    if Backend::Cpu.is_compiled() {
        for name in ["cube", "plummer"] {
            let mut problem: Option<Problem> = None;
            for (p, n) in F64_POINTS {
                if problem.as_ref().is_none_or(|q| q.points.len() != n) {
                    problem = Some(Problem::new(name, n));
                }
                let problem = problem.as_ref().expect("drawn");
                let row = run::<f64>(Backend::Cpu, problem, p, &comm, &mut failures);
                eprintln!("{}: {:.1?}", describe(&row), row.wall);
                rows.push(row);
            }
        }
        ran.push("cpu (f64)");
    }
    // Metal in f32, with the Gaussian clusters at p = 8.
    if Backend::Metal.is_compiled() {
        for name in ["cube", "plummer", "clusters"] {
            let problem = Problem::new(name, N);
            for (p, n) in F32_POINTS {
                assert_eq!(n, problem.points.len());
                if name == "clusters" && p != 8 {
                    continue;
                }
                let row = run::<f32>(Backend::Metal, &problem, p, &comm, &mut failures);
                eprintln!("{}: {:.1?}", describe(&row), row.wall);
                rows.push(row);
            }
        }
        ran.push("metal (f32)");
    }
    // CUDA in f32 and f64, with the Gaussian clusters at p = 8, and the cube at N = 10⁶.
    if Backend::Cuda.is_compiled() {
        for name in ["cube", "plummer", "clusters"] {
            let problem = Problem::new(name, N);
            for (p, n) in F32_POINTS {
                assert_eq!(n, problem.points.len());
                if name == "clusters" && p != 8 {
                    continue;
                }
                let row = run::<f32>(Backend::Cuda, &problem, p, &comm, &mut failures);
                eprintln!("{}: {:.1?}", describe(&row), row.wall);
                rows.push(row);
            }
            for (p, n) in F64_POINTS {
                assert_eq!(n, problem.points.len());
                if name == "clusters" && p != 8 {
                    continue;
                }
                let row = run::<f64>(Backend::Cuda, &problem, p, &comm, &mut failures);
                eprintln!("{}: {:.1?}", describe(&row), row.wall);
                rows.push(row);
            }
        }
        let large = Problem::new("cube", LARGE);
        let row = run::<f32>(Backend::Cuda, &large, 8, &comm, &mut failures);
        eprintln!("{}: {:.1?}", describe(&row), row.wall);
        rows.push(row);
        let row = run::<f64>(Backend::Cuda, &large, 8, &comm, &mut failures);
        eprintln!("{}: {:.1?}", describe(&row), row.wall);
        rows.push(row);
        ran.push("cuda (f32, f64)");
    }

    println!();
    println!(
        "| backend | precision | distribution | p | N | strategy | host φ L2 | device φ L2 | ratio | host φ max | device φ max | ratio | host ∇φ L2 | device ∇φ L2 | ratio | host ∇φ max | device ∇φ max | ratio | device − host φ | device − host ∇φ |"
    );
    println!("|---|---|---|---:|---:|---|{}", "---:|".repeat(14));
    for r in &rows {
        let cells: Vec<String> = r
            .host
            .values()
            .iter()
            .zip(r.device.values())
            .map(|(h, d)| format!("{h:.3e} | {d:.3e} | {:.4}", d / h))
            .collect();
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {:.1e} | {:.1e} |",
            r.backend,
            r.precision,
            r.distribution,
            r.p,
            r.n,
            r.strategy,
            cells.join(" | "),
            r.difference.0,
            r.difference.1
        );
    }
    println!();
    println!(
        "| backend | precision | distribution | p | uploads | upload bytes | downloads | download bytes | launches | syncs | timing windows | host evaluate (1 thread) | device evaluate |"
    );
    println!("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let ms = |d: Duration| format!("{:.1} ms", d.as_secs_f64() * 1e3);
    for r in &rows {
        let c = &r.counters;
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            r.backend,
            r.precision,
            r.distribution,
            r.p,
            c.uploads,
            c.upload_bytes,
            c.downloads,
            c.download_bytes,
            c.launches,
            c.syncs,
            c.windows,
            ms(r.times.0),
            ms(r.times.1)
        );
    }
    // The tuning budget at the C3.2 size.
    let cube = Problem::new("cube", N);
    if Backend::Cpu.is_compiled() {
        let line = tuning_budget::<f64>(Backend::Cpu, &cube, 8, &comm, &mut failures);
        eprintln!("{line}");
        println!("\n{line}");
    }
    if Backend::Metal.is_compiled() {
        let line = tuning_budget::<f32>(Backend::Metal, &cube, 8, &comm, &mut failures);
        eprintln!("{line}");
        println!("\n{line}");
    }
    if Backend::Cuda.is_compiled() {
        for line in [
            tuning_budget::<f32>(Backend::Cuda, &cube, 8, &comm, &mut failures),
            tuning_budget::<f64>(Backend::Cuda, &cube, 8, &comm, &mut failures),
        ] {
            eprintln!("{line}");
            println!("\n{line}");
        }
    }
    let gate = c33_gate(&rows, &mut failures);
    if !gate.is_empty() {
        println!();
        println!(
            "| backend | precision | distribution | p | device φ L2 | cube φ L2 | ratio | host ratio | within 2× |"
        );
        println!("|---|---|---|---:|---:|---:|---:|---:|---|");
        for line in gate {
            println!("{line}");
        }
    }
    println!();
    println!(
        "backends run: host (f32, f64), {}; not run: {}",
        if ran.is_empty() {
            "none".to_owned()
        } else {
            ran.join(", ")
        },
        [
            (!Backend::Cpu.is_compiled()).then_some("cpu (not compiled)"),
            (!Backend::Metal.is_compiled()).then_some("metal (not compiled)"),
            (!Backend::Cuda.is_compiled()).then_some("cuda (not compiled)"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ")
    );
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// One line per row, while the test runs.
fn describe(r: &Row) -> String {
    format!(
        "{} {} {} p = {} N = {} ({}): φ L2 {:.3e} / {:.3e} (device / host), ∇φ L2 {:.3e} / \
         {:.3e}; output within {:.1e} / {:.1e}",
        r.backend,
        r.precision,
        r.distribution,
        r.p,
        r.n,
        r.strategy,
        r.device.potential_l2,
        r.host.potential_l2,
        r.device.gradient_l2,
        r.host.gradient_l2,
        r.difference.0,
        r.difference.1
    )
}

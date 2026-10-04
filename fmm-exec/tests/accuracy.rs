//! The C3.2 gate in its smallest form (docs/phase3/README.md, "Exit gate"): a uniform
//! level-4 tree with N = 10⁵ points uniform in a cube, sources equal to targets, charges
//! uniform in [−1, 1), f64, at p = 3 and 8. The full report, with p = 18, f32, the
//! gradients and the stage timings, is the `fmm_accuracy` example of nd-fmm-validate.
//!
//! Error measure: the relative L2 error of φ over 1,000 targets sampled with a fixed
//! seed (design §8.2), against `direct_sum` in f64 over all sources divided by 4π (the
//! oracle expands 1/|x − y|, `Fmm` returns Σ q / (4π |x − y|)), as the root mean square
//! over eight seeded charge vectors: √((1/8) Σₖ eₖ²) with eₖ = ‖φₖ − φₖ*‖ / ‖φₖ*‖
//! (docs/phase3/README.md, "Error measures"). Each vector counts equally; pooling the
//! sums instead would weight a vector by ‖φₖ*‖², and a vector with a large, smooth
//! potential (a small relative error) would dominate. The gate is twice the
//! single-translation prediction of Phase 1 (design
//! §7, "Single-translation accuracy", P2M → M2L → L2P): 1.77e-3 at p = 3 and 1.08e-5 at
//! p = 8.
//!
//! *Why several charge vectors.* With mixed-sign charges the far field of the coarsest
//! V-list level (64 boxes on level 2) partly cancels at each target, by an amount that
//! depends on the charge vector, while the truncation errors of its pairs do not
//! cancel. On this tree the error of one vector varied by a factor of 2.3 (p = 3) and
//! 2.7 (p = 8) across seven seeds, with every pair within its single-translation
//! error. The mean over several vectors measures the FMM, not the draw.
//!
//! Threads (T10, C3.5): at each p the FMM is built again with four threads, and every
//! evaluation equals the one-thread evaluation bit for bit. MPI is initialised with
//! `Threading::Funneled` for it.
//!
//! P2P kernels (Phase 3S T6, C3S.5): the FMM runs the default kernel, `P2pChoice::Auto`,
//! and at each p is built again with `P2pChoice::Reference`. Both pass the gate; the
//! error with `Auto` lies within 1% of the error with `Reference`, and every output of
//! `Auto` within 1e-13 of that of `Reference` (relative L2 over all targets).
//!
//! Device path (Phase 4 T6, C4.2; T7, C4.3; T8, C4.4; T9, C4.5; T10, C4.6), with a backend
//! feature: the same gate with the default placement (every kind on the device; the gate's
//! p ≤ 8 resolves to `Dense`, so M2L runs as dense GEMMs), on the CPU runtime in f64 at
//! p = 3 and 8 (feature `cpu`) and on Metal in f32 at p = 3 and 8 (feature `metal`, by
//! hand outside the macOS sandbox); on the CPU runtime also in f64 at p = 18, which
//! resolves to `Rotation`, so M2L runs the device rotation kernel (T10; no prediction
//! exists at p = 18, so only the comparison with the host applies). The device run's
//! error lies within 0.1% (f64) or 5% (f32) of the host run's in the same precision
//! (docs/phase4/README.md, "Accuracy measures"), its output within 1e-12 (f64) or 1e-5
//! (f32) of the host output (relative L2 over all targets), and in f64 at p = 3 and 8 it
//! passes the gate itself. The test prints the backends it ran.
//!
//! Its own executable, because it initialises MPI; ignored, because it needs release
//! mode:
//!
//! ```text
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release --test accuracy -- --ignored
//! ```

use mpi::Threading;
use mpi::traits::*;
#[cfg(feature = "gpu")]
use nd_fmm_exec::fmm::Backend;
use nd_fmm_exec::fmm::FmmBuilder;
use nd_fmm_exec::operator::{Isa, P2pChoice};
use nd_fmm_ref::p2p::direct_sum;

/// The single-translation prediction of the relative L2 error of φ (design §7).
const PREDICTION: [(usize, f64); 2] = [(3, 1.77e-3), (8, 1.08e-5)];

/// The degrees of the device gate on the CPU runtime in f64: those of the prediction, and
/// p = 18 under `Rotation` (T10), which has none.
#[cfg(feature = "gpu")]
const DEVICE_F64: [(usize, Option<f64>); 3] = [(3, Some(1.77e-3)), (8, Some(1.08e-5)), (18, None)];

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

    /// Uniform in 0..n (n ≤ 2³², with a negligible bias).
    fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }
}

/// The seed of the points and the sample; charge vector k has the seed `SEED + 1 + k`.
const SEED: u64 = 0xc32;

/// The number of charge vectors the error is averaged over.
const CHARGE_VECTORS: u64 = 8;

/// The thread count compared with one thread.
const THREADS: usize = 4;

/// How far the error with `Auto` may lie from the error with `Reference`, relative.
const KERNEL_ERROR_RATIO: f64 = 0.01;

/// The largest relative L2 difference of the `Auto` output from the `Reference` output.
const KERNEL_DIFFERENCE: f64 = 1e-13;

/// The relative L2 error of `potential` at the sampled targets against `exact`.
fn sampled_error(potential: &[f64], sample: &[usize], exact: &[f64]) -> f64 {
    let (mut e2, mut r2) = (0.0f64, 0.0f64);
    for (&i, &e) in sample.iter().zip(exact) {
        e2 += (potential[i] - e).powi(2);
        r2 += e * e;
    }
    (e2 / r2).sqrt()
}

/// The root mean square of `errors`.
fn rms(errors: &[f64]) -> f64 {
    (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt()
}

#[test]
#[ignore = "release mode: N = 10^5 and eight direct sums at 1,000 targets"]
fn uniform_tree_within_twice_the_prediction() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();
    assert_eq!(comm.size(), 1, "the gate runs on one rank");

    let n = 100_000;
    let mut rng = SplitMix64(SEED);
    let points: Vec<[f64; 3]> = (0..n)
        .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
        .collect();
    // 1,000 distinct targets: a partial Fisher–Yates shuffle.
    let mut order: Vec<usize> = (0..n).collect();
    for i in 0..1000 {
        let j = i + rng.below(n - i);
        order.swap(i, j);
    }
    let sample = &order[..1000];
    let sampled: Vec<[f64; 3]> = sample.iter().map(|&i| points[i]).collect();
    let charges: Vec<Vec<f64>> = (0..CHARGE_VECTORS)
        .map(|k| {
            let mut rng = SplitMix64(SEED + 1 + k);
            (0..n).map(|_| rng.range(-1.0, 1.0)).collect()
        })
        .collect();
    let exact: Vec<Vec<f64>> = charges
        .iter()
        .map(|q| {
            let mut phi = vec![0.0; sampled.len()];
            direct_sum(&points, q, &sampled, &mut phi, None);
            phi.iter()
                .map(|v| v / (4.0 * std::f64::consts::PI))
                .collect()
        })
        .collect();

    // Every p is measured and printed before the gate is checked.
    let detected = P2pChoice::Isa(Isa::detect());
    let mut failures = Vec::new();
    for (p, prediction) in PREDICTION {
        let builder = FmmBuilder::<f64>::new(p)
            .max_level(4)
            .max_points_per_leaf(1);
        let mut fmm = builder
            .build(&points, &points, &comm)
            .expect("the FMM builds");
        assert_eq!(fmm.p2p_kernel(), detected, "Auto by default");
        let leaves = fmm.plan().index().leaves();
        assert_eq!(fmm.nleaves(), 4096);
        assert!((0..fmm.nleaves()).all(|j| leaves.level(j) == 4), "uniform");
        let mut threaded = builder
            .clone()
            .threads(THREADS)
            .build(&points, &points, &comm)
            .expect("the threaded FMM builds");
        assert_eq!(threaded.threading().threads, THREADS);
        let mut reference = builder
            .clone()
            .p2p_kernel(P2pChoice::Reference)
            .build(&points, &points, &comm)
            .expect("the FMM builds with the reference P2P");
        assert_eq!(reference.p2p_kernel(), P2pChoice::Reference);
        let (mut each, mut each_reference) = (Vec::new(), Vec::new());
        let mut largest_difference = 0.0f64;
        for (k, (q, exact)) in charges.iter().zip(&exact).enumerate() {
            let output = fmm.evaluate(q).expect("the FMM evaluates");
            let parallel = threaded.evaluate(q).expect("the threaded FMM evaluates");
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&parallel.potential),
                bits(&output.potential),
                "p = {p}, charge vector {k}: {THREADS} threads differ from one"
            );
            let slow = reference.evaluate(q).expect("the reference FMM evaluates");
            let (d2, r2) = output
                .potential
                .iter()
                .zip(&slow.potential)
                .fold((0.0f64, 0.0f64), |(d2, r2), (a, b)| {
                    (d2 + (a - b).powi(2), r2 + b * b)
                });
            largest_difference = largest_difference.max((d2 / r2).sqrt());
            each.push(sampled_error(&output.potential, sample, exact));
            each_reference.push(sampled_error(&slow.potential, sample, exact));
        }
        let (error, error_reference) = (rms(&each), rms(&each_reference));
        let ratio = error / error_reference;
        let each: Vec<String> = each.iter().map(|e| format!("{e:.2e}")).collect();
        eprintln!(
            "uniform level-4 tree, N = {n}, p = {p}: relative L2 error of φ {error:.3e} \
             with {detected} (root mean square over {CHARGE_VECTORS} charge vectors: {}), \
             {error_reference:.3e} with reference (ratio {ratio:.6}), prediction \
             {prediction:.2e}, ratio {:.2}; outputs within {largest_difference:.1e} of \
             reference; {THREADS} threads bit for bit ({})",
            each.join(", "),
            error / prediction,
            threaded.threading()
        );
        for (kernel, e) in [(detected, error), (P2pChoice::Reference, error_reference)] {
            if e > 2.0 * prediction {
                failures.push(format!(
                    "p = {p}, {kernel}: {e:e} exceeds twice the prediction {prediction:e}"
                ));
            }
        }
        if (ratio - 1.0).abs() > KERNEL_ERROR_RATIO {
            failures.push(format!(
                "p = {p}: the error with {detected} is {ratio} times that with reference"
            ));
        }
        if largest_difference > KERNEL_DIFFERENCE {
            failures.push(format!(
                "p = {p}: {detected} differs from reference by {largest_difference:e}"
            ));
        }
    }
    #[cfg(feature = "gpu")]
    device_gates((&points, sample), (&charges, &exact), &comm, &mut failures);
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// The relative L2 difference of `a` from `b`.
#[cfg(feature = "gpu")]
fn relative_l2(a: &[f64], b: &[f64]) -> f64 {
    let (d2, r2) = a.iter().zip(b).fold((0.0f64, 0.0f64), |(d2, r2), (x, y)| {
        (d2 + (x - y).powi(2), r2 + y * y)
    });
    (d2 / r2).sqrt()
}

/// The gate on each device backend compiled in (module documentation, "Device path"):
/// the CPU runtime in f64, Metal in f32.
#[cfg(feature = "gpu")]
fn device_gates(
    (points, sample): (&[[f64; 3]], &[usize]),
    (charges, exact): (&[Vec<f64>], &[Vec<f64>]),
    comm: &mpi::topology::SimpleCommunicator,
    failures: &mut Vec<String>,
) {
    let mut ran = Vec::new();
    if Backend::Cpu.is_compiled() {
        device_gate::<f64>(
            Backend::Cpu,
            &DEVICE_F64,
            (points, sample),
            (charges, exact),
            comm,
            failures,
        );
        ran.push("cpu (f64)");
    }
    if Backend::Metal.is_compiled() {
        let degrees = PREDICTION.map(|(p, prediction)| (p, Some(prediction)));
        device_gate::<f32>(
            Backend::Metal,
            &degrees,
            (points, sample),
            (charges, exact),
            comm,
            failures,
        );
        ran.push("metal (f32)");
    }
    eprintln!(
        "backends run: host (f64), {}; not run: cuda (type-checked, not run)",
        ran.join(", ")
    );
}

/// The gate with every kind on `backend` in `T` at each degree of `degrees` (with its
/// prediction, if any), against the host path in `T`.
#[cfg(feature = "gpu")]
fn device_gate<
    T: nd_fmm_tables::cache::Stored
        + nd_fmm_exec::operator::SimdScalar
        + Equivalence
        + Default
        + nd_fmm_math::RealScalar,
>(
    backend: Backend,
    degrees: &[(usize, Option<f64>)],
    (points, sample): (&[[f64; 3]], &[usize]),
    (charges, exact): (&[Vec<f64>], &[Vec<f64>]),
    comm: &mpi::topology::SimpleCommunicator,
    failures: &mut Vec<String>,
) {
    let f64_run = size_of::<T>() == 8;
    let (error_ratio, difference_bound) = if f64_run { (1e-3, 1e-12) } else { (5e-2, 1e-5) };
    for &(p, prediction) in degrees {
        let start = std::time::Instant::now();
        let builder = FmmBuilder::<T>::new(p).max_level(4).max_points_per_leaf(1);
        let mut host = builder
            .build(points, points, comm)
            .expect("the host FMM builds");
        let mut device = builder
            .clone()
            .backend(backend)
            .build(points, points, comm)
            .unwrap_or_else(|error| panic!("{backend}: the device FMM does not build: {error}"));
        let report = device.device_report().expect("a device backend");
        let layout = format!(
            "{}; P2P {}, leaf operators {}, M2M/L2L/M2L GEMM {} ({} of {} level calls on the \
             library), rotation M2L {} ({} level calls)",
            report.strategy_name(),
            report.p2p_layout,
            report.leaf_layout,
            report.gemm_layout,
            report
                .translations
                .iter()
                .filter(|t| t.gemm == nd_fmm_exec::device::Gemm::Library)
                .count(),
            report.translations.len(),
            report.rotation_layout,
            report.rotations.len()
        );
        let (mut each_host, mut each_device) = (Vec::new(), Vec::new());
        let mut largest_difference = 0.0f64;
        for (q, exact) in charges.iter().zip(exact) {
            let q: Vec<T> = q.iter().map(|&v| T::from_f64(v)).collect();
            let widen = |v: &[T]| -> Vec<f64> { v.iter().map(|&x| x.to_f64()).collect() };
            let h = widen(&host.evaluate(&q).expect("the host FMM evaluates").potential);
            let d = widen(
                &device
                    .evaluate(&q)
                    .expect("the device FMM evaluates")
                    .potential,
            );
            largest_difference = largest_difference.max(relative_l2(&d, &h));
            each_host.push(sampled_error(&h, sample, exact));
            each_device.push(sampled_error(&d, sample, exact));
        }
        let (error_host, error_device) = (rms(&each_host), rms(&each_device));
        let ratio = error_device / error_host;
        eprintln!(
            "{backend}, {}, p = {p}, every kind on the device ({}): relative L2 error of φ \
             {error_device:.4e} against the host's {error_host:.4e} (ratio {ratio:.6}), \
             prediction {}; outputs within {largest_difference:.1e} of the host; host and \
             device built and evaluated {} times in {:.1?}",
            if f64_run { "f64" } else { "f32" },
            layout,
            prediction.map_or("none".to_owned(), |e| format!("{e:.2e}")),
            charges.len(),
            start.elapsed()
        );
        if (ratio - 1.0).abs() > error_ratio {
            failures.push(format!(
                "{backend}, p = {p}: the device error is {ratio} times the host's"
            ));
        }
        if largest_difference > difference_bound {
            failures.push(format!(
                "{backend}, p = {p}: the device output differs from the host's by \
                 {largest_difference:e}"
            ));
        }
        if let Some(prediction) = prediction
            && f64_run
            && error_device > 2.0 * prediction
        {
            failures.push(format!(
                "{backend}, p = {p}: {error_device:e} exceeds twice the prediction {prediction:e}"
            ));
        }
    }
}

//! The accuracy harness of the Helmholtz FMM (Helmholtz CH3.1c,
//! `nd_fmm_validate::helmholtz_accuracy`) on small cases: the checksum independent of the
//! partition and sensitive to one bit, the blocks a partition, the sharded oracle the
//! unsharded one bit for bit at any thread count, and the FMM within ε of the oracle with
//! the same checksum at 1 and 2 threads and, on several ranks, on one rank. A smoke run of
//! the calibration's plateau scan (Helmholtz CH3.3, `nd_fmm_validate::helmholtz_calibration`):
//! the errors fall with the degree, the smallest degree at ε/2 meets it, and the default
//! rule's degrees are at most S7's at the levels' circumradii.
//!
//! One test initialises MPI (at `Funneled`); it passes on any number of ranks, each check
//! on every rank or on none.

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Communicator;
use nd_fmm_exec::helmholtz::{
    ChargeKind, Degrees, Densities, Output, Wavenumber, WavenumberKind, s7_degree,
};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{Case, Oracle, Problem, checksum, complex_k, run};
use nd_fmm_validate::helmholtz_calibration::Curve;
use nd_fmm_validate::helmholtz_workloads::Workload;

/// An output of `n` targets, 2 complex vectors with gradients, filled with random values.
fn random_output(n: usize, rng: &mut SplitMix64) -> Output<f64> {
    let densities = Densities::new(WavenumberKind::Real, ChargeKind::Complex, 2);
    let mut output = Output::new(&densities, n, true);
    for v in &mut output.potential {
        *v = rng.range(-1.0, 1.0);
    }
    for g in output.gradient.as_mut().expect("gradients") {
        *g = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
    }
    output
}

/// Targets `range` of `output` as an output of their own.
fn part(output: &Output<f64>, range: std::ops::Range<usize>) -> Output<f64> {
    let densities = Densities::new(WavenumberKind::Real, ChargeKind::Complex, 2);
    let n = output.ntargets();
    let mut part = Output::new(&densities, range.len(), true);
    let columns = output.potential.len() / n;
    for o in 0..columns {
        let (from, to) = (o * n, o * range.len());
        part.potential[to..to + range.len()]
            .copy_from_slice(&output.potential[from + range.start..from + range.end]);
        part.gradient.as_mut().expect("gradients")[to..to + range.len()].copy_from_slice(
            &output.gradient.as_ref().expect("gradients")[from + range.start..from + range.end],
        );
    }
    part
}

/// The checksum, the blocks and the oracle (no FMM).
fn check_harness(world: &SimpleCommunicator) {
    let mut rng = SplitMix64::new(11);
    let output = random_output(37, &mut rng);
    let whole = checksum(&output, 0);
    let split =
        checksum(&part(&output, 0..20), 0).wrapping_add(checksum(&part(&output, 20..37), 20));
    assert_eq!(split, whole, "the checksum depends on the partition");
    // The same values at shifted positions, and one changed bit, change it.
    assert_ne!(checksum(&output, 1), whole);
    let mut changed = output.clone();
    changed.gradient.as_mut().expect("gradients")[40][2] =
        f64::from_bits(changed.gradient.as_ref().expect("gradients")[40][2].to_bits() ^ 1);
    assert_ne!(checksum(&changed, 0), whole);

    let case = Case::new(
        Workload::W4 {
            distribution: Distribution::Plummer,
            n: 500,
        },
        (10.0, 2.0),
        Degrees::Tolerance(1e-3),
    );
    let problem = Problem::new(&case);
    assert_eq!(problem.sample.len(), 500, "at most N samples");
    let mut sorted = problem.sample.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 500, "distinct samples");
    for size in [1, 3, 7] {
        let blocks: Vec<_> = (0..size).map(|r| problem.block(r, size)).collect();
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks[size - 1].end, 500);
        for pair in blocks.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
            assert!(pair[0].len().abs_diff(pair[1].len()) <= 1);
        }
    }
    let k = complex_k(&case);
    let all: Vec<usize> = (0..problem.sample.len()).collect();
    let serial = Oracle::of_samples(k, &problem, &all, true, 1);
    assert_eq!(Oracle::of_samples(k, &problem, &all, true, 3), serial);
    assert_eq!(Oracle::sharded(k, &problem, true, 2, world), serial);
}

/// The FMM within ε, at 1 and 2 threads with one checksum, and on several ranks the
/// checksum of one rank.
fn check_fmm(world: &SimpleCommunicator, case: &Case) {
    let problem = Problem::new(case);
    let oracle = Oracle::sharded(complex_k(case), &problem, case.gradients, 2, world);
    let rows: Vec<_> = [1, 2]
        .into_iter()
        .map(|threads| run(case, &problem, &oracle, threads, world).expect("the FMM runs"))
        .collect();
    let Degrees::Tolerance(eps) = case.degrees else {
        panic!("a tolerance")
    };
    for row in &rows {
        for (v, errors) in row.errors.iter().enumerate() {
            assert!(
                errors[0] <= eps && errors[1] <= eps,
                "{:?} k = {:?}, vector {v}: errors {errors:?} above ε = {eps}",
                case.workload,
                case.k,
            );
        }
        assert_eq!(row.errors.len(), case.vectors + 1);
    }
    assert_eq!(
        rows[0].checksum, rows[1].checksum,
        "threads change the output"
    );
    if world.size() > 1 {
        let one = SimpleCommunicator::self_comm();
        let row = run(case, &problem, &oracle, 1, &one).expect("the one-rank FMM runs");
        assert_eq!(row.checksum, rows[0].checksum, "ranks change the output");
        // The same output; the error sums are reduced in another order (relative 1e-12).
        for (one, many) in row.errors.iter().zip(&rows[0].errors) {
            for (a, b) in one.iter().zip(many) {
                assert!(
                    (a - b).abs() <= 1e-12 * a.abs(),
                    "errors {one:?} and {many:?}"
                );
            }
        }
    }
}

/// The plateau scan of the calibration on W1 at kD = 10: one degree p = 1..=8 on every
/// level ≥ 2, as the `helmholtz_calibrate` example runs it.
fn check_calibration_scan(world: &SimpleCommunicator) {
    let mut case = Case::new(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
    let problem = Problem::new(&case);
    let oracle = Oracle::sharded(complex_k(&case), &problem, true, 2, world);
    let default = run(&case, &problem, &oracle, 2, world).expect("the FMM runs");
    // The circumradii are √3 r_l of the run's domain, halving per level.
    let k = Wavenumber::new(10.0, 0.0).unwrap();
    assert_eq!(default.circumradii.len(), default.levels);
    for l in 1..default.levels {
        assert_eq!(default.circumradii[l], default.circumradii[l - 1] / 2.0);
    }
    // The default rule never above S7's.
    for (l, &p) in default.degrees.iter().enumerate().skip(2) {
        assert!(
            p <= s7_degree(&k, default.circumradii[l], 1e-3),
            "level {l}: {p}"
        );
    }
    let mut curve = Curve::new(1);
    while curve.next_degree() <= 8 {
        let mut degrees = vec![curve.next_degree(); 64];
        degrees[..2].fill(0);
        case.degrees = Degrees::PerLevel(degrees);
        let row = run(&case, &problem, &oracle, 2, world).expect("the FMM runs");
        curve.push(*row.errors.last().unwrap());
    }
    let largest = |e: &[f64; 2]| e[0].max(e[1]);
    assert!(
        largest(&curve.errors[7]) < 1e-3 * largest(&curve.errors[0]),
        "{curve:?}"
    );
    let p = curve.smallest_degree(0.5e-3).expect("p ≤ 8 meets 5e-4");
    for e in &curve.errors[p - 1..] {
        assert!(largest(e) <= 0.5e-3, "{curve:?}");
    }
}

#[test]
fn helmholtz_accuracy_harness() {
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
    let world = universe.world();
    check_harness(&world);

    // W1 at kD = 10 (796 points), two complex vectors with gradients.
    let mut w1 = Case::new(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
    w1.vectors = 2;
    check_fmm(&world, &w1);
    // W4's clusters, complex k, real charges, no gradients.
    let mut w4 = Case::new(
        Workload::W4 {
            distribution: Distribution::Clusters,
            n: 2000,
        },
        (10.0, 1.0),
        Degrees::Tolerance(1e-3),
    );
    w4.charges = ChargeKind::Real;
    w4.gradients = false;
    check_fmm(&world, &w4);
    check_calibration_scan(&world);
}

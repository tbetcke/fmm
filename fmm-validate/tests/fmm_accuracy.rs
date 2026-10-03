//! Smoke runs of the cores of the `fmm_accuracy` and `calibrate` examples at N = 500.
//!
//! - `fmm_accuracy`: p = 2 in f64 and f32 on a uniform tree: it runs, is consistent
//!   and converges roughly as expected, and on two threads gives the same errors. It
//!   reports the P2P kernel that ran, `Auto`'s ISA by default; with
//!   `P2pChoice::Reference` the errors agree to rounding. Then
//!   each clustered distribution at p = 4 on an adaptive tree: its leaves span several
//!   levels and its W and X lists are non-empty.
//! - `calibrate`: the sweeps of every distribution at p ≤ 3 in f64 and f32, the
//!   calibration read off them, and a leaf-size study at p = 2.
//!
//! The accuracy figures themselves are reported by the examples. One test initialises
//! MPI and runs both (MPI cannot be initialised twice in one process). Error measure:
//! the root mean squares of the relative L2 and max errors over the charge vectors that
//! `fmm_accuracy::run` reports, compared as plain numbers.

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use nd_fmm_exec::operator::{Isa, P2pChoice};
use nd_fmm_validate::calibration::{
    self, Measure, Precision, Reached, Reference, floor, leaf_study, smallest_p, sweep,
};
use nd_fmm_validate::fmm_accuracy::{
    Config, Distribution, Execution, Oracle, Problem, prediction, run,
};

const SMOKE: Config = Config {
    distribution: Distribution::Cube,
    n: 500,
    max_level: 2,
    max_points_per_leaf: 1,
    sampled: 100,
    charge_vectors: 2,
    seed: 7,
};

#[test]
fn smoke_runs_at_n_500() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();
    fmm_accuracy_at_p_2_and_4(&comm);
    calibration_at_p_up_to_3(&comm);
}

/// The core of `fmm_accuracy`: a uniform tree at p = 2, then the clustered
/// distributions at p = 4.
fn fmm_accuracy_at_p_2_and_4(comm: &SimpleCommunicator) {
    let problem = Problem::new(&SMOKE);
    assert_eq!(problem.points.len(), SMOKE.n);
    assert_eq!(problem.charges.len(), SMOKE.charge_vectors);
    let mut sample = problem.sample.clone();
    sample.sort_unstable();
    sample.dedup();
    assert_eq!(
        sample.len(),
        SMOKE.sampled,
        "the sampled targets are distinct"
    );
    assert_eq!(problem, Problem::new(&SMOKE), "seeded");

    let oracle = Oracle::new(&problem, &problem.charges);
    let one = (2, Execution::threads(1));
    let r64 = run::<f64>(&SMOKE, &problem, &problem.charges, &oracle, one, comm);
    assert_eq!(r64.threading.threads, 1);
    assert_eq!(r64.p2p, P2pChoice::Isa(Isa::detect()), "Auto by default");
    // Two threads: the output is bit-identical (C3.5), so are the errors.
    let two = (2, Execution::threads(2));
    let threaded = run::<f64>(&SMOKE, &problem, &problem.charges, &oracle, two, comm);
    assert_eq!(threaded.threading.threads, 2);
    assert_eq!(
        (threaded.potential, threaded.gradient),
        (r64.potential, r64.gradient),
        "two threads"
    );
    // The reference P2P: the same errors up to rounding (C3S.5).
    let reference = Execution {
        p2p: P2pChoice::Reference,
        ..Execution::threads(1)
    };
    let slow = run::<f64>(
        &SMOKE,
        &problem,
        &problem.charges,
        &oracle,
        (2, reference),
        comm,
    );
    assert_eq!(slow.p2p, P2pChoice::Reference);
    for (a, b) in [
        (slow.potential, r64.potential),
        (slow.gradient, r64.gradient),
    ] {
        assert!((a.l2 - b.l2).abs() <= 1e-10 * b.l2, "{a:?} against {b:?}");
    }
    let charges32 = problem.charges_as::<f32>();
    let rounded: Vec<Vec<f64>> = charges32
        .iter()
        .map(|q| q.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let r32 = run::<f32>(
        &SMOKE,
        &problem,
        &charges32,
        &Oracle::new(&problem, &rounded),
        one,
        comm,
    );
    eprintln!("{r64:?}\n{r32:?}");

    for r in [&r64, &r32] {
        let values = [
            r.potential.l2,
            r.potential.max,
            r.gradient.l2,
            r.gradient.max,
        ];
        assert!(
            values.iter().all(|e| e.is_finite() && *e > 0.0),
            "{values:?}"
        );
        // The error of a p = 2 FMM: well below 1, far above rounding (the uniform
        // cube of nd-fmm-exec's tests gives 6e-3 at p = 2).
        assert!(r.potential.l2 < 5e-2, "{}: {}", r.precision, r.potential.l2);
        let (low, high) = r.potential_l2_range;
        assert!(low <= r.potential.l2 && r.potential.l2 <= high);
        assert_eq!(r.nleaves, 64, "a uniform level-2 tree");
        assert_eq!(r.nlevels, 3);
        assert_eq!((r.lists.w, r.lists.x), (0, 0));
        assert_eq!(
            r.points_per_leaf.1 * r.nleaves as f64,
            SMOKE.n as f64,
            "every point in a leaf"
        );
        assert_eq!(r.ratio(), None, "no prediction at p = 2");
    }
    assert_eq!((r64.precision, r32.precision), ("f64", "f32"));
    // At p = 2 truncation dominates, so f32 and f64 agree to a few per cent.
    assert!((r32.potential.l2 - r64.potential.l2).abs() < 0.05 * r64.potential.l2);
    assert_eq!(prediction(3), Some(1.77e-3));
    assert_eq!(r64.leaf_levels, vec![0, 0, 64]);

    // The clustered distributions on adaptive trees, as in `Config::c33` but small: 500
    // points, eight per leaf as the refinement target.
    for distribution in [
        Distribution::Sphere,
        Distribution::Plummer,
        Distribution::Clusters,
    ] {
        assert_eq!(
            Distribution::from_name(distribution.name()),
            Some(distribution)
        );
        let config = Config {
            n: 500,
            max_points_per_leaf: 8,
            ..Config::c33(distribution)
        };
        let config = Config {
            sampled: 100,
            charge_vectors: 2,
            ..config
        };
        let problem = Problem::new(&config);
        let oracle = Oracle::new(&problem, &problem.charges);
        let r = run::<f64>(
            &config,
            &problem,
            &problem.charges,
            &oracle,
            (4, Execution::threads(1)),
            comm,
        );
        let name = distribution.name();
        eprintln!("{name}: {r:?}");
        assert_eq!(r.distribution, distribution);
        let levels = r.leaf_levels.iter().filter(|&&n| n > 0).count();
        assert!(levels >= 3, "{name}: leaves on {levels} levels");
        assert_eq!(r.leaf_levels.iter().sum::<usize>(), r.nleaves);
        assert!(r.lists.w > 0 && r.lists.x > 0, "{name}: {:?}", r.lists);
        // A loose smoke bound: the p = 4 error of the uniform cube in nd-fmm-exec's
        // tests is 6e-4.
        assert!(r.potential.l2 < 1e-2, "{name}: {}", r.potential.l2);
    }
}

/// The core of `calibrate`: every distribution at p ≤ 3 in f64 and f32, the calibration
/// read off the sweeps, and a leaf-size study at p = 2.
fn calibration_at_p_up_to_3(comm: &SimpleCommunicator) {
    const DEGREES: [usize; 3] = [1, 2, 3];
    for distribution in Distribution::ALL {
        let name = distribution.name();
        // The calibration problem, but small: 500 points, eight per leaf as the
        // refinement target, so that the V lists are not empty.
        let config = Config {
            max_points_per_leaf: 8,
            sampled: 100,
            charge_vectors: 2,
            ..calibration::config(distribution, 500)
        };
        assert_eq!(
            (config.n, config.max_level, config.seed),
            (500, 16, Config::c33(distribution).seed)
        );
        let reference = Reference::new(&config);
        assert_eq!(reference.problem, Problem::new(&config), "{name}: seeded");
        assert_eq!(reference.charges32, reference.problem.charges_as::<f32>());

        let one = Execution::threads(1);
        let f64 = sweep(&config, &reference, Precision::F64, &DEGREES, one, comm);
        let f32 = sweep(&config, &reference, Precision::F32, &DEGREES, one, comm);
        for (runs, precision) in [(&f64, Precision::F64), (&f32, Precision::F32)] {
            let ps: Vec<usize> = runs.iter().map(|r| r.p).collect();
            assert_eq!(ps, DEGREES, "{name}");
            assert!(runs.iter().all(|r| r.precision == precision.name()));
            assert!(runs[0].lists.v > 0, "{name}: {:?}", runs[0].lists);
            for r in runs.iter() {
                assert!(r.potential.l2.is_finite() && r.potential.l2 > 0.0);
                assert!(r.gradient.l2.is_finite() && r.gradient.l2 > 0.0);
            }
            // Truncation dominates at p ≤ 3: the error falls from p = 1 to p = 3.
            assert!(
                runs[2].potential.l2 < runs[0].potential.l2,
                "{name} {}: {} then {}",
                precision.name(),
                runs[0].potential.l2,
                runs[2].potential.l2
            );
        }
        // At p ≤ 3, f32 and f64 agree to a few per cent.
        for (a, b) in f64.iter().zip(&f32) {
            assert!((a.potential.l2 - b.potential.l2).abs() < 0.05 * a.potential.l2);
        }
        // Two threads: the same errors (C3.5).
        let two = Execution::threads(2);
        let threaded = sweep(&config, &reference, Precision::F64, &[2], two, comm);
        assert_eq!(
            (threaded[0].potential, threaded[0].gradient),
            (f64[1].potential, f64[1].gradient),
            "{name}: two threads"
        );

        // The calibration: the first degree below the target, or beyond the sweep.
        for measure in [Measure::Potential, Measure::Gradient] {
            let errors: Vec<f64> = f64.iter().map(|r| measure.of(r)).collect();
            for target in [1.0, errors[1] * 1.0001, errors[2], 1e-12] {
                let expected = errors
                    .iter()
                    .position(|&e| e < target)
                    .map_or(Reached::Beyond(3), |i| Reached::At(DEGREES[i]));
                assert_eq!(smallest_p(&f64, measure, target), expected, "{name}");
            }
            assert_eq!(smallest_p(&f64, measure, 1e-12), Reached::Beyond(3));
            let (e, p) = floor(&f64, measure);
            assert_eq!(e, errors.iter().copied().fold(f64::INFINITY, f64::min));
            assert_eq!(measure.of(&f64[p - 1]), e);
        }

        // The leaf-size study: the refinement target changes the tree, not the problem;
        // at the sweep's own target it repeats the sweep's run.
        if calibration::LEAF_STUDY_DISTRIBUTIONS.contains(&distribution) {
            let study = leaf_study(&config, &reference, 2, &[4, 8, 16], one, comm);
            assert_eq!(study.len(), 3);
            assert!(study.iter().all(|r| r.p == 2 && r.precision == "f64"));
            assert!(
                study[0].nleaves > study[2].nleaves,
                "{name}: smaller leaves"
            );
            assert_eq!(
                (study[1].potential, study[1].gradient, study[1].nleaves),
                (f64[1].potential, f64[1].gradient, f64[1].nleaves),
                "{name}: the sweep's tree"
            );
        }
    }
}

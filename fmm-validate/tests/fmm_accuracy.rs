//! A smoke run of the core of the `fmm_accuracy` example at N = 500, p = 2, in f64 and
//! f32: it runs, is consistent and converges roughly as expected, and on two threads
//! gives the same errors. Then each clustered distribution at N = 500 and p = 4 on an
//! adaptive tree: its leaves span several levels and its W and X lists are non-empty.
//! The accuracy figures themselves are reported by the example.
//!
//! The one test of this executable that initialises MPI (MPI cannot be initialised
//! twice in one process). Error measure: the root mean squares of the relative L2 and
//! max errors over the charge vectors that `fmm_accuracy::run` reports, compared as
//! plain numbers.

use mpi::Threading;
use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Oracle, Problem, prediction, run};

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
fn smoke_run_at_n_500_and_p_2() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();

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
    let r64 = run::<f64>(&SMOKE, &problem, &problem.charges, &oracle, (2, 1), &comm);
    assert_eq!(r64.threading.threads, 1);
    // Two threads: the output is bit-identical (C3.5), so are the errors.
    let threaded = run::<f64>(&SMOKE, &problem, &problem.charges, &oracle, (2, 2), &comm);
    assert_eq!(threaded.threading.threads, 2);
    assert_eq!(
        (threaded.potential, threaded.gradient),
        (r64.potential, r64.gradient),
        "two threads"
    );
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
        (2, 1),
        &comm,
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
        let r = run::<f64>(&config, &problem, &problem.charges, &oracle, (4, 1), &comm);
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

//! A smoke run of the core of the `fmm_accuracy` example at N = 500, p = 2, in f64 and
//! f32: it runs, is consistent and converges roughly as expected, and on two threads
//! gives the same errors. The accuracy figures themselves are reported by the example.
//!
//! The one test of this executable that initialises MPI (MPI cannot be initialised
//! twice in one process). Error measure: the root mean squares of the relative L2 and
//! max errors over the charge vectors that `fmm_accuracy::run` reports, compared as
//! plain numbers.

use mpi::Threading;
use nd_fmm_validate::fmm_accuracy::{Config, Oracle, Problem, prediction, run};

const SMOKE: Config = Config {
    n: 500,
    max_level: 2,
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
}

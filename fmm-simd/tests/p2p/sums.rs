//! Sums against `direct_sum` (design §3, requirement 2): within 1e-14 (f64) and 1e-6
//! (f32) relative to the sum of term magnitudes of each target, for up to 4,096
//! sources, or within twice the reference's own error on the same inputs where that
//! is larger (`Oracle::tolerance`). Seeded uniform sets at every n_t up to 3 · K · W + 1, a cancelling set, and
//! FMM-shaped leaf-scaled sets in the per-pair and the gathered form.

use proptest::prelude::*;

use crate::common::{
    BLOCK, Oracle, Precision, Problem, Rng, Worst, assert_within, fmm_shaped, gathered, kernels,
    problem, run, runner, uniform,
};

/// The source counts of the seeded sets.
const SOURCES: [usize; 5] = [1, 7, 64, 1000, 4096];

/// Seeded uniform sets: every n_s of [`SOURCES`] against every n_t from 0 to
/// 3 · [`BLOCK`] + 1, or, for n_s above `full_up_to`, every `stride`-th n_t.
fn seeded<T: Precision>(test: &str, full_up_to: usize, stride: usize) {
    let kernels = kernels::<T>(test);
    for with_gradient in [false, true] {
        let mut worst = Worst::new(format!(
            "{test} ({}, gradient: {with_gradient})",
            std::any::type_name::<T>()
        ));
        for (seed, n_sources) in SOURCES.into_iter().enumerate() {
            let step = if n_sources > full_up_to { stride } else { 1 };
            for n_targets in (0..=3 * BLOCK + 1).step_by(step) {
                let problem: Problem<T> =
                    uniform(n_sources, n_targets, (100 * seed + n_targets) as u64);
                let oracle = Oracle::new(&problem);
                for &kernel in &kernels {
                    let errors = oracle.errors(&run(kernel, &problem, with_gradient));
                    assert_within(
                        errors,
                        oracle.tolerance::<T>(),
                        &format_args!("{} n_s = {n_sources}, n_t = {n_targets}", kernel.isa()),
                    );
                    worst.record_sum::<T>(kernel.isa(), errors, &oracle, with_gradient);
                }
            }
        }
        worst.print(1.0, "(relative to the term magnitudes)");
    }
}

#[test]
fn seeded_sums_f32() {
    seeded::<f32>("seeded_sums_f32", 1000, 5);
}

#[test]
fn seeded_sums_f64() {
    seeded::<f64>("seeded_sums_f64", 1000, 5);
}

#[test]
#[ignore = "every n_t at 4,096 sources; run in release with --ignored"]
fn seeded_sums_f32_every_n_t() {
    seeded::<f32>("seeded_sums_f32_every_n_t", usize::MAX, 1);
}

#[test]
#[ignore = "every n_t at 4,096 sources; run in release with --ignored"]
fn seeded_sums_f64_every_n_t() {
    seeded::<f64>("seeded_sums_f64_every_n_t", usize::MAX, 1);
}

/// A cancelling set: 1,000 sources in [−1, 1)³ whose charges sum to nearly zero (the
/// last one is minus the sum of the others), seen from targets at a distance, where
/// the potential and gradient are small against their term magnitudes.
fn cancelling<T: Precision>(test: &str) {
    let mut problem: Problem<T> = uniform(1000, 3 * BLOCK + 1, 0x00ca_9ce1);
    let rest: f64 = problem.charges[..999].iter().map(|&q| q.to_f64()).sum();
    problem.charges[999] = T::from_f64(-rest);
    let mut rng = Rng(0x00ca_9ce2);
    for x in &mut problem.targets {
        x[0] = T::from_f64(rng.uniform(2.0, 4.0));
    }
    let oracle = Oracle::new(&problem);
    let total: f64 = problem.charges.iter().map(|&q| q.to_f64()).sum();
    for with_gradient in [false, true] {
        let mut worst = Worst::new(format!(
            "{test} ({}, Σq = {total:.1e}, gradient: {with_gradient})",
            std::any::type_name::<T>()
        ));
        for kernel in kernels::<T>(test) {
            let errors = oracle.errors(&run(kernel, &problem, with_gradient));
            assert_within(errors, oracle.tolerance::<T>(), &kernel.isa());
            worst.record_sum::<T>(kernel.isa(), errors, &oracle, with_gradient);
        }
        worst.print(1.0, "(relative to the term magnitudes)");
    }
}

#[test]
fn cancelling_f32() {
    cancelling::<f32>("cancelling_f32");
}

#[test]
fn cancelling_f64() {
    cancelling::<f64>("cancelling_f64");
}

/// FMM-shaped sets: a target leaf of n points and its 26 neighbours, on one level and
/// on mixed levels. The per-pair form (27 calls into the same outputs, as
/// `LaplaceOperator` makes them) equals the gathered form (one call) bit for bit, and
/// both are within the tolerance of `direct_sum`.
fn fmm<T: Precision>(test: &str) {
    let kernels = kernels::<T>(test);
    for with_gradient in [false, true] {
        let mut worst = Worst::new(format!(
            "{test} ({}, gradient: {with_gradient})",
            std::any::type_name::<T>()
        ));
        for (seed, n) in [1, 8, 13, 20, 24, 44, 64].into_iter().enumerate() {
            for mixed in [false, true] {
                let (leaves, targets) = fmm_shaped::<T>(n, mixed, 0xf33_0000 + seed as u64);
                let (sources, charges) = gathered(&leaves);
                let problem = Problem {
                    sources,
                    charges,
                    potential: vec![T::zero(); n],
                    gradient: vec![[T::zero(); 3]; n],
                    targets,
                };
                let oracle = Oracle::new(&problem);
                for &kernel in &kernels {
                    let gathered = run(kernel, &problem, with_gradient);
                    let (mut potential, mut gradient) = (
                        problem.potential.clone(),
                        with_gradient.then(|| problem.gradient.clone()),
                    );
                    for (points, q) in &leaves {
                        kernel.evaluate(
                            points,
                            q,
                            &problem.targets,
                            &mut potential,
                            gradient.as_deref_mut(),
                        );
                    }
                    let what = format_args!("{} n = {n}, mixed: {mixed}", kernel.isa());
                    assert_eq!(
                        crate::common::bits(&(potential, gradient)),
                        crate::common::bits(&gathered),
                        "{what}: per-pair and gathered differ"
                    );
                    let errors = oracle.errors(&gathered);
                    assert_within(errors, oracle.tolerance::<T>(), &what);
                    worst.record_sum::<T>(kernel.isa(), errors, &oracle, with_gradient);
                }
            }
        }
        worst.print(1.0, "(relative to the term magnitudes)");
    }
}

#[test]
fn fmm_shaped_f32() {
    fmm::<f32>("fmm_shaped_f32");
}

#[test]
fn fmm_shaped_f64() {
    fmm::<f64>("fmm_shaped_f64");
}

/// Random problems (grid points in [−2, 2]³, coincident pairs and duplicated sources,
/// nonzero initial outputs) within the sum tolerance of `direct_sum`.
fn random_sums<T: Precision>(test: &str) {
    for kernel in kernels::<T>(test) {
        let worst = std::cell::RefCell::new(Worst::new(format!(
            "{test} ({}, potential and gradient)",
            std::any::type_name::<T>()
        )));
        runner()
            .run(&problem(64, 3 * BLOCK + 1), |raw| {
                let mut problem = raw.cast::<T>();
                let half = T::from_f64(0.5);
                for p in problem.sources.iter_mut().chain(&mut problem.targets) {
                    *p = p.map(|c| c * half);
                }
                let oracle = Oracle::new(&problem);
                for with_gradient in [false, true] {
                    let (ep, eg) = oracle.errors(&run(kernel, &problem, with_gradient));
                    let (tp, tg) = oracle.tolerance::<T>();
                    prop_assert!(ep <= tp && eg <= tg, "{}: {:e}, {:e}", kernel.isa(), ep, eg);
                    worst.borrow_mut().record_sum::<T>(
                        kernel.isa(),
                        (ep, eg),
                        &oracle,
                        with_gradient,
                    );
                }
                Ok(())
            })
            .unwrap();
        worst.borrow().print(
            1.0,
            "(relative to the initial value and the term magnitudes)",
        );
    }
}

#[test]
fn random_sums_f32() {
    random_sums::<f32>("random_sums_f32");
}

#[test]
fn random_sums_f64() {
    random_sums::<f64>("random_sums_f64");
}

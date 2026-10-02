//! The scalar path, which every ISA runs in this version, against
//! `nd_fmm_ref::p2p::p2p`: the same formula in the same order, so bit for bit,
//! accumulating onto the nonzero initial outputs of each problem.

use nd_fmm_simd::SimdScalar;

use crate::common::{bits, kernels, problem, run, run_reference, runner, seeded};

/// Seeded problems: every n_t from 0 to 17 (each n_t mod W for W ≤ 8, twice) against
/// several n_s, potential only and with gradient.
fn seeded_problems_match<T: SimdScalar>(test: &str) {
    for kernel in kernels::<T>(test) {
        for (seed, n_sources) in [1, 2, 7, 64, 1000].into_iter().enumerate() {
            for n_targets in 0..=17 {
                let raw = seeded(n_sources, n_targets, (seed * 100 + n_targets) as u64);
                let problem = raw.cast::<T>();
                for with_gradient in [false, true] {
                    assert_eq!(
                        bits(&run(kernel, &problem, with_gradient)),
                        bits(&run_reference(&problem, with_gradient)),
                        "{} n_s = {n_sources}, n_t = {n_targets}, gradient: {with_gradient}",
                        kernel.isa()
                    );
                }
            }
        }
    }
}

#[test]
fn seeded_problems_match_reference_f32() {
    seeded_problems_match::<f32>("seeded_problems_match_reference_f32");
}

#[test]
fn seeded_problems_match_reference_f64() {
    seeded_problems_match::<f64>("seeded_problems_match_reference_f64");
}

/// Random problems with coincident pairs and duplicated sources.
fn random_problems_match<T: SimdScalar>(test: &str) {
    for kernel in kernels::<T>(test) {
        runner()
            .run(&problem(40, 17), |raw| {
                let problem = raw.cast::<T>();
                for with_gradient in [false, true] {
                    assert_eq!(
                        bits(&run(kernel, &problem, with_gradient)),
                        bits(&run_reference(&problem, with_gradient)),
                        "{} gradient: {with_gradient}",
                        kernel.isa()
                    );
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn random_problems_match_reference_f32() {
    random_problems_match::<f32>("random_problems_match_reference_f32");
}

#[test]
fn random_problems_match_reference_f64() {
    random_problems_match::<f64>("random_problems_match_reference_f64");
}

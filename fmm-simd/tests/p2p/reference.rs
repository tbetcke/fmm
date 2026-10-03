//! The scalar path against `nd_fmm_ref::p2p::p2p`: the same formula in the same
//! order, so bit for bit wherever r² ≠ 0, accumulating onto the nonzero initial
//! outputs of each problem. The vector paths differ from the reference within the
//! accuracy of `terms` and `sums`.

use nd_fmm_simd::SimdScalar;

use crate::common::{BLOCK, bits, problem, run, run_reference, runner, scalar_kernel, seeded};

/// Seeded problems: every n_t from 0 to 3 · BLOCK + 1 against several n_s, potential
/// only and with gradient.
fn seeded_problems_match<T: SimdScalar>(test: &str) {
    let kernel = scalar_kernel::<T>(test);
    for (seed, n_sources) in [1, 2, 7, 64, 1000].into_iter().enumerate() {
        for n_targets in 0..=3 * BLOCK + 1 {
            let raw = seeded(n_sources, n_targets, (seed * 100 + n_targets) as u64);
            let problem = raw.cast::<T>();
            for with_gradient in [false, true] {
                assert_eq!(
                    bits(&run(kernel, &problem, with_gradient)),
                    bits(&run_reference(&problem, with_gradient)),
                    "n_s = {n_sources}, n_t = {n_targets}, gradient: {with_gradient}"
                );
            }
        }
    }
}

#[test]
fn scalar_matches_reference_on_seeded_problems_f32() {
    seeded_problems_match::<f32>("scalar_matches_reference_on_seeded_problems_f32");
}

#[test]
fn scalar_matches_reference_on_seeded_problems_f64() {
    seeded_problems_match::<f64>("scalar_matches_reference_on_seeded_problems_f64");
}

/// Random problems with coincident pairs and duplicated sources.
fn random_problems_match<T: SimdScalar>(test: &str) {
    let kernel = scalar_kernel::<T>(test);
    runner()
        .run(&problem(40, 17), |raw| {
            let problem = raw.cast::<T>();
            for with_gradient in [false, true] {
                assert_eq!(
                    bits(&run(kernel, &problem, with_gradient)),
                    bits(&run_reference(&problem, with_gradient)),
                    "gradient: {with_gradient}"
                );
            }
            Ok(())
        })
        .unwrap();
}

#[test]
fn scalar_matches_reference_on_random_problems_f32() {
    random_problems_match::<f32>("scalar_matches_reference_on_random_problems_f32");
}

#[test]
fn scalar_matches_reference_on_random_problems_f64() {
    random_problems_match::<f64>("scalar_matches_reference_on_random_problems_f64");
}

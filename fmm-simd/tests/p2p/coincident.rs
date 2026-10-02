//! Coincident points: targets equal to the sources, duplicated sources, and signed
//! zeros. The pairs with r² = 0 contribute nothing, as the reference's coincident pairs
//! do, so the results equal the reference's bit for bit.

use nd_fmm_simd::{P2pKernel, SimdScalar};

use crate::common::{RawProblem, bits, kernels, run, run_reference, seeded};

/// Checks `raw` against the reference with each of `kernels`, potential only and with
/// gradient.
fn check<T: SimdScalar>(kernels: &[P2pKernel<T>], raw: &RawProblem) {
    let problem = raw.cast::<T>();
    for &kernel in kernels {
        for with_gradient in [false, true] {
            assert_eq!(
                bits(&run(kernel, &problem, with_gradient)),
                bits(&run_reference(&problem, with_gradient)),
                "{} gradient: {with_gradient}",
                kernel.isa()
            );
        }
    }
}

/// Targets equal to the sources: every self-interaction pair is skipped.
fn targets_are_sources<T: SimdScalar>(test: &str) {
    let kernels = kernels::<T>(test);
    for n in [1, 2, 5, 16, 33] {
        let mut raw = seeded(n, n, 7 + n as u64);
        raw.targets = raw.sources.clone();
        check(&kernels, &raw);
    }
}

#[test]
fn targets_equal_to_sources_f32() {
    targets_are_sources::<f32>("targets_equal_to_sources_f32");
}

#[test]
fn targets_equal_to_sources_f64() {
    targets_are_sources::<f64>("targets_equal_to_sources_f64");
}

/// Each source three times, and targets at the sources and elsewhere: every copy of a
/// source that coincides with a target is skipped.
fn duplicated_sources<T: SimdScalar>(test: &str) {
    let mut raw = seeded(6, 4, 11);
    raw.sources = raw.sources.repeat(3);
    raw.charges = raw.charges.repeat(3);
    raw.targets.extend_from_slice(&raw.sources[..5]);
    raw.potential.extend_from_slice(&[3, -2, 1, 0, 5]);
    raw.gradient.extend_from_slice(&[[1, 2, 3]; 5]);
    check(&kernels::<T>(test), &raw);
}

#[test]
fn duplicated_sources_f32() {
    duplicated_sources::<f32>("duplicated_sources_f32");
}

#[test]
fn duplicated_sources_f64() {
    duplicated_sources::<f64>("duplicated_sources_f64");
}

/// +0 and −0 compare equal and their difference is ±0, so (0, 0, 0) and (−0, 0, −0)
/// coincide under both rules.
fn signed_zeros<T: SimdScalar>(test: &str) {
    let zero = T::zero();
    let sources = [[-zero, zero, -zero], [T::one(), zero, zero]];
    let charges = [T::one(), T::from_f64(2.0)];
    let targets = [[zero, zero, zero]];
    for kernel in kernels::<T>(test) {
        let mut potential = [zero];
        let mut gradient = [[zero; 3]];
        kernel.evaluate(
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        let outputs = (potential.to_vec(), Some(gradient.to_vec()));
        let two = T::from_f64(2.0);
        let expected = (vec![two], Some(vec![[two, zero, zero]]));
        assert_eq!(bits(&outputs), bits(&expected), "{}", kernel.isa());
    }
}

#[test]
fn signed_zeros_coincide_f32() {
    signed_zeros::<f32>("signed_zeros_coincide_f32");
}

#[test]
fn signed_zeros_coincide_f64() {
    signed_zeros::<f64>("signed_zeros_coincide_f64");
}

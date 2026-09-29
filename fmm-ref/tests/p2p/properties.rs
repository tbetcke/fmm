//! Random properties: random points, charges and coincidences, 64 cases each to keep
//! CI time low. Each property reuses the error measure and tolerance of the
//! deterministic test it generalises.

use nd_fmm_ref::p2p::{direct_sum, p2p};
use proptest::prelude::*;

use crate::common::{
    bits, dd_reference, gradient_bits, gradient_error, magnitudes, naive, potential_error,
};

/// Tolerance of `direct_sum` against the double-double reference, relative to the sum
/// of term magnitudes (as in `brute` and `cancellation`).
const TOL: f64 = 1e-15;

fn point() -> impl Strategy<Value = [f64; 3]> {
    [-10.0..10.0f64, -10.0..10.0f64, -10.0..10.0f64]
}

fn charge() -> impl Strategy<Value = f64> {
    prop_oneof![-1.0..-0.01f64, 0.01..1.0f64]
}

/// Up to 12 charged sources and up to 6 targets, some of which are copies of sources
/// (chosen by index modulo the number of sources).
fn set() -> impl Strategy<Value = (Vec<[f64; 3]>, Vec<f64>, Vec<[f64; 3]>)> {
    let sources = prop::collection::vec((point(), charge()), 1..=12);
    let targets = prop::collection::vec(
        prop_oneof![point().prop_map(Err), any::<usize>().prop_map(Ok)],
        1..=6,
    );
    (sources, targets).prop_map(|(sources, targets)| {
        let (points, charges): (Vec<_>, Vec<_>) = sources.into_iter().unzip();
        let targets = targets
            .into_iter()
            .map(|t| match t {
                Ok(index) => points[index % points.len()],
                Err(x) => x,
            })
            .collect();
        (points, charges, targets)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn p2p_equals_naive_loop((sources, charges, targets) in set()) {
        // Error measure: bit-for-bit equality of potential and gradient.
        let n = targets.len();
        let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
        naive(&sources, &charges, &targets, &mut phi, &mut grad);
        let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
        p2p(&sources, &charges, &targets, &mut potential, Some(&mut gradient));
        prop_assert_eq!(bits(&potential), bits(&phi));
        prop_assert_eq!(gradient_bits(&gradient), gradient_bits(&grad));
    }

    #[test]
    fn direct_sum_matches_double_double((sources, charges, targets) in set()) {
        // Error measure: per target, |error| / Σⱼ |qⱼ| / rᵢⱼ (potential) and
        // / Σⱼ |qⱼ| / rᵢⱼ² (gradient components), against `common::dd_reference`.
        let n = targets.len();
        let (reference, reference_gradient) = dd_reference(&sources, &charges, &targets);
        let (scale, gradient_scale) = magnitudes(&sources, &charges, &targets);
        let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
        direct_sum(&sources, &charges, &targets, &mut potential, Some(&mut gradient));
        prop_assert!(potential.iter().all(|v| v.is_finite()));
        prop_assert!(potential_error(&potential, &reference, &scale) <= TOL);
        prop_assert!(gradient_error(&gradient, &reference_gradient, &gradient_scale) <= TOL);
    }

    #[test]
    fn unit_charges_are_symmetric(x in point(), y in point()) {
        // Error measure: bit-for-bit equality of the potentials; gradients with ==
        // (see `symmetry`).
        prop_assume!(x != y);
        let (mut phi, mut grad) = ([0.0; 2], [[0.0; 3]; 2]);
        p2p(&[y], &[1.0], &[x], &mut phi[..1], Some(&mut grad[..1]));
        p2p(&[x], &[1.0], &[y], &mut phi[1..], Some(&mut grad[1..]));
        prop_assert_eq!(phi[0].to_bits(), phi[1].to_bits());
        prop_assert_eq!(grad[0], grad[1].map(|g| -g));
    }
}

//! Brute force on small sets (up to 64 points): `p2p` in f64 equals the naive double
//! loop bit for bit, and `direct_sum` agrees with it to 1e-15 relative to the sum of
//! term magnitudes.

use nd_fmm_ref::p2p::{direct_sum, p2p};

use crate::common::{
    CUBES, Set, SplitMix64, Worst, bits, gradient_bits, gradient_error, magnitudes, naive,
    potential_error,
};

/// Tolerance of `direct_sum` against the naive loop, relative to the sum of term
/// magnitudes.
const TOL: f64 = 1e-15;

/// (sources, targets) counts of the brute-force sets.
const SIZES: [(usize, usize); 6] = [(1, 1), (1, 64), (64, 1), (7, 13), (33, 64), (64, 64)];

/// Every brute-force set: each size in each cube of [`CUBES`], sources and targets
/// drawn independently.
fn sets() -> Vec<Set> {
    let mut rng = SplitMix64::new(0x7404);
    let mut sets = Vec::new();
    for (centre, half) in CUBES {
        for (ns, nt) in SIZES {
            let (sources, charges) = rng.charged(ns, centre, half);
            let targets = (0..nt).map(|_| rng.in_cube(centre, half)).collect();
            sets.push(Set {
                sources,
                charges,
                targets,
            });
        }
    }
    sets
}

#[test]
fn p2p_equals_naive_loop_bit_for_bit() {
    // Error measure: bit-for-bit equality of potential and gradient.
    for Set {
        sources,
        charges,
        targets,
    } in sets()
    {
        let n = targets.len();
        let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
        naive(&sources, &charges, &targets, &mut phi, &mut grad);

        let mut potential = vec![0.0; n];
        p2p(&sources, &charges, &targets, &mut potential, None);
        assert_eq!(bits(&potential), bits(&phi));

        let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
        p2p(
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        assert_eq!(bits(&potential), bits(&phi));
        assert_eq!(gradient_bits(&gradient), gradient_bits(&grad));
    }
}

#[test]
fn direct_sum_agrees_with_naive_loop() {
    // Error measure: per target, |direct_sum − naive| / Σⱼ |qⱼ| / rᵢⱼ for the potential
    // and per component / Σⱼ |qⱼ| / rᵢⱼ² for the gradient.
    let mut worst = [
        Worst::new("direct_sum vs naive loop, potential"),
        Worst::new("direct_sum vs naive loop, gradient"),
    ];
    for Set {
        sources,
        charges,
        targets,
    } in sets()
    {
        let n = targets.len();
        let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
        naive(&sources, &charges, &targets, &mut phi, &mut grad);
        let (scale, gradient_scale) = magnitudes(&sources, &charges, &targets);

        let mut potential = vec![0.0; n];
        direct_sum(&sources, &charges, &targets, &mut potential, None);
        let e = worst[0].update(potential_error(&potential, &phi, &scale));
        assert!(e <= TOL, "potential error {e:e}");

        let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
        direct_sum(
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        let e = worst[0].update(potential_error(&potential, &phi, &scale));
        assert!(e <= TOL, "potential error {e:e}");
        let e = worst[1].update(gradient_error(&gradient, &grad, &gradient_scale));
        assert!(e <= TOL, "gradient error {e:e}");
    }
}

#[test]
fn gradient_does_not_change_potential() {
    // Error measure: bit-for-bit equality. Asking for the gradient must not change the
    // potential of `direct_sum`; for `p2p` the bit-for-bit test above shows it.
    for Set {
        sources,
        charges,
        targets,
    } in sets()
    {
        let n = targets.len();
        let mut plain = vec![0.0; n];
        let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
        direct_sum(&sources, &charges, &targets, &mut plain, None);
        direct_sum(
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        assert_eq!(bits(&plain), bits(&potential));
    }
}

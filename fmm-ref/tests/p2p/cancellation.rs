//! Cancellation: charges that sum to nearly zero, so the potential is far smaller than
//! the sum of its term magnitudes. `direct_sum` stays within 1e-15 of the
//! double-double reference, relative to the sum of term magnitudes.

use nd_fmm_ref::p2p::{direct_sum, p2p};

use crate::common::{SplitMix64, Worst, dd_reference, gradient_error, magnitudes, potential_error};

/// Tolerance of `direct_sum` against the double-double reference, relative to the sum
/// of term magnitudes.
const TOL: f64 = 1e-15;

/// Neutral charge sets in the unit cube [−1, 1]³: `pairs` dipoles of opposite charges
/// ±q at distance 10⁻³, and `pairs` random charges whose last one balances the f64 sum
/// of the others.
fn neutral_sets(rng: &mut SplitMix64, pairs: usize) -> [(Vec<[f64; 3]>, Vec<f64>); 2] {
    let (mut dipoles, mut dipole_charges) = (Vec::new(), Vec::new());
    for _ in 0..pairs {
        let y = rng.in_cube([0.0; 3], 1.0);
        let q = rng.charge();
        dipoles.extend([y, rng.in_cube(y, 1e-3)]);
        dipole_charges.extend([q, -q]);
    }
    let (balanced, mut charges) = rng.charged(pairs, [0.0; 3], 1.0);
    let rest: f64 = charges[..pairs - 1].iter().sum();
    charges[pairs - 1] = -rest;
    [(dipoles, dipole_charges), (balanced, charges)]
}

/// Targets near the sources (in the cube) and far from them (at distance 10 to 100).
fn targets(rng: &mut SplitMix64) -> Vec<[f64; 3]> {
    let near = (0..8)
        .map(|_| rng.in_cube([0.0; 3], 1.0))
        .collect::<Vec<_>>();
    let far = (0..8).map(|_| {
        let x = rng.in_cube([0.0; 3], 1.0);
        let r = rng.range(10.0, 100.0) / x.iter().map(|c| c * c).sum::<f64>().sqrt();
        x.map(|c| r * c)
    });
    near.into_iter().chain(far).collect()
}

#[test]
fn direct_sum_survives_cancellation() {
    // Error measure: per target, |direct_sum − reference| / Σⱼ |qⱼ| / rᵢⱼ for the
    // potential and per component / Σⱼ |qⱼ| / rᵢⱼ² for the gradient. The reference is
    // computed in double-double arithmetic (`common::dd_reference`). The errors of the
    // plain `p2p` sum, and of `direct_sum` relative to |φ| itself, are only reported.
    let mut worst = [
        Worst::new("direct_sum vs double-double under cancellation, potential"),
        Worst::new("direct_sum vs double-double under cancellation, gradient"),
        Worst::new("p2p (plain sum) vs double-double under cancellation, potential (reported)"),
        Worst::new("direct_sum under cancellation, relative to |phi| (reported)"),
    ];
    let mut rng = SplitMix64::new(0xca2ce1);
    for pairs in [16, 256, 4096] {
        let targets = targets(&mut rng);
        for (sources, charges) in neutral_sets(&mut rng, pairs) {
            let n = targets.len();
            let (reference, reference_gradient) = dd_reference(&sources, &charges, &targets);
            let (scale, gradient_scale) = magnitudes(&sources, &charges, &targets);

            let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
            direct_sum(
                &sources,
                &charges,
                &targets,
                &mut potential,
                Some(&mut gradient),
            );
            let e = worst[0].update(potential_error(&potential, &reference, &scale));
            assert!(e <= TOL, "{pairs} pairs: potential error {e:e}");
            let e = worst[1].update(gradient_error(
                &gradient,
                &reference_gradient,
                &gradient_scale,
            ));
            assert!(e <= TOL, "{pairs} pairs: gradient error {e:e}");

            let magnitude: Vec<f64> = reference.iter().map(|r| r.abs()).collect();
            worst[3].update(potential_error(&potential, &reference, &magnitude));

            let mut plain = vec![0.0; n];
            p2p(&sources, &charges, &targets, &mut plain, None);
            worst[2].update(potential_error(&plain, &reference, &scale));
        }
    }
}

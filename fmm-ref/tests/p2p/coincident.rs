//! Coincident points: targets equal to the sources (the same slice), and duplicated
//! sources. Results are finite and equal the sum over the non-coincident pairs, which
//! the tests select by index, not by comparing coordinates.

use nd_fmm_ref::p2p::{direct_sum, p2p};

use crate::common::{
    CUBES, SplitMix64, Worst, bits, dd_reference, gradient_bits, gradient_error, magnitudes, naive,
    potential_error,
};

/// Tolerance of `direct_sum` against the double-double reference, relative to the sum
/// of term magnitudes.
const TOL: f64 = 1e-15;

/// Checks both functions against the sum over the pairs (i, j) with `keep(i, j)`:
/// finite results; `p2p` equal bit for bit to the naive loop over the kept sources of
/// each target; `direct_sum` within [`TOL`] of the double-double reference over them.
fn check(
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
    keep: impl Fn(usize, usize) -> bool,
    worst: &mut [Worst; 2],
) {
    let n = targets.len();
    let (mut p2p_phi, mut p2p_grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
    p2p(sources, charges, targets, &mut p2p_phi, Some(&mut p2p_grad));
    let (mut sum_phi, mut sum_grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
    direct_sum(sources, charges, targets, &mut sum_phi, Some(&mut sum_grad));

    for (i, &x) in targets.iter().enumerate() {
        let kept: Vec<usize> = (0..sources.len()).filter(|&j| keep(i, j)).collect();
        let s: Vec<[f64; 3]> = kept.iter().map(|&j| sources[j]).collect();
        let q: Vec<f64> = kept.iter().map(|&j| charges[j]).collect();

        assert!(p2p_phi[i].is_finite() && p2p_grad[i].iter().all(|g| g.is_finite()));
        assert!(sum_phi[i].is_finite() && sum_grad[i].iter().all(|g| g.is_finite()));

        let (mut phi, mut grad) = ([0.0], [[0.0; 3]]);
        naive(&s, &q, &[x], &mut phi, &mut grad);
        assert_eq!(bits(&p2p_phi[i..=i]), bits(&phi), "target {i}");
        assert_eq!(
            gradient_bits(&p2p_grad[i..=i]),
            gradient_bits(&grad),
            "target {i}"
        );

        let (reference, reference_gradient) = dd_reference(&s, &q, &[x]);
        let (scale, gradient_scale) = magnitudes(&s, &q, &[x]);
        let e = worst[0].update(potential_error(&sum_phi[i..=i], &reference, &scale));
        assert!(e <= TOL, "target {i}: potential error {e:e}");
        let e = worst[1].update(gradient_error(
            &sum_grad[i..=i],
            &reference_gradient,
            &gradient_scale,
        ));
        assert!(e <= TOL, "target {i}: gradient error {e:e}");
    }
}

#[test]
fn targets_equal_to_sources() {
    // Error measure: bit-for-bit equality for `p2p`; for `direct_sum`, per target
    // |error| / Σⱼ |qⱼ| / rᵢⱼ (potential) and / Σⱼ |qⱼ| / rᵢⱼ² (gradient components)
    // against the double-double sum over j ≠ i.
    let mut worst = [
        Worst::new("direct_sum, targets = sources, potential"),
        Worst::new("direct_sum, targets = sources, gradient"),
    ];
    let mut rng = SplitMix64::new(0xc01c);
    for (centre, half) in CUBES {
        for n in [1, 2, 50] {
            let (points, charges) = rng.charged(n, centre, half);
            check(&points, &charges, &points, |i, j| i != j, &mut worst);
        }
    }
}

#[test]
fn duplicated_sources() {
    // Error measure: as in `targets_equal_to_sources`, against the sum over the
    // sources that are not copies of the target.
    //
    // Sources: 20 base points, a second copy of the first 10 and a third copy of the
    // first 5 (each copy with its own charge), and 10 further points. Targets: the 20
    // base points and 10 points that are not sources. Every copy of a base point is
    // skipped for that target, and counted for every other target.
    let mut worst = [
        Worst::new("direct_sum, duplicated sources, potential"),
        Worst::new("direct_sum, duplicated sources, gradient"),
    ];
    let mut rng = SplitMix64::new(0xd0b1e);
    for (centre, half) in CUBES {
        let base: Vec<[f64; 3]> = (0..20).map(|_| rng.in_cube(centre, half)).collect();
        let mut sources = base.clone();
        let mut origin: Vec<Option<usize>> = (0..20).map(Some).collect();
        for copies in [10, 5] {
            sources.extend_from_slice(&base[..copies]);
            origin.extend((0..copies).map(Some));
        }
        sources.extend((0..10).map(|_| rng.in_cube(centre, half)));
        origin.extend([None; 10]);
        let charges: Vec<f64> = sources.iter().map(|_| rng.charge()).collect();

        let mut targets = base.clone();
        targets.extend((0..10).map(|_| rng.in_cube(centre, half)));

        check(
            &sources,
            &charges,
            &targets,
            |i, j| origin[j] != Some(i),
            &mut worst,
        );
    }
}

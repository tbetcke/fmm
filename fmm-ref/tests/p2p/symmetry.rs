//! Symmetry: a unit charge at y seen at x equals a unit charge at x seen at y, bit for
//! bit, and the gradients are exact negatives of each other.

use nd_fmm_ref::p2p::{direct_sum, p2p};

use crate::common::SplitMix64;

/// Random pairs (x, y): both in the unit cube, far apart, and nearly coincident at a
/// large offset from the origin.
fn pairs() -> Vec<([f64; 3], [f64; 3])> {
    let mut rng = SplitMix64::new(0x5e7);
    let mut pairs = Vec::new();
    for _ in 0..200 {
        pairs.push((rng.in_cube([0.0; 3], 1.0), rng.in_cube([0.0; 3], 1.0)));
        pairs.push((
            rng.in_cube([0.0; 3], 1.0),
            rng.in_cube([50.0, -20.0, 7.0], 30.0),
        ));
        let x = rng.in_cube([1000.0, -250.0, 64.0], 1.0);
        pairs.push((x, rng.in_cube(x, 1e-9)));
    }
    // Pairs that share coordinates, so that some components of x − y are zero.
    pairs.push(([1.0, 2.0, 3.0], [1.0, 2.0, 4.5]));
    pairs.push(([0.0, 0.0, 0.0], [-0.0, 7.0, 0.0]));
    pairs
}

/// Potential and gradient at `x` of a unit charge at `y`, from `p2p` in f64.
fn seen_at_f64(x: [f64; 3], y: [f64; 3]) -> (f64, [f64; 3]) {
    let (mut phi, mut grad) = ([0.0], [[0.0; 3]]);
    p2p(&[y], &[1.0], &[x], &mut phi, Some(&mut grad));
    (phi[0], grad[0])
}

/// As [`seen_at_f64`], from `direct_sum`.
fn seen_at_direct(x: [f64; 3], y: [f64; 3]) -> (f64, [f64; 3]) {
    let (mut phi, mut grad) = ([0.0], [[0.0; 3]]);
    direct_sum(&[y], &[1.0], &[x], &mut phi, Some(&mut grad));
    (phi[0], grad[0])
}

/// As [`seen_at_f64`], from `p2p` in f32.
fn seen_at_f32(x: [f32; 3], y: [f32; 3]) -> (f32, [f32; 3]) {
    let (mut phi, mut grad) = ([0.0], [[0.0; 3]]);
    p2p(&[y], &[1.0], &[x], &mut phi, Some(&mut grad));
    (phi[0], grad[0])
}

#[test]
fn unit_charges_are_symmetric() {
    // Error measure: bit-for-bit equality of the potentials. Gradients are compared
    // with ==, which is bit-for-bit equality except for the sign of zero components:
    // x − y = +0 when the coordinates are equal, so such a component is −0 in both
    // directions.
    for (x, y) in pairs() {
        for seen_at in [seen_at_f64, seen_at_direct] {
            let (phi_xy, g_xy) = seen_at(x, y);
            let (phi_yx, g_yx) = seen_at(y, x);
            assert_eq!(phi_xy.to_bits(), phi_yx.to_bits(), "x = {x:?}, y = {y:?}");
            assert_eq!(g_xy, g_yx.map(|g| -g), "x = {x:?}, y = {y:?}");
        }
        let (x32, y32) = (x.map(|c| c as f32), y.map(|c| c as f32));
        if x32 != y32 {
            let (phi_xy, g_xy) = seen_at_f32(x32, y32);
            let (phi_yx, g_yx) = seen_at_f32(y32, x32);
            assert_eq!(
                phi_xy.to_bits(),
                phi_yx.to_bits(),
                "x = {x32:?}, y = {y32:?}"
            );
            assert_eq!(g_xy, g_yx.map(|g| -g), "x = {x32:?}, y = {y32:?}");
        }
    }
}

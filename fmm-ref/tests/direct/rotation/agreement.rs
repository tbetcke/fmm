//! Rotation against direct on general shifts: random frames (proptest), the 8 octant
//! children, the 316 same-level offsets, and M2M and L2L at high degree.

use nd_fmm_ref::{Frame, Workspace};
use proptest::prelude::*;

use super::{P_GATE, Tracker, compare, tolerance};
use crate::common::{Op, P_MAX, SQRT3, SplitMix64, frames, nested, octants, place};
use crate::common::{random_coefficients, sub};
use crate::m2l::offsets;
use crate::properties::{config, direction, frame, nest, nesting};

/// Random input coefficients of the kind of `op`'s input, from `seed`.
fn input(op: Op, p: usize, seed: u64) -> Vec<f64> {
    random_coefficients(op.input(), p, &mut SplitMix64::new(seed))
}

#[test]
fn m2m_agrees_with_direct_on_random_frames() {
    let worst = Tracker::new(
        "rotation vs direct, random frames, M2M, per degree / terms",
        "rotation vs direct, random frames, M2M, per degree / own norm",
    );
    proptest!(config(), |(
        parent in frame(),
        nesting in nesting(),
        p in 0..=P_GATE,
        seed in any::<u64>(),
    )| {
        // Error measure: per degree, weighted, relative to the term magnitudes (module
        // documentation); the child is nested in the parent, as on an octree.
        let mut ws = Workspace::new(p);
        let child = nest(&parent, nesting);
        let x = input(Op::M2m, p, seed);
        let e = worst.update(&compare(Op::M2m, p, &child, &parent, &mut ws, &x));
        prop_assert!(e <= tolerance(p), "p = {p}: {e:.3e}");
    });
}

#[test]
fn l2l_agrees_with_direct_on_random_frames() {
    let worst = Tracker::new(
        "rotation vs direct, random frames, L2L, per degree / terms",
        "rotation vs direct, random frames, L2L, per degree / own norm",
    );
    proptest!(config(), |(
        parent in frame(),
        nesting in nesting(),
        p in 0..=P_GATE,
        seed in any::<u64>(),
    )| {
        // Error measure: per degree, weighted, relative to the term magnitudes (module
        // documentation); the child is nested in the parent, as on an octree.
        let mut ws = Workspace::new(p);
        let child = nest(&parent, nesting);
        let x = input(Op::L2l, p, seed);
        let e = worst.update(&compare(Op::L2l, p, &parent, &child, &mut ws, &x));
        prop_assert!(e <= tolerance(p), "p = {p}: {e:.3e}");
    });
}

#[test]
fn m2l_agrees_with_direct_on_random_frames() {
    let worst = Tracker::new(
        "rotation vs direct, random frames, M2L, per degree / terms",
        "rotation vs direct, random frames, M2L, per degree / own norm",
    );
    proptest!(config(), |(
        source in frame(),
        sigma in 0.5..=2.0f64,
        direction in direction(),
        margin in 1.1..=3.0f64,
        p in 0..=P_GATE,
        seed in any::<u64>(),
    )| {
        // Error measure: per degree, weighted, relative to the term magnitudes (module
        // documentation). The shift is |b| = margin · √3 (1 + σ), beyond the
        // convergence limit, as in the direct M2L properties.
        let mut ws = Workspace::new(p);
        let b = direction.map(|c| c * margin * SQRT3 * (1.0 + sigma));
        let target = Frame::new(place(&source, b), sigma * source.radius);
        let x = input(Op::M2l, p, seed);
        let e = worst.update(&compare(Op::M2l, p, &source, &target, &mut ws, &x));
        prop_assert!(e <= tolerance(p), "p = {p}: {e:.3e}");
    });
}

#[test]
fn m2m_and_l2l_agree_with_direct_on_octant_children() {
    // The 8 octant children (r′ = r/2, c′ − c = (r/2)(±1, ±1, ±1)) of the parents of
    // `frames` (radii 1, 0.37 and 2⁻¹⁶, two centres each): M2M child to parent and L2L
    // parent to child, for every p ≤ 30 (1e-13 up to p = 20, 1e-11 above). Above
    // p = 20 only the first centre of each radius is used, to bound the run time of
    // the O(p⁴) direct operators in debug builds.
    //
    // Error measure: per degree, weighted, relative to the term magnitudes (module
    // documentation).
    let mut rng = SplitMix64::new(0x7601);
    let mut ws = Workspace::new(P_MAX);
    let gate = Tracker::new(
        "rotation vs direct, octants, p <= 20, per degree / terms",
        "rotation vs direct, octants, p <= 20, per degree / own norm",
    );
    let high = Tracker::new(
        "rotation vs direct, octants, 20 < p <= 30, per degree / terms",
        "rotation vs direct, octants, 20 < p <= 30, per degree / own norm",
    );
    for (index, parent) in frames().into_iter().enumerate() {
        let p_max = if index % 2 == 0 { P_MAX } else { P_GATE };
        for child in octants(&parent) {
            for p in 0..=p_max {
                let worst = if p <= P_GATE { &gate } else { &high };
                for (op, from, to) in [(Op::M2m, child, parent), (Op::L2l, parent, child)] {
                    let x = random_coefficients(op.input(), p, &mut rng);
                    let e = worst.update(&compare(op, p, &from, &to, &mut ws, &x));
                    assert!(
                        e <= tolerance(p),
                        "{op:?} {from:?} -> {to:?}, p = {p}: {e:.3e}"
                    );
                }
            }
        }
    }
}

#[test]
fn translations_agree_with_direct_on_the_316_offsets() {
    // Same-level frames (r′ = r) with c′ − c = 2r · d for all 316 offsets d in
    // {−3..3}³ \ {−1..1}³, at p = 10 and p = 20: M2L, the V-list operator, and M2M
    // and L2L, which are exact for any two frames. The offsets cover the axis
    // directions ±x, ±y, ±z (no rotation for ±z) and every polar angle and azimuth of
    // the V list.
    //
    // Error measure: per degree, weighted, relative to the term magnitudes (module
    // documentation).
    let mut rng = SplitMix64::new(0x7602);
    let mut ws = Workspace::new(P_GATE);
    // In the order of `Op::ALL`.
    let worst = [
        Tracker::new(
            "rotation vs direct, 316 offsets, M2M, per degree / terms",
            "rotation vs direct, 316 offsets, M2M, per degree / own norm",
        ),
        Tracker::new(
            "rotation vs direct, 316 offsets, L2L, per degree / terms",
            "rotation vs direct, 316 offsets, L2L, per degree / own norm",
        ),
        Tracker::new(
            "rotation vs direct, 316 offsets, M2L, per degree / terms",
            "rotation vs direct, 316 offsets, M2L, per degree / own norm",
        ),
    ];
    let offsets = offsets();
    assert_eq!(offsets.len(), 316);
    let source = Frame::new([0.3, -1.1, 0.7], 0.37);
    for d in offsets {
        let target = Frame::new(place(&source, d.map(|c| 2.0 * c)), source.radius);
        for p in [10, P_GATE] {
            for (op, worst) in Op::ALL.into_iter().zip(&worst) {
                let x = random_coefficients(op.input(), p, &mut rng);
                let e = worst.update(&compare(op, p, &source, &target, &mut ws, &x));
                assert!(e <= tolerance(p), "{op:?}, d = {d:?}, p = {p}: {e:.3e}");
            }
        }
    }
}

#[test]
fn m2m_and_l2l_agree_with_direct_at_high_degree() {
    // 20 < p ≤ 30, M2M and L2L only (M2L stays at p ≤ 20, the range of fixture set C):
    // random nested frame pairs, 1e-11, the accuracy of the rotation blocks at n ≤ 30.
    //
    // Error measure: per degree, weighted, relative to the term magnitudes (module
    // documentation).
    let mut rng = SplitMix64::new(0x7603);
    let mut ws = Workspace::new(P_MAX);
    let worst = Tracker::new(
        "rotation vs direct, random nested frames, 20 < p <= 30, per degree / terms",
        "rotation vs direct, random nested frames, 20 < p <= 30, per degree / own norm",
    );
    for p in P_GATE + 1..=P_MAX {
        for _ in 0..4 {
            let centre = rng.in_shell(0.0, 5.0);
            let parent = Frame::new(centre, rng.range(0.01, 3.0));
            let child = nested(&parent, &mut rng);
            // The shift is never on the z axis, so both rotations run.
            assert!(sub(child.centre, parent.centre)[..2] != [0.0, 0.0]);
            for (op, from, to) in [(Op::M2m, child, parent), (Op::L2l, parent, child)] {
                let x = random_coefficients(op.input(), p, &mut rng);
                let e = worst.update(&compare(op, p, &from, &to, &mut ws, &x));
                assert!(e <= tolerance(p), "{op:?}, p = {p}: {e:.3e}");
            }
        }
    }
}

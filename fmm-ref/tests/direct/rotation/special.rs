//! Special shifts: zero, along ±z (the coaxial step alone, no rotation), near the
//! z axis and along ±x and ±y; accumulation onto initial values.

use nd_fmm_ref::{Frame, Workspace, direct};

use super::{COAXIAL_TOL, P_GATE, Tracker, compare, rotated, tolerance};
use crate::common::{
    Op, P_MAX, SplitMix64, Worst, degree_error, degree_norms, len, place, random_coefficients,
};
use crate::linearity::ACCUMULATION_TOL;

/// Tolerance of the zero shift against the radius rescaling, per degree relative to
/// the weighted norm of the expected result, as for `direct` (`special`).
const ZERO_SHIFT_TOL: f64 = 1e-15;

/// xᵏ by repeated multiplication.
fn power(x: f64, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, _| acc * x)
}

#[test]
fn zero_shift_is_the_radius_rescaling() {
    // With c′ = c no rotation runs, and the coaxial forms of CONVENTIONS §3.11 reduce
    // to M̃′ⱼ = ρʲ M̃ⱼ (M2M, ρ = r/r′) and L̃′ⱼ = σʲ⁺¹ L̃ⱼ (L2L, σ = r′/r), p ≤ 30.
    //
    // Error measures: against direct, per degree, weighted, relative to the term
    // magnitudes (module documentation), 1e-14; against the rescaling, per degree,
    // weighted, relative to the weighted norm of the rescaled input, 1e-15.
    let mut rng = SplitMix64::new(0x7604);
    let mut ws = Workspace::new(P_MAX);
    let worst = Tracker::new(
        "rotation vs direct, zero shift, per degree / terms",
        "rotation vs direct, zero shift, per degree / own norm",
    );
    let rescaling = Worst::new("rotation, zero shift vs radius rescaling, per degree");
    let centre = [0.3, -1.1, 0.7];
    for (r, r_out) in [
        (0.37, 0.74),
        (0.37, 0.37),
        (1.0, 0.3),
        (2f64.powi(-16), 3.0),
    ] {
        let (from, to) = (Frame::new(centre, r), Frame::new(centre, r_out));
        for p in 0..=P_MAX {
            let layout = nd_fmm_math::Layout::new(p);
            for op in [Op::M2m, Op::L2l] {
                let x = random_coefficients(op.input(), p, &mut rng);
                let e = worst.update(&compare(op, p, &from, &to, &mut ws, &x));
                assert!(e <= COAXIAL_TOL, "{op:?}, r′ = {r_out}, p = {p}: {e:.3e}");

                let got = rotated(op, p, &from, &to, &mut ws, &x);
                let want: Vec<f64> = (0..len(p))
                    .map(|i| {
                        let n = layout.nm(i).0;
                        let factor = match op {
                            Op::M2m => power(r / r_out, n),
                            _ => power(r_out / r, n + 1),
                        };
                        factor * x[i]
                    })
                    .collect();
                let scale = degree_norms(op.output(), p, &want);
                let e = rescaling.update(degree_error(op.output(), p, &got, &want, &scale));
                assert!(
                    e <= ZERO_SHIFT_TOL,
                    "{op:?}, r′ = {r_out}, p = {p}: {e:.3e}"
                );
            }
        }
    }
}

#[test]
fn m2l_with_coincident_centres_is_not_finite() {
    // Outside the domain of M2L, like `direct::m2l`: the axis value I₀⁰ = 1/|b| is
    // infinite. Error measure: none; the output must contain a non-finite value.
    let mut ws = Workspace::new(4);
    let frame = Frame::new([0.3, -1.1, 0.7], 0.37);
    let x = vec![1.0; len(4)];
    let got = rotated(Op::M2l, 4, &frame, &frame, &mut ws, &x);
    assert!(got.iter().any(|c| !c.is_finite()), "{got:?}");
    let mut want = vec![0.0; len(4)];
    direct::m2l(4, &frame, &frame, &mut ws, &x, &mut want);
    assert!(want.iter().any(|c| !c.is_finite()), "{want:?}");
}

/// Shifts along the z axis: (operator, output radius / r, (c′ − c)_z / r).
const Z_SHIFTS: [(Op, f64, f64); 12] = [
    (Op::M2m, 2.0, 1.0),
    (Op::M2m, 2.0, -1.0),
    (Op::M2m, 1.3, -0.4),
    (Op::M2m, 1.3, 0.4),
    (Op::L2l, 0.5, -0.5),
    (Op::L2l, 0.5, 0.5),
    (Op::L2l, 0.6, 0.3),
    (Op::L2l, 0.6, -0.3),
    (Op::M2l, 1.0, 4.0),
    (Op::M2l, 1.0, -6.0),
    (Op::M2l, 0.6, 3.5),
    (Op::M2l, 1.5, -5.0),
];

#[test]
fn coaxial_step_alone_equals_direct_on_the_z_axis() {
    // For c′ − c = d e_z the operators apply the coaxial form of CONVENTIONS §3.11
    // with no rotation: for d > 0 as the general case does after rotating, and for
    // d < 0 directly, which avoids the z-y-z Euler-angle singularity at θ = π. Both
    // signs, radius ratios ≠ 1, p ≤ 30 (M2L p ≤ 20).
    //
    // Error measure: per degree, weighted, relative to the term magnitudes (module
    // documentation), 1e-14.
    let mut rng = SplitMix64::new(0x7605);
    let mut ws = Workspace::new(P_MAX);
    let plus = Tracker::new(
        "coaxial step vs direct, +z, per degree / terms",
        "coaxial step vs direct, +z, per degree / own norm",
    );
    let minus = Tracker::new(
        "coaxial step vs direct, -z, per degree / terms",
        "coaxial step vs direct, -z, per degree / own norm",
    );
    let from = Frame::new([0.3, -1.1, 0.7], 0.37);
    for (op, ratio, shift) in Z_SHIFTS {
        let to = Frame::new(place(&from, [0.0, 0.0, shift]), ratio * from.radius);
        assert_eq!(to.centre[..2], from.centre[..2]);
        let worst = if shift > 0.0 { &plus } else { &minus };
        for p in 0..=op.p_max() {
            let x = random_coefficients(op.input(), p, &mut rng);
            let e = worst.update(&compare(op, p, &from, &to, &mut ws, &x));
            assert!(e <= COAXIAL_TOL, "{op:?}, shift {shift}, p = {p}: {e:.3e}");
        }
    }
}

#[test]
fn shifts_near_and_along_the_axes_agree_with_direct() {
    // Rotated shifts close to ±z, where the polar angle θ approaches 0 or π and the
    // azimuth φ is set by a tiny transverse offset (relative size 1e-6 and 1e-13),
    // and shifts along ±x and ±y (θ = π/2, φ = 0, π, ±π/2). The shifts of `Z_SHIFTS`
    // with the transverse part added; p ≤ 30 (M2L p ≤ 20), 1e-13 up to p = 20 and
    // 1e-11 above.
    //
    // Error measure: per degree, weighted, relative to the term magnitudes (module
    // documentation).
    let mut rng = SplitMix64::new(0x7606);
    let mut ws = Workspace::new(P_MAX);
    let near = Tracker::new(
        "rotation vs direct, near the z axis, per degree / terms",
        "rotation vs direct, near the z axis, per degree / own norm",
    );
    let axes = Tracker::new(
        "rotation vs direct, along x and y, per degree / terms",
        "rotation vs direct, along x and y, per degree / own norm",
    );
    let from = Frame::new([0.3, -1.1, 0.7], 0.37);
    for (op, ratio, shift) in Z_SHIFTS {
        let mut offsets = Vec::new();
        for eps in [1e-6, 1e-13] {
            let e = eps * shift.abs();
            offsets.push(([e, 0.0, shift], &near));
            offsets.push(([-0.6 * e, 0.8 * e, shift], &near));
        }
        for axis in [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ] {
            offsets.push((axis.map(|c| c * shift.abs()), &axes));
        }
        for (offset, worst) in offsets {
            let to = Frame::new(place(&from, offset), ratio * from.radius);
            assert_ne!(to.centre[..2], from.centre[..2], "{offset:?} must rotate");
            for p in (0..=op.p_max()).step_by(3).chain([P_GATE, op.p_max()]) {
                let x = random_coefficients(op.input(), p, &mut rng);
                let e = worst.update(&compare(op, p, &from, &to, &mut ws, &x));
                assert!(e <= tolerance(p), "{op:?}, {offset:?}, p = {p}: {e:.3e}");
            }
        }
    }
}

#[test]
fn rotation_accumulates_onto_initial_values() {
    // Every operator adds into its output. The result must equal the initial values
    // plus a fresh translation, and a zero input must leave the output unchanged
    // exactly. General shifts (`Op::pairs`) and a shift along −z.
    //
    // Error measure: per degree, weighted, relative to the term magnitudes plus the
    // weighted norm of the initial values, 1e-15, as for `direct` (`linearity`).
    let mut rng = SplitMix64::new(0x7607);
    let mut ws = Workspace::new(P_MAX);
    let worst = Worst::new("rotation accumulation, per degree");
    for op in Op::ALL {
        let mut pairs = op.pairs();
        let (from, _) = pairs[0];
        let shift = if let Op::M2l = op { -4.0 } else { -0.5 };
        pairs.push((
            from,
            Frame::new(place(&from, [0.0, 0.0, shift]), from.radius),
        ));
        for (from, to) in pairs {
            for p in [0, 1, 5, 12, op.p_max()] {
                let x = random_coefficients(op.input(), p, &mut rng);
                let initial = random_coefficients(op.output(), p, &mut rng);
                let fresh = rotated(op, p, &from, &to, &mut ws, &x);
                let mut accumulated = initial.clone();
                op.rotation()(p, &from, &to, &mut ws, &x, &mut accumulated);
                let want: Vec<f64> = initial.iter().zip(&fresh).map(|(a, b)| a + b).collect();
                let scale: Vec<f64> = op
                    .scale(p, &from, &to, &x)
                    .iter()
                    .zip(degree_norms(op.output(), p, &initial))
                    .map(|(s, i)| s + i)
                    .collect();
                let e = worst.update(degree_error(op.output(), p, &accumulated, &want, &scale));
                assert!(
                    e <= ACCUMULATION_TOL,
                    "{op:?} {from:?} -> {to:?}, p = {p}: {e:.3e}"
                );

                let mut unchanged = initial.clone();
                op.rotation()(p, &from, &to, &mut ws, &vec![0.0; len(p)], &mut unchanged);
                assert_eq!(unchanged, initial, "{op:?}, p = {p}: zero input");
            }
        }
    }
}

//! Special cases: zero shift, shifts along the coordinate axes, and p = 0 and p = 1
//! against values worked out by hand.

use nd_fmm_math::Layout;
use nd_fmm_ref::{Frame, Workspace, direct};

use crate::common::{
    EXACT_TOL, Kind, Op, P_MAX, P_MAX_M2L, SplitMix64, Worst, apply, degree_error, degree_norms,
    len, place, random_coefficients,
};
use crate::exactness::{l2l_l2p_scale, l2p_after_l2l, m2m_after_p2m, m2m_p2m_scale};
use crate::m2l::check_against_direct_sum;

/// Tolerance of the zero-shift cases, per degree relative to the weighted norm of the
/// expected result: a few roundings of the radius factor and of one product.
const ZERO_SHIFT_TOL: f64 = 1e-15;

/// Tolerance of the hand-worked cases, absolute per slot; all values are O(1).
const HAND_TOL: f64 = 1e-15;

/// A hand-worked case at p = 1: operator, input frame, output frame and the expected
/// output for the input of `degrees_zero_and_one_match_hand_computed_values`.
type HandCase = (Op, Frame<f64>, Frame<f64>, [f64; 4]);

fn factorial(k: usize) -> f64 {
    (1..=k).map(|j| j as f64).product()
}

/// xᵏ by repeated multiplication, as the operators form their radius factors.
fn power(x: f64, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, _| acc * x)
}

#[test]
fn zero_shift_rescales_by_the_radius_factor() {
    // With c′ = c the shift harmonics are R₀⁰ = 1 and 0 otherwise, so (CONVENTIONS
    // §3.11) M2M gives M̃′ⱼ = ρʲ M̃ⱼ with ρ = r/r′, and L2L gives L̃′ⱼ = σʲ⁺¹ L̃ⱼ
    // with σ = r′/r.
    //
    // Error measure: coefficients per degree in the orthonormal weighting (Nₘ for M2M,
    // Nₘ/Sₘ for L2L), relative to the weighted norm of the expected result.
    let mut rng = SplitMix64::new(0x7508);
    let mut ws = Workspace::new(P_MAX);
    let worst = Worst::new("zero shift vs radius rescaling, per degree");
    let centre = [0.3, -1.1, 0.7];
    for (r, r_out) in [
        (0.37, 0.74),
        (0.37, 0.37),
        (1.0, 0.3),
        (2f64.powi(-16), 3.0),
    ] {
        let (from, to) = (Frame::new(centre, r), Frame::new(centre, r_out));
        for p in 0..=P_MAX {
            let layout = Layout::new(p);
            for op in [Op::M2m, Op::L2l] {
                let input = random_coefficients(op.input(), p, &mut rng);
                let got = apply(op, p, &from, &to, &mut ws, &input);
                let want: Vec<f64> = (0..len(p))
                    .map(|i| {
                        let n = layout.nm(i).0;
                        let factor = match op {
                            Op::M2m => power(r / r_out, n),
                            _ => power(r_out / r, n + 1),
                        };
                        factor * input[i]
                    })
                    .collect();
                let scale = degree_norms(op.output(), p, &want);
                let e = worst.update(degree_error(op.output(), p, &got, &want, &scale));
                assert!(
                    e <= ZERO_SHIFT_TOL,
                    "{op:?}, r = {r}, r′ = {r_out}, p = {p}: {e:.3e}"
                );
            }
        }
    }
}

/// The coaxial forms of CONVENTIONS §3.11 for c′ − c = d e_z, from the input frame
/// radius `r` to the output radius `r_out`. Each keeps the order fixed and multiplies
/// the slots +i and −i by the same real factors.
fn coaxial(op: Op, p: usize, input: &[f64], d: f64, r: f64, r_out: f64) -> Vec<f64> {
    let layout = Layout::new(p);
    let mut out = vec![0.0; len(p)];
    for j in 0..=p {
        for i in -(j as isize)..=j as isize {
            let a = i.unsigned_abs();
            let value: f64 = match op {
                // Σₖ₌|ᵢ|ʲ ρᵏ M̃ₖⁱ (−d/r′)ʲ⁻ᵏ / (j − k)!
                Op::M2m => (a..=j)
                    .map(|k| {
                        power(r / r_out, k) * input[layout.idx(k, i)] * power(-d / r_out, j - k)
                            / factorial(j - k)
                    })
                    .sum(),
                // σʲ⁺¹ Σₙ₌ⱼᵖ L̃ₙⁱ (d/r)ⁿ⁻ʲ / (n − j)!
                Op::L2l => {
                    power(r_out / r, j + 1)
                        * (j..=p)
                            .map(|n| {
                                input[layout.idx(n, i)] * power(d / r, n - j) / factorial(n - j)
                            })
                            .sum::<f64>()
                }
                // (−1)ʲ⁺ⁱ σʲ⁺¹ Σₙ₌|ᵢ|ᵖ M̃ₙⁱ (n + j)! (sgn d)ⁿ⁺ʲ (r/|d|)ⁿ⁺ʲ⁺¹
                Op::M2l => {
                    let sign = if (j + a) % 2 == 0 { 1.0 } else { -1.0 };
                    sign * power(r_out / r, j + 1)
                        * (a..=p)
                            .map(|n| {
                                input[layout.idx(n, i)]
                                    * factorial(n + j)
                                    * power(d.signum(), n + j)
                                    * power(r / d.abs(), n + j + 1)
                            })
                            .sum::<f64>()
                }
            };
            out[layout.idx(j, i)] = value;
        }
    }
    out
}

#[test]
fn shifts_along_z_match_the_coaxial_forms() {
    // Error measure: coefficients per degree in the orthonormal weighting (Nₘ for
    // multipole output, Nₘ/Sₘ for local output), relative to the term magnitudes of
    // the translation (`common`). Both signs of d, and radius factors ≠ 1.
    let mut rng = SplitMix64::new(0x7509);
    let mut ws = Workspace::new(P_MAX);
    let worst = Worst::new("z shifts vs coaxial forms, per degree");
    let centre = [0.3, -1.1, 0.7];
    let r = 0.37;
    // (operator, output radius / r, d / r).
    let cases = [
        (Op::M2m, 2.0, 1.0),
        (Op::M2m, 1.3, -0.4),
        (Op::L2l, 0.5, -0.5),
        (Op::L2l, 0.6, 0.3),
        (Op::M2l, 1.0, 4.0),
        (Op::M2l, 1.0, -6.0),
        (Op::M2l, 0.6, 3.5),
        (Op::M2l, 1.5, -5.0),
    ];
    for (op, ratio, shift) in cases {
        let from = Frame::new(centre, r);
        let to = Frame::new(place(&from, [0.0, 0.0, shift]), ratio * r);
        let d = to.centre[2] - from.centre[2];
        for p in 0..=op.p_max() {
            let input = random_coefficients(op.input(), p, &mut rng);
            let got = apply(op, p, &from, &to, &mut ws, &input);
            let want = coaxial(op, p, &input, d, r, to.radius);
            let scale = op.scale(p, &from, &to, &input);
            let e = worst.update(degree_error(op.output(), p, &got, &want, &scale));
            assert!(e <= EXACT_TOL, "{op:?}, d = {d}, p = {p}: {e:.3e}");
        }
    }
}

#[test]
fn shifts_along_x_and_z_are_exact() {
    // On the z axis only order 0 of the shift harmonics is nonzero, on the x axis
    // every imaginary part is 0. Both are checked with the oracles of the general
    // tests, for shifts along ±x and ±z:
    // - M2M after P2M against P2M at the output frame (error measure of
    //   `exactness::m2m_after_p2m_equals_p2m_at_parent`);
    // - L2P after L2L against L2P (error measure of
    //   `exactness::l2l_is_exact_on_local_polynomials`);
    // - P2M, M2L, L2P against the direct sum within the truncation bound (`m2l`);
    // - on the x axis, a multipole or local with real coefficients (imaginary slots 0)
    //   stays real under each translation, exactly.
    let mut rng = SplitMix64::new(0x750a);
    let mut ws = Workspace::new(P_MAX);
    let exact = Worst::new("axis shifts, M2M and L2L exactness");
    let bound = Worst::new("axis shifts, M2L error / (bound + floor)");
    let frame = Frame::new([0.3, -1.1, 0.7], 0.37);
    let axes = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for axis in axes {
        // M2M: a child at 0.8 r′ from the parent centre along the axis, r′ = 2r.
        let parent = Frame::new(frame.centre, 2.0 * frame.radius);
        let child = Frame::new(place(&parent, axis.map(|c| 0.8 * c)), frame.radius);
        let sources: Vec<[f64; 3]> = (0..8).map(|_| place(&child, rng.in_cube())).collect();
        let charges: Vec<f64> = (0..8).map(|_| rng.charge()).collect();
        for p in [0, 1, 6, P_MAX] {
            let (got, want) = m2m_after_p2m(p, &child, &parent, &sources, &charges, &mut ws);
            let scale = m2m_p2m_scale(p, &child, &parent, &sources, &charges);
            let e = exact.update(degree_error(Kind::Multipole, p, &got, &want, &scale));
            assert!(e <= EXACT_TOL, "M2M along {axis:?}, p = {p}: {e:.3e}");
        }

        // L2L: parent to the same child.
        let targets: Vec<[f64; 3]> = (0..8).map(|_| place(&child, rng.in_cube())).collect();
        for p in [0, 1, 6, P_MAX] {
            let local = random_coefficients(Kind::Local, p, &mut rng);
            let magnitudes = degree_norms(Kind::Local, p, &local);
            let (got, want) = l2p_after_l2l(p, &parent, &child, &local, &targets, &mut ws);
            for ((g, w), &x) in got.iter().zip(&want).zip(&targets) {
                let scale = l2l_l2p_scale(&parent, &child, &magnitudes, x);
                let e = exact.update((g - w).abs() / scale);
                assert!(e <= EXACT_TOL, "L2L along {axis:?}, p = {p}: {e:.3e}");
            }
        }

        // M2L: same-level neighbours at offsets 2 and 3 along the axis.
        for distance in [4.0, 6.0] {
            let target = Frame::new(place(&frame, axis.map(|c| distance * c)), frame.radius);
            let sources: Vec<[f64; 3]> = (0..8).map(|_| place(&frame, rng.in_cube())).collect();
            let targets: Vec<[f64; 3]> = (0..8).map(|_| place(&target, rng.in_cube())).collect();
            for p in [0, 1, 6, P_MAX_M2L] {
                let check = check_against_direct_sum(
                    p, &frame, &target, &sources, &charges, &targets, &mut ws,
                );
                bound.update(check.attained);
            }
        }

        // Real coefficients stay real on the x axis.
        if axis[0] != 0.0 {
            let p = 9;
            let layout = Layout::new(p);
            let pairs = [
                (Op::M2m, child, parent),
                (Op::L2l, parent, child),
                (
                    Op::M2l,
                    frame,
                    Frame::new(place(&frame, axis.map(|c| 4.0 * c)), frame.radius),
                ),
            ];
            for (op, from, to) in pairs {
                let mut input = random_coefficients(op.input(), p, &mut rng);
                for (i, c) in input.iter_mut().enumerate() {
                    if layout.nm(i).1 < 0 {
                        *c = 0.0;
                    }
                }
                let out = apply(op, p, &from, &to, &mut ws, &input);
                for (i, c) in out.iter().enumerate() {
                    if layout.nm(i).1 < 0 {
                        assert_eq!(*c, 0.0, "{op:?} along {axis:?}: slot {i}");
                    }
                }
            }
        }
    }
}

#[test]
fn degrees_zero_and_one_match_hand_computed_values() {
    // Error measure: absolute per slot, 1e-15; all values are O(1). Storage order of
    // degree ≤ 1 is [X₀⁰, Im X₁¹, X₁⁰, Re X₁¹] (CONVENTIONS §3.6). Every input is
    // X = [2, 1/2, −1, 1/4], so X₀⁰ = 2, X₁⁰ = −1 and X₁¹ = 1/4 + i/2.
    //
    // M2M from (0, 1) to ((1/2, −1, 1), 2): b = (c − c′)/r′ = (−1/4, 1/2, −1/2),
    // ρ = 1/2. With conj R₀⁰ = 1, conj R₁⁰(b) = b_z, conj R₁¹(b) = (b_x − i b_y)/2:
    //   M̃′₀⁰ = M̃₀⁰ = 2,
    //   M̃′₁⁰ = ρ M̃₁⁰ + M̃₀⁰ b_z = −1/2 − 1 = −3/2,
    //   M̃′₁¹ = ρ M̃₁¹ + M̃₀⁰ (b_x − i b_y)/2 = (1/8 − 1/4) + i (1/4 − 1/2),
    // so M̃′ = [2, −1/4, −3/2, −1/8]; at p = 0, [2].
    //
    // L2L from (0, 2) to ((1, −1, 1), 1): t = (c′ − c)/r = (1/2, −1/2, 1/2), σ = 1/2.
    // With R₁⁰(t) = t_z and R₁¹(t) = (t_x + i t_y)/2, the doubling rule of §3.6 gives
    //   L̃′₀⁰ = σ (L̃₀⁰ + L̃₁⁰ t_z + Re L̃₁¹ t_x − Im L̃₁¹ t_y)
    //        = (2 − 1/2 + 1/8 + 1/4)/2 = 15/16,
    //   L̃′₁ⁱ = σ² L̃₁ⁱ,
    // so L̃′ = [15/16, 1/8, −1/4, 1/16]; at p = 0, [σ L̃₀⁰] = [1].
    //
    // M2L from (0, 1) to ((3, 0, 4), r′): b = (3, 0, 4), |b| = 5, σ = r′. From §3.3
    // and §3.5 (all real, since b_y = 0): I₀⁰ = 1/5, I₁⁰ = 4/125, I₁¹ = 3/125,
    // I₂⁰ = (3z² − r²)/r⁵ = 23/3125, I₂¹ = 3zρ/r⁵ = 36/3125, I₂² = 3ρ²/r⁵ = 27/3125,
    // and I₁⁻¹ = −I₁¹, I₂⁻¹ = −I₂¹, I₂⁻² = I₂² by the (−1)ᵐ rule. Then, from
    // L̃′ⱼⁱ = (−1)ʲ⁺ⁱ σʲ⁺¹ Σₙ Σₘ M̃ₙᵐ Iₙ₊ⱼᵐ⁻ⁱ(b) with M̃₁⁻¹ = −conj M̃₁¹:
    //   L̃′₀⁰ = σ (M̃₀⁰/5 + 4 M̃₁⁰/125 + 2 · 3 Re M̃₁¹/125) = σ · 19/50,
    //   L̃′₁⁰ = −σ² (4 M̃₀⁰/125 + 23 M̃₁⁰/3125 + 2 · 36 Re M̃₁¹/3125) = −σ² · 39/625,
    //   L̃′₁¹ = σ² (−3 M̃₀⁰/125 − 36 M̃₁⁰/3125 + 23 M̃₁¹/3125 − 27 conj M̃₁¹/3125)
    //        = σ² (−23/625 + i/125),
    // so for σ = 1, L̃′ = [19/50, 1/125, −39/625, −23/625]; at p = 0,
    // [M̃₀⁰/5] = [2/5]. For σ = 2 degree 0 doubles and degree 1 quadruples.
    //
    // M2M from (0, 1) to ((0, 2, 0), 1): b = (0, −2, 0), ρ = 1, so M̃′₁⁰ = M̃₁⁰ = −1
    // and M̃′₁¹ = M̃₁¹ + M̃₀⁰ (0 + 2i)/2 = 1/4 + 5i/2: M̃′ = [2, 5/2, −1, 1/4].
    let mut ws = Workspace::new(1);
    let input = [2.0, 0.5, -1.0, 0.25];
    let unit = Frame::new([0.0; 3], 1.0);
    let cases: [HandCase; 5] = [
        (
            Op::M2m,
            unit,
            Frame::new([0.5, -1.0, 1.0], 2.0),
            [2.0, -0.25, -1.5, -0.125],
        ),
        (
            Op::L2l,
            Frame::new([0.0; 3], 2.0),
            Frame::new([1.0, -1.0, 1.0], 1.0),
            [15.0 / 16.0, 0.125, -0.25, 0.0625],
        ),
        (
            Op::M2l,
            unit,
            Frame::new([3.0, 0.0, 4.0], 1.0),
            [19.0 / 50.0, 1.0 / 125.0, -39.0 / 625.0, -23.0 / 625.0],
        ),
        (
            Op::M2l,
            unit,
            Frame::new([3.0, 0.0, 4.0], 2.0),
            [
                2.0 * 19.0 / 50.0,
                4.0 / 125.0,
                -4.0 * 39.0 / 625.0,
                -4.0 * 23.0 / 625.0,
            ],
        ),
        // M2M from (0, 1) to ((0, 2, 0), 1): b = (0, −2, 0), ρ = 1.
        (
            Op::M2m,
            unit,
            Frame::new([0.0, 2.0, 0.0], 1.0),
            [2.0, 2.5, -1.0, 0.25],
        ),
    ];
    for (op, from, to, want) in cases {
        let got = apply(op, 1, &from, &to, &mut ws, &input);
        for (slot, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= HAND_TOL,
                "{op:?} p = 1, slot {slot}: {g} vs {w}"
            );
        }
        // p = 0: M̃′₀⁰ = M̃₀⁰ (M2M), L̃′₀⁰ = σ L̃₀⁰ (L2L), L̃′₀⁰ = σ M̃₀⁰/|b| (M2L).
        let got = apply(op, 0, &from, &to, &mut ws, &input[..1]);
        let want0 = match op {
            Op::M2m => input[0],
            Op::L2l => (to.radius / from.radius) * input[0],
            Op::M2l => (to.radius / from.radius) * input[0] / 5.0,
        };
        assert!(
            (got[0] - want0).abs() <= HAND_TOL,
            "{op:?} p = 0: {} vs {want0}",
            got[0]
        );
    }

    // Accumulation at p = 0 onto a nonzero value, by hand: M2L adds σ M̃₀⁰/|b|.
    let mut out = [1.0];
    direct::m2l(
        0,
        &unit,
        &Frame::new([3.0, 0.0, 4.0], 1.0),
        &mut ws,
        &[2.0],
        &mut out,
    );
    assert!((out[0] - 1.4).abs() <= HAND_TOL);
}

//! Random properties: random frames, coefficients, points and degrees, 64 cases each.
//! Each property reuses the error measure and tolerance of the deterministic test it
//! generalises, and prints its worst error over the cases with `--nocapture`.

use nd_fmm_ref::{Frame, Workspace, direct, leaf};
use proptest::prelude::*;

use crate::common::{
    EXACT_TOL, Kind, Op, P_MAX, P_MAX_M2L, SQRT3, SplitMix64, Worst, apply, degree_error,
    degree_norms, l2l_shift, len, m2l_scale, norm, place, random_coefficients,
};
use crate::exactness::{
    l2l_chain, l2l_chain_scale, l2l_l2p_scale, l2p_after_l2l, m2m_after_p2m, m2m_chain,
    m2m_chain_scale, m2m_p2m_scale,
};
use crate::linearity::LINEARITY_TOL;
use crate::m2l::check_against_direct_sum;

/// Frames with centres in [−10, 10]³ and radii either powers of two from 2⁻²⁰ to 2³
/// or uniform in [10⁻³, 10].
fn frame() -> impl Strategy<Value = Frame<f64>> {
    let radius = prop_oneof![(-20i32..=3).prop_map(|k| 2f64.powi(k)), 1e-3..10.0f64];
    ([-10.0..10.0f64, -10.0..10.0f64, -10.0..10.0f64], radius)
        .prop_map(|(centre, radius)| Frame::new(centre, radius))
}

/// Directions, not too short to normalise.
fn direction() -> impl Strategy<Value = [f64; 3]> {
    [-1.0..1.0f64, -1.0..1.0f64, -1.0..1.0f64]
        .prop_filter("direction must not be tiny", |v| norm(*v) > 0.1)
        .prop_map(|v| v.map(|c| c / norm(v)))
}

/// A frame nested in an outer one, as (radius ratio in [0.1, 0.9], direction,
/// fraction in [0, 1] of the largest offset √3 (1 − ratio) that keeps the inner
/// sphere inside the outer one).
fn nesting() -> impl Strategy<Value = (f64, [f64; 3], f64)> {
    (0.1..=0.9f64, direction(), 0.0..=1.0f64)
}

fn nest(outer: &Frame<f64>, (ratio, direction, fraction): (f64, [f64; 3], f64)) -> Frame<f64> {
    let offset = direction.map(|c| c * fraction * SQRT3 * (1.0 - ratio));
    Frame::new(place(outer, offset), ratio * outer.radius)
}

/// Scaled positions in the box [−1, 1]³.
fn in_box() -> impl Strategy<Value = [f64; 3]> {
    [-1.0..=1.0f64, -1.0..=1.0f64, -1.0..=1.0f64]
}

fn charge() -> impl Strategy<Value = f64> {
    prop_oneof![-1.0..-0.01f64, 0.01..1.0f64]
}

/// 64 cases per property, as for the other test targets.
fn config() -> ProptestConfig {
    ProptestConfig::with_cases(64)
}

#[test]
fn m2m_after_p2m_equals_p2m_at_output() {
    let worst = Worst::new("property: M2M after P2M vs P2M at output, per degree");
    proptest!(config(), |(
        parent in frame(),
        nesting in nesting(),
        p in 0..=P_MAX,
        sources in prop::collection::vec((in_box(), charge()), 1..=6),
    )| {
        // Error measure: as `exactness::m2m_after_p2m_equals_p2m_at_parent`.
        let mut ws = Workspace::new(p);
        let child = nest(&parent, nesting);
        let (y, q): (Vec<[f64; 3]>, Vec<f64>) =
            sources.iter().map(|&(u, q)| (place(&child, u), q)).unzip();
        let (got, want) = m2m_after_p2m(p, &child, &parent, &y, &q, &mut ws);
        let scale = m2m_p2m_scale(p, &child, &parent, &y, &q);
        let e = worst.update(degree_error(Kind::Multipole, p, &got, &want, &scale));
        prop_assert!(e <= EXACT_TOL, "{e:.3e}");
    });
}

#[test]
fn l2l_is_exact_on_local_polynomials() {
    let worst = Worst::new("property: L2P after L2L vs L2P, relative to term magnitudes");
    proptest!(config(), |(
        parent in frame(),
        nesting in nesting(),
        p in 0..=P_MAX,
        seed in any::<u64>(),
        targets in prop::collection::vec(in_box(), 1..=6),
    )| {
        // Error measure: as `exactness::l2l_is_exact_on_local_polynomials`.
        let mut ws = Workspace::new(p);
        let child = nest(&parent, nesting);
        let local = random_coefficients(Kind::Local, p, &mut SplitMix64::new(seed));
        let magnitudes = degree_norms(Kind::Local, p, &local);
        let x: Vec<[f64; 3]> = targets.iter().map(|&v| place(&child, v)).collect();
        let (got, want) = l2p_after_l2l(p, &parent, &child, &local, &x, &mut ws);
        for ((g, w), &x) in got.iter().zip(&want).zip(&x) {
            let e = worst.update((g - w).abs() / l2l_l2p_scale(&parent, &child, &magnitudes, x));
            prop_assert!(e <= EXACT_TOL, "{e:.3e}");
        }
    });
}

#[test]
fn m2m_and_l2l_compose() {
    let worst_m2m = Worst::new("property: M2M composition, per degree");
    let worst_l2l = Worst::new("property: L2L composition, per degree");
    proptest!(config(), |(
        outer in frame(),
        first in nesting(),
        second in nesting(),
        p in 0..=P_MAX,
        seed in any::<u64>(),
    )| {
        // Error measure: as `exactness::m2m_composes` and `exactness::l2l_composes`.
        let mut ws = Workspace::new(p);
        let mut rng = SplitMix64::new(seed);
        let middle = nest(&outer, first);
        let inner = nest(&middle, second);

        let frames = [&inner, &middle, &outer];
        let input = random_coefficients(Kind::Multipole, p, &mut rng);
        let (twice, once) = m2m_chain(p, frames, &input, &mut ws);
        let scale = m2m_chain_scale(p, frames, &input);
        let e = worst_m2m.update(degree_error(Kind::Multipole, p, &twice, &once, &scale));
        prop_assert!(e <= EXACT_TOL, "M2M: {e:.3e}");

        let frames = [&outer, &middle, &inner];
        let input = random_coefficients(Kind::Local, p, &mut rng);
        let (twice, once) = l2l_chain(p, frames, &input, &mut ws);
        let scale = l2l_chain_scale(p, frames, &input);
        let e = worst_l2l.update(degree_error(Kind::Local, p, &twice, &once, &scale));
        prop_assert!(e <= EXACT_TOL, "L2L: {e:.3e}");
    });
}

#[test]
fn m2l_within_bound_of_direct_sum() {
    let worst = Worst::new("property: M2L vs direct sum, error / (bound + floor)");
    proptest!(config(), |(
        source in frame(),
        sigma in 0.5..=2.0f64,
        direction in direction(),
        margin in 1.1..=3.0f64,
        p in 0..=P_MAX_M2L,
        sources in prop::collection::vec((in_box(), charge()), 1..=4),
        targets in prop::collection::vec(in_box(), 1..=4),
    )| {
        // Error measure: as `m2l::m2l_matches_direct_sum_on_the_v_list`. The shift is
        // |b| = margin · √3 (1 + σ), beyond the convergence limit, so the bound's
        // condition |u| + σ|v′| < |b| holds for all points in the boxes.
        let mut ws = Workspace::new(p);
        let b = direction.map(|c| c * margin * SQRT3 * (1.0 + sigma));
        let target = Frame::new(place(&source, b), sigma * source.radius);
        let (y, q): (Vec<[f64; 3]>, Vec<f64>) =
            sources.iter().map(|&(u, q)| (place(&source, u), q)).unzip();
        let x: Vec<[f64; 3]> = targets.iter().map(|&v| place(&target, v)).collect();
        // Panics if an error exceeds its bound plus floor.
        let check = check_against_direct_sum(p, &source, &target, &y, &q, &x, &mut ws);
        worst.update(check.attained);
    });
}

#[test]
fn m2l_of_a_unit_monopole_equals_p2l() {
    let worst = Worst::new("property: M2L of a unit monopole vs P2L, per degree");
    proptest!(config(), |(
        source in frame(),
        sigma in 0.5..=2.0f64,
        direction in direction(),
        margin in 1.1..=3.0f64,
        p in 0..=P_MAX_M2L,
    )| {
        // Error measure: as `m2l::m2l_of_a_unit_monopole_equals_p2l_of_a_unit_charge`.
        let mut ws = Workspace::new(p);
        let b = direction.map(|c| c * margin * SQRT3 * (1.0 + sigma));
        let target = Frame::new(place(&source, b), sigma * source.radius);
        let mut monopole = vec![0.0; len(p)];
        monopole[0] = 1.0;
        let mut got = vec![0.0; len(p)];
        direct::m2l(p, &source, &target, &mut ws, &monopole, &mut got);
        let mut want = vec![0.0; len(p)];
        leaf::p2l(p, &target, &[source.centre], &[1.0], &mut ws, &mut want);
        let (b, sigma) = l2l_shift(&source, &target);
        let scale = m2l_scale(&degree_norms(Kind::Multipole, p, &monopole), sigma, norm(b));
        let e = worst.update(degree_error(Kind::Local, p, &got, &want, &scale));
        prop_assert!(e <= EXACT_TOL, "{e:.3e}");
    });
}

#[test]
fn translations_are_linear_and_accumulate() {
    let worst = Worst::new("property: linearity with accumulation, per degree");
    proptest!(config(), |(
        op in prop_oneof![Just(Op::M2m), Just(Op::L2l), Just(Op::M2l)],
        outer in frame(),
        nesting in nesting(),
        sigma in 0.5..=2.0f64,
        direction in direction(),
        p in 0..=P_MAX_M2L,
        seed in any::<u64>(),
        alpha in -2.0..2.0f64,
    )| {
        // Error measure: as `linearity::translations_are_linear`, with its tolerance
        // 1e-14. T(x + αy) in one call is compared with T(αy) accumulated onto T(x),
        // which checks linearity and accumulation together. Frames: nested for M2M
        // and L2L, and |b| = 1.5 √3 (1 + σ) for M2L.
        let mut ws = Workspace::new(p);
        let (from, to) = match op {
            Op::M2m => (nest(&outer, nesting), outer),
            Op::L2l => (outer, nest(&outer, nesting)),
            Op::M2l => {
                let b = direction.map(|c| c * 1.5 * SQRT3 * (1.0 + sigma));
                (outer, Frame::new(place(&outer, b), sigma * outer.radius))
            }
        };
        let mut rng = SplitMix64::new(seed);
        let x = random_coefficients(op.input(), p, &mut rng);
        let y = random_coefficients(op.input(), p, &mut rng);
        let combined: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a + alpha * b).collect();
        let want = apply(op, p, &from, &to, &mut ws, &combined);
        let scaled: Vec<f64> = y.iter().map(|b| alpha * b).collect();
        let mut got = apply(op, p, &from, &to, &mut ws, &x);
        op.function()(p, &from, &to, &mut ws, &scaled, &mut got);
        let scale: Vec<f64> = op
            .scale(p, &from, &to, &x)
            .iter()
            .zip(op.scale(p, &from, &to, &y))
            .map(|(sx, sy)| sx + alpha.abs() * sy)
            .collect();
        let e = worst.update(degree_error(op.output(), p, &got, &want, &scale));
        prop_assert!(e <= LINEARITY_TOL, "{op:?}: {e:.3e}");
    });
}

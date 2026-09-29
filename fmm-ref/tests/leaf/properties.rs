//! Random properties: random frames, points, charges and degrees, 64 cases each to
//! keep CI time low. Each property reuses the error measure and tolerance of the
//! deterministic test it generalises.

use nd_fmm_ref::{Frame, Workspace, leaf};
use proptest::prelude::*;

use crate::common::{Kind, P_MAX, SQRT3, Target, degree_error, legendre_series, len, norm, place};
use crate::series::{GRADIENT_TOL, VALUE_TOL, p2l_l2p, p2m_m2p};
use crate::truncation::Case;

/// Frames with centres in [−10, 10]³ and radii either powers of two from 2⁻²⁰ to 2³
/// or uniform in [10⁻³, 10].
fn frame() -> impl Strategy<Value = Frame<f64>> {
    let radius = prop_oneof![(-20i32..=3).prop_map(|k| 2f64.powi(k)), 1e-3..10.0f64];
    ([-10.0..10.0f64, -10.0..10.0f64, -10.0..10.0f64], radius)
        .prop_map(|(centre, radius)| Frame::new(centre, radius))
}

/// Scaled positions with length uniform in [rmin, rmax].
fn shell(rmin: f64, rmax: f64) -> impl Strategy<Value = [f64; 3]> {
    let direction = [-1.0..1.0f64, -1.0..1.0f64, -1.0..1.0f64]
        .prop_filter("direction must not be tiny", |v| norm(*v) > 0.1);
    (direction, rmin..=rmax).prop_map(|(v, r)| {
        let n = norm(v);
        v.map(|c| r * c / n)
    })
}

/// Scaled positions in the box [−1, 1]³.
fn in_box() -> impl Strategy<Value = [f64; 3]> {
    [-1.0..=1.0f64, -1.0..=1.0f64, -1.0..=1.0f64]
}

fn charge() -> impl Strategy<Value = f64> {
    prop_oneof![-1.0..-0.01f64, 0.01..1.0f64]
}

/// Up to four points from `points`, each with a charge.
fn charged(points: impl Strategy<Value = [f64; 3]>) -> impl Strategy<Value = Vec<([f64; 3], f64)>> {
    prop::collection::vec((points, charge()), 1..=4)
}

fn place_all(frame: &Frame<f64>, points: &[([f64; 3], f64)]) -> (Vec<[f64; 3]>, Vec<f64>) {
    points.iter().map(|&(u, q)| (place(frame, u), q)).unzip()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn p2m_m2p_matches_legendre_series(
        frame in frame(),
        p in 0..=P_MAX,
        sources in charged(shell(0.01, SQRT3)),
        target in shell(2.0, 8.0),
    ) {
        // Error measure: as `series::p2m_then_m2p_matches_legendre_series`.
        let mut ws = Workspace::new(p);
        let (y, q) = place_all(&frame, &sources);
        let x = place(&frame, target);
        let reference = legendre_series(p, frame.centre, &y, &q, x, Target::Far);
        let (value, gradient) = p2m_m2p(p, &frame, &y, &q, x, &mut ws);
        prop_assert!(reference.value_error(value) <= VALUE_TOL);
        prop_assert!(reference.gradient_error(gradient) <= GRADIENT_TOL);
    }

    #[test]
    fn p2l_l2p_matches_legendre_series(
        frame in frame(),
        p in 0..=P_MAX,
        sources in charged(shell(2.0, 8.0)),
        target in shell(0.01, SQRT3),
    ) {
        // Error measure: as `series::p2l_then_l2p_matches_legendre_series`.
        let mut ws = Workspace::new(p);
        let (y, q) = place_all(&frame, &sources);
        let x = place(&frame, target);
        let reference = legendre_series(p, frame.centre, &y, &q, x, Target::Near);
        let (value, gradient) = p2l_l2p(p, &frame, &y, &q, x, &mut ws);
        prop_assert!(reference.value_error(value) <= VALUE_TOL);
        prop_assert!(reference.gradient_error(gradient) <= GRADIENT_TOL);
    }

    #[test]
    fn p2m_m2p_within_truncation_bound(
        frame in frame(),
        p in 0..=P_MAX,
        sources in charged(in_box()),
        target in shell(4.0 - SQRT3, 8.0),
    ) {
        // Error measure: as `truncation::p2m_m2p_within_truncation_bound`.
        let mut ws = Workspace::new(p);
        let (y, q) = place_all(&frame, &sources);
        let x = place(&frame, target);
        let got = p2m_m2p(p, &frame, &y, &q, x, &mut ws);
        let case = Case { p, centre: frame.centre, sources: &y, charges: &q, x, which: Target::Far };
        let excess = case.excess(got);
        prop_assert!(excess.passes(), "{excess:?}");
    }

    #[test]
    fn p2l_l2p_within_truncation_bound(
        frame in frame(),
        p in 0..=P_MAX,
        sources in charged(shell(4.0 - SQRT3, 8.0)),
        target in in_box(),
    ) {
        // Error measure: as `truncation::p2l_l2p_within_truncation_bound`.
        let mut ws = Workspace::new(p);
        let (y, q) = place_all(&frame, &sources);
        let x = place(&frame, target);
        let got = p2l_l2p(p, &frame, &y, &q, x, &mut ws);
        let case = Case { p, centre: frame.centre, sources: &y, charges: &q, x, which: Target::Near };
        let excess = case.excess(got);
        prop_assert!(excess.passes(), "{excess:?}");
    }

    #[test]
    fn p2m_and_p2l_accumulate_additively(
        frame in frame(),
        p in 0..=P_MAX,
        near in charged(in_box()),
        far in charged(shell(2.0, 8.0)),
        split in 0usize..=4,
    ) {
        // Error measure: as `linearity::p2m_and_p2l_are_additive_over_sources`; the
        // second part is accumulated onto the first.
        let mut ws = Workspace::new(p);
        for (kind, points) in [(Kind::Multipole, &near), (Kind::Local, &far)] {
            let expand = match kind {
                Kind::Multipole => leaf::p2m::<f64>,
                Kind::Local => leaf::p2l::<f64>,
            };
            let (y, q) = place_all(&frame, points);
            let k = split.min(y.len());
            let mut whole = vec![0.0; len(p)];
            expand(p, &frame, &y, &q, &mut ws, &mut whole);
            let (mut first, mut second) = (vec![0.0; len(p)], vec![0.0; len(p)]);
            expand(p, &frame, &y[..k], &q[..k], &mut ws, &mut first);
            expand(p, &frame, &y[k..], &q[k..], &mut ws, &mut second);
            let mut accumulated = first.clone();
            expand(p, &frame, &y[k..], &q[k..], &mut ws, &mut accumulated);
            let e = degree_error(kind, p, &accumulated, &whole, &[&first, &second]);
            prop_assert!(e <= 1e-14, "{kind:?}: {e:.3e}");
        }
    }
}

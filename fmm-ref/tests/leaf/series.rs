//! Same-degree oracle: P2M then M2P, and P2L then L2P, reproduce the truncated
//! Legendre series Σₙ≤ₚ q aⁿ / dⁿ⁺¹ Pₙ(cos γ) and its gradient, for p ≤ 30.

use nd_fmm_ref::{Frame, Workspace, leaf};

use crate::common::{
    P_MAX, Reference, SQRT3, SplitMix64, Target, Worst, frames, legendre_series, len, place,
};

/// Potential tolerance, relative to the sum of term magnitudes of the series.
pub const VALUE_TOL: f64 = 1e-13;
/// Gradient tolerance: Euclidean norm of the error, relative to the sum of the
/// Euclidean norms of the gradient terms of the series.
pub const GRADIENT_TOL: f64 = 1e-12;

/// P2M at `frame` from the sources, then M2P at the target, with gradient.
pub fn p2m_m2p(
    p: usize,
    frame: &Frame<f64>,
    sources: &[[f64; 3]],
    charges: &[f64],
    target: [f64; 3],
    ws: &mut Workspace<f64>,
) -> (f64, [f64; 3]) {
    let mut multipole = vec![0.0; len(p)];
    leaf::p2m(p, frame, sources, charges, ws, &mut multipole);
    let (mut potential, mut gradient) = ([0.0], [[0.0; 3]]);
    let g = Some(&mut gradient[..]);
    leaf::m2p(p, frame, &multipole, &[target], ws, &mut potential, g);
    (potential[0], gradient[0])
}

/// P2L at `frame` from the sources, then L2P at the target, with gradient.
pub fn p2l_l2p(
    p: usize,
    frame: &Frame<f64>,
    sources: &[[f64; 3]],
    charges: &[f64],
    target: [f64; 3],
    ws: &mut Workspace<f64>,
) -> (f64, [f64; 3]) {
    let mut local = vec![0.0; len(p)];
    leaf::p2l(p, frame, sources, charges, ws, &mut local);
    let (mut potential, mut gradient) = ([0.0], [[0.0; 3]]);
    let g = Some(&mut gradient[..]);
    leaf::l2p(p, frame, &local, &[target], ws, &mut potential, g);
    (potential[0], gradient[0])
}

/// Checks one evaluation against the series and records the errors.
fn check(
    reference: &Reference,
    (value, gradient): (f64, [f64; 3]),
    worst: &mut [Worst; 2],
    what: &str,
) {
    let e = worst[0].update(reference.value_error(value));
    assert!(e <= VALUE_TOL, "{what}: potential error {e:.3e}");
    let e = worst[1].update(reference.gradient_error(gradient));
    assert!(e <= GRADIENT_TOL, "{what}: gradient error {e:.3e}");
}

/// Pairs of scaled positions (nearer, farther) that include the edges of the tested
/// ranges: the box corner at |u| = √3, |v| = 2, and collinear and antipodal points,
/// where cos γ = ±1 and Pₙ′ is largest.
fn edge_pairs() -> Vec<([f64; 3], [f64; 3])> {
    let corner = [1.0, 1.0, 1.0];
    let s = 2.0 / SQRT3;
    vec![
        (corner, corner.map(|c| c * s)),
        (corner, corner.map(|c| -c * s)),
        ([0.0, 0.0, 1.7], [0.0, 0.0, 2.0]),
        ([0.0, 0.0, 1.7], [0.0, 0.0, -2.0]),
        ([1.0, -1.0, 1.0], [2.0, 0.0, 0.0]),
        ([0.2, 0.1, -0.3], [-5.0, 3.0, 1.0]),
    ]
}

#[test]
fn p2m_then_m2p_matches_legendre_series() {
    // Error measure: potential relative to the sum of term magnitudes of the series,
    // gradient (Euclidean norm) relative to the sum of term norms. Sources at
    // |u| ≤ √3 plus a margin, targets at |v| ≥ 2 (CONVENTIONS §3.9).
    let mut rng = SplitMix64::new(0x5e41);
    let mut ws = Workspace::new(P_MAX);
    let mut worst = [
        Worst::new("P2M-M2P potential vs series"),
        Worst::new("P2M-M2P gradient vs series"),
    ];
    for frame in frames() {
        let mut pairs = edge_pairs();
        for _ in 0..12 {
            pairs.push((rng.in_shell(0.05, 1.8), rng.in_shell(2.0, 8.0)));
        }
        for (u, v) in pairs {
            let (y, x, q) = (place(&frame, u), place(&frame, v), rng.charge());
            for p in 0..=P_MAX {
                let reference = legendre_series(p, frame.centre, &[y], &[q], x, Target::Far);
                let got = p2m_m2p(p, &frame, &[y], &[q], x, &mut ws);
                check(&reference, got, &mut worst, &format!("{frame:?} p = {p}"));
            }
        }
    }
}

#[test]
fn p2l_then_l2p_matches_legendre_series() {
    // Error measure: as for P2M-M2P. Sources at |u| ≥ 2, targets at |v| ≤ √3 plus a
    // margin (CONVENTIONS §3.9).
    let mut rng = SplitMix64::new(0x5e42);
    let mut ws = Workspace::new(P_MAX);
    let mut worst = [
        Worst::new("P2L-L2P potential vs series"),
        Worst::new("P2L-L2P gradient vs series"),
    ];
    for frame in frames() {
        let mut pairs = edge_pairs();
        for _ in 0..12 {
            pairs.push((rng.in_shell(0.05, 1.8), rng.in_shell(2.0, 8.0)));
        }
        for (v, u) in pairs {
            let (y, x, q) = (place(&frame, u), place(&frame, v), rng.charge());
            for p in 0..=P_MAX {
                let reference = legendre_series(p, frame.centre, &[y], &[q], x, Target::Near);
                let got = p2l_l2p(p, &frame, &[y], &[q], x, &mut ws);
                check(&reference, got, &mut worst, &format!("{frame:?} p = {p}"));
            }
        }
    }
}

#[test]
fn several_sources_match_legendre_series() {
    // Error measure: as above, with the term magnitudes summed over all sources.
    let mut rng = SplitMix64::new(0x5e43);
    let mut ws = Workspace::new(P_MAX);
    let mut worst = [
        Worst::new("several sources, potential vs series"),
        Worst::new("several sources, gradient vs series"),
    ];
    for frame in frames() {
        let near: Vec<[f64; 3]> = (0..10).map(|_| place(&frame, rng.in_cube())).collect();
        let far: Vec<[f64; 3]> = (0..10)
            .map(|_| place(&frame, rng.in_shell(2.0, 6.0)))
            .collect();
        let charges: Vec<f64> = (0..10).map(|_| rng.charge()).collect();
        for p in [0, 1, 2, 5, 10, 20, P_MAX] {
            for &x in &far {
                let reference = legendre_series(p, frame.centre, &near, &charges, x, Target::Far);
                let got = p2m_m2p(p, &frame, &near, &charges, x, &mut ws);
                check(
                    &reference,
                    got,
                    &mut worst,
                    &format!("M2P {frame:?} p = {p}"),
                );
            }
            for &x in &near {
                let reference = legendre_series(p, frame.centre, &far, &charges, x, Target::Near);
                let got = p2l_l2p(p, &frame, &far, &charges, x, &mut ws);
                check(
                    &reference,
                    got,
                    &mut worst,
                    &format!("L2P {frame:?} p = {p}"),
                );
            }
        }
    }
}

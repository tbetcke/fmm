//! The dipole operators against central differences of the charge operators in the
//! source position: a dipole of moment d at y is the limit d · ∇_y of a unit charge at y
//! (CONVENTIONS §3.7, "Dipole sources"), so
//!
//! d · ∇_y F(y) ≈ (8 (F(y + h d) − F(y − h d)) − (F(y + 2h d) − F(y − 2h d))) / (12 h),
//!
//! the fourth-order central difference, for F the charge P2M, P2L or P2P with q = 1. The
//! step is h |d| = 2 · 10⁻⁴ r |u| for the expansions (sources at 0.5 ≤ |u| ≤ √3 and
//! 2 ≤ |u| ≤ 8) and 10⁻³ of the distance for P2P: relative to the scale on which the
//! function varies, so that the truncation error, of order (p h |d| / (r |u|))⁴, and the
//! rounding error, of order ε r |u| / (h |d|), both stay near 10⁻¹¹. The frames have
//! centres within a few radii of the origin, so that y + t d rounds on the scale of the
//! frame. These checks are independent of the fixtures and of the gradient ladder, but
//! loose.
//!
//! Error measures: coefficients per degree, relative to the largest magnitude of the
//! difference in that degree ([`degree_error`]), f64, p ≤ 12; P2P potentials and
//! gradients relative to the term magnitude |d| / r² and 2 |d| / r³.

use nd_fmm_ref::{Frame, Workspace, leaf, p2p};

use crate::common::{SQRT3, SplitMix64, degree_error, len, norm, place};

/// Tolerance of the coefficients, per degree, relative.
const EXPANSION_TOL: f64 = 1e-10;
/// Tolerance of P2P, relative to the term magnitude.
const P2P_TOL: f64 = 1e-10;
const P: usize = 12;

/// Radii 1, 0.37 and 2⁻¹⁶, each centred within four radii of the origin.
fn frames() -> [Frame<f64>; 3] {
    let deep = 2f64.powi(-16);
    [
        Frame::new([0.3, -1.1, 0.7], 1.0),
        Frame::new([-0.5, 0.25, 0.75], 0.37),
        Frame::new([3.0 * deep, -deep, 2.0 * deep], deep),
    ]
}

/// The fourth-order central difference of `f` along `d` at y with step h.
fn difference<const N: usize>(
    y: [f64; 3],
    d: [f64; 3],
    h: f64,
    mut f: impl FnMut([f64; 3]) -> [Vec<f64>; N],
) -> [Vec<f64>; N] {
    let mut at = |t: f64| f(core::array::from_fn(|k| y[k] + t * d[k]));
    let (a, b, c, e) = (at(h), at(-h), at(2.0 * h), at(-2.0 * h));
    core::array::from_fn(|n| {
        (0..a[n].len())
            .map(|i| (8.0 * (a[n][i] - b[n][i]) - (c[n][i] - e[n][i])) / (12.0 * h))
            .collect()
    })
}

type Charge = fn(usize, &Frame<f64>, &[[f64; 3]], &[f64], &mut Workspace<f64>, &mut [f64]);
type Dipole = fn(usize, &Frame<f64>, &[[f64; 3]], &[[f64; 3]], &mut Workspace<f64>, &mut [f64]);

fn check_expansion(name: &str, charge: Charge, dipole: Dipole, radii: (f64, f64), seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let mut ws = Workspace::new(P);
    let mut worst: f64 = 0.0;
    for frame in frames() {
        for _ in 0..8 {
            let u = rng.in_shell(radii.0, radii.1);
            let y = place(&frame, u);
            let d = rng.moment().map(|c| c * frame.radius);
            let h = 2e-4 * frame.radius * norm(u) / norm(d);
            let [want] = difference(y, d, h, |y| {
                let mut out = vec![0.0; len(P)];
                charge(P, &frame, &[y], &[1.0], &mut ws, &mut out);
                [out]
            });
            let mut got = vec![0.0; len(P)];
            dipole(P, &frame, &[y], &[d], &mut ws, &mut got);
            let e = degree_error(P, &got, &want);
            assert!(e <= EXPANSION_TOL, "{name} at u = {y:?}: error {e:.3e}");
            worst = worst.max(e);
        }
    }
    println!("{name} against differences of the charge operator: worst {worst:.2e}");
}

#[test]
fn p2m_dipoles_is_the_derivative_of_p2m() {
    check_expansion("p2m_dipoles", leaf::p2m, leaf::p2m_dipoles, (0.5, SQRT3), 1);
}

#[test]
fn p2l_dipoles_is_the_derivative_of_p2l() {
    check_expansion("p2l_dipoles", leaf::p2l, leaf::p2l_dipoles, (2.0, 8.0), 2);
}

#[test]
fn p2p_dipoles_is_the_derivative_of_p2p() {
    let mut rng = SplitMix64::new(3);
    let (mut worst_phi, mut worst_grad): (f64, f64) = (0.0, 0.0);
    for _ in 0..64 {
        let x = rng.in_shell(0.0, 2.0);
        let y = rng.in_shell(0.0, 2.0);
        let d = rng.moment();
        let r = norm(core::array::from_fn(|k| x[k] - y[k]));
        let h = 1e-3 * r / norm(d);
        let [phi, gx, gy, gz] = difference(y, d, h, |y| {
            let (mut phi, mut g) = ([0.0], [[0.0; 3]]);
            p2p::p2p(&[y], &[1.0], &[x], &mut phi, Some(&mut g));
            [vec![phi[0]], vec![g[0][0]], vec![g[0][1]], vec![g[0][2]]]
        });
        let (mut got_phi, mut got_g) = ([0.0], [[0.0; 3]]);
        p2p::p2p_dipoles(&[y], &[d], &[x], &mut got_phi, Some(&mut got_g));
        let ep = (got_phi[0] - phi[0]).abs() / (norm(d) / (r * r));
        let eg = norm([
            got_g[0][0] - gx[0],
            got_g[0][1] - gy[0],
            got_g[0][2] - gz[0],
        ]) / (2.0 * norm(d) / (r * r * r));
        assert!(
            ep <= P2P_TOL && eg <= P2P_TOL,
            "x = {x:?}, y = {y:?}: {ep:.3e}, {eg:.3e}"
        );
        worst_phi = worst_phi.max(ep);
        worst_grad = worst_grad.max(eg);
    }
    println!(
        "p2p_dipoles against differences of p2p: worst {worst_phi:.2e} (φ), {worst_grad:.2e} (∇φ)"
    );
}

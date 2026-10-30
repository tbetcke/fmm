//! Shared helpers: evaluation of a family into complex vectors, and the argument grid of
//! the dense sweeps with the orders the operators need there (spike S3's
//! `src/ranges.rs`).

use nd_fmm_math::Complex;
use nd_fmm_math::radial::{bessel_j, hankel_h, modified_i, modified_k};
use num_traits::Zero;

/// The four families of `radial`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    J,
    H,
    I,
    K,
}

impl Family {
    pub fn from_name(name: &str) -> Family {
        match name {
            "j" => Family::J,
            "h" => Family::H,
            "i" => Family::I,
            "k" => Family::K,
            _ => panic!("unknown family {name}"),
        }
    }
}

/// Values and derivatives of `family` at orders 0..=top; for i and k, z is real (its
/// real part is taken).
pub fn evaluate(
    family: Family,
    z: Complex<f64>,
    s: f64,
    top: usize,
) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let mut v = vec![Complex::zero(); top + 1];
    let mut d = vec![Complex::zero(); top + 1];
    match family {
        Family::J => bessel_j(z, s, &mut v, Some(&mut d)),
        Family::H => hankel_h(z, s, &mut v, Some(&mut d)),
        Family::I | Family::K => {
            let mut vr = vec![0.0; top + 1];
            let mut dr = vec![0.0; top + 1];
            if family == Family::I {
                modified_i(z.re, s, &mut vr, Some(&mut dr));
            } else {
                modified_k(z.re, s, &mut vr, Some(&mut dr));
            }
            for n in 0..=top {
                v[n] = Complex::from(vr[n]);
                d[n] = Complex::from(dr[n]);
            }
        }
    }
    (v, d)
}

/// p of proposal §3.5 (Gumerov–Duraiswami 2009 eq. 18, δ = 2), as S3's `degree`.
fn degree(ka: f64, eps: f64) -> usize {
    let p_hi = ka + (1.5 * (1.0 / eps).ln()).powf(2.0 / 3.0) * ka.powf(1.0 / 3.0);
    let p_lo = 1.0 - (eps * 0.5f64.powf(1.5)).ln() / 2.0f64.ln();
    ((p_lo.powi(4) + p_hi.powi(4)).powf(0.25)).ceil() as usize - 1
}

/// The smallest tolerance and the largest k a of a level (level 2 at kD = 3000).
const EPS: f64 = 1e-10;
const KA_MAX: f64 = 650.0;

/// The highest order of j or i any operator needs at |z| = r: the M2M and L2L start
/// values to p_l + p_{l+1} at k a = 2r.
fn regular_top(r: f64) -> usize {
    let ka = (2.0 * r).min(KA_MAX);
    degree(ka, EPS) + degree(ka / 2.0, EPS)
}

/// The highest order of h or k any operator needs at |z| = r: the M2L start values to
/// 2p at t ≥ 4a/√3.
fn singular_top(r: f64) -> usize {
    2 * degree((r * 3.0f64.sqrt() / 4.0).min(KA_MAX), EPS)
}

/// The grid of the dense sweeps: (|z|, regular top, singular top) at 65 moduli,
/// log-uniform from 10⁻⁶ to 4550 (the projection's hₙ(k R), R up to 7a, at k a = 650).
pub fn sweep() -> Vec<(f64, usize, usize)> {
    (0..=64)
        .map(|i| {
            let r = 10f64.powf(-6.0 + i as f64 * (4550f64.log10() + 6.0) / 64.0);
            (r, regular_top(r), singular_top(r))
        })
        .collect()
}

/// The arguments of the sweeps: 0 to π/2, both ends and next to the imaginary axis.
pub const ANGLES: [f64; 7] = [
    0.0,
    0.05,
    std::f64::consts::FRAC_PI_8,
    std::f64::consts::FRAC_PI_4,
    3.0 * std::f64::consts::FRAC_PI_8,
    std::f64::consts::FRAC_PI_2 - 1e-3,
    std::f64::consts::FRAC_PI_2,
];

/// r e^{iθ}, exactly imaginary at θ = π/2.
pub fn polar(r: f64, theta: f64) -> Complex<f64> {
    if theta == std::f64::consts::FRAC_PI_2 {
        Complex::new(0.0, r)
    } else {
        Complex::new(r * theta.cos(), r * theta.sin())
    }
}

/// iᵏ, exactly.
pub fn i_pow(k: i64) -> Complex<f64> {
    match k.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

//! The coaxial coefficients by projection (spike S2, SPIKE_REPORT §2.2).
//!
//! With p̄ₙᵐ the orthonormal associated Legendre functions on [−1, 1] (no Condon–Shortley
//! phase), p̄ₙᵐ = √((2n + 1)/2) P̄ₙᵐ, a wave of order m restricted to the half-plane φ = 0
//! carries the same φ factor on both sides of the translation. At the point
//! r = ρ (sin α, 0, cos α) of a sphere about the new centre, with R and Θ the polar
//! coordinates of r + t about the old centre:
//!
//! ```text
//! fₙ(k R) p̄ₙᵐ(cos Θ) = Σ_{n′} Eᵐ_{n′n} g_{n′}(k ρ) p̄_{n′}ᵐ(cos α)
//! ```
//!
//! Gauss–Legendre quadrature in x = cos α gives A = Σᵢ wᵢ p̄_{n′}ᵐ(xᵢ) fₙ(k Rᵢ) p̄ₙᵐ(uᵢ) =
//! E g_{n′}(kρ), and the same with ∂/∂ρ (divided by k) gives B = E g′_{n′}(kρ). Where
//! g = j, which has zeros, each target degree is solved by least squares,
//! E = (A conj ĝ + B conj ĝ′)/(|ĝ|² + |ĝ′|²); where g = h, E = A/ĝ.
//!
//! Every point value carries a factor that takes the exponential of the scaled radial
//! functions back to a constant, chosen so that no applied exponent is positive:
//! e^{Im k (t − ρ − R)} for (S|R) (R ≥ t − ρ), e^{Im k (R − t − ρ)} for the inner (R|R)
//! (R ≤ t + ρ) and e^{Im k (ρ − t − R)} for the outer (S|S) (R ≥ ρ − t). With the target
//! functions scaled as in [`crate::radial`], A and B are then the stored Ê times ĝ and ĝ′.
//!
//! The Legendre columns come from [`crate::orthonormal::legendre`]'s recursion (accurate
//! near the poles, no overflow at any degree), one order at a time per point. The
//! derivative of p̄ₙᵐ(u) along ρ uses ∂u/∂ρ = −ρ t sin²α / R³ and
//! (1 − u²) p̄ₙᵐ′ = √((2n + 1)(n − m)(n + m)/(2n − 1)) p̄ₙ₋₁ᵐ − n u p̄ₙᵐ, whose sin² factors
//! cancel: ∂ρ p̄ₙᵐ(u) = −t/(ρR) (√(…) p̄ₙ₋₁ᵐ − n u p̄ₙᵐ).

use super::CoaxialTable;
use crate::orthonormal::legendre::Columns;
use crate::radial::{bessel_j, div, hankel_h};
use crate::{Complex, RealScalar};

/// Which projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// (S|R): hₙ onto jₙ on ρ < t, least squares; scaled by s^{n+n′}.
    SingularRegular,
    /// (S|S) = (R|R): hₙ onto hₙ on ρ > t; target degree ≥ source degree.
    RegularOuter,
    /// (R|R): jₙ onto jₙ on any ρ, least squares; target degree ≤ source degree.
    RegularInner,
}

/// The projection's parameters: the kind, k, the shift t, the sphere's radius ρ, the
/// power scale s (1 except for (S|R)) and the number of Gauss–Legendre nodes.
pub(super) struct Sphere<T> {
    pub(super) kind: Kind,
    pub(super) k: Complex<T>,
    pub(super) t: T,
    pub(super) rho: T,
    pub(super) s: T,
    pub(super) nodes: usize,
}

/// One quadrature point and its source functions.
struct Point<T> {
    weight: T,
    /// The new centre's Legendre columns, at x = cos α.
    new: Columns<T>,
    /// The old centre's Legendre columns, at u = cos Θ.
    old: Columns<T>,
    /// u, ∂R/∂ρ and −t/(ρR).
    u: T,
    dr: T,
    du_factor: T,
    /// fₙ(kR) and fₙ′(kR), n ≤ the largest source degree, times the point's factor.
    values: Vec<Complex<T>>,
    derivatives: Vec<Complex<T>>,
}

/// The table of degrees p_src → p_tgt by the projection `sphere`.
pub(super) fn build<T: RealScalar>(
    sphere: Sphere<T>,
    p_src: usize,
    p_tgt: usize,
) -> CoaxialTable<T> {
    let Sphere {
        kind,
        k,
        t,
        rho,
        s,
        nodes,
    } = sphere;
    assert!(nodes >= 1, "a projection needs at least one node");
    let mut table = CoaxialTable::zeros(p_src, p_tgt);
    let (rows, cols) = (table.rows, table.cols);
    let zero = Complex::new(T::zero(), T::zero());
    let least_squares = kind != Kind::RegularOuter;
    // Target and source degree ranges of the half the projection computes.
    let (t_max, s_max) = match kind {
        Kind::RegularOuter => (rows, cols),
        Kind::SingularRegular | Kind::RegularInner => (cols, rows),
    };
    let (xs, ws) = gauss_legendre::<T>(nodes);
    let two = T::from_f64(2.0);
    let mut points: Vec<Point<T>> = xs
        .iter()
        .zip(&ws)
        .map(|(&x, &w)| {
            let sx = ((T::one() - x) * (T::one() + x)).sqrt();
            let r = (rho * rho + t * t + two * rho * t * x).sqrt();
            let u = (rho * x + t) / r;
            let su = rho * sx / r;
            let mut values = vec![zero; s_max + 1];
            let mut derivatives = vec![zero; if least_squares { s_max + 1 } else { 0 }];
            let der = least_squares.then_some(&mut derivatives[..]);
            let exponent = match kind {
                Kind::SingularRegular => {
                    hankel_h(k * r, s, &mut values, der);
                    k.im * (t - rho - r)
                }
                Kind::RegularInner => {
                    bessel_j(k * r, T::one(), &mut values, der);
                    k.im * (r - t - rho)
                }
                Kind::RegularOuter => {
                    hankel_h(k * r, T::one(), &mut values, der);
                    k.im * (rho - t - r)
                }
            };
            let factor = exponent.min(T::zero()).exp();
            for v in values.iter_mut().chain(derivatives.iter_mut()) {
                *v = *v * factor;
            }
            Point {
                weight: w,
                new: Columns::new(x, sx, false),
                old: Columns::new(u, su, false),
                u,
                dr: (rho + t * x) / r,
                du_factor: -t / (rho * r),
                values,
                derivatives,
            }
        })
        .collect();
    // The target functions at kρ: ĵ (scaled by s) and ĵ′, or ĥ. ĵ is evaluated to the
    // square table's degree, max(p_src, p_tgt), since Miller's start order grows with the
    // length; the source functions already are (s_max = rows), and ĥ does not depend on
    // it. A rectangular table is then a block of the square one bit for bit.
    let z = k * rho;
    let target_len = if least_squares { rows + 1 } else { t_max + 1 };
    let mut target = vec![zero; target_len];
    let mut target_der = vec![zero; if least_squares { target_len } else { 0 }];
    match kind {
        Kind::SingularRegular => bessel_j(z, s, &mut target, Some(&mut target_der)),
        Kind::RegularInner => bessel_j(z, T::one(), &mut target, Some(&mut target_der)),
        Kind::RegularOuter => hankel_h(z, T::one(), &mut target, None),
    }
    let width = s_max + 1;
    let mut acc_a = vec![zero; (t_max + 1) * width];
    let mut acc_b = vec![
        zero;
        if least_squares {
            (t_max + 1) * width
        } else {
            0
        }
    ];
    let mut pn = vec![T::zero(); t_max + 1];
    let mut po = vec![T::zero(); s_max + 1];
    let mut f = vec![zero; s_max + 1];
    let mut df = vec![zero; s_max + 1];
    // The source range of target degree n′ at order m.
    let sources = |m: usize, n1: usize| match kind {
        Kind::RegularOuter => (m, n1.min(cols)),
        Kind::SingularRegular | Kind::RegularInner => (n1, rows),
    };
    for m in 0..=cols {
        acc_a.fill(zero);
        acc_b.fill(zero);
        let mf = m as f64;
        for q in points.iter_mut() {
            q.new.next(t_max, |n, v| {
                pn[n] = T::from_f64(((2 * n + 1) as f64 / 2.0).sqrt()) * v;
            });
            q.old.next(s_max, |n, v| {
                po[n] = T::from_f64(((2 * n + 1) as f64 / 2.0).sqrt()) * v;
            });
            for n in m..=s_max {
                f[n] = q.values[n] * po[n];
            }
            if least_squares {
                for n in m..=s_max {
                    let nf = n as f64;
                    let lower = if n > m {
                        T::from_f64(
                            ((2.0 * nf + 1.0) * (nf - mf) * (nf + mf) / (2.0 * nf - 1.0)).sqrt(),
                        ) * po[n - 1]
                    } else {
                        T::zero()
                    };
                    let dpo = q.du_factor * (lower - T::from_f64(nf) * q.u * po[n]);
                    df[n] = q.derivatives[n] * (q.dr * po[n]) + div(q.values[n] * dpo, k);
                }
            }
            for (n1, &p_target) in pn.iter().enumerate().take(t_max + 1).skip(m) {
                let (lo, hi) = sources(m, n1);
                let wp = q.weight * p_target;
                let row = n1 * width;
                for (acc, v) in acc_a[row + lo..=row + hi].iter_mut().zip(&f[lo..=hi]) {
                    *acc = *acc + *v * wp;
                }
                if least_squares {
                    for (acc, v) in acc_b[row + lo..=row + hi].iter_mut().zip(&df[lo..=hi]) {
                        *acc = *acc + *v * wp;
                    }
                }
            }
        }
        for n1 in m..=t_max {
            let (lo, hi) = sources(m, n1);
            for n in lo..=hi {
                let i = n1 * width + n;
                let e = if least_squares {
                    let (g, gd) = (target[n1], target_der[n1]);
                    let num = acc_a[i] * g.conj() + acc_b[i] * gd.conj();
                    num / (g.norm_sqr() + gd.norm_sqr())
                } else {
                    div(acc_a[i], target[n1])
                };
                // Êᵐ_{n′n} with n′ = n1 the target degree; the triangle stores r ≥ c.
                if n1 >= n {
                    table.set(m, n1, n, e);
                } else {
                    table.set(m, n, n1, if (n + n1) % 2 == 1 { -e } else { e });
                }
            }
        }
    }
    table
}

/// Gauss–Legendre nodes and weights on [−1, 1] by Newton's method on P_N, from the
/// asymptotic guesses cos(π (i + 3/4)/(N + 1/2)).
fn gauss_legendre<T: RealScalar>(n: usize) -> (Vec<T>, Vec<T>) {
    let mut x = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];
    let nf = T::from_f64(n as f64);
    for i in 0..n.div_ceil(2) {
        let mut z =
            T::from_f64((std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos());
        let mut dp = T::one();
        for _ in 0..100 {
            let (mut p0, mut p1) = (T::one(), z);
            for j in 2..=n {
                let jf = T::from_f64(j as f64);
                let p2 = ((jf + jf - T::one()) * z * p1 - (jf - T::one()) * p0) / jf;
                p0 = p1;
                p1 = p2;
            }
            let (p, pm1) = if n == 1 { (z, T::one()) } else { (p1, p0) };
            dp = nf * (z * p - pm1) / (z * z - T::one());
            let dz = p / dp;
            z = z - dz;
            if dz.abs() <= T::epsilon() {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        let wi = T::from_f64(2.0) / ((T::one() - z * z) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

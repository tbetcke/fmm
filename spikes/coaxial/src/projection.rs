//! The coaxial coefficients by projection ("point and shoot", as FMM3D's stabilised
//! translations): the translated wave is evaluated on a sphere about the new centre and
//! projected onto the Legendre functions of each order by Gauss–Legendre quadrature in
//! the polar angle. No recursion in the degrees, so no amplification of rounding; the
//! cost is O(N p) per order and target degree, O(N p³) per table.
//!
//! With p̄ₙᵐ the orthonormal associated Legendre functions on [−1, 1] (no
//! Condon–Shortley phase), a wave of order m restricted to φ = 0 carries the same φ
//! factor on both sides, so for the new-centre point r = ρ (sin α, 0, cos α):
//! fₙ(k R) p̄ₙᵐ(cos Θ) = Σ_{n′} Eᵐ_{n′n} g_{n′}(k ρ) p̄_{n′}ᵐ(cos α), with R and Θ the
//! polar coordinates of r + t about the old centre.
//!
//! - (S|R): f = h, g = j, ρ < t (ρ the target box's radius). jₙ(kρ) has zeros for
//!   n < |k|ρ, so each degree is solved by least squares from the value and the radial
//!   derivative: E = (A conj ĵ + B conj ĵ′)/(|ĵ|² + |ĵ′|²), which never divides by a
//!   small number where the evaluation does not also make the coefficient small.
//! - (R|R) = (S|S): f = g = h on a sphere ρ > t, where hₙ(kρ) never vanishes.

use crate::complex::C64;

/// A coaxial table by projection: per order m a dense (p_tgt − m + 1) × (p_src − m + 1)
/// block, row n′, column n.
pub struct Dense {
    /// Largest source degree.
    pub p_src: usize,
    /// Largest target degree.
    pub p_tgt: usize,
    blocks: Vec<Vec<C64>>,
}

impl Dense {
    /// Êᵐ_{n′n} for m ≤ n′ ≤ p_tgt, m ≤ n ≤ p_src.
    pub fn get(&self, m: usize, n1: usize, n: usize) -> C64 {
        self.blocks[m][(n1 - m) * (self.p_src - m + 1) + (n - m)]
    }
}

/// The quadrature order used by default: max(p_src, p_tgt) + 40, converged to the
/// rounding floor on every configuration of the sweep (SPIKE_REPORT.md).
pub fn default_nodes(p_src: usize, p_tgt: usize) -> usize {
    p_src.max(p_tgt) + 40
}

/// Gauss–Legendre nodes and weights on [−1, 1] by Newton's method on P_N.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let p = if n == 0 { 1.0 } else { p1 };
            let pm1 = if n == 1 { 1.0 } else { p0 };
            dp = n as f64 * (z * p - pm1) / (z * z - 1.0);
            let dz = p / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

/// aᵐₙ of the three-term recursion u p̄ₙᵐ = aᵐₙ p̄ᵐₙ₊₁ + aᵐₙ₋₁ p̄ᵐₙ₋₁.
#[inline]
fn a(n: usize, m: usize) -> f64 {
    let (n, m) = (n as f64, m as f64);
    (((n + 1.0 + m) * (n + 1.0 - m)) / ((2.0 * n + 1.0) * (2.0 * n + 3.0))).sqrt()
}

/// p̄ₙᵐ(u) for n = m..=n_max into out[n − m], from the sectorial value `sect` = p̄ₘᵐ(u).
fn legendre_column(u: f64, m: usize, sect: f64, out: &mut [f64]) {
    out[0] = sect;
    if out.len() > 1 {
        out[1] = u * ((2 * m + 3) as f64).sqrt() * sect;
    }
    for k in 2..out.len() {
        let n = m + k - 1;
        out[k] = (u * out[k - 1] - a(n - 1, m) * out[k - 2]) / a(n, m);
    }
}

/// d p̄ₙᵐ / du for n = m..=n_max from the values, by
/// (1 − u²) p̄ₙᵐ′ = √((2n + 1)(n − m)(n + m)/(2n − 1)) p̄ᵐₙ₋₁ − n u p̄ₙᵐ.
fn legendre_derivative(u: f64, m: usize, values: &[f64], out: &mut [f64]) {
    let s2 = 1.0 - u * u;
    for k in 0..values.len() {
        let n = m + k;
        let lower = if k == 0 {
            0.0
        } else {
            let (nf, mf) = (n as f64, m as f64);
            ((2.0 * nf + 1.0) * (nf - mf) * (nf + mf) / (2.0 * nf - 1.0)).sqrt() * values[k - 1]
        };
        out[k] = (lower - n as f64 * u * values[k]) / s2;
    }
}

/// Sectorial values p̄ₘᵐ(u), m = 0..=m_max: p̄₀⁰ = 1/√2, p̄ₘᵐ = p̄ᵐ⁻¹ₘ₋₁ s √((2m + 1)/(2m)).
fn sectorials(u: f64, m_max: usize) -> Vec<f64> {
    let s = (1.0 - u * u).sqrt();
    let mut out = vec![0.0; m_max + 1];
    out[0] = std::f64::consts::FRAC_1_SQRT_2;
    for m in 1..=m_max {
        out[m] = out[m - 1] * s * ((2 * m + 1) as f64 / (2 * m) as f64).sqrt();
    }
    out
}

/// One evaluation point of the projection about the old centre.
struct Point {
    weight: f64,
    /// cos α on the new sphere.
    x: f64,
    /// R = |r + t|, its derivative ∂R/∂ρ, the polar cosine u about the old centre and
    /// ∂u/∂ρ.
    r: f64,
    dr: f64,
    u: f64,
    du: f64,
    /// hₙ(kR) gⁿ (n ≤ p_src + 1) times a constant: e^{Im kt} (inner sphere) or e^{Im kρ}
    /// (outer sphere), stored as ĥₙ(kR) gⁿ e^{Im k (t − R)} or e^{Im k (ρ − R)}.
    h: Vec<C64>,
}

/// The coaxial (S|R) table by projection, in the spike's normalisation
/// Êᵐ_{n′n} = Eᵐ_{n′n} e^{Im kt} g^{n+n′}. `k` the wavenumber, `t` the shift, `rho` the radius of the
/// projection sphere (< t), `nodes` the Gauss–Legendre order.
pub fn singular_regular(
    k: C64,
    t: f64,
    rho: f64,
    g: f64,
    p_src: usize,
    p_tgt: usize,
    nodes: usize,
) -> Dense {
    inner(k, t, rho, g, p_src, p_tgt, nodes, false)
}

/// The coaxial (R|R) table by the same projection on a sphere of radius `rho` < t
/// (jₙ onto jₙ, least squares with the derivative), scaled as [`singular_regular`];
/// real k only. It shares the inner sphere, and so the M2L weights, with
/// [`singular_regular`], which the reality check Re (S|R) = (R|R) needs.
pub fn regular_inner(
    k: C64,
    t: f64,
    rho: f64,
    g: f64,
    p_src: usize,
    p_tgt: usize,
    nodes: usize,
) -> Dense {
    assert_eq!(k.im, 0.0, "real k only");
    inner(k, t, rho, g, p_src, p_tgt, nodes, true)
}

#[allow(clippy::too_many_arguments)]
fn inner(
    k: C64,
    t: f64,
    rho: f64,
    g: f64,
    p_src: usize,
    p_tgt: usize,
    nodes: usize,
    regular: bool,
) -> Dense {
    let (xs, ws) = gauss_legendre(nodes);
    let points = make_points(&xs, &ws, k, t, rho, g, p_src + 1, false, regular);
    // ĵₙ(kρ)/gⁿ and its derivative ĵₙ′ = (n/z) ĵₙ − ĵₙ₊₁, both times e^{−|Im kρ|}.
    let z = k * rho;
    let mut jv = vec![C64::ZERO; p_tgt + 2];
    crate::radial::bessel_j(z, &mut jv);
    let mut gi = 1.0;
    let mut jn = Vec::with_capacity(p_tgt + 1);
    let mut jd = Vec::with_capacity(p_tgt + 1);
    for n in 0..=p_tgt {
        jn.push(jv[n] / gi);
        jd.push((jv[n] * n as f64 / z - jv[n + 1]) / gi);
        gi *= g;
    }
    // The point values carry e^{Im k (t − R)}; ĵ carries e^{−|Im kρ|} relative to j.
    let fix = (-(k.im * rho).abs()).exp();
    project(
        &points,
        k,
        g,
        p_src,
        p_tgt,
        |n1, value, deriv| {
            let (j, d) = (jn[n1], jd[n1]);
            let num = value * C64::new(j.re, -j.im) + deriv * C64::new(d.re, -d.im);
            num / (j.re * j.re + j.im * j.im + d.re * d.re + d.im * d.im) * fix
        },
        true,
    )
}

/// The coaxial (R|R) = (S|S) table by projection on a sphere of radius `rho` > t, in the
/// spike's normalisation Êᵐ_{n′n} = Eᵐ_{n′n} e^{−|Im kt|} (unscaled).
pub fn regular_regular(
    k: C64,
    t: f64,
    rho: f64,
    p_src: usize,
    p_tgt: usize,
    nodes: usize,
) -> Dense {
    let (xs, ws) = gauss_legendre(nodes);
    let points = make_points(&xs, &ws, k, t, rho, 1.0, p_src, true, false);
    let z = k * rho;
    let mut hv = vec![C64::ZERO; p_tgt + 1];
    crate::radial::hankel(z, 1.0, &mut hv);
    // Points carry e^{Im k (ρ − R)}, ĥ(kρ) carries e^{Im kρ} relative to h; the target
    // normalisation is e^{−|Im kt|}.
    let fix = (-(k.im * t).abs()).exp();
    project(
        &points,
        k,
        1.0,
        p_src,
        p_tgt,
        |n1, value, _| value / hv[n1] * fix,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn make_points(
    xs: &[f64],
    ws: &[f64],
    k: C64,
    t: f64,
    rho: f64,
    g: f64,
    n_max: usize,
    outer: bool,
    regular: bool,
) -> Vec<Point> {
    xs.iter()
        .zip(ws)
        .map(|(&x, &w)| {
            let r2 = rho * rho + t * t + 2.0 * rho * t * x;
            let r = r2.sqrt();
            let dr = (rho + t * x) / r;
            let u = (rho * x + t) / r;
            let du = (x * r - (rho * x + t) * dr) / r2;
            let mut h = vec![C64::ZERO; n_max + 1];
            if regular {
                // jₙ(kR) gⁿ; real k only (no exponential factor to carry).
                crate::radial::bessel_j(k * r, &mut h);
                let mut s = 1.0;
                for v in h.iter_mut() {
                    *v = *v * s;
                    s *= g;
                }
            } else {
                crate::radial::hankel(k * r, g, &mut h);
            }
            // ĥ(kR) = h(kR) e^{Im kR}; rescale to e^{Im k (t − R)} (inner sphere, the
            // (S|R) normalisation) or e^{Im k (ρ − R)} (outer sphere).
            let shift = if outer { rho } else { t };
            let e = (-k.im * r + k.im * shift).exp();
            for v in h.iter_mut() {
                *v = *v * e;
            }
            Point {
                weight: w,
                x,
                r,
                dr,
                u,
                du,
                h,
            }
        })
        .collect()
}

/// For each order m: A[n′][n] = Σᵢ wᵢ p̄_{n′}ᵐ(xᵢ) Fₙ(i) and, if `derivative`, the
/// same with ∂Fₙ/∂ρ / k, where Fₙ(i) = ĥₙ(kRᵢ) p̄ₙᵐ(uᵢ); then `finish(n′, A, B)`.
fn project(
    points: &[Point],
    k: C64,
    g: f64,
    p_src: usize,
    p_tgt: usize,
    finish: impl Fn(usize, C64, C64) -> C64,
    derivative: bool,
) -> Dense {
    let top = p_src.min(p_tgt);
    let sect_new: Vec<Vec<f64>> = points.iter().map(|q| sectorials(q.x, top)).collect();
    let sect_old: Vec<Vec<f64>> = points.iter().map(|q| sectorials(q.u, top)).collect();
    let mut out = Vec::with_capacity(top + 1);
    let mut pn = vec![0.0; p_tgt + 1];
    let mut po = vec![0.0; p_src + 1];
    let mut dpo = vec![0.0; p_src + 1];
    let mut f = vec![C64::ZERO; p_src + 1];
    let mut df = vec![C64::ZERO; p_src + 1];
    for m in 0..=top {
        let (rows, cols) = (p_tgt - m + 1, p_src - m + 1);
        let mut acc_a = vec![C64::ZERO; rows * cols];
        let mut acc_b = vec![C64::ZERO; rows * cols];
        for (i, q) in points.iter().enumerate() {
            legendre_column(q.x, m, sect_new[i][m], &mut pn[..rows]);
            legendre_column(q.u, m, sect_old[i][m], &mut po[..cols]);
            for c in 0..cols {
                f[c] = q.h[m + c] * po[c];
            }
            if derivative {
                legendre_derivative(q.u, m, &po[..cols], &mut dpo[..cols]);
                for c in 0..cols {
                    let n = m + c;
                    // hₙ′(z) = (n/z) hₙ − hₙ₊₁ on the scaled values: ĥₙ′ = (n/z) ĥₙ − ĥₙ₊₁/g.
                    let hprime = q.h[n] * (n as f64) / (k * q.r) - q.h[n + 1] / g;
                    // ∂/∂ρ of h(kR) p̄(u), divided by k.
                    df[c] = hprime * po[c] * q.dr + q.h[n] * dpo[c] * q.du / k;
                }
            }
            for r in 0..rows {
                let wp = q.weight * pn[r];
                let row_a = &mut acc_a[r * cols..(r + 1) * cols];
                for c in 0..cols {
                    row_a[c] += f[c] * wp;
                }
                if derivative {
                    let row_b = &mut acc_b[r * cols..(r + 1) * cols];
                    for c in 0..cols {
                        row_b[c] += df[c] * wp;
                    }
                }
            }
        }
        let mut table = vec![C64::ZERO; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                table[r * cols + c] = finish(m + r, acc_a[r * cols + c], acc_b[r * cols + c]);
            }
        }
        out.push(table);
    }
    Dense {
        p_src,
        p_tgt,
        blocks: out,
    }
}

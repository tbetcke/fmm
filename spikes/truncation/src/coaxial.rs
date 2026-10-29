//! Coaxial translation coefficients (E|F)ᵐ_{n′n} along +z in S2's basis and convention
//! (`spikes/coaxial`), in either precision: Fₙᵐ(r + t ẑ) = Σ_{n′} Eᵐ_{n′n} G_{n′}ᵐ(r), one
//! coefficient for the slots ±m; (R|R) = (S|S) from jₗ, (S|R) from hₗ.
//!
//! Two constructions of the same table, as S2 signed off: S2's recursions in the
//! triangle n′ ≥ n (the rest by Eᵐ_{nn′} = (−1)^{n+n′} Eᵐ_{n′n}), and S2's projection on a
//! sphere about the new centre. Both give a square table of degree P, stored as one
//! dense block per order m (rows n′, columns n, m ≤ n, n′ ≤ P); a table for degrees
//! p_src, p_tgt ≤ P is its leading block (S2: rectangular truncation is a block, bit for
//! bit for the recursion).
//!
//! The (S|R) table is scaled symmetrically, Êᵐ_{n′n} = Eᵐ_{n′n} g^{n+n′}, with g the
//! level's scale; (R|R) is unscaled (the chain applies the two levels' scales).

use crate::complex::Complex;
use crate::radial;
use crate::real::Real;

/// Which coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// (R|R) = (S|S): M2M and L2L.
    Regular,
    /// (S|R): M2L.
    Singular,
}

/// A square coaxial table of degree P, a dense block per order.
pub struct Table<T> {
    /// The degree P.
    pub p: usize,
    blocks: Vec<Vec<Complex<T>>>,
}

impl<T: Real> Table<T> {
    /// Êᵐ_{n′n}, m ≤ n′, n ≤ P.
    #[inline]
    pub fn get(&self, m: usize, n1: usize, n: usize) -> Complex<T> {
        self.blocks[m][(n1 - m) * (self.p - m + 1) + (n - m)]
    }

    /// The block of order m: rows n′ = m..=P, columns n = m..=P, row-major.
    pub fn block(&self, m: usize) -> &[Complex<T>] {
        &self.blocks[m]
    }

    /// out[n′, ±m] = Σₙ Êᵐ_{n′n} f(n′, n) c[n, ±m] for n ≤ p_src, n′ ≤ p_tgt,
    /// m ≤ min(p_src, p_tgt), with f = 1 (`scale` None, the M2L, whose table carries the
    /// scale) or a shift's [`ShiftScale`]; the other entries of `out` (orders above
    /// p_src) are zero.
    pub fn apply(
        &self,
        p_src: usize,
        p_tgt: usize,
        scale: Option<&ShiftScale<T>>,
        c: &[Complex<T>],
        out: &mut [Complex<T>],
    ) {
        use crate::harmonics::idx;
        assert!(p_src.max(p_tgt) <= self.p);
        out[..(p_tgt + 1) * (p_tgt + 1)].fill(Complex::ZERO);
        for m in 0..=p_src.min(p_tgt) {
            let mi = m as isize;
            let block = &self.blocks[m];
            let w = self.p - m + 1;
            for n1 in m..=p_tgt {
                let row = &block[(n1 - m) * w..(n1 - m) * w + (p_src - m + 1)];
                let (mut ap, mut am) = (Complex::ZERO, Complex::ZERO);
                for (e, n) in row.iter().zip(m..=p_src) {
                    let e = match scale {
                        Some(s) => *e * s.factor(n1, n),
                        None => *e,
                    };
                    ap += e * c[idx(n, mi)];
                    if m > 0 {
                        am += e * c[idx(n, -mi)];
                    }
                }
                out[idx(n1, mi)] = ap;
                if m > 0 {
                    out[idx(n1, -mi)] = am;
                }
            }
        }
    }
}

/// The scale factors of a shift between levels with scales s_src and s_tgt, on the
/// unscaled (R|R) table, per entry so that no power of a scale alone leaves the exponent
/// range (the unscaled coefficients themselves do, at low frequency and large p):
///
/// - M2M: M̂_{t,n′} = Σ E_{n′n} s_srcⁿ s_tgt^{−n′} M̂_{s,n}, f = (s_src/s_tgt)ⁿ s_tgt^{n−n′};
/// - L2L: L̂_{t,n′} = Σ E_{n′n} s_src^{−n} s_tgt^{n′} L̂_{s,n}, f = (s_tgt/s_src)^{n′} s_src^{n′−n}.
pub struct ShiftScale<T> {
    m2m: bool,
    p: usize,
    ratio: Vec<T>,
    base: Vec<T>,
}

impl<T: Real> ShiftScale<T> {
    /// The factors for degrees to p.
    pub fn new(m2m: bool, s_src: T, s_tgt: T, p: usize) -> Self {
        let (r, b) = if m2m {
            (s_src / s_tgt, s_tgt)
        } else {
            (s_tgt / s_src, s_src)
        };
        let mut ratio = Vec::with_capacity(p + 1);
        let mut v = T::ONE;
        for _ in 0..=p {
            ratio.push(v);
            v *= r;
        }
        // base^e for e = −p..=p at e + p.
        let mut base = vec![T::ONE; 2 * p + 1];
        let inv = T::ONE / b;
        for e in 1..=p {
            base[p + e] = base[p + e - 1] * b;
            base[p - e] = base[p - e + 1] * inv;
        }
        ShiftScale {
            m2m,
            p,
            ratio,
            base,
        }
    }

    /// f(n′, n).
    #[inline]
    pub fn factor(&self, n1: usize, n: usize) -> T {
        if self.m2m {
            self.ratio[n] * self.base[self.p + n - n1]
        } else {
            self.ratio[n1] * self.base[self.p + n1 - n]
        }
    }
}

/// aᵐₙ, zero for n < m.
fn a<T: Real>(n: usize, m: usize) -> T {
    if n < m {
        return T::ZERO;
    }
    (T::from_usize((n + 1 + m) * (n + 1 - m)) / T::from_usize((2 * n + 1) * (2 * n + 3))).sqrt()
}

/// βₙᵐ, zero where (n − m − 1)(n − m) ≤ 0.
fn beta<T: Real>(n: usize, m: usize) -> T {
    if n < m + 2 {
        return T::ZERO;
    }
    (T::from_usize((n - m - 1) * (n - m)) / T::from_usize((2 * n - 1) * (2 * n + 1))).sqrt()
}

/// γₙᵐ.
fn gamma<T: Real>(n: usize, m: usize) -> T {
    (T::from_usize((n + m + 1) * (n + m + 2)) / T::from_usize((2 * n + 1) * (2 * n + 3))).sqrt()
}

/// The start values f̂ₗ, l = 0..=l_max: jₗ(z) gˡ for [`Op::Regular`] (S2's normalisation
/// with g = 1 here), hₗ(z) gˡ for [`Op::Singular`].
pub fn start_values<T: Real>(op: Op, z: Complex<T>, g: T, l_max: usize) -> Vec<Complex<T>> {
    let mut out = vec![Complex::ZERO; l_max + 1];
    match op {
        Op::Regular => {
            // ĵ = j/gˡ from radial; jₗ gˡ needs g^{2l} more; the spike uses g = 1 here.
            radial::bessel_j(z, T::ONE, &mut out);
            let mut s = T::ONE;
            for v in out.iter_mut() {
                *v = *v * s;
                s *= g;
            }
        }
        Op::Singular => radial::hankel(z, g, &mut out),
    }
    out
}

/// The table of degree P by S2's recursions (the triangle variant) from the start values
/// `start` (orders 0..=2P) with scale g.
pub fn recursion<T: Real>(start: &[Complex<T>], g: T, p: usize) -> Table<T> {
    let l_max = 2 * p;
    assert!(start.len() > l_max, "start needs orders 0..=2P");
    let g2 = g * g;
    let len = l_max + 3;
    let at = |n1: usize| n1 + 1;
    let mut blocks: Vec<Vec<Complex<T>>> = (0..=p)
        .map(|m| vec![Complex::ZERO; (p - m + 1) * (p - m + 1)])
        .collect();
    let mut sect = vec![Complex::ZERO; len];
    let mut tmp = vec![Complex::ZERO; len];
    let mut prev = vec![Complex::ZERO; len];
    let mut cur = vec![Complex::ZERO; len];
    let mut next = vec![Complex::ZERO; len];
    for (n1, f) in start[..=l_max].iter().enumerate() {
        let sign = if n1 % 2 == 1 { -T::ONE } else { T::ONE };
        sect[at(n1)] = *f * (sign * T::from_usize(2 * n1 + 1).sqrt());
    }
    // Stores column n (rows from `lo`) of order m into the triangle and its mirror.
    let store =
        |blocks: &mut Vec<Vec<Complex<T>>>, m: usize, n: usize, lo: usize, col: &[Complex<T>]| {
            let w = p - m + 1;
            for n1 in lo.max(n)..=p {
                let v = col[at(n1)];
                blocks[m][(n1 - m) * w + (n - m)] = v;
                if n1 != n {
                    let mirrored = if (n + n1) % 2 == 1 { -v } else { v };
                    blocks[m][(n - m) * w + (n1 - m)] = mirrored;
                }
            }
        };
    for m in 0..=p {
        if m > 0 {
            let mm = m - 1;
            let inv = T::ONE / gamma::<T>(mm, mm);
            tmp.fill(Complex::ZERO);
            for n1 in m..=l_max - m {
                tmp[at(n1)] = (sect[at(n1 + 1)] * beta::<T>(n1 + 1, mm)
                    + sect[at(n1 - 1)] * (g2 * gamma::<T>(n1 - 1, mm)))
                    * inv;
            }
            std::mem::swap(&mut sect, &mut tmp);
        }
        prev.fill(Complex::ZERO);
        cur.fill(Complex::ZERO);
        cur[at(m)..=at(l_max - m)].copy_from_slice(&sect[at(m)..=at(l_max - m)]);
        store(&mut blocks, m, m, m, &cur);
        for n in m..p {
            let lo = n + 1;
            let hi = l_max - (n + 1);
            let inv = T::ONE / a::<T>(n, m);
            let a_prev = g2 * if n == 0 { T::ZERO } else { a::<T>(n - 1, m) };
            next.fill(Complex::ZERO);
            for n1 in lo..=hi {
                let a_lower = if n1 == 0 { T::ZERO } else { a::<T>(n1 - 1, m) };
                next[at(n1)] = (prev[at(n1)] * a_prev - cur[at(n1 + 1)] * a::<T>(n1, m)
                    + cur[at(n1) - 1] * (g2 * a_lower))
                    * inv;
            }
            store(&mut blocks, m, n + 1, lo, &next);
            std::mem::swap(&mut prev, &mut cur);
            std::mem::swap(&mut cur, &mut next);
        }
    }
    Table { p, blocks }
}

/// Gauss–Legendre nodes and weights on [−1, 1]: Newton's method on P_N, first in f64 and
/// then three steps in T.
pub fn gauss_legendre<T: Real>(n: usize) -> (Vec<T>, Vec<T>) {
    let mut x = vec![T::ZERO; n];
    let mut w = vec![T::ZERO; n];
    let eval = |z: T| -> (T, T) {
        let (mut p0, mut p1) = (T::ONE, z);
        for k in 2..=n {
            let p2 =
                (T::from_usize(2 * k - 1) * z * p1 - T::from_usize(k - 1) * p0) / T::from_usize(k);
            p0 = p1;
            p1 = p2;
        }
        let p = if n == 0 { T::ONE } else { p1 };
        let pm1 = if n == 1 { T::ONE } else { p0 };
        (p, T::from_usize(n) * (z * p - pm1) / (z * z - T::ONE))
    };
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = eval(T::from_f64(z));
            let dz = (p / dp).to_f64();
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        let mut zt = T::from_f64(z);
        for _ in 0..3 {
            let (p, dp) = eval(zt);
            zt -= p / dp;
        }
        let (_, dp) = eval(zt);
        x[i] = -zt;
        x[n - 1 - i] = zt;
        let wi = T::from_f64(2.0) / ((T::ONE - zt * zt) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

/// aᵐₙ of the Legendre recursion (the same as the coaxial a).
fn la<T: Real>(n: usize, m: usize) -> T {
    a::<T>(n, m)
}

/// Orthonormal p̄ₙᵐ(u) for n = m..m + out.len() − 1, from the sectorial value.
fn legendre_column<T: Real>(u: T, m: usize, sect: T, out: &mut [T]) {
    out[0] = sect;
    if out.len() > 1 {
        out[1] = u * T::from_usize(2 * m + 3).sqrt() * sect;
    }
    for k in 2..out.len() {
        let n = m + k - 1;
        out[k] = (u * out[k - 1] - la::<T>(n - 1, m) * out[k - 2]) / la::<T>(n, m);
    }
}

/// d p̄ₙᵐ/du from the values: (1 − u²) p̄ₙᵐ′ = √((2n + 1)(n − m)(n + m)/(2n − 1)) p̄ᵐₙ₋₁ − n u p̄ₙᵐ.
fn legendre_derivative<T: Real>(u: T, m: usize, values: &[T], out: &mut [T]) {
    let s2 = T::ONE - u * u;
    for k in 0..values.len() {
        let n = m + k;
        let lower = if k == 0 {
            T::ZERO
        } else {
            (T::from_usize((2 * n + 1) * (n - m) * (n + m)) / T::from_usize(2 * n - 1)).sqrt()
                * values[k - 1]
        };
        out[k] = (lower - T::from_usize(n) * u * values[k]) / s2;
    }
}

/// Sectorial p̄ₘᵐ(u), m = 0..=m_max.
fn sectorials<T: Real>(u: T, m_max: usize) -> Vec<T> {
    let s = (T::ONE - u * u).sqrt();
    let mut out = vec![T::ZERO; m_max + 1];
    out[0] = T::ONE / T::from_f64(2.0).sqrt();
    for m in 1..=m_max {
        out[m] = out[m - 1] * s * (T::from_usize(2 * m + 1) / T::from_usize(2 * m)).sqrt();
    }
    out
}

/// The table of degree P by S2's projection, with `nodes` Gauss–Legendre nodes:
/// (S|R) on the sphere of radius ρ < t (least squares from value and radial derivative;
/// scaled by g^{n+n′}), or (R|R) = (S|S) on a sphere ρ > t (hₙ onto hₙ; unscaled).
pub fn projection<T: Real>(
    op: Op,
    k: Complex<T>,
    t: T,
    rho: T,
    g: T,
    p: usize,
    nodes: usize,
) -> Table<T> {
    let (xs, ws) = gauss_legendre::<T>(nodes);
    let singular = op == Op::Singular;
    let gg = if singular { g } else { T::ONE };
    // The source functions ĥₙ(kR) gⁿ (n ≤ P + 1) at each node, with R, ∂R/∂ρ, the polar
    // cosine u about the old centre and ∂u/∂ρ.
    struct Point<T> {
        w: T,
        x: T,
        r: T,
        dr: T,
        u: T,
        du: T,
        h: Vec<Complex<T>>,
    }
    let points: Vec<Point<T>> = xs
        .iter()
        .zip(&ws)
        .map(|(&x, &w)| {
            let r2 = rho * rho + t * t + T::from_f64(2.0) * rho * t * x;
            let r = r2.sqrt();
            let dr = (rho + t * x) / r;
            let u = (rho * x + t) / r;
            let du = (x * r - (rho * x + t) * dr) / r2;
            let mut h = vec![Complex::ZERO; p + 2];
            radial::hankel(k * r, gg, &mut h);
            Point {
                w,
                x,
                r,
                dr,
                u,
                du,
                h,
            }
        })
        .collect();
    // Target radial functions at ρ: (S|R) ĵₙ(kρ)/gⁿ and its derivative; (R|R) hₙ(kρ).
    let z = k * rho;
    let mut tv = vec![Complex::ZERO; p + 2];
    let mut fin_v = vec![Complex::ZERO; p + 1];
    let mut fin_d = vec![Complex::ZERO; p + 1];
    if singular {
        radial::bessel_j(z, g, &mut tv);
        for n in 0..=p {
            fin_v[n] = tv[n];
            fin_d[n] = tv[n] * T::from_usize(n) / z - tv[n + 1] * g;
        }
    } else {
        radial::hankel(z, T::ONE, &mut tv);
        fin_v[..=p].copy_from_slice(&tv[..=p]);
    }
    let sect_new: Vec<Vec<T>> = points.iter().map(|q| sectorials(q.x, p)).collect();
    let sect_old: Vec<Vec<T>> = points.iter().map(|q| sectorials(q.u, p)).collect();
    let mut blocks = Vec::with_capacity(p + 1);
    let mut pn = vec![T::ZERO; p + 1];
    let mut po = vec![T::ZERO; p + 1];
    let mut dpo = vec![T::ZERO; p + 1];
    let mut f = vec![Complex::ZERO; p + 1];
    let mut df = vec![Complex::ZERO; p + 1];
    for m in 0..=p {
        let w = p - m + 1;
        let mut acc_a = vec![Complex::ZERO; w * w];
        let mut acc_b = vec![Complex::ZERO; if singular { w * w } else { 0 }];
        for (i, q) in points.iter().enumerate() {
            legendre_column(q.x, m, sect_new[i][m], &mut pn[..w]);
            legendre_column(q.u, m, sect_old[i][m], &mut po[..w]);
            for c in 0..w {
                f[c] = q.h[m + c] * po[c];
            }
            if singular {
                legendre_derivative(q.u, m, &po[..w], &mut dpo[..w]);
                for c in 0..w {
                    let n = m + c;
                    // ĥₙ′ = (n/z) ĥₙ − ĥₙ₊₁/g, and ∂/∂ρ of h(kR) p̄(u) over k.
                    let hprime = q.h[n] * T::from_usize(n) / (k * q.r) - q.h[n + 1] / gg;
                    df[c] = hprime * po[c] * q.dr + q.h[n] * dpo[c] * q.du / k;
                }
            }
            for r in 0..w {
                let wp = q.w * pn[r];
                let row = &mut acc_a[r * w..(r + 1) * w];
                for c in 0..w {
                    row[c] += f[c] * wp;
                }
                if singular {
                    let row = &mut acc_b[r * w..(r + 1) * w];
                    for c in 0..w {
                        row[c] += df[c] * wp;
                    }
                }
            }
        }
        let mut block = vec![Complex::ZERO; w * w];
        for r in 0..w {
            let n1 = m + r;
            for c in 0..w {
                block[r * w + c] = if singular {
                    let (j, d) = (fin_v[n1], fin_d[n1]);
                    let num = acc_a[r * w + c] * j.conj() + acc_b[r * w + c] * d.conj();
                    num / Complex::real(j.norm_sqr() + d.norm_sqr())
                } else {
                    acc_a[r * w + c] / fin_v[n1]
                };
            }
        }
        blocks.push(block);
    }
    Table { p, blocks }
}

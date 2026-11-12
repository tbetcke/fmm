//! The transfer function of an offset, its bound, and its interpolation in ŝ · t̂.
//!
//! T_L(ŝ) = Σ_{l≤L} (2l + 1) iˡ hₗ(k|t|) Pₗ(ŝ · t̂) (proposal §3.6, [`crate::signature`]),
//! evaluated in double-double at each sample: hₗ by S7's upward recursion
//! (`nd_fmm_spike_truncation::radial::hankel`), Pₗ by the forward three-term recurrence,
//! which is stable for |μ| ≤ 1. One pass gives T at several bandwidths L (the snapshots),
//! since the partial sums are the lower bandwidths.
//!
//! The bound of §3.6 is E = C_T |k||t| u maxⱼ≤L (2j + 1)|hⱼ(k|t|)|; [`log10_term`] gives
//! log₁₀ of everything but C_T u, from the upward recursion in f64 with rescaling, so that
//! it is finite where hⱼ leaves the exponent range.
//!
//! [`Interp`] is the O(p²)-per-class build that §5.4 and R10 ask about: T is a cosine
//! polynomial of degree L in ψ = arccos(ŝ · t̂), so a table on a uniform grid in ψ,
//! oversampled by σ, determines it; the samples read it by sinc interpolation with a
//! Gaussian window (the regularised Shannon sampling series of Qian, 2003), 2h terms per
//! sample, in f64.

use nd_fmm_spike_truncation::complex::Complex;
use nd_fmm_spike_truncation::radial;
use nd_fmm_spike_truncation::real::{Dd, Real};

use crate::grid::round;

/// The transfer function of one offset at several bandwidths.
#[derive(Clone, Debug)]
pub struct Transfer {
    /// t̂.
    pub that: [Dd; 3],
    /// |t|.
    pub t: Dd,
    /// The bandwidths L, ascending.
    pub snaps: Vec<usize>,
    /// (2l + 1) iˡ hₗ(k|t|), l ≤ the largest snapshot.
    pub coeffs: Vec<Complex<Dd>>,
    /// (2l + 1)/(l + 1) and l/(l + 1), the recurrence's factors.
    alpha: Vec<Dd>,
    beta: Vec<Dd>,
}

impl Transfer {
    /// The transfer function of the offset `t` (in box widths) for wavenumber k (in the
    /// inverse unit), at the bandwidths `snaps` (ascending).
    pub fn new(k: Complex<Dd>, t: [i32; 3], snaps: &[usize]) -> Self {
        assert!(!snaps.is_empty() && snaps.windows(2).all(|w| w[0] < w[1]));
        let l_max = *snaps.last().unwrap();
        let tv = t.map(|x| Dd::from_f64(x as f64));
        let r = (tv[0] * tv[0] + tv[1] * tv[1] + tv[2] * tv[2]).sqrt();
        let that = tv.map(|x| x / r);
        let mut h = vec![Complex::ZERO; l_max + 1];
        radial::hankel(k * r, Dd::ONE, &mut h);
        let coeffs = h
            .iter()
            .enumerate()
            .map(|(l, &hl)| crate::signature::times_i_pow(hl, l).scale(Dd::from_usize(2 * l + 1)))
            .collect();
        let alpha = (0..=l_max)
            .map(|l| Dd::from_usize(2 * l + 1) / Dd::from_usize(l + 1))
            .collect();
        let beta = (0..=l_max)
            .map(|l| Dd::from_usize(l) / Dd::from_usize(l + 1))
            .collect();
        Transfer {
            that,
            t: r,
            snaps: snaps.to_vec(),
            coeffs,
            alpha,
            beta,
        }
    }

    /// ŝ · t̂.
    #[inline]
    pub fn mu(&self, s: [Dd; 3]) -> Dd {
        s[0] * self.that[0] + s[1] * self.that[1] + s[2] * self.that[2]
    }

    /// T_L(μ) for each snapshot L, into `out`.
    pub fn eval(&self, mu: Dd, out: &mut [Complex<Dd>]) {
        let (mut p0, mut p1) = (Dd::ONE, mu);
        let mut acc = self.coeffs[0];
        let mut si = 0;
        if self.snaps[0] == 0 {
            out[0] = acc;
            si = 1;
        }
        for l in 1..self.coeffs.len() {
            let c = self.coeffs[l];
            acc = Complex::new(acc.re + c.re * p1, acc.im + c.im * p1);
            if si < self.snaps.len() && self.snaps[si] == l {
                out[si] = acc;
                si += 1;
            }
            let p2 = self.alpha[l] * mu * p1 - self.beta[l] * p0;
            p0 = p1;
            p1 = p2;
        }
    }
}

/// The transfer function built in f64, as a host table build in working precision would
/// (the seed's model, Appendix B.1): the terms (2l + 1) iˡ hₗ rounded to f64 from
/// double-double, Pₗ(μ) by the recurrence and the sum in f64, μ from the f64 directions.
#[derive(Clone, Debug)]
pub struct TransferF64 {
    /// t̂ in f64.
    pub that: [f64; 3],
    /// The terms in f64, l ≤ L.
    pub coeffs: Vec<Complex<f64>>,
}

impl TransferF64 {
    /// The f64 build of `tr`'s snapshot `si`.
    pub fn new(tr: &Transfer, si: usize) -> Self {
        TransferF64 {
            that: tr.that.map(round::<f64>),
            coeffs: tr.coeffs[..=tr.snaps[si]]
                .iter()
                .map(|z| Complex::new(round::<f64>(z.re), round::<f64>(z.im)))
                .collect(),
        }
    }

    /// ŝ · t̂ in f64.
    pub fn mu(&self, s: [f64; 3]) -> f64 {
        s[0] * self.that[0] + s[1] * self.that[1] + s[2] * self.that[2]
    }

    /// T at the f64 direction ŝ.
    pub fn eval(&self, s: [f64; 3]) -> Complex<f64> {
        self.eval_mu(self.mu(s))
    }

    /// T at μ, in f64.
    pub fn eval_mu(&self, mu: f64) -> Complex<f64> {
        let (mut p0, mut p1) = (1.0, mu);
        let mut acc = self.coeffs[0];
        for (l, &c) in self.coeffs.iter().enumerate().skip(1) {
            acc += c * p1;
            let p2 = ((2 * l + 1) as f64 * mu * p1 - l as f64 * p0) / (l + 1) as f64;
            p0 = p1;
            p1 = p2;
        }
        acc
    }
}

/// log₁₀ of |z| maxⱼ≤L (2j + 1)|hⱼ(z)|, z = k|t| (the bound E of §3.6 without C_T u).
pub fn log10_term(z: Complex<f64>, l_max: usize) -> f64 {
    let e = (Complex::I * z).exp();
    let mut h0 = -(Complex::I * e) / z;
    let mut h1 = -(e * (z + Complex::I)) / (z * z);
    let mut offset = 0.0f64; // the h are 10^offset times the values kept
    let mut best = h0.abs().log10();
    if l_max >= 1 {
        best = best.max((3.0 * h1.abs()).log10());
    }
    for l in 1..l_max {
        let h2 = h1 * (2 * l + 1) as f64 / z - h0;
        h0 = h1;
        h1 = h2;
        best = best.max(((2 * l + 3) as f64 * h1.abs()).log10() + offset);
        if h1.abs() > 1e200 {
            h0 = h0 * 1e-200;
            h1 = h1 * 1e-200;
            offset += 200.0;
        }
    }
    best + z.abs().log10()
}

/// The angle ψ between ŝ and t̂ in double-double: atan2(|ŝ × t̂|, ŝ · t̂) in f64, corrected
/// by sin(ψ − ψ₀) = sin ψ cos ψ₀ − cos ψ sin ψ₀ in double-double. A table built once per
/// class affords it; ψ in f64 alone would cost T about L u |T| (|dT/dψ| ≈ L |T|), and
/// arccos of a rounded ŝ · t̂ about L² u |T| near the poles (Pₗ′(1) = l(l + 1)/2).
pub fn angle(s: [Dd; 3], that: [Dd; 3]) -> Dd {
    let c = [
        s[1] * that[2] - s[2] * that[1],
        s[2] * that[0] - s[0] * that[2],
        s[0] * that[1] - s[1] * that[0],
    ];
    let sin = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let cos = s[0] * that[0] + s[1] * that[1] + s[2] * that[2];
    let psi0 = round::<f64>(sin).atan2(round::<f64>(cos));
    let (s0, c0) = Dd::from_f64(psi0).sin_cos();
    Dd::from_f64(psi0) + (sin * c0 - cos * s0)
}

/// T at one bandwidth by interpolation in ψ = arccos(ŝ · t̂), in f64.
#[derive(Clone, Debug)]
pub struct Interp {
    /// M/π, the grid steps per radian.
    per_radian: Dd,
    /// M.
    m: usize,
    /// T(kπ/M), k = 0..=M, rounded from double-double.
    table: Vec<Complex<f64>>,
    /// Terms on each side of a sample.
    half: usize,
    /// 1/(2r²) of the Gaussian window, in steps.
    inv_two_r2: f64,
}

impl Interp {
    /// The table of `tr`'s snapshot `si` (bandwidth L) on M = ⌈σ L⌉ + 1 steps of [0, π],
    /// read with `half` terms on each side.
    pub fn new(tr: &Transfer, si: usize, sigma: f64, half: usize) -> Self {
        let l = tr.snaps[si];
        let m = (sigma * l as f64).ceil() as usize + 1;
        let step_dd = Dd::pi() / Dd::from_usize(m);
        let mut out = vec![Complex::ZERO; tr.snaps.len()];
        let table = (0..=m)
            .map(|k| {
                let (_, c) = (step_dd * Dd::from_usize(k)).sin_cos();
                tr.eval(c, &mut out);
                Complex::new(round::<f64>(out[si].re), round::<f64>(out[si].im))
            })
            .collect();
        // Band δ = L π/M radians per step; Qian's window r² = (half − 2)/(π − δ).
        let delta = l as f64 * std::f64::consts::PI / m as f64;
        let r2 = (half as f64 - 2.0).max(1.0) / (std::f64::consts::PI - delta);
        Interp {
            per_radian: Dd::from_usize(m) / Dd::pi(),
            m,
            table,
            half,
            inv_two_r2: 0.5 / r2,
        }
    }

    /// max |T| over the table, which includes ψ = 0 and π, where |T| peaks (T(0) = Σ cₗ):
    /// the interpolation's error is about u times this.
    pub fn max_abs(&self) -> f64 {
        self.table.iter().map(|z| z.abs()).fold(0.0, f64::max)
    }

    /// The table entry k of the even, 2π-periodic extension.
    #[inline]
    fn at(&self, k: isize) -> Complex<f64> {
        let period = 2 * self.m as isize;
        let mut k = k.rem_euclid(period);
        if k > self.m as isize {
            k = period - k;
        }
        self.table[k as usize]
    }

    /// T at the angle ψ between ŝ and t̂ ([`angle`]): the position on the grid in
    /// double-double, the sum in f64.
    pub fn eval(&self, psi: Dd) -> Complex<f64> {
        let x = psi * self.per_radian;
        let mut n0 = x.hi.floor();
        let mut f = (x - Dd::from_f64(n0)).to_f64();
        if f < 0.0 {
            n0 -= 1.0;
            f += 1.0;
        }
        // A tiny negative f rounds to f + 1 = 1 (ψ at a node from below, as for ŝ = t̂).
        if f >= 1.0 {
            n0 += 1.0;
            f -= 1.0;
        }
        let n0 = n0 as isize;
        if f == 0.0 {
            return self.at(n0);
        }
        // sin(π(f − d)) = (−1)^d sin(πf); for f near 1 as sin(π(1 − f)), 1 − f exact,
        // since π f rounds next to π and sin would lose its relative accuracy there.
        let g = if f > 0.5 { 1.0 - f } else { f };
        let s = (std::f64::consts::PI * g).sin() / std::f64::consts::PI;
        let mut acc = Complex::ZERO;
        for d in 1 - self.half as isize..=self.half as isize {
            let dist = f - d as f64;
            let sign = if d % 2 == 0 { 1.0 } else { -1.0 };
            let w = sign * s / dist * (-dist * dist * self.inv_two_r2).exp();
            acc += self.at(n0 + d) * w;
        }
        acc
    }
}

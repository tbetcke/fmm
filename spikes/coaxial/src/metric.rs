//! The FMM-weighted error of a coaxial table.
//!
//! A coaxial table spans tens to hundreds of orders of magnitude at large p (hₗ grows
//! and jₗ decays beyond l ≈ |k t|), but its entries only ever multiply coefficients that
//! the geometry bounds. Weighting each entry by the bounds of its input and output
//! measures the error of the field it produces:
//!
//! - **M2L, (S|R)** between boxes of circumradius a: a source multipole of charges within
//!   a has |Mₙ| ≤ Jₙ(a) √(2n + 1) per unit charge, Jₙ(a) = max_{r ≤ a} |jₙ(k r)|, and a
//!   local coefficient of degree n′ reaches targets within a through J_{n′}(a) √(2n′ + 1).
//!   Weight uₙ u_{n′}, uₙ = Jₙ(a) √(2n + 1). For n ≥ |k| a, |jₙ(k r)| grows with r and
//!   Jₙ(a) = |jₙ(k a)|; below, jₙ oscillates in r and the envelope is needed.
//! - **M2M and L2L, (R|R) = (S|S)** from a child (circumradius a/2) to its parent
//!   (circumradius a): the child's multipole is bounded by Jₙ(a/2) √(2n + 1), and the
//!   parent's multipole of degree n′ is evaluated no closer than the nearest M2L distance
//!   t_min = 2 w = 4a/√3, through |h_{n′}(k t_min)| √(2n′ + 1). The same weight serves
//!   L2L with the roles of the indices exchanged (|E_{n′n}| = |E_{nn′}|).
//!
//! The weighted error is max |ΔÊ| w / max |Ê| w over the table, so a common factor of the
//! weights (the exponential normalisation) cancels; in the scaled form Ê = E g^{n+n′} the
//! weights are divided by g^{n+n′}.

use crate::complex::C64;
use crate::radial;

/// Jₙ(R) = max_{0 < r ≤ R} |jₙ(k r)| for n ≤ n_max, up to the common factor
/// e^{|Im k| R}, sampled at 16 points per period π/|k| of the oscillation and at least
/// 16 in all (the maximum of a grid may fall short of the true one by about 2 %).
pub fn envelope(k: C64, radius: f64, n_max: usize) -> Vec<f64> {
    let samples = ((16.0 * k.abs() * radius / std::f64::consts::PI).ceil() as usize).max(16);
    let mut out = vec![0.0f64; n_max + 1];
    let mut j = vec![C64::ZERO; n_max + 1];
    for i in 1..=samples {
        let r = radius * i as f64 / samples as f64;
        radial::bessel_j(k * r, &mut j);
        // ĵ = j e^{−|Im k| r}; relative to e^{|Im k| R}: |j| e^{−|Im k| R}.
        let f = (k.im.abs() * (r - radius)).exp();
        for (o, v) in out.iter_mut().zip(&j) {
            *o = o.max(v.abs() * f);
        }
    }
    out
}

/// Weights of an M2L table at a level with circumradius a: w(n′, n) = uₙ′ uₙ with
/// uₙ = Jₙ(a) √(2n + 1) / gⁿ, n ≤ p.
pub fn m2l_weights(k: C64, a: f64, g: f64, p: usize) -> Vec<f64> {
    let mut s = 1.0;
    envelope(k, a, p)
        .iter()
        .enumerate()
        .map(|(n, v)| {
            let u = v * ((2 * n + 1) as f64).sqrt() / s;
            s *= g;
            u
        })
        .collect()
}

/// Input weights Jₙ(a/2) √(2n + 1), n ≤ p_child, and output weights
/// |ĥₙ′(k t_min)| √(2n′ + 1), n′ ≤ p_parent, of an M2M table (unscaled).
pub fn m2m_weights(k: C64, a: f64, p_child: usize, p_parent: usize) -> (Vec<f64>, Vec<f64>) {
    let mut h = vec![C64::ZERO; p_parent + 1];
    radial::hankel(k * (4.0 * a / 3.0f64.sqrt()), 1.0, &mut h);
    let input = envelope(k, 0.5 * a, p_child)
        .iter()
        .enumerate()
        .map(|(n, v)| v * ((2 * n + 1) as f64).sqrt())
        .collect();
    let output = h
        .iter()
        .enumerate()
        .map(|(n, v)| v.abs() * ((2 * n + 1) as f64).sqrt())
        .collect();
    (input, output)
}

/// Errors of `got` against `exact` over m ≤ min(p_src, p_tgt), m ≤ n′ ≤ p_tgt,
/// m ≤ n ≤ p_src: (weighted, entrywise relative over |exact| ≥ 1e-290), with
/// w(n′, n) = output[n′] input[n].
pub fn compare(
    p_src: usize,
    p_tgt: usize,
    input: &[f64],
    output: &[f64],
    exact: impl Fn(usize, usize, usize) -> C64,
    got: impl Fn(usize, usize, usize) -> C64,
) -> (f64, f64) {
    let (mut num, mut den, mut entry) = (0.0f64, 0.0f64, 0.0f64);
    for m in 0..=p_src.min(p_tgt) {
        for (n1, &w_out) in output.iter().enumerate().take(p_tgt + 1).skip(m) {
            for (n, &w_in) in input.iter().enumerate().take(p_src + 1).skip(m) {
                let (e, g) = (exact(m, n1, n), got(m, n1, n));
                let w = w_out * w_in;
                let err = (g - e).abs();
                num = num.max(err * w);
                den = den.max(e.abs() * w);
                if e.abs() >= 1e-290 {
                    entry = entry.max(err / e.abs());
                }
            }
        }
    }
    (num / den, entry)
}

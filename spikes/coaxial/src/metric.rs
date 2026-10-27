//! The FMM-weighted error of a coaxial table.
//!
//! A coaxial table spans tens to hundreds of orders of magnitude at large p (hₗ grows
//! and jₗ decays beyond l ≈ |k t|), but its entries only ever multiply coefficients that
//! the geometry bounds. Weighting each entry by the bounds of its input and output
//! measures the error of the field it produces:
//!
//! - **M2L, (S|R)** between boxes of circumradius a: a source multipole of charges within
//!   a has |Mₙ| ≤ |jₙ(k a)| √(2n + 1) per unit charge, and a local coefficient of degree
//!   n′ reaches targets within a through |j_{n′}(k a)| √(2n′ + 1). Weight
//!   uₙ u_{n′}, uₙ = |jₙ(k a)| √(2n + 1).
//! - **M2M and L2L, (R|R) = (S|S)** from a child (circumradius a/2) to its parent
//!   (circumradius a): the child's multipole is bounded by |jₙ(k a/2)| √(2n + 1), and the
//!   parent's multipole of degree n′ is evaluated no closer than the nearest M2L distance
//!   t_min = 2 w = 4a/√3, through |h_{n′}(k t_min)| √(2n′ + 1). The same weight serves
//!   L2L with the roles of the indices exchanged (|E_{n′n}| = |E_{nn′}|).
//!
//! The weighted error is max |ΔÊ| w / max |Ê| w over the table, so a common factor of the
//! weights (the exponential normalisation) cancels; in the scaled form Ê = E g^{n+n′} the
//! weights are divided by g^{n+n′}.

use crate::complex::C64;
use crate::radial;

/// Weights of an M2L table at a level with circumradius a: w(n′, n) = uₙ′ uₙ with
/// uₙ = |ĵₙ(k a)| √(2n + 1) / gⁿ, n ≤ p.
pub fn m2l_weights(k: C64, a: f64, g: f64, p: usize) -> Vec<f64> {
    let mut j = vec![C64::ZERO; p + 1];
    radial::bessel_j(k * a, &mut j);
    let mut s = 1.0;
    j.iter()
        .enumerate()
        .map(|(n, v)| {
            let u = v.abs() * ((2 * n + 1) as f64).sqrt() / s;
            s *= g;
            u
        })
        .collect()
}

/// Input weights |ĵₙ(k a/2)| √(2n + 1), n ≤ p_child, and output weights
/// |ĥₙ′(k t_min)| √(2n′ + 1), n′ ≤ p_parent, of an M2M table (unscaled).
pub fn m2m_weights(k: C64, a: f64, p_child: usize, p_parent: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = vec![C64::ZERO; p_child + 1];
    radial::bessel_j(k * (0.5 * a), &mut j);
    let mut h = vec![C64::ZERO; p_parent + 1];
    radial::hankel(k * (4.0 * a / 3.0f64.sqrt()), 1.0, &mut h);
    let input = j
        .iter()
        .enumerate()
        .map(|(n, v)| v.abs() * ((2 * n + 1) as f64).sqrt())
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

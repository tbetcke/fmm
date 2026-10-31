//! The exponential e^x for x ≤ 0 on the vector layer (Helmholtz CH2.4a;
//! docs/design/simd-p2p.md §4.9), the e^{−κr} of the Yukawa P2P.
//!
//! Written once over [`Simd`] from operations every ISA rounds as IEEE 754 does, so its
//! result is the same bit for bit on every ISA and equals the scalar layer's:
//!
//! ```text
//! x ← max(x, x_min)                          clamp to the normal range
//! k = fma(x, 1/ln 2, c)                      c = 1.5 · 2^m: k = c + n, n = round(x / ln 2)
//! n = k − c                                  exact
//! r = fnma(n, ln2_hi, x)                     exact (Cody–Waite: n · ln2_hi is exact)
//! r = fnma(n, ln2_lo, r)                     |r| ≤ ln 2 / 2
//! p = 1 + r + r²/2! + … + r^N/N!             Horner, one fma per degree
//! e^x = p · 2ⁿ = scale_pow2(p, k)            n added to p's exponent field
//! ```
//!
//! m is the number of explicit significand bits (52, 23). For |n| < 2^(m−1), k = c + n
//! holds n, as a two's complement integer, in the low bits of its significand, and the
//! low bits of c are zero; so [`Simd::scale_pow2`] adds `bits(k) << m` to `bits(p)` and
//! obtains p · 2ⁿ without a conversion to integer (AVX2 has none for 64 bits).
//!
//! **Domain.** x_min = (e_min + 1) ln 2, rounded towards zero: −1021 ln 2 ≈ −707.70 in f64
//! and −125 ln 2 ≈ −86.64 in f32. Then n ≥ e_min + 1 and p ≥ e^(−ln 2 / 2) > ½, so every
//! result is a normal number and the exponent addition never leaves the normal range. Below
//! x_min (−∞ included) the result is e^{x_min}'s, slightly above 2^(e_min + 1), at most 2^(e_min + 2), and positive. For
//! x = ±0 the result is exactly 1. Above 0 and for NaN it is unspecified.
//!
//! **Degree.** Taylor coefficients 1/i!, rounded to the precision: N = 13 in f64 and 7 in
//! f32, whose truncation errors on |r| ≤ ln 2 / 2 are 0.08 u₆₄ and 0.17 u₃₂ relative.
//!
//! **Error** (design §4.9 derives it), relative to e^x, in units of u_T, on the domain:
//! r carries 0.35 u (its rounding; n · ln2_hi is exact by ln2_hi's trailing zeros, and so is
//! the first fnma, whose result |x − n ln2_hi| ≤ |x| keeps ln2_hi's quantum),
//! plus the error of ln2_lo times |n| (below 10⁻⁶ u in f64, 0.03 u in f32); Horner's
//! roundings add u Σⱼ |qⱼ| |r|ʲ ≤ u (1 + |r|) e^{|r|} / e^{−|r|} ≤ 2.69 u, with qⱼ the
//! partial sums, and the rounded coefficients (1/i! from i = 3) 0.01 u; the scaling is
//! exact. In all at most 3.2 u (f64) and 3.3 u (f32), within the contract of 4 u_T.

use super::Simd;
use crate::SimdScalar;

/// The constants of the exponential in one precision, as f64 values that the precision
/// represents exactly (but `ln2_lo`, `inv_ln2` and the coefficients, rounded once to it).
///
/// Nominally `pub`, as [`super::AvailableIsa`], because the sealed trait of
/// [`SimdScalar`] names it; it is not reachable from outside the crate.
#[derive(Debug)]
pub struct ExpConstants {
    /// x_min = (e_min + 1) ln 2, rounded towards zero (module documentation).
    pub(crate) x_min: f64,
    /// 1/ln 2.
    pub(crate) inv_ln2: f64,
    /// c = 1.5 · 2^m, the constant that rounds k to an integer.
    pub(crate) round: f64,
    /// The leading part of ln 2, with enough trailing zeros that n · ln2_hi is exact for
    /// every n of the domain.
    pub(crate) ln2_hi: f64,
    /// ln 2 − ln2_hi.
    pub(crate) ln2_lo: f64,
    /// 1/i! for i = 0, …, N.
    pub(crate) coefficients: &'static [f64],
}

/// 1/i!, correctly rounded to f64 (i! is exact in f64 for i ≤ 18).
const fn inverse_factorial(i: u32) -> f64 {
    let mut factorial = 1.0;
    let mut j = 2;
    while j <= i {
        factorial *= j as f64;
        j += 1;
    }
    1.0 / factorial
}

/// 1/i! for i = 0, …, N.
const fn coefficients<const N: usize>() -> [f64; N] {
    let mut c = [0.0; N];
    let mut i = 0;
    while i < N {
        c[i] = inverse_factorial(i as u32);
        i += 1;
    }
    c
}

/// The Taylor coefficients of degree 13 for f64.
const COEFFICIENTS_F64: [f64; 14] = coefficients::<14>();
/// The Taylor coefficients of degree 7 for f32.
const COEFFICIENTS_F32: [f64; 8] = coefficients::<8>();

/// The constants in f64. ln2_hi and ln2_lo are fdlibm's (ln2_hi = 0x3fe62e42fee00000,
/// 32 significant bits, so n · ln2_hi is exact for |n| ≤ 2²¹).
pub(crate) const F64: ExpConstants = ExpConstants {
    x_min: -707.7032713517041,
    inv_ln2: core::f64::consts::LOG2_E,
    round: 6_755_399_441_055_744.0,
    ln2_hi: 6.931_471_803_691_238e-1,
    ln2_lo: 1.908_214_929_270_587_7e-10,
    coefficients: &COEFFICIENTS_F64,
};

/// The constants in f32. ln2_hi = 0.693359375 has 9 significant bits, so n · ln2_hi is
/// exact for |n| ≤ 2¹⁵; ln2_lo is rounded to f32 when splatted.
pub(crate) const F32: ExpConstants = ExpConstants {
    x_min: -86.643_394_470_214_84,
    inv_ln2: core::f64::consts::LOG2_E,
    round: 12_582_912.0,
    ln2_hi: 0.693_359_375,
    ln2_lo: -2.121_944_400_546_905_7e-4,
    coefficients: &COEFFICIENTS_F32,
};

/// e^x lane by lane on the layer `s` (module documentation): within 4 u_T relative for
/// x_min ≤ x ≤ 0, exactly 1 at ±0, e^{x_min}'s value below x_min; bit for bit the same on
/// every ISA.
#[inline(always)]
pub(crate) fn exp<T: SimdScalar, S: Simd<T>>(s: S, x: S::V) -> S::V {
    let c = T::EXP;
    let splat = |v: f64| s.splat(T::from_f64(v));
    let x = s.max(x, splat(c.x_min));
    let round = splat(c.round);
    let k = s.fma(x, splat(c.inv_ln2), round);
    let n = s.sub(k, round);
    let r = s.fnma(n, splat(c.ln2_hi), x);
    let r = s.fnma(n, splat(c.ln2_lo), r);
    let (&last, rest) = c
        .coefficients
        .split_last()
        .expect("at least one coefficient");
    let mut p = splat(last);
    for &ci in rest.iter().rev() {
        p = s.fma(p, r, splat(ci));
    }
    s.scale_pow2(p, k)
}

/// `out[i] = exp(x[i])` for every i, in whole vectors and the tail in a stack copy padded
/// with its last value, as [`super::rsqrt_body`].
///
/// # Panics
///
/// If `x` and `out` have different lengths.
#[inline(always)]
pub(crate) fn exp_body<T: SimdScalar, S: Simd<T>>(s: S, x: &[T], out: &mut [T]) {
    const { assert!(S::W <= super::MAX_LANES) };
    assert_eq!(
        x.len(),
        out.len(),
        "`x` and `out` must have the same length"
    );
    let whole = x.len() - x.len() % S::W;
    let (x_whole, rest) = x.split_at(whole);
    let (out_whole, out_rest) = out.split_at_mut(whole);
    for (xv, ov) in x_whole.chunks(S::W).zip(out_whole.chunks_mut(S::W)) {
        s.store(exp(s, s.load(xv)), ov);
    }
    if let Some(&last) = rest.last() {
        let mut padded = [last; super::MAX_LANES];
        padded[..rest.len()].copy_from_slice(rest);
        let mut result = [T::zero(); super::MAX_LANES];
        s.store(exp(s, s.load(&padded)), &mut result);
        out_rest.copy_from_slice(&result[..rest.len()]);
    }
}

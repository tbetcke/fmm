//! The sine and cosine of one argument on the vector layer (Helmholtz CH3.4a;
//! docs/design/simd-p2p.md §4.11), the e^{i Re(k) r} of the Helmholtz P2P.
//!
//! Written once over [`Simd`] from operations every ISA rounds as IEEE 754 does, so its
//! results are the same bit for bit on every ISA and equal the scalar layer's:
//!
//! ```text
//! k = fma(θ, 1/π, c)                 c = 1.5 · 2^m: k = c + n, n = round(θ / π)
//! n = k − c                          exact
//! t = fnma(n, π₁, θ)                 exact (n π₁ is exact; Sterbenz)
//! t = fnma(n, π₂, t)                 rounded once; |t| ≤ π/2 and a little
//! s = t t
//! S = fma(t s, P(s), t)              sin t = t + t³ (−1/3! + s/5! − …), Horner in s
//! C = Q(s)                           cos t = 1 − s/2! + s²/4! − …, Horner in s
//! h = fma(n, ½, c) − c               round(n / 2), exact
//! b = fnma(2, h, n)                  n − 2h ∈ {−1, 0, 1}, exact
//! σ = fnma(2, b b, 1)                1 − 2b² = (−1)ⁿ, exact
//! sin θ = σ S,  cos θ = σ C          exact
//! ```
//!
//! since sin(nπ + t) = (−1)ⁿ sin t and cos(nπ + t) = (−1)ⁿ cos t. m is the number of
//! explicit significand bits (52, 23); for |n| < 2^(m−1), k = c + n rounds θ / π to the
//! nearest integer, ties to even, as the exponential's k does.
//!
//! **Domain.** |θ| ≤ θ_max = 2¹⁰. Then |n| ≤ 326 has at most 9 bits, and π₁, π truncated
//! to 44 (f64) or 15 (f32) significant bits, makes n π₁ exact; θ − n π₁ is then exact by
//! Sterbenz's lemma (θ / (n π₁) lies in [½, 2] for n ≠ 0). π₂ is π − π₁ rounded to the
//! precision, so the reduced t differs from θ − nπ by its one rounding and by
//! |n| |π − π₁ − π₂| (below 10⁻²⁸ in f64 and 7 · 10⁻¹⁰ = 0.012 u₃₂ in f32). Beyond θ_max
//! and for non-finite θ the results are unspecified.
//!
//! **Symmetry.** Every step is odd or even in θ (the rounding to nearest is symmetric, and
//! its ties go to even n), so sin(−θ) = −sin θ and cos(−θ) = cos θ bit for bit for θ ≠ 0.
//! cos(±0) = 1 exactly and sin(±0) = +0: at t = −0 the correction t s P(s) is +0, and
//! −0 + +0 = +0.
//!
//! **Degree.** Taylor coefficients (−1)ⁱ/(2i + 1)! and (−1)ⁱ/(2i)!, rounded to the
//! precision: sin to t²¹ and cos to t²⁰ in f64, sin to t¹³ and cos to t¹² in f32, whose
//! truncation errors at |t| = π/2 are 10⁻¹⁸ and 0.17 u₆₄, and 0.01 and 0.11 u₃₂.
//!
//! **Error** (design §4.11 derives it; `fmm-simd/tools/sincos_bound.py`), absolute, on the
//! domain: to first order at most 2.7 u_T for the sine and 3.4 u_T for the cosine, in both
//! precisions, within the contract of 4 u_T.

use super::Simd;
use crate::SimdScalar;

/// The constants of the sine and cosine in one precision, as f64 values that the precision
/// represents exactly (but `inv_pi` and the coefficients, rounded once to it).
///
/// Nominally `pub`, as [`super::AvailableIsa`], because the sealed trait of
/// [`SimdScalar`] names it; it is not reachable from outside the crate.
#[derive(Debug)]
pub struct SincosConstants {
    /// 1/π.
    pub(crate) inv_pi: f64,
    /// c = 1.5 · 2^m, the constant that rounds k to an integer.
    pub(crate) round: f64,
    /// π truncated to few enough significant bits that n π₁ is exact for |n| ≤ 2⁹.
    pub(crate) pi_1: f64,
    /// π − π₁, rounded to the precision.
    pub(crate) pi_2: f64,
    /// (−1)ⁱ/(2i + 1)! for i = 1, …, N_s: the polynomial P of sin t = t + t s P(s).
    pub(crate) sin_coefficients: &'static [f64],
    /// (−1)ⁱ/(2i)! for i = 0, …, N_c: the polynomial Q of cos t = Q(s).
    pub(crate) cos_coefficients: &'static [f64],
}

/// (−1)ⁱ/j!, correctly rounded to f64 (j! is exact in f64 for j ≤ 22).
const fn signed_inverse_factorial(i: u32, j: u32) -> f64 {
    let mut factorial = 1.0;
    let mut l = 2;
    while l <= j {
        factorial *= l as f64;
        l += 1;
    }
    if i.is_multiple_of(2) {
        1.0 / factorial
    } else {
        -1.0 / factorial
    }
}

/// (−1)ⁱ/(2i + 1)! for i = 1, …, N.
const fn sin_coefficients<const N: usize>() -> [f64; N] {
    let mut c = [0.0; N];
    let mut i = 0;
    while i < N {
        let j = i as u32 + 1;
        c[i] = signed_inverse_factorial(j, 2 * j + 1);
        i += 1;
    }
    c
}

/// (−1)ⁱ/(2i)! for i = 0, …, N − 1.
const fn cos_coefficients<const N: usize>() -> [f64; N] {
    let mut c = [0.0; N];
    let mut i = 0;
    while i < N {
        c[i] = signed_inverse_factorial(i as u32, 2 * i as u32);
        i += 1;
    }
    c
}

/// sin to t²¹ in f64: the coefficients of t³, …, t²¹.
const SIN_F64: [f64; 10] = sin_coefficients::<10>();
/// cos to t²⁰ in f64: the coefficients of 1, t², …, t²⁰.
const COS_F64: [f64; 11] = cos_coefficients::<11>();
/// sin to t¹³ in f32: the coefficients of t³, …, t¹³.
const SIN_F32: [f64; 6] = sin_coefficients::<6>();
/// cos to t¹² in f32: the coefficients of 1, t², …, t¹².
const COS_F32: [f64; 7] = cos_coefficients::<7>();

/// The constants in f64. π₁ is π truncated to 44 significant bits.
pub(crate) const F64: SincosConstants = SincosConstants {
    inv_pi: core::f64::consts::FRAC_1_PI,
    round: 6_755_399_441_055_744.0,
    pi_1: 3.141_592_653_589_668_8,
    pi_2: 1.244_674_434_379_322_7e-13,
    sin_coefficients: &SIN_F64,
    cos_coefficients: &COS_F64,
};

/// The constants in f32. π₁ = 3.1414794921875 is π truncated to 15 significant bits; π₂
/// is π − π₁ rounded to f32; 1/π and the coefficients are rounded to f32 when splatted.
pub(crate) const F32: SincosConstants = SincosConstants {
    inv_pi: core::f64::consts::FRAC_1_PI,
    round: 12_582_912.0,
    pi_1: 3.141_479_492_187_5,
    pi_2: 1.131_614_044_425_077_7e-4,
    sin_coefficients: &SIN_F32,
    cos_coefficients: &COS_F32,
};

/// Horner in `s` over `coefficients` (constant term first), one fma per degree.
#[inline(always)]
fn horner<T: SimdScalar, S: Simd<T>>(s: S, x: S::V, coefficients: &[f64]) -> S::V {
    let splat = |v: f64| s.splat(T::from_f64(v));
    let (&last, rest) = coefficients.split_last().expect("at least one coefficient");
    let mut p = splat(last);
    for &ci in rest.iter().rev() {
        p = s.fma(p, x, splat(ci));
    }
    p
}

/// (sin θ, cos θ) lane by lane on the layer `s` (module documentation): each within 4 u_T
/// absolute for |θ| ≤ 2¹⁰, odd and even bit for bit; bit for bit the same on every ISA.
#[inline(always)]
pub(crate) fn sincos<T: SimdScalar, S: Simd<T>>(s: S, theta: S::V) -> (S::V, S::V) {
    let c = T::SINCOS;
    let splat = |v: f64| s.splat(T::from_f64(v));
    let round = splat(c.round);
    let k = s.fma(theta, splat(c.inv_pi), round);
    let n = s.sub(k, round);
    let t = s.fnma(n, splat(c.pi_1), theta);
    let t = s.fnma(n, splat(c.pi_2), t);
    let t2 = s.mul(t, t);
    let sin_t = s.fma(s.mul(t, t2), horner(s, t2, c.sin_coefficients), t);
    let cos_t = horner(s, t2, c.cos_coefficients);
    let one = splat(1.0);
    let two = splat(2.0);
    let h = s.sub(s.fma(n, splat(0.5), round), round);
    let b = s.fnma(two, h, n);
    let sign = s.fnma(two, s.mul(b, b), one);
    (s.mul(sign, sin_t), s.mul(sign, cos_t))
}

/// `sin[i] = sin(theta[i])` and `cos[i] = cos(theta[i])` for every i, in whole vectors and
/// the tail in a stack copy padded with its last value, as [`super::exp::exp_body`].
///
/// # Panics
///
/// If `theta`, `sin` and `cos` do not all have the same length.
#[inline(always)]
pub(crate) fn sincos_body<T: SimdScalar, S: Simd<T>>(
    s: S,
    theta: &[T],
    sin: &mut [T],
    cos: &mut [T],
) {
    const { assert!(S::W <= super::MAX_LANES) };
    assert!(
        theta.len() == sin.len() && theta.len() == cos.len(),
        "`theta`, `sin` and `cos` must have the same length"
    );
    let whole = theta.len() - theta.len() % S::W;
    let (theta_whole, rest) = theta.split_at(whole);
    let (sin_whole, sin_rest) = sin.split_at_mut(whole);
    let (cos_whole, cos_rest) = cos.split_at_mut(whole);
    for ((tv, sv), cv) in theta_whole
        .chunks(S::W)
        .zip(sin_whole.chunks_mut(S::W))
        .zip(cos_whole.chunks_mut(S::W))
    {
        let (sin_v, cos_v) = sincos(s, s.load(tv));
        s.store(sin_v, sv);
        s.store(cos_v, cv);
    }
    if let Some(&last) = rest.last() {
        let mut padded = [last; super::MAX_LANES];
        padded[..rest.len()].copy_from_slice(rest);
        let (sin_v, cos_v) = sincos(s, s.load(&padded));
        let mut result = [T::zero(); super::MAX_LANES];
        s.store(sin_v, &mut result);
        sin_rest.copy_from_slice(&result[..rest.len()]);
        s.store(cos_v, &mut result);
        cos_rest.copy_from_slice(&result[..rest.len()]);
    }
}

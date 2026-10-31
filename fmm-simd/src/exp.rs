//! The exponential of the Helmholtz-family P2P kernels, for tests and reports (Helmholtz
//! CH2.4a; docs/design/simd-p2p.md §4.9): e^x for x ≤ 0, the e^{−κr} of the Yukawa
//! kernel.
//!
//! | [`Isa`] | f32 | f64 | bound, relative |
//! | --- | --- | --- | --- |
//! | every ISA | Cody–Waite reduction, Taylor degree 7 | Cody–Waite reduction, Taylor degree 13 | 3.3 u_T, derived |
//!
//! One formulation for every ISA, written once over the vector layer from operations that
//! each ISA rounds as IEEE 754 does (multiplications and additions fused or not, a
//! maximum, integer additions on the bits): n = round(x / ln 2) by adding 1.5 · 2^m, the
//! reduced argument r = x − n ln 2 in two exact-or-once-rounded steps, |r| ≤ ln 2 / 2, the
//! Taylor polynomial of e^r by Horner, and the result's exponent raised by n through an
//! integer addition on its bits. So, unlike the inverse square root, the results are the
//! same bit for bit on every ISA.
//!
//! u_T is the unit roundoff, 2⁻²⁴ (f32) or 2⁻⁵³ (f64). The contract of design §4.9,
//! checked by the tests of `tests/exp` on every ISA a machine offers, is a relative error
//! of at most 4 u_T on the domain [x_min, 0], with x_min = −125 ln 2 ≈ −86.64 (f32) and
//! −1021 ln 2 ≈ −707.70 (f64), where every result is a normal number; exactly 1 at x = ±0;
//! and below x_min the value at x_min, which is positive and at most 2⁻¹²⁴ (f32) or 2⁻¹⁰²⁰
//! (f64).

use crate::{Isa, SimdScalar};

/// `out[i] = e^x[i]` for every i, by the exponential that the Helmholtz-family P2P
/// kernels use on `isa`.
///
/// The result is within 4 u_T relative of e^x for x in the domain [x_min, 0], exactly 1 at
/// x = ±0, and below x_min (−∞ included) the result at x_min (module documentation).
/// Above 0 and for NaN it is unspecified. The result is the same bit for bit on every ISA,
/// and each value's result depends only on the value, not on its position in `x` or the
/// length of `x`: whole vectors are computed in place, and the tail in a padded copy by
/// the same vector code (design §4.5).
///
/// ```
/// use nd_fmm_simd::{Isa, exp::exp_slice};
///
/// let x = [0.0_f64, -1.0, -1000.0];
/// let mut out = [f64::NAN; 3];
/// exp_slice(Isa::detect(), &x, &mut out);
/// assert_eq!(out[0], 1.0);
/// let e = (-1.0_f64).exp();
/// assert!((out[1] - e).abs() <= 4.0 * f64::EPSILON / 2.0 * e);
/// // Below the domain: the value at x_min, tiny and positive.
/// assert!(out[2] > 0.0 && out[2] <= 2f64.powi(-1020));
/// ```
///
/// # Panics
///
/// If `isa` is not available on this machine ([`Isa::is_available`]), or if `x` and
/// `out` have different lengths.
pub fn exp_slice<T: SimdScalar>(isa: Isa, x: &[T], out: &mut [T]) {
    assert_eq!(
        x.len(),
        out.len(),
        "`x` and `out` must have the same length, got {} and {}",
        x.len(),
        out.len()
    );
    T::exp_slice(isa, x, out);
}

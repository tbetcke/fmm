//! The sine and cosine of the Helmholtz P2P kernel, for tests and reports (Helmholtz
//! CH3.4a; docs/design/simd-p2p.md §4.11): sin θ and cos θ of one argument together, the
//! e^{i Re(k) r} of the Helmholtz kernel e^{ikr}/r.
//!
//! | [`Isa`] | f32 | f64 | bound, absolute |
//! | --- | --- | --- | --- |
//! | every ISA | reduction by π (Cody–Waite, two parts), Taylor to t¹³ and t¹² | reduction by π (Cody–Waite, two parts), Taylor to t²¹ and t²⁰ | 2.7 u_T (sin), 3.4 u_T (cos), derived |
//!
//! One formulation for every ISA, written once over the vector layer from operations that
//! each ISA rounds as IEEE 754 does (multiplications and additions, fused or not): n =
//! round(θ / π) by adding 1.5 · 2^m, the reduced argument t = θ − nπ in an exact and a
//! once-rounded step, |t| ≤ π/2, the Taylor polynomials of sin t and cos t in t² by
//! Horner, and the sign (−1)ⁿ formed from n by exact arithmetic. So, unlike the inverse
//! square root, the results are the same bit for bit on every ISA.
//!
//! u_T is the unit roundoff, 2⁻²⁴ (f32) or 2⁻⁵³ (f64). The contract of design §4.11,
//! checked by the tests of `tests/sincos` on every ISA a machine offers, is an absolute
//! error of at most 4 u_T in each of sin θ and cos θ for |θ| ≤ 2¹⁰ (the phase Re(k) r of the
//! Helmholtz P2P stays below about 22 c_leaf radians, proposal §5.7); sin(−θ) = −sin θ and
//! cos(−θ) = cos θ bit for bit for θ ≠ 0; sin(±0) = +0 and cos(±0) = 1 exactly. The error is absolute
//! because the P2P term q e^{iθ}/r is measured relative to |q|/r: near a zero of sin or
//! cos the relative error of that one value is not bounded.

use crate::{Isa, SimdScalar};

/// `sin[i] = sin θᵢ` and `cos[i] = cos θᵢ` for every θᵢ of `theta`, by the sine and cosine
/// that the Helmholtz P2P kernel uses on `isa`.
///
/// Each result is within 4 u_T absolute of the exact value for |θ| ≤ 2¹⁰ (module
/// documentation); beyond 2¹⁰ and for non-finite θ the results are unspecified. The
/// results are the same bit for bit on every ISA, and each value's results depend only on
/// the value, not on its position in `theta` or the length of `theta`: whole vectors are
/// computed in place, and the tail in a padded copy by the same vector code (design §4.5).
///
/// ```
/// use nd_fmm_simd::{Isa, sincos::sincos_slice};
///
/// let theta = [0.0_f64, 1.0, -20.0];
/// let mut sin = [f64::NAN; 3];
/// let mut cos = [f64::NAN; 3];
/// sincos_slice(Isa::detect(), &theta, &mut sin, &mut cos);
/// assert_eq!((sin[0], cos[0]), (0.0, 1.0));
/// let u = f64::EPSILON / 2.0;
/// assert!((sin[1] - 1.0_f64.sin()).abs() <= 4.0 * u);
/// assert!((cos[2] - 20.0_f64.cos()).abs() <= 4.0 * u);
/// ```
///
/// # Panics
///
/// If `isa` is not available on this machine ([`Isa::is_available`]), or if `theta`,
/// `sin` and `cos` do not all have the same length.
pub fn sincos_slice<T: SimdScalar>(isa: Isa, theta: &[T], sin: &mut [T], cos: &mut [T]) {
    assert!(
        theta.len() == sin.len() && theta.len() == cos.len(),
        "`theta`, `sin` and `cos` must have the same length, got {}, {} and {}",
        theta.len(),
        sin.len(),
        cos.len()
    );
    T::sincos_slice(isa, theta, sin, cos);
}

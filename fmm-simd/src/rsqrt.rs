//! The inverse square root of the P2P kernels, for tests and reports
//! (docs/design/simd-p2p.md §4.3 and §5.1).
//!
//! | [`Isa`] | f32 | f64 | bound, relative |
//! | --- | --- | --- | --- |
//! | [`Isa::Scalar`] | `sqrt`, division | `sqrt`, division | 1.5 u_T |
//! | [`Isa::Neon`] | FSQRT, FDIV | FSQRT, FDIV | 1.5 u_T |
//! | [`Isa::Avx2`] | `vrsqrtps`, degree-2 correction | `vrsqrtps` through f32, degree-5 correction | 1.5 u_T, derived |
//!
//! u_T is the unit roundoff, 2⁻²⁴ (f32) or 2⁻⁵³ (f64). The formulations are the ones
//! signed off in Phase 3S T2 (spikes/p2p-simd/SPIKE_REPORT.md, "Recommendation"); the
//! architecture modules document each with its error bound. The contract of
//! design §4.3, checked by the tests of `tests/rsqrt` on every ISA a machine offers, is
//! a relative error of at most 4 u_T on the kernel domain of CONVENTIONS §3.13,
//! 2⁻¹⁰⁸ ≤ x ≤ 2⁷, and exactly +0 at x = 0.

use crate::{Isa, SimdScalar};

/// `out[i] = 1/√x[i]` for every i, by the inverse square root that the P2P kernel uses
/// on `isa`, with the coincident-pair mask: `out[i]` is +0 where `x[i]` is 0
/// (CONVENTIONS §3.13, "Fast kernels"; design §4.4).
///
/// The result is within 4 u_T relative of 1/√x for x in the kernel domain
/// 2⁻¹⁰⁸ ≤ x ≤ 2⁷ (module documentation). Outside it (negative, subnormal, very
/// large, infinite or NaN inputs) the result is unspecified. Each value's result
/// depends only on the value and `isa`, not on its position in `x` or the length of
/// `x`: whole vectors are computed in place, and the tail in a padded copy by the same
/// vector code (design §4.5).
///
/// ```
/// use nd_fmm_simd::{Isa, rsqrt::rsqrt_slice};
///
/// let x = [4.0_f64, 0.0, 0.25];
/// let mut out = [f64::NAN; 3];
/// rsqrt_slice(Isa::detect(), &x, &mut out);
/// assert!((out[0] - 0.5).abs() <= 4.0 * f64::EPSILON / 2.0 * 0.5);
/// assert_eq!(out[1].to_bits(), 0.0_f64.to_bits());
/// assert!((out[2] - 2.0).abs() <= 4.0 * f64::EPSILON / 2.0 * 2.0);
/// ```
///
/// # Panics
///
/// If `isa` is not available on this machine ([`Isa::is_available`]), or if `x` and
/// `out` have different lengths.
pub fn rsqrt_slice<T: SimdScalar>(isa: Isa, x: &[T], out: &mut [T]) {
    assert_eq!(
        x.len(),
        out.len(),
        "`x` and `out` must have the same length, got {} and {}",
        x.len(),
        out.len()
    );
    T::rsqrt_slice(isa, x, out);
}

//! The real scalar type that all numeric code is generic over.

use num_traits::{Float, FloatConst};

/// Real floating-point scalar of the `nd-fmm-*` crates, implemented for [`f32`] and
/// [`f64`].
///
/// The ranges in which each type is tested are fixed in CONVENTIONS §3.9: f64 up to
/// degree p = 30, f32 up to p = 8.
///
/// ```
/// use nd_fmm_math::RealScalar;
///
/// fn half<T: RealScalar>(x: T) -> T {
///     x * T::from_f64(0.5)
/// }
///
/// assert_eq!(half(3.0_f32).to_f64(), 1.5);
/// ```
pub trait RealScalar: Float + FloatConst + Copy + Send + Sync + 'static {
    /// Converts from `f64`, rounding to the nearest representable value.
    fn from_f64(value: f64) -> Self;

    /// Converts to `f64`; exact for both implementations.
    ///
    /// Call it on a value, not a reference: for `v: &T`, `v.to_f64()` resolves to
    /// [`num_traits::ToPrimitive::to_f64`], which returns `Option<f64>`. Write
    /// `(*v).to_f64()` or iterate with `|&v| v.to_f64()` instead.
    fn to_f64(self) -> f64;
}

impl RealScalar for f64 {
    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

impl RealScalar for f32 {
    #[inline]
    fn from_f64(value: f64) -> Self {
        // `as` rounds to nearest, ties to even.
        value as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

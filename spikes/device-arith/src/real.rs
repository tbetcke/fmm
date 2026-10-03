//! The precisions of the spike, and the host references of the primitive operations.
//!
//! Errors are relative and in units of u_T, the unit roundoff (2⁻²⁴ for f32, 2⁻⁵³ for
//! f64): a correctly rounded result is within 1 u_T (half an ulp, relative to a value at
//! the bottom of its binade). For f32 the exact value is approximated in f64 (error
//! 2⁻⁵³, far below an f32 ulp). For f64 the error of a device result y is computed from
//! residuals that are exact with an fma: x y² − 1 for 1/√x, y² − x for √x, q b − a for
//! a / b.

use cubecl::prelude::{CubeElement, Float};
use nd_fmm_simd::SimdScalar;

/// A precision run on the device and on the host.
pub trait Real: Float + CubeElement + SimdScalar + Copy + Send + Sync + std::fmt::Debug {
    /// "f32" or "f64".
    const NAME: &'static str;
    /// The unit roundoff u_T.
    const U: f64;
    /// Significand bits with the hidden bit (24, 53).
    const BITS: i32;
    /// The smallest positive normal value.
    const MIN_NORMAL: Self;
    /// A subnormal value (the smallest normal times 2⁻³).
    const SUBNORMAL: Self;
    /// Rounds an f64 to this type.
    fn narrow(x: f64) -> Self;
    /// Widens to f64 (exact).
    fn widen(self) -> f64;
    /// The raw bits, zero-extended.
    fn bits(self) -> u64;
    /// The value from raw bits.
    fn from_bits(bits: u64) -> Self;
    /// The relative error of `y` as 1/√x, in u_T.
    fn rsqrt_error(x: Self, y: Self) -> f64;
    /// The relative error of `y` as √x, in u_T.
    fn sqrt_error(x: Self, y: Self) -> f64;
    /// The relative error of `y` as a / b, in u_T.
    fn div_error(a: Self, b: Self, y: Self) -> f64;
    /// a · b + c on the host, rounded once.
    fn host_fma(a: Self, b: Self, c: Self) -> Self;
    /// √x on the host, correctly rounded.
    fn host_sqrt(x: Self) -> Self;
}

/// Relative error in u of `y` against `exact`, both in f64; ∞ if `y` is not finite or
/// differs in sign.
fn rel(y: f64, exact: f64, u: f64) -> f64 {
    if !y.is_finite() {
        return f64::INFINITY;
    }
    ((y - exact) / exact).abs() / u
}

impl Real for f32 {
    const NAME: &'static str = "f32";
    const U: f64 = 1.0 / (1u64 << 24) as f64;
    const BITS: i32 = 24;
    const MIN_NORMAL: Self = f32::MIN_POSITIVE;
    const SUBNORMAL: Self = f32::MIN_POSITIVE / 8.0;
    fn narrow(x: f64) -> Self {
        x as f32
    }
    fn widen(self) -> f64 {
        f64::from(self)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_bits(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
    fn rsqrt_error(x: Self, y: Self) -> f64 {
        rel(f64::from(y), 1.0 / f64::from(x).sqrt(), Self::U)
    }
    fn sqrt_error(x: Self, y: Self) -> f64 {
        rel(f64::from(y), f64::from(x).sqrt(), Self::U)
    }
    fn div_error(a: Self, b: Self, y: Self) -> f64 {
        rel(f64::from(y), f64::from(a) / f64::from(b), Self::U)
    }
    fn host_fma(a: Self, b: Self, c: Self) -> Self {
        a.mul_add(b, c)
    }
    fn host_sqrt(x: Self) -> Self {
        x.sqrt()
    }
}

/// The exact x y² as hi + lo + lo2 is not needed: x y² − 1 is computed from y² = h + l
/// (exact, two-product) as fl(fl(x h − 1) + (fma residual of x h) + x l), accurate to a
/// few u² relative when y is within a few ulps of 1/√x.
fn rsqrt_residual(x: f64, y: f64) -> f64 {
    let h = y * y;
    let l = y.mul_add(y, -h);
    let p = x * h;
    let pe = x.mul_add(h, -p);
    (p - 1.0) + pe + x * l
}

impl Real for f64 {
    const NAME: &'static str = "f64";
    const U: f64 = 1.0 / (1u64 << 53) as f64;
    const BITS: i32 = 53;
    const MIN_NORMAL: Self = f64::MIN_POSITIVE;
    const SUBNORMAL: Self = f64::MIN_POSITIVE / 8.0;
    fn narrow(x: f64) -> Self {
        x
    }
    fn widen(self) -> f64 {
        self
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }
    fn rsqrt_error(x: Self, y: Self) -> f64 {
        if !y.is_finite() || y <= 0.0 {
            return f64::INFINITY;
        }
        // y = (1 + e)/√x: x y² − 1 = 2e + e², so e = √(1 + t) − 1 ≈ t/2 − t²/8.
        let t = rsqrt_residual(x, y);
        (t / 2.0 - t * t / 8.0).abs() / Self::U
    }
    fn sqrt_error(x: Self, y: Self) -> f64 {
        if !y.is_finite() || y <= 0.0 {
            return f64::INFINITY;
        }
        // y = (1 + e)√x: (y² − x)/x = 2e + e²; y² − x from the exact two-product.
        let h = y * y;
        let l = y.mul_add(y, -h);
        let t = ((h - x) + l) / x;
        (t / 2.0 - t * t / 8.0).abs() / Self::U
    }
    fn div_error(a: Self, b: Self, y: Self) -> f64 {
        if !y.is_finite() {
            return f64::INFINITY;
        }
        // y = (1 + e) a / b: e = (y b − a) / a, the residual exact with an fma.
        (y.mul_add(b, -a) / a).abs() / Self::U
    }
    fn host_fma(a: Self, b: Self, c: Self) -> Self {
        a.mul_add(b, c)
    }
    fn host_sqrt(x: Self) -> Self {
        x.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The f64 error measures give at most 1 u for correctly rounded results and see a
    /// one-ulp change. Error measure: the measures themselves, against 1 u and 2 u.
    #[test]
    fn f64_measures() {
        let mut x = 1.0f64;
        for _ in 0..1000 {
            x = x * 1.75 % 7.0 + 0.013;
            let s = x.sqrt();
            assert!(f64::sqrt_error(x, s) <= 1.0, "sqrt {x}");
            assert!(f64::rsqrt_error(x, 1.0 / s) <= 3.0, "rsqrt {x}");
            assert!(f64::div_error(1.0, x, 1.0 / x) <= 1.0, "div {x}");
            let off = f64::from_bits(s.to_bits() + 3);
            assert!(f64::sqrt_error(x, off) >= 2.0, "sqrt off {x}");
        }
    }
}

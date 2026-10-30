//! A minimal complex number over a [`RealScalar`], for the arithmetic of the Helmholtz
//! family (CONVENTIONS §3.15).
//!
//! Operator storage keeps complex quantities as real planes (§3.15, "Storage"); this type
//! is only for computing with them. It has what the radial functions and the coaxial
//! coefficients need, and nothing that needs a dependency.

use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use crate::RealScalar;

/// The complex number `re + i im`.
///
/// ```
/// use nd_fmm_math::Complex;
///
/// let z = Complex::new(1.0_f64, 2.0);
/// let w = z * z.conj();
/// assert_eq!(w, Complex::real(5.0));
/// assert_eq!(Complex::<f64>::i_pow(3), Complex::new(0.0, -1.0));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex<T> {
    /// The real part.
    pub re: T,
    /// The imaginary part.
    pub im: T,
}

impl<T: RealScalar> Complex<T> {
    /// `re + i im`.
    #[inline(always)]
    pub const fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }

    /// The real number `re`.
    #[inline(always)]
    pub fn real(re: T) -> Self {
        Complex::new(re, T::zero())
    }

    /// 0.
    #[inline(always)]
    pub fn zero() -> Self {
        Complex::new(T::zero(), T::zero())
    }

    /// 1.
    #[inline(always)]
    pub fn one() -> Self {
        Complex::new(T::one(), T::zero())
    }

    /// The imaginary unit i.
    #[inline(always)]
    pub fn i() -> Self {
        Complex::new(T::zero(), T::one())
    }

    /// iᵏ, exactly.
    pub fn i_pow(k: i64) -> Self {
        let (o, z) = (T::one(), T::zero());
        match k.rem_euclid(4) {
            0 => Complex::new(o, z),
            1 => Complex::new(z, o),
            2 => Complex::new(-o, z),
            _ => Complex::new(z, -o),
        }
    }

    /// e^{iθ} = cos θ + i sin θ.
    #[inline]
    pub fn cis(theta: T) -> Self {
        let (s, c) = theta.sin_cos();
        Complex::new(c, s)
    }

    /// The complex conjugate.
    #[inline(always)]
    pub fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }

    /// The modulus |z|, without overflow of the squares.
    #[inline]
    pub fn abs(self) -> T {
        self.re.hypot(self.im)
    }

    /// Whether both parts are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Both parts as f64 (exact).
    #[inline]
    pub fn to_f64(self) -> Complex<f64> {
        Complex::new(self.re.to_f64(), self.im.to_f64())
    }

    /// Both parts rounded from f64.
    #[inline]
    pub fn from_f64(z: Complex<f64>) -> Self {
        Complex::new(T::from_f64(z.re), T::from_f64(z.im))
    }
}

impl<T: RealScalar> Add for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl<T: RealScalar> AddAssign for Complex<T> {
    #[inline(always)]
    fn add_assign(&mut self, o: Self) {
        self.re = self.re + o.re;
        self.im = self.im + o.im;
    }
}

impl<T: RealScalar> Sub for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl<T: RealScalar> SubAssign for Complex<T> {
    #[inline(always)]
    fn sub_assign(&mut self, o: Self) {
        self.re = self.re - o.re;
        self.im = self.im - o.im;
    }
}

impl<T: RealScalar> Neg for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: RealScalar> Mul for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl<T: RealScalar> Mul<T> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: T) -> Self {
        Complex::new(self.re * s, self.im * s)
    }
}

impl<T: RealScalar> Div for Complex<T> {
    type Output = Self;
    /// Smith's algorithm, which forms no square of either modulus.
    #[inline]
    fn div(self, o: Self) -> Self {
        if o.re.abs() >= o.im.abs() {
            let r = o.im / o.re;
            let d = o.re + o.im * r;
            Complex::new((self.re + self.im * r) / d, (self.im - self.re * r) / d)
        } else {
            let r = o.re / o.im;
            let d = o.re * r + o.im;
            Complex::new((self.re * r + self.im) / d, (self.im * r - self.re) / d)
        }
    }
}

impl<T: RealScalar> Div<T> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: T) -> Self {
        Complex::new(self.re / s, self.im / s)
    }
}

#[cfg(test)]
mod tests {
    use super::Complex;

    #[test]
    fn division_at_large_moduli() {
        // The squares of the moduli overflow f64; Smith's algorithm does not form them.
        let a = Complex::<f64>::new(3e200, 4e200);
        let b = Complex::new(1e200, -2e200);
        let q = a / b;
        // (3 + 4i)/(1 − 2i) = (3 + 4i)(1 + 2i)/5 = (−5 + 10i)/5.
        assert!((q.re + 1.0).abs() <= 1e-15 && (q.im - 2.0).abs() <= 1e-15);
        let q = Complex::new(1.0, 0.0) / Complex::new(1e-200, 3e-200);
        assert!(q.is_finite());
        assert!((q * Complex::new(1e-200, 3e-200) - Complex::one()).abs() <= 1e-15);
    }

    #[test]
    fn powers_of_i() {
        let mut p = Complex::<f64>::one();
        for k in 0..12 {
            assert_eq!(Complex::i_pow(k), p);
            assert_eq!(Complex::i_pow(-k), p.conj());
            p = p * Complex::i();
        }
    }

    #[test]
    fn modulus_without_overflow() {
        assert_eq!(Complex::new(3e300, 4e300).abs(), 5e300);
        assert_eq!(Complex::new(3.0f32, -4.0).abs(), 5.0);
    }
}

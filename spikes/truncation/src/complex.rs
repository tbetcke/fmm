//! A minimal complex number over a [`Real`].

use crate::real::Real;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// re + i im.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T: Real> Complex<T> {
    /// 0.
    pub const ZERO: Complex<T> = Complex {
        re: T::ZERO,
        im: T::ZERO,
    };
    /// 1.
    pub const ONE: Complex<T> = Complex {
        re: T::ONE,
        im: T::ZERO,
    };
    /// i.
    pub const I: Complex<T> = Complex {
        re: T::ZERO,
        im: T::ONE,
    };

    /// re + i im.
    pub const fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }

    /// A real number.
    pub fn real(re: T) -> Self {
        Complex { re, im: T::ZERO }
    }

    /// From two f64.
    pub fn from_f64(re: f64, im: f64) -> Self {
        Complex::new(T::from_f64(re), T::from_f64(im))
    }

    /// Both parts rounded to f64.
    pub fn to_f64(self) -> Complex<f64> {
        Complex::new(self.re.to_f64(), self.im.to_f64())
    }

    /// The complex conjugate.
    pub fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }

    /// |z|².
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// |z| (no guard against overflow of the squares; the spike's values stay far
    /// inside the range).
    pub fn abs(self) -> T {
        self.norm_sqr().sqrt()
    }

    /// max(|re|, |im|), a size that cannot overflow.
    pub fn max_abs(self) -> T {
        self.re.abs().max(self.im.abs())
    }

    /// e^{iθ}.
    pub fn cis(theta: T) -> Self {
        let (s, c) = theta.sin_cos();
        Complex::new(c, s)
    }

    /// e^z.
    pub fn exp(self) -> Self {
        Complex::cis(self.im) * self.re.exp()
    }

    /// z times a real.
    pub fn scale(self, f: T) -> Self {
        Complex::new(self.re * f, self.im * f)
    }
}

impl<T: Real> Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl<T: Real> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        self.re += o.re;
        self.im += o.im;
    }
}

impl<T: Real> Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl<T: Real> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        self.re -= o.re;
        self.im -= o.im;
    }
}

impl<T: Real> Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: Real> Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl<T: Real> Mul<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, f: T) -> Self {
        Complex::new(self.re * f, self.im * f)
    }
}

impl<T: Real> Div for Complex<T> {
    type Output = Self;
    /// Smith's algorithm, which forms no square of the divisor and so neither overflows
    /// nor underflows where the quotient is in range.
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

impl<T: Real> Div<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, f: T) -> Self {
        Complex::new(self.re / f, self.im / f)
    }
}

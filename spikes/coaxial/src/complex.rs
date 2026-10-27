//! A minimal complex f64, enough for the recursions; the workspace has no complex type
//! the spike could use without a new dependency.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// A complex number re + i im.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl C64 {
    /// Zero.
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    /// The imaginary unit.
    pub const I: C64 = C64 { re: 0.0, im: 1.0 };

    /// re + i im.
    pub const fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }

    /// The modulus, without overflow of the squares.
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// e^{iθ}.
    pub fn cis(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        C64::new(c, s)
    }

    /// i^k.
    pub fn i_pow(k: i64) -> Self {
        match k.rem_euclid(4) {
            0 => C64::new(1.0, 0.0),
            1 => C64::new(0.0, 1.0),
            2 => C64::new(-1.0, 0.0),
            _ => C64::new(0.0, -1.0),
        }
    }

    /// Whether both parts are finite.
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl Add for C64 {
    type Output = C64;
    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
}

impl AddAssign for C64 {
    fn add_assign(&mut self, o: C64) {
        self.re += o.re;
        self.im += o.im;
    }
}

impl Sub for C64 {
    type Output = C64;
    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
}

impl Neg for C64 {
    type Output = C64;
    fn neg(self) -> C64 {
        C64::new(-self.re, -self.im)
    }
}

impl Mul for C64 {
    type Output = C64;
    fn mul(self, o: C64) -> C64 {
        C64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl Mul<f64> for C64 {
    type Output = C64;
    fn mul(self, s: f64) -> C64 {
        C64::new(self.re * s, self.im * s)
    }
}

impl Mul<C64> for f64 {
    type Output = C64;
    fn mul(self, z: C64) -> C64 {
        z * self
    }
}

impl Div for C64 {
    type Output = C64;
    /// Smith's algorithm, which forms no square of a large modulus.
    fn div(self, o: C64) -> C64 {
        if o.re.abs() >= o.im.abs() {
            let r = o.im / o.re;
            let d = o.re + o.im * r;
            C64::new((self.re + self.im * r) / d, (self.im - self.re * r) / d)
        } else {
            let r = o.re / o.im;
            let d = o.re * r + o.im;
            C64::new((self.re * r + self.im) / d, (self.im * r - self.re) / d)
        }
    }
}

impl Div<f64> for C64 {
    type Output = C64;
    fn div(self, s: f64) -> C64 {
        C64::new(self.re / s, self.im / s)
    }
}

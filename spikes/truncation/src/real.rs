//! The real scalar of the spike: `f64`, or [`Dd`], a double-double number of about 32
//! significant digits, so that every operator of a chain runs in either precision from
//! one source.
//!
//! A [`Dd`] is an unevaluated sum hi + lo with |lo| ≤ ulp(hi)/2. The arithmetic follows
//! the error-free transformations of Dekker and Knuth as in the QD library of Hida, Li
//! and Bailey (2001): `two_sum` and `two_prod` (the latter with a fused multiply-add),
//! sums and products correct to about 2⁻¹⁰⁴ relative. The exponent range is that of f64,
//! and below about 10⁻²⁹² the low part is subnormal and the precision falls off.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// The arithmetic the chains need, for `f64` and [`Dd`].
pub trait Real:
    Copy
    + Debug
    + PartialEq
    + PartialOrd
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// 0.
    const ZERO: Self;
    /// 1.
    const ONE: Self;
    /// Unit roundoff of the type, 2⁻⁵³ or 2⁻¹⁰⁴ (approximately, for [`Dd`]).
    const EPSILON: f64;
    /// A short name for reports: "f64" or "dd".
    const NAME: &'static str;

    /// The exact value of an f64.
    fn from_f64(x: f64) -> Self;
    /// The nearest f64.
    fn to_f64(self) -> f64;
    /// The square root, x ≥ 0.
    fn sqrt(self) -> Self;
    /// e^x.
    fn exp(self) -> Self;
    /// (sin x, cos x).
    fn sin_cos(self) -> (Self, Self);
    /// π.
    fn pi() -> Self;

    /// |x|.
    fn abs(self) -> Self {
        if self < Self::ZERO { -self } else { self }
    }

    /// The exact value of an integer below 2⁵³.
    fn from_usize(n: usize) -> Self {
        Self::from_f64(n as f64)
    }

    /// x times an f64.
    fn scale(self, f: f64) -> Self {
        self * Self::from_f64(f)
    }

    /// max(x, y).
    fn max(self, other: Self) -> Self {
        if other > self { other } else { self }
    }
}

impl Real for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const EPSILON: f64 = f64::EPSILON / 2.0;
    const NAME: &'static str = "f64";

    fn from_f64(x: f64) -> f64 {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn sin_cos(self) -> (f64, f64) {
        f64::sin_cos(self)
    }
    fn pi() -> f64 {
        std::f64::consts::PI
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
}

/// A double-double number hi + lo.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Dd {
    /// The leading part, the f64 nearest to the value.
    pub hi: f64,
    /// The trailing part, |lo| ≤ ulp(hi)/2.
    pub lo: f64,
}

/// s + e = a + b exactly.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// s + e = a + b exactly, for |a| ≥ |b|.
#[inline(always)]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// p + e = a b exactly (unless it underflows).
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl Dd {
    /// π, rounded to 106 bits.
    pub const PI: Dd = Dd {
        hi: std::f64::consts::PI,
        lo: 1.224_646_799_147_353_2e-16,
    };
    /// ln 2, rounded to 106 bits.
    pub const LN2: Dd = Dd {
        hi: std::f64::consts::LN_2,
        lo: 2.319_046_813_846_299_6e-17,
    };
    /// e, rounded to 106 bits (a test value only).
    pub const E: Dd = Dd {
        hi: std::f64::consts::E,
        lo: 1.445_646_891_729_250_2e-16,
    };

    /// hi + lo, normalised.
    pub fn new(hi: f64, lo: f64) -> Dd {
        let (hi, lo) = two_sum(hi, lo);
        Dd { hi, lo }
    }

    /// The quotient by an f64.
    #[inline]
    pub fn div_f64(self, b: f64) -> Dd {
        let q1 = self.hi / b;
        let (p1, p2) = two_prod(q1, b);
        let (s, e) = two_sum(self.hi, -p1);
        let e = e - p2 + self.lo;
        let q2 = (s + e) / b;
        let (hi, lo) = quick_two_sum(q1, q2);
        Dd { hi, lo }
    }

    /// The nearest integer, as f64 (|x| < 2⁵²).
    fn round(self) -> f64 {
        let r = self.hi.round();
        if r == self.hi {
            // hi is an integer; lo may move the value across a half.
            let (hi, lo) = two_sum(r, self.lo.round());
            hi + lo
        } else if (r - self.hi).abs() == 0.5 && self.lo != 0.0 {
            // A tie of hi broken by lo.
            if (self.lo > 0.0) == (r > self.hi) {
                r
            } else {
                r - (r - self.hi).signum()
            }
        } else {
            r
        }
    }
}

impl Add for Dd {
    type Output = Dd;
    #[inline]
    fn add(self, b: Dd) -> Dd {
        let (s1, s2) = two_sum(self.hi, b.hi);
        let (t1, t2) = two_sum(self.lo, b.lo);
        let (s1, s2) = quick_two_sum(s1, s2 + t1);
        let (hi, lo) = quick_two_sum(s1, s2 + t2);
        Dd { hi, lo }
    }
}

impl Sub for Dd {
    type Output = Dd;
    #[inline]
    fn sub(self, b: Dd) -> Dd {
        self + (-b)
    }
}

impl Neg for Dd {
    type Output = Dd;
    #[inline]
    fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Mul for Dd {
    type Output = Dd;
    #[inline]
    fn mul(self, b: Dd) -> Dd {
        let (p1, p2) = two_prod(self.hi, b.hi);
        let p2 = p2 + (self.hi * b.lo + self.lo * b.hi);
        let (hi, lo) = quick_two_sum(p1, p2);
        Dd { hi, lo }
    }
}

impl Div for Dd {
    type Output = Dd;
    #[inline]
    fn div(self, b: Dd) -> Dd {
        let q1 = self.hi / b.hi;
        let r = self - b * Dd::from_f64(q1);
        let q2 = r.hi / b.hi;
        let r = r - b * Dd::from_f64(q2);
        let q3 = r.hi / b.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        Dd { hi, lo } + Dd::from_f64(q3)
    }
}

impl AddAssign for Dd {
    #[inline]
    fn add_assign(&mut self, b: Dd) {
        *self = *self + b;
    }
}

impl SubAssign for Dd {
    #[inline]
    fn sub_assign(&mut self, b: Dd) {
        *self = *self - b;
    }
}

impl MulAssign for Dd {
    #[inline]
    fn mul_assign(&mut self, b: Dd) {
        *self = *self * b;
    }
}

/// Σ terms of a Taylor series x^{start}/start! · Σᵢ (±x²)ⁱ / ((start + 1) … (start + 2i)),
/// summed until the term is below 2⁻¹¹⁰ of the first.
fn taylor_sin_cos(x: Dd, start: u32) -> Dd {
    let x2 = x * x;
    let mut term = if start == 0 { Dd::ONE } else { x };
    let mut sum = term;
    let mut i = start;
    loop {
        term = (term * x2).div_f64(-(((i + 1) * (i + 2)) as f64));
        sum += term;
        i += 2;
        if term.hi.abs() <= sum.hi.abs() * 7.7e-34 || i > 60 {
            return sum;
        }
    }
}

impl Real for Dd {
    const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };
    const ONE: Dd = Dd { hi: 1.0, lo: 0.0 };
    const EPSILON: f64 = 4.93e-32;
    const NAME: &'static str = "dd";

    #[inline]
    fn from_f64(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    fn sqrt(self) -> Dd {
        if self.hi <= 0.0 {
            assert!(self.hi == 0.0, "square root of a negative number");
            return Dd::ZERO;
        }
        // One Newton step from the f64 root: x + (a − x²)/(2x).
        let x = self.hi.sqrt();
        let (p, e) = two_prod(x, x);
        let r = (self - Dd { hi: p, lo: e }).to_f64();
        Dd::new(x, r / (2.0 * x))
    }

    fn exp(self) -> Dd {
        if self.hi > 709.0 {
            return Dd::from_f64(f64::INFINITY);
        }
        if self.hi < -745.0 {
            return Dd::ZERO;
        }
        // x = k ln 2 + r, |r| ≤ ln 2 / 2; e^r = (e^{r/2¹⁰})^{2¹⁰} with e^{r/2¹⁰} − 1 by its
        // series, squared ten times in the form s ← 2s + s².
        let k = (self.hi / Dd::LN2.hi).round();
        let r = (self - Dd::LN2 * Dd::from_f64(k)) * Dd::from_f64(1.0 / 1024.0);
        let mut term = r;
        let mut s = r;
        let mut i = 1.0;
        loop {
            i += 1.0;
            term = (term * r).div_f64(i);
            s += term;
            if term.hi.abs() <= s.hi.abs() * 1e-34 || i > 30.0 {
                break;
            }
        }
        for _ in 0..10 {
            s = s * Dd::from_f64(2.0) + s * s;
        }
        let e = s + Dd::ONE;
        // Multiply by 2^k in two steps, so that a large |k| neither overflows nor
        // underflows early.
        let half = (k / 2.0).trunc();
        let f1 = 2f64.powi(half as i32);
        let f2 = 2f64.powi((k - half) as i32);
        Dd {
            hi: e.hi * f1 * f2,
            lo: e.lo * f1 * f2,
        }
    }

    fn sin_cos(self) -> (Dd, Dd) {
        // x = 2π j + r, then r = (π/2) q + t, |t| ≤ π/4.
        let two_pi = Dd::PI * Dd::from_f64(2.0);
        let j = (self / two_pi).round();
        let r = self - two_pi * Dd::from_f64(j);
        let half_pi = Dd::PI * Dd::from_f64(0.5);
        let q = (r / half_pi).round();
        let t = r - half_pi * Dd::from_f64(q);
        let (s, c) = (taylor_sin_cos(t, 1), taylor_sin_cos(t, 0));
        match (q as i64).rem_euclid(4) {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        }
    }

    fn pi() -> Dd {
        Dd::PI
    }

    #[inline]
    fn abs(self) -> Dd {
        if self.hi < 0.0 { -self } else { self }
    }

    #[inline]
    fn scale(self, f: f64) -> Dd {
        let (p1, p2) = two_prod(self.hi, f);
        let (hi, lo) = quick_two_sum(p1, p2 + self.lo * f);
        Dd { hi, lo }
    }
}

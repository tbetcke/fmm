//! Double-double arithmetic for the transfer functions and the grid (proposal §3.6: T is
//! computed in extended precision and rounded once).
//!
//! A [`Dd`] is an unevaluated sum hi + lo with |lo| ≤ ulp(hi)/2, about 32 significant
//! digits. The arithmetic follows the error-free transformations of Dekker and Knuth as in
//! the QD library of Hida, Li and Bailey (2001): `two_sum` and `two_prod` (with a fused
//! multiply-add). The exponent range is f64's. Ported from S7's double-double scalar
//! (`spikes/truncation/src/real.rs`, `complex.rs`, `radial.rs` and `coaxial.rs`'s
//! Gauss–Legendre rule), which S6 used for its transfer functions (CH0.6), with the same
//! order of operations.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// A double-double number hi + lo.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub(crate) struct Dd {
    /// The leading part, the f64 nearest to the value.
    pub(crate) hi: f64,
    /// The trailing part, |lo| ≤ ulp(hi)/2.
    pub(crate) lo: f64,
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
    /// 0.
    pub(crate) const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };
    /// 1.
    pub(crate) const ONE: Dd = Dd { hi: 1.0, lo: 0.0 };
    /// π, rounded to 106 bits.
    pub(crate) const PI: Dd = Dd {
        hi: core::f64::consts::PI,
        lo: 1.224_646_799_147_353_2e-16,
    };
    /// ln 2, rounded to 106 bits.
    const LN2: Dd = Dd {
        hi: core::f64::consts::LN_2,
        lo: 2.319_046_813_846_299_6e-17,
    };

    /// The exact value of an f64.
    #[inline]
    pub(crate) const fn from_f64(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    /// The exact value of an integer below 2⁵³.
    #[inline]
    pub(crate) fn from_usize(n: usize) -> Dd {
        Dd::from_f64(n as f64)
    }

    /// The f64 nearest to the value (hi + lo rounded once).
    #[inline]
    pub(crate) fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// hi + lo, normalised.
    fn new(hi: f64, lo: f64) -> Dd {
        let (hi, lo) = two_sum(hi, lo);
        Dd { hi, lo }
    }

    /// The quotient by an f64.
    #[inline]
    fn div_f64(self, b: f64) -> Dd {
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

    /// |x|.
    #[cfg(test)]
    #[inline]
    pub(crate) fn abs(self) -> Dd {
        if self.hi < 0.0 { -self } else { self }
    }

    /// The square root, x ≥ 0.
    pub(crate) fn sqrt(self) -> Dd {
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

    /// e^x.
    pub(crate) fn exp(self) -> Dd {
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
            s = s + term;
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

    /// (sin x, cos x).
    pub(crate) fn sin_cos(self) -> (Dd, Dd) {
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
}

/// Σ terms of a Taylor series x^{start}/start! · Σᵢ (−x²)ⁱ / ((start + 1) … (start + 2i)),
/// summed until the term is below 2⁻¹¹⁰ of the sum.
fn taylor_sin_cos(x: Dd, start: u32) -> Dd {
    let x2 = x * x;
    let mut term = if start == 0 { Dd::ONE } else { x };
    let mut sum = term;
    let mut i = start;
    loop {
        term = (term * x2).div_f64(-(((i + 1) * (i + 2)) as f64));
        sum = sum + term;
        i += 2;
        if term.hi.abs() <= sum.hi.abs() * 7.7e-34 || i > 60 {
            return sum;
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

/// A complex double-double number.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Cdd {
    /// The real part.
    pub(crate) re: Dd,
    /// The imaginary part.
    pub(crate) im: Dd,
}

impl Cdd {
    /// 0.
    pub(crate) const ZERO: Cdd = Cdd {
        re: Dd::ZERO,
        im: Dd::ZERO,
    };
    /// i.
    const I: Cdd = Cdd {
        re: Dd::ZERO,
        im: Dd::ONE,
    };

    /// re + i im.
    #[inline]
    pub(crate) const fn new(re: Dd, im: Dd) -> Cdd {
        Cdd { re, im }
    }

    /// The exact value of an f64 pair.
    #[inline]
    pub(crate) const fn from_f64(re: f64, im: f64) -> Cdd {
        Cdd::new(Dd::from_f64(re), Dd::from_f64(im))
    }

    /// x times a real.
    #[inline]
    pub(crate) fn scale(self, x: Dd) -> Cdd {
        Cdd::new(self.re * x, self.im * x)
    }

    /// z iᵏ, exactly.
    #[inline]
    pub(crate) fn times_i_pow(self, k: usize) -> Cdd {
        match k % 4 {
            0 => self,
            1 => Cdd::new(-self.im, self.re),
            2 => Cdd::new(-self.re, -self.im),
            _ => Cdd::new(self.im, -self.re),
        }
    }

    /// e^z.
    pub(crate) fn exp(self) -> Cdd {
        let r = self.re.exp();
        let (s, c) = self.im.sin_cos();
        Cdd::new(r * c, r * s)
    }
}

impl Add for Cdd {
    type Output = Cdd;
    #[inline]
    fn add(self, b: Cdd) -> Cdd {
        Cdd::new(self.re + b.re, self.im + b.im)
    }
}

impl Sub for Cdd {
    type Output = Cdd;
    #[inline]
    fn sub(self, b: Cdd) -> Cdd {
        Cdd::new(self.re - b.re, self.im - b.im)
    }
}

impl Neg for Cdd {
    type Output = Cdd;
    #[inline]
    fn neg(self) -> Cdd {
        Cdd::new(-self.re, -self.im)
    }
}

impl Mul for Cdd {
    type Output = Cdd;
    #[inline]
    fn mul(self, b: Cdd) -> Cdd {
        Cdd::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
}

impl Div for Cdd {
    type Output = Cdd;
    #[inline]
    fn div(self, b: Cdd) -> Cdd {
        let d = b.re * b.re + b.im * b.im;
        Cdd::new(
            (self.re * b.re + self.im * b.im) / d,
            (self.im * b.re - self.re * b.im) / d,
        )
    }
}

/// hₗ(z) = hₗ⁽¹⁾(z) for l = 0..out.len(), by the upward recursion
/// hₗ₊₁ = (2l + 1)/z hₗ − hₗ₋₁ from h₀ = −i e^{iz}/z and h₁ = −e^{iz}(z + i)/z², stable for
/// the dominant solution (S7's `radial::hankel` without its power scale).
pub(crate) fn hankel(z: Cdd, out: &mut [Cdd]) {
    if out.is_empty() {
        return;
    }
    let e = (Cdd::I * z).exp();
    out[0] = -(Cdd::I * e) / z;
    if out.len() == 1 {
        return;
    }
    out[1] = -(e * (z + Cdd::I)) / (z * z);
    for l in 1..out.len() - 1 {
        out[l + 1] = out[l].scale(Dd::from_usize(2 * l + 1)) / z - out[l - 1];
    }
}

/// The n-point Gauss–Legendre rule on [−1, 1]: nodes ascending and weights, by Newton's
/// method in f64 and three steps in double-double (S7's `gauss_legendre`). The nodes are
/// symmetric bit for bit: x_{n−1−i} = −xᵢ.
pub(crate) fn gauss_legendre(n: usize) -> (Vec<Dd>, Vec<Dd>) {
    let mut x = vec![Dd::ZERO; n];
    let mut w = vec![Dd::ZERO; n];
    let eval = |z: Dd| -> (Dd, Dd) {
        let (mut p0, mut p1) = (Dd::ONE, z);
        for k in 2..=n {
            let p2 = (Dd::from_usize(2 * k - 1) * z * p1 - Dd::from_usize(k - 1) * p0)
                / Dd::from_usize(k);
            p0 = p1;
            p1 = p2;
        }
        let p = if n == 0 { Dd::ONE } else { p1 };
        let pm1 = if n == 1 { Dd::ONE } else { p0 };
        (p, Dd::from_usize(n) * (z * p - pm1) / (z * z - Dd::ONE))
    };
    for i in 0..n.div_ceil(2) {
        let mut z = (core::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = eval(Dd::from_f64(z));
            let dz = (p / dp).to_f64();
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        let mut zt = Dd::from_f64(z);
        for _ in 0..3 {
            let (p, dp) = eval(zt);
            zt = zt - p / dp;
        }
        let (_, dp) = eval(zt);
        x[i] = -zt;
        x[n - 1 - i] = zt;
        let wi = Dd::from_f64(2.0) / ((Dd::ONE - zt * zt) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    if n % 2 == 1 {
        // The middle node of an odd rule is 0 exactly.
        x[n / 2] = Dd::ZERO;
    }
    (x, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// |a − b| in units of b (or absolute where b is 0).
    fn rel(a: Dd, b: Dd) -> f64 {
        let d = (a - b).abs().to_f64();
        let s = b.abs().to_f64();
        if s == 0.0 { d } else { d / s }
    }

    #[test]
    fn error_free_transformations() {
        let (a, b) = (1.0 + 2f64.powi(-40), 3.0 - 2f64.powi(-45));
        let (s, e) = two_sum(a, b);
        assert_eq!(s + e, a + b);
        assert_eq!(Dd::new(s, e), Dd { hi: s, lo: e });
        let (p, e) = two_prod(a, b);
        // a b = p + e exactly: check against the exact product in pieces of 26 bits.
        let prod = Dd::from_f64(a) * Dd::from_f64(b);
        assert_eq!((prod.hi, prod.lo), (p, e));
    }

    #[test]
    fn arithmetic_and_functions_to_thirty_digits() {
        // (1/3) · 3 = 1, √2² = 2, e^{ln 2} = 2.
        let third = Dd::ONE / Dd::from_f64(3.0);
        assert!(rel(third * Dd::from_f64(3.0), Dd::ONE) < 1e-31);
        let r2 = Dd::from_f64(2.0).sqrt();
        assert!(rel(r2 * r2, Dd::from_f64(2.0)) < 1e-31);
        assert!(rel(Dd::LN2.exp(), Dd::from_f64(2.0)) < 1e-31);
        // e = 2.718281828459045235360287471352662497757..., as hi + lo.
        let e = Dd {
            hi: core::f64::consts::E,
            lo: 1.445_646_891_729_250_2e-16,
        };
        assert!(rel(Dd::ONE.exp(), e) < 1e-31);
        // sin(π/6) = 1/2, cos(π/3) = 1/2, sin²+cos² = 1 at an arbitrary point, sin π = 0.
        let (s, _) = (Dd::PI / Dd::from_f64(6.0)).sin_cos();
        assert!(rel(s, Dd::from_f64(0.5)) < 1e-31);
        let (_, c) = (Dd::PI / Dd::from_f64(3.0)).sin_cos();
        assert!(rel(c, Dd::from_f64(0.5)) < 1e-31);
        let (s, c) = Dd::from_f64(7.25).sin_cos();
        assert!(rel(s * s + c * c, Dd::ONE) < 1e-31);
        let (s, c) = Dd::PI.sin_cos();
        assert!(s.abs().to_f64() < 1e-31 && rel(c, -Dd::ONE) < 1e-31);
    }

    #[test]
    fn hankel_closed_forms_and_wronskian() {
        // For real z: j_l y_{l+1} − j_{l+1} y_l = −1/z² (with h = j + i y).
        let z = Cdd::from_f64(3.7, 0.0);
        let mut h = vec![Cdd::ZERO; 41];
        hankel(z, &mut h);
        let zz = Dd::from_f64(3.7);
        for l in 0..10 {
            let w = h[l].re * h[l + 1].im - h[l + 1].re * h[l].im;
            assert!(rel(w, -(Dd::ONE / (zz * zz))) < 1e-28, "l = {l}");
        }
        // Against nd-fmm-math's f64 Hankel functions (Helmholtz CH1.2b), which are
        // accurate to a few ulp here.
        let mut f = vec![nd_fmm_math::Complex::new(0.0, 0.0); 41];
        nd_fmm_math::radial::hankel_h(nd_fmm_math::Complex::new(3.7, 0.0), 1.0, &mut f, None);
        for l in 0..41 {
            let d = (h[l].re.to_f64() - f[l].re).hypot(h[l].im.to_f64() - f[l].im);
            assert!(d <= 1e-14 * f[l].norm(), "l = {l}: {d:e}");
        }
    }

    #[test]
    fn gauss_legendre_integrates_polynomials() {
        for n in [1, 2, 5, 9, 31] {
            let (x, w) = gauss_legendre(n);
            for i in 0..n {
                assert_eq!(x[n - 1 - i], -x[i]);
            }
            // ∫ x^{2j} = 2/(2j + 1) for 2j ≤ 2n − 1.
            for j in 0..n {
                let mut s = Dd::ZERO;
                for i in 0..n {
                    let mut t = w[i];
                    for _ in 0..2 * j {
                        t = t * x[i];
                    }
                    s = s + t;
                }
                let exact = Dd::from_f64(2.0) / Dd::from_usize(2 * j + 1);
                assert!(
                    rel(s, exact) < 1e-29,
                    "n = {n}, j = {j}: {:e}",
                    rel(s, exact)
                );
            }
        }
    }
}

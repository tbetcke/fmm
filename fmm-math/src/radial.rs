//! Scaled spherical Bessel, Hankel and modified spherical Bessel functions of the
//! Helmholtz family (CONVENTIONS §3.15, "Radial functions and scaling").
//!
//! For a power scale s with 0 < s ≤ 1 (a level's sₗ = min(|k| aₗ, 1)), complex z with
//! Im z ≥ 0 and real x ≥ 0, the functions write orders 0 to `values.len() − 1` of
//!
//! ```text
//! ĵₙ(z) = e^{−Im z} jₙ(z) / sⁿ        ĥₙ(z) = e^{Im z} hₙ(z) sⁿ,   hₙ = hₙ⁽¹⁾ = jₙ + i yₙ
//! îₙ(x) = e⁻ˣ iₙ(x) / sⁿ              k̂ₙ(x) = eˣ kₙ(x) sⁿ
//! ```
//!
//! with iₙ(x) = √(π/(2x)) I_{n+½}(x) and kₙ(x) = √(2/(πx)) K_{n+½}(x), so that
//! k₀(x) = e⁻ˣ/x, jₙ(ix) = iⁿ iₙ(x) and hₙ(ix) = −i⁻ⁿ kₙ(x). On request they also write
//! the derivatives with the same factors (ĵ′ₙ = e^{−Im z} jₙ′(z)/sⁿ and so on): the
//! factors multiply the derivative and are not differentiated.
//!
//! # Method (spike S3, spikes/radial/SPIKE_REPORT.md §2)
//!
//! - [`bessel_j`] and [`modified_i`]: Miller's method in ratio form. The ratios
//!   rₙ = fₙ/fₙ₋₁ come from the backward recursion, started at the order where the forward
//!   recursion from the requested length has grown by 10²⁰ (FMM3D's test); then ĵ₀ (or ĵ₁,
//!   whichever is larger) from its closed form, a power series for |z| < 0.5, and the
//!   values upward. No pass overflows. The start order is a function of z and the
//!   length only, computed in f64 for every `T`.
//! - [`hankel_h`] and [`modified_k`]: the upward recursion from the closed forms of orders
//!   0 and 1, the stable direction.
//! - Derivatives from the neighbouring orders: jₙ′ = (n jₙ₋₁ − (n + 1) jₙ₊₁)/(2n + 1),
//!   hₙ′ = hₙ₋₁ − (n + 1) hₙ / z, iₙ′ = (n iₙ₋₁ + (n + 1) iₙ₊₁)/(2n + 1),
//!   kₙ′ = −kₙ₋₁ − (n + 1) kₙ / x; j₀′ = −j₁, h₀′ = −h₁, i₀′ = i₁, k₀′ = −k₁.
//!
//! Nothing allocates; the regular functions use `values` for the ratios on the way.
//! A value depends on (z, s) and the length: Miller's start order grows with the length,
//! so the values of the regular functions at low orders may change in the last bits when
//! more orders are requested. The singular functions do not depend on the length.
//!
//! # The tangential gradient
//!
//! The gradient of fₙ(k|v|) Yₙᵐ(v̂) needs fₙ(k|v|)/|v| = k fₙ(z)/z. Away from z = 0 this is
//! a division. At z = 0 (a target on the centre of a local expansion) the limits are
//! ĵ₁(z)/z → 1/(3s) and ĵₙ(z)/z → 0 for n ≥ 2, and the same for î; the n = 0 term has no
//! tangential part. Equivalently ĵₙ/z = (ĵₙ₋₁/s + s ĵₙ₊₁)/(2n + 1) for n ≥ 1.
//!
//! # Tested ranges
//!
//! `tests/radial/` checks f64 against arbitrary precision (CH1.2b, the sets of S3) on
//! every operator's range of CONVENTIONS §3.15 at ε down to 10⁻¹⁰ and kD up to 3000:
//! |z| from 10⁻¹² (10⁻⁶ for ĥ and k̂) to 4550, 0 ≤ arg z ≤ π/2, orders to 1482, within 10⁻¹³ (relative, or
//! relative to the oscillation envelope max(|jₙ|, |hₙ|) for jₙ where it has real zeros).
//! f32 is checked at s = 1, |z| ≤ 20 and n ≤ 8 only (CONVENTIONS §3.15 reserves it).
//!
//! # Panics
//!
//! Every function panics if s is not in (0, 1] or a derivative slice differs in length
//! from `values`, or if the argument is not finite; [`bessel_j`] and [`hankel_h`] if
//! Im z < 0; [`hankel_h`] at z = 0; [`modified_i`] if x < 0; [`modified_k`] unless x > 0.

use num_traits::{One, Zero};

use crate::{Complex, RealScalar};

/// Growth of the forward recursion at which Miller's start order is taken; the error of
/// the start then falls like its inverse square (FMM3D's `besseljs3d` uses the same).
const GROWTH: f64 = 1e20;

/// Below this modulus ĵ₀ and î₀ come from their power series (the closed forms cancel).
const SERIES: f64 = 0.5;

/// Terms of the power series below [`SERIES`]: the k-th is below (|z|²/2)ᵏ/(k! (2k+1)!!),
/// far below the f64 epsilon at k = 20.
const SERIES_TERMS: usize = 20;

/// The integer `k` as a `T`; exact for every k that occurs here.
#[inline(always)]
fn int<T: RealScalar>(k: usize) -> T {
    T::from_f64(k as f64)
}

fn check_scale<T: RealScalar>(s: T) {
    assert!(
        s > T::zero() && s <= T::one(),
        "the power scale must lie in (0, 1]"
    );
}

fn check_derivatives<U>(values: usize, derivatives: &Option<&mut [U]>) {
    if let Some(d) = derivatives {
        assert_eq!(
            d.len(),
            values,
            "derivatives must have the length of values"
        );
    }
}

/// ĵₙ(z) = e^{−Im z} jₙ(z)/sⁿ for n < `values.len()`, and ĵ′ₙ in `derivatives` if given.
/// Im z ≥ 0; z = 0 gives ĵ₀ = 1, ĵ′₁ = 1/(3s) and zeros elsewhere.
///
/// ```
/// use nd_fmm_math::Complex;
/// use nd_fmm_math::radial::bessel_j;
///
/// let z = Complex::new(2.0_f64, 0.0);
/// let mut j = [Complex::new(0.0, 0.0); 3];
/// bessel_j(z, 1.0, &mut j, None);
/// assert!((j[0].re - 2.0_f64.sin() / 2.0).abs() < 1e-15);
/// ```
pub fn bessel_j<T: RealScalar>(
    z: Complex<T>,
    s: T,
    values: &mut [Complex<T>],
    derivatives: Option<&mut [Complex<T>]>,
) {
    check_scale(s);
    assert!(
        z.re.is_finite() && z.im.is_finite() && z.im >= T::zero(),
        "bessel_j needs a finite z with Im z ≥ 0"
    );
    check_derivatives(values.len(), &derivatives);
    let len = values.len();
    if len == 0 {
        return;
    }
    // The ratios rₙ = jₙ/jₙ₋₁ for n = 1..=len (one beyond, for the derivative), by the
    // backward recursion rₙ = z/((2n + 1) − z rₙ₊₁) from 0 at the start order.
    // values[n − 1] holds rₙ for n < len; r_len is kept apart.
    let start = start_order_j(len, Complex::new(z.re.to_f64(), z.im.to_f64()));
    let mut r = Complex::zero();
    let mut r_top = Complex::zero();
    for n in (1..=start).rev() {
        let mut d = Complex::new(int::<T>(2 * n + 1), T::zero()) - z * r;
        if d == Complex::zero() {
            // jₙ₋₁(z) = 0 to the last bit: a denominator of rounding size keeps rₙ and
            // rₙ₋₁ finite, and their product, which is what the values use, accurate.
            d = Complex::new(T::epsilon() * int(2 * n + 1), T::zero());
        }
        r = div(z, d);
        if n == len {
            r_top = r;
        } else if n < len {
            values[n - 1] = r;
        }
    }
    // ĵ₀ or ĵ₁ from their closed forms, whichever is larger; then ĵₙ = ĵₙ₋₁ rₙ/s upward,
    // overwriting the ratios.
    let (j0, j1) = j0_j1(z);
    let use_j1 = j1.norm() > j0.norm();
    let mut prev = j0;
    for n in 1..len {
        let rn = values[n - 1];
        values[n - 1] = prev;
        prev = if n == 1 && use_j1 {
            j1 / s
        } else {
            prev * rn / s
        };
    }
    values[len - 1] = prev;
    if let Some(der) = derivatives {
        // ĵ_len, for the derivative of the last order.
        let next = if len == 1 && use_j1 {
            j1 / s
        } else {
            prev * r_top / s
        };
        for n in 0..len {
            let up = if n + 1 < len { values[n + 1] } else { next };
            der[n] = if n == 0 {
                -(up * s)
            } else {
                (values[n - 1] * (int::<T>(n) / s) - up * (int::<T>(n + 1) * s))
                    / int::<T>(2 * n + 1)
            };
        }
    }
}

/// ĥₙ(z) = e^{Im z} hₙ⁽¹⁾(z) sⁿ for n < `values.len()`, and ĥ′ₙ in `derivatives` if given.
/// Im z ≥ 0 and z ≠ 0.
///
/// ```
/// use nd_fmm_math::Complex;
/// use nd_fmm_math::radial::hankel_h;
///
/// // ĥ₀(z) = −i e^{i Re z}/z, so ĥ₀(3i) = −1/3.
/// let z = Complex::new(0.0_f64, 3.0);
/// let mut h = [Complex::new(0.0, 0.0); 2];
/// hankel_h(z, 1.0, &mut h, None);
/// assert!((h[0].re + 1.0 / 3.0).abs() < 1e-15 && h[0].im.abs() < 1e-15);
/// ```
pub fn hankel_h<T: RealScalar>(
    z: Complex<T>,
    s: T,
    values: &mut [Complex<T>],
    derivatives: Option<&mut [Complex<T>]>,
) {
    check_scale(s);
    assert!(
        z.re.is_finite() && z.im.is_finite() && z.im >= T::zero(),
        "hankel_h needs a finite z with Im z ≥ 0"
    );
    assert!(z != Complex::zero(), "hankel_h needs z ≠ 0");
    check_derivatives(values.len(), &derivatives);
    let len = values.len();
    if len == 0 {
        return;
    }
    // h₀ = −i e^{iz}/z, h₁ = −e^{iz}(z + i)/z², and e^{iz} e^{Im z} = e^{i Re z}.
    let e = Complex::cis(z.re);
    let h0 = -div(Complex::<T>::i() * e, z);
    let h1 = -div(div(e * (z + Complex::<T>::i()), z), z) * s;
    // hₙ₊₁ = (2n + 1)/z hₙ − hₙ₋₁, scaled.
    let (mut a, mut b) = (h0, h1);
    values[0] = h0;
    for (n, v) in values.iter_mut().enumerate().skip(1) {
        *v = b;
        let c = div(b * (int::<T>(2 * n + 1) * s), z) - a * (s * s);
        a = b;
        b = c;
    }
    if let Some(der) = derivatives {
        // ĥ₀′ = −ĥ₁/s; for n ≥ 1, ĥₙ′ = s ĥₙ₋₁ − (n + 1) ĥₙ / z.
        der[0] = -(if len > 1 { values[1] } else { h1 }) / s;
        for n in 1..len {
            der[n] = values[n - 1] * s - div(values[n] * int::<T>(n + 1), z);
        }
    }
}

/// îₙ(x) = e⁻ˣ iₙ(x)/sⁿ for n < `values.len()`, and î′ₙ in `derivatives` if given.
/// x ≥ 0; x = 0 gives î₀ = 1, î′₁ = 1/(3s) and zeros elsewhere.
pub fn modified_i<T: RealScalar>(x: T, s: T, values: &mut [T], derivatives: Option<&mut [T]>) {
    check_scale(s);
    assert!(
        x.is_finite() && x >= T::zero(),
        "modified_i needs a finite x ≥ 0"
    );
    check_derivatives(values.len(), &derivatives);
    let len = values.len();
    if len == 0 {
        return;
    }
    // Ratios rₙ = iₙ/iₙ₋₁ = x/((2n + 1) + x rₙ₊₁), all positive.
    let start = start_order_i(len, x.to_f64());
    let mut r = T::zero();
    let mut r_top = T::zero();
    for n in (1..=start).rev() {
        r = x / (int::<T>(2 * n + 1) + x * r);
        if n == len {
            r_top = r;
        } else if n < len {
            values[n - 1] = r;
        }
    }
    // î₀ = e⁻ˣ sinh x / x = −expm1(−2x)/(2x), or its power series.
    let two = int::<T>(2);
    let mut prev = if x == T::zero() {
        T::one()
    } else if x.to_f64() < SERIES {
        let w = x * x;
        let (mut sum, mut term) = (T::zero(), T::one());
        for k in 0..SERIES_TERMS {
            sum = sum + term;
            term = term * w / int::<T>((2 * k + 2) * (2 * k + 3));
        }
        sum * (-x).exp()
    } else {
        -(-two * x).exp_m1() / (two * x)
    };
    for n in 1..len {
        let rn = values[n - 1];
        values[n - 1] = prev;
        prev = prev * rn / s;
    }
    values[len - 1] = prev;
    if let Some(der) = derivatives {
        let next = prev * r_top / s;
        for n in 0..len {
            let up = if n + 1 < len { values[n + 1] } else { next };
            der[n] = if n == 0 {
                up * s
            } else {
                (values[n - 1] * (int::<T>(n) / s) + up * (int::<T>(n + 1) * s))
                    / int::<T>(2 * n + 1)
            };
        }
    }
}

/// k̂ₙ(x) = eˣ kₙ(x) sⁿ for n < `values.len()`, with k₀(x) = e⁻ˣ/x, and k̂′ₙ in
/// `derivatives` if given. x > 0.
///
/// ```
/// use nd_fmm_math::radial::modified_k;
///
/// let mut k = [0.0_f64; 2];
/// modified_k(4.0, 1.0, &mut k, None);
/// assert_eq!(k[0], 0.25); // k̂₀(x) = 1/x
/// ```
pub fn modified_k<T: RealScalar>(x: T, s: T, values: &mut [T], derivatives: Option<&mut [T]>) {
    check_scale(s);
    assert!(
        x.is_finite() && x > T::zero(),
        "modified_k needs a finite x > 0"
    );
    check_derivatives(values.len(), &derivatives);
    let len = values.len();
    if len == 0 {
        return;
    }
    // k₀ = e⁻ˣ/x, k₁ = e⁻ˣ(1 + x)/x², kₙ₊₁ = kₙ₋₁ + (2n + 1)/x kₙ.
    let k0 = T::one() / x;
    let k1 = (T::one() + x) / x / x * s;
    let (mut a, mut b) = (k0, k1);
    values[0] = k0;
    for (n, v) in values.iter_mut().enumerate().skip(1) {
        *v = b;
        let c = b * (int::<T>(2 * n + 1) * s) / x + a * (s * s);
        a = b;
        b = c;
    }
    if let Some(der) = derivatives {
        // k̂₀′ = −k̂₁/s; for n ≥ 1, k̂ₙ′ = −s k̂ₙ₋₁ − (n + 1) k̂ₙ / x.
        der[0] = -(if len > 1 { values[1] } else { k1 }) / s;
        for n in 1..len {
            der[n] = -(values[n - 1] * s) - values[n] * int::<T>(n + 1) / x;
        }
    }
}

/// Miller's start order for jₙ to order `len` (inclusive): the forward recursion
/// fₙ₊₁ = (2n + 1)/z fₙ − fₙ₋₁ from (0, 1) at `len` until it has grown by [`GROWTH`];
/// at least `len` + 2.
fn start_order_j(len: usize, z: Complex<f64>) -> usize {
    if z == Complex::zero() {
        return len + 2;
    }
    let (mut g, mut f) = (Complex::<f64>::zero(), Complex::<f64>::one());
    let mut n = len;
    while f.norm() < GROWTH {
        let next = div(f * (2 * n + 1) as f64, z) - g;
        g = f;
        f = next;
        n += 1;
    }
    n.max(len + 2)
}

/// [`start_order_j`] for iₙ, whose recursion iₙ₊₁ = iₙ₋₁ − (2n + 1)/x iₙ has the
/// forward-growing solution (−1)ⁿ kₙ.
fn start_order_i(len: usize, x: f64) -> usize {
    if x == 0.0 {
        return len + 2;
    }
    let (mut g, mut f) = (0.0f64, 1.0f64);
    let mut n = len;
    while f.abs() < GROWTH {
        let next = g - f * (2 * n + 1) as f64 / x;
        g = f;
        f = next;
        n += 1;
    }
    n.max(len + 2)
}

/// ĵ₀ and s ĵ₁ (e^{−Im z} j₁(z), without the power scale): the power series for
/// |z| < [`SERIES`], else sin z / z and sin z / z² − cos z / z with the exponential factor
/// removed. j₁'s closed form is used only where |j₁| > |j₀|, far from its cancellation.
fn j0_j1<T: RealScalar>(z: Complex<T>) -> (Complex<T>, Complex<T>) {
    if z == Complex::zero() {
        return (Complex::one(), Complex::zero());
    }
    let damp = (-z.im).exp();
    if z.norm().to_f64() < SERIES {
        let w = z * z * T::from_f64(-0.5);
        let (mut s0, mut s1) = (Complex::zero(), Complex::zero());
        let mut term = Complex::one(); // wᵏ / k!
        let (mut d0, mut d1) = (T::one(), int::<T>(3)); // (2k + 1)!!, (2k + 3)!!
        for k in 0..SERIES_TERMS {
            s0 = s0 + term / d0;
            s1 = s1 + term / d1;
            term = term * w / int::<T>(k + 1);
            d0 = d0 * int(2 * k + 3);
            d1 = d1 * int(2 * k + 5);
        }
        return (s0 * damp, z * s1 * damp);
    }
    // e^{±iz} e^{−Im z} without overflow (Im z ≥ 0).
    let plus = Complex::cis(z.re) * (-(z.im + z.im)).exp();
    let minus = Complex::cis(-z.re);
    let sin = (plus - minus) * Complex::new(T::zero(), T::from_f64(-0.5)); // ÷ 2i
    let cos = (plus + minus) * T::from_f64(0.5);
    (div(sin, z), div(div(sin, z), z) - div(cos, z))
}

/// a / b by Smith's algorithm, which forms neither |b|² nor a product of the two moduli.
/// `num-complex`'s division forms |b|², and its extra rounding raises the error of the
/// recursions (the Wronskian at |z| = 800 and n = 784 from 5.6e-14 to 1.0e-13).
#[inline]
fn div<T: RealScalar>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if b.re.abs() >= b.im.abs() {
        let r = b.im / b.re;
        let d = b.re + b.im * r;
        Complex::new((a.re + a.im * r) / d, (a.im - a.re * r) / d)
    } else {
        let r = b.re / b.im;
        let d = b.re * r + b.im;
        Complex::new((a.re * r + a.im) / d, (a.im * r - a.re) / d)
    }
}

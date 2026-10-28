//! Spherical Bessel and Hankel functions of complex argument with a power scale, in
//! either precision: S2's spike-grade algorithms (`spikes/coaxial/src/radial.rs`), which
//! S3 (`spikes/radial`) confirmed over every operator's range, without the exponential
//! factor (the spike's ranges of Im z stay far inside the f64 exponent range, §5.5's
//! pruning cut).
//!
//! - ĵₗ = jₗ(z)/gˡ by Miller's downward recursion, normalised by ĵ₀ or ĵ₁;
//! - ĥₗ = hₗ(z) gˡ, hₗ = hₗ⁽¹⁾, by the upward recursion, stable for the dominant solution;
//! - derivatives from the neighbouring order, jₗ′ = (l/z) jₗ − jₗ₊₁ and likewise for h.
//!
//! g is the level's scale s_l = min(|k| a_l, 1) (CH0.3's sign-off) or 1.

use crate::complex::Complex;
use crate::real::Real;

/// ĥₗ = hₗ(z) gˡ for l = 0..out.len().
pub fn hankel<T: Real>(z: Complex<T>, g: T, out: &mut [Complex<T>]) {
    if out.is_empty() {
        return;
    }
    // h₀ = −i e^{iz}/z, h₁ = −e^{iz}(z + i)/z².
    let e = (Complex::I * z).exp();
    out[0] = -(Complex::I * e) / z;
    if out.len() == 1 {
        return;
    }
    out[1] = (-(e * (z + Complex::I)) / (z * z)) * g;
    let g2 = g * g;
    for l in 1..out.len() - 1 {
        out[l + 1] = out[l] * (T::from_usize(2 * l + 1) * g) / z - out[l - 1] * g2;
    }
}

/// ĵ₀ = j₀(z) and j₁(z): their series for |z| < 1/2, else the closed forms
/// sin z / z and sin z / z² − cos z / z.
fn j0_j1<T: Real>(z: Complex<T>) -> (Complex<T>, Complex<T>) {
    if z.abs() < T::from_f64(0.5) {
        let w = z * z * T::from_f64(-0.5);
        let (mut s0, mut s1) = (Complex::ZERO, Complex::ZERO);
        let mut term = Complex::ONE; // w^k / k!
        let (mut d0, mut d1) = (T::ONE, T::from_f64(3.0)); // (2k + 1)!!, (2k + 3)!!
        for k in 0..30 {
            s0 += term / d0;
            s1 += term / d1;
            term = term * w / T::from_usize(k + 1);
            d0 *= T::from_usize(2 * k + 3);
            d1 *= T::from_usize(2 * k + 5);
        }
        return (s0, z * s1);
    }
    let plus = (Complex::I * z).exp();
    let minus = (-(Complex::I * z)).exp();
    let sin = (plus - minus) / Complex::new(T::ZERO, T::from_f64(2.0));
    let cos = (plus + minus) * T::from_f64(0.5);
    (sin / z, sin / (z * z) - cos / z)
}

/// ĵₗ = jₗ(z)/gˡ for l = 0..out.len(), by Miller's downward recursion from an order well
/// above max(L, |z|), rescaled on the way down so nothing overflows, and normalised by
/// ĵ₀ or ĵ₁, whichever is larger. Values below the exponent range underflow to zero.
pub fn bessel_j<T: Real>(z: Complex<T>, g: T, out: &mut [Complex<T>]) {
    let len = out.len();
    if len == 0 {
        return;
    }
    let (j0, j1) = j0_j1(z);
    // f64: S2's start; double-double: further up, for the extra digits.
    let (factor, extra) = if T::EPSILON < 1e-20 {
        (1.3, 120.0)
    } else {
        (1.2, 60.0)
    };
    let start = (factor * ((len - 1) as f64).max(z.abs().to_f64()) + extra) as usize;
    let g2 = g * g;
    let mut upper = Complex::ZERO; // u_{l+1}
    let mut current = Complex::ONE; // u_l
    out.fill(Complex::ZERO);
    if start < len {
        out[start] = current;
    }
    let big = T::from_f64(1e200);
    let small = T::from_f64(1e-200);
    for l in (1..=start).rev() {
        let lower = current * (T::from_usize(2 * l + 1) * g) / z - upper * g2;
        upper = current;
        current = lower;
        if l - 1 < len {
            out[l - 1] = current;
        }
        if current.max_abs() > big {
            current = current * small;
            upper = upper * small;
            for o in out[(l - 1).min(len)..].iter_mut() {
                *o = *o * small;
            }
        }
    }
    let c = if j0.max_abs() >= j1.max_abs() || len < 2 {
        j0 / out[0]
    } else {
        (j1 / g) / out[1]
    };
    for o in out.iter_mut() {
        *o = *o * c;
    }
}

/// Values and derivatives of ĵₗ (`regular`) or ĥₗ for l = 0..=n_max, into `value` and
/// `deriv` (each of length n_max + 1), with `work` of length n_max + 2:
/// ĵₗ′ = jₗ′(z)/gˡ = (l/z) ĵₗ − g ĵₗ₊₁ and ĥₗ′ = hₗ′(z) gˡ = (l/z) ĥₗ − ĥₗ₊₁/g.
pub fn with_derivative<T: Real>(
    regular: bool,
    z: Complex<T>,
    g: T,
    value: &mut [Complex<T>],
    deriv: &mut [Complex<T>],
    work: &mut [Complex<T>],
) {
    let n = value.len();
    assert!(deriv.len() == n && work.len() == n + 1);
    if regular {
        bessel_j(z, g, work);
    } else {
        hankel(z, g, work);
    }
    let up = if regular { g } else { T::ONE / g };
    for l in 0..n {
        value[l] = work[l];
        deriv[l] = work[l] * T::from_usize(l) / z - work[l + 1] * up;
    }
}

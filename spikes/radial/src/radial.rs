//! The radial functions of the Helmholtz family with the spike's scaling (SPIKE_REPORT.md
//! §1), for orders 0..out.len():
//!
//! - ĵₙ = e^{−|Im z|} jₙ(z)/sⁿ and ĥₙ = e^{Im z} hₙ(z) sⁿ, hₙ = hₙ⁽¹⁾, for complex z with
//!   Im z ≥ 0 (the branch Im k ≥ 0);
//! - îₙ = e^{−x} iₙ(x)/sⁿ and k̂ₙ = eˣ kₙ(x) sⁿ for real x ≥ 0 (x > 0 for k), with kₙ
//!   normalised so that k₀(x) = e^{−x}/x and hₙ(ix) = −i⁻ⁿ kₙ(x);
//! - the derivatives with the same factors, ĵ′ₙ = e^{−|Im z|} jₙ′(z)/sⁿ and so on (the
//!   factors multiply the derivative; they are not differentiated).
//!
//! s is the power scale of the level, 0 < s ≤ 1. The regular functions (j, i) come from
//! Miller's method in ratio form, the singular ones (h, k) from the upward recursion,
//! each in its stable direction. Nothing allocates.

use crate::complex::C64;

/// Growth of the forward recursion at which Miller's start order is taken: the start
/// order's error then falls like its inverse square (FMM3D's `besseljs3d` uses the same
/// 10²⁰).
const GROWTH: f64 = 1e20;

/// Below this modulus ĵ₀ and î₀ come from their power series (the closed forms cancel).
const SERIES: f64 = 0.5;

/// ĵₙ, and ĵ′ₙ if `der` is given (same length as `out`).
pub fn bessel_j(z: C64, s: f64, out: &mut [C64], der: Option<&mut [C64]>) {
    let len = out.len();
    if len == 0 {
        return;
    }
    // The ratios rₙ = jₙ/jₙ₋₁ for n = 1..=len (one beyond, for the derivative), by the
    // backward recursion rₙ = z/((2n + 1) − z rₙ₊₁) from rₙ₊₁ = 0 at the start order.
    // out[n − 1] holds rₙ for n < len + 1; r_len is kept apart.
    let start = miller_start(len, |n, f, g| (f * ((2 * n + 1) as f64)) / z - g, z.abs());
    let mut r = C64::ZERO;
    let mut r_top = C64::ZERO;
    for n in (1..=start).rev() {
        let mut d = C64::new((2 * n + 1) as f64, 0.0) - z * r;
        if d == C64::ZERO {
            // jₙ₋₁(z) = 0 to the last bit: any tiny denominator keeps the ratios finite.
            d = C64::new(f64::MIN_POSITIVE, 0.0);
        }
        r = z / d;
        if n == len {
            r_top = r;
        } else if n < len {
            out[n - 1] = r;
        }
    }
    // The normalisation: ĵ₀ or ĵ₁ from their closed forms, whichever is larger; then
    // ĵₙ = ĵₙ₋₁ rₙ/s upward, overwriting the ratios.
    let (j0, j1) = j0_j1(z);
    let use_j1 = j1.abs() > j0.abs();
    let mut prev = j0;
    for n in 1..len {
        let rn = out[n - 1];
        out[n - 1] = prev;
        prev = if n == 1 && use_j1 {
            j1 / s
        } else {
            prev * rn / s
        };
    }
    out[len - 1] = prev;
    // ĵ_len, for the derivative of the last order.
    let next = if len == 1 && use_j1 {
        j1 / s
    } else {
        prev * r_top / s
    };
    if let Some(der) = der {
        assert_eq!(der.len(), len);
        // jₙ′ = (n jₙ₋₁ − (n + 1) jₙ₊₁)/(2n + 1); j₀′ = −j₁.
        for n in 0..len {
            let up = if n + 1 < len { out[n + 1] } else { next };
            der[n] = if n == 0 {
                -(up * s)
            } else {
                (out[n - 1] * (n as f64 / s) - up * ((n + 1) as f64 * s)) / (2 * n + 1) as f64
            };
        }
    }
}

/// ĥₙ, and ĥ′ₙ if `der` is given; z ≠ 0.
pub fn hankel(z: C64, s: f64, out: &mut [C64], der: Option<&mut [C64]>) {
    let len = out.len();
    if len == 0 {
        return;
    }
    // h₀ = −i e^{iz}/z, h₁ = −e^{iz}(z + i)/z², and e^{iz} e^{Im z} = e^{i Re z}.
    let e = C64::cis(z.re);
    let h0 = -(C64::I * e) / z;
    let h1 = -(e * (z + C64::I)) / (z * z) * s;
    // hₙ₊₁ = (2n + 1)/z hₙ − hₙ₋₁, scaled.
    let (mut a, mut b) = (h0, h1);
    out[0] = h0;
    for (n, o) in out.iter_mut().enumerate().skip(1) {
        *o = b;
        let c = (b * ((2 * n + 1) as f64 * s)) / z - a * (s * s);
        a = b;
        b = c;
    }
    let next = if len == 1 { h1 } else { b };
    if let Some(der) = der {
        assert_eq!(der.len(), len);
        // hₙ′ = hₙ₋₁ − (n + 1)/z hₙ; h₀′ = −h₁.
        for n in 0..len {
            der[n] = if n == 0 {
                -(if len > 1 { out[1] } else { next }) / s
            } else {
                out[n - 1] * s - (out[n] * (n + 1) as f64) / z
            };
        }
    }
}

/// îₙ, and î′ₙ if `der` is given; x ≥ 0.
pub fn modified_i(x: f64, s: f64, out: &mut [f64], der: Option<&mut [f64]>) {
    let len = out.len();
    if len == 0 {
        return;
    }
    // Ratios rₙ = iₙ/iₙ₋₁ = x/((2n + 1) + x rₙ₊₁), all positive.
    let start = miller_start_real(len, x);
    let mut r = 0.0;
    let mut r_top = 0.0;
    for n in (1..=start).rev() {
        r = x / ((2 * n + 1) as f64 + x * r);
        if n == len {
            r_top = r;
        } else if n < len {
            out[n - 1] = r;
        }
    }
    // î₀ = e^{−x} sinh x / x = −expm1(−2x)/(2x).
    let mut prev = if x == 0.0 {
        1.0
    } else if x < SERIES {
        let w = x * x;
        let (mut sum, mut term) = (0.0, 1.0);
        for k in 0..20 {
            sum += term;
            term *= w / ((2 * k + 2) * (2 * k + 3)) as f64;
        }
        sum * (-x).exp()
    } else {
        -(-2.0 * x).exp_m1() / (2.0 * x)
    };
    for n in 1..len {
        let rn = out[n - 1];
        out[n - 1] = prev;
        prev = prev * rn / s;
    }
    out[len - 1] = prev;
    let next = prev * r_top / s;
    if let Some(der) = der {
        assert_eq!(der.len(), len);
        // iₙ′ = (n iₙ₋₁ + (n + 1) iₙ₊₁)/(2n + 1); i₀′ = i₁.
        for n in 0..len {
            let up = if n + 1 < len { out[n + 1] } else { next };
            der[n] = if n == 0 {
                up * s
            } else {
                (out[n - 1] * (n as f64 / s) + up * ((n + 1) as f64 * s)) / (2 * n + 1) as f64
            };
        }
    }
}

/// k̂ₙ, and k̂′ₙ if `der` is given; x > 0.
pub fn modified_k(x: f64, s: f64, out: &mut [f64], der: Option<&mut [f64]>) {
    let len = out.len();
    if len == 0 {
        return;
    }
    // k₀ = e^{−x}/x, k₁ = e^{−x}(1 + x)/x², kₙ₊₁ = kₙ₋₁ + (2n + 1)/x kₙ.
    let k0 = 1.0 / x;
    let k1 = (1.0 + x) / (x * x) * s;
    let (mut a, mut b) = (k0, k1);
    out[0] = k0;
    for (n, o) in out.iter_mut().enumerate().skip(1) {
        *o = b;
        let c = b * ((2 * n + 1) as f64 * s) / x + a * (s * s);
        a = b;
        b = c;
    }
    if let Some(der) = der {
        assert_eq!(der.len(), len);
        // kₙ′ = −kₙ₋₁ − (n + 1)/x kₙ; k₀′ = −k₁.
        for n in 0..len {
            der[n] = if n == 0 {
                -(if len > 1 { out[1] } else { k1 }) / s
            } else {
                -(out[n - 1] * s) - out[n] * (n + 1) as f64 / x
            };
        }
    }
}

/// Miller's start order for orders to `len` (inclusive): the forward recursion of the
/// three-term relation from (0, 1) at `len` until it has grown by [`GROWTH`]; at least
/// `len` + 2.
fn miller_start(len: usize, step: impl Fn(usize, C64, C64) -> C64, modulus: f64) -> usize {
    if modulus == 0.0 {
        return len + 2;
    }
    let (mut g, mut f) = (C64::ZERO, C64::new(1.0, 0.0));
    let mut n = len;
    while f.abs() < GROWTH {
        let next = step(n, f, g);
        g = f;
        f = next;
        n += 1;
    }
    n.max(len + 2)
}

/// [`miller_start`] for the modified recursion iₙ₊₁ = iₙ₋₁ − (2n + 1)/x iₙ, whose
/// forward-growing solution is (−1)ⁿ kₙ.
fn miller_start_real(len: usize, x: f64) -> usize {
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

/// ĵ₀ and ĵ₁·s (that is, e^{−|Im z|} j₁(z), unscaled by s): the power series for
/// |z| < [`SERIES`], else the closed forms sin z / z and sin z / z² − cos z / z with the
/// exponential factor removed. j₁'s closed form is only used where |j₁| > |j₀|, far from
/// its cancellation at small |z|.
fn j0_j1(z: C64) -> (C64, C64) {
    if z == C64::ZERO {
        return (C64::new(1.0, 0.0), C64::ZERO);
    }
    let damp = (-z.im.abs()).exp();
    if z.abs() < SERIES {
        let w = z * z * -0.5;
        let (mut s0, mut s1) = (C64::ZERO, C64::ZERO);
        let mut term = C64::new(1.0, 0.0); // w^k / k!
        let (mut d0, mut d1) = (1.0f64, 3.0f64); // (2k + 1)!!, (2k + 3)!!
        for k in 0..20 {
            s0 += term / d0;
            s1 += term / d1;
            term = term * w / (k + 1) as f64;
            d0 *= (2 * k + 3) as f64;
            d1 *= (2 * k + 5) as f64;
        }
        return (s0 * damp, z * s1 * damp);
    }
    // e^{±iz} e^{−|Im z|} without overflow.
    let plus = C64::cis(z.re) * (-z.im - z.im.abs()).exp();
    let minus = C64::cis(-z.re) * (z.im - z.im.abs()).exp();
    let sin = (plus - minus) / C64::new(0.0, 2.0);
    let cos = (plus + minus) * 0.5;
    (sin / z, sin / (z * z) - cos / z)
}

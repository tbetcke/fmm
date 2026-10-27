//! Spike-grade spherical Bessel and Hankel functions of complex argument, normalised by
//! their exponential factor. Spike S3 (CH0.3) studies the radial functions properly;
//! here they only supply the start column of the recursions, and the tests check them
//! against the reference.

use crate::complex::C64;

/// ĥₗ = hₗ(z) e^{Im z} gˡ for l = 0..out.len(), hₗ = hₗ⁽¹⁾, by the upward recursion,
/// which is stable for the dominant solution in l. The factor e^{Im z} removes the decay
/// (or growth) of e^{iz}; g is the power scale (1, or FMM3D's min(|k| a, 1)).
pub fn hankel(z: C64, g: f64, out: &mut [C64]) {
    if out.is_empty() {
        return;
    }
    // h₀ = −i e^{iz}/z, h₁ = −e^{iz}(z + i)/z², and e^{iz} e^{Im z} = e^{i Re z}.
    let e = C64::cis(z.re);
    out[0] = -(C64::I * e) / z;
    if out.len() == 1 {
        return;
    }
    out[1] = -(e * (z + C64::I)) / (z * z) * g;
    for l in 1..out.len() - 1 {
        out[l + 1] = (out[l] * ((2 * l + 1) as f64 * g)) / z - out[l - 1] * (g * g);
    }
}

/// ĵₗ = jₗ(z) e^{−|Im z|} for l = 0..out.len(), by Miller's downward recursion from an
/// order well above max(L, |z|), rescaled on the way down so nothing overflows, and
/// normalised by ĵ₀ or ĵ₁, whichever is larger. Values below the f64 range underflow to
/// zero.
pub fn bessel_j(z: C64, out: &mut [C64]) {
    let len = out.len();
    if len == 0 {
        return;
    }
    let (j0, j1) = j0_j1(z);
    let start = (1.2 * ((len - 1) as f64).max(z.abs())) as usize + 60;
    let mut upper = C64::ZERO; // u_{l+1}
    let mut current = C64::new(1.0, 0.0); // u_l
    for o in out.iter_mut() {
        *o = C64::ZERO;
    }
    if start < len {
        out[start] = current;
    }
    for l in (1..=start).rev() {
        let lower = (current * (2 * l + 1) as f64) / z - upper;
        upper = current;
        current = lower;
        if l - 1 < len {
            out[l - 1] = current;
        }
        if current.abs() > 1e200 {
            let s = 1e-200;
            current = current * s;
            upper = upper * s;
            for o in out[(l - 1).min(len)..].iter_mut() {
                *o = *o * s;
            }
        }
    }
    let c = if j0.abs() >= j1.abs() || len < 2 {
        j0 / out[0]
    } else {
        j1 / out[1]
    };
    for o in out.iter_mut() {
        *o = *o * c;
    }
}

/// ĵ₀ and ĵ₁ (normalised as [`bessel_j`]): their series for |z| < 0.5, else the closed
/// forms sin z / z and sin z / z² − cos z / z with the exponential factor removed.
fn j0_j1(z: C64) -> (C64, C64) {
    let damp = (-z.im.abs()).exp();
    if z.abs() < 0.5 {
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

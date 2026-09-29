//! Harmonicity (CONVENTIONS §3.4): ΔRₙᵐ = 0 and ΔIₙᵐ = 0 away from the origin, checked
//! with a fourth-order finite-difference Laplacian in f64.

use nd_fmm_math::Layout;
use nd_fmm_math::harmonics::{irregular, regular};

use crate::common::{SplitMix64, norm};

const P: usize = 30;

/// Largest, over the degrees n ≤ P, of max_m |Δₕ Xₙᵐ(x)| / S_n with
/// S_n = (n + 1)² max_m |Xₙᵐ(x)| / |x|², for the fourth-order central stencil
/// (−f₋₂ + 16f₋₁ − 30f₀ + 16f₁ − f₂) / (12h²) on each axis.
///
/// S_n is the size of the second derivatives of a degree-n solid harmonic, so the
/// measure is the fraction of them that fails to cancel; a non-harmonic function gives
/// order one. Without the factor (n + 1)² no step h reaches 1e-6 at n = 30: the
/// stencil's truncation error and the values' own rounding error, amplified by 1/h²,
/// both stay near 5e-7.
fn laplacian_residual(x: [f64; 3], h: f64, f: fn(usize, [f64; 3], &mut [f64])) -> f64 {
    let layout = Layout::new(P);
    let eval = |point: [f64; 3]| {
        let mut out = vec![0.0; layout.len()];
        f(P, point, &mut out);
        out
    };
    let center = eval(x);
    let mut lap = vec![0.0; layout.len()];
    for axis in 0..3 {
        for (offset, weight) in [(-2.0, -1.0), (-1.0, 16.0), (1.0, 16.0), (2.0, -1.0)] {
            let mut point = x;
            point[axis] += offset * h;
            for (l, v) in lap.iter_mut().zip(eval(point)) {
                *l += weight * v;
            }
        }
        for (l, v) in lap.iter_mut().zip(&center) {
            *l -= 30.0 * v;
        }
    }
    let r2 = norm(x).powi(2);
    let mut worst = 0.0_f64;
    for (n, range) in layout.degrees() {
        let scale = center[range.clone()]
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let residual = lap[range].iter().fold(0.0_f64, |acc, v| acc.max(v.abs())) / (12.0 * h * h);
        let second_derivatives = ((n + 1) * (n + 1)) as f64 * scale / r2;
        worst = worst.max(residual / second_derivatives);
    }
    worst
}

fn check(seed: u64, rmin: f64, rmax: f64, f: fn(usize, [f64; 3], &mut [f64]), name: &str) {
    let mut rng = SplitMix64::new(seed);
    let mut worst = 0.0_f64;
    for _ in 0..20 {
        let x = rng.point_in_shell(rmin, rmax);
        // Relative to |Xₙᵐ| / |x|², the truncation error of the stencil is about
        // (h/|x|)⁴ n⁶ / 90 and its rounding error about 16 n ε (|x|/h)²; h = 2.5e-4 |x|
        // keeps both below 1e-6 at n = 30.
        worst = worst.max(laplacian_residual(x, 2.5e-4 * norm(x), f));
    }
    println!("harmonicity, {name}, p = {P}: worst scaled residual {worst:.2e}");
    assert!(worst < 1e-6, "{name}: discrete Laplacian {worst:e} >= 1e-6");
}

#[test]
fn regular_harmonics_are_harmonic() {
    check(0x4a4d_0001, 0.5, 3.0_f64.sqrt(), regular, "regular");
}

#[test]
fn irregular_harmonics_are_harmonic() {
    check(0x4a4d_0002, 2.0, 8.0, irregular, "irregular");
}

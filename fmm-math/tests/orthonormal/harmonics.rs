//! The harmonics and their derivatives against the Racah harmonics, identities,
//! quadrature and differences (tests 1, 3, 4, 5 and 10 of the plan).

use nd_fmm_math::harmonics::regular;
use nd_fmm_math::orthonormal::{DerivativeScratch, harmonics, harmonics_derivatives};
use nd_fmm_math::{Layout, RealScalar};
use std::f64::consts::PI;

use crate::common::{SplitMix64, norm};
use crate::support::normalisation;

fn eval(p: usize, x: [f64; 3]) -> Vec<f64> {
    let mut out = vec![f64::NAN; Layout::new(p).len()];
    harmonics(p, x, &mut out);
    out
}

/// (value, ∂θ, (1/sin θ) ∂φ) to degree p.
fn eval_derivatives(p: usize, x: [f64; 3]) -> [Vec<f64>; 3] {
    let len = Layout::new(p).len();
    let (mut v, mut t, mut f) = (
        vec![f64::NAN; len],
        vec![f64::NAN; len],
        vec![f64::NAN; len],
    );
    harmonics_derivatives(p, x, &mut v, &mut t, &mut f, &mut DerivativeScratch::new(p));
    [v, t, f]
}

/// Largest |a − b| over degree n, relative to the largest |b| of the degree (1 if
/// that is zero), maximised over n ≤ p.
fn degree_error(p: usize, a: &[f64], b: &[f64]) -> f64 {
    let layout = Layout::new(p);
    let mut worst = 0.0f64;
    for (_, range) in layout.degrees() {
        let scale = b[range.clone()].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = if scale > 0.0 { scale } else { 1.0 };
        let diff = a[range.clone()]
            .iter()
            .zip(&b[range])
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        worst = worst.max(diff / scale);
    }
    worst
}

/// The point at radius r in the direction (θ, φ).
fn spherical(r: f64, theta: f64, phi: f64) -> [f64; 3] {
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    [r * st * cp, r * st * sp, r * ct]
}

/// Test 1: rⁿ Yₙᵐ = √((2n + 1)/(4π)) times the slot-m entry of N Rₙ (§3.15), p = 30.
#[test]
fn equal_the_racah_harmonics_through_n() {
    let p = 30;
    let layout = Layout::new(p);
    let mut rng = SplitMix64::new(20261029);
    let mut worst = 0.0f64;
    for _ in 0..200 {
        let x = rng.point_in_shell(0.2, 3.0);
        let r = norm(x);
        let mut racah = vec![0.0; layout.len()];
        regular(p, x, &mut racah);
        let expected: Vec<f64> = (0..layout.len())
            .map(|i| {
                let (n, m) = layout.nm(i);
                ((2 * n + 1) as f64 / (4.0 * PI)).sqrt() * normalisation(n, m) * racah[i]
                    / r.powi(n as i32)
            })
            .collect();
        worst = worst.max(degree_error(p, &eval(p, x), &expected));
    }
    assert!(worst <= 1e-13, "worst {worst:e}");
}

/// Test 3: Σₘ Yₙᵐ(a) Yₙᵐ(b) = (2n + 1)/(4π) Pₙ(a · b), with Pₙ(t) = √(4π/(2n + 1)) Yₙ⁰
/// at a third direction with cos θ = t, n ≤ 300.
#[test]
fn satisfy_the_addition_formula() {
    let p = 300;
    let layout = Layout::new(p);
    let mut rng = SplitMix64::new(7);
    let mut worst = 0.0f64;
    for _ in 0..10 {
        let (a, b) = (rng.unit_vector(), rng.unit_vector());
        let t: f64 = a.iter().zip(&b).map(|(u, v)| u * v).sum();
        let c = [(1.0 - t * t).max(0.0).sqrt(), 0.0, t];
        let (ya, yb, yc) = (eval(p, a), eval(p, b), eval(p, c));
        for (n, range) in layout.degrees() {
            let sum: f64 = ya[range.clone()]
                .iter()
                .zip(&yb[range])
                .map(|(u, v)| u * v)
                .sum();
            let factor = (2 * n + 1) as f64 / (4.0 * PI);
            let expected = factor.sqrt() * yc[layout.idx(n, 0)];
            worst = worst.max((sum - expected).abs() / factor);
        }
    }
    assert!(worst <= 1e-12, "worst {worst:e}");
}

/// Gauss–Legendre nodes and weights on [−1, 1] by Newton's method.
fn gauss_legendre(count: usize) -> Vec<(f64, f64)> {
    (1..=count)
        .map(|i| {
            let mut x = (PI * (i as f64 - 0.25) / (count as f64 + 0.5)).cos();
            loop {
                let (mut p0, mut p1) = (1.0, x);
                for j in 2..=count {
                    let p2 = ((2 * j - 1) as f64 * x * p1 - (j - 1) as f64 * p0) / j as f64;
                    p0 = p1;
                    p1 = p2;
                }
                let dp = count as f64 * (x * p1 - p0) / (x * x - 1.0);
                let dx = p1 / dp;
                x -= dx;
                if dx.abs() < 1e-15 {
                    let (mut p0, mut p1) = (1.0, x);
                    for j in 2..=count {
                        let p2 = ((2 * j - 1) as f64 * x * p1 - (j - 1) as f64 * p0) / j as f64;
                        p0 = p1;
                        p1 = p2;
                    }
                    let dp = count as f64 * (x * p1 - p0) / (x * x - 1.0);
                    return (x, 2.0 / ((1.0 - x * x) * dp * dp));
                }
            }
        })
        .collect()
}

/// Test 4: ∫ Yᵢ Yⱼ dΩ = δᵢⱼ for n ≤ 30, by a product rule exact for these degrees.
#[test]
fn are_orthonormal() {
    let p = 30;
    let len = Layout::new(p).len();
    let nodes = gauss_legendre(p + 2);
    let nphi = 2 * p + 2;
    let mut gram = vec![0.0; len * len];
    for &(t, w) in &nodes {
        for k in 0..nphi {
            let phi = 2.0 * PI * k as f64 / nphi as f64;
            let y = eval(p, spherical(1.0, t.acos(), phi));
            let weight = w * 2.0 * PI / nphi as f64;
            for i in 0..len {
                for j in 0..len {
                    gram[i * len + j] += weight * y[i] * y[j];
                }
            }
        }
    }
    let worst = (0..len * len).fold(0.0f64, |m, e| {
        let identity = if e / len == e % len { 1.0 } else { 0.0 };
        m.max((gram[e] - identity).abs())
    });
    assert!(worst <= 1e-13, "worst {worst:e}");
}

/// Test 5: ∂θ Y and (1/sin θ) ∂φ Y against central differences of `harmonics`, n ≤ 20,
/// and the values of `harmonics_derivatives` equal those of `harmonics` bit for bit.
#[test]
fn derivatives_match_differences() {
    let p = 20;
    let h = 1e-5;
    let mut rng = SplitMix64::new(11);
    let (mut worst_t, mut worst_f) = (0.0f64, 0.0f64);
    for _ in 0..20 {
        let (theta, phi, r) = (rng.range(0.2, 2.9), rng.range(-PI, PI), rng.range(0.5, 2.0));
        let x = spherical(r, theta, phi);
        let [v, t, f] = eval_derivatives(p, x);
        assert_eq!(v, eval(p, x));
        let diff = |a: [f64; 3], b: [f64; 3], scale: f64| -> Vec<f64> {
            eval(p, a)
                .iter()
                .zip(eval(p, b))
                .map(|(u, w)| (u - w) / (2.0 * h * scale))
                .collect()
        };
        let dt = diff(
            spherical(r, theta + h, phi),
            spherical(r, theta - h, phi),
            1.0,
        );
        let df = diff(
            spherical(r, theta, phi + h),
            spherical(r, theta, phi - h),
            theta.sin(),
        );
        worst_t = worst_t.max(degree_error(p, &t, &dt));
        worst_f = worst_f.max(degree_error(p, &f, &df));
    }
    assert!(
        worst_t <= 1e-7 && worst_f <= 1e-7,
        "∂θ {worst_t:e}, ∂φ {worst_f:e}"
    );
}

/// Test 10: lower degrees do not depend on p (bit for bit); x = 0 takes the direction
/// +z; on the axis Yₙ⁰(±ẑ) = (±1)ⁿ √((2n + 1)/(4π)) and every other order vanishes.
#[test]
fn are_consistent_across_degrees_and_on_the_axis() {
    let x = [0.3, -0.4, 0.7];
    let (small, large) = (eval(10, x), eval(40, x));
    assert_eq!(small, large[..small.len()]);
    let (small, large) = (eval_derivatives(10, x), eval_derivatives(40, x));
    for (s, l) in small.iter().zip(&large) {
        assert_eq!(s[..], l[..s.len()]);
    }
    assert_eq!(eval(12, [0.0; 3]), eval(12, [0.0, 0.0, 1.0]));
    let layout = Layout::new(40);
    for (sign, point) in [(1.0, [0.0, 0.0, 2.0]), (-1.0f64, [0.0, 0.0, -0.5])] {
        let y = eval(40, point);
        for (i, value) in y.iter().enumerate() {
            let (n, m) = layout.nm(i);
            let expected = if m == 0 {
                sign.powi(n as i32) * ((2 * n + 1) as f64 / (4.0 * PI)).sqrt()
            } else {
                0.0
            };
            assert!(
                (value - expected).abs() <= 1e-14 * expected.abs().max(1.0),
                "n {n} m {m}"
            );
        }
    }
}

/// Test 10: f32 against f64 at p ≤ 8, relative 1e-5 per degree.
#[test]
fn f32_agrees_with_f64_at_low_degree() {
    let p = 8;
    let len = Layout::new(p).len();
    let mut rng = SplitMix64::new(3);
    for _ in 0..20 {
        let x = rng.point_in_shell(0.3, 2.0);
        let reference = eval_derivatives(p, x);
        let xs = x.map(|c| c as f32);
        let (mut v, mut t, mut f) = (vec![0.0f32; len], vec![0.0f32; len], vec![0.0f32; len]);
        harmonics_derivatives(
            p,
            xs,
            &mut v,
            &mut t,
            &mut f,
            &mut DerivativeScratch::new(p),
        );
        for (single, double) in [v, t, f].iter().zip(&reference) {
            let widened: Vec<f64> = single.iter().map(|&s| s.to_f64()).collect();
            let error = degree_error(p, &widened, double);
            assert!(error <= 1e-5, "{error:e}");
        }
    }
}

/// Length checks.
#[test]
#[should_panic(expected = "must have length")]
fn wrong_length_panics() {
    let mut out = vec![0.0; 3];
    harmonics(1, [1.0, 0.0, 0.0], &mut out);
}

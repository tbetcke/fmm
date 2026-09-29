//! The separation identity and the addition theorem of CONVENTIONS §3.4.

use nd_fmm_math::harmonics::{irregular, regular};
use nd_fmm_math::{Layout, RealScalar};

use crate::common::{SplitMix64, complex_at, norm};

fn evaluate<T: RealScalar>(
    p: usize,
    x: [f64; 3],
    f: fn(usize, [T; 3], &mut [T]),
) -> ([f64; 3], Vec<f64>) {
    let x = x.map(T::from_f64);
    let mut out = vec![T::zero(); Layout::new(p).len()];
    f(p, x, &mut out);
    (
        x.map(|c| c.to_f64()),
        out.iter().map(|&v| v.to_f64()).collect(),
    )
}

/// Σₙ≤ₚ |y|ⁿ / |x|ⁿ⁺¹ Pₙ(cos γ), with cos γ = x·y / (|x| |y|): the Legendre expansion
/// of 1/|x − y| truncated after degree p, the same truncation as the harmonic sum.
fn truncated_legendre_series(p: usize, x: [f64; 3], y: [f64; 3]) -> f64 {
    let (rx, ry) = (norm(x), norm(y));
    let t = if ry == 0.0 {
        0.0
    } else {
        (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]) / (rx * ry)
    };
    let ratio = ry / rx;
    // (n + 1) Pₙ₊₁ = (2n + 1) t Pₙ − n Pₙ₋₁
    let (mut prev, mut cur) = (0.0, 1.0);
    let mut power = 1.0 / rx;
    let mut sum = 0.0;
    for n in 0..=p {
        sum += power * cur;
        let next = ((2 * n + 1) as f64 * t * cur - n as f64 * prev) / (n + 1) as f64;
        (prev, cur) = (cur, next);
        power *= ratio;
    }
    sum
}

/// Σₙ≤ₚ Σₘ conj(Rₙᵐ(y)) Iₙᵐ(x) from real storage. The terms of ±m are complex
/// conjugates of each other (§3.3), so each pair contributes 2 Re(conj(Rₙᵐ) Iₙᵐ).
fn harmonic_sum(p: usize, r: &[f64], i: &[f64]) -> f64 {
    let layout = Layout::new(p);
    let mut sum = 0.0;
    for n in 0..=p {
        sum += r[layout.idx(n, 0)] * i[layout.idx(n, 0)];
        for m in 1..=n as isize {
            let (re, im) = (layout.idx(n, m), layout.idx(n, -m));
            sum += 2.0 * (r[re] * i[re] + r[im] * i[im]);
        }
    }
    sum
}

/// Separation identity (§3.4) at `pairs` random pairs with |y|/|x| ≤ 0.5, truncated
/// at degree p.
///
/// Truncating the series leaves an error of up to (|y|/|x|)ᵖ⁺¹, about 4.7e-10 at
/// p = 30 and ratio 0.5, far above the 1e-12 the harmonics must reach. The harmonic
/// sum is therefore compared with the Legendre series truncated at the same degree,
/// which it equals exactly by the addition theorem for Legendre polynomials; the
/// Legendre series itself is checked against 1/|x − y| within its truncation bound.
fn check_separation<T: RealScalar>(seed: u64, p: usize, pairs: usize, tol: f64) {
    let mut rng = SplitMix64::new(seed);
    let mut worst = 0.0_f64;
    for _ in 0..pairs {
        let x = rng.point_in_shell(1.0, 4.0);
        let ratio = rng.range(0.0, 0.5);
        let y = rng.unit_vector().map(|c| ratio * norm(x) * c);
        // The harmonics are those of x and y rounded to T.
        let (x, i) = evaluate::<T>(p, x, irregular);
        let (y, r) = evaluate::<T>(p, y, regular);
        let ratio = norm(y) / norm(x);
        assert!(ratio <= 0.5 + 1e-6);

        let series = truncated_legendre_series(p, x, y);
        let diff = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
        let exact = 1.0 / norm(diff);
        let bound = ratio.powi(p as i32 + 1) / ((1.0 - ratio) * norm(x));
        assert!(
            (series - exact).abs() <= bound + 1e-14 * exact,
            "reference series off by {:e}, bound {bound:e}",
            (series - exact).abs()
        );

        let sum = harmonic_sum(p, &r, &i);
        worst = worst.max((sum - series).abs() / series.abs());
    }
    println!("separation, p = {p}: worst relative error {worst:.2e}");
    assert!(worst <= tol, "separation: {worst:e} > {tol:e}");
}

#[test]
fn separation_identity_f64() {
    check_separation::<f64>(0x5eba_4a7e, 30, 200, 1e-12);
}

/// Not required by the brief: the same identity in f32 in its range, p ≤ 8 (§3.9).
#[test]
fn separation_identity_f32() {
    check_separation::<f32>(0x5eba_4a7f, 8, 200, 1e-5);
}

/// Addition theorem (§3.4), Rₙᵐ(a + b) = Σₖ Σₗ Rₖˡ(a) Rₙ₋ₖᵐ⁻ˡ(b), for every n ≤ p at
/// `trials` random pairs with |a|, |b| ≤ √3/2, so that |a + b| stays in the range of
/// §3.9. Evaluating at degree p also covers every smaller degree: lower-degree values
/// do not depend on p.
///
/// The error of degree n is measured relative to max_m Σₖ Σₗ |Rₖˡ(a)| |Rₙ₋ₖᵐ⁻ˡ(b)|,
/// the scale of any rounding-error bound for the sum. Relative to Rₙᵐ(a + b) itself the
/// sum is ill-conditioned when a and b nearly cancel: the error then grows like
/// ε (|a + b| / (|a| + |b|))⁻ⁿ, reaching 2e-9 at n = 12 for random pairs, whatever the
/// accuracy of the harmonics.
fn check_addition<T: RealScalar>(seed: u64, p: usize, trials: usize, tol: f64) {
    let layout = Layout::new(p);
    let mut rng = SplitMix64::new(seed);
    let half = 0.5 * 3.0_f64.sqrt();
    let mut worst = 0.0_f64;
    for _ in 0..trials {
        let (a, ra) = evaluate::<T>(p, rng.point_in_shell(0.0, half), regular);
        let (b, rb) = evaluate::<T>(p, rng.point_in_shell(0.0, half), regular);
        // a + b is formed in f64; rounding it to T is part of the T error budget.
        let (_, rab) = evaluate::<T>(p, [a[0] + b[0], a[1] + b[1], a[2] + b[2]], regular);

        let mut expected = vec![0.0; layout.len()];
        let mut magnitude = vec![0.0; layout.len()];
        for n in 0..=p {
            for m in 0..=n as isize {
                let (mut re, mut im, mut abs_sum) = (0.0, 0.0, 0.0);
                for k in 0..=n {
                    for l in -(k as isize)..=k as isize {
                        let (ar, ai) = complex_at(&ra, k, l);
                        let (br, bi) = complex_at(&rb, n - k, m - l);
                        re += ar * br - ai * bi;
                        im += ar * bi + ai * br;
                        abs_sum += ar.hypot(ai) * br.hypot(bi);
                    }
                }
                expected[layout.idx(n, m)] = re;
                magnitude[layout.idx(n, m)] = abs_sum;
                if m > 0 {
                    expected[layout.idx(n, -m)] = im;
                }
            }
        }
        for (_, range) in layout.degrees() {
            let scale = magnitude[range.clone()]
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v));
            let err = rab[range.clone()]
                .iter()
                .zip(&expected[range])
                .fold(0.0_f64, |a, (c, e)| a.max((c - e).abs()));
            worst = worst.max(err / scale);
        }
    }
    println!("addition theorem, p = {p}: worst relative error {worst:.2e}");
    assert!(worst <= tol, "addition theorem: {worst:e} > {tol:e}");
}

#[test]
fn addition_theorem_f64() {
    check_addition::<f64>(0xadd1_7104, 12, 100, 1e-13);
}

/// Not required by the brief: the same identity in f32, p ≤ 8 (§3.9).
#[test]
fn addition_theorem_f32() {
    check_addition::<f32>(0xadd1_7105, 8, 100, 1e-5);
}

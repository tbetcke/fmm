//! Regular and irregular solid harmonics and their gradients (CONVENTIONS §3.3–§3.6).
//!
//! Every function evaluates all degrees 0 to p at one point and writes them into caller
//! slices of length (p + 1)² in the real storage of CONVENTIONS §3.6 (see [`Layout`]):
//! slot m = 0 holds the real value, slot +m the real part and slot −m the imaginary
//! part of the order-m harmonic. Gradients use the same layout for each Cartesian
//! component; since ∂x, ∂y and ∂z are real operators, slot +m of ∂x holds ∂x Re Xₙᵐ
//! and slot −m holds ∂x Im Xₙᵐ.
//!
//! Evaluation uses the Cartesian recursions of CONVENTIONS §3.5 on real and imaginary
//! parts separately: no trigonometric functions, no complex type and no allocation.
//! The tested ranges are those of CONVENTIONS §3.9: |x| ≤ √3 for regular and |x| ≥ 2
//! for irregular harmonics, f64 up to p = 30 and f32 up to p = 8.
//!
//! # Panics
//!
//! Every function panics unless each output slice has length (p + 1)².

use crate::{Layout, RealScalar};

/// The integer `k` as a `T`; exact for every k that occurs here.
#[inline(always)]
fn int<T: RealScalar>(k: usize) -> T {
    T::from_f64(k as f64)
}

/// Writes the complex value `(re, im)` of order m ≥ 0 at degree n into real storage.
/// For m = 0 only the real part is stored; it is the whole value there.
#[inline(always)]
fn store<T: RealScalar>(out: &mut [T], layout: Layout, n: usize, m: usize, (re, im): (T, T)) {
    out[layout.idx(n, m as isize)] = re;
    if m > 0 {
        out[layout.idx(n, -(m as isize))] = im;
    }
}

/// Reads the complex value of order m ≥ 0 at degree n from real storage.
#[inline(always)]
fn load<T: RealScalar>(values: &[T], layout: Layout, n: usize, m: usize) -> (T, T) {
    let re = values[layout.idx(n, m as isize)];
    if m == 0 {
        (re, T::zero())
    } else {
        (re, values[layout.idx(n, -(m as isize))])
    }
}

/// Applies the conjugate symmetry Xₙ⁻ᵏ = (−1)ᵏ conj(Xₙᵏ) of CONVENTIONS §3.3 to the
/// value `(re, im)` of order k ≥ 0.
#[inline(always)]
fn negate_order<T: RealScalar>(k: usize, (re, im): (T, T)) -> (T, T) {
    if k.is_multiple_of(2) {
        (re, -im)
    } else {
        (-re, im)
    }
}

/// Complex value of Xₙᵐ for any integer m, from real storage; zero for |m| > n.
#[inline(always)]
fn value_at<T: RealScalar>(values: &[T], layout: Layout, n: usize, m: isize) -> (T, T) {
    let k = m.unsigned_abs();
    if k > n {
        return (T::zero(), T::zero());
    }
    let v = load(values, layout, n, k);
    if m < 0 { negate_order(k, v) } else { v }
}

/// Writes c₁ Xₙ₋₁ᵐ + c₂ Xₙ₋₂ᵐ to Xₙᵐ, real and imaginary parts alike.
#[inline(always)]
fn three_term<T: RealScalar>(out: &mut [T], layout: Layout, n: usize, m: usize, c1: T, c2: T) {
    let (r1, i1) = load(out, layout, n - 1, m);
    let (r2, i2) = load(out, layout, n - 2, m);
    store(out, layout, n, m, (c1 * r1 + c2 * r2, c1 * i1 + c2 * i2));
}

/// ρ · (re, im) with ρ = x + iy, scaled by `s`.
#[inline(always)]
fn times_rho<T: RealScalar>(x: T, y: T, s: T, (re, im): (T, T)) -> (T, T) {
    ((x * re - y * im) * s, (x * im + y * re) * s)
}

/// Checks the length of an output slice.
fn check_len<T>(out: &[T], layout: Layout, name: &str) {
    assert_eq!(
        out.len(),
        layout.len(),
        "`{name}` must have length (p + 1)^2 = {} for p = {}",
        layout.len(),
        layout.p()
    );
}

/// Regular solid harmonics Rₙᵐ(x), n ≤ p (CONVENTIONS §3.3), written to `out` in real
/// storage (§3.6).
///
/// Uses the recursion of CONVENTIONS §3.5, with ρ = x + iy and r² = |x|²:
/// R₀⁰ = 1, Rₘᵐ = ρ Rₘ₋₁ᵐ⁻¹ / (2m), Rₘ₊₁ᵐ = z Rₘᵐ and
/// Rₙᵐ = ((2n − 1) z Rₙ₋₁ᵐ − r² Rₙ₋₂ᵐ) / ((n + m)(n − m)).
/// Values of degree n do not depend on p.
///
/// ```
/// use nd_fmm_math::{Layout, harmonics};
///
/// let (x, y, z): (f64, f64, f64) = (0.3, -0.2, 0.5);
/// let layout = Layout::new(2);
/// let mut r = vec![0.0; layout.len()];
/// harmonics::regular(2, [x, y, z], &mut r);
/// // R₁¹ = (x + iy)/2 and R₂⁰ = (2z² − x² − y²)/4 (CONVENTIONS §3.3)
/// assert_eq!((r[layout.idx(1, 1)], r[layout.idx(1, -1)]), (x / 2.0, y / 2.0));
/// assert!((r[layout.idx(2, 0)] - (2.0 * z * z - x * x - y * y) / 4.0).abs() < 1e-15);
/// ```
///
/// # Panics
///
/// If `out.len() != (p + 1)²`.
pub fn regular<T: RealScalar>(p: usize, x: [T; 3], out: &mut [T]) {
    let layout = Layout::new(p);
    check_len(out, layout, "out");
    let [x, y, z] = x;
    let r2 = x * x + y * y + z * z;
    let mut diagonal = (T::one(), T::zero());
    for m in 0..=p {
        if m > 0 {
            diagonal = times_rho(x, y, T::one() / int(2 * m), diagonal);
        }
        store(out, layout, m, m, diagonal);
        if m == p {
            break;
        }
        store(out, layout, m + 1, m, (z * diagonal.0, z * diagonal.1));
        for n in m + 2..=p {
            let denominator = int::<T>((n + m) * (n - m));
            let c1 = int::<T>(2 * n - 1) * z / denominator;
            let c2 = -r2 / denominator;
            three_term(out, layout, n, m, c1, c2);
        }
    }
}

/// Irregular solid harmonics Iₙᵐ(x), n ≤ p (CONVENTIONS §3.3), written to `out` in real
/// storage (§3.6).
///
/// Uses the recursion of CONVENTIONS §3.5, with ρ = x + iy and r² = |x|²:
/// I₀⁰ = 1/r, Iₘᵐ = (2m − 1) ρ Iₘ₋₁ᵐ⁻¹ / r², Iₘ₊₁ᵐ = (2m + 1) z Iₘᵐ / r² and
/// Iₙᵐ = ((2n − 1) z Iₙ₋₁ᵐ − ((n − 1)² − m²) Iₙ₋₂ᵐ) / r².
/// Values of degree n do not depend on p. At the origin the values are not finite.
///
/// ```
/// use nd_fmm_math::{Layout, harmonics};
///
/// let layout = Layout::new(1);
/// let mut i = vec![0.0; layout.len()];
/// harmonics::irregular(1, [0.0, 0.0, 2.0], &mut i);
/// // I₀⁰ = 1/r and I₁⁰ = z/r³ (CONVENTIONS §3.3)
/// assert_eq!(i, [0.5, 0.0, 0.25, 0.0]);
/// ```
///
/// # Panics
///
/// If `out.len() != (p + 1)²`.
pub fn irregular<T: RealScalar>(p: usize, x: [T; 3], out: &mut [T]) {
    let layout = Layout::new(p);
    check_len(out, layout, "out");
    let [x, y, z] = x;
    let r2 = x * x + y * y + z * z;
    let inv_r2 = T::one() / r2;
    let mut diagonal = (T::one() / r2.sqrt(), T::zero());
    for m in 0..=p {
        if m > 0 {
            diagonal = times_rho(x, y, int::<T>(2 * m - 1) * inv_r2, diagonal);
        }
        store(out, layout, m, m, diagonal);
        if m == p {
            break;
        }
        let c = int::<T>(2 * m + 1) * z * inv_r2;
        store(out, layout, m + 1, m, (c * diagonal.0, c * diagonal.1));
        for n in m + 2..=p {
            let c1 = int::<T>(2 * n - 1) * z * inv_r2;
            let c2 = -int::<T>((n - 1 - m) * (n - 1 + m)) * inv_r2;
            three_term(out, layout, n, m, c1, c2);
        }
    }
}

/// Writes the gradient of Xₙᵐ, m ≥ 0, given `[A, B, C]` = [Xₛᵐ⁻¹, Xₛᵐ⁺¹, Xₛᵐ] of the
/// neighbouring degree s: from (∂x − i∂y) X = A, (∂x + i∂y) X = −B and ∂z X = σ C,
/// ∂x X = (A − B)/2, ∂y X = i (A + B)/2 and ∂z X = σ C.
#[inline(always)]
fn store_ladder<T: RealScalar>(
    [gx, gy, gz]: &mut [&mut [T]; 3],
    layout: Layout,
    (n, m): (usize, usize),
    [(ar, ai), (br, bi), (cr, ci)]: [(T, T); 3],
    sigma: T,
) {
    let half = T::from_f64(0.5);
    store(gx, layout, n, m, ((ar - br) * half, (ai - bi) * half));
    store(gy, layout, n, m, (-(ai + bi) * half, (ar + br) * half));
    store(gz, layout, n, m, (sigma * cr, sigma * ci));
}

/// Checks the lengths of the value and gradient slices.
fn check_grad_len<T>(value: &[T], grad: &[&mut [T]; 3], layout: Layout) {
    check_len(value, layout, "value");
    for (g, name) in grad.iter().zip(["grad[0]", "grad[1]", "grad[2]"]) {
        check_len(g, layout, name);
    }
}

/// Regular solid harmonics Rₙᵐ(x), n ≤ p, and their gradient: values as in
/// [`regular`] into `value`, and ∂x, ∂y, ∂z into `grad[0]`, `grad[1]`, `grad[2]`, all
/// in real storage (CONVENTIONS §3.6).
///
/// Uses the gradient ladder of CONVENTIONS §3.4, ∂z Rₙᵐ = Rₙ₋₁ᵐ,
/// (∂x − i∂y) Rₙᵐ = Rₙ₋₁ᵐ⁻¹ and (∂x + i∂y) Rₙᵐ = −Rₙ₋₁ᵐ⁺¹, hence
///
/// ∂x Rₙᵐ = (Rₙ₋₁ᵐ⁻¹ − Rₙ₋₁ᵐ⁺¹) / 2,  ∂y Rₙᵐ = i (Rₙ₋₁ᵐ⁻¹ + Rₙ₋₁ᵐ⁺¹) / 2,
///
/// with Rₙ₋₁⁻¹ = −conj(Rₙ₋₁¹) (§3.3) and Rⱼⁱ = 0 for |i| > j. The values are
/// computed alongside because the gradient of degree n is built from degree n − 1.
///
/// # Panics
///
/// If `value` or any `grad` component does not have length (p + 1)².
pub fn regular_grad<T: RealScalar>(p: usize, x: [T; 3], value: &mut [T], grad: [&mut [T]; 3]) {
    let layout = Layout::new(p);
    let mut grad = grad;
    check_grad_len(value, &grad, layout);
    regular(p, x, value);
    let zero = (T::zero(), T::zero());
    store_ladder(&mut grad, layout, (0, 0), [zero; 3], T::one());
    for n in 1..=p {
        for m in 0..=n {
            let k = m as isize;
            let a = value_at(value, layout, n - 1, k - 1);
            let b = value_at(value, layout, n - 1, k + 1);
            let c = value_at(value, layout, n - 1, k);
            store_ladder(&mut grad, layout, (n, m), [a, b, c], T::one());
        }
    }
}

/// Complex value of Iₚ₊₁ᵏ for any integer k, from the stored degrees p and p − 1 of
/// `values`, by the recursion of CONVENTIONS §3.5; zero for |k| > p + 1. The
/// operations are those of [`irregular`] at degree p + 1, in the same order.
fn irregular_above<T: RealScalar>(
    values: &[T],
    layout: Layout,
    k: isize,
    [x, y, z]: [T; 3],
    inv_r2: T,
) -> (T, T) {
    let p = layout.p();
    let j = k.unsigned_abs();
    if j > p + 1 {
        return (T::zero(), T::zero());
    }
    let v = if j == p + 1 {
        // diagonal: Iₚ₊₁ᵖ⁺¹ = (2p + 1) ρ Iₚᵖ / r²
        times_rho(
            x,
            y,
            int::<T>(2 * p + 1) * inv_r2,
            load(values, layout, p, p),
        )
    } else if j == p {
        // Iₚ₊₁ᵖ = (2p + 1) z Iₚᵖ / r²
        let c = int::<T>(2 * p + 1) * z * inv_r2;
        let (re, im) = load(values, layout, p, p);
        (c * re, c * im)
    } else {
        // Iₚ₊₁ʲ = ((2p + 1) z Iₚʲ − (p² − j²) Iₚ₋₁ʲ) / r²
        let c1 = int::<T>(2 * p + 1) * z * inv_r2;
        let c2 = -int::<T>((p - j) * (p + j)) * inv_r2;
        let (r1, i1) = load(values, layout, p, j);
        let (r2, i2) = load(values, layout, p - 1, j);
        (c1 * r1 + c2 * r2, c1 * i1 + c2 * i2)
    };
    if k < 0 { negate_order(j, v) } else { v }
}

/// Irregular solid harmonics Iₙᵐ(x), n ≤ p, and their gradient: values as in
/// [`irregular`] into `value`, and ∂x, ∂y, ∂z into `grad[0]`, `grad[1]`, `grad[2]`,
/// all in real storage (CONVENTIONS §3.6).
///
/// # Gradient ladder
///
/// CONVENTIONS §3.4 states the ladder only for regular harmonics. The irregular one is
///
/// ∂z Iₙᵐ = −Iₙ₊₁ᵐ,  (∂x − i∂y) Iₙᵐ = Iₙ₊₁ᵐ⁻¹,  (∂x + i∂y) Iₙᵐ = −Iₙ₊₁ᵐ⁺¹,
///
/// hence ∂x Iₙᵐ = (Iₙ₊₁ᵐ⁻¹ − Iₙ₊₁ᵐ⁺¹) / 2 and ∂y Iₙᵐ = i (Iₙ₊₁ᵐ⁻¹ + Iₙ₊₁ᵐ⁺¹) / 2,
/// with Iₙ₊₁⁻¹ = −conj(Iₙ₊₁¹) (§3.3), valid for all |m| ≤ n.
///
/// *Derivation.* It follows from the regular ladder through the separation identity
/// of §3.4,
///
/// 1/|x − y| = Σₖ Σₗ conj(Rₖˡ(y)) Iₖˡ(x),  |y| < |x|.
///
/// The series converges absolutely and uniformly, with all derivatives, on
/// |y| ≤ c |x| for c < 1, so it may be differentiated term by term. Write
/// D± = ∂x ± i∂y. The left-hand side depends on x − y only, so every derivative in x
/// equals minus the same derivative in y. Since the Cartesian derivatives are real,
/// D∓ conj(f) = conj(D± f).
///
/// - ∂z: Σ conj(Rₖˡ(y)) ∂z Iₖˡ(x) = −Σ conj(∂z Rₖˡ(y)) Iₖˡ(x)
///   = −Σ conj(Rₖ₋₁ˡ(y)) Iₖˡ(x) = −Σₙ Σₘ conj(Rₙᵐ(y)) Iₙ₊₁ᵐ(x).
/// - D−: Σ conj(Rₖˡ(y)) D− Iₖˡ(x) = −Σ conj(D+ Rₖˡ(y)) Iₖˡ(x)
///   = Σ conj(Rₖ₋₁ˡ⁺¹(y)) Iₖˡ(x) = Σₙ Σₘ conj(Rₙᵐ(y)) Iₙ₊₁ᵐ⁻¹(x).
/// - D+: Σ conj(Rₖˡ(y)) D+ Iₖˡ(x) = −Σ conj(D− Rₖˡ(y)) Iₖˡ(x)
///   = −Σ conj(Rₖ₋₁ˡ⁻¹(y)) Iₖˡ(x) = −Σₙ Σₘ conj(Rₙᵐ(y)) Iₙ₊₁ᵐ⁺¹(x).
///
/// Terms with Rⱼⁱ = 0 for |i| > j drop out, and the reindexed sums run over all
/// |m| ≤ n. For each n the conj(Rₙᵐ), −n ≤ m ≤ n, are linearly independent
/// homogeneous polynomials of degree n in y, so the coefficients on both sides agree,
/// which is the ladder above. The mpmath fixtures of tools/fixtures/ confirm it
/// independently, from central differences of the definitions in §3.3.
///
/// The gradient of degree p needs Iₚ₊₁; its values are formed from degrees p and p − 1
/// by the recursion of §3.5 on the fly, so no storage beyond (p + 1)² is needed.
///
/// # Panics
///
/// If `value` or any `grad` component does not have length (p + 1)².
pub fn irregular_grad<T: RealScalar>(p: usize, x: [T; 3], value: &mut [T], grad: [&mut [T]; 3]) {
    let layout = Layout::new(p);
    let mut grad = grad;
    check_grad_len(value, &grad, layout);
    irregular(p, x, value);
    let minus_one = -T::one();
    for n in 0..p {
        for m in 0..=n {
            let k = m as isize;
            let a = value_at(value, layout, n + 1, k - 1);
            let b = value_at(value, layout, n + 1, k + 1);
            let c = value_at(value, layout, n + 1, k);
            store_ladder(&mut grad, layout, (n, m), [a, b, c], minus_one);
        }
    }
    let [x0, x1, x2] = x;
    let inv_r2 = T::one() / (x0 * x0 + x1 * x1 + x2 * x2);
    for m in 0..=p {
        let k = m as isize;
        let a = irregular_above(value, layout, k - 1, x, inv_r2);
        let b = irregular_above(value, layout, k + 1, x, inv_r2);
        let c = irregular_above(value, layout, k, x, inv_r2);
        store_ladder(&mut grad, layout, (p, m), [a, b, c], minus_one);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: [f64; 3] = [0.3, -0.7, 0.45];

    fn eval(p: usize, f: fn(usize, [f64; 3], &mut [f64])) -> Vec<f64> {
        let mut out = vec![f64::NAN; Layout::new(p).len()];
        f(p, X, &mut out);
        out
    }

    #[test]
    fn low_degree_examples_of_conventions() {
        // CONVENTIONS §3.3: R₁⁰ = z, R₁¹ = (x + iy)/2, R₁⁻¹ = −(x − iy)/2,
        // R₂⁰ = (2z² − x² − y²)/4.
        let [x, y, z] = X;
        let layout = Layout::new(2);
        let r = eval(2, regular);
        assert_eq!(r[layout.idx(0, 0)], 1.0);
        assert_eq!(r[layout.idx(1, 0)], z);
        assert_eq!(value_at(&r, layout, 1, 1), (x / 2.0, y / 2.0));
        assert_eq!(value_at(&r, layout, 1, -1), (-x / 2.0, y / 2.0));
        let r20 = (2.0 * z * z - x * x - y * y) / 4.0;
        assert!((r[layout.idx(2, 0)] - r20).abs() < 1e-16);

        // Iₙᵐ = (n − m)! (n + m)! Rₙᵐ / r²ⁿ⁺¹ (CONVENTIONS §3.8) at n ≤ 1.
        let r2: f64 = X.iter().map(|c| c * c).sum();
        let i = eval(2, irregular);
        let rr = r2.sqrt();
        assert!((i[0] - 1.0 / rr).abs() < 1e-15);
        assert!((i[layout.idx(1, 0)] - z / (rr * r2)).abs() < 1e-15);
        assert!((i[layout.idx(1, 1)] - x / (rr * r2)).abs() < 1e-15);
        assert!((i[layout.idx(1, -1)] - y / (rr * r2)).abs() < 1e-15);
    }

    #[test]
    fn lower_degrees_do_not_depend_on_p() {
        for f in [regular, irregular] {
            let full = eval(12, f);
            for p in 0..12 {
                assert_eq!(eval(p, f), full[..Layout::new(p).len()]);
            }
        }
    }

    #[test]
    fn gradients_do_not_depend_on_p() {
        type GradFn = fn(usize, [f64; 3], &mut [f64], [&mut [f64]; 3]);
        let grads = |p: usize, f: GradFn| {
            let len = Layout::new(p).len();
            let mut value = vec![0.0; len];
            let [mut gx, mut gy, mut gz] = [0; 3].map(|_| vec![0.0; len]);
            f(p, X, &mut value, [&mut gx, &mut gy, &mut gz]);
            [gx, gy, gz]
        };
        for f in [regular_grad as GradFn, irregular_grad] {
            let full = grads(10, f);
            for p in 0..10 {
                let len = Layout::new(p).len();
                for (g, reference) in grads(p, f).iter().zip(&full) {
                    // Degree p of the irregular gradient takes Iₚ₊₁ from the on-the-fly
                    // recursion rather than from storage; same operations, same result.
                    assert_eq!(g[..], reference[..len]);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "must have length (p + 1)^2")]
    fn wrong_output_length_panics() {
        regular(3, X, &mut [0.0; 15]);
    }
}

//! Error metrics: relative L2 and max errors, errors relative to the sum of term
//! magnitudes, and per-degree coefficient errors in the orthonormal weighting of
//! CONVENTIONS §3.8.
//!
//! # Which measure to use
//!
//! These are the error measures of docs/phase1/README.md ("Error measures") and design
//! §8. Every reported figure and every test names the one it uses.
//!
//! - **Relative L2 and max error** ([`potential_errors`], [`gradient_errors`],
//!   [`ErrorAccumulator`]): for reported accuracy of potentials and gradients against
//!   the f64 direct sum, as in design §8.2 and the acceptance criteria of C3.2 and C3.4.
//!   They are relative to the size of the exact result, ‖φ*‖₂ and max |φ*|, so they
//!   compare runs, precisions and distributions on one scale. They are the wrong
//!   measure where the exact result is small because its terms cancel: there a tiny
//!   rounding error looks large.
//! - **Error relative to the sum of term magnitudes** ([`potential_magnitudes`],
//!   [`gradient_magnitudes`], [`potential_error_relative_to_magnitude`],
//!   [`gradient_error_relative_to_magnitude`]): per target,
//!   |φ − φ*| / Σⱼ |qⱼ| / |x − yⱼ| (and |∇φ − ∇φ*| / Σⱼ |qⱼ| / |x − yⱼ|² for gradients).
//!   Use it wherever terms can cancel (charges of both signs, a target where the
//!   potential nearly vanishes), and whenever a tolerance is set near the rounding
//!   level: rounding error scales with the magnitudes of the terms, not of their sum.
//! - **Per-degree coefficient error** ([`degree_norms`], [`degree_errors`],
//!   [`max_relative_degree_error`]): for comparing coefficient vectors, e.g. one
//!   translation method against another, or a truncated expansion against a reference
//!   truncated at the same degree (the truncation error is measured separately against
//!   its bound). Slot m of degree n is weighted by Nₘ for multipole-type data
//!   (conjugates of regular harmonics) and by Nₘ/Sₘ for local-type data (conjugates of
//!   irregular harmonics), [`CoefficientKind`], so that every slot of a degree is on
//!   the same scale; raw storage mixes slots whose sizes differ by factorials.
//!
//! ```
//! use nd_fmm_validate::metrics::potential_errors;
//!
//! let exact = [3.0, 4.0];
//! let errors = potential_errors(&[3.0, 4.5], &exact);
//! assert!((errors.l2 - 0.1).abs() < 1e-15); // ‖(0, 0.5)‖ / ‖(3, 4)‖
//! assert!((errors.max - 0.125).abs() < 1e-15); // 0.5 / 4
//! ```

use nd_fmm_math::{Layout, RealScalar};

/// Relative L2 and max error of a set of values (potentials) or 3-vectors (gradients)
/// against exact ones.
///
/// For values, `l2` = ‖a − e‖₂ / ‖e‖₂ and `max` = maxᵢ |aᵢ − eᵢ| / maxᵢ |eᵢ|. For
/// 3-vectors, |·| is the Euclidean norm of each vector. A zero reference gives 0 for a
/// zero error and +∞ otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErrorNorms {
    /// Relative L2 error ‖a − e‖₂ / ‖e‖₂.
    pub l2: f64,
    /// Relative max error maxᵢ |aᵢ − eᵢ| / maxᵢ |eᵢ|.
    pub max: f64,
}

/// Accumulates relative L2 and max errors over several batches of targets, as if they
/// were one set: the sums of squares and the maxima of error and reference are
/// accumulated, and [`finish`](Self::finish) forms the ratios of [`ErrorNorms`].
///
/// ```
/// use nd_fmm_validate::metrics::{ErrorAccumulator, potential_errors};
///
/// let mut acc = ErrorAccumulator::new();
/// acc.add_values(&[3.0_f32], &[3.0]);
/// acc.add_values(&[4.5_f32], &[4.0]);
/// assert_eq!(acc.finish(), potential_errors(&[3.0, 4.5], &[3.0, 4.0]));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErrorAccumulator {
    error_sq: f64,
    reference_sq: f64,
    error_max: f64,
    reference_max: f64,
}

impl ErrorAccumulator {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds values `approx` (any precision) against exact f64 values `exact`.
    ///
    /// # Panics
    ///
    /// If the slices differ in length.
    pub fn add_values<T: RealScalar>(&mut self, approx: &[T], exact: &[f64]) {
        assert_eq!(
            approx.len(),
            exact.len(),
            "approx and exact differ in length"
        );
        for (&a, &e) in approx.iter().zip(exact) {
            self.add_one((RealScalar::to_f64(a) - e).abs(), e.abs());
        }
    }

    /// Adds 3-vectors `approx` (any precision) against exact f64 vectors `exact`,
    /// measuring each by its Euclidean norm.
    ///
    /// # Panics
    ///
    /// If the slices differ in length.
    pub fn add_vectors<T: RealScalar>(&mut self, approx: &[[T; 3]], exact: &[[f64; 3]]) {
        assert_eq!(
            approx.len(),
            exact.len(),
            "approx and exact differ in length"
        );
        for (a, e) in approx.iter().zip(exact) {
            let diff: [f64; 3] = core::array::from_fn(|i| RealScalar::to_f64(a[i]) - e[i]);
            self.add_one(norm(diff), norm(*e));
        }
    }

    fn add_one(&mut self, error: f64, reference: f64) {
        self.error_sq += error * error;
        self.reference_sq += reference * reference;
        self.error_max = self.error_max.max(error);
        self.reference_max = self.reference_max.max(reference);
    }

    /// The relative L2 and max errors of everything added so far.
    pub fn finish(&self) -> ErrorNorms {
        ErrorNorms {
            l2: ratio(self.error_sq.sqrt(), self.reference_sq.sqrt()),
            max: ratio(self.error_max, self.reference_max),
        }
    }
}

/// Relative L2 and max error of potentials `approx` against the exact `exact`
/// (see [`ErrorNorms`]).
///
/// # Panics
///
/// If the slices differ in length.
pub fn potential_errors<T: RealScalar>(approx: &[T], exact: &[f64]) -> ErrorNorms {
    let mut acc = ErrorAccumulator::new();
    acc.add_values(approx, exact);
    acc.finish()
}

/// Relative L2 and max error of gradients `approx` against the exact `exact`,
/// measuring each gradient by its Euclidean norm (see [`ErrorNorms`]).
///
/// # Panics
///
/// If the slices differ in length.
pub fn gradient_errors<T: RealScalar>(approx: &[[T; 3]], exact: &[[f64; 3]]) -> ErrorNorms {
    let mut acc = ErrorAccumulator::new();
    acc.add_vectors(approx, exact);
    acc.finish()
}

/// The sum of term magnitudes of the potential at each target,
/// Σⱼ |qⱼ| / |xᵢ − yⱼ|, skipping pairs with xᵢ == yⱼ as `nd_fmm_ref::p2p` does.
///
/// # Panics
///
/// If `sources` and `charges` differ in length.
pub fn potential_magnitudes(
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
) -> Vec<f64> {
    magnitudes(sources, charges, targets, |r| 1.0 / r)
}

/// The sum of term magnitudes of the gradient at each target,
/// Σⱼ |qⱼ| / |xᵢ − yⱼ|², skipping pairs with xᵢ == yⱼ as `nd_fmm_ref::p2p` does.
///
/// # Panics
///
/// If `sources` and `charges` differ in length.
pub fn gradient_magnitudes(
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
) -> Vec<f64> {
    magnitudes(sources, charges, targets, |r| 1.0 / (r * r))
}

fn magnitudes(
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
    term: impl Fn(f64) -> f64,
) -> Vec<f64> {
    assert_eq!(
        sources.len(),
        charges.len(),
        "sources and charges differ in length"
    );
    targets
        .iter()
        .map(|x| {
            sources
                .iter()
                .zip(charges)
                .filter(|(y, _)| *y != x)
                .map(|(y, q)| q.abs() * term(norm(core::array::from_fn(|i| x[i] - y[i]))))
                .sum()
        })
        .collect()
}

/// The largest error of potentials relative to the sum of term magnitudes,
/// maxᵢ |aᵢ − eᵢ| / magnitudeᵢ, with the magnitudes of [`potential_magnitudes`].
///
/// A target with zero magnitude contributes 0 if its error is 0 and +∞ otherwise.
///
/// ```
/// use nd_fmm_validate::metrics::{potential_error_relative_to_magnitude, potential_magnitudes};
///
/// // Two opposite charges: the potential at the midpoint cancels to 0, but its terms
/// // have magnitude 2.
/// let sources = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
/// let magnitude = potential_magnitudes(&sources, &[1.0, -1.0], &[[0.0; 3]]);
/// assert_eq!(magnitude, [2.0]);
/// let error = potential_error_relative_to_magnitude(&[1e-16], &[0.0], &magnitude);
/// assert_eq!(error, 5e-17);
/// ```
///
/// # Panics
///
/// If the slices differ in length.
pub fn potential_error_relative_to_magnitude<T: RealScalar>(
    approx: &[T],
    exact: &[f64],
    magnitude: &[f64],
) -> f64 {
    relative_to_magnitude(approx.len(), exact.len(), magnitude, |i| {
        (RealScalar::to_f64(approx[i]) - exact[i]).abs()
    })
}

/// The largest error of gradients relative to the sum of term magnitudes,
/// maxᵢ |aᵢ − eᵢ| / magnitudeᵢ with |·| the Euclidean norm, with the magnitudes of
/// [`gradient_magnitudes`].
///
/// A target with zero magnitude contributes 0 if its error is 0 and +∞ otherwise.
///
/// # Panics
///
/// If the slices differ in length.
pub fn gradient_error_relative_to_magnitude<T: RealScalar>(
    approx: &[[T; 3]],
    exact: &[[f64; 3]],
    magnitude: &[f64],
) -> f64 {
    relative_to_magnitude(approx.len(), exact.len(), magnitude, |i| {
        norm(core::array::from_fn(|k| {
            RealScalar::to_f64(approx[i][k]) - exact[i][k]
        }))
    })
}

fn relative_to_magnitude(
    approx: usize,
    exact: usize,
    magnitude: &[f64],
    error: impl Fn(usize) -> f64,
) -> f64 {
    assert_eq!(approx, exact, "approx and exact differ in length");
    assert_eq!(
        approx,
        magnitude.len(),
        "approx and magnitude differ in length"
    );
    (0..approx)
        .map(|i| ratio(error(i), magnitude[i]))
        .fold(0.0, f64::max)
}

/// Which weight of the orthonormal basis of CONVENTIONS §3.8 a coefficient vector
/// takes (docs/phase1/README.md, "Error measures").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoefficientKind {
    /// Multipole-type data, conjugates of regular harmonics (M̃, and R itself):
    /// weight Nₘ = √((n + |m|)! (n − |m|)!) cₘ.
    Multipole,
    /// Local-type data, conjugates of irregular harmonics (L̃, and I itself):
    /// weight Nₘ/Sₘ = cₘ / √((n + |m|)! (n − |m|)!).
    Local,
}

/// The weight of slot m of degree n in the orthonormal basis of CONVENTIONS §3.8: Nₘ
/// for [`CoefficientKind::Multipole`] and Nₘ/Sₘ for [`CoefficientKind::Local`], with
/// c₀ = 1 and cₘ = √2 for m ≠ 0.
///
/// With these weights, and Σₘ (n − |m|)!/(n + |m|)! (Pₙᵐ)² = 1, the weighted norm of
/// degree n of the regular harmonics Rₙ(u) is |u|ⁿ, and that of the irregular
/// harmonics Iₙ(u) is 1/|u|ⁿ⁺¹. Factorials are formed in f64, so n ≤ 85.
///
/// # Panics
///
/// If |m| > n (in debug builds, as an underflow).
pub fn weight(kind: CoefficientKind, n: usize, m: isize) -> f64 {
    let k = m.unsigned_abs();
    let c = if k == 0 {
        1.0
    } else {
        core::f64::consts::SQRT_2
    };
    let f = (factorial(n + k) * factorial(n - k)).sqrt();
    match kind {
        CoefficientKind::Multipole => f * c,
        CoefficientKind::Local => c / f,
    }
}

/// The weighted norm of each degree of the coefficient vector `x` of degree ≤ p in the
/// real storage of CONVENTIONS §3.6: ‖W xₙ‖₂ for n = 0..=p, with the weights of
/// [`weight`].
///
/// # Panics
///
/// If `x.len() != (p + 1)²`.
pub fn degree_norms<T: RealScalar>(kind: CoefficientKind, p: usize, x: &[T]) -> Vec<f64> {
    let layout = Layout::new(p);
    assert_eq!(
        x.len(),
        layout.len(),
        "coefficients must have length (p + 1)²"
    );
    layout
        .degrees()
        .map(|(n, range)| {
            range
                .map(|i| (weight(kind, n, layout.nm(i).1) * RealScalar::to_f64(x[i])).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

/// The weighted norm of each degree of the difference `got − want`, n = 0..=p (see
/// [`degree_norms`]). This is an absolute error; divide it by a per-degree scale, as
/// [`max_relative_degree_error`] does.
///
/// # Panics
///
/// If either vector does not have length (p + 1)².
pub fn degree_errors<T: RealScalar>(
    kind: CoefficientKind,
    p: usize,
    got: &[T],
    want: &[f64],
) -> Vec<f64> {
    assert_eq!(got.len(), want.len(), "got and want differ in length");
    let diff: Vec<f64> = got
        .iter()
        .zip(want)
        .map(|(&a, b)| RealScalar::to_f64(a) - b)
        .collect();
    degree_norms(kind, p, &diff)
}

/// The worst per-degree error relative to a per-degree scale:
/// maxₙ ‖W (got − want)ₙ‖₂ / scaleₙ.
///
/// The scale is the size the error is measured against: the weighted norms of `want`
/// ([`degree_norms`]) where no cancellation occurs, or, where it can, the magnitudes of
/// the terms that form each degree. A degree with zero scale contributes 0 if its error
/// is 0 and +∞ otherwise.
///
/// # Panics
///
/// If either vector does not have length (p + 1)², or `scale.len() != p + 1`.
pub fn max_relative_degree_error<T: RealScalar>(
    kind: CoefficientKind,
    p: usize,
    got: &[T],
    want: &[f64],
    scale: &[f64],
) -> f64 {
    assert_eq!(scale.len(), p + 1, "scale must have length p + 1");
    degree_errors(kind, p, got, want)
        .iter()
        .zip(scale)
        .map(|(&e, &s)| ratio(e, s))
        .fold(0.0, f64::max)
}

/// e / s, with 0/0 = 0 and e/0 = +∞ for e > 0.
fn ratio(e: f64, s: f64) -> f64 {
    if s > 0.0 {
        e / s
    } else if e == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|j| j as f64).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SplitMix64, points};
    use nd_fmm_math::harmonics;

    #[test]
    fn zero_error() {
        // Error measure: exact equality with 0.
        let exact = [1.0, -2.0, 0.5];
        assert_eq!(potential_errors(&exact, &exact), ErrorNorms::default());
        let exact_f32 = [1.0_f32, -2.0, 0.5];
        assert_eq!(potential_errors(&exact_f32, &exact), ErrorNorms::default());
        let g = [[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]];
        assert_eq!(gradient_errors(&g, &g), ErrorNorms::default());
        let magnitude = [1.0, 2.0];
        assert_eq!(
            gradient_error_relative_to_magnitude(&g, &g, &magnitude),
            0.0
        );
        assert_eq!(
            potential_error_relative_to_magnitude(&exact[..2], &exact[..2], &magnitude),
            0.0
        );
        let x = [0.5, -1.0, 2.0, 0.25];
        for kind in [CoefficientKind::Multipole, CoefficientKind::Local] {
            assert_eq!(degree_errors(kind, 1, &x, &x), [0.0, 0.0]);
            assert_eq!(max_relative_degree_error(kind, 1, &x, &x, &[1.0, 1.0]), 0.0);
        }
        // A zero reference with zero error is 0, with nonzero error +∞.
        assert_eq!(potential_errors(&[0.0], &[0.0]), ErrorNorms::default());
        let infinite = potential_errors(&[1.0], &[0.0]);
        assert!(infinite.l2.is_infinite() && infinite.max.is_infinite());
    }

    #[test]
    fn known_potential_error() {
        // Error measure: relative error of the metric against hand-computed values,
        // 1e-15.
        let exact = [3.0, 4.0, 0.0, -12.0];
        let approx = [3.0, 4.5, 0.25, -12.0];
        // ‖(0, 0.5, 0.25, 0)‖ = √0.3125, ‖exact‖ = 13; max: 0.5 / 12.
        let e = potential_errors(&approx, &exact);
        assert!((e.l2 - 0.3125_f64.sqrt() / 13.0).abs() < 1e-15);
        assert!((e.max - 0.5 / 12.0).abs() < 1e-15);
    }

    #[test]
    fn known_gradient_error() {
        // Error measure: relative error of the metric against hand-computed values,
        // 1e-15.
        let exact = [[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]];
        let approx = [[3.0, 4.0, 1.0], [0.0, 2.0, 12.0]];
        // Per-target errors 1 and 2; reference norms 5 and 12.
        let e = gradient_errors(&approx, &exact);
        assert!((e.l2 - (5.0_f64).sqrt() / 13.0).abs() < 1e-15);
        assert!((e.max - 2.0 / 12.0).abs() < 1e-15);
    }

    #[test]
    fn accumulator_equals_one_batch() {
        // Error measure: exact equality; the same sums in the same order.
        let exact = [1.0, -2.0, 0.5, 3.0];
        let approx = [1.1, -2.0, 0.4, 3.3];
        let mut acc = ErrorAccumulator::new();
        acc.add_values(&approx[..1], &exact[..1]);
        acc.add_values(&approx[1..], &exact[1..]);
        assert_eq!(acc.finish(), potential_errors(&approx, &exact));
    }

    #[test]
    fn known_magnitude_error() {
        // Two charges +1 and −1 at distance 1 and 3 from a target: potential
        // 1 − 1/3, term magnitudes 1 + 1/3; gradient magnitudes 1 + 1/9.
        // Error measure: relative error of the metric, 1e-15.
        let sources = [[1.0, 0.0, 0.0], [0.0, -3.0, 0.0]];
        let charges = [1.0, -1.0];
        let targets = [[0.0; 3]];
        let pm = potential_magnitudes(&sources, &charges, &targets);
        assert!((pm[0] - 4.0 / 3.0).abs() < 1e-15);
        let gm = gradient_magnitudes(&sources, &charges, &targets);
        assert!((gm[0] - 10.0 / 9.0).abs() < 1e-15);
        let err = potential_error_relative_to_magnitude(&[1.0], &[1.0 - 1.0 / 3.0], &pm);
        assert!((err - 0.25).abs() < 1e-15);
        let err = gradient_error_relative_to_magnitude(&[[0.0, 1.0, 0.0]], &[[0.0; 3]], &gm);
        assert!((err - 0.9).abs() < 1e-15);
        // A coincident source is skipped, as in P2P.
        let pm = potential_magnitudes(&sources, &charges, &[[1.0, 0.0, 0.0]]);
        assert!((pm[0] - 1.0 / 10.0_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn weights_by_hand() {
        // Error measure: relative error 1e-15 against hand-computed weights.
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-15 * b.abs();
        assert_eq!(weight(CoefficientKind::Multipole, 0, 0), 1.0);
        assert_eq!(weight(CoefficientKind::Local, 0, 0), 1.0);
        // n = 2: m = 0 → √(2! 2!) = 2; m = ±1 → √(3! 1!) √2 = √12; m = ±2 → √(4! 0!) √2 = √48;
        // locals: the reciprocal factorial part, m = 1 → √2 / √(3! 1!).
        assert!(close(weight(CoefficientKind::Multipole, 2, 0), 2.0));
        assert!(close(
            weight(CoefficientKind::Multipole, 2, -1),
            12.0_f64.sqrt()
        ));
        assert!(close(
            weight(CoefficientKind::Multipole, 2, 2),
            48.0_f64.sqrt()
        ));
        assert!(close(weight(CoefficientKind::Local, 2, 0), 0.5));
        assert!(close(
            weight(CoefficientKind::Local, 2, 1),
            2.0_f64.sqrt() / 6.0_f64.sqrt()
        ));
    }

    #[test]
    fn known_degree_error() {
        // p = 1, multipole weights: degree 0 weight 1; degree 1 weights
        // √(2! 0!) √2 = 2 (m = ±1) and √(1! 1!) = 1 (m = 0). A difference of 1 in slot
        // (1, −1) and 2 in slot (1, 0) has degree-1 norm √(4 + 4) = √8.
        // Error measure: relative error of the metric, 1e-15.
        let want = [1.0, 0.0, 0.0, 0.0];
        let got = [1.0, 1.0, 2.0, 0.0];
        let e = degree_errors(CoefficientKind::Multipole, 1, &got, &want);
        assert_eq!(e[0], 0.0);
        assert!((e[1] - 8.0_f64.sqrt()).abs() < 1e-15);
        // Local weights for degree 1: √2/√2 = 1 (m = ±1), 1 (m = 0); norm √(1 + 4).
        let e = degree_errors(CoefficientKind::Local, 1, &got, &want);
        assert!((e[1] - 5.0_f64.sqrt()).abs() < 1e-15);
        let rel = max_relative_degree_error(CoefficientKind::Local, 1, &got, &want, &[1.0, 3.0]);
        assert!((rel - 5.0_f64.sqrt() / 3.0).abs() < 1e-15);
    }

    #[test]
    fn harmonic_norms_in_the_orthonormal_weighting() {
        // The weighted norms of §3.8: ‖Rₙ(u)‖_N = |u|ⁿ and ‖Iₙ(u)‖_{N/S} = 1/|u|ⁿ⁺¹.
        // Error measure: relative error per degree, 1e-13, for p = 20.
        let p = 20;
        let len = Layout::new(p).len();
        let (mut r, mut i) = (vec![0.0; len], vec![0.0; len]);
        let mut rng = SplitMix64::new(9);
        for x in points::ball(&mut rng, 20, [0.0; 3], 3.0) {
            let s = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
            harmonics::regular(p, x, &mut r);
            harmonics::irregular(p, x, &mut i);
            let rn = degree_norms(CoefficientKind::Multipole, p, &r);
            let in_ = degree_norms(CoefficientKind::Local, p, &i);
            for n in 0..=p {
                let want_r = s.powi(n as i32);
                let want_i = s.powi(-(n as i32) - 1);
                assert!((rn[n] - want_r).abs() <= 1e-13 * want_r, "R, n = {n}");
                assert!((in_[n] - want_i).abs() <= 1e-13 * want_i, "I, n = {n}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "approx and exact differ in length")]
    fn rejects_mismatched_lengths() {
        let _ = potential_errors(&[1.0, 2.0], &[1.0]);
    }
}

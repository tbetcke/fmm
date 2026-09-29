//! Particle-to-particle interaction: the near-field operator [`p2p`] and the f64
//! direct-sum oracle [`direct_sum`] (CONVENTIONS §3.1).
//!
//! Both add, at each target xᵢ, the potential and optionally the gradient with
//! respect to xᵢ of point charges qⱼ at sources yⱼ,
//!
//! φ(xᵢ) += Σⱼ qⱼ / |xᵢ − yⱼ|,  ∇φ(xᵢ) += −Σⱼ qⱼ (xᵢ − yⱼ) / |xᵢ − yⱼ|³,
//!
//! for the kernel 1/|x − y| of CONVENTIONS §3.1, without the factor 1/(4π). Both
//! accumulate (+=) into their outputs; `gradient`, if given, holds ∂x, ∂y, ∂z per
//! target.
//!
//! # Coincident points
//!
//! A pair with xᵢ == yⱼ exactly (all three coordinates compare equal) contributes
//! nothing, to the potential or to the gradient. This excludes self-interaction when
//! targets and sources are the same points, and every copy of a duplicated source that
//! coincides with a target. Nearly coincident pairs are *not* skipped: they contribute
//! their (large) terms, and a separation whose square underflows to zero gives
//! non-finite values.
//!
//! # The two functions
//!
//! - [`p2p`] is the near-field operator, generic over [`RealScalar`]. It sums over the
//!   sources in input order with plain floating-point additions, without reordering or
//!   compensation, so its result is reproducible operation by operation. Later fast
//!   P2P kernels are checked against it.
//! - [`direct_sum`] is the f64 oracle that end-to-end tests compare against. It
//!   computes the same terms as [`p2p`] and adds them with compensated (Neumaier)
//!   summation per target, so its summation error does not grow with the number of
//!   sources in practice. It is serial and simple, and costs O(N_s · N_t).
//!
//! # Panics
//!
//! Both functions panic if `sources` and `charges` differ in length, if `potential`
//! and `targets` differ in length, or if a given `gradient` has a length other than
//! `targets.len()`.
//!
//! # Example
//!
//! ```
//! use nd_fmm_ref::p2p::{direct_sum, p2p};
//!
//! let points: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]];
//! let charges = [2.0, -1.0];
//!
//! // Targets equal to the sources: the self-interaction pairs are skipped.
//! let mut potential = [0.0; 2];
//! let mut gradient = [[0.0; 3]; 2];
//! p2p(&points, &charges, &points, &mut potential, Some(&mut gradient));
//! assert_eq!(potential, [-1.0 / 5.0, 2.0 / 5.0]);
//! assert_eq!(gradient[1], [-2.0 * 3.0 / 125.0, -2.0 * 4.0 / 125.0, 0.0]);
//!
//! let mut oracle = [0.0; 2];
//! direct_sum(&points, &charges, &points, &mut oracle, None);
//! assert_eq!(oracle, potential);
//! ```

use nd_fmm_math::RealScalar;

/// P2P: adds the potential of point charges, and optionally its gradient, at each
/// target (CONVENTIONS §3.1).
///
/// For each target xᵢ and each source yⱼ with xᵢ ≠ yⱼ, with d = xᵢ − yⱼ,
/// r² = (d₀² + d₁²) + d₂² and r = √r²,
///
/// φᵢ ← φᵢ + qⱼ / r,  (∇φᵢ)ₖ ← (∇φᵢ)ₖ − qⱼ dₖ / (r² r),
///
/// which is φ(xᵢ) += Σⱼ qⱼ / |xᵢ − yⱼ| and ∇φ(xᵢ) += −Σⱼ qⱼ (xᵢ − yⱼ) / |xᵢ − yⱼ|³,
/// without the factor 1/(4π). The sources are visited in input order and each term is
/// added directly into `potential` and `gradient`, with no reordering and no
/// compensation, exactly as written above (qⱼ dₖ is formed first, then divided by
/// r² r).
///
/// A pair with xᵢ == yⱼ exactly is skipped, which excludes self-interaction when
/// `targets` and `sources` are the same points. Nearly coincident pairs are not
/// skipped; see the [module documentation](self).
///
/// # Panics
///
/// If `sources.len() != charges.len()`, `potential.len() != targets.len()` or a given
/// `gradient` has a length other than `targets.len()`.
pub fn p2p<T: RealScalar>(
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: Option<&mut [[T; 3]]>,
) {
    check_lengths(sources, charges, targets, potential, gradient.as_deref());
    match gradient {
        None => {
            for (&x, phi) in targets.iter().zip(potential.iter_mut()) {
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = Term::new(x, y, q) {
                        *phi = *phi + term.potential();
                    }
                }
            }
        }
        Some(gradient) => {
            for ((&x, phi), g) in targets.iter().zip(potential.iter_mut()).zip(gradient) {
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = Term::new(x, y, q) {
                        *phi = *phi + term.potential();
                        for (gk, dgk) in g.iter_mut().zip(term.gradient()) {
                            *gk = *gk + dgk;
                        }
                    }
                }
            }
        }
    }
}

/// Direct sum: the f64 oracle for the potential of point charges, and optionally its
/// gradient, at each target (CONVENTIONS §3.1).
///
/// Adds φ(xᵢ) = Σⱼ qⱼ / |xᵢ − yⱼ| and ∇φ(xᵢ) = −Σⱼ qⱼ (xᵢ − yⱼ) / |xᵢ − yⱼ|³ to
/// `potential` and `gradient`, without the factor 1/(4π). Each term is computed
/// exactly as in [`p2p`]. The terms of each target are summed in input order with
/// compensated summation (Neumaier's variant of Kahan summation), separately for the
/// potential and each gradient component. The summation error is then at most about
/// one rounding of the result plus N_s ε² times the sum of the term magnitudes (ε the
/// unit roundoff), so for any practical N_s the accuracy is set by the few roundings in
/// each term, not by the number of sources. The compensated sum is then added to the
/// existing output value in one final addition.
///
/// A pair with xᵢ == yⱼ exactly is skipped, which excludes self-interaction when
/// `targets` and `sources` are the same points. Nearly coincident pairs are not
/// skipped; see the [module documentation](self).
///
/// The function is serial and O(N_s · N_t); it is meant as a reference, not for large
/// problems.
///
/// # Panics
///
/// If `sources.len() != charges.len()`, `potential.len() != targets.len()` or a given
/// `gradient` has a length other than `targets.len()`.
pub fn direct_sum(
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
    potential: &mut [f64],
    gradient: Option<&mut [[f64; 3]]>,
) {
    check_lengths(sources, charges, targets, potential, gradient.as_deref());
    match gradient {
        None => {
            for (&x, phi) in targets.iter().zip(potential.iter_mut()) {
                let mut sum = Neumaier::default();
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = Term::new(x, y, q) {
                        sum.add(term.potential());
                    }
                }
                *phi += sum.value();
            }
        }
        Some(gradient) => {
            for ((&x, phi), g) in targets.iter().zip(potential.iter_mut()).zip(gradient) {
                let mut sum = Neumaier::default();
                let mut grad = [Neumaier::default(); 3];
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = Term::new(x, y, q) {
                        sum.add(term.potential());
                        for (gk, dgk) in grad.iter_mut().zip(term.gradient()) {
                            gk.add(dgk);
                        }
                    }
                }
                *phi += sum.value();
                for (gk, sk) in g.iter_mut().zip(&grad) {
                    *gk += sk.value();
                }
            }
        }
    }
}

/// One source–target pair xᵢ ≠ yⱼ: the shared term formula of [`p2p`] and
/// [`direct_sum`].
struct Term<T> {
    /// qⱼ.
    q: T,
    /// d = xᵢ − yⱼ.
    d: [T; 3],
    /// r² = (d₀² + d₁²) + d₂².
    r2: T,
    /// r = √r².
    r: T,
}

impl<T: RealScalar> Term<T> {
    /// The pair (x, y) with charge q, or `None` if x == y exactly.
    #[inline]
    fn new(x: [T; 3], y: [T; 3], q: T) -> Option<Self> {
        if x == y {
            return None;
        }
        let d = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        Some(Self {
            q,
            d,
            r2,
            r: r2.sqrt(),
        })
    }

    /// q / r.
    #[inline]
    fn potential(&self) -> T {
        self.q / self.r
    }

    /// −q d / r³, computed per component as −((q dₖ) / (r² r)).
    #[inline]
    fn gradient(&self) -> [T; 3] {
        let r3 = self.r2 * self.r;
        self.d.map(|dk| -(self.q * dk / r3))
    }
}

/// Neumaier's compensated sum: a running sum and the rounding errors lost from it.
#[derive(Clone, Copy, Default)]
struct Neumaier {
    /// The running floating-point sum.
    sum: f64,
    /// The accumulated rounding errors of the additions into `sum`.
    compensation: f64,
}

impl Neumaier {
    /// Adds `term`, recording the rounding error of the addition exactly (Fast2Sum on
    /// the larger and the smaller summand).
    #[inline]
    fn add(&mut self, term: f64) {
        let t = self.sum + term;
        if self.sum.abs() >= term.abs() {
            self.compensation += (self.sum - t) + term;
        } else {
            self.compensation += (term - t) + self.sum;
        }
        self.sum = t;
    }

    /// The compensated sum.
    #[inline]
    fn value(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// Panics unless the paired slices have matching lengths.
fn check_lengths<T>(
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &[T],
    gradient: Option<&[[T; 3]]>,
) {
    check_pair(sources.len(), charges.len(), "sources", "charges");
    check_pair(targets.len(), potential.len(), "targets", "potential");
    if let Some(gradient) = gradient {
        check_pair(targets.len(), gradient.len(), "targets", "gradient");
    }
}

/// Panics unless the paired slices `a` and `b` have the same length.
fn check_pair(a: usize, b: usize, name_a: &str, name_b: &str) {
    assert_eq!(
        a, b,
        "`{name_a}` and `{name_b}` must have the same length, got {a} and {b}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumaier_recovers_lost_terms() {
        // Error measure: exact equality. Plain summation of 1, 1e100, 1, −1e100 gives 0;
        // Neumaier's variant recovers both units, which Kahan's original loses.
        let mut sum = Neumaier::default();
        for t in [1.0, 1e100, 1.0, -1e100] {
            sum.add(t);
        }
        assert_eq!(sum.value(), 2.0);
    }

    #[test]
    fn direct_sum_is_compensated_and_p2p_is_not() {
        // Error measure: exact equality. All sources are at distance 1 from the target,
        // so each term equals its charge exactly.
        let sources = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ];
        let charges = [1.0, 1e100, 1.0, -1e100];
        let origin = [[0.0; 3]];
        let mut compensated = [0.0];
        direct_sum(&sources, &charges, &origin, &mut compensated, None);
        assert_eq!(compensated, [2.0]);
        let mut plain = [0.0];
        p2p(&sources, &charges, &origin, &mut plain, None);
        assert_eq!(plain, [0.0]);
    }

    #[test]
    fn term_skips_only_exact_coincidence() {
        // Error measure: exact equality.
        assert!(Term::new([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0).is_none());
        assert!(Term::new([0.0, 0.0, 0.0], [-0.0, 0.0, -0.0], 1.0).is_none());
        // Distinct points whose squared separation underflows to zero are not skipped.
        let near = Term::new([1e-200_f64, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0);
        assert!(near.is_some_and(|t| t.potential().is_infinite()));
        let term = Term::new([3.0, 0.0, 4.0], [0.0, 0.0, 0.0], 2.0).unwrap();
        assert_eq!(term.potential(), 0.4);
        assert_eq!(term.gradient(), [-6.0 / 125.0, 0.0, -8.0 / 125.0]);
    }
}

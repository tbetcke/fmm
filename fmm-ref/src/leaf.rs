//! Leaf operators: P2M and P2L form expansions from point charges, L2P and M2P
//! evaluate them at targets (CONVENTIONS §3.6, §3.7).
//!
//! All four take the degree `p` first, then the [`Frame`] of the expansion, their
//! inputs, the [`Workspace`] and their outputs. Coefficients are the scaled ones of
//! CONVENTIONS §3.7, (p + 1)² reals in the real storage of §3.6 (see
//! [`nd_fmm_math::Layout`]). Every operator adds (+=) into its output and none applies
//! 1/(4π) (§3.1). They are shared by the translation methods of later tasks.
//!
//! # Tested range
//!
//! CONVENTIONS §3.9 fixes where the solid harmonics are evaluated, in scaled
//! coordinates: regular harmonics at |u| ≤ √3 (points inside the sphere around a box
//! of half-width r), irregular harmonics at |v| ≥ 2. The operators are tested there,
//! with a margin, for p ≤ 30 in f64 and p ≤ 8 in f32. They compute something outside
//! that range too, but its accuracy is not guaranteed.
//!
//! # Panics
//!
//! Every operator panics if the workspace was built for a smaller degree than `p`, if
//! a coefficient slice does not have length (p + 1)², or if paired slices (sources and
//! charges; targets, potential and gradient) differ in length.
//!
//! # Example
//!
//! ```
//! use nd_fmm_ref::{Frame, Workspace, leaf};
//!
//! let p = 12;
//! let mut ws = Workspace::new(p);
//! let frame = Frame::new([1.0, 1.0, 1.0], 0.5);
//! let sources: [[f64; 3]; 2] = [[1.1, 0.8, 1.2], [0.7, 1.3, 0.9]];
//! let charges = [1.0, -0.5];
//!
//! let mut multipole = vec![0.0; (p + 1) * (p + 1)];
//! leaf::p2m(p, &frame, &sources, &charges, &mut ws, &mut multipole);
//!
//! let targets: [[f64; 3]; 1] = [[4.0, 1.0, 1.0]];
//! let mut potential = [0.0];
//! let mut gradient = [[0.0; 3]];
//! leaf::m2p(p, &frame, &multipole, &targets, &mut ws, &mut potential, Some(&mut gradient));
//!
//! let exact: f64 = sources
//!     .iter()
//!     .zip(charges)
//!     .map(|(y, q)| {
//!         let d: f64 = (0..3).map(|i| (targets[0][i] - y[i]).powi(2)).sum();
//!         q / d.sqrt()
//!     })
//!     .sum();
//! assert!((potential[0] - exact).abs() < 1e-6);
//! ```

use nd_fmm_math::{Layout, RealScalar, harmonics};

use crate::{Frame, Workspace};

/// P2M: adds the scaled multipole coefficients of point charges to `multipole`
/// (CONVENTIONS §3.7).
///
/// With c and r the centre and radius of `frame` and uⱼ = (yⱼ − c)/r,
///
/// M̃ₙᵐ += Σⱼ qⱼ conj(Rₙᵐ(uⱼ)),  n ≤ p,
///
/// written for m ≥ 0 in real storage (§3.6): slot m holds Re M̃ₙᵐ, slot −m holds
/// Im M̃ₙᵐ = −Σⱼ qⱼ Im Rₙᵐ(uⱼ). Each source is added in turn, degree by degree.
/// The harmonics come from [`harmonics::regular`] (§3.5).
///
/// Tested for sources at |uⱼ| ≤ √3 (CONVENTIONS §3.9), i.e. within the sphere around
/// a box of half-width r centred at c; outside it the result is not guaranteed.
///
/// # Panics
///
/// If `ws.p() < p`, `multipole.len() != (p + 1)²` or
/// `sources.len() != charges.len()`.
pub fn p2m<T: RealScalar>(
    p: usize,
    frame: &Frame<T>,
    sources: &[[T; 3]],
    charges: &[T],
    ws: &mut Workspace<T>,
    multipole: &mut [T],
) {
    check_coefficients(p, multipole, "multipole");
    check_pair(sources.len(), charges.len(), "sources", "charges");
    let values = ws.harmonics(p);
    for (&y, &q) in sources.iter().zip(charges) {
        harmonics::regular(p, frame.scaled(y), values);
        add_conjugate(p, q, values, multipole);
    }
}

/// P2L: adds the scaled local coefficients of point charges to `local`
/// (CONVENTIONS §3.7).
///
/// With c and r the centre and radius of `frame` and uⱼ = (yⱼ − c)/r,
///
/// L̃ₙᵐ += Σⱼ qⱼ conj(Iₙᵐ(uⱼ)),  n ≤ p,
///
/// written for m ≥ 0 in real storage (§3.6): slot m holds Re L̃ₙᵐ, slot −m holds
/// Im L̃ₙᵐ = −Σⱼ qⱼ Im Iₙᵐ(uⱼ). Each source is added in turn, degree by degree.
/// The harmonics come from [`harmonics::irregular`] (§3.5).
///
/// Tested for sources at |uⱼ| ≥ 2 (CONVENTIONS §3.9); closer to the centre the result
/// is not guaranteed, and a source at the centre gives non-finite values.
///
/// # Panics
///
/// If `ws.p() < p`, `local.len() != (p + 1)²` or `sources.len() != charges.len()`.
pub fn p2l<T: RealScalar>(
    p: usize,
    frame: &Frame<T>,
    sources: &[[T; 3]],
    charges: &[T],
    ws: &mut Workspace<T>,
    local: &mut [T],
) {
    check_coefficients(p, local, "local");
    check_pair(sources.len(), charges.len(), "sources", "charges");
    let values = ws.harmonics(p);
    for (&y, &q) in sources.iter().zip(charges) {
        harmonics::irregular(p, frame.scaled(y), values);
        add_conjugate(p, q, values, local);
    }
}

/// L2P: adds the potential of a scaled local expansion, and optionally its gradient,
/// at each target (CONVENTIONS §3.6, §3.7).
///
/// With c and r the centre and radius of `frame` and v = (x − c)/r,
///
/// φ(x) += (1/r) Σₙ≤ₚ Σₘ L̃ₙᵐ Rₙᵐ(v),  ∇φ(x) += (1/r²) Σₙ≤ₚ Σₘ L̃ₙᵐ (∇Rₙᵐ)(v),
///
/// where each inner sum over m is evaluated from real storage by the doubling rule of
/// §3.6, L̃ₙ⁰Rₙ⁰ + 2 Σₘ≥₁ (Re L̃ₙᵐ Re Rₙᵐ − Im L̃ₙᵐ Im Rₙᵐ), and the same for each
/// Cartesian component of the gradient. The extra 1/r of the gradient is the chain
/// rule for v = (x − c)/r. The harmonics come from [`harmonics::regular`], and with a
/// gradient from [`harmonics::regular_grad`], whose degree-p gradient needs only
/// degrees below p. `gradient`, if given, holds ∂x, ∂y, ∂z per target.
///
/// Tested for targets at |v| ≤ √3 (CONVENTIONS §3.9); outside that ball the result is
/// not guaranteed.
///
/// # Panics
///
/// If `ws.p() < p`, `local.len() != (p + 1)²`, `potential.len() != targets.len()` or
/// a given `gradient` has a length other than `targets.len()`.
pub fn l2p<T: RealScalar>(
    p: usize,
    frame: &Frame<T>,
    local: &[T],
    targets: &[[T; 3]],
    ws: &mut Workspace<T>,
    potential: &mut [T],
    gradient: Option<&mut [[T; 3]]>,
) {
    check_coefficients(p, local, "local");
    let expansion = Expansion {
        basis: Basis::Regular,
        p,
        frame,
        coefficients: local,
    };
    expansion.evaluate(targets, ws, potential, gradient);
}

/// M2P: adds the potential of a scaled multipole expansion, and optionally its
/// gradient, at each target (CONVENTIONS §3.6, §3.7).
///
/// With c and r the centre and radius of `frame` and v = (x − c)/r,
///
/// φ(x) += (1/r) Σₙ≤ₚ Σₘ M̃ₙᵐ Iₙᵐ(v),  ∇φ(x) += (1/r²) Σₙ≤ₚ Σₘ M̃ₙᵐ (∇Iₙᵐ)(v),
///
/// where each inner sum over m is evaluated from real storage by the doubling rule of
/// §3.6, M̃ₙ⁰Iₙ⁰ + 2 Σₘ≥₁ (Re M̃ₙᵐ Re Iₙᵐ − Im M̃ₙᵐ Im Iₙᵐ), and the same for each
/// Cartesian component of the gradient. The extra 1/r of the gradient is the chain
/// rule for v = (x − c)/r. The harmonics come from [`harmonics::irregular`], and with
/// a gradient from [`harmonics::irregular_grad`], which forms degree p + 1 internally
/// for the gradient of degree p. `gradient`, if given, holds ∂x, ∂y, ∂z per target.
///
/// Tested for targets at |v| ≥ 2 (CONVENTIONS §3.9); closer to the centre the result
/// is not guaranteed, and a target at the centre gives non-finite values.
///
/// # Panics
///
/// If `ws.p() < p`, `multipole.len() != (p + 1)²`, `potential.len() != targets.len()`
/// or a given `gradient` has a length other than `targets.len()`.
pub fn m2p<T: RealScalar>(
    p: usize,
    frame: &Frame<T>,
    multipole: &[T],
    targets: &[[T; 3]],
    ws: &mut Workspace<T>,
    potential: &mut [T],
    gradient: Option<&mut [[T; 3]]>,
) {
    check_coefficients(p, multipole, "multipole");
    let expansion = Expansion {
        basis: Basis::Irregular,
        p,
        frame,
        coefficients: multipole,
    };
    expansion.evaluate(targets, ws, potential, gradient);
}

/// The solid harmonics an expansion is evaluated in.
#[derive(Clone, Copy)]
enum Basis {
    /// Rₙᵐ, for local expansions.
    Regular,
    /// Iₙᵐ, for multipole expansions.
    Irregular,
}

/// An expansion to evaluate: the shared body of [`l2p`] and [`m2p`].
struct Expansion<'a, T: RealScalar> {
    basis: Basis,
    p: usize,
    frame: &'a Frame<T>,
    coefficients: &'a [T],
}

impl<T: RealScalar> Expansion<'_, T> {
    /// Adds the potential, and optionally the gradient, at each target.
    fn evaluate(
        &self,
        targets: &[[T; 3]],
        ws: &mut Workspace<T>,
        potential: &mut [T],
        gradient: Option<&mut [[T; 3]]>,
    ) {
        let Self {
            basis,
            p,
            frame,
            coefficients,
        } = *self;
        check_pair(targets.len(), potential.len(), "targets", "potential");
        let r = frame.radius;
        let r2 = r * r;
        match gradient {
            None => {
                let values = ws.harmonics(p);
                for (&x, phi) in targets.iter().zip(potential.iter_mut()) {
                    let v = frame.scaled(x);
                    match basis {
                        Basis::Regular => harmonics::regular(p, v, values),
                        Basis::Irregular => harmonics::irregular(p, v, values),
                    }
                    *phi = *phi + contract(p, coefficients, values) / r;
                }
            }
            Some(gradient) => {
                check_pair(targets.len(), gradient.len(), "targets", "gradient");
                let (values, mut grad) = ws.harmonics_and_gradient(p);
                for ((&x, phi), g) in targets.iter().zip(potential.iter_mut()).zip(gradient) {
                    let v = frame.scaled(x);
                    let [gx, gy, gz] = &mut grad;
                    let components = [&mut **gx, &mut **gy, &mut **gz];
                    match basis {
                        Basis::Regular => harmonics::regular_grad(p, v, values, components),
                        Basis::Irregular => harmonics::irregular_grad(p, v, values, components),
                    }
                    *phi = *phi + contract(p, coefficients, values) / r;
                    for (gk, component) in g.iter_mut().zip(&grad) {
                        *gk = *gk + contract(p, coefficients, component) / r2;
                    }
                }
            }
        }
    }
}

/// Adds q · conj(X) to `out` in real storage (CONVENTIONS §3.6): conjugation negates
/// the imaginary parts, which live in the slots m < 0.
fn add_conjugate<T: RealScalar>(p: usize, q: T, values: &[T], out: &mut [T]) {
    let layout = Layout::new(p);
    for (i, (o, &x)) in out.iter_mut().zip(values).enumerate() {
        let (_, m) = layout.nm(i);
        *o = if m < 0 { *o - q * x } else { *o + q * x };
    }
}

/// Σₙ≤ₚ Σₘ Cₙᵐ Xₙᵐ from real storage, by the doubling rule of CONVENTIONS §3.6:
/// per degree Cₙ⁰Xₙ⁰ + 2 Σₘ≥₁ (Re Cₙᵐ Re Xₙᵐ − Im Cₙᵐ Im Xₙᵐ).
fn contract<T: RealScalar>(p: usize, coefficients: &[T], values: &[T]) -> T {
    let layout = Layout::new(p);
    let two = T::from_f64(2.0);
    let mut sum = T::zero();
    for n in 0..=p {
        let zero = layout.idx(n, 0);
        let mut orders = T::zero();
        for m in 1..=n as isize {
            let (re, im) = (layout.idx(n, m), layout.idx(n, -m));
            orders = orders + coefficients[re] * values[re] - coefficients[im] * values[im];
        }
        sum = sum + coefficients[zero] * values[zero] + two * orders;
    }
    sum
}

/// Panics unless the coefficient slice `name` has length (p + 1)².
fn check_coefficients<T>(p: usize, coefficients: &[T], name: &str) {
    let len = Layout::new(p).len();
    assert_eq!(
        coefficients.len(),
        len,
        "`{name}` must have length (p + 1)^2 = {len} for p = {p}"
    );
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
    fn contract_applies_doubling_rule() {
        // Error measure: exact equality; all values are small integers.
        // Degree 0: 2·3; degree 1: 1·5 + 2 (4·6 − (−1)·7) = 5 + 62.
        let coefficients = [2.0, -1.0, 1.0, 4.0];
        let values = [3.0, 7.0, 5.0, 6.0];
        assert_eq!(contract(1, &coefficients, &values), 6.0 + 67.0);
    }

    #[test]
    fn add_conjugate_negates_imaginary_slots() {
        // Error measure: exact equality; all values are small integers.
        let mut out = [1.0; 4];
        add_conjugate(1, 2.0, &[3.0, 4.0, 5.0, 6.0], &mut out);
        assert_eq!(out, [7.0, -7.0, 11.0, 13.0]);
    }
}

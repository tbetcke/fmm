//! The degree p_l of each level of a Helmholtz FMM (Helmholtz CH3.1a, calibrated in CH3.3;
//! docs/design/helmholtz-fmm-proposal.md §3.5; spike S7,
//! `spikes/truncation/SPIKE_REPORT.md` §3 and §8, question 1, signed off on 2026-10-29;
//! `fmm-validate/results/helmholtz-ch3-3.md`).
//!
//! The degrees are a setting of the FMM, [`Degrees`]: by default a rule at a tolerance ε,
//! or degrees named per level, which tests and the calibration use. The rule is
//! [`degree`], [`DegreeRule::CALIBRATED`] at x = Re(k) a with a the circumradius √3 r_l of
//! the level's boxes and L = log₁₀(1/ε),
//!
//! ```text
//! p(x, ε) = max(⌈0.30 L² + 0.25 L + 2.60⌉, ⌈x + 0.80 (ln 1/ε)^{2/3} x^{1/3}⌉),
//! ```
//!
//! a low-frequency plateau and a high-frequency branch, calibrated in the full FMM
//! (Helmholtz CH3.3): every degree the smallest that keeps the relative ℓ2 errors of φ and
//! ∇φ over 1,000 sampled targets within ε/2, on W1, W3 and W4 of proposal §9.1, one level
//! scanned at a time (the plateau on trees with every level at x ≤ 2.2, all levels together),
//! over 0.14 ≤ x ≤ 65 and 10⁻⁸ ≤ ε ≤ 10⁻³ (the plateau to 10⁻¹⁰); the rule is at least
//! every measured degree. Outside those ranges it is an extrapolation.
//!
//! S7's closed form, [`s7_degree`], max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈x + (ln 1/ε)^{2/3} x^{1/3}⌉),
//! was the default before CH3.3. It is the per-pair worst case (the worst of the 316
//! offsets and three placements, potential and gradient, per-pair relative ℓ2 norm, margin
//! ε/2) and lies at or above the calibrated rule for 10⁻¹⁸ ≤ ε ≤ 10⁻¹: about twice its plateau
//! at low frequency (34 against 15 at ε = 10⁻⁶), where the full FMM's errors lay 10³ to 10⁶
//! below ε (CH3.1c).
//!
//! The high-frequency branch is measured to x = 65 (kD = 300 on the unit cube) and checked
//! in the full FMM at x = 108 (W1 at kD = 500, 0.63 ε at ε = 10⁻⁸, the tightest case of the
//! verification). Beyond, it is an extrapolation that falls increasingly below S7's
//! per-pair p* as x grows and ε falls (157 against 161 at x = 128, 292 against 297 at
//! x = 256, both at 10⁻⁸): the measured excess p − x grows about linearly in log₁₀(1/ε),
//! faster than (ln 1/ε)^{2/3}. Above kD ≈ 500 at ε ≤ 10⁻⁷, check the errors or name S7's
//! degrees per level; CH4.3 recalibrates the high-frequency levels.
//!
//! Complex k takes Re(k) a, provisionally: S7 found η = Im k / Re k = 0.1 within 10 degrees
//! of real k, and the damping-dependent rule is CH5.2's.
//!
//! Levels 0 and 1 get degree 0 under the rule: no V list lies above level 2, no X list
//! reaches a box on them and no W list holds one, so nothing reads their expansions; their
//! multipoles are only formed by M2M and their locals only passed down by L2L, as zeros.
//! The degrees depend on k, the domain and ε only (proposal §2, constraint 9).
//!
//! ```
//! use nd_fmm_exec::geometry::Domain;
//! use nd_fmm_exec::helmholtz::{Degrees, Wavenumber, degree, s7_degree};
//! use nd_octree::PhysicalBox;
//!
//! // Low frequency: the plateau ⌈0.30 · 36 + 0.25 · 6 + 2.60⌉ = 15 at ε = 10⁻⁶; S7's 34.
//! let k = Wavenumber::new(1.0, 0.0).unwrap();
//! assert_eq!(degree(&k, 0.01, 1e-6), 15);
//! assert_eq!(s7_degree(&k, 0.01, 1e-6), 34);
//!
//! // kD = 100 on the unit cube: level 2 has x = 100 · √3 / 8 ≈ 21.65, so
//! // ⌈21.65 + 0.80 · 13.8^{2/3} · 21.65^{1/3}⌉ = 35; level 3 (x ≈ 10.8) 22.
//! let k = Wavenumber::new(100.0, 0.0).unwrap();
//! let domain = Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap();
//! let degrees = Degrees::Tolerance(1e-6).resolve(&k, &domain, 4).unwrap();
//! assert_eq!(degrees, [0, 0, 35, 22]);
//! ```

use thiserror::Error;

use super::Wavenumber;
use crate::geometry::{Domain, radius};

/// The degrees of an FMM's levels: a rule at a tolerance, or named per level (module
/// documentation).
#[derive(Clone, Debug, PartialEq)]
pub enum Degrees {
    /// [`degree`] at the tolerance ε, 0 < ε < 1, on levels ≥ 2; degree 0 on levels 0 and 1.
    Tolerance(f64),
    /// The degree of each level, from level 0; entries beyond the tree's levels are unused.
    PerLevel(Vec<usize>),
}

/// Why [`Degrees::resolve`] failed.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DegreeError {
    /// The tolerance is not in (0, 1).
    #[error("the tolerance must lie in (0, 1), got {0}")]
    Tolerance(f64),
    /// Fewer degrees named than the tree has levels.
    #[error("{given} degrees named for {levels} levels")]
    TooFewLevels {
        /// The degrees named.
        given: usize,
        /// The levels of the tree.
        levels: usize,
    },
}

impl Degrees {
    /// The degrees of the levels 0..`nlevels` of an octree on `domain` for `k`.
    ///
    /// # Errors
    ///
    /// [`DegreeError::Tolerance`] for a tolerance outside (0, 1) (NaN included), and
    /// [`DegreeError::TooFewLevels`] for fewer named degrees than levels.
    pub fn resolve(
        &self,
        k: &Wavenumber,
        domain: &Domain,
        nlevels: usize,
    ) -> Result<Vec<usize>, DegreeError> {
        match self {
            Self::Tolerance(epsilon) => {
                check_tolerance(*epsilon)?;
                Ok((0..nlevels)
                    .map(|l| {
                        if l < 2 {
                            0
                        } else {
                            degree(k, 3f64.sqrt() * radius(l, domain), *epsilon)
                        }
                    })
                    .collect())
            }
            Self::PerLevel(degrees) if degrees.len() < nlevels => Err(DegreeError::TooFewLevels {
                given: degrees.len(),
                levels: nlevels,
            }),
            Self::PerLevel(degrees) => Ok(degrees[..nlevels].to_vec()),
        }
    }
}

/// The default rule (module documentation), [`DegreeRule::CALIBRATED`] at x = Re(k) a, a
/// the circumradius of the level's boxes in the units of k.
///
/// # Panics
///
/// Unless 0 < ε < 1, and unless `circumradius` is finite and positive.
pub fn degree(k: &Wavenumber, circumradius: f64, epsilon: f64) -> usize {
    check_arguments(circumradius, epsilon);
    DegreeRule::CALIBRATED.degree(k.re() * circumradius, epsilon)
}

/// S7's closed form, max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈x + (ln 1/ε)^{2/3} x^{1/3}⌉) at
/// x = Re(k) a, a the circumradius of the level's boxes in the units of k: the degree of
/// S7's per-pair worst case (module documentation), an upper bound of the full FMM's.
///
/// # Panics
///
/// Unless 0 < ε < 1, and unless `circumradius` is finite and positive.
pub fn s7_degree(k: &Wavenumber, circumradius: f64, epsilon: f64) -> usize {
    check_arguments(circumradius, epsilon);
    let floor = (6.0 * (1.0 / epsilon).log10()).ceil() - 2.0;
    let x = k.re() * circumradius;
    let high = (x + (1.0 / epsilon).ln().powf(2.0 / 3.0) * x.cbrt()).ceil();
    floor.max(high) as usize
}

/// A degree rule of S7's shape (module documentation), with a quadratic plateau,
/// p(x, ε) = max(⌈γ L² + α L + β⌉, ⌈x + C (ln 1/ε)^{2/3} x^{1/3}⌉) at x = Re(k) a and
/// L = log₁₀(1/ε): a low-frequency plateau and a high-frequency branch. S7's closed form is
/// γ = 0, α = 6, β = −2, C = 1 up to the ceilings, which here forgive 10⁻⁹, so that a value
/// an integer up to rounding (α log₁₀(10⁶) with α a multiple of 0.05) stays that integer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DegreeRule {
    /// γ, the plateau's quadratic coefficient in L = log₁₀(1/ε).
    pub gamma: f64,
    /// α, the plateau's linear coefficient in L.
    pub alpha: f64,
    /// β, the plateau's offset.
    pub beta: f64,
    /// C, the constant of the high-frequency branch.
    pub c: f64,
}

impl DegreeRule {
    /// The default rule, calibrated in the full FMM (Helmholtz CH3.3; module
    /// documentation): γ = 0.30, α = 0.25, β = 2.60, C = 0.80.
    pub const CALIBRATED: Self = Self {
        gamma: 0.30,
        alpha: 0.25,
        beta: 2.60,
        c: 0.80,
    };

    /// The plateau ⌈γ L² + α L + β⌉ with L = log₁₀(1/ε), at least 0.
    pub fn plateau(&self, epsilon: f64) -> usize {
        let l = (1.0 / epsilon).log10();
        rounded_up((self.gamma * l + self.alpha) * l + self.beta)
    }

    /// The high-frequency branch ⌈x + C (ln 1/ε)^{2/3} x^{1/3}⌉, at least 0.
    pub fn high(&self, x: f64, epsilon: f64) -> usize {
        rounded_up(x + self.c * (1.0 / epsilon).ln().powf(2.0 / 3.0) * x.cbrt())
    }

    /// The larger of [`DegreeRule::plateau`] and [`DegreeRule::high`].
    pub fn degree(&self, x: f64, epsilon: f64) -> usize {
        self.plateau(epsilon).max(self.high(x, epsilon))
    }
}

/// ⌈v − 10⁻⁹⌉, at least 0.
fn rounded_up(v: f64) -> usize {
    (v - 1e-9).ceil().max(0.0) as usize
}

/// Panics unless 0 < ε < 1 and the circumradius is finite and positive.
fn check_arguments(circumradius: f64, epsilon: f64) {
    assert!(
        check_tolerance(epsilon).is_ok(),
        "the tolerance must lie in (0, 1), got {epsilon}"
    );
    assert!(
        circumradius.is_finite() && circumradius > 0.0,
        "the circumradius must be finite and positive, got {circumradius}"
    );
}

fn check_tolerance(epsilon: f64) -> Result<(), DegreeError> {
    if epsilon > 0.0 && epsilon < 1.0 {
        Ok(())
    } else {
        Err(DegreeError::Tolerance(epsilon))
    }
}

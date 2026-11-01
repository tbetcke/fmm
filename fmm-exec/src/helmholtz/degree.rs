//! The degree p_l of each level of a Helmholtz FMM (Helmholtz CH3.1a;
//! docs/design/helmholtz-fmm-proposal.md §3.5; spike S7,
//! `spikes/truncation/SPIKE_REPORT.md` §3 and §8, question 1, signed off on 2026-10-29).
//!
//! The degrees are a setting of the FMM, [`Degrees`]: by default a rule at a tolerance ε,
//! or degrees named per level, which tests and the calibration of CH3.3 use. The rule is
//! [`degree`], S7's closed form for real k,
//!
//! ```text
//! p(k a, ε) = max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈k a + (ln 1/ε)^{2/3} (k a)^{1/3}⌉)
//! ```
//!
//! with a the circumradius √3 r_l of the level's boxes. S7 fitted it to the per-level
//! degree p* of the multi-level chains (the worst of the 316 offsets and three
//! placements, potential and gradient, per-pair relative ℓ2 norm, margin ε/2) at
//! 0.016 ≤ k a ≤ 256 and 10⁻¹⁰ ≤ ε ≤ 10⁻³, which it never under-predicts and over-predicts
//! by at most 9 degrees. Between those grid points it is an interpolation. Complex k takes
//! Re(k) a, provisionally: S7 found η = Im k / Re k = 0.1 within 10 degrees of real k, and
//! the damping-dependent rule is CH5.2's. CH3.3 confirms the rule in the full FMM.
//!
//! Levels 0 and 1 get degree 0 under the rule: no V list lies above level 2, no X list
//! reaches a box on them and no W list holds one, so nothing reads their expansions; their
//! multipoles are only formed by M2M and their locals only passed down by L2L, as zeros.
//! The degrees depend on k, the domain and ε only (proposal §2, constraint 9).
//!
//! ```
//! use nd_fmm_exec::geometry::Domain;
//! use nd_fmm_exec::helmholtz::{Degrees, Wavenumber, degree};
//! use nd_octree::PhysicalBox;
//!
//! // Low frequency: the floor ⌈6 · 6⌉ − 2 = 34 at ε = 10⁻⁶.
//! let k = Wavenumber::new(1.0, 0.0).unwrap();
//! assert_eq!(degree(&k, 0.01, 1e-6), 34);
//!
//! // kD = 100 on the unit cube: level 2 has k a = 100 · √3 / 8 ≈ 21.7, so
//! // ⌈21.7 + 13.8^{2/3} · 21.7^{1/3}⌉ = 38; level 3 the floor.
//! let k = Wavenumber::new(100.0, 0.0).unwrap();
//! let domain = Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap();
//! let degrees = Degrees::Tolerance(1e-6).resolve(&k, &domain, 4).unwrap();
//! assert_eq!(degrees, [0, 0, 38, 34]);
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

/// S7's closed form (module documentation) at x = Re(k) a, a the circumradius of the
/// level's boxes in the units of k:
/// max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈x + (ln 1/ε)^{2/3} x^{1/3}⌉).
///
/// # Panics
///
/// Unless 0 < ε < 1, and unless `circumradius` is finite and positive.
pub fn degree(k: &Wavenumber, circumradius: f64, epsilon: f64) -> usize {
    s7_degree(k, circumradius, epsilon)
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

/// A degree rule of S7's shape (module documentation),
/// p(x, ε) = max(⌈α log₁₀(1/ε) + β⌉, ⌈x + C (ln 1/ε)^{2/3} x^{1/3}⌉) at x = Re(k) a:
/// a low-frequency plateau and a high-frequency branch. The ceilings forgive 10⁻⁹, so that
/// a value an integer up to rounding (α log₁₀(10⁶) with α a multiple of 0.05) stays that
/// integer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DegreeRule {
    /// α, the plateau's degrees per decade of ε.
    pub alpha: f64,
    /// β, the plateau's offset.
    pub beta: f64,
    /// C, the constant of the high-frequency branch.
    pub c: f64,
}

impl DegreeRule {
    /// The plateau ⌈α log₁₀(1/ε) + β⌉, at least 0.
    pub fn plateau(&self, epsilon: f64) -> usize {
        rounded_up(self.alpha * (1.0 / epsilon).log10() + self.beta)
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

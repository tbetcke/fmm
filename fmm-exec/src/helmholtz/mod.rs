//! The Helmholtz family's builder (docs/design/helmholtz-fmm-proposal.md §6.1, §6.3): the
//! wavenumber k and its kind (CONVENTIONS §3.15), and the leaf rule of proposal §3.7.
//!
//! [`Wavenumber::new`] classifies k = Re k + i Im k as CONVENTIONS §3.15 ("Kernel and
//! factors", "Inputs and outputs") states: Yukawa for Re k = 0, real for Im k = 0,
//! complex otherwise; k = 0, Re k < 0 and Im k < 0 are rejected.
//!
//! The leaf rule (proposal §3.7, Helmholtz CH3.0b): every box that holds a point and is
//! wider than c_leaf λ is refined, in addition to the points-per-leaf rule, so that every
//! leaf that holds a point lies on a low-frequency level and every P2P pair within about
//! 2√3 c_leaf λ. Empty leaves are refined only as completeness and the 2:1 balance
//! require, so they may be wider: they hold no point and no expansion. In
//! `nd-octree`'s key space that width is a level: [`min_leaf_level`] gives the shallowest
//! level whose boxes are at most c_leaf λ wide, which the builder passes to
//! `nd_octree::OctreeOptions::with_min_leaf_level`. The level depends on k, the global
//! domain and c_leaf only, never on a rank's share of the points (proposal §2, constraint
//! 9), so every rank finds the same one. Yukawa has no wavelength and no rule.
//!
//! [`Densities`] (Helmholtz CH3.0c, submodule [`densities`]) fixes the path of an FMM, its
//! leaf chunks and expansion sizes, the complex charges in and the split potentials and
//! gradients out, with the leaf's own P2M and L2P.
//!
//! The module is MPI-free.
//!
//! ```
//! use nd_fmm_exec::geometry::Domain;
//! use nd_fmm_exec::helmholtz::{C_LEAF, Wavenumber, WavenumberKind, min_leaf_level};
//! use nd_octree::PhysicalBox;
//!
//! let domain = Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap();
//!
//! // kD = 500: λ = 2π/500 ≈ 0.0126, and level 7 has boxes 2⁻⁷ ≈ 0.0078 wide.
//! let k = Wavenumber::new(500.0, 0.0).unwrap();
//! assert_eq!(k.kind(), WavenumberKind::Real);
//! assert_eq!(min_leaf_level(&k, &domain, C_LEAF), 7);
//!
//! // Yukawa: no wavelength, no rule.
//! let yukawa = Wavenumber::new(0.0, 500.0).unwrap();
//! assert_eq!(min_leaf_level(&yukawa, &domain, C_LEAF), 0);
//! ```

pub mod densities;

use std::f64::consts::TAU;

use nd_octree::constants::DEEPEST_LEVEL;
use thiserror::Error;

use crate::geometry::Domain;

pub use densities::{ChargeKind, Charges, Densities, DensityError, LeafScratch, Output, Split};

/// The deepest level of a Morton key, 16 (CONVENTIONS §3.12).
const DEEPEST: usize = DEEPEST_LEVEL as usize;

/// The maximum leaf width in wavelengths, c_leaf = 1: decision D11 of
/// docs/helmholtz/roadmap.md (signed off on 2026-10-28; spikes/census/SPIKE_REPORT.md
/// §3), the largest c_leaf with no high-frequency leaf on any reference workload at
/// tolerances 10⁻⁶ and 10⁻⁸.
pub const C_LEAF: f64 = 1.0;

/// The kind of a wavenumber (CONVENTIONS §3.15, "Kernel and factors").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WavenumberKind {
    /// Real k > 0: the oscillating kernel e^{ikr}/(4πr).
    Real,
    /// Complex k with Re k > 0 and Im k > 0: oscillating and decaying like e^{−Im(k) r}.
    Complex,
    /// Yukawa (modified Helmholtz), k = iκ with κ > 0: the kernel e^{−κr}/(4πr).
    Yukawa,
}

/// Why a wavenumber is rejected (CONVENTIONS §3.15, "Kernel and factors").
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum WavenumberError {
    /// A part is NaN or infinite.
    #[error("the wavenumber must be finite, got {re} + {im}i")]
    NotFinite {
        /// Re k.
        re: f64,
        /// Im k.
        im: f64,
    },
    /// k = 0 is the Laplace FMM, not a Helmholtz case.
    #[error("k = 0 is the Laplace kernel, not a Helmholtz case")]
    Zero,
    /// Re k < 0: incoming waves.
    #[error("Re k must not be negative (incoming waves), got {re}")]
    Incoming {
        /// Re k.
        re: f64,
    },
    /// Im k < 0: a kernel that grows with distance.
    #[error("Im k must not be negative (a growing kernel), got {im}")]
    Growing {
        /// Im k.
        im: f64,
    },
}

/// A validated wavenumber k = Re k + i Im k with Re k ≥ 0, Im k ≥ 0 and k ≠ 0, and its
/// kind (CONVENTIONS §3.15).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wavenumber {
    re: f64,
    im: f64,
    kind: WavenumberKind,
}

impl Wavenumber {
    /// Classifies k = `re` + i `im`: [`WavenumberKind::Yukawa`] for Re k = 0,
    /// [`WavenumberKind::Real`] for Im k = 0, [`WavenumberKind::Complex`] otherwise. A
    /// signed zero counts as zero and is stored as +0.
    ///
    /// # Errors
    ///
    /// In this order: [`WavenumberError::NotFinite`] if a part is NaN or infinite;
    /// [`WavenumberError::Zero`] for k = 0; [`WavenumberError::Incoming`] for Re k < 0;
    /// [`WavenumberError::Growing`] for Im k < 0.
    pub fn new(re: f64, im: f64) -> Result<Self, WavenumberError> {
        if !(re.is_finite() && im.is_finite()) {
            return Err(WavenumberError::NotFinite { re, im });
        }
        if re == 0.0 && im == 0.0 {
            return Err(WavenumberError::Zero);
        }
        if re < 0.0 {
            return Err(WavenumberError::Incoming { re });
        }
        if im < 0.0 {
            return Err(WavenumberError::Growing { im });
        }
        // `x + 0.0` turns −0 into +0 and leaves every other value as it is.
        let (re, im) = (re + 0.0, im + 0.0);
        let kind = if re == 0.0 {
            WavenumberKind::Yukawa
        } else if im == 0.0 {
            WavenumberKind::Real
        } else {
            WavenumberKind::Complex
        };
        Ok(Self { re, im, kind })
    }

    /// Re k.
    pub fn re(&self) -> f64 {
        self.re
    }

    /// Im k.
    pub fn im(&self) -> f64 {
        self.im
    }

    /// The kind of k.
    pub fn kind(&self) -> WavenumberKind {
        self.kind
    }

    /// The wavelength λ = 2π / Re k, or `None` for Yukawa. For complex k it is the
    /// wavelength of the oscillation; Im k only damps it.
    pub fn wavelength(&self) -> Option<f64> {
        match self.kind {
            WavenumberKind::Yukawa => None,
            WavenumberKind::Real | WavenumberKind::Complex => Some(TAU / self.re),
        }
    }
}

/// The leaf rule of proposal §3.7: the shallowest level l whose boxes are at most
/// c_leaf λ wide, w 2⁻ˡ ≤ c_leaf λ with w the side of `domain` and λ
/// ([`Wavenumber::wavelength`]) 2π / Re k, capped at level 16; 0 for Yukawa, which has
/// no rule.
///
/// The builder passes it to `nd_octree::OctreeOptions::with_min_leaf_level`, which
/// refines every box that holds a key and lies above it, so no leaf that holds a point is
/// wider than c_leaf λ, unless the octree's `max_level` lies above the level (it is 16 in
/// the FMM) or k w > 2¹⁶ · 2π c_leaf, where the cap leaves the leaves of level 16. Empty
/// leaves may be wider. Level 0, the octree option's default, adds no condition.
/// [`C_LEAF`] is the value of D11.
///
/// # Panics
///
/// Unless `c_leaf` is positive and finite.
pub fn min_leaf_level(k: &Wavenumber, domain: &Domain, c_leaf: f64) -> usize {
    assert!(
        c_leaf > 0.0 && c_leaf.is_finite(),
        "c_leaf must be positive and finite, got {c_leaf}"
    );
    let Some(lambda) = k.wavelength() else {
        return 0;
    };
    let widest = c_leaf * lambda;
    (0..=DEEPEST)
        .find(|&l| domain.side() * 0.5f64.powi(l as i32) <= widest)
        .unwrap_or(DEEPEST)
}

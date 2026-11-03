//! The exponential (plane-wave) M2L of the Helmholtz family on low-frequency levels
//! (docs/design/helmholtz-fmm-proposal.md §5.3, "Exponential (plane-wave) M2L"; Helmholtz
//! H6): the generalisation of CONVENTIONS §3.14 with a quadrature per level.
//!
//! For a target above a source by u = (x, y, z) in box widths, 1 ≤ z ≤ 4 and |x|, |y| ≤ 4
//! (a V pair whose offset along its direction is 2 or 3 boxes, as in §3.14), and κ = k w
//! the wavenumber in box widths (Re κ, Im κ ≥ 0),
//!
//! e^{iκ|u|}/|u| = ∫_Γ e^{−λ z} J₀(t ρ) dλ,  t = √(λ² + κ²),  ρ = √(x² + y²),
//!
//! on a contour Γ from λ = −iκ (t = 0) to +∞, and with the trapezoidal rule in the angle α
//! of J₀(t ρ) = (1/2π) ∫ e^{i t (x cos α + y sin α)} dα a quadrature is
//!
//! e^{iκ|u|}/|u| ≈ Σₖ (wₖ / Nₖ) Σ_{j<Nₖ} e^{−λₖ z + i tₖ (x cos αⱼ + y sin αⱼ)},  αⱼ = 2πj/Nₖ,
//!
//! with complex nodes λₖ and weights wₖ, α counts Nₖ and Fourier counts Fₖ (a source's
//! orders |m| ≤ Fₖ kept at λₖ). Unlike Laplace's single table, the nodes depend on κ, so
//! each level has its own ([`quadrature`]).
//!
//! Helmholtz CH6.1a gives the quadratures ([`quadrature`]); CH6.1c the conversions between
//! expansions and waves, the shifts and the pair translation of one level ([`pair`],
//! CONVENTIONS §3.15, "Exponential M2L"); merge and shift follows in CH6.1d
//! (docs/helmholtz/roadmap.md).

pub mod pair;
pub mod quadrature;

pub use pair::{PlaneWaveLevel, PlaneWaveLevelScratch};
pub use quadrature::{Precision, Quadrature};

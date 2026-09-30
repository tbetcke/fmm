//! # `nd-fmm-math`
//!
//! Mathematical core of the nd-project FMM: scalar trait, index layout, real solid
//! harmonics and rotation blocks. Every other `nd-fmm-*` crate builds on it.
//!
//! ## Conventions
//!
//! All basis functions, phases, storage and scaling are defined in
//! [`docs/CONVENTIONS.md`][conventions], the single source of truth; code cites it as
//! `CONVENTIONS §3.x`. [`CONVENTION_VERSION`] records the version this crate
//! implements. In summary:
//!
//! - Kernel (§3.1): operators expand 1/|x − y|; the factor 1/(4π) is never applied
//!   here.
//! - Coordinates (§3.2): θ from +z, φ from +x towards +y. Harmonics are evaluated by
//!   Cartesian recursion (§3.5), without trigonometric functions.
//! - Basis (§3.3): associated Legendre functions without the Condon–Shortley phase for
//!   m ≥ 0, Racah-type normalisation Rₙᵐ = rⁿ Pₙᵐ(cosθ) e^{imφ} / (n + m)! and
//!   Iₙᵐ = (n − m)! Pₙᵐ(cosθ) e^{imφ} / rⁿ⁺¹, and negative orders from
//!   Xₙ⁻ᵐ = (−1)ᵐ conj(Xₙᵐ) for both families.
//! - Storage (§3.6): (p + 1)² reals at index n² + n + m; slot m = 0 holds the real
//!   value, slot +m the real part and slot −m the imaginary part of the order-m
//!   quantity.
//! - Scaling (§3.7): coefficients are stored as M̃ = M / rⁿ and L̃ = L · rⁿ⁺¹ with r
//!   the box half-width, so translation tables are identical on every level.
//! - Rotations (§3.8): degree-n blocks Dⁿ(Q) with Rₙ(Qx) = Dⁿ(Q) Rₙ(x) in real
//!   storage; Euler angles in the z-y-z convention.
//! - Precision (§3.9): f64 tested for p ≤ 30, f32 for p ≤ 8.
//!
//! ## Contents
//!
//! - [`RealScalar`]: the scalar trait all numeric code is generic over (`f32`, `f64`).
//! - [`Layout`]: the real storage index layout of §3.6.
//! - [`harmonics`]: regular and irregular solid harmonics and their gradients,
//!   evaluated by the Cartesian recursions of §3.5.
//! - [`rotation`]: per-degree rotation blocks Dⁿ(Q) for regular and irregular
//!   harmonics (§3.8) and z-y-z Euler angles.
//!
//! [conventions]: https://github.com/tbetcke/fmm/blob/main/docs/CONVENTIONS.md

pub mod harmonics;
mod layout;
pub mod rotation;
mod scalar;

pub use layout::Layout;
pub use scalar::RealScalar;

/// Version of the conventions in
/// [`docs/CONVENTIONS.md`](https://github.com/tbetcke/fmm/blob/main/docs/CONVENTIONS.md)
/// that this crate implements (CONVENTIONS §3.10).
///
/// Any change to CONVENTIONS §3.1–§3.8, §3.11 or §3.12 bumps this value, which
/// invalidates committed fixtures and cached operator tables. A test checks that it
/// matches the file.
pub const CONVENTION_VERSION: u32 = 1;

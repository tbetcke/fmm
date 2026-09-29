//! # `nd-fmm-ref`
//!
//! Reference operators of the nd-project Laplace FMM: leaf operators, direct O(p⁴)
//! and rotation-based O(p³) translations, P2P and the direct-sum oracle. Every fast
//! path in later phases is tested against this crate, so it favours clarity over
//! speed. It is generic over [`RealScalar`]; f64 is the reference precision and f32
//! is tested for comparison.
//!
//! ## Conventions
//!
//! All basis functions, phases, storage and scaling are defined in
//! [`docs/CONVENTIONS.md`][conventions], the single source of truth; code cites it as
//! `CONVENTIONS §3.x`. This crate relies on:
//!
//! - Kernel (§3.1): operators expand 1/|x − y|.
//! - Storage (§3.6): coefficients of degree ≤ p occupy (p + 1)² reals at index
//!   n² + n + m, as described by [`nd_fmm_math::Layout`].
//! - Scaling (§3.7): an expansion lives in a [`Frame`] with centre c and scaling
//!   radius r, and its coefficients are stored as M̃ = M / rⁿ and L̃ = L · rⁿ⁺¹.
//!
//! ## Rules for every operator
//!
//! - **Scaled coefficients only.** Operators read and write the scaled coefficients of
//!   §3.7; there is no unscaled variant. Each operator takes the [`Frame`] of its input
//!   and of its output expansion, never a bare shift vector.
//! - **Every operator accumulates.** Results are added (+=) into the output, as
//!   `nd-fmm-plan`'s `FmmOperator` does; callers zero the output themselves.
//! - **No 1/(4π).** No operator applies the factor 1/(4π) of §3.1; it is applied once,
//!   by the caller that evaluates the FMM.
//! - **No allocation per call.** Temporaries come from a caller-owned workspace built
//!   for a maximum p.
//!
//! ## Error measures
//!
//! Tests name the measure they use (docs/phase1/README.md, "Error measures"):
//!
//! - Coefficient vectors are compared per degree, in the orthonormal basis of §3.8.
//!   Slot m is weighted by Nₘ for multipole-type data (conjugates of regular harmonics)
//!   and by Nₘ/Sₘ for local-type data (conjugates of irregular harmonics).
//! - Potentials are compared relative to the sum of term magnitudes wherever terms can
//!   cancel.
//! - A truncated result is compared with a reference truncated at the same degree; the
//!   truncation error is checked separately, against its bound.
//!
//! ## Contents
//!
//! - [`Frame`]: the centre and scaling radius of an expansion (§3.7).
//! - [`Workspace`]: caller-owned scratch memory for the operators, built for a
//!   maximum p.
//! - [`leaf`]: the leaf operators P2M, P2L, L2P and M2P (§3.6, §3.7), with gradients.
//! - [`direct`]: the translations M2M, L2L and M2L as direct O(p⁴) sums (§3.11).
//! - [`p2p`]: the near-field operator P2P and the f64 direct-sum oracle (§3.1), with
//!   gradients and self-interaction excluded.
//!
//! [conventions]: https://github.com/tbetcke/fmm/blob/main/docs/CONVENTIONS.md
//! [`RealScalar`]: nd_fmm_math::RealScalar

pub mod direct;
mod frame;
pub mod leaf;
pub mod p2p;
mod workspace;

pub use frame::Frame;
pub use workspace::Workspace;

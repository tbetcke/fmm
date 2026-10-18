//! Plane-wave (exponential) M2L on one uniform level, a spike (Phase 6 T10, C6.5;
//! docs/phase6/T10-plane-wave-spike.md, docs/design/optimisation.md §8.2).
//!
//! - [`quadrature`]: FMM3D's quadratures for separation 1 and their choice per p;
//! - [`geometry`]: the uniform level, its V lists, the six directions and the
//!   merge-and-shift groups;
//! - [`model`]: the cost model of the stop rule;
//! - [`planewave`]: the conversions (multipole → plane waves → local, CONVENTIONS §3.3,
//!   §3.7, §3.11) and the level M2L, direct and with merge-and-shift;
//! - [`baseline`]: the level M2L of `Batched` (T2) and `Rotation`, as the host runs them.
//!
//! SPIKE_REPORT.md has the model, the measurements and the recommendation.

pub mod baseline;
pub mod geometry;
pub mod model;
pub mod planewave;
pub mod quadrature;
// FMM3D's literals keep their 36 digits, as in the Fortran source.
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
mod quadrature_data;

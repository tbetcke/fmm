//! Acceptance tests for the rotation and coaxial tables and the table-driven rotation
//! operators (Phase 2 / T6, C2.3): M2L for all 316 offsets and M2M and L2L for all 8
//! octants against `nd_fmm_ref::rotation` at the canonical frames and against `direct`
//! through the dense tables of T3 and T4, level independence on levels 2, 9 and 16 of a
//! dyadic domain, the on-axis offsets, the distinct-angle and distance counts, the
//! rotation rules on point sources, storage, determinism, f32 tables, random
//! properties, and large degrees in ignored release tests.
//!
//! Every test names its error measure (docs/phase1/README.md and docs/phase2/README.md,
//! "Error measures"). Coefficients are compared per degree in the orthonormal weighting
//! of CONVENTIONS §3.8 (Nₘ for multipoles, Nₘ/Sₘ for locals), relative to the terms
//! |Aᵢₖ xₖ| of the dense table A of the same operator applied to the input x (the
//! "terms" measure of T3, see `support`). Each test prints its worst error with
//! `--nocapture`.

mod common;
mod direct;
mod large;
mod levels;
mod precision;
mod properties;
#[expect(
    dead_code,
    reason = "the point-placement and L2P helpers serve the octant and m2l tests"
)]
#[path = "../support/mod.rs"]
mod support;
mod tables;

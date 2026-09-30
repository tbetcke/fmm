//! Acceptance tests for the cube symmetry group and the 16-class M2L tables (Phase 2 /
//! T5, C2.2): the group and its classes, the coefficient transforms against the
//! harmonics, the z-axis elements, the class form against the dense T4 tables
//! (expanded and applied), the octant tables of T3 from octant 0, f32 and determinism.
//!
//! Every test names its error measure (docs/phase1/README.md and docs/phase2/README.md,
//! "Error measures"). Coefficients are compared per degree in the orthonormal weighting
//! of CONVENTIONS §3.8 (Nₘ for multipoles, Nₘ/Sₘ for locals), relative to the terms of
//! the application (see `support`). The terms of the class form are those of its three
//! products, |T_L(P)| (|A| (|T_M(Pᵀ)| |x|)). Whole matrices are compared per block of
//! output and input degree, relative to the terms matrix |T_L(P)| |A| |T_M(Pᵀ)|
//! (`common::block_error`). Each test prints its worst error with `--nocapture`.

mod classes;
mod common;
mod group;
mod octants;
#[expect(
    dead_code,
    reason = "the domain and L2P helpers serve the octant and m2l tests"
)]
#[path = "../support/mod.rs"]
mod support;
mod transforms;

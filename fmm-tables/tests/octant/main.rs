//! Acceptance tests for the M2M and L2L tables (Phase 2 / T3, C2.1): canonical columns
//! against `direct`, level independence in a dyadic and a generic domain, physical
//! chains, compositions, block structure, hand-worked degrees 0 and 1, the leading
//! block, determinism, f32 tables and random properties.
//!
//! Every test names its error measure (docs/phase1/README.md and docs/phase2/README.md,
//! "Error measures"). Coefficients are compared per degree in the orthonormal weighting
//! of CONVENTIONS §3.8 (Nₘ for multipoles, Nₘ/Sₘ for locals), relative to the terms
//! |Aᵢₖ xₖ| of the table application (see `common`). Each test prints its worst error
//! with `--nocapture`.

mod chains;
mod common;
mod levels;
mod precision;
mod properties;
#[path = "../support/mod.rs"]
mod support;
mod tables;

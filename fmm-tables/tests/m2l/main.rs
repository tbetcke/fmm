//! Acceptance tests for the dense M2L tables (Phase 2 / T4, C2.2): canonical columns
//! against `direct`, level independence on levels 2, 9 and 16 of a dyadic domain,
//! physical chains against `direct::m2l` and against the direct sum within the
//! truncation bound, the monopole column, the leading block, inversion, determinism,
//! f32 tables and random properties.
//!
//! Every test names its error measure (docs/phase1/README.md and docs/phase2/README.md,
//! "Error measures"). Coefficients are compared per degree in the orthonormal weighting
//! of CONVENTIONS §3.8 (Nₘ/Sₘ for the local output), relative to the terms |Aᵢₖ xₖ| of
//! the table application (see `support`). Each test prints its worst error with
//! `--nocapture`.

mod chains;
mod common;
mod levels;
mod precision;
mod properties;
#[path = "../support/mod.rs"]
mod support;
mod tables;

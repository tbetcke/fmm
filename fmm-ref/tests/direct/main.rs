//! Acceptance tests for the direct translations M2M, L2L and M2L (Phase 1 / T5):
//! exactness and composition of M2M and L2L, M2L against the direct sum within a
//! truncation bound and its structure, zero and axis shifts, hand-worked degrees 0
//! and 1, linearity and accumulation, f32 against f64, argument checks and random
//! properties. Module `rotation` holds the acceptance tests of the rotation-based
//! translations (Phase 1 / T6), which the direct ones check.
//!
//! Every test names its error measure (docs/phase1/README.md, "Error measures").
//! Coefficients are compared per degree in the orthonormal weighting of CONVENTIONS
//! §3.8 (Nₘ for multipoles, Nₘ/Sₘ for locals), relative to the magnitude of the terms
//! that the translation sums (see `common`). Each test prints its worst error with
//! `--nocapture`.

mod common;
mod exactness;
mod linearity;
mod m2l;
mod panics;
mod precision;
mod properties;
mod rotation;
mod special;

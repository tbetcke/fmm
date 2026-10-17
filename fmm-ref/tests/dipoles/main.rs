//! Acceptance tests for the dipole operators (Phase 6 T8, C6.4; CONVENTIONS §3.7,
//! "Dipole sources"): `leaf::{p2m_dipoles, p2l_dipoles}`, `p2p::p2p_dipoles` and
//! `p2p::direct_sum_dipoles`, against the mpmath fixtures of tools/fixtures/gen_dipoles.py,
//! against central differences of the charge operators in the source position, through
//! the series (M2P and L2P) against the direct sum, and for accumulation, coincident
//! pairs, f32 and the panics.
//!
//! Every test names its error measure (docs/phase1/README.md, "Error measures").

mod common;
mod differences;
mod fixtures;
mod panics;
mod properties;
mod series;

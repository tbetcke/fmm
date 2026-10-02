//! Acceptance tests for `P2pKernel::evaluate` (Phase 3S T3) on every ISA this machine
//! offers: against `nd_fmm_ref::p2p::p2p` bit for bit, coincident points, empty inputs,
//! accumulation, argument checks, and chunk and target-position invariance.
//!
//! Error measure: exact equality of the bits of every output (potential and gradient
//! components), compared as the f64 widening of each value. The inputs lie on the grid
//! 2⁻¹⁰ ℤ (charges and initial outputs on similar grids), exact in f32 and f64, so a
//! nonzero separation component is at least 2⁻¹⁰ and r² = 0 exactly when the target
//! and the source coincide: the kernel's rule and the reference's agree on them
//! (CONVENTIONS §3.13, "Fast kernels").
//!
//! Every test prints the ISAs it ran; run with `--show-output` to see it.

mod coincident;
mod common;
mod inputs;
mod invariance;
mod panics;
mod reference;

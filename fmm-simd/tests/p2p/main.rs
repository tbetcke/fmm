//! Acceptance tests for `P2pKernel::evaluate` (Phase 3S T5, C3S.4) on every ISA this
//! machine offers, in f32 and f64, potential only and with gradients.
//!
//! Error measures:
//! - *Terms* (`terms`, `domain`): one source and one target; the potential relative to
//!   the reference's term, each gradient component relative to |q| / r², in units of
//!   u_T (2⁻²⁴, 2⁻⁵³). Tolerances 8 u_T and 16 u_T (design §3, requirement 2).
//! - *Sums* (`sums`, `coincident`, `inputs`, `domain`): the error against
//!   `nd_fmm_ref::p2p::direct_sum` on the exactly widened inputs, from the same initial
//!   outputs, relative per target to the initial value's magnitude plus the sum of
//!   term magnitudes over the pairs with xᵢ ≠ yⱼ: Σⱼ |qⱼ| / r for the potential and
//!   Σⱼ |qⱼ| / r² for each gradient component. Tolerances 1e-6 (f32) and 1e-14 (f64),
//!   or twice the error of `nd_fmm_ref::p2p::p2p` on the same inputs where that is
//!   larger (`common::Oracle::tolerance`; signed off in Phase 3S T5).
//! - *Cross-ISA* (`cross_isa`): the difference of two ISAs in the same measures, within
//!   twice the tolerances.
//! - *Bits* (`invariance`, `inputs`, `reference`, `coincident`): exact equality of the
//!   bits of every output, compared as the f64 widening of each value. The scalar path
//!   equals `nd_fmm_ref::p2p::p2p` bit for bit; on grid-valued inputs (2⁻¹⁰ ℤ) r² = 0
//!   exactly when the points coincide, so both rules skip the same pairs (CONVENTIONS
//!   §3.13, "Fast kernels").
//!
//! The large sweeps are `#[ignore]`d, for a release run:
//! `cargo test -p nd-fmm-simd --release -- --ignored --show-output`.
//!
//! Every test prints the ISAs it ran, and the accuracy tests the largest errors per
//! ISA; run with `--show-output` to see it.

mod coincident;
mod common;
mod cross_isa;
mod domain;
mod inputs;
mod invariance;
mod panics;
mod reference;
mod sums;
mod terms;

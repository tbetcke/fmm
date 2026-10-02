//! Acceptance tests for the kernels' inverse square root, `rsqrt::rsqrt_slice`
//! (Phase 3S T4, C3S.3), on every ISA this machine offers: the 4 u_T contract of
//! docs/design/simd-p2p.md §4.3, scale invariance, lane independence and argument
//! checks.
//!
//! Error measure: the relative error |y √x − 1| of a result y, in units of the unit
//! roundoff u_T (2⁻²⁴, 2⁻⁵³). For f32 inputs the reference is 1/√x in f64; for f64
//! inputs it is the residual x y² − 1, formed in double-double arithmetic with
//! error-free products, so the error is exact to a relative 2⁻⁵⁰ or so. The contract
//! holds on the kernel domain 2⁻¹⁰⁸ ≤ x ≤ 2⁷ of CONVENTIONS §3.13, and the result at
//! x = 0 is exactly +0.
//!
//! The exhaustive f32 check and the 10⁷-sample f64 check are `#[ignore]`d, for a
//! release run: `cargo test -p nd-fmm-simd --release -- --ignored --show-output`. The
//! debug run checks every 97th f32 and 10⁵ f64 samples.
//!
//! Every test prints the ISAs it ran, and the contract tests the largest error per ISA
//! and precision; run with `--show-output` to see it. Accuracy counts only from real
//! hardware, not from Rosetta 2, which emulates the x86 estimate.

mod common;
mod contract;
mod invariance;

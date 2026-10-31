//! Acceptance tests for the Helmholtz-family kernels' exponential, `exp::exp_slice`
//! (Helmholtz CH2.4a), on every ISA this machine offers: the 4 u_T contract of
//! docs/design/simd-p2p.md §4.9 on the domain [x_min, 0], the values at ±0 and below the
//! domain, bit identity across ISAs, lane independence and argument checks.
//!
//! Error measure: the relative error |y − e^x| / e^x of a result y, in units of the unit
//! roundoff u_T (2⁻²⁴, 2⁻⁵³). For f32 inputs the reference is `f64::exp`, whose own error
//! (about 2⁻⁵³) is negligible against u₃₂; for f64 inputs it is e^x in double-double
//! arithmetic (`common::exp_dd`: the reduction by ln 2 in triple-double, a Taylor series
//! in double-double), exact to a relative 2⁻¹⁰⁰ or so.
//!
//! The exhaustive f32 check and the 10⁷-sample f64 check are `#[ignore]`d, for a release
//! run: `cargo test -p nd-fmm-simd --release -- --ignored --show-output`. The debug run
//! checks a strided subset of the f32 domain and 10⁵ f64 samples.
//!
//! Every test prints the ISAs it ran, and the contract tests the largest error per ISA
//! and precision; run with `--show-output` to see it. Accuracy counts only from real
//! hardware, not from Rosetta 2.

mod common;
mod contract;
mod invariance;

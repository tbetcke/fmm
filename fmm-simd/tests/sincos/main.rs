//! Acceptance tests for the Helmholtz P2P kernel's sine and cosine,
//! `sincos::sincos_slice` (Helmholtz CH3.4a), on every ISA this machine offers: the 4 u_T
//! contract of docs/design/simd-p2p.md §4.11 on |θ| ≤ 2¹⁰, the symmetries and the values
//! at ±0, bit identity across ISAs, lane independence and argument checks.
//!
//! Error measure: the absolute error |y − sin θ| (and |y − cos θ|) of a result y, in units
//! of the unit roundoff u_T (2⁻²⁴, 2⁻⁵³); absolute because the P2P term q e^{iθ}/r is
//! measured relative to |q|/r. For f32 inputs the reference is `f64::sin` and `f64::cos`,
//! whose own errors (about 2⁻⁵³) are negligible against u₃₂; for f64 inputs it is sin θ and
//! cos θ in double-double arithmetic (`common::sincos_dd`: the reduction by π in
//! triple-double, Taylor series in double-double), exact to an absolute 2⁻¹⁰⁰ or so.
//!
//! The exhaustive f32 check and the 10⁷-sample f64 check are `#[ignore]`d, for a release
//! run: `cargo test -p nd-fmm-simd --release -- --ignored --show-output`. The debug run
//! checks a strided subset of the f32 domain and 10⁵ f64 samples.
//!
//! Every test prints the ISAs it ran, and the contract tests the largest errors per ISA
//! and precision; run with `--show-output` to see them. Accuracy counts only from real
//! hardware, not from Rosetta 2.

mod common;
mod contract;
mod invariance;

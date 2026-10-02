//! # `nd-fmm-validate`
//!
//! Development tooling that measures the nd-project FMM rather than testing it: error
//! metrics, seeded point distributions, accuracy sweeps and (in the examples) timing
//! reports. It grows with every phase; in Phase 1 it measures the reference operators
//! of `nd-fmm-ref` against its direct-sum oracle, and in Phase 2 the operator tables of
//! `nd-fmm-tables`: their cost, and the same accuracy sweep run through them.
//!
//! No library crate depends on this crate, not even as a dev-dependency: once it
//! depends on the executor crates, that would create dependency cycles. Everything
//! here is seeded and deterministic; only the `timing` and `tables` examples measure
//! wall-clock time, and nothing asserts timings.
//!
//! ## Conventions
//!
//! All basis functions, storage and scaling are those of
//! [`docs/CONVENTIONS.md`][conventions]; code cites it as `CONVENTIONS §3.x`. The
//! metrics of [`metrics`] weight coefficients in the orthonormal basis of §3.8 and read
//! them in the real storage of §3.6.
//!
//! ## Error measures
//!
//! Every accuracy figure names its measure (docs/phase1/README.md, "Error measures");
//! [`metrics`] explains when to use which:
//!
//! - relative L2 and max error of potentials and gradients, for reported accuracy
//!   against the direct sum;
//! - error relative to the sum of term magnitudes, wherever terms can cancel;
//! - per-degree coefficient error in the orthonormal weighting of §3.8 (Nₘ for
//!   multipole-type, Nₘ/Sₘ for local-type data), for coefficient vectors.
//!
//! ## Contents
//!
//! - [`SplitMix64`]: a small seeded generator, the algorithm of the `nd-fmm-math`
//!   tests, so no `rand` dependency.
//! - [`points`]: uniform points in a cube and a ball, and on a sphere surface; the
//!   Plummer sphere and Gaussian clusters, truncated; random charges.
//! - [`metrics`]: the error measures above.
//! - [`accuracy`]: the single-translation accuracy sweep of the operator chains
//!   against `nd_fmm_ref::p2p::direct_sum`, with the translations of `nd-fmm-ref` or
//!   of the tables, run by the `accuracy` example.
//! - [`bench`](mod@bench): the helpers of the timing reports (median time per call, fitted
//!   exponent, crossover, machine description).
//! - [`fmm_accuracy`]: the complete FMM of `nd-fmm-exec` against the direct sum, over
//!   several charge vectors, on a uniform tree (C3.2) and on the adaptive trees of the
//!   sphere surface, the Plummer sphere and Gaussian clusters (C3.3), run by the
//!   `fmm_accuracy` example.
//! - [`calibration`]: the degree p against the accuracy of the complete FMM, per
//!   precision and distribution, the smallest p for each target accuracy, and a
//!   leaf-size study (C3.4), run by the `calibrate` example.
//!
//! The examples `accuracy`, `timing`, `tables`, `fmm_accuracy` and `calibrate` print
//! Markdown reports on stdout:
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --example accuracy
//! cargo run --release -p nd-fmm-validate --example accuracy -- --tables
//! cargo run --release -p nd-fmm-validate --example timing
//! cargo run --release -p nd-fmm-validate --example tables
//! cargo run --release -p nd-fmm-validate --example fmm_accuracy
//! cargo run --release -p nd-fmm-validate --example fmm_accuracy -- --distribution plummer
//! cargo run --release -p nd-fmm-validate --example calibrate -- --threads 8
//! ```
//!
//! Since Phase 3 the crate depends on `nd-fmm-exec`, and so on MPI: building it, and
//! `cargo test -p nd-fmm-validate`, need an MPI installation. `fmm_accuracy` and
//! `calibrate` run on one rank.
//!
//! [conventions]: https://github.com/tbetcke/fmm/blob/main/docs/CONVENTIONS.md

pub mod accuracy;
pub mod bench;
pub mod calibration;
pub mod fmm_accuracy;
pub mod metrics;
pub mod points;
mod rng;

pub use rng::SplitMix64;

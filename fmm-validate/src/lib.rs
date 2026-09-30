//! # `nd-fmm-validate`
//!
//! Development tooling that measures the nd-project FMM rather than testing it: error
//! metrics, seeded point distributions, accuracy sweeps and (in the examples) timing
//! reports. It grows with every phase; in Phase 1 it measures the reference operators
//! of `nd-fmm-ref` against its direct-sum oracle.
//!
//! No library crate depends on this crate, not even as a dev-dependency: once it
//! depends on the executor crates, that would create dependency cycles. Everything
//! here is seeded and deterministic; only the timing example measures wall-clock time,
//! and nothing asserts timings.
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
//! - [`points`]: uniform points in a cube and a ball, and on a sphere surface; random
//!   charges.
//! - [`metrics`]: the error measures above.
//! - [`accuracy`]: the single-translation accuracy sweep of the operator chains
//!   against `nd_fmm_ref::p2p::direct_sum`, run by the `accuracy` example.
//!
//! The examples `accuracy` and `timing` print Markdown reports on stdout:
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --example accuracy
//! cargo run --release -p nd-fmm-validate --example timing
//! ```
//!
//! [conventions]: https://github.com/tbetcke/fmm/blob/main/docs/CONVENTIONS.md

pub mod accuracy;
pub mod metrics;
pub mod points;
mod rng;

pub use rng::SplitMix64;

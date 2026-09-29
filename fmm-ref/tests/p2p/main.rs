//! Acceptance tests for P2P and the direct-sum oracle (Phase 1 / T4): brute force,
//! cancellation, coincident points, gradients, symmetry, accumulation and empty
//! inputs, f32 against f64, argument checks and random properties.
//!
//! Every test names its error measure (docs/phase1/README.md, "Error measures").
//! Potentials are compared relative to the target's sum of term magnitudes
//! Σⱼ |qⱼ| / |xᵢ − yⱼ|, gradient components relative to Σⱼ |qⱼ| / |xᵢ − yⱼ|². The
//! high-precision reference is double-double arithmetic (`common::dd_reference`).

mod brute;
mod cancellation;
mod coincident;
mod common;
mod gradient;
mod inputs;
mod panics;
mod properties;
mod single;
mod symmetry;

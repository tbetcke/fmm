//! Acceptance tests for the Helmholtz direct-sum oracle and P2P reference (Helmholtz
//! CH1.3a, CONVENTIONS §3.15 "Kernel and factors"): FMM3D's fixture, the limit k → 0,
//! Yukawa two ways, properties and argument checks.
//!
//! Every test names its error measure (docs/phase1/README.md, "Error measures"). Per
//! target, potentials are compared relative to the sum of term magnitudes Σⱼ |qⱼ| |e^{ikrⱼ}|/rⱼ
//! and gradients (the Euclidean norm of the complex 3-vector) relative to
//! Σⱼ |qⱼ| |ikrⱼ − 1| |e^{ikrⱼ}|/rⱼ², which bound the terms; whole cases are compared in
//! the relative ℓ2 norm of proposal §9.1.

mod common;
mod fixtures;
mod limit;
mod panics;
mod properties;
mod yukawa;

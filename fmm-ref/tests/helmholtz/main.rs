//! Acceptance tests for the Helmholtz reference operators (CONVENTIONS §3.15).
//!
//! The direct-sum oracle and P2P (Helmholtz CH1.3a, "Kernel and factors"): FMM3D's
//! fixture, the limit k → 0, Yukawa two ways, properties and argument checks. Per target,
//! potentials are compared relative to the sum of term magnitudes Σⱼ |qⱼ| |e^{ikrⱼ}|/rⱼ
//! and gradients (the Euclidean norm of the complex 3-vector) relative to
//! Σⱼ |qⱼ| |ikrⱼ − 1| |e^{ikrⱼ}|/rⱼ², which bound the terms; whole cases are compared in
//! the relative ℓ2 norm of proposal §9.1.
//!
//! The leaf operators (Helmholtz CH1.3b, "Radial functions and scaling" and "Storage"),
//! the `leaf_` modules: the series P2M → M2P and P2L → L2P against the direct sum, the
//! stored form against closed forms, strong damping, f32, the limit k → 0 against
//! Laplace's leaf operators under the diagonal map of proposal §3.4, and Yukawa two ways.
//!
//! Every test names its error measure (docs/phase1/README.md, "Error measures").

mod common;
mod fixtures;
mod leaf_common;
mod leaf_limit;
mod leaf_series;
mod leaf_stored;
mod leaf_yukawa;
mod limit;
mod panics;
mod properties;
mod translation_common;
mod translation_direct;
mod translation_limit;
mod translation_series;
mod translation_yukawa;
mod yukawa;

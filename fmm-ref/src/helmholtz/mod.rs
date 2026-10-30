//! Reference operators of the Helmholtz family (CONVENTIONS §3.15; Helmholtz CH1.3), the
//! trusted paths that the Helmholtz and Yukawa fast paths are tested against
//! (docs/design/helmholtz-fmm-proposal.md, §2 constraint 6 and §9).
//!
//! ## Conventions
//!
//! CONVENTIONS §3.15 defines the family; this module relies on:
//!
//! - **Kernel** ("Kernel and factors"): G_k(x, y) = e^{ik|x−y|}/(4π|x − y|) with
//!   Re k ≥ 0 and Im k ≥ 0: real k, complex k, and Yukawa (modified Helmholtz) k = iκ,
//!   κ > 0, where G_k = e^{−κr}/(4πr). As for Laplace (§3.1), no operator applies 1/(4π):
//!   every operator accumulates the potential of e^{ikr}/r, written φ̃ = 4πφ, and its
//!   gradient with respect to the target.
//! - **Charges** are real or complex, and a complex charge enters as it is (the real
//!   orthonormal basis is never conjugated). Yukawa with real charges is real throughout.
//! - **Coincident pairs** are excluded as in §3.13: a pair with x = y exactly contributes
//!   nothing.
//!
//! The complex type is [`nd_fmm_math::Complex`]. The oracle takes and returns complex
//! values per point, for clarity; the operators' storage of §3.15 (two real planes per
//! complex expansion) belongs to the operators that follow (Helmholtz CH1.3b, CH1.3c).
//!
//! ## Contents
//!
//! - [`p2p`]: the near-field operator and the f64 direct-sum oracle, for complex k and
//!   complex charges, and Yukawa's real kernel for real charges (Helmholtz CH1.3a).

pub mod p2p;

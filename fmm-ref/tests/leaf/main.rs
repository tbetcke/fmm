//! Acceptance tests for the leaf operators P2M, P2L, L2P and M2P (Phase 1 / T3):
//! real storage, the same-degree Legendre series, the truncation bound against the
//! exact kernel, linearity and accumulation, f32 against f64, and random properties.
//!
//! Frames use the radii {1, 0.37, 2⁻¹⁶} with centres away from the origin, so that the
//! scaling of CONVENTIONS §3.7 is exercised. Every test names its error measure
//! (docs/phase1/README.md, "Error measures").

mod common;
mod linearity;
mod panics;
mod properties;
mod series;
mod single;
mod storage;
mod truncation;

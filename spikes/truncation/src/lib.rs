//! The truncation degree of the Helmholtz FMM under realistic geometry, a spike
//! (Helmholtz CH0.7, S7; docs/helmholtz/tasks/CH0.7.md, proposal §3.5, §9 item 5).
//!
//! Every operator of the chains P2M → M2M → M2L → L2L → L2P runs in double-double
//! precision ([`real::Dd`], about 32 digits) against the direct sum in the same
//! precision, so that the error measured is the truncation's alone; the same code runs in
//! f64 for comparison.
//!
//! - [`real`], [`complex`]: the scalar types;
//! - [`radial`], [`harmonics`]: jₙ, hₙ and the real orthonormal Yₙᵐ (S2's basis);
//! - [`rotation`]: S1's y-rotation blocks, the z rotation, the rotation onto +z;
//! - [`coaxial`]: S2's coaxial tables by recursion and by projection;
//! - [`chain`]: the operators, the direct sum and the error norms;
//! - [`geometry`]: the 316 offsets, children and the point placements;
//! - [`calibrate`]: the chains C1, C2, C3m, C3l over offsets and placements, and the
//!   search for the smallest p.
//!
//! SPIKE_REPORT.md has the method, the measurements and the answers.

pub mod calibrate;
pub mod chain;
pub mod coaxial;
pub mod complex;
pub mod geometry;
pub mod harmonics;
pub mod radial;
pub mod real;
pub mod rotation;

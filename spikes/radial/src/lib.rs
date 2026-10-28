//! The radial functions of the Helmholtz FMM, a spike (Helmholtz CH0.3, S3;
//! docs/helmholtz/tasks/CH0.3.md, proposal §3.1, §3.4, §5.1, §5.5, §5.6, §8 R4).
//!
//! - [`radial`]: jₙ and hₙ = hₙ⁽¹⁾ of complex argument, iₙ and kₙ of real argument, and
//!   their derivatives, with a power scale and the exponential factor removed;
//! - [`ranges`]: the degrees of the octree's levels and the scale candidates;
//! - [`complex`]: a minimal complex type.
//!
//! SPIKE_REPORT.md has the method, the measurements and the answers.

pub mod complex;
pub mod radial;
pub mod ranges;

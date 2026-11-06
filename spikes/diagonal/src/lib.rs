//! The stability of the diagonal M2L of the Helmholtz FMM, a spike (Helmholtz CH0.6, S6;
//! docs/helmholtz/tasks/CH0.6.md, proposal §3.6, §5.4 and Appendix B).
//!
//! The diagonal M2L of proposal §5.4 takes a box's multipole to its signature on a sphere
//! grid (S → F), multiplies it by the transfer function of the offset, and projects the
//! product back onto a local expansion (F → R). In exact arithmetic it is the truncated
//! M2L that RCR applies; in floating point the transfer function's terms grow with the
//! degree, and their rounding lands in the result. This spike runs the chain in f64 and f32
//! with transfer functions rounded from double-double, on the 34 offset classes, with
//! multipoles from realistic sources, against S7's double-double RCR and the direct sum
//! (`nd_fmm_spike_truncation`), and fits the bound of proposal §3.6.
//!
//! - [`grid`]: the signature grid (Gauss–Legendre in cos θ, uniform in φ) and its tables;
//! - [`signature`]: S → F and F → R in any precision;
//! - [`transfer`]: the transfer function in double-double, its bound and its interpolation
//!   in ŝ · t̂;
//! - [`classes`]: the 34 reflection classes of the 316 offsets and their grid permutations;
//! - [`rules`]: the degree rules (proposal §3.5's formula, S7's p*, CH3.3's calibrated);
//! - [`run`]: one unit of the sweep, every class, placement and variant at one (k, p).
//!
//! SPIKE_REPORT.md has the method, the measurements and the answers.

pub mod classes;
pub mod grid;
pub mod rules;
pub mod run;
pub mod signature;
pub mod transfer;

//! The workload census of the Helmholtz FMM (Helmholtz CH0.8a, spike S8; proposal §10
//! CH0.8, docs/helmholtz/tasks/CH0.8a.md).
//!
//! The trees of the reference workloads W1–W4 (proposal §9.1) are built by `nd-octree`
//! with a prototype maximum leaf width ([`tree`]), their interaction lists by
//! `nd-fmm-plan`, and counted per level ([`census`]). The model of proposal §3.5–§3.6
//! and §7 ([`model`]) turns the counts into degrees, regimes, memory and operation
//! counts. No FMM is run.
//!
//! Lengths are in units of the root cube's side D = 1, so the wavenumber k is kD.

pub mod census;
pub mod model;
pub mod tree;
pub mod workloads;

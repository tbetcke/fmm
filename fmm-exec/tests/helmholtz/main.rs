//! Acceptance tests of the `helmholtz` module (Helmholtz CH3.0b, CH3.0c, CH3.1a): the
//! classification of k (CONVENTIONS §3.15, "Kernel and factors" and "Inputs and
//! outputs"), the leaf rule of docs/design/helmholtz-fmm-proposal.md §3.7, the complex
//! densities (§3.15, "Storage" and "Inputs and outputs"), the degree rule (spike S7),
//! the operator's bodies pair by pair, its P2P on every ISA (CH3.4c), its Dense M2L
//! against Rotation (CH3.2a) and the α counts of its exponential M2L at the largest V
//! offsets (CH6.2g), and the V rule's cost model (CH4.0d2).
//!
//! None of these tests initialises MPI. The trees of the reference workloads under the
//! rule are tested in `nd-fmm-validate` (`tests/helmholtz_leaves.rs`).

mod common;
mod degree;
mod dense;
mod densities;
mod leaf_rule;
mod operator;
mod p2p;
mod plane_wave;
mod plane_wave_reach;
mod v_crossover;
mod wavenumber;

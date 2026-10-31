//! Acceptance tests of the `helmholtz` module (Helmholtz CH3.0b): the classification of k
//! (CONVENTIONS §3.15, "Kernel and factors" and "Inputs and outputs") and the leaf rule
//! of docs/design/helmholtz-fmm-proposal.md §3.7.
//!
//! None of these tests initialises MPI. The trees of the reference workloads under the
//! rule are tested in `nd-fmm-validate` (`tests/helmholtz_leaves.rs`).

mod leaf_rule;
mod wavenumber;

//! Acceptance tests of the `helmholtz` module (Helmholtz CH3.0b, CH3.0c): the
//! classification of k (CONVENTIONS §3.15, "Kernel and factors" and "Inputs and
//! outputs"), the leaf rule of docs/design/helmholtz-fmm-proposal.md §3.7, and the complex
//! densities (§3.15, "Storage" and "Inputs and outputs").
//!
//! None of these tests initialises MPI. The trees of the reference workloads under the
//! rule are tested in `nd-fmm-validate` (`tests/helmholtz_leaves.rs`).

mod densities;
mod leaf_rule;
mod wavenumber;

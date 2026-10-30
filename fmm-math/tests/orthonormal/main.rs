//! Acceptance tests of the real orthonormal harmonics and their rotation blocks
//! (Helmholtz CH1.2a, docs/helmholtz/tasks/CH1.2a.md; CONVENTIONS §3.15, "Angular basis"
//! and "Rotations"). The tests at degrees above 300 are ignored in the default run; run
//! them with `cargo test -p nd-fmm-math --release --test orthonormal -- --ignored`.

#[allow(
    dead_code,
    reason = "shared with tests/harmonics; this suite uses part of it"
)]
#[path = "../harmonics/common.rs"]
mod common;
#[allow(
    dead_code,
    reason = "shared with tests/rotation; this suite uses part of it"
)]
#[path = "../rotation/support.rs"]
mod support;

mod fixtures;
mod harmonics;
mod rotation;

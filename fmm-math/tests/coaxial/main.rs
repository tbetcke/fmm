//! Acceptance tests of the coaxial translation coefficients of the Helmholtz family
//! (Helmholtz CH1.2c, docs/helmholtz/tasks/CH1.2c.md; CONVENTIONS §3.15, "Coaxial
//! translations"). The numbers are those of the plan's "Tests that prove it". The tests
//! that build tables beyond p = 100 are `#[ignore]`d for debug builds; run them with
//! `cargo test -p nd-fmm-math --release --test coaxial -- --ignored`.

mod fixtures;
mod high_frequency;
mod identities;
mod support;

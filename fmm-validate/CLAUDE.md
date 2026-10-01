# nd-fmm-validate

Purpose: error metrics, point distributions, accuracy sweeps and benchmarks for the
nd-fmm-* crates; development tooling, never a dependency of a library crate.
Phase and components: created in Phase 1 (task T7 in docs/phase1/); grows in every
later phase (C3.2–C3.4 calibration, Phase 4 benchmarks).

## Rules
- `publish = false`.
- No library crate depends on this crate, not even as a dev-dependency: that would
  create dependency cycles once it depends on nd-fmm-exec. Crates keep their own small
  test helpers.
- Examples measure and report; they never assert timings. Tests cover the metrics and
  distributions themselves and run small smoke sweeps only.
- Seeded and deterministic, apart from timings. Do not commit CSV or other output
  (`.gitignore` does not cover `*.csv`).
- Before finishing: `cargo clippy -p nd-fmm-validate --all-targets -- -D warnings`
  and `cargo test -p nd-fmm-validate` must pass.

## Allowed dependencies
Phase 1: nd-fmm-math, nd-fmm-ref. Phase 2: nd-fmm-tables. Phase 3: nd-fmm-exec (and so
MPI: `mpi` from [workspace.dependencies]). Later phases add the crates they validate.

Since Phase 3, building this crate, and so `cargo test -p nd-fmm-validate`, needs an MPI
installation. Tests that run an FMM initialise MPI, at most one test per test
executable (`tests/fmm_accuracy.rs`).

Anything else (criterion, rand, plotting) needs a note in the PR.

## Test oracle
`nd_fmm_ref::p2p::direct_sum` for every accuracy figure; hand-made examples for the
metrics.

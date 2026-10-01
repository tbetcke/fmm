# Phase 3 / T9 — nd-fmm-exec: the FMM object and the uniform-tree FMM (C3.2)

This task puts the pieces together. `FmmBuilder` bins the user's points into the
leaves of an octree and builds `LaplaceOperator` and the new `nd-fmm-plan`
evaluator. `Fmm::evaluate`
runs the complete pass and returns potentials and gradients in the user's order, with
1/(4π) applied once. The C3.2 gate measures it on a uniform tree against the direct
sum.

Read first: docs/CONVENTIONS.md §3.1 and §3.13; docs/phase3/README.md ("Design
decisions", all; "Exit gate", C3.2); fmm-exec/CLAUDE.md; the T3 and T8 code; the docs of the new nd-fmm-plan evaluator
and leaf stores (T5, T6: per-leaf counts, source, target-input and target-output
access, the public stage methods and `reset()`); docs/design/fmm-plan-redesign.md §9
(the redistribution API that C5.1 will add, which `Fmm` must not contradict);
the docs of `nd_octree::{Octree, OctreeOptions, points_to_morton,
compute_global_bounding_box}` and `Octree::local_leaf`; fmm-validate/CLAUDE.md and the
nd-fmm-validate code (`metrics`, `points`, the `accuracy` example); design §7, "Single-
translation accuracy".

Do:
- Add rlst (`workspace = true`) to nd-fmm-exec, for `points_to_morton` and
  `compute_global_bounding_box`.
- `FmmBuilder<T>` and `Fmm<'o, T>` (the lifetime follows whatever the evaluator
  borrows; an owning design is fine if it stays simple):
  - builder settings: p, `M2lStrategy`, gradients on or off, `max_level`,
    `max_points_per_leaf` (passed to `OctreeOptions::with_max_fine_keys`), an optional
    domain and an optional table cache directory. Document the defaults (suggested:
    `Auto`, no gradients, `max_level` 16, 64 points per leaf, no cache);
  - `build(sources: &[[f64; 3]], targets: &[[f64; 3]], comm) -> Result<Fmm, FmmError>`,
    collective. It:
    - takes the domain from `compute_global_bounding_box` over sources and targets, or
      checks the supplied one (`geometry::Domain`, and every point strictly inside);
    - builds one octree from the union of both point sets, with
      `with_ghost_children(true)`, and the plan's box index and lists;
    - finds every point's leaf with `points_to_morton` at `DEEPEST_LEVEL` and
      `Octree::local_leaf`, and returns `FmmError::PointsNotOwned` on every rank (one
      all-reduce) if any rank has a point owned elsewhere;
    - sorts points into leaf order, stably, keeping the permutation;
    - converts them to leaf-scaled coordinates (`geometry::leaf_coordinates`) and
      writes the target positions into the target-input store once;
    - all-reduces the largest leaf occupancy, then builds the tables, the operator and
      the evaluator with the per-leaf counts;
  - `evaluate(charges: &[T]) -> Output<T>`, collective: writes the source chunks
    (coordinates, then charges, §3.13), resets the evaluator and runs it stage by
    stage, converts the target output with φ = φ̂ / (4π r_t) and ∇φ = ĝ / (4π r_t²), and returns them in
    the user's target order;
  - `Output<T> { potential, gradient: Option<_>, timings: StageTimings }`, where
    `StageTimings` holds the wall time of each evaluator stage
    (`std::time::Instant`), for reports only;
  - accessors for the octree, the number of leaves, levels and points per leaf, the
    interaction-list sizes, and the cache outcomes.
- `FmmError` (`thiserror`): invalid domain, a point outside the domain, points not
  owned, a charge vector of the wrong length, and invalid settings: p > 20, which §3.9
  does not cover for M2L. f32 with p > 8 is allowed and documented as beyond its
  useful range (design §4).
- Every error that depends on one rank's input is agreed by all ranks before any
  further collective, and no communication sits inside a rank-dependent branch.
- Crate docs: a `no_run` example of a complete evaluation.

Tests that define done:
- tests/mpi_exec.rs, in the T8 `cases` list (no new MPI-initialising test), each on
  one rank in CI. Small N, debug mode, against `direct_sum` over all targets, with the
  error measures of docs/phase3/README.md:
  - uniform cube, N = 2,000, sources equal to targets, p ∈ {2, 4, 6, 8}, every
    strategy: the error decreases with p, and at p = 8 is below 1e-4 (φ, relative L2);
    `Dense`, `Classes` and `Rotation` agree to 1e-12 (relative L2 between them);
  - distinct sources and targets, and sources a subset of targets;
  - gradients against the `direct_sum` gradient;
  - a uniform tree: `max_level` 3 and `max_points_per_leaf` 1 with enough points;
    assert that every leaf lies on level 3 and the W and X lists are empty;
  - a single leaf (the root only): the result equals the direct sum to 1e-14;
  - no targets on this rank; a supplied non-cubic domain and a point outside a
    supplied domain each give the right `FmmError`;
  - f32 at p = 6 against f64;
  - repeatability: `evaluate` twice gives bit-identical output, and a second charge
    vector gives the same result as a fresh build with it.
- tests/accuracy.rs, its own executable with one `#[ignore]` MPI test (release): the
  C3.2 gate in its smallest form, a uniform level-4 tree with N = 10⁵ at p = 3 and 8,
  1,000 sampled targets, f64, within twice the prediction. This keeps the gate next to
  the code; the full report is the example below.
- nd-fmm-validate:
  - add nd-fmm-exec (`workspace = true`) and, in fmm-validate/CLAUDE.md, "Phase 3:
    nd-fmm-exec (and so MPI)" to the allowed dependencies. Note there that
    `cargo test -p nd-fmm-validate` now needs an MPI installation to build;
  - example `fmm_accuracy` (release; one rank; not registered with
    templated-examples): uniform cube, N = 10⁵, sources equal to targets, charges in
    [−1, 1), a uniform level-4 tree (`max_level` 4, `max_points_per_leaf` 1). For
    p ∈ {3, 8, 18} in f64 and p ∈ {3, 8} in f32, it prints the relative L2 and max
    error of φ and ∇φ over 1,000 sampled targets, the prediction and the ratio, the
    tree (levels, leaves, points per leaf) and the stage timings, as Markdown;
  - a smoke test of the example's core at N = 500, p = 2 (one MPI-initialising test in
    its executable).
- Design the public signatures of `build` and `evaluate` so that C5.1 can add point
  redistribution behind them (design §9 of fmm-plan-redesign.md) without changing
  them: points and charges in the caller's order, outputs in the caller's order.
- Multi-rank, by hand, under an external timeout: `tests/mpi_exec.rs` on 2 ranks must
  return `PointsNotOwned` consistently on every rank in the scenarios where points
  are not owned, and must neither hang nor diverge. Add a scenario for this that
  passes on one rank (no error) and on several (the error).

Must pass:
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` (debug run under a minute);
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored`;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`;
- `cargo run --release -p nd-fmm-validate --example fmm_accuracy`, within a few
  minutes; paste its output into the PR;
- the 2-rank run above;
- `cargo clippy -p nd-fmm-exec -p nd-fmm-validate --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-exec --no-deps` without warnings;
- the root checks.

Report the C3.2 table (p, measured, prediction, ratio), f32 next to f64, the stage
timings at N = 10⁵, and the debug test time. If the gate fails at any p, report the
errors per stage that can be separated (for example φ with P2P only, against the near
field of the direct sum) and stop.

Do not: redistribute points between ranks (C5.1); register MPI examples with
templated-examples; add rayon; assert timings; commit output files.

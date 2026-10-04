# Phase 5 / T2 — remove the index FMM from nd-fmm-plan

The index FMM (`IndexFmm`, `BatchedIndexFmm`) was the exact topology test of the Phase 3
rewrite. Its operators propagate leaf indices, so every leaf could be checked to receive
every other leaf's sources exactly once. It served that purpose:
- the rewrite reproduced the old evaluator on every scenario (Phase 3 T6, T7);
- the Laplace FMM in `nd-fmm-exec` has since checked the evaluator's values against the
  direct sum on every scenario.

It is now retired (docs/phase5/README.md, decision 5). This task removes it and keeps
the coverage that still matters through the checks that remain:
- the list oracle;
- the exchange checks;
- the recording operator in `tests/mpi_regressions.rs`;
- the Laplace FMM in `nd-fmm-exec`.

Nothing else in the crate changes.

Read first:
- root CLAUDE.md, fmm-plan/CLAUDE.md;
- docs/phase5/README.md (the paragraph on `IndexFmm`, "Design decisions", the T2 line of
  "Exit gate");
- fmm-plan/src/index_fmm.rs, fmm-plan/src/index_fmm_tests.rs,
  fmm-plan/src/evaluator_tests.rs, fmm-plan/src/lib.rs (the crate docs and the module
  list);
- fmm-plan/tests/mpi_regressions.rs: `check_counts_of_one`, `check_variable_counts`, the
  helpers they use, `check_batches` and the recording operator, and the `cases` loop;
- fmm-plan/examples/test_index_fmm.rs, fmm-plan/examples/evaluator_stage_cost.rs,
  fmm-plan/Cargo.toml (the `[[example]]` and templated-examples entries);
- fmm-exec/tests/mpi_exec.rs (the scenario `batched against per-pair`, and `device
  backends`, the one scenario that runs distributed today).

Do:
- Remove from `nd-fmm-plan`:
  - the module `index_fmm` (`IndexFmm`, `BatchedIndexFmm`, `Walk`, `GlobalLeaves`,
    `check_counts`, `run_index_fmm`, `IndexPath`) and `src/index_fmm_tests.rs`;
  - the example `test_index_fmm`, with its `[[example]]` section and
    `[package.metadata.example.test_index_fmm.templated-examples]` entry;
  - the example `evaluator_stage_cost`, which times the evaluator with the index FMM.
    Its numbers belonged to Phase 3 T6. If you think a stage-cost example with another
    operator is worth keeping, say so in the PR instead of writing one;
  - in `tests/mpi_regressions.rs`: `check_counts_of_one`, `check_variable_counts` and
    every helper only they use.
- Keep, unchanged in what they check:
  - `PairOperator` and `PerPair`. `LaplaceOperator` implements both, and
    `fmm-exec/tests/mpi_exec.rs` compares its per-pair and batched paths bit for bit;
  - every `tests/mpi_regressions.rs` scenario and its plan, list-oracle, exchange,
    coarse-gather and recording-operator checks (`check_batches`). The scenario set must
    not shrink. Also keep the second-evaluation bit-identity check, run with the
    recording operator, or with a minimal operator in the test file whose values do not
    matter, if the recording operator cannot express it;
  - `src/operator_tests.rs`, `src/store_tests.rs`, `src/exchange_tests.rs`,
    `src/plan_tests.rs`.
- `src/evaluator_tests.rs`: the four tests run the evaluator with the index FMM on
  ghost-free trees.
  - Keep what they check about the evaluator that does not depend on index values:
    the call order, `reset` and a repeated evaluation, validation errors, and every batch
    handing the right views and slices. Use a small test-local operator there, for
    example the recording style of `mpi_regressions.rs`.
  - Drop the value checks. Say in the PR which assertions went and which check now
    covers each.
- Documentation:
  - the crate docs in `src/lib.rs` (module list, "testing", the compute graph text);
  - fmm-plan/CLAUDE.md: the paragraph on the index FMM, the code map, "Crate checks",
    "Multi-rank runs" (no `test_index_fmm`; `mpi_regressions` stays the multi-rank
    run), "Conventions";
  - root CLAUDE.md: the parenthesis "(its `IndexFmm` uses `u32`)" under "Working rules".
    The rule it qualifies stays: `FmmOperator::Value` is deliberately generic, so
    nd-fmm-plan is not bound to `RealScalar`. Reword it without the index FMM;
  - any rustdoc link to the removed items.

  Leave the design documents (fmm-plan-redesign, device-path, laplace-fmm-plan,
  workspace-structure) and the Phase 3 and 4 briefs as they are; they record history.
  T10 notes the removal in the design documents at the end of the phase.

Tests that define done:
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan` passes, with fewer unit tests than
  before. The PR lists the removed tests and, for each removed check, what still covers
  it, for example:
  - list coverage: the oracle in `mpi_regressions`;
  - every pair issued once: the recording operator;
  - the values of the distributed passes: `fmm-exec/tests/mpi_exec.rs` against the direct
    sum, and from T6 against one rank;
  - per-pair against batched: the `LaplaceOperator` scenario in `mpi_exec`.
- `tests/mpi_regressions.rs` on 1, 2 and 4 ranks, by hand, under an external timeout
  with the macOS loopback flags (fmm-plan/CLAUDE.md, "Multi-rank runs"): every scenario
  passes.
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` passes unchanged. Nothing outside
  `nd-fmm-plan` used the index FMM.
- `grep -rn "IndexFmm\|index_fmm\|run_index_fmm\|GlobalLeaves"` finds nothing outside
  `docs/design/`, the Phase 3 and 4 briefs and this phase's briefs.

Must pass:
- `cargo fmt --all`, then the root checks (`cargo fmt -- --check`, `cargo clippy -- -D
  warnings`, `cargo clippy --examples -- -D warnings`, `RUST_MIN_STACK=8388608 cargo
  test`, `cargo doc --no-deps`) and the stricter workspace checks;
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p nd-fmm-plan` (no broken links);
- the multi-rank runs above.

Do not:
- change the plan, the lists, the stores, the exchanges, the operator traits or the
  evaluator. This task only removes code and adapts tests;
- weaken a remaining check, or drop a scenario;
- add a replacement test FMM that propagates values exactly. If you find a gap that only
  such an operator could close, report it in the PR instead.

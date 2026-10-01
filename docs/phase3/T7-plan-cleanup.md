# Phase 3 / T7 — nd-fmm-plan: remove the old API and document the new crate (C3.0)

The last rebuild task. T6 showed that the new evaluator reproduces the old one on every
scenario. This task removes the old code, moves the new modules to their final paths,
and brings the docs, `fmm-plan/CLAUDE.md` and the manifest in line with the signed-off
design. It changes no behaviour.

This task works inside fmm-plan/ and follows fmm-plan/CLAUDE.md. It may also touch, in
nd-fmm-exec only, the `use` paths that the move breaks, and nothing else there.

Read first: root CLAUDE.md, fmm-plan/CLAUDE.md, docs/design/fmm-plan-redesign.md (§11
and the recorded sign-off decisions), the T4–T6 code, and every caller of nd-fmm-plan in
the workspace (`grep` the manifests; nd-fmm-exec only, at this point).

Do:
- Remove the old API: `InteractionManager`'s map-based lists, `LevelData`, the old
  `FmmOperator`, `FmmEvaluator`, `FmmGhostCommunicator` and `run_index_fmm`, or
  whatever the design marks as replaced. Keep `V_LIST_DIRECTIONS` at its current path.
- Move the new modules from their temporary path to the final ones of the design.
- Tests:
  - remove the old-against-new comparisons (their reference is gone);
  - keep every list-oracle check and every `IndexFmm` check on the new API;
  - port any old serial test that still tests a kept property.

  The set of scenarios in `tests/mpi_regressions.rs` must not shrink.
- Docs:
  - crate docs in src/lib.rs and src/fmm.rs (the compute graph, now with the batched
    calls), and "Future extensions" (only overlap and device data remain);
  - the README;
  - fmm-plan/CLAUDE.md: the project description, the code map, the test counts, the
    conventions section, and the note on rewriting.
- Manifest, as signed off (design §12, question 5). If approved:
  - move fmm-plan's dependencies to `[workspace.dependencies]` and its package fields
    to `[workspace.package]`;
  - fix the licence form, `homepage` and `repository`;
  - regenerate `Cargo.lock` with the change, and update the root CLAUDE.md sentence
    "octree/ and fmm-plan/ declare theirs directly until migrated" to name octree/
    only.

  If not approved, leave the manifest alone.

Must pass, and report which ran:
- the root CI commands and the stricter workspace checks;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan` and
  `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec`;
- `cargo clippy -p nd-fmm-plan --all-targets -- -D warnings`;
- `RUSTDOCFLAGS="-D warnings" cargo doc -p nd-fmm-plan --no-deps`;
- `cargo run -p nd-fmm-plan --example test_index_fmm`;
- the `mpi_regressions` binary on 1, 2 and 4 ranks, by hand, under an external timeout.

Report the removed and the remaining public API, the line counts before and after, and
the scenario and test counts before and after.

Do not: change behaviour, list rules or the pass order; change nd-octree; touch
nd-fmm-exec beyond broken `use` paths.

# Phase 3 / T6 — nd-fmm-plan: batched operator interface, evaluator and IndexFmm (C3.0, C4.0)

The third of the four rebuild tasks, and the one `nd-fmm-exec` builds on. It adds:
- the level-batched operator interface of the redesign, with its per-pair adapter;
- the new evaluator, which runs the pass order on the T4 index and the T5 stores;
- `IndexFmm` on the new interface.

The old evaluator stays as the reference. On every scenario, the new one must give
the same `IndexFmm` result. This also delivers C4.0, the batched hooks that the
design document had planned for Phase 4.

This task works inside fmm-plan/ and follows fmm-plan/CLAUDE.md. The signed-off
`docs/design/fmm-plan-redesign.md` is the specification; where it and this brief
disagree, the design wins. Report the difference.

Read first: root CLAUDE.md, fmm-plan/CLAUDE.md, docs/phase3/README.md (requirements
1, 6, 7, 8 and 10; "Exit gate", C3.0 and C4.0), docs/design/fmm-plan-redesign.md (§6,
§7 and §11), the T4 and T5 code, fmm-plan/src/fmm.rs, src/fmm/*.rs and
examples/test_index_fmm.rs.

Do:
- The operator trait or traits of design §6, with every method documented: what it
  receives, what it may read and write, the accumulation rule, and the order in which
  the evaluator calls it.
- The per-pair adapter of design §6, so an operator can implement one method per pair
  and get the batched trait.
- The evaluator of design §7:
  - the pass order of `src/fmm.rs` (steps 1–6), with every collective entered by every
    rank;
  - the per-leaf counts of requirement 4 at construction; zero counts allowed;
  - the global coarse levels;
  - the fixed accumulation order of requirement 8, documented;
  - public stage methods and `reset()`, so a caller can time each stage.
- `IndexFmm` on the new interface, through the per-pair adapter:
  - every source point of leaf j carries the value j; P2M, P2L and P2P add one at
    index j for every source point;
  - L2P, M2P and P2P act on every target point of a leaf; the target output size per
    point is the number of leaves;
  - the check: every target point of every leaf holds, at index j, exactly the number
    of source points of leaf j. With counts of one this is today's check;
  - with zero-count leaves a missed interaction can hide. Say so in the docs, and keep
    the test counts mostly nonzero.
- A batched `IndexFmm` variant (or a test operator) that implements the batched trait
  directly, without the adapter, so the batched path is exercised by itself.
- Port `examples/test_index_fmm.rs` to the new evaluator with random counts. Register
  it with an `[[example]]` section and
  `[package.metadata.example.test_index_fmm.templated-examples]` with
  `command = "mpirun -n {{NPROCESSES}}"`, as octree/Cargo.toml does, so the weekly job
  runs it at 3 ranks.

Tests that define done:
- Serial (no MPI): the adapter on hand-made batches; the batched `IndexFmm` operators
  on hand-made chunks with several points per leaf.
- tests/mpi_regressions.rs, in the existing `cases` loop:
  - the old `run_index_fmm` still passes unchanged;
  - the new evaluator with `IndexFmm` and counts of one gives, on every leaf, the same
    target values as the old evaluator;
  - the new evaluator with seeded counts derived from the key (zeros included) passes
    the count check, through the adapter and through the batched variant;
  - every batch the evaluator issues honours its grouping (record the calls in a test
    operator): each target at most once per (level, offset), complete octant batches,
    and every list pair issued exactly once;
  - every level call hands the operator both views of requirement 5, and an operator
    that walks the target-centric view gives the same `IndexFmm` output as one that
    walks the groupings;
  - determinism: two evaluations give bit-identical outputs.
- Multi-rank, by hand, under an external timeout: the `mpi_regressions` binary on 1, 2
  and 4 ranks, and the example on 2, 3 and 4 ranks.

Must pass, and report which ran:
- the root CI commands and the stricter workspace checks;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan` (debug run under a minute);
- `cargo clippy -p nd-fmm-plan --all-targets -- -D warnings`;
- `RUSTDOCFLAGS="-D warnings" cargo doc -p nd-fmm-plan --no-deps`;
- `cargo run -p nd-fmm-plan --example test_index_fmm`;
- the multi-rank runs above, with their output.

Report the public API of the new evaluator and traits, and the wall time per stage of
the old and new evaluator with `IndexFmm` on a 10⁴-leaf tree on one rank (release, not
asserted).

Do not: change or remove the old API (T7); add any kernel arithmetic; add rayon, device
buffers or overlap; put a collective in a rank-dependent branch. If the new evaluator
disagrees with the old one, find which stage diverges, report it and stop. The old one
is the reference unless the list oracle shows it wrong.

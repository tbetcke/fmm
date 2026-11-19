# CHA.3 The superseded M2L configuration removed

Started 2026-11-19 on `tbetcke/h-cha-3`. Roadmap: "CHA.3 The superseded M2L configuration
removed. Review P1. *Done when:* `NodeContext`'s unused fields, `FmmBuilder::dense_node_budget`,
the `AUTO_DENSE_*` constants, the deprecated `resolve(p)` and the harness's `--budget` are gone,
with the tests that only they serve; the history of the Phase 5N rule in
`docs/design/node-m2l.md`, not in rustdoc; `M2lStrategy::Auto`'s choices unchanged on every case
of `tests/operator/strategy.rs`. A public API change, named in the PR." Proposal: none (section
HA: the roadmap's *Done when* with constraint 1 of §2).

## Goal
Remove the configuration that Phase 5N T4's node rule left public and plumbed after Phase 6 T11
replaced that rule (review 2026-11-12-H6, P1), without changing a choice or a bit.

## Acceptance (roadmap *Done when*, verbatim)
`NodeContext`'s unused fields, `FmmBuilder::dense_node_budget`, the `AUTO_DENSE_*` constants, the
deprecated `resolve(p)` and the harness's `--budget` are gone, with the tests that only they
serve; the history of the Phase 5N rule in `docs/design/node-m2l.md`, not in rustdoc;
`M2lStrategy::Auto`'s choices unchanged on every case of `tests/operator/strategy.rs`. A public
API change, named in the PR. With constraint 1 (the HA preamble): Laplace's outputs bit for bit,
its benchmarks within D9's spread wherever a host or device path's code moves; no behaviour or
default changes.

## Read first
- `docs/reviews/2026-11-12-H6.md` P1, checked against the tree on 2026-11-19: `M2lStrategy::choose`
  (`fmm-exec/src/tables.rs:228`) reads `node.p` and `node.value_bytes` only; `NodeContext`
  (`tables.rs:253`) still has `ranks_per_node` and `node_budget`; `AUTO_DENSE_MAX_P` (deprecated,
  `:110`), `AUTO_DENSE_RANK_BYTES` (`:143`), `AUTO_DENSE_NODE_BYTES` (`:154`), `resolve(p)`
  (deprecated, `:161`); `resolve_on_node`'s rustdoc (`:166-219`) narrates T11's and 5N's rules;
  `FmmBuilder::dense_node_budget` (`fmm.rs:1371`); the harness's `--budget`
  (`fmm-validate/examples/scaling.rs:194`, `Settings::dense_node_budget`, the summary's
  `budget_mb`), passed by every launch of `tools/scaling/run.sh` (`:590`).
- `docs/design/node-m2l.md` §4, §10 (the rule and its outcome); `docs/phase6/README.md` decision 7.

## Changes
All in `nd-fmm-exec` and `nd-fmm-validate`; no other crate, no dependency, no convention.
- `fmm-exec/src/tables.rs`:
  - remove `AUTO_DENSE_MAX_P`, `AUTO_DENSE_RANK_BYTES`, `AUTO_DENSE_NODE_BYTES` and `resolve(p)`;
  - `NodeContext` keeps `p` and `value_bytes`; `ranks_per_node`, `node_budget` and
    `dense_node_bytes` (which reads `ranks_per_node`) go; `NodeContext::new(p, value_bytes)`;
    `one_rank::<T>(p)` and `dense_rank_bytes` stay (question 2, signed off), their docs restated;
  - `resolve_on_node`'s rustdoc: the rule and why its default is what it is (T11, decision 7,
    the report), no Phase 5N narration; `NodeContext`'s and `Tables::build`'s docs to match.
- `fmm-exec/src/fmm.rs`: `FmmBuilder::dense_node_budget` and its field and defaults-table row go.
  `build`: both `NodeContext`s lose the two arguments; step 2's all-reduce no longer carries the
  ranks per node for the cost model (read only to build that `NodeContext`), so it reduces one
  value; step 7 still agrees the ranks per node for `ThreadingReport::ranks_per_node` (kept,
  reported). Module docs (step 7, `Fmm::strategy`) restated.
- `fmm-exec/src/cost.rs`: `CostModel::calibrated`'s `Auto` arm builds `NodeContext::new(p, bytes)`.
- `fmm-exec/src/threading.rs`: the module docs' sentence on the budget.
- Tests: `tests/operator/strategy.rs` (the constants' asserts, the ranks × budget loops, the
  deprecated `resolve` block and `one_rank`'s equality with the old 4-argument form go; every
  (precision, p) cell and the named strategies stay); `tests/mpi_exec.rs`
  (`strategy_on_the_node`: the budget loop becomes `Auto`, `Dense` and `Rotation` named, the
  ranks per node agreed and reported, the cold cache once per node as before; its module docs);
  `tests/operator/common.rs` unchanged (`one_rank`).
- `nd-fmm-validate`: `examples/scaling.rs` `--budget` and its docs; `src/scaling.rs`
  `Settings::dense_node_budget`, the builder call, the summary's `budget_mb` (question 3;
  `dense_rank_mb`, `dense_node_mb` kept, computed as ranks × `dense_rank_bytes`), the module docs;
  `examples/device_fmm.rs` the 2-argument `NodeContext`; `fmm-validate/CLAUDE.md`.
- `tools/scaling/run.sh`: `--budget $BUDGET` off the launch line; `BUDGET` kept for
  `auto_resolves` alone (the Phase 5N sweeps' restatement of the old rule, CHA.10's to move).
- Docs: `docs/design/node-m2l.md` §11 "Superseded (Phase 6 T11) and removed (Helmholtz CHA.3)":
  the 5N rule's source from the rustdoc (the per-pair costs, the 194 cells, the rule of Phases
  2–5), what replaced it, what CHA.3 removed. `fmm-exec/CLAUDE.md`'s two paragraphs (the strategy
  rule on the node; the defaults of T11) cut to what holds.
- Public API removed (named in the PR): `M2lStrategy::{AUTO_DENSE_MAX_P, AUTO_DENSE_RANK_BYTES,
  AUTO_DENSE_NODE_BYTES, resolve}`, `NodeContext::{ranks_per_node, node_budget,
  dense_node_bytes}`, `NodeContext::new`'s last two arguments, `FmmBuilder::dense_node_budget`;
  `scaling --budget`, `nd_fmm_validate::scaling::Settings::dense_node_budget`.

## Tests that prove it
- `Auto`'s choices unchanged: on `main` (3468950) and on the branch, a throwaway program (not
  committed) prints `M2lStrategy::Auto.choose(..)` and its `Display` for both precisions and
  p = 0–20 (on `main` over the old grid of 1–128 ranks × the six budgets of
  `tests/operator/strategy.rs`, which must give one line per (precision, p)); the two lists equal.
  The kept `auto_resolves_by_the_node_rule` asserts every (precision, p) cell against `rule`, as
  before, and the reasons' text.
- Laplace bit for bit: no evaluation code changes, and `Fmm::strategy` is the same function of
  (p, precision); every Laplace test passes (`mpi_exec`, `invariance`, `multi_rank`'s one-rank
  references), the MPI list at 2 and 4 ranks.
- `strategy_on_the_node` at 1, 2 and 4 ranks (CI's list): the report agreed on every rank.

## Checks
`/verify` (fmt, Clippy default, examples and workspace, docs, the default and workspace tests at
opt-level 1, the MPI list at 2 and 4 ranks, `cargo deny`); `cargo clippy -p nd-fmm-exec
--all-targets --features cpu,metal -- -D warnings` and `cargo check -p nd-fmm-exec --features
cuda` (the device path's `NodeContext` users compile); `cargo check -p nd-fmm-validate
--examples --features cpu`; `tools/scaling/run.sh` once on a small case without `--budget`
(correctness only, not timed); `bash -n tools/scaling/run.sh`.

## Machine runs
None (question 4, signed off: no D9 runs; no host or device evaluation code moves).

## Constraints and risks
- Constraint 1: `FmmBuilder` is Laplace's builder; its build changes only in what it passes to
  `choose`, which does not read it. Outputs bit for bit by construction; checked by the tests.
- The all-reduce of step 2 carries one value fewer on every rank alike (no rank-dependent branch).
- Removing `budget_mb` from the summary changes the harness's key=value lines; no script in the
  tree reads that key (`git grep`).

## Pending runs
None.

## Open questions
None. Signed off 2026-11-19 at the plan (the user, all four as recommended): (1) "Go ahead
(Recommended)"; (2) `one_rank`: "Keep the name (Recommended)"; (3) the summary: "Drop budget_mb
only (Recommended)"; (4) "Skip D9 runs (Recommended)".

## Results
- Removed as planned (commits `90aebee`, `0d8994a`); the self-review's findings in `e2775ae`
  (stale module docs of step 2's all-reduce and of the cost model's strategy in `fmm.rs`, and
  older sentences that `Auto` depends on the ranks per node, in `fmm.rs`, three test files,
  `scaling.rs`, `fmm-exec/CLAUDE.md`; `workspace-structure.md`'s `tables` line).
- `Auto`'s choices: a throwaway example (not committed) printed `M2lStrategy::Auto.choose`, its
  strategy and its `Display`, for f32 and f64 at p = 0–20: on `main` (3468950) over the old grid
  of 1–128 ranks per node × the six budgets of `tests/operator/strategy.rs` (one line per
  (precision, p), asserted), on the branch with `NodeContext::new(p, value_bytes)`. The 42 lines
  are identical (`diff`), e.g. `Auto → Batched: f32 p = 7 < 8`, `Auto → PlaneWave: f64 p = 7 ≥ 7`.
- The harness: `scaling` at one rank (debug, N = 3000, f64 p = 4, no `--budget`) ran, its summary
  without `budget_mb` and with `dense_rank_mb=1.7 dense_node_mb=1.7`; `scaling --budget 96`
  refused ("unknown option --budget"); `DRY_RUN=1 tools/scaling/run.sh strong-small` printed 24
  launches, none with `--budget`; `node-rule` still names its strategies through `auto_resolves`.
- Debug test time: no executable gained work (`strategy_on_the_node` builds three `Fmm`s where it
  built four), so no opt-level 0 timing was run.

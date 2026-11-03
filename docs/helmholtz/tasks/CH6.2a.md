# CH6.2a The waves buffer bounded

Started 2026-11-03 on `tbetcke/h-ch6-2a`. Roadmap: "CH6.2a The waves buffer bounded" (H6;
priority list item 3, the first sub-task of CH6.2, split on this branch). Proposal: §10
CH6.2; §2 (constraints 2, 7, 9), §5.3 (Exponential M2L), §6.4, §7; CH6.1e's report §7 and
its question 4 (signed off 2026-11-03: "the waves buffer bounded and counted in the node
budget before CH6.2 takes times").

## Goal

Bound the memory of the exponential M2L's level call by a node budget without changing a
bit of its output, so that CH6.2b can time it on the large cases.

## Acceptance (proposal §10, verbatim)

CH6.2: "per-level accuracy and time beside FMM3D's on the same problems; adopted as a
strategy where it wins, else kept opt-in or dropped (D8)". That is CH6.2b's. This
sub-task's, from CH6.1e's sign-off 4 and the roadmap's CH6.2a:

> the output bit for bit the same at every bound (the floor, a middle bound, the whole
> level), every thread count and on P ranks the one-rank output; `plane_wave_bytes` within
> the tables, the bound and the scratch; on W1 at kD = 500 and W3 at kD = 100 at 10⁻⁶ the
> bytes and the waves computed per box at the default and smaller bounds recorded, with the
> evaluation's time against the whole-level buffer (Kathleen).

## Read first

- `fmm-exec/src/helmholtz/operator_plane_wave.rs` (the level call, `PlaneScratch`),
  `operator.rs` ("M2L", `with_plane_wave_boxes`, `plane_wave_bytes`, `select_tables`),
  `fmm.rs` (the builder's step 7).
- `fmm-exec/src/driver.rs` (`Execution::{chunks, parents}`; used, not changed).
- `fmm-exec/tests/helmholtz_fmm.rs` (`check_plane_wave`, `check_one_rank`),
  `fmm-exec/tests/helmholtz/plane_wave.rs`.
- `fmm-validate/src/helmholtz_accuracy.rs` (`Case`, `Row::dense_bytes`),
  `examples/helmholtz_accuracy.rs`, `tools/helmholtz/ch6-1e-runs.sh`.
- `docs/design/node-m2l.md` (the node budget, Laplace's), CH6.1e's report §7.

## Changes

No Laplace file, convention, dependency or `nd-fmm-plan` change. Crates: `nd-fmm-exec`
(the level call, the operator, the builder) and `nd-fmm-validate` (the harness's flag and
columns, the run script).

1. **The level call by chunks of parents** (`operator_plane_wave.rs`). Per vector and frame
   the level is cut into chunks of consecutive whole parents, greedily in index order: a
   parent joins the chunk while the chunk's distinct sources (the V-row sources of its
   computed children, each a slot of the parent) fit the buffer's B boxes, B ≥ 216 (one
   parent's slots). For each chunk: the sources listed in first-seen order with a per-box
   position map (`u32` per box of the level, reset after the chunk), their up and down
   waves computed into the buffer on the pool (`Execution::chunks`), then merge and shift
   over the chunk's parents (`Execution::parents` on the chunk's locals), reading each
   source's waves through the map. Each box's waves are a function of its multipole alone,
   and every accumulator keeps its slots' order, so the output does not depend on B, the
   chunking or the thread count. With one chunk per level the waves are computed for the V
   sources only (today: every box of the level, own and ghost).
2. **The bound** (`operator.rs`): `HelmholtzOperator::with_plane_wave_buffer(bytes)`, the
   waves buffer's byte bound per rank (at least 216 boxes of the largest level with waves,
   at most the largest level's boxes, as `with_plane_wave_boxes` sizes it today), the map
   and the source list sized at the same time (no allocation in a level call);
   `plane_wave_buffer()` reports the bytes held; `plane_wave_bytes` counts the buffer, the
   map and the list. `plane_wave_outgoing()` (doc-hidden or public, decided while writing):
   the waves computed (boxes × frames × vectors) by the last evaluation, so the report can
   give the recomputation per box.
3. **The node budget** (`fmm.rs`): `HelmholtzFmmBuilder::plane_wave_budget(bytes)`, the
   waves buffer's bytes per node, default `DEFAULT_PLANE_WAVE_NODE_BUDGET` (proposed
   4 GiB: the M6's 24 GB with room for the rest, about 100 MB per rank at Kathleen's 40
   ranks per node; for sign-off); step 7 splits the communicator by node (`split_shared`,
   every rank) and agrees the largest ranks per node (an all-reduce max, every rank), and
   the operator's bound is the budget over it. `HelmholtzFmm::ranks_per_node` (or the
   operator's bound) reports it. The output does not depend on it (item 1), so the
   agreement serves the report and the budget's arithmetic only.
4. **The harness** (`nd-fmm-validate`): `Case::plane_wave_budget`, the example's
   `--plane-wave-budget <bytes|whole>`, columns for the buffer's bytes and the waves
   computed per V source; `tools/helmholtz/ch6-2a-runs.sh` for the Kathleen job.
5. Docs: the `helmholtz::operator` module docs ("M2L"), `fmm-exec/CLAUDE.md` (the
   exponential M2L entry), the report `fmm-validate/results/helmholtz-ch6-2a.md`.

## Tests that prove it

- `fmm-exec/tests/helmholtz_fmm.rs`, `check_plane_wave` (both scenarios, one and two
  multipole planes): the FMM at the floor bound (216 boxes) and at a middle bound (a few
  chunks per level) bit for bit the whole-level buffer's output; the floor at 2 threads bit
  for bit; on several ranks the floor bound's FMM against the one-rank FMM bit for bit
  (`check_one_rank`); the node budget's arithmetic (the bound reported = budget / ranks
  per node, floored at one parent); `plane_wave_bytes` with the floor below the whole
  level's.
- `fmm-exec/tests/helmholtz/plane_wave.rs` (operator level, no MPI): the bound's clamps
  (below the floor, above the whole level, zero), `plane_wave_bytes` = tables + buffer +
  map + list + scratch.
- Chunking unit tests in `operator_plane_wave.rs` if the greedy cut is a function of its
  own (each chunk's distinct sources ≤ B, whole parents, every parent once, in order).
- The existing checks unchanged: the pair path within 1e-14, the direct sum and 1.1×
  Rotation's error, two vectors, named precisions.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`,
`cargo test` and `--workspace`, the MPI list at 2 and 4 ranks), `cargo test -p
nd-fmm-validate --release --test helmholtz_accuracy`, `helmholtz_fmm` in release at one
rank. D9 not needed (no Laplace path changed).

## Machine runs

Kathleen, one node (`singlenode`), one rank × 40 threads, release, BLAS threads 1: W1 at
kD = 500 and W3 at kD = 100 at ε = 10⁻⁶ under `--m2l planewave` with the whole-level
buffer, the default budget's bound and two smaller bounds: `plane_wave_bytes`, the waves
computed (against the whole level's: the recomputation), rank 0's peak resident size
(VmHWM, the oracle's included), the evaluation's time (one evaluation per launch; the
second whole and default launches give the run-to-run spread), the error and the checksum
(the same at every bound). And at 40 ranks × 1 thread, the whole level and the default
(4 GiB over 40 ranks per node), the checksum the one-rank runs'. Decides whether
the default budget costs time (the recomputation) and is recorded for CH6.2b. Not a gate:
no quiet-machine rule beyond Kathleen's exclusive node.

## Constraints and risks

- Constraint 2 (determinism, rank invariance): the chunking changes which waves are
  computed when, never a value or an accumulation order; the tests check every bound.
- Constraint 7 (the node budget): the buffer is the binding memory of the strategy; the
  node split is new in the Helmholtz builder (Laplace's builder has had it since Phase 5N
  T4). The dense budget stays per rank (CH4.0 counts it per node).
- Constraint 9: the bound reads the budget and the ranks per node, which change no output;
  the chunking reads the plan's V rows only.
- Laplace: not touched (Helmholtz files of `nd-fmm-exec`; the driver's `Execution` used,
  not changed). Laplace's own plane-wave buffer is whole-level too, but its W is small; not
  in scope.
- Risk: recomputation at small bounds (a chunk's halo of sources); measured on Kathleen.
  Two collectives added to the Helmholtz build (the split and the all-reduce), every rank.

## Pending runs

- Kathleen job 628518 (one node, `singlenode`, 4 h; submitted 2026-11-03 21:24):
  `cargo build --release -p nd-fmm-validate --example helmholtz_accuracy`, then
  `tools/helmholtz/ch6-2a-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch6-2a budgets ranks`.
  Decides the report's bytes, recomputation and time at each bound and whether the
  default budget is kept.

## Open questions

1. The default node budget (4 GiB proposed) for sign-off in the PR.

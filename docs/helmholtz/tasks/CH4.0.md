# CH4.0 The M2L rule's sparse-level term

Started 2026-11-12 on `tbetcke/h-ch4-0`. Roadmap: "CH4.0 The M2L rule's sparse-level term"
(H4; priority list item 1). Proposal: §10 CH4.0; §2 (constraints 1, 2, 7, 9), §5.3, §6.4;
CH3.2d's report §2–§3 (`fmm-validate/results/helmholtz-ch3-2d.md`), CH6.2h's report §5, §6
and §9 (`fmm-validate/results/helmholtz-ch6-2h.md`); the H3 gate §5
(`docs/helmholtz/gates/H3.md`).

## Goal

`HelmholtzM2l::Auto` gives a level to Batched only where its tables are read for enough
columns to beat Rotation, by a term from the global tree, and Batched's level call keeps
every thread busy on levels of few blocks; the dense budget is counted per node.

## Acceptance (proposal §10, verbatim)

> CH3.2d's sweeps within 1.10 on every case of the three machines, sparse levels included

and from the roadmap's *Done when*: "deterministic and rank-invariant"; the Helmholtz dense
budget counted per node, as Laplace's node rule does (constraint 7), before any multi-rank
run with `Auto` as the default; the term covers coarse levels as well as sparse ones (CH6.2h
§6), or Batched's level call splits as CH6.2f's did; and the levels that Batched at p = 7–8
loses at 72 threads on locust below about 3.5·10⁵ pairs (CH6.2h §5), within constraint 2
(the rule cannot read the thread count).

## What the data before this task shows

- **Why Batched loses on sparse levels.** Batched's level call (`src/helmholtz/batched.rs`)
  runs blocks of `M2L_BLOCK_TARGETS` = 64 consecutive targets, and per block, per offset d
  with a pair in the block, one call of the column kernel on that offset's table (n² complex
  entries, n = (p + 1)²) for the block's c pairs of offset d. Its time is about
  α n² V + β n² P over a level of P pairs and V (block, offset) visits, Rotation's γ p³ P.
  So Batched wins where the mean columns per visit m = P / V exceeds
  m*(p) = α n² / (γ p³ − β n²), infinite at and above the crossover per pair. CH3.2d's
  crossover d_b was fitted on dense levels (the cube: m ≈ 64 · 189/316 ≈ 38); W1's levels
  are sparse (fewer V entries per box, so fewer columns per visit), which is the 1.106 miss.
- **Why Batched loses on coarse levels at many threads.** Level 2 (64 boxes) is one block:
  one thread, against Rotation on all of them. At 72 threads every level of fewer than 72
  blocks (about 4,600 targets, or 3.5·10⁵ pairs at about 75 per target) leaves threads idle;
  this is CH6.2h §5's locust finding (1.0–28×). Not a cost per pair: a split.
- **The split is allowed.** Batched's level call is the pair chain bit for bit
  (`Batched::m2l`'s docs; `check_tables`), so the block width changes no bit: constraint 9
  as amended at H6.G lets the execution split read the thread count. `Execution::blocks`
  already takes the width (`driver.rs` unchanged).
- **The term must not read the thread count** (constraint 2: Batched's bits are not
  Rotation's) nor a rank's share of the boxes (constraint 9); it may read the global tree.

## Read first

- `fmm-exec/src/helmholtz/operator.rs`: `auto_batched_max_degree`, `select_tables`,
  `DEFAULT_DENSE_BUDGET`; `src/helmholtz/batched.rs`; `src/helmholtz/fmm.rs` `build` (step
  7, the agreements, `ranks_per_node`).
- `fmm-tables/src/helmholtz/batched.rs`: `select_within`, `level_bytes`.
- `fmm-validate/src/helmholtz_accuracy.rs` (`Row::v_pairs`, `--levels`),
  `fmm-validate/examples/helmholtz_m2l.rs` (`--columns`), `tools/helmholtz/ch3-2d-runs.sh`,
  `ch6-2h-runs.sh`, `ch6-2h-tables.py`.
- Laplace's node rule: `M2lStrategy::resolve_on_node` (Phase 5N T4) for the per-node budget.

## Changes

`nd-fmm-exec` (Helmholtz modules only) and `nd-fmm-validate`; tools.

1. **The term** (`helmholtz::operator`):
   - `LevelShape { pairs, targets }` per level: the V pairs and the targets with a V row, each
     counted once over the ranks by the box's owner (`Plan::computes`, the replicated top
     included), summed by one all-reduce in `HelmholtzFmmBuilder::build` and passed with
     `HelmholtzOperator::with_level_shapes`. Integers summed, so the same on every rank and
     at every rank count.
   - `batched_columns(shape)`: the estimate m̂_l of the columns per visit from the shape and
     the block width 64. The form (P / (blocks × expected distinct offsets per block), or a
     simpler one) is chosen by how well it tracks the exact one-rank count, which the harness
     prints beside it.
   - `auto_batched_min_columns(isa, planes, p)`: m*(p) per machine class, from the pair
     evidence (Machine runs, 1).
   - Under `Auto` a level runs Batched where p_l ≤ d_b (unchanged), m̂_l ≥ m*(p_l), and the
     tables fit the budget; the exponential M2L's levels are as before (D8 unchanged). Without
     shapes (the operator alone) the term is off, as before; a test hook names them.
2. **The split** (`helmholtz::batched`): the block width of a level call
   `min(64, max(w_min, ⌈rows / threads⌉))` (w_min from the measurements, at least 1), read
   from the pool at the call; the output the same bits at every width (tested).
3. **The node budget** (`HelmholtzFmmBuilder::dense_node_budget`,
   `DEFAULT_DENSE_NODE_BUDGET`): each rank's dense budget is
   min(`dense_budget`, node budget / ranks per node), with `ranks_per_node` as agreed now.
   The default node budget is a question for sign-off (proposed: 8 GiB, the waves' node
   budget, with the per-rank 1 GiB kept as a cap, so one rank per node is unchanged). As for
   Laplace's node rule (Phase 5N T4), the levels a rank batches then depend on the ranks per
   node; the one-rank reference of a multi-rank test names the budget the run resolved.
4. **The harness** (`nd-fmm-validate`): `Row::{v_targets, batched_columns, block_visits}`
   (the shape, m̂, and the exact visits of each level at width 64 on the rank's own targets,
   summed: on one rank the level call's own count), columns in `--levels`; `--no-term` for the
   rule without it (`Case::batched_term`); `helmholtz_m2l --columns-list`.
5. **Tools**: `tools/helmholtz/ch4-0-runs.sh` and `ch4-0-tables.py`.

No convention, table format, dependency or `nd-fmm-plan` interface changes. No Laplace path.

## Tests that prove it

- `tests/helmholtz/dense.rs`: the term per level (a sparse shape below m* runs Rotation, a
  dense one Batched, unchanged above d_b and on the exponential M2L's levels), off without
  shapes (bit for bit CH3.2d's choice), m̂'s closed form on hand shapes, m* monotone in p
  and infinite above the per-pair crossover; the split's widths at 1, 2, 7 and 72 threads.
- `tests/helmholtz_fmm.rs` (`check_auto`): the default on a tree with a sparse level against
  the direct sum and Rotation (within `DENSE_TOLERANCE` relative to Rotation's output, as
  now), 2 threads bit for bit 1 thread, the one-rank FMM bit for bit on P ranks, the shapes
  equal on every rank and to the one-rank tree's; Batched at widths 1, 7 and 64 (a test hook)
  bit for bit; the node budget halved per rank at 2 ranks per node (named budget, levels
  reported), the default unchanged at one rank.
- `tests/helmholtz_accuracy.rs` (nd-fmm-validate): the new columns; one rank's
  `block_visits` the hand count on a small tree.

## Checks

`/verify` (fmt, clippy, tests, docs, the MPI list at 2 and 4 ranks; constraint 1: no Laplace
file changed). `cargo clippy -p nd-fmm-validate --all-targets -- -D warnings`,
`cargo test -p nd-fmm-validate`.

## Machine runs

Release, one rank, BLAS threads 1, f64, two planes (complex charges) unless named.

1. **Pair evidence** (`helmholtz_m2l --ps 3,…,15 --columns-list 1,2,4,8,16,32,64`, twice per
   ISA): Batched's ns per pair against Rotation's per column count; m*(p) per class is the
   smallest c at which Batched is at least 1.1× Rotation's speed (CH3.2d §3's headroom). The
   M6 (NEON), locust (NEON), Kathleen (AVX-512, AVX2).
2. **FMM evidence**, ABBA, `--levels`:
   - (a) CH3.2d's nine cases at one thread, `--m2l rotation,batched-fused,auto` with the term
     and `--no-term`, CH3.2d's rule (`auto_plane_wave` off, as CH3.2d ran): the rule against
     the per-level best of Batched and Rotation, within 1.10 (the acceptance);
   - (b) CH6.2h's matrix and large cases under the default (D8 on) at the machine's threads
     (the M6 1 × 6, Kathleen 1 × 40, locust 1 × 72), the term and split against `--no-term`
     without the split: the per-level best of Rotation, Batched and PW, as CH6.2h's tables.
   On the M6 only with the user's go-ahead (they run other jobs there).
3. **Node budget**: no timing; the tests.

The report `fmm-validate/results/helmholtz-ch4-0.md`.

## Constraints and risks

- Constraint 1: no Laplace file; `driver.rs` used as is.
- Constraint 2: the term reads integers summed over the ranks; the split changes no bit.
- Constraint 7: the node budget.
- Constraint 9: k, the global tree (the shape), ε, the machine class.
- Risk: the split narrows blocks at many threads, so m falls below m̂ there (m̂ is at width
  64). If Batched then loses on mid-size levels at 72 threads, a minimum width or the term at
  a reference width (a class constant, not the thread count) is the fix; the report says
  which.
- Risk: m̂ is an estimate of the one-rank count; the report shows how far it is off.

## Pending runs

## Open questions

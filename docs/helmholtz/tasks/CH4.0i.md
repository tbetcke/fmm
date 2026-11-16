# CH4.0i The coarse RCR levels

Started 2026-11-16 on `tbetcke/h-ch4-0i`. Roadmap: "CH4.0i The coarse RCR levels" (H4;
priority list item 6, added on 2026-11-16 by CH4.0e's sign-off). Proposal: no §10 row
(CH4.0e's report §3.2 and §7 question 1, signed off as recommended on #220); §2 (constraints
1, 2, 7, 9), §5.3 (Rotation, the RCR M2L).

## Goal

Let a level of few boxes run its Rotation (RCR) M2L on every thread of the node, by splitting
the level's work below one box, without changing an output bit; and compare our RCR pair with
S6's prototype on the same core.

## Acceptance (roadmap *Done when*; no proposal §10 row)

Level 2 of W1 at kD = 500 at ≥ 70 % of linear on the node, or the report says why not;
deterministic and rank-invariant.

Read as: level 2's M2L seconds (`--levels`, the slowest rank's) on one core over (threads ×
its seconds at the split), at 10⁻³, 10⁻⁶ and 10⁻⁸, on Kathleen at 1 × 40 and locust at 1 × 72
(the other splits of CH4.0e reported beside), ≥ 0.70. "Deterministic and rank-invariant":
the output bit for bit `main`'s (so CH4.0e's and CH4.0h's checksums) at every thread count,
split and rank count, which also keeps Laplace's rule that the Helmholtz output is the same at
every thread count and on P ranks (constraint 2).

## Read first

- CH4.0e's report §3.2 ("Level 2's Rotation M2L": by target box only, 64 boxes, 19× on 40
  cores and 20× on 72; blocks formed per pair at p = 131, 18–22 % of a pair; per pair 1.9–3.3×
  S6's prototype on the same core).
- `fmm-exec/src/driver.rs` `m2l` (→ `Execution::boxes`, one task per target, its V row in
  series) and `m2l_target`; `fmm-exec/src/helmholtz/operator.rs` `FmmOperator::m2l` (Batched
  and PlaneWave have their own level calls; RCR goes through `driver::m2l`).
- `fmm-tables/src/helmholtz.rs` `HelmholtzTables::m2l` and `translate`: per pair and vector,
  (1) rotate each multipole plane into scratch, (2) the coaxial step into `work`, (3) rotate
  back each output plane into scratch `y`, then `output[i] = output[i] + y[i]`. The pair's
  whole contribution exists before the add, and the add is one rounding per element.
- `nd_fmm_plan::lists::GroupedCsr::row_offsets` (the V list's flattened pair numbering).
- S6: `spikes/diagonal/SPIKE_REPORT.md` §8, `spikes/diagonal/examples/times.rs` (its RCR is
  S7's `chain::rcr`, rotations precomputed; 3.0 ms at p = 128 on a Grace core).

## Design

**The pair split.** For an RCR level the level call runs in two passes over the level's V
pairs, flattened in the plan's row order (`row_offsets`), in chunks of consecutive pairs
whose contributions fit a buffer:

1. *contributions*: on the pool, every pair of the chunk computes its contribution (step 3's
   `y` of every vector and plane, steps 1–3 of `translate` unchanged) into its own slot of the
   buffer; any order, any thread;
2. *sums*: on the pool, by target, each target of the chunk adds its pairs' slots in row
   order: `local[i] = local[i] + y[i]`, element by element, plane by plane, vector by vector,
   as `translate` does.

Each element of a local then receives the same terms, each computed by the same operations,
added in the same order as by `m2l_target`: the output is `main`'s bit for bit, at every
thread count and chunk size (a target's row may span two chunks; its pairs still add in row
order). So the choice of split may read the thread count (as Batched's block width does,
CH4.0) and constraint 9 is untouched. No work is repeated; the extra cost is writing and
reading each contribution once (O(p²) against the pair's O(p³)).

Alternatives not taken: V-row segments summed apart (no buffer, but a new summation order:
not `main`'s bits, and bits that depend on the segment length); a split by output degree
(`main`'s bits, but every degree slice repeats the forward rotation, about a third of a pair).

**Where.** `nd-fmm-tables` (Helmholtz only): `HelmholtzTables::m2l_contribution` (steps 1–3
into an output slice, without the add) and `add_m2l_contribution` (the add), with `m2l` their
composition bit for bit. `nd-fmm-exec` (Helmholtz only): the operator's RCR level call,
`RcrSplit::{Boxes, Pairs, Auto}` (`HelmholtzOperator::with_rcr_split`,
`HelmholtzFmmBuilder::rcr_split`, reported by `HelmholtzFmm`), the buffer sized from the
plan's pairs per level (as `with_plane_wave_boxes`) within a node budget over the ranks per
node (`HelmholtzFmmBuilder::rcr_pair_budget`, constraint 7). `Auto`: by pairs on an RCR level
whose targets with a V row are fewer than a share of the pool's threads (`RCR_PAIR_SHARE`, to
measure; by boxes elsewhere and serially). `driver.rs` unchanged (the passes use the pool
directly, as Batched's and PlaneWave's level calls do); Laplace's operator, the plan and the
octree unchanged.

**The pair.** Time the pair by step (forming the y-blocks per call, the two rotations, the
coaxial step, the add) on one core at p = 72 and 131, with the blocks stored and per call, and
S6's `times` at p = 72, 128 and 131 on the same locust core; report where ours spends the
1.9–3.3×. Within the contributions pass the pairs may run in any order: if forming the blocks
per call is a sizeable share, order a chunk's pairs by polar angle and keep the last angle's
blocks in the thread's scratch (the same blocks, so the same bits). Anything larger on the
pair itself (layout, vector kernels) becomes a follow-up line, not this task.

## Changes

- `fmm-tables/src/helmholtz.rs`: `m2l_contribution`, `add_m2l_contribution`,
  `m2l_contribution_len`; possibly the per-angle block reuse in `HelmholtzScratch`.
- `fmm-exec/src/helmholtz/operator.rs`: `RcrSplit`, the pair level call, its buffer and
  setters, `rcr_by_pairs` counts (levels and calls run by pairs), docs ("Execution").
- `fmm-exec/src/helmholtz/fmm.rs`, `mod.rs`: `rcr_split`, `rcr_pair_budget`
  (`DEFAULT_RCR_PAIR_NODE_BUDGET`), the buffer sized from the plan; re-exports; `HelmholtzFmm`
  reports the split.
- `fmm-validate`: `Case::rcr_split`, `Row::rcr_split` (and per level whether it ran by
  pairs), `--rcr-split auto|boxes|pairs,…`, a column in `--levels`; `helmholtz_m2l --steps`
  (the pair by step) if the per-step timing needs the example.
- `tools/helmholtz/ch4-0i-runs.sh`, `ch4-0i-tables.py`; the report
  `fmm-validate/results/helmholtz-ch4-0i.md`; `fmm-exec/CLAUDE.md`, `fmm-tables/CLAUDE.md`,
  `fmm-validate/CLAUDE.md`.
- Public API: additions only. Conventions: none. Dependencies: none.

## Tests that prove it

- `nd-fmm-tables` `tests/helmholtz/`: `m2l_contribution` then `add_m2l_contribution` equals
  `m2l` bit for bit on every offset, one and two multipole planes, m = 1 and 2, real and
  complex k, stored and per-call blocks, raised M2L degrees; the add order across several
  pairs equals successive `m2l` calls bit for bit.
- `nd-fmm-exec` `tests/helmholtz/`: the operator's M2L level call by pairs against by boxes,
  bit for bit, on a level of a real plan, serially and on pools of 2, 3 and 5 threads, with
  buffers of one pair, a few pairs (rows split across chunks) and the whole level, m = 1 and
  2, real and complex charges; `Auto`'s choice per level.
- `tests/helmholtz_fmm.rs`: the FMM under `RcrSplit::Pairs` and `Auto` against `Boxes`, bit
  for bit, at 1 and 2 threads and against the one-rank FMM; the MPI list at 2 and 4 ranks
  (rank invariance).
- `nd-fmm-validate` `tests/helmholtz_accuracy.rs`: the column, and `pairs` the checksum of
  `boxes`.

## Checks

`/verify` (fmt, Clippy default/examples/workspace, docs, default and workspace tests, the MPI
list at 2 and 4 ranks, `nd-fmm-tables`' own checks), and the one-rank opt-level-0 runs of
the changed Helmholtz test files with their times.

## Machine runs

Release, BLAS threads 1, every checksum compared with CH4.0e's and CH4.0h's launches of the
same case, ABBA (by boxes against by pairs):
- Kathleen (`singlenode`, one node, 40 cores): W1 at kD = 500 at 10⁻³, 10⁻⁶, 10⁻⁸ on one core
  and at 1 × 40, 2 × 20, 4 × 10, 8 × 5, `--levels`, by boxes and by pairs: the acceptance;
  the levels 3 and 4 decide `RCR_PAIR_SHARE`. CH3.6's matrix at 1 × 40 under `Auto` against
  `boxes` (nothing slower).
- locust (72 cores, quiet): the same at 1 × 72 … 8 × 9; the pair by step at p = 72, 131 and
  S6's `times` at p = 72, 128, 131 on one core.
- The M6: no timing runs (the user's go-ahead would be needed).

## Constraints and risks

- Constraint 1: no Laplace path changes (`driver.rs`, `operator.rs`, `nd-fmm-plan`,
  `nd-octree`, `nd-fmm-simd` untouched); `HelmholtzTables` is Helmholtz's alone.
- Constraint 2 and 9: bit for bit `main` at every split; the rule may read threads only
  because it changes no bit.
- Constraint 7: the buffer within a node budget over the ranks per node; 279 KB per pair at
  p = 131 (level 2 of W1 at kD = 500: about 3,000 pairs, 0.8 GB for the whole level), so the
  chunks bound it.
- Risk: the pair at p = 131 reads 15 coaxial tables of about 12 MB each and forms its blocks:
  on 40–72 threads the level may be bound by memory bandwidth rather than by the split; then
  the report says so, with the per-step profile, and the angle order is the lever inside this
  task.

## As built (departures from the plan)

- The level call by pairs needs a plan, so its bit-for-bit tests are in the MPI test
  `tests/helmholtz_fmm.rs` (`check_rcr_split`: 1, 2, 3 and 20 threads, the buffer's floor, two
  vectors, Rotation and Auto, the one-rank FMM), not in `tests/helmholtz/`; the rule alone is
  `tests/helmholtz/rcr_split.rs`.
- The pair: the class order and the kept y-blocks were built (the plan's lever); the pair is
  timed by variant (stored, per call, by class; the box path against the pair path) by the
  new `helmholtz_rcr_pair`, not by step: a step profile would need a restatement of the
  private steps. The step costs are argued from their operation counts in the report §4.
- Auto's rule: first "by pairs where targets < 4 × threads"; replaced after the runs by the
  degree rule (p ≥ 48; p ≥ 30 with targets < 4 × threads), report §5. Tried and not kept: one
  run of consecutive slots per task (no gain, report §3).
- Extra stages: `stored` (level 2's blocks stored, CH4.0e's question 2 again), `pairs`,
  `rotation`, `crossover` (Auto's rule).

## Results

Report `fmm-validate/results/helmholtz-ch4-0i.md`. Level 2 of W1 at kD = 500 by pairs:
0.55–0.69× by boxes at one and two ranks per node, 0.23–0.40× at more; against CH4.0e's one
core 0.75–0.78 of linear at locust 1 × 72, 0.68–0.80 at Kathleen 1 × 40, 0.89–0.91 at 2 × 20;
0.40–0.54 against the one core by pairs. The pair 1.80–2.41× S6's on the same core. Every checksum
CH4.0h's.

## Pending runs

None. Kathleen job 634421 (the matrix under the degree rule, `rotation`, `crossover`) finished
and is in the report §4 and §5; W1 at kD = 1000 on locust (after the self-review) in §5.

## Open questions

The PR's questions for sign-off (report §7): (1) the acceptance read against CH4.0e's one
core; (2) `RcrSplit::Auto` by degree as the default; (3) CH4.0l off the priority list.

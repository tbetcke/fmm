# CH4.3b2d The waves' V crossover on raised levels, in one plan

Started 2026-11-21 on `tbetcke/h-ch4-3b2d`. Roadmap: CH4.3b2d "The waves' V crossover on raised
levels, in one plan" (added at CH4.3b2b, its question 3, signed off 2026-11-21 on #245: "Q1-4 as
recommended"; before CH4.3b2c on the priority list). Proposal: §10 CH4.3 ("the time crossover per
class"); §3.6, §3.7 (the near rules); §5.3 and §6.4 (D8, the waves); §2 (constraints 1, 2, 7, 9).
CH4.3b2b's report §4–§7.

## Goal
Price the V near rule's crossover before the plan on a level whose waves are raised (p_w > p_l,
ε < 10⁻⁶) by the waves' cost per box at p_w, so that one plan converts the leaf pairs that
CH4.3b2b's second plan (`VAfterPlan::AllLevels`) converted, and decide its adoption.

## Acceptance (the roadmap's CH4.3b2d)
"(CH4.3b2b's report §5–§7, question 3, signed off 2026-11-21) on a level with waves raised to
p_w > p_l (ε < 10⁻⁶), the V near rule's crossover before the plan priced by the waves' cost per box
at p_w (not the per-pair constant `VCosts::plane_wave`), one plan; against `VAfterPlan::AllLevels`
on the raised levels as the trusted measure: pair evidence (per level, the conversions and the
levels' M2L and P2P) and FMM evidence (W1 at kD = 100–500 and W3 at N = 10⁶, kD = 200 and 500, at
10⁻⁷ and 10⁻⁸, both run orders), W3's dense raised levels included (n_s n_t up to 256² per pair);
the errors within 1.1× the defaults'; adoption for sign-off."

How it is read:
- **"Priced by the waves' cost per box, before the plan."** CH4.3b2b's after-plan crossover is
  c_w(p_w) B / P over the P2P's per interaction: c_w = `DiagonalCosts::plane_wave` (thread-ns per
  box with a V row, a (p + 1)² + b (p + 1)³), B the level's boxes with a V row and P its V pairs
  after the first plan, so the mean per pair. Before the plan B and P are unknown. They are
  counted from the octree's keys: per local leaf t of a raised level, its V entries by the
  per-key rule (the children of the interior same-level neighbours of t's parent that are not
  adjacent to t; nd-fmm-plan's `interaction_manager` docs), less those the empty-box rule leaves
  out (a source box without source points; t without target points) and those between leaves
  that the V rule converts at the per-pair crossover (as built after the first runs, 43f29ad: the
  trusted measure prices the pairs the first plan leaves); B the leaves with an entry left, P
  those entries, summed over the ranks. Leaves only: they are the only targets whose V row can
  empty (only pairs between leaves convert), each local leaf is held by one rank, so the sums are
  the one-rank tree's (constraints 2 and 9); a raised level is a fine level, mostly leaves on W1
  (CH4.3b2b §4: 98.7–100 % of the pairs converted were leaf pairs).
  The crossover is the higher of the per-pair constant's and the per-box one, per (parent,
  source) as before (`v_group`).
- **"One plan."** No second plan: the counts take one pass over the local leaves of the raised
  levels (26 lookups per parent) and one all-reduce.
- **"Against `AllLevels` on the raised levels."** A new `VAfterPlan::RaisedLevels` (the levels
  that keep their waves at a raised degree, the after-plan crossover, the second plan), so that the
  trusted measure changes the raised levels only (`AllLevels` also changes the unraised levels,
  which over-convert: CH4.3b2b question 4).
- **"Pair evidence."** Per raised level: the crossovers (per pair, per box, after the plan), the V
  pairs left and converted, the change of the level's M2L and P2P seconds, per-box against
  per-pair and against the trusted measure.
- **"FMM evidence."** The evaluation, the build (the counts' time; no "replan"), the errors (within
  1.1× the defaults'), on the cases above in both run orders, under the defaults and under the time
  rule (`--diagonal off,auto`: CH4.3b2c makes the time rule the default next).
- **"Adoption for sign-off."** The new pricing as `VRaisedPrice::PerBox` beside CH4.0d2's
  `PerPair`; if the evidence holds, `PerBox` the builder's default in this PR, for sign-off.

## Read first
- `fmm-exec/src/helmholtz/fmm.rs` `build` (the V crossovers, `v_group`, the after-plan block),
  `v_crossover`, `v_after_plan_crossover`, `VCosts`; `operator_diagonal.rs` `DiagonalCosts`.
- `fmm-plan/src/interaction_manager.rs` (the V rule), `near.rs` (`LeafCounts`).
- CH4.3b2b's report and record; `tests/helmholtz_fmm.rs` `check_v_after_plan`.

## Changes
- `nd-fmm-exec` (Helmholtz modules only):
  - `VRaisedPrice` (`PerPair`, CH4.0d2's; `PerBox`), `HelmholtzFmmBuilder::v_raised_price`;
    `raised_crossover` (a pure function of the costs, p_w and the counts, constraint 9);
    `leaf_v_shapes` and `LeafVShape` (per level, the local leaves with a V entry left and those
    entries, from the octree's keys and the counts); in `build`, before the plan, the counts of
    the raised levels (agreed: one all-reduce sum) and the crossovers (one all-reduce min);
    `HelmholtzFmm::{v_raised_price, raised_v_shapes}`; `HelmholtzBuildTimings::counts` includes
    them.
  - `VAfterPlan::RaisedLevels`.
- `nd-fmm-validate`: `Case::v_raised_price` (`None` the library's), `Row::{v_raised_price,
  raised_v_shapes}`, the example's `--v-raised pair|box,…` (the column "M2L" ending `/vbox` per
  box) and `--v-after-plan raised` (`/vafterraised`), the counts as the last column of "Near
  pairs".
- `tools/helmholtz/ch4-3b2d-runs.sh`, `ch4-3b2d-tables.py`; the report
  `fmm-validate/results/helmholtz-ch4-3b2d.md`.
- No convention, no dependency, no `nd-fmm-plan` change.

## Tests that prove it
- Unit (`fmm.rs`): `raised_crossover` by hand (c_w B / P over the P2P, without gradients, no
  pairs).
- `tests/helmholtz_fmm.rs` `check_v_raised` (a sphere at kD = 30, ε = 10⁻⁷, Auto with waves from
  degree 0, NEON's V costs named, so that the plans are the same on every machine class):
  - the counts against the plan: per level, the computed local leaves with a V row and their V
    entries of `PerPair`'s plan equal `raised_v_shapes`, on every rank count (with the empty-box
    rule on); where the raised level's boxes with a V row are leaves, the per-box crossover the
    trusted measure's after-plan crossover;
  - `PerBox`: the crossovers raised exactly on the raised levels with waves and leaf pairs, by
    `raised_crossover`; V pairs per level at most and conversions at least `PerPair`'s; no second
    plan; the errors within `NEAR_ERROR_BOUND` (1.1) of `PerPair`'s; 2 threads and on P ranks the
    one-rank FMM bit for bit; Rotation's plan and output unchanged;
  - `RaisedLevels`: raised exactly the raised levels with waves and leaf pairs; 2 threads, one
    rank.
- `fmm-validate`'s `tests/helmholtz_accuracy.rs`: the columns on W1 at kD = 30, 10⁻⁷.

## Checks
`/verify` (fmt; Clippy default and workspace all-targets; rustdoc `-D warnings`; the default and
workspace tests at opt-level 1; the MPI list at 2 and 4 ranks); `cargo test -p nd-fmm-validate`.

## Machine runs
Kathleen, `singlenode`, 1 × 40, `tools/helmholtz/ch4-3b2d-runs.sh` (release, BLAS threads 1):
W1 at kD = 100, 200, 500 and W3 at N = 10⁶, kD = 200 and 500, at 10⁻⁷ and 10⁻⁸, `--m2l auto-pw
--near-v cost --diagonal off,auto --v-raised pair,box --v-after-plan off,raised`, and the same
in the reverse order. Decides: whether `PerBox` matches the trusted measure on the raised levels
(their M2L and P2P within the spread of `RaisedLevels`' and below `PerPair`'s), costs no second
plan, and keeps the errors within 1.1×; then its adoption as the default.

## Constraints and risks
- Constraint 1: no Laplace path (Helmholtz modules, the harness, scripts).
- Constraints 2 and 9: the counts from keys and agreed counts, summed over the ranks; the
  crossovers agreed (all-reduce min, as before); every rank takes the same branch.
- Constraint 7: the counts hold two integers per level.
- Leaves only: a mixed raised level (W3's level 4, interior boxes beside dense leaves) gets the
  leaves' mean, not the level's; the runs show whether that matters (W3's dense leaves, 256² per
  pair, lie above both).
- Under the time rule the form may take a raised level after the plan; the plan then converts
  more pairs than the form's crossover would, still exact. The runs show the cost.

## Pending runs
(filled at step 6)

## Open questions
(filled when they arise)

# CH4.3b2b The V rule on the form's levels, and the defaults' errors at 10⁻⁶

Started 2026-11-21 on `tbetcke/h-ch4-3b2b`. Roadmap: CH4.3b2b "The V rule on the form's levels,
and the defaults' errors at 10⁻⁶" (the second of CH4.3b2a–c, split on 2026-11-20). Proposal: §10
CH4.3 ("the time crossover per class"); §3.6 (the rule per (level, class)); §3.7 (the near rules);
§5.3 and §6.4 (D8); §2 (constraints 1, 2, 7, 9). CH4.3b1's report §5 (the errors at 10⁻⁶),
CH4.3b2a's report §6 and its question 3 (signed off 2026-11-21 on #244: "the sparse level closed
here and carried to CH4.3b2b").

## Goal
Explain why the defaults err up to 1.35× Rotation's at 10⁻⁶, and decide the V near rule's
crossover on the levels whose M2L the time rule decides after the plan (and on the sparse level
of CH4.3b2a's §6), by the cost model per pair and by the FMM.

## Acceptance (the roadmap's CH4.3b2b)
"(CH4.3b1's report §5) the defaults' errors at 10⁻⁶ (up to 1.35× Rotation's; 0.77–0.95× of them
when the plan saw no waves on the levels the form took, CH4.3b1's first runs) explained, and the
V near rule's crossover on a level the time rule takes from the waves (set for the waves' cost
before the plan) re-decided after the plan or justified as it is; pair evidence and FMM evidence,
both. Also (CH4.3b2a's report §6): the level that keeps 58,551 V pairs under the defaults' plan
against Rotation's plan's 2,296 (W1 at kD = 200, level 5, 10⁻⁸; RCR and the waves 0.30–0.32 s
there at p_w against RCR's 0.012 s on Rotation's plan), CH4.3b1's "31×"."

How it is read:
- **"Explained."** The hypothesis from the archived runs (CH4.3b1's Kathleen job 653681): the
  defaults' and Rotation's M2L err alike (CH6.1e: the waves within 1.07× Rotation's), but their
  plans differ. The V near rule (CH4.0d2, `VNear::CostModel`) converts a V pair between leaves
  into an exact near pair below the crossover of the level's cheapest M2L, per (parent, source)
  on a level with waves (`v_group`), per pair elsewhere. RCR is dear per pair, so Rotation's plan
  converts 2–3× as many pairs (W1 at kD = 200, 10⁻⁶: 63,426 of 136,740 against 19,131), and each
  converted pair loses its truncation error. CH4.3b1's first runs gave the levels up before the
  plan, so the V rule ran at RCR's crossover there (50,902 converted, 1.034× Rotation's error).
  It is confirmed if, with the V rule off (`--near-v off`), the defaults err within CH6.1e's
  1.07× of Rotation's on the same cases, and the ratio follows the converted pairs.
- **"Re-decided after the plan."** Today the builder's V rule reads `wave_levels` before the plan:
  on a level with waves the crossover is the waves' per-pair constant (`VCosts::plane_wave`
  b (p_w + 1)²), grouped per (parent, source). After the plan the operator knows the M2L it runs
  per level (the time rule's choice on the agreed `DiagonalShape`). The candidate: an opt-in
  builder setting `v_after_plan` that, after the operator's choice, raises each level's crossover
  to that of the M2L it runs, by the after-plan model, and builds the plan once more if any
  crossover rose:
  - a level taken from the waves by the form: the form's pair (`DiagonalCosts::pair` at p_w) over
    the P2P's per interaction, per pair (the grouping is the waves' alone);
  - taken by RCR: RCR's pair (`VCosts::rotation` at p_w), per pair;
  - a level that keeps its waves (`AllLevels` only, the Also's level): the waves' level cost by
    the time rule's model (`DiagonalCosts::plane_wave` per box with a V row, times those boxes)
    over the level's pairs, grouped as before;
  - every other level unchanged.
  The second plan converts the first plan's pairs and those the raised crossover adds (the union
  of the two predicates, so its V lists are a subset of the first's: every buffer the operator
  sized from the first plan holds), the operator's M2L per level as decided on the first plan.
  Only under `VNear::CostModel` (a named crossover is the caller's).
- **"Justified as it is."** If the second plan does not pay (its build against the evaluation's
  gain, and the evaluation itself), the report says so with the numbers and the setting stays
  off; the decision is a question for sign-off either way.
- **"Pair evidence."** Per level, the model's crossovers before and after, the pairs the second
  plan converts, and the modelled change (M2L pairs × the run's per-pair cost, P2P interactions ×
  `VCosts::p2p`) against the measured change of the level's M2L and P2P seconds.
- **"FMM evidence."** The evaluation, the build (with the second plan's part), the errors, in the
  same launch, on W1 and W3 at 10⁻³–10⁻⁸ on Kathleen.

## Read first
- `fmm-exec/src/helmholtz/fmm.rs` `build` (the V rule's crossovers, `v_group`, the plan, the
  shapes), `v_crossover`, `VCosts`.
- `fmm-exec/src/helmholtz/operator.rs` `select_tables`, `taken_from_waves`, `wave_levels`;
  `operator_diagonal.rs` `DiagonalCosts`, `wave_level_choice`.
- CH4.0d2's record (the V rule, its grouping), CH4.3b1's report §5, CH4.3b2a's report §6.
- `tests/helmholtz_fmm.rs` `check_near_v`, `check_time_rule_raised`.

## Changes
- `nd-fmm-exec` (Helmholtz modules only):
  - `VAfterPlan` (`Off`, the default; `TakenLevels`; `AllLevels`) and
    `HelmholtzFmmBuilder::v_after_plan`; `v_after_plan_crossover` (a pure function of the costs,
    the M2L run, its degree and the level's shape, constraint 9);
  - in `build`: after the operator, the raised crossovers (agreed: computed from agreed shapes
    and agreed costs, so every rank decides alike and reaches the second plan's collectives or
    none, constraint 2), the second plan on a clone of the index, the evaluator on it;
    `HelmholtzBuildTimings::replan`; `HelmholtzFmm::{v_after_plan, v_crossovers}` (as planned
    and after the plan).
- `nd-fmm-validate`: `Case::v_after_plan`, `Row::{v_after_plan, v_crossovers}`, the build part
  "replan", the example's `--v-after-plan off|taken|all,…` (the column "M2L" ending `/vafter` or
  `/vafterall`), the crossovers in the "Near pairs" table per level.
- `tools/helmholtz/ch4-3b2b-runs.sh` and `ch4-3b2b-tables.py`; the report
  `fmm-validate/results/helmholtz-ch4-3b2b.md`.
- No convention, no dependency, no `nd-fmm-plan` change (`Plan::with_rules_on` as it is).

## Tests that prove it
- Unit (`fmm.rs`): `v_after_plan_crossover` by hand: the form's pair, RCR's pair, the waves' level
  cost over its pairs, never below the planned crossover.
- `tests/helmholtz_fmm.rs` `check_v_after_plan` (a tree with V pairs between leaves on levels with
  waves in the time rule's range; `VNear::CostModel`, `HelmholtzDiagonal::Auto`, named
  `DiagonalCosts` that make the form take a level with a dear pair, and with the form dearer than
  the waves for `AllLevels`): the operator's M2L per level the same as without the setting; the
  crossovers raised where the model says and nowhere else; the V pairs per level at most and the
  near interactions at least those without; the errors against the direct sum within
  `NEAR_ERROR_BOUND` (1.1) of the setting off's; 2 threads and on P ranks the one-rank FMM bit for
  bit; with nothing to raise (Rotation) no second plan and the setting off's output bit for bit.
- `fmm-validate`'s `tests/helmholtz_accuracy.rs`: the new columns on W1 at kD = 30.

## Checks
`/verify` (fmt; Clippy default and workspace all-targets; rustdoc `-D warnings`; the default and
workspace tests at opt-level 1; the MPI list at 2 and 4 ranks); `cargo test -p nd-fmm-validate`.

## Machine runs
Kathleen, `singlenode`, 1 × 40, `tools/helmholtz/ch4-3b2b-runs.sh` (release, BLAS threads 1):
- `errors`: W1 at kD = 100, 200, 500, 1000 and W3 at N = 10⁶, kD = 200, 500, at 10⁻⁶ (and 10⁻³),
  `--m2l rotation,auto-pw --near-v off,cost`: the explanation.
- `w1`, `w3`: W1 at kD = 100, 200, 500 (10⁻³, 10⁻⁶, 10⁻⁷, 10⁻⁸) and 1000 (10⁻⁶), W3 at N = 10⁶,
  kD = 200 and 500 (the same ε), `--m2l auto-pw --near-v cost --diagonal off,auto
  --v-after-plan off,taken,all` and Rotation's plan beside: the decision.
Decides: the explanation, and whether `v_after_plan` (and which form) is proposed for adoption
with CH4.3b2c's default.

## Constraints and risks
- Constraint 1: no Laplace path (Helmholtz modules, the harness, scripts).
- Constraints 2 and 9: the raised crossovers from agreed shapes and costs; the second plan on
  every rank or none; the output on P ranks the one-rank FMM's bit for bit.
- Constraint 7: the second plan holds two plans briefly (the first dropped after it); the plan's
  memory is reported by the build. Its time is the risk: the plan is 4.5 s of W1 at kD = 1000's
  15.6 s build and 14 s of W3 at kD = 500's 17 s (CH4.3b2a's runs), so a second plan may cost
  more than it gains; the report weighs both.
- The waves' after-plan crossover uses the level's mean cost per pair; a converted pair saves a
  box's waves only if its box leaves every V list, so the mean may overstate the gain on dense
  levels. The FMM decides.

## As built
- `VAfterPlan`, `HelmholtzFmmBuilder::v_after_plan`, `v_after_plan_crossover` (`VLevelM2l`) as
  planned. Two changes after the plan: the second plan takes a fresh `BoxIndex::of_octree` (the
  first plan's index has the ghost leaves numbered, which `Plan::with_rules_on` refuses: the MPI
  run at 2 ranks failed with the clone, 3ecfa8a), and a level is raised only where its first plan
  still holds a V pair between leaves (summed over the ranks; c315554), after job 653717 built a
  second, identical plan on W3 at kD = 500 (14 s) and where only levels without leaf pairs rose.
- After the sign-off, CI's x86 runners failed `check_v_after_plan`: the builder takes `VCosts` by
  ISA, x86's waves' crossover per (p + 1)² is about 11× NEON's, so x86's first plan already
  converted every leaf pair of the sphere's level 3 and nothing rose (reproduced on the M6 with
  `V_COSTS_X86` named). `HelmholtzFmmBuilder::v_costs` (default `None`, the ISA's) names the near
  rules' costs; the test names NEON's, so its plans are the same on every machine class.
- The runs got a `reverse` stage (the same runs in the reverse order, job 653724): same-plan pairs
  in one launch differed by up to 30 % in the evaluation, so §5 of the report reads the levels' own
  seconds in both orders.

## Results
The report: fmm-validate/results/helmholtz-ch4-3b2b.md.
- The explanation (§2): with the V rule off the defaults err 0.966–1.024× Rotation's; with it on
  1.000–1.344×, with the pairs converted (Rotation's plan 1.6–9.2× as many).
- The levels taken from the waves (§4, §5): the re-decision's change −0.12 to +0.02 s per case,
  within the spread, against a second plan as dear as the first: justified as it is (question 2).
- The raised levels that keep their waves (§5, §6): the after-plan crossover converts their leaf
  pairs and gains 0.09–0.75 s per case in both orders, linearly in the pairs converted; the
  Also's level is one of them. A one-plan rule is the follow-up CH4.3b2d (question 3).
- Unraised dense levels with waves: the mean over-converts, +0.02–0.45 s (question 4).

## Pending runs
None. Kathleen job 653717 (node-c11a-021, 2026-11-21 00:29–01:36, `singlenode`, 1 × 40, at 9c1f72e:
`ch4-3b2b-runs.sh target/runs/ch4-3b2b errors w1 w3`, 54 launches, every one exit 0) and job 653724
(node-c11a-006, 02:02–02:17, at 5184d30: `reverse`, 15 launches, every one exit 0). Raw outputs in
`~/projects/fmm-archive/kathleen/runs/ch4-3b2b{,-reverse}/`.

## Open questions
None. The report's §7, questions 1–4, signed off 2026-11-21 by the user on #245, "Q1-4 as
recommended": 1, the explanation as read, nothing changes; 2, the levels taken from the waves
justified as they are, `VAfterPlan` stays opt-in; 3, the follow-up CH4.3b2d before CH4.3b2c (in the
roadmap); 4, unraised levels with waves not adopted, no task.

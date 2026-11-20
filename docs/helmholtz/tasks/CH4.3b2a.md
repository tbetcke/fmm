# CH4.3b2a The time rule at matched accuracy

Started 2026-11-20 on `tbetcke/h-ch4-3b2a`. Roadmap: CH4.3b2a "The time rule at matched
accuracy" (CH4.3b2 split on 2026-11-20 on this branch into CH4.3b2a, CH4.3b2b and CH4.3b2c by
the size rule of `/next-task`: a rule change in the operator, an error study of the V rule, and
the default with its runs at 1 × N and N × 1 on two machines). Proposal: §10 CH4.3 ("the time
crossover per class"); §3.6 (the rule per (level, class)); §5.3 and §6.4 (D8, Auto's exponential
M2L, the resolving degree of CH6.1g and CH6.1h); §5.4 (the level call); §2 (constraints 1, 2, 7,
9). CH4.3b1's report §5 and its question 4 (signed off 2026-11-20 on #243: "CH4.3b2 decides at
matched accuracy as its line says (the form at the raised degree on such a level, or the waves
kept below 10⁻⁷ where they are the more accurate)").

## Goal
Make the diagonal form's time rule (`HelmholtzDiagonal::Auto`) give up no accuracy against the
defaults: a level with waves keeps its resolving degree p_w whatever M2L the rule gives it, and
the rule takes the cheapest of the waves, the form and RCR at that degree.

## Acceptance (the roadmap's CH4.3b2a as split; proposal §10 CH4.3, the time crossover)
"Under `HelmholtzDiagonal::Auto` a level with waves keeps its resolving degree p_w (CH6.1g,
CH6.1h) whatever M2L the time rule gives it, and the rule takes the cheapest of the waves, the
form and RCR at p_w by the model (k a ≥ 2.8; the form's scratch budget first); W1 at kD = 100,
200, 500 (and 1000 at 10⁻⁶) and W3 at N = 10⁶, kD = 200 and 500, at 10⁻³, 10⁻⁶, 10⁻⁷ and 10⁻⁸
on Kathleen (1 × 40) and locust (1 × 72): the defaults with the time rule within 1.1× the
defaults' errors (matched accuracy; CH4.3b1: 1.07–2.2× at 10⁻⁸), the choices per level against
each option timed at p_w, the evaluation; every output bit for bit at every thread count, chunk
and rank count. Also (CH4.3b1's report): the raised waves on a sparse level (31× RCR, Kathleen,
W1 at kD = 200, level 5) and the order of waves, form and RCR there."

How it is read:
- **"Keeps its resolving degree."** The degree the defaults give a level with waves
  (`plane_wave_degree(p_l, ε)`: max(p_l, 28) at 10⁻⁷ ≤ ε < 10⁻⁶, max(p_l, 32) at 10⁻⁸ ≤ ε < 10⁻⁷,
  p_l at ε ≥ 10⁻⁶) is the level's degree under the setting, the M2L's choice comes after it. The
  translations and the leaves' operators keep `PlaneWaveRaise`'s form unchanged (`Leaves` at
  10⁻⁸, `M2l` at 10⁻⁷): only the M2L changes on the level. The form's classes are the bound's at
  p_w (more pass at the higher degree), RCR's tables at p_w (as built now for every level).
- **"The cheapest of three."** On a level with waves in the time rule's range (k a ≥ 2.8), by
  `DiagonalCosts` on the agreed `DiagonalShape`: the waves per box at p_w, RCR per pair at p_w,
  the form (if the bound passes a class and its scratch fits the budget) at p_w. Ties keep the
  waves (D8). Outside the range, or under another setting, nothing changes. A level that leaves
  its waves for RCR is a change of D8's choice under the opt-in time rule only; D8 itself is
  CH4.3b2c's (amended there if the default moves it).
- **"Within 1.1× the defaults' errors"**: relative ℓ2 of φ and ∇φ at §9.1's samples, the defaults
  with `--diagonal auto` over the defaults with `--diagonal off`, every case.
- **"Each option timed at p_w"**: the same launch runs the defaults (the waves), the time rule
  with the machine's costs, and the time rule with costs that force the form (where a class
  passes; else RCR) and RCR on every level the rule reaches; the choice is scored against the
  fastest of the three per level, as CH4.3b1 §4 did (within 1.05×).

## Read first
- `fmm-exec/src/helmholtz/operator.rs` `select_tables` (the waves given up, the degrees, the RCR
  levels), `wave_levels`; `operator_diagonal.rs` `select`, `diagonal_beats_*`, `DiagonalCosts`.
- `fmm-tables/src/helmholtz/plane_wave/merge.rs` `plane_wave_degree`.
- CH4.3b1's record and report §3–§5; `tools/helmholtz/ch4-3b1-{runs.sh,fit.py}`.
- `fmm-exec/tests/helmholtz/diagonal.rs` (`the_time_rule_against_the_waves`),
  `tests/helmholtz_fmm.rs` (`check_diagonal_auto`).

## Changes
- `nd-fmm-exec` (Helmholtz modules only):
  - `select_tables`: under `HelmholtzDiagonal::Auto` with `HelmholtzM2l::Auto` and a tolerance,
    on each level with waves in range: the classes at p_w, the three options' costs at p_w
    (`diagonal_level`, `rcr_level`, `plane_wave_level`); the waves given up where one of the
    others is cheaper; the level keeps p_w in `degrees` (a vector of raised levels apart from the
    precisions); `diagonal::select` then takes the form or RCR at p_w by `diagonal_beats_rcr` as
    now. The form's budget check at p_w.
  - A helper `time_rule_on_waves(costs, p_w, shape, classes) -> WaveChoice` (or the like) beside
    `diagonal_beats_plane_waves`, public as the others, so the tests and the harness restate it.
  - Docs: the operator module's "M2L" (the time rule), `HelmholtzDiagonal::Auto`,
    `fmm-exec/CLAUDE.md`.
- `nd-fmm-validate`: the example's `--diagonal-costs machine|form|rcr,…` (each `auto` run with
  each; `Case::diagonal_costs`), the column "M2L" ending `/diagauto:form` or `:rcr`; `CLAUDE.md`.
- `tools/helmholtz/ch4-3b2a-runs.sh` and `ch4-3b2a-tables.py`; the report
  `fmm-validate/results/helmholtz-ch4-3b2a.md`.
- No default change (`HelmholtzDiagonal::Off` stays the default until CH4.3b2c), no convention,
  no dependency, no `nd-fmm-plan` interface, no Laplace file.

## Tests that prove it
- `tests/helmholtz/diagonal.rs`, `the_time_rule_against_the_waves` rewritten: at 10⁻⁶ and 10⁻⁷
  (and 10⁻⁸) on k = 40, every level with waves in range: under free form costs the form at
  p_w = the defaults' degree with the bound's classes at p_w; under free RCR costs RCR at p_w;
  under dear costs (the waves free) the defaults' levels and degrees; a tight budget: the waves or
  RCR by the costs, never the form; below k a 2.8 the defaults' levels; the translation degrees
  the defaults' on every level under every costs.
- Unit (`operator_diagonal.rs`): the three-way choice by hand (each option cheapest in turn, the
  ties to the waves, no classes or no fit leaves two).
- `tests/helmholtz_fmm.rs`, `check_diagonal_auto` gains the defaults at 10⁻⁷ and 10⁻⁸ on a
  scenario with a raised level in range (the cube at kD = 40 + 2i has one at 10⁻⁷, checked when
  writing): the time rule within `DIAGONAL_RATIO` of the defaults' errors (the setting's without
  the form), the degrees the defaults', 2 threads, chunks and the one-rank FMM bit for bit. If no
  scenario of the debug run reaches a raised level in range, a release-only case and the
  operator tests carry it (stated in the record).
- `fmm-validate/tests/helmholtz_accuracy.rs` (`check_diagonal`): an `auto` row with named costs.

## Checks
`/verify` (fmt, Clippy, rustdoc, the default and workspace tests at opt-level 1, the MPI list at
2 and 4 ranks: `helmholtz_fmm` changes); the one-rank `helmholtz_fmm` and `nd-fmm-exec`'s
Helmholtz executable at opt-level 0 against their budget; `nd-fmm-validate`'s
`helmholtz_accuracy` test.

## Machine runs
- **Kathleen** (`singlenode`, 1 × 40, one job) and **locust** (1 × 72, idle checked):
  `ch4-3b2a-runs.sh` (the defaults, `--m2l auto-pw`, × `--diagonal off,auto` × `--diagonal-costs
  machine,form,rcr`): W1 at kD = 100, 200, 500 and W3 at N = 10⁶, kD = 200 and 500 at 10⁻³, 10⁻⁶,
  10⁻⁷, 10⁻⁸; W1 at kD = 1000 at 10⁻⁶. Decides: the acceptance (errors, choices, evaluation).
  `ch4-3b2a-tables.py`: per launch the errors and evaluation of each setting over the defaults';
  per level the three options' seconds and the machine's choice (within 1.05×).

## Constraints and risks
- Constraint 1: no Laplace path (Helmholtz modules of `nd-fmm-exec`, the harness, scripts).
- Constraints 2 and 9: every choice is a function of agreed counts, (p, k̂, ε, T) and agreed
  constants; the degrees are the defaults' (a function of k, the widths and ε).
- Constraint 7: the form's scratch budget first, at p_w (the form's scratch grows with p_w).
- Risk: the waves' per-box constant (fitted at p_w's range on W1 and W3) moves sparse levels to
  RCR where it is wrong; measured per level against the three options' times.
- Risk: the form at p_w costs (p_w/p_l)² more per pair and up to (p_w/p_l)³ per box, so at
  10⁻⁸ it may win fewer levels than CH4.3b1's runs (which ran it at p_l); measured.
- The V rule's crossover on a level the rule takes from the waves stays the waves' (CH4.3b2b).

## As built: the order on a level with waves
The plan took the cheapest of three, RCR included (c02d2b3). The first runs (Kathleen job 653691,
locust) put RCR against the waves on seven levels: 0.45 s lost and 0.11 s gained against keeping
the waves, RCR's constant (CH4.3b1's, fitted at the degrees given) under-predicting RCR at p_w on
sparse, low-k a levels (the report §4). As built (a4bc1c2): the waves unless the form beats them at
p_w, then the cheaper of the form and RCR (`wave_level_choice`); RCR never against the waves alone.
The roadmap's CH4.3b2a line is amended in this PR; the report's question 2. The forced costs of the
harness follow: `RCR_COSTS` forces RCR only where a class passes. The FMM check
(`check_time_rule_raised`) runs the cubes at kD = 20 + 2i (no class at p_w, so the waves stay) in
release builds only, and in debug builds the forced costs alone, 2 threads at free form costs alone,
on 600 points.

## Pending runs
None. The runs are done:
- first runs (the rule of three, c02d2b3): Kathleen job 653691 (node-c11a-044, `singlenode`, 1 × 40,
  2026-11-20 22:05–22:34), `ch4-3b2a-runs.sh target/runs/ch4-3b2a w1 w3`, 21 launches, every one
  exit 0; locust 1 × 72, 22:05–22:29, the same, with W1 at kD = 1000 rerun alone at 22:29–22:32
  (another user's load during the first: 15-minute average 316; the rerun's 1-minute load at most 33);
- final runs (as built, a4bc1c2): Kathleen job 653702 (node-c11a-018, 22:57–23:26, queued 17 min),
  `ch4-3b2a-runs.sh target/runs/ch4-3b2a-final w1 w3`, 21 launches, every one exit 0; locust
  22:39–22:51, the load sampled every 30 s, at most 45, every launch exit 0.
Raw outputs in `~/projects/fmm-archive/{kathleen,locust}/runs/ch4-3b2a{,-final,-rerun,-contaminated}/`.

## Results
The report: fmm-validate/results/helmholtz-ch4-3b2a.md. Under the time rule the errors are
0.990–1.037× the defaults' on every case on both machines, 0.993–1.000× at 10⁻⁸ (CH4.3b1: 1.07–2.2×);
the choices within 1.05× of the fastest on 81/83 levels (Kathleen, 0.54 s lost of 48.0 s) and 82/92
(locust, 0.28 s of 13.8 s), every miss the waves kept where the form was faster (W3's dense levels
and two raised W1 levels); the evaluation 0.38–1.12× the defaults' on Kathleen and 0.43–0.99× on
locust (W3 at kD = 200 on Kathleen 1.03–1.12×: the same waves level slower in the time rule's run of
the launch, question 4). CH4.3b1's sparse level (31× RCR) is the V near rule's plan (58,551 V pairs
against Rotation's plan's 2,296), not the M2L: CH4.3b2b. At the raised degree the bound passes fewer
classes at low k a (the cube at k = 40, level 2: 16 at 10⁻⁷, 5 at 10⁻⁸).

Debug test time (opt-level 0, one rank, the M6): `helmholtz_fmm` 646 s at load 2–7 (CH4.2c2: 611 s at
load 7–10; the first version of the check, 803 s at load 3–35, was trimmed), `nd-fmm-exec`'s
`helmholtz` executable 51.5 s (under its minute). At opt-level 1 `helmholtz_fmm` 33.6 s at one rank
(CH4.3b1: 33 s).

## Open questions
The report's §7, questions 1–4, for sign-off.

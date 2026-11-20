# CH4.3b1 The time rule per level, built and calibrated

Started 2026-11-20 on `tbetcke/h-ch4-3b1`. Roadmap: CH4.3b1 "The time rule per level, built and
calibrated" (CH4.3b split on 2026-11-20 on this branch into CH4.3b1, the rule opt-in, and CH4.3b2,
the default, by the size rule of `/next-task`: a cost model and a rule in the operator, plus
measurements on two machines). Proposal: §10 CH4.3 ("the time crossover per class; the fallback
of a whole level to RCR when the scratch budget binds"); §3.6 (the rule per (level, class)); §5.4
(the level call); §5.3 and §6.4 (D8, Auto's exponential M2L); §2 (constraints 1, 2, 7, 9).

## Goal
Give the diagonal form a rule by time: a cost model of its level call per machine class, fitted on
Kathleen and locust, and an opt-in setting `HelmholtzDiagonal::Auto` that takes the bound's
classes on a level only where the model says the diagonal level beats the M2L the level would
otherwise run (RCR, or the exponential M2L), so that CH4.3b2 can measure it as the default.

## Acceptance (the roadmap's CH4.3b1 as split; proposal §10 CH4.3, its second and third items in part)
(The order against the waves is decided after the plan as built, "As built" below; the roadmap's
line is amended in this PR, the report's question 2.)

"The diagonal level's costs per machine class (S → F and F → R per box, the pair per sample;
thread-ns, as `VCosts`) fitted on Kathleen (1 × 40) and locust (1 × 72) from W1's diagonal levels,
beside `VCosts`' RCR and exponential M2L; the per-class crossover reported (the diagonal pair
against RCR's per pair at every degree); `HelmholtzDiagonal::Auto`, opt-in: on each level the
bound's classes, taken where the model's diagonal level beats the level's M2L under the setting
(RCR after the plan, by the agreed V pairs per class and boxes; the exponential M2L before it, in
`wave_levels`, by degree, k̂ and ε, so that the builder's V rule and the operator agree, CHA.11),
after the scratch budget's whole-level fallback; the model's choice against the measured level
times on W1 and W3; the FMM under it within 1.1× Rotation's errors, bit for bit at every thread
count, chunk and rank count. Also (CH4.3a's question 3): the levels below k a = 2.8 that the bound
takes on W1 and W3 at 10⁻³–10⁻⁸, and the rule bounded to k a ≥ 2.8, or C_T raised there, if any
is."

How it is read:
- **The per-class crossover.** A class's pair costs one complex multiply-add per sample (S(p) =
  rings × azimuths, about 2p² at the level's degree) against RCR's O(p³) per pair; the
  conversions are per box, not per class. So once a level pays its conversions every class that
  passes the bound is cheaper diagonal, and the decision is per level. The report shows it: the
  fitted pair cost against `VCosts::rotation` at every p from 8 to 245 (the ratio, and the p, if
  any, where it reaches 1).
- **The level's decision.** On level l with the bound's classes C (θ ε, the build's C_T):
  T_diag = n_s c_S→F(p) + n_t c_F→R(p) + Σ_{c ∈ C} n_c c_pair S(p) + Σ_{c ∉ C} n_c c_RCR(p),
  against T_RCR = n c_RCR(p) (or, as built, T_waves = n_t c_pw(p_w) per box with a V row where
  Auto gives the level waves; the plan had it per pair).
  The diagonal form takes the level iff T_diag < T_alt. After the plan the counts are the agreed
  ones (each box counted by the rank that computes it, summed over the ranks: the same on every
  rank and at every rank count, as `LevelShape`), n_s taken as n_t (the V relation is symmetric
  on a level where every box is both; the S → F count is the distinct sources, the model's one
  approximation, measured). The plan decided the order against the waves before the plan on a
  uniform interior level; as built it is decided after the plan on the same agreed counts
  ("As built" below).
- **Under Auto the order against Batched is unchanged:** the diagonal form considers the levels
  that are RCR's or the waves' under the setting, not Batched's (p ≤ 11, below the high-frequency
  range where classes pass; reported if a Batched level would have passed).
- **"Within 1.1× Rotation's errors"** as CH4.2b1–CH4.3a read it (`DIAGONAL_RATIO`, relative ℓ2
  of φ and ∇φ at §9.1's samples). Against the defaults' waves at matched accuracy is CH4.3b2's
  (at 10⁻⁸ the raised waves err 0.6× Rotation's, CH4.2b2 §2).

## Read first
- `fmm-exec/src/helmholtz/operator_diagonal.rs` (`select`, `diagonal_rule_classes`), `operator.rs`
  (`select_tables`, `wave_levels`, `WaveSettings`), `fmm.rs` (the shapes' and sources'
  all-reduces, `VCosts`, `v_costs`, `v_crossover`, `WxCosts`).
- `fmm-tables/src/helmholtz/diagonal/` (`class_bounds`, `Conversions::multiply_adds`,
  `DiagonalM2l` sample maps); `nd_fmm_ref::helmholtz::diagonal::classes::class_of`.
- CH4.2b2's report §3–§4 (the level times and parts), CH4.3a's report (the classes per level).
- `fmm-validate/examples/helmholtz_accuracy.rs` (`--diagonal`, `--levels`' "Diagonal M2L").

## Changes
- `nd-fmm-exec` (Helmholtz modules only):
  - `DiagonalCosts { s2f, f2r, pair, rotation, plane_wave }`, each [a, b] of
    a (p + 1)² + b (p + 1)³ thread-ns (S → F and F → R per box, the diagonal pair, RCR's pair
    and the exponential M2L's pair at p_w), `DIAGONAL_COSTS_X86`, `DIAGONAL_COSTS_NEON`,
    `diagonal_costs(isa)`, beside `VCosts`. Its own RCR and waves constants, not `VCosts`':
    the rule compares whole level calls, so every constant is the level call's wall time ×
    threads per unit, idle threads included; `VCosts` fits thread time on the leaf levels for
    the V rule, and its waves per pair is 4–7× below the level call's on W1 at kD = 500
    (CH4.2b2's run: 367 µs per pair at p = 41 against 53). The form: the conversions per box on
    CH4.2b2's run fit a (p + 1)² + b (p + 1)³ within 10 % from p = 16 to 131 (a ≈ 0.45 µs,
    b ≈ 1.6 ns), where a pure (p + 1)³ misses p = 41 by half;
  - `HelmholtzDiagonal::Auto`: the bound's classes, then the level by the model; a new
    `DiagonalShape` per level (boxes with a V row, V pairs per class, `offset_class`), which the
    builder counts whatever the setting (every rank reaches the all-reduce), and the costs agreed
    (all-reduce max of the constants, as m*); `with_diagonal_costs`, `with_diagonal_shapes`, the
    builder's `diagonal_costs`;
  - (as built) the waves given up in the operator's `select_tables` after the plan, by
    `diagonal_beats_plane_waves` on the level's shape; `wave_levels` unchanged (the builder's V
    rule reads a superset);
  - the Also's bound below k a = 2.8 (`DIAGONAL_AUTO_MIN_KA`, the time rule only);
  - the operator and builder docs, `fmm-exec/CLAUDE.md`.
- `nd-fmm-validate`: `--diagonal auto` in `helmholtz_accuracy` (`HelmholtzDiagonal::Auto`), the
  model's prediction per level in `--levels`' "Diagonal M2L" table (T_diag and T_alt), and the
  k a of each level.
- `tools/helmholtz/ch4-3b1-runs.sh` (Kathleen and locust stages) and `ch4-3b1-fit.py` (the fit
  and the report's tables); the report `fmm-validate/results/helmholtz-ch4-3b1.md`.
- No default changes (`HelmholtzDiagonal::Off` stays the default until CH4.3b2), no convention,
  no dependency, no `nd-fmm-plan` interface (the V view's rows and groups are read as they are),
  no Laplace file.

## Tests that prove it
- Unit (`operator_diagonal.rs`): the model by hand (as built: hand constants at p = 1, against
  RCR by the classes taken and against the waves by the pairs per box); the uniform
  level's class mix sums to 189 pairs and to 8 × 189 over the child positions; `diagonal_costs`
  by ISA; the pair against RCR per pair below 1 at every p from 8 to 245 with both machine
  classes' constants.
- `tests/helmholtz/diagonal.rs`: `Auto` on hand `LevelShape`s per level (a subset of `Rule`'s
  levels, the same classes where it takes one, `Off`'s M2L where it takes none; the scratch
  budget's fallback first); without shapes, as `Rule` on the RCR levels; f32 below 10⁻⁴ none.
- `fmm-exec/src/helmholtz/operator.rs`' `wave_levels_are_the_operators` with `Auto`: the
  operator's wave levels within the builder's (as built); `tests/helmholtz/diagonal.rs`
  (`the_time_rule_against_the_waves`): a wave level taken by the diagonal form under `Auto`
  runs at its own degree, a tight budget keeps the waves, none below k a = 2.8.
- `tests/helmholtz_fmm.rs`, `check_diagonal_auto` (in `check_diagonal`; the cube at kD = 40 + 2i
  and the clusters at kD = 20, 10⁻⁴; under Rotation and the library's Auto): the errors within
  `DIAGONAL_RATIO` of the setting's without the form, the levels and classes the same on every
  rank and the one-rank FMM's, 2 threads and chunks bit for bit (at free costs), the one-rank FMM
  bit for bit, and the agreed shapes against the one-rank FMM's.
- `fmm-validate/tests/helmholtz_accuracy.rs` (`check_diagonal`): an `auto` row on W1 at kD = 30.

## Checks
`/verify` (fmt, Clippy, rustdoc, the default and workspace tests at opt-level 1, the MPI list at
2 and 4 ranks: `helmholtz_fmm` changes); the one-rank `helmholtz_fmm` and `nd-fmm-exec`'s
Helmholtz executable at opt-level 0 against their minute (the debug budget); `nd-fmm-validate`'s
`helmholtz_accuracy` test.

## Machine runs
- **Kathleen** (`singlenode`, 1 × 40, one job): (1) the fit: W1 at kD = 100, 200, 500, 1000 at
  10⁻⁶ and 10⁻⁸ (kD = 1000 at 10⁻⁶ only), `--m2l rotation,auto-pw --diagonal off,rule --levels`
  (the diagonal levels' parts per level, RCR's and the waves' level times); a first fit from
  CH4.2b2's archived run (`~/projects/fmm-archive/kathleen/runs/ch4-2b2/`) starts the code; (2) the check:
  the same cases and W3 (N = 10⁶, kD = 200, 500) at 10⁻³, 10⁻⁶, 10⁻⁸ with `--diagonal
  off,rule,auto`: per level the model's choice against the faster measured one, the errors.
- **locust** (1 × 72, idle checked): the same two stages at kD ≤ 500 and kD = 1000 at 10⁻⁶.
- Decides: the constants (adopted only after stage 2 confirms the choice per level); the Also.

## Constraints and risks
- Constraint 1: no Laplace path (Helmholtz modules of `nd-fmm-exec`, the harness).
- Constraints 2 and 9: the choice is a function of agreed counts, (p, k̂, ε, T) and agreed
  constants, so every rank and every rank count chooses the same levels (the scratch budget's
  fallback depends on the ranks per node, as before).
- Constraint 7: the budget fallback is unchanged and comes first.
- Risk: the uniform-level model before the plan misjudges a sparse level (W1 is a surface:
  shorter rows, conversions weigh more). Stage 2 measures it; a wrong call costs time, never
  accuracy (both forms are within the bound).
- Risk: at 10⁻⁸ a level moved from the raised waves to the diagonal form runs at p_l and loses
  the raise's accuracy; measured in stage 2, decided in CH4.3b2.

## As built: the order against the waves after the plan
The plan decided the order against the exponential M2L before the plan, in `wave_levels`, on the
uniform level, with the waves per pair. The first runs (Kathleen job 653672, locust, at 164ddd7)
refuted it: both passes of the waves cost per box with a V row whatever the pairs per box, the
diagonal form per pair, so the order turns on the pairs per box (W1 about 45, W3 about 165); the
uniform level (189) put W3's dense levels on the form at up to 1.9× the waves' time and lost 5.1 s
of 64.8 s on Kathleen. As built (258261a, 14b2103):
- `DiagonalCosts::plane_wave` per box with a V row; `diagonal_beats_plane_waves` on the level's
  agreed shape, in the operator's `select_tables` after the plan; the builder's V rule reads
  `wave_levels` before the plan as before, a superset of the operator's levels with waves (its
  grouping per (parent, source) is exact under every M2L);
- a level taken from the waves runs the cheaper of the form and RCR by the same model (Kathleen,
  kD = 200, 10⁻⁸, level 5: raised waves 0.363 s, RCR 0.012 s, the form with 3 classes 0.020 s);
- the fit weighs each level by its measured time; x86's constants from Kathleen, NEON's from
  locust (`tools/helmholtz/ch4-3b1-fit.py fit`), replayed by `check` from the measured counts:
  122/127 and 133/140 levels within 1.05× the fastest;
- the Also: no level below k a = 2.8 under the time rule (`DIAGONAL_AUTO_MIN_KA`).
The roadmap's CH4.3b1 line is amended in this PR; the report's question 2.

## Pending runs
None. The runs are done:
- Kathleen job 653672 (node-c11a-080, `singlenode`, 1 × 40, 2026-11-20 19:57–20:27, at 164ddd7):
  `ch4-3b1-runs.sh target/runs/ch4-3b1 w1 w3`, 16 launches, every one exit 0 (the fit's data);
- locust (1 × 72, idle, 19:47–19:59, at 164ddd7): the same, 16 launches, exit 0;
- Kathleen job 653681 (node-c11a-080, 20:31–20:51, at 14b2103) and locust (20:30–20:51, at
  14b2103): `auto`, 16 launches each, exit 0 (the final choices and errors);
- (locust at 258261a, the first design after the plan with NEON's refit and the old x86
  constants: archived as `ch4-3b1-auto-258261a`, not used in the report).
Raw outputs in `~/projects/fmm-archive/{kathleen,locust}/runs/ch4-3b1{,-auto}/`.

## Results
The report: fmm-validate/results/helmholtz-ch4-3b1.md. The per-class crossover is never reached
(a diagonal pair 0.08–0.52× RCR's at p = 8–245); the final runs' choices within 1.05× of the
fastest on 123/127 levels (Kathleen) and 132/140 (locust); under Rotation 1.000–1.001× its
errors, evaluation 0.27–1.00×; under the defaults their own errors at 10⁻³ and 10⁻⁶ and
Rotation's at 10⁻⁸ (1.07–2.2× the defaults'). The self-review (a subagent on the branch at
14b2103) found the budget's fallback after the waves were given up (fixed: a level whose form
would not fit keeps its waves, tested), a misplaced doc comment, the FMM check without chunks and
the one-rank shapes (added), the harness's choice column ungated (fixed), and stale wording here
and in the roadmap (fixed).

## Open questions
The report's §7, questions 1–4: signed off 2026-11-20 by the user on #243, "1-4 as recommended".

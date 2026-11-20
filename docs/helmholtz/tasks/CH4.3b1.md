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

## Acceptance (the roadmap's CH4.3b1; proposal §10 CH4.3, its second and third items in part)
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
  against T_RCR = n c_RCR(p) (or T_waves = n c_pw (p_w + 1)² where Auto gives the level waves).
  The diagonal form takes the level iff T_diag < T_alt. After the plan the counts are the agreed
  ones (each box counted by the rank that computes it, summed over the ranks: the same on every
  rank and at every rank count, as `LevelShape`), n_s taken as n_t (the V relation is symmetric
  on a level where every box is both; the S → F count is the distinct sources, the model's one
  approximation, measured). Before the plan (against the waves) there are no counts: the model
  takes an interior box of a uniform level (pairs per class of the 8 child positions' V lists,
  189 pairs per box), so it is a function of (p, k̂, ε, T, the costs) alone, as D8's rule; the
  builder and the operator read the same function (`wave_levels`, CHA.11).
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
  - `DiagonalCosts { s2f: [a, b], f2r: [a, b], pair: f64 }` (thread-ns: S → F and F → R per box
    a + b (p + 1)³, the pair per sample), `DIAGONAL_COSTS_X86`, `DIAGONAL_COSTS_NEON`,
    `diagonal_costs(isa)`, beside `VCosts`;
  - `HelmholtzDiagonal::Auto`: the bound's classes, then the level by the model;
    `LevelShape` gains the V pairs per class (34 counts per level); the builder counts them
    whatever the setting (every rank reaches the all-reduce) and agrees the costs (all-reduce
    max of the constants, as m*), `with_diagonal_costs` names them to the operator;
  - `wave_levels`' `WaveSettings` gains the diagonal setting, tolerance, build and costs: under
    `Auto` a level the waves would take goes to the diagonal form where the uniform-level model
    says so (its degree then the level's own, not CH6.1g's raised one);
  - the Also's bound below k a = 2.8 if the census finds such a level (else recorded);
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
- Unit (`operator_diagonal.rs`, `fmm.rs`): the model by hand on two shapes (every class passing
  at p = 72 with full rows: diagonal; few classes at p = 16 with short rows: RCR); the uniform
  level's class mix sums to 189 pairs and to 8 × 189 over the child positions; `diagonal_costs`
  by ISA; the pair against RCR per pair below 1 at every p from 8 to 245 with both machine
  classes' constants.
- `tests/helmholtz/diagonal.rs`: `Auto` on hand `LevelShape`s per level (a subset of `Rule`'s
  levels, the same classes where it takes one, `Off`'s M2L where it takes none; the scratch
  budget's fallback first); without shapes, as `Rule` on the RCR levels; f32 below 10⁻⁴ none.
- `fmm-exec/src/helmholtz/operator.rs`' `wave_levels_are_the_operators` with `Auto`: the
  builder's wave levels the operator's; `tests/helmholtz/plane_wave.rs`: a wave level taken by
  the diagonal form under `Auto` runs at its own degree.
- `tests/helmholtz_fmm.rs`, `check_diagonal_auto` (in `check_diagonal`; the cube at kD = 40 + 2i
  and the clusters at kD = 20, 10⁻⁴; under Rotation and the library's Auto): the errors within
  `DIAGONAL_RATIO` of the setting's without the form, the levels and classes the same on every
  rank and the one-rank FMM's, 2 threads and chunks bit for bit, the one-rank FMM bit for bit,
  and the agreed shapes per class against one rank's.
- `fmm-validate/tests/helmholtz_accuracy.rs` (`check_diagonal`): an `auto` row on W1 at kD = 30.

## Checks
`/verify` (fmt, Clippy, rustdoc, the default and workspace tests at opt-level 1, the MPI list at
2 and 4 ranks: `helmholtz_fmm` changes); the one-rank `helmholtz_fmm` and `nd-fmm-exec`'s
Helmholtz executable at opt-level 0 against their minute (the debug budget); `nd-fmm-validate`'s
`helmholtz_accuracy` test.

## Machine runs
- **Kathleen** (`singlenode`, 1 × 40, one job): (1) the fit: W1 at kD = 100, 200, 500, 1000 at
  10⁻⁶ and 10⁻⁸ (kD = 1000 at 10⁻⁶ only), `--m2l rotation,auto-pw --diagonal off,rule --levels`
  (the diagonal levels' parts per level, RCR's and the waves' level times); (2) the check:
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

## Pending runs
(filled in step 6)

## Open questions
(filled when they arise)

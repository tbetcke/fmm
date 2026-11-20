# CH4.2c2 The interpolated build in the operator, and W1's tables timed

Started 2026-11-20 on `tbetcke/h-ch4-2c2`. Roadmap: CH4.2c2 "The interpolated build in the
operator, and W1's tables timed" (the second half of CH4.2c, split on 2026-11-20; CH4.2c1, the
build, merged as 2cfed91). Proposal: §10 CH4.2 (CH4.2c is its follow-up, added at CH4.2b2's
question 2); §5.4 (the transfer functions "interpolated in ŝ · t̂ from an extended-precision
table in O(p²) per class with C_T = 0.23"); §3.6 (the bound, θ); §9.1 (W1); §2 (constraints 1,
2, 5, 6, 7, 9).

## Goal
The operator and the builder choose the transfer functions' build (direct or interpolated), the
rule reads the build's constants (C_T = 0.23 interpolated), the harness names it, and W1's
diagonal tables are timed by both builds at one rank per node and one rank per core on Kathleen,
with the FMM's errors and the classes per level, so that the default build can be signed off.

## Acceptance (the roadmap's CH4.2c2)
"the operator's build setting (the rule's constants by the build: C_T = 0.23 interpolated) and
the harness's; the FMM within 1.1× Rotation's errors with it; W1 at kD = 500 and 1000's diagonal
tables timed at 1 × 40 and 40 × 1 against CH4.2b2's (2.3 s and 61 s at p = 131), with the errors
and the classes per level against the direct build's; which build is the default, for sign-off."

With CH4.2c's: "W1 at kD = 500 and 1000's diagonal tables timed at 1 × 40 and 40 × 1 against
CH4.2b2's (2.3 s and 61 s at p = 131)". Proposal §10 has no CH4.2c row; CH4.2's row ("against RCR
on the same class within E(l, c)") was shown for the interpolated tables by CH4.2c1 (≤ 2E at
C_T = 0.23, signed off on #240). Here the FMM measure: "within 1.1× Rotation's errors" as
CH4.2b1 and CH4.2b2 read it, the relative ℓ2 errors of φ and ∇φ against the direct sum over the
Rotation run without the diagonal form.

## Read first
- `fmm-exec/src/helmholtz/operator_diagonal.rs` (`diagonal_rule_classes`, `select`,
  `DiagonalSettings`), `operator.rs` (`with_diagonal*`, `select_tables`, the module docs "M2L"),
  `fmm.rs` (the builder's diagonal setters and `build`).
- `fmm-tables/src/helmholtz/diagonal/transfer.rs` (`TransferBuild`, `bound_constants`,
  `DiagonalM2l::build_by`).
- `fmm-validate/src/helmholtz_accuracy.rs` (`Case::diagonal`, `Row::diagonal*`), the example's
  `--diagonal`, `tests/helmholtz_accuracy.rs` (`check_diagonal`), `tools/helmholtz/ch4-2b2-*`.
- CH4.2b2's report §2, §5, §6 (the baselines) and CH4.2c1's record (the per-class times).

## Changes
- `nd-fmm-exec` (Helmholtz modules only):
  - `nd_fmm_exec::helmholtz::TransferBuild`, re-exported from `nd-fmm-tables` (so a caller
    needs no tables dependency).
  - `DiagonalSettings::build`; `HelmholtzOperator::{with_diagonal_build, diagonal_build}`
    (tables chosen again), `HelmholtzFmmBuilder::diagonal_build` and `HelmholtzFmm`'s report
    through the operator.
  - `diagonal_rule_classes::<T>(p, k̂, ε, build)`: the constants `build.bound_constants(single)`
    (F64, F64_INTERPOLATED, or F32 for an f32 table either way); `select` builds by
    `DiagonalM2l::build_by(p, k̂, classes, build, parts)`.
  - The default: `TransferBuild::Direct` (CH4.2b's tables) until the sign-off; if the measured
    recommendation is Interpolated, the PR asks for it and the default moves on this branch only
    on the user's answer.
- `nd-fmm-validate`: `Case::diagonal_build` (named always, `Direct` in `Case::new`),
  `Row::diagonal_build_kind` (`Row::diagonal_build` already holds the build seconds); the example's `--diagonal-build direct|interpolated,…` (inside
  `--diagonal`'s loop; the column "M2L" ending `/interp`); `check_diagonal` with the interpolated
  build.
- `tools/helmholtz/ch4-2c2-runs.sh` and `ch4-2c2-tables.py`; the report
  `fmm-validate/results/helmholtz-ch4-2c2.md`.
- No convention, no dependency, no `nd-fmm-plan` interface, no Laplace file, no default of the
  diagonal form (still `Off` until CH4.3).

## Tests that prove it
- `fmm-exec/tests/helmholtz/diagonal.rs`:
  - the rule per build: under Interpolated the classes with E(F64_INTERPOLATED) ≤ θ ε, a subset
    of Direct's (C_T 0.23 > 0.15), strictly smaller on some level at a tolerance where a class
    lies between the two bounds; `diagonal_build` reported; f32 the same classes by either build;
  - the pair body with the interpolated tables on every offset of levels 2 and 3 at k = 40 (real
    and damped) within 2E(F64_INTERPOLATED) of Rotation's in CH1.3d1's measure.
- `fmm-exec/tests/helmholtz_fmm.rs`, `check_diagonal` under Rotation: the interpolated build's
  FMM within `DIAGONAL_RATIO` (1.1) of Rotation's errors (φ and ∇φ) against the direct sum, its
  classes per level a subset of the direct build's and the same on every rank, 2 threads bit for
  bit, on P ranks the one-rank FMM bit for bit; the cube at kD = 40 + 2i (damped) and the
  clusters at kD = 20, ε = 10⁻⁴.
- `fmm-validate/tests/helmholtz_accuracy.rs`, `check_diagonal`: the interpolated row's build
  reported, within 1.1× Rotation's errors, 1 and 2 threads the same checksum.

## Checks
`/verify`: fmt; Clippy (default, examples, workspace all-targets); rustdoc `-D warnings`; the
default and workspace tests at opt-level 1; the MPI list at 2 and 4 ranks (`helmholtz_fmm`
changes); `nd-fmm-exec`'s Helmholtz executable, `helmholtz_fmm` and `helmholtz_accuracy` at
opt-level 0 against their budgets. `cargo deny` not needed (no manifest).

## Machine runs
Kathleen (`singlenode`, one job, `run.sbatch`): release `helmholtz_accuracy`, BLAS threads 1,
W1 with §9.1's complex charges and samples, `--near-v cost --near-wx cost --levels
--evaluations 2`, `--m2l rotation --diagonal off,rule --diagonal-build direct,interpolated`:
- `node`: 1 × 40, kD = 500 at 10⁻⁶ and 10⁻⁸, kD = 1000 at 10⁻⁶ (CH4.2b2's `w1` cases);
- `cores`: 40 × 1, the same three cases (kD = 1000 at 40 × 1 not run by CH4.2b2: its direct
  build at p = 245 is estimated at minutes; CH4.1b's census sends levels 2–3 to RCR there).
What they decide: the default build (the tables' build time at both splits against the
evaluation, the errors 1.1× Rotation's, the classes the interpolated rule loses at 10⁻⁸ and the
time they cost). No assert on a time.

Wider scope (the `/next-task` rule): the interpolated build widens the diagonal form's constants
from C_T = 0.15 to 0.23. CH4.2c1 left the large degrees (k a ≥ 54, p ≥ 72) to its ignored tests
at E's floor (1.79–3.04 E against RCR, attributed to RCR's floor); W1 at kD = 1000 (p = 245 on
level 2) brings them into the FMM here. The pair measure is CH4.2c1's (signed off), the FMM's
is this task's.

## Constraints and risks
- Constraint 1: no Laplace path (Helmholtz modules of `nd-fmm-exec`, the harness).
- Constraint 2 and 9: the build a setting, the classes a function of p, k̂, ε, T and the build;
  the interpolated tables depend on libm (`atan2`, `sin`, `exp`), the same on every rank of a run.
- Constraint 6: the trusted paths are Rotation's FMM and the direct sum (FMM), RCR and the
  direct build (CH4.2c1's tables).
- Constraint 7: the 40 × 1 runs at kD = 1000 hold 7.96 million points on 40 ranks; CH4.2b2's
  40 × 1 at kD = 500 peaked at 1.1 GB per rank.
- Risk: under Interpolated fewer classes pass at 10⁻⁸ (CH4.2b2: 12 classes on level 6 by
  Direct), so a level may lose its diagonal classes; reported, a question for the default.
- Risk: the debug cost of `helmholtz_fmm` (CI's 900 s): two interpolated FMM runs per scenario
  and one-rank reference only.

## Pending runs
None. Kathleen job 653605 (node-c11a-058, `singlenode`, 2026-11-20 13:42–14:01, 19 min: the
release build, `node` and `cores`). Raw outputs in `~/projects/fmm-archive/kathleen/runs/ch4-2c2/`.

## Results
- Built as planned; `Row::diagonal_build_kind` in place of the plan's `Row::diagonal_build`
  (taken by the build seconds). After question 1's sign-off the default is `Interpolated`
  (`DEFAULT_DIAGONAL_BUILD`, operator and builder); the tests name `Direct` as the trusted path
  and `check_interpolated` runs the builder's default.
- Errors (report §2): the interpolated build 1.000× Rotation's errors (φ and ∇φ) on W1 at
  kD = 500 (10⁻⁶, 10⁻⁸) and 1000 (10⁻⁶) at 1 × 40 and 40 × 1, as the direct build; in CI's
  `check_interpolated` 1.000× on the cube at kD = 40 + 2i and the clusters at kD = 20.
- Classes (report §2): the same per level under both builds on every W1 case (10⁻⁸ included);
  the operator's unit test shows the looser C_T dropping classes 0 and 1 of level 2 at k = 40,
  10⁻⁴.
- Tables (report §3): at p = 131, CH4.2b2's 2.34 s (1 × 40) and 61.0 s (40 × 1) are now 1.80 and
  53.8 s direct by ring, 0.33 and 6.84 s interpolated; per run the interpolated diagonal tables
  4.0–5.7× faster at 1 × 40 (0.14–0.18 of an evaluation) and 3.3–6.4× at 40 × 1 (still 2.7× an
  evaluation at kD = 500: every rank builds every class, CH8.0's).
- Bits (report §4): each build one checksum at 1 × 40 and 40 × 1; the direct build CH4.2b2's
  `c74ca0e7954cf9ba`, so the ring split is CH4.2b2's FMM bit for bit.
- Debug times at opt-level 0 on the M6: `nd-fmm-exec`'s Helmholtz executable 51 s at load 4–9
  (under its minute; CH4.2b2 116 s at load 39–49); `helmholtz_fmm`'s one-rank run 611 s at load
  7–10 (CH4.2b2 1,807 s at load 39–49); `helmholtz_accuracy` 282 s at load 3–10 (CH4.2b2 849 s at load 39–49).

## Open questions
For sign-off on the PR (report §5): 1, the default build (recommended: `Interpolated`, with
`Direct` named and the tests' trusted path); 2, the acceptance as read.

Signed off 2026-11-20 by the user on #241: "Both questions as recommended". Proposal §5.4
amended (the interpolated build the default).

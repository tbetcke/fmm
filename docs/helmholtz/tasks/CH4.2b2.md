# CH4.2b2 The diagonal M2L on W1, measured

Started 2026-11-20 on `tbetcke/h-ch4-2b2`. Roadmap: CH4.2b2 (the second half of CH4.2b, split on
2026-11-20: "the FMM against the direct sum on W1 (kD = 500 at 10⁻⁶ and 10⁻⁸; kD = 1000 at 10⁻⁶)
within 1.1× RCR's error, the diagonal levels' M2L time against RCR's per level, at 1 × 40
(Kathleen) and 1 × 72 (locust)", with two Alsos: the conversions' rate, and the transfer tables'
build time per level). Proposal: §10 CH4.2; §3.6, §5.4 ("Scratch and chunking", the
interpolated build), §6.7 (the conversions' count), §7, §9.1 (W1), §2 (constraints 1, 2, 7, 9).

## Goal
W1's errors and the diagonal levels' M2L times with `HelmholtzDiagonal::Rule` against RCR on
Kathleen and locust, the level call's parts (S → F, the pairs, F → R, the RCR pairs) and the
conversions' rate, and the transfer tables' build time per level: the data CH4.3 calibrates on.

## Acceptance (roadmap CH4.2b; proposal §10 CH4.2)
Proposal §10, CH4.2: "against RCR on the same class within E(l, c); determinism with the RCR
pairs added after F → R" (shown at the table level by CH4.2a and in the FMM by CH4.2b1). The
roadmap's CH4.2b2: "the FMM against the direct sum on W1 (kD = 500 at 10⁻⁶ and 10⁻⁸; kD = 1000
at 10⁻⁶) within 1.1× RCR's error, the diagonal levels' M2L time against RCR's per level, at
1 × 40 (Kathleen) and 1 × 72 (locust)." No default changes (CH4.3 decides the rule's place).

## Read first
- `fmm-exec/src/helmholtz/operator_diagonal.rs` (`diagonal_m2l`, `select`, `DiagonalCounts`,
  `DiagonalScratch`), `operator_plane_wave.rs` (`PlaneWaveTimes`: wall per pass and the
  workers' thread-seconds drained after the level call, the pattern to follow).
- `fmm-tables/src/helmholtz/diagonal/mod.rs` (`Conversions::multiply_adds`).
- `fmm-validate/src/helmholtz_accuracy.rs` (`Case`, `Row`, `run`), its example (the run loops,
  `level_m2l`, `--levels`), `tests/helmholtz_accuracy.rs` (`check_levels`, `check_rcr_split`).
- The baselines: CH4.1a's report (the conversions at 3–5 GF/s on a Kathleen core, 6–10 on a
  Grace core), CH4.1b's (W1's memory, the scratch census), CH4.0i's (RCR by pairs on levels 2–3).

## Changes
- `nd-fmm-exec` (Helmholtz modules only):
  - `DiagonalTimes` per level (`HelmholtzOperator::diagonal_times`, reset with `m2l_times`):
    the wall seconds of the S → F pass and of the targets' pass on the calling thread's clock,
    and the workers' thread-seconds of S → F (the gather included), the diagonal pairs, F → R
    and the RCR pairs, each worker's clock in its `DiagonalScratch`, drained after the level
    call (as `PlaneWaveTimes`' parts). A few clock reads per block and per chunk; no value
    changed.
  - `DiagonalCounts::back`: the accumulators converted back (target × vector) per level, beside
    `signatures`, so the harness has both conversions' counts.
  - `HelmholtzOperator::diagonal_build_times`: per level the transfer functions' and the
    conversions' build seconds (`select`, the table build already timed in sum by
    `TableTimings::diagonal`), and `diagonal_multiply_adds(level)`: one conversion's real
    multiply-adds (`Conversions::multiply_adds`, both steps).
- `nd-fmm-validate`: `Case::diagonal` (`HelmholtzFmmBuilder::diagonal`, `Off` in `Case::new`)
  and `Row::{diagonal, diagonal_classes, diagonal_pairs, diagonal_times, diagonal_counts,
  diagonal_build, diagonal_multiply_adds}` (per level, the slowest rank's times, the counts
  summed); the example's `--diagonal off|rule|<ε>,…` (the outermost run loop; the column "M2L"
  ending `/diag` or `/diag<ε>`, `--levels`' "ran" `diagonal c/34` where a level took it); a
  table "Diagonal M2L" with `--levels`: per level the classes, the diagonal and RCR pairs, the
  M2L seconds, the two passes' seconds, the four parts' thread-seconds, the conversions' rate
  (GF/s per thread: 2 × multiply-adds × count over thread-seconds) and the tables' build;
  "÷ Rotation" also on a Rotation row with the diagonal form (against the row without it).
  `tests/helmholtz_accuracy.rs`: `check_diagonal` on W1 at kD = 30 (or a size where the rule
  takes some level): the columns, within 1.1× Rotation's error, the counts against the plan.
- `tools/helmholtz/ch4-2b2-runs.sh` (stages `w1`, `ranks`) and `ch4-2b2-tables.py`; the report
  `fmm-validate/results/helmholtz-ch4-2b2.md`.
- No convention, no dependency, no Laplace file, no `nd-fmm-plan` interface, no default.

## Tests that prove it
- `fmm-exec/tests/helmholtz_fmm.rs`, `check_diagonal`: on every diagonal level with V pairs
  both passes and the four parts timed (the RCR part where the level has RCR pairs), the
  passes within the level's M2L time, the parts' sum within the threads × passes; zero on
  every other level; `back` equals the targets × vectors with a diagonal pair (counted from the
  plan's view on one rank); the build times on the diagonal levels alone. The outputs as
  before (the times read no value).
- `fmm-validate/tests/helmholtz_accuracy.rs`, `check_diagonal`: the harness's row against the
  operator's reports, the FMM within 1.1× the Rotation row's errors, the same checksum at 1 and
  2 threads.

## Checks
`/verify`: fmt; Clippy (default, workspace all-targets); rustdoc `-D warnings`; default and
workspace tests at opt-level 1; the MPI list at 2 and 4 ranks (`helmholtz_fmm` changes);
`nd-fmm-exec`'s Helmholtz executable and `helmholtz_fmm` at opt-level 0 (the debug budgets);
`cargo deny` not needed (no manifest).

## Machine runs
Release `helmholtz_accuracy`, BLAS threads 1, `--near-v cost --near-wx cost` (the defaults),
`--levels`, complex charges, §9.1's samples:
- `w1` on Kathleen (`singlenode`, 1 × 40) and locust (1 × 72): W1 at kD = 500 at 10⁻⁶ and
  10⁻⁸ and at kD = 1000 at 10⁻⁶, each under `--m2l rotation` and the library's defaults
  (`auto-pw`), each `--diagonal off,rule`: the errors within 1.1× Rotation's (the acceptance),
  each level's M2L seconds against RCR's, the parts, the conversions' rate, the build per level.
- `ranks` on Kathleen: W1 at kD = 500, 10⁻⁶, `--m2l rotation --diagonal rule` at 2 × 20 and
  40 × 1: the checksum the 1 × 40 run's where the same levels run diagonal (constraint 2 at
  scale), and the levels the budget sends to RCR at 40 ranks per node.
These decide nothing by themselves: CH4.3 takes the crossover per class and the default.

## Constraints and risks
- Constraint 1: no Laplace path (Helmholtz modules of `nd-fmm-exec`, the harness).
- Constraint 2: the timings read clocks only; the bits unchanged (`check_diagonal`'s bit checks).
- Constraint 7: W1 at kD = 1000's levels 2–3 at 40 × 1 fall back (CH4.1b); not run here (Results).
- Risk: the transfer tables at p = 131–245 (kD = 1000's level 2) are built in double-double per
  class (ref's `Transfer`); their build may take minutes. Measured here, the reason for the Also.
- Risk: the debug cost of `helmholtz_fmm` (CI's 900 s): the new checks read reports only.

## Pending runs
None. Kathleen job 653533 (node-c11a-025, `singlenode`, 11 min: the build, `w1` and `ranks`);
locust 2026-11-20 08:44–08:48 (1 × 72, idle before: load 0.00, GPU 0 %). Raw outputs in
`~/projects/fmm-archive/{kathleen,locust}/runs/ch4-2b2/`.

## Results
- Built as planned. Deviations: the counts' `back` and the times live in the operator's
  diagonal buffer (reset with the counts), not in a new operator field; `DiagonalTimes` is the
  passes' wall time and the parts' thread-seconds in one type; the build times are
  `diagonal_build_time(level)` per level; the harness's "÷ Rotation" counts a Rotation row with
  the diagonal form against the one without; the GF/s column on one rank only (the
  self-review: counts summed against the slowest rank's seconds). Not done: the build against
  the interpolated build of §5.4 (none exists yet; proposed as CH4.2c), and kD = 1000 at 40 × 1
  (CH4.1b's fallback levels; its build alone would take minutes, report §5–6).
- The acceptance (report §2): 1.000× Rotation's errors (φ and ∇φ) on all three cases on both
  machines; under the defaults the form takes the levels Auto leaves on RCR, errors unchanged.
- The levels (report §3): 0.08–0.46× RCR's M2L where every class is diagonal, 0.95–1.02× on
  mixed levels (12 or 21 classes); Rotation with the form 1.6–2.8× faster than the defaults on
  the evaluation, its levels 0.29–0.83× the defaults' waves (a question for CH4.3).
- The conversions (report §4): 24–49 % of the call's thread time, 1.7–5.1 GF/s per thread at
  p ≥ 41 on Kathleen (40 threads), 3.3–11 on locust; the pairs 50–80 %.
- The build (report §5): one class at the top level per node (2.3 s at p = 131, 11.9 s at p = 245
  on Kathleen); every class serially at 40 × 1 (61 s at p = 131, against an evaluation of 3.9 s).
- Ranks (report §6): the same checksum at 1 × 40, 2 × 20 and 40 × 1; no level falls back at
  kD = 500 at 40 ranks per node.

## Open questions
For sign-off on the PR (report §7): 1, the acceptance as read; 2, a new task CH4.2c (the
interpolated transfer build in parts finer than a class) before CH4.3; 3, CH4.2d (dispatched
kernels for the pair and the conversions) off the list; 4, an Also on CH4.3: the order between
the exponential M2L (D8) and the diagonal form per level.

Signed off 2026-11-20 by the user on #239, all four as recommended: "All 4 questions as
recommended."

# CH6.1e The quadrature per level, measured

Started 2026-11-03 on `tbetcke/h-ch6-1e`. Roadmap: "CH6.1e The quadrature per level,
measured" (H6; priority list item 2, the last sub-task of CH6.1). Proposal: §10 CH6.1;
§5.3 (Exponential M2L), §2 (constraint 9), §9.1; CH6.1c's record (the degree study, its
question 3 signed off) and CH6.1d's record (the level call, the provisional rule).

## Goal

Set the precision rule of `HelmholtzM2l::PlaneWave` by the FMM's own errors on CH3.1c's
cases, and show the FMM under it within about 1.1× Rotation's error on every case.

## Acceptance (proposal §10, verbatim)

CH6.1: "errors within about 1.1× Rotation's, as T13's budget for Laplace". The roadmap's
CH6.1e:

> the rule's constants set by the FMM's errors with each precision forced per level on
> CH3.1c's cases; the FMM within about 1.1× Rotation's error on every case, asserted in
> the tests; the report names the levels left to Rotation.

With CH6.1c's sign-off (question 3): per level the coarsest precision whose error against
the direct sum stays within about 1.1× RCR's; a level where none does keeps Rotation.

Read as (to be confirmed in the PR): "error" is CH3.1c's, the relative ℓ2 errors of φ and
∇φ over the sampled targets against the direct sum, the ratio the larger of the two ratios
to Rotation's on the same case; "every case" is CH3.1c's matrix as CH3.3 verified it (62
cases, the degrees of the calibrated rule); "asserted in the tests" is the debug-run tests
of `nd-fmm-exec` and `nd-fmm-validate` on their small scenarios, plus the matrix in the
report from the example (the matrix runs for an hour or more and needs gigabytes, beyond a
test; `fmm-validate/CLAUDE.md`: tests run small smoke sweeps only).

## Read first

- `fmm-tables/src/helmholtz/plane_wave/merge.rs` (`precision_for_degree`, `select`,
  `MAX_PLANE_WAVE_DEGREE`), `quadrature.rs` (`Precision`, `domain_of`).
- `fmm-exec/src/helmholtz/operator.rs` (`PlaneWavePrecision`, `select_tables`),
  `fmm.rs` (the builder, `Degrees`).
- `fmm-validate/src/helmholtz_accuracy.rs` (`Case`, `run`, `Row`) and
  `examples/helmholtz_accuracy.rs` (`--m2l`); CH3.3's report §5 and §7 (the matrix and its
  commands).
- `fmm-exec/tests/helmholtz_fmm.rs` (`check_plane_wave`), `tests/helmholtz/plane_wave.rs`,
  `fmm-tables/tests/helmholtz/plane_wave_merge.rs` (the rule's tests).

## Changes

No Laplace file, convention, dependency or `nd-fmm-plan` change.

1. **nd-fmm-validate** (the measurement):
   - `helmholtz_accuracy::Case::plane_wave` (a `PlaneWavePrecision`, read under
     `HelmholtzM2l::PlaneWave`) and `Row::plane_wave` (each level's precision as built);
   - the example's `--m2l planewave` (the rule), `planewave:<1|2|3>` (one precision on
     every level with a quadrature) and `planewave-scan` (Rotation, then each level with a
     quadrature at each precision alone, the others Rotation; a column naming the forced
     level and precision, and the ratio to Rotation's error);
   - `tests/helmholtz_accuracy.rs`: the rule's case within 1.1× Rotation's (below).
2. **nd-fmm-tables** (`helmholtz::plane_wave::merge`): `precision_for_degree` and `select`
   with the measured constants. The rule a function of κ_l = k w_l, p_l and, if the
   measurement needs it, ε (constraint 9); then `select` takes the tolerance and
   `nd-fmm-exec` passes `Degrees::Tolerance`'s ε through `HelmholtzFmmBuilder` (named
   degrees: a stated fallback). The module docs and `fmm-tables/CLAUDE.md`.
3. **nd-fmm-exec**: what the rule's signature needs (above); `check_plane_wave` asserts
   1.1× Rotation's error; the module docs ("M2L") and `fmm-exec/CLAUDE.md`.
4. The report `fmm-validate/results/helmholtz-ch6-1e.md`: the scan, the rule's fit, the
   verification on the matrix, the levels left to Rotation.

## Tests that prove it

1. **The scan** (the example, release, the M6): per case of CH3.1c's matrix (CH3.3's
   degrees) and per level with a quadrature, the FMM with that level at iprec 1, 2 and 3
   alone, against the sampled direct sum (`nd_fmm_ref::helmholtz::p2p::direct_sum` in
   f64, the oracle of CH3.1c), the ratio to Rotation's error. The rule's constants: the
   coarsest precision whose ratio stays within a per-level share of 1.1× on every case
   at that (κ, p, ε), so that the levels together stay within 1.1×.
2. **The verification** (the example): every case of the matrix under the rule, the ratio
   to Rotation's error at most about 1.1×, each level's precision or Rotation named.
3. **Tests** (debug, one rank in CI; by hand at 2 and 4 ranks):
   - `fmm-tables/tests/helmholtz/plane_wave_merge.rs`: the rule at its new breakpoints;
   - `fmm-exec/tests/helmholtz_fmm.rs` `check_plane_wave`: the error ratio to Rotation's
     at most 1.1 on its scenarios (asserted where CH6.1d only printed), with the existing
     checks (direct sum, pair path, threads, ranks, vectors);
   - `fmm-validate/tests/helmholtz_accuracy.rs`: small cases of the matrix (W1 at kD = 10,
     ε = 10⁻³, 10⁻⁶, 10⁻⁸; W4 at N = 2,000) under the rule within 1.1× Rotation's error.

## Checks

`/verify` (nd-fmm-tables, nd-fmm-exec, nd-fmm-validate: format, Clippy default, examples
and workspace all-targets, `cargo doc`, `cargo test` and `--workspace`, the MPI list at 2
and 4 ranks); `cargo test -p nd-fmm-tables`, `-p nd-fmm-exec`, `-p nd-fmm-validate`.

## Machine runs

The M6, release, 1 rank × 6 threads (accuracy, not timing): the scan and the verification
above. The largest cases (W3 at kD = 100, W1 at kD = 500) scan only the levels with a
quadrature and are named in the report if cut for time or memory (W3 at kD = 100 at
ε = 10⁻⁸ needs about 12–21 GB).

## Constraints and risks

- Constraint 1: no Laplace code changes.
- Constraint 9: the rule from k, w_l, p_l (and ε) only, never the charges.
- Risk (CH6.1c's study): at p ≥ 20 no precision may hold 1.1×, so at ε = 10⁻⁸ the
  low-frequency levels (p = 24) keep Rotation and H6's gain shrinks to ε ≥ 10⁻⁶; the
  report names those levels.
- Risk: levels' extra errors add; the per-level share must leave the sum within 1.1×.

## Pending runs

## Open questions

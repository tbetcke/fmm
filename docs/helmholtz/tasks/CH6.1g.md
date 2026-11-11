# CH6.1g The quadrature below 10⁻⁶ in the FMM

Started 2026-11-11 on `tbetcke/h-ch6-1g`. Roadmap: "CH6.1g The quadrature below 10⁻⁶ in the
FMM" (H6; priority list item 6, the last part of CH6.1f "Quadratures for ε below 10⁻⁶").
Proposal: §10 CH6.1 ("errors within about 1.1× Rotation's"); §5.3 (Exponential M2L); §2
(constraints 1, 6, 7, 9); §12 D8. CH6.1f's report §3 and §5
(fmm-validate/results/helmholtz-ch6-1f.md), CH6.1e's report §2–§3, §7
(fmm-validate/results/helmholtz-ch6-1e.md).

## Goal

Run the exponential M2L below ε = 10⁻⁶ by CH6.1f's signed-off rule, the resolving degree:
a level with a quadrature at p_w = max(p_l, 24) with iprec 2 for 10⁻⁷ ≤ ε < 10⁻⁶ and at
p_w = max(p_l, 32) with iprec 3 for 10⁻⁸ ≤ ε < 10⁻⁷, the degree raised on the level; and
measure its errors and times beside Rotation's. (Changed 2026-11-11, "Open questions" 1:
iprec 3 at max(p_l, 28) for 10⁻⁷ ≤ ε < 10⁻⁶.)

## Acceptance (proposal §10, verbatim)

CH6.1: "errors within about 1.1× Rotation's, as T13's budget for Laplace". The roadmap's
CH6.1g:

> *Done when:* the FMM within about 1.1× Rotation's error at ε = 10⁻⁷ and 10⁻⁸ on CH6.1e's
> cases, the clustered trees' deep levels included, asserted in the tests on small cases,
> with the rule still a function of k, w_l, p_l and ε (constraint 9); times beside
> Rotation's; D8's bound ε ≥ 10⁻⁶ revisited as a question; or, if CH6.1f recommends none,
> the rule unchanged and the report's reason recorded.

CH6.1f recommended candidate 4, signed off 2026-11-09 (the user: "1–3 as recommended"):
(1) the resolving degree as above, Rotation below 10⁻⁸; (2) the degree raised on the level
first (`Degrees` per level: no new level call, the leaves' P2M and L2P pay too), on the M2L
alone only if that costs the gain; (3) `generate_for_degree` a trusted path only.

"CH6.1e's cases" at 10⁻⁷ and 10⁻⁸ (its `tools/helmholtz/ch6-1e-runs.sh`, `small` at 10⁻⁸
and `between` at 10⁻⁷; CH6.1e's sign-off 1 left the two large cases at 10⁻⁸ to Rotation):
W1 at kD = 10, 30, 100 and W3 at kD = 10, 30 at both ε; the cube, the sphere, the Plummer
sphere and the clusters (W4, N = 10⁵) at kD = 1, 10, 100 at 10⁻⁸; the cube and the clusters
at kD = 1, 100 at 10⁻⁷. 26 cases.

## Read first

- `fmm-tables/src/helmholtz/plane_wave/merge.rs` (`precision_for_tolerance`, `select`,
  `PLANE_WAVE_MIN_TOLERANCE`, `MergeLevel::new`).
- `fmm-exec/src/helmholtz/operator.rs` (`with_block_budget`, `select_tables`,
  `auto_takes_plane_waves`, the "M2L" module docs), `fmm-exec/src/helmholtz/fmm.rs` (step 7:
  the degrees and the operator; `HelmholtzFmm::degrees`).
- `fmm-exec/tests/helmholtz/plane_wave.rs` (the precisions per level, Auto's rule),
  `fmm-exec/tests/helmholtz_fmm.rs` (`check_plane_wave`, the scenarios),
  `fmm-validate/tests/helmholtz_accuracy.rs` (`check_plane_wave`, W1 at 10⁻⁸).
- `tools/helmholtz/ch6-1e-runs.sh` and `ch6-2h-runs.sh` (the cases, ABBA, the launcher).

## Changes

No convention, dependency, `nd-fmm-plan` or Laplace change.

1. **nd-fmm-tables, `helmholtz::plane_wave::merge`** (the rule):
   - `precision_for_tolerance(ε)`: iprec 2 for ε ≥ 10⁻⁶, iprec 3 for 10⁻⁸ ≤ ε < 10⁻⁶
     (`RESOLVING_MIN_TOLERANCE`, the smallest tolerance with a precision); none below or
     for ε not positive and finite (as built after question 1; the plan first had iprec 2
     to 10⁻⁷).
   - `resolving_degree(ε)`: 0 for ε ≥ 10⁻⁶ (`PLANE_WAVE_MIN_TOLERANCE`, kept: CH6.1e's
     bound, the smallest tolerance at which a level runs at its own degree, and D8's),
     28 for 10⁻⁷ ≤ ε < 10⁻⁶ (`RESOLVING_STEP_TOLERANCE`), 32 for 10⁻⁸ ≤ ε < 10⁻⁷
     (`RESOLVING_DEGREE`, `RESOLVING_DEGREE_TIGHT`);
     `plane_wave_degree(p, ε)` = max(p, d).
   - `select` unchanged in form (it reads `precision_for_tolerance`); the module docs
     record CH6.1f's study and this task's measurement.
   - Named degrees (no tolerance) unchanged: iprec 2 to p = 15, never raised.
2. **nd-fmm-exec, `HelmholtzOperator`**: the kernel keeps the degrees it was given
   (`base_degrees`) and the block budget; `select_tables` computes the precisions on them,
   raises the degree of every level with a precision under `PlaneWavePrecision::Rule` and a
   tolerance (`plane_wave_degree`), and builds `HelmholtzTables` again where the degrees
   change (and back when a later `with_m2l` takes the waves away), so every operator of the
   level (P2M, M2M, L2L, L2P, P2L, M2P and the store sizes) runs at p_w. Named precisions
   (`PerLevel`, the measurements) are not raised. `degree(level)` reports the degree run;
   `base_degree(level)` the one given.
   - **Auto keeps D8's bound** (signed off: "CH6.1e's rule gives a quadrature (ε ≥ 10⁻⁶)"):
     its filter adds that the level needs no raise (the tolerance, if any, at least
     `PLANE_WAVE_MIN_TOLERANCE`). Lifting it is this task's question for sign-off.
3. **nd-fmm-exec, `HelmholtzFmmBuilder`/`HelmholtzFmm`**: `degrees()` the operator's (the
   degrees run), so that the harness's rows show p_w; the module docs say so.
4. **nd-fmm-validate**: `tests/helmholtz_accuracy.rs`'s W1 at kD = 10 and ε = 10⁻⁸ now
   checks the rule's iprec 3 at raised degrees within 1.1× Rotation's (it asserted
   Rotation's output below the old rule's tolerance). The harness itself unchanged (its
   rows print `fmm.degrees()`).
5. **tools**: `tools/helmholtz/ch6-1g-runs.sh` (the 26 cases, `--m2l rotation,planewave`
   ABBA, `--evaluations 2 --levels`), `ch6-1g-tables.py` (the report's tables).
6. Docs: fmm-tables/CLAUDE.md, fmm-exec/CLAUDE.md, the operator's and builder's module
   docs; the report `fmm-validate/results/helmholtz-ch6-1g.md`.

## Tests that prove it

- nd-fmm-tables `tests/helmholtz/plane_wave_merge.rs`: the rule's tolerances and precisions
  (iprec 2 to 10⁻⁷, iprec 3 to 10⁻⁸, none below, the bounds exact), the resolving degree
  and `plane_wave_degree` at and around each bound; `select` as before.
- nd-fmm-exec `tests/helmholtz/plane_wave.rs` (no MPI): an operator at ε = 10⁻⁷ and 10⁻⁸
  under PlaneWave has the rule's precisions and the degrees max(p_l, d) exactly on the
  levels with waves, the base degrees elsewhere; `with_m2l(Rotation)` afterwards restores
  the base degrees and the store sizes; named precisions raise nothing; Auto at 10⁻⁸ takes
  no waves and raises nothing (D8's bound); at 10⁻⁶ nothing raised (bit for bit as before).
- nd-fmm-exec `tests/helmholtz_fmm.rs` (`check_resolving_degree`, debug and CI): the cube at
  kD = 1 and the clusters at kD = 1 (deep levels at κ_l ≤ 0.08), N about 1,000, at
  ε = 10⁻⁷ and 10⁻⁸: PlaneWave's errors against the direct sum within `PLANE_WAVE_RATIO`
  (1.1) of Rotation's at the base degrees, the degrees and precisions per level as the
  rule's, 2 threads bit for bit, on P ranks the one-rank FMM bit for bit.
- nd-fmm-validate `tests/helmholtz_accuracy.rs`: W1 at kD = 10, ε = 10⁻⁸ under the rule
  within 1.1× Rotation's (with the reach and split checks it already runs).
- The report's runs: the 26 cases, every one within 1.1× Rotation's (φ and ∇φ).

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`,
`cargo test`, `cargo test --workspace`, the MPI list at 2 and 4 ranks; Laplace untouched);
`cargo test -p nd-fmm-validate --test helmholtz_accuracy`; the tables' ignored merge test if
it reads the rule (it does not: it checks shifts).

## Machine runs

Kathleen, one node 1 × 40 (exclusive, no quiet gate needed), `tools/helmholtz/ch6-1g-runs.sh`
through `run.sbatch`: errors and times of Rotation and PlaneWave (the rule) on the 26 cases,
each case forward and reversed (A B B A), the M2L per level and the evaluation. Decides the
acceptance (errors) and the times beside Rotation, and so the D8 question. No M6 timing
(the user's machine; errors are deterministic, Kathleen's suffice).

## Constraints and risks

- Constraint 1: Laplace untouched (the Helmholtz merge rule, operator and builder; tests).
- Constraint 9: p_w and the precision are functions of ε and p_l; the levels with a
  quadrature of k and w_l; every rank chooses alike.
- Constraint 7: a raised level's tables and stores are larger (p_w = 32: (33/25)² = 1.74×
  the coefficients per box); `plane_wave_bytes` and the stores count them.
- Constraint 6: errors against the direct sum and Rotation's; on P ranks the one-rank FMM.
- Risk: the raised leaves' P2M and L2P and the M2M and L2L at p_w eat the M2L's gain
  (CH6.1f §3: M2M and L2L 1.69× and 2.30× at the raised levels); then the report says so
  and sign-off 2's option (b), the M2L alone, is the follow-up.
- Risk: the debug cost of `tests/helmholtz_fmm.rs` (255 s one rank on the M6); the new
  check stays small (two trees of about 1,000 points).

## Results

The report `fmm-validate/results/helmholtz-ch6-1g.md`. Kathleen jobs 632472 (ε = 10⁻⁷, 85 s
of runs) and 632473 (10⁻⁸, 270 s), one node 1 × 40, 20 launches, all exit 0; raw outputs in
`~/projects/fmm-archive/kathleen/logs/ch6-1g/`. Every one of CH6.1e's 26 cases within 1.1×
Rotation's error (largest 0.80, φ, W1 at kD = 100, 10⁻⁷). Times beside Rotation: report §3.
The changes after question 1 (iprec 3 at 28) were synced to Kathleen before both jobs ran; the
Auto change of question 2 came after them and does not touch what they ran (`rotation`,
`planewave`).

## Pending runs

None.

## Open questions

1. **iprec 2 at 10⁻⁷ misses in the FMM** (2026-11-11, `check_resolving_degree`'s first
   debug run). iprec 2's own error (stated up to 10⁻⁷) lies above Rotation's on the tests'
   trees at 10⁻⁷ whatever the degree: on the clusters at kD = 1 (N = 1,000 and 4,000) and
   the cube at kD = 20 + 2i (N = 1,000 and 4,000), release, the M6, φ over Rotation's:
   iprec 2 at p_l 4.18×, 6.61×, 1.15×, 1.42×; at max(p_l, 24) 4.09×, 6.56×, 0.42×, 1.25×;
   at 32 the same as at 24. iprec 3 at p_l 0.94–0.99×, at max(p_l, 24) 0.04–0.10×, at 32
   at most 0.002×. At 10⁻⁸ the signed-off iprec 3 at max(p_l, 32) 0.004–0.014×. CH6.1f's
   pair chain compared with RCR's worst-case error (5.8e-8 at p = 20), above iprec 2's
   floor, so it did not show. Asked; **signed off 2026-11-11: "iprec 3 at max(p,28)"**
   (the recommendation: on the pair chain iprec 3 at p + 8 = 28 is 0.02–0.93× RCR's error at
   every κ of CH6.1f's study at 10⁻⁷). The 10⁻⁸ band unchanged.
2. **D8's bound** (the task's question; report §4): **signed off 2026-11-11, "Lift for
   10⁻⁸–10⁻⁷ only (Recommended)"**: Auto takes the waves at 10⁻⁸ ≤ ε < 10⁻⁷, keeps Rotation at
   10⁻⁷ ≤ ε < 10⁻⁶ until CH6.1h. Built on this branch with its tests.
3. **The M2L-alone raise** (CH6.1f's option (b)): **signed off 2026-11-11, "Add CH6.1h, next on
   the list (Recommended)"**: roadmap line and priority list item 6.

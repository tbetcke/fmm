# CH6.2g The counts' reach adopted

Started 2026-11-08 on `tbetcke/h-ch6-2g`. Roadmap: "CH6.2g The counts' reach adopted" (H6;
priority list item 4; added 2026-11-07 by CH6.2e, before CH6.2f by its question 2).
Proposal: §10 CH6.1 and CH6.2; §2 (constraints 1, 6, 9); CONVENTIONS §3.15 "Exponential
M2L"; CH6.2e's report `fmm-validate/results/helmholtz-ch6-2e.md` (§1, §4, §8).

## Goal

Make the α counts for lateral offsets ρ ≤ 4.5 the default of the Helmholtz exponential M2L
in place of the region's corner 4√2, as signed off on 2026-11-07 (CH6.2e, question 1).

**Outcome (2026-11-08): not adopted; the default stays the corner.** The largest-offset pair
check failed at high κ (see "Results"); asked, the user answered "Keep the corner
(Recommended)", and confirmed it after a correction of the question (it said wrongly that
CH6.2e had no level with waves above κ ≈ 25; W1 at kD = 500 has κ = 31.25 on level 4). The
task lands the guard test, the sampling check and the report
`fmm-validate/results/helmholtz-ch6-2g.md`; the default-change steps below were implemented,
then reverted before commit.

## Acceptance (proposal §10, verbatim)

CH6.1: "errors within about 1.1× Rotation's, as T13's budget for Laplace". CH6.2: "per-level
accuracy and time beside FMM3D's on the same problems; adopted as a strategy where it wins,
else kept opt-in or dropped (D8)". This sub-task's, from the roadmap's CH6.2g:

> the α counts for lateral offsets ρ ≤ 4.5 in place of the corner 4√2 as the default
> (`RHO_MAX`'s role), 0.84–0.86× the waves; CH6.1a's acceptance and the quadrature module's
> bound restated on ρ ≤ 4.5, CONVENTIONS §3.15's sentence on the quadrature's region amended
> (its subsection "Exponential M2L" is unversioned, so no version bump), `nd-fmm-ref`'s
> generator's counts left at the corner or moved alike (its tests decide); the sampling of ρ
> in `tail_count` checked (on some strongly damped domains a smaller reach gives more waves,
> CH6.2e's report §1). *Done when:* the default reach changed; the kernel within 2× and the
> chain within 3× the stated error on ρ ≤ 4.5 on every domain (the ignored test, which
> asserts both since CH6.2e); one M2L pair at the largest V offset (3, 3, 2) at the level's
> degree against Rotation on every domain at the new reach (the FMM's chain, which CH6.2e
> did not test apart); CH6.1e's verification (`tools/helmholtz/ch6-1e-runs.sh verify
> verify-large`) within 1.1× Rotation's on every case; nd-fmm-exec's
> `tests/helmholtz_fmm.rs` and nd-fmm-validate's tests at the new default.

Read as:
- **the default reach**: a new `quadrature::DEFAULT_REACH = 4.5`; `Quadrature::new`,
  `MergeLevel::new`, `PlaneWaveM2l::build`, `HelmholtzOperator`'s and
  `HelmholtzFmmBuilder`'s default take it. `RHO_MAX` stays the region's corner 4√2 (a
  named reach, `--reach corner`).
- **CH6.1a's acceptance restated**: the default's kernel within 2× (nine points of every
  domain) and point-to-point chain within 3× (every domain's centre) the stated error on the
  grid points with ρ ≤ 4.5; the whole region's (the corner's) error printed, not asserted.
- **one M2L pair at the largest V offset**: the 24 offsets whose components are a
  permutation of (±3, ±3, ±2), along every direction merge and shift gives them, at the
  level's degree (`nd_fmm_exec::helmholtz::degree`, the calibrated rule, at ε = 10⁻³ and
  10⁻⁶, the ends of iprec 2's range under the rule) on every domain of iprec 2 at its
  centre: the error of the exponential M2L against RCR (Rotation) at the 27 target points,
  sources at the box's corners, centre and seeded points (CH6.1c's measure). At these
  degrees the chain also errs on the truncated multipole's tail (CH6.1c's degree study), so
  the bound is the larger of 3× the stated error and 1.1× the corner's own error on the same
  pair; both printed.
- **the verification**: CH6.1e's `verify verify-large` at the new default on locust; every
  case within 1.1× Rotation's (the report's measure, φ and ∇φ). Since the code is otherwise
  that of CH6.2e's runs, the errors should equal CH6.2e's `ρ 4.5` rows; any difference is
  reported.

## Read first

- `fmm-tables/src/helmholtz/plane_wave/quadrature.rs` (module docs "The counts",
  `Quadrature::{new, with_reach}`, `tail_count`, `RHO_MAX`), `merge.rs`
  (`MergeLevel::new`, `PlaneWaveM2l::build`).
- `fmm-tables/tests/helmholtz/plane_wave_quadrature.rs` (`check_counts`, `kernel_error`,
  `chain_error`, the ignored tests), `plane_wave_merge.rs`
  (`build_with_reach_takes_the_counts_of_the_reach`), `plane_wave_pair.rs` (`pair_error`).
- `fmm-exec/src/helmholtz/{operator.rs, fmm.rs}` (the reach settings),
  `fmm-exec/src/helmholtz/degree.rs`, `fmm-exec/tests/helmholtz_fmm.rs` (`check_plane_wave`).
- `fmm-validate/src/helmholtz_accuracy.rs`, `examples/helmholtz_accuracy.rs`
  (`reach_suffix`, `--reach corner`), `tests/helmholtz_accuracy.rs`.
- `fmm-ref/src/helmholtz/plane_wave.rs` and its tests (the generator's counts).

## Changes

- **nd-fmm-tables** `helmholtz::plane_wave::quadrature`: `DEFAULT_REACH` (4.5); `new` is
  `with_reach(…, DEFAULT_REACH)`; the module docs' "The counts" restated (the bound on
  ρ ≤ 4.5, why the FMM's chain needs about 3√2, the corner's error printed by the tests).
  `merge`: `MergeLevel::new` and `PlaneWaveM2l::build` at `DEFAULT_REACH`. `tail_count`'s
  sampling unchanged (see "Open questions").
- **nd-fmm-exec**: the operator's and the builder's default reach `DEFAULT_REACH`; their
  docs. No change to the level call.
- **nd-fmm-validate**: the example's ` ρ r` suffix off the default (not off the corner);
  `--reach corner` stays 4√2; docs.
- **nd-fmm-ref**: unchanged. The generator's tests check the kernel on the whole region's
  grid (|x|, |y| ≤ 4, the corner included; CH6.1b's acceptance), where counts for ρ ≤ 4.5
  err by up to 10⁵–10⁶× the stated error (CH6.2e §1 for FMM3D's nodes), so its counts stay
  at the corner; it is the trusted path, not the FMM's quadrature.
- **docs/CONVENTIONS.md** §3.15 "The quadrature": the sentence on the region amended, as
  signed off (CH6.2e question 1, "as recommended"): the λ sum on the region, the α counts on
  ρ ≤ 4.5 (FMM3D's tables) or the whole region (the generator). No version bump.
- The crate `CLAUDE.md` files of nd-fmm-tables, nd-fmm-exec and nd-fmm-validate.
- Report `fmm-validate/results/helmholtz-ch6-2g.md`: the sampling check, the largest-offset
  pairs, the verification, the waves.

## Tests that prove it

- nd-fmm-tables, `tests/helmholtz/plane_wave_quadrature.rs`: `check_counts`,
  `the_counts_on_sampled_domains`, the ignored `the_counts_on_the_reference_levels` and
  `the_counts_on_every_domain` on ρ ≤ `q.reach()` (the default 4.5) against the kernel
  e^{iκ|u|}/|u| (2×) and the point-to-point chain (3×); `the_counts_for_a_smaller_reach`
  with `DEFAULT_REACH` named `new` bit for bit and the corner named with at least the
  default's waves; the ignored smaller-reach test's ratios over the corner's named.
  `plane_wave_merge.rs`: `build_with_reach(…, DEFAULT_REACH)` is `build` bit for bit.
- nd-fmm-exec, new `tests/helmholtz/plane_wave_reach.rs` (no MPI): the largest-offset pairs
  against RCR as above, in debug on the fixture's domains at ε = 10⁻⁶, and the ignored
  `…_on_every_domain` on all 437 domains at 10⁻³ and 10⁻⁶ (release).
- nd-fmm-exec `tests/helmholtz_fmm.rs` `check_plane_wave`: the default reach is
  `DEFAULT_REACH`; the default named is the default bit for bit; the corner named within
  `PLANE_WAVE_RATIO` of Rotation's error with at least the default's waves per level; reach
  4 as before (fewer waves, within 1.1×); 2 threads and the one-rank FMM bit for bit; at 1,
  2 and 4 ranks.
- nd-fmm-validate `tests/helmholtz_accuracy.rs`: the row's reach the default.

## Checks

`/verify` (format, Clippy default/examples/workspace all-targets, `cargo doc`, `cargo test`
and `cargo test --workspace`, the MPI list at 2 and 4 ranks, `cargo deny` not needed: no
manifest change); `cargo test -p nd-fmm-tables --release -- --ignored`; the new ignored
nd-fmm-exec test in release.

## Machine runs

Planned: locust (1 × 72, accuracy only): CH6.1e's `verify verify-large` at the new default.
Not run: the default was not changed.

## Constraints and risks

- Constraint 1: Laplace not touched (its plane-wave M2L is §3.14's quadrature,
  `nd_fmm_tables::plane_wave`, separate). The MPI list's Laplace executables run at 2 and 4
  ranks as a check.
- Constraint 9: the reach is a constant; the counts stay a function of κ and the precision.
- Constraint 2: unchanged machinery; `check_plane_wave` checks threads and ranks.
- Risk: a point pair at the corner errs far beyond the stated error at reach 4.5 (CH6.2e
  §1); the largest-offset pair test is the guard for the FMM's chain. If it fails on a
  domain the FMM can reach, stop and report.
- Every level's waves change under `HelmholtzM2l::PlaneWave`, so its outputs change (not
  Rotation's or the other strategies'); D8's times (CH6.2f) follow this task.

## Results

- The new test `fmm-exec/tests/helmholtz/plane_wave_reach.rs`, first written with the bound
  max(3× the stated error, 1.1× the corner's error), failed on fixture domain 172 (κ ≈ 11 +
  2.4i, p = 20, 10⁻⁶: reach 4.5 1.23× the corner's). An exploration on every domain
  (against RCR and the direct sum) showed reach 4.5 adding more than 10× the stated error
  from Re κ = 7π and up to 2·10⁴× near 16π, and against the direct sum up to about 2,000×
  Rotation's error (report §2). Decision as above.
- The test then became the guard of the default counts: against the same nodes with converged
  α counts (Nₖ = ⌈|tₖ| 4√2⌉ + 2Fₖ + 40), the default may add at most 10× the stated error,
  on every domain below Im κ = 5π. Two intermediate bounds were tried and failed for reasons
  the counts do not govern: 1.1× the converged counts' error (domain 192, both errors at 1 %
  of the stated error), then that plus the stated error (domain 199, κ ≈ 23.6 + 4.7i, the
  corner adding 2.4×). The distribution over all domains is the report's §2: the corner adds at
  most 3.2× below Im κ = 5π; from 5π the stated errors fall to round-off while both chains err
  10⁻⁹–10⁻⁵ absolute, so those domains are left out (report §3).
- `tail_count`'s sampling: with 64 steps of [0, reach] the counts at 4.5 exceed the corner's on
  11–16 of 437 domains per precision, all with Im κ ≥ 17, by one or two waves in all; at 1,024
  steps (the same at 4,096) never. Kept at 64 (the bounds hold); report §1.
- No machine run: the default is unchanged, so the FMM's outputs are bit for bit `main`'s.

## Pending runs

None.

## Open questions

None open. Follow-ups for CH6.2f (roadmap): the exponential M2L's largest-offset pairs at the
level's degree err about 10× ε relative at 10⁻⁶ on the low-κ plateau (CH6.1c's truncated
tail), and both counts' chains err 10⁻⁹–10⁻⁵ absolute on domains with Im κ ≥ 5π (report §3).

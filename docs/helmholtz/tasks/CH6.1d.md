# CH6.1d Merge and shift in the FMM

Started 2026-11-03 on `tbetcke/h-ch6-1d`. Roadmap: "CH6.1d Merge and shift in the FMM"
(H6; priority list item 2). Proposal: §10 CH6.1; §5.3 (Exponential M2L), §8 R7;
CONVENTIONS §3.14 ("Merge and shift") and §3.15 ("Exponential M2L"); CH6.1c's record
(docs/helmholtz/tasks/CH6.1c.md, "Results", the degree study).

Split on this branch (the roadmap): CH6.1d is the level call with a provisional
precision rule; CH6.1e the rule by measurement on CH3.1c's cases and CH6.1's acceptance
(1.1× Rotation's errors), with its report. Together they come to about 2,000 lines in
three crates (nd-fmm-tables, nd-fmm-exec, nd-fmm-validate).

## Goal

Run the Helmholtz M2L of a level by plane waves with merge and shift, opt-in as
`HelmholtzM2l::PlaneWave`, on the levels that have a quadrature, deterministic and
rank-invariant, with the pair translation of CH6.1c as its trusted path.

## Acceptance (proposal §10, verbatim)

CH6.1: "errors within about 1.1× Rotation's, as T13's budget for Laplace". That is
CH6.1e's (the split). CH6.1d's own condition (the roadmap):

> merge and shift's shift factors the pair's to rounding on every direction, slot and
> child; the FMM with PlaneWave against the direct sum and against its pair path to
> rounding; deterministic at every thread count and on P ranks the one-rank FMM bit for
> bit; vector v the one-vector FMM bit for bit; the errors against Rotation's reported,
> not yet asserted (CH6.1e).

## Read first

- CONVENTIONS §3.14 "Merge and shift" (slots, children, accumulators, masks, the shifts
  slot → parent → child) and §3.15 "Exponential M2L" (the conversions and shifts; "the
  degree").
- `fmm-exec/src/operator_plane_wave.rs` and `operator.rs`'s `m2l` (Laplace's level call:
  outgoing waves per frame on the pool, then parents whole on one thread), which this
  restates for Helmholtz without `nd-fmm-simd`'s `WaveKernel` (Laplace's real,
  half-node layout).
- `fmm-tables/src/plane_wave/{mod.rs, geometry.rs}`: `MergeGeometry`, `Direction`,
  `slot_of`, `PlaneWaveTables::pair_direction`, the source-shift keys (read, not changed).
- `fmm-tables/src/helmholtz/plane_wave/{pair.rs, quadrature.rs}`: `PlaneWaveLevel`
  (steps, `shift_factors`, `shift_add`, `m2l`), `Quadrature::new(κ, precision)`,
  `domain_of`.
- `fmm-exec/src/helmholtz/{operator.rs, fmm.rs}` (the M2L settings, `select_tables`, the
  scratch, `FmmOperator::m2l`, the builder) and `fmm-exec/src/driver.rs`'s `Execution`
  (`chunks`, `parents`; used, not changed).

## Changes

No Laplace file changes: `crate::plane_wave` (Laplace's tables and geometry) and
`driver.rs` are only used. No convention, dependency or `nd-fmm-plan` change.

**nd-fmm-tables** (`helmholtz::plane_wave`):

- `merge.rs` (new, re-exported): per level the merge's shift factors of §3.14's geometry
  with §3.15's shifts, built in f64 and rounded to T from a `PlaneWaveLevel`: the source
  shifts, slot centre → parent centre (d = (5 − 2a)/2 per axis, rotated into the
  direction's frame), shared by key as Laplace's; the child shifts, parent → child
  (d = (2b − 1)/2), 6 × 8. `PlaneWaveM2l<T>`: the levels' `PlaneWaveLevel`s and shifts,
  `build(k̂₀, degrees, precisions)` (levels 0 and 1 none), `level(l)`, `precision(l)`,
  `bytes()`.
- The rule: `precision_for_degree(p) -> Option<Precision>` (provisional, CH6.1c's degree
  study: the coarsest precision whose chain error against the direct sum stayed within
  1.1× RCR's at every κ of the study: iprec 1 to p = 8, 2 to 12, 3 to 16, none above,
  the untested degrees 9–11 and 13–15 given the finer one), and `select(k̂₀, degrees)`,
  the rule on each level ≥ 2 whose κ_l = 2k̂_l has a domain (`domain_of`). A function of
  k, w_l and p_l only (constraint 9). CH6.1e replaces the constants.

**nd-fmm-exec** (`helmholtz`):

- `src/helmholtz/plane_wave.rs` (new, crate-private): the level call, Laplace's order:
  per vector and frame f, the up and down waves of every box of the level (the multipole
  rotated into f, its modes, the waves: `PlaneWaveLevel::waves_of`, 4W reals per box) on
  the pool by boxes; then per parent (boxes of the level in index order, split at parent
  boundaries) its computed children (with a V row), the slots of their sources, and per
  direction of f every accumulator whose mask holds a computed child summing its slots
  present in ascending order, each source's waves times its source shift
  (`shift_add`, from zero); each computed child its accumulators in ascending mask order,
  times its child shift; then its local: `sums_from_waves` and `add_local_from_sums` of
  the up then the down waves into a zeroed frame local, `add_local_from_frame` into its
  local. Per-thread scratch sized at construction; the waves buffer sized from the plan.
- `operator.rs`: `HelmholtzM2l::PlaneWave`; `PlaneWavePrecision` (`Rule`, the default,
  or `PerLevel(Vec<Option<Precision>>)`), `with_plane_wave_precision`,
  `plane_wave_precision(level)`, `plane_wave_bytes()`, `with_plane_wave_boxes`; `m2l_on`
  reports PlaneWave on those levels; the pair body (`PerPair`) translates along
  `PlaneWaveTables::pair_direction(d, child)` by `PlaneWaveLevel::m2l`; the other levels
  Rotation's level call.
- `fmm.rs`: `HelmholtzFmmBuilder::plane_wave_precision`; the waves buffer sized from the
  plan's largest level (as Laplace's `with_plane_wave_boxes`).
- Module docs ("M2L"), `fmm-exec/CLAUDE.md` and `fmm-tables/CLAUDE.md`.

## Tests that prove it

1. **Shift factorisation** (`fmm-tables/tests/helmholtz/plane_wave_merge.rs`): for every
   direction, accumulator slot and child of its mask, source shift ⊙ child shift against
   `shift_factors` of the pair's offset (f64, relative to the factor's modulus, within a
   few u); the source-shift keys shared only where the factors are equal; real and
   complex k, three levels, iprec 1–3.
2. **The rule**: `precision_for_degree` at its breakpoints; `select` gives none on levels
   0 and 1, outside the tables (Re κ > 16π) and above p = 16, and κ_l's quadrature
   otherwise; `build` with named precisions.
3. **The FMM** (`fmm-exec/tests/helmholtz_fmm.rs`, `check_plane_wave`, on the complex-k
   complex-charge scenario and the real-k real-charge one, so one and two multipole
   planes): the precision per level as the rule gives it (and named per level); against
   the direct sum within the scenario's bound, the ratio to Rotation's error printed;
   against `pair_calls(true)` (CH6.1c's pair along the merge's direction) within a
   rounding tolerance set after the first measurement (Laplace's was 2.2e-16 relative L2);
   2 threads bit for bit; on P ranks the one-rank FMM bit for bit; two vectors, each the
   one-vector FMM bit for bit; a level the rule leaves without a precision reported and
   run as Rotation.
4. **Operator** (`fmm-exec/tests/helmholtz/plane_wave.rs`): `m2l_on`, the precisions and
   the bytes reported; `PerLevel` of the wrong length or a level without a domain refused
   or kept on Rotation (as built, stated).

## Checks

`/verify` (nd-fmm-tables and nd-fmm-exec: format, Clippy default/examples/workspace,
`cargo doc`, `cargo test` and `--workspace`, the MPI list at 2 and 4 ranks, which holds
`helmholtz_fmm`); `cargo test -p nd-fmm-tables` and `-p nd-fmm-exec`.

## Machine runs

None (no timing: CH6.2; the measurement of the rule is CH6.1e's, on the M6).

## Constraints and risks

- Constraint 1: no Laplace code changes; `crate::plane_wave` and `driver.rs` only used.
- Constraint 2: every accumulator sums its slots in ascending order over geometric masks,
  so its sources and order do not depend on the rank or the thread split (Laplace's
  argument, operator module docs).
- Constraint 9: the precision per level from k, w_l and p_l only.
- Risk: the provisional rule may cost accuracy at the FMM's degrees (CH6.1c's study:
  1.3–2.3× at p = 20–24, above the rule's 16); CH6.1e measures it.
- Risk: memory. Per level the shifts are (keys + 48) × 2W reals; W = Σ Nₖ up to about
  2,000 at iprec 3; reported by `plane_wave_bytes`.
- f64 only (§3.15 "Precision and range"); f32's range (|Pₖ| to about 10³⁴ at iprec 3,
  CH6.1c's review) is not tested here.

## Pending runs

None yet.

## Open questions

None yet.

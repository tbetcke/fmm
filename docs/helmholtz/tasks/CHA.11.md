# CHA.11 The Helmholtz wave levels in one place

Started 2026-11-19 on `tbetcke/h-cha-11`. Roadmap: "CHA.11 The Helmholtz wave levels in one
place. Review 2026-11-19, L1 (its new parts; the user, 2026-11-19, on #230: on the priority list
before CH4.1). One crate-private function of `helmholtz/operator.rs` returns the levels with
waves from the settings, called by `HelmholtzFmmBuilder::build` (the V rule) and by the
operator's `select_tables`, in place of the builder's restatement (`helmholtz/fmm.rs:924-945`),
its run-time `assert!` (`:1146-1153`) and the second f64 gate. *Done when:* bit for bit
(checksums of `helmholtz_accuracy`'s cases and the MPI list); no Laplace file touched. H4 adds
the diagonal form's choice per level to the same builder, which reads it from the same place."
Proposal: none (section HA: the roadmap's *Done when* with constraint 1 of §2).

## Goal
Make the choice of the levels whose M2L is the exponential one a single crate-private function
that the builder's V rule and the operator's table selection both call, so the two cannot drift
and the run-time assert that held them together goes.

## Acceptance (roadmap *Done when*, verbatim)
Bit for bit (checksums of `helmholtz_accuracy`'s cases and the MPI list); no Laplace file
touched. With constraint 1 (the HA preamble): no behaviour or default changes.

## Read first
- `docs/reviews/2026-11-19-H1.md` §2.4, L1, checked against the tree on 2026-11-19 (ca278a6):
  the builder's restatement is `helmholtz/fmm.rs:914-945` (the comment and `waves`), the assert
  `:1146-1153`; the f64 gate is written in the builder (`:934`, `size_of::<T>()`) and in the
  operator's `HelmholtzKernel::auto_takes_plane_waves` (`operator.rs:1840`).
- `operator.rs`: `auto_plane_wave_levels` (`:522`, already shared, the Auto arm only),
  `select_tables` (`:1360`; the `precisions` match, `:1371-1389`), `plane_wave_precision`
  (`:1240`).
- The builder's k̂_0 (`Complex::new(k.re() r_0, k.im() r_0)`) is the operator's
  (`HelmholtzOperator::unbuilt`, `:897`; `tables.k_hat(0)` is it times 0.5⁰, the same bits).

## Changes
`nd-fmm-exec` only (`src/helmholtz/operator.rs`, `src/helmholtz/fmm.rs`, `fmm-exec/CLAUDE.md`).
No public API change, no table, no convention, no dependency.
- `operator.rs`: a crate-private `WaveSettings` (the M2L choice, the plane-wave precision
  setting, Auto's switch, Auto's smallest degree, the tolerance) and
  `wave_levels::<T>(&WaveSettings, k_root, base) -> Option<Vec<Option<Precision>>>`: `None`
  where the setting runs no exponential M2L, else each level's quadrature (PlaneWave's rule or
  named precisions; Auto's levels of `auto_plane_wave_levels`, made private). The f64 gate once,
  in `auto_takes_plane_waves::<T>(m2l, on)`, which the kernel's method of the same name calls.
  `select_tables` calls `wave_levels`.
- `fmm.rs`: `build` calls `wave_levels` with the agreed smallest degree; the restatement, the
  assert and its comment go; the comment on the V rule's levels says where they come from.
- `fmm-exec/CLAUDE.md`: the CH4.0d2 sentence "`build` asserts they are the operator's" restated.

## Tests that prove it
- Tests first: a unit test in `operator.rs` (`mod tests`, crate-private access),
  `wave_levels_are_the_operators`: for Rotation, PlaneWave (rule; named precisions), Auto (waves
  on and off, the smallest degree 0 and 15), tolerances none, 10⁻³, 10⁻⁶, 10⁻⁸ and 10⁻⁹, real
  and damped k (Im κ₂ above 5π), f64 and f32, the operator's `plane_wave_precision(l)` on every
  level equals `wave_levels` of the same settings, and `None` exactly where the operator holds no
  exponential M2L. It replaces the assert's run-time check with one over the settings.
- Bit for bit: `helmholtz_accuracy` (release, the M6, 1 and 2 ranks, 1 and 2 threads) on W1 at
  kD = 30 and 100, ε 10⁻³, 10⁻⁶ and 10⁻⁸, `--m2l rotation,auto,auto-pw,planewave,planewave:2`,
  `--near-v off,cost` (the V rule reads the levels with waves): every checksum of the branch
  equal to `main`'s (ca278a6), launched in the same order. Not a timing run.
- The existing operator and FMM tests unchanged: `tests/helmholtz/plane_wave.rs` (Auto's levels
  by rule, damping, f32), `tests/helmholtz_fmm.rs` (`check_near_v` under PlaneWave and Auto with
  waves, the one-rank FMM bit for bit) at 1 rank and in the MPI list at 2 and 4.

## Checks
`/verify` (fmt, Clippy default and workspace all targets, docs, the default and workspace tests,
the MPI list at 2 and 4 ranks since `helmholtz_fmm` runs the changed build).

## Machine runs
None: the change is bit for bit by construction and checked on the M6; no host path's code moves
in a timed loop (two calls at build).

## Constraints and risks
Constraint 1: no Laplace file (`driver.rs`, `pipeline.rs`, `fmm.rs` of Laplace, `nd-fmm-simd`,
`nd-fmm-kernels`, `nd-octree`, `nd-fmm-plan`) changes. Constraint 2 (determinism) and 9 (choices
from the settings and the machine class): the levels stay a function of the agreed settings.
Risk: the builder passing the function other settings than it passes the operator; both read the
same fields and the agreed degree, shown in the code and by the checksums.

## Pending runs
None.

## Open questions
None.

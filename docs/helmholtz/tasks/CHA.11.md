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
touched. With constraint 1 (the HA preamble): no behaviour or default changes (one exception,
found by the work, for sign-off: "Open questions").

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
  named precisions on the levels whose κ_l lies in the tables, as `PlaneWaveM2l` builds them;
  Auto's levels, `auto_plane_wave_levels` inlined and removed). The f64 gate once,
  in `auto_takes_plane_waves::<T>(m2l, on)`, which the kernel's method of the same name calls.
  `select_tables` calls `wave_levels`.
- `fmm.rs`: `build` calls `wave_levels` with the agreed smallest degree; the restatement, the
  assert and its comment go; the comment on the V rule's levels says where they come from.
- `fmm-exec/CLAUDE.md`: the CH4.0d2 sentence "`build` asserts they are the operator's" restated.

## Tests that prove it
- Tests first: a unit test in `operator.rs` (`mod tests`, crate-private access),
  `wave_levels_are_the_operators`: for Rotation, PlaneWave (rule; named precisions, all and
  mixed), Auto (waves on at the smallest degree 0 and 6, and off), tolerances none, 10⁻³, 10⁻⁶,
  10⁻⁸ and 10⁻⁹, real k and, under Auto, damped k (Im κ₂ above 5π), f64 and (Auto, Rotation)
  f32, and k = 400 and 20 + 200i (κ₂ outside the tables by its real and by its imaginary part),
  the operator's `plane_wave_precision(l)` on every level equals `wave_levels` of the same
  settings, and `None` exactly where the operator holds no exponential M2L. The operator calls
  the function itself, so the test guards the tables it builds against the function (the
  mutation without `domain_of` fails at k = 400); Auto's and the rule's choice keep their
  checks in `tests/helmholtz/plane_wave.rs`.
- Bit for bit: `helmholtz_accuracy` (release, the M6, 1 and 2 ranks, 1 and 2 threads) on W1 at
  kD = 30 and 100, ε 10⁻³, 10⁻⁶ and 10⁻⁸, `--m2l rotation,auto,auto-pw,planewave,planewave:2`,
  `--near-v off,cost` (the V rule reads the levels with waves), and kD = 300 under every one of
  them but `planewave:2` with `--near-v cost`, which panics on `main` (below): every checksum of
  the branch equal to `main`'s (ca278a6), launched in the same order. On the branch also that
  case, which runs. Not a timing run.
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
1. Found by the baseline: on `main`, `planewave:<q>` (`PlaneWavePrecision::PerLevel`) with the V
   rule on panics in `build` where a level's κ_l lies outside the quadrature's tables (W1 at
   kD = 300, ε any, `--m2l planewave:2 --near-v cost`: "the V rule's levels with waves [false,
   false, true, true, true, true, true] are the operator's"). The builder's restatement gave the
   named precision to every level from 2; the tables build none where `domain_of` is `None` and
   run RCR there, as `PerLevel`'s rustdoc says. `wave_levels` applies the tables' filter, so the
   builder's levels are the operator's and the case runs. Named precisions are a measurement
   setting; the V rule's cost model is the default since CH4.0d3. Every run that completed on
   `main` keeps its bits. For sign-off: accept that this case now runs (recommended), the one
   behaviour change of this HA task.

## Results (2026-11-19, the M6)
- Checksums (`helmholtz_accuracy`, release, W1, ε 10⁻³, 10⁻⁶, 10⁻⁸, 1 and 2 ranks × 1 and 2
  threads; `main` at ca278a6, the branch at c616bd5, whose later commits change the test and a
  comment only): kD = 30 and 100 under `rotation, auto, auto-pw, planewave, planewave:2` with
  `--near-v off,cost`; kD = 300 under the first four with both and `planewave:2` with the V
  rule off. All 348 rows equal line for line: the M2L and its levels, the waves, φ and the
  checksum. The kD = 300 runs of `main` that panicked
  (`--m2l planewave:2 --near-v cost`, every ε) run on the branch: level 2 (κ₂ = 75, outside
  the tables) runs RCR as the operator always did, and the V rule decides there per pair. The
  12 rows have one checksum per ε at every rank and thread count, and φ is at most the rule
  off's (2.56e-5, 6.45e-8, 1.46e-8 against 2.57e-5, 6.48e-8, 1.46e-8). Script and raw outputs:
  `~/projects/fmm-archive/m6/cha-11/` (`checksums.sh`, `main.txt`, `branch.txt`, the tables).
- `/verify` on the M6: `cargo fmt --all -- --check`; Clippy default, examples and workspace
  all-targets with `-D warnings`; `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`; the default
  (1,303 passed) and workspace (1,347 passed) tests at opt-level 1; the MPI list at 2 and 4 ranks
  at opt-level 1 (`run-mpi-tests.sh`, every executable passed; `helmholtz_fmm` 32.5 s and
  30.5 s). Not run: Laplace's fixtures and benchmarks (no Laplace file changed); `cargo deny`
  (no manifest).
- Debug time: `wave_levels_are_the_operators` 14.6–16.8 s in the lib's unit tests at
  opt-level 0 (the M6 at load about 6, the checksum runs beside it); 54 s before its operators
  were built once at small degrees (`unbuilt`, then `build_tables`).
- Self-review (a subagent on the plan, §2 and the branch): nothing blocking; the plan's stale
  lines, the undocumented panic fix (question 1), the damped level outside the tables in the
  test and a garbled comment fixed.

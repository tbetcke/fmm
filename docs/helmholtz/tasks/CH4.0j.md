# CH4.0j The exponential M2L's tables within a level in parts

Started 2026-11-16 on `tbetcke/h-ch4-0j`. Roadmap: "CH4.0j The exponential M2L's tables within
a level in parts" (H4; priority list item 6, added on 2026-11-16 by CH4.0h's sign-off, #221
question 2). Proposal: no §10 row (a follow-up of CH4.0h, report §4 and §6); §2 (constraints
1, 2, 9), §6.4.

## Goal

Build each level's exponential-M2L tables (`MergeLevel`: its quadrature, its
`PlaneWaveLevel`, its source and child shifts) in parts on the caller's threads, so that a
tree with one or two levels with waves no longer builds them nearly serially; the tables bit
for bit.

## Acceptance (roadmap *Done when*; no proposal §10 row)

The tables bit for bit; on CH3.6's cases under 1 s the build at one rank per node at most
half CH4.0h's, or the report says why not. Helmholtz only.

Read as: "bit for bit" is every plane-wave table of every level the same bits as `main`'s
(the `Debug` text, every value's shortest round trip) on any map of the parts, and every
checksum of the machine runs equal to CH4.0h's launches of the same case. "CH3.6's cases
under 1 s": the cases of CH4.0h's matrix (report §4: W1 at kD = 10, 30, 100, 500; W3 at
kD = 10, 30, 100; W4's four trees at N = 10⁵ and kD = 1, 10, 100; at 10⁻³, 10⁻⁶, 10⁻⁸) whose
build plus two evaluations at one rank per node (Kathleen 1 × 40, locust 1 × 72) took under
1 s in CH4.0h's launches; for each, the build at most half CH4.0h's build of the same case
and split. Where a case's build is not the plane-wave tables (Batched's matrices on W4 at
kD = 1 and 10, the shared steps of CH4.0k), the report says so with the build's parts.

## Read first

- CH4.0h's report `fmm-validate/results/helmholtz-ch4-0h.md` §1, §4 (the small trees: locust
  1 × 72, W1 at kD = 100 at 10⁻³ 0.066 s of plane-wave tables in a 0.084 s build), §6.
- `fmm-tables/src/helmholtz/plane_wave/merge.rs` (`MergeLevel::with_reach`,
  `PlaneWaveM2l::build_with_reach_on`), `pair.rs` (`PlaneWaveLevel::from_nodes`,
  `shift_factors`), `quadrature.rs` (`Quadrature::with_reach`, `tail_count`,
  `bessel_j_norms`); `fmm-tables/src/parts.rs`.
- `fmm-exec/src/helmholtz/operator.rs` `select_tables` (passes the pool's `PoolParts` to
  `build_with_reach_on`: no change needed there).
- `fmm-tables/CLAUDE.md`, "Builds are deterministic" (parts since CH4.0h; rayon not allowed).

Found before any machine run, by a throwaway probe on the M6 (release, one thread, one level
at κ = 1–45, p = 8–32, iprec 1–3): a level takes 4–49 ms; its quadrature 55–60 % (the α and
Fourier counts: three `tail_count`s per node, each 64 Bessel recurrences), its 152 shift
tables (104 distinct source shifts, 48 child shifts, each W complex exponentials) about 40 %,
its `PlaneWaveLevel` 1–6 % (0.06–2.8 ms; the frames' rotation blocks and the factors grow
with p). Every node's counts and every shift table is independent of the others.

## Changes

`nd-fmm-tables` (`helmholtz::plane_wave` only):
- `quadrature.rs`: `Quadrature::with_reach` split into its node values (λ, w, t, the share:
  cheap, serial) and the counts of node k (a crate-private function of the node alone), put
  together in node order; `with_reach` is the serial composition (the same calls in the same
  order per value, so the same bits).
- `merge.rs`: the merge's shift keys (the distinct (rotated offset, up) of the source shifts
  and each slot's index) are a function of the geometry alone, formed once; a crate-private
  staged build of a list of levels on a `Parts`:
  1. every level's every node's counts, one part each (flattened over the levels);
  2. every level's `PlaneWaveLevel`, one part each;
  3. every level's 104 source and 48 child shift tables, one part each (flattened).
  `PlaneWaveM2l::build_with_reach_on` runs it on its levels; `MergeLevel::with_reach` runs
  it on one level with `Serial`. No public API changes; no table, layout or value changes.
- As built: `PlaneWaveLevel` (stage 2) in parts too, each node's values and each frame's
  rotation blocks (`LevelLayout`): its per-node values are about 90 % of it and at p = 32 it
  would have been the longest part at 72 threads (M6 probe); and the shift tables kept one
  allocation each (`Shifts`, its `Debug` the flat list's text): their serial concatenation was
  half the time in parts (7 of 14 ms at κ = 45 on the M6).
- `fmm-tables/CLAUDE.md`: the parts list gains the exponential M2L's stages.

`nd-fmm-validate` / tools: `tools/helmholtz/ch4-0j-runs.sh` (the `parts` stage: the ignored
timing test below; the `matrix` stage: CH4.0h's matrix at one rank per node, twice),
`tools/helmholtz/ch4-0j-tables.py` (against CH4.0h's launches). No library change in
`nd-fmm-exec` or `nd-fmm-validate` expected.

No convention changes; no dependency changes; `nd-fmm-plan` untouched.

## Tests that prove it

- `nd-fmm-tables` `tests/helmholtz/parts.rs`:
  - `the_plane_wave_tables_on_threads_are_the_serial_tables` widened: real and complex k̂₀,
    10⁻³, 10⁻⁶ and 10⁻⁸ (iprec 1–3 and the raised degrees), the corner's reach and reach 4,
    on the reverse-order three-thread map against `Serial`, the `Debug` text equal.
  - new: every `MergeLevel` of a staged build against the level built from the public steps
    one by one (`Quadrature::with_reach`, `PlaneWaveLevel::new`, `shift_factors` of each
    slot's and child's shift), every source and child shift and the `PlaneWaveLevel`'s text
    bit for bit: an oracle independent of the staging.
  - `print_the_hashes` (ignored): the plane-wave hashes `main`'s (on the M6 before the change,
    2026-11-16: bc43acd5206e0682, 2b62dea0a7d6d982, 38d11660252af3c9, b9419e1bdc4071ca, the
    same as CH4.0h's), the RCR and matrix hashes unchanged.
  - new ignored `time_a_levels_parts` (prints, asserts no time): per (iprec, κ, p) over the
    levels' range, one level's quadrature, `PlaneWaveLevel` and shifts on one thread; run on
    the machines. As first written it also timed the build on a map that started its threads
    per call, which at 72 threads timed their start (7 ms whatever the level, locust): dropped;
    the time on many threads is the FMM's, the harness's build part "plane-wave tables".
- `tests/helmholtz/plane_wave_merge.rs` and `plane_wave_quadrature.rs` unchanged and passing
  (the counts, the shifts against the pair's, the rule).
- `nd-fmm-exec` `tests/helmholtz_fmm.rs`: unchanged; every check's 2-thread run builds the
  plane-wave tables on the pool against the 1-thread serial build bit for bit (CH4.0h).
- Machine runs: every checksum CH4.0h's.

## Checks

`/verify` (fmt, Clippy default/examples/workspace, docs, default and workspace tests at
opt-level 1, the MPI list at 2 and 4 ranks); `cargo clippy -p nd-fmm-tables --all-targets --
-D warnings`; `cargo test -p nd-fmm-tables` (debug, under a minute) and `--release --
--ignored` for the new ignored test; no manifest change, so no `cargo deny`.

## Machine runs

- Kathleen (one node, `singlenode`): `time_a_levels_parts` (release); then CH4.0h's matrix
  at 1 × 40, two launches per case (a, b). Decides the acceptance on Cascade Lake.
- locust (check `uptime` and `nvidia-smi` first): the same at 1 × 72. Decides it on Grace.
- Baseline: CH4.0h's launches (`~/projects/fmm-archive/{kathleen,locust}/logs/ch4-0h/`,
  `main` before CH4.0i; CH4.0i changed no build step); a launch of `main` (45dcc10) at the
  same splits in the same session where the CH4.0h numbers look off (another node, load).
- No timing on the M6.

## Constraints and risks

- Constraint 1 (Laplace unchanged): only `nd-fmm-tables`' `helmholtz::plane_wave`; Laplace's
  `plane_wave` (its geometry read only, as before), the drivers, `nd-fmm-plan`, `nd-octree`,
  `nd-fmm-simd` untouched. Laplace's tests and the MPI list run.
- Constraint 2 (determinism) and 9: every part a function of k̂, p, the precision, the reach,
  the node or shift index; results put together in part order; no thread count reaches a
  value.
- Risk: the restructured `Quadrature::with_reach` must keep its order of operations per
  value; the hashes and the independent oracle check it.
- Risk: the build's remaining serial steps (octree, routes, plan: CH4.0k, off the list) bound
  the gain at one rank per node; the report states it per case.

## Pending runs

None. Done: locust 2026-11-16 15:06–15:12 (`parts`, the matrix at 1 × 72, 76 launches);
Kathleen job 634435 (node-c11b-047, 16 min; `parts`, the matrix at 1 × 40, 76 launches). Raw
outputs `~/projects/fmm-archive/{kathleen,locust}/logs/ch4-0j/`.

## Results

Report `fmm-validate/results/helmholtz-ch4-0j.md`. Tables bit for bit (hashes `main`'s, every
checksum CH4.0h's); the plane-wave tables part 0.05–0.29 of CH4.0h's on locust, 0.08–0.42 on
Kathleen; the build at most half CH4.0h's on 8 of 51 case-ε pairs under 1 s (locust) and 12 of
43 (Kathleen), the others above half even without plane-wave tables (39, 28) or within 0.06 of
that floor (4, 3): the shared steps (CH4.0k) and Batched's matrices.

`/verify` on the M6: fmt, Clippy (default, examples, workspace), docs, the default and workspace
tests at opt-level 1 (1,275 and 1,389 passed), the MPI list at 2 and 4 ranks (24 of 24);
`nd-fmm-tables`' `tests/helmholtz` at opt-level 0 37.8 s at load 12 (CH4.0i 36.7 s); the
hashes in release. No Laplace path changed, so no D9 runs.

## Open questions

The report's §6: (1) the acceptance as read; (2) CH4.0k's place.

# CH6.2e The counts' corner

Started 2026-11-07 on `tbetcke/h-ch6-2e`. Roadmap: "CH6.2e The counts' corner" (H6; priority
list item 4; split on 2026-11-06 from CH6.2d). Proposal: §10 CH6.2; §2 (constraints 1, 9),
§5.3; CH6.1a's record (the counts), `spikes/pw-quadrature/SPIKE_REPORT.md` §3.2, CH6.1e's
report (the scan and the verification), CH6.2b's harness, CH6.2d's report §4.

## Goal

Find the lateral reach ρ_max that the α counts of the exponential M2L need by the FMM's own
errors, with its waves and times, and propose fewer waves for sign-off if a smaller reach
holds.

## Acceptance (proposal §10, verbatim)

CH6.2: "per-level accuracy and time beside FMM3D's on the same problems; adopted as a
strategy where it wins, else kept opt-in or dropped (D8)". This sub-task's, from the
roadmap's CH6.2e:

> whether a pair ever needs the corner at the FMM's ε, so fewer waves would do. The counts
> with a smaller reach (ρ_max = 4, FMM3D's) beside the corner's in CH6.1e's scan and
> verification at 10⁻³ and 10⁻⁶ (each level's errors against Rotation's, within 1.1×), with
> the waves per level and the exponential M2L's time per level (CH6.2b's harness). *Done
> when:* the corner's answer: the reach the counts need by the FMM's errors, with its waves
> and times, and a change to the counts proposed for sign-off if fewer waves hold. Also
> (CH6.2d, report §4): […] whether fewer than ours hold at iprec 2 (the rule's) is this
> task's.

Read as:
- **the counts with a reach ρ**: `Quadrature`'s Bessel bound with ρ in place of
  ρ_max = 4√2 (both Nₖ's terms: the count for ρ and the count for ρ − r_s plus Fₖ), the
  Fourier counts Fₖ unchanged (they bound the source's reach r_s, not the pair's);
- **CH6.1e's scan**: each level with a quadrature alone at iprec 2 (the rule's precision),
  the others Rotation, at ρ = 4 and at 4√2, on CH3.1c's matrix at 10⁻³ and 10⁻⁶ (10⁻⁸ has no
  waves under the rule); the ratio to Rotation's error at most 1.1×;
- **the verification**: the rule (iprec 2 on every level with a quadrature) at each reach of
  a ladder 4√2, 5, 4.5, 4, 3.5, 3 on the same matrix, its 10⁻⁴ and 10⁻⁵ cases (CH6.1e's
  `between`) and the two large cases; "the reach the counts need" is the smallest of the
  ladder within 1.1× Rotation's on every case, with the margin below it shown;
- **the waves per level**: Σₖ Nₖ per direction (`wave_count`) per level at each reach, and
  over every domain of iprec 2 the ratio to the corner's (FMM3D's counts sit at 1/1.31 to
  1/1.51 of the corner's where Im κ < π/2, CH6.1a);
- **the times**: CH6.2b's harness (`--levels --evaluations 2`, ABBA) on locust at 1 × 72,
  the exponential M2L per level at the corner and at the proposed reach, on CH6.2b's matrix
  and large cases at 10⁻³ and 10⁻⁶;
- **the worst case**: a pair at the corner is still possible (S4 §3.2: the extreme lateral
  offsets, 3 boxes across in both directions, the nearest along), so the kernel's and the
  point-to-point chain's error at the corner with the smaller reach is recorded beside the
  FMM's errors, for the sign-off.

## Read first

- `fmm-tables/src/helmholtz/plane_wave/quadrature.rs` (`Quadrature::new`, `tail_count`,
  `RHO_MAX`, `SOURCE_REACH`), `merge.rs` (`MergeLevel::new`, `PlaneWaveM2l::build`);
  `fmm-tables/tests/helmholtz/plane_wave_quadrature.rs` (`kernel_error`, `chain_error`).
- `fmm-exec/src/helmholtz/operator.rs` (`select_tables`, `with_plane_wave_precision`),
  `fmm.rs` (the builder's plane-wave settings).
- `fmm-validate/src/helmholtz_accuracy.rs` (`Case`, `Row`), `examples/helmholtz_accuracy.rs`
  (`--m2l planewave-scan`, `--levels`); `tools/helmholtz/ch6-1e-runs.sh`, `ch6-2b-runs.sh`,
  `ch6-2b-tables.py`.

## Changes

No Laplace file, convention, dependency, `nd-fmm-plan` or default change. The counts'
default stays ρ_max = 4√2; a smaller default is a question for sign-off (CONVENTIONS §3.15
leaves the counts to the code: "the counts its own").

1. **nd-fmm-tables** (`helmholtz::plane_wave`):
   - `Quadrature::with_reach(κ, precision, reach)` (the counts for ρ ≤ reach; `new` is
     `with_reach(…, RHO_MAX)` bit for bit) and `Quadrature::reach()`; a reach that is not
     finite or not above `SOURCE_REACH` panics;
   - `MergeLevel::with_reach`, `PlaneWaveM2l::build_with_reach` (one reach for every level;
     `new` and `build` the corner's);
   - the module docs ("The counts") and `fmm-tables/CLAUDE.md`.
2. **nd-fmm-exec**: `HelmholtzOperator::with_plane_wave_reach(reach)` and
   `plane_wave_reach()`, `HelmholtzFmmBuilder::plane_wave_reach(reach)` (default `RHO_MAX`,
   for the measurements; under PlaneWave the tables are built again); the module docs
   ("M2L") and `fmm-exec/CLAUDE.md`.
3. **nd-fmm-validate**: `Case::plane_wave_reach` (`None` the library's), `Row::plane_wave_waves`
   (per level Σₖ Nₖ, 0 without waves) and `Row::plane_wave_reach`; the example's
   `--reach r,…` (each exponential-M2L run at each reach; `corner` for 4√2),
   `planewave-scan:<1|2|3>` (the scan at one precision), the reach in the column "M2L"
   where it is not the corner's, and a last column "waves" in `--levels`' table.
4. `tools/helmholtz/ch6-2e-runs.sh` (`verify`, `verify-large`, `scan`, `times`) and
   `ch6-2e-tables.py`; the report `fmm-validate/results/helmholtz-ch6-2e.md`.

## Tests that prove it

1. **nd-fmm-tables** `tests/helmholtz/plane_wave_quadrature.rs` (debug):
   - `with_reach(RHO_MAX)` equals `new` (`PartialEq`, bit for bit) at the fixture's domains
     and every precision;
   - the counts do not grow as the reach shrinks (4√2, 5, 4, 3) on those domains, Fₖ the
     same at every reach;
   - at reach 4 the kernel within 2× the stated error on the region's grid points with
     ρ ≤ 4 (the trusted path: e^{iκ|u|}/|u|, exactly as `kernel_error`), at the fixture's
     domains' centres at every precision;
   - ignored (release): over every domain of iprec 1–3, the kernel at ρ ≤ 4 within 2×
     at reach 4, and printed per domain: the waves at reach 4 over the corner's, the
     kernel's error on the full region (the corner's) at reach 4 over the stated error.
2. **nd-fmm-tables** `tests/helmholtz/plane_wave_merge.rs`: `build_with_reach(…, RHO_MAX)`
   bit for bit `build`; at reach 4 every level's wave length at most the corner's.
3. **nd-fmm-exec** `tests/helmholtz_fmm.rs` (`check_plane_wave`, one rank in CI, 2 and 4 by
   hand): the reach `RHO_MAX` named gives the default's output bit for bit; at reach 4 the
   FMM's error within 1.1× Rotation's on its scenarios, fewer or equal waves per level,
   deterministic across thread counts; every rank or none.
4. **nd-fmm-validate** `tests/helmholtz_accuracy.rs`: `--reach`'s row (reach 4) within 1.1×
   Rotation's error on W1 at kD = 10, ε = 10⁻³ and 10⁻⁶; `plane_wave_waves` equal to the
   operator's per level and smaller than the corner's.
5. **The machine runs** (below): the scan and the verification against the sampled direct
   sum (`direct_sum` in f64, CH3.1c's oracle), the ratio to Rotation's on every case.

## Checks

`/verify` (nd-fmm-tables, nd-fmm-exec, nd-fmm-validate: format, Clippy default, examples and
workspace all-targets, `cargo doc`, `cargo test` and `--workspace`, the MPI list at 2 and 4
ranks); `cargo test -p nd-fmm-tables --release -- --ignored` for the new ignored test
(`fmm-tables/CLAUDE.md`); `shellcheck` of the script if installed.

Run on the quiet M6 (2026-11-07, from 09:38, branch at 200bd44 plus docs only), all passed:
`cargo fmt --all -- --check`; `cargo clippy -- -D warnings`, `--examples`, `--workspace
--all-targets`; `cargo doc --no-deps`; `RUST_MIN_STACK=8388608 cargo test` (1219 passed, 909 s)
and `--workspace` (1333 passed, 1236 s); `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks
(`mpi_regressions`, `mpi_exec`, `mpi_threading`, `invariance`, `helmholtz_fmm` and the seven
nd-octree examples, 1051 s); `cargo test --release -p nd-fmm-tables -- --ignored` (15 passed,
553 s). `shellcheck`: not installed, not run. Laplace (constraint 1): no Laplace code changed; the bit-identity tests of
`mpi_exec`, `mpi_threading` and `invariance` passed at 2 and 4 ranks.

## Machine runs

locust (release, BLAS threads 1, one rank × 72 threads; after `nvidia-smi`, `uptime` and a
check for other sessions' gridcl tests before the timed part):
- `verify` and `verify-large`: the rule at every reach of the ladder on CH6.1e's cases at
  10⁻³ to 10⁻⁶ (accuracy, not timed): decides the reach;
- `scan`: each level alone at iprec 2 at ρ = 4 and 4√2 on CH3.1c's matrix at 10⁻³ and 10⁻⁶:
  per-level answer;
- `times`: CH6.2b's matrix and large cases at 10⁻³ and 10⁻⁶ with `--m2l planewave
  --reach corner,<proposed>`, then reversed (ABBA), `--levels --evaluations 2`, on a quiet
  node: the waves and times per level.
No M6 timing (accuracy and tests only there). No Kathleen job.

## Constraints and risks

- Constraint 1: no Laplace code; Laplace's plane-wave M2L (`crate::plane_wave`) untouched.
- Constraint 9: the reach is a constant of the counts (a function of the geometry), the
  counts still a function of κ and the precision only.
- Risk: the corner is reachable by a pair, so a smaller reach trades a bounded worst case
  for fewer waves; the FMM's relative ℓ2 error may not see it on the reference workloads
  while a clustered distribution with charges at the extreme corners could. The report
  records the corner's error per domain for the sign-off.
- Risk: on some domains (S4 §3.2: 34, 11) FMM3D's counts miss the stated error within ρ ≤ 4
  as well; ours at reach 4 still bound ρ ≤ 4 per node, which the test checks.

## Results

locust (1 × 72, 2026-11-07 07:43–08:07, source 0a53256) and the M6 (tests, per-pair counts,
and the times at 4.5 at 1 × 6 on a quiet machine, 08:28–08:31, the user: "the m6 is quiet
now"); report `fmm-validate/results/helmholtz-ch6-2e.md`.

- **Per pair** (the ignored test, every domain): at reach 4 the kernel within 2× the stated
  error on ρ ≤ 4 (at most 1.065×, 1.188×, 1.495× at iprec 1–3) and the point-to-point chain
  within 3× (at most 0.784×, 0.716×, 1.175×), but at the corner up to 10⁴, 10⁷ and 10⁹
  times it (Re κ from 14π): a corner point pair is wrong. At 4.5 the same bounds on
  ρ ≤ 4.5, the corner up to 5·10³, 3·10⁵ and 10⁶ times. Where Im κ < π/2 reach
  4 has 0.75–0.81× the corner's waves; on a few strongly damped domains more (to 1.25×; the
  64 sample steps of ρ).
- **The verification** (60 cases at 10⁻³ to 10⁻⁶): within 1.070× Rotation's at the corner,
  5, 4.5 (each case the corner's ratio to three digits) and 4.25 (within 0.003); reach 4
  misses on three cases at 10⁻⁶ (the W4 cube at kD = 100 1.60×, W1 at kD = 500 1.62×, W3 at
  kD = 100 1.69×), 3.5 on 12, 3 on 17. The plan's ladder had no 4.25: added (a third
  launch) to place the threshold, which lies between 4 and 4.25, next to the centres'
  largest lateral offset 3√2.
- **The scan** (166 levels at iprec 2): the corner and 4.5 within 1.044×; at 4 one level
  misses, level 2 of the W4 cube at kD = 100, 10⁻⁶ (κ = 25, p = 35; 1.59×). The plan named
  4 and the corner; 4.5 added after the verification chose it.
- **Times** (ABBA, the second evaluation): on locust at 4.5 the waves per level 0.84–0.86×,
  the M2L per level 0.75–0.96× and per case 0.80–0.92×, the large cases' evaluation
  0.89–0.95×; at 4 0.77–0.80×, 0.69–0.89×, 0.73–0.89× and 0.85–0.90×. On the M6 at 4.5 (the
  matrix only) the M2L 0.79–0.95× per level and 0.80–0.91× per case. The waves per level
  were recorded at the corner, 4 and 4.5 only (the times runs); the ladder's other reaches
  by the bytes of their tables and buffers (the report's §2), which narrows the
  acceptance's "waves per level" to those three reaches.
- **Tests**: `the_counts_for_a_smaller_reach` and the reach panic (nd-fmm-tables, debug),
  `build_with_reach_takes_the_counts_of_the_reach`, the reach check of `check_plane_wave`
  (nd-fmm-exec: corner bit for bit, reach 4 within 1.1× with fewer waves, 2 threads, the
  one-rank FMM), the reach row of the harness test (W1 within 1.1×). Debug times at one rank
  on the quiet M6: `helmholtz_fmm` 211 s (203 s recorded by CH6.2d), `helmholtz_accuracy`
  46 s (42 s). Not as planned: per-node counts may exceed a larger reach's by one (the
  sampled ρ), so the test asserts the waves monotone and each node within one; test 4's
  "`plane_wave_waves` equal to the operator's" is not asserted (the row reads the operator,
  so the check would be the same call twice). After the self-review: the reach checked
  eagerly in every setter (`assert_reach`), and the ignored every-domain test asserts the
  chain within 3× at both reaches.

## Pending runs

None.

## Open questions

For sign-off (the report's §8): 1. the counts' reach 4.5 in place of 4√2 (CH6.2g);
2. CH6.2g before CH6.2f (D8).

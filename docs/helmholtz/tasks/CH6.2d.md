# CH6.2d Beside FMM3D's exponential M2L

Started 2026-11-06 on `tbetcke/h-ch6-2d`. Roadmap: "CH6.2d Beside FMM3D's exponential M2L"
(H6; priority list item 4; split on this branch, `/next-task`'s size rule, into FMM3D's per-level
times and accuracy (CH6.2d) and the counts' corner (CH6.2e), D8 renamed CH6.2f). Proposal:
§10 CH6.2; §2 (constraints 1, 2, 6, 9), §5.3, §9.2 (the H6 row); CH6.2b's report, CH6.1e's
report §2 and appendix, `spikes/pw-quadrature/SPIKE_REPORT.md` §2.

## Goal

Time FMM3D's exponential M2L per level and measure its accuracy level by level on CH6.2b's
problems, beside CH6.2b's per-level times and CH6.1e's per-level scan of ours.

## Acceptance (proposal §10, verbatim)

CH6.2: "per-level accuracy and time beside FMM3D's on the same problems; adopted as a strategy
where it wins, else kept opt-in or dropped (D8)". D8 is CH6.2f's, the corner CH6.2e's. This
sub-task's, from the roadmap's CH6.2d after the split:

> FMM3D's M2L per level from a timed build of `hfmm3d` (per-level timers in its main routine,
> a patch applied by `tools/fmm3d/build.sh`, outside the workspace), against CH6.2b's
> per-level times at matched ε; the per-level accuracy against FMM3D's (each level's waves
> alone, as CH6.1e's scan). *Done when:* per-level accuracy and time beside FMM3D's on the
> same problems.

Read as:
- **matched ε**: the same requested ε, 10⁻³ and 10⁻⁶ (CH6.2b's), as CH3.6 read it; both
  codes' end-to-end errors against the same sampled direct sum printed beside the times, so
  a difference in achieved accuracy shows;
- **the same problems**: CH6.2b's matrix (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the
  four W4 distributions at N = 10⁵ and kD = 1 and 100) and its large cases (W1 at kD = 500,
  W3 at kD = 100), the problem files of CH3.6's `helmholtz_fmm3d problem`;
- **per-level time**: FMM3D's time per level of its M2L as it runs (the multipoles to
  waves, the shifts and the waves to the children's locals), with its per-call setup of the
  level (quadrature, factors, rotation matrices) and its list-3 and list-4 work beside it,
  against CH6.2b's `m2l_seconds` per level (the exponential M2L, Rotation and the per-level
  best) on the same machine at one rank × all cores (FMM3D's OpenMP on all cores);
- **per-level accuracy**: FMM3D's scan, as CH6.1e's: FMM3D with every level on its
  point-and-shoot M2L (its rotation path), then each level that FMM3D runs on plane waves
  alone on plane waves, each run's errors against the sampled direct sum over the
  reference run's (the ratio), beside CH6.1e's ratios of ours at the same precision
  (FMM3D's iprec from ε: 1 at 10⁻³, 2 at 10⁻⁶; CH6.1e's appendix has ours at both).

## Read first

- FMM3D v2.1.0 `src/Helmholtz/hfmm3d.f` (`hfmm3dmain`: step 3, lines 1000–1830; the
  regime test at 1058, 1566, 1723, 1758, 1793), `spikes/pw-quadrature/SPIKE_REPORT.md` §2.
- `tools/fmm3d/{build.sh, driver.f90, run.sh, README.md}`;
  `fmm-validate/src/helmholtz_fmm3d.rs`, `examples/helmholtz_fmm3d.rs`.
- `fmm-validate/results/helmholtz-ch6-2b.md` (its tables, `tools/helmholtz/ch6-2b-tables.py`,
  the raw outputs in `bench-results/ch6-2b/` and the archive), `helmholtz-ch6-1e.md` (§2,
  the appendix), `helmholtz-ch3-6.md` (FMM3D's errors and times on these cases).

## Changes

No convention, dependency, table or Laplace file changes. `tools/`, `nd-fmm-validate`, and
since item 7 the Helmholtz operator of `nd-fmm-exec` (timers only).

1. **The patch** `tools/fmm3d/hfmm3d-levels.patch` (a unified diff of `hfmm3d.f`, applied by
   `build.sh` to the variants `timed` and `fastker-timed`, `default` and `fastker` with it;
   FMM3D outside the workspace as before):
   - per-level wall times (`omp_get_wtime`) in `hfmm3dmain`'s step 3: the level's setup
     (`hwts3e` to the zeroed waves), list 4's waves (`h3dlist4pw`), the multipoles to waves
     (the box loop), the parents' loop (wall), and within it the thread-seconds of the
     plane-wave part (`getpwlistall`, `hprocess*exp`) and of list 3 apart; on a
     point-and-shoot level its M2L loop and its list-3 loop; the high-frequency list-4
     loops per level; written after step 3 of each call as one line per level,
     `hfmm3d-level: ilev= width= zk2= nterms= mode= nlams= nwaves= nboxes= parents=
     setup= list4= m2e= gather= pw_ts= list3_ts= ps= ps_list3= hf_list4=` (`mode` pw,
     ps or none; `nwaves` Σ Nₖ, `nexptotp`), and the call's step times (`timeinfo`);
   - the switch `FMM3D_PW_LEVEL` (environment): unset, FMM3D unchanged; an integer l ≥ 2,
     only level l may take plane waves (where FMM3D's own test allows them), every other
     level of the plane-wave regime takes FMM3D's point-and-shoot path (its M2L
     `h3dmploc`, list 3 and list 4 as on its high-frequency levels); −1, no level on plane
     waves. Read once per call. The values computed are FMM3D's own code paths; the
     switch only chooses between them.
2. **`build.sh`**: the variants `timed` and `fastker-timed` (the tarball unpacked, the patch
   applied with `patch -p1`, its SHA-256 recorded in `build-info.txt`), `driver` relinks
   them; README.md documents the patch, the switch and the lines.
3. **`helmholtz_fmm3d errors`** (the example): FMM3D's errors at the samples (φ and ∇φ,
   §9.1's relative ℓ2) for each `--fmm3d label=PREFIX` and token, against the case's
   oracle, without running our FMM; one Markdown row per label and token, with the
   driver's seconds. The problem file checked against the case byte for byte as `compare`
   does. Single rank, no MPI collective beyond `compare`'s pattern.
4. **The run script** `tools/helmholtz/ch6-2d-runs.sh`: stages `times` (per case and
   variant, the driver at 10⁻³ and 10⁻⁶ with REPEAT 2 on all cores, twice; the
   `hfmm3d-level` lines kept in the logs) and `scan` (per case of the matrix and ε, the
   variant `timed`: `FMM3D_PW_LEVEL=-1`, unset, then each level the unset run put on
   plane waves alone; then `helmholtz_fmm3d errors` over all of them); each launch under a
   timeout, the problem files written first (`helmholtz_fmm3d problem`).
5. **The tables** `tools/helmholtz/ch6-2d-tables.py` (Python 3.9 standard library): per
   machine and case, FMM3D's levels (width, κ, nterms, mode, waves, boxes, its M2L =
   multipoles to waves + the parents' loop's plane-wave share, setup, list 3, list 4) beside
   CH6.2b's levels (p, pairs, the exponential M2L, Rotation, the per-level best, from its
   raw outputs) aligned by box width; the scan's ratios per level beside CH6.1e's; the
   end-to-end errors of both codes.
6. Docs: `fmm-validate/CLAUDE.md` (the `errors` step), the report
   `fmm-validate/results/helmholtz-ch6-2d.md`.
7. **Our level call's passes** (added 2026-11-07, the user: "yes, add the per-pass timers to
   our level call", after locust showed FMM3D's plane waves 2× faster than ours on the
   coarse levels): `HelmholtzOperator::plane_wave_times` (`PlaneWaveTimes { waves, parents }`
   per level: the waves of each chunk's sources and merge and shift, summed over chunks,
   vectors and frames; four clock reads more per chunk, vector and frame, no value changed),
   `helmholtz_accuracy::Row::plane_wave_seconds` (the slowest rank's of each), two columns of
   the example's `--levels` table (waves s, merge and shift s), read by both table scripts
   (CH6.2b's first twelve columns unchanged). The Helmholtz operator only: no Laplace path.

## Tests that prove it

- **The patch changes no value**: on each machine, `timed` with `FMM3D_PW_LEVEL` unset
  against `default` (and `fastker-timed` against `fastker`) on W1 at kD = 30 and the W4
  cube at kD = 100, both ε, FULL = 1: the full outputs byte for byte the same (`cmp`).
- **The switch**: `FMM3D_PW_LEVEL=-1` puts no level on plane waves (`mode=ps` on every
  level that was `pw`, from the lines), and its errors meet FMM3D's own at that ε within
  the scan's ratio (FMM3D's point-and-shoot path is its own; the reference row of every
  scan); a level l alone gives `mode=pw` on l only.
- **The timers**: the per-level parts sum to within the call's step-3 time
  (`timeinfo(3)`); the timed build's whole call against `default`'s (both printed). (As run:
  single calls of the check, 1.00–1.09×; `default` was not repeated, so its spread is not
  known.)
- `helmholtz_fmm3d errors`: its numbers are the module's `sample_errors` and `parse_sampled`
  (behind `read_sampled`), which `tests/helmholtz_accuracy.rs` already checks on synthetic FMM3D
  files (the direct sum in the driver's format: errors zero to round-off); the step itself
  is example code (examples are not tested in this crate) and is exercised by every scan.
  (Changed from "a unit test of the row's numbers": a test of the example's printing would
  restate the existing one.)
- The tables script: run on the outputs, its numbers checked by hand on one case per
  machine in the self-review; on CH6.2b's outputs both scripts give CH6.2b's and the first
  CH6.2d tables byte for byte.
- The passes (item 7): `fmm-exec/tests/helmholtz_fmm.rs`, `check_plane_wave_times` in
  `check_plane_wave`: one entry per level, both passes positive and within the level's M2L
  time on every level with waves and V pairs, zero elsewhere, on every rank or none (the
  existing bit-for-bit checks show no value changed); `fmm-validate/tests/helmholtz_accuracy.rs`,
  `check_levels`: the same on the harness's columns, on the exponential-M2L rows too.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`,
`cargo test` and `--workspace`, the MPI list at 2 and 4 ranks), with
`cargo clippy -p nd-fmm-validate --all-targets -- -D warnings`, `cargo test -p
nd-fmm-validate`; `tools/fmm3d/build.sh build` (VARIANTS `timed fastker-timed`) on the M6,
locust and Kathleen; `shellcheck` of the scripts if installed. No Laplace file: no D9.

## Machine runs

FMM3D's OpenMP on all cores of one node, BLAS threads 1:

- **The M6**: the build, the bit-for-bit check and a smoke `times` and `scan` on W1 at
  kD = 10 and the W4 cube at kD = 100 (accuracy and correctness, not timing).
- **Kathleen** (one `singlenode` job): build, the bit-for-bit check, `times` on the matrix
  and the large cases at 40 threads (CH6.2b's 1 × 40), `scan` on the matrix. Job and node
  in the report.
- **locust**: build, the bit-for-bit check, `times` on the matrix and the large cases at 72
  threads (CH6.2b's 1 × 72); load checked first and stated.

No timing on the M6 (CH6.2b's M6 rows stand without FMM3D beside them unless the user asks).
Decides: nothing by itself; the per-level comparison is CH6.2f's input for D8.

## Constraints and risks

- Constraint 1: no Laplace file; the patch lives outside the workspace's crates.
- Constraint 6: every error against the f64 direct sum at the sampled targets.
- Risk: FMM3D's point-and-shoot path on low-frequency levels (κ_l down to 0.008) may not meet
  ε; then the reference row is FMM3D at a tighter eps on all plane waves (its iprec 3) and
  the report says so.
- Risk: the levels do not align (FMM3D's root box is the points' bounding cube, ours [0, 1]³
  for these workloads; its leaves hold up to `ndiv` points): aligned by width, each level's
  κ printed for both.
- Risk: list 4 (FMM3D's X list on plane waves) is shifted inside `hprocess*exp` with the
  M2L's waves and cannot be separated without changing FMM3D's loops; its share is
  bounded by `h3dlist4pw`'s time and stated.
- Risk: per-box clock reads in the parents' loop add overhead; checked by the whole call's
  time against `default`'s.

## Pending runs

- Done: Kathleen job 630125 (node-c11a-109, 2026-11-07; the four FMM3D builds, `check`,
  `times`, `scan`; 147 launches, exit 0) and job 630164 (node-c11a-111; ours with the passes,
  `ch6-2b-runs.sh matrix large` at 1 × 40; 16 launches, exit 0). The report's §2–§4, §6.
- Done, records only: locust 2026-11-07 00:07–00:22 (FMM3D `check` and `times`, source
  3f683b5) and 00:43–01:10 (FMM3D `times` again and ours with the passes at 1 × 72, source
  9ff114a), beside another session's `gridcl_geometry` test on one core (from 00:12).
- Done, locust 01:24–01:50 (the user, 2026-11-07: "Wait for gridcl"; FMM3D `times` 60
  launches, ours 16, every one exit 0, no gridcl process at either end): the report's §2a,
  §6 and the locust appendix. The chain as it was queued:
  `/data/ucahtbe/logs/ch6-2d-chain2.sh` (pid 1933945) waits until no gridcl test has run
  for 10 minutes and the load is below 2, then runs `THREADS=72 tools/helmholtz/ch6-2d-runs.sh
  /data/ucahtbe/logs/ch6-2d-r3 times` and `THREADS=72 SPLITS=1x72 tools/helmholtz/ch6-2b-runs.sh
  /data/ucahtbe/logs/ch6-2d-ours-r3 matrix large`; log `/data/ucahtbe/logs/ch6-2d-chain2.log`.
  Decides locust's rows (the roadmap's *Done when* names Kathleen and locust); then the
  tables with `locust=bench-results/ch6-2d/locust-r3,bench-results/ch6-2d/locust-ours-r3`,
  the report's §2–§3 and §6, the roadmap tick.

## Open questions

# Helmholtz CH4.0d1b: the empty-box rule's plan built once

Task: `docs/helmholtz/tasks/CH4.0d1b.md`. Roadmap: H4, CH4.0d1b (added at CH4.0d1, #216's
question 2, signed off 2026-11-13). The design the user approved on 2026-11-13 ("Approve
(Recommended)"). Proposal §2 (constraints 1, 2, 9). Builds on
`fmm-validate/results/helmholtz-ch4-0d1.md`.

## 0. Summary

- **The change.** CH4.0d1 built the plan twice: `Plan::new` for the routes and the counts'
  exchange lists, then `Plan::with_rules`. The routes need only the box index's local leaves,
  and every held ghost below another rank's coarse blocks is a local box of its owner, so
  the counts need no rows either. Now the Helmholtz builder builds the octree's index
  (`BoxIndex::of_octree`), routes the points on it (`Redistribution::with_index`), counts
  every held box (`LeafCounts::exchange_with_boxes` on the index: every held ghost that is
  not a top-tree key, and every held ghost leaf, a count each) and builds the plan once on
  the same index (`Plan::with_rules_on`), with the rule or without.
- **The build with the rule is no longer dearer** (one rank × 40 threads, Kathleen; the build
  with the rule over the rule off's, the mean of both orders):
  - W3 at N = 4·10⁶, kD = 100: **1.00–1.03** (CH4.0d1: 1.55–1.57);
  - the under-resolved W3 (N = 10⁶) at kD = 500: **0.80–0.82** (CH4.0d1: 1.49–1.50);
  - W1 at kD = 500: **0.87–0.91** (CH4.0d1: 1.24–1.33);
  - on 4 ranks × 10 threads, the under-resolved W3 at kD = 500: 0.82 (CH4.0d1: 1.45).

  The *Done when*'s bound, 1.1, holds on both named cases. Where boxes are empty the plan with
  the rule is cheaper to build than the plan without it (fewer entries to translate and
  store). The build with the rule off is CH4.0d1's (15.3 s against 15.3 s, 92.1 s against
  92.2 s), so the ratio did not improve by a slower denominator.
- **The output is CH4.0d1's.** Every checksum of the 24 rows (rule on and off, both orders,
  Rotation and the default) equals CH4.0d1's on the same case (jobs 633596 and 633566), and
  the plan is CH4.0d1's bit for bit (`tests/mpi_regressions.rs`, at 1, 2, 4 and 8 ranks: the
  plan of one build equals `Plan::with_rules` on the true counts of every box, with and
  without the near rule; with no rule it is `Plan::new`'s).
- **Laplace is unchanged**: `Plan::new`, `Redistribution::new` and `Fmm`'s routes run the same
  code on the same index (every Laplace test passes); `plan_build_cost` on Kathleen within
  `main`'s spread on all three trees (§3).

## 1. Method

Kathleen, one exclusive node per job (`singlenode`; 2 × 20 Cascade Lake cores, Xeon Gold 6248),
release builds, rustc 1.99.0, AVX-512 P2P, BLAS threads 1, `helmholtz_accuracy` through
`tools/helmholtz/ch4-0d1b-runs.sh` (CH4.0d1's cases and options: Rotation and the default
`auto-pw`, `--empty-boxes on,off` and `off,on`), tables by
`tools/helmholtz/ch4-0d1b-tables.py` (the means of both orders, against CH4.0d1's launches of
the same cases). Raw outputs: `~/projects/fmm-archive/kathleen/logs/ch4-0d1b/`.

| job | node | tree | stages |
| --- | --- | --- | --- |
| 633734 | node-c11a-036 | 98014e8 | `large` (W1 at kD = 500, W3 at N = 4·10⁶ and kD = 100; 10⁻⁶, 10⁻⁸; two evaluations) |
| 633735 | node-c11a-140 | 98014e8; `main` 34c5922 | `w3` (the under-resolved W3 at kD = 500, 10⁻⁶, one evaluation), `traffic` (4 ranks × 10 threads), `plan_build_cost` (`main` and the branch alternately, three times) |

98014e8 differs from the branch's last commit in tests and documentation only.

## 2. The build (one rank × 40 threads unless named)

| case | ε | M2L | build on s | build off s | on/off | CH4.0d1 on s | CH4.0d1 off s | CH4.0d1 on/off | evaluate on s | checksums |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W1, kD = 500 | 1e-6 | Rotation | 9.36 | 10.52 | 0.89 | 14.18 | 10.68 | 1.33 | 9.60 | equal |
| W1, kD = 500 | 1e-6 | default | 9.60 | 11.05 | 0.87 | 14.55 | 11.00 | 1.32 | 9.82 | equal |
| W1, kD = 500 | 1e-8 | Rotation | 9.69 | 11.05 | 0.88 | 14.62 | 11.27 | 1.30 | 18.61 | equal |
| W1, kD = 500 | 1e-8 | default | 13.67 | 15.07 | 0.91 | 18.98 | 15.33 | 1.24 | 20.25 | equal |
| W3 (N = 4031442), kD = 100 | 1e-6 | Rotation | 15.70 | 15.31 | 1.03 | 23.94 | 15.30 | 1.56 | 39.17 | equal |
| W3 (N = 4031442), kD = 100 | 1e-6 | default | 16.09 | 16.16 | 1.00 | 24.45 | 15.57 | 1.57 | 11.06 | equal |
| W3 (N = 4031442), kD = 100 | 1e-8 | Rotation | 15.51 | 15.34 | 1.01 | 23.82 | 15.16 | 1.57 | 127.61 | equal |
| W3 (N = 4031442), kD = 100 | 1e-8 | default | 16.30 | 16.12 | 1.01 | 24.50 | 15.81 | 1.55 | 37.83 | equal |
| W3 (N = 1000000), kD = 500 | 1e-6 | Rotation | 74.10 | 92.13 | 0.80 | 137.66 | 92.17 | 1.49 | 107.04 | equal |
| W3 (N = 1000000), kD = 500 | 1e-6 | default | 76.19 | 92.46 | 0.82 | 139.02 | 92.62 | 1.50 | 63.55 | equal |
| W3 (N = 1000000), kD = 500, 4 × 10 | 1e-6 | default | 21.09 | 25.82 | 0.82 | 37.58 | 25.94 | 1.45 | 67.05 | equal |

"checksums": every checksum of the case (rule on and off, every launch) is CH4.0d1's. CH4.0d1's
columns are its jobs 633596 (`large`), 633566 (`w3`, one order) and 633564 (`traffic`) on
other nodes; the evaluations are not compared (other nodes; the evaluation code is unchanged).
CH4.0d1's report gave the build with the rule plus one evaluation up to 1.33× the rule off's
on uniform trees; with the build at 1.00–1.03 that loss is gone, and the gains of CH4.0d1's
§0 on sparse trees grow by the build's saving.

The 4-rank run reads 95,284 multipoles from other ranks with the rule against 175,800
without, as in CH4.0d1; the counts' exchange now requests every held ghost box, one count each,
and its build is still 0.82 of the rule off's.

## 3. Laplace: the plan's build time (job 633735)

`plan_build_cost` (one rank, the best of several builds per tree, ms), `main` (34c5922) and the
branch alternately on one node:

| tree | `main` runs 1–3 | branch runs 1–3 |
| --- | --- | --- |
| graded corner blob | 7.15, 7.00, 7.09 | 7.11, 7.05, 7.15 |
| 10⁵ uniform, level 4 | 80.33, 80.65, 79.34 | 81.41, 79.63, 80.29 |
| 10⁵ uniform, 32 per leaf | 94.91, 94.35, 95.63 | 95.83, 95.11, 94.23 |

Medians 7.09, 80.33 and 94.91 on `main`, 7.11, 80.29 and 95.11 on the branch: within each
other's spread. The heap columns are the same. No D9 evaluation runs: no evaluation code
changed (`Plan::new` is the same code split into the index and the rows; `Redistribution::new`
calls `with_index` on the plan's index; `Fmm` passes its plan's index to the routes).

## 4. Questions for sign-off

None: the design was approved before the code, and the *Done when* is met.

## 5. Reproduce

```sh
tools/kathleen/sync.sh
tools/kathleen/submit.sh --time=02:30:00 -- run.sbatch \
    'cargo build --release -p nd-fmm-validate --example helmholtz_accuracy' \
    'tools/helmholtz/ch4-0d1b-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch4-0d1b/large large'
# likewise `w3 traffic` into .../w3; then, locally:
python3 tools/helmholtz/ch4-0d1b-tables.py ~/projects/fmm-archive/kathleen/logs/ch4-0d1b/large \
    ~/projects/fmm-archive/kathleen/logs/ch4-0d1/large2
```

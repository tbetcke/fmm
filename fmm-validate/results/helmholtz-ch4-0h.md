# Helmholtz CH4.0h: the Helmholtz build by part, and parallel

Task: `docs/helmholtz/tasks/CH4.0h.md`. Roadmap: H4, CH4.0h (priority list item 6), added on
2026-11-16 by CH4.0e's sign-off (`fmm-validate/results/helmholtz-ch4-0e.md` §3.2, §7 question
1). No proposal §10 row; the acceptance is the roadmap's *Done when*: W1 at kD = 500's build on
the node at most half its one-core time, the output bit for bit, Laplace unchanged.

## 0. Summary

- **Where the build went** (§2, locust, W1 at kD = 500, before this task): the operator's RCR
  tables were 1.5, 1.9 and 4.2 s of a build of 4.2–5.9 s at 10⁻³, 10⁻⁶ and 10⁻⁸, the same on
  one core and on 72 threads: every table was built serially, and at 10⁻⁸ twice (once at the
  degrees given, again where the exponential M2L's rule raises a level). The rest is the plan
  with its rules (1.5 s; 0.6 s at 10⁻⁸, where the leaves hold more points), the routes (0.6 s)
  and the octree (0.3 s), the shared steps that Laplace's builder runs too.
- **What changed** (§1): the operator chooses every level's M2L first and builds its tables
  once, with the settings it is given (`HelmholtzOperator::unbuilt`, `build_tables`); the RCR
  tables store their M2L blocks only for the levels whose M2L is RCR's; Dense, Batched and
  Classes take f64 tables of their own levels' M2L alone, not a second set of every level's;
  and every table build and the scratch sets run in independent parts on the FMM's pool
  (`nd_fmm_tables::parts`), each part a function of its inputs, put together in order, so the
  tables are the serial build's bit for bit.
- **The acceptance** (§3): at the node's best split the build is 0.15–0.19 of its one-core
  time on locust (8 × 9 and 4 × 18) and 0.22–0.23 on Kathleen (4 × 10), at every ε; at one
  rank per node 0.46–0.66 on locust and 0.44–0.50 on Kathleen (10⁻³ on Kathleen 0.505), where
  the shared steps stay serial (2.4 s on locust at 10⁻³). Against CH4.0e's builds of the same
  split: 0.15–0.33 at the best split, 0.28–0.64 at one rank per node. Every checksum of every
  run, 1 × 1 included, is CH4.0e's: the output is bit for bit.
- **CH3.6's matrix** (§4): the build is 0.34–0.59 of CH4.0e's (geometric means by ε and
  split; 0.14–0.98 per case) on both machines, all 228 checksums CH4.0e's. Where a case is
  small, what remains is mostly the exponential M2L's tables, one part per level (W1 at
  kD = 100 on locust: 0.066 of a 0.084 s build); where it is large in N, the shared steps
  (W3 at kD = 100, 4·10⁶ points: the routes 1.7–2.0 s and the plan 1.0–1.5 s of 3.8–4.3 s at
  one rank per node).
- **Memory** (§5): per build the peak resident size is lower than `main`'s (locust, 72 threads,
  W1 at kD = 500: 5.2, 7.6 and 7.7 GB against 5.4, 7.8 and 8.1 GB); three builds in one process
  peak higher (10.1 against 8.2 GB), the freed table work held in the worker threads' arenas.
- **Follow-ups** (§6): CH4.0j (the exponential M2L's tables within a level in parts) and CH4.0k
  (the shared steps on the node: the plan with its rules, the routes, the octree; Laplace's
  paths); the PR's questions.

## 1. What changed

`nd-fmm-tables` (Helmholtz modules and a new `parts`; Laplace's tables untouched):
- `parts::{Parts, Serial}`: a builder cuts its work into parts 0..n and asks the map for
  `[f(0), …, f(n − 1)]`; `Serial` computes them in turn. The crate runs no thread.
- `HelmholtzTables::build_by(k̂₀, degrees, &TableChoice, &impl Parts)`: per level pair the
  M2M table, the L2L table and the two octant block sets; per level its 15 M2L coaxial tables;
  per stored level its 49 M2L block sets, each a part. `TableChoice` names the M2L degrees,
  the block budget, whether to build the translations, the levels with M2L tables and the
  levels whose blocks may be stored (the others spend none of the budget);
  `set_m2l_blocks` changes the stored set of built tables. `build` and
  `build_with_m2l_degrees` are `build_by` with `TableChoice::new` and `Serial`.
- `DenseM2l`, `BatchedM2l`, `ClassM2l::build_on`: their f64 tables hold the M2L of their own
  levels alone (before: a whole `HelmholtzTables<f64>` of every level, the coarse levels at
  p = 131 included), and the matrices are built eight offsets per part.
- `PlaneWaveM2l::build_with_reach_on`: each level's tables a part.

`nd-fmm-exec` (`helmholtz` only):
- `HelmholtzBuildTimings`, `HelmholtzFmm::build_timings`, `TableTimings`,
  `HelmholtzOperator::table_timings`: the build by step and the tables by part (clock reads).
- `HelmholtzOperator::unbuilt` and `build_tables`: the setters record and build nothing until
  `build_tables`; `HelmholtzFmmBuilder` names every setting and the pool first.
  `with_block_budget` is `unbuilt` and the build, as before.
- `select_tables`: each level's M2L first (the exponential M2L's tables, then Dense, Classes
  or Batched), then the RCR tables, built again only where their degrees change, with the M2L
  blocks of the levels whose M2L is RCR's alone; all on the pool (`PoolParts`, rayon's
  indexed collect, which keeps the parts' order); the scratch sets, one per thread, made on
  the pool after the old ones are dropped.

`nd-fmm-validate`: `Row::build_parts`, `BUILD_PARTS` and the table "Build" of
`helmholtz_accuracy`; `tools/helmholtz/ch4-0h-runs.sh` (`build`, `matrix`) and
`ch4-0h-tables.py`.

The bits: every part is computed by the serial build's functions from the same inputs, and
the stored and per-call M2L blocks agree by construction (`HelmholtzTables`' "Blocks and the
budget"), so neither the threads nor the block levels move a value. Checked by
`nd-fmm-tables`' `tests/helmholtz/parts.rs` (every build on a map that runs the parts in
reverse order on three threads against the serial build, through every application), by the
hashes of every application, matrix and plane-wave table, the same on `main` and on the branch
(the M6, release: RCR d99e3221b99a61e3, 71223dfc2b0eba4f, 1520f647ee9b88bd; matrices
e8f14c144df269a6, 9218603869fc1abd; plane waves bc43acd5206e0682, 2b62dea0a7d6d982,
38d11660252af3c9, b9419e1bdc4071ca), by `nd-fmm-exec`'s `tests/helmholtz/build.rs` (the unbuilt
operator against the setters on a built one, every M2L, every pair body bit for bit), by every
two-thread check of `tests/helmholtz_fmm.rs` (the tables on the pool against one thread's serial
build), and by the machine runs' checksums (§3, §4).

## 2. The build by part, before

locust (GH200, 72 Neoverse V2 cores), W1 at kD = 500 (N = 1,989,437) at the defaults, one rank,
`main` with this task's timings alone (`13f2089`, the profile `profile-main-*`) and this
branch, seconds (the operator includes its tables):

| ε | threads | | build | octree | routes | plan (with counts, shapes) | operator | RCR tables | plane-wave tables | scratch |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 1 | before | 4.23 | 0.33 | 0.62 | 1.50 | 1.66 | 1.51 | 0.14 | 0.00 |
| 1e-3 | 72 | before | 4.31 | 0.32 | 0.61 | 1.50 | 1.76 | 1.51 | 0.14 | 0.11 |
| 1e-3 | 72 | after | 2.74 | 0.33 | 0.62 | 1.51 | 0.18 | 0.09 | 0.08 | 0.01 |
| 1e-6 | 1 | before | 4.70 | 0.35 | 0.64 | 1.52 | 2.06 | 1.90 | 0.16 | 0.00 |
| 1e-6 | 72 | before | 4.70 | 0.32 | 0.62 | 1.48 | 2.16 | 1.87 | 0.16 | 0.13 |
| 1e-6 | 72 | after | 2.74 | 0.32 | 0.60 | 1.51 | 0.19 | 0.10 | 0.08 | 0.02 |
| 1e-8 | 1 | before | 5.93 | 0.21 | 0.61 | 0.60 | 4.41 | 4.22 | 0.18 | 0.01 |
| 1e-8 | 72 | before | 5.89 | 0.20 | 0.57 | 0.59 | 4.47 | 4.15 | 0.19 | 0.13 |
| 1e-8 | 72 | after | 1.67 | 0.20 | 0.59 | 0.59 | 0.23 | 0.11 | 0.10 | 0.02 |

("after" are the means of `build-1x72-a` and `-b` of §3; "before" one launch each.) The build
did not use the threads at all; the operator's tables were 39–76 % of it, and at 10⁻⁸ the RCR
tables were built twice. After the change the RCR tables take 0.09–0.11 s on 72 threads (one
coaxial M2L table at p = 131 is the longest part). On one core the RCR tables are 1.49, 1.85
and 2.05 s (the second build gone at 10⁻⁸). No level of W1 at kD = 500 runs Batched (its levels
are Rotation or waves), so its matrices cost nothing here; on W4's trees they do (§4).

## 3. The acceptance: W1 at kD = 500 by split

Each split twice, in order and reversed (`build` stage, one evaluation), the one-core launch
(1 × 1) first and last; "÷ one core" over the mean one-core build of this branch; "CH4.0e" the
mean build of CH4.0e's launches of the split (its one-core run, `t1`, is at 10⁻⁶ only); the
parts the means, seconds; "checksum" whether every launch's equals CH4.0e's.

Kathleen (job 634392, node-c11a-065, 2 × 20 Cascade Lake cores):

| ε | split | build s (a, b) | ÷ one core | CH4.0e build s | octree s | routes s | counts s | plan s | operator s | RCR tables s | scratch s | checksum |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-3 | 1x1 | 5.07 (5.09, 5.04) | 1.00 | – | 0.321 | 0.704 | 0.042 | 0.998 | 2.862 | 2.589 | 0.011 | CH4.0e |
| 1e-3 | 1x40 | 2.56 (2.58, 2.54) | 0.50 | 5.06 | 0.321 | 0.700 | 0.015 | 0.997 | 0.366 | 0.201 | 0.009 | CH4.0e |
| 1e-3 | 2x20 | 1.56 (1.56, 1.56) | 0.31 | 3.99 | 0.188 | 0.398 | 0.010 | 0.485 | 0.393 | 0.224 | 0.013 | CH4.0e |
| 1e-3 | 4x10 | 1.18 (1.18, 1.19) | 0.23 | 3.55 | 0.099 | 0.197 | 0.009 | 0.244 | 0.589 | 0.415 | 0.013 | CH4.0e |
| 1e-3 | 8x5 | 1.65 (1.66, 1.64) | 0.32 | 3.35 | 0.057 | 0.098 | 0.005 | 0.128 | 1.323 | 1.143 | 0.010 | CH4.0e |
| 1e-6 | 1x1 | 5.68 (5.67, 5.69) | 1.00 | 5.54 | 0.317 | 0.712 | 0.013 | 0.998 | 3.501 | 3.231 | 0.013 | CH4.0e |
| 1e-6 | 1x40 | 2.66 (2.69, 2.64) | 0.47 | 5.69 | 0.320 | 0.693 | 0.013 | 0.996 | 0.492 | 0.299 | 0.039 | CH4.0e |
| 1e-6 | 2x20 | 1.62 (1.61, 1.63) | 0.28 | 4.58 | 0.190 | 0.401 | 0.009 | 0.488 | 0.451 | 0.275 | 0.024 | CH4.0e |
| 1e-6 | 4x10 | 1.27 (1.27, 1.26) | 0.22 | 4.08 | 0.099 | 0.197 | 0.007 | 0.253 | 0.673 | 0.498 | 0.021 | CH4.0e |
| 1e-6 | 8x5 | 1.91 (1.91, 1.91) | 0.34 | 3.98 | 0.053 | 0.092 | 0.004 | 0.128 | 1.602 | 1.425 | 0.020 | CH4.0e |
| 1e-8 | 1x1 | 6.15 (6.16, 6.14) | 1.00 | – | 0.321 | 0.706 | 0.013 | 1.017 | 3.970 | 3.610 | 0.014 | CH4.0e |
| 1e-8 | 1x40 | 2.73 (2.72, 2.73) | 0.44 | 9.78 | 0.320 | 0.688 | 0.013 | 1.014 | 0.548 | 0.315 | 0.040 | CH4.0e |
| 1e-8 | 2x20 | 1.69 (1.69, 1.70) | 0.28 | 8.59 | 0.189 | 0.397 | 0.009 | 0.497 | 0.523 | 0.309 | 0.027 | CH4.0e |
| 1e-8 | 4x10 | 1.38 (1.39, 1.36) | 0.22 | 8.36 | 0.099 | 0.193 | 0.005 | 0.257 | 0.780 | 0.561 | 0.022 | CH4.0e |
| 1e-8 | 8x5 | 2.13 (2.12, 2.13) | 0.35 | 8.29 | 0.052 | 0.092 | 0.005 | 0.131 | 1.812 | 1.585 | 0.022 | CH4.0e |

locust (2026-11-16, 07:16–07:39, the load 0.7 before the first launch, from this task's own
earlier runs; no other user's process at the check before them):

| ε | split | build s (a, b) | ÷ one core | CH4.0e build s | octree s | routes s | counts s | plan s | operator s | RCR tables s | scratch s | checksum |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-3 | 1x1 | 4.16 (4.15, 4.17) | 1.00 | – | 0.322 | 0.585 | 0.025 | 1.471 | 1.640 | 1.492 | 0.004 | CH4.0e |
| 1e-3 | 1x72 | 2.74 (2.76, 2.73) | 0.66 | 4.31 | 0.325 | 0.616 | 0.025 | 1.481 | 0.178 | 0.089 | 0.005 | CH4.0e |
| 1e-3 | 2x36 | 1.57 (1.57, 1.56) | 0.38 | 3.08 | 0.186 | 0.422 | 0.013 | 0.710 | 0.178 | 0.091 | 0.003 | CH4.0e |
| 1e-3 | 4x18 | 0.89 (0.90, 0.88) | 0.21 | 2.34 | 0.093 | 0.194 | 0.007 | 0.342 | 0.221 | 0.137 | 0.003 | CH4.0e |
| 1e-3 | 8x9 | 0.62 (0.62, 0.62) | 0.15 | 1.99 | 0.050 | 0.083 | 0.004 | 0.169 | 0.294 | 0.210 | 0.004 | CH4.0e |
| 1e-6 | 1x1 | 4.53 (4.53, 4.54) | 1.00 | 4.54 | 0.327 | 0.587 | 0.024 | 1.476 | 2.006 | 1.849 | 0.002 | CH4.0e |
| 1e-6 | 1x72 | 2.74 (2.74, 2.73) | 0.60 | 4.71 | 0.317 | 0.603 | 0.025 | 1.485 | 0.194 | 0.102 | 0.016 | CH4.0e |
| 1e-6 | 2x36 | 1.59 (1.59, 1.59) | 0.35 | 3.45 | 0.180 | 0.421 | 0.013 | 0.716 | 0.202 | 0.114 | 0.014 | CH4.0e |
| 1e-6 | 4x18 | 0.91 (0.92, 0.91) | 0.20 | 2.69 | 0.091 | 0.197 | 0.007 | 0.344 | 0.242 | 0.151 | 0.012 | CH4.0e |
| 1e-6 | 8x9 | 0.68 (0.68, 0.68) | 0.15 | 2.35 | 0.045 | 0.080 | 0.004 | 0.170 | 0.362 | 0.273 | 0.009 | CH4.0e |
| 1e-8 | 1x1 | 3.66 (3.66, 3.65) | 1.00 | – | 0.200 | 0.553 | 0.008 | 0.583 | 2.239 | 2.048 | 0.004 | CH4.0e |
| 1e-8 | 1x72 | 1.67 (1.69, 1.64) | 0.46 | 5.83 | 0.198 | 0.585 | 0.009 | 0.578 | 0.226 | 0.113 | 0.017 | CH4.0e |
| 1e-8 | 2x36 | 1.08 (1.08, 1.08) | 0.30 | 5.18 | 0.114 | 0.404 | 0.005 | 0.282 | 0.234 | 0.117 | 0.020 | CH4.0e |
| 1e-8 | 4x18 | 0.69 (0.69, 0.69) | 0.19 | 4.72 | 0.057 | 0.188 | 0.004 | 0.136 | 0.284 | 0.175 | 0.009 | CH4.0e |
| 1e-8 | 8x9 | 0.82 (0.59, 1.05) | 0.22 | 4.60 | 0.028 | 0.075 | 0.003 | 0.067 | 0.619 | 0.513 | 0.011 | CH4.0e |

Reading:
- **At the node's best split** (CH4.0e's measure, each code at its best split) the build is
  0.22–0.23 of one core on Kathleen (4 × 10) and 0.15–0.19 on locust (8 × 9 at 10⁻³ and 10⁻⁶,
  4 × 18 at 10⁻⁸): the acceptance holds, at every ε, on both machines.
- **At one rank per node** (1 × 40, 1 × 72) it is 0.44–0.50 on Kathleen (10⁻³: 2.56 over 5.07,
  0.505) and 0.46–0.66 on locust. There the shared steps, the plan with its rules, the routes
  and the octree, stay serial: 2.0, 2.0 and 2.0 s of 2.56–2.73 s on Kathleen, 2.4, 2.4 and 1.4 s
  of 1.67–2.74 s on locust. Several ranks split them by points (their sum falls with the ranks:
  locust 0.30 s at 8 × 9 at 10⁻³ against 2.4 s at 1 × 72), and that is the whole difference.
- **Against CH4.0e** at the same split: 0.15–0.33 at the best split, 0.28–0.64 at one rank.
  W1 at kD = 500's build now takes 0.62–0.69 s on locust and 1.18–1.38 s on Kathleen at the
  best split, against 1.99–4.60 and 3.35–8.29 s at CH4.0e's best splits.
- **With many ranks per node the tables are each rank's own**: at 8 × 5 on Kathleen the RCR
  tables take 1.1–1.6 s per rank on 5 threads (2.6–3.6 s on one core), so 8 × 5 builds slower
  than 4 × 10; with 9 threads on locust 0.21–0.51 s. One copy of the tables per node is CH8.0's.
- 10⁻⁸ on locust at 8 × 9 has one slow launch (b: 1.05 s against a: 0.59 s, the RCR tables
  0.79 against 0.24 s); not repeated.

## 4. CH3.6's matrix

The 19 cases of CH4.0e (W1 at kD = 10, 30, 100, 500; W3 at kD = 10, 30, 100; W4's cube,
sphere, Plummer sphere and clusters at N = 10⁵ and kD = 1, 10, 100) at 10⁻³, 10⁻⁶ and 10⁻⁸, two
evaluations as CH4.0e, one launch per case at one rank per node and at CH4.0e's third split;
the build over CH4.0e's mean build of the same split (CH4.0e printed builds to 0.01 s: no
ratio where its build was below 0.05 s, so 16–18 cases per row):

| machine | split | ε | cases | build ÷ CH4.0e: geometric mean | min | max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Kathleen | 1x40 | 1e-3 | 18 | 0.51 | 0.29 | 0.89 |
| Kathleen | 1x40 | 1e-6 | 18 | 0.59 | 0.47 | 0.94 |
| Kathleen | 1x40 | 1e-8 | 18 | 0.43 | 0.28 | 0.93 |
| Kathleen | 4x10 | 1e-3 | 18 | 0.47 | 0.33 | 0.98 |
| Kathleen | 4x10 | 1e-6 | 18 | 0.51 | 0.31 | 0.95 |
| Kathleen | 4x10 | 1e-8 | 18 | 0.37 | 0.17 | 0.78 |
| locust | 1x72 | 1e-3 | 17 | 0.59 | 0.42 | 0.96 |
| locust | 1x72 | 1e-6 | 17 | 0.58 | 0.46 | 0.97 |
| locust | 1x72 | 1e-8 | 18 | 0.38 | 0.27 | 0.91 |
| locust | 4x18 | 1e-3 | 16 | 0.45 | 0.31 | 0.90 |
| locust | 4x18 | 1e-6 | 17 | 0.45 | 0.31 | 0.90 |
| locust | 4x18 | 1e-8 | 18 | 0.34 | 0.14 | 0.83 |

Every checksum (19 cases × 3 ε × 2 splits per machine, 228 in all) is CH4.0e's. What remains:
- **Small trees**: the exponential M2L's tables, one part per level, so a tree with one or two
  levels with waves builds them nearly serially (locust 1 × 72, W1 at kD = 100 at 10⁻³: 0.066 s
  of a 0.084 s build; W1 at kD = 30: 0.027 of 0.036 s). W4's trees at kD = 1 and 10 at 10⁻³
  build Batched's matrices (W4's Plummer sphere at kD = 10, 4 × 18: 0.038 of 0.068 s).
- **Large N**: W3 at kD = 100 (4·10⁶ points) changes little (0.76–0.97): its build is the
  shared steps (Kathleen 1 × 40 at 10⁻³: the routes 1.72 s, the plan 1.04 s and the octree
  0.52 s of 3.90 s; the operator 0.21 s).
- The evaluations, the errors and the checksums are CH4.0e's (no evaluation path changed).

## 5. Memory

locust, 72 threads, one rank, the peak resident size (`VmHWM`) of one process:

| | 1e-3 | 1e-6 | 1e-8 | the three in one process |
| --- | ---: | ---: | ---: | ---: |
| `main` (c878fd5) | 5,407 MB | 7,772 MB | 8,132 MB | 8,211 MB |
| this branch | 5,237 MB | 7,631 MB | 7,748 MB | 10,082 MB |

One build peaks lower than before: the second RCR build at 10⁻⁸ and the 72 scratch sets built
before the tables are gone, and the scratch sets are dropped before they are made again. Three
FMMs built and dropped one after the other in one process (the harness's `--eps a,b,c`) peak
1.9 GB higher: the parts allocate in the 72 worker threads' arenas, and glibc does not give the
freed memory back to the main thread's later buffers (likely: not measured further; limiting
the arenas, `MALLOC_ARENA_MAX=2`, raised the single builds' peaks by 0.2 GB). On one core the
branch peaks at 6,306 MB against `main`'s 6,490 MB for the three. A program that builds many
FMMs in one process holds more memory than before; not addressed here.

## 6. What it leaves

- **CH4.0j The exponential M2L's tables within a level in parts** (Helmholtz only): a level's
  `MergeLevel` (its quadrature, its `PlaneWaveLevel` and the shift factors of every source
  shift and child) is one part today, so a tree with one or two levels with waves builds them
  nearly serially; it is now most of the build on CH3.6's small cases, where CH4.0e found FMM3D
  faster because of our build. Time its parts, then split them.
- **CH4.0k The shared build steps on the node**: the plan with its rules
  (`Plan::with_rules_on`), the routes (`Points::route`) and the octree are serial in each rank:
  2.0–2.4 s of W1 at kD = 500's build at one rank per node, 2.8–3.8 s of W3 at kD = 100's. They
  are Laplace's paths (`nd-fmm-plan`, the pipeline, `nd-octree`): constraint 1, D9's benchmarks,
  its own sign-off. Meanwhile several ranks per node split them, the documented advice since
  CH4.0 (a few ranks per node with threads).

## 7. Runs

- Kathleen job 634392 (one exclusive node, `singlenode`, node-c11a-065, 2 × 20 Cascade Lake
  cores; 41 min): the release build, `ch4-0h-runs.sh build matrix` from `86c458f`; rustc
  1.99.0, BLAS threads 1, the ranks bound per socket.
- locust (GH200, 72 Neoverse V2 cores, Open MPI 5.0.10), 2026-11-16: the profile of §2
  (`profile-main-*` from `13f2089`, `profile-b1-*` from `3077b5c`), `build` and `matrix` from
  `86c458f` (07:16–07:41), the memory runs of §5 from `86c458f` and `main` (c878fd5, a separate
  tree).
- The M6: the table hashes of §1 (release, `print_the_hashes`), no timing.
- Raw outputs: `~/projects/fmm-archive/{kathleen,locust}/logs/ch4-0h/`; tables from
  `tools/helmholtz/ch4-0h-tables.py build|matrix <these launches> <CH4.0e's launches>`.

## 8. Questions for sign-off

1. **The acceptance as read.** *Recommended*: met. At the node's best split (CH4.0e's measure)
   the build is 0.15–0.23 of one core on both machines at every ε; at one rank per node
   0.44–0.66, the shared steps of CH4.0k the rest. *Alternative*: read it at one rank per
   node, which needs CH4.0k (Laplace's paths) first.
2. **The follow-ups' order.** *Recommended*: CH4.0j on the priority list beside CH4.0i (item 6;
   small, Helmholtz-only, on the cases where FMM3D is faster), CH4.0k off the list (a Laplace
   task with D9 runs; several ranks per node meanwhile). *Alternatives*: both off the list; or
   CH4.0k on it too.
3. **`nd-fmm-tables`' rule "Builds are serial and deterministic"**, amended to "deterministic;
   a build's parts may run on a caller's threads, the same bits; the crate runs no thread"
   (`fmm-tables/CLAUDE.md`; rayon stays out of the crate). *Recommended*: as amended.

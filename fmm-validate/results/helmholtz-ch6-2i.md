# Helmholtz CH6.2i: Laplace's plane-wave M2L on the coarse levels

Helmholtz task CH6.2i ([docs/helmholtz/tasks/CH6.2i.md](../../docs/helmholtz/tasks/CH6.2i.md));
added on 2026-11-08 by CH6.2f's question 2 ([helmholtz-ch6-2f.md](helmholtz-ch6-2f.md) §9;
the user: "schedule a Laplace task for 2"); proposal
[docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md) §2
(constraints 1, 2, 9). Phase 6 T13's plane-wave M2L ([phase6-planewave.md](phase6-planewave.md))
runs merge and shift on the shared drivers' split by parents, which never cuts a piece below
`PARENT_GRAIN` = 64 boxes: level 2 on one thread, level 3 on at most 8, as Helmholtz's did
before CH6.2f. This task measures what that costs Laplace and splits those levels below the
parent. Measured on:

- **locust** (NVIDIA GH200, 72 Neoverse-V2 cores, `tools/gh200/`): one rank × 1 and 72
  threads, 2026-11-09 07:29–08:26: the split by parents (07:29–07:37, source 28599ec, before
  the split existed), the three splits (07:38–07:55, d9449b7, `Auto` at half the threads),
  `Auto` at a quarter (07:58–08:03, bf7c262), D9 (08:03–08:26, `main` b73c362 against
  bf7c262). Under the run script's quiet gate (`QUIET_LINUX=1 QUIET_GPU=0`); another user's
  Python process on 1.6 cores and the H100 at the start, gone from 07:58.
- **Kathleen** (2 × 20 Cascade Lake cores, AVX-512, `tools/kathleen/`): job 630794
  (node-c11b-017, `singlenode`, exclusive, 07:56–08:28, source d9449b7), one rank × 1 and 40
  threads (`--map-by slot:PE=40 --bind-to core`), Open MPI 4.1.6, P2P `Auto` (AVX-512); D9 in
  job 630796 (§5).
- **the M6** (Apple M6, `tools/m6/machine.md`): one rank × 1 and 6 threads (`tools/scaling/run.sh`'s `CORES` on the M6), 08:09–08:15
  (source bf7c262), then D9 (08:15–08:30, every core: 12 threads), with the user's go-ahead
  ("the m6 is quiet now"), under the run script's quiet gate.

Release builds, rustc 1.99.0, BLAS threads 1. Commands: §8.

## Summary

- **The coarse levels were held** (§2). By parents level 2 ran at 0.6–1.4× its one-thread
  time at every pool size (one piece) and level 3 at 2.7–9.9× (8 pieces): at N = 10⁶ levels
  2–3 took 7–16 % of the M2L and 1.9–4.1 % of the evaluation on locust (72 threads), 5.5–10 %
  and 2.2–3.0 % on Kathleen (40), 1.3–2.3 % and 0.5–0.8 % on the M6 (6); at N = 10⁷ at most
  1.9 % of the M2L and 0.4 % of the evaluation. On Kathleen level 2 at 40 threads was
  *slower* than at one (0.6–1.0×).
- **The split by child** (§3), CH6.2f's design on Laplace's level call: each computed child
  runs its parent's call for itself alone (the same arithmetic and order, so the same bits).
  By child levels 2–3 took 0.27–0.49× the split by parents' time on locust and 0.34–0.54×
  on Kathleen; on every finer level it lost (the M2L 1.4–2.3× the split by parents'), since
  siblings repeat shared accumulators.
- **`Auto`, at a quarter of the threads** (§4). `PlaneWaveSplit::Auto` (Laplace's default
  on this branch) takes the split by child on a level call whose split by parents makes at
  most a quarter as many pieces with work as the pool has threads (level 2 from 4 threads,
  level 3 from 32), by parents elsewhere. Helmholtz's rule (half) took level 4 of the
  Plummer sphere at N = 10⁶ (32 pieces) by child at 72 threads on locust, 1.02–1.22× slower there; at
  a quarter it stays by parents. Levels 2–3 under `Auto`: 0.27–0.49× on locust, 0.39–0.54×
  on Kathleen, 0.81–0.96× on the M6 at 6 threads (level 2 alone).
- **D9** (§5): no row slower than `main` on locust and the M6; the errors equal in every run. On
  locust f32 p = 8 (−1.5 %) and f64 p = 18 (−3.8 %) are faster beyond `main`'s spread, f64
  p = 8 (−3.7 %) within it, f64 p = 12 level; the M6 within noise everywhere (−0.5 to
  +0.8 %); Kathleen's D9 is pending (job 630796). One thread (never by child) is unchanged on every
  machine.
- **Correctness** (§6): every split bit for bit the split by parents in `tests/mpi_exec.rs`
  (one thread, the most threads overlapped, four threads; `Children` on 2 and 4 ranks
  against the one-rank `Fmm`); the MPI list at 2 and 4 ranks passed.

## 1. Method

`scaling --kinds` (`KindTiming::Synchronous`, one clock pair per level call) now reports the
M2L per level, with each level's boxes, boxes with a V row and the pieces with work of the
split by parents (`parent_pieces`, the driver's halving restated; `tools/scaling/run.sh
ch6-2i`). A level's time is both passes of its level call, the waves (split box by box over
the pool) and merge and shift; on levels 2–3 merge and shift is almost all of it (the waves
of 64 boxes take microseconds). "Busy" is a level's time at one thread over its time at
many: about the threads merge and shift kept busy. Cases: Phase 6 T13's where `Auto`
resolves to `PlaneWave` (f64 p ≥ 7, f32 p ≥ 8): the cube and the Plummer sphere at N = 10⁶
(1 thread and the sweep's threads: every core on locust and Kathleen, `tools/scaling/run.sh`'s six on the M6) and 10⁷ (the sweep's threads; not on the M6), f64 p = 8, 12, 18 and f32
p = 8; three timed evaluations (two at p = 12, one at p = 18), the median. Each split ran
the same cases (`--split parents|children|auto`). The single-run evaluation times of §3
vary by up to about 10 % between launches on Kathleen and 5 % elsewhere on levels no
split changes; D9 (§5) is the measure of the evaluation.

## 2. Held: the split by parents

Levels 2 and 3 by parents (`--split parents`, or before the split existed), the time at one thread and at the sweep's threads, busy their ratio, and levels 2–3's share of the M2L and of the evaluation at the sweep's threads; at N = 10⁷ only the sweep's threads ran. Level 2 is one piece and level 3 eight on every tree (the Plummer sphere's levels 2–3 are full too).

### locust (1 × 72; one thread for the busy columns)

| case | level 2 ms, 1 / 72 threads | busy | level 3 ms, 1 / 72 | busy | levels 2–3 of M2L | of the evaluation |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| cube N = 1e6 f32 p = 8 | 0.79 / 0.89 | 0.9 | 6.55 / 1.14 | 5.7 | 15.2% | 3.2% |
| cube N = 1e6 f64 p = 8 | 1.07 / 1.05 | 1.0 | 9.38 / 1.41 | 6.7 | 10.8% | 2.4% |
| cube N = 1e6 f64 p = 12 | 2.12 / 1.71 | 1.2 | 17.96 / 2.18 | 8.2 | 9.6% | 2.9% |
| cube N = 1e6 f64 p = 18 | 6.50 / 4.56 | 1.4 | 53.70 / 5.50 | 9.8 | 7.9% | 4.1% |
| cube N = 1e7 f32 p = 8 | – / 0.89 | – | – / 1.16 | – | 1.9% | 0.3% |
| cube N = 1e7 f64 p = 8 | – / 1.04 | – | – / 1.43 | – | 1.2% | 0.2% |
| cube N = 1e7 f64 p = 12 | – / 1.75 | – | – / 2.16 | – | 1.2% | 0.3% |
| cube N = 1e7 f64 p = 18 | – / 4.80 | – | – / 5.57 | – | 1.0% | 0.4% |
| plummer N = 1e6 f32 p = 8 | 0.80 / 0.90 | 0.9 | 6.62 / 1.15 | 5.8 | 9.7% | 2.6% |
| plummer N = 1e6 f64 p = 8 | 1.09 / 1.09 | 1.0 | 9.53 / 1.41 | 6.8 | 7.3% | 1.9% |
| plummer N = 1e6 f64 p = 12 | 2.08 / 1.70 | 1.2 | 18.72 / 2.18 | 8.6 | 12.2% | 2.2% |
| plummer N = 1e6 f64 p = 18 | 6.34 / 4.56 | 1.4 | 53.46 / 5.41 | 9.9 | 10.1% | 3.7% |
| plummer N = 1e7 f32 p = 8 | – / 0.89 | – | – / 1.16 | – | 1.0% | 0.3% |
| plummer N = 1e7 f64 p = 8 | – / 1.09 | – | – / 1.40 | – | 0.6% | 0.2% |
| plummer N = 1e7 f64 p = 12 | – / 1.68 | – | – / 2.18 | – | 1.2% | 0.2% |
| plummer N = 1e7 f64 p = 18 | – / 4.99 | – | – / 6.08 | – | 1.2% | 0.4% |

### Kathleen (1 × 40; job 630794)

| case | level 2 ms, 1 / 40 threads | busy | level 3 ms, 1 / 40 | busy | levels 2–3 of M2L | of the evaluation |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| cube N = 1e6 f32 p = 8 | 1.16 / 1.78 | 0.7 | 9.74 / 3.01 | 3.2 | 10.2% | 3.0% |
| cube N = 1e6 f64 p = 8 | 1.43 / 2.23 | 0.6 | 12.12 / 3.87 | 3.1 | 8.5% | 2.9% |
| cube N = 1e6 f64 p = 12 | 2.96 / 3.79 | 0.8 | 24.99 / 5.51 | 4.5 | 7.8% | 3.0% |
| cube N = 1e6 f64 p = 18 | 9.30 / 8.87 | 1.0 | 78.69 / 11.40 | 6.9 | 6.1% | 3.0% |
| cube N = 1e7 f32 p = 8 | – / 1.78 | – | – / 2.38 | – | 1.3% | 0.3% |
| cube N = 1e7 f64 p = 8 | – / 2.64 | – | – / 2.96 | – | 1.2% | 0.3% |
| cube N = 1e7 f64 p = 12 | – / 4.17 | – | – / 5.52 | – | 1.2% | 0.3% |
| cube N = 1e7 f64 p = 18 | – / 10.06 | – | – / 11.58 | – | 0.9% | 0.4% |
| plummer N = 1e6 f32 p = 8 | 1.14 / 1.99 | 0.6 | 9.47 / 2.82 | 3.4 | 5.9% | 2.2% |
| plummer N = 1e6 f64 p = 8 | 1.46 / 2.20 | 0.7 | 12.21 / 3.68 | 3.3 | 5.5% | 2.2% |
| plummer N = 1e6 f64 p = 12 | 3.03 / 3.69 | 0.8 | 24.90 / 5.07 | 4.9 | 8.2% | 2.4% |
| plummer N = 1e6 f64 p = 18 | 9.45 / 9.86 | 1.0 | 77.11 / 11.89 | 6.5 | 6.8% | 2.9% |
| plummer N = 1e7 f32 p = 8 | – / 1.87 | – | – / 2.65 | – | 0.7% | 0.2% |
| plummer N = 1e7 f64 p = 8 | – / 2.39 | – | – / 3.12 | – | 0.6% | 0.2% |
| plummer N = 1e7 f64 p = 12 | – / 3.68 | – | – / 5.23 | – | 1.1% | 0.3% |
| plummer N = 1e7 f64 p = 18 | – / 11.02 | – | – / 11.66 | – | 1.0% | 0.4% |

### the M6 (1 × 6: `tools/scaling/run.sh`'s six cores of the M6, tools/m6/machine.md)

| case | level 2 ms, 1 / 6 threads | busy | level 3 ms, 1 / 6 | busy | levels 2–3 of M2L | of the evaluation |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| cube N = 1e6 f32 p = 8 | 0.41 / 0.42 | 1.0 | 3.65 / 1.13 | 3.2 | 2.1% | 0.7% |
| cube N = 1e6 f64 p = 8 | 0.52 / 0.49 | 1.1 | 4.48 / 1.53 | 2.9 | 2.0% | 0.6% |
| cube N = 1e6 f64 p = 12 | 1.13 / 0.91 | 1.2 | 9.70 / 2.79 | 3.5 | 1.8% | 0.7% |
| cube N = 1e6 f64 p = 18 | 3.49 / 2.63 | 1.3 | 28.76 / 7.52 | 3.8 | 1.7% | 0.8% |
| plummer N = 1e6 f32 p = 8 | 0.44 / 0.41 | 1.1 | 3.71 / 1.11 | 3.4 | 1.3% | 0.5% |
| plummer N = 1e6 f64 p = 8 | 0.51 / 0.49 | 1.0 | 4.48 / 1.63 | 2.7 | 1.3% | 0.5% |
| plummer N = 1e6 f64 p = 12 | 1.15 / 0.91 | 1.3 | 9.39 / 2.71 | 3.5 | 2.3% | 0.6% |
| plummer N = 1e6 f64 p = 18 | 3.32 / 2.46 | 1.3 | 28.34 / 7.32 | 3.9 | 2.1% | 0.8% |

## 3. The split by child

At the sweep's threads: levels 2–3, the M2L and the evaluation (single runs, the median of the launch's evaluations) by parents, under `Auto` and by child on every level. On locust `Auto` is the quarter rule's run (bf7c262); on Kathleen the half rule's, which takes the same levels at 40 threads (§4); on the M6 the quarter rule's.

### locust (1 × 72; one thread for the busy columns)

| case | levels 2–3 ms: parents / auto / children | M2L ms: parents / auto / children | evaluation ms: parents / auto / children |
| --- | --- | --- | --- |
| cube N = 1e6 f32 p = 8 | 2.03 / 0.96 / 0.98 | 13.37 / 11.94 / 19.64 | 63.98 / 59.76 / 71.92 |
| cube N = 1e6 f64 p = 8 | 2.46 / 1.11 / 1.13 | 22.74 / 21.14 / 38.26 | 100.91 / 99.97 / 120.50 |
| cube N = 1e6 f64 p = 12 | 3.89 / 1.40 / 1.54 | 40.46 / 36.92 / 73.09 | 134.34 / 126.67 / 167.82 |
| cube N = 1e6 f64 p = 18 | 10.06 / 2.81 / 2.99 | 127.29 / 118.97 / 241.52 | 248.16 / 238.68 / 374.15 |
| cube N = 1e7 f32 p = 8 | 2.05 / 0.95 / 0.93 | 109.32 / 105.19 / 163.99 | 725.46 / 654.27 / 781.54 |
| cube N = 1e7 f64 p = 8 | 2.48 / 1.08 / 1.12 | 199.52 / 191.70 / 320.78 | 1193.00 / 1136.17 / 1324.87 |
| cube N = 1e7 f64 p = 12 | 3.91 / 1.43 / 1.41 | 333.03 / 322.55 / 625.25 | 1438.21 / 1369.16 / 1728.25 |
| cube N = 1e7 f64 p = 18 | 10.37 / 2.89 / 2.78 | 1015.12 / 1010.95 / 2347.53 | 2390.30 / 2332.03 / 3717.22 |
| plummer N = 1e6 f32 p = 8 | 2.04 / 1.00 / 0.93 | 21.01 / 20.49 / 32.53 | 78.99 / 77.46 / 92.88 |
| plummer N = 1e6 f64 p = 8 | 2.50 / 1.08 / 1.11 | 34.05 / 31.43 / 64.94 | 130.69 / 124.83 / 163.35 |
| plummer N = 1e6 f64 p = 12 | 3.87 / 1.42 / 1.45 | 31.83 / 29.88 / 55.08 | 176.12 / 162.58 / 196.78 |
| plummer N = 1e6 f64 p = 18 | 9.97 / 2.93 / 2.92 | 98.25 / 91.85 / 169.98 | 271.75 / 264.91 / 345.45 |
| plummer N = 1e7 f32 p = 8 | 2.06 / 0.95 / 1.01 | 212.28 / 211.26 / 326.09 | 777.77 / 780.03 / 921.63 |
| plummer N = 1e7 f64 p = 8 | 2.49 / 1.13 / 1.13 | 402.35 / 385.78 / 690.97 | 1368.70 / 1288.54 / 1657.98 |
| plummer N = 1e7 f64 p = 12 | 3.87 / 1.54 / 1.47 | 319.20 / 322.34 / 580.90 | 1660.97 / 1635.86 / 1883.07 |
| plummer N = 1e7 f64 p = 18 | 11.07 / 2.95 / 2.94 | 960.79 / 913.94 / 2127.64 | 2627.12 / 2543.31 / 3739.93 |

### Kathleen (1 × 40; job 630794)

| case | levels 2–3 ms: parents / auto / children | M2L ms: parents / auto / children | evaluation ms: parents / auto / children |
| --- | --- | --- | --- |
| cube N = 1e6 f32 p = 8 | 4.78 / 1.96 / 1.94 | 46.95 / 49.58 / 73.70 | 158.03 / 169.32 / 185.76 |
| cube N = 1e6 f64 p = 8 | 6.10 / 2.39 / 2.38 | 71.73 / 66.31 / 107.99 | 209.19 / 202.98 / 246.62 |
| cube N = 1e6 f64 p = 12 | 9.31 / 3.60 / 3.31 | 119.81 / 116.41 / 185.60 | 314.82 / 316.86 / 382.01 |
| cube N = 1e6 f64 p = 18 | 20.27 / 8.92 / 8.56 | 334.84 / 325.04 / 653.67 | 684.93 / 676.24 / 997.66 |
| cube N = 1e7 f32 p = 8 | 4.16 / 1.85 / 2.23 | 315.95 / 311.69 / 546.00 | 1494.58 / 1485.89 / 1713.14 |
| cube N = 1e7 f64 p = 8 | 5.60 / 2.20 / 2.20 | 451.44 / 439.49 / 752.39 | 1891.33 / 1895.43 / 2219.76 |
| cube N = 1e7 f64 p = 12 | 9.69 / 4.05 / 3.26 | 809.39 / 800.58 / 1419.84 | 2799.97 / 2784.70 / 3437.37 |
| cube N = 1e7 f64 p = 18 | 21.64 / 9.21 / 10.21 | 2490.92 / 2521.54 / 5762.17 | 5843.10 / 5879.45 / 9126.70 |
| plummer N = 1e6 f32 p = 8 | 4.80 / 1.93 / 1.93 | 82.08 / 76.86 / 120.24 | 222.05 / 211.57 / 255.90 |
| plummer N = 1e6 f64 p = 8 | 5.88 / 2.38 / 2.31 | 106.32 / 112.27 / 155.98 | 264.77 / 280.36 / 309.43 |
| plummer N = 1e6 f64 p = 12 | 8.76 / 3.44 / 3.66 | 107.31 / 106.20 / 154.87 | 359.76 / 357.92 / 407.70 |
| plummer N = 1e6 f64 p = 18 | 21.74 / 11.34 / 9.08 | 317.94 / 292.13 / 523.86 | 749.22 / 710.84 / 958.01 |
| plummer N = 1e7 f32 p = 8 | 4.52 / 1.98 / 1.93 | 636.30 / 617.41 / 1079.30 | 1914.08 / 1878.90 / 2331.61 |
| plummer N = 1e7 f64 p = 8 | 5.51 / 2.36 / 2.32 | 915.05 / 908.87 / 1578.47 | 2487.46 / 2481.18 / 3119.92 |
| plummer N = 1e7 f64 p = 12 | 8.91 / 4.06 / 3.83 | 796.11 / 806.15 / 1349.15 | 3002.36 / 3022.29 / 3596.78 |
| plummer N = 1e7 f64 p = 18 | 22.68 / 12.19 / 8.97 | 2369.65 / 2431.53 / 5598.04 | 5953.44 / 6052.49 / 9201.08 |

### the M6 (1 × 6: `tools/scaling/run.sh`'s six cores of the M6, tools/m6/machine.md)

| case | levels 2–3 ms: parents / auto / children | M2L ms: parents / auto / children | evaluation ms: parents / auto / children |
| --- | --- | --- | --- |
| cube N = 1e6 f32 p = 8 | 1.55 / 1.33 / 1.88 | 72.17 / 71.97 / 137.12 | 233.86 / 238.88 / 306.21 |
| cube N = 1e6 f64 p = 8 | 2.02 / 1.70 / 2.64 | 100.69 / 102.45 / 214.85 | 330.59 / 339.53 / 458.35 |
| cube N = 1e6 f64 p = 12 | 3.69 / 3.35 / 4.66 | 201.86 / 211.98 / 384.70 | 542.82 / 556.02 / 740.00 |
| cube N = 1e6 f64 p = 18 | 10.15 / 9.38 / 12.87 | 595.43 / 613.97 / 1085.94 | 1199.27 / 1233.14 / 1714.63 |
| plummer N = 1e6 f32 p = 8 | 1.52 / 1.35 / 1.85 | 114.08 / 114.00 / 216.69 | 300.05 / 303.45 / 403.69 |
| plummer N = 1e6 f64 p = 8 | 2.12 / 1.71 / 2.69 | 160.81 / 163.38 / 354.22 | 416.03 / 428.29 / 613.67 |
| plummer N = 1e6 f64 p = 12 | 3.62 / 3.09 / 4.43 | 156.57 / 156.72 / 288.15 | 641.23 / 654.26 / 774.82 |
| plummer N = 1e6 f64 p = 18 | 9.78 / 9.39 / 12.38 | 467.75 / 462.43 / 835.82 | 1213.59 / 1247.45 / 1609.28 |

## 4. The rule

`Auto` decides once per level call from the pool's threads and the pieces with work of the
split by parents (`driver::parent_pieces`, moved from Helmholtz's `pieces`): by child where
`PLANE_CHILD_SHARE` (4) × pieces ≤ threads. The levels it takes:

| threads | level 2 (1 piece) | level 3 (8) | Plummer level 4 (32) | cube level 4 (64) |
| ---: | --- | --- | --- | --- |
| 1–3 | parents | parents | parents | parents |
| 4–31 (the M6's 6 and 12) | **child** | parents | parents | parents |
| 32–127 (Kathleen's 40, locust's 72) | **child** | **child** | parents | parents |

Helmholtz's rule (CH6.2f, at most half) would take Plummer's level 4 at N = 10⁶ (32 pieces;
64 at 10⁷) by child from 64 threads: on locust that level took 1.02–1.22× the split by parents' time by child (1.32 /
1.29, 2.22 / 2.01, 5.86 / 4.80 and 1.81 / 1.64 ms at f32 p = 8 and f64 p = 12, 18, 8), so
Laplace's rule takes a quarter; Helmholtz keeps its own (its conversions are a larger share
of its merge and shift, CH6.2f §2). With the quarter rule locust's Plummer level 4 ran by
parents (1.28, 1.99, 4.89, 1.57 ms). At 40 threads both rules take the same levels, so
Kathleen's job (run on the half rule) measured the quarter rule's choice. At 6 threads the
M6 splits level 2 alone: level 3's 8 pieces already keep its 6 threads busy (2.7–3.9×).

## 5. D9

D9 as signed off: five runs per side, A B B A A B B A A B, A `main` (b73c362, a detached
worktree with its own target directory) and B the branch (bf7c262); each run `tools/bench/run.sh`
(nd-fmm-bench) at N = 10⁶ on the cube under `auto` (`PlaneWave`) with 10 timed evaluations
after 2 warm-ups and the kinds timed (`sync`, both sides): f64 p = 8, 12, 18 and f32 p = 8 on
every core, f64 p = 8 on one thread (where `Auto` never splits by child: the code path's
cost alone). Per row the spread is max − min of `main`'s five run medians; the branch fails
if its median of medians is slower than `main`'s by more than the spread, and is faster
beyond noise if faster by more than it. "Errors equal": the φ and ∇φ errors at the 1,000
sampled targets the same in all ten runs.

### locust (72 threads)

| precision | p | threads | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians | within noise | errors equal |
| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| f32 | 8 | 72 | 63.36, 63.06, 62.63, 63.05, 63.33 | 61.98, 62.93, 60.60, 62.59, 62.12 | 0.73 | −0.94 (−1.5 %) | faster beyond | yes |
| f64 | 8 | 72 | 103.36, 99.99, 100.04, 103.10, 104.21 | 102.87, 99.28, 98.70, 98.52, 102.84 | 4.23 | −3.82 (−3.7 %) | yes | yes |
| f64 | 12 | 72 | 127.22, 131.14, 128.50, 127.49, 127.09 | 127.55, 126.03, 124.24, 129.23, 128.29 | 4.05 | +0.06 (+0.0 %) | yes | yes |
| f64 | 18 | 72 | 247.85, 249.66, 245.19, 245.91, 247.50 | 238.17, 237.96, 237.06, 239.00, 241.17 | 4.47 | −9.33 (−3.8 %) | faster beyond | yes |
| f64 | 8 | 1 | 3551.86, 3529.53, 3536.03, 3549.07, 3514.70 | 3552.78, 3534.52, 3507.50, 3522.14, 3549.12 | 37.16 | −1.51 (−0.0 %) | yes | yes |

### Kathleen (40 threads)

Pending: job 630796 (`tools/kathleen/submit.sh --time=01:30:00 -- run.sbatch 'bash
<root>/logs/ch6-2i-d9.sh <root>/src/d9-main-b73c362 <root>/src/tbetcke_h-ch6-2i
<root>/logs/ch6-2i-d9'`), queued for a node since 08:30.

### the M6 (12 threads, every core)

| precision | p | threads | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians | within noise | errors equal |
| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| f32 | 8 | 12 | 164.97, 163.37, 164.50, 165.40, 162.60 | 164.02, 163.39, 163.44, 164.07, 165.83 | 2.80 | −0.48 (−0.3 %) | yes | yes |
| f64 | 8 | 12 | 231.76, 226.49, 226.72, 229.16, 227.20 | 227.88, 229.09, 229.25, 229.03, 227.64 | 5.27 | +1.83 (+0.8 %) | yes | yes |
| f64 | 12 | 12 | 381.60, 376.88, 380.83, 379.68, 380.33 | 378.45, 377.38, 378.38, 377.34, 378.91 | 4.72 | −1.95 (−0.5 %) | yes | yes |
| f64 | 18 | 12 | 869.09, 865.71, 871.98, 874.27, 876.18 | 867.78, 856.97, 876.60, 871.05, 876.90 | 10.47 | −0.93 (−0.1 %) | yes | yes |
| f64 | 8 | 1 | 1332.92, 1340.95, 1345.25, 1354.19, 1341.75 | 1333.64, 1340.68, 1342.40, 1344.57, 1349.24 | 21.27 | +0.65 (+0.0 %) | yes | yes |

## 6. Tests and checks

- `tests/mpi_exec.rs`, `check_plane_wave_split` (in `check_plane_wave`, on every scenario and
  precision it runs: 17 checks at one rank in debug): `Children` and `Parents` named at one
  thread and at the most threads overlapped, the default at four threads, each bit for bit
  the default's output; `Auto` never by child at one thread and by child at four on some rank
  (the doc-hidden `Fmm::plane_wave_by_child`); on 2 and 4 ranks `Children` at two threads
  against the one-rank `Fmm`. One rank in debug: 169 s (the one-rank run of the whole file,
  165–195 s before on the M6 and the M3 Max; the split checks about 4 s). The trusted path is
  the split by parents, itself checked against the direct sum, the exact M2L and the pair
  path (Phase 6 T13).
- `driver::parent_pieces` (unit test, moved from Helmholtz's `pieces`);
  `nd-fmm-validate`: `parent_pieces_restate_the_split_by_parents` and `tests/scaling.rs` (the
  levels sum to the M2L kind's time, a level timed only with a V row, the summary's keys).
- Helmholtz's split is unchanged (`PlaneWaveSplit` re-exported from `helmholtz`, its rule
  calling the moved `parent_pieces`): `tests/helmholtz_fmm.rs` passed at 2 and 4 ranks.
- Laplace's fixtures and bit-identity tests are not edited.

| check | where | result |
| --- | --- | --- |
| `cargo fmt --all -- --check` | the M6 | passed |
| `cargo clippy -- -D warnings`, `--examples`, `--workspace --all-targets` | the M6 | passed |
| `cargo doc --no-deps` | the M6 | passed |
| `RUST_MIN_STACK=8388608 cargo test` | the M6 | passed (1226 tests, 811 s) |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | the M6 | passed (1340 tests, 1037 s) |
| `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks (12 executables × 2) | the M6, bf7c262 | passed (24 of 24) |
| `tests/mpi_exec.rs` at 1, 2, 4, 8 ranks (debug) | the M6, 15a3f5e | passed (169, 229, 220, 218 s) |

## 7. Questions for sign-off

Signed off 2026-11-09 (the user: "ok, 1–3 as recommended"). The questions as asked:

1. **Constraint 1: adopt `PlaneWaveSplit::Auto` as Laplace's default** (a change to a
   Laplace path, the plane-wave level call). (1) No dispatch below the level call: one choice
   per level call; (2) layouts and tables unchanged; (3) outputs bit for bit at every split,
   thread count, overlap setting and rank count (§6); (4) D9: no row slower on the three
   machines, two rows on locust faster beyond the spread (§5). Recommended: adopt (the branch
   has it as the default). Alternative: keep `Parents` as the default and `Auto` opt-in.
2. **Laplace's rule at a quarter of the threads, Helmholtz's at half** (§4). Recommended:
   accept, each by its own measurement; one rule for both would need Helmholtz's runs again.
3. **`driver.rs`**: one crate-private function added (`parent_pieces`, moved from
   Helmholtz's `operator_plane_wave.rs`), no existing driver code changed. Recommended:
   accept.

## 7a. Follow-ups

- None required. A split in two phases (each parent's accumulators once on the pool, then
  its children; CH6.2f §9) would remove the repeated shifts that make the split by child
  lose on the fine levels, but `Auto` never takes it there.

## 8. Reproducing

- locust: `tools/gh200/sync.sh`, then `OUT=… QUIET_LINUX=1 QUIET_GPU=0 tools/scaling/run.sh
  ch6-2i` (and with `CH6_2I_SPLITS="parents children auto" ONLY='r1x72$'`) through
  `tools/gh200/remote.sh`; Kathleen: `tools/kathleen/submit.sh --time=05:00:00
  '--export=ALL,CH6_2I_SPLITS=parents children auto' -- sweep.sbatch ch6-2i`; the M6:
  `CH6_2I_SPLITS="parents children auto" tools/scaling/run.sh ch6-2i` outside the sandbox.
- D9: a detached worktree of `main` with its own target directory, then on each machine a
  script (session scratchpad, not committed, as in CHR.2 and CH2.0) running the ten runs of
  §5 with `tools/bench/run.sh --precision f64 --degree 8,12,18 --strategy auto`,
  `--precision f32 --degree 8 --strategy auto` and `--precision f64 --degree 8 --threads 1
  --strategy auto`; on Kathleen inside `run.sbatch`.
- Raw outputs: `~/projects/fmm-archive/{locust,kathleen,m6}/logs/ch6-2i/`.

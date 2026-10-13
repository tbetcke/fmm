# The work-weighted cut (Phase 5S T7)

S3, as docs/phase5s/T7-work-weighted-cut.md asks and docs/design/scale-out.md §6 (signed
off, question 5) specifies: after the plan of the first cut (by points), every rank counts
the work of each of its coarse blocks by operator kind, weighs it with per-kind constants
(the cost model), and the octree cuts the blocks again by those weights
(`Octree::repartition`); the points move once, after the second cut. Run on 2026-10-13 on
the M3 Max, locust (GH200) and Kathleen.

**Measured** means run here and read from the output; timings are reported, never
asserted. locust is a shared node: its load is in each output file before and after
every launch (idle but for these launches throughout). Kathleen nodes are exclusive; every
job is named. Oversubscribed runs are counted, never timed.

"First cut" is the cut by points that `Octree::new` makes (O3, Phase 5 T4); "second cut"
the cut by modelled cost of this task (`FmmBuilder::work_cut(true)`, the default). Both
are the same binary, launched with `--cut points` and `--cut work`.

## Summary

- **Built as designed**: `Octree::repartition(&weights)` in `nd-octree` (one all-gather-v of
  the block weights, O3's nearest-boundary cut on them, one all-to-all-v of the leaves,
  the partition map and H rebuilt; the leaves, the blocks and every key unchanged); the
  cost model in `nd-fmm-exec` (`cost`: per block the eight counts of the plan's views
  weighed by per-kind constants, Kathleen's table `CostModel::calibrated` per strategy,
  precision and p, `FmmBuilder::cost_model` for others); `FmmBuilder::work_cut`, on by
  default on several ranks (decision 3), moving blocks only where the second cut lowers
  the busiest rank's block cost by at least 2 % (signed off on 2026-10-13, §9, §10); the
  harness's `--cut`, `--alpha`, counts, model and fit, and the sweeps `calibrate`, `cut`,
  `cut-counts`.
- **Bit for bit**: with the second cut on, `mpi_exec` (with the new scenario
  **work-weighted cut**, the cut off against on) at 1, 2, 4 and 8 ranks on the M3 Max and
  locust, `multi_rank` at 1, 2, 4 and 8 (0 values differing), rank-count invariance at 1,
  2, 4, 8, at 512 against 64 (locust), at 80 against 40, 20, 10, at 160 against 80, 40,
  20 and at 640 against 80 (Kathleen): 0 values differing; the leaves equal the first
  cut's key for key on every workload of `test_mpi_weighted_partition` at 1–8 ranks.
- **The model holds**: constants fitted to the kind times reproduce every rank's compute
  within 1–4 % on Kathleen and locust (f32 `Dense` 4–22 %, a configuration the rule does
  not pick at one rank per core), and its max over mean follows the measured one within
  0.03 with one thread per rank (§3, §4).
- **Measured (timed)**: at one rank per core from 40 ranks up the compute stages' max over
  mean falls from 1.06–1.58 to 1.02–1.10 and the evaluation's wall time by 1–32 % (the
  Plummer sphere 14–32 %, at 160 Kathleen ranks 1.51–1.58 → 1.07–1.09); at 2–8 ranks
  (locust 8 × 1, Kathleen 2–8 × 20) it changes −4.9 % to +1.3 % (§4).
- **Counted at 512 and 640 oversubscribed ranks**: the modelled max over mean 1.09–2.79 by
  points, 1.02–1.12 by the second cut; within 1.12 in 17 of 18 rows (the cube at N = 10⁶
  on 512 ranks: 1.124) and within 1.10 in 16 (§5); **accepted as T7's outcome on
  2026-10-13** (§9, question 1).
- **Build cost**: one to one and a half `Plan::new`, 31–41 % of the build at 40–160 ranks,
  above the quarter the design set for the default (§6). Signed off (§9, question 2): on
  by default with the 2 % gain rule, which keeps the cut by points, and its plan, in 16
  of the 18 configurations at 2–8 ranks, where the cut gained nothing, and cuts again in
  every one with one rank per core from 40 ranks up (§10). Kathleen: 5.2 node-hours in
  eight jobs (§8).

## 1. The change

`octree/src/octree.rs` (`Octree::repartition`), `octree/src/octree/implementation.rs`
(`quantize_weights`, `gather_block_weights`, `move_leaves`), `octree/src/lib.rs`, the
examples `test_mpi_weighted_partition` and `test_mpi_locally_essential`;
`fmm-exec/src/cost.rs` (new), `fmm.rs`, `lib.rs`, `tests/mpi_exec.rs`;
`fmm-validate/src/scaling.rs` (with `fmm_accuracy.rs`, which aggregates `BuildTimings`),
`examples/scaling.rs`, `tests/scaling.rs`; `tools/scaling/run.sh` and
`tools/kathleen/README.md`; the crates' CLAUDE.md files. `nd-fmm-plan` is unchanged.

1. **`Octree::repartition(&self, weights: &[f64]) -> Octree`** (scale-out.md §6.3, item 2):
   one weight per own block, finite and not negative (checked before any communication).
   One all-gather-v of the weights (counts from the partition map); the weights scaled to
   integers summing to about 2⁵² (`round(w 2⁵² / W)`), so that `partition_blocks`, O3's
   nearest-boundary cut with ties to the later boundary, cuts them exactly as it cuts keys
   and its tests and bound carry over; the leaves moved to their new owners (one
   all-to-all-v; they arrive sorted, since every rank's leaves lie in contiguous ranges);
   the partition map rebuilt from the same blocks with the new offsets (a block's leaf bit
   does not depend on the cut, so no gather); H rebuilt by `generate_all_keys`'s push. The
   coarse tree, the leaves and every key stay; only ownership changes. When the weights
   cut the blocks as they are already cut (every rank finds this from the same gathered
   weights) it returns a copy of the tree without the move and the push, and so it does
   when the caller's `min_gain` is not reached: the heaviest rank's weight under the new
   cut, from the gathered (scaled) weights, must be at most 1 − `min_gain` times that under
   the current cut. On one rank the tree is unchanged. `nd-octree` never interprets a
   weight.
2. **The cost model** (`nd_fmm_exec::cost`, scale-out.md §6.1): per rank and per block the
   eight counts c_k of the plan's views (P2M and L2P points of the local leaves; M2M and
   L2L pairs; V pairs; X pairs times the source leaf's points; W pairs times the target
   leaf's points; near interactions Σ_leaf n_t Σ_near n_s), the rows of a box or leaf
   counted to the own block at or above it, the rows of the `Global` boxes (shared
   ancestors) to no block (`WorkCounts::rest`: the cut cannot move them); the modelled
   cost Σ_k α_k c_k with constants α_k in ns per counted unit (`CostModel`).
   `CostModel::calibrated(strategy, value bytes, p)` holds Kathleen's constants (§3) for
   f64 p = 3 `Dense`, f64 p = 8 `Rotation`, f32 p = 8 `Rotation` and `Dense`, and scales
   the nearest calibration of the same strategy and precision by (p + 1)² (leaf
   operators), (p + 1)³ (`Rotation`'s translations) or (p + 1)⁴ (dense translations) for
   another p. `FmmBuilder::cost_model` names others (locust's, §3).
3. **`Fmm`**: `FmmBuilder::work_cut` (on by default on several ranks, decision 3; nothing
   on one rank). Step 4 of `build`, after the first `Plan::new`: the points of every local
   leaf counted where they are (`cost::leaf_point_counts`: each rank sorts the keys of its
   sources and targets, merges them into distinct keys with both counts and sends each to
   its owner, one all-to-all-v of 24 bytes per distinct key); the ghost leaves' source
   counts from the build of a `SourceExchange<u8>` (the plan's own exchange, one value per
   point); the work per block (`cost::work_counts`); the weights under the cost model (by
   default `CostModel::calibrated` for the strategy the host rule resolves with the ranks
   per node, which step 2's all-reduce now agrees as a second maximum, so no collective is
   added for it); `Octree::repartition` with a minimum gain of 2 %
   (`WORK_CUT_MIN_GAIN`, §9 question 2: the new cut only where it lowers the busiest rank's
   block cost by at least that much); and, where the cut changed, a second `Plan::new`. Then the points move once, by the redistributions of step 5.
   `BuildTimings::cut` times it; `Fmm::{work_cut, recut, cost_model, kind_counts}` report
   it (the last from the final plan, without communication).
4. **Tests.** `nd-octree`: the unit test `test_cut_by_real_weights` (the scaling, heavy
   blocks, ties in tiny real weights, empty ranges, no weight, randomised against the
   nearest boundaries of the real weights and O3's bound); `test_mpi_weighted_partition`
   cuts every workload a second time by five weights per block (the square of its leaves,
   the levels of its leaves cubed, one heavy block, all zero, all one) and checks **the
   leaves against the first cut's, key for key**, the blocks and leaf flags, the
   nearest-boundary cut of the gathered weights and O3's bound, ownership, the held keys'
   kinds, one rank unchanged, and a repeat by the same weights a fixed point;
   `test_mpi_locally_essential` runs its brute-force checks of H on every tree cut a
   second time. `nd-fmm-exec`: every `mpi_exec` scenario, `invariance` and `multi_rank`
   run with the cut on (the default); the new scenario **work-weighted cut** builds a
   cloud with a dense blob capped at level 5 (whose P2P outweighs its points) with the cut
   off and on, bit for bit each other and the one-rank reference, checks that the cut
   moves points on every rank and that the work by kind summed over the ranks equals the
   one-rank `Fmm`'s for the kinds no rank repeats (P2M, M2M, L2P, M2P, P2P) and is at
   least it for M2L, P2L and L2L; **input distributions** takes the owners' input from an
   `Fmm`'s own octree (both cuts). Unit tests of `CostModel`.
5. **`scaling`**: `--cut work|points`, `--alpha` (eight constants), the build part "second
   cut", per rank the work by kind and the modelled cost, and with `--kinds` the constants
   fitted to the kind times (`PathTimes::fit`, `ModelFit`: per kind the time summed over
   the ranks over the count summed over the ranks, scale-out.md §6.1's estimator) with the
   model's largest error on a rank and the max over mean of the measured kind sum, of the
   fitted model and of the run's model. `tools/scaling/run.sh`: the sweeps `calibrate`,
   `cut` and `cut-counts` (oversubscribed on Kathleen with PSM2's shared contexts) and
   `ALPHA`.

## 2. Readings and departures (for review)

1. **The leaf counts are counted, not taken from `Octree::new`.** The design's §6.3 has
   "the octree records the keys per leaf, with multiplicity": `Octree::new` weighs sources
   and targets together, but the model needs them apart (P2P is n_t × n_s), and nothing of
   the octree's API carries them. `Fmm` counts them itself with one all-to-all-v of
   distinct keys (24 B each), cheaper than a `Redistribution` (12 B per point and set).
2. **The weights are per block, and only blocks move** (§6.3, items 1–2). The rows of the
   shared ancestors (their M2L, P2L, L2L on every sharer, their global M2M on the owner)
   stay with the ranks that share them and are not part of a block's weight; the
   modelled per-rank cost in every report includes them.
3. **No refinement by cost.** The brief forbids changing O1's coarse tree, and the
   signed-off target reads 1.12 without the refinement (question 5). It also cannot be
   added by splitting heavy blocks alone: H's assembly relies on a 2:1 balanced coarse tree
   (§3.3), so split blocks need re-balancing, whose new blocks need weights only their
   owners know: one more round of gathers. §5 shows where it would matter.
4. **An unchanged cut costs no second plan.** Where the second cut lands on the first
   cut's boundaries, or gains less than 2 % (§9, question 2), `repartition` returns a copy
   and `build` keeps its first plan (§6). The gain is measured on the blocks' weights, not
   on the modelled cost with the shared ancestors' rows, which no cut moves.
5. **The strategy of the constants** is the host rule's (named, or `Auto` resolved with
   the ranks per node), also on a device backend, where the tuned strategy may differ;
   only the balance depends on it. A precision or strategy without a calibration falls
   back as item 2 of §1 says.
6. **The model on threaded ranks is looser** (2 × 20 on Kathleen), as the design found
   (§6.1): there the cut changes little anyway, with 2–8 ranks (§4).

## 3. The cost model: calibration and fit

`tools/scaling/run.sh calibrate`: one rank per core with one thread, the first cut, three
timed evaluations with `KindTiming::Synchronous`, the cube, the Plummer sphere and the
clusters at N = 10⁷; per kind α_k = Σ_r t_k(r) / Σ_r c_k(r) (scale-out.md §6.1), and the
model with the fitted α_k against every rank's sum of kind times (`ModelFit`). Kathleen:
40 ranks on one node, job 603049; locust: 72 ranks.

| Configuration | Kathleen α, ns: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P (median of the three workloads) | largest error on a rank (cube / Plummer / clusters) | locust α, ns | largest error |
| --- | --- | --- | --- | --- |
| f64 p = 8 `Rotation` | 664, 2,755, 3,129, 568, 2,799, 1,087, 1,299, 3.09 | 1.0 / 2.6 / 3.8 % | 179, 1,002, 1,118, 171, 1,066, 395, 470, 2.55 | 1.5 / 1.6 / 1.9 % |
| f64 p = 3 `Dense` | 113, 112, 117, 109, 125, 232, 330, 3.07 | 1.1 / 4.2 / 3.2 % | 31.9, 62.1, 67.1, 39.0, 72.0, 89.7, 124, 2.54 | 3.2 / 2.4 / 2.4 % |
| f32 p = 8 `Rotation` | 596, 2,561, 2,806, 600, 2,604, 1,077, 1,273, 1.35 | 2.1 / 3.4 / 2.9 % | 177, 931, 1,041, 168, 946, 389, 465, 0.92 | 1.3 / 1.9 / 2.8 % |
| f32 p = 8 `Dense` | 600, 1,417, 4,322, 629, 1,558, 1,098, 1,288, 1.36 | 4.2 / 8.0 / 5.2 % | 179, 661, 3,650, 173, 768, 402, 477, 0.95 | 7.2 / 22.4 / 10.7 % |

(Medians per kind over the three workloads, on both machines.) The Kathleen column is `CostModel::calibrated`'s
table; the locust f64 p = 8 `Rotation` constants were passed with `ALPHA` to every locust
`cut` and `cut-counts` launch. They agree with scale-out.md §6.1's (fitted at 8–160 ranks
there) within 5 % for every kind but P2M and M2P, which are 2–11 % higher here. Readings:
- **The model holds within 1–4 % per rank** in the three configurations the rule picks at
  one rank per core (f64 `Rotation` and `Dense`, f32 `Rotation`), on both machines. The
  spread between workloads comes from P2P (the cube's full leaves are cheaper per
  interaction, 2.79 against 3.09–3.16 ns on Kathleen).
- **f32 `Dense` at one rank per core is looser** (Kathleen 4–8 %, locust 7–22 %): every
  rank streams its own dense tables (17 MB in f32 at p = 8), so an M2L's cost depends on
  the V rows' locality as well as their number. At 40 or 72 ranks per node the rule picks
  `Rotation` there (the node budget), so the default cut never weighs with these.
- **The machines' ratios differ** (M2L over P2P: 1,014 on Kathleen, 438 on locust), as the
  design found: a constants table is per machine class; locust's runs name theirs.
- The calibration's own imbalance columns: by points the measured compute's max over mean
  at 40 Kathleen ranks is 1.07 (cube), 1.32 (Plummer) and 1.13 (clusters), the fitted
  model's 1.07, 1.31 and 1.11 (f64 p = 8).

## 4. Timed: the first cut against the second

`tools/scaling/run.sh cut`: f64 p = 8 with the strategy of the rule (`Rotation` in every
configuration here), overlap off, three timed evaluations after a warm-up, `--kinds`
(the kind timings add one clock pair per level call on the host), `--cut points` and
`--cut work` from the same binary one after the other. locust with locust's constants
(§3), Kathleen with the library's (Kathleen's). "compute" is the max over mean of the
compute stages (upward local, downward, leaves; the harness's measure since Phase 5 T10),
"model" the modelled cost's max over mean with the run's constants, "points" the owned
sources' max over mean after the second cut; wall and build in ms (the build's max over
the ranks, with the second cut's part); "break-even" the evaluations after which the
second cut's build is paid back by the evaluations' wall time.

| Workload | N | ranks × threads | compute: first → second | model: first → second | points, second | wall, ms | build, ms (second cut) | break-even |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cube | 10⁶ | locust, 8 × 1 | 1.010 → 1.008 | 1.002 → 1.002 | 1.004 | 1146 → 1144 (−0.2 %) | 115 → 186 (71) | 31.6 |
| plummer | 10⁶ | locust, 8 × 1 | 1.050 → 1.050 | 1.049 → 1.049 | 1.027 | 2474 → 2456 (−0.7 %) | 176 → 291 (115) | 6.5 |
| clusters | 10⁶ | locust, 8 × 1 | 1.066 → 1.009 | 1.063 → 1.021 | 1.076 | 2672 → 2541 (−4.9 %) | 188 → 319 (129) | 1.0 |
| cube | 10⁷ | locust, 8 × 1 | 1.022 → 1.008 | 1.002 → 1.002 | 1.002 | 11193 → 11004 (−1.7 %) | 1492 → 2190 (699) | 3.7 |
| plummer | 10⁷ | locust, 8 × 1 | 1.019 → 1.023 | 1.020 → 1.020 | 1.019 | 22576 → 22697 (+0.5 %) | 2390 → 3668 (1286) | never |
| clusters | 10⁷ | locust, 8 × 1 | 1.028 → 1.021 | 1.031 → 1.026 | 1.051 | 23397 → 23163 (−1.0 %) | 2504 → 3898 (1367) | 6.0 |
| cube | 10⁶ | locust, 72 × 1 | 1.147 → 1.019 | 1.141 → 1.015 | 1.149 | 153 → 137 (−10.3 %) | 26 → 34 (10) | 0.5 |
| plummer | 10⁶ | locust, 72 × 1 | 1.238 → 1.055 | 1.246 → 1.054 | 1.382 | 338 → 290 (−14.3 %) | 39 → 59 (19) | 0.4 |
| clusters | 10⁶ | locust, 72 × 1 | 1.331 → 1.056 | 1.347 → 1.056 | 1.206 | 389 → 311 (−20.0 %) | 42 → 66 (21) | 0.3 |
| cube | 10⁷ | locust, 72 × 1 | 1.066 → 1.025 | 1.067 → 1.017 | 1.071 | 1364 → 1308 (−4.0 %) | 204 → 291 (89) | 1.6 |
| plummer | 10⁷ | locust, 72 × 1 | 1.315 → 1.047 | 1.320 → 1.049 | 1.406 | 3385 → 2735 (−19.2 %) | 385 → 564 (173) | 0.3 |
| clusters | 10⁷ | locust, 72 × 1 | 1.191 → 1.048 | 1.196 → 1.044 | 1.284 | 3176 → 2838 (−10.6 %) | 334 → 532 (182) | 0.6 |
| cube | 10⁷ | Kathleen 1N, 40 × 1 | 1.070 → 1.044 | 1.071 → 1.040 | 1.097 | 5660 → 5550 (−1.9 %) | 623 → 947 (329) | 3.0 |
| plummer | 10⁷ | Kathleen 1N, 40 × 1 | 1.317 → 1.058 | 1.306 → 1.058 | 1.207 | 16315 → 13217 (−19.0 %) | 1208 → 1859 (664) | 0.2 |
| clusters | 10⁷ | Kathleen 1N, 40 × 1 | 1.128 → 1.038 | 1.114 → 1.054 | 1.307 | 14532 → 13520 (−7.0 %) | 1067 → 1880 (771) | 0.8 |
| cube | 10⁷ | Kathleen 1N, 2 × 20 | 1.009 → 1.011 | 1.000 → 1.000 | 1.000 | 5411 → 5454 (+0.8 %) | 9980 → 11192 (1200) | never |
| plummer | 10⁷ | Kathleen 1N, 2 × 20 | 1.045 → 1.050 | 1.000 → 1.000 | 1.002 | 12862 → 13031 (+1.3 %) | 15545 → 16717 (1208) | never |
| clusters | 10⁷ | Kathleen 1N, 2 × 20 | 1.051 → 1.052 | 1.037 → 1.037 | 1.050 | 13115 → 13081 (−0.3 %) | 16745 → 18092 (1295) | 39.5 |
| cube | 10⁷ | Kathleen 2N, 80 × 1 | 1.077 → 1.030 | 1.076 → 1.020 | 1.094 | 2842 → 2730 (−3.9 %) | 408 → 493 (177) | 0.8 |
| plummer | 10⁷ | Kathleen 2N, 80 × 1 | 1.384 → 1.060 | 1.371 → 1.047 | 1.371 | 8447 → 6527 (−22.7 %) | 674 → 1032 (344) | 0.2 |
| clusters | 10⁷ | Kathleen 2N, 80 × 1 | 1.210 → 1.058 | 1.202 → 1.054 | 1.385 | 7656 → 6712 (−12.3 %) | 579 → 988 (377) | 0.4 |
| cube | 10⁷ | Kathleen 2N, 4 × 20 | 1.016 → 1.020 | 1.000 → 1.000 | 1.001 | 2674 → 2680 (+0.2 %) | 4631 → 5072 (577) | never |
| plummer | 10⁷ | Kathleen 2N, 4 × 20 | 1.064 → 1.041 | 1.002 → 1.002 | 1.004 | 6501 → 6312 (−2.9 %) | 7253 → 7866 (591) | 3.2 |
| clusters | 10⁷ | Kathleen 2N, 4 × 20 | 1.043 → 1.035 | 1.014 → 1.008 | 1.020 | 6510 → 6466 (−0.7 %) | 7629 → 12424 (4724) | 111.2 |
| cube | 10⁷ | Kathleen 4N, 160 × 1 | 1.091 → 1.032 | 1.091 → 1.034 | 1.132 | 1324 → 1261 (−4.8 %) | 299 → 297 (103) | < 1 |
| plummer | 10⁷ | Kathleen 4N, 160 × 1 | 1.507 → 1.085 | 1.522 → 1.087 | 1.481 | 4284 → 3121 (−27.2 %) | 403 → 607 (205) | 0.2 |
| clusters | 10⁷ | Kathleen 4N, 160 × 1 | 1.301 → 1.102 | 1.271 → 1.093 | 1.545 | 3838 → 3280 (−14.5 %) | 351 → 560 (208) | 0.4 |
| cube | 10⁸ | Kathleen 4N, 160 × 1 | 1.063 → 1.057 | 1.062 → 1.037 | 1.058 | 13546 → 13444 (−0.8 %) | 1734 → 2520 (789) | 7.7 |
| plummer | 10⁸ | Kathleen 4N, 160 × 1 | 1.584 → 1.070 | 1.597 → 1.081 | 1.500 | 43757 → 29956 (−31.5 %) | 4162 → 6186 (2027) | 0.1 |
| clusters | 10⁸ | Kathleen 4N, 160 × 1 | 1.242 → 1.063 | 1.255 → 1.071 | 1.619 | 33087 → 28698 (−13.3 %) | 3343 → 5398 (1869) | 0.5 |
| cube | 10⁷ | Kathleen 4N, 8 × 20 | 1.018 → 1.017 | 1.001 → 1.001 | 1.002 | 1328 → 1335 (+0.5 %) | 2118 → 2400 (265) | never |
| plummer | 10⁷ | Kathleen 4N, 8 × 20 | 1.107 → 1.101 | 1.020 → 1.020 | 1.019 | 3339 → 3313 (−0.8 %) | 3447 → 5661 (2223) | 84.2 |
| clusters | 10⁷ | Kathleen 4N, 8 × 20 | 1.065 → 1.016 | 1.031 → 1.026 | 1.051 | 3274 → 3130 (−4.4 %) | 3691 → 6103 (2408) | 16.8 |
| cube | 10⁸ | Kathleen 4N, 8 × 20 | 1.013 → 1.020 | 1.003 → 1.003 | 1.000 | 14331 → 14339 (+0.1 %) | 27545 → 30935 (3201) | never |
| plummer | 10⁸ | Kathleen 4N, 8 × 20 | 1.099 → 1.096 | 1.006 → 1.006 | 1.007 | 32620 → 32579 (−0.1 %) | 45201 → 69987 (24731) | 604.9 |
| clusters | 10⁸ | Kathleen 4N, 8 × 20 | 1.052 → 1.057 | 1.056 → 1.022 | 1.027 | 29739 → 29958 (+0.7 %) | 46206 → 71657 (25225) | never |

The 2 × 20 and 4 × 20 rows are the repeats on the final code (jobs 603978 and 603986);
the first runs of jobs 603930 and 603931, before an unchanged cut kept its plan, took
5.1–10.1 s (2 × 20) and 2.5–4.8 s (4 × 20) for the second cut at the same balance.

Readings:
- **The second cut balances the compute where the first does not**: at one rank per core
  from 40 ranks up (40, 80 and 160 on Kathleen, 72 on locust) the compute's max over mean
  falls from 1.06–1.58 to 1.02–1.10 on every workload and N, and the evaluation's wall
  time by 1–32 %: the Plummer sphere 1.24–1.58 → 1.05–1.09 and 14–32 % faster, the
  clusters 1.13–1.33 → 1.04–1.10 and 7–20 %, the cube 1.06–1.15 → 1.02–1.06 and 1–10 %.
  The modelled max over mean follows the measured one within 0.03 in every row with one
  thread per rank (within 0.02 but for the clusters at N = 10⁷ on 160 ranks under the
  first cut, 1.271 modelled against 1.301 measured).
- **At 160 ranks it reaches the brief's measured criterion** (C5S.4: the compute stages at
  72 and 160 ranks improved against the first cut): 1.091 → 1.032 (cube), 1.507 → 1.085
  (Plummer) and 1.301 → 1.102 (clusters) at N = 10⁷, 1.063 → 1.057, 1.584 → 1.070 and
  1.242 → 1.063 at N = 10⁸; at 72 locust ranks as above.
- **At 8 ranks (locust) and at 2–8 ranks of 20 threads (Kathleen) it buys little or
  nothing** (−4.9 % to +1.3 %): the first cut is within 1.01–1.11 there already, with
  many blocks per rank, and the model, looser with 20 threads (errors up to 9 % on a
  rank), sees little to gain (modelled 1.000–1.056 before, the same or less after). 7 of
  these 18 rows get 0.1–1.3 % slower.
- **Points follow cost**: after the second cut the busiest rank holds 1.06–1.62 times the
  mean points at 40–160 Kathleen and 72 locust ranks (memory per rank follows points,
  scale-out.md §8.2).
- **The second cut costs one to one and a half `Plan::new`** (§6): 31–41 % of the build
  at 40–160 Kathleen ranks. Where it balances, the evaluations pay it back quickly (the
  Plummer sphere and the clusters after 0.1–0.8 evaluations, the cube after 0.5–7.7);
  where it does not, they never do. Where the second cut moves blocks without a gain
  (threaded ranks), its second plan is expensive: 2.2–4.7 s at N = 10⁷ (4 × 20, 8 × 20)
  and 25 s at N = 10⁸ on 8 × 20 (12.5 million points per rank, a serial `Plan::new`);
  where it keeps the first cut, it costs the counting alone (0.3–3.2 s, 7–11 % of those
  builds).

## 5. Counted at 512 and 640 oversubscribed ranks

`tools/scaling/run.sh cut-counts`: never timed. f64 p = 8 `Rotation`, both cuts, the
modelled cost per rank with locust's constants on locust and Kathleen's on Kathleen; N =
10⁶ and 10⁷. Target (decision 3, scale-out.md §6.4): the modelled max over mean ≤ 1.10
with blocks refined by cost, or ≤ 1.12 without that refinement (this task: without).

| Machine, ranks | Workload | N | modelled: first → second | points, second | leaves, second | V pairs, second |
| --- | --- | --- | --- | --- | --- | --- |
| locust, 512 | cube | 10⁶ | 1.268 → **1.124** | 1.470 | 1.500 | 1.157 |
| locust, 512 | cube | 10⁷ | 1.191 → 1.096 | 1.219 | 1.218 | 1.082 |
| locust, 512 | Plummer | 10⁶ | 1.677 → 1.073 | 1.879 | 1.455 | 1.668 |
| locust, 512 | Plummer | 10⁷ | 1.581 → 1.052 | 1.669 | 1.333 | 1.488 |
| locust, 512 | clusters | 10⁶ | 2.786 → 1.078 | 1.488 | 3.210 | 1.690 |
| locust, 512 | clusters | 10⁷ | 1.580 → 1.048 | 1.614 | 1.542 | 1.424 |
| Kathleen, 320 | cube | 10⁶ | 1.241 → 1.084 | 1.399 | 1.406 | 1.105 |
| Kathleen, 320 | cube | 10⁷ | 1.097 → 1.082 | 1.182 | 1.172 | 1.097 |
| Kathleen, 320 | Plummer | 10⁶ | 1.534 → 1.075 | 1.804 | 1.388 | 1.667 |
| Kathleen, 320 | Plummer | 10⁷ | 1.581 → 1.082 | 1.854 | 1.287 | 1.437 |
| Kathleen, 320 | clusters | 10⁶ | 1.767 → 1.085 | 1.486 | 2.825 | 1.614 |
| Kathleen, 320 | clusters | 10⁷ | 1.367 → 1.078 | 1.626 | 1.218 | 1.379 |
| Kathleen, 640 | cube | 10⁶ | 1.215 → 1.054 | 1.500 | 1.523 | 1.093 |
| Kathleen, 640 | cube | 10⁷ | 1.093 → 1.023 | 1.225 | 1.230 | 1.037 |
| Kathleen, 640 | Plummer | 10⁶ | 1.680 → **1.105** | 2.071 | 1.448 | 1.682 |
| Kathleen, 640 | Plummer | 10⁷ | 1.724 → 1.064 | 1.928 | 1.287 | 1.437 |
| Kathleen, 640 | clusters | 10⁶ | 2.673 → 1.080 | 1.629 | 3.618 | 1.709 |
| Kathleen, 640 | clusters | 10⁷ | 1.473 → 1.057 | 1.740 | 1.359 | 1.412 |

Readings:
- **By points the modelled cost's max over mean is 1.09–2.79 at 320–640 ranks; by the
  second cut 1.02–1.12.** Within 1.12 (the signed-off target without refinement by cost)
  in 17 of the 18 rows; above it the cube at N = 10⁶ on 512 locust ranks, 1.124. Within
  1.10 (the target with refinement) in 16; above it that row and the Plummer sphere at
  N = 10⁶ on 640 Kathleen ranks, 1.105.
- **It is the replay of scale-out.md §6.2, built**: every row the design replayed (N = 10⁷
  at 320 and 640 on Kathleen, N = 10⁶ and 10⁷ at 512 on locust but the cube at 10⁶) lies
  within 0.02 of the replay's "by modelled cost" column (for example 1.020, 1.055, 1.053
  there against 1.023, 1.064, 1.057 here at 640 ranks).
- **Where it stays above 1.10: blocks too coarse for the work per rank.** The two rows are
  the smallest problems per rank (N/P = 1,950 and 1,560 points). On the cube at 512 ranks
  a rank owns 64 leaves and 68 boxes on average: its blocks are level-4 boxes of eight
  leaves (about an eighth of a fair share each, O1's refinement by points stops there)
  and single leaves, and the nearest-boundary cut leaves a rank up to one block above
  the mean (O3's bound); the shared ancestors' rows, which no cut moves, add their part
  (9 per rank on average, 14 at most). Finer blocks where the cost is, the refinement by
  cost of scale-out.md §6.3 item 3, is what would lower it (§2, reading 3; §9).
- **Points and leaves follow cost**: after the second cut the busiest rank holds up to
  2.07 times the mean points (the Plummer sphere at 640) and 3.62 times the mean leaves
  (the clusters at N = 10⁶), so its memory per rank (scale-out.md §8.2) and its share of
  the points' redistribution grow with it.

## 6. The build cost of the second cut

Its parts, measured once with temporary clocks on the M3 Max (8 ranks, N = 10⁶, f64 p = 8
`Rotation`, the Plummer sphere; not kept in the code): the second cut's 107 ms are the
second `Plan::new` (71 ms, 66 %), the points per leaf (22 ms, 20 %: two sorts of the
rank's keys and the all-to-all-v), `Octree::repartition` (10 ms, 9 %), the ghost leaves'
counts (3.5 ms) and the work per block (0.4 ms). On the cube there the cut came out the
same as the first and kept its plan (26 ms in all). On Kathleen the cut costs 1.0–1.5
times the first `Plan::new` at 40–80 ranks, and since `Plan::new` is 29–50 % of the build
there, the second cut is 33–41 % of the build with it (`BuildTimings::cut`, §4).
At 160 ranks (job 603954) the second cut takes 103–208 ms at N = 10⁷ and 0.8–2.0 s at
N = 10⁸, **31–37 % of the build**; the evaluation it shortens takes 1.3–44 s, so on the
Plummer sphere and the clusters it is paid back within half an evaluation, on the cube
within one at N = 10⁷ and after eight at N = 10⁸ (the cube's first cut is 1.06 already).

**So the condition the design attached to the default (scale-out.md §6.4: "on … once T7
has measured its build cost at 160 ranks below a quarter of the build") is not met**: the
second cut is at least a second `Plan::new`, which alone is a quarter of the new build
wherever `Plan::new` is a third of the old one, and it is 29–50 % of it here. What
it buys is an evaluation shorter by 1–32 % at 40–160 ranks, paid back within one
evaluation on the adaptive workloads (§4). Decision 3 as signed off has the second cut on
by default, and this branch keeps it so; §9 asks the question again.

## 7. Correctness

Every run bit for bit, the second cut on unless named:
- **M3 Max** (debug unless named; `mpirun` with the loopback flags): the `nd-octree`
  examples `test_mpi_weighted_partition` and `test_mpi_locally_essential` at 1, 2, 3, 4 and
  8 ranks; `mpi_exec` at 1, 2, 4 and 8 ranks (**work-weighted cut**: points moved on every
  rank, the modelled max over mean 1.31 → 1.05, 1.36 → 1.07 and 2.04 → 1.10 at 2, 4 and 8
  ranks); `invariance` at 1, 2, 4 and 8; `.github/scripts/run-mpi-tests.sh` at 2 and 4
  (every executable and example of CI's list, on the final code); the ignored
  `multi_rank` in release at 1, 2, 4 and 8 ranks, 0 values differing in every row;
  `tests/scaling.rs` at 1 and 2 ranks; `RUST_MIN_STACK=8388608 cargo test` and
  `cargo test --workspace` (58 and 68 test runs, every one passed).
- **locust** (final code): `.github/scripts/run-mpi-tests.sh` at 1, 2, 4 and 8 ranks;
  `multi_rank` at 1, 2, 4 and 8 ranks in release, 0 of 3,200,000 values differing in each
  of the 10 rows at 8 ranks; `mpi_exec` with `--features cpu` in release at 2 and 4 ranks,
  every scenario repeated on the CPU runtime (107 device scenarios at 2 ranks: the host
  fallback bit for bit, the device within the FMM bounds); **rank-count invariance at
  512 against 64 ranks** (oversubscribed, N = 10⁷, f64 p = 8 `Rotation`, the cube and the
  Plummer sphere): 0 of 80,000,000 values differing in each.
- **Kathleen**: the correctness job of T2 at 2 nodes (job 603955, `test`): the MPI list at
  2, 4, 8 and 80 ranks, rank-count invariance at 80 against 40, 20 and 10 ranks (the cube
  and the Plummer sphere f64 p = 8, the cube f32 p = 3), the 4 × 20 smoke run; all passed.
- **Kathleen**, 4 nodes (job 603960, `small`): rank-count invariance at 160 ranks against
  80, 40 and 20 (the cube and the Plummer sphere f64 p = 8 `Rotation`, the cube f32
  p = 3 `Dense`, N = 10⁷) and at 640 ranks against 80 (oversubscribed, the cube f64 p =
  8): 0 of 80,000,000 values differing in every check. Together with job 603955 that is
  T2's correctness job, at 2 and at 4 nodes.

## 8. Kathleen jobs and node-hours

| Job | Nodes, QoS | What | Waited | Ran | Node-hours |
| --- | --- | --- | ---: | ---: | ---: |
| 603049 | 1, `singlenode` | build, `calibrate` | 1 s | 9:30 | 0.16 |
| 603930 | 1, `singlenode` | `cut` at 40 × 1 and 2 × 20 | 2 s | 11:58 | 0.20 |
| 603931 | 2, `test` | `cut` at 80 × 1 and 4 × 20 | 5:33 | 5:58 | 0.20 |
| 603955 | 2, `test` | correctness (mpi, invariance, smoke) | 11:54 | 22:20 | 0.74 |
| 603954 | 4, `small` | `cut` at 160 × 1 and 8 × 20 (N = 10⁷, 10⁸), `cut-counts` at 320 and 640 | 2:25:18 | 51:52 | 3.46 |
| 603960 | 4, `small` | correctness: invariance at 160 and 640 | 1:52:15 | 3:44 | 0.25 |
| 603978 | 1, `singlenode` | `cut` at 2 × 20, final code | 1 s | 6:11 | 0.10 |
| 603986 | 2, `test` | `cut` at 4 × 20, final code | 38:32 | 3:32 | 0.12 |
| **total** | | | | | **5.2** |

Within the task's budget of 20 node-hours (README, "Ranks on Kathleen"). Nodes of the
multi-node jobs: 603931 node-c11b-[018-019], 603955 node-c11a-[065,126], 603954
node-c11a-[006,033,040,086], 603960 node-c11b-[009,013-014,022], 603986
node-c11a-[042,071]. Raw outputs stay in Kathleen's `logs/scaling-<job>` and
`logs/fmm-correctness-<job>.out` and locust's `/data/ucahtbe/fmm/logs/t7-*`.

## 9. Questions for sign-off

**Signed off on 2026-10-13 by the user: question 1, option (a); question 2, option (c)**,
both as recommended; recorded in docs/phase5s/README.md, decision 3. The gain rule is
built on this branch (§1, items 1 and 3); §10 checks it. The tables of §4–§6 were measured
before it, where every second cut that moved blocks moved them.

1. **The target (decision 3; C5S.4).** Without refinement by cost the modelled max over
   mean at 320–640 oversubscribed ranks is 1.02–1.12: 17 of 18 rows within 1.12 (the
   target without the refinement) and 16 within 1.10; the cube at N = 10⁶ on 512 locust
   ranks reaches 1.124, the Plummer sphere at N = 10⁶ on 640 Kathleen ranks 1.105 (§5).
   The measured criterion holds: the compute stages at 72 (locust) and 160 (Kathleen)
   ranks improve on every workload (§4). Options:
   - (a) accept this as T7's outcome; **recommended**: the misses are the two smallest
     problems per rank (1,500–2,000 points), where blocks of an eighth of a fair share
     set the bound, and the timed rows at 40–160 ranks reach 1.02–1.10 measured;
   - (b) build the refinement by cost (scale-out.md §6.3, item 3) as its own task before
     T9: split blocks heavier than W/(kP) in cost into their children (nodes of the tree,
     so the leaves stay), re-balance the blocks 2:1 for H's assembly (§3.3), the new
     blocks' weights from their owners (one more gather round); a change of O1's coarse
     tree, which T7's brief excludes, bits unchanged.
2. **The default (decision 3; scale-out.md §6.4).** The design made "on by default" wait
   for a build cost below a quarter of the build at 160 ranks; it is 31–37 % there and
   31–41 % at 40–160 ranks (§6), because a second `Plan::new` is part of it. Where it
   balances, one evaluation repays it; at 2–8 ranks it repays nothing, and on 8 ranks of
   20 threads at N = 10⁸ it costs 25 s (§4). This branch keeps the signed-off default
   (on). Options:
   - (a) on, as signed off (this branch);
   - (b) off, on request (`FmmBuilder::work_cut(true)`);
   - (c) on with a gain rule: cut again only where the gathered weights lower the busiest
     rank's block weight by at least g, say 2 % (a few lines in `Octree::repartition` or
     `Fmm`, deterministic, the same on every rank). On the modelled columns of §4 it keeps
     every row at 40–160 ranks with one thread per rank (each gains 2.4 % or more) and
     skips 16 of the 18 rows at 2–8 ranks, keeping the clusters on 8 locust ranks (4.9 %
     faster) and on 8 × 20 at N = 10⁸ (0.7 % slower, the threaded model's limit).
     **Recommended**, as a follow-up commit if accepted.

## 10. The gain rule, checked

After the sign-off (§9), with the 2 % rule built, on locust on 2026-10-13: the timed `cut`
sweep again (locust's constants, as in §4), and every Kathleen configuration of §4
replayed with Kathleen's constants, counted only, f64 p = 8 `Rotation`, oversubscribed
above 72 ranks. Whether the rule moves blocks depends only on the tree, the rank count
and the constants, never on where the ranks run, so the replay gives Kathleen's
decisions (the replayed modelled max over mean equals job 603954's to three digits:
1.037, 1.081 and 1.071 at 160 ranks and N = 10⁸). `recut` is `Fmm::recut`.

| Configuration | cube | Plummer | clusters |
| --- | --- | --- | --- |
| locust 8 ranks, N = 10⁶ | kept (16 ms) | kept (19 ms) | **cut** (131 ms) |
| locust 8 ranks, N = 10⁷ | kept (200 ms) | kept (213 ms) | kept (216 ms) |
| locust 72 ranks, N = 10⁶ and 10⁷ | **cut** | **cut** | **cut** |
| Kathleen 2 and 4 ranks (× 20 threads), N = 10⁷ | kept | kept | kept |
| Kathleen 8 ranks (× 20), N = 10⁷ | kept | kept | kept |
| Kathleen 8 ranks (× 20), N = 10⁸ | kept | kept | **cut** |
| Kathleen 40, 80 and 160 ranks, N = 10⁷ | **cut** | **cut** | **cut** |
| Kathleen 160 ranks, N = 10⁸ | **cut** | **cut** | **cut** |

("kept": the first cut stays, with its plan; in brackets the second cut's time on
locust, the counting alone where it is kept.) Readings:
- **As predicted in §9**: every configuration with one rank per core from 40 ranks up
  cuts again, as in §4 and §5; at 2–8 ranks 16 of the 18 configurations keep the first
  cut, the two that cut being those §9 named (the clusters on 8 locust ranks at N = 10⁶,
  4.9 % faster in §4, and on 8 × 20 at N = 10⁸).
- **Where it keeps the first cut, the cost is the counting alone**: on 8 locust ranks
  16–216 ms against 71–1,367 ms before (§4), 8–12 % of the build; the evaluations are
  those of the first cut (wall time within ±1.3 % of this sweep's first-cut rows).
- The rows that cut give §4's balance and times again (72 locust ranks: compute 1.03–1.07
  after the cut against 1.07–1.33 before).
- Bits: the rule only chooses between two cuts that both give the one-rank output; the
  checks of §7 ran again on the M3 Max with it (CI's MPI list at 1, 2, 4 and 8 ranks, 44
  of 44 passed; `cargo test --workspace`, the clippy and documentation checks), and
  `test_mpi_weighted_partition` checks the rule's decision against brute force at 1, 2,
  3, 4 and 8 ranks.


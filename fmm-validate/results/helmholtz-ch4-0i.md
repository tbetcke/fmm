# Helmholtz CH4.0i: the coarse RCR levels

Task: `docs/helmholtz/tasks/CH4.0i.md`. Roadmap: H4, CH4.0i (priority list item 6), added on
2026-11-16 by CH4.0e's sign-off (`fmm-validate/results/helmholtz-ch4-0e.md` §3.2, §7 question
1). No proposal §10 row; the acceptance is the roadmap's *Done when*: level 2 of W1 at kD = 500
at ≥ 70 % of linear on the node, or the report says why not; deterministic and
rank-invariant.

## 0. Summary

- **What changed** (§1): a Rotation (RCR) M2L level call can run by pairs. The level's V
  pairs, in the plan's order and in chunks that fit a buffer, each write their contribution
  (the terms `HelmholtzTables::m2l` adds to the local) into a slot of their own on the pool,
  ordered by polar angle and distance; then every target adds its slots in its row's order.
  Each value of a local receives the same terms, computed by the same operations and added
  in the same order as target by target, so the output is `main`'s bit for bit at every
  split, thread count, buffer and rank count. The scratch also keeps the y-blocks it formed
  last, so pairs of one angle form them once.
- **Level 2 of W1 at kD = 500** (§2), by pairs (the default now) against by boxes:
  - one and two ranks per node: 0.55–0.69× on both machines;
  - 4 and 8 ranks per node: 0.23–0.40×;
  - one core: 0.56–0.60×, from the class order and the kept blocks.
- **The acceptance** (§2, §3), at the node's best split:
  - Against CH4.0e's measure (one core target by target): met on Kathleen, 0.89–0.91 at
    2 × 20, and not on locust, 0.75–0.78 at 1 × 72.
  - Against this task's own one core (by pairs, 0.56–0.60× faster): not met, 0.50–0.54 on
    Kathleen and 0.44–0.45 on locust.
  - The remaining loss is per-pair memory traffic at p = 131 (§3). Storing level 2's blocks
    lifts locust to 0.58–0.69 and Kathleen hardly at all, and costs 601 MB per rank; CH4.0e's
    budget stands.
- **The evaluation of W1 at kD = 500** (§2): 0.82–0.88× of by boxes on Kathleen at 1 × 40
  and 2 × 20, 0.67–0.75× at 4 × 10, 0.50–0.59× at 8 × 5. On locust 0.87–0.91×, 0.86–0.89×,
  0.69–0.76× and 0.53–0.61× at 1 × 72 … 8 × 9.
- **The pair against S6** (§4), on the same Grace core: ours 1.95× S6's prototype at p = 72
  and 2.41× at p = 131 with blocks stored, 1.80× and 1.79× in the pair split's order. The rest
  lies in the pair's loop-carried sum chains, whose order `rcr` and the tables keep in step:
  the follow-up CH4.0l.
- **Auto's rule** (§5): by pairs on levels of degree p ≥ 48 at every thread count, and of
  p ≥ 30 where the level's targets with a V row are fewer than 4 × the threads; by boxes
  below. By pairs loses 1.1–3.8× on levels of p = 7–15 (the overhead of two passes over
  cheap pairs) and is about even at p = 18–24. Under the rule, CH3.6's matrix at one rank per
  node runs 0.974× by boxes on locust (geometric mean over 57 rows, range 0.67–1.04),
  {K_RULE_SUMMARY}. Every checksum of every run equals CH4.0h's.

## 1. What changed

`nd-fmm-tables` (`helmholtz` only; Laplace's tables untouched):
- `HelmholtzTables::m2l_contribution` writes the pair's contribution, steps 1–3 of
  `translate` unchanged, with step 3 storing in place of adding. `add_m2l_contribution` adds
  it. The two in turn are `m2l` bit for bit (`to_bits`, so −0 and +0 apart).
  `m2l_contribution_len` gives the slot's length.
- `HelmholtzTables::m2l_classes(d)` gives an offset's (polar angle, distance) classes.
- `HelmholtzScratch` keeps the per-call y-blocks it formed last, with their degree and
  (cos θ, sin θ), and forms them again only for another degree or angle: the same function
  of the same inputs.

`nd-fmm-exec` (`helmholtz` only; `driver.rs`, Laplace's operator, the plan and the octree
untouched):
- `RcrSplit::{Auto, Boxes, Pairs}`, `rcr_by_pairs`, `RCR_PAIR_MIN_DEGREE` = 48,
  `RCR_PAIR_FEW_DEGREE` = 30, `RCR_PAIR_SHARE` = 4.
- `HelmholtzOperator::{with_rcr_split, with_rcr_buffer, with_rcr_levels, rcr_counts, …}`.
- `HelmholtzFmmBuilder::{rcr_split, rcr_budget}`: by default `Auto` within
  `DEFAULT_RCR_NODE_BUDGET` = 1 GiB per node over the ranks per node.
- The level call itself is the private `operator_rcr.rs`. A chunk is a range of consecutive
  pairs in row order, at least one per thread. A row may span two chunks; its pairs still add
  in its order.

`nd-fmm-validate`:
- `Case::rcr_split`, `Row::{rcr_split, rcr_by_pairs}` and `--rcr-split` of
  `helmholtz_accuracy`; `--levels`' column "ran" says `rotation by pairs`.
- The `helmholtz_rcr_pair` example, which times the pair on one core with blocks stored,
  formed per call and formed per call in class order, and the box path against the pair path.
- `tools/helmholtz/ch4-0i-runs.sh` (stages `levels`, `stored`, `matrix`, `pairs`, `rotation`,
  `crossover`, `pair`) and `ch4-0i-tables.py`.

Tests:
- `nd-fmm-tables` `tests/helmholtz/contribution.rs`: contribution then add against `m2l` on
  every offset of two levels, one and two planes, one and two vectors, real and complex k,
  stored and per-call blocks and raised degrees. Also a row computed in reverse and added in
  order, and kept blocks against fresh ones in three orders, all bit for bit.
- `nd-fmm-exec` `tests/helmholtz/rcr_split.rs`: the rule.
- `tests/helmholtz_fmm.rs` (`check_rcr_split`, one MPI test): by pairs at 1, 2 and 3 threads,
  at the default buffer and at its floor, Auto at 2 and 20 threads, one and two vectors,
  Rotation and the library's Auto, against by boxes bit for bit. On P ranks also against the
  one-rank FMM; passed at 1, 2 and 4 ranks.
- `nd-fmm-validate` `tests/helmholtz_accuracy.rs` (`check_rcr_split`): the checksums.

## 2. Level 2 of W1 at kD = 500 (the acceptance)

`ch4-0i-runs.sh levels`: W1 at kD = 500 at the defaults (`--m2l auto-pw --near-v cost
--near-wx cost`), `--levels`, two evaluations (one on one core), the last one's M2L per level.
Each split runs `--rcr-split boxes,auto,pairs` (one core `boxes,pairs`) in one process, in
order a and again reversed (b); the times are the means of a and b. "auto" is the rule
before §5's change (by pairs where targets < 4 × threads), the same as pairs on level 2 and on
locust's level 3. Under §5's rule `Auto` runs both levels by pairs at every split, the column
"pairs". "Linear" is the one-core time over (ranks × threads × the time). It is given against
the one core by boxes (CH4.0e's measure, the only one before this task) and by pairs (this
task's one core, the best).

Kathleen (jobs 634393, node-c11a-065, 2 × 20 Cascade Lake cores):

Level 2:

| ε | ranks × threads | boxes s | auto s | pairs s | pairs ÷ boxes | linear (boxes, one core by boxes) | linear (pairs, one core by boxes) | linear (pairs, one core by pairs) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 1 × 1 | 41.907 | – | 23.471 | 0.56 | – | – | – |
| 1e-3 | 1 × 40 | 2.299 | 1.549 | 1.312 | 0.57 | 0.46 | 0.80 | 0.45 |
| 1e-3 | 2 × 20 | 2.088 | 1.125 | 1.179 | 0.56 | 0.50 | 0.89 | 0.50 |
| 1e-3 | 4 × 10 | 3.603 | 1.158 | 1.285 | 0.36 | 0.29 | 0.82 | 0.46 |
| 1e-3 | 8 × 5 | 6.112 | 1.545 | 1.559 | 0.26 | 0.17 | 0.67 | 0.38 |
| 1e-6 | 1 × 1 | 49.720 | – | 29.667 | 0.60 | – | – | – |
| 1e-6 | 1 × 40 | 2.884 | 1.859 | 1.829 | 0.63 | 0.43 | 0.68 | 0.41 |
| 1e-6 | 2 × 20 | 2.491 | 1.380 | 1.373 | 0.55 | 0.50 | 0.91 | 0.54 |
| 1e-6 | 4 × 10 | 4.357 | 1.465 | 1.473 | 0.34 | 0.29 | 0.84 | 0.50 |
| 1e-6 | 8 × 5 | 7.450 | 1.801 | 1.915 | 0.26 | 0.17 | 0.65 | 0.39 |
| 1e-8 | 1 × 1 | 54.481 | – | 32.180 | 0.59 | – | – | – |
| 1e-8 | 1 × 40 | 3.094 | 2.053 | 1.987 | 0.64 | 0.44 | 0.69 | 0.40 |
| 1e-8 | 2 × 20 | 2.776 | 1.599 | 1.526 | 0.55 | 0.49 | 0.89 | 0.53 |
| 1e-8 | 4 × 10 | 4.908 | 1.560 | 1.631 | 0.33 | 0.28 | 0.84 | 0.49 |
| 1e-8 | 8 × 5 | 8.599 | 2.088 | 1.944 | 0.23 | 0.16 | 0.70 | 0.41 |

Level 3:

| ε | ranks × threads | boxes s | auto s | pairs s | pairs ÷ boxes | linear (boxes, one core by boxes) | linear (pairs, one core by boxes) | linear (pairs, one core by pairs) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 1 × 1 | 19.659 | – | 16.197 | 0.82 | – | – | – |
| 1e-3 | 1 × 40 | 0.764 | 0.844 | 0.651 | 0.85 | 0.64 | 0.76 | 0.62 |
| 1e-3 | 2 × 20 | 0.734 | 0.761 | 0.627 | 0.85 | 0.67 | 0.78 | 0.65 |
| 1e-3 | 4 × 10 | 0.728 | 0.779 | 0.647 | 0.89 | 0.68 | 0.76 | 0.63 |
| 1e-3 | 8 × 5 | 1.563 | 1.650 | 0.764 | 0.49 | 0.31 | 0.64 | 0.53 |
| 1e-6 | 1 × 1 | 26.230 | – | 20.873 | 0.80 | – | – | – |
| 1e-6 | 1 × 40 | 1.114 | 1.130 | 0.920 | 0.83 | 0.59 | 0.71 | 0.57 |
| 1e-6 | 2 × 20 | 0.951 | 1.002 | 0.880 | 0.93 | 0.69 | 0.75 | 0.59 |
| 1e-6 | 4 × 10 | 0.987 | 1.042 | 0.923 | 0.94 | 0.66 | 0.71 | 0.57 |
| 1e-6 | 8 × 5 | 2.069 | 2.125 | 1.036 | 0.50 | 0.32 | 0.63 | 0.50 |
| 1e-8 | 1 × 1 | 31.215 | – | 24.542 | 0.79 | – | – | – |
| 1e-8 | 1 × 40 | 1.381 | 1.426 | 1.157 | 0.84 | 0.57 | 0.67 | 0.53 |
| 1e-8 | 2 × 20 | 1.186 | 1.199 | 1.054 | 0.89 | 0.66 | 0.74 | 0.58 |
| 1e-8 | 4 × 10 | 1.189 | 1.232 | 1.118 | 0.94 | 0.66 | 0.70 | 0.55 |
| 1e-8 | 8 × 5 | 2.505 | 2.510 | 1.113 | 0.44 | 0.31 | 0.70 | 0.55 |

The evaluation (s):

| ε | ranks × threads | boxes | auto | pairs | auto ÷ boxes |
| --- | --- | ---: | ---: | ---: | ---: |
| 1e-3 | 1 × 1 | 128.007 | nan | 106.147 | nan |
| 1e-3 | 1 × 40 | 5.936 | 5.355 | 4.843 | 0.90 |
| 1e-3 | 2 × 20 | 5.575 | 4.657 | 4.585 | 0.84 |
| 1e-3 | 4 × 10 | 6.851 | 4.712 | 4.679 | 0.69 |
| 1e-3 | 8 × 5 | 10.385 | 5.978 | 5.222 | 0.58 |
| 1e-6 | 1 × 1 | 156.011 | nan | 130.454 | nan |
| 1e-6 | 1 × 40 | 7.374 | 6.374 | 6.113 | 0.86 |
| 1e-6 | 2 × 20 | 6.707 | 5.653 | 5.529 | 0.84 |
| 1e-6 | 4 × 10 | 8.392 | 5.755 | 5.595 | 0.69 |
| 1e-6 | 8 × 5 | 12.778 | 7.237 | 6.427 | 0.57 |
| 1e-8 | 1 × 1 | 256.675 | nan | 227.591 | nan |
| 1e-8 | 1 × 40 | 11.248 | 10.204 | 9.906 | 0.91 |
| 1e-8 | 2 × 20 | 10.546 | 9.434 | 9.194 | 0.89 |
| 1e-8 | 4 × 10 | 12.555 | 9.572 | 9.456 | 0.76 |
| 1e-8 | 8 × 5 | 17.652 | 11.642 | 10.415 | 0.66 |

locust (GH200, 72 Neoverse V2 cores):

Level 2:

| ε | ranks × threads | boxes s | auto s | pairs s | pairs ÷ boxes | linear (boxes, one core by boxes) | linear (pairs, one core by boxes) | linear (pairs, one core by pairs) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 1 × 1 | 21.831 | – | 12.629 | 0.58 | – | – | – |
| 1e-3 | 1 × 72 | 0.611 | 0.391 | 0.388 | 0.64 | 0.50 | 0.78 | 0.45 |
| 1e-3 | 2 × 36 | 0.637 | 0.415 | 0.418 | 0.66 | 0.48 | 0.72 | 0.42 |
| 1e-3 | 4 × 18 | 1.137 | 0.462 | 0.449 | 0.40 | 0.27 | 0.67 | 0.39 |
| 1e-3 | 8 × 9 | 2.002 | 0.522 | 0.518 | 0.26 | 0.15 | 0.59 | 0.34 |
| 1e-6 | 1 × 1 | 26.687 | – | 15.511 | 0.58 | – | – | – |
| 1e-6 | 1 × 72 | 0.748 | 0.495 | 0.493 | 0.66 | 0.50 | 0.75 | 0.44 |
| 1e-6 | 2 × 36 | 0.785 | 0.511 | 0.519 | 0.66 | 0.47 | 0.71 | 0.41 |
| 1e-6 | 4 × 18 | 1.414 | 0.546 | 0.548 | 0.39 | 0.26 | 0.68 | 0.39 |
| 1e-6 | 8 × 9 | 2.495 | 0.643 | 0.641 | 0.26 | 0.15 | 0.58 | 0.34 |
| 1e-8 | 1 × 1 | 29.081 | – | 16.925 | 0.58 | – | – | – |
| 1e-8 | 1 × 72 | 0.820 | 0.539 | 0.536 | 0.65 | 0.49 | 0.75 | 0.44 |
| 1e-8 | 2 × 36 | 0.864 | 0.567 | 0.594 | 0.69 | 0.47 | 0.68 | 0.40 |
| 1e-8 | 4 × 18 | 1.555 | 0.605 | 0.625 | 0.40 | 0.26 | 0.65 | 0.38 |
| 1e-8 | 8 × 9 | 2.772 | 0.704 | 0.708 | 0.26 | 0.15 | 0.57 | 0.33 |

Level 3:

| ε | ranks × threads | boxes s | auto s | pairs s | pairs ÷ boxes | linear (boxes, one core by boxes) | linear (pairs, one core by boxes) | linear (pairs, one core by pairs) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 1 × 1 | 10.153 | – | 9.476 | 0.93 | – | – | – |
| 1e-3 | 1 × 72 | 0.177 | 0.150 | 0.150 | 0.85 | 0.80 | 0.94 | 0.87 |
| 1e-3 | 2 × 36 | 0.200 | 0.155 | 0.153 | 0.77 | 0.71 | 0.92 | 0.86 |
| 1e-3 | 4 × 18 | 0.225 | 0.158 | 0.158 | 0.70 | 0.63 | 0.89 | 0.83 |
| 1e-3 | 8 × 9 | 0.297 | 0.171 | 0.170 | 0.57 | 0.47 | 0.83 | 0.77 |
| 1e-6 | 1 × 1 | 13.884 | – | 12.313 | 0.89 | – | – | – |
| 1e-6 | 1 × 72 | 0.232 | 0.195 | 0.195 | 0.84 | 0.83 | 0.99 | 0.88 |
| 1e-6 | 2 × 36 | 0.277 | 0.201 | 0.215 | 0.78 | 0.70 | 0.90 | 0.80 |
| 1e-6 | 4 × 18 | 0.286 | 0.237 | 0.217 | 0.76 | 0.68 | 0.89 | 0.79 |
| 1e-6 | 8 × 9 | 0.362 | 0.241 | 0.242 | 0.67 | 0.53 | 0.80 | 0.71 |
| 1e-8 | 1 × 1 | 16.871 | – | 14.443 | 0.86 | – | – | – |
| 1e-8 | 1 × 72 | 0.290 | 0.234 | 0.231 | 0.80 | 0.81 | 1.01 | 0.87 |
| 1e-8 | 2 × 36 | 0.332 | 0.260 | 0.242 | 0.73 | 0.71 | 0.97 | 0.83 |
| 1e-8 | 4 × 18 | 0.355 | 0.291 | 0.272 | 0.77 | 0.66 | 0.86 | 0.74 |
| 1e-8 | 8 × 9 | 0.462 | 0.307 | 0.333 | 0.72 | 0.51 | 0.70 | 0.60 |

The evaluation (s):

| ε | ranks × threads | boxes | auto | pairs | auto ÷ boxes |
| --- | --- | ---: | ---: | ---: | ---: |
| 1e-3 | 1 × 1 | 82.722 | nan | 72.951 | nan |
| 1e-3 | 1 × 72 | 1.761 | 1.514 | 1.609 | 0.86 |
| 1e-3 | 2 × 36 | 1.782 | 1.534 | 1.579 | 0.86 |
| 1e-3 | 4 × 18 | 2.240 | 1.538 | 1.550 | 0.69 |
| 1e-3 | 8 × 9 | 3.094 | 1.640 | 1.639 | 0.53 |
| 1e-6 | 1 × 1 | 112.805 | nan | 100.034 | nan |
| 1e-6 | 1 × 72 | 2.312 | 2.018 | 2.016 | 0.87 |
| 1e-6 | 2 × 36 | 2.371 | 2.021 | 2.038 | 0.85 |
| 1e-6 | 4 × 18 | 2.921 | 2.058 | 2.045 | 0.70 |
| 1e-6 | 8 × 9 | 3.966 | 2.189 | 2.192 | 0.55 |
| 1e-8 | 1 × 1 | 164.973 | nan | 150.361 | nan |
| 1e-8 | 1 × 72 | 3.152 | 2.823 | 2.822 | 0.90 |
| 1e-8 | 2 × 36 | 3.236 | 2.854 | 2.873 | 0.88 |
| 1e-8 | 4 × 18 | 3.857 | 2.907 | 2.922 | 0.75 |
| 1e-8 | 8 × 9 | 5.004 | 3.058 | 3.073 | 0.61 |

Level 2 has 64 boxes at p = 123, 131 and 135 (10⁻³, 10⁻⁶, 10⁻⁸) and 2,504 V pairs; level 3
has 512 boxes at p = 66–76 and 12,216 pairs.

By pairs, level 2 at one and two ranks per node takes 0.55–0.69× its time by boxes, and at 4 and 8
ranks per node 0.23–0.40×, where by boxes a rank's 8–16 targets left most of its threads
idle. Level 3 takes 0.69–0.94× by pairs (0.44–0.50× at 8 × 5 on Kathleen). On one core both levels run faster by pairs
(0.56–0.60× and 0.79–0.94×): in class order a thread's consecutive pairs share their y-blocks,
which the scratch keeps where they are formed per call (level 2's, 601 MB, exceed the block
budget), and their coaxial table (§4).

Read against CH4.0e's measure, the acceptance is met on Kathleen: at the best split (2 × 20)
level 2 runs at 0.89–0.91 of linear. On locust it reaches 0.75–0.78 at 1 × 72. Read against
this task's one core, which is itself 0.56–0.60× the old one, it is not met anywhere:
0.50–0.54 at Kathleen's best split, 0.44–0.45 on locust. §3 says why.

## 3. Why level 2 stays below 70 % of its own one core

On one core in class order a pair at p = 131 takes 6.2 ms on locust (15.5 s over 2,504 pairs
at 10⁻⁶). On 72 threads the same pairs take 14.2 thread-ms each, 2.3× as much. Level 3 at
p = 72 takes 1.0 ms per pair on one core and 1.1 thread-ms on 72 threads.

The difference is the data per pair. At p = 131 a pair reads:
- one coaxial table of its distance class, Σ_m (p + 1 − m)² complex values ≈ 12 MB;
- its angle's y-blocks twice, 12 MB.

Each is beyond a core's L2 (1 MB on Grace, 1 MB on Cascade Lake), and a few of them fill an
L3 (114 MB on Grace, 27.5 MB per Cascade Lake socket). On one core in class order the same
table and blocks stay in L3 for many consecutive pairs. With every thread on a different
class, they come from memory. At p = 72 the table is about 2 MB and the blocks as much, and
the parallel cost per pair stays near the serial one.

The two levers measured:
- **Fewer block formations per task** (one run of consecutive slots per task, half a thread's
  share, in place of rayon's splitting): no gain on locust (level 2 0.490 against 0.495 s at
  1 × 72, 10⁻⁶; not kept).
- **Level 2's blocks stored** (`ch4-0i-runs.sh stored`, `--rcr-split pairs`, the block
  budget 2 GiB against the default 512 MiB, so every angle's blocks are shared and none is
  formed). The tables below; "linear" against the one core by pairs at the same budget.

Kathleen (job 634402):

| ε | ranks × threads | level 2 s: 512 MiB, 2 GiB | linear (one core by pairs, same budget) | RCR tables s | build s | evaluate s | build + evaluate ÷ |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| 1e-3 | 1 × 1 | 23.617, 22.904 | 1.00, 1.00 | 2.606, 2.987 | 5.056, 5.434 | 106.349, 105.632 | 1.00 |
| 1e-3 | 1 × 40 | 1.218, 1.148 | 0.48, 0.50 | 0.212, 0.214 | 2.562, 2.566 | 4.715, 4.630 | 0.99 |
| 1e-3 | 2 × 20 | 1.148, 1.013 | 0.51, 0.57 | 0.233, 0.264 | 1.574, 1.589 | 4.560, 4.412 | 0.98 |
| 1e-3 | 4 × 10 | 1.310, 1.251 | 0.45, 0.46 | 0.413, 0.489 | 1.179, 1.261 | 4.723, 4.649 | 1.00 |
| 1e-3 | 8 × 5 | 1.542, 1.500 | 0.38, 0.38 | 1.183, 2.023 | 1.706, 2.523 | 5.196, 5.243 | 1.13 |
| 1e-6 | 1 × 1 | 29.169, 28.887 | 1.00, 1.00 | 3.266, 3.712 | 5.703, 6.155 | 129.844, 129.529 | 1.00 |
| 1e-6 | 1 × 40 | 1.705, 1.624 | 0.43, 0.44 | 0.276, 0.309 | 2.637, 2.673 | 6.049, 5.966 | 0.99 |
| 1e-6 | 2 × 20 | 1.483, 1.217 | 0.49, 0.59 | 0.278, 0.333 | 1.619, 1.669 | 5.564, 5.344 | 0.98 |
| 1e-6 | 4 × 10 | 1.400, 1.280 | 0.52, 0.56 | 0.526, 0.595 | 1.294, 1.363 | 5.594, 5.518 | 1.00 |
| 1e-6 | 8 × 5 | 1.739, 1.683 | 0.42, 0.43 | 1.432, 2.525 | 1.917, 3.026 | 6.431, 6.281 | 1.11 |
| 1e-8 | 1 × 1 | 32.474, 31.770 | 1.00, 1.00 | 3.640, 4.115 | 6.175, 6.652 | 228.143, 227.249 | 1.00 |
| 1e-8 | 1 × 40 | 1.888, 1.764 | 0.43, 0.45 | 0.333, 0.341 | 2.740, 2.744 | 9.771, 9.659 | 0.99 |
| 1e-8 | 2 × 20 | 1.539, 1.376 | 0.53, 0.58 | 0.309, 0.375 | 1.690, 1.761 | 9.220, 9.095 | 1.00 |
| 1e-8 | 4 × 10 | 1.558, 1.434 | 0.52, 0.55 | 0.573, 0.692 | 1.389, 1.514 | 9.449, 9.320 | 1.00 |
| 1e-8 | 8 × 5 | 2.128, 2.056 | 0.38, 0.39 | 1.608, 2.836 | 2.144, 3.373 | 10.639, 10.441 | 1.08 |

locust:

| ε | ranks × threads | level 2 s: 512 MiB, 2 GiB | linear (one core by pairs, same budget) | RCR tables s | build s | evaluate s | build + evaluate ÷ |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| 1e-3 | 1 × 1 | 12.670, 12.589 | 1.00, 1.00 | 1.490, 1.687 | 4.217, 4.389 | 73.309, 72.890 | 1.00 |
| 1e-3 | 1 × 72 | 0.392, 0.252 | 0.45, 0.69 | 0.090, 0.093 | 2.720, 2.702 | 1.609, 1.484 | 0.97 |
| 1e-3 | 2 × 36 | 0.412, 0.300 | 0.43, 0.58 | 0.093, 0.101 | 1.558, 1.571 | 1.566, 1.454 | 0.97 |
| 1e-3 | 4 × 18 | 0.457, 0.360 | 0.39, 0.49 | 0.142, 0.146 | 0.899, 0.905 | 1.557, 1.462 | 0.96 |
| 1e-3 | 8 × 9 | 0.512, 0.446 | 0.34, 0.39 | 0.229, 0.399 | 0.647, 0.817 | 1.645, 1.573 | 1.04 |
| 1e-6 | 1 × 1 | 15.547, 15.507 | 1.00, 1.00 | 1.847, 2.087 | 4.630, 4.856 | 99.981, 99.909 | 1.00 |
| 1e-6 | 1 × 72 | 0.488, 0.327 | 0.44, 0.66 | 0.102, 0.103 | 2.750, 2.712 | 2.018, 1.854 | 0.96 |
| 1e-6 | 2 × 36 | 0.563, 0.386 | 0.38, 0.56 | 0.107, 0.117 | 1.582, 1.597 | 2.080, 1.913 | 0.96 |
| 1e-6 | 4 × 18 | 0.566, 0.460 | 0.38, 0.47 | 0.146, 0.176 | 0.901, 0.925 | 2.077, 1.966 | 0.97 |
| 1e-6 | 8 × 9 | 0.644, 0.570 | 0.34, 0.38 | 0.275, 0.303 | 0.688, 0.711 | 2.195, 2.139 | 0.99 |
| 1e-8 | 1 × 1 | 17.237, 17.007 | 1.00, 1.00 | 2.023, 2.311 | 3.681, 3.990 | 150.475, 150.308 | 1.00 |
| 1e-8 | 1 × 72 | 0.539, 0.409 | 0.44, 0.58 | 0.113, 0.120 | 1.675, 1.672 | 2.821, 2.704 | 0.97 |
| 1e-8 | 2 × 36 | 0.571, 0.438 | 0.42, 0.54 | 0.120, 0.135 | 1.081, 1.097 | 2.844, 2.726 | 0.97 |
| 1e-8 | 4 × 18 | 0.612, 0.508 | 0.39, 0.46 | 0.175, 0.212 | 0.699, 0.734 | 2.912, 2.801 | 0.98 |
| 1e-8 | 8 × 9 | 0.707, 0.691 | 0.34, 0.34 | 0.289, 0.335 | 0.606, 0.653 | 3.102, 3.037 | 1.00 |

Stored blocks lift locust's level 2 to 0.58–0.69 at one rank per node, and Kathleen's by
0.01–0.10 (0.44–0.50 at 1 × 40, 0.57–0.59 at 2 × 20). They cost 601 MB per rank, and at 8 ranks per node
on Kathleen 0.8–1.2 s more build (each rank builds them on 5 threads): build plus evaluation
0.96–1.00× at one to four ranks per node, 1.08–1.13× at 8 × 5. So CH4.0e's block budget
(512 MiB, signed off on #220) stands, and stored blocks are no general fix.

What would lift the parallel level is fewer bytes per pair: one coaxial row applied to every
pair of a class at once (each pair's chain unchanged, so the same bits), reading each table
once per batch rather than once per pair. With the vectorised chains of §4 that is the
follow-up CH4.0l.

## 4. The pair against S6

`helmholtz_rcr_pair` on one core (`taskset -c 2` on locust; one bound core on Kathleen), five
passes over the 316 offsets of one level, the median per pair, ms:

Kathleen:

Machine: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz, 40 physical, 1 logical; target x86_64-linux, release build; one thread; median of 5 passes over the 316 offsets.
| p | k̂ | planes | stored ms | per call ms | by class ms | per call ÷ stored | by class ÷ stored | y-blocks ms | y-blocks ÷ per call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 72 | 31.25 | 1 | 1.854 | 2.696 | 1.669 | 1.45 | 0.90 | 1.192 | 0.44 |
| 72 | 31.25 | 2 | 2.309 | 3.086 | 2.064 | 1.34 | 0.89 | 1.192 | 0.39 |
| 131 | 62.5 | 1 | 11.571 | 16.182 | 10.330 | 1.40 | 0.89 | 6.637 | 0.41 |
| 131 | 62.5 | 2 | 14.901 | 18.452 | 12.610 | 1.24 | 0.85 | 6.637 | 0.36 |

locust:

Machine: Neoverse-V2 (implementer 0x41, part 0xd4f), 72 physical, 1 logical; target aarch64-linux, release build; one thread; median of 5 passes over the 316 offsets.
| p | k̂ | planes | stored ms | per call ms | by class ms | per call ÷ stored | by class ÷ stored | y-blocks ms | y-blocks ÷ per call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 72 | 31.25 | 1 | 0.871 | 1.387 | 0.834 | 1.59 | 0.96 | 0.647 | 0.47 |
| 72 | 31.25 | 2 | 1.182 | 1.680 | 1.092 | 1.42 | 0.92 | 0.647 | 0.39 |
| 131 | 62.5 | 1 | 6.862 | 9.059 | 5.497 | 1.32 | 0.80 | 3.643 | 0.40 |
| 131 | 62.5 | 2 | 9.250 | 10.361 | 6.891 | 1.12 | 0.75 | 3.643 | 0.35 |

S6's prototype (`spikes/diagonal`, `examples/times.rs`, S7's RCR with its rotations built
once outside the timing, one complex expansion) on the same locust core, `taskset -c 2`, seven
repetitions: 0.606 ms at p = 72, 3.33 ms at p = 128 (its report: 3.02 ms) and 3.84 ms at
p = 131.

So per pair, against S6 on the same core, with two planes (S6's expansion is complex):

| p | stored | by class (the pair split's order) |
| ---: | ---: | ---: |
| 72 | 1.95× | 1.80× |
| 131 | 2.41× | 1.79× |

CH4.0e put this at 1.9× and 3.3× (2.5× stored), from the FMM's level times.

Forming one angle's y-blocks costs 35–47 % of a per-call pair on both machines (`y_blocks`
alone: 0.65 ms at p = 72 and 3.64 ms at p = 131 on locust). The class order saves it, and also
reuses the coaxial table: by class is 0.75–0.96× even the stored-blocks pair.

What remains is in the three steps' sum chains (`sum = sum + e · x`, one rounding each, in
`rcr`'s order). They are latency-bound on one core, about 12 M multiply-adds per pair at
p = 131. A vectorised form must change that order, so the Helmholtz outputs change in their
last bits, and `rcr` and `HelmholtzTables` change together (`nd-fmm-tables`' rule "keep the
two in step"). That is a task of its own (CH4.0l), not this one's.

The box path against the pair path on one core (`crossover`; locust; the level's blocks
stored within the default budget, so per call at p = 131 only):

locust:

Machine: Neoverse-V2 (implementer 0x41, part 0xd4f), 72 physical, 1 logical; target aarch64-linux, release build; one thread; median of 5 passes over the 316 offsets.
| p | k̂ | planes | stored ms | per call ms | by class ms | per call ÷ stored | by class ÷ stored | y-blocks ms | y-blocks ÷ per call | boxes ms | pairs ms | pairs ÷ boxes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 1 | 0.002 | 0.004 | 0.002 | 1.96 | 1.16 | 0.002 | 0.46 | 0.002 | 0.002 | 1.02 |
| 8 | 1 | 2 | 0.002 | 0.004 | 0.003 | 1.76 | 1.12 | 0.002 | 0.41 | 0.002 | 0.002 | 1.02 |
| 12 | 2 | 1 | 0.005 | 0.009 | 0.005 | 1.90 | 1.14 | 0.004 | 0.48 | 0.005 | 0.005 | 1.01 |
| 12 | 2 | 2 | 0.006 | 0.010 | 0.007 | 1.72 | 1.11 | 0.004 | 0.42 | 0.006 | 0.006 | 1.01 |
| 16 | 4 | 1 | 0.010 | 0.019 | 0.011 | 1.88 | 1.11 | 0.009 | 0.48 | 0.010 | 0.010 | 1.00 |
| 16 | 4 | 2 | 0.013 | 0.022 | 0.014 | 1.69 | 1.09 | 0.009 | 0.42 | 0.013 | 0.013 | 0.99 |
| 20 | 6 | 1 | 0.019 | 0.034 | 0.020 | 1.83 | 1.07 | 0.016 | 0.48 | 0.019 | 0.018 | 0.95 |
| 20 | 6 | 2 | 0.024 | 0.040 | 0.025 | 1.65 | 1.06 | 0.016 | 0.41 | 0.024 | 0.023 | 0.97 |
| 24 | 8 | 1 | 0.032 | 0.057 | 0.034 | 1.81 | 1.07 | 0.027 | 0.47 | 0.032 | 0.029 | 0.92 |
| 24 | 8 | 2 | 0.040 | 0.065 | 0.042 | 1.62 | 1.04 | 0.027 | 0.41 | 0.040 | 0.038 | 0.96 |
| 32 | 12 | 1 | 0.069 | 0.127 | 0.075 | 1.84 | 1.09 | 0.061 | 0.48 | 0.069 | 0.066 | 0.96 |
| 32 | 12 | 2 | 0.090 | 0.148 | 0.096 | 1.64 | 1.06 | 0.061 | 0.42 | 0.089 | 0.086 | 0.96 |
| 40 | 16 | 1 | 0.130 | 0.246 | 0.144 | 1.89 | 1.11 | 0.116 | 0.47 | 0.130 | 0.126 | 0.97 |
| 40 | 16 | 2 | 0.173 | 0.284 | 0.184 | 1.64 | 1.06 | 0.116 | 0.41 | 0.173 | 0.166 | 0.96 |
| 48 | 20 | 1 | 0.219 | 0.420 | 0.248 | 1.91 | 1.13 | 0.200 | 0.48 | 0.219 | 0.217 | 0.99 |
| 48 | 20 | 2 | 0.297 | 0.492 | 0.321 | 1.65 | 1.08 | 0.200 | 0.41 | 0.298 | 0.293 | 0.98 |
| 56 | 24 | 1 | 0.352 | 0.658 | 0.389 | 1.87 | 1.11 | 0.313 | 0.47 | 0.351 | 0.342 | 0.98 |
| 56 | 24 | 2 | 0.485 | 0.784 | 0.516 | 1.62 | 1.06 | 0.313 | 0.40 | 0.485 | 0.475 | 0.98 |
| 72 | 31.25 | 1 | 0.826 | 1.395 | 0.840 | 1.69 | 1.02 | 0.645 | 0.46 | 0.829 | 0.772 | 0.93 |
| 72 | 31.25 | 2 | 1.121 | 1.661 | 1.095 | 1.48 | 0.98 | 0.645 | 0.39 | 1.131 | 1.047 | 0.93 |
| 131 | 62.5 | 1 | 6.153 | 8.907 | 5.333 | 1.45 | 0.87 | 3.643 | 0.41 | 8.841 | 5.341 | 0.60 |
| 131 | 62.5 | 2 | 8.344 | 10.260 | 6.757 | 1.23 | 0.81 | 3.643 | 0.36 | 10.261 | 6.781 | 0.66 |

{K_CROSSOVER}

On one core the pair path is never slower (0.99–1.02× at p = 8–16, 0.92–0.98× at p = 20–72).
Its losses in the FMM at low degrees (§5) are the parallel pass's.

## 5. Auto's rule

The first rule (by pairs where a level's targets with a V row are fewer than 4 × the threads)
left level 3 of W1 at kD = 500 by boxes on Kathleen at 1 × 40, where by pairs is 0.83–0.94×.
It also took low-degree levels of few targets by pairs, where by pairs loses (W1 at kD = 30,
level 3, p = 8: 1.4–2.5× on locust).

Forced on every Rotation level (`pairs`: CH3.6's matrix at 1 × N and 4 × N,
`--rcr-split boxes,pairs`):
- by pairs against by boxes over the evaluation: geometric mean 1.000 on locust (114 rows,
  range 0.70–1.50) and 0.985 on Kathleen (114 rows, 0.65–1.25);
- it wins on W1 at kD = 500 (0.65–0.88×) and loses on W4's sphere at 10⁻³ (levels 4 and 5,
  p = 7: 1.04–1.18× of a 30–40 ms evaluation).

On Kathleen several rows that ran no level by pairs (the same code path both ways) differ by
up to 1.25×: that is the noise of these sub-0.3 s cases there.

The degrees in between come from W1 and W3 at kD = 100 with Rotation named on every level
(`rotation`), and from the levels and pairs stages. Every Rotation level, by pairs over by
boxes, grouped by degree and boxes:

locust:

| p | boxes | levels × splits | pairs ÷ boxes |
| ---: | --- | ---: | --- |
| 7 | 32,768–80,264 | 14 | 1.55–3.80 |
| 8 | 70,264 | 7 | 1.12–2.99 |
| 11 | 2,368–4,096 | 4 | 1.27–1.95 |
| 15 | 2,368–32,768 | 6 | 1.10–1.52 |
| 18 | 512 | 4 | 0.76–1.12 |
| 22 | 512 | 4 | 0.88–1.02 |
| 24 | 512 | 4 | 0.92–1.00 |
| 24 | 1,408–32,768 | 6 | 0.98–1.12 |
| 30–38 | 64 | 12 | 0.39–0.90 |
| 66–76 | 512 | 21 | 0.69–0.94 |
| 123–135 | 64 | 21 | 0.26–0.73 |

Kathleen:

{K_DEGREES}

Hence the rule as built, a function of the level's degree, its targets with a V row and the
pool's threads:
- by pairs where p_l ≥ `RCR_PAIR_MIN_DEGREE` = 48, at every thread count, serially too;
- by pairs where p_l ≥ `RCR_PAIR_FEW_DEGREE` = 30 and the targets are fewer than
  `RCR_PAIR_SHARE` = 4 × the threads;
- by boxes elsewhere.

48 lies between the many-box levels of p = 24 (1.03–1.12×) and level 3 at p = 66 (0.69–0.94×);
30 is the lowest degree measured with few boxes (64 boxes, 0.39–0.90×). It reads the threads,
which no other Helmholtz rule may (constraint 9), only because no bit depends on it.

CH3.6's matrix under the rule (`matrix`, `--rcr-split boxes,auto`, one rank per node), auto
over boxes:
- locust: geometric mean 0.974 over 57 rows, range 0.67–1.04. Only W1 at kD = 500 ran by
  pairs (levels 2 and 3, 0.87–0.89×); the 1.04 row (W4's cube at kD = 1, 10⁻⁸) ran no level by
  pairs, and the 0.67 rows are 2–3 ms evaluations.
- Kathleen: {K_RULE}

## 6. Runs

- Kathleen, one exclusive node each (`singlenode`, node-c11a-065, 2 × 20 Cascade Lake cores;
  rustc 1.99.0; BLAS threads 1; even rank counts bound per socket):
  - job 634393 (`levels pair matrix`, from 37ddcda);
  - job 634402 (`pairs stored`, from 37ddcda with the script's later stages);
  - job 634421 (`matrix rotation crossover` under §5's rule, from 935dbab with the script's
    later stages).
- locust (GH200, 72 Neoverse V2 cores, Open MPI 5.0.10), 2026-11-16, quiet (load below 1
  before each stage):
  - `levels pair matrix` from 37ddcda;
  - `stored` and `pairs` from 37ddcda with the script's later stages;
  - the two experiments of §3 by hand (`locust-x/`);
  - `rotation` and `crossover` by hand with the stages' commands, from the rule's tree before
    its commit;
  - S6's `times`;
  - `matrix` under §5's rule from 935dbab (`rule/`).
- The M6: the tests and checks only, no timing.
- Raw outputs: `~/projects/fmm-archive/{kathleen,locust}/logs/ch4-0i/`. The tables come from
  `tools/helmholtz/ch4-0i-tables.py levels|stored|matrix|pairs|degrees <these launches>
  <CH4.0h's launches>`; every launch's checksum was compared with CH4.0h's launches of the
  same case and ε, with no difference.

## 7. Questions for sign-off

1. **The acceptance as read.**
   - *Recommended*: met on Kathleen, read against CH4.0e's measure (the one-core level target
     by target, the only baseline when the task was set): level 2 at 0.89–0.91 of linear at
     the node's best split. On locust it reaches 0.75–0.78 at 1 × 72.
   - Against the one core by pairs, which this task made 0.56–0.60× faster, it is 0.44–0.54,
     for the reason of §3. The lever is CH4.0l, recorded off the priority list (question 3).
   - *Alternative*: not met; CH4.0l on the list before H4.
2. **Auto's rule as the default.**
   - *Recommended*: `RcrSplit::Auto` as built (§5): by pairs at p_l ≥ 48, and at p_l ≥ 30 with
     fewer targets than 4 × the threads, within 1 GiB per node. No bit changes.
   - *Alternative*: by boxes by default, by pairs opt-in.
3. **The follow-up CH4.0l (the RCR pair: one coaxial row over a class's pairs, and vectorised
   chains).**
   - *Recommended*: off the priority list. Inside D1 only W1 at kD = 500 has RCR levels at
     these degrees, and H4's diagonal form takes those levels' passing classes (CH4.2, CH4.3).
     CH4.0l is worth it only for the classes left on RCR, which CH4.3 measures.
   - *Alternative*: on the list before CHA.1.

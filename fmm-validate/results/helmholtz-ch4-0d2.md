# Helmholtz CH4.0d2: V pairs between leaves of few points as near pairs

Task: `docs/helmholtz/tasks/CH4.0d2.md`. Roadmap: H4, CH4.0d2 (CH4.0d split on 2026-11-13).
The interface and the crossover approved by the user on 2026-11-13 ("Approve
(Recommended)", "As planned (Recommended)"); the rule on levels with waves decided by the
user on 2026-11-13 ("Group per (parent, source) (Recommended)", §3). Proposal §2
(constraints 1, 2, 9), §9.1. Builds on `fmm-validate/results/helmholtz-ch4-0d1b.md`.

## 0. Summary

- **The rule.** `nd-fmm-plan`'s near rule gains an opt-in V predicate
  (`NearRule::with_v_pairs`, `VPair`): a V entry s of a local leaf t, s a leaf, becomes a near
  pair where the operator's predicate holds; t's V row loses it and its near row gains it in
  (level, key) order, and a ghost s is named in the leaf numbering, so the source exchange
  delivers its points. `NearRule::new` offers no V pair: Laplace's rule and plans are
  unchanged. `HelmholtzFmmBuilder::near_v(VNear)` applies it (`Off`, the default; a named
  crossover of n_s n_t; `CostModel`).
- **A defect found and fixed on the way (§3).** On a level whose M2L is the exponential one,
  merge and shift serves each source of a parent's slots to every child in its geometric
  mask, so a pair converted alone while a sibling keeps its source is counted twice (W1 at
  kD = 500 under the default, a partial crossover: φ 4·10⁵× the rule off's). On those levels
  the rule decides once per parent and source (`v_group`), every child reading the source
  converting or none.
- **Converting pays where the M2L per pair is dear** (Kathleen, one rank × 40 threads, build
  plus one evaluation over the rule off's, the median over the cases that take at least
  0.3 s; §4):

  | M2L | ε = 10⁻³ | 10⁻⁶ | 10⁻⁸ |
  | --- | --- | --- | --- |
  | Rotation, cost model | 0.94 (0.71–1.00) | 0.66 (0.43–0.85) | 0.52 (0.23–0.66) |
  | the default, cost model | 1.03 (0.55–1.06) | 1.01 (0.93–1.14) | 0.85 (0.72–0.95) |
  | the default, every V pair between leaves | 0.94 (0.52–0.98) | 1.00 (0.90–1.03) | 0.85 (0.73–0.96) |

  On locust (NEON, 1 × 72) the default is 1.05, 1.04 and 1.00 under the cost model, Rotation
  0.87 and 0.69 at 10⁻⁶ and 10⁻⁸. Under Rotation nearly every V pair between leaves of 64
  points pays from p ≈ 11; under the default (Batched to p = 11, the waves above) the gain is
  small or negative except at 10⁻⁸ on Kathleen.
- **On the large trees** (W1 at kD = 500, W3 at N = 4·10⁶ and kD = 100, the under-resolved W3
  at kD = 500; 10⁻⁶ and 10⁻⁸) every V pair between leaves converts under the cost model. Under
  Rotation build plus evaluation is 0.23–0.90 of the rule off's; under the default the
  evaluation is 0.43–0.90, but the build rises by up to 1.29 (the decisions over 46 million
  pairs), so build plus one evaluation is 0.69–1.01 (§4.1).
- **The errors** stay within 1.1× the rule off's on every case of every round (worst 1.095,
  W4's Plummer sphere at 10⁻⁸ under the default; converted pairs are exact, most errors fall).
- **Rank-count invariance**: the checksums at 4 × 10 equal those at 1 × 40 with the rule on
  (§5); the tests check P ranks against the one-rank FMM bit for bit at 2 and 4 ranks.
- **Traffic** (§5): on the under-resolved W3 at kD = 500 on 4 ranks, every V pair between
  leaves converted, the multipoles read from other ranks fall from 95,284 to 44,901, the
  source points exchanged rise from 31,288 to 63,259.
- **Not proposed as the default** (§6, question 1, signed off 2026-11-13 as recommended): the
  gains say leaves of 64 points are too small for the degrees of ε ≤ 10⁻⁶ under Rotation, and under the default the rule only pays
  at 10⁻⁸ on one machine class. CH4.0d3 (the leaf size) should sweep the leaf size with the
  rule off and with `CostModel`, and propose the two together.
- **Laplace**: `Plan::new` unchanged (`plan_build_cost` on Kathleen within `main`'s spread),
  Laplace's `Fmm` unchanged, every Laplace test passes.

## 1. Method

Release builds, BLAS threads 1, `helmholtz_accuracy` through
`tools/helmholtz/ch4-0d2-runs.sh` (Rotation and the default `auto-pw`; `--near-v` lists;
`--levels`, whose "M2L per level" and "Near pairs" tables give each level's M2L seconds and V
pairs, and its P2P seconds and near interactions Σ_t n_t Σ_s n_s), tables by
`tools/helmholtz/ch4-0d2-tables.py`. `matrix` is CH3.6's matrix (W1 at kD = 10, 30, 100; W3 at
kD = 10, 30; the four W4 distributions at N = 10⁵ and kD = 1, 10, 100) at 10⁻³, 10⁻⁶ and
10⁻⁸, two evaluations, each list in both orders (ABBA); `large` is W1 at kD = 500 and W3 at
N = 4·10⁶ and kD = 100 at 10⁻⁶ and 10⁻⁸; `w3` the under-resolved W3 (N = 10⁶) at kD = 500;
`traffic` the default at 10⁻⁶ on 4 ranks × 10 threads and 1 × 40. Raw outputs:
`~/projects/fmm-archive/{kathleen,locust}/logs/ch4-0d2/`.

| round | machine | job | node | tree | stages and crossovers |
| --- | --- | --- | --- | --- | --- |
| 1 | Kathleen | 633813–633815 | node-c11a-051 | e6ea63e (per pair) | `matrix`: off, 300, 1000, 3000, 10⁴, inf |
| 1 | Kathleen | 633811 | node-c11a-122 | e6ea63e | `large`: off, 1000, 3000, 10⁴, 3·10⁴, inf |
| 1 | Kathleen | 633812 | node-c11a-036 | e6ea63e | `w3`: off, 3, 30, inf; `traffic` |
| 1 | Kathleen | 633816 | node-c11a-036 | e6ea63e; `main` 7f4466c | `plan_build_cost`, alternately three times |
| 2 | locust | – | GH200 | b7137cb (one class) | `matrix`, `large`: off, cost, inf |
| 2 | Kathleen | 633849 | node-c11a-004 | b7137cb | `matrix`: off, cost, inf |
| 3 | Kathleen | 633854 | node-c11b-014 | 3ead76a (final) | `matrix`: off, cost, inf |
| 3 | Kathleen | 633855 | node-c11b-014 | 3ead76a | `large`, `w3`, `traffic`: off, cost, inf |
| 3 | locust | – | GH200 (load 1.7) | 3ead76a | `matrix`, `large`: off, cost, inf |

Round 1's levels with waves ran the per-pair rule (the defect of §3): their errors at partial
crossovers are not usable, their times are (the same work). Round 2 ran one class's constants
(Kathleen's, the waves' at 11); round 3 the final code. Tree 3ead76a differs from the PR's head
only in debug assertions, tests and docs.

## 2. The cost model's constants

Per level, from each pair of runs of the same case with the rule off and on (the level's M2L
the same): the M2L's thread time saved per V pair removed and the P2P's thread time per near
interaction added (seconds × threads over the counts), medians over the levels with at least
100 pairs removed (`ch4-0d2-tables.py levels`).

Kathleen (round 1, every launch):

| M2L that ran | p | rows | M2L µs saved per pair removed (median, range) | P2P ns per interaction added (median, range) | crossover n_s n_t (medians) |
| --- | ---: | ---: | --- | --- | ---: |
| batched-fused | 7 | 230 | 0.94 (-1.02–12.83) | 6.72 (1.90–17.86) | 139 |
| batched-fused | 8 | 2 | 3.26 (3.04–3.49) | – | – |
| planewave i2 | 15 | 289 | 3.28 (-10.93–27.32) | 6.58 (3.31–209.15) | 499 |
| planewave i3 | 32 | 289 | 10.39 (-50.71–88.32) | 6.30 (3.31–209.70) | 1650 |
| rotation | 7 | 250 | 4.11 (-1.40–6.26) | 6.52 (3.31–17.97) | 630 |
| rotation | 8 | 4 | 5.72 (3.44–6.07) | 5.59 (5.59–5.59) | 1022 |
| rotation | 11 | 52 | 11.98 (9.39–16.70) | 7.04 (5.53–17.11) | 1701 |
| rotation | 15 | 289 | 26.05 (15.32–52.53) | 6.58 (0.00–209.82) | 3958 |
| rotation | 24 | 289 | 95.36 (50.22–137.19) | 6.30 (0.00–209.70) | 15147 |

locust (round 2):

| M2L that ran | p | rows | M2L µs saved per pair removed (median, range) | P2P ns per interaction added (median, range) | crossover n_s n_t (medians) |
| --- | ---: | ---: | --- | --- | ---: |
| batched-fused | 7 | 80 | 0.29 (-2.95–6.43) | 14.51 (0.00–24.12) | 20 |
| planewave i2 | 11 | 4 | 2.08 (1.83–2.22) | 13.65 (12.56–14.73) | 152 |
| planewave i2 | 15 | 106 | 2.08 (-13.12–8.18) | 12.56 (6.08–27.60) | 166 |
| planewave i3 | 32 | 95 | 5.53 (-9.78–37.00) | 13.50 (9.11–32.15) | 410 |
| rotation | 7 | 114 | 1.79 (-44.33–4.81) | 12.32 (0.00–25.67) | 146 |
| rotation | 8 | 4 | 2.90 (2.49–3.03) | 14.75 (14.75–14.75) | 197 |
| rotation | 11 | 12 | 5.07 (2.90–15.27) | 14.37 (10.56–17.35) | 353 |
| rotation | 15 | 100 | 11.13 (-11.79–14.50) | 14.10 (11.89–20.10) | 789 |
| rotation | 24 | 100 | 40.47 (1.73–45.06) | 14.28 (11.89–24.12) | 2835 |

Fitted (`VCosts`, thread-nanoseconds; crossover = M2L per pair over P2P per interaction):

| class | P2P per interaction | Rotation per pair | Batched per pair | waves per pair |
| --- | ---: | --- | --- | --- |
| x86 (`V_COSTS_X86`, Kathleen) | 6.5 | 800 + 6.0 (p+1)³ | 0.25 (p+1)⁴ | 30 (p_w+1)² |
| NEON (`V_COSTS_NEON`, locust) | 14.0 | 600 + 2.5 (p+1)³ | 0.07 (p+1)⁴ | 6 (p_w+1)² |

The waves' per-pair figure is not a marginal cost: removing some of a level's pairs saves
only their shifts, removing all saves the level's waves too. Round 2's 11 (the matrix's
median, mostly partial) converted too few pairs on Kathleen; 30 matches the large cases,
whose wave levels converted whole (9.9 µs at p = 15, 32.6 µs at p = 32). Without gradients the
P2P is charged 0.67 of its time (CH3.4b's kernel ratio). The crossover per level is the
cheapest M2L the setting may run there (Batched below Auto's crossover degree, the waves where
D8's rule gives them), agreed over the ranks (minimum), so Auto's later choice only means more
pairs would have paid.

## 3. The defect on levels with waves

Round 1's default rows at partial crossovers erred by up to 10¹¹× the rule off's (W1 at
kD = 500, 10⁻⁶, crossover 1000: φ 4.4·10⁵×), with every V pair between leaves converted 0.98×,
and Rotation exact everywhere. Merge and shift
(`src/helmholtz/operator_plane_wave.rs`) takes a parent's slots from the union of its computed
children's V rows and serves each accumulator to every child in its geometric mask: a
converted pair still reached its target through the parent's waves. Rotation, Batched, Dense
and Classes follow the rows; the empty-box rule leaves out exact zeros only, so it was never
affected. `check_near_v` reproduced it under PlaneWave (φ 1.8·10⁻² against 1.2·10⁻⁵).

The fix, the user's choice of three (2026-11-13): on a level with waves the rule decides once
per parent P and source s, for every child of P whose V list holds s: they all convert if all
are leaves and n_s n_P < n m_s (n_P the target points under P, m_s the children reading s; so
n_P / m_s bounds their mean target count). The decision reads keys and counts by key only,
the same on every rank. The builder predicts the levels with waves before the plan
(`auto_plane_wave_levels`, the operator's choice as a function) and asserts them against the
operator's. A first version cost about 170 ns a pair in the build (hash lookups per pair, +0.1 s
on W4's cube); it caches the parent's leaf mask for its siblings and rejects on the counts
first.

## 4. The gain

### 4.1 Round 3 on Kathleen

Job 633854 (`matrix`) and 633855 (`large`, `w3`): build plus one evaluation over the rule off's,
median (range) over the cases of at least 0.3 s (the matrix's both orders averaged).

| M2L | ε | cases | cost model: build + evaluate ÷ off | every V pair between leaves | evaluate ÷ off (cost model) | worst error ÷ off (cost model) |
| --- | --- | ---: | --- | --- | --- | ---: |
| Rotation | 10⁻3 | 9 | 0.94 (0.71–1.00) | 0.95 (0.70–0.99) | 0.81 (0.54–0.93) | 0.984 |
| Rotation | 10⁻6 | 14 | 0.66 (0.43–0.84) | 0.66 (0.43–0.84) | 0.57 (0.26–0.74) | 0.987 |
| Rotation | 10⁻8 | 14 | 0.52 (0.23–0.66) | 0.52 (0.23–0.65) | 0.48 (0.17–0.58) | 1.000 |
| the default | 10⁻3 | 14 | 1.03 (0.55–1.06) | 0.94 (0.52–0.98) | 1.00 (0.82–1.00) | 1.000 |
| the default | 10⁻6 | 14 | 1.01 (0.93–1.14) | 1.00 (0.90–1.02) | 0.90 (0.77–1.00) | 0.987 |
| the default | 10⁻8 | 15 | 0.85 (0.72–0.95) | 0.85 (0.73–0.96) | 0.74 (0.45–0.90) | 1.095 |

The large cases (job 633855; one order, two evaluations, the second's time):

| case | ε | M2L | V crossover | V pairs converted | build s | evaluate s | build ÷ off | evaluate ÷ off | (build + evaluate) ÷ off | φ ÷ off |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1, N = 1989437, kD = 500 | 1e-6 | default | off | – | 9.45 | 9.830 | 1.000 | 1.000 | 1.000 | 1.000 |
| W1, N = 1989437, kD = 500 | 1e-6 | default | cost | 3067716 of 3298824 | 10.12 | 8.870 | 1.071 | 0.902 | 0.985 | 0.979 |
| W1, N = 1989437, kD = 500 | 1e-6 | default | inf | 3298824 of 3298824 | 9.94 | 8.100 | 1.052 | 0.824 | 0.936 | 0.979 |
| W1, N = 1989437, kD = 500 | 1e-6 | rotation | off | – | 9.06 | 11.430 | 1.000 | 1.000 | 1.000 | 1.000 |
| W1, N = 1989437, kD = 500 | 1e-6 | rotation | cost | 3298824 of 3298824 | 9.17 | 9.200 | 1.012 | 0.805 | 0.897 | 0.987 |
| W1, N = 1989437, kD = 500 | 1e-6 | rotation | inf | 3298824 of 3298824 | 9.30 | 8.930 | 1.026 | 0.781 | 0.890 | 0.987 |
| W1, N = 1989437, kD = 500 | 1e-8 | default | off | – | 13.45 | 19.450 | 1.000 | 1.000 | 1.000 | 1.000 |
| W1, N = 1989437, kD = 500 | 1e-8 | default | cost | 3298824 of 3298824 | 13.84 | 13.300 | 1.029 | 0.684 | 0.825 | 1.000 |
| W1, N = 1989437, kD = 500 | 1e-8 | default | inf | 3298824 of 3298824 | 14.04 | 14.400 | 1.044 | 0.740 | 0.864 | 1.000 |
| W1, N = 1989437, kD = 500 | 1e-8 | rotation | off | – | 9.52 | 23.960 | 1.000 | 1.000 | 1.000 | 1.000 |
| W1, N = 1989437, kD = 500 | 1e-8 | rotation | cost | 3298824 of 3298824 | 9.78 | 13.640 | 1.027 | 0.569 | 0.700 | 1.000 |
| W1, N = 1989437, kD = 500 | 1e-8 | rotation | inf | 3298824 of 3298824 | 9.48 | 13.460 | 0.996 | 0.562 | 0.685 | 1.000 |
| W3, N = 1000000, kD = 500 | 1e-6 | default | off | – | 74.25 | 59.220 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3, N = 1000000, kD = 500 | 1e-6 | default | cost | 55193392 of 55193392 | 84.61 | 33.680 | 1.140 | 0.569 | 0.886 | 0.973 |
| W3, N = 1000000, kD = 500 | 1e-6 | default | inf | 55193392 of 55193392 | 81.24 | 33.520 | 1.094 | 0.566 | 0.860 | 0.973 |
| W3, N = 1000000, kD = 500 | 1e-6 | rotation | off | – | 73.66 | 141.770 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3, N = 1000000, kD = 500 | 1e-6 | rotation | cost | 55193392 of 55193392 | 75.18 | 97.730 | 1.021 | 0.689 | 0.803 | 0.973 |
| W3, N = 1000000, kD = 500 | 1e-6 | rotation | inf | 55193392 of 55193392 | 75.18 | 91.630 | 1.021 | 0.646 | 0.774 | 0.973 |
| W3, N = 1000000, kD = 500 | 1e-8 | default | off | – | 77.94 | 165.690 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3, N = 1000000, kD = 500 | 1e-8 | default | cost | 55193392 of 55193392 | 85.17 | 82.620 | 1.093 | 0.499 | 0.689 | 1.000 |
| W3, N = 1000000, kD = 500 | 1e-8 | default | inf | 55193392 of 55193392 | 86.32 | 82.020 | 1.108 | 0.495 | 0.691 | 1.000 |
| W3, N = 1000000, kD = 500 | 1e-8 | rotation | off | – | 75.27 | 383.480 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3, N = 1000000, kD = 500 | 1e-8 | rotation | cost | 55193392 of 55193392 | 77.37 | 209.100 | 1.028 | 0.545 | 0.624 | 1.000 |
| W3, N = 1000000, kD = 500 | 1e-8 | rotation | inf | 55193392 of 55193392 | 75.73 | 227.950 | 1.006 | 0.594 | 0.662 | 1.000 |
| W3, N = 4031442, kD = 100 | 1e-6 | default | off | – | 15.88 | 10.140 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3, N = 4031442, kD = 100 | 1e-6 | default | cost | 46298352 of 46298376 | 20.48 | 5.850 | 1.290 | 0.577 | 1.012 | 0.990 |
| W3, N = 4031442, kD = 100 | 1e-6 | default | inf | 46298376 of 46298376 | 20.49 | 5.500 | 1.290 | 0.542 | 0.999 | 0.990 |
| W3, N = 4031442, kD = 100 | 1e-6 | rotation | off | – | 15.45 | 48.810 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3, N = 4031442, kD = 100 | 1e-6 | rotation | cost | 46298376 of 46298376 | 16.95 | 9.850 | 1.097 | 0.202 | 0.417 | 0.988 |
| W3, N = 4031442, kD = 100 | 1e-6 | rotation | inf | 46298376 of 46298376 | 16.60 | 9.850 | 1.074 | 0.202 | 0.412 | 0.988 |
| W3, N = 4031442, kD = 100 | 1e-8 | default | off | – | 16.07 | 35.340 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3, N = 4031442, kD = 100 | 1e-8 | default | cost | 46298376 of 46298376 | 20.79 | 15.160 | 1.294 | 0.429 | 0.699 | 1.000 |
| W3, N = 4031442, kD = 100 | 1e-8 | default | inf | 46298376 of 46298376 | 21.34 | 15.280 | 1.328 | 0.432 | 0.712 | 1.000 |
| W3, N = 4031442, kD = 100 | 1e-8 | rotation | off | – | 15.40 | 174.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3, N = 4031442, kD = 100 | 1e-8 | rotation | cost | 46298376 of 46298376 | 16.95 | 27.020 | 1.101 | 0.155 | 0.231 | 1.000 |
| W3, N = 4031442, kD = 100 | 1e-8 | rotation | inf | 46298376 of 46298376 | 16.58 | 26.780 | 1.077 | 0.153 | 0.228 | 1.000 |

Under Rotation every case gains from 10⁻⁶ on (to 0.23 on W3 at N = 4·10⁶, 10⁻⁸). Under the
default the evaluation falls to 0.43–0.90 on these trees, but the build rises by up to 1.29
(the grouped decision's lookups over 46 million pairs on W3 at N = 4·10⁶, and the larger near
rows), so build plus one evaluation is 0.69–1.01; on the matrix's small trees the default at
10⁻³ and 10⁻⁶ converts few pairs (Batched's and the waves' crossovers) and pays the build.

### 4.2 Round 3 on locust

locust, 1 × 72 (load 1.7 before), `matrix` and `large`, as §4.1:

| M2L | ε | cases | cost model: build + evaluate ÷ off | every V pair between leaves | evaluate ÷ off (cost model) | worst error ÷ off (cost model) |
| --- | --- | ---: | --- | --- | --- | ---: |
| Rotation | 10⁻6 | 8 | 0.87 (0.71–0.98) | 0.89 (0.72–0.99) | 0.77 (0.40–0.91) | 1.009 |
| Rotation | 10⁻8 | 16 | 0.69 (0.40–0.85) | 0.69 (0.39–0.86) | 0.60 (0.21–0.68) | 1.005 |
| the default | 10⁻3 | 9 | 1.05 (1.01–1.06) | 0.98 (0.90–1.17) | 1.00 (0.94–1.06) | 1.002 |
| the default | 10⁻6 | 11 | 1.04 (0.99–1.12) | 1.07 (0.97–1.19) | 1.00 (0.94–1.05) | 1.004 |
| the default | 10⁻8 | 16 | 1.00 (0.98–1.07) | 0.98 (0.92–1.02) | 0.96 (0.88–1.03) | 1.000 |

On NEON the waves and Batched are cheap against the P2P (§2): the default's crossovers are
small, and converting every pair loses (1.07 at 10⁻⁶).

### 4.3 Round 1: the named crossovers (Kathleen)

The build plus one evaluation over the rule off's, median (range) over the cases of at least
0.3 s, at every V pair between leaves converted and at each case's best named crossover:

| M2L | ε | cases | every pair | best crossover |
| --- | --- | ---: | --- | --- |
| Rotation | 10⁻³ | 8 | 0.974 (0.768–1.012) | 0.961 (0.768–0.988) |
| Rotation | 10⁻⁶ | 17 | 0.726 (0.476–0.917) | 0.726 (0.467–0.908) |
| Rotation | 10⁻⁸ | 17 | 0.571 (0.241–0.738) | 0.571 (0.237–0.715) |
| the default | 10⁻³ | 14 | 0.954 (0.554–0.989) | 0.935 (0.539–0.966) |
| the default | 10⁻⁶ | 17 | 0.922 (0.821–0.990) | 0.904 (0.821–0.985) |
| the default | 10⁻⁸ | 18 | 0.831 (0.609–0.968) | 0.827 (0.603–0.952) |

(The default's figures here ran the per-pair rule, which at "every pair" also converted pairs
whose groups the fix keeps; round 3's grouped "every pair" is 0.94, 1.00 and 0.85.) Under
Rotation the best crossover is "every pair" on nearly every case: with leaves of at most 64
points, n_s n_t ≤ 4,096 against Rotation's crossover of 1,700 at p = 11, 3,900 at p = 15 and
15,000 at p = 24.

## 5. Traffic and rank-count invariance

Job 633855 (final code), the default at 10⁻⁶, 4 ranks × 10 threads (`--map-by slot:PE=10
--bind-to core`) against one rank × 40 threads on the same node. The V pairs, the multipoles
the ranks read from other ranks (each rank's distinct V- and W-row boxes it does not compute,
summed), the source points the source exchange moves per evaluation (sent = received, summed),
and the checksum of the whole output:

| case | rule | V pairs converted | V pairs left | remote multipoles | source points moved | checksum 4 × 10 | checksum 1 × 40 | evaluate s 4 × 10 / 1 × 40 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| W3, N = 10⁶, kD = 500 | off | 0 | 105,510,916 | 95,284 | 31,288 | b8196d7f1fe12151 | b8196d7f1fe12151 | 61.0 / 58.8 |
| W3, N = 10⁶, kD = 500 | cost | 55,193,392 | 50,317,524 | 44,901 | 63,259 | 31d2d4a548cfda3c | 31d2d4a548cfda3c | 34.1 / 33.5 |
| W3, N = 10⁶, kD = 500 | every pair | 55,193,392 | 50,317,524 | 44,901 | 63,259 | 31d2d4a548cfda3c | 31d2d4a548cfda3c | 34.0 / 33.4 |
| W1, kD = 500 | off | 0 | 4,476,464 | 8,193 | 62,482 | 30d1687b75dca4d0 | 30d1687b75dca4d0 | 11.2 / 9.8 |
| W1, kD = 500 | cost | 3,067,716 | 1,408,748 | 5,389 | 125,598 | 472c4bb27b46dc7a | 472c4bb27b46dc7a | 10.2 / 8.9 |
| W1, kD = 500 | every pair | 3,298,824 | 1,177,640 | 4,109 | 125,598 | 37ea6f9e41bd425b | 37ea6f9e41bd425b | 9.5 / 8.2 |

Every checksum at 4 ranks equals one rank's: rank-count invariance with the rule on, also where
the cost model converts the groups of a level with waves in part (W1). The multipoles read
from other ranks fall to 47–66 % of the rule off's, the source points moved double (the halo
of the converted V leaves). The tests check the same at 2 and 4 ranks against the one-rank FMM
bit for bit (`tests/helmholtz_fmm.rs`, `check_near_v`; `tests/mpi_regressions.rs`,
`check_near_v_rule`, also at 8).

**Laplace** (job 633816, `plan_build_cost`, one rank, the best of five builds per tree, ms;
`main` 7f4466c and the branch alternately):

| tree | `main` runs 1–3 | branch runs 1–3 |
| --- | --- | --- |
| graded corner blob | 7.13, 7.08, 7.24 | 7.03, 7.11, 7.11 |
| 10⁵ uniform, level 4 | 81.26, 81.55, 81.82 | 80.66, 81.52, 81.34 |
| 10⁵ uniform, 32 per leaf | 96.13, 95.83, 96.08 | 96.08, 94.84, 95.81 |

Medians 7.13, 81.55 and 96.08 on `main`, 7.11, 81.34 and 95.81 on the branch: within `main`'s
spread. `Plan::new` and `Plan::with_near_rule` without a V predicate run the code they ran (an
`Option` test per V entry of a local leaf, with a rule only); Laplace's `Fmm` is unchanged.

## 6. Questions for sign-off

1. **The default.** Signed off on 2026-11-13 (the user: "I agree with your recommendation").
   Recommended: keep `VNear::Off` as the default and hand the decision to
   CH4.0d3, which sweeps the leaf size with the rule off and with `VNear::CostModel` and
   proposes the two together (the Rotation gains are those of larger leaves; under the default
   the rule gains only at 10⁻⁸, and on Kathleen only). Alternatives: `CostModel` on by default
   (median 1.01–1.05 under the default at 10⁻³ and 10⁻⁶, worst 1.14); on at ε < 10⁻⁷ only
   (Kathleen 0.85, locust 1.00).

## 7. Reproduce

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
tools/helmholtz/ch4-0d2-runs.sh <out> matrix large w3 traffic          # round 1
CROSSOVERS=off,cost,inf tools/helmholtz/ch4-0d2-runs.sh <out> matrix large w3 traffic
THREADS=72 CROSSOVERS=off,cost,inf tools/helmholtz/ch4-0d2-runs.sh <out> matrix large  # locust
python3 tools/helmholtz/ch4-0d2-tables.py cases <out>/*.md
python3 tools/helmholtz/ch4-0d2-tables.py levels <out>/*.md
```

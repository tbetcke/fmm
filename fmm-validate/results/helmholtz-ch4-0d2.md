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
- **The errors** stay within 1.1× the rule off's on every case of every round (worst 1.095,
  W4's Plummer sphere at 10⁻⁸ under the default; converted pairs are exact, most errors fall).
- **Rank-count invariance**: the checksums at 4 × 10 equal those at 1 × 40 with the rule on
  (§5); the tests check P ranks against the one-rank FMM bit for bit at 2 and 4 ranks.
- **Traffic** (§5): on the under-resolved W3 at kD = 500 on 4 ranks, every V pair between
  leaves converted, the multipoles read from other ranks fall from 95,284 to 44,901, the
  source points exchanged rise from 31,288 to 63,259.
- **Not proposed as the default** (§6, question 1, signed off 2026-11-13 as recommended): the gains say leaves of 64 points are too
  small for the degrees of ε ≤ 10⁻⁶ under Rotation, and under the default the rule only pays
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

ROUND3_K

### 4.2 Round 3 on locust

ROUND3_L

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

TRAFFIC

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

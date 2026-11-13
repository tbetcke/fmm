# Helmholtz CH4.0d1: empty boxes left out of the lists

Task: `docs/helmholtz/tasks/CH4.0d1.md`. Roadmap: H4, CH4.0d1 (CH4.0d split on 2026-11-13 into
CH4.0d1–d3); the design the user agreed to on #214 (2026-11-13), amended on 2026-11-13 (the
top-tree keys' counts, below). Proposal §2 (constraints 1, 2, 9), §6.1–§6.3, §9.1.

## 0. Summary

- **The rule.** An opt-in rule of `nd-fmm-plan` (`Plan::with_rules` with
  `PlanRules::empty_boxes`, module `nd_fmm_plan::empty`) leaves out of the lists the work of
  the boxes whose whole subtree holds no target point (their V, X, W and L2L rows) or no
  source point (the V, W and M2M entries that name them). Only exact zeros go: under
  Rotation, Batched, PlaneWave and Auto every row keeps its order and loses terms that are a
  zero multipole or feed a local no point reads. The counts come from the local leaves summed
  up the tree, the coarse blocks' counts all-gathered (every top-tree key, `Global` boxes
  included) and the other ghost boxes' from their owners, so every rank leaves out what one
  rank leaves out. `Plan::new` and Laplace are unchanged.
- **The output is the rule off's.** In every Kathleen launch below (118 case × ε × M2L rows
  of the final tables, each with the rule on and off) the checksum of the whole output and the errors against the
  direct sum are the rule off's, bit for bit, wherever both runs used the same M2L on every
  level. They differed only where Auto's term chose another M2L for a level (below). The
  tests check the values (−0 = +0) under every M2L, 2 threads and the one-rank FMM bit for
  bit at 1, 2 and 4 ranks.
- **The evaluation gains where boxes are empty**, never loses beyond noise elsewhere
  (one rank × 40 threads on Kathleen, the evaluation with the rule over the rule off's):
  - W1 (points on a sphere) at kD = 500: 0.43–0.44 under the default, 0.20–0.22 under
    Rotation (V pairs 4.5 million against 28.4 million);
  - the under-resolved W3 (N = 10⁶) at kD = 500: 0.51–0.56 under the default, 0.28–0.36
    under Rotation (V pairs 106 million against 419 million; 1.26 million boxes without
    targets of 2.06 million leaves); level 7's M2L at 10⁻⁸ under Rotation 124 s against 820 s;
  - W1 at kD = 100 and W4's sphere: 0.57–0.73 under the default, 0.36–0.53 under Rotation;
    W4's Plummer sphere and clusters 0.79–1.04;
  - the uniform trees (W3 at N ≤ 4·10⁶ and kD ≤ 200, W4's cube) have no empty box and the
    same pairs: 0.95–1.08 where the evaluation takes at least 0.2 s, noise.
- **The build pays for it: a second plan.** The rule builds the plan twice (`Plan::new` for
  the counts' exchange lists, then `Plan::with_rules`), and the plan is built serially, about
  17 µs per box on Kathleen. The build with the rule over the rule off's build is 1.1–1.7
  (W3 at N = 4·10⁶: 24.5 s against 15.6 s; the under-resolved W3 at kD = 500: 139 s against
  93 s). So build plus one evaluation:
  - gains on sparse trees: W1 at kD = 500 0.64–0.72 (default), 0.30–0.41 (Rotation); the
    under-resolved W3 at kD = 500 0.73 at 10⁻⁸ and 1.00 at 10⁻⁶ (default), 0.38–0.63
    (Rotation); W1 at kD = 100 0.60–0.91;
  - loses on uniform trees: W3 at N = 4·10⁶ 1.10–1.33, W4's cube 1.05–1.21, W3 at 10⁵
    1.10–1.19, W3 at N = 10⁶ and kD = 200 1.10–1.24 (and up to 1.5 on the smallest cases,
    whose build and evaluation take under 0.2 s).

  Taking the leaf counts from the routes the build already makes (one change of this PR) cut
  about a tenth of the extra build; the rest is the second plan. A follow-up (CH4.0d1b) would
  build the rule's plan once.
- **On 4 ranks × 10 threads** (W3, N = 10⁶, kD = 500, 10⁻⁶, the default) the multipoles the
  ranks read from other ranks fall from 175,800 to 95,284, the evaluation from 116.3 s to
  63.7 s, the build rises from 25.9 s to 37.6 s; the same checksum.
- **Auto's term reads the rule's pairs.** Under the default, a level's M2L changed on W1 at
  kD = 100, W4's sphere at kD = 100 (level 4) and W4's clusters (levels 2–4) at 10⁻³: Batched
  became Rotation where the rule left too few pairs for its tables. The evaluation was faster
  in each (0.57–0.95), the errors equal to the digits shown.
- **Laplace**: `Plan::new` unchanged (every Laplace test passes, and with counts without a zero
  the rule's plan is `Plan::new`'s); the plan's build time on Kathleen within the run-to-run
  spread (§4). No D9 evaluation runs (the user, 2026-11-13: "Tests + build time").

## 1. Method

Kathleen, one exclusive node per job (`singlenode`; 2 × 20 Cascade Lake cores, Xeon Gold 6248),
release builds, rustc 1.99.0, AVX-512 P2P, BLAS threads 1, `helmholtz_accuracy` through
`tools/helmholtz/ch4-0d1-runs.sh`; each launch runs Rotation and the default
(`auto-pw`: Auto with the exponential M2L, D8) with `--empty-boxes on,off` and, for `matrix`
and `large`, again with `off,on` (ABBA); the tables are the means of the two orders, made by
`tools/helmholtz/ch4-0d1-tables.py`. `matrix` and `large` ran two evaluations per build (the
row is the second's), `w3` one. Raw outputs: `~/projects/fmm-archive/kathleen/logs/ch4-0d1/`.

| job | node | tree | stages |
| --- | --- | --- | --- |
| 633564 | node-c11a-072 | 8631669 | `matrix`, `traffic` (4 ranks × 10 threads) |
| 633565 | node-c11b-034 | 8631669 | `large` |
| 633566 | node-c11a-009 | 8631669 | `w3` |
| 633567 | node-c11a-017 | f32e5ba (`main`), 8631669 | `plan_build_cost`, alternately three times each |
| 633596 | node-c11a-061 | d2a9efc | `matrix`, `large` again with the counts from the routes |

The trees 8631669 and d2a9efc differ in the build alone (the leaf counts from
`cost::leaf_point_counts` or from the routes); the plan, the counts and the output are the
same. The tables of §2 and §3 are job 633596's (the final code), §3.2 job 633566's.

## 2. CH3.6's matrix (one rank × 40 threads, job 633596)

On over off, the default M2L (Auto with waves) and Rotation. "Build + evaluate" is one build
and one evaluation.

| Workload | N | kD | ε | M2L | ranks × threads | build on / off s | evaluate on / off s | on ÷ off (evaluate) | on ÷ off (build + evaluate) | same checksum | same errors | V pairs on / off | levels whose M2L differs |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| W1 | 796 | 10 | 1e-3 | default | 1 × 40 | 0.08 / 0.08 | 0.00 / 0.00 | – | 1.000 | yes | yes | 2504 / 3096 | – |
| W1 | 796 | 10 | 1e-3 | rotation | 1 × 40 | 0.04 / 0.01 | 0.00 / 0.00 | – | 8.000 | yes | yes | 2504 / 3096 | – |
| W1 | 796 | 10 | 1e-6 | default | 1 × 40 | 0.06 / 0.05 | 0.00 / 0.01 | 0.000 | 1.000 | yes | yes | 2504 / 3096 | – |
| W1 | 796 | 10 | 1e-6 | rotation | 1 × 40 | 0.01 / 0.01 | 0.01 / 0.01 | 1.000 | 1.000 | yes | yes | 2504 / 3096 | – |
| W1 | 796 | 10 | 1e-8 | default | 1 × 40 | 0.10 / 0.11 | 0.01 / 0.01 | 1.000 | 0.957 | yes | yes | 2504 / 3096 | – |
| W1 | 796 | 10 | 1e-8 | rotation | 1 × 40 | 0.02 / 0.01 | 0.01 / 0.01 | 0.667 | 1.000 | yes | yes | 2504 / 3096 | – |
| W1 | 7162 | 30 | 1e-3 | default | 1 × 40 | 0.19 / 0.19 | 0.01 / 0.01 | 1.000 | 1.000 | yes | yes | 14720 / 35352 | – |
| W1 | 7162 | 30 | 1e-3 | rotation | 1 × 40 | 0.02 / 0.02 | 0.01 / 0.01 | 1.000 | 1.000 | yes | yes | 14720 / 35352 | – |
| W1 | 7162 | 30 | 1e-6 | default | 1 × 40 | 0.11 / 0.11 | 0.02 / 0.03 | 0.667 | 0.964 | yes | yes | 14720 / 35352 | – |
| W1 | 7162 | 30 | 1e-6 | rotation | 1 × 40 | 0.03 / 0.02 | 0.02 / 0.04 | 0.500 | 0.750 | yes | yes | 14720 / 35352 | – |
| W1 | 7162 | 30 | 1e-8 | default | 1 × 40 | 0.22 / 0.21 | 0.06 / 0.08 | 0.750 | 0.966 | yes | yes | 14720 / 35352 | – |
| W1 | 7162 | 30 | 1e-8 | rotation | 1 × 40 | 0.04 / 0.04 | 0.06 / 0.12 | 0.480 | 0.606 | yes | yes | 14720 / 35352 | – |
| W1 | 79578 | 100 | 1e-3 | default | 1 × 40 | 0.56 / 0.92 | 0.14 / 0.23 | 0.574 | 0.597 | no | yes | 184796 / 686736 | 4 |
| W1 | 79578 | 100 | 1e-3 | rotation | 1 × 40 | 0.29 / 0.21 | 0.14 / 0.35 | 0.394 | 0.768 | yes | yes | 184796 / 686736 | – |
| W1 | 79578 | 100 | 1e-6 | default | 1 × 40 | 0.57 / 0.47 | 0.35 / 0.54 | 0.648 | 0.906 | yes | yes | 184796 / 686736 | – |
| W1 | 79578 | 100 | 1e-6 | rotation | 1 × 40 | 0.29 / 0.21 | 0.35 / 0.88 | 0.403 | 0.594 | yes | yes | 184796 / 686736 | – |
| W1 | 79578 | 100 | 1e-8 | default | 1 × 40 | 0.78 / 0.68 | 1.07 / 1.72 | 0.622 | 0.769 | yes | yes | 184796 / 686736 | – |
| W1 | 79578 | 100 | 1e-8 | rotation | 1 × 40 | 0.31 / 0.23 | 0.93 / 2.52 | 0.366 | 0.449 | yes | yes | 184796 / 686736 | – |
| W3 (N = 108849) | 108849 | 30 | 1e-3 | default | 1 × 40 | 0.47 / 0.39 | 0.07 / 0.07 | 1.000 | 1.185 | yes | yes | 640584 / 640584 | – |
| W3 (N = 108849) | 108849 | 30 | 1e-3 | rotation | 1 × 40 | 0.23 / 0.15 | 0.12 / 0.12 | 1.000 | 1.273 | yes | yes | 640584 / 640584 | – |
| W3 (N = 108849) | 108849 | 30 | 1e-6 | default | 1 × 40 | 0.36 / 0.27 | 0.21 / 0.22 | 0.977 | 1.173 | yes | yes | 640584 / 640584 | – |
| W3 (N = 108849) | 108849 | 30 | 1e-6 | rotation | 1 × 40 | 0.23 / 0.15 | 0.58 / 0.58 | 1.009 | 1.109 | yes | yes | 640584 / 640584 | – |
| W3 (N = 108849) | 108849 | 30 | 1e-8 | default | 1 × 40 | 0.51 / 0.41 | 0.69 / 0.67 | 1.022 | 1.097 | yes | yes | 640584 / 640584 | – |
| W3 (N = 108849) | 108849 | 30 | 1e-8 | rotation | 1 × 40 | 0.25 / 0.17 | 2.01 / 2.00 | 1.005 | 1.039 | yes | yes | 640584 / 640584 | – |
| W3 (N = 4032) | 4032 | 10 | 1e-3 | default | 1 × 40 | 0.16 / 0.15 | 0.02 / 0.02 | 1.000 | 1.059 | yes | yes | 8640 / 8640 | – |
| W3 (N = 4032) | 4032 | 10 | 1e-3 | rotation | 1 × 40 | 0.01 / 0.01 | 0.02 / 0.01 | 2.000 | 1.500 | yes | yes | 8640 / 8640 | – |
| W3 (N = 4032) | 4032 | 10 | 1e-6 | default | 1 × 40 | 0.08 / 0.08 | 0.04 / 0.04 | 1.000 | 1.000 | yes | yes | 8640 / 8640 | – |
| W3 (N = 4032) | 4032 | 10 | 1e-6 | rotation | 1 × 40 | 0.02 / 0.01 | 0.04 / 0.04 | 1.000 | 1.091 | yes | yes | 8640 / 8640 | – |
| W3 (N = 4032) | 4032 | 10 | 1e-8 | default | 1 × 40 | 0.18 / 0.18 | 0.14 / 0.14 | 1.000 | 1.000 | yes | yes | 8640 / 8640 | – |
| W3 (N = 4032) | 4032 | 10 | 1e-8 | rotation | 1 × 40 | 0.03 / 0.03 | 0.10 / 0.10 | 1.000 | 1.000 | yes | yes | 8640 / 8640 | – |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-3 | default | 1 × 40 | 0.67 / 0.72 | 0.20 / 0.21 | 0.952 | 0.925 | no | yes | 764832 / 1028718 | 2, 3, 4 |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-3 | rotation | 1 × 40 | 0.40 / 0.24 | 0.25 / 0.29 | 0.877 | 1.229 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-6 | default | 1 × 40 | 0.58 / 0.44 | 0.65 / 0.70 | 0.921 | 1.075 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-6 | rotation | 1 × 40 | 0.40 / 0.25 | 1.06 / 1.27 | 0.827 | 0.954 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-8 | default | 1 × 40 | 0.94 / 0.79 | 2.46 / 2.67 | 0.923 | 0.986 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-8 | rotation | 1 × 40 | 0.45 / 0.30 | 3.25 / 4.12 | 0.787 | 0.835 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-3 | default | 1 × 40 | 0.66 / 0.72 | 0.20 / 0.21 | 0.929 | 0.919 | no | yes | 764832 / 1028718 | 2, 3, 4 |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-3 | rotation | 1 × 40 | 0.40 / 0.23 | 0.24 / 0.29 | 0.828 | 1.219 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-6 | default | 1 × 40 | 0.58 / 0.45 | 0.66 / 0.72 | 0.903 | 1.060 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-6 | rotation | 1 × 40 | 0.40 / 0.24 | 1.08 / 1.29 | 0.837 | 0.961 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-8 | default | 1 × 40 | 0.93 / 0.79 | 2.39 / 2.63 | 0.909 | 0.972 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-8 | rotation | 1 × 40 | 0.44 / 0.29 | 3.37 / 4.04 | 0.834 | 0.881 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-3 | default | 1 × 40 | 0.80 / 0.77 | 0.20 / 0.24 | 0.833 | 0.990 | no | yes | 764832 / 1028718 | 4 |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-3 | rotation | 1 × 40 | 0.40 / 0.25 | 0.26 / 0.30 | 0.836 | 1.171 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-6 | default | 1 × 40 | 0.68 / 0.53 | 0.69 / 0.77 | 0.895 | 1.050 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-6 | rotation | 1 × 40 | 0.41 / 0.26 | 1.06 / 1.31 | 0.805 | 0.933 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-8 | default | 1 × 40 | 1.02 / 0.86 | 2.43 / 2.66 | 0.915 | 0.980 | yes | yes | 764832 / 1028718 | – |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-8 | rotation | 1 × 40 | 0.45 / 0.30 | 3.27 / 4.09 | 0.801 | 0.846 | yes | yes | 764832 / 1028718 | – |
| W4 cube (N = 100000) | 100000 | 1 | 1e-3 | default | 1 × 40 | 0.43 / 0.35 | 0.06 / 0.06 | 1.000 | 1.193 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 1 | 1e-3 | rotation | 1 × 40 | 0.23 / 0.15 | 0.11 / 0.11 | 1.045 | 1.327 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 1 | 1e-6 | default | 1 × 40 | 0.33 / 0.24 | 0.21 / 0.20 | 1.051 | 1.205 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 1 | 1e-6 | rotation | 1 × 40 | 0.23 / 0.15 | 0.57 / 0.57 | 1.000 | 1.110 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 1 | 1e-8 | default | 1 × 40 | 0.46 / 0.39 | 0.62 / 0.63 | 0.984 | 1.059 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 1 | 1e-8 | rotation | 1 × 40 | 0.24 / 0.17 | 1.99 / 2.00 | 0.997 | 1.032 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 10 | 1e-3 | default | 1 × 40 | 0.43 / 0.35 | 0.06 / 0.06 | 1.000 | 1.193 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 10 | 1e-3 | rotation | 1 × 40 | 0.23 / 0.14 | 0.11 / 0.11 | 0.957 | 1.333 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 10 | 1e-6 | default | 1 × 40 | 0.33 / 0.24 | 0.20 / 0.20 | 1.000 | 1.180 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 10 | 1e-6 | rotation | 1 × 40 | 0.23 / 0.14 | 0.58 / 0.57 | 1.009 | 1.133 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 10 | 1e-8 | default | 1 × 40 | 0.47 / 0.40 | 0.60 / 0.62 | 0.968 | 1.049 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 10 | 1e-8 | rotation | 1 × 40 | 0.25 / 0.17 | 2.00 / 2.00 | 1.000 | 1.037 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 100 | 1e-3 | default | 1 × 40 | 0.91 / 0.82 | 0.23 / 0.23 | 1.000 | 1.081 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 100 | 1e-3 | rotation | 1 × 40 | 0.25 / 0.17 | 0.36 / 0.36 | 0.986 | 1.151 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 100 | 1e-6 | default | 1 × 40 | 0.50 / 0.41 | 0.32 / 0.33 | 0.985 | 1.116 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 100 | 1e-6 | rotation | 1 × 40 | 0.26 / 0.18 | 0.71 / 0.70 | 1.007 | 1.096 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 100 | 1e-8 | default | 1 × 40 | 0.66 / 0.56 | 0.84 / 0.83 | 1.018 | 1.075 | yes | yes | 640584 / 640584 | – |
| W4 cube (N = 100000) | 100000 | 100 | 1e-8 | rotation | 1 × 40 | 0.27 / 0.21 | 2.04 / 2.02 | 1.007 | 1.036 | yes | yes | 640584 / 640584 | – |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-3 | default | 1 × 40 | 0.79 / 0.67 | 0.21 / 0.20 | 1.025 | 1.144 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-3 | rotation | 1 × 40 | 0.30 / 0.18 | 0.27 / 0.26 | 1.038 | 1.281 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-6 | default | 1 × 40 | 0.49 / 0.39 | 0.64 / 0.63 | 1.008 | 1.113 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-6 | rotation | 1 × 40 | 0.30 / 0.20 | 1.07 / 1.10 | 0.977 | 1.066 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-8 | default | 1 × 40 | 0.83 / 0.72 | 2.29 / 2.33 | 0.987 | 1.026 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-8 | rotation | 1 × 40 | 0.34 / 0.23 | 3.26 / 3.35 | 0.973 | 1.007 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-3 | default | 1 × 40 | 0.79 / 0.67 | 0.20 / 0.21 | 0.952 | 1.125 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-3 | rotation | 1 × 40 | 0.29 / 0.18 | 0.26 / 0.27 | 0.962 | 1.236 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-6 | default | 1 × 40 | 0.51 / 0.40 | 0.64 / 0.64 | 1.000 | 1.111 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-6 | rotation | 1 × 40 | 0.30 / 0.19 | 1.06 / 1.10 | 0.968 | 1.058 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-8 | default | 1 × 40 | 0.82 / 0.71 | 2.33 / 2.31 | 1.011 | 1.046 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-8 | rotation | 1 × 40 | 0.34 / 0.23 | 3.29 / 3.37 | 0.978 | 1.011 | yes | yes | 770660 / 793284 | – |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-3 | default | 1 × 40 | 1.33 / 1.19 | 0.32 / 0.35 | 0.900 | 1.065 | yes | yes | 918828 / 1120110 | – |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-3 | rotation | 1 × 40 | 0.40 / 0.26 | 0.39 / 0.49 | 0.788 | 1.053 | yes | yes | 918828 / 1120110 | – |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-6 | default | 1 × 40 | 0.77 / 0.63 | 0.74 / 0.80 | 0.937 | 1.063 | yes | yes | 918828 / 1120110 | – |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-6 | rotation | 1 × 40 | 0.42 / 0.27 | 1.25 / 1.44 | 0.872 | 0.985 | yes | yes | 918828 / 1120110 | – |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-8 | default | 1 × 40 | 1.10 / 0.96 | 2.49 / 2.62 | 0.950 | 1.003 | yes | yes | 918828 / 1120110 | – |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-8 | rotation | 1 × 40 | 0.46 / 0.32 | 3.71 / 4.29 | 0.864 | 0.903 | yes | yes | 918828 / 1120110 | – |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-3 | default | 1 × 40 | 0.57 / 0.48 | 0.08 / 0.11 | 0.727 | 1.110 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-3 | rotation | 1 × 40 | 0.29 / 0.20 | 0.08 / 0.17 | 0.471 | 1.014 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-6 | default | 1 × 40 | 0.42 / 0.32 | 0.27 / 0.43 | 0.628 | 0.920 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-6 | rotation | 1 × 40 | 0.31 / 0.20 | 0.32 / 0.81 | 0.389 | 0.619 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-8 | default | 1 × 40 | 0.61 / 0.51 | 0.97 / 1.55 | 0.626 | 0.769 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-8 | rotation | 1 × 40 | 0.33 / 0.23 | 0.92 / 2.70 | 0.341 | 0.427 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-3 | default | 1 × 40 | 0.58 / 0.47 | 0.08 / 0.11 | 0.682 | 1.120 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-3 | rotation | 1 × 40 | 0.30 / 0.20 | 0.09 / 0.17 | 0.514 | 1.040 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-6 | default | 1 × 40 | 0.42 / 0.33 | 0.29 / 0.43 | 0.655 | 0.928 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-6 | rotation | 1 × 40 | 0.30 / 0.20 | 0.32 / 0.80 | 0.394 | 0.615 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-8 | default | 1 × 40 | 0.61 / 0.52 | 0.99 / 1.53 | 0.650 | 0.787 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-8 | rotation | 1 × 40 | 0.32 / 0.23 | 0.94 / 2.69 | 0.348 | 0.429 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-3 | default | 1 × 40 | 0.59 / 0.94 | 0.14 / 0.24 | 0.583 | 0.619 | no | yes | 203730 / 767196 | 4 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-3 | rotation | 1 × 40 | 0.33 / 0.23 | 0.15 / 0.38 | 0.387 | 0.783 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-6 | default | 1 × 40 | 0.59 / 0.50 | 0.36 / 0.57 | 0.626 | 0.884 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-6 | rotation | 1 × 40 | 0.33 / 0.23 | 0.37 / 0.94 | 0.396 | 0.601 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-8 | default | 1 × 40 | 0.81 / 0.70 | 1.14 / 1.80 | 0.633 | 0.778 | yes | yes | 203730 / 767196 | – |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-8 | rotation | 1 × 40 | 0.35 / 0.26 | 0.94 / 2.73 | 0.347 | 0.436 | yes | yes | 203730 / 767196 | – |

The levels whose M2L takes at least a tenth of a rule-off evaluation of at least 0.5 s:

| Workload | N | kD | ε | M2L | level | ran on / off | M2L s on / off |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| W1 | 79578 | 100 | 1e-6 | default | 4 | planewave i2 / planewave i2 | 0.058 / 0.129 |
| W1 | 79578 | 100 | 1e-6 | default | 5 | planewave i2 / planewave i2 | 0.091 / 0.130 |
| W1 | 79578 | 100 | 1e-6 | rotation | 3 | rotation / rotation | 0.031 / 0.128 |
| W1 | 79578 | 100 | 1e-6 | rotation | 4 | rotation / rotation | 0.044 / 0.305 |
| W1 | 79578 | 100 | 1e-6 | rotation | 5 | rotation / rotation | 0.092 / 0.193 |
| W1 | 79578 | 100 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.152 / 0.385 |
| W1 | 79578 | 100 | 1e-8 | default | 5 | planewave i3 / planewave i3 | 0.311 / 0.441 |
| W1 | 79578 | 100 | 1e-8 | rotation | 4 | rotation / rotation | 0.165 / 1.113 |
| W1 | 79578 | 100 | 1e-8 | rotation | 5 | rotation / rotation | 0.334 / 0.709 |
| W3 (N = 108849) | 108849 | 30 | 1e-6 | rotation | 4 | rotation / rotation | 0.462 / 0.462 |
| W3 (N = 108849) | 108849 | 30 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.394 / 0.408 |
| W3 (N = 108849) | 108849 | 30 | 1e-8 | rotation | 4 | rotation / rotation | 1.688 / 1.689 |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-6 | default | 8 | planewave i2 / planewave i2 | 0.087 / 0.079 |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-6 | rotation | 6 | rotation / rotation | 0.091 / 0.173 |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-6 | rotation | 7 | rotation / rotation | 0.229 / 0.231 |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-6 | rotation | 8 | rotation / rotation | 0.282 / 0.281 |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-8 | default | 8 | planewave i3 / planewave i3 | 0.298 / 0.302 |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-8 | rotation | 6 | rotation / rotation | 0.330 / 0.628 |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-8 | rotation | 7 | rotation / rotation | 0.848 / 0.848 |
| W4 clusters (N = 100000) | 100000 | 1 | 1e-8 | rotation | 8 | rotation / rotation | 1.031 / 1.030 |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-6 | default | 8 | planewave i2 / planewave i2 | 0.084 / 0.084 |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-6 | rotation | 6 | rotation / rotation | 0.091 / 0.173 |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-6 | rotation | 7 | rotation / rotation | 0.231 / 0.231 |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-6 | rotation | 8 | rotation / rotation | 0.321 / 0.285 |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-8 | default | 8 | planewave i3 / planewave i3 | 0.289 / 0.301 |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-8 | rotation | 6 | rotation / rotation | 0.330 / 0.632 |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-8 | rotation | 7 | rotation / rotation | 0.844 / 0.851 |
| W4 clusters (N = 100000) | 100000 | 10 | 1e-8 | rotation | 8 | rotation / rotation | 1.200 / 1.026 |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-6 | default | 8 | planewave i2 / planewave i2 | 0.091 / 0.091 |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-6 | rotation | 6 | rotation / rotation | 0.091 / 0.172 |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-6 | rotation | 7 | rotation / rotation | 0.250 / 0.231 |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-6 | rotation | 8 | rotation / rotation | 0.279 / 0.281 |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-8 | default | 8 | planewave i3 / planewave i3 | 0.288 / 0.286 |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-8 | rotation | 6 | rotation / rotation | 0.328 / 0.631 |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-8 | rotation | 7 | rotation / rotation | 0.852 / 0.930 |
| W4 clusters (N = 100000) | 100000 | 100 | 1e-8 | rotation | 8 | rotation / rotation | 1.047 / 1.027 |
| W4 cube (N = 100000) | 100000 | 1 | 1e-6 | rotation | 4 | rotation / rotation | 0.462 / 0.461 |
| W4 cube (N = 100000) | 100000 | 1 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.374 / 0.377 |
| W4 cube (N = 100000) | 100000 | 1 | 1e-8 | rotation | 4 | rotation / rotation | 1.700 / 1.700 |
| W4 cube (N = 100000) | 100000 | 10 | 1e-6 | rotation | 4 | rotation / rotation | 0.461 / 0.461 |
| W4 cube (N = 100000) | 100000 | 10 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.359 / 0.360 |
| W4 cube (N = 100000) | 100000 | 10 | 1e-8 | rotation | 4 | rotation / rotation | 1.702 / 1.704 |
| W4 cube (N = 100000) | 100000 | 100 | 1e-6 | rotation | 3 | rotation / rotation | 0.131 / 0.130 |
| W4 cube (N = 100000) | 100000 | 100 | 1e-6 | rotation | 4 | rotation / rotation | 0.461 / 0.460 |
| W4 cube (N = 100000) | 100000 | 100 | 1e-8 | default | 3 | planewave i3 / planewave i3 | 0.096 / 0.096 |
| W4 cube (N = 100000) | 100000 | 100 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.506 / 0.501 |
| W4 cube (N = 100000) | 100000 | 100 | 1e-8 | rotation | 4 | rotation / rotation | 1.703 / 1.698 |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-6 | default | 7 | planewave i2 / planewave i2 | 0.067 / 0.067 |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-6 | rotation | 6 | rotation / rotation | 0.164 / 0.165 |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-6 | rotation | 7 | rotation / rotation | 0.265 / 0.265 |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-8 | default | 7 | planewave i3 / planewave i3 | 0.245 / 0.245 |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-8 | rotation | 6 | rotation / rotation | 0.603 / 0.604 |
| W4 plummer (N = 100000) | 100000 | 1 | 1e-8 | rotation | 7 | rotation / rotation | 0.974 / 0.973 |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-6 | default | 7 | planewave i2 / planewave i2 | 0.073 / 0.075 |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-6 | rotation | 6 | rotation / rotation | 0.165 / 0.166 |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-6 | rotation | 7 | rotation / rotation | 0.265 / 0.268 |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-8 | default | 7 | planewave i3 / planewave i3 | 0.244 / 0.234 |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-8 | rotation | 6 | rotation / rotation | 0.608 / 0.611 |
| W4 plummer (N = 100000) | 100000 | 10 | 1e-8 | rotation | 7 | rotation / rotation | 0.976 / 0.987 |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-6 | default | 4 | planewave i2 / planewave i2 | 0.087 / 0.115 |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-6 | rotation | 4 | rotation / rotation | 0.164 / 0.301 |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-6 | rotation | 6 | rotation / rotation | 0.165 / 0.167 |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-6 | rotation | 7 | rotation / rotation | 0.265 / 0.267 |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.248 / 0.345 |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-8 | rotation | 4 | rotation / rotation | 0.589 / 1.106 |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-8 | rotation | 6 | rotation / rotation | 0.602 / 0.608 |
| W4 plummer (N = 100000) | 100000 | 100 | 1e-8 | rotation | 7 | rotation / rotation | 0.974 / 0.975 |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-6 | rotation | 4 | rotation / rotation | 0.043 / 0.305 |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-6 | rotation | 5 | rotation / rotation | 0.107 / 0.257 |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.112 / 0.287 |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-8 | default | 5 | planewave i3 / planewave i3 | 0.294 / 0.480 |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-8 | rotation | 4 | rotation / rotation | 0.162 / 1.119 |
| W4 sphere (N = 100000) | 100000 | 1 | 1e-8 | rotation | 5 | rotation / rotation | 0.399 / 0.944 |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-6 | rotation | 4 | rotation / rotation | 0.044 / 0.305 |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-6 | rotation | 5 | rotation / rotation | 0.107 / 0.256 |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.111 / 0.276 |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-8 | default | 5 | planewave i3 / planewave i3 | 0.308 / 0.481 |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-8 | rotation | 4 | rotation / rotation | 0.158 / 1.120 |
| W4 sphere (N = 100000) | 100000 | 10 | 1e-8 | rotation | 5 | rotation / rotation | 0.391 / 0.946 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-6 | default | 4 | planewave i2 / planewave i2 | 0.054 / 0.132 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-6 | default | 5 | planewave i2 / planewave i2 | 0.098 / 0.156 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-6 | rotation | 3 | rotation / rotation | 0.032 / 0.130 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-6 | rotation | 4 | rotation / rotation | 0.043 / 0.305 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-6 | rotation | 5 | rotation / rotation | 0.108 / 0.256 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.160 / 0.398 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-8 | default | 5 | planewave i3 / planewave i3 | 0.340 / 0.528 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-8 | rotation | 4 | rotation / rotation | 0.156 / 1.122 |
| W4 sphere (N = 100000) | 100000 | 100 | 1e-8 | rotation | 5 | rotation / rotation | 0.392 / 0.945 |

## 3. The large cases

### 3.1 W1 at kD = 500 and W3 at kD = 100 (one rank × 40 threads, job 633596)

| Workload | N | kD | ε | M2L | ranks × threads | build on / off s | evaluate on / off s | on ÷ off (evaluate) | on ÷ off (build + evaluate) | same checksum | same errors | V pairs on / off | levels whose M2L differs |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| W1 | 1989437 | 500 | 1e-6 | default | 1 × 40 | 14.55 / 11.00 | 9.88 / 23.14 | 0.427 | 0.716 | yes | yes | 4476464 / 28434444 | – |
| W1 | 1989437 | 500 | 1e-6 | rotation | 1 × 40 | 14.18 / 10.68 | 11.04 / 50.34 | 0.219 | 0.413 | yes | yes | 4476464 / 28434444 | – |
| W1 | 1989437 | 500 | 1e-8 | default | 1 × 40 | 18.98 / 15.33 | 21.01 / 47.53 | 0.442 | 0.636 | yes | yes | 4476464 / 28434444 | – |
| W1 | 1989437 | 500 | 1e-8 | rotation | 1 × 40 | 14.62 / 11.27 | 21.82 / 108.84 | 0.200 | 0.303 | yes | yes | 4476464 / 28434444 | – |
| W3 (N = 4031442) | 4031442 | 100 | 1e-6 | default | 1 × 40 | 24.45 / 15.57 | 10.70 / 10.88 | 0.984 | 1.329 | yes | yes | 52337880 / 52337880 | – |
| W3 (N = 4031442) | 4031442 | 100 | 1e-6 | rotation | 1 × 40 | 23.94 / 15.30 | 44.28 / 44.25 | 1.001 | 1.145 | yes | yes | 52337880 / 52337880 | – |
| W3 (N = 4031442) | 4031442 | 100 | 1e-8 | default | 1 × 40 | 24.50 / 15.81 | 37.27 / 37.60 | 0.991 | 1.157 | yes | yes | 52337880 / 52337880 | – |
| W3 (N = 4031442) | 4031442 | 100 | 1e-8 | rotation | 1 × 40 | 23.82 / 15.16 | 166.91 / 158.20 | 1.055 | 1.100 | yes | yes | 52337880 / 52337880 | – |

| Workload | N | kD | ε | M2L | level | ran on / off | M2L s on / off |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| W1 | 1989437 | 500 | 1e-6 | default | 2 | rotation / rotation | 2.778 / 3.242 |
| W1 | 1989437 | 500 | 1e-6 | default | 3 | rotation / rotation | 1.175 / 4.993 |
| W1 | 1989437 | 500 | 1e-6 | default | 6 | planewave i2 / planewave i2 | 0.736 / 3.092 |
| W1 | 1989437 | 500 | 1e-6 | default | 7 | planewave i2 / planewave i2 | 2.037 / 4.637 |
| W1 | 1989437 | 500 | 1e-6 | rotation | 3 | rotation / rotation | 1.174 / 5.128 |
| W1 | 1989437 | 500 | 1e-6 | rotation | 4 | rotation / rotation | 0.781 / 6.953 |
| W1 | 1989437 | 500 | 1e-6 | rotation | 5 | rotation / rotation | 0.737 / 9.528 |
| W1 | 1989437 | 500 | 1e-6 | rotation | 6 | rotation / rotation | 0.826 / 11.758 |
| W1 | 1989437 | 500 | 1e-6 | rotation | 7 | rotation / rotation | 2.606 / 11.425 |
| W1 | 1989437 | 500 | 1e-8 | default | 3 | rotation / rotation | 1.414 / 6.208 |
| W1 | 1989437 | 500 | 1e-8 | default | 6 | planewave i3 / planewave i3 | 2.225 / 9.172 |
| W1 | 1989437 | 500 | 1e-8 | default | 7 | planewave i3 / planewave i3 | 6.707 / 14.741 |
| W1 | 1989437 | 500 | 1e-8 | rotation | 5 | rotation / rotation | 0.977 / 13.938 |
| W1 | 1989437 | 500 | 1e-8 | rotation | 6 | rotation / rotation | 2.561 / 30.209 |
| W1 | 1989437 | 500 | 1e-8 | rotation | 7 | rotation / rotation | 9.524 / 41.887 |
| W3 (N = 4031442) | 4031442 | 100 | 1e-6 | default | 6 | planewave i2 / planewave i2 | 6.704 / 6.792 |
| W3 (N = 4031442) | 4031442 | 100 | 1e-6 | rotation | 6 | rotation / rotation | 36.552 / 36.583 |
| W3 (N = 4031442) | 4031442 | 100 | 1e-8 | default | 6 | planewave i3 / planewave i3 | 24.350 / 24.582 |
| W3 (N = 4031442) | 4031442 | 100 | 1e-8 | rotation | 5 | rotation / rotation | 15.649 / 16.425 |
| W3 (N = 4031442) | 4031442 | 100 | 1e-8 | rotation | 6 | rotation / rotation | 134.375 / 134.193 |

W1 at kD = 500 (N = 1,989,437, 217,897 leaves): 150,933 boxes without targets; their V entries
17.4 million, W 1.7 million, X 0.95 million; V entries without sources 6.6 million. W3 at
N = 4·10⁶ and kD = 100 has no empty box.

### 3.2 The under-resolved W3 (N = 10⁶, one rank × 40 threads, job 633566)

One evaluation per build, one order (`on,off`).

| Workload | N | kD | ε | M2L | ranks × threads | build on / off s | evaluate on / off s | on ÷ off (evaluate) | on ÷ off (build + evaluate) | same checksum | same errors | V pairs on / off | levels whose M2L differs |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| W3 (N = 1000000) | 1000000 | 200 | 1e-6 | default | 1 × 40 | 3.49 / 2.37 | 2.51 / 2.47 | 1.016 | 1.240 | yes | yes | 6039504 / 6039504 | – |
| W3 (N = 1000000) | 1000000 | 200 | 1e-6 | rotation | 1 × 40 | 2.98 / 1.88 | 5.47 / 5.78 | 0.946 | 1.103 | yes | yes | 6039504 / 6039504 | – |
| W3 (N = 1000000) | 1000000 | 200 | 1e-8 | default | 1 × 40 | 4.16 / 3.11 | 6.44 / 6.50 | 0.991 | 1.103 | yes | yes | 6039504 / 6039504 | – |
| W3 (N = 1000000) | 1000000 | 200 | 1e-8 | rotation | 1 × 40 | 2.97 / 1.95 | 16.37 / 15.22 | 1.076 | 1.126 | yes | yes | 6039504 / 6039504 | – |
| W3 (N = 1000000) | 1000000 | 500 | 1e-6 | default | 1 × 40 | 139.02 / 92.62 | 59.62 / 106.71 | 0.559 | 0.997 | yes | yes | 105510916 / 418825452 | – |
| W3 (N = 1000000) | 1000000 | 500 | 1e-6 | rotation | 1 × 40 | 137.66 / 92.17 | 104.89 / 290.52 | 0.361 | 0.634 | yes | yes | 105510916 / 418825452 | – |
| W3 (N = 1000000) | 1000000 | 500 | 1e-8 | default | 1 × 40 | 144.23 / 97.32 | 167.89 / 329.29 | 0.510 | 0.732 | yes | yes | 105510916 / 418825452 | – |
| W3 (N = 1000000) | 1000000 | 500 | 1e-8 | rotation | 1 × 40 | 141.18 / 92.85 | 274.25 / 997.39 | 0.275 | 0.381 | yes | yes | 105510916 / 418825452 | – |

| Workload | N | kD | ε | M2L | level | ran on / off | M2L s on / off |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| W3 (N = 1000000) | 1000000 | 200 | 1e-6 | default | 4 | planewave i2 / planewave i2 | 0.361 / 0.356 |
| W3 (N = 1000000) | 1000000 | 200 | 1e-6 | default | 5 | planewave i2 / planewave i2 | 1.183 / 1.175 |
| W3 (N = 1000000) | 1000000 | 200 | 1e-6 | rotation | 4 | rotation / rotation | 1.004 / 0.998 |
| W3 (N = 1000000) | 1000000 | 200 | 1e-6 | rotation | 5 | rotation / rotation | 3.236 / 3.554 |
| W3 (N = 1000000) | 1000000 | 200 | 1e-8 | default | 4 | planewave i3 / planewave i3 | 0.677 / 0.704 |
| W3 (N = 1000000) | 1000000 | 200 | 1e-8 | default | 5 | planewave i3 / planewave i3 | 3.695 / 3.683 |
| W3 (N = 1000000) | 1000000 | 200 | 1e-8 | rotation | 5 | rotation / rotation | 13.256 / 12.139 |
| W3 (N = 1000000) | 1000000 | 500 | 1e-6 | default | 6 | planewave i2 / planewave i2 | 11.683 / 11.913 |
| W3 (N = 1000000) | 1000000 | 500 | 1e-6 | default | 7 | planewave i2 / planewave i2 | 27.086 / 72.388 |
| W3 (N = 1000000) | 1000000 | 500 | 1e-6 | rotation | 6 | rotation / rotation | 31.516 / 32.997 |
| W3 (N = 1000000) | 1000000 | 500 | 1e-6 | rotation | 7 | rotation / rotation | 38.087 / 220.029 |
| W3 (N = 1000000) | 1000000 | 500 | 1e-8 | default | 6 | planewave i3 / planewave i3 | 34.917 / 35.562 |
| W3 (N = 1000000) | 1000000 | 500 | 1e-8 | default | 7 | planewave i3 / planewave i3 | 87.277 / 234.611 |
| W3 (N = 1000000) | 1000000 | 500 | 1e-8 | rotation | 6 | rotation / rotation | 99.195 / 118.970 |
| W3 (N = 1000000) | 1000000 | 500 | 1e-8 | rotation | 7 | rotation / rotation | 123.684 / 820.199 |

At kD = 500 (2,056,727 leaves on level 7): 1,260,796 boxes without targets, their V entries
225 million, V entries without sources 88 million; V pairs 105.5 million against 418.8 million.
At kD = 200 the leaves lie on level 5 at about 30 points and no box is empty.

### 3.3 Four ranks (W3, N = 10⁶, kD = 500, 10⁻⁶, the default, 4 × 10, job 633564)

| rule | build s | evaluate s | V pairs | multipoles read from other ranks | checksum |
| --- | ---: | ---: | ---: | ---: | --- |
| on | 37.58 | 63.71 | 105,510,916 | 95,284 | b8196d7f1fe12151 |
| off | 25.94 | 116.25 | 418,825,452 | 175,800 | b8196d7f1fe12151 |

"Multipoles read from other ranks" is `Row::remote_multipoles`: per rank the distinct boxes its
V and W rows name that it does not compute, summed over the ranks, which the reduction, the
replicated top and the multipole exchange deliver.

## 4. Laplace: the plan's build time (job 633567)

`plan_build_cost` (one rank, the best of several builds per tree, ms), `main` (f32e5ba) and the
branch alternately on one node:

| tree | `main` runs 1–3 | branch runs 1–3 |
| --- | --- | --- |
| graded corner blob | 7.00, 7.09, 7.07 | 7.01, 7.15, 7.02 |
| 10⁵ uniform, level 4 | 80.10, 79.15, 80.47 | 79.54, 81.36, 80.69 |
| 10⁵ uniform, 32 per leaf | 94.69, 94.70, 94.29 | 94.76, 95.33, 94.41 |

Within the run-to-run spread of `main` (the branch's medians 7.02, 80.69 and 94.76 against
7.07, 80.10 and 94.69). The heap columns are the same.

## 5. Questions for sign-off

1. **The default.** (a) On, as built (recommended): the evaluation is never slower beyond
   noise and 0.2–0.66 of the rule off's where boxes are empty, and an FMM is built once and
   evaluated many times in the solvers it serves; with CH4.0d1b (one plan build) next on the
   priority list, before CH4.0d2. (b) Off until CH4.0d1b: no build cost on uniform trees now,
   and no gain either.
2. **CH4.0d1b, the rule's build cost**, added to the roadmap: build the rule's plan once
   (the counts' exchange lists without a first plan, or the rule applied to the first plan's
   rows), the build with the rule within 1.1 of the rule off's on W3 at N = 4·10⁶; it changes
   `nd-fmm-plan`'s interface again (a stop-and-ask when it starts).

## 6. Reproduce

```sh
tools/kathleen/sync.sh
tools/kathleen/submit.sh --time=06:00:00 -- run.sbatch \
    'cargo build --release -p nd-fmm-validate --example helmholtz_accuracy' \
    'tools/helmholtz/ch4-0d1-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch4-0d1/matrix matrix'
# likewise the stages large, w3 and traffic; then, locally:
python3 tools/helmholtz/ch4-0d1-tables.py ~/projects/fmm-archive/kathleen/logs/ch4-0d1/matrix2/*.md
```

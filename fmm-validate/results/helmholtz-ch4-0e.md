# Helmholtz CH4.0e: the comparison with FMM3D under the defaults, and H4's expected gain

Task: `docs/helmholtz/tasks/CH4.0e.md`. Roadmap: H4, CH4.0e (priority list item 5), added by
the triage of the review after the H6 gate (`docs/reviews/2026-11-12-H6.md` A1, B2, B5;
question 1 of #209). Proposal §9.1, §9.2, §3.6, §5.4, §7. Builds on CH3.6's matrix
(`fmm-validate/results/helmholtz-ch3-6.md`), the defaults of CH4.0d3
(`fmm-validate/results/helmholtz-ch4-0d3.md`) and S6's prototype times
(`spikes/diagonal/SPIKE_REPORT.md` §4, §8).

## 0. Summary

- **Ours under the defaults meets ε everywhere.** 152 launches per machine (CH3.6's 19 cases
  × 4 splits × 2 launches, 3 ε each): the largest error 0.35 ε on Kathleen and 0.40 ε on
  locust (∇φ, W3 at kD = 30, 10⁻⁶). Every case's checksum is the same at every split and in
  both launches on each machine (rank- and thread-count invariance at the defaults). FMM3D's
  errors and times at the requested ε reproduce CH3.6's to the digits shown (W1 at kD = 500:
  1.87e-4, 6.21e-8, 5.19e-11; 61.3 s against CH3.6's 62.9 s at 10⁻⁶).
- **Against FMM3D, each code on the whole node at its best split** (§2): our build plus one
  evaluation over FMM3D's whole call, the geometric mean over the 18 cases other than W1 at
  kD = 10 (FMM3D sums that one directly):

  | | Kathleen 10⁻³ | 10⁻⁶ | 10⁻⁸ | locust 10⁻³ | 10⁻⁶ | 10⁻⁸ |
  | --- | --- | --- | --- | --- | --- | --- |
  | `default`, same ε | 0.59 | 0.32 | 0.42 | 0.73 | 0.39 | 0.43 |
  | `default`, matched accuracy | 0.43 | 0.38 | 0.44 | 0.50 | 0.48 | 0.44 |
  | `fastker`, same ε | 1.16 | 0.61 | 0.90 | 0.72 | 0.26 | 0.38 |
  | `fastker`, matched accuracy | 0.78 | 0.69 | 0.93 | 0.30 | 0.27 | 0.38 |

  On the two cases of N ≥ 10⁶ ours is 0.18–0.50 of `fastker`'s time on Kathleen and 0.08–0.36
  on locust, at either measure: W1 at kD = 500 takes 9.9, 11.7 and 19.9 s on Kathleen (CH3.6:
  32, 52 and 100 s) against `fastker`'s 33.4, 43.5 and 49.1 s; our evaluation alone is 0.16–0.23
  of `fastker`'s whole call there.
- **Where FMM3D is still faster it is our build** (§2.3): on Kathleen against `fastker` at
  matched accuracy ours is slower on 7, 4 and 9 of the 18 cases at 10⁻³, 10⁻⁶ and 10⁻⁸, all of
  at most 0.8 s, where our build is 46–96 % of our time and our evaluation alone is faster than
  FMM3D's whole call in every one. On locust, where `fastker` is slow on the larger trees (W3
  at kD = 100: 38.5 s against `default`'s 11.7 s at 10⁻⁶, both launches), ours is slower at
  matched accuracy only on W3 at kD = 10 (4,032 points, ≤ 0.06 s) and W1 at kD = 30 at 10⁻⁶
  against `default`.
- **H4's expected gain is small inside D1** (§3). Of the 57 case × ε points only W1 at
  kD = 500 has levels with k a ≥ 8 on RCR (levels 2 and 3, p = 123–135 and 66–76); everywhere
  else `Auto` already runs those levels on the exponential M2L. On W1 at kD = 500 the bound
  (those levels' M2L free) is 14–31 % of build plus one evaluation (36–61 % of the evaluation),
  the estimate from S6's prototype times 6–14 % (15–28 % of the evaluation); H4's transfer
  functions add 0.2 s to the build if built in parallel over the classes and about 5.7 s if
  built serially. Ours is already 0.23–0.44× `fastker` there. At kD = 1000 (CH4.4, outside D1;
  §3.3) the bound is 27–42 % and the prototype estimate 9–17 %, with about 21 s of serial
  transfer-function build against it.
- **Two cheaper levers, found on the way** (§3.2): the **build**, which barely scales with
  threads (W1 at kD = 500: 5.5 s on one core, 4.6 s on 40; it grows with the degrees, not N:
  W3 at kD = 100 with 4·10⁶ points builds in 0.8–1.2 s) and is what FMM3D is faster on; and
  **level 2's Rotation M2L**, split by target box only (64 boxes) at 28–48 % parallel
  efficiency, its rotation blocks (601 MB at p = 131) formed per V pair under the 512 MiB
  default budget (storing them saves 18–22 % per pair and costs 0.1–1.3 s of build: net
  0.92–1.03×), and per pair 1.9–3.3× S6's scalar prototype on the same Grace core
  (p = 72–131).
- **Kathleen against locust (review B5), measured** (§4): per core the Grace core runs every
  M2L level 1.9–2.4× faster than a Cascade Lake core (W1 at kD = 500, 10⁻⁶, one core each:
  level 2 50.9 against 26.5 s, level 3 27.1 against 13.2 s, level 4 10.5 against 4.4 s); times
  1.8× the cores that is 3.4–4.3×, at the same parallel efficiency of the evaluation on both
  (54 %); the levels' own efficiencies spread it to the review's 3.3–5.4×. The evaluation is
  2.5× at the best splits, the build 1.7×.
- **Questions for sign-off (§7)**: (1) H4's go-ahead and order: recommended, H4 stays, after
  two new tasks for the build and the coarse RCR levels; (2) the block budget's default kept.

## 1. Method

- **Cases.** CH3.6's matrix: W1 at kD = 10, 30, 100, 500; W3 (ten points per wavelength) at
  kD = 10, 30, 100 (N = 4,032, 108,849, 4,031,442); W4's cube, sphere, Plummer sphere and
  clusters at N = 10⁵ and kD = 1, 10, 100; each at ε = 10⁻³, 10⁻⁶, 10⁻⁸; one complex charge
  vector, §9.1's, with potentials and gradients; the errors are §9.1's relative ℓ2 of φ and
  ∇φ at the 1,000 sampled targets against the f64 direct sum, for both codes against the same
  oracle (`helmholtz_accuracy::Oracle`; FMM3D's outputs through `helmholtz_fmm3d errors`, on
  the problem files `helmholtz_fmm3d problem` writes from the same cases).
- **Ours: the defaults as they stand after CH4.0d3.** `helmholtz_accuracy --m2l auto-pw
  --near-v cost --near-wx cost`, the leaf size and c_leaf the library's
  (`default_max_points_per_leaf`: x86 256; NEON 128 at ε ≥ 10⁻⁷ and 256 below; c_leaf 2), the
  calibrated degree rule, the empty-box rule, `Auto` with the exponential M2L (D8), the P2P
  `Auto` (AVX-512 on Kathleen, NEON on locust); release, BLAS threads 1. One build, two
  evaluations; the time is the build plus the second evaluation ("build plus one
  evaluation"), the build and the evaluation also apart.
- **Ours at its best split.** Every case at four splits of the node into ranks × threads,
  Kathleen 1 × 40, 2 × 20, 4 × 10, 8 × 5 (`mpirun --map-by ppr:<r/2>:socket:PE=<t> --bind-to
  core`, 1 × 40 by `slot:PE=40`), locust 1 × 72, 2 × 36, 4 × 18, 8 × 9 (`slot:PE=<t>`), in
  that order and again reversed (launches a and b); per case and ε the split and launch with
  the smallest build plus one evaluation.
- **FMM3D.** v2.1.0 (`70455e0`, T9's pin), `default` and `fastker`, built as for CH3.6 (gcc
  12.3.0 on Kathleen, the system's GCC 11.4.1 on locust, `-O3 -march=native`). `hfmm3d_s_c_g`
  through `tools/fmm3d/driver.f90`'s `helmholtz` mode on every core (OpenMP, its defaults),
  over the eps grid 10⁻³, 3·10⁻⁴, …, 3·10⁻⁹, 10⁻⁹ (1-3 per decade, the requested ε
  included), each eps the best of two calls (REPEAT 2; FMM3D's setup is in every call),
  launches a and b with the variants in opposite order; the time is the smaller of the two
  launches. FMM3D keeps nothing between calls, so its time is the whole call; ours is given
  as build plus one evaluation and apart.
- **Matched accuracy.** Per case, ε and variant, the fastest grid eps whose φ and ∇φ errors
  are both at most ours (FMM3D at least as accurate in both fields); `>` marks a row where no
  grid eps is, and the most accurate grid eps is shown. The match moves in FMM3D's steps
  (its order and quadrature tables), so it is as close as the grid allows.
- **One core** (`t1`): ours at 1 × 1 and both FMM3D variants on one thread on W1 at
  kD = 100 (CH3.6's one-thread case), and ours on W1 at kD = 500 at 10⁻⁶, for the per-core
  part of the comparison of the machines (§4).
- **Machines.** Kathleen: one exclusive node per job (2 × 20 Cascade Lake cores, Xeon Gold
  6248, `singlenode`), jobs and nodes in §6. locust: GH200, 72 Neoverse V2 cores, the load
  recorded before and after every launch.
- **The estimate (§3).** Per case and ε at the best split, from `--levels` (each level's M2L
  that ran, its V pairs, boxes and seconds): the levels with k a_l ≥ 8 (a_l = √3 w_l / 2;
  below it S6 sends most classes to RCR at ε ≤ 10⁻⁶ whatever the rule) whose M2L runs RCR
  (Rotation or Batched; the exponential M2L's levels are low-frequency and stay). Of each,
  the share of V pairs the diagonal form admits, from S6's table for the calibrated p at the
  level's (k a, ε) (`spikes/diagonal/results/summary.md` §3, "real, calibrated"); then
  - **the bound**: that share of the level's M2L time removed entirely (diagonal pairs and
    the conversions free), an upper bound of H4's gain on our trees;
  - **the prototype estimate**: that share at the cost of S6's prototype diagonal pair plus
    S → F and F → R per box (`spikes/diagonal/results/times.md`, run 1, interpolated in log p),
    scaled by the level's measured RCR seconds over S6's prototype RCR pair on its pairs (so
    the scale carries our vectorisation and threads);
  - **the ceiling**: every level's M2L with k a_l ≥ 8 removed, the exponential M2L's included.
  None includes H4's scratch traffic (proposal §7) or the class-permuted transfer functions'
  reads, which the prototypes do not have.

## 2. The comparison

### 2.1 Kathleen

Per case and ε: ours at its best split (ranks × threads), the leaf size the library chose,
build, evaluation (the second of the build's two) and both in seconds, its errors; FMM3D's
whole call at the same ε for each variant and ours over it (÷); FMM3D at matched accuracy: the
grid eps, its seconds and ours over it (`>`: no grid eps is as accurate in both fields, the
most accurate one shown). W1 at kD = 10 (796 points) is summed directly by FMM3D and holds
one leaf in ours (errors 10⁻¹⁵ for both).

| Case | ε | split | leaf | build s | evaluate s | ours s | φ, ∇φ | default s | ÷ | fastker s | ÷ | default matched: eps, s, ÷ | fastker matched: eps, s, ÷ |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| w1-10 | 1e-3 | 4x10 | 256 | 0.004 | 0.001 | 0.005 | 1.44e-15, 1.74e-15 | 0.008 | 0.63 | 0.003 | 1.87 | 3e-4, 0.007, 0.75 | 1e-3, 0.003, 1.87 |
| w1-10 | 1e-6 | 4x10 | 256 | 0.002 | 0.001 | 0.003 | 1.44e-15, 1.74e-15 | 0.007 | 0.42 | 0.014 | 0.22 | 3e-4, 0.007, 0.44 | 1e-3, 0.003, 1.10 |
| w1-10 | 1e-8 | 8x5 | 256 | 0.002 | 0.001 | 0.003 | 1.44e-15, 1.74e-15 | 0.008 | 0.38 | 0.016 | 0.18 | 3e-4, 0.007, 0.43 | 1e-3, 0.003, 1.06 |
| w1-30 | 1e-3 | 4x10 | 256 | 0.060 | 0.010 | 0.070 | 1.30e-05, 2.26e-05 | 0.054 | 1.31 | 0.025 | 2.87 | 3e-4, 0.058, 1.20 | 3e-4, 0.052, 1.37 |
| w1-30 | 1e-6 | 4x10 | 256 | 0.055 | 0.013 | 0.068 | 1.08e-07, 2.50e-07 | 0.073 | 0.93 | 0.054 | 1.26 | 3e-4, 0.058, 1.16 | 3e-4, 0.052, 1.32 |
| w1-30 | 1e-8 | 2x20 | 256 | 0.107 | 0.025 | 0.132 | 5.59e-12, 6.48e-12 | 0.109 | 1.22 | 0.058 | 2.29 | >1e-9, 0.118, 1.12 | 3e-4, 0.052, 2.57 |
| w1-100 | 1e-3 | 2x20 | 256 | 0.252 | 0.095 | 0.347 | 1.17e-05, 2.59e-05 | 0.919 | 0.38 | 0.417 | 0.83 | 3e-4, 0.725, 0.48 | 3e-4, 0.422, 0.82 |
| w1-100 | 1e-6 | 2x20 | 256 | 0.288 | 0.154 | 0.442 | 7.89e-08, 2.27e-07 | 0.907 | 0.49 | 0.488 | 0.90 | 3e-7, 1.346, 0.33 | 3e-7, 0.591, 0.75 |
| w1-100 | 1e-8 | 2x20 | 256 | 0.430 | 0.324 | 0.754 | 6.28e-10, 8.74e-10 | 1.517 | 0.50 | 0.657 | 1.15 | 3e-7, 1.346, 0.56 | 3e-7, 0.591, 1.28 |
| w1-500 | 1e-3 | 2x20 | 256 | 3.972 | 5.879 | 9.851 | 1.17e-05, 2.10e-05 | 49.378 | 0.20 | 33.434 | 0.29 | 3e-4, 48.650, 0.20 | 3e-4, 34.151, 0.29 |
| w1-500 | 1e-6 | 2x20 | 256 | 4.576 | 7.138 | 11.714 | 1.36e-07, 3.18e-07 | 61.334 | 0.19 | 43.482 | 0.27 | 3e-4, 48.650, 0.24 | 3e-4, 34.151, 0.34 |
| w1-500 | 1e-8 | 2x20 | 256 | 8.575 | 11.312 | 19.886 | 1.62e-09, 3.21e-09 | 69.888 | 0.28 | 49.083 | 0.41 | 3e-7, 63.935, 0.31 | 3e-7, 45.171, 0.44 |
| w3-10 | 1e-3 | 2x20 | 256 | 0.073 | 0.003 | 0.076 | 3.88e-05, 6.58e-05 | 0.021 | 3.64 | 0.011 | 6.75 | 1e-3, 0.021, 3.64 | 1e-3, 0.011, 6.75 |
| w3-10 | 1e-6 | 4x10 | 256 | 0.047 | 0.010 | 0.057 | 4.01e-08, 1.24e-07 | 0.039 | 1.45 | 0.022 | 2.63 | 3e-5, 0.029, 1.94 | 3e-4, 0.020, 2.89 |
| w3-10 | 1e-8 | 4x10 | 256 | 0.088 | 0.018 | 0.106 | 2.16e-12, 1.24e-12 | 0.068 | 1.55 | 0.024 | 4.42 | >3e-9, 0.074, 1.43 | 3e-4, 0.020, 5.36 |
| w3-30 | 1e-3 | 4x10 | 256 | 0.184 | 0.090 | 0.274 | 4.07e-05, 8.96e-05 | 0.349 | 0.79 | 0.212 | 1.30 | 3e-4, 0.702, 0.39 | 3e-4, 0.243, 1.13 |
| w3-30 | 1e-6 | 4x10 | 256 | 0.141 | 0.118 | 0.259 | 1.39e-07, 3.53e-07 | 0.948 | 0.27 | 0.307 | 0.84 | 3e-4, 0.702, 0.37 | 3e-4, 0.243, 1.06 |
| w3-30 | 1e-8 | 4x10 | 256 | 0.263 | 0.276 | 0.539 | 5.04e-12, 6.99e-12 | 1.279 | 0.42 | 0.446 | 1.21 | >1e-9, 1.333, 0.40 | >1e-9, 0.501, 1.08 |
| w3-100 | 1e-3 | 8x5 | 256 | 1.243 | 2.398 | 3.641 | 4.76e-05, 9.67e-05 | 22.046 | 0.17 | 7.290 | 0.50 | 3e-4, 29.285, 0.12 | 3e-4, 20.082, 0.18 |
| w3-100 | 1e-6 | 8x5 | 256 | 0.822 | 4.191 | 5.013 | 1.01e-07, 2.10e-07 | 32.031 | 0.16 | 22.533 | 0.22 | 3e-7, 35.584, 0.14 | 3e-7, 25.228, 0.20 |
| w3-100 | 1e-8 | 8x5 | 256 | 1.022 | 10.710 | 11.732 | 1.19e-09, 1.82e-09 | 38.382 | 0.31 | 27.596 | 0.43 | 3e-7, 35.584, 0.33 | 3e-7, 25.228, 0.47 |
| w4-cube-1 | 1e-3 | 4x10 | 256 | 0.150 | 0.066 | 0.216 | 3.57e-05, 5.06e-05 | 0.218 | 0.99 | 0.154 | 1.40 | 3e-4, 0.690, 0.31 | 3e-4, 0.174, 1.24 |
| w4-cube-1 | 1e-6 | 4x10 | 256 | 0.084 | 0.101 | 0.185 | 5.47e-08, 1.34e-07 | 0.842 | 0.22 | 0.223 | 0.83 | 3e-7, 0.817, 0.23 | 1e-5, 0.198, 0.94 |
| w4-cube-1 | 1e-8 | 2x20 | 256 | 0.180 | 0.210 | 0.390 | 1.49e-12, 3.74e-12 | 0.879 | 0.44 | 0.331 | 1.18 | >3e-8, 0.858, 0.45 | >3e-8, 0.309, 1.26 |
| w4-cube-10 | 1e-3 | 4x10 | 256 | 0.151 | 0.067 | 0.218 | 4.67e-05, 6.56e-05 | 0.241 | 0.90 | 0.156 | 1.40 | 3e-4, 0.773, 0.28 | 3e-4, 0.183, 1.19 |
| w4-cube-10 | 1e-6 | 4x10 | 256 | 0.092 | 0.102 | 0.194 | 5.47e-08, 1.44e-07 | 0.930 | 0.21 | 0.230 | 0.84 | 1e-4, 0.788, 0.25 | 1e-4, 0.191, 1.02 |
| w4-cube-10 | 1e-8 | 2x20 | 256 | 0.189 | 0.210 | 0.399 | 1.61e-11, 4.61e-12 | 1.007 | 0.40 | 0.349 | 1.14 | >1e-9, 1.054, 0.38 | >1e-9, 0.392, 1.02 |
| w4-cube-100 | 1e-3 | 2x20 | 256 | 0.245 | 0.154 | 0.399 | 7.42e-06, 1.06e-05 | 0.699 | 0.57 | 0.381 | 1.05 | 3e-4, 1.295, 0.31 | 3e-4, 0.450, 0.89 |
| w4-cube-100 | 1e-6 | 2x20 | 256 | 0.242 | 0.183 | 0.425 | 3.53e-08, 5.70e-08 | 1.568 | 0.27 | 0.541 | 0.79 | 3e-7, 1.374, 0.31 | 3e-7, 0.651, 0.65 |
| w4-cube-100 | 1e-8 | 2x20 | 256 | 0.337 | 0.294 | 0.631 | 8.42e-10, 1.27e-09 | 1.452 | 0.43 | 0.729 | 0.86 | 3e-7, 1.374, 0.46 | 3e-7, 0.651, 0.97 |
| w4-sphere-1 | 1e-3 | 8x5 | 256 | 0.223 | 0.042 | 0.265 | 2.05e-05, 4.23e-07 | 0.371 | 0.71 | 0.135 | 1.97 | 3e-4, 0.390, 0.68 | 3e-4, 0.250, 1.06 |
| w4-sphere-1 | 1e-6 | 4x10 | 256 | 0.120 | 0.096 | 0.216 | 3.71e-08, 1.11e-09 | 0.510 | 0.42 | 0.297 | 0.73 | 1e-4, 0.413, 0.52 | 1e-4, 0.259, 0.83 |
| w4-sphere-1 | 1e-8 | 4x10 | 256 | 0.239 | 0.268 | 0.507 | 9.44e-13, 2.25e-12 | 1.081 | 0.47 | 0.397 | 1.28 | >1e-8, 1.081, 0.47 | >1e-8, 0.397, 1.28 |
| w4-sphere-10 | 1e-3 | 4x10 | 256 | 0.219 | 0.044 | 0.263 | 2.68e-05, 4.77e-07 | 0.384 | 0.68 | 0.147 | 1.79 | 3e-4, 0.421, 0.62 | 3e-4, 0.254, 1.03 |
| w4-sphere-10 | 1e-6 | 4x10 | 256 | 0.127 | 0.102 | 0.229 | 3.97e-08, 1.18e-09 | 0.537 | 0.43 | 0.302 | 0.76 | 1e-4, 0.438, 0.52 | 3e-4, 0.254, 0.90 |
| w4-sphere-10 | 1e-8 | 4x10 | 256 | 0.242 | 0.269 | 0.511 | 6.73e-12, 2.25e-12 | 1.133 | 0.45 | 0.413 | 1.24 | >1e-9, 1.263, 0.40 | >1e-9, 0.459, 1.11 |
| w4-sphere-100 | 1e-3 | 4x10 | 256 | 0.240 | 0.117 | 0.357 | 7.27e-06, 3.15e-07 | 0.985 | 0.36 | 0.433 | 0.83 | 3e-4, 0.863, 0.41 | 3e-4, 0.502, 0.71 |
| w4-sphere-100 | 1e-6 | 4x10 | 256 | 0.272 | 0.175 | 0.447 | 5.47e-08, 3.01e-09 | 1.058 | 0.42 | 0.587 | 0.76 | 3e-7, 1.589, 0.28 | 3e-7, 0.686, 0.65 |
| w4-sphere-100 | 1e-8 | 4x10 | 256 | 0.431 | 0.368 | 0.799 | 4.36e-10, 1.50e-11 | 1.786 | 0.45 | 0.759 | 1.05 | 3e-7, 1.589, 0.50 | 3e-7, 0.686, 1.17 |
| w4-plummer-1 | 1e-3 | 4x10 | 256 | 0.418 | 0.098 | 0.516 | 6.18e-05, 7.52e-05 | 0.718 | 0.72 | 0.398 | 1.30 | 3e-4, 1.066, 0.48 | 3e-4, 0.767, 0.67 |
| w4-plummer-1 | 1e-6 | 4x10 | 256 | 0.214 | 0.199 | 0.413 | 1.13e-07, 2.19e-07 | 1.577 | 0.26 | 0.960 | 0.43 | 3e-4, 1.066, 0.39 | 3e-4, 0.767, 0.54 |
| w4-plummer-1 | 1e-8 | 2x20 | 256 | 0.489 | 0.513 | 1.002 | 2.11e-12, 9.50e-12 | 2.938 | 0.34 | 1.475 | 0.68 | >3e-9, 3.161, 0.32 | >3e-9, 1.578, 0.64 |
| w4-plummer-10 | 1e-3 | 4x10 | 256 | 0.429 | 0.099 | 0.528 | 4.81e-05, 7.54e-05 | 0.740 | 0.71 | 0.398 | 1.33 | 3e-4, 1.108, 0.48 | 3e-4, 0.787, 0.67 |
| w4-plummer-10 | 1e-6 | 4x10 | 256 | 0.233 | 0.210 | 0.443 | 8.11e-08, 2.20e-07 | 1.609 | 0.28 | 0.984 | 0.45 | 3e-4, 1.108, 0.40 | 3e-4, 0.787, 0.56 |
| w4-plummer-10 | 1e-8 | 2x20 | 256 | 0.507 | 0.512 | 1.019 | 2.09e-12, 5.32e-12 | 2.966 | 0.34 | 1.489 | 0.68 | >3e-8, 2.799, 0.36 | >1e-9, 1.680, 0.61 |
| w4-plummer-100 | 1e-3 | 2x20 | 256 | 0.861 | 0.165 | 1.026 | 7.78e-05, 1.01e-04 | 1.191 | 0.86 | 0.760 | 1.35 | 3e-4, 1.684, 0.61 | 3e-4, 1.079, 0.95 |
| w4-plummer-100 | 1e-6 | 2x20 | 256 | 0.415 | 0.267 | 0.682 | 1.52e-07, 2.61e-07 | 2.305 | 0.30 | 1.350 | 0.51 | 3e-4, 1.684, 0.41 | 3e-4, 1.079, 0.63 |
| w4-plummer-100 | 1e-8 | 2x20 | 256 | 0.674 | 0.586 | 1.260 | 3.05e-10, 1.39e-10 | 3.928 | 0.32 | 1.978 | 0.64 | 3e-7, 3.400, 0.37 | 3e-7, 1.715, 0.73 |
| w4-clusters-1 | 1e-3 | 8x5 | 256 | 0.235 | 0.072 | 0.307 | 4.48e-05, 6.11e-05 | 0.912 | 0.34 | 0.439 | 0.70 | 3e-4, 1.235, 0.25 | 3e-4, 0.873, 0.35 |
| w4-clusters-1 | 1e-6 | 4x10 | 256 | 0.215 | 0.167 | 0.382 | 7.20e-08, 2.09e-07 | 1.868 | 0.20 | 1.163 | 0.33 | 3e-4, 1.235, 0.31 | 3e-4, 0.873, 0.44 |
| w4-clusters-1 | 1e-8 | 4x10 | 256 | 0.491 | 0.490 | 0.981 | 1.15e-12, 6.13e-12 | 3.502 | 0.28 | 1.932 | 0.51 | >1e-7, 3.072, 0.32 | >3e-8, 1.815, 0.54 |
| w4-clusters-10 | 1e-3 | 8x5 | 256 | 0.235 | 0.071 | 0.306 | 4.47e-05, 6.11e-05 | 0.909 | 0.34 | 0.443 | 0.69 | 3e-4, 1.232, 0.25 | 3e-4, 0.876, 0.35 |
| w4-clusters-10 | 1e-6 | 4x10 | 256 | 0.226 | 0.180 | 0.406 | 7.08e-08, 2.09e-07 | 1.900 | 0.21 | 1.171 | 0.35 | 3e-4, 1.232, 0.33 | 3e-4, 0.876, 0.46 |
| w4-clusters-10 | 1e-8 | 2x20 | 256 | 0.506 | 0.455 | 0.961 | 2.25e-12, 5.56e-12 | 3.514 | 0.27 | 1.936 | 0.50 | >1e-7, 3.092, 0.31 | >1e-8, 1.936, 0.50 |
| w4-clusters-100 | 1e-3 | 4x10 | 256 | 0.369 | 0.081 | 0.450 | 4.78e-05, 6.39e-05 | 1.154 | 0.39 | 0.615 | 0.73 | 3e-4, 1.549, 0.29 | 3e-4, 1.063, 0.42 |
| w4-clusters-100 | 1e-6 | 4x10 | 256 | 0.309 | 0.178 | 0.487 | 7.11e-08, 2.11e-07 | 2.248 | 0.22 | 1.375 | 0.35 | 3e-4, 1.549, 0.31 | 3e-4, 1.063, 0.46 |
| w4-clusters-100 | 1e-8 | 4x10 | 256 | 0.573 | 0.480 | 1.053 | 4.13e-12, 5.55e-12 | 3.972 | 0.27 | 2.245 | 0.47 | >1e-8, 3.972, 0.27 | >1e-8, 2.245, 0.47 |

### 2.2 locust

| Case | ε | split | leaf | build s | evaluate s | ours s | φ, ∇φ | default s | ÷ | fastker s | ÷ | default matched: eps, s, ÷ | fastker matched: eps, s, ÷ |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| w1-10 | 1e-3 | 4x18 | 128 | 0.001 | 0.002 | 0.003 | 1.42e-15, 1.75e-15 | 0.031 | 0.08 | 0.017 | 0.15 | 1e-6, 0.018, 0.15 | 1e-3, 0.017, 0.15 |
| w1-10 | 1e-6 | 8x9 | 128 | 0.001 | 0.001 | 0.002 | 1.42e-15, 1.75e-15 | 0.018 | 0.13 | 0.033 | 0.07 | 1e-6, 0.018, 0.13 | 1e-3, 0.017, 0.13 |
| w1-10 | 1e-8 | 8x9 | 256 | 0.001 | 0.001 | 0.002 | 1.42e-15, 1.75e-15 | 0.018 | 0.13 | 0.036 | 0.06 | 1e-6, 0.018, 0.13 | 1e-3, 0.017, 0.13 |
| w1-30 | 1e-3 | 8x9 | 128 | 0.033 | 0.005 | 0.038 | 2.59e-05, 6.91e-05 | 0.061 | 0.62 | 0.048 | 0.79 | 3e-4, 0.056, 0.67 | 3e-4, 0.222, 0.17 |
| w1-30 | 1e-6 | 8x9 | 128 | 0.051 | 0.008 | 0.059 | 1.12e-07, 2.96e-07 | 0.063 | 0.94 | 0.224 | 0.26 | 3e-4, 0.056, 1.06 | 3e-4, 0.222, 0.27 |
| w1-30 | 1e-8 | 8x9 | 256 | 0.052 | 0.015 | 0.067 | 5.45e-12, 6.43e-12 | 0.082 | 0.83 | 0.227 | 0.30 | >1e-9, 0.089, 0.76 | 3e-4, 0.222, 0.30 |
| w1-100 | 1e-3 | 8x9 | 128 | 0.144 | 0.053 | 0.197 | 1.32e-05, 3.14e-05 | 0.453 | 0.43 | 0.339 | 0.58 | 3e-4, 0.380, 0.52 | 3e-4, 0.546, 0.36 |
| w1-100 | 1e-6 | 8x9 | 128 | 0.158 | 0.062 | 0.220 | 8.99e-08, 2.55e-07 | 0.460 | 0.48 | 0.581 | 0.38 | 3e-7, 0.674, 0.33 | 3e-7, 0.627, 0.35 |
| w1-100 | 1e-8 | 8x9 | 256 | 0.206 | 0.131 | 0.337 | 6.28e-10, 8.74e-10 | 0.758 | 0.44 | 0.671 | 0.50 | 3e-7, 0.674, 0.50 | 3e-7, 0.627, 0.54 |
| w1-500 | 1e-3 | 4x18 | 128 | 2.338 | 2.141 | 4.479 | 2.67e-05, 6.96e-05 | 19.123 | 0.23 | 14.637 | 0.31 | 3e-4, 19.041, 0.24 | 3e-4, 19.320, 0.23 |
| w1-500 | 1e-6 | 4x18 | 128 | 2.705 | 2.843 | 5.548 | 1.44e-07, 3.58e-07 | 23.875 | 0.23 | 22.651 | 0.24 | 3e-4, 19.041, 0.29 | 3e-4, 19.320, 0.29 |
| w1-500 | 1e-8 | 2x36 | 256 | 5.168 | 3.169 | 8.337 | 1.62e-09, 3.21e-09 | 27.253 | 0.31 | 24.682 | 0.34 | 3e-7, 24.773, 0.34 | 3e-7, 23.137, 0.36 |
| w3-10 | 1e-3 | 4x18 | 128 | 0.034 | 0.003 | 0.037 | 3.88e-05, 6.58e-05 | 0.010 | 3.88 | 0.008 | 4.55 | 1e-3, 0.010, 3.88 | 1e-3, 0.008, 4.55 |
| w3-10 | 1e-6 | 8x9 | 128 | 0.025 | 0.004 | 0.029 | 4.01e-08, 1.24e-07 | 0.019 | 1.52 | 0.074 | 0.39 | 1e-4, 0.013, 2.24 | 3e-4, 0.073, 0.40 |
| w3-10 | 1e-8 | 4x18 | 256 | 0.048 | 0.007 | 0.055 | 6.12e-12, 2.82e-12 | 0.034 | 1.61 | 0.076 | 0.73 | >3e-9, 0.038, 1.45 | 3e-4, 0.073, 0.75 |
| w3-30 | 1e-3 | 8x9 | 128 | 0.134 | 0.040 | 0.174 | 4.60e-05, 1.31e-04 | 0.135 | 1.29 | 0.345 | 0.50 | 3e-4, 0.305, 0.57 | 3e-4, 0.361, 0.48 |
| w3-30 | 1e-6 | 8x9 | 128 | 0.080 | 0.067 | 0.147 | 1.42e-07, 4.02e-07 | 0.396 | 0.37 | 0.385 | 0.38 | 3e-4, 0.305, 0.48 | 3e-4, 0.361, 0.41 |
| w3-30 | 1e-8 | 4x18 | 256 | 0.141 | 0.137 | 0.278 | 5.04e-12, 6.99e-12 | 0.556 | 0.50 | 0.437 | 0.64 | >1e-9, 0.576, 0.48 | >1e-9, 0.464, 0.60 |
| w3-100 | 1e-3 | 8x9 | 128 | 0.707 | 2.185 | 2.892 | 4.77e-05, 9.85e-05 | 8.272 | 0.35 | 8.104 | 0.36 | 3e-4, 10.816, 0.27 | 3e-4, 37.875, 0.08 |
| w3-100 | 1e-6 | 8x9 | 128 | 0.682 | 2.790 | 3.472 | 1.01e-07, 2.13e-07 | 11.711 | 0.30 | 38.540 | 0.09 | 3e-7, 12.858, 0.27 | 1e-7, 39.531, 0.09 |
| w3-100 | 1e-8 | 8x9 | 256 | 0.608 | 3.981 | 4.588 | 1.19e-09, 1.82e-09 | 14.013 | 0.33 | 40.109 | 0.11 | 3e-7, 12.858, 0.36 | 1e-7, 39.531, 0.12 |
| w4-cube-1 | 1e-3 | 8x9 | 128 | 0.114 | 0.033 | 0.147 | 4.07e-05, 8.96e-05 | 0.084 | 1.74 | 0.188 | 0.78 | 3e-4, 0.315, 0.47 | 3e-4, 0.198, 0.74 |
| w4-cube-1 | 1e-6 | 8x9 | 128 | 0.066 | 0.061 | 0.127 | 6.29e-08, 2.46e-07 | 0.377 | 0.34 | 0.215 | 0.59 | 1e-4, 0.324, 0.39 | 1e-4, 0.200, 0.63 |
| w4-cube-1 | 1e-8 | 8x9 | 256 | 0.075 | 0.130 | 0.205 | 1.49e-12, 3.74e-12 | 0.381 | 0.54 | 0.256 | 0.80 | >3e-8, 0.375, 0.55 | >3e-8, 0.249, 0.82 |
| w4-cube-10 | 1e-3 | 8x9 | 128 | 0.114 | 0.033 | 0.147 | 5.00e-05, 9.95e-05 | 0.091 | 1.62 | 0.228 | 0.65 | 3e-4, 0.347, 0.42 | 3e-4, 0.235, 0.63 |
| w4-cube-10 | 1e-6 | 8x9 | 128 | 0.072 | 0.059 | 0.132 | 6.10e-08, 2.50e-07 | 0.414 | 0.32 | 0.259 | 0.51 | 1e-4, 0.360, 0.36 | 1e-4, 0.239, 0.55 |
| w4-cube-10 | 1e-8 | 4x18 | 256 | 0.095 | 0.118 | 0.213 | 1.61e-11, 4.61e-12 | 0.425 | 0.50 | 0.302 | 0.71 | >1e-9, 0.446, 0.48 | >1e-9, 0.323, 0.66 |
| w4-cube-100 | 1e-3 | 8x9 | 128 | 0.154 | 0.085 | 0.239 | 1.85e-05, 4.04e-05 | 0.283 | 0.84 | 0.404 | 0.59 | 3e-4, 0.588, 0.41 | 3e-4, 0.438, 0.55 |
| w4-cube-100 | 1e-6 | 8x9 | 128 | 0.147 | 0.107 | 0.254 | 9.66e-08, 3.01e-07 | 0.697 | 0.36 | 0.480 | 0.53 | 3e-7, 0.617, 0.41 | 3e-7, 0.527, 0.48 |
| w4-cube-100 | 1e-8 | 4x18 | 256 | 0.177 | 0.138 | 0.315 | 8.42e-10, 1.27e-09 | 0.649 | 0.49 | 0.557 | 0.56 | 3e-7, 0.617, 0.51 | 3e-7, 0.527, 0.60 |
| w4-sphere-1 | 1e-3 | 8x9 | 128 | 0.080 | 0.028 | 0.108 | 2.13e-05, 5.62e-07 | 0.146 | 0.74 | 0.112 | 0.96 | 3e-4, 0.171, 0.63 | 3e-4, 0.398, 0.27 |
| w4-sphere-1 | 1e-6 | 8x9 | 128 | 0.078 | 0.050 | 0.128 | 4.00e-08, 1.75e-09 | 0.218 | 0.59 | 0.418 | 0.31 | 1e-4, 0.176, 0.73 | 1e-4, 0.402, 0.32 |
| w4-sphere-1 | 1e-8 | 8x9 | 256 | 0.112 | 0.111 | 0.223 | 9.78e-13, 2.25e-12 | 0.492 | 0.45 | 0.462 | 0.48 | >1e-8, 0.492, 0.45 | >1e-8, 0.462, 0.48 |
| w4-sphere-10 | 1e-3 | 8x9 | 128 | 0.080 | 0.028 | 0.108 | 2.74e-05, 6.04e-07 | 0.151 | 0.71 | 0.123 | 0.88 | 3e-4, 0.184, 0.59 | 1e-4, 0.478, 0.23 |
| w4-sphere-10 | 1e-6 | 8x9 | 128 | 0.086 | 0.051 | 0.137 | 4.21e-08, 1.80e-09 | 0.231 | 0.59 | 0.495 | 0.28 | 3e-4, 0.184, 0.74 | 1e-4, 0.478, 0.29 |
| w4-sphere-10 | 1e-8 | 8x9 | 256 | 0.118 | 0.110 | 0.227 | 6.77e-12, 2.25e-12 | 0.518 | 0.44 | 0.544 | 0.42 | >1e-9, 0.575, 0.40 | >1e-9, 0.565, 0.40 |
| w4-sphere-100 | 1e-3 | 8x9 | 128 | 0.150 | 0.054 | 0.204 | 1.84e-05, 1.26e-06 | 0.404 | 0.51 | 0.293 | 0.70 | 3e-4, 0.379, 0.54 | 3e-4, 0.767, 0.27 |
| w4-sphere-100 | 1e-6 | 8x9 | 128 | 0.164 | 0.075 | 0.239 | 6.08e-08, 3.60e-09 | 0.465 | 0.51 | 0.806 | 0.30 | 3e-7, 0.745, 0.32 | 3e-7, 0.851, 0.28 |
| w4-sphere-100 | 1e-8 | 4x18 | 256 | 0.219 | 0.125 | 0.344 | 4.36e-10, 1.50e-11 | 0.836 | 0.41 | 0.890 | 0.39 | 3e-7, 0.745, 0.46 | 3e-7, 0.851, 0.40 |
| w4-plummer-1 | 1e-3 | 8x9 | 128 | 0.209 | 0.075 | 0.284 | 6.32e-05, 8.32e-05 | 0.322 | 0.88 | 0.264 | 1.08 | 3e-4, 0.491, 0.58 | 3e-4, 1.056, 0.27 |
| w4-plummer-1 | 1e-6 | 8x9 | 128 | 0.109 | 0.107 | 0.216 | 1.16e-07, 2.73e-07 | 0.754 | 0.29 | 1.154 | 0.19 | 3e-4, 0.491, 0.44 | 3e-4, 1.056, 0.20 |
| w4-plummer-1 | 1e-8 | 4x18 | 256 | 0.236 | 0.246 | 0.482 | 2.12e-12, 9.83e-12 | 1.449 | 0.33 | 1.384 | 0.35 | >3e-9, 1.546, 0.31 | >3e-9, 1.437, 0.34 |
| w4-plummer-10 | 1e-3 | 8x9 | 128 | 0.210 | 0.077 | 0.287 | 4.91e-05, 8.34e-05 | 0.332 | 0.86 | 0.281 | 1.02 | 3e-4, 0.506, 0.57 | 3e-4, 1.168, 0.25 |
| w4-plummer-10 | 1e-6 | 8x9 | 128 | 0.119 | 0.112 | 0.231 | 8.36e-08, 2.74e-07 | 0.770 | 0.30 | 1.269 | 0.18 | 3e-4, 0.506, 0.46 | 3e-4, 1.168, 0.20 |
| w4-plummer-10 | 1e-8 | 4x18 | 256 | 0.239 | 0.243 | 0.482 | 2.23e-12, 7.44e-12 | 1.470 | 0.33 | 1.504 | 0.32 | >3e-8, 1.394, 0.35 | >1e-9, 1.609, 0.30 |
| w4-plummer-100 | 1e-3 | 8x9 | 128 | 0.260 | 0.105 | 0.365 | 7.83e-05, 1.07e-04 | 0.549 | 0.67 | 0.509 | 0.72 | 3e-4, 0.781, 0.47 | 3e-4, 1.856, 0.20 |
| w4-plummer-100 | 1e-6 | 8x9 | 128 | 0.198 | 0.139 | 0.337 | 1.53e-07, 3.07e-07 | 1.102 | 0.31 | 1.988 | 0.17 | 3e-4, 0.781, 0.43 | 3e-4, 1.856, 0.18 |
| w4-plummer-100 | 1e-8 | 4x18 | 256 | 0.345 | 0.266 | 0.611 | 3.05e-10, 1.39e-10 | 1.924 | 0.32 | 2.293 | 0.27 | 3e-7, 1.667, 0.37 | 3e-7, 2.145, 0.28 |
| w4-clusters-1 | 1e-3 | 8x9 | 128 | 0.118 | 0.053 | 0.171 | 4.66e-05, 6.80e-05 | 0.376 | 0.45 | 0.300 | 0.57 | 3e-4, 0.537, 0.32 | 3e-4, 1.144, 0.15 |
| w4-clusters-1 | 1e-6 | 8x9 | 128 | 0.126 | 0.084 | 0.210 | 7.80e-08, 2.35e-07 | 0.851 | 0.25 | 1.294 | 0.16 | 3e-4, 0.537, 0.39 | 3e-4, 1.144, 0.18 |
| w4-clusters-1 | 1e-8 | 4x18 | 256 | 0.253 | 0.174 | 0.427 | 2.35e-12, 1.78e-11 | 1.546 | 0.28 | 1.654 | 0.26 | >1e-7, 1.356, 0.31 | >3e-8, 1.590, 0.27 |
| w4-clusters-10 | 1e-3 | 8x9 | 128 | 0.118 | 0.053 | 0.171 | 4.65e-05, 6.80e-05 | 0.379 | 0.45 | 0.314 | 0.55 | 3e-4, 0.545, 0.31 | 3e-4, 1.259, 0.14 |
| w4-clusters-10 | 1e-6 | 8x9 | 128 | 0.135 | 0.085 | 0.220 | 7.65e-08, 2.34e-07 | 0.858 | 0.26 | 1.414 | 0.16 | 3e-4, 0.545, 0.40 | 3e-4, 1.259, 0.17 |
| w4-clusters-10 | 1e-8 | 8x9 | 256 | 0.241 | 0.185 | 0.426 | 2.48e-12, 9.75e-12 | 1.553 | 0.27 | 1.762 | 0.24 | >1e-8, 1.553, 0.27 | >1e-8, 1.762, 0.24 |
| w4-clusters-100 | 1e-3 | 8x9 | 128 | 0.191 | 0.057 | 0.248 | 4.93e-05, 7.06e-05 | 0.495 | 0.50 | 0.417 | 0.59 | 3e-4, 0.681, 0.36 | 3e-4, 1.546, 0.16 |
| w4-clusters-100 | 1e-6 | 8x9 | 128 | 0.174 | 0.089 | 0.263 | 7.63e-08, 2.37e-07 | 1.013 | 0.26 | 1.715 | 0.15 | 3e-4, 0.681, 0.39 | 3e-4, 1.546, 0.17 |
| w4-clusters-100 | 1e-8 | 8x9 | 256 | 0.289 | 0.185 | 0.474 | 4.27e-12, 6.82e-12 | 1.770 | 0.27 | 2.117 | 0.22 | >1e-8, 1.770, 0.27 | >1e-8, 2.117, 0.22 |

FMM3D's `fastker` on locust is slower than `default` on the larger trees (W3 at kD = 100:
38.5 and 40.1 s at 10⁻⁶ and 10⁻⁸ against `default`'s 11.7 and 14.0 s; both launches within
0.3 %); it is the system GCC 11.4.1's build with FMM3D's own flags (`tools/fmm3d/README.md`),
not investigated here.

### 2.3 The summary and where FMM3D is faster

Our time over FMM3D's, per ε and variant, over the 18 cases other than W1 at kD = 10: the
geometric mean, the range and the cases where ours is slower; the last column the two cases
of N ≥ 10⁶ (W1 at kD = 500; W3 at kD = 100), same ε and matched.

Kathleen:

| ε | variant | cases | same ε: geometric mean (range) | slower | matched: geometric mean (range) | slower | N ≥ 10⁶: same ε, matched |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-3 | default | 18 | 0.59 (0.17–3.64) | 2 | 0.43 (0.12–3.64) | 2 | 0.20, 0.20; 0.17, 0.12 |
| 1e-3 | fastker | 18 | 1.16 (0.29–6.75) | 11 | 0.78 (0.18–6.75) | 7 | 0.29, 0.29; 0.50, 0.18 |
| 1e-6 | default | 18 | 0.32 (0.16–1.45) | 1 | 0.38 (0.14–1.94) | 2 | 0.19, 0.24; 0.16, 0.14 |
| 1e-6 | fastker | 18 | 0.61 (0.22–2.63) | 2 | 0.69 (0.20–2.89) | 4 | 0.27, 0.34; 0.22, 0.20 |
| 1e-8 | default | 18 | 0.42 (0.27–1.55) | 2 | 0.44 (0.27–1.43) | 2 | 0.28, 0.31; 0.31, 0.33 |
| 1e-8 | fastker | 18 | 0.90 (0.41–4.42) | 9 | 0.93 (0.44–5.36) | 9 | 0.41, 0.44; 0.43, 0.47 |

locust:

| ε | variant | cases | same ε: geometric mean (range) | slower | matched: geometric mean (range) | slower | N ≥ 10⁶: same ε, matched |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-3 | default | 18 | 0.73 (0.23–3.88) | 4 | 0.50 (0.24–3.88) | 1 | 0.23, 0.24; 0.35, 0.27 |
| 1e-3 | fastker | 18 | 0.72 (0.31–4.55) | 3 | 0.30 (0.08–4.55) | 1 | 0.31, 0.23; 0.36, 0.08 |
| 1e-6 | default | 18 | 0.39 (0.23–1.52) | 1 | 0.48 (0.27–2.24) | 2 | 0.23, 0.29; 0.30, 0.27 |
| 1e-6 | fastker | 18 | 0.26 (0.09–0.59) | 0 | 0.27 (0.09–0.63) | 0 | 0.24, 0.29; 0.09, 0.09 |
| 1e-8 | default | 18 | 0.43 (0.27–1.61) | 1 | 0.44 (0.27–1.45) | 1 | 0.31, 0.34; 0.33, 0.36 |
| 1e-8 | fastker | 18 | 0.38 (0.11–0.80) | 0 | 0.38 (0.12–0.82) | 0 | 0.34, 0.36; 0.11, 0.12 |

The cases slower than `fastker` at matched accuracy on Kathleen, with our build's share of
our time:

| Case | ε | build s | evaluate s | ours s | `fastker` matched: eps, s | ÷ | build share |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| w1-10 | 1e-3 | 0.004 | 0.001 | 0.005 | 1e-3, 0.003 | 1.87 | 80 % |
| w1-10 | 1e-6 | 0.002 | 0.001 | 0.003 | 1e-3, 0.003 | 1.10 | 67 % |
| w1-10 | 1e-8 | 0.002 | 0.001 | 0.003 | 1e-3, 0.003 | 1.06 | 67 % |
| w1-30 | 1e-3 | 0.060 | 0.010 | 0.070 | 3e-4, 0.052 | 1.37 | 86 % |
| w1-30 | 1e-6 | 0.055 | 0.013 | 0.068 | 3e-4, 0.052 | 1.32 | 81 % |
| w1-30 | 1e-8 | 0.107 | 0.025 | 0.132 | 3e-4, 0.052 | 2.57 | 81 % |
| w1-100 | 1e-8 | 0.430 | 0.324 | 0.754 | 3e-7, 0.591 | 1.28 | 57 % |
| w3-10 | 1e-3 | 0.073 | 0.003 | 0.076 | 1e-3, 0.011 | 6.75 | 96 % |
| w3-10 | 1e-6 | 0.047 | 0.010 | 0.057 | 3e-4, 0.020 | 2.89 | 82 % |
| w3-10 | 1e-8 | 0.088 | 0.018 | 0.106 | 3e-4, 0.020 | 5.36 | 83 % |
| w3-30 | 1e-3 | 0.184 | 0.090 | 0.274 | 3e-4, 0.243 | 1.13 | 67 % |
| w3-30 | 1e-6 | 0.141 | 0.118 | 0.259 | 3e-4, 0.243 | 1.06 | 54 % |
| w3-30 | 1e-8 | 0.263 | 0.276 | 0.539 | >1e-9, 0.501 | 1.08 | 49 % |
| w4-cube-1 | 1e-3 | 0.150 | 0.066 | 0.216 | 3e-4, 0.174 | 1.24 | 69 % |
| w4-cube-1 | 1e-8 | 0.180 | 0.210 | 0.390 | >3e-8, 0.309 | 1.26 | 46 % |
| w4-cube-10 | 1e-3 | 0.151 | 0.067 | 0.218 | 3e-4, 0.183 | 1.19 | 69 % |
| w4-cube-10 | 1e-6 | 0.092 | 0.102 | 0.194 | 1e-4, 0.191 | 1.02 | 47 % |
| w4-cube-10 | 1e-8 | 0.189 | 0.210 | 0.399 | >1e-9, 0.392 | 1.02 | 47 % |
| w4-sphere-1 | 1e-3 | 0.223 | 0.042 | 0.265 | 3e-4, 0.250 | 1.06 | 84 % |
| w4-sphere-1 | 1e-8 | 0.239 | 0.268 | 0.507 | >1e-8, 0.397 | 1.28 | 47 % |
| w4-sphere-10 | 1e-3 | 0.219 | 0.044 | 0.263 | 3e-4, 0.254 | 1.03 | 83 % |
| w4-sphere-10 | 1e-8 | 0.242 | 0.269 | 0.511 | >1e-9, 0.459 | 1.11 | 47 % |
| w4-sphere-100 | 1e-8 | 0.431 | 0.368 | 0.799 | 3e-7, 0.686 | 1.17 | 54 % |

Our evaluation alone is below FMM3D's whole call in every row. FMM3D keeps nothing between
calls, so for a sequence of evaluations on one tree (an iterative solver) the evaluation is
the measure, and there ours is 0.07–0.39 of `fastker`'s call on the two large cases on both
machines.

### 2.4 The splits

Build plus one evaluation per split (the smaller of the two launches) and the checksums of
the eight launches.

Kathleen (4 × 10 is CH3.6's split):

| Case | ε | 1x40 | 2x20 | 4x10 | 8x5 | best | checksums |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| w1-10 | 1e-3 | 0.006 | 0.005 | 0.005 | 0.007 | 4x10 | equal (8) |
| w1-10 | 1e-6 | 0.005 | 0.003 | 0.003 | 0.003 | 4x10 | equal (8) |
| w1-10 | 1e-8 | 0.005 | 0.003 | 0.003 | 0.003 | 8x5 | equal (8) |
| w1-30 | 1e-3 | 0.075 | 0.071 | 0.070 | 0.075 | 4x10 | equal (8) |
| w1-30 | 1e-6 | 0.079 | 0.070 | 0.068 | 0.074 | 4x10 | equal (8) |
| w1-30 | 1e-8 | 0.151 | 0.132 | 0.139 | 0.151 | 2x20 | equal (8) |
| w1-100 | 1e-3 | 0.390 | 0.347 | 0.348 | 0.384 | 2x20 | equal (8) |
| w1-100 | 1e-6 | 0.508 | 0.442 | 0.456 | 0.489 | 2x20 | equal (8) |
| w1-100 | 1e-8 | 0.852 | 0.754 | 0.771 | 0.866 | 2x20 | equal (8) |
| w1-500 | 1e-3 | 10.973 | 9.851 | 10.984 | 14.103 | 2x20 | equal (8) |
| w1-500 | 1e-6 | 12.792 | 11.714 | 12.505 | 17.007 | 2x20 | equal (8) |
| w1-500 | 1e-8 | 20.919 | 19.886 | 21.550 | 26.242 | 2x20 | equal (8) |
| w3-10 | 1e-3 | 0.079 | 0.076 | 0.076 | 0.081 | 2x20 | equal (8) |
| w3-10 | 1e-6 | 0.066 | 0.057 | 0.057 | 0.059 | 4x10 | equal (8) |
| w3-10 | 1e-8 | 0.122 | 0.106 | 0.106 | 0.109 | 4x10 | equal (8) |
| w3-30 | 1e-3 | 0.327 | 0.285 | 0.274 | 0.279 | 4x10 | equal (8) |
| w3-30 | 1e-6 | 0.325 | 0.270 | 0.259 | 0.281 | 4x10 | equal (8) |
| w3-30 | 1e-8 | 0.603 | 0.545 | 0.539 | 0.616 | 4x10 | equal (8) |
| w3-100 | 1e-3 | 7.042 | 5.256 | 4.128 | 3.641 | 8x5 | equal (8) |
| w3-100 | 1e-6 | 8.367 | 6.470 | 5.454 | 5.013 | 8x5 | equal (8) |
| w3-100 | 1e-8 | 13.867 | 12.596 | 11.811 | 11.732 | 8x5 | equal (8) |
| w4-cube-1 | 1e-3 | 0.260 | 0.226 | 0.216 | 0.221 | 4x10 | equal (8) |
| w4-cube-1 | 1e-6 | 0.240 | 0.192 | 0.185 | 0.194 | 4x10 | equal (8) |
| w4-cube-1 | 1e-8 | 0.463 | 0.390 | 0.394 | 0.450 | 2x20 | equal (8) |
| w4-cube-10 | 1e-3 | 0.269 | 0.225 | 0.218 | 0.221 | 4x10 | equal (8) |
| w4-cube-10 | 1e-6 | 0.254 | 0.204 | 0.194 | 0.203 | 4x10 | equal (8) |
| w4-cube-10 | 1e-8 | 0.456 | 0.399 | 0.402 | 0.459 | 2x20 | equal (8) |
| w4-cube-100 | 1e-3 | 0.454 | 0.399 | 0.404 | 0.466 | 2x20 | equal (8) |
| w4-cube-100 | 1e-6 | 0.480 | 0.425 | 0.439 | 0.499 | 2x20 | equal (8) |
| w4-cube-100 | 1e-8 | 0.693 | 0.631 | 0.667 | 0.804 | 2x20 | equal (8) |
| w4-sphere-1 | 1e-3 | 0.313 | 0.275 | 0.267 | 0.265 | 8x5 | equal (8) |
| w4-sphere-1 | 1e-6 | 0.293 | 0.239 | 0.216 | 0.225 | 4x10 | equal (8) |
| w4-sphere-1 | 1e-8 | 0.620 | 0.541 | 0.507 | 0.555 | 4x10 | equal (8) |
| w4-sphere-10 | 1e-3 | 0.316 | 0.273 | 0.263 | 0.264 | 4x10 | equal (8) |
| w4-sphere-10 | 1e-6 | 0.310 | 0.247 | 0.229 | 0.234 | 4x10 | equal (8) |
| w4-sphere-10 | 1e-8 | 0.638 | 0.553 | 0.511 | 0.562 | 4x10 | equal (8) |
| w4-sphere-100 | 1e-3 | 0.413 | 0.362 | 0.357 | 0.392 | 4x10 | equal (8) |
| w4-sphere-100 | 1e-6 | 0.541 | 0.460 | 0.447 | 0.500 | 4x10 | equal (8) |
| w4-sphere-100 | 1e-8 | 0.924 | 0.806 | 0.799 | 0.904 | 4x10 | equal (8) |
| w4-plummer-1 | 1e-3 | 0.596 | 0.537 | 0.516 | 0.534 | 4x10 | equal (8) |
| w4-plummer-1 | 1e-6 | 0.508 | 0.435 | 0.413 | 0.483 | 4x10 | equal (8) |
| w4-plummer-1 | 1e-8 | 1.132 | 1.002 | 1.047 | 1.266 | 2x20 | equal (8) |
| w4-plummer-10 | 1e-3 | 0.596 | 0.533 | 0.528 | 0.535 | 4x10 | equal (8) |
| w4-plummer-10 | 1e-6 | 0.528 | 0.443 | 0.443 | 0.501 | 4x10 | equal (8) |
| w4-plummer-10 | 1e-8 | 1.110 | 1.019 | 1.035 | 1.243 | 2x20 | equal (8) |
| w4-plummer-100 | 1e-3 | 1.097 | 1.026 | 1.032 | 1.121 | 2x20 | equal (8) |
| w4-plummer-100 | 1e-6 | 0.765 | 0.682 | 0.693 | 0.788 | 2x20 | equal (8) |
| w4-plummer-100 | 1e-8 | 1.380 | 1.260 | 1.329 | 1.600 | 2x20 | equal (8) |
| w4-clusters-1 | 1e-3 | 0.407 | 0.337 | 0.312 | 0.307 | 8x5 | equal (8) |
| w4-clusters-1 | 1e-6 | 0.492 | 0.404 | 0.382 | 0.388 | 4x10 | equal (8) |
| w4-clusters-1 | 1e-8 | 1.099 | 0.988 | 0.981 | 0.993 | 4x10 | equal (8) |
| w4-clusters-10 | 1e-3 | 0.406 | 0.337 | 0.310 | 0.306 | 8x5 | equal (8) |
| w4-clusters-10 | 1e-6 | 0.505 | 0.420 | 0.406 | 0.406 | 4x10 | equal (8) |
| w4-clusters-10 | 1e-8 | 1.057 | 0.961 | 0.965 | 0.976 | 2x20 | equal (8) |
| w4-clusters-100 | 1e-3 | 0.556 | 0.479 | 0.450 | 0.453 | 4x10 | equal (8) |
| w4-clusters-100 | 1e-6 | 0.613 | 0.511 | 0.487 | 0.490 | 4x10 | equal (8) |
| w4-clusters-100 | 1e-8 | 1.143 | 1.071 | 1.053 | 1.089 | 4x10 | equal (8) |

locust:

| Case | ε | 1x72 | 2x36 | 4x18 | 8x9 | best | checksums |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| w1-10 | 1e-3 | 0.005 | 0.004 | 0.003 | 0.003 | 4x18 | equal (8) |
| w1-10 | 1e-6 | 0.005 | 0.003 | 0.003 | 0.002 | 8x9 | equal (8) |
| w1-10 | 1e-8 | 0.004 | 0.003 | 0.003 | 0.002 | 8x9 | equal (8) |
| w1-30 | 1e-3 | 0.052 | 0.043 | 0.039 | 0.038 | 8x9 | equal (8) |
| w1-30 | 1e-6 | 0.076 | 0.065 | 0.060 | 0.059 | 8x9 | equal (8) |
| w1-30 | 1e-8 | 0.089 | 0.073 | 0.068 | 0.067 | 8x9 | equal (8) |
| w1-100 | 1e-3 | 0.271 | 0.228 | 0.206 | 0.197 | 8x9 | equal (8) |
| w1-100 | 1e-6 | 0.292 | 0.251 | 0.227 | 0.220 | 8x9 | equal (8) |
| w1-100 | 1e-8 | 0.394 | 0.347 | 0.340 | 0.337 | 8x9 | equal (8) |
| w1-500 | 1e-3 | 6.019 | 4.802 | 4.479 | 4.976 | 4x18 | equal (8) |
| w1-500 | 1e-6 | 6.910 | 5.739 | 5.548 | 6.214 | 4x18 | equal (8) |
| w1-500 | 1e-8 | 8.889 | 8.337 | 8.439 | 9.465 | 2x36 | equal (8) |
| w3-10 | 1e-3 | 0.040 | 0.038 | 0.037 | 0.038 | 4x18 | equal (8) |
| w3-10 | 1e-6 | 0.038 | 0.034 | 0.030 | 0.029 | 8x9 | equal (8) |
| w3-10 | 1e-8 | 0.083 | 0.062 | 0.055 | 0.055 | 4x18 | equal (8) |
| w3-30 | 1e-3 | 0.280 | 0.219 | 0.185 | 0.174 | 8x9 | equal (8) |
| w3-30 | 1e-6 | 0.252 | 0.186 | 0.157 | 0.147 | 8x9 | equal (8) |
| w3-30 | 1e-8 | 0.348 | 0.310 | 0.278 | 0.285 | 4x18 | equal (8) |
| w3-100 | 1e-3 | 6.746 | 4.678 | 3.449 | 2.892 | 8x9 | equal (8) |
| w3-100 | 1e-6 | 7.334 | 5.240 | 4.022 | 3.472 | 8x9 | equal (8) |
| w3-100 | 1e-8 | 6.801 | 5.803 | 4.923 | 4.588 | 8x9 | equal (8) |
| w4-cube-1 | 1e-3 | 0.239 | 0.190 | 0.159 | 0.147 | 8x9 | equal (8) |
| w4-cube-1 | 1e-6 | 0.220 | 0.164 | 0.135 | 0.127 | 8x9 | equal (8) |
| w4-cube-1 | 1e-8 | 0.256 | 0.225 | 0.208 | 0.205 | 8x9 | equal (8) |
| w4-cube-10 | 1e-3 | 0.241 | 0.189 | 0.158 | 0.147 | 8x9 | equal (8) |
| w4-cube-10 | 1e-6 | 0.228 | 0.170 | 0.141 | 0.132 | 8x9 | equal (8) |
| w4-cube-10 | 1e-8 | 0.259 | 0.230 | 0.213 | 0.213 | 4x18 | equal (8) |
| w4-cube-100 | 1e-3 | 0.352 | 0.283 | 0.250 | 0.239 | 8x9 | equal (8) |
| w4-cube-100 | 1e-6 | 0.357 | 0.290 | 0.256 | 0.254 | 8x9 | equal (8) |
| w4-cube-100 | 1e-8 | 0.364 | 0.330 | 0.315 | 0.330 | 4x18 | equal (8) |
| w4-sphere-1 | 1e-3 | 0.187 | 0.144 | 0.120 | 0.108 | 8x9 | equal (8) |
| w4-sphere-1 | 1e-6 | 0.221 | 0.166 | 0.141 | 0.128 | 8x9 | equal (8) |
| w4-sphere-1 | 1e-8 | 0.291 | 0.246 | 0.231 | 0.223 | 8x9 | equal (8) |
| w4-sphere-10 | 1e-3 | 0.187 | 0.144 | 0.120 | 0.108 | 8x9 | equal (8) |
| w4-sphere-10 | 1e-6 | 0.232 | 0.174 | 0.149 | 0.137 | 8x9 | equal (8) |
| w4-sphere-10 | 1e-8 | 0.296 | 0.252 | 0.236 | 0.227 | 8x9 | equal (8) |
| w4-sphere-100 | 1e-3 | 0.314 | 0.252 | 0.217 | 0.204 | 8x9 | equal (8) |
| w4-sphere-100 | 1e-6 | 0.345 | 0.288 | 0.249 | 0.239 | 8x9 | equal (8) |
| w4-sphere-100 | 1e-8 | 0.403 | 0.355 | 0.344 | 0.349 | 4x18 | equal (8) |
| w4-plummer-1 | 1e-3 | 0.351 | 0.312 | 0.292 | 0.284 | 8x9 | equal (8) |
| w4-plummer-1 | 1e-6 | 0.292 | 0.243 | 0.219 | 0.216 | 8x9 | equal (8) |
| w4-plummer-1 | 1e-8 | 0.583 | 0.528 | 0.482 | 0.490 | 4x18 | equal (8) |
| w4-plummer-10 | 1e-3 | 0.351 | 0.312 | 0.294 | 0.287 | 8x9 | equal (8) |
| w4-plummer-10 | 1e-6 | 0.305 | 0.254 | 0.232 | 0.231 | 8x9 | equal (8) |
| w4-plummer-10 | 1e-8 | 0.584 | 0.530 | 0.482 | 0.502 | 4x18 | equal (8) |
| w4-plummer-100 | 1e-3 | 0.450 | 0.394 | 0.368 | 0.365 | 8x9 | equal (8) |
| w4-plummer-100 | 1e-6 | 0.422 | 0.365 | 0.339 | 0.337 | 8x9 | equal (8) |
| w4-plummer-100 | 1e-8 | 0.705 | 0.646 | 0.611 | 0.626 | 4x18 | equal (8) |
| w4-clusters-1 | 1e-3 | 0.249 | 0.205 | 0.179 | 0.171 | 8x9 | equal (8) |
| w4-clusters-1 | 1e-6 | 0.304 | 0.244 | 0.213 | 0.210 | 8x9 | equal (8) |
| w4-clusters-1 | 1e-8 | 0.526 | 0.469 | 0.427 | 0.430 | 4x18 | equal (8) |
| w4-clusters-10 | 1e-3 | 0.250 | 0.208 | 0.179 | 0.171 | 8x9 | equal (8) |
| w4-clusters-10 | 1e-6 | 0.312 | 0.253 | 0.222 | 0.220 | 8x9 | equal (8) |
| w4-clusters-10 | 1e-8 | 0.529 | 0.465 | 0.426 | 0.426 | 8x9 | equal (8) |
| w4-clusters-100 | 1e-3 | 0.346 | 0.294 | 0.257 | 0.248 | 8x9 | equal (8) |
| w4-clusters-100 | 1e-6 | 0.369 | 0.309 | 0.273 | 0.263 | 8x9 | equal (8) |
| w4-clusters-100 | 1e-8 | 0.593 | 0.519 | 0.477 | 0.474 | 8x9 | equal (8) |

The best split is 4 × 10 (31 of 57 rows) or 2 × 20 (19) on Kathleen, 8 × 5 on W3 at kD = 100
and three W4 cases at 10⁻³; 8 × 9 (43) or 4 × 18 (13) on locust; one rank over the node
(1 × 40, 1 × 72) is never the best, and on the large cases the worst split takes up to 1.9×
the best (W3 at kD = 100 at 10⁻³ on Kathleen: 7.0 s at 1 × 40 against 3.6 s at 8 × 5). The two launches of a split
agree within a median 1.3 % (Kathleen) and 0.6 % (locust), at most 6 % (W3 at kD = 30); FMM3D's
within 2 %.

## 3. H4's expected gain

### 3.1 The levels H4 can change

The levels with k a ≥ 8 that run RCR: on the matrix only W1 at kD = 500's levels 2 and 3. W3
at kD = 100 and W1 at kD = 100 have their levels with k a ≥ 8 (x = 21.7, 10.8) on the
exponential M2L, and the other cases have none. W1 at kD = 500 per level at the best split
("ns per pair": the level's wall time over its V pairs, on the node):

Kathleen (2 × 20):

| ε | split | level | p | k a | ran | V pairs | boxes | M2L s | ns per pair |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1e-3 | 2x20 | 2 | 123 | 108.3 | rotation | 2504 | 64 | 2.1888 | 874133 |
| 1e-3 | 2x20 | 3 | 66 | 54.1 | rotation | 12216 | 512 | 0.7835 | 64138 |
| 1e-3 | 2x20 | 4 | 36 | 27.1 | planewave i2 | 54000 | 3840 | 0.4854 | 8989 |
| 1e-3 | 2x20 | 5 | 21 | 13.5 | planewave i2 | 228048 | 17600 | 0.4772 | 2093 |
| 1e-3 | 2x20 | 6 | 13 | 6.8 | planewave i2 | 738264 | 38080 | 0.5892 | 798 |
| 1e-6 | 2x20 | 2 | 131 | 108.3 | rotation | 2504 | 64 | 2.6448 | 1056218 |
| 1e-6 | 2x20 | 3 | 72 | 54.1 | rotation | 12216 | 512 | 1.0309 | 84389 |
| 1e-6 | 2x20 | 4 | 41 | 27.1 | planewave i2 | 54000 | 3840 | 0.5595 | 10362 |
| 1e-6 | 2x20 | 5 | 25 | 13.5 | planewave i2 | 228048 | 17600 | 0.5382 | 2360 |
| 1e-6 | 2x20 | 6 | 16 | 6.8 | planewave i2 | 703169 | 38080 | 0.7180 | 1021 |
| 1e-8 | 2x20 | 2 | 135 | 108.3 | rotation | 2504 | 64 | 2.9620 | 1182922 |
| 1e-8 | 2x20 | 3 | 76 | 54.1 | rotation | 12216 | 512 | 1.2781 | 104623 |
| 1e-8 | 2x20 | 4 | 44 | 27.1 | planewave i3 | 54000 | 3840 | 0.7980 | 14778 |
| 1e-8 | 2x20 | 5 | 32 | 13.5 | planewave i3 | 228048 | 17600 | 1.0644 | 4667 |
| 1e-8 | 2x20 | 6 | 32 | 6.8 | planewave i3 | 517489 | 38080 | 2.0387 | 3940 |

locust:

| ε | split | level | p | k a | ran | V pairs | boxes | M2L s | ns per pair |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1e-3 | 4x18 | 2 | 123 | 108.3 | rotation | 2504 | 64 | 1.0950 | 437287 |
| 1e-3 | 4x18 | 3 | 66 | 54.1 | rotation | 12216 | 512 | 0.2078 | 17011 |
| 1e-3 | 4x18 | 4 | 36 | 27.1 | planewave i2 | 54000 | 3840 | 0.1161 | 2150 |
| 1e-3 | 4x18 | 5 | 21 | 13.5 | planewave i2 | 228048 | 19584 | 0.0818 | 359 |
| 1e-3 | 4x18 | 6 | 13 | 6.8 | planewave i2 | 869120 | 63176 | 0.1189 | 137 |
| 1e-3 | 4x18 | 7 | 8 | 3.4 | rotation | 1434122 | 70264 | 0.0485 | 34 |
| 1e-6 | 4x18 | 2 | 131 | 108.3 | rotation | 2504 | 64 | 1.3329 | 532321 |
| 1e-6 | 4x18 | 3 | 72 | 54.1 | rotation | 12216 | 512 | 0.2662 | 21789 |
| 1e-6 | 4x18 | 4 | 41 | 27.1 | planewave i2 | 54000 | 3840 | 0.1290 | 2389 |
| 1e-6 | 4x18 | 5 | 25 | 13.5 | planewave i2 | 228048 | 19584 | 0.0955 | 419 |
| 1e-6 | 4x18 | 6 | 16 | 6.8 | planewave i2 | 866105 | 63176 | 0.1412 | 163 |
| 1e-6 | 4x18 | 7 | 15 | 3.4 | planewave i2 | 1369333 | 70264 | 0.2389 | 174 |
| 1e-8 | 2x36 | 2 | 135 | 108.3 | rotation | 2504 | 64 | 0.8147 | 325364 |
| 1e-8 | 2x36 | 3 | 76 | 54.1 | rotation | 12216 | 512 | 0.3131 | 25632 |
| 1e-8 | 2x36 | 4 | 44 | 27.1 | planewave i3 | 54000 | 3840 | 0.1685 | 3120 |
| 1e-8 | 2x36 | 5 | 32 | 13.5 | planewave i3 | 228048 | 17600 | 0.1857 | 814 |
| 1e-8 | 2x36 | 6 | 32 | 6.8 | planewave i3 | 777762 | 38080 | 0.4000 | 514 |

The estimate (§1, "The estimate"), per machine, for the rows with levels on RCR (every other
row: no level, no gain):

Kathleen:

| Case | ε | ours s | levels on RCR at k a ≥ 8 (level:p) | their M2L s | share | after H4, bound: s | after H4, prototype: s | ceiling (all M2L at k a ≥ 8) s | fastker s | after H4 ÷ fastker (bound, prototype) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| w1-500 | 1e-3 | 9.851 | 2:123, 3:66 | 2.972 | 30% | 6.878 | 8.548 | 5.916 | 33.434 | 0.21, 0.26 |
| w1-500 | 1e-6 | 11.714 | 2:131, 3:72 | 3.676 | 31% | 8.039 | 10.126 | 6.941 | 43.482 | 0.18, 0.23 |
| w1-500 | 1e-8 | 19.886 | 2:135, 3:76 | 4.240 | 21% | 15.646 | 18.087 | 13.784 | 49.083 | 0.32, 0.37 |

locust:

| Case | ε | ours s | levels on RCR at k a ≥ 8 (level:p) | their M2L s | share | after H4, bound: s | after H4, prototype: s | ceiling (all M2L at k a ≥ 8) s | fastker s | after H4 ÷ fastker (bound, prototype) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| w1-500 | 1e-3 | 4.479 | 2:123, 3:66 | 1.303 | 29% | 3.176 | 3.876 | 2.978 | 14.637 | 0.22, 0.26 |
| w1-500 | 1e-6 | 5.548 | 2:131, 3:72 | 1.599 | 29% | 3.949 | 4.811 | 3.725 | 22.651 | 0.17, 0.21 |
| w1-500 | 1e-8 | 8.337 | 2:135, 3:76 | 1.128 | 14% | 7.209 | 7.851 | 6.855 | 24.682 | 0.29, 0.32 |

On the evaluation alone the bound is 51, 51 and 37 % on Kathleen and 61, 56 and 36 % on locust
at 10⁻³, 10⁻⁶ and 10⁻⁸; the prototype estimate 22, 22 and 16 % and 28, 26 and 15 %. The
"ceiling" (every M2L at k a ≥ 8 free, the exponential M2L's levels 4 and 5 too) adds 1.0–1.9 s
on Kathleen and 0.2–0.4 s on locust to the bound.

**H4's build.** The diagonal form needs a transfer function per class and level (34 classes,
proposal §5.4); S6's prototype built one by interpolation in 124 ms at p = 128 and 32–70 ms at
p = 64–96 on one Grace core (`results/times.md`), so about 4.4 s for level 2 and 1.4 s for level
3 serially, or about 0.2 s with the classes over the node's threads. Against the prototype
estimate's saving of 1.3–1.8 s (Kathleen) and 0.5–0.7 s (locust) on one evaluation, a serial
build would lose; a parallel one keeps most of the saving, and every further evaluation gains
it again.

**What remains to FMM3D after H4.** On W1 at kD = 500 ours would be 0.17–0.37× `fastker`'s time
at the same ε (Kathleen 0.18–0.32 by the bound, 0.23–0.37 by the prototype; locust 0.17–0.29
and 0.21–0.32), against 0.24–0.41× now: H4 does not decide the comparison inside D1. Where FMM3D is
faster (§2.3) H4 changes nothing.

### 3.2 The cheaper levers

**The build.** It is 39–43 % of W1 at kD = 500's build plus one evaluation on Kathleen and
49–62 % on locust, and 46–96 % in every case where `fastker` is faster. It hardly uses the
threads: W1 at kD = 500 at 10⁻⁶ builds in 5.5 s on one Kathleen core and 5.7, 4.6, 4.0 and
4.0 s at 1 × 40, 2 × 20, 4 × 10 and 8 × 5 (locust: 4.5 s on one core, 4.6–2.4 s on the node).
It follows the degrees, not N: 4.0–8.6 s on W1 at kD = 500 (p up to 135, and the raised
degrees at 10⁻⁸), 0.8–1.2 s on W3 at kD = 100 (4·10⁶ points, p up to 38). The harness does not
time the build by part, so which part (most likely the per-level tables, built on every
rank) is not measured here.

**Level 2's Rotation M2L.** It splits a level's work by target box only (`driver::m2l` through
`Execution::boxes`, one rayon task per box, its V row in series), so level 2's 64 boxes keep
at most 64 tasks and the level's time is the busiest box's row: on one core it takes 50.9 s
(Kathleen) and 26.5 s (locust), on the node 2.6 s at 2 × 20 and 1.3 s at 4 × 18, 19× on 40
cores and 20× on 72 (level 3: 26× and 50×). At p = 131 its rotation blocks (49 polar angles,
601 MB) exceed the default block budget of 512 MiB and are formed per V pair. Storing them
(`--block-budget 2048`, the stage `blocks`):

Kathleen:

| ε | split | build s: default, stored | evaluate s: default, stored | level 2 M2L s: default, stored | level 3 M2L s: default, stored | total ÷ | checksums |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| 1e-3 | 1x40 | 5.058, 5.509 | 5.915, 5.325 | 2.180, 1.677 (stored) | 0.798, 0.819 | 0.99 | equal |
| 1e-3 | 2x20 | 3.972, 4.458 | 5.879, 5.150 | 2.189, 1.481 (stored) | 0.784, 0.833 | 0.98 | equal |
| 1e-3 | 4x10 | 3.557, 4.019 | 7.427, 6.055 | 3.684, 2.944 (stored) | 0.825, 0.780 | 0.92 | equal |
| 1e-3 | 8x5 | 3.357, 3.833 | 10.746, 9.144 | 6.348, 4.923 (stored) | 1.597, 1.666 | 0.92 | equal |
| 1e-6 | 1x40 | 5.732, 6.174 | 7.060, 6.694 | 2.554, 2.166 (stored) | 1.062, 1.152 | 1.01 | equal |
| 1e-6 | 2x20 | 4.576, 5.070 | 7.138, 6.333 | 2.645, 1.824 (stored) | 1.031, 1.126 | 0.97 | equal |
| 1e-6 | 4x10 | 4.015, 4.686 | 8.490, 8.079 | 4.730, 3.506 (stored) | 1.020, 1.089 | 1.02 | equal |
| 1e-6 | 8x5 | 3.981, 4.532 | 13.026, 11.300 | 7.585, 5.782 (stored) | 2.216, 2.392 | 0.93 | equal |
| 1e-8 | 1x40 | 9.748, 11.025 | 11.171, 10.564 | 3.052, 2.438 (stored) | 1.300, 1.329 | 1.03 | equal |
| 1e-8 | 2x20 | 8.575, 9.758 | 11.312, 10.184 | 2.962, 2.168 (stored) | 1.278, 1.256 | 1.00 | equal |
| 1e-8 | 4x10 | 8.353, 9.650 | 13.197, 11.819 | 4.859, 3.986 (stored) | 1.193, 1.371 | 1.00 | equal |
| 1e-8 | 8x5 | 8.284, 9.581 | 17.958, 15.743 | 8.733, 6.528 (stored) | 2.518, 2.571 | 0.97 | equal |

locust:

| ε | split | build s: default, stored | evaluate s: default, stored | level 2 M2L s: default, stored | level 3 M2L s: default, stored | total ÷ | checksums |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| 1e-3 | 1x72 | 4.277, 4.439 | 1.742, 1.503 | 0.581, 0.357 (stored) | 0.184, 0.170 | 0.99 | equal |
| 1e-3 | 2x36 | 3.081, 3.222 | 1.721, 1.551 | 0.592, 0.423 (stored) | 0.191, 0.194 | 0.99 | equal |
| 1e-3 | 4x18 | 2.338, 2.532 | 2.141, 1.874 | 1.095, 0.807 (stored) | 0.208, 0.204 | 0.98 | equal |
| 1e-3 | 8x9 | 1.986, 2.169 | 2.990, 2.573 | 1.932, 1.512 (stored) | 0.240, 0.245 | 0.95 | equal |
| 1e-6 | 1x72 | 4.649, 4.921 | 2.261, 2.010 | 0.703, 0.459 (stored) | 0.236, 0.232 | 1.00 | equal |
| 1e-6 | 2x36 | 3.433, 3.645 | 2.307, 2.068 | 0.739, 0.518 (stored) | 0.270, 0.260 | 1.00 | equal |
| 1e-6 | 4x18 | 2.705, 2.899 | 2.843, 2.516 | 1.333, 0.994 (stored) | 0.266, 0.281 | 0.98 | equal |
| 1e-6 | 8x9 | 2.360, 2.564 | 3.854, 3.366 | 2.398, 1.901 (stored) | 0.369, 0.342 | 0.95 | equal |
| 1e-8 | 1x72 | 5.794, 6.189 | 3.095, 2.868 | 0.767, 0.531 (stored) | 0.283, 0.284 | 1.02 | equal |
| 1e-8 | 2x36 | 5.168, 5.594 | 3.169, 2.920 | 0.815, 0.577 (stored) | 0.313, 0.323 | 1.02 | equal |
| 1e-8 | 4x18 | 4.707, 5.137 | 3.732, 3.419 | 1.468, 1.139 (stored) | 0.344, 0.335 | 1.01 | equal |
| 1e-8 | 8x9 | 4.579, 5.026 | 4.886, 4.379 | 2.665, 2.157 (stored) | 0.376, 0.409 | 0.99 | equal |

Storing them costs 0.1–1.3 s more build (the 49 block sets at p = 131 built once per rank)
and saves 0.2–2.2 s of level 2's M2L: build plus one evaluation 0.92–1.03×. On one core level
2's 50.9 s become 41.8 s (Kathleen) and 26.5 s become 20.6 s (locust): forming the blocks is
18–22 % of a pair, the rest the pair's own O(p³) work. Per pair on one core our Rotation takes
20.3 ms at p = 131 and 2.2 ms at p = 72 on Kathleen, 10.6 and 1.08 ms on locust, where S6's
scalar prototype took 3.0 ms at p = 128 and about 0.57 ms at p = 72 on a Grace core: on the
same core ours is 1.9× the prototype at p = 72 and 3.3× at p = 131 (2.6× with the blocks
stored, scaled by (131/128)³). (S6's prototype applied one complex expansion to one class;
whether the two differ in work, not only in speed, is not checked here.)

### 3.3 W1 at kD = 1000 (outside D1)

W1 at kD = 1000 (N = 7,957,748; CH4.4's largest case, outside D1) at the defaults, one split
per machine (Kathleen 2 × 20, locust 4 × 18; launches a and b agree bit for bit), no FMM3D
run. Every error within ε (at most 0.30 ε, ∇φ on locust at 10⁻⁶). CH4.0c's runs of this case (1 × 40, before
CH4.0d1–d3) took 55–93 s to build and 138–300 s to evaluate under the default.

Kathleen:

| Case | ε | split | leaf | build s | evaluate s | ours s | φ, ∇φ |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| w1-1000 | 1e-3 | 2x20 | 256 | 28.953 | 42.756 | 71.709 | 1.07e-05, 1.97e-05 |
| w1-1000 | 1e-6 | 2x20 | 256 | 33.826 | 51.519 | 85.345 | 1.08e-07, 2.60e-07 |
| w1-1000 | 1e-8 | 2x20 | 256 | 68.045 | 67.914 | 135.959 | 1.67e-09, 2.15e-09 |

locust:

| Case | ε | split | leaf | build s | evaluate s | ours s | φ, ∇φ |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| w1-1000 | 1e-3 | 4x18 | 128 | 16.953 | 15.947 | 32.900 | 2.48e-05, 6.74e-05 |
| w1-1000 | 1e-6 | 4x18 | 128 | 20.089 | 19.461 | 39.550 | 1.18e-07, 2.96e-07 |
| w1-1000 | 1e-8 | 4x18 | 256 | 37.771 | 24.032 | 61.803 | 1.67e-09, 2.15e-09 |

The estimate (levels 2–4 on RCR, p = 234–251, 123–135, 66–76):

Kathleen:

| Case | ε | ours s | levels on RCR at k a ≥ 8 (level:p) | their M2L s | share | after H4, bound: s | after H4, prototype: s | ceiling (all M2L at k a ≥ 8) s | fastker s | after H4 ÷ fastker (bound, prototype) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| w1-1000 | 1e-3 | 71.709 | 2:234, 3:123, 4:66 | 30.344 | 42% | 41.365 | 61.365 | 37.939 | – | – |
| w1-1000 | 1e-6 | 85.345 | 2:245, 3:131, 4:72 | 36.180 | 42% | 49.165 | 73.502 | 44.933 | – | – |
| w1-1000 | 1e-8 | 135.959 | 2:251, 3:135, 4:76 | 39.052 | 29% | 96.907 | 123.329 | 90.125 | – | – |

locust:

| Case | ε | ours s | levels on RCR at k a ≥ 8 (level:p) | their M2L s | share | after H4, bound: s | after H4, prototype: s | ceiling (all M2L at k a ≥ 8) s | fastker s | after H4 ÷ fastker (bound, prototype) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| w1-1000 | 1e-3 | 32.900 | 2:234, 3:123, 4:66 | 13.007 | 40% | 19.893 | 27.425 | 19.283 | – | – |
| w1-1000 | 1e-6 | 39.550 | 2:245, 3:131, 4:72 | 15.307 | 39% | 24.243 | 33.273 | 23.529 | – | – |
| w1-1000 | 1e-8 | 61.803 | 2:251, 3:135, 4:76 | 16.884 | 27% | 44.919 | 54.945 | 43.712 | – | – |

Per level at 10⁻⁶:

| ε | split | level | p | k a | ran | V pairs | boxes | M2L s | ns per pair |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1e-6 | 2x20 | 2 | 245 | 216.5 | rotation | 2504 | 64 | 20.0750 | 8017172 |
| 1e-6 | 2x20 | 3 | 131 | 108.3 | rotation | 12216 | 512 | 11.0284 | 902784 |
| 1e-6 | 2x20 | 4 | 72 | 54.1 | rotation | 54000 | 3840 | 5.0764 | 94007 |
| 1e-6 | 2x20 | 5 | 41 | 27.1 | planewave i2 | 228048 | 20160 | 1.9870 | 8713 |
| 1e-6 | 2x20 | 6 | 25 | 13.5 | planewave i2 | 912182 | 73856 | 2.2447 | 2461 |
| 1e-6 | 2x20 | 7 | 16 | 6.8 | planewave i2 | 2813175 | 153104 | 2.8677 | 1019 |
| 1e-6 | 4x18 | 2 | 245 | 216.5 | rotation | 2504 | 64 | 11.0465 | 4411531 |
| 1e-6 | 4x18 | 3 | 131 | 108.3 | rotation | 12216 | 512 | 3.1856 | 260777 |
| 1e-6 | 4x18 | 4 | 72 | 54.1 | rotation | 54000 | 3840 | 1.0744 | 19897 |
| 1e-6 | 4x18 | 5 | 41 | 27.1 | planewave i2 | 228048 | 21120 | 0.3586 | 1573 |
| 1e-6 | 4x18 | 6 | 25 | 13.5 | planewave i2 | 912182 | 84608 | 0.3561 | 390 |
| 1e-6 | 4x18 | 7 | 16 | 6.8 | planewave i2 | 3449399 | 258776 | 0.5535 | 160 |
| 1e-6 | 4x18 | 8 | 15 | 3.4 | planewave i2 | 5542930 | 282632 | 0.9798 | 177 |

Here the bound is 29–42 % of build plus one evaluation on Kathleen and 27–40 % on locust
(58–71 % and 70–82 % of the evaluation), the prototype estimate 9–14 % and 11–17 %
(19–24 % and 29–34 % of the evaluation). H4's transfer functions at p = 245 (about 0.45 s per
class by S6's interpolation) add about 21 s to the build for levels 2–4 serially, about 0.6 s
over the node's threads. Level 2 (p = 245, 64 boxes) alone takes 20.1 s on Kathleen and 11.0 s
on locust: the coarse RCR levels' parallelism (§3.2) matters more as kD grows, and the build
(29–68 s on Kathleen, 40–50 % of build plus one evaluation) as much.

## 4. Kathleen against locust (review B5)

W1 at kD = 500, 10⁻⁶, each at its best split (A Kathleen 2 × 20, B locust 4 × 18; the trees
differ below level 6: 256 points per leaf on x86, 128 on NEON):

| level | p | ran | Kathleen s | locust s | ÷ |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2 | 131 | rotation / rotation | 2.6448 | 1.3329 | 1.98 |
| 3 | 72 | rotation / rotation | 1.0309 | 0.2662 | 3.87 |
| 4 | 41 | planewave i2 / planewave i2 | 0.5595 | 0.1290 | 4.34 |
| 5 | 25 | planewave i2 / planewave i2 | 0.5382 | 0.0955 | 5.64 |
| 6 | 16 | planewave i2 / planewave i2 | 0.7180 | 0.1412 | 5.08 |
| P2P s | – | – | 0.3945 | 0.3019 | 1.31 |
| M2L s | – | – | 5.4914 | 2.2038 | 2.49 |
| P2L s | – | – | 0.0007 | 0.0007 | 1.00 |
| M2P s | – | – | 0.0004 | 0.0004 | 1.00 |
| P2M, M2M, L2L, L2P s | – | – | 0.9983 | 0.2953 | 3.38 |
| build s | – | – | 4.5760 | 2.7050 | 1.69 |
| evaluate s | – | – | 7.1380 | 2.8430 | 2.51 |

On one core each (the stage `t1`, the same configuration on 1 × 1):

| level | p | Kathleen s | locust s | Kathleen ÷ locust | Kathleen 1 × 1 ÷ 2 × 20 | locust 1 × 1 ÷ 4 × 18 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 131 | 50.86 | 26.52 | 1.92 | 19.2 | 19.9 |
| 3 | 72 | 27.10 | 13.22 | 2.05 | 26.3 | 49.7 |
| 4 | 41 | 10.52 | 4.43 | 2.37 | 18.8 | 34.4 |
| 5 | 25 | 11.86 | 5.12 | 2.32 | 22.0 | 53.6 |
| evaluation | – | 153.5 | 110.9 | 1.38 | 21.5 | 39.0 |
| build | – | 5.54 | 4.54 | 1.22 | 1.21 | 1.68 |

(Level 6 and below and the evaluation's total compare different trees.) So the 2–5.6× per
level at the nodes is a per-core factor of 1.9–2.4 on the M2L levels (f64, the same
kernels' NEON and AVX-512 paths) times
the core ratio 1.8, with each node's evaluation at 54 % of linear (21.5× on 40 cores, 39.0×
on 72); the review's 3.3–5.4× (CH6.2b, 1 × 40 on Kathleen, before CH4.0–CH4.0d3) also carried
the one-rank split, 1.05–1.11× slower than 2 × 20 here. Clock, vector width and memory are
not separated here. Every H4 gate runs on Kathleen; its times are about 2–2.5×
locust's at the best split.

## 5. FMM3D's error steps

FMM3D's errors move in steps of its order and quadrature tables: on W3 at kD = 30 (`default`,
Kathleen) the φ error is 1.0e-4 at eps 10⁻³, 2.1e-8 at 3·10⁻⁴, 1.6e-8 from 10⁻⁴ to 10⁻⁶ and
8.0e-12 from 3·10⁻⁷; so at matched accuracy FMM3D runs 3·10⁻⁴ for our 10⁻³ and 10⁻⁶ alike on
several cases, and `>` rows (no grid eps as accurate as ours in both fields down to 10⁻⁹) are
our 10⁻⁸ runs on small trees, whose errors are 10⁻¹¹–10⁻¹² (P2P-dominated). The raw rows are
in `fmm3d-<case>.md` of each machine's archive.

## 6. Runs

- Kathleen (one exclusive node each, `singlenode`, 2 × 20 Cascade Lake cores, Xeon Gold 6248):
  jobs 634358 (ours, W1 at kD = 500 and W3 at kD = 100; node-c11a-089), 634359 (ours, the
  matrix; node-c11b-008), 634367 (FMM3D, W1 at kD = 500; node-c11a-089), 634368 (FMM3D, W3 at
  kD = 100; node-c11b-008), 634369 (FMM3D, the matrix; node-c11b-008), 634370 (one core;
  node-c11b-008), 634373 (blocks; node-c11a-089), 634380 (W1 at kD = 1000; node-c11a-089). Branch `tbetcke/h-ch4-0e`
  at 2f25bcd (ours), 1342417 (FMM3D, one core), 47c5aeb (blocks), 07c2ae7 (kD = 1000); release,
  rustc 1.99.0, BLAS threads 1. Jobs 634360–634363 ran FMM3D on one thread (GNU `nproc`
  honours `OMP_NUM_THREADS`, which the script had set to 1 first) and were cancelled; their
  outputs are kept apart as `invalid-fmm3d-1thread/` and not used.
- locust (GH200, 72 Neoverse V2 cores, Open MPI 5.0.10): the same stages in sequence on
  2026-11-15/16, the load 0.0 before the first launch and 0.1–1.2 between the stages (the
  load averages logged before each launch are this run's own previous launch); no other
  user's process at the checks.
- Raw outputs: `~/projects/fmm-archive/{kathleen,locust}/logs/ch4-0e/`; the scripts
  `tools/helmholtz/ch4-0e-runs.sh` and `ch4-0e-tables.py` (the tables of §2–§4 are its
  `compare`, `summary`, `splits`, `levels`, `estimate`, `blocks`, `machines` and `t1` steps).

## 7. Questions for sign-off

1. **H4's go-ahead and order.** *Recommended*: H4 goes ahead, for kD beyond D1 (CH4.4's
   kD = 1000, §3.3) and the high-frequency classes CH4.3 calibrates, but after two new tasks
   that the measurements point to inside D1, each small and each Helmholtz-only:
   - **CH4.0h The Helmholtz build by part, and parallel**: time the build by part in the
     harness, then parallelise its serial parts (most likely the per-level tables) over the
     threads, bit for bit; *done when* W1 at kD = 500's build at the node is at most half its
     one-core time, the output unchanged.
   - **CH4.0i The coarse RCR levels**: split a level's Rotation work below one box where its
     boxes are fewer than the threads (by V-row segments with a fixed-order sum, or by
     output degrees), and check our RCR per pair against S6's prototype (2–2.7× here); *done
     when* level 2 of W1 at kD = 500 runs at ≥ 70 % of linear on the node or the report shows
     why not, deterministic and rank-invariant (the shared driver's split is Laplace's too:
     constraint 1, so the split lives in the Helmholtz operator or is bit for bit for
     Laplace).

   Then CHA.1–CHA.3, CH1.3d and CH4.1–CH4.5 as listed. The priority list's reason for H4
   ("the far field at kD = 500 is mostly the high-frequency levels' RCR") is replaced by this
   report's §3. *Alternatives*: (a) the list as it stands (CHA.1–CHA.3, CH1.3d, H4), the two
   tasks off the list; (b) H4 off the priority list until a use beyond kD = 500 asks for it,
   the two tasks next.
2. **The block budget.** *Recommended*: keep `DEFAULT_BLOCK_BUDGET` = 512 MiB; storing level 2's
   blocks at p = 131 (601 MB per rank) changes build plus one evaluation by 0.92–1.03× (§3.2)
   and CH4.0i's split is the better fix. *Alternative*: raise it to 1 GiB per rank.

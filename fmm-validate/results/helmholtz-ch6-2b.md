# Helmholtz CH6.2b: the M2L timed per level

Helmholtz task CH6.2b ([docs/helmholtz/tasks/CH6.2b.md](../../docs/helmholtz/tasks/CH6.2b.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH6.2 (split on this branch: our strategies per level here, FMM3D and D8 in CH6.2c),
§2 (constraints 2, 7, 9), §5.3, §6.4. Measured on:

- **the M6** (Apple M6, 24 GB, macOS; `tools/m6/machine.md`): one rank × 6 threads, under
  the quiet gate of `tools/helmholtz/ch6-2b-runs.sh` (one launch repeated for contention),
  on 2026-11-04, binary of cd67415 (the branch's code; the seconds printed to the µs);
- **locust** (NVIDIA GH200, 72 Neoverse-V2 cores, `tools/gh200/`; idle, load average 0.00
  before): 1 × 72 for every case, and 8 × 9 and 72 × 1 for the large ones, on 2026-11-04,
  source 1e22995 (the W3 large cases at 8 × 9 and 72 × 1 from 8fb9f84, whose code is the
  same; §6);
- **Kathleen**: pending (§5).

Release builds, rustc 1.99.0, Open MPI 5.0.11 (M6) and 5.0.10 (locust), BLAS threads 1,
the P2P kernel `Auto` (NEON on both), the default waves budget (8 GiB per node). Commands:
§8.

## Summary

- **The tool.** Every M2L level call of the Helmholtz operator is now timed and summed per
  level (`HelmholtzOperator::m2l_times`; two clock reads per call, no value changed), and
  `helmholtz_accuracy --levels` prints, per run and level, the degree, the method that ran,
  the V pairs, the M2L's seconds (the slowest rank's) and the ns per pair. Each case ran
  Rotation, Batched (fused), Auto and the exponential M2L (the rule, `PlaneWave`) in one
  launch and again in the reverse order (ABBA), each build evaluated twice and the second
  evaluation's times kept.
- **At ε = 10⁻⁶ the exponential M2L is the fastest method on almost every level below the
  coarsest ones.** On the M6 (6 threads) its M2L took 0.19–0.40× Rotation's on W1 at
  kD = 100, W3 at kD = 30 and the eight W4 cases, and the evaluation 0.27–0.52× Auto's,
  within 1.00–1.01× of the per-level best there (1.00–1.16× on all 13 cases). On locust at
  1 × 72 the large cases gave W1 at kD = 500 0.45× and W3 at kD = 100 0.21× Rotation's M2L
  (evaluation 0.49× and 0.30× Auto's), within 1.00–1.04× of the per-level best; the W4
  cases 0.39–1.06× Rotation's, held back by the coarse levels (below) (§2, §3).
- **At ε = 10⁻³ no strategy is best on every level.** Batched (Auto's choice at p ≤ 10)
  wins the fine levels of degree 7, the exponential M2L the intermediate ones (p ≈ 11–36),
  Rotation the coarsest. A per-level choice beats the best single strategy by up to 1.32×
  in M2L on the M6 and 1.24× on locust (most cases 1.0–1.1×, within the runs' noise). The
  exponential M2L alone beats Auto on W1 at kD = 100 and the W4 cube, Plummer and sphere at
  kD = 100 on the M6 (0.43–0.76× Auto's M2L) and on W1 at kD = 500 on locust (0.66×), and
  loses elsewhere (to 2.2×) (§2–§4).
- **Coarse levels, and many threads.** The exponential M2L's level call splits its merge and
  shift over parents, so a level with few parents keeps few threads busy, and its fixed
  cost per level call (the waves of every frame and direction) does not shrink with the
  pairs. Level 2 (64 boxes, 8 parents, 3,096 pairs) took 9–24× Rotation's time on locust at
  72 threads (Rotation splits over its 64 targets) and 0.9–3.3× on the M6 at 6 threads
  (2.2–3.3× at 10⁻³, 0.9–1.4× at 10⁻⁶). On locust that makes the exponential M2L slower
  than Rotation on W1 at kD = 10–100 and W3 at kD = 10 at both ε and on the W4 cube and
  sphere at kD = 100 at 10⁻⁶ (1.03–1.06× Rotation's M2L, 1.65–1.88× the per-level best).
  The rule cannot choose by the thread count, which would make the output depend on it
  (proposal §2, constraint 2): a finer split of the coarse levels' level call, or a rule by
  the level's boxes (a function of the tree, constraint 9), is CH6.2c's to weigh with D8
  (§7).
- **Batched above Auto's crossover is slow**: 2–10× Rotation's M2L at p ≥ 15 on both
  machines (its tables, hundreds of MB per level, streamed per block), as CH3.2c measured;
  Auto never takes it there.
- **Ranks × threads** (locust, §4): one rank × 72 threads evaluated W1 at kD = 500 fastest
  under every strategy (at 10⁻⁶ the exponential M2L 5.0 s against 7.8 s at 8 × 9 and
  16.0 s at 72 × 1; Rotation 10.3, 13.5, 20.7 s); on W3 at kD = 100 1 × 72 and 8 × 9 are
  within 0.90–1.03× of each other and 72 × 1 is the slowest. The exponential M2L's gain at
  10⁻⁶ held at every split (0.23–0.58× Rotation's M2L at 8 × 9, 0.38–0.70× at 72 × 1).
  The default waves budget held every case in memory: rank 0's peak 17.2 GB at 1 × 72 on
  W1 at kD = 500, 3.1 and 1.9 GB per rank at 8 × 9 and 72 × 1.
- **Accuracy unchanged**: the exponential M2L's errors are 0.92–1.06× Rotation's on every
  case (the rule of CH6.1e), and every evaluation's checksum is its strategy's on both
  machines' ABBA pairs.

## 1. Method

The harness is CH3.1c's `nd_fmm_validate::helmholtz_accuracy` (proposal §9.1's workloads,
complex charges, gradients, 1,000 sampled targets against the f64 direct sum) with this
task's `Row::{m2l_seconds, v_pairs, first_evaluate}` and `Case::evaluations`. Per run:

- **M2L per level**: the M2L level calls' seconds of the second evaluation of one build
  (the first touches the buffers: on locust's large cases the first evaluation took up to
  5.0 s more, Batched at 8 × 9 on W1 at kD = 500, the exponential M2L up to 0.63 s more),
  the slowest rank's per level; the Σ rows add these per-level maxima, which on several
  ranks may come from different ranks.
- **V pairs**: each target counted by the rank that owns it, summed over the ranks, so the
  same at every split; "ns per pair" is the slowest rank's time over all pairs, a machine
  throughput, not a per-rank cost, on several ranks.
- **ABBA**: each case's launch with `--m2l rotation,batched-fused,auto,planewave` and again
  with the order reversed; the tables give the mean of the two, "ABBA max" the larger
  ratio of a strategy's two launches on any level of the case.
- **The methods a level ran**: Rotation; Batched (fused) where its tables fit the dense
  budget (1 GiB per rank), Rotation elsewhere; Auto Batched (fused) at p ≤ 10 on these
  machines (NEON, `auto_batched_max_degree`), Rotation elsewhere; the exponential M2L where
  the rule gives a quadrature (iprec 2 at ε ≥ 10⁻⁶ wherever κ_l lies in the tables),
  Rotation elsewhere. The per-level tables mark a run that ran another method there in
  brackets. **Fastest** and **best per level** compare the distinct methods (each timed by
  the first strategy that ran it, so that two runs of Rotation never compete on noise).
- The cases: CH6.1e's matrix at ε = 10⁻³ and 10⁻⁶ (W1 at kD = 10, 30, 100; W3 at kD = 10,
  30; the W4 cube, sphere, Plummer and clusters at N = 10⁵, kD = 1 and 100) on every machine
  at one rank; W1 at kD = 500 and W3 at kD = 100 (N = 2.0 × 10⁶ and 4.0 × 10⁶) on locust
  and Kathleen at each split (not on the M6: 14–17 GB of peak).

`tools/helmholtz/ch6-2b-tables.py` makes the tables from the launches' outputs (§8).

## 2. The summary per case

Per machine, case and split: the M2L summed over the levels per strategy (s), the
per-level best (each level's fastest method), each strategy over it, the evaluation per
strategy (s), the ABBA spread and the exponential M2L's errors over Rotation's (φ/∇φ).
"b-f" is Batched (fused), "pw" the exponential M2L.

| machine | workload | kD | ε | split | M2L Σ rot. | b-f | auto | pw | best per level | rot./best | b-f/best | auto/best | pw/best | evaluate rot. | b-f | auto | pw | ABBA max | pw ÷ rot. error φ/∇φ |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| m6 | W1 | 10 | 1e-3 | 1×6 | 0.001 | 0.002 | 0.002 | 0.002 | 0.001 | 1.00 | 2.30 | 2.11 | 2.59 | 0.002 | 0.003 | 0.003 | 0.003 | 1.40 | 1.000/1.000 |
| m6 | W1 | 10 | 1e-6 | 1×6 | 0.004 | 0.040 | 0.005 | 0.005 | 0.004 | 1.00 | 9.65 | 1.12 | 1.16 | 0.005 | 0.041 | 0.006 | 0.006 | 1.29 | 1.005/0.995 |
| m6 | W1 | 30 | 1e-3 | 1×6 | 0.012 | 0.026 | 0.012 | 0.017 | 0.012 | 1.00 | 2.25 | 1.03 | 1.49 | 0.017 | 0.031 | 0.017 | 0.022 | 1.15 | 1.000/1.000 |
| m6 | W1 | 30 | 1e-6 | 1×6 | 0.048 | 0.181 | 0.055 | 0.026 | 0.025 | 1.95 | 7.35 | 2.23 | 1.08 | 0.056 | 0.189 | 0.063 | 0.034 | 1.43 | 1.000/1.000 |
| m6 | W1 | 100 | 1e-3 | 1×6 | 0.443 | 0.731 | 0.420 | 0.317 | 0.242 | 1.83 | 3.02 | 1.74 | 1.31 | 0.573 | 0.858 | 0.554 | 0.447 | 1.34 | 1.000/1.000 |
| m6 | W1 | 100 | 1e-6 | 1×6 | 1.276 | 2.242 | 1.174 | 0.481 | 0.477 | 2.67 | 4.70 | 2.46 | 1.01 | 1.592 | 2.552 | 1.516 | 0.791 | 1.31 | 0.995/0.980 |
| m6 | W3 (N = 108849) | 30 | 1e-3 | 1×6 | 0.164 | 0.109 | 0.096 | 0.116 | 0.095 | 1.73 | 1.15 | 1.01 | 1.22 | 0.251 | 0.189 | 0.178 | 0.195 | 1.24 | 1.000/1.000 |
| m6 | W3 (N = 108849) | 30 | 1e-6 | 1×6 | 0.944 | 2.132 | 1.109 | 0.203 | 0.201 | 4.69 | 10.59 | 5.51 | 1.01 | 1.077 | 2.251 | 1.236 | 0.331 | 1.40 | 0.973/0.964 |
| m6 | W3 (N = 4032) | 10 | 1e-3 | 1×6 | 0.002 | 0.004 | 0.004 | 0.006 | 0.002 | 1.00 | 2.23 | 2.18 | 2.96 | 0.014 | 0.016 | 0.017 | 0.018 | 1.08 | 1.000/1.000 |
| m6 | W3 (N = 4032) | 10 | 1e-6 | 1×6 | 0.012 | 0.074 | 0.013 | 0.012 | 0.012 | 1.04 | 6.28 | 1.07 | 1.04 | 0.044 | 0.108 | 0.045 | 0.044 | 1.17 | 0.991/0.947 |
| m6 | W4 clusters (N = 100000) | 1 | 1e-3 | 1×6 | 0.264 | 0.164 | 0.167 | 0.190 | 0.162 | 1.63 | 1.01 | 1.03 | 1.17 | 0.529 | 0.412 | 0.415 | 0.436 | 1.22 | 1.000/1.000 |
| m6 | W4 clusters (N = 100000) | 1 | 1e-6 | 1×6 | 1.691 | 3.266 | 1.602 | 0.361 | 0.361 | 4.69 | 9.05 | 4.44 | 1.00 | 2.419 | 3.971 | 2.379 | 1.056 | 1.46 | 0.969/0.916 |
| m6 | W4 clusters (N = 100000) | 100 | 1e-3 | 1×6 | 0.333 | 0.218 | 0.207 | 0.237 | 0.192 | 1.73 | 1.13 | 1.07 | 1.23 | 0.617 | 0.475 | 0.470 | 0.499 | 1.25 | 1.000/1.000 |
| m6 | W4 clusters (N = 100000) | 100 | 1e-6 | 1×6 | 1.700 | 3.218 | 1.767 | 0.419 | 0.418 | 4.06 | 7.69 | 4.22 | 1.00 | 2.465 | 3.925 | 2.520 | 1.141 | 1.35 | 0.970/0.921 |
| m6 | W4 cube (N = 100000) | 1 | 1e-3 | 1×6 | 0.185 | 0.106 | 0.092 | 0.099 | 0.097 | 1.91 | 1.09 | 0.95 | 1.02 | 0.258 | 0.174 | 0.169 | 0.167 | 1.33 | 1.000/1.000 |
| m6 | W4 cube (N = 100000) | 1 | 1e-6 | 1×6 | 0.973 | 1.998 | 0.965 | 0.183 | 0.183 | 5.32 | 10.93 | 5.28 | 1.00 | 1.085 | 2.104 | 1.081 | 0.291 | 1.33 | 0.983/0.928 |
| m6 | W4 cube (N = 100000) | 100 | 1e-3 | 1×6 | 0.566 | 0.945 | 0.670 | 0.287 | 0.277 | 2.04 | 3.41 | 2.42 | 1.04 | 0.678 | 1.047 | 0.776 | 0.393 | 1.35 | 1.000/1.000 |
| m6 | W4 cube (N = 100000) | 100 | 1e-6 | 1×6 | 1.140 | 2.255 | 1.298 | 0.357 | 0.357 | 3.19 | 6.31 | 3.63 | 1.00 | 1.282 | 2.388 | 1.435 | 0.492 | 1.33 | 1.003/0.990 |
| m6 | W4 plummer (N = 100000) | 1 | 1e-3 | 1×6 | 0.217 | 0.139 | 0.128 | 0.129 | 0.124 | 1.75 | 1.12 | 1.03 | 1.04 | 0.470 | 0.374 | 0.367 | 0.367 | 1.37 | 1.000/1.000 |
| m6 | W4 plummer (N = 100000) | 1 | 1e-6 | 1×6 | 1.300 | 2.216 | 1.229 | 0.252 | 0.252 | 5.17 | 8.80 | 4.88 | 1.00 | 1.979 | 2.864 | 1.906 | 0.902 | 1.44 | 1.063/0.940 |
| m6 | W4 plummer (N = 100000) | 100 | 1e-3 | 1×6 | 0.625 | 0.889 | 0.540 | 0.345 | 0.329 | 1.90 | 2.70 | 1.64 | 1.05 | 0.899 | 1.147 | 0.803 | 0.595 | 1.35 | 1.000/1.000 |
| m6 | W4 plummer (N = 100000) | 100 | 1e-6 | 1×6 | 2.063 | 2.968 | 1.895 | 0.513 | 0.513 | 4.02 | 5.79 | 3.69 | 1.00 | 2.745 | 3.613 | 2.575 | 1.152 | 1.31 | 1.013/0.955 |
| m6 | W4 sphere (N = 100000) | 1 | 1e-3 | 1×6 | 0.204 | 0.123 | 0.112 | 0.172 | 0.121 | 1.69 | 1.02 | 0.93 | 1.43 | 0.327 | 0.236 | 0.223 | 0.283 | 1.28 | 1.000/1.000 |
| m6 | W4 sphere (N = 100000) | 1 | 1e-6 | 1×6 | 1.150 | 2.602 | 1.162 | 0.333 | 0.333 | 3.46 | 7.82 | 3.49 | 1.00 | 1.461 | 2.895 | 1.486 | 0.635 | 1.43 | 0.986/0.945 |
| m6 | W4 sphere (N = 100000) | 100 | 1e-3 | 1×6 | 0.500 | 0.792 | 0.472 | 0.360 | 0.272 | 1.84 | 2.91 | 1.73 | 1.32 | 0.654 | 0.931 | 0.619 | 0.503 | 1.35 | 1.000/1.000 |
| m6 | W4 sphere (N = 100000) | 100 | 1e-6 | 1×6 | 1.372 | 2.742 | 1.369 | 0.546 | 0.543 | 2.53 | 5.05 | 2.52 | 1.00 | 1.703 | 3.058 | 1.697 | 0.882 | 1.35 | 1.003/0.978 |
| locust | W1 | 10 | 1e-3 | 1×72 | 0.000 | 0.004 | 0.004 | 0.004 | 0.000 | 1.00 | 22.72 | 22.83 | 23.86 | 0.001 | 0.005 | 0.005 | 0.006 | 1.14 | 1.000/1.000 |
| locust | W1 | 10 | 1e-6 | 1×72 | 0.001 | 0.123 | 0.001 | 0.007 | 0.001 | 1.00 | 178.84 | 1.01 | 10.55 | 0.002 | 0.124 | 0.002 | 0.010 | 1.05 | 1.005/0.995 |
| locust | W1 | 30 | 1e-3 | 1×72 | 0.002 | 0.057 | 0.009 | 0.015 | 0.002 | 1.00 | 33.56 | 5.24 | 8.88 | 0.005 | 0.060 | 0.012 | 0.018 | 1.03 | 1.000/1.000 |
| locust | W1 | 30 | 1e-6 | 1×72 | 0.006 | 0.317 | 0.006 | 0.022 | 0.006 | 1.00 | 49.09 | 0.99 | 3.34 | 0.010 | 0.322 | 0.010 | 0.025 | 1.03 | 1.000/1.000 |
| locust | W1 | 100 | 1e-3 | 1×72 | 0.053 | 0.510 | 0.054 | 0.117 | 0.048 | 1.10 | 10.54 | 1.12 | 2.41 | 0.083 | 0.541 | 0.084 | 0.147 | 1.08 | 1.000/1.000 |
| locust | W1 | 100 | 1e-6 | 1×72 | 0.132 | 0.364 | 0.132 | 0.147 | 0.088 | 1.50 | 4.15 | 1.50 | 1.68 | 0.198 | 0.429 | 0.197 | 0.212 | 1.03 | 0.995/0.980 |
| locust | W1 | 500 | 1e-3 | 1×72 | 5.274 | 5.806 | 5.090 | 3.375 | 3.091 | 1.71 | 1.88 | 1.65 | 1.09 | 5.861 | 6.379 | 5.710 | 3.930 | 1.13 | 1.000/1.000 |
| locust | W1 | 500 | 1e-3 | 72×1 | 13.795 | 14.754 | 13.900 | 13.984 | 12.329 | 1.12 | 1.20 | 1.13 | 1.13 | 13.036 | 14.200 | 13.209 | 12.926 | 1.43 | 1.000/1.000 |
| locust | W1 | 500 | 1e-3 | 8×9 | 8.032 | 8.644 | 9.837 | 6.118 | 5.707 | 1.41 | 1.51 | 1.72 | 1.07 | 8.127 | 8.734 | 9.064 | 6.247 | 2.56 | 1.000/1.000 |
| locust | W1 | 500 | 1e-6 | 1×72 | 9.481 | 15.558 | 9.532 | 4.257 | 4.264 | 2.22 | 3.65 | 2.24 | 1.00 | 10.257 | 16.329 | 10.304 | 5.023 | 1.19 | 0.973/0.984 |
| locust | W1 | 500 | 1e-6 | 72×1 | 23.098 | 32.856 | 22.014 | 16.154 | 15.479 | 1.49 | 2.12 | 1.42 | 1.04 | 20.746 | 29.857 | 20.653 | 16.044 | 1.63 | 0.973/0.984 |
| locust | W1 | 500 | 1e-6 | 8×9 | 13.334 | 21.429 | 13.025 | 7.763 | 7.625 | 1.75 | 2.81 | 1.71 | 1.02 | 13.514 | 21.135 | 13.156 | 7.835 | 1.23 | 0.973/0.984 |
| locust | W3 (N = 108849) | 30 | 1e-3 | 1×72 | 0.017 | 0.078 | 0.024 | 0.032 | 0.015 | 1.11 | 5.07 | 1.58 | 2.12 | 0.039 | 0.100 | 0.047 | 0.055 | 1.08 | 1.000/1.000 |
| locust | W3 (N = 108849) | 30 | 1e-6 | 1×72 | 0.101 | 0.393 | 0.101 | 0.049 | 0.035 | 2.92 | 11.30 | 2.91 | 1.40 | 0.131 | 0.424 | 0.131 | 0.078 | 1.06 | 0.973/0.964 |
| locust | W3 (N = 4031442) | 100 | 1e-3 | 1×72 | 1.333 | 1.344 | 0.935 | 1.319 | 0.876 | 1.52 | 1.53 | 1.07 | 1.51 | 2.023 | 2.024 | 1.615 | 1.999 | 1.12 | 1.000/1.000 |
| locust | W3 (N = 4031442) | 100 | 1e-3 | 72×1 | 1.508 | 1.521 | 1.041 | 2.665 | 1.040 | 1.45 | 1.46 | 1.00 | 2.56 | 1.980 | 1.946 | 1.508 | 3.081 | 1.08 | 1.000/1.000 |
| locust | W3 (N = 4031442) | 100 | 1e-3 | 8×9 | 1.366 | 1.472 | 0.999 | 1.456 | 0.923 | 1.48 | 1.60 | 1.08 | 1.58 | 1.821 | 1.911 | 1.481 | 1.902 | 1.25 | 1.000/1.000 |
| locust | W3 (N = 4031442) | 100 | 1e-6 | 1×72 | 8.375 | 14.701 | 8.288 | 1.772 | 1.712 | 4.89 | 8.59 | 4.84 | 1.04 | 9.344 | 15.669 | 9.270 | 2.778 | 1.26 | 1.010/0.984 |
| locust | W3 (N = 4031442) | 100 | 1e-6 | 72×1 | 9.486 | 19.344 | 9.353 | 3.630 | 3.444 | 2.75 | 5.62 | 2.72 | 1.05 | 10.231 | 19.894 | 10.105 | 4.326 | 1.14 | 1.010/0.984 |
| locust | W3 (N = 4031442) | 100 | 1e-6 | 8×9 | 8.746 | 15.893 | 8.544 | 1.971 | 1.962 | 4.46 | 8.10 | 4.36 | 1.00 | 9.315 | 16.573 | 9.290 | 2.697 | 1.45 | 1.010/0.984 |
| locust | W3 (N = 4032) | 10 | 1e-3 | 1×72 | 0.000 | 0.006 | 0.007 | 0.008 | 0.000 | 1.00 | 15.95 | 16.00 | 18.58 | 0.007 | 0.014 | 0.014 | 0.015 | 1.10 | 1.000/1.000 |
| locust | W3 (N = 4032) | 10 | 1e-6 | 1×72 | 0.002 | 0.202 | 0.002 | 0.014 | 0.002 | 1.00 | 104.11 | 1.01 | 7.24 | 0.019 | 0.219 | 0.019 | 0.031 | 1.08 | 0.991/0.947 |
| locust | W4 clusters (N = 100000) | 1 | 1e-3 | 1×72 | 0.027 | 0.056 | 0.057 | 0.041 | 0.027 | 1.00 | 2.09 | 2.09 | 1.54 | 0.084 | 0.112 | 0.113 | 0.098 | 1.14 | 1.000/1.000 |
| locust | W4 clusters (N = 100000) | 1 | 1e-6 | 1×72 | 0.166 | 0.637 | 0.165 | 0.072 | 0.064 | 2.59 | 9.95 | 2.58 | 1.13 | 0.308 | 0.780 | 0.306 | 0.214 | 1.08 | 0.969/0.916 |
| locust | W4 clusters (N = 100000) | 100 | 1e-3 | 1×72 | 0.031 | 0.115 | 0.055 | 0.070 | 0.031 | 1.00 | 3.78 | 1.81 | 2.31 | 0.087 | 0.173 | 0.114 | 0.128 | 1.11 | 1.000/1.000 |
| locust | W4 clusters (N = 100000) | 100 | 1e-6 | 1×72 | 0.168 | 0.636 | 0.168 | 0.104 | 0.072 | 2.34 | 8.82 | 2.33 | 1.44 | 0.309 | 0.778 | 0.308 | 0.245 | 1.07 | 0.970/0.921 |
| locust | W4 cube (N = 100000) | 1 | 1e-3 | 1×72 | 0.016 | 0.024 | 0.023 | 0.024 | 0.015 | 1.12 | 1.64 | 1.59 | 1.62 | 0.035 | 0.044 | 0.043 | 0.044 | 1.04 | 1.000/1.000 |
| locust | W4 cube (N = 100000) | 1 | 1e-6 | 1×72 | 0.101 | 0.512 | 0.101 | 0.039 | 0.032 | 3.12 | 15.78 | 3.13 | 1.19 | 0.127 | 0.540 | 0.127 | 0.065 | 1.06 | 0.983/0.928 |
| locust | W4 cube (N = 100000) | 100 | 1e-3 | 1×72 | 0.061 | 0.519 | 0.061 | 0.109 | 0.049 | 1.24 | 10.59 | 1.24 | 2.22 | 0.085 | 0.544 | 0.085 | 0.135 | 1.03 | 1.000/1.000 |
| locust | W4 cube (N = 100000) | 100 | 1e-6 | 1×72 | 0.125 | 0.272 | 0.125 | 0.129 | 0.069 | 1.82 | 3.97 | 1.82 | 1.88 | 0.154 | 0.302 | 0.156 | 0.159 | 1.03 | 1.003/0.990 |
| locust | W4 plummer (N = 100000) | 1 | 1e-3 | 1×72 | 0.021 | 0.059 | 0.059 | 0.041 | 0.020 | 1.05 | 2.96 | 2.97 | 2.07 | 0.083 | 0.122 | 0.123 | 0.104 | 1.11 | 1.000/1.000 |
| locust | W4 plummer (N = 100000) | 1 | 1e-6 | 1×72 | 0.129 | 0.648 | 0.130 | 0.071 | 0.062 | 2.09 | 10.47 | 2.10 | 1.15 | 0.291 | 0.805 | 0.292 | 0.232 | 1.08 | 1.063/0.940 |
| locust | W4 plummer (N = 100000) | 100 | 1e-3 | 1×72 | 0.065 | 0.566 | 0.087 | 0.127 | 0.057 | 1.13 | 9.89 | 1.53 | 2.22 | 0.127 | 0.629 | 0.150 | 0.190 | 1.11 | 1.000/1.000 |
| locust | W4 plummer (N = 100000) | 100 | 1e-6 | 1×72 | 0.199 | 0.729 | 0.200 | 0.165 | 0.102 | 1.96 | 7.18 | 1.97 | 1.63 | 0.344 | 0.881 | 0.354 | 0.316 | 1.21 | 1.013/0.955 |
| locust | W4 sphere (N = 100000) | 1 | 1e-3 | 1×72 | 0.019 | 0.028 | 0.028 | 0.034 | 0.018 | 1.06 | 1.53 | 1.54 | 1.85 | 0.047 | 0.057 | 0.058 | 0.063 | 1.09 | 1.000/1.000 |
| locust | W4 sphere (N = 100000) | 1 | 1e-6 | 1×72 | 0.120 | 0.499 | 0.121 | 0.059 | 0.052 | 2.31 | 9.56 | 2.32 | 1.12 | 0.183 | 0.562 | 0.184 | 0.121 | 1.06 | 0.986/0.945 |
| locust | W4 sphere (N = 100000) | 100 | 1e-3 | 1×72 | 0.056 | 0.507 | 0.055 | 0.120 | 0.050 | 1.11 | 10.12 | 1.10 | 2.39 | 0.087 | 0.540 | 0.088 | 0.152 | 1.04 | 1.000/1.000 |
| locust | W4 sphere (N = 100000) | 100 | 1e-6 | 1×72 | 0.144 | 0.380 | 0.145 | 0.152 | 0.092 | 1.57 | 4.14 | 1.58 | 1.65 | 0.211 | 0.447 | 0.212 | 0.218 | 1.04 | 1.003/0.978 |

## 3. Per level

The pattern of the summary, level by level, in four cases (seconds of M2L per level, the
mean of the ABBA pair; every case is in the appendix).

At 10⁻⁶ on the M6 the exponential M2L wins every level, the coarsest with few pairs
excepted:

#### W4 cube (N = 100000), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0532 | 0.0539 (rot.) | 0.0465 (rot.) | 0.0503 (planewave i2) | planewave i2 | 0.95 |
| 3 | 22 | 53352 | 0.2184 | 0.2265 (rot.) | 0.2184 (rot.) | 0.0624 (planewave i2) | planewave i2 | 0.29 |
| 4 | 15 | 584136 | 0.8680 | 1.9750 | 1.0333 (rot.) | 0.2445 (planewave i2) | planewave i2 | 0.28 |
| Σ | | | 1.1396 | 2.2553 | 1.2983 | 0.3573 | best 0.3573 | |
| evaluate | | | 1.282 | 2.388 | 1.435 | 0.492 | | |

The same case on locust at 72 threads: level 2 at 9.3× Rotation's time outweighs the gain
on level 4:

#### W4 cube (N = 100000), kD = 100, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0066 | 0.0067 (rot.) | 0.0066 (rot.) | 0.0619 (planewave i2) | rotation | 9.31 |
| 3 | 22 | 53352 | 0.0266 | 0.0269 (rot.) | 0.0269 (rot.) | 0.0319 (planewave i2) | rotation | 1.20 |
| 4 | 15 | 584136 | 0.0917 | 0.2383 | 0.0914 (rot.) | 0.0353 (planewave i2) | planewave i2 | 0.39 |
| Σ | | | 0.1250 | 0.2720 | 0.1250 | 0.1291 | best 0.0686 | |
| evaluate | | | 0.154 | 0.302 | 0.156 | 0.159 | | |

The large W1 at 10⁻⁶ (locust, 1 × 72): levels 2 and 3 (p = 131 and 72, outside the
quadratures' tables) run Rotation under every strategy; the exponential M2L takes levels
4–7 at 0.22–0.40× Rotation's time:

#### W1 (N = 1989437), kD = 500, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 131 | 3096 | 0.9349 | 0.9290 (rot.) | 0.9509 (rot.) | 0.9248 (rot.) | rotation | nan |
| 3 | 72 | 53352 | 0.9772 | 0.9913 (rot.) | 0.9665 (rot.) | 0.9806 (rot.) | rotation | nan |
| 4 | 41 | 511200 | 1.3946 | 1.4099 (rot.) | 1.4073 (rot.) | 0.3113 (planewave i2) | planewave i2 | 0.22 |
| 5 | 25 | 2935296 | 1.9021 | 1.9167 (rot.) | 1.9197 (rot.) | 0.4623 (planewave i2) | planewave i2 | 0.24 |
| 6 | 16 | 10439280 | 1.9678 | 5.4702 | 1.9835 (rot.) | 0.6577 (planewave i2) | planewave i2 | 0.33 |
| 7 | 15 | 14492220 | 2.3043 | 4.8404 | 2.3043 (rot.) | 0.9203 (planewave i2) | planewave i2 | 0.40 |
| Σ | | | 9.4809 | 15.5576 | 9.5321 | 4.2569 | best 4.2637 | |
| evaluate | | | 10.257 | 16.329 | 10.304 | 5.023 | | |

At 10⁻³ a level of each method: Rotation on levels 2–3, the exponential M2L on level 4,
Batched (Auto) on levels 5–6 of degree 7 (W3 at kD = 100, locust, 1 × 72):

#### W3 (N = 4031442), kD = 100, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0043 | 0.0043 (rot.) | 0.0043 (rot.) | 0.0531 (planewave i2) | rotation | 12.38 |
| 3 | 18 | 53352 | 0.0157 | 0.4502 | 0.0155 (rot.) | 0.0264 (planewave i2) | rotation | 1.68 |
| 4 | 11 | 584136 | 0.0415 | 0.0622 | 0.0415 (rot.) | 0.0286 (planewave i2) | planewave i2 | 0.69 |
| 5 | 7 | 5398920 | 0.1377 | 0.0864 | 0.0864 (batched-fused) | 0.1458 (planewave i2) | batched-fused | 1.69 |
| 6 | 7 | 46298376 | 1.1343 | 0.7408 | 0.7871 (batched-fused) | 1.0649 (planewave i2) | batched-fused | 1.44 |
| Σ | | | 1.3335 | 1.3439 | 0.9347 | 1.3188 | best 0.8758 | |
| evaluate | | | 2.023 | 2.024 | 1.615 | 1.999 | | |

## 4. Ranks × threads (locust)

The large cases at 1 × 72, 8 × 9 (`--map-by slot:PE=9 --bind-to core`) and 72 × 1, the
evaluation's seconds per strategy (ABBA means; the summary has the M2L):

| case | ε | split | Rotation | Batched (fused) | Auto | exponential | rank 0 peak MB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| W1, kD = 500 | 1e-3 | 1 × 72 | 5.861 | 6.379 | 5.710 | 3.930 | 17,244 |
| W1, kD = 500 | 1e-3 | 72 × 1 | 13.036 | 14.200 | 13.209 | 12.926 | 1,937 |
| W1, kD = 500 | 1e-3 | 8 × 9 | 8.127 | 8.734 | 9.064 | 6.247 | 3,074 |
| W1, kD = 500 | 1e-6 | 1 × 72 | 10.257 | 16.329 | 10.304 | 5.023 | 17,244 |
| W1, kD = 500 | 1e-6 | 72 × 1 | 20.746 | 29.857 | 20.653 | 16.044 | 1,937 |
| W1, kD = 500 | 1e-6 | 8 × 9 | 13.514 | 21.135 | 13.156 | 7.835 | 3,074 |
| W3, kD = 100 | 1e-3 | 1 × 72 | 2.023 | 2.024 | 1.615 | 1.999 | 14,287 |
| W3, kD = 100 | 1e-3 | 72 × 1 | 1.980 | 1.946 | 1.508 | 3.081 | 1,622 |
| W3, kD = 100 | 1e-3 | 8 × 9 | 1.821 | 1.911 | 1.481 | 1.902 | 2,600 |
| W3, kD = 100 | 1e-6 | 1 × 72 | 9.344 | 15.669 | 9.270 | 2.778 | 14,287 |
| W3, kD = 100 | 1e-6 | 72 × 1 | 10.231 | 19.894 | 10.105 | 4.326 | 1,622 |
| W3, kD = 100 | 1e-6 | 8 × 9 | 9.315 | 16.573 | 9.290 | 2.697 | 2,600 |

On W1 at kD = 500 one rank × 72 threads is the fastest split under every strategy; on W3
at kD = 100 1 × 72 and 8 × 9 are within 0.90–1.03× of each other, and 72 × 1 is the
slowest (the exponential M2L at 10⁻³ 1.5× its 1 × 72 time). The exponential M2L's gain at
10⁻⁶ holds at every split (0.29–0.60× Auto's evaluation at 8 × 9, 0.43–0.78× at
72 × 1). Rank 0's peak is the 1 × 72 run's whole tree; at 8 × 9 and 72 × 1 a rank holds
its share (and its share of the default 8 GiB waves budget per node).

## 5. Kathleen (pending)

Job 628611 (one node, node-c11a-142, 43 min, 2026-11-04: the build, `matrix` and `large`
at 1 × 40) completed with every launch's exit 0; its outputs are on Kathleen
(`/scratch/scratch/ucahtbe/fmm/logs/ch6-2b/`) and not yet fetched: from 2026-11-04 10:10
the Kathleen login node closed every connection through UCL's gateway
(`kex_exchange_identification: Connection closed by remote host`), so neither the fetch nor
the second job (`large` at 4 × 10 and 40 × 1) could run. Both are the PR's pending runs;
this section and the summary take their numbers when they have run.

## 6. Runs and defects found on the way

- The run script's first version let `mpirun` read the case list from stdin, so a split
  with several ranks ran its first case only (locust's 8 × 9 and 72 × 1 ran W1 at
  kD = 500 alone); fixed (8fb9f84, every launch reads `/dev/null`) and the W3 launches
  run again. The one-rank launches (`matrix` and `large` at 1 × 72 and 1 × 40) had no
  `mpirun` and were complete.
- The macOS gate first counted the launch itself (`timeout` runs the command in a process
  group of its own, which the gate did not leave out), so every M6 launch was taken for
  contended and repeated; that run was stopped and discarded, the gate fixed (cd67415)
  and the M6's runs made again. The one repeat in the M6's log is a real contention.
- Every evaluation's checksum is its strategy's in both launches of a pair (104 pairs on
  the M6, 152 on locust), and the exponential M2L's errors are 0.92–1.06× Rotation's.
- The ABBA spread: up to 1.46× on a level on the M6 (6 threads, its known ±25 % at this
  size; CH3.4c) and up to 2.5× on locust's 8 × 9 split (W1 at kD = 500, 10⁻³); the
  summary's ratios between strategies are larger than that on the cases it draws from.

## 7. Observations for CH6.2c and D8

- **At ε ≥ 10⁻⁶ the exponential M2L is the fastest method on the levels with a quadrature
  and enough parents, and close to the per-level best as a whole** (1.00–1.01× on the M6's
  larger cases at 10⁻⁶, 1.00–1.04× on locust's large ones). A default that took it there
  would cut the evaluation to 0.27–0.52× Auto's at 10⁻⁶ on the M6 and 0.30–0.49× on the
  large cases on locust; at 10⁻³ the gain is smaller and the choice per level matters.
- **The level rule D8 needs** reads the level's degree, ε and its boxes (a function of the
  tree, constraint 9): Batched at p ≤ Auto's crossover, the exponential M2L on levels with
  a quadrature and enough boxes, Rotation on the coarsest levels and on the levels above
  the quadratures' tables. The threads cannot enter it (constraint 2), so the coarse levels'
  loss at many threads is fixed in the level call (a split of the merge and shift below the
  parent, by frame, direction or child, and of the outgoing waves by box), or the rule takes
  Rotation on levels of few parents on every machine. CH6.2c's line records it.
- **Batched above the crossover** stays slow (2–10× Rotation's M2L at p ≥ 15 on both
  machines): Auto is right not to take it.
- **The splits**: on locust one rank with all 72 cores was fastest or within 10 % of
  8 × 9, and 72 × 1 the slowest; on Kathleen (two sockets) CH3.6's 4 × 10 is pending (§5).

## 8. Reproducing

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
THREADS=6 tools/helmholtz/ch6-2b-runs.sh <out> matrix                          # the M6, unsandboxed
THREADS=72 SPLITS="1x72 8x9 72x1" tools/helmholtz/ch6-2b-runs.sh <out> matrix large   # locust
tools/helmholtz/ch6-2b-tables.py m6=<out-m6> locust=<out-locust> > tables.md
```

On Kathleen inside one-node jobs (`tools/kathleen/submit.sh --time=06:00:00 -- run.sbatch
'<build>' 'THREADS=40 tools/helmholtz/ch6-2b-runs.sh <out> matrix large'`, then
`SPLITS="4x10 40x1"` with `large`).

## Appendix: every case per level

Seconds of M2L per level (ABBA means), in brackets the method a strategy ran where it was
not its own ("rot." Rotation), the fastest method and the exponential M2L over the fastest
other method; Σ the levels summed, "evaluate" the evaluation.

### The M6 (1 × 6)

#### W1 (N = 796), kD = 10, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0009 | 0.0022 | 0.0020 (batched-fused) | 0.0024 (planewave i2) | rotation | 2.59 |
| Σ | | | 0.0009 | 0.0022 | 0.0020 | 0.0024 | best 0.0009 | |
| evaluate | | | 0.002 | 0.003 | 0.003 | 0.003 | | |

#### W1 (N = 796), kD = 10, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0041 | 0.0397 | 0.0046 (rot.) | 0.0048 (planewave i2) | rotation | 1.16 |
| Σ | | | 0.0041 | 0.0397 | 0.0046 | 0.0048 | best 0.0041 | |
| evaluate | | | 0.005 | 0.041 | 0.006 | 0.006 | | |

#### W1 (N = 7162), kD = 30, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 12 | 3096 | 0.0025 | 0.0166 | 0.0024 (rot.) | 0.0056 (planewave i2) | rotation | 2.23 |
| 3 | 8 | 32256 | 0.0090 | 0.0093 | 0.0095 (batched-fused) | 0.0115 (planewave i2) | rotation | 1.28 |
| Σ | | | 0.0115 | 0.0259 | 0.0119 | 0.0171 | best 0.0115 | |
| evaluate | | | 0.017 | 0.031 | 0.017 | 0.022 | | |

#### W1 (N = 7162), kD = 30, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 16 | 3096 | 0.0054 | 0.0521 | 0.0048 (rot.) | 0.0073 (planewave i2) | rotation | 1.36 |
| 3 | 15 | 32256 | 0.0426 | 0.1284 | 0.0500 (rot.) | 0.0192 (planewave i2) | planewave i2 | 0.45 |
| Σ | | | 0.0480 | 0.1805 | 0.0547 | 0.0265 | best 0.0246 | |
| evaluate | | | 0.056 | 0.189 | 0.063 | 0.034 | | |

#### W1 (N = 79578), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0269 | 0.0270 (rot.) | 0.0318 (rot.) | 0.0396 (planewave i2) | rotation | 1.47 |
| 3 | 18 | 53352 | 0.1176 | 0.3877 | 0.1165 (rot.) | 0.0490 (planewave i2) | planewave i2 | 0.42 |
| 4 | 11 | 385344 | 0.2426 | 0.2819 | 0.2362 (rot.) | 0.1312 (planewave i2) | planewave i2 | 0.54 |
| 5 | 7 | 244944 | 0.0560 | 0.0349 | 0.0356 (batched-fused) | 0.0968 (planewave i2) | batched-fused | 2.77 |
| Σ | | | 0.4431 | 0.7315 | 0.4202 | 0.3165 | best 0.2420 | |
| evaluate | | | 0.573 | 0.858 | 0.554 | 0.447 | | |

#### W1 (N = 79578), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0426 | 0.0500 (rot.) | 0.0438 (rot.) | 0.0459 (planewave i2) | rotation | 1.08 |
| 3 | 22 | 53352 | 0.2049 | 0.2097 (rot.) | 0.2458 (rot.) | 0.0592 (planewave i2) | planewave i2 | 0.29 |
| 4 | 15 | 385344 | 0.6298 | 1.1686 | 0.5395 (rot.) | 0.1767 (planewave i2) | planewave i2 | 0.28 |
| 5 | 15 | 244944 | 0.3983 | 0.8137 | 0.3451 (rot.) | 0.1989 (planewave i2) | planewave i2 | 0.50 |
| Σ | | | 1.2756 | 2.2419 | 1.1742 | 0.4807 | best 0.4774 | |
| evaluate | | | 1.592 | 2.552 | 1.516 | 0.791 | | |

#### W3 (N = 108849), kD = 30, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 12 | 3096 | 0.0026 | 0.0170 | 0.0023 (rot.) | 0.0058 (planewave i2) | rotation | 2.20 |
| 3 | 8 | 53352 | 0.0194 | 0.0134 | 0.0140 (batched-fused) | 0.0143 (planewave i2) | batched-fused | 1.07 |
| 4 | 7 | 584136 | 0.1417 | 0.0789 | 0.0797 (batched-fused) | 0.0955 (planewave i2) | batched-fused | 1.21 |
| Σ | | | 0.1637 | 0.1092 | 0.0961 | 0.1155 | best 0.0948 | |
| evaluate | | | 0.251 | 0.189 | 0.178 | 0.195 | | |

#### W3 (N = 108849), kD = 30, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 16 | 3096 | 0.0060 | 0.0057 (rot.) | 0.0055 (rot.) | 0.0078 (planewave i2) | rotation | 1.31 |
| 3 | 15 | 53352 | 0.0791 | 0.1925 | 0.0960 (rot.) | 0.0250 (planewave i2) | planewave i2 | 0.32 |
| 4 | 15 | 584136 | 0.8587 | 1.9338 | 1.0079 (rot.) | 0.1703 (planewave i2) | planewave i2 | 0.20 |
| Σ | | | 0.9438 | 2.1320 | 1.1094 | 0.2031 | best 0.2013 | |
| evaluate | | | 1.077 | 2.251 | 1.236 | 0.331 | | |

#### W3 (N = 4032), kD = 10, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0008 | 0.0027 | 0.0027 (batched-fused) | 0.0025 (planewave i2) | rotation | 3.30 |
| 3 | 7 | 5544 | 0.0013 | 0.0018 | 0.0017 (batched-fused) | 0.0035 (planewave i2) | rotation | 2.76 |
| Σ | | | 0.0020 | 0.0045 | 0.0044 | 0.0059 | best 0.0020 | |
| evaluate | | | 0.014 | 0.016 | 0.017 | 0.018 | | |

#### W3 (N = 4032), kD = 10, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0045 | 0.0405 | 0.0043 (rot.) | 0.0050 (planewave i2) | rotation | 1.10 |
| 3 | 15 | 5544 | 0.0079 | 0.0339 | 0.0084 (rot.) | 0.0073 (planewave i2) | planewave i2 | 0.93 |
| Σ | | | 0.0124 | 0.0745 | 0.0127 | 0.0123 | best 0.0119 | |
| evaluate | | | 0.044 | 0.108 | 0.045 | 0.044 | | |

#### W4 clusters (N = 100000), kD = 1, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0008 | 0.0032 | 0.0031 (batched-fused) | 0.0024 (planewave i2) | rotation | 2.99 |
| 3 | 7 | 19782 | 0.0050 | 0.0040 | 0.0040 (batched-fused) | 0.0054 (planewave i2) | batched-fused | 1.32 |
| 4 | 7 | 43392 | 0.0103 | 0.0080 | 0.0078 (batched-fused) | 0.0098 (planewave i2) | batched-fused | 1.22 |
| 5 | 7 | 106500 | 0.0300 | 0.0165 | 0.0168 (batched-fused) | 0.0203 (planewave i2) | batched-fused | 1.24 |
| 6 | 7 | 214818 | 0.0549 | 0.0324 | 0.0321 (batched-fused) | 0.0383 (planewave i2) | batched-fused | 1.18 |
| 7 | 7 | 289194 | 0.0749 | 0.0455 | 0.0483 (batched-fused) | 0.0510 (planewave i2) | batched-fused | 1.12 |
| 8 | 7 | 351936 | 0.0880 | 0.0544 | 0.0547 (batched-fused) | 0.0625 (planewave i2) | batched-fused | 1.15 |
| Σ | | | 0.2638 | 0.1641 | 0.1668 | 0.1898 | best 0.1617 | |
| evaluate | | | 0.529 | 0.412 | 0.415 | 0.436 | | |

#### W4 clusters (N = 100000), kD = 1, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0052 | 0.0053 (rot.) | 0.0045 (rot.) | 0.0046 (planewave i2) | planewave i2 | 0.87 |
| 3 | 15 | 19782 | 0.0369 | 0.0334 (rot.) | 0.0462 (rot.) | 0.0109 (planewave i2) | planewave i2 | 0.29 |
| 4 | 15 | 43392 | 0.0783 | 0.0815 (rot.) | 0.0738 (rot.) | 0.0189 (planewave i2) | planewave i2 | 0.24 |
| 5 | 15 | 106500 | 0.1883 | 0.1927 (rot.) | 0.1637 (rot.) | 0.0380 (planewave i2) | planewave i2 | 0.20 |
| 6 | 15 | 214818 | 0.3843 | 0.7141 | 0.3280 (rot.) | 0.0746 (planewave i2) | planewave i2 | 0.19 |
| 7 | 15 | 289194 | 0.4581 | 0.9598 | 0.4462 (rot.) | 0.0989 (planewave i2) | planewave i2 | 0.22 |
| 8 | 15 | 351936 | 0.5395 | 1.2793 | 0.5399 (rot.) | 0.1149 (planewave i2) | planewave i2 | 0.21 |
| Σ | | | 1.6907 | 3.2660 | 1.6023 | 0.3608 | best 0.3608 | |
| evaluate | | | 2.419 | 3.971 | 2.379 | 1.056 | | |

#### W4 clusters (N = 100000), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 21 | 3096 | 0.0118 | 0.0128 (rot.) | 0.0115 (rot.) | 0.0171 (planewave i2) | rotation | 1.45 |
| 3 | 13 | 19782 | 0.0247 | 0.0399 | 0.0319 (rot.) | 0.0150 (planewave i2) | planewave i2 | 0.61 |
| 4 | 8 | 43392 | 0.0143 | 0.0131 | 0.0127 (batched-fused) | 0.0157 (planewave i2) | batched-fused | 1.20 |
| 5 | 7 | 106500 | 0.0266 | 0.0171 | 0.0171 (batched-fused) | 0.0243 (planewave i2) | batched-fused | 1.42 |
| 6 | 7 | 214818 | 0.0760 | 0.0330 | 0.0325 (batched-fused) | 0.0429 (planewave i2) | batched-fused | 1.30 |
| 7 | 7 | 289194 | 0.0716 | 0.0473 | 0.0464 (batched-fused) | 0.0562 (planewave i2) | batched-fused | 1.19 |
| 8 | 7 | 351936 | 0.1084 | 0.0551 | 0.0544 (batched-fused) | 0.0662 (planewave i2) | batched-fused | 1.20 |
| Σ | | | 0.3335 | 0.2182 | 0.2066 | 0.2375 | best 0.1923 | |
| evaluate | | | 0.617 | 0.475 | 0.470 | 0.499 | | |

#### W4 clusters (N = 100000), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 25 | 3096 | 0.0197 | 0.0192 (rot.) | 0.0200 (rot.) | 0.0202 (planewave i2) | rotation | 1.02 |
| 3 | 16 | 19782 | 0.0427 | 0.0474 (rot.) | 0.0451 (rot.) | 0.0183 (planewave i2) | planewave i2 | 0.43 |
| 4 | 15 | 43392 | 0.0740 | 0.0806 (rot.) | 0.0732 (rot.) | 0.0263 (planewave i2) | planewave i2 | 0.36 |
| 5 | 15 | 106500 | 0.1645 | 0.1695 (rot.) | 0.1949 (rot.) | 0.0453 (planewave i2) | planewave i2 | 0.28 |
| 6 | 15 | 214818 | 0.3958 | 0.7187 | 0.3436 (rot.) | 0.0826 (planewave i2) | planewave i2 | 0.21 |
| 7 | 15 | 289194 | 0.4522 | 0.9849 | 0.5286 (rot.) | 0.1070 (planewave i2) | planewave i2 | 0.24 |
| 8 | 15 | 351936 | 0.5510 | 1.1975 | 0.5613 (rot.) | 0.1190 (planewave i2) | planewave i2 | 0.22 |
| Σ | | | 1.6998 | 3.2177 | 1.7668 | 0.4187 | best 0.4182 | |
| evaluate | | | 2.465 | 3.925 | 2.520 | 1.141 | | |

#### W4 cube (N = 100000), kD = 1, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0009 | 0.0030 | 0.0028 (batched-fused) | 0.0021 (planewave i2) | rotation | 2.49 |
| 3 | 7 | 53352 | 0.0126 | 0.0090 | 0.0087 (batched-fused) | 0.0100 (planewave i2) | batched-fused | 1.12 |
| 4 | 7 | 584136 | 0.1717 | 0.0938 | 0.0803 (batched-fused) | 0.0873 (planewave i2) | planewave i2 | 0.93 |
| Σ | | | 0.1852 | 0.1057 | 0.0918 | 0.0995 | best 0.0971 | |
| evaluate | | | 0.258 | 0.174 | 0.169 | 0.167 | | |

#### W4 cube (N = 100000), kD = 1, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0045 | 0.0418 | 0.0047 (rot.) | 0.0047 (planewave i2) | rotation | 1.03 |
| 3 | 15 | 53352 | 0.0948 | 0.1910 | 0.0918 (rot.) | 0.0188 (planewave i2) | planewave i2 | 0.20 |
| 4 | 15 | 584136 | 0.8739 | 1.7656 | 0.8686 (rot.) | 0.1595 (planewave i2) | planewave i2 | 0.18 |
| Σ | | | 0.9732 | 1.9984 | 0.9651 | 0.1829 | best 0.1828 | |
| evaluate | | | 1.085 | 2.104 | 1.081 | 0.291 | | |

#### W4 cube (N = 100000), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0324 | 0.0317 (rot.) | 0.0293 (rot.) | 0.0424 (planewave i2) | rotation | 1.31 |
| 3 | 18 | 53352 | 0.1295 | 0.4056 | 0.1746 (rot.) | 0.0524 (planewave i2) | planewave i2 | 0.40 |
| 4 | 11 | 584136 | 0.4039 | 0.5073 | 0.4664 (rot.) | 0.1925 (planewave i2) | planewave i2 | 0.48 |
| Σ | | | 0.5659 | 0.9446 | 0.6704 | 0.2873 | best 0.2773 | |
| evaluate | | | 0.678 | 1.047 | 0.776 | 0.393 | | |

#### W4 cube (N = 100000), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0532 | 0.0539 (rot.) | 0.0465 (rot.) | 0.0503 (planewave i2) | planewave i2 | 0.95 |
| 3 | 22 | 53352 | 0.2184 | 0.2265 (rot.) | 0.2184 (rot.) | 0.0624 (planewave i2) | planewave i2 | 0.29 |
| 4 | 15 | 584136 | 0.8680 | 1.9750 | 1.0333 (rot.) | 0.2445 (planewave i2) | planewave i2 | 0.28 |
| Σ | | | 1.1396 | 2.2553 | 1.2983 | 0.3573 | best 0.3573 | |
| evaluate | | | 1.282 | 2.388 | 1.435 | 0.492 | | |

#### W4 plummer (N = 100000), kD = 1, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0009 | 0.0029 | 0.0030 (batched-fused) | 0.0021 (planewave i2) | rotation | 2.49 |
| 3 | 7 | 53352 | 0.0124 | 0.0091 | 0.0087 (batched-fused) | 0.0096 (planewave i2) | batched-fused | 1.05 |
| 4 | 7 | 53352 | 0.0132 | 0.0093 | 0.0087 (batched-fused) | 0.0103 (planewave i2) | batched-fused | 1.10 |
| 5 | 7 | 109548 | 0.0297 | 0.0190 | 0.0183 (batched-fused) | 0.0185 (planewave i2) | planewave i2 | 0.97 |
| 6 | 7 | 205458 | 0.0528 | 0.0349 | 0.0313 (batched-fused) | 0.0321 (planewave i2) | planewave i2 | 0.92 |
| 7 | 7 | 333330 | 0.0993 | 0.0578 | 0.0510 (batched-fused) | 0.0483 (planewave i2) | planewave i2 | 0.84 |
| 8 | 7 | 35148 | 0.0087 | 0.0056 | 0.0067 (batched-fused) | 0.0085 (planewave i2) | batched-fused | 1.50 |
| Σ | | | 0.2170 | 0.1387 | 0.1278 | 0.1292 | best 0.1237 | |
| evaluate | | | 0.470 | 0.374 | 0.367 | 0.367 | | |

#### W4 plummer (N = 100000), kD = 1, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0042 | 0.0055 (rot.) | 0.0052 (rot.) | 0.0046 (planewave i2) | rotation | 1.09 |
| 3 | 15 | 53352 | 0.1116 | 0.1050 (rot.) | 0.1108 (rot.) | 0.0208 (planewave i2) | planewave i2 | 0.19 |
| 4 | 15 | 53352 | 0.0937 | 0.0847 (rot.) | 0.0801 (rot.) | 0.0195 (planewave i2) | planewave i2 | 0.21 |
| 5 | 15 | 109548 | 0.1651 | 0.1648 (rot.) | 0.1640 (rot.) | 0.0352 (planewave i2) | planewave i2 | 0.21 |
| 6 | 15 | 205458 | 0.3664 | 0.6594 | 0.3088 (rot.) | 0.0603 (planewave i2) | planewave i2 | 0.16 |
| 7 | 15 | 333330 | 0.5058 | 1.0663 | 0.5073 (rot.) | 0.0953 (planewave i2) | planewave i2 | 0.19 |
| 8 | 15 | 35148 | 0.0535 | 0.1298 | 0.0530 (rot.) | 0.0163 (planewave i2) | planewave i2 | 0.30 |
| Σ | | | 1.3005 | 2.2155 | 1.2293 | 0.2520 | best 0.2517 | |
| evaluate | | | 1.979 | 2.864 | 1.906 | 0.902 | | |

#### W4 plummer (N = 100000), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0365 | 0.0358 (rot.) | 0.0307 (rot.) | 0.0413 (planewave i2) | rotation | 1.13 |
| 3 | 18 | 53352 | 0.1551 | 0.4200 | 0.1359 (rot.) | 0.0506 (planewave i2) | planewave i2 | 0.33 |
| 4 | 11 | 380178 | 0.2604 | 0.3207 | 0.2619 (rot.) | 0.1310 (planewave i2) | planewave i2 | 0.50 |
| 5 | 7 | 109548 | 0.0279 | 0.0184 | 0.0193 (batched-fused) | 0.0237 (planewave i2) | batched-fused | 1.29 |
| 6 | 7 | 205458 | 0.0525 | 0.0319 | 0.0336 (batched-fused) | 0.0362 (planewave i2) | batched-fused | 1.13 |
| 7 | 7 | 333330 | 0.0837 | 0.0552 | 0.0507 (batched-fused) | 0.0540 (planewave i2) | planewave i2 | 0.98 |
| 8 | 7 | 35148 | 0.0088 | 0.0070 | 0.0074 (batched-fused) | 0.0086 (planewave i2) | batched-fused | 1.22 |
| Σ | | | 0.6249 | 0.8891 | 0.5395 | 0.3453 | best 0.3295 | |
| evaluate | | | 0.899 | 1.147 | 0.803 | 0.595 | | |

#### W4 plummer (N = 100000), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0484 | 0.0497 (rot.) | 0.0574 (rot.) | 0.0479 (planewave i2) | planewave i2 | 0.99 |
| 3 | 21 | 53352 | 0.2024 | 0.2167 (rot.) | 0.2020 (rot.) | 0.0622 (planewave i2) | planewave i2 | 0.31 |
| 4 | 15 | 380178 | 0.6805 | 0.6746 (rot.) | 0.5824 (rot.) | 0.1718 (planewave i2) | planewave i2 | 0.25 |
| 5 | 15 | 109548 | 0.1690 | 0.1711 (rot.) | 0.1718 (rot.) | 0.0460 (planewave i2) | planewave i2 | 0.27 |
| 6 | 15 | 205458 | 0.3115 | 0.6510 | 0.3163 (rot.) | 0.0652 (planewave i2) | planewave i2 | 0.21 |
| 7 | 15 | 333330 | 0.5955 | 1.0748 | 0.5117 (rot.) | 0.1028 (planewave i2) | planewave i2 | 0.17 |
| 8 | 15 | 35148 | 0.0554 | 0.1302 | 0.0534 (rot.) | 0.0169 (planewave i2) | planewave i2 | 0.31 |
| Σ | | | 2.0627 | 2.9681 | 1.8950 | 0.5130 | best 0.5130 | |
| evaluate | | | 2.745 | 3.613 | 2.575 | 1.152 | | |

#### W4 sphere (N = 100000), kD = 1, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0008 | 0.0029 | 0.0030 (batched-fused) | 0.0022 (planewave i2) | rotation | 2.81 |
| 3 | 7 | 53352 | 0.0137 | 0.0086 | 0.0088 (batched-fused) | 0.0092 (planewave i2) | batched-fused | 1.06 |
| 4 | 7 | 385344 | 0.1106 | 0.0590 | 0.0539 (batched-fused) | 0.0659 (planewave i2) | batched-fused | 1.12 |
| 5 | 7 | 325404 | 0.0787 | 0.0524 | 0.0464 (batched-fused) | 0.0950 (planewave i2) | batched-fused | 1.81 |
| Σ | | | 0.2038 | 0.1229 | 0.1121 | 0.1723 | best 0.1208 | |
| evaluate | | | 0.327 | 0.236 | 0.223 | 0.283 | | |

#### W4 sphere (N = 100000), kD = 1, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0046 | 0.0059 (rot.) | 0.0044 (rot.) | 0.0045 (planewave i2) | planewave i2 | 0.98 |
| 3 | 15 | 53352 | 0.1001 | 0.1914 | 0.0859 (rot.) | 0.0184 (planewave i2) | planewave i2 | 0.18 |
| 4 | 15 | 385344 | 0.5668 | 1.3088 | 0.5705 (rot.) | 0.1193 (planewave i2) | planewave i2 | 0.21 |
| 5 | 15 | 325404 | 0.4785 | 1.0958 | 0.5010 (rot.) | 0.1907 (planewave i2) | planewave i2 | 0.40 |
| Σ | | | 1.1500 | 2.6020 | 1.1617 | 0.3329 | best 0.3329 | |
| evaluate | | | 1.461 | 2.895 | 1.486 | 0.635 | | |

#### W4 sphere (N = 100000), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0288 | 0.0385 (rot.) | 0.0299 (rot.) | 0.0439 (planewave i2) | rotation | 1.52 |
| 3 | 18 | 53352 | 0.1292 | 0.4036 | 0.1315 (rot.) | 0.0521 (planewave i2) | planewave i2 | 0.40 |
| 4 | 11 | 385344 | 0.2611 | 0.3002 | 0.2609 (rot.) | 0.1418 (planewave i2) | planewave i2 | 0.54 |
| 5 | 7 | 325404 | 0.0810 | 0.0495 | 0.0498 (batched-fused) | 0.1222 (planewave i2) | batched-fused | 2.47 |
| Σ | | | 0.5002 | 0.7917 | 0.4721 | 0.3600 | best 0.2722 | |
| evaluate | | | 0.654 | 0.931 | 0.619 | 0.503 | | |

#### W4 sphere (N = 100000), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0466 | 0.0545 (rot.) | 0.0476 (rot.) | 0.0491 (planewave i2) | rotation | 1.05 |
| 3 | 22 | 53352 | 0.2279 | 0.2303 (rot.) | 0.2600 (rot.) | 0.0612 (planewave i2) | planewave i2 | 0.27 |
| 4 | 15 | 385344 | 0.6087 | 1.2294 | 0.5747 (rot.) | 0.1959 (planewave i2) | planewave i2 | 0.32 |
| 5 | 15 | 325404 | 0.4892 | 1.2274 | 0.4866 (rot.) | 0.2395 (planewave i2) | planewave i2 | 0.49 |
| Σ | | | 1.3724 | 2.7417 | 1.3690 | 0.5458 | best 0.5433 | |
| evaluate | | | 1.703 | 3.058 | 1.697 | 0.882 | | |

### locust

#### W1 (N = 796), kD = 10, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0002 | 0.0038 | 0.0038 (batched-fused) | 0.0040 (planewave i2) | rotation | 23.86 |
| Σ | | | 0.0002 | 0.0038 | 0.0038 | 0.0040 | best 0.0002 | |
| evaluate | | | 0.001 | 0.005 | 0.005 | 0.006 | | |

#### W1 (N = 796), kD = 10, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0007 | 0.1229 | 0.0007 (rot.) | 0.0073 (planewave i2) | rotation | 10.55 |
| Σ | | | 0.0007 | 0.1229 | 0.0007 | 0.0073 | best 0.0007 | |
| evaluate | | | 0.002 | 0.124 | 0.002 | 0.010 | | |

#### W1 (N = 7162), kD = 30, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 12 | 3096 | 0.0004 | 0.0482 | 0.0004 (rot.) | 0.0084 (planewave i2) | rotation | 19.93 |
| 3 | 8 | 32256 | 0.0013 | 0.0086 | 0.0085 (batched-fused) | 0.0067 (planewave i2) | rotation | 5.24 |
| Σ | | | 0.0017 | 0.0568 | 0.0089 | 0.0150 | best 0.0017 | |
| evaluate | | | 0.005 | 0.060 | 0.012 | 0.018 | | |

#### W1 (N = 7162), kD = 30, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 16 | 3096 | 0.0008 | 0.1987 | 0.0008 (rot.) | 0.0107 (planewave i2) | rotation | 12.86 |
| 3 | 15 | 32256 | 0.0056 | 0.1183 | 0.0056 (rot.) | 0.0109 (planewave i2) | rotation | 1.93 |
| Σ | | | 0.0065 | 0.3170 | 0.0064 | 0.0215 | best 0.0065 | |
| evaluate | | | 0.010 | 0.322 | 0.010 | 0.025 | | |

#### W1 (N = 79578), kD = 100, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0043 | 0.0043 (rot.) | 0.0042 (rot.) | 0.0526 (planewave i2) | rotation | 12.32 |
| 3 | 18 | 53352 | 0.0154 | 0.4556 | 0.0156 (rot.) | 0.0265 (planewave i2) | rotation | 1.72 |
| 4 | 11 | 385344 | 0.0274 | 0.0434 | 0.0272 (rot.) | 0.0223 (planewave i2) | planewave i2 | 0.82 |
| 5 | 7 | 244944 | 0.0064 | 0.0071 | 0.0071 (batched-fused) | 0.0152 (planewave i2) | rotation | 2.38 |
| Σ | | | 0.0534 | 0.5103 | 0.0541 | 0.1166 | best 0.0484 | |
| evaluate | | | 0.083 | 0.541 | 0.084 | 0.147 | | |

#### W1 (N = 79578), kD = 100, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0067 | 0.0067 (rot.) | 0.0067 (rot.) | 0.0612 (planewave i2) | rotation | 9.12 |
| 3 | 22 | 53352 | 0.0269 | 0.0273 (rot.) | 0.0270 (rot.) | 0.0318 (planewave i2) | rotation | 1.18 |
| 4 | 15 | 385344 | 0.0603 | 0.1688 | 0.0599 (rot.) | 0.0277 (planewave i2) | planewave i2 | 0.46 |
| 5 | 15 | 244944 | 0.0380 | 0.1613 | 0.0381 (rot.) | 0.0263 (planewave i2) | planewave i2 | 0.69 |
| Σ | | | 0.1319 | 0.3641 | 0.1318 | 0.1470 | best 0.0877 | |
| evaluate | | | 0.198 | 0.429 | 0.197 | 0.212 | | |

#### W1 (N = 1989437), kD = 500, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 123 | 3096 | 0.7502 | 0.7578 (rot.) | 0.7426 (rot.) | 0.7638 (rot.) | rotation | nan |
| 3 | 66 | 53352 | 0.7408 | 0.7319 (rot.) | 0.7606 (rot.) | 0.7645 (rot.) | rotation | nan |
| 4 | 36 | 511200 | 0.9401 | 0.9912 (rot.) | 0.9378 (rot.) | 0.2832 (planewave i2) | planewave i2 | 0.30 |
| 5 | 21 | 2935296 | 1.2402 | 1.1844 (rot.) | 1.1536 (rot.) | 0.3713 (planewave i2) | planewave i2 | 0.30 |
| 6 | 13 | 10439280 | 1.1170 | 1.7642 | 1.1118 (rot.) | 0.5689 (planewave i2) | planewave i2 | 0.51 |
| 7 | 8 | 14492220 | 0.4855 | 0.3768 | 0.3840 (batched-fused) | 0.6232 (planewave i2) | batched-fused | 1.65 |
| Σ | | | 5.2738 | 5.8063 | 5.0904 | 3.3749 | best 3.0913 | |
| evaluate | | | 5.861 | 6.379 | 5.710 | 3.930 | | |

#### W1 (N = 1989437), kD = 500, ε = 1e-3, 72 × 1

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 123 | 3096 | 3.0783 | 2.9558 (rot.) | 3.3205 (rot.) | 3.3981 (rot.) | rotation | nan |
| 3 | 66 | 53352 | 3.6095 | 3.4004 (rot.) | 3.6386 (rot.) | 4.4012 (rot.) | rotation | nan |
| 4 | 36 | 511200 | 2.3707 | 2.5182 (rot.) | 2.4688 (rot.) | 2.1151 (planewave i2) | planewave i2 | 0.89 |
| 5 | 21 | 2935296 | 2.1912 | 2.2690 (rot.) | 2.1340 (rot.) | 1.5175 (planewave i2) | planewave i2 | 0.69 |
| 6 | 13 | 10439280 | 1.8425 | 3.0170 | 1.7225 (rot.) | 1.4145 (planewave i2) | planewave i2 | 0.77 |
| 7 | 8 | 14492220 | 0.7029 | 0.5942 | 0.6159 (batched-fused) | 1.1376 (planewave i2) | batched-fused | 1.91 |
| Σ | | | 13.7952 | 14.7544 | 13.9002 | 13.9841 | best 12.3292 | |
| evaluate | | | 13.036 | 14.200 | 13.209 | 12.926 | | |

#### W1 (N = 1989437), kD = 500, ε = 1e-3, 8 × 9

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 123 | 3096 | 2.5580 | 2.4833 (rot.) | 2.7210 (rot.) | 2.7006 (rot.) | rotation | nan |
| 3 | 66 | 53352 | 1.2270 | 1.0978 (rot.) | 1.5402 (rot.) | 1.2789 (rot.) | rotation | nan |
| 4 | 36 | 511200 | 1.3562 | 1.1607 (rot.) | 1.4378 (rot.) | 0.4552 (planewave i2) | planewave i2 | 0.34 |
| 5 | 21 | 2935296 | 1.2093 | 1.3796 (rot.) | 1.6731 (rot.) | 0.4418 (planewave i2) | planewave i2 | 0.37 |
| 6 | 13 | 10439280 | 1.1744 | 2.0879 | 1.7695 (rot.) | 0.5898 (planewave i2) | planewave i2 | 0.50 |
| 7 | 8 | 14492220 | 0.5072 | 0.4348 | 0.6956 (batched-fused) | 0.6521 (planewave i2) | batched-fused | 1.50 |
| Σ | | | 8.0323 | 8.6440 | 9.8371 | 6.1184 | best 5.7066 | |
| evaluate | | | 8.127 | 8.734 | 9.064 | 6.247 | | |

#### W1 (N = 1989437), kD = 500, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 131 | 3096 | 0.9349 | 0.9290 (rot.) | 0.9509 (rot.) | 0.9248 (rot.) | rotation | nan |
| 3 | 72 | 53352 | 0.9772 | 0.9913 (rot.) | 0.9665 (rot.) | 0.9806 (rot.) | rotation | nan |
| 4 | 41 | 511200 | 1.3946 | 1.4099 (rot.) | 1.4073 (rot.) | 0.3113 (planewave i2) | planewave i2 | 0.22 |
| 5 | 25 | 2935296 | 1.9021 | 1.9167 (rot.) | 1.9197 (rot.) | 0.4623 (planewave i2) | planewave i2 | 0.24 |
| 6 | 16 | 10439280 | 1.9678 | 5.4702 | 1.9835 (rot.) | 0.6577 (planewave i2) | planewave i2 | 0.33 |
| 7 | 15 | 14492220 | 2.3043 | 4.8404 | 2.3043 (rot.) | 0.9203 (planewave i2) | planewave i2 | 0.40 |
| Σ | | | 9.4809 | 15.5576 | 9.5321 | 4.2569 | best 4.2637 | |
| evaluate | | | 10.257 | 16.329 | 10.304 | 5.023 | | |

#### W1 (N = 1989437), kD = 500, ε = 1e-6, 72 × 1

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 131 | 3096 | 3.5956 | 3.6679 (rot.) | 3.6366 (rot.) | 3.9549 (rot.) | rotation | nan |
| 3 | 72 | 53352 | 5.4151 | 4.8418 (rot.) | 4.5788 (rot.) | 5.7314 (rot.) | rotation | nan |
| 4 | 41 | 511200 | 3.7964 | 4.2675 (rot.) | 3.7683 (rot.) | 2.2224 (planewave i2) | planewave i2 | 0.59 |
| 5 | 25 | 2935296 | 3.6045 | 3.7557 (rot.) | 4.5644 (rot.) | 1.3488 (planewave i2) | planewave i2 | 0.37 |
| 6 | 16 | 10439280 | 3.3823 | 9.5050 | 2.7194 (rot.) | 1.4041 (planewave i2) | planewave i2 | 0.42 |
| 7 | 15 | 14492220 | 3.3042 | 6.8186 | 2.7463 (rot.) | 1.4927 (planewave i2) | planewave i2 | 0.45 |
| Σ | | | 23.0983 | 32.8565 | 22.0139 | 16.1544 | best 15.4787 | |
| evaluate | | | 20.746 | 29.857 | 20.653 | 16.044 | | |

#### W1 (N = 1989437), kD = 500, ε = 1e-6, 8 × 9

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 131 | 3096 | 3.1877 | 3.2026 (rot.) | 3.2372 (rot.) | 3.2890 (rot.) | rotation | nan |
| 3 | 72 | 53352 | 1.7152 | 1.6042 (rot.) | 1.6473 (rot.) | 1.7522 (rot.) | rotation | nan |
| 4 | 41 | 511200 | 1.6159 | 1.7276 (rot.) | 1.6064 (rot.) | 0.5089 (planewave i2) | planewave i2 | 0.31 |
| 5 | 25 | 2935296 | 2.0576 | 2.1178 (rot.) | 2.0547 (rot.) | 0.5078 (planewave i2) | planewave i2 | 0.25 |
| 6 | 16 | 10439280 | 2.1557 | 6.4149 | 2.0879 (rot.) | 0.6838 (planewave i2) | planewave i2 | 0.32 |
| 7 | 15 | 14492220 | 2.6014 | 6.3621 | 2.3912 (rot.) | 1.0217 (planewave i2) | planewave i2 | 0.39 |
| Σ | | | 13.3335 | 21.4293 | 13.0248 | 7.7634 | best 7.6251 | |
| evaluate | | | 13.514 | 21.135 | 13.156 | 7.835 | | |

#### W3 (N = 108849), kD = 30, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 12 | 3096 | 0.0004 | 0.0531 | 0.0004 (rot.) | 0.0084 (planewave i2) | rotation | 21.04 |
| 3 | 8 | 53352 | 0.0021 | 0.0116 | 0.0115 (batched-fused) | 0.0083 (planewave i2) | rotation | 4.05 |
| 4 | 7 | 584136 | 0.0146 | 0.0129 | 0.0123 (batched-fused) | 0.0158 (planewave i2) | batched-fused | 1.23 |
| Σ | | | 0.0171 | 0.0776 | 0.0242 | 0.0325 | best 0.0153 | |
| evaluate | | | 0.039 | 0.100 | 0.047 | 0.055 | | |

#### W3 (N = 108849), kD = 30, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 16 | 3096 | 0.0008 | 0.0008 (rot.) | 0.0008 (rot.) | 0.0109 (planewave i2) | rotation | 13.37 |
| 3 | 15 | 53352 | 0.0094 | 0.1511 | 0.0094 (rot.) | 0.0133 (planewave i2) | rotation | 1.41 |
| 4 | 15 | 584136 | 0.0911 | 0.2410 | 0.0908 (rot.) | 0.0245 (planewave i2) | planewave i2 | 0.27 |
| Σ | | | 0.1014 | 0.3929 | 0.1010 | 0.0488 | best 0.0348 | |
| evaluate | | | 0.131 | 0.424 | 0.131 | 0.078 | | |

#### W3 (N = 4031442), kD = 100, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0043 | 0.0043 (rot.) | 0.0043 (rot.) | 0.0531 (planewave i2) | rotation | 12.38 |
| 3 | 18 | 53352 | 0.0157 | 0.4502 | 0.0155 (rot.) | 0.0264 (planewave i2) | rotation | 1.68 |
| 4 | 11 | 584136 | 0.0415 | 0.0622 | 0.0415 (rot.) | 0.0286 (planewave i2) | planewave i2 | 0.69 |
| 5 | 7 | 5398920 | 0.1377 | 0.0864 | 0.0864 (batched-fused) | 0.1458 (planewave i2) | batched-fused | 1.69 |
| 6 | 7 | 46298376 | 1.1343 | 0.7408 | 0.7871 (batched-fused) | 1.0649 (planewave i2) | batched-fused | 1.44 |
| Σ | | | 1.3335 | 1.3439 | 0.9347 | 1.3188 | best 0.8758 | |
| evaluate | | | 2.023 | 2.024 | 1.615 | 1.999 | | |

#### W3 (N = 4031442), kD = 100, ε = 1e-3, 72 × 1

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0129 | 0.0133 (rot.) | 0.0127 (rot.) | 0.1267 (planewave i2) | rotation | 9.82 |
| 3 | 18 | 53352 | 0.0298 | 0.4646 | 0.0298 (rot.) | 0.0900 (planewave i2) | rotation | 3.02 |
| 4 | 11 | 584136 | 0.0561 | 0.1016 | 0.0572 (rot.) | 0.1177 (planewave i2) | rotation | 2.10 |
| 5 | 7 | 5398920 | 0.1593 | 0.1080 | 0.1080 (batched-fused) | 0.3401 (planewave i2) | batched-fused | 3.15 |
| 6 | 7 | 46298376 | 1.2501 | 0.8334 | 0.8334 (batched-fused) | 1.9908 (planewave i2) | batched-fused | 2.39 |
| Σ | | | 1.5081 | 1.5209 | 1.0411 | 2.6654 | best 1.0402 | |
| evaluate | | | 1.980 | 1.946 | 1.508 | 3.081 | | |

#### W3 (N = 4031442), kD = 100, ε = 1e-3, 8 × 9

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0157 | 0.0164 (rot.) | 0.0156 (rot.) | 0.0151 (planewave i2) | planewave i2 | 0.96 |
| 3 | 18 | 53352 | 0.0152 | 0.5323 | 0.0154 (rot.) | 0.0305 (planewave i2) | rotation | 2.00 |
| 4 | 11 | 584136 | 0.0429 | 0.0681 | 0.0429 (rot.) | 0.0368 (planewave i2) | planewave i2 | 0.86 |
| 5 | 7 | 5398920 | 0.1350 | 0.0918 | 0.0918 (batched-fused) | 0.1701 (planewave i2) | batched-fused | 1.85 |
| 6 | 7 | 46298376 | 1.1575 | 0.7639 | 0.8334 (batched-fused) | 1.2038 (planewave i2) | batched-fused | 1.58 |
| Σ | | | 1.3663 | 1.4725 | 0.9992 | 1.4562 | best 0.9228 | |
| evaluate | | | 1.821 | 1.911 | 1.481 | 1.902 | | |

#### W3 (N = 4031442), kD = 100, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0069 | 0.0069 (rot.) | 0.0069 (rot.) | 0.0621 (planewave i2) | rotation | 9.01 |
| 3 | 22 | 53352 | 0.0269 | 0.0275 (rot.) | 0.0265 (rot.) | 0.0317 (planewave i2) | rotation | 1.18 |
| 4 | 15 | 584136 | 0.0943 | 0.2179 | 0.0917 (rot.) | 0.0403 (planewave i2) | planewave i2 | 0.43 |
| 5 | 15 | 5398920 | 0.8854 | 1.6008 | 0.8476 (rot.) | 0.2025 (planewave i2) | planewave i2 | 0.23 |
| 6 | 15 | 46298376 | 7.3614 | 12.8478 | 7.3151 (rot.) | 1.4352 (planewave i2) | planewave i2 | 0.19 |
| Σ | | | 8.3750 | 14.7009 | 8.2879 | 1.7718 | best 1.7118 | |
| evaluate | | | 9.344 | 15.669 | 9.270 | 2.778 | | |

#### W3 (N = 4031442), kD = 100, ε = 1e-6, 72 × 1

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0255 | 0.0253 (rot.) | 0.0246 (rot.) | 0.1512 (planewave i2) | rotation | 5.94 |
| 3 | 22 | 53352 | 0.0551 | 0.0564 (rot.) | 0.0550 (rot.) | 0.1064 (planewave i2) | rotation | 1.93 |
| 4 | 15 | 584136 | 0.1279 | 0.3829 | 0.1276 (rot.) | 0.1367 (planewave i2) | rotation | 1.07 |
| 5 | 15 | 5398920 | 1.0366 | 2.2810 | 1.0204 (rot.) | 0.4346 (planewave i2) | planewave i2 | 0.42 |
| 6 | 15 | 46298376 | 8.2411 | 16.5980 | 8.1254 (rot.) | 2.8011 (planewave i2) | planewave i2 | 0.34 |
| Σ | | | 9.4862 | 19.3436 | 9.3530 | 3.6300 | best 3.4441 | |
| evaluate | | | 10.231 | 19.894 | 10.105 | 4.326 | | |

#### W3 (N = 4031442), kD = 100, ε = 1e-6, 8 × 9

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0253 | 0.0250 (rot.) | 0.0253 (rot.) | 0.0175 (planewave i2) | planewave i2 | 0.69 |
| 3 | 22 | 53352 | 0.0267 | 0.0287 (rot.) | 0.0275 (rot.) | 0.0363 (planewave i2) | rotation | 1.36 |
| 4 | 15 | 584136 | 0.0932 | 0.2652 | 0.0961 (rot.) | 0.0444 (planewave i2) | planewave i2 | 0.48 |
| 5 | 15 | 5398920 | 1.0312 | 1.8464 | 0.8719 (rot.) | 0.2295 (planewave i2) | planewave i2 | 0.22 |
| 6 | 15 | 46298376 | 7.5698 | 13.7275 | 7.5235 (rot.) | 1.6436 (planewave i2) | planewave i2 | 0.22 |
| Σ | | | 8.7461 | 15.8927 | 8.5443 | 1.9712 | best 1.9616 | |
| evaluate | | | 9.315 | 16.573 | 9.290 | 2.697 | | |

#### W3 (N = 4032), kD = 10, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0002 | 0.0038 | 0.0037 (batched-fused) | 0.0039 (planewave i2) | rotation | 23.36 |
| 3 | 7 | 5544 | 0.0002 | 0.0027 | 0.0028 (batched-fused) | 0.0037 (planewave i2) | rotation | 15.30 |
| Σ | | | 0.0004 | 0.0065 | 0.0065 | 0.0076 | best 0.0004 | |
| evaluate | | | 0.007 | 0.014 | 0.014 | 0.015 | | |

#### W3 (N = 4032), kD = 10, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0007 | 0.1235 | 0.0007 (rot.) | 0.0071 (planewave i2) | rotation | 10.34 |
| 3 | 15 | 5544 | 0.0013 | 0.0783 | 0.0012 (rot.) | 0.0069 (planewave i2) | rotation | 5.53 |
| Σ | | | 0.0019 | 0.2019 | 0.0020 | 0.0140 | best 0.0019 | |
| evaluate | | | 0.019 | 0.219 | 0.019 | 0.031 | | |

#### W4 clusters (N = 100000), kD = 1, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0002 | 0.0043 | 0.0043 (batched-fused) | 0.0035 (planewave i2) | rotation | 23.12 |
| 3 | 7 | 19782 | 0.0007 | 0.0052 | 0.0052 (batched-fused) | 0.0036 (planewave i2) | rotation | 5.32 |
| 4 | 7 | 43392 | 0.0013 | 0.0068 | 0.0068 (batched-fused) | 0.0039 (planewave i2) | rotation | 2.98 |
| 5 | 7 | 106500 | 0.0030 | 0.0089 | 0.0088 (batched-fused) | 0.0051 (planewave i2) | rotation | 1.70 |
| 6 | 7 | 214818 | 0.0056 | 0.0099 | 0.0099 (batched-fused) | 0.0085 (planewave i2) | rotation | 1.52 |
| 7 | 7 | 289194 | 0.0075 | 0.0113 | 0.0117 (batched-fused) | 0.0078 (planewave i2) | rotation | 1.04 |
| 8 | 7 | 351936 | 0.0088 | 0.0099 | 0.0099 (batched-fused) | 0.0092 (planewave i2) | rotation | 1.04 |
| Σ | | | 0.0270 | 0.0563 | 0.0566 | 0.0415 | best 0.0270 | |
| evaluate | | | 0.084 | 0.112 | 0.113 | 0.098 | | |

#### W4 clusters (N = 100000), kD = 1, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0007 | 0.0007 (rot.) | 0.0007 (rot.) | 0.0065 (planewave i2) | rotation | 9.63 |
| 3 | 15 | 19782 | 0.0039 | 0.0039 (rot.) | 0.0037 (rot.) | 0.0064 (planewave i2) | rotation | 1.66 |
| 4 | 15 | 43392 | 0.0076 | 0.0077 (rot.) | 0.0076 (rot.) | 0.0069 (planewave i2) | planewave i2 | 0.90 |
| 5 | 15 | 106500 | 0.0175 | 0.0177 (rot.) | 0.0175 (rot.) | 0.0087 (planewave i2) | planewave i2 | 0.50 |
| 6 | 15 | 214818 | 0.0342 | 0.1903 | 0.0344 (rot.) | 0.0147 (planewave i2) | planewave i2 | 0.43 |
| 7 | 15 | 289194 | 0.0467 | 0.2218 | 0.0463 (rot.) | 0.0136 (planewave i2) | planewave i2 | 0.29 |
| 8 | 15 | 351936 | 0.0556 | 0.1951 | 0.0553 (rot.) | 0.0157 (planewave i2) | planewave i2 | 0.28 |
| Σ | | | 0.1662 | 0.6373 | 0.1654 | 0.0725 | best 0.0641 | |
| evaluate | | | 0.308 | 0.780 | 0.306 | 0.214 | | |

#### W4 clusters (N = 100000), kD = 100, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 21 | 3096 | 0.0016 | 0.0016 (rot.) | 0.0016 (rot.) | 0.0221 (planewave i2) | rotation | 13.96 |
| 3 | 13 | 19782 | 0.0026 | 0.0625 | 0.0027 (rot.) | 0.0091 (planewave i2) | rotation | 3.56 |
| 4 | 8 | 43392 | 0.0017 | 0.0113 | 0.0112 (batched-fused) | 0.0060 (planewave i2) | rotation | 3.46 |
| 5 | 7 | 106500 | 0.0028 | 0.0088 | 0.0088 (batched-fused) | 0.0058 (planewave i2) | rotation | 2.06 |
| 6 | 7 | 214818 | 0.0055 | 0.0099 | 0.0099 (batched-fused) | 0.0092 (planewave i2) | rotation | 1.69 |
| 7 | 7 | 289194 | 0.0074 | 0.0113 | 0.0111 (batched-fused) | 0.0087 (planewave i2) | rotation | 1.18 |
| 8 | 7 | 351936 | 0.0090 | 0.0099 | 0.0099 (batched-fused) | 0.0095 (planewave i2) | rotation | 1.06 |
| Σ | | | 0.0305 | 0.1153 | 0.0552 | 0.0705 | best 0.0305 | |
| evaluate | | | 0.087 | 0.173 | 0.114 | 0.128 | | |

#### W4 clusters (N = 100000), kD = 100, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 25 | 3096 | 0.0026 | 0.0026 (rot.) | 0.0026 (rot.) | 0.0263 (planewave i2) | rotation | 10.19 |
| 3 | 16 | 19782 | 0.0046 | 0.0044 (rot.) | 0.0045 (rot.) | 0.0110 (planewave i2) | rotation | 2.40 |
| 4 | 15 | 43392 | 0.0078 | 0.0076 (rot.) | 0.0077 (rot.) | 0.0096 (planewave i2) | rotation | 1.22 |
| 5 | 15 | 106500 | 0.0179 | 0.0178 (rot.) | 0.0177 (rot.) | 0.0100 (planewave i2) | planewave i2 | 0.56 |
| 6 | 15 | 214818 | 0.0343 | 0.1943 | 0.0340 (rot.) | 0.0161 (planewave i2) | planewave i2 | 0.47 |
| 7 | 15 | 289194 | 0.0458 | 0.2160 | 0.0458 (rot.) | 0.0145 (planewave i2) | planewave i2 | 0.32 |
| 8 | 15 | 351936 | 0.0554 | 0.1936 | 0.0554 (rot.) | 0.0165 (planewave i2) | planewave i2 | 0.30 |
| Σ | | | 0.1685 | 0.6362 | 0.1678 | 0.1040 | best 0.0721 | |
| evaluate | | | 0.309 | 0.778 | 0.308 | 0.245 | | |

#### W4 cube (N = 100000), kD = 1, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0001 | 0.0041 | 0.0040 (batched-fused) | 0.0035 (planewave i2) | rotation | 23.76 |
| 3 | 7 | 53352 | 0.0016 | 0.0070 | 0.0070 (batched-fused) | 0.0058 (planewave i2) | rotation | 3.60 |
| 4 | 7 | 584136 | 0.0146 | 0.0129 | 0.0123 (batched-fused) | 0.0143 (planewave i2) | batched-fused | 1.11 |
| Σ | | | 0.0164 | 0.0240 | 0.0232 | 0.0236 | best 0.0146 | |
| evaluate | | | 0.035 | 0.044 | 0.043 | 0.044 | | |

#### W4 cube (N = 100000), kD = 1, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0007 | 0.1254 | 0.0007 (rot.) | 0.0064 (planewave i2) | rotation | 9.39 |
| 3 | 15 | 53352 | 0.0096 | 0.1511 | 0.0097 (rot.) | 0.0101 (planewave i2) | rotation | 1.06 |
| 4 | 15 | 584136 | 0.0911 | 0.2357 | 0.0911 (rot.) | 0.0222 (planewave i2) | planewave i2 | 0.24 |
| Σ | | | 0.1014 | 0.5122 | 0.1015 | 0.0387 | best 0.0325 | |
| evaluate | | | 0.127 | 0.540 | 0.127 | 0.065 | | |

#### W4 cube (N = 100000), kD = 100, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0042 | 0.0043 (rot.) | 0.0042 (rot.) | 0.0529 (planewave i2) | rotation | 12.54 |
| 3 | 18 | 53352 | 0.0156 | 0.4532 | 0.0155 (rot.) | 0.0268 (planewave i2) | rotation | 1.72 |
| 4 | 11 | 584136 | 0.0412 | 0.0613 | 0.0409 (rot.) | 0.0292 (planewave i2) | planewave i2 | 0.71 |
| Σ | | | 0.0610 | 0.5188 | 0.0606 | 0.1089 | best 0.0490 | |
| evaluate | | | 0.085 | 0.544 | 0.085 | 0.135 | | |

#### W4 cube (N = 100000), kD = 100, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0066 | 0.0067 (rot.) | 0.0066 (rot.) | 0.0619 (planewave i2) | rotation | 9.31 |
| 3 | 22 | 53352 | 0.0266 | 0.0269 (rot.) | 0.0269 (rot.) | 0.0319 (planewave i2) | rotation | 1.20 |
| 4 | 15 | 584136 | 0.0917 | 0.2383 | 0.0914 (rot.) | 0.0353 (planewave i2) | planewave i2 | 0.39 |
| Σ | | | 0.1250 | 0.2720 | 0.1250 | 0.1291 | best 0.0686 | |
| evaluate | | | 0.154 | 0.302 | 0.156 | 0.159 | | |

#### W4 plummer (N = 100000), kD = 1, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0001 | 0.0041 | 0.0042 (batched-fused) | 0.0034 (planewave i2) | rotation | 23.20 |
| 3 | 7 | 53352 | 0.0016 | 0.0070 | 0.0071 (batched-fused) | 0.0055 (planewave i2) | rotation | 3.53 |
| 4 | 7 | 53352 | 0.0015 | 0.0071 | 0.0071 (batched-fused) | 0.0057 (planewave i2) | rotation | 3.66 |
| 5 | 7 | 109548 | 0.0030 | 0.0102 | 0.0103 (batched-fused) | 0.0064 (planewave i2) | rotation | 2.17 |
| 6 | 7 | 205458 | 0.0053 | 0.0117 | 0.0117 (batched-fused) | 0.0076 (planewave i2) | rotation | 1.42 |
| 7 | 7 | 333330 | 0.0083 | 0.0113 | 0.0115 (batched-fused) | 0.0073 (planewave i2) | planewave i2 | 0.88 |
| 8 | 7 | 35148 | 0.0011 | 0.0075 | 0.0076 (batched-fused) | 0.0053 (planewave i2) | rotation | 4.89 |
| Σ | | | 0.0210 | 0.0591 | 0.0594 | 0.0413 | best 0.0200 | |
| evaluate | | | 0.083 | 0.122 | 0.123 | 0.104 | | |

#### W4 plummer (N = 100000), kD = 1, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0007 | 0.0007 (rot.) | 0.0007 (rot.) | 0.0063 (planewave i2) | rotation | 9.32 |
| 3 | 15 | 53352 | 0.0094 | 0.0096 (rot.) | 0.0094 (rot.) | 0.0099 (planewave i2) | rotation | 1.05 |
| 4 | 15 | 53352 | 0.0096 | 0.0094 (rot.) | 0.0096 (rot.) | 0.0099 (planewave i2) | rotation | 1.04 |
| 5 | 15 | 109548 | 0.0180 | 0.0181 (rot.) | 0.0182 (rot.) | 0.0111 (planewave i2) | planewave i2 | 0.62 |
| 6 | 15 | 205458 | 0.0328 | 0.2235 | 0.0328 (rot.) | 0.0121 (planewave i2) | planewave i2 | 0.37 |
| 7 | 15 | 333330 | 0.0523 | 0.2205 | 0.0527 (rot.) | 0.0125 (planewave i2) | planewave i2 | 0.24 |
| 8 | 15 | 35148 | 0.0065 | 0.1661 | 0.0065 (rot.) | 0.0095 (planewave i2) | rotation | 1.46 |
| Σ | | | 0.1293 | 0.6479 | 0.1299 | 0.0712 | best 0.0619 | |
| evaluate | | | 0.291 | 0.805 | 0.292 | 0.232 | | |

#### W4 plummer (N = 100000), kD = 100, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0042 | 0.0043 (rot.) | 0.0044 (rot.) | 0.0521 (planewave i2) | rotation | 12.37 |
| 3 | 18 | 53352 | 0.0155 | 0.4606 | 0.0150 (rot.) | 0.0260 (planewave i2) | rotation | 1.68 |
| 4 | 11 | 380178 | 0.0272 | 0.0601 | 0.0270 (rot.) | 0.0201 (planewave i2) | planewave i2 | 0.74 |
| 5 | 7 | 109548 | 0.0030 | 0.0103 | 0.0102 (batched-fused) | 0.0081 (planewave i2) | rotation | 2.72 |
| 6 | 7 | 205458 | 0.0053 | 0.0118 | 0.0117 (batched-fused) | 0.0076 (planewave i2) | rotation | 1.42 |
| 7 | 7 | 333330 | 0.0083 | 0.0113 | 0.0113 (batched-fused) | 0.0080 (planewave i2) | planewave i2 | 0.96 |
| 8 | 7 | 35148 | 0.0011 | 0.0076 | 0.0076 (batched-fused) | 0.0054 (planewave i2) | rotation | 4.92 |
| Σ | | | 0.0646 | 0.5660 | 0.0873 | 0.1273 | best 0.0572 | |
| evaluate | | | 0.127 | 0.629 | 0.150 | 0.190 | | |

#### W4 plummer (N = 100000), kD = 100, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0066 | 0.0066 (rot.) | 0.0066 (rot.) | 0.0604 (planewave i2) | rotation | 9.12 |
| 3 | 21 | 53352 | 0.0231 | 0.0231 (rot.) | 0.0236 (rot.) | 0.0299 (planewave i2) | rotation | 1.29 |
| 4 | 15 | 380178 | 0.0597 | 0.0593 (rot.) | 0.0595 (rot.) | 0.0247 (planewave i2) | planewave i2 | 0.41 |
| 5 | 15 | 109548 | 0.0180 | 0.0187 (rot.) | 0.0183 (rot.) | 0.0140 (planewave i2) | planewave i2 | 0.78 |
| 6 | 15 | 205458 | 0.0329 | 0.2203 | 0.0331 (rot.) | 0.0131 (planewave i2) | planewave i2 | 0.40 |
| 7 | 15 | 333330 | 0.0525 | 0.2165 | 0.0525 (rot.) | 0.0133 (planewave i2) | planewave i2 | 0.25 |
| 8 | 15 | 35148 | 0.0066 | 0.1847 | 0.0068 (rot.) | 0.0098 (planewave i2) | rotation | 1.48 |
| Σ | | | 0.1994 | 0.7292 | 0.2004 | 0.1652 | best 0.1015 | |
| evaluate | | | 0.344 | 0.881 | 0.354 | 0.316 | | |

#### W4 sphere (N = 100000), kD = 1, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0002 | 0.0042 | 0.0041 (batched-fused) | 0.0035 (planewave i2) | rotation | 23.12 |
| 3 | 7 | 53352 | 0.0015 | 0.0070 | 0.0071 (batched-fused) | 0.0060 (planewave i2) | rotation | 3.86 |
| 4 | 7 | 385344 | 0.0096 | 0.0085 | 0.0085 (batched-fused) | 0.0104 (planewave i2) | batched-fused | 1.23 |
| 5 | 7 | 325404 | 0.0081 | 0.0083 | 0.0085 (batched-fused) | 0.0140 (planewave i2) | rotation | 1.72 |
| Σ | | | 0.0195 | 0.0280 | 0.0281 | 0.0338 | best 0.0183 | |
| evaluate | | | 0.047 | 0.057 | 0.058 | 0.063 | | |

#### W4 sphere (N = 100000), kD = 1, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0007 | 0.0007 (rot.) | 0.0007 (rot.) | 0.0064 (planewave i2) | rotation | 9.49 |
| 3 | 15 | 53352 | 0.0093 | 0.1460 | 0.0095 (rot.) | 0.0100 (planewave i2) | rotation | 1.07 |
| 4 | 15 | 385344 | 0.0601 | 0.1722 | 0.0603 (rot.) | 0.0173 (planewave i2) | planewave i2 | 0.29 |
| 5 | 15 | 325404 | 0.0503 | 0.1798 | 0.0506 (rot.) | 0.0249 (planewave i2) | planewave i2 | 0.50 |
| Σ | | | 0.1203 | 0.4987 | 0.1211 | 0.0586 | best 0.0522 | |
| evaluate | | | 0.183 | 0.562 | 0.184 | 0.121 | | |

#### W4 sphere (N = 100000), kD = 100, ε = 1e-3, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0042 | 0.0043 (rot.) | 0.0042 (rot.) | 0.0527 (planewave i2) | rotation | 12.55 |
| 3 | 18 | 53352 | 0.0154 | 0.4517 | 0.0154 (rot.) | 0.0265 (planewave i2) | rotation | 1.72 |
| 4 | 11 | 385344 | 0.0277 | 0.0430 | 0.0272 (rot.) | 0.0223 (planewave i2) | planewave i2 | 0.81 |
| 5 | 7 | 325404 | 0.0081 | 0.0083 | 0.0085 (batched-fused) | 0.0181 (planewave i2) | rotation | 2.22 |
| Σ | | | 0.0555 | 0.5072 | 0.0553 | 0.1197 | best 0.0501 | |
| evaluate | | | 0.087 | 0.540 | 0.088 | 0.152 | | |

#### W4 sphere (N = 100000), kD = 100, ε = 1e-6, 1 × 72

| level | p | V pairs | rotation | batched-fused | auto | planewave | fastest | planewave ÷ fastest other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0066 | 0.0067 (rot.) | 0.0066 (rot.) | 0.0614 (planewave i2) | rotation | 9.25 |
| 3 | 22 | 53352 | 0.0268 | 0.0268 (rot.) | 0.0274 (rot.) | 0.0320 (planewave i2) | rotation | 1.19 |
| 4 | 15 | 385344 | 0.0601 | 0.1670 | 0.0603 (rot.) | 0.0274 (planewave i2) | planewave i2 | 0.46 |
| 5 | 15 | 325404 | 0.0506 | 0.1795 | 0.0504 (rot.) | 0.0309 (planewave i2) | planewave i2 | 0.61 |
| Σ | | | 0.1442 | 0.3799 | 0.1448 | 0.1517 | best 0.0918 | |
| evaluate | | | 0.211 | 0.447 | 0.212 | 0.218 | | |


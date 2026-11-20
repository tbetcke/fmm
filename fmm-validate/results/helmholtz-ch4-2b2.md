# Helmholtz CH4.2b2: the diagonal M2L on W1, measured

Task: `docs/helmholtz/tasks/CH4.2b2.md`. Roadmap: H4, CH4.2b2 (priority list item 10), the second
half of CH4.2b. Done when: "the FMM against the direct sum on W1 (kD = 500 at 10⁻⁶ and 10⁻⁸;
kD = 1000 at 10⁻⁶) within 1.1× RCR's error, the diagonal levels' M2L time against RCR's per level,
at 1 × 40 (Kathleen) and 1 × 72 (locust)". Two Alsos: the conversions' rate, and the transfer
tables' build time per level.

## 0. Summary

- **The acceptance holds on every case on both machines** (§2). With the diagonal form, the FMM's
  errors against the direct sum are 1.000× Rotation's errors for φ and ∇φ, to the three digits
  printed. Under the library's defaults the diagonal form takes only the levels Auto leaves on
  RCR, and the errors do not change at all.
- **The diagonal levels' M2L** (§3):
  - Where the rule takes every class, or 32 of 34, a level's M2L takes 0.08–0.46× RCR's time:
    0.08 at p = 245, 0.46 at p = 16.
  - On the few levels where it takes only some classes (12 or 21 of 34), the RCR pairs dominate
    the call, and the time is 0.95–1.02× RCR's.
  - Summed over the levels, at 1 × 40 on Kathleen, the M2L drops from 4.50 to 1.10 s at kD = 500
    and 10⁻⁶. At kD = 1000 it drops from 32.3 to 5.2 s.
  - The evaluation, at 1 × 40 on Kathleen, drops from 6.37 to 2.91 s and from 41.4 to 14.2 s. At
    1 × 72 on locust, from 2.02 to 1.14 s and from 13.3 to 5.5 s.
- **The diagonal form against the exponential M2L** (§3.2). Rotation with the diagonal form beats
  the library's defaults by 1.6–2.8× on the evaluation, on every case. The defaults put the
  exponential M2L on W1's levels 4 and below. There the diagonal form's level costs 0.29–0.83×
  the waves' over every case: 0.29–0.50× at kD = 500 and 10⁻⁶ on Kathleen, 0.32–0.58× on
  locust, and up to 0.83× on a level where only 12 classes pass. The rule ordering the
  two is CH4.3's decision; a question below adds it there.
- **The level call by part** (§4). The pairs take 50–80 % of the call's thread time. S → F and
  F → R take the rest, up to 50 % at p = 16. The conversions run at 1.7–5.1 GF/s per thread at
  p ≥ 41 and 0.8–1.4 at p ≤ 25 on Kathleen (40 threads), and 2.7–11 GF/s on locust (72 threads).
  CH4.1a measured 3–5 GF/s on one Kathleen core and 6–10 on one Grace core.
- **The transfer tables' build** (§5).
  - At one rank per node it costs about one class's build at the top level: p = 131 2.3 s and
    p = 245 11.9 s on Kathleen, 0.52 s and 3.3 s on locust. It doubles W1 at kD = 500's build
    (2.5 → 5.6 s at 1 × 40); one evaluation repays it at kD = 500, and at kD = 1000 nearly
    twice over.
  - At one rank per core every rank builds every class on its one thread. Level 2 then takes 61 s
    against an evaluation of 3.9 s (kD = 500, 40 × 1). This is where it dominates.
- **Ranks** (§6). W1 at kD = 500, 10⁻⁶, Rotation with the diagonal form, gives the same checksum at
  1 × 40, 2 × 20 and 40 × 1. At kD = 500 no level falls back to RCR at 40 ranks per node.

## 1. Setup

- **Runs.** `helmholtz_accuracy` (release, BLAS threads 1), launched by
  `tools/helmholtz/ch4-2b2-runs.sh`. Each launch builds an FMM per setting and evaluates it twice;
  the rows are the second evaluation's.
  - W1 (§9.1, complex charges, 1,000 samples, gradients) at kD = 500 (N = 1,989,437) at 10⁻⁶ and
    10⁻⁸, and at kD = 1000 (N = 7,957,748) at 10⁻⁶.
  - `--m2l rotation,auto-pw` (RCR on every level, and the library's defaults), each with
    `--diagonal off,rule`.
  - The near rules of the defaults (`--near-v cost --near-wx cost`), `--levels`.
- **Kathleen.** Job 653533 on node-c11a-025 (`singlenode`, 2 × 20 Cascade Lake cores), 1 × 40
  for the `w1` stage; 2 × 20 (one rank per socket) and 40 × 1 for `ranks`.
- **locust.** The GH200's 72 Grace cores at 1 × 72; idle before the runs (load 0.00, GPU 0 %).
- **Tables.** `tools/helmholtz/ch4-2b2-tables.py` makes them from the outputs. Raw outputs:
  `~/projects/fmm-archive/kathleen/runs/ch4-2b2/` and `~/projects/fmm-archive/locust/runs/ch4-2b2/`.
- **What the harness reports** (new here, `--levels`' table "Diagonal M2L"). Per level:
  - the classes taken and the V pairs on each form;
  - the wall seconds of the S → F pass and of the targets' pass;
  - the workers' thread-seconds of S → F, the pairs, F → R and the RCR pairs;
  - the conversions computed, their rate, and the build of the level's tables.

  The operator records these: `HelmholtzOperator::{diagonal_times, diagonal_build_time,
  diagonal_multiply_adds}`, and `DiagonalCounts::back`. Clock reads only; no value changes.

## 2. The acceptance: W1's errors

The relative ℓ2 errors over the 1,000 samples. "÷ Rotation" is the row's error over the
Rotation row without the diagonal form, for φ and ∇φ.

| machine | kD | ε | setting | diagonal levels (classes) | φ | ∇φ | ÷ Rotation |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| Kathleen 1 × 40 | 500 | 10⁻⁶ | Rotation | – | 1.18e-7 | 2.75e-7 | – |
| | | | Rotation + diagonal | 2–6 (34 each) | 1.18e-7 | 2.75e-7 | **1.000/1.000** |
| | | | defaults | – | 1.36e-7 | 3.18e-7 | 1.151/1.155 |
| | | | defaults + diagonal | 2, 3 (34) | 1.36e-7 | 3.18e-7 | 1.151/1.155 |
| | | 10⁻⁸ | Rotation | – | 2.72e-9 | 6.32e-9 | – |
| | | | Rotation + diagonal | 2, 3 (34); 4, 5 (32); 6 (12) | 2.72e-9 | 6.32e-9 | **1.000/1.000** |
| | | | defaults | – | 1.62e-9 | 3.21e-9 | 0.596/0.508 |
| | | | defaults + diagonal | 2, 3 (34) | 1.62e-9 | 3.21e-9 | 0.596/0.508 |
| | 1000 | 10⁻⁶ | Rotation | – | 1.03e-7 | 2.39e-7 | – |
| | | | Rotation + diagonal | 2–7 (34 each) | 1.03e-7 | 2.39e-7 | **1.000/1.000** |
| | | | defaults | – | 1.08e-7 | 2.60e-7 | 1.050/1.089 |
| | | | defaults + diagonal | 2–4 (34) | 1.08e-7 | 2.60e-7 | 1.050/1.089 |
| locust 1 × 72 | 500 | 10⁻⁶ | Rotation | – | 1.48e-7 | 3.54e-7 | – |
| | | | Rotation + diagonal | 2–6 (34); 7 (21) | 1.48e-7 | 3.54e-7 | **1.000/1.000** |
| | | | defaults | – | 1.44e-7 | 3.58e-7 | 0.970/1.011 |
| | | | defaults + diagonal | 2, 3 (34) | 1.44e-7 | 3.58e-7 | 0.970/1.011 |
| | | 10⁻⁸ | Rotation | – | 2.71e-9 | 6.31e-9 | – |
| | | | Rotation + diagonal | 2, 3 (34); 4, 5 (32); 6 (12) | 2.71e-9 | 6.31e-9 | **1.000/1.000** |
| | | | defaults | – | 1.62e-9 | 3.21e-9 | 0.597/0.510 |
| | | | defaults + diagonal | 2, 3 (34) | 1.62e-9 | 3.21e-9 | 0.597/0.510 |
| | 1000 | 10⁻⁶ | Rotation | – | 1.17e-7 | 2.96e-7 | – |
| | | | Rotation + diagonal | 2–7 (34); 8 (21) | 1.17e-7 | 2.96e-7 | **1.000/1.000** |
| | | | defaults | – | 1.18e-7 | 2.96e-7 | 1.005/1.000 |
| | | | defaults + diagonal | 2–4 (34) | 1.18e-7 | 2.96e-7 | 1.005/1.000 |

The checksums differ between each pair of rows with and without the diagonal form (the form
rounds differently), and the errors agree to three digits. Kathleen's tree and locust's differ:
they use other default leaf sizes (256 points on x86, 128 on NEON at ε ≥ 10⁻⁷), which give
locust one more level at 10⁻⁶.

The defaults' 1.151× at kD = 500, 10⁻⁶ on Kathleen is the exponential M2L's, not the diagonal
form's: it is the same with the form off. Recorded here; it is not this task's.

## 3. The M2L per level

### 3.1 Against RCR

Each level's M2L seconds (the slowest rank's level calls, second evaluation) without and with
the diagonal form, under Rotation. "ran" is `--levels`' column: "by pairs" is CH4.0i's RCR split.

| machine | kD, ε | level | p | without | s | with | s | with ÷ without |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Kathleen | 500, 10⁻⁶ | 2 | 131 | rotation by pairs | 1.555 | diagonal 34/34 | 0.304 | 0.195 |
| | | 3 | 72 | rotation by pairs | 0.996 | diagonal 34/34 | 0.198 | 0.199 |
| | | 4 | 41 | rotation | 0.735 | diagonal 34/34 | 0.177 | 0.240 |
| | | 5 | 25 | rotation | 0.742 | diagonal 34/34 | 0.236 | 0.319 |
| | | 6 | 16 | rotation | 0.472 | diagonal 34/34 | 0.183 | 0.388 |
| | | Σ | | | 4.500 | | 1.098 | 0.244 |
| | 500, 10⁻⁸ | 2 | 135 | rotation by pairs | 1.741 | diagonal 34/34 | 0.251 | 0.144 |
| | | 3 | 76 | rotation by pairs | 1.221 | diagonal 34/34 | 0.243 | 0.199 |
| | | 4 | 44 | rotation | 0.923 | diagonal 32/34 | 0.226 | 0.245 |
| | | 5 | 27 | rotation | 0.913 | diagonal 32/34 | 0.308 | 0.338 |
| | | 6 | 24 | rotation | 0.844 | diagonal 12/34 | 0.862 | 1.021 |
| | | Σ | | | 5.643 | | 1.891 | 0.335 |
| | 1000, 10⁻⁶ | 2 | 245 | rotation by pairs | 12.246 | diagonal 34/34 | 0.997 | 0.081 |
| | | 3 | 131 | rotation by pairs | 7.487 | diagonal 34/34 | 0.935 | 0.125 |
| | | 4 | 72 | rotation by pairs | 4.586 | diagonal 34/34 | 0.824 | 0.180 |
| | | 5 | 41 | rotation | 3.058 | diagonal 34/34 | 0.751 | 0.245 |
| | | 6 | 25 | rotation | 2.959 | diagonal 34/34 | 0.977 | 0.330 |
| | | 7 | 16 | rotation | 1.948 | diagonal 34/34 | 0.737 | 0.378 |
| | | Σ | | | 32.285 | | 5.221 | 0.162 |
| locust | 500, 10⁻⁶ | 2 | 131 | rotation by pairs | 0.504 | diagonal 34/34 | 0.068 | 0.134 |
| | | 3 | 72 | rotation by pairs | 0.192 | diagonal 34/34 | 0.040 | 0.207 |
| | | 4 | 41 | rotation | 0.151 | diagonal 34/34 | 0.038 | 0.254 |
| | | 5 | 25 | rotation | 0.145 | diagonal 34/34 | 0.056 | 0.387 |
| | | 6 | 16 | rotation | 0.140 | diagonal 34/34 | 0.063 | 0.453 |
| | | 7 | 15 | rotation | 0.124 | diagonal 21/34 | 0.117 | 0.948 |
| | | Σ | | | 1.255 | | 0.382 | 0.304 |
| | 500, 10⁻⁸ | 2 | 135 | rotation by pairs | 0.555 | diagonal 34/34 | 0.077 | 0.139 |
| | | 3 | 76 | rotation by pairs | 0.230 | diagonal 34/34 | 0.051 | 0.222 |
| | | 4 | 44 | rotation | 0.188 | diagonal 32/34 | 0.049 | 0.262 |
| | | 5 | 27 | rotation | 0.183 | diagonal 32/34 | 0.069 | 0.377 |
| | | 6 | 24 | rotation | 0.327 | diagonal 12/34 | 0.323 | 0.987 |
| | | Σ | | | 1.483 | | 0.569 | 0.384 |
| | 1000, 10⁻⁶ | 2 | 245 | rotation by pairs | 4.323 | diagonal 34/34 | 0.363 | 0.084 |
| | | 3 | 131 | rotation by pairs | 2.386 | diagonal 34/34 | 0.335 | 0.140 |
| | | 4 | 72 | rotation by pairs | 0.861 | diagonal 34/34 | 0.189 | 0.220 |
| | | 5 | 41 | rotation | 0.624 | diagonal 34/34 | 0.181 | 0.290 |
| | | 6 | 25 | rotation | 0.587 | diagonal 34/34 | 0.238 | 0.406 |
| | | 7 | 16 | rotation | 0.567 | diagonal 34/34 | 0.263 | 0.464 |
| | | 8 | 15 | rotation | 0.484 | diagonal 21/34 | 0.473 | 0.977 |
| | | Σ | | | 9.833 | | 2.043 | 0.208 |

Under the defaults the diagonal form takes the levels Auto leaves on RCR: 2–3 at kD = 500 and
2–4 at kD = 1000. There each level's time against RCR is that of the Rotation rows above
(0.08–0.23×), and the levels with waves are unchanged (0.97–1.03×).

### 3.2 Against the exponential M2L, and the evaluation

| machine | kD, ε | Rotation | Rotation + diagonal | defaults | defaults + diagonal | M2L Σ: Rotation + diagonal / defaults + diagonal |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Kathleen 1 × 40 | 500, 10⁻⁶ | 6.37 | **2.91** | 6.43 | 3.99 | 1.10 / 2.03 s |
| | 500, 10⁻⁸ | 8.40 | **4.62** | 10.13 | 7.14 | 1.89 / 3.71 s |
| | 1000, 10⁻⁶ | 41.44 | **14.25** | 40.14 | 17.77 | 5.22 / 8.81 s |
| locust 1 × 72 | 500, 10⁻⁶ | 2.02 | **1.14** | 2.02 | 1.44 | 0.38 / 0.71 s |
| | 500, 10⁻⁸ | 2.60 | **1.74** | 2.84 | 2.21 | 0.57 / 0.89 s |
| | 1000, 10⁻⁶ | 13.27 | **5.46** | 13.06 | 6.36 | 2.04 / 3.10 s |

Evaluation seconds (the second evaluation, slowest rank). On the levels where the defaults run
the exponential M2L, the diagonal form under Rotation is faster. On Kathleen at kD = 500, 10⁻⁶:

| level | p | waves (s) | diagonal (s) | diagonal ÷ waves |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 41 | 0.496 | 0.177 | 0.36 |
| 5 | 25 | 0.476 | 0.236 | 0.50 |
| 6 | 16 | 0.629 | 0.183 | 0.29 |

- Kathleen, 10⁻⁸: levels 4 and 5 give 0.33 and 0.36, and level 6 (12 classes) 0.52.
- Kathleen, kD = 1000: levels 5–7 give 0.46, 0.55 and 0.30.
- locust, kD = 500, 10⁻⁶: levels 4–6 give 0.32, 0.58 and 0.44.
- locust, 10⁻⁸: levels 4 and 5 give 0.29 and 0.37, and level 6 (12 classes) 0.83.
- locust, kD = 1000: levels 5–8 give 0.52, 0.66, 0.48 and 0.49.

The range over every case is 0.29–0.83. At 10⁻⁸ the waves run at the raised degree (p = 32 on
levels 5 and 6, CH6.1g) and the diagonal form at the levels' own degrees (24–27).

The defaults' errors are lower at 10⁻⁸ (0.6× Rotation's: the raised degree of CH6.1g), so a fair
comparison at matched accuracy is CH4.3's and CH4.5's. At 10⁻⁶ the two are within 1.15×.

## 4. The level call by part, and the conversions

The diagonal level call's thread-seconds by part: S → F with its gather, the pairs with the
accumulators' zeroing, F → R, and the RCR pairs. The rate is 2 × the multiply-adds of every
conversion (`Conversions::multiply_adds`, both steps) over the part's thread-seconds, per
thread. Rotation + diagonal.

| machine | kD, ε | level | p | S → F | pairs | F → R | RCR | conversions' share | S → F, F → R GF/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Kathleen | 500, 10⁻⁶ | 2 | 131 | 0.64 | 5.28 | 1.01 | 0 | 0.24 | 4.06, 2.58 |
| | | 3 | 72 | 0.88 | 4.59 | 1.00 | 0 | 0.29 | 2.45, 2.15 |
| | | 4 | 41 | 1.03 | 4.47 | 1.07 | 0 | 0.32 | 1.74, 1.68 |
| | | 5 | 25 | 1.44 | 6.41 | 1.34 | 0 | 0.30 | 1.25, 1.34 |
| | | 6 | 16 | 2.07 | 3.65 | 1.42 | 0 | 0.49 | 0.76, 1.11 |
| | 1000, 10⁻⁶ | 2 | 245 | 3.31 | 16.06 | 5.58 | 0 | 0.36 | 5.07, 3.01 |
| | | 4 | 72 | 3.76 | 22.94 | 4.53 | 0 | 0.27 | 2.44, 2.02 |
| | | 7 | 16 | 7.81 | 15.31 | 5.91 | 0 | 0.47 | 0.82, 1.09 |
| | 500, 10⁻⁸ | 6 | 24 | 3.30 | 1.03 | 2.70 | 27.12 | 0.18 | 1.16, 1.42 |
| locust | 500, 10⁻⁶ | 2 | 131 | 0.28 | 2.05 | 0.39 | 0 | 0.25 | 9.25, 6.67 |
| | | 4 | 41 | 0.34 | 1.57 | 0.49 | 0 | 0.35 | 5.21, 3.66 |
| | | 6 | 16 | 0.72 | 2.83 | 0.64 | 0 | 0.32 | 2.73, 3.08 |
| | 1000, 10⁻⁶ | 2 | 245 | 1.50 | 10.89 | 2.42 | 0 | 0.26 | 11.18, 6.93 |
| | | 7 | 16 | 2.94 | 13.02 | 2.53 | 0 | 0.30 | 2.66, 3.09 |

Thread-seconds; "conversions' share" is (S → F + F → R) over the four parts.

- **The pairs take most of the call:** 50–80 % of the thread time where every class is diagonal.
  A pair is one complex multiply-add per sample, read through the class's sample map, and so
  bound by bandwidth. On a mixed level the RCR pairs dominate: 79 % of the thread time at
  kD = 500, 10⁻⁸, level 6, where 80 % of the pairs stay RCR.
- **The conversions take 24–49 %**, and the larger share at the smaller degrees.
  - At 40 threads on Kathleen they run at 1.7–5.1 GF/s per thread at p ≥ 41, near CH4.1a's
    3–5 on one core, falling to 0.8–1.4 GF/s at p ≤ 25.
  - On locust they run at 3.3–11 GF/s at p ≥ 41 (CH4.1a: 6–10), and 2.7–4.0 at p ≤ 25.
  - The small degrees lose the most: a box's conversion is short there, and the gather and the
    per-block overheads weigh more.
- **The Also's verdict.** The conversions matter: a quarter to a half of a diagonal level's
  call. A kernel twice as fast (a dispatched column kernel in `nd-fmm-simd`, FMA at 8 lanes on
  AVX-512) would save 12–25 % of a diagonal level. The pairs are the larger share, and a kernel
  for them (two contiguous runs per ring, a complex multiply-add) would gain more. Neither
  changes the order against RCR or the waves (§3). Proposed as a follow-up below, question 3.

## 5. The transfer tables' build

Each diagonal level builds its transfer functions (`DiagonalM2l::build_on`: ref's double-double
`Transfer` per stored class, one part per class on the pool) and its conversions. The seconds,
the slowest rank's:

| machine | split | kD, ε | level | p | classes | transfer functions | conversions |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Kathleen | 1 × 40 | 500, 10⁻⁶ | 2 | 131 | 34 | 2.34 | 0.069 |
| | | | 3 | 72 | 34 | 0.39 | 0.012 |
| | | | 4 | 41 | 34 | 0.08 | 0.003 |
| | | 1000, 10⁻⁶ | 2 | 245 | 34 | 11.90 | 0.43 |
| | | | 3 | 131 | 34 | 1.77 | 0.08 |
| | | | 4 | 72 | 34 | 0.34 | 0.013 |
| | 2 × 20 | 500, 10⁻⁶ | 2 | 131 | 34 | 3.39 | 0.069 |
| | 40 × 1 | 500, 10⁻⁶ | 2 | 131 | 34 | **61.02** | 0.095 |
| | | | 3 | 72 | 34 | 10.24 | 0.016 |
| | | | 4 | 41 | 34 | 1.94 | 0.004 |
| locust | 1 × 72 | 500, 10⁻⁶ | 2 | 131 | 34 | 0.52 | 0.038 |
| | | 1000, 10⁻⁶ | 2 | 245 | 34 | 3.30 | 0.22 |

- **At 1 × 40 a level costs one class's build.** The 34 classes are 34 parts on 40 threads, so
  each thread builds about one class: about 1.8 s at p = 131 and 12 s at p = 245 on a Kathleen
  core. Locust's cores are 3.5–4.5× faster per class here.
- **The build against the evaluation.**
  - The diagonal tables take 2.9 s of W1 at kD = 500's build at 1 × 40 (2.5 → 5.6 s), and one
    evaluation saves 3.5 s.
  - At kD = 1000 they take 14.6 s (12.9 → 28.4 s), and an evaluation saves 27.2 s.
  - So at one rank per node the build is repaid within one evaluation, but it is now the
    largest part of the Helmholtz build at kD = 1000.
- **At one rank per core it dominates.** Each rank builds every class on its one thread: 61 s for
  level 2 and 73.9 s in all at 40 × 1, against an evaluation of 3.87 s. At kD = 1000 the same
  split would take about 34 × 12 s ≈ 7 minutes per rank (an estimate, not run).
- **The interpolated build.** Proposal §5.4 names it: T interpolated in ŝ · t̂ from an
  extended-precision table, O(p²) per class. S6 measured it 11× faster at p = 256, with
  C_T = 0.23 in place of 0.18. Parts finer than a class (by ring) would remove the one-class floor
  at one rank per node. A node-shared copy is CH8.0's; at many ranks per node every rank now
  builds the same tables. Proposed below, question 2.

## 6. Ranks

| split | levels (classes) | evaluate s | build s | diagonal tables s | checksum | rank 0 peak |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 1 × 40 | 2–6 (34 each) | 2.91 | 5.59 | 2.93 | c74ca0e7954cf9ba | 6.9 GB (4 FMMs in one process) |
| 2 × 20 | 2–6 (34 each) | 2.53 | 5.73 | 4.19 | c74ca0e7954cf9ba | 2.8 GB |
| 40 × 1 | 2–6 (34 each) | 3.87 | 78.87 | 73.90 | c74ca0e7954cf9ba | 1.1 GB |

W1 at kD = 500, 10⁻⁶, Rotation with the diagonal form, Kathleen job 653533. The table "Diagonal
M2L" of the runs on several ranks prints GF/s from counts summed over the ranks against the
slowest rank's seconds; the harness prints "–" there since the self-review, and the report quotes
one-rank rates only.
- The output is the same bits at every split (constraint 2).
- At 40 ranks per node every level still fits its 205 MiB share of the budget. CH4.1b's census
  put the fallback at kD = 1000's levels 2–3; not run here at 40 × 1, since its build alone would
  take minutes (§5).

## 7. Questions for sign-off

1. **The acceptance as read.** On W1 at kD = 500 (10⁻⁶, 10⁻⁸) and kD = 1000 (10⁻⁶), at 1 × 40 on
   Kathleen and 1 × 72 on locust:
   - the FMM with the diagonal form is 1.000× Rotation's errors;
   - each diagonal level's M2L time is reported against RCR's: 0.08–0.46× where every class or 32
     of 34 are diagonal, 0.95–1.02× on the levels of 12 or 21 classes.

   *Recommended: accept.*
2. **The transfer tables' build.** At one rank per node it is about one class's build at the
   top level (2.3 s at p = 131, 11.9 s at p = 245 on Kathleen). At one rank per core it is every
   class, serially (61 s against an evaluation of 3.9 s).

   *Recommended:* a new task CH4.2c before CH4.3, "The transfer functions interpolated, and built
   in parts finer than a class". It covers T interpolated in ŝ · t̂ (proposal §5.4, S6: 11× at
   p = 256, C_T = 0.23) against the direct build, within the bound with C_T = 0.23, and parts by
   ring. It goes before CH4.3, because CH4.3 calibrates C_T and θ, and the interpolated T moves
   C_T.

   *Alternatives:* an Also on CH4.3; or leave the build to CH8.0 (tables per node) and CH4.4's
   per-k build report.
3. **Kernels for the pair and the conversions** (CH4.1a's Also, answered: they matter, 24–49 %
   for the conversions and the rest for the pairs).

   *Recommended:* a follow-up task off the list, CH4.2d, "Dispatched kernels for the diagonal
   pair and the conversions" (`nd-fmm-simd`, constraint 1), reconsidered at CH4.5, the comparison
   with FMM3D. The diagonal levels are already 0.08–0.46× RCR's, and the far field now waits
   more on the exponential M2L and the P2P.

   *Alternatives:* on the list before CH4.3; or no task.
4. **The diagonal form against the exponential M2L** (§3.2). Under Rotation the diagonal form
   beats the defaults' waves on W1's levels 4–8: 0.29–0.83× per level, and 1.6–2.8× on the
   evaluation.

   *Recommended:* an Also on CH4.3, "the rule's order between the exponential M2L (D8) and the
   diagonal form, per level, by time at matched accuracy", with D8 amended there if it moves. No
   default changes here.

   *Alternative:* a separate task after CH4.3.

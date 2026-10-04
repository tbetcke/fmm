# Phase 4 benchmarks on the Apple M3 Max (T13)

The device FMM of Phase 4 against the host path, the device kernels against the peak and
the GEMM spike, and the device leaf-size study (docs/phase4/T13-benchmarks.md). Run on
2026-10-04, 18:15–18:51.

**Every number below is measured on the Apple M3 Max: Metal f32 for the device, and NEON
for the host (`P2pChoice::Auto`).** A figure labelled *model* is not a measurement.
The peak used for "% of peak" is the spike report's: 14.3 TFLOP/s f32 for the M3 Max GPU
(derived, not measured) and 400 GB/s. The P2P peak model of device-path.md §13.4 is
720 Gpairs/s (φ) and 480 (φ and ∇φ) (*model*).

**Not measured:**
- **f64 on a GPU.** Metal has no f64, and no other GPU was available (decision 4). f64
  runs only on the CubeCL CPU runtime, a correctness backend, and is not timed here.
- **CUDA.** It is type-checked only (`cargo check -p nd-fmm-validate --features cuda
  --examples`, done for this report). The run that would measure it is one command
  ("The CUDA run" below), and no result is claimed from it.
- x86_64 hosts, external baselines (decision 9 recommends later), and multi-rank device
  runs (C5.1).

## Setup

| item | value |
| --- | --- |
| machine | MacBook Pro 16" (Mac15,9), Apple M3 Max: 12 performance and 4 efficiency cores; 40-core GPU, plane size 32, 32 KB shared memory per cube; 64 GB unified memory, on AC power |
| software | macOS 27.0, rustc 1.99.0, release build, aarch64-macos default target; CubeCL 0.11.0-pre.4 (`metal` = wgpu with the MSL compiler) |
| environment | `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` all 1; `RAYON_NUM_THREADS=1` for the kernel harnesses; every run outside the macOS sandbox (Metal needs GPU access); one run at a time, no other build running |
| host comparisons | the Phase 3S host path: `P2pChoice::Auto` (NEON), 1 thread and 12 threads (the performance cores) |
| device timing | compilation excluded (warm-up launches or evaluations); FMM: the median of 5 evaluations, each queuing every launch and syncing once; stage times from a second build with `synchronous_stages` (a sync after each stage), the median per stage over 5 evaluations; kernel harnesses: the median of 15 batches of at least 20 ms of queued launches |
| runs | one run of each harness; about ±25% run-to-run variance is expected on this laptop GPU (Phase 0 T6), and the rotation harness shows ±25% between two tables of the same run |
| tables | through a table cache (loading measured in the build) |

Commands (from the workspace root; `DIR` a scratch directory):

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cargo run --release -p nd-fmm-validate --features metal --example device_fmm -- \
    --device metal --part fmm --tuning-cache DIR/tune --table-cache DIR/tables
cargo run --release -p nd-fmm-validate --features metal --example device_fmm -- \
    --device metal --part leaf --table-cache DIR/tables
export RAYON_NUM_THREADS=1
cargo run --release -p nd-fmm-validate --features metal --example fmm_accuracy -- --backend metal --distribution plummer
cargo run --release -p nd-fmm-validate --features metal --example p2p_kernels -- --device metal --fmm
cargo run --release -p nd-fmm-validate --features metal --example leaf_kernels -- --device metal
cargo run --release -p nd-fmm-validate --features metal --example translation_kernels -- --device metal
cargo run --release -p nd-fmm-validate --features metal --example m2l_kernels -- --device metal --table-cache DIR/tables
cargo run --release -p nd-fmm-validate --features metal --example rotation_kernels -- --device metal --table-cache DIR/tables
```

## 1. The device FMM against the host path

The T12 problems: the uniform cube and the Plummer sphere (a = 0.1), N = 10⁵, sources
equal to targets, `max_level` 16, 64 points per leaf, eight charge vectors, gradients;
and the cube at N = 10⁶. f32 at p = 3, 6 and 8. The device runs every operator kind.
"Tuned" is `M2lStrategy::Auto` with a fresh tuning cache (the default 10 s budget); the
other device rows are fixed strategies (`Classes` runs as dense on the device). The full
tables are in "Raw output" below.

### Evaluation time

Median ms of one evaluation. Host: its default (`Auto`, which is `Dense` at p ≤ 8). The
speed-ups are of the tuned device over that host.

| problem | p | host 1 thread | host 12 threads | Metal tuned (strategy) | Metal Dense | Metal Rotation | x host 1 | x host 12 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| cube, N = 10⁵ | 3 | 73.2 | 8.39 | 2.08 (Dense) | 2.09 | 3.52 | 35.3 | 4.0 |
| cube, N = 10⁵ | 6 | 227 | 24.7 | 4.41 (Dense) | 4.38 | 5.68 | 51.6 | 5.6 |
| cube, N = 10⁵ | 8 | 514 | 50.9 | 6.71 (Dense) | 7.03 | 8.88 | 76.6 | 7.6 |
| Plummer, N = 10⁵ | 3 | 246 | 27.5 | 3.59 (Dense) | 3.55 | 5.17 | 68.5 | 7.7 |
| Plummer, N = 10⁵ | 6 | 726 | 81.3 | 6.58 (Dense) | 6.84 | 9.34 | 110 | 12.3 |
| Plummer, N = 10⁵ | 8 | 1,364 | 140 | 10.90 (Dense) | 11.47 | 15.00 | 125 | 12.9 |
| cube, N = 10⁶ | 3 | 778 | 75.2 | 15.4 (Dense) | 16.0 | 20.6 | 50.6 | 4.9 |
| cube, N = 10⁶ | 6 | 2,224 | 218 | 41.8 (**Rotation**) | 41.8 | 42.1 | 53.3 | 5.2 |
| cube, N = 10⁶ | 8 | 4,926 | 498 | 73.5 (**Rotation**) | 81.9 | 74.2 | 67.0 | 6.8 |

- The device is 35–125× the host at one thread and 4.0–12.9× at 12 threads. The gain is
  largest where the host's leaf stage is large: the Plummer sphere, whose small leaves
  (17.6 points on average) and W and X lists keep the host's leaf stage at 44–67% of its
  evaluation.
- **The tuner chose dense M2L at every degree at N = 10⁵, and rotation on the cube at
  N = 10⁶ for p = 6 (a tie, 41.75 against 41.82 ms) and p = 8 (73.5 against 81.9 ms
  dense).** The f32 static rule (dense at every p) therefore holds at N = 10⁵ but not at
  N = 10⁶. The dense downward stage per V pair grows from 9.5 ns at N = 10⁵ to 11.6 ns at
  N = 10⁶ (p = 8, synchronous stages), while rotation's falls from 11.8 to 10.3 ns. The
  cause was not analysed here (the dense level calls at N = 10⁶ run in more chunks of
  the 128 MB scratch).
- Tuned evaluations ran at 0.99–1.11× the speed of the static rule's (Dense) at
  N = 10⁵, within the run-to-run variance (T12's single runs gave 1.05–1.8×). Tuning took
  2.9–6.6 s per build at N = 10⁵ and 3.6–5.9 s at N = 10⁶, within the 10 s budget.

### Stages

Medians in ms. Host: its timed evaluations. Device: a build with `synchronous_stages`
(each stage waits for the device, so the stages add up to more than an evaluation).
Upward = P2M and M2M, downward = L2L, M2L and P2L, leaves = L2P, M2P and P2P. The
design's timing resolves stages, not kinds; M2L and P2P alone are in Section 2.

| problem | p | run | upward | downward | leaves | near share | evaluate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| cube, N = 10⁵ | 3 | host 1 thread | 2.55 | 34.3 | 35.9 | 49% | 73.2 |
| | | host 12 threads | 0.81 | 3.52 | 3.56 | 43% | 8.39 |
| | | Metal tuned | 0.58 | 0.93 | 0.85 | 24% | 2.08 |
| cube, N = 10⁵ | 8 | host 1 thread | 16.7 | 433 | 62.6 | 12% | 514 |
| | | host 12 threads | 2.43 | 41.8 | 6.32 | 12% | 50.9 |
| | | Metal tuned | 0.91 | 5.10 | 0.86 | 10% | 6.71 |
| Plummer, N = 10⁵ | 3 | host 1 thread | 2.67 | 78.2 | 164 | 67% | 246 |
| | | host 12 threads | 1.39 | 8.36 | 17.0 | 62% | 27.5 |
| | | Metal tuned | 0.89 | 1.74 | 1.63 | 29% | 3.59 |
| Plummer, N = 10⁵ | 8 | host 1 thread | 17.7 | 750 | 599 | 44% | 1,364 |
| | | host 12 threads | 3.12 | 76.4 | 60.3 | 43% | 140 |
| | | Metal tuned | 0.97 | 7.98 | 2.50 | 19% | 10.90 |
| cube, N = 10⁶ | 8 | host 1 thread | 158 | 4,061 | 701 | 14% | 4,926 |
| | | host 12 threads | 15.6 | 411 | 66.4 | 13% | 498 |
| | | Metal tuned (Rotation) | 1.88 | 62.3 | 5.07 | 7% | 73.5 |

On the device the far field (downward, that is M2L) dominates from p = 6 on: the leaf
stage is 7–29% of the device's stages, against 12–67% on the host.

### Build, transfers, launches, syncs and errors

| item | measured |
| --- | --- |
| build, N = 10⁵ | device 64–106 ms with the tables loaded (0.1–10.7 ms of it) and no tuning; host 51–89 ms with the tables loaded; tuning adds 2.9–6.6 s |
| build, N = 10⁶ | device 743–763 ms, host 621–652 ms; tuning adds 3.6–5.9 s |
| transfers per evaluation | one upload (the charges: 0.4 MB at N = 10⁵, 4 MB at N = 10⁶) and one download (φ and ∇φ: 1.6 / 16 MB): the design's minimum, on every run |
| syncs per evaluation | 1 (the download), on every run |
| launches per evaluation | cube N = 10⁵: 40, 43, 46 at p = 3, 6, 8 (34 under `Rotation`); Plummer: 101, 101, 107 (87); cube N = 10⁶: 64, 97, 133 (41) |
| device memory | N = 10⁵: 68–172 MB (dense), 20–30 MB (rotation); N = 10⁶: 322–358 MB (dense), 190–228 MB (rotation) |
| device − host output | relative L2 over every target and charge vector at most 2.9e-6 (φ) and 1.9e-7 (∇φ); bound 1e-5 |
| errors against the direct sum | each of φ L2, φ max, ∇φ L2 and ∇φ max within 0.991–1.002 of the host's of the same strategy (bound 5%) |
| determinism | two evaluations bit-identical on every run |
| against T11 | `fmm_accuracy --backend metal` (T11's harness) gives the Plummer errors of the Dense rows to every printed digit: 3.202e-3 / 3.984e-3 / 1.116e-3 / 2.606e-4 at p = 3 and 2.189e-5 / 3.885e-5 / 1.394e-5 / 5.180e-6 at p = 8. The cube rows give the error ratio 0.9952 at p = 8 that T11 reported (0.995) |

## 2. Kernel efficiency, Metal f32

From the T6–T10 harnesses, run again here (raw output below). Every device result was
first checked against the host or the reference (all pass).

### P2P (`p2p_kernels --device metal --fmm`)

| cell | output | Gpairs/s (cube 64, the default) | of the peak model | x NEON 1 thread | x NEON 12 threads | C4.2 target |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| W1 n_t = 64, gathered, 4,096 leaves per launch | φ | 352 | 48.8% | 90.8 | 10.1 | 10%: met |
| W1 n_t = 64, gathered | φ, ∇φ | 287 | 59.7% | 121 | 12.9 | 10%: met |
| W2 N = 10⁵, targets ≠ sources | φ | 324 | 45.1% | 81.7 | 7.1 | 25%: met |
| W2 N = 10⁵, targets ≠ sources | φ, ∇φ | 269 | 56.0% | 104 | 9.1 | 25%: met |

- For small leaves (W1 n_t = 8) the cube with 32 units and the plane layout are
  1.2–1.4× faster than the default cube of 64 units. The tuner picks them on the cube
  (plane 2 or cube 32) and keeps cube 64 on the Plummer sphere.
- The leaf stage inside the FMM (p = 3): 0.85 ms (cube, plane layout) and 1.91 ms
  (Plummer) on the device, against 35.4 / 167 ms (1 thread) and 3.79 / 17.7 ms (12
  threads) on the host.

### Leaf operators (`leaf_kernels --device metal`), totals over the levels, ms

| problem | p | P2M | L2P | P2L | M2P | x host 1 thread | x host 12 threads |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| C3.2 cube | 3 | 0.021 | 0.014 | – | – | 110, 484 | 15.1, 57.4 |
| C3.2 cube | 8 | 0.078 | 0.032 | – | – | 171, 1036 | 16.7, 95.9 |
| Plummer | 3 | 0.054 | 0.031 | 0.386 | 0.246 | 44, 219, 91, 509 | 14.6, 40.8, 12.0, 81.6 |
| Plummer | 8 | 0.134 | 0.067 | 1.221 | 1.070 | 101, 501, 157, 500 | 15.2, 61.3, 19.0, 77.2 |

### M2M and L2L GEMM (`translation_kernels --device metal`), the cube's level 4 (L2L, 4,096 pairs)

| p | GEMM of the plan | GFLOP/s | % of peak | spike at (p, 10⁴): hand-written / library | gather + accumulation share of the level call |
| ---: | --- | ---: | ---: | --- | ---: |
| 3 | hand-written | 488 | 3.4% | 1,431 / rejected | 60% |
| 8 | library | 2,469 | 17.3% | 3,585 / 3,792 | 40% |
| 16 | library | 3,748 | 26.2% | 3,572 / 4,475 | 19% |

Small levels are latency-bound: one hand-written GEMM takes about 4 µs at p = 3 whatever
the batch. The level calls of M2M and L2L are a small part of an evaluation (upward
0.6–1.9 ms in the stages above).

### Dense M2L (`m2l_kernels --device metal`)

The hand-written GEMM (the device default) on the cube's level 4 (584,136 pairs), with the
spike (one table, B columns; CubeCL 0.11, results-m3max-0.11.md) at the nearest cell:

| p | hand-written GEMM GFLOP/s | % of peak | library (padded) GFLOP/s, useful columns | spike at (p, 10³): hand-written / library |
| ---: | ---: | ---: | --- | --- |
| 3 | 1,325 | 9.3% | rejected | 239 / rejected (p = 4) |
| 8 | 2,193 | 15.3% | 1,402, 55% | 625 / 2,045 |
| 12 | 2,526 | 17.7% | 2,037, 62% | 1,426 / 3,396 |
| 16 | 2,847 | 19.9% | 2,438, 65% | 2,146 / 3,894 |

The whole M2L stage (every V level, device level calls):

| problem | p | device ms | GEMM / gather / accumulation | launches | per-offset structure (A) ms, launches | x host 1 thread | x host 12 threads |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| cube | 3 | 0.60 | 43% / 18% / 29% | 9 | 24.9, 2,844 | 59.3 | 7.6 |
| cube | 8 | 8.42 | 73% / 13% / 12% | 24 | 23.8, 2,844 | 56.7 | 12.9 |
| Plummer | 3 | 0.89 | 49% / 19% / 39% | 21 | 58.0, 6,636 | 51.2 | 6.1 |
| Plummer | 8 | 18.6 | 83% / 12% / 7% | 36 | 55.9, 6,636 | 30.1 | 7.3 |

The C4.5 gate (the GEMM alone on the spike's shapes, against the spike's recorded 0.11
figures, not a same-day spike run): 26–97% of the spike, 80% met in 1 of 12 cells (p = 4,
B = 10³); the library 43–80% at p ≥ 8. T9 measured 67–87% (library) against a same-day
spike run, which came out 10–25% below the recorded figures; the gate stays accepted as
analysed (decision 12). The scratch budget does not matter between 16 and 512 MB.

### Rotation against dense (`rotation_kernels --device metal`), every V level of the C3.2 cube

| p | rotation ns/pair | % of peak (the spike's flop count, a model) | dense (hand-written) ns/pair | rotation / dense | break-even on the M3, f32 (spike, measured) | A100/H100 f64 break-even (*model*) |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | 2.33 | 0.54% | 0.68 | 3.44 | – | – |
| 4 | 3.44 | 1.69% | 2.75 | 1.25 | 13.9% | – |
| 6 | 7.87 | 2.03% | 5.83 | 1.35 | – | – |
| 8 | 16.6 | 2.05% | 11.6 | 1.43 | 9.8% | 13.2% |
| 12 | 51.9 | 1.97% | 41.6 | 1.25 | 7.1% | 5.4% |
| 16 | 140 | 1.64% | 108 | 1.30 | 4.8% | 3.8% |

- Rotation stays far below every break-even efficiency, as in T10 (1.8–3.0% there). The
  same run timed level 4 alone at p = 8 at 13.0 ns per pair (2.6%), against 16.6 ns over
  every level: the variance between tables is about ±25%.
- The M2L stage at p = 8 under rotation: 9.97 ms on the device against 580 ms (1 thread)
  and 61.9 ms (12 threads) for the host rotation.
- The f64 crossover on a data-centre GPU stays a model: if CUDA f64 rotation reached the
  same 1.6–2.6% of its peak, it would sit below the central break-even values at p = 8,
  12 and 16, so dense would win to beyond p = 16 (T10 and T12 reached the same
  conclusion). No f64 GPU ran.

## 3. The device leaf-size study

`device_fmm --part leaf`: the device under its static rule (dense, no tuning), Metal f32,
the cube and the Plummer sphere of Section 1 at N = 10⁵, p = 3 and 8, with the refinement
targets 16, 32, 64, 128 and 256. Evaluation time in ms (median of 5); the host's one-thread
f32 times from Phase 3S T7 (design §7, Phase 3S, "Leaf size") beside them.

| distribution | p | 16 | 32 | 64 | 128 | 256 | fastest (64 / fastest) | near share at 64, device / host | host 1 thread (Phase 3S T7) at 16, 32, 64, 128, 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| cube | 3 | 7.81 | 2.45 | **2.18** | 2.21 | 3.65 | 64 | 20% / 50% | 386, 152, 75, **74**, 173 |
| cube | 8 | 66.3 | 8.03 | 6.86 | 7.03 | **4.28** | 256 (1.60) | 10% / 12% | 3,946, 854, 499, 506, **243** |
| Plummer | 3 | 6.66 | 4.40 | **3.61** | 3.95 | 6.14 | 64 | 28% / 67% | 358, 288, **246**, 255, 295 |
| Plummer | 8 | 36.1 | 19.4 | 11.4 | **9.30** | 10.7 | 128 (1.23) | 18% / 44% | 3,069, 1,921, 1,352, 1,146, **1,145** |

- The trees: the cube at 64 and 128 is the same uniform level-4 tree (4,096 leaves, 24.4
  points on average); 16 and 32 split it unevenly (31,543 and 5,657 leaves), 256 is the
  level-3 tree (512 leaves, 195 points). The Plummer sphere has 21,652 to 2,157 leaves.
- The near/far balance moves as on the host, but the device's leaf stage is cheap: at
  64 points per leaf it is 10–28% of the device's stages, against 12–67% on the host. Small
  leaves cost the device most through M2L: at 16 points per leaf the cube has 5.6 × 10⁶
  V pairs (at p = 8 the far field takes 64 ms of 66).
- Errors (relative L2, f32): φ barely moves with the size (cube 1.60–1.89e-5, Plummer
  2.09–2.27e-5 at p = 8); ∇φ improves with larger leaves (cube 2.81e-5 at 16 to 1.18e-5
  at 256 at p = 8). The values at 64 equal those of Section 1.

**The device leaf-size rule** (fixed in the T13 brief): the geometric mean of the Metal
evaluation time over the four configurations is 18.79, 6.40, 4.98, 4.89 and 5.66 ms at
16, 32, 64, 128 and 256. 128 is fastest, but only 1.9% faster than 64, below the 5% the
rule requires (its errors equal or improve on 64's). **The rule picks 64: the device
backend gets no default of its own**, and `DEFAULT_MAX_POINTS_PER_LEAF` stays 64 for both.
The choice rests on M3 Max Metal timings only. As on the host, the best size depends on p
(64 at p = 3, 128–256 at p = 8), so a size per p, or a tuned one, would gain more than
any single default (design §9.2).

## 4. The CUDA run (not run; type-checked)

No CUDA machine was available in Phase 4. On an A100- or H100-class machine with CUDA
and the native prerequisites of the root `CLAUDE.md` (MPI, BLAS/LAPACK), this one
command, from the workspace root, runs the device FMM, the autotune tables and the GEMM
spike, in f32 and f64, and writes Markdown into `cuda-run/` (outside the repository's
tracked files; do not commit it):

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1; \
mkdir -p cuda-run && for prec in f32 f64; do \
  cargo run --release -p nd-fmm-validate --features cuda --example device_fmm -- --device cuda \
    --precision $prec --tuning-cache cuda-run/tune --table-cache cuda-run/tables > cuda-run/device_fmm-$prec.md && \
  cargo run --release -p nd-fmm-validate --features cuda --example autotune -- --device cuda \
    --precision $prec --tuning-cache cuda-run/autotune --table-cache cuda-run/tables > cuda-run/autotune-$prec.md || break; \
done && \
RAYON_NUM_THREADS=1 cargo run --release -p nd-fmm-validate --features cuda --example m2l_kernels -- \
  --device cuda --table-cache cuda-run/tables > cuda-run/m2l_kernels-f32.md && \
cargo run -p nd-fmm-spike-cubecl-gemm --release --no-default-features --features cuda -- \
  --backends cuda > cuda-run/gemm-spike.md
```

- `device_fmm` runs f32 at p = 3, 6, 8 and f64 at p = 8, 12, 18 (its defaults), the cube
  and the Plummer sphere at N = 10⁵ and the cube at N = 10⁶, against the host at one
  thread and at the performance cores (on Linux, every logical CPU unless `--threads` is
  given), and the leaf-size study at p = 3 and 8 in each precision.
- `autotune` tunes f32 at p = 3, 6, 8 and f64 at p = 4, 8, 12, 16, with every
  candidate's time.
- The spike measures the f64 GEMM alone: the hand-written kernels against the library's
  scalar path, which is all f64 gets (no f64 MMA in CubeCL).

**The numbers it would settle:**
- the f64 GEMM efficiency of the hand-written kernel on a data-centre card (the spike's
  model predicts 19–36% of the roofline);
- the f64 dense/rotation crossover, from `autotune`'s strategy decision per p and
  `device_fmm`'s fixed-strategy rows at p = 8, 12, 18;
- and so the f64 static rule (dense to p = 11, rotation from 12, provisional; decision
  13), which the tuner replaces per device wherever a cache exists;
- also: the f32 GEMM on CUDA (the input-precision guard is expected to keep the library
  off, since CUDA's MMA paths are TF32 at best), and the device FMM against an x86_64 host.

Checked here: `cargo check -p nd-fmm-validate --features cuda --examples` type-checks
every example above, and `cargo check -p nd-fmm-spike-cubecl-gemm --release
--no-default-features --features cuda` the spike. Nothing was linked against or run on a
CUDA driver, and no result is claimed.

## Raw output

Machine-specific paths are replaced by `DIR`. Every table is as printed by the example.

### `device_fmm --part fmm`

<details><summary>Output</summary>

### The device FMM against the host path (Phase 4 T13)

| item | value |
| --- | --- |
| machine | Apple M3 Max; 16 physical, 16 logical, 12 performance; aarch64-macos, release build; rustc 1.99.0 (b940084d7 2026-09-28) |
| device | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| BLAS threads | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1 |
| precision | f32 |
| host | `P2pChoice::Auto` at 1 and 12 threads; one rank |
| timing | evaluate: median wall time of 5 evaluations of the first charge vector after the eight error evaluations (compilation excluded; one sync per device evaluation); stages: on the device the medians of a synchronous-stages build (a sync after each stage), on the host the medians of the timed evaluations |
| tuning | a fresh tuning cache per (problem, p) under `DIR/tune`, the default budget |
| tables | through the table cache `DIR/tables` |
| wall time of every run | 1144 s |

#### cube, N = 100000, uniform in the cube [-1, 1)^3 (f32), 64 per leaf

##### p = 3: 4096 leaves on 5 levels, points per leaf 7 / 24.4 / 43 (min / mean / max), lists U 93240 V 640584 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 12: the evaluation's speed-up over the host with the same strategy at 1 / 12 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 71.70 | 22.26 B | – | 2.55 | 34.31 | 35.89 | 49% | 73.20 | – | – |
| host, 12 threads, Dense | Dense | 77.39 | 28.37 L | – | 0.81 | 3.52 | 3.56 | 43% | 8.39 | – | – |
| host, 1 thread, Classes | Classes | 56.55 | 7.73 B | – | 2.50 | 85.82 | 35.46 | 29% | 124.12 | – | – |
| host, 12 threads, Classes | Classes | 61.12 | 12.84 L | – | 0.80 | 8.37 | 3.55 | 27% | 13.15 | – | – |
| host, 1 thread, Rotation | Rotation | 58.48 | 8.88 B | – | 3.07 | 97.72 | 36.51 | 27% | 137.37 | – | – |
| host, 12 threads, Rotation | Rotation | 51.80 | 2.91 L | – | 0.82 | 9.70 | 3.53 | 24% | 14.51 | – | – |
| metal, tuned | Dense | 2939.73 | 0.53 L | 2864.86 | 0.58 | 0.93 | 0.85 | 24% | 2.08 | 35.3 | 4.0 |
| metal, Dense | Dense | 64.06 | 0.59 L | – | 0.64 | 0.98 | 0.89 | 23% | 2.09 | 35.0 | 4.0 |
| metal, Classes | Classes | 65.65 | 0.29 L | – | 0.94 | 1.12 | 0.94 | 21% | 2.06 | 60.1 | 6.4 |
| metal, Rotation | Rotation | 67.77 | 0.13 L | – | 0.79 | 1.93 | 0.99 | 19% | 3.52 | 39.1 | 4.1 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 2.613e-3 | 3.872e-3 | 1.446e-3 | 3.294e-4 | | | | | | yes | |
| host, 1 thread, Classes | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | | | | | | yes | |
| host, 1 thread, Rotation | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | | | | | | yes | |
| metal, tuned | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | 1.0000, 0.9999, 1.0000, 1.0000 | 1.35e-6, 1.35e-7 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 94.9 MB |
| metal, Dense | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | 1.0000, 0.9999, 1.0000, 1.0000 | 1.35e-6, 1.35e-7 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 95.1 MB |
| metal, Classes | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.50e-7, 1.35e-7 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 95.1 MB |
| metal, Rotation | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.00e-7, 1.35e-7 | 1 × 400000 / 1 × 1600000 | 34 | 1 | yes | 20.1 MB |

- metal, tuned: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(16 x 4 units, 4 columns per unit))); tuning 2.86 s: M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P plane (2 planes per cube) (tuned)

- metal, Dense: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(16 x 4 units, 4 columns per unit)))

- metal, Classes: Classes, run as dense on the device (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(16 x 4 units, 4 columns per unit)))

- metal, Rotation: Rotation (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)); M2L rotation 3 calls (cube (32 units)))

##### p = 6: 4096 leaves on 5 levels, points per leaf 7 / 24.4 / 43 (min / mean / max), lists U 93240 V 640584 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 12: the evaluation's speed-up over the host with the same strategy at 1 / 12 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 133.31 | 82.54 B | – | 8.96 | 167.61 | 49.50 | 22% | 227.25 | – | – |
| host, 12 threads, Dense | Dense | 78.51 | 29.86 L | – | 1.52 | 17.21 | 5.39 | 22% | 24.66 | – | – |
| host, 1 thread, Classes | Classes | 62.94 | 10.31 B | – | 8.69 | 331.30 | 49.13 | 13% | 389.54 | – | – |
| host, 12 threads, Classes | Classes | 61.55 | 13.24 L | – | 1.43 | 30.05 | 4.91 | 13% | 36.96 | – | – |
| host, 1 thread, Rotation | Rotation | 56.80 | 8.58 B | – | 9.68 | 301.42 | 48.94 | 14% | 360.34 | – | – |
| host, 12 threads, Rotation | Rotation | 51.19 | 3.03 L | – | 1.48 | 28.37 | 4.78 | 14% | 35.39 | – | – |
| metal, tuned | Dense | 3195.23 | 3.85 L | 3116.66 | 0.90 | 2.97 | 0.90 | 14% | 4.41 | 51.6 | 5.6 |
| metal, Dense | Dense | 70.72 | 4.05 L | – | 0.76 | 3.11 | 0.83 | 14% | 4.38 | 51.9 | 5.6 |
| metal, Classes | Classes | 68.77 | 0.77 L | – | 0.72 | 3.11 | 0.83 | 14% | 4.36 | 89.3 | 8.5 |
| metal, Rotation | Rotation | 65.25 | 0.34 L | – | 0.85 | 4.42 | 0.86 | 11% | 5.68 | 63.5 | 6.2 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.150e-4 | 2.757e-4 | 1.006e-4 | 3.202e-5 | | | | | | yes | |
| host, 1 thread, Classes | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | | | | | | yes | |
| host, 1 thread, Rotation | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | | | | | | yes | |
| metal, tuned | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | 1.0000, 0.9995, 1.0000, 1.0000 | 2.06e-6, 1.11e-7 | 1 × 400000 / 1 × 1600000 | 43 | 1 | yes | 158.7 MB |
| metal, Dense | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | 1.0000, 0.9995, 1.0000, 1.0000 | 2.06e-6, 1.11e-7 | 1 × 400000 / 1 × 1600000 | 43 | 1 | yes | 159.2 MB |
| metal, Classes | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | 1.0000, 1.0000, 1.0000, 1.0000 | 4.38e-7, 1.04e-7 | 1 × 400000 / 1 × 1600000 | 43 | 1 | yes | 159.2 MB |
| metal, Rotation | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.22e-7, 1.04e-7 | 1 × 400000 / 1 × 1600000 | 34 | 1 | yes | 22.8 MB |

- metal, tuned: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 3 calls in 4 chunks (0 library, 3 hand-written cube(32 x 2 units, 4 columns per unit))); tuning 3.12 s: M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P plane (2 planes per cube) (tuned)

- metal, Dense: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 3 calls in 4 chunks (0 library, 3 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Classes: Classes, run as dense on the device (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 3 calls in 4 chunks (0 library, 3 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Rotation: Rotation (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)); M2L rotation 3 calls (cube (64 units)))

##### p = 8: 4096 leaves on 5 levels, points per leaf 7 / 24.4 / 43 (min / mean / max), lists U 93240 V 640584 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 12: the evaluation's speed-up over the host with the same strategy at 1 / 12 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 311.22 | 262.27 B | – | 16.72 | 433.08 | 62.64 | 12% | 513.86 | – | – |
| host, 12 threads, Dense | Dense | 89.49 | 41.46 L | – | 2.43 | 41.76 | 6.32 | 12% | 50.93 | – | – |
| host, 1 thread, Classes | Classes | 75.17 | 21.09 B | – | 16.47 | 660.10 | 63.72 | 9% | 740.95 | – | – |
| host, 12 threads, Classes | Classes | 52.95 | 4.50 L | – | 2.16 | 60.76 | 5.95 | 9% | 69.47 | – | – |
| host, 1 thread, Rotation | Rotation | 60.98 | 10.73 B | – | 16.75 | 549.02 | 61.91 | 10% | 628.73 | – | – |
| host, 12 threads, Rotation | Rotation | 60.59 | 12.41 L | – | 2.04 | 48.73 | 5.89 | 10% | 58.59 | – | – |
| metal, tuned | Dense | 3959.10 | 10.52 L | 3873.93 | 0.91 | 5.10 | 0.86 | 10% | 6.71 | 76.6 | 7.6 |
| metal, Dense | Dense | 76.73 | 10.57 L | – | 0.88 | 6.06 | 0.86 | 9% | 7.03 | 73.1 | 7.2 |
| metal, Classes | Classes | 75.86 | 1.68 L | – | 0.88 | 5.60 | 0.87 | 10% | 6.89 | 107.5 | 10.1 |
| metal, Rotation | Rotation | 68.16 | 0.60 L | – | 0.88 | 7.55 | 0.96 | 9% | 8.88 | 70.8 | 6.6 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.827e-5 | 4.623e-5 | 1.910e-5 | 8.109e-6 | | | | | | yes | |
| host, 1 thread, Classes | 1.818e-5 | 4.610e-5 | 1.910e-5 | 8.109e-6 | | | | | | yes | |
| host, 1 thread, Rotation | 1.818e-5 | 4.609e-5 | 1.910e-5 | 8.109e-6 | | | | | | yes | |
| metal, tuned | 1.818e-5 | 4.616e-5 | 1.910e-5 | 8.109e-6 | 0.9952, 0.9986, 1.0000, 1.0000 | 2.24e-6, 1.13e-7 | 1 × 400000 / 1 × 1600000 | 46 | 1 | yes | 165.9 MB |
| metal, Dense | 1.818e-5 | 4.616e-5 | 1.910e-5 | 8.109e-6 | 0.9952, 0.9986, 1.0000, 1.0000 | 2.24e-6, 1.13e-7 | 1 × 400000 / 1 × 1600000 | 46 | 1 | yes | 166.3 MB |
| metal, Classes | 1.818e-5 | 4.617e-5 | 1.910e-5 | 8.109e-6 | 0.9997, 1.0015, 0.9999, 1.0000 | 4.78e-7, 1.03e-7 | 1 × 400000 / 1 × 1600000 | 46 | 1 | yes | 166.3 MB |
| metal, Rotation | 1.817e-5 | 4.610e-5 | 1.910e-5 | 8.109e-6 | 0.9999, 1.0002, 1.0000, 1.0000 | 3.37e-7, 1.03e-7 | 1 × 400000 / 1 × 1600000 | 34 | 1 | yes | 25.9 MB |

- metal, tuned: Dense (M2M 4 calls in 4 chunks (2 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 4 calls in 4 chunks (2 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 3 calls in 5 chunks (0 library, 3 hand-written cube(32 x 2 units, 4 columns per unit))); tuning 3.87 s: M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 4096 pairs: library (f32 inputs), box-major, 128 MB (tuned); M2M GEMM, ≤ 512 pairs: library (f32 inputs), box-major, 128 MB (tuned); M2M GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 128 MB (static rule); M2M GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 128 MB (static rule); L2L GEMM, ≤ 4096 pairs: library (f32 inputs), box-major, 128 MB (tuned); L2L GEMM, ≤ 512 pairs: library (f32 inputs), box-major, 128 MB (tuned); L2L GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 128 MB (static rule); L2L GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 128 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P plane (2 planes per cube) (tuned)

- metal, Dense: Dense (M2M 4 calls in 4 chunks (2 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 4 calls in 4 chunks (2 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 3 calls in 5 chunks (0 library, 3 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Classes: Classes, run as dense on the device (M2M 4 calls in 4 chunks (2 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 4 calls in 4 chunks (2 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 3 calls in 5 chunks (0 library, 3 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Rotation: Rotation (M2M 4 calls in 4 chunks (2 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 4 calls in 4 chunks (2 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L rotation 3 calls (cube (96 units)))

#### plummer, N = 100000, a Plummer sphere about the origin, scale a = 0.1, truncated at 10 a (f32), 64 per leaf

##### p = 3: 5678 leaves on 9 levels, points per leaf 0 / 17.6 / 64 (min / mean / max), lists U 132622 V 813714 W 36642 X 36642

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 12: the evaluation's speed-up over the host with the same strategy at 1 / 12 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 69.78 | 0.57 L | – | 2.67 | 78.23 | 164.30 | 67% | 245.74 | – | – |
| host, 12 threads, Dense | Dense | 69.85 | 0.58 L | – | 1.39 | 8.36 | 16.95 | 62% | 27.51 | – | – |
| host, 1 thread, Classes | Classes | 69.99 | 0.28 L | – | 2.70 | 147.99 | 166.26 | 52% | 317.40 | – | – |
| host, 12 threads, Classes | Classes | 69.57 | 0.27 L | – | 1.46 | 14.83 | 17.31 | 51% | 34.45 | – | – |
| host, 1 thread, Rotation | Rotation | 69.97 | 0.21 L | – | 3.37 | 156.24 | 164.33 | 51% | 324.32 | – | – |
| host, 12 threads, Rotation | Rotation | 69.27 | 0.20 L | – | 1.56 | 16.07 | 17.22 | 49% | 35.23 | – | – |
| metal, tuned | Dense | 4826.23 | 0.54 L | 4723.61 | 0.89 | 1.74 | 1.63 | 29% | 3.59 | 68.5 | 7.7 |
| metal, Dense | Dense | 93.12 | 0.59 L | – | 0.93 | 1.73 | 1.65 | 29% | 3.55 | 69.2 | 7.7 |
| metal, Classes | Classes | 93.41 | 0.27 L | – | 0.95 | 1.74 | 1.68 | 29% | 3.69 | 86.1 | 9.3 |
| metal, Rotation | Rotation | 93.37 | 0.15 L | – | 0.91 | 3.38 | 1.51 | 21% | 5.17 | 62.7 | 6.8 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | | | | | | yes | |
| host, 1 thread, Classes | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | | | | | | yes | |
| host, 1 thread, Rotation | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | | | | | | yes | |
| metal, tuned | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | 1.0000, 1.0001, 1.0000, 1.0000 | 1.53e-6, 1.31e-7 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 67.9 MB |
| metal, Dense | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | 1.0000, 1.0001, 1.0000, 1.0000 | 1.53e-6, 1.31e-7 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 67.9 MB |
| metal, Classes | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.81e-7, 1.24e-7 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 67.9 MB |
| metal, Rotation | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.49e-7, 1.24e-7 | 1 × 400000 / 1 × 1600000 | 87 | 1 | yes | 24.3 MB |

- metal, tuned: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 4 columns per unit))); tuning 4.72 s: M2L strategy: Dense, hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 524288 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 131072 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 2048 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 1024 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 2048 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 1024 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (64 units) (tuned)

- metal, Dense: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 4 columns per unit)))

- metal, Classes: Classes, run as dense on the device (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 4 columns per unit)))

- metal, Rotation: Rotation (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); M2L rotation 7 calls (cube (32 units)))

##### p = 6: 5678 leaves on 9 levels, points per leaf 0 / 17.6 / 64 (min / mean / max), lists U 132622 V 813714 W 36642 X 36642

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 12: the evaluation's speed-up over the host with the same strategy at 1 / 12 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 73.13 | 4.00 L | – | 9.34 | 325.24 | 390.53 | 54% | 725.81 | – | – |
| host, 12 threads, Dense | Dense | 73.68 | 4.09 L | – | 2.09 | 35.70 | 42.46 | 53% | 81.31 | – | – |
| host, 1 thread, Classes | Classes | 76.16 | 0.93 L | – | 9.44 | 532.59 | 389.75 | 42% | 932.35 | – | – |
| host, 12 threads, Classes | Classes | 69.94 | 0.76 L | – | 2.09 | 55.29 | 40.62 | 41% | 97.68 | – | – |
| host, 1 thread, Rotation | Rotation | 73.59 | 0.36 L | – | 10.60 | 495.31 | 393.23 | 44% | 904.13 | – | – |
| host, 12 threads, Rotation | Rotation | 70.12 | 0.35 L | – | 2.28 | 48.22 | 40.28 | 44% | 91.59 | – | – |
| metal, tuned | Dense | 5029.70 | 4.08 L | 4919.87 | 1.41 | 4.39 | 1.90 | 21% | 6.58 | 110.2 | 12.3 |
| metal, Dense | Dense | 96.86 | 3.99 L | – | 0.94 | 5.16 | 2.28 | 23% | 6.84 | 106.0 | 11.9 |
| metal, Classes | Classes | 97.45 | 0.78 L | – | 0.93 | 4.51 | 1.97 | 22% | 6.85 | 136.1 | 14.3 |
| metal, Rotation | Rotation | 93.91 | 0.37 L | – | 0.95 | 7.09 | 1.90 | 17% | 9.34 | 96.8 | 9.8 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | | | | | | yes | |
| host, 1 thread, Classes | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | | | | | | yes | |
| host, 1 thread, Rotation | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | | | | | | yes | |
| metal, tuned | 1.537e-4 | 2.354e-4 | 7.836e-5 | 2.985e-5 | 1.0000, 0.9999, 0.9999, 1.0000 | 2.46e-6, 1.39e-7 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 161.5 MB |
| metal, Dense | 1.537e-4 | 2.354e-4 | 7.836e-5 | 2.985e-5 | 1.0000, 0.9999, 0.9999, 1.0000 | 2.46e-6, 1.39e-7 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 161.8 MB |
| metal, Classes | 1.537e-4 | 2.354e-4 | 7.836e-5 | 2.985e-5 | 1.0000, 1.0001, 0.9999, 1.0000 | 4.56e-7, 1.31e-7 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 161.8 MB |
| metal, Rotation | 1.537e-4 | 2.354e-4 | 7.836e-5 | 2.985e-5 | 1.0000, 1.0000, 0.9999, 1.0000 | 3.84e-7, 1.31e-7 | 1 × 400000 / 1 × 1600000 | 87 | 1 | yes | 27.0 MB |

- metal, tuned: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(32 x 2 units, 4 columns per unit))); tuning 4.92 s: M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 524288 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 262144 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 131072 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 2048 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 1024 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 2048 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 1024 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (64 units) (tuned)

- metal, Dense: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Classes: Classes, run as dense on the device (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Rotation: Rotation (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 4 columns per unit)); M2L rotation 7 calls (cube (64 units)))

##### p = 8: 5678 leaves on 9 levels, points per leaf 0 / 17.6 / 64 (min / mean / max), lists U 132622 V 813714 W 36642 X 36642

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 12: the evaluation's speed-up over the host with the same strategy at 1 / 12 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 80.79 | 11.08 L | – | 17.69 | 749.78 | 598.53 | 44% | 1364.47 | – | – |
| host, 12 threads, Dense | Dense | 82.10 | 12.33 L | – | 3.12 | 76.41 | 60.26 | 43% | 140.03 | – | – |
| host, 1 thread, Classes | Classes | 77.66 | 1.77 L | – | 17.72 | 1036.21 | 600.46 | 36% | 1657.43 | – | – |
| host, 12 threads, Classes | Classes | 72.91 | 1.69 L | – | 3.29 | 104.93 | 57.77 | 35% | 166.89 | – | – |
| host, 1 thread, Rotation | Rotation | 72.00 | 0.68 L | – | 18.50 | 891.75 | 604.14 | 40% | 1516.08 | – | – |
| host, 12 threads, Rotation | Rotation | 70.89 | 0.69 L | – | 3.03 | 83.12 | 57.32 | 40% | 144.49 | – | – |
| metal, tuned | Dense | 6689.44 | 10.74 L | 6570.79 | 0.97 | 7.98 | 2.50 | 19% | 10.90 | 125.2 | 12.9 |
| metal, Dense | Dense | 106.33 | 10.70 L | – | 0.97 | 8.46 | 2.50 | 19% | 11.47 | 118.9 | 12.2 |
| metal, Classes | Classes | 106.02 | 1.71 L | – | 1.20 | 8.46 | 2.49 | 18% | 11.44 | 144.9 | 14.6 |
| metal, Rotation | Rotation | 95.42 | 0.61 L | – | 0.96 | 12.10 | 2.46 | 14% | 15.00 | 101.0 | 9.6 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 2.196e-5 | 3.885e-5 | 1.395e-5 | 5.182e-6 | | | | | | yes | |
| host, 1 thread, Classes | 2.187e-5 | 3.878e-5 | 1.395e-5 | 5.180e-6 | | | | | | yes | |
| host, 1 thread, Rotation | 2.187e-5 | 3.884e-5 | 1.395e-5 | 5.180e-6 | | | | | | yes | |
| metal, tuned | 2.189e-5 | 3.885e-5 | 1.394e-5 | 5.180e-6 | 0.9968, 1.0000, 0.9991, 0.9997 | 2.64e-6, 1.48e-7 | 1 × 400000 / 1 × 1600000 | 107 | 1 | yes | 171.7 MB |
| metal, Dense | 2.189e-5 | 3.885e-5 | 1.394e-5 | 5.180e-6 | 0.9968, 1.0000, 0.9991, 0.9997 | 2.64e-6, 1.48e-7 | 1 × 400000 / 1 × 1600000 | 107 | 1 | yes | 171.9 MB |
| metal, Classes | 2.188e-5 | 3.885e-5 | 1.394e-5 | 5.180e-6 | 1.0006, 1.0019, 0.9992, 1.0000 | 5.27e-7, 1.38e-7 | 1 × 400000 / 1 × 1600000 | 107 | 1 | yes | 171.9 MB |
| metal, Rotation | 2.187e-5 | 3.882e-5 | 1.394e-5 | 5.180e-6 | 1.0002, 0.9994, 0.9991, 0.9999 | 4.08e-7, 1.38e-7 | 1 × 400000 / 1 × 1600000 | 87 | 1 | yes | 30.2 MB |

- metal, tuned: Dense (M2M 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 7 calls in 9 chunks (0 library, 7 hand-written cube(32 x 2 units, 4 columns per unit))); tuning 6.57 s: M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 524288 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 262144 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 131072 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 4096 pairs: library (f32 inputs), box-major, 128 MB (tuned); M2M GEMM, ≤ 2048 pairs: library (f32 inputs), box-major, 128 MB (tuned); M2M GEMM, ≤ 1024 pairs: library (f32 inputs), box-major, 128 MB (tuned); M2M GEMM, ≤ 512 pairs: library (f32 inputs), box-major, 128 MB (tuned); M2M GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 128 MB (static rule); M2M GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 128 MB (static rule); L2L GEMM, ≤ 4096 pairs: library (f32 inputs), box-major, 128 MB (tuned); L2L GEMM, ≤ 2048 pairs: library (f32 inputs), box-major, 128 MB (tuned); L2L GEMM, ≤ 1024 pairs: library (f32 inputs), box-major, 128 MB (tuned); L2L GEMM, ≤ 512 pairs: library (f32 inputs), box-major, 128 MB (tuned); L2L GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 128 MB (static rule); L2L GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 128 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (64 units) (tuned)

- metal, Dense: Dense (M2M 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 7 calls in 9 chunks (0 library, 7 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Classes: Classes, run as dense on the device (M2M 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 7 calls in 9 chunks (0 library, 7 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Rotation: Rotation (M2M 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L rotation 7 calls (cube (96 units)))

#### cube, N = 1000000, uniform in the cube [-1, 1)^3 (f32), 64 per leaf

##### p = 3: 32768 leaves on 6 levels, points per leaf 11 / 30.5 / 57 (min / mean / max), lists U 797816 V 6039504 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 12: the evaluation's speed-up over the host with the same strategy at 1 / 12 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 638.21 | 0.60 L | – | 24.36 | 320.64 | 427.99 | 55% | 778.07 | – | – |
| host, 12 threads, Dense | Dense | 640.81 | 0.58 L | – | 3.03 | 28.95 | 38.53 | 51% | 75.20 | – | – |
| host, 1 thread, Classes | Classes | 632.18 | 0.26 L | – | 24.33 | 810.22 | 430.91 | 34% | 1270.92 | – | – |
| host, 12 threads, Classes | Classes | 643.23 | 0.25 L | – | 2.94 | 71.60 | 38.25 | 33% | 117.56 | – | – |
| host, 1 thread, Rotation | Rotation | 630.23 | 0.25 L | – | 27.99 | 900.18 | 427.62 | 31% | 1360.22 | – | – |
| host, 12 threads, Rotation | Rotation | 632.14 | 0.19 L | – | 3.41 | 84.02 | 38.11 | 29% | 130.40 | – | – |
| metal, tuned | Dense | 4829.39 | 0.58 L | 4060.79 | 0.91 | 5.80 | 4.86 | 28% | 15.39 | 50.6 | 4.9 |
| metal, Dense | Dense | 751.58 | 0.58 L | – | 0.91 | 5.88 | 5.26 | 30% | 15.98 | 48.7 | 4.7 |
| metal, Classes | Classes | 760.40 | 0.26 L | – | 0.90 | 5.88 | 5.24 | 29% | 16.01 | 79.4 | 7.3 |
| metal, Rotation | Rotation | 747.22 | 0.21 L | – | 0.88 | 10.45 | 5.17 | 23% | 20.58 | 66.1 | 6.3 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | | | | | | yes | |
| host, 1 thread, Classes | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | | | | | | yes | |
| host, 1 thread, Rotation | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | | | | | | yes | |
| metal, tuned | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.69e-6, 1.62e-7 | 1 × 4000000 / 1 × 16000000 | 64 | 1 | yes | 322.3 MB |
| metal, Dense | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.69e-6, 1.62e-7 | 1 × 4000000 / 1 × 16000000 | 64 | 1 | yes | 324.4 MB |
| metal, Classes | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.75e-7, 1.61e-7 | 1 × 4000000 / 1 × 16000000 | 64 | 1 | yes | 324.4 MB |
| metal, Rotation | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.29e-7, 1.61e-7 | 1 × 4000000 / 1 × 16000000 | 41 | 1 | yes | 189.7 MB |

- metal, tuned: Dense (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 4 calls in 9 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit))); tuning 4.06 s: M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 128 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 2 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 128 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned)

- metal, Dense: Dense (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 4 calls in 9 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)))

- metal, Classes: Classes, run as dense on the device (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 4 calls in 9 chunks (0 library, 4 hand-written cube(16 x 4 units, 4 columns per unit)))

- metal, Rotation: Rotation (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 4 columns per unit)); M2L rotation 4 calls (cube (32 units)))

##### p = 6: 32768 leaves on 6 levels, points per leaf 11 / 30.5 / 57 (min / mean / max), lists U 797816 V 6039504 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 12: the evaluation's speed-up over the host with the same strategy at 1 / 12 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 634.21 | 4.28 L | – | 84.40 | 1572.74 | 559.50 | 25% | 2223.60 | – | – |
| host, 12 threads, Dense | Dense | 642.63 | 4.19 L | – | 8.58 | 150.92 | 53.45 | 25% | 217.73 | – | – |
| host, 1 thread, Classes | Classes | 637.61 | 0.82 L | – | 84.69 | 3114.96 | 561.56 | 15% | 3763.02 | – | – |
| host, 12 threads, Classes | Classes | 631.80 | 0.78 L | – | 8.70 | 393.95 | 50.37 | 11% | 457.18 | – | – |
| host, 1 thread, Rotation | Rotation | 634.67 | 0.37 L | – | 95.05 | 2810.95 | 560.68 | 16% | 3472.18 | – | – |
| host, 12 threads, Rotation | Rotation | 634.79 | 0.40 L | – | 9.69 | 257.51 | 50.39 | 16% | 323.58 | – | – |
| metal, tuned | Rotation | 4326.05 | 0.32 L | 3555.16 | 1.22 | 31.24 | 4.85 | 11% | 41.75 | 53.3 | 5.2 |
| metal, Dense | Dense | 763.27 | 4.07 L | – | 1.22 | 31.12 | 5.30 | 12% | 41.82 | 53.2 | 5.2 |
| metal, Classes | Classes | 766.54 | 0.81 L | – | 1.22 | 31.14 | 5.28 | 12% | 41.60 | 90.5 | 11.0 |
| metal, Rotation | Rotation | 747.28 | 0.38 L | – | 1.22 | 31.28 | 5.47 | 13% | 42.11 | 82.5 | 7.7 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.212e-4 | 3.101e-4 | 2.099e-4 | 9.303e-5 | | | | | | yes | |
| host, 1 thread, Classes | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | | | | | | yes | |
| host, 1 thread, Rotation | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | | | | | | yes | |
| metal, tuned | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | 1.0000, 1.0001, 1.0000, 1.0000 | 3.51e-7, 1.35e-7 | 1 × 4000000 / 1 × 16000000 | 41 | 1 | yes | 208.5 MB |
| metal, Dense | 1.212e-4 | 3.101e-4 | 2.099e-4 | 9.304e-5 | 1.0000, 1.0002, 1.0000, 1.0000 | 2.64e-6, 1.39e-7 | 1 × 4000000 / 1 × 16000000 | 97 | 1 | yes | 341.7 MB |
| metal, Classes | 1.212e-4 | 3.101e-4 | 2.099e-4 | 9.304e-5 | 0.9999, 0.9999, 1.0000, 1.0000 | 4.77e-7, 1.35e-7 | 1 × 4000000 / 1 × 16000000 | 97 | 1 | yes | 341.7 MB |
| metal, Rotation | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | 1.0000, 1.0001, 1.0000, 1.0000 | 3.51e-7, 1.35e-7 | 1 × 4000000 / 1 × 16000000 | 41 | 1 | yes | 208.5 MB |

- metal, tuned: Rotation (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 4 columns per unit)); M2L rotation 4 calls (cube (64 units))); tuning 3.56 s: M2L strategy: Rotation (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 2 units, 2 columns per unit), box-major, 128 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned)

- metal, Dense: Dense (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 4 calls in 20 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Classes: Classes, run as dense on the device (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 4 calls in 20 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Rotation: Rotation (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 4 columns per unit)); M2L rotation 4 calls (cube (64 units)))

##### p = 8: 32768 leaves on 6 levels, points per leaf 11 / 30.5 / 57 (min / mean / max), lists U 797816 V 6039504 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 12: the evaluation's speed-up over the host with the same strategy at 1 / 12 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 649.65 | 10.67 L | – | 158.23 | 4061.11 | 700.84 | 14% | 4925.75 | – | – |
| host, 12 threads, Dense | Dense | 652.15 | 11.06 L | – | 15.56 | 410.71 | 66.36 | 13% | 498.18 | – | – |
| host, 1 thread, Classes | Classes | 644.73 | 1.75 L | – | 160.68 | 6240.33 | 694.98 | 10% | 7097.29 | – | – |
| host, 12 threads, Classes | Classes | 638.47 | 1.73 L | – | 14.81 | 627.16 | 66.23 | 9% | 713.70 | – | – |
| host, 1 thread, Rotation | Rotation | 628.53 | 0.66 L | – | 163.53 | 5175.02 | 698.76 | 12% | 6042.26 | – | – |
| host, 12 threads, Rotation | Rotation | 621.31 | 0.66 L | – | 14.98 | 448.15 | 61.61 | 12% | 533.31 | – | – |
| metal, tuned | Rotation | 6668.66 | 0.66 L | 5913.52 | 1.88 | 62.28 | 5.07 | 7% | 73.49 | 67.0 | 6.8 |
| metal, Dense | Dense | 762.19 | 10.63 L | – | 1.87 | 70.09 | 5.50 | 7% | 81.94 | 60.1 | 6.1 |
| metal, Classes | Classes | 759.31 | 1.79 L | – | 1.88 | 70.00 | 5.52 | 7% | 81.52 | 87.1 | 8.8 |
| metal, Rotation | Rotation | 743.59 | 0.65 L | – | 1.87 | 62.25 | 5.45 | 7% | 74.18 | 81.4 | 7.2 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.863e-5 | 5.282e-5 | 4.026e-5 | 2.305e-5 | | | | | | yes | |
| host, 1 thread, Classes | 1.846e-5 | 5.275e-5 | 4.026e-5 | 2.305e-5 | | | | | | yes | |
| host, 1 thread, Rotation | 1.845e-5 | 5.272e-5 | 4.026e-5 | 2.305e-5 | | | | | | yes | |
| metal, tuned | 1.845e-5 | 5.279e-5 | 4.026e-5 | 2.305e-5 | 1.0000, 1.0014, 1.0000, 0.9999 | 3.87e-7, 1.77e-7 | 1 × 4000000 / 1 × 16000000 | 41 | 1 | yes | 227.7 MB |
| metal, Dense | 1.846e-5 | 5.277e-5 | 4.026e-5 | 2.305e-5 | 0.9910, 0.9991, 0.9999, 0.9999 | 2.92e-6, 1.88e-7 | 1 × 4000000 / 1 × 16000000 | 133 | 1 | yes | 357.5 MB |
| metal, Classes | 1.846e-5 | 5.277e-5 | 4.026e-5 | 2.305e-5 | 1.0003, 1.0003, 1.0000, 1.0000 | 5.16e-7, 1.77e-7 | 1 × 4000000 / 1 × 16000000 | 133 | 1 | yes | 357.5 MB |
| metal, Rotation | 1.845e-5 | 5.279e-5 | 4.026e-5 | 2.305e-5 | 1.0000, 1.0014, 1.0000, 0.9999 | 3.87e-7, 1.77e-7 | 1 × 4000000 / 1 × 16000000 | 41 | 1 | yes | 227.7 MB |

- metal, tuned: Rotation (M2M 5 calls in 5 chunks (3 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 5 calls in 5 chunks (3 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L rotation 4 calls (cube (96 units))); tuning 5.91 s: M2L strategy: Rotation (tuned); M2M GEMM, ≤ 32768 pairs: library (f32 inputs), box-major, 128 MB (tuned); M2M GEMM, ≤ 4096 pairs: library (f32 inputs), box-major, 128 MB (tuned); M2M GEMM, ≤ 512 pairs: library (f32 inputs), box-major, 128 MB (tuned); M2M GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 128 MB (static rule); M2M GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 128 MB (static rule); L2L GEMM, ≤ 32768 pairs: library (f32 inputs), box-major, 128 MB (tuned); L2L GEMM, ≤ 4096 pairs: library (f32 inputs), box-major, 128 MB (tuned); L2L GEMM, ≤ 512 pairs: library (f32 inputs), box-major, 128 MB (tuned); L2L GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 128 MB (static rule); L2L GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 128 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned)

- metal, Dense: Dense (M2M 5 calls in 5 chunks (3 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 5 calls in 5 chunks (3 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 4 calls in 32 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Classes: Classes, run as dense on the device (M2M 5 calls in 5 chunks (3 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 5 calls in 5 chunks (3 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 4 calls in 32 chunks (0 library, 4 hand-written cube(32 x 2 units, 4 columns per unit)))

- metal, Rotation: Rotation (M2M 5 calls in 5 chunks (3 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 5 calls in 5 chunks (3 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L rotation 4 calls (cube (96 units)))

#### Summary: evaluation time

Median ms. Host: its default (`Auto`, `P2pChoice::Auto`). Device: tuned, and the fastest of its rows.

| problem | p | host 1 thread | host 12 threads | device tuned | device fastest | tuned x 1 | tuned x 12 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| cube, N = 100000 | 3 | 73.20 | 8.39 | 2.08 | 2.06 (metal, Classes) | 35.3 | 4.0 |
| cube, N = 100000 | 6 | 227.25 | 24.66 | 4.41 | 4.36 (metal, Classes) | 51.6 | 5.6 |
| cube, N = 100000 | 8 | 513.86 | 50.93 | 6.71 | 6.71 (metal, tuned) | 76.6 | 7.6 |
| plummer, N = 100000 | 3 | 245.74 | 27.51 | 3.59 | 3.55 (metal, Dense) | 68.5 | 7.7 |
| plummer, N = 100000 | 6 | 725.81 | 81.31 | 6.58 | 6.58 (metal, tuned) | 110.2 | 12.3 |
| plummer, N = 100000 | 8 | 1364.47 | 140.03 | 10.90 | 10.90 (metal, tuned) | 125.2 | 12.9 |
| cube, N = 1000000 | 3 | 778.07 | 75.20 | 15.39 | 15.39 (metal, tuned) | 50.6 | 4.9 |
| cube, N = 1000000 | 6 | 2223.60 | 217.73 | 41.75 | 41.60 (metal, Classes) | 53.3 | 5.2 |
| cube, N = 1000000 | 8 | 4925.75 | 498.18 | 73.49 | 73.49 (metal, tuned) | 67.0 | 6.8 |

Backends run: metal (f32); host (Apple M3 Max); not run: cpu, cuda (type-checked, not run).

</details>

### `device_fmm --part leaf`

<details><summary>Output</summary>

### The device FMM against the host path (Phase 4 T13)

| item | value |
| --- | --- |
| machine | Apple M3 Max; 16 physical, 16 logical, 12 performance; aarch64-macos, release build; rustc 1.99.0 (b940084d7 2026-09-28) |
| device | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| BLAS threads | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1 |
| precision | f32 |
| host | `P2pChoice::Auto` at 1 and 12 threads; one rank |
| timing | evaluate: median wall time of 5 evaluations of the first charge vector after the eight error evaluations (compilation excluded; one sync per device evaluation); stages: on the device the medians of a synchronous-stages build (a sync after each stage), on the host the medians of the timed evaluations |
| tuning | no tuning cache: no tuned rows |
| tables | through the table cache `DIR/tables` |
| wall time of every run | 25 s |

#### Device leaf-size study (metal, f32, static rule)

Times in ms: stages from the synchronous-stages build; far = upward + downward; near share = leaves / sum of the stages; evaluate = the median of the default build.

| distribution | p | max points per leaf | leaves | points per leaf (min / mean / max) | V pairs | φ L2 | ∇φ L2 | far | leaves | near share | evaluate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 3 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 2.769e-3 | 2.232e-3 | 6.57 | 1.89 | 18% | 7.81 |
| cube | 3 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 2.616e-3 | 1.489e-3 | 2.42 | 1.54 | 27% | 2.45 |
| cube | 3 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 1.97 | 0.87 | 20% | 2.18 |
| cube | 3 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 1.65 | 0.87 | 22% | 2.21 |
| cube | 3 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 2.295e-3 | 8.343e-4 | 1.05 | 2.97 | 57% | 3.65 |
| cube | 8 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 1.893e-5 | 2.805e-5 | 64.23 | 2.22 | 3% | 66.32 |
| cube | 8 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 1.816e-5 | 1.957e-5 | 7.18 | 1.24 | 13% | 8.03 |
| cube | 8 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 1.818e-5 | 1.910e-5 | 6.49 | 0.88 | 10% | 6.86 |
| cube | 8 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 1.818e-5 | 1.910e-5 | 6.86 | 0.89 | 10% | 7.03 |
| cube | 8 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 1.604e-5 | 1.181e-5 | 1.76 | 2.98 | 50% | 4.28 |
| plummer | 3 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 3.306e-3 | 1.545e-3 | 4.96 | 2.09 | 24% | 6.66 |
| plummer | 3 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 3.260e-3 | 1.377e-3 | 3.33 | 1.74 | 27% | 4.40 |
| plummer | 3 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 3.202e-3 | 1.116e-3 | 2.89 | 1.66 | 28% | 3.61 |
| plummer | 3 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 3.160e-3 | 1.008e-3 | 2.29 | 2.49 | 40% | 3.95 |
| plummer | 3 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 3.072e-3 | 8.782e-4 | 2.34 | 4.44 | 55% | 6.14 |
| plummer | 8 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 2.269e-5 | 2.051e-5 | 33.08 | 3.44 | 9% | 36.13 |
| plummer | 8 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 2.237e-5 | 1.868e-5 | 16.98 | 2.71 | 13% | 19.41 |
| plummer | 8 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 2.189e-5 | 1.394e-5 | 9.48 | 2.46 | 18% | 11.40 |
| plummer | 8 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 2.143e-5 | 1.298e-5 | 6.39 | 3.37 | 30% | 9.30 |
| plummer | 8 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 2.093e-5 | 1.085e-5 | 5.36 | 5.95 | 46% | 10.73 |

##### Fastest leaf size per configuration

| distribution | p | fastest | evaluate (ms) | at 64 (ms) | 64 / fastest | evaluate at 16, 32, 64, 128, 256 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cube | 3 | 64 | 2.18 | 2.18 | 1.00 | 7.81, 2.45, 2.18, 2.21, 3.65 |
| cube | 8 | 256 | 4.28 | 6.86 | 1.60 | 66.32, 8.03, 6.86, 7.03, 4.28 |
| plummer | 3 | 64 | 3.61 | 3.61 | 1.00 | 6.66, 4.40, 3.61, 3.95, 6.14 |
| plummer | 8 | 128 | 9.30 | 11.40 | 1.23 | 36.13, 19.41, 11.40, 9.30, 10.73 |

##### The device leaf-size rule (T13)

Geometric mean of the metal evaluation time over the 4 configurations (cube and Plummer, p = [3, 8], f32). The fastest size replaces 64 only if it is at least 5% faster by this measure and no φ or ∇φ L2 error at it is more than 10% worse than at 64.

| max points per leaf | geometric mean (ms) | relative to 64 |
| ---: | ---: | ---: |
| 16 | 18.79 | 3.774 |
| 32 | 6.40 | 1.286 |
| 64 | 4.98 | 1.000 |
| 128 | 4.89 | 0.981 |
| 256 | 5.66 | 1.137 |

Fastest: 128 (1.019x faster than 64); largest error growth there: 1.000x. **The rule picks 64.**

Backends run: metal (f32); host (Apple M3 Max); not run: cpu, cuda (type-checked, not run).

</details>

### `fmm_accuracy --backend metal --distribution plummer` (T11's harness)

<details><summary>Output</summary>

### Accuracy of the FMM on uniform and adaptive trees (C3.2, C3.3)

- Problems: N = 100000 points per distribution, sources equal to targets; 8 charge vectors uniform in [-1, 1) (seeds 0xc32 + 1 + k for the cube, 0xc33 + 1 + k for the others); the domain from `compute_global_bounding_box`; the default M2L strategy; gradients on. The cube has a uniform level-4 tree (`max_level` 4, `max_points_per_leaf` 1); the others adaptive trees (`max_level` 16, `max_points_per_leaf` 64, the same for every distribution).
- Error measure: relative L2 and max error of φ and of ∇φ (Euclidean norm per target) at 1000 targets sampled with the seed of the points, against `direct_sum` in f64 over all sources divided by 4π; for each charge vector, then the root mean square over the vectors. "range" is the smallest and largest φ L2 error of one vector. f32 runs use the charges rounded to f32, and the oracle on the rounded charges.
- C3.2: the single-translation φ L2 error of P2M → M2L → L2P (design §7; p = 18 re-derived in T9 as the median over 33 source draws, docs/phase3/README.md, "Predictions"), 1.77e-3 at p = 3, 1.08e-5 at p = 8, 6.74e-9 at p = 18; the gate allows twice that, on the cube.
- C3.3: the φ L2 error of each clustered distribution at most twice that of the uniform cube at the same p and N, in f64; f32 is reported next to it.
- Machine: Apple M3 Max; 16 physical, 16 logical; aarch64-macos, release build; one rank, 1 thread. The direct sums took 6.1 (cube), 6.0 (plummer) s (one thread).
- Threading: rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- P2P kernel: neon (`--p2p auto`).
- Backend: metal (`--backend metal`), device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32. Kinds on the device: all eight; stage timing: host clock, enqueue time only. The device does no f64 arithmetic: f64 rows are left out.

#### C3.3 gate

| distribution | precision | p | φ L2 | cube φ L2 | ratio | within 2× | W pairs | X pairs |
|---|---|---:|---:|---:|---:|---|---:|---:|
| plummer | f32 | 3 | 3.202e-3 | 2.650e-3 | 1.21 | (f32, reported) | 36642 | 36642 |
| plummer | f32 | 8 | 2.189e-5 | 1.778e-5 | 1.23 | (f32, reported) | 36642 | 36642 |

#### plummer

a Plummer sphere about the origin, scale a = 0.1, truncated at 10 a; N = 100000, `max_level` 16, `max_points_per_leaf` 64, seed 0xc33.

##### Tree

| levels | leaves | leaf levels (min – max) | points per leaf (min / mean / max) | U pairs | V pairs | W pairs | X pairs |
|---:|---:|---|---:|---:|---:|---:|---:|
| 9 | 5678 | 3 – 8 | 0 / 17.6 / 64 | 132622 | 813714 | 36642 | 36642 |

Leaves per level: 3: 448, 4: 395, 5: 733, 6: 1322, 7: 2364, 8: 416.

##### Accuracy

| precision | p | strategy | φ L2 | φ L2 range | φ max | ∇φ L2 | ∇φ max | prediction | φ L2 / prediction | cube φ L2 | φ L2 / cube |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| f32 | 3 | Dense | 3.202e-3 | 1.02e-3 – 4.86e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | 1.77e-3 | 1.81 | 2.650e-3 | 1.21 |
| f32 | 8 | Dense | 2.189e-5 | 6.27e-6 – 3.32e-5 | 3.885e-5 | 1.394e-5 | 5.180e-6 | 1.08e-5 | 2.03 | 1.778e-5 | 1.23 |

##### Device

Per evaluation (the last one of each row): transfers, kernel launches, syncs (waits for the device) and timing windows.

| precision | p | strategy on the device | kinds on the device | GEMMs | uploads | upload bytes | downloads | download bytes | launches | syncs | windows |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| f32 | 3 | Dense | 8 | M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 4 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 4 columns per unit)) | 1 | 400000 | 1 | 1600000 | 101 | 1 | 0 |
| f32 | 8 | Dense | 8 | M2M 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); L2L 8 calls in 8 chunks (6 library, 2 hand-written cube(32 x 2 units, 4 columns per unit)); M2L 7 calls in 9 chunks (0 library, 7 hand-written cube(32 x 2 units, 4 columns per unit)) | 1 | 400000 | 1 | 1600000 | 107 | 1 | 0 |

##### Timings

Wall time in ms. Build: once per row. Evaluate: the mean over the 8 charge vectors, by evaluator stage.

| precision | p | build | octree | plan | tables | evaluator | load | evaluate | exchange sources | upward local | upward global | exchange multipoles | downward | leaves | output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f32 | 3 | 103.8 | 8.7 | 48.5 | 2.9 | 4.8 | 1.2 | 12.8 | 0.0 | 0.1 | 0.1 | 0.0 | 0.4 | 3.0 | 9.1 |
| f32 | 8 | 414.4 | 8.8 | 46.6 | 241.4 | 4.4 | 1.2 | 36.1 | 0.0 | 0.1 | 0.1 | 0.0 | 0.3 | 7.1 | 28.4 |

</details>

### `p2p_kernels --device metal --fmm`

<details><summary>Output</summary>

### The device P2P against nd-fmm-simd (Phase 4 T6, C4.2)

| item | value |
| --- | --- |
| machine | Apple M3 Max (16 physical, 16 logical) |
| performance cores | 12 |
| target | aarch64-macos, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| device | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| host kernel | nd-fmm-simd on neon (detected) |
| W1 leaves per launch | 4096 (one-thread rows: 1024) |
| threads of the all-cores rows | 12 |
| timing | median of 15 batches of >= 20 ms; device launches queued between syncs, a warm-up launch first (compilation excluded) |

#### The GPU layouts, f32

Model: the C4.2 peak model of device-path.md §13.4, 720 Gpairs/s (φ) and 480 (φ, ∇φ), from 40 cores × 128 lanes × 1.4 GHz and 10 or 15 operations per pair. Speed-ups over nd-fmm-simd on the same cell at 1 and 12 threads. W1 rows launch the stated leaves per launch; W2 rows split the targets into rows of one tile.

##### potential

| cell | layout | Gpairs/s | of model | x simd 1 thread | x simd 12 threads | max φ | max ∇φ | check |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W1 n_t=8 per-pair | cube (64 units) | 33.954 | 4.7% | 10.7 | 0.93 | 8.8e-8 | – | pass |
| W1 n_t=8 per-pair | cube (32 units) | 48.040 | 6.7% | 15.2 | 1.31 | 8.8e-8 | – | pass |
| W1 n_t=8 per-pair | cube (128 units) | 18.938 | 2.6% | 6.0 | 0.52 | 8.8e-8 | – | pass |
| W1 n_t=8 per-pair | plane (2 planes per cube) | 47.998 | 6.7% | 15.2 | 1.31 | 8.8e-8 | – | pass |
| W1 n_t=8 per-pair | plane (4 planes per cube) | 48.385 | 6.7% | 15.3 | 1.32 | 8.8e-8 | – | pass |
| W1 n_t=8 gathered | cube (64 units) | 61.896 | 8.6% | 16.1 | 1.40 | 8.8e-8 | – | pass |
| W1 n_t=8 gathered | cube (32 units) | 74.427 | 10.3% | 19.3 | 1.68 | 8.8e-8 | – | pass |
| W1 n_t=8 gathered | cube (128 units) | 36.492 | 5.1% | 9.5 | 0.82 | 8.8e-8 | – | pass |
| W1 n_t=8 gathered | plane (2 planes per cube) | 73.789 | 10.2% | 19.2 | 1.66 | 8.8e-8 | – | pass |
| W1 n_t=8 gathered | plane (4 planes per cube) | 74.537 | 10.4% | 19.4 | 1.68 | 8.8e-8 | – | pass |
| W1 n_t=16 per-pair | cube (64 units) | 84.759 | 11.8% | 22.4 | 1.97 | 7.0e-8 | – | pass |
| W1 n_t=16 per-pair | cube (32 units) | 123.311 | 17.1% | 32.6 | 2.86 | 7.0e-8 | – | pass |
| W1 n_t=16 per-pair | cube (128 units) | 49.563 | 6.9% | 13.1 | 1.15 | 7.0e-8 | – | pass |
| W1 n_t=16 per-pair | plane (2 planes per cube) | 121.020 | 16.8% | 32.0 | 2.81 | 7.0e-8 | – | pass |
| W1 n_t=16 per-pair | plane (4 planes per cube) | 121.244 | 16.8% | 32.1 | 2.81 | 7.0e-8 | – | pass |
| W1 n_t=16 gathered | cube (64 units) | 122.860 | 17.1% | 31.1 | 2.88 | 7.0e-8 | – | pass |
| W1 n_t=16 gathered | cube (32 units) | 154.007 | 21.4% | 39.0 | 3.62 | 7.0e-8 | – | pass |
| W1 n_t=16 gathered | cube (128 units) | 74.173 | 10.3% | 18.8 | 1.74 | 7.0e-8 | – | pass |
| W1 n_t=16 gathered | plane (2 planes per cube) | 151.008 | 21.0% | 38.3 | 3.55 | 7.0e-8 | – | pass |
| W1 n_t=16 gathered | plane (4 planes per cube) | 152.873 | 21.2% | 38.8 | 3.59 | 7.0e-8 | – | pass |
| W1 n_t=24 per-pair | cube (64 units) | 138.166 | 19.2% | 35.6 | 3.43 | 8.8e-8 | – | pass |
| W1 n_t=24 per-pair | cube (32 units) | 202.179 | 28.1% | 52.1 | 5.02 | 8.8e-8 | – | pass |
| W1 n_t=24 per-pair | cube (128 units) | 80.132 | 11.1% | 20.6 | 1.99 | 8.8e-8 | – | pass |
| W1 n_t=24 per-pair | plane (2 planes per cube) | 197.810 | 27.5% | 51.0 | 4.91 | 8.8e-8 | – | pass |
| W1 n_t=24 per-pair | plane (4 planes per cube) | 197.676 | 27.5% | 50.9 | 4.91 | 8.8e-8 | – | pass |
| W1 n_t=24 gathered | cube (64 units) | 178.489 | 24.8% | 45.1 | 4.42 | 8.8e-8 | – | pass |
| W1 n_t=24 gathered | cube (32 units) | 233.232 | 32.4% | 58.9 | 5.78 | 8.8e-8 | – | pass |
| W1 n_t=24 gathered | cube (128 units) | 110.087 | 15.3% | 27.8 | 2.73 | 8.8e-8 | – | pass |
| W1 n_t=24 gathered | plane (2 planes per cube) | 229.518 | 31.9% | 57.9 | 5.68 | 8.8e-8 | – | pass |
| W1 n_t=24 gathered | plane (4 planes per cube) | 229.288 | 31.8% | 57.9 | 5.68 | 8.8e-8 | – | pass |
| W1 n_t=32 per-pair | cube (64 units) | 198.766 | 27.6% | 50.8 | 5.24 | 5.8e-8 | – | pass |
| W1 n_t=32 per-pair | cube (32 units) | 289.179 | 40.2% | 73.9 | 7.62 | 5.8e-8 | – | pass |
| W1 n_t=32 per-pair | cube (128 units) | 113.434 | 15.8% | 29.0 | 2.99 | 5.8e-8 | – | pass |
| W1 n_t=32 per-pair | plane (2 planes per cube) | 280.693 | 39.0% | 71.8 | 7.40 | 5.8e-8 | – | pass |
| W1 n_t=32 per-pair | plane (4 planes per cube) | 283.853 | 39.4% | 72.6 | 7.48 | 5.8e-8 | – | pass |
| W1 n_t=32 gathered | cube (64 units) | 235.678 | 32.7% | 59.4 | 6.29 | 5.8e-8 | – | pass |
| W1 n_t=32 gathered | cube (32 units) | 315.303 | 43.8% | 79.4 | 8.42 | 5.8e-8 | – | pass |
| W1 n_t=32 gathered | cube (128 units) | 143.404 | 19.9% | 36.1 | 3.83 | 5.8e-8 | – | pass |
| W1 n_t=32 gathered | plane (2 planes per cube) | 307.047 | 42.6% | 77.4 | 8.20 | 5.8e-8 | – | pass |
| W1 n_t=32 gathered | plane (4 planes per cube) | 304.956 | 42.4% | 76.8 | 8.14 | 5.8e-8 | – | pass |
| W1 n_t=64 per-pair | cube (64 units) | 337.153 | 46.8% | 86.3 | 9.48 | 1.2e-7 | – | pass |
| W1 n_t=64 per-pair | cube (32 units) | 308.515 | 42.8% | 78.9 | 8.68 | 1.2e-7 | – | pass |
| W1 n_t=64 per-pair | cube (128 units) | 221.775 | 30.8% | 56.8 | 6.24 | 1.2e-7 | – | pass |
| W1 n_t=64 per-pair | plane (2 planes per cube) | 296.119 | 41.1% | 75.8 | 8.33 | 1.2e-7 | – | pass |
| W1 n_t=64 per-pair | plane (4 planes per cube) | 300.170 | 41.7% | 76.8 | 8.44 | 1.2e-7 | – | pass |
| W1 n_t=64 gathered | cube (64 units) | 351.701 | 48.8% | 90.8 | 10.08 | 1.2e-7 | – | pass |
| W1 n_t=64 gathered | cube (32 units) | 320.829 | 44.6% | 82.8 | 9.19 | 1.2e-7 | – | pass |
| W1 n_t=64 gathered | cube (128 units) | 256.189 | 35.6% | 66.1 | 7.34 | 1.2e-7 | – | pass |
| W1 n_t=64 gathered | plane (2 planes per cube) | 310.514 | 43.1% | 80.2 | 8.90 | 1.2e-7 | – | pass |
| W1 n_t=64 gathered | plane (4 planes per cube) | 308.909 | 42.9% | 79.8 | 8.85 | 1.2e-7 | – | pass |
| W1 n_t=128 per-pair | cube (64 units) | 346.201 | 48.1% | 92.1 | 10.13 | 1.2e-7 | – | pass |
| W1 n_t=128 per-pair | cube (32 units) | 319.077 | 44.3% | 84.9 | 9.33 | 1.2e-7 | – | pass |
| W1 n_t=128 per-pair | cube (128 units) | 360.961 | 50.1% | 96.0 | 10.56 | 1.2e-7 | – | pass |
| W1 n_t=128 per-pair | plane (2 planes per cube) | 305.124 | 42.4% | 81.2 | 8.93 | 1.2e-7 | – | pass |
| W1 n_t=128 per-pair | plane (4 planes per cube) | 307.567 | 42.7% | 81.8 | 9.00 | 1.2e-7 | – | pass |
| W1 n_t=128 gathered | cube (64 units) | 353.722 | 49.1% | 89.8 | 9.71 | 1.2e-7 | – | pass |
| W1 n_t=128 gathered | cube (32 units) | 321.795 | 44.7% | 81.7 | 8.83 | 1.2e-7 | – | pass |
| W1 n_t=128 gathered | cube (128 units) | 371.131 | 51.5% | 94.2 | 10.19 | 1.2e-7 | – | pass |
| W1 n_t=128 gathered | plane (2 planes per cube) | 308.559 | 42.9% | 78.3 | 8.47 | 1.2e-7 | – | pass |
| W1 n_t=128 gathered | plane (4 planes per cube) | 311.069 | 43.2% | 79.0 | 8.54 | 1.2e-7 | – | pass |
| W2 N=1000 t≠s | cube (64 units) | 12.610 | 1.8% | 3.2 | 0.33 | 1.2e-7 | – | pass |
| W2 N=1000 t≠s | cube (32 units) | 12.179 | 1.7% | 3.1 | 0.32 | 1.2e-7 | – | pass |
| W2 N=1000 t≠s | cube (128 units) | 12.996 | 1.8% | 3.3 | 0.34 | 1.2e-7 | – | pass |
| W2 N=1000 t≠s | plane (2 planes per cube) | 12.203 | 1.7% | 3.1 | 0.32 | 1.2e-7 | – | pass |
| W2 N=1000 t≠s | plane (4 planes per cube) | 12.186 | 1.7% | 3.1 | 0.32 | 1.2e-7 | – | pass |
| W2 N=1000 t=s | cube (64 units) | 12.570 | 1.7% | 3.2 | 0.30 | 1.4e-7 | – | pass |
| W2 N=1000 t=s | cube (32 units) | 12.168 | 1.7% | 3.1 | 0.29 | 1.4e-7 | – | pass |
| W2 N=1000 t=s | cube (128 units) | 12.822 | 1.8% | 3.2 | 0.31 | 1.4e-7 | – | pass |
| W2 N=1000 t=s | plane (2 planes per cube) | 12.198 | 1.7% | 3.1 | 0.29 | 1.4e-7 | – | pass |
| W2 N=1000 t=s | plane (4 planes per cube) | 12.220 | 1.7% | 3.1 | 0.29 | 1.4e-7 | – | pass |
| W2 N=10000 t≠s | cube (64 units) | 133.049 | 18.5% | 34.3 | 3.09 | 8.5e-8 | – | pass |
| W2 N=10000 t≠s | cube (32 units) | 129.884 | 18.0% | 33.4 | 3.01 | 8.5e-8 | – | pass |
| W2 N=10000 t≠s | cube (128 units) | 135.671 | 18.8% | 34.9 | 3.15 | 8.5e-8 | – | pass |
| W2 N=10000 t≠s | plane (2 planes per cube) | 129.312 | 18.0% | 33.3 | 3.00 | 8.5e-8 | – | pass |
| W2 N=10000 t≠s | plane (4 planes per cube) | 128.799 | 17.9% | 33.2 | 2.99 | 8.5e-8 | – | pass |
| W2 N=10000 t=s | cube (64 units) | 133.453 | 18.5% | 33.7 | 3.10 | 1.1e-7 | – | pass |
| W2 N=10000 t=s | cube (32 units) | 130.222 | 18.1% | 32.9 | 3.02 | 1.1e-7 | – | pass |
| W2 N=10000 t=s | cube (128 units) | 135.728 | 18.9% | 34.3 | 3.15 | 1.1e-7 | – | pass |
| W2 N=10000 t=s | plane (2 planes per cube) | 129.662 | 18.0% | 32.7 | 3.01 | 1.1e-7 | – | pass |
| W2 N=10000 t=s | plane (4 planes per cube) | 128.458 | 17.8% | 32.4 | 2.98 | 1.1e-7 | – | pass |
| W2 N=100000 t≠s | cube (64 units) | 324.436 | 45.1% | 81.7 | 7.14 | 8.4e-8 | – | pass |
| W2 N=100000 t≠s | cube (32 units) | 304.603 | 42.3% | 76.7 | 6.71 | 8.4e-8 | – | pass |
| W2 N=100000 t≠s | cube (128 units) | 329.378 | 45.7% | 82.9 | 7.25 | 8.4e-8 | – | pass |
| W2 N=100000 t≠s | plane (2 planes per cube) | 298.656 | 41.5% | 75.2 | 6.58 | 8.4e-8 | – | pass |
| W2 N=100000 t≠s | plane (4 planes per cube) | 296.421 | 41.2% | 74.6 | 6.53 | 8.4e-8 | – | pass |
| W2 N=100000 t=s | cube (64 units) | 326.554 | 45.4% | 82.0 | 7.04 | 1.1e-7 | – | pass |
| W2 N=100000 t=s | cube (32 units) | 304.068 | 42.2% | 76.3 | 6.56 | 1.1e-7 | – | pass |
| W2 N=100000 t=s | cube (128 units) | 330.678 | 45.9% | 83.0 | 7.13 | 1.1e-7 | – | pass |
| W2 N=100000 t=s | plane (2 planes per cube) | 297.328 | 41.3% | 74.6 | 6.41 | 1.1e-7 | – | pass |
| W2 N=100000 t=s | plane (4 planes per cube) | 296.412 | 41.2% | 74.4 | 6.39 | 1.1e-7 | – | pass |

##### potential and gradient

| cell | layout | Gpairs/s | of model | x simd 1 thread | x simd 12 threads | max φ | max ∇φ | check |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W1 n_t=8 per-pair | cube (64 units) | 29.776 | 6.2% | 14.4 | 1.25 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=8 per-pair | cube (32 units) | 42.951 | 8.9% | 20.7 | 1.80 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=8 per-pair | cube (128 units) | 16.079 | 3.3% | 7.8 | 0.67 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=8 per-pair | plane (2 planes per cube) | 42.416 | 8.8% | 20.4 | 1.78 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=8 per-pair | plane (4 planes per cube) | 42.635 | 8.9% | 20.6 | 1.78 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=8 gathered | cube (64 units) | 51.368 | 10.7% | 20.4 | 1.76 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=8 gathered | cube (32 units) | 63.336 | 13.2% | 25.2 | 2.18 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=8 gathered | cube (128 units) | 29.954 | 6.2% | 11.9 | 1.03 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=8 gathered | plane (2 planes per cube) | 61.309 | 12.8% | 24.4 | 2.11 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=8 gathered | plane (4 planes per cube) | 61.476 | 12.8% | 24.5 | 2.11 | 8.8e-8 | 7.8e-8 | pass |
| W1 n_t=16 per-pair | cube (64 units) | 75.712 | 15.8% | 31.5 | 2.87 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=16 per-pair | cube (32 units) | 107.421 | 22.4% | 44.7 | 4.08 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=16 per-pair | cube (128 units) | 40.690 | 8.5% | 16.9 | 1.54 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=16 per-pair | plane (2 planes per cube) | 105.488 | 22.0% | 43.9 | 4.01 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=16 per-pair | plane (4 planes per cube) | 105.920 | 22.1% | 44.1 | 4.02 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=16 gathered | cube (64 units) | 103.086 | 21.5% | 40.1 | 3.79 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=16 gathered | cube (32 units) | 127.989 | 26.7% | 49.7 | 4.71 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=16 gathered | cube (128 units) | 60.936 | 12.7% | 23.7 | 2.24 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=16 gathered | plane (2 planes per cube) | 126.264 | 26.3% | 49.1 | 4.64 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=16 gathered | plane (4 planes per cube) | 126.134 | 26.3% | 49.0 | 4.64 | 7.0e-8 | 1.6e-7 | pass |
| W1 n_t=24 per-pair | cube (64 units) | 124.363 | 25.9% | 49.9 | 4.81 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=24 per-pair | cube (32 units) | 174.839 | 36.4% | 70.2 | 6.76 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=24 per-pair | cube (128 units) | 66.525 | 13.9% | 26.7 | 2.57 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=24 per-pair | plane (2 planes per cube) | 170.233 | 35.5% | 68.3 | 6.58 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=24 per-pair | plane (4 planes per cube) | 169.179 | 35.2% | 67.9 | 6.54 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=24 gathered | cube (64 units) | 152.722 | 31.8% | 59.1 | 5.86 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=24 gathered | cube (32 units) | 194.051 | 40.4% | 75.1 | 7.44 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=24 gathered | cube (128 units) | 89.095 | 18.6% | 34.5 | 3.42 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=24 gathered | plane (2 planes per cube) | 187.936 | 39.2% | 72.8 | 7.21 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=24 gathered | plane (4 planes per cube) | 189.749 | 39.5% | 73.5 | 7.28 | 8.8e-8 | 1.5e-7 | pass |
| W1 n_t=32 per-pair | cube (64 units) | 175.553 | 36.6% | 69.6 | 7.29 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=32 per-pair | cube (32 units) | 245.450 | 51.1% | 97.3 | 10.19 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=32 per-pair | cube (128 units) | 94.062 | 19.6% | 37.3 | 3.91 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=32 per-pair | plane (2 planes per cube) | 237.825 | 49.5% | 94.3 | 9.88 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=32 per-pair | plane (4 planes per cube) | 237.567 | 49.5% | 94.2 | 9.87 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=32 gathered | cube (64 units) | 204.433 | 42.6% | 79.1 | 8.44 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=32 gathered | cube (32 units) | 262.147 | 54.6% | 101.5 | 10.83 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=32 gathered | cube (128 units) | 118.731 | 24.7% | 46.0 | 4.90 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=32 gathered | plane (2 planes per cube) | 253.500 | 52.8% | 98.1 | 10.47 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=32 gathered | plane (4 planes per cube) | 255.514 | 53.2% | 98.9 | 10.55 | 5.8e-8 | 4.6e-7 | pass |
| W1 n_t=64 per-pair | cube (64 units) | 275.677 | 57.4% | 111.6 | 11.78 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=64 per-pair | cube (32 units) | 257.306 | 53.6% | 104.1 | 10.99 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=64 per-pair | cube (128 units) | 188.460 | 39.3% | 76.3 | 8.05 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=64 per-pair | plane (2 planes per cube) | 248.729 | 51.8% | 100.7 | 10.62 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=64 per-pair | plane (4 planes per cube) | 246.875 | 51.4% | 99.9 | 10.54 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=64 gathered | cube (64 units) | 286.521 | 59.7% | 121.5 | 12.92 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=64 gathered | cube (32 units) | 266.975 | 55.6% | 113.3 | 12.04 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=64 gathered | cube (128 units) | 210.655 | 43.9% | 89.4 | 9.50 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=64 gathered | plane (2 planes per cube) | 251.815 | 52.5% | 106.8 | 11.36 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=64 gathered | plane (4 planes per cube) | 248.750 | 51.8% | 105.5 | 11.22 | 1.2e-7 | 2.7e-7 | pass |
| W1 n_t=128 per-pair | cube (64 units) | 283.643 | 59.1% | 114.5 | 12.06 | 1.2e-7 | 5.9e-7 | pass |
| W1 n_t=128 per-pair | cube (32 units) | 263.377 | 54.9% | 106.3 | 11.20 | 1.2e-7 | 5.9e-7 | pass |
| W1 n_t=128 per-pair | cube (128 units) | 288.465 | 60.1% | 116.4 | 12.27 | 1.2e-7 | 5.9e-7 | pass |
| W1 n_t=128 per-pair | plane (2 planes per cube) | 253.767 | 52.9% | 102.4 | 10.79 | 1.2e-7 | 5.9e-7 | pass |
| W1 n_t=128 per-pair | plane (4 planes per cube) | 254.376 | 53.0% | 102.7 | 10.82 | 1.2e-7 | 5.9e-7 | pass |
| W1 n_t=128 gathered | cube (64 units) | 285.719 | 59.5% | 110.2 | 11.35 | 1.2e-7 | 5.9e-7 | pass |
| W1 n_t=128 gathered | cube (32 units) | 268.912 | 56.0% | 103.7 | 10.68 | 1.2e-7 | 5.9e-7 | pass |
| W1 n_t=128 gathered | cube (128 units) | 296.681 | 61.8% | 114.4 | 11.79 | 1.2e-7 | 5.9e-7 | pass |
| W1 n_t=128 gathered | plane (2 planes per cube) | 259.969 | 54.2% | 100.3 | 10.33 | 1.2e-7 | 5.9e-7 | pass |
| W1 n_t=128 gathered | plane (4 planes per cube) | 261.417 | 54.5% | 100.8 | 10.39 | 1.2e-7 | 5.9e-7 | pass |
| W2 N=1000 t≠s | cube (64 units) | 11.933 | 2.5% | 4.6 | 0.46 | 1.2e-7 | 7.9e-7 | pass |
| W2 N=1000 t≠s | cube (32 units) | 11.504 | 2.4% | 4.4 | 0.44 | 1.2e-7 | 7.9e-7 | pass |
| W2 N=1000 t≠s | cube (128 units) | 12.058 | 2.5% | 4.6 | 0.46 | 1.2e-7 | 7.9e-7 | pass |
| W2 N=1000 t≠s | plane (2 planes per cube) | 11.497 | 2.4% | 4.4 | 0.44 | 1.2e-7 | 7.9e-7 | pass |
| W2 N=1000 t≠s | plane (4 planes per cube) | 11.461 | 2.4% | 4.4 | 0.44 | 1.2e-7 | 7.9e-7 | pass |
| W2 N=1000 t=s | cube (64 units) | 11.834 | 2.5% | 4.6 | 0.44 | 1.4e-7 | 9.1e-7 | pass |
| W2 N=1000 t=s | cube (32 units) | 11.505 | 2.4% | 4.4 | 0.42 | 1.4e-7 | 9.1e-7 | pass |
| W2 N=1000 t=s | cube (128 units) | 12.095 | 2.5% | 4.7 | 0.45 | 1.4e-7 | 9.1e-7 | pass |
| W2 N=1000 t=s | plane (2 planes per cube) | 11.666 | 2.4% | 4.5 | 0.43 | 1.4e-7 | 9.1e-7 | pass |
| W2 N=1000 t=s | plane (4 planes per cube) | 11.732 | 2.4% | 4.5 | 0.43 | 1.4e-7 | 9.1e-7 | pass |
| W2 N=10000 t≠s | cube (64 units) | 123.666 | 25.8% | 48.1 | 4.34 | 8.5e-8 | 1.5e-6 | pass |
| W2 N=10000 t≠s | cube (32 units) | 122.260 | 25.5% | 47.5 | 4.29 | 8.5e-8 | 1.5e-6 | pass |
| W2 N=10000 t≠s | cube (128 units) | 119.666 | 24.9% | 46.5 | 4.20 | 8.5e-8 | 1.5e-6 | pass |
| W2 N=10000 t≠s | plane (2 planes per cube) | 121.491 | 25.3% | 47.2 | 4.27 | 8.5e-8 | 1.5e-6 | pass |
| W2 N=10000 t≠s | plane (4 planes per cube) | 115.569 | 24.1% | 44.9 | 4.06 | 8.5e-8 | 1.5e-6 | pass |
| W2 N=10000 t=s | cube (64 units) | 123.686 | 25.8% | 48.1 | 4.32 | 1.1e-7 | 1.2e-6 | pass |
| W2 N=10000 t=s | cube (32 units) | 122.031 | 25.4% | 47.4 | 4.26 | 1.1e-7 | 1.2e-6 | pass |
| W2 N=10000 t=s | cube (128 units) | 119.688 | 24.9% | 46.5 | 4.18 | 1.1e-7 | 1.2e-6 | pass |
| W2 N=10000 t=s | plane (2 planes per cube) | 121.047 | 25.2% | 47.1 | 4.23 | 1.1e-7 | 1.2e-6 | pass |
| W2 N=10000 t=s | plane (4 planes per cube) | 115.806 | 24.1% | 45.0 | 4.04 | 1.1e-7 | 1.2e-6 | pass |
| W2 N=100000 t≠s | cube (64 units) | 268.885 | 56.0% | 103.7 | 9.06 | 8.4e-8 | 3.0e-6 | pass |
| W2 N=100000 t≠s | cube (32 units) | 259.908 | 54.1% | 100.2 | 8.76 | 8.4e-8 | 3.0e-6 | pass |
| W2 N=100000 t≠s | cube (128 units) | 273.707 | 57.0% | 105.6 | 9.23 | 8.4e-8 | 3.0e-6 | pass |
| W2 N=100000 t≠s | plane (2 planes per cube) | 250.508 | 52.2% | 96.6 | 8.44 | 8.4e-8 | 3.0e-6 | pass |
| W2 N=100000 t≠s | plane (4 planes per cube) | 251.724 | 52.4% | 97.1 | 8.48 | 8.4e-8 | 3.0e-6 | pass |
| W2 N=100000 t=s | cube (64 units) | 269.709 | 56.2% | 104.0 | 9.09 | 1.1e-7 | 2.7e-6 | pass |
| W2 N=100000 t=s | cube (32 units) | 258.937 | 53.9% | 99.8 | 8.73 | 1.1e-7 | 2.7e-6 | pass |
| W2 N=100000 t=s | cube (128 units) | 272.192 | 56.7% | 104.9 | 9.18 | 1.1e-7 | 2.7e-6 | pass |
| W2 N=100000 t=s | plane (2 planes per cube) | 251.533 | 52.4% | 97.0 | 8.48 | 1.1e-7 | 2.7e-6 | pass |
| W2 N=100000 t=s | plane (4 planes per cube) | 251.552 | 52.4% | 97.0 | 8.48 | 1.1e-7 | 2.7e-6 | pass |

#### The C4.2 target (measured, never asserted)

| cell | output | target | best layout | Gpairs/s | of model | met |
| --- | --- | ---: | --- | ---: | ---: | --- |
| W2 N=100000 t≠s | φ | 25% (180 Gpairs/s) | cube (128 units) (default cube 64: 45.1%) | 329.378 | 45.7% | yes |
| W2 N=100000 t=s | φ | 25% (180 Gpairs/s) | cube (128 units) (default cube 64: 45.4%) | 330.678 | 45.9% | yes |
| W2 N=100000 t≠s | φ, ∇φ | 25% (120 Gpairs/s) | cube (128 units) (default cube 64: 56.0%) | 273.707 | 57.0% | yes |
| W2 N=100000 t=s | φ, ∇φ | 25% (120 Gpairs/s) | cube (128 units) (default cube 64: 56.2%) | 272.192 | 56.7% | yes |
| W1 n_t=64 per-pair | φ | 10% (72 Gpairs/s) | cube (64 units) (default cube 64: 46.8%) | 337.153 | 46.8% | yes |
| W1 n_t=64 gathered | φ | 10% (72 Gpairs/s) | cube (64 units) (default cube 64: 48.8%) | 351.701 | 48.8% | yes |
| W1 n_t=64 per-pair | φ, ∇φ | 10% (48 Gpairs/s) | cube (64 units) (default cube 64: 57.4%) | 275.677 | 57.4% | yes |
| W1 n_t=64 gathered | φ, ∇φ | 10% (48 Gpairs/s) | cube (64 units) (default cube 64: 59.7%) | 286.521 | 59.7% | yes |

#### The leaf stage inside the FMM

p = 3, gradients, N = 10^5 (the C3.2 cube: uniform level-4 tree; the Plummer sphere: max_level 16, 64 points per leaf). Device runs use synchronous stages, so the leaf stage (L2P, M2P and P2P) is timed whole; L2P and M2P run on the device (Phase 4 T7). The median of 5 evaluations after a warm-up. Difference: relative L2 of the device output from the host output, φ and ∇φ.

| problem | precision | configuration | leaf stage ms | evaluation ms | leaf stage x host 1 thread | x host 12 threads | difference φ / ∇φ |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| C3.2 cube | f32 | host, 1 thread | 35.39 | 72.44 | 1.00 | 0.11 | – |
| C3.2 cube | f32 | host, 12 threads | 3.79 | 8.95 | 9.34 | 1.00 | – |
| C3.2 cube | f32 | metal, P2P on the device (cube 64), 12 threads | 1.34 | 5.33 | 26.46 | 2.83 | 1.0e-6 / 1.2e-7 |
| C3.2 cube | f32 | metal, P2P on the device (plane 2), 12 threads | 0.85 | 3.99 | 41.56 | 4.45 | 1.0e-6 / 1.2e-7 |
| C3.2 cube | f32 | metal, P2P on the host fallback, 12 threads | 4.13 | 7.24 | 8.57 | 0.92 | 9.9e-7 / 3.9e-8 |
| Plummer sphere | f32 | host, 1 thread | 167.41 | 250.84 | 1.00 | 0.11 | – |
| Plummer sphere | f32 | host, 12 threads | 17.70 | 28.53 | 9.46 | 1.00 | – |
| Plummer sphere | f32 | metal, P2P on the device (cube 64), 12 threads | 2.05 | 6.08 | 81.60 | 8.63 | 1.5e-6 / 1.1e-7 |
| Plummer sphere | f32 | metal, P2P on the device (plane 2), 12 threads | 1.91 | 5.88 | 87.78 | 9.28 | 1.5e-6 / 1.1e-7 |
| Plummer sphere | f32 | metal, P2P on the host fallback, 12 threads | 7.88 | 12.03 | 21.25 | 2.25 | 1.5e-6 / 4.0e-8 |

</details>

### `leaf_kernels --device metal`

<details><summary>Output</summary>

### The device leaf operators per level (Phase 4 T7, C4.3)

| item | value |
| --- | --- |
| machine | Apple M3 Max (16 physical, 16 logical) |
| performance cores | 12 |
| target | aarch64-macos, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| device | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| host threads of the many-thread column | 12 |
| timing | median of 15 batches of >= 20 ms; device launches queued between syncs, a warm-up launch first (compilation excluded); host: the host operator's per-target bodies on 1 thread and on scoped threads, each a contiguous share of the rows |
| precision | f32 (f64 not timed: Metal has no f64; CUDA type-checked only) |

#### C3.2 cube, p = 3, f32, gradients

| operator | level | rows | entries | points | cube (64 units, tile 32) µs | difference | cube (32 units, tile 16) µs | difference | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2M | 4 | 4096 | 4096 | 100000 | 20.9 | 5.8e-8 | 16.2 | 5.8e-8 | 2287.7 | 315.8 | 109.5 | 15.11 |
| L2P | 4 | 4096 | 4096 | 100000 | 14.0 | 3.4e-8 | 13.7 | 3.4e-8 | 6763.5 | 802.4 | 483.5 | 57.36 |

Totals over the levels (metal; the speed-ups of the first layout, cube (64 units, tile 32)):

| operator | cube (64 units, tile 32) ms | cube (32 units, tile 16) ms | host 1 thread ms | host 12 threads ms | x host 1 | x host 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2M | 0.021 | 0.016 | 2.288 | 0.316 | 109.5 | 15.11 |
| L2P | 0.014 | 0.014 | 6.763 | 0.802 | 483.5 | 57.36 |

#### C3.2 cube, p = 8, f32, gradients

| operator | level | rows | entries | points | cube (64 units, tile 32) µs | difference | cube (32 units, tile 16) µs | difference | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2M | 4 | 4096 | 4096 | 100000 | 78.1 | 5.8e-8 | 58.8 | 5.8e-8 | 13383.9 | 1304.7 | 171.3 | 16.70 |
| L2P | 4 | 4096 | 4096 | 100000 | 32.3 | 3.5e-8 | 28.8 | 3.5e-8 | 33434.2 | 3093.5 | 1036.3 | 95.89 |

Totals over the levels (metal; the speed-ups of the first layout, cube (64 units, tile 32)):

| operator | cube (64 units, tile 32) ms | cube (32 units, tile 16) ms | host 1 thread ms | host 12 threads ms | x host 1 | x host 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2M | 0.078 | 0.059 | 13.384 | 1.305 | 171.3 | 16.70 |
| L2P | 0.032 | 0.029 | 33.434 | 3.093 | 1036.3 | 95.89 |

#### Plummer sphere, p = 3, f32, gradients

| operator | level | rows | entries | points | cube (64 units, tile 32) µs | difference | cube (32 units, tile 16) µs | difference | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2M | 3 | 512 | 448 | 2690 | 7.3 | 5.4e-8 | 7.7 | 5.4e-8 | 86.0 | 90.4 | 11.8 | 12.42 |
| L2P | 3 | 448 | 448 | 2690 | 4.0 | 3.7e-8 | 4.1 | 3.7e-8 | 209.0 | 115.5 | 52.0 | 28.71 |
| M2P | 3 | 448 | 4472 | 2690 | 35.6 | 1.3e-7 | 57.8 | 1.3e-7 | 8185.8 | 1392.9 | 230.2 | 39.17 |
| P2M | 4 | 512 | 395 | 6172 | 8.0 | 5.6e-8 | 8.6 | 5.6e-8 | 162.9 | 109.0 | 20.5 | 13.68 |
| P2L | 4 | 512 | 4472 | 88755 | 39.0 | 1.6e-7 | 39.4 | 1.6e-7 | 2329.6 | 371.9 | 59.8 | 9.55 |
| L2P | 4 | 395 | 395 | 6172 | 4.1 | 3.4e-8 | 4.0 | 3.4e-8 | 442.2 | 140.6 | 106.7 | 33.94 |
| M2P | 4 | 395 | 6320 | 6172 | 42.8 | 1.2e-7 | 68.9 | 1.2e-7 | 16606.5 | 2377.1 | 388.4 | 55.59 |
| P2M | 5 | 936 | 733 | 14286 | 9.2 | 6.3e-8 | 9.9 | 6.3e-8 | 347.3 | 132.8 | 37.8 | 14.45 |
| P2L | 5 | 936 | 6320 | 181730 | 54.3 | 1.7e-7 | 54.3 | 1.7e-7 | 4673.9 | 606.8 | 86.0 | 11.17 |
| L2P | 5 | 733 | 733 | 14286 | 4.3 | 3.3e-8 | 4.6 | 3.3e-8 | 987.2 | 197.9 | 229.0 | 45.89 |
| M2P | 5 | 733 | 9380 | 14286 | 52.9 | 1.1e-7 | 79.7 | 1.1e-7 | 32741.3 | 4702.6 | 619.3 | 88.94 |
| P2M | 6 | 1624 | 1322 | 27565 | 11.3 | 5.9e-8 | 11.8 | 5.9e-8 | 644.3 | 164.0 | 57.0 | 14.51 |
| P2L | 6 | 1624 | 9380 | 363671 | 92.2 | 2.3e-7 | 72.0 | 2.3e-7 | 9155.1 | 1309.1 | 99.3 | 14.20 |
| L2P | 6 | 1322 | 1322 | 27565 | 5.9 | 3.5e-8 | 6.3 | 3.5e-8 | 1877.4 | 282.2 | 318.7 | 47.90 |
| M2P | 6 | 1322 | 12320 | 27565 | 61.0 | 1.3e-7 | 83.4 | 1.3e-7 | 48157.1 | 6114.2 | 789.8 | 100.28 |
| P2M | 7 | 2416 | 2364 | 45441 | 14.1 | 5.8e-8 | 13.7 | 5.8e-8 | 1043.7 | 194.4 | 74.1 | 13.80 |
| P2L | 7 | 2416 | 12320 | 534269 | 116.2 | 1.9e-7 | 92.3 | 1.9e-7 | 13417.5 | 1654.7 | 115.5 | 14.25 |
| L2P | 7 | 2364 | 2364 | 45441 | 9.0 | 3.4e-8 | 9.4 | 3.4e-8 | 3076.2 | 421.6 | 340.6 | 46.68 |
| M2P | 7 | 2364 | 4150 | 45441 | 53.5 | 1.1e-7 | 89.0 | 1.1e-7 | 19466.4 | 5469.1 | 363.6 | 102.16 |
| P2M | 8 | 416 | 416 | 3846 | 4.3 | 4.4e-8 | 4.4 | 4.4e-8 | 109.9 | 98.4 | 25.8 | 23.09 |
| P2L | 8 | 416 | 4150 | 217325 | 84.8 | 2.5e-7 | 89.1 | 2.5e-7 | 5491.0 | 686.2 | 64.7 | 8.09 |
| L2P | 8 | 416 | 416 | 3846 | 4.0 | 3.4e-8 | 4.0 | 3.4e-8 | 287.6 | 123.1 | 72.1 | 30.89 |

Totals over the levels (metal; the speed-ups of the first layout, cube (64 units, tile 32)):

| operator | cube (64 units, tile 32) ms | cube (32 units, tile 16) ms | host 1 thread ms | host 12 threads ms | x host 1 | x host 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2M | 0.054 | 0.056 | 2.394 | 0.789 | 44.3 | 14.59 |
| L2P | 0.031 | 0.033 | 6.880 | 1.281 | 219.2 | 40.81 |
| M2P | 0.246 | 0.379 | 125.157 | 20.056 | 509.4 | 81.63 |
| P2L | 0.386 | 0.347 | 35.067 | 4.629 | 90.7 | 11.98 |

#### Plummer sphere, p = 8, f32, gradients

| operator | level | rows | entries | points | cube (64 units, tile 32) µs | difference | cube (32 units, tile 16) µs | difference | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2M | 3 | 512 | 448 | 2690 | 12.7 | 5.4e-8 | 17.1 | 5.4e-8 | 388.4 | 137.2 | 30.7 | 10.85 |
| L2P | 3 | 448 | 448 | 2690 | 6.3 | 3.6e-8 | 8.7 | 3.6e-8 | 917.6 | 207.8 | 146.0 | 33.07 |
| M2P | 3 | 448 | 4472 | 2690 | 153.2 | 5.1e-7 | 282.4 | 5.1e-7 | 33955.0 | 4682.9 | 221.7 | 30.57 |
| P2M | 4 | 512 | 395 | 6172 | 15.7 | 5.6e-8 | 19.8 | 5.6e-8 | 867.4 | 175.0 | 55.1 | 11.12 |
| P2L | 4 | 512 | 4472 | 88755 | 103.3 | 4.9e-7 | 92.9 | 4.9e-7 | 12727.2 | 1589.6 | 123.2 | 15.39 |
| L2P | 4 | 395 | 395 | 6172 | 6.9 | 3.5e-8 | 9.0 | 3.5e-8 | 2069.5 | 306.4 | 302.0 | 44.71 |
| M2P | 4 | 395 | 6320 | 6172 | 182.9 | 4.3e-7 | 344.5 | 4.3e-7 | 69581.4 | 9980.6 | 380.4 | 54.57 |
| P2M | 5 | 936 | 733 | 14286 | 22.7 | 6.4e-8 | 22.7 | 6.4e-8 | 1931.6 | 279.3 | 85.0 | 12.29 |
| P2L | 5 | 936 | 6320 | 181730 | 173.0 | 3.9e-7 | 156.8 | 3.9e-7 | 24899.0 | 2937.1 | 143.9 | 16.97 |
| L2P | 5 | 733 | 733 | 14286 | 10.6 | 3.5e-8 | 11.4 | 3.5e-8 | 4853.7 | 637.1 | 459.3 | 60.29 |
| M2P | 5 | 733 | 9380 | 14286 | 240.0 | 4.5e-7 | 401.9 | 4.5e-7 | 140696.2 | 19404.6 | 586.1 | 80.84 |
| P2M | 6 | 1624 | 1322 | 27565 | 31.8 | 5.9e-8 | 29.1 | 5.9e-8 | 3714.6 | 460.3 | 116.7 | 14.46 |
| P2L | 6 | 1624 | 9380 | 363671 | 302.9 | 3.8e-7 | 260.2 | 3.8e-7 | 50696.0 | 6460.4 | 167.4 | 21.33 |
| L2P | 6 | 1322 | 1322 | 27565 | 15.1 | 3.5e-8 | 15.8 | 3.5e-8 | 9147.0 | 1012.7 | 607.6 | 67.27 |
| M2P | 6 | 1322 | 12320 | 27565 | 264.2 | 4.1e-7 | 406.2 | 4.1e-7 | 207616.7 | 25475.4 | 785.7 | 96.41 |
| P2M | 7 | 2416 | 2364 | 45441 | 42.5 | 5.8e-8 | 36.1 | 5.8e-8 | 6174.6 | 831.9 | 145.4 | 19.58 |
| P2L | 7 | 2416 | 12320 | 534269 | 409.3 | 4.1e-7 | 337.6 | 4.1e-7 | 73419.1 | 8842.8 | 179.4 | 21.61 |
| L2P | 7 | 2364 | 2364 | 45441 | 22.0 | 3.5e-8 | 21.7 | 3.5e-8 | 15248.8 | 1740.7 | 691.9 | 78.98 |
| M2P | 7 | 2364 | 4150 | 45441 | 229.6 | 4.0e-7 | 441.7 | 4.0e-7 | 83162.3 | 23093.0 | 362.2 | 100.57 |
| P2M | 8 | 416 | 416 | 3846 | 8.8 | 4.4e-8 | 8.6 | 4.4e-8 | 538.9 | 149.8 | 61.4 | 17.05 |
| P2L | 8 | 416 | 4150 | 217325 | 232.7 | 3.8e-7 | 233.1 | 3.8e-7 | 29666.9 | 3348.8 | 127.5 | 14.39 |
| L2P | 8 | 416 | 416 | 3846 | 6.4 | 3.6e-8 | 6.5 | 3.6e-8 | 1380.9 | 214.9 | 216.8 | 33.74 |

Totals over the levels (metal; the speed-ups of the first layout, cube (64 units, tile 32)):

| operator | cube (64 units, tile 32) ms | cube (32 units, tile 16) ms | host 1 thread ms | host 12 threads ms | x host 1 | x host 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2M | 0.134 | 0.133 | 13.615 | 2.034 | 101.4 | 15.15 |
| L2P | 0.067 | 0.073 | 33.617 | 4.120 | 500.5 | 61.33 |
| M2P | 1.070 | 1.877 | 535.012 | 82.636 | 500.0 | 77.23 |
| P2L | 1.221 | 1.081 | 191.408 | 23.179 | 156.7 | 18.98 |

</details>

### `translation_kernels --device metal`

<details><summary>Output</summary>

### The device M2M and L2L per level (Phase 4 T8, C4.4)

| item | value |
| --- | --- |
| machine | Apple M3 Max (16 physical, 16 logical) |
| target | aarch64-macos, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| device | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| timing | median of 15 batches of >= 20 ms; launches queued between syncs, a warm-up launch and a sync first (compilation excluded) |
| peak | 14300 GFLOP/s (M3 Max GPU f32, derived in the spike report, not measured); GFLOP/s counts 2 n² per column |
| precision | f32 (f64 not timed: Metal has no f64; CUDA type-checked only) |

#### C3.2 cube, p = 3 (n = 16), metal

| view | level | pairs | k_max | plan's GEMM (chunks) | hand-written GEMM µs | GFLOP/s | % peak | library GEMM µs | GFLOP/s | % peak | spike (p, B): hand / library GFLOP/s | gather µs | level call µs | accumulate µs (derived) | gather + accumulate share |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| M2M global | 0 | 8 | 1 | hand-written (1) | 3.7 | 1 | 0.0 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 10.0 | 3.8 | 63% |
| M2M local | 1 | 64 | 8 | hand-written (1) | 3.7 | 9 | 0.1 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 9.9 | 3.5 | 62% |
| L2L | 1 | 8 | 1 | hand-written (1) | 3.6 | 1 | 0.0 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 9.0 | 2.8 | 60% |
| M2M local | 2 | 512 | 64 | hand-written (1) | 3.8 | 69 | 0.5 | – | – | – | (4, 1000): 239 / rejected | 2.7 | 10.2 | 3.8 | 63% |
| L2L | 2 | 64 | 8 | hand-written (1) | 3.7 | 9 | 0.1 | – | – | – | (4, 1000): 239 / rejected | 2.5 | 8.9 | 2.7 | 58% |
| M2M local | 3 | 4096 | 512 | hand-written (1) | 4.4 | 476 | 3.3 | – | – | – | (4, 10000): 1431 / rejected | 3.0 | 11.3 | 3.9 | 61% |
| L2L | 3 | 512 | 64 | hand-written (1) | 3.8 | 69 | 0.5 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 8.9 | 2.5 | 57% |
| L2L | 4 | 4096 | 512 | hand-written (1) | 4.3 | 488 | 3.4 | – | – | – | (4, 10000): 1431 / rejected | 3.0 | 10.8 | 3.5 | 60% |

Totals over the level calls: 79.0 µs per evaluation's M2M and L2L (8 calls), of which GEMM 31.1 µs, gather 21.5 µs, accumulation 26.4 µs (gather and accumulation 61%). The hand-written layout: cube(16 x 4 units, 4 columns per unit).

#### C3.2 cube, p = 8 (n = 81), metal

| view | level | pairs | k_max | plan's GEMM (chunks) | hand-written GEMM µs | GFLOP/s | % peak | library GEMM µs | GFLOP/s | % peak | spike (p, B): hand / library GFLOP/s | gather µs | level call µs | accumulate µs (derived) | gather + accumulate share |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| M2M global | 0 | 8 | 1 | hand-written (1) | 14.9 | 7 | 0.0 | – | – | – | (8, 1000): 625 / 2045 | 2.6 | 21.9 | 4.3 | 32% |
| M2M local | 1 | 64 | 8 | hand-written (1) | 16.0 | 52 | 0.4 | – | – | – | (8, 1000): 625 / 2045 | 2.6 | 23.7 | 5.0 | 32% |
| L2L | 1 | 8 | 1 | hand-written (1) | 15.1 | 7 | 0.0 | – | – | – | (8, 1000): 625 / 2045 | 2.6 | 19.9 | 2.2 | 24% |
| M2M local | 2 | 512 | 64 | library (1) | 16.0 | 420 | 2.9 | 5.5 | 1217 | 8.5 | (8, 1000): 625 / 2045 | 2.6 | 13.2 | 5.1 | 58% |
| L2L | 2 | 64 | 8 | hand-written (1) | 16.1 | 52 | 0.4 | – | – | – | (8, 1000): 625 / 2045 | 2.6 | 21.6 | 2.8 | 25% |
| M2M local | 3 | 4096 | 512 | library (1) | 33.4 | 1607 | 11.2 | 21.3 | 2520 | 17.6 | (8, 10000): 3585 / 3792 | 6.3 | 33.8 | 6.1 | 37% |
| L2L | 3 | 512 | 64 | library (1) | 16.2 | 415 | 2.9 | 5.5 | 1218 | 8.5 | (8, 1000): 625 / 2045 | 2.6 | 11.3 | 3.1 | 51% |
| L2L | 4 | 4096 | 512 | library (1) | 33.6 | 1601 | 11.2 | 21.8 | 2469 | 17.3 | (8, 10000): 3585 / 3792 | 6.3 | 36.4 | 8.3 | 40% |

Totals over the level calls: 181.6 µs per evaluation's M2M and L2L (8 calls), of which GEMM 116.3 µs, gather 28.4 µs, accumulation 37.0 µs (gather and accumulation 36%). The hand-written layout: cube(32 x 2 units, 4 columns per unit).

#### C3.2 cube, p = 16 (n = 289), metal

| view | level | pairs | k_max | plan's GEMM (chunks) | hand-written GEMM µs | GFLOP/s | % peak | library GEMM µs | GFLOP/s | % peak | spike (p, B): hand / library GFLOP/s | gather µs | level call µs | accumulate µs (derived) | gather + accumulate share |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| M2M global | 0 | 8 | 1 | hand-written (1) | 117.9 | 11 | 0.1 | – | – | – | (16, 1000): 2146 / 3894 | 2.5 | 127.2 | 6.7 | 7% |
| M2M local | 1 | 64 | 8 | hand-written (1) | 119.0 | 90 | 0.6 | – | – | – | (16, 1000): 2146 / 3894 | 2.6 | 129.6 | 8.0 | 8% |
| L2L | 1 | 8 | 1 | hand-written (1) | 117.8 | 11 | 0.1 | – | – | – | (16, 1000): 2146 / 3894 | 2.6 | 125.1 | 4.7 | 6% |
| M2M local | 2 | 512 | 64 | library (1) | 119.6 | 715 | 5.0 | 29.2 | 2932 | 20.5 | (16, 1000): 2146 / 3894 | 4.0 | 37.9 | 4.7 | 23% |
| L2L | 2 | 64 | 8 | hand-written (1) | 119.0 | 90 | 0.6 | – | – | – | (16, 1000): 2146 / 3894 | 2.6 | 127.6 | 6.0 | 7% |
| M2M local | 3 | 4096 | 512 | library (1) | 290.1 | 2359 | 16.5 | 182.8 | 3742 | 26.2 | (16, 10000): 3572 / 4475 | 16.9 | 216.7 | 16.9 | 16% |
| L2L | 3 | 512 | 64 | library (1) | 119.2 | 718 | 5.0 | 28.6 | 2987 | 20.9 | (16, 1000): 2146 / 3894 | 4.0 | 38.6 | 6.0 | 26% |
| L2L | 4 | 4096 | 512 | library (1) | 288.5 | 2372 | 16.6 | 182.5 | 3748 | 26.2 | (16, 10000): 3572 / 4475 | 16.6 | 226.2 | 27.1 | 19% |

Totals over the level calls: 1028.8 µs per evaluation's M2M and L2L (8 calls), of which GEMM 896.9 µs, gather 51.7 µs, accumulation 80.2 µs (gather and accumulation 13%). The hand-written layout: cube(32 x 2 units, 4 columns per unit).

#### Plummer sphere, p = 3 (n = 16), metal

| view | level | pairs | k_max | plan's GEMM (chunks) | hand-written GEMM µs | GFLOP/s | % peak | library GEMM µs | GFLOP/s | % peak | spike (p, B): hand / library GFLOP/s | gather µs | level call µs | accumulate µs (derived) | gather + accumulate share |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| M2M global | 0 | 8 | 1 | hand-written (1) | 3.6 | 1 | 0.0 | – | – | – | (4, 1000): 239 / rejected | 2.7 | 10.1 | 3.8 | 65% |
| M2M local | 1 | 64 | 8 | hand-written (1) | 3.7 | 9 | 0.1 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 9.8 | 3.5 | 62% |
| L2L | 1 | 8 | 1 | hand-written (1) | 3.6 | 1 | 0.0 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 8.9 | 2.7 | 60% |
| M2M local | 2 | 512 | 64 | hand-written (1) | 3.8 | 69 | 0.5 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 10.2 | 3.9 | 63% |
| L2L | 2 | 64 | 8 | hand-written (1) | 3.7 | 9 | 0.1 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 8.8 | 2.5 | 58% |
| M2M local | 3 | 512 | 64 | hand-written (1) | 3.8 | 69 | 0.5 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 10.3 | 3.9 | 63% |
| L2L | 3 | 512 | 64 | hand-written (1) | 3.8 | 69 | 0.5 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 9.0 | 2.6 | 58% |
| M2M local | 4 | 936 | 117 | hand-written (1) | 3.9 | 122 | 0.9 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 10.3 | 3.8 | 62% |
| L2L | 4 | 512 | 64 | hand-written (1) | 3.8 | 69 | 0.5 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 8.8 | 2.4 | 57% |
| M2M local | 5 | 1624 | 203 | hand-written (1) | 3.9 | 213 | 1.5 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 10.4 | 4.0 | 63% |
| L2L | 5 | 936 | 117 | hand-written (1) | 3.9 | 123 | 0.9 | – | – | – | (4, 1000): 239 / rejected | 2.5 | 9.1 | 2.7 | 57% |
| M2M local | 6 | 2416 | 302 | hand-written (1) | 4.0 | 311 | 2.2 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 10.7 | 4.1 | 63% |
| L2L | 6 | 1624 | 203 | hand-written (1) | 3.9 | 214 | 1.5 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 8.8 | 2.3 | 56% |
| M2M local | 7 | 416 | 52 | hand-written (1) | 3.8 | 56 | 0.4 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 10.2 | 3.8 | 63% |
| L2L | 7 | 2416 | 302 | hand-written (1) | 4.0 | 310 | 2.2 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 9.3 | 2.7 | 57% |
| L2L | 8 | 416 | 52 | hand-written (1) | 3.8 | 56 | 0.4 | – | – | – | (4, 1000): 239 / rejected | 2.6 | 8.9 | 2.4 | 57% |

Totals over the level calls: 153.7 µs per evaluation's M2M and L2L (16 calls), of which GEMM 61.0 µs, gather 41.5 µs, accumulation 51.3 µs (gather and accumulation 60%). The hand-written layout: cube(16 x 4 units, 4 columns per unit).

#### Plummer sphere, p = 8 (n = 81), metal

| view | level | pairs | k_max | plan's GEMM (chunks) | hand-written GEMM µs | GFLOP/s | % peak | library GEMM µs | GFLOP/s | % peak | spike (p, B): hand / library GFLOP/s | gather µs | level call µs | accumulate µs (derived) | gather + accumulate share |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| M2M global | 0 | 8 | 1 | hand-written (1) | 15.2 | 7 | 0.0 | – | – | – | (8, 1000): 625 / 2045 | 2.6 | 21.8 | 4.0 | 30% |
| M2M local | 1 | 64 | 8 | hand-written (1) | 16.1 | 52 | 0.4 | – | – | – | (8, 1000): 625 / 2045 | 2.6 | 23.6 | 5.0 | 32% |
| L2L | 1 | 8 | 1 | hand-written (1) | 14.9 | 7 | 0.0 | – | – | – | (8, 1000): 625 / 2045 | 2.6 | 19.9 | 2.5 | 26% |
| M2M local | 2 | 512 | 64 | library (1) | 16.3 | 413 | 2.9 | 5.5 | 1218 | 8.5 | (8, 1000): 625 / 2045 | 2.6 | 13.3 | 5.1 | 58% |
| L2L | 2 | 64 | 8 | hand-written (1) | 16.1 | 52 | 0.4 | – | – | – | (8, 1000): 625 / 2045 | 2.6 | 21.8 | 3.0 | 26% |
| M2M local | 3 | 512 | 64 | library (1) | 16.3 | 412 | 2.9 | 5.5 | 1219 | 8.5 | (8, 1000): 625 / 2045 | 2.6 | 13.5 | 5.4 | 59% |
| L2L | 3 | 512 | 64 | library (1) | 16.3 | 412 | 2.9 | 5.5 | 1216 | 8.5 | (8, 1000): 625 / 2045 | 2.7 | 11.4 | 3.2 | 52% |
| M2M local | 4 | 936 | 117 | library (1) | 16.5 | 745 | 5.2 | 7.6 | 1624 | 11.4 | (8, 1000): 625 / 2045 | 2.9 | 15.7 | 5.3 | 52% |
| L2L | 4 | 512 | 64 | library (1) | 16.0 | 420 | 2.9 | 5.4 | 1234 | 8.6 | (8, 1000): 625 / 2045 | 2.7 | 11.4 | 3.3 | 52% |
| M2M local | 5 | 1624 | 203 | library (1) | 17.3 | 1228 | 8.6 | 13.2 | 1618 | 11.3 | (8, 1000): 625 / 2045 | 3.7 | 23.5 | 6.6 | 44% |
| L2L | 5 | 936 | 117 | library (1) | 16.6 | 741 | 5.2 | 7.5 | 1633 | 11.4 | (8, 1000): 625 / 2045 | 3.2 | 14.2 | 3.5 | 47% |
| M2M local | 6 | 2416 | 302 | library (1) | 24.6 | 1291 | 9.0 | 16.1 | 1972 | 13.8 | (8, 1000): 625 / 2045 | 4.6 | 28.0 | 7.4 | 43% |
| L2L | 6 | 1624 | 203 | library (1) | 17.6 | 1210 | 8.5 | 13.1 | 1625 | 11.4 | (8, 1000): 625 / 2045 | 3.8 | 22.7 | 5.7 | 42% |
| M2M local | 7 | 416 | 52 | library (1) | 16.3 | 335 | 2.3 | 5.6 | 969 | 6.8 | (8, 1000): 625 / 2045 | 2.6 | 14.8 | 6.6 | 62% |
| L2L | 7 | 2416 | 302 | library (1) | 24.4 | 1298 | 9.1 | 16.2 | 1953 | 13.7 | (8, 1000): 625 / 2045 | 4.6 | 27.5 | 6.7 | 41% |
| L2L | 8 | 416 | 52 | library (1) | 16.4 | 333 | 2.3 | 5.7 | 959 | 6.7 | (8, 1000): 625 / 2045 | 2.6 | 11.5 | 3.2 | 51% |

Totals over the level calls: 294.8 µs per evaluation's M2M and L2L (16 calls), of which GEMM 169.2 µs, gather 48.9 µs, accumulation 76.6 µs (gather and accumulation 43%). The hand-written layout: cube(32 x 2 units, 4 columns per unit).

#### Plummer sphere, p = 16 (n = 289), metal

| view | level | pairs | k_max | plan's GEMM (chunks) | hand-written GEMM µs | GFLOP/s | % peak | library GEMM µs | GFLOP/s | % peak | spike (p, B): hand / library GFLOP/s | gather µs | level call µs | accumulate µs (derived) | gather + accumulate share |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| M2M global | 0 | 8 | 1 | hand-written (1) | 117.3 | 11 | 0.1 | – | – | – | (16, 1000): 2146 / 3894 | 2.6 | 127.4 | 7.4 | 8% |
| M2M local | 1 | 64 | 8 | hand-written (1) | 119.3 | 90 | 0.6 | – | – | – | (16, 1000): 2146 / 3894 | 2.6 | 129.5 | 7.6 | 8% |
| L2L | 1 | 8 | 1 | hand-written (1) | 117.1 | 11 | 0.1 | – | – | – | (16, 1000): 2146 / 3894 | 2.7 | 125.3 | 5.5 | 7% |
| M2M local | 2 | 512 | 64 | library (1) | 119.5 | 715 | 5.0 | 29.1 | 2936 | 20.5 | (16, 1000): 2146 / 3894 | 4.0 | 37.9 | 4.7 | 23% |
| L2L | 2 | 64 | 8 | hand-written (1) | 119.3 | 90 | 0.6 | – | – | – | (16, 1000): 2146 / 3894 | 2.6 | 127.4 | 5.4 | 6% |
| M2M local | 3 | 512 | 64 | library (1) | 119.4 | 716 | 5.0 | 28.4 | 3011 | 21.1 | (16, 1000): 2146 / 3894 | 4.0 | 39.9 | 7.5 | 29% |
| L2L | 3 | 512 | 64 | library (1) | 119.4 | 716 | 5.0 | 29.2 | 2934 | 20.5 | (16, 1000): 2146 / 3894 | 4.0 | 38.7 | 5.6 | 25% |
| M2M local | 4 | 936 | 117 | library (1) | 121.4 | 1288 | 9.0 | 56.9 | 2746 | 19.2 | (16, 1000): 2146 / 3894 | 5.5 | 71.0 | 8.5 | 20% |
| L2L | 4 | 512 | 64 | library (1) | 119.5 | 715 | 5.0 | 29.1 | 2937 | 20.5 | (16, 1000): 2146 / 3894 | 4.0 | 37.8 | 4.7 | 23% |
| M2M local | 5 | 1624 | 203 | library (1) | 130.0 | 2088 | 14.6 | 88.4 | 3070 | 21.5 | (16, 1000): 2146 / 3894 | 8.0 | 109.1 | 12.8 | 19% |
| L2L | 5 | 936 | 117 | library (1) | 121.4 | 1288 | 9.0 | 57.6 | 2715 | 19.0 | (16, 1000): 2146 / 3894 | 5.5 | 71.3 | 8.2 | 19% |
| M2M local | 6 | 2416 | 302 | library (1) | 174.0 | 2319 | 16.2 | 124.9 | 3230 | 22.6 | (16, 1000): 2146 / 3894 | 10.8 | 151.9 | 16.2 | 18% |
| L2L | 6 | 1624 | 203 | library (1) | 129.7 | 2091 | 14.6 | 88.0 | 3084 | 21.6 | (16, 1000): 2146 / 3894 | 8.0 | 109.2 | 13.3 | 19% |
| M2M local | 7 | 416 | 52 | library (1) | 119.2 | 583 | 4.1 | 30.7 | 2263 | 15.8 | (16, 1000): 2146 / 3894 | 3.7 | 49.4 | 15.1 | 38% |
| L2L | 7 | 2416 | 302 | library (1) | 175.8 | 2296 | 16.1 | 126.4 | 3193 | 22.3 | (16, 1000): 2146 / 3894 | 10.9 | 156.4 | 19.2 | 19% |
| L2L | 8 | 416 | 52 | library (1) | 119.3 | 582 | 4.1 | 32.8 | 2120 | 14.8 | (16, 1000): 2146 / 3894 | 3.7 | 42.1 | 5.7 | 22% |

Totals over the level calls: 1424.5 µs per evaluation's M2M and L2L (16 calls), of which GEMM 1194.6 µs, gather 82.4 µs, accumulation 147.5 µs (gather and accumulation 16%). The hand-written layout: cube(32 x 2 units, 4 columns per unit).

</details>

### `m2l_kernels --device metal`

<details><summary>Output</summary>

### The device dense M2L per level (Phase 4 T9, C4.5)

| item | value |
| --- | --- |
| machine | Apple M3 Max (16 physical, 16 logical) |
| target | aarch64-macos, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| device | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| timing | median of 15 batches of >= 20 ms; launches queued between syncs, a warm-up launch and a sync first (compilation excluded) |
| peak | 14300 GFLOP/s (M3 Max GPU f32, derived in the spike report, not measured); GFLOP/s counts 2 n² per pair (useful flops) |
| host | `LaplaceOperator::m2l_pair` with the dense f32 tables, 1 and 12 scoped threads |
| precision | f32 (f64 not timed: Metal has no f64; CUDA type-checked only) |

#### C3.2 cube, p = 3 (n = 16), metal

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 5.1 | 312 | 2.2 | – | (4, 1000): 239 | 131% |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 24.9 | 1098 | 7.7 | – | (4, 1000): 239 | 459% |
| 4 | 4096 | 584136 | 316 | 1849 / 3584 | hand-written (1) | 100% | 225.7 | 1325 | 9.3 | – | (4, 1000): 239 | 555% |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 2.7 | 5.1 | 15.2 | 24.2 | 3 | 8253.6 | 948 | hand-written | 196.4 | 118.8 | 8.1 | 4.9 | 6.5e-7 (host) |
| 3 | 11.9 | 24.9 | 47.8 | 86.9 | 3 | 8233.0 | 948 | hand-written | 2994.5 | 487.5 | 34.5 | 5.6 | 6.6e-7 (host) |
| 4 | 92.8 | 225.7 | 113.1 | 487.1 | 3 | 8413.3 | 948 | hand-written | 32256.3 | 3933.5 | 66.2 | 8.1 | 8.5e-7 (host) |

#### C3.2 cube, p = 8 (n = 81), metal

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 35.0 | 1162 | 8.1 | – | (8, 1000): 625 | 186% |
|  | library rejected: [316, 32, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | library (1) | 44% | 639.6 | 1095 | 7.7 | 335.4 (2087) | (8, 1000): 2045 | 54% |
| 4 | 4096 | 584136 | 316 | 1849 / 3584 | library (6) | 55% | 5466.7 | 1402 | 9.8 | 3495.3 (2193) | (8, 1000): 2045 | 69% |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 5.3 | 35.0 | 15.8 | 58.7 | 3 | 8585.6 | 948 | hand-written | 2119.7 | 419.6 | 36.1 | 7.1 | 7.7e-7 (host) |
| 3 | 107.8 | 639.6 | 85.3 | 859.5 | 3 | 8333.2 | 948 | hand-written | 41462.2 | 7925.8 | 48.2 | 9.2 | 1.2e-6 (host) |
| 4 | 951.9 | 5466.7 | 950.5 | 7499.9 | 18 | 6891.8 | 948 | library | 433559.3 | 100389.0 | 57.8 | 13.4 | 1.3e-6 (host) |

#### C3.2 cube, p = 12 (n = 169), metal

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 143.3 | 1234 | 8.6 | – | (12, 1000): 1426 | 87% |
|  | library rejected: [316, 32, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | library (2) | 48% | 1907.0 | 1598 | 11.2 | 1362.6 (2237) | (12, 1000): 3396 | 47% |
| 4 | 4096 | 584136 | 316 | 1849 / 3584 | library (10) | 62% | 16383.7 | 2037 | 14.2 | 13209.5 (2526) | (12, 1000): 3396 | 60% |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 8.7 | 143.3 | 16.1 | 177.5 | 3 | 34631.2 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 3 | 219.8 | 1907.0 | 142.7 | 2311.2 | 6 | 34899.9 | 948 | hand-written | – | – | – | – | 0.0e0 (hand-written) |
| 4 | 1682.0 | 16383.7 | 2644.5 | 21007.2 | 30 | 15731.8 | 948 | library | – | – | – | – | 0.0e0 (hand-written) |

#### C3.2 cube, p = 16 (n = 289), metal

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 406.7 | 1272 | 8.9 | – | (16, 1000): 2146 | 59% |
|  | library rejected: [316, 32, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | library (3) | 45% | 5182.2 | 1720 | 12.0 | 3387.8 (2631) | (16, 1000): 3894 | 44% |
| 4 | 4096 | 584136 | 316 | 1849 / 3584 | library (17) | 65% | 40020.3 | 2438 | 17.0 | 34269.4 (2847) | (16, 1000): 3894 | 63% |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 13.8 | 406.7 | 16.1 | 440.6 | 3 | 62443.0 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 3 | 352.2 | 5182.2 | 257.1 | 5904.5 | 9 | 63820.8 | 948 | hand-written | – | – | – | – | 0.0e0 (hand-written) |
| 4 | 2803.2 | 40020.3 | 7180.1 | 49823.7 | 51 | 35466.7 | 948 | library | – | – | – | – | 0.0e0 (hand-written) |

#### C3.2 cube: the M2L stage (every V level)

| p | levels | pairs | device level calls µs | of which GEMM / gather / accumulate | launches | (A) µs | (A) launches | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3 | 640584 | 598.1 | 43% / 18% / 29% | 9 | 24900.0 | 2844 | 35447.2 | 4539.7 | 59.3 | 7.6 |
| 8 | 3 | 640584 | 8418.1 | 73% / 13% / 12% | 24 | 23810.6 | 2844 | 477141.3 | 108734.4 | 56.7 | 12.9 |
| 12 | 3 | 640584 | 23496.0 | 78% / 8% / 12% | 39 | 85262.8 | 2844 | – | – | – | – |
| 16 | 3 | 640584 | 56168.8 | 81% / 6% / 13% | 63 | 161730.5 | 2844 | – | – | – | – |

#### Plummer sphere, p = 3 (n = 16), metal

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 5.1 | 309 | 2.2 | – | (4, 1000): 239 | 129% |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 25.0 | 1092 | 7.6 | – | (4, 1000): 239 | 457% |
| 4 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 29.2 | 934 | 6.5 | – | (4, 1000): 239 | 391% |
| 5 | 936 | 115008 | 316 | 364 / 704 | hand-written (1) | 100% | 58.7 | 1003 | 7.0 | – | (4, 1000): 239 | 420% |
| 6 | 1624 | 215160 | 316 | 681 / 1280 | hand-written (1) | 100% | 116.5 | 945 | 6.6 | – | (4, 1000): 239 | 396% |
| 7 | 2416 | 335952 | 316 | 1063 / 1976 | hand-written (1) | 100% | 175.4 | 981 | 6.9 | – | (4, 1000): 239 | 410% |
| 8 | 416 | 37794 | 316 | 120 / 248 | hand-written (1) | 100% | 21.7 | 893 | 6.2 | – | (4, 1000): 239 | 374% |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 2.6 | 5.1 | 15.9 | 24.4 | 3 | 8140.6 | 948 | hand-written | 194.4 | 117.4 | 8.0 | 4.8 | 6.5e-7 (host) |
| 3 | 12.1 | 25.0 | 48.6 | 86.7 | 3 | 8088.8 | 948 | hand-written | 2977.2 | 409.0 | 34.4 | 4.7 | 6.6e-7 (host) |
| 4 | 14.5 | 29.2 | 48.4 | 86.7 | 3 | 8178.0 | 948 | hand-written | 2963.7 | 400.2 | 34.2 | 4.6 | 6.3e-7 (host) |
| 5 | 22.7 | 58.7 | 51.4 | 130.0 | 3 | 8282.1 | 948 | hand-written | 6382.2 | 782.7 | 49.1 | 6.0 | 7.9e-7 (host) |
| 6 | 41.3 | 116.5 | 66.6 | 192.2 | 3 | 8571.7 | 948 | hand-written | 12318.4 | 1339.7 | 64.1 | 7.0 | 8.0e-7 (host) |
| 7 | 61.1 | 175.4 | 67.7 | 298.4 | 3 | 8135.7 | 948 | hand-written | 18642.0 | 2040.6 | 62.5 | 6.8 | 8.9e-7 (host) |
| 8 | 10.3 | 21.7 | 45.6 | 71.6 | 3 | 8642.4 | 948 | hand-written | 2114.3 | 357.2 | 29.5 | 5.0 | 7.9e-7 (host) |

#### Plummer sphere, p = 8 (n = 81), metal

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 36.0 | 1127 | 7.9 | – | (8, 1000): 625 | 180% |
|  | library rejected: [316, 32, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | library (1) | 44% | 1343.2 | 521 | 3.6 | 478.6 (1463) | (8, 1000): 2045 | 25% |
| 4 | 512 | 53352 | 316 | 169 / 384 | library (1) | 44% | 1015.4 | 689 | 4.8 | 446.2 (1569) | (8, 1000): 2045 | 34% |
| 5 | 936 | 115008 | 316 | 364 / 704 | library (2) | 54% | 2081.5 | 725 | 5.1 | 1019.4 (1480) | (8, 1000): 2045 | 35% |
| 6 | 1624 | 215160 | 316 | 681 / 1280 | library (2) | 53% | 3886.6 | 726 | 5.1 | 2486.0 (1136) | (8, 1000): 2045 | 36% |
| 7 | 2416 | 335952 | 316 | 1063 / 1976 | library (4) | 54% | 6386.0 | 690 | 4.8 | 2866.1 (1538) | (8, 1000): 2045 | 34% |
| 8 | 416 | 37794 | 316 | 120 / 248 | library (1) | 48% | 741.0 | 669 | 4.7 | 311.0 (1595) | (8, 1000): 2045 | 33% |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 5.4 | 36.0 | 16.0 | 58.9 | 3 | 8559.0 | 948 | hand-written | 2130.5 | 370.3 | 36.2 | 6.3 | 7.7e-7 (host) |
| 3 | 192.0 | 1343.2 | 86.4 | 1211.1 | 3 | 8742.4 | 948 | hand-written | 36100.4 | 7771.2 | 29.8 | 6.4 | 1.2e-6 (host) |
| 4 | 141.0 | 1015.4 | 86.7 | 1134.6 | 3 | 8432.6 | 948 | hand-written | 36032.2 | 7889.4 | 31.8 | 7.0 | 1.1e-6 (host) |
| 5 | 296.5 | 2081.5 | 152.0 | 2611.6 | 6 | 7326.4 | 948 | library | 78139.9 | 19419.0 | 29.9 | 7.4 | 1.3e-6 (host) |
| 6 | 732.4 | 3886.6 | 298.7 | 5273.4 | 6 | 7144.3 | 948 | library | 153054.2 | 36589.1 | 29.0 | 6.9 | 1.3e-6 (host) |
| 7 | 811.1 | 6386.0 | 507.8 | 7468.6 | 12 | 7182.7 | 948 | library | 229397.0 | 57104.6 | 30.7 | 7.6 | 1.3e-6 (host) |
| 8 | 75.7 | 741.0 | 81.1 | 872.6 | 3 | 8488.8 | 948 | hand-written | 25658.7 | 6407.0 | 29.4 | 7.3 | 1.1e-6 (host) |

#### Plummer sphere, p = 12 (n = 169), metal

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 207.4 | 853 | 6.0 | – | (12, 1000): 1426 | 60% |
|  | library rejected: [316, 32, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | library (2) | 48% | 3491.0 | 873 | 6.1 | 2161.5 (1410) | (12, 1000): 3396 | 26% |
| 4 | 512 | 53352 | 316 | 169 / 384 | library (2) | 48% | 3054.6 | 998 | 7.0 | 1806.7 (1687) | (12, 1000): 3396 | 29% |
| 5 | 936 | 115008 | 316 | 364 / 704 | library (3) | 55% | 5918.9 | 1110 | 7.8 | 3685.2 (1783) | (12, 1000): 3396 | 33% |
| 6 | 1624 | 215160 | 316 | 681 / 1280 | library (5) | 54% | 9973.3 | 1232 | 8.6 | 6921.8 (1776) | (12, 1000): 3396 | 36% |
| 7 | 2416 | 335952 | 316 | 1063 / 1976 | library (6) | 57% | 15243.6 | 1259 | 8.8 | 10827.6 (1772) | (12, 1000): 3396 | 37% |
| 8 | 416 | 37794 | 316 | 120 / 248 | library (1) | 48% | 2358.4 | 915 | 6.4 | 1320.9 (1634) | (12, 1000): 3396 | 27% |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 9.3 | 207.4 | 16.7 | 173.3 | 3 | 34424.2 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 3 | 220.8 | 3491.0 | 159.3 | 3490.1 | 6 | 34776.8 | 948 | hand-written | – | – | – | – | 0.0e0 (hand-written) |
| 4 | 222.3 | 3054.6 | 154.2 | 3968.3 | 6 | 34931.1 | 948 | hand-written | – | – | – | – | 0.0e0 (hand-written) |
| 5 | 407.5 | 5918.9 | 357.4 | 7346.2 | 9 | 7301.0 | 948 | library | – | – | – | – | 0.0e0 (hand-written) |
| 6 | 1169.0 | 9973.3 | 763.9 | 14257.6 | 15 | 8496.1 | 948 | library | – | – | – | – | 0.0e0 (hand-written) |
| 7 | 1642.5 | 15243.6 | 1282.5 | 19777.4 | 18 | 12644.7 | 948 | library | – | – | – | – | 0.0e0 (hand-written) |
| 8 | 203.1 | 2358.4 | 85.3 | 2776.1 | 3 | 34844.6 | 948 | hand-written | – | – | – | – | 0.0e0 (hand-written) |

#### Plummer sphere, p = 16 (n = 289), metal

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 513.4 | 1007 | 7.0 | – | (16, 1000): 2146 | 47% |
|  | library rejected: [316, 32, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | library (3) | 45% | 8473.8 | 1052 | 7.4 | 5132.2 (1736) | (16, 1000): 3894 | 27% |
| 4 | 512 | 53352 | 316 | 169 / 384 | library (3) | 45% | 9170.7 | 972 | 6.8 | 5205.8 (1712) | (16, 1000): 3894 | 25% |
| 5 | 936 | 115008 | 316 | 364 / 704 | library (4) | 53% | 16751.3 | 1147 | 8.0 | 12083.8 (1590) | (16, 1000): 3894 | 29% |
| 6 | 1624 | 215160 | 316 | 681 / 1280 | library (7) | 63% | 25928.3 | 1386 | 9.7 | 19526.5 (1841) | (16, 1000): 3894 | 36% |
| 7 | 2416 | 335952 | 316 | 1063 / 1976 | library (9) | 66% | 41175.2 | 1363 | 9.5 | 28843.9 (1946) | (16, 1000): 3894 | 35% |
| 8 | 416 | 37794 | 316 | 120 / 248 | library (2) | 48% | 6808.2 | 927 | 6.5 | 3417.8 (1847) | (16, 1000): 3894 | 24% |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 14.8 | 513.4 | 17.8 | 492.9 | 3 | 61923.1 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 3 | 547.4 | 8473.8 | 293.2 | 10349.6 | 9 | 63699.8 | 948 | hand-written | – | – | – | – | 0.0e0 (hand-written) |
| 4 | 539.6 | 9170.7 | 274.5 | 9567.6 | 9 | 63727.1 | 948 | hand-written | – | – | – | – | 0.0e0 (hand-written) |
| 5 | 1031.4 | 16751.3 | 592.0 | 25562.0 | 12 | 10651.5 | 948 | library | – | – | – | – | 0.0e0 (hand-written) |
| 6 | 1646.1 | 25928.3 | 1494.9 | 30169.5 | 21 | 19081.3 | 948 | library | – | – | – | – | 0.0e0 (hand-written) |
| 7 | 2228.6 | 41175.2 | 2711.7 | 45422.8 | 27 | 29831.7 | 948 | library | – | – | – | – | 0.0e0 (hand-written) |
| 8 | 338.0 | 6808.2 | 169.3 | 7912.0 | 6 | 62982.0 | 948 | hand-written | – | – | – | – | 0.0e0 (hand-written) |

#### Plummer sphere: the M2L stage (every V level)

| p | levels | pairs | device level calls µs | of which GEMM / gather / accumulate | launches | (A) µs | (A) launches | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 7 | 813714 | 890.0 | 49% / 19% / 39% | 21 | 58039.3 | 6636 | 45592.2 | 5446.9 | 51.2 | 6.1 |
| 8 | 7 | 813714 | 18630.6 | 83% / 12% / 7% | 36 | 55876.2 | 6636 | 560512.8 | 135550.7 | 30.1 | 7.3 |
| 12 | 7 | 813714 | 51789.1 | 78% / 7% / 5% | 60 | 167418.4 | 6636 | – | – | – | – |
| 16 | 7 | 813714 | 129476.4 | 84% / 5% / 4% | 87 | 311896.5 | 6636 | – | – | – | – |

#### The gate: the GEMM alone on the spike's shapes (one table, B columns)

| p | B | GEMM of the rule | µs | GFLOP/s | % peak | spike GFLOP/s | of the spike | gate (>= 80%) | hand-written µs | GFLOP/s | spike hand-written | of it |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 1000 | hand-written | 5.4 | 233 | 1.6 | 239 | 97% | met | 5.4 | 233 | 239 | 97% |
| 4 | 10000 | hand-written | 17.9 | 698 | 4.9 | 1431 | 49% | not met | 17.9 | 698 | 1431 | 49% |
| 4 | 100000 | hand-written | 156.0 | 801 | 5.6 | 3078 | 26% | not met | 156.0 | 801 | 3078 | 26% |
| 8 | 1000 | library | 8.1 | 1627 | 11.4 | 2045 | 80% | not met | 17.9 | 733 | 625 | 117% |
| 8 | 10000 | library | 70.4 | 1864 | 13.0 | 3792 | 49% | not met | 85.4 | 1536 | 3585 | 43% |
| 8 | 100000 | library | 753.8 | 1741 | 12.2 | 4053 | 43% | not met | 764.7 | 1716 | 2921 | 59% |
| 12 | 1000 | library | 25.5 | 2237 | 15.6 | 3396 | 66% | not met | 62.9 | 908 | 1426 | 64% |
| 12 | 10000 | library | 235.8 | 2422 | 16.9 | 4539 | 53% | not met | 293.8 | 1944 | 3903 | 50% |
| 12 | 100000 | library | 2285.8 | 2499 | 17.5 | 3955 | 63% | not met | 2841.1 | 2011 | 3082 | 65% |
| 16 | 1000 | library | 74.2 | 2252 | 15.7 | 3894 | 58% | not met | 126.7 | 1318 | 2146 | 61% |
| 16 | 10000 | library | 662.0 | 2523 | 17.6 | 4475 | 56% | not met | 773.9 | 2158 | 3572 | 60% |
| 16 | 100000 | library | 6527.1 | 2559 | 17.9 | 4204 | 61% | not met | 7100.8 | 2352 | 3310 | 71% |

#### The scratch budget: the cube's level 4 at p = 8 (584136 pairs)

| budget MB | GEMM | chunks | scratch MB | level call µs | launches |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | library | 66 | 7.8 | 14724.5 | 198 |
| 16 | library | 34 | 15.6 | 11470.6 | 102 |
| 32 | library | 18 | 31.1 | 11189.9 | 54 |
| 64 | library | 9 | 63.1 | 10001.3 | 27 |
| 128 | library | 6 | 126.7 | 11480.7 | 18 |
| 256 | library | 3 | 254.7 | 11555.3 | 9 |
| 512 | library | 2 | 511.6 | 11795.5 | 6 |

#### The orientation of X and Y (T12): box-major against coefficient-major

Whole level call (gather, GEMM, reduction per chunk), µs; `cm/bm` < 1 means the coefficient-major layout is faster. Library columns where the library takes the level's shapes in that orientation (f32, p ≥ 8, a GPU).

| problem | p | level | pairs | hand bm | hand cm | cm/bm | library bm | library cm | cm/bm | best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C3.2 cube | 3 | 2 | 3096 | 24.0 | 27.2 | 1.13 | – | – | – | hand bm |
| C3.2 cube | 3 | 3 | 53352 | 86.4 | 103.2 | 1.19 | – | – | – | hand bm |
| C3.2 cube | 3 | 4 | 584136 | 724.0 | 1346.4 | 1.86 | – | – | – | hand bm |
| Plummer sphere | 3 | 2 | 3096 | 26.7 | 27.1 | 1.01 | – | – | – | hand bm |
| Plummer sphere | 3 | 3 | 53352 | 86.5 | 104.4 | 1.21 | – | – | – | hand bm |
| Plummer sphere | 3 | 4 | 53352 | 86.5 | 103.8 | 1.20 | – | – | – | hand bm |
| Plummer sphere | 3 | 5 | 115008 | 135.9 | 164.3 | 1.21 | – | – | – | hand bm |
| Plummer sphere | 3 | 6 | 215160 | 220.1 | 278.2 | 1.26 | – | – | – | hand bm |
| Plummer sphere | 3 | 7 | 335952 | 349.9 | 459.5 | 1.31 | – | – | – | hand bm |
| Plummer sphere | 3 | 8 | 37794 | 78.3 | 91.9 | 1.17 | – | – | – | hand bm |
| C3.2 cube | 6 | 2 | 3096 | 37.5 | 44.8 | 1.19 | – | – | – | hand bm |
| C3.2 cube | 6 | 3 | 53352 | 347.0 | 390.1 | 1.12 | – | – | – | hand bm |
| C3.2 cube | 6 | 4 | 584136 | 2956.4 | 4838.0 | 1.64 | – | – | – | hand bm |
| Plummer sphere | 6 | 2 | 3096 | 39.3 | 44.6 | 1.14 | – | – | – | hand bm |
| Plummer sphere | 6 | 3 | 53352 | 274.1 | 384.2 | 1.40 | – | – | – | hand bm |
| Plummer sphere | 6 | 4 | 53352 | 307.3 | 369.1 | 1.20 | – | – | – | hand bm |
| Plummer sphere | 6 | 5 | 115008 | 641.0 | 794.0 | 1.24 | – | – | – | hand bm |
| Plummer sphere | 6 | 6 | 215160 | 1217.5 | 1635.2 | 1.34 | – | – | – | hand bm |
| Plummer sphere | 6 | 7 | 335952 | 2002.3 | 2872.0 | 1.43 | – | – | – | hand bm |
| Plummer sphere | 6 | 8 | 37794 | 235.0 | 270.4 | 1.15 | – | – | – | hand bm |
| C3.2 cube | 8 | 2 | 3096 | 58.6 | 65.5 | 1.12 | – | – | – | hand bm |
| C3.2 cube | 8 | 3 | 53352 | 549.7 | 731.9 | 1.33 | 1256.3 | 1023.5 | 0.81 | hand bm |
| C3.2 cube | 8 | 4 | 584136 | 6678.5 | 8726.1 | 1.31 | 11524.9 | 10522.9 | 0.91 | hand bm |
| Plummer sphere | 8 | 2 | 3096 | 65.0 | 65.4 | 1.01 | – | – | – | hand bm |
| Plummer sphere | 8 | 3 | 53352 | 648.6 | 706.9 | 1.09 | 1239.6 | 1105.6 | 0.89 | hand bm |
| Plummer sphere | 8 | 4 | 53352 | 608.8 | 707.5 | 1.16 | 1262.8 | 1129.3 | 0.89 | hand bm |
| Plummer sphere | 8 | 5 | 115008 | 1408.4 | 1595.5 | 1.13 | 2535.9 | 2245.7 | 0.89 | hand bm |
| Plummer sphere | 8 | 6 | 215160 | 2656.0 | 3495.8 | 1.32 | 4448.9 | 4066.8 | 0.91 | hand bm |
| Plummer sphere | 8 | 7 | 335952 | 4082.6 | 5030.4 | 1.23 | 7149.1 | 6047.2 | 0.85 | hand bm |
| Plummer sphere | 8 | 8 | 37794 | 433.9 | 527.0 | 1.21 | 833.5 | 754.1 | 0.90 | hand bm |
| C3.2 cube | 12 | 2 | 3096 | 177.9 | 216.5 | 1.22 | – | – | – | hand bm |
| C3.2 cube | 12 | 3 | 53352 | 2334.4 | 2579.6 | 1.11 | 3597.0 | 3484.0 | 0.97 | hand bm |
| C3.2 cube | 12 | 4 | 584136 | 23448.0 | 26323.3 | 1.12 | 32461.7 | 28231.0 | 0.87 | hand bm |
| Plummer sphere | 12 | 2 | 3096 | 197.6 | 246.6 | 1.25 | – | – | – | hand bm |
| Plummer sphere | 12 | 3 | 53352 | 2272.9 | 2647.0 | 1.16 | 3785.8 | 3326.0 | 0.88 | hand bm |
| Plummer sphere | 12 | 4 | 53352 | 2194.9 | 2725.7 | 1.24 | 3925.0 | 3244.3 | 0.83 | hand bm |
| Plummer sphere | 12 | 5 | 115008 | 5124.4 | 5338.0 | 1.04 | 6912.3 | 6611.8 | 0.96 | hand bm |
| Plummer sphere | 12 | 6 | 215160 | 8472.5 | 10404.2 | 1.23 | 13258.5 | 11558.2 | 0.87 | hand bm |
| Plummer sphere | 12 | 7 | 335952 | 13595.8 | 18664.3 | 1.37 | 20966.3 | 18921.2 | 0.90 | hand bm |
| Plummer sphere | 12 | 8 | 37794 | 1504.6 | 2089.7 | 1.39 | 2865.0 | 2609.5 | 0.91 | hand bm |
| C3.2 cube | 16 | 2 | 3096 | 454.3 | 705.1 | 1.55 | – | – | – | hand bm |
| C3.2 cube | 16 | 3 | 53352 | 6388.6 | 6619.6 | 1.04 | 9522.4 | 9189.1 | 0.96 | hand bm |
| C3.2 cube | 16 | 4 | 584136 | 60454.4 | 66811.0 | 1.11 | 76581.1 | 70850.8 | 0.93 | hand bm |
| Plummer sphere | 16 | 2 | 3096 | 578.2 | 632.7 | 1.09 | – | – | – | hand bm |
| Plummer sphere | 16 | 3 | 53352 | 6272.9 | 6738.6 | 1.07 | 10002.0 | 8750.8 | 0.87 | hand bm |
| Plummer sphere | 16 | 4 | 53352 | 5631.8 | 7213.8 | 1.28 | 9821.3 | 8875.3 | 0.90 | hand bm |
| Plummer sphere | 16 | 5 | 115008 | 13161.4 | 13801.4 | 1.05 | 18684.9 | 16073.1 | 0.86 | hand bm |
| Plummer sphere | 16 | 6 | 215160 | 23198.5 | 24824.0 | 1.07 | 30108.0 | 25510.0 | 0.85 | hand bm |
| Plummer sphere | 16 | 7 | 335952 | 35260.9 | 36520.7 | 1.04 | 45738.2 | 40065.0 | 0.88 | hand bm |
| Plummer sphere | 16 | 8 | 37794 | 4254.2 | 5041.7 | 1.19 | 7406.3 | 6426.6 | 0.87 | hand bm |

</details>

### `rotation_kernels --device metal`

<details><summary>Output</summary>

### The device rotation M2L against dense (Phase 4 T10, C4.6)

| item | value |
| --- | --- |
| machine | Apple M3 Max (16 physical, 16 logical) |
| target | aarch64-macos, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| device | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| problem | the C3.2 cube: N = 10⁵ uniform in a cube, a uniform level-4 tree; its V levels 2 to 4 |
| timing | median of 15 batches of >= 20 ms; launches queued between syncs, a warm-up launch and a sync first (compilation excluded) |
| flops | rotation (20/3)(p + 1)³ per pair (the spike's count), dense 2 (p + 1)⁴ per pair; peak 14300 GFLOP/s (M3 Max GPU f32, derived in the spike report, not measured) |
| host | `LaplaceOperator::m2l_pair` with the f32 rotation tables, 1 and 12 scoped threads |
| precision | f32 (f64 not timed: Metal has no f64, CUDA is type-checked only; the f64 comparison is a model) |

#### p = 2 (n = 9), metal, rotation layout cube (32 units)

| level | boxes | pairs | rotation µs | ns/pair | GFLOP/s (model) | % peak | dense level call µs (chunks) | ns/pair | of which GEMM µs | GFLOP/s | library call µs | rotation / dense best | relative L2 rotation – dense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 3096 | 83.3 | 26.92 | 7 | 0.05 | 23.0 (3) | 7.44 | 3.5 | 143 | – | 3.62 | 1.7e-7 |
| 3 | 512 | 53352 | 253.4 | 4.75 | 38 | 0.27 | 72.2 (3) | 1.35 | 18.3 | 472 | – | 3.51 | 2.0e-7 |
| 4 | 4096 | 584136 | 1153.2 | 1.97 | 91 | 0.64 | 337.9 (3) | 0.58 | 198.4 | 477 | – | 3.41 | 1.9e-7 |

#### p = 4 (n = 25), metal, rotation layout cube (32 units)

| level | boxes | pairs | rotation µs | ns/pair | GFLOP/s (model) | % peak | dense level call µs (chunks) | ns/pair | of which GEMM µs | GFLOP/s | library call µs | rotation / dense best | relative L2 rotation – dense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 3096 | 122.9 | 39.70 | 21 | 0.15 | 28.1 (3) | 9.09 | 9.3 | 417 | – | 4.37 | 1.8e-7 |
| 3 | 512 | 53352 | 392.5 | 7.36 | 113 | 0.79 | 147.6 (3) | 2.77 | 113.7 | 586 | – | 2.66 | 1.9e-7 |
| 4 | 4096 | 584136 | 1689.1 | 2.89 | 288 | 2.02 | 1585.3 (3) | 2.71 | 1044.7 | 699 | – | 1.07 | 2.1e-7 |

#### p = 6 (n = 49), metal, rotation layout cube (64 units)

| level | boxes | pairs | rotation µs | ns/pair | GFLOP/s (model) | % peak | dense level call µs (chunks) | ns/pair | of which GEMM µs | GFLOP/s | library call µs | rotation / dense best | relative L2 rotation – dense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 3096 | 170.0 | 54.92 | 42 | 0.29 | 37.4 (3) | 12.09 | 17.8 | 833 | – | 4.54 | 8.9e-8 |
| 3 | 512 | 53352 | 574.8 | 10.77 | 212 | 1.48 | 339.2 (3) | 6.36 | 230.0 | 1114 | – | 1.69 | 1.2e-7 |
| 4 | 4096 | 584136 | 4298.0 | 7.36 | 311 | 2.17 | 3357.0 (6) | 5.75 | 2641.4 | 1062 | – | 1.28 | 1.2e-7 |

#### p = 8 (n = 81), metal, rotation layout cube (96 units)

| level | boxes | pairs | rotation µs | ns/pair | GFLOP/s (model) | % peak | dense level call µs (chunks) | ns/pair | of which GEMM µs | GFLOP/s | library call µs | rotation / dense best | relative L2 rotation – dense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 3096 | 217.2 | 70.14 | 69 | 0.48 | 58.2 (3) | 18.80 | 38.0 | 1068 | – | 3.73 | 1.0e-7 |
| 3 | 512 | 53352 | 992.3 | 18.60 | 261 | 1.83 | 737.8 (3) | 13.83 | 470.1 | 1489 | 1376.3 | 1.34 | 1.1e-7 |
| 4 | 4096 | 584136 | 9409.3 | 16.11 | 302 | 2.11 | 6635.0 (9) | 11.36 | 5433.6 | 1411 | 16849.3 | 1.42 | 1.1e-7 |

#### p = 12 (n = 169), metal, rotation layout cube (192 units)

| level | boxes | pairs | rotation µs | ns/pair | GFLOP/s (model) | % peak | dense level call µs (chunks) | ns/pair | of which GEMM µs | GFLOP/s | library call µs | rotation / dense best | relative L2 rotation – dense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 3096 | 378.2 | 122.17 | 120 | 0.84 | 176.5 (3) | 57.01 | 212.1 | 834 | – | 2.14 | 1.0e-7 |
| 3 | 512 | 53352 | 2990.1 | 56.04 | 261 | 1.83 | 2226.9 (3) | 41.74 | 2163.3 | 1409 | 3412.5 | 1.34 | 9.5e-8 |
| 4 | 4096 | 584136 | 29900.5 | 51.19 | 286 | 2.00 | 24232.2 (18) | 41.48 | 19654.6 | 1698 | 33423.0 | 1.23 | 1.0e-7 |

#### p = 16 (n = 289), metal, rotation layout cube (320 units)

| level | boxes | pairs | rotation µs | ns/pair | GFLOP/s (model) | % peak | dense level call µs (chunks) | ns/pair | of which GEMM µs | GFLOP/s | library call µs | rotation / dense best | relative L2 rotation – dense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 3096 | 573.3 | 185.18 | 177 | 1.24 | 447.1 (3) | 144.41 | 638.3 | 810 | – | 1.28 | 1.2e-7 |
| 3 | 512 | 53352 | 8554.3 | 160.34 | 204 | 1.43 | 5583.8 (3) | 104.66 | 5381.1 | 1656 | 9867.3 | 1.53 | 1.1e-7 |
| 4 | 4096 | 584136 | 80519.5 | 137.84 | 238 | 1.66 | 62924.5 (33) | 107.72 | 51483.9 | 1895 | 81061.3 | 1.28 | 1.1e-7 |

#### Rotation against dense across p (every V level of the cube, f32)

| p | pairs | rotation µs | ns/pair | % peak (model flops) | dense (hand-written) µs | ns/pair | dense, best µs | rotation / dense best | break-even, M3 f32 measured | break-even, A100/H100 f64 model |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | 640584 | 1490.0 | 2.33 | 0.54% | 433.2 | 0.68 | 433.2 | 3.44 | – | – |
| 4 | 640584 | 2204.5 | 3.44 | 1.69% | 1761.0 | 2.75 | 1761.0 | 1.25 | 13.9% (rotation below) | – |
| 6 | 640584 | 5042.8 | 7.87 | 2.03% | 3733.6 | 5.83 | 3733.6 | 1.35 | – | – |
| 8 | 640584 | 10618.8 | 16.58 | 2.05% | 7431.0 | 11.60 | 7431.0 | 1.43 | 9.8% (rotation below) | 13.2% (rotation below) |
| 12 | 640584 | 33268.8 | 51.94 | 1.97% | 26635.6 | 41.58 | 26635.6 | 1.25 | 7.1% (rotation below) | 5.4% (rotation below) |
| 16 | 640584 | 89647.1 | 139.95 | 1.64% | 68955.4 | 107.64 | 68955.4 | 1.30 | 4.8% (rotation below) | 3.8% (rotation below) |

#### The M2L stage at p = 8: device rotation against host rotation (f32)

| level | pairs | device µs | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 device – host |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3096 | 217.4 | 2799.5 | 413.3 | 12.9 | 1.9 | 7.4e-8 |
| 3 | 53352 | 836.2 | 48421.2 | 5036.3 | 57.9 | 6.0 | 9.0e-8 |
| 4 | 584136 | 8913.3 | 529056.8 | 56437.7 | 59.4 | 6.3 | 9.2e-8 |
| all | 640584 | 9966.8 | 580277.5 | 61887.3 | 58.2 | 6.2 | |

#### Rotation layouts at p = 8, level 4 (584136 pairs)

| layout | µs | ns/pair | % peak (model) | bit for bit the default |
| --- | ---: | ---: | ---: | --- |
| cube (96 units) | 7600.8 | 13.01 | 2.61 | yes |
| cube (128 units) | 9124.2 | 15.62 | 2.18 | yes |
| cube (192 units) | 12858.3 | 22.01 | 1.54 | yes |
| cube (32 units) | 8391.0 | 14.36 | 2.37 | yes |

</details>

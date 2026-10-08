# Phase 4S benchmarks on the NVIDIA GH200 (T8)

The device FMM on locust's GH200 in f32 and f64: the `nd-fmm-bench` sweep, the Phase 4
CUDA run of fmm-validate/results/phase4-m3max.md §4, the device leaf-size rule on CUDA,
and the comparison with the Grace host and with the M3 Max
(docs/phase4s/T8-benchmarks.md). Run on 2026-10-07, 17:36–20:13 UTC.

**Every number below is measured on locust: CUDA (the H100) for the device, NEON on the
72 Grace cores for the host (`P2pChoice::Auto`).** A figure labelled *model* or
*datasheet* is not a measurement; a figure labelled *T7* was measured by Phase 4S T7 on
the same machine and code on the same day (docs/design/device-path.md §18.2) and not
repeated here. M3 Max figures are from phase4-m3max.md (Phase 4 T13, 2026-10-04): a
different run on a different machine.

**Not measured:**
- multi-GPU and multi-rank device runs (Phase 5 and later; locust has one GPU);
- an x86_64 host: no x86_64 machine has been timed in any phase;
- FMM3D, the external baseline (Phase 4 decision 9: a later phase);
- the f64 `device_fmm` run at its default sizes (the cube at N = 10⁶ against the host at
  one thread up to p = 18, about 5 hours by T4's estimate): dropped on request. The
  f64 comparison at N = 10⁶ comes from `nd-fmm-bench` (Sections 1 and 2) instead;
- the kernel examples `m2l_kernels --precision f64`, `rotation_kernels` and
  `p2p_kernels` on CUDA: dropped on request, because T7 measured them on the same code
  the same day; Section 4 cites T7's numbers;
- per-kind times of the host at one thread (dropped on request: `--kinds off`), and of
  the host at 72 threads at N = 10⁷;
- hardware counters: `ncu` is refused on locust (`ERR_NVGPUCTRPERM`, T7); Section 7 uses
  an `nsys` timeline.

## Setup

| item | value |
| --- | --- |
| machine | locust (`locust.rc.ucl.ac.uk`), NVIDIA GH200 480GB: 72 Arm Neoverse-V2 (Grace) cores, one thread per core, 572 GB LPDDR5X; one H100 (compute capability 9.0, 132 SMs, 96 GB HBM3, MIG off); RHEL 9.3, kernel `5.14.0-362.18.1.el9_3.aarch64+64k` (64 KiB pages); tools/gh200/machine.md |
| software | the spack environment of tools/gh200/ (spack v1.2.2, spack-packages `d4f7c711`; `spack.lock` 55 specs, SHA-256 `fb706a3c30f2ac363321ab721363726fbdc377e7d4fb4b85fa03040e366480cc`): Open MPI 5.0.10, OpenBLAS 0.3.33 (`threads=none`), CUDA toolkit 12.6.3 (`nvcc` V12.6.85); NVIDIA driver 565.57.01 (CUDA 12.7); rustc 1.99.0 (b940084d7 2026-09-28), release build, linked by rust-lld; CubeCL 0.11.0-pre.4 through its default LLVM NVPTX path (`cubek-matmul` 0.3.0-pre.4) |
| source | `d3ed6b4` on `tbetcke/phase4s_t8` (main after T7, PR #68), synced by tools/gh200/sync.sh; no code change in T8 |
| device | `cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64`; plane size 32, 232,448 B shared memory and 1,024 units per cube |
| peaks (*datasheet*, at the 1,980 MHz maximum SM clock) | f32 67 TFLOP/s and f64 34 TFLOP/s without tensor cores (CubeCL has no FP64 MMA, and the input-precision guard keeps f32 off TF32), HBM3 4.0 TB/s (`nd_fmm_validate::peaks`). P2P peak model (*model*, T7, as device-path.md §13.4): 3,345 / 2,230 Gpairs/s in f32 and 1,673 / 1,115 in f64 (φ / φ and ∇φ). M3 Max: Phase 4's values (14.3 TFLOP/s f32, derived; 400 GB/s) |
| clocks | not locked (no administrator access). `nvidia-smi -q -d CLOCK` before and after the runs: SM 345 MHz idle, application and maximum clocks 1,980 MHz (SM) and 2,619 MHz (memory) |
| environment | `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `GOTO_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` all 1 (tools/bench/run.sh, and by hand for the examples); `RAYON_NUM_THREADS=1` for `m2l_kernels` and the GEMM spike; MPI at `Funneled`, one rank; `HOME`, caches and `TMPDIR` under `/data/ucahtbe` (tools/gh200/env.sh) |
| host comparisons | the Phase 3S host path, `P2pChoice::Auto` (NEON), at 1 thread and at 72 threads (every core); `M2lStrategy::Auto` on the host (`Dense` to p = 8, `Rotation` above) |
| load | checked before and after every step (`nvidia-smi`, `uptime`, `top`; 46 records, kept on locust under `/data/ucahtbe/logs/t8/load/`): no other user's compute process on the GPU and 0% utilisation (at most 4 MiB in use) at every check; the only other user's process in `top` was an `nvitop` monitor at about 6% of one core; the one-minute load average 0.3–2.6 before each step, except right after this session's own 72-rank MPI runs and 72-thread steps (7.7–32 before a step, up to 55 right after one; the average still decaying). No step had to wait. The f64 tuned CUDA sweep was stopped once and rerun from an empty cache (a script change, not load); its first, partial output is not used |
| timing | `nd-fmm-bench`: compilation excluded by 2 warm-up evaluations, then 10 timed evaluations with kind timings off (min, median, mean, max, standard deviation); kind times from a second build with `KindTiming::Synchronous` (a sync after every level call), its own 2 warm-ups and 10 evaluations; the host at one thread 3 timed evaluations without warm-up or kinds, at 72 threads and N = 10⁷ without kinds. `device_fmm`: the median of 5 evaluations after the 8 error evaluations; stages from a synchronous-stages build. `autotune` and `m2l_kernels`: as their headers state. Tables through a table cache (loading timed in the build) |
| runs | one run of each command; the evaluation's standard deviation over the repeats is at most 5.0% of the mean on CUDA (f64, N = 10⁵, p = 3; at most 3.4% from N = 10⁶ on) and at most 3.0% on the host |

Commands (from the synced tree on locust, inside `tools/gh200/env.sh`; `T` the table cache,
`DIR` a scratch directory under `/data/ucahtbe/logs/t8`):

```sh
# nd-fmm-bench, CUDA: tuned (a fresh tuning cache per precision), then every fixed strategy at N = 10^6
tools/bench/run.sh --backend cuda --n 1e5,1e6,1e7 --precision f32 --degree 3,6,8 --tuning-cache DIR/f32 --table-cache T
tools/bench/run.sh --backend cuda --n 1e5,1e6,1e7 --precision f64 --degree 3,6,8,12,18 --tuning-cache DIR/f64 --table-cache T
tools/bench/run.sh --backend cuda --n 1e6 --precision f32 --degree 3,6,8 --strategy dense|classes|rotation --table-cache T
tools/bench/run.sh --backend cuda --n 1e6 --precision f64 --degree 3,6,8,12,18 --strategy dense|classes|rotation --table-cache T
# nd-fmm-bench, the host on Grace
tools/bench/run.sh --backend host --threads 72 --n 1e5,1e6 --precision f32|f64 --degree ... --table-cache T
tools/bench/run.sh --backend host --threads 72 --n 1e7 --precision f32|f64 --degree ... --kinds off --table-cache T
tools/bench/run.sh --backend host --threads 1 --n 1e5,1e6 --precision f32|f64 --degree ... --repeats 3 --warmup 0 --kinds off --table-cache T
# the Phase 4 CUDA run (phase4-m3max.md §4); f64 device_fmm with --large 0 (see "Not measured")
cargo run --release -p nd-fmm-validate --features cuda --example device_fmm -- --device cuda --precision f32 --tuning-cache DIR/tune --table-cache T
cargo run --release -p nd-fmm-validate --features cuda --example device_fmm -- --device cuda --precision f64 --large 0 --tuning-cache DIR/tune --table-cache T
cargo run --release -p nd-fmm-validate --features cuda --example autotune -- --device cuda --precision f32|f64 --tuning-cache DIR/autotune --table-cache T
RAYON_NUM_THREADS=1 cargo run --release -p nd-fmm-validate --features cuda --example m2l_kernels -- --device cuda --table-cache T
RAYON_NUM_THREADS=1 cargo run -p nd-fmm-spike-cubecl-gemm --release --no-default-features --features cuda -- --backends cuda
# the device leaf-size study at N = 10^6
cargo run --release -p nd-fmm-validate --features cuda --example device_fmm -- --device cuda --precision f32|f64 --part leaf --n 1000000 --large 0 --leaf-degrees 3,8 --table-cache T
```

The problems: `nd-fmm-bench`'s N points uniform in [0, 1]³ (seed `0xbe9c06`), sources
equal to targets, one charge vector, gradients on, `max_level` 16 and 64 points per leaf
(4,096 leaves at N = 10⁵, 32,768 at 10⁶, 262,214 on 8 levels at 10⁷ with 52.3 × 10⁶ V
pairs); and `device_fmm`'s Phase 4 problems (the cube [−1, 1)³ and the Plummer sphere,
a = 0.1, N = 10⁵, eight charge vectors; the cube at N = 10⁶).

## 1. `nd-fmm-bench` on CUDA: summary and kinds

### Evaluation time, the tuned strategy

`tools/bench/run.sh --backend cuda`, one tuning cache per precision, empty at the start
and shared by the three N in order: the N = 10⁵ builds tuned every decision (3.6–4.0 s,
17.5 s at f64 p = 18), and the N = 10⁶ and 10⁷ builds took the M2L strategy and the
smaller levels' GEMMs from the cache and tuned only their new, larger level sizes (1.4–12.3
s). So the strategy at N = 10⁶ and 10⁷ is the one tuned at N = 10⁵; the fixed-strategy
rows below and `device_fmm`'s cube at N = 10⁶ (a cache of its own, Section 3) test it
there. ms per evaluation over 10 timed evaluations; build in s, tuning included; errors:
relative L2 at 1,000 sampled targets against the f64 direct sum. Every row: bit-identical
across the repeats, one upload, one download and one sync per evaluation.

| precision | N | p | strategy | build (s) | min | median | mean | max | std | error φ | error ∇φ |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | 10⁵ | 3 | Dense | 3.98 | 1.475 | 1.572 | **1.559** | 1.599 | 0.041 | 3.39e-3 | 1.99e-3 |
| f32 | 10⁵ | 6 | Dense | 3.77 | 1.816 | 1.889 | **1.909** | 2.028 | 0.090 | 1.42e-4 | 1.30e-4 |
| f32 | 10⁵ | 8 | Dense | 4.02 | 2.558 | 2.657 | **2.648** | 2.712 | 0.053 | 2.12e-5 | 2.37e-5 |
| f32 | 10⁶ | 3 | Dense | 2.52 | 12.433 | 12.755 | **12.829** | 13.549 | 0.331 | 2.00e-3 | 3.80e-3 |
| f32 | 10⁶ | 6 | Dense | 2.63 | 15.667 | 15.917 | **16.019** | 17.498 | 0.542 | 8.53e-5 | 2.56e-4 |
| f32 | 10⁶ | 8 | Dense | 2.66 | 21.333 | 21.599 | **21.669** | 22.735 | 0.415 | 1.36e-5 | 4.59e-5 |
| f32 | 10⁷ | 3 | Dense | 19.2 | 219.081 | 220.777 | **221.470** | 224.153 | 1.777 | 1.87e-3 | 1.87e-3 |
| f32 | 10⁷ | 6 | Dense | 20.5 | 255.265 | 257.726 | **257.534** | 259.646 | 1.366 | 7.90e-5 | 1.23e-4 |
| f32 | 10⁷ | 8 | Dense | 22.5 | 312.733 | 315.430 | **315.042** | 317.466 | 1.640 | 1.21e-5 | 2.26e-5 |
| f64 | 10⁵ | 3 | Dense | 3.98 | 1.911 | 1.940 | **1.994** | 2.225 | 0.100 | 3.39e-3 | 1.99e-3 |
| f64 | 10⁵ | 6 | Dense | 3.75 | 2.497 | 2.555 | **2.544** | 2.566 | 0.025 | 1.42e-4 | 1.30e-4 |
| f64 | 10⁵ | 8 | Dense | 3.75 | 3.760 | 3.785 | **3.792** | 3.849 | 0.025 | 2.12e-5 | 2.37e-5 |
| f64 | 10⁵ | 12 | Dense | 4.15 | 8.475 | 8.492 | **8.499** | 8.545 | 0.023 | 8.07e-7 | 1.11e-6 |
| f64 | 10⁵ | 18 | Dense | 18.6 | 26.412 | 26.461 | **26.473** | 26.621 | 0.060 | 1.01e-8 | 2.37e-8 |
| f64 | 10⁶ | 3 | Dense | 2.54 | 19.892 | 20.560 | **20.504** | 20.903 | 0.341 | 2.00e-3 | 3.80e-3 |
| f64 | 10⁶ | 6 | Dense | 2.42 | 24.508 | 25.475 | **25.430** | 26.175 | 0.446 | 8.53e-5 | 2.56e-4 |
| f64 | 10⁶ | 8 | Dense | 2.50 | 33.878 | 35.268 | **35.055** | 35.370 | 0.457 | 1.36e-5 | 4.59e-5 |
| f64 | 10⁶ | 12 | Dense | 3.55 | 74.312 | 76.389 | **76.204** | 76.680 | 0.702 | 4.35e-7 | 2.17e-6 |
| f64 | 10⁶ | 18 | Dense | 10.8 | 241.604 | 243.452 | **243.242** | 243.825 | 0.633 | 6.32e-9 | 5.58e-8 |
| f64 | 10⁷ | 3 | Dense | 19.7 | 356.594 | 361.077 | **363.693** | 394.636 | 11.053 | 1.87e-3 | 1.87e-3 |
| f64 | 10⁷ | 6 | Dense | 21.6 | 412.692 | 415.426 | **415.423** | 417.725 | 1.807 | 7.90e-5 | 1.23e-4 |
| f64 | 10⁷ | 8 | Dense | 24.5 | 513.336 | 517.091 | **516.715** | 519.400 | 1.943 | 1.21e-5 | 2.26e-5 |
| f64 | 10⁷ | 12 | Dense | 24.5 | 1017.487 | 1022.384 | **1021.767** | 1026.457 | 2.890 | 4.40e-7 | 1.21e-6 |
| f64 | 10⁷ | 18 | Dense | 27.2 | 3073.488 | 3080.903 | **3080.447** | 3083.747 | 3.272 | 6.50e-9 | 3.01e-8 |

- **N = 10⁷ fits, in both precisions, up to f64 p = 18**: 3.8 GB of device memory at
  f32 p = 3, 6.4 GB at f64 p = 18, of 96 GB. Nothing was refused
  (`SettingsError::DeviceMemory` never occurred), so no smaller N was needed.
- **The errors equal the host's.** At N = 10⁵ and 10⁶ every φ and ∇φ error of the device
  equals the host's at 72 threads (Section 2) to the printed digits, except three f32 φ
  values: the host's 2.14e-5 against 2.12e-5 (N = 10⁵, p = 8), 8.54e-5 against 8.53e-5
  and 1.40e-5 against 1.36e-5 (N = 10⁶, p = 6 and 8), within the 5% of Phase 4.
- f64 costs 1.59–1.64× f32 on the device at N = 10⁶ and 10⁷ for p = 3–8, less than the
  2× of the peaks: much of an evaluation is not f64 arithmetic (below).

### Per operator kind, the tuned strategy

Mean ms per evaluation from the second, kind-timed build (`KindTiming::Synchronous`: a
sync after each level call, about 18 µs per call, T7). Other: the time outside the level
calls (loading the charges, the upload, the download, the output's scaling and order).
Evaluation: those evaluations' own mean. Shares of that mean.

| precision | N | p | P2M | M2M + L2L | M2L | P2L + M2P | L2P | P2P | other | evaluation | M2L share | other share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | 10⁵ | 3 | 0.03 | 0.29 | 0.37 | 0.00 | 0.15 | 0.28 | 0.79 | 1.92 | 19% | 41% |
| f32 | 10⁵ | 6 | 0.03 | 0.47 | 0.66 | 0.00 | 0.09 | 0.28 | 0.80 | 2.36 | 28% | 34% |
| f32 | 10⁵ | 8 | 0.03 | 0.56 | 1.22 | 0.00 | 0.22 | 0.28 | 0.84 | 3.20 | 38% | 26% |
| f32 | 10⁶ | 3 | 0.06 | 0.49 | 1.28 | 0.00 | 0.07 | 1.99 | 8.42 | 12.6 | 10% | 67% |
| f32 | 10⁶ | 6 | 0.07 | 0.53 | 4.36 | 0.00 | 0.23 | 2.14 | 8.47 | 16.2 | 27% | 52% |
| f32 | 10⁶ | 8 | 0.11 | 0.87 | 9.37 | 0.00 | 0.42 | 2.14 | 8.47 | 21.9 | 43% | 39% |
| f32 | 10⁷ | 3 | 0.39 | 0.96 | 9.72 | 0.31 | 0.44 | 24.1 | 184 | 223 | 4% | 82% |
| f32 | 10⁷ | 6 | 0.54 | 1.54 | 44.4 | 0.50 | 1.48 | 24.1 | 183 | 261 | 17% | 70% |
| f32 | 10⁷ | 8 | 0.96 | 2.33 | 96.3 | 0.43 | 3.18 | 24.1 | 189 | 322 | 30% | 59% |
| f64 | 10⁵ | 3 | 0.03 | 0.30 | 0.41 | 0.00 | 0.11 | 0.40 | 1.18 | 2.47 | 16% | 48% |
| f64 | 10⁵ | 6 | 0.03 | 0.53 | 0.82 | 0.00 | 0.17 | 0.40 | 1.25 | 3.27 | 25% | 38% |
| f64 | 10⁵ | 8 | 0.05 | 0.50 | 1.85 | 0.00 | 0.13 | 0.40 | 1.22 | 4.25 | 43% | 29% |
| f64 | 10⁵ | 12 | 0.12 | 1.03 | 5.73 | 0.00 | 0.61 | 0.56 | 1.26 | 9.56 | 60% | 13% |
| f64 | 10⁵ | 18 | 0.64 | 2.18 | 21.9 | 0.00 | 1.58 | 0.46 | 1.17 | 28.3 | 77% | 4% |
| f64 | 10⁶ | 3 | 0.08 | 0.46 | 1.64 | 0.00 | 0.19 | 3.14 | 13.0 | 19.1 | 9% | 68% |
| f64 | 10⁶ | 6 | 0.12 | 0.77 | 6.02 | 0.00 | 0.52 | 3.32 | 13.1 | 24.8 | 24% | 53% |
| f64 | 10⁶ | 8 | 0.27 | 1.03 | 14.9 | 0.00 | 0.68 | 3.21 | 13.2 | 34.5 | 43% | 38% |
| f64 | 10⁶ | 12 | 0.91 | 2.00 | 49.9 | 0.00 | 4.82 | 3.22 | 13.2 | 76.2 | 65% | 17% |
| f64 | 10⁶ | 18 | 5.54 | 4.76 | 200 | 0.00 | 12.5 | 3.21 | 13.5 | 244 | 82% | 6% |
| f64 | 10⁷ | 3 | 0.65 | 1.07 | 14.1 | 0.58 | 0.67 | 39.0 | 303 | 366 | 4% | 83% |
| f64 | 10⁷ | 6 | 0.97 | 2.16 | 65.5 | 0.65 | 2.88 | 38.9 | 299 | 419 | 16% | 71% |
| f64 | 10⁷ | 8 | 2.42 | 2.85 | 156 | 0.69 | 5.99 | 38.9 | 300 | 519 | 30% | 58% |
| f64 | 10⁷ | 12 | 8.80 | 7.49 | 602 | 0.89 | 48.7 | 38.9 | 298 | 1,023 | 59% | 29% |
| f64 | 10⁷ | 18 | 53.6 | 22.8 | 2,503 | 1.22 | 129 | 39.1 | 298 | 3,081 | 81% | 10% |

- **At N = 10⁶ and 10⁷ "other" dominates the device at low p**: 8.4 ms (f32) and
  13.0–13.5 ms (f64) per evaluation at N = 10⁶, 183–189 ms and 298–303 ms at N = 10⁷,
  the same at every p. That is 38–68% of an evaluation at N = 10⁶ and 58–83% at 10⁷ for
  p ≤ 8. It grows with N, not p: host work per point (Section 7).
- **M2L dominates the device's own work** from p = 6 on and is 65–82% of an f64
  evaluation from p = 12 at N = 10⁶ (59–82% over every N); P2P is the next kind at
  p ≤ 8 (2.0–2.1 ms in f32 and 3.1–3.3 ms in f64 at N = 10⁶, 24 / 39 ms at 10⁷), and at
  f64 p ≥ 12 L2P overtakes it (12.5 against 3.2 ms at p = 18, N = 10⁶).
- On the host at 72 threads (Section 2) the same split holds for M2L (49–90% from p = 6)
  but "other" is 10–21 ms at N = 10⁶ too: most of it is the evaluation's host part, shared
  by both paths, not the device transfer.

### Every fixed strategy at N = 10⁶ (CUDA)

Mean ms per evaluation; `Classes` runs as dense on the device (Phase 4 decision 5). M2L:
the M2L kind time under dense / under rotation, from the kind-timed builds.

| precision | p | tuned | Dense | Classes (as dense) | Rotation | rotation / dense | M2L dense / rotation (kinds) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f32 | 3 | 12.83 | 12.25 | 12.15 | 14.42 | 1.18 | 1.29 / 3.42 |
| f32 | 6 | 16.02 | 15.84 | 15.66 | 19.96 | 1.26 | 4.36 / 8.57 |
| f32 | 8 | 21.67 | 21.50 | 21.30 | 26.45 | 1.23 | 9.39 / 14.41 |
| f64 | 3 | 20.50 | 19.84 | 19.54 | 21.93 | 1.11 | 1.65 / 3.74 |
| f64 | 6 | 25.43 | 25.11 | 24.91 | 32.55 | 1.30 | 6.03 / 13.28 |
| f64 | 8 | 35.05 | 35.43 | 35.56 | 45.29 | 1.28 | 15.55 / 25.30 |
| f64 | 12 | 76.20 | 78.87 | 78.83 | 105.10 | 1.33 | 52.57 / 78.80 |
| f64 | 18 | 243.24 | 250.15 | 250.32 | 301.55 | 1.21 | 207.67 / 258.74 |

- **Dense is faster than rotation at every (precision, p)**, end to end by 1.11–1.33×
  and on the M2L kind alone by 1.25–2.65× (f64 p = 18: 208 against 259 ms).
- The tuned build is within −3.4% and +4.7% of the static rule's (`Dense`): at f32 and
  f64 p ≤ 6 within the run-to-run spread, at f64 p = 12 and 18 3% faster from its GEMM
  layouts. T7 measured 1.02–1.16× for tuning at N = 10⁶, p = 3–8.

## 2. The device against the host on Grace

`nd-fmm-bench`, the same problems on the host at 1 thread and 72 threads (the host's
`Auto`: `Dense` to p = 8, `Rotation` above). Mean ms per evaluation; the speed-ups are
the tuned device's over each. "Level calls only": the sum of the kind times of the host at
72 threads over the device's, both synchronous (the far and near field without "other").

| precision | N | p | host 1 thread | host 72 threads | CUDA tuned | x host 1 | x host 72 | CUDA level calls only: x host 72 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | 10⁵ | 3 | 92.2 | 4.09 | 1.56 | 59.2 | 2.62 | 2.33 |
| f32 | 10⁵ | 6 | 267 | 7.27 | 1.91 | 140 | 3.81 | 3.85 |
| f32 | 10⁵ | 8 | 540 | 13.0 | 2.65 | 204 | 4.91 | 4.82 |
| f32 | 10⁶ | 3 | 1,063 | 27.5 | 12.8 | 82.9 | 2.15 | 4.34 |
| f32 | 10⁶ | 6 | 2,766 | 58.8 | 16.0 | 173 | 3.67 | 6.42 |
| f32 | 10⁶ | 8 | 5,281 | 111 | 21.7 | 244 | 5.10 | 7.56 |
| f32 | 10⁷ | 3 | – | 338 | 221 | – | 1.52 | – |
| f32 | 10⁷ | 6 | – | 629 | 258 | – | 2.44 | – |
| f32 | 10⁷ | 8 | – | 1,095 | 315 | – | 3.47 | – |
| f64 | 10⁵ | 3 | 198 | 5.67 | 1.99 | 99.4 | 2.85 | 3.35 |
| f64 | 10⁵ | 6 | 467 | 11.4 | 2.54 | 183 | 4.47 | 4.94 |
| f64 | 10⁵ | 8 | 962 | 22.8 | 3.79 | 254 | 6.01 | 7.26 |
| f64 | 10⁵ | 12 | 2,278 | 37.6 | 8.50 | 268 | 4.42 | 4.48 |
| f64 | 10⁵ | 18 | 6,364 | 103 | 26.5 | 240 | 3.91 | 3.78 |
| f64 | 10⁶ | 3 | 2,441 | 54.5 | 20.5 | 119 | 2.66 | 6.68 |
| f64 | 10⁶ | 6 | 5,059 | 109 | 25.4 | 199 | 4.27 | 8.26 |
| f64 | 10⁶ | 8 | 9,653 | 216 | 35.1 | 275 | 6.17 | 9.78 |
| f64 | 10⁶ | 12 | 22,138 | 355 | 76.2 | 291 | 4.66 | 5.48 |
| f64 | 10⁶ | 18 | 60,782 | 958 | 243 | 250 | 3.94 | 4.17 |
| f64 | 10⁷ | 3 | – | 689 | 364 | – | 1.90 | – |
| f64 | 10⁷ | 6 | – | 1,178 | 415 | – | 2.84 | – |
| f64 | 10⁷ | 8 | – | 2,145 | 517 | – | 4.15 | – |
| f64 | 10⁷ | 12 | – | 3,307 | 1,022 | – | 3.24 | – |
| f64 | 10⁷ | 18 | – | 8,718 | 3,080 | – | 2.83 | – |

- **Against one Grace core: 59–244× (f32) and 99–291× (f64)** at N = 10⁵ and 10⁶. The
  one-thread host takes up to 61 s per evaluation (f64, N = 10⁶, p = 18).
- **Against all 72 Grace cores: 2.2–6.2× at N = 10⁵ and 10⁶, 1.5–4.2× at N = 10⁷.** The
  lead is largest at p = 8 (4.9–6.2×), where the device's dense M2L is most efficient
  against the host's; smaller at f64 p = 12 and 18 (3.9–4.7× at N ≤ 10⁶), where the host
  switches to rotation; and smallest at p = 3 and at N = 10⁷, where the host part of the
  device's evaluation dominates (Section 7).
- **The level calls alone: 2.3–9.8×.** Without "other" the device leads by 4.2–9.8× at
  N = 10⁶, up to 2.5× the end-to-end figure at low p: the host part of the evaluation
  costs both paths about the same (8.4–13.5 ms on the device, 10–21 ms on the host at 72
  threads), so it caps the speed-up as N grows.
- At N = 10⁷ the host at 72 threads takes 0.34–1.1 s (f32) and 0.69–8.7 s (f64) per
  evaluation, under the brief's "about a minute", so 10⁷ was run there; not at one thread.
- The one-thread rows are 3 evaluations each (standard deviation under 0.4% of the mean);
  the host's strategy is its default `Auto` (`Rotation` at p = 12 and 18), the device's
  `Dense` (tuned), so those rows compare the best of each path.

The host's runs (kind times at 72 threads; the one-thread rows without kinds):

| run | precision | N | p | strategy | mean | std | M2L | P2P | L2P | other |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 thread | f32 | 10⁵ | 3 | Dense | 92.215 | 0.158 | – | – | – | – |
| 1 thread | f32 | 10⁵ | 6 | Dense | 267.383 | 0.605 | – | – | – | – |
| 1 thread | f32 | 10⁵ | 8 | Dense | 540.317 | 0.958 | – | – | – | – |
| 1 thread | f32 | 10⁶ | 3 | Dense | 1063.422 | 0.992 | – | – | – | – |
| 1 thread | f32 | 10⁶ | 6 | Dense | 2766.035 | 0.336 | – | – | – | – |
| 1 thread | f32 | 10⁶ | 8 | Dense | 5281.421 | 4.443 | – | – | – | – |
| 1 thread | f64 | 10⁵ | 3 | Dense | 198.183 | 0.212 | – | – | – | – |
| 1 thread | f64 | 10⁵ | 6 | Dense | 466.672 | 0.720 | – | – | – | – |
| 1 thread | f64 | 10⁵ | 8 | Dense | 962.232 | 1.512 | – | – | – | – |
| 1 thread | f64 | 10⁵ | 12 | Rotation | 2277.566 | 1.988 | – | – | – | – |
| 1 thread | f64 | 10⁵ | 18 | Rotation | 6363.858 | 5.879 | – | – | – | – |
| 1 thread | f64 | 10⁶ | 3 | Dense | 2441.072 | 1.072 | – | – | – | – |
| 1 thread | f64 | 10⁶ | 6 | Dense | 5058.938 | 5.290 | – | – | – | – |
| 1 thread | f64 | 10⁶ | 8 | Dense | 9652.637 | 5.180 | – | – | – | – |
| 1 thread | f64 | 10⁶ | 12 | Rotation | 22137.715 | 10.346 | – | – | – | – |
| 1 thread | f64 | 10⁶ | 18 | Rotation | 60782.403 | 19.555 | – | – | – | – |
| 72 threads | f32 | 10⁵ | 3 | Dense | 4.089 | 0.124 | 0.648 | 0.831 | 0.256 | 1.429 |
| 72 threads | f32 | 10⁵ | 6 | Dense | 7.272 | 0.085 | 3.646 | 0.846 | 0.467 | 1.459 |
| 72 threads | f32 | 10⁵ | 8 | Dense | 12.990 | 0.394 | 8.460 | 0.865 | 0.707 | 1.547 |
| 72 threads | f32 | 10⁶ | 3 | Dense | 27.522 | 0.212 | 4.294 | 9.297 | 1.382 | 9.982 |
| 72 threads | f32 | 10⁶ | 6 | Dense | 58.802 | 0.402 | 30.956 | 9.303 | 3.620 | 11.147 |
| 72 threads | f32 | 10⁶ | 8 | Dense | 110.536 | 0.275 | 77.658 | 9.379 | 5.792 | 12.287 |
| 72 threads | f32 | 10⁷ | 3 | Dense | 337.548 | 0.850 | – | – | – | – |
| 72 threads | f32 | 10⁷ | 6 | Dense | 628.606 | 1.911 | – | – | – | – |
| 72 threads | f32 | 10⁷ | 8 | Dense | 1094.512 | 15.790 | – | – | – | – |
| 72 threads | f64 | 10⁵ | 3 | Dense | 5.675 | 0.061 | 0.897 | 2.155 | 0.278 | 1.514 |
| 72 threads | f64 | 10⁵ | 6 | Dense | 11.365 | 0.174 | 5.963 | 2.194 | 0.512 | 1.567 |
| 72 threads | f64 | 10⁵ | 8 | Dense | 22.776 | 0.145 | 16.947 | 2.204 | 0.735 | 1.655 |
| 72 threads | f64 | 10⁵ | 12 | Rotation | 37.581 | 0.271 | 30.772 | 2.194 | 1.321 | 1.629 |
| 72 threads | f64 | 10⁵ | 18 | Rotation | 103.483 | 0.903 | 92.627 | 2.225 | 2.698 | 1.828 |
| 72 threads | f64 | 10⁶ | 3 | Dense | 54.505 | 0.356 | 6.313 | 26.875 | 1.699 | 16.761 |
| 72 threads | f64 | 10⁶ | 6 | Dense | 108.691 | 0.554 | 54.602 | 26.778 | 3.895 | 19.508 |
| 72 threads | f64 | 10⁶ | 8 | Dense | 216.403 | 0.418 | 157.780 | 26.842 | 5.860 | 20.219 |
| 72 threads | f64 | 10⁶ | 12 | Rotation | 355.277 | 1.234 | 284.127 | 26.962 | 11.890 | 18.864 |
| 72 threads | f64 | 10⁶ | 18 | Rotation | 958.326 | 1.748 | 864.007 | 27.866 | 26.368 | 20.731 |
| 72 threads | f64 | 10⁷ | 3 | Dense | 689.468 | 3.950 | – | – | – | – |
| 72 threads | f64 | 10⁷ | 6 | Dense | 1178.104 | 14.041 | – | – | – | – |
| 72 threads | f64 | 10⁷ | 8 | Dense | 2145.216 | 27.175 | – | – | – | – |
| 72 threads | f64 | 10⁷ | 12 | Rotation | 3306.811 | 11.072 | – | – | – | – |
| 72 threads | f64 | 10⁷ | 18 | Rotation | 8717.771 | 110.494 | – | – | – | – |

## 3. GH200 against M3 Max, f32

Two different runs on two machines: the M3 Max figures are phase4-m3max.md Section 1
(Phase 4 T13, Metal, 12 performance cores, 2026-10-04), the GH200's this report's
`device_fmm` run (CUDA, 72 Grace cores), with the same command, problems and timing
(median of 5). ms per evaluation; device tuned.

| problem | p | M3 host 1 thread | Grace 1 thread | M3 host 12 threads | Grace 72 threads | Metal | CUDA | CUDA x Metal | Metal x host 12 | CUDA x host 72 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube, N = 10⁵ | 3 | 73.2 | 91.4 | 8.39 | 3.99 | 2.08 | 1.61 | 1.29 | 4.0 | 2.5 |
| cube, N = 10⁵ | 6 | 227 | 267 | 24.7 | 7.47 | 4.41 | 1.91 | 2.31 | 5.6 | 3.9 |
| cube, N = 10⁵ | 8 | 514 | 524 | 50.9 | 13.1 | 6.71 | 2.66 | 2.52 | 7.6 | 4.9 |
| Plummer, N = 10⁵ | 3 | 246 | 307 | 27.5 | 12.8 | 3.59 | 3.12 | 1.15 | 7.7 | 4.1 |
| Plummer, N = 10⁵ | 6 | 726 | 863 | 81.3 | 25.1 | 6.58 | 3.95 | 1.67 | 12.3 | 6.3 |
| Plummer, N = 10⁵ | 8 | 1,364 | 1,522 | 140 | 38.8 | 10.90 | 5.18 | 2.10 | 12.9 | 7.5 |
| cube, N = 10⁶ | 3 | 778 | 1,056 | 75.2 | 27.6 | 15.4 | 13.8 | 1.11 | 4.9 | 2.0 |
| cube, N = 10⁶ | 6 | 2,224 | 2,740 | 218 | 59.5 | 41.8 | 16.8 | 2.48 | 5.2 | 3.5 |
| cube, N = 10⁶ | 8 | 4,926 | 5,260 | 498 | 111 | 73.5 | 22.9 | 3.20 | 6.8 | 4.8 |

- **The H100 is 1.1–3.2× the M3 Max GPU**, least at p = 3 (1.11–1.29×), where the
  evaluation is short and the host part and launches weigh most, and most at p = 8 on the
  large cube (3.2×), where M2L dominates. The peaks differ by 4.7× (67 against 14.3
  TFLOP/s in f32, *datasheet* and derived).
- **One Grace core is 1.02–1.36× slower than one M3 performance core** on these
  problems; 72 Grace cores are 2.1–4.5× the 12 M3 performance cores.
- **A server host narrows the device's lead**: CUDA over 72 Grace cores 2.0–7.5×, Metal
  over 12 M3 cores 4.0–12.9×. The tuner picked `Dense` everywhere on CUDA; on Metal it
  picked rotation for the cube at N = 10⁶, p = 6 and 8.
- The same in `nd-fmm-bench` (N = 10⁶, f32, p = 6, the unit cube; Metal from the T6 run of
  2026-10-07): Metal 42.4 ms, CUDA 16.0 ms (2.65×); the M3 host at 16 threads 195 ms,
  Grace at 72 threads 58.8 ms.

## 4. Kernel efficiency on CUDA

Peaks: f32 67 and f64 34 TFLOP/s, 4.0 TB/s (*datasheet*). "T7": measured by T7 on the same
code and machine (device-path.md §18.2), not repeated here (dropped on request).

### Dense and rotation M2L in the FMM (`nd-fmm-bench`, every V level, synchronous kinds)

Useful flops: 2 n² per V pair (n = (p + 1)²) for dense; the spike's model count
(20/3)(p + 1)³ per pair for rotation (*model* flops). The M2L kind time covers whole level
calls (gather, GEMM, reduction, in chunks of the 2 GB budget). Rotation from the
`--strategy rotation` runs at N = 10⁶.

| precision | N | p | n | V pairs | dense M2L ms | dense TFLOP/s | % of peak | rotation M2L ms | rotation TFLOP/s (model flops) | % of peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | 10⁵ | 3 | 16 | 640,584 | 0.37 | 0.90 | 1.3% | – | – | – |
| f32 | 10⁵ | 6 | 49 | 640,584 | 0.66 | 4.65 | 6.9% | – | – | – |
| f32 | 10⁵ | 8 | 81 | 640,584 | 1.22 | 6.88 | 10.3% | – | – | – |
| f32 | 10⁶ | 3 | 16 | 6,039,504 | 1.28 | 2.41 | 3.6% | 3.42 | 0.75 | 1.1% |
| f32 | 10⁶ | 6 | 49 | 6,039,504 | 4.36 | 6.66 | 9.9% | 8.57 | 1.61 | 2.4% |
| f32 | 10⁶ | 8 | 81 | 6,039,504 | 9.37 | 8.46 | 12.6% | 14.41 | 2.04 | 3.0% |
| f32 | 10⁷ | 3 | 16 | 52,337,880 | 9.72 | 2.76 | 4.1% | – | – | – |
| f32 | 10⁷ | 6 | 49 | 52,337,880 | 44.36 | 5.67 | 8.5% | – | – | – |
| f32 | 10⁷ | 8 | 81 | 52,337,880 | 96.33 | 7.13 | 10.6% | – | – | – |
| f64 | 10⁵ | 3 | 16 | 640,584 | 0.41 | 0.81 | 2.4% | – | – | – |
| f64 | 10⁵ | 6 | 49 | 640,584 | 0.82 | 3.75 | 11.0% | – | – | – |
| f64 | 10⁵ | 8 | 81 | 640,584 | 1.85 | 4.55 | 13.4% | – | – | – |
| f64 | 10⁵ | 12 | 169 | 640,584 | 5.73 | 6.39 | 18.8% | – | – | – |
| f64 | 10⁵ | 18 | 361 | 640,584 | 21.90 | 7.62 | 22.4% | – | – | – |
| f64 | 10⁶ | 3 | 16 | 6,039,504 | 1.64 | 1.89 | 5.6% | 3.74 | 0.69 | 2.0% |
| f64 | 10⁶ | 6 | 49 | 6,039,504 | 6.02 | 4.82 | 14.2% | 13.28 | 1.04 | 3.1% |
| f64 | 10⁶ | 8 | 81 | 6,039,504 | 14.85 | 5.34 | 15.7% | 25.30 | 1.16 | 3.4% |
| f64 | 10⁶ | 12 | 169 | 6,039,504 | 49.86 | 6.92 | 20.4% | 78.80 | 1.12 | 3.3% |
| f64 | 10⁶ | 18 | 361 | 6,039,504 | 200.26 | 7.86 | 23.1% | 258.74 | 1.07 | 3.1% |
| f64 | 10⁷ | 3 | 16 | 52,337,880 | 14.09 | 1.90 | 5.6% | – | – | – |
| f64 | 10⁷ | 6 | 49 | 52,337,880 | 65.53 | 3.84 | 11.3% | – | – | – |
| f64 | 10⁷ | 8 | 81 | 52,337,880 | 156.30 | 4.39 | 12.9% | – | – | – |
| f64 | 10⁷ | 12 | 169 | 52,337,880 | 601.63 | 4.97 | 14.6% | – | – | – |
| f64 | 10⁷ | 18 | 361 | 52,337,880 | 2502.87 | 5.45 | 16.0% | – | – | – |

- Dense M2L reaches 14–23% of the f64 peak and 10–13% of the f32 peak at p ≥ 6 at
  N = 10⁶, whole level calls included. At N = 10⁷ it drops to 11–16% (f64) and 8.5–11%
  (f32): the largest level runs in more chunks, and `Accumulate::Rows` walks every entry
  of a target's row in every chunk (T7's finding, not changed).
- Rotation stays at 2.0–3.4% of the f64 peak (1.1–3.0% of f32) at every p, as on Metal
  (1.8–3.0%, Phase 4 T10).

### The GEMM alone

| measurement | p | GFLOP/s | % of peak | source |
| --- | --- | --- | --- | --- |
| f32, the cube's level 4 at N = 10⁵ (584,136 pairs), `m2l_kernels` | 3 / 8 | 5,883 / 11,384 | 8.8% / 17.0% | this run |
| f32, the spike's shapes, B = 10⁵, `m2l_kernels` gate | 4 / 8 / 12 / 16 | 6,326 / 10,839 / 12,373 / 11,272 | 9.4–18.5% | this run |
| f32, the largest V level at N = 10⁶ | 3 / 6 / 8 | 6,700 / 10,900 / 12,200 | 10–18% | T7 |
| f64, the largest V level at N = 10⁶ | 3 / 6 / 8 / 12 | 4,300 / 7,800 / 7,000 / 8,000 | 13–24% | T7 |
| f64, the GEMM spike's best hand-written kernel, B = 10⁵ | 8 / 12 / 16 | 8,195 / 7,759 / 9,955 | 24% / 23% / 29% | this run (T2: 7.8–9.9 TFLOP/s) |

- **The library GEMM never runs on CUDA in the FMM**: `m2l_kernels` reports "No tile size
  is available for the problem" at every level for p = 8–16 (at p = 3 the library is not a
  candidate) (F28), so the input-precision guard is never reached and f32 runs the
  hand-written kernel at every p, as Phase 4S expected. In the GEMM spike the library's
  scalar `SimpleUnit` path does run, and on the spike's shapes it beats the hand-written
  kernel in f32 at p = 12 and 16 (13.6 and 16.4 against 12.4 and 11.3 TFLOP/s); the
  device path offers only the library's CMMA strategies, named explicitly (device-path.md
  §6.5), so it is not a candidate there.
- The M2L stage on the device (`m2l_kernels`, the C3.2 cube at N = 10⁵, every V level): 0.29,
  1.17, 3.70 and 10.6 ms at p = 3, 8, 12 and 16 (GEMM 22%, 66%, 82%, 91% of it); 123× and
  398× the host at one thread at p = 3 and 8, and 15× and 48× the host at 12 threads (the
  example's default thread count, not 72).

### P2P and the leaf operators (T7)

- P2P (`p2p_kernels`, T7): W1 at n_t = 64 at 21–28% (f32) and 26–34% (f64) of the peak
  model (C4.2 target 10%: met); W2 at N = 10⁵ at 26–33%, except f32 φ alone at 21.5%
  (target 25%: not met). In the FMM: 2.0–2.1 ms (f32) and 3.1–3.3 ms (f64) at N = 10⁶,
  10–17% of an evaluation (this run: 2.0–2.1 and 3.1–3.3 ms, Section 1). In this run the
  tuner timed the P2P level call of the cube at N = 10⁵ (58.1 × 10⁶ pairs, φ and ∇φ,
  f32) at 257 µs, 226 Gpairs/s, 10% of the model.
- L2P at f64 p = 12 (4.8 ms at N = 10⁶ here, T7 4.7 ms) and p = 18 (12.5 ms) is bound by
  its per-unit harmonics arrays (T7).

### Rotation against dense per p

| measure | f32 | f64 | source |
| --- | --- | --- | --- |
| per level, every V level (`rotation_kernels`), N = 10⁵ / 10⁶ | 1.55–3.24 / 1.27–3.72 (p = 2–10) | 1.17–3.28 / 1.06–2.24 (p = 4–20) | T7 |
| the largest level in the tuner (`autotune`, N = 10⁵) | 1.68–2.34 (p = 3–8) | 1.53–2.05 (p = 4–16) | this run |
| end to end, `nd-fmm-bench`, N = 10⁶ | 1.18–1.26 (p = 3–8) | 1.11–1.33 (p = 3–18) | this run |
| end to end, `device_fmm`, cube and Plummer at N = 10⁵, cube at 10⁶ | 1.24, 1.50, 1.34 / 1.32, 1.39, 1.29 / 1.18, 1.25, 1.22 (p = 3, 6, 8) | 1.51, 1.49, 1.30 / 1.46, 1.45, 1.27 (p = 8, 12, 18; N = 10⁵) | this run |

## 5. Autotune decisions

`autotune --device cuda` (a fresh cache per problem and p, the 10 s budget), the C3.2
cube and the Plummer sphere at N = 10⁵; evaluation ms.

| precision | problem | p | static rule | tuned | tuning s | evaluate static | evaluate tuned | tuned / static | rotation / dense at the largest level |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f32 | cube | 3 | Dense | Dense | 3.80 | 1.62 | 1.47 | 0.91 | 1.91 |
| f32 | cube | 6 | Dense | Dense | 3.62 | 1.94 | 1.91 | 0.98 | 2.22 |
| f32 | cube | 8 | Dense | Dense | 3.71 | 2.71 | 2.55 | 0.94 | 1.68 |
| f32 | Plummer | 3 | Dense | Dense | 5.83 | 3.59 | 3.19 | 0.89 | 2.34 |
| f32 | Plummer | 6 | Dense | Dense | 5.49 | 4.70 | 4.06 | 0.86 | 1.94 |
| f32 | Plummer | 8 | Dense | Dense | 5.83 | 6.01 | 5.27 | 0.88 | 1.79 |
| f64 | cube | 4 | Dense | Dense | 3.85 | 2.21 | 2.10 | 0.95 | 1.60 |
| f64 | cube | 8 | Dense | Dense | 3.63 | 3.78 | 3.76 | 0.99 | 1.79 |
| f64 | cube | 12 | Dense | Dense | 3.95 | 8.65 | 8.48 | 0.98 | 1.66 |
| f64 | cube | 16 | Dense | Dense | 10.00 | 21.31 | 17.78 | 0.83 | 1.53 |
| f64 | Plummer | 4 | Dense | Dense | 6.30 | 4.90 | 4.28 | 0.87 | 1.65 |
| f64 | Plummer | 8 | Dense | Dense | 5.76 | 8.17 | 7.50 | 0.92 | 2.05 |
| f64 | Plummer | 12 | Dense | Dense | 5.98 | 16.20 | 15.64 | 0.97 | 1.72 |
| f64 | Plummer | 16 | Dense | Dense | 5.73 | 35.55 | 30.37 | 0.85 | 1.57 |

- **The tuner chose `Dense` in every case** (f32 p = 3–8, f64 p = 4–16), and so did every
  tuned `nd-fmm-bench` and `device_fmm` build (N = 10⁵ to 10⁷, f64 to p = 18): the static
  rule on CUDA (decision 9) is what the tuner picks.
- Tuned evaluations ran 0.83–0.99× the static rule's time, from the GEMM layouts per level
  size and the P2P layout (in `nd-fmm-bench`, f32 at N = 10⁷: the plane layout with 2
  planes); the largest
  gains at p = 16 in f64 (0.83, 0.85).
- Tuning took 3.6–6.3 s per build, except the cube at f64 p = 16, which used the whole
  10 s budget with 2 of 9 decisions made (the rest tuned at the rebuild from the cache,
  3.1 s). A rebuild from the cache takes 0.09–0.43 s.

## 6. f64 on a GPU: the first measurements against the models

| item (device-path.md) | model or estimate | measured on the H100 |
| --- | --- | --- |
| f64 accuracy (§9, Phase 4S requirement 5) | the CPU runtime's f64 bounds: device − host 1e-12, errors within 0.1% | device − host at most 1.2e-14 (φ) and 3.2e-16 (∇φ), error ratios 1.0000 on every `device_fmm` row (N = 10⁵, p = 8, 12, 18, cube and Plummer, every strategy); f32 2.9e-6 and 0.991–1.002 |
| hand-written f64 GEMM (§10.5; the spike's model) | 19–36% of the roofline (central), 9–14% (low) | the GEMM alone 13–24% of peak (T7, p = 3–12); whole dense M2L level calls 14–23% at N = 10⁶, p = 6–18 (Section 4); the spike's kernels 23–29% at p = 8–16 (this run). The production kernel sits at or below the low end of the central model; the spike's best kernel inside it |
| rotation break-even (§10.5, A100 central) | 13.2% of f64 peak at p = 8, 5.4% at p = 12; 10.7%, 8.6%, 6.9% at p = 9, 10, 11 | rotation reaches 3.1–3.4% at p = 6–18 (model flops, whole level calls), below every break-even |
| the f64 static rule (§10.5, decision 13 of Phase 4) | `Dense` to p = 11, `Rotation` from 12, provisional | dense faster at every p from 3 to 18 end to end (1.11–1.33×, N = 10⁶) and per level to p = 20 (T7): **no crossover**; T7's rule (`Dense` at every p, decision 9) confirmed |
| f64 against f32 | 2× (the peaks) | 1.59–1.64× per evaluation (N = 10⁶ and 10⁷, p = 3–8); M2L 1.27–1.58× (N = 10⁶) |
| P2P (§13.4 as derived by T7) | 1,673 / 1,115 Gpairs/s (*model*) | 26–34% of it on W1 (T7) |
| device memory (§4.6) | dense tables of 52 MB at f64 p = 11 | 0.07–6.4 GB per `Fmm` under dense (`nd-fmm-bench` and `device_fmm`, N = 10⁵ to 10⁷; f64 p = 18 at N = 10⁷: 6.4 GB; f64 from 175 MB); rotation from 20 MB (f32) and 37–103 MB (f64) at N = 10⁵ |
| launches per evaluation (§8.1) | about 49 (cube, p = 8) | the cube: 40–43 at N = 10⁵, 49–94 at 10⁶, 75–486 at 10⁷ (f32 p = 3 to f64 p = 18: more levels and chunks); the Plummer sphere at N = 10⁵ 87–101; one sync |

## The Phase 4 CUDA run: what it settles

phase4-m3max.md §4 listed what "the CUDA run" would settle. Number by number:
1. **The f64 GEMM efficiency on a data-centre card** (spike model 19–36% of the
   roofline): the production hand-written GEMM 13–24% alone (T7) and 14–23% as whole
   M2L level calls (this run, p ≥ 6); the spike's own kernels 23–29%. At the low end of
   the model's central range at p ≥ 8 (21–24% alone, 20–23% as level calls at p = 12–18)
   and below it at p ≤ 6, which T7 traced to its register blocks (no shared-memory
   staging); a shared-memory-tiled variant is the open next step (§18.2).
2. **The f64 dense/rotation crossover per p**: none between p = 3 and 18 end to end, or 4
   and 20 per level (T7). Rotation is 1.11–1.51× slower end to end (N = 10⁵ and 10⁶) and
   1.53–2.05× at the tuner's largest level.
3. **So the f64 static rule**: the provisional Phase 4 rule (rotation from p = 12) does not
   hold on Hopper. T7 already moved CUDA to `Dense` at every p (decision 9); this run
   confirms it, including p = 18 end to end and every tuner decision. Metal and the CPU
   runtime keep their rules.
4. **The f32 GEMM with the input-precision guard**: the library is never used on CUDA; it
   fails its own probe on the LLVM path before the guard is reached (F28), so f32 runs the
   hand-written kernel at 9–18.5% of the f32 peak.
5. **The device FMM against a server host** (an aarch64 Grace with 72 cores rather than
   an x86_64 one, which no phase has timed): 2.0–7.5× at N = 10⁵ and 10⁶ (`device_fmm`,
   f32) and 2.2–6.2× at N = 10⁵ and 10⁶ and 1.5–4.2× at N = 10⁷ (`nd-fmm-bench`, f32 and f64); Section 2.

## 7. The host part of a device evaluation

The largest single cost of the device FMM at N ≥ 10⁶ for p ≤ 8 lies outside every level
call ("other" in Section 1): 8.4 ms (f32) and 13.0–13.5 ms (f64) at N = 10⁶, 183–189 ms
and 298–303 ms at N = 10⁷, independent of p and about 1.6× as large in f64 as in f32. It
is host work per point. T7 measured part of it at N = 10⁶ (a pinned host buffer allocated
per evaluation, 1.0–15.6 ms; the 16 MB download itself 56 µs). Here one `nsys` trace per
precision at N = 10⁷, p = 3 (`nd-fmm-bench --backend cuda --n 1e7 --degree 3 --repeats 3
--warmup 1 --kinds off --accuracy off` under `nsys profile --trace=cuda,osrt
--sample=process-tree`; 226.7 ms (f32) and 363.2 ms (f64) per evaluation under the
profiler, as without it):

| item | f32 | f64 |
| --- | --- | --- |
| CPU samples inside `Fmm::evaluate` (about 1 kHz; the warm-up and 3 timed evaluations) | 348 | 511 |
| of them in `scaled_output` (the output pass) | 101 (29%) | 165 (32%) |
| in the download (`DeviceDriver::read_output`, `Device::download`) | 11 (3.2%) | 21 (4.1%) |
| in `memcpy` (inside the two above) | 13 (3.7%) | 24 (4.7%) |
| loading the charges (`LeafOrder::points`, `Device::write`) | under 1% | under 2% |
| broken call stacks (not attributable) | 176 (51%) | 274 (54%) |
| `cuMemAllocHost` (pinned host buffers), the whole run | 6 calls, 73.6 ms, median 2.45 ms, at most 53.3 ms | 6 calls, 23.8 ms, median 2.32 ms, at most 13.9 ms |
| `cuMemcpyDtoHAsync` (the download, enqueue time) | 4 calls, 14 µs on average | 4 calls, 9 µs on average |
| the download on the GPU (`[CUDA memcpy Device-to-Host]`, 160 / 320 MB) | 0.54 ms each | 1.08 ms each |

- **`scaled_output` is the largest identified part**: one serial loop over every target
  that divides each value by its leaf's scale (`4π r`, `4π r²`) and scatters it into the
  caller's order (a random-access store into a 10⁷-element array), into potential and
  gradient vectors allocated and zeroed afresh in every evaluation (160 MB in f32, 320 MB
  in f64, first touched there). The host path runs the same pass, so the host at 72
  threads spends 10–21 ms outside its level calls at N = 10⁶ too (Section 2).
- The pinned buffer (per evaluation, T7) and the download are a smaller part. Half the
  samples have broken call stacks, so the split is a lower bound per item, not a
  breakdown; hardware counters are not available (`ncu` refused).
- Not changed here (T8 changes no code, and transfers are out of Phase 4S's scope).
  Candidates, for Phase 5 or later: the output pass threaded over leaves (it writes
  disjoint targets) and without the fresh allocation; a persistent pinned download buffer
  (CubeCL's read path); the same for the charge gather. They change no value.



## 8. Leaf size on CUDA

The Phase 4 leaf-size rule (T13, decision 8 of Phase 4), applied on CUDA:
`device_fmm --part leaf`, the device under its static rule, the cube and the Plummer
sphere, p = 3 and 8, refinement targets 16 to 256; at N = 10⁶ (the brief's size; `--n
1000000`) and at N = 10⁵ (the documented run's default). The rule takes the size with the
smallest geometric mean of the evaluation time over the four configurations, if it is at
least 5% faster than 64 and no error grows by more than 10%.

Geometric mean of the evaluation time, relative to 64 (ms at 64 in brackets):

| N | precision | 16 | 32 | 64 | 128 | 256 | the rule picks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 10⁶ | f32 | 2.424 | 1.338 | 1.000 (20.62) | 1.017 | 1.449 | 64 |
| 10⁶ | f64 | 2.374 | 1.316 | 1.000 (32.80) | 1.010 | 1.414 | 64 |
| 10⁵ | f32 | 1.941 | 1.033 | 1.000 (3.08) | 1.176 | 2.670 | 64 |
| 10⁵ | f64 | 1.999 | 1.053 | 1.000 (4.10) | 1.177 | 2.601 | 64 |

The fastest size per configuration, evaluation ms (median of 5):

| N | precision | distribution | p | 16 | 32 | 64 | 128 | 256 | fastest (64 / fastest) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10⁶ | f32 | cube | 3 | 27.12 | 16.98 | 13.94 | **13.66** | 22.72 | 128 (1.02) |
| 10⁶ | f32 | cube | 8 | 119.43 | 37.16 | 23.07 | **22.82** | 25.73 | 128 (1.01) |
| 10⁶ | f32 | Plummer | 3 | 23.30 | 18.51 | **16.91** | 20.78 | 33.56 | 64 |
| 10⁶ | f32 | Plummer | 8 | 82.60 | 49.53 | 33.20 | **29.81** | 40.57 | 128 (1.11) |
| 10⁶ | f64 | cube | 3 | 39.21 | 25.78 | **21.78** | 21.85 | 37.19 | 64 |
| 10⁶ | f64 | cube | 8 | 202.72 | 60.89 | 37.30 | **37.10** | 42.05 | 128 (1.01) |
| 10⁶ | f64 | Plummer | 3 | 34.10 | 27.60 | **26.97** | 32.15 | 49.58 | 64 |
| 10⁶ | f64 | Plummer | 8 | 135.69 | 80.09 | 52.82 | **46.20** | 59.70 | 128 (1.14) |
| 10⁵ | f32 | cube | 3 | 3.08 | 1.66 | **1.62** | 1.63 | 5.61 | 64 |
| 10⁵ | f32 | cube | 8 | 11.73 | 3.04 | 2.63 | **2.62** | 5.82 | 128 (1.00) |
| 10⁵ | f32 | Plummer | 3 | 3.58 | **3.10** | 3.52 | 5.47 | 10.91 | 32 (1.14) |
| 10⁵ | f32 | Plummer | 8 | 9.82 | 6.51 | **5.96** | 7.33 | 12.76 | 64 |
| 10⁵ | f64 | cube | 3 | 3.94 | 2.25 | **2.00** | 2.01 | 7.21 | 64 |
| 10⁵ | f64 | cube | 8 | 18.15 | 4.26 | **3.83** | 3.85 | 7.62 | 64 |
| 10⁵ | f64 | Plummer | 3 | 4.34 | **3.87** | 4.51 | 7.06 | 13.95 | 32 (1.17) |
| 10⁵ | f64 | Plummer | 8 | 14.46 | 9.31 | **8.12** | 9.87 | 16.79 | 64 |

- **The rule picks 64 on CUDA in f32 and f64, at N = 10⁵ and 10⁶.** Decision 7 of
  Phase 4S, decided on 2026-10-07 with these numbers: **no CUDA leaf-size default** (64
  stays the default on every backend).
- On the cube, 64 and 128 give the same uniform tree (4,096 leaves at N = 10⁵, 32,768 at
  10⁶), so their times differ only by run-to-run noise (0–2%).
- The best size depends on the distribution and p, as on the host and on Metal: on the
  Plummer sphere 128 is 11–14% faster at p = 8 and N = 10⁶, and 32 is 14–17% faster at
  p = 3 and N = 10⁵. Unlike on Metal (256 at p = 8 on the cube), large leaves cost CUDA
  more: its far field is cheaper relative to P2P, so the near share at 256 points per leaf
  reaches 51–82%. A size per p or a tuned size would gain up to 17%, as Phase 4 found; it
  stays a later decision (laplace-fmm-plan §9.2).
- The errors move as on Metal: φ barely, ∇φ improving with larger leaves (cube at
  N = 10⁶, p = 8, f32: 5.77e-5 at 16 to 2.80e-5 at 256).

## 9. Checks run for T8

T8 changes no code. On locust, after the timing runs (the same tree):
all passed (20:06–20:38 UTC; logs on locust under `/data/ucahtbe/logs/t8/checks/`):
- `RUST_MIN_STACK=8388608 cargo test --workspace` (MPI from the spack environment): 734
  passed, 0 failed, 44 ignored, in 62 test executables;
- `cargo test -p nd-fmm-kernels --release --features cpu,cuda -- --include-ignored
  --show-output`: 150 passed ("backends run: cuda" in 56 tests, "cpu" in 58);
- `cargo clippy -p nd-fmm-kernels --all-targets --features cpu,cuda -- -D warnings` and the
  same for `nd-fmm-exec`: clean;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release`: 77 passed,
  11 ignored; with `-- --ignored` (every CUDA test: `tests/device_cuda.rs`, the CUDA blocks
  of the gates and the operator tests): 11 passed, in 18.6 minutes. Run without
  `--show-output`, so their "backends run" lines were captured by the test harness and are
  not in the logs;
- `nd-fmm-bench`'s smoke test: `cargo test -p nd-fmm-bench` and with `--features cpu
  --release`, 14 passed each (the smoke run on the host and the CPU runtime);
- `tools/gh200/check-home.sh`: only `/home/ucahtbe/.bash_history`, last written at 13:20
  local time, before this task's first command (an interactive login's, README "Checking
  the home directory").

MPI at several ranks on locust (for the Phase 5 briefs, decision 8; before the timing runs,
release builds, `timeout 300 mpirun -n <n> …` inside `tools/gh200/env.sh`, no extra flags):

| program | ranks | result | wall time per run |
| --- | --- | --- | --- |
| `nd-octree`'s `test_mpi_complete_tree`, `test_mpi_construction_edge_cases`, `test_mpi_leaf_lookup` | 1, 2, 4, 8, 16, 32, 64, 72 | all pass | 0.11–0.19 s at 1–8 ranks, 0.23–0.41 s at 16, 0.30–0.65 s at 32, 0.42–1.09 s at 64, 0.59–1.24 s at 72 |
| `nd-fmm-exec`'s `tests/mpi_exec.rs` (release, `RUST_MIN_STACK=8388608`) | 1, 2, 4, 8 | passes | 3.8 s at 1 rank (every scenario), 0.29–0.34 s at 2–8 (the one multi-rank scenario) |

On the M3 Max (sandboxed unless stated), all passed:
- the root checks: `cargo fmt -- --check`, `cargo clippy -- -D warnings`, `cargo clippy
  --examples -- -D warnings`, `RUST_MIN_STACK=8388608 cargo test` (690 passed, 44 ignored),
  `cargo doc --no-deps`;
- the stricter checks: `cargo clippy --workspace --all-targets -- -D warnings`,
  `RUST_MIN_STACK=8388608 cargo test --workspace` (734 passed, 44 ignored);
- `nd-fmm-kernels`: clippy with `cpu,metal`, `cargo check --features cuda`, `cargo test
  --features cpu --release` (92 passed; "backends run: cpu"); by hand outside the sandbox,
  `--features metal --release -- --ignored` (48 passed; "backends run: metal");
- `nd-fmm-exec`: clippy with `cpu,metal`, `cargo check --features cuda`, `cargo test
  --features cpu --release` (83 passed) and with `-- --ignored` (8 passed), the host's
  ignored gates (4 passed); by hand outside the sandbox, `--features metal --release --
  --ignored` (11 passed);
- `nd-fmm-bench`: clippy with `cpu,metal`, `cargo check --features cuda`, `cargo test
  --features cpu --release` (14 passed, the smoke run on the host and the CPU runtime);
- `nd-fmm-simd`: `cargo test --release -- --ignored` (6 passed; NEON and scalar).

## Raw output

The generated Markdown, as printed, with locust's paths shortened to `L` (=
`/data/ucahtbe/logs/t8`). Kept here as the only raw output of T8 (fmm-bench/CLAUDE.md).

### `nsys` at N = 10⁷, p = 3, f32 (Section 7): `nsys stats` excerpts

<details><summary>Output</summary>

`cuda_api_sum` (CUDA API calls, the whole run: build, warm-up and 3 timed evaluations):

| Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60.1 | 285456928 | 10 | 28545692.8 | 1352208.0 | 9504 | 206789600 | 64742610.6 | cuMemAllocAsync |
| 22.3 | 106150336 | 24 | 4422930.7 | 4336.0 | 512 | 35209152 | 11879629.9 | cuEventSynchronize |
| 15.5 | 73553504 | 6 | 12258917.3 | 2451392.0 | 840288 | 53260736 | 20664760.4 | cuMemAllocHost_v2 |
| 1.2 | 5768800 | 240 | 24036.7 | 3136.0 | 1824 | 810144 | 113027.5 | cuMemcpyHtoDAsync_v2 |
| 0.3 | 1633056 | 11 | 148459.6 | 152096.0 | 104256 | 184256 | 26180.8 | cuModuleLoadData |
| 0.2 | 1140096 | 308 | 3701.6 | 2176.0 | 1632 | 49472 | 5381.1 | cuLaunchKernel |
| 0.2 | 1118976 | 2 | 559488.0 | 559488.0 | 31904 | 1087072 | 746116.4 | cuStreamCreate |
| 0.0 | 178240 | 1257 | 141.8 | 96.0 | 32 | 3040 | 266.4 | cuCtxSetCurrent |
| 0.0 | 119776 | 25 | 4791.0 | 4768.0 | 1920 | 14368 | 2424.4 | cuEventRecord |
| 0.0 | 65952 | 25 | 2638.1 | 2752.0 | 1056 | 5728 | 1170.4 | cuEventCreate |
| 0.0 | 56928 | 4 | 14232.0 | 8800.0 | 8160 | 31168 | 11295.8 | cuMemcpyDtoHAsync_v2 |
| 0.0 | 56576 | 24 | 2357.3 | 2160.0 | 384 | 4416 | 1008.3 | cuEventDestroy_v2 |
| 0.0 | 4096 | 2 | 2048.0 | 2048.0 | 1568 | 2528 | 678.8 | cuInit |

`cuda_gpu_mem_time_sum`:

| Time (%) | Total Time (ns) | Count | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Operation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 73.5 | 5984768 | 240 | 24936.5 | 832.0 | 704 | 773664 | 108617.9 | [CUDA memcpy Host-to-Device] |
| 26.5 | 2158432 | 4 | 539608.0 | 539632.0 | 539200 | 539968 | 315.0 | [CUDA memcpy Device-to-Host] |

The CPU-sample counts of Section 7 are from the exported SQLite database
(`SAMPLING_CALLCHAINS`, the samples whose call chain contains `Fmm::evaluate`), not
reproduced here.

</details>

### `nsys` at N = 10⁷, p = 3, f64 (Section 7): `nsys stats` excerpts

<details><summary>Output</summary>

`cuda_api_sum` (CUDA API calls, the whole run: build, warm-up and 3 timed evaluations):

| Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 57.2 | 321459392 | 13 | 24727645.5 | 3360384.0 | 2336 | 211320640 | 57828549.1 | cuMemAllocAsync |
| 36.6 | 205420640 | 24 | 8559193.3 | 5504.0 | 544 | 55250848 | 19736040.2 | cuEventSynchronize |
| 4.2 | 23755392 | 6 | 3959232.0 | 2322512.0 | 825440 | 13881024 | 5009459.7 | cuMemAllocHost_v2 |
| 1.2 | 6502976 | 240 | 27095.7 | 3072.0 | 1696 | 889952 | 128161.1 | cuMemcpyHtoDAsync_v2 |
| 0.3 | 1702784 | 11 | 154798.5 | 153216.0 | 103040 | 244032 | 38237.6 | cuModuleLoadData |
| 0.2 | 1192448 | 344 | 3466.4 | 2112.0 | 1600 | 45216 | 4986.2 | cuLaunchKernel |
| 0.2 | 1163712 | 2 | 581856.0 | 581856.0 | 32928 | 1130784 | 776301.4 | cuStreamCreate |
| 0.0 | 165120 | 1293 | 127.7 | 64.0 | 32 | 2688 | 246.6 | cuCtxSetCurrent |
| 0.0 | 106688 | 25 | 4267.5 | 4448.0 | 1568 | 6336 | 1440.9 | cuEventRecord |
| 0.0 | 59712 | 24 | 2488.0 | 2496.0 | 512 | 4896 | 1018.4 | cuEventDestroy_v2 |
| 0.0 | 57888 | 25 | 2315.5 | 2400.0 | 928 | 3712 | 812.6 | cuEventCreate |
| 0.0 | 36800 | 4 | 9200.0 | 9008.0 | 8224 | 10560 | 979.2 | cuMemcpyDtoHAsync_v2 |
| 0.0 | 4320 | 2 | 2160.0 | 2160.0 | 1472 | 2848 | 973.0 | cuInit |

`cuda_gpu_mem_time_sum`:

| Time (%) | Total Time (ns) | Count | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Operation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 62.9 | 7300960 | 240 | 30420.7 | 864.0 | 704 | 842400 | 127074.4 | [CUDA memcpy Host-to-Device] |
| 37.1 | 4304832 | 4 | 1076208.0 | 1076080.0 | 1075872 | 1076800 | 406.9 | [CUDA memcpy Device-to-Host] |

The CPU-sample counts of Section 7 are from the exported SQLite database
(`SAMPLING_CALLCHAINS`, the samples whose call chain contains `Fmm::evaluate`), not
reproduced here.

</details>

### `nd-fmm-bench`, CUDA, tuned, f32

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:36:02 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:36:02 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e5,1e6,1e7 --precision f32 --degree 3,6,8 --tuning-cache L/tune-bench/f32 --table-cache L/tables --output L/bench/cuda-tuned-f32.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | `L/tune-bench/f32` |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 100000 | 3 | Dense | 3.978 | 1.475 | 1.572 | 1.559 | 1.599 | 0.041 | 3.39e-3 | 1.99e-3 | yes |
| cuda | f32 | 100000 | 6 | Dense | 3.771 | 1.816 | 1.889 | 1.909 | 2.028 | 0.090 | 1.42e-4 | 1.30e-4 | yes |
| cuda | f32 | 100000 | 8 | Dense | 4.020 | 2.558 | 2.657 | 2.648 | 2.712 | 0.053 | 2.12e-5 | 2.37e-5 | yes |
| cuda | f32 | 1000000 | 3 | Dense | 2.517 | 12.433 | 12.755 | 12.829 | 13.549 | 0.331 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 6 | Dense | 2.626 | 15.667 | 15.917 | 16.019 | 17.498 | 0.542 | 8.53e-5 | 2.56e-4 | yes |
| cuda | f32 | 1000000 | 8 | Dense | 2.663 | 21.333 | 21.599 | 21.669 | 22.735 | 0.415 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f32 | 10000000 | 3 | Dense | 19.208 | 219.081 | 220.777 | 221.470 | 224.153 | 1.777 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f32 | 10000000 | 6 | Dense | 20.483 | 255.265 | 257.726 | 257.534 | 259.646 | 1.366 | 7.90e-5 | 1.23e-4 | yes |
| cuda | f32 | 10000000 | 8 | Dense | 22.490 | 312.733 | 315.430 | 315.042 | 317.466 | 1.640 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 100000 | 3 | 0.027 | 0.150 | 0.366 | 0.135 | 0.000 | 0.000 | 0.147 | 0.276 | 0.794 | 1.102 | 1.923 | 0.707 |
| cuda | f32 | 100000 | 6 | 0.029 | 0.169 | 0.662 | 0.298 | 0.000 | 0.000 | 0.088 | 0.276 | 0.803 | 1.524 | 2.364 | 0.798 |
| cuda | f32 | 100000 | 8 | 0.034 | 0.195 | 1.222 | 0.361 | 0.000 | 0.000 | 0.220 | 0.278 | 0.843 | 2.310 | 3.203 | 0.872 |
| cuda | f32 | 1000000 | 3 | 0.056 | 0.191 | 1.285 | 0.299 | 0.000 | 0.000 | 0.073 | 1.991 | 8.416 | 3.894 | 12.568 | 0.304 |
| cuda | f32 | 1000000 | 6 | 0.070 | 0.227 | 4.356 | 0.307 | 0.000 | 0.000 | 0.234 | 2.138 | 8.473 | 7.332 | 16.183 | 0.458 |
| cuda | f32 | 1000000 | 8 | 0.114 | 0.286 | 9.372 | 0.585 | 0.000 | 0.000 | 0.421 | 2.138 | 8.471 | 12.916 | 21.881 | 0.596 |
| cuda | f32 | 10000000 | 3 | 0.392 | 0.347 | 9.718 | 0.613 | 0.072 | 0.239 | 0.441 | 24.125 | 184.238 | 35.947 | 223.485 | 0.162 |
| cuda | f32 | 10000000 | 6 | 0.539 | 0.713 | 44.362 | 0.823 | 0.066 | 0.431 | 1.481 | 24.102 | 183.468 | 72.516 | 260.586 | 0.282 |
| cuda | f32 | 10000000 | 8 | 0.963 | 1.013 | 96.330 | 1.312 | 0.074 | 0.356 | 3.178 | 24.093 | 188.574 | 127.319 | 321.765 | 0.404 |

##### cuda, f32, N = 100000, p = 3

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 3.978 s, of it tables 0.001 s (built) and device 3.901 s.
- Evaluation over 10 repeats, ms: min 1.475, median 1.572, mean 1.559, max 1.599, std 0.041; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 3.39e-3, ∇φ 1.99e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 94.9 MB.
- One evaluation: uploads 1 (400000 B), downloads 1 (1600000 B), launches 40, syncs 1, timing windows 0.
- Tuning: 3.88 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.026 | 0.027 | 0.027 | 2.4% |
| M2M | device | 4 | 0.149 | 0.150 | 0.151 | 13.6% |
| M2L | device | 3 | 0.365 | 0.366 | 0.368 | 33.3% |
| L2L | device | 4 | 0.132 | 0.135 | 0.142 | 12.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.141 | 0.147 | 0.153 | 13.4% |
| P2P | device | 1 | 0.274 | 0.276 | 0.277 | 25.0% |
| other | | | | 0.794 | | |
| sum | | | | 1.102 | | 100.0% |

##### cuda, f32, N = 100000, p = 6

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 3.771 s, of it tables 0.006 s (built) and device 3.692 s.
- Evaluation over 10 repeats, ms: min 1.816, median 1.889, mean 1.909, max 2.028, std 0.090; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.42e-4, ∇φ 1.30e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 253.2 MB.
- One evaluation: uploads 1 (400000 B), downloads 1 (1600000 B), launches 40, syncs 1, timing windows 0.
- Tuning: 3.67 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.029 | 0.029 | 0.030 | 1.9% |
| M2M | device | 4 | 0.167 | 0.169 | 0.173 | 11.1% |
| M2L | device | 3 | 0.660 | 0.662 | 0.665 | 43.5% |
| L2L | device | 4 | 0.156 | 0.298 | 0.361 | 19.6% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.085 | 0.088 | 0.097 | 5.8% |
| P2P | device | 1 | 0.276 | 0.276 | 0.277 | 18.1% |
| other | | | | 0.803 | | |
| sum | | | | 1.524 | | 100.0% |

##### cuda, f32, N = 100000, p = 8

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 4.020 s, of it tables 0.016 s (built) and device 3.930 s.
- Evaluation over 10 repeats, ms: min 2.558, median 2.657, mean 2.648, max 2.712, std 0.053; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.12e-5, ∇φ 2.37e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P plane (4 planes per cube); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 409.7 MB.
- One evaluation: uploads 1 (400000 B), downloads 1 (1600000 B), launches 40, syncs 1, timing windows 0.
- Tuning: 3.91 s; M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P plane (4 planes per cube) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.032 | 0.034 | 0.038 | 1.5% |
| M2M | device | 4 | 0.193 | 0.195 | 0.198 | 8.5% |
| M2L | device | 3 | 1.219 | 1.222 | 1.225 | 52.9% |
| L2L | device | 4 | 0.350 | 0.361 | 0.367 | 15.6% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.213 | 0.220 | 0.226 | 9.5% |
| P2P | device | 1 | 0.272 | 0.278 | 0.281 | 12.0% |
| other | | | | 0.843 | | |
| sum | | | | 2.310 | | 100.0% |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 2.517 s, of it tables 0.000 s (loaded from the cache) and device 1.716 s.
- Evaluation over 10 repeats, ms: min 12.433, median 12.755, mean 12.829, max 13.549, std 0.331; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 878.9 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 49, syncs 1, timing windows 0.
- Tuning: 1.52 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (cache).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.055 | 0.056 | 0.057 | 1.4% |
| M2M | device | 5 | 0.188 | 0.191 | 0.195 | 4.9% |
| M2L | device | 4 | 1.279 | 1.285 | 1.290 | 33.0% |
| L2L | device | 5 | 0.291 | 0.299 | 0.306 | 7.7% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.063 | 0.073 | 0.078 | 1.9% |
| P2P | device | 1 | 1.974 | 1.991 | 2.018 | 51.1% |
| other | | | | 8.416 | | |
| sum | | | | 3.894 | | 100.0% |

##### cuda, f32, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 2.626 s, of it tables 0.004 s (loaded from the cache) and device 1.827 s.
- Evaluation over 10 repeats, ms: min 15.667, median 15.917, mean 16.019, max 17.498, std 0.542; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 2317.0 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 49, syncs 1, timing windows 0.
- Tuning: 1.43 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (cache).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.069 | 0.070 | 0.071 | 1.0% |
| M2M | device | 5 | 0.224 | 0.227 | 0.229 | 3.1% |
| M2L | device | 4 | 4.350 | 4.356 | 4.360 | 59.4% |
| L2L | device | 5 | 0.255 | 0.307 | 0.473 | 4.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.224 | 0.234 | 0.241 | 3.2% |
| P2P | device | 1 | 2.125 | 2.138 | 2.157 | 29.2% |
| other | | | | 8.473 | | |
| sum | | | | 7.332 | | 100.0% |

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 2.663 s, of it tables 0.010 s (loaded from the cache) and device 1.858 s.
- Evaluation over 10 repeats, ms: min 21.333, median 21.599, mean 21.669, max 22.735, std 0.415; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P plane (4 planes per cube); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2365.8 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 52, syncs 1, timing windows 0.
- Tuning: 1.67 s; M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 8388608 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P plane (4 planes per cube) (cache).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 2 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.113 | 0.114 | 0.116 | 0.9% |
| M2M | device | 5 | 0.283 | 0.286 | 0.289 | 2.2% |
| M2L | device | 4 | 9.359 | 9.372 | 9.397 | 72.6% |
| L2L | device | 5 | 0.580 | 0.585 | 0.597 | 4.5% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.404 | 0.421 | 0.435 | 3.3% |
| P2P | device | 1 | 2.114 | 2.138 | 2.155 | 16.5% |
| other | | | | 8.471 | | |
| sum | | | | 12.916 | | 100.0% |

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 19.208 s, of it tables 0.001 s (loaded from the cache) and device 7.155 s.
- Evaluation over 10 repeats, ms: min 219.081, median 220.777, mean 221.470, max 224.153, std 1.777; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P plane (2 planes per cube); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 3834.6 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 75, syncs 1, timing windows 0.
- Tuning: 5.00 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 67108864 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 128 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 128 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 64 points per leaf: P2P plane (2 planes per cube) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 3 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.389 | 0.392 | 0.396 | 1.1% |
| M2M | device | 7 | 0.342 | 0.347 | 0.351 | 1.0% |
| M2L | device | 5 | 9.705 | 9.718 | 9.735 | 27.0% |
| L2L | device | 7 | 0.450 | 0.613 | 0.649 | 1.7% |
| P2L | device | 1 | 0.071 | 0.072 | 0.073 | 0.2% |
| M2P | device | 1 | 0.232 | 0.239 | 0.243 | 0.7% |
| L2P | device | 2 | 0.416 | 0.441 | 0.472 | 1.2% |
| P2P | device | 2 | 24.102 | 24.125 | 24.150 | 67.1% |
| other | | | | 184.238 | | |
| sum | | | | 35.947 | | 100.0% |

##### cuda, f32, N = 10000000, p = 6

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 20.483 s, of it tables 0.004 s (loaded from the cache) and device 8.204 s.
- Evaluation over 10 repeats, ms: min 255.265, median 257.726, mean 257.534, max 259.646, std 1.366; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 7.90e-5, ∇φ 1.23e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P plane (2 planes per cube); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 3916.6 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 93, syncs 1, timing windows 0.
- Tuning: 6.04 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 67108864 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 128 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 128 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 64 points per leaf: P2P plane (2 planes per cube) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 9 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.536 | 0.539 | 0.542 | 0.7% |
| M2M | device | 7 | 0.702 | 0.713 | 0.719 | 1.0% |
| M2L | device | 5 | 44.350 | 44.362 | 44.373 | 61.2% |
| L2L | device | 7 | 0.717 | 0.823 | 0.931 | 1.1% |
| P2L | device | 1 | 0.065 | 0.066 | 0.067 | 0.1% |
| M2P | device | 1 | 0.429 | 0.431 | 0.433 | 0.6% |
| L2P | device | 2 | 1.440 | 1.481 | 1.512 | 2.0% |
| P2P | device | 2 | 24.082 | 24.102 | 24.133 | 33.2% |
| other | | | | 183.468 | | |
| sum | | | | 72.516 | | 100.0% |

##### cuda, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 22.490 s, of it tables 0.010 s (loaded from the cache) and device 10.503 s.
- Evaluation over 10 repeats, ms: min 312.733, median 315.430, mean 315.042, max 317.466, std 1.640; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P plane (2 planes per cube); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 4020.8 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 111, syncs 1, timing windows 0.
- Tuning: 8.31 s; M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 67108864 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 262144 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 128 pairs: library (f32 inputs), box-major, 2048 MB (static rule); M2M GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 262144 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 128 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); P2P layout, ≤ 64 points per leaf: P2P plane (2 planes per cube) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 2 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 14 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.940 | 0.963 | 1.144 | 0.8% |
| M2M | device | 7 | 1.003 | 1.013 | 1.021 | 0.8% |
| M2L | device | 5 | 96.278 | 96.330 | 96.444 | 75.7% |
| L2L | device | 7 | 1.267 | 1.312 | 1.396 | 1.0% |
| P2L | device | 1 | 0.073 | 0.074 | 0.075 | 0.1% |
| M2P | device | 1 | 0.351 | 0.356 | 0.361 | 0.3% |
| L2P | device | 2 | 3.118 | 3.178 | 3.233 | 2.5% |
| P2P | device | 2 | 24.039 | 24.093 | 24.159 | 18.9% |
| other | | | | 188.574 | | |
| sum | | | | 127.319 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, CUDA, tuned, f64

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:42:17 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:42:17 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e5,1e6,1e7 --precision f64 --degree 3,6,8,12,18 --tuning-cache L/tune-bench/f64 --table-cache L/tables --output L/bench/cuda-tuned-f64.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | `L/tune-bench/f64` |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f64 | 100000 | 3 | Dense | 3.984 | 1.911 | 1.940 | 1.994 | 2.225 | 0.100 | 3.39e-3 | 1.99e-3 | yes |
| cuda | f64 | 100000 | 6 | Dense | 3.749 | 2.497 | 2.555 | 2.544 | 2.566 | 0.025 | 1.42e-4 | 1.30e-4 | yes |
| cuda | f64 | 100000 | 8 | Dense | 3.753 | 3.760 | 3.785 | 3.792 | 3.849 | 0.025 | 2.12e-5 | 2.37e-5 | yes |
| cuda | f64 | 100000 | 12 | Dense | 4.151 | 8.475 | 8.492 | 8.499 | 8.545 | 0.023 | 8.07e-7 | 1.11e-6 | yes |
| cuda | f64 | 100000 | 18 | Dense | 18.551 | 26.412 | 26.461 | 26.473 | 26.621 | 0.060 | 1.01e-8 | 2.37e-8 | yes |
| cuda | f64 | 1000000 | 3 | Dense | 2.537 | 19.892 | 20.560 | 20.504 | 20.903 | 0.341 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 6 | Dense | 2.417 | 24.508 | 25.475 | 25.430 | 26.175 | 0.446 | 8.53e-5 | 2.56e-4 | yes |
| cuda | f64 | 1000000 | 8 | Dense | 2.501 | 33.878 | 35.268 | 35.055 | 35.370 | 0.457 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 1000000 | 12 | Dense | 3.554 | 74.312 | 76.389 | 76.204 | 76.680 | 0.702 | 4.35e-7 | 2.17e-6 | yes |
| cuda | f64 | 1000000 | 18 | Dense | 10.806 | 241.604 | 243.452 | 243.242 | 243.825 | 0.633 | 6.32e-9 | 5.58e-8 | yes |
| cuda | f64 | 10000000 | 3 | Dense | 19.693 | 356.594 | 361.077 | 363.693 | 394.636 | 11.053 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f64 | 10000000 | 6 | Dense | 21.596 | 412.692 | 415.426 | 415.423 | 417.725 | 1.807 | 7.90e-5 | 1.23e-4 | yes |
| cuda | f64 | 10000000 | 8 | Dense | 24.530 | 513.336 | 517.091 | 516.715 | 519.400 | 1.943 | 1.21e-5 | 2.26e-5 | yes |
| cuda | f64 | 10000000 | 12 | Dense | 24.521 | 1017.487 | 1022.384 | 1021.767 | 1026.457 | 2.890 | 4.40e-7 | 1.21e-6 | yes |
| cuda | f64 | 10000000 | 18 | Dense | 27.184 | 3073.488 | 3080.903 | 3080.447 | 3083.747 | 3.272 | 6.50e-9 | 3.01e-8 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f64 | 100000 | 3 | 0.030 | 0.158 | 0.407 | 0.139 | 0.000 | 0.000 | 0.111 | 0.398 | 1.182 | 1.242 | 2.472 | 0.623 |
| cuda | f64 | 100000 | 6 | 0.034 | 0.170 | 0.821 | 0.358 | 0.000 | 0.000 | 0.170 | 0.397 | 1.253 | 1.949 | 3.274 | 0.766 |
| cuda | f64 | 100000 | 8 | 0.050 | 0.210 | 1.846 | 0.289 | 0.000 | 0.000 | 0.132 | 0.397 | 1.224 | 2.924 | 4.249 | 0.771 |
| cuda | f64 | 100000 | 12 | 0.118 | 0.348 | 5.727 | 0.686 | 0.000 | 0.000 | 0.615 | 0.556 | 1.262 | 8.050 | 9.561 | 0.947 |
| cuda | f64 | 100000 | 18 | 0.640 | 1.172 | 21.902 | 1.007 | 0.000 | 0.000 | 1.579 | 0.457 | 1.173 | 26.756 | 28.346 | 1.011 |
| cuda | f64 | 1000000 | 3 | 0.084 | 0.203 | 1.636 | 0.257 | 0.000 | 0.000 | 0.192 | 3.136 | 12.979 | 5.507 | 19.139 | 0.269 |
| cuda | f64 | 1000000 | 6 | 0.120 | 0.239 | 6.018 | 0.527 | 0.000 | 0.000 | 0.517 | 3.322 | 13.117 | 10.743 | 24.833 | 0.422 |
| cuda | f64 | 1000000 | 8 | 0.268 | 0.579 | 14.854 | 0.448 | 0.000 | 0.000 | 0.676 | 3.211 | 13.175 | 20.036 | 34.495 | 0.572 |
| cuda | f64 | 1000000 | 12 | 0.906 | 0.889 | 49.858 | 1.109 | 0.000 | 0.000 | 4.824 | 3.217 | 13.183 | 60.804 | 76.151 | 0.798 |
| cuda | f64 | 1000000 | 18 | 5.543 | 2.437 | 200.258 | 2.325 | 0.000 | 0.000 | 12.496 | 3.208 | 13.451 | 226.267 | 243.776 | 0.930 |
| cuda | f64 | 10000000 | 3 | 0.653 | 0.515 | 14.086 | 0.553 | 0.080 | 0.496 | 0.666 | 38.976 | 303.240 | 56.025 | 365.914 | 0.154 |
| cuda | f64 | 10000000 | 6 | 0.973 | 0.888 | 65.534 | 1.270 | 0.077 | 0.569 | 2.875 | 38.930 | 298.952 | 111.116 | 419.269 | 0.267 |
| cuda | f64 | 10000000 | 8 | 2.420 | 1.437 | 156.299 | 1.416 | 0.079 | 0.613 | 5.991 | 38.892 | 299.997 | 207.147 | 518.915 | 0.401 |
| cuda | f64 | 10000000 | 12 | 8.798 | 3.602 | 601.628 | 3.886 | 0.107 | 0.779 | 48.687 | 38.865 | 298.321 | 706.351 | 1023.363 | 0.691 |
| cuda | f64 | 10000000 | 18 | 53.623 | 11.284 | 2502.867 | 11.513 | 0.293 | 0.929 | 128.599 | 39.123 | 298.470 | 2748.232 | 3080.584 | 0.892 |

##### cuda, f64, N = 100000, p = 3

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 3.984 s, of it tables 0.001 s (loaded from the cache) and device 3.909 s.
- Evaluation over 10 repeats, ms: min 1.911, median 1.940, mean 1.994, max 2.225, std 0.100; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 3.39e-3, ∇φ 1.99e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 175.4 MB.
- One evaluation: uploads 1 (800000 B), downloads 1 (3200000 B), launches 40, syncs 1, timing windows 0.
- Tuning: 3.89 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.029 | 0.030 | 0.030 | 2.4% |
| M2M | device | 4 | 0.155 | 0.158 | 0.161 | 12.7% |
| M2L | device | 3 | 0.405 | 0.407 | 0.409 | 32.8% |
| L2L | device | 4 | 0.136 | 0.139 | 0.142 | 11.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.102 | 0.111 | 0.116 | 8.9% |
| P2P | device | 1 | 0.392 | 0.398 | 0.414 | 32.0% |
| other | | | | 1.182 | | |
| sum | | | | 1.242 | | 100.0% |

##### cuda, f64, N = 100000, p = 6

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 3.749 s, of it tables 0.007 s (loaded from the cache) and device 3.670 s.
- Evaluation over 10 repeats, ms: min 2.497, median 2.555, mean 2.544, max 2.566, std 0.025; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.42e-4, ∇φ 1.30e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 492.0 MB.
- One evaluation: uploads 1 (800000 B), downloads 1 (3200000 B), launches 40, syncs 1, timing windows 0.
- Tuning: 3.65 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.033 | 0.034 | 0.034 | 1.7% |
| M2M | device | 4 | 0.168 | 0.170 | 0.172 | 8.7% |
| M2L | device | 3 | 0.819 | 0.821 | 0.824 | 42.1% |
| L2L | device | 4 | 0.351 | 0.358 | 0.363 | 18.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.165 | 0.170 | 0.178 | 8.7% |
| P2P | device | 1 | 0.393 | 0.397 | 0.414 | 20.4% |
| other | | | | 1.253 | | |
| sum | | | | 1.949 | | 100.0% |

##### cuda, f64, N = 100000, p = 8

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 3.753 s, of it tables 0.019 s (loaded from the cache) and device 3.661 s.
- Evaluation over 10 repeats, ms: min 3.760, median 3.785, mean 3.792, max 3.849, std 0.025; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.12e-5, ∇φ 2.37e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 804.8 MB.
- One evaluation: uploads 1 (800000 B), downloads 1 (3200000 B), launches 40, syncs 1, timing windows 0.
- Tuning: 3.64 s; M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.049 | 0.050 | 0.053 | 1.7% |
| M2M | device | 4 | 0.206 | 0.210 | 0.211 | 7.2% |
| M2L | device | 3 | 1.844 | 1.846 | 1.850 | 63.1% |
| L2L | device | 4 | 0.279 | 0.289 | 0.295 | 9.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.122 | 0.132 | 0.144 | 4.5% |
| P2P | device | 1 | 0.394 | 0.397 | 0.405 | 13.6% |
| other | | | | 1.224 | | |
| sum | | | | 2.924 | | 100.0% |

##### cuda, f64, N = 100000, p = 12

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 4.151 s, of it tables 0.084 s (loaded from the cache) and device 3.994 s.
- Evaluation over 10 repeats, ms: min 8.475, median 8.492, mean 8.499, max 8.545, std 0.023; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.07e-7, ∇φ 1.11e-6.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(64 x 1 units, 8 columns per unit); rotation cube (192 units); device memory 1692.7 MB.
- One evaluation: uploads 1 (800000 B), downloads 1 (3200000 B), launches 40, syncs 1, timing windows 0.
- Tuning: 3.95 s; M2L strategy: Dense, hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.118 | 0.118 | 0.119 | 1.5% |
| M2M | device | 4 | 0.328 | 0.348 | 0.508 | 4.3% |
| M2L | device | 3 | 5.718 | 5.727 | 5.736 | 71.1% |
| L2L | device | 4 | 0.662 | 0.686 | 0.706 | 8.5% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.566 | 0.615 | 0.765 | 7.6% |
| P2P | device | 1 | 0.545 | 0.556 | 0.573 | 6.9% |
| other | | | | 1.262 | | |
| sum | | | | 8.050 | | 100.0% |

##### cuda, f64, N = 100000, p = 18

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 18.551 s, of it tables 0.836 s (built) and device 17.641 s.
- Evaluation over 10 repeats, ms: min 26.412, median 26.461, mean 26.473, max 26.621, std 0.060; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.01e-8, ∇φ 2.37e-8.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(128 x 1 units, 8 columns per unit); rotation cube (384 units); device memory 2545.4 MB.
- One evaluation: uploads 1 (800000 B), downloads 1 (3200000 B), launches 43, syncs 1, timing windows 0.
- Tuning: 17.53 s; M2L strategy: Dense, hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (static rule).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 2 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.637 | 0.640 | 0.646 | 2.4% |
| M2M | device | 4 | 1.164 | 1.172 | 1.196 | 4.4% |
| M2L | device | 3 | 21.889 | 21.902 | 21.912 | 81.9% |
| L2L | device | 4 | 0.996 | 1.007 | 1.017 | 3.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 1.563 | 1.579 | 1.607 | 5.9% |
| P2P | device | 1 | 0.432 | 0.457 | 0.492 | 1.7% |
| other | | | | 1.173 | | |
| sum | | | | 26.756 | | 100.0% |

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 2.537 s, of it tables 0.001 s (loaded from the cache) and device 1.751 s.
- Evaluation over 10 repeats, ms: min 19.892, median 20.560, mean 20.504, max 20.903, std 0.341; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 1623.1 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 49, syncs 1, timing windows 0.
- Tuning: 1.54 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (cache).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.083 | 0.084 | 0.085 | 1.5% |
| M2M | device | 5 | 0.200 | 0.203 | 0.205 | 3.7% |
| M2L | device | 4 | 1.629 | 1.636 | 1.643 | 29.7% |
| L2L | device | 5 | 0.252 | 0.257 | 0.263 | 4.7% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.187 | 0.192 | 0.198 | 3.5% |
| P2P | device | 1 | 3.116 | 3.136 | 3.163 | 56.9% |
| other | | | | 12.979 | | |
| sum | | | | 5.507 | | 100.0% |

##### cuda, f64, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 2.417 s, of it tables 0.007 s (loaded from the cache) and device 1.599 s.
- Evaluation over 10 repeats, ms: min 24.508, median 25.475, mean 25.430, max 26.175, std 0.446; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 2414.0 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 52, syncs 1, timing windows 0.
- Tuning: 1.41 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (cache).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 2 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.119 | 0.120 | 0.121 | 1.1% |
| M2M | device | 5 | 0.238 | 0.239 | 0.242 | 2.2% |
| M2L | device | 4 | 6.011 | 6.018 | 6.022 | 56.0% |
| L2L | device | 5 | 0.359 | 0.527 | 0.553 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.509 | 0.517 | 0.529 | 4.8% |
| P2P | device | 1 | 3.291 | 3.322 | 3.359 | 30.9% |
| other | | | | 13.117 | | |
| sum | | | | 10.743 | | 100.0% |

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 2.501 s, of it tables 0.020 s (loaded from the cache) and device 1.697 s.
- Evaluation over 10 repeats, ms: min 33.878, median 35.268, mean 35.055, max 35.370, std 0.457; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2446.5 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 58, syncs 1, timing windows 0.
- Tuning: 1.44 s; M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 8388608 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (cache).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 4 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.267 | 0.268 | 0.269 | 1.3% |
| M2M | device | 5 | 0.398 | 0.579 | 0.605 | 2.9% |
| M2L | device | 4 | 14.843 | 14.854 | 14.863 | 74.1% |
| L2L | device | 5 | 0.437 | 0.448 | 0.459 | 2.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.652 | 0.676 | 0.841 | 3.4% |
| P2P | device | 1 | 3.181 | 3.211 | 3.247 | 16.0% |
| other | | | | 13.175 | | |
| sum | | | | 20.036 | | 100.0% |

##### cuda, f64, N = 1000000, p = 12

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 3.554 s, of it tables 0.085 s (loaded from the cache) and device 2.688 s.
- Evaluation over 10 repeats, ms: min 74.312, median 76.389, mean 76.204, max 76.680, std 0.702; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 4.35e-7, ∇φ 2.17e-6.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(64 x 1 units, 8 columns per unit); rotation cube (192 units); device memory 2562.2 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 67, syncs 1, timing windows 0.
- Tuning: 2.42 s; M2L strategy: Dense, hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 8388608 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 32768 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (cache).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 7 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.902 | 0.906 | 0.910 | 1.5% |
| M2M | device | 5 | 0.877 | 0.889 | 0.896 | 1.5% |
| M2L | device | 4 | 49.742 | 49.858 | 50.620 | 82.0% |
| L2L | device | 5 | 1.058 | 1.109 | 1.262 | 1.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 4.797 | 4.824 | 4.873 | 7.9% |
| P2P | device | 1 | 3.168 | 3.217 | 3.270 | 5.3% |
| other | | | | 13.183 | | |
| sum | | | | 60.804 | | 100.0% |

##### cuda, f64, N = 1000000, p = 18

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 10.806 s, of it tables 0.384 s (loaded from the cache) and device 9.642 s.
- Evaluation over 10 repeats, ms: min 241.604, median 243.452, mean 243.242, max 243.825, std 0.633; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 6.32e-9, ∇φ 5.58e-8.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(128 x 1 units, 8 columns per unit); rotation cube (384 units); device memory 2947.5 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 94, syncs 1, timing windows 0.
- Tuning: 9.30 s; M2L strategy: Dense, hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 8388608 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 32768 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (cache).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 2 chunks
  - M2L level 5: 5398920 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 15 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 5.526 | 5.543 | 5.557 | 2.4% |
| M2M | device | 5 | 2.416 | 2.437 | 2.447 | 1.1% |
| M2L | device | 4 | 200.193 | 200.258 | 200.322 | 88.5% |
| L2L | device | 5 | 2.298 | 2.325 | 2.353 | 1.0% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 12.410 | 12.496 | 12.622 | 5.5% |
| P2P | device | 1 | 3.128 | 3.208 | 3.359 | 1.4% |
| other | | | | 13.451 | | |
| sum | | | | 226.267 | | 100.0% |

##### cuda, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 19.693 s, of it tables 0.001 s (loaded from the cache) and device 7.353 s.
- Evaluation over 10 repeats, ms: min 356.594, median 361.077, mean 363.693, max 394.636, std 11.053; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 4353.3 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 84, syncs 1, timing windows 0.
- Tuning: 5.18 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 67108864 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 262144 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 128 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 128 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 64 points per leaf: P2P cube (32 units) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 2 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 8 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 6 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.643 | 0.653 | 0.664 | 1.2% |
| M2M | device | 7 | 0.510 | 0.515 | 0.521 | 0.9% |
| M2L | device | 5 | 14.066 | 14.086 | 14.099 | 25.1% |
| L2L | device | 7 | 0.527 | 0.553 | 0.577 | 1.0% |
| P2L | device | 1 | 0.079 | 0.080 | 0.085 | 0.1% |
| M2P | device | 1 | 0.488 | 0.496 | 0.501 | 0.9% |
| L2P | device | 2 | 0.620 | 0.666 | 0.776 | 1.2% |
| P2P | device | 2 | 38.943 | 38.976 | 39.042 | 69.6% |
| other | | | | 303.240 | | |
| sum | | | | 56.025 | | 100.0% |

##### cuda, f64, N = 10000000, p = 6

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 21.596 s, of it tables 0.007 s (loaded from the cache) and device 9.243 s.
- Evaluation over 10 repeats, ms: min 412.692, median 415.426, mean 415.423, max 417.725, std 1.807; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 7.90e-5, ∇φ 1.23e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 4517.3 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 120, syncs 1, timing windows 0.
- Tuning: 7.06 s; M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 67108864 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 128 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 128 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 64 points per leaf: P2P cube (32 units) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 2 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 17 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.968 | 0.973 | 0.982 | 0.9% |
| M2M | device | 7 | 0.879 | 0.888 | 0.896 | 0.8% |
| M2L | device | 5 | 65.515 | 65.534 | 65.568 | 59.0% |
| L2L | device | 7 | 1.232 | 1.270 | 1.306 | 1.1% |
| P2L | device | 1 | 0.076 | 0.077 | 0.082 | 0.1% |
| M2P | device | 1 | 0.565 | 0.569 | 0.575 | 0.5% |
| L2P | device | 2 | 2.832 | 2.875 | 3.002 | 2.6% |
| P2P | device | 2 | 38.893 | 38.930 | 38.991 | 35.0% |
| other | | | | 298.952 | | |
| sum | | | | 111.116 | | 100.0% |

##### cuda, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 24.530 s, of it tables 0.020 s (loaded from the cache) and device 12.071 s.
- Evaluation over 10 repeats, ms: min 513.336, median 517.091, mean 516.715, max 519.400, std 1.943; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 4705.7 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 159, syncs 1, timing windows 0.
- Tuning: 9.09 s; M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 67108864 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 262144 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 128 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 262144 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 128 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 64 points per leaf: P2P cube (32 units) (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 4 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 28 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 2.367 | 2.420 | 2.582 | 1.2% |
| M2M | device | 7 | 1.429 | 1.437 | 1.452 | 0.7% |
| M2L | device | 5 | 156.257 | 156.299 | 156.395 | 75.5% |
| L2L | device | 7 | 1.342 | 1.416 | 1.533 | 0.7% |
| P2L | device | 1 | 0.077 | 0.079 | 0.080 | 0.0% |
| M2P | device | 1 | 0.608 | 0.613 | 0.623 | 0.3% |
| L2P | device | 2 | 5.885 | 5.991 | 6.074 | 2.9% |
| P2P | device | 2 | 38.833 | 38.892 | 38.992 | 18.8% |
| other | | | | 299.997 | | |
| sum | | | | 207.147 | | 100.0% |

##### cuda, f64, N = 10000000, p = 12

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 24.521 s, of it tables 0.084 s (loaded from the cache) and device 12.378 s.
- Evaluation over 10 repeats, ms: min 1017.487, median 1022.384, mean 1021.767, max 1026.457, std 2.890; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 4.40e-7, ∇φ 1.21e-6.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(64 x 1 units, 8 columns per unit); rotation cube (192 units); device memory 5221.6 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 261, syncs 1, timing windows 0.
- Tuning: 10.12 s; M2L strategy: Dense, hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 67108864 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 1048576 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 262144 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 32768 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 128 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 262144 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 128 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 64 points per leaf: P2P cube (32 units) (static rule).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 1 units, 2 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(64 x 2 units, 4 columns per unit), 7 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 59 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 8.770 | 8.798 | 8.920 | 1.2% |
| M2M | device | 7 | 3.584 | 3.602 | 3.617 | 0.5% |
| M2L | device | 5 | 601.588 | 601.628 | 601.743 | 85.2% |
| L2L | device | 7 | 3.705 | 3.886 | 4.012 | 0.6% |
| P2L | device | 1 | 0.104 | 0.107 | 0.113 | 0.0% |
| M2P | device | 1 | 0.698 | 0.779 | 0.868 | 0.1% |
| L2P | device | 2 | 48.570 | 48.687 | 48.921 | 6.9% |
| P2P | device | 2 | 38.836 | 38.865 | 38.966 | 5.5% |
| other | | | | 298.321 | | |
| sum | | | | 706.351 | | 100.0% |

##### cuda, f64, N = 10000000, p = 18

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 27.184 s, of it tables 0.383 s (loaded from the cache) and device 14.587 s.
- Evaluation over 10 repeats, ms: min 3073.488, median 3080.903, mean 3080.447, max 3083.747, std 3.272; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 6.50e-9, ∇φ 3.01e-8.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(128 x 1 units, 8 columns per unit); rotation cube (384 units); device memory 6420.7 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 486, syncs 1, timing windows 0.
- Tuning: 12.28 s; M2L strategy: Dense, hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 67108864 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 1048576 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 65536 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 262144 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 32768 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (cache); M2M GEMM, ≤ 128 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 262144 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (cache); L2L GEMM, ≤ 128 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 64 points per leaf: P2P cube (32 units) (static rule).
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 2 chunks
  - M2L level 5: 5398920 pairs, hand-written cube(128 x 1 units, 4 columns per unit), 15 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 125 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 53.544 | 53.623 | 53.756 | 2.0% |
| M2M | device | 7 | 11.228 | 11.284 | 11.364 | 0.4% |
| M2L | device | 5 | 2502.674 | 2502.867 | 2503.385 | 91.1% |
| L2L | device | 7 | 11.286 | 11.513 | 11.696 | 0.4% |
| P2L | device | 1 | 0.289 | 0.293 | 0.301 | 0.0% |
| M2P | device | 1 | 0.873 | 0.929 | 1.021 | 0.0% |
| L2P | device | 2 | 128.368 | 128.599 | 128.746 | 4.7% |
| P2P | device | 2 | 39.054 | 39.123 | 39.183 | 1.4% |
| other | | | | 298.470 | | |
| sum | | | | 2748.232 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, CUDA, `--strategy dense`, f32

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:50:58 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:50:58 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6 --precision f32 --degree 3,6,8 --strategy dense --table-cache L/tables --output L/bench/cuda-dense-f32.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | dense |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Dense | 1.034 | 12.011 | 12.247 | 12.251 | 12.547 | 0.173 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 6 | Dense | 1.176 | 15.259 | 15.886 | 15.842 | 16.335 | 0.266 | 8.53e-5 | 2.56e-4 | yes |
| cuda | f32 | 1000000 | 8 | Dense | 0.983 | 21.210 | 21.533 | 21.502 | 21.955 | 0.230 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.055 | 0.190 | 1.287 | 0.298 | 0.000 | 0.000 | 0.071 | 1.997 | 8.431 | 3.898 | 12.581 | 0.318 |
| cuda | f32 | 1000000 | 6 | 0.071 | 0.244 | 4.363 | 0.304 | 0.000 | 0.000 | 0.234 | 2.136 | 8.623 | 7.352 | 16.352 | 0.464 |
| cuda | f32 | 1000000 | 8 | 0.115 | 0.308 | 9.388 | 0.643 | 0.000 | 0.000 | 0.430 | 2.137 | 8.573 | 13.022 | 22.093 | 0.606 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.034 s, of it tables 0.000 s (loaded from the cache) and device 0.251 s.
- Evaluation over 10 repeats, ms: min 12.011, median 12.247, mean 12.251, max 12.547, std 0.173; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 878.9 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 49, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.053 | 0.055 | 0.056 | 1.4% |
| M2M | device | 5 | 0.188 | 0.190 | 0.194 | 4.9% |
| M2L | device | 4 | 1.283 | 1.287 | 1.292 | 33.0% |
| L2L | device | 5 | 0.286 | 0.298 | 0.304 | 7.6% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.063 | 0.071 | 0.077 | 1.8% |
| P2P | device | 1 | 1.976 | 1.997 | 2.017 | 51.2% |
| other | | | | 8.431 | | |
| sum | | | | 3.898 | | 100.0% |

##### cuda, f32, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.176 s, of it tables 0.003 s (loaded from the cache) and device 0.392 s.
- Evaluation over 10 repeats, ms: min 15.259, median 15.886, mean 15.842, max 16.335, std 0.266; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 2317.0 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 49, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.070 | 0.071 | 0.073 | 1.0% |
| M2M | device | 5 | 0.241 | 0.244 | 0.247 | 3.3% |
| M2L | device | 4 | 4.356 | 4.363 | 4.374 | 59.3% |
| L2L | device | 5 | 0.277 | 0.304 | 0.489 | 4.1% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.225 | 0.234 | 0.242 | 3.2% |
| P2P | device | 1 | 2.122 | 2.136 | 2.148 | 29.1% |
| other | | | | 8.623 | | |
| sum | | | | 7.352 | | 100.0% |

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.983 s, of it tables 0.010 s (loaded from the cache) and device 0.190 s.
- Evaluation over 10 repeats, ms: min 21.210, median 21.533, mean 21.502, max 21.955, std 0.230; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2365.8 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 52, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 2 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.114 | 0.115 | 0.116 | 0.9% |
| M2M | device | 5 | 0.305 | 0.308 | 0.314 | 2.4% |
| M2L | device | 4 | 9.375 | 9.388 | 9.411 | 72.1% |
| L2L | device | 5 | 0.634 | 0.643 | 0.649 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.411 | 0.430 | 0.443 | 3.3% |
| P2P | device | 1 | 2.122 | 2.137 | 2.153 | 16.4% |
| other | | | | 8.573 | | |
| sum | | | | 13.022 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, CUDA, `--strategy dense`, f64

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:51:27 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:51:27 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6 --precision f64 --degree 3,6,8,12,18 --strategy dense --table-cache L/tables --output L/bench/cuda-dense-f64.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | dense |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f64 | 1000000 | 3 | Dense | 1.216 | 19.004 | 19.908 | 19.836 | 20.384 | 0.368 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 6 | Dense | 0.972 | 23.656 | 25.257 | 25.110 | 25.513 | 0.537 | 8.53e-5 | 2.56e-4 | yes |
| cuda | f64 | 1000000 | 8 | Dense | 1.004 | 34.465 | 35.512 | 35.434 | 35.974 | 0.406 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 1000000 | 12 | Dense | 1.185 | 77.662 | 78.698 | 78.868 | 81.385 | 0.957 | 4.35e-7 | 2.17e-6 | yes |
| cuda | f64 | 1000000 | 18 | Dense | 1.484 | 248.996 | 250.043 | 250.153 | 251.993 | 0.744 | 6.32e-9 | 5.58e-8 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f64 | 1000000 | 3 | 0.084 | 0.209 | 1.647 | 0.271 | 0.000 | 0.000 | 0.194 | 3.138 | 12.889 | 5.544 | 19.081 | 0.279 |
| cuda | f64 | 1000000 | 6 | 0.120 | 0.338 | 6.029 | 0.556 | 0.000 | 0.000 | 0.502 | 3.326 | 13.076 | 10.871 | 24.913 | 0.433 |
| cuda | f64 | 1000000 | 8 | 0.268 | 0.621 | 15.546 | 0.462 | 0.000 | 0.000 | 0.765 | 3.226 | 13.254 | 20.888 | 35.424 | 0.589 |
| cuda | f64 | 1000000 | 12 | 0.905 | 0.895 | 52.573 | 1.203 | 0.000 | 0.000 | 4.839 | 3.202 | 13.330 | 63.617 | 79.111 | 0.807 |
| cuda | f64 | 1000000 | 18 | 5.534 | 2.429 | 207.673 | 2.338 | 0.000 | 0.000 | 12.483 | 3.264 | 13.335 | 233.721 | 251.108 | 0.934 |

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.216 s, of it tables 0.001 s (loaded from the cache) and device 0.443 s.
- Evaluation over 10 repeats, ms: min 19.004, median 19.908, mean 19.836, max 20.384, std 0.368; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 1623.1 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 49, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.083 | 0.084 | 0.086 | 1.5% |
| M2M | device | 5 | 0.202 | 0.209 | 0.215 | 3.8% |
| M2L | device | 4 | 1.641 | 1.647 | 1.654 | 29.7% |
| L2L | device | 5 | 0.264 | 0.271 | 0.278 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.184 | 0.194 | 0.202 | 3.5% |
| P2P | device | 1 | 3.099 | 3.138 | 3.175 | 56.6% |
| other | | | | 12.889 | | |
| sum | | | | 5.544 | | 100.0% |

##### cuda, f64, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.972 s, of it tables 0.007 s (loaded from the cache) and device 0.192 s.
- Evaluation over 10 repeats, ms: min 23.656, median 25.257, mean 25.110, max 25.513, std 0.537; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 2413.9 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 52, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 2 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.119 | 0.120 | 0.121 | 1.1% |
| M2M | device | 5 | 0.252 | 0.338 | 0.467 | 3.1% |
| M2L | device | 4 | 6.020 | 6.029 | 6.044 | 55.5% |
| L2L | device | 5 | 0.541 | 0.556 | 0.571 | 5.1% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.332 | 0.502 | 0.530 | 4.6% |
| P2P | device | 1 | 3.285 | 3.326 | 3.367 | 30.6% |
| other | | | | 13.076 | | |
| sum | | | | 10.871 | | 100.0% |

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.004 s, of it tables 0.020 s (loaded from the cache) and device 0.201 s.
- Evaluation over 10 repeats, ms: min 34.465, median 35.512, mean 35.434, max 35.974, std 0.406; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2446.4 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 58, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 4 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.267 | 0.268 | 0.270 | 1.3% |
| M2M | device | 5 | 0.618 | 0.621 | 0.626 | 3.0% |
| M2L | device | 4 | 15.539 | 15.546 | 15.555 | 74.4% |
| L2L | device | 5 | 0.450 | 0.462 | 0.472 | 2.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.756 | 0.765 | 0.773 | 3.7% |
| P2P | device | 1 | 3.205 | 3.226 | 3.285 | 15.4% |
| other | | | | 13.254 | | |
| sum | | | | 20.888 | | 100.0% |

##### cuda, f64, N = 1000000, p = 12

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.185 s, of it tables 0.083 s (loaded from the cache) and device 0.333 s.
- Evaluation over 10 repeats, ms: min 77.662, median 78.698, mean 78.868, max 81.385, std 0.957; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 4.35e-7, ∇φ 2.17e-6.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(64 x 1 units, 8 columns per unit); rotation cube (192 units); device memory 2562.2 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 67, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 7 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.904 | 0.905 | 0.908 | 1.4% |
| M2M | device | 5 | 0.890 | 0.895 | 0.899 | 1.4% |
| M2L | device | 4 | 52.538 | 52.573 | 52.631 | 82.6% |
| L2L | device | 5 | 1.071 | 1.203 | 1.270 | 1.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 4.765 | 4.839 | 4.890 | 7.6% |
| P2P | device | 1 | 3.158 | 3.202 | 3.240 | 5.0% |
| other | | | | 13.330 | | |
| sum | | | | 63.617 | | 100.0% |

##### cuda, f64, N = 1000000, p = 18

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.484 s, of it tables 0.387 s (loaded from the cache) and device 0.304 s.
- Evaluation over 10 repeats, ms: min 248.996, median 250.043, mean 250.153, max 251.993, std 0.744; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 6.32e-9, ∇φ 5.58e-8.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(128 x 1 units, 8 columns per unit); rotation cube (384 units); device memory 2947.5 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 94, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 2 chunks
  - M2L level 5: 5398920 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 15 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 5.521 | 5.534 | 5.546 | 2.4% |
| M2M | device | 5 | 2.412 | 2.429 | 2.443 | 1.0% |
| M2L | device | 4 | 207.525 | 207.673 | 207.718 | 88.9% |
| L2L | device | 5 | 2.312 | 2.338 | 2.364 | 1.0% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 12.441 | 12.483 | 12.524 | 5.3% |
| P2P | device | 1 | 3.118 | 3.264 | 3.329 | 1.4% |
| other | | | | 13.335 | | |
| sum | | | | 233.721 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, CUDA, `--strategy classes`, f32

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:52:11 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:52:11 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6 --precision f32 --degree 3,6,8 --strategy classes --table-cache L/tables --output L/bench/cuda-classes-f32.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | classes |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Classes, run as dense on the device | 1.032 | 11.839 | 12.131 | 12.153 | 12.587 | 0.237 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 6 | Classes, run as dense on the device | 1.183 | 15.387 | 15.602 | 15.657 | 16.347 | 0.283 | 8.53e-5 | 2.56e-4 | yes |
| cuda | f32 | 1000000 | 8 | Classes, run as dense on the device | 1.011 | 21.012 | 21.241 | 21.295 | 21.806 | 0.242 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.055 | 0.192 | 1.286 | 0.295 | 0.000 | 0.000 | 0.069 | 2.006 | 8.372 | 3.903 | 12.532 | 0.321 |
| cuda | f32 | 1000000 | 6 | 0.070 | 0.242 | 4.361 | 0.324 | 0.000 | 0.000 | 0.239 | 2.139 | 8.326 | 7.374 | 16.077 | 0.471 |
| cuda | f32 | 1000000 | 8 | 0.114 | 0.310 | 9.387 | 0.643 | 0.000 | 0.000 | 0.431 | 2.141 | 8.565 | 13.026 | 22.120 | 0.612 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Classes, run as dense on the device; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.032 s, of it tables 0.001 s (built) and device 0.242 s.
- Evaluation over 10 repeats, ms: min 11.839, median 12.131, mean 12.153, max 12.587, std 0.237; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 878.9 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 49, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.054 | 0.055 | 0.057 | 1.4% |
| M2M | device | 5 | 0.188 | 0.192 | 0.197 | 4.9% |
| M2L | device | 4 | 1.283 | 1.286 | 1.291 | 32.9% |
| L2L | device | 5 | 0.291 | 0.295 | 0.302 | 7.6% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.059 | 0.069 | 0.080 | 1.8% |
| P2P | device | 1 | 1.980 | 2.006 | 2.031 | 51.4% |
| other | | | | 8.372 | | |
| sum | | | | 3.903 | | 100.0% |

##### cuda, f32, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Classes, run as dense on the device; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.183 s, of it tables 0.004 s (built) and device 0.400 s.
- Evaluation over 10 repeats, ms: min 15.387, median 15.602, mean 15.657, max 16.347, std 0.283; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 2317.0 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 49, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.069 | 0.070 | 0.071 | 0.9% |
| M2M | device | 5 | 0.239 | 0.242 | 0.244 | 3.3% |
| M2L | device | 4 | 4.356 | 4.361 | 4.369 | 59.1% |
| L2L | device | 5 | 0.279 | 0.324 | 0.490 | 4.4% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.225 | 0.239 | 0.253 | 3.2% |
| P2P | device | 1 | 2.115 | 2.139 | 2.156 | 29.0% |
| other | | | | 8.326 | | |
| sum | | | | 7.374 | | 100.0% |

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Classes, run as dense on the device; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.011 s, of it tables 0.014 s (built) and device 0.200 s.
- Evaluation over 10 repeats, ms: min 21.012, median 21.241, mean 21.295, max 21.806, std 0.242; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2365.8 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 52, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 2 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.112 | 0.114 | 0.115 | 0.9% |
| M2M | device | 5 | 0.306 | 0.310 | 0.314 | 2.4% |
| M2L | device | 4 | 9.379 | 9.387 | 9.400 | 72.1% |
| L2L | device | 5 | 0.626 | 0.643 | 0.655 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.400 | 0.431 | 0.445 | 3.3% |
| P2P | device | 1 | 2.122 | 2.141 | 2.157 | 16.4% |
| other | | | | 8.565 | | |
| sum | | | | 13.026 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, CUDA, `--strategy classes`, f64

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:52:40 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:52:40 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6 --precision f64 --degree 3,6,8,12,18 --strategy classes --table-cache L/tables --output L/bench/cuda-classes-f64.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | classes |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f64 | 1000000 | 3 | Classes, run as dense on the device | 1.221 | 18.475 | 19.698 | 19.535 | 20.108 | 0.533 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 6 | Classes, run as dense on the device | 0.997 | 23.910 | 24.948 | 24.911 | 25.643 | 0.436 | 8.53e-5 | 2.56e-4 | yes |
| cuda | f64 | 1000000 | 8 | Classes, run as dense on the device | 1.036 | 34.525 | 35.632 | 35.565 | 36.069 | 0.404 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 1000000 | 12 | Classes, run as dense on the device | 1.295 | 77.653 | 79.059 | 78.831 | 79.204 | 0.495 | 4.35e-7 | 2.17e-6 | yes |
| cuda | f64 | 1000000 | 18 | Classes, run as dense on the device | 2.539 | 248.739 | 250.247 | 250.321 | 252.232 | 0.926 | 6.32e-9 | 5.58e-8 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f64 | 1000000 | 3 | 0.085 | 0.205 | 1.641 | 0.268 | 0.000 | 0.000 | 0.197 | 3.147 | 13.052 | 5.543 | 19.239 | 0.284 |
| cuda | f64 | 1000000 | 6 | 0.120 | 0.269 | 6.028 | 0.558 | 0.000 | 0.000 | 0.483 | 3.321 | 13.310 | 10.780 | 25.058 | 0.433 |
| cuda | f64 | 1000000 | 8 | 0.268 | 0.607 | 15.544 | 0.481 | 0.000 | 0.000 | 0.772 | 3.239 | 13.351 | 20.911 | 35.544 | 0.588 |
| cuda | f64 | 1000000 | 12 | 0.907 | 0.895 | 52.571 | 1.216 | 0.000 | 0.000 | 4.847 | 3.205 | 13.161 | 63.640 | 78.957 | 0.807 |
| cuda | f64 | 1000000 | 18 | 5.543 | 2.471 | 207.653 | 2.348 | 0.000 | 0.000 | 12.497 | 3.214 | 14.192 | 233.725 | 252.234 | 0.934 |

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Classes, run as dense on the device; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.221 s, of it tables 0.001 s (built) and device 0.446 s.
- Evaluation over 10 repeats, ms: min 18.475, median 19.698, mean 19.535, max 20.108, std 0.533; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 1623.1 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 49, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.084 | 0.085 | 0.087 | 1.5% |
| M2M | device | 5 | 0.200 | 0.205 | 0.210 | 3.7% |
| M2L | device | 4 | 1.635 | 1.641 | 1.646 | 29.6% |
| L2L | device | 5 | 0.260 | 0.268 | 0.275 | 4.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.190 | 0.197 | 0.203 | 3.6% |
| P2P | device | 1 | 3.120 | 3.147 | 3.174 | 56.8% |
| other | | | | 13.052 | | |
| sum | | | | 5.543 | | 100.0% |

##### cuda, f64, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Classes, run as dense on the device; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.997 s, of it tables 0.005 s (built) and device 0.196 s.
- Evaluation over 10 repeats, ms: min 23.910, median 24.948, mean 24.911, max 25.643, std 0.436; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 2413.9 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 52, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 2 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.120 | 0.120 | 0.122 | 1.1% |
| M2M | device | 5 | 0.247 | 0.269 | 0.450 | 2.5% |
| M2L | device | 4 | 6.019 | 6.028 | 6.036 | 55.9% |
| L2L | device | 5 | 0.547 | 0.558 | 0.565 | 5.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.330 | 0.483 | 0.531 | 4.5% |
| P2P | device | 1 | 3.301 | 3.321 | 3.372 | 30.8% |
| other | | | | 13.310 | | |
| sum | | | | 10.780 | | 100.0% |

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Classes, run as dense on the device; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.036 s, of it tables 0.016 s (built) and device 0.214 s.
- Evaluation over 10 repeats, ms: min 34.525, median 35.632, mean 35.565, max 36.069, std 0.404; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2446.4 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 58, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 4 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.267 | 0.268 | 0.269 | 1.3% |
| M2M | device | 5 | 0.432 | 0.607 | 0.630 | 2.9% |
| M2L | device | 4 | 15.536 | 15.544 | 15.558 | 74.3% |
| L2L | device | 5 | 0.461 | 0.481 | 0.491 | 2.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.760 | 0.772 | 0.782 | 3.7% |
| P2P | device | 1 | 3.183 | 3.239 | 3.284 | 15.5% |
| other | | | | 13.351 | | |
| sum | | | | 20.911 | | 100.0% |

##### cuda, f64, N = 1000000, p = 12

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Classes, run as dense on the device; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.295 s, of it tables 0.107 s (built) and device 0.414 s.
- Evaluation over 10 repeats, ms: min 77.653, median 79.059, mean 78.831, max 79.204, std 0.495; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 4.35e-7, ∇φ 2.17e-6.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(64 x 1 units, 8 columns per unit); rotation cube (192 units); device memory 2562.2 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 67, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 7 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.903 | 0.907 | 0.911 | 1.4% |
| M2M | device | 5 | 0.890 | 0.895 | 0.899 | 1.4% |
| M2L | device | 4 | 52.546 | 52.571 | 52.591 | 82.6% |
| L2L | device | 5 | 1.055 | 1.216 | 1.262 | 1.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 4.789 | 4.847 | 4.907 | 7.6% |
| P2P | device | 1 | 3.149 | 3.205 | 3.244 | 5.0% |
| other | | | | 13.161 | | |
| sum | | | | 63.640 | | 100.0% |

##### cuda, f64, N = 1000000, p = 18

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Classes, run as dense on the device; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 2.539 s, of it tables 0.913 s (built) and device 0.834 s.
- Evaluation over 10 repeats, ms: min 248.739, median 250.247, mean 250.321, max 252.232, std 0.926; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 6.32e-9, ∇φ 5.58e-8.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(128 x 1 units, 8 columns per unit); rotation cube (384 units); device memory 2947.5 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 94, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 2 chunks
  - M2L level 5: 5398920 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 15 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 5.525 | 5.543 | 5.562 | 2.4% |
| M2M | device | 5 | 2.422 | 2.471 | 2.617 | 1.1% |
| M2L | device | 4 | 207.502 | 207.653 | 207.764 | 88.8% |
| L2L | device | 5 | 2.323 | 2.348 | 2.362 | 1.0% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 12.441 | 12.497 | 12.541 | 5.3% |
| P2P | device | 1 | 3.143 | 3.214 | 3.331 | 1.4% |
| other | | | | 14.192 | | |
| sum | | | | 233.725 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, CUDA, `--strategy rotation`, f32

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:53:26 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:53:26 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6 --precision f32 --degree 3,6,8 --strategy rotation --table-cache L/tables --output L/bench/cuda-rotation-f32.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | rotation |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Rotation | 1.034 | 13.983 | 14.433 | 14.418 | 14.922 | 0.300 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 6 | Rotation | 0.991 | 19.648 | 19.973 | 19.957 | 20.529 | 0.259 | 8.53e-5 | 2.56e-4 | yes |
| cuda | f32 | 1000000 | 8 | Rotation | 0.992 | 26.166 | 26.384 | 26.450 | 26.989 | 0.251 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.055 | 0.196 | 3.417 | 0.380 | 0.000 | 0.000 | 0.172 | 1.995 | 8.406 | 6.216 | 14.872 | 0.431 |
| cuda | f32 | 1000000 | 6 | 0.072 | 0.245 | 8.570 | 0.557 | 0.000 | 0.000 | 0.232 | 2.145 | 8.374 | 11.822 | 20.553 | 0.592 |
| cuda | f32 | 1000000 | 8 | 0.115 | 0.356 | 14.406 | 0.617 | 0.000 | 0.000 | 0.443 | 2.142 | 8.616 | 18.080 | 27.189 | 0.684 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.034 s, of it tables 0.000 s (loaded from the cache) and device 0.233 s.
- Evaluation over 10 repeats, ms: min 13.983, median 14.433, mean 14.418, max 14.922, std 0.300; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 189.6 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 41, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L (rotation) level 2: 3096 pairs, 64 rows, cube (32 units)
  - M2L (rotation) level 3: 53352 pairs, 512 rows, cube (32 units)
  - M2L (rotation) level 4: 584136 pairs, 4096 rows, cube (32 units)
  - M2L (rotation) level 5: 5398920 pairs, 32768 rows, cube (32 units)

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.055 | 0.055 | 0.057 | 0.9% |
| M2M | device | 5 | 0.192 | 0.196 | 0.202 | 3.2% |
| M2L | device | 4 | 3.409 | 3.417 | 3.435 | 55.0% |
| L2L | device | 5 | 0.355 | 0.380 | 0.558 | 6.1% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.147 | 0.172 | 0.186 | 2.8% |
| P2P | device | 1 | 1.984 | 1.995 | 2.021 | 32.1% |
| other | | | | 8.406 | | |
| sum | | | | 6.216 | | 100.0% |

##### cuda, f32, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.991 s, of it tables 0.000 s (loaded from the cache) and device 0.191 s.
- Evaluation over 10 repeats, ms: min 19.648, median 19.973, mean 19.957, max 20.529, std 0.259; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 208.5 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 41, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L (rotation) level 2: 3096 pairs, 64 rows, cube (64 units)
  - M2L (rotation) level 3: 53352 pairs, 512 rows, cube (64 units)
  - M2L (rotation) level 4: 584136 pairs, 4096 rows, cube (64 units)
  - M2L (rotation) level 5: 5398920 pairs, 32768 rows, cube (64 units)

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.070 | 0.072 | 0.072 | 0.6% |
| M2M | device | 5 | 0.242 | 0.245 | 0.250 | 2.1% |
| M2L | device | 4 | 8.536 | 8.570 | 8.608 | 72.5% |
| L2L | device | 5 | 0.541 | 0.557 | 0.588 | 4.7% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.182 | 0.232 | 0.271 | 2.0% |
| P2P | device | 1 | 2.126 | 2.145 | 2.168 | 18.1% |
| other | | | | 8.374 | | |
| sum | | | | 11.822 | | 100.0% |

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.992 s, of it tables 0.001 s (loaded from the cache) and device 0.188 s.
- Evaluation over 10 repeats, ms: min 26.166, median 26.384, mean 26.450, max 26.989, std 0.251; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 227.2 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 41, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2L (rotation) level 2: 3096 pairs, 64 rows, cube (96 units)
  - M2L (rotation) level 3: 53352 pairs, 512 rows, cube (96 units)
  - M2L (rotation) level 4: 584136 pairs, 4096 rows, cube (96 units)
  - M2L (rotation) level 5: 5398920 pairs, 32768 rows, cube (96 units)

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.115 | 0.115 | 0.116 | 0.6% |
| M2M | device | 5 | 0.313 | 0.356 | 0.519 | 2.0% |
| M2L | device | 4 | 14.356 | 14.406 | 14.442 | 79.7% |
| L2L | device | 5 | 0.581 | 0.617 | 0.640 | 3.4% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.414 | 0.443 | 0.479 | 2.5% |
| P2P | device | 1 | 2.122 | 2.142 | 2.150 | 11.8% |
| other | | | | 8.616 | | |
| sum | | | | 18.080 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, CUDA, `--strategy rotation`, f64

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:53:55 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:53:55 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6 --precision f64 --degree 3,6,8,12,18 --strategy rotation --table-cache L/tables --output L/bench/cuda-rotation-f64.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | rotation |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f64 | 1000000 | 3 | Rotation | 1.022 | 21.439 | 21.935 | 21.926 | 22.552 | 0.310 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 6 | Rotation | 0.978 | 31.015 | 32.465 | 32.547 | 34.471 | 0.828 | 8.53e-5 | 2.56e-4 | yes |
| cuda | f64 | 1000000 | 8 | Rotation | 0.978 | 44.198 | 45.156 | 45.294 | 47.303 | 0.788 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 1000000 | 12 | Rotation | 1.032 | 104.074 | 104.895 | 105.099 | 107.501 | 0.902 | 4.35e-7 | 2.17e-6 | yes |
| cuda | f64 | 1000000 | 18 | Rotation | 1.043 | 299.760 | 301.298 | 301.546 | 303.823 | 1.203 | 6.32e-9 | 5.58e-8 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f64 | 1000000 | 3 | 0.086 | 0.208 | 3.739 | 0.577 | 0.000 | 0.000 | 0.155 | 3.137 | 13.143 | 7.902 | 21.684 | 0.360 |
| cuda | f64 | 1000000 | 6 | 0.121 | 0.254 | 13.284 | 0.495 | 0.000 | 0.000 | 0.409 | 3.335 | 13.420 | 17.897 | 32.276 | 0.550 |
| cuda | f64 | 1000000 | 8 | 0.268 | 0.629 | 25.298 | 0.576 | 0.000 | 0.000 | 0.762 | 3.245 | 13.483 | 30.779 | 45.522 | 0.680 |
| cuda | f64 | 1000000 | 12 | 0.908 | 0.901 | 78.797 | 0.999 | 0.000 | 0.000 | 4.797 | 3.189 | 13.328 | 89.591 | 105.073 | 0.852 |
| cuda | f64 | 1000000 | 18 | 5.538 | 2.439 | 258.736 | 2.235 | 0.000 | 0.000 | 12.492 | 3.153 | 13.914 | 284.593 | 302.873 | 0.944 |

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.022 s, of it tables 0.000 s (loaded from the cache) and device 0.240 s.
- Evaluation over 10 repeats, ms: min 21.439, median 21.935, mean 21.926, max 22.552, std 0.310; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 246.7 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 41, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L (rotation) level 2: 3096 pairs, 64 rows, cube (32 units)
  - M2L (rotation) level 3: 53352 pairs, 512 rows, cube (32 units)
  - M2L (rotation) level 4: 584136 pairs, 4096 rows, cube (32 units)
  - M2L (rotation) level 5: 5398920 pairs, 32768 rows, cube (32 units)

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.084 | 0.086 | 0.088 | 1.1% |
| M2M | device | 5 | 0.202 | 0.208 | 0.212 | 2.6% |
| M2L | device | 4 | 3.726 | 3.739 | 3.748 | 47.3% |
| L2L | device | 5 | 0.494 | 0.577 | 0.698 | 7.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.140 | 0.155 | 0.170 | 2.0% |
| P2P | device | 1 | 3.091 | 3.137 | 3.178 | 39.7% |
| other | | | | 13.143 | | |
| sum | | | | 7.902 | | 100.0% |

##### cuda, f64, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.978 s, of it tables 0.001 s (loaded from the cache) and device 0.188 s.
- Evaluation over 10 repeats, ms: min 31.015, median 32.465, mean 32.547, max 34.471, std 0.828; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (64 units); device memory 284.3 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 41, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L (rotation) level 2: 3096 pairs, 64 rows, cube (64 units)
  - M2L (rotation) level 3: 53352 pairs, 512 rows, cube (64 units)
  - M2L (rotation) level 4: 584136 pairs, 4096 rows, cube (64 units)
  - M2L (rotation) level 5: 5398920 pairs, 32768 rows, cube (64 units)

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.120 | 0.121 | 0.122 | 0.7% |
| M2M | device | 5 | 0.249 | 0.254 | 0.260 | 1.4% |
| M2L | device | 4 | 13.250 | 13.284 | 13.339 | 74.2% |
| L2L | device | 5 | 0.476 | 0.495 | 0.518 | 2.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.351 | 0.409 | 0.528 | 2.3% |
| P2P | device | 1 | 3.299 | 3.335 | 3.381 | 18.6% |
| other | | | | 13.420 | | |
| sum | | | | 17.897 | | 100.0% |

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.978 s, of it tables 0.001 s (loaded from the cache) and device 0.199 s.
- Evaluation over 10 repeats, ms: min 44.198, median 45.156, mean 45.294, max 47.303, std 0.788; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 321.2 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 41, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L (rotation) level 2: 3096 pairs, 64 rows, cube (96 units)
  - M2L (rotation) level 3: 53352 pairs, 512 rows, cube (96 units)
  - M2L (rotation) level 4: 584136 pairs, 4096 rows, cube (96 units)
  - M2L (rotation) level 5: 5398920 pairs, 32768 rows, cube (96 units)

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.267 | 0.268 | 0.270 | 0.9% |
| M2M | device | 5 | 0.624 | 0.629 | 0.638 | 2.0% |
| M2L | device | 4 | 25.221 | 25.298 | 25.413 | 82.2% |
| L2L | device | 5 | 0.497 | 0.576 | 0.693 | 1.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.712 | 0.762 | 0.809 | 2.5% |
| P2P | device | 1 | 3.210 | 3.245 | 3.282 | 10.5% |
| other | | | | 13.483 | | |
| sum | | | | 30.779 | | 100.0% |

##### cuda, f64, N = 1000000, p = 12

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.032 s, of it tables 0.003 s (loaded from the cache) and device 0.247 s.
- Evaluation over 10 repeats, ms: min 104.074, median 104.895, mean 105.099, max 107.501, std 0.902; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 4.35e-7, ∇φ 2.17e-6.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(64 x 1 units, 8 columns per unit); rotation cube (192 units); device memory 424.5 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 41, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(64 x 1 units, 8 columns per unit), 1 chunk
  - M2L (rotation) level 2: 3096 pairs, 64 rows, cube (192 units)
  - M2L (rotation) level 3: 53352 pairs, 512 rows, cube (192 units)
  - M2L (rotation) level 4: 584136 pairs, 4096 rows, cube (192 units)
  - M2L (rotation) level 5: 5398920 pairs, 32768 rows, cube (192 units)

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.904 | 0.908 | 0.913 | 1.0% |
| M2M | device | 5 | 0.895 | 0.901 | 0.914 | 1.0% |
| M2L | device | 4 | 78.644 | 78.797 | 78.918 | 88.0% |
| L2L | device | 5 | 0.903 | 0.999 | 1.075 | 1.1% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 4.676 | 4.797 | 4.876 | 5.4% |
| P2P | device | 1 | 3.159 | 3.189 | 3.215 | 3.6% |
| other | | | | 13.328 | | |
| sum | | | | 89.591 | | 100.0% |

##### cuda, f64, N = 1000000, p = 18

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.043 s, of it tables 0.034 s (built) and device 0.223 s.
- Evaluation over 10 repeats, ms: min 299.760, median 301.298, mean 301.546, max 303.823, std 1.203; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 6.32e-9, ∇φ 5.58e-8.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(128 x 1 units, 8 columns per unit); rotation cube (384 units); device memory 658.1 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 41, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(128 x 1 units, 8 columns per unit), 1 chunk
  - M2L (rotation) level 2: 3096 pairs, 64 rows, cube (384 units)
  - M2L (rotation) level 3: 53352 pairs, 512 rows, cube (384 units)
  - M2L (rotation) level 4: 584136 pairs, 4096 rows, cube (384 units)
  - M2L (rotation) level 5: 5398920 pairs, 32768 rows, cube (384 units)

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 5.527 | 5.538 | 5.555 | 1.9% |
| M2M | device | 5 | 2.417 | 2.439 | 2.453 | 0.9% |
| M2L | device | 4 | 258.484 | 258.736 | 259.042 | 90.9% |
| L2L | device | 5 | 2.149 | 2.235 | 2.312 | 0.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 12.399 | 12.492 | 12.644 | 4.4% |
| P2P | device | 1 | 3.113 | 3.153 | 3.312 | 1.1% |
| other | | | | 13.914 | | |
| sum | | | | 284.593 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, host, 72 threads, N = 10⁵ and 10⁶, f32

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:54:43 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:54:43 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | none asked for (host only) |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL not compiled in |
| compiled backends | host |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend host --threads 72 --n 1e5,1e6 --precision f32 --degree 3,6,8 --table-cache L/tables --output L/bench/host72-f32.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72; host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| host | f32 | 100000 | 3 | Dense | 0.070 | 3.979 | 4.058 | 4.089 | 4.422 | 0.124 | 3.39e-3 | 1.99e-3 | yes |
| host | f32 | 100000 | 6 | Dense | 0.073 | 7.152 | 7.284 | 7.272 | 7.432 | 0.085 | 1.42e-4 | 1.30e-4 | yes |
| host | f32 | 100000 | 8 | Dense | 0.079 | 12.610 | 12.866 | 12.990 | 14.006 | 0.394 | 2.14e-5 | 2.37e-5 | yes |
| host | f32 | 1000000 | 3 | Dense | 0.790 | 27.231 | 27.511 | 27.522 | 27.859 | 0.212 | 2.00e-3 | 3.80e-3 | yes |
| host | f32 | 1000000 | 6 | Dense | 0.788 | 58.089 | 58.762 | 58.802 | 59.530 | 0.402 | 8.54e-5 | 2.56e-4 | yes |
| host | f32 | 1000000 | 8 | Dense | 0.784 | 110.104 | 110.590 | 110.536 | 110.997 | 0.275 | 1.40e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f32 | 100000 | 3 | 0.325 | 0.169 | 0.648 | 0.338 | 0.000 | 0.000 | 0.256 | 0.831 | 1.429 | 2.568 | 4.049 | 0.628 |
| host | f32 | 100000 | 6 | 0.397 | 0.178 | 3.646 | 0.341 | 0.000 | 0.000 | 0.467 | 0.846 | 1.459 | 5.875 | 7.395 | 0.808 |
| host | f32 | 100000 | 8 | 0.526 | 0.204 | 8.460 | 0.377 | 0.000 | 0.000 | 0.707 | 0.865 | 1.547 | 11.138 | 12.757 | 0.857 |
| host | f32 | 1000000 | 3 | 0.839 | 0.378 | 4.294 | 0.703 | 0.000 | 0.000 | 1.382 | 9.297 | 9.982 | 16.892 | 27.161 | 0.614 |
| host | f32 | 1000000 | 6 | 2.035 | 0.424 | 30.956 | 0.764 | 0.000 | 0.000 | 3.620 | 9.303 | 11.147 | 47.101 | 58.642 | 0.801 |
| host | f32 | 1000000 | 8 | 3.186 | 0.643 | 77.658 | 0.976 | 0.000 | 0.000 | 5.792 | 9.379 | 12.287 | 97.634 | 110.443 | 0.883 |

##### host, f32, N = 100000, p = 3

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.070 s, of it tables 0.000 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 3.979, median 4.058, mean 4.089, max 4.422, std 0.124; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 3.39e-3, ∇φ 1.99e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.288 | 0.325 | 0.347 | 12.7% |
| M2M | host | 4 | 0.154 | 0.169 | 0.184 | 6.6% |
| M2L | host | 3 | 0.619 | 0.648 | 0.693 | 25.2% |
| L2L | host | 4 | 0.301 | 0.338 | 0.371 | 13.2% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 0.230 | 0.256 | 0.289 | 10.0% |
| P2P | host | 1 | 0.815 | 0.831 | 0.853 | 32.3% |
| other | | | | 1.429 | | |
| sum | | | | 2.568 | | 100.0% |

##### host, f32, N = 100000, p = 6

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.073 s, of it tables 0.004 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 7.152, median 7.284, mean 7.272, max 7.432, std 0.085; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.42e-4, ∇φ 1.30e-4.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.381 | 0.397 | 0.419 | 6.8% |
| M2M | host | 4 | 0.162 | 0.178 | 0.205 | 3.0% |
| M2L | host | 3 | 3.511 | 3.646 | 4.536 | 62.1% |
| L2L | host | 4 | 0.294 | 0.341 | 0.390 | 5.8% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 0.434 | 0.467 | 0.485 | 7.9% |
| P2P | host | 1 | 0.828 | 0.846 | 0.858 | 14.4% |
| other | | | | 1.459 | | |
| sum | | | | 5.875 | | 100.0% |

##### host, f32, N = 100000, p = 8

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.079 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 12.610, median 12.866, mean 12.990, max 14.006, std 0.394; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.14e-5, ∇φ 2.37e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.493 | 0.526 | 0.648 | 4.7% |
| M2M | host | 4 | 0.186 | 0.204 | 0.225 | 1.8% |
| M2L | host | 3 | 8.400 | 8.460 | 8.601 | 76.0% |
| L2L | host | 4 | 0.348 | 0.377 | 0.408 | 3.4% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 0.678 | 0.707 | 0.756 | 6.3% |
| P2P | host | 1 | 0.849 | 0.865 | 0.886 | 7.8% |
| other | | | | 1.547 | | |
| sum | | | | 11.138 | | 100.0% |

##### host, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.790 s, of it tables 0.000 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 27.231, median 27.511, mean 27.522, max 27.859, std 0.212; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.804 | 0.839 | 0.870 | 5.0% |
| M2M | host | 5 | 0.344 | 0.378 | 0.405 | 2.2% |
| M2L | host | 4 | 4.264 | 4.294 | 4.316 | 25.4% |
| L2L | host | 5 | 0.656 | 0.703 | 0.760 | 4.2% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 1.358 | 1.382 | 1.417 | 8.2% |
| P2P | host | 1 | 9.274 | 9.297 | 9.329 | 55.0% |
| other | | | | 9.982 | | |
| sum | | | | 16.892 | | 100.0% |

##### host, f32, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.788 s, of it tables 0.003 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 58.089, median 58.762, mean 58.802, max 59.530, std 0.402; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.54e-5, ∇φ 2.56e-4.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 1.938 | 2.035 | 2.119 | 4.3% |
| M2M | host | 5 | 0.406 | 0.424 | 0.449 | 0.9% |
| M2L | host | 4 | 30.817 | 30.956 | 31.102 | 65.7% |
| L2L | host | 5 | 0.694 | 0.764 | 0.840 | 1.6% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 3.530 | 3.620 | 3.695 | 7.7% |
| P2P | host | 1 | 9.272 | 9.303 | 9.337 | 19.8% |
| other | | | | 11.147 | | |
| sum | | | | 47.101 | | 100.0% |

##### host, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.784 s, of it tables 0.009 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 110.104, median 110.590, mean 110.536, max 110.997, std 0.275; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.40e-5, ∇φ 4.59e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 3.129 | 3.186 | 3.363 | 3.3% |
| M2M | host | 5 | 0.608 | 0.643 | 0.772 | 0.7% |
| M2L | host | 4 | 77.501 | 77.658 | 77.793 | 79.5% |
| L2L | host | 5 | 0.881 | 0.976 | 1.040 | 1.0% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 5.658 | 5.792 | 6.364 | 5.9% |
| P2P | host | 1 | 9.256 | 9.379 | 10.061 | 9.6% |
| other | | | | 12.287 | | |
| sum | | | | 97.634 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f32); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, host, 72 threads, N = 10⁵ and 10⁶, f64

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 17:55:14 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 17:55:14 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | none asked for (host only) |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL not compiled in |
| compiled backends | host |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend host --threads 72 --n 1e5,1e6 --precision f64 --degree 3,6,8,12,18 --table-cache L/tables --output L/bench/host72-f64.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72; host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| host | f64 | 100000 | 3 | Dense | 0.071 | 5.595 | 5.661 | 5.675 | 5.820 | 0.061 | 3.39e-3 | 1.99e-3 | yes |
| host | f64 | 100000 | 6 | Dense | 0.075 | 11.220 | 11.307 | 11.365 | 11.701 | 0.174 | 1.42e-4 | 1.30e-4 | yes |
| host | f64 | 100000 | 8 | Dense | 0.087 | 22.550 | 22.767 | 22.776 | 23.075 | 0.145 | 2.12e-5 | 2.37e-5 | yes |
| host | f64 | 100000 | 12 | Rotation | 0.071 | 37.277 | 37.547 | 37.581 | 38.113 | 0.271 | 8.07e-7 | 1.11e-6 | yes |
| host | f64 | 100000 | 18 | Rotation | 0.077 | 102.524 | 103.143 | 103.483 | 105.227 | 0.903 | 1.01e-8 | 2.37e-8 | yes |
| host | f64 | 1000000 | 3 | Dense | 0.772 | 53.931 | 54.438 | 54.505 | 55.088 | 0.356 | 2.00e-3 | 3.80e-3 | yes |
| host | f64 | 1000000 | 6 | Dense | 0.779 | 108.021 | 108.568 | 108.691 | 109.914 | 0.554 | 8.53e-5 | 2.56e-4 | yes |
| host | f64 | 1000000 | 8 | Dense | 0.807 | 215.625 | 216.369 | 216.403 | 217.143 | 0.418 | 1.36e-5 | 4.59e-5 | yes |
| host | f64 | 1000000 | 12 | Rotation | 0.785 | 354.153 | 354.926 | 355.277 | 358.428 | 1.234 | 4.35e-7 | 2.17e-6 | yes |
| host | f64 | 1000000 | 18 | Rotation | 0.784 | 955.242 | 958.508 | 958.326 | 961.427 | 1.748 | 6.32e-9 | 5.58e-8 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f64 | 100000 | 3 | 0.313 | 0.173 | 0.897 | 0.347 | 0.000 | 0.000 | 0.278 | 2.155 | 1.514 | 4.162 | 5.774 | 0.733 |
| host | f64 | 100000 | 6 | 0.413 | 0.180 | 5.963 | 0.360 | 0.000 | 0.000 | 0.512 | 2.194 | 1.567 | 9.622 | 11.309 | 0.847 |
| host | f64 | 100000 | 8 | 0.578 | 0.360 | 16.947 | 0.415 | 0.000 | 0.000 | 0.735 | 2.204 | 1.655 | 21.240 | 23.046 | 0.933 |
| host | f64 | 100000 | 12 | 0.931 | 0.347 | 30.772 | 0.487 | 0.000 | 0.000 | 1.321 | 2.194 | 1.629 | 36.052 | 37.940 | 0.959 |
| host | f64 | 100000 | 18 | 1.895 | 0.848 | 92.627 | 0.911 | 0.000 | 0.000 | 2.698 | 2.225 | 1.828 | 101.204 | 103.452 | 0.978 |
| host | f64 | 1000000 | 3 | 0.850 | 0.374 | 6.313 | 0.649 | 0.000 | 0.000 | 1.699 | 26.875 | 16.761 | 36.761 | 54.102 | 0.674 |
| host | f64 | 1000000 | 6 | 2.106 | 0.502 | 54.602 | 0.880 | 0.000 | 0.000 | 3.895 | 26.778 | 19.508 | 88.763 | 109.160 | 0.817 |
| host | f64 | 1000000 | 8 | 3.214 | 0.918 | 157.780 | 1.287 | 0.000 | 0.000 | 5.860 | 26.842 | 20.219 | 195.900 | 217.296 | 0.905 |
| host | f64 | 1000000 | 12 | 6.709 | 1.707 | 284.127 | 2.013 | 0.000 | 0.000 | 11.890 | 26.962 | 18.864 | 333.408 | 354.259 | 0.938 |
| host | f64 | 1000000 | 18 | 15.857 | 4.898 | 864.007 | 5.222 | 0.000 | 0.000 | 26.368 | 27.866 | 20.731 | 944.217 | 968.869 | 0.985 |

##### host, f64, N = 100000, p = 3

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.071 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 5.595, median 5.661, mean 5.675, max 5.820, std 0.061; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 3.39e-3, ∇φ 1.99e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.291 | 0.313 | 0.326 | 7.5% |
| M2M | host | 4 | 0.153 | 0.173 | 0.201 | 4.1% |
| M2L | host | 3 | 0.877 | 0.897 | 0.921 | 21.5% |
| L2L | host | 4 | 0.321 | 0.347 | 0.377 | 8.3% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 0.259 | 0.278 | 0.331 | 6.7% |
| P2P | host | 1 | 2.145 | 2.155 | 2.172 | 51.8% |
| other | | | | 1.514 | | |
| sum | | | | 4.162 | | 100.0% |

##### host, f64, N = 100000, p = 6

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.075 s, of it tables 0.007 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 11.220, median 11.307, mean 11.365, max 11.701, std 0.174; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.42e-4, ∇φ 1.30e-4.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.376 | 0.413 | 0.546 | 4.3% |
| M2M | host | 4 | 0.169 | 0.180 | 0.198 | 1.9% |
| M2L | host | 3 | 5.917 | 5.963 | 6.007 | 62.0% |
| L2L | host | 4 | 0.330 | 0.360 | 0.404 | 3.7% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 0.473 | 0.512 | 0.614 | 5.3% |
| P2P | host | 1 | 2.174 | 2.194 | 2.243 | 22.8% |
| other | | | | 1.567 | | |
| sum | | | | 9.622 | | 100.0% |

##### host, f64, N = 100000, p = 8

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.087 s, of it tables 0.019 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 22.550, median 22.767, mean 22.776, max 23.075, std 0.145; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.12e-5, ∇φ 2.37e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.486 | 0.578 | 0.762 | 2.7% |
| M2M | host | 4 | 0.231 | 0.360 | 1.403 | 1.7% |
| M2L | host | 3 | 16.804 | 16.947 | 17.274 | 79.8% |
| L2L | host | 4 | 0.375 | 0.415 | 0.476 | 2.0% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 0.697 | 0.735 | 0.845 | 3.5% |
| P2P | host | 1 | 2.167 | 2.204 | 2.355 | 10.4% |
| other | | | | 1.655 | | |
| sum | | | | 21.240 | | 100.0% |

##### host, f64, N = 100000, p = 12

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.071 s, of it tables 0.003 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 37.277, median 37.547, mean 37.581, max 38.113, std 0.271; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.07e-7, ∇φ 1.11e-6.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.798 | 0.931 | 0.988 | 2.6% |
| M2M | host | 4 | 0.333 | 0.347 | 0.365 | 1.0% |
| M2L | host | 3 | 30.459 | 30.772 | 31.830 | 85.4% |
| L2L | host | 4 | 0.450 | 0.487 | 0.554 | 1.4% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 1.246 | 1.321 | 1.663 | 3.7% |
| P2P | host | 1 | 2.158 | 2.194 | 2.265 | 6.1% |
| other | | | | 1.629 | | |
| sum | | | | 36.052 | | 100.0% |

##### host, f64, N = 100000, p = 18

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.077 s, of it tables 0.008 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 102.524, median 103.143, mean 103.483, max 105.227, std 0.903; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.01e-8, ∇φ 2.37e-8.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 1.755 | 1.895 | 2.806 | 1.9% |
| M2M | host | 4 | 0.808 | 0.848 | 0.890 | 0.8% |
| M2L | host | 3 | 91.836 | 92.627 | 93.578 | 91.5% |
| L2L | host | 4 | 0.784 | 0.911 | 0.971 | 0.9% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 2.643 | 2.698 | 2.792 | 2.7% |
| P2P | host | 1 | 2.153 | 2.225 | 2.317 | 2.2% |
| other | | | | 1.828 | | |
| sum | | | | 101.204 | | 100.0% |

##### host, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.772 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 53.931, median 54.438, mean 54.505, max 55.088, std 0.356; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.788 | 0.850 | 0.936 | 2.3% |
| M2M | host | 5 | 0.341 | 0.374 | 0.457 | 1.0% |
| M2L | host | 4 | 6.228 | 6.313 | 6.454 | 17.2% |
| L2L | host | 5 | 0.595 | 0.649 | 0.698 | 1.8% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 1.461 | 1.699 | 3.552 | 4.6% |
| P2P | host | 1 | 26.746 | 26.875 | 27.587 | 73.1% |
| other | | | | 16.761 | | |
| sum | | | | 36.761 | | 100.0% |

##### host, f64, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.779 s, of it tables 0.007 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 108.021, median 108.568, mean 108.691, max 109.914, std 0.554; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 2.032 | 2.106 | 2.235 | 2.4% |
| M2M | host | 5 | 0.463 | 0.502 | 0.544 | 0.6% |
| M2L | host | 4 | 54.336 | 54.602 | 55.688 | 61.5% |
| L2L | host | 5 | 0.818 | 0.880 | 0.955 | 1.0% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 3.559 | 3.895 | 4.671 | 4.4% |
| P2P | host | 1 | 26.718 | 26.778 | 26.878 | 30.2% |
| other | | | | 19.508 | | |
| sum | | | | 88.763 | | 100.0% |

##### host, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.807 s, of it tables 0.020 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 215.625, median 216.369, mean 216.403, max 217.143, std 0.418; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 3.166 | 3.214 | 3.288 | 1.6% |
| M2M | host | 5 | 0.876 | 0.918 | 0.961 | 0.5% |
| M2L | host | 4 | 157.379 | 157.780 | 159.425 | 80.5% |
| L2L | host | 5 | 1.192 | 1.287 | 1.552 | 0.7% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 5.753 | 5.860 | 5.934 | 3.0% |
| P2P | host | 1 | 26.772 | 26.842 | 26.925 | 13.7% |
| other | | | | 20.219 | | |
| sum | | | | 195.900 | | 100.0% |

##### host, f64, N = 1000000, p = 12

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.785 s, of it tables 0.003 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 354.153, median 354.926, mean 355.277, max 358.428, std 1.234; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 4.35e-7, ∇φ 2.17e-6.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 6.405 | 6.709 | 7.172 | 2.0% |
| M2M | host | 5 | 1.652 | 1.707 | 1.916 | 0.5% |
| M2L | host | 4 | 283.700 | 284.127 | 284.739 | 85.2% |
| L2L | host | 5 | 1.947 | 2.013 | 2.092 | 0.6% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 11.835 | 11.890 | 11.947 | 3.6% |
| P2P | host | 1 | 26.937 | 26.962 | 26.984 | 8.1% |
| other | | | | 18.864 | | |
| sum | | | | 333.408 | | 100.0% |

##### host, f64, N = 1000000, p = 18

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.784 s, of it tables 0.009 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 955.242, median 958.508, mean 958.326, max 961.427, std 1.748; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 6.32e-9, ∇φ 5.58e-8.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 15.253 | 15.857 | 16.422 | 1.7% |
| M2M | host | 5 | 4.763 | 4.898 | 5.034 | 0.5% |
| M2L | host | 4 | 856.186 | 864.007 | 909.744 | 91.5% |
| L2L | host | 5 | 5.013 | 5.222 | 5.527 | 0.6% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 25.653 | 26.368 | 27.511 | 2.8% |
| P2P | host | 1 | 27.069 | 27.866 | 28.485 | 3.0% |
| other | | | | 20.731 | | |
| sum | | | | 944.217 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, host, 72 threads, N = 10⁷, f32

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 19:57:21 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 19:57:21 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | none asked for (host only) |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL not compiled in |
| compiled backends | host |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend host --threads 72 --n 1e7 --precision f32 --degree 3,6,8 --kinds off --table-cache L/tables --output L/bench/host72-1e7-f32.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72; host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `off` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| host | f32 | 10000000 | 3 | Dense | 12.568 | 336.251 | 337.847 | 337.548 | 338.643 | 0.850 | 1.87e-3 | 1.87e-3 | yes |
| host | f32 | 10000000 | 6 | Dense | 11.987 | 624.614 | 628.716 | 628.606 | 630.676 | 1.911 | 7.91e-5 | 1.23e-4 | yes |
| host | f32 | 10000000 | 8 | Dense | 12.336 | 1083.441 | 1087.616 | 1094.512 | 1126.589 | 15.790 | 1.26e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `off` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f32 | 10000000 | 3 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f32 | 10000000 | 6 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f32 | 10000000 | 8 | off | – | – | – | – | – | – | – | – | – | – | – |

##### host, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.568 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 336.251, median 337.847, mean 337.548, max 338.643, std 0.850; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.

##### host, f32, N = 10000000, p = 6

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 11.987 s, of it tables 0.004 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 624.614, median 628.716, mean 628.606, max 630.676, std 1.911; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 7.91e-5, ∇φ 1.23e-4.

##### host, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.336 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 1083.441, median 1087.616, mean 1094.512, max 1126.589, std 15.790; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.26e-5, ∇φ 2.26e-5.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `off`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f32); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, host, 72 threads, N = 10⁷, f64

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 20:00:05 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 20:00:05 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | none asked for (host only) |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL not compiled in |
| compiled backends | host |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend host --threads 72 --n 1e7 --precision f64 --degree 3,6,8,12,18 --kinds off --table-cache L/tables --output L/bench/host72-1e7-f64.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72; host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `off` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| host | f64 | 10000000 | 3 | Dense | 12.195 | 683.889 | 688.842 | 689.468 | 695.187 | 3.950 | 1.87e-3 | 1.87e-3 | yes |
| host | f64 | 10000000 | 6 | Dense | 12.039 | 1168.798 | 1173.325 | 1178.104 | 1215.813 | 14.041 | 7.90e-5 | 1.23e-4 | yes |
| host | f64 | 10000000 | 8 | Dense | 12.128 | 2122.782 | 2131.946 | 2145.216 | 2203.087 | 27.175 | 1.21e-5 | 2.26e-5 | yes |
| host | f64 | 10000000 | 12 | Rotation | 11.909 | 3292.543 | 3307.182 | 3306.811 | 3328.196 | 11.072 | 4.40e-7 | 1.21e-6 | yes |
| host | f64 | 10000000 | 18 | Rotation | 12.024 | 8626.804 | 8694.455 | 8717.771 | 9023.119 | 110.494 | 6.50e-9 | 3.01e-8 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `off` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f64 | 10000000 | 3 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 10000000 | 6 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 10000000 | 8 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 10000000 | 12 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 10000000 | 18 | off | – | – | – | – | – | – | – | – | – | – | – |

##### host, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.195 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 683.889, median 688.842, mean 689.468, max 695.187, std 3.950; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.

##### host, f64, N = 10000000, p = 6

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.039 s, of it tables 0.007 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 1168.798, median 1173.325, mean 1178.104, max 1215.813, std 14.041; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 7.90e-5, ∇φ 1.23e-4.

##### host, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.128 s, of it tables 0.019 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 2122.782, median 2131.946, mean 2145.216, max 2203.087, std 27.175; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.

##### host, f64, N = 10000000, p = 12

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 11.909 s, of it tables 0.003 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 3292.543, median 3307.182, mean 3306.811, max 3328.196, std 11.072; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 4.40e-7, ∇φ 1.21e-6.

##### host, f64, N = 10000000, p = 18

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Rotation; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.024 s, of it tables 0.009 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 8626.804, median 8694.455, mean 8717.771, max 9023.119, std 110.494; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 6.50e-9, ∇φ 3.01e-8.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `off`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, host, 1 thread, f32

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 19:50:31 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 19:50:31 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | none asked for (host only) |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL not compiled in |
| compiled backends | host |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend host --threads 1 --n 1e5,1e6 --precision f32 --degree 3,6,8 --repeats 3 --warmup 0 --kinds off --table-cache L/tables --output L/bench/host1-f32.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 1; host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 0 warm-up, then 3 timed evaluations; kind timings `off` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 3 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| host | f32 | 100000 | 3 | Dense | 0.069 | 92.067 | 92.198 | 92.215 | 92.380 | 0.158 | 3.39e-3 | 1.99e-3 | yes |
| host | f32 | 100000 | 6 | Dense | 0.072 | 266.700 | 267.598 | 267.383 | 267.850 | 0.605 | 1.42e-4 | 1.30e-4 | yes |
| host | f32 | 100000 | 8 | Dense | 0.079 | 539.630 | 539.910 | 540.317 | 541.411 | 0.958 | 2.14e-5 | 2.37e-5 | yes |
| host | f32 | 1000000 | 3 | Dense | 0.788 | 1062.752 | 1062.952 | 1063.422 | 1064.561 | 0.992 | 2.00e-3 | 3.80e-3 | yes |
| host | f32 | 1000000 | 6 | Dense | 0.794 | 2765.692 | 2766.050 | 2766.035 | 2766.364 | 0.336 | 8.54e-5 | 2.56e-4 | yes |
| host | f32 | 1000000 | 8 | Dense | 0.828 | 5278.533 | 5279.194 | 5281.421 | 5286.537 | 4.443 | 1.40e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `off` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f32 | 100000 | 3 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f32 | 100000 | 6 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f32 | 100000 | 8 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f32 | 1000000 | 3 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f32 | 1000000 | 6 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f32 | 1000000 | 8 | off | – | – | – | – | – | – | – | – | – | – | – |

##### host, f32, N = 100000, p = 3

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.069 s, of it tables 0.000 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 92.067, median 92.198, mean 92.215, max 92.380, std 0.158; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 3.39e-3, ∇φ 1.99e-3.

##### host, f32, N = 100000, p = 6

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.072 s, of it tables 0.004 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 266.700, median 267.598, mean 267.383, max 267.850, std 0.605; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.42e-4, ∇φ 1.30e-4.

##### host, f32, N = 100000, p = 8

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.079 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 539.630, median 539.910, mean 540.317, max 541.411, std 0.958; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.14e-5, ∇φ 2.37e-5.

##### host, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.788 s, of it tables 0.000 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 1062.752, median 1062.952, mean 1063.422, max 1064.561, std 0.992; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

##### host, f32, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.794 s, of it tables 0.004 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 2765.692, median 2766.050, mean 2766.035, max 2766.364, std 0.336; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.54e-5, ∇φ 2.56e-4.

##### host, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.828 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 5278.533, median 5279.194, mean 5281.421, max 5286.537, std 4.443; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.40e-5, ∇φ 4.59e-5.

##### Method

Each combination builds the FMM once and runs 0 warm-up evaluations, which compile the device kernels and are not counted, then times 3 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `off`, which runs its own warm-up and 3 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f32); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `nd-fmm-bench`, host, 1 thread, f64

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-07 19:51:25 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 19:51:25 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | none asked for (host only) |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL not compiled in |
| compiled backends | host |
| source revision | d3ed6b4 on tbetcke/phase4s_t8 (.source-revision) |
| command | `nd-fmm-bench --backend host --threads 1 --n 1e5,1e6 --precision f64 --degree 3,6,8,12,18 --repeats 3 --warmup 0 --kinds off --table-cache L/tables --output L/bench/host1-f64.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 1; host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 0 warm-up, then 3 timed evaluations; kind timings `off` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 3 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| host | f64 | 100000 | 3 | Dense | 0.069 | 198.024 | 198.101 | 198.183 | 198.424 | 0.212 | 3.39e-3 | 1.99e-3 | yes |
| host | f64 | 100000 | 6 | Dense | 0.074 | 466.142 | 466.383 | 466.672 | 467.491 | 0.720 | 1.42e-4 | 1.30e-4 | yes |
| host | f64 | 100000 | 8 | Dense | 0.087 | 960.593 | 962.531 | 962.232 | 963.572 | 1.512 | 2.12e-5 | 2.37e-5 | yes |
| host | f64 | 100000 | 12 | Rotation | 0.070 | 2275.534 | 2277.657 | 2277.566 | 2279.507 | 1.988 | 8.07e-7 | 1.11e-6 | yes |
| host | f64 | 100000 | 18 | Rotation | 0.076 | 6359.424 | 6361.623 | 6363.858 | 6370.526 | 5.879 | 1.01e-8 | 2.37e-8 | yes |
| host | f64 | 1000000 | 3 | Dense | 0.778 | 2439.880 | 2441.375 | 2441.072 | 2441.960 | 1.072 | 2.00e-3 | 3.80e-3 | yes |
| host | f64 | 1000000 | 6 | Dense | 0.799 | 5054.895 | 5056.994 | 5058.938 | 5064.926 | 5.290 | 8.53e-5 | 2.56e-4 | yes |
| host | f64 | 1000000 | 8 | Dense | 0.799 | 9648.188 | 9651.400 | 9652.637 | 9658.324 | 5.180 | 1.36e-5 | 4.59e-5 | yes |
| host | f64 | 1000000 | 12 | Rotation | 0.784 | 22129.937 | 22133.751 | 22137.715 | 22149.456 | 10.346 | 4.35e-7 | 2.17e-6 | yes |
| host | f64 | 1000000 | 18 | Rotation | 0.785 | 60763.181 | 60781.751 | 60782.403 | 60802.276 | 19.555 | 6.32e-9 | 5.58e-8 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `off` (min and max in each combination's section). Other: the rest of the stages (loading the charges, transfers, the download, scaling). Sum: the kinds together. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f64 | 100000 | 3 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 100000 | 6 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 100000 | 8 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 100000 | 12 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 100000 | 18 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 1000000 | 3 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 1000000 | 6 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 1000000 | 8 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 1000000 | 12 | off | – | – | – | – | – | – | – | – | – | – | – |
| host | f64 | 1000000 | 18 | off | – | – | – | – | – | – | – | – | – | – | – |

##### host, f64, N = 100000, p = 3

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.069 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 198.024, median 198.101, mean 198.183, max 198.424, std 0.212; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 3.39e-3, ∇φ 1.99e-3.

##### host, f64, N = 100000, p = 6

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.074 s, of it tables 0.007 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 466.142, median 466.383, mean 466.672, max 467.491, std 0.720; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.42e-4, ∇φ 1.30e-4.

##### host, f64, N = 100000, p = 8

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.087 s, of it tables 0.020 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 960.593, median 962.531, mean 962.232, max 963.572, std 1.512; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.12e-5, ∇φ 2.37e-5.

##### host, f64, N = 100000, p = 12

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.070 s, of it tables 0.003 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 2275.534, median 2277.657, mean 2277.566, max 2279.507, std 1.988; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.07e-7, ∇φ 1.11e-6.

##### host, f64, N = 100000, p = 18

- Tree: 4096 leaves on 5 levels; points per leaf 9 / 24.4 / 45 (min / mean / max); lists U 93240, V 640584, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.076 s, of it tables 0.009 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 6359.424, median 6361.623, mean 6363.858, max 6370.526, std 5.879; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.01e-8, ∇φ 2.37e-8.

##### host, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.778 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 2439.880, median 2441.375, mean 2441.072, max 2441.960, std 1.072; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

##### host, f64, N = 1000000, p = 6

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.799 s, of it tables 0.007 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 5054.895, median 5056.994, mean 5058.938, max 5064.926, std 5.290; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 8.53e-5, ∇φ 2.56e-4.

##### host, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.799 s, of it tables 0.020 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 9648.188, median 9651.400, mean 9652.637, max 9658.324, std 5.180; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.

##### host, f64, N = 1000000, p = 12

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.784 s, of it tables 0.003 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 22129.937, median 22133.751, mean 22137.715, max 22149.456, std 10.346; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 4.35e-7, ∇φ 2.17e-6.

##### host, f64, N = 1000000, p = 18

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Rotation; P2P kernel neon; rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.785 s, of it tables 0.009 s (loaded from the cache) and device 0.000 s.
- Evaluation over 3 repeats, ms: min 60763.181, median 60781.751, mean 60782.403, max 60802.276, std 19.555; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 6.32e-9, ∇φ 5.58e-8.

##### Method

Each combination builds the FMM once and runs 0 warm-up evaluations, which compile the device kernels and are not counted, then times 3 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `off`, which runs its own warm-up and 3 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### `device_fmm --device cuda --precision f32`

<details><summary>Output</summary>

#### The device FMM against the host path (Phase 4 T13)

| item | value |
| --- | --- |
| machine | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical; aarch64-linux, release build; rustc 1.99.0 (b940084d7 2026-09-28) |
| device | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| BLAS threads | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1 |
| precision | f32 |
| host | `P2pChoice::Auto` at 1 and 72 threads; one rank |
| timing | evaluate: median wall time of 5 evaluations of the first charge vector after the eight error evaluations (compilation excluded; one sync per device evaluation); stages: on the device the medians of a synchronous-stages build (a sync after each stage), on the host the medians of the timed evaluations |
| tuning | a fresh tuning cache per (problem, p) under `L/cuda-run/tune`, the default budget |
| tables | through the table cache `L/tables` |
| wall time of every run | 1392 s |

##### cube, N = 100000, uniform in the cube [-1, 1)^3 (f32), 64 per leaf

###### p = 3: 4096 leaves on 5 levels, points per leaf 7 / 24.4 / 43 (min / mean / max), lists U 93240 V 640584 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 68.10 | 0.42 L | – | 3.11 | 28.58 | 59.15 | 65% | 91.39 | – | – |
| host, 72 threads, Dense | Dense | 73.32 | 0.47 L | – | 0.75 | 1.32 | 1.33 | 34% | 3.99 | – | – |
| host, 1 thread, Classes | Classes | 66.92 | 0.14 L | – | 3.11 | 89.69 | 59.28 | 39% | 152.66 | – | – |
| host, 72 threads, Classes | Classes | 69.46 | 0.14 L | – | 0.84 | 2.27 | 1.42 | 28% | 5.11 | – | – |
| host, 1 thread, Rotation | Rotation | 67.25 | 0.15 L | – | 3.70 | 111.39 | 59.45 | 34% | 175.11 | – | – |
| host, 72 threads, Rotation | Rotation | 70.14 | 0.15 L | – | 0.78 | 2.46 | 1.33 | 26% | 5.15 | – | – |
| cuda, tuned | Dense | 3910.78 | 0.42 L | 3822.90 | 0.13 | 0.41 | 0.31 | 19% | 1.61 | 56.8 | 2.5 |
| cuda, Dense | Dense | 85.99 | 0.41 L | – | 0.13 | 0.41 | 0.30 | 18% | 1.60 | 57.2 | 2.5 |
| cuda, Classes | Classes | 85.52 | 0.15 L | – | 0.13 | 0.41 | 0.30 | 18% | 1.60 | 95.5 | 3.2 |
| cuda, Rotation | Rotation | 85.08 | 0.09 L | – | 0.13 | 0.82 | 0.47 | 21% | 1.99 | 87.9 | 2.6 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 2.613e-3 | 3.872e-3 | 1.446e-3 | 3.294e-4 | | | | | | yes | |
| host, 1 thread, Classes | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | | | | | | yes | |
| host, 1 thread, Rotation | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | | | | | | yes | |
| cuda, tuned | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | 1.0000, 0.9999, 1.0000, 1.0000 | 1.35e-6, 2.51e-8 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 94.9 MB |
| cuda, Dense | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | 1.0000, 0.9999, 1.0000, 1.0000 | 1.35e-6, 2.51e-8 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 94.9 MB |
| cuda, Classes | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.36e-7, 7.72e-9 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 94.9 MB |
| cuda, Rotation | 2.613e-3 | 3.871e-3 | 1.446e-3 | 3.294e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 2.84e-7, 7.42e-9 | 1 × 400000 / 1 × 1600000 | 34 | 1 | yes | 20.1 MB |

- cuda, tuned: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(16 x 4 units, 8 columns per unit))); tuning 3.82 s: M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned)

- cuda, Dense: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); M2L rotation 3 calls (cube (32 units)))

###### p = 6: 4096 leaves on 5 levels, points per leaf 7 / 24.4 / 43 (min / mean / max), lists U 93240 V 640584 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 70.99 | 3.56 L | – | 11.08 | 181.63 | 74.01 | 28% | 267.43 | – | – |
| host, 72 threads, Dense | Dense | 73.80 | 3.39 L | – | 1.03 | 4.22 | 1.63 | 22% | 7.47 | – | – |
| host, 1 thread, Classes | Classes | 67.90 | 0.61 L | – | 11.22 | 346.00 | 74.78 | 17% | 432.64 | – | – |
| host, 72 threads, Classes | Classes | 71.82 | 0.60 L | – | 0.98 | 6.15 | 1.69 | 18% | 9.39 | – | – |
| host, 1 thread, Rotation | Rotation | 67.85 | 0.33 L | – | 12.60 | 357.20 | 76.11 | 17% | 446.43 | – | – |
| host, 72 threads, Rotation | Rotation | 70.40 | 0.32 L | – | 0.98 | 6.02 | 1.65 | 18% | 9.30 | – | – |
| cuda, tuned | Dense | 3880.29 | 3.50 L | 3788.77 | 0.15 | 0.72 | 0.41 | 18% | 1.91 | 139.7 | 3.9 |
| cuda, Dense | Dense | 90.86 | 3.46 L | – | 0.17 | 0.74 | 0.39 | 17% | 2.01 | 132.7 | 3.7 |
| cuda, Classes | Classes | 89.56 | 0.60 L | – | 0.16 | 0.75 | 0.39 | 17% | 2.01 | 214.9 | 4.7 |
| cuda, Rotation | Rotation | 85.98 | 0.26 L | – | 0.16 | 1.72 | 0.39 | 12% | 3.02 | 147.9 | 3.1 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.150e-4 | 2.757e-4 | 1.006e-4 | 3.202e-5 | | | | | | yes | |
| host, 1 thread, Classes | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | | | | | | yes | |
| host, 1 thread, Rotation | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | | | | | | yes | |
| cuda, tuned | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | 1.0000, 0.9995, 1.0000, 1.0000 | 2.06e-6, 4.13e-8 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 253.2 MB |
| cuda, Dense | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | 1.0000, 0.9995, 1.0000, 1.0000 | 2.06e-6, 4.13e-8 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 253.2 MB |
| cuda, Classes | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | 1.0000, 1.0000, 1.0000, 1.0000 | 4.28e-7, 9.00e-9 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 253.2 MB |
| cuda, Rotation | 1.150e-4 | 2.756e-4 | 1.006e-4 | 3.202e-5 | 1.0000, 0.9999, 1.0000, 1.0000 | 3.07e-7, 7.67e-9 | 1 × 400000 / 1 × 1600000 | 34 | 1 | yes | 22.7 MB |

- cuda, tuned: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(16 x 4 units, 8 columns per unit))); tuning 3.79 s: M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned)

- cuda, Dense: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)); M2L rotation 3 calls (cube (64 units)))

###### p = 8: 4096 leaves on 5 levels, points per leaf 7 / 24.4 / 43 (min / mean / max), lists U 93240 V 640584 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 76.35 | 9.40 L | – | 19.87 | 413.74 | 89.24 | 17% | 523.56 | – | – |
| host, 72 threads, Dense | Dense | 80.54 | 9.38 L | – | 1.11 | 9.19 | 1.90 | 15% | 13.12 | – | – |
| host, 1 thread, Classes | Classes | 68.35 | 1.45 L | – | 19.93 | 761.53 | 89.50 | 10% | 871.50 | – | – |
| host, 72 threads, Classes | Classes | 72.10 | 1.37 L | – | 1.05 | 12.67 | 1.82 | 11% | 16.22 | – | – |
| host, 1 thread, Rotation | Rotation | 68.01 | 0.51 L | – | 21.34 | 655.27 | 89.88 | 12% | 767.25 | – | – |
| host, 72 threads, Rotation | Rotation | 72.91 | 0.52 L | – | 1.10 | 10.47 | 1.84 | 13% | 14.10 | – | – |
| cuda, tuned | Dense | 3756.43 | 9.53 L | 3654.11 | 0.19 | 1.33 | 0.44 | 15% | 2.66 | 196.7 | 4.9 |
| cuda, Dense | Dense | 97.58 | 9.46 L | – | 0.23 | 1.40 | 0.37 | 12% | 2.80 | 187.2 | 4.7 |
| cuda, Classes | Classes | 100.34 | 1.42 L | – | 0.23 | 1.39 | 0.37 | 12% | 2.79 | 312.0 | 5.8 |
| cuda, Rotation | Rotation | 86.85 | 0.51 L | – | 0.21 | 2.40 | 0.36 | 9% | 3.74 | 205.2 | 3.8 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.827e-5 | 4.623e-5 | 1.910e-5 | 8.109e-6 | | | | | | yes | |
| host, 1 thread, Classes | 1.818e-5 | 4.610e-5 | 1.910e-5 | 8.109e-6 | | | | | | yes | |
| host, 1 thread, Rotation | 1.818e-5 | 4.609e-5 | 1.910e-5 | 8.109e-6 | | | | | | yes | |
| cuda, tuned | 1.818e-5 | 4.617e-5 | 1.910e-5 | 8.109e-6 | 0.9951, 0.9987, 1.0001, 1.0001 | 2.24e-6, 4.89e-8 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 409.7 MB |
| cuda, Dense | 1.818e-5 | 4.617e-5 | 1.910e-5 | 8.109e-6 | 0.9951, 0.9987, 1.0001, 1.0001 | 2.24e-6, 4.89e-8 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 409.7 MB |
| cuda, Classes | 1.818e-5 | 4.617e-5 | 1.910e-5 | 8.109e-6 | 0.9997, 1.0016, 1.0000, 1.0000 | 4.68e-7, 9.70e-9 | 1 × 400000 / 1 × 1600000 | 40 | 1 | yes | 409.7 MB |
| cuda, Rotation | 1.817e-5 | 4.611e-5 | 1.910e-5 | 8.109e-6 | 0.9999, 1.0004, 1.0000, 1.0000 | 3.21e-7, 7.76e-9 | 1 × 400000 / 1 × 1600000 | 34 | 1 | yes | 25.5 MB |

- cuda, tuned: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(32 x 2 units, 8 columns per unit))); tuning 3.65 s: M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned)

- cuda, Dense: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); M2L rotation 3 calls (cube (96 units)))

##### plummer, N = 100000, a Plummer sphere about the origin, scale a = 0.1, truncated at 10 a (f32), 64 per leaf

###### p = 3: 5678 leaves on 9 levels, points per leaf 0 / 17.6 / 64 (min / mean / max), lists U 132622 V 813714 W 36642 X 36642

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 90.77 | 0.43 L | – | 3.27 | 83.78 | 219.36 | 71% | 306.96 | – | – |
| host, 72 threads, Dense | Dense | 92.76 | 0.40 L | – | 2.25 | 3.13 | 6.84 | 54% | 12.81 | – | – |
| host, 1 thread, Classes | Classes | 90.06 | 0.15 L | – | 3.39 | 161.62 | 220.03 | 57% | 385.47 | – | – |
| host, 72 threads, Classes | Classes | 94.28 | 0.15 L | – | 2.16 | 4.24 | 6.94 | 50% | 14.02 | – | – |
| host, 1 thread, Rotation | Rotation | 90.10 | 0.16 L | – | 4.06 | 188.55 | 218.99 | 53% | 412.18 | – | – |
| host, 72 threads, Rotation | Rotation | 92.51 | 0.10 L | – | 2.21 | 4.67 | 6.97 | 48% | 14.53 | – | – |
| cuda, tuned | Dense | 5842.01 | 0.41 L | 5721.94 | 0.26 | 0.95 | 1.46 | 41% | 3.12 | 98.5 | 4.1 |
| cuda, Dense | Dense | 117.03 | 0.41 L | – | 0.26 | 0.96 | 1.93 | 48% | 3.57 | 86.1 | 3.6 |
| cuda, Classes | Classes | 116.58 | 0.14 L | – | 0.26 | 0.95 | 1.93 | 48% | 3.70 | 104.2 | 3.8 |
| cuda, Rotation | Rotation | 115.84 | 0.09 L | – | 0.25 | 2.06 | 1.76 | 35% | 4.70 | 87.6 | 3.1 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | | | | | | yes | |
| host, 1 thread, Classes | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | | | | | | yes | |
| host, 1 thread, Rotation | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | | | | | | yes | |
| cuda, tuned | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | 1.0000, 1.0001, 1.0000, 1.0000 | 1.53e-6, 4.13e-8 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 67.6 MB |
| cuda, Dense | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | 1.0000, 1.0001, 1.0000, 1.0000 | 1.53e-6, 4.13e-8 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 67.6 MB |
| cuda, Classes | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.65e-7, 1.28e-8 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 67.6 MB |
| cuda, Rotation | 3.202e-3 | 3.984e-3 | 1.116e-3 | 2.606e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.30e-7, 1.25e-8 | 1 × 400000 / 1 × 1600000 | 87 | 1 | yes | 24.3 MB |

- cuda, tuned: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 8 columns per unit))); tuning 5.72 s: M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 524288 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 131072 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 2048 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 1024 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 2048 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 1024 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (64 units) (tuned)

- cuda, Dense: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); M2L rotation 7 calls (cube (32 units)))

###### p = 6: 5678 leaves on 9 levels, points per leaf 0 / 17.6 / 64 (min / mean / max), lists U 132622 V 813714 W 36642 X 36642

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 94.29 | 3.51 L | – | 11.65 | 369.61 | 480.79 | 56% | 862.64 | – | – |
| host, 72 threads, Dense | Dense | 98.14 | 3.54 L | – | 2.36 | 8.61 | 13.48 | 54% | 25.06 | – | – |
| host, 1 thread, Classes | Classes | 91.05 | 0.60 L | – | 11.75 | 581.37 | 485.76 | 45% | 1080.07 | – | – |
| host, 72 threads, Classes | Classes | 95.45 | 0.61 L | – | 2.26 | 10.78 | 13.76 | 50% | 27.50 | – | – |
| host, 1 thread, Rotation | Rotation | 90.48 | 0.34 L | – | 13.32 | 582.68 | 483.26 | 45% | 1080.15 | – | – |
| host, 72 threads, Rotation | Rotation | 94.83 | 0.28 L | – | 2.32 | 10.65 | 13.27 | 50% | 27.06 | – | – |
| cuda, tuned | Dense | 5539.40 | 3.43 L | 5417.25 | 0.29 | 1.42 | 1.75 | 41% | 3.95 | 218.5 | 6.3 |
| cuda, Dense | Dense | 120.70 | 3.45 L | – | 0.53 | 1.46 | 2.18 | 41% | 4.57 | 188.9 | 5.5 |
| cuda, Classes | Classes | 119.82 | 0.59 L | – | 0.52 | 1.47 | 2.18 | 41% | 4.52 | 238.7 | 6.1 |
| cuda, Rotation | Rotation | 117.86 | 0.28 L | – | 0.52 | 3.34 | 2.23 | 32% | 6.36 | 169.9 | 4.3 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | | | | | | yes | |
| host, 1 thread, Classes | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | | | | | | yes | |
| host, 1 thread, Rotation | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | | | | | | yes | |
| cuda, tuned | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | 1.0000, 1.0000, 1.0000, 1.0000 | 2.46e-6, 6.54e-8 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 160.8 MB |
| cuda, Dense | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | 1.0000, 1.0000, 1.0000, 1.0000 | 2.46e-6, 6.54e-8 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 160.8 MB |
| cuda, Classes | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | 1.0000, 1.0002, 1.0000, 1.0000 | 4.45e-7, 1.44e-8 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 160.8 MB |
| cuda, Rotation | 1.537e-4 | 2.354e-4 | 7.837e-5 | 2.985e-5 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.68e-7, 1.39e-8 | 1 × 400000 / 1 × 1600000 | 87 | 1 | yes | 26.9 MB |

- cuda, tuned: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 8 columns per unit))); tuning 5.42 s: M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 524288 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 262144 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 131072 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 2048 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 1024 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 2048 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 1024 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (64 units) (tuned)

- cuda, Dense: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(16 x 4 units, 8 columns per unit)); M2L rotation 7 calls (cube (64 units)))

###### p = 8: 5678 leaves on 9 levels, points per leaf 0 / 17.6 / 64 (min / mean / max), lists U 132622 V 813714 W 36642 X 36642

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 101.31 | 9.50 L | – | 21.49 | 770.36 | 729.66 | 48% | 1522.09 | – | – |
| host, 72 threads, Dense | Dense | 104.09 | 9.45 L | – | 2.35 | 16.67 | 19.20 | 50% | 38.83 | – | – |
| host, 1 thread, Classes | Classes | 92.10 | 1.40 L | – | 21.33 | 1215.91 | 725.73 | 37% | 1963.76 | – | – |
| host, 72 threads, Classes | Classes | 96.06 | 1.42 L | – | 2.43 | 20.86 | 18.90 | 44% | 42.78 | – | – |
| host, 1 thread, Rotation | Rotation | 91.60 | 0.52 L | – | 23.59 | 1086.01 | 740.97 | 40% | 1852.11 | – | – |
| host, 72 threads, Rotation | Rotation | 95.91 | 0.52 L | – | 2.36 | 18.59 | 19.01 | 47% | 40.68 | – | – |
| cuda, tuned | Dense | 5954.70 | 9.54 L | 5818.13 | 0.53 | 2.24 | 2.01 | 34% | 5.18 | 293.8 | 7.5 |
| cuda, Dense | Dense | 129.15 | 9.51 L | – | 0.53 | 2.45 | 2.50 | 39% | 5.94 | 256.3 | 6.5 |
| cuda, Classes | Classes | 129.65 | 1.40 L | – | 0.53 | 2.45 | 2.51 | 39% | 5.91 | 332.2 | 7.2 |
| cuda, Rotation | Rotation | 118.75 | 0.51 L | – | 0.53 | 4.26 | 2.40 | 29% | 7.68 | 241.2 | 5.3 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 2.196e-5 | 3.885e-5 | 1.395e-5 | 5.182e-6 | | | | | | yes | |
| host, 1 thread, Classes | 2.187e-5 | 3.878e-5 | 1.395e-5 | 5.180e-6 | | | | | | yes | |
| host, 1 thread, Rotation | 2.187e-5 | 3.884e-5 | 1.395e-5 | 5.180e-6 | | | | | | yes | |
| cuda, tuned | 2.189e-5 | 3.884e-5 | 1.395e-5 | 5.180e-6 | 0.9969, 0.9997, 1.0000, 0.9997 | 2.64e-6, 7.72e-8 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 254.4 MB |
| cuda, Dense | 2.189e-5 | 3.884e-5 | 1.395e-5 | 5.180e-6 | 0.9969, 0.9997, 1.0000, 0.9997 | 2.64e-6, 7.72e-8 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 254.4 MB |
| cuda, Classes | 2.189e-5 | 3.884e-5 | 1.395e-5 | 5.180e-6 | 1.0007, 1.0016, 1.0001, 1.0000 | 5.14e-7, 1.59e-8 | 1 × 400000 / 1 × 1600000 | 101 | 1 | yes | 254.4 MB |
| cuda, Rotation | 2.187e-5 | 3.880e-5 | 1.395e-5 | 5.180e-6 | 1.0002, 0.9991, 1.0001, 1.0000 | 3.92e-7, 1.33e-8 | 1 × 400000 / 1 × 1600000 | 87 | 1 | yes | 29.7 MB |

- cuda, tuned: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(32 x 2 units, 8 columns per unit))); tuning 5.82 s: M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 524288 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 262144 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 131072 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 2048 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 1024 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 2048 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 1024 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (64 units) (tuned)

- cuda, Dense: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); M2L rotation 7 calls (cube (96 units)))

##### cube, N = 1000000, uniform in the cube [-1, 1)^3 (f32), 64 per leaf

###### p = 3: 32768 leaves on 6 levels, points per leaf 11 / 30.5 / 57 (min / mean / max), lists U 797816 V 6039504 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 792.87 | 0.45 L | – | 31.04 | 273.96 | 741.38 | 70% | 1055.98 | – | – |
| host, 72 threads, Dense | Dense | 797.34 | 0.44 L | – | 1.82 | 5.62 | 11.13 | 41% | 27.61 | – | – |
| host, 1 thread, Classes | Classes | 788.16 | 0.18 L | – | 30.67 | 850.30 | 730.55 | 45% | 1620.71 | – | – |
| host, 72 threads, Classes | Classes | 797.17 | 0.24 L | – | 1.84 | 13.77 | 11.07 | 31% | 35.99 | – | – |
| host, 1 thread, Rotation | Rotation | 831.23 | 0.12 L | – | 36.08 | 1078.78 | 747.80 | 40% | 1871.28 | – | – |
| host, 72 threads, Rotation | Rotation | 791.69 | 0.12 L | – | 1.82 | 16.39 | 11.11 | 29% | 38.79 | – | – |
| cuda, tuned | Dense | 6127.86 | 0.44 L | 5130.19 | 0.19 | 1.34 | 2.17 | 18% | 13.83 | 76.3 | 2.0 |
| cuda, Dense | Dense | 979.19 | 0.44 L | – | 0.19 | 1.34 | 2.18 | 18% | 13.56 | 77.8 | 2.0 |
| cuda, Classes | Classes | 978.98 | 0.17 L | – | 0.19 | 1.34 | 2.17 | 18% | 13.66 | 118.7 | 2.6 |
| cuda, Rotation | Rotation | 974.80 | 0.12 L | – | 0.19 | 3.46 | 2.22 | 15% | 15.99 | 117.0 | 2.4 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | | | | | | yes | |
| host, 1 thread, Classes | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | | | | | | yes | |
| host, 1 thread, Rotation | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | | | | | | yes | |
| cuda, tuned | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.69e-6, 3.31e-8 | 1 × 4000000 / 1 × 16000000 | 49 | 1 | yes | 878.9 MB |
| cuda, Dense | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.69e-6, 3.31e-8 | 1 × 4000000 / 1 × 16000000 | 49 | 1 | yes | 878.9 MB |
| cuda, Classes | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.63e-7, 9.28e-9 | 1 × 4000000 / 1 × 16000000 | 49 | 1 | yes | 878.9 MB |
| cuda, Rotation | 2.808e-3 | 4.933e-3 | 3.014e-3 | 8.399e-4 | 1.0000, 1.0000, 1.0000, 1.0000 | 3.16e-7, 8.93e-9 | 1 × 4000000 / 1 × 16000000 | 41 | 1 | yes | 189.6 MB |

- cuda, tuned: Dense (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit))); tuning 5.13 s: M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P plane (2 planes per cube) (tuned)

- cuda, Dense: Dense (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); M2L rotation 4 calls (cube (32 units)))

###### p = 6: 32768 leaves on 6 levels, points per leaf 11 / 30.5 / 57 (min / mean / max), lists U 797816 V 6039504 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 809.75 | 3.55 L | – | 110.37 | 1726.35 | 893.24 | 33% | 2739.57 | – | – |
| host, 72 threads, Dense | Dense | 800.96 | 3.63 L | – | 2.96 | 32.55 | 13.55 | 23% | 59.46 | – | – |
| host, 1 thread, Classes | Classes | 789.71 | 0.65 L | – | 109.16 | 3270.97 | 895.18 | 21% | 4285.14 | – | – |
| host, 72 threads, Classes | Classes | 801.50 | 0.64 L | – | 2.89 | 50.10 | 13.45 | 18% | 76.18 | – | – |
| host, 1 thread, Rotation | Rotation | 774.81 | 0.36 L | – | 120.40 | 3355.64 | 896.92 | 20% | 4382.53 | – | – |
| host, 72 threads, Rotation | Rotation | 795.40 | 0.36 L | – | 3.02 | 48.78 | 13.45 | 18% | 74.47 | – | – |
| cuda, tuned | Dense | 6048.02 | 3.60 L | 5063.06 | 0.24 | 4.46 | 2.19 | 14% | 16.83 | 162.8 | 3.5 |
| cuda, Dense | Dense | 971.78 | 3.55 L | – | 0.44 | 4.47 | 2.17 | 14% | 16.96 | 161.5 | 3.5 |
| cuda, Classes | Classes | 981.55 | 0.71 L | – | 0.44 | 4.48 | 2.17 | 14% | 16.91 | 253.5 | 4.5 |
| cuda, Rotation | Rotation | 979.90 | 0.30 L | – | 0.44 | 8.66 | 2.20 | 11% | 21.19 | 206.8 | 3.5 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.212e-4 | 3.101e-4 | 2.099e-4 | 9.303e-5 | | | | | | yes | |
| host, 1 thread, Classes | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | | | | | | yes | |
| host, 1 thread, Rotation | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | | | | | | yes | |
| cuda, tuned | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | 1.0001, 1.0002, 1.0000, 1.0000 | 2.64e-6, 5.44e-8 | 1 × 4000000 / 1 × 16000000 | 49 | 1 | yes | 2317.0 MB |
| cuda, Dense | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | 1.0001, 1.0002, 1.0000, 1.0000 | 2.64e-6, 5.44e-8 | 1 × 4000000 / 1 × 16000000 | 49 | 1 | yes | 2317.0 MB |
| cuda, Classes | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | 0.9999, 1.0000, 1.0000, 1.0000 | 4.68e-7, 1.05e-8 | 1 × 4000000 / 1 × 16000000 | 49 | 1 | yes | 2317.0 MB |
| cuda, Rotation | 1.212e-4 | 3.102e-4 | 2.099e-4 | 9.304e-5 | 1.0000, 1.0002, 1.0000, 1.0000 | 3.39e-7, 9.32e-9 | 1 × 4000000 / 1 × 16000000 | 41 | 1 | yes | 208.5 MB |

- cuda, tuned: Dense (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit))); tuning 5.06 s: M2L strategy: Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P plane (2 planes per cube) (tuned)

- cuda, Dense: Dense (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); M2L 4 calls in 4 chunks (0 library, 4 hand-written cube(16 x 4 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(16 x 4 units, 8 columns per unit)); M2L rotation 4 calls (cube (64 units)))

###### p = 8: 32768 leaves on 6 levels, points per leaf 11 / 30.5 / 57 (min / mean / max), lists U 797816 V 6039504 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 804.39 | 9.53 L | – | 196.12 | 4006.14 | 1048.79 | 20% | 5260.29 | – | – |
| host, 72 threads, Dense | Dense | 801.46 | 9.56 L | – | 4.16 | 79.95 | 15.55 | 14% | 110.68 | – | – |
| host, 1 thread, Classes | Classes | 798.16 | 1.53 L | – | 196.59 | 7200.55 | 1051.25 | 12% | 8458.45 | – | – |
| host, 72 threads, Classes | Classes | 799.32 | 1.50 L | – | 4.17 | 110.66 | 15.39 | 11% | 140.99 | – | – |
| host, 1 thread, Rotation | Rotation | 788.23 | 0.54 L | – | 205.06 | 6174.94 | 1042.86 | 14% | 7432.57 | – | – |
| host, 72 threads, Rotation | Rotation | 793.54 | 0.60 L | – | 4.24 | 89.97 | 15.47 | 13% | 119.92 | – | – |
| cuda, tuned | Dense | 6058.42 | 9.66 L | 5062.10 | 0.46 | 9.53 | 2.36 | 11% | 22.94 | 229.3 | 4.8 |
| cuda, Dense | Dense | 1001.33 | 9.58 L | – | 0.46 | 9.60 | 2.51 | 12% | 22.94 | 229.3 | 4.8 |
| cuda, Classes | Classes | 993.83 | 1.44 L | – | 0.46 | 9.60 | 2.50 | 12% | 22.84 | 370.3 | 6.2 |
| cuda, Rotation | Rotation | 977.89 | 0.55 L | – | 0.45 | 14.61 | 2.48 | 9% | 27.90 | 266.4 | 4.3 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.05), the relative L2 difference of the output from that host's over every target and vector (bound 1e-5), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.863e-5 | 5.282e-5 | 4.026e-5 | 2.305e-5 | | | | | | yes | |
| host, 1 thread, Classes | 1.846e-5 | 5.275e-5 | 4.026e-5 | 2.305e-5 | | | | | | yes | |
| host, 1 thread, Rotation | 1.845e-5 | 5.272e-5 | 4.026e-5 | 2.305e-5 | | | | | | yes | |
| cuda, tuned | 1.846e-5 | 5.277e-5 | 4.026e-5 | 2.305e-5 | 0.9911, 0.9990, 0.9999, 1.0000 | 2.92e-6, 6.45e-8 | 1 × 4000000 / 1 × 16000000 | 52 | 1 | yes | 2365.8 MB |
| cuda, Dense | 1.846e-5 | 5.277e-5 | 4.026e-5 | 2.305e-5 | 0.9911, 0.9990, 0.9999, 1.0000 | 2.92e-6, 6.45e-8 | 1 × 4000000 / 1 × 16000000 | 52 | 1 | yes | 2365.8 MB |
| cuda, Classes | 1.846e-5 | 5.276e-5 | 4.026e-5 | 2.305e-5 | 1.0004, 1.0001, 1.0000, 1.0000 | 5.07e-7, 1.19e-8 | 1 × 4000000 / 1 × 16000000 | 52 | 1 | yes | 2365.8 MB |
| cuda, Rotation | 1.845e-5 | 5.279e-5 | 4.026e-5 | 2.305e-5 | 1.0000, 1.0014, 1.0000, 1.0000 | 3.76e-7, 1.01e-8 | 1 × 4000000 / 1 × 16000000 | 41 | 1 | yes | 227.2 MB |

- cuda, tuned: Dense (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 4 calls in 5 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit))); tuning 5.06 s: M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 8388608 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 32768 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: library (f32 inputs), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: library (f32 inputs), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P plane (2 planes per cube) (tuned)

- cuda, Dense: Dense (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 4 calls in 5 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 4 calls in 5 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 5 calls in 5 chunks (0 library, 5 hand-written cube(32 x 2 units, 8 columns per unit)); M2L rotation 4 calls (cube (96 units)))

##### Summary: evaluation time

Median ms. Host: its default (`Auto`, `P2pChoice::Auto`). Device: tuned, and the fastest of its rows.

| problem | p | host 1 thread | host 72 threads | device tuned | device fastest | tuned x 1 | tuned x 72 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| cube, N = 100000 | 3 | 91.39 | 3.99 | 1.61 | 1.60 (cuda, Dense) | 56.8 | 2.5 |
| cube, N = 100000 | 6 | 267.43 | 7.47 | 1.91 | 1.91 (cuda, tuned) | 139.7 | 3.9 |
| cube, N = 100000 | 8 | 523.56 | 13.12 | 2.66 | 2.66 (cuda, tuned) | 196.7 | 4.9 |
| plummer, N = 100000 | 3 | 306.96 | 12.81 | 3.12 | 3.12 (cuda, tuned) | 98.5 | 4.1 |
| plummer, N = 100000 | 6 | 862.64 | 25.06 | 3.95 | 3.95 (cuda, tuned) | 218.5 | 6.3 |
| plummer, N = 100000 | 8 | 1522.09 | 38.83 | 5.18 | 5.18 (cuda, tuned) | 293.8 | 7.5 |
| cube, N = 1000000 | 3 | 1055.98 | 27.61 | 13.83 | 13.56 (cuda, Dense) | 76.3 | 2.0 |
| cube, N = 1000000 | 6 | 2739.57 | 59.46 | 16.83 | 16.83 (cuda, tuned) | 162.8 | 3.5 |
| cube, N = 1000000 | 8 | 5260.29 | 110.68 | 22.94 | 22.84 (cuda, Classes) | 229.3 | 4.8 |

##### Device leaf-size study (cuda, f32, static rule)

Times in ms: stages from the synchronous-stages build; far = upward + downward; near share = leaves / sum of the stages; evaluate = the median of the default build.

| distribution | p | max points per leaf | leaves | points per leaf (min / mean / max) | V pairs | φ L2 | ∇φ L2 | far | leaves | near share | evaluate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 3 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 2.769e-3 | 2.232e-3 | 1.48 | 0.69 | 21% | 3.08 |
| cube | 3 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 2.616e-3 | 1.489e-3 | 0.67 | 0.44 | 23% | 1.66 |
| cube | 3 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 0.55 | 0.31 | 18% | 1.62 |
| cube | 3 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 0.55 | 0.30 | 18% | 1.63 |
| cube | 3 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 2.295e-3 | 8.343e-4 | 0.29 | 4.57 | 79% | 5.61 |
| cube | 8 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 1.893e-5 | 2.805e-5 | 9.47 | 1.11 | 9% | 11.73 |
| cube | 8 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 1.816e-5 | 1.957e-5 | 2.02 | 0.50 | 14% | 3.04 |
| cube | 8 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 1.818e-5 | 1.910e-5 | 1.62 | 0.37 | 13% | 2.63 |
| cube | 8 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 1.818e-5 | 1.910e-5 | 1.62 | 0.37 | 14% | 2.62 |
| cube | 8 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 1.604e-5 | 1.181e-5 | 0.57 | 4.62 | 76% | 5.82 |
| plummer | 3 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 3.306e-3 | 1.545e-3 | 1.75 | 1.08 | 28% | 3.58 |
| plummer | 3 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 3.260e-3 | 1.377e-3 | 1.37 | 0.97 | 29% | 3.10 |
| plummer | 3 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 3.202e-3 | 1.116e-3 | 1.22 | 1.93 | 48% | 3.52 |
| plummer | 3 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 3.160e-3 | 1.008e-3 | 1.02 | 3.97 | 67% | 5.47 |
| plummer | 3 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 3.072e-3 | 8.782e-4 | 1.14 | 9.15 | 82% | 10.91 |
| plummer | 8 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 2.269e-5 | 2.051e-5 | 7.13 | 1.79 | 18% | 9.82 |
| plummer | 8 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 2.237e-5 | 1.869e-5 | 4.28 | 1.53 | 22% | 6.51 |
| plummer | 8 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 2.189e-5 | 1.395e-5 | 2.98 | 2.49 | 39% | 5.96 |
| plummer | 8 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 2.143e-5 | 1.298e-5 | 2.26 | 4.74 | 61% | 7.33 |
| plummer | 8 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 2.093e-5 | 1.085e-5 | 2.07 | 10.29 | 78% | 12.76 |

###### Fastest leaf size per configuration

| distribution | p | fastest | evaluate (ms) | at 64 (ms) | 64 / fastest | evaluate at 16, 32, 64, 128, 256 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cube | 3 | 64 | 1.62 | 1.62 | 1.00 | 3.08, 1.66, 1.62, 1.63, 5.61 |
| cube | 8 | 128 | 2.62 | 2.63 | 1.00 | 11.73, 3.04, 2.63, 2.62, 5.82 |
| plummer | 3 | 32 | 3.10 | 3.52 | 1.14 | 3.58, 3.10, 3.52, 5.47, 10.91 |
| plummer | 8 | 64 | 5.96 | 5.96 | 1.00 | 9.82, 6.51, 5.96, 7.33, 12.76 |

###### The device leaf-size rule (T13)

Geometric mean of the cuda evaluation time over the 4 configurations (cube and Plummer, p = [3, 8], f32). The fastest size replaces 64 only if it is at least 5% faster by this measure and no φ or ∇φ L2 error at it is more than 10% worse than at 64.

| max points per leaf | geometric mean (ms) | relative to 64 |
| ---: | ---: | ---: |
| 16 | 5.97 | 1.941 |
| 32 | 3.18 | 1.033 |
| 64 | 3.08 | 1.000 |
| 128 | 3.61 | 1.176 |
| 256 | 8.21 | 2.670 |

Fastest: 64 (1.000x faster than 64); largest error growth there: 1.000x. **The rule picks 64.**

Backends run: cuda (f32); host (Neoverse-V2 (implementer 0x41, part 0xd4f)); not run: cpu, metal.

</details>

### `device_fmm --device cuda --precision f64 --large 0`

<details><summary>Output</summary>

#### The device FMM against the host path (Phase 4 T13)

| item | value |
| --- | --- |
| machine | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical; aarch64-linux, release build; rustc 1.99.0 (b940084d7 2026-09-28) |
| device | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| BLAS threads | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1 |
| precision | f64 |
| host | `P2pChoice::Auto` at 1 and 72 threads; one rank |
| timing | evaluate: median wall time of 5 evaluations of the first charge vector after the eight error evaluations (compilation excluded; one sync per device evaluation); stages: on the device the medians of a synchronous-stages build (a sync after each stage), on the host the medians of the timed evaluations |
| tuning | a fresh tuning cache per (problem, p) under `L/cuda-run/tune`, the default budget |
| tables | through the table cache `L/tables` |
| wall time of every run | 3997 s |

##### cube, N = 100000, uniform in the cube [-1, 1)^3 (f64), 64 per leaf

###### p = 8: 4096 leaves on 5 levels, points per leaf 7 / 24.4 / 43 (min / mean / max), lists U 93240 V 640584 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 87.93 | 19.61 L | – | 22.50 | 735.41 | 183.30 | 19% | 941.94 | – | – |
| host, 72 threads, Dense | Dense | 93.08 | 19.34 L | – | 1.15 | 17.61 | 3.30 | 15% | 22.85 | – | – |
| host, 1 thread, Classes | Classes | 68.76 | 2.67 L | – | 22.59 | 1184.40 | 184.97 | 13% | 1393.38 | – | – |
| host, 72 threads, Classes | Classes | 72.56 | 2.70 L | – | 1.18 | 19.88 | 3.16 | 13% | 24.98 | – | – |
| host, 1 thread, Rotation | Rotation | 67.43 | 0.97 L | – | 21.86 | 709.57 | 184.90 | 20% | 917.17 | – | – |
| host, 72 threads, Rotation | Rotation | 71.16 | 0.95 L | – | 1.13 | 11.41 | 3.11 | 19% | 16.40 | – | – |
| cuda, tuned | Dense | 3893.98 | 18.92 L | 3781.63 | 0.22 | 1.98 | 0.56 | 14% | 3.73 | 252.9 | 6.1 |
| cuda, Dense | Dense | 111.10 | 19.59 L | – | 0.43 | 2.02 | 0.51 | 12% | 3.86 | 244.3 | 5.9 |
| cuda, Classes | Classes | 107.56 | 2.78 L | – | 0.42 | 2.02 | 0.52 | 13% | 3.89 | 358.6 | 6.4 |
| cuda, Rotation | Rotation | 88.49 | 1.02 L | – | 0.43 | 3.92 | 0.61 | 10% | 5.82 | 157.6 | 2.8 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.001), the relative L2 difference of the output from that host's over every target and vector (bound 1e-12), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.816e-5 | 4.619e-5 | 1.903e-5 | 8.082e-6 | | | | | | yes | |
| host, 1 thread, Classes | 1.816e-5 | 4.619e-5 | 1.903e-5 | 8.082e-6 | | | | | | yes | |
| host, 1 thread, Rotation | 1.816e-5 | 4.619e-5 | 1.903e-5 | 8.082e-6 | | | | | | yes | |
| cuda, tuned | 1.816e-5 | 4.619e-5 | 1.903e-5 | 8.082e-6 | 1.0000, 1.0000, 1.0000, 1.0000 | 5.23e-15, 9.97e-17 | 1 × 800000 / 1 × 3200000 | 40 | 1 | yes | 804.8 MB |
| cuda, Dense | 1.816e-5 | 4.619e-5 | 1.903e-5 | 8.082e-6 | 1.0000, 1.0000, 1.0000, 1.0000 | 5.23e-15, 9.97e-17 | 1 × 800000 / 1 × 3200000 | 40 | 1 | yes | 804.8 MB |
| cuda, Classes | 1.816e-5 | 4.619e-5 | 1.903e-5 | 8.082e-6 | 1.0000, 1.0000, 1.0000, 1.0000 | 9.33e-16, 1.77e-17 | 1 × 800000 / 1 × 3200000 | 40 | 1 | yes | 804.8 MB |
| cuda, Rotation | 1.816e-5 | 4.619e-5 | 1.903e-5 | 8.082e-6 | 1.0000, 1.0000, 1.0000, 1.0000 | 6.19e-16, 1.68e-17 | 1 × 800000 / 1 × 3200000 | 34 | 1 | yes | 36.8 MB |

- cuda, tuned: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(32 x 2 units, 8 columns per unit))); tuning 3.78 s: M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P plane (2 planes per cube) (tuned)

- cuda, Dense: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(32 x 2 units, 8 columns per unit)); M2L rotation 3 calls (cube (96 units)))

###### p = 12: 4096 leaves on 5 levels, points per leaf 7 / 24.4 / 43 (min / mean / max), lists U 93240 V 640584 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 153.10 | 84.51 L | – | 63.40 | 4657.07 | 229.55 | 5% | 4951.22 | – | – |
| host, 72 threads, Dense | Dense | 156.74 | 84.97 L | – | 2.06 | 77.88 | 3.76 | 4% | 84.82 | – | – |
| host, 1 thread, Classes | Classes | 78.16 | 10.36 L | – | 60.64 | 4352.75 | 225.96 | 5% | 4640.54 | – | – |
| host, 72 threads, Classes | Classes | 81.68 | 10.62 L | – | 2.01 | 81.97 | 3.70 | 4% | 88.62 | – | – |
| host, 1 thread, Rotation | Rotation | 69.66 | 2.74 L | – | 50.09 | 1956.30 | 226.46 | 10% | 2233.58 | – | – |
| host, 72 threads, Rotation | Rotation | 73.72 | 2.82 L | – | 1.54 | 31.55 | 3.72 | 10% | 37.68 | – | – |
| cuda, tuned | Dense | 4211.42 | 84.22 L | 4015.23 | 0.45 | 6.06 | 0.97 | 11% | 8.68 | 257.3 | 4.3 |
| cuda, Dense | Dense | 195.55 | 84.38 L | – | 0.45 | 6.06 | 0.95 | 11% | 8.68 | 570.3 | 9.8 |
| cuda, Classes | Classes | 200.79 | 10.37 L | – | 0.45 | 6.06 | 1.15 | 13% | 8.68 | 534.9 | 10.2 |
| cuda, Rotation | Rotation | 95.10 | 2.82 L | – | 0.46 | 10.44 | 1.14 | 9% | 12.93 | 172.8 | 2.9 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.001), the relative L2 difference of the output from that host's over every target and vector (bound 1e-12), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 7.119e-7 | 2.536e-6 | 1.038e-6 | 6.011e-7 | | | | | | yes | |
| host, 1 thread, Classes | 7.119e-7 | 2.536e-6 | 1.038e-6 | 6.011e-7 | | | | | | yes | |
| host, 1 thread, Rotation | 7.119e-7 | 2.536e-6 | 1.038e-6 | 6.011e-7 | | | | | | yes | |
| cuda, tuned | 7.119e-7 | 2.536e-6 | 1.038e-6 | 6.011e-7 | 1.0000, 1.0000, 1.0000, 1.0000 | 7.52e-15, 1.39e-16 | 1 × 800000 / 1 × 3200000 | 40 | 1 | yes | 1692.7 MB |
| cuda, Dense | 7.119e-7 | 2.536e-6 | 1.038e-6 | 6.011e-7 | 1.0000, 1.0000, 1.0000, 1.0000 | 7.52e-15, 1.39e-16 | 1 × 800000 / 1 × 3200000 | 40 | 1 | yes | 1692.7 MB |
| cuda, Classes | 7.119e-7 | 2.536e-6 | 1.038e-6 | 6.011e-7 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.14e-15, 2.05e-17 | 1 × 800000 / 1 × 3200000 | 40 | 1 | yes | 1692.7 MB |
| cuda, Rotation | 7.119e-7 | 2.536e-6 | 1.038e-6 | 6.011e-7 | 1.0000, 1.0000, 1.0000, 1.0000 | 7.09e-16, 1.58e-17 | 1 × 800000 / 1 × 3200000 | 34 | 1 | yes | 53.5 MB |

- cuda, tuned: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(64 x 1 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(64 x 1 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(64 x 1 units, 8 columns per unit))); tuning 4.02 s: M2L strategy: Dense, hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned)

- cuda, Dense: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(64 x 1 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(64 x 1 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(64 x 1 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(64 x 1 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(64 x 1 units, 8 columns per unit)); M2L 3 calls in 3 chunks (0 library, 3 hand-written cube(64 x 1 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(64 x 1 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(64 x 1 units, 8 columns per unit)); M2L rotation 3 calls (cube (192 units)))

###### p = 18: 4096 leaves on 5 levels, points per leaf 7 / 24.4 / 43 (min / mean / max), lists U 93240 V 640584 W 0 X 0

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 455.34 | 386.14 L | – | 211.82 | 30714.69 | 324.89 | 1% | 31253.15 | – | – |
| host, 72 threads, Dense | Dense | 456.38 | 383.62 L | – | 5.83 | 718.90 | 5.14 | 1% | 731.76 | – | – |
| host, 1 thread, Classes | Classes | 112.84 | 43.96 L | – | 204.18 | 17572.65 | 324.61 | 2% | 18102.62 | – | – |
| host, 72 threads, Classes | Classes | 115.54 | 43.99 L | – | 5.39 | 372.00 | 5.07 | 1% | 383.80 | – | – |
| host, 1 thread, Rotation | Rotation | 76.16 | 8.47 L | – | 138.02 | 5907.45 | 330.39 | 5% | 6377.27 | – | – |
| host, 72 threads, Rotation | Rotation | 80.64 | 8.52 L | – | 2.94 | 93.21 | 5.07 | 5% | 102.21 | – | – |
| cuda, tuned | Dense | 5071.68 | 384.67 L | 4505.29 | 1.29 | 22.20 | 2.03 | 8% | 26.60 | 239.7 | 3.8 |
| cuda, Dense | Dense | 567.12 | 384.11 L | – | 1.29 | 22.21 | 2.01 | 8% | 26.64 | 1173.2 | 27.5 |
| cuda, Classes | Classes | 775.52 | 43.88 L | – | 1.29 | 22.16 | 1.89 | 7% | 26.65 | 679.3 | 14.4 |
| cuda, Rotation | Rotation | 123.16 | 8.62 L | – | 1.30 | 30.26 | 1.98 | 6% | 34.70 | 183.8 | 2.9 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.001), the relative L2 difference of the output from that host's over every target and vector (bound 1e-12), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.022e-8 | 3.866e-8 | 2.401e-8 | 1.774e-8 | | | | | | yes | |
| host, 1 thread, Classes | 1.022e-8 | 3.866e-8 | 2.401e-8 | 1.774e-8 | | | | | | yes | |
| host, 1 thread, Rotation | 1.022e-8 | 3.866e-8 | 2.401e-8 | 1.774e-8 | | | | | | yes | |
| cuda, tuned | 1.022e-8 | 3.866e-8 | 2.401e-8 | 1.774e-8 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.03e-14, 2.02e-16 | 1 × 800000 / 1 × 3200000 | 43 | 1 | yes | 2545.4 MB |
| cuda, Dense | 1.022e-8 | 3.866e-8 | 2.401e-8 | 1.774e-8 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.03e-14, 2.02e-16 | 1 × 800000 / 1 × 3200000 | 43 | 1 | yes | 2545.4 MB |
| cuda, Classes | 1.022e-8 | 3.866e-8 | 2.401e-8 | 1.774e-8 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.48e-15, 2.50e-17 | 1 × 800000 / 1 × 3200000 | 43 | 1 | yes | 2545.4 MB |
| cuda, Rotation | 1.022e-8 | 3.866e-8 | 2.401e-8 | 1.774e-8 | 1.0000, 1.0000, 1.0000, 1.0000 | 9.20e-16, 1.86e-17 | 1 × 800000 / 1 × 3200000 | 34 | 1 | yes | 98.3 MB |

- cuda, tuned: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(128 x 1 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(128 x 1 units, 8 columns per unit)); M2L 3 calls in 4 chunks (0 library, 3 hand-written cube(128 x 1 units, 8 columns per unit))); tuning 4.51 s: M2L strategy: Dense, hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 1048576 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (32 units) (tuned)

- cuda, Dense: Dense (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(128 x 1 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(128 x 1 units, 8 columns per unit)); M2L 3 calls in 4 chunks (0 library, 3 hand-written cube(128 x 1 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(128 x 1 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(128 x 1 units, 8 columns per unit)); M2L 3 calls in 4 chunks (0 library, 3 hand-written cube(128 x 1 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 4 calls in 4 chunks (0 library, 4 hand-written cube(128 x 1 units, 8 columns per unit)); L2L 4 calls in 4 chunks (0 library, 4 hand-written cube(128 x 1 units, 8 columns per unit)); M2L rotation 3 calls (cube (384 units)))

##### plummer, N = 100000, a Plummer sphere about the origin, scale a = 0.1, truncated at 10 a (f64), 64 per leaf

###### p = 8: 5678 leaves on 9 levels, points per leaf 0 / 17.6 / 64 (min / mean / max), lists U 132622 V 813714 W 36642 X 36642

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 110.53 | 19.29 L | – | 25.09 | 1171.12 | 840.99 | 41% | 2038.89 | – | – |
| host, 72 threads, Dense | Dense | 112.61 | 18.97 L | – | 2.41 | 27.62 | 20.31 | 40% | 51.18 | – | – |
| host, 1 thread, Classes | Classes | 91.36 | 2.73 L | – | 24.68 | 1784.64 | 830.09 | 31% | 2640.03 | – | – |
| host, 72 threads, Classes | Classes | 96.38 | 2.73 L | – | 2.37 | 30.59 | 20.77 | 38% | 54.34 | – | – |
| host, 1 thread, Rotation | Rotation | 89.98 | 1.04 L | – | 23.35 | 1110.37 | 829.83 | 42% | 1964.83 | – | – |
| host, 72 threads, Rotation | Rotation | 94.46 | 0.96 L | – | 2.47 | 19.07 | 20.93 | 49% | 43.34 | – | – |
| cuda, tuned | Dense | 5913.50 | 18.84 L | 5769.41 | 0.54 | 3.22 | 2.93 | 37% | 7.42 | 274.7 | 6.9 |
| cuda, Dense | Dense | 141.40 | 18.98 L | – | 0.53 | 3.33 | 3.36 | 40% | 8.15 | 250.1 | 6.3 |
| cuda, Classes | Classes | 138.09 | 2.71 L | – | 0.53 | 3.33 | 3.34 | 40% | 8.06 | 327.6 | 6.7 |
| cuda, Rotation | Rotation | 118.56 | 0.96 L | – | 0.54 | 7.09 | 3.38 | 28% | 11.88 | 165.4 | 3.6 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.001), the relative L2 difference of the output from that host's over every target and vector (bound 1e-12), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 2.187e-5 | 3.882e-5 | 1.387e-5 | 5.180e-6 | | | | | | yes | |
| host, 1 thread, Classes | 2.187e-5 | 3.882e-5 | 1.387e-5 | 5.180e-6 | | | | | | yes | |
| host, 1 thread, Rotation | 2.187e-5 | 3.882e-5 | 1.387e-5 | 5.180e-6 | | | | | | yes | |
| cuda, tuned | 2.187e-5 | 3.882e-5 | 1.387e-5 | 5.180e-6 | 1.0000, 1.0000, 1.0000, 1.0000 | 5.71e-15, 1.59e-16 | 1 × 800000 / 1 × 3200000 | 101 | 1 | yes | 489.7 MB |
| cuda, Dense | 2.187e-5 | 3.882e-5 | 1.387e-5 | 5.180e-6 | 1.0000, 1.0000, 1.0000, 1.0000 | 5.71e-15, 1.59e-16 | 1 × 800000 / 1 × 3200000 | 101 | 1 | yes | 489.7 MB |
| cuda, Classes | 2.187e-5 | 3.882e-5 | 1.387e-5 | 5.180e-6 | 1.0000, 1.0000, 1.0000, 1.0000 | 9.72e-16, 2.90e-17 | 1 × 800000 / 1 × 3200000 | 101 | 1 | yes | 489.7 MB |
| cuda, Rotation | 2.187e-5 | 3.882e-5 | 1.387e-5 | 5.180e-6 | 1.0000, 1.0000, 1.0000, 1.0000 | 7.58e-16, 2.44e-17 | 1 × 800000 / 1 × 3200000 | 87 | 1 | yes | 41.1 MB |

- cuda, tuned: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(32 x 2 units, 8 columns per unit))); tuning 5.77 s: M2L strategy: Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 524288 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 262144 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 131072 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 2048 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 1024 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 2048 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 1024 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (64 units) (tuned)

- cuda, Dense: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(32 x 2 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(32 x 2 units, 8 columns per unit)); M2L rotation 7 calls (cube (96 units)))

###### p = 12: 5678 leaves on 9 levels, points per leaf 0 / 17.6 / 64 (min / mean / max), lists U 132622 V 813714 W 36642 X 36642

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 176.05 | 84.46 L | – | 70.91 | 6202.45 | 1503.97 | 19% | 7778.47 | – | – |
| host, 72 threads, Dense | Dense | 180.32 | 84.58 L | – | 3.28 | 109.33 | 35.49 | 24% | 149.45 | – | – |
| host, 1 thread, Classes | Classes | 101.00 | 10.39 L | – | 69.61 | 6171.59 | 1481.25 | 19% | 7723.36 | – | – |
| host, 72 threads, Classes | Classes | 104.94 | 10.53 L | – | 3.25 | 115.84 | 36.04 | 23% | 156.15 | – | – |
| host, 1 thread, Rotation | Rotation | 92.63 | 2.78 L | – | 54.63 | 2984.59 | 1480.76 | 33% | 4521.47 | – | – |
| host, 72 threads, Rotation | Rotation | 97.33 | 2.86 L | – | 2.74 | 49.39 | 35.77 | 40% | 89.15 | – | – |
| cuda, tuned | Dense | 6174.47 | 84.96 L | 5946.16 | 0.97 | 8.93 | 4.84 | 30% | 15.63 | 289.2 | 5.7 |
| cuda, Dense | Dense | 226.64 | 84.99 L | – | 0.95 | 8.97 | 5.44 | 33% | 16.23 | 479.2 | 9.2 |
| cuda, Classes | Classes | 230.42 | 10.40 L | – | 0.95 | 8.96 | 5.31 | 32% | 16.26 | 475.1 | 9.6 |
| cuda, Rotation | Rotation | 125.60 | 2.84 L | – | 0.96 | 16.28 | 5.38 | 22% | 23.50 | 192.4 | 3.8 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.001), the relative L2 difference of the output from that host's over every target and vector (bound 1e-12), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 8.058e-7 | 1.572e-6 | 7.064e-7 | 3.446e-7 | | | | | | yes | |
| host, 1 thread, Classes | 8.058e-7 | 1.572e-6 | 7.064e-7 | 3.446e-7 | | | | | | yes | |
| host, 1 thread, Rotation | 8.058e-7 | 1.572e-6 | 7.064e-7 | 3.446e-7 | | | | | | yes | |
| cuda, tuned | 8.058e-7 | 1.572e-6 | 7.064e-7 | 3.446e-7 | 1.0000, 1.0000, 1.0000, 1.0000 | 8.56e-15, 2.28e-16 | 1 × 800000 / 1 × 3200000 | 101 | 1 | yes | 1030.9 MB |
| cuda, Dense | 8.058e-7 | 1.572e-6 | 7.064e-7 | 3.446e-7 | 1.0000, 1.0000, 1.0000, 1.0000 | 8.56e-15, 2.28e-16 | 1 × 800000 / 1 × 3200000 | 101 | 1 | yes | 1030.9 MB |
| cuda, Classes | 8.058e-7 | 1.572e-6 | 7.064e-7 | 3.446e-7 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.18e-15, 3.53e-17 | 1 × 800000 / 1 × 3200000 | 101 | 1 | yes | 1030.9 MB |
| cuda, Rotation | 8.058e-7 | 1.572e-6 | 7.064e-7 | 3.446e-7 | 1.0000, 1.0000, 1.0000, 1.0000 | 8.27e-16, 3.14e-17 | 1 × 800000 / 1 × 3200000 | 87 | 1 | yes | 58.0 MB |

- cuda, tuned: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(64 x 1 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(64 x 1 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(64 x 1 units, 8 columns per unit))); tuning 5.95 s: M2L strategy: Dense, hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 524288 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 262144 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 131072 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 2048 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 1024 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 2048 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 1024 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (64 units) (tuned)

- cuda, Dense: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(64 x 1 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(64 x 1 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(64 x 1 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(64 x 1 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(64 x 1 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(64 x 1 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(64 x 1 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(64 x 1 units, 8 columns per unit)); M2L rotation 7 calls (cube (192 units)))

###### p = 18: 5678 leaves on 9 levels, points per leaf 0 / 17.6 / 64 (min / mean / max), lists U 132622 V 813714 W 36642 X 36642

Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and the tuning. x 1 / x 72: the evaluation's speed-up over the host with the same strategy at 1 / 72 threads (the tuned row: over the host default, `Auto`).

| run | strategy | build | tables | tuning | upward | downward | leaves | near share | evaluate | x 1 | x 72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host, 1 thread, Dense | Dense | 477.59 | 385.04 L | – | 254.20 | 41191.36 | 2989.85 | 7% | 44439.41 | – | – |
| host, 72 threads, Dense | Dense | 482.44 | 386.12 L | – | 7.94 | 1083.88 | 71.22 | 6% | 1162.79 | – | – |
| host, 1 thread, Classes | Classes | 135.47 | 43.75 L | – | 247.35 | 22868.82 | 3000.69 | 11% | 26118.51 | – | – |
| host, 72 threads, Classes | Classes | 139.63 | 44.41 L | – | 7.68 | 502.35 | 68.34 | 12% | 580.23 | – | – |
| host, 1 thread, Rotation | Rotation | 99.51 | 8.49 L | – | 153.09 | 8718.82 | 2956.03 | 25% | 11829.12 | – | – |
| host, 72 threads, Rotation | Rotation | 104.31 | 8.53 L | – | 4.35 | 141.81 | 70.28 | 32% | 217.98 | – | – |
| cuda, tuned | Dense | 7103.80 | 378.79 L | 6513.69 | 2.62 | 30.96 | 9.67 | 22% | 44.49 | 265.9 | 4.9 |
| cuda, Dense | Dense | 594.90 | 383.58 L | – | 2.41 | 30.89 | 10.41 | 23% | 44.83 | 991.2 | 25.9 |
| cuda, Classes | Classes | 805.75 | 43.86 L | – | 2.42 | 30.87 | 10.30 | 23% | 45.00 | 580.4 | 12.9 |
| cuda, Rotation | Rotation | 153.93 | 8.57 L | – | 2.41 | 43.50 | 10.19 | 18% | 57.03 | 207.4 | 3.8 |

Errors: root mean squares over the charge vectors of the relative L2 and max errors at the sampled targets. Device rows: the four errors over the host's of the same strategy at one thread (bound 1 ± 0.001), the relative L2 difference of the output from that host's over every target and vector (bound 1e-12), and an evaluation's transfers, launches and syncs (every kind on the device).

| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | up / down (bytes) | launches | syncs | bit-identical | device memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| host, 1 thread, Dense | 1.379e-8 | 5.676e-8 | 1.624e-8 | 1.146e-8 | | | | | | yes | |
| host, 1 thread, Classes | 1.379e-8 | 5.676e-8 | 1.624e-8 | 1.146e-8 | | | | | | yes | |
| host, 1 thread, Rotation | 1.379e-8 | 5.676e-8 | 1.624e-8 | 1.146e-8 | | | | | | yes | |
| cuda, tuned | 1.379e-8 | 5.676e-8 | 1.624e-8 | 1.146e-8 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.19e-14, 3.23e-16 | 1 × 800000 / 1 × 3200000 | 101 | 1 | yes | 2353.2 MB |
| cuda, Dense | 1.379e-8 | 5.676e-8 | 1.624e-8 | 1.146e-8 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.19e-14, 3.23e-16 | 1 × 800000 / 1 × 3200000 | 101 | 1 | yes | 2353.2 MB |
| cuda, Classes | 1.379e-8 | 5.676e-8 | 1.624e-8 | 1.146e-8 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.57e-15, 4.03e-17 | 1 × 800000 / 1 × 3200000 | 101 | 1 | yes | 2353.2 MB |
| cuda, Rotation | 1.379e-8 | 5.676e-8 | 1.624e-8 | 1.146e-8 | 1.0000, 1.0000, 1.0000, 1.0000 | 1.07e-15, 3.15e-17 | 1 × 800000 / 1 × 3200000 | 87 | 1 | yes | 103.2 MB |

- cuda, tuned: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(128 x 1 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(128 x 1 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(128 x 1 units, 8 columns per unit))); tuning 6.51 s: M2L strategy: Dense, hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 524288 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 262144 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 131072 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 65536 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 2048 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 1024 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); M2M GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); M2M GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 4096 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 2048 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 1024 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 512 pairs: hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB (tuned); L2L GEMM, ≤ 64 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); L2L GEMM, ≤ 8 pairs: hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB (static rule); P2P layout, ≤ 32 points per leaf: P2P cube (64 units) (tuned)

- cuda, Dense: Dense (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(128 x 1 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(128 x 1 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(128 x 1 units, 8 columns per unit)))

- cuda, Classes: Classes, run as dense on the device (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(128 x 1 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(128 x 1 units, 8 columns per unit)); M2L 7 calls in 7 chunks (0 library, 7 hand-written cube(128 x 1 units, 8 columns per unit)))

- cuda, Rotation: Rotation (M2M 8 calls in 8 chunks (0 library, 8 hand-written cube(128 x 1 units, 8 columns per unit)); L2L 8 calls in 8 chunks (0 library, 8 hand-written cube(128 x 1 units, 8 columns per unit)); M2L rotation 7 calls (cube (384 units)))

##### Summary: evaluation time

Median ms. Host: its default (`Auto`, `P2pChoice::Auto`). Device: tuned, and the fastest of its rows.

| problem | p | host 1 thread | host 72 threads | device tuned | device fastest | tuned x 1 | tuned x 72 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| cube, N = 100000 | 8 | 941.94 | 22.85 | 3.73 | 3.73 (cuda, tuned) | 252.9 | 6.1 |
| cube, N = 100000 | 12 | 2233.58 | 37.68 | 8.68 | 8.68 (cuda, Classes) | 257.3 | 4.3 |
| cube, N = 100000 | 18 | 6377.27 | 102.21 | 26.60 | 26.60 (cuda, tuned) | 239.7 | 3.8 |
| plummer, N = 100000 | 8 | 2038.89 | 51.18 | 7.42 | 7.42 (cuda, tuned) | 274.7 | 6.9 |
| plummer, N = 100000 | 12 | 4521.47 | 89.15 | 15.63 | 15.63 (cuda, tuned) | 289.2 | 5.7 |
| plummer, N = 100000 | 18 | 11829.12 | 217.98 | 44.49 | 44.49 (cuda, tuned) | 265.9 | 4.9 |

##### Device leaf-size study (cuda, f64, static rule)

Times in ms: stages from the synchronous-stages build; far = upward + downward; near share = leaves / sum of the stages; evaluate = the median of the default build.

| distribution | p | max points per leaf | leaves | points per leaf (min / mean / max) | V pairs | φ L2 | ∇φ L2 | far | leaves | near share | evaluate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 3 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 2.769e-3 | 2.232e-3 | 1.88 | 0.86 | 19% | 3.94 |
| cube | 3 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 2.616e-3 | 1.489e-3 | 0.77 | 0.48 | 19% | 2.25 |
| cube | 3 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 0.62 | 0.58 | 23% | 2.00 |
| cube | 3 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 0.62 | 0.57 | 23% | 2.01 |
| cube | 3 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 2.295e-3 | 8.342e-4 | 0.32 | 5.85 | 79% | 7.21 |
| cube | 8 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 1.893e-5 | 2.804e-5 | 15.02 | 1.42 | 8% | 18.15 |
| cube | 8 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 1.816e-5 | 1.950e-5 | 2.70 | 0.64 | 14% | 4.26 |
| cube | 8 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 1.816e-5 | 1.903e-5 | 2.47 | 0.51 | 12% | 3.83 |
| cube | 8 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 1.816e-5 | 1.903e-5 | 2.45 | 0.52 | 13% | 3.85 |
| cube | 8 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 1.596e-5 | 1.087e-5 | 0.88 | 6.06 | 74% | 7.62 |
| plummer | 3 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 3.306e-3 | 1.545e-3 | 2.10 | 1.38 | 28% | 4.34 |
| plummer | 3 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 3.260e-3 | 1.377e-3 | 1.64 | 1.35 | 32% | 3.87 |
| plummer | 3 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 3.202e-3 | 1.116e-3 | 1.45 | 2.40 | 48% | 4.51 |
| plummer | 3 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 3.160e-3 | 1.008e-3 | 1.21 | 5.13 | 68% | 7.06 |
| plummer | 3 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 3.072e-3 | 8.782e-4 | 1.29 | 11.80 | 82% | 13.95 |
| plummer | 8 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 2.269e-5 | 2.047e-5 | 10.74 | 2.48 | 17% | 14.46 |
| plummer | 8 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 2.236e-5 | 1.864e-5 | 6.04 | 2.08 | 22% | 9.31 |
| plummer | 8 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 2.187e-5 | 1.387e-5 | 3.86 | 3.34 | 40% | 8.12 |
| plummer | 8 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 2.142e-5 | 1.290e-5 | 2.85 | 6.28 | 60% | 9.87 |
| plummer | 8 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 2.092e-5 | 1.072e-5 | 2.54 | 13.47 | 78% | 16.79 |

###### Fastest leaf size per configuration

| distribution | p | fastest | evaluate (ms) | at 64 (ms) | 64 / fastest | evaluate at 16, 32, 64, 128, 256 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cube | 3 | 64 | 2.00 | 2.00 | 1.00 | 3.94, 2.25, 2.00, 2.01, 7.21 |
| cube | 8 | 64 | 3.83 | 3.83 | 1.00 | 18.15, 4.26, 3.83, 3.85, 7.62 |
| plummer | 3 | 32 | 3.87 | 4.51 | 1.17 | 4.34, 3.87, 4.51, 7.06, 13.95 |
| plummer | 8 | 64 | 8.12 | 8.12 | 1.00 | 14.46, 9.31, 8.12, 9.87, 16.79 |

###### The device leaf-size rule (T13)

Geometric mean of the cuda evaluation time over the 4 configurations (cube and Plummer, p = [3, 8], f64). The fastest size replaces 64 only if it is at least 5% faster by this measure and no φ or ∇φ L2 error at it is more than 10% worse than at 64.

| max points per leaf | geometric mean (ms) | relative to 64 |
| ---: | ---: | ---: |
| 16 | 8.18 | 1.999 |
| 32 | 4.31 | 1.053 |
| 64 | 4.10 | 1.000 |
| 128 | 4.82 | 1.177 |
| 256 | 10.65 | 2.601 |

Fastest: 64 (1.000x faster than 64); largest error growth there: 1.000x. **The rule picks 64.**

Backends run: cuda (f64); host (Neoverse-V2 (implementer 0x41, part 0xd4f)); not run: cpu, metal.

</details>

### `autotune --device cuda --precision f32`

<details><summary>Output</summary>

#### Device autotune tables (Phase 4 T12, C4.7)

| item | value |
| --- | --- |
| machine | Neoverse-V2 (implementer 0x41, part 0xd4f) (72 physical, 72 logical) |
| target | aarch64-linux, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| device | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| precision | f32 |
| timing | each candidate: a warm-up launch and a sync (compilation excluded), then the median of 5 batches of >= 10 ms of launches queued between syncs |
| budget | 10 s per build; a fresh tuning cache per (problem, p) |

Static rule (no tuning cache): f32 `Dense` at every p.

##### cube, p = 3

Tuning 3.80 s of a 10 s budget; build 3.90 s; cache `L/cuda-run/autotune/cube-cuda-f32-p03`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 194.5 | **chosen** (tuned) |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 200.9 | 1.03× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 239.0 | 1.23× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 199.0 | 1.02× |
| M2L strategy | 4 | 584136 | Rotation | 371.3 | 1.91× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 194.5 | **chosen** (tuned) |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 200.9 | 1.03× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 239.0 | 1.23× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 199.0 | 1.02× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 71.5 | 1.00× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 71.2 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 75.3 | 1.06× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 71.5 | 1.00× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 26.9 | 1.06× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 27.1 | 1.07× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 25.3 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 25.7 | 1.01× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 15.1 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 16.2 | 1.07× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 16.1 | 1.07× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 16.2 | 1.07× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 16.0 | 1.02× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 16.0 | 1.02× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 15.8 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 16.0 | 1.02× |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 13.8 | 1.01× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 13.6 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 13.7 | 1.00× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 13.8 | 1.01× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 13.9 | 1.02× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 13.7 | 1.00× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 13.7 | 1.00× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 13.7 | **chosen** (tuned) |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (32 units) | 257.0 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (64 units) | 313.8 | 1.22× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (2 planes per cube) | 260.5 | 1.01× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (4 planes per cube) | 266.6 | 1.04× |

##### cube, p = 6

Tuning 3.62 s of a 10 s budget; build 3.72 s; cache `L/cuda-run/autotune/cube-cuda-f32-p06`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 469.1 | **chosen** (tuned) |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 576.1 | 1.23× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 696.8 | 1.49× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 537.8 | 1.15× |
| M2L strategy | 4 | 584136 | Rotation | 1039.8 | 2.22× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 469.1 | **chosen** (tuned) |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 576.1 | 1.23× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 696.8 | 1.49× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 537.8 | 1.15× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 96.6 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 106.2 | 1.10× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 109.2 | 1.13× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 105.7 | 1.09× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 34.7 | 1.27× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 34.4 | 1.26× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 27.4 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 40.9 | 1.49× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 22.8 | 1.45× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 21.8 | 1.38× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.7 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 27.5 | 1.75× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 22.1 | 1.40× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 20.6 | 1.30× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.8 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 26.4 | 1.66× |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 21.1 | 1.50× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 20.1 | 1.43× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 14.1 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 25.7 | 1.83× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 19.8 | 1.45× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 18.4 | 1.34× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 13.7 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 24.5 | 1.79× |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (32 units) | 254.5 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (64 units) | 308.0 | 1.21× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (2 planes per cube) | 260.8 | 1.02× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (4 planes per cube) | 257.5 | 1.01× |

##### cube, p = 8

Tuning 3.71 s of a 10 s budget; build 3.81 s; cache `L/cuda-run/autotune/cube-cuda-f32-p08`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 976.1 | **chosen** (tuned) |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 1099.0 | 1.13× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 1225.0 | 1.25× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 1061.4 | 1.09× |
| M2L strategy | 4 | 584136 | Dense, library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 3584, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2L strategy | 4 | 584136 | Rotation | 1635.9 | 1.68× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 976.1 | **chosen** (tuned) |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 1099.0 | 1.13× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 1225.0 | 1.25× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 1061.4 | 1.09× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 3584, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 160.1 | 1.02× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 175.9 | 1.12× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 172.6 | 1.10× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 157.0 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 384, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 45.5 | 1.47× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 40.8 | 1.31× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 31.1 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 35.0 | 1.13× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 32, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 32.6 | 1.61× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 28.3 | 1.40× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 20.2 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 22.4 | 1.11× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 31.3 | 1.88× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 26.8 | 1.61× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 16.6 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 20.7 | 1.24× |
| M2M GEMM, ≤ 64 pairs | – | 0 | library (f32 inputs), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | library (f32 inputs), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 31.4 | 1.69× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 27.0 | 1.45× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 18.6 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 21.1 | 1.13× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 29.1 | 2.02× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 24.6 | 1.71× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 14.4 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 18.2 | 1.26× |
| L2L GEMM, ≤ 64 pairs | – | 0 | library (f32 inputs), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | library (f32 inputs), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (32 units) | 257.5 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (64 units) | 308.5 | 1.20× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (2 planes per cube) | 260.6 | 1.01× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (4 planes per cube) | 267.1 | 1.04× |

##### plummer, p = 3

Tuning 5.83 s of a 10 s budget; build 5.95 s; cache `L/cuda-run/autotune/plummer-cuda-f32-p03`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 124.9 | **chosen** (tuned) |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 129.6 | 1.04× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 157.3 | 1.26× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 128.8 | 1.03× |
| M2L strategy | 7 | 335952 | Rotation | 292.6 | 2.34× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 124.9 | **chosen** (tuned) |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 129.6 | 1.04× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 157.3 | 1.26× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 128.8 | 1.03× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 98.9 | **chosen** (tuned) |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 100.3 | 1.01× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 113.4 | 1.15× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 100.3 | 1.01× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 79.9 | **chosen** (tuned) |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 80.3 | 1.00× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 87.6 | 1.10× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 80.3 | 1.00× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 71.4 | 1.00× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 71.3 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 75.2 | 1.05× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 72.0 | 1.01× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 27.5 | 1.08× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 26.7 | 1.05× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 25.5 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 26.4 | 1.04× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 14.0 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 15.0 | 1.07× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 15.1 | 1.08× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 14.9 | 1.06× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 14.9 | 1.00× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 14.8 | **chosen** (tuned) |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 14.8 | 1.00× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 14.9 | 1.00× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 15.0 | 1.01× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 14.9 | **chosen** (tuned) |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 15.1 | 1.02× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 15.0 | 1.01× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 14.9 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 15.1 | 1.01× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 15.0 | 1.01× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 15.0 | 1.00× |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 14.2 | 1.02× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 14.0 | 1.00× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 13.9 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 14.0 | 1.01× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 14.2 | 1.03× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 13.8 | 1.00× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 13.8 | **chosen** (tuned) |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 13.9 | 1.01× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 14.1 | 1.02× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 13.8 | 1.00× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 13.8 | **chosen** (tuned) |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 13.9 | 1.01× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 14.0 | 1.02× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 4 units, 4 columns per unit), box-major, 2048 MB | 13.8 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 2 units, 2 columns per unit), box-major, 2048 MB | 14.0 | 1.02× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 13.9 | 1.01× |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (32 units) | 315.1 | 1.32× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (64 units) | 238.3 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (2 planes per cube) | 314.8 | 1.32× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (4 planes per cube) | 321.2 | 1.35× |

##### plummer, p = 6

Tuning 5.49 s of a 10 s budget; build 5.61 s; cache `L/cuda-run/autotune/plummer-cuda-f32-p06`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 330.2 | **chosen** (tuned) |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 387.4 | 1.17× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 462.4 | 1.40× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 368.4 | 1.12× |
| M2L strategy | 7 | 335952 | Rotation | 639.5 | 1.94× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 330.2 | **chosen** (tuned) |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 387.4 | 1.17× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 462.4 | 1.40× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 368.4 | 1.12× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 245.4 | **chosen** (tuned) |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 286.0 | 1.17× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 337.1 | 1.37× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 280.6 | 1.14× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 148.8 | **chosen** (tuned) |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 172.1 | 1.16× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 201.1 | 1.35× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 169.5 | 1.14× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 96.6 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 106.3 | 1.10× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 109.0 | 1.13× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 104.7 | 1.08× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 35.4 | 1.30× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 34.6 | 1.27× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 27.2 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 41.5 | 1.53× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 22.3 | 1.45× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 21.2 | 1.38× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.4 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 26.9 | 1.75× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 22.4 | 1.48× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 21.1 | 1.40× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.1 | **chosen** (tuned) |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 26.7 | 1.77× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 22.4 | 1.41× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 20.5 | 1.29× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.9 | **chosen** (tuned) |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 26.9 | 1.69× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 21.8 | 1.35× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 20.3 | 1.26× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 16.2 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 26.5 | 1.64× |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 20.6 | 1.44× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 19.1 | 1.34× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 14.3 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 25.1 | 1.75× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 20.3 | 1.43× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 18.9 | 1.33× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 14.2 | **chosen** (tuned) |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 24.9 | 1.76× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 20.7 | 1.43× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 18.7 | 1.29× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 14.5 | **chosen** (tuned) |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 24.9 | 1.72× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 19.8 | 1.38× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 18.0 | 1.26× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 14.3 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 24.5 | 1.71× |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (32 units) | 314.9 | 1.32× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (64 units) | 239.0 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (2 planes per cube) | 314.9 | 1.32× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (4 planes per cube) | 314.8 | 1.32× |

##### plummer, p = 8

Tuning 5.83 s of a 10 s budget; build 5.97 s; cache `L/cuda-run/autotune/plummer-cuda-f32-p08`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 582.7 | **chosen** (tuned) |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 672.4 | 1.15× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 735.0 | 1.26× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 634.6 | 1.09× |
| M2L strategy | 7 | 335952 | Dense, library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 1976, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2L strategy | 7 | 335952 | Rotation | 1041.8 | 1.79× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 582.7 | **chosen** (tuned) |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 672.4 | 1.15× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 735.0 | 1.26× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 634.6 | 1.09× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 1976, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 425.6 | **chosen** (tuned) |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 475.7 | 1.12× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 513.7 | 1.21× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 450.5 | 1.06× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 1280, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 289.4 | 1.00× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 302.4 | 1.04× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 331.0 | 1.14× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 289.4 | **chosen** (tuned) |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 704, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 160.4 | 1.02× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 176.3 | 1.12× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 176.3 | 1.12× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 157.8 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 384, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 45.1 | 1.47× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 40.8 | 1.33× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 30.8 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 34.9 | 1.13× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [316, 32, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 302, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 32.2 | 1.72× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 27.9 | 1.49× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 18.7 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 22.2 | 1.18× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 203, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 31.5 | 1.81× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 27.4 | 1.58× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 17.4 | **chosen** (tuned) |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 21.0 | 1.21× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 117, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 31.8 | 1.91× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 27.1 | 1.63× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 16.7 | **chosen** (tuned) |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 21.0 | 1.26× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 30.9 | 1.87× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 26.6 | 1.61× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 16.5 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 20.4 | 1.24× |
| M2M GEMM, ≤ 64 pairs | – | 0 | library (f32 inputs), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | library (f32 inputs), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 302, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 30.2 | 1.79× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 25.7 | 1.52× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 16.9 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 20.1 | 1.19× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 203, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 29.6 | 1.91× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 25.5 | 1.64× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.5 | **chosen** (tuned) |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 18.9 | 1.22× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 117, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 30.0 | 2.04× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 25.2 | 1.72× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 14.6 | **chosen** (tuned) |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 18.8 | 1.28× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | library (f32 inputs), box-major, 2048 MB | – | not registered: the library does not take the level's shapes: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 29.0 | 2.01× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 24.6 | 1.71× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 14.4 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 18.3 | 1.27× |
| L2L GEMM, ≤ 64 pairs | – | 0 | library (f32 inputs), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | library (f32 inputs), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (32 units) | 314.8 | 1.30× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (64 units) | 242.3 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (2 planes per cube) | 314.9 | 1.30× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (4 planes per cube) | 314.9 | 1.30× |

##### Summary

| problem | p | strategy (static rule) | strategy (tuned) | tuned decisions | cut | tuning s | build s | build from the cache s | evaluate static ms | evaluate tuned ms |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 3 | Dense | Dense | 9 | 0 | 3.80 | 3.90 | 0.09 | 1.62 | 1.47 |
| cube | 6 | Dense | Dense | 9 | 0 | 3.62 | 3.72 | 0.09 | 1.94 | 1.91 |
| cube | 8 | Dense | Dense | 9 | 0 | 3.71 | 3.81 | 0.10 | 2.71 | 2.55 |
| plummer | 3 | Dense | Dense | 15 | 0 | 5.83 | 5.95 | 0.12 | 3.59 | 3.19 |
| plummer | 6 | Dense | Dense | 15 | 0 | 5.49 | 5.61 | 0.12 | 4.70 | 4.06 |
| plummer | 8 | Dense | Dense | 15 | 0 | 5.83 | 5.97 | 0.13 | 6.01 | 5.27 |

Backends run: cuda (f32); not run: cpu, metal.

</details>

### `autotune --device cuda --precision f64`

<details><summary>Output</summary>

#### Device autotune tables (Phase 4 T12, C4.7)

| item | value |
| --- | --- |
| machine | Neoverse-V2 (implementer 0x41, part 0xd4f) (72 physical, 72 logical) |
| target | aarch64-linux, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| device | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| precision | f64 |
| timing | each candidate: a warm-up launch and a sync (compilation excluded), then the median of 5 batches of >= 10 ms of launches queued between syncs |
| budget | 10 s per build; a fresh tuning cache per (problem, p) |

Static rule (no tuning cache): f64 `Dense` at every p (Phase 4S decision 9).

##### cube, p = 4

Tuning 3.85 s of a 10 s budget; build 3.95 s; cache `L/cuda-run/autotune/cube-cuda-f64-p04`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 353.5 | **chosen** (tuned) |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 390.6 | 1.11× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 467.5 | 1.32× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 386.2 | 1.09× |
| M2L strategy | 4 | 584136 | Rotation | 566.4 | 1.60× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 353.5 | **chosen** (tuned) |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 390.6 | 1.11× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 467.5 | 1.32× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 386.2 | 1.09× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 81.4 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 86.7 | 1.07× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 88.4 | 1.09× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 88.3 | 1.08× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 33.0 | 1.24× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 31.0 | 1.16× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 26.7 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 33.8 | 1.27× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 19.3 | 1.22× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 16.1 | 1.02× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.8 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 20.3 | 1.28× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 18.1 | 1.19× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 15.2 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.7 | 1.03× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 18.9 | 1.24× |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 17.4 | 1.27× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 14.2 | 1.03× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 13.7 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 18.2 | 1.33× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 16.3 | 1.20× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 13.7 | 1.01× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 13.6 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 17.0 | 1.25× |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (32 units) | 390.0 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (64 units) | 484.3 | 1.24× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (2 planes per cube) | 393.4 | 1.01× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (4 planes per cube) | 394.9 | 1.01× |

##### cube, p = 8

Tuning 3.63 s of a 10 s budget; build 3.75 s; cache `L/cuda-run/autotune/cube-cuda-f64-p08`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 1505.3 | **chosen** (tuned) |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 1552.3 | 1.03× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 1810.2 | 1.20× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 1555.2 | 1.03× |
| M2L strategy | 4 | 584136 | Rotation | 2701.5 | 1.79× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 1505.3 | **chosen** (tuned) |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 1552.3 | 1.03× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 1810.2 | 1.20× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 1555.2 | 1.03× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 244.2 | 1.03× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 238.0 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 275.5 | 1.16× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 238.2 | 1.00× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 47.5 | 1.33× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 43.8 | 1.22× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 35.8 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 45.0 | 1.26× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 34.0 | 1.43× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 31.0 | 1.30× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 23.8 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 29.8 | 1.26× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 32.5 | 1.53× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 28.7 | 1.35× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 21.2 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 27.4 | 1.29× |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 33.1 | 1.47× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 29.9 | 1.33× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 22.5 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 28.7 | 1.28× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 31.0 | 1.66× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 26.5 | 1.42× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 18.7 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 25.6 | 1.37× |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (32 units) | 392.5 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (64 units) | 484.5 | 1.23× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (2 planes per cube) | 392.8 | 1.00× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (4 planes per cube) | 395.7 | 1.01× |

##### cube, p = 12

Tuning 3.95 s of a 10 s budget; build 4.14 s; cache `L/cuda-run/autotune/cube-cuda-f64-p12`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 4989.0 | **chosen** (tuned) |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 9995.7 | 2.00× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 5908.0 | 1.18× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 4991.1 | 1.00× |
| M2L strategy | 4 | 584136 | Rotation | 8298.3 | 1.66× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 4989.0 | **chosen** (tuned) |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 9995.7 | 2.00× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 5908.0 | 1.18× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 4991.1 | 1.00× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 633.7 | 1.03× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 1199.3 | 1.96× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 761.9 | 1.24× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 612.8 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 117.9 | 1.04× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 310.9 | 2.75× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 116.4 | 1.03× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 113.2 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 61.7 | 1.15× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 141.3 | 2.63× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 56.2 | 1.05× |
| M2M GEMM, ≤ 4096 pairs | 3 | 4096 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 53.8 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 56.7 | 1.18× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 135.0 | 2.80× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 49.8 | 1.03× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 48.2 | **chosen** (tuned) |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 62.0 | 1.14× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 141.0 | 2.60× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 56.5 | 1.04× |
| L2L GEMM, ≤ 4096 pairs | 4 | 4096 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 54.2 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 54.2 | 1.17× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 133.1 | 2.88× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 47.2 | 1.02× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 46.3 | **chosen** (tuned) |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (32 units) | 387.4 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P cube (64 units) | 484.7 | 1.25× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (2 planes per cube) | 393.6 | 1.02× |
| P2P layout, ≤ 32 points per leaf | 4 | 58082534 | P2P plane (4 planes per cube) | 394.7 | 1.02× |

##### cube, p = 16

Tuning 10.00 s of a 10 s budget, deadline reached; build 10.62 s; cache `L/cuda-run/autotune/cube-cuda-f64-p16`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 15237.1 | 1.27× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 11983.1 | **chosen** (tuned) |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 32540.9 | 2.72× |
| M2L strategy | 4 | 584136 | Dense, hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 16212.2 | 1.35× |
| M2L strategy | 4 | 584136 | Rotation | 18345.1 | 1.53× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 15237.1 | 1.27× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 11983.1 | **chosen** (tuned) |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 32540.9 | 2.72× |
| M2L GEMM, ≤ 1048576 pairs | 4 | 584136 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 16212.2 | 1.35× |
| M2L GEMM, ≤ 65536 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; past the tuning deadline: not tuned, not stored) |
| M2L GEMM, ≤ 4096 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; past the tuning deadline: not tuned, not stored) |
| M2M GEMM, ≤ 4096 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; past the tuning deadline: not tuned, not stored) |
| M2M GEMM, ≤ 512 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; past the tuning deadline: not tuned, not stored) |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; past the tuning deadline: not tuned, not stored) |
| L2L GEMM, ≤ 512 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; past the tuning deadline: not tuned, not stored) |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | – | 0 | P2P cube (32 units) | – | chosen (static rule; past the tuning deadline: not tuned, not stored) |

##### plummer, p = 4

Tuning 6.30 s of a 10 s budget; build 6.42 s; cache `L/cuda-run/autotune/plummer-cuda-f64-p04`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 252.0 | **chosen** (tuned) |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 274.2 | 1.09× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 324.8 | 1.29× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 270.9 | 1.08× |
| M2L strategy | 7 | 335952 | Rotation | 416.2 | 1.65× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 252.0 | **chosen** (tuned) |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 274.2 | 1.09× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 324.8 | 1.29× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 270.9 | 1.08× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 201.2 | **chosen** (tuned) |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 219.7 | 1.09× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 251.0 | 1.25× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 216.7 | 1.08× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 123.0 | **chosen** (tuned) |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 130.9 | 1.06× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 151.4 | 1.23× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 128.6 | 1.05× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 81.9 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 85.4 | 1.04× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 88.9 | 1.09× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 87.0 | 1.06× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 32.8 | 1.23× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 30.5 | 1.15× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 26.6 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 33.7 | 1.27× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 18.8 | 1.18× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 16.4 | 1.03× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.9 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 19.2 | 1.21× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 18.9 | 1.19× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 16.2 | 1.02× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 15.9 | **chosen** (tuned) |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 19.8 | 1.24× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 18.5 | 1.19× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 15.5 | **chosen** (tuned) |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 16.0 | 1.03× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 19.1 | 1.23× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 18.2 | 1.15× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 15.9 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 16.1 | 1.01× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 19.0 | 1.20× |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 16.7 | 1.23× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 14.3 | 1.05× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 13.6 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 17.2 | 1.26× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 16.8 | 1.23× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 14.2 | 1.04× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 13.6 | **chosen** (tuned) |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 17.4 | 1.28× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 16.4 | 1.21× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 13.6 | **chosen** (tuned) |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 13.6 | 1.01× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 17.1 | 1.26× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | 16.4 | 1.21× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 13.7 | 1.01× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 13.5 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(16 x 8 units, 4 columns per unit), box-major, 2048 MB | 17.2 | 1.27× |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(16 x 4 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (32 units) | 421.7 | 1.16× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (64 units) | 362.9 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (2 planes per cube) | 421.7 | 1.16× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (4 planes per cube) | 421.7 | 1.16× |

##### plummer, p = 8

Tuning 5.76 s of a 10 s budget; build 5.91 s; cache `L/cuda-run/autotune/plummer-cuda-f64-p08`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 886.3 | **chosen** (tuned) |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 924.0 | 1.04× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 1086.5 | 1.23× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 937.0 | 1.06× |
| M2L strategy | 7 | 335952 | Rotation | 1819.6 | 2.05× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 886.3 | **chosen** (tuned) |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 924.0 | 1.04× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 1086.5 | 1.23× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 937.0 | 1.06× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 634.3 | **chosen** (tuned) |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 646.8 | 1.02× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 747.7 | 1.18× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 659.0 | 1.04× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 399.6 | **chosen** (tuned) |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 412.7 | 1.03× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 463.1 | 1.16× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 419.0 | 1.05× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 244.7 | 1.04× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 235.5 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 276.2 | 1.17× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 238.6 | 1.01× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 48.2 | 1.35× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 43.5 | 1.22× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 35.6 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 44.2 | 1.24× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 33.5 | 1.48× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 29.7 | 1.32× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 22.6 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 30.3 | 1.34× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 33.3 | 1.53× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 29.4 | 1.35× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 21.8 | **chosen** (tuned) |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 29.6 | 1.36× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 32.7 | 1.53× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 28.9 | 1.35× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 21.4 | **chosen** (tuned) |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 28.9 | 1.35× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 32.6 | 1.55× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 28.3 | 1.35× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 21.0 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 27.7 | 1.32× |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 32.0 | 1.55× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 27.8 | 1.34× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 20.7 | **chosen** (tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 28.5 | 1.38× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 31.1 | 1.56× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 27.2 | 1.37× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 19.9 | **chosen** (tuned) |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 27.5 | 1.38× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 30.7 | 1.62× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 27.1 | 1.42× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 19.0 | **chosen** (tuned) |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 27.0 | 1.42× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | 30.5 | 1.64× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 26.5 | 1.42× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 18.6 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 4 units, 4 columns per unit), box-major, 2048 MB | 25.6 | 1.38× |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(32 x 2 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (32 units) | 421.9 | 1.17× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (64 units) | 361.9 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (2 planes per cube) | 421.5 | 1.16× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (4 planes per cube) | 415.6 | 1.15× |

##### plummer, p = 12

Tuning 5.98 s of a 10 s budget; build 6.21 s; cache `L/cuda-run/autotune/plummer-cuda-f64-p12`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 2957.0 | **chosen** (tuned) |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 5812.8 | 1.97× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 3606.5 | 1.22× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 3007.6 | 1.02× |
| M2L strategy | 7 | 335952 | Rotation | 5082.8 | 1.72× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 2957.0 | **chosen** (tuned) |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 5812.8 | 1.97× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 3606.5 | 1.22× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 3007.6 | 1.02× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 1938.2 | **chosen** (tuned) |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 3949.7 | 2.04× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 2366.7 | 1.22× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 1962.7 | 1.01× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 1100.6 | **chosen** (tuned) |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 2213.5 | 2.01× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 1408.4 | 1.28× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 1150.1 | 1.04× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 634.5 | 1.04× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 1199.5 | 1.97× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 760.5 | 1.25× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 607.7 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 117.2 | 1.03× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 312.3 | 2.74× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 117.3 | 1.03× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 114.0 | **chosen** (tuned) |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 59.4 | 1.15× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 139.2 | 2.69× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 53.9 | 1.04× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 51.7 | **chosen** (tuned) |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 57.4 | 1.14× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 137.7 | 2.74× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 51.8 | 1.03× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 50.3 | **chosen** (tuned) |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 57.0 | 1.15× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 135.5 | 2.75× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 49.9 | 1.01× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 49.4 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 55.9 | 1.16× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 134.3 | 2.78× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 49.1 | 1.02× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 48.3 | **chosen** (tuned) |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 58.2 | 1.15× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 137.7 | 2.72× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 53.3 | 1.05× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 50.6 | **chosen** (tuned) |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 56.3 | 1.16× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 136.0 | 2.81× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 50.0 | 1.03× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 48.5 | **chosen** (tuned) |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 55.3 | 1.16× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 134.5 | 2.82× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 47.9 | 1.00× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 47.7 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | 54.2 | 1.17× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 132.8 | 2.86× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 47.6 | 1.03× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(64 x 2 units, 4 columns per unit), box-major, 2048 MB | 46.4 | **chosen** (tuned) |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(64 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (32 units) | 409.2 | 1.13× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (64 units) | 362.4 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (2 planes per cube) | 421.7 | 1.16× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (4 planes per cube) | 409.3 | 1.13× |

##### plummer, p = 16

Tuning 5.73 s of a 10 s budget; build 6.16 s; cache `L/cuda-run/autotune/plummer-cuda-f64-p16`: no file for the key, stored.

| decision | level | pairs | candidate | time µs | |
| --- | ---: | ---: | --- | ---: | --- |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 8824.5 | 1.28× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 6911.9 | **chosen** (tuned) |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 19044.1 | 2.76× |
| M2L strategy | 7 | 335952 | Dense, hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 9840.1 | 1.42× |
| M2L strategy | 7 | 335952 | Rotation | 10836.8 | 1.57× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 8824.5 | 1.28× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 6911.9 | **chosen** (tuned) |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 19044.1 | 2.76× |
| M2L GEMM, ≤ 524288 pairs | 7 | 335952 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 9840.1 | 1.42× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 5796.6 | 1.28× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 4517.3 | **chosen** (tuned) |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 12602.2 | 2.79× |
| M2L GEMM, ≤ 262144 pairs | 6 | 215160 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 6614.6 | 1.46× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 3162.2 | 1.24× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 2558.2 | **chosen** (tuned) |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 7109.3 | 2.78× |
| M2L GEMM, ≤ 131072 pairs | 5 | 115008 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 3704.2 | 1.45× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 1659.6 | 1.24× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 1333.3 | **chosen** (tuned) |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 3609.7 | 2.71× |
| M2L GEMM, ≤ 65536 pairs | 3 | 53352 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 1886.1 | 1.41× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 189.1 | **chosen** (tuned) |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 208.5 | 1.10× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 595.9 | 3.15× |
| M2L GEMM, ≤ 4096 pairs | 2 | 3096 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 207.7 | 1.10× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 97.3 | 1.03× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 101.9 | 1.08× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 264.7 | 2.81× |
| M2M GEMM, ≤ 4096 pairs | 6 | 2416 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 94.1 | **chosen** (tuned) |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 92.6 | 1.06× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 97.0 | 1.11× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 264.4 | 3.02× |
| M2M GEMM, ≤ 2048 pairs | 5 | 1624 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 87.7 | **chosen** (tuned) |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 89.4 | 1.08× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 94.0 | 1.13× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 261.2 | 3.14× |
| M2M GEMM, ≤ 1024 pairs | 4 | 936 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 83.1 | **chosen** (tuned) |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 87.6 | 1.08× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 94.0 | 1.16× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 257.7 | 3.17× |
| M2M GEMM, ≤ 512 pairs | 2 | 512 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 81.3 | **chosen** (tuned) |
| M2M GEMM, ≤ 64 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| M2M GEMM, ≤ 8 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 97.2 | 1.04× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 101.7 | 1.08× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 265.9 | 2.83× |
| L2L GEMM, ≤ 4096 pairs | 7 | 2416 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 93.9 | **chosen** (tuned) |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 92.3 | 1.07× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 95.6 | 1.11× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 262.7 | 3.04× |
| L2L GEMM, ≤ 2048 pairs | 6 | 1624 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 86.4 | **chosen** (tuned) |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 87.6 | 1.07× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 92.7 | 1.14× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 260.0 | 3.18× |
| L2L GEMM, ≤ 1024 pairs | 5 | 936 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 81.7 | **chosen** (tuned) |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | 86.5 | 1.10× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 2 units, 4 columns per unit), box-major, 2048 MB | 91.0 | 1.16× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(32 x 1 units, 2 columns per unit), box-major, 2048 MB | 259.3 | 3.31× |
| L2L GEMM, ≤ 512 pairs | 3 | 512 | hand-written cube(128 x 1 units, 4 columns per unit), box-major, 2048 MB | 78.4 | **chosen** (tuned) |
| L2L GEMM, ≤ 64 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| L2L GEMM, ≤ 8 pairs | – | 0 | hand-written cube(128 x 1 units, 8 columns per unit), box-major, 2048 MB | – | chosen (static rule; fewer than 512 pairs: not tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (32 units) | 422.2 | 1.16× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P cube (64 units) | 362.5 | **chosen** (tuned) |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (2 planes per cube) | 422.1 | 1.16× |
| P2P layout, ≤ 32 points per leaf | 7 | 31377217 | P2P plane (4 planes per cube) | 415.8 | 1.15× |

##### Summary

| problem | p | strategy (static rule) | strategy (tuned) | tuned decisions | cut | tuning s | build s | build from the cache s | evaluate static ms | evaluate tuned ms |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 4 | Dense | Dense | 9 | 0 | 3.85 | 3.95 | 0.09 | 2.21 | 2.10 |
| cube | 8 | Dense | Dense | 9 | 0 | 3.63 | 3.75 | 0.14 | 3.78 | 3.76 |
| cube | 12 | Dense | Dense | 9 | 0 | 3.95 | 4.14 | 0.20 | 8.65 | 8.48 |
| cube | 16 | Dense | Dense | 2 | 0 | 10.00 | 10.62 | 3.10 (7 left past the deadline tuned then) | 21.31 | 17.78 |
| plummer | 4 | Dense | Dense | 15 | 0 | 6.30 | 6.42 | 0.12 | 4.90 | 4.28 |
| plummer | 8 | Dense | Dense | 15 | 0 | 5.76 | 5.91 | 0.14 | 8.17 | 7.50 |
| plummer | 12 | Dense | Dense | 15 | 0 | 5.98 | 6.21 | 0.23 | 16.20 | 15.64 |
| plummer | 16 | Dense | Dense | 15 | 0 | 5.73 | 6.16 | 0.43 | 35.55 | 30.37 |

Backends run: cuda (f64); not run: cpu, metal.

</details>

### `m2l_kernels --device cuda`

<details><summary>Output</summary>

#### The device dense M2L per level (Phase 4 T9, C4.5)

| item | value |
| --- | --- |
| machine | Neoverse-V2 (implementer 0x41, part 0xd4f) (72 physical, 72 logical) |
| target | aarch64-linux, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| device | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| timing | median of 15 batches of >= 20 ms; launches queued between syncs, a warm-up launch and a sync first (compilation excluded) |
| peak | 67000 GFLOP/s (H100 of the GH200, f32 without tensor cores, datasheet, at the 1,980 MHz maximum SM clock (clocks not locked)); GFLOP/s counts 2 n² per pair (useful flops) |
| host | `LaplaceOperator::m2l_pair` with the dense f32 tables, 1 and 12 scoped threads |
| precision | f32 on cuda |

##### C3.2 cube, p = 3 (n = 16), cuda

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 5.3 | 300 | 0.4 | – | – | – |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 7.9 | 3447 | 5.1 | – | – | – |
| 4 | 4096 | 584136 | 316 | 1849 / 3584 | hand-written (1) | 100% | 50.8 | 5883 | 8.8 | – | – | – |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 4.6 | 5.3 | 16.2 | 26.6 | 3 | 4422.6 | 948 | hand-written | 209.9 | 474.4 | 7.9 | 17.8 | 6.5e-7 (host) |
| 3 | 4.8 | 7.9 | 50.9 | 71.0 | 3 | 4430.9 | 948 | hand-written | 3076.6 | 647.1 | 43.4 | 9.1 | 6.6e-7 (host) |
| 4 | 31.4 | 50.8 | 57.4 | 192.2 | 3 | 4359.7 | 948 | hand-written | 32440.1 | 3178.1 | 168.8 | 16.5 | 8.5e-7 (host) |

##### C3.2 cube, p = 8 (n = 81), cuda

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 23.4 | 1739 | 2.6 | – | – | – |
|  | library rejected: [316, 32, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 79.5 | 8811 | 13.2 | – | – | – |
|  | library rejected: [316, 384, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 4 | 4096 | 584136 | 316 | 1849 / 3584 | hand-written (1) | 100% | 673.3 | 11384 | 17.0 | – | – | – |
|  | library rejected: [316, 3584, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 4.6 | 23.4 | 15.6 | 44.5 | 3 | 8223.9 | 948 | hand-written | 2325.8 | 674.2 | 52.2 | 15.1 | 7.7e-7 (host) |
| 3 | 14.2 | 79.5 | 47.4 | 158.5 | 3 | 8769.2 | 948 | hand-written | 39759.6 | 4592.4 | 250.8 | 29.0 | 1.2e-6 (host) |
| 4 | 133.5 | 673.3 | 157.1 | 965.7 | 3 | 9209.0 | 948 | hand-written | 423504.3 | 50197.7 | 438.6 | 52.0 | 1.3e-6 (host) |

##### C3.2 cube, p = 12 (n = 169), cuda

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 62.8 | 2815 | 4.2 | – | – | – |
|  | library rejected: [316, 32, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 310.1 | 9829 | 14.7 | – | – | – |
|  | library rejected: [316, 384, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 4 | 4096 | 584136 | 316 | 1849 / 3584 | hand-written (1) | 100% | 2658.7 | 12550 | 18.7 | – | – | – |
|  | library rejected: [316, 3584, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 4.7 | 62.8 | 15.4 | 93.9 | 3 | 19456.1 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 3 | 30.6 | 310.1 | 47.7 | 436.1 | 3 | 21365.1 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 4 | 253.3 | 2658.7 | 248.3 | 3168.1 | 3 | 23835.2 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |

##### C3.2 cube, p = 16 (n = 289), cuda

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 140.7 | 3675 | 5.5 | – | – | – |
|  | library rejected: [316, 32, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 899.3 | 9910 | 14.8 | – | – | – |
|  | library rejected: [316, 384, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 4 | 4096 | 584136 | 316 | 1849 / 3584 | hand-written (1) | 100% | 8586.0 | 11364 | 17.0 | – | – | – |
|  | library rejected: [259, 3584, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 5.0 | 140.7 | 15.4 | 166.9 | 3 | 41567.0 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 3 | 55.4 | 899.3 | 90.9 | 1052.0 | 3 | 38667.0 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 4 | 411.1 | 8586.0 | 373.7 | 9346.6 | 3 | 39113.6 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |

##### C3.2 cube: the M2L stage (every V level)

| p | levels | pairs | device level calls µs | of which GEMM / gather / accumulate | launches | (A) µs | (A) launches | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3 | 640584 | 289.8 | 22% / 14% / 43% | 9 | 13213.2 | 2844 | 35726.5 | 4299.6 | 123.3 | 14.8 |
| 8 | 3 | 640584 | 1168.7 | 66% / 13% / 19% | 9 | 26202.1 | 2844 | 465589.7 | 55464.2 | 398.4 | 47.5 |
| 12 | 3 | 640584 | 3698.1 | 82% / 8% / 8% | 9 | 64656.4 | 2844 | – | – | – | – |
| 16 | 3 | 640584 | 10565.5 | 91% / 4% / 5% | 9 | 119347.6 | 2844 | – | – | – | – |

##### Plummer sphere, p = 3 (n = 16), cuda

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 5.3 | 301 | 0.4 | – | – | – |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 7.9 | 3446 | 5.1 | – | – | – |
| 4 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 7.9 | 3444 | 5.1 | – | – | – |
| 5 | 936 | 115008 | 316 | 364 / 704 | hand-written (1) | 100% | 12.8 | 4583 | 6.8 | – | – | – |
| 6 | 1624 | 215160 | 316 | 681 / 1280 | hand-written (1) | 100% | 20.1 | 5470 | 8.2 | – | – | – |
| 7 | 2416 | 335952 | 316 | 1063 / 1976 | hand-written (1) | 100% | 30.2 | 5690 | 8.5 | – | – | – |
| 8 | 416 | 37794 | 316 | 120 / 248 | hand-written (1) | 100% | 6.9 | 2806 | 4.2 | – | – | – |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 4.5 | 5.3 | 16.2 | 25.9 | 3 | 4334.6 | 948 | hand-written | 208.2 | 473.2 | 8.0 | 18.2 | 6.5e-7 (host) |
| 3 | 4.8 | 7.9 | 49.9 | 71.0 | 3 | 4338.5 | 948 | hand-written | 3073.2 | 658.2 | 43.3 | 9.3 | 6.6e-7 (host) |
| 4 | 4.8 | 7.9 | 50.3 | 71.0 | 3 | 4327.0 | 948 | hand-written | 3032.7 | 647.4 | 42.7 | 9.1 | 6.3e-7 (host) |
| 5 | 7.4 | 12.8 | 49.3 | 79.9 | 3 | 4323.4 | 948 | hand-written | 6518.3 | 955.9 | 81.6 | 12.0 | 7.9e-7 (host) |
| 6 | 11.7 | 20.1 | 49.9 | 92.8 | 3 | 4338.5 | 948 | hand-written | 12102.4 | 1442.1 | 130.5 | 15.5 | 8.0e-7 (host) |
| 7 | 16.8 | 30.2 | 50.4 | 122.9 | 3 | 4338.0 | 948 | hand-written | 18557.7 | 1953.1 | 151.0 | 15.9 | 8.9e-7 (host) |
| 8 | 4.5 | 6.9 | 48.5 | 65.1 | 3 | 4339.0 | 948 | hand-written | 2164.7 | 594.2 | 33.2 | 9.1 | 7.9e-7 (host) |

##### Plummer sphere, p = 8 (n = 81), cuda

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 23.4 | 1736 | 2.6 | – | – | – |
|  | library rejected: [316, 32, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 79.5 | 8809 | 13.1 | – | – | – |
|  | library rejected: [316, 384, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 4 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 79.5 | 8802 | 13.1 | – | – | – |
|  | library rejected: [316, 384, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 5 | 936 | 115008 | 316 | 364 / 704 | hand-written (1) | 100% | 165.3 | 9127 | 13.6 | – | – | – |
|  | library rejected: [316, 704, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 6 | 1624 | 215160 | 316 | 681 / 1280 | hand-written (1) | 100% | 258.0 | 10945 | 16.3 | – | – | – |
|  | library rejected: [316, 1280, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 7 | 2416 | 335952 | 316 | 1063 / 1976 | hand-written (1) | 100% | 391.9 | 11248 | 16.8 | – | – | – |
|  | library rejected: [316, 1976, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 8 | 416 | 37794 | 316 | 120 / 248 | hand-written (1) | 100% | 59.2 | 8375 | 12.5 | – | – | – |
|  | library rejected: [316, 248, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 4.5 | 23.4 | 15.6 | 45.0 | 3 | 8224.1 | 948 | hand-written | 2326.7 | 677.2 | 51.7 | 15.0 | 7.7e-7 (host) |
| 3 | 14.2 | 79.5 | 46.9 | 158.0 | 3 | 8769.9 | 948 | hand-written | 39481.6 | 4602.1 | 249.9 | 29.1 | 1.2e-6 (host) |
| 4 | 14.2 | 79.5 | 46.9 | 158.0 | 3 | 8768.9 | 948 | hand-written | 38778.1 | 4546.7 | 245.5 | 28.8 | 1.1e-6 (host) |
| 5 | 33.0 | 165.3 | 48.5 | 289.6 | 3 | 8815.3 | 948 | hand-written | 83534.7 | 10064.2 | 288.5 | 34.8 | 1.3e-6 (host) |
| 6 | 63.2 | 258.0 | 93.2 | 420.9 | 3 | 9004.2 | 948 | hand-written | 158861.9 | 17983.9 | 377.4 | 42.7 | 1.3e-6 (host) |
| 7 | 85.3 | 391.9 | 96.4 | 578.6 | 3 | 9091.4 | 948 | hand-written | 243423.2 | 28045.4 | 420.7 | 48.5 | 1.3e-6 (host) |
| 8 | 10.9 | 59.2 | 44.5 | 123.3 | 3 | 8753.9 | 948 | hand-written | 28596.1 | 4204.0 | 232.0 | 34.1 | 1.1e-6 (host) |

##### Plummer sphere, p = 12 (n = 169), cuda

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 63.5 | 2786 | 4.2 | – | – | – |
|  | library rejected: [316, 32, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 309.7 | 9840 | 14.7 | – | – | – |
|  | library rejected: [316, 384, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 4 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 309.9 | 9835 | 14.7 | – | – | – |
|  | library rejected: [316, 384, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 5 | 936 | 115008 | 316 | 364 / 704 | hand-written (1) | 100% | 568.5 | 11557 | 17.2 | – | – | – |
|  | library rejected: [316, 704, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 6 | 1624 | 215160 | 316 | 681 / 1280 | hand-written (1) | 100% | 1025.2 | 11988 | 17.9 | – | – | – |
|  | library rejected: [316, 1280, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 7 | 2416 | 335952 | 316 | 1063 / 1976 | hand-written (1) | 100% | 1570.7 | 12218 | 18.2 | – | – | – |
|  | library rejected: [316, 1976, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 8 | 416 | 37794 | 316 | 120 / 248 | hand-written (1) | 100% | 237.6 | 9084 | 13.6 | – | – | – |
|  | library rejected: [316, 248, 169] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 4.7 | 63.5 | 15.8 | 93.9 | 3 | 19363.3 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 3 | 30.6 | 309.7 | 47.9 | 434.2 | 3 | 21252.9 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 4 | 30.6 | 309.9 | 48.6 | 434.2 | 3 | 21328.7 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 5 | 66.4 | 568.5 | 93.1 | 733.4 | 3 | 22385.2 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 6 | 104.4 | 1025.2 | 97.4 | 1229.7 | 3 | 23405.5 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 7 | 152.3 | 1570.7 | 163.2 | 1892.9 | 3 | 23588.5 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 8 | 19.9 | 237.6 | 44.9 | 342.3 | 3 | 20976.0 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |

##### Plummer sphere, p = 16 (n = 289), cuda

GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's padding, and the spike at the nearest (p, B) for the mean columns per offset k.

| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) GFLOP/s | of the spike |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2 | 64 | 3096 | 316 | 10 / 32 | hand-written (1) | 100% | 140.7 | 3675 | 5.5 | – | – | – |
|  | library rejected: [316, 32, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 3 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 899.3 | 9910 | 14.8 | – | – | – |
|  | library rejected: [316, 384, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 4 | 512 | 53352 | 316 | 169 / 384 | hand-written (1) | 100% | 899.7 | 9905 | 14.8 | – | – | – |
|  | library rejected: [316, 384, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 5 | 936 | 115008 | 316 | 364 / 704 | hand-written (1) | 100% | 1807.4 | 10629 | 15.9 | – | – | – |
|  | library rejected: [316, 704, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 6 | 1624 | 215160 | 316 | 681 / 1280 | hand-written (1) | 100% | 3263.2 | 11014 | 16.4 | – | – | – |
|  | library rejected: [316, 1280, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 7 | 2416 | 335952 | 316 | 1063 / 1976 | hand-written (1) | 100% | 5010.0 | 11201 | 16.7 | – | – | – |
|  | library rejected: [316, 1976, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |
| 8 | 416 | 37794 | 316 | 120 / 248 | hand-written (1) | 100% | 660.3 | 9562 | 14.3 | – | – | – |
|  | library rejected: [316, 248, 289] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

 | | | | | | | | | | | |

Stages per level: gather, GEMM and accumulation (each over every chunk), the level call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, with the plan's GEMM unless the library rejects an offset's shape), and the host.

| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) launches | (A) GEMM | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 | relative L2 (against) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 5.0 | 140.7 | 15.6 | 167.8 | 3 | 41589.7 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 3 | 55.2 | 899.3 | 91.7 | 1048.7 | 3 | 38575.3 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 4 | 55.3 | 899.7 | 91.0 | 1049.1 | 3 | 38584.4 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 5 | 95.8 | 1807.4 | 95.8 | 2004.8 | 3 | 37992.9 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 6 | 162.0 | 3263.2 | 164.7 | 3595.6 | 3 | 38177.5 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 7 | 240.7 | 5010.0 | 235.6 | 5521.2 | 3 | 38364.0 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |
| 8 | 40.1 | 660.3 | 52.5 | 795.6 | 3 | 38683.8 | 948 | hand-written | – | – | – | – | 0.0e0 (none) |

##### Plummer sphere: the M2L stage (every V level)

| p | levels | pairs | device level calls µs | of which GEMM / gather / accumulate | launches | (A) µs | (A) launches | host 1 thread µs | host 12 threads µs | x host 1 | x host 12 |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 7 | 813714 | 528.6 | 17% / 10% / 60% | 21 | 30338.8 | 6636 | 45657.4 | 6724.1 | 86.4 | 12.7 |
| 8 | 7 | 813714 | 1773.3 | 60% / 13% / 22% | 21 | 61427.7 | 6636 | 595002.3 | 70123.5 | 335.5 | 39.5 |
| 12 | 7 | 813714 | 5160.5 | 79% / 8% / 10% | 21 | 152300.0 | 6636 | – | – | – | – |
| 16 | 7 | 813714 | 14182.8 | 89% / 5% / 5% | 21 | 271967.6 | 6636 | – | – | – | – |

##### The gate: the GEMM alone on the spike's shapes (one table, B columns)

| p | B | GEMM of the rule | µs | GFLOP/s | % peak | spike GFLOP/s | of the spike | gate (>= 80%) | hand-written µs | GFLOP/s | spike hand-written | of it |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 1000 | hand-written | 7.0 | 178 | 0.3 | – | – | – | 7.0 | 178 | – | – |
| 4 | 10000 | hand-written | 7.3 | 1715 | 2.6 | – | – | – | 7.3 | 1715 | – | – |
| 4 | 100000 | hand-written | 19.8 | 6326 | 9.4 | – | – | – | 19.8 | 6326 | – | – |
| 8 | 1000 | hand-written | 22.3 | 587 | 0.9 | – | – | – | 22.3 | 587 | – | – |
| 8 | 10000 | hand-written | 24.5 | 5351 | 8.0 | – | – | – | 24.5 | 5351 | – | – |
| 8 | 100000 | hand-written | 121.1 | 10839 | 16.2 | – | – | – | 121.1 | 10839 | – | – |
| 12 | 1000 | hand-written | 39.2 | 1457 | 2.2 | – | – | – | 39.2 | 1457 | – | – |
| 12 | 10000 | hand-written | 55.6 | 10271 | 15.3 | – | – | – | 55.6 | 10271 | – | – |
| 12 | 100000 | hand-written | 461.7 | 12373 | 18.5 | – | – | – | 461.7 | 12373 | – | – |
| 16 | 1000 | hand-written | 64.3 | 2596 | 3.9 | – | – | – | 64.3 | 2596 | – | – |
| 16 | 10000 | hand-written | 184.2 | 9071 | 13.5 | – | – | – | 184.2 | 9071 | – | – |
| 16 | 100000 | hand-written | 1481.9 | 11272 | 16.8 | – | – | – | 1481.9 | 11272 | – | – |

##### The scratch budget: the cube's level 4 at p = 8 (584136 pairs)

| budget MB | GEMM | chunks | scratch MB | level call µs | launches |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | hand-written | 46 | 8.0 | 2898.0 | 138 |
| 16 | hand-written | 23 | 16.0 | 1884.6 | 69 |
| 32 | hand-written | 12 | 32.0 | 1575.2 | 36 |
| 64 | hand-written | 6 | 64.0 | 1235.8 | 18 |
| 128 | hand-written | 3 | 128.0 | 1139.9 | 9 |
| 256 | hand-written | 2 | 256.0 | 1049.5 | 6 |
| 512 | hand-written | 1 | 361.0 | 966.2 | 3 |

##### The orientation of X and Y (T12): box-major against coefficient-major

Whole level call (gather, GEMM, reduction per chunk), µs; `cm/bm` < 1 means the coefficient-major layout is faster. Library columns where the library takes the level's shapes in that orientation (f32, p ≥ 8, a GPU).

| problem | p | level | pairs | hand bm | hand cm | cm/bm | library bm | library cm | cm/bm | best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C3.2 cube | 3 | 2 | 3096 | 26.6 | 30.1 | 1.13 | – | – | – | hand bm |
| C3.2 cube | 3 | 3 | 53352 | 70.9 | 90.1 | 1.27 | – | – | – | hand bm |
| C3.2 cube | 3 | 4 | 584136 | 192.0 | 232.8 | 1.21 | – | – | – | hand bm |
| Plummer sphere | 3 | 2 | 3096 | 27.0 | 30.1 | 1.12 | – | – | – | hand bm |
| Plummer sphere | 3 | 3 | 53352 | 70.7 | 90.5 | 1.28 | – | – | – | hand bm |
| Plummer sphere | 3 | 4 | 53352 | 70.9 | 90.3 | 1.27 | – | – | – | hand bm |
| Plummer sphere | 3 | 5 | 115008 | 79.8 | 102.7 | 1.29 | – | – | – | hand bm |
| Plummer sphere | 3 | 6 | 215160 | 92.5 | 125.7 | 1.36 | – | – | – | hand bm |
| Plummer sphere | 3 | 7 | 335952 | 122.5 | 163.7 | 1.34 | – | – | – | hand bm |
| Plummer sphere | 3 | 8 | 37794 | 65.0 | 85.5 | 1.31 | – | – | – | hand bm |
| C3.2 cube | 6 | 2 | 3096 | 35.3 | 41.9 | 1.19 | – | – | – | hand bm |
| C3.2 cube | 6 | 3 | 53352 | 95.2 | 126.4 | 1.33 | – | – | – | hand bm |
| C3.2 cube | 6 | 4 | 584136 | 463.9 | 751.6 | 1.62 | – | – | – | hand bm |
| Plummer sphere | 6 | 2 | 3096 | 35.0 | 43.2 | 1.24 | – | – | – | hand bm |
| Plummer sphere | 6 | 3 | 53352 | 95.3 | 126.6 | 1.33 | – | – | – | hand bm |
| Plummer sphere | 6 | 4 | 53352 | 95.4 | 126.3 | 1.32 | – | – | – | hand bm |
| Plummer sphere | 6 | 5 | 115008 | 147.7 | 205.8 | 1.39 | – | – | – | hand bm |
| Plummer sphere | 6 | 6 | 215160 | 243.4 | 308.2 | 1.27 | – | – | – | hand bm |
| Plummer sphere | 6 | 7 | 335952 | 325.7 | 429.8 | 1.32 | – | – | – | hand bm |
| Plummer sphere | 6 | 8 | 37794 | 84.6 | 110.0 | 1.30 | – | – | – | hand bm |
| C3.2 cube | 8 | 2 | 3096 | 46.3 | 57.5 | 1.24 | – | – | – | hand bm |
| C3.2 cube | 8 | 3 | 53352 | 158.0 | 241.1 | 1.53 | – | – | – | hand bm |
| C3.2 cube | 8 | 4 | 584136 | 966.0 | 1853.5 | 1.92 | – | – | – | hand bm |
| Plummer sphere | 8 | 2 | 3096 | 44.5 | 56.9 | 1.28 | – | – | – | hand bm |
| Plummer sphere | 8 | 3 | 53352 | 157.9 | 240.9 | 1.53 | – | – | – | hand bm |
| Plummer sphere | 8 | 4 | 53352 | 157.8 | 240.9 | 1.53 | – | – | – | hand bm |
| Plummer sphere | 8 | 5 | 115008 | 289.4 | 405.0 | 1.40 | – | – | – | hand bm |
| Plummer sphere | 8 | 6 | 215160 | 421.2 | 626.4 | 1.49 | – | – | – | hand bm |
| Plummer sphere | 8 | 7 | 335952 | 578.3 | 985.6 | 1.70 | – | – | – | hand bm |
| Plummer sphere | 8 | 8 | 37794 | 122.9 | 174.9 | 1.42 | – | – | – | hand bm |
| C3.2 cube | 12 | 2 | 3096 | 96.4 | 108.3 | 1.12 | – | – | – | hand bm |
| C3.2 cube | 12 | 3 | 53352 | 436.4 | 515.0 | 1.18 | – | – | – | hand bm |
| C3.2 cube | 12 | 4 | 584136 | 3169.6 | 5315.4 | 1.68 | – | – | – | hand bm |
| Plummer sphere | 12 | 2 | 3096 | 96.4 | 108.5 | 1.13 | – | – | – | hand bm |
| Plummer sphere | 12 | 3 | 53352 | 436.9 | 515.4 | 1.18 | – | – | – | hand bm |
| Plummer sphere | 12 | 4 | 53352 | 437.1 | 515.6 | 1.18 | – | – | – | hand bm |
| Plummer sphere | 12 | 5 | 115008 | 733.6 | 966.2 | 1.32 | – | – | – | hand bm |
| Plummer sphere | 12 | 6 | 215160 | 1230.5 | 1855.3 | 1.51 | – | – | – | hand bm |
| Plummer sphere | 12 | 7 | 335952 | 1892.1 | 2969.3 | 1.57 | – | – | – | hand bm |
| Plummer sphere | 12 | 8 | 37794 | 342.6 | 398.8 | 1.16 | – | – | – | hand bm |
| C3.2 cube | 16 | 2 | 3096 | 166.8 | 184.8 | 1.11 | – | – | – | hand bm |
| C3.2 cube | 16 | 3 | 53352 | 1052.0 | 1242.8 | 1.18 | – | – | – | hand bm |
| C3.2 cube | 16 | 4 | 584136 | 9343.8 | 13755.3 | 1.47 | – | – | – | hand bm |
| Plummer sphere | 16 | 2 | 3096 | 166.5 | 185.3 | 1.11 | – | – | – | hand bm |
| Plummer sphere | 16 | 3 | 53352 | 1052.8 | 1242.1 | 1.18 | – | – | – | hand bm |
| Plummer sphere | 16 | 4 | 53352 | 1050.2 | 1241.8 | 1.18 | – | – | – | hand bm |
| Plummer sphere | 16 | 5 | 115008 | 2004.5 | 2613.4 | 1.30 | – | – | – | hand bm |
| Plummer sphere | 16 | 6 | 215160 | 3593.4 | 4883.7 | 1.36 | – | – | – | hand bm |
| Plummer sphere | 16 | 7 | 335952 | 5521.0 | 7834.9 | 1.42 | – | – | – | hand bm |
| Plummer sphere | 16 | 8 | 37794 | 795.2 | 903.6 | 1.14 | – | – | – | hand bm |

</details>

### the GEMM spike, `--backends cuda`

<details><summary>Output</summary>

| backend | precision | p | Nc | B | implementation | median ms | reps | GFLOP/s | rel. error | check |
|---|---|---|---|---|---|---|---|---|---|---|
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_auto | 0.0148 | 51200 | 84.5 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0148 | 51200 | 84.5 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0161 | 51200 | 77.6 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | 0.0050 | 51200 | 250.1 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0077 | 51200 | 161.7 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0076 | 51200 | 164.3 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_auto | 0.0148 | 51200 | 846.5 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0148 | 51200 | 844.8 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0172 | 51200 | 727.7 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 10000 | tiled-smem(tm=2) | 0.0052 | 51200 | 2383.2 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0078 | 51200 | 1604.2 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0089 | 51200 | 1397.4 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_auto | 0.0384 | 26624 | 3256.1 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.0383 | 26624 | 3262.0 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_double_unit_max_tile_size | 0.0863 | 12288 | 1449.2 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 100000 | tiled-smem(tm=2) | 0.0177 | 51200 | 7079.7 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.0133 | 51200 | 9409.1 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.0333 | 30720 | 3750.0 | 2.03e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_auto | 0.0165 | 51200 | 797.4 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0165 | 51200 | 797.6 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0311 | 32768 | 421.3 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | 0.0112 | 51200 | 1171.6 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0199 | 51200 | 660.6 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0203 | 50176 | 647.5 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_auto | 0.0170 | 51200 | 7707.5 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0170 | 51200 | 7712.9 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0457 | 22528 | 2868.9 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 10000 | tiled-smem(tm=6) | 0.0177 | 51200 | 7414.0 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0226 | 45056 | 5816.2 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0251 | 39936 | 5224.8 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_auto | 0.1157 | 8704 | 11346.3 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.1156 | 8704 | 11354.8 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_double_unit_max_tile_size | 0.2648 | 3840 | 4955.7 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 100000 | tiled-smem(tm=6) | 0.0930 | 11264 | 14113.5 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.0979 | 10240 | 13398.8 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.1695 | 6144 | 7741.8 | 4.03e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_auto | 0.0297 | 33792 | 1923.3 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0297 | 33792 | 1922.2 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0639 | 16384 | 894.2 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 1000 | tiled-smem(tm=11) | 0.0274 | 36864 | 2083.1 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0398 | 25600 | 1435.6 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0443 | 23552 | 1288.5 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_auto | 0.0531 | 19456 | 10757.4 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0531 | 19456 | 10767.4 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_double_unit_max_tile_size | 0.1157 | 8704 | 4937.0 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 10000 | tiled-smem(tm=11) | 0.0460 | 22528 | 12414.9 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0511 | 20480 | 11181.3 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0696 | 15360 | 8210.7 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_auto | 0.4208 | 2432 | 13574.3 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.4208 | 2432 | 13575.6 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_double_unit_max_tile_size | 1.0402 | 1024 | 5491.2 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 100000 | tiled-smem(tm=11) | 0.2671 | 3840 | 21389.9 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.3486 | 3072 | 16385.6 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.6155 | 1664 | 9280.3 | 6.95e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_auto | 0.0464 | 22528 | 3600.4 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0464 | 22528 | 3600.6 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_double_unit_max_tile_size | 0.1097 | 9216 | 1522.3 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 1000 | tiled-smem(tm=10) | 0.0454 | 22528 | 3679.7 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0680 | 15360 | 2457.3 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0755 | 13312 | 2213.0 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_auto | 0.0924 | 11264 | 18071.9 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0924 | 11264 | 18078.3 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_double_unit_max_tile_size | 0.2837 | 3584 | 5887.7 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 10000 | tiled-smem(tm=10) | 0.1191 | 8704 | 14026.4 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0952 | 11264 | 17550.7 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.1868 | 5632 | 8941.9 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_auto | 1.0161 | 1024 | 16439.7 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 1.0152 | 1024 | 16454.2 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 2.7446 | 384 | 6086.3 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 0.9096 | 1152 | 18363.9 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.9770 | 1024 | 17097.1 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 1.7796 | 576 | 9386.5 | 7.88e-7 | pass |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_auto | 0.0170 | 51200 | 73.4 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0170 | 51200 | 73.6 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0213 | 47104 | 58.8 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 1000 | tiled-smem(tm=2) | 0.0051 | 51200 | 247.0 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0091 | 51200 | 136.6 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0118 | 51200 | 105.5 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_auto | 0.0174 | 51200 | 720.1 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0173 | 51200 | 721.6 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0362 | 28672 | 345.8 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 10000 | tiled-smem(tm=2) | 0.0068 | 51200 | 1842.0 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0098 | 51200 | 1270.5 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0128 | 51200 | 975.2 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_auto | 0.0989 | 10240 | 1264.2 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.0989 | 10240 | 1264.0 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_double_unit_max_tile_size | 0.2139 | 4864 | 584.4 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 100000 | tiled-smem(tm=2) | 0.0287 | 35840 | 4360.4 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.0395 | 25600 | 3163.9 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.0714 | 14336 | 1751.9 | 3.83e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_auto | 0.0359 | 28672 | 366.0 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0359 | 28672 | 365.5 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0531 | 19456 | 247.0 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 1000 | tiled-smem(tm=6) | 0.0165 | 51200 | 795.4 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0243 | 41984 | 539.0 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0347 | 29696 | 377.7 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_auto | 0.0372 | 27648 | 3531.3 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0371 | 27648 | 3533.7 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0864 | 12288 | 1519.4 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 10000 | tiled-smem(tm=6) | 0.0292 | 34816 | 4490.9 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0292 | 34816 | 4486.3 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0418 | 24576 | 3139.9 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_auto | 0.2379 | 4352 | 5516.8 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.2378 | 4352 | 5519.2 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_double_unit_max_tile_size | 0.5369 | 1920 | 2444.0 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 100000 | tiled-smem(tm=6) | 0.1601 | 6656 | 8194.6 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.2411 | 4352 | 5442.0 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.2977 | 3584 | 4408.3 | 4.25e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_auto | 0.0643 | 16384 | 888.4 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0643 | 16384 | 889.0 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0984 | 10240 | 580.4 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 1000 | tiled-smem(tm=11) | 0.0591 | 17408 | 966.8 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0524 | 19456 | 1090.5 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0716 | 14336 | 797.9 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_auto | 0.1202 | 8704 | 4752.0 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.1200 | 8704 | 4758.4 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_double_unit_max_tile_size | 0.2349 | 4352 | 2431.7 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 10000 | tiled-smem(tm=11) | 0.1166 | 8704 | 4899.3 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0937 | 11264 | 6096.3 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0983 | 10240 | 5811.5 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_auto | 0.8586 | 1216 | 6652.7 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.8582 | 1216 | 6655.8 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_double_unit_max_tile_size | 1.9096 | 544 | 2991.3 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 100000 | tiled-smem(tm=11) | 0.7363 | 1408 | 7758.5 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.8957 | 1152 | 6377.1 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 1.1660 | 896 | 4899.1 | 6.00e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_auto | 0.1004 | 10240 | 1663.1 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.1003 | 10240 | 1664.9 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_double_unit_max_tile_size | 0.1628 | 6144 | 1026.1 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 1000 | tiled-smem(tm=10) | 0.0705 | 14336 | 2369.4 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0882 | 12288 | 1893.6 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.1306 | 7680 | 1279.3 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_auto | 0.2265 | 4608 | 7373.8 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.2265 | 4608 | 7374.7 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_double_unit_max_tile_size | 0.5421 | 1920 | 3081.4 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 10000 | tiled-smem(tm=10) | 0.2109 | 4864 | 7920.6 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.3428 | 3072 | 4872.3 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.3786 | 2816 | 4411.6 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_auto | 2.1399 | 480 | 7806.2 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 2.1374 | 480 | 7815.1 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 5.0307 | 208 | 3320.4 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 100000 | tiled-smem(tm=10) | 1.6781 | 608 | 9954.5 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 2.5529 | 416 | 6543.1 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 3.2039 | 320 | 5213.8 | 6.78e-16 | pass |

</details>

### `device_fmm --part leaf --n 1000000`, f32

<details><summary>Output</summary>

#### The device FMM against the host path (Phase 4 T13)

| item | value |
| --- | --- |
| machine | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical; aarch64-linux, release build; rustc 1.99.0 (b940084d7 2026-09-28) |
| device | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| BLAS threads | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1 |
| precision | f32 |
| host | `P2pChoice::Auto` at 1 and 72 threads; one rank |
| timing | evaluate: median wall time of 5 evaluations of the first charge vector after the eight error evaluations (compilation excluded; one sync per device evaluation); stages: on the device the medians of a synchronous-stages build (a sync after each stage), on the host the medians of the timed evaluations |
| tuning | no tuning cache: no tuned rows |
| tables | through the table cache `L/tables` |
| wall time of every run | 404 s |

##### Device leaf-size study (cuda, f32, static rule)

Times in ms: stages from the synchronous-stages build; far = upward + downward; near share = leaves / sum of the stages; evaluate = the median of the default build.

| distribution | p | max points per leaf | leaves | points per leaf (min / mean / max) | V pairs | φ L2 | ∇φ L2 | far | leaves | near share | evaluate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 3 | 16 | 261241 | 0 / 3.8 / 16 | 51981072 | 2.889e-3 | 4.526e-3 | 10.45 | 3.09 | 12% | 27.12 |
| cube | 3 | 32 | 113450 | 0 / 8.8 / 32 | 11756358 | 2.828e-3 | 3.548e-3 | 3.51 | 2.34 | 15% | 16.98 |
| cube | 3 | 64 | 32768 | 11 / 30.5 / 57 | 6039504 | 2.808e-3 | 3.014e-3 | 1.53 | 2.17 | 18% | 13.94 |
| cube | 3 | 128 | 32768 | 11 / 30.5 / 57 | 6039504 | 2.808e-3 | 3.014e-3 | 1.53 | 2.17 | 17% | 13.66 |
| cube | 3 | 256 | 10347 | 16 / 96.6 / 256 | 895614 | 2.658e-3 | 2.093e-3 | 1.29 | 11.31 | 52% | 22.72 |
| cube | 8 | 16 | 261241 | 0 / 3.8 / 16 | 51981072 | 1.915e-5 | 5.772e-5 | 97.93 | 5.51 | 5% | 119.43 |
| cube | 8 | 32 | 113450 | 0 / 8.8 / 32 | 11756358 | 1.879e-5 | 4.822e-5 | 20.56 | 4.52 | 13% | 37.16 |
| cube | 8 | 64 | 32768 | 11 / 30.5 / 57 | 6039504 | 1.846e-5 | 4.026e-5 | 10.06 | 2.48 | 12% | 23.07 |
| cube | 8 | 128 | 32768 | 11 / 30.5 / 57 | 6039504 | 1.846e-5 | 4.026e-5 | 10.05 | 2.51 | 12% | 22.82 |
| cube | 8 | 256 | 10347 | 16 / 96.6 / 256 | 895614 | 1.754e-5 | 2.801e-5 | 3.27 | 12.48 | 51% | 25.73 |
| plummer | 3 | 16 | 205402 | 0 / 4.9 / 16 | 37110816 | 3.588e-3 | 2.547e-3 | 7.70 | 3.49 | 16% | 23.30 |
| plummer | 3 | 32 | 108893 | 0 / 9.2 / 32 | 19888158 | 3.566e-3 | 2.396e-3 | 4.82 | 2.95 | 18% | 18.51 |
| plummer | 3 | 64 | 53117 | 0 / 18.8 / 64 | 9461436 | 3.539e-3 | 2.040e-3 | 3.25 | 4.24 | 27% | 16.91 |
| plummer | 3 | 128 | 26083 | 0 / 38.3 / 128 | 4552950 | 3.487e-3 | 1.666e-3 | 2.48 | 8.86 | 45% | 20.78 |
| plummer | 3 | 256 | 13784 | 0 / 72.5 / 256 | 2262846 | 3.458e-3 | 1.555e-3 | 2.29 | 22.39 | 68% | 33.56 |
| plummer | 8 | 16 | 205402 | 0 / 4.9 / 16 | 37110816 | 2.434e-5 | 3.086e-5 | 61.30 | 6.82 | 9% | 82.60 |
| plummer | 8 | 32 | 108893 | 0 / 9.2 / 32 | 19888158 | 2.418e-5 | 2.884e-5 | 32.51 | 4.84 | 10% | 49.53 |
| plummer | 8 | 64 | 53117 | 0 / 18.8 / 64 | 9461436 | 2.405e-5 | 2.480e-5 | 16.48 | 6.21 | 20% | 33.20 |
| plummer | 8 | 128 | 26083 | 0 / 38.3 / 128 | 4552950 | 2.395e-5 | 2.069e-5 | 9.23 | 10.63 | 38% | 29.81 |
| plummer | 8 | 256 | 13784 | 0 / 72.5 / 256 | 2262846 | 2.379e-5 | 1.937e-5 | 6.04 | 24.86 | 63% | 40.57 |

###### Fastest leaf size per configuration

| distribution | p | fastest | evaluate (ms) | at 64 (ms) | 64 / fastest | evaluate at 16, 32, 64, 128, 256 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cube | 3 | 128 | 13.66 | 13.94 | 1.02 | 27.12, 16.98, 13.94, 13.66, 22.72 |
| cube | 8 | 128 | 22.82 | 23.07 | 1.01 | 119.43, 37.16, 23.07, 22.82, 25.73 |
| plummer | 3 | 64 | 16.91 | 16.91 | 1.00 | 23.30, 18.51, 16.91, 20.78, 33.56 |
| plummer | 8 | 128 | 29.81 | 33.20 | 1.11 | 82.60, 49.53, 33.20, 29.81, 40.57 |

###### The device leaf-size rule (T13)

Geometric mean of the cuda evaluation time over the 4 configurations (cube and Plummer, p = [3, 8], f32). The fastest size replaces 64 only if it is at least 5% faster by this measure and no φ or ∇φ L2 error at it is more than 10% worse than at 64.

| max points per leaf | geometric mean (ms) | relative to 64 |
| ---: | ---: | ---: |
| 16 | 49.96 | 2.424 |
| 32 | 27.58 | 1.338 |
| 64 | 20.62 | 1.000 |
| 128 | 20.96 | 1.017 |
| 256 | 29.87 | 1.449 |

Fastest: 64 (1.000x faster than 64); largest error growth there: 1.000x. **The rule picks 64.**

Backends run: cuda (f32); host (Neoverse-V2 (implementer 0x41, part 0xd4f)); not run: cpu, metal.

</details>

### `device_fmm --part leaf --n 1000000`, f64

<details><summary>Output</summary>

#### The device FMM against the host path (Phase 4 T13)

| item | value |
| --- | --- |
| machine | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical; aarch64-linux, release build; rustc 1.99.0 (b940084d7 2026-09-28) |
| device | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| BLAS threads | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1 |
| precision | f64 |
| host | `P2pChoice::Auto` at 1 and 72 threads; one rank |
| timing | evaluate: median wall time of 5 evaluations of the first charge vector after the eight error evaluations (compilation excluded; one sync per device evaluation); stages: on the device the medians of a synchronous-stages build (a sync after each stage), on the host the medians of the timed evaluations |
| tuning | no tuning cache: no tuned rows |
| tables | through the table cache `L/tables` |
| wall time of every run | 415 s |

##### Device leaf-size study (cuda, f64, static rule)

Times in ms: stages from the synchronous-stages build; far = upward + downward; near share = leaves / sum of the stages; evaluate = the median of the default build.

| distribution | p | max points per leaf | leaves | points per leaf (min / mean / max) | V pairs | φ L2 | ∇φ L2 | far | leaves | near share | evaluate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 3 | 16 | 261241 | 0 / 3.8 / 16 | 51981072 | 2.889e-3 | 4.526e-3 | 14.72 | 3.97 | 11% | 39.21 |
| cube | 3 | 32 | 113450 | 0 / 8.8 / 32 | 11756358 | 2.828e-3 | 3.548e-3 | 5.13 | 3.58 | 16% | 25.78 |
| cube | 3 | 64 | 32768 | 11 / 30.5 / 57 | 6039504 | 2.808e-3 | 3.014e-3 | 1.95 | 3.38 | 18% | 21.78 |
| cube | 3 | 128 | 32768 | 11 / 30.5 / 57 | 6039504 | 2.808e-3 | 3.014e-3 | 1.95 | 3.38 | 18% | 21.85 |
| cube | 3 | 256 | 10347 | 16 / 96.6 / 256 | 895614 | 2.658e-3 | 2.093e-3 | 2.44 | 18.13 | 53% | 37.19 |
| cube | 8 | 16 | 261241 | 0 / 3.8 / 16 | 51981072 | 1.915e-5 | 5.771e-5 | 168.62 | 8.28 | 4% | 202.72 |
| cube | 8 | 32 | 113450 | 0 / 8.8 / 32 | 11756358 | 1.879e-5 | 4.822e-5 | 33.92 | 7.32 | 13% | 60.89 |
| cube | 8 | 64 | 32768 | 11 / 30.5 / 57 | 6039504 | 1.846e-5 | 4.024e-5 | 16.30 | 3.89 | 12% | 37.30 |
| cube | 8 | 128 | 32768 | 11 / 30.5 / 57 | 6039504 | 1.846e-5 | 4.024e-5 | 16.29 | 3.89 | 12% | 37.10 |
| cube | 8 | 256 | 10347 | 16 / 96.6 / 256 | 895614 | 1.744e-5 | 2.793e-5 | 4.88 | 20.36 | 53% | 42.05 |
| plummer | 3 | 16 | 205402 | 0 / 4.9 / 16 | 37110816 | 3.587e-3 | 2.547e-3 | 10.44 | 4.92 | 16% | 34.10 |
| plummer | 3 | 32 | 108893 | 0 / 9.2 / 32 | 19888158 | 3.566e-3 | 2.396e-3 | 6.35 | 4.20 | 17% | 27.60 |
| plummer | 3 | 64 | 53117 | 0 / 18.8 / 64 | 9461436 | 3.539e-3 | 2.040e-3 | 4.23 | 6.10 | 26% | 26.97 |
| plummer | 3 | 128 | 26083 | 0 / 38.3 / 128 | 4552950 | 3.487e-3 | 1.666e-3 | 3.27 | 12.40 | 44% | 32.15 |
| plummer | 3 | 256 | 13784 | 0 / 72.5 / 256 | 2262846 | 3.458e-3 | 1.555e-3 | 3.00 | 30.27 | 65% | 49.58 |
| plummer | 8 | 16 | 205402 | 0 / 4.9 / 16 | 37110816 | 2.432e-5 | 3.084e-5 | 101.71 | 10.49 | 8% | 135.69 |
| plummer | 8 | 32 | 108893 | 0 / 9.2 / 32 | 19888158 | 2.417e-5 | 2.882e-5 | 53.01 | 7.49 | 10% | 80.09 |
| plummer | 8 | 64 | 53117 | 0 / 18.8 / 64 | 9461436 | 2.405e-5 | 2.473e-5 | 25.87 | 9.19 | 19% | 52.82 |
| plummer | 8 | 128 | 26083 | 0 / 38.3 / 128 | 4552950 | 2.390e-5 | 2.060e-5 | 13.92 | 15.53 | 36% | 46.20 |
| plummer | 8 | 256 | 13784 | 0 / 72.5 / 256 | 2262846 | 2.374e-5 | 1.928e-5 | 8.68 | 34.37 | 61% | 59.70 |

###### Fastest leaf size per configuration

| distribution | p | fastest | evaluate (ms) | at 64 (ms) | 64 / fastest | evaluate at 16, 32, 64, 128, 256 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cube | 3 | 64 | 21.78 | 21.78 | 1.00 | 39.21, 25.78, 21.78, 21.85, 37.19 |
| cube | 8 | 128 | 37.10 | 37.30 | 1.01 | 202.72, 60.89, 37.30, 37.10, 42.05 |
| plummer | 3 | 64 | 26.97 | 26.97 | 1.00 | 34.10, 27.60, 26.97, 32.15, 49.58 |
| plummer | 8 | 128 | 46.20 | 52.82 | 1.14 | 135.69, 80.09, 52.82, 46.20, 59.70 |

###### The device leaf-size rule (T13)

Geometric mean of the cuda evaluation time over the 4 configurations (cube and Plummer, p = [3, 8], f64). The fastest size replaces 64 only if it is at least 5% faster by this measure and no φ or ∇φ L2 error at it is more than 10% worse than at 64.

| max points per leaf | geometric mean (ms) | relative to 64 |
| ---: | ---: | ---: |
| 16 | 77.88 | 2.374 |
| 32 | 43.16 | 1.316 |
| 64 | 32.80 | 1.000 |
| 128 | 33.12 | 1.010 |
| 256 | 46.38 | 1.414 |

Fastest: 64 (1.000x faster than 64); largest error growth there: 1.000x. **The rule picks 64.**

Backends run: cuda (f64); host (Neoverse-V2 (implementer 0x41, part 0xd4f)); not run: cpu, metal.

</details>

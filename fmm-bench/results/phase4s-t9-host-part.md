# Phase 4S T9: the host part of an evaluation, before and after

The output pass, the charge load and reusable outputs (docs/phase4s/T9-output-pass.md,
C4S.8). These are the measurements behind docs/design/device-path.md §18.4, which
summarises them. Run on 2026-10-08: locust 06:10–07:20 UTC, the M3 Max around 07:10 and
07:45 local time.

**Every number is measured**: on locust (CUDA on the H100; NEON on the 72 Grace cores
for the host, `P2pChoice::Auto`) or on the M3 Max (Metal). Timings are reported and never
asserted.

## Setup

| item | value |
| --- | --- |
| machines | locust: GH200 480GB, 72 Neoverse-V2 cores, one H100 (96 GB), driver 565.57.01, the spack environment of tools/gh200/ (CUDA 12.6), rustc 1.99.0, CubeCL 0.11.0-pre.4 (LLVM NVPTX); the M3 Max: Metal through wgpu, rustc 1.99.0 |
| code | before: main `efdcb8d` with T9 item 1 alone (the `load` and `output` columns and `StageTimings::download`; no change to an evaluation), synced to `/data/ucahtbe/fmm/t9-before`; after: this task's working tree, synced to `/data/ucahtbe/fmm/tbetcke/phase4s_t9`. Both report `efdcb8d-dirty` |
| problem | `nd-fmm-bench`: N points uniform in [0, 1]³ (seed `0xbe9c06`), sources equal to targets, gradients on, `max_level` 16, 64 points per leaf, the static rule (`Dense`), tables from a table cache |
| timing | 2 warm-ups, 10 timed evaluations (min, median, mean, max, standard deviation); load and output from a second build with `KindTiming::Synchronous` (a sync after the charge upload and after every level call, so `load` holds the upload and the zeroing on a device); `BLAS` thread variables 1; the host at 72 threads |
| load | checked before and after every step (`L/before/load`, `L/after/load`; L = `/data/ucahtbe/logs/t9`): no compute process on the GPU at any check, and the only other user's process an `nvitop` monitor at 0.8% of a core. Each step waited for no GPU process and a load average below 4 (twice for up to three minutes, after this run's own 72-thread steps). The utilisation and SM clock read right after a step are this run's own |
| clocks | not locked (no administrator access): the application and maximum clocks are 1,980 MHz (SM) and 2,619 MHz (memory), 345 MHz idle (`nvidia-smi -q -d CLOCK` before and after) |
| M3 Max | Metal f32, N = 10⁶, p = 3 and 8, 16 threads; a desktop session with a load average of 6–8 (Activity Monitor, WebKit, Defender), so the Metal numbers are indicative |

Commands (`tools/bench/run.sh`, with `--table-cache L/tables --output …`):

```sh
# locust, before (t9-before tree) and after (branch tree)
tools/bench/run.sh --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8
tools/bench/run.sh --backend host --threads 72 --n 1e6,1e7 --precision f32,f64 --degree 3,8
# locust, after only
tools/bench/run.sh --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8 --reuse-output
tools/bench/run.sh --backend cuda --n 1e7 --precision f32,f64 --degree 3,8 --output-pass host
tools/bench/run.sh --backend host --threads 72 --n 1e6,1e7 --precision f32,f64 --degree 3,8 --reuse-output
nsys profile --trace=cuda,osrt --sample=process-tree target/release/nd-fmm-bench --backend cuda \
    --n 1e7 --precision f32 --degree 3 --repeats 3 --warmup 1 --kinds off --accuracy off
# the M3 Max, outside the sandbox: before, after, after with --reuse-output
tools/bench/run.sh --backend metal --precision f32 --degree 3,8 --n 1e6 [--reuse-output]
```

## Whole outputs: the code before T9 against the code after

The C3.2 tree of `fmm-exec/tests/accuracy.rs` (N = 10⁵ uniform in [−1, 1)³, a uniform
level-4 tree) at p = 6, first charge vector. The FNV-1a hash of the output bits is printed
by `fmm-exec/tests/output_common` (after) and by a scratch test with the same problem and
hash, run on the code before T9 (not committed). The test after also asserts the output
against the old pass on the same build (`Fmm::reference_output`).

| backend | precision | gradients | before | after |
| --- | --- | --- | --- | --- |
| host, 1 and 4 threads (M3 Max and Grace) | f64 | off | `0x94e1f03747ea1f20` | `0x94e1f03747ea1f20` |
| host, 1 and 4 threads (M3 Max and Grace) | f64 | on | `0xaf5fddbe12a92f52` | `0xaf5fddbe12a92f52` |
| CPU runtime (M3 Max), device pass and host pass | f64 | off | `0xa71c8a351574c16d` | `0xa71c8a351574c16d` |
| CPU runtime (M3 Max), device pass and host pass | f64 | on | `0x7fd66135393ff4b1` | `0x7fd66135393ff4b1` |
| Metal (M3 Max), host pass | f32 | off | `0x7fb4b39db44e3d21` | `0x7fb4b39db44e3d21` |
| Metal (M3 Max), host pass | f32 | on | `0x0dc405d9b331a414` | `0x0dc405d9b331a414` |
| CUDA, device pass and host pass | f32 | off | `0xc57633de922c6e64` | `0xc57633de922c6e64` |
| CUDA, device pass and host pass | f32 | on | `0xcc311d1921bfb320` | `0xcc311d1921bfb320` |
| CUDA, device pass and host pass | f64 | off | `0xa71c8a351574c16d` | `0xa71c8a351574c16d` |
| CUDA, device pass and host pass | f64 | on | `0x284e59e787590ed6` | `0x284e59e787590ed6` |

## Summary

Means in ms per evaluation; device-path.md §18.4 has the discussion.

| N | precision | p | CUDA before | CUDA after | `--reuse-output` | host pass on CUDA | host 72 threads before | after | `--reuse-output` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10⁶ | f32 | 3 | 12.30 | 6.63 | 6.62 | – | 27.40 | 20.44 | 20.65 |
| 10⁶ | f32 | 8 | 21.43 | 15.62 | 15.60 | – | 110.12 | 101.17 | 100.97 |
| 10⁷ | f32 | 3 | 221.34 | 60.35 | 59.71 | 65.50 | 339.93 | 186.67 | 185.36 |
| 10⁷ | f32 | 8 | 314.64 | 153.95 | 153.46 | 159.48 | 1,085.58 | 903.12 | 901.91 |
| 10⁶ | f64 | 3 | 20.10 | 10.19 | 10.21 | – | 52.57 | 40.30 | 40.40 |
| 10⁶ | f64 | 8 | 35.43 | 25.79 | 25.63 | – | 217.42 | 200.92 | 199.69 |
| 10⁷ | f64 | 3 | 357.57 | 102.99 | 102.83 | 111.15 | 690.33 | 436.29 | 445.40 |
| 10⁷ | f64 | 8 | 523.87 | 268.82 | 274.76 | 275.86 | 2,121.75 | 1,846.81 | 1,847.62 |

Load and output (ms; on CUDA the download's part in brackets):

| N | precision | p | CUDA load | CUDA output | host load | host output |
| --- | --- | ---: | --- | --- | --- | --- |
| 10⁶ | f32 | 3 | 3.17 → 0.72 | 5.23 (1.19) → 2.33 (1.25) | 3.19 → 0.13 | 5.04 → 0.95 |
| 10⁶ | f32 | 8 | 3.24 → 0.71 | 5.26 (1.20) → 2.40 (1.26) | 3.16 → 0.13 | 6.90 → 1.02 |
| 10⁷ | f32 | 3 | 63.18 → 6.62 | 119.58 (12.00) → 15.39 (12.69) | 51.70 → 1.47 | 110.55 → 8.52 |
| 10⁷ | f32 | 8 | 62.89 → 6.67 | 120.32 (12.06) → 15.32 (12.66) | 57.30 → 1.52 | 137.24 → 12.97 |
| 10⁶ | f64 | 3 | 5.07 → 1.17 | 7.87 (2.19) → 3.27 (2.25) | 4.58 → 0.17 | 10.06 → 1.10 |
| 10⁶ | f64 | 8 | 5.20 → 1.25 | 7.96 (2.20) → 3.31 (2.28) | 4.91 → 0.19 | 12.81 → 1.19 |
| 10⁷ | f64 | 3 | 92.49 → 13.58 | 203.79 (24.10) → 28.57 (24.61) | 73.98 → 2.15 | 197.31 → 11.86 |
| 10⁷ | f64 | 8 | 92.91 → 13.30 | 203.58 (24.10) → 32.82 (24.81) | 78.66 → 2.38 | 207.81 → 11.83 |

The host pass on CUDA at N = 10⁷ (`--output-pass host`): output 20.37 / 20.53 ms (f32,
p = 3 / 8) and 36.18 / 36.26 ms (f64), of it the download 11.8 / 24.0 ms.

Metal (M3 Max, f32, N = 10⁶): p = 3 16.04 → 14.06 ms (`--reuse-output` 14.09), load
2.45 → 1.67, output 3.15 → 2.15 (download 0.73 → 0.67); p = 8 82.21 → 83.96 ms
(`--reuse-output` 83.23), with M2L 71.31 → 72.72 ms of it.

## The `nsys` trace after (N = 10⁷, f32, p = 3)

60.7 ms per evaluation under the profiler (60.3 without). The CPU samples in the 219 ms
between the first and the last sample inside `Fmm::evaluate` (one warm-up and three timed
evaluations), from the exported SQLite database:
- pool threads (72): the copy into the `Output` 197 samples, the charge load 179, rayon
  and crossbeam scheduling and unresolved addresses about 130; the copy and the load
  under 1 ms of wall time per evaluation each;
- the main thread: at least 130 samples, of them 34 in `memcpy` with no attributable
  stack, 9 in `Device::download`, about 45 at unresolved addresses, 23 in the benchmark's
  comparison of outputs between evaluations, 9 in `memset`. The main thread waits in the
  download's sync most of the time, and a waiting thread is not sampled.

The CUDA API summary (the whole run) is as in T8: `cuMemAllocHost` 6 calls, 75.0 ms,
median 2.46 ms, at most 54.0 ms (the pinned buffer of each download, CubeCL's read path),
and the device-to-host copy of 160 MB 0.54 ms on the GPU.

## Checks run for T9

On the M3 Max (sandboxed unless stated):
- `cargo fmt -- --check`, `cargo clippy -- -D warnings`, `cargo clippy --examples -- -D
  warnings`, `RUST_MIN_STACK=8388608 cargo test` (the debug `tests/mpi_exec.rs` in
  53.3 s), `cargo doc --no-deps`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `RUST_MIN_STACK=8388608 cargo test --workspace`: pass;
- nd-fmm-kernels: clippy with `cpu` and with `cpu,metal`, `cargo check --features cuda`,
  `cargo test`, `cargo test --features cpu --release -- --show-output`, `cargo doc`: pass;
  by hand outside the sandbox `cargo test --release --features metal -- --ignored
  --show-output`: 49 passed (the output gather's f64 refusal on Metal among them);
- nd-fmm-exec: clippy without features and with `cpu,metal`, `cargo check --features
  cuda`, `cargo test`, `cargo test --features cpu --release`, `cargo test --release --
  --ignored`, `cargo test --features cpu --release -- --ignored` (667 s), `cargo doc
  --features cpu`: pass; by hand outside the sandbox `cargo test --features metal
  --release -- --ignored`: pass;
- nd-fmm-bench: clippy without features and with `cpu,metal`, `cargo check --features
  cuda`, `cargo test`, `cargo test --features cpu --release`: pass.

On locust (after `tools/gh200/sync.sh`, `L/after/checks`): `cargo test -p nd-fmm-kernels
--release --features cpu,cuda -- --include-ignored --show-output` (126 passed, 58 of them
on CUDA), the clippy runs of nd-fmm-kernels and nd-fmm-exec with `cpu,cuda`,
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release` and the same
with `-- --ignored` (19 min): pass. Also `RUST_MIN_STACK=8388608 cargo test --workspace` and the nd-fmm-bench tests
without features and with `cpu --release`: pass. `tools/gh200/check-home.sh` exits 1 for
one file, `/home/ucahtbe/.bash_history`, last written on 2026-10-07 at 13:20, before this
task (T8's check listed it too); nothing of T9's runs went into the home directory.

## Raw output

The generated Markdown, as printed, with locust's paths shortened to `L`
(= `/data/ucahtbe/logs/t9`). Headings are demoted by three levels.

### CUDA, before

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 06:11:09 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 06:11:09 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | efdcb8d-dirty on tbetcke/phase4s_t9 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8 --table-cache L/tables --output L/before/cuda.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Dense | 1.039 | 12.094 | 12.263 | 12.298 | 12.680 | 0.168 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 8 | Dense | 1.206 | 21.004 | 21.450 | 21.431 | 21.993 | 0.293 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f32 | 10000000 | 3 | Dense | 14.484 | 219.060 | 221.858 | 221.340 | 223.124 | 1.545 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f32 | 10000000 | 8 | Dense | 14.281 | 312.465 | 315.056 | 314.643 | 316.978 | 1.572 | 1.21e-5 | 2.26e-5 | yes |
| cuda | f64 | 1000000 | 3 | Dense | 0.984 | 19.105 | 20.012 | 20.100 | 21.739 | 0.671 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 8 | Dense | 1.007 | 34.054 | 35.450 | 35.430 | 35.997 | 0.538 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 10000000 | 3 | Dense | 14.540 | 354.020 | 358.357 | 357.570 | 359.622 | 2.105 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f64 | 10000000 | 8 | Dense | 13.945 | 519.533 | 523.265 | 523.869 | 531.207 | 3.757 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.055 | 0.192 | 1.282 | 0.291 | 0.000 | 0.000 | 0.064 | 1.997 | 3.165 | 5.233 | 0.026 | 3.881 | 12.563 | 0.316 |
| cuda | f32 | 1000000 | 8 | 0.114 | 0.475 | 9.382 | 0.636 | 0.000 | 0.000 | 0.434 | 2.138 | 3.242 | 5.263 | 0.027 | 13.179 | 22.225 | 0.615 |
| cuda | f32 | 10000000 | 3 | 0.394 | 0.349 | 9.725 | 0.611 | 0.072 | 0.239 | 0.403 | 24.364 | 63.184 | 119.579 | 0.190 | 36.156 | 222.409 | 0.163 |
| cuda | f32 | 10000000 | 8 | 0.937 | 1.037 | 96.316 | 1.353 | 0.073 | 0.353 | 3.176 | 24.321 | 62.886 | 120.315 | 0.204 | 127.566 | 316.849 | 0.405 |
| cuda | f64 | 1000000 | 3 | 0.084 | 0.207 | 1.646 | 0.263 | 0.000 | 0.000 | 0.189 | 3.147 | 5.069 | 7.869 | 0.027 | 5.536 | 19.172 | 0.275 |
| cuda | f64 | 1000000 | 8 | 0.269 | 0.617 | 15.555 | 0.466 | 0.000 | 0.000 | 0.765 | 3.231 | 5.200 | 7.960 | 0.032 | 20.902 | 35.357 | 0.590 |
| cuda | f64 | 10000000 | 3 | 0.649 | 0.527 | 14.088 | 0.551 | 0.079 | 0.499 | 0.658 | 38.955 | 92.486 | 203.785 | 0.199 | 56.005 | 359.050 | 0.157 |
| cuda | f64 | 10000000 | 8 | 2.422 | 1.655 | 166.247 | 1.574 | 0.080 | 0.592 | 5.774 | 38.867 | 92.908 | 203.582 | 0.195 | 217.211 | 525.649 | 0.415 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.039 s, of it tables 0.001 s (loaded from the cache) and device 0.246 s.
- Evaluation over 10 repeats, ms: min 12.094, median 12.263, mean 12.298, max 12.680, std 0.168; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.054 | 0.055 | 0.056 | 1.4% |
| M2M | device | 5 | 0.190 | 0.192 | 0.196 | 4.9% |
| M2L | device | 4 | 1.279 | 1.282 | 1.286 | 33.0% |
| L2L | device | 5 | 0.286 | 0.291 | 0.301 | 7.5% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.063 | 0.064 | 0.067 | 1.7% |
| P2P | device | 1 | 1.980 | 1.997 | 2.028 | 51.4% |
| load | | | 2.997 | 3.165 | 3.265 | |
| output | | | 5.058 | 5.233 | 5.418 | |
| other | | | | 0.026 | | |
| sum | | | | 3.881 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.185 ms on average.

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.206 s, of it tables 0.010 s (loaded from the cache) and device 0.402 s.
- Evaluation over 10 repeats, ms: min 21.004, median 21.450, mean 21.431, max 21.993, std 0.293; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.114 | 0.114 | 0.115 | 0.9% |
| M2M | device | 5 | 0.315 | 0.475 | 0.519 | 3.6% |
| M2L | device | 4 | 9.374 | 9.382 | 9.390 | 71.2% |
| L2L | device | 5 | 0.628 | 0.636 | 0.652 | 4.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.426 | 0.434 | 0.444 | 3.3% |
| P2P | device | 1 | 2.126 | 2.138 | 2.153 | 16.2% |
| load | | | 3.137 | 3.242 | 3.351 | |
| output | | | 5.072 | 5.263 | 5.382 | |
| other | | | | 0.027 | | |
| sum | | | | 13.179 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.195 ms on average.

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.484 s, of it tables 0.001 s (loaded from the cache) and device 2.158 s.
- Evaluation over 10 repeats, ms: min 219.060, median 221.858, mean 221.340, max 223.124, std 1.545; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 3834.6 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 75, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 3 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.389 | 0.394 | 0.403 | 1.1% |
| M2M | device | 7 | 0.339 | 0.349 | 0.355 | 1.0% |
| M2L | device | 5 | 9.710 | 9.725 | 9.743 | 26.9% |
| L2L | device | 7 | 0.596 | 0.611 | 0.625 | 1.7% |
| P2L | device | 1 | 0.071 | 0.072 | 0.073 | 0.2% |
| M2P | device | 1 | 0.230 | 0.239 | 0.241 | 0.7% |
| L2P | device | 2 | 0.371 | 0.403 | 0.444 | 1.1% |
| P2P | device | 2 | 24.335 | 24.364 | 24.383 | 67.4% |
| load | | | 61.693 | 63.184 | 63.795 | |
| output | | | 116.769 | 119.579 | 122.106 | |
| other | | | | 0.190 | | |
| sum | | | | 36.156 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.004 ms on average.

##### cuda, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.281 s, of it tables 0.010 s (loaded from the cache) and device 2.146 s.
- Evaluation over 10 repeats, ms: min 312.465, median 315.056, mean 314.643, max 316.978, std 1.572; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 4021.0 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 111, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 32768, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

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
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 32768, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 2 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 14 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.933 | 0.937 | 0.942 | 0.7% |
| M2M | device | 7 | 1.027 | 1.037 | 1.044 | 0.8% |
| M2L | device | 5 | 96.268 | 96.316 | 96.366 | 75.5% |
| L2L | device | 7 | 1.306 | 1.353 | 1.404 | 1.1% |
| P2L | device | 1 | 0.073 | 0.073 | 0.075 | 0.1% |
| M2P | device | 1 | 0.343 | 0.353 | 0.362 | 0.3% |
| L2P | device | 2 | 3.137 | 3.176 | 3.308 | 2.5% |
| P2P | device | 2 | 24.272 | 24.321 | 24.375 | 19.1% |
| load | | | 61.790 | 62.886 | 63.817 | |
| output | | | 117.499 | 120.315 | 121.805 | |
| other | | | | 0.204 | | |
| sum | | | | 127.566 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.060 ms on average.

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.984 s, of it tables 0.001 s (loaded from the cache) and device 0.200 s.
- Evaluation over 10 repeats, ms: min 19.105, median 20.012, mean 20.100, max 21.739, std 0.671; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.083 | 0.084 | 0.085 | 1.5% |
| M2M | device | 5 | 0.201 | 0.207 | 0.211 | 3.7% |
| M2L | device | 4 | 1.641 | 1.646 | 1.654 | 29.7% |
| L2L | device | 5 | 0.258 | 0.263 | 0.268 | 4.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.183 | 0.189 | 0.192 | 3.4% |
| P2P | device | 1 | 3.112 | 3.147 | 3.176 | 56.8% |
| load | | | 4.797 | 5.069 | 5.177 | |
| output | | | 7.545 | 7.869 | 8.147 | |
| other | | | | 0.027 | | |
| sum | | | | 5.536 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.191 ms on average.

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.007 s, of it tables 0.020 s (loaded from the cache) and device 0.199 s.
- Evaluation over 10 repeats, ms: min 34.054, median 35.450, mean 35.430, max 35.997, std 0.538; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.268 | 0.269 | 0.270 | 1.3% |
| M2M | device | 5 | 0.606 | 0.617 | 0.628 | 3.0% |
| M2L | device | 4 | 15.546 | 15.555 | 15.561 | 74.4% |
| L2L | device | 5 | 0.455 | 0.466 | 0.477 | 2.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.751 | 0.765 | 0.787 | 3.7% |
| P2P | device | 1 | 3.185 | 3.231 | 3.279 | 15.5% |
| load | | | 5.128 | 5.200 | 5.309 | |
| output | | | 7.609 | 7.960 | 8.157 | |
| other | | | | 0.032 | | |
| sum | | | | 20.902 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.201 ms on average.

##### cuda, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.540 s, of it tables 0.001 s (loaded from the cache) and device 2.202 s.
- Evaluation over 10 repeats, ms: min 354.020, median 358.357, mean 357.570, max 359.622, std 2.105; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 4353.3 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 84, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 6 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.643 | 0.649 | 0.655 | 1.2% |
| M2M | device | 7 | 0.516 | 0.527 | 0.531 | 0.9% |
| M2L | device | 5 | 14.078 | 14.088 | 14.098 | 25.2% |
| L2L | device | 7 | 0.534 | 0.551 | 0.564 | 1.0% |
| P2L | device | 1 | 0.078 | 0.079 | 0.081 | 0.1% |
| M2P | device | 1 | 0.490 | 0.499 | 0.502 | 0.9% |
| L2P | device | 2 | 0.626 | 0.658 | 0.688 | 1.2% |
| P2P | device | 2 | 38.927 | 38.955 | 38.986 | 69.6% |
| load | | | 91.237 | 92.486 | 93.328 | |
| output | | | 199.665 | 203.785 | 207.611 | |
| other | | | | 0.199 | | |
| sum | | | | 56.005 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.099 ms on average.

##### cuda, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 13.945 s, of it tables 0.019 s (loaded from the cache) and device 2.197 s.
- Evaluation over 10 repeats, ms: min 519.533, median 523.265, mean 523.869, max 531.207, std 3.757; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 4701.5 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 159, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 4 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 28 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 2.375 | 2.422 | 2.584 | 1.1% |
| M2M | device | 7 | 1.645 | 1.655 | 1.660 | 0.8% |
| M2L | device | 5 | 166.191 | 166.247 | 166.397 | 76.5% |
| L2L | device | 7 | 1.548 | 1.574 | 1.654 | 0.7% |
| P2L | device | 1 | 0.079 | 0.080 | 0.083 | 0.0% |
| M2P | device | 1 | 0.559 | 0.592 | 0.607 | 0.3% |
| L2P | device | 2 | 5.695 | 5.774 | 5.898 | 2.7% |
| P2P | device | 2 | 38.804 | 38.867 | 38.950 | 17.9% |
| load | | | 91.529 | 92.908 | 93.510 | |
| output | | | 199.699 | 203.582 | 205.787 | |
| other | | | | 0.195 | | |
| sum | | | | 217.211 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.101 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### host, 72 threads, before

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 06:17:26 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 06:17:26 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | none asked for (host only) |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL not compiled in |
| compiled backends | host |
| source revision | efdcb8d-dirty on tbetcke/phase4s_t9 (.source-revision) |
| command | `nd-fmm-bench --backend host --threads 72 --n 1e6,1e7 --precision f32,f64 --degree 3,8 --table-cache L/tables --output L/before/host.md` |
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
| host | f32 | 1000000 | 3 | Dense | 0.795 | 26.672 | 27.386 | 27.397 | 28.035 | 0.382 | 2.00e-3 | 3.80e-3 | yes |
| host | f32 | 1000000 | 8 | Dense | 0.801 | 109.306 | 110.101 | 110.118 | 110.731 | 0.445 | 1.40e-5 | 4.59e-5 | yes |
| host | f32 | 10000000 | 3 | Dense | 12.230 | 337.873 | 339.762 | 339.928 | 342.169 | 1.657 | 1.87e-3 | 1.87e-3 | yes |
| host | f32 | 10000000 | 8 | Dense | 12.232 | 1077.712 | 1082.362 | 1085.577 | 1119.934 | 12.279 | 1.26e-5 | 2.26e-5 | yes |
| host | f64 | 1000000 | 3 | Dense | 0.765 | 51.176 | 52.639 | 52.567 | 53.142 | 0.547 | 2.00e-3 | 3.80e-3 | yes |
| host | f64 | 1000000 | 8 | Dense | 0.793 | 216.178 | 217.251 | 217.421 | 219.516 | 0.907 | 1.36e-5 | 4.59e-5 | yes |
| host | f64 | 10000000 | 3 | Dense | 12.248 | 683.876 | 686.830 | 690.325 | 727.264 | 13.034 | 1.87e-3 | 1.87e-3 | yes |
| host | f64 | 10000000 | 8 | Dense | 12.119 | 2106.817 | 2111.616 | 2121.746 | 2181.995 | 25.102 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f32 | 1000000 | 3 | 0.815 | 0.372 | 4.328 | 0.680 | 0.000 | 0.000 | 1.446 | 9.318 | 3.192 | 5.044 | 1.859 | 16.960 | 27.345 | 0.619 |
| host | f32 | 1000000 | 8 | 2.837 | 0.611 | 77.608 | 0.987 | 0.000 | 0.000 | 5.766 | 9.255 | 3.156 | 6.903 | 1.956 | 97.063 | 109.602 | 0.881 |
| host | f32 | 10000000 | 3 | 5.129 | 1.315 | 35.108 | 1.338 | 0.107 | 0.814 | 12.784 | 113.100 | 51.699 | 110.554 | 3.287 | 169.695 | 338.526 | 0.499 |
| host | f32 | 10000000 | 8 | 29.042 | 3.687 | 677.978 | 3.926 | 0.338 | 0.789 | 55.225 | 113.609 | 57.304 | 137.243 | 3.653 | 884.593 | 1088.716 | 0.815 |
| host | f64 | 1000000 | 3 | 0.799 | 0.396 | 5.858 | 0.694 | 0.000 | 0.000 | 1.524 | 26.833 | 4.584 | 10.063 | 2.056 | 36.103 | 53.384 | 0.687 |
| host | f64 | 1000000 | 8 | 3.048 | 0.930 | 158.164 | 1.244 | 0.000 | 0.000 | 5.870 | 26.828 | 4.911 | 12.814 | 1.921 | 196.085 | 216.908 | 0.902 |
| host | f64 | 10000000 | 3 | 5.012 | 1.392 | 49.096 | 1.371 | 0.104 | 0.708 | 13.073 | 339.519 | 73.982 | 197.310 | 3.293 | 410.273 | 691.555 | 0.594 |
| host | f64 | 10000000 | 8 | 24.817 | 6.484 | 1370.076 | 6.494 | 0.325 | 0.824 | 56.270 | 342.790 | 78.661 | 207.808 | 3.538 | 1808.081 | 2110.264 | 0.852 |

##### host, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.795 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 26.672, median 27.386, mean 27.397, max 28.035, std 0.382; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.773 | 0.815 | 0.878 | 4.8% |
| M2M | host | 5 | 0.339 | 0.372 | 0.408 | 2.2% |
| M2L | host | 4 | 4.284 | 4.328 | 4.389 | 25.5% |
| L2L | host | 5 | 0.626 | 0.680 | 0.777 | 4.0% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 1.418 | 1.446 | 1.472 | 8.5% |
| P2P | host | 1 | 9.234 | 9.318 | 9.669 | 54.9% |
| load | | | 3.117 | 3.192 | 3.340 | |
| output | | | 4.951 | 5.044 | 5.221 | |
| other | | | | 1.859 | | |
| sum | | | | 16.960 | | 100.0% |

##### host, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.801 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 109.306, median 110.101, mean 110.118, max 110.731, std 0.445; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.40e-5, ∇φ 4.59e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 2.738 | 2.837 | 2.914 | 2.9% |
| M2M | host | 5 | 0.584 | 0.611 | 0.635 | 0.6% |
| M2L | host | 4 | 77.253 | 77.608 | 78.228 | 80.0% |
| L2L | host | 5 | 0.944 | 0.987 | 1.043 | 1.0% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 5.704 | 5.766 | 5.937 | 5.9% |
| P2P | host | 1 | 9.233 | 9.255 | 9.343 | 9.5% |
| load | | | 3.065 | 3.156 | 3.241 | |
| output | | | 6.777 | 6.903 | 7.015 | |
| other | | | | 1.956 | | |
| sum | | | | 97.063 | | 100.0% |

##### host, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.230 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 337.873, median 339.762, mean 339.928, max 342.169, std 1.657; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 4.998 | 5.129 | 5.372 | 3.0% |
| M2M | host | 7 | 1.178 | 1.315 | 1.644 | 0.8% |
| M2L | host | 5 | 34.909 | 35.108 | 36.007 | 20.7% |
| L2L | host | 7 | 1.259 | 1.338 | 1.449 | 0.8% |
| P2L | host | 1 | 0.091 | 0.107 | 0.123 | 0.1% |
| M2P | host | 1 | 0.630 | 0.814 | 1.938 | 0.5% |
| L2P | host | 2 | 12.587 | 12.784 | 13.878 | 7.5% |
| P2P | host | 2 | 112.840 | 113.100 | 113.304 | 66.6% |
| load | | | 50.948 | 51.699 | 52.477 | |
| output | | | 109.331 | 110.554 | 111.511 | |
| other | | | | 3.287 | | |
| sum | | | | 169.695 | | 100.0% |

##### host, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.232 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 1077.712, median 1082.362, mean 1085.577, max 1119.934, std 12.279; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.26e-5, ∇φ 2.26e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 24.420 | 29.042 | 68.619 | 3.3% |
| M2M | host | 7 | 3.618 | 3.687 | 3.780 | 0.4% |
| M2L | host | 5 | 674.733 | 677.978 | 692.800 | 76.6% |
| L2L | host | 7 | 3.866 | 3.926 | 3.984 | 0.4% |
| P2L | host | 1 | 0.299 | 0.338 | 0.414 | 0.0% |
| M2P | host | 1 | 0.752 | 0.789 | 0.832 | 0.1% |
| L2P | host | 2 | 55.026 | 55.225 | 55.917 | 6.2% |
| P2P | host | 2 | 113.170 | 113.609 | 114.060 | 12.8% |
| load | | | 56.454 | 57.304 | 58.068 | |
| output | | | 135.118 | 137.243 | 138.799 | |
| other | | | | 3.653 | | |
| sum | | | | 884.593 | | 100.0% |

##### host, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.765 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 51.176, median 52.639, mean 52.567, max 53.142, std 0.547; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.757 | 0.799 | 0.858 | 2.2% |
| M2M | host | 5 | 0.360 | 0.396 | 0.474 | 1.1% |
| M2L | host | 4 | 5.794 | 5.858 | 5.991 | 16.2% |
| L2L | host | 5 | 0.636 | 0.694 | 0.758 | 1.9% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 1.462 | 1.524 | 1.801 | 4.2% |
| P2P | host | 1 | 26.673 | 26.833 | 26.895 | 74.3% |
| load | | | 4.463 | 4.584 | 4.640 | |
| output | | | 9.809 | 10.063 | 10.405 | |
| other | | | | 2.056 | | |
| sum | | | | 36.103 | | 100.0% |

##### host, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.793 s, of it tables 0.019 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 216.178, median 217.251, mean 217.421, max 219.516, std 0.907; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 2.787 | 3.048 | 4.217 | 1.6% |
| M2M | host | 5 | 0.904 | 0.930 | 1.050 | 0.5% |
| M2L | host | 4 | 157.887 | 158.164 | 158.890 | 80.7% |
| L2L | host | 5 | 1.208 | 1.244 | 1.265 | 0.6% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 5.695 | 5.870 | 6.157 | 3.0% |
| P2P | host | 1 | 26.786 | 26.828 | 26.887 | 13.7% |
| load | | | 4.838 | 4.911 | 5.031 | |
| output | | | 12.511 | 12.814 | 13.057 | |
| other | | | | 1.921 | | |
| sum | | | | 196.085 | | 100.0% |

##### host, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.248 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 683.876, median 686.830, mean 690.325, max 727.264, std 13.034; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 4.786 | 5.012 | 5.309 | 1.2% |
| M2M | host | 7 | 1.295 | 1.392 | 1.850 | 0.3% |
| M2L | host | 5 | 48.882 | 49.096 | 49.750 | 12.0% |
| L2L | host | 7 | 1.305 | 1.371 | 1.443 | 0.3% |
| P2L | host | 1 | 0.093 | 0.104 | 0.114 | 0.0% |
| M2P | host | 1 | 0.670 | 0.708 | 0.752 | 0.2% |
| L2P | host | 2 | 12.949 | 13.073 | 13.298 | 3.2% |
| P2P | host | 2 | 339.065 | 339.519 | 340.142 | 82.8% |
| load | | | 72.704 | 73.982 | 78.134 | |
| output | | | 190.368 | 197.310 | 235.616 | |
| other | | | | 3.293 | | |
| sum | | | | 410.273 | | 100.0% |

##### host, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.119 s, of it tables 0.019 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 2106.817, median 2111.616, mean 2121.746, max 2181.995, std 25.102; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 24.335 | 24.817 | 26.611 | 1.4% |
| M2M | host | 7 | 6.315 | 6.484 | 6.847 | 0.4% |
| M2L | host | 5 | 1367.035 | 1370.076 | 1372.372 | 75.8% |
| L2L | host | 7 | 6.356 | 6.494 | 6.893 | 0.4% |
| P2L | host | 1 | 0.291 | 0.325 | 0.373 | 0.0% |
| M2P | host | 1 | 0.779 | 0.824 | 0.876 | 0.0% |
| L2P | host | 2 | 55.877 | 56.270 | 57.111 | 3.1% |
| P2P | host | 2 | 341.772 | 342.790 | 347.369 | 19.0% |
| load | | | 77.765 | 78.661 | 79.960 | |
| output | | | 205.111 | 207.808 | 210.190 | |
| other | | | | 3.538 | | |
| sum | | | | 1808.081 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### CUDA, after

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 06:42:38 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 06:42:38 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | efdcb8d-dirty on tbetcke/phase4s_t9 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8 --table-cache L/tables --output L/after/cuda.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Dense | 1.086 | 6.394 | 6.655 | 6.630 | 6.748 | 0.111 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 8 | Dense | 1.211 | 15.297 | 15.646 | 15.619 | 15.838 | 0.158 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f32 | 10000000 | 3 | Dense | 14.287 | 59.905 | 60.334 | 60.345 | 60.753 | 0.244 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f32 | 10000000 | 8 | Dense | 14.863 | 153.255 | 153.919 | 153.952 | 154.448 | 0.382 | 1.21e-5 | 2.26e-5 | yes |
| cuda | f64 | 1000000 | 3 | Dense | 1.018 | 9.746 | 10.236 | 10.194 | 10.496 | 0.194 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 8 | Dense | 1.024 | 25.476 | 25.723 | 25.790 | 26.575 | 0.315 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 10000000 | 3 | Dense | 14.427 | 102.245 | 102.990 | 102.986 | 103.943 | 0.473 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f64 | 10000000 | 8 | Dense | 14.359 | 268.121 | 268.820 | 268.818 | 269.230 | 0.326 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.062 | 0.217 | 1.288 | 0.300 | 0.000 | 0.000 | 0.067 | 1.998 | 0.716 | 2.327 | 0.030 | 3.934 | 7.233 | 0.593 |
| cuda | f32 | 1000000 | 8 | 0.121 | 0.523 | 9.381 | 0.640 | 0.000 | 0.000 | 0.432 | 2.144 | 0.705 | 2.395 | 0.032 | 13.241 | 16.821 | 0.848 |
| cuda | f32 | 10000000 | 3 | 0.403 | 0.446 | 9.732 | 0.599 | 0.074 | 0.233 | 0.397 | 24.382 | 6.615 | 15.393 | 0.210 | 36.265 | 61.767 | 0.601 |
| cuda | f32 | 10000000 | 8 | 1.137 | 1.059 | 96.362 | 1.337 | 0.076 | 0.348 | 3.175 | 24.313 | 6.666 | 15.322 | 0.214 | 127.806 | 155.842 | 0.830 |
| cuda | f64 | 1000000 | 3 | 0.093 | 0.225 | 1.648 | 0.270 | 0.000 | 0.000 | 0.189 | 3.140 | 1.170 | 3.265 | 0.032 | 5.565 | 10.598 | 0.546 |
| cuda | f64 | 1000000 | 8 | 0.275 | 0.622 | 15.555 | 0.468 | 0.000 | 0.000 | 0.769 | 3.221 | 1.251 | 3.308 | 0.029 | 20.912 | 26.588 | 0.811 |
| cuda | f64 | 10000000 | 3 | 0.664 | 0.537 | 14.104 | 0.563 | 0.080 | 0.491 | 0.714 | 38.988 | 13.583 | 28.567 | 0.208 | 56.141 | 105.013 | 0.545 |
| cuda | f64 | 10000000 | 8 | 2.544 | 1.676 | 166.228 | 1.597 | 0.083 | 0.597 | 5.773 | 38.848 | 13.295 | 32.816 | 0.212 | 217.345 | 275.254 | 0.809 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.086 s, of it tables 0.001 s (loaded from the cache) and device 0.268 s.
- Evaluation over 10 repeats, ms: min 6.394, median 6.655, mean 6.630, max 6.748, std 0.111; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 903.6 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 50, syncs 1, timing windows 0.
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
| P2M | device | 1 | 0.060 | 0.062 | 0.064 | 1.6% |
| M2M | device | 5 | 0.213 | 0.217 | 0.224 | 5.5% |
| M2L | device | 4 | 1.285 | 1.288 | 1.293 | 32.8% |
| L2L | device | 5 | 0.295 | 0.300 | 0.305 | 7.6% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.064 | 0.067 | 0.071 | 1.7% |
| P2P | device | 1 | 1.982 | 1.998 | 2.022 | 50.8% |
| load | | | 0.623 | 0.716 | 0.864 | |
| output | | | 2.224 | 2.327 | 2.547 | |
| other | | | | 0.030 | | |
| sum | | | | 3.934 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.247 ms on average.

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.211 s, of it tables 0.010 s (loaded from the cache) and device 0.407 s.
- Evaluation over 10 repeats, ms: min 15.297, median 15.646, mean 15.619, max 15.838, std 0.158; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2390.4 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 53, syncs 1, timing windows 0.
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
| P2M | device | 1 | 0.119 | 0.121 | 0.123 | 0.9% |
| M2M | device | 5 | 0.518 | 0.523 | 0.527 | 3.9% |
| M2L | device | 4 | 9.374 | 9.381 | 9.389 | 70.8% |
| L2L | device | 5 | 0.633 | 0.640 | 0.653 | 4.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.425 | 0.432 | 0.438 | 3.3% |
| P2P | device | 1 | 2.119 | 2.144 | 2.180 | 16.2% |
| load | | | 0.581 | 0.705 | 0.774 | |
| output | | | 2.225 | 2.395 | 2.801 | |
| other | | | | 0.032 | | |
| sum | | | | 13.241 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.264 ms on average.

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.287 s, of it tables 0.001 s (loaded from the cache) and device 2.228 s.
- Evaluation over 10 repeats, ms: min 59.905, median 60.334, mean 60.345, max 60.753, std 0.244; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 4079.8 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 76, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 3 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.393 | 0.403 | 0.411 | 1.1% |
| M2M | device | 7 | 0.361 | 0.446 | 0.572 | 1.2% |
| M2L | device | 5 | 9.713 | 9.732 | 9.750 | 26.8% |
| L2L | device | 7 | 0.443 | 0.599 | 0.638 | 1.7% |
| P2L | device | 1 | 0.073 | 0.074 | 0.075 | 0.2% |
| M2P | device | 1 | 0.231 | 0.233 | 0.236 | 0.6% |
| L2P | device | 2 | 0.348 | 0.397 | 0.415 | 1.1% |
| P2P | device | 2 | 24.355 | 24.382 | 24.421 | 67.2% |
| load | | | 6.432 | 6.615 | 6.773 | |
| output | | | 14.999 | 15.393 | 15.807 | |
| other | | | | 0.210 | | |
| sum | | | | 36.265 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.685 ms on average.

##### cuda, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.863 s, of it tables 0.010 s (loaded from the cache) and device 2.218 s.
- Evaluation over 10 repeats, ms: min 153.255, median 153.919, mean 153.952, max 154.448, std 0.382; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 4266.3 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 112, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 32768, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

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
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 32768, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 2 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 14 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.955 | 1.137 | 1.166 | 0.9% |
| M2M | device | 7 | 1.054 | 1.059 | 1.065 | 0.8% |
| M2L | device | 5 | 96.313 | 96.362 | 96.413 | 75.4% |
| L2L | device | 7 | 1.253 | 1.337 | 1.479 | 1.0% |
| P2L | device | 1 | 0.075 | 0.076 | 0.077 | 0.1% |
| M2P | device | 1 | 0.341 | 0.348 | 0.356 | 0.3% |
| L2P | device | 2 | 3.152 | 3.175 | 3.194 | 2.5% |
| P2P | device | 2 | 24.298 | 24.313 | 24.332 | 19.0% |
| load | | | 6.540 | 6.666 | 6.899 | |
| output | | | 15.104 | 15.322 | 15.509 | |
| other | | | | 0.214 | | |
| sum | | | | 127.806 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.660 ms on average.

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.018 s, of it tables 0.001 s (loaded from the cache) and device 0.211 s.
- Evaluation over 10 repeats, ms: min 9.746, median 10.236, mean 10.194, max 10.496, std 0.194; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 1663.8 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 50, syncs 1, timing windows 0.
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
| P2M | device | 1 | 0.090 | 0.093 | 0.099 | 1.7% |
| M2M | device | 5 | 0.220 | 0.225 | 0.236 | 4.1% |
| M2L | device | 4 | 1.640 | 1.648 | 1.654 | 29.6% |
| L2L | device | 5 | 0.266 | 0.270 | 0.279 | 4.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.183 | 0.189 | 0.193 | 3.4% |
| P2P | device | 1 | 3.098 | 3.140 | 3.182 | 56.4% |
| load | | | 1.100 | 1.170 | 1.302 | |
| output | | | 3.036 | 3.265 | 3.406 | |
| other | | | | 0.032 | | |
| sum | | | | 5.565 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.252 ms on average.

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.024 s, of it tables 0.020 s (loaded from the cache) and device 0.210 s.
- Evaluation over 10 repeats, ms: min 25.476, median 25.723, mean 25.790, max 26.575, std 0.315; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2487.1 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 59, syncs 1, timing windows 0.
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
| P2M | device | 1 | 0.273 | 0.275 | 0.278 | 1.3% |
| M2M | device | 5 | 0.618 | 0.622 | 0.628 | 3.0% |
| M2L | device | 4 | 15.544 | 15.555 | 15.566 | 74.4% |
| L2L | device | 5 | 0.462 | 0.468 | 0.474 | 2.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.763 | 0.769 | 0.778 | 3.7% |
| P2P | device | 1 | 3.191 | 3.221 | 3.254 | 15.4% |
| load | | | 1.213 | 1.251 | 1.285 | |
| output | | | 3.197 | 3.308 | 3.478 | |
| other | | | | 0.029 | | |
| sum | | | | 20.912 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.276 ms on average.

##### cuda, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.427 s, of it tables 0.001 s (loaded from the cache) and device 2.273 s.
- Evaluation over 10 repeats, ms: min 102.245, median 102.990, mean 102.986, max 103.943, std 0.473; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 4758.5 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 85, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 6 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.660 | 0.664 | 0.672 | 1.2% |
| M2M | device | 7 | 0.532 | 0.537 | 0.541 | 1.0% |
| M2L | device | 5 | 14.088 | 14.104 | 14.116 | 25.1% |
| L2L | device | 7 | 0.525 | 0.563 | 0.727 | 1.0% |
| P2L | device | 1 | 0.079 | 0.080 | 0.081 | 0.1% |
| M2P | device | 1 | 0.489 | 0.491 | 0.495 | 0.9% |
| L2P | device | 2 | 0.624 | 0.714 | 0.828 | 1.3% |
| P2P | device | 2 | 38.947 | 38.988 | 39.011 | 69.4% |
| load | | | 13.146 | 13.583 | 13.801 | |
| output | | | 28.241 | 28.567 | 29.082 | |
| other | | | | 0.208 | | |
| sum | | | | 56.141 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.610 ms on average.

##### cuda, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.359 s, of it tables 0.019 s (loaded from the cache) and device 2.278 s.
- Evaluation over 10 repeats, ms: min 268.121, median 268.820, mean 268.818, max 269.230, std 0.326; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 5106.8 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 160, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 4 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 28 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 2.367 | 2.544 | 2.592 | 1.2% |
| M2M | device | 7 | 1.655 | 1.676 | 1.683 | 0.8% |
| M2L | device | 5 | 166.199 | 166.228 | 166.299 | 76.5% |
| L2L | device | 7 | 1.520 | 1.597 | 1.702 | 0.7% |
| P2L | device | 1 | 0.081 | 0.083 | 0.089 | 0.0% |
| M2P | device | 1 | 0.586 | 0.597 | 0.602 | 0.3% |
| L2P | device | 2 | 5.702 | 5.773 | 5.903 | 2.7% |
| P2P | device | 2 | 38.827 | 38.848 | 38.869 | 17.9% |
| load | | | 12.954 | 13.295 | 13.535 | |
| output | | | 28.200 | 32.816 | 68.658 | |
| other | | | | 0.212 | | |
| sum | | | | 217.345 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.806 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### CUDA, after, `--reuse-output`

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 06:48:22 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 06:48:22 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | efdcb8d-dirty on tbetcke/phase4s_t9 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8 --reuse-output --table-cache L/tables --output L/after/cuda-reuse.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, `--reuse-output`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Dense | 1.062 | 6.389 | 6.572 | 6.615 | 6.943 | 0.156 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 8 | Dense | 1.236 | 15.373 | 15.558 | 15.597 | 15.858 | 0.164 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f32 | 10000000 | 3 | Dense | 14.688 | 59.307 | 59.504 | 59.707 | 60.535 | 0.396 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f32 | 10000000 | 8 | Dense | 14.677 | 153.209 | 153.394 | 153.461 | 154.031 | 0.240 | 1.21e-5 | 2.26e-5 | yes |
| cuda | f64 | 1000000 | 3 | Dense | 1.000 | 10.145 | 10.226 | 10.214 | 10.257 | 0.040 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 8 | Dense | 1.017 | 25.457 | 25.643 | 25.625 | 25.761 | 0.107 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 10000000 | 3 | Dense | 14.893 | 102.007 | 102.886 | 102.830 | 103.710 | 0.445 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f64 | 10000000 | 8 | Dense | 14.542 | 267.815 | 268.651 | 274.764 | 329.598 | 19.271 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.061 | 0.202 | 1.281 | 0.293 | 0.000 | 0.000 | 0.067 | 1.995 | 0.722 | 2.242 | 0.032 | 3.899 | 7.119 | 0.589 |
| cuda | f32 | 1000000 | 8 | 0.122 | 0.524 | 9.375 | 0.643 | 0.000 | 0.000 | 0.436 | 2.146 | 0.688 | 2.337 | 0.034 | 13.246 | 16.767 | 0.849 |
| cuda | f32 | 10000000 | 3 | 0.403 | 0.444 | 9.737 | 0.617 | 0.074 | 0.232 | 0.409 | 24.383 | 6.414 | 14.822 | 0.208 | 36.299 | 61.014 | 0.608 |
| cuda | f32 | 10000000 | 8 | 1.114 | 1.056 | 96.295 | 1.384 | 0.075 | 0.349 | 3.184 | 24.298 | 6.461 | 14.908 | 0.213 | 127.755 | 155.206 | 0.832 |
| cuda | f64 | 1000000 | 3 | 0.092 | 0.219 | 1.648 | 0.266 | 0.000 | 0.000 | 0.187 | 3.122 | 1.190 | 3.325 | 0.032 | 5.534 | 10.648 | 0.542 |
| cuda | f64 | 1000000 | 8 | 0.274 | 0.625 | 15.555 | 0.467 | 0.000 | 0.000 | 0.773 | 3.232 | 1.097 | 3.446 | 0.033 | 20.926 | 26.667 | 0.817 |
| cuda | f64 | 10000000 | 3 | 0.664 | 0.541 | 14.100 | 0.550 | 0.081 | 0.492 | 0.689 | 38.972 | 13.271 | 28.234 | 0.209 | 56.089 | 104.370 | 0.545 |
| cuda | f64 | 10000000 | 8 | 2.585 | 1.673 | 166.244 | 1.601 | 0.081 | 0.595 | 5.743 | 38.834 | 17.519 | 28.145 | 0.213 | 217.355 | 274.884 | 0.791 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.062 s, of it tables 0.000 s (loaded from the cache) and device 0.264 s.
- Evaluation over 10 repeats, ms: min 6.389, median 6.572, mean 6.615, max 6.943, std 0.156; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 903.6 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 50, syncs 1, timing windows 0.
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
| P2M | device | 1 | 0.056 | 0.061 | 0.065 | 1.6% |
| M2M | device | 5 | 0.176 | 0.202 | 0.230 | 5.2% |
| M2L | device | 4 | 1.268 | 1.281 | 1.291 | 32.9% |
| L2L | device | 5 | 0.281 | 0.293 | 0.302 | 7.5% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.062 | 0.067 | 0.071 | 1.7% |
| P2P | device | 1 | 1.974 | 1.995 | 2.033 | 51.2% |
| load | | | 0.632 | 0.722 | 0.814 | |
| output | | | 2.105 | 2.242 | 2.458 | |
| other | | | | 0.032 | | |
| sum | | | | 3.899 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.225 ms on average.

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.236 s, of it tables 0.010 s (loaded from the cache) and device 0.404 s.
- Evaluation over 10 repeats, ms: min 15.373, median 15.558, mean 15.597, max 15.858, std 0.164; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2390.4 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 53, syncs 1, timing windows 0.
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
| P2M | device | 1 | 0.120 | 0.122 | 0.123 | 0.9% |
| M2M | device | 5 | 0.518 | 0.524 | 0.529 | 4.0% |
| M2L | device | 4 | 9.370 | 9.375 | 9.380 | 70.8% |
| L2L | device | 5 | 0.633 | 0.643 | 0.657 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.431 | 0.436 | 0.444 | 3.3% |
| P2P | device | 1 | 2.125 | 2.146 | 2.168 | 16.2% |
| load | | | 0.555 | 0.688 | 0.734 | |
| output | | | 2.124 | 2.337 | 3.510 | |
| other | | | | 0.034 | | |
| sum | | | | 13.246 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.237 ms on average.

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.688 s, of it tables 0.001 s (loaded from the cache) and device 2.243 s.
- Evaluation over 10 repeats, ms: min 59.307, median 59.504, mean 59.707, max 60.535, std 0.396; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 4079.8 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 76, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 3 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.399 | 0.403 | 0.410 | 1.1% |
| M2M | device | 7 | 0.359 | 0.444 | 0.569 | 1.2% |
| M2L | device | 5 | 9.711 | 9.737 | 9.770 | 26.8% |
| L2L | device | 7 | 0.591 | 0.617 | 0.639 | 1.7% |
| P2L | device | 1 | 0.073 | 0.074 | 0.075 | 0.2% |
| M2P | device | 1 | 0.230 | 0.232 | 0.235 | 0.6% |
| L2P | device | 2 | 0.365 | 0.409 | 0.528 | 1.1% |
| P2P | device | 2 | 24.351 | 24.383 | 24.447 | 67.2% |
| load | | | 6.126 | 6.414 | 6.754 | |
| output | | | 14.547 | 14.822 | 15.152 | |
| other | | | | 0.208 | | |
| sum | | | | 36.299 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.661 ms on average.

##### cuda, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.677 s, of it tables 0.010 s (loaded from the cache) and device 2.180 s.
- Evaluation over 10 repeats, ms: min 153.209, median 153.394, mean 153.461, max 154.031, std 0.240; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 4266.3 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 112, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 32768, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

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
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 32768, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 2 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 14 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.954 | 1.114 | 1.159 | 0.9% |
| M2M | device | 7 | 1.050 | 1.056 | 1.062 | 0.8% |
| M2L | device | 5 | 96.283 | 96.295 | 96.314 | 75.4% |
| L2L | device | 7 | 1.356 | 1.384 | 1.415 | 1.1% |
| P2L | device | 1 | 0.075 | 0.075 | 0.076 | 0.1% |
| M2P | device | 1 | 0.345 | 0.349 | 0.358 | 0.3% |
| L2P | device | 2 | 3.160 | 3.184 | 3.209 | 2.5% |
| P2P | device | 2 | 24.280 | 24.298 | 24.322 | 19.0% |
| load | | | 6.175 | 6.461 | 6.765 | |
| output | | | 14.691 | 14.908 | 15.209 | |
| other | | | | 0.213 | | |
| sum | | | | 127.755 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.684 ms on average.

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.000 s, of it tables 0.001 s (loaded from the cache) and device 0.210 s.
- Evaluation over 10 repeats, ms: min 10.145, median 10.226, mean 10.214, max 10.257, std 0.040; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 1663.8 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 50, syncs 1, timing windows 0.
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
| P2M | device | 1 | 0.090 | 0.092 | 0.095 | 1.7% |
| M2M | device | 5 | 0.186 | 0.219 | 0.229 | 4.0% |
| M2L | device | 4 | 1.636 | 1.648 | 1.655 | 29.8% |
| L2L | device | 5 | 0.248 | 0.266 | 0.277 | 4.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.183 | 0.187 | 0.193 | 3.4% |
| P2P | device | 1 | 3.093 | 3.122 | 3.149 | 56.4% |
| load | | | 1.114 | 1.190 | 1.297 | |
| output | | | 3.243 | 3.325 | 3.466 | |
| other | | | | 0.032 | | |
| sum | | | | 5.534 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.257 ms on average.

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.017 s, of it tables 0.019 s (loaded from the cache) and device 0.208 s.
- Evaluation over 10 repeats, ms: min 25.457, median 25.643, mean 25.625, max 25.761, std 0.107; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 2487.1 MB.
- One evaluation: uploads 1 (8000000 B), downloads 1 (32000000 B), launches 59, syncs 1, timing windows 0.
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
| P2M | device | 1 | 0.271 | 0.274 | 0.276 | 1.3% |
| M2M | device | 5 | 0.618 | 0.625 | 0.638 | 3.0% |
| M2L | device | 4 | 15.542 | 15.555 | 15.565 | 74.3% |
| L2L | device | 5 | 0.461 | 0.467 | 0.471 | 2.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.762 | 0.773 | 0.796 | 3.7% |
| P2P | device | 1 | 3.210 | 3.232 | 3.253 | 15.4% |
| load | | | 0.991 | 1.097 | 1.499 | |
| output | | | 3.204 | 3.446 | 4.261 | |
| other | | | | 0.033 | | |
| sum | | | | 20.926 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.308 ms on average.

##### cuda, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.893 s, of it tables 0.001 s (loaded from the cache) and device 2.242 s.
- Evaluation over 10 repeats, ms: min 102.007, median 102.886, mean 102.830, max 103.710, std 0.445; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 4758.5 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 85, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 6 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.660 | 0.664 | 0.671 | 1.2% |
| M2M | device | 7 | 0.532 | 0.541 | 0.548 | 1.0% |
| M2L | device | 5 | 14.084 | 14.100 | 14.111 | 25.1% |
| L2L | device | 7 | 0.533 | 0.550 | 0.571 | 1.0% |
| P2L | device | 1 | 0.081 | 0.081 | 0.082 | 0.1% |
| M2P | device | 1 | 0.491 | 0.492 | 0.498 | 0.9% |
| L2P | device | 2 | 0.625 | 0.689 | 0.823 | 1.2% |
| P2P | device | 2 | 38.921 | 38.972 | 39.013 | 69.5% |
| load | | | 12.894 | 13.271 | 13.644 | |
| output | | | 27.708 | 28.234 | 29.039 | |
| other | | | | 0.209 | | |
| sum | | | | 56.089 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.824 ms on average.

##### cuda, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.542 s, of it tables 0.019 s (loaded from the cache) and device 2.246 s.
- Evaluation over 10 repeats, ms: min 267.815, median 268.651, mean 274.764, max 329.598, std 19.271; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 5106.8 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 160, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 4 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 28 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 2.579 | 2.585 | 2.598 | 1.2% |
| M2M | device | 7 | 1.667 | 1.673 | 1.690 | 0.8% |
| M2L | device | 5 | 166.227 | 166.244 | 166.294 | 76.5% |
| L2L | device | 7 | 1.502 | 1.601 | 1.696 | 0.7% |
| P2L | device | 1 | 0.080 | 0.081 | 0.083 | 0.0% |
| M2P | device | 1 | 0.591 | 0.595 | 0.604 | 0.3% |
| L2P | device | 2 | 5.693 | 5.743 | 5.903 | 2.6% |
| P2P | device | 2 | 38.803 | 38.834 | 38.863 | 17.9% |
| load | | | 12.640 | 17.519 | 55.767 | |
| output | | | 27.693 | 28.145 | 28.566 | |
| other | | | | 0.213 | | |
| sum | | | | 217.355 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.742 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate_into` (one reused `Output`), with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### CUDA, after, `--output-pass host`

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 06:54:10 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 06:54:10 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | efdcb8d-dirty on tbetcke/phase4s_t9 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e7 --precision f32,f64 --degree 3,8 --output-pass host --table-cache L/tables --output L/after/cuda-hostpass.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `host` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 10000000 | 3 | Dense | 14.747 | 64.964 | 65.595 | 65.500 | 66.085 | 0.352 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f32 | 10000000 | 8 | Dense | 14.348 | 158.811 | 159.381 | 159.484 | 160.573 | 0.441 | 1.21e-5 | 2.26e-5 | yes |
| cuda | f64 | 10000000 | 3 | Dense | 14.239 | 110.585 | 111.014 | 111.153 | 111.839 | 0.384 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f64 | 10000000 | 8 | Dense | 14.538 | 274.509 | 275.887 | 275.863 | 277.168 | 0.680 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 10000000 | 3 | 0.400 | 0.368 | 9.725 | 0.594 | 0.072 | 0.231 | 0.399 | 24.402 | 6.670 | 20.373 | 0.209 | 36.191 | 66.752 | 0.553 |
| cuda | f32 | 10000000 | 8 | 1.116 | 1.059 | 96.313 | 1.384 | 0.076 | 0.347 | 3.187 | 24.299 | 6.800 | 20.533 | 0.212 | 127.781 | 161.203 | 0.801 |
| cuda | f64 | 10000000 | 3 | 0.662 | 0.525 | 14.079 | 0.539 | 0.077 | 0.489 | 0.698 | 38.986 | 13.471 | 36.176 | 0.209 | 56.056 | 112.501 | 0.504 |
| cuda | f64 | 10000000 | 8 | 2.571 | 1.667 | 166.199 | 1.535 | 0.080 | 0.592 | 5.764 | 38.843 | 13.398 | 36.259 | 0.209 | 217.250 | 278.877 | 0.788 |

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.747 s, of it tables 0.001 s (loaded from the cache) and device 2.466 s.
- Evaluation over 10 repeats, ms: min 64.964, median 65.595, mean 65.500, max 66.085, std 0.352; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 3834.6 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 75, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 3 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.388 | 0.400 | 0.413 | 1.1% |
| M2M | device | 7 | 0.333 | 0.368 | 0.568 | 1.0% |
| M2L | device | 5 | 9.706 | 9.725 | 9.751 | 26.9% |
| L2L | device | 7 | 0.561 | 0.594 | 0.637 | 1.6% |
| P2L | device | 1 | 0.070 | 0.072 | 0.074 | 0.2% |
| M2P | device | 1 | 0.228 | 0.231 | 0.234 | 0.6% |
| L2P | device | 2 | 0.347 | 0.399 | 0.528 | 1.1% |
| P2P | device | 2 | 24.369 | 24.402 | 24.470 | 67.4% |
| load | | | 6.424 | 6.670 | 6.789 | |
| output | | | 19.871 | 20.373 | 20.682 | |
| other | | | | 0.209 | | |
| sum | | | | 36.191 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 11.807 ms on average.

##### cuda, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.348 s, of it tables 0.010 s (loaded from the cache) and device 2.141 s.
- Evaluation over 10 repeats, ms: min 158.811, median 159.381, mean 159.484, max 160.573, std 0.441; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 4021.0 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 111, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 64, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 512, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 4096, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 32768, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

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
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 32768, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk (library rejected: [8, 10, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: No tile size is available for the problem.

)
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 2 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 14 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.944 | 1.116 | 1.166 | 0.9% |
| M2M | device | 7 | 1.053 | 1.059 | 1.072 | 0.8% |
| M2L | device | 5 | 96.262 | 96.313 | 96.486 | 75.4% |
| L2L | device | 7 | 1.268 | 1.384 | 1.424 | 1.1% |
| P2L | device | 1 | 0.074 | 0.076 | 0.077 | 0.1% |
| M2P | device | 1 | 0.345 | 0.347 | 0.351 | 0.3% |
| L2P | device | 2 | 3.163 | 3.187 | 3.201 | 2.5% |
| P2P | device | 2 | 24.282 | 24.299 | 24.318 | 19.0% |
| load | | | 6.567 | 6.800 | 7.100 | |
| output | | | 20.007 | 20.533 | 21.006 | |
| other | | | | 0.212 | | |
| sum | | | | 127.781 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 11.747 ms on average.

##### cuda, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.239 s, of it tables 0.001 s (loaded from the cache) and device 2.228 s.
- Evaluation over 10 repeats, ms: min 110.585, median 111.014, mean 111.153, max 111.839, std 0.384; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 4353.3 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 84, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 6 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 0.649 | 0.662 | 0.677 | 1.2% |
| M2M | device | 7 | 0.507 | 0.525 | 0.539 | 0.9% |
| M2L | device | 5 | 14.070 | 14.079 | 14.095 | 25.1% |
| L2L | device | 7 | 0.499 | 0.539 | 0.576 | 1.0% |
| P2L | device | 1 | 0.076 | 0.077 | 0.079 | 0.1% |
| M2P | device | 1 | 0.487 | 0.489 | 0.492 | 0.9% |
| L2P | device | 2 | 0.626 | 0.698 | 0.805 | 1.2% |
| P2P | device | 2 | 38.943 | 38.986 | 39.048 | 69.5% |
| load | | | 13.268 | 13.471 | 13.983 | |
| output | | | 35.680 | 36.176 | 36.715 | |
| other | | | | 0.209 | | |
| sum | | | | 56.056 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.021 ms on average.

##### cuda, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.538 s, of it tables 0.019 s (loaded from the cache) and device 2.194 s.
- Evaluation over 10 repeats, ms: min 274.509, median 275.887, mean 275.863, max 277.168, std 0.680; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(32 x 2 units, 8 columns per unit); rotation cube (96 units); device memory 4701.5 MB.
- One evaluation: uploads 1 (80000000 B), downloads 1 (320000000 B), launches 159, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 4 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(32 x 2 units, 8 columns per unit), 28 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 2.381 | 2.571 | 2.601 | 1.2% |
| M2M | device | 7 | 1.650 | 1.667 | 1.675 | 0.8% |
| M2L | device | 5 | 166.150 | 166.199 | 166.328 | 76.5% |
| L2L | device | 7 | 1.472 | 1.535 | 1.629 | 0.7% |
| P2L | device | 1 | 0.079 | 0.080 | 0.084 | 0.0% |
| M2P | device | 1 | 0.586 | 0.592 | 0.597 | 0.3% |
| L2P | device | 2 | 5.703 | 5.764 | 5.903 | 2.7% |
| P2P | device | 2 | 38.806 | 38.843 | 38.874 | 17.9% |
| load | | | 13.213 | 13.398 | 13.635 | |
| output | | | 36.004 | 36.259 | 36.515 | |
| other | | | | 0.209 | | |
| sum | | | | 217.250 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.046 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### host, 72 threads, after

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 06:59:37 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 06:59:37 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | none asked for (host only) |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL not compiled in |
| compiled backends | host |
| source revision | efdcb8d-dirty on tbetcke/phase4s_t9 (.source-revision) |
| command | `nd-fmm-bench --backend host --threads 72 --n 1e6,1e7 --precision f32,f64 --degree 3,8 --table-cache L/tables --output L/after/host.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72; host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| host | f32 | 1000000 | 3 | Dense | 0.787 | 19.805 | 20.528 | 20.443 | 20.907 | 0.363 | 2.00e-3 | 3.80e-3 | yes |
| host | f32 | 1000000 | 8 | Dense | 0.792 | 100.698 | 101.162 | 101.173 | 101.579 | 0.255 | 1.40e-5 | 4.59e-5 | yes |
| host | f32 | 10000000 | 3 | Dense | 12.706 | 185.071 | 185.882 | 186.671 | 191.341 | 1.846 | 1.87e-3 | 1.87e-3 | yes |
| host | f32 | 10000000 | 8 | Dense | 12.509 | 894.550 | 896.876 | 903.120 | 961.020 | 20.376 | 1.26e-5 | 2.26e-5 | yes |
| host | f64 | 1000000 | 3 | Dense | 0.786 | 39.373 | 40.245 | 40.300 | 41.141 | 0.527 | 2.00e-3 | 3.80e-3 | yes |
| host | f64 | 1000000 | 8 | Dense | 0.800 | 199.898 | 200.294 | 200.920 | 206.138 | 1.874 | 1.36e-5 | 4.59e-5 | yes |
| host | f64 | 10000000 | 3 | Dense | 12.601 | 435.215 | 436.183 | 436.289 | 437.891 | 0.874 | 1.87e-3 | 1.87e-3 | yes |
| host | f64 | 10000000 | 8 | Dense | 12.202 | 1836.856 | 1841.190 | 1846.811 | 1876.574 | 13.692 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f32 | 1000000 | 3 | 0.861 | 0.342 | 4.318 | 0.630 | 0.000 | 0.000 | 1.722 | 9.383 | 0.125 | 0.945 | 1.761 | 17.255 | 20.308 | 0.844 |
| host | f32 | 1000000 | 8 | 2.668 | 0.629 | 78.083 | 1.033 | 0.000 | 0.000 | 5.872 | 9.332 | 0.134 | 1.020 | 1.951 | 97.617 | 101.240 | 0.965 |
| host | f32 | 10000000 | 3 | 4.902 | 1.313 | 35.009 | 1.456 | 0.111 | 0.744 | 12.749 | 113.364 | 1.471 | 8.519 | 3.467 | 169.648 | 186.140 | 0.909 |
| host | f32 | 10000000 | 8 | 22.989 | 3.846 | 680.523 | 4.067 | 0.290 | 0.814 | 55.223 | 113.971 | 1.516 | 12.972 | 3.855 | 881.724 | 905.628 | 0.976 |
| host | f64 | 1000000 | 3 | 0.926 | 0.378 | 6.304 | 0.712 | 0.000 | 0.000 | 1.614 | 26.949 | 0.170 | 1.097 | 2.042 | 36.884 | 40.649 | 0.915 |
| host | f64 | 1000000 | 8 | 2.780 | 0.962 | 158.369 | 1.327 | 0.000 | 0.000 | 6.166 | 26.866 | 0.188 | 1.186 | 2.026 | 196.469 | 200.985 | 0.978 |
| host | f64 | 10000000 | 3 | 5.293 | 1.392 | 50.696 | 1.391 | 0.105 | 0.677 | 13.541 | 339.609 | 2.149 | 11.858 | 3.450 | 412.704 | 436.600 | 0.946 |
| host | f64 | 10000000 | 8 | 23.574 | 6.579 | 1377.611 | 6.359 | 0.286 | 0.810 | 58.469 | 343.663 | 2.378 | 11.830 | 3.855 | 1817.351 | 1846.935 | 0.984 |

##### host, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.787 s, of it tables 0.000 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 19.805, median 20.528, mean 20.443, max 20.907, std 0.363; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.736 | 0.861 | 1.047 | 5.0% |
| M2M | host | 5 | 0.308 | 0.342 | 0.377 | 2.0% |
| M2L | host | 4 | 4.249 | 4.318 | 4.450 | 25.0% |
| L2L | host | 5 | 0.572 | 0.630 | 0.714 | 3.7% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 1.391 | 1.722 | 3.580 | 10.0% |
| P2P | host | 1 | 9.252 | 9.383 | 9.824 | 54.4% |
| load | | | 0.110 | 0.125 | 0.153 | |
| output | | | 0.829 | 0.945 | 1.050 | |
| other | | | | 1.761 | | |
| sum | | | | 17.255 | | 100.0% |

##### host, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.792 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 100.698, median 101.162, mean 101.173, max 101.579, std 0.255; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.40e-5, ∇φ 4.59e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 2.572 | 2.668 | 3.009 | 2.7% |
| M2M | host | 5 | 0.597 | 0.629 | 0.732 | 0.6% |
| M2L | host | 4 | 77.983 | 78.083 | 78.245 | 80.0% |
| L2L | host | 5 | 0.918 | 1.033 | 1.169 | 1.1% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 5.703 | 5.872 | 6.334 | 6.0% |
| P2P | host | 1 | 9.242 | 9.332 | 9.841 | 9.6% |
| load | | | 0.127 | 0.134 | 0.142 | |
| output | | | 0.900 | 1.020 | 1.160 | |
| other | | | | 1.951 | | |
| sum | | | | 97.617 | | 100.0% |

##### host, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.706 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 185.071, median 185.882, mean 186.671, max 191.341, std 1.846; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 4.677 | 4.902 | 5.352 | 2.9% |
| M2M | host | 7 | 1.162 | 1.313 | 1.707 | 0.8% |
| M2L | host | 5 | 34.854 | 35.009 | 35.175 | 20.6% |
| L2L | host | 7 | 1.306 | 1.456 | 1.852 | 0.9% |
| P2L | host | 1 | 0.096 | 0.111 | 0.122 | 0.1% |
| M2P | host | 1 | 0.604 | 0.744 | 0.837 | 0.4% |
| L2P | host | 2 | 12.535 | 12.749 | 14.044 | 7.5% |
| P2P | host | 2 | 113.135 | 113.364 | 113.876 | 66.8% |
| load | | | 1.432 | 1.471 | 1.507 | |
| output | | | 8.281 | 8.519 | 8.698 | |
| other | | | | 3.467 | | |
| sum | | | | 169.648 | | 100.0% |

##### host, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.509 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 894.550, median 896.876, mean 903.120, max 961.020, std 20.376; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.26e-5, ∇φ 2.26e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 22.501 | 22.989 | 23.418 | 2.6% |
| M2M | host | 7 | 3.736 | 3.846 | 3.942 | 0.4% |
| M2L | host | 5 | 679.201 | 680.523 | 681.720 | 77.2% |
| L2L | host | 7 | 3.917 | 4.067 | 4.210 | 0.5% |
| P2L | host | 1 | 0.250 | 0.290 | 0.315 | 0.0% |
| M2P | host | 1 | 0.771 | 0.814 | 0.856 | 0.1% |
| L2P | host | 2 | 55.005 | 55.223 | 55.970 | 6.3% |
| P2P | host | 2 | 113.745 | 113.971 | 114.440 | 12.9% |
| load | | | 1.473 | 1.516 | 1.567 | |
| output | | | 8.431 | 12.972 | 51.892 | |
| other | | | | 3.855 | | |
| sum | | | | 881.724 | | 100.0% |

##### host, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.786 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 39.373, median 40.245, mean 40.300, max 41.141, std 0.527; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.807 | 0.926 | 1.088 | 2.5% |
| M2M | host | 5 | 0.328 | 0.378 | 0.506 | 1.0% |
| M2L | host | 4 | 6.126 | 6.304 | 6.911 | 17.1% |
| L2L | host | 5 | 0.627 | 0.712 | 0.875 | 1.9% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 1.504 | 1.614 | 1.798 | 4.4% |
| P2P | host | 1 | 26.789 | 26.949 | 27.492 | 73.1% |
| load | | | 0.155 | 0.170 | 0.183 | |
| output | | | 0.928 | 1.097 | 1.207 | |
| other | | | | 2.042 | | |
| sum | | | | 36.884 | | 100.0% |

##### host, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.800 s, of it tables 0.019 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 199.898, median 200.294, mean 200.920, max 206.138, std 1.874; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 2.622 | 2.780 | 3.524 | 1.4% |
| M2M | host | 5 | 0.916 | 0.962 | 1.124 | 0.5% |
| M2L | host | 4 | 157.663 | 158.369 | 162.695 | 80.6% |
| L2L | host | 5 | 1.223 | 1.327 | 1.696 | 0.7% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 5.895 | 6.166 | 6.694 | 3.1% |
| P2P | host | 1 | 26.788 | 26.866 | 26.953 | 13.7% |
| load | | | 0.178 | 0.188 | 0.202 | |
| output | | | 1.005 | 1.186 | 1.275 | |
| other | | | | 2.026 | | |
| sum | | | | 196.469 | | 100.0% |

##### host, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.601 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 435.215, median 436.183, mean 436.289, max 437.891, std 0.874; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 4.890 | 5.293 | 5.672 | 1.3% |
| M2M | host | 7 | 1.232 | 1.392 | 1.699 | 0.3% |
| M2L | host | 5 | 50.292 | 50.696 | 51.490 | 12.3% |
| L2L | host | 7 | 1.256 | 1.391 | 1.639 | 0.3% |
| P2L | host | 1 | 0.100 | 0.105 | 0.112 | 0.0% |
| M2P | host | 1 | 0.597 | 0.677 | 0.767 | 0.2% |
| L2P | host | 2 | 13.435 | 13.541 | 13.610 | 3.3% |
| P2P | host | 2 | 339.313 | 339.609 | 340.235 | 82.3% |
| load | | | 2.114 | 2.149 | 2.196 | |
| output | | | 11.764 | 11.858 | 11.942 | |
| other | | | | 3.450 | | |
| sum | | | | 412.704 | | 100.0% |

##### host, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.202 s, of it tables 0.019 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 1836.856, median 1841.190, mean 1846.811, max 1876.574, std 13.692; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 23.145 | 23.574 | 24.305 | 1.3% |
| M2M | host | 7 | 6.371 | 6.579 | 7.049 | 0.4% |
| M2L | host | 5 | 1365.766 | 1377.611 | 1436.349 | 75.8% |
| L2L | host | 7 | 6.247 | 6.359 | 6.592 | 0.3% |
| P2L | host | 1 | 0.278 | 0.286 | 0.304 | 0.0% |
| M2P | host | 1 | 0.766 | 0.810 | 0.861 | 0.0% |
| L2P | host | 2 | 58.218 | 58.469 | 58.763 | 3.2% |
| P2P | host | 2 | 341.843 | 343.663 | 347.845 | 18.9% |
| load | | | 2.295 | 2.378 | 2.609 | |
| output | | | 11.691 | 11.830 | 11.978 | |
| other | | | | 3.855 | | |
| sum | | | | 1817.351 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### host, 72 threads, after, `--reuse-output`

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 07:09:10 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 07:09:10 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | none asked for (host only) |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL not compiled in |
| compiled backends | host |
| source revision | efdcb8d-dirty on tbetcke/phase4s_t9 (.source-revision) |
| command | `nd-fmm-bench --backend host --threads 72 --n 1e6,1e7 --precision f32,f64 --degree 3,8 --reuse-output --table-cache L/tables --output L/after/host-reuse.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72; host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, `--reuse-output`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| host | f32 | 1000000 | 3 | Dense | 0.793 | 19.846 | 20.434 | 20.649 | 21.895 | 0.720 | 2.00e-3 | 3.80e-3 | yes |
| host | f32 | 1000000 | 8 | Dense | 0.826 | 100.472 | 100.905 | 100.969 | 101.780 | 0.386 | 1.40e-5 | 4.59e-5 | yes |
| host | f32 | 10000000 | 3 | Dense | 12.687 | 184.515 | 185.209 | 185.360 | 186.535 | 0.596 | 1.87e-3 | 1.87e-3 | yes |
| host | f32 | 10000000 | 8 | Dense | 12.545 | 893.141 | 896.624 | 901.905 | 940.452 | 14.491 | 1.26e-5 | 2.26e-5 | yes |
| host | f64 | 1000000 | 3 | Dense | 0.769 | 39.520 | 40.381 | 40.401 | 41.519 | 0.531 | 2.00e-3 | 3.80e-3 | yes |
| host | f64 | 1000000 | 8 | Dense | 0.798 | 199.032 | 199.664 | 199.691 | 200.651 | 0.452 | 1.36e-5 | 4.59e-5 | yes |
| host | f64 | 10000000 | 3 | Dense | 12.513 | 435.521 | 441.527 | 445.402 | 486.845 | 14.870 | 1.87e-3 | 1.87e-3 | yes |
| host | f64 | 10000000 | 8 | Dense | 11.929 | 1838.546 | 1843.588 | 1847.615 | 1888.086 | 14.434 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f32 | 1000000 | 3 | 0.861 | 0.365 | 4.357 | 0.670 | 0.000 | 0.000 | 1.600 | 9.456 | 0.125 | 0.895 | 2.002 | 17.310 | 20.594 | 0.838 |
| host | f32 | 1000000 | 8 | 2.671 | 0.621 | 77.957 | 1.063 | 0.000 | 0.000 | 5.754 | 9.253 | 0.125 | 0.884 | 1.885 | 97.319 | 100.675 | 0.964 |
| host | f32 | 10000000 | 3 | 4.824 | 1.307 | 35.396 | 1.590 | 0.114 | 0.721 | 12.872 | 113.522 | 1.203 | 8.345 | 3.511 | 170.346 | 186.451 | 0.919 |
| host | f32 | 10000000 | 8 | 23.035 | 3.851 | 679.007 | 8.326 | 0.300 | 0.778 | 55.186 | 113.819 | 1.210 | 8.415 | 3.900 | 884.302 | 903.375 | 0.980 |
| host | f64 | 1000000 | 3 | 0.975 | 0.353 | 6.363 | 0.667 | 0.000 | 0.000 | 1.653 | 26.908 | 0.177 | 1.001 | 1.808 | 36.920 | 40.387 | 0.914 |
| host | f64 | 1000000 | 8 | 2.843 | 0.919 | 157.535 | 1.285 | 0.000 | 0.000 | 6.062 | 26.858 | 0.189 | 1.038 | 1.932 | 195.500 | 199.706 | 0.979 |
| host | f64 | 10000000 | 3 | 5.298 | 1.310 | 49.619 | 1.431 | 0.105 | 0.723 | 13.837 | 340.100 | 1.937 | 11.926 | 3.339 | 412.423 | 436.091 | 0.926 |
| host | f64 | 10000000 | 8 | 23.646 | 6.658 | 1375.997 | 6.566 | 0.294 | 0.825 | 58.914 | 344.996 | 2.016 | 11.806 | 3.769 | 1817.897 | 1847.115 | 0.984 |

##### host, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.793 s, of it tables 0.000 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 19.846, median 20.434, mean 20.649, max 21.895, std 0.720; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.751 | 0.861 | 0.953 | 5.0% |
| M2M | host | 5 | 0.309 | 0.365 | 0.426 | 2.1% |
| M2L | host | 4 | 4.256 | 4.357 | 4.902 | 25.2% |
| L2L | host | 5 | 0.579 | 0.670 | 0.857 | 3.9% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 1.418 | 1.600 | 2.270 | 9.2% |
| P2P | host | 1 | 9.259 | 9.456 | 10.176 | 54.6% |
| load | | | 0.106 | 0.125 | 0.134 | |
| output | | | 0.799 | 0.895 | 0.993 | |
| other | | | | 2.002 | | |
| sum | | | | 17.310 | | 100.0% |

##### host, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.826 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 100.472, median 100.905, mean 100.969, max 101.780, std 0.386; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.40e-5, ∇φ 4.59e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 2.582 | 2.671 | 3.111 | 2.7% |
| M2M | host | 5 | 0.578 | 0.621 | 0.648 | 0.6% |
| M2L | host | 4 | 77.672 | 77.957 | 78.348 | 80.1% |
| L2L | host | 5 | 0.949 | 1.063 | 1.136 | 1.1% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 5.692 | 5.754 | 5.859 | 5.9% |
| P2P | host | 1 | 9.222 | 9.253 | 9.288 | 9.5% |
| load | | | 0.111 | 0.125 | 0.145 | |
| output | | | 0.817 | 0.884 | 0.953 | |
| other | | | | 1.885 | | |
| sum | | | | 97.319 | | 100.0% |

##### host, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.687 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 184.515, median 185.209, mean 185.360, max 186.535, std 0.596; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 4.685 | 4.824 | 5.023 | 2.8% |
| M2M | host | 7 | 1.224 | 1.307 | 1.403 | 0.8% |
| M2L | host | 5 | 34.979 | 35.396 | 37.202 | 20.8% |
| L2L | host | 7 | 1.288 | 1.590 | 3.016 | 0.9% |
| P2L | host | 1 | 0.097 | 0.114 | 0.131 | 0.1% |
| M2P | host | 1 | 0.662 | 0.721 | 0.903 | 0.4% |
| L2P | host | 2 | 12.567 | 12.872 | 13.961 | 7.6% |
| P2P | host | 2 | 113.134 | 113.522 | 114.017 | 66.6% |
| load | | | 1.159 | 1.203 | 1.277 | |
| output | | | 8.173 | 8.345 | 8.724 | |
| other | | | | 3.511 | | |
| sum | | | | 170.346 | | 100.0% |

##### host, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.545 s, of it tables 0.010 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 893.141, median 896.624, mean 901.905, max 940.452, std 14.491; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.26e-5, ∇φ 2.26e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 22.698 | 23.035 | 23.679 | 2.6% |
| M2M | host | 7 | 3.732 | 3.851 | 4.061 | 0.4% |
| M2L | host | 5 | 677.182 | 679.007 | 680.274 | 76.8% |
| L2L | host | 7 | 3.932 | 8.326 | 46.138 | 0.9% |
| P2L | host | 1 | 0.257 | 0.300 | 0.338 | 0.0% |
| M2P | host | 1 | 0.747 | 0.778 | 0.829 | 0.1% |
| L2P | host | 2 | 54.599 | 55.186 | 55.718 | 6.2% |
| P2P | host | 2 | 113.521 | 113.819 | 114.131 | 12.9% |
| load | | | 1.161 | 1.210 | 1.352 | |
| output | | | 8.197 | 8.415 | 8.694 | |
| other | | | | 3.900 | | |
| sum | | | | 884.302 | | 100.0% |

##### host, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.769 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 39.520, median 40.381, mean 40.401, max 41.519, std 0.531; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.765 | 0.975 | 1.366 | 2.6% |
| M2M | host | 5 | 0.310 | 0.353 | 0.451 | 1.0% |
| M2L | host | 4 | 6.171 | 6.363 | 6.993 | 17.2% |
| L2L | host | 5 | 0.609 | 0.667 | 0.832 | 1.8% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 1.498 | 1.653 | 1.909 | 4.5% |
| P2P | host | 1 | 26.799 | 26.908 | 27.264 | 72.9% |
| load | | | 0.162 | 0.177 | 0.236 | |
| output | | | 0.917 | 1.001 | 1.203 | |
| other | | | | 1.808 | | |
| sum | | | | 36.920 | | 100.0% |

##### host, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.798 s, of it tables 0.019 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 199.032, median 199.664, mean 199.691, max 200.651, std 0.452; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 2.616 | 2.843 | 3.764 | 1.5% |
| M2M | host | 5 | 0.875 | 0.919 | 1.076 | 0.5% |
| M2L | host | 4 | 157.309 | 157.535 | 157.757 | 80.6% |
| L2L | host | 5 | 1.221 | 1.285 | 1.399 | 0.7% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 5.987 | 6.062 | 6.213 | 3.1% |
| P2P | host | 1 | 26.817 | 26.858 | 26.884 | 13.7% |
| load | | | 0.173 | 0.189 | 0.211 | |
| output | | | 0.983 | 1.038 | 1.094 | |
| other | | | | 1.932 | | |
| sum | | | | 195.500 | | 100.0% |

##### host, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 12.513 s, of it tables 0.001 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 435.521, median 441.527, mean 445.402, max 486.845, std 14.870; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 4.897 | 5.298 | 5.753 | 1.3% |
| M2M | host | 7 | 1.211 | 1.310 | 1.491 | 0.3% |
| M2L | host | 5 | 49.266 | 49.619 | 50.209 | 12.0% |
| L2L | host | 7 | 1.296 | 1.431 | 1.589 | 0.3% |
| P2L | host | 1 | 0.099 | 0.105 | 0.114 | 0.0% |
| M2P | host | 1 | 0.641 | 0.723 | 0.938 | 0.2% |
| L2P | host | 2 | 13.447 | 13.837 | 14.475 | 3.4% |
| P2P | host | 2 | 339.095 | 340.100 | 340.631 | 82.5% |
| load | | | 1.798 | 1.937 | 2.425 | |
| output | | | 11.542 | 11.926 | 13.142 | |
| other | | | | 3.339 | | |
| sum | | | | 412.423 | | 100.0% |

##### host, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 11.929 s, of it tables 0.019 s (loaded from the cache) and device 0.000 s.
- Evaluation over 10 repeats, ms: min 1838.546, median 1843.588, mean 1847.615, max 1888.086, std 14.434; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.21e-5, ∇φ 2.26e-5.

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 2 | 23.001 | 23.646 | 25.260 | 1.3% |
| M2M | host | 7 | 6.458 | 6.658 | 7.160 | 0.4% |
| M2L | host | 5 | 1363.597 | 1375.997 | 1450.703 | 75.7% |
| L2L | host | 7 | 6.226 | 6.566 | 7.226 | 0.4% |
| P2L | host | 1 | 0.281 | 0.294 | 0.306 | 0.0% |
| M2P | host | 1 | 0.799 | 0.825 | 0.889 | 0.0% |
| L2P | host | 2 | 58.164 | 58.914 | 60.438 | 3.2% |
| P2P | host | 2 | 341.879 | 344.996 | 347.602 | 19.0% |
| load | | | 1.912 | 2.016 | 2.530 | |
| output | | | 11.579 | 11.806 | 12.037 | |
| other | | | | 3.769 | | |
| sum | | | | 1817.897 | | 100.0% |

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate_into` (one reused `Output`), with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### CUDA under `nsys`, after

<details><summary>Output</summary>

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 07:18:46 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 07:18:46 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | efdcb8d-dirty on tbetcke/phase4s_t9 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e7 --precision f32 --degree 3 --repeats 3 --warmup 1 --kinds off --accuracy off --table-cache L/tables --output L/after/nsys/bench-f32.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 1 warm-up, then 3 timed evaluations; kind timings `off` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | off |
| table cache | `L/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 3 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 10000000 | 3 | Dense | 14.752 | 60.335 | 60.844 | 60.683 | 60.871 | 0.302 | – | – | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `off` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 10000000 | 3 | off | – | – | – | – | – | – | – | – | – | – | – | – | – |

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.752 s, of it tables 0.001 s (loaded from the cache) and device 2.516 s.
- Evaluation over 3 repeats, ms: min 60.335, median 60.844, mean 60.683, max 60.871, std 0.302; bit-identical across the repeats: yes.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (32 units); leaf cube (32 units, tile 32); GEMM cube(16 x 4 units, 8 columns per unit); rotation cube (32 units); device memory 4079.8 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 76, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 1 chunk
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 8 columns per unit), 3 chunks

##### Method

Each combination builds the FMM once and runs 1 warm-up evaluations, which compile the device kernels and are not counted, then times 3 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `off`, which runs its own warm-up and 3 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

</details>

### Metal (M3 Max), before

<details><summary>Output</summary>

#### FMM benchmark: Mac, 2026-10-08 06:11:55 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 06:11:55 UTC |
| host | Mac |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | efdcb8d-dirty (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --degree 3,8 --n 1e6 --output bench-results/t9-before-metal.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | none |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 1000000 | 3 | Dense | 0.851 | 15.819 | 15.995 | 16.037 | 16.304 | 0.175 | 2.00e-3 | 3.80e-3 | yes |
| metal | f32 | 1000000 | 8 | Dense | 1.048 | 81.887 | 82.219 | 82.211 | 82.595 | 0.204 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 1000000 | 3 | 0.517 | 1.878 | 6.816 | 1.671 | 0.000 | 0.000 | 0.444 | 5.203 | 2.448 | 3.151 | 0.024 | 16.528 | 22.244 | 1.031 |
| metal | f32 | 1000000 | 8 | 1.112 | 2.054 | 71.306 | 2.008 | 0.000 | 0.000 | 0.710 | 5.075 | 2.633 | 3.254 | 0.027 | 82.266 | 88.371 | 1.001 |

##### metal, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.851 s, of it tables 0.003 s (built) and device 0.199 s.
- Evaluation over 10 repeats, ms: min 15.819, median 15.995, mean 16.037, max 16.304, std 0.175; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(16 x 4 units, 4 columns per unit); rotation cube (32 units); device memory 324.4 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 64, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 6 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.397 | 0.517 | 0.612 | 3.1% |
| M2M | device | 5 | 1.807 | 1.878 | 2.067 | 11.4% |
| M2L | device | 4 | 6.753 | 6.816 | 6.926 | 41.2% |
| L2L | device | 5 | 1.584 | 1.671 | 1.898 | 10.1% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.400 | 0.444 | 0.493 | 2.7% |
| P2P | device | 1 | 5.159 | 5.203 | 5.411 | 31.5% |
| load | | | 2.332 | 2.448 | 2.643 | |
| output | | | 3.088 | 3.151 | 3.398 | |
| other | | | | 0.024 | | |
| sum | | | | 16.528 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.732 ms on average.

##### metal, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.048 s, of it tables 0.239 s (built) and device 0.167 s.
- Evaluation over 10 repeats, ms: min 81.887, median 82.219, mean 82.211, max 82.595, std 0.204; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(32 x 2 units, 4 columns per unit); rotation cube (96 units); device memory 357.5 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 133, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - M2M (Local) level 2: 512 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2M (Local) level 3: 4096 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2M (Local) level 4: 32768 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - L2L level 3: 512 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - L2L level 4: 4096 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - L2L level 5: 32768 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 3 chunks
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 27 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 1.038 | 1.112 | 1.189 | 1.4% |
| M2M | device | 5 | 1.910 | 2.054 | 2.366 | 2.5% |
| M2L | device | 4 | 70.933 | 71.306 | 71.516 | 86.7% |
| L2L | device | 5 | 1.875 | 2.008 | 2.129 | 2.4% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.605 | 0.710 | 0.820 | 0.9% |
| P2P | device | 1 | 5.014 | 5.075 | 5.252 | 6.2% |
| load | | | 2.512 | 2.633 | 2.767 | |
| output | | | 3.028 | 3.254 | 3.465 | |
| other | | | | 0.027 | | |
| sum | | | | 82.266 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.702 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Mac; never asserted.

</details>

### Metal (M3 Max), after

<details><summary>Output</summary>

#### FMM benchmark: Timos-MacBook-Pro.local, 2026-10-08 06:42:55 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 06:42:55 UTC |
| host | Timos-MacBook-Pro.local |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | efdcb8d-dirty (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --degree 3,8 --n 1e6 --output bench-results/t9-after-metal.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | none |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 1000000 | 3 | Dense | 0.886 | 13.910 | 14.035 | 14.064 | 14.316 | 0.116 | 2.00e-3 | 3.80e-3 | yes |
| metal | f32 | 1000000 | 8 | Dense | 1.079 | 82.500 | 84.102 | 83.956 | 84.920 | 0.717 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 1000000 | 3 | 0.520 | 1.668 | 7.200 | 1.947 | 0.000 | 0.000 | 0.566 | 5.281 | 1.666 | 2.145 | 0.028 | 17.182 | 21.124 | 1.222 |
| metal | f32 | 1000000 | 8 | 1.161 | 2.222 | 72.715 | 2.665 | 0.000 | 0.000 | 0.715 | 5.305 | 1.529 | 2.476 | 0.033 | 84.784 | 89.018 | 1.010 |

##### metal, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.886 s, of it tables 0.003 s (built) and device 0.219 s.
- Evaluation over 10 repeats, ms: min 13.910, median 14.035, mean 14.064, max 14.316, std 0.116; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(16 x 4 units, 4 columns per unit); rotation cube (32 units); device memory 324.4 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 64, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 6 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.424 | 0.520 | 0.625 | 3.0% |
| M2M | device | 5 | 1.539 | 1.668 | 1.769 | 9.7% |
| M2L | device | 4 | 6.650 | 7.200 | 8.274 | 41.9% |
| L2L | device | 5 | 1.620 | 1.947 | 3.440 | 11.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.382 | 0.566 | 1.492 | 3.3% |
| P2P | device | 1 | 5.061 | 5.281 | 5.592 | 30.7% |
| load | | | 1.378 | 1.666 | 2.340 | |
| output | | | 1.958 | 2.145 | 2.328 | |
| other | | | | 0.028 | | |
| sum | | | | 17.182 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.665 ms on average.

##### metal, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.079 s, of it tables 0.233 s (built) and device 0.178 s.
- Evaluation over 10 repeats, ms: min 82.500, median 84.102, mean 83.956, max 84.920, std 0.717; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(32 x 2 units, 4 columns per unit); rotation cube (96 units); device memory 357.5 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 133, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - M2M (Local) level 2: 512 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2M (Local) level 3: 4096 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2M (Local) level 4: 32768 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - L2L level 3: 512 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - L2L level 4: 4096 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - L2L level 5: 32768 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 3 chunks
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 27 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 1.008 | 1.161 | 1.705 | 1.4% |
| M2M | device | 5 | 1.979 | 2.222 | 3.006 | 2.6% |
| M2L | device | 4 | 71.026 | 72.715 | 75.315 | 85.8% |
| L2L | device | 5 | 1.928 | 2.665 | 3.814 | 3.1% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.635 | 0.715 | 0.795 | 0.8% |
| P2P | device | 1 | 5.141 | 5.305 | 6.023 | 6.3% |
| load | | | 1.385 | 1.529 | 1.703 | |
| output | | | 2.080 | 2.476 | 3.395 | |
| other | | | | 0.033 | | |
| sum | | | | 84.784 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.721 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MacBook-Pro.local; never asserted.

</details>

### Metal (M3 Max), after, `--reuse-output`

<details><summary>Output</summary>

#### FMM benchmark: Timos-MacBook-Pro.local, 2026-10-08 06:43:07 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 06:43:07 UTC |
| host | Timos-MacBook-Pro.local |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | efdcb8d-dirty (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --degree 3,8 --n 1e6 --reuse-output --output bench-results/t9-after-metal-reuse.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, `--reuse-output`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | none |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 1000000 | 3 | Dense | 0.850 | 13.859 | 14.114 | 14.090 | 14.260 | 0.147 | 2.00e-3 | 3.80e-3 | yes |
| metal | f32 | 1000000 | 8 | Dense | 1.027 | 79.958 | 83.355 | 83.229 | 86.565 | 2.693 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 1000000 | 3 | 0.513 | 1.595 | 6.780 | 1.790 | 0.000 | 0.000 | 0.431 | 5.098 | 1.449 | 2.767 | 0.024 | 16.209 | 20.545 | 1.150 |
| metal | f32 | 1000000 | 8 | 1.125 | 2.015 | 70.783 | 2.094 | 0.000 | 0.000 | 0.657 | 5.177 | 1.591 | 2.666 | 0.026 | 81.851 | 86.331 | 0.983 |

##### metal, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.850 s, of it tables 0.003 s (built) and device 0.191 s.
- Evaluation over 10 repeats, ms: min 13.859, median 14.114, mean 14.090, max 14.260, std 0.147; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 2.00e-3, ∇φ 3.80e-3.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(16 x 4 units, 4 columns per unit); rotation cube (32 units); device memory 324.4 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 64, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 6 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 0.453 | 0.513 | 0.585 | 3.2% |
| M2M | device | 5 | 1.573 | 1.595 | 1.682 | 9.8% |
| M2L | device | 4 | 6.569 | 6.780 | 6.889 | 41.8% |
| L2L | device | 5 | 1.721 | 1.790 | 1.868 | 11.0% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.368 | 0.431 | 0.669 | 2.7% |
| P2P | device | 1 | 4.973 | 5.098 | 5.143 | 31.5% |
| load | | | 1.340 | 1.449 | 1.513 | |
| output | | | 2.670 | 2.767 | 2.904 | |
| other | | | | 0.024 | | |
| sum | | | | 16.209 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.661 ms on average.

##### metal, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.027 s, of it tables 0.227 s (built) and device 0.160 s.
- Evaluation over 10 repeats, ms: min 79.958, median 83.355, mean 83.229, max 86.565, std 2.693; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.36e-5, ∇φ 4.59e-5.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(32 x 2 units, 4 columns per unit); rotation cube (96 units); device memory 357.5 MB.
- One evaluation: uploads 1 (4000000 B), downloads 1 (16000000 B), launches 133, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - M2M (Local) level 2: 512 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2M (Local) level 3: 4096 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2M (Local) level 4: 32768 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - L2L level 1: 8 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 1, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - L2L level 2: 64 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk (library rejected: [8, 8, 81] (groups, columns per group, order): Unable to launch matmul because a required feature is unavailable: Plane dimension unsupported: 32. Only 32 & 64 are supported.

)
  - L2L level 3: 512 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - L2L level 4: 4096 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - L2L level 5: 32768 pairs, library (matmul_simple_cyclic_cmma), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 3 chunks
  - M2L level 5: 5398920 pairs, hand-written cube(32 x 2 units, 4 columns per unit), 27 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 1 | 1.016 | 1.125 | 1.164 | 1.4% |
| M2M | device | 5 | 1.915 | 2.015 | 2.200 | 2.5% |
| M2L | device | 4 | 70.623 | 70.783 | 71.059 | 86.5% |
| L2L | device | 5 | 1.877 | 2.094 | 2.208 | 2.6% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.560 | 0.657 | 0.786 | 0.8% |
| P2P | device | 1 | 5.120 | 5.177 | 5.248 | 6.3% |
| load | | | 1.403 | 1.591 | 1.705 | |
| output | | | 2.040 | 2.666 | 2.881 | |
| other | | | | 0.026 | | |
| sum | | | | 81.851 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.642 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate_into` (one reused `Output`), with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MacBook-Pro.local; never asserted.

</details>

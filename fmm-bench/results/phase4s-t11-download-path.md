# Phase 4S T11: the download and upload paths, before and after

The download read in place and the charges uploaded without a copy
(docs/phase4s/T11-download-path.md, C4S.10; decisions 14 and 15 of docs/phase4s/README.md;
the spike in spikes/download-path/REPORT.md). These are the measurements behind
docs/design/device-path.md §18.5, which summarises them. Run on 2026-10-08: locust
10:32–11:30 UTC, the M3 Max 11:51–11:56 BST.

**Every number is measured**: on locust (CUDA on the H100) or on the M3 Max (Metal).
Timings are reported and never asserted.

## Setup

| item | value |
| --- | --- |
| machines | locust: GH200 480GB, 72 Neoverse-V2 cores, one H100 (96 GB), driver 565.57.01, the spack environment of tools/gh200/ (CUDA 12.6), rustc 1.99.0, CubeCL 0.11.0-pre.4 (LLVM NVPTX); the M3 Max: Metal through wgpu (MSL), rustc 1.99.0 |
| code | before: main `a12d9d6` (T9 merged, the T11 brief), a detached worktree synced to `/data/ucahtbe/fmm/detached-a12d9d6`; after: this task's working tree, synced to `/data/ucahtbe/fmm/tbetcke/phase4s_t11-2` (reports `a12d9d6-dirty`) |
| problem | `nd-fmm-bench`: N points uniform in [0, 1]³ (seed `0xbe9c06`), sources equal to targets, gradients on, `max_level` 16, 64 points per leaf, the static rule (`Dense`), tables from a table cache (T9's on locust, a fresh one on the M3 Max) |
| timing | 2 warm-ups, 10 timed evaluations (min, median, mean, max, standard deviation); load and output from a second build with `KindTiming::Synchronous` (a sync after the charge upload, so `load` holds the gather, the upload, the zeroing and the scatter on a device); every BLAS thread variable 1 |
| load, locust | checked before and after every step by the run script (`/data/ucahtbe/logs/t11/bench/*.load-*`): no compute process on the GPU at any check and no other user's job (the only other user's process an `nvitop` monitor below 1% of a core); load average 0.03 before the first step and at most 1.40 after (this run's own work). Each step waited for no GPU process and a load average below 4. The checks right after a step read this run's own utilisation (50–100%, 2 MiB, SM 1,980 MHz) |
| clocks | not locked (no administrator access): SM 345 MHz idle, application and maximum 1,980 MHz, memory 2,619 MHz (`nvidia-smi -q -d CLOCK` before and after) |
| load, M3 Max | a desktop session (Defender, WindowServer); load average 14.6 at the start, most of it from this task's build just before, and 4.0 at the end. The Metal numbers are indicative |

Commands (`tools/bench/run.sh`; on locust with `--table-cache /data/ucahtbe/logs/t9/tables`,
on the M3 Max with a scratch table cache; `--output` per step):

```sh
# locust, in the before tree and in the after tree, each with and without --reuse-output
tools/bench/run.sh --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8 [--reuse-output]
nsys profile --trace=cuda,osrt --sample=process-tree target/release/nd-fmm-bench --backend cuda \
    --n 1e7 --precision f32 --degree 3 --repeats 3 --warmup 1 --kinds off --accuracy off
# the M3 Max, outside the sandbox, in both trees, each with and without --reuse-output
tools/bench/run.sh --backend metal --precision f32 --n 1e6 --degree 3,8 [--reuse-output]
tools/bench/run.sh --backend metal --precision f32 --n 1e7 --degree 3 [--reuse-output]
```

## Whole outputs

Unchanged bit for bit. The C3.2 hashes of `tests/output_common` (the ignored gates, p = 6,
first charge vector) equal T9's (fmm-bench/results/phase4s-t9-host-part.md) on every
backend that ran:

| backend | precision | gradients | T9 | T11 |
| --- | --- | --- | --- | --- |
| host, 1 and 4 threads (M3 Max) | f64 | off | `0x94e1f03747ea1f20` | `0x94e1f03747ea1f20` |
| host, 1 and 4 threads (M3 Max) | f64 | on | `0xaf5fddbe12a92f52` | `0xaf5fddbe12a92f52` |
| CPU runtime (M3 Max), device pass and host pass | f64 | off | `0xa71c8a351574c16d` | `0xa71c8a351574c16d` |
| CPU runtime (M3 Max), device pass and host pass | f64 | on | `0x7fd66135393ff4b1` | `0x7fd66135393ff4b1` |
| Metal (M3 Max), host pass | f32 | off | `0x7fb4b39db44e3d21` | `0x7fb4b39db44e3d21` |
| Metal (M3 Max), host pass | f32 | on | `0x0dc405d9b331a414` | `0x0dc405d9b331a414` |
| CUDA (locust), device pass and host pass | f32 | off | `0xc57633de922c6e64` | `0xc57633de922c6e64` |
| CUDA (locust), device pass and host pass | f32 | on | `0xcc311d1921bfb320` | `0xcc311d1921bfb320` |
| CUDA (locust), device pass and host pass | f64 | off | `0xa71c8a351574c16d` | `0xa71c8a351574c16d` |
| CUDA (locust), device pass and host pass | f64 | on | `0x284e59e787590ed6` | `0x284e59e787590ed6` |
| host, 1 and 4 threads (Grace) | f64 | off / on | `0x94e1f03747ea1f20` / `0xaf5fddbe12a92f52` | the same |

Every test of `tests/mpi_exec.rs`, `tests/device_common` and the ignored gates asserts the
output against `Fmm::reference_output` with both output passes. In the benchmark, every
evaluation of every step gave the bits of the first, and the errors against the direct sum
are the same before and after in every row, on CUDA and on Metal.

## Summary

Medians in ms per evaluation (means in the raw output; three runs had one slow evaluation
each, marked *):

| N | precision | p | CUDA before | CUDA after | speed-up | `--reuse-output` before → after |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 10⁶ | f32 | 3 | 6.66 | 5.49 | 1.21× | 6.60 → 5.57 |
| 10⁶ | f32 | 8 | 15.47 | 14.52 | 1.07× | 15.55 → 14.42 |
| 10⁷ | f32 | 3 | 60.40* | 45.78 | 1.32× | 59.82 → 45.59 |
| 10⁷ | f32 | 8 | 153.91 | 139.44 | 1.10× | 153.44 → 139.11* |
| 10⁶ | f64 | 3 | 10.17 | 7.79* | 1.31× | 10.21 → 7.77 |
| 10⁶ | f64 | 8 | 25.60 | 23.21 | 1.10× | 25.66 → 23.18 |
| 10⁷ | f64 | 3 | 103.43 | 74.10 | 1.40× | 102.59 → 73.67 |
| 10⁷ | f64 | 8 | 269.20 | 240.22 | 1.12× | 268.60 → 239.67 |

\* one evaluation of the ten at 87.4, 175.5 and 10.7 ms (max), the others within 2% of the
median.

Load and output (means in ms; the download's part of the output in brackets), measured
(locust, CUDA):

| N | precision | p | load before → after | output before → after (download) |
| --- | --- | ---: | --- | --- |
| 10⁶ | f32 | 3 | 0.70 → 0.61 | 2.31 (1.25) → 1.32 (0.28) |
| 10⁶ | f32 | 8 | 0.69 → 0.63 | 2.25 (1.26) → 1.36 (0.26) |
| 10⁷ | f32 | 3 | 6.60 → 3.64 | 15.32 (12.69) → 3.71 (1.16) |
| 10⁷ | f32 | 8 | 6.63 → 3.64 | 15.35 (12.66) → 3.80 (1.16) |
| 10⁶ | f64 | 3 | 1.23 → 0.75 | 3.41 (2.32) → 1.42 (0.31) |
| 10⁶ | f64 | 8 | 1.22 → 0.74 | 3.47 (2.34) → 1.45 (0.29) |
| 10⁷ | f64 | 3 | 13.51 → 7.27 | 28.74 (24.72) → 5.70 (1.80) |
| 10⁷ | f64 | 8 | 13.45 → 7.30 | 28.70 (24.75) → 6.67 (1.81) |

- **The download** went from 12.7 to 1.16 ms at N = 10⁷ in f32 (160 MB) and from 24.7 to
  1.80 ms in f64 (320 MB), and from 1.25 / 2.3 to 0.26–0.31 ms at N = 10⁶. What remains
  is the wait for the last kernels, CubeCL's `read_one` (the GPU copy into its pinned
  buffer, 0.55 ms for 160 MB in the spike) and nothing else. The serial copy into the
  operator's buffer is gone; the parallel copy into the `Output` reads CubeCL's pinned
  buffer instead, at the same cost (output less download: 2.6 → 2.5 ms in f32 and
  4.0 → 3.9 ms in f64 at N = 10⁷).
- **The load** went from 6.6 to 3.6 ms (f32) and from 13.5 to 7.3 ms (f64) at N = 10⁷:
  the 3.0 / 6.4 ms of `to_vec` the spike measured, less nothing measurable for gathering
  into a fresh buffer. At N = 10⁶: 0.70 → 0.61 and 1.23 → 0.75 ms.
- **Evaluations** got 1.32× (f32) and 1.40× (f64) faster at N = 10⁷ and p = 3, 1.10–1.12×
  at p = 8; 1.21× and 1.31× at N = 10⁶ and p = 3, 1.07–1.10× at p = 8. The level calls
  did not change. Against the spike's model (REPORT.md, "Where the evaluation's time goes"),
  which removed the copy alone: 45.8 against ≈ 49.3 ms and 74.1 against ≈ 80.6 ms; with the
  `to_vec` also removed the model gives 46.4 and 74.2 ms.
- **`--reuse-output`** changes the medians by −1.0% to +1.4%, before and after: within
  the noise, as in T9.
- **Host memory**: with every kind on the device the operator no longer holds the
  target output on the host: 16 / 32 MB at N = 10⁶ and 160 / 320 MB at N = 10⁷ (f32 /
  f64) less per `Fmm` (`DeviceReport::host_mirror_bytes` is 0; tested in
  `tests/device_common`). Nor does `Fmm` keep the charges' upload buffer (N_s s bytes): each
  evaluation fills a fresh one that CubeCL takes over and frees after the transfer.

Metal (the M3 Max, f32; the host pass; medians in ms, means of load and output with the
download in brackets), measured (M3 Max, Metal):

| N | p | before | after | `--reuse-output` before → after | load before → after | output before → after (download) |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 10⁶ | 3 | 14.20 | 13.89 | 14.30 → 13.90 | 1.49 → 1.52 | 2.31 (0.76) → 2.00 (0.51) |
| 10⁶ | 8 | 80.62 | 80.26 | 80.96 → 80.28 | 1.58 → 1.59 | 2.82 (0.72) → 2.68 (0.50) |
| 10⁷ | 3 | 221.52 | 216.97 | 221.08 → 218.55 | 15.42 → 14.62 | 33.26 (3.94) → 30.96 (1.24) |

On Metal the download went from 0.76 to 0.51 ms at N = 10⁶ and from 3.9 to 1.2 ms at
N = 10⁷; the host pass reads wgpu's mapped staging buffer as fast as it read the
operator's store (output less download: 29.3 → 29.7 ms at N = 10⁷). The load is wgpu's
write, which T11 does not change (`to_vec` was 0.06 ms at 4 MB). Evaluations at N = 10⁶ and
p = 8 differ within the M2L level calls' noise.

## The `nsys` trace after (N = 10⁷, f32, p = 3)

46.1 ms per evaluation under the profiler (one warm-up, three timed; 45.8 ms without). The
CUDA API summary of the whole run: `cuMemAllocHost_v2` 6 calls, 78.7 ms, median 2.4 ms, at
most 58.1 ms, as in T9's trace (6 calls): the pinned pool grows during the build and the
warm-up and never again (spikes/download-path/REPORT.md, "CubeCL's host pools").
`cuMemcpyDtoHAsync_v2` 4 calls, the warm-up's and the three timed evaluations' downloads.
The host time left in the download is the 1.16 ms of `StageTimings::download` above: the
wait for the end of the last level calls and CubeCL's read, which is mostly the GPU's copy.

## Checks run for T11

On the M3 Max (every exit status 0 unless said):
- `cargo fmt --all` and `cargo fmt -- --check`;
- the root CI checks: `cargo clippy -- -D warnings`, `cargo clippy --examples -- -D
  warnings`, `RUST_MIN_STACK=8388608 cargo test`, `cargo doc --no-deps`;
- the stricter checks: `cargo clippy --workspace --all-targets -- -D warnings` (it first
  failed on the launch-error test helpers, unused without a backend feature; they moved
  into a module behind the backend features, and the rerun is clean) and
  `RUST_MIN_STACK=8388608 cargo test --workspace` (737 passed, 44 ignored);
- `nd-fmm-kernels`: clippy with no feature, `cpu` and `cpu,metal` (and `cpu,metal,cuda`),
  `cargo check --features cuda`, `cargo test -p nd-fmm-kernels`, `cargo test --features cpu
  --release -- --show-output` (32 unit and 69 integration tests, 2 ignored), `cargo doc`;
  by hand outside the sandbox `cargo test --release --features metal -- --ignored
  --show-output` (51 Metal tests and the Metal launch-error test);
- `nd-fmm-exec`: clippy with no feature, `cpu`, `cpu,metal` and `cpu,metal,cuda`,
  `cargo check --features cuda`, `cargo test --features cpu --release` and with
  `-- --ignored` (every gate, the CPU runtime and the host), `cargo doc --features cpu`; by
  hand outside the sandbox `cargo test --features metal --release -- --ignored`
  (`tests/device_metal.rs` and the gates on Metal);
- `nd-fmm-bench`: clippy with no feature and `cpu,metal`, `cargo check --features cuda`,
  `cargo test` and `cargo test --features cpu --release` (the smoke test, unchanged).

On locust (`/data/ucahtbe/logs/t11/checks/`): `cargo test -p nd-fmm-kernels --release
--features cpu,cuda -- --include-ignored --show-output` (165 passed, the CUDA launch-error
test and the view tests among them; rerun for the unit tests after the last sync),
`cargo clippy -p nd-fmm-kernels --all-targets --features cpu,cuda -- -D warnings`,
`cargo clippy -p nd-fmm-exec --all-targets --features cpu,cuda -- -D warnings`,
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release` (79 passed) and
with `-- --ignored` (11 passed: `tests/device_cuda.rs`, the CUDA blocks of the gates and the
operator tests), `RUST_MIN_STACK=8388608 cargo test --workspace` (737 passed, 44 ignored).
Every device test printed the backends it ran; none reports a backend that did not run.

## Raw output

The reports of `nd-fmm-bench` as written, the headings moved down three levels.

### CUDA, before

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 10:34:29 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:34:29 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | a12d9d6 on detached-a12d9d6 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8 --table-cache /data/ucahtbe/logs/t9/tables --output /data/ucahtbe/logs/t11/bench/before.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/data/ucahtbe/logs/t9/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Dense | 1.059 | 6.449 | 6.657 | 6.646 | 6.880 | 0.141 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 8 | Dense | 1.220 | 15.293 | 15.469 | 15.475 | 15.683 | 0.110 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f32 | 10000000 | 3 | Dense | 14.756 | 60.021 | 60.397 | 63.059 | 87.430 | 8.566 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f32 | 10000000 | 8 | Dense | 14.058 | 153.651 | 153.907 | 153.995 | 154.396 | 0.256 | 1.21e-5 | 2.26e-5 | yes |
| cuda | f64 | 1000000 | 3 | Dense | 0.993 | 10.036 | 10.165 | 10.187 | 10.458 | 0.140 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 8 | Dense | 1.022 | 25.416 | 25.604 | 25.602 | 25.847 | 0.131 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 10000000 | 3 | Dense | 14.352 | 102.855 | 103.426 | 103.494 | 104.503 | 0.579 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f64 | 10000000 | 8 | Dense | 14.761 | 268.669 | 269.199 | 269.207 | 270.473 | 0.548 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.063 | 0.212 | 1.285 | 0.299 | 0.000 | 0.000 | 0.069 | 1.998 | 0.699 | 2.308 | 0.031 | 3.926 | 7.191 | 0.591 |
| cuda | f32 | 1000000 | 8 | 0.120 | 0.504 | 9.380 | 0.643 | 0.000 | 0.000 | 0.440 | 2.142 | 0.689 | 2.254 | 0.030 | 13.228 | 16.651 | 0.855 |
| cuda | f32 | 10000000 | 3 | 0.407 | 0.472 | 9.736 | 0.621 | 0.075 | 0.233 | 0.386 | 24.374 | 6.600 | 15.324 | 0.207 | 36.302 | 61.717 | 0.576 |
| cuda | f32 | 10000000 | 8 | 1.058 | 1.063 | 96.313 | 1.339 | 0.076 | 0.349 | 3.184 | 24.300 | 6.629 | 15.347 | 0.209 | 127.683 | 155.683 | 0.829 |
| cuda | f64 | 1000000 | 3 | 0.092 | 0.224 | 1.649 | 0.271 | 0.000 | 0.000 | 0.190 | 3.143 | 1.226 | 3.405 | 0.033 | 5.569 | 10.789 | 0.547 |
| cuda | f64 | 1000000 | 8 | 0.276 | 0.627 | 15.555 | 0.475 | 0.000 | 0.000 | 0.774 | 3.243 | 1.219 | 3.473 | 0.032 | 20.950 | 26.800 | 0.818 |
| cuda | f64 | 10000000 | 3 | 0.661 | 0.543 | 14.098 | 0.576 | 0.080 | 0.491 | 0.692 | 38.973 | 13.511 | 28.735 | 0.209 | 56.116 | 105.123 | 0.542 |
| cuda | f64 | 10000000 | 8 | 2.590 | 1.684 | 166.241 | 1.575 | 0.081 | 0.599 | 5.739 | 38.844 | 13.453 | 28.702 | 0.212 | 217.353 | 271.371 | 0.807 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.059 s, of it tables 0.000 s (loaded from the cache) and device 0.263 s.
- Evaluation over 10 repeats, ms: min 6.449, median 6.657, mean 6.646, max 6.880, std 0.141; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.060 | 0.063 | 0.065 | 1.6% |
| M2M | device | 5 | 0.181 | 0.212 | 0.220 | 5.4% |
| M2L | device | 4 | 1.269 | 1.285 | 1.290 | 32.7% |
| L2L | device | 5 | 0.284 | 0.299 | 0.307 | 7.6% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.066 | 0.069 | 0.072 | 1.7% |
| P2P | device | 1 | 1.981 | 1.998 | 2.013 | 50.9% |
| load | | | 0.580 | 0.699 | 0.925 | |
| output | | | 2.100 | 2.308 | 2.454 | |
| other | | | | 0.031 | | |
| sum | | | | 3.926 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.248 ms on average.

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.220 s, of it tables 0.010 s (loaded from the cache) and device 0.404 s.
- Evaluation over 10 repeats, ms: min 15.293, median 15.469, mean 15.475, max 15.683, std 0.110; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.118 | 0.120 | 0.123 | 0.9% |
| M2M | device | 5 | 0.329 | 0.504 | 0.528 | 3.8% |
| M2L | device | 4 | 9.375 | 9.380 | 9.394 | 70.9% |
| L2L | device | 5 | 0.631 | 0.643 | 0.648 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.428 | 0.440 | 0.449 | 3.3% |
| P2P | device | 1 | 2.126 | 2.142 | 2.170 | 16.2% |
| load | | | 0.567 | 0.689 | 0.772 | |
| output | | | 2.106 | 2.254 | 2.373 | |
| other | | | | 0.030 | | |
| sum | | | | 13.228 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.257 ms on average.

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.756 s, of it tables 0.001 s (loaded from the cache) and device 2.226 s.
- Evaluation over 10 repeats, ms: min 60.021, median 60.397, mean 63.059, max 87.430, std 8.566; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.403 | 0.407 | 0.415 | 1.1% |
| M2M | device | 7 | 0.369 | 0.472 | 0.576 | 1.3% |
| M2L | device | 5 | 9.717 | 9.736 | 9.757 | 26.8% |
| L2L | device | 7 | 0.448 | 0.621 | 0.771 | 1.7% |
| P2L | device | 1 | 0.073 | 0.075 | 0.078 | 0.2% |
| M2P | device | 1 | 0.230 | 0.233 | 0.237 | 0.6% |
| L2P | device | 2 | 0.331 | 0.386 | 0.528 | 1.1% |
| P2P | device | 2 | 24.351 | 24.374 | 24.395 | 67.1% |
| load | | | 6.415 | 6.600 | 6.762 | |
| output | | | 15.009 | 15.324 | 15.509 | |
| other | | | | 0.207 | | |
| sum | | | | 36.302 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.686 ms on average.

##### cuda, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.058 s, of it tables 0.010 s (loaded from the cache) and device 2.233 s.
- Evaluation over 10 repeats, ms: min 153.651, median 153.907, mean 153.995, max 154.396, std 0.256; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.951 | 1.058 | 1.167 | 0.8% |
| M2M | device | 7 | 1.059 | 1.063 | 1.069 | 0.8% |
| M2L | device | 5 | 96.281 | 96.313 | 96.388 | 75.4% |
| L2L | device | 7 | 1.269 | 1.339 | 1.448 | 1.0% |
| P2L | device | 1 | 0.075 | 0.076 | 0.079 | 0.1% |
| M2P | device | 1 | 0.345 | 0.349 | 0.358 | 0.3% |
| L2P | device | 2 | 3.119 | 3.184 | 3.220 | 2.5% |
| P2P | device | 2 | 24.277 | 24.300 | 24.365 | 19.0% |
| load | | | 6.300 | 6.629 | 6.871 | |
| output | | | 15.064 | 15.347 | 15.984 | |
| other | | | | 0.209 | | |
| sum | | | | 127.683 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.664 ms on average.

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.993 s, of it tables 0.001 s (loaded from the cache) and device 0.210 s.
- Evaluation over 10 repeats, ms: min 10.036, median 10.165, mean 10.187, max 10.458, std 0.140; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.090 | 0.092 | 0.094 | 1.6% |
| M2M | device | 5 | 0.215 | 0.224 | 0.232 | 4.0% |
| M2L | device | 4 | 1.643 | 1.649 | 1.654 | 29.6% |
| L2L | device | 5 | 0.265 | 0.271 | 0.277 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.184 | 0.190 | 0.195 | 3.4% |
| P2P | device | 1 | 3.102 | 3.143 | 3.179 | 56.4% |
| load | | | 1.114 | 1.226 | 1.322 | |
| output | | | 3.211 | 3.405 | 3.588 | |
| other | | | | 0.033 | | |
| sum | | | | 5.569 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.315 ms on average.

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.022 s, of it tables 0.019 s (loaded from the cache) and device 0.208 s.
- Evaluation over 10 repeats, ms: min 25.416, median 25.604, mean 25.602, max 25.847, std 0.131; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.273 | 0.276 | 0.278 | 1.3% |
| M2M | device | 5 | 0.623 | 0.627 | 0.631 | 3.0% |
| M2L | device | 4 | 15.542 | 15.555 | 15.568 | 74.2% |
| L2L | device | 5 | 0.466 | 0.475 | 0.481 | 2.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.762 | 0.774 | 0.805 | 3.7% |
| P2P | device | 1 | 3.223 | 3.243 | 3.272 | 15.5% |
| load | | | 1.094 | 1.219 | 1.333 | |
| output | | | 3.292 | 3.473 | 3.961 | |
| other | | | | 0.032 | | |
| sum | | | | 20.950 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.335 ms on average.

##### cuda, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.352 s, of it tables 0.001 s (loaded from the cache) and device 2.214 s.
- Evaluation over 10 repeats, ms: min 102.855, median 103.426, mean 103.494, max 104.503, std 0.579; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.656 | 0.661 | 0.667 | 1.2% |
| M2M | device | 7 | 0.537 | 0.543 | 0.552 | 1.0% |
| M2L | device | 5 | 14.069 | 14.098 | 14.116 | 25.1% |
| L2L | device | 7 | 0.537 | 0.576 | 0.729 | 1.0% |
| P2L | device | 1 | 0.079 | 0.080 | 0.081 | 0.1% |
| M2P | device | 1 | 0.488 | 0.491 | 0.495 | 0.9% |
| L2P | device | 2 | 0.640 | 0.692 | 0.829 | 1.2% |
| P2P | device | 2 | 38.940 | 38.973 | 39.011 | 69.5% |
| load | | | 13.196 | 13.511 | 13.781 | |
| output | | | 28.143 | 28.735 | 29.142 | |
| other | | | | 0.209 | | |
| sum | | | | 56.116 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.721 ms on average.

##### cuda, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.761 s, of it tables 0.019 s (loaded from the cache) and device 2.256 s.
- Evaluation over 10 repeats, ms: min 268.669, median 269.199, mean 269.207, max 270.473, std 0.548; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 2.582 | 2.590 | 2.598 | 1.2% |
| M2M | device | 7 | 1.670 | 1.684 | 1.697 | 0.8% |
| M2L | device | 5 | 166.218 | 166.241 | 166.352 | 76.5% |
| L2L | device | 7 | 1.512 | 1.575 | 1.725 | 0.7% |
| P2L | device | 1 | 0.080 | 0.081 | 0.083 | 0.0% |
| M2P | device | 1 | 0.588 | 0.599 | 0.618 | 0.3% |
| L2P | device | 2 | 5.696 | 5.739 | 5.894 | 2.6% |
| P2P | device | 2 | 38.797 | 38.844 | 38.874 | 17.9% |
| load | | | 13.013 | 13.453 | 13.784 | |
| output | | | 28.261 | 28.702 | 29.263 | |
| other | | | | 0.212 | | |
| sum | | | | 217.353 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.753 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

### CUDA, after

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 10:40:22 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:40:22 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | a12d9d6-dirty on tbetcke/phase4s_t11-2 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8 --table-cache /data/ucahtbe/logs/t9/tables --output /data/ucahtbe/logs/t11/bench/after.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/data/ucahtbe/logs/t9/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Dense | 1.074 | 5.376 | 5.494 | 5.491 | 5.668 | 0.084 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 8 | Dense | 1.211 | 14.436 | 14.521 | 14.560 | 14.855 | 0.140 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f32 | 10000000 | 3 | Dense | 14.593 | 45.396 | 45.778 | 45.805 | 46.241 | 0.241 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f32 | 10000000 | 8 | Dense | 14.709 | 139.306 | 139.442 | 139.435 | 139.561 | 0.075 | 1.21e-5 | 2.26e-5 | yes |
| cuda | f64 | 1000000 | 3 | Dense | 0.987 | 7.678 | 7.787 | 8.356 | 10.738 | 1.198 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 8 | Dense | 1.016 | 23.104 | 23.213 | 23.241 | 23.466 | 0.108 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 10000000 | 3 | Dense | 14.189 | 73.729 | 74.104 | 74.094 | 74.451 | 0.231 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f64 | 10000000 | 8 | Dense | 14.464 | 239.769 | 240.215 | 240.204 | 240.673 | 0.334 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.059 | 0.196 | 1.280 | 0.290 | 0.000 | 0.000 | 0.067 | 1.990 | 0.609 | 1.318 | 0.031 | 3.882 | 6.124 | 0.707 |
| cuda | f32 | 1000000 | 8 | 0.121 | 0.525 | 9.386 | 0.636 | 0.000 | 0.000 | 0.430 | 2.146 | 0.627 | 1.361 | 0.035 | 13.244 | 15.747 | 0.910 |
| cuda | f32 | 10000000 | 3 | 0.406 | 0.530 | 9.734 | 0.629 | 0.074 | 0.236 | 0.394 | 24.392 | 3.639 | 3.705 | 0.209 | 36.395 | 47.212 | 0.795 |
| cuda | f32 | 10000000 | 8 | 1.115 | 1.057 | 96.320 | 1.366 | 0.075 | 0.352 | 3.173 | 24.308 | 3.640 | 3.803 | 0.210 | 127.765 | 141.296 | 0.916 |
| cuda | f64 | 1000000 | 3 | 0.093 | 0.225 | 1.648 | 0.271 | 0.000 | 0.000 | 0.190 | 3.134 | 0.753 | 1.420 | 0.031 | 5.561 | 8.397 | 0.665 |
| cuda | f64 | 1000000 | 8 | 0.276 | 0.643 | 15.556 | 0.473 | 0.000 | 0.000 | 0.771 | 3.231 | 0.744 | 1.452 | 0.033 | 20.949 | 24.419 | 0.901 |
| cuda | f64 | 10000000 | 3 | 0.662 | 0.536 | 14.099 | 0.576 | 0.080 | 0.492 | 0.790 | 38.971 | 7.265 | 5.699 | 0.211 | 56.206 | 75.945 | 0.759 |
| cuda | f64 | 10000000 | 8 | 2.588 | 1.670 | 166.219 | 1.623 | 0.081 | 0.601 | 5.733 | 38.854 | 7.295 | 6.665 | 0.212 | 217.369 | 243.197 | 0.905 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.074 s, of it tables 0.000 s (loaded from the cache) and device 0.274 s.
- Evaluation over 10 repeats, ms: min 5.376, median 5.494, mean 5.491, max 5.668, std 0.084; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.057 | 0.059 | 0.062 | 1.5% |
| M2M | device | 5 | 0.172 | 0.196 | 0.217 | 5.0% |
| M2L | device | 4 | 1.270 | 1.280 | 1.291 | 33.0% |
| L2L | device | 5 | 0.277 | 0.290 | 0.302 | 7.5% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.062 | 0.067 | 0.076 | 1.7% |
| P2P | device | 1 | 1.969 | 1.990 | 2.021 | 51.3% |
| load | | | 0.577 | 0.609 | 0.662 | |
| output | | | 1.230 | 1.318 | 1.465 | |
| other | | | | 0.031 | | |
| sum | | | | 3.882 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.276 ms on average.

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.211 s, of it tables 0.010 s (loaded from the cache) and device 0.402 s.
- Evaluation over 10 repeats, ms: min 14.436, median 14.521, mean 14.560, max 14.855, std 0.140; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.119 | 0.121 | 0.124 | 0.9% |
| M2M | device | 5 | 0.520 | 0.525 | 0.528 | 4.0% |
| M2L | device | 4 | 9.372 | 9.386 | 9.396 | 70.9% |
| L2L | device | 5 | 0.628 | 0.636 | 0.643 | 4.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.421 | 0.430 | 0.439 | 3.2% |
| P2P | device | 1 | 2.122 | 2.146 | 2.167 | 16.2% |
| load | | | 0.559 | 0.627 | 0.688 | |
| output | | | 1.170 | 1.361 | 1.854 | |
| other | | | | 0.035 | | |
| sum | | | | 13.244 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.259 ms on average.

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.593 s, of it tables 0.001 s (loaded from the cache) and device 2.235 s.
- Evaluation over 10 repeats, ms: min 45.396, median 45.778, mean 45.805, max 46.241, std 0.241; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.400 | 0.406 | 0.413 | 1.1% |
| M2M | device | 7 | 0.372 | 0.530 | 0.573 | 1.5% |
| M2L | device | 5 | 9.720 | 9.734 | 9.758 | 26.7% |
| L2L | device | 7 | 0.588 | 0.629 | 0.795 | 1.7% |
| P2L | device | 1 | 0.074 | 0.074 | 0.075 | 0.2% |
| M2P | device | 1 | 0.231 | 0.236 | 0.241 | 0.6% |
| L2P | device | 2 | 0.354 | 0.394 | 0.523 | 1.1% |
| P2P | device | 2 | 24.366 | 24.392 | 24.426 | 67.0% |
| load | | | 3.588 | 3.639 | 3.788 | |
| output | | | 3.567 | 3.705 | 3.892 | |
| other | | | | 0.209 | | |
| sum | | | | 36.395 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.160 ms on average.

##### cuda, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.709 s, of it tables 0.010 s (loaded from the cache) and device 2.209 s.
- Evaluation over 10 repeats, ms: min 139.306, median 139.442, mean 139.435, max 139.561, std 0.075; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.952 | 1.115 | 1.161 | 0.9% |
| M2M | device | 7 | 1.047 | 1.057 | 1.072 | 0.8% |
| M2L | device | 5 | 96.261 | 96.320 | 96.399 | 75.4% |
| L2L | device | 7 | 1.299 | 1.366 | 1.480 | 1.1% |
| P2L | device | 1 | 0.074 | 0.075 | 0.077 | 0.1% |
| M2P | device | 1 | 0.346 | 0.352 | 0.362 | 0.3% |
| L2P | device | 2 | 3.139 | 3.173 | 3.190 | 2.5% |
| P2P | device | 2 | 24.281 | 24.308 | 24.357 | 19.0% |
| load | | | 3.534 | 3.640 | 3.882 | |
| output | | | 3.633 | 3.803 | 4.060 | |
| other | | | | 0.210 | | |
| sum | | | | 127.765 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.163 ms on average.

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.987 s, of it tables 0.001 s (loaded from the cache) and device 0.201 s.
- Evaluation over 10 repeats, ms: min 7.678, median 7.787, mean 8.356, max 10.738, std 1.198; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.092 | 0.093 | 0.095 | 1.7% |
| M2M | device | 5 | 0.221 | 0.225 | 0.231 | 4.1% |
| M2L | device | 4 | 1.641 | 1.648 | 1.654 | 29.6% |
| L2L | device | 5 | 0.268 | 0.271 | 0.274 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.182 | 0.190 | 0.195 | 3.4% |
| P2P | device | 1 | 3.095 | 3.134 | 3.158 | 56.4% |
| load | | | 0.629 | 0.753 | 0.815 | |
| output | | | 1.196 | 1.420 | 1.584 | |
| other | | | | 0.031 | | |
| sum | | | | 5.561 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.305 ms on average.

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.016 s, of it tables 0.019 s (loaded from the cache) and device 0.209 s.
- Evaluation over 10 repeats, ms: min 23.104, median 23.213, mean 23.241, max 23.466, std 0.108; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.274 | 0.276 | 0.277 | 1.3% |
| M2M | device | 5 | 0.622 | 0.643 | 0.687 | 3.1% |
| M2L | device | 4 | 15.547 | 15.556 | 15.567 | 74.3% |
| L2L | device | 5 | 0.464 | 0.473 | 0.478 | 2.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.757 | 0.771 | 0.787 | 3.7% |
| P2P | device | 1 | 3.205 | 3.231 | 3.285 | 15.4% |
| load | | | 0.603 | 0.744 | 0.818 | |
| output | | | 1.309 | 1.452 | 1.586 | |
| other | | | | 0.033 | | |
| sum | | | | 20.949 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.286 ms on average.

##### cuda, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.189 s, of it tables 0.001 s (loaded from the cache) and device 2.219 s.
- Evaluation over 10 repeats, ms: min 73.729, median 74.104, mean 74.094, max 74.451, std 0.231; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.655 | 0.662 | 0.669 | 1.2% |
| M2M | device | 7 | 0.512 | 0.536 | 0.545 | 1.0% |
| M2L | device | 5 | 14.081 | 14.099 | 14.121 | 25.1% |
| L2L | device | 7 | 0.532 | 0.576 | 0.719 | 1.0% |
| P2L | device | 1 | 0.077 | 0.080 | 0.083 | 0.1% |
| M2P | device | 1 | 0.475 | 0.492 | 0.502 | 0.9% |
| L2P | device | 2 | 0.740 | 0.790 | 0.820 | 1.4% |
| P2P | device | 2 | 38.920 | 38.971 | 39.045 | 69.3% |
| load | | | 7.179 | 7.265 | 7.519 | |
| output | | | 5.383 | 5.699 | 5.962 | |
| other | | | | 0.211 | | |
| sum | | | | 56.206 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.801 ms on average.

##### cuda, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.464 s, of it tables 0.019 s (loaded from the cache) and device 2.270 s.
- Evaluation over 10 repeats, ms: min 239.769, median 240.215, mean 240.204, max 240.673, std 0.334; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 2.577 | 2.588 | 2.608 | 1.2% |
| M2M | device | 7 | 1.664 | 1.670 | 1.687 | 0.8% |
| M2L | device | 5 | 166.200 | 166.219 | 166.234 | 76.5% |
| L2L | device | 7 | 1.540 | 1.623 | 1.678 | 0.7% |
| P2L | device | 1 | 0.078 | 0.081 | 0.085 | 0.0% |
| M2P | device | 1 | 0.592 | 0.601 | 0.615 | 0.3% |
| L2P | device | 2 | 5.699 | 5.733 | 5.882 | 2.6% |
| P2P | device | 2 | 38.832 | 38.854 | 38.879 | 17.9% |
| load | | | 7.113 | 7.295 | 7.491 | |
| output | | | 5.559 | 6.665 | 15.340 | |
| other | | | | 0.212 | | |
| sum | | | | 217.369 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.805 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

### CUDA, before, `--reuse-output`

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 10:46:05 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:46:05 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | a12d9d6 on detached-a12d9d6 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8 --reuse-output --table-cache /data/ucahtbe/logs/t9/tables --output /data/ucahtbe/logs/t11/bench/before-reuse.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, `--reuse-output`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/data/ucahtbe/logs/t9/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Dense | 1.061 | 6.481 | 6.604 | 6.698 | 7.448 | 0.287 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 8 | Dense | 1.218 | 15.310 | 15.547 | 15.592 | 15.821 | 0.173 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f32 | 10000000 | 3 | Dense | 15.031 | 59.381 | 59.823 | 59.809 | 60.079 | 0.177 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f32 | 10000000 | 8 | Dense | 14.832 | 153.131 | 153.438 | 153.471 | 154.037 | 0.240 | 1.21e-5 | 2.26e-5 | yes |
| cuda | f64 | 1000000 | 3 | Dense | 0.998 | 10.081 | 10.214 | 10.266 | 10.797 | 0.200 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 8 | Dense | 1.031 | 25.418 | 25.663 | 25.683 | 25.953 | 0.167 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 10000000 | 3 | Dense | 14.614 | 102.419 | 102.591 | 102.649 | 103.278 | 0.249 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f64 | 10000000 | 8 | Dense | 14.869 | 267.816 | 268.601 | 268.603 | 269.263 | 0.399 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.058 | 0.185 | 1.278 | 0.289 | 0.000 | 0.000 | 0.064 | 2.006 | 0.726 | 2.190 | 0.028 | 3.879 | 7.046 | 0.579 |
| cuda | f32 | 1000000 | 8 | 0.121 | 0.524 | 9.383 | 0.640 | 0.000 | 0.000 | 0.431 | 2.133 | 0.682 | 2.270 | 0.032 | 13.234 | 16.676 | 0.849 |
| cuda | f32 | 10000000 | 3 | 0.402 | 0.484 | 9.726 | 0.628 | 0.074 | 0.234 | 0.401 | 24.375 | 6.465 | 14.892 | 0.209 | 36.323 | 61.163 | 0.607 |
| cuda | f32 | 10000000 | 8 | 1.096 | 1.055 | 96.297 | 1.389 | 0.075 | 0.352 | 3.197 | 24.299 | 6.462 | 14.889 | 0.210 | 127.759 | 155.134 | 0.832 |
| cuda | f64 | 1000000 | 3 | 0.092 | 0.225 | 1.651 | 0.271 | 0.000 | 0.000 | 0.192 | 3.121 | 1.204 | 3.298 | 0.032 | 5.551 | 10.646 | 0.541 |
| cuda | f64 | 1000000 | 8 | 0.275 | 0.629 | 15.554 | 0.477 | 0.000 | 0.000 | 0.774 | 3.229 | 1.252 | 3.314 | 0.033 | 20.937 | 26.636 | 0.815 |
| cuda | f64 | 10000000 | 3 | 0.661 | 0.535 | 14.101 | 0.584 | 0.080 | 0.491 | 0.758 | 38.998 | 13.380 | 28.308 | 0.209 | 56.208 | 104.668 | 0.548 |
| cuda | f64 | 10000000 | 8 | 2.487 | 1.678 | 166.217 | 1.592 | 0.080 | 0.599 | 5.799 | 38.862 | 13.087 | 28.134 | 0.213 | 217.313 | 270.469 | 0.809 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.061 s, of it tables 0.000 s (loaded from the cache) and device 0.267 s.
- Evaluation over 10 repeats, ms: min 6.481, median 6.604, mean 6.698, max 7.448, std 0.287; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.056 | 0.058 | 0.063 | 1.5% |
| M2M | device | 5 | 0.174 | 0.185 | 0.217 | 4.8% |
| M2L | device | 4 | 1.272 | 1.278 | 1.290 | 32.9% |
| L2L | device | 5 | 0.279 | 0.289 | 0.303 | 7.4% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.062 | 0.064 | 0.066 | 1.6% |
| P2P | device | 1 | 1.993 | 2.006 | 2.023 | 51.7% |
| load | | | 0.598 | 0.726 | 0.926 | |
| output | | | 2.073 | 2.190 | 2.350 | |
| other | | | | 0.028 | | |
| sum | | | | 3.879 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.209 ms on average.

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.218 s, of it tables 0.010 s (loaded from the cache) and device 0.407 s.
- Evaluation over 10 repeats, ms: min 15.310, median 15.547, mean 15.592, max 15.821, std 0.173; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.118 | 0.121 | 0.123 | 0.9% |
| M2M | device | 5 | 0.520 | 0.524 | 0.531 | 4.0% |
| M2L | device | 4 | 9.371 | 9.383 | 9.391 | 70.9% |
| L2L | device | 5 | 0.635 | 0.640 | 0.644 | 4.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.424 | 0.431 | 0.439 | 3.3% |
| P2P | device | 1 | 2.123 | 2.133 | 2.145 | 16.1% |
| load | | | 0.564 | 0.682 | 0.744 | |
| output | | | 2.188 | 2.270 | 2.345 | |
| other | | | | 0.032 | | |
| sum | | | | 13.234 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.257 ms on average.

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 15.031 s, of it tables 0.001 s (loaded from the cache) and device 2.252 s.
- Evaluation over 10 repeats, ms: min 59.381, median 59.823, mean 59.809, max 60.079, std 0.177; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.398 | 0.402 | 0.408 | 1.1% |
| M2M | device | 7 | 0.363 | 0.484 | 0.565 | 1.3% |
| M2L | device | 5 | 9.718 | 9.726 | 9.738 | 26.8% |
| L2L | device | 7 | 0.616 | 0.628 | 0.645 | 1.7% |
| P2L | device | 1 | 0.073 | 0.074 | 0.075 | 0.2% |
| M2P | device | 1 | 0.229 | 0.234 | 0.241 | 0.6% |
| L2P | device | 2 | 0.370 | 0.401 | 0.426 | 1.1% |
| P2P | device | 2 | 24.344 | 24.375 | 24.401 | 67.1% |
| load | | | 6.246 | 6.465 | 6.711 | |
| output | | | 14.535 | 14.892 | 15.082 | |
| other | | | | 0.209 | | |
| sum | | | | 36.323 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.689 ms on average.

##### cuda, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.832 s, of it tables 0.010 s (loaded from the cache) and device 2.218 s.
- Evaluation over 10 repeats, ms: min 153.131, median 153.438, mean 153.471, max 154.037, std 0.240; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.949 | 1.096 | 1.162 | 0.9% |
| M2M | device | 7 | 1.050 | 1.055 | 1.060 | 0.8% |
| M2L | device | 5 | 96.276 | 96.297 | 96.360 | 75.4% |
| L2L | device | 7 | 1.323 | 1.389 | 1.466 | 1.1% |
| P2L | device | 1 | 0.073 | 0.075 | 0.077 | 0.1% |
| M2P | device | 1 | 0.347 | 0.352 | 0.361 | 0.3% |
| L2P | device | 2 | 3.175 | 3.197 | 3.235 | 2.5% |
| P2P | device | 2 | 24.273 | 24.299 | 24.364 | 19.0% |
| load | | | 6.192 | 6.462 | 6.752 | |
| output | | | 14.539 | 14.889 | 15.055 | |
| other | | | | 0.210 | | |
| sum | | | | 127.759 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 12.639 ms on average.

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.998 s, of it tables 0.001 s (loaded from the cache) and device 0.209 s.
- Evaluation over 10 repeats, ms: min 10.081, median 10.214, mean 10.266, max 10.797, std 0.200; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.090 | 0.092 | 0.093 | 1.6% |
| M2M | device | 5 | 0.221 | 0.225 | 0.232 | 4.1% |
| M2L | device | 4 | 1.641 | 1.651 | 1.659 | 29.7% |
| L2L | device | 5 | 0.264 | 0.271 | 0.275 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.184 | 0.192 | 0.199 | 3.5% |
| P2P | device | 1 | 3.106 | 3.121 | 3.156 | 56.2% |
| load | | | 1.127 | 1.204 | 1.319 | |
| output | | | 3.213 | 3.298 | 3.450 | |
| other | | | | 0.032 | | |
| sum | | | | 5.551 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.223 ms on average.

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.031 s, of it tables 0.019 s (loaded from the cache) and device 0.208 s.
- Evaluation over 10 repeats, ms: min 25.418, median 25.663, mean 25.683, max 25.953, std 0.167; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.274 | 0.275 | 0.277 | 1.3% |
| M2M | device | 5 | 0.622 | 0.629 | 0.639 | 3.0% |
| M2L | device | 4 | 15.543 | 15.554 | 15.563 | 74.3% |
| L2L | device | 5 | 0.466 | 0.477 | 0.491 | 2.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.758 | 0.774 | 0.789 | 3.7% |
| P2P | device | 1 | 3.209 | 3.229 | 3.278 | 15.4% |
| load | | | 1.112 | 1.252 | 1.355 | |
| output | | | 3.125 | 3.314 | 3.492 | |
| other | | | | 0.033 | | |
| sum | | | | 20.937 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 2.300 ms on average.

##### cuda, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.614 s, of it tables 0.001 s (loaded from the cache) and device 2.241 s.
- Evaluation over 10 repeats, ms: min 102.419, median 102.591, mean 102.649, max 103.278, std 0.249; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.654 | 0.661 | 0.666 | 1.2% |
| M2M | device | 7 | 0.529 | 0.535 | 0.542 | 1.0% |
| M2L | device | 5 | 14.087 | 14.101 | 14.114 | 25.1% |
| L2L | device | 7 | 0.530 | 0.584 | 0.733 | 1.0% |
| P2L | device | 1 | 0.079 | 0.080 | 0.081 | 0.1% |
| M2P | device | 1 | 0.488 | 0.491 | 0.497 | 0.9% |
| L2P | device | 2 | 0.626 | 0.758 | 0.821 | 1.3% |
| P2P | device | 2 | 38.966 | 38.998 | 39.029 | 69.4% |
| load | | | 13.032 | 13.380 | 13.799 | |
| output | | | 27.753 | 28.308 | 28.605 | |
| other | | | | 0.209 | | |
| sum | | | | 56.208 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.886 ms on average.

##### cuda, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.869 s, of it tables 0.019 s (loaded from the cache) and device 2.249 s.
- Evaluation over 10 repeats, ms: min 267.816, median 268.601, mean 268.603, max 269.263, std 0.399; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 2.376 | 2.487 | 2.617 | 1.1% |
| M2M | device | 7 | 1.672 | 1.678 | 1.685 | 0.8% |
| M2L | device | 5 | 166.198 | 166.217 | 166.237 | 76.5% |
| L2L | device | 7 | 1.521 | 1.592 | 1.726 | 0.7% |
| P2L | device | 1 | 0.079 | 0.080 | 0.081 | 0.0% |
| M2P | device | 1 | 0.593 | 0.599 | 0.611 | 0.3% |
| L2P | device | 2 | 5.705 | 5.799 | 5.891 | 2.7% |
| P2P | device | 2 | 38.813 | 38.862 | 38.914 | 17.9% |
| load | | | 12.660 | 13.087 | 13.989 | |
| output | | | 27.573 | 28.134 | 28.950 | |
| other | | | | 0.213 | | |
| sum | | | | 217.313 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 24.712 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate_into` (one reused `Output`), with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

### CUDA, after, `--reuse-output`

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 10:51:53 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:51:53 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | a12d9d6-dirty on tbetcke/phase4s_t11-2 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e6,1e7 --precision f32,f64 --degree 3,8 --reuse-output --table-cache /data/ucahtbe/logs/t9/tables --output /data/ucahtbe/logs/t11/bench/after-reuse.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, `--reuse-output`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/data/ucahtbe/logs/t9/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 1000000 | 3 | Dense | 1.083 | 5.418 | 5.573 | 5.586 | 5.769 | 0.123 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f32 | 1000000 | 8 | Dense | 1.211 | 14.334 | 14.416 | 14.476 | 14.932 | 0.178 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f32 | 10000000 | 3 | Dense | 15.560 | 45.500 | 45.586 | 45.662 | 46.294 | 0.237 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f32 | 10000000 | 8 | Dense | 14.636 | 138.620 | 139.108 | 142.674 | 175.546 | 11.552 | 1.21e-5 | 2.26e-5 | yes |
| cuda | f64 | 1000000 | 3 | Dense | 1.010 | 7.704 | 7.771 | 7.831 | 8.285 | 0.173 | 2.00e-3 | 3.80e-3 | yes |
| cuda | f64 | 1000000 | 8 | Dense | 1.020 | 23.056 | 23.179 | 23.211 | 23.582 | 0.156 | 1.36e-5 | 4.59e-5 | yes |
| cuda | f64 | 10000000 | 3 | Dense | 14.427 | 73.370 | 73.665 | 73.957 | 76.745 | 0.996 | 1.87e-3 | 1.87e-3 | yes |
| cuda | f64 | 10000000 | 8 | Dense | 14.582 | 239.434 | 239.668 | 239.677 | 240.121 | 0.221 | 1.21e-5 | 2.26e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 1000000 | 3 | 0.063 | 0.217 | 1.289 | 0.307 | 0.000 | 0.000 | 0.069 | 1.995 | 0.602 | 1.383 | 0.032 | 3.941 | 6.227 | 0.705 |
| cuda | f32 | 1000000 | 8 | 0.121 | 0.492 | 9.386 | 0.643 | 0.000 | 0.000 | 0.433 | 2.141 | 0.625 | 1.442 | 0.031 | 13.216 | 15.793 | 0.913 |
| cuda | f32 | 10000000 | 3 | 0.408 | 0.454 | 9.736 | 0.626 | 0.074 | 0.237 | 0.394 | 24.378 | 3.731 | 3.422 | 0.206 | 36.306 | 46.925 | 0.795 |
| cuda | f32 | 10000000 | 8 | 1.137 | 1.066 | 96.320 | 1.378 | 0.076 | 0.353 | 3.196 | 24.293 | 3.803 | 3.354 | 0.209 | 127.819 | 141.002 | 0.896 |
| cuda | f64 | 1000000 | 3 | 0.090 | 0.215 | 1.644 | 0.264 | 0.000 | 0.000 | 0.190 | 3.137 | 0.701 | 1.416 | 0.032 | 5.539 | 8.322 | 0.707 |
| cuda | f64 | 1000000 | 8 | 0.276 | 0.629 | 15.560 | 0.475 | 0.000 | 0.000 | 0.765 | 3.230 | 0.855 | 1.382 | 0.033 | 20.935 | 24.433 | 0.902 |
| cuda | f64 | 10000000 | 3 | 0.667 | 0.546 | 14.104 | 0.605 | 0.079 | 0.496 | 0.668 | 38.973 | 7.358 | 5.196 | 0.209 | 56.137 | 75.475 | 0.759 |
| cuda | f64 | 10000000 | 8 | 2.591 | 1.681 | 166.246 | 1.570 | 0.081 | 0.602 | 5.744 | 38.854 | 7.345 | 5.222 | 0.214 | 217.368 | 241.818 | 0.907 |

##### cuda, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.083 s, of it tables 0.000 s (loaded from the cache) and device 0.263 s.
- Evaluation over 10 repeats, ms: min 5.418, median 5.573, mean 5.586, max 5.769, std 0.123; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.057 | 0.063 | 0.067 | 1.6% |
| M2M | device | 5 | 0.180 | 0.217 | 0.231 | 5.5% |
| M2L | device | 4 | 1.271 | 1.289 | 1.298 | 32.7% |
| L2L | device | 5 | 0.287 | 0.307 | 0.325 | 7.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.064 | 0.069 | 0.080 | 1.7% |
| P2P | device | 1 | 1.975 | 1.995 | 2.021 | 50.6% |
| load | | | 0.542 | 0.602 | 0.691 | |
| output | | | 1.206 | 1.383 | 1.495 | |
| other | | | | 0.032 | | |
| sum | | | | 3.941 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.272 ms on average.

##### cuda, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.211 s, of it tables 0.010 s (loaded from the cache) and device 0.404 s.
- Evaluation over 10 repeats, ms: min 14.334, median 14.416, mean 14.476, max 14.932, std 0.178; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.118 | 0.121 | 0.124 | 0.9% |
| M2M | device | 5 | 0.335 | 0.492 | 0.543 | 3.7% |
| M2L | device | 4 | 9.378 | 9.386 | 9.394 | 71.0% |
| L2L | device | 5 | 0.637 | 0.643 | 0.650 | 4.9% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.419 | 0.433 | 0.444 | 3.3% |
| P2P | device | 1 | 2.123 | 2.141 | 2.162 | 16.2% |
| load | | | 0.475 | 0.625 | 0.783 | |
| output | | | 1.245 | 1.442 | 2.034 | |
| other | | | | 0.031 | | |
| sum | | | | 13.216 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.264 ms on average.

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 15.560 s, of it tables 0.001 s (loaded from the cache) and device 2.238 s.
- Evaluation over 10 repeats, ms: min 45.500, median 45.586, mean 45.662, max 46.294, std 0.237; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.404 | 0.408 | 0.413 | 1.1% |
| M2M | device | 7 | 0.370 | 0.454 | 0.576 | 1.2% |
| M2L | device | 5 | 9.727 | 9.736 | 9.763 | 26.8% |
| L2L | device | 7 | 0.595 | 0.626 | 0.649 | 1.7% |
| P2L | device | 1 | 0.072 | 0.074 | 0.075 | 0.2% |
| M2P | device | 1 | 0.230 | 0.237 | 0.243 | 0.7% |
| L2P | device | 2 | 0.336 | 0.394 | 0.420 | 1.1% |
| P2P | device | 2 | 24.343 | 24.378 | 24.423 | 67.1% |
| load | | | 3.591 | 3.731 | 4.116 | |
| output | | | 3.297 | 3.422 | 3.576 | |
| other | | | | 0.206 | | |
| sum | | | | 36.306 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.161 ms on average.

##### cuda, f32, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.636 s, of it tables 0.010 s (loaded from the cache) and device 2.197 s.
- Evaluation over 10 repeats, ms: min 138.620, median 139.108, mean 142.674, max 175.546, std 11.552; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.960 | 1.137 | 1.165 | 0.9% |
| M2M | device | 7 | 1.060 | 1.066 | 1.070 | 0.8% |
| M2L | device | 5 | 96.288 | 96.320 | 96.407 | 75.4% |
| L2L | device | 7 | 1.321 | 1.378 | 1.471 | 1.1% |
| P2L | device | 1 | 0.075 | 0.076 | 0.078 | 0.1% |
| M2P | device | 1 | 0.341 | 0.353 | 0.362 | 0.3% |
| L2P | device | 2 | 3.159 | 3.196 | 3.220 | 2.5% |
| P2P | device | 2 | 24.266 | 24.293 | 24.320 | 19.0% |
| load | | | 3.687 | 3.803 | 3.938 | |
| output | | | 3.215 | 3.354 | 3.677 | |
| other | | | | 0.209 | | |
| sum | | | | 127.819 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.163 ms on average.

##### cuda, f64, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.010 s, of it tables 0.001 s (loaded from the cache) and device 0.206 s.
- Evaluation over 10 repeats, ms: min 7.704, median 7.771, mean 7.831, max 8.285, std 0.173; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.085 | 0.090 | 0.095 | 1.6% |
| M2M | device | 5 | 0.190 | 0.215 | 0.235 | 3.9% |
| M2L | device | 4 | 1.628 | 1.644 | 1.662 | 29.7% |
| L2L | device | 5 | 0.253 | 0.264 | 0.279 | 4.8% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.182 | 0.190 | 0.196 | 3.4% |
| P2P | device | 1 | 3.115 | 3.137 | 3.165 | 56.6% |
| load | | | 0.614 | 0.701 | 0.787 | |
| output | | | 1.246 | 1.416 | 1.795 | |
| other | | | | 0.032 | | |
| sum | | | | 5.539 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.267 ms on average.

##### cuda, f64, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.020 s, of it tables 0.019 s (loaded from the cache) and device 0.207 s.
- Evaluation over 10 repeats, ms: min 23.056, median 23.179, mean 23.211, max 23.582, std 0.156; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.274 | 0.276 | 0.277 | 1.3% |
| M2M | device | 5 | 0.621 | 0.629 | 0.659 | 3.0% |
| M2L | device | 4 | 15.554 | 15.560 | 15.573 | 74.3% |
| L2L | device | 5 | 0.471 | 0.475 | 0.480 | 2.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.757 | 0.765 | 0.776 | 3.7% |
| P2P | device | 1 | 3.201 | 3.230 | 3.250 | 15.4% |
| load | | | 0.636 | 0.855 | 1.008 | |
| output | | | 1.244 | 1.382 | 1.544 | |
| other | | | | 0.033 | | |
| sum | | | | 20.935 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.304 ms on average.

##### cuda, f64, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.427 s, of it tables 0.001 s (loaded from the cache) and device 2.255 s.
- Evaluation over 10 repeats, ms: min 73.370, median 73.665, mean 73.957, max 76.745, std 0.996; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 0.663 | 0.667 | 0.671 | 1.2% |
| M2M | device | 7 | 0.526 | 0.546 | 0.556 | 1.0% |
| M2L | device | 5 | 14.090 | 14.104 | 14.119 | 25.1% |
| L2L | device | 7 | 0.541 | 0.605 | 0.727 | 1.1% |
| P2L | device | 1 | 0.077 | 0.079 | 0.080 | 0.1% |
| M2P | device | 1 | 0.488 | 0.496 | 0.504 | 0.9% |
| L2P | device | 2 | 0.624 | 0.668 | 0.803 | 1.2% |
| P2P | device | 2 | 38.943 | 38.973 | 39.011 | 69.4% |
| load | | | 7.251 | 7.358 | 7.482 | |
| output | | | 4.985 | 5.196 | 5.337 | |
| other | | | | 0.209 | | |
| sum | | | | 56.137 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.800 ms on average.

##### cuda, f64, N = 10000000, p = 8

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.582 s, of it tables 0.019 s (loaded from the cache) and device 2.249 s.
- Evaluation over 10 repeats, ms: min 239.434, median 239.668, mean 239.677, max 240.121, std 0.221; bit-identical across the repeats: yes.
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
| P2M | device | 2 | 2.586 | 2.591 | 2.595 | 1.2% |
| M2M | device | 7 | 1.676 | 1.681 | 1.686 | 0.8% |
| M2L | device | 5 | 166.230 | 166.246 | 166.261 | 76.5% |
| L2L | device | 7 | 1.514 | 1.570 | 1.704 | 0.7% |
| P2L | device | 1 | 0.080 | 0.081 | 0.083 | 0.0% |
| M2P | device | 1 | 0.594 | 0.602 | 0.617 | 0.3% |
| L2P | device | 2 | 5.693 | 5.744 | 5.868 | 2.6% |
| P2P | device | 2 | 38.819 | 38.854 | 38.919 | 17.9% |
| load | | | 7.268 | 7.345 | 7.523 | |
| output | | | 5.022 | 5.222 | 5.704 | |
| other | | | | 0.214 | | |
| sum | | | | 217.368 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.802 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate_into` (one reused `Output`), with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: cuda (f32, f64); refused or failed: none.

Measured on locust.rc.ucl.ac.uk; never asserted.

### CUDA under `nsys`, after

#### FMM benchmark: locust.rc.ucl.ac.uk, 2026-10-08 10:57:38 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:57:38 UTC |
| host | locust.rc.ucl.ac.uk |
| CPU | Neoverse-V2 (implementer 0x41, part 0xd4f); 72 physical, 72 logical |
| devices | cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64 |
| CUDA | nvidia-smi: NVIDIA GH200 480GB, 565.57.01, 97871 MiB, 1980 MHz, 1980 MHz (name, driver, memory, SM clock, max SM clock); nvcc: Cuda compilation tools, release 12.6, V12.6.85 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-linux, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, cuda |
| source revision | a12d9d6-dirty on tbetcke/phase4s_t11-2 (.source-revision) |
| command | `nd-fmm-bench --backend cuda --n 1e7 --precision f32 --degree 3 --repeats 3 --warmup 1 --kinds off --accuracy off --table-cache /data/ucahtbe/logs/t9/tables --output /data/ucahtbe/logs/t11/bench/nsys-after.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 72 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 1 warm-up, then 3 timed evaluations; kind timings `off` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | off |
| table cache | `/data/ucahtbe/logs/t9/tables` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 3 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuda | f32 | 10000000 | 3 | Dense | 14.892 | 46.032 | 46.141 | 46.148 | 46.272 | 0.120 | – | – | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `off` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | f32 | 10000000 | 3 | off | – | – | – | – | – | – | – | – | – | – | – | – | – |

##### cuda, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 72; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 14.892 s, of it tables 0.001 s (loaded from the cache) and device 2.526 s.
- Evaluation over 3 repeats, ms: min 46.032, median 46.141, mean 46.148, max 46.272, std 0.120; bit-identical across the repeats: yes.
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

### Metal (M3 Max), before, N = 10⁶

#### FMM benchmark: Timos-MBP, 2026-10-08 10:51:42 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:51:42 UTC |
| host | Timos-MBP |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | a12d9d6 (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --n 1e6 --degree 3,8 --table-cache /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal --output /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/metal-bench/before-1e6.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 1000000 | 3 | Dense | 0.843 | 14.060 | 14.195 | 14.224 | 14.495 | 0.140 | 2.00e-3 | 3.80e-3 | yes |
| metal | f32 | 1000000 | 8 | Dense | 1.068 | 79.394 | 80.621 | 80.513 | 81.207 | 0.623 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 1000000 | 3 | 0.547 | 1.806 | 6.820 | 1.692 | 0.000 | 0.000 | 0.446 | 5.181 | 1.493 | 2.306 | 0.027 | 16.493 | 20.419 | 1.159 |
| metal | f32 | 1000000 | 8 | 1.158 | 1.986 | 71.219 | 2.019 | 0.000 | 0.000 | 0.719 | 5.083 | 1.581 | 2.819 | 0.027 | 82.185 | 86.822 | 1.021 |

##### metal, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.843 s, of it tables 0.020 s (built) and device 0.182 s.
- Evaluation over 10 repeats, ms: min 14.060, median 14.195, mean 14.224, max 14.495, std 0.140; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.480 | 0.547 | 0.620 | 3.3% |
| M2M | device | 5 | 1.745 | 1.806 | 1.895 | 11.0% |
| M2L | device | 4 | 6.745 | 6.820 | 6.914 | 41.4% |
| L2L | device | 5 | 1.555 | 1.692 | 2.056 | 10.3% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.382 | 0.446 | 0.670 | 2.7% |
| P2P | device | 1 | 4.949 | 5.181 | 5.448 | 31.4% |
| load | | | 1.373 | 1.493 | 1.665 | |
| output | | | 1.972 | 2.306 | 2.626 | |
| other | | | | 0.027 | | |
| sum | | | | 16.493 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.756 ms on average.

##### metal, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 1.068 s, of it tables 0.261 s (built) and device 0.161 s.
- Evaluation over 10 repeats, ms: min 79.394, median 80.621, mean 80.513, max 81.207, std 0.623; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 1.023 | 1.158 | 1.244 | 1.4% |
| M2M | device | 5 | 1.907 | 1.986 | 2.218 | 2.4% |
| M2L | device | 4 | 70.970 | 71.219 | 71.525 | 86.7% |
| L2L | device | 5 | 1.910 | 2.019 | 2.131 | 2.5% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.597 | 0.719 | 0.964 | 0.9% |
| P2P | device | 1 | 5.016 | 5.083 | 5.237 | 6.2% |
| load | | | 1.465 | 1.581 | 1.696 | |
| output | | | 2.684 | 2.819 | 2.932 | |
| other | | | | 0.027 | | |
| sum | | | | 82.185 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.716 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MBP; never asserted.

### Metal (M3 Max), before, N = 10⁷

#### FMM benchmark: Timos-MBP, 2026-10-08 10:51:54 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:51:54 UTC |
| host | Timos-MBP |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | a12d9d6 (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --n 1e7 --degree 3 --table-cache /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal --output /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/metal-bench/before-1e7.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 10000000 | 3 | Dense | 8.859 | 219.979 | 221.516 | 222.932 | 230.989 | 3.597 | 1.87e-3 | 1.87e-3 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 10000000 | 3 | 2.496 | 4.271 | 111.669 | 2.821 | 0.396 | 1.547 | 1.933 | 57.748 | 15.415 | 33.261 | 0.116 | 182.880 | 232.619 | 0.820 |

##### metal, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 8.859 s, of it tables 0.002 s (loaded from the cache) and device 1.784 s.
- Evaluation over 10 repeats, ms: min 219.979, median 221.516, mean 222.932, max 230.989, std 3.597; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(16 x 4 units, 4 columns per unit); rotation cube (32 units); device memory 1841.2 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 216, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 6 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 45 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 1.950 | 2.496 | 3.308 | 1.4% |
| M2M | device | 7 | 3.151 | 4.271 | 5.462 | 2.3% |
| M2L | device | 5 | 111.294 | 111.669 | 112.419 | 61.1% |
| L2L | device | 7 | 2.613 | 2.821 | 3.356 | 1.5% |
| P2L | device | 1 | 0.368 | 0.396 | 0.504 | 0.2% |
| M2P | device | 1 | 1.469 | 1.547 | 1.730 | 0.8% |
| L2P | device | 2 | 1.753 | 1.933 | 2.073 | 1.1% |
| P2P | device | 2 | 57.048 | 57.748 | 57.987 | 31.6% |
| load | | | 14.580 | 15.415 | 18.340 | |
| output | | | 32.747 | 33.261 | 34.582 | |
| other | | | | 0.116 | | |
| sum | | | | 182.880 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 3.935 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MBP; never asserted.

### Metal (M3 Max), after, N = 10⁶

#### FMM benchmark: Timos-MBP, 2026-10-08 10:52:57 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:52:57 UTC |
| host | Timos-MBP |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | a12d9d6-dirty (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --n 1e6 --degree 3,8 --table-cache /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal --output /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/metal-bench/after-1e6.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 1000000 | 3 | Dense | 0.841 | 13.745 | 13.886 | 13.936 | 14.208 | 0.146 | 2.00e-3 | 3.80e-3 | yes |
| metal | f32 | 1000000 | 8 | Dense | 0.820 | 79.348 | 80.264 | 81.075 | 90.037 | 3.189 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 1000000 | 3 | 0.532 | 1.789 | 6.840 | 1.659 | 0.000 | 0.000 | 0.466 | 5.199 | 1.519 | 2.001 | 0.026 | 16.485 | 20.132 | 1.183 |
| metal | f32 | 1000000 | 8 | 1.179 | 1.997 | 71.180 | 2.032 | 0.000 | 0.000 | 0.753 | 5.130 | 1.587 | 2.675 | 0.030 | 82.271 | 86.764 | 1.015 |

##### metal, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.841 s, of it tables 0.003 s (loaded from the cache) and device 0.185 s.
- Evaluation over 10 repeats, ms: min 13.745, median 13.886, mean 13.936, max 14.208, std 0.146; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.396 | 0.532 | 0.612 | 3.2% |
| M2M | device | 5 | 1.744 | 1.789 | 1.813 | 10.9% |
| M2L | device | 4 | 6.753 | 6.840 | 7.063 | 41.5% |
| L2L | device | 5 | 1.583 | 1.659 | 1.843 | 10.1% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.394 | 0.466 | 0.582 | 2.8% |
| P2P | device | 1 | 5.063 | 5.199 | 5.278 | 31.5% |
| load | | | 1.405 | 1.519 | 1.719 | |
| output | | | 1.894 | 2.001 | 2.157 | |
| other | | | | 0.026 | | |
| sum | | | | 16.485 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.513 ms on average.

##### metal, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.820 s, of it tables 0.012 s (loaded from the cache) and device 0.164 s.
- Evaluation over 10 repeats, ms: min 79.348, median 80.264, mean 81.075, max 90.037, std 3.189; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 1.089 | 1.179 | 1.300 | 1.4% |
| M2M | device | 5 | 1.924 | 1.997 | 2.179 | 2.4% |
| M2L | device | 4 | 70.830 | 71.180 | 71.459 | 86.5% |
| L2L | device | 5 | 1.897 | 2.032 | 2.195 | 2.5% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.596 | 0.753 | 0.897 | 0.9% |
| P2P | device | 1 | 5.017 | 5.130 | 5.473 | 6.2% |
| load | | | 1.451 | 1.587 | 1.788 | |
| output | | | 2.146 | 2.675 | 3.014 | |
| other | | | | 0.030 | | |
| sum | | | | 82.271 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.496 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MBP; never asserted.

### Metal (M3 Max), after, N = 10⁷

#### FMM benchmark: Timos-MBP, 2026-10-08 10:53:09 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:53:09 UTC |
| host | Timos-MBP |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | a12d9d6-dirty (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --n 1e7 --degree 3 --table-cache /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal --output /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/metal-bench/after-1e7.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate` over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 10000000 | 3 | Dense | 9.028 | 216.299 | 216.970 | 217.320 | 220.139 | 1.166 | 1.87e-3 | 1.87e-3 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 10000000 | 3 | 2.181 | 3.348 | 111.601 | 3.012 | 0.386 | 1.662 | 1.950 | 57.693 | 14.615 | 30.958 | 0.121 | 181.832 | 228.457 | 0.837 |

##### metal, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 9.028 s, of it tables 0.002 s (loaded from the cache) and device 1.793 s.
- Evaluation over 10 repeats, ms: min 216.299, median 216.970, mean 217.320, max 220.139, std 1.166; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(16 x 4 units, 4 columns per unit); rotation cube (32 units); device memory 1841.2 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 216, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 6 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 45 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 1.975 | 2.181 | 2.560 | 1.2% |
| M2M | device | 7 | 2.696 | 3.348 | 4.325 | 1.8% |
| M2L | device | 5 | 111.412 | 111.601 | 111.804 | 61.4% |
| L2L | device | 7 | 2.619 | 3.012 | 3.928 | 1.7% |
| P2L | device | 1 | 0.357 | 0.386 | 0.429 | 0.2% |
| M2P | device | 1 | 1.485 | 1.662 | 1.840 | 0.9% |
| L2P | device | 2 | 1.836 | 1.950 | 2.031 | 1.1% |
| P2P | device | 2 | 56.648 | 57.693 | 57.914 | 31.7% |
| load | | | 13.757 | 14.615 | 15.420 | |
| output | | | 29.899 | 30.958 | 32.938 | |
| other | | | | 0.121 | | |
| sum | | | | 181.832 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.237 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MBP; never asserted.

### Metal (M3 Max), before, `--reuse-output`, N = 10⁶

#### FMM benchmark: Timos-MBP, 2026-10-08 10:54:12 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:54:12 UTC |
| host | Timos-MBP |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | a12d9d6 (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --n 1e6 --degree 3,8 --reuse-output --table-cache /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal --output /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/metal-bench/before-reuse-1e6.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, `--reuse-output`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 1000000 | 3 | Dense | 0.838 | 14.123 | 14.302 | 14.288 | 14.482 | 0.108 | 2.00e-3 | 3.80e-3 | yes |
| metal | f32 | 1000000 | 8 | Dense | 0.823 | 80.658 | 80.958 | 81.031 | 81.375 | 0.217 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 1000000 | 3 | 0.482 | 1.874 | 6.846 | 1.693 | 0.000 | 0.000 | 0.464 | 5.159 | 1.507 | 2.168 | 0.024 | 16.518 | 20.315 | 1.156 |
| metal | f32 | 1000000 | 8 | 1.110 | 2.010 | 71.302 | 2.053 | 0.000 | 0.000 | 0.688 | 5.049 | 1.653 | 2.920 | 0.030 | 82.213 | 87.011 | 1.015 |

##### metal, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.838 s, of it tables 0.003 s (loaded from the cache) and device 0.187 s.
- Evaluation over 10 repeats, ms: min 14.123, median 14.302, mean 14.288, max 14.482, std 0.108; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.420 | 0.482 | 0.633 | 2.9% |
| M2M | device | 5 | 1.676 | 1.874 | 2.071 | 11.3% |
| M2L | device | 4 | 6.763 | 6.846 | 6.952 | 41.4% |
| L2L | device | 5 | 1.598 | 1.693 | 1.802 | 10.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.419 | 0.464 | 0.608 | 2.8% |
| P2P | device | 1 | 4.964 | 5.159 | 5.214 | 31.2% |
| load | | | 1.358 | 1.507 | 1.644 | |
| output | | | 2.010 | 2.168 | 2.415 | |
| other | | | | 0.024 | | |
| sum | | | | 16.518 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.746 ms on average.

##### metal, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.823 s, of it tables 0.014 s (loaded from the cache) and device 0.162 s.
- Evaluation over 10 repeats, ms: min 80.658, median 80.958, mean 81.031, max 81.375, std 0.217; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.991 | 1.110 | 1.254 | 1.4% |
| M2M | device | 5 | 1.891 | 2.010 | 2.410 | 2.4% |
| M2L | device | 4 | 70.978 | 71.302 | 71.732 | 86.7% |
| L2L | device | 5 | 1.918 | 2.053 | 2.226 | 2.5% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.593 | 0.688 | 0.763 | 0.8% |
| P2P | device | 1 | 5.004 | 5.049 | 5.205 | 6.1% |
| load | | | 1.444 | 1.653 | 2.038 | |
| output | | | 2.802 | 2.920 | 3.115 | |
| other | | | | 0.030 | | |
| sum | | | | 82.213 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.729 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate_into` (one reused `Output`), with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MBP; never asserted.

### Metal (M3 Max), before, `--reuse-output`, N = 10⁷

#### FMM benchmark: Timos-MBP, 2026-10-08 10:54:24 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:54:24 UTC |
| host | Timos-MBP |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | a12d9d6 (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --n 1e7 --degree 3 --reuse-output --table-cache /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal --output /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/metal-bench/before-reuse-1e7.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, `--reuse-output`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 10000000 | 3 | Dense | 8.997 | 219.935 | 221.080 | 222.428 | 229.084 | 3.123 | 1.87e-3 | 1.87e-3 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 10000000 | 3 | 2.651 | 4.034 | 112.171 | 3.000 | 0.397 | 1.565 | 1.981 | 57.782 | 15.024 | 33.569 | 0.116 | 183.581 | 233.232 | 0.825 |

##### metal, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 8.997 s, of it tables 0.002 s (loaded from the cache) and device 1.821 s.
- Evaluation over 10 repeats, ms: min 219.935, median 221.080, mean 222.428, max 229.084, std 3.123; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(16 x 4 units, 4 columns per unit); rotation cube (32 units); device memory 1841.2 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 216, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 6 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 45 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 2.014 | 2.651 | 3.140 | 1.4% |
| M2M | device | 7 | 2.694 | 4.034 | 6.004 | 2.2% |
| M2L | device | 5 | 110.980 | 112.171 | 117.089 | 61.1% |
| L2L | device | 7 | 2.712 | 3.000 | 3.383 | 1.6% |
| P2L | device | 1 | 0.354 | 0.397 | 0.565 | 0.2% |
| M2P | device | 1 | 1.483 | 1.565 | 1.715 | 0.9% |
| L2P | device | 2 | 1.905 | 1.981 | 2.114 | 1.1% |
| P2P | device | 2 | 57.555 | 57.782 | 57.906 | 31.5% |
| load | | | 14.646 | 15.024 | 15.510 | |
| output | | | 32.925 | 33.569 | 34.711 | |
| other | | | | 0.116 | | |
| sum | | | | 183.581 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 3.955 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate_into` (one reused `Output`), with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MBP; never asserted.

### Metal (M3 Max), after, `--reuse-output`, N = 10⁶

#### FMM benchmark: Timos-MBP, 2026-10-08 10:55:27 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:55:27 UTC |
| host | Timos-MBP |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | a12d9d6-dirty (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --n 1e6 --degree 3,8 --reuse-output --table-cache /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal --output /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/metal-bench/after-reuse-1e6.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, `--reuse-output`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 1000000 | 3 | Dense | 0.835 | 13.651 | 13.895 | 13.947 | 14.651 | 0.271 | 2.00e-3 | 3.80e-3 | yes |
| metal | f32 | 1000000 | 8 | Dense | 0.815 | 79.382 | 80.280 | 81.144 | 90.161 | 3.197 | 1.36e-5 | 4.59e-5 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 1000000 | 3 | 0.527 | 1.871 | 6.789 | 1.677 | 0.000 | 0.000 | 0.471 | 5.184 | 1.479 | 1.993 | 0.025 | 16.519 | 20.114 | 1.184 |
| metal | f32 | 1000000 | 8 | 1.156 | 2.007 | 71.265 | 2.033 | 0.000 | 0.000 | 0.712 | 5.108 | 1.598 | 2.589 | 0.027 | 82.281 | 86.687 | 1.014 |

##### metal, f32, N = 1000000, p = 3

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.835 s, of it tables 0.002 s (loaded from the cache) and device 0.183 s.
- Evaluation over 10 repeats, ms: min 13.651, median 13.895, mean 13.947, max 14.651, std 0.271; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 0.386 | 0.527 | 0.602 | 3.2% |
| M2M | device | 5 | 1.792 | 1.871 | 2.042 | 11.3% |
| M2L | device | 4 | 6.712 | 6.789 | 6.880 | 41.1% |
| L2L | device | 5 | 1.598 | 1.677 | 1.892 | 10.2% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.415 | 0.471 | 0.559 | 2.9% |
| P2P | device | 1 | 5.123 | 5.184 | 5.231 | 31.4% |
| load | | | 1.339 | 1.479 | 1.610 | |
| output | | | 1.841 | 1.993 | 2.685 | |
| other | | | | 0.025 | | |
| sum | | | | 16.519 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.496 ms on average.

##### metal, f32, N = 1000000, p = 8

- Tree: 32768 leaves on 6 levels; points per leaf 11 / 30.5 / 55 (min / mean / max); lists U 797816, V 6039504, W 0, X 0.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 0.815 s, of it tables 0.012 s (loaded from the cache) and device 0.163 s.
- Evaluation over 10 repeats, ms: min 79.382, median 80.280, mean 81.144, max 90.161, std 3.197; bit-identical across the repeats: yes.
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
| P2M | device | 1 | 1.016 | 1.156 | 1.309 | 1.4% |
| M2M | device | 5 | 1.903 | 2.007 | 2.190 | 2.4% |
| M2L | device | 4 | 71.030 | 71.265 | 71.597 | 86.6% |
| L2L | device | 5 | 1.889 | 2.033 | 2.299 | 2.5% |
| P2L | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | device | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | device | 1 | 0.621 | 0.712 | 0.854 | 0.9% |
| P2P | device | 1 | 4.988 | 5.108 | 5.261 | 6.2% |
| load | | | 1.514 | 1.598 | 1.757 | |
| output | | | 1.893 | 2.589 | 3.054 | |
| other | | | | 0.027 | | |
| sum | | | | 82.281 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 0.494 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate_into` (one reused `Output`), with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MBP; never asserted.

### Metal (M3 Max), after, `--reuse-output`, N = 10⁷

#### FMM benchmark: Timos-MBP, 2026-10-08 10:55:39 UTC

| item | value |
| --- | --- |
| date | 2026-10-08 10:55:39 UTC |
| host | Timos-MBP |
| CPU | Apple M3 Max; 16 physical, 16 logical, 12 performance |
| devices | metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0 (b940084d7 2026-09-28); aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | a12d9d6-dirty (git describe) |
| command | `nd-fmm-bench --backend metal --precision f32 --n 1e7 --degree 3 --reuse-output --table-cache /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal --output /private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/metal-bench/after-reuse-1e7.md` |
| environment | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, GOTO_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c06), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 16 (default); refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 16 (default: every core); host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 2 warm-up, then 10 timed evaluations; kind timings `sync` in a second build |
| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, `--reuse-output`); output pass `auto` |
| accuracy | relative L2 at 1000 sampled targets against the f64 direct sum |
| table cache | `/private/tmp/claude-501/-Users-betcke-projects-worktrees-fmm-phase4s-t11-2/87e27501-c68e-4c5a-9097-6c5e26774864/scratchpad/tables-metal` |
| tuning cache | none |

##### Summary

Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 10 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| metal | f32 | 10000000 | 3 | Dense | 8.972 | 216.570 | 218.546 | 220.382 | 228.518 | 4.177 | 1.87e-3 | 1.87e-3 | yes |

##### Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal | f32 | 10000000 | 3 | 2.301 | 3.652 | 111.719 | 2.962 | 0.426 | 1.537 | 1.954 | 57.770 | 14.363 | 30.363 | 0.117 | 182.321 | 228.096 | 0.827 |

##### metal, f32, N = 10000000, p = 3

- Tree: 262214 leaves on 8 levels; points per leaf 1 / 38.1 / 64 (min / mean / max); lists U 6597988, V 52337880, W 1408, X 1408.
- Run: strategy Dense; P2P kernel neon; rayon threads 16; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Build: 8.972 s, of it tables 0.002 s (loaded from the cache) and device 1.802 s.
- Evaluation over 10 repeats, ms: min 216.570, median 218.546, mean 220.382, max 228.518, std 4.177; bit-identical across the repeats: yes.
- Accuracy at 1000 sampled targets: φ 1.87e-3, ∇φ 1.87e-3.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube (64 units); leaf cube (64 units, tile 32); GEMM cube(16 x 4 units, 4 columns per unit); rotation cube (32 units); device memory 1841.2 MB.
- One evaluation: uploads 1 (40000000 B), downloads 1 (160000000 B), launches 216, syncs 1, timing windows 0.
- Device level calls of M2M, L2L and M2L:
  - M2M (Local) level 1: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 2: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 3: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 4: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 5: 262144 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Local) level 6: 80 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2M (Global) level 0: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 1: 8 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 2: 64 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 3: 512 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 4: 4096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 5: 32768 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 6: 262144 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - L2L level 7: 80 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 2: 3096 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 3: 53352 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 4: 584136 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 1 chunk
  - M2L level 5: 5398920 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 6 chunks
  - M2L level 6: 46298376 pairs, hand-written cube(16 x 4 units, 4 columns per unit), 45 chunks

Kind timings `sync` (synchronous, a sync after each call on a device), over 10 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | device | 2 | 1.984 | 2.301 | 2.685 | 1.3% |
| M2M | device | 7 | 2.643 | 3.652 | 5.605 | 2.0% |
| M2L | device | 5 | 111.454 | 111.719 | 112.309 | 61.3% |
| L2L | device | 7 | 2.722 | 2.962 | 3.308 | 1.6% |
| P2L | device | 1 | 0.357 | 0.426 | 0.649 | 0.2% |
| M2P | device | 1 | 1.475 | 1.537 | 1.703 | 0.8% |
| L2P | device | 2 | 1.836 | 1.954 | 2.065 | 1.1% |
| P2P | device | 2 | 57.579 | 57.770 | 58.055 | 31.7% |
| load | | | 13.743 | 14.363 | 14.720 | |
| output | | | 29.886 | 30.363 | 31.240 | |
| other | | | | 0.117 | | |
| sum | | | | 182.321 | | 100.0% |

Output: of it the download (`read_output`, with its sync) 1.248 ms on average.

##### Method

Each combination builds the FMM once and runs 2 warm-up evaluations, which compile the device kernels and are not counted, then times 10 evaluations of the same charges by the wall clock around `Fmm::evaluate_into` (one reused `Output`), with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 10 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: metal (f32); refused or failed: none.

Measured on Timos-MBP; never asserted.

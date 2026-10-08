# Spike report: the download and upload paths (Phase 4S / T11, item 1)

The brief is docs/phase4s/T11-download-path.md. This spike measures where the host time
of a device evaluation's one download and one upload goes, step by step, on CUDA (locust's
H100) and Metal (the M3 Max), before any change to the library. It brings the numbers for
decisions 14 and 15 of docs/phase4s/README.md. The raw output is in `results-gh200.md`
and `results-m3max.md`, and every table below comes from them.

Every number is labelled: **measured (locust, CUDA)**, **measured (M3 Max, Metal)**,
**measured (T9)** for fmm-bench/results/phase4s-t9-host-part.md, or **model** for an
estimate computed from measured parts. Timings are reported and never asserted. Sizes are
in MB (10⁶ bytes). A size of 160 MB is the output of N = 10⁷ targets with gradients in
f32, and 320 MB the same in f64. 16 MB is N = 10⁶ in f32.

## Summary

- **On CUDA the download is almost all our own copy.** `Device::download` of 160 MB takes
  11.66 ms. Of that, CubeCL's `read_one` takes 0.55 ms: the GPU copy into pinned memory,
  at 289 GB/s. The serial `copy_from_slice` into the caller's slice takes 11.03 ms, at
  14.5 GB/s on one Grace core. The copy is 93–97% of the download at 16, 160 and 320 MB,
  in f32 and in f64. The same copy on rayon's 72 threads takes 1.09 ms (measured
  (locust, CUDA)).
  - T9's `copy_output` already copies from the operator's host buffer into the `Output`
    in parallel. A view of CubeCL's pinned bytes therefore removes the serial copy and
    leaves the parallel one.
  - Model: at N = 10⁷ an evaluation drops from 60.3 to about 49 ms (f32, p = 3) and from
    103 to about 81 ms (f64). At N = 10⁶ it drops from 6.6 to about 5.7 ms (f32).
- **On Metal the copy is half the download, but small.** At 16 MB (N = 10⁶) the copy
  takes 0.25 ms of a 0.56 ms `Device::download`, and `read_one` takes 0.47 ms (measured
  (M3 Max, Metal)). That is about 2% of T9's 14 ms evaluation at p = 3.
- **CubeCL's host pools grow, and then they are reused.**
  - On CUDA there is one `cuMemAllocHost` per growth of the pinned pool: 58–61 ms for the
    first 160 MB download in a process. After that, `read_one` takes 0.55 ms every time.
    Under `nsys`, 1 and 10 downloads both make exactly 1 call, and 10 evaluation-like
    rounds (a 40 MB write and a 160 MB download) make 2.
  - On Metal, the first `read_one` of each new size grows wgpu's staging pool (5–11 ms).
    It is flat afterwards.
  - `Client::memory_persistent_allocation` does not reach either host pool. The source
    switches only the device pool's mode, and the measurement is identical.
  - **Decision 15 is not needed.**
- **The upload: `to_vec` is half of `Device::write` on CUDA.** For 40 MB (the charges of
  N = 10⁷ in f32) the steps are:
  - `to_vec` 2.97 ms;
  - `Client::write` returns in 1 µs;
  - the transfer, to a following sync, 2.73 ms;
  - in all, 5.70 ms. A `Vec` handed over without `to_vec` takes 2.58 ms (measured
    (locust, CUDA)).

  On Metal, `to_vec` is 7% of the upload: 0.06 of 0.76 ms at 4 MB. wgpu's own write path
  is the rest.
- **The zeroing of `begin_evaluation` costs little.** On CUDA it takes 0.03–0.20 ms for all
  three buffers; the windows are CUDA events. On Metal it takes 0.25–0.31 ms at
  N = 10⁶.
- **Recommendation for decision 14.**
  - Yes to a view of the downloaded bytes, with the API below, and the device operator's
    host output buffer removed. This saves o N_t s bytes of host memory: 160 MB at
    N = 10⁷ in f32, 320 MB in f64.
  - Yes to an owned upload (`Device::write_owned`). Its gain on CUDA is about 3 ms at
    N = 10⁷ in f32 and 6 ms in f64.

## Setup

| item | value |
| --- | --- |
| code | `main` at `a12d9d6` (T9 merged, the T11 brief), with this spike added and no library change |
| locust | GH200 480GB, 72 Neoverse-V2 cores, H100 (96 GB), driver 565.57.01, the spack environment of tools/gh200/ (CUDA 12.6), rustc 1.99.0, CubeCL 0.11.0-pre.4 (LLVM NVPTX), rayon 72 threads |
| load on locust | checked before and after (`load-before.txt` and `load-after.txt`, 10:02:59 and 10:03:50 UTC on 2026-10-08): no compute process on the GPU, 0% utilisation, load average 2.75 → 1.54 (the build and the run itself), and the only other user's process `nvitop` at 0.8% of a core |
| clocks | not locked (no administrator access): SM 345 MHz idle, application and maximum 1,980 MHz (SM) and 2,619 MHz (memory), from `nvidia-smi -q -d CLOCK` before and after |
| the M3 Max | Metal through wgpu (MSL), f32, rayon 16 threads, run outside the sandbox. A desktop session with a load average of 2.6–3.1 (Defender, WindowServer). An earlier run under a load average of 13 agreed within 10%. The Metal numbers are indicative |
| timing | release builds; 2 warm-ups and 10 timed repetitions; median (max) in ms; every BLAS thread variable 1 |
| commands | locust: `spikes/download-path/run-gh200.sh` (all sections in f32 and f64, then the pool section per precision in its own process, then six `nsys profile --trace=cuda,osrt` runs of the pool sequences). The M3 Max: `cargo run --release -p nd-fmm-spike-download-path --features metal -- --backends metal --precisions f32`, then the same with `--sections pool` |

The CPU runtime ran only during development, at small sizes inside the sandbox, as a smoke
test. It is not reported.

## What the source of CubeCL 0.11.0-pre.4 says (reading, before measuring)

- **Read, CUDA.** `Client::read_one` calls `read_async`, which goes to the server's
  `read`. That runs `copy_to_bytes`:
  - `reserve_cpu(size)` takes a slice of the stream's pinned pool
    (`memory_management_cpu`, `PinnedMemoryStorage`, `cuMemAllocHost_v2` when a page is
    added). If the pool cannot serve it, the fallback is `vec![0; size]`;
  - then a `cuMemcpyDtoHAsync` into that slice, and a fence;
  - the returned `Bytes` owns the pinned slice. Dropping it returns the slice to the pool
    (cubecl-server `command/base.rs`, cubecl-cuda `compute/storage/cpu.rs`).
- **Read, wgpu.** `read_resources` copies each buffer into a slice of the staging pool
  (`reserve_staging`, `MAP_READ | COPY_DST` buffers, `MemoryConfiguration::ExclusivePages`,
  mode `Auto`). It then flushes and `map_async`s. The `Bytes` wraps the mapped range
  through a `WgpuAllocController` (cubecl-wgpu `compute/stream.rs`, `mem_manager.rs`).
- **The cast.** `Bytes` derefs to `[u8]`. `CubeElement::from_bytes` (a `bytemuck` cast,
  safe) gives `&[E]`. `Device::download` does that today and then copies. **A view needs no
  `unsafe`**: it keeps the `Bytes` and derefs through the same cast.
- **Write, CUDA.** `Client::write` submits `(descriptor, Bytes)` to the server, whose
  `write_to_gpu` picks a `Staging`:
  - data under `STAGE_MAX` (100 MiB) that is not already pinned is first copied into a
    slice of the pinned pool, then sent;
  - larger data is sent from where it is;
  - the source goes into the drop queue until the copy has run.

  `Bytes::from_elems(Vec<E>)` takes the `Vec` without copying.
- **`memory_persistent_allocation`.** The client submits `allocation_mode(Persistent)`.
  On CUDA, `Command::allocation_mode` sets `streams.current().device_memory().mode(..)`:
  the device pool only, not `host_memory()`. On wgpu, `WgpuMemManager::mode` sets
  `memory_pool` (the main pool) only, and the staging pool was built with mode `Auto`. **It
  does not cover the host pools.** The measurement below agrees.

## The download, step by step

A fill kernel writes the buffer before each download. The steps are:
- `sync`: the wait for that kernel;
- `read_one`: after that sync;
- `copy`: `copy_from_slice` into a reused, touched slice on one thread;
- `par copy`: the same copy on rayon's pool, in chunks of 2¹⁶ elements;
- `cold copy`: a fresh `vec!` and the copy into it, with its page faults;
- `drop`: dropping CubeCL's `Bytes`.

Every copy is checked against the fill and against the others.
`Device::download` is the production call from an idle stream, which includes its own
sync. Median (max) in ms.

Measured (locust, CUDA):

| precision | MB | sync | `read_one` | copy | par copy (72 thr.) | cold copy | drop | `Device::download` | copy share of it |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | 16 | 0.140 | 0.070 | 0.933 | 0.170 | 0.800 | 0.000 | 1.003 | 93% |
| f32 | 160 | 0.182 | 0.554 (0.555) | 11.027 (11.167) | 1.090 | 13.207 | 0.002 | 11.658 (12.479) | 95% |
| f32 | 320 | 0.183 | 1.091 | 22.415 | 2.327 | 27.080 | 0.002 | 23.691 | 95% |
| f64 | 16 | 0.059 | 0.068 | 0.999 | 0.305 | 1.051 | 0.000 | 1.034 | 97% |
| f64 | 160 | 0.138 | 0.554 | 11.149 | 1.108 | 12.978 | 0.001 | 11.282 | 99% |
| f64 | 320 | 0.199 | 1.091 (1.093) | 22.414 (22.675) | 2.256 | 26.666 | 0.002 | 23.086 (23.799) | 97% |

- `read_one` moves 289–293 GB/s at 160 and 320 MB. That is the GPU's copy into pinned
  memory: T8's `nsys` gave 0.54 ms for the copy of 160 MB.
- The serial copy moves 14.3–17.2 GB/s. In parallel it moves 52–147 GB/s (52–94 at 16 MB, 138–147 at 160 and 320 MB).
- A fresh destination adds 15–20% in page faults at 160 and 320 MB.
- Dropping the `Bytes` costs nothing.
- The download in T9's benchmark at N = 10⁷ (12.69 ms in f32, 24.61 ms in f64, measured
  (T9)) is this `Device::download` plus the last kernels' tail.

Measured (M3 Max, Metal, f32):

| MB | sync | `read_one` | copy | par copy (16 thr.) | cold copy | drop | `Device::download` | copy share of it |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.539 | 0.466 (0.576) | 0.249 (0.280) | 0.339 | 0.364 | 0.000 | 0.557 (0.624) | 45% |
| 160 | 1.244 | 1.694 | 2.529 | 1.870 | 3.311 | 0.001 | 4.222 | 60% |
| 320 | 1.760 | 2.246 | 5.156 | 2.900 | 7.348 | 0.003 | 7.768 | 66% |

- On Metal, `read_one` (the GPU copy into a staging buffer and the mapping) moves 34 GB/s
  at 16 MB and 94–143 GB/s at 160 and 320 MB.
- The serial copy moves 62–64 GB/s, four times Grace's single-core rate.
- At 16 MB the parallel copy is not faster than the serial one.

## CubeCL's host pools

`read_one` per download in sequence (ms), after a fill and a sync. Each precision ran in a
process of its own that had run nothing else. The sequences, in this order in each
process:
- `steady`: 10 downloads of the N = 10⁷ output;
- `mixed`: 16, 160 and 320 MB in turn;
- `evaluation`: per round a `Client::write` of the N = 10⁷ charges, a fill, and a
  download of the output;
- `persistent`: `steady` with each `read_one` inside `memory_persistent_allocation`.

Measured (locust, CUDA):

| process | sequence | `read_one` per download, ms |
| --- | --- | --- |
| f32 | steady (160 MB) | **61.14**, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55 |
| f32 | mixed | **14.63** (16), 0.55 (160), 1.11 (320), 0.08 (16), 0.55, 1.09, 0.08, 0.55, 1.09, 0.07 |
| f32 | evaluation (40 MB up, 160 MB down) | 0.55 every round (0.55–0.56) |
| f32 | persistent | 0.55 every download |
| f64 | steady (320 MB) | **16.44**, then 1.09–1.10 |
| f64 | mixed | 0.07 (16), **60.60** (160), 1.09 (320), 0.07, 0.55, 1.09, 0.07, 0.55, 1.09, 0.07 |
| f64 | evaluation (80 MB up, 320 MB down) | 1.09 every round |
| f64 | persistent | 1.09–1.10 every download |

Under `nsys` (`--trace=cuda,osrt`), one sequence per process in f32, measured (locust,
CUDA):

| run | `cuMemAllocHost_v2` calls | their time, ms | HtoD copies | DtoH copies |
| --- | ---: | --- | ---: | ---: |
| steady, 1 download | 1 | 57.9 | 0 | 1 |
| steady, 10 downloads | 1 | 60.5 | 0 | 10 |
| evaluation, 1 round | 2 | 15.3 + 59.6 | 1 | 1 |
| evaluation, 10 rounds | 2 | 14.8 + 57.8 | 10 | 10 |
| mixed, 10 downloads | 2 | 15.3 + 59.3 | 0 | 10 |
| persistent, 10 downloads | 1 | 58.7 | 0 | 10 |

- The pinned pool allocates only when it grows: one page for the download, and one for
  the upload's staging, below `STAGE_MAX`. More downloads or rounds add no call.
  - T9's whole benchmark run made 6 calls (measured (T9)). That fits this: the build's
    uploads, the warm-up's first download, and no more.
  - A page costs about 15 or about 60 ms, depending on the pool that serves the size
    rather than on the size. In the f64 process the 320 MB download was the 16 ms kind
    and the 160 MB one the 60 ms kind. This spike does not resolve the page sizes;
    CubeCL logs them only with a `cubecl.toml` memory logger.
- `memory_persistent_allocation` changes nothing, as the source says.

Measured (M3 Max, Metal, f32):

| sequence | `read_one` per download, ms |
| --- | --- |
| steady (160 MB) | **5.17**, 1.56, 1.24, 1.36, 1.21, 1.22, 1.30, 1.21, 1.22, 1.21 |
| mixed | 0.70 (16), 1.19 (160), **11.01** (320), 0.42 (16), 1.30, 2.25, 0.45, 1.18, 2.22, 0.46 |
| evaluation (40 MB up, 160 MB down) | 1.21–1.44 |
| persistent | 1.96, then 1.32–1.68 |

wgpu's staging pool grows on the first download of a larger size and is reused after it.
The staging pool has no counter like `nsys` gives on CUDA, so this rests on times alone.

**Conclusion for decision 15:** neither pool allocates again in steady state.
- The growth falls in the first evaluation of a process: the warm-up in every benchmark
  and test. It costs about 60–75 ms on CUDA and 5–11 ms on Metal.
- Nothing needs changing. Through the public API, `Client::staging` with a buffer of the
  output's size at build would move the CUDA growth from the first evaluation into the
  build (not measured). That is optional and outside this task.

## The upload, step by step

Each upload starts from an idle stream. The steps are:
- `to_vec`: `Device::write`'s copy of the caller's slice, a fresh allocation;
- `write`: `Client::write` until it returns;
- `sync`: the transfer, to a following sync;
- `owned`: `from_elems`, `write` and `sync` of a `Vec` made before the timing.

`Bytes::from_elems` took under 1 µs at every size. Each path's device contents were read
back once and compared bit for bit with the source. Median in ms.

Measured (locust, CUDA):

| precision | MB | `to_vec` | `write` | `sync` | total | owned | `Device::write` + sync | `to_vec` share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | 4 | 0.183 | 0.001 | 0.384 | 0.571 | 0.266 | 0.525 | 32% |
| f32 | 8 | 0.305 | 0.000 | 0.655 | 0.961 | 0.458 | 0.739 | 32% |
| f32 | 40 | 2.973 | 0.001 | 2.732 | 5.699 | 2.578 | 5.681 | 52% |
| f32 | 80 | 6.268 | 0.001 | 5.137 | 11.414 | 5.071 | 12.163 | 55% |
| f32 | 160 | 12.242 | 0.001 | 1.650 | 13.930 | 1.460 | 14.322 | 88% |
| f64 | 4 | 0.185 | 0.001 | 0.389 | 0.577 | 0.272 | 0.595 | 32% |
| f64 | 8 | 0.359 | 0.001 | 0.683 | 1.042 | 0.539 | 0.760 | 34% |
| f64 | 40 | 3.001 | 0.001 | 2.818 | 5.843 | 2.759 | 5.675 | 51% |
| f64 | 80 | 6.384 | 0.001 | 4.986 | 11.371 | 4.843 | 12.198 | 56% |
| f64 | 160 | 12.609 | 0.001 | 1.908 | 14.456 | 1.749 | 14.946 | 87% |

The charges of the benchmark are 4 MB (N = 10⁶, f32), 8 MB (N = 10⁶, f64), 40 MB
(N = 10⁷, f32) and 80 MB (N = 10⁷, f64).
- `to_vec` runs at 13 GB/s on one Grace core. It includes the fresh allocation's page
  faults.
- `Client::write` only queues. The transfer happens before the next sync completes.
- Below 100 MiB the transfer includes CubeCL's copy into its pinned staging buffer: 40 MB
  take 2.73 ms in `sync`. At 160 MB (above `STAGE_MAX`) CubeCL sends from the `Vec`
  directly, and `sync` takes only 1.65 ms. That is an inference from these times: the
  staging copy (serial, on CubeCL's server thread, about 16 GB/s) costs more on the GH200
  than the transfer from pageable memory. It is CubeCL's threshold, not ours; see
  "Upstream".
- T9's load at N = 10⁷ (6.62 ms in f32, 13.58 ms in f64, measured (T9)) is mostly this
  write: 5.68 and 12.20 ms. The rest is the zeroing (below), the charge scatter and the
  charge gather, which this spike does not split.

Measured (M3 Max, Metal, f32):

| MB | `to_vec` | `write` | `sync` | total | owned | `Device::write` + sync | `to_vec` share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.056 | 0.000 | 0.707 | 0.762 | 0.673 | 0.763 | 7% |
| 8 | 0.125 | 0.000 | 1.256 | 1.387 | 1.158 | 1.805 | 9% |
| 40 | 0.644 | 0.000 | 5.003 | 5.647 | 4.965 | 5.272 | 11% |
| 80 | 1.331 | 0.001 | 10.007 | 11.349 | 9.806 | 12.211 | 12% |
| 160 | 2.619 | 0.001 | 18.493 | 21.115 | 18.209 | 22.036 | 12% |

wgpu's write (`queue.write_buffer` with its own staging) runs at about 8 GB/s and is
most of the upload on Metal. `to_vec` is a small part of it.

## The zeroing of `begin_evaluation`

The three zero kernels run at the benchmark's sizes:
- the multipoles and the locals: (p + 1)² values per box on every level, with 37,449
  boxes at N = 10⁶ and 299,673 at N = 10⁷, counted from the level calls in T9's reports;
- the target output: 4 N values.

Each repetition opens one window per kernel, then one window over all three as
`begin_evaluation` runs them. The windows are CUDA events and Metal timestamps,
through `Device::open_window` and `close_window`. Median in ms.

| backend | precision | N | p | multipoles + locals MB | output MB | all three | multipoles | locals | output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA | f32 | 10⁶ | 3 | 4.8 | 16 | 0.031 | 0.017 | 0.016 | 0.025 |
| CUDA | f32 | 10⁶ | 8 | 24.3 | 16 | 0.043 | 0.023 | 0.018 | 0.021 |
| CUDA | f32 | 10⁷ | 3 | 38.4 | 160 | 0.112 | 0.025 | 0.017 | 0.081 |
| CUDA | f32 | 10⁷ | 8 | 194.2 | 160 | 0.197 | 0.068 | 0.059 | 0.081 |
| CUDA | f64 | 10⁶ | 3 | 9.6 | 32 | 0.031 | 0.017 | 0.017 | 0.025 |
| CUDA | f64 | 10⁶ | 8 | 48.5 | 32 | 0.043 | 0.023 | 0.019 | 0.021 |
| CUDA | f64 | 10⁷ | 3 | 76.7 | 320 | 0.115 | 0.028 | 0.018 | 0.087 |
| CUDA | f64 | 10⁷ | 8 | 388.4 | 320 | 0.201 | 0.067 | 0.059 | 0.084 |
| Metal | f32 | 10⁶ | 3 | 4.8 | 16 | 0.253 | 0.996 | 0.085 | 0.088 |
| Metal | f32 | 10⁶ | 8 | 24.3 | 16 | 0.314 | 0.773 | 0.178 | 0.076 |
| Metal | f32 | 10⁷ | 3 | 38.4 | 160 | 1.067 | 0.995 | 0.188 | 0.437 |
| Metal | f32 | 10⁷ | 8 | 194.2 | 160 | 0.443 | 1.020 | 0.280 | 0.276 |

Measured (locust, CUDA) and measured (M3 Max, Metal).
- On CUDA the zeroing is under 2% of the load at every size.
- On Metal the first window of each repetition (the multipoles) reads about 1 ms whatever
  its size. That is the window, not the kernel, and the window over all three is the
  figure that counts.
- At N = 10⁶ on Metal the zeroing takes 0.25–0.31 ms of T9's 1.67 ms load. The rest is
  wgpu's write of the charges (0.76 ms above), the scatter and the gather.
- The Metal times are noisy: the N = 10⁷ row at p = 3 is above the one at p = 8. Nothing
  here suggests changing the zeroing.

## Where the evaluation's time goes, and what the changes would remove

Model, from the measured parts above and T9's means (measured (T9)). It assumes that the
parallel copy into the `Output`, which T9's `copy_output` already runs, costs the same
reading CubeCL's pinned bytes as reading the operator's host buffer. Both are ordinary
LPDDR5X on Grace, and the spike's par copy reads the pinned bytes.

| backend, N, precision, p | evaluation (T9) | download (T9) | serial copy removed by the view | evaluation, model | load (T9) | `to_vec` removed by `write_owned` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA 10⁶ f32 p = 3 | 6.63 | 1.25 | 0.93 | ≈ 5.7 | 0.72 | 0.18 |
| CUDA 10⁶ f64 p = 3 | 10.19 | 2.25 | ≈ 1.9 (32 MB; not measured) | ≈ 8.3 | 1.17 | 0.36 |
| CUDA 10⁷ f32 p = 3 | 60.35 | 12.69 | 11.03 | ≈ 49.3 | 6.62 | 2.97 |
| CUDA 10⁷ f32 p = 8 | 153.95 | 12.66 | 11.03 | ≈ 142.9 | 6.67 | 2.97 |
| CUDA 10⁷ f64 p = 3 | 102.99 | 24.61 | 22.41 | ≈ 80.6 | 13.58 | 6.38 |
| CUDA 10⁷ f64 p = 8 | 268.82 | 24.81 | 22.41 | ≈ 246.4 | 13.30 | 6.38 |
| Metal 10⁶ f32 p = 3 | 14.06 | 0.67 | 0.25 | ≈ 13.8 | 1.67 | 0.06 |

The "evaluation, model" column removes the copy only. With `write_owned` the `to_vec`
column would also go, less the cost of filling a fresh buffer that is not zeroed. On
Grace a fresh destination added 15–20% to a serial copy. In the parallel gather that
cost spreads over the threads, but it is not measured here; item 3 measures it.

## Proposal for decision 14

Signed off on 2026-10-08 as proposed, with one change in the building: `HostValues<E>`
owns CubeCL's bytes without a lifetime. A view that borrows `&mut Device` cannot be
returned from `nd-fmm-exec`'s `read_output` on its success path while the error path still
reads the device's counters (the borrow checker's conditional-return limitation, with no
safe way round it), so the borrow moved up a level: `DeviceOutput` borrows the operator,
and no view outlives its evaluation. The rest is as below.

### `nd-fmm-kernels`

```rust
/// Values downloaded from a device, read in place in CubeCL's host memory (pinned memory
/// from CUDA's pool, a mapped staging buffer on wgpu, host memory on the CPU runtime),
/// without a copy into a caller's slice. Derefs to `&[E]`. It borrows the device, so it
/// is dropped, and its memory returned to CubeCL's pool, before the device is used again.
pub struct HostValues<'d, E: DeviceElement> {
    bytes: Option<cubecl::bytes::Bytes>, // None for an empty range
    _device: PhantomData<&'d mut Device>,
    _element: PhantomData<E>,
}

impl<E: DeviceElement> Deref for HostValues<'_, E> {
    type Target = [E];
    // `E::from_bytes(&bytes)` (CubeElement's safe bytemuck cast, as `download` uses now),
    // or `&[]` for an empty range.
}

impl Device {
    /// Downloads `slice` and lends CubeCL's host copy of it: waits for every queued launch
    /// (one sync) and returns any launch error attached to the buffer (device-path.md §12).
    /// Counts one download of `slice.len()` elements and one sync, as `download` does.
    ///
    /// # Errors
    /// As `download`: `WrongDevice`; `Device` with CubeCL's message.
    pub fn download_view<E: DeviceElement>(
        &mut self,
        slice: DeviceSlice<'_, E>,
    ) -> Result<HostValues<'_, E>, KernelError>;
}
```

- **No new `unsafe`.** The cast is `CubeElement::from_bytes`, which `download` already
  uses. Every function stays safe, and no dependency is added.
- **The view borrows `&mut Device`.** That is a choice and could be a shared borrow. With
  it, a view cannot outlive the next device operation, so it cannot hold a pinned or
  staging slice across a later download. The pools never need a second slot because of a
  view, and the steady state measured above stays as it is.
- **`download` becomes `out.copy_from_slice(&self.download_view(slice)?)`**, with the
  same counters, checks, errors and length assertion. There is then one code path, and
  the test "the view equals `download` bit for bit" compares two uses of it with the
  data.
- **Tests, as in the brief.**
  - f32, f64 and u32, at 0, 1, 7 and 10⁵ elements, at odd offsets inside larger buffers;
  - the counters;
  - `WrongDevice`;
  - a launch error that surfaces at the view;
  - the CPU runtime in CI, Metal and CUDA by hand.

### `nd-fmm-exec`

- **`DeviceOutput<'a, T>` carries views.** `CallerOrder(HostValues<'a, T>)` holds the
  output pass on the device (CUDA, the CPU runtime). `copy_output` reads the view in
  parallel, as it reads the host buffer now. `LeafOrder { values: HostValues<'a, T>,
  layout }` holds the host pass (Metal, `OutputPass::Host`).
- **The host pass reads a slice.** The pass generalised over the leaf-ordered slice and
  the store's point offsets and point size, in place of `&LeafStore<T>`, with no change
  to `nd-fmm-plan`.
- **The host output buffer moves into `Mirrors`.** `self.output: LeafStore<T>` is today
  the download target and also the host-fallback mirror of the target output
  (`fetch_output`, `send_output`). It moves into `Mirrors`, which exist only when some
  kind has `Placement::Host`. Its layout (offsets, point size) stays for
  `output_values` and for `download_target_output` (the test oracle, which then builds
  its `LeafStore` from the layout).
  - Without a host fallback, which is the default on every backend at the benchmark,
    this saves o N_t s bytes of host memory: 16 / 32 MB at N = 10⁶ and 160 / 320 MB at
    N = 10⁷ (f32 / f64, gradients).
  - A test checks that the operator holds no host output store without a fallback.
- **The transfer formula of device-path.md §4.1 does not change.** One upload, one
  download and one sync per evaluation; only the destination of the download changes.

### The owned upload (item 3)

`Device::write_owned<E>(&mut self, slice: DeviceSliceMut<'_, E>, data: Vec<E>) ->
Result<(), KernelError>` has the checks, counters, panics and errors of `write`, and hands
`data` to `Bytes::from_elems`. `write` stays.

`DeviceOperator::begin_evaluation` would take the charges as a `Vec<T>`. The operator is
public but is not `Fmm`'s API, and `Fmm::evaluate` and `evaluate_into` do not change.
`Fmm` would gather into a fresh `Vec` with rayon's `collect_into_vec`, which fills spare
capacity with no zero fill, on its pool. It gives up the reused `leaf_charges` buffer on
the device path only.

Recommended, from the CUDA numbers (2.97 / 6.38 ms at N = 10⁷). Item 3 measures the fresh
allocation's faults against them.

## Decision 15

Not needed: neither host pool allocates again in steady state (above). Nothing in CubeCL
is changed.

## Upstream (an observation, no question filed)

Recorded as entry 1 of docs/design/cubecl-upstream.md (2026-10-08), the collection of
CubeCL observations for a later summary to the CubeCL developers.

The pinned staging of uploads below `STAGE_MAX` (100 MiB, a constant in cubecl-server
`command/staging.rs`) cost more than it saved on the GH200 here:
- a 40 MB write spends 2.73 ms after `Client::write` returns;
- a 160 MB write, sent without staging, spends 1.65 ms (measured (locust, CUDA)).

The public API has no switch for it. `Bytes` that are already pinned skip the staging,
but only CubeCL makes such `Bytes`: `read`'s results, or `Client::staging`, which copies.
On a machine with a coherent CPU–GPU link the threshold may want to be lower, or
configurable. It goes to the CubeCL developers in that summary, not as a question of
its own. It is not a pool reallocation, so it is not decision 15. Moving off
`=0.11.0-pre.4` stays a separate decision either way.

## Checks run for the spike

- `cargo fmt --all`;
- `cargo clippy -p nd-fmm-spike-download-path -- -D warnings`, and the same with
  `--features cpu,metal,cuda`. Both are clean on the M3 Max, and the CUDA code was
  type-checked there;
- `cargo build --release -p nd-fmm-spike-download-path --features cuda` on locust, and
  the runs above, which are the spike's own checks: every download is compared with the
  fill and between the copies, and every upload is read back bit for bit;
- no library code changed, so the root, kernel and exec checks were not run for this
  step. They belong to items 2 and 3.

The spike is a workspace member and not a default member, so CI does not build it.
`Cargo.lock` gains its package entry.

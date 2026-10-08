# Raw output: locust (GH200), CUDA, 2026-10-08

Written by `spikes/download-path/run-gh200.sh` into `/data/ucahtbe/logs/t11/` (10:02:59–10:03:50 UTC); REPORT.md has the setup, the load checks and the discussion. Each block below is one process.

### `cuda-main.md`

#### Download and upload paths (Phase 4S T11 spike)

- Host: Neoverse-V2 (implementer 0x41, part 0xd4f) (72 physical, 72 logical), aarch64-linux, release build, rustc 1.99.0 (b940084d7 2026-09-28); rayon threads 72.
- Arguments: --backends cuda --precisions f32,f64.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64.

#### Download, cuda, f32

Median (max) over 10 downloads after 2 warm-ups, ms. A fill kernel writes the buffer before each download. `sync`: the wait for it; `read_one`: `Client::read_one` after that sync; `copy`: `copy_from_slice` into a reused, touched host slice (one thread); `par copy`: the same on rayon's pool (72 threads); `cold copy`: a fresh `vec!` and the copy into it; `drop`: dropping CubeCL's `Bytes`; `unsynced`: `read_one` and the copy without the sync first, as `Device::download` does it, the fill's wait included.

| MB | sync | read_one | copy | par copy | cold copy | drop | unsynced | read_one GB/s | copy GB/s | par copy GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.140 (0.155) | 0.070 (0.077) | 0.933 (0.936) | 0.170 (0.178) | 0.800 (0.829) | 0.000 (0.000) | 1.108 (1.231) | 228.6 | 17.2 | 94.4 |
| 160 | 0.182 (0.251) | 0.554 (0.555) | 11.027 (11.167) | 1.090 (1.166) | 13.207 (13.454) | 0.002 (0.004) | 11.779 (12.436) | 289.0 | 14.5 | 146.8 |
| 320 | 0.183 (0.222) | 1.091 (1.092) | 22.415 (22.588) | 2.327 (2.367) | 27.080 (27.607) | 0.002 (0.002) | 23.691 (23.850) | 293.3 | 14.3 | 137.5 |

#### Upload, cuda, f32

Median (max) over 10 uploads after 2 warm-ups, ms, each from an idle stream. `to_vec`: `Device::write`'s copy of the caller's slice; `from_elems`: `Bytes::from_elems` of that `Vec`; `write`: `Client::write` until it returns; `sync`: the transfer, to a following sync; `owned`: `from_elems`, `write` and `sync` of a `Vec` made before the timing (no `to_vec`). Each path's device contents are read back once and compared bit for bit with the source.

| MB | to_vec | from_elems | write | sync | total | owned | to_vec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.183 (0.188) | 0.000 (0.000) | 0.001 (0.001) | 0.384 (0.396) | 0.571 (0.578) | 0.266 (0.275) | 21.8 |
| 8 | 0.305 (0.308) | 0.000 (0.000) | 0.000 (0.001) | 0.655 (0.664) | 0.961 (0.973) | 0.458 (0.467) | 26.2 |
| 40 | 2.973 (3.048) | 0.001 (0.001) | 0.001 (0.002) | 2.732 (2.796) | 5.699 (5.827) | 2.578 (2.645) | 13.5 |
| 80 | 6.268 (6.313) | 0.001 (0.001) | 0.001 (0.001) | 5.137 (5.243) | 11.414 (11.529) | 5.071 (5.126) | 12.8 |
| 160 | 12.242 (14.302) | 0.001 (0.003) | 0.001 (0.003) | 1.650 (2.048) | 13.930 (16.357) | 1.460 (1.645) | 13.1 |

#### Zeroing, cuda, f32

The three zero kernels of `begin_evaluation` at the benchmark's sizes (gradients on: the target output 4 N values; multipoles and locals (p + 1)² values per box, every level), median (max) over 10 evaluations after 2 warm-ups, ms, by timing windows (on the device): one window per kernel, then one over the three.

| N | p | multipoles MB | output MB | multipoles | locals | output | all three | GB/s (all) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000000 | 3 | 2.4 | 16.0 | 0.017 (0.017) | 0.016 (0.017) | 0.025 (0.025) | 0.031 (0.032) | 680.4 |
| 1000000 | 8 | 12.1 | 16.0 | 0.023 (0.023) | 0.018 (0.020) | 0.021 (0.021) | 0.043 (0.044) | 937.7 |
| 10000000 | 3 | 19.2 | 160.0 | 0.025 (0.026) | 0.017 (0.017) | 0.081 (0.081) | 0.112 (0.112) | 1775.9 |
| 10000000 | 8 | 97.1 | 160.0 | 0.068 (0.068) | 0.059 (0.059) | 0.081 (0.081) | 0.197 (0.197) | 1802.1 |

#### `Device::write` and `Device::download`, cuda, f32

The production path, median (max) over 10 calls after 2 warm-ups, ms. `write`: `Device::write` and a `Device::sync`; `download`: `Device::download` from an idle stream (its own sync included) into a reused slice; counters per call.

| MB | write + sync | download | download GB/s | counters (write) | counters (download) |
| ---: | ---: | ---: | ---: | --- | --- |
| 4 | 0.525 (0.530) | 0.258 (0.264) | 15.5 | up 1 (4000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (4000000 B), syncs 1 |
| 8 | 0.739 (1.500) | 0.505 (0.516) | 15.8 | up 1 (8000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (8000000 B), syncs 1 |
| 16 | 1.673 (1.680) | 1.003 (1.637) | 15.9 | up 1 (16000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (16000000 B), syncs 1 |
| 40 | 5.681 (5.768) | 2.620 (2.687) | 15.3 | up 1 (40000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (40000000 B), syncs 1 |
| 80 | 12.163 (12.540) | 5.630 (5.728) | 14.2 | up 1 (80000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (80000000 B), syncs 1 |
| 160 | 14.322 (17.689) | 11.658 (12.479) | 13.7 | up 1 (160000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (160000000 B), syncs 1 |
| 320 | 27.556 (30.108) | 23.691 (25.140) | 13.5 | up 1 (320000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (320000000 B), syncs 1 |

#### Download, cuda, f64

Median (max) over 10 downloads after 2 warm-ups, ms. A fill kernel writes the buffer before each download. `sync`: the wait for it; `read_one`: `Client::read_one` after that sync; `copy`: `copy_from_slice` into a reused, touched host slice (one thread); `par copy`: the same on rayon's pool (72 threads); `cold copy`: a fresh `vec!` and the copy into it; `drop`: dropping CubeCL's `Bytes`; `unsynced`: `read_one` and the copy without the sync first, as `Device::download` does it, the fill's wait included.

| MB | sync | read_one | copy | par copy | cold copy | drop | unsynced | read_one GB/s | copy GB/s | par copy GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.059 (0.081) | 0.068 (0.069) | 0.999 (1.017) | 0.305 (0.385) | 1.051 (1.085) | 0.000 (0.001) | 1.202 (1.283) | 235.5 | 16.0 | 52.4 |
| 160 | 0.138 (0.249) | 0.554 (0.556) | 11.149 (11.358) | 1.108 (1.856) | 12.978 (13.170) | 0.001 (0.006) | 11.936 (13.179) | 288.9 | 14.4 | 144.4 |
| 320 | 0.199 (0.246) | 1.091 (1.093) | 22.414 (22.675) | 2.256 (2.369) | 26.666 (27.237) | 0.002 (0.006) | 23.783 (23.833) | 293.2 | 14.3 | 141.8 |

#### Upload, cuda, f64

Median (max) over 10 uploads after 2 warm-ups, ms, each from an idle stream. `to_vec`: `Device::write`'s copy of the caller's slice; `from_elems`: `Bytes::from_elems` of that `Vec`; `write`: `Client::write` until it returns; `sync`: the transfer, to a following sync; `owned`: `from_elems`, `write` and `sync` of a `Vec` made before the timing (no `to_vec`). Each path's device contents are read back once and compared bit for bit with the source.

| MB | to_vec | from_elems | write | sync | total | owned | to_vec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.185 (0.198) | 0.000 (0.000) | 0.001 (0.001) | 0.389 (2.948) | 0.577 (3.138) | 0.272 (0.290) | 21.6 |
| 8 | 0.359 (0.363) | 0.000 (0.000) | 0.001 (0.001) | 0.683 (0.689) | 1.042 (1.047) | 0.539 (0.544) | 22.3 |
| 40 | 3.001 (3.143) | 0.001 (0.001) | 0.001 (0.001) | 2.818 (2.967) | 5.843 (6.112) | 2.759 (2.841) | 13.3 |
| 80 | 6.384 (6.393) | 0.000 (0.001) | 0.001 (0.002) | 4.986 (5.118) | 11.371 (11.513) | 4.843 (4.984) | 12.5 |
| 160 | 12.609 (14.282) | 0.001 (0.001) | 0.001 (0.002) | 1.908 (2.441) | 14.456 (16.726) | 1.749 (1.857) | 12.7 |

#### Zeroing, cuda, f64

The three zero kernels of `begin_evaluation` at the benchmark's sizes (gradients on: the target output 4 N values; multipoles and locals (p + 1)² values per box, every level), median (max) over 10 evaluations after 2 warm-ups, ms, by timing windows (on the device): one window per kernel, then one over the three.

| N | p | multipoles MB | output MB | multipoles | locals | output | all three | GB/s (all) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000000 | 3 | 4.8 | 32.0 | 0.017 (0.017) | 0.017 (0.017) | 0.025 (0.026) | 0.031 (0.035) | 1343.9 |
| 1000000 | 8 | 24.3 | 32.0 | 0.023 (0.024) | 0.019 (0.020) | 0.021 (0.022) | 0.043 (0.044) | 1863.5 |
| 10000000 | 3 | 38.4 | 320.0 | 0.028 (0.028) | 0.018 (0.019) | 0.087 (0.088) | 0.115 (0.115) | 3454.3 |
| 10000000 | 8 | 194.2 | 320.0 | 0.067 (0.068) | 0.059 (0.060) | 0.084 (0.085) | 0.201 (0.201) | 3532.0 |

#### `Device::write` and `Device::download`, cuda, f64

The production path, median (max) over 10 calls after 2 warm-ups, ms. `write`: `Device::write` and a `Device::sync`; `download`: `Device::download` from an idle stream (its own sync included) into a reused slice; counters per call.

| MB | write + sync | download | download GB/s | counters (write) | counters (download) |
| ---: | ---: | ---: | ---: | --- | --- |
| 4 | 0.595 (0.598) | 0.267 (0.269) | 15.0 | up 1 (4000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (4000000 B), syncs 1 |
| 8 | 0.760 (1.958) | 0.525 (0.540) | 15.2 | up 1 (8000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (8000000 B), syncs 1 |
| 16 | 1.594 (1.598) | 1.034 (1.056) | 15.5 | up 1 (16000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (16000000 B), syncs 1 |
| 40 | 5.675 (5.767) | 2.679 (2.708) | 14.9 | up 1 (40000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (40000000 B), syncs 1 |
| 80 | 12.198 (13.069) | 5.641 (5.700) | 14.2 | up 1 (80000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (80000000 B), syncs 1 |
| 160 | 14.946 (15.602) | 11.282 (11.375) | 14.2 | up 1 (160000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (160000000 B), syncs 1 |
| 320 | 28.498 (29.305) | 23.086 (23.799) | 13.9 | up 1 (320000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (320000000 B), syncs 1 |

### `cuda-pool-f32.md`

#### Download and upload paths (Phase 4S T11 spike)

- Host: Neoverse-V2 (implementer 0x41, part 0xd4f) (72 physical, 72 logical), aarch64-linux, release build, rustc 1.99.0 (b940084d7 2026-09-28); rayon threads 72.
- Arguments: --backends cuda --precisions f32 --sections pool.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64.

#### Host pool, cuda, f32

`read_one` per download in sequence (ms), after a fill and a sync, in a process that ran the sections pool in that order. `steady`: 10 downloads of 160 MB (the output of N = 10⁷ with gradients); `mixed`: the sizes [16.0, 160.0, 320.0] MB in turn; `evaluation`: per iteration a `Client::write` of 40 MB (the charges of N = 10⁷), a fill and a download of 160 MB; `persistent`: `steady` with each `read_one` inside `Client::memory_persistent_allocation`.

- steady: 61.14, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55
- mixed: 14.63 (16), 0.55 (160), 1.11 (320), 0.08 (16), 0.55 (160), 1.09 (320), 0.08 (16), 0.55 (160), 1.09 (320), 0.07 (16)
- evaluation: 0.55, 0.55, 0.55, 0.55, 0.55, 0.56, 0.55, 0.55, 0.55, 0.55
- persistent: 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55

### `cuda-pool-f64.md`

#### Download and upload paths (Phase 4S T11 spike)

- Host: Neoverse-V2 (implementer 0x41, part 0xd4f) (72 physical, 72 logical), aarch64-linux, release build, rustc 1.99.0 (b940084d7 2026-09-28); rayon threads 72.
- Arguments: --backends cuda --precisions f64 --sections pool.
- Device: cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64.

#### Host pool, cuda, f64

`read_one` per download in sequence (ms), after a fill and a sync, in a process that ran the sections pool in that order. `steady`: 10 downloads of 320 MB (the output of N = 10⁷ with gradients); `mixed`: the sizes [16.0, 160.0, 320.0] MB in turn; `evaluation`: per iteration a `Client::write` of 80 MB (the charges of N = 10⁷), a fill and a download of 320 MB; `persistent`: `steady` with each `read_one` inside `Client::memory_persistent_allocation`.

- steady: 16.44, 1.10, 1.09, 1.10, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09
- mixed: 0.07 (16), 60.60 (160), 1.09 (320), 0.07 (16), 0.55 (160), 1.09 (320), 0.07 (16), 0.55 (160), 1.09 (320), 0.07 (16)
- evaluation: 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09
- persistent: 1.10, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09

### CUDA API summary under `nsys` (`nsys stats --report cuda_api_sum`), pinned allocations and copies

| run | cuMemAllocHost_v2 calls | total ms | min ms | max ms | cuMemcpyHtoDAsync_v2 calls | cuMemcpyDtoHAsync_v2 calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| steady-1 | 1 | 57.9 | 57.9 | 57.9 | 0 | 1 |
| steady-10 | 1 | 60.5 | 60.5 | 60.5 | 0 | 10 |
| evaluation-1 | 2 | 74.9 | 15.3 | 59.6 | 1 | 1 |
| evaluation-10 | 2 | 72.6 | 14.8 | 57.8 | 10 | 10 |
| mixed-10 | 2 | 74.6 | 15.3 | 59.3 | 0 | 10 |
| persistent-10 | 1 | 58.7 | 58.7 | 58.7 | 0 | 10 |

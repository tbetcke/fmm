# Raw output: the M3 Max, Metal, 2026-10-08

Run outside the sandbox at 11:04:49–11:04:55 BST (load average 2.6–3.1, a desktop session; an earlier run at 10:59 under a load average of 13 gave the same numbers within 10%). REPORT.md has the discussion. Each block below is one process.

### `metal-main.md`

#### Download and upload paths (Phase 4S T11 spike)

- Host: Apple M3 Max (16 physical, 16 logical), aarch64-macos, release build, rustc 1.99.0 (b940084d7 2026-09-28); rayon threads 16.
- Arguments: --backends metal --precisions f32.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32.

#### Download, metal, f32

Median (max) over 10 downloads after 2 warm-ups, ms. A fill kernel writes the buffer before each download. `sync`: the wait for it; `read_one`: `Client::read_one` after that sync; `copy`: `copy_from_slice` into a reused, touched host slice (one thread); `par copy`: the same on rayon's pool (16 threads); `cold copy`: a fresh `vec!` and the copy into it; `drop`: dropping CubeCL's `Bytes`; `unsynced`: `read_one` and the copy without the sync first, as `Device::download` does it, the fill's wait included.

| MB | sync | read_one | copy | par copy | cold copy | drop | unsynced | read_one GB/s | copy GB/s | par copy GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.539 (0.734) | 0.466 (0.576) | 0.249 (0.280) | 0.339 (1.903) | 0.364 (0.446) | 0.000 (0.001) | 1.318 (1.459) | 34.4 | 64.1 | 47.2 |
| 160 | 1.244 (1.402) | 1.694 (1.756) | 2.529 (2.624) | 1.870 (2.141) | 3.311 (6.377) | 0.001 (0.003) | 4.924 (5.020) | 94.5 | 63.3 | 85.6 |
| 320 | 1.760 (1.852) | 2.246 (2.777) | 5.156 (5.741) | 2.900 (3.083) | 7.348 (12.120) | 0.003 (0.005) | 9.205 (9.585) | 142.5 | 62.1 | 110.3 |

#### Upload, metal, f32

Median (max) over 10 uploads after 2 warm-ups, ms, each from an idle stream. `to_vec`: `Device::write`'s copy of the caller's slice; `from_elems`: `Bytes::from_elems` of that `Vec`; `write`: `Client::write` until it returns; `sync`: the transfer, to a following sync; `owned`: `from_elems`, `write` and `sync` of a `Vec` made before the timing (no `to_vec`). Each path's device contents are read back once and compared bit for bit with the source.

| MB | to_vec | from_elems | write | sync | total | owned | to_vec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.056 (0.062) | 0.000 (0.000) | 0.000 (0.000) | 0.707 (0.831) | 0.762 (0.886) | 0.673 (0.794) | 70.9 |
| 8 | 0.125 (0.139) | 0.000 (0.000) | 0.000 (0.000) | 1.256 (1.326) | 1.387 (1.453) | 1.158 (1.236) | 63.9 |
| 40 | 0.644 (0.651) | 0.000 (0.000) | 0.000 (0.000) | 5.003 (5.265) | 5.647 (5.911) | 4.965 (5.225) | 62.1 |
| 80 | 1.331 (1.367) | 0.000 (0.001) | 0.001 (0.001) | 10.007 (11.496) | 11.349 (12.794) | 9.806 (11.859) | 60.1 |
| 160 | 2.619 (2.830) | 0.000 (0.001) | 0.001 (0.002) | 18.493 (21.027) | 21.115 (23.859) | 18.209 (21.094) | 61.1 |

#### Zeroing, metal, f32

The three zero kernels of `begin_evaluation` at the benchmark's sizes (gradients on: the target output 4 N values; multipoles and locals (p + 1)² values per box, every level), median (max) over 10 evaluations after 2 warm-ups, ms, by timing windows (on the device): one window per kernel, then one over the three.

| N | p | multipoles MB | output MB | multipoles | locals | output | all three | GB/s (all) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000000 | 3 | 2.4 | 16.0 | 0.996 (1.123) | 0.085 (0.152) | 0.088 (0.092) | 0.253 (0.337) | 82.3 |
| 1000000 | 8 | 12.1 | 16.0 | 0.773 (1.236) | 0.178 (0.203) | 0.076 (0.118) | 0.314 (0.553) | 128.1 |
| 10000000 | 3 | 19.2 | 160.0 | 0.995 (1.160) | 0.188 (0.467) | 0.437 (0.577) | 1.067 (1.152) | 185.9 |
| 10000000 | 8 | 97.1 | 160.0 | 1.020 (1.070) | 0.280 (0.298) | 0.276 (0.496) | 0.443 (0.496) | 798.6 |

#### `Device::write` and `Device::download`, metal, f32

The production path, median (max) over 10 calls after 2 warm-ups, ms. `write`: `Device::write` and a `Device::sync`; `download`: `Device::download` from an idle stream (its own sync included) into a reused slice; counters per call.

| MB | write + sync | download | download GB/s | counters (write) | counters (download) |
| ---: | ---: | ---: | ---: | --- | --- |
| 4 | 0.763 (0.955) | 0.291 (0.302) | 13.8 | up 1 (4000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (4000000 B), syncs 1 |
| 8 | 1.805 (2.956) | 0.385 (0.486) | 20.8 | up 1 (8000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (8000000 B), syncs 1 |
| 16 | 2.447 (2.584) | 0.557 (0.624) | 28.7 | up 1 (16000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (16000000 B), syncs 1 |
| 40 | 5.272 (7.054) | 1.480 (1.902) | 27.0 | up 1 (40000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (40000000 B), syncs 1 |
| 80 | 12.211 (15.146) | 2.430 (2.792) | 32.9 | up 1 (80000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (80000000 B), syncs 1 |
| 160 | 22.036 (25.328) | 4.222 (4.486) | 37.9 | up 1 (160000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (160000000 B), syncs 1 |
| 320 | 42.824 (47.310) | 7.768 (8.592) | 41.2 | up 1 (320000000 B), down 0 (0 B), syncs 1 | up 0 (0 B), down 1 (320000000 B), syncs 1 |

### `metal-pool.md`

#### Download and upload paths (Phase 4S T11 spike)

- Host: Apple M3 Max (16 physical, 16 logical), aarch64-macos, release build, rustc 1.99.0 (b940084d7 2026-09-28); rayon threads 16.
- Arguments: --backends metal --precisions f32 --sections pool.
- Device: metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32.

#### Host pool, metal, f32

`read_one` per download in sequence (ms), after a fill and a sync, in a process that ran the sections pool in that order. `steady`: 10 downloads of 160 MB (the output of N = 10⁷ with gradients); `mixed`: the sizes [16.0, 160.0, 320.0] MB in turn; `evaluation`: per iteration a `Client::write` of 40 MB (the charges of N = 10⁷), a fill and a download of 160 MB; `persistent`: `steady` with each `read_one` inside `Client::memory_persistent_allocation`.

- steady: 5.17, 1.56, 1.24, 1.36, 1.21, 1.22, 1.30, 1.21, 1.22, 1.21
- mixed: 0.70 (16), 1.19 (160), 11.01 (320), 0.42 (16), 1.30 (160), 2.25 (320), 0.45 (16), 1.18 (160), 2.22 (320), 0.46 (16)
- evaluation: 1.24, 1.21, 1.32, 1.28, 1.25, 1.25, 1.21, 1.38, 1.25, 1.44
- persistent: 1.96, 1.47, 1.44, 1.42, 1.42, 1.32, 1.32, 1.43, 1.43, 1.68

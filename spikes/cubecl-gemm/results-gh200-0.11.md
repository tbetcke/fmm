# Raw results: NVIDIA GH200 (locust), CUDA, CubeCL 0.11.0-pre.4, 2026-10-06

Output of the spike binary (`src/main.rs`), unchanged since Phase 4 T2, built against
`cubecl =0.11.0-pre.4`, `cubek-matmul =0.3.0-pre.4` and `cubek-std =0.3.0-pre.4` with
`--no-default-features --features cuda` (Phase 4S T2), unedited. Median time per GEMM
over batches of back-to-back launches with one sync per batch. Every measured row was
checked against the plain-Rust f64 reference (`rel. error` = max-norm relative error;
tolerance 1e-12 for f64, 1e-5 for f32). The comparison with Metal, the CPU runtime and
the roofline predictions is in SPIKE_REPORT.md, section "CUDA on GH200".

Machine: locust (`locust.rc.ucl.ac.uk`), NVIDIA GH200 480GB: one H100 (compute
capability 9.0, 132 SMs as CubeCL reports them, 96 GB HBM3, driver 565.57.01), 72
Neoverse-V2 cores, RHEL 9.3; the environment of `tools/gh200/` (CUDA 12.6.3, rustc
1.99.0, release profile, linked by rust-lld). CUDA through CubeCL's default LLVM NVPTX
path. `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`RAYON_NUM_THREADS` set to 1. The spike's device line (stderr): `runtime cuda: f64
supported = true, plane size 32..32, max shared memory 232448 B, SMs Some(132), CPU
cores None, tensor-core min dim Some(8)`.

Load, checked before and after (Phase 4S README, "Timing"): at 07:05 BST, `nvidia-smi`
listed no compute process, 0% utilisation, 1 MiB used, 91 W; `uptime` load average 1.5
(this session's builds just before); the only other user's process was an `nvitop`
monitor at 0.8% of one core. At 07:09, after the run: no compute process, 2 MiB, the
utilisation sample 23% and 403 W from the run that had just ended; load average 1.2, the
same `nvitop`. The GPU clocks are not locked (no administrator access): `nvidia-smi -q
-d CLOCK` before the run showed the idle SM clock at 345 MHz, application and maximum
clocks 1980 MHz (SM) and 2619 MHz (memory); the clock during the run was not recorded.

Run (the built binary was run directly; `cargo run -p nd-fmm-spike-cubecl-gemm --release
--no-default-features --features cuda -- --backends cuda` is equivalent): `--backends
cuda`, both precisions (f64 runs without `--force-f64`), 155 s wall. First launches
(compilation included, stderr): 144, 4.4 s in all, the slowest 0.89 s
(`lib:matmul_double_unit_max_tile_size`, f64 p = 4).

| backend | precision | p | Nc | B | implementation | median ms | reps | GFLOP/s | rel. error | check |
|---|---|---|---|---|---|---|---|---|---|---|
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_auto | 0.0159 | 51200 | 78.6 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0152 | 51200 | 82.4 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0162 | 51200 | 77.3 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | 0.0049 | 51200 | 253.2 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0077 | 51200 | 163.1 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0077 | 51200 | 162.2 | 2.22e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_auto | 0.0152 | 51200 | 822.2 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0152 | 51200 | 822.9 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0172 | 51200 | 727.0 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 10000 | tiled-smem(tm=2) | 0.0053 | 51200 | 2372.4 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0079 | 51200 | 1588.5 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0090 | 51200 | 1394.8 | 1.93e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_auto | 0.0385 | 26624 | 3248.4 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.0384 | 26624 | 3251.7 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_double_unit_max_tile_size | 0.0864 | 12288 | 1447.2 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 4 | 25 | 100000 | tiled-smem(tm=2) | 0.0178 | 51200 | 7038.7 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.0133 | 51200 | 9378.8 | 2.03e-7 | pass |
| cuda | f32 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.0334 | 30720 | 3743.9 | 2.03e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_auto | 0.0165 | 51200 | 795.0 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0165 | 51200 | 795.0 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0311 | 32768 | 422.3 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | 0.0112 | 51200 | 1172.0 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0200 | 50176 | 655.4 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0202 | 50176 | 648.7 | 3.27e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_auto | 0.0171 | 51200 | 7692.6 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0171 | 51200 | 7691.7 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0456 | 22528 | 2880.7 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 10000 | tiled-smem(tm=6) | 0.0178 | 51200 | 7377.3 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0226 | 45056 | 5802.1 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0251 | 39936 | 5221.9 | 4.38e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_auto | 0.1159 | 8704 | 11320.5 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.1159 | 8704 | 11325.3 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_double_unit_max_tile_size | 0.2644 | 3840 | 4963.5 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 8 | 81 | 100000 | tiled-smem(tm=6) | 0.0930 | 11264 | 14115.6 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.0970 | 11264 | 13523.0 | 4.03e-7 | pass |
| cuda | f32 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.1696 | 6144 | 7736.7 | 4.03e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_auto | 0.0298 | 33792 | 1917.1 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0298 | 33792 | 1919.7 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0642 | 16384 | 890.3 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 1000 | tiled-smem(tm=11) | 0.0274 | 36864 | 2084.8 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0399 | 25600 | 1432.7 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0443 | 23552 | 1289.6 | 6.71e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_auto | 0.0532 | 19456 | 10731.0 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0531 | 19456 | 10750.2 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_double_unit_max_tile_size | 0.1158 | 8704 | 4931.1 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 10000 | tiled-smem(tm=11) | 0.0461 | 22528 | 12396.5 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0513 | 20480 | 11139.5 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0696 | 15360 | 8202.4 | 7.27e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_auto | 0.4208 | 2432 | 13574.1 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.4208 | 2432 | 13574.2 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_double_unit_max_tile_size | 1.0401 | 1024 | 5491.9 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 12 | 169 | 100000 | tiled-smem(tm=11) | 0.2671 | 3840 | 21383.5 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.3490 | 3072 | 16369.3 | 6.95e-7 | pass |
| cuda | f32 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.6161 | 1664 | 9271.4 | 6.95e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_auto | 0.0464 | 22528 | 3598.8 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0464 | 22528 | 3600.6 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_double_unit_max_tile_size | 0.1097 | 9216 | 1522.6 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 1000 | tiled-smem(tm=10) | 0.0453 | 22528 | 3690.4 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0672 | 15360 | 2485.6 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0754 | 13312 | 2214.4 | 6.94e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_auto | 0.0925 | 11264 | 18063.3 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0924 | 11264 | 18081.7 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_double_unit_max_tile_size | 0.2811 | 3584 | 5942.5 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 10000 | tiled-smem(tm=10) | 0.1192 | 8704 | 14012.8 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0955 | 11264 | 17499.4 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.1868 | 5632 | 8940.3 | 7.38e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_auto | 1.0169 | 1024 | 16426.5 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 1.0162 | 1024 | 16438.4 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 2.7409 | 384 | 6094.4 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 0.9114 | 1152 | 18328.2 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.9772 | 1024 | 17093.6 | 7.88e-7 | pass |
| cuda | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 1.7800 | 576 | 9384.5 | 7.88e-7 | pass |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_auto | 0.0170 | 51200 | 73.6 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0169 | 51200 | 73.9 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0213 | 47104 | 58.6 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 1000 | tiled-smem(tm=2) | 0.0049 | 51200 | 253.0 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0092 | 51200 | 135.4 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0119 | 51200 | 105.3 | 2.62e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_auto | 0.0174 | 51200 | 719.8 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0173 | 51200 | 721.1 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0362 | 27648 | 344.8 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 10000 | tiled-smem(tm=2) | 0.0068 | 51200 | 1832.6 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0103 | 51200 | 1218.1 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0128 | 51200 | 976.3 | 3.28e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_auto | 0.0994 | 10240 | 1257.6 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.0993 | 10240 | 1258.3 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_double_unit_max_tile_size | 0.2140 | 4864 | 584.2 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 4 | 25 | 100000 | tiled-smem(tm=2) | 0.0287 | 35840 | 4351.8 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.0400 | 25600 | 3122.2 | 3.83e-16 | pass |
| cuda | f64 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.0708 | 14336 | 1765.3 | 3.83e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_auto | 0.0359 | 28672 | 365.1 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0359 | 28672 | 365.3 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0531 | 19456 | 247.2 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 1000 | tiled-smem(tm=6) | 0.0166 | 51200 | 792.8 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0244 | 41984 | 538.6 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0348 | 29696 | 376.7 | 3.33e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_auto | 0.0372 | 27648 | 3529.1 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0371 | 27648 | 3537.8 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0860 | 12288 | 1525.5 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 10000 | tiled-smem(tm=6) | 0.0292 | 34816 | 4492.3 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0295 | 34816 | 4448.0 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0417 | 24576 | 3143.4 | 3.89e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_auto | 0.2379 | 4352 | 5516.8 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.2377 | 4352 | 5519.7 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_double_unit_max_tile_size | 0.5389 | 1920 | 2435.2 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 8 | 81 | 100000 | tiled-smem(tm=6) | 0.1603 | 6656 | 8187.4 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.2412 | 4352 | 5440.4 | 4.25e-16 | pass |
| cuda | f64 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.2971 | 3584 | 4416.2 | 4.25e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_auto | 0.0643 | 16384 | 887.8 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0643 | 16384 | 888.4 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0984 | 10240 | 580.3 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 1000 | tiled-smem(tm=11) | 0.0591 | 17408 | 966.9 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0527 | 19456 | 1083.5 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0719 | 14336 | 794.1 | 4.82e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_auto | 0.1209 | 8704 | 4726.3 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.1209 | 8704 | 4725.6 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_double_unit_max_tile_size | 0.2353 | 4352 | 2427.2 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 10000 | tiled-smem(tm=11) | 0.1167 | 8704 | 4894.7 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0941 | 11264 | 6071.3 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0984 | 10240 | 5803.1 | 5.15e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_auto | 0.8582 | 1216 | 6656.2 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.8578 | 1216 | 6659.5 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_double_unit_max_tile_size | 1.9154 | 544 | 2982.3 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 12 | 169 | 100000 | tiled-smem(tm=11) | 0.7365 | 1408 | 7756.2 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.8981 | 1152 | 6360.2 | 6.00e-16 | pass |
| cuda | f64 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 1.1683 | 896 | 4889.5 | 6.00e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_auto | 0.1005 | 10240 | 1661.6 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.1005 | 10240 | 1661.4 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_double_unit_max_tile_size | 0.1627 | 6144 | 1026.5 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 1000 | tiled-smem(tm=10) | 0.0705 | 14336 | 2368.2 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0881 | 12288 | 1895.2 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.1299 | 8192 | 1286.3 | 4.43e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_auto | 0.2265 | 4608 | 7375.9 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.2266 | 4608 | 7373.0 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_double_unit_max_tile_size | 0.5421 | 1920 | 3081.6 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 10000 | tiled-smem(tm=10) | 0.2105 | 4864 | 7934.4 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.3428 | 3072 | 4873.0 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.3843 | 2816 | 4346.3 | 6.14e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_auto | 2.1403 | 480 | 7804.5 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 2.1388 | 480 | 7810.0 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 5.0386 | 208 | 3315.3 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: No tile size  |
| cuda | f64 | 16 | 289 | 100000 | tiled-smem(tm=10) | 1.6808 | 608 | 9938.4 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 2.5543 | 416 | 6539.6 | 6.78e-16 | pass |
| cuda | f64 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 3.2072 | 320 | 5208.4 | 6.78e-16 | pass |

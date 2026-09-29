# Raw results: Apple M3 Max (Mac15,9), 2026-09-29

Output of the spike binary (`src/main.rs`), unedited apart from concatenation. Median time
per GEMM over batches of back-to-back launches with one sync per batch. Every measured row
was checked against the plain-Rust f64 reference (`rel. error` = max-norm relative error).

Runs:

1. `cargo run -p nd-fmm-spike-cubecl-gemm --release -- --backends metal` (full sweep, Metal f32)
2. `... -- --backends cpu --precisions f64` (CPU runtime f64)
3. `... -- --backends cpu --precisions f32 --no-lib` (CPU runtime f32, hand-written kernels
   only: the library's f32 kernels for B = 1e4 did not finish compiling in over 5 minutes,
   twice; the B = 1e3 library rows below are from the interrupted full run)
4. Metal repeat runs of p = 8 and 16 at B = 1e5 (three runs), to show run-to-run variance.

## Sweep

| backend | precision | p | Nc | B | implementation | median ms | reps | GFLOP/s | rel. error | check |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_auto | 0.0379 | 26624 | 33.0 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0374 | 27648 | 33.4 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0333 | 30720 | 37.5 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | 0.0079 | 51200 | 157.8 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0092 | 51200 | 135.2 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0094 | 51200 | 133.0 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_auto | 0.0871 | 12288 | 143.5 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0868 | 12288 | 144.0 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0493 | 20480 | 253.6 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 10000 | tiled-smem(tm=2) | 0.0108 | 51200 | 1153.0 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0107 | 51200 | 1163.0 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0109 | 51200 | 1147.6 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_auto | 0.4956 | 2048 | 252.2 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.4931 | 2048 | 253.5 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_double_unit_max_tile_size | 0.4161 | 2432 | 300.4 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 100000 | tiled-smem(tm=2) | 0.0701 | 14336 | 1783.1 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.0420 | 24576 | 2977.6 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.0424 | 24576 | 2948.3 | 2.03e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_auto | 0.0093 | 51200 | 1412.3 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0888 | 12288 | 147.8 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0728 | 14336 | 180.3 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | 0.0091 | 51200 | 1441.2 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | 0.0110 | 51200 | 1195.3 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | 0.0232 | 44032 | 565.9 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0225 | 45056 | 583.7 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0225 | 45056 | 583.7 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_auto | 0.0358 | 28672 | 3668.5 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.2181 | 4608 | 601.6 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_double_unit_max_tile_size | 0.1336 | 7680 | 982.0 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_simple_cyclic_cmma | 0.0362 | 28672 | 3629.7 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_double_cyclic_cmma | 0.0433 | 23552 | 3032.4 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | tiled-smem(tm=6) | 0.0543 | 19456 | 2417.8 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0373 | 27648 | 3516.2 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0492 | 20480 | 2667.8 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_auto | 0.3468 | 3072 | 3784.2 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_simple_unit_max_tile_size | 1.3067 | 768 | 1004.2 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_double_unit_max_tile_size | 1.1862 | 896 | 1106.2 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_simple_cyclic_cmma | 0.3620 | 2816 | 3624.4 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_double_cyclic_cmma | 0.4588 | 2176 | 2860.3 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-smem(tm=6) | 0.5159 | 2048 | 2543.3 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.4563 | 2304 | 2875.5 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.4840 | 2176 | 2710.9 | 4.03e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_auto | 0.0194 | 51200 | 2937.2 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.1720 | 6144 | 332.0 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_double_unit_max_tile_size | 0.1304 | 8192 | 438.0 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_simple_cyclic_cmma | 0.0186 | 51200 | 3071.7 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_double_cyclic_cmma | 0.0396 | 25600 | 1441.5 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | tiled-smem(tm=11) | 0.0662 | 15360 | 863.3 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0420 | 24576 | 1360.1 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0453 | 22528 | 1261.8 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_auto | 0.1295 | 8192 | 4412.4 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.5206 | 1920 | 1097.1 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_double_unit_max_tile_size | 0.4538 | 2304 | 1258.7 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_simple_cyclic_cmma | 0.1385 | 7680 | 4125.7 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_double_cyclic_cmma | 0.3253 | 3328 | 1755.9 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | tiled-smem(tm=11) | 0.1826 | 5632 | 3127.6 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.1609 | 6656 | 3551.2 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.2636 | 3840 | 2167.1 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_auto | 1.4502 | 704 | 3938.9 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_simple_unit_max_tile_size | 5.2949 | 192 | 1078.8 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_double_unit_max_tile_size | 4.5403 | 224 | 1258.1 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_simple_cyclic_cmma | 1.5065 | 704 | 3791.8 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_double_cyclic_cmma | 3.3717 | 304 | 1694.2 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | tiled-smem(tm=11) | 1.8801 | 544 | 3038.2 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 1.9171 | 544 | 2979.6 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 3.5035 | 288 | 1630.4 | 6.95e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_auto | 0.0474 | 22528 | 3523.2 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.2822 | 3584 | 591.9 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_double_unit_max_tile_size | 0.2322 | 4352 | 719.3 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_simple_cyclic_cmma | 0.0451 | 22528 | 3704.8 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_double_cyclic_cmma | 0.1153 | 8704 | 1448.8 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | tiled-smem(tm=10) | 0.1033 | 9728 | 1617.1 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0789 | 13312 | 2116.5 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0863 | 12288 | 1936.5 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_auto | 0.4080 | 2560 | 4093.7 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_simple_unit_max_tile_size | 1.5443 | 672 | 1081.7 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_double_unit_max_tile_size | 1.2963 | 832 | 1288.6 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_simple_cyclic_cmma | 0.4203 | 2432 | 3974.2 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_double_cyclic_cmma | 0.8807 | 1152 | 1896.7 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | tiled-smem(tm=10) | 0.5597 | 1792 | 2984.6 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.5614 | 1792 | 2975.3 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 1.3960 | 768 | 1196.5 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_auto | 4.7824 | 224 | 3492.9 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 14.9195 | 68 | 1119.6 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 13.5122 | 76 | 1236.2 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | 4.8623 | 208 | 3435.5 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | 10.2981 | 96 | 1622.1 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 6.0803 | 176 | 2747.2 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 6.5717 | 160 | 2541.9 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 17.9391 | 56 | 931.2 | 7.88e-7 | pass |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_auto | 0.4424 | 2304 | 2.8 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.4392 | 2304 | 2.8 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.2010 | 5120 | 6.2 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 1000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0911 | 11264 | 13.7 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0391 | 25600 | 32.0 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_auto | 3.7688 | 272 | 3.3 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_simple_unit_max_tile_size | 3.7648 | 272 | 3.3 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_double_unit_max_tile_size | 1.7509 | 576 | 7.1 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 10000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.1375 | 7168 | 90.9 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.1319 | 7680 | 94.7 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_auto | 37.2921 | 28 | 3.4 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_simple_unit_max_tile_size | 37.2318 | 28 | 3.4 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_double_unit_max_tile_size | 18.6301 | 56 | 6.7 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 100000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.9785 | 1024 | 127.7 | 0.00e0 | pass |
| cpu | f64 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 1.1720 | 896 | 106.7 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_auto | 2.1483 | 480 | 6.1 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 2.1470 | 480 | 6.1 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 1.5707 | 640 | 8.4 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 1000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.1325 | 7680 | 99.0 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.1275 | 8192 | 102.9 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_auto | 20.5593 | 52 | 6.4 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_simple_unit_max_tile_size | 20.5862 | 52 | 6.4 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_double_unit_max_tile_size | 16.6485 | 64 | 7.9 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 10000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.7138 | 1408 | 183.8 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.7552 | 1408 | 173.8 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_auto | 206.4896 | 5 | 6.4 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_simple_unit_max_tile_size | 205.5645 | 5 | 6.4 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_double_unit_max_tile_size | 164.9667 | 7 | 8.0 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 100000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 15.4790 | 68 | 84.8 | 0.00e0 | pass |
| cpu | f64 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 8.1977 | 128 | 160.1 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_auto | 5.7336 | 176 | 10.0 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_simple_unit_max_tile_size | 5.7404 | 176 | 10.0 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_double_unit_max_tile_size | 6.3913 | 160 | 8.9 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 1000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.3312 | 3072 | 172.5 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.3406 | 3072 | 167.7 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_auto | 56.0880 | 18 | 10.2 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_simple_unit_max_tile_size | 56.0800 | 18 | 10.2 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_double_unit_max_tile_size | 65.7458 | 16 | 8.7 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 10000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 3.4977 | 288 | 163.3 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 3.5066 | 288 | 162.9 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_auto | 560.7797 | 5 | 10.2 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_simple_unit_max_tile_size | 562.3012 | 5 | 10.2 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_double_unit_max_tile_size | 630.6780 | 5 | 9.1 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 100000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 121.1522 | 9 | 47.1 | 0.00e0 | pass |
| cpu | f64 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 37.2270 | 28 | 153.4 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_auto | 16.3933 | 64 | 10.2 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_simple_unit_max_tile_size | 16.3086 | 64 | 10.2 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_double_unit_max_tile_size | 17.5167 | 60 | 9.5 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 1000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.8695 | 1216 | 192.1 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.8358 | 1216 | 199.8 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_auto | 161.2020 | 7 | 10.4 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_simple_unit_max_tile_size | 159.7281 | 7 | 10.5 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_double_unit_max_tile_size | 179.6840 | 6 | 9.3 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 10000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 8.6870 | 120 | 192.3 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 9.0166 | 112 | 185.3 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_auto | 1606.9893 | 5 | 10.4 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 1605.8876 | 5 | 10.4 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 1727.6406 | 5 | 9.7 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 100000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 453.4387 | 5 | 36.8 | 0.00e0 | pass |
| cpu | f64 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 133.9390 | 8 | 124.7 | 0.00e0 | pass |
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_auto | 0.4367 | 2304 | 2.9 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.4386 | 2304 | 2.9 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.1917 | 5632 | 6.5 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0893 | 11264 | 14.0 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0363 | 28672 | 34.5 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 10000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.1155 | 9216 | 108.2 | 1.93e-7 | pass |
| cpu | f32 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0933 | 11264 | 134.0 | 1.93e-7 | pass |
| cpu | f32 | 4 | 25 | 100000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.5093 | 2048 | 245.4 | 2.06e-7 | pass |
| cpu | f32 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.6779 | 1536 | 184.4 | 2.06e-7 | pass |
| cpu | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.1131 | 9216 | 116.0 | 3.29e-7 | pass |
| cpu | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.1007 | 10240 | 130.3 | 3.29e-7 | pass |
| cpu | f32 | 8 | 81 | 10000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.4531 | 2304 | 289.6 | 3.79e-7 | pass |
| cpu | f32 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.4433 | 2304 | 296.0 | 3.79e-7 | pass |
| cpu | f32 | 8 | 81 | 100000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 7.6490 | 136 | 171.6 | 3.97e-7 | pass |
| cpu | f32 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 4.4047 | 240 | 297.9 | 3.97e-7 | pass |
| cpu | f32 | 12 | 169 | 1000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.2264 | 4608 | 252.3 | 6.79e-7 | pass |
| cpu | f32 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.2376 | 4352 | 240.4 | 6.79e-7 | pass |
| cpu | f32 | 12 | 169 | 10000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 1.8452 | 544 | 309.6 | 7.27e-7 | pass |
| cpu | f32 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 1.7782 | 576 | 321.2 | 7.27e-7 | pass |
| cpu | f32 | 12 | 169 | 100000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 55.0497 | 18 | 103.8 | 6.14e-7 | pass |
| cpu | f32 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 19.3369 | 52 | 295.4 | 6.14e-7 | pass |
| cpu | f32 | 16 | 289 | 1000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.5561 | 1920 | 300.4 | 7.62e-7 | pass |
| cpu | f32 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.5501 | 1920 | 303.7 | 7.62e-7 | pass |
| cpu | f32 | 16 | 289 | 10000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 5.5159 | 192 | 302.8 | 8.08e-7 | pass |
| cpu | f32 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 4.9864 | 208 | 335.0 | 8.08e-7 | pass |
| cpu | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 252.6316 | 5 | 66.1 | 8.01e-7 | pass |
| cpu | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 62.1503 | 17 | 268.8 | 8.01e-7 | pass |

## Metal repeat runs (p = 8, 16; B = 1e5; three runs, in order)

| backend | precision | p | Nc | B | implementation | median ms | reps | GFLOP/s | rel. error | check |
|---|---|---|---|---|---|---|---|---|---|---|
| metal | f32 | 8 | 81 | 100000 | lib:matmul_auto | 0.3759 | 2560 | 3491.1 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-smem(tm=6) | 0.5236 | 1920 | 2506.3 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.4674 | 2176 | 2807.2 | 4.03e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_auto | 3.7495 | 288 | 4455.1 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 5.0578 | 208 | 3302.7 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 5.1161 | 192 | 3265.0 | 7.88e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_auto | 0.3840 | 2816 | 3417.1 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-smem(tm=6) | 0.5295 | 1920 | 2478.0 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.4704 | 2176 | 2789.8 | 4.03e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_auto | 3.7277 | 288 | 4481.1 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 5.0354 | 208 | 3317.4 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 5.1278 | 208 | 3257.6 | 7.88e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_auto | 0.4070 | 2560 | 3223.9 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-smem(tm=6) | 0.5354 | 1920 | 2451.0 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.4747 | 2176 | 2764.5 | 4.03e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_auto | 3.9572 | 272 | 4221.2 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 5.2323 | 192 | 3192.5 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 5.1939 | 208 | 3216.1 | 7.88e-7 | pass |

# Raw results: Apple M3 Max (Mac15,9), CubeCL 0.11.0-pre.4, 2026-10-03

Output of the spike binary (`src/main.rs`) built against `cubecl =0.11.0-pre.4`,
`cubek-matmul =0.3.0-pre.4` and `cubek-std =0.3.0-pre.4` (Phase 4 T2), unedited apart from
concatenation. Median time per GEMM over batches of back-to-back launches with one sync per
batch. Every measured row was checked against the plain-Rust f64 reference (`rel. error` =
max-norm relative error; tolerance 1e-12 for f64, 1e-5 for f32). The 0.10.0 results are in
[results-m3max.md](results-m3max.md); the comparison is in SPIKE_REPORT.md, section
"CubeCL 0.11.0-pre.4".

Machine: MacBook Pro 16" (Mac15,9), Apple M3 Max (12 performance + 4 efficiency cores,
40-core GPU), 64 GB, macOS 27.0.1 (26A434), on AC power; rustc 1.99.0 (2026-09-28),
release profile. Every run had `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `MKL_NUM_THREADS` and `RAYON_NUM_THREADS` set to 1, and ran
outside the macOS sandbox (Metal needs GPU access). Before each run no cargo, rustc or
spike process was running and the 1-minute load average was below 6 (it was 2.0–6.0;
the remaining load was Microsoft Defender, WindowServer, Orca, Safari and Discord, each
under 20% of one core). `pmset` recorded no thermal or performance warning.

Runs (the built binary was run directly; `cargo run -p nd-fmm-spike-cubecl-gemm --release --`
is equivalent):

1. `--backends metal` (full sweep, Metal f32; f64 is skipped, the device reports none)
2. `--backends cpu --precisions f64` (CPU runtime f64, library included)
3. `--backends cpu --precisions f32 --no-lib` (CPU runtime f32, hand-written kernels only,
   as for 0.10.0; the library's f32 kernels now compile, see the probe below)
4. Repeats of the flagged cells, three runs each, interleaved with the same commands on a
   0.10.0 build of the spike (the parent commit, built in a separate target directory) as
   a same-day control.
5. Probes: library f32 on the CPU runtime (`--quick`, 20-minute limit), the worker stack
   at its 64 MB default, launch overhead (p = 1, B = 4) and single-unit throughput
   (p = 1, B = 1e6); the last two also on the 0.10.0 build.

Device facts (stderr):

```
[metal] runtime wgpu<msl>: f64 supported = false, plane size 32..32, max shared memory 32768 B, SMs None, CPU cores None, tensor-core min dim None
[cpu] runtime cpu: f64 supported = true, plane size 1..1, max shared memory 65536 B, SMs None, CPU cores Some(16), tensor-core min dim None
```

## Sweep

| backend | precision | p | Nc | B | implementation | median ms | reps | GFLOP/s | rel. error | check |
|---|---|---|---|---|---|---|---|---|---|---|
| metal | f32 | 4 | 25 | 1000 | lib:matmul_auto | 0.0338 | 30720 | 36.9 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0339 | 30720 | 36.8 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0331 | 30720 | 37.8 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | 0.0052 | 51200 | 239.0 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0080 | 51200 | 157.1 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0081 | 51200 | 154.3 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_auto | 0.0818 | 12288 | 152.9 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.0824 | 12288 | 151.8 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_double_unit_max_tile_size | 0.0501 | 20480 | 249.6 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 10000 | tiled-smem(tm=2) | 0.0092 | 51200 | 1354.0 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0093 | 51200 | 1348.4 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0087 | 51200 | 1431.4 | 1.93e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_auto | 0.4765 | 2176 | 262.3 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_simple_unit_max_tile_size | 0.4766 | 2176 | 262.3 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_double_unit_max_tile_size | 0.4245 | 2432 | 294.5 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 100000 | tiled-smem(tm=2) | 0.0693 | 15360 | 1803.6 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.0406 | 24576 | 3078.5 | 2.03e-7 | pass |
| metal | f32 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.0415 | 24576 | 3012.3 | 2.03e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_auto | 0.0064 | 51200 | 2042.3 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0808 | 13312 | 162.5 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0701 | 14336 | 187.1 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | 0.0064 | 51200 | 2045.2 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | 0.0095 | 51200 | 1378.7 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | 0.0224 | 45056 | 584.7 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0210 | 48128 | 625.4 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0215 | 47104 | 610.2 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_auto | 0.0351 | 28672 | 3742.8 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.2066 | 4864 | 635.2 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_double_unit_max_tile_size | 0.1317 | 7680 | 996.5 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_simple_cyclic_cmma | 0.0346 | 29696 | 3792.3 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | lib:matmul_double_cyclic_cmma | 0.0443 | 23552 | 2960.9 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | tiled-smem(tm=6) | 0.0527 | 19456 | 2491.6 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.0366 | 27648 | 3585.1 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.0479 | 21504 | 2741.7 | 4.38e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_auto | 0.3146 | 3328 | 4171.5 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_simple_unit_max_tile_size | 1.2539 | 832 | 1046.5 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_double_unit_max_tile_size | 1.1770 | 896 | 1114.9 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_simple_cyclic_cmma | 0.3238 | 3328 | 4052.7 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | lib:matmul_double_cyclic_cmma | 0.4589 | 2304 | 2859.7 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-smem(tm=6) | 0.5171 | 2048 | 2537.8 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 0.4503 | 2304 | 2914.0 | 4.03e-7 | pass |
| metal | f32 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 0.4492 | 2304 | 2921.2 | 4.03e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_auto | 0.0167 | 51200 | 3419.9 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.1564 | 6656 | 365.3 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_double_unit_max_tile_size | 0.1225 | 8192 | 466.3 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_simple_cyclic_cmma | 0.0168 | 51200 | 3396.3 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | lib:matmul_double_cyclic_cmma | 0.0416 | 24576 | 1373.5 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | tiled-smem(tm=11) | 0.0650 | 16384 | 878.3 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0401 | 25600 | 1426.1 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0435 | 23552 | 1312.5 | 6.71e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_auto | 0.1178 | 8704 | 4850.2 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_simple_unit_max_tile_size | 0.4969 | 2048 | 1149.6 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_double_unit_max_tile_size | 0.4451 | 2304 | 1283.4 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_simple_cyclic_cmma | 0.1259 | 8192 | 4538.9 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | lib:matmul_double_cyclic_cmma | 0.3374 | 3072 | 1692.8 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | tiled-smem(tm=11) | 0.1816 | 5632 | 3146.0 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.1463 | 7168 | 3903.3 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.2504 | 4096 | 2281.0 | 7.27e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_auto | 1.3481 | 768 | 4237.1 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_simple_unit_max_tile_size | 4.8911 | 208 | 1167.9 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_double_unit_max_tile_size | 4.4073 | 240 | 1296.1 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_simple_cyclic_cmma | 1.4442 | 704 | 3955.4 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | lib:matmul_double_cyclic_cmma | 3.4132 | 304 | 1673.6 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | tiled-smem(tm=11) | 1.8536 | 544 | 3081.7 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 1.8908 | 544 | 3021.0 | 6.95e-7 | pass |
| metal | f32 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 3.4835 | 288 | 1639.8 | 6.95e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_auto | 0.0451 | 22528 | 3700.0 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.2593 | 4096 | 644.3 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_double_unit_max_tile_size | 0.2221 | 4608 | 752.0 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_simple_cyclic_cmma | 0.0429 | 24576 | 3894.3 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | lib:matmul_double_cyclic_cmma | 0.1225 | 8192 | 1363.3 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | tiled-smem(tm=10) | 0.1002 | 10240 | 1666.9 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0779 | 13312 | 2145.5 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0814 | 12288 | 2051.4 | 6.94e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_auto | 0.3633 | 2816 | 4598.5 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_simple_unit_max_tile_size | 1.4308 | 704 | 1167.5 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_double_unit_max_tile_size | 1.1240 | 896 | 1486.2 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_simple_cyclic_cmma | 0.3733 | 2816 | 4474.6 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | lib:matmul_double_cyclic_cmma | 0.9468 | 1088 | 1764.3 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | tiled-smem(tm=10) | 0.5032 | 2048 | 3319.4 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.4676 | 2176 | 3572.2 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 1.4677 | 704 | 1138.1 | 7.38e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_auto | 3.7102 | 272 | 4502.2 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 12.2283 | 88 | 1366.0 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 11.1526 | 96 | 1497.8 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | 3.9737 | 256 | 4203.7 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | 10.0725 | 104 | 1658.4 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 5.0665 | 208 | 3297.0 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 5.0469 | 208 | 3309.8 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 17.8334 | 56 | 936.7 | 7.88e-7 | pass |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_auto | 0.3096 | 3072 | 4.0 | 2.62e-16 | pass |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.3033 | 3328 | 4.1 | 2.62e-16 | pass |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0780 | 13312 | 16.0 | 2.62e-16 | pass |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 1000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.2309 | 3840 | 5.4 | 2.62e-16 | pass |
| cpu | f64 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.3153 | 3072 | 4.0 | 2.62e-16 | pass |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_auto | 2.8569 | 352 | 4.4 | 3.28e-16 | pass |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_simple_unit_max_tile_size | 2.8859 | 352 | 4.3 | 3.28e-16 | pass |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_double_unit_max_tile_size | 0.7049 | 1408 | 17.7 | 3.28e-16 | pass |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 10000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.4155 | 2432 | 30.1 | 3.28e-16 | pass |
| cpu | f64 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.5568 | 2048 | 22.5 | 3.28e-16 | pass |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_auto | 29.4079 | 34 | 4.3 | 3.83e-16 | pass |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_simple_unit_max_tile_size | 29.1961 | 36 | 4.3 | 3.83e-16 | pass |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_double_unit_max_tile_size | 7.4656 | 136 | 16.7 | 3.83e-16 | pass |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 4 | 25 | 100000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 2.7288 | 200 | 45.8 | 3.83e-16 | pass |
| cpu | f64 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 1.8666 | 480 | 67.0 | 3.83e-16 | pass |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_auto | 1.4520 | 704 | 9.0 | 3.33e-16 | pass |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 1.4574 | 704 | 9.0 | 3.33e-16 | pass |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.6234 | 1664 | 21.1 | 3.33e-16 | pass |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 1000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.3394 | 2560 | 38.7 | 3.33e-16 | pass |
| cpu | f64 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.4654 | 2048 | 28.2 | 3.33e-16 | pass |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_auto | 13.9058 | 76 | 9.4 | 3.89e-16 | pass |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_simple_unit_max_tile_size | 14.0772 | 72 | 9.3 | 3.89e-16 | pass |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_double_unit_max_tile_size | 5.9790 | 176 | 21.9 | 3.89e-16 | pass |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 10000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 1.0072 | 896 | 130.3 | 3.89e-16 | pass |
| cpu | f64 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 2.2499 | 448 | 58.3 | 3.89e-16 | pass |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_auto | 141.9212 | 7 | 9.2 | 4.25e-16 | pass |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_simple_unit_max_tile_size | 140.4311 | 8 | 9.3 | 4.25e-16 | pass |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_double_unit_max_tile_size | 62.2642 | 16 | 21.1 | 4.25e-16 | pass |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 8 | 81 | 100000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 20.5323 | 44 | 63.9 | 4.25e-16 | pass |
| cpu | f64 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 14.7014 | 68 | 89.3 | 4.25e-16 | pass |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_auto | 3.9784 | 256 | 14.4 | 4.82e-16 | pass |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_simple_unit_max_tile_size | 3.8310 | 272 | 14.9 | 4.82e-16 | pass |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_double_unit_max_tile_size | 2.4360 | 416 | 23.4 | 4.82e-16 | pass |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 1000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.5019 | 1920 | 113.8 | 4.82e-16 | pass |
| cpu | f64 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.6560 | 1408 | 87.1 | 4.82e-16 | pass |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_auto | 40.0978 | 20 | 14.2 | 5.15e-16 | pass |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_simple_unit_max_tile_size | 36.9781 | 23 | 15.4 | 5.15e-16 | pass |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_double_unit_max_tile_size | 23.9865 | 44 | 23.8 | 5.15e-16 | pass |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 10000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 4.0084 | 256 | 142.5 | 5.15e-16 | pass |
| cpu | f64 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 8.0129 | 136 | 71.3 | 5.15e-16 | pass |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_auto | 367.8950 | 5 | 15.5 | 6.00e-16 | pass |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_simple_unit_max_tile_size | 369.0886 | 5 | 15.5 | 6.00e-16 | pass |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_double_unit_max_tile_size | 242.5128 | 5 | 23.6 | 6.00e-16 | pass |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 12 | 169 | 100000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 69.9935 | 14 | 81.6 | 6.00e-16 | pass |
| cpu | f64 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 40.5469 | 26 | 140.9 | 6.00e-16 | pass |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_auto | 10.8261 | 96 | 15.4 | 4.43e-16 | pass |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_simple_unit_max_tile_size | 10.7477 | 96 | 15.5 | 4.43e-16 | pass |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_double_unit_max_tile_size | 6.8488 | 144 | 24.4 | 4.43e-16 | pass |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 1000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 1.1816 | 832 | 141.4 | 4.43e-16 | pass |
| cpu | f64 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 1.4009 | 704 | 119.2 | 4.43e-16 | pass |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_auto | 104.4923 | 10 | 16.0 | 6.14e-16 | pass |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_simple_unit_max_tile_size | 106.0210 | 10 | 15.8 | 6.14e-16 | pass |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_double_unit_max_tile_size | 67.2586 | 15 | 24.8 | 6.14e-16 | pass |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 10000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 14.0602 | 80 | 118.8 | 6.14e-16 | pass |
| cpu | f64 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 17.2927 | 56 | 96.6 | 6.14e-16 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_auto | 1069.1377 | 5 | 15.6 | 6.78e-16 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 1078.0498 | 5 | 15.5 | 6.78e-16 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 678.9003 | 5 | 24.6 | 6.78e-16 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 100000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 300.0880 | 5 | 55.7 | 6.78e-16 | pass |
| cpu | f64 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 113.4494 | 9 | 147.2 | 6.78e-16 | pass |
| cpu | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.2240 | 3840 | 5.6 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.2125 | 3840 | 5.9 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 10000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.3538 | 3072 | 35.3 | 1.93e-7 | pass |
| cpu | f32 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.3113 | 2816 | 40.2 | 1.93e-7 | pass |
| cpu | f32 | 4 | 25 | 100000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 4 | 25 | 100000 | tiled-reg-cols(4x8, vec 4) | 1.0471 | 768 | 119.4 | 2.03e-7 | pass |
| cpu | f32 | 4 | 25 | 100000 | tiled-reg-rows(4x8, vec 4) | 1.3588 | 768 | 92.0 | 2.03e-7 | pass |
| cpu | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.3516 | 2560 | 37.3 | 3.27e-7 | pass |
| cpu | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.3274 | 3072 | 40.1 | 3.27e-7 | pass |
| cpu | f32 | 8 | 81 | 10000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.8299 | 1280 | 158.1 | 4.38e-7 | pass |
| cpu | f32 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 1.2312 | 832 | 106.6 | 4.38e-7 | pass |
| cpu | f32 | 8 | 81 | 100000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 8 | 81 | 100000 | tiled-reg-cols(4x8, vec 4) | 12.5549 | 76 | 104.5 | 4.03e-7 | pass |
| cpu | f32 | 8 | 81 | 100000 | tiled-reg-rows(4x8, vec 4) | 9.0853 | 112 | 144.4 | 4.03e-7 | pass |
| cpu | f32 | 12 | 169 | 1000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.4494 | 1920 | 127.1 | 6.71e-7 | pass |
| cpu | f32 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.8723 | 1280 | 65.5 | 6.71e-7 | pass |
| cpu | f32 | 12 | 169 | 10000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 2.9459 | 352 | 193.9 | 7.27e-7 | pass |
| cpu | f32 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 3.5396 | 288 | 161.4 | 7.27e-7 | pass |
| cpu | f32 | 12 | 169 | 100000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 12 | 169 | 100000 | tiled-reg-cols(4x8, vec 4) | 42.6020 | 24 | 134.1 | 6.95e-7 | pass |
| cpu | f32 | 12 | 169 | 100000 | tiled-reg-rows(4x8, vec 4) | 25.1335 | 44 | 227.3 | 6.95e-7 | pass |
| cpu | f32 | 16 | 289 | 1000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.7761 | 1216 | 215.2 | 6.94e-7 | pass |
| cpu | f32 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 1.1689 | 768 | 142.9 | 6.94e-7 | pass |
| cpu | f32 | 16 | 289 | 10000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 7.9188 | 120 | 210.9 | 7.38e-7 | pass |
| cpu | f32 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 9.5655 | 104 | 174.6 | 7.38e-7 | pass |
| cpu | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 150.8895 | 7 | 110.7 | 7.88e-7 | pass |
| cpu | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 51.8465 | 20 | 322.2 | 7.88e-7 | pass |

## Metal repeat runs, CPU repeats and same-day 0.10.0 controls

Not part of the sweep (`analyse.py` stops at this heading). GFLOP/s per run, in order.
"0.10.0 file" is the sweep in results-m3max.md (2026-09-29); "0.10.0 same day" is the
0.10.0 build run between the 0.11 repeats today. Commands: Metal `--precisions f32 --p 4,8
--b 1000` and `--p 16 --b 100000`; CPU `--precisions f64,f32 --no-lib` (every p and B) and
`--precisions f64 --p 8 --b 10000` (library included). Every repeated result passed its check.

| backend | precision | p | B | implementation | 0.10.0 file | 0.10.0 same day (runs) | 0.11 sweep | 0.11 repeats | median ratio 0.11/0.10 (same day) |
|---|---|---|---|---|---|---|---|---|---|
| metal | f32 | 4 | 1000 | lib:matmul_auto | 33.0 | 32.8 | 36.9 | 37.4, 37.1, 36.9 | 1.13 |
| metal | f32 | 4 | 1000 | lib:matmul_double_unit | 37.5 | 37.6 | 37.8 | 37.8, 38.1, 38.1 | 1.01 |
| metal | f32 | 4 | 1000 | lib:matmul_simple_unit | 33.4 | 33.6 | 36.8 | 37.0, 37.7, 36.9 | 1.10 |
| metal | f32 | 4 | 1000 | tiled-reg-cols | 135.2 | 143.3 | 157.1 | 156.1, 160.1, 157.8 | 1.10 |
| metal | f32 | 4 | 1000 | tiled-reg-rows | 133.0 | 142.3 | 154.3 | 153.6, 157.3, 162.5 | 1.11 |
| metal | f32 | 4 | 1000 | tiled-smem(tm=2) | 157.8 | 169.7 | 239.0 | 238.6, 238.2, 235.0 | 1.40 |
| metal | f32 | 8 | 1000 | lib:matmul_auto | 1412.3 | 1446.4 | 2042.3 | 2043.6, 2043.9, 2077.3 | 1.41 |
| metal | f32 | 8 | 1000 | lib:matmul_double_cyclic_cmma | 1195.3 | 1236.6 | 1378.7 | 1379.4, 1347.4, 1411.0 | 1.12 |
| metal | f32 | 8 | 1000 | lib:matmul_double_unit | 180.3 | 177.3 | 187.1 | 186.8, 186.2, 187.5 | 1.05 |
| metal | f32 | 8 | 1000 | lib:matmul_simple_cyclic_cmma | 1441.2 | 1444.2 | 2045.2 | 2045.2, 2064.8, 2031.3 | 1.42 |
| metal | f32 | 8 | 1000 | lib:matmul_simple_unit | 147.8 | 145.2 | 162.5 | 161.1, 161.9, 163.2 | 1.12 |
| metal | f32 | 8 | 1000 | tiled-reg-cols | 583.7 | 589.8 | 625.4 | 621.8, 632.0, 628.9 | 1.07 |
| metal | f32 | 8 | 1000 | tiled-reg-rows | 583.7 | 566.5 | 610.2 | 604.9, 622.0, 615.8 | 1.09 |
| metal | f32 | 8 | 1000 | tiled-smem(tm=6) | 565.9 | 559.0 | 584.7 | 573.9, 590.4, 607.2 | 1.06 |
| metal | f32 | 16 | 100000 | lib:matmul_auto | 3492.9 | 5415.0 | 4502.2 | 5376.2, 5215.9, 5220.0 | 0.96 |
| metal | f32 | 16 | 100000 | lib:matmul_double_cyclic_cmma | 1622.1 | 1762.7 | 1658.4 | 1659.0, 1662.9, 1662.8 | 0.94 |
| metal | f32 | 16 | 100000 | lib:matmul_double_unit | 1236.2 | 1608.4 | 1497.8 | 1610.6, 1609.9, 1617.2 | 1.00 |
| metal | f32 | 16 | 100000 | lib:matmul_simple_cyclic_cmma | 3435.5 | 5302.5 | 4203.7 | 5460.5, 5441.8, 5457.8 | 1.03 |
| metal | f32 | 16 | 100000 | lib:matmul_simple_unit | 1119.6 | 1337.3 | 1366.0 | 1397.9, 1396.0, 1402.4 | 1.05 |
| metal | f32 | 16 | 100000 | tiled-reg-cols | 2541.9 | 3715.2 | 3309.8 | 3749.6, 3741.7, 3750.2 | 1.01 |
| metal | f32 | 16 | 100000 | tiled-reg-rows | 931.2 | 1020.9 | 936.7 | 1049.8, 1047.5, 1050.4 | 1.03 |
| metal | f32 | 16 | 100000 | tiled-smem(tm=10) | 2747.2 | 3326.1 | 3297.0 | 3336.4, 3345.0, 3341.7 | 1.00 |
| cpu | f32 | 4 | 1000 | tiled-reg-cols | 14.0 | 14.0, 14.2, 13.9 | 5.6 | 4.4, 4.5, 3.4 | 0.31 |
| cpu | f32 | 4 | 1000 | tiled-reg-rows | 34.5 | 36.0, 34.9, 35.0 | 5.9 | 4.6, 2.3, 5.4 | 0.13 |
| cpu | f32 | 4 | 10000 | tiled-reg-cols | 108.2 | 111.2, 108.8, 109.3 | 35.3 | 20.9, 35.4, 40.1 | 0.32 |
| cpu | f32 | 4 | 10000 | tiled-reg-rows | 134.0 | 134.7, 133.9, 133.6 | 40.2 | 28.4, 30.8, 29.1 | 0.22 |
| cpu | f32 | 4 | 100000 | tiled-reg-cols | 245.4 | 225.8, 227.8, 224.5 | 119.4 | 60.4, 73.9, 88.7 | 0.33 |
| cpu | f32 | 4 | 100000 | tiled-reg-rows | 184.4 | 160.3, 162.6, 165.4 | 92.0 | 92.5, 95.5, 91.3 | 0.57 |
| cpu | f32 | 8 | 1000 | tiled-reg-cols | 116.0 | 115.4, 116.9, 114.2 | 37.3 | 31.1, 40.6, 61.4 | 0.35 |
| cpu | f32 | 8 | 1000 | tiled-reg-rows | 130.3 | 131.5, 131.7, 130.3 | 40.1 | 23.9, 38.3, 31.5 | 0.24 |
| cpu | f32 | 8 | 10000 | tiled-reg-cols | 289.6 | 265.9, 266.4, 266.2 | 158.1 | 151.5, 167.6, 113.3 | 0.57 |
| cpu | f32 | 8 | 10000 | tiled-reg-rows | 296.0 | 263.2, 271.2, 270.3 | 106.6 | 117.2, 85.4, 112.9 | 0.42 |
| cpu | f32 | 8 | 100000 | tiled-reg-cols | 171.6 | 154.4, 149.7, 154.4 | 104.5 | 102.4, 86.6, 86.7 | 0.56 |
| cpu | f32 | 8 | 100000 | tiled-reg-rows | 297.9 | 271.7, 272.0, 269.2 | 144.4 | 147.9, 139.0, 132.9 | 0.51 |
| cpu | f32 | 12 | 1000 | tiled-reg-cols | 252.3 | 244.6, 248.8, 247.7 | 127.1 | 152.6, 100.0, 85.1 | 0.40 |
| cpu | f32 | 12 | 1000 | tiled-reg-rows | 240.4 | 233.6, 233.3, 234.6 | 65.5 | 91.8, 143.9, 100.1 | 0.43 |
| cpu | f32 | 12 | 10000 | tiled-reg-cols | 309.6 | 290.1, 289.3, 288.2 | 193.9 | 175.6, 210.1, 192.4 | 0.67 |
| cpu | f32 | 12 | 10000 | tiled-reg-rows | 321.2 | 304.4, 301.0, 306.1 | 161.4 | 179.9, 172.0, 136.2 | 0.57 |
| cpu | f32 | 12 | 100000 | tiled-reg-cols | 103.8 | 101.8, 110.4, 108.3 | 134.1 | 128.9, 116.5, 136.5 | 1.19 |
| cpu | f32 | 12 | 100000 | tiled-reg-rows | 295.4 | 277.4, 270.5, 280.1 | 227.3 | 243.3, 227.5, 226.4 | 0.82 |
| cpu | f32 | 16 | 1000 | tiled-reg-cols | 300.4 | 280.8, 284.6, 286.3 | 215.2 | 155.7, 177.5, 168.3 | 0.59 |
| cpu | f32 | 16 | 1000 | tiled-reg-rows | 303.7 | 291.2, 289.6, 290.3 | 142.9 | 145.2, 194.1, 192.7 | 0.66 |
| cpu | f32 | 16 | 10000 | tiled-reg-cols | 302.8 | 282.8, 286.0, 286.3 | 210.9 | 166.6, 189.1, 176.1 | 0.62 |
| cpu | f32 | 16 | 10000 | tiled-reg-rows | 335.0 | 300.1, 311.4, 312.6 | 174.6 | 155.3, 167.8, 160.2 | 0.51 |
| cpu | f32 | 16 | 100000 | tiled-reg-cols | 66.1 | 60.9, 65.4, 65.3 | 110.7 | 111.8, 102.0, 96.1 | 1.56 |
| cpu | f32 | 16 | 100000 | tiled-reg-rows | 268.8 | 269.5, 271.4, 272.7 | 322.2 | 292.5, 306.4, 319.7 | 1.13 |
| cpu | f64 | 4 | 1000 | tiled-reg-cols | 13.7 | 13.3, 13.4, 13.3 | 5.4 | 5.1, 4.9, 4.8 | 0.37 |
| cpu | f64 | 4 | 1000 | tiled-reg-rows | 32.0 | 30.5, 30.7, 30.2 | 4.0 | 5.6, 3.5, 2.6 | 0.11 |
| cpu | f64 | 4 | 10000 | tiled-reg-cols | 90.9 | 89.7, 89.4, 90.6 | 30.1 | 34.4, 29.2, 31.9 | 0.36 |
| cpu | f64 | 4 | 10000 | tiled-reg-rows | 94.7 | 94.2, 94.7, 94.6 | 22.5 | 33.5, 28.3, 25.9 | 0.30 |
| cpu | f64 | 4 | 100000 | tiled-reg-cols | 127.7 | 131.6, 131.8, 131.6 | 45.8 | 32.6, 36.5, 37.3 | 0.28 |
| cpu | f64 | 4 | 100000 | tiled-reg-rows | 106.7 | 107.2, 108.4, 106.7 | 67.0 | 43.0, 47.3, 57.3 | 0.44 |
| cpu | f64 | 8 | 1000 | tiled-reg-cols | 99.0 | 97.7, 98.2, 98.9 | 38.7 | 42.1, 19.3, 17.0 | 0.20 |
| cpu | f64 | 8 | 1000 | tiled-reg-rows | 102.9 | 106.7, 105.0, 102.5 | 28.2 | 32.1, 24.0, 35.9 | 0.31 |
| cpu | f64 | 8 | 10000 | lib:matmul_auto | 6.4 | 6.5, 6.4, 6.5 | 9.4 | 9.3, 8.0, 9.5 | 1.43 |
| cpu | f64 | 8 | 10000 | lib:matmul_double_unit | 7.9 | 9.1, 9.2, 9.1 | 21.9 | 21.9, 21.9, 22.0 | 2.41 |
| cpu | f64 | 8 | 10000 | lib:matmul_simple_unit | 6.4 | 6.5, 6.1, 6.5 | 9.3 | 9.4, 9.3, 9.4 | 1.45 |
| cpu | f64 | 8 | 10000 | tiled-reg-cols | 183.8 | 186.9, 183.9, 185.0, 182.5, 185.4, 184.8 | 130.3 | 75.8, 108.2, 81.0, 99.9, 96.7, 87.3 | 0.50 |
| cpu | f64 | 8 | 10000 | tiled-reg-rows | 173.8 | 183.6, 185.0, 185.4, 185.3, 185.9, 185.5 | 58.3 | 50.8, 47.6, 53.3, 54.8, 43.9, 53.4 | 0.28 |
| cpu | f64 | 8 | 100000 | tiled-reg-cols | 84.8 | 91.8, 91.0, 90.8 | 63.9 | 59.9, 58.4, 60.4 | 0.66 |
| cpu | f64 | 8 | 100000 | tiled-reg-rows | 160.1 | 179.8, 180.9, 173.9 | 89.3 | 72.2, 79.7, 81.5 | 0.44 |
| cpu | f64 | 12 | 1000 | tiled-reg-cols | 172.5 | 171.0, 172.4, 171.3 | 113.8 | 86.9, 48.9, 111.4 | 0.51 |
| cpu | f64 | 12 | 1000 | tiled-reg-rows | 167.7 | 168.1, 169.0, 166.7 | 87.1 | 86.8, 69.6, 85.0 | 0.51 |
| cpu | f64 | 12 | 10000 | tiled-reg-cols | 163.3 | 159.2, 160.8, 155.1 | 142.5 | 120.4, 148.8, 135.7 | 0.85 |
| cpu | f64 | 12 | 10000 | tiled-reg-rows | 162.9 | 156.3, 160.8, 163.1 | 71.3 | 71.9, 57.7, 71.9 | 0.45 |
| cpu | f64 | 12 | 100000 | tiled-reg-cols | 47.1 | 52.7, 51.0, 52.2 | 81.6 | 82.1, 83.8, 82.3 | 1.58 |
| cpu | f64 | 12 | 100000 | tiled-reg-rows | 153.4 | 170.7, 170.9, 167.1 | 140.9 | 142.5, 147.1, 141.9 | 0.83 |
| cpu | f64 | 16 | 1000 | tiled-reg-cols | 192.1 | 191.4, 191.3, 189.1 | 141.4 | 148.8, 98.0, 140.8 | 0.74 |
| cpu | f64 | 16 | 1000 | tiled-reg-rows | 199.8 | 201.3, 201.1, 199.0 | 119.2 | 108.0, 126.2, 110.7 | 0.55 |
| cpu | f64 | 16 | 10000 | tiled-reg-cols | 192.3 | 176.0, 180.2, 175.8 | 118.8 | 131.9, 140.5, 112.7 | 0.75 |
| cpu | f64 | 16 | 10000 | tiled-reg-rows | 185.3 | 175.6, 185.5, 180.0 | 96.6 | 106.0, 85.3, 112.6 | 0.59 |
| cpu | f64 | 16 | 100000 | tiled-reg-cols | 36.8 | 37.2, 37.3, 36.8 | 55.7 | 56.1, 56.8, 56.8 | 1.53 |
| cpu | f64 | 16 | 100000 | tiled-reg-rows | 124.7 | 104.9, 131.3, 135.0 | 147.2 | 179.1, 141.9, 144.2 | 1.10 |


### Launch-overhead probe (CPU runtime, f64, p = 1, B = 4, `--no-lib`; three runs each)

`tiled-reg-rows` launches one cube of one unit, `tiled-reg-cols` one cube of 32 units (31
of which exit at once). Median ms per launch:

| build | tiled-reg-rows (1 unit) | tiled-reg-cols (32 units) |
|---|---|---|
| 0.10.0 | 0.0061, 0.0063, 0.0062 | 0.0845, 0.0847, 0.0863 |
| 0.11.0-pre.4 | 0.0020, 0.0018, 0.0022 | 0.2482, 0.2491, 0.2014 |

### Single-unit probe (CPU runtime, p = 1, B = 1e6, `--no-lib`; three runs each)

With p = 1, `tiled-reg-rows` launches cubes of one unit, so one worker thread runs the whole
GEMM and the launch cost is negligible: this compares the generated code alone. Median ms
per GEMM (GFLOP/s):

| build | f64 | f32 |
|---|---|---|
| 0.10.0 | 2.1239 ms (15.1), 1.9914 ms (16.1), 1.9949 ms (16.0) | 1.4043 ms (22.8), 1.3498 ms (23.7), 1.3420 ms (23.8) |
| 0.11.0-pre.4 | 0.8192 ms (39.1), 0.8079 ms (39.6), 0.8115 ms (39.4) | 0.4613 ms (69.4), 0.4520 ms (70.8), 0.4493 ms (71.2) |

### Metal repeat runs, raw (0.11.0-pre.4)

| backend | precision | p | Nc | B | implementation | median ms | reps | GFLOP/s | rel. error | check |
|---|---|---|---|---|---|---|---|---|---|---|
| metal | f32 | 4 | 25 | 1000 | lib:matmul_auto | 0.0335 | 30720 | 37.4 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0338 | 29696 | 37.0 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0331 | 30720 | 37.8 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | 0.0052 | 51200 | 238.6 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0080 | 51200 | 156.1 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0081 | 51200 | 153.6 | 2.22e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_auto | 0.0064 | 51200 | 2043.6 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0815 | 13312 | 161.1 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0702 | 14336 | 186.8 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | 0.0064 | 51200 | 2045.2 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | 0.0095 | 51200 | 1379.4 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | 0.0229 | 45056 | 573.9 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0211 | 48128 | 621.8 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0217 | 47104 | 604.9 | 3.27e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_auto | 3.1071 | 320 | 5376.2 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 11.9498 | 88 | 1397.9 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 10.3713 | 104 | 1610.6 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | 3.0591 | 352 | 5460.5 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | 10.0691 | 104 | 1659.0 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 5.0066 | 208 | 3336.4 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 4.4550 | 240 | 3749.6 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 15.9123 | 64 | 1049.8 | 7.88e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_auto | 0.0337 | 30720 | 37.1 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0332 | 30720 | 37.7 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0328 | 30720 | 38.1 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | 0.0052 | 51200 | 238.2 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0078 | 51200 | 160.1 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0079 | 51200 | 157.3 | 2.22e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_auto | 0.0064 | 51200 | 2043.9 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0810 | 13312 | 161.9 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0705 | 14336 | 186.2 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | 0.0064 | 51200 | 2064.8 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | 0.0097 | 51200 | 1347.4 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | 0.0222 | 46080 | 590.4 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0208 | 49152 | 632.0 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0211 | 48128 | 622.0 | 3.27e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_auto | 3.2025 | 320 | 5215.9 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 11.9657 | 88 | 1396.0 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 10.3761 | 104 | 1609.9 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | 3.0696 | 352 | 5441.8 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | 10.0454 | 104 | 1662.9 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 4.9938 | 208 | 3345.0 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 4.4644 | 224 | 3741.7 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 15.9474 | 64 | 1047.5 | 7.88e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_auto | 0.0339 | 29696 | 36.9 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0339 | 30720 | 36.9 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0328 | 30720 | 38.1 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| metal | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | 0.0053 | 51200 | 235.0 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0079 | 51200 | 157.8 | 2.22e-7 | pass |
| metal | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0077 | 51200 | 162.5 | 2.22e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_auto | 0.0063 | 51200 | 2077.3 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.0804 | 13312 | 163.2 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0700 | 14336 | 187.5 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | 0.0065 | 51200 | 2031.3 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | 0.0093 | 51200 | 1411.0 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | 0.0216 | 46080 | 607.2 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.0209 | 49152 | 628.9 | 3.27e-7 | pass |
| metal | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.0213 | 48128 | 615.8 | 3.27e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_auto | 3.2000 | 320 | 5220.0 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 11.9110 | 88 | 1402.4 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 10.3291 | 104 | 1617.2 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | 3.0606 | 352 | 5457.8 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | 10.0461 | 104 | 1662.8 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | 4.9988 | 208 | 3341.7 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 4.4542 | 240 | 3750.2 | 7.88e-7 | pass |
| metal | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 15.9032 | 64 | 1050.4 | 7.88e-7 | pass |

## First launch of each implementation (stderr)

The spike times the warm-up launch, which includes the kernel's compilation the first time
a kernel is needed in the process (one sync each). A kernel is compiled once per comptime
configuration (p, layout, precision, line size) and reused for later B, so the B = 1e3 row
holds the compile time and later rows only the launch. The library's `Auto` shares kernels
with the explicit strategies. On Metal the driver's shader cache may persist between
processes, so the Metal figures are a lower bound on a cold compile.


Sweeps, B = 1000 (seconds; the first case of a kernel includes its compilation):

| backend | precision | implementation | p = 4 | p = 8 | p = 12 | p = 16 |
|---|---|---|---|---|---|---|
| cpu | f32 | tiled-reg-cols | 0.015 | 0.012 | 0.012 | 0.012 |
| cpu | f32 | tiled-reg-rows | 0.012 | 0.011 | 0.011 | 0.012 |
| cpu | f64 | lib:matmul_auto | 0.558 | 0.002 | 0.004 | 0.012 |
| cpu | f64 | lib:matmul_double_unit | 3.430 | 0.001 | 0.003 | 0.008 |
| cpu | f64 | lib:matmul_simple_unit | 0.000 | 0.002 | 0.004 | 0.011 |
| cpu | f64 | tiled-reg-cols | 0.014 | 0.013 | 0.013 | 0.014 |
| cpu | f64 | tiled-reg-rows | 0.013 | 0.013 | 0.013 | 0.014 |
| metal | f32 | lib:matmul_auto | 0.158 | 0.057 | 0.001 | 0.002 |
| metal | f32 | lib:matmul_double_cyclic_cmma | – | 0.155 | 0.320 | 0.001 |
| metal | f32 | lib:matmul_double_unit | 0.377 | 0.001 | 0.001 | 0.001 |
| metal | f32 | lib:matmul_simple_cyclic_cmma | – | 0.000 | 0.000 | 0.000 |
| metal | f32 | lib:matmul_simple_unit | 0.000 | 0.000 | 0.001 | 0.001 |
| metal | f32 | tiled-reg-cols | 0.026 | 0.027 | 0.028 | 0.027 |
| metal | f32 | tiled-reg-rows | 0.026 | 0.027 | 0.027 | 0.026 |
| metal | f32 | tiled-smem | 0.065 | 0.145 | 0.280 | 0.285 |

Sweeps, B = 10000 (seconds; the first case of a kernel includes its compilation):

| backend | precision | implementation | p = 4 | p = 8 | p = 12 | p = 16 |
|---|---|---|---|---|---|---|
| cpu | f32 | tiled-reg-cols | 0.004 | 0.001 | 0.002 | 0.004 |
| cpu | f32 | tiled-reg-rows | 0.000 | 0.001 | 0.001 | 0.004 |
| cpu | f64 | lib:matmul_auto | 0.006 | 0.028 | 0.039 | 0.113 |
| cpu | f64 | lib:matmul_double_unit | 0.001 | 0.007 | 0.026 | 0.070 |
| cpu | f64 | lib:matmul_simple_unit | 0.003 | 0.015 | 0.106 | 0.109 |
| cpu | f64 | tiled-reg-cols | 0.000 | 0.001 | 0.002 | 0.006 |
| cpu | f64 | tiled-reg-rows | 0.000 | 0.001 | 0.002 | 0.008 |
| metal | f32 | lib:matmul_auto | 0.001 | 0.057 | 0.002 | 0.005 |
| metal | f32 | lib:matmul_double_cyclic_cmma | – | 0.153 | 0.331 | 0.002 |
| metal | f32 | lib:matmul_double_unit | 0.001 | 0.001 | 0.001 | 0.002 |
| metal | f32 | lib:matmul_simple_cyclic_cmma | – | 0.000 | 0.000 | 0.001 |
| metal | f32 | lib:matmul_simple_unit | 0.000 | 0.001 | 0.001 | 0.003 |
| metal | f32 | tiled-reg-cols | 0.001 | 0.001 | 0.001 | 0.001 |
| metal | f32 | tiled-reg-rows | 0.000 | 0.000 | 0.001 | 0.002 |
| metal | f32 | tiled-smem | 0.001 | 0.001 | 0.001 | 0.001 |

Sweeps, B = 100000 (seconds; the first case of a kernel includes its compilation):

| backend | precision | implementation | p = 4 | p = 8 | p = 12 | p = 16 |
|---|---|---|---|---|---|---|
| cpu | f32 | tiled-reg-cols | 0.015 | 0.056 | 0.034 | 0.147 |
| cpu | f32 | tiled-reg-rows | 0.001 | 0.003 | 0.014 | 0.047 |
| cpu | f64 | lib:matmul_auto | 0.576 | 0.145 | 0.596 | 1.153 |
| cpu | f64 | lib:matmul_double_unit | 3.384 | 0.063 | 0.244 | 0.738 |
| cpu | f64 | lib:matmul_simple_unit | 0.029 | 0.146 | 0.362 | 1.045 |
| cpu | f64 | tiled-reg-cols | 0.001 | 0.010 | 0.075 | 0.283 |
| cpu | f64 | tiled-reg-rows | 0.001 | 0.005 | 0.032 | 0.105 |
| metal | f32 | lib:matmul_auto | 0.008 | 0.008 | 0.043 | 0.044 |
| metal | f32 | lib:matmul_double_cyclic_cmma | – | 0.001 | 0.005 | 0.011 |
| metal | f32 | lib:matmul_double_unit | 0.002 | 0.002 | 0.005 | 0.011 |
| metal | f32 | lib:matmul_simple_cyclic_cmma | – | 0.001 | 0.002 | 0.004 |
| metal | f32 | lib:matmul_simple_unit | 0.001 | 0.002 | 0.006 | 0.014 |
| metal | f32 | tiled-reg-cols | 0.001 | 0.002 | 0.003 | 0.005 |
| metal | f32 | tiled-reg-rows | 0.000 | 0.002 | 0.004 | 0.016 |
| metal | f32 | tiled-smem | 0.001 | 0.002 | 0.002 | 0.006 |


## Library f32 on the CPU runtime (compile probe)

`--backends cpu --precisions f32 --p 4,8,12,16 --b 1000,10000 --quick`, under a 20-minute
limit. The whole run finished in 50 s (0.10.0: the B = 1e4 library kernels did not finish
compiling in over 5 minutes, twice). Short timing (`--quick`), so the GFLOP/s are rough.

| backend | precision | p | Nc | B | implementation | median ms | reps | GFLOP/s | rel. error | check |
|---|---|---|---|---|---|---|---|---|---|---|
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_auto | 0.2680 | 768 | 4.7 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_simple_unit_max_tile_size | 0.2754 | 768 | 4.5 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_double_unit_max_tile_size | 0.0634 | 3328 | 19.7 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 4 | 25 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 4 | 25 | 1000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 4 | 25 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.2323 | 896 | 5.4 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.1079 | 1280 | 11.6 | 2.22e-7 | pass |
| cpu | f32 | 4 | 25 | 10000 | lib:matmul_auto | 1.8703 | 112 | 6.7 | 1.93e-7 | pass |
| cpu | f32 | 4 | 25 | 10000 | lib:matmul_simple_unit_max_tile_size | 1.8620 | 112 | 6.7 | 1.93e-7 | pass |
| cpu | f32 | 4 | 25 | 10000 | lib:matmul_double_unit_max_tile_size | 1.0813 | 192 | 11.6 | 1.93e-7 | pass |
| cpu | f32 | 4 | 25 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 4 | 25 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 4 | 25 | 10000 | tiled-smem(tm=2) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 4 | 25 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.3552 | 640 | 35.2 | 1.93e-7 | pass |
| cpu | f32 | 4 | 25 | 10000 | tiled-reg-rows(4x8, vec 4) | 0.3759 | 640 | 33.3 | 1.93e-7 | pass |
| cpu | f32 | 8 | 81 | 1000 | lib:matmul_auto | 1.2680 | 160 | 10.3 | 3.27e-7 | pass |
| cpu | f32 | 8 | 81 | 1000 | lib:matmul_simple_unit_max_tile_size | 1.2299 | 176 | 10.7 | 3.27e-7 | pass |
| cpu | f32 | 8 | 81 | 1000 | lib:matmul_double_unit_max_tile_size | 0.4391 | 512 | 29.9 | 3.27e-7 | pass |
| cpu | f32 | 8 | 81 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 8 | 81 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 8 | 81 | 1000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 8 | 81 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.4472 | 512 | 29.3 | 3.27e-7 | pass |
| cpu | f32 | 8 | 81 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.2376 | 896 | 55.2 | 3.27e-7 | pass |
| cpu | f32 | 8 | 81 | 10000 | lib:matmul_auto | 4.6963 | 40 | 27.9 | 4.38e-7 | pass |
| cpu | f32 | 8 | 81 | 10000 | lib:matmul_simple_unit_max_tile_size | 4.5666 | 44 | 28.7 | 4.38e-7 | pass |
| cpu | f32 | 8 | 81 | 10000 | lib:matmul_double_unit_max_tile_size | 4.2790 | 48 | 30.7 | 4.38e-7 | pass |
| cpu | f32 | 8 | 81 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 8 | 81 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 8 | 81 | 10000 | tiled-smem(tm=6) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 8 | 81 | 10000 | tiled-reg-cols(4x8, vec 4) | 0.6553 | 256 | 200.2 | 4.38e-7 | pass |
| cpu | f32 | 8 | 81 | 10000 | tiled-reg-rows(4x8, vec 4) | 1.3765 | 144 | 95.3 | 4.38e-7 | pass |
| cpu | f32 | 12 | 169 | 1000 | lib:matmul_auto | 3.0771 | 72 | 18.6 | 6.71e-7 | pass |
| cpu | f32 | 12 | 169 | 1000 | lib:matmul_simple_unit_max_tile_size | 3.2053 | 64 | 17.8 | 6.71e-7 | pass |
| cpu | f32 | 12 | 169 | 1000 | lib:matmul_double_unit_max_tile_size | 1.6872 | 128 | 33.9 | 6.71e-7 | pass |
| cpu | f32 | 12 | 169 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 12 | 169 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 12 | 169 | 1000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 12 | 169 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.3417 | 576 | 167.2 | 6.71e-7 | pass |
| cpu | f32 | 12 | 169 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.4519 | 512 | 126.4 | 6.71e-7 | pass |
| cpu | f32 | 12 | 169 | 10000 | lib:matmul_auto | 15.8495 | 14 | 36.0 | 7.27e-7 | pass |
| cpu | f32 | 12 | 169 | 10000 | lib:matmul_simple_unit_max_tile_size | 15.8214 | 14 | 36.1 | 7.27e-7 | pass |
| cpu | f32 | 12 | 169 | 10000 | lib:matmul_double_unit_max_tile_size | 9.7272 | 24 | 58.7 | 7.27e-7 | pass |
| cpu | f32 | 12 | 169 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 12 | 169 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 12 | 169 | 10000 | tiled-smem(tm=11) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 12 | 169 | 10000 | tiled-reg-cols(4x8, vec 4) | 2.6169 | 80 | 218.3 | 7.27e-7 | pass |
| cpu | f32 | 12 | 169 | 10000 | tiled-reg-rows(4x8, vec 4) | 3.8484 | 64 | 148.4 | 7.27e-7 | pass |
| cpu | f32 | 16 | 289 | 1000 | lib:matmul_auto | 8.3989 | 24 | 19.9 | 6.94e-7 | pass |
| cpu | f32 | 16 | 289 | 1000 | lib:matmul_simple_unit_max_tile_size | 8.2828 | 28 | 20.2 | 6.94e-7 | pass |
| cpu | f32 | 16 | 289 | 1000 | lib:matmul_double_unit_max_tile_size | 4.3959 | 48 | 38.0 | 6.94e-7 | pass |
| cpu | f32 | 16 | 289 | 1000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 16 | 289 | 1000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 16 | 289 | 1000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 16 | 289 | 1000 | tiled-reg-cols(4x8, vec 4) | 0.6877 | 256 | 242.9 | 6.94e-7 | pass |
| cpu | f32 | 16 | 289 | 1000 | tiled-reg-rows(4x8, vec 4) | 0.8141 | 288 | 205.2 | 6.94e-7 | pass |
| cpu | f32 | 16 | 289 | 10000 | lib:matmul_auto | 38.5076 | 6 | 43.4 | 7.38e-7 | pass |
| cpu | f32 | 16 | 289 | 10000 | lib:matmul_simple_unit_max_tile_size | 36.3593 | 6 | 45.9 | 7.38e-7 | pass |
| cpu | f32 | 16 | 289 | 10000 | lib:matmul_double_unit_max_tile_size | 27.2023 | 8 | 61.4 | 7.38e-7 | pass |
| cpu | f32 | 16 | 289 | 10000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 16 | 289 | 10000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f32 | 16 | 289 | 10000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 16 | 289 | 10000 | tiled-reg-cols(4x8, vec 4) | 10.1551 | 24 | 164.5 | 7.38e-7 | pass |
| cpu | f32 | 16 | 289 | 10000 | tiled-reg-rows(4x8, vec 4) | 8.6255 | 24 | 193.7 | 7.38e-7 | pass |

First launches (stderr):

```
[cpu] f32 p = 4, B = 1000
  lib:matmul_auto: first launch (compilation included) 0.475 s
  lib:matmul_simple_unit_max_tile_size: first launch (compilation included) 0.000 s
  lib:matmul_double_unit_max_tile_size: first launch (compilation included) 3.265 s
  tiled-reg-cols(4x8, vec 4): first launch (compilation included) 0.012 s
  tiled-reg-rows(4x8, vec 4): first launch (compilation included) 0.012 s
[cpu] f32 p = 4, B = 10000
  lib:matmul_auto: first launch (compilation included) 3.215 s
  lib:matmul_simple_unit_max_tile_size: first launch (compilation included) 0.002 s
  lib:matmul_double_unit_max_tile_size: first launch (compilation included) 31.587 s
  tiled-reg-cols(4x8, vec 4): first launch (compilation included) 0.000 s
  tiled-reg-rows(4x8, vec 4): first launch (compilation included) 0.000 s
[cpu] f32 p = 8, B = 1000
  lib:matmul_auto: first launch (compilation included) 0.002 s
  lib:matmul_simple_unit_max_tile_size: first launch (compilation included) 0.001 s
  lib:matmul_double_unit_max_tile_size: first launch (compilation included) 0.001 s
  tiled-reg-cols(4x8, vec 4): first launch (compilation included) 0.011 s
  tiled-reg-rows(4x8, vec 4): first launch (compilation included) 0.010 s
[cpu] f32 p = 8, B = 10000
  lib:matmul_auto: first launch (compilation included) 0.017 s
  lib:matmul_simple_unit_max_tile_size: first launch (compilation included) 0.006 s
  lib:matmul_double_unit_max_tile_size: first launch (compilation included) 0.005 s
  tiled-reg-cols(4x8, vec 4): first launch (compilation included) 0.001 s
  tiled-reg-rows(4x8, vec 4): first launch (compilation included) 0.001 s
[cpu] f32 p = 12, B = 1000
  lib:matmul_auto: first launch (compilation included) 0.004 s
  lib:matmul_simple_unit_max_tile_size: first launch (compilation included) 0.003 s
  lib:matmul_double_unit_max_tile_size: first launch (compilation included) 0.003 s
  tiled-reg-cols(4x8, vec 4): first launch (compilation included) 0.011 s
  tiled-reg-rows(4x8, vec 4): first launch (compilation included) 0.011 s
[cpu] f32 p = 12, B = 10000
  lib:matmul_auto: first launch (compilation included) 0.017 s
  lib:matmul_simple_unit_max_tile_size: first launch (compilation included) 0.017 s
  lib:matmul_double_unit_max_tile_size: first launch (compilation included) 0.011 s
  tiled-reg-cols(4x8, vec 4): first launch (compilation included) 0.002 s
  tiled-reg-rows(4x8, vec 4): first launch (compilation included) 0.001 s
[cpu] f32 p = 16, B = 1000
  lib:matmul_auto: first launch (compilation included) 0.010 s
  lib:matmul_simple_unit_max_tile_size: first launch (compilation included) 0.009 s
  lib:matmul_double_unit_max_tile_size: first launch (compilation included) 0.005 s
  tiled-reg-cols(4x8, vec 4): first launch (compilation included) 0.012 s
  tiled-reg-rows(4x8, vec 4): first launch (compilation included) 0.012 s
[cpu] f32 p = 16, B = 10000
  lib:matmul_auto: first launch (compilation included) 0.042 s
  lib:matmul_simple_unit_max_tile_size: first launch (compilation included) 0.042 s
  lib:matmul_double_unit_max_tile_size: first launch (compilation included) 0.033 s
  tiled-reg-cols(4x8, vec 4): first launch (compilation included) 0.004 s
  tiled-reg-rows(4x8, vec 4): first launch (compilation included) 0.004 s
```

## Worker stack probe

`CUBECL_CPU_STACK_MB=64` (the runtime's default) at the shape that overflowed the 64 MB
worker stack on 0.10.0: `--backends cpu --precisions f64 --p 16 --b 100000 --quick`, then
`--precisions f32 --no-lib` with the same shape. Both completed.

| backend | precision | p | Nc | B | implementation | median ms | reps | GFLOP/s | rel. error | check |
|---|---|---|---|---|---|---|---|---|---|---|
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_auto | 1055.6637 | 3 | 15.8 | 6.78e-16 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_simple_unit_max_tile_size | 1053.4920 | 3 | 15.9 | 6.78e-16 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_double_unit_max_tile_size | 677.7582 | 3 | 24.6 | 6.78e-16 | pass |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_simple_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 100000 | lib:matmul_double_cyclic_cmma | – | – | – | – | n/a: Unable to launch matmul because a required feature is unavailable: Plane dimensi |
| cpu | f64 | 16 | 289 | 100000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f64 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 299.2808 | 3 | 55.8 | 6.78e-16 | pass |
| cpu | f64 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 94.4009 | 3 | 176.9 | 6.78e-16 | pass |
| cpu | f32 | 16 | 289 | 100000 | tiled-smem(tm=10) | – | – | – | – | n/a: skipped on the CPU runtime (too slow) |
| cpu | f32 | 16 | 289 | 100000 | tiled-reg-cols(4x8, vec 4) | 141.3809 | 3 | 118.2 | 7.88e-7 | pass |
| cpu | f32 | 16 | 289 | 100000 | tiled-reg-rows(4x8, vec 4) | 45.2598 | 5 | 369.1 | 7.88e-7 | pass |

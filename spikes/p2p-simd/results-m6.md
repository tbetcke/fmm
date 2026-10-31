# P2P SIMD spike: raw results

| item | value |
| --- | --- |
| CPU | Apple M6 |
| cores | 12 physical, 12 logical |
| target | aarch64-macos, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| ISAs with prototypes | scalar, neon |
| green-kernels | commit `7d757c579f8633d58163cd5b2d011a9468541f17`, pulp backend `Neon(Neon { neon: Neon })` |
| thread variables | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, MKL_NUM_THREADS=unset, BLIS_NUM_THREADS=unset, RAYON_NUM_THREADS=1 |
| clock for cycles | 4.05 GHz assumed (checked by the FMA latency below) |
| mode | full |
| timing | median of 15 batches of ≥ 20 ms, one thread |

## Single instructions

Latency: one dependent chain. Throughput: 16 independent chains. Cycles at the assumed clock.

| ISA | precision | instruction | latency (cycles) | throughput (per cycle) |
| --- | --- | --- | --- | --- |
| neon | f32 | fma (FMLA / vfmadd) | 2.79 | 4.43 |
| neon | f32 | mul (FMUL / vmulp) | 2.48 | 4.51 |
| neon | f32 | add (FADD / vaddp) | 2.12 | 4.53 |
| neon | f32 | estimate (FRSQRTE / vrsqrtps) | 2.51 | 1.18 |
| neon | f32 | step (FRSQRTS / fnmadd) | 2.79 | 4.42 |
| neon | f32 | sqrt (FSQRT / vsqrtp) | 6.68 | 0.59 |
| neon | f32 | div (FDIV / vdivp) | 5.01 | 1.18 |
| neon | f64 | fma (FMLA / vfmadd) | 2.79 | 4.24 |
| neon | f64 | mul (FMUL / vmulp) | 2.48 | 4.51 |
| neon | f64 | add (FADD / vaddp) | 2.03 | 4.52 |
| neon | f64 | estimate (FRSQRTE / vrsqrtps) | 2.51 | 1.18 |
| neon | f64 | step (FRSQRTS / fnmadd) | 2.79 | 4.43 |
| neon | f64 | sqrt (FSQRT / vsqrtp) | 10.02 | 0.59 |
| neon | f64 | div (FDIV / vdivp) | 7.52 | 1.18 |

## Inverse square root: measured

Error: max |relative error| in u_T; f32 exhaustively over [1, 4), f64 on 10⁷ log-uniform samples over [2⁻¹⁰⁸, 2⁷]; plus the powers of two of the domain, their neighbours and the ends. Edge inputs show the raw result before the r² = 0 mask. Throughput on a 4,096-element buffer in L1; latency in a dependent chain.

| ISA | precision | candidate | FP ops | max error (u_T) | at x | ≤ 4 u_T | ns/elem | cycles/vector | latency (cycles) | edge inputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neon | f32 | est | 2 | 54975.965 | 2.093750e0 | no | 0.056 | 0.90 | 2.5 | 0 → +inf; min sub → 2.664e22; 1e-40 → 1.000e20 (rel 1.6e-4) |
| neon | f32 | est+N1 | 5 | 271.160 | 2.093750e0 | no | 0.083 | 1.34 | 10.9 | 0 → NaN; min sub → NaN; 1e-40 → +inf (rel -inf) |
| neon | f32 | est+N2 | 8 | 1.498 | 3.998046e0 | yes | 0.138 | 2.23 | 18.8 | 0 → NaN; min sub → NaN; 1e-40 → NaN (rel NaN) |
| neon | f32 | est+N3 | 11 | 1.498 | 3.996619e0 | yes | 0.192 | 3.12 | 26.5 | 0 → NaN; min sub → NaN; 1e-40 → NaN (rel NaN) |
| neon | f32 | est+S1 | 4 | 271.510 | 2.093746e0 | no | 0.065 | 1.05 | 10.0 | 0 → +inf; min sub → +inf; 1e-40 → +inf (rel -inf) |
| neon | f32 | est+S2 | 7 | 2.433 | 3.922365e0 | yes | 0.100 | 1.63 | 18.7 | 0 → +inf; min sub → +inf; 1e-40 → +inf (rel inf) |
| neon | f32 | est+S3 | 10 | 2.096 | 1.771779e0 | yes | 0.143 | 2.32 | 25.4 | 0 → +inf; min sub → +inf; 1e-40 → +inf (rel -inf) |
| neon | f32 | est+P2 | 6 | 2.195 | 2.093741e0 | yes | 0.110 | 1.78 | 14.6 | 0 → NaN; min sub → +inf; 1e-40 → +inf (rel inf) |
| neon | f32 | est+P3 | 7 | 1.000 | 3.985106e0 | yes | 0.138 | 2.23 | 16.5 | 0 → NaN; min sub → +inf; 1e-40 → +inf (rel -inf) |
| neon | f32 | est+P4 | 8 | 1.000 | 4.000000e0 | yes | 0.166 | 2.68 | 19.1 | 0 → NaN; min sub → +inf; 1e-40 → +inf (rel inf) |
| neon | f32 | sqrt+div | 2 | 1.500 | 4.000000e0 | yes | 0.169 | 2.73 | 11.8 | 0 → +inf; min sub → 2.671e22; 1e-40 → 1.000e20 (rel 2.7e-6) |

| ISA | precision | candidate | FP ops | max error (u_T) | at x | ≤ 4 u_T | ns/elem | cycles/vector | latency (cycles) | edge inputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neon | f64 | est | 2 | 29515321864122.965 | 4.874891e-10 | no | 0.111 | 0.90 | 2.5 | 0 → +inf; min sub → 4.490e161; 1e-310 → 9.993e154 (rel -7.0e-4) |
| neon | f64 | est+N2 | 8 | 3512695.315 | 4.874891e-10 | no | 0.275 | 2.23 | 18.8 | 0 → NaN; min sub → NaN; 1e-310 → NaN (rel NaN) |
| neon | f64 | est+N3 | 11 | 1.495 | 3.630836e-12 | yes | 0.385 | 3.12 | 26.5 | 0 → NaN; min sub → NaN; 1e-310 → NaN (rel NaN) |
| neon | f64 | est+N4 | 14 | 1.487 | 1.483026e-8 | yes | 0.495 | 4.01 | 34.6 | 0 → NaN; min sub → NaN; 1e-310 → NaN (rel NaN) |
| neon | f64 | est+S2 | 7 | 3512695.315 | 4.874891e-10 | no | 0.201 | 1.62 | 17.8 | 0 → +inf; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+S3 | 10 | 2.452 | 9.817949e-1 | yes | 0.286 | 2.32 | 25.4 | 0 → +inf; min sub → +inf; 1e-310 → +inf (rel -inf) |
| neon | f64 | est+S4 | 13 | 2.094 | 1.571777e-15 | yes | 0.372 | 3.01 | 32.9 | 0 → +inf; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+P4 | 8 | 26946.656 | 4.874891e-10 | no | 0.331 | 2.68 | 19.1 | 0 → NaN; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+P6 | 10 | 1.702 | 6.606679e-30 | yes | 0.389 | 3.15 | 24.4 | 0 → NaN; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+P7 | 11 | 1.000 | 9.764367e-4 | yes | 0.418 | 3.38 | 27.1 | 0 → NaN; min sub → +inf; 1e-310 → +inf (rel -inf) |
| neon | f64 | est+P8 | 12 | 1.000 | 9.764367e-4 | yes | 0.447 | 3.62 | 29.8 | 0 → NaN; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+N1+P2 | 10 | 95.144 | 1.996754e-6 | no | 0.358 | 2.90 | 22.2 | 0 → NaN; min sub → NaN; 1e-310 → +inf (rel -inf) |
| neon | f64 | est+N1+P3 | 11 | 1.498 | 3.469212e-18 | yes | 0.413 | 3.34 | 24.3 | 0 → NaN; min sub → NaN; 1e-310 → NaN (rel NaN) |
| neon | f64 | sqrt+div | 2 | 1.500 | 1.232595e-32 | yes | 0.376 | 3.05 | 17.7 | 0 → +inf; min sub → 4.499e161; 1e-310 → 1.000e155 (rel 0.0e0) |

## Lane utilisation of targets in lanes at the W1 leaf sizes

n_t / (lanes used per call), per ISA, precision and K (K·W targets per block; the last block is padded).

| ISA | precision | K | n_t=8 | n_t=16 | n_t=20 | n_t=24 | n_t=32 | n_t=44 | n_t=64 | n_t=128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neon | f32 | 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| neon | f32 | 2 | 100% | 100% | 83% | 100% | 100% | 92% | 100% | 100% |
| neon | f32 | 4 | 50% | 100% | 62% | 75% | 100% | 92% | 100% | 100% |
| neon | f64 | 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| neon | f64 | 2 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| neon | f64 | 4 | 100% | 100% | 83% | 100% | 100% | 92% | 100% | 100% |
| avx2 | f32 | 1 | 100% | 100% | 83% | 100% | 100% | 92% | 100% | 100% |
| avx2 | f32 | 2 | 50% | 100% | 62% | 75% | 100% | 92% | 100% | 100% |
| avx2 | f32 | 4 | 25% | 50% | 62% | 75% | 100% | 69% | 100% | 100% |
| avx2 | f64 | 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| avx2 | f64 | 2 | 100% | 100% | 83% | 100% | 100% | 92% | 100% | 100% |
| avx2 | f64 | 4 | 50% | 100% | 62% | 75% | 100% | 92% | 100% | 100% |

## Kernels: every row

Gpairs/s counts n_s n_t pairs, coincident ones included. Accuracy against `direct_sum` relative to the term magnitudes (max) and relative L2, worst of the checked sets; "= gathered": per-pair calls give the same bits as one gathered call.

### f32, potential

| cell | variant | ISA | order | K/T | rsqrt | Gpairs/s | pairs/cycle | max err φ | max err ∇ | L2 φ | L2 ∇ | check | = gathered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | reference | scalar |  |  | sqrt+div | 1.307 | 0.323 | 8.8e-8 |  | 4.5e-7 |  | pass |  |
| W1 n_t=8 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.753 | 0.680 | 3.0e-8 |  | 3.9e-7 |  | pass |  |
| W1 n_t=8 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.704 | 0.668 | 7.7e-8 |  | 5.9e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.020 | 0.993 | 7.7e-8 |  | 5.9e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 1.442 | 0.356 | 7.7e-8 |  | 5.9e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 2.712 | 0.670 | 9.9e-8 |  | 6.3e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.387 | 0.836 | 9.9e-8 |  | 6.3e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 1.256 | 0.310 | 9.9e-8 |  | 6.3e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.337 | 0.824 | 2.2e-8 |  | 4.5e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.756 | 0.927 | 2.2e-8 |  | 4.5e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.186 | 1.034 | 2.2e-8 |  | 4.5e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.993 | 0.739 | 2.1e-8 |  | 5.6e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.167 | 0.782 | 2.1e-8 |  | 5.6e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.376 | 0.834 | 2.1e-8 |  | 5.6e-7 |  | pass | no |
| W1 n_t=8 gathered | reference | scalar |  |  | sqrt+div | 0.542 | 0.134 | 8.8e-8 |  | 4.5e-7 |  | pass |  |
| W1 n_t=8 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.635 | 0.897 | 3.5e-8 |  | 6.1e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.438 | 1.096 | 7.7e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.274 | 1.302 | 7.7e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.739 | 0.676 | 7.7e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.715 | 0.917 | 9.9e-8 |  | 6.3e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.860 | 0.953 | 9.9e-8 |  | 6.3e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 2.046 | 0.505 | 9.9e-8 |  | 6.3e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.740 | 1.170 | 3.1e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.397 | 1.333 | 3.1e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.660 | 1.398 | 3.1e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.640 | 0.899 | 2.6e-8 |  | 5.5e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.982 | 0.983 | 2.6e-8 |  | 5.5e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.263 | 1.053 | 2.6e-8 |  | 5.5e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 3.379 | 0.834 | 8.8e-8 |  | 7.8e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.876 | 0.710 | 8.8e-8 |  | 7.1e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.860 | 0.953 | 9.9e-8 |  | 6.3e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 3.231 | 0.798 | 8.8e-8 |  | 7.2e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.780 | 0.933 | 7.7e-8 |  | 6.4e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 3.421 | 0.845 | 8.8e-8 |  | 7.2e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 3.135 | 0.774 | 8.8e-8 |  | 7.2e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 5.273 | 1.302 | 7.7e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=16 per-pair | reference | scalar |  |  | sqrt+div | 1.215 | 0.300 | 8.7e-8 |  | 9.4e-7 |  | pass |  |
| W1 n_t=16 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.181 | 0.785 | 1.3e-8 |  | 2.9e-7 |  | pass |  |
| W1 n_t=16 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.525 | 0.870 | 7.6e-8 |  | 8.2e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.952 | 1.223 | 7.6e-8 |  | 8.2e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 3.996 | 0.987 | 7.6e-8 |  | 8.2e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.474 | 0.858 | 5.9e-8 |  | 7.5e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.712 | 0.917 | 5.9e-8 |  | 7.5e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.206 | 0.792 | 5.9e-8 |  | 7.5e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.215 | 1.041 | 1.3e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.785 | 1.181 | 1.3e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.103 | 1.260 | 1.3e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.337 | 0.824 | 1.1e-8 |  | 2.9e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.608 | 0.891 | 1.1e-8 |  | 2.9e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.868 | 0.955 | 1.1e-8 |  | 2.9e-7 |  | pass | no |
| W1 n_t=16 gathered | reference | scalar |  |  | sqrt+div | 1.143 | 0.282 | 8.7e-8 |  | 9.4e-7 |  | pass |  |
| W1 n_t=16 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.686 | 0.910 | 3.8e-8 |  | 6.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.502 | 1.112 | 7.6e-8 |  | 8.2e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.278 | 1.303 | 7.6e-8 |  | 8.2e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.583 | 1.379 | 7.6e-8 |  | 8.2e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.747 | 0.925 | 5.9e-8 |  | 7.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.888 | 0.960 | 5.9e-8 |  | 7.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 4.145 | 1.023 | 5.9e-8 |  | 7.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.800 | 1.185 | 3.0e-8 |  | 5.7e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.450 | 1.346 | 3.0e-8 |  | 5.7e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.697 | 1.407 | 3.0e-8 |  | 5.7e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.669 | 0.906 | 3.6e-8 |  | 6.4e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 4.006 | 0.989 | 3.6e-8 |  | 6.4e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.282 | 1.057 | 3.6e-8 |  | 6.4e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 3.400 | 0.839 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.880 | 0.711 | 6.5e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.890 | 0.960 | 5.9e-8 |  | 7.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 3.239 | 0.800 | 7.0e-8 |  | 7.8e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.800 | 0.938 | 7.0e-8 |  | 7.6e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 3.439 | 0.849 | 7.6e-8 |  | 8.0e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 3.135 | 0.774 | 7.6e-8 |  | 8.0e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 5.277 | 1.303 | 7.6e-8 |  | 8.2e-7 |  | pass |  |
| W1 n_t=24 per-pair | reference | scalar |  |  | sqrt+div | 1.046 | 0.258 | 8.7e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.313 | 0.818 | 2.1e-8 |  | 2.2e-7 |  | pass |  |
| W1 n_t=24 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.890 | 0.960 | 8.1e-8 |  | 7.0e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.097 | 1.259 | 8.1e-8 |  | 7.0e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 3.318 | 0.819 | 8.1e-8 |  | 7.0e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.551 | 0.877 | 8.0e-8 |  | 6.8e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.794 | 0.937 | 8.0e-8 |  | 6.8e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 2.618 | 0.647 | 8.0e-8 |  | 6.8e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.424 | 1.092 | 2.6e-8 |  | 1.8e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.029 | 1.242 | 2.6e-8 |  | 1.8e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.328 | 1.316 | 2.6e-8 |  | 1.8e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.459 | 0.854 | 2.4e-8 |  | 1.6e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.733 | 0.922 | 2.4e-8 |  | 1.6e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.018 | 0.992 | 2.4e-8 |  | 1.6e-7 |  | pass | no |
| W1 n_t=24 gathered | reference | scalar |  |  | sqrt+div | 1.223 | 0.302 | 8.7e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.693 | 0.912 | 3.5e-8 |  | 3.1e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.517 | 1.115 | 8.1e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.297 | 1.308 | 8.1e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.214 | 1.040 | 8.1e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.755 | 0.927 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.899 | 0.963 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.124 | 0.771 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.813 | 1.188 | 3.5e-8 |  | 3.5e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.464 | 1.349 | 3.5e-8 |  | 3.5e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.710 | 1.410 | 3.5e-8 |  | 3.5e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.683 | 0.909 | 2.7e-8 |  | 3.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 4.010 | 0.990 | 2.7e-8 |  | 3.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.287 | 1.058 | 2.7e-8 |  | 3.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 3.408 | 0.841 | 8.6e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.885 | 0.712 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.898 | 0.962 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 3.244 | 0.801 | 7.2e-8 |  | 7.1e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.809 | 0.941 | 8.1e-8 |  | 6.6e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 3.443 | 0.850 | 8.1e-8 |  | 6.7e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 3.142 | 0.776 | 8.1e-8 |  | 6.7e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 5.299 | 1.308 | 8.1e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=32 per-pair | reference | scalar |  |  | sqrt+div | 0.974 | 0.240 | 5.6e-8 |  | 9.1e-7 |  | pass |  |
| W1 n_t=32 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.418 | 0.844 | 1.9e-8 |  | 3.4e-7 |  | pass |  |
| W1 n_t=32 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.104 | 1.013 | 7.0e-8 |  | 8.1e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.162 | 1.275 | 7.0e-8 |  | 8.1e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.665 | 1.152 | 7.0e-8 |  | 8.1e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.627 | 0.896 | 6.1e-8 |  | 9.2e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.825 | 0.944 | 6.1e-8 |  | 9.2e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.633 | 0.897 | 6.1e-8 |  | 9.2e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.472 | 1.104 | 1.8e-8 |  | 2.8e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.143 | 1.270 | 1.8e-8 |  | 2.8e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.444 | 1.344 | 1.8e-8 |  | 2.8e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.492 | 0.862 | 2.8e-8 |  | 3.2e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.824 | 0.944 | 2.8e-8 |  | 3.2e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.085 | 1.009 | 2.8e-8 |  | 3.2e-7 |  | pass | no |
| W1 n_t=32 gathered | reference | scalar |  |  | sqrt+div | 1.238 | 0.306 | 5.6e-8 |  | 9.1e-7 |  | pass |  |
| W1 n_t=32 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.699 | 0.913 | 3.2e-8 |  | 5.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.530 | 1.118 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.309 | 1.311 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.635 | 1.391 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.760 | 0.928 | 6.1e-8 |  | 9.2e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.904 | 0.964 | 6.1e-8 |  | 9.2e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 4.176 | 1.031 | 6.1e-8 |  | 9.2e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.821 | 1.190 | 3.0e-8 |  | 4.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.470 | 1.351 | 3.0e-8 |  | 4.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.727 | 1.414 | 3.0e-8 |  | 4.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.684 | 0.910 | 3.0e-8 |  | 4.8e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 4.014 | 0.991 | 3.0e-8 |  | 4.8e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.292 | 1.060 | 3.0e-8 |  | 4.8e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 3.412 | 0.842 | 5.3e-8 |  | 8.3e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.887 | 0.713 | 5.8e-8 |  | 7.8e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.903 | 0.964 | 6.1e-8 |  | 9.2e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 3.247 | 0.802 | 6.6e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.810 | 0.941 | 5.8e-8 |  | 8.6e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 3.447 | 0.851 | 5.8e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 3.144 | 0.776 | 5.8e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 5.304 | 1.310 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=64 per-pair | reference | scalar |  |  | sqrt+div | 0.787 | 0.194 | 1.3e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.581 | 0.884 | 2.1e-8 |  | 3.3e-7 |  | pass |  |
| W1 n_t=64 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.327 | 1.068 | 1.2e-7 |  | 1.1e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.236 | 1.293 | 1.2e-7 |  | 1.1e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.111 | 1.262 | 1.2e-7 |  | 1.1e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.707 | 0.915 | 1.2e-7 |  | 1.2e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.867 | 0.955 | 1.2e-7 |  | 1.2e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.891 | 0.961 | 1.2e-7 |  | 1.2e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.663 | 1.151 | 1.7e-8 |  | 3.7e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.319 | 1.313 | 1.7e-8 |  | 3.7e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.590 | 1.380 | 1.7e-8 |  | 3.7e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.608 | 0.891 | 1.6e-8 |  | 3.5e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.925 | 0.969 | 1.6e-8 |  | 3.5e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.193 | 1.035 | 1.6e-8 |  | 3.5e-7 |  | pass | no |
| W1 n_t=64 gathered | reference | scalar |  |  | sqrt+div | 1.285 | 0.317 | 1.3e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.711 | 0.916 | 4.0e-8 |  | 8.7e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.541 | 1.121 | 1.2e-7 |  | 1.1e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.322 | 1.314 | 1.2e-7 |  | 1.1e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.665 | 1.399 | 1.2e-7 |  | 1.1e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.772 | 0.931 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.914 | 0.966 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 4.188 | 1.034 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.844 | 1.196 | 4.9e-8 |  | 8.8e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.494 | 1.357 | 4.9e-8 |  | 8.8e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.745 | 1.419 | 4.9e-8 |  | 8.8e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.697 | 0.913 | 4.9e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 4.023 | 0.993 | 4.9e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.304 | 1.063 | 4.9e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 3.419 | 0.844 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.893 | 0.714 | 1.2e-7 |  | 1.3e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.913 | 0.966 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 3.256 | 0.804 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.820 | 0.943 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 3.456 | 0.853 | 1.1e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 3.151 | 0.778 | 1.1e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 5.323 | 1.314 | 1.2e-7 |  | 1.1e-6 |  | pass |  |
| W1 n_t=128 per-pair | reference | scalar |  |  | sqrt+div | 0.600 | 0.148 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.650 | 0.901 | 1.6e-8 |  | 3.5e-7 |  | pass |  |
| W1 n_t=128 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.447 | 1.098 | 1.1e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.287 | 1.305 | 1.1e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.386 | 1.330 | 1.1e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.744 | 0.924 | 1.0e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.894 | 0.961 | 1.0e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 4.042 | 0.998 | 1.0e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.757 | 1.175 | 1.2e-8 |  | 3.3e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.409 | 1.336 | 1.2e-8 |  | 3.3e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.680 | 1.402 | 1.2e-8 |  | 3.3e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.655 | 0.902 | 1.2e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.980 | 0.983 | 1.2e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.253 | 1.050 | 1.2e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=128 gathered | reference | scalar |  |  | sqrt+div | 1.305 | 0.322 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.713 | 0.917 | 4.5e-8 |  | 1.3e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.546 | 1.123 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.329 | 1.316 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.680 | 1.402 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.775 | 0.932 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.917 | 0.967 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 4.198 | 1.036 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.852 | 1.198 | 4.8e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.498 | 1.358 | 4.8e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.751 | 1.420 | 4.8e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.703 | 0.914 | 4.6e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 4.030 | 0.995 | 4.6e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.308 | 1.064 | 4.6e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 3.422 | 0.845 | 1.2e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.895 | 0.715 | 1.2e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.916 | 0.967 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 3.259 | 0.805 | 1.1e-7 |  | 1.7e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.826 | 0.945 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 3.459 | 0.854 | 1.2e-7 |  | 1.7e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 3.154 | 0.779 | 1.2e-7 |  | 1.7e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 5.330 | 1.316 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W2 N=1000 t≠s | reference | scalar |  |  | sqrt+div | 1.339 | 0.331 | 1.1e-7 |  | 6.2e-7 |  | pass |  |
| W2 N=1000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.712 | 0.916 | 5.9e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.541 | 1.121 | 1.2e-7 |  | 6.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.325 | 1.315 | 1.2e-7 |  | 6.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.607 | 1.385 | 1.2e-7 |  | 6.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.773 | 0.932 | 1.2e-7 |  | 6.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.916 | 0.967 | 1.2e-7 |  | 6.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 4.150 | 1.025 | 1.2e-7 |  | 6.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.823 | 1.191 | 5.7e-8 |  | 3.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.493 | 1.356 | 5.7e-8 |  | 3.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.743 | 1.418 | 5.7e-8 |  | 3.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.697 | 0.913 | 7.0e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 4.024 | 0.994 | 7.0e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.304 | 1.063 | 7.0e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t=s | reference | scalar |  |  | sqrt+div | 1.234 | 0.305 | 1.2e-7 |  | 6.6e-7 |  | pass |  |
| W2 N=1000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.707 | 0.915 | 5.7e-8 |  | 3.4e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.539 | 1.121 | 1.3e-7 |  | 6.6e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.325 | 1.315 | 1.3e-7 |  | 6.6e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.603 | 1.383 | 1.3e-7 |  | 6.6e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.770 | 0.931 | 1.5e-7 |  | 6.5e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.913 | 0.966 | 1.5e-7 |  | 6.5e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 4.148 | 1.024 | 1.5e-7 |  | 6.5e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.837 | 1.194 | 4.8e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.493 | 1.356 | 4.8e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.749 | 1.419 | 4.8e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.695 | 0.912 | 5.6e-8 |  | 3.4e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 4.024 | 0.994 | 5.6e-8 |  | 3.4e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.306 | 1.063 | 5.6e-8 |  | 3.4e-7 |  | pass |  |
| W2 N=10000 t≠s | reference | scalar |  |  | sqrt+div | 1.333 | 0.329 | 1.5e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.702 | 0.914 | 8.2e-8 |  | 8.1e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.547 | 1.123 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.327 | 1.315 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.687 | 1.404 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.771 | 0.931 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.866 | 0.954 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 4.198 | 1.036 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.829 | 1.192 | 9.4e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.492 | 1.356 | 9.4e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.754 | 1.421 | 9.4e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.686 | 0.910 | 8.7e-8 |  | 8.1e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 4.018 | 0.992 | 8.7e-8 |  | 8.1e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.300 | 1.062 | 8.7e-8 |  | 8.1e-7 |  | pass |  |
| W2 N=10000 t=s | reference | scalar |  |  | sqrt+div | 1.324 | 0.327 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.703 | 0.914 | 7.2e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 4.547 | 1.123 | 1.5e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 5.333 | 1.317 | 1.5e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 5.689 | 1.405 | 1.5e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.775 | 0.932 | 1.7e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.918 | 0.967 | 1.7e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 4.201 | 1.037 | 1.7e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 4.829 | 1.192 | 7.1e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 5.492 | 1.356 | 7.1e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 5.747 | 1.419 | 7.1e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.687 | 0.910 | 6.7e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 4.022 | 0.993 | 6.7e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 4.309 | 1.064 | 6.7e-8 |  | 8.0e-7 |  | pass |  |

### f32, potential and gradient

| cell | variant | ISA | order | K/T | rsqrt | Gpairs/s | pairs/cycle | max err φ | max err ∇ | L2 φ | L2 ∇ | check | = gathered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | reference | scalar |  |  | sqrt+div | 0.718 | 0.177 | 8.8e-8 | 1.5e-7 | 4.5e-7 | 3.0e-7 | pass |  |
| W1 n_t=8 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.767 | 0.436 | 3.0e-8 | 2.4e-7 | 3.9e-7 | 4.5e-7 | pass |  |
| W1 n_t=8 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.017 | 0.498 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.030 | 0.501 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.038 | 0.256 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 1.869 | 0.462 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 1.653 | 0.408 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 0.912 | 0.225 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.188 | 0.540 | 3.1e-8 | 5.6e-8 | 3.4e-7 | 2.1e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.148 | 0.530 | 3.1e-8 | 5.6e-8 | 3.4e-7 | 2.1e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.002 | 0.494 | 3.1e-8 | 5.6e-8 | 3.4e-7 | 2.1e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 1.900 | 0.469 | 2.2e-8 | 7.9e-8 | 4.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 1.884 | 0.465 | 2.2e-8 | 7.9e-8 | 4.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 1.804 | 0.445 | 2.2e-8 | 7.9e-8 | 4.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=8 gathered | reference | scalar |  |  | sqrt+div | 0.477 | 0.118 | 8.8e-8 | 1.5e-7 | 4.5e-7 | 3.0e-7 | pass |  |
| W1 n_t=8 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.606 | 0.643 | 3.5e-8 | 1.1e-7 | 6.1e-7 | 2.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.129 | 0.773 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.177 | 0.785 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.677 | 0.414 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.581 | 0.637 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.667 | 0.659 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 1.379 | 0.340 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.018 | 0.745 | 3.1e-8 | 4.1e-8 | 7.8e-7 | 1.8e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.298 | 0.814 | 3.1e-8 | 4.1e-8 | 7.8e-7 | 1.8e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.281 | 0.810 | 3.1e-8 | 4.1e-8 | 7.8e-7 | 1.8e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.551 | 0.630 | 2.1e-8 | 4.9e-8 | 6.6e-7 | 2.0e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.701 | 0.667 | 2.1e-8 | 4.9e-8 | 6.6e-7 | 2.0e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.782 | 0.687 | 2.1e-8 | 4.9e-8 | 6.6e-7 | 2.0e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.454 | 0.606 | 9.9e-8 | 1.3e-7 | 8.1e-7 | 2.8e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.122 | 0.524 | 7.7e-8 | 1.7e-7 | 5.8e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.666 | 0.658 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.328 | 0.575 | 8.8e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.661 | 0.657 | 8.8e-8 | 1.3e-7 | 6.1e-7 | 2.6e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.482 | 0.613 | 9.9e-8 | 1.3e-7 | 5.9e-7 | 2.9e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.315 | 0.572 | 9.9e-8 | 1.3e-7 | 5.9e-7 | 2.9e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 3.177 | 0.784 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=16 per-pair | reference | scalar |  |  | sqrt+div | 0.679 | 0.168 | 8.7e-8 | 2.2e-7 | 9.4e-7 | 4.5e-7 | pass |  |
| W1 n_t=16 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.132 | 0.527 | 1.3e-8 | 6.5e-8 | 2.9e-7 | 2.5e-7 | pass |  |
| W1 n_t=16 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.872 | 0.709 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.655 | 0.656 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.453 | 0.606 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.341 | 0.578 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.262 | 0.559 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.116 | 0.523 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.561 | 0.632 | 2.1e-8 | 7.2e-8 | 3.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.747 | 0.678 | 2.1e-8 | 7.2e-8 | 3.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.702 | 0.667 | 2.1e-8 | 7.2e-8 | 3.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.197 | 0.542 | 1.7e-8 | 8.1e-8 | 4.2e-7 | 2.9e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.288 | 0.565 | 1.7e-8 | 8.1e-8 | 4.2e-7 | 2.9e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.320 | 0.573 | 1.7e-8 | 8.1e-8 | 4.2e-7 | 2.9e-7 | pass | no |
| W1 n_t=16 gathered | reference | scalar |  |  | sqrt+div | 0.493 | 0.122 | 8.7e-8 | 2.2e-7 | 9.4e-7 | 4.5e-7 | pass |  |
| W1 n_t=16 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.659 | 0.656 | 3.8e-8 | 6.8e-8 | 6.5e-7 | 2.5e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.160 | 0.780 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.239 | 0.800 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 3.386 | 0.836 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.608 | 0.644 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.710 | 0.669 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.783 | 0.687 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.057 | 0.755 | 3.6e-8 | 1.1e-7 | 6.2e-7 | 3.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.330 | 0.822 | 3.6e-8 | 1.1e-7 | 6.2e-7 | 3.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.324 | 0.821 | 3.6e-8 | 1.1e-7 | 6.2e-7 | 3.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.578 | 0.637 | 3.5e-8 | 1.2e-7 | 5.6e-7 | 3.7e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.728 | 0.674 | 3.5e-8 | 1.2e-7 | 5.6e-7 | 3.7e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.813 | 0.695 | 3.5e-8 | 1.2e-7 | 5.6e-7 | 3.7e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.492 | 0.615 | 6.5e-8 | 2.4e-7 | 8.5e-7 | 4.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.148 | 0.530 | 5.3e-8 | 2.0e-7 | 8.3e-7 | 4.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.711 | 0.669 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.361 | 0.583 | 5.3e-8 | 2.4e-7 | 7.7e-7 | 5.0e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.702 | 0.667 | 6.5e-8 | 2.0e-7 | 8.9e-7 | 4.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.512 | 0.620 | 6.5e-8 | 2.4e-7 | 9.3e-7 | 4.9e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.330 | 0.575 | 6.5e-8 | 2.4e-7 | 9.3e-7 | 4.9e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 3.241 | 0.800 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass |  |
| W1 n_t=24 per-pair | reference | scalar |  |  | sqrt+div | 0.609 | 0.150 | 8.7e-8 | 1.7e-7 | 6.8e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.288 | 0.565 | 2.1e-8 | 4.9e-8 | 2.2e-7 | 2.0e-7 | pass |  |
| W1 n_t=24 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.939 | 0.726 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.901 | 0.716 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.099 | 0.518 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.484 | 0.613 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.447 | 0.604 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 1.776 | 0.439 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.700 | 0.667 | 2.4e-8 | 7.5e-8 | 1.7e-7 | 2.1e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.913 | 0.719 | 2.4e-8 | 7.5e-8 | 1.7e-7 | 2.1e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.895 | 0.715 | 2.4e-8 | 7.5e-8 | 1.7e-7 | 2.1e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.320 | 0.573 | 2.9e-8 | 4.8e-8 | 2.1e-7 | 2.0e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.440 | 0.602 | 2.9e-8 | 4.8e-8 | 2.1e-7 | 2.0e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.495 | 0.616 | 2.9e-8 | 4.8e-8 | 2.1e-7 | 2.0e-7 | pass | no |
| W1 n_t=24 gathered | reference | scalar |  |  | sqrt+div | 0.502 | 0.124 | 8.7e-8 | 1.7e-7 | 6.8e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.674 | 0.660 | 3.5e-8 | 7.6e-8 | 3.1e-7 | 3.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.177 | 0.784 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.253 | 0.803 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.558 | 0.632 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.615 | 0.646 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.720 | 0.672 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.101 | 0.519 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.064 | 0.756 | 3.4e-8 | 9.1e-8 | 3.8e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.342 | 0.825 | 3.4e-8 | 9.1e-8 | 3.8e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.336 | 0.824 | 3.4e-8 | 9.1e-8 | 3.8e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.585 | 0.638 | 3.5e-8 | 6.0e-8 | 3.4e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.737 | 0.676 | 3.5e-8 | 6.0e-8 | 3.4e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.828 | 0.698 | 3.5e-8 | 6.0e-8 | 3.4e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.502 | 0.618 | 8.0e-8 | 1.5e-7 | 7.7e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.155 | 0.532 | 8.4e-8 | 1.5e-7 | 7.1e-7 | 5.7e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.720 | 0.672 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.371 | 0.585 | 6.6e-8 | 1.5e-7 | 7.0e-7 | 5.5e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.714 | 0.670 | 6.6e-8 | 1.5e-7 | 6.5e-7 | 5.8e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.527 | 0.624 | 6.6e-8 | 1.5e-7 | 6.6e-7 | 5.7e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.355 | 0.581 | 6.6e-8 | 1.5e-7 | 6.6e-7 | 5.7e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 3.254 | 0.804 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=32 per-pair | reference | scalar |  |  | sqrt+div | 0.566 | 0.140 | 5.6e-8 | 4.6e-7 | 9.1e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.348 | 0.580 | 1.9e-8 | 2.2e-7 | 3.4e-7 | 2.8e-7 | pass |  |
| W1 n_t=32 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.001 | 0.741 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.020 | 0.746 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.892 | 0.714 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.513 | 0.620 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.533 | 0.625 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.440 | 0.602 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.778 | 0.686 | 1.2e-8 | 7.7e-8 | 2.6e-7 | 1.9e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.017 | 0.745 | 1.2e-8 | 7.7e-8 | 2.6e-7 | 1.9e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.009 | 0.743 | 1.2e-8 | 7.7e-8 | 2.6e-7 | 1.9e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.368 | 0.585 | 2.8e-8 | 1.4e-7 | 3.1e-7 | 2.3e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.517 | 0.621 | 2.8e-8 | 1.4e-7 | 3.1e-7 | 2.3e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.581 | 0.637 | 2.8e-8 | 1.4e-7 | 3.1e-7 | 2.3e-7 | pass | no |
| W1 n_t=32 gathered | reference | scalar |  |  | sqrt+div | 0.510 | 0.126 | 5.6e-8 | 4.6e-7 | 9.1e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.680 | 0.662 | 3.2e-8 | 2.0e-7 | 5.4e-7 | 4.7e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.181 | 0.785 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.262 | 0.805 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 3.418 | 0.844 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.618 | 0.646 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.726 | 0.673 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.806 | 0.693 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.068 | 0.758 | 3.5e-8 | 1.7e-7 | 4.2e-7 | 4.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.347 | 0.826 | 3.5e-8 | 1.7e-7 | 4.2e-7 | 4.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.339 | 0.824 | 3.5e-8 | 1.7e-7 | 4.2e-7 | 4.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.587 | 0.639 | 3.0e-8 | 1.8e-7 | 5.6e-7 | 4.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.741 | 0.677 | 3.0e-8 | 1.8e-7 | 5.6e-7 | 4.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.831 | 0.699 | 3.0e-8 | 1.8e-7 | 5.6e-7 | 4.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.508 | 0.619 | 5.9e-8 | 4.9e-7 | 9.0e-7 | 5.5e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.159 | 0.533 | 5.8e-8 | 4.9e-7 | 7.2e-7 | 5.7e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.727 | 0.673 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.375 | 0.586 | 7.0e-8 | 4.9e-7 | 8.4e-7 | 5.9e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.721 | 0.672 | 7.0e-8 | 4.9e-7 | 8.6e-7 | 5.7e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.533 | 0.625 | 7.0e-8 | 4.9e-7 | 9.0e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.361 | 0.583 | 7.0e-8 | 4.9e-7 | 9.0e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 3.265 | 0.806 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass |  |
| W1 n_t=64 per-pair | reference | scalar |  |  | sqrt+div | 0.514 | 0.127 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.9e-7 | pass |  |
| W1 n_t=64 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.515 | 0.621 | 2.1e-8 | 9.0e-8 | 3.3e-7 | 2.2e-7 | pass |  |
| W1 n_t=64 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.106 | 0.767 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.178 | 0.785 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 3.154 | 0.779 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.576 | 0.636 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.648 | 0.654 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.625 | 0.648 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.927 | 0.723 | 1.5e-8 | 1.0e-7 | 3.5e-7 | 1.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.182 | 0.786 | 1.5e-8 | 1.0e-7 | 3.5e-7 | 1.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.176 | 0.784 | 1.5e-8 | 1.0e-7 | 3.5e-7 | 1.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.485 | 0.614 | 1.8e-8 | 2.6e-7 | 3.3e-7 | 2.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.627 | 0.649 | 1.8e-8 | 2.6e-7 | 3.3e-7 | 2.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.710 | 0.669 | 1.8e-8 | 2.6e-7 | 3.3e-7 | 2.8e-7 | pass | no |
| W1 n_t=64 gathered | reference | scalar |  |  | sqrt+div | 0.547 | 0.135 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.9e-7 | pass |  |
| W1 n_t=64 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.694 | 0.665 | 4.0e-8 | 3.8e-7 | 8.7e-7 | 4.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.192 | 0.788 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.272 | 0.808 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 3.434 | 0.848 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.624 | 0.648 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.735 | 0.675 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.819 | 0.696 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.081 | 0.761 | 5.5e-8 | 2.6e-7 | 8.7e-7 | 4.3e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.358 | 0.829 | 5.5e-8 | 2.6e-7 | 8.7e-7 | 4.3e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.353 | 0.828 | 5.5e-8 | 2.6e-7 | 8.7e-7 | 4.3e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.595 | 0.641 | 5.5e-8 | 3.4e-7 | 9.3e-7 | 4.6e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.751 | 0.679 | 5.5e-8 | 3.4e-7 | 9.3e-7 | 4.6e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.840 | 0.701 | 5.5e-8 | 3.4e-7 | 9.3e-7 | 4.6e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.513 | 0.621 | 1.2e-7 | 4.4e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.165 | 0.535 | 1.3e-7 | 4.4e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.736 | 0.675 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.381 | 0.588 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.728 | 0.674 | 1.2e-7 | 2.9e-7 | 1.1e-6 | 7.1e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.538 | 0.627 | 1.2e-7 | 2.9e-7 | 1.1e-6 | 7.0e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.365 | 0.584 | 1.2e-7 | 2.9e-7 | 1.1e-6 | 7.0e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 3.274 | 0.808 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=128 per-pair | reference | scalar |  |  | sqrt+div | 0.495 | 0.122 | 1.0e-7 | 6.2e-7 | 1.6e-6 | 1.2e-6 | pass |  |
| W1 n_t=128 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.613 | 0.645 | 1.6e-8 | 1.6e-7 | 3.5e-7 | 2.4e-7 | pass |  |
| W1 n_t=128 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.156 | 0.779 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.231 | 0.798 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 3.297 | 0.814 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.604 | 0.643 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.700 | 0.667 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.724 | 0.673 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.011 | 0.743 | 1.4e-8 | 1.0e-7 | 3.0e-7 | 2.3e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.275 | 0.809 | 1.4e-8 | 1.0e-7 | 3.0e-7 | 2.3e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.275 | 0.809 | 1.4e-8 | 1.0e-7 | 3.0e-7 | 2.3e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.544 | 0.628 | 1.4e-8 | 2.2e-7 | 3.3e-7 | 2.9e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.691 | 0.665 | 1.4e-8 | 2.2e-7 | 3.3e-7 | 2.9e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.778 | 0.686 | 1.4e-8 | 2.2e-7 | 3.3e-7 | 2.9e-7 | pass | no |
| W1 n_t=128 gathered | reference | scalar |  |  | sqrt+div | 0.609 | 0.150 | 1.0e-7 | 6.2e-7 | 1.6e-6 | 1.2e-6 | pass |  |
| W1 n_t=128 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.698 | 0.666 | 4.5e-8 | 4.0e-7 | 1.3e-6 | 6.4e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.197 | 0.789 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.280 | 0.810 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 3.442 | 0.850 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.629 | 0.649 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.740 | 0.676 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.824 | 0.697 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.086 | 0.762 | 4.7e-8 | 2.8e-7 | 1.4e-6 | 5.9e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.363 | 0.830 | 4.7e-8 | 2.8e-7 | 1.4e-6 | 5.9e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.360 | 0.830 | 4.7e-8 | 2.8e-7 | 1.4e-6 | 5.9e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.598 | 0.642 | 4.4e-8 | 4.0e-7 | 1.4e-6 | 6.6e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.754 | 0.680 | 4.4e-8 | 4.0e-7 | 1.4e-6 | 6.6e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.845 | 0.703 | 4.4e-8 | 4.0e-7 | 1.4e-6 | 6.6e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.517 | 0.622 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.168 | 0.535 | 1.0e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.739 | 0.676 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.384 | 0.589 | 1.1e-7 | 5.3e-7 | 1.6e-6 | 1.0e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.732 | 0.675 | 1.1e-7 | 5.8e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.541 | 0.627 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.369 | 0.585 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 3.279 | 0.810 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W2 N=1000 t≠s | reference | scalar |  |  | sqrt+div | 0.496 | 0.122 | 1.1e-7 | 7.9e-7 | 6.2e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.683 | 0.663 | 5.9e-8 | 5.0e-7 | 3.3e-7 | 2.3e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.188 | 0.787 | 1.1e-7 | 7.4e-7 | 6.2e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.270 | 0.807 | 1.1e-7 | 7.4e-7 | 6.2e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 3.400 | 0.839 | 1.1e-7 | 7.4e-7 | 6.2e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.624 | 0.648 | 1.2e-7 | 9.4e-7 | 6.1e-7 | 4.5e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.732 | 0.675 | 1.2e-7 | 9.4e-7 | 6.1e-7 | 4.5e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.791 | 0.689 | 1.2e-7 | 9.4e-7 | 6.1e-7 | 4.5e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.079 | 0.760 | 6.9e-8 | 3.8e-7 | 3.2e-7 | 3.0e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.354 | 0.828 | 6.9e-8 | 3.8e-7 | 3.2e-7 | 3.0e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.351 | 0.827 | 6.9e-8 | 3.8e-7 | 3.2e-7 | 3.0e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.595 | 0.641 | 6.2e-8 | 4.8e-7 | 3.3e-7 | 2.3e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.743 | 0.677 | 6.2e-8 | 4.8e-7 | 3.3e-7 | 2.3e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.834 | 0.700 | 6.2e-8 | 4.8e-7 | 3.3e-7 | 2.3e-7 | pass |  |
| W2 N=1000 t=s | reference | scalar |  |  | sqrt+div | 0.513 | 0.127 | 1.2e-7 | 9.6e-7 | 6.6e-7 | 6.6e-7 | pass |  |
| W2 N=1000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.685 | 0.663 | 5.7e-8 | 3.6e-7 | 3.4e-7 | 3.2e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | check failed | – | 1.2e-7 | 1.0e-6 | 6.6e-7 | 6.7e-7 | FAIL |  |
| W2 N=1000 t=s | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | check failed | – | 1.2e-7 | 1.0e-6 | 6.6e-7 | 6.7e-7 | FAIL |  |
| W2 N=1000 t=s | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | check failed | – | 1.2e-7 | 1.0e-6 | 6.6e-7 | 6.7e-7 | FAIL |  |
| W2 N=1000 t=s | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.624 | 0.648 | 1.2e-7 | 8.6e-7 | 6.6e-7 | 6.3e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.732 | 0.674 | 1.2e-7 | 8.6e-7 | 6.6e-7 | 6.3e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.757 | 0.681 | 1.2e-7 | 8.6e-7 | 6.6e-7 | 6.3e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.077 | 0.760 | 4.8e-8 | 4.2e-7 | 3.3e-7 | 3.6e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.352 | 0.828 | 4.8e-8 | 4.2e-7 | 3.3e-7 | 3.6e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.351 | 0.827 | 4.8e-8 | 4.2e-7 | 3.3e-7 | 3.6e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.593 | 0.640 | 5.5e-8 | 4.2e-7 | 3.4e-7 | 3.2e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.745 | 0.678 | 5.5e-8 | 4.2e-7 | 3.4e-7 | 3.2e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.836 | 0.700 | 5.5e-8 | 4.2e-7 | 3.4e-7 | 3.2e-7 | pass |  |
| W2 N=10000 t≠s | reference | scalar |  |  | sqrt+div | 0.606 | 0.150 | 1.5e-7 | 2.0e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.681 | 0.662 | 8.2e-8 | 1.4e-6 | 8.1e-7 | 8.9e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.194 | 0.789 | 1.6e-7 | 2.0e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.279 | 0.810 | 1.6e-7 | 2.0e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 3.443 | 0.850 | 1.6e-7 | 2.0e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.626 | 0.648 | 1.6e-7 | 2.1e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.739 | 0.676 | 1.6e-7 | 2.1e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.825 | 0.697 | 1.6e-7 | 2.1e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.073 | 0.759 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.2e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.350 | 0.827 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.2e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.363 | 0.830 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.2e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.589 | 0.639 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.3e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.750 | 0.679 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.3e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.841 | 0.702 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.3e-7 | pass |  |
| W2 N=10000 t=s | reference | scalar |  |  | sqrt+div | 0.685 | 0.169 | 1.6e-7 | 2.7e-6 | 1.9e-6 | 1.4e-6 | FAIL |  |
| W2 N=10000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.678 | 0.661 | 7.2e-8 | 1.3e-6 | 8.0e-7 | 8.2e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 3.191 | 0.788 | 1.8e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 3.277 | 0.809 | 1.8e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 3.442 | 0.850 | 1.8e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.625 | 0.648 | 1.7e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.739 | 0.676 | 1.7e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.824 | 0.697 | 1.7e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.072 | 0.758 | 7.1e-8 | 1.3e-6 | 8.0e-7 | 8.0e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.360 | 0.830 | 7.1e-8 | 1.3e-6 | 8.0e-7 | 8.0e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.360 | 0.830 | 7.1e-8 | 1.3e-6 | 8.0e-7 | 8.0e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.587 | 0.639 | 7.5e-8 | 1.3e-6 | 8.0e-7 | 8.1e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.748 | 0.679 | 7.5e-8 | 1.3e-6 | 8.0e-7 | 8.1e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.841 | 0.701 | 7.5e-8 | 1.3e-6 | 8.0e-7 | 8.1e-7 | pass |  |

### f64, potential

| cell | variant | ISA | order | K/T | rsqrt | Gpairs/s | pairs/cycle | max err φ | max err ∇ | L2 φ | L2 ∇ | check | = gathered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | reference | scalar |  |  | sqrt+div | 1.271 | 0.314 | 8.2e-17 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.364 | 0.337 | 4.4e-17 |  | 8.6e-16 |  | pass |  |
| W1 n_t=8 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.069 | 0.264 | 1.2e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 1.694 | 0.418 | 1.2e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.261 | 0.558 | 1.2e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.022 | 0.252 | 1.2e-16 |  | 1.1e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.395 | 0.344 | 1.2e-16 |  | 1.1e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.529 | 0.378 | 1.2e-16 |  | 1.1e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.047 | 0.505 | 4.4e-17 |  | 3.8e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.268 | 0.560 | 4.4e-17 |  | 3.8e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.465 | 0.609 | 4.4e-17 |  | 3.8e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.423 | 0.351 | 4.4e-17 |  | 7.4e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.522 | 0.376 | 4.4e-17 |  | 7.4e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.647 | 0.407 | 4.4e-17 |  | 7.4e-16 |  | pass | no |
| W1 n_t=8 gathered | reference | scalar |  |  | sqrt+div | 0.521 | 0.129 | 8.2e-17 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.520 | 0.375 | 1.2e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.203 | 0.544 | 1.2e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.554 | 0.631 | 1.2e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.604 | 0.643 | 1.2e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.504 | 0.371 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.563 | 0.386 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.703 | 0.421 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.289 | 0.565 | 8.1e-17 |  | 1.6e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.552 | 0.630 | 8.1e-17 |  | 1.6e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.603 | 0.643 | 8.1e-17 |  | 1.6e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.510 | 0.373 | 1.2e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.633 | 0.403 | 1.2e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.764 | 0.435 | 1.2e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.421 | 0.351 | 1.2e-16 |  | 9.9e-16 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 1.205 | 0.297 | 1.2e-16 |  | 9.6e-16 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.595 | 0.394 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.398 | 0.345 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.415 | 0.349 | 1.2e-16 |  | 1.0e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.367 | 0.337 | 1.2e-16 |  | 1.0e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.302 | 0.321 | 1.2e-16 |  | 1.0e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.380 | 0.341 | 1.2e-16 |  | 7.8e-16 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.551 | 0.630 | 1.2e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.843 | 0.455 | 6.5e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.961 | 0.484 | 6.5e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 2.088 | 0.516 | 6.5e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.492 | 0.368 | 6.1e-14 |  | 6.6e-13 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.566 | 0.387 | 6.1e-14 |  | 6.6e-13 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.639 | 0.405 | 6.1e-14 |  | 6.6e-13 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.429 | 0.353 | 1.3e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.499 | 0.370 | 1.3e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.531 | 0.378 | 1.3e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=16 per-pair | reference | scalar |  |  | sqrt+div | 1.159 | 0.286 | 1.1e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=16 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.453 | 0.359 | 4.4e-17 |  | 9.9e-16 |  | pass |  |
| W1 n_t=16 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.692 | 0.418 | 1.2e-16 |  | 2.0e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.392 | 0.591 | 1.2e-16 |  | 2.0e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.509 | 0.619 | 1.2e-16 |  | 2.0e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.454 | 0.359 | 1.1e-16 |  | 1.8e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.548 | 0.382 | 1.1e-16 |  | 1.8e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.677 | 0.414 | 1.1e-16 |  | 1.8e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.197 | 0.543 | 3.2e-17 |  | 6.9e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.502 | 0.618 | 3.2e-17 |  | 6.9e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.636 | 0.651 | 3.2e-17 |  | 6.9e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.481 | 0.366 | 3.2e-17 |  | 8.3e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.593 | 0.393 | 3.2e-17 |  | 8.3e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.720 | 0.425 | 3.2e-17 |  | 8.3e-16 |  | pass | no |
| W1 n_t=16 gathered | reference | scalar |  |  | sqrt+div | 0.598 | 0.148 | 1.1e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=16 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.528 | 0.377 | 6.6e-17 |  | 1.4e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.239 | 0.553 | 1.2e-16 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.568 | 0.634 | 1.2e-16 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.679 | 0.661 | 1.2e-16 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.555 | 0.384 | 1.1e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.618 | 0.399 | 1.1e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.758 | 0.434 | 1.1e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.332 | 0.576 | 9.7e-17 |  | 1.1e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.593 | 0.640 | 9.7e-17 |  | 1.1e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.630 | 0.649 | 9.7e-17 |  | 1.1e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.530 | 0.378 | 6.3e-17 |  | 1.3e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.650 | 0.407 | 6.3e-17 |  | 1.3e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.780 | 0.440 | 6.3e-17 |  | 1.3e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.439 | 0.355 | 8.6e-17 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 1.222 | 0.302 | 9.5e-17 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.618 | 0.400 | 1.1e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.415 | 0.350 | 8.6e-17 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.434 | 0.354 | 9.9e-17 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.379 | 0.340 | 9.9e-17 |  | 1.9e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.309 | 0.323 | 9.9e-17 |  | 1.9e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.386 | 0.342 | 9.9e-17 |  | 1.9e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.569 | 0.634 | 1.2e-16 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.864 | 0.460 | 5.8e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.971 | 0.487 | 5.8e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 2.096 | 0.518 | 5.8e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.504 | 0.371 | 4.7e-14 |  | 8.0e-13 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.572 | 0.388 | 4.7e-14 |  | 8.0e-13 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.644 | 0.406 | 4.7e-14 |  | 8.0e-13 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.440 | 0.356 | 1.4e-16 |  | 3.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.504 | 0.371 | 1.4e-16 |  | 3.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.534 | 0.379 | 1.4e-16 |  | 3.0e-15 |  | pass |  |
| W1 n_t=24 per-pair | reference | scalar |  |  | sqrt+div | 1.013 | 0.250 | 1.3e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=24 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.469 | 0.363 | 5.1e-17 |  | 4.0e-16 |  | pass |  |
| W1 n_t=24 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.885 | 0.465 | 1.6e-16 |  | 1.3e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.452 | 0.605 | 1.6e-16 |  | 1.3e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.587 | 0.639 | 1.6e-16 |  | 1.3e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.472 | 0.363 | 1.5e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.578 | 0.390 | 1.5e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.713 | 0.423 | 1.5e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.231 | 0.551 | 5.6e-17 |  | 4.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.532 | 0.625 | 5.6e-17 |  | 4.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.642 | 0.652 | 5.6e-17 |  | 4.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.492 | 0.368 | 5.6e-17 |  | 5.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.612 | 0.398 | 5.6e-17 |  | 5.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.744 | 0.430 | 5.6e-17 |  | 5.1e-16 |  | pass | no |
| W1 n_t=24 gathered | reference | scalar |  |  | sqrt+div | 0.698 | 0.172 | 1.3e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=24 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.532 | 0.378 | 1.6e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.251 | 0.556 | 1.6e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.579 | 0.637 | 1.6e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.685 | 0.663 | 1.6e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.560 | 0.385 | 1.5e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.624 | 0.401 | 1.5e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.764 | 0.436 | 1.5e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.343 | 0.579 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.604 | 0.643 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.637 | 0.651 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.533 | 0.379 | 1.5e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.654 | 0.408 | 1.5e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.784 | 0.440 | 1.5e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.442 | 0.356 | 1.4e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 1.225 | 0.303 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.624 | 0.401 | 1.5e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.419 | 0.350 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.439 | 0.355 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.383 | 0.342 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.313 | 0.324 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.391 | 0.343 | 1.3e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.581 | 0.637 | 1.6e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.872 | 0.462 | 5.4e-12 |  | 4.3e-11 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.977 | 0.488 | 5.4e-12 |  | 4.3e-11 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 2.101 | 0.519 | 5.4e-12 |  | 4.3e-11 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.509 | 0.372 | 2.8e-14 |  | 2.7e-13 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.576 | 0.389 | 2.8e-14 |  | 2.7e-13 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.647 | 0.407 | 2.8e-14 |  | 2.7e-13 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.444 | 0.357 | 1.3e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.506 | 0.372 | 1.3e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.538 | 0.380 | 1.3e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=32 per-pair | reference | scalar |  |  | sqrt+div | 0.960 | 0.237 | 1.7e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=32 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.489 | 0.368 | 3.2e-17 |  | 6.0e-16 |  | pass |  |
| W1 n_t=32 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.962 | 0.485 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.475 | 0.611 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.613 | 0.645 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.509 | 0.373 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.583 | 0.391 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.730 | 0.427 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.261 | 0.558 | 3.1e-17 |  | 5.8e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.549 | 0.629 | 3.1e-17 |  | 5.8e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.640 | 0.652 | 3.1e-17 |  | 5.8e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.503 | 0.371 | 3.9e-17 |  | 5.6e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.622 | 0.400 | 3.9e-17 |  | 5.6e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.752 | 0.433 | 3.9e-17 |  | 5.6e-16 |  | pass | no |
| W1 n_t=32 gathered | reference | scalar |  |  | sqrt+div | 0.760 | 0.188 | 1.7e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=32 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.535 | 0.379 | 1.2e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.258 | 0.558 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.585 | 0.638 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.694 | 0.665 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.563 | 0.386 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.626 | 0.401 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.766 | 0.436 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.347 | 0.580 | 1.4e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.606 | 0.643 | 1.4e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.638 | 0.651 | 1.4e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.535 | 0.379 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.656 | 0.409 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.785 | 0.441 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.443 | 0.356 | 1.5e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 1.226 | 0.303 | 1.4e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.625 | 0.401 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.420 | 0.351 | 1.5e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.440 | 0.355 | 1.5e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.385 | 0.342 | 1.5e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.314 | 0.324 | 1.5e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.393 | 0.344 | 1.5e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.585 | 0.638 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.875 | 0.463 | 4.2e-12 |  | 9.7e-11 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.980 | 0.489 | 4.2e-12 |  | 9.7e-11 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 2.105 | 0.520 | 4.2e-12 |  | 9.7e-11 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.510 | 0.373 | 2.4e-14 |  | 6.0e-13 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.577 | 0.389 | 2.4e-14 |  | 6.0e-13 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.649 | 0.407 | 2.4e-14 |  | 6.0e-13 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.446 | 0.357 | 1.9e-16 |  | 2.1e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.509 | 0.373 | 1.9e-16 |  | 2.1e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.539 | 0.380 | 1.9e-16 |  | 2.1e-15 |  | pass |  |
| W1 n_t=64 per-pair | reference | scalar |  |  | sqrt+div | 0.782 | 0.193 | 3.1e-16 |  | 2.4e-15 |  | pass |  |
| W1 n_t=64 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.514 | 0.374 | 3.2e-17 |  | 8.0e-16 |  | pass |  |
| W1 n_t=64 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.119 | 0.523 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.541 | 0.627 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.653 | 0.655 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.540 | 0.380 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.610 | 0.397 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.750 | 0.432 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.312 | 0.571 | 3.3e-17 |  | 6.5e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.582 | 0.638 | 3.3e-17 |  | 6.5e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.651 | 0.655 | 3.3e-17 |  | 6.5e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.520 | 0.375 | 3.3e-17 |  | 7.8e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.642 | 0.405 | 3.3e-17 |  | 7.8e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.771 | 0.437 | 3.3e-17 |  | 7.8e-16 |  | pass | no |
| W1 n_t=64 gathered | reference | scalar |  |  | sqrt+div | 0.890 | 0.220 | 3.1e-16 |  | 2.4e-15 |  | pass |  |
| W1 n_t=64 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.536 | 0.379 | 1.3e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.269 | 0.560 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.588 | 0.639 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.698 | 0.666 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.565 | 0.387 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.628 | 0.402 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.768 | 0.437 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.354 | 0.581 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.611 | 0.645 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.649 | 0.654 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.536 | 0.379 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.656 | 0.409 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.789 | 0.442 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.445 | 0.357 | 2.2e-16 |  | 2.6e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 1.228 | 0.303 | 2.3e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.629 | 0.402 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.422 | 0.351 | 2.4e-16 |  | 2.6e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.442 | 0.356 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.387 | 0.343 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.316 | 0.325 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.393 | 0.344 | 2.4e-16 |  | 2.6e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.592 | 0.640 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.879 | 0.464 | 3.2e-12 |  | 1.0e-10 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.981 | 0.489 | 3.2e-12 |  | 1.0e-10 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 2.108 | 0.520 | 3.2e-12 |  | 1.0e-10 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.512 | 0.373 | 2.1e-14 |  | 5.6e-13 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.578 | 0.390 | 2.1e-14 |  | 5.6e-13 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.651 | 0.408 | 2.1e-14 |  | 5.6e-13 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.449 | 0.358 | 2.9e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.512 | 0.373 | 2.9e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.541 | 0.380 | 2.9e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=128 per-pair | reference | scalar |  |  | sqrt+div | 0.596 | 0.147 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.525 | 0.377 | 2.6e-17 |  | 7.4e-16 |  | pass |  |
| W1 n_t=128 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.196 | 0.542 | 2.2e-16 |  | 3.3e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.572 | 0.635 | 2.2e-16 |  | 3.3e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.681 | 0.662 | 2.2e-16 |  | 3.3e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.555 | 0.384 | 2.2e-16 |  | 3.2e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.621 | 0.400 | 2.2e-16 |  | 3.2e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.761 | 0.435 | 2.2e-16 |  | 3.2e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.335 | 0.577 | 2.7e-17 |  | 6.6e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.598 | 0.641 | 2.7e-17 |  | 6.6e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.598 | 0.641 | 2.7e-17 |  | 6.6e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.526 | 0.377 | 2.7e-17 |  | 6.8e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.651 | 0.408 | 2.7e-17 |  | 6.8e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.780 | 0.440 | 2.7e-17 |  | 6.8e-16 |  | pass | no |
| W1 n_t=128 gathered | reference | scalar |  |  | sqrt+div | 1.305 | 0.322 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.537 | 0.379 | 1.7e-16 |  | 2.8e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.272 | 0.561 | 2.2e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.595 | 0.641 | 2.2e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.698 | 0.666 | 2.2e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.564 | 0.386 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.629 | 0.402 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.769 | 0.437 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.356 | 0.582 | 1.7e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.612 | 0.645 | 1.7e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.647 | 0.653 | 1.7e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.536 | 0.379 | 1.6e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.658 | 0.409 | 1.6e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.788 | 0.442 | 1.6e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.446 | 0.357 | 2.3e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 1.228 | 0.303 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.629 | 0.402 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.423 | 0.351 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.443 | 0.356 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.388 | 0.343 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.316 | 0.325 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.395 | 0.344 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.595 | 0.641 | 2.2e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.882 | 0.465 | 2.5e-12 |  | 9.8e-11 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.985 | 0.490 | 2.5e-12 |  | 9.8e-11 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 2.108 | 0.520 | 2.5e-12 |  | 9.8e-11 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.513 | 0.374 | 1.8e-14 |  | 5.9e-13 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.579 | 0.390 | 1.8e-14 |  | 5.9e-13 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.651 | 0.408 | 1.8e-14 |  | 5.9e-13 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.450 | 0.358 | 2.4e-16 |  | 3.4e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.512 | 0.373 | 2.4e-16 |  | 3.4e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.541 | 0.381 | 2.4e-16 |  | 3.4e-15 |  | pass |  |
| W2 N=1000 t≠s | reference | scalar |  |  | sqrt+div | 1.214 | 0.300 | 2.1e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.537 | 0.379 | 1.8e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.264 | 0.559 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.594 | 0.641 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.705 | 0.668 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.566 | 0.387 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.629 | 0.402 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.770 | 0.437 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.357 | 0.582 | 1.8e-16 |  | 9.0e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.614 | 0.645 | 1.8e-16 |  | 9.0e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.654 | 0.655 | 1.8e-16 |  | 9.0e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.537 | 0.380 | 1.8e-16 |  | 9.1e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.659 | 0.410 | 1.8e-16 |  | 9.1e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.789 | 0.442 | 1.8e-16 |  | 9.1e-16 |  | pass |  |
| W2 N=1000 t=s | reference | scalar |  |  | sqrt+div | 0.771 | 0.190 | 3.0e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.537 | 0.379 | 2.0e-16 |  | 9.4e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.267 | 0.560 | 2.5e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.596 | 0.641 | 2.5e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.703 | 0.667 | 2.5e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.566 | 0.387 | 2.4e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.629 | 0.402 | 2.4e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.769 | 0.437 | 2.4e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.358 | 0.582 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.615 | 0.646 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.653 | 0.655 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.538 | 0.380 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.659 | 0.410 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.789 | 0.442 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=10000 t≠s | reference | scalar |  |  | sqrt+div | 1.288 | 0.318 | 2.6e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.533 | 0.378 | 2.1e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.275 | 0.562 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.599 | 0.642 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.706 | 0.668 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.567 | 0.387 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.629 | 0.402 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.771 | 0.437 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.348 | 0.580 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.611 | 0.645 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.647 | 0.654 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.531 | 0.378 | 2.3e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.658 | 0.409 | 2.3e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.789 | 0.442 | 2.3e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t=s | reference | scalar |  |  | sqrt+div | 1.180 | 0.291 | 6.2e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.532 | 0.378 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.272 | 0.561 | 6.1e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.595 | 0.641 | 6.1e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.702 | 0.667 | 6.1e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.567 | 0.387 | 6.2e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.629 | 0.402 | 6.2e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.770 | 0.437 | 6.2e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.348 | 0.580 | 1.7e-16 |  | 2.1e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.609 | 0.644 | 1.7e-16 |  | 2.1e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.594 | 0.640 | 1.7e-16 |  | 2.1e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.530 | 0.378 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.657 | 0.409 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.788 | 0.442 | 1.9e-16 |  | 2.2e-15 |  | pass |  |

### f64, potential and gradient

| cell | variant | ISA | order | K/T | rsqrt | Gpairs/s | pairs/cycle | max err φ | max err ∇ | L2 φ | L2 ∇ | check | = gathered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | reference | scalar |  |  | sqrt+div | 0.663 | 0.164 | 8.2e-17 | 4.3e-16 | 1.2e-15 | 8.1e-16 | pass |  |
| W1 n_t=8 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.990 | 0.244 | 4.4e-17 | 4.3e-16 | 8.6e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 0.883 | 0.218 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.120 | 0.277 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.068 | 0.264 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.811 | 0.200 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.891 | 0.220 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.848 | 0.209 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.274 | 0.315 | 4.4e-17 | 2.6e-16 | 7.7e-16 | 4.6e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.366 | 0.337 | 4.4e-17 | 2.6e-16 | 7.7e-16 | 4.6e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.308 | 0.323 | 4.4e-17 | 2.6e-16 | 7.7e-16 | 4.6e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.004 | 0.248 | 4.4e-17 | 4.3e-16 | 6.3e-16 | 8.4e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.050 | 0.259 | 4.4e-17 | 4.3e-16 | 6.3e-16 | 8.4e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.012 | 0.250 | 4.4e-17 | 4.3e-16 | 6.3e-16 | 8.4e-16 | pass | no |
| W1 n_t=8 gathered | reference | scalar |  |  | sqrt+div | 0.470 | 0.116 | 8.2e-17 | 4.3e-16 | 1.2e-15 | 8.1e-16 | pass |  |
| W1 n_t=8 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.154 | 0.285 | 1.2e-16 | 2.6e-16 | 1.4e-15 | 6.1e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.418 | 0.350 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.366 | 0.337 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.429 | 0.353 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.037 | 0.256 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.087 | 0.269 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.186 | 0.293 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.432 | 0.353 | 8.1e-17 | 1.4e-16 | 1.3e-15 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.580 | 0.390 | 8.1e-17 | 1.4e-16 | 1.3e-15 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.594 | 0.394 | 8.1e-17 | 1.4e-16 | 1.3e-15 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.106 | 0.273 | 4.4e-17 | 1.8e-16 | 5.5e-16 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.177 | 0.291 | 4.4e-17 | 1.8e-16 | 5.5e-16 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.210 | 0.299 | 4.4e-17 | 1.8e-16 | 5.5e-16 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.058 | 0.261 | 8.2e-17 | 4.3e-16 | 1.3e-15 | 8.0e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.927 | 0.229 | 8.2e-17 | 3.5e-16 | 8.7e-16 | 6.6e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.162 | 0.287 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.016 | 0.251 | 8.2e-17 | 3.5e-16 | 7.3e-16 | 6.5e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.091 | 0.269 | 8.2e-17 | 4.3e-16 | 7.8e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.000 | 0.247 | 8.2e-17 | 4.3e-16 | 8.0e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.983 | 0.243 | 8.2e-17 | 4.3e-16 | 8.0e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.056 | 0.261 | 8.2e-17 | 3.5e-16 | 6.0e-16 | 6.6e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.554 | 0.384 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.290 | 0.319 | 6.5e-12 | 4.0e-11 | 1.2e-10 | 1.3e-10 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.344 | 0.332 | 6.5e-12 | 4.0e-11 | 1.2e-10 | 1.3e-10 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.369 | 0.338 | 6.5e-12 | 4.0e-11 | 1.2e-10 | 1.3e-10 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.129 | 0.279 | 6.1e-14 | 3.1e-13 | 6.6e-13 | 8.9e-13 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.166 | 0.288 | 6.1e-14 | 3.1e-13 | 6.6e-13 | 8.9e-13 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.201 | 0.296 | 6.1e-14 | 3.1e-13 | 6.6e-13 | 8.9e-13 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.084 | 0.268 | 1.5e-16 | 8.2e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.088 | 0.269 | 1.5e-16 | 8.2e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.098 | 0.271 | 1.5e-16 | 8.2e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=16 per-pair | reference | scalar |  |  | sqrt+div | 0.634 | 0.157 | 1.1e-16 | 2.0e-16 | 1.7e-15 | 7.4e-16 | pass |  |
| W1 n_t=16 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.068 | 0.264 | 4.4e-17 | 1.0e-16 | 9.9e-16 | 4.7e-16 | pass |  |
| W1 n_t=16 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.301 | 0.321 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.363 | 0.336 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.317 | 0.325 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.976 | 0.241 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.053 | 0.260 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.026 | 0.253 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.362 | 0.336 | 3.2e-17 | 1.9e-16 | 4.9e-16 | 6.6e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.483 | 0.366 | 3.2e-17 | 1.9e-16 | 4.9e-16 | 6.6e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.473 | 0.364 | 3.2e-17 | 1.9e-16 | 4.9e-16 | 6.6e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.059 | 0.262 | 2.8e-17 | 1.5e-16 | 5.8e-16 | 4.4e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.122 | 0.277 | 2.8e-17 | 1.5e-16 | 5.8e-16 | 4.4e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.137 | 0.281 | 2.8e-17 | 1.5e-16 | 5.8e-16 | 4.4e-16 | pass | no |
| W1 n_t=16 gathered | reference | scalar |  |  | sqrt+div | 0.475 | 0.117 | 1.1e-16 | 2.0e-16 | 1.7e-15 | 7.4e-16 | pass |  |
| W1 n_t=16 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.174 | 0.290 | 6.6e-17 | 2.6e-16 | 1.4e-15 | 8.1e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.500 | 0.370 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.570 | 0.388 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.621 | 0.400 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.148 | 0.283 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.182 | 0.292 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.216 | 0.300 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.456 | 0.360 | 9.7e-17 | 1.9e-16 | 1.3e-15 | 8.8e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.607 | 0.397 | 9.7e-17 | 1.9e-16 | 1.3e-15 | 8.8e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.621 | 0.400 | 9.7e-17 | 1.9e-16 | 1.3e-15 | 8.8e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.123 | 0.277 | 5.5e-17 | 2.6e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.197 | 0.296 | 5.5e-17 | 2.6e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.229 | 0.303 | 5.5e-17 | 2.6e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.075 | 0.265 | 9.9e-17 | 1.5e-16 | 1.5e-15 | 7.4e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.944 | 0.233 | 9.9e-17 | 1.7e-16 | 1.5e-15 | 6.4e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.182 | 0.292 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.036 | 0.256 | 1.1e-16 | 1.6e-16 | 1.5e-15 | 7.1e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.099 | 0.271 | 1.1e-16 | 1.5e-16 | 1.6e-15 | 6.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.010 | 0.249 | 1.2e-16 | 1.5e-16 | 1.7e-15 | 6.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.990 | 0.244 | 1.2e-16 | 1.5e-16 | 1.7e-15 | 6.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.063 | 0.263 | 1.1e-16 | 1.5e-16 | 1.6e-15 | 6.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.571 | 0.388 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.303 | 0.322 | 5.8e-12 | 4.3e-11 | 1.2e-10 | 1.2e-10 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.357 | 0.335 | 5.8e-12 | 4.3e-11 | 1.2e-10 | 1.2e-10 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.386 | 0.342 | 5.8e-12 | 4.3e-11 | 1.2e-10 | 1.2e-10 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.140 | 0.281 | 4.7e-14 | 3.2e-13 | 8.0e-13 | 7.7e-13 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.176 | 0.290 | 4.7e-14 | 3.2e-13 | 8.0e-13 | 7.7e-13 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.212 | 0.299 | 4.7e-14 | 3.2e-13 | 8.0e-13 | 7.7e-13 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.093 | 0.270 | 1.3e-16 | 8.3e-16 | 2.6e-15 | 2.1e-15 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.097 | 0.271 | 1.3e-16 | 8.3e-16 | 2.6e-15 | 2.1e-15 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.109 | 0.274 | 1.3e-16 | 8.3e-16 | 2.6e-15 | 2.1e-15 | pass |  |
| W1 n_t=24 per-pair | reference | scalar |  |  | sqrt+div | 0.583 | 0.144 | 1.3e-16 | 3.2e-16 | 1.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.109 | 0.274 | 5.1e-17 | 1.3e-16 | 4.0e-16 | 3.8e-16 | pass |  |
| W1 n_t=24 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.331 | 0.329 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.441 | 0.356 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.414 | 0.349 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.063 | 0.262 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.100 | 0.272 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.091 | 0.269 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.387 | 0.343 | 3.6e-17 | 1.3e-16 | 3.8e-16 | 3.5e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.524 | 0.376 | 3.6e-17 | 1.3e-16 | 3.8e-16 | 3.5e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.531 | 0.378 | 3.6e-17 | 1.3e-16 | 3.8e-16 | 3.5e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.080 | 0.267 | 4.3e-17 | 1.3e-16 | 3.7e-16 | 4.9e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.149 | 0.284 | 4.3e-17 | 1.3e-16 | 3.7e-16 | 4.9e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.172 | 0.289 | 4.3e-17 | 1.3e-16 | 3.7e-16 | 4.9e-16 | pass | no |
| W1 n_t=24 gathered | reference | scalar |  |  | sqrt+div | 0.478 | 0.118 | 1.3e-16 | 3.2e-16 | 1.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.177 | 0.291 | 1.6e-16 | 3.0e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.506 | 0.372 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.576 | 0.389 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.631 | 0.403 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.152 | 0.284 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.187 | 0.293 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.221 | 0.301 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.461 | 0.361 | 1.4e-16 | 2.7e-16 | 1.3e-15 | 7.2e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.611 | 0.398 | 1.4e-16 | 2.7e-16 | 1.3e-15 | 7.2e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.627 | 0.402 | 1.4e-16 | 2.7e-16 | 1.3e-15 | 7.2e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.125 | 0.278 | 1.7e-16 | 3.0e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.200 | 0.296 | 1.7e-16 | 3.0e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.231 | 0.304 | 1.7e-16 | 3.0e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.079 | 0.266 | 1.3e-16 | 3.5e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.947 | 0.234 | 1.3e-16 | 3.5e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.187 | 0.293 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.039 | 0.257 | 1.1e-16 | 3.5e-16 | 1.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.103 | 0.272 | 1.3e-16 | 3.7e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.014 | 0.250 | 1.3e-16 | 3.7e-16 | 1.2e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.995 | 0.246 | 1.3e-16 | 3.7e-16 | 1.2e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.067 | 0.263 | 1.6e-16 | 3.7e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.576 | 0.389 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.309 | 0.323 | 5.4e-12 | 3.1e-11 | 4.3e-11 | 1.1e-10 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.363 | 0.336 | 5.4e-12 | 3.1e-11 | 4.3e-11 | 1.1e-10 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.391 | 0.343 | 5.4e-12 | 3.1e-11 | 4.3e-11 | 1.1e-10 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.145 | 0.283 | 2.8e-14 | 2.1e-13 | 2.7e-13 | 7.2e-13 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.180 | 0.291 | 2.8e-14 | 2.1e-13 | 2.7e-13 | 7.2e-13 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.217 | 0.300 | 2.8e-14 | 2.1e-13 | 2.7e-13 | 7.2e-13 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.097 | 0.271 | 1.6e-16 | 6.9e-16 | 1.6e-15 | 2.4e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.100 | 0.272 | 1.6e-16 | 6.9e-16 | 1.6e-15 | 2.4e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.112 | 0.275 | 1.6e-16 | 6.9e-16 | 1.6e-15 | 2.4e-15 | pass |  |
| W1 n_t=32 per-pair | reference | scalar |  |  | sqrt+div | 0.550 | 0.136 | 1.7e-16 | 4.5e-16 | 1.8e-15 | 1.0e-15 | pass |  |
| W1 n_t=32 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.125 | 0.278 | 3.2e-17 | 2.4e-16 | 6.0e-16 | 4.6e-16 | pass |  |
| W1 n_t=32 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.391 | 0.344 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.482 | 0.366 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.468 | 0.362 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.085 | 0.268 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.128 | 0.279 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.127 | 0.278 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.411 | 0.348 | 2.6e-17 | 3.1e-16 | 7.3e-16 | 4.2e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.545 | 0.382 | 2.6e-17 | 3.1e-16 | 7.3e-16 | 4.2e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.556 | 0.384 | 2.6e-17 | 3.1e-16 | 7.3e-16 | 4.2e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.094 | 0.270 | 3.3e-17 | 3.1e-16 | 5.7e-16 | 5.1e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.161 | 0.287 | 3.3e-17 | 3.1e-16 | 5.7e-16 | 5.1e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.189 | 0.294 | 3.3e-17 | 3.1e-16 | 5.7e-16 | 5.1e-16 | pass | no |
| W1 n_t=32 gathered | reference | scalar |  |  | sqrt+div | 0.479 | 0.118 | 1.7e-16 | 4.5e-16 | 1.8e-15 | 1.0e-15 | pass |  |
| W1 n_t=32 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.179 | 0.291 | 1.2e-16 | 3.3e-16 | 1.4e-15 | 7.4e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.508 | 0.372 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.578 | 0.390 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.633 | 0.403 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.154 | 0.285 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.188 | 0.293 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.223 | 0.302 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.463 | 0.361 | 1.3e-16 | 3.1e-16 | 1.2e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.612 | 0.398 | 1.3e-16 | 3.1e-16 | 1.2e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.628 | 0.402 | 1.3e-16 | 3.1e-16 | 1.2e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.126 | 0.278 | 1.3e-16 | 2.9e-16 | 1.4e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.201 | 0.297 | 1.3e-16 | 2.9e-16 | 1.4e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.234 | 0.305 | 1.3e-16 | 2.9e-16 | 1.4e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.080 | 0.267 | 1.9e-16 | 6.2e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.948 | 0.234 | 1.6e-16 | 6.2e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.188 | 0.293 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.040 | 0.257 | 2.1e-16 | 6.2e-16 | 1.9e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.104 | 0.273 | 1.6e-16 | 3.9e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.014 | 0.250 | 1.6e-16 | 3.6e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.996 | 0.246 | 1.6e-16 | 3.6e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.069 | 0.264 | 1.9e-16 | 6.2e-16 | 1.9e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.578 | 0.390 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.311 | 0.324 | 4.2e-12 | 7.4e-11 | 9.7e-11 | 1.5e-10 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.365 | 0.337 | 4.2e-12 | 7.4e-11 | 9.7e-11 | 1.5e-10 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.394 | 0.344 | 4.2e-12 | 7.4e-11 | 9.7e-11 | 1.5e-10 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.146 | 0.283 | 2.4e-14 | 4.2e-13 | 6.0e-13 | 7.6e-13 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.180 | 0.291 | 2.4e-14 | 4.2e-13 | 6.0e-13 | 7.6e-13 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.218 | 0.301 | 2.4e-14 | 4.2e-13 | 6.0e-13 | 7.6e-13 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.099 | 0.271 | 2.1e-16 | 8.1e-16 | 1.8e-15 | 2.0e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.101 | 0.272 | 2.1e-16 | 8.1e-16 | 1.8e-15 | 2.0e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.114 | 0.275 | 2.1e-16 | 8.1e-16 | 1.8e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 per-pair | reference | scalar |  |  | sqrt+div | 0.509 | 0.126 | 3.1e-16 | 1.7e-15 | 2.4e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.156 | 0.286 | 3.2e-17 | 1.6e-16 | 8.0e-16 | 4.3e-16 | pass |  |
| W1 n_t=64 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.455 | 0.359 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.537 | 0.379 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.551 | 0.383 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.124 | 0.277 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.162 | 0.287 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.177 | 0.291 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.440 | 0.355 | 3.2e-17 | 2.2e-16 | 6.9e-16 | 4.0e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.581 | 0.390 | 3.2e-17 | 2.2e-16 | 6.9e-16 | 4.0e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.594 | 0.394 | 3.2e-17 | 2.2e-16 | 6.9e-16 | 4.0e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.111 | 0.274 | 3.3e-17 | 4.4e-16 | 7.5e-16 | 4.7e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.183 | 0.292 | 3.3e-17 | 4.4e-16 | 7.5e-16 | 4.7e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.213 | 0.300 | 3.3e-17 | 4.4e-16 | 7.5e-16 | 4.7e-16 | pass | no |
| W1 n_t=64 gathered | reference | scalar |  |  | sqrt+div | 0.483 | 0.119 | 3.1e-16 | 1.7e-15 | 2.4e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.181 | 0.292 | 1.3e-16 | 2.3e-15 | 1.8e-15 | 2.5e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.513 | 0.374 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.583 | 0.391 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.638 | 0.405 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.156 | 0.286 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.191 | 0.294 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.226 | 0.303 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.464 | 0.362 | 1.2e-16 | 2.5e-15 | 1.9e-15 | 2.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.616 | 0.399 | 1.2e-16 | 2.5e-15 | 1.9e-15 | 2.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.630 | 0.402 | 1.2e-16 | 2.5e-15 | 1.9e-15 | 2.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.127 | 0.278 | 1.4e-16 | 2.2e-15 | 1.8e-15 | 2.4e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.202 | 0.297 | 1.4e-16 | 2.2e-15 | 1.8e-15 | 2.4e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.234 | 0.305 | 1.4e-16 | 2.2e-15 | 1.8e-15 | 2.4e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.081 | 0.267 | 2.9e-16 | 1.5e-15 | 2.4e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.950 | 0.234 | 2.9e-16 | 1.5e-15 | 2.3e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.190 | 0.294 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.042 | 0.257 | 2.9e-16 | 1.6e-15 | 2.5e-15 | 1.8e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.106 | 0.273 | 2.9e-16 | 1.5e-15 | 2.5e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.017 | 0.251 | 2.9e-16 | 1.5e-15 | 2.5e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.999 | 0.247 | 2.9e-16 | 1.5e-15 | 2.5e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.070 | 0.264 | 2.9e-16 | 1.5e-15 | 2.4e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.582 | 0.391 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.314 | 0.324 | 3.2e-12 | 2.9e-11 | 1.0e-10 | 1.0e-10 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.368 | 0.338 | 3.2e-12 | 2.9e-11 | 1.0e-10 | 1.0e-10 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.396 | 0.345 | 3.2e-12 | 2.9e-11 | 1.0e-10 | 1.0e-10 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.149 | 0.284 | 2.1e-14 | 1.4e-13 | 5.6e-13 | 5.7e-13 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.182 | 0.292 | 2.1e-14 | 1.4e-13 | 5.6e-13 | 5.7e-13 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.221 | 0.302 | 2.1e-14 | 1.4e-13 | 5.6e-13 | 5.7e-13 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.101 | 0.272 | 3.3e-16 | 1.5e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.103 | 0.272 | 3.3e-16 | 1.5e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.117 | 0.276 | 3.3e-16 | 1.5e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 per-pair | reference | scalar |  |  | sqrt+div | 0.492 | 0.122 | 2.1e-16 | 7.7e-16 | 3.3e-15 | 2.1e-15 | pass |  |
| W1 n_t=128 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.169 | 0.289 | 2.6e-17 | 1.7e-16 | 7.4e-16 | 5.4e-16 | pass |  |
| W1 n_t=128 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.487 | 0.367 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.563 | 0.386 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.595 | 0.394 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.140 | 0.282 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.178 | 0.291 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.202 | 0.297 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.453 | 0.359 | 2.7e-17 | 3.6e-16 | 6.6e-16 | 6.6e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.600 | 0.395 | 2.7e-17 | 3.6e-16 | 6.6e-16 | 6.6e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.614 | 0.399 | 2.7e-17 | 3.6e-16 | 6.6e-16 | 6.6e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.120 | 0.277 | 2.2e-17 | 2.3e-16 | 7.5e-16 | 5.0e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.193 | 0.295 | 2.2e-17 | 2.3e-16 | 7.5e-16 | 5.0e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.225 | 0.303 | 2.2e-17 | 2.3e-16 | 7.5e-16 | 5.0e-16 | pass | no |
| W1 n_t=128 gathered | reference | scalar |  |  | sqrt+div | 0.492 | 0.121 | 2.1e-16 | 7.7e-16 | 3.3e-15 | 2.1e-15 | pass |  |
| W1 n_t=128 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.181 | 0.292 | 1.7e-16 | 1.5e-15 | 2.8e-15 | 2.1e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.515 | 0.374 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.584 | 0.391 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.639 | 0.405 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.157 | 0.286 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.192 | 0.294 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.226 | 0.303 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.465 | 0.362 | 1.6e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.616 | 0.399 | 1.6e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.631 | 0.403 | 1.6e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.128 | 0.279 | 1.8e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.204 | 0.297 | 1.8e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.236 | 0.305 | 1.8e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.083 | 0.267 | 2.0e-16 | 9.5e-16 | 3.3e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.950 | 0.235 | 2.0e-16 | 9.9e-16 | 3.3e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.192 | 0.294 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.043 | 0.258 | 2.0e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.107 | 0.273 | 2.0e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.017 | 0.251 | 2.0e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.998 | 0.246 | 2.0e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.070 | 0.264 | 2.0e-16 | 9.5e-16 | 3.3e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.583 | 0.391 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.315 | 0.325 | 2.5e-12 | 5.5e-11 | 9.8e-11 | 1.2e-10 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.369 | 0.338 | 2.5e-12 | 5.5e-11 | 9.8e-11 | 1.2e-10 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.399 | 0.345 | 2.5e-12 | 5.5e-11 | 9.8e-11 | 1.2e-10 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.149 | 0.284 | 1.8e-14 | 4.0e-13 | 5.9e-13 | 6.7e-13 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.184 | 0.292 | 1.8e-14 | 4.0e-13 | 5.9e-13 | 6.7e-13 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.222 | 0.302 | 1.8e-14 | 4.0e-13 | 5.9e-13 | 6.7e-13 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.101 | 0.272 | 2.7e-16 | 1.4e-15 | 3.4e-15 | 2.9e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.104 | 0.273 | 2.7e-16 | 1.4e-15 | 3.4e-15 | 2.9e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.117 | 0.276 | 2.7e-16 | 1.4e-15 | 3.4e-15 | 2.9e-15 | pass |  |
| W2 N=1000 t≠s | reference | scalar |  |  | sqrt+div | 0.482 | 0.119 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.3e-16 | pass |  |
| W2 N=1000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.181 | 0.292 | 1.8e-16 | 8.5e-16 | 9.2e-16 | 6.2e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.512 | 0.373 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.2e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.582 | 0.391 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.2e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.636 | 0.404 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.2e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.156 | 0.286 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.1e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.190 | 0.294 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.1e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.225 | 0.303 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.1e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.465 | 0.362 | 1.7e-16 | 6.5e-16 | 8.9e-16 | 5.8e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.615 | 0.399 | 1.7e-16 | 6.5e-16 | 8.9e-16 | 5.8e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.631 | 0.403 | 1.7e-16 | 6.5e-16 | 8.9e-16 | 5.8e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.126 | 0.278 | 1.8e-16 | 5.8e-16 | 9.0e-16 | 5.7e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.202 | 0.297 | 1.8e-16 | 5.8e-16 | 9.0e-16 | 5.7e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.235 | 0.305 | 1.8e-16 | 5.8e-16 | 9.0e-16 | 5.7e-16 | pass |  |
| W2 N=1000 t=s | reference | scalar |  |  | sqrt+div | 0.482 | 0.119 | 3.0e-16 | 1.6e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.181 | 0.292 | 2.0e-16 | 1.1e-15 | 9.4e-16 | 8.3e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.513 | 0.374 | 3.0e-16 | 1.5e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.583 | 0.391 | 3.0e-16 | 1.5e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.636 | 0.404 | 3.0e-16 | 1.5e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.156 | 0.285 | 2.6e-16 | 1.7e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.191 | 0.294 | 2.6e-16 | 1.7e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.226 | 0.303 | 2.6e-16 | 1.7e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.464 | 0.362 | 1.7e-16 | 9.5e-16 | 9.2e-16 | 7.7e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.615 | 0.399 | 1.7e-16 | 9.5e-16 | 9.2e-16 | 7.7e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.632 | 0.403 | 1.7e-16 | 9.5e-16 | 9.2e-16 | 7.7e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.128 | 0.278 | 1.5e-16 | 1.1e-15 | 9.2e-16 | 8.0e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.202 | 0.297 | 1.5e-16 | 1.1e-15 | 9.2e-16 | 8.0e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.234 | 0.305 | 1.5e-16 | 1.1e-15 | 9.2e-16 | 8.0e-16 | pass |  |
| W2 N=10000 t≠s | reference | scalar |  |  | sqrt+div | 0.513 | 0.127 | 2.6e-16 | 4.7e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.179 | 0.291 | 2.1e-16 | 3.4e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.514 | 0.374 | 2.8e-16 | 4.6e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.584 | 0.391 | 2.8e-16 | 4.6e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.640 | 0.405 | 2.8e-16 | 4.6e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.157 | 0.286 | 2.6e-16 | 4.7e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.192 | 0.294 | 2.6e-16 | 4.7e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.228 | 0.303 | 2.6e-16 | 4.7e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.464 | 0.361 | 2.0e-16 | 3.3e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.616 | 0.399 | 2.0e-16 | 3.3e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.632 | 0.403 | 2.0e-16 | 3.3e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.127 | 0.278 | 2.2e-16 | 3.4e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.204 | 0.297 | 2.2e-16 | 3.4e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.237 | 0.305 | 2.2e-16 | 3.4e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t=s | reference | scalar |  |  | sqrt+div | 0.534 | 0.132 | 6.2e-16 | 6.8e-15 | 3.5e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.178 | 0.291 | 1.9e-16 | 3.6e-15 | 2.2e-15 | 2.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.514 | 0.374 | 5.9e-16 | 6.7e-15 | 3.5e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.584 | 0.391 | 5.9e-16 | 6.7e-15 | 3.5e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.642 | 0.405 | 5.9e-16 | 6.7e-15 | 3.5e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 1.157 | 0.286 | 6.1e-16 | 6.9e-15 | 3.5e-15 | 3.4e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 1.192 | 0.294 | 6.1e-16 | 6.9e-15 | 3.5e-15 | 3.4e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 1.228 | 0.303 | 6.1e-16 | 6.9e-15 | 3.5e-15 | 3.4e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.463 | 0.361 | 1.7e-16 | 3.4e-15 | 2.1e-15 | 2.2e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.616 | 0.399 | 1.7e-16 | 3.4e-15 | 2.1e-15 | 2.2e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.633 | 0.403 | 1.7e-16 | 3.4e-15 | 2.1e-15 | 2.2e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.126 | 0.278 | 1.8e-16 | 3.5e-15 | 2.1e-15 | 2.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.203 | 0.297 | 1.8e-16 | 3.5e-15 | 2.1e-15 | 2.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.236 | 0.305 | 1.8e-16 | 3.5e-15 | 2.1e-15 | 2.3e-15 | pass |  |

## Inverse square root inside the kernel

Targets in lanes, K = 2, gathered W1 cells, Gpairs/s; the geometric mean over n_t decides the formulation among those within 4 u_T in the study.

### f32, potential

| rsqrt | FP ops | study error (u_T) | W1 n_t=8 gathered | W1 n_t=16 gathered | W1 n_t=24 gathered | W1 n_t=32 gathered | W1 n_t=64 gathered | W1 n_t=128 gathered | geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| est+N2 | 17 | 1.50 | 3.379 | 3.400 | 3.408 | 3.412 | 3.419 | 3.422 | 3.406 |
| est+N3 | 20 | 1.50 | 2.876 | 2.880 | 2.885 | 2.887 | 2.893 | 2.895 | 2.886 |
| est+S2 | 16 | 2.43 | 3.860 | 3.890 | 3.898 | 3.903 | 3.913 | 3.916 | 3.896 |
| est+S3 | 19 | 2.10 | 3.231 | 3.239 | 3.244 | 3.247 | 3.256 | 3.259 | 3.246 |
| est+P2 | 15 | 2.19 | 3.780 | 3.800 | 3.809 | 3.810 | 3.820 | 3.826 | 3.808 |
| est+P3 | 16 | 1.00 | 3.421 | 3.439 | 3.443 | 3.447 | 3.456 | 3.459 | 3.444 |
| est+P4 | 17 | 1.00 | 3.135 | 3.135 | 3.142 | 3.144 | 3.151 | 3.154 | 3.144 |
| sqrt+div | 11 | 1.50 | 5.273 | 5.277 | 5.299 | 5.304 | 5.323 | 5.330 | 5.301 |

### f32, potential and gradient

| rsqrt | FP ops | study error (u_T) | W1 n_t=8 gathered | W1 n_t=16 gathered | W1 n_t=24 gathered | W1 n_t=32 gathered | W1 n_t=64 gathered | W1 n_t=128 gathered | geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| est+N2 | 23 | 1.50 | 2.454 | 2.492 | 2.502 | 2.508 | 2.513 | 2.517 | 2.498 |
| est+N3 | 26 | 1.50 | 2.122 | 2.148 | 2.155 | 2.159 | 2.165 | 2.168 | 2.153 |
| est+S2 | 22 | 2.43 | 2.666 | 2.711 | 2.720 | 2.727 | 2.736 | 2.739 | 2.716 |
| est+S3 | 25 | 2.10 | 2.328 | 2.361 | 2.371 | 2.375 | 2.381 | 2.384 | 2.367 |
| est+P2 | 21 | 2.19 | 2.661 | 2.702 | 2.714 | 2.721 | 2.728 | 2.732 | 2.710 |
| est+P3 | 22 | 1.00 | 2.482 | 2.512 | 2.527 | 2.533 | 2.538 | 2.541 | 2.522 |
| est+P4 | 23 | 1.00 | 2.315 | 2.330 | 2.355 | 2.361 | 2.365 | 2.369 | 2.349 |
| sqrt+div | 17 | 1.50 | 3.177 | 3.241 | 3.254 | 3.265 | 3.274 | 3.279 | 3.248 |

### f64, potential

| rsqrt | FP ops | study error (u_T) | W1 n_t=8 gathered | W1 n_t=16 gathered | W1 n_t=24 gathered | W1 n_t=32 gathered | W1 n_t=64 gathered | W1 n_t=128 gathered | geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| est+N3 | 20 | 1.50 | 1.421 | 1.439 | 1.442 | 1.443 | 1.445 | 1.446 | 1.439 |
| est+N4 | 23 | 1.49 | 1.205 | 1.222 | 1.225 | 1.226 | 1.228 | 1.228 | 1.222 |
| est+S3 | 19 | 2.45 | 1.595 | 1.618 | 1.624 | 1.625 | 1.629 | 1.629 | 1.620 |
| est+S4 | 22 | 2.09 | 1.398 | 1.415 | 1.419 | 1.420 | 1.422 | 1.423 | 1.416 |
| est+P6 | 19 | 1.70 | 1.415 | 1.434 | 1.439 | 1.440 | 1.442 | 1.443 | 1.435 |
| est+P7 | 20 | 1.00 | 1.367 | 1.379 | 1.383 | 1.385 | 1.387 | 1.388 | 1.382 |
| est+P8 | 21 | 1.00 | 1.302 | 1.309 | 1.313 | 1.314 | 1.316 | 1.316 | 1.312 |
| est+N1+P3 | 20 | 1.50 | 1.380 | 1.386 | 1.391 | 1.393 | 1.393 | 1.395 | 1.390 |
| sqrt+div | 11 | 1.50 | 2.551 | 2.569 | 2.581 | 2.585 | 2.592 | 2.595 | 2.579 |

### f64, potential and gradient

| rsqrt | FP ops | study error (u_T) | W1 n_t=8 gathered | W1 n_t=16 gathered | W1 n_t=24 gathered | W1 n_t=32 gathered | W1 n_t=64 gathered | W1 n_t=128 gathered | geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| est+N3 | 26 | 1.50 | 1.058 | 1.075 | 1.079 | 1.080 | 1.081 | 1.083 | 1.076 |
| est+N4 | 29 | 1.49 | 0.927 | 0.944 | 0.947 | 0.948 | 0.950 | 0.950 | 0.944 |
| est+S3 | 25 | 2.45 | 1.162 | 1.182 | 1.187 | 1.188 | 1.190 | 1.192 | 1.183 |
| est+S4 | 28 | 2.09 | 1.016 | 1.036 | 1.039 | 1.040 | 1.042 | 1.043 | 1.036 |
| est+P6 | 25 | 1.70 | 1.091 | 1.099 | 1.103 | 1.104 | 1.106 | 1.107 | 1.102 |
| est+P7 | 26 | 1.00 | 1.000 | 1.010 | 1.014 | 1.014 | 1.017 | 1.017 | 1.012 |
| est+P8 | 27 | 1.00 | 0.983 | 0.990 | 0.995 | 0.996 | 0.999 | 0.998 | 0.993 |
| est+N1+P3 | 26 | 1.50 | 1.056 | 1.063 | 1.067 | 1.069 | 1.070 | 1.070 | 1.066 |
| sqrt+div | 17 | 1.50 | 1.554 | 1.571 | 1.576 | 1.578 | 1.582 | 1.583 | 1.574 |

## Kernels: summary

Gpairs/s. TIL = targets in lanes (K vectors per block), SIL = sources in lanes (T targets per block), both with the "best" inverse square root; "TIL gk" is the best K with green-kernels' formulation. Ratios: best TIL over green-kernels, over the reference, and best SIL over best TIL.

### f32, potential

| cell | reference | green-kernels | TIL K=1 | K=2 | K=4 | SIL T=1 | T=2 | T=4 | TIL gk | TIL/gk | TIL/ref | SIL/TIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | 1.307 | 2.753 | 2.704 | 4.020 | 1.442 | 3.337 | 3.756 | 4.186 | 3.387 (K=2) | 1.46 | 3.08 | 1.04 |
| W1 n_t=8 gathered | 0.542 | 3.635 | 4.438 | 5.274 | 2.739 | 4.740 | 5.397 | 5.660 | 3.860 (K=2) | 1.45 | 9.73 | 1.07 |
| W1 n_t=16 per-pair | 1.215 | 3.181 | 3.525 | 4.952 | 3.996 | 4.215 | 4.785 | 5.103 | 3.712 (K=2) | 1.56 | 4.08 | 1.03 |
| W1 n_t=16 gathered | 1.143 | 3.686 | 4.502 | 5.278 | 5.583 | 4.800 | 5.450 | 5.697 | 4.145 (K=4) | 1.51 | 4.88 | 1.02 |
| W1 n_t=24 per-pair | 1.046 | 3.313 | 3.890 | 5.097 | 3.318 | 4.424 | 5.029 | 5.328 | 3.794 (K=2) | 1.54 | 4.87 | 1.05 |
| W1 n_t=24 gathered | 1.223 | 3.693 | 4.517 | 5.297 | 4.214 | 4.813 | 5.464 | 5.710 | 3.899 (K=2) | 1.43 | 4.33 | 1.08 |
| W1 n_t=32 per-pair | 0.974 | 3.418 | 4.104 | 5.162 | 4.665 | 4.472 | 5.143 | 5.444 | 3.825 (K=2) | 1.51 | 5.30 | 1.05 |
| W1 n_t=32 gathered | 1.238 | 3.699 | 4.530 | 5.309 | 5.635 | 4.821 | 5.470 | 5.727 | 4.176 (K=4) | 1.52 | 4.55 | 1.02 |
| W1 n_t=64 per-pair | 0.787 | 3.581 | 4.327 | 5.236 | 5.111 | 4.663 | 5.319 | 5.590 | 3.891 (K=4) | 1.46 | 6.65 | 1.07 |
| W1 n_t=64 gathered | 1.285 | 3.711 | 4.541 | 5.322 | 5.665 | 4.844 | 5.494 | 5.745 | 4.188 (K=4) | 1.53 | 4.41 | 1.01 |
| W1 n_t=128 per-pair | 0.600 | 3.650 | 4.447 | 5.287 | 5.386 | 4.757 | 5.409 | 5.680 | 4.042 (K=4) | 1.48 | 8.97 | 1.05 |
| W1 n_t=128 gathered | 1.305 | 3.713 | 4.546 | 5.329 | 5.680 | 4.852 | 5.498 | 5.751 | 4.198 (K=4) | 1.53 | 4.35 | 1.01 |
| W2 N=1000 t≠s | 1.339 | 3.712 | 4.541 | 5.325 | 5.607 | 4.823 | 5.493 | 5.743 | 4.150 (K=4) | 1.51 | 4.19 | 1.02 |
| W2 N=1000 t=s | 1.234 | 3.707 | 4.539 | 5.325 | 5.603 | 4.837 | 5.493 | 5.749 | 4.148 (K=4) | 1.51 | 4.54 | 1.03 |
| W2 N=10000 t≠s | 1.333 | 3.702 | 4.547 | 5.327 | 5.687 | 4.829 | 5.492 | 5.754 | 4.198 (K=4) | 1.54 | 4.27 | 1.01 |
| W2 N=10000 t=s | 1.324 | 3.703 | 4.547 | 5.333 | 5.689 | 4.829 | 5.492 | 5.747 | 4.201 (K=4) | 1.54 | 4.30 | 1.01 |

Targets in lanes, geometric mean over the W1 cells (Gpairs/s): K=1: 4.131, K=2: 5.116, K=4: 4.183.

### f32, potential and gradient

| cell | reference | green-kernels | TIL K=1 | K=2 | K=4 | SIL T=1 | T=2 | T=4 | TIL gk | TIL/gk | TIL/ref | SIL/TIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | 0.718 | 1.767 | 2.017 | 2.030 | 1.038 | 2.188 | 2.148 | 2.002 | 1.869 (K=1) | 1.15 | 2.83 | 1.08 |
| W1 n_t=8 gathered | 0.477 | 2.606 | 3.129 | 3.177 | 1.677 | 3.018 | 3.298 | 3.281 | 2.667 (K=2) | 1.22 | 6.67 | 1.04 |
| W1 n_t=16 per-pair | 0.679 | 2.132 | 2.872 | 2.655 | 2.453 | 2.561 | 2.747 | 2.702 | 2.341 (K=1) | 1.35 | 4.23 | 0.96 |
| W1 n_t=16 gathered | 0.493 | 2.659 | 3.160 | 3.239 | 3.386 | 3.057 | 3.330 | 3.324 | 2.783 (K=4) | 1.27 | 6.87 | 0.98 |
| W1 n_t=24 per-pair | 0.609 | 2.288 | 2.939 | 2.901 | 2.099 | 2.700 | 2.913 | 2.895 | 2.484 (K=1) | 1.28 | 4.82 | 0.99 |
| W1 n_t=24 gathered | 0.502 | 2.674 | 3.177 | 3.253 | 2.558 | 3.064 | 3.342 | 3.336 | 2.720 (K=2) | 1.22 | 6.47 | 1.03 |
| W1 n_t=32 per-pair | 0.566 | 2.348 | 3.001 | 3.020 | 2.892 | 2.778 | 3.017 | 3.009 | 2.533 (K=2) | 1.29 | 5.33 | 1.00 |
| W1 n_t=32 gathered | 0.510 | 2.680 | 3.181 | 3.262 | 3.418 | 3.068 | 3.347 | 3.339 | 2.806 (K=4) | 1.28 | 6.70 | 0.98 |
| W1 n_t=64 per-pair | 0.514 | 2.515 | 3.106 | 3.178 | 3.154 | 2.927 | 3.182 | 3.176 | 2.648 (K=2) | 1.26 | 6.18 | 1.00 |
| W1 n_t=64 gathered | 0.547 | 2.694 | 3.192 | 3.272 | 3.434 | 3.081 | 3.358 | 3.353 | 2.819 (K=4) | 1.27 | 6.28 | 0.98 |
| W1 n_t=128 per-pair | 0.495 | 2.613 | 3.156 | 3.231 | 3.297 | 3.011 | 3.275 | 3.275 | 2.724 (K=4) | 1.26 | 6.66 | 0.99 |
| W1 n_t=128 gathered | 0.609 | 2.698 | 3.197 | 3.280 | 3.442 | 3.086 | 3.363 | 3.360 | 2.824 (K=4) | 1.28 | 5.66 | 0.98 |
| W2 N=1000 t≠s | 0.496 | 2.683 | 3.188 | 3.270 | 3.400 | 3.079 | 3.354 | 3.351 | 2.791 (K=4) | 1.27 | 6.85 | 0.99 |
| W2 N=1000 t=s | 0.513 | 2.685 | check failed | check failed | check failed | 3.077 | 3.352 | 3.351 | 2.757 (K=4) | – | – | – |
| W2 N=10000 t≠s | 0.606 | 2.681 | 3.194 | 3.279 | 3.443 | 3.073 | 3.350 | 3.363 | 2.825 (K=4) | 1.28 | 5.68 | 0.98 |
| W2 N=10000 t=s | 0.685 | 2.678 | 3.191 | 3.277 | 3.442 | 3.072 | 3.360 | 3.360 | 2.824 (K=4) | 1.29 | 5.03 | 0.98 |

Targets in lanes, geometric mean over the W1 cells (Gpairs/s): K=1: 2.990, K=2: 3.017, K=4: 2.597.

### f64, potential

| cell | reference | green-kernels | TIL K=1 | K=2 | K=4 | SIL T=1 | T=2 | T=4 | TIL gk | TIL/gk | TIL/ref | SIL/TIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | 1.271 | 1.364 | 1.069 | 1.694 | 2.261 | 2.047 | 2.268 | 2.465 | 1.529 (K=4) | 1.66 | 1.78 | 1.09 |
| W1 n_t=8 gathered | 0.521 | 1.520 | 2.203 | 2.554 | 2.604 | 2.289 | 2.552 | 2.603 | 1.703 (K=4) | 1.71 | 4.99 | 1.00 |
| W1 n_t=16 per-pair | 1.159 | 1.453 | 1.692 | 2.392 | 2.509 | 2.197 | 2.502 | 2.636 | 1.677 (K=4) | 1.73 | 2.16 | 1.05 |
| W1 n_t=16 gathered | 0.598 | 1.528 | 2.239 | 2.568 | 2.679 | 2.332 | 2.593 | 2.630 | 1.758 (K=4) | 1.75 | 4.48 | 0.98 |
| W1 n_t=24 per-pair | 1.013 | 1.469 | 1.885 | 2.452 | 2.587 | 2.231 | 2.532 | 2.642 | 1.713 (K=4) | 1.76 | 2.55 | 1.02 |
| W1 n_t=24 gathered | 0.698 | 1.532 | 2.251 | 2.579 | 2.685 | 2.343 | 2.604 | 2.637 | 1.764 (K=4) | 1.75 | 3.84 | 0.98 |
| W1 n_t=32 per-pair | 0.960 | 1.489 | 1.962 | 2.475 | 2.613 | 2.261 | 2.549 | 2.640 | 1.730 (K=4) | 1.75 | 2.72 | 1.01 |
| W1 n_t=32 gathered | 0.760 | 1.535 | 2.258 | 2.585 | 2.694 | 2.347 | 2.606 | 2.638 | 1.766 (K=4) | 1.75 | 3.54 | 0.98 |
| W1 n_t=64 per-pair | 0.782 | 1.514 | 2.119 | 2.541 | 2.653 | 2.312 | 2.582 | 2.651 | 1.750 (K=4) | 1.75 | 3.39 | 1.00 |
| W1 n_t=64 gathered | 0.890 | 1.536 | 2.269 | 2.588 | 2.698 | 2.354 | 2.611 | 2.649 | 1.768 (K=4) | 1.76 | 3.03 | 0.98 |
| W1 n_t=128 per-pair | 0.596 | 1.525 | 2.196 | 2.572 | 2.681 | 2.335 | 2.598 | 2.598 | 1.761 (K=4) | 1.76 | 4.50 | 0.97 |
| W1 n_t=128 gathered | 1.305 | 1.537 | 2.272 | 2.595 | 2.698 | 2.356 | 2.612 | 2.647 | 1.769 (K=4) | 1.76 | 2.07 | 0.98 |
| W2 N=1000 t≠s | 1.214 | 1.537 | 2.264 | 2.594 | 2.705 | 2.357 | 2.614 | 2.654 | 1.770 (K=4) | 1.76 | 2.23 | 0.98 |
| W2 N=1000 t=s | 0.771 | 1.537 | 2.267 | 2.596 | 2.703 | 2.358 | 2.615 | 2.653 | 1.769 (K=4) | 1.76 | 3.51 | 0.98 |
| W2 N=10000 t≠s | 1.288 | 1.533 | 2.275 | 2.599 | 2.706 | 2.348 | 2.611 | 2.647 | 1.771 (K=4) | 1.77 | 2.10 | 0.98 |
| W2 N=10000 t=s | 1.180 | 1.532 | 2.272 | 2.595 | 2.702 | 2.348 | 2.609 | 2.594 | 1.770 (K=4) | 1.76 | 2.29 | 0.97 |

Targets in lanes, geometric mean over the W1 cells (Gpairs/s): K=1: 1.997, K=2: 2.452, K=4: 2.610.

### f64, potential and gradient

| cell | reference | green-kernels | TIL K=1 | K=2 | K=4 | SIL T=1 | T=2 | T=4 | TIL gk | TIL/gk | TIL/ref | SIL/TIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | 0.663 | 0.990 | 0.883 | 1.120 | 1.068 | 1.274 | 1.366 | 1.308 | 0.891 (K=2) | 1.13 | 1.69 | 1.22 |
| W1 n_t=8 gathered | 0.470 | 1.154 | 1.418 | 1.366 | 1.429 | 1.432 | 1.580 | 1.594 | 1.186 (K=4) | 1.24 | 3.04 | 1.12 |
| W1 n_t=16 per-pair | 0.634 | 1.068 | 1.301 | 1.363 | 1.317 | 1.362 | 1.483 | 1.473 | 1.053 (K=2) | 1.28 | 2.15 | 1.09 |
| W1 n_t=16 gathered | 0.475 | 1.174 | 1.500 | 1.570 | 1.621 | 1.456 | 1.607 | 1.621 | 1.216 (K=4) | 1.38 | 3.41 | 1.00 |
| W1 n_t=24 per-pair | 0.583 | 1.109 | 1.331 | 1.441 | 1.414 | 1.387 | 1.524 | 1.531 | 1.100 (K=2) | 1.30 | 2.47 | 1.06 |
| W1 n_t=24 gathered | 0.478 | 1.177 | 1.506 | 1.576 | 1.631 | 1.461 | 1.611 | 1.627 | 1.221 (K=4) | 1.39 | 3.41 | 1.00 |
| W1 n_t=32 per-pair | 0.550 | 1.125 | 1.391 | 1.482 | 1.468 | 1.411 | 1.545 | 1.556 | 1.128 (K=2) | 1.32 | 2.69 | 1.05 |
| W1 n_t=32 gathered | 0.479 | 1.179 | 1.508 | 1.578 | 1.633 | 1.463 | 1.612 | 1.628 | 1.223 (K=4) | 1.39 | 3.41 | 1.00 |
| W1 n_t=64 per-pair | 0.509 | 1.156 | 1.455 | 1.537 | 1.551 | 1.440 | 1.581 | 1.594 | 1.177 (K=4) | 1.34 | 3.05 | 1.03 |
| W1 n_t=64 gathered | 0.483 | 1.181 | 1.513 | 1.583 | 1.638 | 1.464 | 1.616 | 1.630 | 1.226 (K=4) | 1.39 | 3.39 | 0.99 |
| W1 n_t=128 per-pair | 0.492 | 1.169 | 1.487 | 1.563 | 1.595 | 1.453 | 1.600 | 1.614 | 1.202 (K=4) | 1.36 | 3.24 | 1.01 |
| W1 n_t=128 gathered | 0.492 | 1.181 | 1.515 | 1.584 | 1.639 | 1.465 | 1.616 | 1.631 | 1.226 (K=4) | 1.39 | 3.33 | 1.00 |
| W2 N=1000 t≠s | 0.482 | 1.181 | 1.512 | 1.582 | 1.636 | 1.465 | 1.615 | 1.631 | 1.225 (K=4) | 1.38 | 3.39 | 1.00 |
| W2 N=1000 t=s | 0.482 | 1.181 | 1.513 | 1.583 | 1.636 | 1.464 | 1.615 | 1.632 | 1.226 (K=4) | 1.39 | 3.39 | 1.00 |
| W2 N=10000 t≠s | 0.513 | 1.179 | 1.514 | 1.584 | 1.640 | 1.464 | 1.616 | 1.632 | 1.228 (K=4) | 1.39 | 3.20 | 1.00 |
| W2 N=10000 t=s | 0.534 | 1.178 | 1.514 | 1.584 | 1.642 | 1.463 | 1.616 | 1.633 | 1.228 (K=4) | 1.39 | 3.07 | 0.99 |

Targets in lanes, geometric mean over the W1 cells (Gpairs/s): K=1: 1.388, K=2: 1.474, K=4: 1.490.

## Loop order: the decision rule of design §4.1

Ratio best SIL / best TIL (each over its K or T, best inverse square root), geometric mean over the W1 cells (both forms, all n_t). Sources in lanes is recommended only if ≥ 1.15 in both precisions.

| precision | potential | potential and gradient | both outputs |
| --- | --- | --- | --- |
| f32 | 1.042 | 1.000 | 1.021 |
| f64 | 1.003 | 1.045 | 1.024 |

## Fraction of the design §4.6 model (targets in lanes, chosen K)

Model: 4 FP pipes × W lanes / (FP ops per pair and lane, design §4.2 with this inverse square root). Corrected: also limited by the measured throughput of the estimate and of FRSQRTS, and for sqrt+div by the divider (1/rate(FSQRT) + 1/rate(FDIV) cycles per vector; the two share one unit), from the single-instruction table. Rows for the chosen formulation and for green-kernels' (gk), at the chosen K. Pairs per cycle at 4.05 GHz.

| precision | output | K | rsqrt | ops/pair/lane | model pairs/cycle | corrected | cell | measured pairs/cycle | of model | of corrected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.455 | W1 n_t=8 gathered | 1.302 | 90% | 90% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=8 gathered | 0.953 | 95% | 95% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.455 | W1 n_t=16 gathered | 1.303 | 90% | 90% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=16 gathered | 0.960 | 96% | 96% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.455 | W1 n_t=24 gathered | 1.308 | 90% | 90% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=24 gathered | 0.963 | 96% | 96% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.455 | W1 n_t=32 gathered | 1.311 | 90% | 90% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=32 gathered | 0.964 | 96% | 96% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.455 | W1 n_t=64 gathered | 1.314 | 90% | 90% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=64 gathered | 0.966 | 97% | 97% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.455 | W1 n_t=128 gathered | 1.316 | 90% | 90% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=128 gathered | 0.967 | 97% | 97% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.455 | W2 N=1000 t≠s | 1.315 | 90% | 90% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W2 N=1000 t≠s | 0.967 | 97% | 97% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.455 | W2 N=10000 t≠s | 1.315 | 90% | 90% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W2 N=10000 t≠s | 0.954 | 95% | 95% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=8 gathered | 0.785 | 83% | 83% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=8 gathered | 0.659 | 91% | 91% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=16 gathered | 0.800 | 85% | 85% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=16 gathered | 0.669 | 92% | 92% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=24 gathered | 0.803 | 85% | 85% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=24 gathered | 0.672 | 92% | 92% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=32 gathered | 0.805 | 86% | 86% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=32 gathered | 0.673 | 93% | 93% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=64 gathered | 0.808 | 86% | 86% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=64 gathered | 0.675 | 93% | 93% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=128 gathered | 0.810 | 86% | 86% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=128 gathered | 0.676 | 93% | 93% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W2 N=1000 t≠s | 0.807 | 86% | 86% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W2 N=1000 t≠s | 0.675 | 93% | 93% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W2 N=10000 t≠s | 0.810 | 86% | 86% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W2 N=10000 t≠s | 0.676 | 93% | 93% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.727 | W1 n_t=8 gathered | 0.643 | 88% | 88% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=8 gathered | 0.421 | 100% | 100% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.727 | W1 n_t=16 gathered | 0.661 | 91% | 91% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=16 gathered | 0.434 | 103% | 103% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.727 | W1 n_t=24 gathered | 0.663 | 91% | 91% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=24 gathered | 0.436 | 103% | 103% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.727 | W1 n_t=32 gathered | 0.665 | 91% | 91% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=32 gathered | 0.436 | 104% | 104% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.727 | W1 n_t=64 gathered | 0.666 | 92% | 92% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=64 gathered | 0.437 | 104% | 104% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.727 | W1 n_t=128 gathered | 0.666 | 92% | 92% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=128 gathered | 0.437 | 104% | 104% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.727 | W2 N=1000 t≠s | 0.668 | 92% | 92% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W2 N=1000 t≠s | 0.437 | 104% | 104% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.727 | W2 N=10000 t≠s | 0.668 | 92% | 92% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W2 N=10000 t≠s | 0.437 | 104% | 104% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=8 gathered | 0.353 | 75% | 75% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=8 gathered | 0.293 | 92% | 92% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=16 gathered | 0.400 | 85% | 85% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=16 gathered | 0.300 | 94% | 94% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=24 gathered | 0.403 | 86% | 86% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=24 gathered | 0.301 | 94% | 94% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=32 gathered | 0.403 | 86% | 86% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=32 gathered | 0.302 | 94% | 94% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=64 gathered | 0.405 | 86% | 86% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=64 gathered | 0.303 | 95% | 95% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=128 gathered | 0.405 | 86% | 86% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=128 gathered | 0.303 | 95% | 95% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W2 N=1000 t≠s | 0.404 | 86% | 86% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W2 N=1000 t≠s | 0.303 | 95% | 95% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W2 N=10000 t≠s | 0.405 | 86% | 86% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W2 N=10000 t≠s | 0.303 | 95% | 95% |

## Relaxed f64 levels (targets in lanes, chosen K, W1 gathered)

| output | n_t | full: rsqrt, Gpairs/s, max err | relaxed S2: Gpairs/s (gain), max err | relaxed P4: Gpairs/s (gain), max err |
| --- | --- | --- | --- | --- |
| φ | 8 | sqrt+div, 2.604, 1.2e-16 | 2.088 (-20%), 6.5e-12 | 1.639 (-37%), 6.1e-14 |
| φ | 16 | sqrt+div, 2.679, 1.2e-16 | 2.096 (-22%), 5.8e-12 | 1.644 (-39%), 4.7e-14 |
| φ | 24 | sqrt+div, 2.685, 1.6e-16 | 2.101 (-22%), 5.4e-12 | 1.647 (-39%), 2.8e-14 |
| φ | 32 | sqrt+div, 2.694, 1.6e-16 | 2.105 (-22%), 4.2e-12 | 1.649 (-39%), 2.4e-14 |
| φ | 64 | sqrt+div, 2.698, 2.4e-16 | 2.108 (-22%), 3.2e-12 | 1.651 (-39%), 2.1e-14 |
| φ | 128 | sqrt+div, 2.698, 2.2e-16 | 2.108 (-22%), 2.5e-12 | 1.651 (-39%), 1.8e-14 |
| φ, ∇φ | 8 | sqrt+div, 1.429, 4.3e-16 | 1.369 (-4%), 4.0e-11 | 1.201 (-16%), 3.1e-13 |
| φ, ∇φ | 16 | sqrt+div, 1.621, 2.0e-16 | 1.386 (-15%), 4.3e-11 | 1.212 (-25%), 3.2e-13 |
| φ, ∇φ | 24 | sqrt+div, 1.631, 3.0e-16 | 1.391 (-15%), 3.1e-11 | 1.217 (-25%), 2.1e-13 |
| φ, ∇φ | 32 | sqrt+div, 1.633, 3.9e-16 | 1.394 (-15%), 7.4e-11 | 1.218 (-25%), 4.2e-13 |
| φ, ∇φ | 64 | sqrt+div, 1.638, 1.9e-15 | 1.396 (-15%), 2.9e-11 | 1.221 (-25%), 1.4e-13 |
| φ, ∇φ | 128 | sqrt+div, 1.639, 9.5e-16 | 1.399 (-15%), 5.5e-11 | 1.222 (-25%), 4.0e-13 |


# P2P SIMD spike: raw results

| item | value |
| --- | --- |
| CPU | Apple M3 Max |
| cores | 16 physical, 16 logical |
| target | aarch64-macos, release build |
| toolchain | rustc 1.98.0 (88d9e12ae 2026-08-18) |
| ISAs with prototypes | scalar, neon |
| green-kernels | commit `7d757c579f8633d58163cd5b2d011a9468541f17`, pulp backend `Neon(Neon { neon: Neon })` |
| thread variables | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, RAYON_NUM_THREADS=1 |
| clock for cycles | 4.05 GHz assumed (checked by the FMA latency below) |
| mode | full |
| timing | median of 15 batches of ≥ 20 ms, one thread |

## Single instructions

Latency: one dependent chain. Throughput: 16 independent chains. Cycles at the assumed clock.

| ISA | precision | instruction | latency (cycles) | throughput (per cycle) |
| --- | --- | --- | --- | --- |
| neon | f32 | fma (FMLA / vfmadd) | 3.66 | 3.93 |
| neon | f32 | mul (FMUL / vmulp) | 3.95 | 3.76 |
| neon | f32 | add (FADD / vaddp) | 2.69 | 3.95 |
| neon | f32 | estimate (FRSQRTE / vrsqrtps) | 2.98 | 1.00 |
| neon | f32 | step (FRSQRTS / fnmadd) | 3.94 | 3.79 |
| neon | f32 | sqrt (FSQRT / vsqrtp) | 9.93 | 0.50 |
| neon | f32 | div (FDIV / vdivp) | 7.95 | 1.00 |
| neon | f64 | fma (FMLA / vfmadd) | 3.95 | 3.96 |
| neon | f64 | mul (FMUL / vmulp) | 3.96 | 3.94 |
| neon | f64 | add (FADD / vaddp) | 2.70 | 3.95 |
| neon | f64 | estimate (FRSQRTE / vrsqrtps) | 2.99 | 1.00 |
| neon | f64 | step (FRSQRTS / fnmadd) | 3.94 | 3.98 |
| neon | f64 | sqrt (FSQRT / vsqrtp) | 12.92 | 0.50 |
| neon | f64 | div (FDIV / vdivp) | 9.94 | 1.00 |

## Inverse square root: measured

Error: max |relative error| in u_T; f32 exhaustively over [1, 4), f64 on 10⁷ log-uniform samples over [2⁻¹⁰⁸, 2⁷]; plus the powers of two of the domain, their neighbours and the ends. Edge inputs show the raw result before the r² = 0 mask. Throughput on a 4,096-element buffer in L1; latency in a dependent chain.

| ISA | precision | candidate | FP ops | max error (u_T) | at x | ≤ 4 u_T | ns/elem | cycles/vector | latency (cycles) | edge inputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neon | f32 | est | 2 | 54975.965 | 2.093750e0 | no | 0.072 | 1.16 | 3.0 | 0 → +inf; min sub → 2.664e22; 1e-40 → 1.000e20 (rel 1.6e-4) |
| neon | f32 | est+N1 | 5 | 271.160 | 2.093750e0 | no | 0.103 | 1.67 | 14.9 | 0 → NaN; min sub → NaN; 1e-40 → +inf (rel -inf) |
| neon | f32 | est+N2 | 8 | 1.498 | 3.998046e0 | yes | 0.172 | 2.78 | 26.9 | 0 → NaN; min sub → NaN; 1e-40 → NaN (rel NaN) |
| neon | f32 | est+N3 | 11 | 1.498 | 3.996619e0 | yes | 0.239 | 3.88 | 39.0 | 0 → NaN; min sub → NaN; 1e-40 → NaN (rel NaN) |
| neon | f32 | est+S1 | 4 | 271.510 | 2.093746e0 | no | 0.076 | 1.22 | 15.0 | 0 → +inf; min sub → +inf; 1e-40 → +inf (rel -inf) |
| neon | f32 | est+S2 | 7 | 2.433 | 3.922365e0 | yes | 0.131 | 2.11 | 26.9 | 0 → +inf; min sub → +inf; 1e-40 → +inf (rel inf) |
| neon | f32 | est+S3 | 10 | 2.096 | 1.771779e0 | yes | 0.179 | 2.90 | 39.0 | 0 → +inf; min sub → +inf; 1e-40 → +inf (rel -inf) |
| neon | f32 | est+P2 | 6 | 2.195 | 2.093741e0 | yes | 0.137 | 2.22 | 19.2 | 0 → NaN; min sub → +inf; 1e-40 → +inf (rel inf) |
| neon | f32 | est+P3 | 7 | 1.000 | 3.985106e0 | yes | 0.176 | 2.85 | 23.0 | 0 → NaN; min sub → +inf; 1e-40 → +inf (rel -inf) |
| neon | f32 | est+P4 | 8 | 1.000 | 4.000000e0 | yes | 0.206 | 3.33 | 27.0 | 0 → NaN; min sub → +inf; 1e-40 → +inf (rel inf) |
| neon | f32 | sqrt+div | 2 | 1.500 | 4.000000e0 | yes | 0.188 | 3.05 | 17.9 | 0 → +inf; min sub → 2.671e22; 1e-40 → 1.000e20 (rel 2.7e-6) |

| ISA | precision | candidate | FP ops | max error (u_T) | at x | ≤ 4 u_T | ns/elem | cycles/vector | latency (cycles) | edge inputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neon | f64 | est | 2 | 29515321864122.965 | 4.874891e-10 | no | 0.158 | 1.28 | 3.0 | 0 → +inf; min sub → 4.490e161; 1e-310 → 9.993e154 (rel -7.0e-4) |
| neon | f64 | est+N2 | 8 | 3512695.315 | 4.874891e-10 | no | 0.342 | 2.77 | 26.9 | 0 → NaN; min sub → NaN; 1e-310 → NaN (rel NaN) |
| neon | f64 | est+N3 | 11 | 1.495 | 3.630836e-12 | yes | 0.477 | 3.86 | 38.9 | 0 → NaN; min sub → NaN; 1e-310 → NaN (rel NaN) |
| neon | f64 | est+N4 | 14 | 1.487 | 1.483026e-8 | yes | 0.611 | 4.95 | 51.2 | 0 → NaN; min sub → NaN; 1e-310 → NaN (rel NaN) |
| neon | f64 | est+S2 | 7 | 3512695.315 | 4.874891e-10 | no | 0.259 | 2.09 | 26.9 | 0 → +inf; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+S3 | 10 | 2.452 | 9.817949e-1 | yes | 0.356 | 2.88 | 38.9 | 0 → +inf; min sub → +inf; 1e-310 → +inf (rel -inf) |
| neon | f64 | est+S4 | 13 | 2.094 | 1.571777e-15 | yes | 0.469 | 3.80 | 51.2 | 0 → +inf; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+P4 | 8 | 26946.656 | 4.874891e-10 | no | 0.407 | 3.30 | 27.0 | 0 → NaN; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+P6 | 10 | 1.702 | 6.606679e-30 | yes | 0.490 | 3.97 | 35.0 | 0 → NaN; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+P7 | 11 | 1.000 | 9.764367e-4 | yes | 0.515 | 4.17 | 38.9 | 0 → NaN; min sub → +inf; 1e-310 → +inf (rel -inf) |
| neon | f64 | est+P8 | 12 | 1.000 | 9.764367e-4 | yes | 0.561 | 4.54 | 42.9 | 0 → NaN; min sub → +inf; 1e-310 → +inf (rel inf) |
| neon | f64 | est+N1+P2 | 10 | 95.144 | 1.996754e-6 | no | 0.445 | 3.60 | 31.1 | 0 → NaN; min sub → NaN; 1e-310 → +inf (rel -inf) |
| neon | f64 | est+N1+P3 | 11 | 1.498 | 3.469212e-18 | yes | 0.510 | 4.13 | 35.0 | 0 → NaN; min sub → NaN; 1e-310 → NaN (rel NaN) |
| neon | f64 | sqrt+div | 2 | 1.500 | 1.232595e-32 | yes | 0.380 | 3.08 | 22.9 | 0 → +inf; min sub → 4.499e161; 1e-310 → 1.000e155 (rel 0.0e0) |

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
| W1 n_t=8 per-pair | reference | scalar |  |  | sqrt+div | 0.798 | 0.197 | 8.8e-8 |  | 4.5e-7 |  | pass |  |
| W1 n_t=8 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.966 | 0.485 | 3.0e-8 |  | 3.9e-7 |  | pass |  |
| W1 n_t=8 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.057 | 0.508 | 7.7e-8 |  | 5.9e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.454 | 0.606 | 7.7e-8 |  | 5.9e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 1.062 | 0.262 | 7.7e-8 |  | 5.9e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 1.890 | 0.467 | 9.9e-8 |  | 6.3e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 2.089 | 0.516 | 9.9e-8 |  | 6.3e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 0.893 | 0.221 | 9.9e-8 |  | 6.3e-7 |  | pass | yes |
| W1 n_t=8 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.682 | 0.662 | 2.2e-8 |  | 4.5e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.489 | 0.615 | 2.2e-8 |  | 4.5e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.376 | 0.587 | 2.2e-8 |  | 4.5e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.254 | 0.557 | 2.1e-8 |  | 5.6e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.171 | 0.536 | 2.1e-8 |  | 5.6e-7 |  | pass | no |
| W1 n_t=8 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.153 | 0.532 | 2.1e-8 |  | 5.6e-7 |  | pass | no |
| W1 n_t=8 gathered | reference | scalar |  |  | sqrt+div | 0.371 | 0.092 | 8.8e-8 |  | 4.5e-7 |  | pass |  |
| W1 n_t=8 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.906 | 0.717 | 3.5e-8 |  | 6.1e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.398 | 0.839 | 7.7e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 3.741 | 0.924 | 7.7e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.033 | 0.502 | 7.7e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 2.908 | 0.718 | 9.9e-8 |  | 6.3e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.051 | 0.753 | 9.9e-8 |  | 6.3e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 1.624 | 0.401 | 9.9e-8 |  | 6.3e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.469 | 0.857 | 3.1e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.919 | 0.968 | 3.1e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.220 | 1.042 | 3.1e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.898 | 0.716 | 2.6e-8 |  | 5.5e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.200 | 0.790 | 2.6e-8 |  | 5.5e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.361 | 0.830 | 2.6e-8 |  | 5.5e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.674 | 0.660 | 8.8e-8 |  | 7.8e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.253 | 0.556 | 8.8e-8 |  | 7.1e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.052 | 0.754 | 9.9e-8 |  | 6.3e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.578 | 0.637 | 8.8e-8 |  | 7.2e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.012 | 0.744 | 7.7e-8 |  | 6.4e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.729 | 0.674 | 8.8e-8 |  | 7.2e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.507 | 0.619 | 8.8e-8 |  | 7.2e-7 |  | pass |  |
| W1 n_t=8 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 3.741 | 0.924 | 7.7e-8 |  | 5.9e-7 |  | pass |  |
| W1 n_t=16 per-pair | reference | scalar |  |  | sqrt+div | 0.658 | 0.163 | 8.7e-8 |  | 9.4e-7 |  | pass |  |
| W1 n_t=16 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.474 | 0.611 | 1.3e-8 |  | 2.9e-7 |  | pass |  |
| W1 n_t=16 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 2.788 | 0.688 | 7.6e-8 |  | 8.2e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 3.220 | 0.795 | 7.6e-8 |  | 8.2e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.699 | 0.666 | 7.6e-8 |  | 8.2e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 2.413 | 0.596 | 5.9e-8 |  | 7.5e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 2.668 | 0.659 | 5.9e-8 |  | 7.5e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 2.290 | 0.565 | 5.9e-8 |  | 7.5e-7 |  | pass | yes |
| W1 n_t=16 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.129 | 0.773 | 1.3e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.343 | 0.826 | 1.3e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.317 | 0.819 | 1.3e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.585 | 0.638 | 1.1e-8 |  | 2.9e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.740 | 0.676 | 1.1e-8 |  | 2.9e-7 |  | pass | no |
| W1 n_t=16 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.780 | 0.686 | 1.1e-8 |  | 2.9e-7 |  | pass | no |
| W1 n_t=16 gathered | reference | scalar |  |  | sqrt+div | 0.353 | 0.087 | 8.7e-8 |  | 9.4e-7 |  | pass |  |
| W1 n_t=16 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.949 | 0.728 | 3.8e-8 |  | 6.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.532 | 0.872 | 7.6e-8 |  | 8.2e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.012 | 0.991 | 7.6e-8 |  | 8.2e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.252 | 1.050 | 7.6e-8 |  | 8.2e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.023 | 0.746 | 5.9e-8 |  | 7.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.226 | 0.797 | 5.9e-8 |  | 7.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.324 | 0.821 | 5.9e-8 |  | 7.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.510 | 0.867 | 3.0e-8 |  | 5.7e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.007 | 0.989 | 3.0e-8 |  | 5.7e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.333 | 1.070 | 3.0e-8 |  | 5.7e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.985 | 0.737 | 3.6e-8 |  | 6.4e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.283 | 0.811 | 3.6e-8 |  | 6.4e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.424 | 0.845 | 3.6e-8 |  | 6.4e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.760 | 0.681 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.308 | 0.570 | 6.5e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.227 | 0.797 | 5.9e-8 |  | 7.5e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.679 | 0.661 | 7.0e-8 |  | 7.8e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.096 | 0.764 | 7.0e-8 |  | 7.6e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.799 | 0.691 | 7.6e-8 |  | 8.0e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.554 | 0.631 | 7.6e-8 |  | 8.0e-7 |  | pass |  |
| W1 n_t=16 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 4.009 | 0.990 | 7.6e-8 |  | 8.2e-7 |  | pass |  |
| W1 n_t=24 per-pair | reference | scalar |  |  | sqrt+div | 0.567 | 0.140 | 8.7e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.582 | 0.638 | 2.1e-8 |  | 2.2e-7 |  | pass |  |
| W1 n_t=24 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.032 | 0.749 | 8.1e-8 |  | 7.0e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 3.547 | 0.876 | 8.1e-8 |  | 7.0e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.391 | 0.590 | 8.1e-8 |  | 7.0e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 2.569 | 0.634 | 8.0e-8 |  | 6.8e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 2.891 | 0.714 | 8.0e-8 |  | 6.8e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 1.952 | 0.482 | 8.0e-8 |  | 6.8e-7 |  | pass | yes |
| W1 n_t=24 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.210 | 0.793 | 2.6e-8 |  | 1.8e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.620 | 0.894 | 2.6e-8 |  | 1.8e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.735 | 0.922 | 2.6e-8 |  | 1.8e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.676 | 0.661 | 2.4e-8 |  | 1.6e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.938 | 0.726 | 2.4e-8 |  | 1.6e-7 |  | pass | no |
| W1 n_t=24 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.021 | 0.746 | 2.4e-8 |  | 1.6e-7 |  | pass | no |
| W1 n_t=24 gathered | reference | scalar |  |  | sqrt+div | 0.348 | 0.086 | 8.7e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.016 | 0.745 | 3.5e-8 |  | 3.1e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.563 | 0.880 | 8.1e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.039 | 0.997 | 8.1e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 3.224 | 0.796 | 8.1e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.056 | 0.755 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.247 | 0.802 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 2.509 | 0.619 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.537 | 0.873 | 3.5e-8 |  | 3.5e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.049 | 1.000 | 3.5e-8 |  | 3.5e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.342 | 1.072 | 3.5e-8 |  | 3.5e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.978 | 0.735 | 2.7e-8 |  | 3.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.277 | 0.809 | 2.7e-8 |  | 3.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.437 | 0.849 | 2.7e-8 |  | 3.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.775 | 0.685 | 8.6e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.316 | 0.572 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.246 | 0.802 | 8.0e-8 |  | 6.8e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.691 | 0.664 | 7.2e-8 |  | 7.1e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.115 | 0.769 | 8.1e-8 |  | 6.6e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.816 | 0.695 | 8.1e-8 |  | 6.7e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.565 | 0.633 | 8.1e-8 |  | 6.7e-7 |  | pass |  |
| W1 n_t=24 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 4.040 | 0.998 | 8.1e-8 |  | 7.0e-7 |  | pass |  |
| W1 n_t=32 per-pair | reference | scalar |  |  | sqrt+div | 0.528 | 0.130 | 5.6e-8 |  | 9.1e-7 |  | pass |  |
| W1 n_t=32 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.710 | 0.669 | 1.9e-8 |  | 3.4e-7 |  | pass |  |
| W1 n_t=32 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.211 | 0.793 | 7.0e-8 |  | 8.1e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 3.628 | 0.896 | 7.0e-8 |  | 8.1e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 3.378 | 0.834 | 7.0e-8 |  | 8.1e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 2.704 | 0.668 | 6.1e-8 |  | 9.2e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 2.954 | 0.729 | 6.1e-8 |  | 9.2e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 2.765 | 0.683 | 6.1e-8 |  | 9.2e-7 |  | pass | yes |
| W1 n_t=32 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.313 | 0.818 | 1.8e-8 |  | 2.8e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.680 | 0.909 | 1.8e-8 |  | 2.8e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 3.932 | 0.971 | 1.8e-8 |  | 2.8e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.773 | 0.685 | 2.8e-8 |  | 3.2e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.005 | 0.742 | 2.8e-8 |  | 3.2e-7 |  | pass | no |
| W1 n_t=32 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.143 | 0.776 | 2.8e-8 |  | 3.2e-7 |  | pass | no |
| W1 n_t=32 gathered | reference | scalar |  |  | sqrt+div | 0.345 | 0.085 | 5.6e-8 |  | 9.1e-7 |  | pass |  |
| W1 n_t=32 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.035 | 0.749 | 3.2e-8 |  | 5.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.578 | 0.883 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.056 | 1.002 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.310 | 1.064 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.067 | 0.757 | 6.1e-8 |  | 9.2e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.255 | 0.804 | 6.1e-8 |  | 9.2e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.354 | 0.828 | 6.1e-8 |  | 9.2e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.553 | 0.877 | 3.0e-8 |  | 4.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.070 | 1.005 | 3.0e-8 |  | 4.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.358 | 1.076 | 3.0e-8 |  | 4.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.996 | 0.740 | 3.0e-8 |  | 4.8e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.292 | 0.813 | 3.0e-8 |  | 4.8e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.443 | 0.850 | 3.0e-8 |  | 4.8e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.782 | 0.687 | 5.3e-8 |  | 8.3e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.321 | 0.573 | 5.8e-8 |  | 7.8e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.253 | 0.803 | 6.1e-8 |  | 9.2e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.698 | 0.666 | 6.6e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.121 | 0.771 | 5.8e-8 |  | 8.6e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.800 | 0.691 | 5.8e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.570 | 0.635 | 5.8e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=32 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 4.058 | 1.002 | 7.0e-8 |  | 8.1e-7 |  | pass |  |
| W1 n_t=64 per-pair | reference | scalar |  |  | sqrt+div | 0.483 | 0.119 | 1.3e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.865 | 0.707 | 2.1e-8 |  | 3.3e-7 |  | pass |  |
| W1 n_t=64 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.455 | 0.853 | 1.2e-7 |  | 1.1e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 3.878 | 0.958 | 1.2e-7 |  | 1.1e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 3.806 | 0.940 | 1.2e-7 |  | 1.1e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 2.890 | 0.714 | 1.2e-7 |  | 1.2e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.039 | 0.750 | 1.2e-7 |  | 1.2e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.049 | 0.753 | 1.2e-7 |  | 1.2e-6 |  | pass | yes |
| W1 n_t=64 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.459 | 0.854 | 1.7e-8 |  | 3.7e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 3.919 | 0.968 | 1.7e-8 |  | 3.7e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.161 | 1.027 | 1.7e-8 |  | 3.7e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.900 | 0.716 | 1.6e-8 |  | 3.5e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.178 | 0.785 | 1.6e-8 |  | 3.5e-7 |  | pass | no |
| W1 n_t=64 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.306 | 0.816 | 1.6e-8 |  | 3.5e-7 |  | pass | no |
| W1 n_t=64 gathered | reference | scalar |  |  | sqrt+div | 0.341 | 0.084 | 1.3e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.056 | 0.755 | 4.0e-8 |  | 8.7e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.601 | 0.889 | 1.2e-7 |  | 1.1e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.074 | 1.006 | 1.2e-7 |  | 1.1e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.330 | 1.069 | 1.2e-7 |  | 1.1e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.086 | 0.762 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.266 | 0.806 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.368 | 0.832 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.574 | 0.883 | 4.9e-8 |  | 8.8e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.084 | 1.008 | 4.9e-8 |  | 8.8e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.388 | 1.083 | 4.9e-8 |  | 8.8e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.018 | 0.745 | 4.9e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.311 | 0.817 | 4.9e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.462 | 0.855 | 4.9e-8 |  | 8.4e-7 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.792 | 0.689 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.322 | 0.573 | 1.2e-7 |  | 1.3e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.267 | 0.807 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.707 | 0.668 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.134 | 0.774 | 1.2e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.829 | 0.699 | 1.1e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.577 | 0.636 | 1.1e-7 |  | 1.2e-6 |  | pass |  |
| W1 n_t=64 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 4.076 | 1.006 | 1.2e-7 |  | 1.1e-6 |  | pass |  |
| W1 n_t=128 per-pair | reference | scalar |  |  | sqrt+div | 0.392 | 0.097 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.913 | 0.719 | 1.6e-8 |  | 3.5e-7 |  | pass |  |
| W1 n_t=128 per-pair | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.427 | 0.846 | 1.1e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 3.918 | 0.967 | 1.1e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.071 | 1.005 | 1.1e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 2.913 | 0.719 | 1.0e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.156 | 0.779 | 1.0e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.208 | 0.792 | 1.0e-7 |  | 1.6e-6 |  | pass | yes |
| W1 n_t=128 per-pair | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.527 | 0.871 | 1.2e-8 |  | 3.3e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.023 | 0.993 | 1.2e-8 |  | 3.3e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.288 | 1.059 | 1.2e-8 |  | 3.3e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.950 | 0.728 | 1.2e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.258 | 0.804 | 1.2e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=128 per-pair | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.399 | 0.839 | 1.2e-8 |  | 3.4e-7 |  | pass | no |
| W1 n_t=128 gathered | reference | scalar |  |  | sqrt+div | 0.339 | 0.084 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.072 | 0.759 | 4.5e-8 |  | 1.3e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.615 | 0.892 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.087 | 1.009 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.353 | 1.075 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.095 | 0.764 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.274 | 0.808 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.378 | 0.834 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.590 | 0.886 | 4.8e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.114 | 1.016 | 4.8e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.405 | 1.088 | 4.8e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.031 | 0.748 | 4.6e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.320 | 0.820 | 4.6e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.472 | 0.857 | 4.6e-8 |  | 1.4e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 2.796 | 0.690 | 1.2e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 2.331 | 0.576 | 1.2e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 3.275 | 0.809 | 1.0e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 2.711 | 0.669 | 1.1e-7 |  | 1.7e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 3.142 | 0.776 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.836 | 0.700 | 1.2e-7 |  | 1.7e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 2.583 | 0.638 | 1.2e-7 |  | 1.7e-6 |  | pass |  |
| W1 n_t=128 gathered | f32_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 4.088 | 1.009 | 1.1e-7 |  | 1.6e-6 |  | pass |  |
| W2 N=1000 t≠s | reference | scalar |  |  | sqrt+div | 0.893 | 0.221 | 1.1e-7 |  | 6.2e-7 |  | pass |  |
| W2 N=1000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.051 | 0.753 | 5.9e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.594 | 0.887 | 1.2e-7 |  | 6.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.074 | 1.006 | 1.2e-7 |  | 6.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.287 | 1.059 | 1.2e-7 |  | 6.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.080 | 0.761 | 1.2e-7 |  | 6.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.262 | 0.805 | 1.2e-7 |  | 6.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.332 | 0.823 | 1.2e-7 |  | 6.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.568 | 0.881 | 5.7e-8 |  | 3.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.093 | 1.011 | 5.7e-8 |  | 3.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.385 | 1.083 | 5.7e-8 |  | 3.2e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.010 | 0.743 | 7.0e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.303 | 0.816 | 7.0e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t≠s | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.436 | 0.849 | 7.0e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t=s | reference | scalar |  |  | sqrt+div | 0.344 | 0.085 | 1.2e-7 |  | 6.6e-7 |  | pass |  |
| W2 N=1000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 3.049 | 0.753 | 5.7e-8 |  | 3.4e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.593 | 0.887 | 1.3e-7 |  | 6.6e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.068 | 1.004 | 1.3e-7 |  | 6.6e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.280 | 1.057 | 1.3e-7 |  | 6.6e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.082 | 0.761 | 1.5e-7 |  | 6.5e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.265 | 0.806 | 1.5e-7 |  | 6.5e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.321 | 0.820 | 1.5e-7 |  | 6.5e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.571 | 0.882 | 4.8e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.095 | 1.011 | 4.8e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.383 | 1.082 | 4.8e-8 |  | 3.3e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 3.010 | 0.743 | 5.6e-8 |  | 3.4e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.304 | 0.816 | 5.6e-8 |  | 3.4e-7 |  | pass |  |
| W2 N=1000 t=s | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.459 | 0.854 | 5.6e-8 |  | 3.4e-7 |  | pass |  |
| W2 N=10000 t≠s | reference | scalar |  |  | sqrt+div | 0.337 | 0.083 | 1.5e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.968 | 0.733 | 8.2e-8 |  | 8.1e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.570 | 0.882 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.062 | 1.003 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.326 | 1.068 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.047 | 0.752 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.246 | 0.801 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.370 | 0.832 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.470 | 0.857 | 9.4e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.041 | 0.998 | 9.4e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.336 | 1.071 | 9.4e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.917 | 0.720 | 8.7e-8 |  | 8.1e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.235 | 0.799 | 8.7e-8 |  | 8.1e-7 |  | pass |  |
| W2 N=10000 t≠s | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.402 | 0.840 | 8.7e-8 |  | 8.1e-7 |  | pass |  |
| W2 N=10000 t=s | reference | scalar |  |  | sqrt+div | 0.522 | 0.129 | 1.6e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.973 | 0.734 | 7.2e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 3.556 | 0.878 | 1.5e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 4.061 | 1.003 | 1.5e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 4.323 | 1.067 | 1.5e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k1_gk | neon | TIL | 1 | est+S2 | 3.047 | 0.752 | 1.7e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k2_gk | neon | TIL | 2 | est+S2 | 3.245 | 0.801 | 1.7e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::til_k4_gk | neon | TIL | 4 | est+S2 | 3.372 | 0.833 | 1.7e-7 |  | 1.9e-6 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 3.472 | 0.857 | 7.1e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 4.042 | 0.998 | 7.1e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 4.344 | 1.073 | 7.1e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.917 | 0.720 | 6.7e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t2_gk | neon | SIL | 2 | est+S2 | 3.255 | 0.804 | 6.7e-8 |  | 8.0e-7 |  | pass |  |
| W2 N=10000 t=s | f32_pot::sil_t4_gk | neon | SIL | 4 | est+S2 | 3.418 | 0.844 | 6.7e-8 |  | 8.0e-7 |  | pass |  |

### f32, potential and gradient

| cell | variant | ISA | order | K/T | rsqrt | Gpairs/s | pairs/cycle | max err φ | max err ∇ | L2 φ | L2 ∇ | check | = gathered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | reference | scalar |  |  | sqrt+div | 0.500 | 0.124 | 8.8e-8 | 1.5e-7 | 4.5e-7 | 3.0e-7 | pass |  |
| W1 n_t=8 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.282 | 0.317 | 3.0e-8 | 2.4e-7 | 3.9e-7 | 4.5e-7 | pass |  |
| W1 n_t=8 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.301 | 0.321 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.410 | 0.348 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 0.753 | 0.186 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 1.235 | 0.305 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 1.223 | 0.302 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 0.667 | 0.165 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass | yes |
| W1 n_t=8 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.662 | 0.410 | 3.1e-8 | 5.6e-8 | 3.4e-7 | 2.1e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.552 | 0.383 | 3.1e-8 | 5.6e-8 | 3.4e-7 | 2.1e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.389 | 0.343 | 3.1e-8 | 5.6e-8 | 3.4e-7 | 2.1e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 1.433 | 0.354 | 2.2e-8 | 7.9e-8 | 4.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 1.361 | 0.336 | 2.2e-8 | 7.9e-8 | 4.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=8 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 1.127 | 0.278 | 2.2e-8 | 7.9e-8 | 4.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=8 gathered | reference | scalar |  |  | sqrt+div | 0.332 | 0.082 | 8.8e-8 | 1.5e-7 | 4.5e-7 | 3.0e-7 | pass |  |
| W1 n_t=8 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.106 | 0.520 | 3.5e-8 | 1.1e-7 | 6.1e-7 | 2.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.308 | 0.570 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.421 | 0.598 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.273 | 0.314 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 1.973 | 0.487 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.059 | 0.509 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 1.068 | 0.264 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.363 | 0.584 | 3.1e-8 | 4.1e-8 | 7.8e-7 | 1.8e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.560 | 0.632 | 3.1e-8 | 4.1e-8 | 7.8e-7 | 1.8e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.572 | 0.635 | 3.1e-8 | 4.1e-8 | 7.8e-7 | 1.8e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.038 | 0.503 | 2.1e-8 | 4.9e-8 | 6.6e-7 | 2.0e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.092 | 0.516 | 2.1e-8 | 4.9e-8 | 6.6e-7 | 2.0e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.088 | 0.516 | 2.1e-8 | 4.9e-8 | 6.6e-7 | 2.0e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 1.894 | 0.468 | 9.9e-8 | 1.3e-7 | 8.1e-7 | 2.8e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.603 | 0.396 | 7.7e-8 | 1.7e-7 | 5.8e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.049 | 0.506 | 7.7e-8 | 1.5e-7 | 3.4e-7 | 3.1e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.727 | 0.426 | 8.8e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.102 | 0.519 | 8.8e-8 | 1.3e-7 | 6.1e-7 | 2.6e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 1.957 | 0.483 | 9.9e-8 | 1.3e-7 | 5.9e-7 | 2.9e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 1.816 | 0.448 | 9.9e-8 | 1.3e-7 | 5.9e-7 | 2.9e-7 | pass |  |
| W1 n_t=8 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.411 | 0.595 | 7.7e-8 | 1.7e-7 | 7.1e-7 | 3.3e-7 | pass |  |
| W1 n_t=16 per-pair | reference | scalar |  |  | sqrt+div | 0.474 | 0.117 | 8.7e-8 | 2.2e-7 | 9.4e-7 | 4.5e-7 | pass |  |
| W1 n_t=16 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.648 | 0.407 | 1.3e-8 | 6.5e-8 | 2.9e-7 | 2.5e-7 | pass |  |
| W1 n_t=16 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.776 | 0.439 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.934 | 0.478 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.826 | 0.451 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 1.624 | 0.401 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 1.626 | 0.402 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 1.593 | 0.393 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass | yes |
| W1 n_t=16 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.023 | 0.500 | 2.1e-8 | 7.2e-8 | 3.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.097 | 0.518 | 2.1e-8 | 7.2e-8 | 3.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.916 | 0.473 | 2.1e-8 | 7.2e-8 | 3.6e-7 | 2.2e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 1.732 | 0.428 | 1.7e-8 | 8.1e-8 | 4.2e-7 | 2.9e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 1.751 | 0.432 | 1.7e-8 | 8.1e-8 | 4.2e-7 | 2.9e-7 | pass | no |
| W1 n_t=16 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 1.538 | 0.380 | 1.7e-8 | 8.1e-8 | 4.2e-7 | 2.9e-7 | pass | no |
| W1 n_t=16 gathered | reference | scalar |  |  | sqrt+div | 0.328 | 0.081 | 8.7e-8 | 2.2e-7 | 9.4e-7 | 4.5e-7 | pass |  |
| W1 n_t=16 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.161 | 0.534 | 3.8e-8 | 6.8e-8 | 6.5e-7 | 2.5e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.409 | 0.595 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.612 | 0.645 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.628 | 0.649 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.067 | 0.510 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.136 | 0.527 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.182 | 0.539 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.389 | 0.590 | 3.6e-8 | 1.1e-7 | 6.2e-7 | 3.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.635 | 0.651 | 3.6e-8 | 1.1e-7 | 6.2e-7 | 3.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.643 | 0.652 | 3.6e-8 | 1.1e-7 | 6.2e-7 | 3.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.061 | 0.509 | 3.5e-8 | 1.2e-7 | 5.6e-7 | 3.7e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.025 | 0.500 | 3.5e-8 | 1.2e-7 | 5.6e-7 | 3.7e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.164 | 0.534 | 3.5e-8 | 1.2e-7 | 5.6e-7 | 3.7e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 1.969 | 0.486 | 6.5e-8 | 2.4e-7 | 8.5e-7 | 4.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.665 | 0.411 | 5.3e-8 | 2.0e-7 | 8.3e-7 | 4.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.135 | 0.527 | 4.2e-8 | 2.2e-7 | 7.1e-7 | 4.6e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.793 | 0.443 | 5.3e-8 | 2.4e-7 | 7.7e-7 | 5.0e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.174 | 0.537 | 6.5e-8 | 2.0e-7 | 8.9e-7 | 4.8e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.016 | 0.498 | 6.5e-8 | 2.4e-7 | 9.3e-7 | 4.9e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 1.869 | 0.462 | 6.5e-8 | 2.4e-7 | 9.3e-7 | 4.9e-7 | pass |  |
| W1 n_t=16 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.611 | 0.645 | 5.3e-8 | 2.4e-7 | 8.4e-7 | 5.0e-7 | pass |  |
| W1 n_t=24 per-pair | reference | scalar |  |  | sqrt+div | 0.417 | 0.103 | 8.7e-8 | 1.7e-7 | 6.8e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.777 | 0.439 | 2.1e-8 | 4.9e-8 | 2.2e-7 | 2.0e-7 | pass |  |
| W1 n_t=24 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.961 | 0.484 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.147 | 0.530 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.570 | 0.388 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 1.762 | 0.435 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 1.789 | 0.442 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 1.340 | 0.331 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass | yes |
| W1 n_t=24 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.125 | 0.525 | 2.4e-8 | 7.5e-8 | 1.7e-7 | 2.1e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.255 | 0.557 | 2.4e-8 | 7.5e-8 | 1.7e-7 | 2.1e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.149 | 0.530 | 2.4e-8 | 7.5e-8 | 1.7e-7 | 2.1e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 1.816 | 0.448 | 2.9e-8 | 4.8e-8 | 2.1e-7 | 2.0e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 1.851 | 0.457 | 2.9e-8 | 4.8e-8 | 2.1e-7 | 2.0e-7 | pass | no |
| W1 n_t=24 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 1.725 | 0.426 | 2.9e-8 | 4.8e-8 | 2.1e-7 | 2.0e-7 | pass | no |
| W1 n_t=24 gathered | reference | scalar |  |  | sqrt+div | 0.323 | 0.080 | 8.7e-8 | 1.7e-7 | 6.8e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.182 | 0.539 | 3.5e-8 | 7.6e-8 | 3.1e-7 | 3.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.433 | 0.601 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.631 | 0.650 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.990 | 0.491 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.083 | 0.514 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.148 | 0.530 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 1.649 | 0.407 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.412 | 0.596 | 3.4e-8 | 9.1e-8 | 3.8e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.642 | 0.652 | 3.4e-8 | 9.1e-8 | 3.8e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.654 | 0.655 | 3.4e-8 | 9.1e-8 | 3.8e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.081 | 0.514 | 3.5e-8 | 6.0e-8 | 3.4e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.150 | 0.531 | 3.5e-8 | 6.0e-8 | 3.4e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.177 | 0.538 | 3.5e-8 | 6.0e-8 | 3.4e-7 | 3.2e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 1.967 | 0.486 | 8.0e-8 | 1.5e-7 | 7.7e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.636 | 0.404 | 8.4e-8 | 1.5e-7 | 7.1e-7 | 5.7e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.147 | 0.530 | 9.2e-8 | 1.4e-7 | 7.2e-7 | 5.6e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.801 | 0.445 | 6.6e-8 | 1.5e-7 | 7.0e-7 | 5.5e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.191 | 0.541 | 6.6e-8 | 1.5e-7 | 6.5e-7 | 5.8e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.027 | 0.500 | 6.6e-8 | 1.5e-7 | 6.6e-7 | 5.7e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 1.878 | 0.464 | 6.6e-8 | 1.5e-7 | 6.6e-7 | 5.7e-7 | pass |  |
| W1 n_t=24 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.632 | 0.650 | 8.1e-8 | 1.3e-7 | 6.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=32 per-pair | reference | scalar |  |  | sqrt+div | 0.386 | 0.095 | 5.6e-8 | 4.6e-7 | 9.1e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.858 | 0.459 | 1.9e-8 | 2.2e-7 | 3.4e-7 | 2.8e-7 | pass |  |
| W1 n_t=32 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.026 | 0.500 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.249 | 0.555 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.192 | 0.541 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 1.854 | 0.458 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 1.868 | 0.461 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 1.782 | 0.440 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass | yes |
| W1 n_t=32 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.120 | 0.524 | 1.2e-8 | 7.7e-8 | 2.6e-7 | 1.9e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.303 | 0.569 | 1.2e-8 | 7.7e-8 | 2.6e-7 | 1.9e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.275 | 0.562 | 1.2e-8 | 7.7e-8 | 2.6e-7 | 1.9e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 1.889 | 0.466 | 2.8e-8 | 1.4e-7 | 3.1e-7 | 2.3e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 1.918 | 0.473 | 2.8e-8 | 1.4e-7 | 3.1e-7 | 2.3e-7 | pass | no |
| W1 n_t=32 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 1.831 | 0.452 | 2.8e-8 | 1.4e-7 | 3.1e-7 | 2.3e-7 | pass | no |
| W1 n_t=32 gathered | reference | scalar |  |  | sqrt+div | 0.316 | 0.078 | 5.6e-8 | 4.6e-7 | 9.1e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.192 | 0.541 | 3.2e-8 | 2.0e-7 | 5.4e-7 | 4.7e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.442 | 0.603 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.641 | 0.652 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.660 | 0.657 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.092 | 0.516 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.153 | 0.532 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.204 | 0.544 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.423 | 0.598 | 3.5e-8 | 1.7e-7 | 4.2e-7 | 4.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.652 | 0.655 | 3.5e-8 | 1.7e-7 | 4.2e-7 | 4.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.665 | 0.658 | 3.5e-8 | 1.7e-7 | 4.2e-7 | 4.1e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.090 | 0.516 | 3.0e-8 | 1.8e-7 | 5.6e-7 | 4.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.157 | 0.533 | 3.0e-8 | 1.8e-7 | 5.6e-7 | 4.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.189 | 0.541 | 3.0e-8 | 1.8e-7 | 5.6e-7 | 4.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 1.984 | 0.490 | 5.9e-8 | 4.9e-7 | 9.0e-7 | 5.5e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.676 | 0.414 | 5.8e-8 | 4.9e-7 | 7.2e-7 | 5.7e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.153 | 0.532 | 5.8e-8 | 4.0e-7 | 7.5e-7 | 5.4e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.805 | 0.446 | 7.0e-8 | 4.9e-7 | 8.4e-7 | 5.9e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.195 | 0.542 | 7.0e-8 | 4.9e-7 | 8.6e-7 | 5.7e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.032 | 0.502 | 7.0e-8 | 4.9e-7 | 9.0e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 1.883 | 0.465 | 7.0e-8 | 4.9e-7 | 9.0e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.641 | 0.652 | 5.9e-8 | 4.9e-7 | 8.0e-7 | 5.1e-7 | pass |  |
| W1 n_t=64 per-pair | reference | scalar |  |  | sqrt+div | 0.347 | 0.086 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.9e-7 | pass |  |
| W1 n_t=64 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.012 | 0.497 | 2.1e-8 | 9.0e-8 | 3.3e-7 | 2.2e-7 | pass |  |
| W1 n_t=64 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.272 | 0.561 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.439 | 0.602 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.420 | 0.598 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 1.980 | 0.489 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.015 | 0.497 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.036 | 0.503 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass | yes |
| W1 n_t=64 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.335 | 0.577 | 1.5e-8 | 1.0e-7 | 3.5e-7 | 1.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.516 | 0.621 | 1.5e-8 | 1.0e-7 | 3.5e-7 | 1.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.472 | 0.610 | 1.5e-8 | 1.0e-7 | 3.5e-7 | 1.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 1.996 | 0.493 | 1.8e-8 | 2.6e-7 | 3.3e-7 | 2.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.009 | 0.496 | 1.8e-8 | 2.6e-7 | 3.3e-7 | 2.8e-7 | pass | no |
| W1 n_t=64 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.005 | 0.495 | 1.8e-8 | 2.6e-7 | 3.3e-7 | 2.8e-7 | pass | no |
| W1 n_t=64 gathered | reference | scalar |  |  | sqrt+div | 0.315 | 0.078 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.9e-7 | pass |  |
| W1 n_t=64 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.206 | 0.545 | 4.0e-8 | 3.8e-7 | 8.7e-7 | 4.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.455 | 0.606 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.652 | 0.655 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.671 | 0.659 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.100 | 0.519 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.160 | 0.533 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.211 | 0.546 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.437 | 0.602 | 5.5e-8 | 2.6e-7 | 8.7e-7 | 4.3e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.665 | 0.658 | 5.5e-8 | 2.6e-7 | 8.7e-7 | 4.3e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.682 | 0.662 | 5.5e-8 | 2.6e-7 | 8.7e-7 | 4.3e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.099 | 0.518 | 5.5e-8 | 3.4e-7 | 9.3e-7 | 4.6e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.165 | 0.535 | 5.5e-8 | 3.4e-7 | 9.3e-7 | 4.6e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.198 | 0.543 | 5.5e-8 | 3.4e-7 | 9.3e-7 | 4.6e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 1.990 | 0.491 | 1.2e-7 | 4.4e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.680 | 0.415 | 1.3e-7 | 4.4e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.162 | 0.534 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.7e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.811 | 0.447 | 1.3e-7 | 3.8e-7 | 1.2e-6 | 6.8e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.152 | 0.531 | 1.2e-7 | 2.9e-7 | 1.1e-6 | 7.1e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 1.981 | 0.489 | 1.2e-7 | 2.9e-7 | 1.1e-6 | 7.0e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 1.838 | 0.454 | 1.2e-7 | 2.9e-7 | 1.1e-6 | 7.0e-7 | pass |  |
| W1 n_t=64 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.581 | 0.637 | 1.3e-7 | 3.2e-7 | 1.1e-6 | 6.8e-7 | pass |  |
| W1 n_t=128 per-pair | reference | scalar |  |  | sqrt+div | 0.329 | 0.081 | 1.0e-7 | 6.2e-7 | 1.6e-6 | 1.2e-6 | pass |  |
| W1 n_t=128 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.105 | 0.520 | 1.6e-8 | 1.6e-7 | 3.5e-7 | 2.4e-7 | pass |  |
| W1 n_t=128 per-pair | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.318 | 0.572 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.539 | 0.627 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.547 | 0.629 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.010 | 0.496 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.082 | 0.514 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.126 | 0.525 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass | yes |
| W1 n_t=128 per-pair | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.398 | 0.592 | 1.4e-8 | 1.0e-7 | 3.0e-7 | 2.3e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.600 | 0.642 | 1.4e-8 | 1.0e-7 | 3.0e-7 | 2.3e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.573 | 0.635 | 1.4e-8 | 1.0e-7 | 3.0e-7 | 2.3e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.054 | 0.507 | 1.4e-8 | 2.2e-7 | 3.3e-7 | 2.9e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.112 | 0.521 | 1.4e-8 | 2.2e-7 | 3.3e-7 | 2.9e-7 | pass | no |
| W1 n_t=128 per-pair | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.084 | 0.515 | 1.4e-8 | 2.2e-7 | 3.3e-7 | 2.9e-7 | pass | no |
| W1 n_t=128 gathered | reference | scalar |  |  | sqrt+div | 0.315 | 0.078 | 1.0e-7 | 6.2e-7 | 1.6e-6 | 1.2e-6 | pass |  |
| W1 n_t=128 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.214 | 0.547 | 4.5e-8 | 4.0e-7 | 1.3e-6 | 6.4e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.462 | 0.608 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.662 | 0.657 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.679 | 0.662 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.106 | 0.520 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.166 | 0.535 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.217 | 0.547 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.450 | 0.605 | 4.7e-8 | 2.8e-7 | 1.4e-6 | 5.9e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.675 | 0.660 | 4.7e-8 | 2.8e-7 | 1.4e-6 | 5.9e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.690 | 0.664 | 4.7e-8 | 2.8e-7 | 1.4e-6 | 5.9e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.106 | 0.520 | 4.4e-8 | 4.0e-7 | 1.4e-6 | 6.6e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.171 | 0.536 | 4.4e-8 | 4.0e-7 | 1.4e-6 | 6.6e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.207 | 0.545 | 4.4e-8 | 4.0e-7 | 1.4e-6 | 6.6e-7 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_n2 | neon | TIL | 2 | est+N2 | 1.996 | 0.493 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.686 | 0.416 | 1.0e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_s2 | neon | TIL | 2 | est+S2 | 2.166 | 0.535 | 1.1e-7 | 7.0e-7 | 1.6e-6 | 1.3e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.814 | 0.448 | 1.1e-7 | 5.3e-7 | 1.6e-6 | 1.0e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_p2 | neon | TIL | 2 | est+P2 | 2.198 | 0.543 | 1.1e-7 | 5.8e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_p3 | neon | TIL | 2 | est+P3 | 2.044 | 0.505 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_p4 | neon | TIL | 2 | est+P4 | 1.894 | 0.468 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W1 n_t=128 gathered | f32_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.661 | 0.657 | 1.1e-7 | 5.9e-7 | 1.6e-6 | 1.1e-6 | pass |  |
| W2 N=1000 t≠s | reference | scalar |  |  | sqrt+div | 0.309 | 0.076 | 1.1e-7 | 7.9e-7 | 6.2e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.200 | 0.543 | 5.9e-8 | 5.0e-7 | 3.3e-7 | 2.3e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.449 | 0.605 | 1.1e-7 | 7.4e-7 | 6.2e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.650 | 0.654 | 1.1e-7 | 7.4e-7 | 6.2e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.643 | 0.653 | 1.1e-7 | 7.4e-7 | 6.2e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.096 | 0.517 | 1.2e-7 | 9.4e-7 | 6.1e-7 | 4.5e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.088 | 0.515 | 1.2e-7 | 9.4e-7 | 6.1e-7 | 4.5e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.189 | 0.541 | 1.2e-7 | 9.4e-7 | 6.1e-7 | 4.5e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.435 | 0.601 | 6.9e-8 | 3.8e-7 | 3.2e-7 | 3.0e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.665 | 0.658 | 6.9e-8 | 3.8e-7 | 3.2e-7 | 3.0e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.676 | 0.661 | 6.9e-8 | 3.8e-7 | 3.2e-7 | 3.0e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.096 | 0.518 | 6.2e-8 | 4.8e-7 | 3.3e-7 | 2.3e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.163 | 0.534 | 6.2e-8 | 4.8e-7 | 3.3e-7 | 2.3e-7 | pass |  |
| W2 N=1000 t≠s | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.191 | 0.541 | 6.2e-8 | 4.8e-7 | 3.3e-7 | 2.3e-7 | pass |  |
| W2 N=1000 t=s | reference | scalar |  |  | sqrt+div | 0.315 | 0.078 | 1.2e-7 | 9.6e-7 | 6.6e-7 | 6.6e-7 | pass |  |
| W2 N=1000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.202 | 0.544 | 5.7e-8 | 3.6e-7 | 3.4e-7 | 3.2e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | check failed | – | 1.2e-7 | 1.0e-6 | 6.6e-7 | 6.7e-7 | FAIL |  |
| W2 N=1000 t=s | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | check failed | – | 1.2e-7 | 1.0e-6 | 6.6e-7 | 6.7e-7 | FAIL |  |
| W2 N=1000 t=s | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | check failed | – | 1.2e-7 | 1.0e-6 | 6.6e-7 | 6.7e-7 | FAIL |  |
| W2 N=1000 t=s | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.096 | 0.518 | 1.2e-7 | 8.6e-7 | 6.6e-7 | 6.3e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.157 | 0.533 | 1.2e-7 | 8.6e-7 | 6.6e-7 | 6.3e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.191 | 0.541 | 1.2e-7 | 8.6e-7 | 6.6e-7 | 6.3e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.432 | 0.601 | 4.8e-8 | 4.2e-7 | 3.3e-7 | 3.6e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.664 | 0.658 | 4.8e-8 | 4.2e-7 | 3.3e-7 | 3.6e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.676 | 0.661 | 4.8e-8 | 4.2e-7 | 3.3e-7 | 3.6e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.097 | 0.518 | 5.5e-8 | 4.2e-7 | 3.4e-7 | 3.2e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.163 | 0.534 | 5.5e-8 | 4.2e-7 | 3.4e-7 | 3.2e-7 | pass |  |
| W2 N=1000 t=s | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.191 | 0.541 | 5.5e-8 | 4.2e-7 | 3.4e-7 | 3.2e-7 | pass |  |
| W2 N=10000 t≠s | reference | scalar |  |  | sqrt+div | 0.313 | 0.077 | 1.5e-7 | 2.0e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.145 | 0.530 | 8.2e-8 | 1.4e-6 | 8.1e-7 | 8.9e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.432 | 0.601 | 1.6e-7 | 2.0e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.637 | 0.651 | 1.6e-7 | 2.0e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.666 | 0.658 | 1.6e-7 | 2.0e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.073 | 0.512 | 1.6e-7 | 2.1e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.166 | 0.535 | 1.6e-7 | 2.1e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.218 | 0.548 | 1.6e-7 | 2.1e-6 | 1.9e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.395 | 0.591 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.2e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.626 | 0.648 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.2e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.661 | 0.657 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.2e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.043 | 0.504 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.3e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.130 | 0.526 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.3e-7 | pass |  |
| W2 N=10000 t≠s | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.186 | 0.540 | 9.4e-8 | 1.5e-6 | 8.0e-7 | 9.3e-7 | pass |  |
| W2 N=10000 t=s | reference | scalar |  |  | sqrt+div | 0.313 | 0.077 | 1.6e-7 | 2.7e-6 | 1.9e-6 | 1.4e-6 | FAIL |  |
| W2 N=10000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 2.143 | 0.529 | 7.2e-8 | 1.3e-6 | 8.0e-7 | 8.2e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 2.433 | 0.601 | 1.8e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 2.637 | 0.651 | 1.8e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 2.681 | 0.662 | 1.8e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k1_gk | neon | TIL | 1 | est+S2 | 2.103 | 0.519 | 1.7e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k2_gk | neon | TIL | 2 | est+S2 | 2.167 | 0.535 | 1.7e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::til_k4_gk | neon | TIL | 4 | est+S2 | 2.218 | 0.548 | 1.7e-7 | 2.6e-6 | 1.9e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 2.397 | 0.592 | 7.1e-8 | 1.3e-6 | 8.0e-7 | 8.0e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.627 | 0.649 | 7.1e-8 | 1.3e-6 | 8.0e-7 | 8.0e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.660 | 0.657 | 7.1e-8 | 1.3e-6 | 8.0e-7 | 8.0e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t1_gk | neon | SIL | 1 | est+S2 | 2.043 | 0.504 | 7.5e-8 | 1.3e-6 | 8.0e-7 | 8.1e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t2_gk | neon | SIL | 2 | est+S2 | 2.130 | 0.526 | 7.5e-8 | 1.3e-6 | 8.0e-7 | 8.1e-7 | pass |  |
| W2 N=10000 t=s | f32_grad::sil_t4_gk | neon | SIL | 4 | est+S2 | 2.180 | 0.538 | 7.5e-8 | 1.3e-6 | 8.0e-7 | 8.1e-7 | pass |  |

### f64, potential

| cell | variant | ISA | order | K/T | rsqrt | Gpairs/s | pairs/cycle | max err φ | max err ∇ | L2 φ | L2 ∇ | check | = gathered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | reference | scalar |  |  | sqrt+div | 0.780 | 0.193 | 8.2e-17 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.022 | 0.252 | 4.4e-17 |  | 8.6e-16 |  | pass |  |
| W1 n_t=8 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 0.737 | 0.182 | 1.2e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 1.154 | 0.285 | 1.2e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 1.379 | 0.341 | 1.2e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 0.684 | 0.169 | 1.2e-16 |  | 1.1e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 0.851 | 0.210 | 1.2e-16 |  | 1.1e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 0.989 | 0.244 | 1.2e-16 |  | 1.1e-15 |  | pass | yes |
| W1 n_t=8 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.506 | 0.372 | 4.4e-17 |  | 3.8e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.610 | 0.398 | 4.4e-17 |  | 3.8e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.683 | 0.416 | 4.4e-17 |  | 3.8e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.113 | 0.275 | 4.4e-17 |  | 7.4e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.168 | 0.288 | 4.4e-17 |  | 7.4e-16 |  | pass | no |
| W1 n_t=8 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.175 | 0.290 | 4.4e-17 |  | 7.4e-16 |  | pass | no |
| W1 n_t=8 gathered | reference | scalar |  |  | sqrt+div | 0.359 | 0.089 | 8.2e-17 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.231 | 0.304 | 1.2e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.677 | 0.414 | 1.2e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 1.976 | 0.488 | 1.2e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.150 | 0.531 | 1.2e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.232 | 0.304 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.315 | 0.325 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.389 | 0.343 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.724 | 0.426 | 8.1e-17 |  | 1.6e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.024 | 0.500 | 8.1e-17 |  | 1.6e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.127 | 0.525 | 8.1e-17 |  | 1.6e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.258 | 0.311 | 1.2e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.335 | 0.330 | 1.2e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.409 | 0.348 | 1.2e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.138 | 0.281 | 1.2e-16 |  | 9.9e-16 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.963 | 0.238 | 1.2e-16 |  | 9.6e-16 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.312 | 0.324 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.104 | 0.273 | 1.2e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.139 | 0.281 | 1.2e-16 |  | 1.0e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.089 | 0.269 | 1.2e-16 |  | 1.0e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.044 | 0.258 | 1.2e-16 |  | 1.0e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.106 | 0.273 | 1.2e-16 |  | 7.8e-16 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.972 | 0.487 | 1.2e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.463 | 0.361 | 6.5e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.571 | 0.388 | 6.5e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.655 | 0.409 | 6.5e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.193 | 0.295 | 6.1e-14 |  | 6.6e-13 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.256 | 0.310 | 6.1e-14 |  | 6.6e-13 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.328 | 0.328 | 6.1e-14 |  | 6.6e-13 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.139 | 0.281 | 1.3e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.199 | 0.296 | 1.3e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=8 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.227 | 0.303 | 1.3e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=16 per-pair | reference | scalar |  |  | sqrt+div | 0.645 | 0.159 | 1.1e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=16 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.123 | 0.277 | 4.4e-17 |  | 9.9e-16 |  | pass |  |
| W1 n_t=16 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.165 | 0.288 | 1.2e-16 |  | 2.0e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 1.578 | 0.390 | 1.2e-16 |  | 2.0e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 1.816 | 0.448 | 1.2e-16 |  | 2.0e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.004 | 0.248 | 1.1e-16 |  | 1.8e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.083 | 0.267 | 1.1e-16 |  | 1.8e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.221 | 0.301 | 1.1e-16 |  | 1.8e-15 |  | pass | yes |
| W1 n_t=16 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.577 | 0.389 | 3.2e-17 |  | 6.9e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.860 | 0.459 | 3.2e-17 |  | 6.9e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.003 | 0.495 | 3.2e-17 |  | 6.9e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.172 | 0.289 | 3.2e-17 |  | 8.3e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.276 | 0.315 | 3.2e-17 |  | 8.3e-16 |  | pass | no |
| W1 n_t=16 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.320 | 0.326 | 3.2e-17 |  | 8.3e-16 |  | pass | no |
| W1 n_t=16 gathered | reference | scalar |  |  | sqrt+div | 0.345 | 0.085 | 1.1e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=16 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.266 | 0.313 | 6.6e-17 |  | 1.4e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.721 | 0.425 | 1.2e-16 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.025 | 0.500 | 1.2e-16 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.184 | 0.539 | 1.2e-16 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.264 | 0.312 | 1.1e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.338 | 0.330 | 1.1e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.403 | 0.346 | 1.1e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.757 | 0.434 | 9.7e-17 |  | 1.1e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.056 | 0.508 | 9.7e-17 |  | 1.1e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.159 | 0.533 | 9.7e-17 |  | 1.1e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.264 | 0.312 | 6.3e-17 |  | 1.3e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.355 | 0.335 | 6.3e-17 |  | 1.3e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.420 | 0.351 | 6.3e-17 |  | 1.3e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.155 | 0.285 | 8.6e-17 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.974 | 0.241 | 9.5e-17 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.338 | 0.330 | 1.1e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.121 | 0.277 | 8.6e-17 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.157 | 0.286 | 9.9e-17 |  | 1.8e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.107 | 0.273 | 9.9e-17 |  | 1.9e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.058 | 0.261 | 9.9e-17 |  | 1.9e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.121 | 0.277 | 9.9e-17 |  | 1.9e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.035 | 0.503 | 1.2e-16 |  | 2.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.509 | 0.373 | 5.8e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.609 | 0.397 | 5.8e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.675 | 0.414 | 5.8e-12 |  | 1.2e-10 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.218 | 0.301 | 4.7e-14 |  | 8.0e-13 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.278 | 0.315 | 4.7e-14 |  | 8.0e-13 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.341 | 0.331 | 4.7e-14 |  | 8.0e-13 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.164 | 0.287 | 1.4e-16 |  | 3.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.218 | 0.301 | 1.4e-16 |  | 3.0e-15 |  | pass |  |
| W1 n_t=16 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.237 | 0.306 | 1.4e-16 |  | 3.0e-15 |  | pass |  |
| W1 n_t=24 per-pair | reference | scalar |  |  | sqrt+div | 0.564 | 0.139 | 1.3e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=24 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.175 | 0.290 | 5.1e-17 |  | 4.0e-16 |  | pass |  |
| W1 n_t=24 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.365 | 0.337 | 1.6e-16 |  | 1.3e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 1.739 | 0.430 | 1.6e-16 |  | 1.3e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 1.979 | 0.489 | 1.6e-16 |  | 1.3e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.090 | 0.269 | 1.5e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.197 | 0.296 | 1.5e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.298 | 0.320 | 1.5e-16 |  | 1.2e-15 |  | pass | yes |
| W1 n_t=24 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.648 | 0.407 | 5.6e-17 |  | 4.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.925 | 0.475 | 5.6e-17 |  | 4.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.080 | 0.513 | 5.6e-17 |  | 4.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.212 | 0.299 | 5.6e-17 |  | 5.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.300 | 0.321 | 5.6e-17 |  | 5.1e-16 |  | pass | no |
| W1 n_t=24 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.358 | 0.335 | 5.6e-17 |  | 5.1e-16 |  | pass | no |
| W1 n_t=24 gathered | reference | scalar |  |  | sqrt+div | 0.347 | 0.086 | 1.3e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=24 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.272 | 0.314 | 1.6e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.738 | 0.429 | 1.6e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.045 | 0.505 | 1.6e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.192 | 0.541 | 1.6e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.270 | 0.314 | 1.5e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.342 | 0.331 | 1.5e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.408 | 0.348 | 1.5e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.770 | 0.437 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.072 | 0.512 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.170 | 0.536 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.270 | 0.314 | 1.5e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.361 | 0.336 | 1.5e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.424 | 0.352 | 1.5e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.159 | 0.286 | 1.4e-16 |  | 1.1e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.977 | 0.241 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.345 | 0.332 | 1.5e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.125 | 0.278 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.162 | 0.287 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.111 | 0.274 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.061 | 0.262 | 1.4e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.124 | 0.278 | 1.3e-16 |  | 1.2e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.050 | 0.506 | 1.6e-16 |  | 1.3e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.522 | 0.376 | 5.4e-12 |  | 4.3e-11 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.617 | 0.399 | 5.4e-12 |  | 4.3e-11 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.681 | 0.415 | 5.4e-12 |  | 4.3e-11 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.227 | 0.303 | 2.8e-14 |  | 2.7e-13 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.281 | 0.316 | 2.8e-14 |  | 2.7e-13 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.345 | 0.332 | 2.8e-14 |  | 2.7e-13 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.172 | 0.289 | 1.3e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.223 | 0.302 | 1.3e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=24 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.241 | 0.306 | 1.3e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=32 per-pair | reference | scalar |  |  | sqrt+div | 0.509 | 0.126 | 1.7e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=32 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.206 | 0.298 | 3.2e-17 |  | 6.0e-16 |  | pass |  |
| W1 n_t=32 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.461 | 0.361 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 1.802 | 0.445 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.023 | 0.500 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.111 | 0.274 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.234 | 0.305 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.241 | 0.306 | 1.6e-16 |  | 1.7e-15 |  | pass | yes |
| W1 n_t=32 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.584 | 0.391 | 3.1e-17 |  | 5.8e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.921 | 0.474 | 3.1e-17 |  | 5.8e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.089 | 0.516 | 3.1e-17 |  | 5.8e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.235 | 0.305 | 3.9e-17 |  | 5.6e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.320 | 0.326 | 3.9e-17 |  | 5.6e-16 |  | pass | no |
| W1 n_t=32 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.370 | 0.338 | 3.9e-17 |  | 5.6e-16 |  | pass | no |
| W1 n_t=32 gathered | reference | scalar |  |  | sqrt+div | 0.338 | 0.083 | 1.7e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=32 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.274 | 0.315 | 1.2e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.744 | 0.431 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.053 | 0.507 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.196 | 0.542 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.276 | 0.315 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.349 | 0.333 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.409 | 0.348 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.775 | 0.438 | 1.4e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.080 | 0.514 | 1.4e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.173 | 0.537 | 1.4e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.241 | 0.306 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.363 | 0.337 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.427 | 0.352 | 1.3e-16 |  | 1.4e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.161 | 0.287 | 1.5e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.979 | 0.242 | 1.4e-16 |  | 1.6e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.348 | 0.333 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.127 | 0.278 | 1.5e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.164 | 0.287 | 1.5e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.113 | 0.275 | 1.5e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.063 | 0.263 | 1.5e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.127 | 0.278 | 1.5e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.057 | 0.508 | 1.6e-16 |  | 1.7e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.530 | 0.378 | 4.2e-12 |  | 9.7e-11 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.622 | 0.401 | 4.2e-12 |  | 9.7e-11 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.677 | 0.414 | 4.2e-12 |  | 9.7e-11 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.229 | 0.303 | 2.4e-14 |  | 6.0e-13 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.286 | 0.318 | 2.4e-14 |  | 6.0e-13 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.347 | 0.333 | 2.4e-14 |  | 6.0e-13 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.175 | 0.290 | 1.9e-16 |  | 2.1e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.225 | 0.303 | 1.9e-16 |  | 2.1e-15 |  | pass |  |
| W1 n_t=32 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.242 | 0.307 | 1.9e-16 |  | 2.1e-15 |  | pass |  |
| W1 n_t=64 per-pair | reference | scalar |  |  | sqrt+div | 0.452 | 0.112 | 3.1e-16 |  | 2.4e-15 |  | pass |  |
| W1 n_t=64 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.236 | 0.305 | 3.2e-17 |  | 8.0e-16 |  | pass |  |
| W1 n_t=64 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.619 | 0.400 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 1.942 | 0.480 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.119 | 0.523 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.209 | 0.299 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.297 | 0.320 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.374 | 0.339 | 2.4e-16 |  | 2.5e-15 |  | pass | yes |
| W1 n_t=64 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.731 | 0.427 | 3.3e-17 |  | 6.5e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.037 | 0.503 | 3.3e-17 |  | 6.5e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.144 | 0.529 | 3.3e-17 |  | 6.5e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.252 | 0.309 | 3.3e-17 |  | 7.8e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.348 | 0.333 | 3.3e-17 |  | 7.8e-16 |  | pass | no |
| W1 n_t=64 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.404 | 0.347 | 3.3e-17 |  | 7.8e-16 |  | pass | no |
| W1 n_t=64 gathered | reference | scalar |  |  | sqrt+div | 0.340 | 0.084 | 3.1e-16 |  | 2.4e-15 |  | pass |  |
| W1 n_t=64 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.282 | 0.317 | 1.3e-16 |  | 1.8e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.752 | 0.433 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.067 | 0.510 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.200 | 0.543 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.279 | 0.316 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.353 | 0.334 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.411 | 0.349 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.775 | 0.438 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.090 | 0.516 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.117 | 0.523 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.276 | 0.315 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.367 | 0.337 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.430 | 0.353 | 1.3e-16 |  | 1.9e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.132 | 0.280 | 2.2e-16 |  | 2.6e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.933 | 0.230 | 2.3e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.293 | 0.319 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.125 | 0.278 | 2.4e-16 |  | 2.6e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.167 | 0.288 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.117 | 0.276 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.063 | 0.262 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.130 | 0.279 | 2.4e-16 |  | 2.6e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.069 | 0.511 | 2.4e-16 |  | 2.5e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.539 | 0.380 | 3.2e-12 |  | 1.0e-10 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.628 | 0.402 | 3.2e-12 |  | 1.0e-10 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.687 | 0.417 | 3.2e-12 |  | 1.0e-10 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.236 | 0.305 | 2.1e-14 |  | 5.6e-13 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.287 | 0.318 | 2.1e-14 |  | 5.6e-13 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.349 | 0.333 | 2.1e-14 |  | 5.6e-13 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.181 | 0.292 | 2.9e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.228 | 0.303 | 2.9e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=64 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.244 | 0.307 | 2.9e-16 |  | 3.1e-15 |  | pass |  |
| W1 n_t=128 per-pair | reference | scalar |  |  | sqrt+div | 0.382 | 0.094 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.259 | 0.311 | 2.6e-17 |  | 7.4e-16 |  | pass |  |
| W1 n_t=128 per-pair | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.661 | 0.410 | 2.2e-16 |  | 3.3e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 1.983 | 0.490 | 2.2e-16 |  | 3.3e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.159 | 0.533 | 2.2e-16 |  | 3.3e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.231 | 0.304 | 2.2e-16 |  | 3.2e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.316 | 0.325 | 2.2e-16 |  | 3.2e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.386 | 0.342 | 2.2e-16 |  | 3.2e-15 |  | pass | yes |
| W1 n_t=128 per-pair | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.705 | 0.421 | 2.7e-17 |  | 6.6e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.067 | 0.510 | 2.7e-17 |  | 6.6e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.169 | 0.536 | 2.7e-17 |  | 6.6e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.272 | 0.314 | 2.7e-17 |  | 6.8e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.359 | 0.336 | 2.7e-17 |  | 6.8e-16 |  | pass | no |
| W1 n_t=128 per-pair | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.414 | 0.349 | 2.7e-17 |  | 6.8e-16 |  | pass | no |
| W1 n_t=128 gathered | reference | scalar |  |  | sqrt+div | 0.892 | 0.220 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.279 | 0.316 | 1.7e-16 |  | 2.8e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.755 | 0.433 | 2.2e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.065 | 0.510 | 2.2e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.204 | 0.544 | 2.2e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.280 | 0.316 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.353 | 0.334 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.412 | 0.349 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.785 | 0.441 | 1.7e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.092 | 0.516 | 1.7e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.182 | 0.539 | 1.7e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.278 | 0.316 | 1.6e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.359 | 0.336 | 1.6e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.431 | 0.353 | 1.6e-16 |  | 2.9e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 1.165 | 0.288 | 2.3e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.982 | 0.242 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 1.352 | 0.334 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 1.130 | 0.279 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 1.168 | 0.289 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 1.116 | 0.276 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 1.066 | 0.263 | 2.1e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 1.130 | 0.279 | 2.2e-16 |  | 3.2e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 2.070 | 0.511 | 2.2e-16 |  | 3.3e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.540 | 0.380 | 2.5e-12 |  | 9.8e-11 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.627 | 0.402 | 2.5e-12 |  | 9.8e-11 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.688 | 0.417 | 2.5e-12 |  | 9.8e-11 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 1.238 | 0.306 | 1.8e-14 |  | 5.9e-13 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 1.290 | 0.319 | 1.8e-14 |  | 5.9e-13 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 1.350 | 0.333 | 1.8e-14 |  | 5.9e-13 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 1.181 | 0.292 | 2.4e-16 |  | 3.4e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 1.228 | 0.303 | 2.4e-16 |  | 3.4e-15 |  | pass |  |
| W1 n_t=128 gathered | f64_pot::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 1.246 | 0.308 | 2.4e-16 |  | 3.4e-15 |  | pass |  |
| W2 N=1000 t≠s | reference | scalar |  |  | sqrt+div | 0.344 | 0.085 | 2.1e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.281 | 0.316 | 1.8e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.752 | 0.432 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.067 | 0.510 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.204 | 0.544 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.279 | 0.316 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.352 | 0.334 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.413 | 0.349 | 2.3e-16 |  | 1.1e-15 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.786 | 0.441 | 1.8e-16 |  | 9.0e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.091 | 0.516 | 1.8e-16 |  | 9.0e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.184 | 0.539 | 1.8e-16 |  | 9.0e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.279 | 0.316 | 1.8e-16 |  | 9.1e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.367 | 0.338 | 1.8e-16 |  | 9.1e-16 |  | pass |  |
| W2 N=1000 t≠s | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.431 | 0.353 | 1.8e-16 |  | 9.1e-16 |  | pass |  |
| W2 N=1000 t=s | reference | scalar |  |  | sqrt+div | 0.344 | 0.085 | 3.0e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.282 | 0.316 | 2.0e-16 |  | 9.4e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.752 | 0.433 | 2.5e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.069 | 0.511 | 2.5e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.208 | 0.545 | 2.5e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.283 | 0.317 | 2.4e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.353 | 0.334 | 2.4e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.412 | 0.349 | 2.4e-16 |  | 1.2e-15 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.785 | 0.441 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.091 | 0.516 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.186 | 0.540 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.278 | 0.316 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.367 | 0.338 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=1000 t=s | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.430 | 0.353 | 2.0e-16 |  | 9.2e-16 |  | pass |  |
| W2 N=10000 t≠s | reference | scalar |  |  | sqrt+div | 0.327 | 0.081 | 2.6e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.271 | 0.314 | 2.1e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.755 | 0.433 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.071 | 0.511 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.212 | 0.546 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.279 | 0.316 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.354 | 0.334 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.412 | 0.349 | 2.7e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.750 | 0.432 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.074 | 0.512 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.189 | 0.540 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.264 | 0.312 | 2.3e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.364 | 0.337 | 2.3e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t≠s | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.432 | 0.354 | 2.3e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t=s | reference | scalar |  |  | sqrt+div | 0.331 | 0.082 | 6.2e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 1.268 | 0.313 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k1_best | neon | TIL | 1 | sqrt+div | 1.755 | 0.433 | 6.1e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k2_best | neon | TIL | 2 | sqrt+div | 2.046 | 0.505 | 6.1e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k4_best | neon | TIL | 4 | sqrt+div | 2.208 | 0.545 | 6.1e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k1_gk | neon | TIL | 1 | est+S3 | 1.279 | 0.316 | 6.2e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k2_gk | neon | TIL | 2 | est+S3 | 1.351 | 0.334 | 6.2e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::til_k4_gk | neon | TIL | 4 | est+S3 | 1.414 | 0.349 | 6.2e-16 |  | 3.5e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.745 | 0.431 | 1.7e-16 |  | 2.1e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t2_best | neon | SIL | 2 | sqrt+div | 2.069 | 0.511 | 1.7e-16 |  | 2.1e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t4_best | neon | SIL | 4 | sqrt+div | 2.189 | 0.541 | 1.7e-16 |  | 2.1e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t1_gk | neon | SIL | 1 | est+S3 | 1.262 | 0.312 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t2_gk | neon | SIL | 2 | est+S3 | 1.362 | 0.336 | 1.9e-16 |  | 2.2e-15 |  | pass |  |
| W2 N=10000 t=s | f64_pot::sil_t4_gk | neon | SIL | 4 | est+S3 | 1.428 | 0.353 | 1.9e-16 |  | 2.2e-15 |  | pass |  |

### f64, potential and gradient

| cell | variant | ISA | order | K/T | rsqrt | Gpairs/s | pairs/cycle | max err φ | max err ∇ | L2 φ | L2 ∇ | check | = gathered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | reference | scalar |  |  | sqrt+div | 0.477 | 0.118 | 8.2e-17 | 4.3e-16 | 1.2e-15 | 8.1e-16 | pass |  |
| W1 n_t=8 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.720 | 0.178 | 4.4e-17 | 4.3e-16 | 8.6e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 0.560 | 0.138 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 0.719 | 0.177 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 0.783 | 0.193 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.487 | 0.120 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.531 | 0.131 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.620 | 0.153 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass | yes |
| W1 n_t=8 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 0.997 | 0.246 | 4.4e-17 | 2.6e-16 | 7.7e-16 | 4.6e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.019 | 0.252 | 4.4e-17 | 2.6e-16 | 7.7e-16 | 4.6e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 0.907 | 0.224 | 4.4e-17 | 2.6e-16 | 7.7e-16 | 4.6e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.749 | 0.185 | 4.4e-17 | 4.3e-16 | 6.3e-16 | 8.4e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.757 | 0.187 | 4.4e-17 | 4.3e-16 | 6.3e-16 | 8.4e-16 | pass | no |
| W1 n_t=8 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.689 | 0.170 | 4.4e-17 | 4.3e-16 | 6.3e-16 | 8.4e-16 | pass | no |
| W1 n_t=8 gathered | reference | scalar |  |  | sqrt+div | 0.322 | 0.079 | 8.2e-17 | 4.3e-16 | 1.2e-15 | 8.1e-16 | pass |  |
| W1 n_t=8 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.901 | 0.222 | 1.2e-16 | 2.6e-16 | 1.4e-15 | 6.1e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.142 | 0.282 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.249 | 0.308 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.276 | 0.315 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.832 | 0.205 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.882 | 0.218 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.904 | 0.223 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.170 | 0.289 | 8.1e-17 | 1.4e-16 | 1.3e-15 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.278 | 0.315 | 8.1e-17 | 1.4e-16 | 1.3e-15 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.283 | 0.317 | 8.1e-17 | 1.4e-16 | 1.3e-15 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.861 | 0.213 | 4.4e-17 | 1.8e-16 | 5.5e-16 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.893 | 0.221 | 4.4e-17 | 1.8e-16 | 5.5e-16 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.916 | 0.226 | 4.4e-17 | 1.8e-16 | 5.5e-16 | 6.4e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 0.822 | 0.203 | 8.2e-17 | 4.3e-16 | 1.3e-15 | 8.0e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.696 | 0.172 | 8.2e-17 | 3.5e-16 | 8.7e-16 | 6.6e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 0.883 | 0.218 | 8.2e-17 | 5.2e-16 | 1.2e-15 | 1.0e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 0.745 | 0.184 | 8.2e-17 | 3.5e-16 | 7.3e-16 | 6.5e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 0.828 | 0.204 | 8.2e-17 | 4.3e-16 | 7.8e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 0.795 | 0.196 | 8.2e-17 | 4.3e-16 | 8.0e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.754 | 0.186 | 8.2e-17 | 4.3e-16 | 8.0e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 0.825 | 0.204 | 8.2e-17 | 3.5e-16 | 6.0e-16 | 6.6e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.241 | 0.306 | 8.2e-17 | 4.3e-16 | 8.9e-16 | 7.8e-16 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.001 | 0.247 | 6.5e-12 | 4.0e-11 | 1.2e-10 | 1.3e-10 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.049 | 0.259 | 6.5e-12 | 4.0e-11 | 1.2e-10 | 1.3e-10 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.090 | 0.269 | 6.5e-12 | 4.0e-11 | 1.2e-10 | 1.3e-10 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 0.886 | 0.219 | 6.1e-14 | 3.1e-13 | 6.6e-13 | 8.9e-13 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 0.919 | 0.227 | 6.1e-14 | 3.1e-13 | 6.6e-13 | 8.9e-13 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 0.946 | 0.233 | 6.1e-14 | 3.1e-13 | 6.6e-13 | 8.9e-13 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 0.832 | 0.205 | 1.5e-16 | 8.2e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 0.873 | 0.216 | 1.5e-16 | 8.2e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=8 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 0.860 | 0.212 | 1.5e-16 | 8.2e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=16 per-pair | reference | scalar |  |  | sqrt+div | 0.459 | 0.113 | 1.1e-16 | 2.0e-16 | 1.7e-15 | 7.4e-16 | pass |  |
| W1 n_t=16 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.800 | 0.198 | 4.4e-17 | 1.0e-16 | 9.9e-16 | 4.7e-16 | pass |  |
| W1 n_t=16 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 0.817 | 0.202 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 0.950 | 0.234 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.002 | 0.248 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.680 | 0.168 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.665 | 0.164 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.754 | 0.186 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass | yes |
| W1 n_t=16 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.089 | 0.269 | 3.2e-17 | 1.9e-16 | 4.9e-16 | 6.6e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.162 | 0.287 | 3.2e-17 | 1.9e-16 | 4.9e-16 | 6.6e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.106 | 0.273 | 3.2e-17 | 1.9e-16 | 4.9e-16 | 6.6e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.804 | 0.198 | 2.8e-17 | 1.5e-16 | 5.8e-16 | 4.4e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.829 | 0.205 | 2.8e-17 | 1.5e-16 | 5.8e-16 | 4.4e-16 | pass | no |
| W1 n_t=16 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.808 | 0.200 | 2.8e-17 | 1.5e-16 | 5.8e-16 | 4.4e-16 | pass | no |
| W1 n_t=16 gathered | reference | scalar |  |  | sqrt+div | 0.317 | 0.078 | 1.1e-16 | 2.0e-16 | 1.7e-15 | 7.4e-16 | pass |  |
| W1 n_t=16 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.907 | 0.224 | 6.6e-17 | 2.6e-16 | 1.4e-15 | 8.1e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.175 | 0.290 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.265 | 0.312 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.296 | 0.320 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.851 | 0.210 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.895 | 0.221 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.915 | 0.226 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.191 | 0.294 | 9.7e-17 | 1.9e-16 | 1.3e-15 | 8.8e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.290 | 0.318 | 9.7e-17 | 1.9e-16 | 1.3e-15 | 8.8e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.300 | 0.321 | 9.7e-17 | 1.9e-16 | 1.3e-15 | 8.8e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.870 | 0.215 | 5.5e-17 | 2.6e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.900 | 0.222 | 5.5e-17 | 2.6e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.921 | 0.227 | 5.5e-17 | 2.6e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 0.833 | 0.206 | 9.9e-17 | 1.5e-16 | 1.5e-15 | 7.4e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.704 | 0.174 | 9.9e-17 | 1.7e-16 | 1.5e-15 | 6.4e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 0.892 | 0.220 | 1.0e-16 | 2.6e-16 | 1.8e-15 | 9.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 0.753 | 0.186 | 1.1e-16 | 1.6e-16 | 1.5e-15 | 7.1e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 0.840 | 0.208 | 1.1e-16 | 1.5e-16 | 1.6e-15 | 6.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 0.807 | 0.199 | 1.2e-16 | 1.5e-16 | 1.7e-15 | 6.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.764 | 0.189 | 1.2e-16 | 1.5e-16 | 1.7e-15 | 6.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 0.837 | 0.207 | 1.1e-16 | 1.5e-16 | 1.6e-15 | 6.9e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.278 | 0.315 | 1.3e-16 | 2.0e-16 | 1.8e-15 | 9.0e-16 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.028 | 0.254 | 5.8e-12 | 4.3e-11 | 1.2e-10 | 1.2e-10 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.065 | 0.263 | 5.8e-12 | 4.3e-11 | 1.2e-10 | 1.2e-10 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.105 | 0.273 | 5.8e-12 | 4.3e-11 | 1.2e-10 | 1.2e-10 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 0.909 | 0.224 | 4.7e-14 | 3.2e-13 | 8.0e-13 | 7.7e-13 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 0.934 | 0.231 | 4.7e-14 | 3.2e-13 | 8.0e-13 | 7.7e-13 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 0.956 | 0.236 | 4.7e-14 | 3.2e-13 | 8.0e-13 | 7.7e-13 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 0.851 | 0.210 | 1.3e-16 | 8.3e-16 | 2.6e-15 | 2.1e-15 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 0.888 | 0.219 | 1.3e-16 | 8.3e-16 | 2.6e-15 | 2.1e-15 | pass |  |
| W1 n_t=16 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 0.871 | 0.215 | 1.3e-16 | 8.3e-16 | 2.6e-15 | 2.1e-15 | pass |  |
| W1 n_t=24 per-pair | reference | scalar |  |  | sqrt+div | 0.406 | 0.100 | 1.3e-16 | 3.2e-16 | 1.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.839 | 0.207 | 5.1e-17 | 1.3e-16 | 4.0e-16 | 3.8e-16 | pass |  |
| W1 n_t=24 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 0.926 | 0.229 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.049 | 0.259 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.091 | 0.269 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.745 | 0.184 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.724 | 0.179 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.805 | 0.199 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass | yes |
| W1 n_t=24 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.135 | 0.280 | 3.6e-17 | 1.3e-16 | 3.8e-16 | 3.5e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.200 | 0.296 | 3.6e-17 | 1.3e-16 | 3.8e-16 | 3.5e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.177 | 0.291 | 3.6e-17 | 1.3e-16 | 3.8e-16 | 3.5e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.829 | 0.205 | 4.3e-17 | 1.3e-16 | 3.7e-16 | 4.9e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.849 | 0.210 | 4.3e-17 | 1.3e-16 | 3.7e-16 | 4.9e-16 | pass | no |
| W1 n_t=24 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.847 | 0.209 | 4.3e-17 | 1.3e-16 | 3.7e-16 | 4.9e-16 | pass | no |
| W1 n_t=24 gathered | reference | scalar |  |  | sqrt+div | 0.317 | 0.078 | 1.3e-16 | 3.2e-16 | 1.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.910 | 0.225 | 1.6e-16 | 3.0e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.186 | 0.293 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.286 | 0.317 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.301 | 0.321 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.856 | 0.211 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.899 | 0.222 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.919 | 0.227 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.197 | 0.296 | 1.4e-16 | 2.7e-16 | 1.3e-15 | 7.2e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.296 | 0.320 | 1.4e-16 | 2.7e-16 | 1.3e-15 | 7.2e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.306 | 0.322 | 1.4e-16 | 2.7e-16 | 1.3e-15 | 7.2e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.847 | 0.209 | 1.7e-16 | 3.0e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.902 | 0.223 | 1.7e-16 | 3.0e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.926 | 0.229 | 1.7e-16 | 3.0e-16 | 1.4e-15 | 8.0e-16 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 0.800 | 0.197 | 1.3e-16 | 3.5e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.672 | 0.166 | 1.3e-16 | 3.5e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 0.875 | 0.216 | 1.4e-16 | 4.2e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 0.759 | 0.187 | 1.1e-16 | 3.5e-16 | 1.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 0.844 | 0.208 | 1.3e-16 | 3.7e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 0.811 | 0.200 | 1.3e-16 | 3.7e-16 | 1.2e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.768 | 0.190 | 1.3e-16 | 3.7e-16 | 1.2e-15 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 0.839 | 0.207 | 1.6e-16 | 3.7e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.285 | 0.317 | 1.3e-16 | 3.0e-16 | 1.2e-15 | 1.1e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.035 | 0.255 | 5.4e-12 | 3.1e-11 | 4.3e-11 | 1.1e-10 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.071 | 0.264 | 5.4e-12 | 3.1e-11 | 4.3e-11 | 1.1e-10 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.110 | 0.274 | 5.4e-12 | 3.1e-11 | 4.3e-11 | 1.1e-10 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 0.915 | 0.226 | 2.8e-14 | 2.1e-13 | 2.7e-13 | 7.2e-13 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 0.938 | 0.232 | 2.8e-14 | 2.1e-13 | 2.7e-13 | 7.2e-13 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 0.961 | 0.237 | 2.8e-14 | 2.1e-13 | 2.7e-13 | 7.2e-13 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 0.856 | 0.211 | 1.6e-16 | 6.9e-16 | 1.6e-15 | 2.4e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 0.892 | 0.220 | 1.6e-16 | 6.9e-16 | 1.6e-15 | 2.4e-15 | pass |  |
| W1 n_t=24 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 0.875 | 0.216 | 1.6e-16 | 6.9e-16 | 1.6e-15 | 2.4e-15 | pass |  |
| W1 n_t=32 per-pair | reference | scalar |  |  | sqrt+div | 0.379 | 0.094 | 1.7e-16 | 4.5e-16 | 1.8e-15 | 1.0e-15 | pass |  |
| W1 n_t=32 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.860 | 0.212 | 3.2e-17 | 2.4e-16 | 6.0e-16 | 4.6e-16 | pass |  |
| W1 n_t=32 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 0.992 | 0.245 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.107 | 0.273 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.136 | 0.281 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.777 | 0.192 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.763 | 0.188 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.831 | 0.205 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass | yes |
| W1 n_t=32 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.155 | 0.285 | 2.6e-17 | 3.1e-16 | 7.3e-16 | 4.2e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.231 | 0.304 | 2.6e-17 | 3.1e-16 | 7.3e-16 | 4.2e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.208 | 0.298 | 2.6e-17 | 3.1e-16 | 7.3e-16 | 4.2e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.842 | 0.208 | 3.3e-17 | 3.1e-16 | 5.7e-16 | 5.1e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.866 | 0.214 | 3.3e-17 | 3.1e-16 | 5.7e-16 | 5.1e-16 | pass | no |
| W1 n_t=32 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.867 | 0.214 | 3.3e-17 | 3.1e-16 | 5.7e-16 | 5.1e-16 | pass | no |
| W1 n_t=32 gathered | reference | scalar |  |  | sqrt+div | 0.315 | 0.078 | 1.7e-16 | 4.5e-16 | 1.8e-15 | 1.0e-15 | pass |  |
| W1 n_t=32 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.911 | 0.225 | 1.2e-16 | 3.3e-16 | 1.4e-15 | 7.4e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.191 | 0.294 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.289 | 0.318 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.304 | 0.322 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.858 | 0.212 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.900 | 0.222 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.920 | 0.227 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.200 | 0.296 | 1.3e-16 | 3.1e-16 | 1.2e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.298 | 0.320 | 1.3e-16 | 3.1e-16 | 1.2e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.308 | 0.323 | 1.3e-16 | 3.1e-16 | 1.2e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.873 | 0.216 | 1.3e-16 | 2.9e-16 | 1.4e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.899 | 0.222 | 1.3e-16 | 2.9e-16 | 1.4e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.928 | 0.229 | 1.3e-16 | 2.9e-16 | 1.4e-15 | 7.2e-16 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 0.838 | 0.207 | 1.9e-16 | 6.2e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.708 | 0.175 | 1.6e-16 | 6.2e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 0.901 | 0.222 | 1.6e-16 | 4.2e-16 | 2.1e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 0.759 | 0.187 | 2.1e-16 | 6.2e-16 | 1.9e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 0.845 | 0.209 | 1.6e-16 | 3.9e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 0.812 | 0.200 | 1.6e-16 | 3.6e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.769 | 0.190 | 1.6e-16 | 3.6e-16 | 2.0e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 0.810 | 0.200 | 1.9e-16 | 6.2e-16 | 1.9e-15 | 1.2e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.242 | 0.307 | 1.6e-16 | 3.9e-16 | 1.8e-15 | 1.1e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.039 | 0.256 | 4.2e-12 | 7.4e-11 | 9.7e-11 | 1.5e-10 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.075 | 0.265 | 4.2e-12 | 7.4e-11 | 9.7e-11 | 1.5e-10 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.112 | 0.274 | 4.2e-12 | 7.4e-11 | 9.7e-11 | 1.5e-10 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 0.918 | 0.227 | 2.4e-14 | 4.2e-13 | 6.0e-13 | 7.6e-13 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 0.940 | 0.232 | 2.4e-14 | 4.2e-13 | 6.0e-13 | 7.6e-13 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 0.962 | 0.237 | 2.4e-14 | 4.2e-13 | 6.0e-13 | 7.6e-13 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 0.858 | 0.212 | 2.1e-16 | 8.1e-16 | 1.8e-15 | 2.0e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 0.894 | 0.221 | 2.1e-16 | 8.1e-16 | 1.8e-15 | 2.0e-15 | pass |  |
| W1 n_t=32 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 0.875 | 0.216 | 2.1e-16 | 8.1e-16 | 1.8e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 per-pair | reference | scalar |  |  | sqrt+div | 0.344 | 0.085 | 3.1e-16 | 1.7e-15 | 2.4e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.890 | 0.220 | 3.2e-17 | 1.6e-16 | 8.0e-16 | 4.3e-16 | pass |  |
| W1 n_t=64 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.098 | 0.271 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.198 | 0.296 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.222 | 0.302 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.821 | 0.203 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.830 | 0.205 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.876 | 0.216 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass | yes |
| W1 n_t=64 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.184 | 0.292 | 3.2e-17 | 2.2e-16 | 6.9e-16 | 4.0e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.271 | 0.314 | 3.2e-17 | 2.2e-16 | 6.9e-16 | 4.0e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.263 | 0.312 | 3.2e-17 | 2.2e-16 | 6.9e-16 | 4.0e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.862 | 0.213 | 3.3e-17 | 4.4e-16 | 7.5e-16 | 4.7e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.881 | 0.218 | 3.3e-17 | 4.4e-16 | 7.5e-16 | 4.7e-16 | pass | no |
| W1 n_t=64 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.900 | 0.222 | 3.3e-17 | 4.4e-16 | 7.5e-16 | 4.7e-16 | pass | no |
| W1 n_t=64 gathered | reference | scalar |  |  | sqrt+div | 0.314 | 0.078 | 3.1e-16 | 1.7e-15 | 2.4e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.915 | 0.226 | 1.3e-16 | 2.3e-15 | 1.8e-15 | 2.5e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.200 | 0.296 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.294 | 0.319 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.308 | 0.323 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.862 | 0.213 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.903 | 0.223 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.922 | 0.228 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.204 | 0.297 | 1.2e-16 | 2.5e-15 | 1.9e-15 | 2.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.302 | 0.322 | 1.2e-16 | 2.5e-15 | 1.9e-15 | 2.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.311 | 0.324 | 1.2e-16 | 2.5e-15 | 1.9e-15 | 2.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.876 | 0.216 | 1.4e-16 | 2.2e-15 | 1.8e-15 | 2.4e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.905 | 0.224 | 1.4e-16 | 2.2e-15 | 1.8e-15 | 2.4e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.929 | 0.229 | 1.4e-16 | 2.2e-15 | 1.8e-15 | 2.4e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 0.841 | 0.208 | 2.9e-16 | 1.5e-15 | 2.4e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.710 | 0.175 | 2.9e-16 | 1.5e-15 | 2.3e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 0.904 | 0.223 | 2.6e-16 | 1.6e-15 | 2.3e-15 | 1.9e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 0.760 | 0.188 | 2.9e-16 | 1.6e-15 | 2.5e-15 | 1.8e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 0.847 | 0.209 | 2.9e-16 | 1.5e-15 | 2.5e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 0.814 | 0.201 | 2.9e-16 | 1.5e-15 | 2.5e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.770 | 0.190 | 2.9e-16 | 1.5e-15 | 2.5e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 0.844 | 0.208 | 2.9e-16 | 1.5e-15 | 2.4e-15 | 1.7e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.293 | 0.319 | 2.9e-16 | 1.9e-15 | 2.4e-15 | 2.1e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.041 | 0.257 | 3.2e-12 | 2.9e-11 | 1.0e-10 | 1.0e-10 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.080 | 0.267 | 3.2e-12 | 2.9e-11 | 1.0e-10 | 1.0e-10 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.115 | 0.275 | 3.2e-12 | 2.9e-11 | 1.0e-10 | 1.0e-10 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 0.922 | 0.228 | 2.1e-14 | 1.4e-13 | 5.6e-13 | 5.7e-13 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 0.943 | 0.233 | 2.1e-14 | 1.4e-13 | 5.6e-13 | 5.7e-13 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 0.964 | 0.238 | 2.1e-14 | 1.4e-13 | 5.6e-13 | 5.7e-13 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 0.858 | 0.212 | 3.3e-16 | 1.5e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 0.897 | 0.221 | 3.3e-16 | 1.5e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 0.878 | 0.217 | 3.3e-16 | 1.5e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 per-pair | reference | scalar |  |  | sqrt+div | 0.327 | 0.081 | 2.1e-16 | 7.7e-16 | 3.3e-15 | 2.1e-15 | pass |  |
| W1 n_t=128 per-pair | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.903 | 0.223 | 2.6e-17 | 1.7e-16 | 7.4e-16 | 5.4e-16 | pass |  |
| W1 n_t=128 per-pair | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.129 | 0.279 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.241 | 0.306 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.266 | 0.313 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.826 | 0.204 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.867 | 0.214 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.900 | 0.222 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass | yes |
| W1 n_t=128 per-pair | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.196 | 0.295 | 2.7e-17 | 3.6e-16 | 6.6e-16 | 6.6e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.289 | 0.318 | 2.7e-17 | 3.6e-16 | 6.6e-16 | 6.6e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.276 | 0.315 | 2.7e-17 | 3.6e-16 | 6.6e-16 | 6.6e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.870 | 0.215 | 2.2e-17 | 2.3e-16 | 7.5e-16 | 5.0e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.897 | 0.222 | 2.2e-17 | 2.3e-16 | 7.5e-16 | 5.0e-16 | pass | no |
| W1 n_t=128 per-pair | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.912 | 0.225 | 2.2e-17 | 2.3e-16 | 7.5e-16 | 5.0e-16 | pass | no |
| W1 n_t=128 gathered | reference | scalar |  |  | sqrt+div | 0.314 | 0.077 | 2.1e-16 | 7.7e-16 | 3.3e-15 | 2.1e-15 | pass |  |
| W1 n_t=128 gathered | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.913 | 0.226 | 1.7e-16 | 1.5e-15 | 2.8e-15 | 2.1e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.197 | 0.296 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.293 | 0.319 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.307 | 0.323 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.861 | 0.213 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.905 | 0.223 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.922 | 0.228 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.206 | 0.298 | 1.6e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.292 | 0.319 | 1.6e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.308 | 0.323 | 1.6e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.876 | 0.216 | 1.8e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.905 | 0.223 | 1.8e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.928 | 0.229 | 1.8e-16 | 1.3e-15 | 3.0e-15 | 2.0e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_n3 | neon | TIL | 2 | est+N3 | 0.840 | 0.208 | 2.0e-16 | 9.5e-16 | 3.3e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_n4 | neon | TIL | 2 | est+N4 | 0.710 | 0.175 | 2.0e-16 | 9.9e-16 | 3.3e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_s3 | neon | TIL | 2 | est+S3 | 0.905 | 0.223 | 2.2e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_s4 | neon | TIL | 2 | est+S4 | 0.758 | 0.187 | 2.0e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_p6 | neon | TIL | 2 | est+P6 | 0.847 | 0.209 | 2.0e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_p7 | neon | TIL | 2 | est+P7 | 0.815 | 0.201 | 2.0e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_p8 | neon | TIL | 2 | est+P8 | 0.769 | 0.190 | 2.0e-16 | 8.1e-16 | 3.3e-15 | 2.2e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_n1p3 | neon | TIL | 2 | est+N1+P3 | 0.843 | 0.208 | 2.0e-16 | 9.5e-16 | 3.3e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_c_sd | neon | TIL | 2 | sqrt+div | 1.292 | 0.319 | 1.9e-16 | 9.5e-16 | 3.2e-15 | 2.4e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_rel_s2 | neon | TIL | 1 | est+S2 | 1.042 | 0.257 | 2.5e-12 | 5.5e-11 | 9.8e-11 | 1.2e-10 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_rel_s2 | neon | TIL | 2 | est+S2 | 1.079 | 0.266 | 2.5e-12 | 5.5e-11 | 9.8e-11 | 1.2e-10 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_rel_s2 | neon | TIL | 4 | est+S2 | 1.114 | 0.275 | 2.5e-12 | 5.5e-11 | 9.8e-11 | 1.2e-10 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_rel_p4 | neon | TIL | 1 | est+P4 | 0.922 | 0.228 | 1.8e-14 | 4.0e-13 | 5.9e-13 | 6.7e-13 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_rel_p4 | neon | TIL | 2 | est+P4 | 0.944 | 0.233 | 1.8e-14 | 4.0e-13 | 5.9e-13 | 6.7e-13 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_rel_p4 | neon | TIL | 4 | est+P4 | 0.964 | 0.238 | 1.8e-14 | 4.0e-13 | 5.9e-13 | 6.7e-13 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k1_rel_n1p2 | neon | TIL | 1 | est+N1+P2 | 0.861 | 0.213 | 2.7e-16 | 1.4e-15 | 3.4e-15 | 2.9e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k2_rel_n1p2 | neon | TIL | 2 | est+N1+P2 | 0.897 | 0.221 | 2.7e-16 | 1.4e-15 | 3.4e-15 | 2.9e-15 | pass |  |
| W1 n_t=128 gathered | f64_grad::til_k4_rel_n1p2 | neon | TIL | 4 | est+N1+P2 | 0.877 | 0.217 | 2.7e-16 | 1.4e-15 | 3.4e-15 | 2.9e-15 | pass |  |
| W2 N=1000 t≠s | reference | scalar |  |  | sqrt+div | 0.314 | 0.078 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.3e-16 | pass |  |
| W2 N=1000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.915 | 0.226 | 1.8e-16 | 8.5e-16 | 9.2e-16 | 6.2e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.195 | 0.295 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.2e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.292 | 0.319 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.2e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.308 | 0.323 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.2e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.861 | 0.213 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.1e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.903 | 0.223 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.1e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.921 | 0.227 | 2.1e-16 | 1.2e-15 | 1.1e-15 | 9.1e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.205 | 0.297 | 1.7e-16 | 6.5e-16 | 8.9e-16 | 5.8e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.302 | 0.322 | 1.7e-16 | 6.5e-16 | 8.9e-16 | 5.8e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.311 | 0.324 | 1.7e-16 | 6.5e-16 | 8.9e-16 | 5.8e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.877 | 0.216 | 1.8e-16 | 5.8e-16 | 9.0e-16 | 5.7e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.906 | 0.224 | 1.8e-16 | 5.8e-16 | 9.0e-16 | 5.7e-16 | pass |  |
| W2 N=1000 t≠s | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.929 | 0.229 | 1.8e-16 | 5.8e-16 | 9.0e-16 | 5.7e-16 | pass |  |
| W2 N=1000 t=s | reference | scalar |  |  | sqrt+div | 0.315 | 0.078 | 3.0e-16 | 1.6e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.914 | 0.226 | 2.0e-16 | 1.1e-15 | 9.4e-16 | 8.3e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.197 | 0.295 | 3.0e-16 | 1.5e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.293 | 0.319 | 3.0e-16 | 1.5e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.308 | 0.323 | 3.0e-16 | 1.5e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.857 | 0.212 | 2.6e-16 | 1.7e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.903 | 0.223 | 2.6e-16 | 1.7e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.922 | 0.228 | 2.6e-16 | 1.7e-15 | 1.2e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.205 | 0.297 | 1.7e-16 | 9.5e-16 | 9.2e-16 | 7.7e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.275 | 0.315 | 1.7e-16 | 9.5e-16 | 9.2e-16 | 7.7e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.306 | 0.323 | 1.7e-16 | 9.5e-16 | 9.2e-16 | 7.7e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.872 | 0.215 | 1.5e-16 | 1.1e-15 | 9.2e-16 | 8.0e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.904 | 0.223 | 1.5e-16 | 1.1e-15 | 9.2e-16 | 8.0e-16 | pass |  |
| W2 N=1000 t=s | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.929 | 0.229 | 1.5e-16 | 1.1e-15 | 9.2e-16 | 8.0e-16 | pass |  |
| W2 N=10000 t≠s | reference | scalar |  |  | sqrt+div | 0.312 | 0.077 | 2.6e-16 | 4.7e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.906 | 0.224 | 2.1e-16 | 3.4e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.195 | 0.295 | 2.8e-16 | 4.6e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.293 | 0.319 | 2.8e-16 | 4.6e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.310 | 0.324 | 2.8e-16 | 4.6e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.861 | 0.213 | 2.6e-16 | 4.7e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.905 | 0.223 | 2.6e-16 | 4.7e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.921 | 0.227 | 2.6e-16 | 4.7e-15 | 3.5e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.196 | 0.295 | 2.0e-16 | 3.3e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.299 | 0.321 | 2.0e-16 | 3.3e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.310 | 0.323 | 2.0e-16 | 3.3e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.869 | 0.215 | 2.2e-16 | 3.4e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.905 | 0.223 | 2.2e-16 | 3.4e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t≠s | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.929 | 0.229 | 2.2e-16 | 3.4e-15 | 2.2e-15 | 1.8e-15 | pass |  |
| W2 N=10000 t=s | reference | scalar |  |  | sqrt+div | 0.312 | 0.077 | 6.2e-16 | 6.8e-15 | 3.5e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | green-kernels | Neon(Neon { neon: Neon }) |  |  |  | 0.905 | 0.223 | 1.9e-16 | 3.6e-15 | 2.2e-15 | 2.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k1_best | neon | TIL | 1 | sqrt+div | 1.200 | 0.296 | 5.9e-16 | 6.7e-15 | 3.5e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k2_best | neon | TIL | 2 | sqrt+div | 1.294 | 0.320 | 5.9e-16 | 6.7e-15 | 3.5e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k4_best | neon | TIL | 4 | sqrt+div | 1.311 | 0.324 | 5.9e-16 | 6.7e-15 | 3.5e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k1_gk | neon | TIL | 1 | est+S3 | 0.861 | 0.213 | 6.1e-16 | 6.9e-15 | 3.5e-15 | 3.4e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k2_gk | neon | TIL | 2 | est+S3 | 0.902 | 0.223 | 6.1e-16 | 6.9e-15 | 3.5e-15 | 3.4e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::til_k4_gk | neon | TIL | 4 | est+S3 | 0.923 | 0.228 | 6.1e-16 | 6.9e-15 | 3.5e-15 | 3.4e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t1_best | neon | SIL | 1 | sqrt+div | 1.195 | 0.295 | 1.7e-16 | 3.4e-15 | 2.1e-15 | 2.2e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t2_best | neon | SIL | 2 | sqrt+div | 1.231 | 0.304 | 1.7e-16 | 3.4e-15 | 2.1e-15 | 2.2e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t4_best | neon | SIL | 4 | sqrt+div | 1.311 | 0.324 | 1.7e-16 | 3.4e-15 | 2.1e-15 | 2.2e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t1_gk | neon | SIL | 1 | est+S3 | 0.869 | 0.215 | 1.8e-16 | 3.5e-15 | 2.1e-15 | 2.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t2_gk | neon | SIL | 2 | est+S3 | 0.905 | 0.223 | 1.8e-16 | 3.5e-15 | 2.1e-15 | 2.3e-15 | pass |  |
| W2 N=10000 t=s | f64_grad::sil_t4_gk | neon | SIL | 4 | est+S3 | 0.930 | 0.230 | 1.8e-16 | 3.5e-15 | 2.1e-15 | 2.3e-15 | pass |  |

## Inverse square root inside the kernel

Targets in lanes, K = 2, gathered W1 cells, Gpairs/s; the geometric mean over n_t decides the formulation among those within 4 u_T in the study.

### f32, potential

| rsqrt | FP ops | study error (u_T) | W1 n_t=8 gathered | W1 n_t=16 gathered | W1 n_t=24 gathered | W1 n_t=32 gathered | W1 n_t=64 gathered | W1 n_t=128 gathered | geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| est+N2 | 17 | 1.50 | 2.674 | 2.760 | 2.775 | 2.782 | 2.792 | 2.796 | 2.763 |
| est+N3 | 20 | 1.50 | 2.253 | 2.308 | 2.316 | 2.321 | 2.322 | 2.331 | 2.308 |
| est+S2 | 16 | 2.43 | 3.052 | 3.227 | 3.246 | 3.253 | 3.267 | 3.275 | 3.219 |
| est+S3 | 19 | 2.10 | 2.578 | 2.679 | 2.691 | 2.698 | 2.707 | 2.711 | 2.677 |
| est+P2 | 15 | 2.19 | 3.012 | 3.096 | 3.115 | 3.121 | 3.134 | 3.142 | 3.103 |
| est+P3 | 16 | 1.00 | 2.729 | 2.799 | 2.816 | 2.800 | 2.829 | 2.836 | 2.801 |
| est+P4 | 17 | 1.00 | 2.507 | 2.554 | 2.565 | 2.570 | 2.577 | 2.583 | 2.559 |
| sqrt+div | 11 | 1.50 | 3.741 | 4.009 | 4.040 | 4.058 | 4.076 | 4.088 | 4.000 |

### f32, potential and gradient

| rsqrt | FP ops | study error (u_T) | W1 n_t=8 gathered | W1 n_t=16 gathered | W1 n_t=24 gathered | W1 n_t=32 gathered | W1 n_t=64 gathered | W1 n_t=128 gathered | geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| est+N2 | 23 | 1.50 | 1.894 | 1.969 | 1.967 | 1.984 | 1.990 | 1.996 | 1.966 |
| est+N3 | 26 | 1.50 | 1.603 | 1.665 | 1.636 | 1.676 | 1.680 | 1.686 | 1.657 |
| est+S2 | 22 | 2.43 | 2.049 | 2.135 | 2.147 | 2.153 | 2.162 | 2.166 | 2.135 |
| est+S3 | 25 | 2.10 | 1.727 | 1.793 | 1.801 | 1.805 | 1.811 | 1.814 | 1.792 |
| est+P2 | 21 | 2.19 | 2.102 | 2.174 | 2.191 | 2.195 | 2.152 | 2.198 | 2.169 |
| est+P3 | 22 | 1.00 | 1.957 | 2.016 | 2.027 | 2.032 | 1.981 | 2.044 | 2.009 |
| est+P4 | 23 | 1.00 | 1.816 | 1.869 | 1.878 | 1.883 | 1.838 | 1.894 | 1.863 |
| sqrt+div | 17 | 1.50 | 2.411 | 2.611 | 2.632 | 2.641 | 2.581 | 2.661 | 2.588 |

### f64, potential

| rsqrt | FP ops | study error (u_T) | W1 n_t=8 gathered | W1 n_t=16 gathered | W1 n_t=24 gathered | W1 n_t=32 gathered | W1 n_t=64 gathered | W1 n_t=128 gathered | geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| est+N3 | 20 | 1.50 | 1.138 | 1.155 | 1.159 | 1.161 | 1.132 | 1.165 | 1.152 |
| est+N4 | 23 | 1.49 | 0.963 | 0.974 | 0.977 | 0.979 | 0.933 | 0.982 | 0.968 |
| est+S3 | 19 | 2.45 | 1.312 | 1.338 | 1.345 | 1.348 | 1.293 | 1.352 | 1.331 |
| est+S4 | 22 | 2.09 | 1.104 | 1.121 | 1.125 | 1.127 | 1.125 | 1.130 | 1.122 |
| est+P6 | 19 | 1.70 | 1.139 | 1.157 | 1.162 | 1.164 | 1.167 | 1.168 | 1.160 |
| est+P7 | 20 | 1.00 | 1.089 | 1.107 | 1.111 | 1.113 | 1.117 | 1.116 | 1.109 |
| est+P8 | 21 | 1.00 | 1.044 | 1.058 | 1.061 | 1.063 | 1.063 | 1.066 | 1.059 |
| est+N1+P3 | 20 | 1.50 | 1.106 | 1.121 | 1.124 | 1.127 | 1.130 | 1.130 | 1.123 |
| sqrt+div | 11 | 1.50 | 1.972 | 2.035 | 2.050 | 2.057 | 2.069 | 2.070 | 2.042 |

### f64, potential and gradient

| rsqrt | FP ops | study error (u_T) | W1 n_t=8 gathered | W1 n_t=16 gathered | W1 n_t=24 gathered | W1 n_t=32 gathered | W1 n_t=64 gathered | W1 n_t=128 gathered | geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| est+N3 | 26 | 1.50 | 0.822 | 0.833 | 0.800 | 0.838 | 0.841 | 0.840 | 0.829 |
| est+N4 | 29 | 1.49 | 0.696 | 0.704 | 0.672 | 0.708 | 0.710 | 0.710 | 0.700 |
| est+S3 | 25 | 2.45 | 0.883 | 0.892 | 0.875 | 0.901 | 0.904 | 0.905 | 0.893 |
| est+S4 | 28 | 2.09 | 0.745 | 0.753 | 0.759 | 0.759 | 0.760 | 0.758 | 0.756 |
| est+P6 | 25 | 1.70 | 0.828 | 0.840 | 0.844 | 0.845 | 0.847 | 0.847 | 0.842 |
| est+P7 | 26 | 1.00 | 0.795 | 0.807 | 0.811 | 0.812 | 0.814 | 0.815 | 0.809 |
| est+P8 | 27 | 1.00 | 0.754 | 0.764 | 0.768 | 0.769 | 0.770 | 0.769 | 0.766 |
| est+N1+P3 | 26 | 1.50 | 0.825 | 0.837 | 0.839 | 0.810 | 0.844 | 0.843 | 0.833 |
| sqrt+div | 17 | 1.50 | 1.241 | 1.278 | 1.285 | 1.242 | 1.293 | 1.292 | 1.272 |

## Kernels: summary

Gpairs/s. TIL = targets in lanes (K vectors per block), SIL = sources in lanes (T targets per block), both with the "best" inverse square root; "TIL gk" is the best K with green-kernels' formulation. Ratios: best TIL over green-kernels, over the reference, and best SIL over best TIL.

### f32, potential

| cell | reference | green-kernels | TIL K=1 | K=2 | K=4 | SIL T=1 | T=2 | T=4 | TIL gk | TIL/gk | TIL/ref | SIL/TIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | 0.798 | 1.966 | 2.057 | 2.454 | 1.062 | 2.682 | 2.489 | 2.376 | 2.089 (K=2) | 1.25 | 3.08 | 1.09 |
| W1 n_t=8 gathered | 0.371 | 2.906 | 3.398 | 3.741 | 2.033 | 3.469 | 3.919 | 4.220 | 3.051 (K=2) | 1.29 | 10.07 | 1.13 |
| W1 n_t=16 per-pair | 0.658 | 2.474 | 2.788 | 3.220 | 2.699 | 3.129 | 3.343 | 3.317 | 2.668 (K=2) | 1.30 | 4.89 | 1.04 |
| W1 n_t=16 gathered | 0.353 | 2.949 | 3.532 | 4.012 | 4.252 | 3.510 | 4.007 | 4.333 | 3.324 (K=4) | 1.44 | 12.04 | 1.02 |
| W1 n_t=24 per-pair | 0.567 | 2.582 | 3.032 | 3.547 | 2.391 | 3.210 | 3.620 | 3.735 | 2.891 (K=2) | 1.37 | 6.26 | 1.05 |
| W1 n_t=24 gathered | 0.348 | 3.016 | 3.563 | 4.039 | 3.224 | 3.537 | 4.049 | 4.342 | 3.247 (K=2) | 1.34 | 11.61 | 1.08 |
| W1 n_t=32 per-pair | 0.528 | 2.710 | 3.211 | 3.628 | 3.378 | 3.313 | 3.680 | 3.932 | 2.954 (K=2) | 1.34 | 6.87 | 1.08 |
| W1 n_t=32 gathered | 0.345 | 3.035 | 3.578 | 4.056 | 4.310 | 3.553 | 4.070 | 4.358 | 3.354 (K=4) | 1.42 | 12.51 | 1.01 |
| W1 n_t=64 per-pair | 0.483 | 2.865 | 3.455 | 3.878 | 3.806 | 3.459 | 3.919 | 4.161 | 3.049 (K=4) | 1.35 | 8.03 | 1.07 |
| W1 n_t=64 gathered | 0.341 | 3.056 | 3.601 | 4.074 | 4.330 | 3.574 | 4.084 | 4.388 | 3.368 (K=4) | 1.42 | 12.72 | 1.01 |
| W1 n_t=128 per-pair | 0.392 | 2.913 | 3.427 | 3.918 | 4.071 | 3.527 | 4.023 | 4.288 | 3.208 (K=4) | 1.40 | 10.40 | 1.05 |
| W1 n_t=128 gathered | 0.339 | 3.072 | 3.615 | 4.087 | 4.353 | 3.590 | 4.114 | 4.405 | 3.378 (K=4) | 1.42 | 12.84 | 1.01 |
| W2 N=1000 t≠s | 0.893 | 3.051 | 3.594 | 4.074 | 4.287 | 3.568 | 4.093 | 4.385 | 3.332 (K=4) | 1.40 | 4.80 | 1.02 |
| W2 N=1000 t=s | 0.344 | 3.049 | 3.593 | 4.068 | 4.280 | 3.571 | 4.095 | 4.383 | 3.321 (K=4) | 1.40 | 12.43 | 1.02 |
| W2 N=10000 t≠s | 0.337 | 2.968 | 3.570 | 4.062 | 4.326 | 3.470 | 4.041 | 4.336 | 3.370 (K=4) | 1.46 | 12.83 | 1.00 |
| W2 N=10000 t=s | 0.522 | 2.973 | 3.556 | 4.061 | 4.323 | 3.472 | 4.042 | 4.344 | 3.372 (K=4) | 1.45 | 8.28 | 1.00 |

Targets in lanes, geometric mean over the W1 cells (Gpairs/s): K=1: 3.236, K=2: 3.688, K=4: 3.108.

### f32, potential and gradient

| cell | reference | green-kernels | TIL K=1 | K=2 | K=4 | SIL T=1 | T=2 | T=4 | TIL gk | TIL/gk | TIL/ref | SIL/TIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | 0.500 | 1.282 | 1.301 | 1.410 | 0.753 | 1.662 | 1.552 | 1.389 | 1.235 (K=1) | 1.10 | 2.82 | 1.18 |
| W1 n_t=8 gathered | 0.332 | 2.106 | 2.308 | 2.421 | 1.273 | 2.363 | 2.560 | 2.572 | 2.059 (K=2) | 1.15 | 7.28 | 1.06 |
| W1 n_t=16 per-pair | 0.474 | 1.648 | 1.776 | 1.934 | 1.826 | 2.023 | 2.097 | 1.916 | 1.626 (K=2) | 1.17 | 4.08 | 1.08 |
| W1 n_t=16 gathered | 0.328 | 2.161 | 2.409 | 2.612 | 2.628 | 2.389 | 2.635 | 2.643 | 2.182 (K=4) | 1.22 | 8.01 | 1.01 |
| W1 n_t=24 per-pair | 0.417 | 1.777 | 1.961 | 2.147 | 1.570 | 2.125 | 2.255 | 2.149 | 1.789 (K=2) | 1.21 | 5.15 | 1.05 |
| W1 n_t=24 gathered | 0.323 | 2.182 | 2.433 | 2.631 | 1.990 | 2.412 | 2.642 | 2.654 | 2.148 (K=2) | 1.21 | 8.15 | 1.01 |
| W1 n_t=32 per-pair | 0.386 | 1.858 | 2.026 | 2.249 | 2.192 | 2.120 | 2.303 | 2.275 | 1.868 (K=2) | 1.21 | 5.83 | 1.02 |
| W1 n_t=32 gathered | 0.316 | 2.192 | 2.442 | 2.641 | 2.660 | 2.423 | 2.652 | 2.665 | 2.204 (K=4) | 1.21 | 8.42 | 1.00 |
| W1 n_t=64 per-pair | 0.347 | 2.012 | 2.272 | 2.439 | 2.420 | 2.335 | 2.516 | 2.472 | 2.036 (K=4) | 1.21 | 7.04 | 1.03 |
| W1 n_t=64 gathered | 0.315 | 2.206 | 2.455 | 2.652 | 2.671 | 2.437 | 2.665 | 2.682 | 2.211 (K=4) | 1.21 | 8.49 | 1.00 |
| W1 n_t=128 per-pair | 0.329 | 2.105 | 2.318 | 2.539 | 2.547 | 2.398 | 2.600 | 2.573 | 2.126 (K=4) | 1.21 | 7.74 | 1.02 |
| W1 n_t=128 gathered | 0.315 | 2.214 | 2.462 | 2.662 | 2.679 | 2.450 | 2.675 | 2.690 | 2.217 (K=4) | 1.21 | 8.51 | 1.00 |
| W2 N=1000 t≠s | 0.309 | 2.200 | 2.449 | 2.650 | 2.643 | 2.435 | 2.665 | 2.676 | 2.189 (K=4) | 1.20 | 8.58 | 1.01 |
| W2 N=1000 t=s | 0.315 | 2.202 | check failed | check failed | check failed | 2.432 | 2.664 | 2.676 | 2.191 (K=4) | – | – | – |
| W2 N=10000 t≠s | 0.313 | 2.145 | 2.432 | 2.637 | 2.666 | 2.395 | 2.626 | 2.661 | 2.218 (K=4) | 1.24 | 8.51 | 1.00 |
| W2 N=10000 t=s | 0.313 | 2.143 | 2.433 | 2.637 | 2.681 | 2.397 | 2.627 | 2.660 | 2.218 (K=4) | 1.25 | 8.55 | 0.99 |

Targets in lanes, geometric mean over the W1 cells (Gpairs/s): K=1: 2.148, K=2: 2.328, K=4: 1.983.

### f64, potential

| cell | reference | green-kernels | TIL K=1 | K=2 | K=4 | SIL T=1 | T=2 | T=4 | TIL gk | TIL/gk | TIL/ref | SIL/TIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | 0.780 | 1.022 | 0.737 | 1.154 | 1.379 | 1.506 | 1.610 | 1.683 | 0.989 (K=4) | 1.35 | 1.77 | 1.22 |
| W1 n_t=8 gathered | 0.359 | 1.231 | 1.677 | 1.976 | 2.150 | 1.724 | 2.024 | 2.127 | 1.389 (K=4) | 1.75 | 5.99 | 0.99 |
| W1 n_t=16 per-pair | 0.645 | 1.123 | 1.165 | 1.578 | 1.816 | 1.577 | 1.860 | 2.003 | 1.221 (K=4) | 1.62 | 2.81 | 1.10 |
| W1 n_t=16 gathered | 0.345 | 1.266 | 1.721 | 2.025 | 2.184 | 1.757 | 2.056 | 2.159 | 1.403 (K=4) | 1.73 | 6.34 | 0.99 |
| W1 n_t=24 per-pair | 0.564 | 1.175 | 1.365 | 1.739 | 1.979 | 1.648 | 1.925 | 2.080 | 1.298 (K=4) | 1.68 | 3.51 | 1.05 |
| W1 n_t=24 gathered | 0.347 | 1.272 | 1.738 | 2.045 | 2.192 | 1.770 | 2.072 | 2.170 | 1.408 (K=4) | 1.72 | 6.31 | 0.99 |
| W1 n_t=32 per-pair | 0.509 | 1.206 | 1.461 | 1.802 | 2.023 | 1.584 | 1.921 | 2.089 | 1.241 (K=4) | 1.68 | 3.97 | 1.03 |
| W1 n_t=32 gathered | 0.338 | 1.274 | 1.744 | 2.053 | 2.196 | 1.775 | 2.080 | 2.173 | 1.409 (K=4) | 1.72 | 6.50 | 0.99 |
| W1 n_t=64 per-pair | 0.452 | 1.236 | 1.619 | 1.942 | 2.119 | 1.731 | 2.037 | 2.144 | 1.374 (K=4) | 1.71 | 4.68 | 1.01 |
| W1 n_t=64 gathered | 0.340 | 1.282 | 1.752 | 2.067 | 2.200 | 1.775 | 2.090 | 2.117 | 1.411 (K=4) | 1.72 | 6.47 | 0.96 |
| W1 n_t=128 per-pair | 0.382 | 1.259 | 1.661 | 1.983 | 2.159 | 1.705 | 2.067 | 2.169 | 1.386 (K=4) | 1.71 | 5.65 | 1.00 |
| W1 n_t=128 gathered | 0.892 | 1.279 | 1.755 | 2.065 | 2.204 | 1.785 | 2.092 | 2.182 | 1.412 (K=4) | 1.72 | 2.47 | 0.99 |
| W2 N=1000 t≠s | 0.344 | 1.281 | 1.752 | 2.067 | 2.204 | 1.786 | 2.091 | 2.184 | 1.413 (K=4) | 1.72 | 6.41 | 0.99 |
| W2 N=1000 t=s | 0.344 | 1.282 | 1.752 | 2.069 | 2.208 | 1.785 | 2.091 | 2.186 | 1.412 (K=4) | 1.72 | 6.41 | 0.99 |
| W2 N=10000 t≠s | 0.327 | 1.271 | 1.755 | 2.071 | 2.212 | 1.750 | 2.074 | 2.189 | 1.412 (K=4) | 1.74 | 6.75 | 0.99 |
| W2 N=10000 t=s | 0.331 | 1.268 | 1.755 | 2.046 | 2.208 | 1.745 | 2.069 | 2.189 | 1.414 (K=4) | 1.74 | 6.67 | 0.99 |

Targets in lanes, geometric mean over the W1 cells (Gpairs/s): K=1: 1.494, K=2: 1.847, K=4: 2.034.

### f64, potential and gradient

| cell | reference | green-kernels | TIL K=1 | K=2 | K=4 | SIL T=1 | T=2 | T=4 | TIL gk | TIL/gk | TIL/ref | SIL/TIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 n_t=8 per-pair | 0.477 | 0.720 | 0.560 | 0.719 | 0.783 | 0.997 | 1.019 | 0.907 | 0.620 (K=4) | 1.09 | 1.64 | 1.30 |
| W1 n_t=8 gathered | 0.322 | 0.901 | 1.142 | 1.249 | 1.276 | 1.170 | 1.278 | 1.283 | 0.904 (K=4) | 1.42 | 3.97 | 1.01 |
| W1 n_t=16 per-pair | 0.459 | 0.800 | 0.817 | 0.950 | 1.002 | 1.089 | 1.162 | 1.106 | 0.754 (K=4) | 1.25 | 2.18 | 1.16 |
| W1 n_t=16 gathered | 0.317 | 0.907 | 1.175 | 1.265 | 1.296 | 1.191 | 1.290 | 1.300 | 0.915 (K=4) | 1.43 | 4.09 | 1.00 |
| W1 n_t=24 per-pair | 0.406 | 0.839 | 0.926 | 1.049 | 1.091 | 1.135 | 1.200 | 1.177 | 0.805 (K=4) | 1.30 | 2.69 | 1.10 |
| W1 n_t=24 gathered | 0.317 | 0.910 | 1.186 | 1.286 | 1.301 | 1.197 | 1.296 | 1.306 | 0.919 (K=4) | 1.43 | 4.11 | 1.00 |
| W1 n_t=32 per-pair | 0.379 | 0.860 | 0.992 | 1.107 | 1.136 | 1.155 | 1.231 | 1.208 | 0.831 (K=4) | 1.32 | 3.00 | 1.08 |
| W1 n_t=32 gathered | 0.315 | 0.911 | 1.191 | 1.289 | 1.304 | 1.200 | 1.298 | 1.308 | 0.920 (K=4) | 1.43 | 4.14 | 1.00 |
| W1 n_t=64 per-pair | 0.344 | 0.890 | 1.098 | 1.198 | 1.222 | 1.184 | 1.271 | 1.263 | 0.876 (K=4) | 1.37 | 3.55 | 1.04 |
| W1 n_t=64 gathered | 0.314 | 0.915 | 1.200 | 1.294 | 1.308 | 1.204 | 1.302 | 1.311 | 0.922 (K=4) | 1.43 | 4.17 | 1.00 |
| W1 n_t=128 per-pair | 0.327 | 0.903 | 1.129 | 1.241 | 1.266 | 1.196 | 1.289 | 1.276 | 0.900 (K=4) | 1.40 | 3.88 | 1.02 |
| W1 n_t=128 gathered | 0.314 | 0.913 | 1.197 | 1.293 | 1.307 | 1.206 | 1.292 | 1.308 | 0.922 (K=4) | 1.43 | 4.17 | 1.00 |
| W2 N=1000 t≠s | 0.314 | 0.915 | 1.195 | 1.292 | 1.308 | 1.205 | 1.302 | 1.311 | 0.921 (K=4) | 1.43 | 4.16 | 1.00 |
| W2 N=1000 t=s | 0.315 | 0.914 | 1.197 | 1.293 | 1.308 | 1.205 | 1.275 | 1.306 | 0.922 (K=4) | 1.43 | 4.16 | 1.00 |
| W2 N=10000 t≠s | 0.312 | 0.906 | 1.195 | 1.293 | 1.310 | 1.196 | 1.299 | 1.310 | 0.921 (K=4) | 1.45 | 4.20 | 1.00 |
| W2 N=10000 t=s | 0.312 | 0.905 | 1.200 | 1.294 | 1.311 | 1.195 | 1.231 | 1.311 | 0.923 (K=4) | 1.45 | 4.20 | 1.00 |

Targets in lanes, geometric mean over the W1 cells (Gpairs/s): K=1: 1.030, K=2: 1.147, K=4: 1.179.

## Loop order: the decision rule of design §4.1

Ratio best SIL / best TIL (each over its K or T, best inverse square root), geometric mean over the W1 cells (both forms, all n_t). Sources in lanes is recommended only if ≥ 1.15 in both precisions.

| precision | potential | potential and gradient | both outputs |
| --- | --- | --- | --- |
| f32 | 1.054 | 1.039 | 1.046 |
| f64 | 1.026 | 1.057 | 1.041 |

## Fraction of the design §4.6 model (targets in lanes, chosen K)

Model: 4 FP pipes × W lanes / (FP ops per pair and lane, design §4.2 with this inverse square root). Corrected: also limited by the measured throughput of the estimate and of FRSQRTS, and for sqrt+div by the divider (1/rate(FSQRT) + 1/rate(FDIV) cycles per vector; the two share one unit), from the single-instruction table. Rows for the chosen formulation and for green-kernels' (gk), at the chosen K. Pairs per cycle at 4.05 GHz.

| precision | output | K | rsqrt | ops/pair/lane | model pairs/cycle | corrected | cell | measured pairs/cycle | of model | of corrected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.330 | W1 n_t=8 gathered | 0.924 | 64% | 69% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=8 gathered | 0.753 | 75% | 75% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.330 | W1 n_t=16 gathered | 0.991 | 68% | 74% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=16 gathered | 0.797 | 80% | 80% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.330 | W1 n_t=24 gathered | 0.997 | 69% | 75% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=24 gathered | 0.802 | 80% | 80% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.330 | W1 n_t=32 gathered | 1.002 | 69% | 75% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=32 gathered | 0.804 | 80% | 80% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.330 | W1 n_t=64 gathered | 1.006 | 69% | 76% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=64 gathered | 0.806 | 81% | 81% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.330 | W1 n_t=128 gathered | 1.009 | 69% | 76% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W1 n_t=128 gathered | 0.808 | 81% | 81% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.330 | W2 N=1000 t≠s | 1.006 | 69% | 76% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W2 N=1000 t≠s | 0.805 | 81% | 81% |
| f32 | φ | 2 | sqrt+div | 11 | 1.455 | 1.330 | W2 N=10000 t≠s | 1.003 | 69% | 75% |
| f32 | φ | 2 | est+S2 | 16 | 1.000 | 1.000 | W2 N=10000 t≠s | 0.801 | 80% | 80% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=8 gathered | 0.598 | 64% | 64% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=8 gathered | 0.509 | 70% | 70% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=16 gathered | 0.645 | 69% | 69% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=16 gathered | 0.527 | 73% | 73% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=24 gathered | 0.650 | 69% | 69% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=24 gathered | 0.530 | 73% | 73% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=32 gathered | 0.652 | 69% | 69% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=32 gathered | 0.532 | 73% | 73% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=64 gathered | 0.655 | 70% | 70% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=64 gathered | 0.533 | 73% | 73% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W1 n_t=128 gathered | 0.657 | 70% | 70% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W1 n_t=128 gathered | 0.535 | 74% | 74% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W2 N=1000 t≠s | 0.654 | 70% | 70% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W2 N=1000 t≠s | 0.515 | 71% | 71% |
| f32 | φ, ∇φ | 2 | sqrt+div | 17 | 0.941 | 0.941 | W2 N=10000 t≠s | 0.651 | 69% | 69% |
| f32 | φ, ∇φ | 2 | est+S2 | 22 | 0.727 | 0.727 | W2 N=10000 t≠s | 0.535 | 74% | 74% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.665 | W1 n_t=8 gathered | 0.531 | 73% | 80% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=8 gathered | 0.343 | 81% | 81% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.665 | W1 n_t=16 gathered | 0.539 | 74% | 81% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=16 gathered | 0.346 | 82% | 82% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.665 | W1 n_t=24 gathered | 0.541 | 74% | 81% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=24 gathered | 0.348 | 83% | 83% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.665 | W1 n_t=32 gathered | 0.542 | 75% | 82% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=32 gathered | 0.348 | 83% | 83% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.665 | W1 n_t=64 gathered | 0.543 | 75% | 82% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=64 gathered | 0.349 | 83% | 83% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.665 | W1 n_t=128 gathered | 0.544 | 75% | 82% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W1 n_t=128 gathered | 0.349 | 83% | 83% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.665 | W2 N=1000 t≠s | 0.544 | 75% | 82% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W2 N=1000 t≠s | 0.349 | 83% | 83% |
| f64 | φ | 4 | sqrt+div | 11 | 0.727 | 0.665 | W2 N=10000 t≠s | 0.546 | 75% | 82% |
| f64 | φ | 4 | est+S3 | 19 | 0.421 | 0.421 | W2 N=10000 t≠s | 0.349 | 83% | 83% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=8 gathered | 0.315 | 67% | 67% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=8 gathered | 0.223 | 70% | 70% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=16 gathered | 0.320 | 68% | 68% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=16 gathered | 0.226 | 71% | 71% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=24 gathered | 0.321 | 68% | 68% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=24 gathered | 0.227 | 71% | 71% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=32 gathered | 0.322 | 68% | 68% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=32 gathered | 0.227 | 71% | 71% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=64 gathered | 0.323 | 69% | 69% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=64 gathered | 0.228 | 71% | 71% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W1 n_t=128 gathered | 0.323 | 69% | 69% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W1 n_t=128 gathered | 0.228 | 71% | 71% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W2 N=1000 t≠s | 0.323 | 69% | 69% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W2 N=1000 t≠s | 0.227 | 71% | 71% |
| f64 | φ, ∇φ | 4 | sqrt+div | 17 | 0.471 | 0.471 | W2 N=10000 t≠s | 0.324 | 69% | 69% |
| f64 | φ, ∇φ | 4 | est+S3 | 25 | 0.320 | 0.320 | W2 N=10000 t≠s | 0.227 | 71% | 71% |

## Relaxed f64 levels (targets in lanes, chosen K, W1 gathered)

| output | n_t | full: rsqrt, Gpairs/s, max err | relaxed S2: Gpairs/s (gain), max err | relaxed P4: Gpairs/s (gain), max err |
| --- | --- | --- | --- | --- |
| φ | 8 | sqrt+div, 2.150, 1.2e-16 | 1.655 (-23%), 6.5e-12 | 1.328 (-38%), 6.1e-14 |
| φ | 16 | sqrt+div, 2.184, 1.2e-16 | 1.675 (-23%), 5.8e-12 | 1.341 (-39%), 4.7e-14 |
| φ | 24 | sqrt+div, 2.192, 1.6e-16 | 1.681 (-23%), 5.4e-12 | 1.345 (-39%), 2.8e-14 |
| φ | 32 | sqrt+div, 2.196, 1.6e-16 | 1.677 (-24%), 4.2e-12 | 1.347 (-39%), 2.4e-14 |
| φ | 64 | sqrt+div, 2.200, 2.4e-16 | 1.687 (-23%), 3.2e-12 | 1.349 (-39%), 2.1e-14 |
| φ | 128 | sqrt+div, 2.204, 2.2e-16 | 1.688 (-23%), 2.5e-12 | 1.350 (-39%), 1.8e-14 |
| φ, ∇φ | 8 | sqrt+div, 1.276, 4.3e-16 | 1.090 (-15%), 4.0e-11 | 0.946 (-26%), 3.1e-13 |
| φ, ∇φ | 16 | sqrt+div, 1.296, 2.0e-16 | 1.105 (-15%), 4.3e-11 | 0.956 (-26%), 3.2e-13 |
| φ, ∇φ | 24 | sqrt+div, 1.301, 3.0e-16 | 1.110 (-15%), 3.1e-11 | 0.961 (-26%), 2.1e-13 |
| φ, ∇φ | 32 | sqrt+div, 1.304, 3.9e-16 | 1.112 (-15%), 7.4e-11 | 0.962 (-26%), 4.2e-13 |
| φ, ∇φ | 64 | sqrt+div, 1.308, 1.9e-15 | 1.115 (-15%), 2.9e-11 | 0.964 (-26%), 1.4e-13 |
| φ, ∇φ | 128 | sqrt+div, 1.307, 9.5e-16 | 1.114 (-15%), 5.5e-11 | 0.964 (-26%), 4.0e-13 |


## Inner loops, aarch64 release build (inner_loops.py)

| ISA | precision | output | entry | loop insns | FP | loads | spills | moves | shuffles | other | call | FP per pair-vector | model | FP / model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neon | f32 | grad | sil_t1_best | 22 | 18 | 2 | 0 | 0 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f32 | grad | sil_t1_gk | 27 | 23 | 2 | 0 | 0 | 0 | 2 | no | 23.0 | 22 (S2) | 1.05 |
| neon | f32 | grad | sil_t2_best | 40 | 36 | 2 | 0 | 0 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f32 | grad | sil_t2_gk | 50 | 46 | 2 | 0 | 0 | 0 | 2 | no | 23.0 | 22 (S2) | 1.05 |
| neon | f32 | grad | sil_t4_best | 92 | 72 | 2 | 6 | 10 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f32 | grad | sil_t4_gk | 140 | 92 | 2 | 10 | 34 | 0 | 2 | no | 23.0 | 22 (S2) | 1.05 |
| neon | f32 | grad | til_k1_best | 22 | 18 | 2 | 0 | 0 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f32 | grad | til_k1_gk | 27 | 23 | 2 | 0 | 0 | 0 | 2 | no | 23.0 | 22 (S2) | 1.05 |
| neon | f32 | grad | til_k2_best | 40 | 36 | 2 | 0 | 0 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f32 | grad | til_k2_c_n2 | 56 | 48 | 2 | 0 | 0 | 0 | 6 | no | 24.0 | 23 (N2) | 1.04 |
| neon | f32 | grad | til_k2_c_n3 | 64 | 54 | 2 | 0 | 0 | 0 | 8 | no | 27.0 | 26 (N3) | 1.04 |
| neon | f32 | grad | til_k2_c_p2 | 52 | 44 | 2 | 0 | 2 | 0 | 4 | no | 22.0 | 21 (P2) | 1.05 |
| neon | f32 | grad | til_k2_c_p3 | 56 | 46 | 2 | 0 | 4 | 0 | 4 | no | 23.0 | 22 (P3) | 1.05 |
| neon | f32 | grad | til_k2_c_p4 | 60 | 48 | 2 | 0 | 6 | 0 | 4 | no | 24.0 | 23 (P4) | 1.04 |
| neon | f32 | grad | til_k2_c_s2 | 50 | 46 | 2 | 0 | 0 | 0 | 2 | no | 23.0 | 22 (S2) | 1.05 |
| neon | f32 | grad | til_k2_c_s3 | 56 | 52 | 2 | 0 | 0 | 0 | 2 | no | 26.0 | 25 (S3) | 1.04 |
| neon | f32 | grad | til_k4_best | 82 | 72 | 2 | 6 | 0 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f32 | grad | til_k4_gk | 104 | 92 | 2 | 8 | 0 | 0 | 2 | no | 23.0 | 22 (S2) | 1.05 |
| neon | f32 | pot | sil_t1_best | 15 | 11 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f32 | pot | sil_t1_gk | 20 | 16 | 2 | 0 | 0 | 0 | 2 | no | 16.0 | 16 (S2) | 1.00 |
| neon | f32 | pot | sil_t2_best | 26 | 22 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f32 | pot | sil_t2_gk | 36 | 32 | 2 | 0 | 0 | 0 | 2 | no | 16.0 | 16 (S2) | 1.00 |
| neon | f32 | pot | sil_t4_best | 48 | 44 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f32 | pot | sil_t4_gk | 68 | 64 | 2 | 0 | 0 | 0 | 2 | no | 16.0 | 16 (S2) | 1.00 |
| neon | f32 | pot | til_k1_best | 15 | 11 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f32 | pot | til_k1_gk | 20 | 16 | 2 | 0 | 0 | 0 | 2 | no | 16.0 | 16 (S2) | 1.00 |
| neon | f32 | pot | til_k2_best | 26 | 22 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f32 | pot | til_k2_c_n2 | 42 | 34 | 2 | 0 | 0 | 0 | 6 | no | 17.0 | 17 (N2) | 1.00 |
| neon | f32 | pot | til_k2_c_n3 | 50 | 40 | 2 | 0 | 0 | 0 | 8 | no | 20.0 | 20 (N3) | 1.00 |
| neon | f32 | pot | til_k2_c_p2 | 38 | 30 | 2 | 0 | 2 | 0 | 4 | no | 15.0 | 15 (P2) | 1.00 |
| neon | f32 | pot | til_k2_c_p3 | 42 | 32 | 2 | 0 | 4 | 0 | 4 | no | 16.0 | 16 (P3) | 1.00 |
| neon | f32 | pot | til_k2_c_p4 | 46 | 34 | 2 | 0 | 6 | 0 | 4 | no | 17.0 | 17 (P4) | 1.00 |
| neon | f32 | pot | til_k2_c_s2 | 36 | 32 | 2 | 0 | 0 | 0 | 2 | no | 16.0 | 16 (S2) | 1.00 |
| neon | f32 | pot | til_k2_c_s3 | 42 | 38 | 2 | 0 | 0 | 0 | 2 | no | 19.0 | 19 (S3) | 1.00 |
| neon | f32 | pot | til_k4_best | 48 | 44 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f32 | pot | til_k4_gk | 68 | 64 | 2 | 0 | 0 | 0 | 2 | no | 16.0 | 16 (S2) | 1.00 |
| neon | f64 | grad | sil_t1_best | 22 | 18 | 2 | 0 | 0 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f64 | grad | sil_t1_gk | 30 | 26 | 2 | 0 | 0 | 0 | 2 | no | 26.0 | 25 (S3) | 1.04 |
| neon | f64 | grad | sil_t2_best | 40 | 36 | 2 | 0 | 0 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f64 | grad | sil_t2_gk | 56 | 52 | 2 | 0 | 0 | 0 | 2 | no | 26.0 | 25 (S3) | 1.04 |
| neon | f64 | grad | sil_t4_best | 90 | 72 | 2 | 4 | 10 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f64 | grad | sil_t4_gk | 133 | 104 | 2 | 10 | 15 | 0 | 2 | no | 26.0 | 25 (S3) | 1.04 |
| neon | f64 | grad | til_k1_best | 22 | 18 | 2 | 0 | 0 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f64 | grad | til_k1_gk | 30 | 26 | 2 | 0 | 0 | 0 | 2 | no | 26.0 | 25 (S3) | 1.04 |
| neon | f64 | grad | til_k1_rel_n1p2 | 33 | 26 | 2 | 0 | 3 | 0 | 2 | no | 26.0 | 25 (N1P2) | 1.04 |
| neon | f64 | grad | til_k1_rel_p4 | 32 | 24 | 2 | 0 | 4 | 0 | 2 | no | 24.0 | 23 (P4) | 1.04 |
| neon | f64 | grad | til_k1_rel_s2 | 27 | 23 | 2 | 0 | 0 | 0 | 2 | no | 23.0 | 22 (S2) | 1.05 |
| neon | f64 | grad | til_k2_best | 40 | 36 | 2 | 0 | 0 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f64 | grad | til_k2_c_n1p3 | 66 | 54 | 2 | 0 | 8 | 0 | 2 | no | 27.0 | 26 (N1P3) | 1.04 |
| neon | f64 | grad | til_k2_c_n3 | 64 | 54 | 2 | 0 | 6 | 0 | 2 | no | 27.0 | 26 (N3) | 1.04 |
| neon | f64 | grad | til_k2_c_n4 | 72 | 60 | 2 | 0 | 8 | 0 | 2 | no | 30.0 | 29 (N4) | 1.03 |
| neon | f64 | grad | til_k2_c_p6 | 68 | 52 | 2 | 0 | 12 | 0 | 2 | no | 26.0 | 25 (P6) | 1.04 |
| neon | f64 | grad | til_k2_c_p7 | 72 | 54 | 2 | 0 | 14 | 0 | 2 | no | 27.0 | 26 (P7) | 1.04 |
| neon | f64 | grad | til_k2_c_p8 | 81 | 56 | 2 | 4 | 17 | 0 | 2 | no | 28.0 | 27 (P8) | 1.04 |
| neon | f64 | grad | til_k2_c_s3 | 56 | 52 | 2 | 0 | 0 | 0 | 2 | no | 26.0 | 25 (S3) | 1.04 |
| neon | f64 | grad | til_k2_c_s4 | 62 | 58 | 2 | 0 | 0 | 0 | 2 | no | 29.0 | 28 (S4) | 1.04 |
| neon | f64 | grad | til_k2_rel_n1p2 | 62 | 52 | 2 | 0 | 6 | 0 | 2 | no | 26.0 | 25 (N1P2) | 1.04 |
| neon | f64 | grad | til_k2_rel_p4 | 60 | 48 | 2 | 0 | 8 | 0 | 2 | no | 24.0 | 23 (P4) | 1.04 |
| neon | f64 | grad | til_k2_rel_s2 | 50 | 46 | 2 | 0 | 0 | 0 | 2 | no | 23.0 | 22 (S2) | 1.05 |
| neon | f64 | grad | til_k4_best | 90 | 72 | 2 | 6 | 8 | 0 | 2 | no | 18.0 | 17 (SD) | 1.06 |
| neon | f64 | grad | til_k4_gk | 122 | 104 | 2 | 6 | 8 | 0 | 2 | no | 26.0 | 25 (S3) | 1.04 |
| neon | f64 | grad | til_k4_rel_n1p2 | 162 | 104 | 2 | 19 | 35 | 0 | 2 | no | 26.0 | 25 (N1P2) | 1.04 |
| neon | f64 | grad | til_k4_rel_p4 | 129 | 96 | 2 | 10 | 19 | 0 | 2 | no | 24.0 | 23 (P4) | 1.04 |
| neon | f64 | grad | til_k4_rel_s2 | 112 | 92 | 2 | 6 | 10 | 0 | 2 | no | 23.0 | 22 (S2) | 1.05 |
| neon | f64 | pot | sil_t1_best | 15 | 11 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f64 | pot | sil_t1_gk | 23 | 19 | 2 | 0 | 0 | 0 | 2 | no | 19.0 | 19 (S3) | 1.00 |
| neon | f64 | pot | sil_t2_best | 26 | 22 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f64 | pot | sil_t2_gk | 42 | 38 | 2 | 0 | 0 | 0 | 2 | no | 19.0 | 19 (S3) | 1.00 |
| neon | f64 | pot | sil_t4_best | 48 | 44 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f64 | pot | sil_t4_gk | 80 | 76 | 2 | 0 | 0 | 0 | 2 | no | 19.0 | 19 (S3) | 1.00 |
| neon | f64 | pot | til_k1_best | 15 | 11 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f64 | pot | til_k1_gk | 23 | 19 | 2 | 0 | 0 | 0 | 2 | no | 19.0 | 19 (S3) | 1.00 |
| neon | f64 | pot | til_k1_rel_n1p2 | 26 | 19 | 2 | 0 | 3 | 0 | 2 | no | 19.0 | 19 (N1P2) | 1.00 |
| neon | f64 | pot | til_k1_rel_p4 | 25 | 17 | 2 | 0 | 4 | 0 | 2 | no | 17.0 | 17 (P4) | 1.00 |
| neon | f64 | pot | til_k1_rel_s2 | 20 | 16 | 2 | 0 | 0 | 0 | 2 | no | 16.0 | 16 (S2) | 1.00 |
| neon | f64 | pot | til_k2_best | 26 | 22 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f64 | pot | til_k2_c_n1p3 | 52 | 40 | 2 | 0 | 8 | 0 | 2 | no | 20.0 | 20 (N1P3) | 1.00 |
| neon | f64 | pot | til_k2_c_n3 | 50 | 40 | 2 | 0 | 6 | 0 | 2 | no | 20.0 | 20 (N3) | 1.00 |
| neon | f64 | pot | til_k2_c_n4 | 58 | 46 | 2 | 0 | 8 | 0 | 2 | no | 23.0 | 23 (N4) | 1.00 |
| neon | f64 | pot | til_k2_c_p6 | 54 | 38 | 2 | 0 | 12 | 0 | 2 | no | 19.0 | 19 (P6) | 1.00 |
| neon | f64 | pot | til_k2_c_p7 | 58 | 40 | 2 | 0 | 14 | 0 | 2 | no | 20.0 | 20 (P7) | 1.00 |
| neon | f64 | pot | til_k2_c_p8 | 62 | 42 | 2 | 0 | 16 | 0 | 2 | no | 21.0 | 21 (P8) | 1.00 |
| neon | f64 | pot | til_k2_c_s3 | 42 | 38 | 2 | 0 | 0 | 0 | 2 | no | 19.0 | 19 (S3) | 1.00 |
| neon | f64 | pot | til_k2_c_s4 | 48 | 44 | 2 | 0 | 0 | 0 | 2 | no | 22.0 | 22 (S4) | 1.00 |
| neon | f64 | pot | til_k2_rel_n1p2 | 48 | 38 | 2 | 0 | 6 | 0 | 2 | no | 19.0 | 19 (N1P2) | 1.00 |
| neon | f64 | pot | til_k2_rel_p4 | 46 | 34 | 2 | 0 | 8 | 0 | 2 | no | 17.0 | 17 (P4) | 1.00 |
| neon | f64 | pot | til_k2_rel_s2 | 36 | 32 | 2 | 0 | 0 | 0 | 2 | no | 16.0 | 16 (S2) | 1.00 |
| neon | f64 | pot | til_k4_best | 48 | 44 | 2 | 0 | 0 | 0 | 2 | no | 11.0 | 11 (SD) | 1.00 |
| neon | f64 | pot | til_k4_gk | 80 | 76 | 2 | 0 | 0 | 0 | 2 | no | 19.0 | 19 (S3) | 1.00 |
| neon | f64 | pot | til_k4_rel_n1p2 | 105 | 76 | 2 | 1 | 24 | 0 | 2 | no | 19.0 | 19 (N1P2) | 1.00 |
| neon | f64 | pot | til_k4_rel_p4 | 105 | 68 | 2 | 1 | 32 | 0 | 2 | no | 17.0 | 17 (P4) | 1.00 |
| neon | f64 | pot | til_k4_rel_s2 | 68 | 64 | 2 | 0 | 0 | 0 | 2 | no | 16.0 | 16 (S2) | 1.00 |

Entries missing from the table were merged by LLVM with an identical function (for example a `gk` entry with the `cand` entry of the same formulation and K); the surviving name is listed.

## Inner loops, x86_64-apple-darwin release build (inner_loops.py; compiled on the M3 Max, not run)

| ISA | precision | output | entry | loop insns | FP | loads | spills | moves | shuffles | other | call | FP per pair-vector | model | FP / model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avx2 | f32 | grad | sil_t1_best | 55 | 21 | 25 | 1 | 0 | 3 | 5 | no | 21.0 | 21 (P2) | 1.00 |
| avx2 | f32 | grad | sil_t1_gk | 54 | 20 | 25 | 0 | 0 | 3 | 6 | no | 20.0 | 20 (S1) | 1.00 |
| avx2 | f32 | grad | sil_t2_best | 104 | 42 | 29 | 21 | 1 | 3 | 8 | no | 21.0 | 21 (P2) | 1.00 |
| avx2 | f32 | grad | sil_t2_gk | 99 | 40 | 27 | 21 | 0 | 3 | 8 | no | 20.0 | 20 (S1) | 1.00 |
| avx2 | f32 | grad | sil_t4_best | 184 | 84 | 34 | 50 | 2 | 3 | 11 | no | 21.0 | 21 (P2) | 1.00 |
| avx2 | f32 | grad | sil_t4_gk | 171 | 80 | 31 | 46 | 0 | 3 | 11 | no | 20.0 | 20 (S1) | 1.00 |
| avx2 | f32 | grad | til_k1_best | 33 | 21 | 5 | 1 | 0 | 0 | 6 | no | 21.0 | 21 (P2) | 1.00 |
| avx2 | f32 | grad | til_k1_gk | 31 | 20 | 5 | 0 | 0 | 0 | 6 | no | 20.0 | 20 (S1) | 1.00 |
| avx2 | f32 | grad | til_k2_best | 82 | 42 | 10 | 14 | 8 | 0 | 8 | no | 21.0 | 21 (P2) | 1.00 |
| avx2 | f32 | grad | til_k2_gk | 70 | 40 | 7 | 15 | 0 | 0 | 8 | no | 20.0 | 20 (S1) | 1.00 |
| avx2 | f32 | grad | til_k4_best | 165 | 84 | 12 | 56 | 3 | 0 | 10 | no | 21.0 | 21 (P2) | 1.00 |
| avx2 | f32 | grad | til_k4_gk | 155 | 80 | 10 | 52 | 2 | 0 | 11 | no | 20.0 | 20 (S1) | 1.00 |
| avx2 | f32 | pot | sil_t1_best | 46 | 15 | 24 | 0 | 0 | 3 | 4 | no | 15.0 | 15 (P2) | 1.00 |
| avx2 | f32 | pot | sil_t1_gk | 45 | 14 | 24 | 0 | 0 | 3 | 4 | no | 14.0 | 14 (S1) | 1.00 |
| avx2 | f32 | pot | sil_t2_best | 69 | 30 | 25 | 2 | 4 | 3 | 5 | no | 15.0 | 15 (P2) | 1.00 |
| avx2 | f32 | pot | sil_t2_gk | 66 | 28 | 25 | 1 | 4 | 3 | 5 | no | 14.0 | 14 (S1) | 1.00 |
| avx2 | f32 | pot | sil_t4_best | 123 | 60 | 30 | 15 | 10 | 3 | 5 | no | 15.0 | 15 (P2) | 1.00 |
| avx2 | f32 | pot | sil_t4_gk | 110 | 56 | 27 | 15 | 4 | 3 | 5 | no | 14.0 | 14 (S1) | 1.00 |
| avx2 | f32 | pot | til_k1_best | 23 | 15 | 4 | 0 | 0 | 0 | 4 | no | 15.0 | 15 (P2) | 1.00 |
| avx2 | f32 | pot | til_k1_gk | 22 | 14 | 4 | 0 | 0 | 0 | 4 | no | 14.0 | 14 (S1) | 1.00 |
| avx2 | f32 | pot | til_k2_best | 43 | 30 | 5 | 2 | 2 | 0 | 4 | no | 15.0 | 15 (P2) | 1.00 |
| avx2 | f32 | pot | til_k2_gk | 40 | 28 | 7 | 1 | 0 | 0 | 4 | no | 14.0 | 14 (S1) | 1.00 |
| avx2 | f32 | pot | til_k4_best | 90 | 60 | 13 | 12 | 0 | 0 | 5 | no | 15.0 | 15 (P2) | 1.00 |
| avx2 | f32 | pot | til_k4_gk | 81 | 56 | 8 | 12 | 0 | 0 | 5 | no | 14.0 | 14 (S1) | 1.00 |
| avx2 | f64 | grad | sil_t1_best | 69 | 26 | 15 | 3 | 16 | 3 | 6 | no | 26.0 | 26 (P5) | 1.00 |
| avx2 | f64 | grad | sil_t1_gk | 47 | 25 | 13 | 0 | 0 | 3 | 6 | no | 25.0 | 25 (S2) | 1.00 |
| avx2 | f64 | grad | sil_t2_best | 108 | 52 | 25 | 18 | 2 | 3 | 8 | no | 26.0 | 26 (P5) | 1.00 |
| avx2 | f64 | grad | sil_t2_gk | 97 | 50 | 16 | 18 | 2 | 3 | 8 | no | 25.0 | 25 (S2) | 1.00 |
| avx2 | f64 | grad | sil_t4_best | 209 | 104 | 30 | 57 | 5 | 3 | 10 | no | 26.0 | 26 (P5) | 1.00 |
| avx2 | f64 | grad | sil_t4_gk | 185 | 100 | 16 | 51 | 3 | 3 | 12 | no | 25.0 | 25 (S2) | 1.00 |
| avx2 | f64 | grad | til_k1_best | 41 | 26 | 5 | 4 | 1 | 0 | 5 | no | 26.0 | 26 (P5) | 1.00 |
| avx2 | f64 | grad | til_k1_gk | 36 | 25 | 4 | 1 | 0 | 0 | 6 | no | 25.0 | 25 (S2) | 1.00 |
| avx2 | f64 | grad | til_k2_best | 94 | 52 | 16 | 18 | 0 | 0 | 8 | no | 26.0 | 26 (P5) | 1.00 |
| avx2 | f64 | grad | til_k2_gk | 82 | 50 | 7 | 15 | 2 | 0 | 8 | no | 25.0 | 25 (S2) | 1.00 |
| avx2 | f64 | grad | til_k4_best | 201 | 104 | 21 | 60 | 5 | 0 | 11 | no | 26.0 | 26 (P5) | 1.00 |
| avx2 | f64 | grad | til_k4_gk | 177 | 100 | 10 | 56 | 0 | 0 | 11 | no | 25.0 | 25 (S2) | 1.00 |
| avx2 | f64 | pot | sil_t1_best | 51 | 20 | 12 | 0 | 11 | 3 | 5 | no | 20.0 | 20 (P5) | 1.00 |
| avx2 | f64 | pot | sil_t1_gk | 38 | 19 | 12 | 0 | 0 | 3 | 4 | no | 19.0 | 19 (S2) | 1.00 |
| avx2 | f64 | pot | sil_t2_best | 68 | 40 | 13 | 5 | 2 | 3 | 5 | no | 20.0 | 20 (P5) | 1.00 |
| avx2 | f64 | pot | sil_t2_gk | 70 | 38 | 14 | 1 | 8 | 3 | 6 | no | 19.0 | 19 (S2) | 1.00 |
| avx2 | f64 | pot | sil_t4_best | 135 | 80 | 20 | 25 | 2 | 3 | 5 | no | 20.0 | 20 (P5) | 1.00 |
| avx2 | f64 | pot | sil_t4_gk | 120 | 76 | 15 | 16 | 4 | 3 | 6 | no | 19.0 | 19 (S2) | 1.00 |
| avx2 | f64 | pot | til_k1_best | 29 | 20 | 4 | 0 | 1 | 0 | 4 | no | 20.0 | 20 (P5) | 1.00 |
| avx2 | f64 | pot | til_k1_gk | 27 | 19 | 4 | 0 | 0 | 0 | 4 | no | 19.0 | 19 (S2) | 1.00 |
| avx2 | f64 | pot | til_k2_best | 56 | 40 | 4 | 6 | 2 | 0 | 4 | no | 20.0 | 20 (P5) | 1.00 |
| avx2 | f64 | pot | til_k2_gk | 47 | 38 | 4 | 1 | 0 | 0 | 4 | no | 19.0 | 19 (S2) | 1.00 |
| avx2 | f64 | pot | til_k4_best | 122 | 80 | 22 | 15 | 0 | 0 | 5 | no | 20.0 | 20 (P5) | 1.00 |
| avx2 | f64 | pot | til_k4_gk | 100 | 76 | 7 | 12 | 0 | 0 | 5 | no | 19.0 | 19 (S2) | 1.00 |

Entries missing from the table were merged by LLVM with an identical function (for example a `gk` entry with the `cand` entry of the same formulation and K); the surviving name is listed.

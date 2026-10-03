# Device arithmetic (Phase 4 T3): raw output

- Machine: Apple M3 Max (16 physical, 16 logical cores; performance cores Some(12))
- Build: aarch64-macos, release build, rustc 1.99.0 (b940084d7 2026-09-28)
- CubeCL 0.11.0-pre.4 (pinned)
- Environment: OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS=1
- Sections: primitives, compiler, domain, p2p, cpu-p2p, leafops
- metal: runtime `wgpu<msl>`, f64 supported = false, plane size 32..32, max shared memory 32768 B, max units per cube 1024
- cpu: runtime `cpu`, f64 supported = true, plane size 1..1, max shared memory 65536 B, max units per cube 16

## Primitive accuracy: metal

| backend | precision | operation | inputs | max error (u_T) | at x | ≠ host |
| --- | --- | --- | --- | --- | --- | --- |
| metal | f32 | `sqrt(x)` | [1, 4) exhaustive (16777216 values) | 1.783 | 3.875714e0 | 4769584 of 16777216 |
| metal | f32 | `a / x` | [1, 4) exhaustive (16777216 values) | 2.257 | 1.930468e0 | 4769283 of 16777216 |
| metal | f32 | `1 / x` | [1, 4) exhaustive (16777216 values) | 1.285 | 1.927701e0 | 1736348 of 16777216 |
| metal | f32 | `recip(x)` | [1, 4) exhaustive (16777216 values) | 1.285 | 1.927701e0 | 1736348 of 16777216 |
| metal | f32 | `inverse_sqrt(x)` | [1, 4) exhaustive (16777216 values) | 1.500 | 3.998047e0 | 4416128 of 16777216 |
| metal | f32 | `inverse_sqrt(x) + Newton` | [1, 4) exhaustive (16777216 values) | 1.244 | 3.974914e0 | 5467899 of 16777216 |
| metal | f32 | `1 / sqrt(x)` | [1, 4) exhaustive (16777216 values) | 1.500 | 3.998047e0 | 4416128 of 16777216 |
| metal | f32 | `sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.651 | 3.145134e-27 | 13345 of 51968 |
| metal | f32 | `a / x` | 2^k m, k = −108..7 (51968 values) | 2.123 | 1.813348e-28 | 13814 of 51968 |
| metal | f32 | `1 / x` | 2^k m, k = −108..7 (51968 values) | 1.269 | 1.946692e-28 | 3086 of 51968 |
| metal | f32 | `recip(x)` | 2^k m, k = −108..7 (51968 values) | 1.269 | 1.946692e-28 | 3086 of 51968 |
| metal | f32 | `inverse_sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.249 | 9.462347e-4 | 15886 of 51968 |
| metal | f32 | `inverse_sqrt(x) + Newton` | 2^k m, k = −108..7 (51968 values) | 1.217 | 4.811375e-32 | 16659 of 51968 |
| metal | f32 | `1 / sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.249 | 9.462347e-4 | 15886 of 51968 |

"≠ host": results whose bits differ from the correctly rounded host value (`sqrt`, divisions) or from the host's fl(1/fl(√x)) (inverse square roots).

Edge cases (metal, f32; a = 1):

| x | `sqrt(x)` | `a / x` | `1 / x` | `recip(x)` | `inverse_sqrt(x)` | `inverse_sqrt(x) + Newton` | `1 / sqrt(x)` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +0 | +0 | +inf | +inf | +inf | +inf | **NaN** (IEEE +inf) | +inf |
| -0 | -0 | -inf | -inf | -inf | -inf | **NaN** (IEEE -inf) | -inf |
| min normal | 1.084202e-19 | 8.507059e37 | 8.507059e37 | 8.507059e37 | 9.223372e18 | **1.383506e19** (IEEE 9.223372e18) | 9.223372e18 |
| subnormal | **+0** (IEEE 3.833233e-20) | +inf | +inf | +inf | **+inf** (IEEE 2.608764e19) | **NaN** (IEEE 2.608764e19) | **+inf** (IEEE 2.608764e19) |
| +inf | +inf | +0 | +0 | +0 | +0 | **NaN** (IEEE +0) | +0 |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

## Compiler behaviour: metal

| backend | precision | probe | device | IEEE as written | verdict |
| --- | --- | --- | --- | --- | --- |
| metal | f32 | a·b + c | 5.960464477539063e-8 | 0e0 | **fused** |
| metal | f32 | c + a·b | 5.960464477539063e-8 | 0e0 | **fused** |
| metal | f32 | a·b − c | 5.960464477539063e-8 | 0e0 | **fused** |
| metal | f32 | c − a·b | -5.960464477539063e-8 | 0e0 | **fused** |
| metal | f32 | p = a·b (two uses); p + c | 0e0 | 0e0 | as written |
| metal | f32 | p (the second use) | 1.00048828125e0 | 1.00048828125e0 | as written |
| metal | f32 | fma(a, b, c) explicit | 5.960464477539063e-8 | 5.960464477539063e-8 | as written |
| metal | f32 | a·a + (−a)·a | 5.960464477539063e-8 | 0e0 | **fused (either product)** |
| metal | f32 | (x + b) − x (one value x = 1) | 9.313225746154785e-10 | 0e0 | **simplified to b** |
| metal | f32 | x − (x + z) (one value x = 1) | -9.313225746154785e-10 | 0e0 | **simplified to −z** |
| metal | f32 | (x − y) − z (control) | -9.313225746154785e-10 | -9.313225746154785e-10 | as written |
| metal | f32 | 1.5 min − 1.25 min (subnormal out) | 0e0 | 2.938735877055719e-39 | **flushed** |
| metal | f32 | small · small (subnormal out) | 0e0 | 7.346839692639297e-40 | **flushed** |
| metal | f32 | subnormal · 1 (subnormal in) | 0e0 | 1.4693679385278594e-39 | **flushed** |
| metal | f32 | subnormal + subnormal | 0e0 | 2.938735877055719e-39 | **flushed** |
| metal | f32 | select(−0 == 0, 1, 2) | 1e0 | 1e0 | as written |
| metal | f32 | select(subnormal == 0, 1, 2) | 1e0 | 2e0 | **flushed in compare** |
| metal | f32 | q · select(r² == 0, 0, 1/√r²), r² = 0 | 0e0 | 0e0 | as written |
| metal | f32 | t = q/√r²; if r² == 0 { t = 0 } | 0e0 | 0e0 | as written |
| metal | f32 | ∞ − ∞ (same value) | 0e0 | NaN | **folded x − x → 0** |
| metal | f32 | −0 + 0.0 (literal) | -0e0 | 0e0 | **folded x + 0 → x** |
| metal | f32 | (x + b) − y (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |
| metal | f32 | x − (y + z) (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |

## The §3.13 argument on the device: metal

metal f32 adversarial pairs: 209303 pairs, 30403 coincident (30385 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-53.00. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 164601 components and gain or lose a zero in 123571.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 209042 | 209134 |
| fma forward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |
| fma backward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208420 | 208140 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |

metal f32 random pairs: 216000 pairs, 3526 coincident (0 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-17.70. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 126828 components and gain or lose a zero in 0.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169414 | 184146 |
| fma forward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |
| fma backward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 152271 | 153148 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |


## P2P candidates: metal

Pair terms against `nd_fmm_ref::p2p` (1000000 pairs over the kernel domain; gradient over r² ≥ 2^-84), and sums against `direct_sum` on W1 (worst over n_t ∈ [8, 32, 64, 128], 4 sets each, with gradients; relative to the term magnitudes; the reference's own error in brackets):

| backend | precision | candidate | φ term (u_T) | ∇φ term (u_T) | φ, ∇φ term, nonzero |dₖ| ≥ 2^-53 (u_T) | φ sum | ∇φ sum | sum check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metal | f32 | sqrt + div, as written | 3.974 | 8.629 | 3.974, 8.629 | 1.16e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| metal | f32 | sqrt + div, explicit fma | 6.606 | 8.621 | 3.974, 8.621 | 1.21e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| metal | f32 | inverse_sqrt, as written | 3.974 | 8.629 | 3.974, 8.629 | 1.16e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| metal | f32 | inverse_sqrt, explicit fma | 6.606 | 8.621 | 3.974, 8.621 | 1.21e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| metal | f32 | inverse_sqrt + Newton, as written | 3.974 | 9.124 | 3.974, 9.124 | 1.17e-7 [1.28e-7] | 5.17e-7 [6.23e-7] | pass |
| metal | f32 | inverse_sqrt + Newton, explicit fma | 7.928 | 8.905 | 3.974, 8.905 | 1.17e-7 [1.28e-7] | 5.17e-7 [6.23e-7] | pass |

Quick throughput, Gpairs/s (4096 target leaves per launch, one unit per target, 64 per cube, launches queued between syncs, compilation excluded, median of 15 batches of at least 20 ms):

| backend | precision | candidate | output | n_t = 8 | n_t = 32 | n_t = 64 | n_t = 128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metal | f32 | sqrt + div, as written | φ | 185.0 | 235.8 | 249.5 | 256.9 |
| metal | f32 | sqrt + div, as written | φ, ∇φ | 153.8 | 199.7 | 211.8 | 218.7 |
| metal | f32 | sqrt + div, explicit fma | φ | 175.3 | 224.5 | 240.7 | 249.0 |
| metal | f32 | sqrt + div, explicit fma | φ, ∇φ | 151.6 | 193.9 | 205.1 | 209.3 |
| metal | f32 | inverse_sqrt, as written | φ | 183.9 | 234.8 | 247.7 | 249.5 |
| metal | f32 | inverse_sqrt, as written | φ, ∇φ | 158.7 | 193.8 | 199.7 | 202.6 |
| metal | f32 | inverse_sqrt, explicit fma | φ | 178.0 | 221.6 | 229.0 | 231.3 |
| metal | f32 | inverse_sqrt, explicit fma | φ, ∇φ | 147.2 | 182.2 | 188.7 | 191.8 |
| metal | f32 | inverse_sqrt + Newton, as written | φ | 168.3 | 205.5 | 214.1 | 217.1 |
| metal | f32 | inverse_sqrt + Newton, as written | φ, ∇φ | 136.9 | 167.0 | 173.9 | 177.7 |
| metal | f32 | inverse_sqrt + Newton, explicit fma | φ | 161.7 | 196.4 | 205.8 | 209.3 |
| metal | f32 | inverse_sqrt + Newton, explicit fma | φ, ∇φ | 133.0 | 162.6 | 169.5 | 173.6 |

## Harmonics and GEMM: metal

| backend | precision | test | bit-identical to host | max diff from host (u_T) | device vs f64 (u_T) | host vs f64 (u_T) |
| --- | --- | --- | --- | --- | --- | --- |
| metal | f32 | regular harmonics, p = 8 | 34530 of 81000 | 48.399 (48.399 where normal; 0 host values subnormal) | 27.761 | 43.965 |
| metal | f32 | irregular harmonics, p = 8 | 20387 of 81000 | 60.431 (60.431 where normal; 0 host values subnormal) | 40.568 | 69.803 |
| metal | f32 | regular harmonics, p = 20 | 88943 of 441000 | 1186862.523 (273.571 where normal; 351 host values subnormal) | 1186862.110 | 143.170 |
| metal | f32 | irregular harmonics, p = 20 | 66689 of 441000 | 267.948 (267.948 where normal; 0 host values subnormal) | 317.364 | 243.180 |
| metal | f32 | y += A x, n = 81, 512 columns | 10724 of 41472 (= fused host: 41472) | 3.489 | – | – |

Harmonics errors per point and degree, relative to the largest |value| of that degree; GEMM relative to max |y|.

## Primitive accuracy: cpu

| backend | precision | operation | inputs | max error (u_T) | at x | ≠ host |
| --- | --- | --- | --- | --- | --- | --- |
| cpu | f32 | `sqrt(x)` | [1, 4) exhaustive (16777216 values) | 1.000 | 1.000000e0 | 0 of 16777216 |
| cpu | f32 | `a / x` | [1, 4) exhaustive (16777216 values) | 1.000 | 2.731115e0 | 0 of 16777216 |
| cpu | f32 | `1 / x` | [1, 4) exhaustive (16777216 values) | 1.000 | 2.000000e0 | 0 of 16777216 |
| cpu | f32 | `recip(x)` | [1, 4) exhaustive (16777216 values) | 1.000 | 2.000000e0 | 0 of 16777216 |
| cpu | f32 | `inverse_sqrt(x)` | [1, 4) exhaustive (16777216 values) | 1.500 | 4.000000e0 | 0 of 16777216 |
| cpu | f32 | `inverse_sqrt(x) + Newton` | [1, 4) exhaustive (16777216 values) | 1.244 | 3.974914e0 | 5514308 of 16777216 |
| cpu | f32 | `1 / sqrt(x)` | [1, 4) exhaustive (16777216 values) | 1.500 | 4.000000e0 | 0 of 16777216 |
| cpu | f32 | `sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.000 | 3.081488e-33 | 0 of 51968 |
| cpu | f32 | `a / x` | 2^k m, k = −108..7 (51968 values) | 1.000 | 1.220701e-4 | 0 of 51968 |
| cpu | f32 | `1 / x` | 2^k m, k = −108..7 (51968 values) | 1.000 | 6.162975e-33 | 0 of 51968 |
| cpu | f32 | `recip(x)` | 2^k m, k = −108..7 (51968 values) | 1.000 | 6.162975e-33 | 0 of 51968 |
| cpu | f32 | `inverse_sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.500 | 1.232595e-32 | 0 of 51968 |
| cpu | f32 | `inverse_sqrt(x) + Newton` | 2^k m, k = −108..7 (51968 values) | 1.217 | 4.811375e-32 | 18578 of 51968 |
| cpu | f32 | `1 / sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.500 | 1.232595e-32 | 0 of 51968 |

"≠ host": results whose bits differ from the correctly rounded host value (`sqrt`, divisions) or from the host's fl(1/fl(√x)) (inverse square roots).

Edge cases (cpu, f32; a = 1):

| x | `sqrt(x)` | `a / x` | `1 / x` | `recip(x)` | `inverse_sqrt(x)` | `inverse_sqrt(x) + Newton` | `1 / sqrt(x)` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +0 | +0 | +inf | +inf | +inf | +inf | **NaN** (IEEE +inf) | +inf |
| -0 | -0 | -inf | -inf | -inf | -inf | **NaN** (IEEE -inf) | -inf |
| min normal | 1.084202e-19 | 8.507059e37 | 8.507059e37 | 8.507059e37 | 9.223372e18 | 9.223372e18 | 9.223372e18 |
| subnormal | 3.833233e-20 | +inf | +inf | +inf | 2.608764e19 | 2.608764e19 | 2.608764e19 |
| +inf | +inf | +0 | +0 | +0 | +0 | **NaN** (IEEE +0) | +0 |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| backend | precision | operation | inputs | max error (u_T) | at x | ≠ host |
| --- | --- | --- | --- | --- | --- | --- |
| cpu | f64 | `sqrt(x)` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.000 | 1.490889e-8 | 0 of 10000000 |
| cpu | f64 | `a / x` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.000 | 5.182455e-7 | 0 of 10000000 |
| cpu | f64 | `1 / x` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.000 | 3.387567e-21 | 0 of 10000000 |
| cpu | f64 | `recip(x)` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.000 | 3.387567e-21 | 0 of 10000000 |
| cpu | f64 | `inverse_sqrt(x)` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.497 | 3.088107e-33 | 0 of 10000000 |
| cpu | f64 | `inverse_sqrt(x) + Newton` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.248 | 5.958421e-8 | 3427150 of 10000000 |
| cpu | f64 | `1 / sqrt(x)` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.497 | 3.088107e-33 | 0 of 10000000 |
| cpu | f64 | `sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.000 | 3.081488e-33 | 0 of 51968 |
| cpu | f64 | `a / x` | 2^k m, k = −108..7 (51968 values) | 1.000 | 1.220703e-4 | 0 of 51968 |
| cpu | f64 | `1 / x` | 2^k m, k = −108..7 (51968 values) | 1.000 | 6.162976e-33 | 0 of 51968 |
| cpu | f64 | `recip(x)` | 2^k m, k = −108..7 (51968 values) | 1.000 | 6.162976e-33 | 0 of 51968 |
| cpu | f64 | `inverse_sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.500 | 1.232595e-32 | 0 of 51968 |
| cpu | f64 | `inverse_sqrt(x) + Newton` | 2^k m, k = −108..7 (51968 values) | 1.227 | 6.092721e-2 | 17074 of 51968 |
| cpu | f64 | `1 / sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.500 | 1.232595e-32 | 0 of 51968 |

"≠ host": results whose bits differ from the correctly rounded host value (`sqrt`, divisions) or from the host's fl(1/fl(√x)) (inverse square roots).

Edge cases (cpu, f64; a = 1):

| x | `sqrt(x)` | `a / x` | `1 / x` | `recip(x)` | `inverse_sqrt(x)` | `inverse_sqrt(x) + Newton` | `1 / sqrt(x)` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +0 | +0 | +inf | +inf | +inf | +inf | **NaN** (IEEE +inf) | +inf |
| -0 | -0 | -inf | -inf | -inf | -inf | **NaN** (IEEE -inf) | -inf |
| min normal | 1.491668e-154 | 4.494233e307 | 4.494233e307 | 4.494233e307 | 6.703904e153 | 6.703904e153 | 6.703904e153 |
| subnormal | 5.273843e-155 | +inf | +inf | +inf | 1.896150e154 | **1.896150e154** (IEEE 1.896150e154) | 1.896150e154 |
| +inf | +inf | +0 | +0 | +0 | +0 | **NaN** (IEEE +0) | +0 |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

## Compiler behaviour: cpu

| backend | precision | probe | device | IEEE as written | verdict |
| --- | --- | --- | --- | --- | --- |
| cpu | f32 | a·b + c | 5.960464477539063e-8 | 0e0 | **fused** |
| cpu | f32 | c + a·b | 5.960464477539063e-8 | 0e0 | **fused** |
| cpu | f32 | a·b − c | 5.960464477539063e-8 | 0e0 | **fused** |
| cpu | f32 | c − a·b | -5.960464477539063e-8 | 0e0 | **fused** |
| cpu | f32 | p = a·b (two uses); p + c | 0e0 | 0e0 | as written |
| cpu | f32 | p (the second use) | 1.00048828125e0 | 1.00048828125e0 | as written |
| cpu | f32 | fma(a, b, c) explicit | 5.960464477539063e-8 | 5.960464477539063e-8 | as written |
| cpu | f32 | a·a + (−a)·a | 5.960464477539063e-8 | 0e0 | **fused (either product)** |
| cpu | f32 | (x + b) − x (one value x = 1) | 0e0 | 0e0 | as written |
| cpu | f32 | x − (x + z) (one value x = 1) | 0e0 | 0e0 | as written |
| cpu | f32 | (x − y) − z (control) | -9.313225746154785e-10 | -9.313225746154785e-10 | as written |
| cpu | f32 | 1.5 min − 1.25 min (subnormal out) | 2.938735877055719e-39 | 2.938735877055719e-39 | as written |
| cpu | f32 | small · small (subnormal out) | 7.346839692639297e-40 | 7.346839692639297e-40 | as written |
| cpu | f32 | subnormal · 1 (subnormal in) | 1.4693679385278594e-39 | 1.4693679385278594e-39 | as written |
| cpu | f32 | subnormal + subnormal | 2.938735877055719e-39 | 2.938735877055719e-39 | as written |
| cpu | f32 | select(−0 == 0, 1, 2) | 1e0 | 1e0 | as written |
| cpu | f32 | select(subnormal == 0, 1, 2) | 2e0 | 2e0 | as written |
| cpu | f32 | q · select(r² == 0, 0, 1/√r²), r² = 0 | 0e0 | 0e0 | as written |
| cpu | f32 | t = q/√r²; if r² == 0 { t = 0 } | 0e0 | 0e0 | as written |
| cpu | f32 | ∞ − ∞ (same value) | 0e0 | NaN | **folded x − x → 0** |
| cpu | f32 | −0 + 0.0 (literal) | -0e0 | 0e0 | **folded x + 0 → x** |
| cpu | f32 | (x + b) − y (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |
| cpu | f32 | x − (y + z) (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |
| backend | precision | probe | device | IEEE as written | verdict |
| --- | --- | --- | --- | --- | --- |
| cpu | f64 | a·b + c | 5.551115123125783e-17 | 0e0 | **fused** |
| cpu | f64 | c + a·b | 5.551115123125783e-17 | 0e0 | **fused** |
| cpu | f64 | a·b − c | 5.551115123125783e-17 | 0e0 | **fused** |
| cpu | f64 | c − a·b | -5.551115123125783e-17 | 0e0 | **fused** |
| cpu | f64 | p = a·b (two uses); p + c | 0e0 | 0e0 | as written |
| cpu | f64 | p (the second use) | 1.0000000149011612e0 | 1.0000000149011612e0 | as written |
| cpu | f64 | fma(a, b, c) explicit | 5.551115123125783e-17 | 5.551115123125783e-17 | as written |
| cpu | f64 | a·a + (−a)·a | 5.551115123125783e-17 | 0e0 | **fused (either product)** |
| cpu | f64 | (x + b) − x (one value x = 1) | 0e0 | 0e0 | as written |
| cpu | f64 | x − (x + z) (one value x = 1) | 0e0 | 0e0 | as written |
| cpu | f64 | (x − y) − z (control) | -1.734723475976807e-18 | -1.734723475976807e-18 | as written |
| cpu | f64 | 1.5 min − 1.25 min (subnormal out) | 5.562684646268003e-309 | 5.562684646268003e-309 | as written |
| cpu | f64 | small · small (subnormal out) | 8.095e-320 | 8.095e-320 | as written |
| cpu | f64 | subnormal · 1 (subnormal in) | 2.781342323134e-309 | 2.781342323134e-309 | as written |
| cpu | f64 | subnormal + subnormal | 5.562684646268003e-309 | 5.562684646268003e-309 | as written |
| cpu | f64 | select(−0 == 0, 1, 2) | 1e0 | 1e0 | as written |
| cpu | f64 | select(subnormal == 0, 1, 2) | 2e0 | 2e0 | as written |
| cpu | f64 | q · select(r² == 0, 0, 1/√r²), r² = 0 | 0e0 | 0e0 | as written |
| cpu | f64 | t = q/√r²; if r² == 0 { t = 0 } | 0e0 | 0e0 | as written |
| cpu | f64 | ∞ − ∞ (same value) | 0e0 | NaN | **folded x − x → 0** |
| cpu | f64 | −0 + 0.0 (literal) | -0e0 | 0e0 | **folded x + 0 → x** |
| cpu | f64 | (x + b) − y (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |
| cpu | f64 | x − (y + z) (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |

## The §3.13 argument on the device: cpu

cpu f32 adversarial pairs: 209303 pairs, 30403 coincident (30385 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-53.00. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 164601 components and gain or lose a zero in 123571.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 209107 | 209081 |
| fma forward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |
| fma backward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208420 | 208140 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |

cpu f32 random pairs: 216000 pairs, 3526 coincident (0 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-17.70. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 126828 components and gain or lose a zero in 0.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 193764 | 188412 |
| fma forward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |
| fma backward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 152271 | 153148 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |

cpu f64 adversarial pairs: 439192 pairs, 484 coincident (154 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-53.00. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 3118 components and gain or lose a zero in 900.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 437883 | 437005 |
| fma forward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 435717 | 439192 |
| fma backward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 434096 | 432767 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 435717 | 439192 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 435717 | 439192 |

cpu f64 random pairs: 216000 pairs, 3526 coincident (0 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-17.69. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 1656 components and gain or lose a zero in 0.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 194026 | 188849 |
| fma forward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 170154 | 216000 |
| fma backward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 152728 | 153516 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 170154 | 216000 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 170154 | 216000 |


## P2P candidates: cpu

Pair terms against `nd_fmm_ref::p2p` (1000000 pairs over the kernel domain; gradient over r² ≥ 2^-84), and sums against `direct_sum` on W1 (worst over n_t ∈ [8, 32, 64, 128], 4 sets each, with gradients; relative to the term magnitudes; the reference's own error in brackets):

| backend | precision | candidate | φ term (u_T) | ∇φ term (u_T) | φ, ∇φ term, nonzero |dₖ| ≥ 2^-53 (u_T) | φ sum | ∇φ sum | sum check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | f32 | sqrt + div, as written | 3.997 | 10.296 | 3.997, 10.296 | 1.28e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| cpu | f32 | sqrt + div, explicit fma | 3.986 | 10.210 | 3.986, 10.210 | 1.23e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| cpu | f32 | inverse_sqrt, as written | 3.997 | 10.296 | 3.997, 10.296 | 1.28e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| cpu | f32 | inverse_sqrt, explicit fma | 3.986 | 10.210 | 3.986, 10.210 | 1.23e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| cpu | f32 | inverse_sqrt + Newton, as written | 4.321 | 9.296 | 4.321, 9.296 | 1.28e-7 [1.28e-7] | 5.41e-7 [6.23e-7] | pass |
| cpu | f32 | inverse_sqrt + Newton, explicit fma | 3.974 | 8.905 | 3.974, 8.905 | 1.11e-7 [1.28e-7] | 5.41e-7 [6.23e-7] | pass |
Pair terms against `nd_fmm_ref::p2p` (1000000 pairs over the kernel domain; gradient over r² ≥ 2^-108), and sums against `direct_sum` on W1 (worst over n_t ∈ [8, 32, 64, 128], 4 sets each, with gradients; relative to the term magnitudes; the reference's own error in brackets):

| backend | precision | candidate | φ term (u_T) | ∇φ term (u_T) | φ, ∇φ term, nonzero |dₖ| ≥ 2^-53 (u_T) | φ sum | ∇φ sum | sum check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | f64 | sqrt + div, as written | 3.998 | 10.195 | 3.998, 10.195 | 2.86e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cpu | f64 | sqrt + div, explicit fma | 4.054 | 9.657 | 4.054, 9.657 | 2.42e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cpu | f64 | inverse_sqrt, as written | 3.998 | 10.195 | 3.998, 10.195 | 2.86e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cpu | f64 | inverse_sqrt, explicit fma | 4.054 | 9.657 | 4.054, 9.657 | 2.42e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cpu | f64 | inverse_sqrt + Newton, as written | 4.606 | 9.296 | 4.606, 9.296 | 2.86e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cpu | f64 | inverse_sqrt + Newton, explicit fma | 4.000 | 9.268 | 4.000, 9.268 | 2.31e-16 [3.08e-16] | 1.46e-15 [1.75e-15] | pass |

## CPU-shaped P2P on the CPU runtime against nd-fmm-simd

W1 gathered, 768 target leaves per launch (the 64-set pool cycled); 1 unit against `P2pKernel::detect()` (neon) on one thread, 12 units against it on 12 rayon threads; kernel compilation excluded; median of 15 batches of at least 20 ms. Ratio = CPU-runtime time / nd-fmm-simd time.

f32: N = 4, K = 2; pair terms over the kernel domain (1000000 pairs): φ 3.995 u_T, ∇φ 9.252 u_T.

| precision | output | n_t | CPU runtime 1 unit (Gpairs/s) | nd-fmm-simd 1 thread | ratio (time) | CPU runtime 12 units | nd-fmm-simd 12 threads | ratio (time) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | φ | 8 | 3.681 | 3.739 | 1.02 | 7.431 | 20.008 | 2.69 |
| f32 | φ | 16 | 3.732 | 3.793 | 1.02 | 10.626 | 31.491 | 2.96 |
| f32 | φ | 24 | 3.721 | 3.791 | 1.02 | 14.338 | 34.676 | 2.42 |
| f32 | φ | 32 | 3.712 | 3.832 | 1.03 | 14.668 | 35.802 | 2.44 |
| f32 | φ | 64 | 3.745 | 3.840 | 1.03 | 17.088 | 37.997 | 2.22 |
| f32 | φ | 128 | 3.754 | 3.849 | 1.03 | 32.167 | 41.066 | 1.28 |
| f32 | φ, ∇φ | 8 | 2.424 | 2.450 | 1.01 | 3.310 | 20.078 | 6.07 |
| f32 | φ, ∇φ | 16 | 2.410 | 2.485 | 1.03 | 7.157 | 25.561 | 3.57 |
| f32 | φ, ∇φ | 24 | 2.430 | 2.494 | 1.03 | 7.841 | 26.629 | 3.40 |
| f32 | φ, ∇φ | 32 | 2.420 | 2.499 | 1.03 | 7.198 | 26.535 | 3.69 |
| f32 | φ, ∇φ | 64 | 2.435 | 2.504 | 1.03 | 11.669 | 28.369 | 2.43 |
| f32 | φ, ∇φ | 128 | 2.435 | 2.506 | 1.03 | 22.335 | 29.545 | 1.32 |

f32: outputs bit-identical to `nd-fmm-simd` (neon) on the checked sets of every cell: 4456 of 5440.

f64: N = 2, K = 4; pair terms over the kernel domain (1000000 pairs): φ 3.987 u_T, ∇φ 9.484 u_T.

| precision | output | n_t | CPU runtime 1 unit (Gpairs/s) | nd-fmm-simd 1 thread | ratio (time) | CPU runtime 12 units | nd-fmm-simd 12 threads | ratio (time) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 | φ | 8 | 2.037 | 2.040 | 1.00 | 1.468 | 13.649 | 9.30 |
| f64 | φ | 16 | 2.072 | 2.063 | 1.00 | 6.309 | 22.148 | 3.51 |
| f64 | φ | 24 | 2.060 | 2.048 | 0.99 | 6.008 | 22.871 | 3.81 |
| f64 | φ | 32 | 2.066 | 2.080 | 1.01 | 7.489 | 21.956 | 2.93 |
| f64 | φ | 64 | 2.082 | 2.081 | 1.00 | 12.051 | 23.731 | 1.97 |
| f64 | φ | 128 | 2.088 | 2.079 | 1.00 | 19.368 | 24.706 | 1.28 |
| f64 | φ, ∇φ | 8 | 1.220 | 1.157 | 0.95 | 1.921 | 10.869 | 5.66 |
| f64 | φ, ∇φ | 16 | 1.209 | 1.167 | 0.97 | 4.286 | 11.997 | 2.80 |
| f64 | φ, ∇φ | 24 | 1.203 | 1.175 | 0.98 | 3.840 | 11.754 | 3.06 |
| f64 | φ, ∇φ | 32 | 1.206 | 1.170 | 0.97 | 3.948 | 12.012 | 3.04 |
| f64 | φ, ∇φ | 64 | 1.207 | 1.176 | 0.97 | 6.180 | 12.605 | 2.04 |
| f64 | φ, ∇φ | 128 | 1.207 | 1.174 | 0.97 | 14.477 | 13.103 | 0.91 |

f64: outputs bit-identical to `nd-fmm-simd` (neon) on the checked sets of every cell: 4455 of 5440.

Geometric mean over 24 cells: one thread 1.004, all performance cores 2.729. Rule: at most 1.5 sets a CPU-runtime target; above 1.5 the CPU runtime stays correctness-only.

Per-launch overhead (f32, φ):

| launch | units | queued (µs per launch) | launch + sync (µs) |
| --- | --- | --- | --- |
| empty (no leaves) | 1 | 2.4 | 9.2 |
| empty (no leaves) | 12 | 190.9 | 27.5 |
| one target leaf, n_t = 64, f32 φ | 1 | 32.0 | 43.6 |
| one target leaf, n_t = 64, f32 φ | 12 | 230.9 | 51.3 |


## Harmonics and GEMM: cpu

| backend | precision | test | bit-identical to host | max diff from host (u_T) | device vs f64 (u_T) | host vs f64 (u_T) |
| --- | --- | --- | --- | --- | --- | --- |
| cpu | f32 | regular harmonics, p = 8 | 38963 of 81000 | 34.761 (34.761 where normal; 0 host values subnormal) | 24.877 | 43.965 |
| cpu | f32 | irregular harmonics, p = 8 | 32955 of 81000 | 74.922 (74.922 where normal; 0 host values subnormal) | 36.827 | 69.803 |
| cpu | f32 | regular harmonics, p = 20 | 115132 of 441000 | 216.509 (216.509 where normal; 351 host values subnormal) | 118.124 | 143.170 |
| cpu | f32 | irregular harmonics, p = 20 | 105338 of 441000 | 306.227 (306.227 where normal; 0 host values subnormal) | 202.057 | 243.180 |
| cpu | f32 | y += A x, n = 81, 512 columns | 10724 of 41472 (= fused host: 41472) | 3.489 | – | – |
| cpu | f64 | regular harmonics, p = 8 | 39240 of 81000 | 40.162 (40.162 where normal; 0 host values subnormal) | 40.162 | 0.000 |
| cpu | f64 | irregular harmonics, p = 8 | 32911 of 81000 | 65.340 (65.340 where normal; 0 host values subnormal) | 65.340 | 0.000 |
| cpu | f64 | regular harmonics, p = 20 | 117006 of 441000 | 106.625 (106.625 where normal; 0 host values subnormal) | 106.625 | 0.000 |
| cpu | f64 | irregular harmonics, p = 20 | 105935 of 441000 | 213.648 (213.648 where normal; 0 host values subnormal) | 213.648 | 0.000 |
| cpu | f64 | y += A x, n = 81, 512 columns | 10524 of 41472 (= fused host: 41472) | 3.172 | – | – |

Harmonics errors per point and degree, relative to the largest |value| of that degree; GEMM relative to max |y|.

## Summary

Every check passed on: metal, cpu.

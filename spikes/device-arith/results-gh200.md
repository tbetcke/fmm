# Device arithmetic (Phase 4 T3): raw output

- Machine: unknown (72 logical cores; performance cores None)
- Build: aarch64-linux, release build, rustc 1.99.0 (b940084d7 2026-09-28)
- CubeCL 0.11.0-pre.4 (pinned)
- Environment: OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, RAYON_NUM_THREADS=1
- Sections: primitives, compiler, domain, p2p, cpu-p2p, leafops
- cuda: device `NVIDIA GH200 480GB`, runtime `cuda`, compiler LLVM NVPTX, f64 supported = true, plane size 32..32, max shared memory 232448 B, max units per cube 1024

## Primitive accuracy: cuda

| backend | precision | operation | inputs | max error (u_T) | at x | ≠ host |
| --- | --- | --- | --- | --- | --- | --- |
| cuda | f32 | `sqrt(x)` | [1, 4) exhaustive (16777216 values) | 1.000 | 1.000000e0 | 0 of 16777216 |
| cuda | f32 | `a / x` | [1, 4) exhaustive (16777216 values) | 1.000 | 2.731115e0 | 0 of 16777216 |
| cuda | f32 | `1 / x` | [1, 4) exhaustive (16777216 values) | 1.000 | 2.000000e0 | 0 of 16777216 |
| cuda | f32 | `recip(x)` | [1, 4) exhaustive (16777216 values) | 1.000 | 2.000000e0 | 0 of 16777216 |
| cuda | f32 | `inverse_sqrt(x)` | [1, 4) exhaustive (16777216 values) | 1.500 | 4.000000e0 | 0 of 16777216 |
| cuda | f32 | `inverse_sqrt(x) + Newton` | [1, 4) exhaustive (16777216 values) | 1.244 | 3.974914e0 | 5514308 of 16777216 |
| cuda | f32 | `1 / sqrt(x)` | [1, 4) exhaustive (16777216 values) | 1.500 | 4.000000e0 | 0 of 16777216 |
| cuda | f32 | `sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.000 | 3.081488e-33 | 0 of 51968 |
| cuda | f32 | `a / x` | 2^k m, k = −108..7 (51968 values) | 1.000 | 1.220701e-4 | 0 of 51968 |
| cuda | f32 | `1 / x` | 2^k m, k = −108..7 (51968 values) | 1.000 | 6.162975e-33 | 0 of 51968 |
| cuda | f32 | `recip(x)` | 2^k m, k = −108..7 (51968 values) | 1.000 | 6.162975e-33 | 0 of 51968 |
| cuda | f32 | `inverse_sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.500 | 1.232595e-32 | 0 of 51968 |
| cuda | f32 | `inverse_sqrt(x) + Newton` | 2^k m, k = −108..7 (51968 values) | 1.217 | 4.811375e-32 | 18578 of 51968 |
| cuda | f32 | `1 / sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.500 | 1.232595e-32 | 0 of 51968 |

"≠ host": results whose bits differ from the correctly rounded host value (`sqrt`, divisions) or from the host's fl(1/fl(√x)) (inverse square roots).

Edge cases (cuda, f32; a = 1):

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
| cuda | f64 | `sqrt(x)` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.000 | 1.490889e-8 | 0 of 10000000 |
| cuda | f64 | `a / x` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.000 | 5.182455e-7 | 0 of 10000000 |
| cuda | f64 | `1 / x` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.000 | 3.387567e-21 | 0 of 10000000 |
| cuda | f64 | `recip(x)` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.000 | 3.387567e-21 | 0 of 10000000 |
| cuda | f64 | `inverse_sqrt(x)` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.497 | 3.088107e-33 | 0 of 10000000 |
| cuda | f64 | `inverse_sqrt(x) + Newton` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.248 | 5.958421e-8 | 3427247 of 10000000 |
| cuda | f64 | `1 / sqrt(x)` | log-uniform on [2^-108, 2^7] (10000000 samples) | 1.497 | 3.088107e-33 | 0 of 10000000 |
| cuda | f64 | `sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.000 | 3.081488e-33 | 0 of 51968 |
| cuda | f64 | `a / x` | 2^k m, k = −108..7 (51968 values) | 1.000 | 1.220703e-4 | 0 of 51968 |
| cuda | f64 | `1 / x` | 2^k m, k = −108..7 (51968 values) | 1.000 | 6.162976e-33 | 0 of 51968 |
| cuda | f64 | `recip(x)` | 2^k m, k = −108..7 (51968 values) | 1.000 | 6.162976e-33 | 0 of 51968 |
| cuda | f64 | `inverse_sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.500 | 1.232595e-32 | 0 of 51968 |
| cuda | f64 | `inverse_sqrt(x) + Newton` | 2^k m, k = −108..7 (51968 values) | 1.227 | 6.092721e-2 | 17074 of 51968 |
| cuda | f64 | `1 / sqrt(x)` | 2^k m, k = −108..7 (51968 values) | 1.500 | 1.232595e-32 | 0 of 51968 |

"≠ host": results whose bits differ from the correctly rounded host value (`sqrt`, divisions) or from the host's fl(1/fl(√x)) (inverse square roots).

Edge cases (cuda, f64; a = 1):

| x | `sqrt(x)` | `a / x` | `1 / x` | `recip(x)` | `inverse_sqrt(x)` | `inverse_sqrt(x) + Newton` | `1 / sqrt(x)` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +0 | +0 | +inf | +inf | +inf | +inf | **NaN** (IEEE +inf) | +inf |
| -0 | -0 | -inf | -inf | -inf | -inf | **NaN** (IEEE -inf) | -inf |
| min normal | 1.491668e-154 | 4.494233e307 | 4.494233e307 | 4.494233e307 | 6.703904e153 | 6.703904e153 | 6.703904e153 |
| subnormal | 5.273843e-155 | +inf | +inf | +inf | 1.896150e154 | **1.896150e154** (IEEE 1.896150e154) | 1.896150e154 |
| +inf | +inf | +0 | +0 | +0 | +0 | **NaN** (IEEE +0) | +0 |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

## Compiler behaviour: cuda

| backend | precision | probe | device | IEEE as written | verdict |
| --- | --- | --- | --- | --- | --- |
| cuda | f32 | a·b + c | 5.960464477539063e-8 | 0e0 | **fused** |
| cuda | f32 | c + a·b | 5.960464477539063e-8 | 0e0 | **fused** |
| cuda | f32 | a·b − c | 5.960464477539063e-8 | 0e0 | **fused** |
| cuda | f32 | c − a·b | -5.960464477539063e-8 | 0e0 | **fused** |
| cuda | f32 | p = a·b (two uses); p + c | 5.960464477539063e-8 | 0e0 | **fused** |
| cuda | f32 | p (the second use) | 1.00048828125e0 | 1.00048828125e0 | as written |
| cuda | f32 | fma(a, b, c) explicit | 5.960464477539063e-8 | 5.960464477539063e-8 | as written |
| cuda | f32 | a·a + (−a)·a | 5.960464477539063e-8 | 0e0 | **fused (either product)** |
| cuda | f32 | (x + b) − x (one value x = 1) | 0e0 | 0e0 | as written |
| cuda | f32 | x − (x + z) (one value x = 1) | 0e0 | 0e0 | as written |
| cuda | f32 | (x − y) − z (control) | -9.313225746154785e-10 | -9.313225746154785e-10 | as written |
| cuda | f32 | 1.5 min − 1.25 min (subnormal out) | 2.938735877055719e-39 | 2.938735877055719e-39 | as written |
| cuda | f32 | small · small (subnormal out) | 7.346839692639297e-40 | 7.346839692639297e-40 | as written |
| cuda | f32 | subnormal · 1 (subnormal in) | 1.4693679385278594e-39 | 1.4693679385278594e-39 | as written |
| cuda | f32 | subnormal + subnormal | 2.938735877055719e-39 | 2.938735877055719e-39 | as written |
| cuda | f32 | select(−0 == 0, 1, 2) | 1e0 | 1e0 | as written |
| cuda | f32 | select(subnormal == 0, 1, 2) | 2e0 | 2e0 | as written |
| cuda | f32 | q · select(r² == 0, 0, 1/√r²), r² = 0 | 0e0 | 0e0 | as written |
| cuda | f32 | t = q/√r²; if r² == 0 { t = 0 } | 0e0 | 0e0 | as written |
| cuda | f32 | ∞ − ∞ (same value) | 0e0 | NaN | **folded x − x → 0** |
| cuda | f32 | −0 + 0.0 (literal) | -0e0 | 0e0 | **folded x + 0 → x** |
| cuda | f32 | (x + b) − y (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |
| cuda | f32 | x − (y + z) (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |
| cuda | f32 | fma(z, z, a·b) + c (z = 0) | 0e0 | 0e0 | as written |
| backend | precision | probe | device | IEEE as written | verdict |
| --- | --- | --- | --- | --- | --- |
| cuda | f64 | a·b + c | 5.551115123125783e-17 | 0e0 | **fused** |
| cuda | f64 | c + a·b | 5.551115123125783e-17 | 0e0 | **fused** |
| cuda | f64 | a·b − c | 5.551115123125783e-17 | 0e0 | **fused** |
| cuda | f64 | c − a·b | -5.551115123125783e-17 | 0e0 | **fused** |
| cuda | f64 | p = a·b (two uses); p + c | 5.551115123125783e-17 | 0e0 | **fused** |
| cuda | f64 | p (the second use) | 1.0000000149011612e0 | 1.0000000149011612e0 | as written |
| cuda | f64 | fma(a, b, c) explicit | 5.551115123125783e-17 | 5.551115123125783e-17 | as written |
| cuda | f64 | a·a + (−a)·a | 5.551115123125783e-17 | 0e0 | **fused (either product)** |
| cuda | f64 | (x + b) − x (one value x = 1) | 0e0 | 0e0 | as written |
| cuda | f64 | x − (x + z) (one value x = 1) | 0e0 | 0e0 | as written |
| cuda | f64 | (x − y) − z (control) | -1.734723475976807e-18 | -1.734723475976807e-18 | as written |
| cuda | f64 | 1.5 min − 1.25 min (subnormal out) | 5.562684646268003e-309 | 5.562684646268003e-309 | as written |
| cuda | f64 | small · small (subnormal out) | 8.095e-320 | 8.095e-320 | as written |
| cuda | f64 | subnormal · 1 (subnormal in) | 2.781342323134e-309 | 2.781342323134e-309 | as written |
| cuda | f64 | subnormal + subnormal | 5.562684646268003e-309 | 5.562684646268003e-309 | as written |
| cuda | f64 | select(−0 == 0, 1, 2) | 1e0 | 1e0 | as written |
| cuda | f64 | select(subnormal == 0, 1, 2) | 2e0 | 2e0 | as written |
| cuda | f64 | q · select(r² == 0, 0, 1/√r²), r² = 0 | 0e0 | 0e0 | as written |
| cuda | f64 | t = q/√r²; if r² == 0 { t = 0 } | 0e0 | 0e0 | as written |
| cuda | f64 | ∞ − ∞ (same value) | 0e0 | NaN | **folded x − x → 0** |
| cuda | f64 | −0 + 0.0 (literal) | -0e0 | 0e0 | **folded x + 0 → x** |
| cuda | f64 | (x + b) − y (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |
| cuda | f64 | x − (y + z) (x = y = 1, distinct loads) | 0e0 | 0e0 | as written |
| cuda | f64 | fma(z, z, a·b) + c (z = 0) | 0e0 | 0e0 | as written |

## The §3.13 argument on the device: cuda

cuda f32 adversarial pairs: 209303 pairs, 30403 coincident (30385 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-53.00. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 164601 components and gain or lose a zero in 123571.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |
| fma forward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |
| fma backward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208420 | 208140 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 208885 | 209303 |

cuda f32 random pairs: 216000 pairs, 3526 coincident (0 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-17.70. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 126828 components and gain or lose a zero in 0.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |
| fma forward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |
| fma backward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 152271 | 153148 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 169442 | 216000 |

cuda f64 adversarial pairs: 439192 pairs, 484 coincident (154 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-53.00. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 3118 components and gain or lose a zero in 900.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 435717 | 439192 |
| fma forward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 435717 | 439192 |
| fma backward | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 434096 | 432767 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 435717 | 439192 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-106.00 | 12.0000 | 435717 | 439192 |

cuda f64 random pairs: 216000 pairs, 3526 coincident (0 from distinct points); ŷ ≠ host in 0 components, d ≠ host in 0 (unfused ŷ: 0), d = 0 wrong in 0; smallest nonzero |d| 2^-17.69. On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change 1656 components and gain or lose a zero in 0.

| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| written | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 170154 | 216000 |
| fma forward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 170154 | 216000 |
| fma backward | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 152728 | 153516 |
| explicit fma ŷ | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 170154 | 216000 |
| unfused ŷ (product with a second use) | yes | 0 | 0 | 0 | 2^-8.92 | 72.6506 | 170154 | 216000 |


## P2P candidates: cuda

Pair terms against `nd_fmm_ref::p2p` (1000000 pairs over the kernel domain; gradient over r² ≥ 2^-84), and sums against `direct_sum` on W1 (worst over n_t ∈ [8, 32, 64, 128], 4 sets each, with gradients; relative to the term magnitudes; the reference's own error in brackets):

| backend | precision | candidate | φ term (u_T) | ∇φ term (u_T) | φ, ∇φ term, nonzero |dₖ| ≥ 2^-53 (u_T) | φ sum | ∇φ sum | sum check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cuda | f32 | sqrt + div, as written | 3.997 | 10.296 | 3.997, 10.296 | 1.17e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| cuda | f32 | sqrt + div, explicit fma | 3.986 | 10.210 | 3.986, 10.210 | 1.23e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| cuda | f32 | inverse_sqrt, as written | 3.997 | 10.296 | 3.997, 10.296 | 1.17e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| cuda | f32 | inverse_sqrt, explicit fma | 3.986 | 10.210 | 3.986, 10.210 | 1.23e-7 [1.28e-7] | 5.93e-7 [6.23e-7] | pass |
| cuda | f32 | inverse_sqrt + Newton, as written | 4.321 | 9.296 | 4.321, 9.296 | 1.17e-7 [1.28e-7] | 5.41e-7 [6.23e-7] | pass |
| cuda | f32 | inverse_sqrt + Newton, explicit fma | 3.974 | 8.905 | 3.974, 8.905 | 1.11e-7 [1.28e-7] | 5.41e-7 [6.23e-7] | pass |

Quick throughput, Gpairs/s (4096 target leaves per launch, one unit per target, 64 per cube, launches queued between syncs, compilation excluded, median of 15 batches of at least 20 ms):

| backend | precision | candidate | output | n_t = 8 | n_t = 32 | n_t = 64 | n_t = 128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cuda | f32 | sqrt + div, as written | φ | 135.8 | 373.0 | 433.9 | 475.0 |
| cuda | f32 | sqrt + div, as written | φ, ∇φ | 122.2 | 327.6 | 375.1 | 414.7 |
| cuda | f32 | sqrt + div, explicit fma | φ | 136.9 | 370.4 | 433.9 | 476.7 |
| cuda | f32 | sqrt + div, explicit fma | φ, ∇φ | 122.7 | 328.0 | 366.2 | 414.9 |
| cuda | f32 | inverse_sqrt, as written | φ | 135.7 | 372.4 | 433.6 | 474.5 |
| cuda | f32 | inverse_sqrt, as written | φ, ∇φ | 122.2 | 327.9 | 380.5 | 414.7 |
| cuda | f32 | inverse_sqrt, explicit fma | φ | 137.0 | 372.6 | 436.4 | 477.1 |
| cuda | f32 | inverse_sqrt, explicit fma | φ, ∇φ | 123.1 | 327.8 | 369.5 | 415.0 |
| cuda | f32 | inverse_sqrt + Newton, as written | φ | 138.2 | 350.1 | 411.6 | 450.8 |
| cuda | f32 | inverse_sqrt + Newton, as written | φ, ∇φ | 122.0 | 310.8 | 360.9 | 394.1 |
| cuda | f32 | inverse_sqrt + Newton, explicit fma | φ | 139.1 | 350.3 | 411.5 | 451.1 |
| cuda | f32 | inverse_sqrt + Newton, explicit fma | φ, ∇φ | 123.0 | 308.2 | 359.3 | 392.1 |
Pair terms against `nd_fmm_ref::p2p` (1000000 pairs over the kernel domain; gradient over r² ≥ 2^-108), and sums against `direct_sum` on W1 (worst over n_t ∈ [8, 32, 64, 128], 4 sets each, with gradients; relative to the term magnitudes; the reference's own error in brackets):

| backend | precision | candidate | φ term (u_T) | ∇φ term (u_T) | φ, ∇φ term, nonzero |dₖ| ≥ 2^-53 (u_T) | φ sum | ∇φ sum | sum check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cuda | f64 | sqrt + div, as written | 3.998 | 10.195 | 3.998, 10.195 | 2.53e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cuda | f64 | sqrt + div, explicit fma | 4.054 | 9.657 | 4.054, 9.657 | 2.42e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cuda | f64 | inverse_sqrt, as written | 3.998 | 10.195 | 3.998, 10.195 | 2.53e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cuda | f64 | inverse_sqrt, explicit fma | 4.054 | 9.657 | 4.054, 9.657 | 2.42e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cuda | f64 | inverse_sqrt + Newton, as written | 4.606 | 9.296 | 4.606, 9.296 | 2.53e-16 [3.08e-16] | 1.89e-15 [1.75e-15] | pass |
| cuda | f64 | inverse_sqrt + Newton, explicit fma | 4.000 | 9.268 | 4.000, 9.268 | 2.31e-16 [3.08e-16] | 1.46e-15 [1.75e-15] | pass |

Quick throughput, Gpairs/s (4096 target leaves per launch, one unit per target, 64 per cube, launches queued between syncs, compilation excluded, median of 15 batches of at least 20 ms):

| backend | precision | candidate | output | n_t = 8 | n_t = 32 | n_t = 64 | n_t = 128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cuda | f64 | sqrt + div, as written | φ | 87.7 | 241.9 | 273.8 | 301.9 |
| cuda | f64 | sqrt + div, as written | φ, ∇φ | 85.9 | 214.5 | 242.1 | 263.4 |
| cuda | f64 | sqrt + div, explicit fma | φ | 86.9 | 238.8 | 272.6 | 298.3 |
| cuda | f64 | sqrt + div, explicit fma | φ, ∇φ | 85.1 | 215.9 | 243.5 | 263.9 |
| cuda | f64 | inverse_sqrt, as written | φ | 87.4 | 242.1 | 275.0 | 302.3 |
| cuda | f64 | inverse_sqrt, as written | φ, ∇φ | 86.4 | 214.7 | 242.0 | 263.3 |
| cuda | f64 | inverse_sqrt, explicit fma | φ | 86.8 | 238.7 | 272.0 | 298.6 |
| cuda | f64 | inverse_sqrt, explicit fma | φ, ∇φ | 85.0 | 215.8 | 243.6 | 263.8 |
| cuda | f64 | inverse_sqrt + Newton, as written | φ | 86.9 | 226.7 | 261.0 | 284.3 |
| cuda | f64 | inverse_sqrt + Newton, as written | φ, ∇φ | 85.2 | 201.3 | 229.2 | 250.9 |
| cuda | f64 | inverse_sqrt + Newton, explicit fma | φ | 86.0 | 223.9 | 257.8 | 280.2 |
| cuda | f64 | inverse_sqrt + Newton, explicit fma | φ, ∇φ | 84.4 | 203.6 | 229.9 | 251.2 |

## Harmonics and GEMM: cuda

| backend | precision | test | bit-identical to host | max diff from host (u_T) | device vs f64 (u_T) | host vs f64 (u_T) |
| --- | --- | --- | --- | --- | --- | --- |
| cuda | f32 | regular harmonics, p = 8 | 38963 of 81000 | 34.761 (34.761 where normal; 0 host values subnormal) | 24.877 | 43.965 |
| cuda | f32 | irregular harmonics, p = 8 | 32955 of 81000 | 74.922 (74.922 where normal; 0 host values subnormal) | 36.827 | 69.803 |
| cuda | f32 | regular harmonics, p = 20 | 115132 of 441000 | 216.509 (216.509 where normal; 351 host values subnormal) | 118.124 | 143.170 |
| cuda | f32 | irregular harmonics, p = 20 | 105338 of 441000 | 306.227 (306.227 where normal; 0 host values subnormal) | 202.057 | 243.180 |
| cuda | f32 | y += A x, n = 81, 512 columns | 10724 of 41472 (= fused host: 41472) | 3.489 | – | – |
| cuda | f64 | regular harmonics, p = 8 | 39240 of 81000 | 40.162 (40.162 where normal; 0 host values subnormal) | 40.162 | 0.000 |
| cuda | f64 | irregular harmonics, p = 8 | 32911 of 81000 | 65.340 (65.340 where normal; 0 host values subnormal) | 65.340 | 0.000 |
| cuda | f64 | regular harmonics, p = 20 | 117006 of 441000 | 106.625 (106.625 where normal; 0 host values subnormal) | 106.625 | 0.000 |
| cuda | f64 | irregular harmonics, p = 20 | 105935 of 441000 | 213.648 (213.648 where normal; 0 host values subnormal) | 213.648 | 0.000 |
| cuda | f64 | y += A x, n = 81, 512 columns | 10524 of 41472 (= fused host: 41472) | 3.172 | – | – |

Harmonics errors per point and degree, relative to the largest |value| of that degree; GEMM relative to max |y|.

## Summary

Every check passed on: cuda.

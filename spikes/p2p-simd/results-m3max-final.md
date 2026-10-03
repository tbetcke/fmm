# Phase 3S T7: final runs on the Apple M3 Max

Raw output of the four runs of docs/phase3s/T7-benchmarks.md, taken on 2026-10-03 one after
the other on an otherwise idle Apple M3 Max (NEON), from the commit of the T7 pull request,
with every BLAS thread variable set to 1:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
RAYON_NUM_THREADS=1 cargo run --release -p nd-fmm-validate --example p2p_kernels
RAYON_NUM_THREADS=1 cargo run --release -p nd-fmm-spike-p2p-simd --example compare
cargo run --release -p nd-fmm-validate --example p2p_fmm -- --threads 1
cargo run --release -p nd-fmm-validate --example p2p_fmm -- --threads 12
```

Wall times: `p2p_kernels` 2.5 min, `compare` 2.2 min, `p2p_fmm` 8.4 min (1 thread) and 1.3 min
(12 threads). **No x86_64 path was timed**: no x86_64 machine is available in Phase 3S. Both
examples build for x86_64 (checked with `cargo clippy --target x86_64-apple-darwin`) and
time the AVX2 path when run there. The summary and the analysis are in the T7 pull
request and in docs/design/laplace-fmm-plan.md §7, Phase 3S.

Headings of the reports are demoted by one level below.


## P2P kernels: nd-fmm-simd against nd_fmm_ref::p2p (Phase 3S T7)

| item | value |
| --- | --- |
| CPU | Apple M3 Max |
| cores | 16 physical, 16 logical, 12 performance |
| target | aarch64-macos, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| ISAs available | scalar, neon (detected: neon) |
| thread variables | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, RAYON_NUM_THREADS=1 |
| clock for cycles | 4.05 GHz assumed |
| mode | full |
| timing | median of 15 batches of ≥ 20 ms, one thread |

- Workloads (design §8.2): W1, a target leaf of n_t points in [-1, 1]^3 and the 27 leaves of its 3 x 3 x 3 block, n_t sources each (the centre leaf equal to the targets), a pool of 64 sets cycled; per-pair = 27 calls, gathered = one call. W2, N sources in [0, 1]^3, targets distinct (t≠s) or equal (t=s). Charges in [-1, 1]. The inputs of the T2 spike.
- Gpairs/s counts n_s n_t pairs per evaluation, coincident ones included. "x ref": pairs per second over the reference's on the same cell.
- Accuracy against `direct_sum` on the first 4 sets of the cell: "max" is the largest error relative to the term magnitudes of its target (Σ|q|/r for φ, Σ|q|/r² per component of ∇φ), "L2" the relative L2 error. Check: requirement 2 (design §3) as decided in T5, max within 1e-6 (f32) or 1e-14 (f64), or within twice the reference's error on the same inputs.
- Model: pairs per cycle of design §4.6 as corrected by the T2 spike (NEON: 4 FP pipes and the divider, W / max(ops/4, 3); AVX2: 2 FMA pipes, 2W / ops), with the measured pairs per cycle at the assumed clock.
- The reference's speed is bimodal on the M3 Max (T2 spike, "Other findings"): potential-only cells run at about 0.34 or about 0.86 Gpairs/s on the same data, depending on the run, because its compiled loop keeps φ in memory across the source loop. The speed-ups use the measured values as they are.

### f32, potential

| cell | kernel | Gpairs/s | x ref | model ops | pairs/cycle | of model | max φ | L2 φ | max ∇φ | L2 ∇φ | check | per-pair = gathered |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| W1 n_t=8 per-pair | reference | 0.762 | 1.00 | – | 0.188 | – | 8.8e-8 | 4.5e-7 |  |  | pass |  |
| W1 n_t=8 per-pair | scalar | 0.953 | 1.25 | – | 0.235 | – | 8.8e-8 | 4.5e-7 |  |  | pass | yes |
| W1 n_t=8 per-pair | neon | 3.215 | 4.22 | 11 | 0.794 | 60% | 7.7e-8 | 5.9e-7 |  |  | pass | yes |
| W1 n_t=8 gathered | reference | 0.364 | 1.00 | – | 0.090 | – | 8.8e-8 | 4.5e-7 |  |  | pass |  |
| W1 n_t=8 gathered | scalar | 0.341 | 0.94 | – | 0.084 | – | 8.8e-8 | 4.5e-7 |  |  | pass |  |
| W1 n_t=8 gathered | neon | 3.936 | 10.83 | 11 | 0.972 | 73% | 7.7e-8 | 5.9e-7 |  |  | pass |  |
| W1 n_t=16 per-pair | reference | 0.633 | 1.00 | – | 0.156 | – | 8.7e-8 | 9.4e-7 |  |  | pass |  |
| W1 n_t=16 per-pair | scalar | 0.799 | 1.26 | – | 0.197 | – | 8.7e-8 | 9.4e-7 |  |  | pass | yes |
| W1 n_t=16 per-pair | neon | 3.865 | 6.10 | 11 | 0.954 | 72% | 7.6e-8 | 8.2e-7 |  |  | pass | yes |
| W1 n_t=16 gathered | reference | 0.353 | 1.00 | – | 0.087 | – | 8.7e-8 | 9.4e-7 |  |  | pass |  |
| W1 n_t=16 gathered | scalar | 1.095 | 3.10 | – | 0.270 | – | 8.7e-8 | 9.4e-7 |  |  | pass |  |
| W1 n_t=16 gathered | neon | 3.999 | 11.34 | 11 | 0.987 | 74% | 7.6e-8 | 8.2e-7 |  |  | pass |  |
| W1 n_t=24 per-pair | reference | 0.565 | 1.00 | – | 0.139 | – | 8.7e-8 | 6.8e-7 |  |  | pass |  |
| W1 n_t=24 per-pair | scalar | 0.657 | 1.16 | – | 0.162 | – | 8.7e-8 | 6.8e-7 |  |  | pass | yes |
| W1 n_t=24 per-pair | neon | 3.944 | 6.98 | 11 | 0.974 | 73% | 8.1e-8 | 7.0e-7 |  |  | pass | yes |
| W1 n_t=24 gathered | reference | 0.820 | 1.00 | – | 0.202 | – | 8.7e-8 | 6.8e-7 |  |  | pass |  |
| W1 n_t=24 gathered | scalar | 1.111 | 1.36 | – | 0.274 | – | 8.7e-8 | 6.8e-7 |  |  | pass |  |
| W1 n_t=24 gathered | neon | 4.053 | 4.95 | 11 | 1.001 | 75% | 8.1e-8 | 7.0e-7 |  |  | pass |  |
| W1 n_t=32 per-pair | reference | 0.517 | 1.00 | – | 0.128 | – | 5.6e-8 | 9.1e-7 |  |  | pass |  |
| W1 n_t=32 per-pair | scalar | 0.578 | 1.12 | – | 0.143 | – | 5.6e-8 | 9.1e-7 |  |  | pass | yes |
| W1 n_t=32 per-pair | neon | 4.006 | 7.76 | 11 | 0.989 | 74% | 7.0e-8 | 8.1e-7 |  |  | pass | yes |
| W1 n_t=32 gathered | reference | 0.848 | 1.00 | – | 0.209 | – | 5.6e-8 | 9.1e-7 |  |  | pass |  |
| W1 n_t=32 gathered | scalar | 1.111 | 1.31 | – | 0.274 | – | 5.6e-8 | 9.1e-7 |  |  | pass |  |
| W1 n_t=32 gathered | neon | 4.032 | 4.76 | 11 | 0.996 | 75% | 7.0e-8 | 8.1e-7 |  |  | pass |  |
| W1 n_t=64 per-pair | reference | 0.476 | 1.00 | – | 0.117 | – | 1.3e-7 | 1.2e-6 |  |  | pass |  |
| W1 n_t=64 per-pair | scalar | 0.499 | 1.05 | – | 0.123 | – | 1.3e-7 | 1.2e-6 |  |  | pass | yes |
| W1 n_t=64 per-pair | neon | 3.876 | 8.15 | 11 | 0.957 | 72% | 1.2e-7 | 1.1e-6 |  |  | pass | yes |
| W1 n_t=64 gathered | reference | 0.853 | 1.00 | – | 0.211 | – | 1.3e-7 | 1.2e-6 |  |  | pass |  |
| W1 n_t=64 gathered | scalar | 1.125 | 1.32 | – | 0.278 | – | 1.3e-7 | 1.2e-6 |  |  | pass |  |
| W1 n_t=64 gathered | neon | 4.072 | 4.77 | 11 | 1.005 | 75% | 1.2e-7 | 1.1e-6 |  |  | pass |  |
| W1 n_t=128 per-pair | reference | 0.397 | 1.00 | – | 0.098 | – | 1.0e-7 | 1.6e-6 |  |  | pass |  |
| W1 n_t=128 per-pair | scalar | 0.412 | 1.04 | – | 0.102 | – | 1.0e-7 | 1.6e-6 |  |  | pass | yes |
| W1 n_t=128 per-pair | neon | 3.910 | 9.86 | 11 | 0.965 | 72% | 1.1e-7 | 1.6e-6 |  |  | pass | yes |
| W1 n_t=128 gathered | reference | 0.338 | 1.00 | – | 0.083 | – | 1.0e-7 | 1.6e-6 |  |  | pass |  |
| W1 n_t=128 gathered | scalar | 1.130 | 3.34 | – | 0.279 | – | 1.0e-7 | 1.6e-6 |  |  | pass |  |
| W1 n_t=128 gathered | neon | 4.072 | 12.05 | 11 | 1.005 | 75% | 1.1e-7 | 1.6e-6 |  |  | pass |  |
| W2 N=1000 t≠s | reference | 0.386 | 1.00 | – | 0.095 | – | 1.1e-7 | 6.2e-7 |  |  | pass |  |
| W2 N=1000 t≠s | scalar | 1.117 | 2.89 | – | 0.276 | – | 1.1e-7 | 6.2e-7 |  |  | pass |  |
| W2 N=1000 t≠s | neon | 4.081 | 10.56 | 11 | 1.008 | 76% | 1.2e-7 | 6.3e-7 |  |  | pass |  |
| W2 N=1000 t=s | reference | 0.834 | 1.00 | – | 0.206 | – | 1.2e-7 | 6.6e-7 |  |  | pass |  |
| W2 N=1000 t=s | scalar | 1.071 | 1.28 | – | 0.264 | – | 1.2e-7 | 6.6e-7 |  |  | pass |  |
| W2 N=1000 t=s | neon | 4.036 | 4.84 | 11 | 0.997 | 75% | 1.3e-7 | 6.6e-7 |  |  | pass |  |
| W2 N=10000 t≠s | reference | 0.805 | 1.00 | – | 0.199 | – | 1.5e-7 | 1.9e-6 |  |  | pass |  |
| W2 N=10000 t≠s | scalar | 1.113 | 1.38 | – | 0.275 | – | 1.5e-7 | 1.9e-6 |  |  | pass |  |
| W2 N=10000 t≠s | neon | 4.046 | 5.02 | 11 | 0.999 | 75% | 1.6e-7 | 1.9e-6 |  |  | pass |  |
| W2 N=10000 t=s | reference | 0.858 | 1.00 | – | 0.212 | – | 1.6e-7 | 1.9e-6 |  |  | pass |  |
| W2 N=10000 t=s | scalar | 1.111 | 1.29 | – | 0.274 | – | 1.6e-7 | 1.9e-6 |  |  | pass |  |
| W2 N=10000 t=s | neon | 4.046 | 4.72 | 11 | 0.999 | 75% | 1.5e-7 | 1.9e-6 |  |  | pass |  |

### f32, potential and gradient

| cell | kernel | Gpairs/s | x ref | model ops | pairs/cycle | of model | max φ | L2 φ | max ∇φ | L2 ∇φ | check | per-pair = gathered |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| W1 n_t=8 per-pair | reference | 0.519 | 1.00 | – | 0.128 | – | 8.8e-8 | 4.5e-7 | 1.5e-7 | 3.0e-7 | pass |  |
| W1 n_t=8 per-pair | scalar | 0.521 | 1.00 | – | 0.129 | – | 8.8e-8 | 4.5e-7 | 1.5e-7 | 3.0e-7 | pass | yes |
| W1 n_t=8 per-pair | neon | 2.121 | 4.08 | 17 | 0.524 | 56% | 7.7e-8 | 7.1e-7 | 1.7e-7 | 3.3e-7 | pass | yes |
| W1 n_t=8 gathered | reference | 0.323 | 1.00 | – | 0.080 | – | 8.8e-8 | 4.5e-7 | 1.5e-7 | 3.0e-7 | pass |  |
| W1 n_t=8 gathered | scalar | 0.322 | 1.00 | – | 0.080 | – | 8.8e-8 | 4.5e-7 | 1.5e-7 | 3.0e-7 | pass |  |
| W1 n_t=8 gathered | neon | 2.564 | 7.94 | 17 | 0.633 | 67% | 7.7e-8 | 7.1e-7 | 1.7e-7 | 3.3e-7 | pass |  |
| W1 n_t=16 per-pair | reference | 0.479 | 1.00 | – | 0.118 | – | 8.7e-8 | 9.4e-7 | 2.2e-7 | 4.5e-7 | pass |  |
| W1 n_t=16 per-pair | scalar | 0.484 | 1.01 | – | 0.120 | – | 8.7e-8 | 9.4e-7 | 2.2e-7 | 4.5e-7 | pass | yes |
| W1 n_t=16 per-pair | neon | 2.461 | 5.14 | 17 | 0.608 | 65% | 5.3e-8 | 8.4e-7 | 2.4e-7 | 5.0e-7 | pass | yes |
| W1 n_t=16 gathered | reference | 0.311 | 1.00 | – | 0.077 | – | 8.7e-8 | 9.4e-7 | 2.2e-7 | 4.5e-7 | pass |  |
| W1 n_t=16 gathered | scalar | 0.316 | 1.02 | – | 0.078 | – | 8.7e-8 | 9.4e-7 | 2.2e-7 | 4.5e-7 | pass |  |
| W1 n_t=16 gathered | neon | 2.622 | 8.42 | 17 | 0.647 | 69% | 5.3e-8 | 8.4e-7 | 2.4e-7 | 5.0e-7 | pass |  |
| W1 n_t=24 per-pair | reference | 0.419 | 1.00 | – | 0.104 | – | 8.7e-8 | 6.8e-7 | 1.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 per-pair | scalar | 0.415 | 0.99 | – | 0.103 | – | 8.7e-8 | 6.8e-7 | 1.7e-7 | 5.2e-7 | pass | yes |
| W1 n_t=24 per-pair | neon | 2.437 | 5.81 | 17 | 0.602 | 64% | 8.1e-8 | 6.7e-7 | 1.3e-7 | 5.2e-7 | pass | yes |
| W1 n_t=24 gathered | reference | 0.317 | 1.00 | – | 0.078 | – | 8.7e-8 | 6.8e-7 | 1.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | scalar | 0.311 | 0.98 | – | 0.077 | – | 8.7e-8 | 6.8e-7 | 1.7e-7 | 5.2e-7 | pass |  |
| W1 n_t=24 gathered | neon | 2.599 | 8.21 | 17 | 0.642 | 68% | 8.1e-8 | 6.7e-7 | 1.3e-7 | 5.2e-7 | pass |  |
| W1 n_t=32 per-pair | reference | 0.385 | 1.00 | – | 0.095 | – | 5.6e-8 | 9.1e-7 | 4.6e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 per-pair | scalar | 0.383 | 0.99 | – | 0.094 | – | 5.6e-8 | 9.1e-7 | 4.6e-7 | 5.8e-7 | pass | yes |
| W1 n_t=32 per-pair | neon | 2.563 | 6.66 | 17 | 0.633 | 67% | 5.9e-8 | 8.0e-7 | 4.9e-7 | 5.1e-7 | pass | yes |
| W1 n_t=32 gathered | reference | 0.315 | 1.00 | – | 0.078 | – | 5.6e-8 | 9.1e-7 | 4.6e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 gathered | scalar | 0.315 | 1.00 | – | 0.078 | – | 5.6e-8 | 9.1e-7 | 4.6e-7 | 5.8e-7 | pass |  |
| W1 n_t=32 gathered | neon | 2.558 | 8.13 | 17 | 0.632 | 67% | 5.9e-8 | 8.0e-7 | 4.9e-7 | 5.1e-7 | pass |  |
| W1 n_t=64 per-pair | reference | 0.345 | 1.00 | – | 0.085 | – | 1.3e-7 | 1.2e-6 | 3.8e-7 | 6.9e-7 | pass |  |
| W1 n_t=64 per-pair | scalar | 0.344 | 1.00 | – | 0.085 | – | 1.3e-7 | 1.2e-6 | 3.8e-7 | 6.9e-7 | pass | yes |
| W1 n_t=64 per-pair | neon | 2.618 | 7.58 | 17 | 0.646 | 69% | 1.3e-7 | 1.1e-6 | 3.2e-7 | 6.8e-7 | pass | yes |
| W1 n_t=64 gathered | reference | 0.314 | 1.00 | – | 0.078 | – | 1.3e-7 | 1.2e-6 | 3.8e-7 | 6.9e-7 | pass |  |
| W1 n_t=64 gathered | scalar | 0.314 | 1.00 | – | 0.078 | – | 1.3e-7 | 1.2e-6 | 3.8e-7 | 6.9e-7 | pass |  |
| W1 n_t=64 gathered | neon | 2.650 | 8.44 | 17 | 0.654 | 70% | 1.3e-7 | 1.1e-6 | 3.2e-7 | 6.8e-7 | pass |  |
| W1 n_t=128 per-pair | reference | 0.323 | 1.00 | – | 0.080 | – | 1.0e-7 | 1.6e-6 | 6.2e-7 | 1.2e-6 | pass |  |
| W1 n_t=128 per-pair | scalar | 0.331 | 1.03 | – | 0.082 | – | 1.0e-7 | 1.6e-6 | 6.2e-7 | 1.2e-6 | pass | yes |
| W1 n_t=128 per-pair | neon | 2.619 | 8.11 | 17 | 0.647 | 69% | 1.1e-7 | 1.6e-6 | 5.9e-7 | 1.1e-6 | pass | yes |
| W1 n_t=128 gathered | reference | 0.314 | 1.00 | – | 0.077 | – | 1.0e-7 | 1.6e-6 | 6.2e-7 | 1.2e-6 | pass |  |
| W1 n_t=128 gathered | scalar | 0.315 | 1.01 | – | 0.078 | – | 1.0e-7 | 1.6e-6 | 6.2e-7 | 1.2e-6 | pass |  |
| W1 n_t=128 gathered | neon | 2.653 | 8.46 | 17 | 0.655 | 70% | 1.1e-7 | 1.6e-6 | 5.9e-7 | 1.1e-6 | pass |  |
| W2 N=1000 t≠s | reference | 0.316 | 1.00 | – | 0.078 | – | 1.1e-7 | 6.2e-7 | 7.9e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | scalar | 0.319 | 1.01 | – | 0.079 | – | 1.1e-7 | 6.2e-7 | 7.9e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t≠s | neon | 2.651 | 8.39 | 17 | 0.655 | 70% | 1.1e-7 | 6.2e-7 | 7.4e-7 | 4.4e-7 | pass |  |
| W2 N=1000 t=s | reference | 0.314 | 1.00 | – | 0.077 | – | 1.2e-7 | 6.6e-7 | 9.6e-7 | 6.6e-7 | pass |  |
| W2 N=1000 t=s | scalar | 0.317 | 1.01 | – | 0.078 | – | 1.2e-7 | 6.6e-7 | 9.6e-7 | 6.6e-7 | pass |  |
| W2 N=1000 t=s | neon | 2.597 | 8.28 | 17 | 0.641 | 68% | 1.2e-7 | 6.6e-7 | 1.0e-6 | 6.7e-7 | pass |  |
| W2 N=10000 t≠s | reference | 0.312 | 1.00 | – | 0.077 | – | 1.5e-7 | 1.9e-6 | 2.0e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | scalar | 0.313 | 1.00 | – | 0.077 | – | 1.5e-7 | 1.9e-6 | 2.0e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t≠s | neon | 2.632 | 8.42 | 17 | 0.650 | 69% | 1.6e-7 | 1.9e-6 | 2.0e-6 | 1.2e-6 | pass |  |
| W2 N=10000 t=s | reference | 0.312 | 1.00 | – | 0.077 | – | 1.6e-7 | 1.9e-6 | 2.7e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | scalar | 0.313 | 1.00 | – | 0.077 | – | 1.6e-7 | 1.9e-6 | 2.7e-6 | 1.4e-6 | pass |  |
| W2 N=10000 t=s | neon | 2.628 | 8.41 | 17 | 0.649 | 69% | 1.8e-7 | 1.9e-6 | 2.6e-6 | 1.4e-6 | pass |  |

### f64, potential

| cell | kernel | Gpairs/s | x ref | model ops | pairs/cycle | of model | max φ | L2 φ | max ∇φ | L2 ∇φ | check | per-pair = gathered |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| W1 n_t=8 per-pair | reference | 0.752 | 1.00 | – | 0.186 | – | 8.2e-17 | 1.2e-15 |  |  | pass |  |
| W1 n_t=8 per-pair | scalar | 0.922 | 1.23 | – | 0.228 | – | 8.2e-17 | 1.2e-15 |  |  | pass | yes |
| W1 n_t=8 per-pair | neon | 1.954 | 2.60 | 11 | 0.483 | 72% | 1.2e-16 | 1.2e-15 |  |  | pass | yes |
| W1 n_t=8 gathered | reference | 0.369 | 1.00 | – | 0.091 | – | 8.2e-17 | 1.2e-15 |  |  | pass |  |
| W1 n_t=8 gathered | scalar | 0.346 | 0.94 | – | 0.086 | – | 8.2e-17 | 1.2e-15 |  |  | pass |  |
| W1 n_t=8 gathered | neon | 2.129 | 5.77 | 11 | 0.526 | 79% | 1.2e-16 | 1.2e-15 |  |  | pass |  |
| W1 n_t=16 per-pair | reference | 0.636 | 1.00 | – | 0.157 | – | 1.1e-16 | 1.7e-15 |  |  | pass |  |
| W1 n_t=16 per-pair | scalar | 0.758 | 1.19 | – | 0.187 | – | 1.1e-16 | 1.7e-15 |  |  | pass | yes |
| W1 n_t=16 per-pair | neon | 2.124 | 3.34 | 11 | 0.524 | 79% | 1.2e-16 | 2.0e-15 |  |  | pass | yes |
| W1 n_t=16 gathered | reference | 0.352 | 1.00 | – | 0.087 | – | 1.1e-16 | 1.7e-15 |  |  | pass |  |
| W1 n_t=16 gathered | scalar | 0.335 | 0.95 | – | 0.083 | – | 1.1e-16 | 1.7e-15 |  |  | pass |  |
| W1 n_t=16 gathered | neon | 2.184 | 6.21 | 11 | 0.539 | 81% | 1.2e-16 | 2.0e-15 |  |  | pass |  |
| W1 n_t=24 per-pair | reference | 0.553 | 1.00 | – | 0.136 | – | 1.3e-16 | 1.1e-15 |  |  | pass |  |
| W1 n_t=24 per-pair | scalar | 0.660 | 1.19 | – | 0.163 | – | 1.3e-16 | 1.1e-15 |  |  | pass | yes |
| W1 n_t=24 per-pair | neon | 2.147 | 3.89 | 11 | 0.530 | 80% | 1.6e-16 | 1.3e-15 |  |  | pass | yes |
| W1 n_t=24 gathered | reference | 0.347 | 1.00 | – | 0.086 | – | 1.3e-16 | 1.1e-15 |  |  | pass |  |
| W1 n_t=24 gathered | scalar | 1.102 | 3.17 | – | 0.272 | – | 1.3e-16 | 1.1e-15 |  |  | pass |  |
| W1 n_t=24 gathered | neon | 2.192 | 6.31 | 11 | 0.541 | 81% | 1.6e-16 | 1.3e-15 |  |  | pass |  |
| W1 n_t=32 per-pair | reference | 0.496 | 1.00 | – | 0.123 | – | 1.7e-16 | 1.8e-15 |  |  | pass |  |
| W1 n_t=32 per-pair | scalar | 0.537 | 1.08 | – | 0.133 | – | 1.7e-16 | 1.8e-15 |  |  | pass | yes |
| W1 n_t=32 per-pair | neon | 2.120 | 4.27 | 11 | 0.524 | 79% | 1.6e-16 | 1.7e-15 |  |  | pass | yes |
| W1 n_t=32 gathered | reference | 0.592 | 1.00 | – | 0.146 | – | 1.7e-16 | 1.8e-15 |  |  | pass |  |
| W1 n_t=32 gathered | scalar | 1.061 | 1.79 | – | 0.262 | – | 1.7e-16 | 1.8e-15 |  |  | pass |  |
| W1 n_t=32 gathered | neon | 2.102 | 3.55 | 11 | 0.519 | 78% | 1.6e-16 | 1.7e-15 |  |  | pass |  |
| W1 n_t=64 per-pair | reference | 0.451 | 1.00 | – | 0.111 | – | 3.1e-16 | 2.4e-15 |  |  | pass |  |
| W1 n_t=64 per-pair | scalar | 0.502 | 1.11 | – | 0.124 | – | 3.1e-16 | 2.4e-15 |  |  | pass | yes |
| W1 n_t=64 per-pair | neon | 2.090 | 4.64 | 11 | 0.516 | 77% | 2.4e-16 | 2.5e-15 |  |  | pass | yes |
| W1 n_t=64 gathered | reference | 0.653 | 1.00 | – | 0.161 | – | 3.1e-16 | 2.4e-15 |  |  | pass |  |
| W1 n_t=64 gathered | scalar | 1.074 | 1.64 | – | 0.265 | – | 3.1e-16 | 2.4e-15 |  |  | pass |  |
| W1 n_t=64 gathered | neon | 2.128 | 3.26 | 11 | 0.525 | 79% | 2.4e-16 | 2.5e-15 |  |  | pass |  |
| W1 n_t=128 per-pair | reference | 0.384 | 1.00 | – | 0.095 | – | 2.1e-16 | 3.3e-15 |  |  | pass |  |
| W1 n_t=128 per-pair | scalar | 0.398 | 1.03 | – | 0.098 | – | 2.1e-16 | 3.3e-15 |  |  | pass | yes |
| W1 n_t=128 per-pair | neon | 2.189 | 5.70 | 11 | 0.541 | 81% | 2.2e-16 | 3.3e-15 |  |  | pass | yes |
| W1 n_t=128 gathered | reference | 0.816 | 1.00 | – | 0.202 | – | 2.1e-16 | 3.3e-15 |  |  | pass |  |
| W1 n_t=128 gathered | scalar | 1.101 | 1.35 | – | 0.272 | – | 2.1e-16 | 3.3e-15 |  |  | pass |  |
| W1 n_t=128 gathered | neon | 2.114 | 2.59 | 11 | 0.522 | 78% | 2.2e-16 | 3.3e-15 |  |  | pass |  |
| W2 N=1000 t≠s | reference | 0.320 | 1.00 | – | 0.079 | – | 2.1e-16 | 1.1e-15 |  |  | pass |  |
| W2 N=1000 t≠s | scalar | 1.127 | 3.52 | – | 0.278 | – | 2.1e-16 | 1.1e-15 |  |  | pass |  |
| W2 N=1000 t≠s | neon | 2.129 | 6.65 | 11 | 0.526 | 79% | 2.3e-16 | 1.1e-15 |  |  | pass |  |
| W2 N=1000 t=s | reference | 0.332 | 1.00 | – | 0.082 | – | 3.0e-16 | 1.2e-15 |  |  | pass |  |
| W2 N=1000 t=s | scalar | 1.093 | 3.30 | – | 0.270 | – | 3.0e-16 | 1.2e-15 |  |  | pass |  |
| W2 N=1000 t=s | neon | 2.199 | 6.63 | 11 | 0.543 | 81% | 2.5e-16 | 1.2e-15 |  |  | pass |  |
| W2 N=10000 t≠s | reference | 0.523 | 1.00 | – | 0.129 | – | 2.6e-16 | 3.5e-15 |  |  | pass |  |
| W2 N=10000 t≠s | scalar | 1.075 | 2.06 | – | 0.265 | – | 2.6e-16 | 3.5e-15 |  |  | pass |  |
| W2 N=10000 t≠s | neon | 2.196 | 4.20 | 11 | 0.542 | 81% | 2.7e-16 | 3.5e-15 |  |  | pass |  |
| W2 N=10000 t=s | reference | 0.753 | 1.00 | – | 0.186 | – | 6.2e-16 | 3.5e-15 |  |  | pass |  |
| W2 N=10000 t=s | scalar | 1.116 | 1.48 | – | 0.275 | – | 6.2e-16 | 3.5e-15 |  |  | pass |  |
| W2 N=10000 t=s | neon | 2.202 | 2.92 | 11 | 0.544 | 82% | 6.1e-16 | 3.5e-15 |  |  | pass |  |

### f64, potential and gradient

| cell | kernel | Gpairs/s | x ref | model ops | pairs/cycle | of model | max φ | L2 φ | max ∇φ | L2 ∇φ | check | per-pair = gathered |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| W1 n_t=8 per-pair | reference | 0.493 | 1.00 | – | 0.122 | – | 8.2e-17 | 1.2e-15 | 4.3e-16 | 8.1e-16 | pass |  |
| W1 n_t=8 per-pair | scalar | 0.494 | 1.00 | – | 0.122 | – | 8.2e-17 | 1.2e-15 | 4.3e-16 | 8.1e-16 | pass | yes |
| W1 n_t=8 per-pair | neon | 0.970 | 1.97 | 17 | 0.239 | 51% | 8.2e-17 | 8.9e-16 | 4.3e-16 | 7.8e-16 | pass | yes |
| W1 n_t=8 gathered | reference | 0.304 | 1.00 | – | 0.075 | – | 8.2e-17 | 1.2e-15 | 4.3e-16 | 8.1e-16 | pass |  |
| W1 n_t=8 gathered | scalar | 0.311 | 1.02 | – | 0.077 | – | 8.2e-17 | 1.2e-15 | 4.3e-16 | 8.1e-16 | pass |  |
| W1 n_t=8 gathered | neon | 1.205 | 3.97 | 17 | 0.297 | 63% | 8.2e-17 | 8.9e-16 | 4.3e-16 | 7.8e-16 | pass |  |
| W1 n_t=16 per-pair | reference | 0.465 | 1.00 | – | 0.115 | – | 1.1e-16 | 1.7e-15 | 2.0e-16 | 7.4e-16 | pass |  |
| W1 n_t=16 per-pair | scalar | 0.467 | 1.00 | – | 0.115 | – | 1.1e-16 | 1.7e-15 | 2.0e-16 | 7.4e-16 | pass | yes |
| W1 n_t=16 per-pair | neon | 1.141 | 2.45 | 17 | 0.282 | 60% | 1.3e-16 | 1.8e-15 | 2.0e-16 | 9.0e-16 | pass | yes |
| W1 n_t=16 gathered | reference | 0.306 | 1.00 | – | 0.075 | – | 1.1e-16 | 1.7e-15 | 2.0e-16 | 7.4e-16 | pass |  |
| W1 n_t=16 gathered | scalar | 0.318 | 1.04 | – | 0.079 | – | 1.1e-16 | 1.7e-15 | 2.0e-16 | 7.4e-16 | pass |  |
| W1 n_t=16 gathered | neon | 1.236 | 4.04 | 17 | 0.305 | 65% | 1.3e-16 | 1.8e-15 | 2.0e-16 | 9.0e-16 | pass |  |
| W1 n_t=24 per-pair | reference | 0.407 | 1.00 | – | 0.101 | – | 1.3e-16 | 1.1e-15 | 3.2e-16 | 1.2e-15 | pass |  |
| W1 n_t=24 per-pair | scalar | 0.406 | 1.00 | – | 0.100 | – | 1.3e-16 | 1.1e-15 | 3.2e-16 | 1.2e-15 | pass | yes |
| W1 n_t=24 per-pair | neon | 1.169 | 2.87 | 17 | 0.289 | 61% | 1.3e-16 | 1.2e-15 | 3.0e-16 | 1.1e-15 | pass | yes |
| W1 n_t=24 gathered | reference | 0.313 | 1.00 | – | 0.077 | – | 1.3e-16 | 1.1e-15 | 3.2e-16 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | scalar | 0.312 | 0.99 | – | 0.077 | – | 1.3e-16 | 1.1e-15 | 3.2e-16 | 1.2e-15 | pass |  |
| W1 n_t=24 gathered | neon | 1.194 | 3.81 | 17 | 0.295 | 63% | 1.3e-16 | 1.2e-15 | 3.0e-16 | 1.1e-15 | pass |  |
| W1 n_t=32 per-pair | reference | 0.318 | 1.00 | – | 0.079 | – | 1.7e-16 | 1.8e-15 | 4.5e-16 | 1.0e-15 | pass |  |
| W1 n_t=32 per-pair | scalar | 0.356 | 1.12 | – | 0.088 | – | 1.7e-16 | 1.8e-15 | 4.5e-16 | 1.0e-15 | pass | yes |
| W1 n_t=32 per-pair | neon | 1.133 | 3.56 | 17 | 0.280 | 59% | 1.6e-16 | 1.8e-15 | 3.9e-16 | 1.1e-15 | pass | yes |
| W1 n_t=32 gathered | reference | 0.308 | 1.00 | – | 0.076 | – | 1.7e-16 | 1.8e-15 | 4.5e-16 | 1.0e-15 | pass |  |
| W1 n_t=32 gathered | scalar | 0.311 | 1.01 | – | 0.077 | – | 1.7e-16 | 1.8e-15 | 4.5e-16 | 1.0e-15 | pass |  |
| W1 n_t=32 gathered | neon | 1.193 | 3.88 | 17 | 0.294 | 63% | 1.6e-16 | 1.8e-15 | 3.9e-16 | 1.1e-15 | pass |  |
| W1 n_t=64 per-pair | reference | 0.334 | 1.00 | – | 0.083 | – | 3.1e-16 | 2.4e-15 | 1.7e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 per-pair | scalar | 0.341 | 1.02 | – | 0.084 | – | 3.1e-16 | 2.4e-15 | 1.7e-15 | 2.0e-15 | pass | yes |
| W1 n_t=64 per-pair | neon | 1.223 | 3.66 | 17 | 0.302 | 64% | 2.9e-16 | 2.4e-15 | 1.9e-15 | 2.1e-15 | pass | yes |
| W1 n_t=64 gathered | reference | 0.309 | 1.00 | – | 0.076 | – | 3.1e-16 | 2.4e-15 | 1.7e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 gathered | scalar | 0.312 | 1.01 | – | 0.077 | – | 3.1e-16 | 2.4e-15 | 1.7e-15 | 2.0e-15 | pass |  |
| W1 n_t=64 gathered | neon | 1.239 | 4.01 | 17 | 0.306 | 65% | 2.9e-16 | 2.4e-15 | 1.9e-15 | 2.1e-15 | pass |  |
| W1 n_t=128 per-pair | reference | 0.324 | 1.00 | – | 0.080 | – | 2.1e-16 | 3.3e-15 | 7.7e-16 | 2.1e-15 | pass |  |
| W1 n_t=128 per-pair | scalar | 0.324 | 1.00 | – | 0.080 | – | 2.1e-16 | 3.3e-15 | 7.7e-16 | 2.1e-15 | pass | yes |
| W1 n_t=128 per-pair | neon | 1.234 | 3.80 | 17 | 0.305 | 65% | 1.9e-16 | 3.2e-15 | 9.5e-16 | 2.4e-15 | pass | yes |
| W1 n_t=128 gathered | reference | 0.311 | 1.00 | – | 0.077 | – | 2.1e-16 | 3.3e-15 | 7.7e-16 | 2.1e-15 | pass |  |
| W1 n_t=128 gathered | scalar | 0.309 | 0.99 | – | 0.076 | – | 2.1e-16 | 3.3e-15 | 7.7e-16 | 2.1e-15 | pass |  |
| W1 n_t=128 gathered | neon | 1.239 | 3.98 | 17 | 0.306 | 65% | 1.9e-16 | 3.2e-15 | 9.5e-16 | 2.4e-15 | pass |  |
| W2 N=1000 t≠s | reference | 0.471 | 1.00 | – | 0.116 | – | 2.1e-16 | 1.1e-15 | 1.2e-15 | 9.3e-16 | pass |  |
| W2 N=1000 t≠s | scalar | 0.310 | 0.66 | – | 0.077 | – | 2.1e-16 | 1.1e-15 | 1.2e-15 | 9.3e-16 | pass |  |
| W2 N=1000 t≠s | neon | 1.240 | 2.63 | 17 | 0.306 | 65% | 2.1e-16 | 1.1e-15 | 1.2e-15 | 9.2e-16 | pass |  |
| W2 N=1000 t=s | reference | 0.309 | 1.00 | – | 0.076 | – | 3.0e-16 | 1.2e-15 | 1.6e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | scalar | 0.303 | 0.98 | – | 0.075 | – | 3.0e-16 | 1.2e-15 | 1.6e-15 | 1.1e-15 | pass |  |
| W2 N=1000 t=s | neon | 1.242 | 4.02 | 17 | 0.307 | 65% | 3.0e-16 | 1.2e-15 | 1.5e-15 | 1.1e-15 | pass |  |
| W2 N=10000 t≠s | reference | 0.314 | 1.00 | – | 0.078 | – | 2.6e-16 | 3.5e-15 | 4.7e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | scalar | 0.316 | 1.01 | – | 0.078 | – | 2.6e-16 | 3.5e-15 | 4.7e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t≠s | neon | 1.241 | 3.95 | 17 | 0.306 | 65% | 2.8e-16 | 3.5e-15 | 4.6e-15 | 2.8e-15 | pass |  |
| W2 N=10000 t=s | reference | 0.311 | 1.00 | – | 0.077 | – | 6.2e-16 | 3.5e-15 | 6.8e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | scalar | 0.312 | 1.00 | – | 0.077 | – | 6.2e-16 | 3.5e-15 | 6.8e-15 | 3.3e-15 | pass |  |
| W2 N=10000 t=s | neon | 1.238 | 3.98 | 17 | 0.306 | 65% | 5.9e-16 | 3.5e-15 | 6.7e-15 | 3.3e-15 | pass |  |

### Summary

Geometric means over the cells of each group: Gpairs/s, the speed-up over the reference, and for W1 gathered the fraction of the model. Worst accuracy over the cells of all groups, and the checks passed.

| precision | output | kernel | W1 per-pair Gpairs/s (x ref) | W1 gathered Gpairs/s (x ref) | W2 Gpairs/s (x ref) | W1 gathered of model | worst max φ | worst max ∇φ | checks |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f32 | φ | reference | 0.546 (1.00) | 0.543 (1.00) | 0.687 (1.00) | – | 1.6e-7 |  | 16/16 |
| f32 | φ | scalar | 0.625 (1.14) | 0.915 (1.68) | 1.102 (1.61) | – | 1.6e-7 |  | 16/16 |
| f32 | φ | neon | 3.792 (6.94) | 4.027 (7.41) | 4.052 (5.90) | 75% | 1.6e-7 |  | 16/16 |
| f32 | φ, ∇φ | reference | 0.406 (1.00) | 0.315 (1.00) | 0.314 (1.00) | – | 1.6e-7 | 2.7e-6 | 16/16 |
| f32 | φ, ∇φ | scalar | 0.407 (1.00) | 0.316 (1.00) | 0.315 (1.01) | – | 1.6e-7 | 2.7e-6 | 16/16 |
| f32 | φ, ∇φ | neon | 2.463 (6.07) | 2.608 (8.26) | 2.627 (8.38) | 68% | 1.8e-7 | 2.6e-6 | 16/16 |
| f64 | φ | reference | 0.532 (1.00) | 0.492 (1.00) | 0.452 (1.00) | – | 6.2e-16 |  | 16/16 |
| f64 | φ | scalar | 0.606 (1.14) | 0.737 (1.50) | 1.103 (2.44) | – | 6.2e-16 |  | 16/16 |
| f64 | φ | neon | 2.103 (3.95) | 2.141 (4.35) | 2.181 (4.82) | 79% | 6.1e-16 |  | 16/16 |
| f64 | φ, ∇φ | reference | 0.384 (1.00) | 0.308 (1.00) | 0.345 (1.00) | – | 6.2e-16 | 6.8e-15 | 16/16 |
| f64 | φ, ∇φ | scalar | 0.393 (1.02) | 0.312 (1.01) | 0.310 (0.90) | – | 6.2e-16 | 6.8e-15 | 16/16 |
| f64 | φ, ∇φ | neon | 1.142 (2.97) | 1.217 (3.95) | 1.240 (3.59) | 64% | 5.9e-16 | 6.7e-15 | 16/16 |


## P2P: nd-fmm-simd against green-kernels (Phase 3S T7)

| item | value |
| --- | --- |
| CPU | Apple M3 Max |
| cores | 16 physical, 16 logical, 12 performance |
| target | aarch64-macos, release build |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| nd-fmm-simd | default `P2pKernel::detect()` = neon; ISAs available: scalar, neon |
| green-kernels | commit `7d757c579f8633d58163cd5b2d011a9468541f17`, `Laplace3dKernel::evaluate_st`, pulp backend `Neon(Neon { neon: Neon })` |
| thread variables | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, RAYON_NUM_THREADS=1 |
| mode | full |
| timing | median of 15 batches of ≥ 20 ms, one thread |

- Inputs: those of `p2p_kernels` (`nd_fmm_validate::p2p_kernels`, design §8.2). Gpairs/s counts n_s n_t pairs per evaluation, coincident ones included. "time ours/gk": our time per evaluation over green-kernels' (below 1: ours is faster).
- Accuracy against `direct_sum` on the first 4 sets of the cell, green-kernels' output times 4π: "max" is the largest error relative to the term magnitudes (Σ|q|/r for φ, Σ|q|/r² per component of ∇φ), "L2" the relative L2 error. "req. 2": within 1e-6 (f32) or 1e-14 (f64), or twice the error of `nd_fmm_ref::p2p` on the same inputs (design §3, as decided in T5).
- Target (C3S.6): speed, ours ≥ green-kernels' Gpairs/s; accuracy, our largest error (max of φ and ∇φ) ≤ green-kernels'.

### f32, potential

| cell | ours | ours Gpairs/s | green-kernels Gpairs/s | time ours/gk | max φ ours / gk | L2 φ ours / gk | max ∇φ ours / gk | L2 ∇φ ours / gk | req. 2 ours / gk | speed | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| W1 n_t=8 per-pair | neon (default) | 3.101 | 1.931 | 0.62 | 7.7e-8 / 3.0e-8 | 5.9e-7 / 3.9e-7 |  |  | pass / pass | met | worse |
| W1 n_t=8 per-pair | scalar | 0.957 | 1.931 | 2.02 | 8.8e-8 / 3.0e-8 | 4.5e-7 / 3.9e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=8 gathered | neon (default) | 3.930 | 2.886 | 0.73 | 7.7e-8 / 3.5e-8 | 5.9e-7 / 6.1e-7 |  |  | pass / pass | met | worse |
| W1 n_t=8 gathered | scalar | 0.319 | 2.886 | 9.06 | 8.8e-8 / 3.5e-8 | 4.5e-7 / 6.1e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=16 per-pair | neon (default) | 3.858 | 2.422 | 0.63 | 7.6e-8 / 1.3e-8 | 8.2e-7 / 2.9e-7 |  |  | pass / pass | met | worse |
| W1 n_t=16 per-pair | scalar | 0.779 | 2.422 | 3.11 | 8.7e-8 / 1.3e-8 | 9.4e-7 / 2.9e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=16 gathered | neon (default) | 3.920 | 2.857 | 0.73 | 7.6e-8 / 3.8e-8 | 8.2e-7 / 6.5e-7 |  |  | pass / pass | met | worse |
| W1 n_t=16 gathered | scalar | 1.095 | 2.857 | 2.61 | 8.7e-8 / 3.8e-8 | 9.4e-7 / 6.5e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=24 per-pair | neon (default) | 3.967 | 2.513 | 0.63 | 8.1e-8 / 2.1e-8 | 7.0e-7 / 2.2e-7 |  |  | pass / pass | met | worse |
| W1 n_t=24 per-pair | scalar | 0.680 | 2.513 | 3.70 | 8.7e-8 / 2.1e-8 | 6.8e-7 / 2.2e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=24 gathered | neon (default) | 4.039 | 3.006 | 0.74 | 8.1e-8 / 3.5e-8 | 7.0e-7 / 3.1e-7 |  |  | pass / pass | met | worse |
| W1 n_t=24 gathered | scalar | 1.072 | 3.006 | 2.80 | 8.7e-8 / 3.5e-8 | 6.8e-7 / 3.1e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=32 per-pair | neon (default) | 4.011 | 2.664 | 0.66 | 7.0e-8 / 1.9e-8 | 8.1e-7 / 3.4e-7 |  |  | pass / pass | met | worse |
| W1 n_t=32 per-pair | scalar | 0.582 | 2.664 | 4.57 | 5.6e-8 / 1.9e-8 | 9.1e-7 / 3.4e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=32 gathered | neon (default) | 4.063 | 3.025 | 0.74 | 7.0e-8 / 3.2e-8 | 8.1e-7 / 5.4e-7 |  |  | pass / pass | met | worse |
| W1 n_t=32 gathered | scalar | 1.059 | 3.025 | 2.86 | 5.6e-8 / 3.2e-8 | 9.1e-7 / 5.4e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=64 per-pair | neon (default) | 4.029 | 2.833 | 0.70 | 1.2e-7 / 2.1e-8 | 1.1e-6 / 3.3e-7 |  |  | pass / pass | met | worse |
| W1 n_t=64 per-pair | scalar | 0.513 | 2.833 | 5.52 | 1.3e-7 / 2.1e-8 | 1.2e-6 / 3.3e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=64 gathered | neon (default) | 4.075 | 3.052 | 0.75 | 1.2e-7 / 4.0e-8 | 1.1e-6 / 8.7e-7 |  |  | pass / pass | met | worse |
| W1 n_t=64 gathered | scalar | 1.124 | 3.052 | 2.71 | 1.3e-7 / 4.0e-8 | 1.2e-6 / 8.7e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=128 per-pair | neon (default) | 4.038 | 2.929 | 0.73 | 1.1e-7 / 1.6e-8 | 1.6e-6 / 3.5e-7 |  |  | pass / pass | met | worse |
| W1 n_t=128 per-pair | scalar | 0.463 | 2.929 | 6.33 | 1.0e-7 / 1.6e-8 | 1.6e-6 / 3.5e-7 |  |  | pass / pass | BELOW | worse |
| W1 n_t=128 gathered | neon (default) | 3.929 | 2.975 | 0.76 | 1.1e-7 / 4.5e-8 | 1.6e-6 / 1.3e-6 |  |  | pass / pass | met | worse |
| W1 n_t=128 gathered | scalar | 1.097 | 2.975 | 2.71 | 1.0e-7 / 4.5e-8 | 1.6e-6 / 1.3e-6 |  |  | pass / pass | BELOW | worse |
| W2 N=1000 t≠s | neon (default) | 4.034 | 2.861 | 0.71 | 1.2e-7 / 5.9e-8 | 6.3e-7 / 3.3e-7 |  |  | pass / pass | met | worse |
| W2 N=1000 t≠s | scalar | 1.114 | 2.861 | 2.57 | 1.1e-7 / 5.9e-8 | 6.2e-7 / 3.3e-7 |  |  | pass / pass | BELOW | worse |
| W2 N=1000 t=s | neon (default) | 4.077 | 3.021 | 0.74 | 1.3e-7 / 5.7e-8 | 6.6e-7 / 3.4e-7 |  |  | pass / pass | met | worse |
| W2 N=1000 t=s | scalar | 1.109 | 3.021 | 2.72 | 1.2e-7 / 5.7e-8 | 6.6e-7 / 3.4e-7 |  |  | pass / pass | BELOW | worse |
| W2 N=10000 t≠s | neon (default) | 4.037 | 2.961 | 0.73 | 1.6e-7 / 8.2e-8 | 1.9e-6 / 8.1e-7 |  |  | pass / pass | met | worse |
| W2 N=10000 t≠s | scalar | 1.120 | 2.961 | 2.64 | 1.5e-7 / 8.2e-8 | 1.9e-6 / 8.1e-7 |  |  | pass / pass | BELOW | worse |
| W2 N=10000 t=s | neon (default) | 4.037 | 2.966 | 0.73 | 1.5e-7 / 7.2e-8 | 1.9e-6 / 8.0e-7 |  |  | pass / pass | met | worse |
| W2 N=10000 t=s | scalar | 1.118 | 2.966 | 2.65 | 1.6e-7 / 7.2e-8 | 1.9e-6 / 8.0e-7 |  |  | pass / pass | BELOW | worse |

### f32, potential and gradient

| cell | ours | ours Gpairs/s | green-kernels Gpairs/s | time ours/gk | max φ ours / gk | L2 φ ours / gk | max ∇φ ours / gk | L2 ∇φ ours / gk | req. 2 ours / gk | speed | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| W1 n_t=8 per-pair | neon (default) | 2.114 | 1.231 | 0.58 | 7.7e-8 / 3.0e-8 | 7.1e-7 / 3.9e-7 | 1.7e-7 / 2.4e-7 | 3.3e-7 / 4.5e-7 | pass / pass | met | met |
| W1 n_t=8 per-pair | scalar | 0.521 | 1.231 | 2.36 | 8.8e-8 / 3.0e-8 | 4.5e-7 / 3.9e-7 | 1.5e-7 / 2.4e-7 | 3.0e-7 / 4.5e-7 | pass / pass | BELOW | met |
| W1 n_t=8 gathered | neon (default) | 2.575 | 2.097 | 0.81 | 7.7e-8 / 3.5e-8 | 7.1e-7 / 6.1e-7 | 1.7e-7 / 1.1e-7 | 3.3e-7 / 2.3e-7 | pass / pass | met | worse |
| W1 n_t=8 gathered | scalar | 0.321 | 2.097 | 6.54 | 8.8e-8 / 3.5e-8 | 4.5e-7 / 6.1e-7 | 1.5e-7 / 1.1e-7 | 3.0e-7 / 2.3e-7 | pass / pass | BELOW | worse |
| W1 n_t=16 per-pair | neon (default) | 2.454 | 1.611 | 0.66 | 5.3e-8 / 1.3e-8 | 8.4e-7 / 2.9e-7 | 2.4e-7 / 6.5e-8 | 5.0e-7 / 2.5e-7 | pass / pass | met | worse |
| W1 n_t=16 per-pair | scalar | 0.484 | 1.611 | 3.33 | 8.7e-8 / 1.3e-8 | 9.4e-7 / 2.9e-7 | 2.2e-7 / 6.5e-8 | 4.5e-7 / 2.5e-7 | pass / pass | BELOW | worse |
| W1 n_t=16 gathered | neon (default) | 2.632 | 2.152 | 0.82 | 5.3e-8 / 3.8e-8 | 8.4e-7 / 6.5e-7 | 2.4e-7 / 6.8e-8 | 5.0e-7 / 2.5e-7 | pass / pass | met | worse |
| W1 n_t=16 gathered | scalar | 0.317 | 2.152 | 6.79 | 8.7e-8 / 3.8e-8 | 9.4e-7 / 6.5e-7 | 2.2e-7 / 6.8e-8 | 4.5e-7 / 2.5e-7 | pass / pass | BELOW | worse |
| W1 n_t=24 per-pair | neon (default) | 2.550 | 1.750 | 0.69 | 8.1e-8 / 2.1e-8 | 6.7e-7 / 2.2e-7 | 1.3e-7 / 4.9e-8 | 5.2e-7 / 2.0e-7 | pass / pass | met | worse |
| W1 n_t=24 per-pair | scalar | 0.416 | 1.750 | 4.21 | 8.7e-8 / 2.1e-8 | 6.8e-7 / 2.2e-7 | 1.7e-7 / 4.9e-8 | 5.2e-7 / 2.0e-7 | pass / pass | BELOW | worse |
| W1 n_t=24 gathered | neon (default) | 2.636 | 2.116 | 0.80 | 8.1e-8 / 3.5e-8 | 6.7e-7 / 3.1e-7 | 1.3e-7 / 7.6e-8 | 5.2e-7 / 3.6e-7 | pass / pass | met | worse |
| W1 n_t=24 gathered | scalar | 0.315 | 2.116 | 6.71 | 8.7e-8 / 3.5e-8 | 6.8e-7 / 3.1e-7 | 1.7e-7 / 7.6e-8 | 5.2e-7 / 3.6e-7 | pass / pass | BELOW | worse |
| W1 n_t=32 per-pair | neon (default) | 2.583 | 1.823 | 0.71 | 5.9e-8 / 1.9e-8 | 8.0e-7 / 3.4e-7 | 4.9e-7 / 2.2e-7 | 5.1e-7 / 2.8e-7 | pass / pass | met | worse |
| W1 n_t=32 per-pair | scalar | 0.372 | 1.823 | 4.90 | 5.6e-8 / 1.9e-8 | 9.1e-7 / 3.4e-7 | 4.6e-7 / 2.2e-7 | 5.8e-7 / 2.8e-7 | pass / pass | BELOW | worse |
| W1 n_t=32 gathered | neon (default) | 2.650 | 2.183 | 0.82 | 5.9e-8 / 3.2e-8 | 8.0e-7 / 5.4e-7 | 4.9e-7 / 2.0e-7 | 5.1e-7 / 4.7e-7 | pass / pass | met | worse |
| W1 n_t=32 gathered | scalar | 0.315 | 2.183 | 6.93 | 5.6e-8 / 3.2e-8 | 9.1e-7 / 5.4e-7 | 4.6e-7 / 2.0e-7 | 5.8e-7 / 4.7e-7 | pass / pass | BELOW | worse |
| W1 n_t=64 per-pair | neon (default) | 2.603 | 1.995 | 0.77 | 1.3e-7 / 2.1e-8 | 1.1e-6 / 3.3e-7 | 3.2e-7 / 9.0e-8 | 6.8e-7 / 2.2e-7 | pass / pass | met | worse |
| W1 n_t=64 per-pair | scalar | 0.338 | 1.995 | 5.91 | 1.3e-7 / 2.1e-8 | 1.2e-6 / 3.3e-7 | 3.8e-7 / 9.0e-8 | 6.9e-7 / 2.2e-7 | pass / pass | BELOW | worse |
| W1 n_t=64 gathered | neon (default) | 2.653 | 2.197 | 0.83 | 1.3e-7 / 4.0e-8 | 1.1e-6 / 8.7e-7 | 3.2e-7 / 3.8e-7 | 6.8e-7 / 4.8e-7 | pass / pass | met | met |
| W1 n_t=64 gathered | scalar | 0.315 | 2.197 | 6.98 | 1.3e-7 / 4.0e-8 | 1.2e-6 / 8.7e-7 | 3.8e-7 / 3.8e-7 | 6.9e-7 / 4.8e-7 | pass / pass | BELOW | worse |
| W1 n_t=128 per-pair | neon (default) | 2.562 | 2.074 | 0.81 | 1.1e-7 / 1.6e-8 | 1.6e-6 / 3.5e-7 | 5.9e-7 / 1.6e-7 | 1.1e-6 / 2.4e-7 | pass / pass | met | worse |
| W1 n_t=128 per-pair | scalar | 0.329 | 2.074 | 6.31 | 1.0e-7 / 1.6e-8 | 1.6e-6 / 3.5e-7 | 6.2e-7 / 1.6e-7 | 1.2e-6 / 2.4e-7 | pass / pass | BELOW | worse |
| W1 n_t=128 gathered | neon (default) | 2.654 | 2.164 | 0.82 | 1.1e-7 / 4.5e-8 | 1.6e-6 / 1.3e-6 | 5.9e-7 / 4.0e-7 | 1.1e-6 / 6.4e-7 | pass / pass | met | worse |
| W1 n_t=128 gathered | scalar | 0.312 | 2.164 | 6.93 | 1.0e-7 / 4.5e-8 | 1.6e-6 / 1.3e-6 | 6.2e-7 / 4.0e-7 | 1.2e-6 / 6.4e-7 | pass / pass | BELOW | worse |
| W2 N=1000 t≠s | neon (default) | 2.658 | 2.174 | 0.82 | 1.1e-7 / 5.9e-8 | 6.2e-7 / 3.3e-7 | 7.4e-7 / 5.0e-7 | 4.4e-7 / 2.3e-7 | pass / pass | met | worse |
| W2 N=1000 t≠s | scalar | 0.312 | 2.174 | 6.97 | 1.1e-7 / 5.9e-8 | 6.2e-7 / 3.3e-7 | 7.9e-7 / 5.0e-7 | 4.4e-7 / 2.3e-7 | pass / pass | BELOW | worse |
| W2 N=1000 t=s | neon (default) | 2.605 | 2.195 | 0.84 | 1.2e-7 / 5.7e-8 | 6.6e-7 / 3.4e-7 | 1.0e-6 / 3.6e-7 | 6.7e-7 / 3.2e-7 | pass / pass | met | worse |
| W2 N=1000 t=s | scalar | 0.313 | 2.195 | 7.01 | 1.2e-7 / 5.7e-8 | 6.6e-7 / 3.4e-7 | 9.6e-7 / 3.6e-7 | 6.6e-7 / 3.2e-7 | pass / pass | BELOW | worse |
| W2 N=10000 t≠s | neon (default) | 2.505 | 2.141 | 0.85 | 1.6e-7 / 8.2e-8 | 1.9e-6 / 8.1e-7 | 2.0e-6 / 1.4e-6 | 1.2e-6 / 8.9e-7 | pass / pass | met | worse |
| W2 N=10000 t≠s | scalar | 0.313 | 2.141 | 6.85 | 1.5e-7 / 8.2e-8 | 1.9e-6 / 8.1e-7 | 2.0e-6 / 1.4e-6 | 1.2e-6 / 8.9e-7 | pass / pass | BELOW | worse |
| W2 N=10000 t=s | neon (default) | 2.630 | 2.140 | 0.81 | 1.8e-7 / 7.2e-8 | 1.9e-6 / 8.0e-7 | 2.6e-6 / 1.3e-6 | 1.4e-6 / 8.2e-7 | pass / pass | met | worse |
| W2 N=10000 t=s | scalar | 0.312 | 2.140 | 6.85 | 1.6e-7 / 7.2e-8 | 1.9e-6 / 8.0e-7 | 2.7e-6 / 1.3e-6 | 1.4e-6 / 8.2e-7 | pass / pass | BELOW | worse |

### f64, potential

| cell | ours | ours Gpairs/s | green-kernels Gpairs/s | time ours/gk | max φ ours / gk | L2 φ ours / gk | max ∇φ ours / gk | L2 ∇φ ours / gk | req. 2 ours / gk | speed | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| W1 n_t=8 per-pair | neon (default) | 1.899 | 0.998 | 0.53 | 1.2e-16 / 4.4e-17 | 1.2e-15 / 8.6e-16 |  |  | pass / pass | met | worse |
| W1 n_t=8 per-pair | scalar | 0.924 | 0.998 | 1.08 | 8.2e-17 / 4.4e-17 | 1.2e-15 / 8.6e-16 |  |  | pass / pass | BELOW | worse |
| W1 n_t=8 gathered | neon (default) | 2.123 | 1.209 | 0.57 | 1.2e-16 / 1.2e-16 | 1.2e-15 / 1.4e-15 |  |  | pass / pass | met | worse |
| W1 n_t=8 gathered | scalar | 1.041 | 1.209 | 1.16 | 8.2e-17 / 1.2e-16 | 1.2e-15 / 1.4e-15 |  |  | pass / pass | BELOW | met |
| W1 n_t=16 per-pair | neon (default) | 2.117 | 1.121 | 0.53 | 1.2e-16 / 4.4e-17 | 2.0e-15 / 9.9e-16 |  |  | pass / pass | met | worse |
| W1 n_t=16 per-pair | scalar | 0.761 | 1.121 | 1.47 | 1.1e-16 / 4.4e-17 | 1.7e-15 / 9.9e-16 |  |  | pass / pass | BELOW | worse |
| W1 n_t=16 gathered | neon (default) | 2.163 | 1.261 | 0.58 | 1.2e-16 / 6.6e-17 | 2.0e-15 / 1.4e-15 |  |  | pass / pass | met | worse |
| W1 n_t=16 gathered | scalar | 0.315 | 1.261 | 4.00 | 1.1e-16 / 6.6e-17 | 1.7e-15 / 1.4e-15 |  |  | pass / pass | BELOW | worse |
| W1 n_t=24 per-pair | neon (default) | 2.148 | 1.176 | 0.55 | 1.6e-16 / 5.1e-17 | 1.3e-15 / 4.0e-16 |  |  | pass / pass | met | worse |
| W1 n_t=24 per-pair | scalar | 0.662 | 1.176 | 1.78 | 1.3e-16 / 5.1e-17 | 1.1e-15 / 4.0e-16 |  |  | pass / pass | BELOW | worse |
| W1 n_t=24 gathered | neon (default) | 2.196 | 1.268 | 0.58 | 1.6e-16 / 1.6e-16 | 1.3e-15 / 1.4e-15 |  |  | pass / pass | met | worse |
| W1 n_t=24 gathered | scalar | 1.113 | 1.268 | 1.14 | 1.3e-16 / 1.6e-16 | 1.1e-15 / 1.4e-15 |  |  | pass / pass | BELOW | met |
| W1 n_t=32 per-pair | neon (default) | 2.116 | 1.168 | 0.55 | 1.6e-16 / 3.2e-17 | 1.7e-15 / 6.0e-16 |  |  | pass / pass | met | worse |
| W1 n_t=32 per-pair | scalar | 0.578 | 1.168 | 2.02 | 1.7e-16 / 3.2e-17 | 1.8e-15 / 6.0e-16 |  |  | pass / pass | BELOW | worse |
| W1 n_t=32 gathered | neon (default) | 2.199 | 1.273 | 0.58 | 1.6e-16 / 1.2e-16 | 1.7e-15 / 1.4e-15 |  |  | pass / pass | met | worse |
| W1 n_t=32 gathered | scalar | 1.113 | 1.273 | 1.14 | 1.7e-16 / 1.2e-16 | 1.8e-15 / 1.4e-15 |  |  | pass / pass | BELOW | worse |
| W1 n_t=64 per-pair | neon (default) | 2.185 | 1.233 | 0.56 | 2.4e-16 / 3.2e-17 | 2.5e-15 / 8.0e-16 |  |  | pass / pass | met | worse |
| W1 n_t=64 per-pair | scalar | 0.509 | 1.233 | 2.42 | 3.1e-16 / 3.2e-17 | 2.4e-15 / 8.0e-16 |  |  | pass / pass | BELOW | worse |
| W1 n_t=64 gathered | neon (default) | 2.206 | 1.282 | 0.58 | 2.4e-16 / 1.3e-16 | 2.5e-15 / 1.8e-15 |  |  | pass / pass | met | worse |
| W1 n_t=64 gathered | scalar | 1.117 | 1.282 | 1.15 | 3.1e-16 / 1.3e-16 | 2.4e-15 / 1.8e-15 |  |  | pass / pass | BELOW | worse |
| W1 n_t=128 per-pair | neon (default) | 2.191 | 1.256 | 0.57 | 2.2e-16 / 2.6e-17 | 3.3e-15 / 7.4e-16 |  |  | pass / pass | met | worse |
| W1 n_t=128 per-pair | scalar | 0.390 | 1.256 | 3.22 | 2.1e-16 / 2.6e-17 | 3.3e-15 / 7.4e-16 |  |  | pass / pass | BELOW | worse |
| W1 n_t=128 gathered | neon (default) | 2.209 | 1.275 | 0.58 | 2.2e-16 / 1.7e-16 | 3.3e-15 / 2.8e-15 |  |  | pass / pass | met | worse |
| W1 n_t=128 gathered | scalar | 1.122 | 1.275 | 1.14 | 2.1e-16 / 1.7e-16 | 3.3e-15 / 2.8e-15 |  |  | pass / pass | BELOW | worse |
| W2 N=1000 t≠s | neon (default) | 2.201 | 1.280 | 0.58 | 2.3e-16 / 1.8e-16 | 1.1e-15 / 9.2e-16 |  |  | pass / pass | met | worse |
| W2 N=1000 t≠s | scalar | 1.101 | 1.280 | 1.16 | 2.1e-16 / 1.8e-16 | 1.1e-15 / 9.2e-16 |  |  | pass / pass | BELOW | worse |
| W2 N=1000 t=s | neon (default) | 2.201 | 1.278 | 0.58 | 2.5e-16 / 2.0e-16 | 1.2e-15 / 9.4e-16 |  |  | pass / pass | met | worse |
| W2 N=1000 t=s | scalar | 1.108 | 1.278 | 1.15 | 3.0e-16 / 2.0e-16 | 1.2e-15 / 9.4e-16 |  |  | pass / pass | BELOW | worse |
| W2 N=10000 t≠s | neon (default) | 2.205 | 1.266 | 0.57 | 2.7e-16 / 2.1e-16 | 3.5e-15 / 2.2e-15 |  |  | pass / pass | met | worse |
| W2 N=10000 t≠s | scalar | 1.112 | 1.266 | 1.14 | 2.6e-16 / 2.1e-16 | 3.5e-15 / 2.2e-15 |  |  | pass / pass | BELOW | worse |
| W2 N=10000 t=s | neon (default) | 2.205 | 1.265 | 0.57 | 6.1e-16 / 1.9e-16 | 3.5e-15 / 2.2e-15 |  |  | pass / pass | met | worse |
| W2 N=10000 t=s | scalar | 1.112 | 1.265 | 1.14 | 6.2e-16 / 1.9e-16 | 3.5e-15 / 2.2e-15 |  |  | pass / pass | BELOW | worse |

### f64, potential and gradient

| cell | ours | ours Gpairs/s | green-kernels Gpairs/s | time ours/gk | max φ ours / gk | L2 φ ours / gk | max ∇φ ours / gk | L2 ∇φ ours / gk | req. 2 ours / gk | speed | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| W1 n_t=8 per-pair | neon (default) | 0.950 | 0.716 | 0.75 | 8.2e-17 / 4.4e-17 | 8.9e-16 / 8.6e-16 | 4.3e-16 / 4.3e-16 | 7.8e-16 / 7.8e-16 | pass / pass | met | met |
| W1 n_t=8 per-pair | scalar | 0.478 | 0.716 | 1.50 | 8.2e-17 / 4.4e-17 | 1.2e-15 / 8.6e-16 | 4.3e-16 / 4.3e-16 | 8.1e-16 / 7.8e-16 | pass / pass | BELOW | met |
| W1 n_t=8 gathered | neon (default) | 1.229 | 0.876 | 0.71 | 8.2e-17 / 1.2e-16 | 8.9e-16 / 1.4e-15 | 4.3e-16 / 2.6e-16 | 7.8e-16 / 6.1e-16 | pass / pass | met | worse |
| W1 n_t=8 gathered | scalar | 0.319 | 0.876 | 2.74 | 8.2e-17 / 1.2e-16 | 1.2e-15 / 1.4e-15 | 4.3e-16 / 2.6e-16 | 8.1e-16 / 6.1e-16 | pass / pass | BELOW | worse |
| W1 n_t=16 per-pair | neon (default) | 1.129 | 0.790 | 0.70 | 1.3e-16 / 4.4e-17 | 1.8e-15 / 9.9e-16 | 2.0e-16 / 1.0e-16 | 9.0e-16 / 4.7e-16 | pass / pass | met | worse |
| W1 n_t=16 per-pair | scalar | 0.464 | 0.790 | 1.70 | 1.1e-16 / 4.4e-17 | 1.7e-15 / 9.9e-16 | 2.0e-16 / 1.0e-16 | 7.4e-16 / 4.7e-16 | pass / pass | BELOW | worse |
| W1 n_t=16 gathered | neon (default) | 1.238 | 0.905 | 0.73 | 1.3e-16 / 6.6e-17 | 1.8e-15 / 1.4e-15 | 2.0e-16 / 2.6e-16 | 9.0e-16 / 8.1e-16 | pass / pass | met | met |
| W1 n_t=16 gathered | scalar | 0.316 | 0.905 | 2.86 | 1.1e-16 / 6.6e-17 | 1.7e-15 / 1.4e-15 | 2.0e-16 / 2.6e-16 | 7.4e-16 / 8.1e-16 | pass / pass | BELOW | met |
| W1 n_t=24 per-pair | neon (default) | 1.173 | 0.832 | 0.71 | 1.3e-16 / 5.1e-17 | 1.2e-15 / 4.0e-16 | 3.0e-16 / 1.3e-16 | 1.1e-15 / 3.8e-16 | pass / pass | met | worse |
| W1 n_t=24 per-pair | scalar | 0.405 | 0.832 | 2.05 | 1.3e-16 / 5.1e-17 | 1.1e-15 / 4.0e-16 | 3.2e-16 / 1.3e-16 | 1.2e-15 / 3.8e-16 | pass / pass | BELOW | worse |
| W1 n_t=24 gathered | neon (default) | 1.241 | 0.909 | 0.73 | 1.3e-16 / 1.6e-16 | 1.2e-15 / 1.4e-15 | 3.0e-16 / 3.0e-16 | 1.1e-15 / 8.0e-16 | pass / pass | met | met |
| W1 n_t=24 gathered | scalar | 0.312 | 0.909 | 2.91 | 1.3e-16 / 1.6e-16 | 1.1e-15 / 1.4e-15 | 3.2e-16 / 3.0e-16 | 1.2e-15 / 8.0e-16 | pass / pass | BELOW | worse |
| W1 n_t=32 per-pair | neon (default) | 1.196 | 0.854 | 0.71 | 1.6e-16 / 3.2e-17 | 1.8e-15 / 6.0e-16 | 3.9e-16 / 2.4e-16 | 1.1e-15 / 4.6e-16 | pass / pass | met | worse |
| W1 n_t=32 per-pair | scalar | 0.376 | 0.854 | 2.27 | 1.7e-16 / 3.2e-17 | 1.8e-15 / 6.0e-16 | 4.5e-16 / 2.4e-16 | 1.0e-15 / 4.6e-16 | pass / pass | BELOW | worse |
| W1 n_t=32 gathered | neon (default) | 1.241 | 0.910 | 0.73 | 1.6e-16 / 1.2e-16 | 1.8e-15 / 1.4e-15 | 3.9e-16 / 3.3e-16 | 1.1e-15 / 7.4e-16 | pass / pass | met | worse |
| W1 n_t=32 gathered | scalar | 0.316 | 0.910 | 2.88 | 1.7e-16 / 1.2e-16 | 1.8e-15 / 1.4e-15 | 4.5e-16 / 3.3e-16 | 1.0e-15 / 7.4e-16 | pass / pass | BELOW | worse |
| W1 n_t=64 per-pair | neon (default) | 1.223 | 0.886 | 0.72 | 2.9e-16 / 3.2e-17 | 2.4e-15 / 8.0e-16 | 1.9e-15 / 1.6e-16 | 2.1e-15 / 4.3e-16 | pass / pass | met | worse |
| W1 n_t=64 per-pair | scalar | 0.343 | 0.886 | 2.58 | 3.1e-16 / 3.2e-17 | 2.4e-15 / 8.0e-16 | 1.7e-15 / 1.6e-16 | 2.0e-15 / 4.3e-16 | pass / pass | BELOW | worse |
| W1 n_t=64 gathered | neon (default) | 1.247 | 0.912 | 0.73 | 2.9e-16 / 1.3e-16 | 2.4e-15 / 1.8e-15 | 1.9e-15 / 2.3e-15 | 2.1e-15 / 2.5e-15 | pass / pass | met | met |
| W1 n_t=64 gathered | scalar | 0.314 | 0.912 | 2.90 | 3.1e-16 / 1.3e-16 | 2.4e-15 / 1.8e-15 | 1.7e-15 / 2.3e-15 | 2.0e-15 / 2.5e-15 | pass / pass | BELOW | met |
| W1 n_t=128 per-pair | neon (default) | 1.236 | 0.899 | 0.73 | 1.9e-16 / 2.6e-17 | 3.2e-15 / 7.4e-16 | 9.5e-16 / 1.7e-16 | 2.4e-15 / 5.4e-16 | pass / pass | met | worse |
| W1 n_t=128 per-pair | scalar | 0.327 | 0.899 | 2.75 | 2.1e-16 / 2.6e-17 | 3.3e-15 / 7.4e-16 | 7.7e-16 / 1.7e-16 | 2.1e-15 / 5.4e-16 | pass / pass | BELOW | worse |
| W1 n_t=128 gathered | neon (default) | 1.241 | 0.912 | 0.73 | 1.9e-16 / 1.7e-16 | 3.2e-15 / 2.8e-15 | 9.5e-16 / 1.5e-15 | 2.4e-15 / 2.1e-15 | pass / pass | met | met |
| W1 n_t=128 gathered | scalar | 0.314 | 0.912 | 2.90 | 2.1e-16 / 1.7e-16 | 3.3e-15 / 2.8e-15 | 7.7e-16 / 1.5e-15 | 2.1e-15 / 2.1e-15 | pass / pass | BELOW | met |
| W2 N=1000 t≠s | neon (default) | 1.240 | 0.913 | 0.74 | 2.1e-16 / 1.8e-16 | 1.1e-15 / 9.2e-16 | 1.2e-15 / 8.5e-16 | 9.2e-16 / 6.2e-16 | pass / pass | met | worse |
| W2 N=1000 t≠s | scalar | 0.313 | 0.913 | 2.91 | 2.1e-16 / 1.8e-16 | 1.1e-15 / 9.2e-16 | 1.2e-15 / 8.5e-16 | 9.3e-16 / 6.2e-16 | pass / pass | BELOW | worse |
| W2 N=1000 t=s | neon (default) | 1.245 | 0.913 | 0.73 | 3.0e-16 / 2.0e-16 | 1.2e-15 / 9.4e-16 | 1.5e-15 / 1.1e-15 | 1.1e-15 / 8.3e-16 | pass / pass | met | worse |
| W2 N=1000 t=s | scalar | 0.313 | 0.913 | 2.92 | 3.0e-16 / 2.0e-16 | 1.2e-15 / 9.4e-16 | 1.6e-15 / 1.1e-15 | 1.1e-15 / 8.3e-16 | pass / pass | BELOW | worse |
| W2 N=10000 t≠s | neon (default) | 1.247 | 0.892 | 0.71 | 2.8e-16 / 2.1e-16 | 3.5e-15 / 2.2e-15 | 4.6e-15 / 3.4e-15 | 2.8e-15 / 1.8e-15 | pass / pass | met | worse |
| W2 N=10000 t≠s | scalar | 0.311 | 0.892 | 2.86 | 2.6e-16 / 2.1e-16 | 3.5e-15 / 2.2e-15 | 4.7e-15 / 3.4e-15 | 2.8e-15 / 1.8e-15 | pass / pass | BELOW | worse |
| W2 N=10000 t=s | neon (default) | 1.236 | 0.903 | 0.73 | 5.9e-16 / 1.9e-16 | 3.5e-15 / 2.2e-15 | 6.7e-15 / 3.6e-15 | 3.3e-15 / 2.3e-15 | pass / pass | met | worse |
| W2 N=10000 t=s | scalar | 0.312 | 0.903 | 2.90 | 6.2e-16 / 1.9e-16 | 3.5e-15 / 2.2e-15 | 6.8e-15 / 3.6e-15 | 3.3e-15 / 2.3e-15 | pass / pass | BELOW | worse |

### The C3S.6 target, default kernel

Speed-up over green-kernels (its time over ours), geometric mean over the cells of each group, its smallest value, and the cells where the target is not met.

| precision | output | W1 per-pair | W1 gathered | W2 | smallest (cell) | speed met | accuracy met | cells below in speed | cells worse in accuracy (ours / gk max error) |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| f32 | φ | 1.51 | 1.35 | 1.37 | 1.32 (W1 n_t=128 gathered) | 16/16 | 0/16 | none | W1 n_t=8 per-pair (7.7e-8 / 3.0e-8); W1 n_t=8 gathered (7.7e-8 / 3.5e-8); W1 n_t=16 per-pair (7.6e-8 / 1.3e-8); W1 n_t=16 gathered (7.6e-8 / 3.8e-8); W1 n_t=24 per-pair (8.1e-8 / 2.1e-8); W1 n_t=24 gathered (8.1e-8 / 3.5e-8); W1 n_t=32 per-pair (7.0e-8 / 1.9e-8); W1 n_t=32 gathered (7.0e-8 / 3.2e-8); W1 n_t=64 per-pair (1.2e-7 / 2.1e-8); W1 n_t=64 gathered (1.2e-7 / 4.0e-8); W1 n_t=128 per-pair (1.1e-7 / 1.6e-8); W1 n_t=128 gathered (1.1e-7 / 4.5e-8); W2 N=1000 t≠s (1.2e-7 / 5.9e-8); W2 N=1000 t=s (1.3e-7 / 5.7e-8); W2 N=10000 t≠s (1.6e-7 / 8.2e-8); W2 N=10000 t=s (1.5e-7 / 7.2e-8) |
| f32 | φ, ∇φ | 1.43 | 1.22 | 1.20 | 1.17 (W2 N=10000 t≠s) | 16/16 | 2/16 | none | W1 n_t=8 gathered (1.7e-7 / 1.1e-7); W1 n_t=16 per-pair (2.4e-7 / 6.5e-8); W1 n_t=16 gathered (2.4e-7 / 6.8e-8); W1 n_t=24 per-pair (1.3e-7 / 4.9e-8); W1 n_t=24 gathered (1.3e-7 / 7.6e-8); W1 n_t=32 per-pair (4.9e-7 / 2.2e-7); W1 n_t=32 gathered (4.9e-7 / 2.0e-7); W1 n_t=64 per-pair (3.2e-7 / 9.0e-8); W1 n_t=128 per-pair (5.9e-7 / 1.6e-7); W1 n_t=128 gathered (5.9e-7 / 4.0e-7); W2 N=1000 t≠s (7.4e-7 / 5.0e-7); W2 N=1000 t=s (1.0e-6 / 3.6e-7); W2 N=10000 t≠s (2.0e-6 / 1.4e-6); W2 N=10000 t=s (2.6e-6 / 1.3e-6) |
| f64 | φ | 1.82 | 1.73 | 1.73 | 1.72 (W1 n_t=16 gathered) | 16/16 | 0/16 | none | W1 n_t=8 per-pair (1.2e-16 / 4.4e-17); W1 n_t=8 gathered (1.2e-16 / 1.2e-16); W1 n_t=16 per-pair (1.2e-16 / 4.4e-17); W1 n_t=16 gathered (1.2e-16 / 6.6e-17); W1 n_t=24 per-pair (1.6e-16 / 5.1e-17); W1 n_t=24 gathered (1.6e-16 / 1.6e-16); W1 n_t=32 per-pair (1.6e-16 / 3.2e-17); W1 n_t=32 gathered (1.6e-16 / 1.2e-16); W1 n_t=64 per-pair (2.4e-16 / 3.2e-17); W1 n_t=64 gathered (2.4e-16 / 1.3e-16); W1 n_t=128 per-pair (2.2e-16 / 2.6e-17); W1 n_t=128 gathered (2.2e-16 / 1.7e-16); W2 N=1000 t≠s (2.3e-16 / 1.8e-16); W2 N=1000 t=s (2.5e-16 / 2.0e-16); W2 N=10000 t≠s (2.7e-16 / 2.1e-16); W2 N=10000 t=s (6.1e-16 / 1.9e-16) |
| f64 | φ, ∇φ | 1.39 | 1.37 | 1.37 | 1.33 (W1 n_t=8 per-pair) | 16/16 | 5/16 | none | W1 n_t=8 gathered (4.3e-16 / 2.6e-16); W1 n_t=16 per-pair (2.0e-16 / 1.0e-16); W1 n_t=24 per-pair (3.0e-16 / 1.3e-16); W1 n_t=32 per-pair (3.9e-16 / 2.4e-16); W1 n_t=32 gathered (3.9e-16 / 3.3e-16); W1 n_t=64 per-pair (1.9e-15 / 1.6e-16); W1 n_t=128 per-pair (9.5e-16 / 1.7e-16); W2 N=1000 t≠s (1.2e-15 / 8.5e-16); W2 N=1000 t=s (1.5e-15 / 1.1e-15); W2 N=10000 t≠s (4.6e-15 / 3.4e-15); W2 N=10000 t=s (6.7e-15 / 3.6e-15) |


## The SIMD P2P kernel in the FMM, and the leaf size (Phase 3S T7)

- Problems: the T12 calibration problems, the uniform cube and the Plummer sphere, N = 100000, sources equal to targets, 8 charge vectors uniform in [-1, 1), an adaptive tree with `max_level` 16, the default M2L strategy (`Auto`), gradients on; p = [3, 8], f64 and f32.
- Error measure: relative L2 error of φ and of ∇φ at 1000 sampled targets against `direct_sum` in f64, the root mean square over the charge vectors (f32 runs: the charges rounded to f32 and their oracle).
- Timings: wall time in ms of one evaluation, the mean over the charge vectors; upward = P2M and M2M, downward = L2L, M2L and P2L, leaves = L2P, M2P and P2P; far = upward + downward, near share = leaves / evaluate.
- Machine: Apple M3 Max; 16 physical, 16 logical, 12 performance; aarch64-macos, release build; rustc 1.99.0 (b940084d7 2026-09-28); one rank, 1 thread. Wall time of all runs 503.6 s.
- Threading: rayon threads 1; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- P2P kernels: `auto` = neon on this machine; ISAs available: scalar, neon.

### P2P kernels in the FMM

64 points per leaf as the refinement target. "x": the reference's time over the kernel's, for the leaf stage and the evaluation. Error ratios: the kernel's error over the reference's.

| distribution | precision | p | P2P | ran as | φ L2 | ∇φ L2 | φ / ref | ∇φ / ref | build | upward | downward | leaves | evaluate | near share | leaves x | evaluate x |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | f64 | 3 | reference | reference | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 53.2 | 2.6 | 38.2 | 160.2 | 201.5 | 80% | 1.00 | 1.00 |
| cube | f64 | 3 | auto | neon | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 49.4 | 2.5 | 37.7 | 60.4 | 101.0 | 60% | 2.65 | 1.99 |
| cube | f64 | 3 | scalar | scalar | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 49.1 | 2.6 | 37.9 | 159.1 | 200.0 | 80% | 1.01 | 1.01 |
| cube | f64 | 8 | reference | reference | 1.816e-5 | 1.903e-5 | 1.000000 | 1.000000 | 275.6 | 18.5 | 742.0 | 181.2 | 942.2 | 19% | 1.00 | 1.00 |
| cube | f64 | 8 | auto | neon | 1.816e-5 | 1.903e-5 | 1.000000 | 1.000000 | 267.7 | 18.6 | 740.3 | 86.6 | 845.9 | 10% | 2.09 | 1.11 |
| cube | f64 | 8 | scalar | scalar | 1.816e-5 | 1.903e-5 | 1.000000 | 1.000000 | 267.1 | 19.5 | 757.3 | 186.2 | 963.6 | 19% | 0.97 | 0.98 |
| cube | f32 | 3 | reference | reference | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 49.7 | 2.5 | 33.3 | 151.1 | 187.3 | 81% | 1.00 | 1.00 |
| cube | f32 | 3 | auto | neon | 2.613e-3 | 1.446e-3 | 1.000001 | 1.000001 | 49.3 | 2.7 | 34.7 | 36.7 | 74.4 | 49% | 4.12 | 2.52 |
| cube | f32 | 3 | scalar | scalar | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 57.6 | 2.6 | 35.4 | 163.6 | 202.0 | 81% | 0.92 | 0.93 |
| cube | f32 | 8 | reference | reference | 1.826e-5 | 1.910e-5 | 1.000000 | 1.000000 | 281.9 | 16.6 | 427.1 | 180.8 | 624.9 | 29% | 1.00 | 1.00 |
| cube | f32 | 8 | auto | neon | 1.827e-5 | 1.910e-5 | 1.000167 | 1.000050 | 280.9 | 16.3 | 436.4 | 62.2 | 515.4 | 12% | 2.90 | 1.21 |
| cube | f32 | 8 | scalar | scalar | 1.826e-5 | 1.910e-5 | 1.000000 | 1.000000 | 285.2 | 17.0 | 432.0 | 181.2 | 630.7 | 29% | 1.00 | 0.99 |
| plummer | f64 | 3 | reference | reference | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 71.6 | 2.7 | 84.2 | 305.0 | 392.5 | 78% | 1.00 | 1.00 |
| plummer | f64 | 3 | auto | neon | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 70.3 | 2.7 | 83.7 | 193.4 | 280.2 | 69% | 1.58 | 1.40 |
| plummer | f64 | 3 | scalar | scalar | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 70.2 | 2.7 | 83.6 | 303.5 | 390.5 | 78% | 1.00 | 1.01 |
| plummer | f64 | 8 | reference | reference | 2.187e-5 | 1.387e-5 | 1.000000 | 1.000000 | 303.4 | 21.1 | 1148.9 | 739.2 | 1909.7 | 39% | 1.00 | 1.00 |
| plummer | f64 | 8 | auto | neon | 2.187e-5 | 1.387e-5 | 1.000000 | 1.000000 | 297.7 | 21.2 | 1157.2 | 632.5 | 1811.4 | 35% | 1.17 | 1.05 |
| plummer | f64 | 8 | scalar | scalar | 2.187e-5 | 1.387e-5 | 1.000000 | 1.000000 | 297.5 | 21.1 | 1167.2 | 745.9 | 1934.7 | 39% | 0.99 | 0.99 |
| plummer | f32 | 3 | reference | reference | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 70.9 | 2.7 | 78.2 | 302.2 | 383.6 | 79% | 1.00 | 1.00 |
| plummer | f32 | 3 | auto | neon | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000001 | 70.6 | 2.6 | 78.1 | 165.4 | 246.6 | 67% | 1.83 | 1.56 |
| plummer | f32 | 3 | scalar | scalar | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 74.8 | 2.6 | 78.1 | 303.3 | 384.3 | 79% | 1.00 | 1.00 |
| plummer | f32 | 8 | reference | reference | 2.196e-5 | 1.394e-5 | 1.000000 | 1.000000 | 294.8 | 17.9 | 751.6 | 740.9 | 1510.8 | 49% | 1.00 | 1.00 |
| plummer | f32 | 8 | auto | neon | 2.196e-5 | 1.395e-5 | 0.999940 | 1.000406 | 299.3 | 17.5 | 731.4 | 602.4 | 1351.8 | 45% | 1.23 | 1.12 |
| plummer | f32 | 8 | scalar | scalar | 2.196e-5 | 1.394e-5 | 1.000000 | 1.000000 | 291.6 | 17.2 | 727.4 | 724.8 | 1470.0 | 49% | 1.02 | 1.03 |

### Leaf-size study (P2P `auto`)

| distribution | precision | p | max points per leaf | leaves | points per leaf (min / mean / max) | V pairs | φ L2 | ∇φ L2 | upward | downward | far | leaves | evaluate | near share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | f64 | 3 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 2.769e-3 | 2.232e-3 | 4.7 | 339.2 | 344.0 | 70.1 | 414.8 | 17% |
| cube | f64 | 3 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 2.616e-3 | 1.489e-3 | 2.7 | 56.7 | 59.4 | 119.6 | 179.5 | 67% |
| cube | f64 | 3 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 2.5 | 38.2 | 40.7 | 61.1 | 102.2 | 60% |
| cube | f64 | 3 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 2.6 | 38.8 | 41.4 | 61.4 | 103.3 | 59% |
| cube | f64 | 3 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 2.295e-3 | 8.342e-4 | 2.3 | 3.4 | 5.7 | 344.8 | 351.0 | 98% |
| cube | f64 | 8 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 1.893e-5 | 2.804e-5 | 55.0 | 6517.2 | 6572.2 | 191.6 | 6764.6 | 3% |
| cube | f64 | 8 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 1.816e-5 | 1.950e-5 | 20.6 | 851.5 | 872.1 | 335.2 | 1207.7 | 28% |
| cube | f64 | 8 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 1.816e-5 | 1.903e-5 | 18.6 | 739.3 | 757.9 | 86.2 | 844.5 | 10% |
| cube | f64 | 8 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 1.816e-5 | 1.903e-5 | 18.4 | 746.8 | 765.2 | 87.4 | 853.1 | 10% |
| cube | f64 | 8 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 1.596e-5 | 1.087e-5 | 15.4 | 65.4 | 80.8 | 366.7 | 448.0 | 82% |
| cube | f32 | 3 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 2.769e-3 | 2.232e-3 | 4.5 | 307.9 | 312.5 | 73.1 | 386.3 | 19% |
| cube | f32 | 3 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 2.616e-3 | 1.489e-3 | 2.7 | 52.6 | 55.3 | 96.6 | 152.2 | 63% |
| cube | f32 | 3 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 2.6 | 34.9 | 37.6 | 37.3 | 75.3 | 50% |
| cube | f32 | 3 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 2.5 | 34.4 | 37.0 | 36.5 | 73.8 | 49% |
| cube | f32 | 3 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 2.295e-3 | 8.343e-4 | 2.3 | 3.0 | 5.3 | 167.4 | 173.0 | 97% |
| cube | f32 | 8 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 1.910e-5 | 2.805e-5 | 37.7 | 3717.0 | 3754.7 | 190.2 | 3945.7 | 5% |
| cube | f32 | 8 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 1.825e-5 | 1.957e-5 | 18.0 | 524.3 | 542.3 | 310.9 | 853.6 | 36% |
| cube | f32 | 8 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 1.827e-5 | 1.910e-5 | 16.7 | 420.8 | 437.5 | 61.1 | 498.9 | 12% |
| cube | f32 | 8 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 1.827e-5 | 1.910e-5 | 17.0 | 426.9 | 443.8 | 61.8 | 505.9 | 12% |
| cube | f32 | 8 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 1.607e-5 | 1.181e-5 | 13.7 | 37.4 | 51.1 | 191.3 | 242.7 | 79% |
| plummer | f64 | 3 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 3.306e-3 | 1.545e-3 | 4.1 | 243.4 | 247.5 | 143.3 | 391.4 | 37% |
| plummer | f64 | 3 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 3.260e-3 | 1.377e-3 | 3.2 | 138.1 | 141.3 | 167.0 | 308.8 | 54% |
| plummer | f64 | 3 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 3.202e-3 | 1.116e-3 | 2.8 | 83.7 | 86.5 | 193.9 | 280.8 | 69% |
| plummer | f64 | 3 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 3.160e-3 | 1.008e-3 | 2.5 | 63.0 | 65.5 | 240.2 | 306.1 | 78% |
| plummer | f64 | 3 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 3.072e-3 | 8.782e-4 | 2.4 | 55.9 | 58.3 | 326.8 | 385.5 | 85% |
| plummer | f64 | 8 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 2.269e-5 | 2.047e-5 | 42.4 | 4370.6 | 4413.0 | 502.5 | 4916.2 | 10% |
| plummer | f64 | 8 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 2.236e-5 | 1.864e-5 | 28.0 | 2244.3 | 2272.3 | 581.9 | 2854.7 | 20% |
| plummer | f64 | 8 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 2.187e-5 | 1.387e-5 | 20.9 | 1132.0 | 1152.9 | 623.0 | 1776.5 | 35% |
| plummer | f64 | 8 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 2.142e-5 | 1.290e-5 | 18.3 | 716.3 | 734.6 | 696.6 | 1431.6 | 49% |
| plummer | f64 | 8 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 2.092e-5 | 1.072e-5 | 16.1 | 524.3 | 540.4 | 826.8 | 1367.7 | 60% |
| plummer | f32 | 3 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 3.306e-3 | 1.545e-3 | 3.9 | 215.4 | 219.2 | 138.0 | 357.8 | 39% |
| plummer | f32 | 3 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 3.260e-3 | 1.377e-3 | 3.1 | 127.8 | 130.9 | 157.0 | 288.4 | 54% |
| plummer | f32 | 3 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 3.202e-3 | 1.116e-3 | 2.6 | 77.3 | 79.9 | 165.5 | 245.9 | 67% |
| plummer | f32 | 3 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 3.160e-3 | 1.008e-3 | 2.5 | 59.9 | 62.4 | 192.2 | 255.0 | 75% |
| plummer | f32 | 3 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 3.072e-3 | 8.782e-4 | 2.4 | 53.5 | 55.8 | 239.1 | 295.2 | 81% |
| plummer | f32 | 8 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 2.286e-5 | 2.051e-5 | 30.2 | 2547.4 | 2577.6 | 490.6 | 3068.9 | 16% |
| plummer | f32 | 8 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 2.250e-5 | 1.869e-5 | 22.1 | 1335.4 | 1357.5 | 562.5 | 1920.5 | 29% |
| plummer | f32 | 8 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 2.196e-5 | 1.395e-5 | 17.6 | 735.5 | 753.1 | 598.6 | 1352.1 | 44% |
| plummer | f32 | 8 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 2.149e-5 | 1.298e-5 | 15.6 | 491.8 | 507.4 | 637.9 | 1145.7 | 56% |
| plummer | f32 | 8 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 2.096e-5 | 1.085e-5 | 15.2 | 397.9 | 413.2 | 731.2 | 1144.8 | 64% |

#### Fastest leaf size per configuration

| distribution | precision | p | fastest | evaluate (ms) | at 64 (ms) | 64 / fastest | evaluate at 16, 32, 64, 128, 256 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cube | f64 | 3 | 64 | 102.2 | 102.2 | 1.00 | 414.8, 179.5, 102.2, 103.3, 351.0 |
| cube | f64 | 8 | 256 | 448.0 | 844.5 | 1.89 | 6764.6, 1207.7, 844.5, 853.1, 448.0 |
| cube | f32 | 3 | 128 | 73.8 | 75.3 | 1.02 | 386.3, 152.2, 75.3, 73.8, 173.0 |
| cube | f32 | 8 | 256 | 242.7 | 498.9 | 2.06 | 3945.7, 853.6, 498.9, 505.9, 242.7 |
| plummer | f64 | 3 | 64 | 280.8 | 280.8 | 1.00 | 391.4, 308.8, 280.8, 306.1, 385.5 |
| plummer | f64 | 8 | 256 | 1367.7 | 1776.5 | 1.30 | 4916.2, 2854.7, 1776.5, 1431.6, 1367.7 |
| plummer | f32 | 3 | 64 | 245.9 | 245.9 | 1.00 | 357.8, 288.4, 245.9, 255.0, 295.2 |
| plummer | f32 | 8 | 256 | 1144.8 | 1352.1 | 1.18 | 3068.9, 1920.5, 1352.1, 1145.7, 1144.8 |

#### The leaf-size rule of T7

Geometric mean of the evaluation time over the 8 configurations (cube and Plummer, p = 3 and 8, f64 and f32), one thread: the measure of the rule. The fastest size replaces 64 only if it is at least 5% faster by this measure and no φ or ∇φ L2 error at it is more than 10% worse than at 64.

| max points per leaf | geometric mean (ms) | relative to 64 |
| ---: | ---: | ---: |
| 16 | 1316.8 | 3.374 |
| 32 | 585.2 | 1.500 |
| 64 | 390.2 | 1.000 |
| 128 | 378.5 | 0.970 |
| 256 | 430.3 | 1.103 |

Fastest: 128 (1.031x faster than 64); largest error growth there: 1.000x. **Chosen: 64**.

## The SIMD P2P kernel in the FMM, and the leaf size (Phase 3S T7)

- Problems: the T12 calibration problems, the uniform cube and the Plummer sphere, N = 100000, sources equal to targets, 8 charge vectors uniform in [-1, 1), an adaptive tree with `max_level` 16, the default M2L strategy (`Auto`), gradients on; p = [3, 8], f64 and f32.
- Error measure: relative L2 error of φ and of ∇φ at 1000 sampled targets against `direct_sum` in f64, the root mean square over the charge vectors (f32 runs: the charges rounded to f32 and their oracle).
- Timings: wall time in ms of one evaluation, the mean over the charge vectors; upward = P2M and M2M, downward = L2L, M2L and P2L, leaves = L2P, M2P and P2P; far = upward + downward, near share = leaves / evaluate.
- Machine: Apple M3 Max; 16 physical, 16 logical, 12 performance; aarch64-macos, release build; rustc 1.99.0 (b940084d7 2026-09-28); one rank, 12 threads. Wall time of all runs 77.8 s.
- Threading: rayon threads 12; MPI Funneled; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- P2P kernels: `auto` = neon on this machine; ISAs available: scalar, neon.

### P2P kernels in the FMM

64 points per leaf as the refinement target. "x": the reference's time over the kernel's, for the leaf stage and the evaluation. Error ratios: the kernel's error over the reference's.

| distribution | precision | p | P2P | ran as | φ L2 | ∇φ L2 | φ / ref | ∇φ / ref | build | upward | downward | leaves | evaluate | near share | leaves x | evaluate x |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | f64 | 3 | reference | reference | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 51.7 | 0.7 | 4.1 | 14.4 | 19.8 | 73% | 1.00 | 1.00 |
| cube | f64 | 3 | auto | neon | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 50.9 | 0.8 | 4.4 | 6.0 | 11.7 | 52% | 2.39 | 1.70 |
| cube | f64 | 3 | scalar | scalar | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 53.8 | 0.7 | 4.3 | 14.8 | 20.3 | 73% | 0.97 | 0.97 |
| cube | f64 | 8 | reference | reference | 1.816e-5 | 1.903e-5 | 1.000000 | 1.000000 | 268.4 | 2.5 | 72.9 | 17.3 | 93.2 | 19% | 1.00 | 1.00 |
| cube | f64 | 8 | auto | neon | 1.816e-5 | 1.903e-5 | 1.000000 | 1.000000 | 282.7 | 2.6 | 72.0 | 8.8 | 84.0 | 10% | 1.96 | 1.11 |
| cube | f64 | 8 | scalar | scalar | 1.816e-5 | 1.903e-5 | 1.000000 | 1.000000 | 283.2 | 2.6 | 72.2 | 17.3 | 92.7 | 19% | 1.00 | 1.01 |
| cube | f32 | 3 | reference | reference | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 52.9 | 0.7 | 3.8 | 13.8 | 18.8 | 74% | 1.00 | 1.00 |
| cube | f32 | 3 | auto | neon | 2.613e-3 | 1.446e-3 | 1.000001 | 1.000001 | 51.2 | 0.8 | 3.7 | 3.7 | 8.6 | 43% | 3.72 | 2.18 |
| cube | f32 | 3 | scalar | scalar | 2.613e-3 | 1.446e-3 | 1.000000 | 1.000000 | 50.0 | 0.8 | 3.7 | 13.9 | 18.7 | 74% | 1.00 | 1.00 |
| cube | f32 | 8 | reference | reference | 1.826e-5 | 1.910e-5 | 1.000000 | 1.000000 | 270.0 | 2.4 | 44.5 | 18.3 | 65.6 | 28% | 1.00 | 1.00 |
| cube | f32 | 8 | auto | neon | 1.827e-5 | 1.910e-5 | 1.000167 | 1.000050 | 285.2 | 2.4 | 44.6 | 7.1 | 54.6 | 13% | 2.59 | 1.20 |
| cube | f32 | 8 | scalar | scalar | 1.826e-5 | 1.910e-5 | 1.000000 | 1.000000 | 288.1 | 2.4 | 44.7 | 18.1 | 65.6 | 28% | 1.01 | 1.00 |
| plummer | f64 | 3 | reference | reference | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 73.4 | 1.4 | 10.0 | 30.1 | 42.2 | 71% | 1.00 | 1.00 |
| plummer | f64 | 3 | auto | neon | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 72.0 | 1.4 | 9.5 | 20.7 | 32.2 | 64% | 1.46 | 1.31 |
| plummer | f64 | 3 | scalar | scalar | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 76.2 | 1.5 | 9.7 | 30.7 | 42.4 | 72% | 0.98 | 0.99 |
| plummer | f64 | 8 | reference | reference | 2.187e-5 | 1.387e-5 | 1.000000 | 1.000000 | 292.8 | 3.7 | 129.1 | 74.1 | 207.5 | 36% | 1.00 | 1.00 |
| plummer | f64 | 8 | auto | neon | 2.187e-5 | 1.387e-5 | 1.000000 | 1.000000 | 295.2 | 3.5 | 127.3 | 65.0 | 196.4 | 33% | 1.14 | 1.06 |
| plummer | f64 | 8 | scalar | scalar | 2.187e-5 | 1.387e-5 | 1.000000 | 1.000000 | 295.8 | 3.5 | 132.0 | 75.7 | 211.7 | 36% | 0.98 | 0.98 |
| plummer | f32 | 3 | reference | reference | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 76.3 | 1.4 | 9.0 | 30.9 | 41.7 | 74% | 1.00 | 1.00 |
| plummer | f32 | 3 | auto | neon | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000001 | 74.6 | 1.4 | 11.6 | 21.7 | 35.3 | 62% | 1.42 | 1.18 |
| plummer | f32 | 3 | scalar | scalar | 3.202e-3 | 1.116e-3 | 1.000000 | 1.000000 | 71.8 | 1.3 | 9.5 | 30.6 | 41.9 | 73% | 1.01 | 1.00 |
| plummer | f32 | 8 | reference | reference | 2.196e-5 | 1.394e-5 | 1.000000 | 1.000000 | 290.0 | 4.7 | 86.1 | 80.4 | 171.7 | 47% | 1.00 | 1.00 |
| plummer | f32 | 8 | auto | neon | 2.196e-5 | 1.395e-5 | 0.999940 | 1.000406 | 299.6 | 4.7 | 86.1 | 64.0 | 155.3 | 41% | 1.26 | 1.11 |
| plummer | f32 | 8 | scalar | scalar | 2.196e-5 | 1.394e-5 | 1.000000 | 1.000000 | 297.2 | 3.3 | 82.3 | 75.9 | 162.0 | 47% | 1.06 | 1.06 |

### Leaf-size study (P2P `auto`)

| distribution | precision | p | max points per leaf | leaves | points per leaf (min / mean / max) | V pairs | φ L2 | ∇φ L2 | upward | downward | far | leaves | evaluate | near share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | f64 | 3 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 2.769e-3 | 2.232e-3 | 1.3 | 34.3 | 35.6 | 7.7 | 44.2 | 17% |
| cube | f64 | 3 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 2.616e-3 | 1.489e-3 | 1.0 | 6.3 | 7.3 | 11.6 | 19.5 | 59% |
| cube | f64 | 3 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 0.8 | 4.2 | 5.0 | 6.0 | 11.5 | 52% |
| cube | f64 | 3 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 0.8 | 4.4 | 5.3 | 6.1 | 11.9 | 52% |
| cube | f64 | 3 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 2.295e-3 | 8.342e-4 | 0.6 | 0.7 | 1.3 | 30.8 | 32.6 | 95% |
| cube | f64 | 8 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 1.893e-5 | 2.804e-5 | 6.7 | 674.5 | 681.2 | 19.4 | 701.6 | 3% |
| cube | f64 | 8 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 1.816e-5 | 1.950e-5 | 3.1 | 87.4 | 90.5 | 32.8 | 123.9 | 26% |
| cube | f64 | 8 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 1.816e-5 | 1.903e-5 | 3.1 | 74.5 | 77.6 | 9.6 | 87.8 | 11% |
| cube | f64 | 8 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 1.816e-5 | 1.903e-5 | 2.7 | 75.2 | 77.9 | 9.7 | 88.3 | 11% |
| cube | f64 | 8 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 1.596e-5 | 1.087e-5 | 2.0 | 7.9 | 9.9 | 35.3 | 45.7 | 77% |
| cube | f32 | 3 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 2.769e-3 | 2.232e-3 | 1.3 | 28.8 | 30.2 | 7.7 | 38.7 | 20% |
| cube | f32 | 3 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 2.616e-3 | 1.489e-3 | 0.9 | 5.6 | 6.5 | 9.2 | 16.2 | 57% |
| cube | f32 | 3 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 0.8 | 3.8 | 4.5 | 3.5 | 8.5 | 42% |
| cube | f32 | 3 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 2.613e-3 | 1.446e-3 | 0.8 | 3.7 | 4.5 | 3.6 | 8.5 | 42% |
| cube | f32 | 3 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 2.295e-3 | 8.343e-4 | 0.6 | 0.8 | 1.3 | 15.1 | 16.8 | 90% |
| cube | f32 | 8 | 16 | 31543 | 0 / 3.2 / 16 | 5578602 | 1.910e-5 | 2.805e-5 | 4.9 | 391.7 | 396.7 | 20.4 | 418.0 | 5% |
| cube | f32 | 8 | 32 | 5657 | 0 / 17.7 / 32 | 656322 | 1.825e-5 | 1.957e-5 | 2.8 | 58.1 | 60.9 | 30.8 | 92.4 | 33% |
| cube | f32 | 8 | 64 | 4096 | 7 / 24.4 / 43 | 640584 | 1.827e-5 | 1.910e-5 | 2.4 | 47.7 | 50.1 | 7.2 | 57.8 | 12% |
| cube | f32 | 8 | 128 | 4096 | 7 / 24.4 / 43 | 640584 | 1.827e-5 | 1.910e-5 | 2.3 | 46.2 | 48.5 | 7.0 | 56.0 | 13% |
| cube | f32 | 8 | 256 | 512 | 159 / 195.3 / 230 | 56448 | 1.607e-5 | 1.181e-5 | 1.8 | 4.8 | 6.6 | 18.6 | 25.6 | 73% |
| plummer | f64 | 3 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 3.306e-3 | 1.545e-3 | 1.9 | 25.9 | 27.8 | 15.6 | 44.2 | 35% |
| plummer | f64 | 3 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 3.260e-3 | 1.377e-3 | 1.7 | 15.4 | 17.1 | 19.3 | 37.0 | 52% |
| plummer | f64 | 3 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 3.202e-3 | 1.116e-3 | 1.4 | 9.6 | 11.0 | 20.2 | 31.8 | 64% |
| plummer | f64 | 3 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 3.160e-3 | 1.008e-3 | 1.1 | 7.3 | 8.5 | 24.3 | 33.3 | 73% |
| plummer | f64 | 3 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 3.072e-3 | 8.782e-4 | 1.4 | 7.0 | 8.4 | 33.7 | 42.7 | 79% |
| plummer | f64 | 8 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 2.269e-5 | 2.047e-5 | 7.1 | 543.8 | 550.8 | 53.5 | 605.3 | 9% |
| plummer | f64 | 8 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 2.236e-5 | 1.864e-5 | 5.0 | 261.8 | 266.8 | 61.5 | 329.0 | 19% |
| plummer | f64 | 8 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 2.187e-5 | 1.387e-5 | 6.5 | 129.4 | 135.9 | 69.9 | 206.4 | 34% |
| plummer | f64 | 8 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 2.142e-5 | 1.290e-5 | 3.3 | 78.9 | 82.2 | 75.3 | 158.0 | 48% |
| plummer | f64 | 8 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 2.092e-5 | 1.072e-5 | 3.2 | 57.4 | 60.5 | 90.1 | 151.2 | 60% |
| plummer | f32 | 3 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 3.306e-3 | 1.545e-3 | 2.0 | 24.4 | 26.4 | 16.0 | 43.1 | 37% |
| plummer | f32 | 3 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 3.260e-3 | 1.377e-3 | 1.5 | 13.6 | 15.1 | 16.8 | 32.4 | 52% |
| plummer | f32 | 3 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 3.202e-3 | 1.116e-3 | 1.3 | 9.1 | 10.4 | 18.3 | 29.1 | 63% |
| plummer | f32 | 3 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 3.160e-3 | 1.008e-3 | 1.3 | 7.5 | 8.9 | 20.1 | 29.4 | 68% |
| plummer | f32 | 3 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 3.072e-3 | 8.782e-4 | 4.6 | 7.6 | 12.3 | 27.9 | 40.6 | 69% |
| plummer | f32 | 8 | 16 | 21652 | 0 / 4.6 / 16 | 3574518 | 2.286e-5 | 2.051e-5 | 5.0 | 282.6 | 287.5 | 50.8 | 339.2 | 15% |
| plummer | f32 | 8 | 32 | 11201 | 0 / 8.9 / 32 | 1750788 | 2.250e-5 | 1.869e-5 | 3.9 | 153.0 | 156.9 | 59.0 | 216.6 | 27% |
| plummer | f32 | 8 | 64 | 5678 | 0 / 17.6 / 64 | 813714 | 2.196e-5 | 1.395e-5 | 3.3 | 81.2 | 84.5 | 63.0 | 148.1 | 43% |
| plummer | f32 | 8 | 128 | 3298 | 0 / 30.3 / 128 | 434322 | 2.149e-5 | 1.298e-5 | 2.9 | 53.3 | 56.2 | 68.3 | 125.0 | 55% |
| plummer | f32 | 8 | 256 | 2157 | 0 / 46.4 / 256 | 256332 | 2.096e-5 | 1.085e-5 | 2.7 | 41.6 | 44.4 | 77.1 | 121.9 | 63% |

#### Fastest leaf size per configuration

| distribution | precision | p | fastest | evaluate (ms) | at 64 (ms) | 64 / fastest | evaluate at 16, 32, 64, 128, 256 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cube | f64 | 3 | 64 | 11.5 | 11.5 | 1.00 | 44.2, 19.5, 11.5, 11.9, 32.6 |
| cube | f64 | 8 | 256 | 45.7 | 87.8 | 1.92 | 701.6, 123.9, 87.8, 88.3, 45.7 |
| cube | f32 | 3 | 128 | 8.5 | 8.5 | 1.00 | 38.7, 16.2, 8.5, 8.5, 16.8 |
| cube | f32 | 8 | 256 | 25.6 | 57.8 | 2.26 | 418.0, 92.4, 57.8, 56.0, 25.6 |
| plummer | f64 | 3 | 64 | 31.8 | 31.8 | 1.00 | 44.2, 37.0, 31.8, 33.3, 42.7 |
| plummer | f64 | 8 | 256 | 151.2 | 206.4 | 1.37 | 605.3, 329.0, 206.4, 158.0, 151.2 |
| plummer | f32 | 3 | 64 | 29.1 | 29.1 | 1.00 | 43.1, 32.4, 29.1, 29.4, 40.6 |
| plummer | f32 | 8 | 256 | 121.9 | 148.1 | 1.21 | 339.2, 216.6, 148.1, 125.0, 121.9 |

#### The leaf-size rule of T7

Geometric mean of the evaluation time over the 8 configurations (cube and Plummer, p = 3 and 8, f64 and f32); the rule reads one-thread runs, so this run only informs it. The fastest size replaces 64 only if it is at least 5% faster by this measure and no φ or ∇φ L2 error at it is more than 10% worse than at 64.

| max points per leaf | geometric mean (ms) | relative to 64 |
| ---: | ---: | ---: |
| 16 | 145.1 | 3.296 |
| 32 | 64.7 | 1.471 |
| 64 | 44.0 | 1.000 |
| 128 | 42.0 | 0.954 |
| 256 | 46.1 | 1.048 |

Fastest: 128 (1.048x faster than 64); largest error growth there: 1.000x. **Chosen: 64** (informative; not the measure of the rule).

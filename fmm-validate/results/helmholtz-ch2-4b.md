# Helmholtz CH2.4b: the Yukawa P2P on a shared skeleton

Helmholtz task CH2.4b ([docs/helmholtz/tasks/CH2.4b.md](../../docs/helmholtz/tasks/CH2.4b.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH2.4, §2 (constraint 1), §5.7, §6.2; design
[docs/design/simd-p2p.md](../../docs/design/simd-p2p.md) §4.10. Measured on 2026-10-31 from
branch `tbetcke/h-ch2-4b` against `main` at 32a2c94, release builds, rustc 1.99.0:

- the M3 Max (NEON, scalar), outside the sandbox;
- locust (Neoverse-V2, NEON, scalar), idle (load below 0.4, the H100 at 0 %);
- Kathleen (Intel Xeon Gold 6248: scalar, AVX2, AVX-512): job 616123 (node-c11a-039,
  `check.sbatch simd`) and job 616122 (node-c11a-034, the runs of §2–§5).

## Summary

- **`P2pKernel::evaluate_yukawa`** adds q e^{−κr}/r and its gradient
  −q (κr + 1) e^{−κr} d/r³ (CONVENTIONS §3.15, k = iκ), with the semantics of
  `nd_fmm_ref::helmholtz::p2p::p2p_yukawa`. It runs on every ISA in both precisions. The
  scalar ISA is the reference's loop, bit for bit.
- **One loop skeleton for Laplace and Yukawa.** `arch::p2p::p2p_skeleton` holds the blocks,
  tails, loads, broadcast, d, r² and stores. The arithmetic after r² comes from a pair
  kernel, a compile-time type parameter. Laplace's pair kernel is zero-sized and is the old
  source loop operation for operation; `p2p_body` is the skeleton with it. Yukawa's forms
  r = r² ρ, e^{−κr} by CH2.4a's exponential, and the terms. At κ = 0 every step equals
  Laplace's, so the vector kernels then give `evaluate`'s bits (a test).
- **Constraint 1 holds in all four conditions** (§2–§4):
  1. no run-time dispatch below the call: the pair kernel is a type parameter, K and the
     gradient flag stay const generics;
  2. no layout or table change;
  3. Laplace's bits unchanged: every existing function of `nd-fmm-simd` is the same
     instruction for instruction as `main`'s on four targets (macOS and Linux, aarch64 and
     x86_64) at the default settings and with one codegen unit, and the 18 full FMM outputs
     are byte for byte `main`'s on each of the three machines;
  4. Laplace's benchmarks within noise (D9) on the three machines: the P2P kernel, the
     plane-wave M2L steps and the FMM. C3S.4: 95–101 % of the Phase 3S prototype on the
     M3 Max.
- **The accuracy contract** (design §4.10, derived to first order): each potential term
  within (20 + 12 κr) u_T of the reference's relative to |q| e^{−κr}/r, each gradient
  component within (52 + 12 κr) u_T relative to |q| (1 + κr) e^{−κr}/r², for κr ≤ 64 (f32)
  or 512 (f64). The κr part is the conditioning of e^{−κr} in r, which the reference
  shares. Measured over 10⁶ pairs per precision on NEON, AVX2 and AVX-512: at most 10.2 u
  at small κr and about 3 u per unit of κr beyond (§5.1).
- **Cost per pair against Laplace's** (§5.2): on NEON 2.7–3.0× in f32 and 3.4–4.5× in f64
  (M3 Max), 3.3–3.6× and 3.1–3.2× (locust); on Kathleen 2.3–3.6× (AVX-512) and 2.6–3.5× (AVX2).
  Ours is faster than green-kernels' modified Helmholtz everywhere measured: 2.7–3.2× in
  f32 and 1.7–1.95× in f64 on the M3 Max, 2.3–3.2× and 1.2–1.3× on locust, and on Kathleen
  3.1–4.1× and 1.8–2.3× with both on AVX2.

## 1. What changed

- `fmm-simd/src/arch/p2p.rs`: `PairKernel`, `Laplace`, `Yukawa`, `p2p_skeleton`; `p2p_body`
  is the skeleton with `Laplace`. The several-vector and dipole bodies are untouched (each
  could become a later user; rule of two, proposal §6.2).
- `arch/{neon,avx2,avx512}.rs`: the Yukawa entry points (`p2p_yukawa_f32`, `…_gradient`,
  `…_f64`, `…_f64_gradient`) with `YUKAWA_K = 1`; `arch/mod.rs` their dispatch;
  `arch/scalar.rs` `p2p_yukawa`, the reference's loop.
- `kernel.rs` `P2pKernel::evaluate_yukawa` (panics unless κ is finite and κ ≥ 0, besides the
  reference's length checks); `lib.rs` the sealed method and the crate docs' "Yukawa".
- Tests: `tests/p2p/yukawa.rs` (17 tests and 4 ignored release tests), and
  `src/arch/tests.rs` `p2p_yukawa_body_is_independent_of_k`.
- `spikes/p2p-simd/examples/yukawa.rs`: the cost per pair (§5.2).
- No dependency, convention, table, layout or `nd-fmm-plan` change; the one manifest
  change is the spike's `[[example]]` for `yukawa`. `nd-fmm-exec`
  does not call the Yukawa kernel yet (CH2.1 will).

## 2. Laplace's code (constraint 1, conditions 1–3)

Every function under `nd_fmm_simd::arch` in the release rlib, `main` against the branch,
disassembled, with addresses, relocations, symbol targets and objdump's comments
normalised, compared one by one:

| target | codegen units | identical | differ | only on the branch (new) |
| --- | --- | ---: | ---: | ---: |
| aarch64-apple-darwin | default | 68 | 0 | 8 |
| aarch64-apple-darwin | 1 | 69 | 0 | 6 |
| x86_64-apple-darwin | default | 130 | 0 | 12 |
| x86_64-apple-darwin | 1 | 115 | 0 | 10 |
| aarch64-unknown-linux-gnu | default | 73 | 0 | 8 |
| aarch64-unknown-linux-gnu | 1 | 70 | 0 | 6 |
| x86_64-unknown-linux-gnu | default | 130 | 0 | 12 |
| x86_64-unknown-linux-gnu | 1 | 115 | 0 | 10 |

The Linux rlibs were cross-built on the M3 Max with the remote machines' toolchain (rustc
1.99.0; their flags are link arguments only). Kathleen built the x86_64 Linux pair natively
in job 616122 with the same result (245 functions identical). The skeleton alone, before
the Yukawa entry points existed, was identical too. The new symbols are the Yukawa entry
points and dispatch functions, and on aarch64 at the default settings
`scalar::p2p_dipoles::<f64>`, which on `main` is instantiated outside the crate's own
objects; in the linked test binaries its code is identical.

**Bits.** `bits.sh` (as in CH2.0): the cube at N = 10⁶, then `nd-fmm-bench --write-output`
(f64, every point, φ and ∇φ) from each tree's release binary, for Dense, Classes, Rotation,
Batched, PlaneWave and Auto at p = 3, 6 and 8:

| machine | ISA of P2P | cases | identical |
| --- | --- | ---: | ---: |
| M3 Max | NEON | 18 | 18 |
| locust | NEON | 18 | 18 |
| Kathleen (job 616122, node-c11a-034) | AVX-512 | 18 | 18 |

The test suite passes unchanged: no existing test was edited.

## 3. Laplace's benchmarks (D9)

D9 as signed off: five runs per side, A B B A A B B A A B (A `main`, B the branch), each
tree's own release binaries. Per row the spread is max − min of `main`'s five values; a row
fails only if the branch's median is slower than `main`'s by more than that spread. Each
run:

1. `p2p_kernels --isa <ISA>` (one thread): the SIMD kernel's gathered W1 rows at n_t = 16,
   32 and 64, f32 and f64, φ and φ with ∇φ, in ns per pair (1 / Gpairs/s): 12 rows per ISA;
2. `plane_wave_parts --ps 8,12 --repeats 15` (one thread), f32 and f64: `waves`, `merge`,
   `gather`, `sums` and their sum per box (µs): 20 rows;
3. `tools/bench/run.sh --backend host --strategy auto`, f32 and f64, p = 3 (Batched) and 8
   (PlaneWave), N = 10⁶, 10 evaluations after 2 warm-ups, every core (ms): 4 rows.

"outputs equal": the error columns agree in every run of both sides.

### 3.1 M3 Max

15:33–15:55. The load was 18 at run 1 (the preceding byte comparison's FMMs), then 4–7.

**The P2P kernel** (`p2p_kernels`, ns per pair):

| ISA | precision | output | n_t | main ns/pair | branch ns/pair | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neon | f32 | φ | 16 | 0.2531, 0.2480, 0.2478, 0.2483, 0.2479 | 0.2521, 0.2476, 0.2481, 0.2483, 0.2482 | 0.0053 | +0.0001847 (+0.1 %) | yes | yes |
| neon | f32 | φ | 32 | 0.2618, 0.2458, 0.2458, 0.2458, 0.2459 | 0.2462, 0.2458, 0.2458, 0.2458, 0.2458 | 0.0160 | -6.041e-05 (-0.0 %) | yes | yes |
| neon | f32 | φ | 64 | 0.2579, 0.2452, 0.2452, 0.2457, 0.2451 | 0.2455, 0.2450, 0.2501, 0.2454, 0.2451 | 0.0128 | +0.0001805 (+0.1 %) | yes | yes |
| neon | f32 | φ, ∇φ | 16 | 0.3874, 0.3801, 0.3795, 0.3799, 0.3795 | 0.3805, 0.3794, 0.3798, 0.3801, 0.3799 | 0.0079 | +0 (+0.0 %) | yes | yes |
| neon | f32 | φ, ∇φ | 32 | 0.3982, 0.3771, 0.3772, 0.3772, 0.3769 | 0.3775, 0.3769, 0.3772, 0.3771, 0.3771 | 0.0213 | -0.0001422 (-0.0 %) | yes | yes |
| neon | f32 | φ, ∇φ | 64 | 0.3976, 0.3764, 0.3759, 0.3762, 0.3761 | 0.3833, 0.3772, 0.3764, 0.3761, 0.3762 | 0.0217 | +0.0001416 (+0.0 %) | yes | yes |
| neon | f64 | φ | 16 | 0.4796, 0.4570, 0.4701, 0.4572, 0.4566 | 0.4583, 0.4572, 0.4575, 0.4579, 0.4575 | 0.0230 | +0.0002092 (+0.0 %) | yes | yes |
| neon | f64 | φ | 32 | 0.4812, 0.4552, 0.4543, 0.4550, 0.4539 | 0.4562, 0.4545, 0.4550, 0.4545, 0.4552 | 0.0273 | +0 (+0.0 %) | yes | yes |
| neon | f64 | φ | 64 | 0.4801, 0.4543, 0.4537, 0.4541, 0.4531 | 0.4762, 0.4545, 0.4535, 0.4533, 0.4543 | 0.0270 | +0.0002063 (+0.0 %) | yes | yes |
| neon | f64 | φ, ∇φ | 16 | 0.8532, 0.8104, 0.8123, 0.8091, 0.8084 | 0.8183, 0.8097, 0.8123, 0.8091, 0.8097 | 0.0448 | -0.0006562 (-0.1 %) | yes | yes |
| neon | f64 | φ, ∇φ | 32 | 1.0965, 0.8065, 0.8058, 0.8026, 0.8052 | 0.8097, 0.8065, 0.8091, 0.8058, 0.8084 | 0.2939 | +0.002606 (+0.3 %) | yes | yes |
| neon | f64 | φ, ∇φ | 64 | 0.9183, 0.8045, 0.8078, 0.8065, 0.8052 | 0.8039, 0.8052, 0.8091, 0.8045, 0.8039 | 0.1138 | -0.001946 (-0.2 %) | yes | yes |

All rows within noise.

**The plane-wave M2L steps** (`plane_wave_parts`, µs per box):

| precision | p | step | main µs | branch µs | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 8 | gather | 0.749, 0.746, 0.745, 0.745, 0.749 | 0.750, 0.746, 0.746, 0.746, 0.754 | 0.004 | +0 (+0.0 %) | yes | yes |
| f32 | 8 | merge | 1.843, 1.836, 1.844, 1.829, 1.838 | 1.845, 1.818, 1.838, 1.833, 1.853 | 0.015 | +0 (+0.0 %) | yes | yes |
| f32 | 8 | sum | 10.014, 9.942, 9.989, 9.970, 9.970 | 9.988, 9.950, 10.046, 9.971, 10.024 | 0.072 | +0.018 (+0.2 %) | yes | yes |
| f32 | 8 | sums | 1.955, 1.937, 1.948, 1.931, 1.924 | 1.946, 1.941, 1.928, 1.929, 1.937 | 0.031 | +0 (+0.0 %) | yes | yes |
| f32 | 8 | waves | 1.605, 1.604, 1.607, 1.612, 1.610 | 1.607, 1.608, 1.600, 1.611, 1.629 | 0.008 | +0.001 (+0.1 %) | yes | yes |
| f32 | 12 | gather | 1.226, 1.221, 1.227, 1.228, 1.222 | 1.225, 1.226, 1.225, 1.222, 1.227 | 0.007 | -0.001 (-0.1 %) | yes | yes |
| f32 | 12 | merge | 3.564, 3.545, 3.535, 3.513, 3.514 | 3.520, 3.521, 3.536, 3.532, 3.529 | 0.051 | -0.006 (-0.2 %) | yes | yes |
| f32 | 12 | sum | 19.320, 19.213, 19.207, 19.182, 19.232 | 19.201, 19.208, 19.292, 19.184, 19.443 | 0.138 | -0.005 (-0.0 %) | yes | yes |
| f32 | 12 | sums | 3.101, 3.097, 3.094, 3.091, 3.140 | 3.097, 3.099, 3.082, 3.082, 3.092 | 0.049 | -0.005 (-0.2 %) | yes | yes |
| f32 | 12 | waves | 2.787, 2.781, 2.787, 2.775, 2.776 | 2.777, 2.781, 2.774, 2.777, 2.783 | 0.012 | -0.004 (-0.1 %) | yes | yes |
| f64 | 8 | gather | 1.463, 1.461, 1.462, 1.463, 1.465 | 1.462, 1.463, 1.472, 1.462, 1.499 | 0.004 | +0 (+0.0 %) | yes | yes |
| f64 | 8 | merge | 4.073, 4.049, 4.053, 4.058, 4.042 | 4.066, 4.048, 4.055, 4.063, 4.522 | 0.031 | +0.01 (+0.2 %) | yes | yes |
| f64 | 8 | sum | 13.976, 13.959, 14.001, 13.959, 13.940 | 13.948, 13.923, 13.926, 13.969, 15.133 | 0.061 | -0.011 (-0.1 %) | yes | yes |
| f64 | 8 | sums | 2.000, 1.979, 1.984, 1.987, 2.020 | 1.979, 1.986, 1.990, 1.978, 2.393 | 0.041 | -0.001 (-0.1 %) | yes | yes |
| f64 | 8 | waves | 2.115, 2.111, 2.184, 2.109, 2.111 | 2.109, 2.119, 2.110, 2.110, 2.212 | 0.075 | -0.001 (-0.0 %) | yes | yes |
| f64 | 12 | gather | 2.325, 2.322, 2.323, 2.327, 2.326 | 2.323, 2.327, 2.322, 2.321, 2.323 | 0.005 | -0.002 (-0.1 %) | yes | yes |
| f64 | 12 | merge | 8.138, 7.975, 7.937, 7.944, 7.917 | 7.969, 7.966, 8.004, 7.969, 7.976 | 0.221 | +0.025 (+0.3 %) | yes | yes |
| f64 | 12 | sum | 29.045, 28.731, 28.697, 28.706, 28.657 | 28.708, 28.711, 28.746, 28.673, 28.733 | 0.388 | +0.005 (+0.0 %) | yes | yes |
| f64 | 12 | sums | 4.489, 4.444, 4.449, 4.452, 4.446 | 4.445, 4.439, 4.424, 4.422, 4.477 | 0.045 | -0.01 (-0.2 %) | yes | yes |
| f64 | 12 | waves | 3.931, 3.878, 3.878, 3.882, 3.860 | 3.877, 3.883, 3.900, 3.871, 3.852 | 0.071 | -0.001 (-0.0 %) | yes | yes |

All rows within noise.

**The FMM** (`run.sh`, host, Auto, ms per evaluation):

| precision | p | strategy | main ms | branch ms | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 3 | Batched | 59.58, 60.73, 58.02, 58.97, 60.12 | 59.71, 58.07, 57.48, 61.86, 58.39 | 2.71 | -1.189 (-2.0 %) | yes | yes |
| f32 | 8 | PlaneWave | 119.54, 121.30, 121.09, 118.61, 118.38 | 118.68, 119.74, 119.80, 119.23, 119.34 | 2.92 | -0.192 (-0.2 %) | yes | yes |
| f64 | 3 | Batched | 99.04, 94.99, 94.19, 95.25, 94.43 | 95.41, 95.58, 95.31, 97.66, 100.97 | 4.85 | +0.59 (+0.6 %) | yes | yes |
| f64 | 8 | PlaneWave | 167.36, 168.65, 167.03, 170.74, 167.49 | 169.28, 170.17, 169.06, 167.49, 167.41 | 3.71 | +1.571 (+0.9 %) | yes | yes |

All rows within noise.

### 3.2 locust

15:36–16:04, D9 restarted after a first attempt stopped in run 1 (an incomplete copy of
`main`'s tree lacked `tools/bench/run.sh`; re-synced). The loads logged at each run's start
(0.3–12 on 72 cores) are the preceding run's all-core FMM step; no other user's process
was running.

**The P2P kernel** (`p2p_kernels`, ns per pair):

| ISA | precision | output | n_t | main ns/pair | branch ns/pair | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neon | f32 | φ | 16 | 0.5249, 0.5263, 0.5249, 0.5263, 0.5252 | 0.5230, 0.5263, 0.5249, 0.5252, 0.5260 | 0.0014 | +0 (+0.0 %) | yes | yes |
| neon | f32 | φ | 32 | 0.5233, 0.5247, 0.5230, 0.5247, 0.5230 | 0.5214, 0.5241, 0.5230, 0.5230, 0.5244 | 0.0016 | -0.0002737 (-0.1 %) | yes | yes |
| neon | f32 | φ | 64 | 0.5227, 0.5244, 0.5227, 0.5241, 0.5227 | 0.5208, 0.5238, 0.5225, 0.5227, 0.5241 | 0.0016 | +0 (+0.0 %) | yes | yes |
| neon | f32 | φ, ∇φ | 16 | 0.7252, 0.7267, 0.7246, 0.7267, 0.7252 | 0.7225, 0.7267, 0.7252, 0.7252, 0.7267 | 0.0021 | +0 (+0.0 %) | yes | yes |
| neon | f32 | φ, ∇φ | 32 | 0.7220, 0.7236, 0.7215, 0.7236, 0.7215 | 0.7194, 0.7236, 0.7215, 0.7215, 0.7231 | 0.0021 | -0.0005209 (-0.1 %) | yes | yes |
| neon | f32 | φ, ∇φ | 64 | 0.7210, 0.7231, 0.7205, 0.7225, 0.7205 | 0.7184, 0.7225, 0.7205, 0.7205, 0.7220 | 0.0026 | -0.0005194 (-0.1 %) | yes | yes |
| neon | f64 | φ | 16 | 1.8315, 1.8382, 1.8315, 1.8382, 1.8315 | 1.8248, 1.8349, 1.8315, 1.8315, 1.8382 | 0.0067 | +0 (+0.0 %) | yes | yes |
| neon | f64 | φ | 32 | 1.8315, 1.8382, 1.8315, 1.8349, 1.8315 | 1.8248, 1.8349, 1.8315, 1.8315, 1.8349 | 0.0067 | +0 (+0.0 %) | yes | yes |
| neon | f64 | φ | 64 | 1.8315, 1.8382, 1.8315, 1.8349, 1.8315 | 1.8248, 1.8349, 1.8315, 1.8315, 1.8349 | 0.0067 | +0 (+0.0 %) | yes | yes |
| neon | f64 | φ, ∇φ | 16 | 2.3810, 2.3866, 2.3810, 2.3866, 2.3753 | 2.3697, 2.3866, 2.3810, 2.3810, 2.3866 | 0.0113 | +0 (+0.0 %) | yes | yes |
| neon | f64 | φ, ∇φ | 32 | 2.3753, 2.3810, 2.3753, 2.3810, 2.3753 | 2.3697, 2.3810, 2.3753, 2.3753, 2.3810 | 0.0057 | +0 (+0.0 %) | yes | yes |
| neon | f64 | φ, ∇φ | 64 | 2.3753, 2.3810, 2.3697, 2.3810, 2.3697 | 2.3641, 2.3753, 2.3697, 2.3697, 2.3810 | 0.0113 | -0.005629 (-0.2 %) | yes | yes |

All rows within noise.

**The plane-wave M2L steps** (`plane_wave_parts`, µs per box):

| precision | p | step | main µs | branch µs | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 8 | gather | 0.985, 0.981, 0.992, 1.001, 1.005 | 0.999, 1.003, 1.003, 1.005, 0.993 | 0.024 | +0.011 (+1.1 %) | yes | yes |
| f32 | 8 | merge | 3.198, 3.220, 3.254, 3.290, 3.226 | 3.342, 3.198, 3.243, 3.235, 3.192 | 0.092 | +0.009 (+0.3 %) | yes | yes |
| f32 | 8 | sum | 13.287, 13.362, 13.272, 13.529, 13.470 | 13.446, 13.361, 13.490, 13.658, 13.336 | 0.257 | +0.084 (+0.6 %) | yes | yes |
| f32 | 8 | sums | 2.380, 2.421, 2.379, 2.439, 2.452 | 2.386, 2.361, 2.434, 2.686, 2.417 | 0.073 | -0.004 (-0.2 %) | yes | yes |
| f32 | 8 | waves | 2.002, 2.013, 1.988, 2.039, 2.014 | 2.000, 2.023, 2.035, 2.001, 2.000 | 0.051 | -0.012 (-0.6 %) | yes | yes |
| f32 | 12 | gather | 1.543, 1.551, 1.551, 1.552, 1.552 | 1.552, 1.560, 1.565, 1.549, 1.542 | 0.009 | +0.001 (+0.1 %) | yes | yes |
| f32 | 12 | merge | 5.848, 6.074, 6.072, 6.353, 6.059 | 6.288, 6.143, 6.071, 6.326, 6.088 | 0.505 | +0.071 (+1.2 %) | yes | yes |
| f32 | 12 | sum | 26.035, 26.261, 26.176, 26.800, 26.437 | 26.456, 26.582, 26.812, 26.840, 26.241 | 0.765 | +0.321 (+1.2 %) | yes | yes |
| f32 | 12 | sums | 4.214, 4.218, 4.375, 4.392, 4.303 | 4.161, 4.255, 4.338, 4.571, 4.205 | 0.178 | -0.048 (-1.1 %) | yes | yes |
| f32 | 12 | waves | 3.555, 3.552, 3.487, 3.588, 3.565 | 3.557, 3.594, 3.600, 3.524, 3.529 | 0.101 | +0.002 (+0.1 %) | yes | yes |
| f64 | 8 | gather | 1.814, 1.829, 1.829, 1.832, 1.828 | 1.822, 1.843, 1.838, 1.829, 1.824 | 0.018 | +0 (+0.0 %) | yes | yes |
| f64 | 8 | merge | 6.916, 8.027, 7.547, 7.562, 7.375 | 7.029, 6.996, 7.563, 7.148, 7.524 | 1.111 | -0.399 (-5.3 %) | yes | yes |
| f64 | 8 | sum | 19.672, 20.932, 20.516, 20.562, 20.096 | 19.695, 19.806, 20.586, 19.867, 20.457 | 1.260 | -0.649 (-3.2 %) | yes | yes |
| f64 | 8 | sums | 2.899, 2.935, 2.987, 2.937, 2.879 | 2.854, 2.846, 3.031, 2.874, 2.958 | 0.108 | -0.061 (-2.1 %) | yes | yes |
| f64 | 8 | waves | 2.991, 3.099, 3.087, 3.150, 3.034 | 2.998, 3.016, 3.086, 3.041, 3.101 | 0.159 | -0.046 (-1.5 %) | yes | yes |
| f64 | 12 | gather | 3.008, 3.023, 3.030, 3.024, 3.011 | 3.021, 3.041, 3.025, 3.018, 3.021 | 0.022 | -0.002 (-0.1 %) | yes | yes |
| f64 | 12 | merge | 12.675, 14.456, 14.456, 14.152, 13.858 | 12.809, 12.801, 14.374, 13.649, 14.244 | 1.781 | -0.503 (-3.6 %) | yes | yes |
| f64 | 12 | sum | 38.660, 40.749, 40.809, 40.507, 39.826 | 38.500, 39.040, 40.740, 39.571, 40.568 | 2.149 | -0.936 (-2.3 %) | yes | yes |
| f64 | 12 | sums | 5.801, 5.944, 5.942, 5.976, 5.905 | 5.712, 5.824, 5.969, 5.850, 5.944 | 0.175 | -0.092 (-1.5 %) | yes | yes |
| f64 | 12 | waves | 5.707, 5.858, 5.865, 5.888, 5.759 | 5.632, 5.772, 5.850, 5.755, 5.885 | 0.181 | -0.086 (-1.5 %) | yes | yes |

All rows within noise.

**The FMM** (`run.sh`, host, Auto, ms per evaluation):

| precision | p | strategy | main ms | branch ms | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 3 | Batched | 47.50, 53.83, 53.59, 53.87, 53.93 | 47.68, 48.09, 53.65, 54.18, 53.64 | 6.43 | -0.186 (-0.3 %) | yes | yes |
| f32 | 8 | PlaneWave | 66.36, 70.74, 71.54, 70.24, 72.27 | 66.56, 66.38, 69.73, 70.29, 70.85 | 5.92 | -1.006 (-1.4 %) | yes | yes |
| f64 | 3 | Batched | 80.33, 86.92, 87.71, 86.63, 86.22 | 82.52, 79.11, 87.03, 87.57, 86.65 | 7.38 | +0.024 (+0.0 %) | yes | yes |
| f64 | 8 | PlaneWave | 104.11, 110.66, 111.25, 110.26, 109.85 | 107.67, 103.75, 110.88, 109.07, 110.84 | 7.14 | -1.184 (-1.1 %) | yes | yes |

All rows within noise.

### 3.3 Kathleen

Job 616122, node-c11a-034 (exclusive), 15:20–16:12; AVX-512 is the default ISA, AVX2 is
the other one timed; the plane-wave steps on AVX-512.

**The P2P kernel** (`p2p_kernels`, ns per pair):

| ISA | precision | output | n_t | main ns/pair | branch ns/pair | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avx2 | f32 | φ | 16 | 0.3405, 0.3389, 0.3409, 0.3408, 0.3408 | 0.3400, 0.3408, 0.3406, 0.3407, 0.3411 | 0.0021 | -0.0001161 (-0.0 %) | yes | yes |
| avx2 | f32 | φ | 32 | 0.3367, 0.3350, 0.3368, 0.3369, 0.3369 | 0.3369, 0.3367, 0.3368, 0.3367, 0.3369 | 0.0019 | +0 (+0.0 %) | yes | yes |
| avx2 | f32 | φ | 64 | 0.3343, 0.3346, 0.3346, 0.3346, 0.3349 | 0.3346, 0.3344, 0.3346, 0.3346, 0.3346 | 0.0006 | +0 (+0.0 %) | yes | yes |
| avx2 | f32 | φ, ∇φ | 16 | 0.5935, 0.6042, 0.5858, 0.6057, 0.5750 | 0.5858, 0.5845, 0.5747, 0.6039, 0.5750 | 0.0307 | -0.009018 (-1.5 %) | yes | yes |
| avx2 | f32 | φ, ∇φ | 32 | 0.5872, 0.5995, 0.5807, 0.6010, 0.5705 | 0.5807, 0.5794, 0.5705, 0.5995, 0.5705 | 0.0305 | -0.007825 (-1.3 %) | yes | yes |
| avx2 | f32 | φ, ∇φ | 64 | 0.5845, 0.5992, 0.5787, 0.5992, 0.5688 | 0.5784, 0.5784, 0.5688, 0.5992, 0.5688 | 0.0303 | -0.006085 (-1.0 %) | yes | yes |
| avx2 | f64 | φ | 16 | 1.2092, 1.2092, 1.2210, 1.2210, 1.2210 | 1.2225, 1.2210, 1.2210, 1.2092, 1.2210 | 0.0118 | +0 (+0.0 %) | yes | yes |
| avx2 | f64 | φ | 32 | 1.2063, 1.2077, 1.2180, 1.2195, 1.2180 | 1.2195, 1.2180, 1.2195, 1.1933, 1.2195 | 0.0132 | +0.001485 (+0.1 %) | yes | yes |
| avx2 | f64 | φ | 64 | 1.2048, 1.2048, 1.2165, 1.2165, 1.2151 | 1.2165, 1.2165, 1.2165, 1.1905, 1.2165 | 0.0117 | +0.001478 (+0.1 %) | yes | yes |
| avx2 | f64 | φ, ∇φ | 16 | 1.7953, 1.7953, 1.8018, 1.7986, 1.7986 | 1.7986, 1.7953, 1.7986, 1.7921, 1.7986 | 0.0065 | +0 (+0.0 %) | yes | yes |
| avx2 | f64 | φ, ∇φ | 32 | 1.7889, 1.7921, 1.7953, 1.7953, 1.7953 | 1.7953, 1.7921, 1.7953, 1.7857, 1.7953 | 0.0064 | +0 (+0.0 %) | yes | yes |
| avx2 | f64 | φ, ∇φ | 64 | 1.7825, 1.7825, 1.7889, 1.7889, 1.7889 | 1.7889, 1.7889, 1.7889, 1.7825, 1.7889 | 0.0064 | +0 (+0.0 %) | yes | yes |
| avx512 | f32 | φ | 16 | 0.1610, 0.1610, 0.1609, 0.1610, 0.1610 | 0.1609, 0.1615, 0.1596, 0.1609, 0.1617 | 0.0001 | -5.181e-05 (-0.0 %) | yes | yes |
| avx512 | f32 | φ | 32 | 0.1605, 0.1604, 0.1603, 0.1603, 0.1602 | 0.1600, 0.1603, 0.1601, 0.1603, 0.1604 | 0.0003 | -2.57e-05 (-0.0 %) | yes | yes |
| avx512 | f32 | φ | 64 | 0.1593, 0.1577, 0.1591, 0.1591, 0.1591 | 0.1588, 0.1593, 0.1591, 0.1592, 0.1592 | 0.0016 | +0.0001013 (+0.1 %) | yes | yes |
| avx512 | f32 | φ, ∇φ | 16 | 0.2627, 0.2573, 0.2627, 0.2627, 0.2627 | 0.2627, 0.2627, 0.2627, 0.2627, 0.2631 | 0.0055 | +0 (+0.0 %) | yes | yes |
| avx512 | f32 | φ, ∇φ | 32 | 0.2531, 0.2525, 0.2531, 0.2530, 0.2522 | 0.2529, 0.2530, 0.2530, 0.2530, 0.2531 | 0.0009 | +6.401e-05 (+0.0 %) | yes | yes |
| avx512 | f32 | φ, ∇φ | 64 | 0.2520, 0.2504, 0.2509, 0.2517, 0.2518 | 0.2517, 0.2520, 0.2518, 0.2520, 0.2520 | 0.0016 | +0.0002537 (+0.1 %) | yes | yes |
| avx512 | f64 | φ | 16 | 0.4095, 0.4065, 0.4097, 0.4062, 0.4087 | 0.4083, 0.4073, 0.4090, 0.4093, 0.4093 | 0.0035 | +0.0003343 (+0.1 %) | yes | yes |
| avx512 | f64 | φ | 32 | 0.4039, 0.4034, 0.4045, 0.4031, 0.4055 | 0.4052, 0.4040, 0.4054, 0.4060, 0.4062 | 0.0025 | +0.001473 (+0.4 %) | yes | yes |
| avx512 | f64 | φ | 64 | 0.4013, 0.4011, 0.4024, 0.4010, 0.4032 | 0.4032, 0.4016, 0.4032, 0.4036, 0.4036 | 0.0023 | +0.001942 (+0.5 %) | yes | yes |
| avx512 | f64 | φ, ∇φ | 16 | 0.6039, 0.6035, 0.6039, 0.6031, 0.6042 | 0.6042, 0.6039, 0.6046, 0.6053, 0.6050 | 0.0011 | +0.0007302 (+0.1 %) | yes | yes |
| avx512 | f64 | φ, ∇φ | 32 | 0.5995, 0.5988, 0.6002, 0.5988, 0.5999 | 0.5995, 0.5995, 0.5984, 0.6006, 0.6006 | 0.0014 | +0 (+0.0 %) | yes | yes |
| avx512 | f64 | φ, ∇φ | 64 | 0.5959, 0.5945, 0.5956, 0.5956, 0.5970 | 0.5952, 0.5963, 0.5967, 0.5970, 0.5970 | 0.0025 | +0.001066 (+0.2 %) | yes | yes |

All rows within noise.

**The plane-wave M2L steps** (`plane_wave_parts`, µs per box):

| precision | p | step | main µs | branch µs | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 8 | gather | 1.061, 1.049, 1.056, 1.060, 1.045 | 1.041, 1.051, 1.046, 1.055, 1.067 | 0.016 | -0.005 (-0.5 %) | yes | yes |
| f32 | 8 | merge | 4.058, 3.862, 3.884, 4.290, 3.845 | 3.862, 4.020, 3.855, 4.033, 4.061 | 0.445 | +0.136 (+3.5 %) | yes | yes |
| f32 | 8 | sum | 19.254, 18.926, 19.307, 19.538, 18.978 | 19.019, 19.179, 18.990, 19.143, 19.241 | 0.612 | -0.111 (-0.6 %) | yes | yes |
| f32 | 8 | sums | 3.286, 3.176, 3.368, 3.271, 3.237 | 3.249, 3.261, 3.253, 3.181, 3.258 | 0.192 | -0.018 (-0.6 %) | yes | yes |
| f32 | 8 | waves | 2.180, 2.178, 2.182, 2.238, 2.177 | 2.173, 2.186, 2.166, 2.187, 2.183 | 0.061 | +0.003 (+0.1 %) | yes | yes |
| f32 | 12 | gather | 1.651, 1.578, 1.579, 1.589, 1.571 | 1.574, 1.786, 1.569, 1.645, 1.632 | 0.080 | +0.053 (+3.4 %) | yes | yes |
| f32 | 12 | merge | 6.890, 7.667, 6.489, 7.006, 7.114 | 7.097, 6.843, 6.221, 6.996, 7.095 | 1.178 | -0.01 (-0.1 %) | yes | yes |
| f32 | 12 | sum | 36.790, 37.544, 36.272, 36.745, 36.807 | 36.916, 36.785, 35.896, 36.763, 36.904 | 1.272 | -0.005 (-0.0 %) | yes | yes |
| f32 | 12 | sums | 4.821, 4.845, 4.946, 4.829, 4.797 | 4.825, 4.809, 4.784, 4.703, 4.827 | 0.149 | -0.02 (-0.4 %) | yes | yes |
| f32 | 12 | waves | 3.871, 3.953, 3.849, 3.895, 3.902 | 3.932, 3.910, 3.822, 3.884, 3.901 | 0.104 | +0.006 (+0.2 %) | yes | yes |
| f64 | 8 | gather | 1.622, 1.676, 1.667, 1.675, 1.663 | 1.679, 1.677, 1.675, 1.618, 1.676 | 0.054 | +0.009 (+0.5 %) | yes | yes |
| f64 | 8 | merge | 9.347, 9.319, 10.428, 9.590, 10.121 | 9.356, 9.741, 9.550, 9.335, 9.740 | 1.109 | -0.04 (-0.4 %) | yes | yes |
| f64 | 8 | sum | 27.563, 27.598, 28.662, 27.856, 28.480 | 27.601, 28.077, 27.715, 27.601, 28.102 | 1.099 | -0.141 (-0.5 %) | yes | yes |
| f64 | 8 | sums | 4.156, 4.095, 4.147, 4.099, 4.195 | 4.161, 4.161, 4.122, 4.169, 4.307 | 0.100 | +0.014 (+0.3 %) | yes | yes |
| f64 | 8 | waves | 3.025, 3.115, 3.117, 3.155, 3.196 | 3.093, 3.168, 3.018, 3.151, 3.047 | 0.171 | -0.024 (-0.8 %) | yes | yes |
| f64 | 12 | gather | 2.582, 2.659, 2.662, 2.644, 2.658 | 2.812, 2.668, 2.792, 2.582, 2.650 | 0.080 | +0.01 (+0.4 %) | yes | yes |
| f64 | 12 | merge | 22.333, 22.633, 22.698, 22.234, 22.505 | 22.653, 22.449, 22.246, 22.219, 22.733 | 0.464 | -0.056 (-0.2 %) | yes | yes |
| f64 | 12 | sum | 60.144, 60.669, 60.686, 60.120, 60.481 | 60.739, 60.473, 60.166, 60.159, 60.973 | 0.566 | -0.008 (-0.0 %) | yes | yes |
| f64 | 12 | sums | 7.529, 7.505, 7.507, 7.486, 7.524 | 7.503, 7.507, 7.506, 7.504, 7.877 | 0.043 | -0.001 (-0.0 %) | yes | yes |
| f64 | 12 | waves | 5.786, 5.923, 5.861, 5.902, 5.943 | 5.899, 5.944, 5.782, 5.943, 5.854 | 0.157 | -0.003 (-0.1 %) | yes | yes |

All rows within noise.

**The FMM** (`run.sh`, host, Auto, ms per evaluation):

| precision | p | strategy | main ms | branch ms | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 3 | Batched | 75.77, 77.62, 73.93, 74.98, 77.89 | 76.09, 75.09, 75.48, 77.98, 79.84 | 3.96 | +0.324 (+0.4 %) | yes | yes |
| f32 | 8 | PlaneWave | 150.10, 153.14, 153.26, 150.87, 151.98 | 153.35, 157.00, 155.50, 150.51, 154.36 | 3.16 | +2.386 (+1.6 %) | yes | yes |
| f64 | 3 | Batched | 110.24, 115.87, 110.64, 112.48, 114.53 | 108.09, 107.59, 112.71, 120.58, 115.25 | 5.63 | +0.23 (+0.2 %) | yes | yes |
| f64 | 8 | PlaneWave | 209.34, 199.39, 201.30, 199.86, 207.99 | 202.27, 201.98, 208.51, 205.48, 208.27 | 9.94 | +4.18 (+2.1 %) | yes | yes |

All rows within noise.

**D9 holds on all three machines: 36 + 36 + 48 rows, none outside noise, no repeat
needed, the outputs equal in every run.**

## 4. C3S.4: the Phase 3S prototype

The gate: on NEON at least 90 % of the spike prototype's throughput on the gathered
FMM-shaped workload. The branch's NEON rows (median of its five D9 runs on the M3 Max)
against the prototype's targets-in-lanes kernel at its chosen K (f32 K = 2, f64 K = 4),
as recorded in Phase 3S T2 (`spikes/p2p-simd/results-m3max.md`) and as run today
(`cargo run --release -p nd-fmm-spike-p2p-simd -- --quick`, n_t = 24 only):

| precision | output | n_t | branch Gpairs/s | T2 prototype | of T2 | prototype today (n_t = 24) | of today |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | φ | 16 | 4.029 | 4.012 | 100 % | – | – |
| f32 | φ | 24 | 4.056 | 4.039 | 100 % | 4.040 | 100 % |
| f32 | φ | 32 | 4.069 | 4.056 | 100 % | – | – |
| f32 | φ | 64 | 4.075 | 4.074 | 100 % | – | – |
| f32 | φ, ∇φ | 16 | 2.632 | 2.612 | 101 % | – | – |
| f32 | φ, ∇φ | 24 | 2.646 | 2.631 | 101 % | 2.628 | 101 % |
| f32 | φ, ∇φ | 32 | 2.652 | 2.641 | 100 % | – | – |
| f32 | φ, ∇φ | 64 | 2.657 | 2.652 | 100 % | – | – |
| f64 | φ | 16 | 2.186 | 2.184 | 100 % | – | – |
| f64 | φ | 24 | 2.199 | 2.192 | 100 % | 2.190 | 100 % |
| f64 | φ | 32 | 2.198 | 2.196 | 100 % | – | – |
| f64 | φ | 64 | 2.201 | 2.200 | 100 % | – | – |
| f64 | φ, ∇φ | 16 | 1.235 | 1.296 | 95 % | – | – |
| f64 | φ, ∇φ | 24 | 1.239 | 1.301 | 95 % | 1.301 | 95 % |
| f64 | φ, ∇φ | 32 | 1.237 | 1.304 | 95 % | – | – |
| f64 | φ, ∇φ | 64 | 1.243 | 1.308 | 95 % | – | – |

The f64 rows with gradients are at 95 % on `main` too (§3.1: the branch's medians equal
`main`'s). **C3S.4 holds.**

## 5. The Yukawa kernel

### 5.1 Accuracy

The contract and its derivation are design §4.10. Largest errors over 10⁶ seeded pairs per
precision (`yukawa_terms_*_large`; r² log-uniform over the domain, κr log-uniform in
[10⁻⁶, 64 or 512] and κ = 0 for every sixteenth pair, 2⁻²⁰ ≤ |q| ≤ 1), in u_T, potential /
gradient:

| ISA (machine) | precision | κr < 0.01 | 0.01–1 | 1–10 | 10–100 | 100–512 |
| --- | --- | --- | --- | --- | --- | --- |
| NEON (M3 Max) | f32 | 5.4 / 9.2 | 7.3 / 9.2 | 35.4 / 31.9 | 194 / 168 | – |
| AVX2 (Kathleen) | f32 | 4.7 / 8.9 | 7.3 / 9.0 | 35.4 / 30.7 | 194 / 168 | – |
| AVX-512 (Kathleen) | f32 | 4.2 / 8.8 | 7.3 / 9.3 | 35.4 / 31.9 | 194 / 165 | – |
| NEON (M3 Max) | f64 | 4.9 / 10.2 | 5.9 / 9.9 | 34.9 / 27.4 | 385 / 258 | 1537 / 1520 |
| AVX2 (Kathleen) | f64 | 4.0 / 8.5 | 5.9 / 9.1 | 34.9 / 31.0 | 261 / 258 | 1537 / 1520 |
| AVX-512 (Kathleen) | f64 | 4.1 / 8.5 | 5.9 / 8.3 | 34.9 / 31.0 | 261 / 258 | 1537 / 1045 |

Every pair is within the contract on every ISA. The growth is about 3u per unit of κr at
large κr, against the derived 11u; the constant part 5–10u against 18u and 50u. The
debug-mode runs (4,096 pairs) are within the same bounds on every ISA.

Sums against `direct_sum_yukawa` (FMM-shaped W1 sets, same level and mixed, and uniform
sets up to 4,096 sources; κ = 0.1, 1 and 10; with initial outputs), relative to the term
magnitudes: at most 7.85e-7 (f32) and 2.9e-15 (f64) on every ISA, within C3S.4's rule
(1e-6 and 1e-14, or twice the reference's error). The scalar ISA equals `p2p_yukawa` bit
for bit on grid-valued inputs. Chunk and target-position invariance hold bit for bit, and
the skeleton gives the same bits at K = 1, 2 and 4, on every ISA.

### 5.2 Cost per pair

`spikes/p2p-simd/examples/yukawa.rs`, one thread, on the W1 cells of `p2p_kernels`
(n_t = 8 to 128, per-pair and gathered), κ = 1 in leaf-scaled units (leaf width 2; the time
does not depend on κ, the exponential has no branch). Geometric means over the six cells
of each form; "Yukawa / Laplace" is each code's time per pair for Yukawa over its own
Laplace kernel's; "ours / gk" our default kernel's pairs per second over green-kernels' on
the same kernel. green-kernels' modified Helmholtz is e^{−ωr}/(4πr), ω = κ, with the
gradient with respect to the target, checked against `direct_sum_yukawa` on one pair
(1.8e-16).

Columns: the default kernel's Yukawa Gpairs/s; Yukawa / Laplace for our default kernel and
for green-kernels; our default kernel over green-kernels' for Yukawa and for Laplace.

**M3 Max** (NEON; green-kernels' pulp backend NEON; outside the sandbox, 15:55):

| precision | output | form | default Yukawa Gpairs/s | default Yukawa / Laplace | gk Yukawa / Laplace | ours / gk (Yukawa) | ours / gk (Laplace) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f32 | φ | per-pair | 1.312 | 2.87 | 6.10 | 3.21 | 1.51 |
| f32 | φ | gathered | 1.350 | 2.97 | 6.92 | 3.14 | 1.35 |
| f32 | φ, ∇φ | per-pair | 0.918 | 2.67 | 5.02 | 2.68 | 1.43 |
| f32 | φ, ∇φ | gathered | 0.955 | 2.74 | 6.08 | 2.69 | 1.21 |
| f64 | φ | per-pair | 0.475 | 4.45 | 4.78 | 1.95 | 1.82 |
| f64 | φ | gathered | 0.482 | 4.47 | 5.02 | 1.91 | 1.70 |
| f64 | φ, ∇φ | per-pair | 0.338 | 3.39 | 4.10 | 1.68 | 1.39 |
| f64 | φ, ∇φ | gathered | 0.346 | 3.56 | 4.34 | 1.66 | 1.36 |

**locust** (Neoverse-V2, NEON; pulp NEON; 15:29, load 4.3 at the start, then idle):

| precision | output | form | default Yukawa Gpairs/s | default Yukawa / Laplace | gk Yukawa / Laplace | ours / gk (Yukawa) | ours / gk (Laplace) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f32 | φ | per-pair | 0.555 | 3.33 | 7.40 | 3.23 | 1.46 |
| f32 | φ | gathered | 0.566 | 3.36 | 8.70 | 3.23 | 1.25 |
| f32 | φ, ∇φ | per-pair | 0.362 | 3.59 | 5.06 | 2.27 | 1.61 |
| f32 | φ, ∇φ | gathered | 0.383 | 3.62 | 6.02 | 2.32 | 1.39 |
| f64 | φ | per-pair | 0.174 | 3.12 | 4.24 | 1.29 | 0.95 |
| f64 | φ | gathered | 0.176 | 3.10 | 4.47 | 1.28 | 0.89 |
| f64 | φ, ∇φ | per-pair | 0.128 | 3.13 | 3.56 | 1.22 | 1.07 |
| f64 | φ, ∇φ | gathered | 0.131 | 3.21 | 3.77 | 1.22 | 1.04 |

**Kathleen** (job 616122, node-c11a-034; default AVX-512; green-kernels' pulp backend is V3,
that is AVX2, so its rows compare our AVX-512 with its AVX2):

| precision | output | form | default Yukawa Gpairs/s | default Yukawa / Laplace | gk Yukawa / Laplace | ours / gk (Yukawa) | ours / gk (Laplace) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f32 | φ | per-pair | 1.344 | 2.80 | 3.93 | 7.53 | 5.36 |
| f32 | φ | gathered | 1.574 | 3.26 | 5.22 | 8.11 | 5.06 |
| f32 | φ, ∇φ | per-pair | 1.050 | 2.26 | 3.22 | 6.33 | 4.44 |
| f32 | φ, ∇φ | gathered | 1.273 | 2.52 | 4.21 | 6.89 | 4.13 |
| f64 | φ | per-pair | 0.608 | 3.39 | 4.70 | 6.04 | 4.36 |
| f64 | φ | gathered | 0.666 | 3.60 | 5.40 | 6.27 | 4.18 |
| f64 | φ, ∇φ | per-pair | 0.455 | 3.05 | 3.73 | 4.80 | 3.92 |
| f64 | φ, ∇φ | gathered | 0.496 | 3.27 | 4.15 | 4.96 | 3.90 |

and our AVX2 kernel on the same node, beside green-kernels' (both AVX2):

| precision, output | form | AVX2 Yukawa Gpairs/s | AVX2 Yukawa / Laplace | AVX2 / gk (Yukawa) | AVX2 / gk (Laplace) |
| --- | --- | ---: | ---: | ---: | ---: |
| f32 φ | per-pair | 0.737 | 3.17 | 4.13 | 3.34 |
| f32 φ | gathered | 0.799 | 3.53 | 4.12 | 2.78 |
| f32 φ, ∇φ | per-pair | 0.513 | 2.58 | 3.10 | 2.48 |
| f32 φ, ∇φ | gathered | 0.575 | 2.80 | 3.11 | 2.07 |
| f64 φ | per-pair | 0.233 | 3.16 | 2.32 | 1.56 |
| f64 φ | gathered | 0.245 | 3.23 | 2.31 | 1.38 |
| f64 φ, ∇φ | per-pair | 0.171 | 2.91 | 1.81 | 1.41 |
| f64 φ, ∇φ | gathered | 0.177 | 3.08 | 1.76 | 1.31 |

**Cost per pair against Laplace's: 2.3–4.5× over every ISA, precision and output measured**
(the proposal's "several times Laplace's", §5.7). The exponential's 14 or 20 operations and
the three or five others (design §4.10) take NEON from a divider-bound loop (Laplace's
FSQRT and FDIV, §4.6) to an FP-pipe-bound one, which costs most in f64 on the M3 Max
(4.5× without gradients). green-kernels' modified Helmholtz costs 3.6–8.8× its own
Laplace kernel.

The accuracy of the Yukawa kernels on these cells (largest error against
`direct_sum_yukawa` relative to the term magnitudes, first four sets): ours within 1.1× of
`p2p_yukawa`'s (for example 2.4e-7 and 4.0e-16 at n_t = 64 gathered, φ); green-kernels'
potential is more accurate (1.4e-7 and 3.5e-16), from its lane-wise partial sums, as Phase
3S found for its Laplace kernel.

**The block size.** K = 1 on every ISA (`YUKAWA_K`). On the M3 Max (sandboxed quick runs,
n_t = 24), K = 1 against K = 2: 0.95 against 0.79 Gpairs/s in f32 and 0.34 against 0.26 in
f64 with gradients, within ±4 % without. The exponential's constants (19 vectors in f64)
take most of the 32 registers. On Kathleen (AVX-512, `yukawa --quick` from builds with
`YUKAWA_K` = 1, then 2, then 1 again, in job 616122): K = 2 against K = 1 within 2 % in f32
(1.413 against 1.400 Gpairs/s gathered φ, 1.153 against 1.132 with gradients) and 5–9 %
slower in f64 (0.613 against 0.665, 0.474 against 0.497); the second K = 1 build repeated
the first within 0.5 %. K = 1 stays on every ISA.

### 5.3 Inner loops

The source loops (the whole blocks', then the padded tail's), f32 φ, f32 φ and ∇φ, f64 φ,
f64 φ and ∇φ: NEON 40/40, 61/50, 52/52 and 78/65 instructions; AVX2 49/49, 78/64, 71/68 and
87/84; AVX-512 37/37, 48/46, 46/46 and 58/56. None contains a call. Laplace's, per target
vector: NEON 15, 22, 15 and 22; AVX2 24, 35, 29 and 42; AVX-512 18, 25, 21 and 28,
unchanged.

## 6. Checks

| command | where | result |
| --- | --- | --- |
| `cargo fmt --all -- --check` | M3 Max | passed |
| `cargo clippy -- -D warnings`, `cargo clippy --examples -- -D warnings` | M3 Max | passed |
| `RUST_MIN_STACK=8388608 cargo test` | M3 Max | passed (1,006 tests) |
| `cargo doc --no-deps` | M3 Max | passed |
| `cargo clippy --workspace --all-targets -- -D warnings` | M3 Max | passed |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | M3 Max | passed (1,086 tests, 98 executables) |
| `cargo clippy -p nd-fmm-simd --all-targets` native and `--target x86_64-apple-darwin` | M3 Max | passed |
| `cargo test -p nd-fmm-simd -- --show-output` (debug, 10.6 s) | M3 Max (scalar, NEON) | passed |
| `cargo test -p nd-fmm-simd --release -- --ignored --show-output` | M3 Max (scalar, NEON) | passed |
| `check.sbatch simd` (Clippy, tests, ignored release tests) | Kathleen job 616123, node-c11a-039 (scalar, AVX2, AVX-512) | passed |
| `cargo test -p nd-fmm-spike-p2p-simd --release --example yukawa`, its Clippy | M3 Max | passed |
| `run-mpi-tests.sh run … 2 4` (`mpi_regressions`, `mpi_exec`, `mpi_threading`, `invariance`, 7 octree examples) | M3 Max | passed |
| the inlining check on the Yukawa entry points (§5.3) | NEON; AVX2 and AVX-512 from the x86_64 build | no call |
| constraint 1: disassembly (§2) | 4 targets × 2 settings | every existing function identical |
| constraint 1: `bits.sh` (§2) | M3 Max, locust, Kathleen job 616122 | 18/18 identical on each |
| D9 (§3) | M3 Max, locust, Kathleen job 616122 | 120 rows, all within noise |
| C3S.4 (§4) | M3 Max | 95–101 % |

Not run: the licence check (no manifest of a default member changed; the spike's
`Cargo.toml` gained an `[[example]]` only); locust's test suite (no ISA there that the M3
Max and CI's arm64 runner do not cover; its timings are §3.2 and §5.2).

## 7. Questions for sign-off

1. **The Yukawa contract** (design §4.10): each potential term within (20 + 12 κr) u_T of
   the reference's relative to |q| e^{−κr}/r, each gradient component within
   (52 + 12 κr) u_T relative to |q| (1 + κr) e^{−κr}/r², for κr ≤ 64 (f32) and 512 (f64).
   The constants are the first-order derivation's, rounded up; the κr term is the
   conditioning of e^{−κr} in r, which the reference shares (3.5u κr of its own). Measured:
   10.2 u at most at small κr and about 3 u per unit of κr, on every ISA. Recommendation:
   accept as derived, as §4.8 did for dipoles. Alternative: contracts at the measured level
   with a margin (for example 16 + 6 κr and 24 + 6 κr), which no derivation backs.
2. **κ ≥ 0 required.** `evaluate_yukawa` panics unless κ is finite and κ ≥ 0, where the
   reference takes any finite κ: the exponential is specified for x ≤ 0 only, and §3.15's
   range is Im k ≥ 0. Recommendation: keep it.

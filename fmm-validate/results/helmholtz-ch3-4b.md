# Helmholtz CH3.4b: the Helmholtz P2P on the skeleton

Task record: `docs/helmholtz/tasks/CH3.4b.md`. Design: `docs/design/simd-p2p.md` §4.12.
Branch `tbetcke/h-ch3-4b`, 2026-11-01.

Machines:
- M6 (Apple M6, NEON; `tools/m6/machine.md`): every check, the contract on NEON, the
  disassembly comparison (cross-built for every target), the cost per pair.
- locust (NVIDIA GH200, Neoverse-V2, NEON): the cost per pair.
- Kathleen (Intel Xeon Gold 6248: scalar, AVX2, AVX-512): jobs 617314 and 617322 (node-c11b-024, the sampling widened after the self-review; the numbers of §4 are
  its and the M6's rerun)
  (`check.sbatch simd`, the contract on x86), job 617315 (node-c11b-015, D9 for the AVX-512
  entry points whose code changed at p = 3, the cost per pair, AVX-512's K = 2) and job
  617321 (node-c11a-090, D9 at p = 8 and the byte comparison of the FMM outputs).

## Summary

- **`P2pKernel::evaluate_helmholtz`** adds q e^{ikr}/r and its gradient
  q (ikr − 1) e^{ikr} d/r³ for Re k ≥ 0, Im k ≥ 0, real or complex charges
  (`HelmholtzCharges`), with the outputs in split planes (the real part of every target,
  then the imaginary part), on every ISA in both precisions. The scalar ISA is the
  reference's loop, bit for bit.
- **The skeleton has planes.** `PairKernel<T, S, QP, OP>`: QP charge planes and OP output
  planes, compile-time; Laplace and Yukawa are QP = OP = 1. Every Laplace function on NEON
  and AVX2 is `main`'s instruction for instruction at the default settings (macOS and
  Linux), and so is every other existing function but one: NEON's Yukawa
  `p2p_yukawa_f32_gradient` in the default macOS build, whose source loop exchanges the
  registers of two gradient accumulators (v3, v4; the same instructions). AVX-512's two Laplace gradient entry points keep their source loops (registers
  renamed) but not the block loop's control; D9 on Kathleen: 16 of 16 rows within noise,
  and the FMM outputs of `main` and the branch byte for byte identical in 18 of 18 cases
  (§2, §3).
- **With real charges** the real plane at k = 0 is `evaluate`'s and at k = iκ
  `evaluate_yukawa`'s, bit for bit on every vector ISA: the skeleton is shared, not copied.
- **Contract** (§4.12): (32 + 12 |k| r) u_T per part of φ and (66 + 12 |k| r) u_T per part of
  each ∇φ component, for Re(k) r ≤ 2¹⁰ (above 22 c_leaf for any c_leaf ≤ 46) and Im(k) r ≤ 64
  (f32) or 512 (f64). Measured over 10⁶ pairs per precision on NEON, AVX2 and AVX-512: at most
  10.3u at |k| r < 1 and 9.0u per unit of |k| r beyond (§4).
- **Cost per pair** (§5): the complex-charge kernel at real k takes 3.8–5.9× Laplace's time
  per pair and 1.4–1.7× Yukawa's on the three machines' default ISAs (AVX2: 3.6–4.9× and
  1.2–1.5×); damping (Im k > 0) costs another 1.2–1.5×. It is 1.7–11.9× faster than green-kernels' Helmholtz on the same
  machine.

## 1. What changed

`nd-fmm-simd`:
- `arch/p2p.rs`: `PairKernel<T, S, QP, OP>`; `p2p_skeleton`, `sub_block`, `block` with
  charges `&[[T; QP]]` and OP output planes of stride n; for one plane the tail and
  `sub_block`'s range are CH2.4b's expressions (`if OP == 1`). `Laplace` and `Yukawa`
  implement the one-plane trait with their arithmetic unchanged. `Helmholtz<V, DAMPED>`,
  the pair kernel of design §4.12.
- `arch/{neon,avx2,avx512}.rs`: 16 entry points per ISA (precision × real/complex charges
  × undamped/damped × potential/gradient, `#[inline(never)]`), `HELMHOLTZ_K = 1`, and one
  selector per precision (`p2p_helmholtz_f32`, `_f64`; `#[target_feature]`, safe calls).
- `arch/mod.rs` `p2p_helmholtz_dispatch!`; `arch/scalar.rs` `p2p_helmholtz` (the reference's
  `Term` loop in `Complex`); `lib.rs` `Sealed::p2p_helmholtz`, the crate docs' "Helmholtz";
  `kernel.rs` `evaluate_helmholtz` and `HelmholtzCharges`.
- Tests: `tests/p2p/helmholtz.rs` (23 tests, 4 ignored release tests); `src/arch/tests.rs`
  `p2p_helmholtz_body_is_independent_of_k`; the existing Yukawa block test's call adapted to
  the new skeleton signature (its checks unchanged).

`spikes/p2p-simd/examples/helmholtz.rs` (registered in the spike's manifest): the cost per
pair, with green-kernels' convention check.

## 2. Laplace's code (constraint 1)

Every function of `nd-fmm-simd`'s release rlib on `main` (dc2fefb) and on the branch,
disassembled for aarch64-apple-darwin, aarch64-unknown-linux-gnu, x86_64-apple-darwin and
x86_64-unknown-linux-gnu, at the default settings and with `-C codegen-units=1` (branch
targets and Mach-O literal comments normalised), compared function by function:

| target | default settings | one codegen unit |
| --- | --- | --- |
| aarch64 macOS | 152 of 153 identical; `neon::p2p_yukawa_f32_gradient` registers v3 and v4 swapped | 94 of 98; NEON `merge_f32`, `merge_f64` (and their dispatchers) shorter |
| aarch64 Linux | 153 of 153 identical | 94 of 98; as macOS |
| x86_64 macOS | 220 of 222; `avx512::p2p_f32_gradient`, `avx512::p2p_f64_gradient` | 158 of 160; the same two |
| x86_64 Linux | 220 of 222; the same two | 158 of 160; the same two |

- **How it got there.** Written generally (planes as slices of stride n, two-dimensional
  padded tails), the skeleton left the source loops as they were but moved stack slots, a
  bounds check and the tail's copies in every Laplace and Yukawa P2P entry point. For one plane
  the padded tail now runs CH2.4b's statements and `sub_block` slices `start..end`, a
  compile-time choice; with them NEON's and AVX2's Laplace code equals `main`'s, and every
  other function too but NEON's `p2p_yukawa_f32_gradient` in the default macOS build: in
  its source loop the gradient accumulators of y and z exchange registers (v3 and v4, in
  the two `fmla`s and the moves around the loop), the instructions otherwise the same; on
  Linux, with the same toolchain, it is `main`'s. Other formulations tried for
  the rest (one-plane addressing in `block`, plane-major accumulators, one-plane asserts)
  moved the differences elsewhere.
- **AVX-512's Laplace gradient entry points** (K = 2): their source loops (50, 25, 25 in f32;
  56, 28, 28 in f64) are `main`'s instruction for instruction up to register numbers; the
  loop over the K-blocks is controlled differently (`main` computes a trip count, the branch
  tests per block). So D9 ran for them (§3).
- **One codegen unit, NEON `merge`**: a copy loop of the plane-wave merge loses an unrolled
  path when the Helmholtz entry points are added (the codegen-unit effect of CH2.4a, in the
  other setting). The default build, the one every profile here uses (no codegen-unit
  override), is `main`'s.

Bits: Laplace's operations are unchanged, so its outputs are; the whole existing suite passes
(`cargo test --workspace`, the MPI list at 2 and 4 ranks), the D9 runs below compare the
outputs, and `p2p_body_is_independent_of_k` and the Yukawa tests run through the new skeleton.

## 3. Laplace's benchmarks (D9, AVX-512 on Kathleen)

D9 as signed off, where the code changed: AVX-512 on Kathleen, jobs 617315 (node-c11b-015;
the kernel and p = 3) and 617321 (node-c11a-090; p = 8),
`main` (dc2fefb, its own tree and target directory) A and the branch B in the order
A B B A A B B A A B. Per row the spread is max − min of `main`'s five values; a row fails
only if the branch's median is slower than `main`'s by more than that spread. Each run:
`p2p_kernels --isa avx512` (one thread; the gathered W1 rows, ns per pair) and
`tools/bench/run.sh --backend host --precision f32,f64 --degree 3` and, in the second job,
`--degree 8` (N = 10⁶, Auto, 40 threads, gradients on, 10 evaluations after 2 warm-ups;
ms). CH2.4b's plane-wave steps (`plane_wave_parts`) are left out: their code is `main`'s on
every target (§2). NEON and AVX2 need no run: their
Laplace code is `main`'s.

| ISA | precision | output | n_t | main ns/pair | branch ns/pair | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avx512 | f32 | φ | 16 | 0.1614, 0.1646, 0.1614, 0.1614, 0.1621 | 0.1608, 0.1608, 0.1608, 0.1607, 0.161 | 0.003215 | -0.0006229 (-0.4 %) | yes | yes |
| avx512 | f32 | φ | 32 | 0.1604, 0.1603, 0.1603, 0.1603, 0.1603 | 0.1602, 0.1603, 0.1601, 0.1602, 0.1602 | 0.0001543 | -0.0001284 (-0.1 %) | yes | yes |
| avx512 | f32 | φ | 64 | 0.1594, 0.1594, 0.1593, 0.1594, 0.1593 | 0.1592, 0.1592, 0.159, 0.1592, 0.1592 | 0.0001524 | -0.000203 (-0.1 %) | yes | yes |
| avx512 | f32 | φ, ∇φ | 16 | 0.2625, 0.2674, 0.2625, 0.2625, 0.2627 | 0.2627, 0.2627, 0.2627, 0.2693, 0.2583 | 0.004912 | +0.0002758 (+0.1 %) | yes | yes |
| avx512 | f32 | φ, ∇φ | 32 | 0.2534, 0.2532, 0.2531, 0.2532, 0.2532 | 0.2528, 0.2529, 0.2527, 0.2529, 0.2528 | 0.0003207 | -0.0003842 (-0.2 %) | yes | yes |
| avx512 | f32 | φ, ∇φ | 64 | 0.2521, 0.2522, 0.2512, 0.2521, 0.2513 | 0.2518, 0.2519, 0.2516, 0.2518, 0.2519 | 0.001014 | -0.0002539 (-0.1 %) | yes | yes |
| avx512 | f64 | φ | 16 | 0.407, 0.4098, 0.4072, 0.4095, 0.4087 | 0.4095, 0.4097, 0.409, 0.4105, 0.41 | 0.002836 | +0.001004 (+0.2 %) | yes | yes |
| avx512 | f64 | φ | 32 | 0.4034, 0.4055, 0.4029, 0.4055, 0.405 | 0.406, 0.406, 0.4052, 0.406, 0.406 | 0.002614 | +0.0009867 (+0.2 %) | yes | yes |
| avx512 | f64 | φ | 64 | 0.4011, 0.4034, 0.4032, 0.4032, 0.4031 | 0.4037, 0.4039, 0.4034, 0.4037, 0.4037 | 0.002265 | +0.0004884 (+0.1 %) | yes | yes |
| avx512 | f64 | φ, ∇φ | 16 | 0.6039, 0.6057, 0.6039, 0.605, 0.6053 | 0.605, 0.6035, 0.6031, 0.6061, 0.6039 | 0.001829 | -0.001096 (-0.2 %) | yes | yes |
| avx512 | f64 | φ, ∇φ | 32 | 0.5995, 0.6006, 0.6002, 0.6006, 0.5984 | 0.6002, 0.6002, 0.5995, 0.6002, 0.601 | 0.002157 | +0 (+0.0 %) | yes | yes |
| avx512 | f64 | φ, ∇φ | 64 | 0.5959, 0.5959, 0.5956, 0.5963, 0.597 | 0.597, 0.597, 0.5967, 0.597, 0.597 | 0.001422 | +0.001067 (+0.2 %) | yes | yes |

0 rows outside noise

| precision | p | strategy | main ms | branch ms | spread | Δ median | within noise | outputs equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 3 | Batched | 76.04, 75.91, 77.56, 76.56, 74.08 | 77.33, 75.29, 76.18, 76.91, 78.07 | 3.476 | +0.864 (+1.1 %) | yes | yes |
| f64 | 3 | Batched | 110, 110, 112.9, 109.5, 112.4 | 116, 113, 112.8, 112, 112.7 | 3.458 | +2.746 (+2.5 %) | yes | yes |
| f32 | 8 | PlaneWave | 156.4, 151.4, 155.6, 153.5, 155.1 | 157.2, 156.5, 157.4, 154.8, 155.1 | 5.062 | +1.398 (+0.9 %) | yes | yes |
| f64 | 8 | PlaneWave | 210.6, 211.5, 212, 208.3, 208.1 | 216.3, 205.1, 208.2, 204.8, 207.3 | 3.901 | -3.261 (-1.5 %) | yes | yes |

0 rows outside noise

All 16 rows within noise; "outputs equal": the error columns agree in every run of both
sides.

**The FMM outputs byte for byte** (job 617321; CH2.0's procedure): `nd-fmm-bench
--write-problem` once (the cube, N = 10⁵), then `--write-output` from `main`'s and the
branch's binaries (f64, φ and ∇φ at every point, 40 threads, AVX-512) at p = 3, 6 and 8 under
dense, classes, rotation, batched, planewave and auto, `cmp`: 18 of 18 identical
(25,600,024 bytes each). `--write-output` is f64 only; f32's outputs rest on the D9 rows'
error columns and the test suite. C3S.4 (NEON's prototype ratio) needs no run: NEON's code is `main`'s.

## 4. Accuracy

The contract, design §4.12: each part of each potential term within (32 + 12 |k| r) u_T of the
reference's term relative to |q| e^{−Im(k) r}/r, each part of each gradient component within
(66 + 12 |k| r) u_T relative to |q| (1 + |k| r) e^{−Im(k) r}/r²; Re(k) r ≤ 2¹⁰, Im(k) r ≤ 64
(f32) or 512 (f64), 2⁻²⁰ ≤ |q| ≤ 1, in f32 the gradient for (1 + |k| r)/r³ ≤ 2¹²⁶.

`helmholtz_terms_*_large`, 10⁶ seeded pairs per precision (real and complex charges, a third
undamped), largest errors in u_T, potential / gradient, over both charge kinds:

| ISA (machine) | precision | \|k\| r < 1 | per unit \|k\| r, \|k\| r ≥ 1 | largest (\|k\| r ≤ 1,145) |
| --- | --- | --- | --- | --- |
| NEON (M6) | f32 | 6.38 / 9.72 | 6.37 / 9.04 | 3,063 / 2,753 |
| NEON (M6) | f64 | 7.13 / 9.57 | 6.88 / 7.24 | 3,072 / 2,382 |
| AVX2 (Kathleen) | f32 | 5.95 / 9.44 | 6.38 / 6.44 | 3,063 / 2,309 |
| AVX2 (Kathleen) | f64 | 7.13 / 9.44 | 6.88 / 7.24 | 3,071 / 2,382 |
| AVX-512 (Kathleen) | f32 | 6.38 / 9.09 | 6.38 / 6.71 | 3,063 / 2,311 |
| AVX-512 (Kathleen) | f64 | 7.13 / 10.31 | 6.88 / 7.36 | 3,071 / 2,368 |
| scalar (both) | both | 0 / 0 | 0 / 0 | 0 / 0 |

Against the derived 30u and 64u at small |k| r and 11.8u per unit of |k| r. The sums
(`helmholtz_sums_*_large`, FMM-shaped sets and up to 4,096 sources, k = 0.1, 1, 10,
1 + 0.5i, 10 + i in leaf units, real and complex charges, relative to the term magnitudes):
f32 at most 4.1e-7 (φ) and 1.7e-6 (∇φ, within twice the reference's own error), f64
1.8e-15 and 1.8e-15, on every ISA. The ISAs run, from the test output: M6 scalar and NEON;
Kathleen scalar, AVX2 and AVX-512.

## 5. Cost per pair

`spikes/p2p-simd/examples/helmholtz.rs` (full mode: the W1 cells n_t = 16, 32, 64, per-pair
and gathered; one thread; k = 1 and the damped k = 1 + 0.1i in leaf units; BLAS threads 1).
"real q" and "complex q" are the Helmholtz kernel at real k, "damped" complex charges at the
damped k; ratios are times per pair, except "ours cq / green cq", our Gpairs/s over
green-kernels' (above 1: ours faster). green-kernels' `Helmholtz3dKernel` (real k, complex
charges) agrees with `direct_sum` on a pair within 2.4e-16 (4π × its φ and ∂/∂x).

### 5.1 Geometric means over the cells

M6 (load 2.7–3 during the run):

| precision | output | code | real q / L | cq / L | damped / L | cq / Y | ours cq / green cq |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f32 | φ | nd-fmm-simd neon (default) | 4.12 | 4.28 | 6.36 | 1.50 | 5.27 |
| f32 | φ | nd-fmm-simd scalar | 6.32 | 6.38 | 6.36 | 3.34 | 0.70 |
| f32 | φ | green-kernels | – | 14.91 | – | – | – |
| f32 | φ, ∇φ | nd-fmm-simd neon (default) | 3.86 | 3.93 | 5.52 | 1.63 | 3.71 |
| f32 | φ, ∇φ | nd-fmm-simd scalar | 3.99 | 4.02 | 4.01 | 3.22 | 0.61 |
| f32 | φ, ∇φ | green-kernels | – | 11.21 | – | – | – |
| f64 | φ | nd-fmm-simd neon (default) | 5.80 | 5.93 | 9.12 | 1.52 | 3.01 |
| f64 | φ | nd-fmm-simd scalar | 7.97 | 8.01 | 8.02 | 4.23 | 0.79 |
| f64 | φ | green-kernels | – | 9.92 | – | – | – |
| f64 | φ, ∇φ | nd-fmm-simd neon (default) | 4.68 | 4.79 | 6.75 | 1.53 | 2.31 |
| f64 | φ, ∇φ | nd-fmm-simd scalar | 5.46 | 5.42 | 5.43 | 3.67 | 0.69 |
| f64 | φ, ∇φ | green-kernels | – | 8.16 | – | – | – |

locust (load 0.65 before, one other user's single-threaded process):

| precision | output | code | real q / L | cq / L | damped / L | cq / Y | ours cq / green cq |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f32 | φ | nd-fmm-simd neon (default) | 5.19 | 5.58 | 8.06 | 1.67 | 2.01 |
| f32 | φ | nd-fmm-simd scalar | 6.27 | 6.27 | 6.27 | 2.78 | 0.47 |
| f32 | φ | green-kernels | – | 8.29 | – | – | – |
| f32 | φ, ∇φ | nd-fmm-simd neon (default) | 5.52 | 5.73 | 7.34 | 1.59 | 1.66 |
| f32 | φ, ∇φ | nd-fmm-simd scalar | 6.43 | 6.43 | 6.43 | 2.25 | 0.39 |
| f32 | φ, ∇φ | green-kernels | – | 6.36 | – | – | – |
| f64 | φ | nd-fmm-simd neon (default) | 4.57 | 4.90 | 6.38 | 1.57 | 2.09 |
| f64 | φ | nd-fmm-simd scalar | 10.22 | 10.26 | 10.51 | 4.45 | 0.76 |
| f64 | φ | green-kernels | – | 10.97 | – | – | – |
| f64 | φ, ∇φ | nd-fmm-simd neon (default) | 4.36 | 4.51 | 5.92 | 1.42 | 1.84 |
| f64 | φ, ∇φ | nd-fmm-simd scalar | 5.26 | 5.26 | 5.42 | 3.06 | 0.70 |
| f64 | φ, ∇φ | green-kernels | – | 7.86 | – | – | – |

Kathleen (job 617315; green-kernels' pulp backend is AVX2 on this CPU):

| precision | output | code | real q / L | cq / L | damped / L | cq / Y | ours cq / green cq |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f32 | φ | nd-fmm-simd avx512 (default) | 3.85 | 4.08 | 5.44 | 1.36 | 11.86 |
| f32 | φ | nd-fmm-simd scalar | 8.74 | 8.84 | 8.85 | 2.66 | 0.53 |
| f32 | φ | nd-fmm-simd avx2 | 4.74 | 4.90 | 6.14 | 1.45 | 5.78 |
| f32 | φ | green-kernels | – | 9.32 | – | – | – |
| f32 | φ, ∇φ | nd-fmm-simd avx512 (default) | 3.68 | 3.79 | 4.79 | 1.59 | 8.50 |
| f32 | φ, ∇φ | nd-fmm-simd scalar | 5.26 | 5.31 | 5.31 | 2.16 | 0.46 |
| f32 | φ, ∇φ | nd-fmm-simd avx2 | 3.72 | 3.86 | 4.63 | 1.45 | 4.41 |
| f32 | φ, ∇φ | green-kernels | – | 7.55 | – | – | – |
| f64 | φ | nd-fmm-simd avx512 (default) | 4.64 | 5.30 | 6.80 | 1.51 | 11.26 |
| f64 | φ | nd-fmm-simd scalar | 12.74 | 12.88 | 13.21 | 3.32 | 0.75 |
| f64 | φ | nd-fmm-simd avx2 | 4.38 | 4.38 | 5.61 | 1.38 | 4.67 |
| f64 | φ | green-kernels | – | 14.00 | – | – | – |
| f64 | φ, ∇φ | nd-fmm-simd avx512 (default) | 4.15 | 4.46 | 5.41 | 1.42 | 9.55 |
| f64 | φ, ∇φ | nd-fmm-simd scalar | 8.40 | 8.41 | 8.86 | 2.89 | 0.66 |
| f64 | φ, ∇φ | nd-fmm-simd avx2 | 3.54 | 3.63 | 5.47 | 1.21 | 4.09 |
| f64 | φ, ∇φ | green-kernels | – | 10.95 | – | – | – |

### 5.2 Gpairs/s at the gathered W1 cell n_t = 64

| machine | precision | output | code | real q | complex q | damped | Yukawa | Laplace | max φ | max ∇φ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M6 | f32 | φ | nd-fmm-simd neon (default) | 1.292 | 1.241 | 0.835 | 1.859 | 5.324 | 8.5e-8 | – |
| M6 | f32 | φ | green-kernels | – | 0.217 | – | – | 3.712 | 5.7e-8 | – |
| M6 | f32 | φ, ∇φ | nd-fmm-simd neon (default) | 0.852 | 0.832 | 0.593 | 1.357 | 3.278 | 7.8e-8 | 5.0e-7 |
| M6 | f32 | φ, ∇φ | green-kernels | – | 0.204 | – | – | 2.720 | 3.1e-8 | 2.0e-7 |
| M6 | f64 | φ | nd-fmm-simd neon (default) | 0.468 | 0.457 | 0.298 | 0.694 | 2.704 | 1.5e-16 | – |
| M6 | f64 | φ | green-kernels | – | 0.147 | – | – | 1.533 | 1.6e-16 | – |
| M6 | f64 | φ, ∇φ | nd-fmm-simd neon (default) | 0.335 | 0.327 | 0.232 | 0.499 | 1.598 | 1.5e-16 | 7.0e-16 |
| M6 | f64 | φ, ∇φ | green-kernels | – | 0.138 | – | – | 1.181 | 1.2e-16 | 5.1e-16 |
| locust | f32 | φ | nd-fmm-simd neon (default) | 0.370 | 0.344 | 0.240 | 0.575 | 1.938 | 8.5e-8 | – |
| locust | f32 | φ | green-kernels | – | 0.173 | – | – | 1.563 | 5.7e-8 | – |
| locust | f32 | φ, ∇φ | nd-fmm-simd neon (default) | 0.251 | 0.242 | 0.188 | 0.388 | 1.410 | 7.8e-8 | 5.0e-7 |
| locust | f32 | φ, ∇φ | green-kernels | – | 0.148 | – | – | 1.015 | 3.2e-8 | 2.0e-7 |
| locust | f64 | φ | nd-fmm-simd neon (default) | 0.122 | 0.113 | 0.087 | 0.179 | 0.554 | 1.5e-16 | – |
| locust | f64 | φ | green-kernels | – | 0.054 | – | – | 0.617 | 1.6e-16 | – |
| locust | f64 | φ, ∇φ | nd-fmm-simd neon (default) | 0.097 | 0.093 | 0.071 | 0.133 | 0.427 | 1.5e-16 | 7.0e-16 |
| locust | f64 | φ, ∇φ | green-kernels | – | 0.050 | – | – | 0.412 | 9.9e-17 | 5.1e-16 |
| Kathleen | f32 | φ | nd-fmm-simd avx512 (default) | 1.453 | 1.374 | 1.002 | 1.867 | 6.255 | 7.8e-8 | – |
| Kathleen | f32 | φ | nd-fmm-simd avx2 | 0.558 | 0.541 | 0.429 | 0.798 | 2.872 | 8.2e-8 | – |
| Kathleen | f32 | φ | green-kernels | – | 0.091 | – | – | 1.026 | 2.9e-8 | – |
| Kathleen | f32 | φ, ∇φ | nd-fmm-simd avx512 (default) | 1.024 | 0.987 | 0.762 | 1.514 | 3.955 | 7.8e-8 | 5.0e-7 |
| Kathleen | f32 | φ, ∇φ | nd-fmm-simd avx2 | 0.424 | 0.409 | 0.336 | 0.577 | 1.619 | 8.5e-8 | 5.0e-7 |
| Kathleen | f32 | φ, ∇φ | green-kernels | – | 0.088 | – | – | 0.786 | 2.6e-8 | 1.0e-7 |
| Kathleen | f64 | φ | nd-fmm-simd avx512 (default) | 0.497 | 0.434 | 0.337 | 0.665 | 2.479 | 1.3e-16 | – |
| Kathleen | f64 | φ | nd-fmm-simd avx2 | 0.176 | 0.176 | 0.138 | 0.246 | 0.796 | 1.3e-16 | – |
| Kathleen | f64 | φ | green-kernels | – | 0.037 | – | – | 0.575 | 8.4e-17 | – |
| Kathleen | f64 | φ, ∇φ | nd-fmm-simd avx512 (default) | 0.390 | 0.364 | 0.295 | 0.500 | 1.674 | 1.4e-16 | 8.6e-16 |
| Kathleen | f64 | φ, ∇φ | nd-fmm-simd avx2 | 0.155 | 0.152 | 0.098 | 0.176 | 0.548 | 1.3e-16 | 8.6e-16 |
| Kathleen | f64 | φ, ∇φ | green-kernels | – | 0.035 | – | – | 0.415 | 5.8e-17 | 2.9e-16 |

### 5.3 Blocks and inner loops

- **K.** K = 2 against K = 1: on the M6 (NEON, `--quick`) K = 2 equal without gradients
  and up to 1.19× slower with them; on Kathleen (AVX-512, the full run with
  `HELMHOLTZ_K = 2` built on the node) the Gpairs/s of K = 2 over K = 1 have geometric means
  0.98, 1.00, 1.00 and 0.97 (f32 φ, f32 φ and ∇φ, f64 φ, f64 φ and ∇φ), per cell 0.90–1.11.
  K = 1 on every ISA.
- **Source loops** (the inlining check on the linked release test binaries, NEON natively,
  AVX2 and AVX-512 from the `x86_64-apple-darwin` build; whole blocks / padded tail; no call
  in any):

| ISA | precision | output | real q | real q, damped | complex q | complex q, damped |
| --- | --- | --- | --- | --- | --- | --- |
| NEON | f32 | φ | 59 / 59 | 116 / 88 | 62 / 62 | 118 / 94 |
| NEON | f32 | φ, ∇φ | 88 / 81 | 122 / 119 | 93 / 83 | 138 / 123 |
| NEON | f64 | φ | 108 / 81 | 143 / 122 | 103 / 84 | 153 / 126 |
| NEON | f64 | φ, ∇φ | 109 / 100 | 164 / 143 | 111 / 103 | 167 / 148 |
| AVX2 | f32 | φ | 76 / 70 | 101 / 97 | 77 / 73 | 109 / 98 |
| AVX2 | f32 | φ, ∇φ | 106 / 106 | 138 / 136 | 110 / 110 | 144 / 140 |
| AVX2 | f64 | φ | 102 / 94 | 136 / 139 | 105 / 97 | 141 / 138 |
| AVX2 | f64 | φ, ∇φ | 129 / 133 | 176 / 174 | 132 / 135 | 181 / 176 |
| AVX-512 | f32 | φ | 55 / 53 | 75 / 71 | 58 / 56 | 78 / 74 |
| AVX-512 | f32 | φ, ∇φ | 73 / 70 | 94 / 92 | 76 / 73 | 97 / 95 |
| AVX-512 | f64 | φ | 72 / 65 | 103 / 88 | 75 / 68 | 106 / 91 |
| AVX-512 | f64 | φ, ∇φ | 84 / 81 | 119 / 119 | 87 / 84 | 122 / 122 |

## 6. Checks

| command | where | result |
| --- | --- | --- |
| `cargo fmt --all -- --check` | M6 | passed |
| `cargo clippy -- -D warnings`, `--examples` | M6 | passed |
| `RUST_MIN_STACK=8388608 cargo test` | M6 | passed (1,104) |
| `cargo doc --no-deps` | M6 | passed |
| `cargo clippy --workspace --all-targets -- -D warnings` | M6 | passed (the spike included) |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | M6 | passed (1,186) |
| `cargo clippy -p nd-fmm-simd --all-targets --target x86_64-apple-darwin -- -D warnings` | M6 | passed |
| `cargo test -p nd-fmm-simd --release -- --ignored --show-output` | M6 | passed (21; the Helmholtz ones again after the review's sampling change) |
| `cargo test --release -p nd-fmm-spike-p2p-simd --example helmholtz` | M6 | passed |
| `.github/scripts/run-mpi-tests.sh` build, run at 2 and 4 ranks | M6 | passed (24 of 24) |
| `check.sbatch simd` (Clippy, tests, ignored release tests) | Kathleen 617314, 617322 | passed |
| disassembly comparison, 4 targets × 2 settings | M6 | §2 |
| inlining check | M6 | §5.3, no call |
| D9 (AVX-512) | Kathleen 617315, 617321 | 16 of 16 within noise |
| FMM outputs of `main` and the branch, byte for byte (18 cases, f64) | Kathleen 617321 | 18 of 18 identical |

Not run: the licence check (no default member's manifest changed); D9 on the M6 and locust
and C3S.4 (NEON's and AVX2's code is `main`'s, §2).

## 7. Questions for sign-off

1. The contract as derived, (32 + 12 |k| r) u_T and (66 + 12 |k| r) u_T, with the f32
   gradient range (1 + |k| r)/r³ ≤ 2¹²⁶ (Laplace's 2⁻⁸⁴ ≤ r² at k = 0; the reference
   overflows beyond).
2. Constraint 1 as met: NEON's and AVX2's Laplace code `main`'s (and Yukawa's, but for a
   register exchange in NEON's f32 gradient loop on macOS); the FMM's outputs byte for byte
   `main`'s on AVX-512 (18 cases); AVX-512's two Laplace gradient entry
   points with `main`'s source loops but other block-loop control, within noise in D9 on
   Kathleen (D9 run only there, where the code changed); NEON's `merge` differs with one
   codegen unit only.
3. `evaluate_helmholtz` panics for Re k < 0 (the reference accepts any finite k; §3.15's
   range is Re k ≥ 0, Im k ≥ 0).

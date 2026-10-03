# Spike report: device arithmetic per CubeCL backend (Phase 4 / T3)

The brief is docs/phase4/T3-device-arithmetic.md. This spike measures what each CubeCL
backend does to floating-point arithmetic. From that it drafts the CONVENTIONS §3.13
addition "Device kernels" and proposes the device P2P contract. It also times a
CPU-shaped P2P on the CubeCL CPU runtime against `nd-fmm-simd` (decision 10). The raw
output of the full run is in `results-m3max.md`. Every table below comes from it unless
marked otherwise.

## Summary

- **The §3.13 rule survives on both backends that ran.** r² = 0 exactly for coincident
  stored points, and every other r² ≥ 2⁻¹⁰⁶. Checked on 209,303 adversarial and 216,000
  random f32 pairs on Metal and the CPU runtime, and 439,192 + 216,000 f64 pairs on the
  CPU runtime, in five formulations of ŷ and r². There were no exceptions, and the
  device ŷ and dₖ equal the host's bit for bit in every component. This holds although
  Metal flushes subnormals to zero and uses approximate `sqrt` and division.
- **Metal (wgpu-msl) compiles with Apple's default fast math, and it shows.**
  - `sqrt` and division are not correctly rounded: up to 1.78 and 2.26 u_T, with 28% of
    results differing from the correctly rounded value.
  - `1 / sqrt(x)` is compiled to the hardware `rsqrt` (1.50 u_T, the same bits as
    `inverse_sqrt`).
  - Subnormal inputs, results and compared values flush to zero.
  - `(x + b) − x` is simplified to b when x is one value. No reassociation across
    distinct values was observed.
- **The CPU runtime is IEEE apart from contraction.**
  - `sqrt`, division and `recip` are correctly rounded (exhaustive f32 over [1, 4), 10⁷
    f64 samples, 0 differences).
  - `inverse_sqrt` is polyfilled to fl(1 / fl(√x)) (1.50 u_T).
  - Subnormals are kept.
- **Contraction everywhere.** cubecl-opt's `InstCombinePass` fuses every a · b ± c whose
  product has no other use, on both backends, as T2 found. A product with a second use
  is not fused, by CubeCL or by Metal. An explicit `fma` is honoured on both.
- **cubecl-opt folds that break IEEE**, on both backends: `∞ − ∞` (one value) → 0, and
  `−0 + 0.0` → −0. They are harmless for P2P but rule out some idioms (see the
  formulation rules).
- **No option turns fast math off.**
  - `#[cube(fast_math = …)]` is stored but never read by any emitter.
  - wgpu-hal compiles the MSL with a bare `MTLCompileOptions`.
  - `metal-native` (`cubecl-metal`, `MTLMathMode::Safe`) is the only safe-math
    alternative, and its 0.11.0-pre.4 release **does not compile**: `DebugInformation`
    is not imported at `cubecl-metal-0.11.0-pre.4/src/compute/context.rs:139`. So the
    cost of safe math in a P2P loop could not be measured. It is also not needed (see
    above).
- **The device P2P meets the C3S.4 contract on every backend run.**
  - Pair terms over the whole kernel domain (10⁶ pairs): ≤ 4.6 u_T (φ) and ≤ 10.3 u_T
    (∇φ), against 8 / 16.
  - On Metal with explicit-fma r², a lone dₖ² that flushes raises φ to 6.6–7.9 u_T. That
    needs |dₖ| < 2⁻⁶³, which leaf-scaled data cannot produce. On pairs whose nonzero |dₖ|
    are ≥ 2⁻⁵³, every candidate is at 3.97 u_T.
  - Sums are within the reference's own error on W1.
- **Decision 10: the rule says "target".** The CPU-shaped P2P on the CPU runtime takes
  **1.004×** the time of `nd_fmm_simd::P2pKernel` (NEON) at one thread, the geometric
  mean over the 24 W1 cells. The threshold is 1.5.
  - Its inner loop has the NEON kernel's FP operations, and its potentials equal
    `nd-fmm-simd`'s bit for bit.
  - On 12 units it takes **2.73×** the time of `nd-fmm-simd` on 12 rayon threads, because
    of the runtime's scheduling. That figure is reported, not a target.
- **Leaf operators and GEMM.**
  - The device GEMM y ← y + A x in `MatrixSet::apply`'s order equals the host with every
    multiply–add fused, bit for bit, on both backends and in both precisions. So T8 and
    T9 can test bit for bit against an fma-emulating host loop.
  - The harmonics recursion is not bit-identical (25–48% of values). At p = 20 it stays
    within the precision's noise, except on Metal f32, where values below 2⁻¹²⁶ flush (an
    error of 0.07 relative, per degree). At the tested f32 range, p ≤ 8, Metal is within
    60 u_T of the host.

## Recommendation, for sign-off

**Status: signed off on 2026-10-03**, every item below as recommended (docs/phase4/README.md,
decisions 3 and 10).

### 1. The §3.13 addition

As drafted in `docs/CONVENTIONS.md` §3.13, "Device kernels" (this PR):
- the arithmetic every device kernel may assume (the minimum over the backends);
- the coincident-pair rule under four formulation conditions;
- the domain and the f32 ranges carried over, with one addition for flushing backends:
  the contracts hold for q = 0 or |q| ≥ 2⁻¹⁰⁰;
- §3.10 unchanged, `CONVENTION_VERSION` stays 1.

### 2. The device P2P contract

| backend | precision | pair terms (proposed) | measured worst, kernel domain | measured worst, nonzero \|dₖ\| ≥ 2⁻⁵³ | sums (proposed) | measured W1 sum error [reference] |
| --- | --- | --- | --- | --- | --- | --- |
| Metal (wgpu-msl) | f32 | 8 u_T φ, 16 u_T ∇φ (C3S.4) | 3.97 / 8.63 (recommended formulation) | 3.97 / 8.63 | 1e-6, or 2× the reference's error | φ 1.16e-7 [1.28e-7], ∇φ 5.93e-7 [6.23e-7] |
| CPU runtime | f32 | 8 u_T φ, 16 u_T ∇φ | 4.00 / 10.30 | 4.00 / 10.30 | 1e-6, or 2× the reference's error | 1.28e-7 [1.28e-7], 5.93e-7 [6.23e-7] |
| CPU runtime | f64 | 8 u_T φ, 16 u_T ∇φ | 4.00 / 10.20 | 4.00 / 10.20 | 1e-14, or 2× the reference's error | 2.86e-16 [3.08e-16], 1.89e-15 [1.75e-15] |
| CUDA | f32, f64 | 8 / 16 u_T, unverified (not run) | — | — | as above | — |

- Each backend meets the provisional contract of C3S.4, so the proposal keeps it
  unchanged.
- On a backend that flushes subnormals (Metal), the contract applies to q = 0 or
  |q| ≥ 2⁻¹⁰⁰, as the §3.13 addition states.
- The gradient margin is smaller than on the host: 10.3 of 16 u_T, against the NEON
  kernel's 1.5 u_T for 1/r alone. The reasons are the device's r² (fused differently
  from the reference) and, on Metal, its `rsqrt`.

**Recommended formulation, every backend and precision** (T6):
- ŷ = `fma(r̂, u_s, ĉ)` (u_s itself for s = t), and dₖ = u_t,k − ŷₖ.
- r² = `fma(d₂, d₂, fma(d₁, d₁, d₀ · d₀))`.
- ρ = `inverse_sqrt(r²)`, then ρ = `select(r² == 0, 0, ρ)`. On the CPU runtime this is
  fl(1 / fl(√r²)), the same bits as `sqrt` and division. On Metal it is the hardware
  `rsqrt`, the same bits as `1 / sqrt`, which Metal rewrites anyway. Writing
  `inverse_sqrt` states what runs.
- No Newton step: it costs 5–15% on Metal and buys nothing the contract needs.
- φ += q ρ and ∇φ −= (q ρ)(ρ ρ) d, written as `fma` calls. The accumulation into the
  owner's register follows device-path.md §6.1.

**Compiler options:** none exists, and none is required.

**Reproducibility** (in the manner of simd-p2p.md §5.5):
- One backend, device and build: bit-identical from run to run. Nothing in the kernel
  depends on scheduling.
- Between backends, results differ within the contract:
  - Metal's `sqrt`, division and `rsqrt` are not correctly rounded, and it flushes
    subnormals;
  - the CPU runtime is correctly rounded;
  - CUDA (unverified) is presumably correctly rounded on its default LLVM path, and its
    NVRTC path would emit CUDA's approximate `rsqrt`.
- The CPU runtime against the host:
  - The recommended formulation (with `inverse_sqrt`) is not bit-identical to
    `nd-fmm-simd`.
  - The CPU-shaped spike kernel (`sqrt` and division, NEON's operation order) equals
    `nd-fmm-simd` NEON bit for bit in every potential, and in about 70% of gradient
    components.
  - Metal against Apple's `rsqrt` on other GPU families: not measured, since only the
    M3 Max ran.

### 3. Formulation rules for the later kernel tasks

1. **ŷ by explicit fma** (T6, and T7's frames). Write `fma(r̂, u_s, ĉ)`, never `ĉ + r̂ u_s`
   inline in a subtraction. The product is exact, so the value is the host's. An fma
   cannot be reassociated, while the inline form could become (u_t − ĉ) − r̂ u_s. That
   would change 164,601 of 627,909 f32 adversarial components and gain or lose a zero in
   123,571. Neither backend did it, but nothing guarantees that.
2. **Mask by compare and select**, never by multiplying with a 0/1 mask. ∞ · 0 = NaN, and
   Metal's fast math may assume no infinities.
3. **Do not write an expression that uses one value twice and relies on IEEE**:
   - no compensated (Kahan or Neumaier) summation: Metal simplifies (x + b) − x to b;
   - no x − x to produce a NaN or a zero of known sign: cubecl-opt folds it to 0;
   - no x + 0.0 to normalise −0: cubecl-opt folds it to x.
4. **Do not rely on a product staying unfused or being fused.** Any lone a · b ± c may be
   fused (`InstCombinePass`, every backend). Where the result must be pinned, write the
   `fma`. A product with another use stays unfused on both backends, but CSE can merge
   two identical products into one value with two uses on the CPU runtime (CSE runs
   before InstCombine there).
5. **Assume nothing about subnormals.** Metal flushes them, the CPU runtime keeps them.
   Kernels whose intermediates can leave the normal range of f32 (high-degree harmonics
   at small |x|, p ≥ 20) lose those values on Metal (section "Leaf operators and GEMM").
6. **Bit identity, where tests may ask for it** (T7, T8):
   - **GEMM** in the host order equals a host loop with `mul_add` in the same order,
     bit for bit, on both backends;
   - **harmonics, P2M, L2P, P2L, M2P and rotation**: tolerances only. About half the
     values differ in the last bits, and on Metal `sqrt` and division are inexact
     anyway;
   - **frames** (products of powers of two and small integers) and **the scatter and
     zeroing kernels**: bit for bit (device-path.md §9.2; T3 confirms ŷ bit for bit).

### 4. Decision 10: CPU-runtime P2P against `nd-fmm-simd`

Geometric mean of (CPU-runtime time) / (`nd-fmm-simd` time) over the 24 W1 cells (f32 and
f64, φ and φ + ∇φ, n_t ∈ {8, 16, 24, 32, 64, 128}):

| | one thread | all performance cores (12) |
| --- | --- | --- |
| geometric mean | **1.004** | 2.729 |
| range over cells | 0.95–1.03 | 0.91–9.30 |

**The rule's outcome: at most 1.5, so recommend a CPU-runtime performance target.** The
CPU runtime's P2P in the FMM should be within 1.5× of `nd_fmm_simd::P2pKernel` per pair at
one thread, through the CPU layout of the device P2P (device-path.md §6.2, "CPU layout"),
added to T6. This kernel is its prototype: targets in `Vector<T, N>` lanes, K vectors per
block, sources broadcast, no shared memory, one unit per core.

The all-cores ratio is 2.7, not 1.0. The runtime's threading is the cause:
- A queued launch of 12 units costs 190–230 µs, against 2.4 µs for one unit.
- Each unit becomes a channel message to a pool of 16 workers, which includes the 4
  efficiency cores. Pinning is only a hint on macOS. A launch waits for its slowest unit.
- The ratio falls from 2.7–9.3 at n_t = 8 to 0.9–1.3 at n_t = 128, as the work per
  launch grows.

For `threads(n)` and MPI ranks per node (design §6.8):
- The CPU runtime always starts one worker per logical CPU, and nothing configures that
  count.
- `threads(n)` can only cap the units per cube of the CPU layout. It cannot keep the pool
  off the efficiency cores or off another rank's cores.
- Several ranks per node with the CPU backend would oversubscribe the node, one pool of
  16 workers each. The device backend runs on one rank until C5.1 anyway
  (`SettingsError::DeviceNeedsOneRank`). When C5.1 lifts that, the CPU backend should
  run one rank per node.
- The target therefore applies at one thread, as the rule says. The multi-core ratio is
  reported, and T6 should measure it again with larger launches (a whole level, not 768
  leaves).

## Setup

- **Machine:** Apple M3 Max, 16 cores (12 performance, 4 efficiency), macOS; Metal
  device with 40 GPU cores, plane size 32 and 32 KB of shared memory per cube.
- **Software:** CubeCL `=0.11.0-pre.4` (the workspace pin); the CPU runtime through
  `cubecl-llvm` with the tracel-llvm 23.1.0-3 bundle; Metal through wgpu 30 with the MSL
  compiler (`runtime wgpu<msl>`, f64 unsupported). Release build, rustc 1.99.0.
- **Backends run:** Metal (f32) and the CPU runtime (f32, f64). **CUDA:**
  `cargo check -p nd-fmm-spike-device-arith --no-default-features --features cuda`
  type-checks. It was not run, and nothing CUDA-specific in this report is measured.
  **metal-native:** does not build at the pin (above).
- **Run:** outside the macOS sandbox (Metal needs GPU access; `sysctl` for the machine
  line), with every BLAS thread variable and `RAYON_NUM_THREADS` set to 1. The
  12-thread rayon pool of the host baseline is built explicitly. The command is under
  "Reproducing".
- **Timing:** kernel compilation excluded (a warm-up launch and sync); launches queued
  between syncs; the median of 15 batches of at least 20 ms
  (`nd_fmm_validate::bench::median_time_per_call`). The Metal P2P rates are quick
  rankings: one run each, about ±25% run-to-run variance expected (Phase 0 T6).
- **Inputs:** all seeded. The §3.13 pairs are rebuilt in Rust from the description in
  `tools/fixtures/check_p2p_domain.py`, with the same four domains, levels 0–16,
  target-leaf patterns, U lists of a 2:1-balanced tree, positions, ulp windows and leaf
  faces as `points_to_morton` draws them. No fixture files. The generic and far domains
  come from seeded clouds of this spike's own generator, so they are not the script's
  exact boxes.

## How each backend lowers the float operations (from the 0.11.0-pre.4 sources)

Paths are relative to the cargo registry. Established by reading the sources, and
confirmed by the measurements below where a backend ran.

| operation | frontend | CPU runtime (cubecl-llvm, CPU target) | Metal (cubecl-cpp MSL, wgpu-msl) | CUDA, default LLVM NVPTX path (not run) | CUDA, `cuda-cpp` NVRTC path (not run) |
| --- | --- | --- | --- | --- | --- |
| `x.sqrt()` | `SqrtOp` | `llvm.sqrt` (`cubecl-llvm/src/shared/to_llvm/math.rs:55`) | `sqrt(x)` (`cubecl-cpp/src/shared/unary.rs:88`) | `llvm.sqrt`, no fast-math flags | `sqrt(x)` |
| `x.inverse_sqrt()` | `RsqrtOp` | polyfill `1 / x.sqrt()` (`cubecl-llvm/src/shared/polyfill/math.rs:185`) | `rsqrt(x)` (`unary.rs:89`) | the same polyfill | `rsqrt(x)` (CUDA's approximate rsqrt) |
| `x.recip()` | `RecipOp` | polyfill `1 / x` | `1 / x` (`cubecl-core/src/frontend/polyfills.rs:249`) | `1 / x` | `1 / x` |
| `a / b` | `FDivOp` | `fdiv`, no fast-math flags (`math.rs:380`) | `a / b` | `fdiv`, no flags | `a / b` |
| `fma(a, b, c)` | `FmaOp` | `llvm.fmuladd` (`math.rs:485`): fusion permitted, not required; fused on AArch64 (measured, and in the dumped assembly) | `fma(a, b, c)` | `llvm.fmuladd` | `fma` |
| a · b ± c, product used once | `InstCombinePass` → `FmaOp`, with no fast-math check (`cubecl-opt/src/passes/inst_combine.rs:30`) | runs (`cubecl-llvm/src/shared/base.rs:420-425`, after CSE) | runs (`cubecl-cpp/src/shared/base.rs:272`, before CSE) | runs; also `contract` on fadd, fsub and fmul (`math.rs:336-345`) | runs |
| fast math | `#[cube(fast_math)]` sets `fp_math_mode`, which no emitter reads | O3 (`cpu/jit/engine.rs:141`), no fast-math or denormal attributes | MSL compiled by wgpu-hal with a bare `MTLCompileOptions::new()` (`wgpu-hal-30.0.1/src/metal/device.rs:1303`): Apple's default, fast math | no ftz or `nvptx-f32ftz` attributes | NVRTC options are only arch, includes and `-lineinfo` (`cubecl-cuda/src/compute/context.rs:415`): fmad on, prec-div and prec-sqrt on, ftz off by NVRTC's defaults |

Also in cubecl-opt, with no flag check:
- SCCP and SimplifyOps fold `x − x → 0` (one SSA value), `0 · x → 0`, `0 / x → 0`,
  `x + 0 → x` (the pattern also matches −0), `x · 1 → x` and `x / 1 → x`
  (`cubecl-ir/src/dialect/math.rs:250–478`).
- Nothing folds `(a + b) − a`.
- The CPU runtime dumps only LLVM IR (`CUBECL_DEBUG_PLIRON` with the `pliron-dump`
  feature), and has no assembly or object dump.

## Primitive accuracy

Maximum relative error in u_T against the exact value, over normal arguments. "≠ host"
counts results whose bits differ from the correctly rounded host value (`sqrt`, the
divisions) or from the host's fl(1 / fl(√x)) (the inverse square roots). Newton:
y + y (½ − ½ x y²) with two fmas.

| operation | Metal f32, [1, 4) exhaustive (16.8 M) | Metal f32, 2^k m (k = −108…7) | CPU f32, [1, 4) exhaustive | CPU f32, 2^k m | CPU f64, 10⁷ log-uniform on [2⁻¹⁰⁸, 2⁷] | CPU f64, 2^k m |
| --- | --- | --- | --- | --- | --- | --- |
| `sqrt(x)` | 1.783 (28.4% ≠) | 1.651 | 1.000 (0 ≠) | 1.000 | 1.000 (0 ≠) | 1.000 |
| `a / x` | 2.257 (28.4% ≠) | 2.123 | 1.000 (0 ≠) | 1.000 | 1.000 (0 ≠) | 1.000 |
| `1 / x`, `recip(x)` | 1.285 (10.3% ≠) | 1.269 | 1.000 (0 ≠) | 1.000 | 1.000 (0 ≠) | 1.000 |
| `inverse_sqrt(x)` | 1.500 | 1.249 | 1.500 (= fl(1/fl(√x))) | 1.500 | 1.497 (= fl(1/fl(√x))) | 1.500 |
| `inverse_sqrt` + Newton | 1.244 | 1.217 | 1.244 | 1.217 | 1.248 | 1.227 |
| `1 / sqrt(x)` | 1.500 (bits of `inverse_sqrt`) | 1.249 | 1.500 | 1.500 | 1.497 | 1.500 |

On Metal, `a / x` is about two roundings (presumably a · rcp(x)) and `1 / x` one-and-a-bit.
`1 / sqrt(x)` gives exactly the bits of `inverse_sqrt(x)`, so fast math rewrote it.

Edge cases, IEEE where not marked:
- **Metal f32:** `sqrt(subnormal)` = +0; `inverse_sqrt(subnormal)` and `1 / sqrt` = +∞
  (the input flushed); Newton gives NaN at 0, −0, subnormal and ∞ (y = ∞ or 0 makes
  x y² NaN), and a wrong value at the smallest normal (½ x flushes).
- **CPU runtime:** Newton gives NaN at ±0 and ∞, as the IEEE formula does; f32 and f64
  subnormals are handled exactly.
- **Both:** ±0, ∞ and NaN otherwise give the IEEE results.
- Neither matters for P2P, which applies these operations only to r² in the domain.

## Compiler behaviour

| probe | Metal f32 | CPU f32 | CPU f64 |
| --- | --- | --- | --- |
| a · b + c, c + a · b, a · b − c, c − a · b (e² = half an ulp) | fused | fused | fused |
| p = a · b with a second use; p + c | not fused | not fused | not fused |
| explicit `fma(a, b, c)` | fused (honoured) | fused | fused |
| a · a + (−a) · a | fused | fused | fused |
| (x + b) − x, one value x | **simplified to b** | as written | as written |
| x − (x + z), one value x | **simplified to −z** | as written | as written |
| (x + b) − y, x − (y + z), x = y = 1 in distinct slots | as written | as written | as written |
| subnormal result of − and ×; subnormal input; subnormal + subnormal | **flushed** | kept | kept |
| `select(subnormal == 0, …)` | **true (flushed)** | false | false |
| `select(−0 == 0, …)` | true | true | true |
| q · select(r² == 0, 0, 1/√r²) and `if r² == 0 { t = 0 }`, at r² = 0 | 0 | 0 | 0 |
| ∞ − ∞ (one value) | **0** (cubecl-opt fold) | **0** | **0** |
| −0 + 0.0 (literal) | **−0** (cubecl-opt fold) | **−0** | **−0** |

The contraction probes give each product its own copy of a. On the CPU runtime, CSE runs
before InstCombine, so a shared a · b would have several uses and never fuse.

## The §3.13 argument on the device

Kernel `domain::domain_kernel` computes per pair, as a P2P kernel would:
- ŷ in two forms: written inline, d = u_t − (ĉ + r̂ u_s), and by an explicit fma;
- an unfused ŷ, whose product has a second use, so that only a reassociating compiler
  could change it;
- r² in three orders.

| backend, precision | pairs (adversarial + random) | coincident | ŷ, d ≠ host | rule holds, all five formulations | smallest nonzero \|d\| | smallest nonzero r² | largest r² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Metal f32 | 209,303 + 216,000 | 30,403 + 3,526 | 0, 0 | yes (0 violations) | 2⁻⁵³ | 2⁻¹⁰⁶ | 72.65 |
| CPU f32 | 209,303 + 216,000 | 30,403 + 3,526 | 0, 0 | yes | 2⁻⁵³ | 2⁻¹⁰⁶ | 72.65 |
| CPU f64 | 439,192 + 216,000 | 484 + 3,526 | 0, 0 | yes | 2⁻⁵³ | 2⁻¹⁰⁶ | 72.65 |

- The extremes are those of check_p2p_domain.py: smallest nonzero |d| 2⁻⁵³, r² 2⁻¹⁰⁶,
  both attained.
- The largest r² of the random pairs is 72.65, below 2⁷.
- In f32, 30,385 of the adversarial coincidences come from distinct doubles that round
  to the same f32 (§3.13, "Coincident pairs", first consequence).
- Reassociation would show: computed on the host, (u_t − ĉ) − r̂ u_s changes 164,601 f32
  and 3,118 f64 adversarial components and gains or loses a zero in 123,571 and 900. No
  device result differs from the host's, so neither backend reassociated d.

## P2P candidates

Pair terms over 10⁶ seeded pairs spanning the domain (the f32 gradient over
r² ≥ 2⁻⁸⁴), in u_T against `nd_fmm_ref::p2p`. Sums on W1 in leaf-scaled form against
`direct_sum`, relative to the term magnitudes, worst over n_t ∈ {8, 32, 64, 128}; the
reference's own error is in brackets.

| backend | candidate | φ term | ∇φ term | φ, ∇φ with nonzero \|dₖ\| ≥ 2⁻⁵³ | φ sum | ∇φ sum |
| --- | --- | --- | --- | --- | --- | --- |
| Metal f32 | sqrt + div / inverse_sqrt, as written | 3.974 | 8.629 | 3.974, 8.629 | 1.16e-7 [1.28e-7] | 5.93e-7 [6.23e-7] |
| Metal f32 | sqrt + div / inverse_sqrt, explicit fma | 6.606 | 8.621 | 3.974, 8.621 | 1.21e-7 | 5.93e-7 |
| Metal f32 | + Newton, as written / explicit fma | 3.974 / 7.928 | 9.124 / 8.905 | 3.974, 9.124 / 3.974, 8.905 | 1.17e-7 | 5.17e-7 |
| CPU f32 | sqrt + div / inverse_sqrt, as written / explicit fma | 3.997 / 3.986 | 10.296 / 10.210 | same | 1.28e-7 / 1.23e-7 [1.28e-7] | 5.93e-7 [6.23e-7] |
| CPU f32 | + Newton, as written / explicit fma | 4.321 / 3.974 | 9.296 / 8.905 | same | 1.28e-7 / 1.11e-7 | 5.41e-7 |
| CPU f64 | sqrt + div / inverse_sqrt, as written / explicit fma | 3.998 / 4.054 | 10.195 / 9.657 | same | 2.86e-16 / 2.42e-16 [3.08e-16] | 1.89e-15 [1.75e-15] |
| CPU f64 | + Newton, as written / explicit fma | 4.606 / 4.000 | 9.296 / 9.268 | same | 2.86e-16 / 2.31e-16 | 1.89e-15 / 1.46e-15 |

- `sqrt` + division and `inverse_sqrt` give identical bits on both backends: Metal
  rewrites one into the other, and the CPU runtime polyfills the other way.
- Every candidate passes the sum check (`p2p_kernels::passes`).

Metal throughput, quick ranking (Gpairs/s), with a naive GPU-shaped kernel: one unit per
target, sources read from global memory and mapped on the fly, 64 units per cube, 4,096
target leaves per launch.

| candidate | φ n_t = 8 / 32 / 64 / 128 | φ, ∇φ n_t = 8 / 32 / 64 / 128 |
| --- | --- | --- |
| sqrt + div, as written | 185 / 236 / 250 / 257 | 154 / 200 / 212 / 219 |
| inverse_sqrt, as written | 184 / 235 / 248 / 250 | 159 / 194 / 200 / 203 |
| inverse_sqrt, explicit fma | 178 / 222 / 229 / 231 | 147 / 182 / 189 / 192 |
| inverse_sqrt + Newton, explicit fma | 162 / 196 / 206 / 209 | 133 / 163 / 170 / 174 |

(Every row is in `results-m3max.md`.)
- At n_t = 128, 257 Gpairs/s is 36% of the 720 Gpairs/s model of device-path.md §13.4,
  with no shared-memory tiling. T6 times the production kernel.
- The differences between the forms that compile to the same arithmetic (5–15%) are at
  the level of the expected run-to-run variance.

## CPU-shaped P2P against `nd-fmm-simd`

Kernel `cpu_p2p::cpu_p2p_kernel`:
- targets in `Vector<T, N>` lanes, with N = 4 (f32) or 2 (f64), K = 2 or 4 vectors per
  block;
- sources broadcast in input order;
- `fma` for r² and the terms, `sqrt` and division, ρ cleared by `select_many` where
  r² = 0;
- no shared memory, no `sync_cube`; one cube whose units take contiguous shares of the
  target leaves.

Inputs: W1 gathered, the 64-set pool cycled to 768 target leaves per launch.
- One thread: one unit, against `P2pKernel::detect()` (NEON) called on one thread.
- All cores: 12 units, against it on 12 rayon threads over the same leaves.

Every timed cell first passed the `direct_sum` check at 1 and 12 units. Pair terms over
10⁶ domain pairs: f32 φ 3.995, ∇φ 9.252 u_T; f64 φ 3.987, ∇φ 9.484 u_T.

| precision | output | n_t | CPU runtime 1 unit (Gpairs/s) | nd-fmm-simd 1 thread | ratio | CPU runtime 12 units | nd-fmm-simd 12 threads | ratio |
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

Geometric mean over the 24 cells: **one thread 1.004, all performance cores 2.729**.

Bit identity with `nd-fmm-simd` (NEON) on the checked sets of every cell:
- every potential output is identical (φ-only cells: all of them);
- of the cells with gradients, about 77% of all outputs are identical (4,456 of 5,440 in
  f32 and 4,455 of 5,440 in f64, potentials included);
- the gradient's operation order differs from the NEON kernel's in some lanes. Not
  pursued.

Per-launch overhead (f32 φ; queued: launches back to back with one sync per batch;
latency: one launch and a sync):

| launch | units | queued (µs per launch) | launch + sync (µs) |
| --- | --- | --- | --- |
| empty (no leaves) | 1 | 2.4 | 9.2 |
| empty (no leaves) | 12 | 190.9 | 27.5 |
| one target leaf, n_t = 64 | 1 | 32.0 | 43.6 |
| one target leaf, n_t = 64 | 12 | 230.9 | 51.3 |

**The JIT output.** The runtime has no assembly dump. A scratch crate outside the
workspace, with `cubecl/pliron-dump` (which cannot be enabled inside it; T2), ran the same
kernel source with `CUBECL_DEBUG_PLIRON`. The dumped `llvm.opt.ll` was lowered with
Homebrew's LLVM 22.1.8 `llc -O3 -mcpu=apple-m3`, after the LLVM 23 attribute
`target_mem: none` was removed. This approximates the JIT's code generation (the same IR,
the host CPU), but it is not the JIT's own object code.

Inner loop per source:

| kernel | instructions | FP and mask ops | loads | stack references | calls | NEON spike loop (Phase 3S, `til_k*_best`): instructions / FP |
| --- | --- | --- | --- | --- | --- | --- |
| f32 φ, K = 2 | 32 | 22 (6 `fsub`, 2 `fmul`, 6 `fmla`, 2 `fsqrt`, 2 `fdiv`, 2 `fcmeq`, 2 `bic`) | 4 (3 `ld1r`, 1 `ldr`) | 0 | 0 | 26 / 22 |
| f32 φ, ∇φ, K = 2 | 46 | 36 | 4 | 0 | 0 | 40 / 36 |
| f64 φ, K = 4 | 54 | 44 | 4 | 0 | 0 | 48 / 44 |
| f64 φ, ∇φ, K = 4 | 86 | 72 | 8 | 4 | 0 | 90 / 72 (6 spills) |

- The vector width is the kernel's (4 × f32, 2 × f64).
- `fsqrt` and `fdiv` are inline vector instructions, and the masks are `fcmeq` + `bic`.
- The FP work is exactly the NEON kernel's. The extra 6 instructions per source are
  loads and address arithmetic: three `ld1r` broadcasts and a scalar charge load
  against one structured load, and a `madd` per source.
- The FP pipes and the divider are the bottleneck, so these cost nothing measurable
  (ratio 1.00).

Where the multi-core time goes, should a later CubeCL version be re-checked with
`--sections cpu-p2p`: launch scheduling (190–230 µs per queued 12-unit launch), units landing
on efficiency cores, and a launch waiting for its slowest unit. It is not in the
generated code.

## Leaf operators and GEMM

Harmonics: one unit per point, the `nd-fmm-math` recursion ported operation by operation,
1,000 points (regular in [−1, 1]³, irregular at |x| ∈ [2, 11]). Error per point and
degree relative to the largest |value| of that degree, in u_T.

| backend | precision | test | bit-identical to host | device − host | where every value is normal | device − f64 | host − f64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Metal | f32 | regular, p = 8 | 34,530 of 81,000 | 48.4 | 48.4 | 27.8 | 44.0 |
| Metal | f32 | irregular, p = 8 | 20,387 of 81,000 | 60.4 | 60.4 | 40.6 | 69.8 |
| Metal | f32 | regular, p = 20 | 88,943 of 441,000 | **1.19e6** | 273.6 (351 host values subnormal) | 1.19e6 | 143.2 |
| Metal | f32 | irregular, p = 20 | 66,689 of 441,000 | 267.9 | 267.9 | 317.4 | 243.2 |
| CPU | f32 | regular / irregular, p = 8 | 38,963 / 32,955 of 81,000 | 34.8 / 74.9 | same | 24.9 / 36.8 | 44.0 / 69.8 |
| CPU | f32 | regular / irregular, p = 20 | 115,132 / 105,338 of 441,000 | 216.5 / 306.2 | same | 118.1 / 202.1 | 143.2 / 243.2 |
| CPU | f64 | regular / irregular, p = 8 | 39,240 / 32,911 of 81,000 | 40.2 / 65.3 | same | (= device − host) | 0 |
| CPU | f64 | regular / irregular, p = 20 | 117,006 / 105,935 of 441,000 | 106.6 / 213.6 | same | (= device − host) | 0 |

- **Contraction moves the recursion by rounding noise.** The device is as accurate as
  the host against f64, or better (CPU f32: 118 against 143 u_T at p = 20). In f64 the
  device and host differ by at most 214 u_T = 2.4e-14 per degree at p = 20, inside the
  1e-13 leaf-operator bound, though not by much. T7 compares with tolerances.
- **Flushing does harm** where values leave the f32 normal range: regular harmonics of
  high degree at small |x| on Metal. Within the tested f32 range (p ≤ 8, §3.9) no value
  is subnormal, and Metal is within 60 u_T (3.6e-6) of the host, inside the 1e-5 f32
  bound.

GEMM y ← y + A x, n = 81 (p = 8), 512 columns, one unit per column, in
`MatrixSet::apply`'s order:

| backend | precision | = `MatrixSet::apply` | = host loop with `mul_add` in the same order | max \|device − apply\| / max \|y\| |
| --- | --- | --- | --- | --- |
| Metal | f32 | 10,724 of 41,472 | **41,472 of 41,472** | 3.49 u_T |
| CPU | f32 | 10,724 of 41,472 | **41,472 of 41,472** | 3.49 u_T |
| CPU | f64 | 10,524 of 41,472 | **41,472 of 41,472** | 3.17 u_T |

So the GEMM is contracted exactly as a host `mul_add` loop is, and not reassociated (not
even on Metal), far inside the C4.4/C4.5 tolerances.

## Other findings

- `cubecl-metal` 0.11.0-pre.4 does not compile (`DebugInformation` is not imported in
  `src/compute/context.rs:139`). Any later use of `metal-native` (decision 11) needs a
  CubeCL release that fixes it.
- Metal's `recip(x)` and `1 / x` are the same instruction sequence (1.29 u_T). A division
  a / x is not the correctly rounded quotient (2.26 u_T).
- The pre-release's facts F16 and F17 of device-path.md §1.2 stand as read. One detail
  to add: explicit `fma` lowers to `llvm.fmuladd`, which permits but does not require
  fusion. It fuses on AArch64. On an x86_64 CPU without FMA (the CI runner has it), the
  CPU runtime's "fma" would round twice. T4's CI job should print the CPU features.

## Reproducing

```sh
# Smoke test (CPU runtime, reduced sizes; seconds once built)
cargo test -p nd-fmm-spike-device-arith --features cpu
# Full run (outside the sandbox: Metal needs GPU access)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
    cargo run --release -p nd-fmm-spike-device-arith --features cpu,metal -- \
    --backends metal,cpu > spikes/device-arith/results-m3max.md
# One section, quick: --sections primitives,compiler,domain,p2p,cpu-p2p,leafops --quick
# CUDA, type-check only
cargo check -p nd-fmm-spike-device-arith --no-default-features --features cuda
```

The JIT dump needs a scratch crate outside the workspace with
`cubecl = { version = "=0.11.0-pre.4", features = ["cpu", "pliron-dump"] }`, the kernel of
`src/cpu_p2p.rs`, `CUBECL_DEBUG_PLIRON=<dir>`, and `llc` from an LLVM ≥ 22.

## Files

- `src/primitives.rs`: primitive accuracy and edge cases.
- `src/compiler.rs`: compiler probes.
- `src/geometry.rs`, `src/pairs.rs`: the §3.13 pairs.
- `src/domain.rs`: the device check of the r² rule.
- `src/p2p.rs`: P2P candidates (pair terms, sums, Metal rate).
- `src/cpu_p2p.rs`: the CPU-shaped kernel and the comparison with `nd-fmm-simd`.
- `src/leafops.rs`: harmonics and GEMM.
- `src/main.rs`: the report driver.
- `src/lib.rs`: the smoke test.
- `results-m3max.md`: the raw output of the full run.

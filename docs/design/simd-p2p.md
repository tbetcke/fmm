# SIMD P2P on the host: design for Phase 3S

As of 2026-10-02. Proposed with [docs/phase3s/README.md](../phase3s/README.md), for
sign-off by hand before Phase 3S T4 starts. Decisions marked *provisional* are confirmed
or overturned by the spike (T2) and recorded here when it is signed off.

Decided on 2026-10-02:
- No x86_64 machine is available for timings. Every timing is NEON on the Apple M3 Max.
  The x86_64 paths are tested for correctness and accuracy on real hardware in CI only,
  and judged for speed by their inner-loop instruction counts against the model of
  Section 4.6.
- CI gains a job for `nd-fmm-simd` on an arm64 and an x86_64 runner (Section 7).
- T7 adopts its own leaf-size recommendation.
- AVX-512 is deferred until hardware to test and time it is available. Phase 3S ships
  NEON, AVX2 + FMA and scalar. What an AVX-512 path would use is kept, marked
  *deferred*, in Sections 4.3 and 4.6 and summarised in Section 4.7.

> Where this document and `docs/CONVENTIONS.md` differ, **the conventions file takes
> precedence.** Phase 3S changes no convention except the addition to §3.13 that T1
> drafts (Section 4.4).

Recommendation in one paragraph: write the near-field kernel of the Laplace FMM by hand
with `core::arch` intrinsics for aarch64 NEON and for x86_64 AVX2 + FMA,
in a new MPI-free crate `nd-fmm-simd`, with a scalar fallback and runtime dispatch.
Vectorise over **targets**: each lane holds one target, each source is broadcast, and
every target adds its sources in input order, exactly as `nd_fmm_ref::p2p` does. Compute
1/r from the hardware reciprocal-square-root estimate and refine it with Newton steps
(or an equivalent polynomial correction) until the result is within 4 units of roundoff,
and exclude coincident pairs by r² = 0. The kernel keeps the signature of
`nd_fmm_ref::p2p`, so `LaplaceOperator` swaps it in without touching the leaf layout of
CONVENTIONS §3.13. It is benchmarked against the Laplace kernels of
[green-kernels](https://github.com/bempp/green-kernels) on the same machine (NEON),
workloads and accuracy measures.

## 1. Purpose and scope

Phase 3 delivered a complete, threaded host FMM whose near field is the reference
operator `nd_fmm_ref::p2p`. T12 measured where the time goes on one thread
(design §7, Phase 3, "Per-pair cost"):

- At p = 3 the leaf stage, dominated by P2P, takes 72–81% of an evaluation, on every
  distribution and in both precisions.
- At p = 8 it is still 20–51%: 20% on the uniform cube in f64, 51% on the Plummer
  sphere in f32.
- On the uniform cube at p = 3 (N = 10⁵, 4,096 leaves of 8–44 points, about 5.8 × 10⁷
  near pairs) the 158 ms leaf stage works out at about 2.6 ns, or 10 cycles, per pair
  in f64 with gradients. This is an estimate from the stage timing; T2 measures the
  kernel alone.

The reference loop is scalar by construction. It skips coincident pairs with a branch,
uses a square root and divisions, and adds the terms of each target in order. LLVM
cannot vectorise that sum without reassociating floating-point additions, which it
never does without fast-math. Phase 3 also documented that the GPU tests of Phase 4
"cannot speed up the low-p runs" with a faster M2L alone (design §7, Phase 3,
"Recommendation for Phase 4"). The same holds on the host.

**In scope**

- `nd-fmm-simd`: P2P for the Laplace kernel 1/|x − y| (CONVENTIONS §3.1), potential and
  gradient, f32 and f64, on aarch64 NEON and x86_64 AVX2 + FMA, plus a scalar fallback
  and runtime dispatch.
- The inverse square root those kernels need, with a stated accuracy.
- Use of the kernel in `nd-fmm-exec`'s `LaplaceOperator`, selectable, with the
  reference path kept as a choice.
- Benchmarks against `nd_fmm_ref::p2p` and against green-kernels, and the effect on the
  complete FMM, including a new look at the leaf size.

**Out of scope**

- SIMD for the other leaf operators (P2M, L2P, P2L, M2P) and for the translations. At
  p ≥ 8 the far field dominates, and its host speed-up is the batched GEMM path of
  Phase 4, not hand-written vector code.
- Portable SIMD libraries (`pulp`, `wide`, `std::simd`, which is unstable) and C or
  assembly sources. Only `core::arch` intrinsics are used, behind the crate's own thin
  internal layer (Section 5.3).
- AVX-512F, deferred until hardware is available (Section 4.7). An AVX-512 machine
  runs the AVX2 path meanwhile.
- SVE and SVE2 (Graviton 3/4, Grace, A64FX). Their Rust intrinsics are not stable in the
  pinned toolchain; NEON runs on every aarch64 CPU. AVX-512 FP16, BF16 and the Xeon Phi
  extensions (`avx512er`) are out too.
- Threading inside the kernel. The host path already threads over target leaves
  (C3.5); the kernel runs on whichever thread calls it.
- Other kernels (Helmholtz, Yukawa) and other source types (dipoles, C6.4). The crate
  is laid out so that they can be added later.
- Changes to the leaf layout of CONVENTIONS §3.13 (Section 4.5 explains why none is
  needed).

## 2. Starting point

### 2.1 The Phase 3 P2P path

- `LaplaceOperator::p2p_target` walks the near list of a target leaf (U list and the
  leaf itself, by leaf index) and calls `Kernels::p2p` once per source leaf.
- For a source leaf s ≠ t it first maps the sources into scratch,
  ŷ = ĉ(s|t) + r̂(s|t) u_s, in the target's leaf-scaled coordinates (CONVENTIONS §3.13,
  "Operators in scaled coordinates"). For s = t it passes the stored chunk unchanged.
- It then calls `nd_fmm_ref::p2p::p2p(sources: &[[T; 3]], charges: &[T],
  targets: &[[T; 3]], potential: &mut [T], gradient: Option<&mut [[T; 3]]>)`, which adds
  into the target output.
- Layouts (§3.13): a source chunk holds the n coordinate triples, then the n charges.
  Target input is n triples. Target output holds n potentials, then, with gradients,
  n triples.
- Order and identity: each target adds its source leaves in the order of its row and
  each leaf's sources in chunk order. The output is bit-identical for any number of
  threads (C3.5), and the per-pair adapter (`PerPair<LaplaceOperator>`) is
  bit-identical to the batched path (C3.1). Phase 3S keeps both properties.
- Coincident pairs: a pair with x == y exactly contributes nothing (§3.13, "Coincident
  pairs").

### 2.2 green-kernels

Read at commit `7d757c579f8633d58163cd5b2d011a9468541f17` (2025-11-23) of
`bempp/green-kernels`. The library is not on crates.io at that version: the crates.io
release 0.2.2 dates from 2024 and uses pulp 0.18 and rlst 0.2. Its `rlst` dependency is
rlst's git `main`. The SIMD primitives below are in `rlst::simd` (`src/simd.rs`), checked
in rlst 0.9.0; T2 checks that the git `main` it builds against has the same code.

- `Laplace3dKernel::<T>::evaluate_st(eval_type, sources, targets, charges, result)`
  takes sources and targets as interleaved triples. `GreenKernelEvalType::Value` returns
  one value per target; `ValueDeriv` returns four interleaved values (φ and the
  gradient). Results are added into `result` and include 1/(4π).
- **Loop order: sources in lanes.** For each target, `evaluate_laplace_one_target`
  broadcasts the target and vectorises over the sources: it deinterleaves a vector of
  sources, accumulates per lane, reduces horizontally at the end and handles the
  remaining sources with a scalar tail (`pulp::Scalar`). `pulp::Arch::new().dispatch`
  runs once per target.
- **Coincident pairs:** `select(r² == 0, 0, rsqrt(r²))`.
- **Inverse square root** (`approx_recip_sqrt`):

  | Backend | f32 | f64 |
  | --- | --- | --- |
  | x86 V3 (AVX2 + FMA) | `rsqrt_ps` (12 bits), 1 Newton step | r² converted to f32, `rsqrt_ps`, back to f64, 2 Newton steps (the rlst test asks for 1e-13) |
  | x86 V4 (AVX-512) | `rsqrt14_ps`, **no** Newton step (14 bits); only with pulp's `nightly` feature | no V4 branch: per-lane scalar `sqrt`, then a vector division |
  | aarch64 NEON | `frsqrte` (≈ 8 bits), 2 `frsqrts` steps | `frsqrte`, 3 `frsqrts` steps |
  | other | `1 / sqrt` | `1 / sqrt` |

- **Dispatch:** pulp detects the CPU at run time. Without the `nightly` feature it
  never selects V4, so on stable Rust an AVX-512 machine runs the AVX2 code.
- Multi-threaded variants (`evaluate_mt`) use rayon's global pool.

What carries over: the r² = 0 mask, the estimate-plus-Newton scheme and its iteration
counts as a starting point, FMA throughout. What does not: the loop order (Section 4.1),
the scalar tails (Section 4.5), the f64 accuracy of the AVX2 path (Section 4.3), and the
dependency on a portable SIMD layer.

## 3. Requirements

Phase 3S is designed and accepted against these. T1 or T2 may propose changing one,
with reasons, for sign-off.

1. **Same operator.** The kernel computes what `nd_fmm_ref::p2p` computes,
   φᵢ += Σⱼ qⱼ / |xᵢ − yⱼ| and ∇φᵢ += −Σⱼ qⱼ (xᵢ − yⱼ) / |xᵢ − yⱼ|³, with no 1/(4π),
   adding into its outputs. It has the same signature.
2. **Accuracy.** Per pair, the potential term is within 8 u_T of the reference term,
   relative, and each gradient component within 16 u_T relative to the term magnitude
   |q| / r². Here u_T is the unit roundoff (2⁻²⁴ for f32, 2⁻⁵³ for f64). Sums agree
   with `direct_sum` to 1e-14 (f64) and 1e-6 (f32) relative to the sum of term
   magnitudes, for up to 4,096 sources (the tolerance of `nd_fmm_ref::p2p` itself in
   Phase 1 T4). *Decided on 2026-10-03 (T5):* where `nd_fmm_ref::p2p` itself exceeds
   that tolerance on the same inputs, the sum passes within twice the reference's
   error. In-order summation (requirement 4) puts f32 sums with gradients above 1e-6 on
   some FMM-shaped sets of 44 or more points per leaf (1,188 or more sources), for the
   reference as for the kernel; the f32 terms summed in f64 stay below 1e-7.
3. **Coincident pairs.** A pair contributes nothing exactly when r² = 0 (Section 4.4).
   On leaf-scaled data this is the rule of §3.13.
4. **Order.** Each target adds its sources in input order, starting from the value
   already in the output. Consequences, each tested bit for bit:
   - *chunk invariance:* evaluating sources `[..k]` and then `[k..]` equals evaluating
     all of them at once;
   - *target-position invariance:* a target's result does not depend on its position
     in `targets` or on how many targets there are;
   - per-pair = batched in `LaplaceOperator`, and bit-identity for every thread count,
     as in Phase 3.
5. **Deterministic.** The same inputs on the same machine and ISA give the same bits.
   Results may differ between ISAs and from the reference, within requirement 2.
6. **No allocation, no threads, no MPI** inside the kernel. It writes only into the
   caller's slices.
7. **ISAs.** NEON (f32 × 4, f64 × 2 lanes), AVX2 + FMA (8 / 4) and a scalar fallback;
   AVX-512F (16 / 8) later (Section 4.7). The best available ISA is chosen once at run
   time. Every path is selectable explicitly, for tests and benchmarks.
8. **Tested on every ISA** that a machine offers, against the reference and the
   oracle. The report of each run says which ISAs ran.
9. **Unsafe confined** to the architecture modules of `nd-fmm-simd` (Section 5.4).
   `nd-fmm-exec` stays free of `unsafe`.
10. **Measured, not asserted.** No test asserts a throughput. Examples and the spike
    report it, against `nd_fmm_ref::p2p` and green-kernels, together with accuracy.

## 4. Kernel design

### 4.1 Loop order: targets in lanes (provisional)

The kernel holds W targets per vector (W the lane count) and K vectors per block, and
loops over all sources:

```text
for each block of K·W targets:                       // K vectors of W targets
    load the targets (x, y, z), the potentials and, with gradients, the gradients
    for each source j in order:
        broadcast yⱼ (3 values) and qⱼ                // NEON ld3r + ld1r; x86 vbroadcast
        for each of the K vectors:                    // independent chains
            d = x − yⱼ;  r² = dx·dx + dy·dy + dz·dz   // 3 sub, mul + 2 fma
            ρ = rsqrt(r²), with ρ = 0 where r² = 0    // Section 4.3, 4.4
            φ += qⱼ ρ;  g −= (qⱼ ρ³) d                // with gradients
    store the potentials and gradients of the block
```

Why this order rather than green-kernels' sources-in-lanes:

- Each lane adds one target's terms in source order, which is requirement 4. Chunk
  invariance makes per-pair calls and gathered calls bit-identical, so the Phase 3
  identity between `PerPair` and the batched path survives. With sources in lanes, a
  target's sum is split into W partial sums and a horizontal reduction, so the result
  depends on where the source stream is cut, and that identity would have to become a
  tolerance.
- The accumulators stay in registers for the whole source loop. No horizontal
  reductions are needed, and no source tails: sources are broadcast one at a time.
- Sources are read in the §3.13 layout as they are (triples, then charges), and so are
  targets and gradients. Only the load and the store of a block convert between
  interleaved and per-lane form (NEON `ld3`/`st3`; x86 shuffles or scalar inserts),
  once per block and call.

Costs of this order:

- Lanes are wasted when a leaf's target count is not a multiple of W. Uniform leaves
  hold 8–44 points (24 on average) at N = 10⁵, and the adaptive trees up to 64. AVX2 in
  f32 (W = 8) puts a leaf of 20 targets into 24 lanes; a later AVX-512 path in f32
  (W = 16) would put 24 targets into 32.
- Four broadcasts per source per block. They are cheap on x86, where a broadcast from
  memory uses only a load port, and on NEON (`ld3r`, `ld1r`).

The spike (T2) measures both orders on NEON, the only ISA that can be timed.
**Decision rule, fixed now:** targets in lanes stays unless sources in lanes is at least
15% faster, as the geometric mean over the FMM-shaped workloads of Section 8.2, in both
precisions on NEON. If so, T2 recommends the change, and adopting it needs sign-off
because it relaxes the per-pair identity of C3.1 for P2P. The chosen order is used on
every ISA. For AVX2 the spike reports the lane utilisation at the W1 leaf sizes, since
it cannot time it.

### 4.2 Arithmetic per pair

Per pair and lane, with k refinement steps of the inverse square root (Section 4.3):

| Step | Operations | Potential only | With gradient |
| --- | --- | --- | --- |
| d = x − y | 3 sub | 3 | 3 |
| r² | mul + 2 fma | 3 | 3 |
| ρ ≈ 1/√r² | estimate + k steps of 3 (x86: + 1 for ½r²) | 1 + 3k (x86: 2 + 3k) | same |
| coincident mask | compare + and-not | 2 | 2 |
| φ += q ρ | mul + add (or one fma without gradient) | 1 | 2 |
| g −= (q ρ · ρ²) d | mul (ρ²), mul, 3 fnma | — | 5 |
| **total** | | **10 + 3k** (x86 + 1) | **16 + 3k** (x86 + 1) |

The x86 f64 path that starts from an f32 estimate adds two conversions. Broadcasts,
loads and stores of the block run on other ports and are not counted.

### 4.3 The inverse square root

Each ISA offers a hardware estimate of 1/√x:

| ISA | f32 estimate | f64 estimate | relative error of the estimate |
| --- | --- | --- | --- |
| NEON | `vrsqrteq_f32` (FRSQRTE) | `vrsqrteq_f64` | ≈ 2⁻⁸ |
| AVX2 | `_mm256_rsqrt_ps` | none: convert r² to f32 and back, or an integer bit-trick guess | ≤ 1.5 · 2⁻¹² |
| AVX-512F (*deferred*) | `_mm512_rsqrt14_ps` | `_mm512_rsqrt14_pd` | < 2⁻¹⁴ |

**Refinement.** With h = ½ x and an estimate y, one Newton–Raphson step for
f(y) = y⁻² − x is

```math
y' = y + y\,(\tfrac12 - h\,y^2), \qquad e' \approx -\tfrac32\, e^2
```

for the relative error e of y. It costs a multiply (y²), an fma (½ − h y²) and an fma
(y + y r), so the rounding of the last step is about one unit. NEON's `vrsqrtsq`
computes (3 − a b)/2 fused, which gives the step as y' = y · frsqrts(x, y · y) in three
operations as well. Steps needed for full precision (relative error below u_T), from
the error bounds above:

| ISA | f32 | f64 |
| --- | --- | --- |
| NEON | 2 (2⁻⁸ → 2⁻¹⁵ → 2⁻²⁹) | 3 (→ 2⁻⁵⁷) |
| AVX2 | 1 gives ≈ 2.0e-7 = 3.4 u₃₂ before rounding, at the edge of the contract; 2 are safe | from the f32 estimate: 2 give ≈ 6e-14 (green-kernels), 3 give full precision |
| AVX-512F (*deferred*) | 1 (2⁻¹⁴ → 2⁻²⁷) | 2 (→ 2⁻⁵⁴) |

**Polynomial correction.** With r = 1 − x y² (one mul, one fma), the exact
correction is y (1 − r)^(−½) = y (1 + r/2 + 3r²/8 + 5r³/16 + 35r⁴/128 + …). Truncated
after rᵐ, its error is about the first omitted term. From the 12-bit AVX2 estimate
(|r| ≲ 7.3e-4), the terms up to r² reach 1.2e-10 in f32 and those up to r⁴ about 5e-17
in f64. That is one evaluation of 4–7 operations with a shorter dependency chain than
two or three Newton steps. It is a candidate wherever it reaches the contract more
cheaply; T2 measures it.

**Square root and division.** `sqrt` then `div` (or `1/sqrt`) rounds twice and is
within about 1.5 u_T. It is the scalar fallback and the yardstick for the estimates. On
the cores of interest its throughput is several times lower than the FMA pipes'. T2
measures it on every ISA; if it is not slower on some core, the dispatch uses it there.

**The contract** (C3S.3): on every ISA and precision, the inverse square root used by
the kernel is within 4 u_T relative, for 0 and for every r² in the domain of Section
4.4. It returns 0 at r² = 0 after the mask.

- f32 is checked exhaustively over [1, 4). The estimates and every refinement above
  commute with scaling by 4ᵏ for normal inputs, because they act on the significand
  and the exponent parity only. One period therefore covers every normal input, and
  the ends of the domain are checked separately.
- f64 is checked on at least 10⁷ seeded log-uniform samples over the domain, plus
  powers of two, their neighbours, and the domain ends.

T2 chooses, per ISA and precision, the cheapest formulation that meets the contract:
- on NEON by measured error (exhaustive in f32) and measured throughput;
- on AVX2, which cannot be timed, from the documented worst-case error of its
  estimate, with a margin of at least a factor 2 below 4 u_T after rounding, and
  by operation count and dependency-chain length. For example, AVX2 f32 then takes two
  Newton steps or the r² polynomial rather than one step. Rosetta 2 emulates the x86
  estimates, so measurements there do not count. The exhaustive and sampled tests of
  T4 confirm the choice on real x86_64 hardware in CI. T2 also
measures a relaxed level for f64 (for example green-kernels' two steps, about 1e-13).
The FMM's own error is at least 1e-8 at p ≤ 20 (C3.4), so a relaxed P2P would be
invisible in its output. Whether to ship it, as an opt-in beside the full-precision
default, is a sign-off question of T2, answered with the measured gain.

### 4.4 Coincident pairs and the domain of r²

The reference excludes a pair when xᵢ == yⱼ in all three components. The kernel excludes
it when r² = 0, which costs one compare and one and-not per pair instead of three
compares and two ands. The two rules differ only when x ≠ y but every dₖ² underflows to
zero, and there the reference returns a non-finite value.

On the leaf-scaled data of §3.13 that case does not arise. A stored coordinate is
u = fl(fl(d · 2^(l+1)/w) − (2i + 1)), rounded to T. Its significand lies on a grid no
finer than that of the operands, which are about 1 in size, and a mapped source
ŷ = ĉ + r̂ u keeps a comparable grid, because ĉ is a dyadic rational of at most 17 bits
and r̂ a power of two. A nonzero component of d is therefore bounded below by a small
power of two, about 2⁻⁶⁰ for leaf-scaled data, and r² is bounded below by about 2⁻¹²⁰,
which is normal in f32 (2⁻¹²⁶). T1 derives the exact bound and checks it with
adversarial constructions.

T1 drafts an addition to CONVENTIONS §3.13, "Coincident pairs", for sign-off:

- the fast kernels exclude a pair by r² = 0;
- on leaf-scaled data this equals the exact-coincidence rule, with the derivation of
  the bound;
- the kernel domain: r² = 0 or r² ∈ [2^(−120), 2^(120)] (or the bound T1 finds). Pairs
  outside it are outside the kernel's contract, whatever the reference does there.
  Within the FMM, |u| ≤ 1 + β and |ŷ| ≤ 5 under 2:1 balance, so r² < 2⁷.

Masking happens after the estimate. At r² = 0 the estimate is +∞ and the Newton step
produces NaN; the bitwise and-not with the mask clears it. No floating-point exception
is trapped (Rust never unmasks them). Subnormal r² never reaches the estimate
instructions, which may flush it to zero (x86 `rsqrtps` does).

### 4.5 Layout, blocks and tails

- **No layout change.** The kernel reads sources as `&[[T; 3]]` plus `&[T]` charges,
  as stored, and broadcasts them. A structure-of-arrays chunk layout (all x, then all
  y, …) would suit only the sources-in-lanes order, and would cost P2M, P2L and the
  exchange a transposition, since `nd_fmm_ref::leaf` takes `[[T; 3]]`. If T2 chooses
  sources in lanes, the kernel transposes into its own scratch: O(n) against
  O(n_s n_t).
- **Blocks.** Targets are loaded per block and converted to per-lane form, and so are
  the potentials and gradients, which are added into in place. On NEON `vld3q`/`vst3q`
  do it directly. On x86 any shuffle sequence will do, since it runs once per block and
  call.
- **Tails.** The last block of a call, with fewer than K·W targets, uses the same vector
  code on a padded stack copy. Padding lanes repeat the last target, so they compute
  finite values that are discarded. AVX2 may use masked loads and stores
  (`_mm256_maskload`, `_mm256_maskstore`) instead. The tail never falls back to scalar
  code, because target-position invariance (requirement 4) needs every target to see the
  same instruction sequence.
- **Register blocking.** K ∈ {1, 2, 4} independent vectors per source hide the latency
  of the estimate and the Newton chain (about 20–40 cycles) and reuse each broadcast K
  times. The register budget limits K: NEON has 32 vector registers, AVX2 16; with
  gradients a vector of targets needs 7 live registers (x, y, z, φ, three gradient
  components). T2 measures K per ISA and precision. The kernel takes K as a const
  generic, so a choice per ISA costs no code.

### 4.6 Throughput model

An upper bound per core, from Section 4.2, assuming every vector operation issues on one
of the FP pipes and nothing else limits. **These are models, not measurements**. T2 and
T7 measure the NEON row; the AVX2 row stays a model, checked only through the
instruction counts of the compiled loops (T5). The AVX-512 row is for the deferred path
(Section 4.7).

| Core | pipes × lanes (f32 / f64) | f32, k: pairs/cycle, with gradient | f64, k: pairs/cycle, with gradient |
| --- | --- | --- | --- |
| Apple M3 Max P-core, NEON | 4 × 4 / 4 × 2 | k = 2: 16/22 = 0.73 | k = 3: 8/25 = 0.32 |
| AVX2 (2 FMA pipes, Zen 3 or Skylake class) | 2 × 8 / 2 × 4 | k = 1: 16/20 = 0.80 | k = 3, + 2 conversions: 8/28 = 0.29 |
| AVX-512F, *deferred* (2 × 512-bit, Ice Lake or Sapphire Rapids class; Zen 4 splits into halves) | 2 × 16 / 2 × 8 | k = 1: 32/20 = 1.60 | k = 2: 16/23 = 0.70 |

Against the estimated 10 cycles per pair of the Phase 3 path on the M3 Max (Section 1),
the model allows about 3× in f64 and 7× in f32 at full efficiency. On the uniform cube
at p = 3, where the leaf stage is 158 of 199 ms, a 2.5× faster P2P would bring an
evaluation to about 105 ms.

### 4.7 AVX-512, deferred

AVX-512F is left out of Phase 3S (decided on 2026-10-02), because no hardware is
available to test or time it, and GitHub's x86_64 runners typically lack it. What a
later task needs, already recorded above:

- the estimates `_mm512_rsqrt14_ps` and `_mm512_rsqrt14_pd` (< 2⁻¹⁴; Intel documents a
  bit-exact reference), so f64 needs no f32 detour: one Newton step for f32, two for
  f64 (Section 4.3);
- 16 / 8 lanes, 32 registers, native masks for tails and for the r² = 0 mask;
- the model of Section 4.6: about 2× the AVX2 bound per core in f32 and 2.4× in f64,
  less on Zen 4, which splits 512-bit operations;
- the worst lane waste of all ISAs at the leaf sizes of the FMM (Section 4.1).

Adding it is one module `arch::avx512` with the vector layer and its inverse square
root, a new `Isa::Avx512` (the enum is `#[non_exhaustive]`, so this is not a breaking
change), detection of `avx512f`, and its tests in a CI or local run on hardware that has
it. Until then an AVX-512 machine runs the AVX2 path. green-kernels on stable Rust does
the same (Section 2.2).

## 5. The crate `nd-fmm-simd`

Directory `fmm-simd/`, package `nd-fmm-simd`, library `nd_fmm_simd`, a default member.
It depends only on `nd-fmm-math` (for `RealScalar`) and `thiserror`; `nd-fmm-ref` and
`proptest` are dev-dependencies. It is MPI-free, so `cargo test -p nd-fmm-simd` needs no
MPI runtime.

### 5.1 Public surface (planned)

```rust
/// Instruction sets with a P2P path. `Avx2` means AVX2 and FMA. AVX-512 is deferred
/// (Section 4.7); `non_exhaustive` lets it be added without a breaking change.
#[non_exhaustive]
pub enum Isa { Scalar, Neon, Avx2 }

impl Isa {
    pub fn detect() -> Isa;                        // the widest available
    pub fn is_available(self) -> bool;
    pub fn available() -> impl Iterator<Item = Isa>;
    pub fn lanes<T: SimdScalar>(self) -> usize;
}

/// f32 and f64; sealed.
pub trait SimdScalar: RealScalar + sealed::Sealed {}

/// A P2P kernel bound to one ISA; `Copy`, `Send` and `Sync`.
pub struct P2pKernel<T: SimdScalar> { /* isa */ }

impl<T: SimdScalar> P2pKernel<T> {
    pub fn new(isa: Isa) -> Result<Self, IsaUnavailable>;
    pub fn detect() -> Self;
    pub fn isa(&self) -> Isa;
    /// The signature and semantics of `nd_fmm_ref::p2p::p2p` (requirements 1–4).
    pub fn evaluate(&self, sources: &[[T; 3]], charges: &[T], targets: &[[T; 3]],
                    potential: &mut [T], gradient: Option<&mut [[T; 3]]>);
}

/// The inverse square root of the kernel, for tests and reports.
pub mod rsqrt { pub fn rsqrt_slice<T: SimdScalar>(isa: Isa, x: &[T], out: &mut [T]); }
```

`evaluate` panics on mismatched lengths, like the reference. Names may change in T3;
the semantics may not.

### 5.2 Detection and dispatch

- aarch64: NEON is part of the base architecture, so `Isa::Neon` is always available
  and needs no detection.
- x86_64: `std::arch::is_x86_feature_detected!` for `avx2` and `fma` (`Isa::Avx2`),
  once at construction. The result is cached by `std`.
- `Isa::Scalar` is available everywhere. It is a plain loop with `sqrt` and division
  and the r² = 0 rule, written for clarity, not speed.
- `P2pKernel::new(isa)` fails for an ISA the CPU lacks. That check is what makes the
  `unsafe` call of a `#[target_feature]` entry point sound (Section 5.4).
- `evaluate` matches on the stored ISA and calls one entry point per (ISA, precision,
  gradient or not). The branch runs once per call and is perfectly predicted.
- `Isa::detect()` picks AVX2, else scalar, on x86_64, and NEON on aarch64. When
  AVX-512 is added, whether it should outrank AVX2 is decided by measurement then.
- The library never reads an environment variable to choose an ISA.
  `FmmBuilder::p2p_kernel` (Section 6) is the only override.

### 5.3 Code structure

- `arch::{neon, avx2, scalar}`, each behind `#[cfg(target_arch = …)]`. A module
  implements a small internal trait over its vector types: splat, load and store,
  add, sub, mul, fma, fnma, compare-equal, and-not, the inverse-square-root estimate,
  and the block load and store. All methods are `#[inline(always)]`.
- The kernel body (Section 4.1) is written once, generic over that trait and over K,
  and instantiated inside one `#[target_feature(enable = "…")]` function per ISA and
  precision. The intrinsics then inline into a function that has the features.
- The inverse square root is written per ISA. It is the only part that differs in
  substance between ISAs.
- A crate-private trait, sealed over f32 and f64, selects the vector type per
  precision.
- **Inlining check.** A missing `#[inline(always)]` turns every intrinsic into an
  out-of-line call, and the kernel runs many times slower while still passing every
  test. T5 inspects the release disassembly of each entry point's inner loop (for
  example `cargo objdump` or `objdump -d` on the test binary). It reports the
  instruction count and that the loop contains no call.
- Toolchain: stable Rust 1.98 (root `CLAUDE.md`). Safe `#[target_feature]` functions
  (Rust 1.86) and safe intrinsics inside functions with the matching features (1.87)
  are both stable; so are the AVX-512 target features (1.89), for the later path. T3
  confirms each on the pinned toolchain and uses them where they remove an `unsafe` block.

### 5.4 Unsafe policy

The workspace rule (workspace-structure §5.1) allows `unsafe` only in `nd-fmm-kernels`.
Phase 3S proposes a second, narrow exception:

- `unsafe` appears only in `nd-fmm-simd`'s `arch` modules and in the dispatch that calls
  `#[target_feature]` entry points.
- The remaining uses are calling an entry point after `P2pKernel::new` has checked the
  ISA, and unaligned vector loads and stores from slices whose bounds the code has
  checked, for example with `chunks_exact`, `as_chunks` or explicit length
  assertions.
- Every `unsafe` block carries a `// SAFETY:` comment. The crate sets
  `#![deny(unsafe_op_in_unsafe_fn)]` and `#![deny(clippy::undocumented_unsafe_blocks)]`.
- Every public function is safe. `nd-fmm-exec` and every other crate stay free of
  `unsafe`.
- The spike (T2) is exempt as throwaway code, but still comments each block.

### 5.5 Reproducibility

- On one machine, one ISA and one build, results are bit-identical from run to run,
  for every thread count of `Fmm`, and between `PerPair` and the batched path.
- Across ISAs they are not: estimates, step counts and lane-wise fma contraction
  differ. The results stay within requirement 2 of each other.
- Across CPUs with the same ISA:
  - NEON's FRSQRTE is defined by the architecture's pseudocode, so every aarch64 CPU
    gives the same estimate.
  - The 12-bit `rsqrtps` estimate of AVX2 is not architecturally defined, and Intel
    and AMD CPUs return different values. On that path results can differ between
    vendors in the last bits. The `sqrt` and division path would not.
  - The repository requires bit-identity only on one machine (C3.5), so this is
    recorded, not fixed. T7 notes the CPUs measured.

## 6. Integration into `nd-fmm-exec`

- `enum P2pChoice { Auto, Reference, Isa(Isa) }` (name final in T6):
  - `Auto`, the default: `P2pKernel::detect()`;
  - `Reference`: `nd_fmm_ref::p2p::p2p`, the Phase 3 path, kept as the trusted slower
    path and for comparisons;
  - `Isa(isa)`: that ISA, or a `SettingsError` on every rank if the CPU lacks it. The
    check is local, and on a cluster with mixed CPUs ranks can disagree, so it goes
    through the same agreement (one all-reduce) that `build` already applies to every
    input error. No new collective is added.
- `FmmBuilder::p2p_kernel(choice)` and `LaplaceOperator::with_p2p(choice)` set it.
  `Fmm::p2p_kernel()` reports the resolved choice, and the `nd-fmm-validate` examples
  print it next to the `ThreadingReport`.
- `LaplaceOperator<T>` gains the bound `T: SimdScalar`. f32 and f64 are the only
  `RealScalar` types, so no caller changes.
- `Kernels::p2p` calls `P2pKernel::evaluate` where it called `nd_fmm_ref::p2p::p2p`.
  The mapping of s ≠ t sources into scratch stays as it is. Calls stay per source leaf.
  Gathering a target's whole near field into one call is measured in T6 (Section 8.2,
  per-pair against gathered form), and adopted only if it gains more than 5%. Thanks to
  chunk invariance the choice does not change a bit of the output.
- Threading is unchanged: rayon over target leaves, one kernel call at a time per
  thread. The kernel is stateless, so no per-thread state is added.
- No BLAS is involved, so the "Threads and BLAS" rule (design §6.8) is unaffected.

## 7. Testing

Every fast path is tested against a slower trusted one:

| Layer | Check | Oracle | Where |
| --- | --- | --- | --- |
| Inverse square root | contract of Section 4.3 (f32 exhaustive on [1, 4), f64 sampled, edges, r² = 0) | `1/√x` in f64 (for f32 inputs), or in double-double arithmetic (for f64 inputs) | `nd-fmm-simd`, per ISA |
| Kernel terms | one source and one target, potential and gradient, requirement 2 | `nd_fmm_ref::p2p` | `nd-fmm-simd`, per ISA |
| Kernel sums | random sets, n_s ≤ 4,096, every n_t mod W, both precisions | `direct_sum` (f64, compensated) | `nd-fmm-simd`, per ISA |
| Semantics | coincident pairs (targets = sources, duplicated sources), empty inputs, accumulation onto nonzero output, chunk and target-position invariance bit for bit, determinism | the rule itself; the kernel's own results | `nd-fmm-simd`, per ISA |
| Operator | P2P of `LaplaceOperator` on levels 2, 9 and 16 to 1e-13 (terms), as in T8 | `nd_fmm_ref::p2p` at absolute frames | `nd-fmm-exec`, per ISA |
| FMM | the C3.2 and C3.3 scenarios: errors within 1% (f64) and 2% (f32) of the reference-P2P run; outputs within 1e-13 (f64) and 1e-6 (f32) relative L2 of it; gates still met; bit-identity for 1, 2, 4 and 8 threads; per-pair = batched | the Phase 3 path (`P2pChoice::Reference`), `direct_sum` | `nd-fmm-exec` |

**Which ISAs run where.**

- *CI, the `nd-fmm-simd` job* (approved 2026-10-02; added in T3). It is a matrix over an
  x86_64 runner (`ubuntu-latest`) and an arm64 runner (`macos-latest`, or
  `ubuntu-24.04-arm` if the repository qualifies). It needs no MPI, and runs
  `cargo clippy -p nd-fmm-simd --all-targets -- -D warnings`, `cargo test -p
  nd-fmm-simd` and `cargo test -p nd-fmm-simd --release -- --ignored`, the exhaustive and
  sampled accuracy tests. The x86_64 leg is the only x86_64 hardware of the phase: it
  covers scalar and AVX2. At the time of writing the runners are typically AMD EPYC
  7763 (Zen 3). The job prints the CPU
  (`lscpu`, `sysctl -n machdep.cpu.brand_string`) and the tests print the ISAs they ran.
- *The existing CI job* (`ubuntu-latest`) also runs the `nd-fmm-simd` and `nd-fmm-exec`
  debug tests as default members, so the AVX2 path is exercised inside the FMM there.
- *Development machine* (Apple M3 Max, aarch64): scalar and NEON, run by hand in every
  task, and every timing. On macOS 15 and later, Rosetta 2 also runs x86_64 binaries
  with AVX2 and FMA, so `cargo test -p nd-fmm-simd --target
  x86_64-apple-darwin` gives a quick check before CI. Rosetta emulates the estimate
  instructions, and their bits need not match hardware, so accuracy contracts count
  only from CI. T3 confirms that this works on the machine.
- Every architecture module also passes `cargo clippy -p nd-fmm-simd --all-targets
  --target <the other architecture> -- -D warnings`, so that both compile on either
  machine.

The test binary prints the ISAs it ran, and a task report lists them. A path that did
not run is reported as not run, never as passing.

## 8. Benchmarking

### 8.1 Rules

- Timings are reported, never asserted, and taken only on the Apple M3 Max (NEON).
  CI runners never time. The harnesses build and run on x86_64 too, so that x86_64
  timings, including the green-kernels comparison, can be added later by anyone with a
  machine.
- Release builds with the default target (no `-C target-cpu=native`), because runtime
  dispatch is what ships; a second run with `native` for both libraries is optional and
  labelled.
- One thread per kernel. FMM runs at 1 thread and at the number of performance cores,
  with every BLAS variable set to 1 (design §6.8).
- Every figure is the median of 15 batches of at least 20 ms
  (`nd_fmm_validate::bench::median_time_per_call`).
- Inputs are seeded. Each report gives the machine (`bench::cpu_model`, `cores`,
  `target`), the toolchain, the ISA of every row and, for green-kernels, the commit and
  the pulp backend it dispatched to.

### 8.2 Workloads

- **W1, FMM-shaped.** A target leaf of n_t ∈ {8, 16, 24, 32, 64, 128} points uniform in
  [−1, 1]³ and its 3 × 3 × 3 block of 27 leaves of n_t sources each, in [−3, 3]³ (the
  target leaf among them, so the self pair has coincident points). A pool of 64 such
  sets is cycled, so the data come from L2 or L3 as in an FMM, not from L1. Two forms:
  - *per-pair*: 27 calls of n_t sources each, as `LaplaceOperator` calls today;
  - *gathered*: one call of 27 n_t sources.
- **W2, all-pairs.** N_s = N_t = N ∈ {10³, 10⁴}, uniform in the unit cube, with
  targets distinct from and equal to the sources. 10³ with targets equal to sources
  is green-kernels' own benchmark.
- **W3, FMM.** The T12 problems (cube and Plummer, N = 10⁵, 64 points per leaf), p = 3
  and 8, f32 and f64, with gradients: leaf stage and evaluation, `Reference` against
  `Auto`. Also the T12 leaf-size study (16–256 points per leaf), which the faster P2P
  shifts.

Each W1 and W2 cell is run with potential only and with gradients, in f32 and f64.

### 8.3 Metrics

- Pairs per second, counting n_s n_t pairs including coincident ones, and the
  ratio to green-kernels.
- Accuracy of both libraries against `direct_sum`: relative L2 and max error relative
  to the term magnitudes, so that speed is compared at stated accuracy. green-kernels'
  results are divided by 1/(4π) first.
- For green-kernels, inputs are given in its layout: interleaved triples, which are
  also our layout, and an interleaved [φ, ∇φ] output. Our gradients are converted, and
  the conversion is excluded on both sides.
- The fraction of the Section 4.6 model reached, per precision, on NEON.
- For W3, stage times, the speed-up of the leaf stage and of the evaluation, and the
  near/far balance per leaf size.

### 8.4 green-kernels in the workspace

green-kernels is used only by the spike crate `spikes/p2p-simd`, a workspace member
outside the default members, so CI never builds it. Its dependency is pinned by
commit in `[workspace.dependencies]`, with a comment that only spikes may use it.

It brings rlst from git `main` and pulp 0.22 into the root `Cargo.lock`, beside the
crates.io rlst 0.9.0 of the default members. rlst has a build script. T2 checks that the
workspace still builds and links with both (default members unchanged and both rlst
versions in one spike binary), and reports what the lock file gained. If they conflict,
T2 stops and reports, with options: patching green-kernels' rlst to 0.9.0, a separate
clone outside the workspace run on the same inputs, or vendoring the two Laplace
functions with attribution.

## 9. Risks and sign-off questions

| Risk | Mitigation |
| --- | --- |
| An ISA path is never run | report the ISAs of every run; the `nd-fmm-simd` CI job on x86_64 (AVX2) and arm64 (NEON) with the release accuracy tests |
| The x86_64 kernels are correct but slow, and nothing measures it | inner-loop instruction counts against Section 4.2 (T5); the inlining check; the same generic body as the timed NEON path; x86_64 throughput reported as not measured |
| Undefined behaviour in intrinsic code (out-of-bounds load, missing feature) | the unsafe policy of Section 5.4; bounds by construction; `P2pKernel::new` checks the ISA; tests at every n_t mod W and with empty slices on each ISA |
| An estimate is less accurate on some CPU than its documented bound, or differs between vendors | the contract is measured on every machine used, exhaustively in f32; Section 5.5 |
| Intrinsics fail to inline and the kernel is silently slow | the disassembly check of Section 5.3; throughput against the spike's prototype (T5) |
| The gain is smaller than the model (memory, broadcasts, lane waste in small leaves) | the spike measures before the production kernel is written; W1 uses realistic leaf sizes and cycles data through L2 |
| green-kernels cannot be built beside rlst 0.9.0 | spike only; the fallbacks of Section 8.4 |
| A faster P2P moves the optimal leaf size, and the default of 64 is no longer good | measured in T7 on the M3 Max, which adopts its recommendation (approved in advance); x86_64 may favour another size, which is not measured |

Questions for sign-off (Phase 3S README, "Decisions to sign off"):

1. This design, and the unsafe exception of Section 5.4 (with T1).
2. The §3.13 addition on coincident pairs and the kernel domain (T1).
3. The spike's choices: loop order, inverse-square-root formulation, K per ISA and
   precision, and whether to ship a relaxed f64 level (T2).
4. A CI job on an arm64 runner for `nd-fmm-simd` (T3). *Decided 2026-10-02: yes,
   together with an x86_64 leg.*
5. Which x86_64 machines run the x86_64 benchmarks. *Decided 2026-10-02: none; x86_64
   is correctness-only, in CI.*
6. Whether to change `max_points_per_leaf` after the leaf-size study (T7). *Decided
   2026-10-02: T7 adopts its own recommendation.*
7. AVX-512. *Decided 2026-10-02: deferred until hardware is available (Section 4.7).*

## 10. References

- green-kernels, [bempp/green-kernels](https://github.com/bempp/green-kernels), commit
  `7d757c579f8633d58163cd5b2d011a9468541f17`: `src/laplace_3d.rs`
  (`evaluate_laplace_one_target`, `assemble_pairwise_st`), `benches/laplace_f64.rs`.
- rlst 0.9.0, `src/simd.rs` (`simd_approx_recip_sqrt` for f32 and f64), and
  [linalg-rs/rlst](https://github.com/linalg-rs/rlst).
- Intel, [Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html):
  `_mm256_rsqrt_ps` (relative error ≤ 1.5 · 2⁻¹²); for the deferred AVX-512 path,
  `_mm512_rsqrt14_ps` and `_mm512_rsqrt14_pd` (< 2⁻¹⁴).
- Arm, [Intrinsics reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/):
  `vrsqrteq_f32`, `vrsqrteq_f64`, `vrsqrtsq_f32`, `vrsqrtsq_f64`, `vld3q`, `vst3q`; Arm
  Architecture Reference Manual, FRSQRTE and FRSQRTS.
- Rust, [`core::arch`](https://doc.rust-lang.org/core/arch/index.html),
  `is_x86_feature_detected!`, and the release notes of Rust 1.86 (target_feature 1.1),
  1.87 (safe intrinsics in matching contexts) and 1.89 (AVX-512 target features).
- L. Nyland, M. Harris, J. Prins (2007). Fast N-body simulation with CUDA. GPU Gems 3,
  ch. 31 (target-per-thread loop order and rsqrt in the all-pairs kernel).

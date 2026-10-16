# Phase 6 T6: AVX-512 P2P (C6.7)

Phase 6, task T6 ([docs/phase6/T6-avx512-p2p.md](../../docs/phase6/T6-avx512-p2p.md));
design [docs/design/optimisation.md](../../docs/design/optimisation.md) §5, signed off on
2026-10-15 (decision 4; §5.5 records what was built). Measured on 2026-10-16 from branch
`tbetcke/phase6_t6` (main at 2c9ca3b plus this task's changes), release builds, rustc
1.99.0, every BLAS thread variable at 1. Kathleen is usable (docs/phase6/README.md): every
Kathleen job used one node (`singlenode`, nodes exclusive), so the path "with Kathleen"
of the README; no AVX-512 code runs on the M3 Max or locust (aarch64).

## Summary

1. **`Isa::Avx512`**: every kernel of `nd-fmm-simd` (P2P, P2P of several charge vectors,
   the batched M2L's column kernel, the inverse square root) on AVX-512F, in
   `arch::avx512` (`avx512f` only; `vrsqrt14ps`/`pd` with one Newton step in f32 and two
   in f64; masks in mask registers). Available where the CPU has AVX-512F, AVX2 and FMA;
   selectable as `P2pChoice::Isa(Isa::Avx512)` (text `avx512`), which on the host also runs
   the batched M2L's kernel; **never chosen by `Isa::detect` or `P2pChoice::Auto`**, which
   keep AVX2 until T11 (decision 4). The rule is a function of the feature set, tested with
   forced sets (§1).
2. **Accuracy, on Kathleen's Cascade Lake** (§2): the inverse square root within 1.498 u₃₂
   (every f32 in [1, 4)) and 1.488 u₆₄ (10⁷ samples), against the contract of 4 u_T and
   AVX2's 1.500 / 1.000; P2P terms within 4.18 u (φ) and 9.60 u (∇φ) over 10⁶ pairs per
   precision, against 8 and 16 u; every `tests/mpi_exec.rs` scenario with AVX-512 within
   the C3S.5 tolerance of the reference kernel and bit for bit at 1, 2, 4 and 8 threads,
   at 1, 2, 4 and 8 ranks. The batched M2L kernel is bit for bit the scalar layer (and so
   `Dense`) on AVX-512 too.
3. **The kernel, one core** (§3; `p2p_kernels`, job 604547): AVX-512 against AVX2 on the
   FMM-shaped gathered cells (geometric means), **1.85× (f32 φ), 2.03× (f32 φ, ∇φ), 3.08×
   (f64 φ), 2.98× (f64 φ, ∇φ)**; from 16 targets per leaf up 2.2–2.5× in f32 and
   2.8–3.1× in f64. f64 gains more than the 2.4× of simd-p2p §4.6's model because AVX2's
   f64 inverse square root goes through f32 and a degree-5 correction (71–80 % of its
   model), while AVX-512 reaches 94–104 % of its. Below 16 targets f32 loses lanes: at
   n_t = 8 AVX-512 is 0.55–0.71× AVX2 per pair (27 calls of 8 targets), 1.03–1.08× gathered.
4. **The kernel, every core at once** (§4; job 604550): with all 40 cores running the same
   kernel, AVX2 keeps 0.73–0.76 of its one-core throughput and AVX-512 0.64–0.66: the
   512-bit licence costs about 13 % more than AVX2's. AVX-512 still leads by **1.37–1.47×
   in f32 and 2.60–2.63× in f64**.
5. **Inside the FMM at 40 ranks per node** (§5; the `scaling` harness, job 604547), the
   default strategy, AVX-512 against AVX2:
   - the **P2P kind 1.40–1.72× (f32) and 2.03–2.33× (f64)** faster; P2P's fitted constant
     falls from 2.78–3.08 to 1.19–1.50 ns per pair in f64 and from 1.09–1.37 to 0.65–0.96 ns
     in f32 (T1's baseline: 3.09 and 1.35);
   - the **evaluation 1.49–1.63× faster on the cube at f64 p = 3**, 1.17–1.30× at f32 p = 3,
     1.00–1.07× at p = 8 (where M2L dominates), and 0.99–1.11× on the Plummer sphere, where
     M2P dominates the leaves;
   - the stages that run no AVX-512 code (Dense and Rotation M2L) within 0.97–1.03 of their
     AVX2-run times in the sweep and 0.99–1.02 in three repeats on another node: no
     slowdown of the rest of a fully loaded node shows above run-to-run variation (§5.4).
6. **The batched M2L's kernel** (§6; `--strategy batched`, the N = 10⁶ cube): with R = 4
   row vectors (tuned, §7) the M2L kind 1.26–1.42× (f32) and 1.33–1.38× (f64) faster at 40
   ranks per node, 1.47–1.71× and 1.53–1.58× on one; the evaluation 1.19–1.49× at 40 ranks.
7. **C6.7 is met**: AVX-512 is faster than AVX2 on Kathleen in f32 and f64 at FMM leaf
   sizes, in the kernel and inside the FMM on a fully loaded node, and no evaluation
   measured is slower beyond run-to-run variation. **Proposal for T11**: make
   AVX-512 `Auto`'s ISA on CPUs with AVX-512F (measured on Cascade Lake only; the
   one-rank outputs change in the last bits, requirement 6), with the cost model's P2P
   constant refitted (§8).

## 1. What T6 adds

- **`nd-fmm-simd`**:
  - `Isa::Avx512` (the enum is `#[non_exhaustive]`, so no breaking change): 16 f32 / 8 f64
    lanes, name `avx512`, available where `is_x86_feature_detected!` finds `avx512f`, `avx2`
    and `fma` (the entry points enable `avx512f`, which implies the other two to the
    compiler). `Isa::detect()` never returns it (decision 4, optimisation.md §5.3): the
    detection is a function of the feature set (`isa::Features`), and a unit test runs
    every combination of the three features and checks that detection picks an ISA the set
    runs, never AVX-512, and AVX2 wherever AVX2 and FMA are present (§5.4's dispatch test
    "with a forced feature set").
  - `arch::avx512`: the token `Avx512` and the vector layer `Simd<f32>`/`Simd<f64>` on
    `__m512`/`__m512d`, every intrinsic AVX-512F; `Mask` is `__mmask16`/`__mmask8`, `eq` a
    compare into a mask register, `and_not(m, v)` the zero-masked move under `!m`; tails
    run on the padded stack copies of the generic bodies as on every ISA (`MAX_LANES` raised
    from 8 to 16), so a target's instruction sequence does not depend on its position. The
    inverse square root: `vrsqrt14ps`/`pd` (below 2⁻¹⁴), then y ← y + y (½ − (x/2) y²) once
    (f32) or twice (f64): 5 and 8 operations against AVX2's 6 and 11, derived bounds 1.6 u₃₂
    and 1.9 u₆₄. Entry points for the inverse square root, P2P (K = 2 target vectors per
    block), P2P of several charge vectors (T5's groups, 1 × 8 and 1 × 4) and the column
    kernel (R = 4 row vectors, separate and fused), each `#[target_feature(enable =
    "avx512f")]` and `#[inline(never)]`, dispatched from `arch/mod.rs` with `// SAFETY:`
    comments. `unsafe` stays in `arch` and the dispatch.
  - Every test that runs a kernel prints the ISAs it ran and those this machine cannot run
    ("not available here: neon" on Kathleen, "avx2, avx512" on the M3 Max); the AVX-512
    tests skip on a CPU without AVX-512F, never fail.
- **`nd-fmm-exec`**: nothing but documentation. `P2pChoice` parses any ISA name, so
  `P2pChoice::Isa(Isa::Avx512)` and the text `avx512` work without a change, and
  `tests/mpi_exec.rs` runs every available ISA against the reference already.
- **`nd-fmm-validate`**: `scaling --p2p auto|reference|scalar|neon|avx2|avx512`
  (`Settings::p2p`, `Report::p2p`, the report's "P2P kernel" and the summary's `p2p=`;
  an ISA the machine lacks is refused before MPI starts), tested in `tests/scaling.rs` (the
  scalar ISA's errors equal the reference's bit for bit); `p2p_kernels --isa name` (the
  reference and one ISA, for loading every core with one ISA) and the AVX-512 row of its
  model (2 FMA pipes, 2W / ops, ops 9 or 15 plus 5 (f32) or 8 (f64)).
- **Tools**: the sweeps `t6-fmm` and `t6-m2l` of `tools/scaling/run.sh` (each
  configuration with `--p2p avx2` and `--p2p avx512` one after the other, `--kinds`,
  overlap off) and the step `p2p-node` of `tools/kathleen/jobs/sweep.sbatch`.

## 2. Correctness

Error measures as named; "bit for bit" is exact equality of every output value's bits.

| check | where | result |
| --- | --- | --- |
| the inverse square root against 1/√x (`tests/rsqrt`): every f32 in [1, 4) (16,777,216 values), 10⁷ log-uniform f64 samples of the kernel domain, the powers of two, their neighbours and the domain ends, +0 at x = 0 | Kathleen (job 604543, node-c11b-003; again in job 604549) | AVX-512: **1.498 u₃₂, 1.488 u₆₄** (powers of two 0.701 and 0.798 u); AVX2 1.500 and 1.000 u; scalar 1.500 and 1.496 u; contract 4 u_T |
| P2P terms, 10⁶ seeded pairs across the kernel domain per precision and output (`terms_*_large`) | Kathleen | AVX-512 φ 3.99–4.18 u, ∇φ 8.66 u (f32) and 9.60 u (f64); AVX2 3.96–4.18 u and 7.95–8.70 u; contract 8 and 16 u |
| P2P sums against `direct_sum`, every n_t at 4,096 sources (`seeded_sums_*`) | Kathleen | AVX-512 f32 2.0e-7 (φ), 5.8e-7 (∇φ); f64 2.5e-16, 1.7e-15; as AVX2 |
| the vector layer against the scalar layer lane by lane (arithmetic, masks, loads and stores, interleaved blocks, short slices panic), the estimate within 2⁻¹⁴, the P2P body bit for bit for K = 1, 2, 4 at every n_t up to 4 W + 1, the several-vector body bit for bit the single body for every K and group (`src/arch/tests.rs`) | Kathleen (scalar, AVX2, AVX-512) | pass |
| the public kernels: chunk and target-position invariance, coincident pairs, the domain ends, accumulation onto nonzero outputs, panics on wrong lengths, `evaluate_many` against `evaluate` (`tests/p2p`) | Kathleen (scalar, AVX2, AVX-512) | pass, bit for bit where required |
| `MatrixKernel` against the matrix–vector loop of `MatrixSet::apply`, both modes, p = 0–20 and 1–64 columns (the ignored sweep) | Kathleen (scalar, AVX2, AVX-512) | bit for bit |
| the dispatch rule with every forced feature set (`isa::tests`) | M3 Max, Kathleen | pass |
| every `tests/mpi_exec.rs` scenario with `P2pChoice::Isa(Isa::Avx512)` (`evaluate_every_kernel`): bit for bit at 1, 2, 4 and 8 threads, overlapped bit for bit blocking, and within 1e-13 (f64) and 1e-6 (f32) relative L2 of the `Reference` output (C3S.5, inside requirement 1) | Kathleen at 1, 2, 4 and 8 ranks (job 604548, debug) | pass: e.g. f64 4.4e-16 (φ), 2.5e-16 (∇φ), f32 1.8e-7, 1.6e-7, as AVX2 |
| the FMM against the direct sum, AVX-512 against the reference kernel (`p2p_fmm --part kernels`, N = 10⁵, p = 3 and 8) | Kathleen (job 604547) | error ratios 1.000000 in f64, 0.9996–1.0001 in f32, as AVX2's |

The tests skip AVX-512 cleanly where the CPU lacks it: on the M3 Max every line says "not
available here: avx2, avx512". **CI**: the x86_64 leg of `run-tests-simd` runs on a mixed
pool; of its 10 most recent jobs (2026-10-15 to 16, runs 38794243923 to 38941255101) 6
ran on AMD EPYC 9V74 (Zen 4) or 9V45 (Zen 5) with `avx512f detected: true` and 4 on EPYC
7763 (Zen 3) without (optimisation.md §5.4 had 14 of 40). So CI runs the AVX-512 tests,
the exhaustive f32 check included, about half the time, on AMD's AVX-512 (Zen 4 splits
512-bit operations, so its timings would differ; its accuracy is checked). The timings
and the in-FMM measurement are Kathleen's only.

## 3. The kernel on one core

`p2p_kernels --ghz 2.5` (every cell, f32 and f64, φ and φ with ∇φ), one thread on core 0 of
node-c11a-022 (job 604547), the other cores idle. Gpairs/s, geometric means over the cells
of each group ("x": over AVX2), and the fraction of simd-p2p §4.6's model at the assumed
2.5 GHz (a core with one busy core turbos higher, so the AVX2 fractions above 100 % are
the clock, not the model):

| precision | output | AVX2 W1 gathered | AVX-512 W1 gathered (x) | AVX2 W2 | AVX-512 W2 (x) | of model AVX2 / AVX-512 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| f32 | φ | 2.826 | 5.223 (1.85) | 2.879 | 6.305 (2.19) | 106 % / 91 % |
| f32 | φ, ∇φ | 1.603 | 3.247 (2.03) | 1.595 | 3.968 (2.49) | 84 % / 81 % |
| f64 | φ | 0.795 | 2.445 (3.08) | 0.798 | 2.499 (3.13) | 79 % / 104 % |
| f64 | φ, ∇φ | 0.548 | 1.634 (2.98) | 0.550 | 1.682 (3.06) | 71 % / 94 % |

By leaf size (W1 gathered, AVX-512 over AVX2): f32 φ 1.03 (n_t = 8), 2.17 (16), 1.62 (24),
2.28 (32), 2.19 (64), 2.20 (128); f32 φ, ∇φ 1.08, 2.38, 1.75, 2.48, 2.48, 2.49; f64 φ 2.95,
3.09, 3.07, 3.11, 3.12, 3.13; f64 φ, ∇φ 2.78, 3.02, 2.96, 3.04, 3.05, 3.05. Per pair (27
calls of n_t targets) at n_t = 8 f32 loses: 0.55 (φ) and 0.71 (φ, ∇φ), where 8 targets
fill half of the 16 lanes and every call runs the padded tail. The worst lane waste of
simd-p2p §4.1 is real but confined to f32 at fewer than 16 targets; at n_t = 24 (16 + a
padded 8) the gain halves against n_t = 32. Every check passed (16/16 per group): AVX-512's
worst max error 1.6e-7 (f32 φ), 2.6e-6 (f32 ∇φ, the reference's 2.7e-6), 6.3e-16 and
6.7e-15 (f64). The scalar ISA runs the reference's loop (0.89–1.06× the reference).

**The model** (simd-p2p §4.6): AVX-512 1.60 (f32) and 0.70 (f64) pairs per cycle with ∇φ,
2.0× and 2.4× AVX2's 0.80 and 0.29 (the model rows of `p2p_kernels`: 2 × 16 / 20 and 2 × 8
/ 23 with the 5- and 8-operation inverse square root). Measured with ∇φ, 1.30 and 0.65
pairs per cycle at 2.5 GHz. f32 gains 2.0×, as modelled; f64 gains 3.0×, more than
modelled, because AVX2's f64 kernel reaches only 71 % of its model (the conversions to and
from f32 around `vrsqrtps` and the degree-5 correction lengthen its chain) and AVX-512's
94 %.

## 4. The kernel on a fully loaded node

`p2p_kernels --quick --isa <isa> --ghz 2.5` on all 40 cores at once, one copy pinned to
each core, first every copy with AVX2, then every copy with AVX-512 (job 604550,
node-c11a-060; the reference also runs in each copy, between the ISA's cells). Gpairs/s
per core on the W1 gathered cell (n_t = 24) and W2 (N = 1,000, t = s), the median over the
40 cores (min–max), against one copy alone (job 604543, node-c11b-003, the same `--quick`
cells):

| precision | output | AVX2: 40 cores (one core) | AVX-512: 40 cores (one core) | AVX-512 / AVX2, 40 cores | W2, 40 cores: AVX2 / AVX-512 |
| --- | --- | --- | --- | ---: | --- |
| f32 | φ | 2.183, 2.085–2.193 (2.970) | 2.988, 2.948–3.043 (4.685) | 1.37 | 2.194 / 4.238 (1.93) |
| f32 | φ, ∇φ | 1.273, 1.152–1.302 (1.745) | 1.871, 1.810–1.926 (2.929) | 1.47 | 1.280 / 2.584 (2.02) |
| f64 | φ | 0.603, 0.583–0.615 (0.795) | 1.584, 1.561–1.655 (2.415) | 2.63 | 0.604 / 1.626 (2.69) |
| f64 | φ, ∇φ | 0.409, 0.402–0.411 (0.546) | 1.064, 1.058–1.094 (1.619) | 2.60 | 0.410 / 1.159 (2.83) |

With every core busy AVX2 keeps 0.73–0.76 of its one-core rate and AVX-512 0.64–0.66: the
all-core clock is lower for both, and lower again under 512-bit code (Cascade Lake's
frequency licences). The licence takes about 13 % of AVX-512's lead, which stays 1.37–1.47×
in f32 and 2.6× in f64 on the FMM-shaped cell. (n_t = 24 is the `--quick` W1 cell, the one
where f32's 16 lanes waste most; the full sweep of §3 has the others.)

## 5. Inside the FMM

The `scaling` harness (`tools/scaling/run.sh t6-fmm`, job 604547, node-c11a-022, `EVALS=5`):
the cube and the Plummer sphere, f64 and f32 at p = 3 and 8, gradients, the default
strategy (`auto`: the node rule, with the budget of 96 MB), the work-weighted cut, overlap
off, `KindTiming::Synchronous`; one rank (`--map-by ppr:1:node:PE=1 --bind-to core`) at
N = 10⁶, and one rank per core (`--map-by ppr:20:socket:PE=1 --bind-to core`, 40 ranks) at
N = 10⁶ and 10⁷; `--p2p avx2` and `--p2p avx512` launched one after the other per
configuration. Times in ms, each stage's slowest rank, the median over 5 evaluations; "P2P
kind" the slowest rank's P2P time; "ns/pair" the P2P constant fitted to the kind times
(`PathTimes::fit`, Σ_r time over Σ_r pairs).

### 5.1 One rank per core (40 × 1): the frequency licence included

| dist | N | prec | p | strategy | evaluation AVX2 → AVX-512 (×) | leaves (×) | P2P kind (×) | P2P ns/pair | downward (×) |
| --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| cube | 10⁶ | f64 | 3 | Dense | 95.4 → 64.1 (1.49) | 66.0 → 34.3 (1.93) | 59.6 → 27.2 (2.19) | 2.887 → 1.323 | 22.9 → 22.9 (1.00) |
| cube | 10⁷ | f64 | 3 | Dense | 1052.6 → 646.4 (1.63) | 790.1 → 376.5 (2.10) | 723.2 → 313.3 (2.31) | 2.779 → 1.200 | 188.9 → 192.7 (0.98) |
| cube | 10⁶ | f64 | 8 | Rotation | 605.7 → 576.1 (1.05) | 87.6 → 56.3 (1.56) | 60.9 → 27.3 (2.23) | 2.887 → 1.294 | 491.3 → 493.5 (1.00) |
| cube | 10⁷ | f64 | 8 | Rotation | 5407.3 → 5040.8 (1.07) | 1031.3 → 624.4 (1.65) | 742.2 → 319.1 (2.33) | 2.782 → 1.194 | 4192.6 → 4210.5 (1.00) |
| cube | 10⁶ | f32 | 3 | Dense | 47.8 → 40.9 (1.17) | 30.9 → 23.3 (1.33) | 24.5 → 16.3 (1.50) | 1.192 → 0.794 | 12.8 → 12.4 (1.03) |
| cube | 10⁷ | f32 | 3 | Dense | 522.5 → 401.2 (1.30) | 351.6 → 232.2 (1.51) | 293.1 → 170.3 (1.72) | 1.091 → 0.652 | 112.7 → 111.8 (1.01) |
| cube | 10⁶ | f32 | 8 | Rotation | 511.3 → 510.5 (1.00) | 54.7 → 44.7 (1.22) | 26.4 → 15.7 (1.68) | 1.189 → 0.750 | 437.9 → 442.4 (0.99) |
| cube | 10⁷ | f32 | 8 | Rotation | 4532.5 → 4383.2 (1.03) | 578.6 → 474.7 (1.22) | 290.5 → 173.7 (1.67) | 1.098 → 0.653 | 3769.3 → 3718.1 (1.01) |
| plummer | 10⁶ | f64 | 3 | Dense | 207.8 → 193.4 (1.07) | 140.4 → 118.6 (1.18) | 80.4 → 39.7 (2.03) | 3.073 → 1.504 | 77.3 → 79.3 (0.97) |
| plummer | 10⁷ | f64 | 3 | Dense | 1789.0 → 1616.4 (1.11) | 1168.1 → 955.1 (1.22) | 675.4 → 322.9 (2.09) | 3.073 → 1.501 | 647.4 → 646.6 (1.00) |
| plummer | 10⁶ | f64 | 8 | Rotation | 1337.5 → 1311.6 (1.02) | 511.2 → 492.2 (1.04) | 68.8 → 32.7 (2.11) | 3.082 → 1.502 | 1049.7 → 1044.1 (1.01) |
| plummer | 10⁷ | f64 | 8 | Rotation | 12870.6 → 12630.7 (1.02) | 3555.8 → 3352.0 (1.06) | 702.9 → 332.4 (2.11) | 3.069 → 1.502 | 9869.5 → 9749.4 (1.01) |
| plummer | 10⁶ | f32 | 3 | Dense | 168.3 → 160.5 (1.05) | 119.0 → 111.4 (1.07) | 35.1 → 25.1 (1.40) | 1.360 → 0.955 | 50.5 → 50.4 (1.00) |
| plummer | 10⁷ | f32 | 3 | Dense | 1382.2 → 1344.2 (1.03) | 928.3 → 870.5 (1.07) | 297.0 → 203.5 (1.46) | 1.365 → 0.949 | 453.4 → 454.3 (1.00) |
| plummer | 10⁶ | f32 | 8 | Rotation | 1214.6 → 1227.8 (0.99) | 494.1 → 489.7 (1.01) | 29.9 → 20.5 (1.46) | 1.368 → 0.956 | 953.2 → 980.3 (0.97) |
| plummer | 10⁷ | f32 | 8 | Rotation | 11633.6 → 11425.2 (1.02) | 3336.6 → 3288.9 (1.01) | 309.6 → 200.1 (1.55) | 1.355 → 0.949 | 8731.6 → 8738.3 (1.00) |

### 5.2 One rank (1 × 1), N = 10⁶

| dist | prec | p | strategy | evaluation (×) | leaves (×) | P2P kind (×) | P2P ns/pair |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| cube | f64 | 3 | Dense | 2533.3 → 1539.3 (1.65) | 1825.0 → 834.5 (2.19) | 1647.8 → 658.0 (2.50) | 2.127 → 0.849 |
| cube | f64 | 8 | Rotation | 16655.0 → 15643.8 (1.06) | 2440.9 → 1453.7 (1.68) | 1647.7 → 659.6 (2.50) | 2.127 → 0.851 |
| cube | f32 | 3 | Dense | 1332.9 → 1022.0 (1.30) | 872.2 → 559.9 (1.56) | 693.8 → 382.0 (1.82) | 0.895 → 0.493 |
| cube | f32 | 8 | Dense | 10902.7 → 10611.6 (1.03) | 1478.6 → 1186.4 (1.25) | 673.6 → 381.9 (1.76) | 0.869 → 0.493 |
| plummer | f64 | 3 | Dense | 5854.0 → 4985.1 (1.17) | 3960.4 → 3100.2 (1.28) | 1523.7 → 660.7 (2.31) | 2.264 → 0.982 |
| plummer | f64 | 8 | Rotation | 38396.6 → 37457.4 (1.03) | 11558.0 → 10703.0 (1.08) | 1525.0 → 671.3 (2.27) | 2.265 → 0.997 |
| plummer | f32 | 3 | Dense | 4553.2 → 4290.3 (1.06) | 3136.8 → 2875.8 (1.09) | 681.0 → 419.7 (1.62) | 1.012 → 0.623 |
| plummer | f32 | 8 | Dense | 29521.9 → 29301.2 (1.01) | 10849.9 → 10603.9 (1.02) | 666.7 → 420.9 (1.58) | 0.990 → 0.625 |

`p2p_fmm --part kernels` (N = 10⁵, one thread, the same job) agrees: on the cube the
leaves 2.06× (f64 p = 3), 1.62× (f64 p = 8), 1.50× (f32 p = 3) and 1.23× (f32 p = 8) faster
than AVX2's, the evaluation 1.54×, 1.05×, 1.28× and 1.02×; on the Plummer sphere the leaves
1.01–1.21×, the evaluation 1.01–1.15×.

### 5.3 What it means

- **P2P inside the FMM gains less than the kernel alone, and less on a full node.** One
  rank: 2.27–2.50× (f64) and 1.58–1.82× (f32), against the kernel's 3.0× and 2.0× on the
  gathered cells: the harness's leaves hold at most 64 points (fewer on average, and many
  below 16, where f32 loses lanes, §3), and each P2P call maps its sources first. 40
  ranks: 2.03–2.33× and 1.40–1.72×, the licence's share (§4).
- **The evaluation gains where P2P is the evaluation**: the cube at p = 3 (P2P 51–69 % of
  the AVX2 evaluation at 40 ranks), 1.17–1.63×. At p = 8 M2L is 77–85 % of it and the gain
  is 0–7 %; on the Plummer sphere M2P is most of the leaves (T12's subject) and the gain is
  1–11 %.
- **The stages without AVX-512 code** moved by 0.97–1.03 at 40 ranks in the sweep (1.00 at
  one rank), and the one evaluation below 1 (Plummer, f32 p = 8, 40 ranks: 0.99, its
  Rotation M2L 0.97) came from that. §5.4 repeats the three configurations with the lowest
  ratios: there they are within run-to-run variation.

### 5.4 Repeats

The three configurations of §5.1 whose downward pass moved most (the cube at N = 10⁷ f64
p = 3: 0.98; the Plummer sphere at N = 10⁶ f64 p = 3: 0.97, and f32 p = 8: 0.97, the one
slower evaluation), each launched three times with AVX2 then AVX-512, 40 ranks, on
another node (job 604551, node-c11b-045, `EVALS=5`):

| configuration | repeat | evaluation AVX2 → AVX-512 (×) | P2P kind (×) | downward (×) |
| --- | ---: | --- | --- | --- |
| cube, 10⁷, f64 p = 3 | 1 | 1066.5 → 668.3 (1.60) | 723.8 → 326.9 (2.21) | 199.3 → 200.5 (0.99) |
| | 2 | 1071.1 → 673.9 (1.59) | 727.4 → 328.9 (2.21) | 199.8 → 200.8 (1.00) |
| | 3 | 1074.1 → 675.2 (1.59) | 728.5 → 330.0 (2.21) | 202.7 → 201.2 (1.01) |
| Plummer, 10⁶, f64 p = 3 | 1 | 216.3 → 199.1 (1.09) | 81.0 → 39.5 (2.05) | 82.1 → 80.4 (1.02) |
| | 2 | 219.7 → 200.4 (1.10) | 80.5 → 39.5 (2.04) | 82.2 → 81.5 (1.01) |
| | 3 | 221.0 → 201.0 (1.10) | 80.5 → 39.8 (2.02) | 82.1 → 82.5 (0.99) |
| Plummer, 10⁶, f32 p = 8 | 1 | 1257.5 → 1244.3 (1.01) | 30.5 → 20.6 (1.48) | 987.8 → 987.1 (1.00) |
| | 2 | 1255.2 → 1258.6 (1.00) | 30.2 → 20.5 (1.47) | 987.7 → 989.2 (1.00) |
| | 3 | 1267.1 → 1255.6 (1.01) | 30.1 → 20.5 (1.47) | 987.7 → 986.8 (1.00) |

The downward pass, which runs no AVX-512 code, is at 0.99–1.02 of its AVX2-run time, and
the Plummer sphere at f32 p = 8 is not slower (1.00–1.01): the 0.97–0.98 of the sweep were
run-to-run variation, about ±3 % between launches (the two nodes differ by 5 % in their
absolute times). If running P2P under the 512-bit licence slows the stages that follow it,
it does so by less than that. The P2P and evaluation gains reproduce within 2 %.

## 6. The batched M2L's kernel

`tools/scaling/run.sh t6-m2l` (job 604547): the N = 10⁶ cube with `--strategy batched`
(separate mode, bit for bit `Dense`), the column kernel and P2P both on the named ISA.

| prec | p | ranks | evaluation AVX2 → AVX-512 (×) | M2L kind, slowest rank (×) | leaves (×) |
| --- | ---: | ---: | --- | --- | --- |
| f64 | 6 | 1 | 3926.8 → 2457.5 (1.60) | 1419.5 → 926.2 (1.53) | 2139.8 → 1155.4 (1.85) |
| f64 | 6 | 40 | 153.8 → 103.4 (1.49) | 58.9 → 42.5 (1.38) | 78.8 → 44.3 (1.78) |
| f64 | 8 | 1 | 7066.3 → 4615.4 (1.53) | 3967.3 → 2508.8 (1.58) | 2442.3 → 1455.2 (1.68) |
| f64 | 8 | 40 | 284.7 → 211.2 (1.35) | 167.8 → 126.0 (1.33) | 89.1 → 57.4 (1.55) |
| f32 | 6 | 1 | 2329.3 → 1757.9 (1.33) | 819.3 → 555.8 (1.47) | 1182.2 → 878.5 (1.35) |
| f32 | 6 | 40 | 89.5 → 75.3 (1.19) | 33.0 → 26.2 (1.26) | 43.3 → 35.0 (1.24) |
| f32 | 8 | 1 | 4077.6 → 2943.2 (1.39) | 2018.5 → 1182.2 (1.71) | 1499.0 → 1186.7 (1.26) |
| f32 | 8 | 40 | 162.2 → 131.2 (1.24) | 81.9 → 57.7 (1.42) | 56.4 → 48.9 (1.15) |

With both kernels on AVX-512, `Batched` at 40 ranks per node takes the cube at f64 p = 8 to
211 ms against `Rotation`'s 576 ms (the default, §5.1) and 285 ms with AVX2: T2's strategy
question at p = 6–8 (T11) now has an AVX-512 column.

## 7. Choices made by measurement

On one core of node-c11b-003 (job 604543), each variant built from a copy of the tree with
the constant changed:

- **P2P, K target vectors per block** (`p2p_kernels --quick`, AVX-512, the gathered W1
  cell, Gpairs/s): K = 1: 4.730 / 2.869 / 2.404 / 1.560 (f32 φ, f32 φ ∇φ, f64 φ, f64 φ
  ∇φ); K = 2: 4.685 / 2.929 / 2.415 / 1.619; K = 4: 4.688 / 2.869 / 2.413 / 1.618. Within
  5 %; **K = 2** is best or tied with ∇φ, which the FMM computes. (On W2, K = 4 is best for
  φ alone and worst with ∇φ.)
- **The column kernel, R row vectors per block** (the N = 10⁶ cube, `Batched`, one rank,
  M2L inside the downward pass): R = 2: 986 / 2896 / 563 / 1348 ms (f64 p = 6, f64 p = 8,
  f32 p = 6, f32 p = 8); **R = 4**: 929 / 2501 / 571 / 1211 ms. R = 4 uses 21 of the 32
  registers (16 accumulators, 4 of A, the broadcast); R = 8 would need 32 accumulators.
- **The several-vector groups** keep AVX2's (1 target vector with 8 vectors without
  gradients, 4 with): not re-tuned; AVX-512's 32 registers hold the 16 accumulators of a
  group with ∇φ that spilled on AVX2.

**Instruction counts** of the compiled loops (fmm-simd/CLAUDE.md, "Inlining check";
`objdump` of the release test binaries built for `x86_64-apple-darwin` on the M3 Max, so
an LLVM build for a generic x86_64 CPU with the features of each entry point), no call in
any inner loop:

| loop | AVX2 | AVX-512 |
| --- | ---: | ---: |
| P2P source loop, f32 φ, per target vector | 24 | 18 (36 at K = 2) |
| f32 φ, ∇φ | 35 | 25 (50) |
| f64 φ | 29 | 21 (42) |
| f64 φ, ∇φ | 42 | 28 (56) |
| inverse square root, per vector, f32 / f64 (`rsqrt_slice`) | 15 / 21 | 14 / 17 |
| column kernel k loop, separate / fused | 27 / 19 (R = 2: 8 mul + 8 add, 8 fma) | 45 / 29 (R = 4: 16 mul + 16 add, 16 fma) |

The f64 φ, ∇φ loop (K = 1 shown, 28 instructions) holds the model's 23 floating-point
operations: 3 sub, mul and 2 fma for r²; `vrsqrt14pd`, the mul of h and twice (mul, fmsub,
fma) for the Newton steps; the compare, whose mask LLVM applies inside the last fma by
zero-masking (`{k1}{z}`) instead of a separate move; 3 mul, an add and 3 fma for the
accumulation; the source's coordinates enter the subtractions as embedded broadcasts. The
rest is a sign flip, a mask negation and the loop. LLVM computes ½ − h y² as an fmsub with
negated operands, the same value rounded once; neither rewrite changes a bit.

## 8. For T11

- **The default ISA** (decision 4, optimisation.md §5.3): on Kathleen AVX-512 is faster
  than AVX2, or equal within run-to-run variation, in every configuration measured inside
  the FMM on a fully loaded node (§5). Proposed: `Isa::detect` picks AVX-512 where the CPU
  has AVX-512F. Measured on Cascade Lake only; other AVX-512 CPUs (later Intel cores, AMD
  Zen 4 and 5, which CI's runners are) are untimed. The one-rank outputs change
  in the last bits of P2P's terms (requirement 6), within C3S.5's tolerance of the
  reference.
- **The cost model** (`nd_fmm_exec::cost`, Kathleen's constants): P2P's fitted constant
  at 40 ranks per node falls from 2.78–3.08 to 1.19–1.50 ns (f64) and from 1.09–1.37 to
  0.65–0.96 ns (f32) with AVX-512; the column kernel's M2L constant by 1.26–1.42×. With
  AVX-512 the default, the constants need refitting (T11 refits them anyway).
- **f32 leaves below 16 points** waste half of AVX-512's lanes (§3); a 256-bit tail on
  AVX-512VL with the same estimate (`vrsqrt14ps` on `ymm`, the same bits per lane) would
  recover it without changing a result, if T11's leaf sizes leave many such leaves.
- **`Batched` with AVX-512** (§6) beats `Rotation` at f64 p = 8 on 40 ranks per node by
  2.7× (211 against 576 ms; with AVX2 285 against 606, 2.1×); the strategy rule of T11
  should be measured with the ISA it will run.

## 9. The checks

Run for this task on 2026-10-16 (M3 Max unless named; Kathleen in the environment of
`tools/kathleen/`):

- `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets -- -D warnings`;
  `RUST_MIN_STACK=8388608 cargo test --workspace` (69 test executables, all passed);
  `cargo doc --no-deps`: clean and passing on the M3 Max. On Kathleen (job 604549,
  node-c11a-060): fmt, `cargo clippy -- -D warnings`, `--examples`, `cargo test`, `cargo
  doc --no-deps` and `cargo clippy --workspace --all-targets` pass; `cargo test --workspace`
  passes every crate, and `spikes/p2p-simd` fails its two known tests
  (`kernel_rsqrt_within_contract`, `compare_core_on_a_small_cell`), as on every Kathleen
  check job since 238839: the spike's AVX2 estimate, not touched here.
- nd-fmm-simd: `cargo clippy -p nd-fmm-simd --all-targets -- -D warnings`, the same with
  `--target x86_64-apple-darwin` (the AVX2 and AVX-512 code), `cargo test -p nd-fmm-simd
  -- --show-output` and the ignored release run on the M3 Max (scalar, NEON); on Kathleen
  (scalar, AVX2, AVX-512; jobs 604543 and 604549) the same tests, the ignored release run
  with the AVX-512 rows of §2, and clippy.
- The MPI list (`.github/scripts/run-mpi-tests.sh`, debug): M3 Max at 2 and 4 ranks;
  Kathleen at 1, 2, 4 and 8 (job 604548, node-c11a-060), every executable passing.
- nd-fmm-validate: `cargo clippy -p nd-fmm-validate --all-targets -- -D warnings`;
  `tests/scaling.rs` at 1 and 2 ranks (M3 Max).

Kathleen jobs (one node each, `singlenode`, waits from seconds to 24 min): 604543 (the
nd-fmm-simd tests and the K and R variants, node-c11b-003, 11 min), 604547 (the sweeps,
node-c11a-022, 48 min), 604548 (the MPI list, node-c11a-060, 24 min), 604549 (the checks,
node-c11a-060, 21 min), 604550 (the full-node kernel, node-c11a-060, 1 min), 604551 (the
repeats, node-c11b-045, 2 min): 1.8 node-hours.

## 10. The raw data

Not in the repository: the launches' files (`tools/scaling/run.sh`, one per launch with the
command, the job, the binding and every table) and the examples' reports on Kathleen in
`/scratch/scratch/ucahtbe/fmm/logs/scaling-604547` (`p2p_kernels.md`, `p2p_fmm.md`, the
`t6-fmm-*` and `t6-m2l-*` launches), `scaling-604550/p2p-node` (one file per core and ISA),
`t6-tune` (the K and R variants) and `t6-repeats`; the jobs' output as
`logs/<job name>-<job id>.out`.

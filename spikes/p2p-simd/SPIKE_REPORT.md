# Spike report: SIMD P2P loop order, inverse square root and the green-kernels baseline (Phase 3S / T2, C3S.2)

Date: 2026-10-02. Brief: [docs/phase3s/T2-simd-spike.md](../../docs/phase3s/T2-simd-spike.md).
Design: [docs/design/simd-p2p.md](../../docs/design/simd-p2p.md). Raw output of the final run, with
the disassembly tables appended: [results-m3max.md](results-m3max.md).

**No x86_64 path was timed.** There is no x86_64 machine, and Rosetta 2 is not installed on the
M3 Max used: `arch -x86_64` fails with "Bad CPU type", and `/Library/Apple/usr/share/rosetta` is
absent. So the AVX2 prototypes were compiled, linted and disassembled, but **never run**, not even
for correctness. The AVX2 inverse-square-root choices rest on the documented estimate bound and on
operation counts until T4's CI tests on real x86_64 hardware confirm them.

## Summary

- **Loop order: targets in lanes stays.** The decision rule of design §4.1 compares best sources
  in lanes with best targets in lanes, as a geometric mean over the 12 FMM-shaped (W1) cells. The
  ratio is 1.046 in f32 and 1.041 in f64, far below the threshold of 1.15. Targets in lanes
  reproduces the per-pair calls bit for bit in one gathered call in every cell; sources in lanes
  does not.
- **Inverse square root on NEON: `sqrt` then `div`, not the estimate.** This is the result that
  matters most, and it overturns the provisional choice of design §4.3.
  - On their own, the estimate formulations are faster: FRSQRTE + 2/3 FRSQRTS takes 2.1/2.9
    cycles per vector, against 3.1 for FSQRT + FDIV.
  - Inside the kernel, `sqrt+div` is faster in every W1 cell and both precisions, by 19% (f32,
    φ+∇φ) to 53% (f64, φ) over the best estimate formulation that meets the contract.
  - The reason: FSQRT and FDIV run on a divider beside the four FP pipes, about 3 cycles per
    vector. The cheapest estimate routes put 5 (f32) to 8 (f64) more operations per pair and
    lane on the pipes, which are already the bottleneck.
  - `sqrt+div` is also accurate, at 1.50 u_T; only the higher-order polynomials do better
    (1.00 u_T). Design §4.3 anticipated this case: "if it is not slower on some core, the dispatch uses it there".
- **Inverse square root on AVX2: the polynomial correction**, chosen from the documented bound and
  the operation count, since nothing can be timed.
  - **f32: `est+P2`.** `rsqrtps` + r = 1 − x y² + a degree-2 correction: derived bound 1.50 u_T, 6
    operations.
  - **f64: `est+P5`.** Via an f32 estimate + degree-5 correction: derived bound 1.50 u_T, 11
    operations. `est+P4` (10 operations) derives 1.97 u_T, right at the factor-2 margin.
  - green-kernels' AVX2 formulations do **not** meet the 4 u_T contract. f32 (1 step) derives
    5.9 u_T; f64 (2 steps from the f32 estimate) derives 550 u_T, a relaxed level of about 6e-14.
- **Register blocking.**
  - NEON: K = 2 in f32 and K = 4 in f64, by the geometric mean over W1.
  - AVX2: K = 1. It is the only blocking without spills in the gradient kernels (16 ymm
    registers).
- **No relaxed f64 level.** On NEON every relaxed level is slower than full-precision
  `sqrt+div`: green-kernels' 2 steps by 15–23%, the degree-4 polynomial by 26–39%. They would
  only have gained (34–48%) against the slower full estimate route.
- **green-kernels on NEON** (pulp backend `Neon`): the recommended prototype beats it in every
  cell.
  - Gathered W1, geometric mean over n_t: 1.33× (f32 φ), 1.20× (f32 φ+∇φ), 1.73× (f64 φ), 1.43×
    (f64 φ+∇φ).
  - Per-pair form: 1.33×, 1.18×, 1.62×, 1.29×.
  - The smallest ratio in any cell is 1.09× (f64, gradient, n_t = 8, per-pair).
  - green-kernels' sums are more accurate, because it adds in W partial sums. Ours adds in source
    order, like the reference. Both are within requirement 2 (see "Accuracy").
- **Against `nd_fmm_ref::p2p`:** 11.5× (f32 φ), 8.1× (f32 φ+∇φ), 5.4× (f64 φ), 4.1× (f64 φ+∇φ),
  gathered W1, geometric mean. The per-pair form gives 6.1×, 5.2×, 3.5×, 2.7×.

## Recommendation, for sign-off before T4

1. **Loop order: targets in lanes**, on every ISA (design §4.1 unchanged). Rule numbers, as SIL /
   TIL geometric means over the W1 cells:

   | precision | φ | φ, ∇φ | both |
   | --- | --- | --- | --- |
   | f32 | 1.054 | 1.039 | 1.046 |
   | f64 | 1.026 | 1.057 | 1.041 |

   Sources in lanes wins only at the smallest per-pair calls (n_t = 8: 1.09–1.30×). There, the
   per-call cost of building a padded block of targets weighs most. T6's per-pair against
   gathered measurement addresses that, and gathering does not change a bit of the output.
2. **Inverse square root per ISA and precision:**

   | ISA | f32 | f64 | evidence |
   | --- | --- | --- | --- |
   | NEON | `sqrt+div` (FSQRT, FDIV): 1.50 u_T | `sqrt+div`: 1.50 u_T | measured, exhaustive f32 / 10⁷ samples f64; fastest in the kernel (table "Inside the kernel") |
   | AVX2 + FMA | `est+P2`: `vrsqrtps`, r = 1 − x y², y + (y r)(½ + ⅜ r); derived ≤ 1.50 u_T; 6 ops | `est+P5`: `vcvtpd2ps`, `vrsqrtps`, `vcvtps2pd`, degree-5 correction; derived ≤ 1.50 u_T; 11 ops | documented bound 1.5·2⁻¹² and operation count; not run |

   Fallbacks, if a later core or measurement favours the estimate:
   - NEON: the fastest estimate formulations inside the kernel are FRSQRTE + 2 FRSQRTS (f32,
     2.43 u_T) and + 3 FRSQRTS (f64, 2.45 u_T). These are green-kernels' NEON formulations.
   - AVX2 f64: `est+P4` (1.97 u_T derived) saves one operation, at the edge of the margin.
   - AVX2 `sqrt+div` is not ruled out by anything measured. The NEON result shows that the
     divider can win even though it looks slower in isolation. It should be timed when an x86_64
     machine is available. Until then the design rule (bound and operation count) picks `est+P2`
     and `est+P5`. It would also remove the vendor dependence of `rsqrtps` (design §5.5).

   For T4 this means:
   - NEON's "vector inverse square root" is `vsqrtq` + `vdivq`. The 4 u_T contract tests still
     apply and pass at 1.50 u_T.
   - At r² = 0 it gives +∞, which the and-not mask clears.
3. **K (target vectors per block, targets in lanes):** NEON f32 K = 2; NEON f64 K = 4; AVX2 K = 1
   in both precisions.
   - NEON, geometric mean over W1 (Gpairs/s):

     | output | f32 K = 1 / 2 / 4 | f64 K = 1 / 2 / 4 |
     | --- | --- | --- |
     | φ | 3.24 / 3.69 / 3.11 | 1.49 / 1.85 / 2.03 |
     | φ, ∇φ | 2.15 / 2.33 / 1.98 | 1.03 / 1.15 / 1.18 |

     f32 K = 4 loses at n_t = 8 and 24 to lane waste (16 targets per block).
   - AVX2: K = 1 is spill-free in all four entry points. K = 2 and 4 spill 14–60 stack references
     per iteration with gradients.
4. **Relaxed f64 level: do not ship.**
   - Measured NEON effect at the gathered W1 cells, against full-precision `sqrt+div` at the
     same K:
     - green-kernels' FRSQRTE + 2 FRSQRTS: −23% (φ) and −15% (φ, ∇φ), at errors of 6.5e-12 and
       7.4e-11 relative to the term magnitudes;
     - degree-4 polynomial: −39% and −26%, at errors of 6.1e-14 and 4.2e-13;
     - estimate + 1 Newton step + degree-2 polynomial (95 u_T per pair): −43% and −33%.
   - On AVX2 the relaxed level would be green-kernels' `est+S2` (10 ops, about 550 u_T derived)
     against `est+P5` (11 ops). That is one operation per pair, not worth a second accuracy level.
5. **Expected production throughput (T5 must reach 90%).** NEON, targets in lanes, chosen
   formulation and K, release, one thread:

   | precision | output | K | gathered W1, Gpairs/s (geomean over n_t ∈ {8 … 128}) | of which n_t = 64 | per-pair W1 (geomean) | 90% target (gathered) |
   | --- | --- | --- | --- | --- | --- | --- |
   | f32 | φ | 2 | 4.00 | 4.07 | 3.40 | 3.60 |
   | f32 | φ, ∇φ | 2 | 2.60 | 2.65 | 2.08 | 2.34 |
   | f64 | φ | 4 | 2.19 | 2.20 | 1.89 | 1.97 |
   | f64 | φ, ∇φ | 4 | 1.30 | 1.31 | 1.07 | 1.17 |

   Expected AVX2 inner-loop instruction counts per source iteration, targets in lanes, K = 1, from
   the release disassembly of the spike (`inner_loops.py`). These are the counts T5 compares
   with, within 25% of the operation count:

   | precision | output | loop instructions | FP ops | model (design §4.2 with the chosen rsqrt) | loads/broadcasts | spills |
   | --- | --- | --- | --- | --- | --- | --- |
   | f32 | φ | 23 | 15 | 15 = 9 + 6 (`est+P2`) | 4 | 0 |
   | f32 | φ, ∇φ | 33 | 21 | 21 = 15 + 6 | 5 | 1 |
   | f64 | φ | 29 | 20 | 20 = 9 + 11 (`est+P5`) | 4 | 0 |
   | f64 | φ, ∇φ | 41 | 26 | 26 = 15 + 11 | 5 | 4 |

   The 4 loads per source are the three coordinate broadcasts and the charge; the rest is loop
   control. No loop contains a call.

Proposed changes to the requirements and design, for the same sign-off (T2 may propose them,
design §3):

- **Design §4.3 / §4.6 / §5.3.** On NEON the kernel uses `sqrt` + `div`. The "inverse square
  root written per ISA" of T4 becomes `vsqrtq_f*` + `vdivq_f*` on NEON. The §4.6 model for the
  M3 Max must count the divider: about 3 cycles per vector (FSQRT 0.5 per cycle, FDIV 1 per
  cycle, sharing one unit). It is reached at 69–82% (section "Fraction of the model").
- **C3S.6, "at equal or better accuracy" than green-kernels.**
  - Speed holds in every cell; accuracy does not, on the sum measure. green-kernels adds each
    target's terms in W partial sums, so its sum error is about half of an in-order sum's
    (f64 φ: 2.1e-16 against 6.1e-16; f32 φ+∇φ, gradient: 1.4e-6 against 2.6e-6).
  - Targets in lanes adds in source order by design, which is requirement 4 and the C3.1
    identity. It cannot match green-kernels' summation without giving that up.
  - Proposal: state the C3S.6 accuracy condition as "within requirement 2, and per pair within
    8/16 u_T". Both libraries meet that.
- **Requirement 2, f32 gradient sums.** On W2 N = 1,000 with targets equal to sources, f32 with
  gradients, in-order summation lands at the 1e-6 tolerance (relative to Σ|q|/r²):
  - the reference: 9.6e-7;
  - targets in lanes with `sqrt+div`: 1.0e-6 (check failed, row not timed);
  - targets in lanes with FRSQRTS: 8.6e-7;
  - sources in lanes: 4.2e-7.

  The per-pair terms are fine; it is the summation of 1,000 terms of mixed sign in f32. T5's
  sum tests (design §7, n_s ≤ 4,096) should either use a tolerance relative to the reference's
  own error on the same inputs (this spike passes a row within 2× the reference's error above
  4,096 sources), or keep 1e-6 and use inputs that are not all-pairs-with-self.

## Setup

| item | value |
| --- | --- |
| Machine | Apple M3 Max (`sysctl machdep.cpu.brand_string`), 16 cores (12 performance, 4 efficiency) |
| Build | `cargo run --release -p nd-fmm-spike-p2p-simd`, default target `aarch64-apple-darwin`, no `target-cpu=native` |
| Toolchain | rustc 1.98.0 (88d9e12ae 2026-08-18) |
| ISAs with prototypes | scalar (the reference), NEON. AVX2 + FMA compiled for `x86_64-apple-darwin` only, not run |
| green-kernels | commit `7d757c579f8633d58163cd5b2d011a9468541f17`, `Laplace3dKernel::evaluate_st`, pulp backend `Neon` |
| Threads | one; `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `RAYON_NUM_THREADS` all set to 1 |
| Timing | median of 15 batches of ≥ 20 ms (`nd_fmm_validate::bench::median_time_per_call`, copied so that the spike needs no MPI) |
| Cycles | at an assumed 4.05 GHz (P-core maximum). FMUL and f64 FMLA latencies measure 3.95 cycles at that clock, against the documented 4, so the assumption holds within about 1% |

Method:
- **Inverse square root.**
  - Every candidate is written once over the spike's vector trait (`src/simd/`).
  - f32 errors are measured exhaustively over [1, 4) against 1/√x in f64, plus the powers of two
    of the kernel domain [2⁻¹⁰⁸, 2⁷] (CONVENTIONS §3.13), their neighbours and the ends.
  - f64 errors: 10⁷ seeded log-uniform samples over the domain plus the same special inputs. The
    error comes from the residual 1 − x y², computed with error-free products.
  - Throughput: a 4,096-element buffer in L1. Latency: a dependent chain y ← rsqrt(y).
- **Prototypes** (`src/kernels.rs`):
  - targets in lanes (TIL), K ∈ {1, 2, 4} vectors per block;
  - sources in lanes (SIL), T ∈ {1, 2, 4} targets per block;
  - each with the chosen ("best") inverse square root and with green-kernels' formulation ("gk").
    On NEON "gk" is FRSQRTE + 2 (f32) or 3 (f64) FRSQRTS; on AVX2 it is `rsqrtps` + 1 step (f32)
    and the f32 estimate + 2 steps (f64);
  - "cand" rows (TIL, K = 2, gathered W1) compare every candidate within 4 u_T inside the kernel;
  - "relaxed" rows (TIL, f64) measure the levels below full precision.
  - Every prototype is checked against `direct_sum` on 4 sets of its cell before it is timed. A
    failed check aborts the row.
- **Workloads (design §8.2).**
  - W1: n_t ∈ {8, 16, 24, 32, 64, 128}; 27 leaves of n_t sources each, the centre leaf equal to
    the targets; a pool of 64 sets; per-pair (27 calls) and gathered (1 call) forms.
  - W2: N ∈ {10³, 10⁴}, targets distinct from or equal to the sources.
  - Charges are uniform in [−1, 1]. Each cell runs φ alone and φ with ∇φ, in f32 and f64.
- **Accuracy** is measured against `direct_sum` on the same rounded inputs:
  - the largest error relative to the sum of term magnitudes of its target (Σ|q|/r for φ, Σ|q|/r²
    for each ∇φ component), and the relative L2 error;
  - green-kernels' output is multiplied by 4π first;
  - the check tolerance is requirement 2: 1e-6 (f32) or 1e-14 (f64) up to 4,096 sources;
  - above that (W2, N = 10⁴) a row also passes within 2× the reference's own error;
  - relaxed rows are checked against 1e-8 (the FMM's floor, design §4.3).

Run it with `cargo run --release -p nd-fmm-spike-p2p-simd` (about 25 minutes; `-- --quick` takes
about 2 minutes). Inner-loop counts:
`python3 spikes/p2p-simd/inner_loops.py <binary>`, for the aarch64 and the
`--target x86_64-apple-darwin` release builds. The spike builds and runs unchanged on x86_64, so
x86_64 timings, including green-kernels, need just one command on such a machine.

## green-kernels build check

- **First attempt: green-kernels at the pinned commit, as is. Dependency resolution fails.**
  - green-kernels depends on rlst from git `main`. At the time of the check that was commit
    `4bb29a3037d2baaaaca52370eca99bc320429315` (2025-12-15), package version `0.4.0-dev`.
  - Its build dependency `cc = "=1.2"` cannot coexist with `cc` 1.5.1, which the default members
    lock through rlst 0.9.0 (`cc = "^1.5"`) and mpi-sys 0.2.4 (`cc = "^1.2.66"`).
  - One lock file holds one semver-compatible `cc`, so with the spike as a member the whole
    workspace stopped resolving, the default members included.
- I reported this with the options of design §8.4. **Decided on 2026-10-02: patch green-kernels'
  rlst to the crates.io rlst 0.9.0** (`[patch."https://github.com/linalg-rs/rlst.git"] rlst =
  "0.9.0"` in the root `Cargo.toml`). Only green-kernels uses that source.
- `rlst::simd` (`src/simd.rs`) has the same code in git `main` and in 0.9.0; only doc comments
  differ. So green-kernels' `approx_recip_sqrt` here is the one design §2.2 describes.
- **With the patch, everything builds and links:**
  - `cargo build` (default members);
  - `cargo build --release -p nd-fmm-spike-p2p-simd`, for `aarch64-apple-darwin` and for
    `x86_64-apple-darwin`.

  There is one rlst in the lock file (0.9.0). rlst 0.9.0 has no default features, so the spike
  links no MPI; green-kernels' own git rlst would have enabled rlst's default `mpi` feature.
- **`Cargo.lock` gained:** `green-kernels 0.2.2-dev` (git, pinned commit), `approx 0.5.1`,
  `hexf 0.2.1`, `syn 1.0.109`, and the spike itself. `pulp 0.22.3`, `rayon`, `coe-rs` and
  `bytemuck` were already locked through rlst 0.9.0. Nothing in the default members' dependency
  graph changed.
- **Root `Cargo.toml` gained** `green-kernels` (pinned by rev) and `pulp = "0.22"` in
  `[workspace.dependencies]`, both commented as spikes-only. The spike uses `pulp` only to print
  the backend green-kernels dispatches to.

## Inverse square root, NEON (measured)

Single instructions, cycles at 4.05 GHz:

| instruction | f32 latency | f32 per cycle | f64 latency | f64 per cycle |
| --- | --- | --- | --- | --- |
| FMLA (accumulator chain) | 3.7 | 3.93 | 3.95 | 3.96 |
| FMUL | 3.95 | 3.76 | 3.96 | 3.94 |
| FADD | 2.7 | 3.95 | 2.7 | 3.95 |
| **FRSQRTE** | 3.0 | **1.00** | 3.0 | **1.00** |
| FRSQRTS | 3.9 | 3.79 | 3.9 | 3.98 |
| FSQRT | 9.9 | 0.50 | 12.9 | 0.50 |
| FDIV | 8.0 | 1.00 | 9.9 | 1.00 |

Two port limits matter for the model:
- **FRSQRTE issues on one pipe** (1 per cycle), while FMLA, FMUL and FRSQRTS issue on all four.
  It does not bind in the kernel: there is one estimate per vector, against 16 or more pipe
  operations.
- **FSQRT and FDIV share the divider.** Together they take 1/0.5 + 1/1 = 3 cycles per vector,
  which matches the 3.1 cycles measured for `sqrt+div` alone.

Candidates. Error: max |relative error| in u_T, before the r² = 0 mask. "Ops" counts vector FP
operations including the estimate; the "est" row's count includes the unused ½x.

| precision | candidate | ops | max error (u_T) | ≤ 4 u_T | ns/elem | cycles/vector | latency (cycles) | at 0 / subnormals (raw) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | est (FRSQRTE) | 1 | 54,976 (= 3.3e-3) | no | 0.072 | 1.16 | 3.0 | +∞ / finite (2.7e22 at the smallest) |
| f32 | est+N1 (fma form) | 5 | 271 | no | 0.103 | 1.67 | 14.9 | NaN / NaN or +∞ |
| f32 | est+N2 | 8 | 1.50 | yes | 0.172 | 2.78 | 26.9 | NaN / NaN |
| f32 | est+N3 | 11 | 1.50 | yes | 0.239 | 3.88 | 39.0 | NaN / NaN |
| f32 | est+S1 (FRSQRTS form) | 4 | 272 | no | 0.076 | 1.22 | 15.0 | +∞ / +∞ |
| f32 | est+S2 (gk) | 7 | 2.43 | yes | 0.131 | 2.11 | 26.9 | +∞ / +∞ |
| f32 | est+S3 | 10 | 2.10 | yes | 0.179 | 2.90 | 39.0 | +∞ / +∞ |
| f32 | est+P2 | 6 | 2.20 | yes | 0.137 | 2.22 | 19.2 | NaN / +∞ |
| f32 | est+P3 | 7 | 1.00 | yes | 0.176 | 2.85 | 23.0 | NaN / +∞ |
| f32 | est+P4 | 8 | 1.00 | yes | 0.206 | 3.33 | 27.0 | NaN / +∞ |
| f32 | **sqrt+div** | 2 | **1.50** | yes | 0.188 | 3.05 | 17.9 | +∞ / finite, 2.7e-6 relative at 1e-40 |
| f64 | est (FRSQRTE) | 1 | 2.95e13 (= 3.3e-3) | no | 0.158 | 1.28 | 3.0 | +∞ / finite |
| f64 | est+N2 / est+S2 | 8 / 7 | 3.5e6 (= 7.8e-10) | no | 0.342 / 0.259 | 2.77 / 2.09 | 26.9 | NaN / NaN; +∞ / +∞ |
| f64 | est+N3 | 11 | 1.50 | yes | 0.477 | 3.86 | 38.9 | NaN / NaN |
| f64 | est+N4 | 14 | 1.49 | yes | 0.611 | 4.95 | 51.2 | NaN / NaN |
| f64 | est+S3 (gk) | 10 | 2.45 | yes | 0.356 | 2.88 | 38.9 | +∞ / +∞ |
| f64 | est+S4 | 13 | 2.09 | yes | 0.469 | 3.80 | 51.2 | +∞ / +∞ |
| f64 | est+P4 | 8 | 26,947 (= 6.0e-12) | no | 0.407 | 3.30 | 27.0 | NaN / +∞ |
| f64 | est+P6 | 10 | 1.70 | yes | 0.490 | 3.97 | 35.0 | NaN / +∞ |
| f64 | est+P7 | 11 | 1.00 | yes | 0.515 | 4.17 | 38.9 | NaN / +∞ |
| f64 | est+P8 | 12 | 1.00 | yes | 0.561 | 4.54 | 42.9 | NaN / +∞ |
| f64 | est+N1+P2 | 10 | 95 (= 1.1e-14) | no | 0.445 | 3.60 | 31.1 | NaN / NaN or +∞ |
| f64 | est+N1+P3 | 11 | 1.50 | yes | 0.510 | 4.13 | 35.0 | NaN / NaN |
| f64 | **sqrt+div** | 2 | **1.50** | yes | 0.380 | 3.08 | 22.9 | +∞ / finite, exact at 1e-310 |

Notation:
- est = FRSQRTE;
- N_k = k Newton steps in the fma form (h = ½x, y + y(½ − h y²));
- S_k = k steps y·FRSQRTS(x, y²);
- P_m = r = 1 − x y², then y + (y r)(c₁ + c₂ r + … + c_m r^(m−1)) with c_k = C(2k, k)/4ᵏ.

The FRSQRTE error of 3.3e-3 (2^−8.25) is the same in both precisions, as the architecture
defines (design §5.5).

**At r² = 0 every candidate returns +∞ or NaN, which the and-not mask turns into 0.** Subnormal
inputs, which lie outside the kernel domain (CONVENTIONS §3.13), give +∞ or NaN for every refined
estimate: ½x·y² or x·y² overflows. FRSQRTE alone and `sqrt+div` handle them correctly. Nothing
traps.

## Inverse square root, AVX2 (derived, not measured)

Worst-case relative error derived from the documented bound of `rsqrtps`, |e₀| ≤ 1.5·2⁻¹²
(Intel Intrinsics Guide), and the rounding of each step:
- Truncation: Newton gives e' = −(3/2)e² − ½e³; P_m leaves (1 + e)·Σ_{k>m} c_k r^k with
  |r| ≤ 2|e₀| + e₀².
- Rounding, first order:
  - fma form and polynomial: 1.5 u_T (½ u_T from rounding y², propagated, plus 1 u_T for the
    final fma);
  - step form (rlst's `1.5 − ½x·y²` and NEON's FRSQRTS): 2.5 u_T (the step result is rounded once
    more).
- For f64 via f32, the conversion of x to f32 adds 2⁻²⁵ to e₀ (negligible); the conversion back
  is exact.
- Ops count vector FP operations including the estimate, with 2 conversions for f64. "Chain" is
  the dependent path in operations.

| precision | candidate | ops | chain | truncation | bound (u_T) | ≤ 4 u_T | ≤ 2 u_T (margin) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | est | 1 | 1 | 3.7e-4 | 6,145 | no | no |
| f32 | est+N1 | 5 | 4 | 2.0e-7 | 4.9 | no | no |
| f32 | est+S1 (green-kernels / rlst) | 5 | 4 | 2.0e-7 | 5.9 | no | no |
| f32 | **est+P2** | **6** | 5 | 1.2e-10 | **1.50** | yes | **yes** |
| f32 | est+P3 | 7 | 6 | 7.9e-14 | 1.50 | yes | yes |
| f32 | est+N2 | 8 | 7 | 6.1e-14 | 1.50 | yes | yes |
| f32 | est+S2 | 8 | 7 | 6.1e-14 | 2.50 | yes | no |
| f32 | sqrt+div | 2 | 2 | 0 | 1.50 | yes | yes (divider, not the FMA pipes) |
| f64 | est (via f32) | 3 | 3 | 3.7e-4 | 3.3e12 | no | no |
| f64 | est+P3 | 9 | 8 | 7.9e-14 | 712 | no | no |
| f64 | est+N2 | 10 | 9 | 6.1e-14 | 549 | no | no |
| f64 | est+S2 (green-kernels / rlst) | 10 | 9 | 6.1e-14 | 550 | no | no |
| f64 | est+P4 | 10 | 9 | 5.2e-17 | 1.97 | yes | at the edge |
| f64 | **est+P5** | **11** | 10 | 3.5e-20 | **1.50** | yes | **yes** |
| f64 | bits+N3 (integer guess 0x5FE6EB50C7B537A9, 3.4e-2) | 12 | 11 | 3.4e-11 | 3.0e5 | no | no |
| f64 | est+N1+P2 | 13 | 12 | 2.0e-20 | 1.50 | yes | yes |
| f64 | est+N3 | 13 | 12 | 5.5e-27 | 1.50 | yes | yes |
| f64 | est+S3 | 13 | 12 | 5.5e-27 | 2.50 | yes | no |
| f64 | bits+N4 | 15 | 14 | 1.7e-21 | 1.50 | yes | yes |
| f64 | sqrt+div | 2 | 2 | 0 | 1.50 | yes | yes (divider) |

Notes on the AVX2 analysis:
- The integer bit-trick guess costs more than the f32-estimate route at every accuracy (15
  against 11 operations at full precision).
- The x86_64 versions of every candidate are in `src/simd/avx2.rs` and `src/rsqrt_study.rs`.
  They compile and pass `clippy --target x86_64-apple-darwin`, but were never run (no Rosetta).
- The NEON bound analysis predicts the measured values. For example, `est+P2` from
  e₀ = 3.3e-3 predicts 1.5 + 1.4 = 2.9 u_T, and 2.20 u_T was measured.

## Inside the kernel (NEON)

Every candidate within 4 u_T, TIL K = 2, gathered W1 cells, Gpairs/s, geometric mean over n_t
(full rows in results-m3max.md):

| candidate | ops per pair (φ / φ,∇φ) | f32 φ | f32 φ,∇φ | f64 φ | f64 φ,∇φ |
| --- | --- | --- | --- | --- | --- |
| **sqrt+div** | 11 / 17 | **4.000** | **2.588** | **2.042** | **1.272** |
| est+S2 (gk f32) | 16 / 22 | 3.219 | 2.135 | — | — |
| est+P2 | 15 / 21 | 3.103 | 2.169 | — | — |
| est+P3 | 16 / 22 | 2.801 | 2.009 | — | — |
| est+N2 | 17 / 23 | 2.763 | 1.966 | — | — |
| est+S3 (gk f64) | 19 / 25 | 2.677 | 1.792 | 1.331 | 0.893 |
| est+P4 | 17 / 23 | 2.559 | 1.863 | — | — |
| est+N3 | 20 / 26 | 2.308 | 1.657 | 1.152 | 0.829 |
| est+P6 | 19 / 25 | — | — | 1.160 | 0.842 |
| est+N1+P3 | 20 / 26 | — | — | 1.123 | 0.833 |
| est+S4 | 22 / 28 | — | — | 1.122 | 0.756 |
| est+P7 | 20 / 26 | — | — | 1.109 | 0.809 |
| est+P8 | 21 / 27 | — | — | 1.059 | 0.766 |
| est+N4 | 23 / 29 | — | — | 0.968 | 0.700 |

Among the estimate routes the order follows the operation count, and the FRSQRTS step form is
the cheapest (it fuses (3 − x y²)/2 and needs no register copy of a constant accumulator).
`sqrt+div` is ahead of all of them because its two divider operations do not occupy the FP pipes.

A mixed scheme, half the lanes or sources on the divider and half on the estimate, could use both
units. It is excluded: a target's result would then depend on its position in the block, or on
where the source stream is cut, which breaks requirement 4.

## Kernels on NEON

Gpairs/s, one thread, with the best inverse square root (`sqrt+div`). "TIL gk" is targets in
lanes with green-kernels' formulation, at its best K. The ratios use the best K or T per cell. The
reference column shows two outliers, explained under "Other findings".

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

## Instruction counts against design §4.2

FP operations per source iteration, divided by K (TIL) or T (SIL), against the model of §4.2
(9 for φ or 15 for φ, ∇φ, plus the inverse square root):
- **AVX2:** every entry point equals the model exactly, 1.00, for both orders, both precisions
  and K, T ∈ {1, 2, 4}.
- **NEON:** φ equals the model; φ, ∇φ is 1.04–1.06. LLVM rewrites the three FMLS into one `fneg`
  plus three FMLA, one operation more.
- **No inner loop contains a call** (`bl`/`call`).

Spills (stack references per iteration):

| ISA | output | K = 1 | K = 2 | K = 4 |
| --- | --- | --- | --- | --- |
| NEON f32 / f64 | φ | 0 | 0 | 0 |
| NEON f32 / f64 | φ, ∇φ | 0 | 0 | 6 |
| AVX2 f32 | φ / φ, ∇φ | 0 / 1 | 2 / 14 | 12 / 56 |
| AVX2 f64 | φ / φ, ∇φ | 0 / 4 | 6 / 18 | 15 / 60 |

Sources in lanes on AVX2 needs 24 scalar loads per source vector to deinterleave the triples
(the prototype uses `_mm256_setr_*`), against 4 broadcasts per source for targets in lanes.
Recommended AVX2 loops: see the recommendation, item 5. All entry points are listed in
results-m3max.md. Entries missing there were merged by LLVM with an identical function, for
example `til_k2_gk` with `til_k2_c_s2`.

## Fraction of the design §4.6 model (NEON, chosen K)

"Model": 4 pipes × W lanes / (FP ops per pair and lane). "Corrected" adds the measured port
limits: the estimate (1 per cycle), FRSQRTS, and for `sqrt+div` the divider (3 cycles per vector).

| precision | output | rsqrt, K | model pairs/cycle | corrected | measured (W1 gathered n_t = 64) | of model | of corrected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | φ | sqrt+div, 2 | 1.455 | 1.330 | 1.006 | 69% | 76% |
| f32 | φ | est+S2, 2 | 1.000 | 1.000 | 0.806 | 81% | 81% |
| f32 | φ, ∇φ | sqrt+div, 2 | 0.941 | 0.941 | 0.655 | 70% | 70% |
| f32 | φ, ∇φ | est+S2, 2 | 0.727 | 0.727 | 0.533 | 73% | 73% |
| f64 | φ | sqrt+div, 4 | 0.727 | 0.665 | 0.543 | 75% | 82% |
| f64 | φ | est+S3, 4 | 0.421 | 0.421 | 0.349 | 83% | 83% |
| f64 | φ, ∇φ | sqrt+div, 4 | 0.471 | 0.471 | 0.323 | 69% | 69% |
| f64 | φ, ∇φ | est+S3, 4 | 0.320 | 0.320 | 0.227 | 71% | 71% |

The design §4.6 row for the M3 Max (f32 k = 2 with gradients: 0.73; f64 k = 3: 0.32) is reached
at 73% and 71% by the estimate route. The `sqrt+div` kernel reaches 0.66 (f32) and 0.32 (f64) pairs per
cycle with gradients (2.65 and 1.31 Gpairs/s): 90% and 101% of those design figures. The remaining gap
of 20–30% is loop overhead, the broadcasts and the block set-up, and is the same in every cell
from n_t = 16 up.

## Lane utilisation (targets in lanes) at the W1 leaf sizes

n_t / lanes computed, per call (n_t = 20 and 44 added as typical uniform leaf sizes):

| ISA | precision | K | 8 | 16 | 20 | 24 | 32 | 44 | 64 | 128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NEON | f32 | 2 (chosen) | 100% | 100% | 83% | 100% | 100% | 92% | 100% | 100% |
| NEON | f32 | 4 | 50% | 100% | 62% | 75% | 100% | 92% | 100% | 100% |
| NEON | f64 | 4 (chosen) | 100% | 100% | 83% | 100% | 100% | 92% | 100% | 100% |
| AVX2 | f32 | 1 (chosen) | 100% | 100% | 83% | 100% | 100% | 92% | 100% | 100% |
| AVX2 | f32 | 2 | 50% | 100% | 62% | 75% | 100% | 92% | 100% | 100% |
| AVX2 | f32 | 4 | 25% | 50% | 62% | 75% | 100% | 69% | 100% | 100% |
| AVX2 | f64 | 1 (chosen) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| AVX2 | f64 | 4 | 50% | 100% | 62% | 75% | 100% | 92% | 100% | 100% |

The full table is in results-m3max.md.

## Accuracy of the timed rows

Worst over all cells and checked sets: the largest error relative to the term magnitudes. The
relative L2 error and every row are in results-m3max.md. "Failed" counts rows whose check failed
(they were not timed, except the reference and green-kernels, which are baselines).

| precision | output | variant | max err φ | max err ∇φ | failed / rows | per-pair = gathered, bit for bit |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | φ | reference | 1.6e-7 | | 0/16 | |
| f32 | φ | green-kernels | 8.2e-8 | | 0/16 | |
| f32 | φ | TIL sqrt+div (K = 1, 2, 4) | 1.6e-7 | | 0/48 | yes, every cell |
| f32 | φ | TIL gk | 1.7e-7 | | 0/48 | yes |
| f32 | φ | SIL sqrt+div / gk | 9.4e-8 / 8.7e-8 | | 0/48 each | no |
| f32 | φ, ∇φ | reference | 1.6e-7 | 2.7e-6 | 1/16 (W2 N = 10⁴, t = s: 2.7e-6 against a scaled 2.4e-6) | |
| f32 | φ, ∇φ | green-kernels | 8.2e-8 | 1.4e-6 | 0/16 | |
| f32 | φ, ∇φ | TIL sqrt+div | 1.8e-7 | 2.6e-6 | 3/48 (W2 N = 1,000, t = s: 1.0e-6) | yes |
| f32 | φ, ∇φ | TIL gk | 1.7e-7 | 2.6e-6 | 0/48 | yes |
| f32 | φ, ∇φ | SIL sqrt+div / gk | 9.4e-8 | 1.5e-6 | 0/48 each | no |
| f64 | φ | reference | 6.2e-16 | | 0/16 | |
| f64 | φ | green-kernels | 2.1e-16 | | 0/16 | |
| f64 | φ | TIL sqrt+div / gk | 6.1e-16 / 6.2e-16 | | 0/48 each | yes |
| f64 | φ | SIL sqrt+div / gk | 2.0e-16 / 2.3e-16 | | 0/48 each | no |
| f64 | φ | relaxed S2 / P4 / N1+P2 | 6.5e-12 / 6.1e-14 / 2.9e-16 | | 0/18 each (relaxed check 1e-8) | |
| f64 | φ, ∇φ | reference | 6.2e-16 | 6.8e-15 | 0/16 | |
| f64 | φ, ∇φ | green-kernels | 2.1e-16 | 3.6e-15 | 0/16 | |
| f64 | φ, ∇φ | TIL sqrt+div / gk | 5.9e-16 / 6.1e-16 | 6.7e-15 / 6.9e-15 | 0/48 each | yes |
| f64 | φ, ∇φ | SIL sqrt+div / gk | 2.0e-16 / 2.2e-16 | 3.4e-15 / 3.5e-15 | 0/48 each | no |
| f64 | φ, ∇φ | relaxed S2 / P4 / N1+P2 | 6.5e-12 / 6.1e-14 / 3.3e-16 | 7.4e-11 / 4.2e-13 / 1.5e-15 | 0/18 each | |

The only failed prototype check is TIL `sqrt+div`, f32 with gradients, W2 N = 1,000 with targets
equal to sources. It fails at 1.0e-6 against 1e-6; the reference is at 9.6e-7 on the same cell.
See the proposal on requirement 2 above. Targets in lanes has the reference's error level
everywhere, because it adds in the same order. Sources in lanes and green-kernels are about twice
as accurate on the sums, because they add W partial sums.

## Other findings

- **The reference baseline is bimodal.**
  - In f64 potential-only, `nd_fmm_ref::p2p` runs at about 0.34 or about 0.86 Gpairs/s on the
    same data, depending on the run. A repeated probe reproduced it: 0.336 and 0.855 on the same
    cell. Four cells of the final run caught the fast mode, in f64 (W1 n_t = 128 gathered,
    W2 N = 10⁴ t = s) and f32 (W2 N = 10³ t ≠ s, W2 N = 10⁴ t = s), all potential-only.
  - The cause is in its compiled loop. The potential is kept in memory across the source loop
    (`ldr`, `fadd`, `str` on the same address for every pair), so its speed is set by store-to-load
    forwarding, which the M3 resolves at different speeds depending on the addresses.
  - nd-fmm-ref is out of scope, so this is recorded, not fixed. T7 should report the reference
    with this caveat, or take the slow mode as the baseline. The ratios against the reference in
    this report use the measured values as they are.
- **Per-pair against gathered.** For the recommended prototypes, the per-pair form is 4%
  (n_t = 128) to 39% (n_t = 8) slower than the gathered form; the cost is the per-call block
  set-up. This is the gain T6 measures
  by gathering a target's near field into one call.
- **NEON intrinsics need `unsafe` without `#[target_feature]`.** rustc 1.98 reports that "the neon
  target feature being enabled in the build configuration does not remove the requirement to list
  it in `#[target_feature]`". The vector layer of `nd-fmm-simd` (T3, T4) should mark its NEON entry
  points `#[target_feature(enable = "neon")]` to use the value intrinsics safely (design §5.3).
- **Identical functions are merged**, so a disassembly check by symbol name (T5) must expect an
  entry point to share its code, and its symbol, with an identical one.
- **The AVX2 f64 estimate route keeps the domain safe.** r² ∈ [2⁻¹⁰⁸, 2⁷] converts to a normal
  f32, as CONVENTIONS §3.13 requires.

## Files

- `Cargo.toml`, `build.rs` (records the compiler version).
- `src/main.rs`: the report binary.
- `src/simd/`: vector layer, NEON and AVX2 types, the candidates.
- `src/kernels.rs`: prototypes and entry points.
- `src/rsqrt_study.rs`, `src/workloads.rs`, `src/bench.rs`, `src/green.rs`, `src/machine.rs`,
  `src/rng.rs`.
- `src/tests.rs`: the smoke test. For one W1 cell per available ISA and precision it checks every
  prototype against `direct_sum`, checks targets in lanes per-pair = gathered bit for bit, and
  checks the kernels' inverse square roots on a strided subset.
- `inner_loops.py`: disassembly counts.
- `results-m3max.md`: raw output of the final run.

Added in Phase 3S T7 (docs/phase3s/T7-benchmarks.md):

- `examples/compare.rs`: the production kernel of `nd-fmm-simd` against green-kernels, on
  the inputs of `nd-fmm-validate`'s `p2p_kernels` example (the W1 and W2 inputs of this
  spike), with a smoke test of its core that `cargo test -p nd-fmm-spike-p2p-simd` runs.
  The spike now depends on `nd-fmm-simd` and `nd-fmm-validate` (which links MPI;
  `compare` never initialises it).
- `results-m3max-final.md`: the T7 runs on the M3 Max (`p2p_kernels`, `compare` and the
  FMM runs of `p2p_fmm`).

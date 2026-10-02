# Phase 3S / T5 — nd-fmm-simd: the SIMD P2P kernel (C3S.4)

The heart of the phase: the Laplace P2P kernel with potential and gradient, in f32 and
f64, on NEON and AVX2 + FMA, behind `P2pKernel::evaluate`. It is written once,
generic over the T4 vector layer, in the loop order and register blocking signed off
from the spike.

Read first: root CLAUDE.md, fmm-simd/CLAUDE.md; docs/design/simd-p2p.md §3, §4 and §7;
CONVENTIONS §3.1 and §3.13 with the signed-off T1 addition (coincident pairs and the
domain of r²); spikes/p2p-simd/SPIKE_REPORT.md and its recommendation (loop order, K per
ISA and precision, the expected throughput); docs/phase3s/README.md; the T3 and T4 code;
`nd_fmm_ref::p2p` and its tests (Phase 1 T4).

Do:
- The kernel body (design §4.1, with the order signed off from T2), generic over the
  vector layer and a const K:
  - per block of K·W targets, load positions, potentials and, with gradients, the
    gradients from the §3.13 layout;
  - for each source in input order: broadcast, d = x − y, r² with fma, the inverse
    square root of T4 with the r² = 0 mask, then φ and g as in design §4.2;
  - store the block.

  Two variants, potential only and with gradients, chosen by whether `gradient` is
  `Some`.
- Tails: the last block runs the same vector code on a padded stack copy (or AVX2 masked
  loads and stores, `_mm256_maskload`/`maskstore`); never a scalar tail (design §4.5).
  With fewer targets than K·W, smaller K is allowed if it is the same instruction
  sequence per lane. Show this by the target-position invariance test.
- Entry points per (ISA, precision, potential or gradient) with `#[target_feature]`,
  and `P2pKernel::evaluate` dispatching to them. The scalar ISA runs the same generic
  body at W = 1, or the T3 loop if the generic body at W = 1 is not bit-identical to it.
  Say which, and keep the other as a test.
- Panics and documentation as `nd_fmm_ref::p2p`: length checks, the formula, the
  exclusion rule (r² = 0, §3.13), the order of accumulation, the accuracy contract, the
  domain, and that results differ between ISAs within that contract.
- Inlining check, as in T4, for every kernel entry point: no call in the source loop.
  Report the instruction counts of the loop body per ISA, precision and variant, and
  compare them with the operation counts of design §4.2. For AVX2, which nothing
  times, this is the performance check: a count more than 25% above the model
  is investigated (spills, shuffles, missed fma) and reported. Build the x86_64
  disassembly on the Mac with `--target x86_64-apple-darwin`, or take it from a CI
  artifact.
- Throughput check against the spike, on NEON only: time the production kernel on the
  spike's W1 gathered workload on the M3 Max, and compare with the spike's best
  prototype of the same order and K. The target is at least 90%. A shortfall is
  investigated (disassembly, register spills) and reported; it is not a test failure.

Tests that define done (on every available ISA, f32 and f64, potential and gradient;
each test prints the ISAs it ran):
- Terms: one source and one target over seeded separations across the domain. The
  potential within 8 u_T of `nd_fmm_ref::p2p`, relative; each gradient component within
  16 u_T relative to |q| / r².
- Sums: seeded sets with n_s ∈ {1, 7, 64, 1000, 4096} and every n_t from 0 to 3·K·W + 1,
  within 1e-14 (f64) and 1e-6 (f32) of `direct_sum`, relative to the sum of term
  magnitudes. Also a cancelling set (charges summing to nearly zero), and FMM-shaped
  leaf-scaled sets (a target leaf and its 26 neighbours, mapped as `LaplaceOperator`
  maps them).
- Coincident points: targets equal to sources (the same slice), duplicated sources, a
  mapped neighbour source that rounds onto a target. The results are finite and equal
  the sum over the non-coincident pairs, to the tolerance above.
- Domain ends: pairs at the smallest and largest r² of the signed-off domain give
  finite results within tolerance.
- Accumulation onto nonzero initial values; empty sources and targets are no-ops; the
  length checks panic.
- Invariance, bit for bit:
  - chunk invariance (sources split at every k for small sets, at random k for large
    ones);
  - target-position invariance (each target alone, and in shuffled and shifted slices,
    gives the same bits);
  - determinism (repeated calls).
- Cross-ISA: on a machine with several ISAs, every ISA agrees with every other within
  twice the tolerances above.
- Property tests (`proptest`) over random point sets, charges and lengths for the
  invariances and the sums.

Must pass: `cargo test -p nd-fmm-simd` (debug under 30 seconds) and
`cargo test -p nd-fmm-simd --release -- --ignored`:
- on the M3 Max (NEON, and AVX2 under Rosetta 2 as a quick check if T3 found that it
  works);
- in the `run-tests-simd` CI job on both legs, which is where the x86_64 results count
  (AVX2).

Also `cargo clippy -p nd-fmm-simd --all-targets -- -D warnings`, the cross-target clippy,
`cargo doc -p nd-fmm-simd --no-deps` without warnings, and the root checks. Report the
ISAs run on each machine and CI leg, the maximum measured errors per ISA (terms in u_T,
sums), the loop instruction counts and the NEON throughput check. State that the x86_64
paths were not timed.

Do not:
- change nd-fmm-exec (T6), §3.13 or `nd_fmm_ref`;
- change the loop order, the inverse-square-root formulation or K from what was signed
  off without asking first, even if a variant measures faster. Report it instead;
- allocate, spawn threads, or read environment variables in the kernel;
- relax a tolerance to make a path pass. If one fails, find whether the term formula,
  the inverse square root or the order is at fault, report it and stop.

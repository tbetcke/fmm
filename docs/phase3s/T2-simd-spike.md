# Phase 3S / T2 — spike: loop order, inverse square root and the green-kernels baseline (C3S.2)

Before writing the production kernel, measure the choices it rests on: the loop order
(targets in lanes or sources in lanes), the formulation of the inverse square root per
ISA and precision, and the register blocking. Also measure where green-kernels stands on
the same machines and workloads. It is the Phase 3S counterpart of the Phase 0 CubeCL
spike: throwaway code, a report, and a recommendation that is signed off before T4.

The development machine is an Apple M3 Max (NEON), and it is the only machine that
times anything: no x86_64 machine is available (docs/phase3s/README.md, "Machines").
The x86_64 choices are therefore made from documented error bounds and operation counts,
and confirmed for accuracy later by the CI tests of T4. The spike still builds and runs
on x86_64, so its timings can be added there later with one command.

Read first: root CLAUDE.md; docs/design/simd-p2p.md (all of it; §2.2, §4 and §8
closely); docs/phase3s/README.md; spikes/cubecl-gemm/ (spike conventions,
SPIKE_REPORT.md); the green-kernels sources at commit
`7d757c579f8633d58163cd5b2d011a9468541f17` (`src/laplace_3d.rs`, `src/traits.rs`,
`benches/laplace_f64.rs`), and `simd_approx_recip_sqrt` in rlst's `src/simd.rs`, both
in rlst 0.9.0 (`~/.cargo/registry/src/*/rlst-0.9.0/`) and in the git `main` that
green-kernels resolves to; `nd_fmm_ref::p2p`; `nd_fmm_validate::bench`; the Intel
Intrinsics Guide and the Arm intrinsics reference for the instructions of design §4.3.

Do:
- `spikes/p2p-simd/` (package `nd-fmm-spike-p2p-simd`, `publish = false`), in
  workspace members but NOT default-members.
  - Add `green-kernels = { git = "https://github.com/bempp/green-kernels", rev =
    "7d757c579f8633d58163cd5b2d011a9468541f17" }` to [workspace.dependencies], with a
    comment that only spikes may use it, and use it from the spike only.
  - The spike may also depend on `nd-fmm-ref` (oracle) and `nd-fmm-math`.
  - **First** check the build: the default members unchanged (`cargo build`), the spike
    with green-kernels and rlst 0.9.0 in one binary (`cargo build --release -p
    nd-fmm-spike-p2p-simd`). Report which crates `Cargo.lock` gained. If it does not
    build or link, stop and report the error with the options of design §8.4. Do not
    pick one on your own.
  - Unsafe is allowed in the spike's prototypes; comment each block anyway.
- **Inverse square root, NEON.** For each precision, implement the candidates of design
  §4.3 on NEON:
  - the estimate with 1, 2 and 3 Newton steps, in the fma form and in the `vrsqrtsq`
    form;
  - the polynomial correction from the estimate, at the truncation orders that design
    §4.3 suggests;
  - `sqrt` then `div`.

  For each candidate, measure:
  - the maximum relative error in units of u_T: f32 exhaustively over [1, 4), f64 on
    10⁷ seeded log-uniform samples over the domain of design §4.4, plus powers of two,
    their neighbours and the domain ends;
  - the behaviour at r² = 0 and at a subnormal input;
  - throughput and latency (independent streams, and a dependent chain), in ns per
    element and cycles per vector.
- **Inverse square root, AVX2 (analysis, no timing).** For AVX2 f32 and f64 (the
  f32-estimate route, an integer bit-trick initial guess), list the same candidates
  with:
  - the worst-case relative error derived from the documented estimate bound
    (≤ 1.5 · 2⁻¹² for `rsqrtps`) and the rounding of each step;
  - the operation count and dependency-chain length;
  - the cheapest candidate whose derived bound stays a factor 2 below 4 u_T.

  Write the x86_64 prototypes too, so that they compile (cross-target clippy). Run them
  under Rosetta 2 against `direct_sum` for correctness, if T3's check or your own shows
  it works. Rosetta emulates the estimates, so do not report errors measured there as
  the hardware's.
- **Prototype kernels**, potential only and with gradients, f32 and f64, timed on NEON
  (and written for AVX2, for correctness and instruction counts only; AVX-512 is
  deferred):
  - targets in lanes (design §4.1) with K ∈ {1, 2, 4} target vectors per block;
  - sources in lanes (green-kernels' order) with 1, 2 and 4 targets per block;
  - each with the inverse square root that meets the 4 u_T contract most cheaply on that
    ISA, and with the green-kernels formulation for comparison;
  - for the x86_64 prototypes, the inner-loop instruction count from the release
    disassembly (`--target x86_64-apple-darwin` builds on the Mac), against design §4.2,
    and the lane utilisation at the W1 leaf sizes.

  Check every prototype against `direct_sum` (relative to the term magnitudes) before
  timing it.
- **Workloads** of design §8.2: W1 (FMM-shaped, n_t ∈ {8, 16, 24, 32, 64, 128}, per-pair
  and gathered forms, a pool of 64 sets) and W2 (all-pairs, N ∈ {10³, 10⁴}, targets
  distinct from and equal to the sources). Also time `nd_fmm_ref::p2p` on every cell:
  that is the Phase 3 baseline.
- **green-kernels baseline.** `Laplace3dKernel::<T>::evaluate_st` with `Value` and
  `ValueDeriv` on the same inputs, f32 and f64, single-threaded:
  - in its layout: interleaved triples, and four interleaved outputs for `ValueDeriv`;
  - on NEON only; report the pulp backend it dispatched to;
  - measure its accuracy against `direct_sum` after removing its 1/(4π), relative to the
    term magnitudes.
- One binary that prints the whole report as Markdown: the machine (`bench::cpu_model`,
  `cores`, `target`), the toolchain, the ISAs detected, then every table. `--quick`
  runs a reduced set for a smoke check. Keep the raw output of each machine as
  `results-<machine>.md`.
- `spikes/p2p-simd/SPIKE_REPORT.md`:
  - the inverse-square-root tables: on NEON, candidate, measured error in u_T and
    throughput; on x86_64, candidate, derived error bound and operation count;
  - the NEON kernel tables: pairs per second per (precision, output, workload, loop
    order, K), with green-kernels and `nd_fmm_ref::p2p` beside them and the ratios;
  - the x86_64 prototypes' instruction counts against design §4.2;
  - the fraction of the design §4.6 model reached, with the model corrected where
    measurement shows a port limit (for example an estimate instruction that issues on
    one pipe only);
  - lane utilisation at the W1 leaf sizes;
  - the accuracy of every timed row;
  - the green-kernels build check and what it brought into `Cargo.lock`;
  - a note that no x86_64 path was timed, and that the x86_64 formulations rest on
    documented bounds until T4's CI tests confirm them.
- **Recommendation**, for sign-off before T4:
  - the loop order, by the decision rule of design §4.1 on NEON, stated with the
    numbers;
  - per ISA and precision: the inverse-square-root formulation that meets the 4 u_T
    contract, and K;
  - whether a relaxed f64 level is worth shipping: its measured NEON gain at the
    FMM-shaped workload, and its error. The design recommends not shipping one unless
    the gain is large;
  - the expected NEON throughput of the production kernel per precision, which T5 must
    reach to within 90%, and the expected x86_64 instruction counts per loop.

Tests that define done:
- Every prototype passes its `direct_sum` check before it is timed (a failed check
  aborts that row, and is reported).
- A smoke test of the spike's core (one W1 cell per available ISA and precision, every
  prototype against `direct_sum`) in `cargo test -p nd-fmm-spike-p2p-simd`, on the M3
  Max and, under Rosetta 2 if it works, for the AVX2 prototypes.

Must pass: `cargo build` and the root checks for the default members (unchanged by this
task apart from the lock file); `cargo clippy -p nd-fmm-spike-p2p-simd --all-targets --
-D warnings`; `cargo clippy -p nd-fmm-spike-p2p-simd --all-targets --target x86_64-apple-darwin --
-D warnings`; `cargo test -p nd-fmm-spike-p2p-simd`; `cargo run --release -p
nd-fmm-spike-p2p-simd` on the M3 Max. Run with every BLAS thread variable set to 1, and
say so.

Do not:
- add the spike to default-members or to CI, or let a default member depend on
  green-kernels;
- write production code in `fmm-simd/`, or change any other crate;
- assert timings;
- commit CSV or other generated output beyond the `results-<machine>.md` reports.

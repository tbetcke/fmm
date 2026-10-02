# Phase 3S / T7 — benchmarks against the reference and green-kernels, FMM timings (C3S.6)

Measure what the phase delivered, on the Apple M3 Max (NEON), the only machine that
times anything (no x86_64 machine is available; docs/phase3s/README.md, "Machines").
That covers three things:
- the SIMD kernel against `nd_fmm_ref::p2p` and against the Laplace kernels of
  green-kernels, at stated accuracy, on NEON;
- the effect on the complete FMM;
- the leaf size, which a faster P2P moves. T7 adopts its own recommendation (approved
  in advance on 2026-10-02).

The task ends the phase with the design-document update, as Phase 3 T12 did.

Read first: root CLAUDE.md, fmm-validate/CLAUDE.md, fmm-simd/CLAUDE.md;
docs/design/simd-p2p.md §4.6, §8 and §9; docs/phase3s/README.md ("Exit gate" C3S.6,
"Machines", "Exit checklist"); spikes/p2p-simd/ (the T2 harness, workloads and
`SPIKE_REPORT.md`); design §7, Phase 3 ("Per-pair cost", "Leaf size", "Recommendation
for Phase 4") and §8.3; `nd_fmm_validate::{bench, calibration}`; the T5 and T6 reports.

Do:
- **Kernel benchmark in the workspace**: an `nd-fmm-validate` example `p2p_kernels`
  (release; no MPI needed in its code path, although the crate links MPI). It times
  `P2pKernel` on every available ISA against `nd_fmm_ref::p2p`:
  - the workloads W1 (both forms) and W2 of design §8.2, f32 and f64, potential only
    and with gradients, with the conventions of design §8.1;
  - per cell: pairs per second, the speed-up over the reference, the fraction of the
    design §4.6 model (as corrected by the spike), and the accuracy against
    `direct_sum` (relative L2 and max, relative to the term magnitudes);
  - the machine, toolchain and detected ISAs at the top. `--quick` runs a reduced set.
- **green-kernels comparison**: a spike example `compare` in `spikes/p2p-simd` (the only
  place green-kernels may be used), with `nd-fmm-simd` as a dependency:
  - the default `P2pKernel` (`detect()`) and every other available ISA against
    `Laplace3dKernel::evaluate_st` (`Value` and `ValueDeriv`), on the same inputs as
    `p2p_kernels`;
  - pairs per second, our time over theirs, and both accuracies;
  - which pulp backend green-kernels dispatched to;
  - green-kernels' 1/(4π) and output layout handled as in design §8.3, with the
    conversion of our gradients excluded from the timing.
- **FMM**, with `fmm_accuracy` or a new `nd-fmm-validate` example, as you see fit:
  - the T12 cube and Plummer problems (N = 10⁵, 64 points per leaf, `max_level` 16),
    p = 3 and 8, f32 and f64, with gradients;
  - `--p2p reference` against `--p2p auto` (and each other ISA);
  - at 1 thread and at the number of performance cores;
  - stage timings (upward, downward, leaves), evaluation time, the speed-up of the leaf
    stage and of the evaluation, and the errors (which must match T6's).
- **Leaf-size study**: the T12 study (`calibration::leaf_study`, p = 8 in f64, cube and
  Plummer, `max_points_per_leaf` ∈ {16, 32, 64, 128, 256}) repeated with `Auto`, plus
  p = 3, in f32 and f64, at 1 thread and at the number of performance cores. Report the
  near/far balance and the fastest leaf size per distribution, p and precision, next to
  T12's numbers, and the φ and ∇φ errors at each size.
- **Leaf-size default.** The recommendation follows this rule, fixed now:
  - take the size from {16, 32, 64, 128, 256} with the smallest geometric mean of
    one-thread evaluation time over cube and Plummer, p = 3 and 8, f32 and f64;
  - keep 64 unless that size is at least 5% faster by this measure, and no relative
    L2 error of φ or ∇φ at it is more than 10% worse than at 64.

  If the rule picks a new size:
  - change `DEFAULT_MAX_POINTS_PER_LEAF` in nd-fmm-exec, in its own commit, with its
    docs;
  - pin every Phase 3 test, example and calibration configuration that relied on the
    default to mean 64 to 64 explicitly, so the published Phase 3 numbers stay
    reproducible;
  - re-run the nd-fmm-exec and nd-fmm-validate tests, including the ignored release
    gates, and report them.

  State in the PR and in design §7 that the choice rests on M3 Max timings only.
- **Runs**: on the M3 Max only, with all BLAS thread variables set to 1. Paste the
  output into the PR, and keep it as `spikes/p2p-simd/results-m3max-final.md`. Both
  examples build and run on x86_64 (check with the cross-target clippy), so x86_64
  numbers can be added later; the report states that none were taken.
- **The performance target** (C3S.6): in every cell of W1 (both forms) and W2, on NEON,
  the default kernel reaches at least green-kernels' pairs per second, at equal or
  better accuracy. For every cell below it:
  - analyse the cause: lane utilisation, the loop's instruction count against design
    §4.2, the inverse square root's share, memory;
  - report it for a decision.

  Do not change the kernel in this task.
- **Design documents**, with the Phase 3S outcome:
  - laplace-fmm-plan.md:
    - the revision note at the top;
    - §7, Phase 3S: status per component C3S.1–C3S.6, with the measured errors (from
      T4–T6), the kernel tables (ours, reference, green-kernels), the FMM speed-ups and
      the leaf-size study;
    - §7, the "Recommendation for Phase 4": the host P2P baseline the GPU kernel (C4.2)
      is compared with, and how the stage shares at p = 3 and 8 have moved;
    - §8.3 (P2P benchmarking as done);
    - §9.1 (the Phase 3S risks, with what was found);
    - §9.2 (the questions answered: arm64 CI, x86 machines, relaxed level, leaf size;
      x86_64 timings remain open until a machine is available);
  - docs/design/simd-p2p.md: the decisions as taken (loop order, inverse square root
    per ISA, K), the measured numbers replacing the models where they differ, and its
    §9 questions answered;
  - workspace-structure.md: §3 and §3.1 (the built surface of `nd-fmm-simd`, the
    `nd-fmm-exec` additions, the new examples), §6 (Phase 3S done);
  - docs/phase3s/README.md: tick the exit checklist.

Tests that define done:
- A smoke test of the `p2p_kernels` core (one small W1 cell per available ISA, checked
  against `direct_sum`) in `nd-fmm-validate`'s tests. It needs no MPI initialisation.
- `cargo test -p nd-fmm-spike-p2p-simd` still passes, including a smoke run of
  `compare`'s core.

Must pass: `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`; the root checks and
the stricter workspace checks; `cargo run --release -p nd-fmm-validate --example
p2p_kernels`; `cargo run --release -p nd-fmm-spike-p2p-simd --example compare`; the FMM
runs above, on every machine listed.

Do not:
- assert timings, or commit CSV or other output beyond the `results-*.md` reports;
- change nd-fmm-simd, or nd-fmm-exec beyond the leaf-size default above;
- make a default member depend on green-kernels;
- add criterion or any other dependency without asking.

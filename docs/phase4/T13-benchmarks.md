# Phase 4 / T13 — device benchmarks, leaf size, design-document update (gate: benchmarks published)

This task measures what the phase delivered, on the only GPU available: Metal, f32, on
the Apple M3 Max (docs/phase4/README.md, "Machines"). It covers four things:
- the device FMM against the host path, stage by stage;
- the kernels against the device's peak and against the spike;
- the leaf size, which a batched GPU path moves (design §8.3);
- what was not measured, stated plainly: f64 on a GPU, and CUDA.

The task ends the phase with the design-document update, as Phase 3 T12 and Phase 3S
T7 did.

Read first:
- root CLAUDE.md, fmm-validate/CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase4/README.md ("Timing", "Peaks", "Exit gate", "Decisions to sign off"
  questions 8 and 9, "Exit checklist");
- docs/design/device-path.md (section 13, and every decision as signed off);
- docs/design/laplace-fmm-plan.md §6, §7 (Phase 3 "Per-pair cost" and "Leaf size",
  Phase 3S "FMM" and "Leaf size") and §8.3;
- spikes/cubecl-gemm/SPIKE_REPORT.md (the 0.10.0 and 0.11 tables);
- spikes/device-arith/REPORT.md;
- the T6–T12 reports;
- `nd_fmm_validate::{bench, calibration, p2p_kernels}` and the `p2p_fmm` example
  (Phase 3S T7: the stage timings and the leaf-size rule).

Do:
- **FMM benchmark**: an `nd-fmm-validate` example `device_fmm` (behind `gpu`), or
  `p2p_fmm` extended, as you see fit.
  - Problems: the T12 cube and Plummer problems (N = 10⁵, `max_level` 16, 64 points per
    leaf, eight charge vectors, gradients), and N = 10⁶ on the cube if it fits in
    memory.
  - Points: f32 at p = 3, 6 and 8 with the autotuned strategy, and with each fixed
    strategy.
  - Backends: Metal against the host path (`P2pChoice::Auto`, NEON) at 1 thread and 12
    threads.
  - Per run:
    - stage times (upward, downward, leaves, with M2L and P2P separately where the
      design's timing allows);
    - evaluation time and speed-up over the host;
    - build time, with table loading and tuning separate;
    - transfers, launches and syncs per evaluation;
    - the errors, which must equal T11's.
- **Kernel efficiency** on Metal f32, collected from the T6–T10 harnesses into one
  report:
  - M2L GEMM GFLOP/s and % of peak per p and batch size, against the spike (0.11 run);
  - M2M/L2L GEMM likewise;
  - P2P pairs per second against host NEON and the peak model of the design;
  - the leaf kernels' times;
  - rotation against dense per p, with the efficiency against the spike's break-even
    values.
- **Leaf-size study** on the device: the Phase 3S T7 study (`max_points_per_leaf` ∈
  {16, 32, 64, 128, 256}, cube and Plummer, p = 3 and 8, f32) on Metal. Report the
  near/far balance, the fastest size per distribution and p, and the φ and ∇φ errors
  at each size, beside the host's numbers from Phase 3S T7.
- **Device leaf-size default** (README, decision 8; recommended to apply this rule to
  the device backend only). The rule is fixed now:
  - take the size from {16, 32, 64, 128, 256} with the smallest geometric mean of Metal
    evaluation time over cube and Plummer, p = 3 and 8;
  - keep 64 unless that size is at least 5% faster by this measure, and no relative
    L2 error of φ or ∇φ at it is more than 10% worse than at 64.

  If the sign-off of decision 8 approved a device default and the rule picks a new
  size, add it as the device backend's default, in its own commit, with docs. The host
  default (`DEFAULT_MAX_POINTS_PER_LEAF` = 64) does not change. State that the choice
  rests on M3 Max Metal timings only.
- **The CUDA run, documented**: one command that runs `device_fmm`, the kernel harness
  and the autotune table on a CUDA machine in f32 and f64. Name the numbers it would
  settle: the f64 GEMM efficiency, the f64 dense/rotation crossover and the static
  rule. Check that it type-checks (`--features cuda`). Do not claim any result from
  it.
- **Runs**: on the M3 Max only, release, outside the sandbox (Metal), with every BLAS
  thread variable set to 1, following the README's timing conventions. Paste the
  output into the PR and keep it as `fmm-validate/results/phase4-m3max.md`. If you
  would rather keep results under `spikes/` as before, say why.
- **Design documents**, with the Phase 4 outcome:
  - laplace-fmm-plan.md:
    - the revision note at the top;
    - §4 (the GPU defaults as measured in f32, and the f64 rule still provisional);
    - §6.1 and §6.2 (CubeCL 0.11.0-pre.4, the f64 capability per backend, the CPU
      runtime facts from T2);
    - §6.4–§6.7 (the kernel mapping, batching and autotune as built);
    - §7, Phase 4: status per component C4.1–C4.8 with the measured errors and the
      benchmark tables, and the "Recommendation for Phase 5";
    - §8.1 and §8.3 (the device test layers and benchmarking as done; the CPU-runtime CI
      job as decided);
    - §9.1 (the Phase 4 risks, with what was found);
    - §9.2 (the questions answered: the CubeCL pin, GPU hardware, the f64 default still
      open without a CUDA run, the leaf size);
  - docs/design/device-path.md: the decisions as taken, and the measured numbers
    replacing the models where they differ;
  - workspace-structure.md: §2 and §3 (`nd-fmm-kernels` as built; whether it is a
    default member), §3.1 (the built surfaces of `nd-fmm-kernels` and the device
    additions of `nd-fmm-exec` and `nd-fmm-validate`), §5 (the CubeCL pin) and §6
    (Phase 4 done);
  - docs/phase4/README.md: tick the exit checklist.

Tests that define done:
- A smoke test of the `device_fmm` core in `nd-fmm-validate` (behind `gpu` and `cpu`): one
  tiny problem on the CPU runtime against the host path, within the README's bounds.
  It initialises MPI only if the harness needs it, in its own test executable.
- The T6–T12 test suites still pass.

Must pass:
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`, without features and with
  `--features cpu --release`;
- `cargo run --release -p nd-fmm-validate --features metal --example device_fmm` (or
  the extended `p2p_fmm`) on the M3 Max;
- `cargo check -p nd-fmm-validate --features cuda`;
- clippy on `nd-fmm-validate`, without features and with `--features cpu,metal`;
- the root checks and the stricter workspace checks.

Do not:
- assert timings, or commit CSV or other output beyond the results report;
- change kernels, the device operator or autotune beyond the device leaf-size default
  above. A defect found here is reported, and fixed in its own commit with its test;
- change the host defaults;
- add criterion or any other dependency without asking;
- present a model as a measurement. Every number is labelled "measured (machine,
  backend)" or "model".

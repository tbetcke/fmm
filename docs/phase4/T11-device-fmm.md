# Phase 4 / T11 — the device FMM end to end (C4.8, new)

After T6–T10 every operator kind has a device kernel. This task makes the device path
what the design promises:
- every kind on the device by default;
- data resident for the whole evaluation;
- many launches per sync, and one sync per evaluation if the design achieves it;
- the small top levels handled as the design decides.

Then it runs the phase's accuracy gate: the GPU result equals the CPU FMM. C4.8 is new
in Phase 4. Design §7 lists C4.1–C4.7 as kernels and infrastructure, but the gate "GPU
result equals CPU FMM" needs its own task, as C3.2 did for the host path.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase4/README.md ("Requirements", "Accuracy measures", "Workloads", "Exit gate"
  C4.8);
- docs/design/device-path.md, signed off: sections 4, 7, 8 and 9;
- docs/design/laplace-fmm-plan.md §6.5 (batching, launch overhead, top levels) and §7
  ("Recommendation for Phase 4": the workloads and expected errors);
- docs/phase3/README.md ("Error measures", "Predictions", the C3.2 and C3.3 gates);
- the T5–T10 code and reports;
- fmm-exec/tests/{mpi_exec, accuracy, adaptive}.rs;
- nd-fmm-validate's `fmm_accuracy` and `calibrate` examples.

Do:
- **Defaults**: every operator kind runs on the device when the backend is a device.
  The host fallback per kind stays as a test aid (requirement 8). `Fmm` reports which
  kinds run where; in the default configuration that is all of them on the device.
- **Launch scheduling**, as the design's section 8 specifies:
  - queue every level call's launches without intermediate syncs, so that an
    evaluation syncs only where the design requires (the target is once, at the
    download of the output);
  - the policy for top levels with few boxes (merged launches, or a fixed host
    fallback for levels below a threshold), fixed at build and reported;
  - count launches and syncs per evaluation, and expose them beside the transfers.
- **Transfers at the minimum**: per evaluation, the charges up and the output down,
  plus whatever else the design's formula states. Check that no fallback transfer
  remains in the default configuration.
- **Stage timings** on the device as the design specifies, without adding syncs to
  normal evaluations. A synchronised reporting mode is opt-in.
- `nd-fmm-validate`: `fmm_accuracy` (and `calibrate`, where useful) accepts
  `--backend host|cpu|metal` (under the `gpu` features), and prints the backend and
  device, the kinds on the device, the resolved strategies and GEMMs, and transfers,
  launches and syncs per evaluation.
- Crate docs: the device path as delivered (residency, scheduling, determinism,
  differences from the host path, errors).

Tests that define done (each prints the backends it ran):
- **Debug scenarios** (CPU runtime, f32 and f64): every `tests/mpi_exec.rs` scenario
  with every kind on the device:
  - within the README's FMM bounds of the host output (relative L2 1e-12 in f64, 1e-5 in
    f32, φ and ∇φ);
  - bit-identical across two evaluations and across two `Fmm` builds of the same
    input;
  - transfers per evaluation equal to the design's minimum; syncs per evaluation as
    designed.
- **The C4.8 gate** (ignored release tests, in `tests/accuracy.rs` and
  `tests/adaptive.rs` or a new executable with its own single MPI test). On the uniform
  cube and the Plummer sphere (N = 10⁵, `max_level` 16, 64 points per leaf, eight
  charge vectors, gradients), at the README's workload points:
  - f32 at p = 3 and 8 on Metal, by hand;
  - f64 at p = 8, 12 and 18 on the CPU runtime (N = 10⁴ where N = 10⁵ exceeds the
    README's time limit; say which);
  - each run checks the device output against the host output of the same settings,
    within the FMM bounds, and the errors against `direct_sum` within 0.1% (f64) and 5%
    (f32) of the host run's;
  - the C3.2 gate (twice the single-translation prediction at p = 3, 8 and 18) and the
    C3.3 gate (adaptive within twice the uniform error) pass on the device path where
    their points are run.
- The optional third distribution, the Gaussian clusters, at f32 p = 8 on Metal, if
  time allows, reported.
- The host path is unchanged: the existing tests pass without features.

Must pass:
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` without features (unchanged),
  `--features cpu --release`, and `--features cpu --release -- --ignored` (the f64
  gates); `--features metal --release -- --ignored` by hand (the f32 gates);
- `cargo test -p nd-fmm-kernels --features cpu --release`;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`, and `cargo run --release -p
  nd-fmm-validate --features metal --example fmm_accuracy -- --backend metal` on the
  M3 Max;
- clippy on the three crates, without features and with `--features cpu,metal`;
  `cargo check -p nd-fmm-exec --features cuda`; `cargo doc` without warnings;
- `tests/mpi_exec.rs` with `--features cpu` on 2 ranks, by hand, under an external
  timeout;
- the CPU-runtime CI job, if kept; the root checks and the stricter workspace checks.

Report:
- a table of φ L2, φ max and ∇φ L2 per distribution, precision and p for host and device
  side by side, with the device/host ratio;
- the device-vs-host output difference;
- transfers, launches and syncs per evaluation;
- the evaluation time on the device against the host at 1 and 12 threads (reported, not
  asserted; T13 measures properly).

If a gate fails, isolate the operator kind with the host fallback, report the breakdown
and stop.

Do not:
- overlap P2P with the far field on another stream, or change any accumulation order;
- change kernels beyond what scheduling needs. A kernel defect found here is reported,
  and fixed in its own commit with its test;
- change the host path or its defaults, or the M2L default rule (T12).

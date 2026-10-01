# Phase 3 / T12 — nd-fmm-validate: calibration of p against accuracy (C3.4)

Design §4 starts from the Gumerov–Duraiswami figures (relative L2 errors near 1e-4,
1e-7 and 1e-10 at p = 3, 8 and 18) and asks Phase 3 to recalibrate them for this
library. This task measures the error of the full FMM as a function of p, precision
and distribution, and publishes the smallest p for each target accuracy. It also
records what the per-pair path costs, as context for Phase 4. It ends the phase with
the design-document update.

Read first: root CLAUDE.md, fmm-validate/CLAUDE.md, docs/phase3/README.md ("Exit
gate", C3.4; "Exit checklist"); design §4, §7 (Phases 1–3), §8.2 and §8.3; the T9,
T10 and T11 code and their reports; docs/design/fmm-plan-redesign.md; docs/CONVENTIONS.md §3.9 (the tested range, p ≤ 20 for M2L).

Do:
- Example `calibrate` (release; one rank; not registered with templated-examples).
  For each of the four distributions (cube, sphere surface, Plummer, Gaussian
  clusters), N = 10⁵, sources equal to targets, charges in [−1, 1), `max_level` 16 and
  the `max_points_per_leaf` of T11, and `--threads n` (T10):
  - f64 for p = 1..=20 with `Auto`, f32 for p = 1..=8;
  - the relative L2 and max error of φ and ∇φ over 1,000 sampled targets, with the
    direct sum computed once per distribution;
  - the stage timings and the total;
  - a Markdown table per distribution and precision, then the calibration table: for
    each target 10⁻ᵏ, k = 3..=12, the smallest p whose relative L2 error of φ is below
    it, per distribution and as the worst over all four; the same for ∇φ. A target
    not reached at p ≤ 20 (f64) or p ≤ 8 (f32) is printed as "> 20" or "> 8"; f32 also
    prints its floor;
  - the machine (CPU model, cores), the thread count and the `ThreadingReport` (T10),
    as in the Phase 2 examples. Run with all BLAS thread variables set to 1.
    Threads change only the run time: the error columns are bit-identical for every
    thread count (T10), so run the sweep threaded.
  
  If the full run takes more than about 30 minutes, reduce N for the sweep, check the
  chosen p at N = 10⁵, and say which N each number used.
- A short leaf-size study, at p = 8 in f64 on the cube and the Plummer sphere:
  `max_points_per_leaf` ∈ {16, 32, 64, 128, 256}, printing error and stage timings.
  The error should barely move; the timings show the near-field/far-field balance of
  design §8.3 for the per-pair path. Report, do not tune defaults.
- A smoke test of the calibration core at N = 500, p ≤ 3 (in the executable's one
  MPI-initialising test).
- Update the design documents with the Phase 3 outcome, as Phase 2 T8 did:
  - laplace-fmm-plan.md: the revision note at the top; §4 (the calibrated starting
    points of p replace "to be recalibrated in Phase 3"); §5.2 (extension 1 done);
    §5.1 and §5.2 (the new nd-fmm-plan interface replaces `FmmOperator`, with a link to
    fmm-plan-redesign.md; extensions 1 and 2 done); §5.3 (Morton-ordered columns,
    leaf-scaled data, §3.13, replacing interleaved absolute coordinates); §7 (Phase 3
    status per component, C3.0 as the rewrite, C4.0 as delivered in Phase 3 and
    removed from Phase 4, the C3.2 and C3.3 tables from T9 and T11, the T10 speed-up table, the calibration
    table and the per-pair timings); §9.1 (the adaptive-list and nd-fmm-plan-extension
    risks, with what Phase 3 found); §9.2 (Morton column order and variable-size leaf
    data answered); §6 or §8.3 (the "Threads and BLAS" rule of docs/phase3/README.md,
    as a constraint on the Phase 4 host batched path, the CubeCL CPU runtime and the
    SVD of C6.2, with the rlst `set_blas_threads` defect and whether it was reported
    upstream);
  - workspace-structure.md: §1.1 (what nd-fmm-plan now provides), §3 (the
    dependencies of nd-fmm-exec and nd-fmm-plan as built), §3.1 (the built surface of
    nd-fmm-exec, nd-fmm-plan and nd-fmm-validate's new examples and points), §4 (the
    seam is the batched interface; variable-size leaf data and batched hooks exist),
    §6 (Phase 3 done; the "Still open" items it answered);
  - docs/phase3/README.md: tick the exit checklist.
- A recommendation for Phase 4, in the PR and in design §7: the p values and
  distributions that the GPU tests should use, with their expected f32 and f64
  errors, and which stage dominates the per-pair run at those p.

Must pass: the root CI commands and the stricter workspace checks;
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`;
`cargo run --release -p nd-fmm-validate --example calibrate`. Paste its output into
the PR.

Do not:
- assert timings or calibration values in tests;
- commit CSV or other output;
- add criterion or any other dependency without asking;
- change nd-fmm-exec, nd-fmm-plan, nd-fmm-tables or nd-fmm-ref. If a measurement shows a defect,
  report it.

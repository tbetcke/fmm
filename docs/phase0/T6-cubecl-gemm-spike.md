# Phase 0 / T6 — CubeCL GEMM spike for dense M2L shapes

Can run any time after T1.

The development machine is an Apple Silicon Mac with no CUDA driver, so the GPU
backend for this spike is Metal. Metal has no f64, so f64 is measured and checked on
the CubeCL CPU runtime instead. The GPUs that matter for f64 are NVIDIA data-centre
cards (A100, H100 class). Their f64 behaviour is predicted from a roofline model here and
measured later, when a CUDA machine is available (see "Recommendation" below).

Do:
- spikes/cubecl-gemm/ (publish = false), in workspace members but NOT default-members.
- Pin CubeCL to the latest stable release (v0.10.0 when this brief was written); record
  the exact version and the matmul crate used (cubecl-matmul or its successor).
- Shapes: A is Nc x Nc with Nc = (p+1)^2 for p in {4, 8, 12, 16}; X is Nc x B for
  B in {1e3, 1e4, 1e5}; C = A X.
- Backends and precisions, as cargo features of the spike:
  - `metal` (required, default): CubeCL's `metal` feature (wgpu with the MSL
    compiler), f32 only. Confirm at start-up that the device reports no f64 support
    and record that in the report.
  - `cpu` (required): the CubeCL CPU runtime, f64 and f32, timed as well as checked.
  - `cuda` (optional, not built by default): the same f32/f64 benchmark, so the f64
    GPU measurement can be run later on a CUDA machine without new code.
- Compare the library matmul with a hand-written tiled kernel that takes p as comptime,
  on every backend and precision above.
- Check every result against a plain-Rust CPU reference computed in f64: relative
  1e-12 for f64 results, 1e-5 for f32 results.
- Write spikes/cubecl-gemm/SPIKE_REPORT.md: device (GPU and CPU model, core counts),
  macOS and Metal versions, CubeCL version, a table of GFLOP/s per
  (p, B, precision, backend, implementation), percentage of device peak (Apple GPU f32
  peak for Metal; CPU f64/f32 peak for the CPU runtime, stating how it was derived),
  and a recommendation (see below).

Roofline prediction for data-centre CUDA cards (in SPIKE_REPORT.md):
- For each p, give the arithmetic intensity of C = A X in f64 and f32 (flops per byte
  moved, counting A once and X and C once each; about Nc/8 in f64 for large B).
- Take peak f64 and f32 throughput and memory bandwidth for the A100 and H100 (SXM)
  from the vendor datasheets and cite them. Give f64 peak both with and without FP64
  tensor cores, and state whether the pinned CubeCL matmul can use them.
- Predict achieved GFLOP/s per (card, p, B, precision) as the roofline bound times an
  efficiency factor. Take the factor from the measured fraction of peak on Metal (f32)
  and the CPU runtime (f64), and state which one was used and why.
- Compare against rotation M2L using the operation counts in
  docs/design/laplace-fmm-plan.md §4 (dense 2(p+1)^4 flops per pair, rotation about
  (10/3)(p+1)^3) at a stated rotation efficiency. For each card and p, give the
  break-even fraction of f64 GEMM peak above which dense GEMM wins. This is the number
  a later CUDA run confirms or overturns.
- Label every predicted figure as a model, not a measurement, and list its assumptions.

Recommendation:
- f32: dense-GEMM or rotation M2L as the Phase 4 default, based on the Metal numbers.
- f64: a provisional recommendation for data-centre cards based on the roofline
  prediction, marked as such.
- Preferred but not required: run the `cuda` feature on an A100 or H100 before Phase 4
  fixes the f64 default, and add the measured numbers next to the predictions. If that
  run has not happened, Phase 4 proceeds on the provisional recommendation.

Do not: build this crate in CI; do not add CubeCL to any other crate.

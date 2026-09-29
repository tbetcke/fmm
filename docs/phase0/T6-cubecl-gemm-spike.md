# Phase 0 / T6 — CubeCL GEMM spike for dense M2L shapes

Can run any time after T1.

Do:
- spikes/cubecl-gemm/ (publish = false), in workspace members but NOT default-members.
- Pin CubeCL to the latest stable release (v0.10.0 when this brief was written); record
  the exact version and the matmul crate used (cubecl-matmul or its successor).
- Shapes: A is Nc x Nc with Nc = (p+1)^2 for p in {4, 8, 12, 16}; X is Nc x B for
  B in {1e3, 1e4, 1e5}; C = A X. Precisions f32 and f64. Backends: cuda (required),
  wgpu f32 if available, cpu runtime for correctness only.
- Compare the library matmul with a hand-written tiled kernel that takes p as comptime.
- Check results against a CPU reference (relative 1e-12 f64, 1e-5 f32).
- Write spikes/cubecl-gemm/SPIKE_REPORT.md: device, driver, CubeCL version, a table of
  GFLOP/s per (p, B, precision, implementation), percentage of device peak, and a
  recommendation: dense-GEMM or rotation M2L as the Phase 4 default in f64.

Do not: build this crate in CI; do not add CubeCL to any other crate.

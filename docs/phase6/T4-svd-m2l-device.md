# Phase 6 / T4 — SVD-compressed M2L on the device (C6.2, device)

T3 compresses the M2L tables and applies them on the host. On the device the dense M2L is
gathered GEMMs per level (device-path.md §6.4), with the tables uploaded once per `Fmm`;
at high p their size limits what fits beside large particle sets (laplace-fmm-plan §9.1;
the device refuses configurations that do not fit with `SettingsError::DeviceMemory`).
This task runs the compressed M2L on Metal, CUDA and the CubeCL CPU runtime, as
`docs/design/optimisation.md` §3 designs its device form.

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/design/optimisation.md §3; T3's PR and results section;
- docs/design/device-path.md §6.4–§6.6, §13, §18.2 (CUDA layouts and the chunk budget);
- the code: `nd_fmm_kernels::translate` (`grouped`, `grouped_stage`, `Gemm`,
  `GemmLayout`, `Accumulate`, `TileSchedule`, `Tables`), `movement::gather_columns`,
  `nd_fmm_exec::device` (the M2L level call, the tuner's M2L candidates in
  `nd_fmm_exec::tune`, `DeviceOperator`'s table upload and memory sum).

Do:
- **The compressed M2L level call** in `nd-fmm-kernels`: per chunk of offsets the gather
  as today, then two grouped products (the factors of T3) instead of one, and the
  reduction in row order, so each target meets its offsets in index order. Reuse T8's
  GEMM kernels and layouts (Phase 4) where the shapes allow; state the new shapes
  ((p + 1)² × k and k × (p + 1)²).
- **In `nd-fmm-exec`**: the strategy T3 introduced runs on the device; the factors are
  uploaded once per `Fmm` through `table_cache`; the memory sum before allocation counts
  them; the tuner may offer the compressed strategy as a candidate (state whether it
  does).
- **Measurements**: the M2L level calls and the evaluation against dense and rotation on
  Metal f32 (the M3 Max), CUDA f32 and f64 (locust), at p = 6, 8, 12, 18, N = 10⁶; the
  device memory; the error against the host compressed path and the direct sum. Report
  in the Phase 6 results file.

Tests that define done:
- The device compressed M2L against the host compressed M2L on the CPU runtime within
  Phase 4's device-against-host bounds (1e-12 f64, 1e-5 f32), and against the dense
  device path within T3's budget, on every `tests/mpi_exec.rs` scenario it runs.
- Two evaluations and two builds bit-identical; transfers per evaluation unchanged (the
  tables upload once).
- Metal and CUDA by hand (`#[ignore]`d tests as in Phase 4 and 4S), at 1 rank and at 2
  ranks sharing the GPU.

Must pass: the root checks, the stricter workspace checks, and the `nd-fmm-kernels`
checks of the root CLAUDE.md (`cpu,metal` clippy, `cuda` type-check, the CPU-runtime
tests); on locust the CUDA checks of the root CLAUDE.md; Metal outside the sandbox.

Do not: change the dense device path's defaults; assert a timing; add a dependency.

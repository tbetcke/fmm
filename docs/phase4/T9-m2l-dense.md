# Phase 4 / T9 — dense M2L on the device (C4.5)

The step that sets the budget at p ≥ 8: about 189 translations per box. For each level
and each of the 316 V-list offsets:
- gather the batch's source multipoles;
- multiply by the offset's table with one GEMM;
- scatter-add into the batch's target locals.

Each target appears at most once per offset, so a batch needs no atomics. Offsets are
added in index order, which is the order of every V row. The GEMM is the library matmul
(CMMA) for f32 at p ≥ 8, and the hand-written comptime-p kernel from T8 for small p and
for every f64 run (Phase 0 T6). The phase gate requires at least 80% of the spike's GEMM
throughput, and the efficiency profiled.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase4/README.md ("M2L strategies", "Strategy selection", "Exit gate" C4.5);
- docs/design/device-path.md, signed off: the dense M2L part of section 6 (the launch
  structure chosen there), sections 8, 9 and 13;
- docs/design/laplace-fmm-plan.md §3.6, §4 and §6.4–§6.6;
- docs/design/fmm-plan-redesign.md §4.3 (the V grouping, `batch_targets`,
  `batch_sources`, `offset_indices`) and §6.4;
- spikes/cubecl-gemm/SPIKE_REPORT.md (the throughput per (p, B) on Metal; CMMA needs
  (p + 1)² ≳ 64; launch- and occupancy-limited at B = 10³) with the 0.11 numbers from
  T2;
- `nd_fmm_tables::{M2lTables, M2lClasses, MatrixSet}`; `LaplaceOperator`'s M2L;
- the T8 GEMM, gather and scatter code.

Do:
- **Dense M2L level call** in `nd-fmm-kernels`, with the launch structure the design
  chose. By default, per offset in index order on one stream: gather, GEMM, scatter-add.
  A grouped or stacked variant that the design chose instead must keep each target's
  offsets in index order and stay conflict-free (requirements 4 and 5).
  - Batches with zero pairs are skipped.
  - Many launches are queued per sync.
- **GEMM choice per (precision, p)**, explicit and fixed at construction:
  - f32, p ≥ 8: the library matmul with the CMMA strategy that T2 found `Auto` to
    pick, named explicitly;
  - f32, p < 8, and f64: the hand-written kernel, with the per-backend layout.

  Where the library rejects a shape (for example a batch too narrow for CMMA), the
  hand-written kernel takes it, decided by the shape alone so that the choice is the
  same on every evaluation. Report how often that happens on the workloads.

  **Input precision guard** (README, "M2L strategies"): the library path is allowed only
  for a strategy whose inputs are multiplied in T, never one that rounds f32 to TF32,
  F16 or BF16 (CUDA's only MMA input types in CubeCL). Check this from the strategy's
  configuration (its input and compute types), not from the backend's name. Where it
  cannot be established, the hand-written kernel runs. Document the check, and add a
  test that a strategy with a lower input precision is rejected for f32.
- **Tables**: the 316 dense tables (`M2lTables`) uploaded once per `Fmm`, through
  `table_cache` when given; the memory per p and precision stated. `Classes` on the
  device as the design decided: run as `Dense` from `M2lClasses::expand`, or refused
  with `SettingsError`.
- The device operator runs M2L on the device by default for `Dense`; the host fallback
  stays selectable. For `Rotation`, M2L stays on the host fallback until T10.
- **Profiling and efficiency**, on Metal f32, reported only:
  - the GEMM alone: GFLOP/s and % of peak per level and offset-batch size (k) for the
    C3.2 cube and the Plummer sphere at p = 3 and 8, and at p = 12 and 16 in f32 for
    the kernel only (as model input for f64; f32 FMMs stop at p = 8);
  - against the spike at the same (p, k): the gate is at least 80% of the spike's
    throughput where the batch sizes match the spike's B (10³, 10⁴, 10⁵), GEMM only.
    Below that: analyse it (occupancy, layout, launch overhead) and report;
  - gather and scatter time, and launches per level;
  - the whole M2L stage time on the device against the host M2L at 1 and 12 threads.

Tests that define done (CPU runtime f32 and f64, hand-written kernel only, small shapes;
Metal f32 with both GEMMs by hand, `#[ignore]`; each test prints the backends and GEMMs
it ran):
- Operator check, as C2.2 and T8 of Phase 3. For all 316 offsets on levels 2, 9 and 16
  of a dyadic domain, the device M2L against `nd_fmm_ref::direct` at the canonical
  frames:
  - f64, CPU runtime, p ≤ 20: within 1e-14 relative to the term magnitudes, per degree
    in the §3.8 weighting for locals (Nₘ/Sₘ);
  - f32, p ≤ 8: within 1e-5 against f64, with both GEMMs on Metal.
- Batches: a level call equals applying each target's V row in offset order on the
  host, within the GEMM tolerance of T8. Each target is written once per offset, and
  ghost rows stay zero.
- The library and the hand-written GEMM agree within the GEMM tolerance on the same
  batches (Metal, f32, p = 8).
- Determinism: repeated level calls bit-identical, with both GEMMs. This confirms
  that the library strategy is fixed.
- FMM: with M2L dense on the device (the other kinds as merged so far, or on the host),
  every `tests/mpi_exec.rs` scenario within the README's FMM bounds of the host output,
  bit-identical across two evaluations. The ignored C3.2 and C3.3 gates pass on the
  device path at p = 3 and 8 (f32 on Metal by hand; f64 on the CPU runtime), with
  errors within 0.1% (f64) and 5% (f32) of the host run's.

Must pass:
- `cargo test -p nd-fmm-kernels --features cpu --release` (within the T4 budget) and
  `--features metal --release -- --ignored` by hand;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` without features (unchanged), and
  `--features cpu --release` (with `-- --ignored`), and on Metal by hand;
- clippy on both crates, without features and with `--features cpu,metal`;
  `cargo check -p nd-fmm-exec --features cuda`; `cargo doc` without warnings;
- the CPU-runtime CI job, if kept; the root checks and the stricter workspace checks.

Report: the backends and GEMMs run; the maximum errors per backend, precision and level;
the efficiency tables against the spike, with the gate stated as met or not per cell;
the launch counts.

Do not:
- fuse the gather into the GEMM loads (C6.1) or compress the tables (C6.2);
- use atomics, or a library strategy chosen per call;
- change the offset order, `nd-fmm-tables` or the host path;
- tune tolerances. If a check fails, isolate the offset and level, report and stop.

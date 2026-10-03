# Phase 4 / T10 — rotation M2L on the device (C4.6)

M2L by point and shoot on the device. Each translation:
- rotates the multipole so that the shift lies along +z;
- translates it coaxially;
- rotates the result back.

All three steps come from the precomputed `RotationTables` of Phase 2 (C2.3). It costs
about (10/3)(p + 1)³ multiply-adds per pair against dense GEMM's (p + 1)⁴. In f32 on
Metal the spike expects dense to win at every p (design §4). For f64 on data-centre
cards the model expects rotation to win above p ≈ 10, but nothing in this phase can run
f64 on a GPU. The task therefore delivers a correct, reasonably fast kernel, timed
against dense in f32 on Metal. It records the rotation efficiency that the f64 model of
the spike report needs, without claiming the f64 crossover.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase4/README.md ("M2L strategies", "Strategy selection", "Exit gate" C4.6);
- docs/design/device-path.md, signed off: the rotation part of section 6, sections 9 and
  13;
- docs/design/laplace-fmm-plan.md §3.2, §4, §6.4 and §6.7;
- CONVENTIONS §3.8 and §3.11 ("Coaxial translations", "Rotation of coefficients");
- `nd_fmm_tables::rotation` in full: the storage of `ShiftTables` (y-rotation blocks per
  polar angle, azimuth factors, coaxial factors, `Alignment`), `RotationTables::m2l`,
  `RotationScratch`;
- spikes/cubecl-gemm/SPIKE_REPORT.md, "Dense against rotation M2L in f64" (the
  break-even rotation efficiencies);
- `LaplaceOperator`'s M2L under `Rotation`; the T8 and T9 code.

Do:
- **Rotation tables on the device**: the M2L family of `RotationTables` (blocks,
  factors and the per-offset indices of polar angle, azimuth, distance and alignment)
  in a device layout the design fixes, uploaded once per `Fmm`, and its memory stated
  per p. M2M and L2L stay the dense octant GEMM of T8 under `Rotation` too (README,
  "Design decisions").
- **The kernel** in `nd-fmm-kernels`, with the structure the design fixes (per pair or
  per target; per-degree products in shared memory or registers; p comptime):
  - the azimuth rotation as plane rotations of the (+m, −m) pairs, the y-rotation by
    the per-degree blocks, the coaxial translation per order m, and the way back, as
    `ShiftTables::apply` does;
  - axis-aligned offsets (`Alignment::Up`, `Down`) by the coaxial step alone;
  - each target's V row in offset-index order (requirement 4). With one cube per
    target walking its row, the order is the row's. With per-offset batches, as in
    T9, it is the batches' index order. No atomics.
- The device operator runs M2L on the device for `Rotation` by default; the host
  fallback stays selectable.
- **Timing**, on Metal f32, reported only:
  - per pair, rotation against the T9 dense M2L (with the GEMM it selects) for
    p = 2, 4, 6, 8, 12 and 16, on the V batches of the C3.2 cube;
  - the rotation kernel's achieved flop rate as a fraction of f32 peak, counting
    (20/3)(p + 1)³ flops per pair as the spike does, against the break-even
    efficiencies of the spike report;
  - the whole M2L stage, device rotation against host rotation at 1 and 12 threads, at
    p = 8 (f32).

  What this implies for the f64 crossover is a model statement for T12 and T13, marked
  as such.

Tests that define done (CPU runtime f32 and f64; Metal f32 by hand, `#[ignore]`; each
test prints the backends it ran):
- Operator check, as C2.3 and T8 of Phase 3. For all 316 offsets on levels 2, 9 and 16
  of a dyadic domain, the device rotation M2L against `nd_fmm_ref::direct` and against
  `RotationTables::m2l` on the host, relative to the term magnitudes, per degree in the
  §3.8 weighting for locals:
  - f64, CPU runtime, p ≤ 20: within 1e-13;
  - f32, p ≤ 8: within 1e-5 against f64.
- Axis-aligned offsets, and the offsets with the largest and smallest polar angles,
  explicitly.
- Rotation against dense on the device (T9) on the same batches: within 1e-13 (f64,
  CPU runtime) and the f32 bound.
- Determinism: repeated level calls bit-identical.
- FMM: with `Rotation` and every kind merged so far on the device, every
  `tests/mpi_exec.rs` scenario within the README's FMM bounds of the host `Rotation`
  output, bit-identical across two evaluations. The ignored C3.2 gate passes on the
  device path at p = 18 in f64 on the CPU runtime (at N = 10⁴ if N = 10⁵ exceeds the
  README's time limit), with errors within 0.1% of the host run's.

Must pass:
- `cargo test -p nd-fmm-kernels --features cpu --release` (within the T4 budget; the p
  sweep to 20 `#[ignore]`d if needed) and `--features metal --release -- --ignored` by
  hand;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` without features (unchanged), and
  `--features cpu --release` (with `-- --ignored`), and on Metal by hand;
- clippy on both crates, without features and with `--features cpu,metal`;
  `cargo check -p nd-fmm-exec --features cuda`; `cargo doc` without warnings;
- the CPU-runtime CI job, if kept; the root checks and the stricter workspace checks.

Report: the backends run; the maximum errors per backend, precision and level; the
timing table against dense; the efficiency against the spike's break-even values. State
that the f64 comparison is a model and that no f64 GPU run was possible.

Do not:
- change `nd-fmm-tables` or its storage. If the device needs another layout, convert at
  upload;
- use atomics or change the offset order;
- change the M2L default rule (T12 does, through autotune).

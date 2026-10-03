# Phase 4 / T7 — device leaf expansion operators: P2M, L2P, P2L, M2P (C4.3)

The four operators that evaluate solid harmonics at points, as `#[cube]` kernels in
`nd-fmm-kernels`:
- P2M at every local leaf;
- L2P at every local leaf, with gradients optionally;
- P2L along the X list;
- M2P along the W list, with gradients optionally.

The design's C4.3 names only P2M and L2P. This task covers all four, because on
adaptive trees L2P and M2P are a growing part of the leaf stage once P2P is fast
(design §7, "After Phase 3S"). It also means no operator kind of an adaptive tree stays
on the host. Each kernel evaluates the harmonics by the Cartesian recursion of
CONVENTIONS §3.5, with p as a comptime parameter, and is checked against
`nd_fmm_ref::leaf`.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase4/README.md ("Accuracy measures", "Exit gate" C4.3);
- docs/design/device-path.md, signed off: the leaf-operator part of section 6, sections
  9 and 13;
- CONVENTIONS §3.3–§3.7 (with §3.5, the recursion), §3.11 ("L2P and M2P") and §3.13
  ("Operators in scaled coordinates"), with the T3 addition;
- spikes/device-arith/REPORT.md (the harmonics check and the arithmetic rules);
- `nd_fmm_math::harmonics` (`regular`, `irregular`, `regular_grad`, `irregular_grad`)
  and `Layout`; `nd_fmm_ref::leaf` (`p2m`, `l2p`, `p2l`, `m2p`) and `Frame`;
- `LaplaceOperator`'s leaf calls (`p2m_target`, `l2p_target`, `p2l_target`,
  `m2p_target`, the frames from `geometry::relative_frame`);
- the T4 and T5 code, and T6's if merged (the shared launch and frame conventions).

Do:
- **Harmonics on the device**: regular and irregular solid harmonics of all degrees up
  to p (M2P up to p + 1 for its gradient), with their gradients, at one point. They
  follow the recursion of `nd_fmm_math::harmonics` operation for operation where that
  is practical, so that host and device agree to rounding. Where the design chose
  precomputed recursion coefficients, build them on the host from `nd-fmm-math` and
  upload them once.
  - Registers or shared memory, as the design fixes per backend.
  - p comptime, up to 20.
  - Unit tests against `nd-fmm-math` at seeded points inside and outside the unit ball,
    per degree, in the orthonormal weighting of CONVENTIONS §3.8.
- **Kernels**, each generic over the runtime and f32/f64, with the structure the design
  fixes. By default:
  - P2M: one cube per leaf, one unit per point (or per block of points), the
    per-coefficient sum over the leaf's points reduced in shared memory or by plane
    operations in a fixed order;
  - L2P and M2P: one unit per target point. L2P adds its leaf's local, and M2P then
    adds the boxes of its W row in box-index order. The evaluator calls L2P before M2P
    (fmm-plan-redesign §7.5);
  - P2L: one cube per target box, walking its X row in leaf-index order;
  - frames (ĉ, r̂) exact from integer keys (§3.13), as T6 forms them; never a
    floating-point shift.
- Safe launch wrappers per level call, and the device operator running each of the four
  kinds on the device by default, the host fallback selectable.
- Timing on Metal f32, reported only: each kernel per level of the C3.2 cube and the
  Plummer sphere at p = 3 and 8, against the host stage time at 1 and 12 threads.

Tests that define done (CPU runtime f32 and f64; Metal f32 by hand, `#[ignore]`; each
test prints the backends it ran):
- Operator check, as T8 of Phase 3. On levels 2, 9 and 16 of a dyadic domain, each of
  P2M, L2P, P2L and M2P on the device against `nd_fmm_ref::leaf` at the absolute
  frames, with and without gradients:
  - f64, CPU runtime, p ≤ 20: within 1e-13 relative to the term magnitudes, per degree
    in the §3.8 weighting for coefficients;
  - f32, p ≤ 8: within 1e-5 against the f64 reference.
- Against the host operator: the device result within twice the measured host-vs-ref
  error of the same check, so that a device-only drift shows.
- Leaves with 0, 1 and many points; points on leaf faces and at leaf centres; M2P and
  P2L with source and target boxes on every level difference the W and X lists allow.
- Determinism: repeated launches bit-identical, including the reduction of P2M.
- FMM: with these four kinds on the device (and P2P on the device if T6 has merged, else
  on the host), every `tests/mpi_exec.rs` scenario within the README's FMM bounds of
  the host output, and bit-identical across two evaluations. The ignored C3.3 gate
  (`tests/adaptive.rs`) passes on the device path, with errors within 0.1% (f64) and 5%
  (f32) of the host run's.

Must pass:
- `cargo test -p nd-fmm-kernels --features cpu --release` (within the T4 budget; long
  sweeps `#[ignore]`) and `--features metal --release -- --ignored` by hand;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` without features (unchanged);
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release`, also with
  `-- --ignored`, and on Metal by hand;
- clippy on both crates, without features and with `--features cpu,metal`;
  `cargo check -p nd-fmm-exec --features cuda`; `cargo doc` without warnings;
- the CPU-runtime CI job, if kept; the root checks and the stricter workspace checks.

Report: the backends run; the maximum errors per operator, backend, precision and level
(with the worst degree); the timing table.

Do not:
- re-derive a formula: the recursion is that of `nd-fmm-math` and the operators those of
  `nd_fmm_ref::leaf`, in their conventions. If a discrepancy points at either crate,
  report it and stop;
- change the accumulation order of any target, or use atomics;
- touch the translation kernels (T8–T10) or the host path.

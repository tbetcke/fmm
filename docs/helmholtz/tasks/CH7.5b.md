# CH7.5b The device's `PlaneWave`

Started 2026-11-05 on `tbetcke/h-ch7-5b`. Roadmap: "CH7.5b The device's `PlaneWave`"
(priority list item 1, CH7.5a's split: "a/b/c, Helmholtz → d", the user, 2026-11-05).
Proposal: §10 CH7.5a–c; §2 constraints 1 and 2; §8 (the device path). Continues
docs/helmholtz/tasks/CH7.5a.md and docs/phase6/T13-plane-wave-m2l.md, "Afterwards".

## Goal
`nd-fmm-exec`'s device path runs M2L under `PlaneWave` on CH7.5a's merge and shift kernels
when a builder setting names it, and keeps running it as `Dense` otherwise.

## Acceptance (proposal §10, CH7.5a–c, verbatim)
T13's requirement 1 on the device (within 1.1× Dense's errors against the direct sum), bit
for bit run to run and on P ranks against one rank; adopted where it wins at p ≥ 12 (T13's
gate), else dropped.

The roadmap's line for CH7.5b, which this task meets (the adoption is CH7.5c's):
`PlaneWave` on the device (CUDA f32 and f64, Metal f32, the CubeCL CPU runtime in CI)
within 1.1× `Dense`'s errors against the direct sum at the C3.2 cube and the C3.3 Plummer
sphere (T13's requirement 1, f64 p = 8, 12, 18 and f32 p = 8), device locals against the
host `PlaneWave`'s within the stated bound, bit for bit from run to run and on P ranks
against the one-rank device run; constraint 1: without the setting every strategy, the
device's `PlaneWave` and Laplace's defaults bit for bit, the device `Auto` unchanged.

## Read first
- fmm-exec/CLAUDE.md (the device path's rules; "The plane-wave M2L"), fmm-kernels/CLAUDE.md
  ("Plane-wave M2L").
- `fmm-exec/src/device.rs`: `DeviceOperator::new` (the M2L tables by strategy, the memory
  check, the report), `m2l_on_device`, `DeviceReport::strategy_name`; the module docs
  "Dense M2L", "Rotation M2L", "Determinism".
- `fmm-exec/src/operator_plane_wave.rs` (the host order) and `operator.rs`'s `m2l` under
  `PlaneWave` (per vector and frame, parents in index order).
- `fmm-kernels/src/plane_wave.rs` (the kernels' API) and the lists of CH7.5a's test
  (`Lists::new` in `fmm-kernels/tests/kernels/translate/plane_wave.rs`).
- `fmm-exec/tests/device_common/mod.rs` (`check_backend`, `expected_evaluation`),
  `tests/mpi_exec.rs` (`check_plane_wave`, `check_one_rank`), `tests/device_fmm.rs` (the
  C4.8 gate's problems), `tests/device_metal.rs`, `tests/device_cuda.rs`.

## Changes
`nd-fmm-exec` (feature `gpu`):
- `fmm::DevicePlaneWave` (`Dense`, the default: "PlaneWave, run as dense on the device" as
  before; `Kernels`: merge and shift on the device), `FmmBuilder::device_plane_wave` and
  `FmmBuilder::device_plane_wave_budget` (bytes of one chunk's waves, accumulators and
  incoming blocks; default the device's translation scratch budget,
  `nd_fmm_kernels::translate::default_scratch_bytes`); both exist without `gpu`, as
  `DeviceGather` does, and are ignored by the host and by every other strategy.
- `device::DeviceOptions::{plane_wave, plane_wave_budget}`; in `DeviceOperator::new`, under
  `PlaneWave` with M2L on the device and `Kernels`: no dense M2L tables; the host's
  `PlaneWaveTables` uploaded once (`nd_fmm_kernels::plane_wave::PlaneWaveTables`); per
  level with a V pair the lists from the V view in the order of CH7.5a's test and of
  `src/operator_plane_wave.rs` (parents in index order; per parent the slots from the
  computed children's rows; per direction the accumulators whose mask meets the computed
  children, each over its slots ascending; per computed child its accumulators in order),
  cut into chunks of whole parents whose scratch fits the budget (at least one parent), the
  m vectors as columns (each chunk's lists repeated per vector on columns b m + v); one
  scratch for the widest chunk; everything in the memory check before allocating. The
  level call runs `plane_wave::m2l` per chunk (four launches each). `DeviceReport` gains
  `plane_wave_layout` and `plane_waves` (per level: pairs, chunks, sources, accumulators,
  incoming blocks, locals, launches) and the name "PlaneWave, merge and shift on the
  device"; module docs ("Plane-wave M2L on the device").
- `nd-fmm-kernels` (CH7.5a's module only, which no path calls before this task):
  `PlaneWaveScratch` sized for the widest of several plans (`covering`; a launch checks
  the scratch holds at least its plan's buffers), and the bytes of tables and lists before
  upload (`PlaneWaveTables::bytes_of`, `PlaneWaveLists::bytes`) for the memory check.
- No change to the host path, `nd-fmm-tables`, `nd-fmm-plan`, a convention or a
  dependency. The device's `Auto` and its tuner are untouched (`PlaneWave` is only named).

## Tests that prove it
- `tests/device_common::check_plane_waves` (new), in every device scenario that
  `check_plane_wave` of `tests/mpi_exec.rs` reaches (the CPU runtime, f32 and f64, every
  rank count in CI's MPI list) and in `tests/device_metal.rs` (f32) and
  `tests/device_cuda.rs` (f32, f64): the setting reported; every kind on the device; the
  output within the FMM bounds of the host `PlaneWave`'s (relative L2 1e-12 f64, 1e-5 f32,
  φ and ∇φ) and the multipoles and locals of every level within them
  (`check_expansions`); two evaluations and a second build bit for bit; a budget of one
  parent per chunk bit for bit the default with more chunks; the transfers, launches and
  syncs of `expected_evaluation` (four launches per chunk, from the report); three vectors
  each the one-vector device run bit for bit.
- Constraint 1: `check_plane_wave` keeps asserting that without the setting the device's
  `PlaneWave` is "run as dense" and its output `Dense`'s bit for bit; every existing device
  check unchanged.
- Rank invariance: in `check_plane_wave` on several ranks, the device `Fmm` with the
  setting and every kind but M2L on the host fallback against the one-rank `Fmm` of the
  same settings over the union, bit for bit (`check_one_rank`; the other kinds' device
  kernels are not rank-invariant bit for bit, the host's are).
- The accuracy gate, ignored `tests/device_fmm.rs` (new section): the cube and the Plummer
  sphere at N = 10⁵, f64 p = 8, 12, 18 on the CPU runtime and CUDA, f32 p = 8 on Metal and
  CUDA: the φ and ∇φ L2 errors against the direct sum (eight vectors, rms) under
  `PlaneWave` with the setting within 1.1× those of the device's `Dense`.

## Checks
`/verify` (format, Clippy default, examples and workspace, `cargo test`, the workspace
tests, docs, the MPI list at 2 and 4 ranks), and the device path's own (fmm-exec/CLAUDE.md,
"Before finishing"): Clippy with `cpu,metal`, `cargo check --features cuda`,
`cargo test -p nd-fmm-exec --features cpu --release` (and by hand at 2 and 4 ranks),
`cargo doc --features cpu`, the ignored `device_fmm` gate on the CPU runtime; the kernels'
Clippy and tests after the scratch change.

## Machine runs
- The M6, Metal (outside the sandbox; correctness, no timing): `cargo test -p nd-fmm-exec
  --features metal --release -- --ignored` (`device_metal`, `device_fmm`).
- locust, CUDA (correctness, no timing): `tests/device_cuda.rs` and `tests/device_fmm.rs`
  with `--features cuda --release -- --ignored`, the crate's Clippy with `cpu,cuda`.
Timings are CH7.5c's.

## Constraints and risks
- Constraint 1 (agreed 2026-11-05 at the split, "Go ahead": the device's `PlaneWave`
  opt-in until CH7.5c's sign-off): the new code runs only under `PlaneWave` with
  `DevicePlaneWave::Kernels`; every other build keeps its tables, launches and bits. The
  branch is at the level call (no dispatch below it); no Laplace layout or table changes.
  D9 timing runs belong to CH7.5c.
- Constraint 2: each accumulator's slots and order come from the geometry and the V rows
  of its computed children, which are the same on every rank (the host's argument), so
  the device M2L is rank-invariant; chunks do not change bits (each item writes its own
  output from zero, the local adds its three frames in order).
- Risk: memory. The waves of a chunk (6 × 2H values per source and vector) are bounded by
  the budget; one parent always fits a chunk (at most 216 sources).
- Risk: the f32 Metal error ratio (the quadrature's error at f32 p = 8 is near the f32
  floor; T13 met 1.1× on the host in f32).

## As built
As planned. The review added a cap on each chunk's buffers at the kernels' 32-bit index
range besides the budget, and a check that the setting under `Dense` changes nothing.
The rank check runs on several ranks only.

## Runs (2026-11-05)
Correctness only; the evaluation times the gate prints are reported, never asserted, and
are CH7.5c's to measure properly.
- The M6, CPU runtime (release): `tests/mpi_exec.rs` at 1, 2 and 4 ranks (21 and 24
  device plane-wave checks, the output within 2.3e-7 of the host's `PlaneWave` in f32,
  M2L alone on the device bit for bit the one-rank run at 2 and 4 ranks; 163 s, 122 s,
  322 s); `tests/device_ranks.rs` at 2 and 4 ranks (f64 within 5.9e-16, f32 within
  3.2e-7); the `tests/device_fmm.rs` gate (820 s): PlaneWave over Dense, L2 errors φ
  0.941–1.010 and ∇φ 0.949–1.011 on the cube and the Plummer sphere, f64 p = 8, 12, 18.
- The M6, Metal (outside the sandbox): `tests/device_metal.rs` (every scenario, within
  3.6e-7 of the host's `PlaneWave`, 1–5 chunks; 150 s); the gate (62 s): f32 p = 8, φ
  1.011 (cube) and 0.999 (Plummer), ∇φ 1.004 and 1.005.
- locust, CUDA (H100, idle): Clippy of nd-fmm-exec with `cpu,cuda`; `tests/device_cuda.rs`
  (248 s); the gate (1274 s): f32 p = 8 and f64 p = 8, 12, 18, φ 0.941–1.010, ∇φ
  0.949–1.011. After the review's fixes (9bf419b): Clippy of nd-fmm-exec and nd-fmm-kernels
  with `cpu,cuda`, `tests/device_cuda.rs` again (245 s; f64 within 4.6e-16, f32 within
  2.6e-7 of the host's `PlaneWave`, 1–5 chunks) and the kernels' seven plane-wave tests
  with `--include-ignored` (CPU runtime and CUDA), all passed.

## Pending runs
None.

## Open questions
1. The roadmap's "the CubeCL CPU runtime in CI": CI runs no `nd-fmm-exec` device test (no
   job builds it with `cpu`); the CPU runtime ran by hand, as every device check of
   `nd-fmm-exec` does. The kernels' plane-wave tests, the covering scratch included, do
   run in CI (`run-tests-kernels`). Accept, or add an exec CPU-runtime job (a separate
   task)?

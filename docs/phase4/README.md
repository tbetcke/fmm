# Phase 4: CubeCL kernels on the device

Phase 4 runs the Laplace FMM on a GPU. It adds the crate `nd-fmm-kernels`, which holds
every `#[cube]` kernel, and a device path in `nd-fmm-exec`. The device path plugs into
the level-batched `FmmOperator` interface that Phase 3 built, as the host path does
(design §5.2; C4.0 was delivered in Phase 3). It reads the same Phase 2 tables and
leaf-scaled data, and `nd-fmm-plan`'s `Evaluator` keeps running the pass order.

The Phase 0 spike (`spikes/cubecl-gemm/SPIKE_REPORT.md`) and the Phase 3 and 3S
measurements set the targets:
- In f32, dense GEMM M2L is the default. The library matmul (CMMA) is used for
  p ≥ 8, and a hand-written kernel with p as a comptime parameter for small p, where
  the library falls back to a scalar path that is 10× slower.
- In f64, the dense path is a hand-written kernel (CubeCL has no FP64 tensor cores).
  Rotation M2L is expected to win above p ≈ 10. Autotune sets the crossover.
- The leaf stage (P2P with L2P and M2P) is still 49–69% of a host evaluation at
  p = 3 after Phase 3S, so the GPU P2P kernel is measured alongside M2L from the start.
  On adaptive trees L2P and M2P are a growing part of it (design §7, "Recommendation
  for Phase 4").

Hardware (decided on 2026-10-03): **no GPU besides the development machine's** is
available. Every timing in this phase is **Metal, f32, on the Apple M3 Max**. f64 runs
on the **CubeCL CPU runtime**, which is a correctness backend, never a throughput one
(design §6.1). The CUDA backend is built and type-checked but never run. A CUDA
measurement stays a documented, single command for whoever later has an A100- or
H100-class card, as x86_64 timings were in Phase 3S.

CubeCL version (decided on 2026-10-03): the workspace moves from 0.10.0 to
**0.11.0-pre.4** at the start of the phase (T2). 0.11 re-enables f64 on CUDA, which
0.10.0 removed, and changes the frontend. Phase 4 is written against 0.11 only.
Moving to the final 0.11.0 release, once it appears, is a separate decision.

The phase has four parts:
1. **Decisions.** A design document for the device path (signed off by hand), the
   move to CubeCL 0.11.0-pre.4, and a study of device arithmetic with an addition to
   CONVENTIONS §3.13 on what device kernels may assume.
2. **Infrastructure.** The crate `nd-fmm-kernels` (backends, capability check, device
   buffers, the gather and scatter primitives). A device path in `nd-fmm-exec` in which
   every operator kind first runs on the host, so the FMM runs end to end from the
   start.
3. **Kernels.** P2P; the leaf expansion operators (P2M, L2P, P2L, M2P); M2M and L2L as
   batched GEMM; M2L as dense GEMM; M2L by rotation. Each moves one operator kind onto
   the device and is checked against the host operator and `nd-fmm-ref`.
4. **The device FMM, autotune and measurement.** Every operator on the device, with the
   gate "GPU result equals CPU FMM"; strategy autotune with a persistent cache; the
   benchmark report and the design-document update.

Companion documents:
- [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md): Sections 4, 5.2,
  5.3, all of 6, 7 (Phase 3 "Per-pair cost" and "Recommendation for Phase 4", Phase 3S,
  Phase 4), 8 and 9;
- [docs/design/fmm-plan-redesign.md](../design/fmm-plan-redesign.md): §5, §6.4 (the GEMM
  check), §7.5 and §10;
- [docs/design/workspace-structure.md](../design/workspace-structure.md): Sections 2, 3,
  3.1 and 5;
- [docs/design/simd-p2p.md](../design/simd-p2p.md): §3, §4.4 and §5.5, as the model for
  the device P2P;
- `spikes/cubecl-gemm/SPIKE_REPORT.md`.

T1 adds a further companion, `docs/design/device-path.md`, signed off on 2026-10-03.

Prerequisite: Phase 3S is complete. T7 is merged (PR #47), and the design documents
carry the Phase 3S outcome. The briefs assume the workspace as it is after that merge.

## Scope

In scope:
- `docs/design/device-path.md`, signed off. The design-document updates at the end of
  the phase.
- The CubeCL pin: `cubecl =0.11.0-pre.4` with the matching `cubek-matmul` and
  `cubek-std` (0.3.0-pre.4), and `spikes/cubecl-gemm` ported to it (T2).
- `docs/CONVENTIONS.md` §3.13: an addition on device kernels (the coincident-pair rule,
  the domain and the arithmetic they may assume), with the spike `spikes/device-arith/`
  that measures it (T3). The same spike times a CPU-shaped P2P on the CPU runtime
  against `nd-fmm-simd`, for decision 10.
- `nd-fmm-kernels` (new crate, `fmm-kernels/`):
  - backend selection and the f64 capability check;
  - device buffers, and the gather, scatter-add and zeroing kernels;
  - the P2P, P2M, L2P, P2L, M2P, M2M, L2L and M2L kernels: the hand-written GEMM, the
    library matmul, and rotation.
- `nd-fmm-exec`:
  - a device path behind the cargo feature `gpu`, with a backend setting on
    `FmmBuilder`;
  - the device operator, with a host fallback per operator kind;
  - autotune of the M2L strategy with a persistent cache (C4.7).
- `nd-fmm-validate`: the device rows of the kernel and FMM benchmarks, the device
  accuracy gates, and a device leaf-size study.
- `.github/workflows/`: a CPU-runtime job for `nd-fmm-kernels`, if T4's measurement and
  the sign-off keep it.

Out of scope:
- Multi-rank device runs (C5.1, device part, after C4.7). Also overlap of exchange and
  computation, and device-resident ghost buffers (C5.2). The design must not preclude
  them.
- A host batched-GEMM path through BLAS (decided on 2026-10-03: deferred; a candidate
  for Phase 6). Phase 4 makes no BLAS call, so the host path and its threading rules do
  not change.
- Fusing the gather into the GEMM loads (C6.1), SVD-compressed M2L (C6.2), several
  charge vectors (C6.3), dipoles (C6.4), and plane-wave M2L (C6.5).
- FP64 tensor cores, and f16, bf16 or tf32 arithmetic.
- Backends other than Metal (wgpu with the MSL compiler) and the CubeCL CPU runtime.
  CUDA is type-checked only (`cargo check … --features cuda`). HIP, Vulkan and WebGPU
  are not built, although the kernels stay runtime-generic.
- Overlapping P2P with the far field on separate streams. It changes the accumulation
  order of fmm-plan-redesign §7.5. T1 may propose it for later.
- Comparisons with FMM3D, ExaFMM-t or kifmm-rs (design §8.3), unless the sign-off
  question below moves them into T13.
- Changes to the host path's defaults (strategy `Auto`, `P2pChoice::Auto`, the leaf
  size of 64) or to its results.
- Changes to `nd-fmm-math`, `nd-fmm-ref`, `nd-fmm-tables`, `nd-fmm-simd` and `nd-octree`
  beyond fixing a defect that a test exposes (stop and report first). Changes to
  `nd-fmm-plan`: the signed-off design needs none (no T4b, device-path.md §4.3; the
  C5.1 hook of device-path.md §4.5 is for later).
- p > 20 (`MAX_DEGREE`).

## Requirements on the device path

T1 designs against these, and T4–T13 are accepted against them. T1 may propose changing
one, with reasons, for sign-off.

1. **Same interface.** The device path is an `FmmOperator` driven by `nd-fmm-plan`'s
   `Evaluator`. It does not re-implement the pass order, the lists or the exchanges.
   `Fmm` keeps its API; the backend is one more builder setting, and the host path
   stays the default. Until C5.1 the device backend runs on one rank only: on more
   than one rank `build` returns `SettingsError::DeviceNeedsOneRank`, after the
   `PointsNotOwned` check (clarified at sign-off, device-path.md §2 and §4.4).
2. **Same data and conventions.** The leaf-scaled chunks of §3.13 and the scaled
   coefficients of §3.7 in the real storage of §3.6. The tables are those of
   `nd-fmm-tables`, built in f64 and rounded to T. Frames come from integer keys, and
   tables are looked up by octant and offset index, never by a floating-point shift.
   1/(4π) and r_t are applied once, by `Fmm`, on the way out.
3. **Device-resident data.** Within one evaluation on one rank, coefficients and leaf
   data stay on the device:
   - plan views and tables are uploaded once per `Fmm`;
   - points are uploaded once per build, and charges once per evaluation;
   - the output is downloaded once per evaluation.

   The design states the transfers per evaluation, and a test counts them.
4. **Accumulation rule kept.** Every device kernel adds each target's contributions in
   the order of that target's row in the target-centric view (`nd_fmm_plan::operator`,
   "Accumulation rule"; fmm-plan-redesign §7.5): offsets in index order, octants in
   order, leaves and boxes in index order. Inside one contribution (one matrix–vector
   product, one leaf's P2M), the summation order is the kernel's, fixed and documented.
5. **No atomics.** M2M and L2L gather, M2L batches are conflict-free per offset, and
   P2P and the leaf operators are target-centric (design §6.6). Where a reduction is
   needed, it uses shared memory or plane operations, in a fixed order.
6. **Deterministic.** For a fixed tree, backend, device, build and resolved strategy,
   the output is bit-identical from evaluation to evaluation and from run to run. No
   kernel or matmul strategy is chosen per call. Every autotuned choice is resolved
   when the `Fmm` is built, reported, and then fixed.
7. **Precision is a capability.** f32 runs on every backend. f64 runs where the device
   reports it (the CPU runtime here; CUDA on 0.11 in principle). Where it does not, as
   on Metal, f64 is refused at build with a `SettingsError` agreed on every rank, never
   by a panic or a silent cast.
8. **Host fallback.** Every operator kind can run on the host (the `LaplaceOperator`
   kernels, with explicit transfers), selectable per kind. This is how the device path
   grows task by task, and it stays as a test aid. With every kind on the host, the
   device path's output equals the host path's bit for bit.
9. **Safe at the boundary.** `unsafe` only inside `nd-fmm-kernels` (CubeCL launches,
   buffer views), each block with `// SAFETY:`. Every public function is safe, and
   `nd-fmm-exec` stays free of `unsafe` (root `CLAUDE.md`). `nd-fmm-exec` reaches CubeCL
   only through `nd-fmm-kernels`, and has no direct `cubecl` dependency.
10. **Tested on what can run.** Every kernel is checked on the CPU runtime in f32 and
    f64 and on Metal in f32. Every test run prints the backends it exercised. A report
    never presents a backend that did not run as passing, and CUDA is "type-checked,
    not run".
11. **Measured, never asserted.** Timings are reported on the M3 Max (Metal), with
    compilation excluded and many launches queued between syncs. No timing is taken in
    CI or asserted in a test.

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **A new crate.** `fmm-kernels/`, package `nd-fmm-kernels`, library `nd_fmm_kernels`,
  version `0.1.0-dev`, inheriting `[workspace.package]`. It is a workspace member.
  Whether it is a default member follows from T4's CI decision.
  - Dependencies: `cubecl`, `cubek-matmul`, `cubek-std` (the pinned workspace entries),
    `nd-fmm-math` and `thiserror`.
  - Dev-dependencies: `nd-fmm-ref`, `nd-fmm-tables` and `proptest`.
  - No MPI, no `nd-fmm-plan`, no `rayon`: it builds and tests without MPI.
  - Backends are cargo features: `cpu` (the CubeCL CPU runtime), `metal` (wgpu with the
    MSL compiler) and `cuda` (type-checked only). No feature is on by default.
- **The device path in `nd-fmm-exec`.** It sits behind the cargo feature `gpu`, which
  enables `nd-fmm-kernels`. The feature is off by default, so the default members and
  the existing CI job never build CubeCL. Backend features pass through (`metal`, `cpu`,
  `cuda`). `FmmBuilder` gets a backend setting with `Host` as the default, and `Fmm`
  reports the backend and device it runs on. The names are those of device-path.md
  §3.3 (`Backend`, `FmmBuilder::backend`, `OperatorKind`, `Fmm::device_report`).
- **Runtime-generic kernels.** Every kernel runs on every CubeCL runtime and is generic
  over f32/f64. The runtime is a run-time value (`Device` in `nd-fmm-kernels`, `Backend`
  in `nd-fmm-exec`), never an `R: Runtime` type parameter (device-path.md §3.2). p,
  tile sizes and cube layouts are comptime parameters, and every layout choice is per
  backend (design §6.5: the best layout differs between Metal and the CPU
  runtime by up to 3.4×).
- **M2M and L2L on the device always use the dense octant tables**, gathered and
  applied as batched GEMM per (level, octant), whatever the M2L strategy (design §4,
  §6.4). They are small: 8 (p + 1)⁴ reals per family. The device path builds them for
  the `Rotation` strategy too.
- **M2L strategies on the device:**
  - `Dense`, the 316 offset tables as GEMM. The library matmul (CMMA) is used for f32
    at p ≥ 8 where CMMA is available, and the hand-written comptime-p kernel otherwise
    and for every f64 run. **Changed on 2026-10-04 (after T9's measurement):** by
    default M2L runs the hand-written kernel at every p (the library pads each offset
    chunk to its widest batch and was slower on every FMM level measured on Metal);
    the library stays selectable as `DeviceGemm::Library`, and M2M and L2L keep the rule.
  - `Rotation`, the tables of `RotationTables` in a rotation kernel.
  - `Classes` is a host memory option. The device runs it as `Dense`, from
    `M2lClasses::expand`, and reports it as such; the host keeps its class tables for
    the fallback (device-path.md §6.8).
  - The matmul strategy is always chosen explicitly, never by the library's own
    per-call `Strategy::Auto` (requirement 6).
  - A library matmul strategy is used only if its inputs are multiplied in T itself. A
    strategy that rounds f32 inputs to TF32, F16 or BF16 is never used, by default or
    by autotune. On Metal the CMMA path uses f32 matrix units. On CUDA the MMA paths
    CubeCL offers are F16, BF16 and TF32 only (Phase 0 spike), and TF32 keeps a 10-bit
    mantissa, far coarser than the 1e-5 that f32 reaches at p = 8.
- **Strategy selection.** Before C4.7, the device default follows design §4:
  - f32: dense at every p ≤ 8;
  - f64: dense for p ≤ 11 and rotation for p ≥ 12, provisional until a GPU measures
    rotation (device-path.md §10.5).

  C4.7 replaces the rule with autotune where a tuning cache exists. The rule stays as
  the fallback for an untuned (backend, precision, p), including f64 on CUDA, which
  nothing here can measure.
- **The CubeCL CPU runtime** is the correctness backend for f64, and the CI candidate.
  - How it maps a launch (CubeCL 0.11.0-pre.4 sources, docs/design/device-path.md F18
    and F19; confirmed by T2): one task per unit of a cube, each unit's code looping over
    every cube in turn. The tasks run on a pool of one worker per logical CPU,
    efficiency cores included (pinning is only a hint on macOS); only kernels with
    `sync_cube` or shared memory get a dedicated worker per unit. A plane is one unit,
    and `sync_cube` is a spin barrier. SIMD comes only from
    `Vector<T, N>` vectors and from LLVM's vectorisation inside one unit's code, never
    from the units. A GPU-shaped kernel therefore runs poorly there by construction.
  - 0.10.0 compiled at LLVM optimisation level 0; 0.11 uses the O3 pipeline. The Phase
    0 CPU-runtime figures (design §6.1) are superseded by T2's re-measurement.
  - In 0.11, cubecl-opt's `InstCombinePass` fuses every product whose only use is an add
    or subtract into an fma, on every backend, the CPU runtime included, with no switch
    (T2; device-path.md F16, §5.3). Device kernels are therefore not bit-identical to
    the unfused host code wherever an inexact product feeds an add, unless T3 finds a
    formulation that keeps the fusion out.
  - It also gets a performance target (decision 10, signed off on 2026-10-03). At one
    thread, T3's CPU-shaped P2P took 1.004× the time of `nd-fmm-simd`, so T6 adds a CPU
    layout of the device P2P kernel within 1.5× of `nd_fmm_simd::P2pKernel` per pair at
    one thread. The all-cores ratio (2.73× in T3) is reported, not targeted. Nothing
    else on the CPU runtime is timed as a result.
  - Shapes are small: shared-memory kernels are emulated very slowly there, and some
    library f32 kernels took minutes to compile (design §6.1). Library-matmul tests run
    on Metal; the CPU runtime tests the hand-written kernels.
  - It has its own worker pool, so the device path never runs it inside a rayon worker
    (design §6.8). With the CPU backend no rayon pool is built, `threads(n)` caps the
    units per cube of the CPU layouts, and host-fallback kinds run serially. With
    Metal or CUDA the pool of `threads(n)` serves host-fallback kinds only and is idle
    while device work runs (device-path.md §11).
  - Its build downloads the `tracel-llvm` bundle. That fails inside a sandbox; run such
    builds outside it and say so.
- **Metal** needs a process with GPU access. It fails inside the macOS sandbox ("No
  possible adapter available"); run Metal tests outside it and say so. It reports no
  f64. The `metal` feature is wgpu with the MSL compiler; a device that comes up
  without it is refused (device-path.md §3.1). Switching to `metal-native` needs a
  separate sign-off (decision 11).
- **Accuracy measures.** Phases 1–3S define the measures, and every test names its own.
  Every f64 bound below applies to the CubeCL CPU runtime, the only backend here that
  runs f64 (and to CUDA, if someone later runs it). Metal runs f32 only, so on Metal
  only the f32 bounds apply.
  - **Operators**, against `nd-fmm-ref` and the host `LaplaceOperator` at the same
    frames, per degree in the §3.8 weighting, relative to the term magnitudes, on
    levels 2, 9 and 16 of a dyadic domain:
    - f64 on the CPU runtime: for dense M2M, L2L and M2L, 1e-14, or twice the host
      operator's measured error on the same cell, whichever is larger (the device GEMM
      sums in its own order; device-path.md §2); 1e-13 for rotation and the leaf
      operators (the C3.1 bounds);
    - f32 on Metal and on the CPU runtime: 1e-5 against the f64 reference (the
      Phase 1 f32 bound).
  - **P2P:** the contract T3 proposed, signed off on 2026-10-03 (decision 3). It is that
    of C3S.4, unchanged: pair terms within 8 u_T (potential) and 16 u_T (gradient) of
    `nd_fmm_ref::p2p`, and sums within 1e-14 (f64, CPU runtime) and 1e-6 (f32, Metal
    and CPU runtime) of `direct_sum`, relative to the term magnitudes, or twice the
    reference's error where that is larger. On a backend that flushes subnormals
    (Metal), it applies to charges q = 0 or |q| ≥ 2⁻¹⁰⁰ (CONVENTIONS §3.13, "Device
    kernels"). Kernels follow T3's formulation rules (spikes/device-arith/REPORT.md,
    "Recommendation").
  - **FMM:**
    - the device output against the host output of the same settings (precision, p,
      strategy, tree): relative L2 within 1e-12 (f64, CPU runtime) and 1e-5 (f32,
      Metal and CPU runtime), for φ and ∇φ;
    - the errors against `direct_sum` as in docs/phase3/README.md ("Error measures": a
      root mean square over eight charge vectors, 1,000 sampled targets): within 0.1%
      (f64, CPU runtime) and 5% (f32, Metal and CPU runtime) of the host run's;
    - the design's expectation is "f64 to the printed digits, f32 within a few per
      cent" (design §7, "Recommendation for Phase 4").
  - If a check fails, the task reports the measured errors and their breakdown (by
    operator kind, using the host fallback to isolate it) and stops. It does not tune
    tolerances to pass.
- **Workloads** (design §7, "Recommendation for Phase 4"):
  - the uniform cube and the Plummer sphere (a = 0.1), N = 10⁵, `max_level` 16, 64
    points per leaf, eight charge vectors, gradients on;
  - at (f32, p = 3), (f32, p = 8), (f64, p = 8), (f64, p = 12) and (f64, p = 18);
  - the Gaussian clusters are the optional third, a stress case for batch sizes;
  - the f64 points run on the CPU runtime. If a point takes more than about ten
    minutes there, it runs at N = 10⁴ with the host comparison at the same N, and the
    report says so;
  - f32 gradient tests avoid the sphere surface (its f32 floor, design §7).
- **Tables** come through `table_cache` in every timed run (dense table building is
  230 ms at p = 8 and 8.5 s at p = 16 on the host). Upload happens once, at build.
- **Timing** (design §8.3, requirement 11). Metal on the M3 Max only, release build:
  - kernel compilation excluded (a warm-up launch);
  - many launches queued per sync (a sync costs about 1.5 ms on wgpu/Metal);
  - the median over repeated batches, with the machine, backend, CubeCL version and
    device printed;
  - about ±25% run-to-run variance expected on a laptop GPU (Phase 0 T6), so each
    figure states the number of runs.

  Host comparisons use the Phase 3S host path (`P2pChoice::Auto`, NEON) at 1 thread and
  at 12 threads (the performance cores), with every BLAS thread variable set to 1.
- **Peaks** used for "% of peak" are those of the spike report (M3 Max GPU f32 14.3
  TFLOP/s, derived, not measured; 400 GB/s), restated in every report that uses them.
- **Errors.** Device failures at build (no adapter, unsupported precision, out of
  memory) are `SettingsError` or `FmmError` values, agreed on every rank by `build`'s
  existing all-reduce of input errors, with no new collective. A device failure during
  `evaluate` surfaces at the evaluation's one download as `FmmError::Device`, and the
  `Fmm` returns it from every later `evaluate` (device-path.md §12).
- **MPI.** The root rules apply. Device work never calls MPI, and every collective
  stays on the calling thread. On more than one rank `Fmm` still reports
  `PointsNotOwned` where it applies until C5.1, for either backend. Where it does not
  (every point in a leaf of its own rank), the host path runs distributed, and a
  device backend returns `SettingsError::DeviceNeedsOneRank` on every rank, with no
  collective (device-path.md §4.4). Device tests that build an `Fmm` go
  into the existing MPI-owning executables (`tests/mpi_exec.rs` and the ignored
  `accuracy.rs` and `adaptive.rs`), or a new executable with its own single
  MPI-initialising test.
- **Test time.**
  - `cargo test -p nd-fmm-kernels` without features builds and passes in seconds.
    Every runtime test is behind its backend feature.
  - T4 measures the CPU-runtime suite and fixes its budget. The target is under five
    minutes on the M3 Max, warm, in release, with large shapes in `#[ignore]` tests.
  - Metal tests are `#[ignore]` and run by hand.
  - The debug run of `cargo test -p nd-fmm-exec` without `gpu` keeps its one-minute
    budget and does not change.
- **Dependencies.** No new external dependency in any crate beyond the pinned CubeCL
  entries, used by `nd-fmm-kernels` and spikes only. Anything else needs asking first.

## Exit gate
- Every acceptance test in the task briefs passes:
  - in CI for the default members;
  - in the CPU-runtime job, if it is kept;
  - on the M3 Max for the CPU runtime and Metal, run by hand and reported with the
    backends run.
- `docs/design/device-path.md` is signed off before T4 starts. The §3.13 addition and
  the device P2P contract are signed off before T6 and T7.
- The CubeCL pin is `=0.11.0-pre.4` (with the matching `cubek-*`), and
  `spikes/cubecl-gemm` passes its tests on it. Its CPU-runtime figures are re-measured
  on 0.11, replacing the level-0 figures of Phase 0.
- T3 reports the CPU-runtime P2P against `nd-fmm-simd` with the outcome of its rule,
  and decision 10 is signed off before T6.
- C4.1:
  - device buffers round-trip bit for bit in f32 and f64 (sizes 0, 1, odd and large;
    −0, subnormals and the extremes);
  - f64 is refused cleanly on Metal and accepted on the CPU runtime;
  - plan views and tables are uploaded once per `Fmm`;
  - with every kind on the host fallback, the device path equals the host path bit for
    bit on every `tests/mpi_exec.rs` scenario.
- C4.2: the device P2P meets the signed-off contract on the CPU runtime (f32, f64) and
  on Metal (f32), with chunk and target-position invariance and determinism. It is
  timed on Metal against the host NEON kernel (C3S.4) at 1 and 12 threads, with its
  fraction of the peak model of device-path.md §13.4 (720 Gpairs/s φ, 480 φ and
  ∇φ). The target, measured and reported, never asserted: Metal f32 at least 25% of
  the model on W2 at N = 10⁵, and at least 10% on W1 at n_t = 64 with at least 4,096
  target leaves per launch. If decision 10 set a CPU-runtime target, the CPU
  layout of the kernel is within 1.5× of `nd_fmm_simd::P2pKernel` per pair at one
  thread on the W1 workload (geometric mean, measured on the M3 Max), and its
  all-cores ratio is reported.
- C4.3: P2M, L2P, P2L and M2P match `nd_fmm_ref::leaf` within the operator bounds above,
  with and without gradients, on levels 2, 9 and 16.
- C4.4: M2M and L2L match the C2.1 tables per level and octant, for the local and the
  global pass.
- C4.5: dense M2L matches C2.2 for every offset on levels 2, 9 and 16. The GEMM reaches
  at least 80% of the spike's throughput at the same (p, columns) on Metal (GEMM only,
  gather and reduction reported separately), and its efficiency is profiled and
  reported. (T9 met the 80% in 2 of 12 cells; accepted as analysed on 2026-10-04,
  decision 12.)
- C4.6: rotation M2L matches C2.3 (f64 on the CPU runtime to p = 20, f32 to p = 8), and
  is timed against C4.5 across p on Metal f32.
- C4.8 (new): with every operator on the device, on the cube and the Plummer sphere at
  the workload points above:
  - the output is within the FMM bounds of the host output;
  - the errors against the direct sum are within 0.1% (f64) and 5% (f32) of the host
    run's;
  - the C3.2 and C3.3 gates still pass;
  - two evaluations are bit-identical;
  - the transfers per evaluation are the design's minimum.
- C4.7: autotune picks the fastest measured strategy per (backend, precision, p), with
  the choice resolved at build and reported. The cache round-trips and is used only
  when the caller passes a directory, and a stale cache is rejected. Without a cache,
  the static rule applies.
- The T13 report gives, on Metal f32, stage timings of the device FMM against the host
  path, GEMM and P2P throughput against peak, launch and sync counts, the device
  leaf-size study, and states that no f64 GPU and no CUDA run was timed.

## Tasks

One pull request each.
- T1 and T2 touch disjoint files and can start at once. T3 needs T2 (it measures on
  0.11).
- T4 needs T1 signed off and T2. T5 needs T4.
- T6 and T7 need T5 and the T3 sign-off. T8 needs T5. T6, T7 and T8
  touch different kernels but all add to the device operator in `nd-fmm-exec`: merge
  one, then rebase the next.
- T9 needs T8 (it reuses its gather, GEMM and scatter). T10 needs T5 and, for its
  timing comparison, T9.
- T11 needs T6–T10. T12 needs T9, T10 and T11. T13 needs T11 and T12.

The signed-off design needs no `nd-fmm-plan` extension, so there is no task **T4b** in
Phase 4: on one rank the only evaluator-side write outside operator calls is `reset`'s
zeroing, which the device operator mirrors (device-path.md §4.3). The hook that C5.1
needs is sketched in device-path.md §4.5.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-device-design.md](T1-device-design.md) | `docs/design/device-path.md`: device residency, crate surfaces, kernel mapping, batching and order, precision, fallback, autotune, tests, task check | design for C4.1–C4.8 | Phase 3S complete |
| T2 | [T2-cubecl-011.md](T2-cubecl-011.md) | CubeCL pin to 0.11.0-pre.4, `spikes/cubecl-gemm` ported and re-measured, migration notes, root `CLAUDE.md` points to Phase 4 | prerequisite of C4.1 | none |
| T3 | [T3-device-arithmetic.md](T3-device-arithmetic.md) | `spikes/device-arith`: device arithmetic per backend; CONVENTIONS §3.13 addition for device kernels; the device P2P contract; a CPU-shaped P2P on the CPU runtime against `nd-fmm-simd` | prerequisite of C4.2, C4.3 | T2 |
| T4 | [T4-kernels-scaffold.md](T4-kernels-scaffold.md) | `fmm-kernels` crate, backends and capability check, device buffers, gather/scatter/zero kernels, crate `CLAUDE.md`, the CPU-runtime CI job measured | part of C4.1 | T1 (signed off), T2 |
| T5 | [T5-exec-device-path.md](T5-exec-device-path.md) | `nd-fmm-exec` feature `gpu`: backend setting, device operator with device-resident data and host fallback per kind, transfer accounting | C4.1 | T4 |
| T6 | [T6-p2p-kernel.md](T6-p2p-kernel.md) | device P2P kernel, potential and gradient, f32 and f64 | C4.2 | T5, T3 (signed off) |
| T7 | [T7-leaf-kernels.md](T7-leaf-kernels.md) | device P2M, L2P, P2L, M2P with harmonics by recursion | C4.3 | T5, T3 (signed off) |
| T8 | [T8-m2m-l2l-gemm.md](T8-m2m-l2l-gemm.md) | grouped translation per level: gather, grouped GEMM over the octants, row-ordered reduction (M2M) or scatter-add (L2L); the hand-written comptime-p GEMM | C4.4 | T5 |
| T9 | [T9-m2l-dense.md](T9-m2l-dense.md) | dense M2L per (level, offset): library CMMA and hand-written GEMM, launch batching, efficiency profile | C4.5 | T8 |
| T10 | [T10-m2l-rotation.md](T10-m2l-rotation.md) | rotation M2L kernel; timing against dense across p | C4.6 | T5, T9 |
| T11 | [T11-device-fmm.md](T11-device-fmm.md) | every operator on the device by default; launch scheduling, syncs, transfers; the device accuracy gates | C4.8 (new) | T6–T10 |
| T12 | [T12-autotune.md](T12-autotune.md) | strategy autotune keyed by (backend, precision, p, …), persistent cache, static fallback rule | C4.7 | T9, T10, T11 |
| T13 | [T13-benchmarks.md](T13-benchmarks.md) | device benchmark report, leaf-size study, design-document update | gate: benchmarks published | T11, T12 |

Review T1 and T3 yourself before the tasks that build on them. T4–T11 encode the
residency, batching and order of the design, and T6 and T7 encode the device arithmetic
and the P2P contract.

T2, T3, T4 and T5 change the root `Cargo.toml` or `Cargo.lock` (T3 adds a spike
member), and T4 may change `.github/workflows/run-tests.yml`. Rebase whichever merges later, and regenerate the
lock file with the manifest change (root `CLAUDE.md`, "Layout").

## Machines

| Machine | Backends | Used for |
| --- | --- | --- |
| Apple M3 Max (development) | Metal (f32; 40 GPU cores, plane size 32, 32 KB shared memory per cube); CubeCL CPU runtime (f32, f64) | every task's tests; every timing (Metal only) |
| GitHub Actions `ubuntu-latest` | CubeCL CPU runtime (f32, f64), if T4's job is kept | kernel correctness on small shapes; never timings |
| CUDA (A100/H100 class) | none available (decided on 2026-10-03) | `cuda` feature type-checked on the Mac; the CUDA run documented as one command for later |

## Decisions to sign off

Each is recorded in the exit checklist when made:
1. The design document `docs/design/device-path.md`, including any change it proposes
   to the requirements or tolerances above (T1; before T4). **Signed off on
   2026-10-03**, with every recommendation of device-path.md §16.
2. An `nd-fmm-plan` extension for the device path, if T1 proposes one (with T1).
   **Decided on 2026-10-03: none, no T4b** (device-path.md §4.3).
3. The §3.13 addition for device kernels and the device P2P contract (T3; before T6 and
   T7). **Signed off on 2026-10-03**, with every recommendation of
   spikes/device-arith/REPORT.md, "Recommendation":
   - the addition "Device kernels" to CONVENTIONS §3.13, with `CONVENTION_VERSION`
     still 1;
   - the C3S.4 contract unchanged (on flushing backends for q = 0 or |q| ≥ 2⁻¹⁰⁰);
   - the formulation rules: ŷ by an explicit `fma(r̂, u_s, ĉ)`, `inverse_sqrt` with no
     Newton step, and masking by compare and select;
   - no compiler option.
4. GPU hardware. **Decided on 2026-10-03: none besides the M3 Max.** Metal f32 is the
   only timed backend. f64 is checked on the CubeCL CPU runtime and never timed on a
   GPU. CUDA is type-checked, and its run is documented for later.
5. The CubeCL version. **Decided on 2026-10-03: move to 0.11.0-pre.4 now** (T2).
   Moving to 0.11.0 once it is released is a separate decision.
6. A host batched-GEMM path through BLAS. **Decided on 2026-10-03: deferred**, not in
   Phase 4 (a candidate for Phase 6).
7. The CPU-runtime CI job for `nd-fmm-kernels`. **Decided on 2026-10-03: T4 measures
   it**, and it is kept, changed or dropped at sign-off with T4's numbers (build time
   cold and cached, test time, bundle download). The answer also decides whether
   `nd-fmm-kernels` becomes a default member. **Signed off on 2026-10-03: kept as built
   in T4** (`run-tests-kernels`; measured on PR #51: 4 min 53 s cold, 51 s warm, of
   which 17.9 s bundle download cold and 0.5 s kernel compilation). `nd-fmm-kernels`
   stays a member, **not a default member**. Also accepted at the T4 sign-off: with
   `metal`, CubeCL's `persistence` feature is on through CubeCL's own manifests, and a
   Metal run creates an empty store `target/environment/default.db` (device-path.md
   §3.4).
8. The device leaf-size default (T13). Recommended: T13 applies the Phase 3S rule to the
   device backend only, and the host default stays 64.
9. External baselines (FMM3D, ExaFMM-t, kifmm-rs; design §8.3). Recommended: later, not
   in T13.
10. A performance target for the CubeCL CPU runtime (T3; before T6). **Approved on
    2026-10-03: T3 measures it.** T3 times a CPU-shaped P2P on the 0.11 CPU runtime
    (targets in `Vector<T, N>` lanes, one unit per core, no shared memory) against
    `nd_fmm_simd::P2pKernel` on NEON, on the Phase 3S W1 workload. The rule is fixed in
    the T3 brief: if the geometric-mean time ratio at one thread is at most 1.5, the CPU
    runtime gets a target in Phase 4 (its P2P within 1.5× of the host SIMD P2P per
    pair), and T6 adds a CPU layout of the device P2P kernel to meet it. Otherwise it
    stays a correctness backend. Either way, the result is signed off before T6.
    **Signed off on 2026-10-03: CPU-runtime target set.** The one-thread geometric mean
    is 1.004 (24 W1 cells) and the all-cores ratio 2.73 (spikes/device-arith/REPORT.md).
    T6 adds the CPU layout and meets the 1.5× target at one thread. The all-cores ratio
    is reported only.
11. The Metal runtime (device-path.md §16, question 10). **Decided on 2026-10-03: keep
    `metal` = wgpu-msl.** If T3 finds that its default math mode breaks the §3.13
    argument and no formulation restores it, switching the `metal` feature to
    `metal-native` (`cubecl-metal`, safe math mode) needs a separate sign-off
    (device-path.md §5.3).
12. The C4.5 GEMM gate (T9, PR #55). T9 measured the GEMM alone on the spike's shapes
    at 67–87% of a same-day spike run for the library (p ≥ 8) and 38–102% for the
    hand-written kernel (p = 4): met in 2 of 12 cells. The cause, measured in the same
    process, is the operand orientation fixed by device-path.md §6.4 (each box's n
    coefficients contiguous), which costs the library 16–29%. **Decided on 2026-10-04:
    the gate is accepted as analysed, and the coefficient-major layout that would give
    the GEMMs the spike's orientation is deferred to T12** (T12 brief, "Do").

## Risks

| Risk | Mitigation |
| --- | --- |
| CubeCL 0.11.0-pre.4 is a pre-release; its API or behaviour changes again before 0.11.0 | one pin for the whole workspace; CubeCL code only in `nd-fmm-kernels` and spikes, behind thin wrappers; T2 records the API notes; moving to 0.11.0 is a separate, small task |
| f64 is never run on a GPU, so f64 device performance and the f64 dense/rotation crossover stay unmeasured | f64 correctness on the CPU runtime; the static rule of design §4 as the untuned fallback; the CUDA command documented; T13 states plainly what was not measured |
| GPU compilers use fast math (reassociation, approximate `rsqrt`, flush to zero) and break the r² = 0 rule or the accuracy contracts | T3 measures each backend before any production kernel; the §3.13 addition states what kernels may assume; every kernel is tested against `nd-fmm-ref` on each backend run |
| The evaluator writes host buffers outside operator calls, and device-resident data goes stale | T1 listed every such write per stage, on one rank and on several (device-path.md §4.2): on one rank only `reset`'s zeroing, which the device operator mirrors, and no T4b; T5 tests the device path with host fallback bit for bit against the host path |
| Launch overhead and syncs dominate at small levels and small N (13 MFLOP took 9–22 µs in the spike; a sync costs 1.5 ms) | many launches per sync, batched offsets (design §6.5), top levels on the device, merged across levels if measured worthwhile (T11, device-path.md §6.7); launch and sync counts reported |
| The CPU runtime is too slow to build or to compile kernels for CI | T4 measures it; small shapes; hand-written kernels only there; a dropped CI job means the kernel tests run by hand, as the GPU tests do |
| The library matmul picks its kernel per call and breaks determinism, or is slow or unavailable for some shapes | explicit strategies only (never the library's per-call `Auto`); the hand-written kernel as the fallback for every shape; determinism tests |
| Device memory at high p (dense M2L 492 MB in f64 at p = 20) | rotation above the crossover; T1 states memory per (N, p, strategy); `Fmm` refuses a configuration that does not fit, at build |
| Unified memory on the M3 Max hides transfer costs that a discrete GPU would pay | T5 counts transfers in bytes and calls whatever the backend does; T13 reports them, so the cost on a discrete card can be estimated |
| The leaf-operator kernels differ from `nd-fmm-math`'s recursion and drift in accuracy at high p | T7 ports the recursion, tests it against `nd-fmm-ref` per degree to p = 20 in f64, and compares with the Phase 1 f32 bounds |

## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase4/T<k>-<name>.md and do that task." Review and merge before the next.

## Exit checklist
- [x] T1 merged: `docs/design/device-path.md` drafted, sign-off questions listed
- [x] Device-path design signed off, including any requirement or tolerance change; T4b decided (needed or not): signed off 2026-10-03 with every recommendation of device-path.md §16; T4b not needed
- [x] GPU hardware: none besides the M3 Max; Metal f32 timed, f64 on the CPU runtime untimed, CUDA type-checked (decided 2026-10-03)
- [x] CubeCL pin: 0.11.0-pre.4 (decided 2026-10-03)
- [x] Host BLAS GEMM path: deferred (decided 2026-10-03)
- [x] Metal runtime (decision 11): `metal` stays wgpu-msl; `metal-native` only by a separate sign-off (decided 2026-10-03)
- [x] C4.5 GEMM gate (decision 12): accepted as analysed (met in 2 of 12 cells, cause measured); the coefficient-major GEMM layout deferred to T12 (decided 2026-10-04)
- [x] T2 merged: pin at 0.11.0-pre.4, `spikes/cubecl-gemm` passing and re-measured, migration notes, root `CLAUDE.md` points to Phase 4
- [x] T3 merged: device arithmetic measured per backend; §3.13 addition and device P2P contract drafted
- [x] §3.13 addition and device P2P contract signed off (2026-10-03)
- [x] CPU-runtime target (decision 10): T3's ratio against `nd-fmm-simd` reported; target set (≤ 1.5×, CPU layout in T6) or CPU runtime correctness-only: target set, signed off 2026-10-03 (one thread 1.004×, all cores 2.73×)
- [x] T4 merged: `nd-fmm-kernels` skeleton, round trips and capability check pass, CPU-runtime CI job measured
- [x] CPU-runtime CI job: kept / changed / dropped; default-member status of `nd-fmm-kernels` decided: kept, not a default member (signed off 2026-10-03)
- [x] T4b merged (only if needed): `nd-fmm-plan` device hooks, `IndexFmm` on 1, 2 and 4 ranks, host path bit-identical: not needed (decision 2, 2026-10-03)
- [x] T5 merged: device path with full host fallback bit-identical to the host path; transfers counted
- [x] T6 merged: device P2P within the contract on every backend run; timed against host NEON
- [x] T7 merged: P2M, L2P, P2L and M2P within the operator bounds on levels 2, 9 and 16
- [x] T8 merged: M2M and L2L equal the C2.1 tables per level and octant, both passes
- [x] T9 merged: dense M2L equals C2.2; GEMM at least 80% of the spike's throughput; efficiency profiled
- [x] T10 merged: rotation M2L equals C2.3; timed against dense across p
- [ ] T11 merged: every operator on the device; C4.8 gates on cube and Plummer; deterministic; minimal transfers
- [ ] T12 merged: autotune with persistent cache; static fallback rule; choice resolved at build and reported
- [ ] Device leaf-size default decided (T13)
- [ ] External baselines: later / in T13
- [ ] T13 merged: device benchmark report published
- [ ] Design documents updated: laplace-fmm-plan §4, §6, §7 (Phase 4 status and numbers), §8.3, §9.1 and §9.2; workspace-structure §2, §3, §3.1, §5 and §6; device-path.md decisions and measurements recorded

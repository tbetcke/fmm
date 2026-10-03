# Phase 4 / T1 — design of the device path (design for C4.1–C4.8)

Phase 4 runs the Laplace FMM on a GPU through CubeCL. Design §6 covers the kernels in
outline. fmm-plan-redesign §6.4 sketches a GEMM M2L on the batched interface, and §10
leaves the device hook to Phase 4. What is missing is a design that ties them together:
- where data lives during an evaluation;
- how a device operator stays consistent with the `Evaluator`, which owns host
  buffers and writes some of them itself;
- how kernels batch and order their work under the plan's accumulation rule;
- how precision, fallback, autotune and testing fit.

This task writes that design as `docs/design/device-path.md`, for sign-off by hand
before any production code. It is the Phase 4 counterpart of Phase 3 T1 and of
simd-p2p.md, and it writes no Rust.

The development machine is an Apple M3 Max: Metal (f32 only) and the CubeCL CPU
runtime. No CUDA card is available (docs/phase4/README.md, "Machines"). The design
targets CUDA as well, but nothing in the phase measures it.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-plan/CLAUDE.md, fmm-simd/CLAUDE.md;
- docs/phase4/README.md, closely: "Requirements on the device path" is what this design
  answers; also "Design decisions" and "Exit gate";
- docs/design/laplace-fmm-plan.md:
  - §4 and §5 (interfaces, data layout);
  - all of §6 (CubeCL facts, precision, kernel mapping, batching, atomics, autotune,
    threads);
  - §7, Phase 2 recommendation, Phase 3 ("Per-pair cost", "Recommendation for Phase
    4"), Phase 3S and the Phase 4 table;
  - §8 and §9;
- docs/design/fmm-plan-redesign.md §4.3–§4.5, §5, §6 (with the GEMM check of §6.4), §7
  (with §7.5) and §10;
- docs/design/simd-p2p.md §3 and §5.5 (requirements and reproducibility of a fast P2P,
  the model for the device P2P);
- spikes/cubecl-gemm/SPIKE_REPORT.md and its kernels (`src/tiled.rs`, `src/bench.rs`);
- the code the device path plugs into:
  - `nd_fmm_plan::{operator, lists, store, evaluator, exchange}`: the batch types,
    `Csr`, `GroupedCsr` and its raw accessors, `LevelBuffers`, `LeafStore`, the stage
    methods, `reset`, the coarse gather;
  - `nd_fmm_exec::{operator, fmm, tables, geometry, threading}`: `LaplaceOperator`,
    its `Kernels::*_target` bodies, `FmmBuilder`, `Fmm::evaluate`, `StageTimings`;
  - `nd_fmm_tables::{MatrixSet, RotationTables, ShiftTables}`, with the column-major
    layout of CONVENTIONS §3.12 "Matrix layout";
- CONVENTIONS §3.6–§3.8, §3.11 ("L2P and M2P", "Coaxial translations", "Rotation of
  coefficients"), §3.12 and §3.13;
- the CubeCL 0.11.0-pre.4 and `cubek-matmul` 0.3.0-pre.4 sources and docs: the runtime
  and client API, buffers and handles, launch and comptime parameters, `Line`, shared
  memory, plane operations, streams and synchronisation, device properties and
  `supports_type` (or its 0.11 successor), autotune and its cache, compilation options
  (fast math), and the matmul strategies.
  - Fetch them with `cargo fetch` on T2's branch, or download the crates from crates.io
    into the scratch directory.
  - If T2 has merged, also read its migration notes in the spike report.

Write `docs/design/device-path.md` with these sections:

1. **Starting point.**
   - What Phase 3 and 3S built that the device path uses. The batched calls and both
     views of each list; the level buffers (column-major (p + 1)² × boxes, a GEMM
     operand); the CSR leaf stores; `LaplaceOperator` and its per-target bodies.
   - The CubeCL 0.11 facts the design relies on, each with a source file or doc
     reference. Mark anything inferred rather than read.
2. **Requirements.** Restate requirements 1–11 of docs/phase4/README.md and say how the
   design meets each. If one should change, say why and propose the change as a
   sign-off question. Do the same for the tolerances of "Accuracy measures".
3. **Crates and public surfaces.**
   - `nd-fmm-kernels`: the backend and device handle; the capability query; device
     buffers; the launch wrappers per kernel family, as Rust signatures with doc
     comments; the error type.
   - What is generic over the CubeCL runtime and what is chosen at run time. Compare
     generic code (`R: Runtime` threaded through `nd-fmm-exec`) with an enum over the
     compiled-in runtimes, and recommend one.
   - The `nd-fmm-exec` side: the backend setting on `FmmBuilder`; the device operator
     type and how it relates to `LaplaceOperator` (wraps it for the fallback, or shares
     its tables); what `Fmm` reports (backend, device, resolved strategies, transfers).
   - The cargo features (`gpu` in `nd-fmm-exec` and `nd-fmm-validate`; `cpu`, `metal`
     and `cuda` in `nd-fmm-kernels`) and what each builds.
4. **Device data and residency.**
   - Which buffers live on the device (level buffers, leaf stores, plan views, tables,
     frames, scratch), when each is uploaded or downloaded, and the transfers per
     evaluation in bytes and calls, as a formula in N, the boxes per level and p.
   - **Every host-side write the `Evaluator` makes outside operator calls**, per stage:
     `reset`, the source exchange, the coarse gather, the multipole exchange. For each,
     say what it does on one rank and on several (read the code; the coarse gather may
     copy a rank's own blocks even on one rank). Then show how the device copy stays
     consistent with it.
   - Weigh the options, at least:
     - (a) the operator keeps device mirrors, and `Fmm` tells it when host data
       changed, with no plan change;
     - (b) an `nd-fmm-plan` hook that routes the four data movements of
       fmm-plan-redesign §10 through a backend;
     - (c) anything better that you find.

     Recommend one. If it needs an `nd-fmm-plan` change, design it as a general,
     kernel-agnostic extension checked with `IndexFmm`, show that the host path stays
     bit-identical, and write its task brief (`docs/phase4/T4b-plan-device-hooks.md`, in
     the style of the other briefs) in this PR.
   - Device memory per (N, p, strategy, precision), against the M3 Max and a 40 or 80 GB
     card, and what `Fmm` does when a configuration does not fit.
5. **Precision and capability.** The f64 check on each backend, the refusal path
   (requirement 7), and the CPU runtime as the f64 correctness backend.
6. **Kernel mapping.** For each operator, refining design §6.4:
   - the parallel unit, the comptime parameters, the per-backend cube layout and the
     shared-memory use;
   - how the kernel follows the accumulation rule (requirement 4), and the fixed
     internal order of each contribution;
   - the data it reads (which views, which raw arrays).

   In particular:
   - **P2P:** one cube per target leaf (or per block of targets); sources tiled through
     shared memory in near-row order; the mapping ŷ = ĉ + r̂ u_s of §3.13 with exact
     frames, computed on the device from integer keys or uploaded per near entry; the
     r² = 0 rule; leaves larger than a tile; empty leaves. Keep room for a second,
     CPU-shaped layout on the CubeCL CPU runtime (targets in `Line<T>` lanes, one unit
     per core, no shared memory), which decision 10 of the README adds to T6 if T3's
     measurement against `nd-fmm-simd` meets its rule. Say how a per-backend layout is
     selected; do not design the CPU layout in detail.
   - **P2M, L2P, P2L, M2P:** harmonics and their gradients by the Cartesian recursion
     of CONVENTIONS §3.5 in registers. Whether recursion coefficients are precomputed
     on the host (from `nd-fmm-math`) or formed in the kernel. M2P needs I_(p+1).
   - **M2M and L2L:** gather → GEMM → scatter-add per (level, octant), the global pass
     included.
   - **Dense M2L:** gather → GEMM → scatter-add per (level, offset), with the launch
     structure of design §6.5. Compare one launch per offset in index order, a grouped
     launch, and the "stacked" variant with a reduction per target in row order, by
     launch count, temporary memory, conflict-freedom and order. Then choose. A grouped
     launch whose cubes write the same target from different offsets breaks
     requirement 5.
   - Library matmul (CMMA) against the hand-written comptime-p kernel per (precision,
     p), and how the matmul strategy is fixed explicitly (requirement 6).
   - **Rotation M2L:** rotation, coaxial translation and rotation back from
     `RotationTables`, per pair or per target, the per-degree products in shared
     memory, and the table layout on the device.
   - **Top levels** with few boxes: merged launches, or the host fallback (design
     §6.5).
   - The `Classes` strategy on the device: run as `Dense` from `M2lClasses::expand`, or
     refuse (README, "M2L strategies").
7. **Host fallback and migration.** How each operator kind runs on the host with
   explicit transfers (requirement 8), how it is selected per kind, and how T5–T11 move
   the kinds onto the device one at a time. With every kind on the host the output must
   equal the host path bit for bit; say why it does.
8. **Launches, synchronisation and timing.**
   - Launches per level and per evaluation, and the syncs: the target is one sync per
     evaluation, when the output is downloaded.
   - How `StageTimings` are measured without a sync per stage (device timestamps, if
     CubeCL offers them, or an opt-in synchronised mode for reports).
9. **Determinism and accuracy.**
   - What makes the output bit-identical run to run (requirement 6), and what makes it
     differ from the host path: summation order inside a product, fma contraction,
     the inverse square root.
   - The tolerances of the README, confirmed or replaced with a reasoned proposal.
   - What T3 must establish about device arithmetic before T6 and T7.
10. **Autotune (C4.7).**
    - CubeCL's own autotune against a strategy-level tuner in `nd-fmm-exec`.
    - The key: backend, device, precision, p, and boxes or pairs per level, bucketed.
    - The candidates: dense with the library or the hand-written GEMM, rotation, and the
      cube layouts and tile sizes.
    - When tuning runs: at build, timed outside `evaluate`.
    - Persistence. The rule of the host table cache applies: a directory only when the
      caller passes one, no default directory, no environment variable. If CubeCL's own
      cache cannot honour that, say so and design around it.
    - Stale-cache rejection, and the static fallback rule (README, "Strategy
      selection"), with the f64 rule for p = 9–11 stated.
11. **Threads and BLAS.** The CubeCL CPU runtime's worker pool against `threads(n)`
    (design §6.8). The runtime starts one OS thread per unit of a cube, and each loops
    over every cube, so its thread count is the cube size, not a setting. Refuse
    `threads > 1` with that backend, or keep rayon idle while device work runs.
    Recommend one, and say how cube sizes keep ranks × runtime threads within the
    physical cores. Phase 4 makes no BLAS call.
12. **Errors.** Device failures at build and during `evaluate`: which are values, which
    are panics, and how they are agreed on every rank without a new collective
    (README, "Errors").
13. **Testing and benchmarking.**
    - The test layers per task: the kernel against `nd-fmm-ref` and the host operator;
      the device operator against the host operator per kind; the FMM against the host
      FMM and the direct sum.
    - Which backend runs what: the CPU runtime with small shapes; Metal `#[ignore]`d
      and run by hand.
    - Compile-time budgets, and how every test prints the backends it ran.
    - The benchmark harness and its metrics (design §8.3): stage times, GFLOP/s and
      GB/s against the peaks of the spike report, P2P pairs per second against host
      NEON, launches, syncs and transfers.
    - The performance targets: C4.5's 80% of the spike's GEMM throughput, and a stated
      target for C4.2 (a fraction of a stated peak model, derived here).
14. **Multi-rank and overlap compatibility.** What C5.1 (device on several ranks:
    exchanges staged through host buffers) and C5.2 (overlap) need from this design,
    and why it does not preclude them. Do not design them. State whether overlapping
    P2P with the far field on a second stream is worth proposing later, given that it
    changes the accumulation order of fmm-plan-redesign §7.5.
15. **Task check.** Map the design onto T4–T13 (and T4b, if needed). Name each task's
    modules and tests. Say where a brief needs changing, and propose the change; do not
    rewrite the briefs except for T4b.
16. **Questions for sign-off.** At least:
    1. the requirements and tolerances, with any change proposed;
    2. the residency option, and whether an `nd-fmm-plan` extension (T4b) is needed;
    3. runtime-generic code or a run-time backend enum;
    4. the dense M2L launch structure (per offset, grouped, stacked);
    5. `Classes` on the device: dense or refused;
    6. the f64 static rule for p = 9–11;
    7. the CPU runtime against `threads(n)`;
    8. autotune: CubeCL's own or a strategy-level tuner, and its persistence;
    9. the C4.2 performance target.

Keep the document in the style of the other design documents: short sections, tables
where they help, Rust signatures where they decide something, no code beyond sketches.
Mark every number that is a model or an estimate as such.

The PR description must contain:
- a one-page summary of the design;
- the sign-off questions with a recommendation for each;
- the list of evaluator-side host writes found in section 4, with file and function
  names.

Must pass: nothing builds differently. Run `cargo fmt -- --check` and
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` once, to confirm the starting
point, and report the result.

Do not:
- write or change any Rust, or the CubeCL pin (T2);
- change docs/phase4/ briefs other than adding T4b if needed, or fixing a factual error
  (say which);
- change nd-fmm-plan or nd-octree. If a requirement cannot be met without changing
  them, say so in the document and in the PR, and propose the smallest change as T4b.

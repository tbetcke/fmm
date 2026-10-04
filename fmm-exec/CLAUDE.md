# nd-fmm-exec

Purpose: the Laplace FMM on top of `nd-fmm-plan`: `LaplaceOperator`, which implements
the plan's level-batched operator interface from the Phase 2 tables and the
`nd-fmm-ref` leaf operators, P2P through the SIMD kernel of `nd-fmm-simd`, box geometry
from Morton keys, and the user-facing `FmmBuilder` and `Fmm`; behind the feature `gpu`,
the device path (`device`: `DeviceOperator`, its report and transfer accounting).
Phase and components: Phase 3, C3.1–C3.3 and C3.5 (tasks T3 and T8–T11 in docs/phase3/);
Phase 3S, C3S.5 (task T6 in docs/phase3s/); Phase 4, C4.1 (task T5 in docs/phase4/;
design docs/design/device-path.md), C4.2 (task T6: P2P on the device), C4.3 (task
T7: P2M, L2P, P2L and M2P on the device), C4.4 (task T8: M2M and L2L on the device),
C4.5 (task T9: dense M2L on the device), C4.6 (task T10: rotation M2L on the device) and
C4.8 (task T11: the device FMM end to end, every kind on the device), with C4.7 to
follow.

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  - Box geometry and table order follow §3.12; leaf data, relative frames and output
    scaling follow §3.13.
  - Operator formulas are reached only through nd-fmm-tables and nd-fmm-ref, and for
    P2P also through nd-fmm-simd (`P2pKernel::evaluate`). Do not re-derive one here.
  - P2P runs the kernel of `P2pChoice` (default `Auto`, the widest ISA). Keep
    `P2pChoice::Reference` (`nd_fmm_ref::p2p::p2p`) as the trusted slower path: tests
    compare every ISA with it. `nd-fmm-exec` itself stays free of `unsafe`.
- Geometry comes from integer keys: tables by `morton::child_index` and V-list offset,
  frames between boxes from `geometry::relative_frame`. Never form a shift as a
  difference of floating-point centres in library code.
- Every operator accumulates (+=). 1/(4π) and the leaf radius are applied once, when
  `Fmm` produces output (§3.1, §3.13), never inside an operator.
- Generic over `T: SimdScalar + Equivalence` (`SimdScalar` of nd-fmm-simd, re-exported
  from `operator`; f32 and f64, like `RealScalar`). No allocation in operators: scratch
  is owned by the operator and sized at construction.
- The host path executes each level target by target, through the plan's
  target-centric view, each target's contributions in a documented order.
- Threading (from T10): rayon over the targets of a level call, in a pool owned by
  `Fmm`, opt-in (`threads`, default 1). The output is bit-identical for every thread
  count. No `unsafe`, no atomics, no global pool; per-thread scratch created at
  construction; worker threads never call MPI, which must provide at least
  `Threading::Funneled` when threads > 1.
- Threads and BLAS: no BLAS or LAPACK call inside a rayon worker in Phase 3; any later
  one runs single-threaded. Ranks × rayon threads × BLAS threads stay within the
  physical cores. BLAS threads are set by the launcher (`OPENBLAS_NUM_THREADS`,
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`);
  this crate reads and reports them and never sets an environment variable.
- Written only against the new nd-fmm-plan API (docs/design/fmm-plan-redesign.md),
  never against the old one that T7 removes.
- MPI: the root rules apply.
  - At most one MPI-initialising test per test executable. `tests/mpi_exec.rs` owns it
    for the scenario list; add scenarios to its `cases`, not new `#[test]`s. Large
    ignored tests get their own executable. `tests/mpi_threading.rs` owns MPI at
    `Threading::Single` (the error path of `threads`); `tests/mpi_exec.rs`,
    `tests/accuracy.rs` (the ignored C3.2 gate), `tests/adaptive.rs` (the ignored
    C3.3 error per list) and `tests/device_metal.rs` (the ignored Metal run of the
    device path, feature `metal`) initialise it at `Threading::Funneled`;
    `tests/device_fmm.rs` (the ignored C4.8 gate, feature `gpu`) at the default level.
    The device checks shared by `tests/mpi_exec.rs` and `tests/device_metal.rs` live in
    `tests/device_common/`.
  - New `Fmm` scenarios evaluate through `evaluate_threaded` in `tests/mpi_exec.rs`,
    which repeats them at 2, 4 and 8 threads and checks the output bit for bit, or
    through `evaluate_every_kernel`, which also repeats them with `Reference` and every
    available ISA and compares each ISA with `Reference`. It roughly triples the cost
    of a scenario, so it runs on the small ones only (debug run under a minute).
  - Operator and geometry tests do not initialise MPI.
  - Every error that depends on one rank's input is agreed by all ranks before the
    next collective.
  - Doctests that initialise MPI are `no_run`.
  - Until C5.1, `Fmm` does not redistribute points; on several ranks it reports
    `PointsNotOwned` on every rank.
- Tests name their error measure (docs/phase1/README.md, "Error measures", and
  docs/phase3/README.md). Operators are compared with nd-fmm-ref in a dyadic domain for
  tight tolerances.
- Keep the debug-mode test run under a minute. Large-N checks go into `#[ignore]`
  tests or nd-fmm-validate examples.
- The device path (Phase 4; docs/phase4/README.md requirements 3–9, device-path.md):
  - features: `gpu` (`nd-fmm-kernels`), and `cpu`, `metal`, `cuda`, each enabling `gpu`
    and the backend of `nd-fmm-kernels`; none on by default. `Backend::Host` stays the
    default, and without the features the host path builds and runs unchanged;
  - residency (requirement 3): plan views, geometry and tables uploaded once per `Fmm`,
    points once per build, charges and output once per evaluation; `Fmm::evaluate`
    drives `begin_evaluation` (after `reset`) and `read_output`. No `nd-fmm-plan`
    change (no T4b); a device backend runs on one rank (`DeviceNeedsOneRank` after
    step 5) until C5.1;
  - accumulation, no atomics, determinism (requirements 4–6): the kernels' rules in
    fmm-kernels/CLAUDE.md; every launch and transfer from the calling thread; nothing
    chosen per call;
  - precision (requirement 7): f64 refused where the device does no f64 arithmetic
    (`PrecisionUnsupported`), agreed by step 1's all-reduce, never a panic or a cast;
  - host fallback (requirement 8): every kind runs on the host through the wrapped
    `LaplaceOperator`'s own method with explicit transfers; with every kind there,
    bit for bit the host path (tested on every `tests/mpi_exec.rs` scenario);
  - device kernels (from T6): P2P runs on the device by default (`nd_fmm_kernels::p2p`,
    one launch per level, the layout by backend or `FmmBuilder::device_p2p_layout`), and
    from T7 so do P2M, L2P, P2L and M2P (`nd_fmm_kernels::leaf`, one launch per level
    call, the layout by backend or `FmmBuilder::device_leaf_layout`), and from T8 M2M
    (both passes) and L2L (`nd_fmm_kernels::translate::grouped` with the dense octant
    tables under every strategy, three launches per chunk, plans and scratch built at
    build, the GEMM by `FmmBuilder::device_gemm` and the chunks by
    `device_scratch_budget`, reported in `DeviceReport::translations`), and from T9 M2L
    under `Dense` and `Classes` (the same grouped translation over the level's V view
    and the 316 dense tables, `Classes` from `M2lClasses::expand` and reported as
    "Classes, run as dense on the device"; under the default `DeviceGemm::Auto` M2L runs
    the hand-written GEMM, the library only under `DeviceGemm::Library`, decided after
    T9's measurement), and from T10 M2L under `Rotation` (`nd_fmm_kernels::rotation::m2l`,
    one launch per level, the M2L family of the host's `RotationTables` uploaded once in
    its own storage through `device::RotationHostArrays`, the rows of each V view at
    build, the layout by backend in `DeviceReport::rotation_layout`, the calls in
    `DeviceReport::rotations`; M2M and L2L stay the dense octant GEMMs);
    `host_fallback` names a kind to keep it on the host.
    `tests/operator/device_leaf.rs` checks the four leaf operators against
    `nd_fmm_ref::leaf` (p ≤ 20 in f64) and the host operator;
    `tests/operator/device_translate.rs` checks M2M and L2L, and
    `tests/operator/device_m2l.rs` M2L for all 316 offsets on levels 2, 9 and 16, against
    `nd_fmm_ref::direct` and the host operator's error (p ≤ 20 in f64, the sweeps
    ignored), and `tests/operator/device_rotation.rs` rotation M2L the same way, against
    `direct`, the host's `RotationTables::m2l` and the device dense M2L (p ≤ 20 in f64,
    the sweep ignored). With a kind on the
    device the output agrees with the host path within the FMM bounds of
    docs/phase4/README.md, not bit for bit: `tests/device_common` checks both, every kind on the fallback bit for
    bit and the default within the bounds, with the transfer formula of each;
  - the device FMM (T11, C4.8): every kind on the device by default under every
    strategy; an evaluation moves the charges up and the output down and syncs once, at
    that download (`tests/device_common` asserts it on every scenario, with two builds
    bit for bit); level calls queue their launches without a sync, the top levels stay
    on the device, dense M2L is not merged across levels (measured negligible); stage
    timing (`device::StageTiming`, `FmmBuilder::device_timestamps`, opt-in): one
    timing window per stage with device work (`DeviceStage`, five per evaluation) where
    the device times on itself (Metal, CUDA), read after the download, in
    `StageTimings::device` (spans that overlap on Metal, not a breakdown); no window on
    the CPU runtime, whose windows drain the stream; `synchronous_stages` disables them. Never add a sync, a download or a host call to a
    default evaluation; a new launch belongs in the formula of `tests/device_common`;
  - safety (requirement 9): no `unsafe` here and no direct `cubecl` dependency;
    CubeCL only through `nd-fmm-kernels`;
  - threads: with `Backend::Cpu` no rayon pool, `threads(n)` caps the CPU runtime's
    units per cube; with Metal and CUDA the pool serves host-fallback kinds only;
  - every device test prints the backends it ran; Metal tests are ignored and run by
    hand outside the macOS sandbox (`tests/device_metal.rs`); CUDA is type-checked.
- Before finishing: `cargo clippy -p nd-fmm-exec --all-targets -- -D warnings` and
  `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` must pass. Tasks that add
  ignored tests must also pass
  `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored`. A change to
  the device path also needs:
  - `cargo clippy -p nd-fmm-exec --all-targets --features cpu,metal -- -D warnings`;
  - `cargo check -p nd-fmm-exec --features cuda` (type-checked, never run);
  - `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release` (every
    `tests/mpi_exec.rs` scenario repeated on the CPU runtime);
  - `cargo doc -p nd-fmm-exec --no-deps --features cpu`;
  - by hand on the M3 Max, outside the sandbox (build the tests sandboxed first):
    `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features metal --release --
    --ignored`;
  - from T6, the ignored gates on the device path too: `RUST_MIN_STACK=8388608 cargo
    test -p nd-fmm-exec --features cpu --release -- --ignored` (`tests/accuracy.rs` and
    `tests/adaptive.rs` repeat their problems with every kind on the device: on the CPU
    runtime in f64, `tests/accuracy.rs` also at p = 18 under `Rotation` (T10) with the
    C3.2 gate there (T11), and on Metal in f32 with `--features metal`; from T11
    `tests/device_fmm.rs`, the C4.8 gate: the cube and the Plummer sphere at N = 10⁵, f64
    p = 8, 12, 18 on the CPU runtime, f32 p = 3, 8 on Metal (with the Gaussian clusters
    at p = 8), against the host of the same settings, and the C3.3 gate on the device).

## Allowed dependencies
nd-fmm-math, nd-fmm-ref, nd-fmm-tables, nd-fmm-plan, nd-fmm-simd (from Phase 3S T6),
nd-octree, mpi, rlst, thiserror, rayon (from T10; no other crate gets it), and
nd-fmm-kernels (from Phase 4 T5, optional, feature `gpu`), all through
[workspace.dependencies]; dev-dependencies: proptest.
Not allowed:
- CubeCL directly (only through nd-fmm-kernels);
- nd-fmm-validate, not even as a dev-dependency (it depends on this crate).

Anything else needs a note in the PR.

## Test oracle
- `nd_fmm_ref::{direct, rotation, leaf, p2p}` at the absolute frames of §3.12, for
  every operator; for P2P with every `P2pChoice` the machine offers.
- The `P2pChoice::Reference` run of the same `Fmm` settings, for every ISA's output.
- `nd_fmm_ref::p2p::direct_sum` in f64 over the original coordinates, for every
  complete FMM.
- `nd-octree` and `nd-fmm-plan` for the octant and offset order of the tables.
- The single-translation prediction of Phase 1 (design §7) for the C3.2 gate.

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
C4.5 (task T9: dense M2L on the device), C4.6 (task T10: rotation M2L on the device),
C4.8 (task T11: the device FMM end to end, every kind on the device) and C4.7 (task T12:
autotune with a persistent cache, module `tune`); Phase 4S, C4S.4 (task T4: the device
FMM on CUDA), C4S.5 (task T5: per-kind timings, `FmmBuilder::kind_timings`) and C4S.8
(task T9: the output pass and the charge load in parallel, `Fmm::evaluate_into`, the output
pass on the device, `FmmBuilder::output_pass`) and C4S.10 (task T11: the download read in
place, the charges uploaded without a copy); Phase 5, C5.1 host (task T6 in docs/phase5/:
`Fmm` on any number of ranks through `nd_fmm_plan::redistribute::Redistribution`; design
docs/design/distributed-fmm.md) and C5.1 device with C5.2's device-resident ghost
buffers (task T8: the device operator on any number of ranks through the host-data
hook) and C5.2 (task T9: the exchanges overlapped with local work, `FmmBuilder::overlap`);
Phase 5N, C5N.3 (task T4 in docs/phase5n/: the M2L strategy rule on the node, the table
loading with a cache at scale, the ranks × threads guidance; design docs/design/node-m2l.md);
Phase 5S, C5S.3 (T6: the owned top tree in `Fmm`) and C5S.4 (T7: the work-weighted cut,
module `cost`; design docs/design/scale-out.md §6); Phase 6, C6.6 (T2 in docs/phase6/: the
host batched M2L, `M2lStrategy::Batched`, with `nd-fmm-simd`'s `MatrixKernel` and the
opt-in fused mode `FmmBuilder::m2l_mul_add`; design docs/design/optimisation.md §2) and
C6.3 (T5: several charge vectors, `FmmBuilder::vectors`, `Fmm::evaluate_many`,
`OutputMany`, host and device; design docs/design/optimisation.md §4) and C6.1 (T7: the
device's fused gather, reduction by chunk and shared-memory GEMM,
`FmmBuilder::{device_gather, device_reduction}`, `DeviceGemm::Shared`; design
docs/design/optimisation.md §6) and C6.4 (T8: dipole sources, `SourceType`,
`FmmBuilder::sources`, `Sources`, `Fmm::evaluate_sources`; T8d: the dipole kinds on the
device; design docs/design/optimisation.md §7; report fmm-validate/results/phase6-dipoles.md)
and C6.9 (T12: the near rule, `FmmBuilder::near_rule`, `NearRuleChoice`, `Fmm::near_rule`;
design docs/design/optimisation.md §8a; report fmm-validate/results/phase6-near.md) and
C6.5 (T13: the plane-wave M2L, `M2lStrategy::PlaneWave`, the level call in
`src/operator_plane_wave.rs`, `cost::M2lUnit`; CONVENTIONS §3.14; design
docs/design/optimisation.md §8.2, "As built in T13"; report
fmm-validate/results/phase6-planewave.md); and the defaults of T11 (decision 7 of
docs/phase6/README.md; report fmm-validate/results/phase6.md).

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  - Box geometry and table order follow §3.12; leaf data, relative frames and output
    scaling follow §3.13.
  - Operator formulas are reached only through nd-fmm-tables and nd-fmm-ref, and for
    P2P also through nd-fmm-simd (`P2pKernel::evaluate`). Do not re-derive one here.
  - P2P runs the kernel of `P2pChoice` (default `Auto`, `Isa::detect`: AVX-512 on an
    AVX-512 machine since Phase 6 T11; selectable as `Isa(Isa::Avx512)` since T6). Keep
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
  target-centric view, each target's contributions in a documented order. The one
  exception is M2L under `Batched` (Phase 6 T2): blocks of `M2L_BLOCK_TARGETS`
  consecutive targets, offsets in index order through the grouped view's batches, each
  local receiving exactly the chain of `Dense` (the `operator` module docs, "The batched
  M2L"). Keep `Batched` bit for bit `Dense` in its default mode; `Dense` stays the
  trusted path. The other is M2L under `PlaneWave` (Phase 6 T13, C6.5; the `operator`
  module docs, "The plane-wave M2L"): per vector and frame the waves of every box of the
  level, then merge and shift parent by parent, the pool splitting at parent boundaries.
  It approximates M2L (CONVENTIONS §3.14; the quadrature of each degree keeps the FMM's
  errors within 1.1× the exact M2L's, `nd_fmm_tables::plane_wave::nodes_for_degree`), so
  it is not `Dense`'s bits; keep it deterministic, the same at every thread count, overlap
  setting and ISA, and on P ranks the one-rank `Fmm`'s bit for bit: every accumulator sums
  its slots in ascending order over geometric masks, so its sources and order do not depend
  on the rank. Its formulas stay in `nd-fmm-tables` and its loops in `nd-fmm-simd`
  (`WaveKernel`, on the P2P choice's ISA); the pair path (`m2l_pair`, `PairOperator`)
  translates each pair directly along the merge's direction (`Tables::m2l_at`), equal to
  the level call to rounding. M2M and L2L are `Rotation`'s; on a device it runs as `Dense`
  (the device path builds the dense tables). `Fmm` sizes its waves buffer from the plan
  (`with_plane_wave_boxes`), so no level call allocates; the cost model counts its M2L per
  box with a V row (`cost::M2lUnit::Boxes`). `Auto` resolves to it on the host at f64 p ≥ 7
  and f32 p ≥ 8 since Phase 6 T11 (below, "The defaults of Phase 6 T11").
- Several charge vectors (Phase 6 T5, C6.3; the `operator` and `fmm` module docs,
  "Several charge vectors"; CONVENTIONS §3.13, "Several charge vectors"):
  `LaplaceOperator::with_vectors(m)` sizes every store for m vectors (m (p + 1)² values per
  box, the vectors one after the other; 3 + m per source, the charges point-major; m or
  4 m per target, the vectors' φ̂ one after the other, then their ĝ), and every operator
  gives each vector exactly the one-vector chain, sharing only what does not depend on
  the charge: the harmonics (`nd_fmm_ref::leaf::*_many`), a pair's distance
  (`P2pKernel::evaluate_many`; the reference P2P vector by vector), a table (the
  translations column by column; `Batched` with m columns per pair). So vector v of
  every output is the one-vector `Fmm`'s on vector v, bit for bit; keep it so. The plan,
  the exchanges and the redistribution carry m through the sizes alone (no `nd-fmm-plan`
  change). `Fmm::evaluate` is refused with `FmmError::VectorCount` when m > 1, agreed in
  the charge-length all-reduce; `evaluate_many` packs the charges point-major on the
  pool. The cost model weighs one vector's work (only the partition depends on it). On
  the device the grouped GEMMs and rotation M2L run on views with m columns per box
  (`device::Columns`), the leaf and P2P kernels take the vectors along z, and the GEMM
  decisions are keyed by the plan's pairs, with a chunk budget per vector unless the builder
  names one (`vector_settings`; the `device` module docs).
- Dipole sources (Phase 6 T8, C6.4; the `operator` and `fmm` module docs, "Dipoles";
  CONVENTIONS §3.7, "Dipole sources", and §3.13, "Source chunks"): `SourceType` is fixed at
  build (`FmmBuilder::sources`, `LaplaceOperator::with_sources`) and sets the source chunk
  (3 + m with charges, 6 with dipoles, 7 with both: u, q, d̂). Only P2M, P2L and P2P change:
  each adds the leaf's charge terms first, then its dipole terms (`leaf::p2m_dipoles`,
  `leaf::p2l_dipoles`, `P2pKernel::evaluate_dipoles` or `p2p::p2p_dipoles` with
  `Reference`), the moments of another leaf mapped by r̂(s|t) into scratch (exact). The
  charge load scales each moment into its leaf's units, `T::from_f64(d.to_f64() / r)`;
  the forward moves 1, 3 or 4 strengths per source (the build's `strengths`), the charges and
  moments packed q, d₀, d₁, d₂ per source. With `SourceType::Charges` (the default) nothing
  changes: `evaluate_sources(Sources::charges(c))` is `evaluate(c)` bit for bit; keep it so.
  `evaluate` and `evaluate_many` on a dipole FMM, and a `Sources` that does not match the
  build, give `FmmError::SourceType`, a short moment slice `FmmError::DipolesLength`, agreed
  in the length all-reduce; dipoles with m > 1 are `SettingsError::SourcesWithVectors`. On a
  device (Phase 6 T8d; the device module docs, "Dipoles") P2M, P2L and P2P run the dipole
  kernels of `nd-fmm-kernels` (`SourceKind`; `host_fallback` keeps any kind on the host, bit
  for bit the host path as before): the device's source store holds the chunk layout with
  the strengths, one upload of w = 3 or 4 values per source per evaluation
  (`DataKind::Charges`, scattered by the strength slots, `DeviceViews::charge_slots`), the
  moments scaled into their leaf's units on the host first (`scale_moments` in `fmm`, the
  bits of the charge load; the sent leaves' chunks from the scaled upload); the P2P layout
  under `DeviceP2pLayout::Auto` is the static rule with dipoles (the tuning cache's P2P
  decisions are the charge kernel's). The cost model weighs the charge kernels. `tests/operator/dipoles.rs` checks the operators against `nd_fmm_ref` (C3.1
  criterion, every P2P kernel); `tests/mpi_exec.rs`, **dipole sources**, the FMM against the
  direct sum, threads, overlap, `Reference`, f32, the one-rank reference, linearity and the
  errors, with the `gpu` feature `device_common::check_dipoles` and `dipole_scenario` (also
  in `tests/device_metal.rs`, `tests/device_cuda.rs` and `tests/device_ranks.rs`: every kind
  on the fallback bit for bit the host, every kind on the device within the FMM bounds, two
  evaluations and blocking against overlapped bit for bit, the transfers of the formula with
  w strengths per source); `tests/operator/device_p2p.rs` and `device_leaf.rs` the dipole
  kernels at the absolute frames (P2P also against the host kernel; P2M, P2L p ≤ 20 in f64);
  the ignored `tests/dipoles.rs` (its own MPI test, any rank count) the gate: twice each source
  type's single-translation prediction on the C3.2 cube, twice the cube's error on the C3.3
  Plummer sphere, f64 p = 3, 8, 12 and f32 p = 3, 8, with a device feature each case also
  with every kind on the device (the CPU runtime f64, Metal f32, CUDA both) within the
  device-against-host bounds of docs/phase4/README.md. The dipole scenario added about 20 s
  to the one-rank debug run of `tests/mpi_exec.rs` (137 → 157 s on the M3 Max).
- The near rule (Phase 6 T12, C6.9; the `fmm` module docs, "The near rule (C6.9)"):
  `FmmBuilder::near_rule` (`NearRuleChoice::CostModel` by default since Phase 6 T11, `Off`
  before; `CostModel`: N_W =
  α_M2P / α_P2P and N_X = α_P2L / α_P2P of the FMM's `CostModel`,
  `CostModel::near_crossovers`; `Crossovers { w, x }` named, negative or NaN refused with
  `SettingsError::NearCrossover`). Step 4 counts every local leaf's points where they are
  (`cost::leaf_point_counts`, shared with the cut), fetches the W rows' ghost leaves' source
  counts (`nd_fmm_plan::near::LeafCounts::exchange`) and builds the plan again with the
  rule (`near_plan`, `Plan::with_near_rule`); the cut weighs that plan (converted pairs as
  P2P), and a recut makes the counts and the rule's plan again. Converted pairs lose their
  truncation error, so the output is not the rule off's; it stays deterministic, the same
  at every thread count and overlap setting, and on P ranks the one-rank `Fmm` of the same
  rule bit for bit. `Fmm::near_rule` reports the setting, the crossovers and the plan's
  `NearPairs`; `BuildTimings::plan` includes the rule's counts and plan. Tested in
  `tests/mpi_exec.rs` (`check_near_rule` on every scenario with W or X lists, **work-weighted
  cut**, **input errors**), `tests/invariance.rs` (the Plummer sphere and the graded tree
  with the rule) and `tests/multi_rank.rs` (the variant "auto, near rule" with the C6.9
  check). A converted pair is exact, but where the truncation errors of pairs cancel the
  error against the direct sum can grow by a few per cent (2.1 % on the work-weighted
  cut's tree at p = 5 with the cost model's rule, 0.6 % at p = 16 with every pair
  converted); the tests bound it at 1.1 times the rule off's (`NEAR_ERROR_BOUND`), C6.9's
  "at most" being a sign-off question (fmm-validate/results/phase6-near.md).
- The shared host drivers (Helmholtz CH2.0; docs/design/helmholtz-fmm-proposal.md §2,
  constraint 1, and §6.2; the `operator` module docs, "The shared drivers"): the level
  calls' walking of the views, threads, scratch sets and batched-M2L blocks live in the
  crate-private `src/driver.rs`, generic over `driver::HostKernel<T>`. Its methods take
  the level call's level. `operator::Kernels` is Laplace's kernel: it ignores the level,
  keeps its runtime p and its per-pair `Tables` dispatch, and delegates to its own
  methods. The plane-wave M2L stays in `operator`. A change to `driver.rs` is a change
  to Laplace's paths: Laplace's outputs stay bit for bit, and its benchmarks within noise
  on the M3 Max, locust and Kathleen (D9 of docs/helmholtz/roadmap.md). Add no runtime
  dispatch below the level call, and keep Laplace's strides and sizes as they are
  (runtime p): never turn a compile-time constant into a runtime value. A box block's
  layout is the kernel's (Helmholtz CH3.5, docs/helmholtz/tasks/CH3.5.md): the
  target-by-target drivers take the block sizes from the plan's stores and pass whole
  blocks, so complex expansions in two planes, multipoles and locals of different sizes
  and real shifts on both planes need no driver change. Only `driver::m2l_block` assumes
  Laplace's layout (m expansions of n values per box, multipole and local alike, square
  tables). Any other kernel returns `None` from `batched_m2l` (and keeps an empty
  `Columns` for `columns`, which is then never called) until CH3.2 adds the
  two-plane case and the epilogue there, under constraint 1.
- The Helmholtz builder (Helmholtz CH3.0b; docs/design/helmholtz-fmm-proposal.md §3.7,
  §6.3; CONVENTIONS §3.15): `helmholtz::Wavenumber` classifies k (Yukawa for Re k = 0,
  real for Im k = 0, complex otherwise; k = 0, Re k < 0, Im k < 0 and non-finite parts
  are `WavenumberError`s), and `helmholtz::min_leaf_level` is the leaf rule, the
  shallowest level whose boxes are at most c_leaf λ wide (λ = 2π/Re k, `C_LEAF` = 1 from
  D11, capped at 16; 0 for Yukawa), for `OctreeOptions::with_min_leaf_level`. It reads k,
  the domain and c_leaf only (constraint 9). `FmmBuilder` does not use it: it stays
  Laplace's builder, and the Helmholtz FMM (CH3.1) passes the level to its tree.
  `tests/helmholtz/` checks it against the census spike's search; the trees of the
  reference workloads are checked in `nd-fmm-validate` (`tests/helmholtz_leaves.rs`).
- Complex densities (Helmholtz CH3.0c; `helmholtz::densities`, re-exported from
  `helmholtz`; CONVENTIONS §3.15, "Storage" and "Inputs and outputs"): `Densities` (k's
  kind, `ChargeKind`, m) fixes the sizes of §3.15's table and the leaf chunks, which the
  module docs define (§3.15 leaves them to CH3.0, not versioned): a complex vector is two
  adjacent real columns, Re then Im, of §3.13's several-vector layout, so every chunk is a
  §3.13 chunk of `columns` source and `output_columns` output vectors, and Yukawa's
  complex vectors are its real path's 2m vectors value for value; keep both so. One
  complex expansion per complex vector. `Charges` in (`check_charges`,
  `write_leaf_charges`), `Output` out in split planes (§3.15's order;
  `write_leaf_output`, Laplace's scales 4π r and 4π r²); `p2m` and `l2p` in the leaf's own
  frame at k̂ = k r, one reference call per point (`LeafScratch`). The operators with a
  second frame (P2L, M2P, P2P) and the FMM are CH3.1's. `tests/helmholtz/densities.rs`
  checks the sizes and indices against closed forms, P2M and L2P bit for bit against
  `nd_fmm_ref::helmholtz::leaf`, and the chunks end to end against the direct sum.
- The Helmholtz operator (Helmholtz CH3.1a; `helmholtz::{degree, operator}`, the module
  docs; docs/helmholtz/tasks/CH3.1a.md): `Degrees` (a tolerance, S7's closed form
  `degree` at Re(k) a, degree 0 on levels 0 and 1, which nothing reads; or degrees named
  per level) and `HelmholtzOperator` for real and complex k, real and complex charges and
  m vectors (Yukawa is `OperatorError::Yukawa`, CH2.1's path). Level l works in its own
  units at k̂_l = k r_l; a box holds r_l φ̃ there as a chunk does, so P2L scales its
  charges by r̂(t|s) and the L2L tables carry ½ (`nd_fmm_tables::helmholtz`). M2M, L2L
  and M2L are `nd_fmm_tables::helmholtz::HelmholtzTables` (RCR, real shifts for real k,
  per-level tables, M2L y-blocks stored within a budget); P2M and L2P `Densities`'; P2L
  and M2P `nd_fmm_ref::helmholtz::leaf` in the points' units with the frame's radius as
  the other box's half-width (the levels of X and W boxes from the frame's radius); P2P
  the reference `nd_fmm_ref::helmholtz::p2p::p2p` until CH3.4. It runs on the unchanged
  shared drivers (`batched_m2l` `None`) and implements `PairOperator`.
  `tests/helmholtz/operator.rs` checks the bodies pair by pair against the direct sum and,
  at low degrees, against `nd-fmm-ref`'s chains in absolute units; the level calls are
  tested with the FMM (CH3.1b).
- The point pipeline (Helmholtz CH3.1b; the crate-private `src/pipeline.rs`, its module
  docs; docs/helmholtz/tasks/CH3.1b.md): steps 3, 4's keys and octree, 5, 6 and 8 of
  `FmmBuilder::build` (`pipeline::{domain, octree}`, `Points::{route, load_coordinates}`)
  and the evaluation's helpers (`agree`, the checks, `LeafRanges`, `OutputOrder`, the
  charge load, `pack_charges`, `gather_output`, `unpack_output_many` and the rest of the
  moves), moved verbatim from `fmm.rs`; `Fmm` holds a `Points`. It is on Laplace's path:
  a change there keeps `Fmm`'s outputs bit for bit (constraint 1, as for `driver.rs`).
  Keep it operator-independent: anything that names Laplace's operator, strategy, cost
  model or device stays in `fmm.rs`.
- The Helmholtz FMM (Helmholtz CH3.1b; `helmholtz::fmm`, its module docs):
  `HelmholtzFmmBuilder` (k, `Degrees`, `ChargeKind`, vectors, gradients, threads, domain,
  points per leaf, `max_level`, the block budget, c_leaf) and `HelmholtzFmm` on any number
  of ranks, the pipeline's steps with the leaf rule (`min_leaf_level` into
  `OctreeOptions::with_min_leaf_level`) and `HelmholtzOperator` on `Evaluator`; no near
  rule, work cut, overlap, device or table cache. `evaluate` packs the c m columns
  (Re q, Im q per complex vector) point-major, so the charge load, `gather_output` and
  `unpack_output_many` run unchanged with m replaced by the columns and output columns
  (CH3.0c's chunks); `HelmholtzFmmError` wraps `FmmError` for the shared steps. Doc-hidden
  for the tests: `pair_calls` (the evaluator runs `PerPair` of the operator) and
  `reference_output` (the pass of `Densities::write_leaf_output`). `tests/helmholtz_fmm.rs`
  (its own MPI test at `Funneled`, in CI's MPI list) checks it against the direct sum, at 1,
  2 and 4 threads (4 only in debug), pair by pair, against the reference pass and the
  one-rank FMM bit for bit, two vectors, real charges against complex ones with Im q = 0,
  and the errors; ε = 10⁻⁶ and kD = 100 in release only (about 21 s at one rank in debug on
  the M6).
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
    C3.3 error per list), `tests/device_metal.rs` (the ignored Metal run of the
    device path, feature `metal`) and `tests/device_cuda.rs` (the ignored CUDA run,
    feature `cuda`; Phase 4S T4) initialise it at `Threading::Funneled`;
    `tests/device_fmm.rs` (the ignored C4.8 gate, feature `gpu`) and
    `tests/device_tune.rs` (the tuner, feature `cpu`) at the default level, and
    `tests/multi_rank.rs` (the ignored C5.1 host gate, Phase 5 T6),
    `tests/device_ranks.rs` (the ignored C5.1 device gate, Phase 5 T8, feature `gpu`) and
    `tests/invariance.rs` (rank-count invariance, Phase 5S T2, below) and
    `tests/helmholtz_fmm.rs` (the Helmholtz FMM, Helmholtz CH3.1b) at `Funneled`.
    The device checks shared by `tests/mpi_exec.rs`, `tests/device_metal.rs` and
    `tests/device_cuda.rs` live in `tests/device_common/`, the tuner's in
    `tests/tune_common/`, the shadow operator of the host-data hook (Phase 5 T7) in
    `tests/shadow/` (used by `tests/mpi_exec.rs`), the per-kind timings' (Phase 4S T5)
    in `tests/kind_common/` (also used by `tests/accuracy.rs`), with the call-window
    measurement of the Metal and
    CUDA executables in `tests/kind_common/windows.rs`, and the whole-output check of the
    C3.2 tree (Phase 4S T9) in `tests/output_common/` (used by `tests/accuracy.rs` and
    `tests/device_fmm.rs`).
  - The batched M2L (Phase 6 T2): `evaluate_threaded` runs `check_batched` wherever the
    scenario resolved `Dense`: the same settings under `Batched` bit for bit, serially,
    once per scenario, precision, p and output kind; once per scenario and precision
    also at its most threads and in the fused mode (within 1e-12 / 1e-5 of `Dense`, the
    same bits at those threads, the one-rank fused reference on several ranks), on one
    build (`Fmm::set_serial`, `Fmm::set_m2l_mul_add`). It added about 15 s to the
    one-rank debug run (91 → 106 s on the M3 Max). `tests/multi_rank.rs` runs every
    workload point with the default strategy, `Batched` and `Batched` fused.
  - The plane-wave M2L (Phase 6 T13): `evaluate_threaded` runs `check_plane_wave` once
    per scenario and precision: under `PlaneWave` within `PLANE_ERROR_BOUND` (1.1) of the
    scenario's error against the direct sum, two evaluations and a build at the most
    threads, overlapped, bit for bit, the kernels on the P2P ISA, M2L counted per box, and
    on several ranks the one-rank `Fmm` bit for bit; **batched against per-pair** runs
    merge and shift against `PerPair` (to rounding, 2.2e-16 on the M3 Max) and threads. In
    debug it checks only the scenarios whose evaluation takes at most `DEBUG_PLANE_BUDGET`
    (0.5 s, the slowest rank's time, agreed; CI's 4-rank run had 496 of its 600 s before
    T13) and names the others, which the release runs check. It added about 15 s to the
    one-rank debug run (about 180 → 195 s on the M3 Max).
    `tests/invariance.rs` runs every workload under `PlaneWave` (f64 p = 8, the graded
    tree also f32 p = 6); `tests/multi_rank.rs` the variant `planewave` with C6.5's check;
    the ignored `tests/accuracy.rs` and `tests/adaptive.rs` its gates at f64 p = 8, 12,
    18 (within 1.1× `Rotation`'s error, threads bit for bit).
  - Several charge vectors (Phase 6 T5): `evaluate_threaded` offers every one-thread
    evaluation to `offer_vectors`, which keeps the scenario's cheapest (the slowest
    rank's time, agreed) per slot (the default P2P kernel; the reference kernel once per
    precision), and after the scenario `check_vectors` runs on it: `evaluate_many` with
    one vector is `evaluate` bit for bit, a wrong vector count is `VectorCount` on every
    rank, and each vector of an `Fmm` of 3 vectors (in release builds also 8, blocking
    and overlapped, and `Batched` with 3 where the strategy is `Dense`) equals its
    single-vector evaluation bit for bit, with the output pass the oracle's
    (`Fmm::reference_output_many`); with the `gpu` feature
    `device_common::check_three_vectors` repeats 3 vectors on the device (each the
    one-vector device `Fmm`'s, blocking and overlapped, the transfers of the formula with
    m, and on the host fallback the host's). The debug run checks 3 vectors overlapped
    only, and leaves a scenario whose cheapest evaluation takes over
    `DEBUG_VECTORS_BUDGET` (0.5 s; the strongly graded tree and multipole routing on the
    M3 Max) to the release runs: it added about 14 s to the one-rank debug run on the M3
    Max (106 → 120 s; the full check 210 s). `tests/multi_rank.rs` runs `auto` and `Batched` on an
    `Fmm` of the eight vectors too, each bit for bit the one-rank single-vector
    references; `tests/device_metal.rs`, `tests/device_cuda.rs` and
    `tests/device_ranks.rs` run `check_three_vectors` in every scenario.
  - New `Fmm` scenarios evaluate through `evaluate_threaded` in `tests/mpi_exec.rs`,
    which repeats them at 2, 4 and 8 threads and checks the output bit for bit, or
    through `evaluate_every_kernel`, which also repeats them with `Reference` and every
    available ISA and compares each ISA with `Reference`. It roughly triples the cost
    of a scenario, so it runs on the small ones only (debug run under a minute).
  - Every scenario runs on every rank count (Phase 5 T6). Each rank passes its share of
    the points (`share`, every P-th), or a distribution of the README (`distribute`);
    `Fmm` moves them. A test that needs a rank count says so on every rank and returns
    before any collective (as `input distributions` does on one rank).
  - The one-rank reference (docs/phase5/README.md, "The oracles"; design §5): on several
    ranks `evaluate_threaded` checks every one-thread evaluation bit for bit against the
    `Fmm` of the same settings at one thread over the union of every rank's points and
    charges in rank order, which every rank builds on `SimpleCommunicator::self_comm()`
    (`check_one_rank`; the union gathered by all-gathers). Two input distributions agree
    within 100 u_T relative L2 (1.1e-14 f64, 6.0e-6 f32), not bit for bit: the order
    within a leaf is (origin rank, origin position). Not run on one rank, where a run is
    its own reference, so the one-rank debug run costs nothing more.
  - Rank-count invariance (Phase 5S T2, C5S.1; requirement 3 of docs/phase5s/README.md;
    docs/design/scale-out.md §5.1): `tests/invariance.rs`, its own executable (so that
    `mpi_exec` keeps its minute), checks that the `Fmm` on P ranks equals the `Fmm` on P/k
    ranks bit for bit for every divisor k > 1 of P (k = 1 on one rank), when small rank r
    holds the inputs of world ranks k r … k r + k − 1 concatenated: every input generated
    by index (a seeded random share), the small runs on a `split` of the world (the other
    ranks enter nothing of it), the outputs moved back point to point by index, one
    all-reduce of the count. The cube, the Plummer sphere and a graded tree with one empty
    rank, N = 10⁴, f64 p = 8 `Rotation` and f32 p = 3 `Dense` (the strategy named: `Auto`
    resolves by the ranks per node), each comparison blocking against overlapped; since
    Phase 6 T12 the Plummer sphere and the graded tree again with the near rule of the
    cost model. 39 s at one rank in debug on the M3 Max (69 s with T12's runs); 31, 38 and
    43 s at 2, 4 and 8; in CI (PR #101) 64 s at
    one rank (`run-tests`), 56 and 110 s at 2 and 4 (`run-tests-mpi`). Keep it passing in
    every structural task (T3–T8): it is the oracle where no one-rank run fits, and its
    at-scale form is `nd_fmm_validate::scaling::invariance`.
  - CI runs `tests/mpi_exec.rs`, `tests/mpi_threading.rs`, `tests/invariance.rs` and
    `tests/helmholtz_fmm.rs` in debug at 2 and 4 ranks on every pull request (the `run-tests-mpi` job, root
    `CLAUDE.md`, "Checks";
    `mpi_threading` checks there that its `MpiThreading` error, carried by step 1's
    agreement all-reduce, reaches every rank). 8 ranks and the ignored tests
    (`accuracy`, `adaptive`, `multi_rank`, the device executables) stay by hand, on the
    M3 Max and on locust, under an external timeout; `multi_rank` at 1, 2, 4 and 8 ranks
    in release (its module docs give the command).
  - Operator and geometry tests do not initialise MPI.
  - Every error that depends on one rank's input is agreed by all ranks before the
    next collective.
  - Doctests that initialise MPI are `no_run`.
  - Since Phase 5 T6 `Fmm` runs on any number of ranks: `build` moves every point to the
    rank that owns its leaf (two `Redistribution`s, the coordinates forwarded), and
    `evaluate` forwards the charges and moves the output back (one all-to-all-v each
    until Phase 5S T8, since then point to point on the routes fixed at build, rlst's
    `SparseRoutes`;
    `StageTimings::{forward_charges, backward_output}`, `BuildTimings::redistribute`).
    `PointsNotOwned` is gone; `FmmError::Redistribution` carries a redistribution that
    overflows MPI's counts. On P ranks the host output is bit for bit the one-rank `Fmm`
    over the union of the points in rank order; keep it so (the `fmm` module docs,
    "Several ranks").
  - The work-weighted cut (Phase 5S T7, C5S.4, S3; docs/design/scale-out.md §6; the
    `cost` and `fmm` module docs): on several ranks `build` cuts the coarse blocks a
    second time by modelled cost, `FmmBuilder::work_cut` (on by default, decision 3 of
    docs/phase5s/README.md; off: by points only, as before). Step 4 counts the points of
    every local leaf where they are (one all-to-all-v of distinct keys with their source
    and target counts, `cost::leaf_point_counts`), learns the ghost leaves' source counts
    from the build of a `SourceExchange<u8>`, counts every own block's work by kind on the
    first plan (`cost::work_counts`: P2M and L2P points, M2M and L2L pairs, V pairs, X
    pairs × source points, W pairs × target points, near interactions n_t Σ n_s; the rows
    of `Global` boxes belong to no block), weighs it with a `CostModel` (`cost_model`, by
    default `CostModel::calibrated` for the strategy the host rule resolves with the ranks
    per node agreed in step 2's all-reduce, the precision and p: Kathleen's constants,
    scaled by powers of p + 1 away from the calibrated p), calls
    `nd_octree::Octree::repartition` with `min_gain` `WORK_CUT_MIN_GAIN` (2 %: the new cut
    only where it lowers the busiest rank's block cost by that much; signed off on
    2026-10-13, `fmm-validate/results/phase5s-t7.md` §9) and builds the plan again where
    the cut changed, before any point moves (`BuildTimings::cut`, `Fmm::recut`). `nd-octree` receives weights only. The constants are fixed
    inputs, never timed in a build: the partition is deterministic, and only it depends
    on them, never a bit of the output. `Fmm::{work_cut, recut, cost_model, kind_counts}` report
    it. `tests/mpi_exec.rs` runs every scenario with it on, and **work-weighted cut**
    checks it off against on bit for bit, that it moves points, and the counts summed over
    the ranks against one rank; `tests/invariance.rs` and `tests/multi_rank.rs` run with
    it on (the default). The input distribution "owners" of **input distributions** takes
    the owners from an `Fmm`'s own octree (both cuts).
  - The host-data hook (Phase 5 T7; `nd_fmm_plan::operator::HostData`, design §6): the
    evaluator tells the operator of `reset` and of every exchange's reads and writes.
    `LaplaceOperator` keeps the default (nothing), and `ExecOperator` forwards the events
    to its engine, so the host path is unchanged bit for bit; the device operator mirrors
    them (from T8, below). `tests/mpi_exec.rs` checks the coverage with a shadow operator
    (`tests/shadow/`) that keeps its own stores and learns of host data only through the
    events: bit for bit the `Fmm` on every scenario (once per tree, precision and output
    kind) on every rank count, and different with the hook disabled on several ranks
    (**host-data hook**), and (Phase 5S T6) different with only the owned top tree's
    events disabled (`shadow::Hook::WithoutTopTree`: `SendReduction`,
    `ReceivedReduction`, `ReceivedTop`) on several ranks. Keep it so when the evaluator,
    the exchanges or `Fmm`'s stages change; the device operator uses the events since T8.
  - Overlap (Phase 5 T9, C5.2; docs/design/distributed-fmm.md §8; the `fmm` module docs,
    "Overlap (C5.2)"): `FmmBuilder::overlap` (off by default until T9's sign-off decides
    the default) and `Fmm::set_overlap` run `Evaluator::exchange_sources_and_upward_local`
    and `far_field` in place of stages 1–5: the same level calls and events, bit for bit
    the blocking output. `StageTimings` keeps its fields, split so that the two paths
    compare (the work parts from `OverlapTimes`, the rest of each overlapped stage in
    `exchange_sources` and `exchange_multipoles`), and adds `overlap: Option<OverlapTimes>`
    (per exchange post to completion, exposed wait, `test` calls; the wait per level; the
    all-gather of the replicated top, the coarse gather before Phase 5S T5).
    `StageTimings::{reduction, replicated_top}` (Phase 5S T6) are parts of `upward_global`
    in both paths, timed in `ExecOperator::host_data` from the events that bracket them
    (`EventClock`: each reduction level from `SendReduction` to the end of
    `ReceivedReduction`; the all-gather of R, from the end of `SendMultipoles` blocking
    or `OverlapTimes::top_gather` overlapped, plus `ReceivedTop`).
    `Fmm::exchange_traffic` gives the messages and values per exchange (since Phase 5S T5
    also the upward reduction per level, the replicated top's values and threshold, and the
    multipole exchange's neighbours, destinations per level and largest fan-out of a box;
    since T8s its depth and forwarded chunks).
  - The forwarding tree (Phase 5S T8s, S8; fmm-validate/results/phase5s-t8s.md §1;
    `nd_fmm_plan::exchange::MultipoleRouting`, re-exported by `fmm`):
    `FmmBuilder::multipole_routing` (default `MultipoleRouting::Tree` with `TREE_FAN_OUT`
    = 8; `Direct` is the oracle) and `Fmm::multipole_routing`. `build` takes each rank's
    node from step 1's `split_shared` (`Group::translate_rank`, no communication) and
    passes it to `Evaluator::with_routing`. The output is the same bit for bit under every
    routing, and no event changes: the forwards resend received chunks, so the device and
    the shadow see the direct exchange's `ReceivedMultipoles` lists (fewer duplicates in
    `send_boxes`). The shadow check builds its evaluator with the `Fmm`'s routing.
    `tests/mpi_exec.rs`, **multipole routing**, checks the narrow tree (fan-out 2), the
    default and `Direct` against each other and one rank, blocking and overlapped, with
    the shadow check on the narrow tree.
    The device runs the overlapped stages too, for correctness (design §13, question 10;
    the stage windows `UpwardLocal` and `Downward`). `tests/mpi_exec.rs` checks overlap on
    every scenario that evaluates (its module docs, "Overlap"), the shadow check runs
    overlapped, and the device's fallback check runs with `overlap(true)`.
    `examples/overlap_timing.rs` (not registered; release, by hand) measures the C5.2
    criterion: blocking and overlapped evaluations of one build in turn, the medians over
    evaluations of the maxima over ranks, the traffic per rank and the copy bandwidth.
  - The defaults of Phase 6 T11 (decision 7 of docs/phase6/README.md, signed off on
    2026-10-20; fmm-validate/results/phase6.md): `Auto` resolves on the host by p and the
    precision alone, `Batched` below `M2lStrategy::auto_plane_wave_min_p` (f64 7, f32 8) and
    `PlaneWave` from it (`StrategyChoice`'s reason `Auto → PlaneWave: f64 p = 8 ≥ 7`; the
    ranks per node and the node budget agreed and reported, no longer read); `Batched`'s mode
    `fmm::default_m2l_mul_add(p)` (fused from `FUSED_MUL_ADD_MIN_P` = 6) unless named; the
    near rule `NearRuleChoice::CostModel`; `Isa::detect` AVX-512 where the CPU has it; the
    leaf size `fmm::default_max_points_per_leaf(p, backend)` unless named
    (`Fmm::max_points_per_leaf`); `CostModel::calibrated` with T11's constants (`Batched`
    its own, `Auto` that of its host rule's strategy). The device path keeps its rule.
    With the near rule on, each strategy's cost model gives its own crossovers: a test that
    compares two strategies, or the device with the host, at equal near pairs names them
    (`near_as` in `tests/mpi_exec.rs`, `device_common::same_near_pairs`) or the rule off,
    and where `Auto` resolves differently on the device (`Dense`, `Rotation`) and the host
    (`Batched`, `PlaneWave`), the host reference names the device's strategy
    (`device_common::check_default`, `tests/device_fmm.rs`). `check_batched`,
    `check_plane_wave` and `check_near_rule` of `tests/mpi_exec.rs` build their own exact,
    `Dense` or rule-off references where the scenario now resolves `Batched`, `PlaneWave` or
    the rule (reusing the scenario's `Fmm` where they can: about 9 % more on the 4-rank debug
    run, 224 → 244 s on the M3 Max; `run-tests-mpi`'s timeout rose to 900 s).
  - The strategy rule on the node (Phase 5N T4, C5N.3; docs/design/node-m2l.md §4, §5;
    until Phase 6 T11, which replaced its rule:
    `tables` and `fmm` module docs): under `Auto` the host resolves the strategy by
    `M2lStrategy::resolve_on_node` (rule B: Dense iff the dense tables of a rank, 332
    (p + 1)⁴ values, are at most `AUTO_DENSE_RANK_BYTES` = 9 MB and ranks per node × those
    at most the node budget, `FmmBuilder::dense_node_budget`, default
    `AUTO_DENSE_NODE_BYTES` = 64 MB; Rotation otherwise), with a `NodeContext` of p, the
    value's bytes, the ranks per node and the budget. `build` splits the communicator by
    node once on every backend (step 1; the device path reuses it for `RankPlacement`) and
    agrees the largest ranks per node in step 7's all-reduce (two maxima), so every rank
    resolves the same strategy; `ThreadingReport::{ranks_per_node, m2l}` report it with
    the reason (`StrategyChoice`). The device path keeps the `tune` rule. `resolve(p)` and
    `AUTO_DENSE_MAX_P` are deprecated. The guarantee is per *resolved* strategy: a
    one-rank reference names `Fmm::strategy()` of the multi-rank run (`check_one_rank` in
    `tests/mpi_exec.rs`, `tests/multi_rank.rs`). With `table_cache` the lowest rank of
    each node loads or builds and stores first, the others after a barrier of the node's
    communicator, then load (decision 7; `TableCache` unchanged). `tests/mpi_exec.rs`,
    **the strategy rule on the node**, checks the agreement, a named strategy kept, and
    the cold cache filled once per node; `tests/operator/strategy.rs` the rule over both
    precisions, p = 0–20, 1–128 ranks per node and the budgets.
  - `examples/basic_evaluation.rs` is the user-facing example of calling `Fmm`
    (registered with `templated-examples`, so the weekly job runs it at 3 ranks): it
    must run on any number of ranks, each passing its own points, and keep to the
    public API of `fmm`.
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
    points once per build, charges and output once per evaluation; the device zeroes its
    stores at the evaluator's `HostData::Reset`, and `Fmm::evaluate` drives
    `begin_evaluation` (the charges, after `reset`) and `read_output`;
  - several ranks (Phase 5 T8, C5.1 device, C5.2's device-resident ghost buffers;
    docs/design/distributed-fmm.md §7; the `device` module docs, "Several ranks"): a
    device backend runs on any number of ranks; `DeviceNeedsOneRank` is gone. The device
    stores have the plan's layouts, ghost slots included, so `build` builds the device
    operator after the evaluator, from its source exchange's counts (every leaf of the
    numbering) and `device::ExchangeLists` (copied from the evaluator's exchanges through
    `Evaluator::{source_exchange, reduction_exchange, multipole_exchange, top_exchange}`),
    swaps it in for the host operator, loads the points and agrees the outcome (one
    all-reduce). At each event the device moves exactly what the exchange sends and
    receives, packed on the device in buffers allocated at build: `SendReduction(l)` (Phase
    5S T6) one `gather_columns` of the level's sent boxes and one download (one sync per
    reduction level at which the rank sends: the owned boxes of the level are formed on
    the device just before), `ReceivedReduction(l)` one upload and one `scatter_columns`
    before the global M2M that reads them, `SendMultipoles` one `gather_columns` and one
    download (one more sync), `ReceivedTop` (Phase 5S T5; `ReceivedCoarse`
    before) and each
    `ReceivedMultipoles(l)` one upload and one `nd_fmm_kernels::movement::scatter_columns`
    (assignment, so a −0.0 stays −0.0), `ReceivedSources` one upload of the ghost tail;
    `SendSources` nothing (`Fmm` writes the sent leaves' charges into the host chunks).
    The counters split them by `DataKind` (`GhostSources`, `SentReduction`,
    `ReceivedReduction`, `SentMultipoles`, `TopMultipoles`, `ReceivedMultipoles`; per
    level in `DeviceCounters::received_levels` and `reduction_levels`);
    `Fmm::exchange_sizes` gives the sizes (`sent_reduction`, `received_reduction` per
    level since Phase 5S T6), and `tests/device_common::expected_evaluation` the formula,
    checked on every scenario and rank count: 1 + [anything sent] + r_s syncs per
    evaluation, at most 2 + r_s, r_s the reduction levels at which the rank sends
    (docs/design/scale-out.md §4.5; Phase 5's two before T5). One rank is unchanged: every
    list empty, Phase 4's formula. (Between Phase 5S T5 and T6 the device did not mirror
    the reduction and the device scenarios on several ranks were not run.) A device per
    rank: one `split_shared` per build (`DeviceReport::ranks`, `device::RankPlacement`);
    every rank opens the default device (`Device::open` takes no index); on the CPU
    runtime the units per cube are `threads(n)` capped at the node's cores over its
    ranks. Errors (decision 12): one all-reduce after the output's download agrees a
    device error of any rank (`Device` on the rank, `OtherRank` elsewhere), and the kept
    error rides on the charge-length agreement of later evaluations;
    `Fmm::inject_device_error` (doc-hidden) is the test hook. Tuning (design §7.6): with a
    tuning cache on several ranks rank 0 alone tunes and uses the cache and broadcasts its
    M2L strategy; the others take it and the static rule for the rest
    (`RankPlacement::follows_rank0`). Ranks sharing one GPU are correctness only, never
    timed as scaling (docs/phase5/README.md, decision 3);
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
    tables under every strategy, three launches per chunk (two with the fused gather,
    Phase 6 T7), plans and scratch built at build, the GEMM by `FmmBuilder::device_gemm` and the chunks by
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
    `StageTimings::device` (spans that overlap on Metal, not a breakdown; on CUDA, one
    stream, they do not overlap: "CUDA (Phase 4S)" below); no window on
    the CPU runtime, whose windows drain the stream; `synchronous_stages` disables them. Never add a sync, a download or a host call to a
    default evaluation; a new launch belongs in the formula of `tests/device_common`;
  - per-kind timings (Phase 4S T5, C4S.5; `fmm::KindTiming`, `FmmBuilder::kind_timings`,
    `Fmm::set_kind_timings`, `StageTimings::kinds`, device-path.md §8.3): off by default,
    and with `Off` each level call only checks the mode. The one place that times is the
    `FmmOperator` implementation of `fmm::ExecOperator` (its `delegate!` macro), around
    the engine's method: `LaplaceOperator`, the device operator and the kernels do not
    change. Only calls with an entry in their view are timed and counted (at most 148 per
    evaluation; 3–119 on the `tests/mpi_exec.rs` trees). `Synchronous`: the host clock per
    call, and on a device one sync after the charge upload and one after every such call;
    `Device`: one window per call of a device kind (`DeviceOperator::{open_call_window,
    close_call_window}`), no sync, resolved after the download, refused at build
    (`SettingsError::KindTimingUnsupported`, step 1's agreement) where the device does not
    time on itself (the host, the CPU runtime). Both compose with `synchronous_stages`
    and `device_timestamps`. Every mode gives the bits of `Off`: `tests/mpi_exec.rs`
    checks `Synchronous` on every scenario at 1 and 4 threads and on the CPU runtime,
    `tests/device_common` both modes on Metal and CUDA with the syncs and windows of
    each, `tests/accuracy.rs` reports the sum over the kinds against the operator stages
    on the C3.2 tree at p = 6 (0.9999 at one thread on the M3 Max). On Metal the `Device`
    windows overlap and misattribute time between calls, so they are not a breakdown
    there (M2L 0.25 ms against 6.15 ms with `Synchronous`, L2P 4.67 against 0.37 ms; the
    cube at N = 10⁵, p = 8, f32; `tests/kind_common/windows.rs`); use `Synchronous`.
    On CUDA they do not overlap and are a breakdown ("CUDA (Phase 4S)" below).
    Never time a level call inside `nd-fmm-plan` or `nd-fmm-kernels`, and never assert a
    timing (the overlapped stages of `nd-fmm-plan` read the clock around their `test`
    calls, waits and work parts for `OverlapTimes`, Phase 5 T9: the one exception);
  - the grouped translations' Phase 6 T7 variants (C6.1; the `device` and `tune` module
    docs): `FmmBuilder::device_gather` (`DeviceGather::{Auto, Separate, Fused}`),
    `device_reduction` (`DeviceReduction::{ByChunk, Walk}`, by chunk by default) and
    `DeviceGemm::Shared` (the shared-memory GEMM, `GemmLayout::shared_for`); none changes a
    bit. A plan with the fused gather needs no gathered columns and has chunks twice as
    wide; the scratch holds X and Y for the widest plans and candidates of each
    (`ScratchColumns`), X only where one gathers. Under `DeviceGemm::Auto` the static rule (`tune::static_layout`,
    `tune::static_gather`) takes the shared-memory layout to p = 9 and the fused gather on
    Metal, the default layout and the separate gather elsewhere (on CUDA the best variant
    alternates with p, so the rule waits for T11), the several-vector chunk budget is
    capped at 8 vectors on Metal and at one vector's on CUDA, and the tuner offers the
    shared-memory layout and the other gather (`CANDIDATE_SET_VERSION` 3; the cache's
    `FORMAT_VERSION` 2 names the gather and the shared-memory layout); a named gather holds
    for every call and candidate. `TranslationReport::{gather, reduction, launches}` report
    them, and `tests/device_common::check_variants` builds, in every device scenario (the
    CPU runtime in `tests/mpi_exec.rs`, Metal, CUDA), the hand-written reference and four
    variants (fused; walking; shared; shared, fused and walking), each bit for bit the
    reference with the launches of the formula (the device line counts the evaluations);
  - autotune (T12, C4.7; module `tune`, device-path.md §10): `FmmBuilder::tuning_cache`
    (no default directory, no environment variable) and `tuning_budget` (10 s); the M2L
    strategy under `Auto` with M2L on the device is decided before the tables are built
    (`device::tune_strategy`: cache, timing rotation against dense on the largest V level,
    or the static rule; at p ≥ 12 without `table_cache` the static rule, not stored), the
    GEMM of each (kind, pair bucket) under `DeviceGemm::Auto` at the end of
    `DeviceOperator::new`, the P2P layout under `DeviceP2pLayout::Auto` in `load_points`;
    settings the builder names are never tuned. Static rule (`tune::static_*`, keyed by
    backend since Phase 4S T7): f32 `Dense`; f64 `Dense` at every p on CUDA (decision 9,
    signed off on 2026-10-07), on Metal and the CPU runtime `Dense` to p = 11 and
    `Rotation` from 12 (provisional); the GEMMs of
    `DeviceGemm::Auto`, `P2pLayout::default_for`; with M2L on the host fallback `Auto`
    keeps the host rule (requirement 8). Candidates are registered only past the device's
    checks and the input-precision guard (a library below T's precision never), timed
    after a warm-up launch as the median of 5 batches of ≥ 10 ms, the static rule's
    first, no candidate started past the deadline; level calls under 512 pairs keep the
    static rule. The cache file (one per backend, device, precision, p) is text with a
    checksum, the key and every version, written atomically; a stale or corrupt file is
    rejected and re-tuned. Every choice is fixed at build and reported in
    `DeviceReport::tuning`; never tune inside `evaluate`, never call CubeCL's autotune.
    The coefficient-major layout is measured and not a candidate (`tune` docs); a change
    to any candidate set bumps `tune::CANDIDATE_SET_VERSION`;
  - safety (requirement 9): no `unsafe` here and no direct `cubecl` dependency;
    CubeCL only through `nd-fmm-kernels`;
  - threads: with `Backend::Cpu` no rayon pool, `threads(n)` caps the CPU runtime's
    units per cube; with Metal and CUDA the pool serves host-fallback kinds, the charge
    load and the host output pass only (Phase 4S T9);
  - the output pass and the charge load (Phase 4S T9, C4S.8; `fmm` module docs, "The
    output pass", device-path.md §18.4): `build` stores per target (caller's order) its
    leaf-order point and leaf (`u32`, 8 N_t bytes) and per leaf 4π r and 4π r²; the host
    pass is one loop over the targets on the `Fmm`'s pool (serially without one), collected
    into the output buffers without a zero fill (`collect_into_vec`, `unzip_into_vecs`),
    each value `T::from_f64(x.to_f64() / scale)`; the charge load splits the source chunks
    by leaves and the device's upload buffer by ranges on the pool, and skips the host's
    source store when P2M, P2L and P2P run on the device. `OutputPass` (decision 12,
    `FmmBuilder::output_pass`): `Auto` runs the pass on the device where it does f64
    arithmetic (`nd_fmm_kernels::movement::gather_output` on an `OutputOrder` uploaded
    once, f64 division, one rounding; the CPU runtime and CUDA), the host pass otherwise
    (Metal); `Host` on every backend; `Device` refused without f64
    (`SettingsError::OutputPassUnsupported`, step 1's agreement). The device pass is one
    launch more and the same download (o N_t values) and sync; `DeviceReport::output_pass`
    and `Fmm::output_pass` say where it ran. `Fmm::evaluate_into` (decision 11) writes into
    the caller's `Output` (wrong lengths and gradients resized, the output untouched on an
    error); `evaluate` wraps it. `StageTimings::download` is the part of `output` in
    `read_output`. Every pass is bit for bit the pass before T9, kept as the doc-hidden test
    oracle `Fmm::reference_output` (`tests/mpi_exec.rs` at 1 and 4 threads and on the CPU
    runtime with both passes, `tests/device_common` on every device scenario, the unit tests
    of `fmm`, and the C3.2 tree at p = 6 in the ignored gates). Never change the scales'
    arithmetic, the division or the rounding;
  - the output's path from the device to the `Output` (Phase 4S T11, C4S.10, decision 14;
    device-path.md §4.1, §18.5): `read_output` downloads with
    `nd_fmm_kernels::Device::download_view` and returns `DeviceOutput` carrying
    `OutputValues` (CubeCL's host copy, pinned memory on CUDA, behind a box, since
    `DeviceDriver` knows only `T: SimdScalar`), borrowing the operator: `CallerOrder(values)`
    after the device pass, which `copy_output` copies into the `Output` on the pool;
    `LeafOrder { values, offsets }` for the host pass, which `gather_output` reads as a
    slice in the `LeafStore` layout with the target leaves' point offsets. The operator
    keeps no host copy of the target output: its host-fallback mirror exists only with
    L2P, M2P or P2P on the host (`Mirrors::target_output`), and `DeviceReport::host_mirror_bytes`
    says what the mirrors hold (0 with every kind on the device; checked in
    `tests/device_common`). The charges: `gather_charges` collects them in leaf order into
    a fresh `Vec` on the pool (no zero fill), which `begin_evaluation` hands to
    `Device::write_owned` (no copy). Still one upload, one download and one sync per
    evaluation, and the output bit for bit as before;
  - every device test prints the backends it ran, and on several ranks the ranks; Metal
    tests are ignored and run by hand outside the macOS sandbox (`tests/device_metal.rs`); CUDA tests are ignored,
    type-checked in CI and run by hand on locust (`tests/device_cuda.rs`, the CUDA blocks
    of `tests/device_fmm.rs`, `tests/accuracy.rs` and `tests/adaptive.rs`, and
    `tests/operator/device_*.rs`; "CUDA (Phase 4S)" below).
- Before finishing: `cargo clippy -p nd-fmm-exec --all-targets -- -D warnings` and
  `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` must pass. Tasks that add
  ignored tests must also pass
  `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored`. A change to
  `Fmm`, the moves or the evaluator also runs `tests/mpi_exec.rs` at 2, 4 and 8 ranks and
  the ignored `tests/multi_rank.rs` at 2, 4 and 8 ranks in release, by hand under an
  external timeout (root `CLAUDE.md`, "MPI"). A change to
  the device path also needs:
  - `cargo clippy -p nd-fmm-exec --all-targets --features cpu,metal -- -D warnings`;
  - `cargo check -p nd-fmm-exec --features cuda` (type-checked here; run on locust);
  - `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release` (every
    `tests/mpi_exec.rs` scenario repeated on the CPU runtime), and from Phase 5 T8 by hand
    at 2 and 4 ranks under an external timeout (every scenario's device repeat runs on
    every rank count, with **device errors and tuning on several ranks**), and the
    ignored `tests/device_ranks.rs` at 2 and 4 ranks (its module docs give the commands;
    Metal at 2 ranks outside the sandbox, CUDA at 2 and 4 on locust);
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
    at p = 8), against the host of the same settings, and the C3.3 gate on the device;
    from T12 also the tuning budget at the C3.2 size, the cube at p = 8 tuned with a fresh
    cache);
  - from T12, `tests/device_tune.rs`: the tuning cache without MPI (round trip, stale and
    corrupt files, concurrent writers) and, with `--features cpu`, the scenarios of
    `tests/tune_common` on the CPU runtime (f64 p = 6 in full, f32 p = 3; about a minute
    in release), which `tests/device_metal.rs` also runs on Metal (f32 p = 8 and 3) and
    `tests/device_cuda.rs` on CUDA (f32 p = 8 and f64 p = 6 in full, f32 p = 3);
  - from Phase 4S T9, the whole output on the C3.2 tree in the ignored gates
    (`tests/output_common`): the host in `tests/accuracy.rs`, the device backends in
    `tests/device_fmm.rs`, each with the output hash printed;
  - from Phase 4S T4, by hand on locust (root CLAUDE.md, "Checks"; after
    `tools/gh200/sync.sh`, each as `tools/gh200/remote.sh '<command>'`):
    `cargo clippy -p nd-fmm-exec --all-targets --features cpu,cuda -- -D warnings`,
    `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release`, and the
    same with `-- --ignored` (every CUDA test: `tests/device_cuda.rs`, the CUDA blocks of
    the gates and the operator tests).

## CUDA (Phase 4S)
Measured on locust's H100 (GH200) on 2026-10-07 (Phase 4S T4, C4S.4; CubeCL
0.11.0-pre.4 through LLVM NVPTX, CUDA 12.6; docs/design/device-path.md §18.1). The pin is
CubeCL 0.11.0 since Phase 5S T1s, re-run on locust there (fmm-validate/results/phase5s-t1s.md).
- Running: on the M3 Max, outside the sandbox, `tools/gh200/sync.sh`, then
  `tools/gh200/remote.sh 'RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features
  cuda --release -- --ignored --nocapture'`. CUDA tests are `#[ignore = "CUDA: run by
  hand on locust"]`: `tests/device_cuda.rs` (its own MPI test, at `Funneled`), the CUDA
  blocks of `tests/device_fmm.rs`, `tests/accuracy.rs` and `tests/adaptive.rs` (behind
  `Backend::Cuda.is_compiled()`), and `tests/operator/device_*.rs` (`…_on_cuda`). The
  tuner scenarios and the strategy at f64 p = 12 run in `tests/device_cuda.rs`, not in
  `tests/device_tune.rs`, whose one MPI test is the CPU runtime's.
- Both precisions run on CUDA, every check with the f64 bounds in f64 and the f32 bounds
  in f32. Every kind on the host fallback is bit for bit the host path, with the
  transfers of the formula; the default placement (every kind on the device) stays within
  the FMM bounds, two evaluations and two builds bit for bit.
- The library GEMM never runs on CUDA: its probe fails on the LLVM path (F28), so every
  level call the static rule gives the library (f32, n ≥ 81: M2M and L2L under `Auto`,
  every kind under `Library`) reports a `library_rejection` and runs the hand-written
  kernel; the tests assert it. The tuner's candidates are the hand-written layouts, the
  chunk budget and the P2P layouts; a library candidate is unregistered. (Since CubeCL
  0.11.0, Phase 5S T1s, the probe succeeds and runs f32 at TF32 precision; the
  input-precision guard rejects it because CUDA registers no f32 CMMA form,
  `docs/design/cubecl-upstream.md` entry 4, so the outcome is unchanged.)
- C4.8 (`tests/device_fmm.rs`): the device output within 6.7e-15 (f64) and 2.9e-6 (f32)
  of the host's (relative L2, φ), the errors against the direct sum within a ratio of
  1.0000 of the host's in f64 and 0.991–1.010 in f32, over the cube, the Plummer sphere
  and the clusters at N = 10⁵ and the cube at N = 10⁶; every evaluation moves the
  charges up and the output down with one sync. The tuning budget at the C3.2 size: 3.2 s
  (f32) and 3.4 s (f64) of the 10 s.
- Stage windows (`device_timestamps(true)`, `tests/device_cuda.rs`, the cube at N = 10⁵,
  p = 8): five windows and one sync per evaluation, the output bit for bit the default's.
  On CUDA the windows do not overlap (one stream, events in order): the five spans add up
  to 0.71–0.77 of the wall time of `evaluate` in f32 and f64 (on Metal, T11, they added
  up to more than it). `KindTiming::Device` (T5) was not checked: T5 had not merged.
- Kind timings (Phase 4S T5, run on 2026-10-07; GPU idle before and after, clocks not
  locked): `Synchronous` and `Device` (alone and inside the stage windows) bit for bit
  the default in every `tests/device_cuda.rs` scenario, with the syncs and windows of
  each (44 and 88 evaluations, 8–40 calls). On the cube at N = 10⁵, p = 8
  (`tests/kind_common/windows.rs`) the call windows do not overlap: every stage's call
  windows add up to less than its own window in all 10 evaluations (f32 downward
  1.64 ms of calls in a 1.65 ms span), and their sum is 0.68–0.71 (f32) and 0.72–0.74
  (f64) of the wall time of `evaluate`. Medians per kind, ms, `Device` / `Synchronous`,
  f32: P2M 0.036 / 0.039, M2M 0.145 / 0.219, M2L 1.475 / 1.556, L2L 0.152 / 0.374, L2P
  0.078 / 0.176, P2P 0.307 / 0.328; f64: P2M 0.063 / 0.068, M2M 0.135 / 0.226, M2L
  2.244 / 2.342, L2L 0.149 / 0.302, L2P 0.135 / 0.181, P2P 0.480 / 0.666. The ignored
  run took 989 s (`device_fmm.rs` 659 s, `device_cuda.rs` 104 s).
- Defaults and tuner candidates (Phase 4S T7, C4S.7; measured on 2026-10-07 with
  `nd-fmm-validate`'s `layout_sweep`, the uniform cube at N = 10⁶; device-path.md §18.2):
  - the CUDA defaults come from `nd-fmm-kernels` by backend: P2P `Cube` of 32 units, the
    leaf operators `Cube` of 32 units with tiles of 32, the GEMM's CUDA rule
    (`GemmLayout::default_for`), the rotation layout as on Metal, and the chunk budget
    `default_scratch_bytes` (2 GB on CUDA; `FmmBuilder::device_scratch_budget` names
    another), which `tune::static_gemm` and the scratch sizing take. Metal and the CPU
    runtime keep theirs;
  - CUDA GEMM candidates (`tune::gemm_candidates`, `cuda_gemm_layouts`): the CUDA default,
    the Metal default, one plane of 32 units with 2 columns each (small level calls) and
    the default's rows with 128 units and 4 columns, all at the device's budget; no
    16 MB candidate on CUDA (a smaller budget only adds chunks). CUDA P2P candidates:
    cube 32 (the default) and 64, planes 2 and 4 (128 units measured slower everywhere).
    The library stays unregistered (F28). `CANDIDATE_SET_VERSION` is 2;
  - the static M2L rule on CUDA (decision 9, signed off on 2026-10-07): `Dense` at every p
    in f32 and f64; measured dense faster than rotation at every p (f64 p = 4 to 20, f32
    p = 2 to 10) per level and end to end, rotation/dense 1.05–1.90 in f64 and 1.15–1.56
    in f32 (`nd-fmm-bench`, N = 10⁵ and 10⁶). `tests/device_cuda.rs` asserts `Dense` at
    f64 p = 12 without a table cache;
  - the tuning hook may offer a GEMM candidate with wider chunks than the operator's
    scratch: the strategy decision times it on a scratch of its own, a GEMM decision skips
    it (`Timing::Skipped`, "scratch"), and a budget the builder names holds for every
    level call (`tests/tune_common`, "a wider candidate").
- Test budget (release, H100, one process per test file): `tests/device_cuda.rs` 95 s,
  `tests/device_fmm.rs` 656 s (its slowest point the host at f64 p = 18), `accuracy.rs`
  91 s, `adaptive.rs` 71 s, the five operator tests 57 s; the whole ignored run 16 min.

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
- On several ranks, the one-rank `Fmm` of the same settings over the union of every
  rank's points in rank order, bit for bit (Phase 5 T6, decision 7).
- The `P2pChoice::Reference` run of the same `Fmm` settings, for every ISA's output.
- `nd_fmm_ref::p2p::direct_sum` in f64 over the original coordinates, for every
  complete FMM.
- `nd-octree` and `nd-fmm-plan` for the octant and offset order of the tables.
- The single-translation prediction of Phase 1 (design §7) for the C3.2 gate.

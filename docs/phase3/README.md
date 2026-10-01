# Phase 3: CPU FMM on a redesigned nd-fmm-plan

Phase 3 runs the first complete Laplace FMM, on the host, through `nd-fmm-plan`.

`nd-fmm-plan` was partly written before the library design existed. It plans the
right things (interaction lists, pass order, ghost exchange, global levels), but its
data model does not suit the rest of the design:
- boxes are found through `HashMap`s keyed by Morton key;
- level columns follow insertion order;
- every leaf carries the same amount of data;
- operators act on one pair of boxes at a time, through `&self`;
- target positions have nowhere to live.

Extending it piecemeal (variable leaf data now, batched hooks in Phase 4,
device-friendly buffers in Phase 5) would mean writing the Laplace operator against an
interface that is then replaced twice. The crate may be rewritten completely, so this
phase redesigns it first.

The phase has three parts:
1. **Design and conventions.** A design document for `nd-fmm-plan` (signed off by
   hand), and CONVENTIONS §3.13 on Laplace leaf data and relative box geometry.
2. **The new `nd-fmm-plan`.** Built in four tasks next to the old code, which stays
   as the reference until the new code reproduces it on every scenario and then is
   removed. The new crate has:
   - a Morton-ordered integer index of the boxes of every level;
   - index-based interaction lists, grouped by V-list offset and by child octant;
   - variable-size leaf data;
   - a level-batched operator interface.

   This absorbs C3.0 (variable-size leaf data) and C4.0 (batched hooks).
3. **The Laplace FMM.** `nd-fmm-exec` implements the new interface with the Phase 2
   tables and the `nd-fmm-ref` leaf operators. It is checked operator by operator, then
   on uniform and clustered distributions against the direct sum. It is then threaded
   with rayon inside each rank, bit-identical to the serial path, and finally calibrated
   (p against accuracy).

Companion documents: [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md)
(Sections 2.4, 2.5, 4, 5.2, 5.3, 6.5, 8 and 9; components C3.0–C3.4 and C4.0 in Section
7; the Phase 1 "Single-translation accuracy" table and the Phase 2 recommendation for
the CPU M2L) and [docs/design/workspace-structure.md](../design/workspace-structure.md)
(Section 1.1, `nd-fmm-plan`; Section 3, `nd-fmm-exec`; Section 4; Section 5; Section 6,
"Still open"). T1 adds a third: `docs/design/fmm-plan-redesign.md`.

Prerequisite: Phase 2 is complete. T8 (tables report and table-path accuracy) is merged
through PR #23, and the design documents carry the Phase 2 outcome. The briefs assume
the workspace as it is after that merge.

## Scope

In scope:
- `docs/design/fmm-plan-redesign.md`: an assessment of the current crate and the design
  of its replacement.
- `nd-fmm-plan`, rewritten:
  - the box index and the interaction lists;
  - the leaf and level data stores and the ghost exchange;
  - the operator interface, the evaluator and `IndexFmm`;
  - removal of the old API.

  It stays kernel-agnostic and keeps its package and library names.
- `docs/CONVENTIONS.md`: a new section §3.13 on Laplace leaf data and relative box
  geometry, and `tools/fixtures/check_leaf_geometry.py`.
- `nd-fmm-exec` (host path only):
  - the crate, box geometry from Morton keys, and the cross-check of the table order;
  - `LaplaceOperator<T>` on the new interface (C3.1);
  - `FmmBuilder` and `Fmm` (C3.2);
  - host threading with rayon inside each rank, opt-in and bit-identical to the serial
    path (C3.5, new);
  - adaptive trees through the W and X lists (C3.3).
- `nd-fmm-validate`: FMM accuracy reports for four distributions and the calibration
  sweep (C3.2–C3.4).

Out of scope:
- Changes to `nd-octree`. If the redesign needs one, T1 proposes it as a separate
  decision, and the phase works around it until then.
- GEMM, CubeCL and device buffers (Phase 4). The host operator executes each level
  pair by pair, serially until T10 and then optionally threaded.
- Threading inside `nd-fmm-plan` (list building, exchange staging) and in
  `nd-fmm-tables` (table building).
- Performance work. Phase 3 gates on accuracy only (design §9.1); timings are reported,
  never asserted. Plan building is measured, not tuned.
- Redistributing points to their owning ranks, and multi-rank Laplace runs. T1
  *designs* the redistribution API, so `Fmm`'s interface does not change later; C5.1
  implements it. The new `nd-fmm-plan` itself is checked on 1, 2 and 4 ranks.
- Overlap of exchange and computation (C5.2). The design must not preclude it.
- SVD-compressed M2L (C6.2), dipoles and other source types (C6.4), several charge
  vectors at once (C6.3).
- Changes to `nd-fmm-tables` and `nd-fmm-ref` beyond fixing a defect that a test
  exposes (stop and report first).

## Requirements on the new nd-fmm-plan

T1 designs against these, and T4–T7 are accepted against them. T1 may propose changing
one, with reasons, for sign-off.

1. **Kernel-agnostic.** No kernel arithmetic. `IndexFmm` stays the exact topology test.
2. **Integer indexing.** On every level, the boxes that a rank holds (local, `Global` and
   ghost) are numbered 0..n in Morton order. Hot paths never look a key up in a
   `HashMap`. Numbering is deterministic: the same tree on the same ranks gives the same
   numbers, which answers design §9.2 (Morton column order) with yes.
3. **Level-contiguous coefficients.** Multipoles and locals are one buffer per level, the
   coefficients of box i at i · size, so a level is a column-major size × n matrix and
   a GEMM operand.
4. **Variable-size leaf data.** Per leaf, CSR-style, with zero counts allowed:
   - source data, exchanged for ghost leaves of the U and X lists;
   - target input (for example target positions), local only and never exchanged;
   - target output, local only.

   Sizes per point are fixed; counts per leaf vary.
5. **Index-based lists.**
   - U, W and X are CSR over box indices.
   - V is grouped by the 316 offsets in the order of CONVENTIONS §3.12
     (`V_LIST_DIRECTIONS`): per (level, offset), parallel arrays of target and source
     indices, in which each target appears at most once.
   - M2M and L2L are grouped by (level, child octant), with octant
     o = `morton::child_index`.
   - The same lists are also available target-centrically: per level, CSR from each
     target box to its V-list sources with each entry's offset index, from each parent
     to its children with their octants, and from each child to its parent. A host
     operator can then give every target's output slice to one thread, borrowed
     mutably with safe Rust (`par_chunks_mut` on a level buffer, disjoint leaf chunks),
     without `unsafe` and without atomics.
   - The list rules are those of today's `InteractionManager`; its brute-force oracle
     in `tests/mpi_regressions.rs` is the specification.
6. **Level-batched operator interface.**
   - One call per level and operator kind, which receives the level's buffers and both
     views of requirement 5: the groupings per octant or offset (for the GEMM path of
     Phase 4) and the target-centric CSR (for the host path). The operator chooses its
     execution order and documents it. If T1 finds finer calls (per octant, per offset)
     better for Phase 4, it may keep them as well, provided the target-centric view
     stays available.
   - Operators can read the keys of the boxes they act on, for geometry.
   - Operators get mutable access to their own state (for scratch), so no `RefCell`
     is needed.
   - A per-pair adapter lets a simple operator (`IndexFmm`, tests, a reference path)
     implement one method per pair instead.
7. **Distributed semantics unchanged.**
   - The same pass order, global coarse levels and ghost exchange as today.
   - Every rank reaches every collective, including ranks with empty input.
   - Exchange buffers are flat and contiguous per level, in index order, so they can be
     staged to and from a device.
8. **Deterministic.** For a fixed tree and number of ranks, results are bit-identical
   from run to run, and for the Laplace host path also for any number of threads. Every
   accumulation into a target happens in a documented, fixed order.
9. **Redistribution designed.** An API that moves per-item payloads (points with their
   data) to the ranks that own their leaves, collectively. It is implemented in C5.1.
10. **Tested like today, and more.** The existing serial tests, the `mpi_regressions`
    scenarios and their list oracle carry over. Every scenario runs the new evaluator
    with `IndexFmm`, with counts of one and with seeded variable counts, on 1, 2 and 4
    ranks by hand. Until T7, the old evaluator stays the reference, and the new one
    must reproduce its `IndexFmm` result on every scenario.

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **Old code as reference.** T4–T6 add the new modules beside the old ones, under a
  new module path. Nothing of the old API changes until T7 removes it, after T6 has
  shown that both agree. `nd-fmm-exec` is written only against the new API.
- **Leaf-scaled data (§3.13, drafted in T2, signed off on 2026-10-01).** This refines
  design §5.3, which interleaved absolute (x, y, z, q):
  - A point x in leaf b is stored as u = (x − c_b)/r_b ∈ [−1, 1]³, computed in f64
    from the user's coordinates and the domain, then rounded to T.
  - A source chunk of n points holds the n coordinate triples, then the n charges, so
    `as_chunks::<3>()` gives `[[T; 3]]` without copying. Target positions use the
    target-input store (requirement 4), in the same leaf-scaled form.
  - Target output holds leaf-scaled potentials φ̂ = r_t Σ q/|x − y| and, with
    gradients, ĝ = r_t² ∇ₓ Σ q/|x − y|.
  - Another box's frame, seen from box t, has centre (c_s − c_t)/r_t and radius
    r_s/r_t. Both are exact dyadic rationals computed from the integer key indices,
    in f32 as in f64. With these frames, P2M, L2P, P2L and M2P call
    `nd_fmm_ref::leaf` unchanged. Only P2P between two different leaves maps its
    sources into scratch first.
  - The domain enters only when points are loaded and when output leaves the FMM:
    φ = φ̂ / (4π r_t) and ∇φ = ĝ / (4π r_t²), applied once by `Fmm` (§3.1).

  Why: a shift formed from floating-point centres carries a relative error of
  ε |c| / r_l (§3.12), up to 6.5e4 ε at level 16, which costs four digits in f32.
- **Tables by integer key, never by shift.** M2M and L2L use the octant of the batch,
  M2L the offset index of the batch. T8 checks that the order of `nd-fmm-tables`, of
  `nd-octree` and of the new `nd-fmm-plan` agree.
- **Host execution.** `LaplaceOperator` executes each level target by target, through
  the target-centric view, with each target's contributions in CSR order. Scratch is
  sized at construction from the largest leaf (a global maximum, one all-reduce). No
  allocation per pair or per level.
- **Threading (T10).** rayon parallelises the target loop of each level call in
  `nd-fmm-exec` only. Each thread writes only its own targets' slices, and every
  target sees the same accumulation order as in the serial loop, so the output is
  bit-identical for every thread count.
  - Opt-in: `FmmBuilder::threads(n)`, default 1, so that several MPI ranks per node do
    not oversubscribe cores by default.
  - A pool owned by the `Fmm` (`rayon::ThreadPoolBuilder`), never the global pool.
  - Per-thread scratch created at construction, with no allocation in the loop and no
    `unsafe`.
  - Worker threads never call MPI. With more than one thread, MPI must be initialised
    with at least `Threading::Funneled`; `build` checks the provided level and returns
    an error otherwise.
- **Threads and BLAS.** Nested thread pools oversubscribe cores: a GEMM called from
  inside a rayon worker would start its own BLAS threads on every worker.
  - Phase 3 makes no BLAS call in any compute path. Tables are applied by
    `MatrixSet::apply`, a hand-written loop; `nd-fmm-ref` is plain Rust; `nd-fmm-plan`
    uses rlst only for its distributed tools. T10 audits this and reports it.
  - The rule for this and every later phase: a matrix product called inside a rayon
    worker runs single-threaded. Ranks × rayon threads × BLAS threads (and any other
    pool, such as the CubeCL CPU runtime of Phase 4) stay at or below the physical
    cores. Large products outside rayon (for example the SVD of C6.2 at table build)
    may use BLAS threads. Never nest two pools that both fill the machine.
  - BLAS threads are set by the launcher through environment variables, before the
    process starts: `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
    `BLIS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` for Accelerate on macOS. With
    Open MPI they are passed with `mpirun -x`.
  - The library never sets environment variables: OpenBLAS reads them once at
    initialisation, so setting them later has no effect, and Rust 2024 makes
    `std::env::set_var` unsafe. The library reads and reports them instead (T10).
  - rlst's runtime control, `rlst::threading::set_blas_threads`, needs no feature
    since rlst 0.9.0: on Linux and macOS it finds a dynamically linked OpenBLAS, MKL,
    BLIS, FlexiBLAS or Accelerate at runtime, and returns
    `BlasThreadingError::NoBackendFound` if there is none. A statically linked
    backend needs its feature (`openblas_threading` for OpenBLAS), which now links
    the thread-control functions itself; in 0.8.0 the `extern` declaration of
    `openblas_set_num_threads` sat under `cfg(feature = "mkl_threading")`, so that
    feature alone failed to link.
- **Strategy.** `M2lStrategy::{Dense, Classes, Rotation, Auto}`:
  - `Dense`: dense octant tables and `M2lTables`;
  - `Classes`: dense octant tables and `M2lClasses` (memory, not speed);
  - `Rotation`: `RotationTables` for all three translations;
  - `Auto`: `Dense` for p ≤ 8 and `Rotation` for p ≥ 9 (Phase 2 T8; single-threaded
    per-pair timing, within about 20% in between).

  Only the tables a strategy uses are built. `Compressed` comes with C6.2.
- **Table cache.** Used only when the caller passes a directory
  (`FmmBuilder::table_cache(dir)`): no default directory, no environment variable.
  Every rank calls `load_or_build`, which Phase 2 T7 made safe for concurrent writers.
- **Precision.** `T` is f32 or f64, with the bound `T: RealScalar + Equivalence`. Tables
  are the cast f64 tables. f32 runs use p ≤ 8 (design §4). The oracle is always the
  f64 `direct_sum` over the original f64 coordinates.
- **Domain.** `Fmm` uses `compute_global_bounding_box` unless the caller supplies a
  domain, which must be cubic and contain every point strictly; anything else is an
  error on every rank.
  - That function forms each side as the difference of two rounded corner coordinates,
    so its sides agree only up to about ε (|corner| + w). Cubic therefore means equal
    sides up to a small multiple of that, not to a fixed relative tolerance, which
    would reject its own output for points far from the origin. T3 fixes the constant.
  - The same ε (|x| + |a|) / w already limits the user's coordinates, so the mismatch
    never costs more than the input's own precision.
- **MPI.** The root rules apply. In addition:
  - At most one MPI-initialising test per test executable. In `nd-fmm-exec` it is
    `tests/mpi_exec.rs` (T8) with a `cases` list; large ignored tests get their own
    executable. Operator and geometry tests do not initialise MPI.
  - Until C5.1, `Fmm` on more than one rank returns `FmmError::PointsNotOwned` on
    every rank, agreed by one all-reduce. T9 checks by hand on 2 ranks that this
    neither hangs nor diverges.
  - Examples that initialise MPI are not registered with templated-examples (the
    weekly job runs at 3 ranks), except the `nd-fmm-plan` examples, which support
    any rank count.
- **Error measures.** The Phase 1 and Phase 2 measures carry over for operators: per
  degree, in the §3.8 weighting, relative to the term magnitudes ("terms"), and a
  dyadic domain for tight tolerances. For the full FMM:
  - relative L2 and max error of φ and ∇φ (`nd_fmm_validate::metrics`) against
    `direct_sum` in f64 over all sources;
  - over all targets when N ≤ 10⁴, otherwise over 1,000 targets sampled with a fixed
    seed (design §8.2);
  - charges uniform in [−1, 1), as in the Phase 1 prediction;
  - the oracle expands 1/|x − y|; tests compare with 4π times the `Fmm` output, or
    divide by 4π, and say which.
- **Predictions.** The single-translation table of Phase 1 T7 (design §7) gives
  relative L2 errors of 1.77e-3, 1.08e-5 and 2.71e-9 at p = 3, 8 and 18. The C3.2 gate
  allows twice that. If a gate fails, the task reports the measured errors and their
  breakdown and stops. It does not tune tolerances or operators to pass.
- **Test time.** Keep the debug-mode runs of `cargo test -p nd-fmm-plan` and
  `cargo test -p nd-fmm-exec` under a minute each. Large-N checks go into `#[ignore]`
  tests or `nd-fmm-validate` examples, run in release mode and reported.
- **Dependencies.**
  - `nd-fmm-exec`: `nd-fmm-math`, `nd-fmm-ref`, `nd-fmm-tables`, `nd-fmm-plan`,
    `nd-octree`, `mpi`, `rlst` and `thiserror`, through `[workspace.dependencies]`;
    dev-dependency `proptest`. T3 adds `mpi` 0.8.2 (feature `derive`), `rlst` 0.9.0
    (no features; `nd-octree` enables `mpi` for the build) and path entries for
    `nd-octree` and `nd-fmm-plan`.
  - `nd-fmm-plan`: whether the rewrite moves it to `[workspace.dependencies]` and
    `[workspace.package]`, and fixes its stale metadata (licence form, `homepage`
    typo, `repository`), is a T1 sign-off question; T7 carries out the answer.
    `nd-octree` stays as it is.
  - `rayon` (`"1"`, through `[workspace.dependencies]`) in `nd-fmm-exec` only, from
    T10.
  - No CubeCL, and no library dependency on `nd-fmm-validate`.

## Exit gate
- Every acceptance test in the task briefs passes in CI. `nd-fmm-exec` is a default
  member; CI runs everything on one rank.
- `docs/design/fmm-plan-redesign.md` is signed off before T4 starts, and CONVENTIONS
  §3.13 before T3 starts.
- C3.0 (now the redesign): the new `nd-fmm-plan` meets requirements 1–10. On 1, 2
  and 4 ranks, every `mpi_regressions` scenario passes its list oracle. `IndexFmm`
  passes with counts of one and with seeded variable counts (zero included), and
  equals the old evaluator's result before T7 removes it.
- C4.0 (absorbed): every batched call honours its grouping. Each target appears at
  most once per (level, offset), every octant batch is complete, and the per-pair
  adapter equals the batched path for `IndexFmm`.
- C3.1: on levels 2, 9 and 16 of a dyadic domain, each of the eight operators equals
  `nd-fmm-ref` for the same geometry, in f64:
  - M2M, L2L and M2L against `direct`: to 1e-14 (terms, per degree) with `Dense`, and
    to 1e-13 with `Classes` and `Rotation`;
  - P2M, P2L, L2P, M2P and P2P against `nd_fmm_ref::leaf` and `p2p` at the absolute
    frames: to 1e-13 (terms).

  The octant and offset order of `nd-fmm-tables`, `nd-octree` and `nd-fmm-plan` agree.
  A non-cubic domain is rejected.
- C3.2: on a uniform level-4 tree with N = 10⁵, the relative L2 error of φ against the
  direct sum is at most twice the single-translation prediction at p = 3, 8 and 18
  (f64): 3.5e-3, 2.2e-5 and 5.4e-9.
- C3.3: on a sphere surface, a Plummer sphere and Gaussian clusters (N = 10⁵, adaptive
  trees with non-empty W and X lists), the relative L2 error of φ is at most twice the
  uniform-tree error at the same p.
- C3.5 (new): with 1, 2, 4 and 8 threads, `Fmm` output is bit-identical to the serial
  path on every T9 and T11 scenario, in f32 and f64. The T10 PR reports the speed-up
  per stage at N = 10⁵ (not asserted), measured with every BLAS thread variable set
  to 1, and confirms that no BLAS routine is called inside a rayon worker.
- C3.4: the T12 PR contains the calibration table: the smallest p for relative L2 errors
  from 1e-3 to 1e-12 in f64 and to the f32 floor, per distribution, for φ and ∇φ. It
  is copied into the design document.

## Tasks

One pull request each.
- T1, T2 and T3 touch disjoint files and can start at once. T3 needs §3.13 (T2)
  signed off before it starts its geometry module; its scaffold part does not.
- T4 needs T1 signed off. T5 needs T4, T6 needs T5, and T7 needs T6.
- T8 needs T3 and T6 (not T7: it uses only the new API). T9 needs T8.
- T10 and T11 both need T9 and can run in parallel. Both add scenarios to
  `tests/mpi_exec.rs`; merge one, then rebase the other.
- T12 needs T10, T11 and T7. It runs the calibration threaded.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-plan-design.md](T1-plan-design.md) | `docs/design/fmm-plan-redesign.md`: assessment of the current crate, design of the new one, migration and test plan | design for C3.0, C4.0 | Phase 2 complete |
| T2 | [T2-leaf-data-conventions.md](T2-leaf-data-conventions.md) | CONVENTIONS §3.13 draft, `check_leaf_geometry.py` | prerequisite of C3.1 | none |
| T3 | [T3-exec-geometry.md](T3-exec-geometry.md) | `fmm-exec` crate, workspace entries, `geometry` module, table order against `nd-octree` | part of C3.1 | T2 (signed off) |
| T4 | [T4-plan-index.md](T4-plan-index.md) | `nd-fmm-plan`: Morton-ordered box index, index-based U/V/W/X lists, offset and octant groupings | C3.0 | T1 (signed off) |
| T5 | [T5-plan-data-exchange.md](T5-plan-data-exchange.md) | `nd-fmm-plan`: level and leaf data stores, variable-size ghost exchange, global levels | C3.0 | T4 |
| T6 | [T6-plan-evaluator.md](T6-plan-evaluator.md) | `nd-fmm-plan`: batched operator trait, per-pair adapter, evaluator, `IndexFmm`; equality with the old evaluator | C3.0, C4.0 | T5 |
| T7 | [T7-plan-cleanup.md](T7-plan-cleanup.md) | `nd-fmm-plan`: old API removed, docs, `CLAUDE.md`, manifest as signed off, registered example | C3.0 | T6 |
| T8 | [T8-laplace-operator.md](T8-laplace-operator.md) | `LaplaceOperator<T>` on the batched interface, table selection | C3.1 | T3, T6 |
| T9 | [T9-uniform-fmm.md](T9-uniform-fmm.md) | `FmmBuilder`, `Fmm`, `Output`; uniform-tree accuracy report in `nd-fmm-validate` | C3.2 | T8 |
| T10 | [T10-host-threads.md](T10-host-threads.md) | rayon threading of the host path, opt-in, bit-identical to serial; speed-up report | C3.5 (new) | T9 |
| T11 | [T11-adaptive-fmm.md](T11-adaptive-fmm.md) | adaptive-tree scenarios, Plummer and Gaussian-cluster distributions, clustered accuracy | C3.3 | T9 |
| T12 | [T12-calibration.md](T12-calibration.md) | calibration of p against accuracy, stage timings, design-document update | C3.4 | T10, T11, T7 |

Review T1 and T2 yourself before the tasks that build on them. T4–T6 encode the
redesign, and T3, T8, T9 and T11 encode §3.13, so a change afterwards means revisiting them.

T4–T7 work inside fmm-plan/ and follow fmm-plan/CLAUDE.md, including its multi-rank
runs. Rewriting the crate is authorised by this phase; the "preserve in-progress work"
rule there means: keep the old code until T7 removes it.

T3, T7, T9 and T10 change the root `Cargo.toml` or `Cargo.lock`. Rebase whichever merges
later, and regenerate the lock file with the manifest change (root `CLAUDE.md`,
"Layout").

## Risks

| Risk | Mitigation |
| --- | --- |
| The rewrite takes longer than the extension it replaces and delays the Laplace work | T2 and T3 run alongside T1–T6; T8 starts as soon as T6 lands; the old code keeps working until T7 |
| The rewrite loses a correctness property the old code had (list edge cases, empty ranks, ghost-children layer) | the old tests and list oracle are the specification; T6 requires equality with the old evaluator on every scenario, on 1, 2 and 4 ranks |
| The batched interface is shaped by the host Laplace operator and does not fit GEMM | requirements 3, 5 and 6 come from design §5.3 and §6.4–§6.6; T1 checks the interface against a sketch of the Phase 4 GEMM M2L and states the gathers it needs |
| A multi-rank defect hangs a collective | every multi-rank run under an external timeout; validation before any collective agreed on all ranks |
| Rayon threads and BLAS threads oversubscribe the cores, now or once a later phase adds a GEMM inside a worker | the "Threads and BLAS" rule; T10 audits that Phase 3 calls no BLAS and reports the BLAS variables on every run; T12 writes the rule into the design document for Phase 4 and C6.2 |

## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase3/T<k>-<name>.md and do that task." Review and merge before the next.

## Exit checklist
- [x] T1 merged: redesign signed off; sign-off decisions recorded in the document
- [x] T2 merged: `check_leaf_geometry.py` passes; §3.13 drafted
- [x] CONVENTIONS §3.13 reviewed and signed off; `CONVENTION_VERSION` and §3.10 decisions recorded
  - Signed off by Timo Betcke on 2026-10-01. Every T2 recommendation is accepted as
    written: (1) §3.13 as merged in PR #27; (2) `CONVENTION_VERSION` stays 1, and the
    §3.10 bump rule is not extended to §3.13; (3) source chunks hold coordinates, then
    charges; (4) target chunks hold leaf-scaled values.
- [x] T3 merged: `nd-fmm-exec` skeleton, CI green, root `CLAUDE.md` points to Phase 3; relative frames exact in f32 and f64; table order agrees with `nd-octree`
- [x] T4 merged: new index and lists equal the brute-force oracle on 1, 2 and 4 ranks
- [x] T5 merged: variable-size ghost exchange passes on 1, 2 and 4 ranks
- [x] T6 merged: new evaluator equals the old one for `IndexFmm` on every scenario; variable counts pass; on 1, 2 and 4 ranks
- [x] T7 merged: old API removed; `fmm-plan/CLAUDE.md` and crate docs describe the new crate
- [x] T8 merged: each operator equals `nd-fmm-ref` on levels 2, 9 and 16 (1e-14 dense; 1e-13 classes, rotation and leaf operators)
- [ ] T9 merged: uniform tree, N = 10⁵, within twice the prediction at p = 3, 8 and 18
- [ ] T10 merged: threaded output bit-identical to serial for 1, 2, 4 and 8 threads; speed-up reported
- [ ] T11 merged: sphere, Plummer and Gaussian clusters within twice the uniform error
- [ ] T12 merged: calibration table in the PR
- [ ] Design documents updated: laplace-fmm-plan §5.1–§5.3 (the new `nd-fmm-plan` interface, leaf-scaled data), §7 (Phase 3 status and numbers; C3.0, C4.0 and the new C3.5 as delivered; Phase 4 without C4.0), §9.1 and §9.2; workspace-structure §1.1, §3, §3.1, §4 and §6 match the result

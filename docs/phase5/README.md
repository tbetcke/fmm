# Phase 5: the distributed FMM

Phase 5 runs the Laplace FMM on several MPI ranks. Most of the machinery already exists:
- `nd-octree` builds and partitions the tree across a communicator.
- `nd-fmm-plan`'s `Evaluator` runs the distributed pass order: the ghost-source exchange
  for U and X, the ghost-multipole exchange for V and W, and the replicated coarse
  (`Global`) levels (fmm-plan-redesign §7, §8).

What is missing (design §7, "Phase 5"; laplace-fmm-plan §7, "Recommendation for Phase
5"):
- **Points never move.** `Fmm::build` bins each rank's points into keys and builds the
  octree from them. The octree then owns Morton ranges that do not match where the
  points were passed in, so on several ranks `build` returns `PointsNotOwned` unless the
  caller happened to pass every point on its owning rank. The redistribution API of
  fmm-plan-redesign §9 is designed but not implemented.
- **No multi-rank Laplace result has been checked against a one-rank result.**
  `tests/mpi_exec.rs` runs a single hand-shaped scenario (`device backends`, octant
  shares) distributed against the direct sum. Every other scenario stops with
  `PointsNotOwned` on several ranks.
- **The device path runs on one rank only** (`DeviceNeedsOneRank`). On several ranks the
  evaluator moves host data around the exchanges, and a device operator does not see
  those moves. device-path.md §4.5 sketches the `nd-fmm-plan` hook it needs.
- **Every exchange blocks.** rlst 0.9.0 offers only the blocking
  `MPI_Neighbor_alltoallv`, and nothing overlaps communication with computation (C5.2).
- **Nothing has been measured on several ranks**, and CI runs one rank only.

Hardware (decided on 2026-10-04): **the Apple M3 Max is the only machine.** It has 12
performance and 4 efficiency cores, 64 GB, and Open MPI 5.0.10 from Homebrew. Every
multi-rank run and every scaling figure of the phase is therefore one node over shared
memory, with at most 12 ranks × threads on the performance cores. Inter-node scaling is
documented as a single command for whoever later has a cluster, as CUDA was in Phase 4.
CI adds a multi-rank job on GitHub's `ubuntu-latest` runners (decision 4), for
correctness only.

The device path goes to several ranks in this phase (decided on 2026-10-04), **for
correctness only**: each rank opens its own device. On the M3 Max that means the CubeCL
CPU runtime per rank, or Metal with every rank sharing the one GPU. No device scaling
figure is claimed.

`IndexFmm` is retired (decided on 2026-10-04). The index FMM in `nd-fmm-plan`
(`IndexFmm`, `BatchedIndexFmm`, `run_index_fmm` and their tests and examples) was a
Phase 3 topology test. It is removed (T2), and nothing in this phase is designed or
checked with it. Its two jobs move:
- the values of the distributed evaluator are checked by the Laplace FMM in
  `nd-fmm-exec`, on N ranks against one rank and against the direct sum (T6);
- the call order and the groupings stay checked by the recording operator of
  `tests/mpi_regressions.rs`, which needs no index FMM.

`nd-octree` may change where the distributed FMM needs it (decided on 2026-10-04). This
reverses the "no nd-octree change" rule of Phases 3 and 4. T1 lists the changes, they are
signed off with the design, and T4 makes them under octree/CLAUDE.md.

The phase has four parts:
1. **Decisions.** A design document for the distributed FMM (signed off by hand), the
   retirement of `IndexFmm`, and a multi-rank CI job.
2. **Points on their owners (C5.1, host).** The `nd-octree` changes that the design
   asks for (load balance, robustness), the redistribution of points in
   `nd-fmm-plan`, and `Fmm` on any number of ranks, equal to its one-rank result.
3. **The device on several ranks (C5.1, device).** A kernel-agnostic data-movement hook
   in `nd-fmm-plan`, and the device operator on several ranks with device-resident ghost
   buffers.
4. **Overlap and scaling (C5.2, C5.3).** Non-blocking exchanges overlapped with local
   work, then strong and weak scaling on the M3 Max, and the design-document update.

Companion documents:
- [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md): Sections 5, 6.8, 7
  (Phase 4 "Recommendation for Phase 5", the Phase 5 table), 8 and 9;
- [docs/design/fmm-plan-redesign.md](../design/fmm-plan-redesign.md): §3.3, §3.5, §7
  (with §7.3 collectives and §7.5 accumulation order), §8, §9 (redistribution) and §10
  (device and overlap compatibility);
- [docs/design/device-path.md](../design/device-path.md): §4.2–§4.5 (the evaluator's host
  writes and the C5.1 hook), §11 (threads), §12 (errors), §14 (multi-rank and overlap)
  and §17;
- [docs/design/workspace-structure.md](../design/workspace-structure.md): Sections 3, 4
  and 6.

T1 adds a further companion, `docs/design/distributed-fmm.md`.

Prerequisite: Phase 4 is complete. T13 is merged (PR #59), and the design documents carry
the Phase 4 outcome. The briefs assume the workspace as it is after that merge.

## Scope

In scope:
- `docs/design/distributed-fmm.md`, signed off; the design-document updates at the end of
  the phase.
- `nd-fmm-plan`:
  - removal of the index FMM (T2);
  - `Redistribution`, as designed in fmm-plan-redesign §9 and refined by T1 (T5);
  - the data-movement hook of device-path.md §4.5, as refined by T1 (T7);
  - non-blocking exchanges and overlapped stages (T9).
- `nd-octree`: the changes T1 proposes and the sign-off accepts, for example load
  balance by point count or by a cost weight, finer coarse partitions, and errors
  instead of panics on inputs that a distributed `Fmm` can produce (T4).
- `nd-fmm-exec`:
  - `Fmm` redistributes sources and targets, on any number of ranks, with the same
    `build` and `evaluate` signatures; `PointsNotOwned` goes (T6);
  - multi-rank checks against the one-rank result in `tests/mpi_exec.rs` and the
    ignored gates (T6);
  - the device operator on several ranks, through the hook (T8);
  - overlap on the host path, and on the device path as far as T1 decides (T9).
- `nd-fmm-validate`: multi-rank accuracy runs (T6), and the scaling harness and report
  (T10).
- `.github/workflows/`: a multi-rank job (T3), kept, changed or dropped at its sign-off.

Out of scope:
- Clusters, inter-node runs, and any GPU besides the M3 Max's (decision 2). The
  inter-node scaling run is one documented command.
- Device scaling. Multi-rank device runs are checked for correctness and their transfers
  counted. They are not timed as scaling figures, since every rank shares one GPU.
- Overlapping P2P with the far field on a second device stream (device-path.md §14:
  wgpu's streams on Metal submit to one queue). T1 may propose it for Phase 6.
- Dynamic load balancing between evaluations, and rebuilding the tree while points move.
  A new point set means a new `Fmm`, as today.
- Threading inside `nd-fmm-plan` or `nd-octree`. Threads stay in `nd-fmm-exec`'s rayon
  pool, and worker threads never call MPI.
- FMM3D comparisons (decision 9 of Phase 4: FMM3D in a later phase), C6.1–C6.5, and p
  > 20.
- Changes to `nd-fmm-math`, `nd-fmm-ref`, `nd-fmm-tables`, `nd-fmm-simd` and
  `nd-fmm-kernels` beyond fixing a defect that a test exposes (stop and report first).
  `nd-fmm-kernels` may gain gather and scatter launches for packed exchange buffers if
  T8 needs them and the existing `movement` module does not cover them; T1 says which.
- Changes to the host path's defaults (strategy `Auto`, `P2pChoice::Auto`, the leaf size
  of 64) or to its one-rank results.

## Requirements on the distributed FMM

T1 designs against these, and T2–T10 are accepted against them. T1 may propose changing
one, with reasons, for sign-off.

1. **Same interface.** `FmmBuilder::build(sources, targets, comm)` and
   `Fmm::evaluate(charges)` keep their signatures. Each rank passes any subset of the
   points, empty included, and receives the output for its own targets in the order it
   passed them. Where the points are is the library's business, not the caller's.
   `PointsNotOwned` disappears.
2. **Equal to one rank.** On 2, 4 and 8 ranks the output equals the one-rank output of
   the same settings to precision, under a tolerance that T1 derives and the sign-off
   fixes. Bit identity across rank counts is not required: the order of points within a
   leaf, and of local and ghost entries in the near and X rows, depends on the
   distribution (fmm-plan-redesign §3.3, §9). The errors against the direct sum match
   the one-rank run's.
3. **Deterministic.** For fixed input on each rank, a fixed rank count and fixed
   settings, the output is bit-identical from evaluation to evaluation, from run to run,
   and for every thread count (C3.5). Overlap changes no bit unless the sign-off accepts
   a documented new order (decision 9).
4. **Collective discipline.** Every rank enters every collective in the same order,
   including ranks with no points (root `CLAUDE.md`, "MPI"). Every error that depends on
   one rank's input or device is agreed before the next collective. Worker threads never
   call MPI. A multi-rank test never hangs: if one rank fails, every rank fails.
5. **Kernel-agnostic plan.** Redistribution, the data-movement hook and overlap live in
   `nd-fmm-plan` and know nothing of Laplace. They are tested there with operators that
   carry no kernel arithmetic, and in `nd-fmm-exec` with the Laplace operator.
6. **Safe.** No `unsafe` in `nd-octree`, `nd-fmm-plan` or `nd-fmm-exec` (root
   `CLAUDE.md`). A non-blocking exchange therefore uses safe rsmpi calls (scoped
   requests), or an addition to rlst that the sign-off approves. It never uses raw MPI
   FFI in the workspace.
7. **Minimal device traffic.** On several ranks the device operator moves exactly the
   data each exchange sends and receives, packed on the device, plus the one-rank
   minimum (charges up, output down). The design states the transfers and syncs per
   evaluation as a formula, and a test counts them.
8. **Memory per rank.** Points, leaves and coefficients per rank scale as N/P plus the
   ghost layer. The design states what is replicated on every rank (the coarse tree and
   coarse blocks, `Global` boxes, the O(P) partition data) and how it grows with P.
9. **Tested on what can run.** Every multi-rank test runs at 1, 2 and 4 ranks (in the CI
   job, if kept) and at 8 by hand on the M3 Max, under an external timeout, with the
   macOS loopback flags. Every report states the rank counts, threads per rank, the
   cores used, and the backends.
10. **Measured, never asserted.** Timings are reported from the M3 Max in release
    builds, with every BLAS thread variable set to 1 and ranks × threads at most 12 (the
    performance cores). No timing is asserted in a test or taken in CI.

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **No new crate.** Redistribution, the hook and overlap belong in `nd-fmm-plan`
  (workspace-structure §3, "fmm-dist" dropped). The Laplace side stays in `nd-fmm-exec`.
  `nd-octree` changes only as T1 proposes and the sign-off accepts.
- **The oracles.** The trusted slower path of a multi-rank run is the **one-rank `Fmm`
  of the same settings over the union of every rank's points**, and behind it the f64
  `direct_sum`. In a test every rank can build it on a one-rank communicator
  (`SimpleCommunicator::self_comm()`, or a split of the world). The one-rank FMM then
  needs no collective across ranks, and each rank compares its own targets. The
  `tests/mpi_regressions.rs` list oracle, exchange checks and recording operator stay
  the topology oracles of `nd-fmm-plan`.
- **Accuracy measures.** As in Phase 3 ("Error measures") and Phase 4:
  - **multi-rank against one rank:** relative L2 of the output difference over every
    target, for φ and ∇φ, summed over ranks. Provisional bounds until T1's sign-off:
    1e-12 in f64 and 1e-5 in f32, the device-versus-host bounds of Phase 4. The relative
    max difference is reported;
  - **against the direct sum:** the root mean square over eight charge vectors at 1,000
    sampled targets, within 0.1% (f64) and 1% (f32) of the one-rank run's;
  - if a check fails, the task reports the measured differences and their breakdown and
    stops. To break a difference down, run every stage with one rank's data, by operator
    kind and by list (local against ghost entries). It does not tune tolerances to pass.
- **Workloads.**
  - The debug scenarios of `tests/mpi_exec.rs`, now on every rank count.
  - The C3.2 cube and the C3.3 Plummer sphere (a = 0.1) at N = 10⁵ and 10⁶,
    `max_level` 16, 64 points per leaf, eight charge vectors, gradients. These run at
    (f64, p = 3, 8 and 18) and (f32, p = 3 and 8), the Phase 4 workload points.
  - The Gaussian clusters, the optional third case: a stress case for load balance.
  - Input distributions over ranks: every point on rank 0; a random share per rank;
    the octree's own partition; and one empty rank. Each distribution must give the same
    result within requirement 2.
- **Ranks on the M3 Max.** At most 12 ranks × threads per rank on the performance
  cores. macOS has no process binding, so efficiency cores can be scheduled anyway; every
  report prints the rank count, threads per rank and the BLAS variables. 16 ranks
  (efficiency cores included) are allowed only in the scaling report, labelled as such.
  `mpirun` always runs under an external `timeout` and with `--mca btl_tcp_if_include lo0
  --mca oob_tcp_if_include lo0` (root `CLAUDE.md`). Open MPI 5 uses its shared-memory
  transport between ranks on one node; the TCP flags only stop it from hanging on
  interface selection.
- **Device ranks.** One device per rank, opened by the local rank index (a
  `split_shared` communicator). On the CPU runtime the cores are shared: ranks × units
  per cube stay within the cores (device-path.md §11), and the `threads(n)` cap applies
  per rank. Metal runs with several ranks share the GPU and run by hand, outside the
  sandbox, `#[ignore]`d.
- **Threads and BLAS.** The Phase 3 rule stands: ranks × rayon threads × BLAS threads
  (× CubeCL CPU-runtime workers) stay within the physical cores. With threads > 1, MPI
  is initialised at `Threading::Funneled`, and every MPI call, the non-blocking ones
  included, is made on the calling thread.
- **Non-blocking communication.** rsmpi 0.8.2 has no neighbourhood collectives (rlst's
  blocking `MPI_Neighbor_alltoallv` wraps them with its own `unsafe`). It has scoped
  point-to-point requests (`request::scope`, `immediate_send`, `immediate_receive_into`)
  and non-blocking dense collectives (`immediate_all_to_all_varcount_into`,
  `immediate_all_gather_varcount_into`). `GhostCommunicator` exposes the neighbour ranks,
  counts and offsets (`out_ranks`, `in_ranks`, `send_counts`, `send_offsets`, …), but
  its value counts are private. T1 chooses among:
  - scoped point-to-point over the ghost communicator's neighbours;
  - an addition to rlst (`MPI_Ineighbor_alltoallv` behind a safe API), which is an
    upstream change and needs asking;
  - anything better it finds.

  A scoped request cannot outlive the function that opened its scope. An overlapped
  stage is therefore one evaluator method that posts, works and waits, not a "start"
  and a "finish" method. T1 also states how messages progress while the rank computes
  (MPI makes progress only inside MPI calls unless the library runs a progress thread),
  for example with `test` calls between level calls.
- **Accumulation order.** fmm-plan-redesign §7.5 stays the order unless decision 9
  changes it. An order-preserving overlap exists:
  - the source exchange behind `upward_local` (P2M and the local M2M read only local
    sources);
  - the multipole exchange of the finer levels behind the downward pass of the coarser
    ones (level l needs the ghost multipoles of l and l + 1 only).

  Running P2P before L2P, or splitting rows into local and ghost parts, changes the order
  (fmm-plan-redesign §10). Either needs decision 9.
- **Tests and MPI.** The root rules apply. In addition:
  - `tests/mpi_exec.rs` keeps its one MPI-initialising test. Its scenarios now run on
    every rank count, and each one checks the multi-rank output against the one-rank
    output. Keep the debug run within its one-minute budget at one rank; the CI job
    (T3) states its own budget at 2 and 4 ranks.
  - Large multi-rank gates are `#[ignore]` release tests in their own executable (one
    MPI test each), or `nd-fmm-validate` examples, run by hand at 2, 4 and 8 ranks.
  - A test that needs a given rank count says so and skips cleanly otherwise. It must
    not fail or hang at 1 rank, or at any rank count the CI job uses.
- **Examples.** Multi-rank examples in `nd-fmm-validate` and `nd-fmm-exec` that run at
  any rank count are registered with templated-examples (`command = "mpirun -n
  {{NPROCESSES}}"`), so the weekly job runs them at 3 ranks. Timing examples are not
  registered.
- **Errors.** `build` keeps its one agreement all-reduce of input errors (step 1) and
  adds no collective for errors where an existing one can carry them. A new collective
  for agreement, for example after a mid-evaluation device sync, is allowed if T1
  justifies it, and is counted in the collectives table of the design.
- **Dependencies.** No new external dependency. rlst stays at 0.9.0 and mpi at 0.8.2. An
  rlst change (a non-blocking neighbourhood exchange, or public value counts) is asked
  for first, and goes upstream with a new pinned release, not into a fork.

## Exit gate
- Every acceptance test in the task briefs passes:
  - in CI for the default members, on one rank;
  - in the multi-rank CI job at 2 and 4 ranks, if it is kept;
  - on the M3 Max at 1, 2, 4 and 8 ranks by hand, reported with the ranks, threads and
    backends run.
- `docs/design/distributed-fmm.md` is signed off before T4 starts, including the
  `nd-octree` changes, the tolerance of requirement 2 and the overlap order.
- `IndexFmm` and its tests, examples and registrations are gone (T2). The scenario set
  of `tests/mpi_regressions.rs` does not shrink, and every scenario still passes its
  list oracle, exchange checks and recording-operator checks on 1, 2 and 4 ranks.
- C5.1, host (T4–T6):
  - `Fmm` builds and evaluates on any rank count for points in any distribution,
    including all on one rank and one empty rank;
  - the output equals the one-rank output within requirement 2 on 2, 4 and 8 ranks on
    every `tests/mpi_exec.rs` scenario, and on the cube and Plummer sphere at N = 10⁵ at
    the workload points;
  - the errors against the direct sum are within 0.1% (f64) and 1% (f32) of the one-rank
    run's;
  - the output is bit-identical across evaluations and thread counts on a fixed rank
    count.
- C5.1, device (T7, T8):
  - the hook covers every host-side write and read of the evaluator. A shadow operator
    that learns of host data only through the hook equals the plain operator bit for bit
    on 1, 2 and 4 ranks, and with the hook disabled it differs on 2 ranks;
  - the device on 2 and 4 ranks (CPU runtime; Metal by hand) is within Phase 4's FMM
    bounds of the host on the same ranks, and bit for bit with every kind on the host
    fallback;
  - its transfers per evaluation equal the design's formula;
  - `DeviceNeedsOneRank` is gone.
- C5.2 (T8, T9): exchanges overlap local work as the design specifies, with the output
  as decision 9 fixes, and the communication is hidden for the benchmark case by the
  measure T1 defines. Device ghost buffers are packed and unpacked on the device.
- C5.3 (T10): the scaling report (strong and weak, 1–12 ranks, host path, stage by
  stage, with load imbalance and communication) is in `fmm-validate/results/` and the
  design documents. It states that every figure is one node of the M3 Max, and gives the
  inter-node run as a documented command.

## Tasks

One pull request each.
- T1, T2 and T3 touch disjoint files and can start at once.
- T4 needs T1 signed off. T5 needs T1 signed off and T2 (both change `nd-fmm-plan`'s
  tests).
- T6 needs T5, and T4 if T4 changes the `Octree::new` signature or options that `build`
  uses.
- T7 needs T6, because its shadow check runs `Fmm` on several ranks.
- T8 needs T7.
- T9 needs T6 and T7 (both change the evaluator; merge T7 first). Its device part, if T1
  keeps one, needs T8.
- T10 needs T6 and T9, and T8 for its correctness table.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-distributed-design.md](T1-distributed-design.md) | `docs/design/distributed-fmm.md`: redistribution, `nd-octree` changes, multi-rank tolerance, the data-movement hook, device ranks, overlap, collectives, memory per rank, tests, scaling method, task check | design for C5.1–C5.3 | Phase 4 complete |
| T2 | [T2-remove-index-fmm.md](T2-remove-index-fmm.md) | `nd-fmm-plan` without `IndexFmm`: code, tests, examples, registration and docs removed; the recording operator and the exchange checks keep the coverage | retirement of the Phase 3 test FMM | none |
| T3 | [T3-multi-rank-ci.md](T3-multi-rank-ci.md) | a CI job that runs the MPI test executables at 2 and 4 ranks under a timeout; measured; root `CLAUDE.md` checks updated | test infrastructure for C5.1–C5.2 | none |
| T4 | [T4-octree-adaptations.md](T4-octree-adaptations.md) | the `nd-octree` changes T1 specifies (load balance by weight, partition granularity, errors instead of panics), with MPI examples | part of C5.1 | T1 (signed off) |
| T5 | [T5-redistribution.md](T5-redistribution.md) | `nd_fmm_plan::redistribute::Redistribution`: points to owning ranks and results back, collective, deterministic | part of C5.1 | T1 (signed off), T2 |
| T6 | [T6-multi-rank-fmm.md](T6-multi-rank-fmm.md) | `Fmm` on any number of ranks through `Redistribution`; multi-rank against one-rank checks in every scenario; the C5.1 host gate at N = 10⁵ | C5.1 (host) | T5; T4 if it changes the octree API |
| T7 | [T7-host-data-hook.md](T7-host-data-hook.md) | `FmmOperator::host_data`, a kernel-agnostic hook around the evaluator's data movements; the shadow-operator check | part of C5.1 (device) | T6 |
| T8 | [T8-device-multi-rank.md](T8-device-multi-rank.md) | the device operator on several ranks: packed exchange buffers on the device, device per rank, errors agreed; `DeviceNeedsOneRank` removed | C5.1 (device), C5.2 (device-resident ghost buffers) | T7 |
| T9 | [T9-overlap.md](T9-overlap.md) | non-blocking exchanges and overlapped stages in `nd-fmm-plan`; `Fmm` option; communication-hidden measurement | C5.2 | T6, T7; T8 for a device part |
| T10 | [T10-scaling.md](T10-scaling.md) | strong and weak scaling on the M3 Max, load balance and communication by stage, the inter-node command, design-document update | C5.3; gate: scaling report | T6, T8, T9 |

Review T1 yourself before the tasks that build on it. T4–T9 encode its decisions on the
octree, the redistribution, the hook and the overlap order.

T3 changes `.github/workflows/`, and T4–T6 may change `Cargo.lock` if a manifest changes.
T2, T5, T7 and T9 all change `nd-fmm-plan`: merge one, then rebase the next. T6, T8 and
T9 all change `fmm-exec/src/fmm.rs` and `tests/mpi_exec.rs`.

## Machines

| Machine | Ranks | Used for |
| --- | --- | --- |
| Apple M3 Max (development; 12 performance and 4 efficiency cores, 64 GB, Open MPI 5.0.10) | 1–12 on the performance cores; 16 only labelled in the scaling report | every task's multi-rank tests by hand; every timing; Metal with ranks sharing the GPU |
| GitHub Actions `ubuntu-latest` (4 vCPUs), if T3's job is kept | 2 and 4 | the MPI test executables, debug, correctness only |
| A cluster | none available (decided on 2026-10-04) | the inter-node run, documented as one command for later |

## Decisions to sign off

Each is recorded in the exit checklist when made:
1. The design document `docs/design/distributed-fmm.md`, including any change it proposes
   to the requirements or tolerances above (T1; before T4).
2. Hardware. **Decided on 2026-10-04: the M3 Max only.** Every multi-rank run is one
   node. The inter-node scaling run is a documented command.
3. The device on several ranks. **Decided on 2026-10-04: in Phase 5, correctness only**
   (T7, T8). It is never timed as a scaling figure.
4. A multi-rank CI job. **Decided on 2026-10-04: add one.** T3 builds and measures it,
   and it is kept, changed or dropped at sign-off with T3's numbers (time per job, flakes
   over repeated runs).
5. The index FMM. **Decided on 2026-10-04: removed from `nd-fmm-plan`** (T2). Nothing in
   Phase 5 is checked with it.
6. `nd-octree` changes. **Decided on 2026-10-04: allowed where the distributed FMM needs
   them.** T1 lists each change with its reason, and the sign-off accepts or rejects each
   one.
7. The tolerance of requirement 2 (multi-rank against one rank), with T1's derivation,
   and whether a rank-count-independent order within leaves is worth providing (T1).
8. The redistribution API as refined by T1, and whether `build` keeps a cheap path for
   callers whose points are already on their owners (T1).
9. The overlap order (T1). The default recommendation is the order-preserving overlap of
   "Accumulation order", bit-identical to the blocking path. A reordering (P2P before
   L2P, local before ghost row parts) is opt-in at most, with its order documented, and
   only if T1's model shows the order-preserving overlap leaves communication exposed.
10. The non-blocking mechanism: scoped rsmpi point-to-point, or an rlst addition (T1;
    an rlst addition needs asking).
11. The C5.2 criterion "communication hidden", made measurable (T1). For example: the
    exposed wait of every exchange in an overlapped evaluation at most 10% of the
    blocking exchange time, at 8 ranks on the benchmark case.
12. Device errors on several ranks: how an error at a mid-evaluation sync is agreed
    without a hang, and whether that costs a collective per evaluation (T1).

## Risks

| Risk | Mitigation |
| --- | --- |
| A multi-rank defect hangs a collective, in a test or in CI | every `mpirun` under an external `timeout`; errors agreed before the next collective (requirement 4); T3's job has a step timeout; scenarios fail on every rank or on none |
| One node over shared memory hides communication costs that a network would show, so overlap looks unnecessary or scaling looks better than it is | T10 reports bytes and messages per exchange next to the times, so a network's cost can be estimated; T1 models the exchanges at network bandwidths; the inter-node command is documented |
| Load imbalance from coarse-block granularity or from weighting distinct keys instead of points (`nd-octree` counts each finest key once, so coincident points weigh as one) | T1 measures the imbalance on the workloads before T4; T4 adds the weighting the sign-off accepts; T10 reports per-rank work |
| Small problems on many ranks: `nd-octree` panics if the coarse tree has fewer blocks than ranks | T1 decides between an error agreed on every rank and a refinement that makes enough blocks; T4 implements it; tests cover a tiny problem on 8 ranks |
| The multi-rank result differs from one rank by more than rounding, and the cause is a ghost or global-level defect hidden behind the tolerance | the tolerance is derived, not tuned (T1); breakdown by kind and by list on failure; the recording operator checks every list pair once on every rank count; bit identity on a fixed rank count |
| Retiring `IndexFmm` leaves the evaluator's values unchecked in `nd-fmm-plan` | the Laplace FMM against one rank and the direct sum in `nd-fmm-exec` (T6) is the value check; the recording operator and the exchange checks stay in `nd-fmm-plan`; T2 lists every check it removes and what replaces it |
| Non-blocking exchanges do not progress while the rank computes, so the overlap gains nothing | T1 designs the progress (test calls between level calls), and T9 measures the exposed wait per exchange |
| Several ranks share the one GPU on the M3 Max and time-slice it | device multi-rank runs are correctness checks only (decision 3); transfers are counted, not timed |
| macOS schedules ranks on efficiency cores without binding, so timings vary | at most 12 ranks × threads; medians over repeated runs; every report prints the placement settings and labels 16-rank runs |
| The O(P) replicated data (coarse tree, coarse gather) dominates at large P | T1 states the growth with P (requirement 8); irrelevant at P ≤ 12, recorded for a cluster run |

## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase5/T<k>-<name>.md and do that task." Review and merge before the next.

## Exit checklist
- [ ] T1 merged: `docs/design/distributed-fmm.md` drafted, sign-off questions listed
- [ ] Distributed design signed off, including the `nd-octree` changes, the tolerance of requirement 2, the redistribution API, the overlap order and the non-blocking mechanism
- [x] Hardware: the M3 Max only; inter-node run documented for later (decided 2026-10-04)
- [x] Device on several ranks: in Phase 5, correctness only (decided 2026-10-04)
- [x] Multi-rank CI job: add one, measured in T3 (decided 2026-10-04)
- [x] Index FMM: removed from `nd-fmm-plan` (decided 2026-10-04)
- [x] `nd-octree` changes: allowed where the distributed FMM needs them, each signed off with T1 (decided 2026-10-04)
- [ ] T2 merged: `IndexFmm` removed; the scenario set unchanged; list, exchange and recording-operator checks pass on 1, 2 and 4 ranks
- [ ] T3 merged: multi-rank CI job measured
- [ ] Multi-rank CI job: kept / changed / dropped
- [ ] T4 merged: `nd-octree` changes as signed off; MPI examples pass on 1 and 3 ranks
- [ ] T5 merged: `Redistribution` round-trips on 1, 2 and 4 ranks for every input distribution
- [ ] T6 merged: `Fmm` on any rank count; equal to one rank within requirement 2 on 2, 4 and 8 ranks; C5.1 host gate
- [ ] T7 merged: `host_data` hook; shadow operator bit for bit on 1, 2 and 4 ranks, and differs with the hook disabled
- [ ] T8 merged: device on 2 and 4 ranks within the FMM bounds of the host; transfers as the formula; `DeviceNeedsOneRank` removed
- [ ] T9 merged: overlapped exchanges, output as decision 9 fixes; communication hidden by the T1 measure
- [ ] T10 merged: scaling report published
- [ ] Design documents updated: laplace-fmm-plan §5, §7 (Phase 5 status and numbers), §8.1, §8.3, §9.1 and §9.2; fmm-plan-redesign §9 and §10 outcome notes; device-path.md §14; workspace-structure §3, §4 and §6; distributed-fmm.md decisions and measurements recorded

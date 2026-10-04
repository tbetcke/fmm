# Phase 5 / T1 — design of the distributed FMM (design for C5.1–C5.3)

The parts of a distributed Laplace FMM exist separately:
- `nd-octree` partitions the tree;
- `nd-fmm-plan`'s `Evaluator` exchanges ghost sources and multipoles and replicates the
  coarse levels;
- fmm-plan-redesign §9 designs a `Redistribution` of points;
- device-path.md §4.5 sketches a hook for the device on several ranks;
- fmm-plan-redesign §10 notes what overlap would need.

What is missing is a design that ties them together and settles the decisions that
change code:
- where points go, and how the octree balances them;
- what "equal to one rank" means numerically;
- how a device operator sees the evaluator's data movements;
- how exchanges overlap computation without breaking determinism;
- how the result is tested and measured on the one machine available.

This task writes that design as `docs/design/distributed-fmm.md`, for sign-off by hand
before T4. It is the Phase 5 counterpart of Phase 3 T1 and Phase 4 T1. It writes no
Rust, apart from throwaway measurements in the scratch directory, which are not
committed.

The machine is the Apple M3 Max, and multi-rank means one node over shared memory
(docs/phase5/README.md, decision 2). `IndexFmm` is retired (decision 5), so do not use
it as an oracle or a check anywhere in the design. `nd-octree` may change where the FMM
needs it (decision 6).

Read first:
- root CLAUDE.md, octree/CLAUDE.md, fmm-plan/CLAUDE.md, fmm-exec/CLAUDE.md,
  fmm-kernels/CLAUDE.md, fmm-validate/CLAUDE.md;
- docs/phase5/README.md, closely: "Requirements on the distributed FMM" is what this
  design answers; also "Design decisions", "Exit gate" and "Decisions to sign off";
- docs/design/fmm-plan-redesign.md §1.3, §3.3, §3.5, §7 (with §7.3 and §7.5), §8, §9 and
  §10;
- docs/design/device-path.md §4 (all of it), §8, §11, §12, §14 and §17;
- docs/design/laplace-fmm-plan.md §5, §6.8, §7 (the Phase 4 "Recommendation for Phase 5"
  and the Phase 5 table), §8 and §9;
- the code:
  - `nd_octree`: `Octree::new`, `OctreeOptions`, `owner_rank`, `local_leaf`,
    `lookup_leaves`, `coarse_tree_bounds`, `points_to_morton`,
    `compute_global_bounding_box`; in the private `octree/implementation.rs`, the
    coarse tree (`compute_coarse_tree`), the weights
    (`compute_coarse_tree_weights`, which count linearized, so distinct, keys),
    `load_balance` and `partition_blocks` (the panic with fewer blocks than ranks),
    the redistribution of keys, and `generate_all_keys` (what every rank receives);
  - `nd_fmm_plan::{plan, index, evaluator, exchange, operator, store}`: the stages, the
    three exchanges and their accessors, `CoarseExchange` (no `own_blocks()` yet), the
    leaf numbering (local, then ghost);
  - rlst 0.9.0's `distributed_tools::ghost_communicator` (from the registry, not
    `../rlst`): `build`, the public neighbour data (`out_ranks`, `in_ranks`,
    `send_counts`, `send_offsets`, …), the private value counts, `forward_comm`, the
    blocking `neighbor_alltoallv`;
  - rsmpi 0.8.2: `request::scope`, `immediate_send`, `immediate_receive_into`, the
    `immediate_*` collectives, `SimpleCommunicator::self_comm`, `split_shared`;
  - `nd_fmm_exec::fmm`: `build` steps 1–8 (step 5 is `PointsNotOwned`, then
    `DeviceNeedsOneRank`), `Fmm::evaluate`, `StageTimings`, `BuildTimings`;
  - `nd_fmm_exec::device`: `DeviceOperator::new` (the ghost-free assertion),
    `begin_evaluation`, `read_output`, the transfer accounting;
  - `fmm-exec/tests/mpi_exec.rs` (`cases`, `share`, `built`, the `ownership` and `device
    backends` scenarios) and `fmm-plan/tests/mpi_regressions.rs` (the scenarios, the
    recording operator).

Measure before you design where a decision rests on a number. The tree and plan
statistics below need no FMM evaluation, only `Octree::new` and `Plan::new` on 2, 4 and
8 ranks under `mpirun`: write a throwaway example in the scratch directory, or run an
existing one with printing. Report each number with how it was obtained:
- the load imbalance of the current partition on the workloads (the cube, the Plummer
  sphere and the Gaussian clusters at N = 10⁵ and 10⁶; and a set with many coincident
  points): leaves, points and near-field pairs per rank, max over mean;
- coarse blocks per rank;
- ghost leaves and ghost boxes per rank and level, and the bytes of each exchange at p = 8
  (f32 and f64);
- the size of the replicated data: coarse blocks, `Global` boxes, the coarse gather per
  evaluation;
- the time of `Octree::new`, `Plan::new` and `Evaluator::new` on 1–8 ranks (release),
  the latter including the rlst graph creation per exchange.

Write `docs/design/distributed-fmm.md` with these sections:

1. **Starting point.**
   - What exists and what each crate does on several ranks today, with file and function
     names: partition, ghosts, coarse tree, the three exchanges, the collectives of
     `Plan::new`, `Evaluator::new` and an evaluation (fmm-plan-redesign §7.3, checked
     against the code), and `Fmm::build`'s steps.
   - The measurements above.
   - What `IndexFmm` checked, and where each of those checks lives once T2 has removed it.
2. **Requirements.** Restate requirements 1–10 of the README and say how the design meets
   each. If one should change, say why and propose the change as a sign-off question.
3. **Partition and `nd-octree`.**
   - How the octree weighs and partitions today, and what that costs on the workloads
     (the measured imbalance, coarse-block granularity, coincident points counted once,
     the panic with fewer blocks than ranks, the one-rank empty-input case).
   - The changes you propose, each with its reason, its API (Rust signatures with doc
     comments), its collectives and its tests. Candidates, to accept or reject with
     reasons:
     - weights per finest key, either point multiplicities (duplicates counted) or a
       caller-supplied cost such as sources and targets weighted separately;
     - a finer coarse partition, for example a minimum number of blocks per rank, so
       that the balance is not limited by a few heavy blocks;
     - an error agreed on every rank, or an automatic refinement, instead of the panic
       when the coarse tree has fewer blocks than ranks;
     - anything `Redistribution` needs that the octree does not offer (a batched
       `owner_rank`, say).
   - What stays as it is, and why. Keep to the octree's contract: it stores topology,
     never points (octree/CLAUDE.md).
   - The weight the FMM should pass: a model of per-leaf work (near-field pairs, leaf
     operators, far-field boxes), checked against the measured per-rank work, kept
     simple enough to compute from keys and counts.
4. **Redistribution (C5.1).**
   - fmm-plan-redesign §9 as the starting point: `Redistribution::new`, `counts`,
     `origins`, `forward`, `backward`. Confirm or refine the API (Rust signatures), its
     errors, its collectives (counts, the `i32` overflow agreement, the
     all-to-all-v), memory and traffic.
   - The order of received items within a leaf, and whether a rank-count-independent
     order is possible and worth it. For example, order by finest key and then by the f64
     coordinates. Weigh what it buys (multipoles and P2P sums independent of the
     distribution) against what still differs (local before ghost in the near and X
     rows).
   - How `Fmm` uses it: one `Redistribution` for sources and one for targets inside
     `build`; f64 coordinates forwarded and converted to leaf-scaled values on the
     owner; charges forwarded and output returned in `evaluate`. Give the collectives
     per build and per evaluation, and the new `StageTimings` and `BuildTimings`
     fields.
   - A cheap path when every point is already on its owner, if the all-to-all of counts
     is measurably worth skipping. It must be agreed on every rank.
   - What replaces `PointsNotOwned` in the error enum and docs.
5. **Equal to one rank (requirement 2).**
   - Every source of difference between N ranks and one rank: point order within a leaf,
     local before ghost entries in the near and X rows, the coarse gather and the global
     pass, and anything else you find in the code.
   - For each, whether it changes bits, and a bound on the difference it can cause.
   - The tolerance you propose for φ and ∇φ in f32 and f64, derived from those bounds and
     checked against a measured difference. Compute the N-rank results with the
     existing host path by hand: give each rank the points of its own leaves, as the
     `device backends` scenario does. Compare against the README's provisional 1e-12
     and 1e-5.
   - How tests build the one-rank reference (README, "The oracles"), and its cost in the
     debug budget.
6. **The data-movement hook (C5.1, device).**
   - device-path.md §4.5 as the starting point: `HostData`, `FmmOperator::host_data`
     with a no-op default. Re-derive every host-side read and write of the evaluator
     outside operator calls on several ranks from the code (device-path.md §4.2 did it
     for one rank and listed the several-rank column). Confirm the events, their
     arguments, the index lists each needs, and `CoarseExchange::own_blocks()`.
   - Show that the host path stays bit-identical: the default does nothing, and the
     evaluator's own movements do not change.
   - **The check without `IndexFmm`:** a shadow operator in `nd-fmm-exec`'s tests wraps
     `LaplaceOperator`. It keeps its own copies of the five stores, runs each batch on its
     copies, and learns of host data only through `host_data`. It writes the values to
     send into the host store at the "send" events. It must equal the plain
     `LaplaceOperator` bit for bit on 1, 2 and 4 ranks, and differ on 2 ranks with the
     hook disabled. Say what the batch types need for that: public constructors, or the
     fields already public. Say where the shadow lives.
   - How overlap (section 8) fires the events when an exchange is split across a stage.
7. **The device on several ranks.**
   - What `DeviceOperator` does at each event:
     - the source exchange reads the host source chunks, which already hold the
       coordinates and this evaluation's charges, so nothing is downloaded; the ghost
       tail is uploaded after it;
     - the multipoles and coarse blocks to send are gathered into a packed buffer on the
       device and downloaded;
     - the received ones are uploaded packed and scattered on the device.

     Check this against the code and correct it where it is wrong.
   - The transfers and syncs per evaluation, as a formula in the exchange sizes, beside
     Phase 4's one-rank minimum. Which syncs are unavoidable: at least the downloads of
     the coarse blocks and of the multipoles to send.
   - Which `nd-fmm-kernels` launches it uses (the existing `movement` gather and scatter,
     or new ones), and the device memory added for the ghost slots and packed buffers.
   - The device per rank (`split_shared`, local rank modulo device count), the CPU
     runtime's cores per rank, and Metal with ranks sharing the GPU.
   - Errors (requirement 4): a device failure at a sync in the middle of an evaluation
     must not leave other ranks blocked in the next exchange. Weigh agreeing it at the end
     of `evaluate` (one all-reduce per evaluation) against carrying a flag in an
     existing exchange, and recommend one.
   - "Device-resident ghost buffers" (C5.2): what remains of it after the above, and
     whether anything more is worth doing on unified memory.
8. **Overlap (C5.2).**
   - The order-preserving overlap of the README ("Accumulation order"): the source
     exchange behind `upward_local`, and the multipole exchange of level l + 1 and finer
     behind the downward pass of level l. Which evaluator methods change. How the public
     stages and their debug-checked order change, or which new combined stage they gain.
   - Its expected gain, as a model from the measured exchange sizes and the Phase 3S
     and Phase 4 stage times, at shared-memory bandwidth and at a stated network
     bandwidth (for example 10 and 100 Gbit/s). Mark it *model*. Say whether the
     multipole exchange can be hidden at all: the coarse levels hold little downward
     work.
   - The reordering options (P2P before L2P; local before ghost parts of near and X rows;
     P2P into its own buffer added at the end) and what each does to the accumulation
     order of fmm-plan-redesign §7.5. Recommend whether any is offered, and as what
     (opt-in, documented order), for decision 9.
   - The non-blocking mechanism (decision 10). Compare:
     - scoped rsmpi point-to-point over the ghost communicator's neighbours, with the
       value counts rebuilt from `send_counts` and `send_offsets`;
     - an rlst addition (`MPI_Ineighbor_alltoallv` behind a safe API);
     - anything better.

     Cover safety (no `unsafe` here), the lifetime of scoped requests (one method posts,
     works and waits), message tags and ordering, the rank's own messages, empty
     neighbours, and progress (test calls between level calls, and how often).
   - The coarse gather: whether `immediate_all_gather_varcount_into` can hide it behind
     the last levels of `upward_local`, or why not.
   - The criterion "communication hidden" made measurable (decision 11), and how
     `StageTimings` reports exposed wait against total exchange time.
9. **Collectives and memory.**
   - Every collective of `Octree::new`, `Plan::new`, `Redistribution`, `Evaluator::new`,
     `Fmm::build` and an evaluation, in order, with what ranks with no points do. This is
     the table of fmm-plan-redesign §7.3 extended.
   - Memory per rank as a formula in N/P, the ghost layer and P (requirement 8), with
     what is replicated and how it grows. State the P at which the replicated part
     matters, as a model.
10. **Testing.**
    - The test layers per task:
      - `nd-octree` examples at 1 and 3 ranks;
      - `mpi_regressions` (lists, exchanges, recording operator) at 1, 2 and 4;
      - redistribution round trips;
      - the multi-rank `Fmm` against one rank in every `mpi_exec` scenario;
      - the shadow check;
      - the device on several ranks;
      - overlap against blocking;
      - the ignored gates.
    - The input distributions (README, "Workloads") and how each scenario derives them.
    - The CI job's place (T3) and budgets: debug at 1 rank under a minute, and at 2 and
      4 ranks as T3 measures.
    - How a test skips a rank count it cannot run.
11. **Scaling method (C5.3).**
    - Strong and weak scaling on the M3 Max: problems, ranks (1, 2, 4, 8, 12), ranks ×
      threads mixes at 12 cores, precisions and p, repetitions, statistics.
    - What is reported per stage and per rank: max, min and mean, load imbalance,
      exchange bytes and messages, exposed wait, build time by part, memory.
    - How shared memory and the missing process binding on macOS bias the numbers, and
      what the report says about them.
    - The inter-node command, and the numbers it would settle.
12. **Task check.** Map the design onto T2–T10. Name each task's modules and tests. Say
    where a brief needs changing, and propose the change; do not rewrite the briefs.
13. **Questions for sign-off.** At least decisions 1 and 6–12 of the README, each with a
    recommendation:
    1. the requirements and tolerances, with any change proposed;
    2. each `nd-octree` change;
    3. the tolerance of requirement 2, and the order within leaves;
    4. the redistribution API and the cheap path;
    5. the hook's events and the shadow check;
    6. device errors on several ranks;
    7. the overlap order, and any opt-in reordering;
    8. the non-blocking mechanism;
    9. the "communication hidden" criterion;
    10. whether the device gets an overlap part in T9, or none in Phase 5.

Also update root CLAUDE.md, "Source of truth", to point to the Phase 5 briefs
(docs/phase5/README.md) and to `docs/design/distributed-fmm.md` as the current phase's
design. Keep the Phase 4 and earlier pointers.

Keep the document in the style of the other design documents:
- short sections, with tables where they help;
- Rust signatures where they decide something, and no code beyond sketches;
- every number that is a model or an estimate marked as such, and every measured number
  with its source (rank count, build, command).

The PR description must contain:
- a one-page summary of the design;
- the sign-off questions with a recommendation for each;
- the measured partition statistics;
- the list of evaluator-side host reads and writes on several ranks found in section 6,
  with file and function names.

Must pass: nothing builds differently. Run `cargo fmt -- --check`,
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan` and `RUST_MIN_STACK=8388608 cargo test
-p nd-fmm-exec` once, to confirm the starting point, and report the results. Run every
multi-rank measurement under an external timeout with the macOS loopback flags.

Do not:
- write or change any Rust in the workspace, or any brief except to fix a factual error
  (say which);
- commit the measurement code or its output;
- use `IndexFmm` in the design, its checks or its measurements;
- propose an `nd-octree` change without a measured or stated reason, or a new external
  dependency.

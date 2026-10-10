# Phase 5S: scale-out on Kathleen

Phase 5S makes the distributed FMM of Phase 5 run on thousands of MPI ranks across the
nodes of a cluster. It builds the packages of docs/design/distributed-fmm.md §14 (S1, S2,
S7 and S3; S4 and S5 only if measured to be needed). It measures them on Kathleen, UCL's
Omni-Path cluster, at 1, 2 and 4 nodes (40, 80 and 160 cores; **no job is larger than 4
nodes, ever**, decided on 2026-10-10). Beyond 160 ranks it measures correctness, memory,
message counts and bytes on oversubscribed runs (locust to 512 ranks, Kathleen's 4 nodes
to 640–1,280 ranks), never timed, and gives times there as *models*, the way
distributed-fmm.md §14.5 ("A cluster may not be available") describes.

Phase 5 ran on one node only (the M3 Max at up to 12 ranks, locust at up to 72, 256 and
512 oversubscribed). Its structure is correct at any rank count: the host output on P
ranks is bit for bit the one-rank `Fmm` over the union of the points in rank order. But
it replicates the top of the tree on every rank, and what is replicated grows with P:
- `nd-octree`'s `generate_all_keys` (octree/src/octree/implementation.rs) gathers every
  rank's coarse-block ancestors to every rank (`gather_to_all`), marks them `Global`, and
  with the ghost-children layer advertises every coarse block to all ranks and every key
  next to a `Global` key with its children;
- `nd-fmm-plan`'s `CoarseExchange` gathers every coarse block's multipole to every rank
  in every evaluation (`all_gather_varcount_into`), and every rank runs the global M2M
  over every `Global` box (`upward_global`);
- P2 (Phase 5 T5) already removed the downward work on `Global` boxes that no own block
  descends from, without changing a value.

What this costs, measured (Phase 5 T10; the N = 10⁶ cube, f64, p = 8; locust;
`fmm-validate/results/phase5-gh200.md` §9 and §10):

| Ranks | held boxes not local | held boxes per rank (max) | coarse gather received per rank and evaluation | coefficient memory per rank (*model*) |
| --- | --- | --- | --- | --- |
| 8 | 37% | 7,417 | 0.19 MB | 9.6 MB |
| 72 | 92% | 6,673 | 2.6 MB | 8.6 MB |
| 256 (oversubscribed) | 97.3% | 5,737 | 2.64 MB | 7.4 MB |
| 512 (oversubscribed) | 99.8% | **37,449, every box of the one-rank tree** | **11.6 MB** | 48.5 MB, the one-rank figure |

At 512 ranks, with about 2,000 points per rank, the replicated top tree has taken over
completely. distributed-fmm.md §14.2 models the same growth at cluster scale (cube,
N = 10⁸, p = 8, f64, *model*): at 8,192 ranks the global M2M on every rank is about
300,000 M2M, three times a rank's useful work; the coarse gather is about 170 MB per rank
per evaluation; the replicated held boxes are about 1.2 million per rank, 1.5 GB of
coefficient slots; and `Plan::new` builds lists over all of them. That does not run on a
node with 4.8 GB per core.

What is missing:
- **A locally essential top tree (S1).** Each rank should hold its own boxes and their
  halo, the ancestors of its own blocks, and the boxes their V, W and X rows name, as
  ghosts of their owners, with the partition only as replicated *keys* (blocks and
  owners, about 2 MB at 8,192 ranks).
- **An owned top tree (S2).** Every `Global` box gets an owner; the children's owners send
  their exact multipoles to the parent's owner, which adds them by octant, so every
  multipole keeps its one-rank bits. A fixed-size top (L_r) is replicated by one
  fixed-size all-gather; below it the multipole exchange carries what a rank needs. The
  coarse gather goes.
- **Validation at scale (S7).** No one-rank reference exists at N = 10⁹. Rank-count
  invariance (P ranks against P/k ranks holding concatenated inputs, bit for bit) and the
  distributed direct-sum sample become the oracles.
- **A work-weighted cut (S3).** At 72 ranks the points stay within 1.07 of the mean, but
  the M2L work reaches 1.17 (cube), 1.39 (Plummer) and 1.29 (clusters) times the mean,
  and the compute stages follow (1.17–1.26; T10). distributed-fmm.md §3.6 predicted it
  within 0.12 and grows with P.
- **Construction and collectives at scale (S4, S5)**, if measured to need it: two sample
  sorts per `Octree::new` and the O(P) count all-to-alls of `Redistribution` and of rlst's
  ghost-communicator builds (32 kB per rank at 8,192 ranks, *model*).
- **Inter-node measurements of the new structures.** Phase 5N (T2) runs the Phase 5 code
  on Kathleen at up to 4 nodes, if Kathleen is usable (Phase 5N decision 0). Nothing will
  run timed at 1,000 or more ranks: per-rank structures and traffic beyond 160 ranks come
  from oversubscribed runs, and times there from models whose inputs (per-rank compute,
  counted messages and bytes, an assumed latency and bandwidth, the measured imbalance)
  are stated.

Hardware: **Kathleen is the primary machine of this phase if Phase 5N decision 0 found it
usable**, with locust and the M3 Max for development, oversubscribed correctness and
regressions (see "Without Kathleen" below for the other case):
- Kathleen (`ssh kathleen`, Slurm; tools/kathleen/ from Phase 5N T1): 190 nodes, each two
  Intel Xeon Gold 6248 (Cascade Lake, 20 cores each, AVX-512, 2.5 GHz), 40 cores per node
  in two NUMA domains (27.5 MB L3 each), 188 GB, Intel Omni-Path; diskless nodes, no
  local `$TMPDIR`; RHEL 9.6; no GPUs. Whole nodes per job. **At most 4 nodes per job**:
  1–2 nodes in the `test` QoS (1 h), 4 nodes in `small` (48 h, at most 6 nodes); the
  larger QoS are not used. The queue is busy (about 2,000 jobs pending on 2026-10-10);
  Phase 5N T1's probe measured the wait for 2- and 4-node jobs.
  Oversubscribed runs on the 4 nodes (for example 640–1,280 ranks, `--oversubscribe`)
  serve correctness, memory and counts only, never timings.
- locust (tools/gh200/): 72 Neoverse-V2 cores, one NUMA node; up to 72 ranks timed, up to
  512 oversubscribed for correctness and memory (the open-file limit raised, as
  `tools/scaling/run.sh` does since Phase 5 T10).
- the M3 Max: development and multi-rank tests by hand up to 12 ranks.

The phase has four parts:
1. **Design and oracles.** A design document for the scale-out, measured on Kathleen with
   the Phase 5 code (T1, signed off by hand), then validation at scale (S7) before any
   structural change, so that every later task is checked by rank-count invariance (T2).
2. **A locally essential top tree (S1)** in `nd-octree` (T3) and `nd-fmm-plan` (T4).
3. **An owned top tree (S2)** in `nd-fmm-plan` (T5) and `nd-fmm-exec` (T6), and the
   **work-weighted cut (S3)** (T7).
4. **Construction and collectives (S4, S5), conditional (T8), and scaling at scale (T9)**,
   with the design-document update and the revision of the Phase 6 briefs.

Companion documents:
- [docs/design/distributed-fmm.md](../design/distributed-fmm.md): §3.6 (the replicated
  global pass, P2, the cut by work), §9 (collectives, memory per rank), §11 (scaling
  method), **§14 (all of it: what grows with P, the packages S1–S7, the staging)** and §15
  (Phase 5's outcome);
- [docs/design/fmm-plan-redesign.md](../design/fmm-plan-redesign.md): §3.3–§3.5 (the box
  index, the held keys, decision 6: every coarse block held), §7 (the pass order and the
  collectives of §7.3) and §8;
- [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md): §5.4, §7 (Phase 5 and
  its "Recommendation for Phase 6", Phase 5N), §8 and §9;
- [docs/design/device-path.md](../design/device-path.md): §14 (the device on several
  ranks: the hook's events);
- `fmm-validate/results/phase5-m3max.md`, `phase5-gh200.md` and Phase 5N's
  `phase5n-kathleen.md`.

T1 adds a further companion, `docs/design/scale-out.md`.

Prerequisite: **Phase 5N is complete**: its decision 0 ("Kathleen usable?") is made; if
Kathleen is used, its environment is reproducible from `tools/kathleen/`, every existing
check passes there and the Kathleen baseline report is published; and the node-level M2L
rule is built. The briefs assume the workspace as it is after Phase 5N's last merge.

## Without Kathleen

If Phase 5N decision 0 is "without Kathleen", or the queue makes Kathleen unusable during
this phase, the phase proceeds on the M3 Max and locust and is accepted as
distributed-fmm.md §14.5 ("A cluster may not be available") describes:
- **Measured on one node**: correctness and bit identity at every rank count (locust to
  72 ranks timed, to 512 oversubscribed; the M3 Max to 12); the per-rank structures,
  memory, message counts and bytes at those counts, which do not depend on where the
  ranks run; the per-rank compute at the per-rank sizes of a large run (N/P = 10⁴–10⁶, on
  one rank or a few).
- **Estimated**: times beyond one node, from a model that adds the measured per-rank
  compute, the counted messages and bytes at a stated latency and bandwidth (for example
  1–2 µs and 10–100 Gbit/s) and the measured imbalance. Every such figure is *model*.
- **Acceptance**: the per-rank quantities (held boxes, top-tree traffic, replicated work)
  bounded in P at fixed N/P on oversubscribed runs, and rank-count invariance up to the
  largest P run; timing criteria replaced by the model, to be checked on Kathleen's 1–4
  nodes when the queue allows (each brief says which runs those are).
- **Open**: network effects a model misses (contention, many small messages, collective
  algorithms); the report lists them.

Every brief's Kathleen runs are then skipped, and the task states it; the briefs mark
them "(Kathleen)".

## Scope

In scope:
- `docs/design/scale-out.md`, signed off; the design-document updates at the end of the
  phase.
- `nd-octree` (octree/CLAUDE.md applies):
  - the replicated partition map as keys and the locally essential key set per rank,
    replacing the all-gathers of `generate_all_keys` and the ghost-children layer's
    replication of every block (S1, T3);
  - the second cut by modelled work after the local refinement (S3, T7);
  - keys routed to block owners instead of the sample sorts (S4, T8), only if T8's trigger
    fires and the sign-off accepts it.
- `nd-fmm-plan`:
  - `Plan::new` over the locally essential set, the held-key check replacing "every coarse
    block held" (S1, T4);
  - the owned top tree: owners, the order-preserving upward reduction, the replicated top
    of fixed size, the coarse gather removed, new `HostData` events, overlap of the
    reduction (S2, T5);
  - a sparse dynamic exchange for the O(P) count all-to-alls (S5, T8), conditional.
- `nd-fmm-exec`: `Fmm` on the new plan and exchanges, the device operator's handling of the
  new events, overlap, `StageTimings` and traffic fields, the work weight's cost model
  (T6, T7).
- `nd-fmm-validate`: the rank-count-invariance check, the `scaling` harness at N ≥ 10⁸
  without a one-rank reference, the per-rank measures of the new structures (T2, T6, T9).
- `tools/kathleen/` and `tools/scaling/`: correctness and scaling jobs at scale (T2, T9).

Out of scope:
- Phase 6 operator work: the host batched M2L, SVD compression, AVX-512 P2P, several
  charge vectors, dipoles, plane-wave M2L, FMM3D (docs/phase6/). The node-level M2L rule
  of Phase 5N stays as built.
- GPUs on the cluster (Kathleen has none). The device on several ranks stays a
  correctness check on locust and the M3 Max (Phase 5 decision 3); multi-GPU device
  indices (S6's second half) wait for GPU nodes.
- Dynamic load balancing between evaluations, and rebuilding the tree while points move.
  A new point set means a new `Fmm`.
- Threading inside `nd-octree` or `nd-fmm-plan`; worker threads never call MPI.
- Changes to `nd-fmm-math`, `nd-fmm-ref`, `nd-fmm-tables`, `nd-fmm-simd` and
  `nd-fmm-kernels` beyond fixing a defect a test exposes (stop and report first).
- Changes to one-rank results, or to the host path's defaults beyond Phase 5N's rule.

## Requirements on the scale-out

T1 designs against these, and T2–T9 are accepted against them. T1 may propose changing
one, with reasons, for sign-off. Phase 5's requirements 1–10 (docs/phase5/README.md) stand
unless restated here.

1. **Same interface.** `FmmBuilder::build(sources, targets, comm)` and
   `Fmm::evaluate(charges)` keep their signatures and semantics (Phase 5 requirement 1).
   New options are additive and off or neutral by default where they change bits.
2. **Bit for bit, at every rank count.** The host output on P ranks stays bit for bit the
   one-rank `Fmm` over the union of the points in rank order, wherever a one-rank run is
   possible (Phase 5 decision 7), and two input distributions stay within 100 u_T. Every
   package keeps this: S1 and S3 change only what is held and where, S2's reduction adds
   children by octant in the one-rank order. A package that cannot keep it stops and
   reports.
3. **Rank-count invariance at scale.** A run on P ranks equals a run on P/k ranks bit for
   bit, when rank r of the small run holds the concatenation of the inputs of ranks
   k r … k r + k − 1 of the large one (distributed-fmm.md §14.4, S7). This is the oracle
   wherever no one-rank run fits.
4. **Bounded per-rank cost at fixed N/P.** After S1 and S2, the held boxes, the top-tree
   traffic, the replicated work and the memory per rank stay bounded as P grows at fixed
   N/P (weak scaling), apart from the replicated partition map (keys and owners, O(P), a
   few MB at 10⁴ ranks) and the fixed-size replicated top. T1 states every remaining
   O(P) item with its size at 10², 10³ and 10⁴ ranks, as a model, and T4, T6 and T9
   measure it: timed to 160 ranks on Kathleen, and in counts and bytes on oversubscribed
   runs beyond (locust to 512, Kathleen's 4 nodes to 1,280).
5. **No per-evaluation cost that grows with P.** Every per-evaluation pattern after S2 is
   a neighbourhood exchange or a fixed-size all-gather (§14.4, S5). The coarse gather is
   gone.
6. **Collective discipline.** Phase 5 requirement 4: every rank enters every collective in
   the same order, empty ranks included; every input-dependent error is agreed before the
   next collective; worker threads never call MPI; a multi-rank test fails on every rank
   or on none, and runs under an external timeout (on Kathleen: the job's time limit and
   a `timeout` inside it).
7. **Kernel-agnostic and safe.** S1, S2 and S5 live in `nd-octree` and `nd-fmm-plan` and
   know nothing of Laplace; they are tested there with operators without kernel
   arithmetic (the recording operator, the shadow operator). No `unsafe` outside the
   crates the root CLAUDE.md allows; non-blocking and sparse exchanges use safe rsmpi
   calls (scoped requests, `immediate_*`), never raw MPI FFI.
8. **The hook, the device and overlap still work.** New data movements (the reduction's
   sends and receives, the replicated top) are `HostData` events, and the device operator
   mirrors them (correctness only, on locust and the M3 Max). Overlap stays
   order-preserving and bit for bit the blocking path.
9. **Tested on what can run.** Every multi-rank test runs at 1, 2 and 4 ranks in CI
   (`run-tests-mpi`), and by hand at 1, 2, 4 and 8 ranks on the M3 Max and on locust, and
   on Kathleen at 2 nodes (80 ranks, and 2 × 2 ranks × 20 threads) in the `test` QoS.
   Larger counts run where a brief asks for them: Kathleen at 4 nodes (160 ranks, timed),
   oversubscribed runs (locust to 512 ranks, Kathleen's 4 nodes to 1,280), never more
   than 4 nodes. Every report states the machine, nodes, ranks per node, threads per
   rank, the binding, the MPI library and the job id.
10. **Measured, never asserted.** Timings in release builds, every BLAS thread variable
    at 1, ranks × threads per node within the physical cores (40 on Kathleen, 72 on
    locust, 12 on the M3 Max). No timing in a test or in CI. Oversubscribed runs are
    never timed. A time beyond 160 ranks is a *model*, with its inputs stated.

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **No new crate.** The packages live in `nd-octree` and `nd-fmm-plan`, the Laplace side in
  `nd-fmm-exec`, the measurements in `nd-fmm-validate` and `tools/`.
- **The oracles.**
  - Up to N ≈ 10⁷: the one-rank `Fmm` over the union in rank order, bit for bit, as in
    Phase 5 (built on `self_comm` in tests, on rank 0 in the `scaling` harness).
  - At any scale: rank-count invariance (requirement 3), and the errors against the
    direct sum from the distributed direct-sum sample (each rank sums its own sources at
    the sampled targets, one all-reduce; the `scaling` harness has it since Phase 5 T10)
    within 0.1% (f64) and 1% (f32) of the errors at the smaller rank count.
  - In `nd-fmm-plan`: the list oracle, the exchange checks and the recording operator of
    `tests/mpi_regressions.rs`, extended to the new structures.
- **Workloads.** The Phase 5 workloads (the cube, the Plummer sphere with a = 0.1, the
  Gaussian clusters; `max_level` 16, 64 points per leaf; f64 p = 3 and 8, f32 p = 8),
  every point and charge generated by index (`nd_fmm_validate::scaling::Workload`), so no
  rank ever holds the global set: N = 10⁶–10⁸ (10⁸ on 4 Kathleen nodes; oversubscribed
  runs at N = 10⁶–10⁷ to fit locust's and Kathleen's memory). Weak scaling at 10⁶ points
  per socket (two ranks × 20 threads per node) and 2.5 × 10⁴ points per rank at 40 ranks
  per node; per-rank sizes of a large run (N/P = 10⁴–10⁶) also timed on one node, as the
  model's input.
- **Ranks on Kathleen.**
  - Configurations: 40 ranks × 1 thread per node (flat MPI), and 2 ranks × 20 threads per
    node (one rank per socket, distributed-fmm.md §14.4, S6), with the strategy that
    Phase 5N's rule picks. Others only where a brief asks.
  - Launch with the launcher Phase 5N T1 chose (`srun` or `mpirun` inside the
    allocation), cores bound, the binding printed, every BLAS thread variable at 1.
  - Builds and runs go in jobs, never on the login nodes (UCL's rule: login nodes only
    for short, light tests). **No job larger than 4 nodes, ever.** Correctness at 1–2
    nodes in the `test` QoS; 4-node runs in `small`. Oversubscribed runs on the 4 nodes
    (`--oversubscribe`, at most 1,280 ranks) for correctness, memory and counts only.
  - **Node-hours.** Each report states the node-hours it used. Budget per task: T1 and T9
    at most 60 node-hours each, the others at most 20, unless the sign-off raises it. The
    total is decision 6.
  - Storage: the 250 GB quota covers home and Scratch together and is hard: output goes to
    Scratch, raw outputs are summarised and deleted after the report, and no task writes
    per-rank files at thousands of ranks.
- **Ranks on locust.** As in Phase 5 (docs/phase5/README.md, "Ranks on locust"): the load
  checked and stated, `--report-bindings`; oversubscribed runs (`--map-by :OVERSUBSCRIBE
  --bind-to none`, the open-file limit raised) for correctness and memory up to 512 ranks,
  never timed.
- **Tests and MPI.** The root rules apply. `run-tests-mpi` stays at 2 and 4 ranks; a new
  multi-rank test runs there within the job's 15-minute budget, or is an ignored gate or
  an example run by hand. A test that needs a rank count says so and skips cleanly
  otherwise. One MPI-initialising test per test executable.
- **Errors.** No new collective for agreement where an existing one can carry the error;
  a new one is justified in T1 and counted in the collectives table.
- **Dependencies.** No new external dependency. rlst 0.9.0 and mpi 0.8.2 stay pinned.
  rsmpi 0.8.2 already has what a sparse exchange needs (`immediate_synchronous_send`,
  `immediate_barrier`, scoped requests). An rlst or rsmpi addition (for example a
  neighbourhood collective, or replacing rlst's ghost-communicator build) is asked for
  first and goes upstream with a pinned release, never into a fork.
- **What stays.** O1–O4 (Phase 5 T4), P1 and P2 (T5), the redistribution (T5, T6), the
  hook's six events (T7) and the overlap order (T9) stay, extended rather than replaced.

## Exit gate
- Every acceptance test in the task briefs passes: in CI for the default members on one
  rank; in `run-tests-mpi` at 2 and 4 ranks; by hand at 1, 2, 4 and 8 ranks on the M3 Max
  and on locust; (Kathleen) at 2 nodes.
- `docs/design/scale-out.md` is signed off before T3 starts.
- C5S.1 (T2): the rank-count-invariance check passes at 1, 2 and 4 ranks in CI, at 512
  against 64 on locust (oversubscribed) and, on Kathleen, at 160 against 20 (and 1,280
  against 160 oversubscribed); the `scaling` harness runs at N = 10⁸ without a one-rank
  reference.
- C5S.2 (T3, T4): held boxes per rank independent of P at fixed N/P within T1's bound,
  measured from 2 to 512 ranks on locust (oversubscribed above 72) and, on Kathleen, from
  40 to 160 timed and to 1,280 oversubscribed; bit for bit Phase 5's results at 1–72
  ranks.
- C5S.3 (T5, T6): no per-rank evaluation cost that grows with P (requirement 5); the coarse
  gather removed; bit for bit; the device on several ranks still within Phase 4's bounds
  and its transfers as the formula; overlap bit for bit the blocking path.
- C5S.4 (T7): the modelled per-rank cost (per-rank counts times the fitted constants)
  within T1's target at 512–1,280 oversubscribed ranks (distributed-fmm.md §14.5: ≤ 1.10
  at 1,024 ranks), and the measured compute-stage max/mean at 72 (locust) and 160
  (Kathleen) ranks reported against the first cut; bit for bit.
- C5S.5 (T8): S4 and S5 evaluated on oversubscribed counts, bytes and models, and built
  or closed as the sign-off decides; their timing at scale stays open.
- C5S.6 (T9): the scaling report (strong and weak at 1, 2 and 4 Kathleen nodes and 1–72
  locust ranks, Phase 5 against Phase 5S, stage by stage; memory, counts and bytes per
  rank to 512–1,280 oversubscribed ranks; *model* times beyond 160 ranks; rank-count
  invariance and the direct-sum checks) in `fmm-validate/results/phase5s-kathleen.md` and
  the design documents; the Phase 6 briefs revised.

## Tasks

One pull request each.
- T1 needs Phase 5N complete. T2 needs T1 signed off.
- T3 needs T1 signed off and T2 (its acceptance runs T2's invariance check).
- T4 needs T3. T5 needs T4. T6 needs T5.
- T7 needs T4 (the held set and the plan it re-cuts), and T6 for timing the cost model's
  constants on the final structure.
- T8 needs T6 and T7 (it measures the final construction and collectives).
- T9 needs T6, T7 and T8.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-scale-out-design.md](T1-scale-out-design.md) | `docs/design/scale-out.md`: S1, S2, S3, S7, the S4/S5 triggers, collectives and memory per rank at 10²–10⁴ ranks, testing and scaling method, the time model beyond 160 ranks, task check, sign-off questions; measured with the Phase 5 code on Kathleen at 1–4 nodes and in oversubscribed counts (locust to 512, Kathleen to 1,280) | design for C5S.1–C5S.6 | Phase 5N complete |
| T2 | [T2-validation-at-scale.md](T2-validation-at-scale.md) | S7: the rank-count-invariance check (a test within 4 ranks, an example at P against P/8), the `scaling` harness at N ≥ 10⁸ without a one-rank reference, a Kathleen correctness job | C5S.1 | T1 (signed off) |
| T3 | [T3-octree-partition-map.md](T3-octree-partition-map.md) | S1 in `nd-octree`: the partition map as keys, local owner and kind of any top-tree key, the locally essential key set replacing `generate_all_keys`'s all-gathers by sparse advertisements | C5S.2 | T1 (signed off), T2 |
| T4 | [T4-plan-locally-essential.md](T4-plan-locally-essential.md) | S1 in `nd-fmm-plan`: `Plan::new` over the locally essential set, the held-key check, the exchanges built from it; held boxes measured bounded in P | C5S.2 | T3 |
| T5 | [T5-owned-top-tree.md](T5-owned-top-tree.md) | S2 in `nd-fmm-plan`: owners, the order-preserving upward reduction, the replicated top of fixed size, the coarse gather removed, new `HostData` events, overlap of the reduction | C5S.3 | T4 |
| T6 | [T6-owned-top-tree-exec.md](T6-owned-top-tree-exec.md) | S2 in `nd-fmm-exec`: `Fmm`, the device operator's events, overlap, timings and traffic, the harness's new measures | C5S.3 | T5 |
| T7 | [T7-work-weighted-cut.md](T7-work-weighted-cut.md) | S3: the cost model from per-kind timings, the second cut after the local refinement, an `Fmm` option; the modelled cost balance at 512–1,280 oversubscribed ranks, the measured balance at 72 and 160 | C5S.4 | T4; T6 for timing |
| T8 | [T8-construction-and-collectives.md](T8-construction-and-collectives.md) | S4 and S5, conditional: evaluated on oversubscribed counts and bytes (to 512–1,280 ranks) and models; built only where T1's trigger fires and the sign-off accepts, else closed by a report | C5S.5 | T6, T7 |
| T9 | [T9-scaling-at-scale.md](T9-scaling-at-scale.md) | scaling on Kathleen at 1, 2 and 4 nodes and on locust to 72 ranks, Phase 5 against Phase 5S, counts and memory to 512–1,280 oversubscribed ranks, *model* times beyond, design-document update, Phase 6 briefs revised | C5S.6; gate: scaling report | T6, T7, T8 |

Review T1 yourself before the tasks that build on it: T3–T8 encode its decisions on the
held set, the owners, L_r and the cost model.

T3 and T7 both change `nd-octree`; T4, T5 and T8 change `nd-fmm-plan`; T6 and T7 change
`fmm-exec/src/fmm.rs` and `tests/mpi_exec.rs`. Merge one, then rebase the next. T2 and T9
change `nd-fmm-validate`'s `scaling` module and `tools/`.

## Machines

| Machine | Ranks | Used for |
| --- | --- | --- |
| Kathleen (190 nodes × 40 cores, 2 sockets, 188 GB, Omni-Path; Slurm; `ssh kathleen`, tools/kathleen/), if Phase 5N decision 0 found it usable | 2 nodes routinely (`test` QoS), 4 nodes at most (`small`), timed to 160 ranks; oversubscribed on the 4 nodes to 1,280 ranks, never timed | every task's multi-node tests, the measurements of T1, T4, T6–T9, the scaling report |
| locust (GH200, 72 Neoverse-V2 cores, one H100; shared, no scheduler; tools/gh200/) | 1–72 timed; up to 512 oversubscribed, never timed | multi-rank tests by hand at 1, 2, 4, 8; held boxes and memory beyond 72 ranks; the device on several ranks (CUDA, correctness only) |
| Apple M3 Max (12 performance cores) | 1–12 | development; multi-rank tests by hand at 1, 2, 4, 8; Metal device ranks (correctness) |
| GitHub Actions `ubuntu-latest` (`run-tests-mpi`) | 2 and 4 | the MPI test executables, debug, correctness only |

## Decisions to sign off

Each is recorded in the exit checklist when made:
1. The design document `docs/design/scale-out.md`, including any change it proposes to
   the requirements above (T1; before T3).
2. The replicated top: its fixed size L_r (a few thousand boxes, chosen once, independent
   of P) and how it is formed and replicated (T1).
3. The work weight of S3: the cost model, where its constants come from (per-kind timings
   per machine, precision and p), the balance target (of the modelled cost at 512–1,280
   oversubscribed ranks, and of the measured compute stages at 72 and 160), and whether the
   second cut is the default on several ranks (bits do not change; only the partition
   does) (T1, T7).
4. The triggers of S4 and S5, stated on what can be measured: counts, bytes and memory at
   512–1,280 oversubscribed ranks and times to 160 ranks, with a model beyond (for example
   the sample sorts' message volume or a count all-to-all's modelled time at 10⁴ ranks
   above a stated share of the build); whether either needs an rsmpi addition (T1, T8).
   Their timing at scale stays open.
5. CI: which new checks run at 2 and 4 ranks in `run-tests-mpi`, and what runs only by
   hand on Kathleen and locust (T1, T2).
6. The node-hour budget per task and for the phase, within the standing limit of 4 nodes
   per job, and T9's job plan (T1; T9).
7. The Phase 6 briefs as revised by T9 (T9).

## Risks

| Risk | Mitigation |
| --- | --- |
| A defect appears only at hundreds of ranks or across nodes, and the cluster queue makes each attempt slow | T2's invariance check and the oversubscribed locust runs (to 512 ranks, no queue) before any Kathleen run; correctness at 2 nodes in the `test` QoS first; each structural task runs the full Phase 5 suites |
| No timing exists beyond 160 ranks (the 4-node limit), so effects that appear only at thousands of ranks across many nodes (network contention, many small messages, collective algorithms at scale) stay unmeasured | per-rank counts and bytes measured to 1,280 oversubscribed ranks; times beyond 160 ranks as a stated *model* (distributed-fmm.md §14.5); the open effects listed in T9's report; the documented command for a larger cluster run kept |
| A multi-node run hangs and burns its allocation | every launch under `timeout` inside the job and a job time limit sized from a smaller run; errors agreed before the next collective (requirement 6); tests fail on every rank or on none |
| S2's reduction changes bits (the order of additions depends on the rank count or the MPI library) | children added by octant on the parent's owner, never through an `MPI_Reduce` or all-reduce of partial sums (§14.4); bit for bit against one rank in every scenario, and rank-count invariance |
| The locally essential set misses a key that a list names | the plan's held-key check (every listed key held) replaces "every coarse block held", fails on every rank, and is tested with the list oracle on graded trees, coincident points and one empty rank |
| The replicated top L_r is chosen badly: too small leaves many senders per needed multipole near the root, too large replicates more than needed | T1 measures the needed top-tree multipoles and their owners per rank on oversubscribed runs to 512–1,280 ranks (with the Phase 5 plan's lists) and §14.2's simulation beyond, and proposes L_r for sign-off; T9 reports the traffic split |
| The work-weighted cut moves points and changes nothing measurable, or unbalances P2P | the cost model is fitted to per-kind timings and checked against measured per-rank stage times before it becomes the default; the target is stated (≤ 1.10) and the point balance reported |
| Kathleen's busy queue delays even 2- and 4-node runs | Phase 5N T1's probe and decision 0; many configurations packed into each job; the "Without Kathleen" path: acceptance on locust's oversubscribed runs and models, with Kathleen's runs added when the queue allows |
| The 250 GB quota fills with build trees and outputs, and jobs fail | one target directory per branch in Scratch, deleted after merge; summarised outputs only; no per-rank files |
| Omni-Path and Open MPI 4.1 behave differently from the one-node runs (progress, eager limits, PSM2 settings) | Phase 5N T1 and T2 record the transport and its settings; T1 measures the exchanges at network bandwidth; overlap's value there is reported, not assumed |
| `Plan::new`, rlst's graph communicators or the octree's sample sorts become the bottleneck at thousands of ranks | T1 times each to 160 ranks and counts their messages and bytes to 1,280 oversubscribed ranks; T8 evaluates the triggers on those counts and models and builds S4/S5 only where they fire |
| The scale-out changes the structures the device operator and overlap rely on | new movements are `HostData` events (requirement 8); the shadow check and the device gate run at the end of T5 and T6 |

## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase5s/T<k>-<name>.md and do that task." Review and merge before the next.

## Exit checklist
- [ ] T1 merged: `docs/design/scale-out.md` drafted, sign-off questions listed
- [ ] Scale-out design signed off, including L_r, the work weight, the S4/S5 triggers, CI and the node-hour budget (decisions 1–6)
- [ ] T2 merged: rank-count invariance at 1, 2 and 4 ranks in CI, at 512 against 64 on locust, and at 160 against 20 on Kathleen; the harness at N = 10⁸
- [ ] T3 merged: the partition map as keys and the locally essential key set in `nd-octree`; MPI examples pass on 1, 3 and 8 ranks
- [ ] T4 merged: `Plan::new` over the locally essential set; bit for bit; held boxes bounded in P to 512 (locust) and 1,280 (Kathleen) oversubscribed ranks
- [ ] T5 merged: the owned top tree in `nd-fmm-plan`; the coarse gather removed; bit for bit
- [ ] T6 merged: `Fmm`, device events and overlap on the owned top tree; bit for bit; device gate passes
- [ ] T7 merged: the work-weighted cut; modelled cost max/mean within the target at 512–1,280 oversubscribed ranks; measured imbalance at 72 and 160 ranks
- [ ] T8 merged: S4 and S5 built or closed by counts, bytes and models
- [ ] T9 merged: scaling report published; Phase 6 briefs revised (decision 7)
- [ ] Design documents updated: laplace-fmm-plan §7 (Phase 5S status and numbers), §8, §9; distributed-fmm.md §14/§15 notes; `scale-out.md` outcome section; workspace-structure §3, §3.1, §4 and §6

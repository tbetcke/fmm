# Phase 5S / T9 — scaling, counts at scale and models, design-document update (C5S.6; gate: scaling report)

This task measures what the phase delivered and ends the phase with the design-document
update and the revision of the Phase 6 briefs, as Phase 5 T10 and Phase 4S T8 did. No job
is larger than 4 nodes (docs/phase5s/README.md), so the report has three kinds of number,
each labelled:
- **timed across nodes** on Kathleen at 1, 2 and 4 nodes (40, 80 and 160 cores), and
  **timed on one node** on locust (to 72 ranks) and the M3 Max (to 12);
- **counted, never timed**, on oversubscribed runs: memory, held boxes, messages and bytes
  per rank on locust to 512 ranks and on Kathleen's 4 nodes to 640;
- **modelled** times beyond 160 ranks, from the measured per-rank compute, the counted
  messages and bytes at a stated latency and bandwidth, and the measured imbalance
  (distributed-fmm.md §14.5, "A cluster may not be available"; `scale-out.md` §10).

It covers strong and weak scaling stage by stage, Phase 5 against Phase 5S; memory, held
boxes and top-tree traffic per rank against P; load balance with and without the
work-weighted cut; overlap at network bandwidth; the correctness checks (rank-count
invariance and the direct-sum sample); regressions on locust and the M3 Max.

If Phase 5N decision 0 was "without Kathleen" and the queue still does not allow 4-node
jobs, the Kathleen runs are skipped and the report says so; the phase is then accepted on
locust's runs and the models (README, "Without Kathleen"), and the Kathleen runs below are
listed as the ones to add when the queue allows.

**Added on 2026-10-14 (T8s, at the user's request): the exposed collectives.** T8s built
S8 (the forwarding tree) and an analysis of its measurements
(`fmm-validate/results/phase5s-t8s.md` §7, §9) found that at the measured scale the
communication is 1–4 % of an evaluation, that the overlapped path hides the source and
multipole exchanges (0.00–0.02 ms exposed at 160 Kathleen ranks), and that the one
per-evaluation term left that grows with P is the replicated top's all-gather-v
(`TopExchange::gather`): 1.7–7 ms (f64 p = 8) and 0.3–0.7 ms (f32 p = 3) at 160 ranks on
the fastest rank, on the critical path of both paths. If Open MPI 4.1.6 picks a P/2-step
algorithm for R's size (inferred from its fixed decision rule, not measured), it takes
10–17 ms per evaluation at 10⁴ ranks (*model*), 5–8 times `scale-out.md` §10.4's
log₂ P model, and crosses 1 % of an evaluation at f32 p = 3 from a few hundred ranks,
before S8 did. T9 therefore also does the following (the steps "The exposed collectives"
below), and the "Do not" on library changes has the exceptions named there.

Read first:
- root CLAUDE.md, fmm-validate/CLAUDE.md, tools/kathleen/README.md;
- docs/phase5s/README.md ("Requirements" 4, 5, 9, 10; "Ranks on Kathleen", node-hours;
  "Without Kathleen"; "Exit gate"; decisions 6 and 7);
- `docs/design/scale-out.md` §10 (the scaling method, the time model, the job plan) and
  every decision as signed off;
- distributed-fmm.md §11, §14 and §15; laplace-fmm-plan.md §7 (Phase 5, Phase 5N);
- `fmm-validate/results/phase5-m3max.md`, `phase5-gh200.md` and `phase5n-kathleen.md`
  (the format, and the numbers to compare against; its part "Phase 5S, T0: four nodes" is
  the Phase 5 code at 1, 2 and 4 Kathleen nodes);
- `nd_fmm_validate::scaling` and `tools/scaling/` (the Kathleen driver of Phase 5N T2);
- docs/phase6/README.md and its briefs (to revise).

Do:
- **Timed runs on Kathleen** (release; BLAS variables at 1; cores bound and the binding
  printed; each configuration's job id, nodes, ranks per node and threads per rank
  recorded; at most 4 nodes per job, many configurations packed into each job):
  - **Strong scaling**: the cube and the Plummer sphere at N = 10⁷ and 10⁸, f64 p = 3 and
    8, f32 p = 8, at 1, 2 and 4 nodes; 40 ranks × 1 thread and 2 × 20 per node; the
    strategy Phase 5N's rule picks; overlap off and on; the second cut on and off (T7).
  - **Weak scaling**: 2.5 × 10⁴ points per rank (40 × 1) and 10⁶ per socket (2 × 20), at
    1, 2 and 4 nodes.
  - **Phase 5 against Phase 5S**: the Phase 5 code (its merge commit, built in its own
    directory) on the same configurations, to show what S1 and S2 changed (held boxes,
    the gather, `upward_global`, evaluation time). T0's runs are that side at 1, 2 and 4
    nodes where the configuration matches; rerun only the rest, and a few of T0's as
    repeats.
  - **The model's inputs**: the per-rank compute at the per-rank sizes of large runs
    (N/P = 10⁴–10⁶) on one node, and the Omni-Path latency and bandwidth seen by the
    exchanges at 2 and 4 nodes.
- **Timed runs on locust and the M3 Max**: Phase 5 T10's strong sweeps (N = 10⁶, 1–72
  ranks on locust, 1–12 on the M3 Max) on the Phase 5S code, as regression checks
  against `phase5-gh200.md` and `phase5-m3max.md`, with the load stated on locust.
- **Counted runs, never timed**: oversubscribed on locust at 128, 256 and 512 ranks and
  (Kathleen) on the 4 nodes at 320 and 640 ranks, on the cube and the Plummer
  sphere at fixed N (10⁶, 10⁷) and at fixed N/P where memory allows: held boxes by kind,
  the reduction's and the multipole exchange's messages and bytes, the replicated top's
  bytes, the memory model and the peak resident size, the modelled per-rank cost's
  balance; against Phase 5's numbers (at 512 ranks: 37,449 held boxes and an 11.6 MB
  gather per rank).
- **Modelled times** to 10³ and 10⁴ ranks for the strong and weak configurations, with
  their inputs and method stated, each figure marked *model*, and the effects a model
  misses listed as open.
- **Correctness**: T2's invariance check at 512 against 64 on locust and (Kathleen) at 160
  against 20 and 640 against 80 oversubscribed; the direct-sum errors at every timed
  configuration against the smallest run (f64 to the printed digits, f32 within 1%),
  stating which runs were checked.
- **Node-hours**: state the node-hours per sweep and in total (budget for this task: at
  most 60 node-hours unless the sign-off raises it; the exposed collectives' measurements
  and checks, about 3–5 of them, come out of it).
- **The report**: `fmm-validate/results/phase5s-kathleen.md` (or `phase5s.md` without
  Kathleen), in the format of the Phase 5 reports. It states which figures are timed
  across nodes (Kathleen, 1–4 nodes, Omni-Path, the MPI library and settings named),
  which are one node, which are counted on oversubscribed runs, and which are models.
  Paste the summary into the PR.
- **Design documents**, with the Phase 5S outcome:
  - `docs/design/scale-out.md`: an outcome section (decisions as taken, measured numbers
    against its models), as distributed-fmm.md §15 does;
  - distributed-fmm.md: a short note at §14 and §15 pointing to the outcome;
  - laplace-fmm-plan.md: the revision note; §7, Phase 5S (status per component
    C5S.1–C5S.6, the scaling tables, a "Recommendation for Phase 6"); §8.1 (the
    "Distributed" and "Topology" rows: invariance and the new structures); §8.3; §9.1
    (the Phase 5S risks, with what was found) and §9.2 (questions answered and open, the
    timing beyond 160 ranks among them);
  - fmm-plan-redesign.md: a short note in §3.5 (decision 6 replaced by the held-key
    check) and §7.3 (the collectives);
  - device-path.md §14: the new events and transfers;
  - workspace-structure.md §3, §3.1 (the new surfaces of `nd-octree`, `nd-fmm-plan`,
    `nd-fmm-exec`, `nd-fmm-validate`), §4 and §6 (Phase 5S done);
  - docs/phase5s/README.md: tick the exit checklist.
- **Revise the Phase 6 briefs** (docs/phase6/README.md and its tasks) with what this phase
  measured: the per-node and per-rank costs, the strategy and configuration that scaled,
  and any task whose premise changed. Record the revision as decision 7.

- **The exposed collectives** (added on 2026-10-14 from T8s; the analysis is summarised in
  `fmm-validate/results/phase5s-t8s.md` §9):
  1. **Measure first** (one 4-node job, about 0.5 node-hours, before the scaling sweeps):
     - `MPI_Allgatherv` at R's sizes (0.26 MB and 2.65 MB in all, R at the cap in f32
       p = 3 and f64 p = 8) at 40, 80 and 160 ranks with 10, 20 and 40 ranks per node,
       with Open MPI's default and each forced algorithm
       (`OMPI_MCA_coll_tuned_use_dynamic_rules=1`,
       `OMPI_MCA_coll_tuned_allgatherv_algorithm`), fitted as a·P + bytes/β; refit
       `scale-out.md` §10.2's all-gather term and §10.4 with it;
     - the overlapped path's wait per level (`OverlapTimes::level_waits`, reported) and its
       per-message cost (`progress`, posting) at 160 ranks timed and 640 counted, to see
       whether the forwarding tree's forwards (posted between level calls) lag a level
       at depth 3;
     - the blocking tree's receive copy (`copy_into` in `TreeExchange::forward`), timed.
  2. **Decide** (sign-off, before building): the all-gather's model and remedy; whether
     overlap becomes the default; whether the blocking path keeps a default anywhere.
  3. **Build** what the sign-off accepts, bit for bit, safe rsmpi only, no new event, no
     collective in a branch:
     - **a two-level gather of R** in `TopExchange`: the node communicator from
       `Fmm::build`'s `split_shared` (passed in as the tree's node is) and a leaders'
       communicator (the node's first rank); per evaluation the leaders all-gather-v R,
       then each node broadcasts the gathered buffer from its leader; the same `gathered`
       layout, slots and `ReceivedTop` (*model* at 10⁴: 1.5–2 ms at f64 p = 8, about
       0.4 ms at f32 p = 3); `TopExchange` keeps the one-communicator gather as the
       oracle, as `MultipoleRouting::Direct` is the tree's;
     - **overlap on by default** (`FmmBuilder::overlap`), `overlap(false)` kept as the
       oracle, if the sweeps at N = 10⁷ and 10⁸ confirm it (measured at 160 ranks: the
       exchanges' 0.4–1.5 % of an evaluation hidden, bit for bit);
     - only if the blocking path keeps a default: the tree's blocking path posting every
       level at once and scattering from its staging buffer (the measured +0.3–0.6 ms
       against the direct exchange at 160 ranks; T8s §7.3).
  4. **Test**: `check_top` in `mpi_regressions` against the one-communicator gather, with
     made-up nodes of two ranks at 4–8 ranks; the Phase 5S suites, the shadow and device
     checks (CPU runtime, Metal, CUDA), T2's correctness job at 2 nodes and invariance at
     160 and 640 (as T8s did); the collectives in order in `scale-out.md` §8.1 and the
     `exchange` module docs.
  5. **Measure** before and after in the sweeps above (both paths, both precisions, the
     three workloads at 40–160 ranks timed, 320–640 counted), and the S8 trigger at
     f32 p = 3 (left fired by T8s, `scale-out.md` §7.3) on the overlapped path.
  - **Record, not build** (for the Phase 6 revision or their own briefs): pacing dense
    bursts in rlst's sparse exchanges (the 4-node packet losses, T8 §9.3; upstream); the
    octree's O1 and weight-round all-reduces (the largest O(P) build term left, *model*
    about 57 ms per build at 10⁴ ranks); `ReductionExchange::new` on two sparse
    exchanges as the tree's build; level aggregation, not as specified (it would expose
    the overlapped path's per-level waits; T8s §2.2); the imbalance at f32 p = 3
    (26–42 % of wall time at 160 ranks: the cost model underweights P2P there); and an
    explanation of the cube's leaf time at 80 Kathleen ranks (1,068 ms at f64 p = 8 against
    88 ms at 40 and 72 ms at 160 ranks for 1.15 times the P2P count; T8s job 604104).

Tests that define done:
- The scaling report is published with every run's configuration, job id and the
  node-hours used, and every number labelled timed, counted or *model*.
- Rank-count invariance: 0 differing values at every pair run; the direct-sum errors as
  stated.
- The Phase 5 regressions on locust and the M3 Max within run-to-run noise of their Phase 5
  values, or the difference explained.
- The smoke tests of the harness and the Phase 5S suites still pass.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
  `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate` at 1 rank and its smoke test at
  2 ranks;
- (Kathleen) the correctness job at 2 nodes.

Do not:
- submit a job larger than 4 nodes;
- time an oversubscribed run, or present a model as a measurement;
- assert timings, or commit CSV or raw outputs beyond the report;
- change the library to improve a number (a defect found is reported and fixed in its own
  commit with its test), except the exposed collectives' changes above, once signed off;
- run builds or measurements on Kathleen's login nodes, or exceed the node-hour budget
  without asking.

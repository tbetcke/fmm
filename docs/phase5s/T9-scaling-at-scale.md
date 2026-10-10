# Phase 5S / T9 — scaling, counts at scale and models, design-document update (C5S.6; gate: scaling report)

This task measures what the phase delivered and ends the phase with the design-document
update and the revision of the Phase 6 briefs, as Phase 5 T10 and Phase 4S T8 did. No job
is larger than 4 nodes (docs/phase5s/README.md), so the report has three kinds of number,
each labelled:
- **timed across nodes** on Kathleen at 1, 2 and 4 nodes (40, 80 and 160 cores), and
  **timed on one node** on locust (to 72 ranks) and the M3 Max (to 12);
- **counted, never timed**, on oversubscribed runs: memory, held boxes, messages and bytes
  per rank on locust to 512 ranks and on Kathleen's 4 nodes to 1,280;
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

Read first:
- root CLAUDE.md, fmm-validate/CLAUDE.md, tools/kathleen/README.md;
- docs/phase5s/README.md ("Requirements" 4, 5, 9, 10; "Ranks on Kathleen", node-hours;
  "Without Kathleen"; "Exit gate"; decisions 6 and 7);
- `docs/design/scale-out.md` §10 (the scaling method, the time model, the job plan) and
  every decision as signed off;
- distributed-fmm.md §11, §14 and §15; laplace-fmm-plan.md §7 (Phase 5, Phase 5N);
- `fmm-validate/results/phase5-m3max.md`, `phase5-gh200.md` and `phase5n-kathleen.md`
  (the format, and the numbers to compare against);
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
    the gather, `upward_global`, evaluation time).
  - **The model's inputs**: the per-rank compute at the per-rank sizes of large runs
    (N/P = 10⁴–10⁶) on one node, and the Omni-Path latency and bandwidth seen by the
    exchanges at 2 and 4 nodes.
- **Timed runs on locust and the M3 Max**: Phase 5 T10's strong sweeps (N = 10⁶, 1–72
  ranks on locust, 1–12 on the M3 Max) on the Phase 5S code, as regression checks
  against `phase5-gh200.md` and `phase5-m3max.md`, with the load stated on locust.
- **Counted runs, never timed**: oversubscribed on locust at 128, 256 and 512 ranks and
  (Kathleen) on the 4 nodes at 320, 640 and 1,280 ranks, on the cube and the Plummer
  sphere at fixed N (10⁶, 10⁷) and at fixed N/P where memory allows: held boxes by kind,
  the reduction's and the multipole exchange's messages and bytes, the replicated top's
  bytes, the memory model and the peak resident size, the modelled per-rank cost's
  balance; against Phase 5's numbers (at 512 ranks: 37,449 held boxes and an 11.6 MB
  gather per rank).
- **Modelled times** to 10³ and 10⁴ ranks for the strong and weak configurations, with
  their inputs and method stated, each figure marked *model*, and the effects a model
  misses listed as open.
- **Correctness**: T2's invariance check at 512 against 64 on locust and (Kathleen) at 160
  against 20 and 1,280 against 160 oversubscribed; the direct-sum errors at every timed
  configuration against the smallest run (f64 to the printed digits, f32 within 1%),
  stating which runs were checked.
- **Node-hours**: state the node-hours per sweep and in total (budget for this task: at
  most 60 node-hours unless the sign-off raises it).
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
  commit with its test);
- run builds or measurements on Kathleen's login nodes, or exceed the node-hour budget
  without asking.

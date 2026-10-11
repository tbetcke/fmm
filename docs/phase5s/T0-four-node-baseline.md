# Phase 5S / T0 — the code on 4 Kathleen nodes, before the design (C5S.0)

Phase 5N ran under the user's 2-node cap (2026-10-10, "for now"): T2 measured the Phase 5
code on Kathleen at 1 and 2 nodes (`fmm-validate/results/phase5n-kathleen.md` §7–§9), and
everything above 2 nodes stayed "not measured", the 4-node queue wait included (the
probe's 4-node `small` job was cancelled after 7 min 34 s pending;
docs/phase5n/kathleen-probe.md). On 2026-10-11 the user lifted the cap to 4 nodes for
Phase 5S. This task takes the first 4-node look before T1 designs on it:
- the queue: how long 4-node jobs in `small` wait, so that T1 and decision 0 can say how
  much of the phase's Kathleen work runs at 4 nodes;
- the code as Phase 5N left it (Phase 5 with the node rule of Phase 5N T4) at 1, 2 and 4
  nodes, strong and weak, with the same harness and sweeps as Phase 5N T2, so the 2-node
  figures extend to 4 and the T4 rule's strategies are in every row;
- what grows with P across 160 ranks: the replicated part of the held boxes, the coarse
  gather, `Plan::new`, the exchanges at network bandwidth (Phase 5N T2's 0.5 GB/s per rank
  "until a run above 2 nodes says otherwise").

It changes no library code and designs nothing: its report is an input of T1 (which then
times at 4 nodes only what this task does not give) and the "Phase 5" side of T9's
comparison at 1–4 nodes.

**Without Kathleen** (the queue makes 4-node jobs impractical: no 4-node job started
within a day, or the user stops it), the task reports the waits it saw and stops;
decision 0 then records that Phase 5S runs at 1–2 nodes, and the briefs' 4-node runs
become "when the queue allows" (README, "Without Kathleen").

Read first:
- root CLAUDE.md, fmm-validate/CLAUDE.md, tools/kathleen/README.md (the jobs, the QoS
  table, the launcher and binding of Phase 5N decision 4, "Scaling sweeps");
- docs/phase5s/README.md ("Ranks on Kathleen", node-hours, "Without Kathleen", decision
  0);
- docs/phase5n/T2-kathleen-baseline.md (the across-nodes part: what this task repeats at
  4 nodes) and `fmm-validate/results/phase5n-kathleen.md` Part B (§7–§9), §12–§14 and
  "Phase 5N, T4"; docs/phase5n/kathleen-probe.md;
- `tools/scaling/run.sh` (the sweeps `nodes-strong`, `nodes-weak`, `nodes-check`, their
  `SPLITS` and `BUDGET`) and `tools/kathleen/jobs/sweep.sbatch`;
- docs/design/distributed-fmm.md §8.2 (the exchange model), §9.2 (memory per rank and the
  crossover *model* P ≈ √(N / 4,700)), §14.2 (what grows with P) and §15.4.

Do:
- **Tools for 4 nodes.** `tools/kathleen/README.md` and the job scripts' headers: the cap
  of 4 nodes per job (the user, 2026-10-11), 4-node jobs in `small` (2–6 nodes, 48 h; `test`
  takes exactly 2), a time limit sized from the 2-node runs (short limits backfill
  sooner), and an example `submit.sh --qos=small --nodes=4 …` line per job script that
  runs at 4 nodes. Change a script only where it assumes 2 nodes; `run.sh` takes the node
  count from the allocation already.
- **The queue.** The first 4-node job is a short smoke job (`mpi.sbatch 160`: the MPI test
  list at 4 × 40 ranks under `timeout`, then one `scaling` launch at 4 × 2 × 20), which is
  also the 4-node probe; then one job per sweep below. Record Submit, Start and the wait of
  every job (`sacct -X`), in the QoS table of tools/kathleen/README.md and in the report.
  At most 2 jobs pending or running at a time. If the first 4-node job has not started
  after 24 hours, stop and report (decision 0).
- **Strong scaling at 1, 2 and 4 nodes** (`nodes-strong`, `SPLITS="40x1 2x20"`, the
  harness's budget of 96 MB, so the strategy Phase 5N T4's rule picks, with Rotation and
  Dense at p = 8 as the sweep names them): the cube and the Plummer sphere at N = 10⁶ and
  10⁷, f64 p = 3 and 8, overlap off and on. Run 1 and 2 nodes again on the current code
  (the T4 rule changed `Auto` since Phase 5N T2's runs), so that each row of 1, 2 and 4
  nodes comes from one revision; state the revision.
- **N = 10⁸ at 1, 2 and 4 nodes**: the cube and the Plummer sphere at f64 p = 3 and the
  rule's strategy at p = 8, at 2 × 20 per node, and at 40 × 1 where memory allows (state
  the peak resident size per node); two timed evaluations; one launch per configuration,
  under `timeout`. This is the first N = 10⁸ run of the code, and the size T1's design and
  T9's strong scaling use.
- **Weak scaling** (`nodes-weak`): 10⁶ points per node at 1, 2 and 4 nodes. Phase 5N T2
  found the cube's weak efficiency set by its tree (a seventh level at N = 2·10⁶), not by
  the network: report the levels, leaves and V, W, X pairs per rank beside each cube row,
  and read the network from the Plummer sphere and from the per-rank stage times at equal
  work.
- **Per run at 4 nodes**, as Phase 5N T2 §8–§9 at 2: the stages by rank (max, min,
  mean), the compute imbalance, the bytes and messages per exchange, the blocking
  exchange times against §8.2's model and against Phase 5N's 0.5 GB/s per rank, the
  exposed waits with overlap on (the C5.2 measure), the coarse gather's bytes, held boxes
  split into local and not local, `Plan::new`'s time, the memory per rank (`VmHWM` and
  §9.2's model). Tabulate them against P at 40, 80 and 160 ranks, beside §9.2's crossover
  and §14.2's growth.
- **Errors** (`nodes-check` at 4 nodes): the direct-sum errors and the one-rank reference
  at N = 10⁶ (f64 to the printed digits, f32 within 1%, the reference within 100 u_T),
  and the direct-sum errors at N = 10⁷.
- **Repeats**: four configurations of the strong sweep at 4 nodes run twice, in different
  jobs on different nodes; report the spread as Phase 5N T2 §7 does.
- **Report** as a new part of `fmm-validate/results/phase5n-kathleen.md`, "Phase 5S, T0:
  four nodes" (after "Phase 5N, T4"), in that report's format: setup and revision, the
  jobs with nodes, QoS, waits and node-hours; strong scaling at 1, 2 and 4 nodes with
  speed-up and efficiency against 1 and against 2 nodes; N = 10⁸; weak; per run at 4
  nodes; the replicated part against P; errors; repeats; "For T1": what changed from 2 to
  4 nodes, the exchange rate per rank to model with, where the replicated part and the
  coarse gather stand at 160 ranks against §9.2 and §14.2, the 4-node queue waits, and
  what T1 still has to time at 4 nodes. Every figure measured on Kathleen, every model
  marked *model*. Update fmm-validate/CLAUDE.md's report paragraph with the new part.

Tests that define done:
- The MPI test list passes at 4 × 40 ranks (`mpi.sbatch 160`), and the `scaling` smoke run
  at 4 nodes.
- Every launch of the sweeps above has a file, or the report lists the missing ones with
  the reason (a failed launch is reported, never retried silently).
- The error checks at 4 nodes match the one-rank run as Phase 5N T2's did at 2.

Must pass: nothing builds differently. `cargo fmt -- --check` on the M3 Max; the sweeps
and the MPI test list on Kathleen in jobs, every launch under `timeout`, the binding
printed.

Do not:
- change any library crate, a default or the harness's figures to improve a number; a
  defect found here is reported, and fixed in its own commit with its test after asking;
- assert timings, or commit CSV or raw output beyond the report;
- submit a job larger than 4 nodes, or more than 20 node-hours in all (report the use);
- run builds or measurements on Kathleen's login nodes;
- present 4-node figures as scaling to thousands of ranks, or a model as a measurement.

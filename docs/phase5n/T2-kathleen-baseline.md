# Phase 5N / T2 — the Phase 5 code on Kathleen: one node, 2 and 4 nodes, the x86 host (C5N.2; skipped without Kathleen)

Phase 5 T10 measured the distributed FMM on two single nodes (the M3 Max to 12 ranks,
locust to 72; `fmm-validate/results/phase5-m3max.md`, `phase5-gh200.md`). Every figure was
one node over shared memory, every host time was NEON, and distributed-fmm.md §8.2's
numbers at network bandwidth are a *model*. This task runs the same harness on Kathleen,
unchanged in what it computes, and measures:
- one Kathleen node (40 Cascade Lake cores in two sockets) at 1–40 ranks, the ranks ×
  threads splits and the M2L strategies, to set the node's baseline beside the M3 Max and
  locust;
- 1, 2 and 4 nodes, the first inter-node figures of the code: exchanges over Omni-Path
  against §8.2's model, overlap where the network makes it matter, the replicated data and
  memory per rank up to 160 ranks (no Kathleen job is larger than 4 nodes, README);
- the x86 host on one rank: the AVX2 + FMA P2P kernels, never timed on Intel, and the
  one-rank evaluation against the M3 Max and Grace.

It changes no library code. Its report is the input of T3 (the M2L study) and of Phase 5S
T1 (the scale-out design). It does not optimise anything (README, "Out of scope").

**Without Kathleen** (decision 0: proceed without it), this task is skipped: Phase 5
T10's reports are the M3 Max and locust baseline. If the user asks for a specific sweep on
the M3 Max or locust instead (for example N = 10⁷ on locust), it runs with the existing
harness and is reported as a section of the relevant T10 report; the Slurm work, the x86
profile and the inter-node figures wait until Kathleen is revisited.

Read first:
- docs/phase5n/README.md ("Design decisions": "Working on Kathleen", "Ranks on Kathleen",
  "The workloads"; decisions 3 and 4) and `tools/kathleen/README.md` (T1);
- docs/phase5/T10-scaling.md and both T10 reports: the format and the sweeps to repeat;
- `fmm-validate/src/scaling.rs` (`Workload`, `Settings`, `run`, `Report`, `machine`,
  `resident_memory`) and `examples/scaling.rs` (its options), `tools/scaling/run.sh` (the
  sweeps, `launch`, the Linux binding flags);
- docs/design/distributed-fmm.md §8.2 (the exchange model at 10 and 100 Gbit/s), §9.2
  (memory per rank), §11 (scaling method), §14.2 (what grows with P) and §15.2 (Phase 5's
  measured numbers);
- `fmm-validate/examples/p2p_kernels.rs` and `p2p_fmm.rs`, and docs/design/simd-p2p.md §4.6
  (the throughput model, AVX2 rows "not timed") and §8 (benchmarking rules);
- laplace-fmm-plan §7, Phase 4S "Benchmarks" (the Grace host times at 1 and 72 threads).

Do:
- **The harness on Slurm.**
  - `tools/scaling/run.sh` (or a sibling `tools/scaling/slurm.sh`, stated) runs a sweep
    inside one allocation: the launcher and binding of decision 4 (`srun --nodes
    --ntasks-per-node --cpus-per-task --cpu-bind=verbose,cores`, or `mpirun` inside the
    allocation), an external `timeout` per launch, one file per launch as today. In place
    of the load check (Kathleen's nodes are exclusive), each file records the job id, the
    node list, the QoS and the binding. A job script per sweep (with `module purge` and
    `. tools/kathleen/env.sh`) submits it.
  - The `scaling` report's block gains, where it runs under Slurm: nodes, ranks per node
    (from `split_shared`), threads per rank, the binding as Slurm reports it, the job id
    (`SLURM_JOB_ID`) and the MPI library version. Keep the M3 Max and locust output as it
    is apart from the new lines. Test the additions in `tests/scaling.rs` where they do
    not depend on Slurm.
  - Batch launches so a sweep is one allocation per node count, not one job per launch.
- **One node** (N = 10⁶, the cube and the Plummer sphere; f64 p = 3 and 8, f32 p = 8; the
  default strategy; overlap off and on from one build, as in T10):
  - strong scaling at 1, 2, 4, 8, 10, 16, 20 and 40 ranks × 1 thread, with the ranks
    spread over both sockets (state the mapping);
  - ranks × threads at 40 cores: 1 × 40, 2 × 20 (one rank per socket), 4 × 10, 8 × 5,
    10 × 4, 20 × 2, 40 × 1, f64 p = 8;
  - the strategies Dense, Rotation and Classes at f64 p = 8 at every rank count of the
    strong sweep (T10's `strategy` sweep);
  - the stage table against ranks (T10's "Stages against ranks"): does the downward pass
    stop scaling as on the M3 Max and locust, and at how many ranks per socket?
- **Across nodes** (the first inter-node figures; `test` for 2 nodes, `small` for 4;
  never more than 4 nodes):
  - strong scaling of the N = 10⁷ cube and Plummer sphere (and N = 10⁶ for comparison with
    one node) at 1, 2 and 4 nodes, at 40 × 1 and 2 × 20 per node, f64 p = 3 and 8
    (the default strategy, and Rotation at p = 8 so that the table effect is separable);
  - weak scaling at 10⁶ points per node (N = 10⁶ × nodes), the same configurations;
  - per run, as in T10: the stages by rank (max, min, mean), the compute imbalance, the
    bytes and messages per exchange, the blocking exchange times and the exposed waits
    with overlap on, the coarse gather, memory per rank (`VmHWM` and the §9.2 model);
  - **the exchanges against the model:** for each exchange, the measured blocking time
    against distributed-fmm.md §8.2's model at Omni-Path's nominal bandwidth (100 Gbit/s;
    state the latency you assume) and the bytes measured; where they differ by more than
    a factor of two, say what the per-message cost explains;
  - **overlap across nodes:** overlapped over blocking wall time, and the exposed waits
    against the blocking exchange times (the C5.2 measure, design §8.6), at 2 and 4 nodes;
  - **the replicated data at 80–160 ranks:** held boxes not local, the coarse gather's
    bytes, `Plan::new`'s time and peak memory, against T10's locust figures at 64–72 ranks
    (and its oversubscribed 256 and 512) and design §14.2;
  - the errors against the direct sum and the one-rank reference (`--errors 8
    --reference`) at 2 and 4 nodes at N = 10⁶ (f64 to the printed digits, f32 within 1%,
    and within 100 u_T of the one-rank `Fmm`), and with `--errors 8` alone at N = 10⁷.
- **The x86 host on one rank** (one node, one rank, release, BLAS variables at 1):
  - `p2p_kernels` (every ISA the machine runs: AVX2 + FMA, scalar, the reference; φ and
    φ with ∇φ; f32 and f64), in pairs per second against simd-p2p.md §4.6's AVX2 model;
    the accuracy columns as the example prints them;
  - `p2p_fmm`: the kernels inside the FMM (the leaf stage) at p = 3 and 8;
  - the one-rank evaluation of the N = 10⁶ cube and Plummer sphere at f64 p = 3 and 8
    (Dense and Rotation) and f32 p = 8, at 1 thread and 40 threads, beside the M3 Max's
    and locust's T10 one-rank figures and the Phase 4S Grace host table.
- **Report** `fmm-validate/results/phase5n-kathleen.md`, in the format of the T10
  reports:
  - it states that every figure is measured on Kathleen, names nodes × ranks per node ×
    threads per rank, the binding, the launcher, the transport and the job ids, and marks
    every model as *model*;
  - setup (machine facts from `tools/kathleen/machine.md`, software, modules, environment);
  - sections: one node (strong, stages, ranks × threads, strategies), across nodes (strong,
    weak, exchanges against the model, overlap, replicated data and memory, errors), the
    x86 host, and "Not measured";
  - "For T3 and Phase 5S": the node's M2L behaviour against the M3 Max and locust, the
    inter-node exchange costs per byte and per message, where the replicated part starts
    to dominate, and the configurations the scale-out design should assume;
  - the node-hours used, and one run's raw output in full.
- Add `tools/scaling/` documentation for the Slurm mode (README or the script's header)
  and update `fmm-validate/CLAUDE.md`'s Phase 5 T10 paragraph with the Kathleen report.

Tests that define done:
- `tests/scaling.rs` still passes at 1 and 2 ranks on the M3 Max, with the new report
  fields checked where they do not need Slurm.
- Every launch of the sweeps above has a file, or the report lists the missing ones with
  the reason (a failed launch is reported, never retried silently).
- The error checks of the sweeps match the one-rank run as T10's did (f64 to the printed
  digits, f32 within 1%, reference within 100 u_T), at 2 and 4 nodes.

Must pass: `cargo fmt --all`, the root checks and the stricter workspace checks on the
M3 Max; `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`; the smoke test at 2 ranks
by hand; the sweeps on Kathleen in jobs.

Do not:
- change any library crate, or a default, to improve a number; a defect found here is
  reported and fixed in its own commit with its test, after asking;
- assert timings, or commit CSV or raw output beyond the report;
- submit any job larger than 4 nodes (160 cores), or exceed the node-hours of decision 3
  (proposed: 100); report the use;
- present a model as a measurement, or a Kathleen figure at 4 nodes as scaling to
  thousands of ranks.

# Phase 5N: one node — Kathleen, and the M2L tables per node

Phase 5 ran the Laplace FMM on any number of ranks, bit for bit the one-rank result over
the union of the points, and measured it on two single nodes: the Apple M3 Max (up to 12
ranks) and locust (up to 72; 256 and 512 oversubscribed, never timed). Its scaling report
(`fmm-validate/results/phase5-m3max.md`, `phase5-gh200.md`; laplace-fmm-plan §7, Phase 5)
left three things open that a cluster now makes urgent:

- **Nothing has run across nodes.** Every Phase 5 figure is one node over shared memory.
Kathleen, UCL's Slurm cluster (`ssh kathleen`), is now available: 190 nodes of two
Intel Xeon Gold 6248 (Cascade Lake) with Intel Omni-Path. No Rust, no build of this
workspace and no run exists there yet, and the scale-out design (Phase 5S) must start
from measured inter-node numbers, not from distributed-fmm.md §8.2's network *model*.
- **No Intel CPU has run the host path.** `nd-fmm-simd`'s AVX2 + FMA kernels are checked
in CI on AMD runners and were never timed; no Intel CPU and no AVX-512 path ran
(docs/design/simd-p2p.md, "Outcome of Phase 3S", §5.5). Kathleen is the first Intel
machine.
- **At p = 8 the per-rank dense M2L tables limit strong scaling on one node.** Every rank
holds its own 316 dense tables (16.6 MB in f64, 8.3 MB in f32). On the M3 Max the f64
evaluation is fastest at 4 ranks, on locust it stops improving after 32; Rotation
keeps 0.75–0.79 efficiency at 12 and 72 ranks, Classes (Dense's products from 16 class
matrices) 0.69–0.77, and threads, which share one copy, beat ranks (4 × 18 is 4.1×
faster than 72 × 1 on locust). Rotation was faster than Dense at every rank count,
even on one rank (0.79× and 0.85× Dense's time on the M3 Max, 0.95× and 0.96× on
locust), yet `M2lStrategy::Auto` picks Dense up to p = 8 from Phase 2's one-rank,
one-pair measurements (`fmm-exec/src/tables.rs`, `M2lStrategy::resolve`). A Kathleen
node runs 40 ranks over two 27.5 MB L3 caches: 20 copies of 16.6 MB per socket would
dominate any cluster measurement taken before this is fixed.

This phase therefore comes before the scale-out (Phase 5S) and before Phase 6. It first
finds out whether Kathleen is usable at all: a queue probe of trivial 2-node jobs, then a
stop for the user's decision 0 (Kathleen usable?). If yes, it
builds the Kathleen environment, measures the Phase 5 code there on one node and on a few
nodes, studies M2L on a node on all three machines, and fixes the default for many ranks
per node. The deep fix of table pressure, a batched host M2L that reads each table once
per level (Phase 4 decision 6, "a host batched-GEMM path through BLAS, deferred; a
candidate for Phase 6"), stays in Phase 6 as C6.6.

The phase has three parts, after a probe:
0. **The queue probe (T1, first step).** Trivial jobs at 2 nodes (`test` QoS), with the
   scheduler's start estimates and the queue depth by QoS; a report; then **stop** for
   decision 0. No environment is set up before that decision. (The 4-node `small` jobs and
   the day of repeats first planned were dropped on 2026-10-10 with the 2-node cap, below.)

1. **Kathleen (C5N.1).** A reproducible environment from files in the repository
 (`tools/kathleen/`), how a session on the M3 Max drives builds and Slurm jobs there,
 and every existing CPU-side check run on Kathleen.
2. **The baseline (C5N.2).** Phase 5's code on Kathleen: one node at 1–40 ranks and the
 ranks × threads splits, a first multi-node look at 1 and 2 nodes, the exchanges at
 network bandwidth against the model, overlap where it can matter, and the x86 host
 profile (P2P on AVX2).
3. **M2L on a node (C5N.3).** A measured study on the three machines and a short design
 (`docs/design/node-m2l.md`), signed off, then the strategy rule that sees the node,
 safe table loading at scale, and the ranks × threads guidance.

Companion documents:

- [docs/design/distributed-fmm.md](../design/distributed-fmm.md): §8.2 (the network
model), §9.2 (memory per rank), §11 (scaling method), §14 (toward thousands of ranks;
S6, ranks × threads) and §15 (Phase 5 outcome, measured numbers);
- [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md): Section 6.8 (threads
and BLAS), Section 7 (Phase 2 T8, the source of the `Auto` rule; Phase 5 and its
"Recommendation for Phase 6"), Sections 8.3 and 9;
- [docs/design/simd-p2p.md](../design/simd-p2p.md): "Outcome of Phase 3S", §4.7 and §5.5 (x86_64, AVX2, AVX-512
deferred);
- [docs/design/workspace-structure.md](../design/workspace-structure.md): Sections 3 and
6;
- `tools/gh200/README.md`: the locust environment, the model for `tools/kathleen/`;
- `fmm-validate/results/phase5-m3max.md` and `phase5-gh200.md`: the measurements this
phase starts from.

T3 adds a further companion, `docs/design/node-m2l.md`.

## Decision 0 and the path without Kathleen

T1 starts with the queue probe and stops with its report. The user then decides:

- **Kathleen usable:** T1 continues with the environment (C5N.1), and T2–T4 run as written,
Kathleen included, every job at most 2 nodes.
- **Without Kathleen (for now):** T1 ends with the probe report (merged as such: the
probe script and the report, no environment); T2 is skipped, since Phase 5 T10's reports
are the M3 Max and locust baseline, unless the user asks for a specific M3 Max or
locust sweep there; T3 and T4 run on the M3 Max and locust only, and the x86 profile,
the inter-node figures and the Kathleen columns are marked "not measured". Kathleen is
revisited later (a new probe), and its parts of T1 and T2 run then.

Every brief says what it does on each path. The exit gate and the checklist mark the
Kathleen items "not applicable" on the second path.

Prerequisite: Phase 5 is complete. T10 is merged (PR #89), and the design documents carry
the Phase 5 outcome. The briefs assume the workspace as it is after that merge.

## Scope

In scope:

- `tools/kathleen/`: the queue probe (`probe.sh`, T1 step 0); then, with Kathleen, setup,
environment, sync and remote scripts, Slurm job scripts, machine facts, README (T1).
- `tools/scaling/`: the sweeps on Slurm, and the new knobs of T4 (T2, T4).
- `nd-fmm-validate`: the `scaling` example's additions for Kathleen (nodes, ranks per
node, binding, job id in the report), and the Kathleen baseline report (T2); the M2L
study's measurements, run through existing examples or small additions (T3).
- `nd-fmm-exec`: the strategy rule with the node context, the table loading at scale, and
the `threading` guidance, as signed off (T4).
- `docs/design/node-m2l.md` (T3), and the design-document update at the end of the phase
(T4).
- Root `CLAUDE.md`: Kathleen in "Build environment", "Checks" and the current phase (T1).

Out of scope:

- The scale-out packages S1–S7 of distributed-fmm.md §14 (Phase 5S). T2 measures what
they will need; it changes no structure.
- A batched host M2L, SVD-compressed tables, AVX-512 P2P and the other Phase 6 items
(C6.1–C6.8). T3 may recommend their order; it does not build them.
- Sharing one copy of the tables between the ranks of a node through MPI shared memory:
rsmpi 0.8.2 has no RMA windows (no `MPI_Win_allocate_shared`; checked in the registry
source), and memory mapping needs `unsafe` or a new dependency. T3 states the options;
an rsmpi addition is an upstream change and is asked for first (decision 6).
- Changes to `nd-octree`, `nd-fmm-plan`, `nd-fmm-math`, `nd-fmm-ref`, `nd-fmm-tables`,
`nd-fmm-simd` and `nd-fmm-kernels`, beyond fixing a defect a test exposes (stop and
report first). `nd-fmm-tables`' cache may gain what T4's table loading at scale needs,
if T3 shows the existing `TableCache` is not enough.
- GPUs on Kathleen (it has none); device scaling.
- Changes to the one-rank results, other than the strategy that `Auto` picks if decision
5 allows it.

## Requirements

T1–T4 are accepted against these; a task may propose a change, with reasons, for
sign-off.

1. **Probe first.** Nothing is installed or built on Kathleen before decision 0. The probe
 writes only job output under `~/Scratch/fmm-probe/`.
2. **Reproducible.** The Kathleen environment is recreated from the repository by one
 script, with the toolchain and the modules pinned and recorded.
3. **Within the rules of the machine.** Builds and runs go in Slurm jobs. The login node
 runs only short (under 15 minutes), light commands: syncing, submitting, reading
 output, `cargo fmt`, `cargo metadata`. Nothing outside the chosen root on Kathleen;
 the 250 GB quota of home and Scratch together is respected and its use reported.
4. **Phase 5's guarantees hold.** The host output on P ranks is bit for bit the one-rank
 `Fmm` over the union in rank order; 100 u\_T between input distributions;
 deterministic from evaluation to evaluation and for every thread count. The strategy
 rule of T4 is a function of settings that every rank agrees, so every rank resolves the
 same strategy.
5. **Collective discipline.** As root `CLAUDE.md`, "MPI": every rank reaches every
 collective in the same order; any new collective for the node context (a
 `split_shared` on the host path, an agreement) is counted and justified.
6. **Measured, never asserted.** Every timing is a release build with every BLAS thread
 variable at 1, names its machine, and on Kathleen also the nodes, ranks per node,
 threads per rank, the binding, the MPI library and transport and the job id. No
 timing in a test or in CI.
7. **Tested on what can run.** With Kathleen, every existing multi-rank test passes there
 at 1, 2, 4 and 8 ranks on one node and at 2 nodes × 40 ranks; every multi-rank test
 still passes by hand on the M3 Max and locust; the CI job stays at 2 and 4 ranks on
 GitHub's runners. No Kathleen job exceeds 2 nodes.
8. **Defaults by measurement.** The strategy rule rests on T3's measurements on every
 machine available after decision 0 (three with Kathleen, the M3 Max and locust without), at the ranks-per-node counts each machine runs, with the rule and its source
 stated in `M2lStrategy::resolve`'s documentation.

## Design decisions for this phase

These hold for every task:

- **The machines.** The M3 Max and locust as in Phase 5 (docs/phase5/README.md, "Ranks
on the M3 Max", "Ranks on locust"), and Kathleen, the third machine, for x86, AVX2,
two NUMA domains per node and Omni-Path between nodes.
- **Working on Kathleen.** A Claude Code session runs on the M3 Max and drives Kathleen
over `ssh kathleen` (outside the sandbox, which denies `~/.ssh`), as Phase 4S did for
locust:
  - `tools/kathleen/sync.sh` copies the working tree, uncommitted work included, to the
  chosen root on Kathleen; `tools/kathleen/remote.sh` runs a short command there inside
  `env.sh`; job scripts build, test and run inside Slurm jobs, and a helper submits one
  and waits for it (T1 chooses the form). Kathleen never commits or pushes.
  - **Login node:** only short, light commands. A build is a job (the `singlenode` QoS,
  6 h, or `test`, 1 h); so is every test and every timed run.
  - **Storage:** home (`/home/ucahtbe`, Lustre) and Scratch (`~/Scratch -> /scratch/scratch/ucahtbe`) share one hard quota of 250 GB, and jobs fail once it is
  full. Compute nodes are diskless and have no `$TMPDIR`: Open MPI's session directory,
  cargo's and rustc's temporary files must go elsewhere (T1 decides: `/dev/shm` or
  Scratch, and measures). Target directories are large; T1 reports their size and
  keeps at most two (one per branch in use).
  - **Jobs:** whole nodes only (jobs never share nodes). QoS as probed: `test` (1 h, ≤ 2
  nodes, ≤ 2 jobs), `singlenode` (6 h), `small` (48 h, ≤ 6 nodes), `medium` (24 h, ≤ 12
  nodes), `large` (12 h, any size; at most 144 nodes per job by the documentation).
  **No Kathleen job is larger than 2 nodes (80 cores), for now** (the user's rule, from
  2026-10-10; it replaces the earlier cap of 4 nodes, and only the user lifts it):
  correctness and multi-node runs use `test` (≤ 2 nodes, 1 h) or, for longer 2-node runs,
  `small`; `medium` and `large` are not used. Figures at more than 2 nodes are "not
  measured" in this phase.
  - **Node-hours:** every report states the node-hours its jobs used. The budget is
  decision 3.
- **Ranks on Kathleen.** Ranks × threads per rank at most 40 per node (the physical
cores; hyperthreading stays off, the default `--hint=nomultithread`), every rank bound
to its cores and the binding printed (`srun --cpu-bind=verbose,cores` or `mpirun --report-bindings --bind-to core`, as T1 decides). Splits are named nodes × ranks per
node × threads per rank, and the default study set per node is 40 × 1, 20 × 2, 10 × 4,
4 × 10 and 2 × 20 (one rank per socket). Every `srun` or `mpirun` runs under the job's
time limit and an external `timeout` per launch. Open MPI 4.1.6 is the module default;
T1 records the transport it uses over Omni-Path (PSM2 or OFI) and whether `srun`
launches it (PMIx).
- **Ranks on the M3 Max and locust.** As in Phase 5. On locust the load is checked before
and after every timing run and stated; take no timings while another user's job could
influence them. Kathleen's nodes are exclusive, so a Kathleen report states the node
list and needs no load check.
- **Threads and BLAS.** The Phase 3 rule stands (`fmm-exec/src/threading.rs`): ranks ×
rayon threads × BLAS threads stay within the physical cores of a node; with threads
  > 1, MPI is initialised at `Threading::Funneled`. On Kathleen, OpenBLAS comes from the
  > module (`openblas/0.3.28/gcc-12.3.0`, not the `-omp` build) and every BLAS variable is
  > still set to 1.
- **The workloads.** Those of Phase 5 T10 (`nd_fmm_validate::scaling::Workload`: the
cube, the Plummer sphere, the Gaussian clusters, every point generated by index, seed
0x5ca1e), at N = 10⁶ and 10⁷, f64 at p = 3 and 8 and f32 at p = 8; T3 adds p = 6, 10
and 12.
- **The oracles.** Unchanged from Phase 5: the one-rank `Fmm` over the union in rank
order where it fits, the distributed direct-sum sampling of the `scaling` harness, and
the errors of the one-rank run (to the printed digits in f64, within 1% in f32).
- **Tests and MPI.** The root rules apply. No new MPI-initialising test executable unless
a task needs one; a test that needs a rank count skips cleanly otherwise.
- **Dependencies.** No new external dependency. rlst stays at 0.9.0 and mpi at 0.8.2.
Kathleen's modules are system software, not crate dependencies; `tools/kathleen/` pins
their versions.

## Exit gate

- T1's queue probe is reported and decision 0 is made. The Kathleen items below apply only
if Kathleen was found usable; otherwise they are "not applicable", and T3 and T4 meet
theirs on the M3 Max and locust.
- The Kathleen environment rebuilds from the repository into an empty root by one script,
and every existing CPU-side check passes there (T1).
- The baseline report `fmm-validate/results/phase5n-kathleen.md` is published, with the
one-node sweeps, the multi-node first look at 1 and 2 nodes, the exchanges against the
network model, and the x86 P2P profile (T2).
- `docs/design/node-m2l.md` is signed off before T4 starts (T3).
- The strategy rule is built as signed off; on each machine, at every ranks-per-node
count measured, the default configuration at p = 8 is within the margin T3 proposes of
the best strategy; every Phase 5 test still passes, and the host output stays bit for
bit the one-rank `Fmm` of the same resolved strategy (T4).
- The design documents carry the Phase 5N outcome (T4).

## Tasks

One pull request each.

- T1 first, and its probe before anything else: decision 0 follows its probe report.
- T2 needs T1 (and is skipped without Kathleen).
- T3 needs T2 for its Kathleen numbers; its M3 Max and locust measurements can start once
decision 0 is made. Without Kathleen it needs only decision 0. Its design is signed off
before T4.
- T4 needs T3 signed off.


| Task | Brief                                                    | Delivers                                                                                                                                                                                                                                                                                             | Component      | Depends on                              |
| ---- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | --------------------------------------- |
| T1   | [T1-kathleen-environment.md](T1-kathleen-environment.md) | first the queue probe (2-node trivial jobs; report; stop for decision 0); then, if Kathleen is usable, `tools/kathleen/` (setup, env, sync, remote, Slurm job scripts, machine facts, README), root `CLAUDE.md`, every CPU-side check on Kathleen in jobs, including the MPI tests at 2 nodes        | C5N.1          | none                                    |
| T2   | [T2-kathleen-baseline.md](T2-kathleen-baseline.md)       | the sweeps on Slurm; Phase 5's code on one Kathleen node (1–40 ranks, splits, strategies) and on 1 and 2 nodes; the exchanges at network bandwidth against distributed-fmm.md §8.2; overlap; replicated data and memory to 80 ranks; the x86 P2P profile; `fmm-validate/results/phase5n-kathleen.md` | C5N.2          | T1; skipped without Kathleen            |
| T3   | [T3-m2l-node-study.md](T3-m2l-node-study.md)             | the M2L study on a node on the three machines (strategies × p × precision × ranks per node × threads), table sizes against the caches, table build and cache load at scale; `docs/design/node-m2l.md` with the proposed rule and the sign-off questions                                              | C5N.3 (design) | decision 0; T2 for the Kathleen numbers |
| T4   | [T4-node-m2l.md](T4-node-m2l.md)                         | the signed-off strategy rule in `nd-fmm-exec`, table loading at scale, the ranks × threads guidance, the harness knobs; before/after runs on the three machines; the design-document update                                                                                                          | C5N.3          | T3 (signed off)                         |


T1 changes `tools/`, the root `CLAUDE.md` and `.gitignore` only. T2 and T4 both change
`tools/scaling/` and `fmm-validate/src/scaling.rs`: merge T2 first. T4 changes
`fmm-exec/src/tables.rs` and `fmm.rs`.

## Machines


| Machine                                                                                                                                        | Ranks                                                          | Used for                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Apple M3 Max (development; 12 performance and 4 efficiency cores, 64 GB, Open MPI 5.0.10)                                                      | 1–12                                                           | development; every task's multi-rank tests by hand; T3's study                                                              |
| locust (NVIDIA GH200: 72 Neoverse-V2 cores in one NUMA node, 572 GB, one H100; Open MPI 5.0.10 from spack; shared, no scheduler; tools/gh200/) | 1–72, one rank per core                                        | T3's study, with the load checked and stated; regression runs                                                               |
| **Kathleen** (UCL; `ssh kathleen`, Slurm; tools/kathleen/ from T1)                                                                             | 1–40 per node; at most 2 nodes (80 cores) for any job, for now | the queue probe and decision 0 (T1); if usable: builds and checks in jobs (T1), the baseline (T2), T3's study and T4's runs |
| GitHub Actions `ubuntu-latest` (4 vCPUs)                                                                                                       | 2 and 4                                                        | the `run-tests-mpi` job, unchanged                                                                                          |


### Kathleen, as probed read-only on 2026-10-10

From the M3 Max, without writing anything or submitting a job; T1 re-checks and records
the facts in `tools/kathleen/machine.md`.


| Item         | Value                                                                                                                                                                                                                                                                                                                |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nodes        | 190 compute nodes (Slurm partition `kathleen`), diskless, plus two login nodes of the same kind (login22.kathleen.ucl.ac.uk answered)                                                                                                                                                                                |
| CPU          | 2 × Intel Xeon Gold 6248 (Cascade Lake), 20 cores each, 2.5 GHz; 40 cores per node, 2 threads per core (hyperthreading off for jobs by default); AVX-512F and AVX2                                                                                                                                                   |
| Caches, NUMA | L3 27.5 MB per socket (55 MiB in 2 instances); 2 NUMA nodes                                                                                                                                                                                                                                                          |
| Memory       | 192 GB per node (188 GB visible)                                                                                                                                                                                                                                                                                     |
| Network      | Intel Omni-Path                                                                                                                                                                                                                                                                                                      |
| OS           | RHEL 9.6, kernel 5.14.0-570, x86\_64                                                                                                                                                                                                                                                                                 |
| Modules      | `ucl-stack/2026-03` (default): gcc 12.3.0, `mpi/openmpi/4.1.6/gcc-12.3.0` (default; also 4.1.2, 4.1.8 and Intel oneAPI MPI 2021.14), `openblas/0.3.28/gcc-12.3.0` (and `-omp`), `fftw/3.3.10/gcc-12.3.0`, `cmake/3.30.5`, `llvm/17.0.6` (hidden: `module avail --all`), `python/3.11.6`, `intel-oneapi-mkl/2023.2.0` |
| Rust         | none installed; static.rust-lang.org, index.crates.io and github.com reachable from the login node                                                                                                                                                                                                                   |
| Storage      | HOME `/home/ucahtbe` (Lustre); `~/Scratch -> /scratch/scratch/ucahtbe`; 250 GB hard quota for both together (`lquota`); no node-local disk, no `$TMPDIR`                                                                                                                                                             |
| Slurm        | whole-node jobs; QoS `test` 1 h ≤ 2 nodes ≤ 2 jobs, `singlenode` 6 h 1 node, `small` 48 h ≤ 6 nodes, `medium` 24 h ≤ 12 nodes, `large` 12 h any size; about 2,000 jobs pending at the probe                                                                                                                          |
| Policy       | login nodes for short (&lt; 15 min), light work only; UCL documentation: [https://www.rc.ucl.ac.uk/docs/Clusters/Kathleen/](https://www.rc.ucl.ac.uk/docs/Clusters/Kathleen/)                                                                                                                                        |


## Decisions to sign off

Each is recorded in the exit checklist when made:
0. **Kathleen usable?** From T1's probe report (start delays at 2 nodes, the scheduler's
   estimates, the queue depth by QoS): proceed with Kathleen, or
   proceed without it (the M3 Max and locust; "Decision 0 and the path without
   Kathleen") and revisit later.

1. **The order and the names.** Phase 5N (this phase) before Phase 5S (the scale-out on
 Kathleen) and Phase 6 (optimisation and extensions); the inserted phases keep the
 numbers that code and documents cite (C6.x), as 3S and 4S did.
2. **Kathleen as the third machine:** the root directory on Kathleen and its layout, how
 the 250 GB quota is shared between the environment, the source copies and the target
 directories (T1 proposes).
3. **The node-hour budget:** per task and for the phase. Proposed: T1 at most 20
 node-hours (the probe's trivial jobs included), T2 at most 100, T3 at most 50 on
 Kathleen, T4 at most 30, every job at most 2 nodes; each report
 states its use.
4. **The launcher on Kathleen:** `srun` (PMIx) or `mpirun` inside an allocation, and the
 binding options (T1 measures both on 2 nodes and proposes one).
5. **The strategy rule** (T3), including whether the one-rank default may change. A new
 `Auto` changes which strategy runs at fixed settings, and so the output bits (within
 the accuracy of each strategy); Phase 5 kept the one-rank defaults fixed (docs/phase5/
 README.md, "Out of scope"). Options: change `Auto` everywhere as measured; change it
 only when ranks per node exceed a threshold; or keep `Auto` and document the
 configuration (threads per rank, an explicit strategy) instead.
6. **Shared tables per node** (T3): none in this phase; or an rsmpi addition
 (`MPI_Win_allocate_shared` behind a safe API) asked for upstream; or left to C6.6, the
 batched host M2L of Phase 6, which reads each table once per level and makes the copy
 per rank cheap.
7. **The table cache at scale** (T3): every rank building its own tables (216 ms at
 p = 8 on locust, 63–83% of the build at 8–72 ranks; distributed-fmm.md §15.2), the
 existing `TableCache` on Lustre warmed by one job, or one rank per node building and
 storing first; what T4 builds.
8. **The margin of the exit gate** (T3): how far the default may lie from the best
 strategy at p = 8 at each ranks-per-node count (for example 10%).

## Risks


| Risk                                                                                                                              | Mitigation                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Builds on the login node break the machine's rules, or fill the quota                                                             | builds in Slurm jobs; target directories counted and limited (T1); `lquota` in every report                                                                                                                                  |
| Diskless nodes: Open MPI, cargo or rustc write to a `/tmp` that is RAM, or to a path that does not exist                          | T1 sets `TMPDIR` and Open MPI's session directory explicitly and measures what lands where                                                                                                                                   |
| The queue delays jobs so long that work on Kathleen stalls                                                                        | T1's probe measures the delays at 2 nodes before any setup, and decision 0 can drop Kathleen; correctness in the `test` QoS (≤ 2 nodes, 1 h); no job above 2 nodes; jobs batched so one allocation runs a sweep              |
| Open MPI 4.1.6 behaves differently from the 5.0.10 of the M3 Max and locust (transport selection, `srun` launch, threading level) | T1 runs the MPI test list at 1–8 ranks on one node and 2 × 40 across nodes; records the transport; MPI initialised at `Funneled` checked                                                                                     |
| The AVX2 kernels behave differently on Intel (the `rsqrtps` estimate is not architecturally defined)                              | T1 runs `run-tests-simd` with `--ignored` on Kathleen; the 4 u\_T contract is tested exhaustively in f32 (simd-p2p.md §5.5)                                                                                                  |
| A strategy rule that depends on the node gives different bits at different launch configurations of the same input                | the rule's inputs are agreed and reported (`Fmm::strategy()`, the threading report); bit identity is required at fixed resolved strategy, and decision 5 states what may change                                              |
| Kathleen's cache hierarchy (two sockets, 1 MB L2 per core, 27.5 MB L3) gives a different crossover from the M3 Max and Grace      | the rule is measured on all three machines (requirement 7) and may name the machine class                                                                                                                                    |
| Inter-node numbers at 2 nodes (80 ranks) say little about 1,000+ ranks                                                            | T2 is a first look for the scale-out design; beyond 2 nodes the evidence is oversubscribed locust runs (correctness, memory, counts) and models, marked as such (distributed-fmm.md §14.5, "A cluster may not be available") |


## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase5n/T<k>-<name>.md and do that task." Review and merge before the next.
Commands on Kathleen need `ssh`, so the session runs them outside the sandbox.

## Exit checklist

- [x] T1's queue probe reported; Kathleen usable or not (decision 0: usable, every job at most 2 nodes for now; the user, 2026-10-10; docs/phase5n/kathleen-probe.md)
- [x] Phase order and names: 5N, then 5S, then 6 (decision 1)
- [x] Kathleen as the third machine; root, layout and quota use (decision 2: everything under `/scratch/scratch/ucahtbe/fmm`, at most two source trees with their `target/`, about 20 GB each, under 50 GB of the 250 GB quota in all; the user, 2026-10-10; tools/kathleen/README.md)
- [x] Node-hour budget (decision 3: as proposed, T1 at most 20 node-hours, T2 at most 100, T3 at most 50 on Kathleen, T4 at most 30, 200 for the phase; every job at most 2 nodes; each report states its use; the user, 2026-10-10. Used: T1 1.54, T2 7.31, fmm-validate/results/phase5n-kathleen.md §13)
- [x] T1 merged: the probe report; with Kathleen, `tools/kathleen/`, the environment rebuilt from the repository, every CPU-side check passing on Kathleen and the MPI tests at 2 nodes (without Kathleen: not applicable)
- [x] Launcher and binding on Kathleen (decision 4: `mpirun` inside the allocation, `--bind-to core` with an explicit `--map-by` (`core`, or `ppr:<r>:socket:PE=<t>` with threads); `srun --mpi=pmix --cpu-bind=cores` the tested alternative; the user, 2026-10-10; tools/kathleen/README.md)
- [x] T2 merged: `fmm-validate/results/phase5n-kathleen.md` published (without Kathleen: skipped)
- [ ] T3 merged: `docs/design/node-m2l.md` drafted, sign-off questions listed
- [ ] Strategy rule (decision 5), shared tables (decision 6), table cache (decision 7) and margin (decision 8) signed off
- [ ] T4 merged: the rule built; before/after on the three machines; Phase 5 tests pass
- [ ] Design documents updated: laplace-fmm-plan §7 (Phase 5N status), §8.3, §9.1, §9.2; distributed-fmm.md §15 note; workspace-structure §6; node-m2l.md outcome

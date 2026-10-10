# tools/kathleen: the build and test environment on Kathleen

Kathleen (`ssh kathleen`, login nodes `login21`/`login22.kathleen.ucl.ac.uk`) is UCL's
Slurm cluster: 190 diskless compute nodes, each with two Intel Xeon Gold 6248 (Cascade
Lake, 2 × 20 cores, AVX-512 and AVX2), 192 GB, Intel Omni-Path between them; RHEL 9.6.
machine.md has the measured facts. These files build the workspace's environment there
from the repository (Phase 5N T1, docs/phase5n/T1-kathleen-environment.md) and let a
session on the M3 Max drive builds, tests and runs, all as Slurm jobs. UCL's
documentation: https://www.rc.ucl.ac.uk/docs/Clusters/Kathleen/.

**Everything lives under `/scratch/scratch/ucahtbe/fmm`** (the root; `~/Scratch` points
to `/scratch/scratch/ucahtbe`). Nothing goes into the home directory. **No job is larger
than 2 nodes (80 cores)** for now (docs/phase5n/README.md, "Working on Kathleen").

## Files

| File | Runs on | Does |
| --- | --- | --- |
| `probe.sh` | M3 Max / login node | the queue probe of T1 step 0 (docs/phase5n/kathleen-probe.md); writes only `~/Scratch/fmm-probe/` |
| `setup.sh` | login node | from nothing to a working environment (bash): the layout, the root marker, a check that every pinned module loads, rustup and the pinned toolchain. Downloads only; idempotent |
| `env.sh` | login node, jobs | sourced from bash: module purge and the pinned modules, `HOME` and every cache moved under the root, the Rust toolchain on `PATH`, `LIBCLANG_PATH`, `CC`/`CXX`, the BLAS thread and stack variables, `TMPDIR` and Open MPI's session directory; prints a summary line |
| `sync.sh` | M3 Max | copies the working tree (uncommitted work included) to `kathleen:/scratch/scratch/ucahtbe/fmm/src/<branch>` (slashes in the branch become `_`) |
| `remote.sh` | M3 Max | runs a short command in that copy on the login node, inside `env.sh` |
| `submit.sh` | M3 Max | submits a job script from that copy, waits for it (polling `sacct`), prints its times, wait, state and the output's tail |
| `jobs/lib.sh` | jobs | shared by the job scripts: `env.sh`, the per-job `TMPDIR` on every node, the batch shell bound to the physical cores, `step` (timed, recorded, carries on), the summary table at exit |
| `jobs/build.sbatch` | 1 node, `singlenode` | fetch, release build, `cargo test -p nd-fmm-math`, every test executable, the release examples, sizes |
| `jobs/check.sbatch` | 1 node, `singlenode` | the CPU-side checks of root CLAUDE.md in groups: `root`, `strict`, `simd`, `kernels` |
| `jobs/mpi.sbatch` | 1 node `singlenode`, or 2 nodes `test` | `.github/scripts/run-mpi-tests.sh` build, then run at the given rank counts with `mpirun --bind-to core` |
| `jobs/launcher.sbatch` | 2 nodes, `test` | decision 4: `mpirun` against `srun --mpi=pmix` at 2 × 40 ranks, the bindings, the transport between and within nodes |
| `jobs/scaling.sbatch` | 2 nodes, `test` | one `scaling` launch, 2 nodes × 2 ranks × 20 threads, the N = 10⁶ cube, f64, p = 3, `--reference --errors 8` |
| `jobs/run.sbatch` | 1 node, `singlenode` (40 tasks, so `mpirun` has slots) | each argument as one step, for anything else |
| `jobs/sweep.sbatch` | 1 node `singlenode` 6 h, or 2 nodes `test`/`small` | sweeps of `tools/scaling/run.sh` in one allocation, and `p2p` (the x86 P2P examples); one file per launch in `logs/scaling-<job id>/` (Phase 5N T2; "Scaling sweeps") |
| `check-root.sh` | login node | lists what was written outside the root since the setup |
| `machine.md` | | the machine's facts, measured, and what the environment resolved to |

The scripts run by `sh` are POSIX `sh`; `setup.sh`, `env.sh` and the job scripts are bash
(Kathleen's `module` is a bash function, and job scripts start with `#!/bin/bash -l`).

## Pins

| What | Pin | Where |
| --- | --- | --- |
| Rust | 1.99.0 with rustfmt and clippy, through rustup (the M3 Max's and CI's stable) | `env.sh` (`FMM_KATHLEEN_RUST`, exported as `RUSTUP_TOOLCHAIN`) |
| modules | `ucl-stack/2026-03`, `compilers/gcc/12.3.0/gcc-12.3.0`, `mpi/openmpi/4.1.6/gcc-12.3.0`, `openblas/0.3.28/gcc-12.3.0`, `fftw/3.3.10/gcc-12.3.0`, `cmake/3.30.5/gcc-12.3.0`, `llvm/17.0.6/gcc-12.3.0-zzkohqr` (hidden; `module avail --all`), `userscripts/2026-03` (UCL's, for `lquota`) | `env.sh` (`FMM_KATHLEEN_MODULES`) |

There is no `rust-toolchain.toml` in the repository: CI floats on stable. When CI's
stable moves, change `FMM_KATHLEEN_RUST` and rerun `setup.sh`. The stack also has
`rust/1.81.0`, too old for Rust 2024 and the workspace; it is not used.

What the build needs from the modules: Open MPI (rsmpi's `mpi-sys` finds it through
`mpicc`), libclang for bindgen (the llvm module; RHEL 9 has none in `/usr/lib64`), and
gcc and cmake for build scripts. The workspace links no BLAS or FFTW library (rlst is
built without those features, and only rlst's own tests link OpenBLAS); the `openblas`
and `fftw` modules are loaded as the brief pins them, so that a later need finds them.
The llvm module sets `CC` and `CXX` to its clang; `env.sh` sets them back to the gcc
module's `gcc` and `g++`. The gcc module adds no library path, so a binary with C++
compiled by its `g++` (CubeCL's CPU runtime in nd-fmm-kernels) loaded RHEL 9's older
`/usr/lib64/libstdc++.so.6` and failed (`GLIBCXX_3.4.30' not found`, job 238824);
`env.sh` puts the gcc module's runtime directory first on `LD_LIBRARY_PATH`. rustc links with rust-lld by default on x86_64 Linux, through
the gcc module's `cc` driver; nothing needed GNU ld.

## Layout under /scratch/scratch/ucahtbe/fmm

| Path | Holds |
| --- | --- |
| `rust/cargo/`, `rust/rustup/` | `CARGO_HOME` (with the crate registry), `RUSTUP_HOME` |
| `home/` | the stand-in `HOME`: what resolves its paths from `HOME` alone (the tracel-llvm bundle's download cache `home/.cache/tracel/`, git, the shells) stays here. Kathleen's home and Scratch share one quota, so the reason is tidiness, not space |
| `cache/`, `share/`, `config/`, `state/` | `XDG_CACHE_HOME`, `XDG_DATA_HOME` (the installed tracel-llvm bundle, `share/tracel/`), `XDG_CONFIG_HOME`, `XDG_STATE_HOME` |
| `tmp/` | `TMPDIR` on the login node |
| `src/<branch>/` | the trees `sync.sh` copies, each with its own `target/`; keep at most two |
| `logs/` | every job's output, `<job name>-<job id>.out` |
| `.root-marker` | the time of the first setup, for `check-root.sh` |

Scratch is Lustre, like home, and neither is backed up (UCL's storage documentation:
Scratch "should not be relied on for secure long-term permanent storage"); nothing here
needs a backup, since `setup.sh` and the repository recreate it. The documentation
names no purge rule for Scratch.

## Temporary files

The nodes are diskless: `/`, `/tmp` and `/dev/shm` are RAM (95 GB each, from the node's
192 GB), and Kathleen offers no `$TMPDIR` space (requesting `tmpfs` is refused). Slurm
sets `TMPDIR=/tmp` in a job. `env.sh` instead uses, inside a job,
`/dev/shm/fmm-$USER-$SLURM_JOB_ID` for cargo's and rustc's temporary files, and
`OMPI_MCA_orte_tmpdir_base=/dev/shm` for Open MPI 4.1's session directories (and so for
the shared-memory transport's backing files), never Lustre. `jobs/lib.sh` creates the
directory on every node of the job and, at exit, prints its size per node and removes
it. On the login node `TMPDIR` is the root's `tmp/`.

Measured (jobs 238823–238841): cargo and rustc leave nothing behind in `TMPDIR` (0 bytes
at every job's end; 36 KiB when a step wrote its own log there). Inside every `srun`
step Slurm sets `TMPDIR=/tmp` again, so processes that `srun` or `mpirun` start (the MPI
ranks) see `/tmp`; Slurm gives each job a private `/tmp` (`NamespaceType=namespace/tmpfs`,
`TmpFS=/tmp`), so nothing there outlives the job or meets another job. What lands there:
`srun --mpi=pmix` writes `spmix_appdir_<uid>_<job>.<step>` (60–840 bytes each); the
tests wrote nothing. `jobs/lib.sh` lists and removes what the job created in `/tmp` per
node. The session directories of Open MPI went to `/dev/shm/ompi.*` and were removed by
Open MPI itself.

## One-time setup

From the M3 Max, outside the sandbox (`ssh` reads `~/.ssh`):

```sh
tools/kathleen/sync.sh
ssh kathleen 'cd /scratch/scratch/ucahtbe/fmm/src/<branch> && bash -l tools/kathleen/setup.sh'
```

`setup.sh` runs on the login node: it only downloads and unpacks (rustup and the
toolchain: 20 seconds, 543 MB). The first build job downloads the crates and, for
nd-fmm-kernels' `cpu` feature, the x86_64 Linux tracel-llvm bundle (compute nodes reach
crates.io and GitHub).

## Daily loop

On the M3 Max, outside the sandbox:

```sh
tools/kathleen/sync.sh
tools/kathleen/submit.sh build.sbatch                      # build; nd-fmm-math's tests
tools/kathleen/submit.sh check.sbatch root                 # the root checks
tools/kathleen/submit.sh run.sbatch 'cargo test -p nd-fmm-exec'
tools/kathleen/submit.sh --qos=test --nodes=2 -- mpi.sbatch 80
tools/kathleen/remote.sh 'cargo fmt -- --check; lquota'   # login node: light only
```

`submit.sh` prints the job id at once, a line at every state change, and at the end the
wait, the run time, the state and the last 40 lines of the output, whose full path it
names. It does not cancel the job when interrupted. Options for `sbatch` (QoS, nodes,
time) go before a `--`. Kathleen never commits or pushes; work is synced, and commits
happen on the M3 Max.

Interactively on Kathleen: `cd /scratch/scratch/ucahtbe/fmm/src/<branch> && .
tools/kathleen/env.sh`, for light commands only.

## Build times and the checks

Measured in jobs on one node (40 cores, the batch shell bound to them), release and debug
as the job scripts build them:

| Job | What | Time |
| --- | --- | --- |
| 238823, cold (empty registry and `target/`) | `cargo fetch` (crates onto Lustre) | 243 s |
| | `cargo build --release` (default members) | 53 s |
| | `cargo test -p nd-fmm-math` (build and run) | 15 s |
| | `cargo test --workspace --no-run` (every test executable, debug; downloads the tracel-llvm bundle) | 131 s |
| | release examples (nd-octree's, `scaling`) | 15 s |
| | the whole job | 8 min 21 s |
| 238844, warm (unchanged tree) | the same steps | 74 s in all, 50 s of them `du` over `target/` |
| 238824, first `check.sbatch` | every clippy, `cargo doc`, the CUDA type-checks, the CPU-runtime build of nd-fmm-kernels | 8 min 55 s with the tests |

Results of the checks on Kathleen (jobs 238824, 238833, 238839, 238830, 238825, 238841):

| Check | Result |
| --- | --- |
| `cargo fmt -- --check`; `cargo clippy -- -D warnings`; `--examples`; `cargo doc --no-deps` | pass |
| `cargo test` (default members) | pass: 57 executables, 702 passed, 45 ignored (as on the M3 Max) |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| `cargo test --workspace --no-fail-fast` | 67 executables, 744 passed, 45 ignored, **2 failed**, both in the Phase 3S spike `spikes/p2p-simd` (not a default member, never built in CI): `kernel_rsqrt_within_contract` (its AVX2 f64 candidate "est+S2", the `rsqrtps` estimate and two Newton steps, reaches 289 u against 4 u: the estimate on this Intel CPU is coarser than where the spike was measured) and the `compare` example's `compare_core_on_a_small_cell` (green-kernels' AVX2 f64 gradient 1.17e-14 against the 1e-14 bound; ours 4.3e-16). nd-fmm-simd's own kernels pass (below) |
| run-tests-simd: clippy, tests, `--release -- --ignored` | pass. ISAs run: scalar, avx2 (detected avx2; avx512f present, not built). Contract: f32 every float in [1, 4), avx2 max 1.500 u (scalar 1.500 u); f64 10⁷ samples, avx2 1.000 u (scalar 1.496 u); large term tests within 4.2 u (potential) and 8.7 u (gradient) |
| run-tests-kernels: clippy `--features cpu`, check `--features cuda` (kernels, exec), CPU-runtime tests in release | pass: 32 and 73 tests passed, 2 ignored (after the `LD_LIBRARY_PATH` fix of `env.sh`, "Pins") |
| MPI test list, one node, 1, 2, 4, 8 ranks (`mpi.sbatch`) | 36 of 36 launches pass (job 238825) |
| MPI test list, 2 nodes × 40 ranks | 9 of 9 pass (job 238841; `mpi_exec` 242 s in debug) |
| `scaling`, 2 nodes × 2 ranks × 20 threads, N = 10⁶ cube, f64, p = 3, `--reference --errors 8` | pass (job 238836): φ L2 2.913e-3, ∇φ 4.673e-3 against the direct sum; 1.04e-15 relative L2 against the one-rank FMM (a random input share, so not bit for bit; within 100 u_T); overlap on and off identical in every bit |

## The login-node rule

UCL: "Very short (<15mins) and non-resource-intensive software tests can be run on the
login nodes, but anything more should be submitted as a job." Here: syncing, `sbatch`,
`squeue`, `sacct`, reading output, `cargo fmt`, `cargo metadata`, `module`, `lquota`,
and `setup.sh` (downloads). Every build, test and run is a job.

## Jobs and QoS

Whole nodes only: jobs never share nodes, and `--hint=nomultithread` is the default (40
cores per node, no hyperthreads). From `sacctmgr show qos` (2026-10-10):

| QoS | Nodes | Wall time | Per user | Priority | Use here |
| --- | --- | --- | --- | --- | --- |
| `test` | exactly 2 | 1 h | 2 jobs | 1000 | 2-node MPI tests and runs |
| `singlenode` | 1 | 6 h | 2 jobs | 0 | builds, checks, one-node tests and runs |
| `small` | 2–6 | 48 h | | 100 | 2-node runs longer than an hour (≤ 2 nodes here) |
| `medium` | 7–12 | 24 h | | 200 | not used |
| `large` | 13+ | 12 h | | 400 | not used |

`test` refuses a 1-node job (`QOSMinNode`). Waits measured on Saturday 2026-10-10,
14:59–16:10 (from `sacct`, Start − Submit):

| QoS | Jobs | Waits |
| --- | --- | --- |
| `singlenode` (1 node) | 9 | 1–2 s for eight; 5 min 38 s (238825), when no node was idle; median 1 s |
| `test` (2 nodes) | 6 | 2 min 24 s, 3 min 43 s, 3 min 43 s, 4 min 10 s, 3 min 32 s, 4 min 5 s; median 3 min 43 s |
| `small` (4 nodes, probe only) | 1 | still pending after 7 min 34 s, cancelled (docs/phase5n/kathleen-probe.md) |

A job's binding: the batch shell is bound to the 40 physical cores by `jobs/lib.sh`
(Slurm gives it all 80 hardware threads); MPI ranks are bound per core by the launcher
(decision 4, below).

T1 used 1.54 node-hours in 16 jobs (238815–238845, the probe's included; `sacct -X
--format=JobID,NNodes,ElapsedRaw`, nodes × elapsed), of the 20 that decision 3 proposes.

Cancel a job: `ssh kathleen scancel <id>` (all of them: `scancel -u $USER`).
Node-hours of the jobs: `ssh kathleen 'sacct -X -S <date> --format=JobID,JobName,NNodes,Elapsed,State'`.

## MPI and the launcher

Open MPI 4.1.6 from the module, MPI initialised at `Funneled` where threads are used (the
`scaling` example's report says so). Measured on 2 nodes in `test` (jobs 238826 and
238840, `jobs/launcher.sbatch`):

| | `mpirun` inside the allocation | `srun --mpi=pmix` |
| --- | --- | --- |
| works | yes (Open MPI's `plm/slurm` starts its daemons through `srun`) | yes (`pmix_v3`; Open MPI's `pmix/ext3x`) |
| start-up and run of `test_mpi_global_bounding_box`, 2 × 40 ranks, three launches | 0.908, 0.894, 0.904 s | 1.284, 1.290, 1.328 s |
| default placement with one rank per core | by core: ranks 0–19 socket 0, 20–39 socket 1 of node 1, 40–79 node 2 (`--map-by core --bind-to core`) | cyclic over the sockets within a node: task 0 core 0, task 1 core 20, … (`--cpu-bind=cores`) |
| binding printed by | `--report-bindings` | `--cpu-bind=verbose,cores` |

The transport, from the selection messages (`OMPI_MCA_pml_base_verbose=10`,
`OMPI_MCA_mtl_base_verbose=10`): PML `cm` with MTL **`psm2`** (Omni-Path's native PSM2,
priority 40 over `ofi` at 25), both between the nodes and within one node, and with
either launcher; `ob1` with the `vader` BTL initialises but is not selected. Nothing uses
TCP. The module sets `OMPI_MCA_btl=self,vader`.

**Decision 4 (signed off by the user, 2026-10-10): `mpirun` inside the allocation**,
with `--bind-to core` and the placement explicit: `--map-by core` for one thread per
rank, `--map-by ppr:<ranks per socket>:socket:PE=<threads> --bind-to core` for ranks with
threads (for example `ppr:1:socket:PE=20`, one rank per socket, as `jobs/scaling.sbatch`
runs). Reasons: it starts 0.4 s faster at 80 ranks; `.github/scripts/run-mpi-tests.sh`
and `tools/scaling/run.sh` launch with `mpirun` already; its mapping names the ranks ×
threads splits of docs/phase5n/README.md directly; and it reports the binding in one
line per rank. `srun --mpi=pmix --cpu-bind=cores` works as well and is what UCL's
examples use; T2 may compare the two under load.

**MPI singletons in a job.** In the batch step Slurm 26.05 sets `SLURM_STEP_ID=-5`, and
Open MPI 4.1 then takes a singleton (a test executable that `cargo test` starts) for a
process launched by `srun`, and aborts in `MPI_Init` ("OMPI was not built with SLURM's
PMI support"). `env.sh` unsets `SLURM_STEP_ID` and `SLURM_STEPID` in the batch step
only; `srun` steps and `mpirun`'s daemons get their own. With it, `cargo test` passes in
a job (job 238833). `mpirun` and `srun` launches work either way.

**Timeouts.** `jobs/mpi.sbatch` runs every launch under `timeout`, 900 s by default:
`mpi_exec` at 80 ranks in debug took 242 s in one job and more than CI's 300 s in
another.

## Scaling sweeps

Phase 5N T2 (docs/phase5n/T2-kathleen-baseline.md, the report
`fmm-validate/results/phase5n-kathleen.md`). `jobs/sweep.sbatch` runs sweeps of
`tools/scaling/run.sh` inside one allocation, one step per sweep, with every launch under
`timeout`; under Slurm, `run.sh` maps the ranks evenly over the job's nodes and each
node's two sockets (`mpirun --map-by ppr:<ranks per socket>:socket:PE=<threads>
--bind-to core`, or `ppr:<ranks per node>:node` when the ranks per node are odd), and
writes in each file, in place of the load check, the job id, QoS, node list and mapping;
mpirun's `--report-bindings` lines and the report's "Placement per rank" give the binding.
A core binding includes both hardware threads of the core (`Cpus_allowed_list` "0,40"):
a rank's threads are placed by the kernel, one per core while cores are idle.

```sh
tools/kathleen/submit.sh sweep.sbatch strong threads p2p          # one node, ~2.5 h
tools/kathleen/submit.sh sweep.sbatch strategy check host         # one node, ~2 h
tools/kathleen/submit.sh sweep.sbatch nodes-strong nodes-check    # one node, N = 10^7
tools/kathleen/submit.sh --qos=test --nodes=2 '--export=ALL,SPLITS=40x1' -- sweep.sbatch nodes-strong
tools/kathleen/submit.sh --qos=test --nodes=2 -- sweep.sbatch nodes-weak nodes-check
```

`singlenode` and `test` take two jobs per user each, so two one-node and two two-node
sweeps run at once. The output stays in the root's `logs/`, never in a tree; the report
quotes what it needs.

## Quota

`lquota` shows the 250 GB shared by home and Scratch. Measured on 2026-10-10 after the
jobs of T1:

| What | Size |
| --- | --- |
| `rust/rustup/` (Rust 1.99.0, minimal profile, rustfmt, clippy) | 617 MB |
| `rust/cargo/` (the crate registry, 907 MB, and rustup's binaries) | 929 MB |
| the tracel-llvm bundle: download `home/.cache/tracel/` and installed `share/tracel/` | 44 MB and 320 MB |
| one `target/` after every job of T1 (debug and release, the kernels' CPU runtime, the CUDA type-check) | 20 GB |
| `logs/` | 1.2 MB |
| **`lquota`** | **21.85 GiB of 250 GiB** (from 100 KiB before) |

Decision 2 (signed off by the user, 2026-10-10): at most two trees with their `target/`
(about 20 GB each), plus about 2 GB of toolchain and caches, so under 50 GB of the
250 GB; the rest stays free for T2's and T3's outputs (raw output is kept in `logs/`,
never in a tree).

## Checking for writes outside the root

```sh
tools/kathleen/remote.sh tools/kathleen/check-root.sh
```

lists what under the real home and the real Scratch (outside the root and the probe's
`fmm-probe/`), and what this user owns in the login node's `/tmp`, `/var/tmp` and
`/dev/shm`, is newer than `.root-marker`. Expected: nothing, or what an interactive login
writes itself.

## Recreating the environment into another root

```sh
ssh kathleen 'cd /scratch/scratch/ucahtbe/fmm/src/<branch> &&
    FMM_KATHLEEN_ROOT=/scratch/scratch/ucahtbe/<other> bash -l tools/kathleen/setup.sh'
```

`setup.sh` refuses a root outside `/scratch/scratch/ucahtbe`. `env.sh` honours
`FMM_KATHLEEN_ROOT` in a shell on the login node (`FMM_KATHLEEN_ROOT=<other> .
tools/kathleen/env.sh`, then `rustc --version`); `sbatch` passes the submitting shell's
environment to the job (`--export=ALL`, the default), so a job submitted from such a
shell uses that root too. `submit.sh` always submits with the default root.

## Removing everything

Cancel any job first (`scancel -u $USER`), then on the login node:

```sh
rm -rf /scratch/scratch/ucahtbe/fmm /scratch/scratch/ucahtbe/fmm-probe
```

Nothing else on the machine was changed: no system software, no files in the home
directory.

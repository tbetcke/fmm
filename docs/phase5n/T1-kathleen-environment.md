# Phase 5N / T1 — the Kathleen queue probe, then a reproducible environment (C5N.1; decision 0)

Kathleen is UCL's Slurm cluster (README, "Machines"): 190 diskless nodes, each with two
Intel Xeon Gold 6248 (Cascade Lake, 40 cores, AVX-512 and AVX2), Intel Omni-Path between
them. Nothing from this workspace has been built there: no Rust, no build, no job. The
queue was busy at the probe (about 2,000 jobs pending), and nobody knows yet how long a
multi-node job waits. This task therefore has two steps:

- **Step 0, the queue probe.** Trivial jobs only, then a report, then **stop** for the
  user's decision 0 ("Kathleen usable?", README). Nothing is installed or built before
  that decision.
- **Step 1, the environment** (only if decision 0 says Kathleen is usable). It:
  - sets Kathleen up from files kept in the repository, so that the environment can be
    recreated by one script;
  - establishes how a Claude Code session on the M3 Max drives builds, tests and Slurm
    jobs there, as `tools/gh200/` does for locust;
  - runs every existing CPU-side check there, in jobs, on one node and across two.

Step 1 is the Phase 5N counterpart of Phase 4S T1 (docs/phase4s/T1-locust-environment.md).
Timings, scaling and the strategy study are T2 and T3; this task records no timing beyond
the probe's queue times and the build times.

**No Kathleen job in this task, or in this phase, is larger than 2 nodes (80 cores), for
now** (README, "Working on Kathleen"; the user's rule from 2026-10-10, which replaced the
earlier cap of 4 nodes while the probe ran).

## Step 0 — the queue probe (report and stop)

Measure how long Kathleen's queue makes small 2-node jobs wait, before anything is set
up. Run from the M3 Max over `ssh kathleen` (outside the sandbox), on the login node
only for submitting and reading.

- **Where.** Job scripts and their output under `~/Scratch/fmm-probe/` only: the probe
  writes nothing else on Kathleen (no toolchain, no clone, no cache). The job scripts are
  kept in the repository as `tools/kathleen/probe.sh` (one script that writes the two job
  files, submits them and collects `sacct`; POSIX `sh`, header comment), so the probe can
  be repeated later.
- **The jobs.** Each runs `srun hostname` (one line per node) and `sleep 30`, nothing else,
  with `--time=00:05:00`, `--ntasks-per-node=1`, `--exclusive` implied, on 2 nodes in the
  `test` QoS (at most 2 such jobs at a time, by the QoS). Never more than the QoS allows at
  once; cancel any job still pending after 12 hours, recording that.
  - As run on 2026-10-10: the probe first submitted pairs of a 2-node `test` job and a
    4-node `small` job, to repeat every 2.5 hours over a day. After the first pair (the
    2-node job waited 2 min 24 s) the user stopped the repeats and capped every job at 2
    nodes; the 4-node job, still pending after 7 minutes, was cancelled and is reported
    as cancelled, not as a wait. `probe.sh` now submits only the 2-node shape; its
    `drive` loop repeats it, for a later probe.
- **The estimates**, at each submission: `sbatch --test-only` for both shapes (the
  scheduler's predicted start), `squeue --start -j <id>` once the job is pending, and the
  queue depth by QoS (`squeue -h -t PD -o %q | sort | uniq -c`, and the running jobs and
  idle nodes from `sinfo -s`).
- **The record**, from `sacct -X -j <ids> --format=JobID,QOS,NNodes,Submit,Start,End,
  Elapsed,State,NodeList`: per job the wait (Start − Submit), the run time and the state;
  per shape the minimum, median and maximum wait; and how the estimates compared with the
  actual starts.
- **The report** (`docs/phase5n/kathleen-probe.md`, short; or the PR description and a
  section of `tools/kathleen/README.md`, stated): the table of jobs, the waits per shape
  and time of day, the queue depth at each submission, the node-hours used (trivial: about
  2 × 5 minutes × nodes per job), what the waits mean for the phase (T1's build and test
  jobs, T2's sweeps at 1 and 2 nodes, T3's runs, Phase 5S), and a recommendation for decision
  0 with its reason. Then **stop**: open the PR with the probe script and the report, and
  ask the user for decision 0. Do not start Step 1 in the same session unless the user has
  answered.

Without Kathleen (decision 0: proceed without it), T1 ends here: the PR contains the
probe script and the report, root `CLAUDE.md` gains one line that Kathleen was probed and
deferred (with the date and the report's path), and the README's checklist records the
decision. The rest of this brief waits until Kathleen is revisited (a new probe first).

## Step 1 — the environment (only if Kathleen is usable)

**Constraints:**
- Everything this task installs, downloads, builds or writes on Kathleen lives under one
  root that you choose and record (decision 2), inside the 250 GB quota that home and
  Scratch share. Propose the root with its reason: for example `~/Scratch/fmm-env` for
  the toolchain and caches and `~/Scratch/fmm/<branch>` for source copies and their
  `target/` (Scratch is Lustre, like home; check the UCL documentation for backup and
  purge rules of each, and say what you found).
- Builds, tests and runs go in Slurm jobs. On the login node only short (under 15
  minutes), light commands: `sync.sh`, `sbatch`, `squeue`, `sacct`, reading output, `cargo
  fmt`, `cargo metadata`, `module`. A rustup install is a download, not a build; say where
  you ran it.
- Compute nodes have no local disk and no `$TMPDIR` (the documentation says not to
  request `tmpfs`). Every temporary file goes to a place you set explicitly.

Read first:
- root CLAUDE.md ("Build environment", "Checks", "MPI");
- docs/phase5n/README.md ("Machines" with the probe table, "Design decisions": "Working
  on Kathleen", "Ranks on Kathleen"; decisions 2–4);
- tools/gh200/README.md, `setup.sh`, `env.sh`, `sync.sh`, `remote.sh`, `check-home.sh`,
  `machine.md`: the model. `sync.sh` already takes the host and the directory as
  arguments (`sync.sh [host [directory]]`), but refuses directories outside
  `/data/ucahtbe`; `remote.sh` reads `FMM_GH200_HOST` and `FMM_GH200_DIR`;
- `.github/workflows/run-tests.yml` and `.github/scripts/run-mpi-tests.sh`, for the native
  prerequisites and the MPI test list;
- fmm-kernels/CLAUDE.md ("Sandbox"): the `tracel-llvm` bundle (download cache
  `$HOME/.cache/tracel`, installed under `$XDG_DATA_HOME/tracel`);
- the UCL documentation: https://www.rc.ucl.ac.uk/docs/Clusters/Kathleen/, the Slurm pages
  (`Supplementary/Slurm/`, `Supplementary/Slurm_Example_Jobscripts/`) and the storage
  pages.

Facts (read-only probe from the M3 Max, 2026-10-10; T1 re-checks them and records them in
`tools/kathleen/machine.md` with the commands that produced them: `lscpu`, `numactl -H`,
`free -g`, `cat /etc/os-release`, `module -t avail`, `sinfo`, `sacctmgr show qos`,
`lquota`, `ompi_info`):

| Item | Value |
| --- | --- |
| Login | `ssh kathleen` reached login22.kathleen.ucl.ac.uk as ucahtbe; HOME `/home/ucahtbe` (Lustre) |
| CPU | 2 × Xeon Gold 6248 @ 2.50 GHz, 20 cores each, 2 threads per core; `avx512f` present; 2 NUMA nodes; L3 55 MiB in 2 instances |
| Memory | 188 GB visible per node |
| OS | RHEL 9.6, kernel 5.14.0-570.140.1.el9_6.x86_64 |
| Modules loaded by default | `ucl-stack/2026-03`, `compilers/gcc/12.3.0`, `mpi/openmpi/4.1.6/gcc-12.3.0`, `cmake/3.30.5`, git 2.46.2, numactl 2.0.18, `default-modules/2026-03` |
| Other modules | `mpi/openmpi/4.1.8/gcc-12.3.0`, Intel oneAPI MPI 2021.14, `openblas/0.3.28/gcc-12.3.0` (and `-omp`), `fftw/3.3.10/gcc-12.3.0`, `llvm/17.0.6/gcc-12.3.0-zzkohqr` (hidden), `python/3.11.6`, `intel-oneapi-mkl/2023.2.0`; no Rust |
| libclang | not in `/usr/lib64`; from the `llvm/17.0.6` module (for rsmpi's bindgen) |
| Network from the login node | static.rust-lang.org, index.crates.io, github.com: HTTP 200 |
| Storage | `~/Scratch -> /scratch/scratch/ucahtbe`; `lquota`: 250 GB for home and Scratch together, 64 kB used |
| Slurm | partition `kathleen` (190 nodes, exclusive); QoS `test` (1 h, ≤ 2 nodes, ≤ 2 jobs), `singlenode` (6 h, 1 node, ≤ 2 jobs), `small` (48 h, ≤ 6 nodes), `medium` (24 h, ≤ 12 nodes), `large` (12 h); account `allusers` |

Do:
- **Layout under the root** (record it in `tools/kathleen/README.md`), for example:

  | Path | Holds |
  | --- | --- |
  | `rust/cargo/`, `rust/rustup/` | `CARGO_HOME`, `RUSTUP_HOME` |
  | `cache/`, `share/`, `config/` | `XDG_CACHE_HOME`, `XDG_DATA_HOME` (the `tracel-llvm` bundle), `XDG_CONFIG_HOME` |
  | `home/` | a stand-in `HOME` for build and job shells, if you decide the real home must stay clean (say whether: on Kathleen home and Scratch share one quota, so the reason is tidiness, not space) |
  | `tmp/` | `TMPDIR` for builds on the login node and in jobs, unless `/dev/shm` is chosen for jobs |
  | `fmm/<branch>/` | the source copies from `sync.sh`, with their `target/` |
  | `logs/` | job output (`#SBATCH --output`) |

- **Modules, pinned.** `env.sh` purges and loads an explicit list with full versions:
  `ucl-stack/2026-03`, gcc 12.3.0, `mpi/openmpi/4.1.6/gcc-12.3.0`,
  `openblas/0.3.28/gcc-12.3.0`, `fftw/3.3.10/gcc-12.3.0`, `cmake/3.30.5`,
  `llvm/17.0.6/gcc-12.3.0-zzkohqr` (or what `module avail --all` shows), and whatever
  else the build needs (`pkgconf`, `python`). It sets `LIBCLANG_PATH` from the llvm
  module, `PKG_CONFIG_PATH` if needed, every BLAS thread variable to 1,
  `RUST_MIN_STACK=8388608`, `TMPDIR`, Open MPI's session directory (for 4.1:
  `OMPI_MCA_orte_tmpdir_base`; check `ompi_info --all`), and prints a one-line summary
  (modules, rustc, mpirun version, transport). Sourcing it twice is harmless; it works in
  a login shell and inside a job.
- **Rust.** rustup into the root, the toolchain pinned in `env.sh` at the M3 Max's
  `rustc --version` and CI's stable (as `tools/gh200/env.sh` pins it), with `rustfmt` and
  `clippy`. No `rust-toolchain.toml` in the repository. rustc links with rust-lld by
  default on x86_64 Linux; say whether anything needed GNU ld instead.
- **Scripts** in `tools/kathleen/`, POSIX `sh` (or `bash`, stated), idempotent, each with
  a header comment:
  - `setup.sh`: from nothing to a working environment (layout, rustup, toolchain), run
    once on the login node or in a short job (say which); refuses a root outside the one
    recorded;
  - `env.sh`, sourced, as above;
  - `sync.sh` and `remote.sh`, run on the M3 Max: either thin wrappers that call
    `tools/gh200/sync.sh` with Kathleen's host and directory (then lift its
    `/data/ucahtbe` check into a per-host rule without changing locust's behaviour), or
    their own copies. Do not break `tools/gh200/`: its README's commands must work as
    before;
  - job scripts (`#SBATCH` templates, one file each, with `module purge` and `. env.sh`
    at the top): a build job (`cargo build --release` and the test executables, on one
    node, `singlenode` or `test` QoS), a check job (the root checks and the stricter
    workspace checks), and an MPI job that runs `.github/scripts/run-mpi-tests.sh run
    <list> <ranks…>` with the launcher of decision 4;
  - a submit helper, run on the M3 Max through `ssh`: submits a job script, waits for it
    (polling `squeue` or `sacct` at a modest interval, or `sbatch --wait`), and prints the
    output file's path and the job's exit state, so a session drives a job in one
    command;
  - `check-root.sh` or an equivalent, if you keep the real home clean: lists what was
    written outside the root since the setup.
- **The launcher (decision 4).** Measure on 2 nodes in the `test` QoS: `srun` (does the
  Open MPI 4.1.6 module support PMIx launch under this Slurm? `srun --mpi=list`) against
  `mpirun` inside the allocation; the binding (`srun --cpu-bind=verbose,cores` or `mpirun
  --report-bindings --bind-to core --map-by ppr:20:socket`); which transport Open MPI
  selects between nodes (`ompi_info`, `OMPI_MCA_pml`/`mtl` verbosity: PSM2, OFI or TCP;
  TCP between nodes would be a defect to fix in `env.sh`) and within a node (vader/sm).
  Propose one launcher and its flags; record the measured start-up time at 2 × 40 ranks.
- **tools/kathleen/README.md:** what each file does, the one-time setup, the daily loop
  (sync, submit a build job, submit a test or run job, read the output), the login-node
  rule, the quota (how to check it, the sizes measured: toolchain, bundle, one `target/`
  with debug and release builds), the QoS table and which to use for what, how to cancel
  a job, and how to remove everything.
- **tools/kathleen/machine.md:** the facts table above re-measured, plus what the
  environment resolved to (rustc, cargo, Open MPI's `ompi_info` summary, OpenBLAS, the
  transport) and the cache sizes per core (`lscpu -C`: L1, L2 and L3).
- **Run the CPU-side checks on Kathleen**, each in a job, from `env.sh`; report each
  result, any failure with its output, and the cold and warm build times:
  - the root checks and the stricter workspace checks (`cargo fmt -- --check`, every
    clippy, `RUST_MIN_STACK=8388608 cargo test` and `cargo test --workspace` with the
    module's Open MPI, `cargo doc --no-deps`);
  - the `run-tests-simd` commands, `--release -- --ignored` included: this is **the first
    Intel CPU to run `nd-fmm-simd`** (AVX2 + FMA and scalar; AVX-512 is not built,
    simd-p2p.md §4.7). Report the ISAs the tests print and the 4 u_T contract results;
  - the `run-tests-kernels` commands (the CubeCL CPU runtime on 40 cores). The first build
    downloads the x86_64 Linux `tracel-llvm` bundle; report where it landed and its size;
  - the MPI test list (`.github/scripts/run-mpi-tests.sh build`, then `run <list> 1 2 4 8`)
    on one node, and the same list at 2 nodes × 40 ranks (80) in the `test` QoS (no job
    above 2 nodes), under
    `timeout`, with the launcher of decision 4. Report every executable's result and time;
  - one `nd-fmm-validate` `scaling` launch at 2 nodes × 2 ranks × 20 threads on the N =
    10⁶ cube, f64, p = 3, with `--reference` and `--errors 8`, to show threads, binding
    and the errors across nodes (its timings are not reported as results; T2 times).
- **Root CLAUDE.md:**
  - "Source of truth": the current phase is docs/phase5n/README.md (Phase 5N), with
    Phase 5 and the older phases kept listed;
  - "Build environment": a paragraph on Kathleen and `tools/kathleen/` (the login-node
    rule, jobs, the quota, `ssh` outside the sandbox);
  - "Checks": a short "On Kathleen (by hand, in jobs)" block, as the locust block.
- Add any new generated file (for example a `.source-revision` variant) to `.gitignore`.

Tests that define done:
- Step 0: the probe's report with every job's submit, start and end times from `sacct`,
  the waits per shape, the estimates and the queue depth; the PR opened and decision 0
  asked for. Without Kathleen, this is all of T1.
- Step 1 (with Kathleen):
  - From a fresh login shell on Kathleen, `. tools/kathleen/env.sh` then a build job of
    `cargo test -p nd-fmm-math` passes, and the checks listed above pass in jobs (report any
    that cannot run, with the reason).
  - Recreating the environment into an empty second root with `setup.sh` gives the same
    toolchain and module versions. Reported; remove the second root afterwards.
  - The MPI test list passes at 1, 2, 4 and 8 ranks on one node and at 2 × 40 across two
    nodes.
  - `lquota` before and after, reported; nothing written outside the root (or the list of
    what was, with the reason).

Must pass:
- on the M3 Max: the root checks (this task changes only `tools/`, docs, `.gitignore` and
  the root CLAUDE.md), and `tools/gh200/sync.sh` and `remote.sh` still work against locust
  as before (one `remote.sh 'rustc --version'`);
- on Kathleen: the checks listed under "Run the CPU-side checks", in jobs.

Do not:
- install, clone or build anything on Kathleen before decision 0, or write outside
  `~/Scratch/fmm-probe/` during Step 0;
- submit any job larger than 2 nodes (80 cores), in Step 0 or Step 1;
- build or run tests on the login node beyond short, light commands; request `tmpfs`; use
  hyperthreading;
- write outside the recorded root on Kathleen, install system software, or store
  credentials there; Kathleen never commits or pushes;
- change any crate, `Cargo.toml` or `Cargo.lock`, or the CI workflows;
- spend more than the node-hours decision 3 gives this task (proposed: 20); report the
  node-hours used (`sacct -X --format=JobID,NNodes,Elapsed`).

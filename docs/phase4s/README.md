# Phase 4S: CUDA on NVIDIA Grace Hopper

Phase 4S runs the device path of Phase 4 on an NVIDIA GPU. Phase 4 built a device path
that is generic over CubeCL runtimes, but ran only two backends: Metal (f32, timed, on
the Apple M3 Max) and the CubeCL CPU runtime (f32 and f64, correctness). CUDA was
type-checked and never run. f64 has never run on a GPU here, and the f64 dense/rotation
rule is provisional for that reason (Phase 4, decision 13). The phase sits between
Phase 4 and Phase 5. It is called 4S rather than 5 so that the phase and component
numbers the code and documents already cite (Phase 5, C5.x) keep their meaning, as
Phase 3S did.

A GPU is now available: **locust**, a GH200 node (`ssh locust`; "Machines"). The phase
has three goals:
1. **Environment.** A reproducible build and test environment on locust, set up from
   files kept in the repository (a spack environment and scripts). Everything on the
   machine lives under `/data/ucahtbe`, never in the home directory.
2. **CUDA, correct and then fast.** The kernels and the device FMM run on the H100 in
   f32 and f64 and pass the Phase 4 gates there. The device arithmetic rules of
   CONVENTIONS §3.13 are checked on CUDA. Layouts and the static rules are tuned for
   Hopper where measurement shows a gain. Metal and the CPU runtime keep their results
   bit for bit.
3. **A benchmark anyone can run.** One command:
   - N random points in the unit cube (default 10⁶), f32 or f64, a chosen expansion
     degree, a chosen backend;
   - the overall evaluation time as min, max and mean, and the time per operator kind
     (P2P, M2L, …);
   - written as a Markdown table.

The phase has four parts:
1. **Environment** (T1). The spack environment, the Rust toolchain, the scripts that keep
   everything under `/data/ucahtbe`, the sync-and-run loop from the M3 Max, and the
   existing CPU-side checks run on Grace.
2. **CUDA correctness** (T2–T4). The kernel suite on CUDA; the device-arithmetic spike on
   CUDA with a sign-off on the §3.13 rules; the device FMM gates on CUDA in both
   precisions.
3. **Timing and benchmark** (T5, T6). Timings per operator kind in `nd-fmm-exec`, and the
   new crate `nd-fmm-bench`. Both can be developed on the M3 Max in parallel with
   part 2.
4. **Tuning and measurement** (T7, T8). CUDA layouts and tuner candidates, the static
   rules on CUDA, the GH200 benchmark report and the design-document update.

Companion documents:
- [docs/design/device-path.md](../design/device-path.md): all of it, especially §1.2
  (F2, F12–F14, F17, F21), §3, §5, §6, §8, §10, §13 and §17. It stays the design of the
  device path. T2, T3 and T7 add a §18, "CUDA on Grace Hopper";
- [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md): §6, §7 (Phase 4),
  §8.3 and §9;
- docs/phase4/README.md: its requirements, accuracy measures and exit gate are this
  phase's yardstick;
- spikes/device-arith/REPORT.md and spikes/cubecl-gemm/SPIKE_REPORT.md;
- fmm-validate/results/phase4-m3max.md (§4, "The CUDA run").

No new design document. The phase adds no new structure to the device path. It runs the
existing design on a new backend, adds an opt-in measurement, and adds a tool. T3's
report and T7's measurements, signed off, are recorded in device-path.md §18.

Prerequisite: Phase 4 is complete (T13 merged, PR #59), and the Phase 5 briefs are
merged (PR #60). Phase 5 starts after Phase 4S. Phase 5 tests on both the M3 Max and
locust (decision 8). T8 updates the Phase 5 briefs to match.

## Scope

In scope:
- `tools/gh200/`: the spack environment (`spack.yaml`, `spack.lock`), the setup,
  environment, sync, remote-run and home-check scripts, the machine facts (T1).
- `tools/bench/run.sh`: the one-command benchmark (T6).
- `nd-fmm-kernels`:
  - a CUDA arm in the test harness and every kernel test on CUDA (T2);
  - fixes for defects that CUDA exposes (T2, T4);
  - CUDA-specific layouts and limits, where measured better (T7).
- `nd-fmm-exec`:
  - CUDA test executables and blocks beside the Metal ones (T4);
  - `FmmBuilder::kind_timings` and `StageTimings::kinds` (T5);
  - CUDA tuner candidates and a backend-keyed static rule (T7).
- `nd-fmm-validate`:
  - the machine lines on Linux aarch64 (T4);
  - per-device peaks in the kernel examples (T7).
- `nd-fmm-bench` (new crate, `fmm-bench/`): the benchmark binary and its library (T6).
- `spikes/device-arith`, `spikes/cubecl-gemm`: their CUDA runs and reports (T2, T3).
- CONVENTIONS §3.13, "Device kernels": a CUDA note, only if T3 shows one is needed and
  the sign-off accepts it.
- Root CLAUDE.md: Phase 4S as the current phase, locust's build environment and checks
  (T1, T2, T4); `nd-fmm-bench` (T6).
- `.github/workflows/run-tests.yml`: a `--features cuda` type-check step in
  `run-tests-kernels` (decision 6, T2).

Out of scope:
- Multi-GPU, multi-node and multi-rank device runs (Phase 5 and later). locust has one
  GPU.
- Grace-specific host work: SVE/SVE2 kernels in `nd-fmm-simd` (NEON runs on Grace as it
  is), BLAS on the host path (still deferred, Phase 4 decision 6), NUMA tuning. Grace is
  used as it is, as the host baseline.
- Using the GH200's coherent memory (ATS, NVLink-C2C) to replace explicit transfers.
  The transfers are the Phase 4 minimum (charges up, output down) and are reported, not
  redesigned.
- FP64 tensor cores (CubeCL has no FP64 MMA), and TF32, F16 or BF16 arithmetic (the
  input-precision guard stays).
- Overlapping kinds on several CUDA streams (changes the accumulation order;
  device-path.md §14).
- The NVRTC compiler path (`cubecl/cuda-cpp`) in production code, unless decision 4
  changes it.
- Moving off CubeCL `=0.11.0-pre.4` (a separate decision, as in Phase 4).
- Changes to:
  - formulations, accumulation orders, tolerances or the P2P contract;
  - the host path's defaults and results;
  - `nd-fmm-math`, `nd-fmm-ref`, `nd-fmm-tables`, `nd-fmm-simd`, `nd-fmm-plan` and
    `nd-octree`, beyond fixing a defect a test exposes (stop and report first);
  - the docs/phase5 briefs, except T8's update for decision 8.
- A CUDA job in CI (no GPU runner).

## Requirements

T1–T8 are accepted against these. A task may propose changing one, with reasons, for
sign-off.

1. **Reproducible environment.** locust's environment is built by one script from files
   in the repository: a pinned spack release and package repository, `spack.yaml` with
   its `spack.lock`, and a pinned Rust toolchain. It can be rebuilt into an empty
   prefix. Every byte it writes is under `/data/ucahtbe`, and a check script shows that
   nothing went into the home directory.
2. **One device path, one more backend.** CUDA runs through the existing `Backend`/
   `BackendKind` values and the existing kernels. There is no CUDA-only kernel, no
   `R: Runtime` parameter, and no change to `Fmm`'s API beyond T5's opt-in setting.
   Per-backend choices are layouts, limits and rules looked up from `DeviceInfo`, as in
   Phase 4.
3. **Metal and the CPU runtime unchanged.** Every Phase 4 test still passes. With the
   default settings, the Metal and CPU-runtime outputs are bit-identical to before the
   phase.
4. **The Phase 4 requirements hold on CUDA**, 1–11 of docs/phase4/README.md:
   device-resident data, the accumulation rule, no atomics, determinism, precision as a
   capability (here f32 and f64 accepted), the host fallback bit for bit, `unsafe`
   confined, tested on what can run, and measured, never asserted.
5. **f64 on a GPU meets the f64 bounds.** CUDA f64 meets the f64 tolerances that
   Phase 4 applied to the CPU runtime: operators, P2P and FMM. If the arithmetic forbids
   it, T3 says so with numbers before any tolerance is discussed.
6. **Per-kind timings change nothing.** Off by default. When on, the output is
   bit-identical to when off, and the syncs added are exactly those the mode documents.
7. **The benchmark is one command** with the defaults of the user request: N = 10⁶ points
   uniform in the unit cube, f32 or f64, a chosen degree, a chosen backend. It writes a
   Markdown file with the summary and per-kind tables, and its machine header names
   everything needed to repeat the run.
8. **Honest reporting.** Every test run prints the backends it ran. A report never
   presents a backend or a check that did not run as passing. Every number is "measured
   (machine, backend)", "model" or "datasheet".

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **Working on locust.** A Claude Code session runs on the M3 Max and drives locust over
  `ssh`:
  - `tools/gh200/sync.sh` copies the working tree, and `tools/gh200/remote.sh` runs a
    command there inside the environment (T1);
  - `ssh` reads `~/.ssh`, which the Claude Code sandbox denies. Every remote command runs
    outside the sandbox and so goes through the permission prompt. Keep remote commands
    few and batched, and say when one ran;
  - work is synced, not committed. Commits and pushes happen from the M3 Max only, and
    only when asked (root CLAUDE.md, "Working agreement"). locust never pushes and holds
    no credentials (the repository is public);
  - on locust, write only under `/data/ucahtbe`. No `sudo`, no `dnf`, no files in
    `/home/ucahtbe`;
  - locust is a shared, interactive node with no scheduler. Before a timing run check
    `nvidia-smi` and `uptime`, and state the load in the report. Run long jobs under
    `timeout`, and never leave processes behind.
- **The CUDA compiler path.** CubeCL's default, LLVM to NVPTX through the `tracel-llvm`
  bundle (F17). On it `inverse_sqrt` is the accurate `1 / sqrt(x)` polyfill, and fadd,
  fsub and fmul carry LLVM's `contract` flag. NVRTC (`cubecl/cuda-cpp`) is not enabled
  (decision 4). The cudarc build reads the CUDA version from `nvcc` on the `PATH`, which
  T1's environment script provides; without it, cudarc silently assumes CUDA 13.4.
- **Precisions on CUDA.** f32 and f64 are both accepted (F12). The library matmul stays
  off for f32 on CUDA: the input-precision guard rejects TF32 stages, and the LLVM
  backend offers f16 CMMA only. The hand-written GEMM runs at every p and precision.
  This is expected, and T2 and T4 check it.
- **Defaults on CUDA.** Until T7, CUDA uses the Metal GPU defaults it already shares
  (`Metal | Cuda` arms). T7 adds CUDA arms only where measured better. Static rules
  become keyed by backend; Metal's rule and defaults do not change.
- **Tests on CUDA.** CUDA tests are `#[ignore]`d, behind the `cuda` feature, and run by
  hand on locust, as Metal tests are run by hand on the M3 Max. The CPU-runtime CI job
  and the default members are unchanged. A device test that opens CUDA prints its
  `DeviceInfo` and the closing "backends run: …; not run: …" line.
- **Accuracy measures.** Those of Phase 4 ("Accuracy measures"), with the CPU-runtime f64
  bounds applying to CUDA f64 and the f32 bounds to CUDA f32:
  - operators: 1e-14, or twice the host's error, for dense f64; 1e-13 for rotation and
    the leaf operators in f64; 1e-5 in f32;
  - P2P: the C3S.4 contract;
  - FMM: against the host output within 1e-12 (f64) and 1e-5 (f32) relative L2, and
    errors against the direct sum within 0.1% (f64) and 5% (f32) of the host run's.

  If a check fails, the task reports the measured errors and their breakdown by kind
  (host fallback per kind) and stops. It does not tune tolerances to pass.
- **Workloads.**
  - The Phase 4 problems: the uniform cube and the Plummer sphere (a = 0.1), N = 10⁵,
    `max_level` 16, 64 points per leaf, eight charge vectors, gradients, at f32 p = 3
    and 8, and f64 p = 8, 12 and 18.
  - On CUDA, N = 10⁶ (and 10⁷ in T8) as well; the GPU has 96 GB.
  - The benchmark problem: N points uniform in [0, 1]³, sources equal targets,
    gradients on, default N = 10⁶, f32, p = 6.
- **Timing** (requirement 8):
  - release builds; kernel compilation excluded (warm-up evaluations);
  - every BLAS thread variable set to 1; the GPU otherwise idle, checked and stated;
  - the overall evaluation time as min, max, mean, median and standard deviation over
    the repeats, with the number of repeats stated;
  - kind timings from a separate set of evaluations with T5's mode on, since a sync per
    call perturbs the total;
  - host comparisons on Grace at 1 thread and at 72 threads (NEON, `P2pChoice::Auto`).

  The GPU clocks are not locked (no administrator access): the report prints them
  (`nvidia-smi -q -d CLOCK`) and states that. Timings are never asserted and never taken
  in CI.
- **Peaks.** For the GH200's H100, the datasheet values (f32 and f64 non-tensor, HBM3
  bandwidth), labelled "datasheet" with the clock they assume. For the M3 Max, Phase 4's
  values. A P2P peak model for the H100 is derived in T7 as device-path.md §13.4 did for
  the M3 Max.
- **The benchmark crate** (decision 5). `fmm-bench/`, package `nd-fmm-bench`, a default
  member, `publish = false`, a binary over a small library:
  - it depends on `nd-fmm-exec`, `nd-fmm-validate` and `mpi` (workspace entries) and
    passes the backend features through;
  - no new external dependency: the command line is parsed by hand, as the
    `nd-fmm-validate` examples do;
  - its output files go to `bench-results/` (ignored by git). Phase reports are kept in
    `fmm-bench/results/`.
- **MPI.** The root rules apply. Every device run is on one rank. Device work never calls
  MPI. CUDA tests that build an `Fmm` go into their own MPI-owning executable
  (`tests/device_cuda.rs`) or into the existing CUDA blocks, one MPI test per
  executable. On Linux, `mpirun` needs no loopback flags.
- **Dependencies.** No new external dependency in any crate. The spack environment is
  tooling, not a crate dependency. Anything else needs asking first.

## Exit gate
- Every acceptance test in the task briefs passes:
  - in CI for the default members (now including `nd-fmm-bench`) and in the CPU-runtime
    job;
  - on the M3 Max for the host, the CPU runtime and Metal (Metal by hand);
  - on locust for the host and CUDA (by hand).

  Each is reported with the backends run.
- C4S.1 (T1): locust's environment rebuilds from the committed spack lock and pinned
  toolchain into an empty prefix. The home check is clean. The root checks, the
  `nd-fmm-simd` checks and the CPU-runtime kernel checks pass on Grace.
- C4S.2 (T2): every `nd-fmm-kernels` test passes on CUDA in f32 and f64, the GEMM spike
  passes on CUDA, and the CUDA facts are recorded (device-path.md §18.1).
- C4S.3 (T3): the device-arithmetic spike runs on CUDA in f32 and f64. The §3.13 rules
  are confirmed for CUDA, or a CUDA note is signed off.
- C4S.4 (T4): the Phase 4 device gates pass on CUDA in f32 and f64:
  - the host fallback bit for bit;
  - C4.8 on the cube, the Plummer sphere and the clusters at N = 10⁵, and the cube at
    N = 10⁶;
  - determinism and transfers;
  - the tuner;
  - the ignored accuracy gates.
- C4S.5 (T5): per-kind timings on the host and every device backend, bit-identical
  output, documented syncs.
- C4S.6 (T6): `tools/bench/run.sh` gives the Markdown report on the M3 Max (host, Metal)
  and on locust (host, CUDA), with summary and kind tables.
- C4S.7 (T7):
  - the CUDA layouts and candidates adopted by measurement, each with the gain that
    justified it;
  - the static M2L rule on CUDA signed off (decision 9);
  - the Metal and CPU-runtime outputs unchanged bit for bit.
- The T8 report, `fmm-bench/results/phase4s-gh200.md`, gives on the GH200:
  - the device FMM in f32 and f64 against Grace at 1 and 72 threads, per kind;
  - kernel efficiency against the datasheet peaks;
  - the f64 dense/rotation crossover;
  - GH200 against the M3 Max in f32;
  - what was not measured.

  The design documents carry the outcome.

## Tasks

One pull request each.
- **Start at once:** T1 (locust, `tools/`) and T5 (`nd-fmm-exec`, on the M3 Max) touch
  disjoint files.
- **After T1:** T2 and T3 can run in parallel. T2 is `nd-fmm-kernels` and the GEMM spike;
  T3 is the device-arith spike.
- **T4** needs T2, and the T3 sign-off for its f64 gates. Its f32 work can start after
  T2.
- **T6** needs T5. Its locust run needs T1, and T4 for the CUDA rows (otherwise host
  only, said so).
- **T7** needs T4, T5 and T6. It changes the static rule only after decision 9 is signed
  off.
- **T8** needs T7.
- **Overlaps.** T4, T5 and T7 all change `nd-fmm-exec`: merge one, then rebase the next.
  T2 and T7 both change `nd-fmm-kernels`. T2 changes `.github/workflows/run-tests.yml`.
  T1 and T6 change the root CLAUDE.md and `.gitignore`. T6 changes the root `Cargo.toml` and `Cargo.lock` (a new member);
  regenerate the lock file with the manifest change.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-locust-environment.md](T1-locust-environment.md) | `tools/gh200/`: spack environment and lock, setup, env, sync, remote and home-check scripts, machine facts; the CPU-side checks on Grace; root `CLAUDE.md` points to Phase 4S | C4S.1 | none |
| T2 | [T2-cuda-kernels.md](T2-cuda-kernels.md) | CUDA arm of the kernel test harness; every kernel test on CUDA in f32 and f64; fixes; the GEMM spike on CUDA; device-path.md §18.1 facts; the CUDA type-check in CI (decision 6) | C4S.2 | T1 |
| T3 | [T3-cuda-arithmetic.md](T3-cuda-arithmetic.md) | `spikes/device-arith` on CUDA: rounding, contraction, sqrt/division/rsqrt, subnormals, the r² = 0 argument, the P2P contract in f64; recommendation for §3.13 | C4S.3 | T1 |
| T4 | [T4-cuda-device-fmm.md](T4-cuda-device-fmm.md) | `tests/device_cuda.rs`, the CUDA blocks of the C4.8, tuner and accuracy gates in f32 and f64; device timestamps on CUDA; Grace machine lines | C4S.4 | T2, T3 (signed off) |
| T5 | [T5-kind-timings.md](T5-kind-timings.md) | `FmmBuilder::kind_timings`, `StageTimings::kinds`: per-kind times on the host and every device backend, bit-identical, documented syncs | C4S.5 | none |
| T6 | [T6-bench-crate.md](T6-bench-crate.md) | `nd-fmm-bench` crate and `tools/bench/run.sh`: N points in the unit cube, f32/f64, degree, backend; min/max/mean and per-kind tables in Markdown | C4S.6 | T5 (T1, T4 for the locust rows) |
| T7 | [T7-hopper-tuning.md](T7-hopper-tuning.md) | CUDA layouts, limits and tuner candidates by measurement; H100 peaks and P2P model; the static M2L rule on CUDA; device-path.md §18.2 | C4S.7 | T4, T5, T6 |
| T8 | [T8-benchmarks.md](T8-benchmarks.md) | `fmm-bench/results/phase4s-gh200.md`: the GH200 report, the Phase 4 "CUDA run", the CUDA leaf-size study; design-document update | gate: benchmarks published | T7 |

Review T1's environment and T3's recommendation yourself before the tasks that build on
them. T4's f64 gates and T7's rules rest on T3.

## Machines

| Machine | Backends | Used for |
| --- | --- | --- |
| Apple M3 Max (development; 12 performance and 4 efficiency cores, 64 GB) | host (NEON); Metal (f32); CubeCL CPU runtime (f32, f64) | every task's editing, commits and M3 checks; T5 and T6 development; Metal by hand |
| **locust** (`ssh locust`, `locust.rc.ucl.ac.uk`): NVIDIA GH200 480GB, shared interactive node, no scheduler | host (NEON on 72 Neoverse-V2 cores); CUDA (f32, f64) on one H100 (sm_90, 96 GB HBM3); CubeCL CPU runtime | every CUDA test and timing; the Grace host baseline; T1–T4, T7, T8 runs |
| GitHub Actions `ubuntu-latest`, `ubuntu-24.04-arm` | host; CubeCL CPU runtime | CI as today, plus the CUDA type-check step (decision 6); never CUDA runs |

### locust, as probed read-only on 2026-10-05

| Item | Value |
| --- | --- |
| OS | RHEL 9.3, kernel 5.14.0-362 `aarch64+64k` (64 KiB pages) |
| CPU, memory | 72 × Neoverse-V2 (Grace), one thread per core; 572 GB |
| GPU | GH200 480GB (H100, compute capability 9.0), 96 GB HBM3; driver 565.57.01 (CUDA 12.7); MIG off; ATS |
| CUDA toolkits | 11.7, 12.2, 12.3, 12.4, 12.6 under `/usr/local`; `nvcc` not on `PATH` |
| System software | gcc 11.4.1, clang 17.0.6 (libclang), Open MPI 4.1.7, OpenBLAS 0.3.21, FFTW 3.3.8 (RPMs, not on `PATH`); environment modules without site modules; no Rust, no spack |
| Storage | `/data/ucahtbe` (on `/data`, 472 GB free) for everything; `/home` not to be used |
| Network | GitHub, crates.io, static.rust-lang.org and PyPI reachable |

T1 re-checks these and keeps them in tools/gh200/machine.md.

## Decisions to sign off

Each is recorded in the exit checklist when made:
1. **Rust on locust.** **Decided on 2026-10-05: spack's `rust` if spack offers the
   current stable, otherwise rustup.**
   - "Current" means the M3 Max's stable when T1 runs. CI floats on stable, and the
     local toolchain must match it before a push.
   - On 2026-10-05 the newest `rust` in spack's package repository (`develop`) was
     1.97.1 and stable was 1.99.0. Unless spack has caught up when T1 runs, T1 uses
     rustup.
   - With spack: `rust@<stable>` in `spack.yaml`, so the lock pins it.
   - With rustup: install under `/data/ucahtbe/rust`, with the version pinned in
     `tools/gh200/env.sh` (`RUSTUP_TOOLCHAIN`), plus `rustfmt` and `clippy`.
   - Either way, no `rust-toolchain.toml` goes into the repository. T1 reports which
     source it used and why.
2. **Keeping the home directory clean.** **Decided on 2026-10-05: as recommended.** Some
   tools resolve their paths from `HOME` alone: the `tracel-llvm` bundle caches in
   `$HOME/.cache/tracel`, and spack, cargo, rustup, CUDA and git also default to
   `$HOME`. So:
   - `env.sh` points `HOME` at `/data/ucahtbe/home` inside the activated shell;
   - it also sets every specific variable (`CARGO_HOME`, `RUSTUP_HOME`,
     `SPACK_USER_CONFIG_PATH`, `SPACK_USER_CACHE_PATH`, `XDG_*`, `CUDA_CACHE_PATH`,
     `TMPDIR`);
   - `check-home.sh` verifies the result.
3. **What spack builds and what it takes from the system.** **Decided on 2026-10-05: as
   recommended.**
   - Externals for the compiler (gcc 11.4.1) and libclang (clang 17.0.6).
   - Spack builds `openmpi`, `openblas threads=none`, `fftw`, `cmake`, `pkgconf` and
     `cuda@12.6`, the newest toolkit the driver's 12.7 supports. Spack builds CUDA
     rather than taking `/usr/local/cuda-12.6` as an external, for reproducibility.
   - A generic aarch64 target if gcc 11 cannot target `neoverse_v2`.
   - T1 reports the concretisation and install times.
4. **The CUDA compiler path.** **Decided on 2026-10-05: CubeCL's default, LLVM to
   NVPTX.**
   - Enabling NVRTC (`cuda-cpp`) is considered only if T3 finds a problem that NVRTC
     avoids, and then by its own sign-off.
   - NVRTC would turn `inverse_sqrt` into CUDA's approximate `rsqrt`.
5. **The benchmark's home.** **Decided on 2026-10-05: a new default-member crate
   `nd-fmm-bench`**, with `tools/bench/run.sh` as the one command.
   - It is a binary over a small library, with no new external dependencies.
   - It reuses `nd-fmm-validate`'s points, references and machine lines.
   - Confirmed: "degree of freedom" in the request is the expansion degree p
     (`--degree`), and the unit cube is [0, 1]³.
6. **A CUDA type-check in CI.** **Decided on 2026-10-05: add it.** T2 adds `cargo check
   -p nd-fmm-kernels --features cuda` and `cargo check -p nd-fmm-exec --features cuda`
   to the `run-tests-kernels` job, and measures the extra time.
   - That job already caches the `tracel-llvm` bundle that `cubecl-cuda` needs.
   - CUDA is type-checked in CI, never run.
   - Without the check, CUDA builds could silently rot between hand runs on locust.
7. **A per-backend leaf-size default.** Recommended:
   - T8 applies the Phase 4 rule on CUDA and reports it;
   - a CUDA leaf-size default is added only by a separate sign-off with T8's numbers;
   - the host default stays 64.
8. **Phase 5 hardware.** **Decided on 2026-10-05: Phase 5 tests on both the M3 Max and
   locust.** This revises Phase 5's decision 2 of 2026-10-04 (the M3 Max only).
   - locust adds 72 Grace cores for multi-rank runs and an H100 for device ranks.
   - The Phase 5 README carries a note now. T8 updates the Phase 5 briefs in full with
     what Phase 4S learned: the machines table, decision 2, requirements 9 and 10, rank
     counts and timeouts, the T3 CI note, the T8 device ranks on CUDA, and the T10
     scaling runs. Those changes are reviewed with T8.
9. **The static M2L rule on CUDA** (T7). The f64 boundary (dense to p = 11, rotation
   from 12) is provisional (Phase 4 decision 13), and the f32 rule is "dense at every
   p". T7 measures the crossover on the H100 and proposes a CUDA rule, keyed by
   backend, with Metal's unchanged. Signed off with T7's numbers before T7 changes the
   code.
10. **T3's recommendation.** The §3.13 "Device kernels" rules for CUDA: confirmed as
    written, a CUDA note, or a formulation change. Signed off before T4's f64 gates.

## Risks

| Risk | Mitigation |
| --- | --- |
| The LLVM NVPTX path or the aarch64 Linux `tracel-llvm` bundle fails to build or run on locust (CUDA was never run; the bundle exists, but nothing has used it here) | T1 builds `--features cuda` and opens the device before any other CUDA task; T2 records any defect with a reproducer; NVRTC as a fallback only by decision 4 |
| cudarc picks the wrong CUDA version at build time (it assumes 13.4 without `nvcc`) and fails at run time against the 12.7 driver | `env.sh` puts the spack (or `/usr/local`) CUDA 12.6 `nvcc` on `PATH` and sets `CUDA_PATH`; T1 checks the version cudarc resolved |
| 64 KiB pages break something built for 4 KiB pages (allocators, JIT memory, shared-memory transports) | T1 runs every existing check on Grace before CUDA work and reports anything that fails |
| f64 on CUDA misses the f64 bounds: LLVM's `contract` flag fuses products that the host rounds, and division or square root may lower differently | T3 measures it before T4 relies on it; the bit-identity tests of T2 show any fused replica at once; a §3.13 note or a formulation change by sign-off, never a looser tolerance |
| Kernels written and tuned for a 32 KB, 40-core GPU leave the H100 idle (P2P cubes of 64, GEMM tiles of 32 rows) | T7 measures every kernel against datasheet peaks and adds CUDA layouts where the gain is at least 10%, inside the signed-off formulations |
| The shared node is busy, and timings vary with other users or with clocks that cannot be locked | check `nvidia-smi` and the load before each timing run, and state them; min/max/mean/median over repeats; GPU clocks printed |
| Profiling counters are restricted to administrators (`ncu` refused) | T7 falls back to CUDA event timings and says so |
| Remote work from a sandboxed session is slow and error-prone (every `ssh` needs the sandbox off and a permission prompt) | `sync.sh` and `remote.sh` batch the work into one command per step; the briefs say which commands run remotely |
| spack concretisation or builds fail on RHEL 9 aarch64, or take hours | externals for the compiler and libclang (decision 3); the lock committed once it works; T1 reports times |
| A CubeCL pre-release defect on CUDA blocks a kernel | minimal reproducer, a small workaround in `nd-fmm-kernels` only, otherwise stop and report; moving CubeCL is a separate decision |
| Per-kind timings perturb what they measure (a sync per call) | the overall time comes from untimed evaluations; T5's `Device` mode (CUDA events) adds no sync; the report shows the sum of kinds against the overall mean |

## How to run a task with Claude Code

In the repository root on the M3 Max, start `claude` and say:
"Read docs/phase4s/T<k>-<name>.md and do that task." Review and merge before the next.
Tasks that run on locust use `tools/gh200/sync.sh` and `tools/gh200/remote.sh` from T1.
Before T1 has merged, T1 itself uses `ssh locust` directly, outside the sandbox.

## Exit checklist
- [x] Rust on locust (decision 1): spack's `rust` if it has the current stable, otherwise rustup under `/data/ucahtbe` pinned in `env.sh` (decided 2026-10-05; T1 records which)
- [x] Home directory (decision 2): stand-in `HOME` plus explicit variables (decided 2026-10-05)
- [x] Spack externals (decision 3): gcc and libclang external; openmpi, openblas, fftw, cmake, pkgconf, cuda@12.6 built by spack (decided 2026-10-05)
- [x] CUDA compiler path (decision 4): LLVM NVPTX (decided 2026-10-05)
- [x] Benchmark home (decision 5): `nd-fmm-bench` crate; "degree" = expansion degree p; unit cube [0, 1]³ (decided 2026-10-05)
- [x] CUDA type-check in CI (decision 6): add it, in T2 (decided 2026-10-05)
- [ ] T1 merged: environment rebuilt from the lock; home check clean; CPU-side checks on Grace reported; root `CLAUDE.md` points to Phase 4S
- [ ] T5 merged: per-kind timings on the host, the CPU runtime and Metal, bit-identical, syncs as documented
- [ ] T2 merged: every kernel test on CUDA in f32 and f64; GEMM spike on CUDA; §18.1 facts
- [ ] T3 merged: device arithmetic on CUDA measured; recommendation drafted
- [ ] T3 recommendation signed off (decision 10)
- [ ] T4 merged: Phase 4 device gates on CUDA in f32 and f64; device timestamps on CUDA checked
- [ ] T6 merged: `nd-fmm-bench` and `tools/bench/run.sh`; reports from the M3 Max and locust
- [ ] Static M2L rule on CUDA signed off (decision 9)
- [ ] T7 merged: CUDA layouts and candidates by measurement; H100 peaks; Metal and CPU runtime unchanged bit for bit
- [ ] T8 merged: `fmm-bench/results/phase4s-gh200.md` published
- [ ] CUDA leaf-size default (decision 7): none / added by sign-off
- [x] Phase 5 hardware (decision 8): the M3 Max and locust (decided 2026-10-05)
- [ ] Phase 5 briefs updated for decision 8 (T8), reviewed
- [ ] Design documents updated: laplace-fmm-plan §6.1, §6.2, §7 (Phase 4S), §8.3, §9.1, §9.2; device-path.md §17 note and §18; workspace-structure §2, §3, §3.1, §6

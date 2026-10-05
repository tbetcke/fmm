# Phase 4S / T1 — a reproducible build and test environment on locust (C4S.1)

locust is a GH200 node (README, "Machines"). Nothing from this workspace has been built
there: no Rust, no MPI on the `PATH`, no spack. This task:
- sets the machine up from files kept in the repository, so that the environment can be
  recreated by one script;
- establishes how a Claude Code session on the M3 Max drives builds and runs on locust;
- runs every existing CPU-side check there.

CUDA itself is T2. This task only checks that a `--features cuda` build links and opens
the device.

**Hard constraint:** everything this phase installs, downloads, builds or writes on
locust lives under **`/data/ucahtbe`**. Nothing goes into the home directory
(`/home/ucahtbe`), whose quota is shared and 78% full. That includes caches that tools
put there by default.

Read first:
- root CLAUDE.md ("Build environment", "Checks", "MPI");
- docs/phase4s/README.md ("Machines", "Design decisions": "Working on locust", decisions 1–3);
- `.github/workflows/run-tests.yml`, for the native prerequisites CI installs;
- fmm-kernels/CLAUDE.md ("Sandbox"): where the `tracel-llvm` bundle goes;
- the spack documentation for environments (`spack.yaml`, `spack.lock`, `spack env
  create`, `concretizer:unify`, views, `config:install_tree`, externals) at the release
  you pin.

Facts measured on 2026-10-05 (read-only probe; T1 re-checks and records them in
tools/gh200/machine.md):

| Item | Value |
| --- | --- |
| OS | RHEL 9.3, kernel 5.14.0-362 `aarch64+64k`: **64 KiB pages** |
| CPU | 72 × Arm Neoverse-V2 (Grace), 1 thread per core, one NUMA node with CPUs (9 nodes in all) |
| Memory | 572 GB LPDDR5X host, plus the GPU's 96 GB HBM3 |
| GPU | NVIDIA GH200 480GB, compute capability 9.0, driver 565.57.01 (CUDA 12.7), MIG off, ATS addressing |
| CUDA toolkits | `/usr/local/cuda-11.7`, `12.2`, `12.3`, `12.4`, `12.6`; `nvcc` not on the `PATH` |
| Compilers | system gcc 11.4.1, clang 17.0.6 with `clang-devel` (libclang) |
| Libraries (RPM) | openmpi 4.1.7 (not on the `PATH`), openblas-devel 0.3.21, fftw-devel 3.3.8 |
| Tools | cmake, git, python3; environment modules (no site modules); no Rust, no spack; no batch scheduler (an interactive, shared node) |
| Network | github.com, static.rust-lang.org, index.crates.io, pypi.org reachable |
| Storage | `/data` 1.8 TB, 472 GB free; `/data/ucahtbe` exists and is empty |

Do:
- **Layout under `/data/ucahtbe`** (names are suggestions; record what you choose in
  tools/gh200/README.md):

  | Path | Holds |
  | --- | --- |
  | `spack/` | a spack checkout at a pinned release tag |
  | `spack-config/`, `spack-cache/` | spack's user configuration and cache (`SPACK_USER_CONFIG_PATH`, `SPACK_USER_CACHE_PATH`) |
  | `spack-install/`, `spack-stage/`, `spack-sources/` | `config:install_tree`, `build_stage`, `source_cache` |
  | `envs/fmm-gh200/` | the spack environment created from the repository's `spack.yaml` |
  | `rust/cargo/`, `rust/rustup/` | `CARGO_HOME`, `RUSTUP_HOME` |
  | `home/` | a stand-in `HOME` for build and run shells (decision 2) |
  | `cache/`, `share/`, `config/` | `XDG_CACHE_HOME`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME` |
  | `tmp/` | `TMPDIR` (Open MPI's session directories, cargo, nvcc) |
  | `nv/` | `CUDA_CACHE_PATH` (the driver's PTX JIT cache) |
  | `fmm/` | the source tree (`git clone https://github.com/tbetcke/fmm`; the repository is public) and its `target/` |

- **Keep the home directory clean.** Defaults that write under `$HOME`:
  - cargo and rustup;
  - spack (`~/.spack`);
  - the `tracel-llvm` bundle. Its download cache is `$HOME/.cache/tracel`, from
    `dirs::home_dir()`, so only `HOME` moves it. It is installed under
    `dirs::data_local_dir()`, that is `$XDG_DATA_HOME`, else `~/.local/share/tracel`
    (`tracel-llvm-bundler-23.1.0-3/src/config.rs`);
  - the CUDA JIT cache (`~/.nv/ComputeCache`);
  - git's global configuration;
  - pip.

  The environment script sets every variable above and, by decision 2, also points
  `HOME` at `/data/ucahtbe/home` inside the activated shell. Write
  `tools/gh200/check-home.sh`: it lists files under the real home that are newer than a
  marker the setup script creates. Run it at the end of the task. The expected output
  is nothing, apart from the shell's own history and `.ssh`, which the report names.
- **The spack environment**, `tools/gh200/spack.yaml`, committed with its
  `spack.lock`:
  - **Spack itself:** a release tag (the latest 1.x at the time), and the commit of the
    package repository that tag uses (spack ≥ 1.0 keeps packages in a separate
    repository; pin it in `repos:`). The setup script checks both out.
  - **Specs:** what the root CLAUDE.md lists as native prerequisites (the CI
    `apt-get` line):
    - `openmpi` (Open MPI 5 if it concretises; it serves single-node runs, and Phase 5
      may use it for multi-rank runs);
    - `openblas threads=none`;
    - `fftw`, `cmake`, `pkgconf`;
    - libclang from the system clang 17.0.6, as an external (decision 3);
    - `cuda@12.6`, installed by spack: the newest toolkit at or below the driver's
      12.7 (decision 3; not the `/usr/local` toolkits, for reproducibility).
  - **Compiler and target:** the system gcc 11.4.1 as an external compiler (decision 3),
    unless a spec needs newer. A generic `aarch64` target (or `neoverse_n1`) if gcc 11 cannot target
    `neoverse_v2`. Nothing timed in this phase runs through these libraries' hot loops:
    Phase 4 makes no BLAS call. Say which target you used.
  - `concretizer: unify: true`, and a view (`envs/fmm-gh200/.spack-env/view`, or a path
    under `/data/ucahtbe`) that the environment script puts on `PATH`,
    `LD_LIBRARY_PATH`, `PKG_CONFIG_PATH`, `CMAKE_PREFIX_PATH` and `LIBCLANG_PATH`.
  - Comment the file: why each spec is there and which crate needs it. Report
    `spack concretize` and the install time.
- **Rust** (decision 1, decided 2026-10-05): spack's `rust` if spack offers the current
  stable, otherwise rustup.
  - "Current stable" is the M3 Max's `rustc --version` when you run this task: rustc
    1.99.0 on 2026-10-05, when spack's newest `rust` was 1.97.1.
  - Check the `rust` package at the spack-packages commit you pin.
  - If it has that version: add `rust@<version>` (with `rustfmt` and `clippy`) to
    `spack.yaml`, so the lock pins it.
  - Otherwise: install rustup into `/data/ucahtbe/rust`, pin the version in the
    environment script (`RUSTUP_TOOLCHAIN`), and add `rustfmt` and `clippy`.
  - Report which source you used and why.
  - No `rust-toolchain.toml` in the repository: CI keeps floating stable (root
    CLAUDE.md).
- **Scripts** in `tools/gh200/`, POSIX `sh` (or `bash`, stated), idempotent, each with a
  header comment:
  - `setup.sh`: from nothing to a working environment. It creates the layout, clones
    spack at the pin, creates and installs the environment from `spack.yaml` and
    `spack.lock`, installs Rust, clones the repository, and writes the home marker. It
    refuses to run if `/data/ucahtbe` is missing, and never writes outside it except
    through the stand-in `HOME`, which also lives there.
  - `env.sh`, to be sourced: activates the spack environment, sets the variables above,
    `CUDA_PATH`/`CUDA_HOME`, `nvcc` on `PATH` (cudarc reads the CUDA version from
    `nvcc --version` at build time and silently assumes 13.4 without it,
    `cudarc-0.19.10/build.rs`; set `CUDARC_CUDA_VERSION` too, say which), every BLAS
    thread variable to 1, and `RUST_MIN_STACK=8388608`. It prints a one-line summary.
  - `sync.sh`, run on the M3 Max: rsyncs the working tree of the current worktree to
    `locust:/data/ucahtbe/fmm/<branch>`, excluding `target/` and `.git`. It writes
    `.source-revision` (`git describe --always --dirty` and the branch). Uncommitted
    work can then be built on locust without a commit (root CLAUDE.md, "Working
    agreement"). Host and directory are arguments with locust's as defaults. Add
    `.source-revision` to `.gitignore`.
  - `remote.sh`, run on the M3 Max: `ssh locust 'cd … && . tools/gh200/env.sh && <cmd>'`
    with the branch directory, so that a session runs `tools/gh200/remote.sh cargo test
    -p nd-fmm-math` in one line.
  - `check-home.sh`, as above.
- **tools/gh200/README.md:** what each file does, the one-time setup, the daily loop
  (sync, remote build, remote test), how to recreate the environment from the lock, the
  GPU etiquette on a shared node (check `nvidia-smi` for other users before timing), and
  how to remove everything (`rm -rf /data/ucahtbe/<dirs>`).
- **tools/gh200/machine.md:** the facts table above, re-measured with the commands that
  produced it (`lscpu`, `nvidia-smi -q`, `getconf PAGESIZE`, `rpm -q`, `df -h`), and the
  versions the environment resolved to.
- **Run the CPU-side checks on locust** (no GPU yet), from `env.sh`:
  - the root checks and the stricter workspace checks: fmt, clippy, the tests with
    MPI from spack, docs;
  - the `run-tests-simd` commands, including `--release -- --ignored`: NEON on a second
    aarch64 CPU (Neoverse-V2 against the M3 Max);
  - the `run-tests-kernels` commands (the CubeCL CPU runtime on 72 cores). The first
    build downloads the aarch64 Linux `tracel-llvm` bundle (23.1.0-3); check where it
    landed;
  - one MPI example at 2 ranks under `timeout`, with no loopback flags (Linux), to show
    the MPI works.

  Report each result, any failure with its output, and the cold build time.

  Watch for anything that the 64 KiB pages break. Candidates: the CubeCL CPU runtime's
  JIT memory, Open MPI's shared-memory transport, and allocators with 4 KiB
  assumptions. Report what you saw, even if it was nothing.
- **CUDA link check only.** `cargo build -p nd-fmm-kernels --release --features cuda
  --tests` succeeds. A ten-line throwaway program (not committed, or an `#[ignore]`d
  test T2 will own) opens `Device::open(BackendKind::Cuda)` and prints `DeviceInfo`.
  Report the line. Leave the kernel tests to T2.
- **Root CLAUDE.md:**
  - "Current phase and task briefs" points to docs/phase4s/README.md (Phase 4S, CUDA on
    Grace Hopper), keeping Phase 4 and the older phases listed. Phase 5 follows 4S;
  - "Build environment": a paragraph on locust and tools/gh200/;
  - "Checks": a short "On locust (by hand)" block, filled in by T2 and T4.

Tests that define done:
- From a fresh shell on locust, `. tools/gh200/env.sh` then `cargo test -p nd-fmm-math`
  passes, and the root checks pass as listed above (report any that cannot run).
- `tools/gh200/check-home.sh` shows nothing written into the home directory by the setup
  or the builds (apart from what the report names).
- Recreating the environment into a second prefix from the committed `spack.lock`
  (`spack env create … spack.lock`) installs from the cache and resolves the same hashes.
  Reported. Remove the second prefix afterwards.
- The CUDA `DeviceInfo` line printed on locust.

Must pass:
- on the M3 Max: the root checks (this task changes only `tools/`, docs, `.gitignore`
  and the root CLAUDE.md);
- on locust: the checks listed under "Run the CPU-side checks", with their results.

Do not:
- write anywhere on locust outside `/data/ucahtbe`, or install system packages (no
  `sudo`, no `dnf`);
- store credentials on locust. The repository is public; the M3 Max pushes nothing from
  there, and locust never pushes;
- change any crate, `Cargo.toml` or `Cargo.lock`, or the CI workflows;
- leave GPU processes running, or start long jobs while someone else is using the GPU
  (`nvidia-smi`).

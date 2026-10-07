# tools/gh200: the build and test environment on locust

locust (`ssh locust`, `locust.rc.ucl.ac.uk`) is an NVIDIA GH200 480GB node: 72
Neoverse-V2 (Grace) cores, one H100 (sm_90, 96 GB HBM3), RHEL 9.3 with 64 KiB pages.
It is shared and interactive, with no scheduler. machine.md has the measured facts.
These files build the workspace's environment there from the repository (Phase 4S T1,
docs/phase4s/T1-locust-environment.md) and let a session on the M3 Max drive it.

**Everything on locust lives under `/data/ucahtbe`.** Nothing goes into
`/home/ucahtbe`, whose quota is shared. `env.sh` points `HOME`, every cache and
`TMPDIR` there, and `check-home.sh` shows whether anything escaped.

## Files

| File | Runs on | Does |
| --- | --- | --- |
| `spack.yaml` | locust | the spack environment: `openmpi@5`, `openblas threads=none`, `fftw`, `cmake`, `pkgconf`, `cuda@12.6`, and the system gcc 11.4.1 and libclang 17.0.6 as externals; each spec commented with the crate that needs it |
| `spack.lock` | locust | the concretised environment: every hash. Written by `setup.sh --update-lock`, committed |
| `setup.sh` | locust | from nothing to a working environment (bash): layout, home marker, spack at the pinned tag, the environment from the lock, rustup and the pinned toolchain, a clone of the repository. Idempotent |
| `env.sh` | locust | sourced: moves `HOME` and the caches, activates the environment, sets the Rust, CUDA, libclang, BLAS-thread and stack variables, prints a summary line |
| `sync.sh` | M3 Max | copies the working tree (uncommitted work included) to `locust:/data/ucahtbe/fmm/<branch>` |
| `remote.sh` | M3 Max | runs a command in that copy on locust, inside `env.sh` |
| `check-home.sh` | locust | lists what was written into the home directory (and `/tmp`, `/var/tmp`, `/dev/shm`) since the setup |
| `machine.md` | | the machine's facts, measured, and the versions the environment resolved to |

The scripts are POSIX `sh`, apart from `setup.sh` (bash).

## Pins

| What | Pin | Where |
| --- | --- | --- |
| spack | v1.2.2 (`3e19345b`) | `setup.sh` |
| package repository | spack/spack-packages `d4f7c711` (`releases/v2026.06`, tag v2026.06.0) | `setup.sh`, `spack.yaml` (`repos:`), checked to agree |
| every package | hashes | `spack.lock` |
| Rust | 1.99.0 with rustfmt and clippy, through rustup | `env.sh` (`FMM_GH200_RUST`, exported as `RUSTUP_TOOLCHAIN`) |
| target | `neoverse_v2` (gcc 11.4.1 has `-mcpu=neoverse-v2`) | `spack.yaml` |

Rust comes from rustup, not spack (Phase 4S decision 1): at the pinned package
repository spack's newest `rust` is 1.96.0 (1.97.1 on `develop`), and the current stable
was 1.99.0. There is no `rust-toolchain.toml` in the repository: CI floats on stable.
When CI's stable moves, change `FMM_GH200_RUST` and rerun `setup.sh`.

## The linker

`env.sh` sets `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS` so that cargo links
with the toolchain's own `rust-lld` (through rustup's `gcc-ld/ld.lld` wrapper), as
rustc already does by default on x86_64 Linux. With the system's GNU ld (2.35.2), every
binary that links CubeCL's CPU runtime or CUDA fails: `undefined reference to
LLVMOrcCreateLLJIT`, `LLVM_InitializeNativeAsmParser` and the like, from
`pliron-llvm`. GNU ld resolves static archives in command-line order, and the link line
that `tracel-llvm-bundler` emits puts the LLVM archives before `pliron-llvm`; lld does
not depend on the order. CI's x86_64 runners already use rust-lld, and macOS's linker
does not depend on the order either, which is why this shows only on aarch64 Linux
(also GitHub's `ubuntu-24.04-arm`, if a CubeCL build ever ran there). Changing the
variable rebuilds everything.

## Layout under /data/ucahtbe

| Path | Holds |
| --- | --- |
| `spack/` | spack at v1.2.2 (`SPACK_ROOT`) |
| `spack-config/` | spack's user scope (`SPACK_USER_CONFIG_PATH`): `config.yaml` (install tree, stage, source cache, 36 build jobs) and `repos.yaml` (the package repository pin), written by `setup.sh` |
| `spack-cache/` | `SPACK_USER_CACHE_PATH`: the package repository clone, clingo's bootstrap, misc caches |
| `spack-install/` | `config:install_tree` |
| `spack-stage/` | `config:build_stage` |
| `spack-sources/` | `config:source_cache`: every downloaded tarball |
| `envs/fmm-gh200/` | the environment (`spack.yaml`, `spack.lock`), its view in `.spack-env/view` |
| `rust/cargo/`, `rust/rustup/` | `CARGO_HOME`, `RUSTUP_HOME` |
| `home/` | the stand-in `HOME` (decision 2) |
| `cache/`, `share/`, `config/`, `state/` | `XDG_CACHE_HOME`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `XDG_STATE_HOME` |
| `tmp/` | `TMPDIR` (cargo, rustc, nvcc, spack, Open MPI's session directories) |
| `nv/` | `CUDA_CACHE_PATH` (the driver's PTX JIT cache) |
| `fmm/repo/` | a clone of https://github.com/tbetcke/fmm (read-only; locust never pushes) |
| `fmm/<branch>/` | the trees `sync.sh` copies, each with its own `target/` |
| `logs/` | setup and long-run logs |
| `.home-marker` | the time of the first setup, for `check-home.sh` |

Where things land that would otherwise go to the home directory:
- the `tracel-llvm` bundle (CubeCL's CPU runtime and CUDA's LLVM): downloaded into
  `$HOME/.cache/tracel` (only `HOME` moves it), so `home/.cache/tracel/`; installed
  into `$XDG_DATA_HOME/tracel`, so `share/tracel/`;
- cargo and rustup: `rust/`;
- spack: `spack-config/`, `spack-cache/`;
- the CUDA JIT cache: `nv/`;
- git's and pip's configuration and caches: `home/`, `cache/`.

## One-time setup

From the M3 Max, outside the sandbox (`ssh` reads `~/.ssh`):

```sh
tools/gh200/sync.sh                     # the tree, with tools/gh200, to locust
ssh locust 'cd /data/ucahtbe/fmm/<branch> && mkdir -p /data/ucahtbe/logs && { setsid nohup
    timeout 5h tools/gh200/setup.sh > /data/ucahtbe/logs/setup.log 2>&1 < /dev/null & }'
ssh locust 'tail -n 20 /data/ucahtbe/logs/setup.log'   # until "=== …: done"
```

The first full setup took about 15 minutes, nearly all of it `spack install` (CUDA's
runfile installer alone 10 minutes); machine.md has the breakdown. The first cargo build
that needs CubeCL downloads the aarch64 Linux `tracel-llvm` bundle.

## Daily loop

On the M3 Max, outside the sandbox:

```sh
tools/gh200/sync.sh
tools/gh200/remote.sh cargo test -p nd-fmm-math
tools/gh200/remote.sh 'RUST_MIN_STACK=8388608 cargo test 2>&1 | tail -n 30'
```

`remote.sh` runs `cd /data/ucahtbe/fmm/<branch> && . tools/gh200/env.sh && <command>`
over `ssh`. Its arguments are read by the remote shell, as with `ssh`. Keep remote
commands few and batched: each one goes through the permission prompt. Work is synced,
never committed on locust; commits and pushes happen on the M3 Max.

Interactively on locust: `cd /data/ucahtbe/fmm/<branch> && . tools/gh200/env.sh`.

Long runs: under `timeout`, detached (`setsid nohup … &`), with the output in
`/data/ucahtbe/logs/`. An MPI run on Linux needs no loopback flags:
`timeout 300 mpirun -n 2 target/release/examples/<example>`.

## The benchmark

`tools/bench/run.sh` (Phase 4S T6, fmm-bench/CLAUDE.md) builds `nd-fmm-bench` with the
features of its `--backend` list, sets every BLAS thread variable to 1, sources
`env.sh` if it is not active, and writes a Markdown report to `bench-results/` under the
current directory. From the M3 Max, outside the sandbox, after the checks of "GPU
etiquette" below:

```sh
tools/gh200/sync.sh
tools/gh200/remote.sh 'nvidia-smi; uptime; ps -eo user,pcpu,pmem,etime,cmd --sort=-pcpu | head'
tools/gh200/remote.sh 'timeout 3600 tools/bench/run.sh --backend host,cuda --precision f32,f64 --degree 6'
tools/gh200/remote.sh 'cat bench-results/*.md'
```

`sync.sh` deletes what exists only on locust (`rsync --delete`), `bench-results/`
included: fetch a report before the next sync.

## GPU etiquette

The node and its one GPU are shared:
- before every benchmark or timing run, check whether other users run jobs that can
  influence it (docs/phase4s/README.md, "Timing"): `nvidia-smi` (other processes,
  utilisation, memory), `uptime` and `ps -eo user,pcpu,pmem,etime,cmd --sort=-pcpu`.
  If one could, take no timings: wait and check again, or stop and report. Check again
  after the run, and state what was checked and found in the report. Correctness runs
  that need little memory may share the GPU;
- run long jobs under `timeout`, and leave no process behind (`nvidia-smi`, `ps -u
  $USER`);
- the clocks are not locked (no administrator access): reports print
  `nvidia-smi -q -d CLOCK`.

## Recreating the environment from the lock

`setup.sh` copies `spack.yaml` and `spack.lock` into `envs/fmm-gh200/`, concretises
(a no-op with the lock; it stops if the lock changes, which means the manifest and the
lock disagree) and installs. Into a fresh prefix:

```sh
FMM_GH200_ROOT=/data/ucahtbe/<other> FMM_GH200_SOURCE_CACHE=/data/ucahtbe/spack-sources \
    tools/gh200/setup.sh
```

builds everything again from the cached sources, with the same hashes. An environment
made from the lock alone also works, because the install tree and the package
repository pin are in spack's user scope:

```sh
. tools/gh200/env.sh
spack env create -d /data/ucahtbe/<dir> tools/gh200/spack.lock
spack -e /data/ucahtbe/<dir> install
```

## Changing the environment

Edit `spack.yaml`, then on locust `tools/gh200/setup.sh --update-lock`. It concretises
afresh (`--fresh`), installs, and leaves the new lock in
`/data/ucahtbe/envs/fmm-gh200/spack.lock`. Copy it back and commit both files together:

```sh
scp locust:/data/ucahtbe/envs/fmm-gh200/spack.lock tools/gh200/spack.lock
```

## Checking the home directory

```sh
tools/gh200/remote.sh tools/gh200/check-home.sh
```

lists every path under the real home (from the password database; `env.sh` moves
`$HOME`) newer than `/data/ucahtbe/.home-marker`, and what the user owns in `/tmp`,
`/var/tmp` and `/dev/shm` newer than it. Expected: nothing, or what an interactive login
writes itself (`.bash_history`, `.ssh`). Known writers outside `/data/ucahtbe`:
- NVIDIA's CUDA runfile installer, run by spack: `/tmp/cuda-installer.log` (spack's
  cuda package deletes it), `/var/tmp/dnf-$USER-*` from its `dnf list --installed
  nvidia-driver`, and an empty `/tmp/hsperfdata_$USER` (`setup.sh` removes both);
- Open MPI 5's shared-memory backing files, `/dev/shm/sm_segment.*` by default, of
  which every MPI-initialising test executable leaves one (16 MiB) behind. `env.sh`
  points them at `$TMPDIR` (`OMPI_MCA_*_backing_directory`); delete
  `/data/ucahtbe/tmp/sm_segment.*` now and then.

## Removing everything

Stop any running job first (`ps -u $USER`), then:

```sh
rm -rf /data/ucahtbe/{spack,spack-config,spack-cache,spack-install,spack-stage,spack-sources} \
       /data/ucahtbe/{envs,rust,home,cache,share,config,state,tmp,nv,fmm,logs} \
       /data/ucahtbe/.home-marker
```

Nothing else on the machine was changed: no system packages, no files in the home
directory, no `sudo`.

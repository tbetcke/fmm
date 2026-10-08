# nd-project/fmm

Fast multipole methods in Rust. Workspace crates: octree (distributed octree),
fmm-plan (nd-fmm-plan), the nd-fmm-* transfer-operator crates, and fmm-bench
(nd-fmm-bench, the one-command benchmark; Phase 4S T6). Each crate has its own CLAUDE.md
for crate-specific rules; everything here applies to all of them.

## Source of truth
- docs/CONVENTIONS.md defines every basis function, phase, storage layout and scaling.
  Never change a convention in code; propose changes in the PR description instead.
- Cite conventions in doc comments as `CONVENTIONS §3.x`.
- CONVENTION_VERSION in nd-fmm-math must match the file.
- Current phase and task briefs: docs/phase5/README.md (Phase 5, the distributed FMM;
  design docs/design/distributed-fmm.md, from T1). Phase 4S briefs:
  docs/phase4s/README.md (CUDA on NVIDIA Grace Hopper; the device path of Phase 4 on
  locust's H100, design docs/design/device-path.md). Phase 4 briefs:
  docs/phase4/README.md (CubeCL kernels). Phase 3S briefs:
  docs/phase3s/README.md (SIMD P2P on the host; design docs/design/simd-p2p.md).
  Phase 3 briefs: docs/phase3/README.md.
- Background: docs/design/laplace-fmm-plan.md (operators, plan) and
  docs/design/workspace-structure.md (crate layout). CONVENTIONS.md wins on any conflict.

## Working agreement
- Approval for one action is not approval for the next one of its kind. A request to
  commit, push, delete, publish or open a PR covers the message it appears in and
  nothing after it; ask again, even when the next action looks like an obvious
  continuation.
- Default flow: make the change, run the checks, report exactly which ones ran and what
  they said, and leave the result uncommitted unless the current message asks for a
  commit. Never report an unrun check as passing; say plainly when something could not
  run (missing MPI, missing native libraries).
- One task per branch and PR; stay inside the crates the task names.
- Keep changes targeted: no drive-by reformatting or refactors.

## Working rules
- Run `cargo fmt --all` after every Rust edit, before the other checks or committing.
  CI rejects unformatted code, and mechanical edits (renames, `sed`) easily break
  import ordering.
- Tests first: write the acceptance tests from the task brief, then the implementation.
- Every fast path is tested against a slower trusted path (fixtures, identities, nd-fmm-ref).
- Numeric code is generic over `T: RealScalar`; no allocation in hot loops. This does
  not apply to nd-fmm-plan, whose `FmmOperator::Value` is deliberately generic, not
  bound to `RealScalar` (its test operators use `u32`).
- New dependencies of new crates only through [workspace.dependencies]; octree/
  declares its own directly until migrated (a separate decision). CubeCL only
  in nd-fmm-kernels and spikes/; it is pinned to `=0.11.0-pre.4` (matmul: `cubek-matmul`
  `=0.3.0-pre.4`, with `cubek-std` `=0.3.0-pre.4`) in [workspace.dependencies], a
  pre-release chosen on 2026-10-03 for f64 on CUDA; see
  spikes/cubecl-gemm/SPIKE_REPORT.md.
- `unsafe` only in nd-fmm-simd's `arch` modules and its ISA dispatch, in
  nd-fmm-kernels, and in spikes/. Every block carries a `// SAFETY:` comment and
  every public function stays safe; all other crates, nd-fmm-exec included, are free
  of `unsafe` (docs/design/simd-p2p.md §5.4).
- Document every public item; follow the crate's existing rustdoc style.

## Layout
- New crates: directories fmm-<name>, packages nd-fmm-<name>, library names
  nd_fmm_<name>. octree/ (package nd-octree, lib nd_octree) predates this rule.
- Fixtures live in <crate>/fixtures/ and are regenerated only by tools/fixtures/.
- spikes/ is excluded from default members and never built in CI.
- `Cargo.lock` lives at the root and is committed; there are no crate-level lock files.
  Commit lock-file changes with the manifest change that causes them.
- The target directory is shared at the root; run every command from there.
- `.gitignore` covers `target/` but not VTK output or `*.csv`; keep those out by hand.

## Build environment
Stable Rust with Rust 2024, plus `rustfmt` and `clippy`. Native prerequisites, as
installed in Linux CI: `libclang-dev cmake libfftw3-dev libopenblas-dev openmpi-bin
libopenmpi-dev`. `mpi` and `rlst` build against a real MPI installation and
BLAS/LAPACK, so a dependency build failure is far more often a missing native library
than a defect in a crate.

locust (`ssh locust`, an NVIDIA GH200: 72 Grace cores, one H100) builds and runs in an
environment made by `tools/gh200/` (README.md there): a pinned spack environment
(`spack.yaml`, `spack.lock`) with Open MPI, OpenBLAS, FFTW, cmake, pkgconf and CUDA
12.6, plus rustup with the toolchain pinned in `env.sh`. Everything on locust lives
under `/data/ucahtbe`, never in the home directory; `env.sh` moves `HOME` and every
cache there, and `check-home.sh` verifies it. It also links with rust-lld: GNU ld
cannot link CubeCL's LLVM on aarch64 Linux (README.md there). A session on the M3 Max drives locust
with `tools/gh200/sync.sh` (copies the working tree, uncommitted work included) and
`tools/gh200/remote.sh <command>` (runs it in that copy inside the environment). Both
use `ssh`, which the sandbox denies, so they run outside it. locust never commits or
pushes; it is a shared node, so check `nvidia-smi` and `uptime` before timing anything.

## Checks
CI (GitHub Actions, `.github/workflows/run-tests.yml`, on pull requests to `main` and on
pushes to `main`, so that `main` holds the caches every pull request restores; Phase 4S
T10) runs exactly this for the default members:

```sh
cargo fmt -- --check
cargo clippy -- -D warnings
cargo clippy --examples -- -D warnings
RUST_MIN_STACK=8388608 cargo test
cargo doc --no-deps
```

A second job, `run-tests-simd`, runs nd-fmm-simd alone, without MPI, on an x86_64
(`ubuntu-latest`) and an arm64 (`ubuntu-24.04-arm`) runner: it prints the CPU, then

```sh
cargo clippy -p nd-fmm-simd --all-targets -- -D warnings
cargo test -p nd-fmm-simd -- --show-output
cargo test -p nd-fmm-simd --release -- --ignored --show-output
```

A third job, `run-tests-kernels` (Phase 4 T4, kept at its sign-off on 2026-10-03), runs
nd-fmm-kernels alone on the CubeCL CPU runtime on `ubuntu-latest`, with the cargo
registry, the target directory and the `tracel-llvm` bundle cached. Since Phase 4S T2
(decision 6) it also type-checks CUDA, never runs it; for the nd-fmm-exec check it
installs MPI and libclang, and its tests still run without MPI:

```sh
cargo clippy -p nd-fmm-kernels --all-targets --features cpu -- -D warnings
cargo check -p nd-fmm-kernels --features cuda
cargo check -p nd-fmm-exec --features cuda
cargo test -p nd-fmm-kernels --features cpu --release -- --show-output
```

Its one test run (since Phase 4S T10) sets `CUBECL_DEBUG_LOG`, and the next step prints
the kernel compile times from that log with `fmm-kernels/tools/compile_times.awk`.
A fourth job, `run-tests-mpi` (Phase 5 T3; README decision 4), runs the MPI test
executables and the nd-octree MPI examples at 2 and at 4 ranks on `ubuntu-latest`, for
correctness only, never timings, never ignored or release tests and never more than 4
ranks. `.github/scripts/run-mpi-tests.sh` builds them in debug without running them
(`cargo test -p … --test … --no-run`, `cargo build -p nd-octree --example …`), takes
their paths from cargo's `--message-format=json` output, and launches each under
`timeout` (300 s), the tests with `--test-threads=1 --nocapture`, with
`RUST_MIN_STACK=8388608` and the BLAS thread variables at 1; the run steps have their own
`timeout-minutes`. It covers `nd-fmm-plan`'s `mpi_regressions`, `nd-fmm-exec`'s
`mpi_exec` and `mpi_threading`, and the five registered nd-octree examples. The runner's
4 vCPUs are 2 cores, so the ranks oversubscribe (`OMPI_MCA_rmaps_base_oversubscribe=1`);
it needs no interface flags. By hand, the same script runs a list at any rank count
(`run-mpi-tests.sh build <list>`, then `run-mpi-tests.sh run <list> 2 4`, on macOS with
`MPIRUN_FLAGS` set to the loopback flags below).

`run-tests`, `run-tests-kernels` and `run-tests-mpi` cache cargo (keyed on the job, the
rustc version and `Cargo.lock`, with a prefix fallback); `run-tests` and `run-tests-mpi`
also cache the `.deb` files of their packages (keyed on the runner image and the package
list, which the two share; `.github/scripts/apt-install.sh` installs them offline on a
hit). Every job has a `timeout-minutes`, and so has each package install; a failed
install is a mirror problem, and its error says to re-run.

nd-fmm-kernels is a workspace member, not a default member: the default and
`--workspace` checks build it without a backend. For a change to it, also run

```sh
cargo clippy -p nd-fmm-kernels --all-targets --features cpu,metal -- -D warnings
cargo check -p nd-fmm-kernels --features cuda   # CUDA: type-checked here, run on locust
cargo test -p nd-fmm-kernels --features cpu --release -- --show-output
# By hand on the M3 Max, outside the sandbox (Metal has no adapter inside it):
cargo test -p nd-fmm-kernels --release --features metal -- --ignored --show-output
```

Before finishing any task, also run the stricter

```sh
cargo clippy --workspace --all-targets -- -D warnings
RUST_MIN_STACK=8388608 cargo test --workspace
```

For a change to nd-fmm-simd, also clippy the other architecture's code, after
`rustup target add <other>` (`x86_64-apple-darwin` on Apple silicon,
`aarch64-unknown-linux-gnu` on x86_64 Linux):

```sh
cargo clippy -p nd-fmm-simd --all-targets --target <other> -- -D warnings
```

On locust (by hand, Phase 4S): after `tools/gh200/sync.sh`, the checks above that are
not Metal run there too, each as `tools/gh200/remote.sh '<command>'`; Linux needs no
loopback flags for `mpirun`. The CUDA tests are `#[ignore]`d and run there by hand;
for nd-fmm-kernels (Phase 4S T2) and nd-fmm-exec (Phase 4S T4; `tests/device_cuda.rs`,
the CUDA blocks of the ignored gates and the operator tests):

```sh
cargo test -p nd-fmm-kernels --release --features cpu,cuda -- --include-ignored --show-output
cargo clippy -p nd-fmm-kernels --all-targets --features cpu,cuda -- -D warnings
cargo clippy -p nd-fmm-exec --all-targets --features cpu,cuda -- -D warnings
RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release
RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release -- --ignored
```

`cargo test --workspace` needs a working MPI runtime, because nd-octree and
nd-fmm-plan tests initialise MPI; crates without MPI (e.g. `cargo test -p nd-fmm-math`)
test without one. Keep `RUST_MIN_STACK=8388608` on every test invocation of an MPI
crate; some tests need it. Crate CLAUDE.md files add crate-specific checks. Keep clippy
clean without blanket `#[allow]`s.

Weekly jobs: `run-examples` runs `.github/scripts/run-examples.sh 3` (the examples
registered with templated-examples metadata, found through `cargo metadata`, at 3 ranks
only; locally on macOS export `OMPI_MCA_btl_tcp_if_include=lo0` and
`OMPI_MCA_oob_tcp_if_include=lo0` first); `run-dependency-checks`
runs `cargo upgrades` (not `cargo audit`).

## MPI
- MPI is a required dependency of nd-octree and nd-fmm-plan, also for one-rank runs.
  nd-fmm-math, nd-fmm-ref and nd-fmm-tables stay free of MPI and the octree.
- Every rank must reach every collective in the same order, including ranks with empty
  input. Never put communication inside a branch only some ranks take.
- Keep MPI initialisation to a single test per test executable: MPI cannot be
  re-initialised after finalisation in one process. Add scenarios to the existing
  MPI-owning test, or move them to `tests/` or `examples/`.
- Doctests that initialise MPI are marked `no_run`.
- CI runs the MPI test executables and the nd-octree MPI examples at 2 and 4 ranks on
  every pull request (`run-tests-mpi`, correctness only), and the registered examples
  weekly at 3; everything else in CI runs on one rank. For changes to distributed code,
  still run multi-rank by hand (8 ranks and the ignored tests run only by hand, on the
  M3 Max and on locust), and always under an external timeout: an assertion on one rank
  leaves the others blocked in a collective. Write a multi-rank test to fail on every
  rank or on none.
- On macOS, plain `mpiexec`/`mpirun -n 2 …` hangs or aborts because Open MPI picks a
  non-loopback interface. Add `--mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0`.
  This is environmental, not a bug to chase.
- Register every example that CI should run in the crate's `Cargo.toml` with both an
  `[[example]]` section and `[package.metadata.example.<name>.templated-examples]`
  with `command = "mpirun -n {{NPROCESSES}}"`, as in `octree/Cargo.toml`.

## Navigating the code
Prefer LSP tools (go-to-definition, find-references, symbols) over `grep`/`find` when
chasing a symbol; they resolve through the real dependency graph. `rlst` 0.9.0 comes
from crates.io, so its sources are in `~/.cargo/registry/src/*/rlst-0.9.0/`; a `../rlst`
checkout on the machine, if any, is an unrelated development tree and may differ. Use
`grep` for what LSP does not index: comments, CI YAML, `Cargo.toml`. Read the
implementation and tests rather than trusting prose when checking how an API behaves.

## Commands
- Test one crate: cargo test -p nd-fmm-<name>
- Regenerate fixtures: see tools/fixtures/README.md
- Benchmark (a Markdown report in bench-results/, ignored by git; fmm-bench/CLAUDE.md):
  `tools/bench/run.sh --backend host,metal --precision f32 --degree 6`; on locust
  `tools/gh200/remote.sh 'tools/bench/run.sh --backend host,cuda --precision f32,f64'`

---
name: verify
description: Runs the checks for the current branch of the nd-fmm workspace, chosen by what it changed - format, Clippy, tests, docs, the MPI tests at 2 and 4 ranks, crate-specific and machine checks, and the Laplace protection of the Helmholtz proposal's constraint 1 - and reports exactly what ran. Use before calling any task done, and from /next-task.
---

# Verify

Runs what root `CLAUDE.md` ("Checks", "MPI") requires for the change and reports each
command with its result. An unrun check is never reported as passing.

## 1. What changed

`git diff --name-only main...HEAD` and the uncommitted files give the crates touched. If a
task is in progress, read its plan (`docs/helmholtz/tasks/<ID>.md`): "Checks" and "Machine
runs" add to what follows.

## 2. Always

```sh
cargo fmt --all -- --check
cargo clippy -- -D warnings
cargo clippy --examples -- -D warnings
RUST_MIN_STACK=8388608 cargo test
cargo doc --no-deps
cargo clippy --workspace --all-targets -- -D warnings
RUST_MIN_STACK=8388608 cargo test --workspace
```

The tests of MPI crates need a working MPI.

## 3. By what changed

- **nd-fmm-simd**: `cargo clippy -p nd-fmm-simd --all-targets -- -D warnings`,
  `cargo test -p nd-fmm-simd -- --show-output`,
  `cargo test -p nd-fmm-simd --release -- --ignored --show-output`, and the other
  architecture: `cargo clippy -p nd-fmm-simd --all-targets --target <other> -- -D warnings`
  (`x86_64-apple-darwin` on Apple silicon).
- **nd-fmm-kernels**: the commands of root `CLAUDE.md` (Clippy with `cpu,metal`, the CUDA
  check, the CPU-runtime tests; the Metal tests by hand outside the sandbox; CUDA on locust).
- **nd-octree, nd-fmm-plan, nd-fmm-exec, or code they call**:
  `.github/scripts/run-mpi-tests.sh build <list>`, then `run <list> 2 4`, on macOS with
  `MPIRUN_FLAGS="--mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0"`, under an
  external timeout. For distributed changes also 8 ranks by hand.
- **Spikes**: their own tests (they are not default members), reported apart.
- **Manifests or pins**: `cargo tree -e normal` on the default members lists no `fftw-sys`;
  `cargo deny check licenses` once CI has it; `cargo check --workspace --all-targets`.

## 4. Laplace protection (proposal §2, constraint 1)

If the change touches code Laplace runs (`nd-fmm-exec`'s Laplace operator or the shared
drivers, `nd-fmm-tables`, `nd-fmm-simd`, `nd-fmm-math` as Laplace uses it, `nd-octree`,
`nd-fmm-plan`, the Laplace kernels of `nd-fmm-kernels`) or a dependency pin:
- the Laplace fixtures and the bit-identity tests across threads, overlap and ranks
  (`mpi_exec`, `mpi_threading`, `invariance` through the MPI script at 2 and 4 ranks); where
  they do not cover the changed path, Laplace's outputs on `main` and on the branch with the
  same settings, compared bit for bit;
- Laplace's benchmarks within noise as D9 defines it (`docs/helmholtz/roadmap.md`), against
  `main` on the M3 Max, locust and Kathleen: `tools/bench/run.sh` (on locust through
  `tools/gh200/remote.sh`) and Kathleen's jobs. These are machine runs: pending, with their
  commands, until they have run.

## 5. Machines

What the plan names: on locust `tools/gh200/sync.sh`, then the checks above that are not
Metal through `tools/gh200/remote.sh '<command>'`; on Kathleen `tools/kathleen/sync.sh`,
then `tools/kathleen/submit.sh check.sbatch`, `mpi.sbatch 1 2 4 8`, and
`correctness.sbatch` for structural changes (never on the login node, never more nodes than
`tools/kathleen/README.md` allows); Metal on the M3 Max outside the sandbox.

## 6. Report

A table: command | where | result (passed; failed, with the first error; not run, and why).
Fix failures and run again. Never weaken, skip or delete a test to get green; if a test
seems wrong, explain why and ask.

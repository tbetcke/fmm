# CHR.2 Move to RLST 0.12

Started 2026-10-26 on `tbetcke/h-chr-2`. Roadmap: "CHR.2 Move to RLST 0.12" (HR). Proposal:
§10 CHR.2; §2 constraint 1; §6.7 ("FFTW stays optional", "RLST's FFTW interface"); §8 R21.

## Goal

nd-fmm on RLST 0.12.0 (the workspace pin and `octree/`'s own pin together), with Laplace
unchanged bit for bit and its benchmarks within noise, and a licence check in CI that keeps
GPL code out of the default build.

## Acceptance (proposal §10, verbatim)

> compiled and run, since the review of draft 0.7 compiled nothing: the default members
> build, pass Clippy with `-D warnings`, test and document; the MPI tests (fmm-plan's
> `mpi_regressions`, octree's MPI examples) and the weekly examples job pass;
> `cargo check --workspace --all-targets` builds the spikes, green-kernels on its own RLST
> 0.10.2; `cargo tree -e normal` on the default members lists no fftw-sys; a `cargo deny`
> licence check of the default set added to CI; the `cmake` install of CI and the `cmake`
> entry of `tools/gh200/spack.yaml`, which names RLST's build as its reason, kept or dropped
> by a build without them, since RLST 0.11 dropped its `cmake`, `cc`, `bindgen` and `git2`
> build dependencies (other crates may still need `cmake`: checked, not assumed); the weekly
> `run-dependency-checks` job (`cargo upgrades`) no longer reports `rlst`; constraint 1 of
> Section 2: Laplace fixtures and bit identity across threads, overlap and ranks, and
> Laplace benchmarks within noise on the M3 Max, locust and Kathleen (D9). Fallback: if any
> condition fails, nd-fmm stays on RLST 0.10.2 and does not offer the `fftw` feature;
> nothing else is affected

D9, signed off 2026-10-26: the run-to-run spread of the Laplace baseline, repeated on each
machine of record before the change, per benchmark; plus the C3S.4 throughput gate.

## Read first

- `Cargo.toml` (`[workspace.dependencies]` `rlst = "0.10.2"`, the `[patch]` for
  green-kernels' git rlst), `octree/Cargo.toml` (`rlst = { version = "0.10.2", features =
  ["mpi"] }`), `fmm-plan/Cargo.toml`, `fmm-exec/Cargo.toml`.
- What nd-fmm names in RLST: `distributed_tools` (`all_to_allv`, `gather_to_all`,
  `is_sorted_array`, `sort_to_bins`, `SparseRoutes`, the sparse exchanges), `SliceArray`,
  `rlst_dynamic_array!`, `threading::set_blas_threads`. Found at planning:
  `src/distributed_tools/` and `src/threading.rs` of 0.12.0 are identical to 0.10.2's
  (`diff -r`), as R21 expects.
- `.github/workflows/run-tests.yml` (`APT_PACKAGES` with `cmake`, three jobs),
  `run-examples.yml`, `run-dependency-checks.yml`; `tools/gh200/spack.yaml` and its README.
- Found at planning: `cargo tree -e build,normal -i cmake --workspace` on `main` shows
  `cmake` only as rlst 0.10.2's build dependency.

## Changes

- `Cargo.toml`: `rlst = "0.12.0"`; the `[patch."https://github.com/linalg-rs/rlst.git"]`
  entry stays `rlst = "0.10.2"`, its comment rewritten (green-kernels only, because 0.12
  removed `rlst::simd`; the spikes alone build a second RLST).
- `octree/Cargo.toml`: `rlst = { version = "0.12.0", features = ["mpi"] }`.
- `Cargo.lock`: the resolution that follows (committed with the manifests).
- Source: only what the compiler demands (nothing expected).
- `deny.toml` at the root: a licence allow-list for the default members' normal and build
  dependencies with no GPL entry; CI step `cargo deny check licenses` in `run-tests`
  (installed by `taiki-e/install-action` or `cargo install --locked cargo-deny`, decided by
  what the other jobs use).
- CI's `APT_PACKAGES`: `cmake` dropped where a build without it shows that nothing in the
  default members needs it (expected: nothing after the move); kept, with its reason, where
  something does. `tools/gh200/spack.yaml`: the `cmake` comment names its real user
  (green-kernels' rlst 0.10.2 in the spikes, which `--workspace` builds on locust); the
  entry is kept unless nothing needs it, since dropping it means re-concretising
  `spack.lock`.
- Docs: root `CLAUDE.md` (Navigating: the rlst 0.12.0 sources; Build environment if cmake
  goes), `fmm-plan/CLAUDE.md` (the rlst version line), `docs/design/workspace-structure.md`
  (the rlst mentions), `tools/kathleen/README.md` if it names rlst's build needs.
- Public API: none. Conventions: none. No `fftw` feature (that is CHR.3).

## Tests that prove it

No new test: the move must leave every existing one passing unchanged.
- The default checks and the strict workspace checks (`/verify` §2) on the M3 Max.
- MPI: `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks (`mpi_regressions`, `mpi_exec`,
  `mpi_threading`, `invariance`, the seven nd-octree examples); 8 ranks by hand; the weekly
  examples (`run-examples.sh 3`).
- Laplace bit for bit across the change: on `main` (a detached worktree of `f4a2c71` with
  its own `CARGO_TARGET_DIR`) and on the branch, `nd-fmm-bench --write-problem` once, then
  `--write-output` at f64 for p = 3, 6, 8 with `Batched` and `Rotation`, one thread and all
  threads; `cmp` the output files. Plus the existing fixtures and the bit-identity tests
  across threads, overlap and ranks.
- `cargo tree -e normal` on the default members: no `fftw-sys`.
- `cargo deny check licenses` passes; a check that it fails on a GPL licence (run once
  locally with a temporary `fftw` feature enabled, not committed).
- `cargo check --workspace --all-targets` (spikes, green-kernels on rlst 0.10.2).
- `cargo upgrades` reports no `rlst`.
- A default-members build with `cmake` hidden from `PATH`.

## Checks

`/verify` (Always, nd-octree/nd-fmm-plan/nd-fmm-exec MPI list at 2 and 4, Manifests and
pins, Laplace protection). nd-fmm-simd's and nd-fmm-kernels' crate checks too, since every
crate rebuilds against the new lock file: `cargo test -p nd-fmm-simd --release -- --ignored
--show-output` (the C3S.4 throughput gate), `cargo clippy -p nd-fmm-kernels --all-targets
--features cpu,metal -- -D warnings`, `cargo test -p nd-fmm-kernels --features cpu
--release`.

## Machine runs

D9's baseline is measured here for the first time, and CH2.0 reuses its method. Per
machine, `main` (detached worktree, own target dir) and the branch alternate,
A B A B A B, each run `tools/bench/run.sh` with its 10 repeats; the spread of a benchmark
is max − min of `main`'s three run medians, and the branch is within noise when its median
of medians differs from `main`'s by no more than that spread.

- **M3 Max** (outside the sandbox): `tools/bench/run.sh --backend host,metal --precision
  f32,f64 --degree 3,6,8` (Metal f32 only), N = 10⁶; the C3S.4 gate by the nd-fmm-simd
  ignored tests.
- **locust**: `tools/gh200/sync.sh`; the checks (`/verify` §2 and the CUDA checks of root
  `CLAUDE.md`); the benchmark `tools/bench/run.sh --backend host,cuda --precision f32,f64
  --degree 3,6,8`, after `nvidia-smi` and `uptime`; the cmake question on aarch64 (a
  default build with spack's cmake out of `PATH`).
- **Kathleen**: `tools/kathleen/sync.sh`; `check.sbatch`; `correctness.sbatch` (2 nodes,
  `test`); the benchmark as a one-node `run.sbatch` step of `tools/bench/run.sh --backend
  host --precision f32,f64 --degree 3,6,8`, `main` and the branch alternating in one job.

What they decide: whether CHR.2 passes or falls back to 0.10.2.

## Constraints and risks

- Constraint 1 binds in full: the pin is a dependency of nd-octree, nd-fmm-plan and
  nd-fmm-exec, which Laplace runs. Expected effect: none, since the RLST modules Laplace
  calls are source-identical and no Laplace hot loop calls RLST; shown by the bit-for-bit
  comparison and the D9 runs, not assumed.
- R21: two RLST copies in the default build if only one pin moves (`cargo tree -d` on the
  default members shows one rlst); green-kernels must keep 0.10.2.
- Transitive changes in the lock file (mpi-sys, bindgen, cc) may change the build on the
  machines: the machine runs cover it.
- Fallback as above.

## Pending runs

- **locust** (`/data/ucahtbe/logs/chr2/`, `summary.txt`; `run.sh`, then `run2.sh` after the
  user dropped locust's MPI list on 2026-10-26): `cargo test -p nd-fmm-exec --features cuda
  --release -- --ignored` (running), the cmake trap build, the D9 benchmark
  (`bench-{A,B}{1,2,3}.md`). Decides constraint 1 on locust and the cmake check on aarch64.
  Done there already: fmt, clippy, test, doc, the strict workspace checks, nd-fmm-simd's
  ignored tests, nd-fmm-kernels' and nd-fmm-exec's CUDA checks and tests.
- Done elsewhere: M3 Max (every check, MPI 2 and 4, the bits, D9); Kathleen jobs 608725
  (D9), 608726 (check; only the known `spikes/p2p-simd` failures), 608727 (correctness, 2
  nodes), 608733 (weekly examples).

## Open questions

All three answered by the user on 2026-10-26 ("agree with all three"):

1. D9 on Kathleen f32 (the branch 2.0–3.4 % faster than `main`, outside its three-run
   spread, bits identical): D9 is one-sided; slower than the spread fails, faster passes
   when the outputs are bit for bit identical. CHR.2's Kathleen rows pass; no rerun.
2. Runs per side: from CH2.0 on, five per side in the order A B B A A B B A A B, the spread
   from `main`'s runs alone, never pooled. CHR.2 keeps its three runs as planned.
3. The licence check stays a separate CI job (`run-licence-check`), not a step of
   `run-tests` as "Changes" said: it needs only `cargo metadata`, fails fast under its own
   name, and CHR.3's `fftw` job can follow its pattern.

# Helmholtz CHR.2: nd-fmm on RLST 0.12

Helmholtz task CHR.2 ([docs/helmholtz/tasks/CHR.2.md](../../docs/helmholtz/tasks/CHR.2.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CHR.2, §2 constraint 1, §6.7 and R21. Measured on 2026-10-26 and 27 from branch
`tbetcke/h-chr-2` against `main` at `f4a2c71` (a detached worktree with its own target
directory, `detached-f4a2c71` on the remote machines), release builds, rustc 1.99.0, every
BLAS thread variable at 1.

## Summary

1. **The move.** `rlst` 0.10.2 → 0.12.0 for the workspace and `octree/`, with no source
   change; green-kernels keeps 0.10.2 through the `[patch]` (spikes only). The default
   members build, lint, test and document on the M3 Max, Kathleen and locust; the MPI tests
   pass at 2 and 4 ranks (M3 Max) and at 2, 4, 8 and 80 (Kathleen), the weekly examples at
   3 ranks (Kathleen), `cargo check --workspace --all-targets` builds the spikes.
2. **Laplace bit for bit**: 48 full outputs of `main` and the branch (two distributions,
   p = 3, 6, 8, four strategies, 1 and 8 threads) identical byte for byte (§2); the
   fixtures, the bit-identity and the rank-invariance tests pass.
3. **Laplace's benchmarks (D9)**: within noise on the M3 Max (all nine) and in f64 on
   Kathleen; Kathleen's three f32 rows are 2.0–3.4 % *faster* than `main`, which passes
   under D9 as clarified on 2026-10-26 (one-sided, §7); within noise on locust (all
   twelve, host and CUDA).
4. **The licence check** (`deny.toml`, CI job `run-licence-check`) passes on the default
   set and fails when FFTW enters it, on fftw-sys stated as `GPL-2.0-or-later` (§3).
5. **cmake**: no crate of the workspace runs it (a trap build, §4); CI drops it, locust's
   and Kathleen's environments keep it unchanged. `cargo upgrades` no longer reports `rlst`; `cargo tree -e
   normal` lists no fftw-sys.

## 1. The change

- `rlst` 0.10.2 → 0.12.0 in `[workspace.dependencies]` and in `octree/Cargo.toml`, together.
  The `[patch]` that points green-kernels' git rlst at crates.io stays on 0.10.2: 0.12
  removed `rlst::simd`, which green-kernels uses. Only the spikes build it.
- No source change. What nd-fmm names in RLST (`distributed_tools`, `threading`,
  `SliceArray`, `rlst_dynamic_array!`) is unchanged: `src/distributed_tools/` and
  `src/threading.rs` of 0.12.0 are identical to 0.10.2's (`diff -r`).
- `Cargo.lock`: rlst and rlst-proc-macro 0.12.0 added beside the spikes' 0.10.2.
  `cargo tree -e normal,build` on the default members shows one rlst, 0.12.0.
- `deny.toml` and the CI job `run-licence-check`: `cargo deny --locked check licenses` on
  the default members with default features, permissive licences only. fftw-sys is stated
  as `GPL-2.0-or-later` (§3).
- `cmake` dropped from CI's package lists (§4); locust's spack `cmake` and Kathleen's
  module stay, unused.

## 2. Laplace bit for bit

`nd-fmm-bench --write-problem` wrote two problems once (the cube and the Plummer sphere,
N = 2 × 10⁵, one charge vector, `scaling`'s seed); `--write-output` then evaluated each
with `main`'s binary and the branch's, f64, p = 3, 6 and 8, the strategies Batched,
Rotation, Dense and PlaneWave, at 1 and 8 threads: 48 full outputs (φ and ∇φ at every
point, 51 MB each at N = 2 × 10⁵) per binary, on the M3 Max. `cmp`: **all 48 pairs
identical**. The comparison has teeth: outputs of different strategies or degrees differ
from the 25th byte on. The byte comparison ran on the M3 Max (aarch64) only; on x86
(Kathleen) the evidence is the equal error columns of §6 and the branch's own identity and
invariance tests, with no source change and the RLST modules nd-fmm calls identical.

Beside it, the existing fixtures and bit-identity tests pass (§5): across threads and
overlap (`mpi_exec`, `mpi_threading`), across ranks (`invariance` at 2 and 4 ranks, and
on Kathleen at 80 ranks against 40, 20 and 10), and the error columns of every benchmark
run below are equal, digit for digit, between `main` and the branch on all three machines.

## 3. The licence check

`deny.toml` leaves out the non-default members (nd-fmm-kernels and the spikes) and allows
Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, MIT, Unicode-3.0 and Unlicense. The default
set's licences (`cargo deny list -l license`) are exactly these, plus r-efi's
`MIT OR Apache-2.0 OR LGPL-2.1-or-later`, which passes on its MIT term. `cargo deny
--locked check licenses` passes on the branch with no warning.

The negative check, in a throwaway worktree with `rlst = { workspace = true, features =
["fftw_system"] }` in `nd-fmm-exec`: fftw-sys 0.8.0 enters the default graph and the check
fails (exit 4). Without a clarification it failed for the wrong reason: fftw-sys declares
its licence only through `LICENSE.md`, which holds both the GPL terms (the FFTW back end)
and MIT ones (MKL), and cargo-deny read it as MIT at confidence 0.77, rejected only for
falling below the 0.8 threshold ("unlicensed"). A more confident match would have passed
it. `deny.toml` therefore states fftw-sys as `GPL-2.0-or-later` (a `[[licenses.clarify]]`
tied to the file's hash); with it the check fails with "rejected: GPL-2.0-or-later is not
explicitly allowed". CHR.3's `fftw` job will need its own allowance for that entry.

## 4. cmake

rlst 0.10.2 lists the `cmake` crate as a build dependency, but its `build.rs` only probes
FFTW through pkg-config (behind `fftw_system`) and never calls it. On the branch the crate
is in the graph only under green-kernels' rlst 0.10.2 (the spikes), and no build script of
the default graph mentions cmake (libffi-sys builds with autotools). The trap, Kathleen job
608744 (node-c11a-139, 3.5 min): with a `cmake` first on `PATH` that records every call
and exits 127, in a fresh target directory, `cargo build --all-targets` on the default
members and then `cargo check --workspace --all-targets` (the spikes included) both
succeed and no call is recorded (no `calls.txt`); on locust (aarch64) the same trap on the
default members' `build --all-targets` (113 crates from scratch) records no call either. **No crate of the workspace runs cmake.**
So CI's three package lists (`run-tests`, `run-tests-mpi`, `run-examples`) drop it, and the
CI run of the pull request repeats the check on GitHub's runner. locust's spack `cmake`
and Kathleen's module stay, with their comments corrected: no crate needs them, and
dropping the spack entry means re-concretising `spack.lock`, a change of its own.

## 5. Checks

| check | where | result |
| --- | --- | --- |
| `cargo fmt --all -- --check`, `cargo clippy -- -D warnings`, `cargo clippy --examples -- -D warnings`, `cargo test`, `cargo doc --no-deps` | M3 Max; Kathleen job 608726; locust | passed on each |
| `cargo clippy --workspace --all-targets -- -D warnings` | M3 Max; Kathleen 608726; locust | passed on each |
| `cargo check --workspace --all-targets` (the spikes, green-kernels on rlst 0.10.2) | M3 Max | passed |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | M3 Max (878 passed, 0 failed); locust | passed |
| the same | Kathleen 608726 | failed in `spikes/p2p-simd` only, known before this task: `kernel_rsqrt_within_contract` (avx2 f64 est+S2 at 289 u against 4 u: two Newton steps from Intel's 1.5 · 2⁻¹² estimate give about 270 u) and `compare_core_on_a_small_cell` (green-kernels' f64 gradient 1.17 · 10⁻¹⁴ over its bound; ours 4.3 · 10⁻¹⁶). The spike builds on the unchanged rlst 0.10.2 and calls no RLST in either computation; the same values were recorded on Kathleen before |
| nd-fmm-simd: clippy, tests, `--release -- --ignored` (C3S.3 and C3S.4's accuracy and invariance) | Kathleen 608726 (AVX2, AVX-512); locust (NEON); M3 Max (NEON, the ignored tests) | passed |
| C3S.4's throughput clause (NEON, at least 90 % of the spike prototype, reported by `p2p_kernels`) | – | not re-measured: unchanged by construction. nd-fmm-simd has no RLST in its graph, neither it nor `spikes/p2p-simd` changed, and the lock file adds only rlst and rlst-proc-macro 0.12.0 (no other package moved), so its kernels compile from the same sources and dependencies |
| nd-fmm-kernels clippy with `cpu,metal` | M3 Max | not run (the M3 Max was loaded by other jobs); nd-fmm-kernels has no RLST in its graph and did not change; its `cpu` Clippy and `cuda` check passed on Kathleen, `cpu,cuda` on locust |
| nd-fmm-kernels: clippy `cpu`, `cpu,cuda`; `check --features cuda`; CPU-runtime tests; CUDA tests `--include-ignored` | Kathleen 608726; locust | passed |
| nd-fmm-exec: `check --features cuda`; clippy `cpu,cuda`; `test --features cuda --release` | Kathleen 608726; locust | passed |
| nd-fmm-exec: `--features cuda --release -- --ignored` | locust | passed: every executable ok (`accuracy` 411 s, `adaptive` 196 s, `device_cuda` 237 s, `device_fmm_gate` 1186 s, `device_ranks` 72 s, `dipoles` 363 s, `multi_rank` 2567 s, `operator`'s seven CUDA tests) |
| MPI list at 2 and 4 ranks (`mpi_regressions`, `mpi_exec`, `mpi_threading`, `invariance`, the seven nd-octree examples) | M3 Max | passed (`mpi_exec` 282 s at 2 ranks, near the script's 300 s default) |
| the same | locust | not run: dropped by the user on 2026-10-26 to shorten the locust run, covered on the M3 Max and Kathleen |
| `correctness.sbatch`: the MPI list at 2, 4, 8 and 80 ranks, invariance at 80 against 40, 20, 10, the `scaling` smoke run | Kathleen 608727, 2 nodes `test` (node-c11b-[037-038]), 46 min | passed |
| weekly examples, `.github/scripts/run-examples.sh 3` | Kathleen 608733 (`run.sbatch`, one node) | 9 passed, 0 failed |
| `cargo tree -e normal` on the default members: no fftw-sys; one rlst (0.12.0) | M3 Max | passed |
| `cargo deny --locked check licenses`; the FFTW negative check | M3 Max | passed; fails as intended |
| the cmake trap: default members `build --all-targets`, `check --workspace --all-targets`, a failing `cmake` first on `PATH` | Kathleen 608744; locust (default members only) | passed, no call |
| `cargo upgrades` | M3 Max | no `rlst` (on `main`: rlst in nd-fmm-plan, nd-octree, nd-fmm-exec); left: the cubecl pins (0.11.0 → 0.11.3) |

## 6. Laplace's benchmarks (D9)

D9, signed off 2026-10-26: the run-to-run spread of the Laplace baseline, repeated on each
machine before the change, per benchmark. Method: per machine, `main` (A) and the branch
(B) alternate, A B A B A B, each run `tools/bench/run.sh` at N = 10⁶ with its 10 timed
evaluations (2 warm-ups); per benchmark the spread is max − min of `main`'s three run
medians, and the branch is within noise when its median of medians differs from
`main`'s by no more than that.

### M3 Max

Outside the sandbox, after the user's go-ahead on a quiet machine (load 3.7 at the start;
before each pair no other process above 31 % of a core). `--backend host,metal
--precision f32,f64 --degree 3,6,8`; Metal refuses f64.

| backend | precision | p | strategy | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians (ms) | within noise | errors equal |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| host | f32 | 3 | Batched | 61.05, 58.02, 60.20 | 58.51, 58.57, 58.66 | 3.03 | -1.63 (-2.7 %) | yes | yes |
| host | f32 | 6 | Batched | 101.20, 98.76, 99.27 | 99.55, 99.80, 100.22 | 2.44 | +0.53 (+0.5 %) | yes | yes |
| host | f32 | 8 | PlaneWave | 117.69, 118.58, 120.24 | 117.81, 122.29, 120.77 | 2.55 | +2.19 (+1.8 %) | yes | yes |
| host | f64 | 3 | Batched | 94.48, 97.41, 98.41 | 96.15, 102.53, 99.85 | 3.93 | +2.44 (+2.5 %) | yes | yes |
| host | f64 | 6 | Batched | 157.04, 170.30, 169.20 | 162.78, 170.92, 178.56 | 13.25 | +1.72 (+1.0 %) | yes | yes |
| host | f64 | 8 | PlaneWave | 171.57, 181.83, 182.64 | 175.59, 183.67, 187.28 | 11.06 | +1.85 (+1.0 %) | yes | yes |
| metal | f32 | 3 | Dense | 16.89, 16.55, 16.40 | 16.73, 16.32, 16.68 | 0.49 | +0.12 (+0.8 %) | yes | yes |
| metal | f32 | 6 | Dense | 29.99, 30.35, 29.97 | 30.20, 29.94, 30.00 | 0.39 | +0.01 (+0.0 %) | yes | yes |
| metal | f32 | 8 | Dense | 38.78, 38.68, 39.00 | 38.71, 38.89, 38.83 | 0.32 | +0.05 (+0.1 %) | yes | yes |

All nine within noise.

### Kathleen

Job 608725, one node in `singlenode` (node-c11a-132, Xeon Gold 6248, 40 cores, exclusive),
both binaries built in the job first. `--backend host --precision f32,f64 --degree 3,6,8`.

| backend | precision | p | strategy | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians (ms) | within noise | errors equal |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| host | f32 | 3 | Batched | 77.45, 77.09, 77.46 | 73.75, 78.42, 75.87 | 0.37 | -1.58 (-2.0 %) | NO | yes |
| host | f32 | 6 | Batched | 112.92, 114.49, 114.33 | 108.03, 113.64, 110.49 | 1.56 | -3.84 (-3.4 %) | NO | yes |
| host | f32 | 8 | PlaneWave | 150.57, 152.19, 152.91 | 146.73, 147.95, 148.90 | 2.35 | -4.25 (-2.8 %) | NO | yes |
| host | f64 | 3 | Batched | 116.16, 109.08, 115.08 | 117.59, 110.71, 112.90 | 7.08 | -2.18 (-1.9 %) | yes | yes |
| host | f64 | 6 | Batched | 162.76, 161.42, 157.29 | 155.77, 156.28, 158.88 | 5.47 | -5.14 (-3.2 %) | yes | yes |
| host | f64 | 8 | PlaneWave | 201.71, 206.49, 200.87 | 204.57, 203.61, 211.63 | 5.62 | +2.87 (+1.4 %) | yes | yes |

The f64 rows are within noise. The three f32 rows are outside the two-sided rule: the
branch is **faster** than `main` by 2.0–3.4 %, beyond a spread of `main` that three runs put
at 0.4–2.4 ms, while the branch's own three runs spread 1.9–4.7 ms. With the outputs bit
for bit identical and no RLST call in the timed evaluation, this is a placement effect of
the two binaries or drift, not a change in Laplace's work. Under D9 as clarified on
2026-10-26 (§7: one-sided, faster passes with identical bits) all six rows pass.

### locust

After the checks, in `/data/ucahtbe/logs/chr2/` (`run.sh`, then `run2.sh`); locust idle
before timing (load 4.5 from the two builds, GPU unused, no other user's process).
`--backend host,cuda --precision f32,f64 --degree 3,6,8`, Neoverse-V2 (72 cores) and the
H100.

| backend | precision | p | strategy | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians (ms) | within noise | errors equal |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| host | f32 | 3 | Batched | 45.57, 44.91, 46.42 | 45.26, 45.76, 50.61 | 1.51 | +0.19 (+0.4 %) | yes | yes |
| host | f32 | 6 | Batched | 58.87, 59.00, 136.66 | 58.30, 58.80, 70.41 | 77.80 | -0.20 (-0.3 %) | yes | yes |
| host | f32 | 8 | PlaneWave | 65.49, 65.16, 150.77 | 65.43, 65.31, 80.19 | 85.62 | -0.06 (-0.1 %) | yes | yes |
| host | f64 | 3 | Batched | 78.79, 79.76, 80.08 | 79.56, 79.72, 100.10 | 1.29 | -0.04 (-0.1 %) | yes | yes |
| host | f64 | 6 | Batched | 99.16, 100.50, 101.30 | 99.60, 98.86, 112.40 | 2.14 | -0.90 (-0.9 %) | yes | yes |
| host | f64 | 8 | PlaneWave | 101.97, 104.53, 107.62 | 103.94, 104.86, 140.42 | 5.65 | +0.33 (+0.3 %) | yes | yes |
| cuda | f32 | 3 | Dense | 21.96, 21.70, 24.40 | 22.14, 21.46, 36.39 | 2.71 | +0.18 (+0.8 %) | yes | yes |
| cuda | f32 | 6 | Dense | 24.93, 25.44, 26.48 | 25.77, 24.52, 34.03 | 1.55 | +0.34 (+1.3 %) | yes | yes |
| cuda | f32 | 8 | Dense | 30.87, 30.94, 35.76 | 30.43, 31.32, 41.01 | 4.89 | +0.38 (+1.2 %) | yes | yes |
| cuda | f64 | 3 | Dense | 43.48, 39.70, 45.63 | 43.73, 39.84, 42.00 | 5.93 | -1.48 (-3.4 %) | yes | yes |
| cuda | f64 | 6 | Dense | 48.19, 45.16, 82.37 | 48.66, 45.35, 49.96 | 37.20 | +0.47 (+1.0 %) | yes | yes |
| cuda | f64 | 8 | Dense | 58.06, 54.63, 95.52 | 58.30, 54.91, 59.31 | 40.89 | +0.24 (+0.4 %) | yes | yes |

All twelve within noise. The third pair was disturbed on both sides (`main`'s f32 p = 6
at 136.7 ms against 59.0, the branch's f64 p = 8 at 140.4 against 104.9; the cause is not
recorded), which inflates some spreads to 37–86 ms and so weakens the rule for those rows.
On the two undisturbed pairs alone every benchmark differs from `main` by at most
1.1 % and is within `main`'s two-run spread or faster: the same verdict without the
disturbance.

## 7. Signed off

The user, 2026-10-26, on the three questions of the pull request ("agree with all three"),
recorded in the decision table (D9) and the plan:

1. **D9 is one-sided.** A branch slower than `main` by more than the spread fails; a faster
   one passes when its outputs are bit for bit identical. Kathleen's three f32 rows
   (2.0–3.4 % faster, bits identical) therefore pass.
2. **Five runs per side from CH2.0 on**, in the order A B B A A B B A A B, which cancels a
   linear drift (every pair here ran `main` first); the spread is max − min of `main`'s five
   run medians, never pooled with the branch's. Three runs underestimated it here (Kathleen
   f32 p = 3: `main` 0.37 ms, the branch 4.67 ms).
3. **The licence check is its own CI job**, `run-licence-check`, not a step of `run-tests`.

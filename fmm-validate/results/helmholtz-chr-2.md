# Helmholtz CHR.2: nd-fmm on RLST 0.12

Helmholtz task CHR.2 ([docs/helmholtz/tasks/CHR.2.md](../../docs/helmholtz/tasks/CHR.2.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CHR.2, §2 constraint 1, §6.7 and R21. Measured on 2026-10-26 from branch
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
   under D9 as clarified on 2026-10-26 (one-sided, §7). locust pending.
4. **The licence check** (`deny.toml`, CI job `run-licence-check`) passes on the default
   set and fails when FFTW enters it, on fftw-sys stated as `GPL-2.0-or-later` (§3).
5. **cmake**: the default members no longer need it; CI drops it, locust and Kathleen keep
   it for the spikes (§4). `cargo upgrades` no longer reports `rlst`; `cargo tree -e
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
  module stay, for the spikes.

## 2. Laplace bit for bit

`nd-fmm-bench --write-problem` wrote two problems once (the cube and the Plummer sphere,
N = 2 × 10⁵, one charge vector, `scaling`'s seed); `--write-output` then evaluated each
with `main`'s binary and the branch's, f64, p = 3, 6 and 8, the strategies Batched,
Rotation, Dense and PlaneWave, at 1 and 8 threads: 48 full outputs (φ and ∇φ at every
point, 51 MB each at N = 2 × 10⁵) per binary, on the M3 Max. `cmp`: **all 48 pairs
identical**. The comparison has teeth: outputs of different strategies or degrees differ
from the 25th byte on.

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

On `main`, `cargo tree -e build,normal -i cmake --workspace` shows the `cmake` crate only
as rlst 0.10.2's build dependency; on the branch only under green-kernels' rlst 0.10.2
(the spikes), and not at all for the default members. No build script of the default
graph mentions cmake (searched in each package with a `build.rs`; libffi-sys builds with
autotools). A default-members build on locust with a `cmake` on `PATH` that records any call and fails is pending. So CI's three package lists (`run-tests`,
`run-tests-mpi`, `run-examples`) drop `cmake`, and the CI run of the pull request is
the final check on x86_64. locust's spack `cmake` and Kathleen's module stay, with their
reason corrected: `--workspace` builds the spikes there.

## 5. Checks

| check | where | result |
| --- | --- | --- |
| `cargo fmt --all -- --check`, `cargo clippy -- -D warnings`, `cargo clippy --examples -- -D warnings`, `cargo test`, `cargo doc --no-deps` | M3 Max; Kathleen job 608726; locust | passed on each |
| `cargo clippy --workspace --all-targets -- -D warnings` | M3 Max; Kathleen 608726; locust | passed on each |
| `cargo check --workspace --all-targets` (the spikes, green-kernels on rlst 0.10.2) | M3 Max | passed |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | M3 Max (878 passed, 0 failed); locust | passed |
| the same | Kathleen 608726 | failed in `spikes/p2p-simd` only, known before this task: `kernel_rsqrt_within_contract` (avx2 f64 est+S2 at 289 u against 4 u: two Newton steps from Intel's 1.5 · 2⁻¹² estimate give about 270 u) and `compare_core_on_a_small_cell` (green-kernels' f64 gradient 1.17 · 10⁻¹⁴ over its bound; ours 4.3 · 10⁻¹⁶). The spike builds on the unchanged rlst 0.10.2 and calls no RLST in either computation; the same values were recorded on Kathleen before |
| nd-fmm-simd: clippy, tests, `--release -- --ignored` (C3S.3 and C3S.4) | Kathleen 608726 (AVX2, AVX-512); locust (NEON) | passed |
| nd-fmm-kernels: clippy `cpu`, `cpu,cuda`; `check --features cuda`; CPU-runtime tests; CUDA tests `--include-ignored` | Kathleen 608726; locust | passed |
| nd-fmm-exec: `check --features cuda`; clippy `cpu,cuda`; `test --features cuda --release` | Kathleen 608726; locust | passed |
| nd-fmm-exec: `--features cuda --release -- --ignored` | locust | pending (running) |
| MPI list at 2 and 4 ranks (`mpi_regressions`, `mpi_exec`, `mpi_threading`, `invariance`, the seven nd-octree examples) | M3 Max | passed (`mpi_exec` 282 s at 2 ranks, near the script's 300 s default) |
| the same | locust | not run: dropped by the user on 2026-10-26 to shorten the locust run, covered on the M3 Max and Kathleen |
| `correctness.sbatch`: the MPI list at 2, 4, 8 and 80 ranks, invariance at 80 against 40, 20, 10, the `scaling` smoke run | Kathleen 608727, 2 nodes `test` (node-c11b-[037-038]), 46 min | passed |
| weekly examples, `.github/scripts/run-examples.sh 3` | Kathleen 608733 (`run.sbatch`, one node) | 9 passed, 0 failed |
| `cargo tree -e normal` on the default members: no fftw-sys; one rlst (0.12.0) | M3 Max | passed |
| `cargo deny --locked check licenses`; the FFTW negative check | M3 Max | passed; fails as intended |
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

Pending: `tools/bench/run.sh --backend host,cuda --precision f32,f64 --degree 3,6,8`, A B A B A B, after the checks (`/data/ucahtbe/logs/chr2/` on locust).

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

# Helmholtz CH2.0: Laplace's host drivers made generic over the kernel

Helmholtz task CH2.0 ([docs/helmholtz/tasks/CH2.0.md](../../docs/helmholtz/tasks/CH2.0.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH2.0, §2 constraint 1, §6.2; D9 of [docs/helmholtz/roadmap.md](../../docs/helmholtz/roadmap.md).
Measured on 2026-10-31 from branch `tbetcke/h-ch2-0` against `main` at `2b73f74` (a
detached worktree with its own target directory, `detached-2b73f74` on the remote
machines). Release builds, rustc 1.99.0, every BLAS thread variable at 1, every core as a
host thread (16 on the M3 Max, 72 on locust, 40 on Kathleen).

## Summary

1. **The change.** The level calls' walking of the views, the threads and scratch sets and
   the batched M2L's blocks moved from `operator.rs` into the crate-private
   `nd-fmm-exec/src/driver.rs`. They are now generic over `driver::HostKernel<T>`, whose
   methods take the level call's level. Laplace's `Kernels` implements it by delegating to
   its unchanged methods and ignoring the level. The plane-wave M2L stays in `operator`.
   There is no public API, table, layout, manifest or convention change.
2. **Condition 1 (no new dispatch).** The kernel is a type parameter. No `dyn`, function
   pointer or new enum match was added. The one runtime branch of the drivers, batched M2L
   or not, replaces the same branch of `main` at the level call.
3. **How p reaches Laplace's inner loops** (§1): at run time, as before. No host loop of
   Laplace is specialised on p, and the trait supplies p through the level argument, which
   Laplace ignores.
4. **Condition 3 (bit for bit).** On each machine, 18 full outputs of `main` and the branch
   are identical byte for byte (§2). The existing tests pass: the fixtures and operator
   tests, threads, overlap, Batched against Dense, every ISA, vectors, dipoles, the
   one-rank reference, and rank invariance at 2, 4 and 8 ranks.
5. **Condition 4 (D9).**
   - M3 Max: all 12 within noise.
   - locust: 11 of 12 in the first pass. f32 p = 3 Batched was +2.6 % against a spread of
     2.2 %; a repeat passes (−0.4 %).
   - Kathleen: 10 of 12. f64 p = 6 Dense and Classes are +5.4 %, beyond the spread. The
     cause is code placement (§3.3): with every loop aligned to 64 bytes, `main` and the
     branch are equal. These two rows are put to the user (§4).

## 1. How p reaches Laplace's inner loops

At run time, never as a compile-time constant, before and after this change:
- `Kernels::p`, a `usize` fixed at construction, goes to `nd_fmm_ref::leaf::*_many(p, …)`
  and to `each_vector`'s chunk length (p + 1)².
- The tables carry their own p: `M2mTables`, `L2lTables`, `M2lTables` (`MatrixSet`,
  n = (p + 1)²), `M2lClasses` and `RotationTables` read it from themselves, and
  `MatrixKernel::accumulate_columns` takes n as an argument.
- There is no const generic over p in `nd-fmm-exec`, `nd-fmm-tables`, `nd-fmm-ref`'s leaf
  operators or `nd-fmm-simd`'s matrix kernel.
- The strategy is a run-time choice too: `Tables` matches on its families per pair, and
  that match is unchanged.

`HostKernel`'s methods take the level call's level. Laplace's implementation ignores it
and reads its one p as before, so its loops receive the same runtime values. A Helmholtz
kernel (CH2.1, CH3.1) reads p_l and its tables from the level. A const generic would be
introduced only where an inner loop is specialised on p, and none of Laplace's is.

## 2. Laplace bit for bit

`nd-fmm-bench --write-problem` (the cube, N = 10⁶), then `--write-output` from both
trees' release binaries: f64, φ and ∇φ at every point, 256 000 024 bytes per file,
compared with `cmp`. The cases are every strategy (dense, classes, rotation, batched,
planewave, auto) at p = 3, 6 and 8, on every core.

| machine | ISA of P2P | cases | identical |
| --- | --- | ---: | ---: |
| M3 Max | NEON | 18 | 18 |
| locust | NEON | 18 | 18 |
| Kathleen (job 614707, node-c11a-110) | AVX-512 | 18 | 18 |

f32 is not in the file format. It runs the same generic code, and the D9 runs' f32 error
columns are equal on both sides in every run. The test suite covers the bit identities
of the branch with itself (§5).

## 3. Laplace's benchmarks (D9)

D9 as signed off (2026-10-26, clarified the same day): five runs per side in the order
A B B A A B B A A B, A `main` and B the branch. Each run is `tools/bench/run.sh` at
N = 10⁶ with 10 timed evaluations after 2 warm-ups. Per benchmark, the spread is
max − min of `main`'s five run medians. The branch fails only if its median of medians is
slower than `main`'s by more than that spread; faster passes with identical bits. There
are 12 benchmarks per machine, host only:
- `auto`, f32 and f64, at p = 3, 6 and 8 (Batched at 3 and 6, PlaneWave at 8);
- `dense`, `classes` and `rotation`, f32 and f64, at p = 6.

The scripts are `d9_run.sh`, `d9.py`, `bits.sh`, `d9_one.sh` and `d9_aligned.sh`, in the
session's scratchpad (not committed), as in CHR.2.

### 3.1 M3 Max

Outside the sandbox, 00:39–01:01. At the start the load was 4.8, with `appstoreagent` at
92 % of one core. During the runs the load was 15–21, mostly the benchmark's own 16
threads.

| backend | precision | p | strategy | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians (ms) | within noise | errors equal |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| host | f32 | 3 | Batched | 59.81, 60.73, 59.72, 59.97, 59.31 | 60.18, 60.84, 60.33, 57.77, 59.87 | 1.43 | +0.37 (+0.6 %) | yes | yes |
| host | f32 | 6 | Batched | 98.28, 102.17, 98.78, 97.98, 98.02 | 111.29, 97.86, 103.79, 101.77, 101.87 | 4.19 | +3.59 (+3.6 %) | yes | yes |
| host | f32 | 6 | Classes | 398.99, 363.51, 351.90, 363.28, 367.35 | 316.66, 315.64, 328.91, 317.07, 310.98 | 47.09 | -46.85 (-12.9 %) | yes | yes |
| host | f32 | 6 | Dense | 201.49, 196.04, 201.62, 194.51, 193.00 | 198.76, 197.18, 197.82, 197.68, 195.78 | 8.62 | +1.64 (+0.8 %) | yes | yes |
| host | f32 | 6 | Rotation | 352.10, 347.63, 339.24, 305.55, 343.16 | 282.18, 303.69, 306.50, 307.81, 281.43 | 46.55 | -39.47 (-11.5 %) | yes | yes |
| host | f32 | 8 | PlaneWave | 119.47, 118.58, 123.02, 119.87, 120.82 | 124.85, 127.21, 117.21, 118.65, 119.47 | 4.43 | -0.40 (-0.3 %) | yes | yes |
| host | f64 | 3 | Batched | 95.53, 96.30, 96.21, 96.30, 98.45 | 97.49, 94.71, 95.39, 96.05, 96.29 | 2.92 | -0.25 (-0.3 %) | yes | yes |
| host | f64 | 6 | Batched | 162.35, 157.09, 158.80, 156.25, 157.20 | 166.99, 158.01, 159.14, 159.60, 158.85 | 6.10 | +1.94 (+1.2 %) | yes | yes |
| host | f64 | 6 | Classes | 476.75, 429.49, 438.49, 433.48, 429.01 | 434.01, 459.91, 429.05, 434.49, 428.74 | 47.74 | +0.53 (+0.1 %) | yes | yes |
| host | f64 | 6 | Dense | 365.68, 322.57, 322.79, 320.80, 327.35 | 328.40, 327.65, 327.31, 324.43, 326.81 | 44.88 | +4.51 (+1.4 %) | yes | yes |
| host | f64 | 6 | Rotation | 333.34, 321.05, 325.10, 335.08, 321.54 | 328.68, 329.33, 330.12, 326.60, 325.48 | 14.03 | +3.58 (+1.1 %) | yes | yes |
| host | f64 | 8 | PlaneWave | 178.36, 169.22, 176.05, 169.99, 169.25 | 171.53, 169.53, 173.83, 169.31, 173.08 | 9.13 | +1.54 (+0.9 %) | yes | yes |

All 12 are within noise, and the errors are equal in every run. The two Classes and
Rotation f32 rows are 11–13 % faster on the branch, all of it in M2L (Classes 249–256 ms
against `main`'s 280–327). The f64 rows show nothing similar. `main`'s run 8 of Rotation
(234 ms) equals the branch, so this is run-to-run variance on the M3 Max's mixed cores,
or layout as in §3.3; it is not counted as a gain.

### 3.2 locust

Idle before timing: load 0.0, the H100 at 0 %, no other user's process. Runs from
00:42 to 01:05, in `/data/ucahtbe/logs/ch2-0/`.

| backend | precision | p | strategy | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians (ms) | within noise | errors equal |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| host | f32 | 3 | Batched | 44.36, 44.59, 45.34, 45.00, 44.62 | 45.80, 46.43, 44.54, 44.30, 45.82 | 0.98 | +1.17 (+2.6 %) | NO | yes |
| host | f32 | 6 | Batched | 59.63, 58.77, 59.00, 58.92, 59.17 | 58.76, 58.77, 58.63, 59.01, 59.67 | 0.86 | -0.23 (-0.4 %) | yes | yes |
| host | f32 | 6 | Classes | 92.66, 93.13, 93.90, 92.67, 93.70 | 93.21, 92.87, 93.26, 92.95, 92.79 | 1.24 | -0.18 (-0.2 %) | yes | yes |
| host | f32 | 6 | Dense | 78.20, 77.33, 76.79, 77.23, 76.64 | 77.67, 79.00, 76.95, 77.26, 76.83 | 1.56 | +0.03 (+0.0 %) | yes | yes |
| host | f32 | 6 | Rotation | 92.07, 91.92, 92.02, 92.19, 92.75 | 93.41, 92.62, 92.03, 92.67, 92.14 | 0.83 | +0.56 (+0.6 %) | yes | yes |
| host | f32 | 8 | PlaneWave | 65.22, 65.31, 65.91, 64.47, 65.14 | 65.01, 64.66, 64.40, 65.25, 64.84 | 1.44 | -0.38 (-0.6 %) | yes | yes |
| host | f64 | 3 | Batched | 80.38, 80.98, 80.73, 79.22, 76.76 | 78.71, 78.20, 81.39, 80.49, 78.83 | 4.22 | -1.56 (-1.9 %) | yes | yes |
| host | f64 | 6 | Batched | 97.49, 100.88, 99.04, 98.51, 98.56 | 99.71, 100.25, 100.30, 100.58, 99.20 | 3.40 | +1.69 (+1.7 %) | yes | yes |
| host | f64 | 6 | Classes | 146.08, 148.73, 145.44, 144.79, 150.12 | 143.78, 145.38, 145.81, 145.42, 145.53 | 5.32 | -0.66 (-0.5 %) | yes | yes |
| host | f64 | 6 | Dense | 139.44, 138.77, 137.98, 138.31, 137.68 | 138.43, 137.91, 138.22, 137.77, 137.92 | 1.76 | -0.39 (-0.3 %) | yes | yes |
| host | f64 | 6 | Rotation | 127.16, 127.55, 124.39, 122.62, 124.39 | 126.22, 124.59, 125.91, 124.77, 126.07 | 4.93 | +1.52 (+1.2 %) | yes | yes |
| host | f64 | 8 | PlaneWave | 101.51, 104.65, 101.82, 100.79, 101.55 | 101.84, 100.77, 103.86, 102.19, 104.58 | 3.86 | +0.64 (+0.6 %) | yes | yes |

f32 p = 3 Batched fails the rule by 0.2 ms. The operator kinds the drivers run are equal
on both sides to about 0.1 ms in every run: P2M 0.79–0.85 ms, M2M 0.36–0.41, M2L
3.19–3.28, L2L 0.66–0.72, L2P 1.42–1.56, P2P 9.36–9.44. About 65 % of this benchmark's
evaluation is outside the operators (the moves, the output pass), code the change does
not touch. The same benchmark was repeated at 01:06–01:08, alone, idle, in the same order:

| backend | precision | p | strategy | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians (ms) | within noise | errors equal |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| host | f32 | 3 | Batched | 45.65, 45.69, 45.30, 45.49, 45.14 | 44.91, 45.30, 45.60, 45.26, 45.53 | 0.55 | -0.19 (-0.4 %) | yes | yes |

The repeat passes. `main`'s own medians moved from 44.4–45.3 to 45.1–45.7 ms between the
two passes: a drift of about 1 ms against a five-run spread of 0.5–1.0 ms.

### 3.3 Kathleen

Job 614707, one node in `singlenode` (node-c11a-110, Xeon Gold 6248, 40 cores, exclusive).
Both trees were built in the job first; the bit check (§2) ran before the timings.

| backend | precision | p | strategy | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians (ms) | within noise | errors equal |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| host | f32 | 3 | Batched | 75.64, 76.45, 73.05, 76.12, 76.79 | 73.91, 74.98, 77.25, 75.80, 75.19 | 3.74 | -0.93 (-1.2 %) | yes | yes |
| host | f32 | 6 | Batched | 111.26, 110.29, 108.91, 109.97, 108.26 | 108.90, 111.16, 109.93, 108.21, 111.25 | 3.00 | -0.05 (-0.0 %) | yes | yes |
| host | f32 | 6 | Classes | 285.83, 291.30, 288.23, 287.74, 286.05 | 291.49, 289.68, 290.65, 289.26, 291.26 | 5.47 | +2.91 (+1.0 %) | yes | yes |
| host | f32 | 6 | Dense | 190.76, 190.11, 195.34, 191.72, 189.43 | 192.65, 189.37, 189.81, 189.55, 190.68 | 5.91 | -0.94 (-0.5 %) | yes | yes |
| host | f32 | 6 | Rotation | 314.17, 313.43, 308.25, 310.11, 314.51 | 306.07, 300.58, 302.59, 301.59, 305.22 | 6.26 | -10.84 (-3.5 %) | yes | yes |
| host | f32 | 8 | PlaneWave | 148.64, 151.09, 152.20, 150.14, 154.84 | 152.34, 151.00, 152.06, 148.88, 154.06 | 6.20 | +0.97 (+0.6 %) | yes | yes |
| host | f64 | 3 | Batched | 112.50, 110.48, 110.99, 104.44, 104.59 | 105.18, 109.08, 112.13, 110.27, 107.59 | 8.06 | -1.40 (-1.3 %) | yes | yes |
| host | f64 | 6 | Batched | 151.85, 157.55, 155.69, 158.08, 160.54 | 149.68, 156.92, 155.60, 159.27, 149.31 | 8.68 | -1.95 (-1.2 %) | yes | yes |
| host | f64 | 6 | Classes | 364.21, 370.35, 378.58, 373.55, 371.03 | 391.23, 390.48, 394.93, 390.69, 395.25 | 14.37 | +20.20 (+5.4 %) | NO | yes |
| host | f64 | 6 | Dense | 310.76, 307.38, 311.79, 305.92, 317.77 | 322.86, 324.30, 330.77, 327.41, 331.85 | 11.85 | +16.65 (+5.4 %) | NO | yes |
| host | f64 | 6 | Rotation | 363.42, 367.35, 372.39, 365.40, 369.57 | 356.26, 358.81, 360.92, 356.66, 356.04 | 8.97 | -10.69 (-2.9 %) | yes | yes |
| host | f64 | 8 | PlaneWave | 203.82, 207.02, 203.44, 203.18, 209.85 | 207.42, 201.06, 204.37, 199.45, 202.24 | 6.68 | -1.59 (-0.8 %) | yes | yes |

f64 p = 6 Dense and Classes fail, and not by noise: every branch run is slower than every
`main` run. The per-kind times put the difference in M2L's target-by-target path:
- Dense: M2L 190–197 ms on `main`, 201–207 on the branch;
- Classes: M2L 243–252 ms on `main`, 272–282 on the branch.

In the same runs the branch's P2M (8.7 against 9.4 ms) and L2P (16.4 against 17.4) are
faster, and so is Rotation by 3 % in both precisions.

**The cause: code placement.** The functions that do the work are the same machine code
in both binaries:
- `MatrixSet<f64>::apply` (Dense), 0x262 bytes;
- `M2lClasses<f64>::apply`, 0x449 bytes;
- each in two copies, called out of line from the target-by-target closure, in both
  binaries.

Everything is generic over T, so it is monomorphised in `nd-fmm-bench`'s 16 codegen units.
Moving the bodies into the driver changed how the units are partitioned, and so where
those copies lie. Their 64-byte alignment differs between the binaries: one
`MatrixSet::apply` copy at offset 48 on `main` and 0 on the branch, one
`M2lClasses::apply` copy at 0 and 32.

The test, job 614723 (node-c11b-040), 01:27–01:42: both trees rebuilt with
`RUSTFLAGS="-C llvm-args=-align-loops=64"` (every loop on a 64-byte boundary, separate
target directories), then the same A B B A order for the f64 p = 6 rows:

| backend | precision | p | strategy | main medians (ms) | branch medians (ms) | spread (ms) | Δ median of medians (ms) | within noise | errors equal |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| host | f64 | 6 | Classes | 394.72, 396.98, 405.79, 406.54, 399.31 | 403.85, 404.83, 404.28, 407.45, 401.28 | 11.82 | +4.98 (+1.2 %) | yes | yes |
| host | f64 | 6 | Dense | 332.76, 331.22, 339.91, 330.33, 325.63 | 331.60, 330.57, 321.23, 331.14, 324.79 | 14.28 | -0.65 (-0.2 %) | yes | yes |
| host | f64 | 6 | Rotation | 363.54, 373.53, 368.25, 365.14, 363.91 | 365.44, 373.11, 366.55, 360.69, 366.23 | 10.00 | +1.09 (+0.3 %) | yes | yes |

With placement taken out, `main` and the branch are equal. M2L Dense: `main` 206–210 ms,
branch 209–215. Classes: 279–290 and 280–286. Rotation: 241–245 and 239–245. The default
layout happened to favour `main` in Dense and Classes and the branch elsewhere: `main`'s
Dense M2L also loses about 7 % once aligned.

The difference is the default build's code layout, not more work or worse code in the
drivers. It is still a measured +5.4 % on the default build of a machine of record. D9 as
written fails these two rows, so they are put to the user (§4).

## 4. Questions for sign-off

1. **Kathleen f64 p = 6 Dense and Classes, +5.4 %** (§3.3). The recommendation is to accept
   CH2.0. The rows differ by code placement, shown by the aligned build. Dense and Classes
   are not the defaults (`Auto` resolves to Batched and PlaneWave on the host), and the
   same layout effect makes Rotation 3 % faster.

   The alternatives:
   - Hold CH2.0 until a layout that passes everywhere is found. That is fragile: the next
     change to any generic code can move it again.
   - The fallback of §10: leave Laplace's operator as on `main` and give the Helmholtz
     operators copies of the drivers.
2. **locust f32 p = 3 Batched** (§3.2): failed by 0.2 ms in the first pass and passed on
   the repeat. The recommendation is to count the repeat, and to record in D9 that a
   failing row may be repeated once, with both results reported.

## 5. Checks

| command | where | result |
| --- | --- | --- |
| `cargo fmt --all -- --check` | M3 Max | passed |
| `cargo clippy -- -D warnings`, `--examples` | M3 Max | passed |
| `cargo clippy --workspace --all-targets -- -D warnings` | M3 Max | passed |
| `cargo clippy -p nd-fmm-exec --all-targets --features cpu,metal -- -D warnings` | M3 Max | passed |
| `cargo check -p nd-fmm-exec --features cuda` | M3 Max | passed |
| `cargo doc --no-deps`; `RUSTDOCFLAGS=-D warnings cargo doc -p nd-fmm-exec --document-private-items` | M3 Max | passed |
| `RUST_MIN_STACK=8388608 cargo test` | M3 Max | passed (976) |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | M3 Max | passed (1055) |
| `run-mpi-tests.sh run … 2 4` (`mpi_regressions`, `mpi_exec`, `mpi_threading`, `invariance`, 7 octree examples) | M3 Max | passed |
| `run-mpi-tests.sh run … 8` (`mpi_exec`) | M3 Max | passed (216 s) |
| `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored` (`accuracy`, `adaptive`, `dipoles`, `multi_rank` at one rank, …) | M3 Max | passed |
| `multi_rank` at 2, 4 and 8 ranks, release, `mpirun` with the loopback flags | M3 Max | pending (running at commit time) |
| the bit check of §2 | M3 Max, locust, Kathleen | passed (18 of 18 each) |
| D9 (§3) | M3 Max, locust, Kathleen | M3 Max passed; locust passed on the repeat; Kathleen two rows for sign-off |

Not run:
- the licence check (no manifest changed);
- the device tests and gates (no device code changed: `device.rs` uses `LaplaceOperator`'s
  public API and `Tables`, both unchanged, and the default device evaluation does not call
  the host drivers; CI type-checks CUDA);
- the C3S.4 P2P throughput gate (`nd-fmm-simd` unchanged).

The tuner's and node-m2l.md's cost constants are not re-measured. They are fixed inputs
that change no bit of any output, and the default strategies (Batched, PlaneWave) moved by
no more than their noise.

# Validation at scale (Phase 5S T2, C5S.1; S7)

S7 of docs/design/distributed-fmm.md §14.4, as docs/phase5s/T2-validation-at-scale.md asks and
docs/design/scale-out.md §5 designs: rank-count invariance as a test and at scale, the
`scaling` harness at N = 10⁸ without a one-rank reference, and a Kathleen correctness job.
Run on 2026-10-12 on the M3 Max, locust (GH200) and Kathleen. No library code changed
(`nd-octree`, `nd-fmm-plan` and `nd-fmm-exec`'s library are untouched); no defect was found.

**Measured** means run here and read from the output; times are for the record, never
asserted, and oversubscribed runs are never timings. Kathleen nodes are exclusive; every job
is named.

## Summary

- **The invariance test** (`nd-fmm-exec`, `tests/invariance.rs`, its own MPI executable,
  registered in `.github/scripts/run-mpi-tests.sh`): 0 values differ on every rank count
  run: 1, 2, 4 and 8 ranks on the M3 Max and on locust, 2, 4, 8 and 80 ranks on Kathleen
  (2 nodes; at 80 ranks every divisor k of 80, 54 checks). A concatenation in the wrong
  order fails on every rank (§2.3).
- **At scale** (`scaling --invariance`): 0 values differ at 512 against 64 ranks on locust
  (oversubscribed), and on Kathleen at 80 against 40, 20, 10 (2 nodes), 160 against 80,
  40, 20 (4 nodes) and 640 against 80 (4 nodes, oversubscribed): 8 × 10⁷ values compared
  per check at N = 10⁷, none gathered.
- **The harness at N = 10⁸** (Kathleen, 4 nodes, no one-rank reference): the errors against
  the distributed direct sum at 160 ranks equal those at 40 ranks (a quarter) in every
  printed digit, f64 p = 8 and f32 p = 3, so within 0.05% (§4). The direct sum's part per
  rank is the rank's sources times 1,000 targets, 6.3 × 10⁸ pairs at 160 ranks and
  2.5 × 10⁹ at 40 (4×). The harness itself holds 56–62 MB per rank at 160 ranks and
  113–118 MB at 40, after the input and before any build (O(N/P); the 10⁸ points alone
  would be 2.4 GB). `--reference` is refused above N = 2 × 10⁷.
- **The Kathleen correctness job** (`tools/kathleen/jobs/correctness.sbatch`) passes at
  2 nodes in `test` (job 447987, 19 min 38 s): the MPI list at 2, 4, 8 and 80 ranks, the
  invariance at 80 against 40, 20, 10 and the smoke run. The first attempt (447359) met a
  node whose Omni-Path HFI refused contexts (§5.2).
- **CI** (the pull request's run 38252400249, every job passed): the invariance test took
  55.6 s at 2 ranks and 110.1 s at 4 in `run-tests-mpi` (run steps of 15 minutes, launches
  of 300 s) and 64.0 s at one rank in `run-tests`; `run-tests-mpi` took 10 min 58 s in all.
- Node-hours on Kathleen: 1.86 (three jobs, §6), of the 10 planned (20 budgeted).

## 1. What was built

| Piece | Where | What |
| --- | --- | --- |
| the invariance test | `fmm-exec/tests/invariance.rs` (new; `run-mpi-tests.sh`) | P ranks against P/k for every divisor k > 1 (k = 1 on one rank); every input by index (a seeded random share); the small runs on `split` of the world (colour 0 on ranks k r, none elsewhere); the outputs moved back point to point by index; one all-reduce of the counts, so it fails on every rank or none. The cube, the Plummer sphere (a = 0.1) and a graded tree with one empty rank (a fifth of the points over the unit cube, the rest in a cube of side 2⁻⁶; the last rank's share to rank 0), N = 10⁴, `max_level` 16, 64 points per leaf, gradients; f64 p = 8 `Rotation` with the large run blocking and the small ones overlapped, f32 p = 3 `Dense` the other way round |
| the check at scale | `nd_fmm_validate::scaling::invariance`, `scaling --invariance k[,k…]` | the same in one launch for each k; the strategy of the large run named in the small ones (`Auto` resolves by the ranks per node); a waiting rank sleeps between tests of its receive, so oversubscribed small runs get the cores; the values that differ per check reduced (sum, most on a rank, ranks with any, the first); exit status 1 on every rank if any differs |
| inputs by index | `Input::Ranges` (`--input ranges`), `input_by_index` | contiguous index ranges in O(N/P) time (the share hashes every index on every rank, O(N); scale-out.md §5.2: the share to N = 10⁹, ranges beyond); the concatenated ranges of k ranks are the ranges at P/k |
| refusals | `HarnessError`; `run` and `invariance` return a `Result` | `--reference` above `REFERENCE_MAX_POINTS` = 2 × 10⁷; a factor that does not divide P; the owners' input for invariance. Each decided from the arguments, on every rank before any collective; the example prints the reason on rank 0 and exits 2 on every rank |
| the direct sum's cost and the harness's memory | `Errors::{direct_pairs, direct_seconds}`, `RankRow::resident_input` | the pairs and time of a rank's part of the direct sum (each rank over its own sources, Neumaier-compensated in f64, one all-reduce); the resident size after the input, before the builds |
| the correctness job | `tools/kathleen/jobs/correctness.sbatch` | parts `mpi`, `invariance`, `smoke` (default, 2 nodes in `test`), `large`, `oversubscribed` (4 nodes in `small`); every launch a step under `timeout`, the job ending with a table of results and wall times (tools/kathleen/README.md, "The correctness job") |

Two choices against the brief's wording, both from T1's design (scale-out.md §5.1, §11):
the test is its own executable, so that `mpi_exec` keeps its minute, and the strategy is
named in every run. And one for the debug budget: the brief asks for "overlap off and on";
each `Fmm` of the test evaluates once, and every comparison is blocking against overlapped
across rank counts (f64: the large run blocking, the small runs overlapped; f32 the other
way round). Evaluating both paths on both sides took 57 s at one rank against 39 s.

## 2. The invariance test

### 2.1 Runs

Debug builds, `--test-threads=1 --nocapture`, under `timeout`; seconds of the whole test.

| Machine | 1 rank | 2 | 4 | 8 | 80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M3 Max (Open MPI 5.0.10, loopback flags) | 39.0 | 30.9 | 38.2 | 42.9 | – |
| locust (Open MPI 5.0.10; load 0.1 before) | 76.0 | 58.6 | 70.5 | 78.6 | – |
| Kathleen 447987 (2 nodes, `--bind-to core`; 80 ranks across both) | – | 49.2 | 59.3 | 69.1 | 135.6 |

Every run: 0 values differ, in every check (6 at 1 and 2 ranks, 12 at 4, 18 at 8, 54 at
80: k = 2, 4, 5, 8, 10, 16, 20, 40, 80). On the M3 Max at 4 ranks per check (seconds for the
large run, then for the small one on rank 0): the cube f64 1.3 / 2.6 (k = 2) and 5.1
(k = 4); the graded tree f64 5.7 / 8.5 and 16.0 (before the one-evaluation change of §1).
In Kathleen job 447359 (the failed attempt, §5.2) the same test passed at 2, 4 and 8 ranks
(50.2, 59.8, 69.3 s) on the node that later refused contexts.

### 2.2 CI

`run-tests-mpi` runs it at 2 and 4 ranks, `run-tests` (`cargo test`) at one. Measured in
the pull request's CI run (run 38252400249, `ubuntu-latest`, 4 vCPUs on 2 cores, the ranks
oversubscribed; every job passed):

| Job | Ranks | `invariance`, s | `mpi_exec`, s (for scale) |
| --- | ---: | ---: | ---: |
| `run-tests` (`cargo test`) | 1 | 64.0 | – |
| `run-tests-mpi` | 2 | 55.6 | 187.8 |
| `run-tests-mpi` | 4 | 110.1 | 230.2 |

0 values differ in every check (6 at 1 and 2 ranks, 12 at 4). The added time is within
`RUN_TIMEOUT` (300 s per launch) and the run steps' 15 minutes; `run-tests-mpi` took
10 min 58 s in all, `run-tests` 7 min 28 s. Before the run it was estimated at 50–105 s per
step from `mpi_exec`'s ratio of CI to the M3 Max (166.1 / 224.8 s in CI run 38240370553
on `main` against 98.4 / 82.9 s locally; `invariance` 30.9 / 38.1 s locally): close, the
4-rank run 5% over. The one-rank run in `run-tests` is 64 s, a little over the minute that
fmm-exec/CLAUDE.md asks of the debug run of a test; it is its own executable (the brief's
alternative to `mpi_exec`'s budget).

### 2.3 A broken input order fails on every rank

By hand on the M3 Max, not committed: the small run's concatenation in reverse rank order
(`(0..k).rev()`, the slices still sent back by their offsets), 4 ranks. The first check
fails on all four ranks, each with its own count, and every rank panics on the all-reduced
total (test exit 101):

```text
rank 0: cube f64 p = 8 Rotation: 15962 of 19952 values differ between 4 ranks and 2 ranks (k = 2)
rank 2: cube f64 p = 8 Rotation: 15958 of 19968 values differ between 4 ranks and 2 ranks (k = 2)
rank 1: cube f64 p = 8 Rotation: 15928 of 20016 values differ between 4 ranks and 2 ranks (k = 2)
rank 3: cube f64 p = 8 Rotation: 15994 of 20064 values differ between 4 ranks and 2 ranks (k = 2)
cube f64 p = 8 Rotation: 4 ranks against 2 (k = 2): 63842 of 80000 values differ, on 4 rank(s)
```

(Then a version that compared both paths of the large run, hence 80,000 values.) The order
within a leaf is (origin rank, origin position), so the reversed concatenation reorders the
points of shared leaves and every sum over them changes in its last bits: the check sees
the input order, as requirement 3 needs.

## 3. Invariance at scale

`scaling --invariance`, release, f64 p = 8 `Rotation` unless stated, the random share,
`--overlap both` (the large run evaluates both paths, each compared with the small run's
overlapped output: values compared = 2 paths × 4 values × N). Seconds: the small run's
build and evaluation on its slowest rank.

| Machine, job | P | against P/k | Workload | values compared per check | differing | points per small rank (max) | small run, s |
| --- | ---: | --- | --- | ---: | ---: | --- | --- |
| locust, oversubscribed (`--map-by :OVERSUBSCRIBE --bind-to none`) | 512 | 64 | cube N = 10⁷ | 8 × 10⁷ | 0 | 157,154 | 1.82 |
| locust, oversubscribed | 512 | 64 | Plummer N = 10⁶ | 8 × 10⁶ | 0 | 15,846 | 0.48 |
| Kathleen 447987, 2 nodes × 40 | 80 | 40, 20, 10 | cube N = 10⁶ | 8 × 10⁶ | 0, 0, 0 | 25,307–100,613 | 0.72, 1.33, 2.55 |
| Kathleen 447987 | 80 | 40, 20, 10 | Plummer N = 10⁶ | 8 × 10⁶ | 0, 0, 0 | 25,307–100,613 | 1.62, 2.95, 5.84 |
| Kathleen 447987 | 80 | 40, 20, 10 | cube N = 10⁶, f32 p = 3 `Dense` | 8 × 10⁶ | 0, 0, 0 | 25,307–100,613 | 0.12, 0.21, 0.38 |
| Kathleen 447366, 4 nodes × 40 | 160 | 80, 40, 20 | cube N = 10⁷ | 8 × 10⁷ | 0, 0, 0 | 126,107–501,425 | 3.23, 6.24, 12.33 |
| Kathleen 447366 | 160 | 80, 40, 20 | Plummer N = 10⁷ | 8 × 10⁷ | 0, 0, 0 | 126,107–501,425 | 8.81, 16.47, 29.63 |
| Kathleen 447366 | 160 | 80, 40, 20 | cube N = 10⁷, f32 p = 3 `Dense` | 8 × 10⁷ | 0, 0, 0 | 126,107–501,425 | 0.63, 1.09, 2.07 |
| Kathleen 447366, 4 nodes × 160, oversubscribed (`--oversubscribe --map-by ppr:160:node --bind-to none -x PSM2_SHAREDCONTEXTS=1`) | 640 | 80 | cube N = 10⁷ | 8 × 10⁷ | 0 | 125,807 | 3.48 |

Largest resident size of a rank at the end: 349 MB (locust, cube) and 228 MB (Plummer);
on Kathleen 381 MB (160, cube), 528 MB (160, Plummer), 293 MB (160, f32) and 302 MB (640).
The 640-rank launch took 97 s in all, 87 s of it the large run on 4 × oversubscribed
cores; the small run's 80 ranks took 3.5 s, while the other 560 ranks slept in their
receives. The small runs at P/8 hold 5 × 10⁵ points per rank at N = 10⁷ (scale-out.md
§5.1's "under 2 GB").

## 4. The harness at N = 10⁸ (Kathleen 447366, 4 nodes)

The cube, N = 10⁸, the random share, `Rotation` named, overlap off, two timed evaluations,
the errors of one charge vector at 1,000 sampled targets against the distributed direct sum,
no one-rank reference. 160 ranks = 4 × 40 (`ppr:20:socket`), 40 ranks = 4 × 10
(`ppr:5:socket`), one thread each.

| Precision, p | Ranks | φ L2 | φ max | ∇φ L2 | ∇φ max | direct sum: pairs per rank (max) | direct sum, s (slowest rank) | harness after the input, MB (min–max) | resident at the end, MB (max) | evaluation, ms |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| f64, 8 | 160 | 2.468e-5 | 6.205e-5 | 1.669e-5 | 8.059e-6 | 627,710,000 | 8.44 | 56.4–61.8 | 446 | 14,274 |
| f64, 8 | 40 | 2.468e-5 | 6.205e-5 | 1.669e-5 | 8.059e-6 | 2,503,424,000 | 27.15 | 113.8–117.4 | 1,324 | 44,620 |
| f32, 3 | 160 | 3.781e-3 | 6.496e-3 | 1.377e-3 | 2.996e-4 | 627,710,000 | 8.36 | 56.6–62.5 | 307 | 2,858 |
| f32, 3 | 40 | 3.781e-3 | 6.496e-3 | 1.377e-3 | 2.996e-4 | 2,503,424,000 | 27.16 | 112.8–118.0 | 859 | 8,930 |

- **Against a quarter of the ranks**: every error equal in all four printed digits, so the
  relative differences are below 5 × 10⁻⁴ (the brief: 0.1% in f64, 1% in f32). The FMM
  outputs differ between the two runs in the last bits (a random share at each rank count,
  so the leaves' orders differ), and so do the direct sums (the parts added by one
  all-reduce in an order that depends on P); neither shows at four digits. f32 runs at p = 3,
  where truncation, not f32 rounding, sets the error.
- **The direct sum is O(N/P) per rank**: the largest part is the rank's sources times the
  1,000 targets, 4.0× more pairs at 40 ranks than at 160 (2.50 × 10⁹ against 6.28 × 10⁸),
  3.2× the time (27.2 s against 8.4 s; 10 ranks per node get more memory bandwidth than 40).
  It is Neumaier-compensated over each rank's sources in f64 (`nd_fmm_ref::p2p::direct_sum`);
  the P parts are added by one all-reduce of 4,000 values per vector.
- **The harness holds O(N/P)**: after the input, before any build, a rank's peak resident
  size (`VmHWM`) is 56–62 MB at 2.5–6.3 × 10⁵ points per rank and 113–118 MB at 2.5 × 10⁶:
  55 MB more for 1.9 × 10⁶ more points, 29 bytes per point (8 for the index, 24 for the
  coordinates). Rank 0 is no exception. The O(N) items would be 0.8 GB (every index) or
  2.4 GB (every point); none exists. The share still hashes all 10⁸ indices on every rank
  (O(N) time, not timed separately here); `--input ranges` avoids it.
- **The one-rank reference** is refused above 2 × 10⁷ points (M3 Max, 2 ranks, N = 3 × 10⁷:
  the reason on rank 0, exit status 2 on both ranks, no point generated).

## 5. The Kathleen correctness job

### 5.1 Job 447987 (2 nodes, `test`, node-c11b-[027-028], waited 0 s, ran 19 min 38 s)

| Step | Result | Seconds |
| --- | --- | ---: |
| mpi-build (debug, the MPI list) | passed | 17.8 |
| mpi-2-4-8-80 (10 executables × 4 rank counts) | passed (40 of 40) | 1,103.7 |
| scaling-build (release) | passed | 3.7 |
| invariance, cube f64 p = 8, 80 against 40, 20, 10 | passed | 6.2 |
| invariance, Plummer f64 p = 8 | passed | 13.6 |
| invariance, cube f32 p = 3 | passed | 1.6 |
| smoke, 2 nodes × 2 ranks × 20 threads | passed | 23.5 |

The MPI list at 80 ranks: `mpi_regressions` 8.9 s, `mpi_exec` 241.9 s, `invariance`
135.6 s, the octree examples 0.8–35.5 s. The smoke run (the N = 10⁶ cube, f64 p = 3,
`Auto` resolved to Dense, 4 ranks × 20 threads): φ L2 2.913e-3, ∇φ L2 4.673e-3 against the
direct sum (8 vectors), 1.04e-15 relative L2 against the one-rank FMM (a random share, so
not bit for bit), overlap and blocking identical in every bit; the harness 42.4 MB per rank
after the input.

### 5.2 Job 447359 (the first attempt): a node refused HFI contexts

On node-c11a-011 every launch at 80 ranks failed in `MPI_Init`: PSM2 could not open an
endpoint ("assign_context command failed: Device or resource busy", 260 times, all on that
node), from the first 80-rank launch on; the same executables passed at 2, 4 and 8 ranks on
it, and the smoke run at 2 ranks per node passed. That is the stale-context state that T1
saw after a failed 1,280-rank launch (scale-out.md §10.1), here apparently present before
the job started: the first 80-rank launch of the job already failed, and nothing before it
had failed on that node. The rerun excluded it
(`--exclude=node-c11a-011`) and passed (447987). The 4-node job (447366) ran on other nodes.

### 5.3 Job 447366 (4 nodes, `small`, node-c11a-[020,025,044,068], waited 7 min 54 s, ran 10 min 39 s)

`correctness.sbatch invariance large oversubscribed` with `INV_N=10000000` and
`LAUNCH_TIMEOUT=3600`: 11 of 11 steps passed (§3, §4); the large launches took 69 s (f64,
160), 217 s (f64, 40), 23 s (f32, 160) and 72 s (f32, 40), the 640-rank invariance 97 s.

## 6. Checks and node-hours

| Check | Where | Result |
| --- | --- | --- |
| `cargo fmt --all -- --check` | M3 Max | pass |
| `cargo clippy -- -D warnings`, `--examples`, `cargo doc --no-deps` | M3 Max | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | M3 Max | pass |
| `RUST_MIN_STACK=8388608 cargo test` | M3 Max | pass: 710 passed, 53 ignored (58 executables) |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | M3 Max | pass: 755 passed, 53 ignored |
| `nd-fmm-validate`'s `tests/scaling.rs` at 2 and 4 ranks (and 1 in `cargo test`) | M3 Max | pass |
| `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks, the new executable included | M3 Max, locust | pass (20 of 20 launches each) |
| the same at 2, 4, 8 and 80 ranks | Kathleen 447987 | pass (40 of 40) |

| Job | Nodes | QoS | Waited | Ran | Node-hours | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 447359 | 2 (node-c11a-[011,059]) | test | 1 s | 14:57 | 0.50 | 4 steps failed: node-c11a-011's HFI contexts (§5.2) |
| 447366 | 4 (node-c11a-[020,025,044,068]) | small | 7:54 | 10:39 | 0.71 | 11 of 11 steps pass |
| 447987 | 2 (node-c11b-[027-028]) | test | 0 s | 19:38 | 0.65 | 8 of 8 steps pass |
| **total** | | | | | **1.86** | of 10 planned (20 budgeted) |

Every Kathleen job: Open MPI 4.1.6 over PSM2 (`PSM2_TID=0`, env.sh), `mpirun` inside the
allocation, ranks bound per core (`--map-by ppr:<k>:socket:PE=1 --bind-to core`) except the
oversubscribed launch, 1 thread per rank except the smoke run (20), every BLAS thread
variable at 1.

# Phase 6 T13: the plane-wave M2L as a host strategy (C6.5)

Phase 6, task T13 ([docs/phase6/T13-plane-wave-m2l.md](../../docs/phase6/T13-plane-wave-m2l.md));
the spike and decision 6 are `spikes/plane-wave/SPIKE_REPORT.md` (Phase 6 T10, signed off
on 2026-10-18: C6.5 proceeds). Measured on 2026-10-19 from branch `tbetcke/phase6_t13`
(main at c681404 plus this task's changes), release builds, rustc 1.99.0, every BLAS
thread variable at 1 (no BLAS is linked). The path taken: **with Kathleen**
(docs/phase6/README.md, "Kathleen is usable"): one node for the timings, two for the
correctness job; no job used more than two nodes.

## Summary

- **Built**: `M2lStrategy::PlaneWave`, an opt-in host strategy (`Auto` never resolves to it),
  in f64 and f32 at every p ≤ 20: FMM3D's quadratures with s per p chosen on a whole
  level (s = 10, 15, 17, 21, 32 at p = 3, 6, 8, 12, 18), merge and shift by geometric masks,
  one frame at a time, SIMD kernels on NEON, AVX2 and AVX-512. The convention is
  CONVENTIONS §3.14, outside the versioned set: **`CONVENTION_VERSION` stays 2** (proposed in
  the PR).
- **Correct**: the FMM's error under `PlaneWave` is **0.94–1.03× the exact M2L's** in every
  test and sweep (bound 1.1: the C3.2 and C3.3 gates at f64 p = 8, 12, 18, every
  `tests/mpi_exec.rs` scenario, the C5.1 gate, f32 included; in f32 at p = 12 and 18, where
  f32's rounding dominates every strategy's error, 0.81–1.05× the most accurate exact M2L's,
  `Rotation`). The output is the same bit
  for bit across evaluations, thread counts, overlap, ISAs (every kernel equals the scalar
  path bit for bit) and ranks (the one-rank run's on 2, 4, 8 and 80 ranks; rank-count
  invariance at 80 ranks against 40, 20 and 10, 0 of 8,000,000 values differ).
- **Faster from f64 p = 8 on every machine**: the evaluation, against the fastest of
  `Batched` (separate and fused, AVX2 and AVX-512 on Kathleen) and `Rotation`:

  | f64 | M3 Max | locust | Kathleen |
  | --- | --- | --- | --- |
  | p = 3, one rank | 0.94–0.97 | 0.94–1.01 | 0.83–0.99 |
  | p = 6, one rank | **1.08–1.20** | **1.03–1.13** | 0.96–0.99 |
  | p = 8, one rank / node splits | **1.28–1.80** / **1.23–1.73** | **1.14–1.48** / **1.05–1.50** | **1.08–1.25** / **1.03–1.24** |
  | p = 12, one rank / node splits | **1.82–3.47** / **2.81–3.54** | **1.80–3.08** / **2.44–3.79** | **1.35–2.32** / **1.95–2.32** |
  | p = 18, one rank | **2.81–6.38** | **2.41–5.89** | **2.04–5.55** |

  (the cube and the Plummer sphere at N = 10⁶, and 10⁷ on locust and Kathleen; one rank
  at one thread and every core; T2's ranks × threads splits.) The M2L stage alone is 1.5–3.2×
  faster at p = 8 and 6.5–11.4× at p = 18; the evaluation gains less where the leaves
  dominate (the Plummer sphere). In f32 it is 1.12–1.40× faster at p = 8 on the arm machines
  and 0.94–1.03× on Kathleen (§9).
- **Memory**: one frame at a time, 4H values per box (8.1 KB at f64 p = 8, 32 KB at
  p = 18) and 119–191 KB of scratch per thread at p = 8–12: 21–35 MB per rank at one rank
  per Kathleen core (N = 10⁶), +249 MB at one rank on the 10⁶ cube at p = 8 (§7).
- **The cost model** counts `PlaneWave`'s M2L per box (`cost::M2lUnit::Boxes`), fitted on
  Kathleen; with it the work cut takes the Plummer sphere's and the clusters' imbalance
  from 1.33–1.41 to 1.05–1.13 (evaluations 1.17–1.31× faster, §8).
- **For T11** (§10): `PlaneWave` as `Auto`'s choice for f64 at p ≥ 8 on every machine and
  from p = 6 on NEON, for f32 from p = 8 on NEON; the exact M2L elsewhere; the waves in
  the node budget.
- **Not run**: CUDA (on a device `PlaneWave` runs as `Dense`, checked bit for bit on the
  CPU runtime; the CUDA build type-checks).

## 1. What T13 adds

- **The convention** (step 1): CONVENTIONS §3.14, "Plane-wave M2L" (box widths, the
  quadrature, multipole to waves, half the α nodes, shifts, waves to local, directions,
  merge and shift), the spike's derivation (`spikes/plane-wave/src/planewave.rs`) written
  out. It changes no quantity of §3.1–§3.13, no fixture depends on it and its tables are
  not cached, so **`CONVENTION_VERSION` stays 2**; §3.10 says why §3.14 is outside the
  versioned set (like §3.13).
- **The quadrature and the tables** (step 2, `nd_fmm_tables::plane_wave`): FMM3D v2.1.0's
  quadratures for separation 1 (NIST's `lwtsexp_sep1.f`, its notice kept;
  `tools/fixtures/plane_wave_quadrature.sh`, from the spike's `tools/extract_quadrature.sh`), the quadrature per
  degree `nodes_for_degree` (§3), and `PlaneWaveTables<T>`: the outgoing powers
  (wₖ/Nₖ') λₖⁿ 2^{−(n+1)} by degree, the incoming powers 2 λₖⁿ 2^{−n}, cos mαⱼ and sin mαⱼ
  per λ node, the multipole and local rotations of the frames ±y and ±x, the 104 distinct
  source shifts (slot to parent, 2H values each) and the 48 child shifts, all built in f64
  and rounded to T; the merge geometry (`MergeGeometry`: the direction of each of the
  216 slots × 8 children by FMM3D's rule, the accumulators by geometric mask, a = 10 at
  most per direction); a pair translation `m2l` (the trusted slower path). Built in
  milliseconds, so not cached: 0.7 MB in f64 at p = 8, 3.0 MB at p = 18.
- **The strategy** (step 3, `nd-fmm-exec`): `M2lStrategy::PlaneWave` (`Auto` never
  resolves to it). Each M2L level call runs, per charge vector and per frame (±z, ±y,
  ±x), the waves of every box of the level (own and ghost, from their multipoles: no new
  exchange) and then merge and shift parent by parent (`src/operator_plane_wave.rs`; the
  `operator` module docs, "The plane-wave M2L"). M2M and L2L are `Rotation`'s. On a device
  it runs as `Dense` (the device path builds the dense tables; "PlaneWave, run as dense on
  the device"). `Tables::m2l_at` translates a pair along merge and shift's direction, so
  `PerPair` equals the level call to rounding. The quadratures end at p = 20, the
  builder's `MAX_DEGREE` (a compile-time assertion ties the two), so `FmmBuilder` refuses
  a larger p under `PlaneWave` as under every strategy (`SettingsError::DegreeTooLarge`,
  checked in `tests/mpi_exec.rs`).
- **Memory** (step 4): one frame at a time, so the waves of a level take 4H values per box
  (a third of the six directions'); `Fmm` sizes the buffer from the plan
  (`LaplaceOperator::with_plane_wave_boxes`), so no level call allocates (§7).
- **The SIMD bodies** (step 5, `nd_fmm_simd::WaveKernel`): `merge` (a parent's accumulator
  over its sources' shifted waves, R vectors of each part held in registers across the
  sources), `gather` (a child's accumulators and its shift), `waves` and `sums` (the
  Fourier passes, over the kernel's own tables padded to its lanes, `WaveTables`), on
  NEON, AVX2, AVX-512 and the scalar layer, one fma per product and sum in a fixed order
  per value: **every ISA equals the scalar path bit for bit** (§2), so the output does not
  depend on the ISA either. The kernels run on the P2P choice's ISA
  (`ThreadingReport::wave_kernel`).
- **The cost model** (step 6): under `PlaneWave` M2L counts the boxes with a V row
  (`cost::M2lUnit::Boxes`), with constants fitted on Kathleen (§8).
- **f32** (step 7): offered at every degree the builder accepts (p ≤ 20, as f64), measured
  at p = 3, 6, 8 on the three machines and at p = 12, 18 on the M3 Max; its errors meet the
  budget (§9).
- **The harness**: `planewave` in `scaling` and `nd-fmm-bench`, the plane waves' memory in
  the memory model, the example `plane_wave_parts` (every step per box per ISA), the sweeps
  `t13-host`, `t13-node` and `t13-calibrate` of `tools/scaling/run.sh`, PlaneWave in
  `tools/kathleen/jobs/correctness.sbatch`'s invariance part.

## 2. Correctness

Error measures as in each test's documentation: potentials against
`nd_fmm_ref::direct::m2l` relative to E Σ|q| (the tables), the bits (the kernels, threads,
overlap, ranks), relative L2 against the direct sum (the FMM).

**The tables** (`nd-fmm-tables`, `plane_wave`): the pair translation (each pair directly
along its merge direction) against `nd_fmm_ref::direct::m2l` on levels 2, 9 and 16 of the
dyadic unit domain (a parent's 216 slots into its 8 children), as potentials at four random
points of every child, within the bound E Σ_V |q| (E the quadrature's measured error, or
the waves' rounding floor of 5e-10 where E lies below it, from s ≈ 27): at p = 0, 3 and 8
in the debug run, at **every p ≤ 20** in the ignored `pair_m2l_every_degree` (release,
9.7 s on the M3 Max), the error 0.001–0.04 of the bound at p ≥ 16 and the same to every
printed digit on the three levels (the tables are level-independent). A source shift
followed by a child shift equals the pair's shift within 1e-15 (`source_and_child_
shifts_compose_to_the_pair_shift`); the geometry gives every V pair of an interior child
exactly one direction with an offset of 2 or 3 boxes along it, and an interior parent 248
entries, 152 of them serving all eight children (the spike's counts).

**The kernels** (`nd-fmm-simd`, `tests/waves.rs`): `merge`, `gather`, `waves` and `sums`
equal the reference loops (restated from `nd_fmm_tables::plane_wave`) **bit for bit** on
every ISA, f32 and f64, with wave lengths and λ-node shapes that are no multiples of any
lane count (every tail and padding path): scalar and NEON on the M3 Max and locust, scalar,
AVX2 and AVX-512 on Kathleen (job 605553, debug and release). A kernel refuses another lane
count's tables. Inner loops (release, `fmm-simd/tools/inner_loops.awk`; instructions /
fmas, no call in any): `merge` 42/16 on NEON, 51/16 on AVX2 and AVX-512 (four vectors of
each part); `waves` 32/16 (NEON), 25/8 (AVX2, a block of two vectors: four of each output
would exceed its 16 registers, and did, 5.8 against 4.2 µs per box), 35/16 (AVX-512);
`sums` 16/8 (NEON), 18/8 (AVX2, AVX-512).

**The operator and the FMM** (`nd-fmm-exec`):
- `tests/mpi_exec.rs`, `check_plane_wave` on every scenario, once per scenario and
  precision: the error against the direct sum at most **1.017** (f64) and **1.003** (f32)
  times the scenario's exact M2L's (φ and ∇φ; bound 1.1), the change from its output within
  its error, two evaluations and a build at 8 threads, overlapped, bit for bit, the kernels
  on the P2P ISA, M2L counted per box; on 2, 4 and 8 ranks also the one-rank `PlaneWave`
  `Fmm` over the union **bit for bit** (21 f64 and 3 f32 scenarios each; M3 Max, 264.6,
  221.7 and 204.9 s in debug for the whole test). **batched against per-pair**: merge and
  shift against `PerPair` (each pair directly) **2.2e-16** relative L2, and 2, 4, 8 threads
  overlapped bit for bit, at every rank count. In the debug run the check skips a
  scenario whose evaluation takes over 0.5 s (to keep CI's 4-rank run under its 600 s
  timeout; main took 496 s there): on the M3 Max at one rank 16 f64 and 2 f32 scenarios
  are checked (worst ratios 1.0133 and 1.0027, 14.6 s; the whole test 195.3 s, 204.1 s
  without the budget), and the strongly graded tree, the sources-only leaves next to
  targets-only ones and the multipole routing are left to the release runs (here the
  device run below, at one rank); Kathleen's correctness job 605561 ran before the budget
  existed and checked every scenario at 2, 4, 8 and 80 ranks in debug (below).
- `tests/invariance.rs`: every workload under `PlaneWave` at f64 p = 8 and the graded tree
  at f32 p = 6, blocking against overlapped, bit for bit across 1, 2, 4 and 8 ranks (M3
  Max; 79.6, 99.0, 111.5 s at 2, 4, 8 ranks for the whole test); at scale in Kathleen's
  correctness job (`correctness.sbatch`, job 605561, 2 nodes, `test`; PlaneWave added to its
  invariance part): at 80 ranks against 40, 20 and 10, the N = 10⁶ cube and Plummer sphere
  at f64 p = 8 under `PlaneWave`, blocking against overlapped, **0 of 8,000,000 values
  differ** in each of the six comparisons. The same job ran the MPI test list at 2, 4, 8
  and 80 ranks (every launch passes; `check_plane_wave` with the one-rank `Fmm` bit for
  bit at each, 21 f64 and 3 f32 scenarios, worst error ratio 1.0172 and 1.0027) and the
  smoke run.
- The C3.2 gate (`tests/accuracy.rs`, the uniform level-4 tree, N = 10⁵, eight charge
  vectors) and the C3.3 gate (`tests/adaptive.rs`, the sphere, the Plummer sphere and the
  clusters at N = 10⁵), `PlaneWave` against the exact M2L (`Rotation`, whose output is
  `Dense`'s within 1e-12), the ratio of the relative L2 errors of φ (bound 1.1), and the
  threads bit for bit:

  | problem | p = 8 | p = 12 | p = 18 |
  | --- | ---: | ---: | ---: |
  | C3.2 cube (M3 Max) | 1.0085 | 1.0172 | 0.9417 |
  | C3.3 sphere (M3 Max) | 1.0036 | 1.0116 | 0.9952 |
  | C3.3 Plummer (M3 Max) | 0.9875 | 0.9908 | 0.9438 |
  | C3.3 clusters (M3 Max) | 0.9995 | 0.9873 | 0.9561 |

  Kathleen (job 605564, node-c11a-003, the AVX2 kernels by `Auto`; 230 and 157 s) gives
  every ratio to the printed digit: the output does not depend on the ISA, and both runs
  pass with the threads bit for bit.

  Below 1 at p = 18: the plane waves' error is a few per cent of the truncation's and of
  either sign, and the exact M2L's own rounding at p = 18 is no smaller.
- `tests/multi_rank.rs` (the C5.1 host gate, release) with the variant `planewave`: every
  check passes on the M3 Max at 1 and 2 ranks (every output bit for bit the one-rank
  `PlaneWave` `Fmm` over the union, the C3.2 and C3.3 gates), and C6.5's ratio of the φ
  error under `PlaneWave` to the default run's (an exact M2L), at most 1.1:

  | problem | f64 p = 3 | f64 p = 8 | f64 p = 18 | f32 p = 3 | f32 p = 8 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | cube (N = 10⁵, 8 vectors) | 1.0014 | 1.0085 | 0.9417 | 1.0014 | 1.0036 |
  | Plummer (N = 10⁵, 8 vectors) | 1.0004 | 0.9982 | 0.9415 | 1.0004 | 0.9951 |

  (the 2-rank runs; in f64 the same at 1 and 2 ranks to the printed digit, in f32 1.0038
  and 0.9952 at one rank, where the order of the points within the leaves, and with it
  f32's rounding, differs). On Kathleen every check passes at 1 rank
  (job 605564, 85 min), at 4 ranks and at 8 ranks (job 605634, node-c11a-072, 110 and
  105 min), with the same ratios to the printed digit in f64 (f32 at p = 8: 1.0039 and
  0.9954 at 4 ranks, 1.0079 and 0.9993 at 8, the order of the points within the leaves
  again). The 2-rank run on Kathleen did not finish within its 2-hour timeout, twice (job
  605564 with every rank bound to one core, which also slowed the one-rank run; job
  605636 unbound) and printed no failure; at 2 ranks the test passes on the M3 Max
  (50 min).
- The device (the CPU runtime, `--features cpu`): `cargo test -p nd-fmm-exec --features cpu --release --test mpi_exec` on
  the M3 Max (66.4 s, passes): in every scenario `check_plane_wave` builds the device `Fmm`
  on the CPU runtime under `PlaneWave`, which reports "PlaneWave, run as dense on the
  device" and equals the device `Fmm` under `Dense` bit for bit; the release run checks
  every scenario (19 f64 and 2 f32 at one rank, worst ratios 1.0172 and 1.0027, 2.7 s).
  CUDA was not run (the device path takes the dense branch, which this task does not
  change; `cargo check -p nd-fmm-exec --features cuda` passes in Kathleen's `check.sbatch`).
- The checks of the root CLAUDE.md: on the M3 Max `RUST_MIN_STACK=8388608 cargo test
  --workspace` (75 executables, 876 passed, 0 failed, 64 ignored), the strict clippy, the
  default clippy and docs, the other architecture's clippy of `nd-fmm-simd`
  (`x86_64-apple-darwin`); `.github/scripts/run-mpi-tests.sh` at 2, 4 and 8 ranks (33 of
  33 launches pass); on Kathleen `check.sbatch` (job 605563, node-c11a-003): fmt, the default clippy,
  clippy of the examples, `cargo test`, the docs, the strict clippy, nd-fmm-simd's clippy,
  tests and ignored release tests, nd-fmm-kernels' clippy, tests and CUDA type check and
  `cargo check -p nd-fmm-exec --features cuda` pass; `cargo test --workspace` has 75
  executables, 874 passed, 64 ignored and **2 failed**, the two failures of every Kathleen
  run in the Phase 3S spike `spikes/p2p-simd` (`kernel_rsqrt_within_contract` and
  `compare_core_on_a_small_cell`: an rsqrt estimate and green-kernels' bound on this CPU,
  tools/kathleen/README.md), which this task does not touch.

## 3. The quadrature per degree

`nodes_for_degree(p)` is the spike's "matched" rule on a whole level (the ignored
`level_every_degree` of `nd_fmm_tables::plane_wave`, release, 74 s on the M3 Max): on the
uniform level 3 (512 boxes, four random charges each, 53,352 V pairs, every pair translated
directly along its merge direction), the fewest λ nodes whose error (the largest over four
points of every box, relative to the box's Σ_V |q|) is at most 0.1× the M2L truncation
error there, made non-decreasing in p:

| p | s | H | modes | error (Σ\|q\|) | truncation (Σ\|q\|) | p | s | H | error | truncation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 5 | 3 | 3.65e-3 | 3.77e-2 | 11 | 21 | 399 | 1.68e-8 | 2.33e-7 |
| 1 | 7 | 34 | 13 | 1.86e-4 | 2.99e-3 | 12 | 21 | 399 | 1.61e-8 | 1.69e-7 |
| 2 | 9 | 57 | 24 | 2.79e-5 | 6.69e-4 | 13 | 23 | 494 | 3.36e-9 | 5.00e-8 |
| 3 | 10 | 73 | 34 | 1.62e-5 | 3.05e-4 | 14 | 26 | 641 | 1.94e-9 | 2.44e-8 |
| 4 | 12 | 111 | 52 | 3.49e-6 | 6.10e-5 | 15 | 27 | 699 | 8.26e-10 | 8.52e-9 |
| 5 | 13 | 133 | 66 | 1.79e-6 | 2.26e-5 | 16 | 27 | 699 | 4.18e-10 | 4.90e-9 |
| 6 | 15 | 186 | 89 | 6.42e-7 | 8.80e-6 | 17 | 32 | 1009 | 3.51e-10 | 5.69e-9 |
| 7 | 16 | 217 | 106 | 2.57e-7 | 4.49e-6 | 18 | 32 | 1009 | 1.84e-10 (s = 30) | 2.02e-9 |
| 8 | 17 | 252 | 124 | 1.20e-7 | 1.82e-6 | 19 | 36 | 1311 | 6.47e-11 | 9.18e-10 |
| 9 | 18 | 284 | 142 | 7.95e-8 | 8.52e-7 | 20 | 36 | 1311 | 2.76e-11 | 3.52e-10 |
| 10 | 20 | 358 | 176 | 2.15e-8 | 2.30e-7 | | | | | |

The spike found 10, 15, 18, 21 and 32 at p = 3, 6, 8, 12 and 18 the same way; here 17 at
p = 8 and 30 at p = 18, which the monotone table raises to 32 (from p = 17's 32; at
p ≥ 17 the waves meet their rounding floor, 2–4e-10 Σ|q|, so the matched s jumps). Even
p = 19 and 20 match, at s = 36. The FMM's errors confirm the choice (§2): at most 1.017
times the exact M2L's in every scenario of `tests/mpi_exec.rs`, 0.94–1.02 in the C3.2 and
C3.3 gates and 0.95–1.03 in φ in every sweep of §5 and §6.

## 4. The kernels

`plane_wave_parts` times every step of the level call per box on one core, synthetic
data (4,096 boxes of random multipoles, the sources of `merge` at random boxes: more
memory traffic than the FMM's spatially close sources), the median of five passes; µs
per box, every frame (the three frames' rotations, modes, waves, sums and locals, 31
merge entries and 8 gathers per box of an interior parent):

| machine, ISA | p | s | H | rotate in | modes | waves | merge | gather | sums | local | rotate out | sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 Max, NEON | 3 | 10 | 73 | 0.115 | 0.270 | 0.579 | 1.129 | 0.471 | 0.651 | 0.328 | 0.105 | 3.65 |
| | 8 | 17 | 252 | 0.597 | 1.219 | 2.112 | 4.103 | 1.463 | 1.996 | 1.928 | 0.586 | 14.00 |
| | 12 | 21 | 399 | 1.449 | 2.869 | 3.873 | 8.198 | 2.324 | 4.426 | 4.352 | 1.419 | 28.91 |
| | 18 | 32 | 1009 | 3.802 | 6.790 | 16.865 | 30.757 | 5.699 | 18.005 | 13.560 | 3.672 | 99.15 |
| Kathleen, AVX2 | 3 | 10 | 73 | 0.174 | 0.743 | 0.999 | 2.202 | 0.733 | 0.952 | 0.846 | 0.157 | 6.81 |
| | 8 | 17 | 252 | 0.877 | 3.206 | 3.913 | 11.581 | 2.218 | 4.484 | 4.998 | 0.844 | 32.12 |
| | 12 | 21 | 399 | 2.175 | 6.510 | 7.069 | 27.276 | 3.578 | 7.902 | 11.663 | 2.280 | 68.45 |
| | 18 | 32 | 1009 | 5.906 | 17.424 | 21.372 | 89.122 | 8.937 | 21.086 | 38.463 | 5.883 | 208.19 |
| Kathleen, AVX-512 | 3 | 10 | 73 | 0.175 | 0.739 | 0.985 | 1.968 | 0.457 | 1.249 | 0.847 | 0.157 | 6.58 |
| | 8 | 17 | 252 | 0.884 | 3.176 | 3.319 | 10.592 | 1.545 | 4.159 | 4.993 | 0.868 | 29.54 |
| | 12 | 21 | 399 | 2.168 | 6.480 | 5.959 | 22.886 | 2.567 | 6.414 | 11.623 | 2.147 | 60.24 |
| | 18 | 32 | 1009 | 5.835 | 17.435 | 14.600 | 71.526 | 6.860 | 22.559 | 38.556 | 5.871 | 183.24 |
| locust, NEON | 3 | 10 | 73 | 0.081 | 0.312 | 0.666 | 2.063 | 0.582 | 0.757 | 0.431 | 0.079 | 4.97 |
| | 8 | 17 | 252 | 0.504 | 1.482 | 2.884 | 7.168 | 1.800 | 2.836 | 2.518 | 0.490 | 19.68 |
| | 12 | 21 | 399 | 1.252 | 3.107 | 5.634 | 13.356 | 2.978 | 5.699 | 5.715 | 1.222 | 38.96 |
| | 18 | 32 | 1009 | 3.599 | 8.026 | 19.014 | 50.859 | 7.318 | 21.907 | 17.858 | 3.486 | 132.07 |

(M3 Max on 2026-10-19 before the sweeps, nothing else running; Kathleen job 605558,
node-c11a-002; locust after its sweeps, load 1.1, its H100 busy with another user's job.)
`merge` is the largest part everywhere (29 % on the M3 Max, 36 % on Kathleen and on locust
at p = 8) and the only one bounded by memory traffic; in the FMM, at f64 p = 8
on the M3 Max's 10⁶ cube, the shares are similar (a `sample` profile: `merge` 28 %,
`sums` 14 %, `waves` 13 %, `local` 13 %, the rotations 9 %, `gather` 8 %, `modes` 8 %), and
the plane waves' M2L is no longer the largest stage: P2P and the leaf operators take more.
On Kathleen the steps outside the vector kernels (`modes`, `local` and the rotations, plain
Rust compiled for the x86-64 baseline) cost 2.5–3× the M3 Max's per core: at p = 18
`local` is a fifth of the sum there, a candidate for a vector body after T13.

How the kernels got there (one core, f64 p = 8, M2L per V pair inside the FMM; the 10⁶
cube): the first version, the spike's loops through the kernel API, took 0.136 µs on the
M3 Max (the spike's matched level: 0.152) and 0.446 (AVX2) and 0.427 µs (AVX-512) on
Kathleen, where `Batched`'s kernels take 0.653 and 0.395: no gain against AVX-512. Then,
measured step by step with `plane_wave_parts`:
- the tables' own steps rounded products and sums apart instead of `mul_add`, which an
  x86_64 build without `fma` turns into a library call (Kathleen 0.446 → 0.226 µs with the
  next two items);
- `waves` with the four outputs of R vectors in registers and the coefficients splatted
  once per mode, the rows padded to the kernel's lanes (`WaveTables`), the parity of the
  mode named per call (a choice of accumulators at run time kept them in memory), and the
  rows sliced once per node so the loop checks no bound (M3 Max 6.8 → 2.1 µs per box);
- `sums` with both directions per row and each parity a pass of its own (3.7 → 2.0 µs);
- `merge` and `gather` sliced once per block (`gather` 2.0 → 1.5 µs);
- `modes` as sweeps over the λ nodes for each (m, n) (3.3 → 1.2 µs on the M3 Max, 6.2 →
  3.1 on Kathleen); `local` kept its direct form (the sweep forms were slower at p ≤ 12);
- AVX2's `waves` with blocks of two vectors (four spilled: 5.8 → 4.2 µs at p = 8).

So **0.094 µs on the M3 Max and 0.202 (AVX2) and 0.182 µs (AVX-512) on Kathleen** (§5), 1.5×
and 2.2–2.4× the first version, against the spike's 0.152 and 0.364 (AVX2 build) on its one
level.

## 5. One rank (t13-host)

`tools/scaling/run.sh t13-host`: one rank at one thread and at every core (the M3 Max's 12
performance cores, locust's 72, Kathleen's 40), the cube and the Plummer sphere at N = 10⁶,
and at N = 10⁷ at every core on locust and Kathleen; `PlaneWave` against `Batched`
(separate and fused) and `Rotation`, every launch with `--kinds` (`KindTiming::Synchronous`),
overlap off and the errors of one charge vector; three timed evaluations (two at p = 12,
one at p = 18), the median. On Kathleen `PlaneWave` and `Batched` ran with each of AVX2 and
AVX-512 (`--p2p`, which sets the P2P and the M2L kernels' ISA; `Rotation`'s M2L does not
depend on it), and the table shows each strategy's faster. *Ratio* is the best other
strategy's time over `PlaneWave`'s (above 1: `PlaneWave` faster); the M2L time per V pair
is that of the slowest rank; the last column the φ error (relative L2 against the direct
sum) under `PlaneWave` over the exact M2L's (`Batched`). Labels: PW `PlaneWave`, B
`Batched`, F `Batched` fused, R `Rotation`. f32 is in §9.

**M3 Max** (12 threads; bench-results/t13/m3max):

| dist | N | prec | p | threads | PW: evaluation ms | best other ms | **ratio** | PW: M2L µs per V pair | best other | **ratio** | φ error, PW / exact M2L |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 10^6 | f64 | 3 | 1 | 995.2 (PW) | 938.3 (F) | **0.94** | 0.0273 | 0.0210 (F) | **0.77** | 1.0024 |
| cube | 10^6 | f64 | 3 | 12 | 105.9 (PW) | 100.0 (F) | **0.94** | 0.0030 | 0.0022 (F) | **0.73** | 1.0024 |
| cube | 10^6 | f64 | 6 | 1 | 1428.8 (PW) | 1713.3 (F) | **1.20** | 0.0667 | 0.1148 (F) | **1.72** | 1.0032 |
| cube | 10^6 | f64 | 6 | 12 | 148.0 (PW) | 172.5 (F) | **1.17** | 0.0071 | 0.0112 (F) | **1.58** | 1.0032 |
| cube | 10^6 | f64 | 8 | 1 | 1788.8 (PW) | 3100.9 (F) | **1.73** | 0.0936 | 0.3009 (F) | **3.21** | 1.0071 |
| cube | 10^6 | f64 | 8 | 12 | 184.1 (PW) | 331.1 (F) | **1.80** | 0.0098 | 0.0317 (F) | **3.23** | 1.0071 |
| cube | 10^6 | f64 | 12 | 1 | 2982.9 (PW) | 10081.9 (F) | **3.38** | 0.1860 | 1.3302 (F) | **7.15** | 0.9815 |
| cube | 10^6 | f64 | 12 | 12 | 305.7 (PW) | 1061.6 (F) | **3.47** | 0.0204 | 0.1412 (F) | **6.92** | 0.9815 |
| cube | 10^6 | f64 | 18 | 1 | 7209.1 (PW) | 43725.5 (R) | **6.07** | 0.6088 | 6.6634 (R) | **10.95** | 0.9846 |
| cube | 10^6 | f64 | 18 | 12 | 681.4 (PW) | 4350.0 (R) | **6.38** | 0.0588 | 0.6614 (R) | **11.25** | 0.9846 |
| plummer | 10^6 | f64 | 3 | 1 | 2137.4 (PW) | 2073.1 (F) | **0.97** | 0.0265 | 0.0206 (F) | **0.78** | 0.9958 |
| plummer | 10^6 | f64 | 3 | 12 | 222.4 (PW) | 212.4 (F) | **0.95** | 0.0032 | 0.0022 (F) | **0.69** | 0.9958 |
| plummer | 10^6 | f64 | 6 | 1 | 4723.7 (PW) | 5175.9 (F) | **1.10** | 0.0673 | 0.1164 (F) | **1.73** | 1.0000 |
| plummer | 10^6 | f64 | 6 | 12 | 475.5 (PW) | 514.2 (F) | **1.08** | 0.0074 | 0.0110 (F) | **1.49** | 1.0000 |
| plummer | 10^6 | f64 | 8 | 1 | 7108.4 (PW) | 9100.9 (F) | **1.28** | 0.0953 | 0.3043 (F) | **3.19** | 1.0033 |
| plummer | 10^6 | f64 | 8 | 12 | 716.4 (PW) | 932.6 (F) | **1.30** | 0.0102 | 0.0310 (F) | **3.04** | 1.0033 |
| plummer | 10^6 | f64 | 12 | 1 | 13879.5 (PW) | 25314.6 (F) | **1.82** | 0.1916 | 1.3573 (F) | **7.08** | 1.0315 |
| plummer | 10^6 | f64 | 12 | 12 | 1340.9 (PW) | 2941.0 (F) | **2.19** | 0.0196 | 0.1610 (F) | **8.21** | 1.0315 |
| plummer | 10^6 | f64 | 18 | 1 | 32260.3 (PW) | 90518.2 (R) | **2.81** | 0.5860 | 6.7031 (R) | **11.44** | 0.9792 |
| plummer | 10^6 | f64 | 18 | 12 | 3106.8 (PW) | 8738.1 (R) | **2.81** | 0.0607 | 0.6359 (R) | **10.48** | 0.9792 |

**locust** (GH200's Grace, 72 threads; its H100 busy with another user's job, the CPU
quiet):

| dist | N | prec | p | threads | PW: evaluation ms | best other ms | **ratio** | PW: M2L µs per V pair | best other | **ratio** | φ error, PW / exact M2L |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 10^6 | f64 | 3 | 1 | 2375.1 (PW) | 2355.9 (F) | **0.99** | 0.0385 | 0.0377 (F) | **0.98** | 1.0024 |
| cube | 10^6 | f64 | 3 | 72 | 79.1 (PW) | 74.3 (F) | **0.94** | 0.0011 | 0.0006 (F) | **0.55** | 1.0024 |
| cube | 10^6 | f64 | 6 | 1 | 3027.7 (PW) | 3434.1 (F) | **1.13** | 0.1052 | 0.1742 (F) | **1.66** | 1.0032 |
| cube | 10^6 | f64 | 6 | 72 | 91.7 (PW) | 95.0 (F) | **1.04** | 0.0026 | 0.0031 (F) | **1.19** | 1.0032 |
| cube | 10^6 | f64 | 8 | 1 | 3579.8 (PW) | 5301.1 (F) | **1.48** | 0.1532 | 0.4361 (F) | **2.85** | 1.0071 |
| cube | 10^6 | f64 | 8 | 72 | 102.8 (PW) | 132.4 (F) | **1.29** | 0.0038 | 0.0079 (F) | **2.08** | 1.0071 |
| cube | 10^6 | f64 | 12 | 1 | 5201.9 (PW) | 16024.9 (F) | **3.08** | 0.2914 | 2.0513 (F) | **7.04** | 0.9815 |
| cube | 10^6 | f64 | 12 | 72 | 130.4 (PW) | 323.4 (F) | **2.48** | 0.0067 | 0.0365 (F) | **5.45** | 0.9815 |
| cube | 10^6 | f64 | 18 | 1 | 10367.8 (PW) | 61101.2 (R) | **5.89** | 0.8057 | 9.2029 (R) | **11.42** | 0.9846 |
| cube | 10^6 | f64 | 18 | 72 | 249.1 (PW) | 968.4 (R) | **3.89** | 0.0213 | 0.1392 (R) | **6.54** | 0.9846 |
| cube | 10^7 | f64 | 3 | 72 | 913.8 (PW) | 904.8 (F) | **0.99** | 0.0010 | 0.0007 (F) | **0.70** | 1.0068 |
| cube | 10^7 | f64 | 6 | 72 | 1037.7 (PW) | 1089.4 (F) | **1.05** | 0.0026 | 0.0028 (F) | **1.08** | 0.9991 |
| cube | 10^7 | f64 | 8 | 72 | 1121.8 (PW) | 1344.0 (F) | **1.20** | 0.0035 | 0.0068 (F) | **1.94** | 1.0082 |
| cube | 10^7 | f64 | 12 | 72 | 1504.5 (PW) | 2712.5 (F) | **1.80** | 0.0063 | 0.0302 (F) | **4.79** | 0.9973 |
| cube | 10^7 | f64 | 18 | 72 | 2440.9 (PW) | 8751.8 (R) | **3.59** | 0.0208 | 0.1409 (R) | **6.77** | 0.9454 |
| plummer | 10^6 | f64 | 3 | 1 | 3796.1 (PW) | 3845.4 (F) | **1.01** | 0.0359 | 0.0356 (F) | **0.99** | 0.9958 |
| plummer | 10^6 | f64 | 3 | 72 | 113.3 (PW) | 112.6 (F) | **0.99** | 0.0012 | 0.0007 (F) | **0.58** | 0.9958 |
| plummer | 10^6 | f64 | 6 | 1 | 7177.5 (PW) | 7882.9 (F) | **1.10** | 0.1005 | 0.1752 (F) | **1.74** | 1.0000 |
| plummer | 10^6 | f64 | 6 | 72 | 169.6 (PW) | 183.9 (F) | **1.08** | 0.0025 | 0.0033 (F) | **1.32** | 1.0000 |
| plummer | 10^6 | f64 | 8 | 1 | 10150.5 (PW) | 12911.6 (F) | **1.27** | 0.1496 | 0.4386 (F) | **2.93** | 1.0033 |
| plummer | 10^6 | f64 | 8 | 72 | 225.3 (PW) | 277.3 (F) | **1.23** | 0.0035 | 0.0086 (F) | **2.46** | 1.0033 |
| plummer | 10^6 | f64 | 12 | 1 | 18642.6 (PW) | 35916.7 (F) | **1.93** | 0.2895 | 2.0743 (F) | **7.17** | 1.0315 |
| plummer | 10^6 | f64 | 12 | 72 | 374.5 (PW) | 693.6 (F) | **1.85** | 0.0066 | 0.0389 (F) | **5.89** | 1.0315 |
| plummer | 10^6 | f64 | 18 | 1 | 41787.0 (PW) | 121957.9 (R) | **2.92** | 0.8321 | 9.2607 (R) | **11.13** | 0.9792 |
| plummer | 10^6 | f64 | 18 | 72 | 813.7 (PW) | 1959.6 (R) | **2.41** | 0.0208 | 0.1403 (R) | **6.75** | 0.9792 |
| plummer | 10^7 | f64 | 3 | 72 | 1075.3 (PW) | 1040.7 (F) | **0.97** | 0.0010 | 0.0007 (F) | **0.70** | 1.0039 |
| plummer | 10^7 | f64 | 6 | 72 | 1552.0 (PW) | 1601.3 (F) | **1.03** | 0.0025 | 0.0028 (F) | **1.12** | 1.0027 |
| plummer | 10^7 | f64 | 8 | 72 | 2051.7 (PW) | 2342.5 (F) | **1.14** | 0.0044 | 0.0068 (F) | **1.55** | 0.9993 |
| plummer | 10^7 | f64 | 12 | 72 | 3131.0 (PW) | 5718.6 (F) | **1.83** | 0.0064 | 0.0301 (F) | **4.70** | 0.9755 |
| plummer | 10^7 | f64 | 18 | 72 | 6847.4 (PW) | 19793.0 (R) | **2.89** | 0.0200 | 0.1420 (R) | **7.10** | 0.9485 |

**Kathleen** (one node, 2 × 20 Cascade Lake cores, 40 threads; jobs 605559, node-c11a-003,
the cube, and 605560, node-c11a-002, the Plummer sphere):

| dist | N | prec | p | threads | PW: evaluation ms | best other ms | **ratio** | PW: M2L µs per V pair | best other | **ratio** | φ error, PW / exact M2L |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 10^6 | f64 | 3 | 1 | 1306.6 (PW avx512) | 1211.1 (F avx512) | **0.93** | 0.0501 | 0.0367 (F avx512) | **0.73** | 1.0024 |
| cube | 10^6 | f64 | 3 | 40 | 122.1 (PW avx512) | 101.8 (F avx512) | **0.83** | 0.0037 | 0.0019 (F avx2) | **0.51** | 1.0024 |
| cube | 10^6 | f64 | 6 | 1 | 2274.7 (PW avx512) | 2251.6 (F avx512) | **0.99** | 0.1235 | 0.1226 (F avx512) | **0.99** | 1.0032 |
| cube | 10^6 | f64 | 6 | 40 | 159.1 (PW avx512) | 154.0 (F avx512) | **0.97** | 0.0077 | 0.0065 (F avx512) | **0.84** | 1.0032 |
| cube | 10^6 | f64 | 8 | 1 | 3196.6 (PW avx512) | 4007.8 (F avx512) | **1.25** | 0.1822 | 0.3142 (F avx512) | **1.72** | 1.0071 |
| cube | 10^6 | f64 | 8 | 40 | 202.6 (PW avx512) | 230.9 (F avx512) | **1.14** | 0.0103 | 0.0157 (F avx512) | **1.52** | 1.0071 |
| cube | 10^6 | f64 | 12 | 1 | 5967.1 (PW avx512) | 13856.2 (F avx512) | **2.32** | 0.3736 | 1.5919 (F avx512) | **4.26** | 0.9815 |
| cube | 10^6 | f64 | 12 | 40 | 311.1 (PW avx512) | 678.9 (F avx512) | **2.18** | 0.0196 | 0.0737 (F avx512) | **3.76** | 0.9815 |
| cube | 10^6 | f64 | 18 | 1 | 14775.1 (PW avx512) | 82014.8 (F avx512) | **5.55** | 1.1251 | 11.7610 (F avx512) | **10.45** | 0.9846 |
| cube | 10^6 | f64 | 18 | 40 | 692.1 (PW avx512) | 3034.0 (F avx512) | **4.38** | 0.0565 | 0.4332 (F avx512) | **7.67** | 0.9846 |
| cube | 10^7 | f64 | 3 | 40 | 1284.8 (PW avx512) | 1268.0 (B avx512) | **0.99** | 0.0026 | 0.0020 (B avx512) | **0.77** | 1.0068 |
| cube | 10^7 | f64 | 6 | 40 | 1659.0 (PW avx512) | 1592.1 (F avx512) | **0.96** | 0.0061 | 0.0054 (F avx512) | **0.89** | 0.9991 |
| cube | 10^7 | f64 | 8 | 40 | 1938.6 (PW avx512) | 2325.8 (B avx512) | **1.20** | 0.0086 | 0.0132 (F avx512) | **1.53** | 1.0082 |
| cube | 10^7 | f64 | 12 | 40 | 2858.6 (PW avx512) | 5011.0 (F avx512) | **1.75** | 0.0159 | 0.0549 (F avx512) | **3.45** | 0.9973 |
| cube | 10^7 | f64 | 18 | 40 | 5755.1 (PW avx512) | 23182.1 (F avx512) | **4.03** | 0.0460 | 0.3666 (F avx512) | **7.97** | 0.9454 |
| plummer | 10^6 | f64 | 3 | 1 | 4675.9 (PW avx512) | 4521.2 (B avx512) | **0.97** | 0.0494 | 0.0350 (F avx512) | **0.71** | 0.9958 |
| plummer | 10^6 | f64 | 3 | 40 | 306.9 (PW avx512) | 279.9 (F avx512) | **0.91** | 0.0040 | 0.0021 (F avx2) | **0.52** | 0.9958 |
| plummer | 10^6 | f64 | 6 | 1 | 11087.5 (PW avx512) | 11014.6 (F avx512) | **0.99** | 0.1274 | 0.1239 (F avx512) | **0.97** | 1.0000 |
| plummer | 10^6 | f64 | 6 | 40 | 563.6 (PW avx512) | 541.9 (B avx512) | **0.96** | 0.0081 | 0.0077 (F avx2) | **0.95** | 1.0000 |
| plummer | 10^6 | f64 | 8 | 1 | 17281.4 (PW avx512) | 18698.9 (F avx512) | **1.08** | 0.1943 | 0.3389 (F avx512) | **1.74** | 1.0033 |
| plummer | 10^6 | f64 | 8 | 40 | 818.0 (PW avx512) | 907.3 (B avx512) | **1.11** | 0.0112 | 0.0177 (F avx512) | **1.58** | 1.0033 |
| plummer | 10^6 | f64 | 12 | 1 | 34822.2 (PW avx512) | 47075.2 (F avx512) | **1.35** | 0.3856 | 1.6250 (F avx512) | **4.21** | 1.0315 |
| plummer | 10^6 | f64 | 12 | 40 | 1492.7 (PW avx512) | 2113.9 (F avx512) | **1.42** | 0.0195 | 0.0738 (F avx512) | **3.78** | 1.0315 |
| plummer | 10^6 | f64 | 18 | 1 | 80175.1 (PW avx512) | 186887.9 (F avx512) | **2.33** | 1.1400 | 11.8491 (F avx512) | **10.39** | 0.9792 |
| plummer | 10^6 | f64 | 18 | 40 | 3473.8 (PW avx512) | 7079.0 (F avx512) | **2.04** | 0.0565 | 0.4265 (F avx512) | **7.55** | 0.9792 |
| plummer | 10^7 | f64 | 3 | 40 | 2341.9 (PW avx512) | 2173.3 (B avx512) | **0.93** | 0.0028 | 0.0019 (B avx512) | **0.68** | 1.0039 |
| plummer | 10^7 | f64 | 6 | 40 | 4285.2 (PW avx512) | 4194.2 (F avx512) | **0.98** | 0.0060 | 0.0054 (F avx512) | **0.90** | 1.0027 |
| plummer | 10^7 | f64 | 8 | 40 | 6088.4 (PW avx512) | 6778.4 (F avx512) | **1.11** | 0.0086 | 0.0135 (F avx512) | **1.57** | 0.9993 |
| plummer | 10^7 | f64 | 12 | 40 | 11587.6 (PW avx512) | 16624.4 (F avx512) | **1.43** | 0.0156 | 0.0615 (F avx512) | **3.94** | 0.9755 |
| plummer | 10^7 | f64 | 18 | 40 | 26128.2 (PW avx512) | 61716.6 (F avx512) | **2.36** | 0.0455 | 0.3687 (F avx512) | **8.10** | 0.9485 |

In f64 the evaluation under `PlaneWave` is, against the fastest other strategy:

| p | M3 Max | locust | Kathleen |
| ---: | --- | --- | --- |
| 3 | 0.94–0.97 | 0.94–1.01 | 0.83–0.99 |
| 6 | **1.08–1.20** | **1.03–1.13** | 0.96–0.99 |
| 8 | **1.28–1.80** | **1.14–1.48** | **1.08–1.25** |
| 12 | **1.82–3.47** | **1.80–3.08** | **1.35–2.32** |
| 18 | **2.81–6.38** | **2.41–5.89** | **2.04–5.55** |

and its M2L stage per V pair 0.51–0.99 (p = 3), 0.84–1.74 (p = 6), 1.52–3.23 (p = 8),
3.45–8.21 (p = 12) and 6.54–11.44 (p = 18) times faster. The evaluation gains less than
the stage, and less on the Plummer sphere than on the cube, as the leaves (P2P above all)
take more of it: at f64 p = 8 and every core the leaves are 47–58 % of a Plummer
evaluation under `PlaneWave` and the downward pass 32–39 % (on locust's cube 32 % and
24 %). At p = 18 the fastest other is `Rotation` on the arm machines and fused
`Batched` (AVX-512) on Kathleen. Against T10's spike, which measured the M2L stage of one
level (2.8–3.3× at p = 8 on the M3 Max and locust, 1.14–1.44× against AVX-512 on
Kathleen), the strategy's M2L keeps that gain inside the FMM: 3.0–3.2× and 1.6–2.9× at
p = 8 on the M3 Max and locust and 1.5–1.7× on Kathleen. Kathleen gains least, at every p: its
`Batched` kernel runs on AVX-512, while the plane waves' steps outside the vector kernels
are plain x86-64 code (§4).

## 6. Many ranks per node (t13-node)

`tools/scaling/run.sh t13-node`: T2's splits of each node's cores into ranks × threads
(`T2_SPLITS`), the N = 10⁶ cube and Plummer sphere at f64 p = 8 and the cube at f64 p = 12,
the same variants and options as §5, three timed evaluations, the median of the slowest
rank; the `downward` column is the pass with M2L (`PlaneWave` / the fastest other). The
work cut uses `PlaneWave`'s own constants (§8).

**M3 Max:**

| dist | p | ranks × threads | PW neon ms | Batched ms | Fused ms | Rotation ms | best other / PW | downward ms PW / best other | M2L µs/pair PW |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| cube | 8 | 1 × 1 | 1796.9 | 4321.1 | 3054.8 | 6357.2 | **1.70** (F) | 591.6 / 1846.9 | 0.0931 |
| cube | 8 | 2 × 1 | 932.1 | 2226.7 | 1610.3 | 3235.9 | **1.73** (F) | 314.6 / 982.5 | 0.0992 |
| cube | 8 | 4 × 1 | 493.7 | 1160.0 | 851.1 | 1681.3 | **1.72** (F) | 173.3 / 525.5 | 0.1096 |
| cube | 8 | 8 × 1 | 275.5 | 611.1 | 476.8 | 852.7 | **1.73** (F) | 99.7 / 293.9 | 0.1269 |
| cube | 8 | 1 × 12 | 200.3 | 461.9 | 303.0 | 580.2 | **1.51** (F) | 65.3 / 179.6 | 0.0102 |
| cube | 8 | 12 × 1 | 213.3 | 444.3 | 354.9 | 686.2 | **1.66** (F) | 81.3 / 214.4 | 0.1535 |
| cube | 12 | 1 × 1 | 2968.2 | 15428.6 | 10096.2 | 15409.0 | **3.40** (F) | 1198.0 / 8232.7 | 0.1860 |
| cube | 12 | 2 × 1 | 1550.6 | 7996.0 | 5249.2 | 7814.4 | **3.39** (F) | 641.6 / 4291.4 | 0.1999 |
| cube | 12 | 4 × 1 | 830.3 | 4100.8 | 2747.0 | 4006.0 | **3.31** (F) | 358.9 / 2253.0 | 0.2248 |
| cube | 12 | 8 × 1 | 467.8 | 2227.3 | 1656.6 | 2248.7 | **3.54** (F) | 205.6 / 1366.6 | 0.2591 |
| cube | 12 | 1 × 12 | 306.7 | 1436.7 | 1021.4 | 1437.8 | **3.33** (F) | 124.5 / 836.7 | 0.0193 |
| cube | 12 | 12 × 1 | 429.6 | 1695.6 | 1206.0 | 1662.2 | **2.81** (F) | 202.9 / 987.9 | 0.3795 |
| plummer | 8 | 1 × 1 | 7185.5 | 11099.9 | 9107.3 | 14287.2 | **1.27** (F) | 2384.1 / 4323.9 | 0.0958 |
| plummer | 8 | 2 × 1 | 3731.3 | 5725.8 | 4842.6 | 7591.7 | **1.30** (F) | 1241.4 / 2334.5 | 0.1060 |
| plummer | 8 | 4 × 1 | 1935.3 | 3032.5 | 2497.6 | 3831.5 | **1.29** (F) | 646.9 / 1240.9 | 0.1149 |
| plummer | 8 | 8 × 1 | 1054.2 | 1666.3 | 1446.1 | 2174.5 | **1.37** (F) | 369.4 / 690.8 | 0.1290 |
| plummer | 8 | 1 × 12 | 744.6 | 1084.7 | 916.1 | 1377.1 | **1.23** (F) | 262.1 / 447.5 | 0.0105 |
| plummer | 8 | 12 × 1 | 858.6 | 1243.1 | 1094.5 | 1658.3 | **1.27** (F) | 320.6 / 525.2 | 0.1714 |

**locust:**

| dist | p | ranks × threads | PW neon ms | Batched ms | Fused ms | Rotation ms | best other / PW | downward ms PW / best other | M2L µs/pair PW |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| cube | 8 | 1 × 1 | 3537.1 | 6556.6 | 5260.1 | 9315.6 | **1.49** (F) | 965.7 / 2636.6 | 0.1537 |
| cube | 8 | 8 × 1 | 472.8 | 877.9 | 703.4 | 1158.7 | **1.49** (F) | 141.7 / 369.2 | 0.1804 |
| cube | 8 | 16 × 1 | 248.6 | 454.0 | 372.2 | 577.5 | **1.50** (F) | 84.2 / 200.8 | 0.2055 |
| cube | 8 | 32 × 1 | 141.0 | 240.6 | 200.8 | 298.8 | **1.42** (F) | 55.9 / 110.6 | 0.2608 |
| cube | 8 | 64 × 1 | 93.4 | 135.0 | 116.4 | 158.5 | **1.25** (F) | 47.8 / 68.2 | 0.4190 |
| cube | 8 | 1 × 72 | 98.2 | 151.8 | 130.8 | 174.1 | **1.33** (F) | 24.7 / 50.1 | 0.0038 |
| cube | 8 | 8 × 9 | 71.5 | 116.6 | 96.0 | 143.3 | **1.34** (F) | 26.9 / 49.6 | 0.0343 |
| cube | 8 | 72 × 1 | 95.2 | 121.1 | 106.9 | 138.6 | **1.12** (F) | 52.9 / 63.1 | 0.5358 |
| cube | 12 | 1 × 1 | 5190.4 | 22118.1 | 16045.0 | 22209.4 | **3.09** (F) | 1844.8 / 12590.9 | 0.2888 |
| cube | 12 | 8 × 1 | 693.6 | 3387.9 | 2625.3 | 2789.3 | **3.79** (F) | 270.0 / 2190.6 | 0.3406 |
| cube | 12 | 16 × 1 | 371.2 | 1790.6 | 1398.9 | 1393.6 | **3.75** (R) | 160.9 / 1185.1 | 0.3883 |
| cube | 12 | 32 × 1 | 212.0 | 955.1 | 764.4 | 743.5 | **3.51** (R) | 103.4 / 637.9 | 0.4774 |
| cube | 12 | 64 × 1 | 145.0 | 538.7 | 449.5 | 447.7 | **3.09** (R) | 85.4 / 384.9 | 0.7510 |
| cube | 12 | 1 × 72 | 131.1 | 418.3 | 320.6 | 374.0 | **2.44** (F) | 43.2 / 223.7 | 0.0067 |
| cube | 12 | 8 × 9 | 104.7 | 411.8 | 305.0 | 341.8 | **2.91** (F) | 47.2 / 247.0 | 0.0598 |
| cube | 12 | 72 × 1 | 148.8 | 493.5 | 422.1 | 402.4 | **2.70** (R) | 92.6 / 347.7 | 0.9508 |
| plummer | 8 | 1 × 1 | 10151.5 | 15086.1 | 12923.8 | 19073.2 | **1.27** (F) | 3141.4 / 5893.6 | 0.1502 |
| plummer | 8 | 8 × 1 | 1330.5 | 2004.0 | 1720.3 | 2473.0 | **1.29** (F) | 454.0 / 827.7 | 0.1767 |
| plummer | 8 | 16 × 1 | 697.8 | 1006.7 | 877.6 | 1220.8 | **1.26** (F) | 262.7 / 453.8 | 0.2042 |
| plummer | 8 | 32 × 1 | 394.3 | 536.0 | 471.1 | 619.4 | **1.19** (F) | 166.0 / 246.8 | 0.2514 |
| plummer | 8 | 64 × 1 | 248.2 | 291.6 | 261.5 | 323.1 | **1.05** (F) | 132.8 / 149.5 | 0.3658 |
| plummer | 8 | 1 × 72 | 220.7 | 315.7 | 275.4 | 345.0 | **1.25** (F) | 71.1 / 118.6 | 0.0035 |
| plummer | 8 | 8 × 9 | 203.4 | 284.7 | 249.6 | 318.0 | **1.23** (F) | 87.2 / 132.2 | 0.0263 |
| plummer | 8 | 72 × 1 | 228.3 | 263.6 | 240.9 | 290.9 | **1.06** (F) | 127.9 / 142.1 | 0.3829 |

**Kathleen** (job 605562, node-c11a-003; `PlaneWave` and `Batched` with AVX2 and AVX-512,
`Rotation` with AVX2's P2P):

| dist | p | ranks × threads | PW avx2 ms | PW avx512 ms | Batched ms | Fused ms | Rotation ms | best other / PW | downward ms PW / best other | M2L µs/pair PW |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| cube | 8 | 1 × 1 | 4351.6 | 3229.0 | 4511.5 | 3991.5 | 16487.8 | **1.24** (F) | 1211.9 / 1969.6 | 0.1882 |
| cube | 8 | 10 × 1 | 531.6 | 406.9 | 556.4 | 505.8 | 1820.6 | **1.24** (F) | 177.5 / 273.3 | 0.2745 |
| cube | 8 | 20 × 1 | 290.4 | 237.9 | 318.8 | 286.8 | 953.7 | **1.21** (F) | 115.3 / 158.9 | 0.3582 |
| cube | 8 | 1 × 40 | 227.8 | 199.7 | 254.3 | 225.0 | 657.0 | **1.13** (F) | 69.2 / 97.9 | 0.0107 |
| cube | 8 | 2 × 20 | 181.6 | 158.2 | 211.8 | 192.9 | 582.9 | **1.22** (F) | 51.7 / 87.6 | 0.0160 |
| cube | 8 | 40 × 1 | 193.2 | 162.8 | 214.7 | 195.8 | 602.0 | **1.20** (F) | 87.7 / 111.3 | 0.5587 |
| cube | 12 | 1 × 1 | 7138.0 | 5972.2 | 16899.2 | 13846.6 | 43492.0 | **2.32** (F) | 2459.1 / 10076.0 | 0.3732 |
| cube | 12 | 10 × 1 | 879.3 | 743.6 | 1912.6 | 1582.0 | 4771.5 | **2.13** (F) | 349.0 / 1156.4 | 0.5295 |
| cube | 12 | 20 × 1 | 481.3 | 434.5 | 1083.6 | 897.3 | 2515.9 | **2.06** (F) | 225.8 / 662.6 | 0.6909 |
| cube | 12 | 1 × 40 | 339.4 | 316.7 | 686.7 | 617.2 | 1526.9 | **1.95** (F) | 128.9 / 425.6 | 0.0198 |
| cube | 12 | 2 × 20 | 284.1 | 266.0 | 616.3 | 526.3 | 1511.0 | **1.98** (F) | 103.3 / 360.9 | 0.0315 |
| cube | 12 | 40 × 1 | 343.3 | 300.7 | 742.1 | 630.7 | 1644.4 | **2.10** (F) | 164.8 / 466.7 | 0.9313 |
| plummer | 8 | 1 × 1 | 18343.6 | 17268.1 | 19377.9 | 18495.5 | 37444.8 | **1.07** (F) | 6046.7 / 7277.2 | 0.1956 |
| plummer | 8 | 10 × 1 | 2118.6 | 1992.9 | 2285.0 | 2206.3 | 4337.9 | **1.11** (F) | 776.3 / 897.6 | 0.2584 |
| plummer | 8 | 20 × 1 | 1110.9 | 1064.1 | 1263.6 | 1215.9 | 2175.4 | **1.14** (F) | 440.9 / 490.1 | 0.3316 |
| plummer | 8 | 1 × 40 | 888.0 | 792.0 | 853.6 | 818.7 | 1408.1 | **1.03** (F) | 318.6 / 360.7 | 0.0109 |
| plummer | 8 | 2 × 20 | 786.7 | 764.4 | 849.5 | 822.8 | 1412.4 | **1.08** (F) | 322.3 / 374.6 | 0.0171 |
| plummer | 8 | 40 × 1 | 737.3 | 715.4 | 862.6 | 894.9 | 1320.5 | **1.21** (B) | 326.5 / 368.6 | 0.4152 |

At the node's ranks × threads the gain holds at every split: f64 p = 8 on the cube 1.51–1.73
(M3 Max), 1.12–1.50 (locust) and 1.13–1.24 (Kathleen), on the Plummer sphere 1.23–1.37,
1.05–1.29 and 1.03–1.21, and the cube at p = 12 2.81–3.54, 2.44–3.79 and 1.95–2.32. The
gain is smallest at one rank per core (locust 72 × 1: 1.12 and 1.06 at p = 8): there the
waves of the ghost boxes, which every rank computes for its halo, are a larger share
(on the cube at p = 8 the M2L time per V pair grows from 1 × 1 to one rank per core by
3.5× under `PlaneWave`, 0.15 → 0.54 µs, and 1.7× under fused `Batched`, 0.43 → 0.75 µs,
on locust; by 2.6× and 2.1× on Kathleen). Each strategy at its own fastest split (1 × 12
on the M3 Max; 8 × 9 on locust but for fused `Batched` on the Plummer sphere, 72 × 1; 2 × 20
on Kathleen's cube, 40 × 1 and 1 × 40 on its Plummer sphere): `PlaneWave` is 1.51, 1.34 and
1.22× faster on the cube at p = 8 (M3 Max, locust, Kathleen), 1.23, 1.18 and 1.14× on the
Plummer sphere, and 3.33, 2.91 and 1.98× on the cube at p = 12.

## 7. Memory

**The formula** (`LaplaceOperator::plane_wave_bytes`; the `operator` module docs, "The
plane-wave M2L"). The level call computes one frame (two opposite directions) at a time,
so its buffer holds **4H values per box** of the largest level a rank holds (own and
ghost boxes, one charge vector at a time): 8.1 KB at f64 p = 8, 12.8 KB at p = 12 and
32.3 KB at p = 18, a third of the six directions' 6 × 16H bytes the brief estimated
(against 0.6–2.9 KB of multipole). `Fmm` sizes it
from the plan (`with_plane_wave_boxes`), so no level call allocates. Each thread's scratch
(`PlaneScratch`) is

  (2a + 32) H + 2 (p + 1)² + 8F + 4 (p + 1) s values

(a ≤ 10 accumulators of a direction, 16 incoming waves of 2H values for a parent's eight
children, the rotated multipole and the local, the modes and sums of a frame, F modes,
and the Fourier passes' work): 119 KB at f64 p = 8 and 191 KB at p = 12, measured as the
difference of the memory model between 1 × 12 and 1 × 1 on the M3 Max (1.3 and 2.1 MB for
11 threads). The tables are 0.7 MB (f64 p = 8) to 3.0 MB (p = 18) per rank, not cached.

**Measured** (the memory model's plane-wave part, the largest rank's, and the resident
size, `VmHWM` on Kathleen; f64; `t13-node` at N = 10⁶):

| problem | split | plane waves, MB | resident `PlaneWave` / `Batched` / `Rotation`, MB |
| --- | --- | ---: | --- |
| cube p = 8 | 1 × 1 | 264.4 | 810 / 561 / 539 |
| | 1 × 40 | 269.2 | 837 / 580 / 569 |
| | 2 × 20 | 151.1 | 462 / 327 / 311 |
| | 40 × 1 | 21.3 | 109 / 105 / 88 |
| cube p = 12 | 1 × 1 | 418.6 | 1017 / 672 / 600 |
| | 2 × 20 | 239.2 | 592 / 421 / 346 |
| | 40 × 1 | 33.7 | 132 / 171 / 96 |
| Plummer p = 8 | 1 × 1 | 186.4 | 843 / 667 / 650 |
| | 2 × 20 | 115.2 | 490 / 402 / 384 |
| | 40 × 1 | 34.5 | 136 / 117 / 105 |

(`Batched` and `Rotation` with AVX2; the M3 Max and locust give the same plane-wave
column, e.g. 264.4, 47.2 and 45.7 MB on the cube at p = 8 at 1 × 1, 8 × 1 and 12 × 1 on
the M3 Max, 14.8–16.2 MB at 64 and 72 ranks on locust.) At one rank the waves are most
of what `PlaneWave` adds (+249 MB resident over `Batched` at p = 8 for 264 MB of waves);
at one rank per core they are 21–35 MB per rank, 4–19 MB more resident than `Batched` at
p = 8 and 39 MB less at p = 12, where `Batched`'s dense tables (each rank's own) outweigh
them. At 40 ranks the waves of a rank's ghosts take 0.5–1× as much as its own boxes' (the
halo; §8).
At the calibration's N = 10⁷ on 40 ranks (§8) the plane waves take 104–181 MB per rank at
f64 p = 8 and 417–726 MB at p = 18 (the cube to the clusters), resident 310–452 MB and
761–1,221 MB: at most 49 GB of a Kathleen node's 192 GB at p = 18. A window of parents in
Morton order (the brief's other bound) would cut the buffer further, at the price of
waves computed twice at the windows' edges; at these sizes it is not needed.

## 8. The cost model

`PlaneWave`'s work is per box, not per V pair: a box's waves, its share of its parent's
merge entries and the conversion of its incoming waves cost about the same for an
interior box with 189 V pairs as for a box at the domain's boundary with half as many.
So under `PlaneWave` the model counts M2L as the boxes with a V row
(`cost::M2lUnit::Boxes`, `work_counts_in`; `CostModel::calibrated` sets it), and its
α_M2L is in ns per box. The constants were fitted as Phase 5S T7 did
(`tools/scaling/run.sh t13-calibrate`, Kathleen job 605559, node-c11a-003: 40 ranks × 1
thread, N = 10⁷, the first cut, overlap off, `--kinds`; each run's fit is Σ_r time over
Σ_r count per kind), and the table takes the median of the three workloads per kind:

| run | α_M2L, ns per box (the run's fit) | M2L µs per V pair | model error, the run's fit (max over ranks) | M2L time max / mean | V pairs max / mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| cube, f64 p = 8 | 64,175 | 0.374 | 4.97 % | 1.085 | 1.081 |
| Plummer, f64 p = 8 | 60,190 | 0.347 | 3.05 % | 1.239 | 1.270 |
| clusters, f64 p = 8 | 57,162 | 0.328 | 7.07 % | 1.102 | 1.155 |
| cube, f64 p = 18 | 387,683 | 2.247 | 5.85 % | 1.068 | 1.081 |
| Plummer, f64 p = 18 | 346,076 | 1.998 | 4.31 % | 1.242 | 1.270 |
| clusters, f64 p = 18 | 331,365 | 1.929 | 8.07 % | 1.116 | 1.155 |
| cube, f32 p = 8 | 42,538 | 0.254 | 4.48 % | 1.108 | 1.081 |
| Plummer, f32 p = 8 | 41,331 | 0.237 | 2.00 % | 1.233 | 1.270 |
| clusters, f32 p = 8 | 39,335 | 0.233 | 5.85 % | 1.080 | 1.155 |

The calibration (`CALIBRATIONS`): f64 p = 8 [586.07, 2586.97, 60190.34, 597.55, 2734.66,
1037.53, 1230.28, 3.0825], f64 p = 18 [3367.09, 20960.12, 346075.79, 3439.55, 22713.93,
4507.78, 5069.52, 3.0750], f32 p = 8 [644.45, 2666.24, 41330.64, 639.69, 2755.85, 1067.26,
1271.82, 1.3652] ns (P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P). Between them α_M2L per box
scales by (p + 1)² (measured: (p + 1)^1.5 from p = 3 to 8, (p + 1)^1.9 from 8 to 12 and
(p + 1)^2.3 from 8 to 18, as s and H grow with p), M2M and L2L by (p + 1)³ as under
`Rotation`. Per box the constant is about twice the one-core cost of §4's parts at p = 8
(32 µs on AVX2): it carries the waves of the ghost boxes (the halo is 0.5–1.1 times the
own boxes on the cube's 40 ranks at N = 10⁷, and every held box of the level gets its
waves) and the node's memory bandwidth shared by 40 ranks. The model is within 2.0–8.1 % of every
rank's sum of kind times with each run's own constants, as T7's were (1.1–8.0 %); the
spread of α_M2L over the workloads (±6 % at p = 8, ±8 % at p = 18) is of the size one
expects from the ghosts' waves, which the count of own boxes does not see.

**The cut with these constants** (`tools/scaling/run.sh t13-cut`, job 605581,
node-c11a-002: the calibration's launches with the second cut, `--cut work`): the compute
imbalance (max / mean of the ranks' compute stages) and the evaluation, against the first
cut of job 605559:

| run | imbalance, first cut → work cut | evaluation ms, first cut → work cut | gain |
| --- | --- | --- | ---: |
| Plummer, f64 p = 8 | 1.352 → 1.127 | 6758.5 → 5769.4 | **1.17** |
| Plummer, f64 p = 18 | 1.390 → 1.078 | 31464.2 → 24403.1 | **1.29** |
| Plummer, f32 p = 8 | 1.407 → 1.072 | 6417.1 → 4892.2 | **1.31** |
| clusters, f64 p = 8 | 1.349 → 1.062 | 6943.4 → 5696.1 | **1.22** |
| clusters, f64 p = 18 | 1.333 → 1.066 | 31264.7 → 26084.4 | **1.20** |
| clusters, f32 p = 8 | 1.391 → 1.053 | 6561.1 → 5131.5 | **1.28** |
| cube, f64 p = 8 | 1.046 → 1.093 (no recut) | 1757.0 → 1861.4 | 0.94 |
| cube, f64 p = 18 | 1.043 → 1.046 (no recut) | 6248.8 → 6336.8 | 0.99 |
| cube, f32 p = 8 | 1.059 → 1.053 (no recut) | 1162.8 → 1149.5 | 1.01 |

On the Plummer sphere and the clusters the work cut takes the imbalance from 1.33–1.41
to 1.05–1.13 and the evaluation 1.17–1.31× faster, as T7's cut did under `Rotation`; the
model is then within 1.8–3.5 % of every rank's sum of kind times. On the cube the
library keeps the first cut (a cut by work would gain less than `WORK_CUT_MIN_GAIN`, 2 %,
of the modelled maximum; the modelled imbalance is 1.03), so the same cut ran on another
node: the differences there (0.94–1.01) are
the two nodes', not the cut's.

## 9. f32

**Offered** at every degree the builder accepts, p ≤ 20 as in f64 (step 7 asked for p = 3,
6, 8 if the errors meet requirement 1); measured at p = 3, 6, 8 on the three machines and at
p = 12 and 18 on the M3 Max (below). The waves in f32 lose nothing measurable. The FMM's error under `PlaneWave` over the exact
M2L's: at most 1.0027 in every f32 scenario of `tests/mpi_exec.rs` (2–3 per rank count, at
1, 2, 4, 8 and 80 ranks), 1.0014 (p = 3) and 1.0036 (p = 8) on the C3.2 cube and 1.0004
and 0.9951 on the C3.3 Plummer sphere in `tests/multi_rank.rs` (N = 10⁵, eight charge
vectors, the 2-rank run), and 0.993–1.0024 in φ at N = 10⁶ in every f32 sweep below. The
graded tree at f32 p = 6 is in `tests/invariance.rs` (bit for bit across 1–8 ranks). The
brief's concern, wave terms of about 3 × 10⁸ at p = 8, does not arise in this form: the
multipoles are scaled by 2^{−(n+1)} before the outgoing powers (CONVENTIONS §3.14), and
the measured errors show no loss from f32's rounding. So no settings error: f32 is
offered like f64.

The f32 rows of `t13-host` (as §5):

**M3 Max:**

| dist | N | prec | p | threads | PW: evaluation ms | best other ms | **ratio** | PW: M2L µs per V pair | best other | **ratio** | φ error, PW / exact M2L |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 10^6 | f32 | 3 | 1 | 585.5 (PW) | 554.1 (F) | **0.95** | 0.0193 | 0.0154 (F) | **0.80** | 1.0024 |
| cube | 10^6 | f32 | 3 | 12 | 64.7 (PW) | 60.5 (F) | **0.94** | 0.0023 | 0.0017 (F) | **0.74** | 1.0024 |
| cube | 10^6 | f32 | 6 | 1 | 970.1 (PW) | 1065.7 (F) | **1.10** | 0.0485 | 0.0662 (F) | **1.36** | 1.0024 |
| cube | 10^6 | f32 | 6 | 12 | 98.5 (PW) | 110.3 (F) | **1.12** | 0.0050 | 0.0064 (F) | **1.28** | 1.0024 |
| cube | 10^6 | f32 | 8 | 1 | 1308.6 (PW) | 1827.9 (F) | **1.40** | 0.0688 | 0.1575 (F) | **2.29** | 1.0005 |
| cube | 10^6 | f32 | 8 | 12 | 128.9 (PW) | 177.7 (F) | **1.38** | 0.0069 | 0.0149 (F) | **2.16** | 1.0005 |
| plummer | 10^6 | f32 | 3 | 1 | 1790.3 (PW) | 1744.6 (F) | **0.97** | 0.0195 | 0.0150 (F) | **0.77** | 0.9958 |
| plummer | 10^6 | f32 | 3 | 12 | 186.0 (PW) | 185.3 (F) | **1.00** | 0.0025 | 0.0017 (F) | **0.68** | 0.9958 |
| plummer | 10^6 | f32 | 6 | 1 | 4328.9 (PW) | 4543.7 (F) | **1.05** | 0.0493 | 0.0670 (F) | **1.36** | 0.9995 |
| plummer | 10^6 | f32 | 6 | 12 | 456.3 (PW) | 480.7 (F) | **1.05** | 0.0053 | 0.0063 (F) | **1.19** | 0.9995 |
| plummer | 10^6 | f32 | 8 | 1 | 6768.1 (PW) | 7555.7 (F) | **1.12** | 0.0686 | 0.1619 (F) | **2.36** | 0.9932 |
| plummer | 10^6 | f32 | 8 | 12 | 646.0 (PW) | 747.1 (F) | **1.16** | 0.0072 | 0.0151 (F) | **2.10** | 0.9932 |

**locust:**

| dist | N | prec | p | threads | PW: evaluation ms | best other ms | **ratio** | PW: M2L µs per V pair | best other | **ratio** | φ error, PW / exact M2L |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 10^6 | f32 | 3 | 1 | 991.0 (PW) | 995.9 (F) | **1.01** | 0.0254 | 0.0281 (F) | **1.11** | 1.0024 |
| cube | 10^6 | f32 | 3 | 72 | 44.4 (PW) | 42.8 (F) | **0.96** | 0.0008 | 0.0005 (B) | **0.62** | 1.0024 |
| cube | 10^6 | f32 | 6 | 1 | 1502.0 (PW) | 1690.7 (F) | **1.13** | 0.0692 | 0.1039 (F) | **1.50** | 1.0024 |
| cube | 10^6 | f32 | 6 | 72 | 53.4 (PW) | 54.1 (F) | **1.01** | 0.0016 | 0.0018 (F) | **1.12** | 1.0024 |
| cube | 10^6 | f32 | 8 | 1 | 1943.0 (PW) | 2711.0 (F) | **1.40** | 0.1001 | 0.2315 (F) | **2.31** | 1.0005 |
| cube | 10^6 | f32 | 8 | 72 | 60.2 (PW) | 73.2 (F) | **1.22** | 0.0022 | 0.0041 (F) | **1.86** | 1.0005 |
| plummer | 10^6 | f32 | 3 | 1 | 2585.6 (PW) | 2567.6 (F) | **0.99** | 0.0257 | 0.0254 (F) | **0.99** | 0.9958 |
| plummer | 10^6 | f32 | 3 | 72 | 82.4 (PW) | 76.8 (F) | **0.93** | 0.0011 | 0.0005 (F) | **0.45** | 0.9958 |
| plummer | 10^6 | f32 | 6 | 1 | 5693.7 (PW) | 5998.2 (F) | **1.05** | 0.0674 | 0.1026 (F) | **1.52** | 0.9995 |
| plummer | 10^6 | f32 | 6 | 72 | 136.7 (PW) | 138.1 (F) | **1.01** | 0.0017 | 0.0019 (F) | **1.12** | 0.9995 |
| plummer | 10^6 | f32 | 8 | 1 | 8530.1 (PW) | 9796.0 (F) | **1.15** | 0.0960 | 0.2333 (F) | **2.43** | 0.9932 |
| plummer | 10^6 | f32 | 8 | 72 | 182.0 (PW) | 204.1 (F) | **1.12** | 0.0022 | 0.0044 (F) | **2.00** | 0.9932 |

**Kathleen:**

| dist | N | prec | p | threads | PW: evaluation ms | best other ms | **ratio** | PW: M2L µs per V pair | best other | **ratio** | φ error, PW / exact M2L |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 10^6 | f32 | 3 | 1 | 981.5 (PW avx512) | 866.5 (F avx512) | **0.88** | 0.0423 | 0.0280 (F avx512) | **0.66** | 1.0024 |
| cube | 10^6 | f32 | 3 | 40 | 86.3 (PW avx512) | 74.7 (F avx512) | **0.87** | 0.0029 | 0.0016 (F avx2) | **0.55** | 1.0024 |
| cube | 10^6 | f32 | 6 | 1 | 1848.2 (PW avx512) | 1734.5 (F avx512) | **0.94** | 0.0981 | 0.0870 (F avx512) | **0.89** | 1.0024 |
| cube | 10^6 | f32 | 6 | 40 | 126.0 (PW avx2) | 111.4 (F avx512) | **0.88** | 0.0056 | 0.0043 (F avx512) | **0.77** | 1.0024 |
| cube | 10^6 | f32 | 8 | 1 | 2660.3 (PW avx512) | 2727.6 (F avx512) | **1.03** | 0.1368 | 0.1577 (F avx512) | **1.15** | 1.0005 |
| cube | 10^6 | f32 | 8 | 40 | 161.6 (PW avx512) | 151.1 (F avx512) | **0.94** | 0.0077 | 0.0079 (F avx512) | **1.03** | 1.0005 |
| plummer | 10^6 | f32 | 3 | 1 | 4292.9 (PW avx512) | 4098.9 (F avx512) | **0.95** | 0.0418 | 0.0270 (F avx512) | **0.65** | 0.9958 |
| plummer | 10^6 | f32 | 3 | 40 | 249.6 (PW avx512) | 227.5 (B avx512) | **0.91** | 0.0032 | 0.0017 (B avx512) | **0.53** | 0.9958 |
| plummer | 10^6 | f32 | 6 | 1 | 10576.2 (PW avx512) | 10382.2 (F avx512) | **0.98** | 0.0984 | 0.0858 (F avx512) | **0.87** | 0.9995 |
| plummer | 10^6 | f32 | 6 | 40 | 504.2 (PW avx2) | 499.8 (B avx512) | **0.99** | 0.0059 | 0.0048 (F avx2) | **0.81** | 0.9995 |
| plummer | 10^6 | f32 | 8 | 1 | 16871.7 (PW avx512) | 16890.9 (F avx512) | **1.00** | 0.1387 | 0.1595 (F avx512) | **1.15** | 0.9932 |
| plummer | 10^6 | f32 | 8 | 40 | 780.1 (PW avx2) | 756.0 (B avx512) | **0.97** | 0.0079 | 0.0075 (F avx512) | **0.95** | 0.9928 |

In f32 the evaluation under `PlaneWave` against the fastest other strategy: p = 3
0.94–1.00 (M3 Max), 0.93–1.01 (locust), 0.87–0.95 (Kathleen); p = 6 **1.05–1.12**,
1.01–1.13 and 0.88–0.99; p = 8 **1.12–1.40**, **1.12–1.40** and 0.94–1.03. The plane
waves gain less in f32 than in f64, probably because their steps outside the vector
kernels (`modes`, `local`, the rotations; §4) are scalar code, which f32 does not make
faster, while `Batched`'s kernels process twice the values per vector (α_M2L per box is
1.46× smaller in f32 than in f64 on Kathleen, §8).

**f32 at p = 12 and 18** (M3 Max, one rank at 12 threads, N = 10⁶, the errors of one charge
vector; `bench-results/t13/m3max-f32`, run on 2026-10-19 at 21:2x while other work loaded the
machine, load average 2.5–7.7, so the times are indicative only). Here every strategy's
error in f32 is at f32's rounding level, about 10⁻⁶, not at the truncation error, and the
exact M2Ls differ among themselves: `Batched` (whose tables are `Dense`'s) has the largest
φ error, `Rotation` the smallest. Relative L2 errors against the direct sum:

| problem | p | φ: `PlaneWave` / `Batched` / `Rotation` | ∇φ: `PlaneWave` / `Batched` / `Rotation` | evaluation ms: `PlaneWave` / fused `Batched` / `Rotation` |
| --- | ---: | --- | --- | --- |
| cube | 12 | 9.80e-7 / 2.86e-6 / 1.08e-6 | 2.84e-6 / 2.86e-6 / 2.85e-6 | 217 / 541 / 1,504 |
| cube | 18 | 7.19e-7 / 2.73e-6 / 8.09e-7 | 9.93e-7 / 1.06e-6 / 9.91e-7 | 493 / 2,685 / 3,749 |
| Plummer | 12 | 1.46e-6 / 3.38e-6 / 1.51e-6 | 1.80e-6 / 1.75e-6 / 1.72e-6 | 1,330 / 1,838 / 2,919 |
| Plummer | 18 | 7.29e-7 / 3.15e-6 / 8.95e-7 | 7.71e-7 / 8.07e-7 / 7.73e-7 | 2,879 / 6,354 / 8,617 |

`PlaneWave`'s φ error is 0.81–0.97× `Rotation`'s and 0.23–0.43× `Batched`'s, its ∇φ error
0.94–1.05× either's: within requirement 1's 1.1× of the most accurate exact M2L in every
case. Its evaluation was 1.38–5.44× faster than the fastest other (fused `Batched`).

## 10. For T11

Requirement 6 keeps `PlaneWave` opt-in until T11. From §5, §6 and §9, the proposal for
`Auto`'s rule on the host (the device keeps its own rule; `PlaneWave` runs there as
`Dense`):

| precision | p | M3 Max, locust (NEON) | Kathleen (AVX-512) | an AVX2-only x86 (Kathleen at AVX2) |
| --- | --- | --- | --- | --- |
| f64 | ≥ 12 | **`PlaneWave`** (1.80–6.38×) | **`PlaneWave`** (1.35–5.55×) | **`PlaneWave`** (1.48–8.10×) |
| f64 | 8 | **`PlaneWave`** (1.14–1.80×; node splits 1.05–1.73×) | **`PlaneWave`** (1.08–1.25×; splits 1.03–1.24×) | **`PlaneWave`** (1.13–1.39×) |
| f64 | 6 | **`PlaneWave`** (1.03–1.20×) | the exact M2L (0.96–0.99×) | either (0.98–1.05×) |
| f64 | ≤ 5 | the exact M2L (p = 3: 0.94–1.01×) | the exact M2L (0.83–0.99×) | the exact M2L (0.84–0.97×) |
| f32 | 8 | **`PlaneWave`** (1.12–1.40×) | the exact M2L (0.94–1.03×) | either (1.00–1.17×) |
| f32 | 6 | either (1.01–1.13×) | the exact M2L (0.88–0.99×) | the exact M2L (0.95–1.02×) |
| f32 | ≤ 5 | the exact M2L (0.93–1.01×) | the exact M2L (0.87–0.95×) | the exact M2L (0.88–0.97×) |

(Each figure is the evaluation's time under the fastest other strategy over `PlaneWave`'s,
over the one-rank runs at 1 thread and every core, N = 10⁶ and 10⁷, and where stated the
node splits of §6.) So, as a rule: **`PlaneWave` for f64 at p ≥ 8 on every machine, and
from p = 6 on NEON; for f32 from p = 8 on NEON**; otherwise the rule of today (`Batched`,
`Rotation` by the node budget). p = 7 was not timed here, f32 at p ≥ 12 only on the M3 Max
under load (1.38–5.44×, §9), and the clusters only under `PlaneWave` (§8); T11 should measure them before fixing the thresholds, and refit the cut's constants
there with the others (the T2 handoff's open item). Two further points for T11:
- **Memory in `Auto`'s budget**: the waves add 4H values per held box of the largest
  level (8.1 KB per box at f64 p = 8), in place of `Batched`'s dense tables (17–72 MB per
  rank in f64 at p = 8–12); at one rank on a large problem that is more (+249 MB on the
  10⁶ cube at p = 8), at one rank per core less or about the same (§7). The node budget
  should count them.
- **The x86 gap**: on Kathleen `PlaneWave`'s steps outside the vector kernels (`modes`,
  `local`, the rotations) cost 2.5–3× the M3 Max's per core (§4), so vector bodies for them
  (as for `merge` and the Fourier passes) would move Kathleen's threshold down; and at one
  rank per core the ghosts' waves (§6) are the other cost to look at. Neither is in T13's
  scope.

## Appendix: runs

Every run on 2026-10-19 from the working tree of `tbetcke/phase6_t13` (main at c681404
plus this task's changes, uncommitted; the remote copies by `tools/kathleen/sync.sh` and
`tools/gh200/sync.sh`). Raw output in `bench-results/t13/` (ignored by git).

**M3 Max** (Apple M3 Max, 12 performance and 4 efficiency cores, macOS; unsandboxed, the
machine otherwise quiet):
- `RUST_MIN_STACK=8388608 cargo test --workspace` (to 10:21), `.github/scripts/run-mpi-tests.sh`
  at 2, 4 and 8 ranks (to 10:39), the ignored `level_every_degree` and
  `pair_m2l_every_degree`, `plane_wave_parts` (before the sweeps);
- the chain from 10:40: the ignored `tests/accuracy.rs` and `tests/adaptive.rs` (release),
  `tests/multi_rank.rs` at 1 and 2 ranks (release, ignored; 35 and 50 min; 4 and 8 ranks
  ran on Kathleen, below; a 4-rank run here was stopped to free the machine for the
  sweeps), `tests/mpi_exec.rs` in debug at one rank with the budget and
  with `--features cpu --release`;
- `tools/scaling/run.sh t13-host t13-node` (12:13–13:53; 200 launches; 12 threads, the
  performance cores) and `plane_wave_parts --ps 3,6,8,12,18`.

**locust** (GH200, 72 Neoverse V2 cores; detached runs under `/data/ucahtbe/logs/p6t13-*`,
the H100 in use by another user throughout, the CPU otherwise idle, load ≈ 1 before the
sweeps): `cargo test -p nd-fmm-simd --release --test waves` (scalar and NEON bit for bit) and
`cargo test -p nd-fmm-tables plane_wave`; `tools/scaling/run.sh t13-host` (10:21–13:39,
168 launches), `t13-node` (13:42–14:06, 96 launches; the tree re-synced first for the cost
constants), `plane_wave_parts --ps 3,6,8,12,18` (15:34).

**Kathleen** (one node of 2 × 20 Xeon Gold 6248 cores, 192 GB, unless stated; every job
excluding node-c11a-011):

| job | what | nodes | elapsed | node-hours |
| --- | --- | ---: | ---: | ---: |
| 605553 | `nd-fmm-simd`'s `tests/waves.rs`, debug and release (scalar, AVX2, AVX-512 bit for bit) | 1 | 0:05:20 | 0.09 |
| 605556, 605557 | the kernels' release tests, a build | 1 | 0:02:09 | 0.04 |
| 605558 | `plane_wave_parts` per ISA (§4; cancelled after the parts) | 1 | 0:13:51 | 0.23 |
| 605559 | `t13-calibrate` (§8), `t13-host` on the cube (§5) | 1 | 3:05:35 | 3.09 |
| 605560 | `t13-host` on the Plummer sphere (§5, §9) | 1 | 5:00:20 | 5.01 |
| 605561 | `correctness.sbatch`, `test` QoS (§2) | 2 | 0:49:32 | 1.65 |
| 605562 | `t13-node` (§6, §7) | 1 | 0:52:40 | 0.88 |
| 605563 | `check.sbatch` (§2; the spike's two known failures) | 1 | 0:33:21 | 0.56 |
| 605564 | the ignored C3.2 and C3.3 gates; `tests/multi_rank.rs` at 1 rank, then 2 (timed out at 2 h: every rank bound to one core; cancelled) | 1 | 3:33:23 | 3.56 |
| 605634 | `tests/multi_rank.rs` at 4 and 8 ranks, unbound | 1 | 3:34:54 | 3.58 |
| 605636 | `tests/multi_rank.rs` at 2 ranks, unbound (timed out at 2 h) | 1 | 2:00:24 | 2.01 |
| 605581 | `t13-cut` (§8) | 1 | 0:06:10 | 0.10 |

In all 20.8 node-hours; no job used more than two nodes.

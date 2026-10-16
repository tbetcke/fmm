# Phase 6 T5: several right-hand sides (C6.3)

Phase 6, task T5 ([docs/phase6/T5-multiple-rhs.md](../../docs/phase6/T5-multiple-rhs.md));
design [docs/design/optimisation.md](../../docs/design/optimisation.md) §4, signed off on
2026-10-15 (§4.5 records what was built). Measured on 2026-10-16 from branch
`tbetcke/phase6_t5` (main at 1038c7f plus this task's changes), release builds, rustc
1.99.0, every BLAS thread variable at 1. Kathleen is usable (docs/phase6/README.md): one
node for the throughput, two for the exchanges; no job used more than two nodes.

## Summary

1. **Each vector bit for bit its single-vector evaluation.** An `Fmm` of m vectors
   (`FmmBuilder::vectors`, `Fmm::evaluate_many`) gives every vector exactly the output of
   the one-vector `Fmm` on that vector's charges: on every `tests/mpi_exec.rs` scenario at
   m = 3 (debug and release) and 8 (release), blocking and overlapped, at 1, 2, 4 and 8
   ranks on the M3 Max, 1–8 on locust and 2–80 on Kathleen; on the CPU runtime, Metal
   and CUDA, with the device's transfers the formula of device-path §14 with m
   times the bytes; and in the C5.1 host gate with eight vectors under `auto` and `Batched`
   at 2, 4 and 8 ranks (§2). The plan is unchanged: every exchange sends the messages it
   sent for one vector, with m times the values (§5).
2. **Throughput per vector** (the N = 10⁶ cube, φ and ∇φ, one rank per core, m = 8 against
   m = 1): with `Batched` **0.61–0.64 at p = 3** (locust f32 0.71) and **0.69–0.83 at p =
   8** on the three machines; the gain saturates at m = 4–8 (§3). With the default
   strategy at p = 8, `Rotation` (which `Auto` resolves with one rank per core), only
   0.89–0.94: Rotation shares nothing across the vectors, so with eight vectors `Batched`
   is 1.4–4.5× faster per vector than the default. A proposal for T11: take m into the
   host rule (§6).
3. **Where it comes from** (§3.4, §4): P2P with ∇φ at 0.51–0.70 per vector (the
   several-vector kernel 1.5–1.9× per vector with ∇φ and 3.9–4.6× for φ alone, bounded by
   the registers), P2M and L2P at 0.52–0.68 (the harmonics shared), `Batched`'s M2L at p = 8 at
   0.75–0.82 on Kathleen and locust and 0.95 on the M3 Max, `Dense`'s at 0.84–0.92,
   `Rotation`'s at 0.98–1.00. T1's model (0.3–0.5 at p = 3, 0.6–0.8 at p = 8) holds at
   p = 8 and was optimistic at p = 3, where it took P2P's gain from the operation count.
4. **The device** (§3.5): the first sweep found the device path's M2L quadratic in m (on
   Metal 10× per vector at m = 32): the grouped GEMM's row reduction walks every entry
   once per chunk, and a fixed chunk budget gave m times the chunks. The default budget is
   now per vector (m times its bytes, at most an eighth of the device's memory; the bits do
   not depend on it). After it the evaluation per vector at m = 32 is 0.40–0.65 of one
   vector's on CUDA (0.63–1.02 before; mostly the host's forward and backward, which fall
   per vector, while dense M2L stays at one vector's cost) and flat on Metal (0.91–1.04,
   dense M2L 78 % of it at p = 8). On CUDA `Rotation` gains as much per vector as `Dense`
   (0.45–0.62 at m = 32; its M2L 0.81–0.90, unlike the host's) and stays 2–16 % slower.
5. **The exchanges at 80 Kathleen ranks** (§5): the same messages for every m, the bytes
   as designed ((3 + m)/4 for the sources, m for the rest); the replicated top's
   all-gather per vector falls from 4.8 to 3.0 ms (N = 10⁶) and from 17.2 to 8.2 ms
   (N = 10⁷) between m = 1 and 32, and the exchanges together stay at 1–4 % of the
   evaluation.

## 1. What T5 adds

- **The interface** (optimisation.md §4.2): `FmmBuilder::vectors(m)` (default 1, fixed at
  build; `SettingsError::ZeroVectors` for 0), `Fmm::evaluate_many(&[&[T]])` and
  `evaluate_many_into`, and `OutputMany` (`vectors`, `potential` and `gradient` vector after
  vector, `potential_of(v)`, `gradient_of(v)`, the timings). `Fmm::evaluate` is unchanged
  and refused on an `Fmm` of m > 1 with `FmmError::VectorCount { expected, actual }`, as is
  `evaluate_many` with a wrong number of vectors; the count rides on the charge-length
  all-reduce every evaluation already makes, so every rank returns the same error.
  `evaluate_many` with m = 1 is `evaluate`.
- **The layouts** (CONVENTIONS §3.13, "Several charge vectors", proposed in the PR, no
  version bump: the in-memory layout of a chunk, not a stored format): a source point
  holds its coordinates, then its m charges (3 + m values, `q[i m + v]`); a target leaf's
  output chunk holds the m vectors' φ̂ one after the other, then their ĝ (m n or 4 m n
  values); a box holds m expansions vector after vector (m (p + 1)² values). With m = 1
  every layout is the old one. The output is vector after vector, not point-major as
  optimisation.md §4.1 proposed: P2P keeps the targets in vector lanes, so it loads and
  stores one vector's potentials of W consecutive targets directly (§4.5 there).
- **The plan unchanged**: `LaplaceOperator::with_vectors(m)` reports m-times sizes
  (`FmmSizes`), so the stores, the source exchange, the upward reduction, the replicated
  top's all-gather, the multipole exchange (along the forwarding tree) and both
  redistributions carry m vectors in the messages they carried one in. No `nd-fmm-plan`
  change.
- **The operators**, each vector given exactly the single-vector chain, only what does
  not depend on the charge shared:
  - P2M, P2L, L2P, M2P: `nd_fmm_ref::leaf::{p2m, p2l, l2p, m2p}_many`, the harmonics of a
    point formed once (the single-vector operators are their m = 1 case);
  - M2M, L2L and M2L under `Dense`, `Classes` and `Rotation`: the table (or rotation)
    applied to each vector's column in turn; under `Batched` each pair contributes m
    columns to the block's kernel call, so a table is read once per block for m times the
    columns;
  - P2P: `nd_fmm_simd::P2pKernel::evaluate_many` (every ISA; `unsafe` only in `arch`),
    the vectors in register groups with ρ formed once per source and target vector per
    group (NEON: blocks of 2 target vectors with groups of 8 vectors for φ, 1 and 4 with
    ∇φ; AVX2: 1 and 8, 1 and 4; chosen by timing, §4); the reference P2P vector by vector;
  - the output pass: per vector, the single-vector pass's scales and division.
- **The device** (device-path.md §14, "Several charge vectors"): the grouped GEMMs (M2M,
  L2L, dense M2L) and rotation M2L run on views with m columns per box (box b's vector v
  is column b m + v), with no kernel change; P2M, P2L, L2P, M2P and P2P take the vectors
  along the grid's z (one launch per level call as before); the output pass
  (`gather_output`) loops over the vectors; the exchanges move m n values per box; the
  GEMM decisions are keyed by the plan's pairs, so FMMs of any m take the same tuned
  choices, and the default chunk budget is per vector (§3.5). The transfers are
  device-path §14's with m times the bytes: on one rank one upload of m N_s charges, one
  download of o m N_t values (o = 1, or 4 with gradients) and one sync per evaluation; on
  several ranks the exchanges' packed buffers m times as large, the same launches and
  syncs.
- **The harness**: `scaling --rhs m` (an `Fmm` of m vectors, `evaluate_many`; the summary's
  `rhs`, the traffic and memory models with m) and `--backend host|cpu|metal|cuda`;
  `--errors`, `--reference` and `--invariance` refuse m > 1 (`HarnessError::SeveralVectors`).
  `tools/scaling/run.sh` sweeps `t5-host`, `t5-threads`, `t5-device` and `t5-nodes`, each
  launch at m = 1, 2, 4, 8, 16, 32 (`T5_RHS`), overlap off, three timed evaluations, on
  the host with `--kinds`.

## 2. Correctness

Error measure: exact equality of bit patterns (of every output value as its f64 widening),
vector v of an `Fmm` of m vectors against the single-vector evaluation of vector v's
charges, unless a bound is named.

| check | where | result |
| --- | --- | --- |
| `nd_fmm_ref::leaf::*_many` against the single-vector operators, p = 0, 3, 8, m = 1, 3, 8, and their refusals (`tests/leaf/many.rs`) | M3 Max | bit for bit |
| `P2pKernel::evaluate_many` against `evaluate` vector by vector, every ISA, f32 and f64, φ and φ with ∇φ, 0–40 sources, 0–33 targets, m = 1–17, nonzero initial outputs, coincident and repeated sources; the generic body against the single body (`tests/p2p/many.rs`, `arch::tests`) | M3 Max (scalar, NEON), locust (scalar, NEON), Kathleen (scalar, AVX2; job 604424) | bit for bit |
| the device kernels: P2P and the four leaf operators on m = 3 vectors against m one-vector launches, every layout, `gather_output` at m = 1 and 3 (`several_vectors` in `tests/kernels/{p2p, leaf}.rs`, `movement.rs`) | M3 Max (CPU runtime, Metal), locust (CPU runtime, CUDA), Kathleen (CPU runtime; job 604424) | bit for bit |
| every `tests/mpi_exec.rs` scenario (`check_vectors` after each, on its cheapest one-thread evaluation): `evaluate_many` with one vector is `evaluate`; a wrong vector count is `VectorCount` on every rank; each vector of an `Fmm` of 3 vectors (release: also 8, blocking and overlapped, and `Batched` with 3 where the strategy is `Dense`) equals its single-vector evaluation; the output pass the oracle's (`reference_output_many`). Debug: 3 vectors overlapped, scenarios over 0.5 s left to the release runs | debug: M3 Max at 1, 2, 4, 8 ranks; locust at 2, 4, 8 (the MPI list); Kathleen at 2, 4, 8 and 80 (job 604425). Release (`--features cpu`): M3 Max at 1, 2, 4, 8; locust at 1 | bit for bit: per debug run 19–22 evaluations of one vector and as many of three; per release run 21–23 of one vector, 42–46 of three, 38–42 of eight, 34–36 of three under `Batched`, 63–69 of three on the CPU runtime (after the per-vector chunk budget, §3.5: again at 1, 2 and 4 ranks on the M3 Max) |
| the device repeat of every scenario (`device_common::check_three_vectors`): 3 vectors on the device, each vector the one-vector device `Fmm`'s, blocking and overlapped; the transfers the formula of device-path §14 with m; on the host fallback the host's bits | CPU runtime (`mpi_exec --features cpu`) on the M3 Max at 1, 2, 4, 8 ranks and locust at 1; CUDA (`tests/device_cuda.rs`) on locust, f32 and f64 (again after the per-vector chunk budget); `tests/device_ranks.rs` on the CPU runtime at 2 and 4 ranks (M3 Max) and on CUDA at 2 and 4 (locust), each before and after the per-vector chunk budget; Metal (`tests/device_metal.rs`, f32, every scenario) and `device_ranks` on Metal at 2 ranks, after it | bit for bit, the counters as the formula |
| the C5.1 host gate (`tests/multi_rank.rs`) with `auto` and `Batched` on an `Fmm` of the eight charge vectors: N = 10⁵ cube and Plummer sphere, f64 p = 3, 8, 18 and f32 p = 3, 8, every vector bit for bit the one-rank single-vector `Fmm` over the union, within 100 u_T of the one-rank run in point order, the C3.3 gate | M3 Max at 2, 4 and 8 ranks; locust at 4 and 8 ranks, at 2 ranks 17 of the 20 eight-vector workload points before the run's 3,600 s limit (the Plummer sphere under `Batched` at f64 p = 18 and f32 p = 3 and 8 not reached) | bit for bit; the C3.3 ratios 1.21–1.26, as for one vector |
| rank-count invariance (`tests/invariance.rs`, `scaling --invariance` at 80 against 40, 20, 10) | the MPI list on the three machines; Kathleen job 604425 | passes (one vector; unchanged by T5) |
| `scaling --rhs m`: the summary, the refusals of `--errors`, `--reference` and `--invariance` with m > 1 (`tests/scaling.rs`) | M3 Max at 1 and 2 ranks | passes |

At one rank the release `multi_rank` first compared its eight-vector variants with
themselves (the one-rank run is its own reference for one vector, not for eight); the
test now builds the single-vector references for them (`size == 1 && vectors == 1`), and
the M3 Max runs at 2, 4 and 8 ranks used the fixed binary. The 1-rank rows of locust's
ignored run and of the M3 Max's first run are therefore no check of the eight vectors;
on several ranks the check was always against the single-vector union.

The debug run of `tests/mpi_exec.rs` at one rank on the M3 Max grew from 106 s (T2) to 120
s: the vector checks take 12.8 s (18 evaluations of one vector and 18 of three; the
strongly graded tree and multipole routing, whose cheapest evaluation takes over 0.5 s in
debug, are left to the release runs). The full check (3 and 8 vectors, blocking and
overlapped, `Batched`) would have added 210 s.

## 3. Throughput per right-hand side

The N = 10⁶ cube, φ and ∇φ, overlap off, on the host `KindTiming::Synchronous` (its cost
is negligible on the host); the wall time of an evaluation (the median of three after a
warm-up, the slowest rank) over m, in ms, and in brackets against m = 1.

### 3.1 Kathleen, one node (40 × 1; job 604430, node-c11b-044)

| problem | strategy | m = 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 3 | `auto` (Dense) | 94.0 | 74.1 (0.79) | 62.3 (0.66) | 62.1 (0.66) | 62.3 (0.66) | 62.0 (0.66) |
| f64 p = 3 | `batched` | 83.4 | 63.5 (0.76) | 51.3 (0.62) | 51.0 (0.61) | 51.1 (0.61) | 50.9 (0.61) |
| f64 p = 8 | `auto` (Rotation) | 570.5 | 543.2 (0.95) | 525.0 (0.92) | 523.1 (0.92) | 519.9 (0.91) | 518.8 (0.91) |
| f64 p = 8 | `batched` | 297.2 | 251.1 (0.85) | 225.3 (0.76) | 219.3 (0.74) | 217.1 (0.73) | 222.3 (0.75) |
| f32 p = 3 | `auto` (Dense) | 49.2 | 37.8 (0.77) | 33.6 (0.68) | 32.9 (0.67) | 32.9 (0.67) | 32.8 (0.67) |
| f32 p = 3 | `batched` | 43.3 | 32.6 (0.75) | 27.3 (0.63) | 26.3 (0.61) | 26.7 (0.62) | 26.1 (0.60) |
| f32 p = 8 | `auto` (Rotation) | 565.9 | 544.5 (0.96) | 536.9 (0.95) | 533.4 (0.94) | 529.6 (0.94) | 530.5 (0.94) |
| f32 p = 8 | `batched` | 169.7 | 139.4 (0.82) | 124.3 (0.73) | 119.2 (0.70) | 114.8 (0.68) | 115.8 (0.68) |

### 3.2 locust, the host (72 × 1; 2026-10-16 15:38–15:41, load average 5.6 before, GPU idle)

| problem | strategy | m = 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 3 | `auto` (Dense) | 39.4 | 25.6 (0.65) | 25.2 (0.64) | 25.3 (0.64) | 25.0 (0.63) | 25.0 (0.64) |
| f64 p = 3 | `batched` | 36.7 | 23.4 (0.64) | 22.8 (0.62) | 23.2 (0.63) | 22.9 (0.62) | 23.0 (0.63) |
| f64 p = 8 | `auto` (Rotation) | 138.9 | 124.1 (0.89) | 124.1 (0.89) | 125.7 (0.90) | 125.5 (0.90) | 125.8 (0.91) |
| f64 p = 8 | `batched` | 127.3 | 98.8 (0.78) | 87.4 (0.69) | 87.3 (0.69) | 87.2 (0.69) | 90.1 (0.71) |
| f32 p = 3 | `auto` (Dense) | 17.6 | 14.0 (0.79) | 12.8 (0.73) | 12.7 (0.72) | 12.8 (0.73) | 12.8 (0.72) |
| f32 p = 3 | `batched` | 15.5 | 12.2 (0.78) | 11.3 (0.73) | 11.0 (0.71) | 11.0 (0.70) | 11.1 (0.71) |
| f32 p = 8 | `auto` (Rotation) | 116.1 | 107.9 (0.93) | 106.8 (0.92) | 108.2 (0.93) | 108.5 (0.93) | 109.9 (0.95) |
| f32 p = 8 | `batched` | 59.7 | 51.0 (0.85) | 46.8 (0.78) | 45.3 (0.76) | 44.7 (0.75) | 45.2 (0.76) |

### 3.3 M3 Max, the host (12 × 1, the performance cores; 2026-10-16 16:30–16:40, unsandboxed, the harness's quiet gate: other processes at 20–40 % of a core, Microsoft Defender)

| problem | strategy | m = 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 3 | `auto` (Dense) | 121.9 | 92.9 (0.76) | 82.1 (0.67) | 80.2 (0.66) | 77.7 (0.64) | 86.2 (0.71) |
| f64 p = 3 | `batched` | 109.4 | 81.0 (0.74) | 71.0 (0.65) | 67.6 (0.62) | 66.0 (0.60) | 70.1 (0.64) |
| f64 p = 8 | `auto` (Rotation) | 672.8 | 634.9 (0.94) | 610.9 (0.91) | 605.8 (0.90) | 597.0 (0.89) | 603.6 (0.90) |
| f64 p = 8 | `batched` | 452.2 | 396.7 (0.88) | 381.5 (0.84) | 375.6 (0.83) | 390.7 (0.86) | 381.0 (0.84) |
| f32 p = 3 | `auto` (Dense) | 79.6 | 58.2 (0.73) | 54.2 (0.68) | 52.7 (0.66) | 51.3 (0.65) | 51.6 (0.65) |
| f32 p = 3 | `batched` | 61.4 | 45.3 (0.74) | 39.8 (0.65) | 39.6 (0.64) | 38.6 (0.63) | 37.8 (0.62) |
| f32 p = 8 | `auto` (Rotation) | 568.2 | 508.8 (0.90) | 497.9 (0.88) | 511.5 (0.90) | 532.3 (0.94) | 542.4 (0.95) |
| f32 p = 8 | `batched` | 245.5 | 214.1 (0.87) | 203.8 (0.83) | 202.8 (0.83) | 199.7 (0.81) | 199.4 (0.81) |

At m = 32 the two f64 p = 8 launches spent 6.7 and 9.5 ms per vector in the charges'
forward (1.2–4.0 at every other m and problem, 2.2–2.4 at m = 1); the machine had 7.5 GB of its
64 GB free before the sweep (other sessions' processes), and the outlier was not
investigated.

### 3.4 Where the time per vector goes (the host)

The kinds' mean over the ranks per vector (ms; in brackets against m = 1) at m = 8; m = 32
is within 0.05 of it in every column but the backward output (which absorbs the ranks'
imbalance).

| machine | problem | strategy | P2M | M2L | L2P | P2P | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kathleen | f64 p = 3 | Batched | 1.6 (0.67) | 6.3 (0.70) | 3.2 (0.53) | 31.1 (0.55) | 51.0 (0.61) |
| Kathleen | f64 p = 3 | Dense | 1.6 (0.67) | 15.7 (0.89) | 3.4 (0.56) | 31.2 (0.56) | 62.1 (0.66) |
| Kathleen | f64 p = 8 | Batched | 10.2 (0.68) | 136.6 (0.82) | 14.8 (0.55) | 31.2 (0.56) | 219.3 (0.74) |
| Kathleen | f64 p = 8 | Rotation | 10.1 (0.68) | 420.6 (0.99) | 14.1 (0.55) | 31.1 (0.56) | 523.1 (0.92) |
| Kathleen | f32 p = 3 | Batched | 1.6 (0.61) | 4.0 (0.66) | 3.2 (0.52) | 13.3 (0.58) | 26.3 (0.61) |
| Kathleen | f32 p = 8 | Batched | 9.9 (0.65) | 65.5 (0.79) | 14.6 (0.53) | 13.3 (0.58) | 119.2 (0.70) |
| Kathleen | f32 p = 8 | Rotation | 9.9 (0.65) | 452.0 (1.00) | 14.2 (0.54) | 13.3 (0.58) | 533.4 (0.94) |
| locust | f64 p = 3 | Batched | 0.2 (0.62) | 2.8 (0.85) | 0.7 (0.61) | 15.5 (0.58) | 23.2 (0.63) |
| locust | f64 p = 8 | Batched | 1.3 (0.61) | 55.9 (0.75) | 3.1 (0.57) | 15.6 (0.58) | 87.3 (0.69) |
| locust | f64 p = 8 | Rotation | 1.3 (0.61) | 91.5 (0.98) | 3.1 (0.57) | 15.6 (0.58) | 125.7 (0.90) |
| locust | f32 p = 3 | Batched | 0.2 (0.62) | 1.6 (0.80) | 0.7 (0.59) | 6.5 (0.70) | 11.0 (0.71) |
| locust | f32 p = 8 | Batched | 1.4 (0.61) | 28.7 (0.79) | 3.1 (0.56) | 6.5 (0.70) | 45.3 (0.76) |
| M3 Max | f64 p = 3 | Batched | 1.3 (0.66) | 13.4 (0.90) | 3.9 (0.67) | 37.1 (0.54) | 67.6 (0.62) |
| M3 Max | f64 p = 8 | Batched | 7.4 (0.64) | 286.1 (0.95) | 18.1 (0.58) | 36.4 (0.53) | 375.6 (0.83) |
| M3 Max | f64 p = 8 | Rotation | 7.3 (0.63) | 514.1 (0.99) | 18.4 (0.55) | 36.2 (0.51) | 605.8 (0.90) |
| M3 Max | f32 p = 3 | Batched | 1.3 (0.64) | 7.1 (0.75) | 3.5 (0.61) | 19.3 (0.62) | 39.6 (0.64) |
| M3 Max | f32 p = 8 | Batched | 7.2 (0.63) | 139.9 (0.96) | 18.4 (0.59) | 20.3 (0.63) | 202.8 (0.83) |

- **P2P** (φ and ∇φ) runs at 0.51–0.58 of its one-vector time per vector in f64 and at
  0.58 (AVX2), 0.62 (M3 Max) and 0.70 (locust) in f32, the kernel's gain with gradients
  (§4); it is most of the evaluation at p = 3.
- **P2M and L2P** share the harmonics: 0.61–0.68 and 0.52–0.67.
- **M2L** gains only where a table is read for several columns at once: `Batched` 0.70–0.85
  on Kathleen and locust and 0.75–0.96 on the M3 Max (its one-vector form already reads
  each table once per block of 64 targets, so the vectors add only the table's reuse
  across a pair's columns; on the M3 Max the kernel already ran near the core's compute
  bound at m = 1, phase6-m2l.md §3), `Dense` 0.84–0.92 (the table from cache for the next
  column), **`Rotation` nothing** (0.98–1.00: its rotations
  and translations are applied to each vector in turn, with nothing shared but the
  coefficients' cache lines). Since `Auto` resolves Rotation at f64 and f32 p = 8 with one
  rank per core on all three machines (the node budget), the default gains only 6–10 % at
  p = 8, and per vector at m = 8 `Batched` is 2.4× (f64) and 4.5× (f32) faster than the
  default on Kathleen, 1.4× and 2.4× on locust and 1.6× and 2.5× on the M3 Max (at m = 1:
  1.9×, 3.3×, 1.1×, 1.9×, 1.5×, 2.3×).
  §6 takes this up.
- The gain saturates at m = 4–8: above it every kind's time per vector is flat, so no
  operator has a per-call or per-box overhead left that further vectors would share.

### 3.5 The device

One rank with every core for the host's parts, every kind on the device, the device's
own strategy rule (`auto`, Dense at these p on CUDA and in f32 on Metal), without
`--kinds` (whose synchronous kind timings would add a sync per level call).

locust's H100 (CUDA), with the chunk budget per vector (2026-10-16 17:41–17:46, the GPU
idle: no other compute process, 0.66 GB used, load average 1.3), and in brackets the first
sweep, under the one-vector budget (15:41–15:46, GPU idle):

| problem | m = 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 3 | 48.3 (36.6) | 36.9 (0.76; 0.83) | 31.1 (0.64; 0.77) | 29.7 (0.62; 0.73) | 25.2 (0.52; 0.65) | 24.9 (0.52; 0.67) |
| f64 p = 8 | 63.5 (53.3) | 51.8 (0.82; 0.84) | 46.1 (0.73; 0.83) | 44.5 (0.70; 0.85) | 40.3 (0.64; 0.87) | 41.3 (0.65; 1.02) |
| f32 p = 3 | 33.9 (21.2) | 26.8 (0.79; 1.06) | 18.4 (0.54; 0.80) | 16.6 (0.49; 0.64) | 15.5 (0.46; 0.68) | 13.5 (0.40; 0.63) |
| f32 p = 8 | 40.1 (29.5) | 34.5 (0.86; 1.02) | 27.0 (0.67; 0.84) | 25.3 (0.63; 0.86) | 24.6 (0.61; 0.91) | 23.1 (0.58; 1.00) |

The one-vector times of the second sweep are 20–60 % above the first's, all of it in the
host's charge forward and output backward (at f64 p = 8: 14.0 and 28.6 ms against 11.0 and
22.2; the device's kinds the same to 0.1 ms): the host's copies on locust ran slower in
the evening, with the GPU idle both times. Compare within a sweep.

Where it goes (a repeat with `--kinds`, `KindTiming::Synchronous`, which on CUDA is a
breakdown, fmm-exec/CLAUDE.md "CUDA (Phase 4S)"; 17:45–17:46; ms per vector, in brackets
against m = 1; "forward" and "backward" are the host's charge packing and forward and the
output's move into `OutputMany`, on one rank copies on the pool):

| problem | m | M2M + L2L | M2L | P2P | P2M + L2P | forward | backward | evaluation |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| f64 p = 8 | 1 | 1.1 | 15.5 | 3.3 | 1.0 | 14.0 | 28.6 | 65.2 |
| f64 p = 8 | 8 | 0.4 (0.33) | 15.0 (0.97) | 2.9 (0.90) | 0.9 (0.92) | 5.0 (0.36) | 19.4 (0.68) | 44.9 (0.69) |
| f64 p = 8 | 32 | 0.2 (0.20) | 16.9 (1.10) | 2.9 (0.89) | 0.9 (0.88) | 5.0 (0.35) | 15.0 (0.53) | 41.5 (0.64) |
| f32 p = 3 | 1 | 0.5 | 1.3 | 2.0 | 0.2 | 7.8 | 21.4 | 35.4 |
| f32 p = 3 | 8 | 0.1 (0.21) | 1.0 (0.76) | 1.9 (0.92) | 0.1 (0.60) | 2.9 (0.37) | 9.1 (0.42) | 15.7 (0.44) |
| f32 p = 3 | 32 | 0.0 (0.08) | 1.0 (0.80) | 1.8 (0.91) | 0.0 (0.45) | 2.4 (0.31) | 7.8 (0.36) | 13.6 (0.38) |

- **Dense M2L** now costs per vector what it costs for one vector (0.97–1.10 at f64 p = 8,
  0.76–0.80 at f32 p = 3); under the one-vector budget it cost 1.21× at m = 8 and 2.04× at
  m = 32 (the first sweep's repeat: 15.5, 18.7 and 31.5 ms), the chunking found on Metal
  (below). It gains nothing from the vectors at p = 8: the GEMM's gather and reduction are
  per column, and presumably its tiles read the table per tile of columns, as on Metal.
- **The host's share falls per vector**: the forward (one packing pass and one move) to
  0.31–0.37, the backward to 0.36–0.68. At m = 1 the two are 65–82 % of the evaluation on
  one rank (the device path's host overhead), so on the device the vectors pay mostly
  through the host; M2M and L2L (one octant table per call, m columns per box) fall to
  0.08–0.33, P2P to 0.89–0.92 (its time is in the pair loop, which each vector repeats),
  the leaf kinds to 0.45–0.92.

**Rotation M2L on CUDA** (`--strategy rotation`, the device's rotation kernel; 2026-10-16
18:16–18:21, the GPU idle before and after, no other compute process): the evaluation per
vector in ms at m = 1, then against it, and the time against `Dense` of the sweep above:

| problem | m = 1 | 2 | 4 | 8 | 16 | 32 | Rotation / Dense at m = 1, 8, 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f64 p = 3 | 52.0 | 39.0 (0.75) | 33.6 (0.64) | 32.0 (0.61) | 27.2 (0.52) | 26.9 (0.52) | 1.08, 1.08, 1.08 |
| f64 p = 8 | 73.1 | 59.4 (0.81) | 53.7 (0.73) | 51.4 (0.70) | 45.8 (0.63) | 45.6 (0.62) | 1.15, 1.16, 1.10 |
| f32 p = 3 | 34.5 | 28.6 (0.83) | 20.5 (0.59) | 18.0 (0.52) | 17.6 (0.51) | 15.5 (0.45) | 1.02, 1.08, 1.15 |
| f32 p = 8 | 45.2 | 39.2 (0.87) | 31.6 (0.70) | 28.7 (0.64) | 28.5 (0.63) | 26.1 (0.58) | 1.13, 1.13, 1.13 |

With `--kinds` (ms per vector, against m = 1): the rotation M2L at f64 p = 8 takes 24.3,
21.9 (0.90) and 21.4 (0.88) ms at m = 1, 8 and 32 (dense: 15.5, 15.0, 16.9), at f32 p = 3
3.7, 3.1 (0.83) and 3.0 (0.81); the other kinds as under `Dense`. So on the device Rotation
gains about as much per vector as `Dense` (0.45–0.62 at m = 32, the host's forward and
backward again most of it), unlike on the host (0.98–1.00 for its M2L): one launch per
level serves the m vectors' columns, so the launch and the rotation tables' reads are
shared, while each column's rotations are still its own. `Dense` stays 2–16 % faster at
every m, as the device rule assumes (Phase 4S decision 9). The one-vector times vary
between the two sweeps' runs by up to 40 % in the host's forward and backward (the f32 p = 3
repeat with `--kinds`: 24.8 ms, against 35.4 under `Dense` an hour earlier); the kinds on
the device agree to 0.1 ms.

The M3 Max's GPU (Metal, f32; 2026-10-16, unsandboxed, the quiet gate), with the chunk
budget per vector (below), and in brackets the first sweep, under the one-vector budget:

| problem | m = 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 p = 3 | 19.3 | 18.1 (0.94; first 0.98) | 17.5 (0.91; 1.06) | 19.4 (1.01; 1.40) | 19.6 (1.01; 1.81) | 19.4 (1.00; 2.70) |
| f32 p = 8 | 84.1 | 83.5 (0.99; 1.29) | 84.0 (1.00; 1.87) | 87.6 (1.04; 3.06) | 87.4 (1.04; 5.38) | 87.5 (1.04; 10.00) |

**The chunk budget per vector.** The first sweep found the device path's M2L quadratic in
m: the evaluation per vector grew with m to 10× the one vector's at m = 32 (27.5 s an
evaluation, 18.9 s of it in the downward pass's enqueueing). The grouped GEMM's
reduction (`Accumulate::Rows`) walks every entry of each target's row once per chunk
(`nd_fmm_kernels::translate`), and a level call of m vectors has m times the entries; under
the one-vector chunk budget (128 MB on Metal, 2 GB on CUDA) it also has m times the chunks,
so the walk costs m². T5 now makes the default budget per vector (`device::vector_settings`:
m times the choice's bytes, at most an eighth of the device's memory, never below the
choice's own; a budget the builder names is a total, as before), which keeps the chunks
as many as for one vector; the chunks change no bit (the device checks compare m vectors
with one-vector evaluations that chunk differently), and m = 1 is unchanged. It costs
scratch: m times the one-vector budget (up to 4 GB on the M3 Max and about 12 GB on the
H100 at m = 32). The fix at the root, a reduction whose cost does not grow with the chunks,
is already in T7's brief (`Accumulate::Rows` by chunk); T5 added to it the m-vector case and
the revisiting of the per-vector budget once that lands (docs/phase6/T7-fused-gather.md,
"Revised after T5"). After it the
time per vector is flat in m on Metal: the GPU gains nothing from the vectors at these
sizes, but no longer loses. Where it goes, with the budget per vector (`--kinds`,
`KindTiming::Synchronous`, the breakdown on Metal; ms per vector, in brackets against
m = 1; 17:44–17:47, just after the checks below, load average falling from 20):

| problem | m | M2M + L2L | M2L | P2P | P2M + L2P | forward | backward | evaluation |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| f32 p = 8 | 1 | 4.2 | 71.2 | 5.0 | 1.9 | 1.4 | 4.6 | 91.1 |
| f32 p = 8 | 8 | 1.2 (0.29) | 71.9 (1.01) | 4.8 (0.95) | 1.2 (0.63) | 1.4 (0.95) | 6.3 (1.38) | 87.8 (0.96) |
| f32 p = 8 | 32 | 0.8 (0.19) | 73.1 (1.03) | 4.8 (0.95) | 1.6 (0.84) | 1.5 (1.03) | 6.0 (1.32) | 90.0 (0.99) |
| f32 p = 3 | 1 | 3.7 | 7.0 | 5.1 | 0.9 | 1.6 | 4.8 | 25.1 |
| f32 p = 3 | 8 | 0.9 (0.24) | 5.6 (0.80) | 4.8 (0.93) | 0.5 (0.56) | 1.2 (0.76) | 6.4 (1.33) | 20.3 (0.81) |
| f32 p = 3 | 32 | 0.2 (0.05) | 5.7 (0.82) | 4.8 (0.93) | 0.3 (0.33) | 1.4 (0.88) | 6.2 (1.28) | 19.8 (0.79) |

Dense M2L is 78 % of the evaluation at p = 8 and costs the same per vector at every m: presumably the
hand-written GEMM on Metal reads its table per tile of columns, so more columns per table
buy nothing (not measured further), and its gather and reduction are per column. P2P repeats the pair loop per
vector (0.93–0.95); M2M and L2L, small calls whose launches and per-call costs the vectors
share, fall to 0.05–0.34; the host's output pass (Metal has no f64, so it runs on the host)
costs 1.3× per vector. With `--kinds` the evaluation at p = 3 gains 0.79–0.81 (each call
then waits for its own completion, so the small calls' fixed costs show), without it
0.91–1.01.

## 4. The P2P kernel of several vectors

`P2pKernel::evaluate_many` keeps the single-vector kernel's targets in vector lanes (W = 4
f32 or 2 f64 on NEON, 8 or 4 on AVX2) and runs, per block of K target vectors and per
group of VG charge vectors, the source loop: d, r² and ρ = 1/r once per source and target
vector, then per vector of the group the single-vector kernel's steps (φ_v = fma(q_v, ρ,
φ_v); with ∇φ, t = q_v ρ and its three fnma). Its accumulators (K · VG, times four with
∇φ) must stay in registers, which caps the gain well below the operation count's (T1's
15–20× and 3.5–4×, optimisation.md §4.4). K and VG were chosen by timing a throwaway
driver (64 sources and 64 targets in L1, m = 8 and 32, per vector against m calls of
`evaluate`; the constants' docs in `arch::{neon, avx2}` record every candidate):

| ISA | output | K, VG | f32 | f64 | the runner-up |
| --- | --- | --- | ---: | ---: | --- |
| NEON (M3 Max) | φ | 2, 8 | 4.6× | 4.1× | 1, 8: 4.4×, 4.0× |
| NEON (M3 Max) | φ, ∇φ | 1, 4 | 1.8× | 1.9× | 1, 2: 1.4–1.5× (8 vectors do not fit the 32 registers) |
| AVX2 (Kathleen, job 604422) | φ | 1, 8 | 3.9× | 4.4× | 2, 8: 3.4×, 4.5×, slower in absolute time |
| AVX2 (Kathleen, job 604422) | φ, ∇φ | 1, 4 | 1.5× | 1.9× | 1, 2: 1.4× (16 accumulators spill from 16 registers, less than the shared work saves) |

Inside the FMM, with ∇φ, the P2P kind's time per vector at m ≥ 8 is 0.55–0.58 of m = 1
in f64 (1.7–2.0×), 0.58 in f32 on AVX2 and 0.62–0.70 on NEON (§3.4). Vectors past a multiple of
VG run in smaller groups (8, 4, 2, 1), each vector still bit for bit `evaluate` (tested at
m = 1, 2, 3, 4, 5, 7, 8, 9, 12 and 17 on every ISA, `tests/p2p/many.rs`).

## 5. The exchanges at two Kathleen nodes

80 ranks (2 × 40, one per core; job 604431 on node-c11a-[007,015], the N = 10⁷, m = 16
and 32 launches repeated in job 604432 on the same nodes), the cube at f64 p = 8, `auto`
(Rotation at 40 ranks per node), overlap off. Bytes are the largest rank's per
evaluation: received for the source and multipole exchanges and the replicated top's
all-gather, sent for the reduction (its largest receiver), the charges' forward and the
output's backward; times are each stage's slowest rank over m (ms per vector), and they
include the wait for the slowest rank of the collective or neighbour.

| N | m | ms per vector | sources: msgs, MB | multipoles: msgs, MB | reduction kB | top MB | forward MB | backward MB | sources stage | multipole stage | reduction | top |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10⁶ | 1 | 278.2 | 30, 0.55 | 30, 0.93 | 11.7 | 2.39 | 0.10 | 0.47 | 0.845 | 1.710 | 3.109 | 4.779 |
| 10⁶ | 2 | 267.5 | 30, 0.68 | 30, 1.86 | 23.3 | 4.78 | 0.20 | 0.94 | 0.647 | 1.799 | 2.333 | 4.187 |
| 10⁶ | 4 | 257.6 | 30, 0.96 | 30, 3.71 | 46.7 | 9.56 | 0.41 | 1.88 | 0.515 | 1.794 | 1.907 | 3.806 |
| 10⁶ | 8 | 261.0 | 30, 1.50 | 30, 7.42 | 93.3 | 19.12 | 0.82 | 3.77 | 0.489 | 1.778 | 1.987 | 3.150 |
| 10⁶ | 16 | 252.7 | 30, 2.59 | 30, 14.85 | 186.6 | 38.25 | 1.63 | 7.54 | 0.427 | 1.673 | 2.004 | 3.062 |
| 10⁶ | 32 | 252.4 | 30, 4.78 | 30, 29.69 | 373.2 | 76.50 | 3.26 | 15.07 | 0.456 | 1.653 | 1.907 | 2.980 |
| 10⁷ | 1 | 2,538.2 | 30, 2.44 | 61, 3.84 | 13.6 | 2.30 | 1.01 | 4.38 | 4.791 | 7.932 | 9.657 | 17.214 |
| 10⁷ | 2 | 2,362.0 | 30, 3.05 | 61, 7.69 | 27.2 | 4.61 | 2.01 | 8.75 | 2.652 | 8.399 | 8.649 | 17.230 |
| 10⁷ | 4 | 2,272.7 | 30, 4.27 | 61, 15.38 | 54.4 | 9.22 | 4.03 | 17.50 | 2.128 | 8.050 | 9.144 | 12.834 |
| 10⁷ | 8 | 2,259.4 | 30, 6.70 | 61, 30.76 | 108.9 | 18.44 | 8.05 | 35.00 | 1.787 | 7.843 | 8.637 | 10.760 |
| 10⁷ | 16 | 2,241.0 | 30, 11.58 | 61, 61.51 | 217.7 | 36.88 | 16.11 | 70.01 | 1.386 | 6.678 | 8.010 | 10.356 |
| 10⁷ | 32 | 2,236.9 | 30, 21.33 | 61, 123.03 | 435.5 | 73.76 | 32.22 | 140.01 | 2.348 | 6.811 | 8.093 | 8.183 |

- **The messages do not change with m**: every exchange sends the same messages to the
  same ranks (30 source messages, 30 and 61 multipole messages at the largest rank), and
  the bytes grow as the design said: the sources by (3 + m) / 4 (the coordinates once, m
  charges), the multipoles, the reduction, the replicated top and the output's backward by
  m exactly, the charges' forward by m.
- **The replicated top's all-gather per vector** falls from 4.8 to 3.0 ms at N = 10⁶ (2.4
  to 76.5 MB received) and from 17.2 to 8.2 ms at N = 10⁷: its latency and its wait for
  the slowest rank are shared by the vectors, and its bandwidth term (about 76 MB in
  3 × 32 ms at m = 32, 0.8 GB/s per rank, every rank receiving at once) does not yet
  dominate. Per vector, every exchange costs less at m = 32 than at m = 1; the four
  together are 3.7 % of the evaluation at N = 10⁶ and m = 1 and 2.8 % at m = 32 (1.6 % and
  1.1 % at N = 10⁷).
- At N = 10⁷, m = 32 the first run (job 604431) took 2,917.5 ms per vector: one rank (7,
  on node-c11a-007) ran every kind of the far field 1.39× slower than the others (82.1 s
  against 59–60 s of downward pass; V pairs within 1.025 of the mean), and the others
  waited in the output's backward. The repeat on the same nodes (job 604432) has compute
  imbalance 1.02 and 2,236.9 ms; the table gives the repeat. Memory at N = 10⁷ (the
  largest rank, the harness's model / resident): 54 / 163 MB at m = 1, 263 / 513 MB at
  m = 8, 980 / 1,789 MB at m = 32.

## 6. Against T1's model, and what follows

**The model** (optimisation.md §4.4) put the evaluation per vector at m = 8 at 0.3–0.5 of
one vector's at p = 3 and 0.6–0.8 at p = 8 with `Batched`. Measured with `Batched`, one rank
per core: **0.61–0.64 at p = 3** (f64 and f32, all three machines but locust's f32, 0.71)
and **0.69–0.83 at p = 8** (Kathleen 0.70 and 0.74, locust 0.69 and 0.76, the M3 Max 0.83
and 0.83). p = 8 is within the model, p = 3 above it: the model took P2P's gain from its
operation count (3.5–4× per vector with ∇φ), and the kernel's is 1.7–2.0× (§4), its
accumulators bounded by the registers; P2P is 51–73 % of the evaluation at p = 3 and m = 1. The
leaf operators came in as modelled (about 2×: 1.5–1.9× here). At p = 8 `Batched`'s M2L
gains 0.75–0.82 on Kathleen and locust, and on the M3 Max, where its kernel already ran
near the core's compute bound at m = 1, 0.95.

**What it means for the defaults** (proposals, not made here):

1. **The strategy rule does not see m.** `Auto` resolves `Rotation` at p = 8 with one rank
   per core (the node budget) on all three machines, and `Rotation` shares nothing across
   the vectors: the evaluation 0.89–0.94 per vector at m = 8, so with eight vectors `Batched` is 1.4–4.5×
   faster per vector than the default (§3.4; 1.1–3.3× at m = 1, phase6-m2l.md). T11, which
   refits the cut's constants and revisits the defaults per machine, should take m into the
   host rule (for example `Batched` where m > 1 and p ≤ 12, as T2 measured it winning for
   one vector at p = 6–8 anyway), or the docs should tell users of `vectors` to name it.
2. **Rotation of several vectors** could share its matrices: a pair's rotation, coaxial
   translation and rotation back are the same for every vector, so a pair could apply each to m
   columns at once (still O(p³) per vector, each matrix read once per pair). Not built; a later task if Rotation stays
   the default at high p.
3. **The device gains less than the host**, and on Metal nothing: on one rank the device
   path's host work (the charges' packing and forward, the output's move into
   `OutputMany`) was 60 % of a CUDA evaluation at m = 1 and falls per vector, but the
   device's own kinds share little (P2P repeats the pair loop per vector; the GEMMs' gather
   and reduction per column): dense M2L, most of a device evaluation at p = 8, costs the
   same per vector at every m on both GPUs. Making the device GEMM take more columns per
   table read (one tile over several vectors' columns) is a candidate for T7 (the fused
   gather) or T11.
4. **The cost model weighs one vector's work** (only the partition depends on it): the
   per-kind ratios above (P2P 0.5–0.7, M2L 0.75–1.0 per vector) say the balance between
   leaves and far field moves with m; T11's refit can take m into account if FMMs of many
   vectors are a target.

## 7. The checks

Run for this task on 2026-10-16 (M3 Max unless named; "locust" and "Kathleen" in the
environments of `tools/gh200/` and `tools/kathleen/`):

- `cargo fmt --all -- --check`; `cargo clippy -- -D warnings`; `cargo clippy --examples --
  -D warnings`; `RUST_MIN_STACK=8388608 cargo test`; `cargo doc --no-deps`; `cargo clippy
  --workspace --all-targets -- -D warnings`: clean and passing on the M3 Max; on Kathleen
  (job 604424: fmt, test, doc; clippy failed there on a `type_complexity` lint the tree had
  before the sync, fixed, and the three clippy steps passed in job 604429).
- `RUST_MIN_STACK=8388608 cargo test --workspace`: passes on the M3 Max (350 s, after the
  per-vector chunk budget); on Kathleen (job
  604424) every crate passes, and `spikes/p2p-simd` fails two tests
  (`kernel_rsqrt_within_contract`, `compare_core_on_a_small_cell`), as it did in T2's job
  604317 and every Kathleen check job since 238839 (2026-10-10): the spike's AVX2 rsqrt estimate,
  not touched by T5.
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks (debug, every executable of the
  list and the nd-octree examples): passes on the M3 Max; on locust at 2, 4 and 8; on
  Kathleen at 2, 4, 8 and 80 (job 604425, with rank-count invariance at 80 and the smoke
  run).
- nd-fmm-simd: `cargo clippy -p nd-fmm-simd --all-targets -- -D warnings`, the same with
  `--target x86_64-apple-darwin`, `cargo test -p nd-fmm-simd -- --show-output` and the
  ignored release run: M3 Max, locust and Kathleen (AVX2, job 604424).
- nd-fmm-kernels: `cargo clippy -p nd-fmm-kernels --all-targets --features cpu,metal -- -D
  warnings`, `cargo check -p nd-fmm-kernels --features cuda`, `cargo test -p
  nd-fmm-kernels --features cpu --release -- --show-output` (75 passed), the Metal ignored
  run (55 passed); on locust `--features cpu,cuda -- --include-ignored`; on Kathleen the
  CPU runtime (job 604424).
- nd-fmm-exec: `cargo clippy -p nd-fmm-exec --all-targets --features cpu,metal -- -D
  warnings`, `cargo check -p nd-fmm-exec --features cuda`, `cargo doc -p nd-fmm-exec
  --no-deps --features cpu`; `tests/mpi_exec.rs` in release with `--features cpu` at 1, 2,
  4 and 8 ranks and in debug at 8; `tests/multi_rank.rs` at 1, 2, 4 and 8 ranks;
  `tests/device_ranks.rs` on the CPU runtime at 2 and 4; the Metal ignored run
  (`--features metal --release -- --ignored`, 2,236 s: `device_metal`, the gates on Metal,
  `multi_rank`, the operator tests) and `device_ranks` on Metal at 2 ranks. On locust: `--features cpu,cuda --release` (every test passes but
  `device_tune`'s `tuning_on_the_cpu_runtime`, below), `--features cuda --release --
  --ignored` (3,156 s, every CUDA test, the gates included), `device_ranks` on CUDA at 2
  and 4 ranks, `multi_rank` at 2 (stopped by its 3,600 s limit, above), 4 and 8.
- nd-fmm-validate: `cargo clippy -p nd-fmm-validate --all-targets -- -D warnings`;
  `tests/scaling.rs` at 1 and 2 ranks.

**`device_tune` on locust.** `tuning_on_the_cpu_runtime` fails on locust's CPU runtime
("wide candidate not offered or not skipped": the tuner resolves `Rotation` there, so no
dense M2L decision is made to offer the candidate to); it passes on the M3 Max, and it fails
the same way on locust at main (1038c7f, a separate worktree and target directory, twice),
so it predates T5. Not fixed here; noted for the PR.

## 8. The raw data

Not in the repository (raw output stays out of it): the launches' files of the sweeps
(`tools/scaling/run.sh`, one file per launch with the command, the load, the machine and
every table) on Kathleen in `/scratch/scratch/ucahtbe/fmm/logs/scaling-604430`, `-604431`
and `-604432`, on locust in `/data/ucahtbe/logs/p6t5/` (`t5-host`, `t5-device`,
`t5-device-kinds`, and the repeats `t5-device3`, `t5-device-kinds3`), and the M3 Max's in
the session's scratch directory (copied with this report's tables; ask for them). The
CUDA sweep `t5-device2` and the runs of `t5-device-kinds2` overlapped another user's GPU
job and are not used.

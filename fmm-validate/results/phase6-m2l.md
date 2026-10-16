# Phase 6 T2: the host batched M2L (C6.6)

Phase 6, task T2 ([docs/phase6/T2-host-batched-m2l.md](../../docs/phase6/T2-host-batched-m2l.md));
design [docs/design/optimisation.md](../../docs/design/optimisation.md) §2, signed off on
2026-10-15. Measured on 2026-10-15 and 2026-10-16 from branch `tbetcke/phase6_t2` (main at
623974f plus this task's changes), release builds, rustc 1.99.0, every BLAS thread
variable at 1. The path taken: **with Kathleen** (docs/phase6/README.md, "Kathleen is
usable"): one Kathleen node for the timings, two for the C5.1 gate; no job used more than
two nodes.

## Summary

1. **Bit for bit `Dense`.** `M2lStrategy::Batched` gives `Dense`'s output bit for bit on
   every `tests/mpi_exec.rs` scenario that resolves `Dense`, at 1, 2, 4 and 8 ranks and up
   to 8 threads on the M3 Max and locust (NEON) and at 1–8 and 80 ranks on Kathleen
   (AVX2), and in the C5.1 host gate at every workload point; the column kernel equals
   the matrix–vector loop bit for bit on every ISA for p = 0–20 and 1–64 columns. The
   opt-in fused mode is within 3.2e-16 (f64) and 1.0e-7 (f32) of `Dense` (relative L2;
   bounds 1e-12 and 1e-5), bit for bit across thread counts, and on P ranks bit for bit
   its own one-rank run (§2).
2. **The node's ranks, f64 p = 8** (the cube, N = 10⁶, one rank per core): Batched takes
   423 ms on the M3 Max (12 × 1), 119 ms on locust (72 × 1) and 298 ms on Kathleen
   (40 × 1), against today's best strategy (Rotation) 664, 139 and 615 ms: **1.57×,
   1.17× and 2.06×**; the fused mode 2.03×, 1.33× and 2.38×. On one rank at one thread
   1.47×, 1.40× and 2.37× (fused 2.07×, 1.76×, 2.80×). At f32 p = 8 on one rank at every
   core 2.03×, 1.55× and 1.97× against Dense (§5, §6).
3. **It scales with the ranks per node** (C6.6's second criterion): the downward pass's
   total over the cores grows from one core to the node's ranks by 1.12× on the M3 Max,
   1.32× on locust and 1.68× on Kathleen, where Dense's grows by 4.51×, 7.38× and 3.28×
   (Rotation 1.20×, 1.02×, 1.37×; the cube at f64 p = 8). Its M2L per V pair goes from
   0.51 to 0.57 µs (M3 Max), 0.67 to 0.87 (locust) and 0.66 to 1.12 (Kathleen), Dense's
   from 1.14 to 5.12, 1.18 to 10.5 and 2.91 to 9.42 (§6).
4. **Where it does not win** (C6.6's first criterion, *faster than every existing host
   strategy at p ≥ 6 on one rank*): met at f64 and f32 p = 6 and 8 on all three machines,
   at one thread and at every core; at f64 p = 12 only on Kathleen (1.43×, 1.49×), a tie
   with Rotation on the M3 Max (1.00×, 1.06×) and locust (1.01×, 0.90× at 72 threads),
   and at locust's 72 ranks 0.83×; **at f64 p = 18 Rotation wins everywhere** (Batched
   0.39–0.71× its speed): the dense product is O(p⁴) against Rotation's O(p³), as T1's
   model said (optimisation.md §2.10, the M3 Max at p = 12). At p = 3, where M2L is a
   quarter of an evaluation, 0.99–1.34×.
5. **The kernel** runs at 26 (M3 Max), 20 (locust) and 20 (Kathleen, AVX2) GFLOP/s per
   core in f64 at p = 8 inside the FMM (fused 44, 30, 27), 1,131 GFLOP/s over locust's 72
   threads; its k loops have no call and no bounds check (§3). **B = 64** stays: larger
   blocks are 2–8 % faster with one rank per core and 9–17 % slower with many threads per
   rank (§4).
6. **T1's model holds where its inputs hold**: within 0.02–0.13× at one rank (above it
   but for the M3 Max's f64 cell) and on the Kathleen and locust cubes at f64 p = 8 with
   one rank per core (2.06× against 2.07×, 1.17× against 1.13×). It
   overestimated Kathleen at 40 ranks on the Plummer sphere and at p = 12 (1.43× against
   1.87× and 1.96×): there the work-weighted cut weighs Batched's M2L with `Dense`'s
   constant (3,009 ns against 1,101 ns fitted at f64 p = 8), so the compute imbalance on
   the Plummer sphere is 1.13 against Rotation's 1.04 (a refit for T11), and the AVX2 kernel runs p = 12 at
   4.24 µs per pair against T1's native-build benchmark's 2.75 (§7).

## 1. What T2 adds

- **`nd_fmm_simd::MatrixKernel`** (optimisation.md §2.11): `accumulate_columns` adds
  A · B[:, b_j] into C[:, c_j] for lists of column offsets, A an n × n column-major
  matrix. Four B columns are packed k-major into the caller's buffer (`pack_len(n)` = 4 n
  values); a block of R vectors of rows of their four C columns stays in registers while k
  runs from 0 to n − 1 (R = 4 on NEON, 2 on AVX2, four rows on the scalar path); rows
  left over run as single vectors, then one at a time on the scalar layer. Every element
  receives c ← c + B[k, b_j] · A[r, k] for k ascending: in `MulAdd::Separate` (the
  default) the product and the sum rounded apart, the chain of `MatrixSet::apply`; in
  `MulAdd::Fused` one fma per step. One generic body over the crate's vector layer, one
  `#[target_feature]` entry point per ISA, precision and mode (`arch::{neon, avx2}`;
  `unsafe` only there and in the dispatch, every block with `// SAFETY:`), the ISA from
  `AvailableIsa`, the C columns checked ascending and disjoint, nothing allocated.
- **`M2lStrategy::Batched`** in `nd-fmm-exec`: `Dense`'s tables (`Tables::batched_m2l`),
  and M2L per level call by blocks of `M2L_BLOCK_TARGETS` = 64 consecutive targets
  (rayon `par_chunks_mut` over blocks), for each block the offsets d = 0 … 315 in order,
  the pairs of `batch(d)` whose target lies in the block (two binary searches), one
  kernel call with the table of d, the multipoles and the block's locals read in place.
  Per-thread scratch made at construction: the packing buffer (4 (p + 1)² values) and two
  lists of at most 64 column offsets. M2M, L2L, Classes and Rotation unchanged; `Auto`
  never resolves to it. On the device it runs as `Dense` ("Batched, run as dense on the
  device"); the cost model takes `Dense`'s constants.
- **The fused mode**, opt-in: `FmmBuilder::m2l_mul_add(MulAdd::Fused)`,
  `Fmm::set_m2l_mul_add`, `LaplaceOperator::{with_m2l_mul_add, set_m2l_mul_add}`. The
  column kernel runs on the ISA of the P2P choice (`P2pChoice`; the scalar path for
  `Reference`); `ThreadingReport::m2l_kernel` and `LaplaceOperator::m2l_kernel` report
  ISA and mode ("M2L Batched (named), batched on neon, separate").
- **The harnesses**: `scaling --strategy batched --mul-add separate|fused` (the summary's
  `mul_add`), `nd-fmm-bench --strategy batched`, and in `tools/scaling/run.sh` the sweeps
  `t2-host` and `t2-node` (every launch with `--kinds`, overlap off) and `batched` in the
  `strategy` sweep.

## 2. Correctness

Error measures: exact equality of bit patterns, unless a bound is named.

| check | where | result |
| --- | --- | --- |
| the kernel of every ISA against the matrix–vector loop of `MatrixSet::apply` (restated in `fmm-simd/tests/matrix.rs`), both modes, p = 0–20 (n = 1–441), 1–64 columns at offsets anywhere in B, with repeats and gaps in C (the ignored sweep; the debug run a strided subset) | M3 Max (scalar, NEON), locust (scalar, NEON), Kathleen (scalar, AVX2: Xeon Gold 6248, job 604317) | bit for bit, every cell |
| splitting the columns over calls, one column per call in reverse order | the same | bit for bit |
| the fused mode against the separate one, per element, relative to \|c₀\| + Σ\|terms\| | the same | at most 0.126 n u (f64) and 0.102 n u (f32), bound 2 n u |
| `Batched` against `Dense` on every `tests/mpi_exec.rs` scenario that resolves `Dense`, serially, once per scenario, precision, p and output kind; at the scenario's most threads (2–8) once per scenario and precision | M3 Max at 1, 2, 4, 8 ranks; locust at 1, 2, 4, 8; Kathleen at 1, 2, 4, 8, 80 (job 604318) | bit for bit: 41 evaluations at one rank, 43 per run on 2–8 ranks, 19 at 80 (where `Auto` resolves fewer scenarios to `Dense`) |
| the fused mode on the same scenarios: against `Dense` (relative L2 of φ and ∇φ over every rank), at the most threads against one thread, on several ranks against the one-rank fused `Fmm` over the union | the same | within 3.2e-16 (f64) and 1.0e-7 (f32) of `Dense` (bounds 1e-12, 1e-5); bit for bit across threads and ranks: 34–36 evaluations per run, 13 at 80 |
| `tests/operator/strategy.rs`: `Batched` holds `Dense`'s tables (every pair translation bit for bit), reports its kernel, follows the P2P ISA | M3 Max, locust, Kathleen | passes |
| the C5.1 host gate (`tests/multi_rank.rs`), now with every workload point also under `Batched` and `Batched` fused: N = 10⁵ cube and Plummer sphere, f64 p = 3, 8, 18 and f32 p = 3, 8, eight charge vectors | M3 Max at 1, 2, 4, 8 ranks; locust at 2, 4, 8; Kathleen at 2 × 1 (cube only, the job's time ran out) and 2 × 40 (job 604377) | every run bit for bit the one-rank `Fmm` over the union, within 100 u_T of the one-rank run in point order, errors within 0.1 % (f64) and 1 % (f32) of it; the C3.2 and C3.3 gates pass under all three |
| rank-count invariance (`tests/invariance.rs`, `scaling --invariance` at 80 against 40, 20, 10 ranks) | the MPI list at 2, 4, 8 on the three machines; Kathleen job 604318 | passes (`Auto`; unchanged by T2) |

The errors against the direct sum are those of `Dense` (the bits are), at f64 p = 3, 8
and 18 and f32 p = 3 and 8 (the gate's table, the same φ and ∇φ errors under every
variant to the printed digits). The fused mode's are the same to four digits.

The debug run of `tests/mpi_exec.rs` at one rank on the M3 Max grew from 91 s to 106 s
with the batched checks (one build per check: `Fmm::set_serial` and
`Fmm::set_m2l_mul_add` reuse it).

## 3. The kernel

Inlining check (fmm-simd/CLAUDE.md): every entry point's k loop has no call and no bounds
check. NEON f64 at R = 4: 43 instructions, 16 `fmul.2d` (by element) and 16 `fadd.2d`
per k; AVX2 f64 at R = 2: 27 instructions (2 loads, 4 broadcasts, 8 `vmulpd`, 8
`vaddpd`), 19 fused (8 `vfmadd`); f32 the same shapes.

One table times 64 columns in L1/L2, one M3 Max performance core (a throwaway driver,
`accumulate_columns` repeated for about 1 s; GFLOP/s, 2 n² per column):

| ISA | mode | f64 p = 3 / 6 / 8 / 12 / 18 | f32 p = 3 / 6 / 8 / 12 / 18 |
| --- | --- | --- | --- |
| scalar | separate | 10.5 / 14.6 / 14.7 / 14.3 / 14.3 | 13.4 / 14.6 / 14.8 / 14.6 / 14.7 |
| scalar | fused | 23.8 / 26.9 / 26.8 / 27.4 / 25.8 | 23.9 / 27.4 / 27.7 / 28.5 / 28.6 |
| NEON | separate | 24.0 / 27.2 / 27.9 / 28.1 / 27.0 | 43.2 / 53.0 / 54.7 / 55.3 / 52.3 |
| NEON | fused | 40.0 / 41.5 / 44.4 / 48.0 / 43.0 | 67.0 / 70.7 / 93.5 / 86.5 / 82.5 |

T1's throwaway kernel reached 26 (separate) and 35 (fused) GFLOP/s at f64 p = 8
(optimisation.md Appendix B). The separate form's ceiling on this core is 32 GFLOP/s in
f64 (four pipes of two lanes, a multiply and an add per step) and 65 with fma.

## 4. The block size B

`M2L_BLOCK_TARGETS` sets how many consecutive targets share a table read; the bits do
not depend on it. Measured by a throwaway build that read B from the environment
(not committed), the N = 10⁶ cube, `--kinds`, M2L per V pair of the slowest rank (µs;
with threads the wall time per pair):

| machine | problem | ranks × threads | B = 16 | B = 32 | B = 64 | B = 128 | B = 256 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| M3 Max | cube f32 p = 8 | 12 × 1 | 0.3670 | 0.2992 | 0.2780 | 0.2680 | 0.2639 |
| M3 Max | cube f64 p = 12 | 12 × 1 | 3.4528 | 2.9973 | 2.8178 | 2.6543 | 2.6128 |
| M3 Max | cube f64 p = 3 | 1 × 1 | 0.0412 | 0.0339 | 0.0296 | 0.0276 | 0.0261 |
| M3 Max | cube f64 p = 8 | 1 × 1 | 0.5450 | 0.5219 | 0.5082 | 0.5049 | 0.4988 |
| M3 Max | cube f64 p = 8 | 1 × 12 | 0.0526 | 0.0486 | 0.0473 | 0.0476 | 0.0506 |
| M3 Max | cube f64 p = 8 | 12 × 1 | 0.7973 | 0.6300 | 0.5654 | 0.5386 | 0.5292 |
| M3 Max | cube f64 p = 8 | 2 × 6 | – | 0.1000 | 0.0963 | 0.0953 | 0.0989 |
| M3 Max | cube f64 p = 8 | 4 × 3 | – | 0.2369 | 0.2786 | 0.2694 | 0.2717 |
| M3 Max | plummer f64 p = 8 | 1 × 12 | – | 0.0490 | 0.0478 | 0.0522 | 0.0524 |
| M3 Max | plummer f64 p = 8 | 12 × 1 | – | 0.6389 | 0.6095 | 0.5816 | 0.5677 |
| M3 Max | plummer f64 p = 8 | 2 × 6 | – | 0.1016 | 0.0990 | 0.0998 | 0.1039 |
| locust | cube f32 p = 8 | 72 × 1 | – | 0.5105 | 0.4213 | 0.3993 | 0.3863 |
| locust | cube f64 p = 12 | 72 × 1 | – | 10.9812 | 18.3023 | 9.5521 | 11.8889 |
| locust | cube f64 p = 8 | 1 × 1 | – | 0.6887 | 0.6643 | 0.6493 | 0.6390 |
| locust | cube f64 p = 8 | 1 × 72 | – | 0.0112 | 0.0115 | 0.0135 | 0.0172 |
| locust | cube f64 p = 8 | 72 × 1 | – | 1.1618 | 0.8709 | 0.7992 | 0.7660 |
| locust | cube f64 p = 8 | 8 × 9 | – | 0.0928 | 0.0913 | 0.1016 | 0.1217 |

Ranks per core prefer large blocks (fewer table reads per pair; 128 and 256 are 2–8 %
faster than 64 at one rank per core), threads per rank prefer smaller ones (a level of a
few thousand boxes has only n / B blocks to share out: with 128, 9–17 % slower at 1 × 12
on the Plummer sphere, 8 × 9 and 1 × 72). **B = 64 stays**, the design's proposal, the
balance of the two; T11 may revisit it with the default split per machine. Two rows
are noisy: locust's f64 p = 12 at 72 × 1 (its own previous launches still loaded the
machine: the load average before them rose from 1.5 to 33) and the M3 Max's 4 × 3 (a
split that puts ranks on the efficiency cores).

## 5. One rank: every strategy (t2-host)

N = 10⁶, one rank at one thread and at every core of the machine, `KindTiming::Synchronous`
(each level call timed; its cost is negligible on the host), overlap off; the evaluation's
wall time (ms, the median of three timed evaluations after a warm-up; two at p = 12, one
at p = 18), "Batched / best other" the best of Dense, Classes and Rotation over Batched
(above 1: Batched faster), the M2L kind time per V pair, and the products' achieved rate
2 (p + 1)⁴ flops per pair over that time (all threads together; the harness prints the
time per pair to four decimals of a µs, so at p = 3 with every core the rate is rough,
±20 %).

### M3 Max (12 performance cores; unsandboxed, the harness's quiet gate)

| problem | threads | Dense | Classes | Rotation | Batched | fused | Batched / best other | M2L µs per pair: Dense / Batched / fused | Batched GFLOP/s (fused) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cube f64 p = 3 | 1 | 1,151 | 1,741 | 1,762 | 983 | 928 | 1.17 | 0.058 / 0.029 / 0.021 | 17.5 (24.7) |
| cube f64 p = 3 | 12 | 118 | 167 | 170 | 101 | 98 | 1.17 | 0.005 / 0.003 / 0.002 | 170.7 (232.7) |
| cube f64 p = 6 | 1 | 3,622 | 4,754 | 3,825 | 2,157 | 1,718 | 1.68 | 0.431 / 0.190 / 0.116 | 25.3 (41.4) |
| cube f64 p = 6 | 12 | 358 | 456 | 450 | 210 | 169 | 1.71 | 0.041 / 0.018 / 0.011 | 269.8 (428.8) |
| cube f64 p = 8 | 1 | 8,142 | 9,953 | 6,324 | 4,309 | 3,050 | 1.47 | 1.143 / 0.505 / 0.297 | 26.0 (44.2) |
| cube f64 p = 8 | 12 | 799 | 979 | 599 | 413 | 295 | 1.45 | 0.111 / 0.047 / 0.028 | 279.2 (463.7) |
| cube f64 p = 12 | 1 | 30,796 | 34,716 | 15,288 | 15,240 | 9,968 | 1.00 | 4.763 / 2.186 / 1.310 | 26.1 (43.6) |
| cube f64 p = 12 | 12 | 5,142 | 3,840 | 1,625 | 1,532 | 1,066 | 1.06 | 0.817 / 0.219 / 0.142 | 261.3 (402.6) |
| cube f64 p = 18 | 1 | 135,683 | 142,757 | 43,705 | 80,974 | 55,888 | 0.54 | 21.693 / 12.633 / 8.482 | 20.6 (30.7) |
| cube f64 p = 18 | 12 | 26,860 | 16,204 | 3,938 | 8,719 | 6,092 | 0.45 | 4.366 / 1.374 / 0.934 | 189.6 (279.0) |
| cube f32 p = 3 | 1 | 766 | 1,219 | 1,311 | 571 | 544 | 1.34 | 0.052 / 0.019 / 0.015 | 26.5 (34.1) |
| cube f32 p = 3 | 12 | 77 | 117 | 125 | 61 | 60 | 1.26 | 0.005 / 0.002 / 0.002 | 256.0 (301.2) |
| cube f32 p = 6 | 1 | 2,181 | 3,564 | 3,373 | 1,261 | 1,053 | 1.73 | 0.253 / 0.101 / 0.065 | 47.5 (73.4) |
| cube f32 p = 6 | 12 | 220 | 324 | 308 | 124 | 104 | 1.78 | 0.024 / 0.009 / 0.006 | 516.3 (762.2) |
| cube f32 p = 8 | 1 | 4,800 | 6,788 | 5,872 | 2,432 | 1,822 | 1.97 | 0.650 / 0.258 / 0.158 | 50.8 (83.0) |
| cube f32 p = 8 | 12 | 466 | 649 | 531 | 230 | 176 | 2.03 | 0.062 / 0.024 / 0.015 | 551.3 (874.8) |
| plummer f64 p = 3 | 1 | 2,418 | 3,339 | 3,376 | 2,141 | 2,061 | 1.13 | 0.058 / 0.029 / 0.020 | 17.5 (25.3) |
| plummer f64 p = 3 | 12 | 247 | 354 | 325 | 218 | 213 | 1.13 | 0.006 / 0.003 / 0.002 | 170.7 (243.8) |
| plummer f64 p = 6 | 1 | 8,167 | 9,949 | 8,509 | 5,991 | 5,250 | 1.36 | 0.429 / 0.189 / 0.117 | 25.4 (41.2) |
| plummer f64 p = 6 | 12 | 863 | 994 | 811 | 582 | 538 | 1.39 | 0.047 / 0.018 / 0.011 | 263.8 (425.0) |
| plummer f64 p = 8 | 1 | 17,132 | 19,995 | 14,236 | 11,170 | 9,102 | 1.27 | 1.145 / 0.504 / 0.300 | 26.0 (43.7) |
| plummer f64 p = 8 | 12 | 1,733 | 2,223 | 1,448 | 1,138 | 987 | 1.27 | 0.117 / 0.051 / 0.036 | 258.3 (362.5) |
| plummer f64 p = 12 | 1 | 57,649 | 63,832 | 33,179 | 33,219 | 25,660 | 1.00 | 4.763 / 2.193 / 1.377 | 26.0 (41.5) |
| plummer f64 p = 12 | 12 | 10,829 | 7,852 | 3,689 | 3,442 | 2,633 | 1.07 | 0.989 / 0.231 / 0.146 | 247.7 (390.4) |
| plummer f64 p = 18 | 1 | 235,784 | 247,937 | 90,728 | 150,339 | 111,221 | 0.60 | 21.753 / 12.759 / 8.672 | 20.4 (30.1) |
| plummer f64 p = 18 | 12 | 53,958 | 27,869 | 9,243 | 16,712 | 11,176 | 0.55 | 5.359 / 1.466 / 0.880 | 177.7 (296.2) |
| plummer f32 p = 3 | 1 | 2,053 | 2,776 | 2,921 | 1,755 | 1,702 | 1.17 | 0.051 / 0.019 / 0.015 | 26.9 (34.8) |
| plummer f32 p = 3 | 12 | 206 | 276 | 285 | 185 | 200 | 1.12 | 0.005 / 0.002 / 0.002 | 269.5 (284.4) |
| plummer f32 p = 6 | 1 | 6,151 | 8,328 | 8,056 | 4,696 | 4,396 | 1.31 | 0.253 / 0.101 / 0.066 | 47.6 (72.5) |
| plummer f32 p = 6 | 12 | 641 | 817 | 788 | 465 | 443 | 1.38 | 0.025 / 0.009 / 0.006 | 505.5 (774.5) |
| plummer f32 p = 8 | 1 | 12,071 | 15,224 | 13,811 | 8,322 | 7,451 | 1.45 | 0.650 / 0.257 / 0.160 | 51.1 (81.9) |
| plummer f32 p = 8 | 12 | 1,177 | 1,472 | 1,317 | 829 | 748 | 1.42 | 0.063 / 0.025 / 0.016 | 527.0 (820.1) |

### locust (GH200, 72 Neoverse-V2 cores)

| problem | threads | Dense | Classes | Rotation | Batched | fused | Batched / best other | M2L µs per pair: Dense / Batched / fused | Batched GFLOP/s (fused) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cube f64 p = 3 | 1 | 2,465 | 2,976 | 3,132 | 2,410 | 2,352 | 1.02 | 0.062 / 0.047 / 0.037 | 10.9 (13.7) |
| cube f64 p = 3 | 72 | 75 | 82 | 86 | 74 | 76 | 1.01 | 0.001 / 0.001 / 0.001 | 640.0 (853.3) |
| cube f64 p = 6 | 1 | 5,143 | 6,833 | 5,835 | 3,964 | 3,431 | 1.30 | 0.458 / 0.263 / 0.175 | 18.3 (27.4) |
| cube f64 p = 6 | 72 | 136 | 143 | 125 | 103 | 96 | 1.21 | 0.009 / 0.004 / 0.003 | 1,091.4 (1,549.0) |
| cube f64 p = 8 | 1 | 9,794 | 14,437 | 9,339 | 6,689 | 5,299 | 1.40 | 1.182 / 0.668 / 0.438 | 19.7 (30.0) |
| cube f64 p = 8 | 72 | 244 | 258 | 176 | 153 | 130 | 1.15 | 0.026 / 0.012 / 0.008 | 1,131.2 (1,661.0) |
| cube f64 p = 12 | 1 | 49,810 | 45,427 | 22,163 | 21,992 | 15,982 | 1.01 | 7.649 / 3.045 / 2.050 | 18.8 (27.9) |
| cube f64 p = 12 | 72 | 816 | 843 | 372 | 414 | 317 | 0.90 | 0.118 / 0.052 / 0.036 | 1,100.6 (1,586.7) |
| cube f64 p = 18 | 1 | 303,670 | 170,915 | 61,051 | 155,935 | 131,458 | 0.39 | 49.194 / 24.736 / 20.685 | 10.5 (12.6) |
| cube f64 p = 18 | 72 | 6,701 | 3,609 | 974 | 2,526 | 2,187 | 0.39 | 1.084 / 0.393 / 0.337 | 663.9 (773.6) |
| cube f32 p = 3 | 1 | 1,074 | 1,727 | 1,897 | 1,011 | 982 | 1.06 | 0.045 / 0.032 / 0.027 | 16.1 (18.8) |
| cube f32 p = 3 | 72 | 42 | 51 | 54 | 42 | 43 | 0.99 | 0.001 / 0.001 / 0.001 | 1,024.0 (1,024.0) |
| cube f32 p = 6 | 1 | 2,814 | 4,429 | 4,416 | 1,922 | 1,674 | 1.46 | 0.292 / 0.144 / 0.103 | 33.3 (46.5) |
| cube f32 p = 6 | 72 | 75 | 93 | 89 | 57 | 54 | 1.31 | 0.005 / 0.002 / 0.002 | 2,000.8 (2,667.8) |
| cube f32 p = 8 | 1 | 5,355 | 8,719 | 7,547 | 3,410 | 2,705 | 1.57 | 0.670 / 0.348 / 0.232 | 37.7 (56.7) |
| cube f32 p = 8 | 72 | 130 | 159 | 134 | 84 | 74 | 1.55 | 0.013 / 0.006 / 0.004 | 2,224.1 (3,200.5) |
| plummer f64 p = 3 | 1 | 4,116 | 4,912 | 5,068 | 3,905 | 3,820 | 1.05 | 0.066 / 0.044 / 0.035 | 11.6 (14.6) |
| plummer f64 p = 3 | 72 | 111 | 122 | 124 | 109 | 109 | 1.02 | 0.001 / 0.001 / 0.001 | 640.0 (731.4) |
| plummer f64 p = 6 | 1 | 10,633 | 13,388 | 11,614 | 8,728 | 7,912 | 1.22 | 0.460 / 0.260 / 0.174 | 18.5 (27.6) |
| plummer f64 p = 6 | 72 | 247 | 263 | 228 | 198 | 184 | 1.15 | 0.009 / 0.005 / 0.003 | 980.0 (1,455.2) |
| plummer f64 p = 8 | 1 | 20,109 | 27,745 | 19,332 | 15,222 | 13,058 | 1.27 | 1.179 / 0.665 / 0.439 | 19.7 (29.9) |
| plummer f64 p = 8 | 72 | 448 | 486 | 346 | 321 | 280 | 1.08 | 0.026 / 0.013 / 0.009 | 1,041.4 (1,525.8) |
| plummer f64 p = 12 | 1 | 87,457 | 82,224 | 46,046 | 46,015 | 36,292 | 1.00 | 7.453 / 3.091 / 2.074 | 18.5 (27.5) |
| plummer f64 p = 12 | 72 | 1,450 | 1,514 | 768 | 877 | 703 | 0.88 | 0.117 / 0.057 / 0.039 | 1,007.4 (1,472.2) |
| plummer f64 p = 18 | 1 | 503,034 | 296,469 | 121,936 | 275,408 | 237,498 | 0.44 | 49.081 / 25.159 / 21.170 | 10.4 (12.3) |
| plummer f64 p = 18 | 72 | 11,907 | 6,123 | 1,981 | 4,707 | 4,116 | 0.42 | 1.179 / 0.424 / 0.362 | 615.2 (720.6) |
| plummer f32 p = 3 | 1 | 2,649 | 3,653 | 3,868 | 2,535 | 2,497 | 1.04 | 0.045 / 0.032 / 0.025 | 16.2 (20.6) |
| plummer f32 p = 3 | 72 | 76 | 91 | 96 | 77 | 77 | 0.99 | 0.001 / 0.001 / 0.001 | 853.3 (1,024.0) |
| plummer f32 p = 6 | 1 | 7,798 | 10,373 | 10,348 | 6,352 | 5,981 | 1.23 | 0.291 / 0.143 / 0.102 | 33.7 (47.1) |
| plummer f32 p = 6 | 72 | 171 | 199 | 194 | 145 | 141 | 1.18 | 0.005 / 0.003 / 0.002 | 1,846.9 (2,527.4) |
| plummer f32 p = 8 | 1 | 13,975 | 19,166 | 17,443 | 10,924 | 9,820 | 1.28 | 0.670 / 0.346 / 0.232 | 37.9 (56.6) |
| plummer f32 p = 8 | 72 | 291 | 342 | 302 | 226 | 207 | 1.29 | 0.013 / 0.006 / 0.004 | 2,018.8 (2,982.3) |

### Kathleen (one node, 2 × 20 Xeon Gold 6248; jobs 604320 and 604321, `singlenode`)

| problem | threads | Dense | Classes | Rotation | Batched | fused | Batched / best other | M2L µs per pair: Dense / Batched / fused | Batched GFLOP/s (fused) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cube f64 p = 3 | 1 | 2,439 | 3,668 | 4,015 | 2,243 | 2,209 | 1.09 | 0.077 / 0.045 / 0.039 | 11.3 (13.0) |
| cube f64 p = 3 | 40 | 141 | 201 | 214 | 141 | 131 | 1.00 | 0.004 / 0.003 / 0.002 | 189.6 (284.4) |
| cube f64 p = 6 | 1 | 8,898 | 10,006 | 9,552 | 3,912 | 3,520 | 2.27 | 1.060 / 0.235 / 0.171 | 20.4 (28.0) |
| cube f64 p = 6 | 40 | 364 | 435 | 425 | 218 | 190 | 1.67 | 0.036 / 0.011 / 0.008 | 425.0 (600.2) |
| cube f64 p = 8 | 1 | 20,596 | 21,243 | 16,736 | 7,062 | 5,971 | 2.37 | 2.902 / 0.659 / 0.479 | 19.9 (27.4) |
| cube f64 p = 8 | 40 | 759 | 853 | 674 | 337 | 291 | 2.00 | 0.098 / 0.028 / 0.021 | 465.3 (630.9) |
| cube f64 p = 12 | 1 | 123,326 | 89,846 | 44,193 | 30,818 | 23,633 | 1.43 | 19.555 / 4.238 / 3.049 | 13.5 (18.7) |
| cube f64 p = 12 | 40 | 5,190 | 3,115 | 1,641 | 1,100 | 827 | 1.49 | 0.821 / 0.141 / 0.098 | 404.0 (585.3) |
| cube f64 p = 18 | 1 | 574,178 | 440,893 | 130,451 | 182,545 | 137,368 | 0.71 | 93.092 / 28.247 / 20.763 | 9.2 (12.6) |
| cube f64 p = 18 | 40 | 33,798 | 14,075 | 4,488 | 6,453 | 4,461 | 0.70 | 5.523 / 0.996 / 0.664 | 261.8 (392.3) |
| cube f32 p = 3 | 1 | 1,244 | 2,309 | 3,040 | 1,161 | 1,136 | 1.07 | 0.045 / 0.032 / 0.029 | 16.2 (18.0) |
| cube f32 p = 3 | 40 | 81 | 125 | 161 | 79 | 79 | 1.02 | 0.002 / 0.002 / 0.002 | 320.0 (320.0) |
| cube f32 p = 6 | 1 | 4,725 | 6,906 | 8,050 | 2,278 | 2,076 | 2.07 | 0.539 / 0.135 / 0.101 | 35.5 (47.5) |
| cube f32 p = 6 | 40 | 210 | 299 | 347 | 130 | 125 | 1.61 | 0.020 / 0.006 / 0.005 | 738.8 (980.0) |
| cube f32 p = 8 | 1 | 10,810 | 14,043 | 14,395 | 4,026 | 3,469 | 2.68 | 1.459 / 0.335 / 0.242 | 39.2 (54.1) |
| cube f32 p = 8 | 40 | 398 | 572 | 557 | 202 | 179 | 1.97 | 0.048 / 0.015 / 0.011 | 892.7 (1,226.4) |
| plummer f64 p = 3 | 1 | 5,724 | 7,631 | 8,203 | 5,396 | 5,325 | 1.06 | 0.079 / 0.044 / 0.037 | 11.7 (13.8) |
| plummer f64 p = 3 | 40 | 316 | 384 | 403 | 306 | 306 | 1.03 | 0.004 / 0.003 / 0.002 | 189.6 (256.0) |
| plummer f64 p = 6 | 1 | 20,677 | 22,545 | 21,793 | 12,877 | 12,269 | 1.61 | 1.057 / 0.235 / 0.172 | 20.5 (27.9) |
| plummer f64 p = 6 | 40 | 785 | 891 | 904 | 600 | 577 | 1.31 | 0.033 / 0.011 / 0.009 | 425.0 (545.7) |
| plummer f64 p = 8 | 1 | 43,755 | 44,837 | 37,759 | 22,436 | 21,054 | 1.68 | 2.894 / 0.657 / 0.513 | 20.0 (25.6) |
| plummer f64 p = 8 | 40 | 1,555 | 1,714 | 1,449 | 931 | 875 | 1.56 | 0.090 / 0.025 / 0.019 | 516.6 (690.6) |
| plummer f64 p = 12 | 1 | 210,158 | 165,593 | 93,710 | 73,972 | 62,054 | 1.27 | 18.674 / 4.364 / 3.115 | 13.1 (18.3) |
| plummer f64 p = 12 | 40 | 8,830 | 5,662 | 3,415 | 2,660 | 2,297 | 1.28 | 0.784 / 0.138 / 0.098 | 415.1 (584.1) |
| plummer f64 p = 18 | 1 | 886,037 | 726,975 | 261,292 | 340,603 | 271,809 | 0.77 | 85.259 / 27.938 / 20.713 | 9.3 (12.6) |
| plummer f64 p = 18 | 40 | 61,236 | 24,703 | 9,322 | 12,273 | 9,576 | 0.76 | 6.118 / 0.975 / 0.691 | 267.3 (377.5) |
| plummer f32 p = 3 | 1 | 4,388 | 5,988 | 7,217 | 4,245 | 4,214 | 1.03 | 0.045 / 0.031 / 0.028 | 16.4 (18.5) |
| plummer f32 p = 3 | 40 | 252 | 314 | 356 | 227 | 252 | 1.11 | 0.002 / 0.002 / 0.002 | 301.2 (301.2) |
| plummer f32 p = 6 | 1 | 14,663 | 18,140 | 19,941 | 10,805 | 10,616 | 1.36 | 0.537 / 0.135 / 0.102 | 35.6 (47.3) |
| plummer f32 p = 6 | 40 | 612 | 747 | 820 | 527 | 524 | 1.16 | 0.018 / 0.006 / 0.005 | 738.8 (980.0) |
| plummer f32 p = 8 | 1 | 29,092 | 34,265 | 34,739 | 18,434 | 17,592 | 1.58 | 1.455 / 0.335 / 0.243 | 39.2 (54.1) |
| plummer f32 p = 8 | 40 | 1,137 | 1,337 | 1,346 | 830 | 738 | 1.37 | 0.049 / 0.016 / 0.010 | 830.5 (1,366.9) |

Notes:

- At p ≥ 6 the gain follows the share of M2L: on the cube it is most of an evaluation,
  on the Plummer sphere M2P (in the leaves) is about half of Batched's time at f64 p = 8
  (its slowest rank 445 of the 958 ms at Kathleen's 40 × 1), which T12 (C6.9) addresses.
- At p = 3 Batched is 0.99–1.34× Dense, the best other strategy there; with every core
  of locust or Kathleen the two are within 2 %.
- The fused mode's gain over the separate one is 1.005–1.54× of the evaluation at p ≥ 6
  (never slower; up to 1.7× of the M2L), largest on the M3 Max, whose four FP pipes
  make an fma as cheap as a multiply. At p = 3 it lies between 0.90× and 1.02×: noise on
  evaluations of 40–250 ms at every core, where M2L is a quarter.
- With every thread of one rank (the threads share one copy of the tables) Dense's M2L
  per pair is 2.2–3.5× Batched's at f64 p = 8. At p = 18 Dense takes 1.7–5.2× and
  Classes 1.1–2.4× Batched's time, Rotation 0.38–0.71× of it.

## 6. Many ranks per node (t2-node)

N = 10⁶, f64 p = 8 (the cube and the Plummer sphere) and p = 12 (the cube), every split of
ranks × threads on one node, `--kinds`, overlap off, the median of three timed
evaluations. "downward × cores / 1 core" is the downward pass's total over the cores
against one rank at one thread (Phase 5 T10's "Stages against ranks": the mean over the
ranks times the ranks, with threads the slowest rank's time times its threads; 1.00 is
perfect scaling). The criterion of C6.6: *scaling with ranks per node where Dense does
not*.

### M3 Max

| problem | ranks × threads | total ms: Dense / Classes / Rotation / **Batched** / fused | Batched over best other | M2L µs per pair: Dense / Rotation / Batched / fused | downward × cores / 1 core: Dense / Rotation / Batched |
| --- | --- | --- | ---: | --- | --- |
| cube f64 p = 8 | 1 × 1 | 8,164 / 9,986 / 6,329 / **4,319** / 3,063 | 1.47 | 1.143 / 0.845 / 0.506 / 0.298 | 1.00 / 1.00 / 1.00 |
| cube f64 p = 8 | 2 × 1 | 4,195 / 5,147 / 3,235 / **2,212** / 1,589 | 1.46 | 1.176 / 0.861 / 0.518 / 0.313 | 1.03 / 1.02 / 1.02 |
| cube f64 p = 8 | 4 × 1 | 2,265 / 2,660 / 1,659 / **1,155** / 847 | 1.44 | 1.281 / 0.883 / 0.543 / 0.339 | 1.11 / 1.04 / 1.07 |
| cube f64 p = 8 | 8 × 1 | 2,601 / 1,528 / 935 / **636** / 499 | 1.47 | 3.213 / 0.982 / 0.584 / 0.397 | 2.77 / 1.14 / 1.12 |
| cube f64 p = 8 | 12 × 1 | 2,767 / 1,075 / 664 / **423** / 327 | 1.57 | 5.124 / 1.019 / 0.567 / 0.374 | 4.51 / 1.20 / 1.12 |
| cube f64 p = 8 | 1 × 12 | 790 / 997 / 611 / **417** / 316 | 1.47 | 0.109 / 0.080 / 0.047 / 0.031 | 1.15 / 1.14 / 1.13 |
| plummer f64 p = 8 | 1 × 1 | 17,128 / 19,971 / 14,252 / **11,078** / 9,257 | 1.29 | 1.143 / 0.844 / 0.506 / 0.302 | 1.00 / 1.00 / 1.00 |
| plummer f64 p = 8 | 2 × 1 | 8,871 / 10,282 / 7,314 / **5,682** / 4,770 | 1.29 | 1.178 / 0.864 / 0.519 / 0.316 | 1.03 / 1.02 / 1.02 |
| plummer f64 p = 8 | 4 × 1 | 4,636 / 5,457 / 3,815 / **2,966** / 2,498 | 1.29 | 1.215 / 0.883 / 0.540 / 0.332 | 1.06 / 1.05 / 1.06 |
| plummer f64 p = 8 | 8 × 1 | 4,387 / 2,969 / 2,066 / **1,621** / 1,381 | 1.27 | 2.794 / 0.945 / 0.555 / 0.367 | 2.36 / 1.11 / 1.09 |
| plummer f64 p = 8 | 12 × 1 | 4,631 / 2,249 / 1,381 / **1,102** / 979 | 1.25 | 4.683 / 0.899 / 0.562 / 0.371 | 3.88 / 1.07 / 1.10 |
| plummer f64 p = 8 | 1 × 12 | 1,770 / 2,127 / 1,428 / **1,103** / 947 | 1.29 | 0.119 / 0.084 / 0.050 / 0.034 | 1.27 / 1.22 / 1.21 |
| cube f64 p = 12 | 1 × 1 | 30,992 / 34,700 / 15,287 / **15,250** / 9,932 | 1.00 | 4.782 / 2.226 / 2.188 / 1.306 | 1.00 / 1.00 / 1.00 |
| cube f64 p = 12 | 2 × 1 | 16,147 / 17,695 / 7,791 / **7,908** / 5,240 | 0.99 | 4.999 / 2.265 / 2.269 / 1.387 | 1.05 / 1.02 / 1.04 |
| cube f64 p = 12 | 4 × 1 | 11,124 / 9,227 / 3,995 / **4,082** / 2,731 | 0.98 | 7.006 / 2.320 / 2.345 / 1.450 | 1.46 / 1.04 / 1.07 |
| cube f64 p = 12 | 8 × 1 | 11,392 / 5,236 / 2,101 / **2,119** / 1,529 | 0.99 | 14.717 / 2.422 / 2.420 / 1.628 | 3.06 / 1.07 / 1.09 |
| cube f64 p = 12 | 12 × 1 | 11,424 / 4,589 / 1,604 / **1,573** / 1,147 | 1.02 | 21.877 / 2.716 / 2.625 / 1.807 | 4.61 / 1.21 / 1.20 |
| cube f64 p = 12 | 1 × 12 | 5,454 / 3,761 / 1,553 / **1,445** / 1,045 | 1.07 | 0.869 / 0.226 / 0.206 / 0.139 | 2.17 / 1.22 / 1.13 |

### locust

| problem | ranks × threads | total ms: Dense / Classes / Rotation / **Batched** / fused | Batched over best other | M2L µs per pair: Dense / Rotation / Batched / fused | downward × cores / 1 core: Dense / Rotation / Batched |
| --- | --- | --- | ---: | --- | --- |
| cube f64 p = 8 | 1 × 1 | 9,800 / 14,509 / 9,302 / **6,682** / 5,302 | 1.39 | 1.182 / 1.102 / 0.666 / 0.438 | 1.00 / 1.00 / 1.00 |
| cube f64 p = 8 | 8 × 1 | 2,038 / 1,823 / 1,166 / **877** / 707 | 1.33 | 2.249 / 1.103 / 0.710 / 0.486 | 1.88 / 0.98 / 1.05 |
| cube f64 p = 8 | 16 × 1 | 1,299 / 927 / 581 / **451** / 371 | 1.29 | 2.958 / 1.084 / 0.725 / 0.510 | 2.48 / 0.98 / 1.09 |
| cube f64 p = 8 | 32 × 1 | 942 / 480 / 300 / **239** / 199 | 1.25 | 4.325 / 1.087 / 0.758 / 0.545 | 3.68 / 0.98 / 1.14 |
| cube f64 p = 8 | 64 × 1 | 948 / 265 / 159 / **133** / 115 | 1.19 | 9.167 / 1.103 / 0.829 / 0.644 | 7.02 / 1.00 / 1.25 |
| cube f64 p = 8 | 72 × 1 | 948 / 233 / 139 / **119** / 105 | 1.17 | 10.501 / 1.139 / 0.871 / 0.734 | 7.38 / 1.02 / 1.32 |
| cube f64 p = 8 | 1 × 72 | 243 / 259 / 173 / **152** / 130 | 1.14 | 0.026 / 0.016 / 0.011 / 0.008 | 1.59 / 1.03 / 1.26 |
| cube f64 p = 8 | 8 × 9 | 243 / 228 / 144 / **116** / 96 | 1.24 | 0.260 / 0.130 / 0.091 / 0.065 | 1.98 / 1.06 / 1.23 |
| plummer f64 p = 8 | 1 × 1 | 19,876 / 27,580 / 19,387 / **15,223** / 13,060 | 1.27 | 1.158 / 1.108 / 0.664 / 0.438 | 1.00 / 1.00 / 1.00 |
| plummer f64 p = 8 | 8 × 1 | 3,744 / 3,516 / 2,479 / **1,988** / 1,710 | 1.25 | 2.087 / 1.073 / 0.685 / 0.468 | 1.72 / 0.98 / 1.03 |
| plummer f64 p = 8 | 16 × 1 | 2,304 / 1,881 / 1,237 / **1,017** / 890 | 1.22 | 2.532 / 1.075 / 0.714 / 0.501 | 2.16 / 0.98 / 1.07 |
| plummer f64 p = 8 | 32 × 1 | 1,583 / 943 / 632 / **542** / 478 | 1.17 | 3.728 / 1.097 / 0.758 / 0.546 | 3.11 / 0.98 / 1.12 |
| plummer f64 p = 8 | 64 × 1 | 1,504 / 518 / 330 / **294** / 264 | 1.12 | 11.830 / 1.106 / 0.821 / 0.624 | 5.92 / 1.00 / 1.22 |
| plummer f64 p = 8 | 72 × 1 | 1,518 / 484 / 298 / **264** / 242 | 1.13 | 8.157 / 1.127 / 0.838 / 0.636 | 6.19 / 1.02 / 1.27 |
| plummer f64 p = 8 | 1 × 72 | 453 / 489 / 352 / **320** / 281 | 1.10 | 0.026 / 0.016 / 0.013 / 0.009 | 1.63 / 1.13 / 1.42 |
| plummer f64 p = 8 | 8 × 9 | 445 / 452 / 327 / **294** / 257 | 1.11 | 0.226 / 0.130 / 0.097 / 0.065 | 1.82 / 1.21 / 1.53 |
| cube f64 p = 12 | 1 × 1 | 49,207 / 46,306 / 22,169 / **22,037** / 15,966 | 1.01 | 7.551 / 3.106 / 3.052 / 2.046 | 1.00 / 1.00 / 1.00 |
| cube f64 p = 12 | 8 × 1 | 10,008 / 5,957 / 2,736 / **3,350** / 2,589 | 0.82 | 12.649 / 3.066 / 3.846 / 2.837 | 1.66 / 0.98 / 1.25 |
| cube f64 p = 12 | 16 × 1 | 6,140 / 3,518 / 1,399 / **1,777** / 1,399 | 0.79 | 15.411 / 3.087 / 4.053 / 3.037 | 2.04 / 0.99 / 1.30 |
| cube f64 p = 12 | 32 × 1 | 4,371 / 2,263 / 746 / **954** / 765 | 0.78 | 21.729 / 3.217 / 4.257 / 3.283 | 2.90 / 1.03 / 1.36 |
| cube f64 p = 12 | 64 × 1 | 4,276 / 1,832 / 454 / **538** / 442 | 0.84 | 43.550 / 3.789 / 4.648 / 3.674 | 5.42 / 1.23 / 1.49 |
| cube f64 p = 12 | 72 × 1 | 4,319 / 1,830 / 405 / **487** / 409 | 0.83 | 50.961 / 4.013 / 5.032 / 4.092 | 6.05 / 1.29 / 1.57 |
| cube f64 p = 12 | 1 × 72 | 838 / 860 / 380 / **418** / 322 | 0.91 | 0.121 / 0.047 / 0.053 / 0.036 | 1.16 / 1.09 / 1.25 |
| cube f64 p = 12 | 8 × 9 | 2,245 / 817 / 347 / **425** / 322 | 0.82 | 2.886 / 0.377 / 0.472 / 0.325 | 3.43 / 1.09 / 1.39 |

### Kathleen (job 604322, node-c11b-008)

| problem | ranks × threads | total ms: Dense / Classes / Rotation / **Batched** / fused | Batched over best other | M2L µs per pair: Dense / Rotation / Batched / fused | downward × cores / 1 core: Dense / Rotation / Batched |
| --- | --- | --- | ---: | --- | --- |
| cube f64 p = 8 | 1 × 1 | 20,648 / 21,259 / 16,720 / **7,035** / 5,969 | 2.38 | 2.909 / 2.258 / 0.656 / 0.479 | 1.00 / 1.00 / 1.00 |
| cube f64 p = 8 | 10 × 1 | 3,295 / 2,337 / 1,848 / **840** / 727 | 2.20 | 4.854 / 2.474 / 0.810 / 0.621 | 1.61 / 1.09 / 1.23 |
| cube f64 p = 8 | 20 × 1 | 1,951 / 1,258 / 969 / **447** / 387 | 2.17 | 5.669 / 2.524 / 0.844 / 0.653 | 1.95 / 1.12 / 1.29 |
| cube f64 p = 8 | 40 × 1 | 1,635 / 842 / 615 / **298** / 258 | 2.06 | 9.422 / 3.086 / 1.116 / 0.869 | 3.28 / 1.37 / 1.68 |
| cube f64 p = 8 | 1 × 40 | 735 / 851 / 663 / **335** / 286 | 1.98 | 0.094 / 0.083 / 0.028 / 0.020 | 1.30 / 1.48 / 1.73 |
| cube f64 p = 8 | 2 × 20 | 676 / 777 / 671 / **281** / 243 | 2.39 | 0.176 / 0.176 / 0.047 / 0.035 | 1.21 / 1.56 / 1.45 |
| plummer f64 p = 8 | 1 × 1 | 44,482 / 44,904 / 37,718 / **22,501** / 20,804 | 1.68 | 2.972 / 2.266 / 0.660 / 0.485 | 1.00 / 1.00 / 1.00 |
| plummer f64 p = 8 | 10 × 1 | 6,602 / 5,276 / 4,367 / **2,647** / 2,459 | 1.65 | 4.480 / 2.458 / 0.786 / 0.585 | 1.47 / 1.09 / 1.17 |
| plummer f64 p = 8 | 20 × 1 | 3,797 / 2,711 / 2,217 / **1,440** / 1,359 | 1.54 | 5.391 / 2.535 / 0.853 / 0.657 | 1.78 / 1.12 / 1.24 |
| plummer f64 p = 8 | 40 × 1 | 2,985 / 1,807 / 1,372 / **958** / 914 | 1.43 | 7.873 / 3.091 / 1.069 / 0.827 | 2.84 / 1.38 / 1.59 |
| plummer f64 p = 8 | 1 × 40 | 1,587 / 1,760 / 1,447 / **1,008** / 912 | 1.44 | 0.092 / 0.076 / 0.028 / 0.020 | 1.35 / 1.48 / 1.87 |
| plummer f64 p = 8 | 2 × 20 | 1,584 / 1,790 / 1,443 / **976** / 902 | 1.48 | 0.180 / 0.156 / 0.054 / 0.039 | 1.35 / 1.49 / 1.87 |
| cube f64 p = 12 | 1 × 1 | 122,524 / 89,752 / 43,960 / **30,840** / 23,646 | 1.43 | 19.422 / 6.500 / 4.243 / 3.050 | 1.00 / 1.00 / 1.00 |
| cube f64 p = 12 | 10 × 1 | 15,597 / 10,236 / 4,840 / **3,495** / 2,701 | 1.39 | 24.746 / 7.096 / 4.797 / 3.483 | 1.25 / 1.08 / 1.11 |
| cube f64 p = 12 | 20 × 1 | 8,857 / 6,463 / 2,558 / **1,922** / 1,462 | 1.33 | 27.380 / 7.298 / 5.294 / 3.770 | 1.41 / 1.12 / 1.17 |
| cube f64 p = 12 | 40 × 1 | 7,047 / 4,853 / 1,688 / **1,182** / 930 | 1.43 | 42.799 / 9.435 / 6.207 / 4.579 | 2.22 / 1.45 / 1.43 |
| cube f64 p = 12 | 1 × 40 | 5,007 / 3,059 / 1,670 / **995** / 865 | 1.68 | 0.792 / 0.239 / 0.127 / 0.103 | 1.63 / 1.47 / 1.21 |
| cube f64 p = 12 | 2 × 20 | 4,486 / 3,077 / 1,528 / **935** / 713 | 1.63 | 1.413 / 0.439 / 0.238 / 0.167 | 1.45 / 1.35 / 1.13 |

Notes:

- With one rank per core Dense's M2L per pair grows 4.5×, 8.9× and 3.2× from one rank to
  the node's ranks (f64 p = 8), Batched's 1.1×, 1.3× and 1.7×: each table is read once per
  block of 64 targets, and the node's 12 to 72 copies of 17.4 MB no longer stream once per
  target. Kathleen's 1.7× includes its all-core clock, which costs every strategy about
  1.4× from 20 to 40 ranks (phase5n-kathleen.md §1).
- At f64 p = 12 on locust the tables are 76 MB per rank, 5.5 GB at 72 ranks against 114
  MB of L3: Batched's M2L per pair grows from 3.05 to 5.03 µs (Rotation's from 3.11 to
  4.01), and Rotation is 1.2× faster there; the fused mode ties it (0.99×). On the M3 Max
  and Kathleen Batched stays ahead at p = 12 (1.02× and 1.43×; fused 1.40× and 1.81×).
- With threads (1 × n, 2 × 20, 8 × 9) Batched is 1.10–2.4× the best other strategy at
  f64 p = 8; at locust's 8 × 9, today's best split there (threading.rs), 116 ms against
  Rotation's 144 (1.24×).
- The Plummer sphere gains less (1.13–1.43× at the node's ranks), since M2P is half of
  its evaluation under Batched (§5, notes).

## 7. Against T1's model

T1 modelled each evaluation with the best strategy's M2L replaced by the time per pair of
its level benchmark (optimisation.md §2.10, Appendix C). Measured (the same problems and
splits, N = 10⁶, today's best strategy measured in the same sweep):

| machine | ranks × threads | problem | best today, measured (ms) | Batched (ms): measured × / *model* × | fused (ms): measured × / *model* × |
| --- | --- | --- | --- | --- | --- |
| M3 Max | 1 × 1 | cube f64 p = 8 | Rotation 6,324 | 4,309 (1.47×) / *1.50×* | 3,050 (2.07×) / *1.85×* |
| M3 Max | 1 × 1 | cube f32 p = 8 | Dense 4,800 | 2,432 (1.97×) / *1.84×* | 1,822 (2.64×) / *2.02×* |
| M3 Max | 12 × 1 | cube f64 p = 8 | Rotation 664 | 423 (1.57×) / *1.30×* | 327 (2.03×) / *–* |
| M3 Max | 12 × 1 | cube f64 p = 12 | Rotation 1,604 | 1,573 (1.02×) / *0.96×* | 1,147 (1.40×) / *–* |
| M3 Max | 12 × 1 | plummer f64 p = 8 | Rotation 1,381 | 1,102 (1.25×) / *1.20×* | 979 (1.41×) / *–* |
| locust | 1 × 1 | cube f64 p = 8 | Rotation 9,339 | 6,689 (1.40×) / *1.38×* | 5,299 (1.76×) / *1.62×* |
| locust | 72 × 1 | cube f64 p = 8 | Rotation 139 | 119 (1.17×) / *1.13×* | 105 (1.33×) / *–* |
| locust | 72 × 1 | cube f64 p = 12 | Rotation 405 | 487 (0.83×) / *1.18×* | 409 (0.99×) / *–* |
| locust | 72 × 1 | plummer f64 p = 8 | Rotation 298 | 264 (1.13×) / *1.13×* | 242 (1.23×) / *–* |
| Kathleen | 1 × 1 | cube f64 p = 8 | Rotation 16,736 | 7,062 (2.37×) / *2.28×* | 5,971 (2.80×) / *2.70×* |
| Kathleen | 40 × 1 | cube f64 p = 8 | Rotation 615 | 298 (2.06×) / *2.07×* | 258 (2.38×) / *2.42×* |
| Kathleen | 40 × 1 | cube f64 p = 12 | Rotation 1,688 | 1,182 (1.43×) / *1.96×* | 930 (1.81×) / *2.40×* |
| Kathleen | 40 × 1 | plummer f64 p = 8 | Rotation 1,372 | 958 (1.43×) / *1.87×* | 914 (1.50×) / *2.11×* |

- At one rank the model is within 0.02–0.13× for Batched, the measurement above it but
  for the M3 Max's f64 cell (1.47× against 1.50×); the fused mode is 0.1–0.6× better than
  modelled (its kernel runs at 44 against T1's 35 GFLOP/s on the M3 Max, §3).
- The M3 Max at 12 ranks is better than modelled (1.57× against 1.30×): its Rotation
  baseline is the same, Batched's M2L per pair 0.57 µs against T1's 0.66.
- Kathleen at 40 ranks matches on the cube at p = 8 (2.06× against 2.07×; M2L per pair
  1.116 µs against T1's 1.13) and falls short at p = 12 and on the Plummer sphere:
  - **the cut**: under `Batched` the work-weighted cut (Phase 5S T7, on by default) uses
    `CostModel::calibrated`'s `Dense` constants, M2L 3,009 ns per pair at f64 p = 8,
    while Batched's fitted constant is 1,101 ns (the same run, `--kinds`). The cut so
    gives a rank too few leaves for its V work: compute max over mean 1.127 against
    Rotation's 1.036 on the Plummer sphere at 40 × 1 (1.081 against 1.065 on the cube).
    A cut with Batched's constants would bring the Plummer cell to about 1.6× (the
    compute mean over the max, *model*). The constants are refitted with the new
    operators in T11 (README, "The work weight of the cut"); the fitted values are in
    every launch's summary (`alpha_ns`);
  - **the kernel at p = 12**: 4.24 µs per pair at one process on Kathleen against T1's
    2.75 (its `-C target-cpu=native` build, in which the compiler may use any feature of
    the Xeon Gold 6248, AVX-512 included), while at p = 6 and 8 it is faster than T1's
    (0.235 against 0.274, 0.659 against 0.71). Not investigated in T2; T6 adds the AVX-512
    body of the kernel and measures it.
- locust at 72 ranks matches at p = 8 (1.17× against 1.13×) and falls short at p = 12
  (0.83× against 1.18×): T1 modelled it from 72 processes of its level benchmark, each
  with one level's working set; the FMM's ranks also hold the tree and the other stages,
  and at 76 MB of tables per rank the L3 no longer holds a block's tables (§6, notes).

## 8. Scratch per thread and memory

Per thread: the packing buffer, 4 (p + 1)² values (2.6 kB at f64 p = 8, 11.6 kB at
p = 18), and two lists of 64 `usize`, 1 kB; the rest of the operator's scratch is
unchanged. Per rank the tables are `Dense`'s, 332 (p + 1)⁴ values (17.4 MB at f64 p = 8,
76 MB at p = 12), every rank its own: at 40 ranks per Kathleen node 697 MB at p = 8 (the
summary's `dense_node_mb`), which the batched layout reads once per block of 64 targets.

## 9. For T11

- **The strategy rule.** Batched was never slower than Dense by more than 1.1 % in any
  cell (§5, §6), so it can take Dense's place wherever rule B picks Dense, bit for bit
  (f64 p ≤ 6, f32 p ≤ 8 under the default budget). It is the fastest strategy at f64
  p = 6–8 on every machine and split measured, and at p = 12 on Kathleen; Rotation stays
  the choice at p = 18, and at p = 12 on locust's 72 ranks; at p = 12 elsewhere they tie.
  Its tables are Dense's, 17.4 MB per rank at f64 p = 8 and 76 MB at p = 12: a rule
  that picks Batched for its speed has to keep the node's memory in mind, not its cache.
- **The fused mode** is 1.0–1.5× faster again at p ≥ 6 (never slower), within 3.2e-16 (f64) and
  1.0e-7 (f32) of Dense, deterministic and rank-count invariant: the candidate default
  (a one-rank output change, requirement 6).
- **The cost model**: `CostModel::calibrated` maps `Batched` to `Dense`'s constants
  (T2 kept it so, optimisation.md §2.9); with them the cut misbalances Batched by up to
  9 % on Kathleen (§7). Refit M2L for Batched (and M2P and P2P where T12 and T6 change
  them).
- **The block size** (§4): 64 is the balance; with one rank per core 128–256 would gain
  2–8 % on the M2L.

## Appendix: runs

| machine | runs | when | load |
| --- | --- | --- | --- |
| M3 Max (12 P + 4 E cores, 64 GB) | `OUT=bench-results/t2-m3max tools/scaling/run.sh t2-host t2-node` (250 launches), unsandboxed; the block-size sweeps by hand (§4) | 2026-10-15 21:13–21:35 (block sizes) and 21:36–23:51 | the harness's quiet gate (other processes below 200 % CPU before each launch) |
| locust (GH200, 72 Neoverse-V2 cores) | the same sweeps from `/data/ucahtbe/fmm/tbetcke/phase6_t2` (280 launches: `t2-node` has 8 splits there); the block sizes from a copy with the throwaway patch | 2026-10-15 23:27 – 2026-10-16 02:14 (block sizes 23:23–23:27) | no other user's process above 1 % CPU before any launch but one (a root `find` at 70 % of one core before the cube f64 p = 3 Dense 1 × 1 cell); every launch's file records it |
| Kathleen (Xeon Gold 6248, 2 × 20 cores) | `sweep.sbatch t2-host` with `ONLY=t2-host-cube` (job 604320, node-c11b-008) and `ONLY=t2-host-plummer` (604321, node-c11a-025); `sweep.sbatch t2-node` (604322, node-c11b-008); `singlenode`, exclusive, `--exclude=node-c11a-011` | 2026-10-15 21:36 – 2026-10-16 00:43 | exclusive nodes |

The checks: on the M3 Max `cargo test -p nd-fmm-simd` (and `--release -- --ignored`),
the root and strict checks, the other-architecture clippy, `cargo test -p nd-fmm-exec`
(and `--release -- --ignored`, and `--features cpu --release`),
`.github/scripts/run-mpi-tests.sh` at 2, 4 and 8 ranks, `tests/multi_rank.rs` at 1, 2,
4 and 8 ranks in release; on locust the nd-fmm-simd checks, `cargo test -p nd-fmm-exec`,
the MPI list at 2, 4 and 8 ranks and the gate at 2, 4 and 8; on Kathleen `check.sbatch`
(job 604317: every group; the strict step's only failures are the two
`spikes/p2p-simd` tests known to fail on Intel AVX2, outside the workspace's members),
`correctness.sbatch` at 2 nodes (604318: the MPI list at 2, 4, 8 and 80 ranks, the
invariance at 80 against 40, 20 and 10, the smoke run) and the C5.1 gate at 2 nodes
(604319: 2 × 1 and 2 × 40, cut short by the step's own timeout after the cube's 15 runs
at each, every row exact; 604377, `small`: 2 × 40 in full).

Kathleen node-hours: 604317 0.42, 604318 0.93, 604319 1.70, 604377 1.97, 604320 1.51,
604321 2.74, 604322 0.88: **10.2** in all. Raw output stays outside the repository
(`bench-results/`, git-ignored; locust `/data/ucahtbe/logs/p6t2-*`; Kathleen
`$FMM_KATHLEEN_ROOT/logs/scaling-60432{0,1,2}`, `fmm-*-6043{17,18,19,77}.out`).

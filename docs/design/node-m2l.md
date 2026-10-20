# M2L on a node: design for Phase 5N T4 (C5N.3)

> **Superseded on the host by Phase 6 T11** (2026-10-20, decision 7 of
> docs/phase6/README.md; `fmm-validate/results/phase6.md`): with the batched M2L (C6.6,
> each table read once per block of targets) and the plane-wave M2L (C6.5, no dense
> tables), `Auto` resolves by p and the precision alone (`Batched` at f64 p ≤ 6 and f32
> p ≤ 7, `PlaneWave` above); the ranks per node and the node budget of this document's rule
> are agreed and reported but no longer decide. The measurements below stand for `Dense`
> and `Rotation`, which remain selectable.

As of 2026-10-11 (Section 10, the outcome of T4, added). Written in Phase 5N T3 ([docs/phase5n/T3-m2l-node-
study.md](../phase5n/T3-m2l-node-study.md)); signed off by the user on 2026-10-11, every
recommendation as proposed, with the budget set per machine class and the far-field build
for the CPU in Phase 6 (Section 9; docs/phase5n/README.md, decisions 5–8). It
proposes the rule `M2lStrategy::Auto` should follow on a node, how the build learns the
node, how tables are loaded when many ranks build at once, what to do about one copy of
the tables per rank, and the ranks × threads guidance. Every number is measured on the
machine named, release, every BLAS variable at 1, unless marked *model*; the raw output
stays outside the repository (root logs on Kathleen and locust, `bench-results/` on the M3
Max), and Section 2 names the jobs and commands.

Machines: the Apple M3 Max (12 performance cores in two clusters of 6, 16 MB L2 per
cluster; the system-level cache is not reported by `sysctl`), locust (GH200: 72
Neoverse-V2 cores, 1 MB L2 each, one 114 MB L3), and one Kathleen node (2 × Xeon Gold
6248: 20 cores, 1 MB L2 each, and 27.5 MB L3 per socket); `lscpu -C` and `sysctl hw`.

## 1. Starting point

**The rule.** `M2lStrategy::resolve` gives Dense for p ≤ 8 (`AUTO_DENSE_MAX_P`) and
Rotation above. Its source is Phase 2 T8 (laplace-fmm-plan §7): one thread, one rank, one
product or one rotation per pair, dense faster than table rotation up to p = 8 (ratio 0.93
at p = 8, 1.22 at p = 10). The device path has its own rule (`tune`) and is not touched.

**The evidence that it is wrong on a node.** Phase 5 T10 (phase5-m3max.md,
phase5-gh200.md, "The M2L strategy at p = 8") and Phase 5N T2 (phase5n-kathleen.md §1–3):
at p = 8 every rank holds its own dense tables (316 M2L and 16 octant matrices, 17.4 MB in
f64), and the downward pass stops scaling: its total over the ranks grows 2.4–4.4× by
8–12 ranks on the M3 Max, 5.8–6.9× by 72 on locust and 3.0–3.2× by 40 on Kathleen. Rotation
keeps 0.60–0.79 efficiency at the largest count, threads (which share one copy) beat ranks,
and Rotation was faster than Dense even on one rank (0.72–0.96× its time). No counters were
read; the cause was inferred.

**What one copy per rank costs, as measured here** (Section 2.1): at f64 p = 8 on the
cube, Dense's M2L costs per V pair 1.15 → 4.93 µs from 1 to 12 ranks (M3 Max), 1.18 →
9.63 µs from 1 to 72 (locust) and 2.92 → 8.83 µs from 1 to 40 (Kathleen); Rotation's 0.87
→ 0.92, 1.09 → 1.12 and 2.22 → 3.01 µs. At the largest one-thread count `Auto` (Dense) is
4.49× (M3 Max), 6.30× (locust) and 2.59× (Kathleen) the best strategy's time.

## 2. Measurements

### 2.1 The grid

N = 10⁶, the cube and the Plummer sphere, the three strategies, `--kinds`
(`KindTiming::Synchronous`), overlap off, five evaluations after a warm-up (three on one
rank), one thread per rank unless a split is named, the ranks spread over both sockets on
Kathleen (`mpirun --map-by ppr:<k>:socket:PE=<t> --bind-to core`), bound per core on
locust, unbound on the M3 Max. Commands: `tools/scaling/run.sh`'s `launch` driven by a
throwaway script, one `scaling` launch per cell, for example

```sh
mpirun --report-bindings --map-by ppr:10:socket:PE=1 --bind-to core -n 20 \
  target/release/examples/scaling --threads 1 --dist cube --n 1000000 --precision f64 \
  --p 8 --strategy dense --overlap off --kinds --evals 5
```

Where: the M3 Max on 2026-10-10, 21:39–23:22 (unsandboxed, the quiet gate of `run.sh`; one
cell repeated, below); locust 2026-10-10 21:37 to 2026-10-11 00:10, idle before and after
every launch (no other user's process above 5% CPU in any load record, the GPU idle);
Kathleen jobs 239031 (p = 8, node-c11a-058), 239038 (the other degrees, node-c11a-014) and
239059 (the crossover, node-c11b-047), `singlenode`, exclusive.

**f64 p = 8, wall time per evaluation (ms), the best strategy and the others over it.**
The headline: today's `Auto` is Dense.

| machine | ranks × threads | cube: best (ms) | Dense / best | Classes / best | Plummer: best (ms) | Dense / best | Classes / best |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| M3 Max | 1 × 1 | Rotation 6,460 | 1.27 | 1.56 | Rotation 14,471 | 1.18 | 1.39 |
| M3 Max | 2 × 1 | Rotation 3,317 | 1.28 | 1.55 | Rotation 7,415 | 1.19 | 1.39 |
| M3 Max | 4 × 1 | Rotation 1,703 | 1.33 | 1.54 | Rotation 3,852 | 1.20 | 1.39 |
| M3 Max | 8 × 1 | Rotation 869 | 2.99 | 1.64 | Rotation 2,016 | 2.17 | 1.50 |
| M3 Max | 12 × 1 | Rotation 613 | 4.49 | 1.67 | Rotation 1,473 | 3.12 | 1.42 |
| M3 Max | 1 × 12 | Rotation 607 | 1.55 | 1.67 | Rotation 1,448 | 1.24 | 1.48 |
| M3 Max | 2 × 6 | Rotation 618 | 1.99 | 1.75 | Rotation 1,475 | 1.54 | 1.48 |
| locust | 1 × 1 | Rotation 9,173 | 1.07 | 1.59 | Rotation 19,111 | 1.03 | 1.41 |
| locust | 8 × 1 | Rotation 1,157 | 1.76 | 1.59 | Rotation 2,483 | 1.50 | 1.44 |
| locust | 16 × 1 | Rotation 603 | 2.28 | 1.58 | Rotation 1,263 | 1.92 | 1.44 |
| locust | 32 × 1 | Rotation 316 | 3.30 | 1.61 | Rotation 678 | 2.68 | 1.46 |
| locust | 64 × 1 | Rotation 178 | 5.58 | 1.63 | Rotation 403 | 3.99 | 1.53 |
| locust | 72 × 1 | Rotation 160 | 6.30 | 1.66 | Rotation 346 | 4.70 | 1.51 |
| locust | 1 × 72 | Rotation 177 | 1.37 | 1.49 | Rotation 348 | 1.29 | 1.38 |
| locust | 4 × 18 | Rotation 153 | 1.48 | 1.54 | Rotation 334 | 1.27 | 1.39 |
| locust | 8 × 9 | Rotation 143 | 1.72 | 1.60 | Rotation 327 | 1.39 | 1.40 |
| Kathleen | 1 × 1 | Rotation 16,463 | 1.26 | 1.28 | Rotation 37,263 | 1.18 | 1.19 |
| Kathleen | 2 × 1 | Rotation 8,271 | 1.31 | 1.27 | Rotation 18,801 | 1.19 | 1.19 |
| Kathleen | 10 × 1 | Rotation 1,856 | 1.77 | 1.26 | Rotation 4,330 | 1.52 | 1.19 |
| Kathleen | 20 × 1 | Rotation 987 | 2.01 | 1.28 | Rotation 2,314 | 1.65 | 1.20 |
| Kathleen | 40 × 1 | Rotation 638 | 2.59 | 1.34 | Rotation 1,519 | 2.05 | 1.25 |
| Kathleen | 1 × 40 | Rotation 649 | 1.13 | 1.23 | Rotation 1,440 | 1.10 | 1.15 |
| Kathleen | 2 × 20 | Rotation 587 | 1.17 | 1.29 | Rotation 1,431 | 1.10 | 1.23 |
| Kathleen | 4 × 10 | Rotation 573 | 1.49 | 1.31 | Rotation 1,435 | 1.22 | 1.20 |

**Rotation is the fastest strategy at f64 p = 8 in all 48 cells**, one rank and every
split included. The best configuration per machine: Rotation 1 × 12 (M3 Max, within 1% of
12 × 1), 8 × 9 (locust, within 12% of 72 × 1) and 4 × 10 (Kathleen; 2 × 20 within 0.5%).
Kathleen repeats T2 (other nodes) to within 2–4% (Dense) and 5–8% (Rotation).

**The other degrees and f32 p = 8**: the three strategies at one rank, the largest
one-thread count and the best split; then Dense against Rotation at the counts in between
where Dense wins on one rank (the *crossover* runs, f32 p = 8 and f64 p = 6). Wall-time ratio
over the best, cube / Plummer:

| case | machine | 1 × 1: Dense, Rotation | largest × 1: Dense, Rotation | best split: Dense, Rotation | where Rotation overtakes Dense |
| --- | --- | --- | --- | --- | --- |
| f64 p = 3 | M3 Max | 1.00/1.00, 1.53/1.41 | 1.00/1.00, 1.43/1.33 | 1.00/1.00, 1.43/1.36 | never (Dense best everywhere) |
| f64 p = 3 | locust | 1.00/1.00, 1.24/1.25 | 1.00/1.00, 1.24/1.15 | 1.00/1.00, 1.20/1.16 | never |
| f64 p = 3 | Kathleen | 1.00/1.00, 1.60/1.40 | 1.00/1.00, 1.60/1.23 | 1.00/1.00, 1.59/1.30 | never |
| f64 p = 6 | M3 Max | 1.00/1.00, 1.09/1.05 | 2.45/1.76, 1.00/1.00 | 1.06/1.08, 1.00/1.00 | between 6 (38 MB) and 8 ranks (51 MB) |
| f64 p = 6 | locust | 1.00/1.00, 1.13/1.10 | 2.81/2.21, 1.00/1.00 | 1.11/1.05, 1.00/1.00 | between 8 (51 MB) and 16 (102 MB) |
| f64 p = 6 | Kathleen | 1.00/1.00, 1.05/1.04 | 1.62/1.36, 1.00/1.00 | 1.00/1.00, 1.09/1.02 | at 10 (64 MB; 1.01/1.00) |
| f32 p = 8 | M3 Max | 1.00/1.00, 1.24/1.15 | 2.29/1.68, 1.00/1.00 | 1.00/1.03, 1.10/1.00 | between 6 (52 MB) and 8 (70 MB) |
| f32 p = 8 | locust | 1.00/1.00, 1.43/1.28 | 3.01/2.26, 1.00/1.00 | 1.00/1.00, 1.04/1.04 | between 8 (70 MB) and 16 (139 MB) |
| f32 p = 8 | Kathleen | 1.00/1.00, 1.32/1.18 | 1.44/1.20, 1.00/1.00 | 1.00/1.00, 1.25/1.16 | between 10 (87 MB) and 20 (174 MB) |
| f64 p = 10 | all three | Rotation, Dense 1.21–1.82 | Rotation, Dense 2.69–9.07 | Rotation, Dense 1.87–4.27 | — (Rotation best everywhere) |
| f64 p = 12 | all three | Rotation, Dense 1.70–2.73 | Rotation, Dense 3.29–10.19 | Rotation, Dense 2.62–7.35 | — |

The MB are the private dense tables of the node, ranks × 332 (p + 1)⁴ × the value's bytes
(6.4 MB per rank at f64 p = 6, 8.7 at f32 p = 8). Classes is never the best strategy. One
cell was repeated: the M3 Max's cube f64 p = 6, Rotation, 12 × 1, whose first run (743 ms)
had one rank at 2.2× the mean downward time (the only multi-rank run of the M3 Max above
1.15); the repeat took 375 ms, max/mean 1.05, and replaces it.

**Per V pair** (the M2L kind's time of the slowest rank over its V pairs), cube, f64 p = 8,
one thread per rank, with the private dense tables per node:

| machine | ranks: dense MB per node → Dense / Classes / Rotation, µs per pair |
| --- | --- |
| M3 Max | 1: 17 → 1.15 / 1.46 / 0.87; 2: 35 → 1.19 / 1.49 / 0.89; 4: 70 → 1.28 / 1.52 / 0.91; 8: 139 → 3.19 / 1.63 / 0.92; 12: 209 → 4.93 / 1.68 / 0.92 |
| locust | 1: 17 → 1.18 / 1.98 / 1.09; 8: 139 → 2.25 / 2.00 / 1.10; 16: 278 → 3.03 / 1.97 / 1.10; 32: 557 → 4.55 / 2.01 / 1.10; 72: 1,253 → 9.63 / 2.17 / 1.12 |
| Kathleen | 1: 17 → 2.92 / 2.98 / 2.22; 2: 35 → 3.07 / 2.97 / 2.22; 10: 174 → 4.67 / 3.16 / 2.40; 20: 348 → 5.53 / 3.33 / 2.47; 40: 696 → 8.83 / 4.27 / 3.01 |

M2M and L2L (the kind timings) are under 1.3% of the downward stage in every cell (cube,
f64 p = 8, one rank: M2M 30–72 ms and L2L 30–74 ms against M2L 5,249–17,983 ms); the
downward stage is M2L. With threads the M2L time per pair times the threads (core-µs) is
Dense 1.58, 2.36 and 4.94 at 1 × 12, 8 × 9 and 4 × 10 against 1.15, 1.18 and 2.92 on one
rank; Rotation 0.96, 1.15 and 2.97 against 0.87, 1.09 and 2.22.

**Errors** (8 charge vectors, 1,000 targets, at the largest one-thread count; the
one-rank Dense `Fmm` as reference for Classes and Rotation, `--reference-strategy dense`):
the three strategies give the same errors against the direct sum to the four printed
digits in f64 at every p (cube φ L2 2.913e-3, 1.257e-4, 1.889e-5, 3.415e-6, 6.906e-7 at
p = 3, 6, 8, 10, 12, identical on all three machines), and differ from Dense by 2.8e-15 to
9.6e-15 (φ) and at most 3.6e-16 (∇φ) relative L2. In f32 p = 8 they differ by 2.7–3.1e-6
(φ) and 1.3–1.6e-7 (∇φ); Dense's φ error is 1% larger (1.910e-5 against 1.891e-5 on the
cube, 2.773e-5 against 2.754e-5 on the Plummer sphere, on every machine): the f32 dense
matrices are rounded once each, the rotation factors are smaller numbers.

### 2.2 Table sizes against the caches

Bytes a rank reads in the M2L level calls (the `tables` example's memory table, every
family of the strategy):

| p | Dense f64 (f32) | Classes f64 | Rotation f64 |
| ---: | ---: | ---: | ---: |
| 3 | 0.68 MB (0.34) | 0.13 MB | 0.07 MB |
| 6 | 6.4 MB (3.2) | 0.96 MB | 0.39 MB |
| 8 | 17.4 MB (8.7) | 2.4 MB | 0.83 MB |
| 10 | 38.9 MB (19.4) | 5.1 MB | 1.5 MB |
| 12 | 75.8 MB (37.9) | 9.5 MB | 2.5 MB |

(Dense: 316 M2L and 2 × 8 octant matrices of (p + 1)⁴ values; Classes: 16 class matrices,
the 2 × 48 transforms and the octant matrices; Rotation: `ShiftTables::storage_len` of
its three families.)

| machine | per core | shared | ranks sharing it | Dense f64 p = 8 copies that fit |
| --- | --- | --- | --- | ---: |
| M3 Max | L1d 128 kB | L2 16 MB per cluster of 6 cores (the system-level cache unreported) | 6 | 0 |
| locust | L1d 64 kB, L2 1 MB | L3 114 MB | 72 | 6 |
| Kathleen | L1d 32 kB, L2 1 MB | L3 27.5 MB per socket | 20 | 1 per socket |

Rotation's 0.83 MB fits in every core's L2 (1 MB) and in the M3 Max's L2 share; Dense's
17.4 MB fits none of the private levels.

### 2.3 Isolating the cause

One rank of Dense on core 0 (the N = 10⁶ cube, f64 p = 8, three evaluations), alone and
while other cores run (a) the Dense evaluation (an `mpirun` of N = 10⁷ on those cores,
evaluating until killed), (b) the Rotation evaluation, (c) a copy load without tables (a C
`memcpy` loop over two 64 MiB arrays per core); each process pinned with `taskset`. The
measured rank's M2L time per V pair:

| background | Kathleen (job 239032, node-c11a-093) | locust (2026-10-10, 21:51–22:11) |
| --- | --- | --- |
| alone | 2.91 µs (repeat 2.90) | 1.16 µs (repeat 1.16) |
| Dense on 3 / 4 cores of the same socket | 4.37 (4 cores; 1.50×) | 1.36 (3 cores; 1.18×) |
| Dense on 7 / 9 cores | 5.52 (9; 1.90×) | 1.87 (7; 1.61×) |
| Dense on 15 cores | | 2.47 (2.13×) |
| Dense on 19 / 31 cores | 8.78 (19, the rest of the socket; 3.02×) | 3.67 (31; 3.16×) |
| Dense on the other socket only (20 cores) | 2.92 (1.00×) | |
| Dense on every other core | 8.07 (39; 2.78×) | 5.21 (71; 4.50×) |
| Rotation on the rest of the socket / node | 3.74 (19; 1.29×) | 1.34 (71; 1.16×); 1.17 (7) |
| copy load on the rest of the socket / node | 16.99 (19; 5.84×) | 8.36 (71; 7.21×); 1.32 (7; 1.14×) |

- **The cost is the socket's memory path, not the clock.** Dense neighbours on the *other*
  socket leave the rank untouched (2.92 against 2.91 µs), the same count on its own socket
  triples it. Rotation neighbours, which read 0.83 MB tables, cost 1.16–1.29×.
- **Capacity and bandwidth both matter.** A pure bandwidth load costs most (5.8–7.2× at a
  full socket): each dense product reads its 52 kB matrix, so once the copies leave the
  shared cache every pair streams from memory at the share of the bandwidth that is left.
  Before saturation the copies cost more than the same number of copy loads (locust, 7
  neighbours: Dense 1.61×, copy 1.14×): the tables are evicted and re-read.
- **Counters** (`perf stat` on locust, user mode: `instructions`, `cycles`,
  `l2d_cache_refill`, `ll_cache_miss_rd`, `mem_access`; Kathleen has no counter tool, no
  `perf` and no module): the instructions are the same in every case (6.10e11) and the
  cycles grow with the slowdown (1.32e11 alone, 4.43e11 with 71 Dense neighbours, 6.87e11
  with 71 copy loads, 1.40e11 with 71 Rotation neighbours). `ll_cache_miss_rd` stays at
  1.91–1.95e10 and `l2d_cache_refill` at 5.5–6.9e7 in every case, so these core events
  do not see the traffic beyond L2 (*inferred*: hardware prefetches are not counted as
  refills, and the 114 MB L3 belongs to the interconnect, whose own PMU was not
  available to the user). The last-level misses of (a) and (b) are therefore **not
  measured**; the cycles at fixed instructions are.

### 2.4 Tables at build and at scale

**Build time per strategy** (the `tables` example, one thread, `--ps 3,6,8,10,12 --parts
build,cache`; Dense = M2M + L2L + dense M2L, Rotation = all three families):

| p | M3 Max: Dense / Classes / Rotation | locust | Kathleen |
| ---: | --- | --- | --- |
| 3 | 3.7 / 0.6 / 0.3 ms | 2.4 / 0.4 / 0.3 ms | 7.1 / 1.4 / 0.6 ms |
| 6 | 63 / 5.3 / 0.7 ms | 49 / 4.3 / 0.8 ms | 141 / 11 / 1.7 ms |
| 8 | 218 / 19 / 1.6 ms | 210 / 18 / 1.6 ms | 591 / 45 / 3.6 ms |
| 10 | 659 / 56 / 2.3 ms | 690 / 53 / 2.9 ms | 1,896 / 143 / 6.6 ms |
| 12 | 1,734 / 132 / 3.7 ms | 1,821 / 136 / 5.1 ms | 5,098 / 383 / 11.5 ms |

**Their share of `build` at the largest one-thread count** (the grid's build parts, cube,
max over ranks; the part includes step 7's all-reduce wait): Dense 58% (M3 Max, 12 ranks),
82% (locust, 72) and 87% (Kathleen, 40) at f64 p = 8, 81–95% at p = 10 and 92–98% at p = 12;
Classes 11–34% at p = 8; Rotation 2–5% at p = 8 and 4–12% at p = 12. With every rank
building at once the Dense build slows on Kathleen (450 → 578 ms at 40 ranks, 3.87 →
5.14 s at p = 12) and not on locust (210 → 211 ms) or the M3 Max (236 → 248 ms).

**Loading from `TableCache`** (the new `node_tables` example: every rank at once after a
barrier; f64; max over the ranks, ms):

| | M3 Max, local SSD | locust, `/data` | Kathleen, Lustre |
| --- | --- | --- | --- |
| ranks | 1 / 12 | 1 / 8 / 72 | 1 / 40 / 80 (2 nodes) |
| Dense p = 8: every rank builds (no cache) | 236 / 248 | 210 / 211 / 211 | 451 / 578 / 607 |
| Dense p = 8: every rank stores at once (empty directory) | 274 / 429 | 233 / 274 / 663 | 588 / 2,791 / 829 |
| Dense p = 8: one rank per node first, the others wait, then load | 274 / 345 | 234 / 252 / 256 | 588 / 839 / 812 |
| Dense p = 8: load, files present | 21 / 24 | 18 / 19 / 19 | 24 / 30 / 30 |
| Dense p = 8: first load on nodes that did not write the files | | | – / – / 119 |
| Dense p = 12: build / node first / load | 1,865 / 2,199 / 97 | 1,824 / 2,017 / 84 | 3,871 / 5,560 / 127 (40); 489 first load at 80 |
| Rotation p = 8: build / load | 1.3 / 1.4 | 1.6 / 1.0 | 3.6 / 2.5 (40); 10.5 first load at 80 |

- Warm loads run at 0.7–1.0 GB/s for files above 1 MB, bound by decoding (`tables`
  example, "load rate"); a load is 11–19× faster than building Dense at p = 8 and 19–37× at
  p = 12.
- Storing from every rank at once is the worst choice on Lustre: 40 ranks writing the same
  16.6 MB file took 2.8 s once (a Lustre outlier: the 80-rank and the one-node-first runs
  took 0.8 s, and one single-rank store of the class tables 2.7 s), and every rank pays the
  build anyway.
- **Concurrent stores succeed.** In every run (1–80 ranks; two processes on one node; two
  processes on two Kathleen nodes, which may share a process id) every rank's outcome was
  `Built` or `Loaded`, never `Rebuilt` or `NotStored`; no temporary file was left; every
  rank's tables of the first pass equal, bit for bit, those it loads afterwards.

### 2.5 The node context

`split_shared` of the world and one all-reduce of the ranks per node (min and max),
each after a barrier, timed per rank, max over the ranks (`node_tables`):

| machine | ranks | first `split_shared` | later ones (median of 20) | the all-reduce |
| --- | ---: | ---: | ---: | ---: |
| M3 Max (Open MPI 5.0.10, TCP loopback) | 1 / 12 | 0.01 / 3.4–6.4 ms | 0.002 / 0.9–1.1 ms | 0.25–0.29 ms |
| locust (Open MPI 5.0.10) | 1 / 8 / 72 | 0.02 / 0.07 / 0.18–0.26 ms | 0.03–0.04 ms (72) | 0.006 ms |
| Kathleen (Open MPI 4.1.6, PSM2) | 1 / 40 / 80 | 0.04 / 1.3–1.4 / 1.5–1.6 ms | 0.06–0.08 ms | 0.013–0.019 ms |

On locust and Kathleen at most 1.6 ms, under 1% of a build at these counts (Rotation's
46–122 ms at the largest count); on the M3 Max, whose Open MPI goes through TCP loopback,
up to 6.4 ms, 4% of its 168-ms Rotation build at 12 ranks.

### 2.6 One copy per node: a spike with `MPI_Win_allocate_shared`

With the user's permission (2026-10-10) to call MPI through `mpi-sys` where rsmpi has no
safe API, a throwaway spike (not committed) measured what one copy per node, or per
socket, would give: 316 random matrices of the dense M2L tables' size applied with the loop
of `MatrixSet::apply`, 189 pairs per target, each with another matrix, 2,000 targets per
rank; the tables private (a `Vec` per rank), in one `MPI_Win_allocate_shared` window per
`split_shared` node, or one per socket. M2L µs per pair at the largest count (max over the
ranks), and the estimate for the FMM: its one-rank Dense time scaled by the spike's ratio:

| machine, ranks | case | spike private 1 → largest | spike shared, node / socket | FMM: Dense with a shared copy (*estimate*) | FMM: Rotation, measured |
| --- | --- | --- | --- | ---: | ---: |
| M3 Max, 12 | f64 p = 8 | 0.99 → 5.74 | 1.38 / 1.43 | 1.59 | 0.92–0.98 |
| | f32 p = 8 | 0.54 → 2.95 | 0.74 / 0.73 | 0.87 | 0.88 |
| | f64 p = 6 | 0.39 → 2.19 | 0.50 / 0.50 | 0.56 | 0.51 |
| locust, 72 | f64 p = 8 | 1.29 → 12.11 | 2.10 / 2.07 | 1.87–1.90 | 1.12 |
| | f32 p = 8 | 0.70 → 5.77 | 1.07 / 1.07 | 1.01 | 1.04 |
| | f64 p = 6 | 0.53 → 4.06 | 0.79 / 0.80 | 0.68 | 0.57–0.58 |
| Kathleen, 40 (job 239062) | f64 p = 8 | 3.00 → 10.18 | 4.30 / 4.25 | 4.11–4.13 | 3.00–3.01 |
| | f32 p = 8 | 1.49 → 5.21 | 2.09 / 2.35 | 2.05 | 2.72–2.74 |
| | f64 p = 6 | 1.09 → 3.62 | 1.52 / 1.50 | 1.45–1.46 | 1.53–1.54 |

A shared copy cuts Dense's cost at the largest count 2.4–5.8× against private copies, but
at f64 p = 8 it stays 1.4–1.7× Rotation's; it beats Rotation only on Kathleen at f32 p = 8
(0.75×) and f64 p = 6 (0.95×), and ties on the M3 Max and locust at f32 p = 8. Threads give
the same sharing now: Dense at 4 × 10 on Kathleen costs 2.03 (f32 p = 8) and 1.41 (f64 p = 6)
core-µs per pair, the window's estimates 2.05 and 1.46. A window per socket gave nothing
over one per node (the first touch puts a node window on socket 0; the spike's Kathleen
node and socket rows agree within 12%).

### 2.7 Node-hours and runs

Kathleen (`sacct -X`, nodes × elapsed): 239031 (the p = 8 grid, 35 min), 239032 (the
isolating experiment, `tables` and `node_tables` on one node, 33 min), 239038 (the other
degrees, 2 h 22 min), 239041 (`node_tables` on 2 nodes, `test`, 1.5 min), 239059 (the
crossover, 18 min), 239060 (a duplicate of it, cancelled after 1 min), 239061 (the spike's
build failed offline, 9 s) and 239062 (the spike, 5 min): **3.94 node-hours** of the 50 that
decision 3 gives T3; every job at most 2 nodes. locust: 2026-10-10 21:37 to 2026-10-11
00:10, the load checked before and after every grid launch; one crossover sweep overlapped
a second copy of itself and a root `find` scan and was discarded and rerun alone. The M3
Max: 21:39–23:22, unsandboxed.

## 3. The model

*Model.* Let D(p, T) = 332 (p + 1)⁴ · size_of(T) be the dense tables of one rank, r the ranks
of a node and C the node's shared cache. A dense M2L product reads one 81 × 81 matrix
(52 kB in f64 at p = 8) per V pair, a different one for each pair of a target, so the
tables are streamed per pair from the nearest level that holds them:

- while r · D fits the shared cache, the per-pair cost stays near the one-rank cost (it
  grows with the shared level's bandwidth per rank: locust's isolated rank 1.18× with four
  copies, 70 MB, in a 114 MB L3);
- once r · D exceeds it, the tables stream from memory and the per-pair cost grows with r,
  bounded by the socket's memory bandwidth per rank (Kathleen 2.9 → 8.8 µs per pair at 20
  per socket; locust 1.16 → 5.21 µs at 72);
- Rotation (0.83 MB) and Classes (2.4 MB) stay in each core's L2 and do not grow, apart from
  the all-core clock (Kathleen's Rotation 2.22 → 3.01 µs from 20 to 40 ranks, every stage
  1.4×, T2 §1).

Measured crossover, the private dense bytes per node where Rotation overtakes Dense (the
crossover runs, Section 2.1):

| machine | shared cache per node | f64 p = 6 (6.4 MB per rank) | f32 p = 8 (8.7 MB per rank) | f64 p = 8 (17.4 MB) |
| --- | ---: | --- | --- | --- |
| M3 Max | 2 × 16 MB L2, plus the system cache | 38–51 MB (6–8 ranks) | 52–70 MB (6–8) | below 17 MB: Rotation wins on one rank |
| locust | 114 MB | 51–102 MB (8–16) | 70–139 MB (8–16) | below 17 MB |
| Kathleen | 2 × 27.5 MB | about 64 MB (10; a tie) | 87–174 MB (10–20) | below 17 MB |

At f64 p = 8 Rotation wins on one rank on every machine (the arithmetic: 0.87 against
1.15 µs on the M3 Max, 1.09 against 1.18 on locust, 2.22 against 2.92 on Kathleen with the
default x86-64 target, where Dense's product is compiled for SSE2; T2 §11), so the cache
model decides only where Dense wins on one rank: f64 p ≤ 6 and f32 p ≤ 8.

A budget of 64 MB of private dense tables per node lies inside the crossover of every
machine for f64 p = 6 and of the M3 Max for f32 p = 8, and below locust's and Kathleen's
crossover for f32 p = 8 (70–139 and 87–174 MB). No machine-independent budget fits all
three at f32 p = 8: at 8 ranks per node (70 MB) Dense is 1.12–1.39× Rotation's time on the M3
Max and 0.78–0.86× on locust and Kathleen.

## 4. The rule

**Inputs and how every rank agrees them.** p and the precision are the builder's; the
ranks per node come from one `split_shared` on the host path, the maximum over the nodes
agreed in step 7's existing all-reduce (today the maximum leaf size: it becomes a two-value
maximum), so every rank resolves the same strategy; threads per rank do not enter (threads
share their rank's copy, and the rule counts copies). No machine property enters: the grid
does not need one except in the band named below, and a rule that reads the cache sizes
would resolve differently on machines that run the same job.

**The options**, each scored over every measured cell (194 cells: both distributions, every
precision, p and ranks × threads of Section 2.1), as the worst wall-time ratio to the best
strategy per machine:

| option | rule | worst: M3 Max | locust | Kathleen | one-rank output bits |
| --- | --- | ---: | ---: | ---: | --- |
| today | Dense for p ≤ 8 | 4.49 (f64 p = 8, 12 × 1) | 6.30 (f64 p = 8, 72 × 1) | 2.59 (f64 p = 8, 40 × 1) | unchanged |
| A | Dense where it wins on one rank (f64 p ≤ 6, f32 p ≤ 8), else Rotation | 2.45 (f64 p = 6, 12 × 1) | 3.01 (f32 p = 8, 72 × 1) | 1.62 (f64 p = 6, 40 × 1) | f64 p = 7, 8 change |
| A′ | Rotation for p ≥ 4 | 1.24 (f32 p = 8, 1 × 1) | 1.43 (f32 p = 8, 1 × 1) | 1.33 (f32 p = 8, 4 × 1) | p = 4–8 change |
| **B** | **Dense iff D ≤ 9 MB and r · D ≤ 64 MB, else Rotation** | **1.08** (f64 p = 6, 1 × 12) | **1.28** (f32 p = 8, 8 × 1) | **1.18** (f32 p = 8, 8 × 1) | f64 p = 7, 8 change |
| C | `Auto` unchanged, configuration documented | as today with `Auto`; 1.00 for a user who names Rotation | | | unchanged |

Under B every cell is within 1.10 of the best except: f32 p = 8 at 8 ranks per node on
locust (1.21–1.28) and at 8–10 on Kathleen (1.11–1.18), where Dense would still win, and
locust's cube f64 p = 6 at 8 × 9 (1.11). Budgets of 48, 56 and 72 MB were scored too: 56 MB
gives the same worst cases, 48 MB worsens the M3 Max (1.16), 72 MB puts the M3 Max's f32
p = 8 at 8 ranks on Dense (1.39).

**Recommendation: B**, with the two constants named in `tables.rs` and an override:

```rust
impl M2lStrategy {
    /// The dense tables of one rank at which Dense stops winning on one rank (f64 p ≤ 6,
    /// f32 p ≤ 8 qualify; f64 p = 7 and f32 p = 9 do not).
    pub const AUTO_DENSE_RANK_BYTES: usize = 9_000_000;
    /// The private dense tables of one node above which Rotation wins.
    pub const AUTO_DENSE_NODE_BYTES: usize = 64_000_000;

    /// `self`, or for `Auto`: Dense iff D(p, T) ≤ AUTO_DENSE_RANK_BYTES and
    /// ranks_per_node · D(p, T) ≤ node_bytes; Rotation otherwise. Never `Auto`.
    pub fn resolve_on_node(self, p: usize, value_bytes: usize, ranks_per_node: usize,
                           node_bytes: usize) -> Self;
}
// FmmBuilder::dense_node_budget(bytes): the node budget of `Auto` (default
// AUTO_DENSE_NODE_BYTES); reported with the resolved strategy.
```

f64 p = 7 (10.9 MB) and f32 p = 9 (13.3 MB) were not measured: the 9 MB cap puts them on
Rotation, as the one-rank trend between p = 6 and p = 8 suggests (*extrapolated*); T4 adds
f64 p = 7 to its before/after runs to confirm or move the cap. `resolve(p)` stays for the
one-rank case (ranks per node 1, f64) or is deprecated, as T4 finds simpler;
`AUTO_DENSE_MAX_P` is deprecated (it no longer describes the rule). The budget override
lets a machine with a larger shared cache (locust, Kathleen at f32 p = 8) keep Dense to
8–10 ranks per node; the default is the value that never puts the M3 Max on the wrong side
by more than 8%.

**Signed off: the budget per machine class.** The library default stays 64 MB and
machine-independent; the harness (`tools/scaling/run.sh` and the reports) sets it per
machine class with `dense_node_budget`: 64 MB on the M3 Max, 96 MB on locust and on
Kathleen. Rescored over the same cells, 96 MB keeps every Kathleen cell within 1.01 of the
best (its window is above 87 MB, where f32 p = 8 at 10 ranks still wants Dense, and below
128 MB, where f64 p = 6 at 20 ranks wants Rotation: 128 MB would cost 1.24) and leaves
locust's worst at 1.11 (cube f64 p = 6 at 8 × 9, 51 MB, under any budget from 64 MB); any
budget above 64 MB puts the M3 Max's f64 p = 6 at 12 ranks on Dense (2.45).

**The margin it keeps** (decision 8): within 1.10 of the best at f64 p = 8 at every count
and split measured on the three machines (it resolves to Rotation, the best in all 48
cells), and at f32 p = 8 at one rank, the largest count and the best split; in the f32
p = 8 band of 8–10 ranks per node on locust and Kathleen it is 1.11–1.28 (Dense would still
win there and today's rule is at the best).

**What it does to Phase 5's guarantee and to what is reported.** The guarantee is
restated per resolved strategy: on P ranks the output is bit for bit the one-rank `Fmm` of
the same *resolved* strategy over the union in rank order. The one-rank default changes for
f64 p = 7 and 8 (Dense → Rotation; the outputs differ by 5.8–6.9e-15 relative L2 at p = 8,
Section 2.1), and the multi-rank default changes wherever r · D exceeds the budget, so the
same input on 1 and on 12 ranks may resolve differently: each run's `Fmm::strategy()` says
which, and a test or a user who needs the same bits across rank counts names the strategy.
`ThreadingReport` gains the node context (`ranks_per_node`, the maximum agreed) and its
`Display` the reason ("Auto → Rotation: 12 ranks per node × 8.7 MB > 64 MB"), so every
benchmark line shows it.

## 5. Tables at build and at scale

| option | cost at the largest count, Dense p = 8 (p = 12) | collectives |
| --- | --- | --- |
| every rank builds (today) | 211–607 ms per rank (1.8–6.7 s) | none |
| `TableCache` warmed by one job, then every rank loads | 19–30 ms (84–127 ms); 119 ms (489) on nodes that did not write the files | none |
| one rank per node builds and stores, the others wait at a node barrier, then load | 256–839 ms (2.0–5.6 s), once; then as above | one node barrier |
| every rank stores into an empty cache | 429–2,791 ms | none |

**Recommendation** (decision 7): keep `TableCache` as it is (Section 2.4 confirmed its
concurrent stores on Lustre and on two nodes) and make *one rank per node first* the
behaviour of `FmmBuilder::table_cache` when the directory lacks a file: the node's lowest
rank `load_or_build`s, the node barrier (on the `split_shared` communicator the rule
already makes), then the others load. Without `table_cache` nothing changes (every rank
builds). The cost falls away with rule B, which builds Dense only where the tables are at
most 9 MB per rank (at most 0.6 s on Kathleen, 0.2 s elsewhere); it matters for an
explicit Dense or Classes at large p and for every later run, which loads in 19–30 ms. The
first load on fresh nodes (119 ms at p = 8 on 80 ranks) is Lustre's client cache; no
change to the cache is needed.

## 6. One copy per node

| option | verdict |
| --- | --- |
| MPI shared memory (`MPI_Win_allocate_shared`) | possible now: rsmpi 0.8.2 has no window API, but its `mpi-sys` 0.2.4 declares `MPI_Win_allocate_shared`, `MPI_Win_shared_query` and `MPI_Win_fence`, and the user allows calling them (2026-10-10) under a documented exception to the `unsafe` rule. Measured (Section 2.6): a shared copy cuts Dense's per-pair cost 2.4–5.8× against private copies, but Rotation stays faster at f64 p = 8 (by 1.4–1.7×) and ties at f32 p = 8 on the M3 Max and locust; the gain over B is up to 25% of M2L on Kathleen at f32 p = 8 and 5% at f64 p = 6, and only at one rank per core: Dense with threads (4 × 10) gets the same per-pair cost today |
| memory-mapped cache files shared through the page cache | needs `mmap` (`unsafe` in `nd-fmm-tables`) or a new dependency; would give what the window gives, with the same verdict |
| threads per rank (few ranks per node) | available now; the measured equivalent of a shared copy (Section 7) |
| the batched host M2L of Phase 6 (C6.6) | reads each table once per level for many targets; changes the bytes per pair, so Dense's per-rank copy stops mattering whatever the sharing |

**Recommendation** (decision 6): no shared window in this phase. B and ranks × threads
recover the loss on every machine measured, and the window's remaining gain (up to 25% of
M2L, Kathleen, f32 p = 8, one rank per core only) does not pay for an `unsafe` exception.
Revisit in Phase 5S, where the replicated top tree (distributed-fmm.md §14, S1 and S2)
grows per rank with P and a node window would hold it once; the spike's code shape
(Section 2.6) is the starting point, and the permission is recorded.

## 7. Ranks × threads

Measured best configurations at N = 10⁶, f64 p = 8 (Section 2.1): Rotation 1 × 12 on the
M3 Max (607 ms; 12 × 1 within 1%), 8 × 9 on locust (143 ms; 4 × 18 152, 72 × 1 160),
4 × 10 on Kathleen (573 ms; 2 × 20 587, 40 × 1 638). With Dense (f32 p = 8, f64 p = 6) the
best splits keep one copy per rank and few ranks: 1 × 12 (M3 Max, 481 ms against 12 × 1's
1,268), 8 × 9 (locust, 108 against 388), 4 × 10 (Kathleen, 392 against 788).

Guidance for the `threading` module documentation (S6 of distributed-fmm.md §14.4,
confirmed): **on a node with many cores, a few ranks with threads** — one rank per socket or
NUMA domain, or a few per socket (4 × 10 on 2 × 20 cores, 8 × 9 on 72) — ranks × threads ×
BLAS threads within the physical cores; with one rank per core the strategy rule (B) keeps
the default near the best, but every replicated per-rank structure (the tables, the top
tree, distributed-fmm.md §9.2) scales with the ranks, not the cores. Kathleen's 20 → 40
ranks step (1.4× in every stage, T2 §1) is the all-core clock and is paid by any split.

## 8. Task check (what T4 builds)

T4 builds, in `nd-fmm-exec` only:
- the rule B: `M2lStrategy::resolve_on_node` (or a `NodeContext` struct of p, value
  bytes, ranks per node and budget, documented), the two constants with their source (this
  document, Sections 2.1 and 3), `AUTO_DENSE_MAX_P` deprecated, `resolve`'s documentation
  replaced; `FmmBuilder::dense_node_budget`;
- the node context in `build` on the host path: one `split_shared` (kept for the table
  loading), the ranks per node agreed in step 7's all-reduce; the device path keeps its own
  `split_shared` and the `tune` rule; the `fmm` module's "Collectives" and
  distributed-fmm.md §9.1 updated (one communicator split more per build);
- `ThreadingReport`: `ranks_per_node` and the reason of the resolution in its `Display`;
- the table loading: with `table_cache`, one rank per node first, a node barrier, the others
  load; no change to `TableCache`;
- the tests: the rule's unit tests over the table of Section 4 (every precision, p = 2–20,
  ranks per node 1–128, the budget); in `tests/mpi_exec.rs`, at 2 and 4 ranks, every rank
  resolves the same strategy, an explicit strategy is never overridden, and the output is
  bit for bit the one-rank `Fmm` of the resolved strategy; a cold cache filled once and
  loaded equal at 2 and 4 ranks; the tests that assumed Dense at p = 7 or 8 under `Auto`
  name Dense;
- the harness: `scaling`'s `--kinds` and `--reference-strategy` (T3) stay, the summary
  gains `ranks_per_node` and the reason; `tools/scaling/run.sh` gains the crossover sweep
  (Dense against Rotation at the counts in between) and `--budget`, set per machine class
  as signed off (64 MB on the M3 Max, 96 MB on locust and Kathleen; Section 4);
- the threading guidance (Section 7) in the `threading` module documentation;
- the before/after runs of its brief, plus f64 p = 7 and f32 p = 9 at one rank (the cap).

Stays for Phase 6: the batched host M2L (C6.6), a build of the far field for the CPU
(signed off, Section 9; T2 §11: `-C target-cpu` makes Dense 1.19–1.24× faster on one Kathleen rank and Rotation
1.05–1.12× slower), AVX-512; for Phase 5S: a node window for the replicated top tree, if
S1/S2 measure it worth it.

## 9. Questions for sign-off

1. **Decision 5, the strategy rule.** Recommended: B (Dense iff the rank's dense tables ≤
   9 MB and ranks per node × those ≤ 64 MB, else Rotation), with the budget overridable.
   It changes the one-rank default at f64 p = 7 and 8 (to Rotation, faster at p = 8 on
   every machine by 3–27%; p = 7 not measured) and the multi-rank default above the
   budget; Phase 5's guarantee is restated per resolved strategy.
2. **Decision 6, shared tables per node.** Recommended: none in this phase (Section 6: the
   window's measured gain over B and threads is at most 25% of M2L, at f32 p = 8 on
   Kathleen with one rank per core); revisit in Phase 5S for the replicated top tree,
   under the user's `mpi-sys` permission.
3. **Decision 7, the table cache at scale.** Recommended: with `table_cache`, one rank per
   node builds and stores first, the others wait at a node barrier and load; `TableCache`
   unchanged; without a cache, every rank builds as today.
4. **Decision 8, the margin.** Recommended: 10% at p = 8 (f64 and f32) at every ranks-per-
   node count of T3's grid (one rank, the largest count, the splits), with the f32 p = 8
   band of 8–10 ranks per node on locust and Kathleen (1.11–1.28 under the default budget)
   named as the known exception and covered by the budget override.
5. **The budget's value.** 64 MB by default (the M3 Max's crossover); should T4 instead
   set it per machine class in the harness (for example 128 MB on locust and Kathleen),
   keeping the library default machine-independent?
6. **`AUTO_DENSE_MAX_P`**: deprecate (recommended) or keep with a new meaning?
7. **A far-field build for the CPU** (T2 §11): a Phase 6 question, raised here because it
   moves the one-rank comparison on Kathleen (Dense and Rotation within 0.5% with `-C
   target-cpu=cascadelake`); the rule B does not depend on it at f64 p = 8 (Rotation still
   1.9–2.4× faster at 40 ranks).

**Signed off** by the user on 2026-10-11: questions 1–4 and 6 as recommended; question 5
per machine class (the library default 64 MB, the harness 64 MB on the M3 Max and 96 MB on
locust and Kathleen; Section 4); question 7 a far-field build for the CPU, in Phase 6
(Section 8).

## 10. Outcome (Phase 5N T4)

Built and measured in T4 on 2026-10-11 (docs/phase5n/T4-node-m2l.md; the report is
`fmm-validate/results/phase5n-kathleen.md`, "Phase 5N, T4": the M3 Max, locust, and one
Kathleen node in job 241833). The sections above stay as signed off; this one records what
was built where it differs, and the numbers.

### 10.1 Decisions as taken

| Topic (section) | As signed off | As built |
| --- | --- | --- |
| the rule (§4, decision 5) | B: Dense iff D(p, T) ≤ 9 MB and r · D ≤ the node budget | as designed: `M2lStrategy::resolve_on_node` over a `NodeContext` (p, value bytes, ranks per node, budget) and `M2lStrategy::choose`, which also returns the reason (`StrategyChoice`: named, the node rule with its inputs, or the device path); `AUTO_DENSE_RANK_BYTES` = 9 MB and `AUTO_DENSE_NODE_BYTES` = 64 MB; `AUTO_DENSE_MAX_P` and `resolve(p)` deprecated (`resolve` is the one-rank f64 case of the rule); `Tables::build` and `load_or_build` resolve `Auto` on one rank in their own precision |
| the 9 MB cap (§4: f64 p = 7 and f32 p = 9 extrapolated) | T4 to confirm or move it | **kept, by the user (2026-10-11)**: at one rank f64 p = 7 splits by machine (Dense 6–10% faster on the M3 Max and locust, 10–15% slower on Kathleen), f32 p = 9 is Dense's by 14–31% everywhere but beyond f32's useful degree |
| the node context (§4, §8) | one `split_shared` on the host path; ranks per node in step 7's all-reduce | as designed: one split on every backend in step 1 (the device path's `RankPlacement` takes the same communicator, so the device path splits once as before), step 7's all-reduce of two maxima; threads do not enter |
| what is reported (§4) | `ThreadingReport` with the ranks per node and the reason | as designed: `ThreadingReport::{ranks_per_node, m2l}`, the `Display` ending "ranks per node 40; M2L Auto → Rotation: 40 ranks per node × 8.7 MB > 96 MB"; `Fmm::strategy` the resolved strategy, as before |
| the budget per machine class (§4, question 5) | the library 64 MB; the harness 64 MB on the M3 Max, 96 MB on locust and Kathleen | `FmmBuilder::dense_node_budget`, the library default 64 MB; the harness (`tools/scaling/run.sh`, `BUDGET`) **56 MB on the M3 Max, set by the user on 2026-10-11** after T4 measured f64 p = 6 at 8 and 10 ranks per node (Dense 1.11 and 1.83× Rotation's time under 64 MB; §10.2), and 96 MB on locust and Kathleen |
| tables at scale (§5, decision 7) | with `table_cache`, one rank per node first, a node barrier, the others load; `TableCache` unchanged | as designed: the node's lowest rank `load_or_build`s, `barrier` on the node's communicator, the others `load_or_build` (a load, unless a file was rejected); with no cache every rank builds. A warm cache costs one load more on the first rank. `nd-fmm-tables` unchanged |
| one copy per node (§6, decision 6) | none in this phase | none; nothing built |
| the guarantee (§4) | per resolved strategy | as designed: the one-rank references of `tests/mpi_exec.rs`, `tests/multi_rank.rs` and the `scaling` harness name the strategy the multi-rank run resolved |
| the harness (§8) | the summary gains ranks per node and the reason; the crossover sweep; `--budget` | `scaling --budget MB`; the summary's `requested`, `budget_mb`, `dense_rank_mb`, `dense_node_mb` (the reason in the threading line); the sweeps `node-rule` (T3's grid: `auto` and the other strategy, the cap, the errors) and `crossover`; the T2 sweep `nodes-strong` names `8:dense` for its former `8:auto` |

### 10.2 Measured

| quantity | M3 Max | locust | Kathleen |
| --- | --- | --- | --- |
| the exit gate: the default over the best at p = 8, every count of T3's grid, f64 / f32 | 1.000 / 1.015 (64 MB) | 1.000 / 1.011 | 1.000 / 1.029 |
| the old default (Dense) over the best at f64 p = 8: one rank / largest one-thread count | 1.28 / 4.03 | 1.02 / 6.45 | 1.34 / 2.83 |
| the f32 p = 8 band (the budget's switch; decision 8's exception) | 1.000 at 6, 8, 10 ranks (64 and 56 MB) | 1.058 / 1.101 at 12 ranks per node | 1.158 / 1.144 at 12, 1.017 / 1.067 at 16 |
| f64 p = 6 where the budget decides | 1.111 / 1.019 at 8, 1.832 / 1.312 at 10 (64 MB); 1.118 / 1.010 and 1.000 / 1.001 (56 MB) | 1.000 at 8–24 | 1.044 / 1.000 to 1.092 / 1.038 at 8–12; 1.000 at 16, 20 |
| f64 p = 3; f64 p = 12 | Dense, the best everywhere; Rotation, as before | the same | the same |
| errors, f64 p = 8 (φ L2) | 1.889e-5, Dense's to 4 digits | the same | the same |
| errors, f32 p = 8 at the largest count (φ L2, Rotation / Dense) | 1.891e-5 / 1.910e-5 | 1.891e-5 / 1.911e-5 | 1.891e-5 / 1.910e-5 |

(cube / Plummer where two values; ratios of wall time per evaluation, N = 10⁶.) The model of
§3 holds where it was tested: at f64 p = 8 Rotation wins on one rank and at every count; in
the band the crossover moves with the machine and the precision: in bytes per node about
38–51 MB for f64 p = 6 and 52–70 MB for f32 p = 8 on the M3 Max, 76–102 and 105–139 MB on
locust, about 51–64 and 140–175 MB on Kathleen. So no budget in bytes fits both precisions on any of
the three machines (on locust the two windows just miss), and the rule keeps one budget per
machine, which costs at most 1.12 on the M3 Max (56 MB), 1.10 on locust and 1.16 on
Kathleen (96 MB) in the measured band; a rule that weighs the
precision, or the batched M2L of Phase 6 (C6.6), which removes the per-pair table traffic,
is the next step, not this phase's.

**Collectives.** One communicator split more per build on the host path (§2.5: at most
1.6 ms on locust and Kathleen, 6.4 ms on the M3 Max at 12 ranks), one barrier of the node's
communicator with a table cache; the step 7 all-reduce carries two values. The evaluation is
unchanged. **Tests.** The rule's unit tests (both precisions, p = 0–20, 1–128 ranks per node,
six budgets, the reasons); `tests/mpi_exec.rs`'s **the strategy rule on the node** at 1, 2,
4, 8 ranks (M3 Max, locust at 8, Kathleen at 1–8 and 2 × 40): the agreement, named
strategies kept, the cold cache filled once per node (twice on two Kathleen nodes sharing
Lustre). The debug run at one rank: 67.5 s (69.6 s before T4).

**For Phase 5S** (laplace-fmm-plan §7, Phase 5N): take the strategy the rule picks with the
harness's budget (Kathleen 96 MB: Rotation at f64 p = 8 in every configuration, Dense at
f64 p = 3; f32 p = 8 Dense at 2 × 20 and Rotation at 40 × 1), prefer a few ranks per node
with threads (4 × 10 the fastest split at f64 p = 8 on one Kathleen node), and pass a
`table_cache` on Lustre for builds at scale.

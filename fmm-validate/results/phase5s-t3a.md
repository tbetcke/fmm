# Keys routed by weight instead of sample sorts (Phase 5S T3a)

S4 of docs/design/scale-out.md §7.1, as docs/phase5s/T3a-octree-key-routing.md asks: `Octree::new`
no longer sorts any key across the ranks. The first sample sort (`linearize`'s `parsort` and
its successor exchange) is replaced by routing the input keys to provisional owners by
weight; the second (inside `balance`) and the second key move are replaced by routing each
rank's 2:1 closure by the partition bounds. Run on 2026-10-12 on the M3 Max, locust (GH200)
and Kathleen. The octree is the Phase 5 octree key for key on every run below; only the
construction's communication changed.

**Measured** means run here and read from the output; timings are reported, never asserted.
locust is a shared node: the load before and after each launch is in the driver's log (0.0
before the timed runs, which ran one at a time). Kathleen nodes are exclusive; every job is
named.

## Summary

- **Key for key.** The leaves, the coarse blocks with their weights, the partition bounds
  and every key with its `KeyType` equal the Phase 5 construction's (rlst 0.10.0) on every
  workload of `test_mpi_weighted_partition` and on every case of
  `test_mpi_construction_edge_cases`, at 1–8 ranks (M3 Max, locust), 80 (Kathleen, 2 nodes),
  160 (4 nodes), 512 (locust, oversubscribed) and 640 (Kathleen, oversubscribed), and in the
  probe at every timed and counted run (N up to 10⁸, both inputs). `Fmm` outputs are bit for
  bit (`multi_rank` at 2, 4, 8 ranks: 0 values differ).
- **`Octree::new`** (release, slowest rank), the cube / Plummer sphere at N = 10⁷ on the
  random share: 30–31 / 46–51 ms at 160 Kathleen ranks against 69 / 87–92 ms with rlst
  0.10.0's balanced `parsort` (T1s) and 2,607 / 2,655 ms with rlst 0.9.0 (T1); at N = 10⁸,
  283–289 / 399–462 ms against 675–679 / 820–886 ms (T1s) and 9,656 / 10,093 ms (T1). At 72
  locust ranks, N = 10⁷: 29–30 / 43–48 ms against 91–92 / 101–106 ms (T1s). Removing the
  sorts gains another 1.8–2.4× at 160 ranks on top of rlst #40's 11–38×.
- **Memory.** The step that moves the input keys peaks at 268 MB on the heaviest rank
  instead of 482 MB (40 Kathleen ranks, N = 10⁸, cube); with every point on rank 0, rank 0
  peaks at 4.9 GB instead of 10.5 GB (N = 10⁸) and 0.55 GB instead of 1.1 GB (N = 10⁷).
- **Bytes and messages.** The keys travel deduplicated, 16-byte (key, multiplicity) pairs
  instead of 24-byte sort records, in one all-to-all-v: at 640 ranks, N = 10⁷, 639 messages
  and 0.26 MB per rank instead of `parsort`'s 4,473 messages, 0.77 MB sent and 1.84 MB
  received per rank (its candidate all-gathers alone grow with P). The second sort's
  2,857–3,378 messages per rank become 143–417, and the second key move goes.
- **Two findings on the way:** a closure key coarser than the blocks must be routed by its
  first finest-level descendant, not by its own Morton key (§1); and the edge-case
  example's own key generator overflowed level 16 above about 227 ranks (§2.3, fixed).

## 1. The change

`octree/src/octree/implementation.rs`; `Octree::new` (octree/src/octree.rs) calls the new
functions in the Phase 5 order. Every rank enters every collective; the number of rounds is
global.

1. **One local sort.** `route_keys` sorts the rank's input keys once (duplicates kept). The
   rounds count them; their runs of equal keys become the pairs. `compute_coarse_tree` no
   longer sorts its own copy, and nothing re-sorts received keys.
2. **Routing by weight.** From the root, each round counts the sorted keys under every
   provisional block in one merge pass and sums the counts in one all-reduce
   (`weight_rounds`); every block heavier than W/(kP) and above `max_level` splits: O1's
   rule without the distinct-key rule (`refine_heavy_blocks` with the weights as the
   distinct counts and `max_fine_keys` 0). The blocks are cut by `partition_blocks`; each
   rank sends its pairs (`count_duplicates`) to the provisional owners
   (`rank_counts`: one binary search per rank, as `sort_to_bins` places keys) in one
   all-to-all-v, and merges the received sorted runs, summing the multiplicities
   (`merge_key_counts`, a k-way merge). Every copy of a key is then on one rank; the input
   keys are all on level 16, so the distinct keys are the linearized keys.
3. **O1 unchanged.** `compute_coarse_tree(linear_keys, multiplicities, …)` counts d(b) from
   the routed keys and w(b) as the sum of their multiplicities (or d(b) with
   `PartitionWeight::DistinctKeys`): the global counts, hence the blocks, weights and cut of
   Phase 5. The key move to the O1 owners stays; it is nearly empty now (at most 2 messages
   per rank, mean 0.002–0.005 at 512–640 ranks), because both cuts are by weight.
4. **Balance without a sort.** `balance(refined, bounds, comm)` builds the local closure
   (`balance_closure`, the Phase 5 loop), linearizes it locally, routes each key by the
   bounds and linearizes what it receives. Routing a key by its own Morton key is wrong for
   a key coarser than a rank's first block: it shares that block's anchor and sorts before
   the block, hence before the rank's bound, and would reach the previous rank, which holds
   none of its descendants and keeps it. Each key is therefore placed by its first finest
   descendant (`first_finest_descendant`, the finest key at its anchor): a key coarser than
   the blocks reaches the owner of its first descendant block, whose leaves (or their
   descendants in the closure) are there too, and is removed as the global linearization
   removes it; a key at or below a block lies in that block's range with all its
   descendants; duplicates meet on one rank. The local linearization before sending removes
   only keys that have a descendant in the closure anyway (the union of the local
   linearizations linearizes to the same tree) and cuts the bytes. The result has no key
   coarser than a block, so every leaf is already on its owner: the second
   `redistribute_with_respect_to_coarse_tree` is gone.
5. **Debug checks.** `is_linear_tree` and `is_complete_linear_tree` are unchanged. New: the
   routed keys are distinct and ordered across the ranks (`communicate_back` and an
   all-reduce, debug only), and every balanced leaf is owned by the rank holding it (local).
6. **Docs.** `Octree::new`'s collective list and octree/CLAUDE.md's construction contract.

The distributed `linearize` (with `parsort`) is removed; `nd-octree` no longer calls rlst's
`parallel_sort`, nor uses `rand_chacha` outside `tools`. No public API changed.

## 2. Key for key

### 2.1 Unit tests (`nd-octree` lib, 65 pass)

- `test_weight_rounds_on_hand_made_counts`: the rounds on hand-made counts (heavy blocks
  split level by level, `max_level` 2 keeping them whole, a single block, no keys,
  `max_level` 0), one repeated key splitting down to `max_level` (no distinct-key rule:
  7 L + 1 blocks, L + 1 rounds), and clustered keys against the rule.
- `test_merge_key_counts_sums_the_runs`: duplicates across runs, empty runs first, in the
  middle and last, no runs; randomised against a map.
- `test_rank_counts_and_first_finest_descendants`: `rank_counts` equals `sort_to_bins`
  (repeated and invalid bounds included); keys of every level go to the owner of their block
  or first descendant block.
- `test_local_block_counts_against_the_expanded_keys`: d(b) and w(b) from the pairs equal the
  counts over the keys with their duplicates.
- `test_routed_balance_against_morton_balance`: S4's balance replayed serially on 1, 2, 3,
  5, 8 and 13 "ranks" (O1's blocks, the cut, each rank's refined tree, its closure routed
  and linearized): the union equals `morton::balance` of all leaves (and the Phase 5 global
  linearization of the closures), each key on its owner; the replay routes keys coarser than
  a rank's first block, the case of §1.4.

### 2.2 Against the Phase 5 construction

`octree/examples/phase5_reference/mod.rs` is a private copy of the Phase 5 construction
(the private functions of implementation.rs as of T1s, without their debug checks; rlst
0.10.0's `parsort`). Both MPI examples build every tree with `Octree::new` and with the copy
and assert on every rank the same leaves, own blocks, gathered blocks, bounds and
`all_keys` (each key with its `KeyType`), and the copy's block weights against the weights
of the same blocks counted from the gathered input keys (the weights are not part of the
octree). `test_mpi_weighted_partition` does this for its four workloads (cube, Plummer
sphere, clusters, duplicates) under its three inputs (the share, every key on rank 0, the
last rank empty), the duplicates under the other weights and refinement factors, and the
root-only trees; `test_mpi_construction_edge_cases` for each of its cases, to which T3a adds
every key on one rank, every key identical (with `max_level` 16 and 4) and a root-only tree
with many distinct keys (`max_level` 0). The copy is removed when T3 lands.

Two mutants were caught: routing the closure by the raw key (one rank panics in
`sort_to_bins`, the key precedes the first bound), and O1 weights without the
multiplicities (the duplicates workload fails on every rank: leaves, blocks, bounds,
`all_keys`).

### 2.3 Runs

| Where | Ranks | What | Result |
| --- | --- | --- | --- |
| M3 Max | 1–8 | the six registered examples, debug, under `timeout` (48 launches) | pass |
| M3 Max | 1, 2, 4, 8 | `run-mpi-tests.sh` (`mpi_regressions`, `mpi_exec`, `mpi_threading`, the six examples; 36 launches) | pass |
| M3 Max | 2, 4, 8 | `multi_rank` (release, ignored) | pass; 0 values differ from the union in rank order in every row |
| locust | 1–8 | the six examples, debug (48 launches) | pass |
| locust | 1, 2, 4, 8 | `run-mpi-tests.sh` (36 launches) | pass |
| locust | 2, 4, 8 | `multi_rank` | pass |
| locust | 512, oversubscribed | `test_mpi_weighted_partition` (release, debug), `test_mpi_construction_edge_cases` (release, then debug after the fix below) | pass; see below |
| Kathleen 415614 | 80 (2 × 40) | `mpi.sbatch 80` (T2 has not merged) | 9 of 9 pass |
| Kathleen 415613 | 160 (4 × 40) | both examples, debug | pass |
| Kathleen 415613 | 640, oversubscribed | both examples, release | pass (1,252 s and 1,316 s) |
| Kathleen 422926 | 160 (4 × 40), 640 | `test_mpi_construction_edge_cases` after the fix, debug at 160, release at 640 | pass (82 s, 1,225 s) |

At 512 ranks the debug edge cases stopped at `morton::from_index_and_level`'s assertion: the
example's own clustered keys are at `20000 + 200 rank`, beyond the 2¹⁶ cells of level 16
from rank 228 on (release wraps silently, so the 640-rank release run of job 415613 used
wrapped indices on the upper ranks). The generator now takes `rank % 200` (and
`rank % 500` for the far keys); the octree code was not involved. After the fix both examples pass at 512 ranks on locust
in release and in debug (edge cases 538 s and 2,345 s) and at 160 and 640 on Kathleen (job
422926).

## 3. `Octree::new` before and after

The probe (throwaway, in the session's scratch directory, not committed, as T1's): for a
`scaling` workload (`Workload::new(dist, n)`; the random share, or every point on rank 0;
the domain by `compute_global_bounding_box`; the keys of sources and targets as `Fmm` passes
them, so every key twice; `max_level` 16, 64 points per leaf, the ghost-children layer) it
replays the Phase 5 construction (rlst 0.10.0's `parsort`, copied with counters, its draws
and result unchanged) and the T3a construction step by step, a barrier before each step,
with the time, the peak resident size during the step (`VmHWM` reset through
`/proc/self/clear_refs`), the collectives, messages and bytes; then times the real
`Octree::new` (a barrier before; it also builds the neighbour map, which the replays do not).
Two rounds, the second reported. Both replays and `Octree::new` are compared key for key on
every rank: **identical in every run** (84 timed and counted runs: 42 on each machine).

Columns: the slowest rank's time in ms; "after" splits the routing into the local sort, the
rounds and the pairs' all-to-all-v with the merge; "all steps" is the slowest rank's sum of
its steps (generate_all_keys included); `Octree::new` fastest–slowest rank; the peak of the
step moving the input keys (the first sort before, the pairs after), max / mean over ranks.

### 3.1 locust (one node, Open MPI 5.0.10, `--map-by core --bind-to core`)

| N | dist | input | ranks | first sort | second sort | before, all steps | after: local sort + rounds + pairs | balance by the bounds | after, all steps | `Octree::new` after | peak, first sort → pairs (MB, max / mean) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 10⁶ | cube | share | 8 | 13.8 | 0.24 | 24.5 | 3.09 + 1.15 + 2.22 = 6.47 | 0.07 | 12.7 | 15.1–15.3 | 50.1 / 46.4 → 36.6 / 35.0 |
| 10⁶ | cube | share | 32 | 5.22 | 0.40 | 9.51 | 0.74 + 0.33 + 1.05 = 2.12 | 0.10 | 4.55 | 4.97–5.13 | 56.9 / 54.7 → 59.9 / 57.8 |
| 10⁶ | cube | share | 72 | 3.46 | 0.55 | 6.95 | 0.32 + 0.32 + 0.76 = 1.40 | 0.15 | 3.68 | 4.09–4.19 | 39.1 / 37.5 → 40.6 / 37.9 |
| 10⁶ | plummer | share | 8 | 13.8 | 0.53 | 30.2 | 3.16 + 1.77 + 2.26 = 7.19 | 0.14 | 18.4 | 23.2–23.8 | 49.7 / 49.5 → 36.0 / 35.3 |
| 10⁶ | plummer | share | 32 | 5.25 | 0.59 | 11.7 | 0.75 + 0.66 + 1.11 = 2.51 | 0.13 | 7.18 | 8.26–8.58 | 59.4 / 56.3 → 62.5 / 59.7 |
| 10⁶ | plummer | share | 72 | 3.66 | 0.78 | 8.69 | 0.35 + 0.44 + 1.13 = 1.92 | 0.21 | 5.96 | 5.62–7.02 | 42.1 / 40.0 → 43.3 / 42.1 |
| 10⁶ | clusters | share | 8 | 13.8 | 0.47 | 31.7 | 3.16 + 3.11 + 2.19 = 8.45 | 0.15 | 21.2 | 25.2–26.0 | 49.7 / 49.3 → 36.1 / 35.3 |
| 10⁶ | clusters | share | 32 | 5.12 | 0.53 | 13.2 | 0.78 + 0.66 + 1.19 = 2.62 | 0.16 | 8.79 | 11.1–12.0 | 58.5 / 56.0 → 62.5 / 60.7 |
| 10⁶ | clusters | share | 72 | 3.64 | 0.74 | 9.84 | 0.40 + 0.43 + 0.80 = 1.62 | 0.22 | 6.49 | 7.12–7.72 | 42.9 / 40.9 → 42.9 / 40.9 |
| 10⁷ | cube | share | 8 | 184 | 1.15 | 298 | 39.4 + 9.90 + 26.1 = 75.4 | 0.48 | 142 | 171–173 | 234 / 229 → 136 / 135 |
| 10⁷ | cube | share | 32 | 76.5 | 0.82 | 107 | 9.83 + 2.97 + 10.9 = 23.7 | 0.24 | 40.6 | 47.4–48.8 | 156 / 153 → 134 / 131 |
| 10⁷ | cube | share | 72 | 61.1 | 1.04 | 83.0 | 5.23 + 1.94 + 7.74 = 14.9 | 0.27 | 26.8 | 29.4–30.1 | 64.4 / 63.6 → 54.2 / 52.2 |
| 10⁷ | plummer | share | 8 | 183 | 3.21 | 364 | 37.5 + 16.9 + 26.5 = 80.9 | 1.08 | 210 | 287–292 | 239 / 238 → 148 / 148 |
| 10⁷ | plummer | share | 32 | 76.7 | 1.68 | 129 | 10.1 + 5.40 + 12.6 = 28.1 | 0.45 | 64.6 | 78.6–83.9 | 163 / 156 → 140 / 136 |
| 10⁷ | plummer | share | 72 | 60.4 | 1.56 | 92.1 | 5.35 + 2.91 + 8.75 = 17.0 | 0.43 | 37.9 | 42.7–48.4 | 69.3 / 64.4 → 57.7 / 53.5 |
| 10⁷ | clusters | share | 8 | 184 | 3.02 | 373 | 38.4 + 17.9 + 26.5 = 82.8 | 1.14 | 220 | 300–307 | 241 / 237 → 148 / 142 |
| 10⁷ | clusters | share | 32 | 76.5 | 1.60 | 129 | 10.2 + 5.58 + 11.9 = 27.7 | 0.47 | 65.6 | 80.1–83.6 | 163 / 156 → 141 / 135 |
| 10⁷ | clusters | share | 72 | 60.4 | 3.54 | 96.6 | 5.24 + 2.72 + 9.10 = 17.1 | 0.45 | 38.2 | 45.8–48.6 | 71.8 / 67.8 → 60.7 / 57.6 |
| 10⁶ | cube | rank0 | 8 | 93.2 | 0.24 | 138 | 28.8 + 8.08 + 3.50 = 40.4 | 0.07 | 46.5 | 51.5–51.7 | 142 / 40.5 → 97.3 / 30.4 |
| 10⁶ | cube | rank0 | 32 | 99.1 | 0.37 | 141 | 28.7 + 10.1 + 2.76 = 41.6 | 0.10 | 43.8 | 44.4–44.5 | 133 / 31.6 → 87.2 / 30.9 |
| 10⁶ | cube | rank0 | 72 | 106 | 0.75 | 150 | 29.4 + 13.5 + 3.45 = 46.3 | 0.22 | 48.6 | 47.7–47.8 | 146 / 30.5 → 102 / 29.9 |
| 10⁶ | plummer | rank0 | 8 | 92.4 | 0.53 | 151 | 28.6 + 14.1 + 3.46 = 46.2 | 0.14 | 57.8 | 64.0–64.7 | 142 / 40.4 → 98.0 / 32.5 |
| 10⁶ | plummer | rank0 | 32 | 99.0 | 0.56 | 157 | 28.9 + 17.4 + 3.31 = 49.6 | 0.14 | 54.1 | 57.1–57.4 | 149 / 33.3 → 105 / 33.3 |
| 10⁶ | plummer | rank0 | 72 | 108 | 0.92 | 162 | 29.3 + 20.1 + 3.45 = 52.9 | 0.30 | 56.4 | 58.4–58.7 | 149 / 31.5 → 103 / 31.1 |
| 10⁶ | clusters | rank0 | 8 | 93.9 | 0.50 | 153 | 29.6 + 14.4 + 3.49 = 47.5 | 0.16 | 59.8 | 66.4–67.8 | 142 / 40.5 → 97.3 / 33.0 |
| 10⁶ | clusters | rank0 | 32 | 100.0 | 0.55 | 158 | 30.4 + 18.2 + 3.20 = 51.8 | 0.17 | 57.6 | 59.0–59.9 | 142 / 30.5 → 94.7 / 30.1 |
| 10⁶ | clusters | rank0 | 72 | 106 | 0.85 | 162 | 30.6 + 19.9 + 5.43 = 55.9 | 2.21 | 69.6 | 60.5–62.9 | 148 / 28.9 → 101 / 29.2 |
| 10⁷ | cube | rank0 | 8 | 1,220 | 1.20 | 1,713 | 351 + 77.2 + 40.6 = 469 | 0.53 | 538 | 581–583 | 1,113 / 252 → 553 / 146 |
| 10⁷ | cube | rank0 | 32 | 1,271 | 0.86 | 1,722 | 352 + 78.0 + 37.1 = 467 | 0.24 | 484 | 501–502 | 1,091 / 79.5 → 530 / 64.8 |
| 10⁷ | cube | rank0 | 72 | 1,347 | 1.92 | 1,816 | 354 + 105 + 35.5 = 494 | 0.51 | 506 | 508–509 | 1,077 / 51.4 → 514 / 41.0 |
| 10⁷ | plummer | rank0 | 8 | 1,242 | 3.25 | 1,850 | 354 + 136 + 40.8 = 531 | 1.12 | 661 | 748–754 | 1,113 / 257 → 556 / 160 |
| 10⁷ | plummer | rank0 | 32 | 1,269 | 1.97 | 1,824 | 344 + 152 + 37.3 = 533 | 0.47 | 570 | 596–602 | 1,090 / 85.7 → 537 / 73.8 |
| 10⁷ | plummer | rank0 | 72 | 1,307 | 1.84 | 1,844 | 345 + 152 + 35.3 = 532 | 0.54 | 552 | 568–574 | 1,080 / 53.0 → 520 / 43.5 |
| 10⁷ | clusters | rank0 | 8 | 1,212 | 3.06 | 1,845 | 352 + 137 + 43.9 = 532 | 1.14 | 668 | 755–762 | 1,114 / 254 → 561 / 157 |
| 10⁷ | clusters | rank0 | 32 | 1,277 | 1.76 | 1,835 | 345 + 152 + 36.1 = 533 | 0.47 | 571 | 596–600 | 1,086 / 82.2 → 530 / 67.0 |
| 10⁷ | clusters | rank0 | 72 | 1,357 | 1.77 | 1,893 | 349 + 151 + 34.3 = 535 | 0.55 | 556 | 572–575 | 1,080 / 50.6 → 525 / 41.4 |

With rlst 0.10.0 (T1s, the same machine, the real `Octree::new`): cube 327–330 ms at 8 ranks
and 91–92 ms at 72; Plummer 437–444 and 101–106 ms (N = 10⁷).

### 3.2 Kathleen (job 415613, node-c11a-[073,105,118,126]; 40 ranks per node, `--map-by ppr:20:socket:PE=1 --bind-to core`, Open MPI 4.1.6 over PSM2, `PSM2_TID=0`)

| N | dist | input | ranks | first sort | second sort | before, all steps | after: local sort + rounds + pairs | balance by the bounds | after, all steps | `Octree::new` after | peak, first sort → pairs (MB, max / mean) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 10⁷ | cube | share | 40 | 135 | 2.12 | 186 | 13.3 + 5.45 + 22.4 = 41.2 | 0.56 | 70.8 | 82.8–84.6 | 91.9 / 86.2 → 72.2 / 66.4 |
| 10⁷ | cube | share | 80 | 78.3 | 2.51 | 115 | 6.75 + 4.10 + 16.0 = 26.9 | 0.63 | 48.1 | 52.6–53.8 | 91.1 / 83.0 → 79.1 / 72.9 |
| 10⁷ | cube | share | 160 | 49.7 | 4.41 | 72.2 | 3.09 + 2.28 + 10.8 = 16.2 | 0.59 | 27.6 | 30.4–31.0 | 80.3 / 72.3 → 79.4 / 71.9 |
| 10⁷ | plummer | share | 40 | 135 | 3.34 | 218 | 13.3 + 10.7 + 22.6 = 46.7 | 1.64 | 111 | 121–133 | 96.9 / 89.8 → 77.6 / 72.1 |
| 10⁷ | plummer | share | 80 | 72.6 | 3.64 | 126 | 6.41 + 6.06 + 16.3 = 28.8 | 1.00 | 68.0 | 73.1–81.7 | 91.6 / 84.7 → 82.7 / 76.2 |
| 10⁷ | plummer | share | 160 | 42.6 | 5.37 | 77.3 | 3.17 + 4.76 + 11.3 = 19.2 | 0.86 | 42.5 | 45.7–50.5 | 80.7 / 74.9 → 79.5 / 74.1 |
| 10⁷ | clusters | share | 40 | 138 | 2.89 | 219 | 13.3 + 10.6 + 24.8 = 48.7 | 0.83 | 111 | 124–130 | 91.4 / 85.4 → 73.5 / 65.2 |
| 10⁷ | clusters | share | 80 | 74.8 | 3.34 | 128 | 6.32 + 5.88 + 18.4 = 30.6 | 0.99 | 66.5 | 77.0–81.7 | 90.2 / 83.2 → 80.6 / 72.5 |
| 10⁷ | clusters | share | 160 | 44.0 | 5.00 | 80.6 | 2.80 + 4.40 + 11.7 = 18.9 | 0.85 | 43.9 | 48.5–53.1 | 82.9 / 74.6 → 81.4 / 74.5 |
| 10⁸ | cube | share | 40 | 1,910 | 9.84 | 2,501 | 200 + 52.0 + 249 = 501 | 2.30 | 805 | 942–969 | 482 / 466 → 268 / 256 |
| 10⁸ | cube | share | 80 | 990 | 6.20 | 1,275 | 90.2 + 33.8 + 161 = 285 | 1.60 | 434 | 502–515 | 292 / 276 → 197 / 181 |
| 10⁸ | cube | share | 160 | 560 | 8.22 | 780 | 57.1 + 31.0 + 106 = 194 | 1.37 | 314 | 283–289 | 178 / 160 → 129 / 110 |
| 10⁸ | plummer | share | 40 | 1,924 | 23.5 | 3,050 | 202 + 104 + 246 = 552 | 7.26 | 1,348 | 1,646–1,922 | 511 / 478 → 306 / 271 |
| 10⁸ | plummer | share | 80 | 989 | 13.7 | 1,520 | 89.2 + 53.7 + 173 = 315 | 4.05 | 690 | 800–945 | 300 / 276 → 212 / 190 |
| 10⁸ | plummer | share | 160 | 501 | 11.2 | 773 | 39.7 + 31.5 + 95.9 = 167 | 2.41 | 367 | 399–462 | 180 / 163 → 128 / 116 |
| 10⁸ | clusters | share | 40 | 1,915 | 20.9 | 2,958 | 201 + 104 + 256 = 561 | 5.94 | 1,281 | 1,574–1,775 | 499 / 468 → 292 / 262 |
| 10⁸ | clusters | share | 80 | 989 | 12.9 | 1,471 | 89.6 + 53.5 + 169 = 312 | 3.34 | 642 | 750–860 | 300 / 273 → 212 / 187 |
| 10⁸ | clusters | share | 160 | 499 | 10.5 | 751 | 42.0 + 33.9 + 95.4 = 171 | 1.98 | 338 | 376–428 | 188 / 164 → 136 / 116 |
| 10⁷ | cube | rank0 | 40 | 3,011 | 2.12 | 3,845 | 600 + 196 + 91.7 = 888 | 0.58 | 918 | 978–980 | 1,095 / 81.5 → 552 / 73.0 |
| 10⁷ | cube | rank0 | 80 | 3,138 | 2.66 | 4,008 | 601 + 244 + 120 = 965 | 0.67 | 984 | 1,051–1,052 | 1,100 / 79.9 → 544 / 73.6 |
| 10⁷ | cube | rank0 | 160 | 3,116 | 5.12 | 3,981 | 601 + 245 + 96.4 = 943 | 0.74 | 954 | 1,021–1,022 | 1,093 / 65.9 → 537 / 65.7 |
| 10⁷ | plummer | rank0 | 40 | 3,021 | 3.60 | 4,126 | 603 + 392 + 91.4 = 1,086 | 1.07 | 1,145 | 1,237–1,248 | 1,104 / 89.1 → 559 / 79.4 |
| 10⁷ | plummer | rank0 | 80 | 3,142 | 3.82 | 4,227 | 604 + 392 + 120 = 1,116 | 1.01 | 1,151 | 1,232–1,241 | 1,102 / 83.2 → 548 / 77.4 |
| 10⁷ | plummer | rank0 | 160 | 3,126 | 5.88 | 4,250 | 600 + 441 + 96.0 = 1,138 | 0.99 | 1,162 | 1,251–1,256 | 1,095 / 69.2 → 539 / 68.1 |
| 10⁷ | clusters | rank0 | 40 | 3,017 | 3.08 | 4,119 | 600 + 393 + 91.8 = 1,084 | 0.90 | 1,140 | 1,243–1,249 | 1,101 / 83.8 → 558 / 73.2 |
| 10⁷ | clusters | rank0 | 80 | 3,141 | 3.68 | 4,218 | 600 + 391 + 119 = 1,109 | 1.00 | 1,144 | 1,228–1,233 | 1,107 / 81.8 → 549 / 73.6 |
| 10⁷ | clusters | rank0 | 160 | 3,124 | 5.37 | 4,244 | 599 + 441 + 95.6 = 1,135 | 0.90 | 1,161 | 1,251–1,254 | 1,104 / 71.0 → 548 / 69.7 |
| 10⁸ | cube | rank0 | 40 | 35,987 | 9.56 | 45,262 | 6,933 + 1,973 + 852 = 9,758 | 2.26 | 10,064 | 10,676–10,707 | 10,497 / 461 → 4,924 / 292 |
| 10⁸ | cube | rank0 | 80 | 37,428 | 6.54 | 46,970 | 6,936 + 2,463 + 1,180 = 10,579 | 1.71 | 10,728 | 11,318–11,332 | 10,505 / 302 → 4,909 / 197 |
| 10⁸ | cube | rank0 | 160 | 36,922 | 7.60 | 46,425 | 6,933 + 2,462 + 917 = 10,312 | 1.19 | 10,389 | 10,976–10,981 | 10,487 / 159 → 4,902 / 133 |
| 10⁸ | plummer | rank0 | 40 | 36,028 | 23.8 | 48,215 | 6,949 + 3,945 + 903 = 11,797 | 7.06 | 12,651 | 13,550–13,834 | 10,498 / 476 → 4,944 / 310 |
| 10⁸ | plummer | rank0 | 80 | 37,143 | 14.5 | 48,874 | 6,928 + 3,942 + 1,181 = 12,051 | 4.03 | 12,417 | 13,259–13,393 | 10,508 / 312 → 4,916 / 210 |
| 10⁸ | plummer | rank0 | 160 | 36,925 | 11.7 | 48,981 | 6,954 + 4,433 + 924 = 12,312 | 2.42 | 12,504 | 13,345–13,412 | 10,485 / 167 → 4,900 / 138 |
| 10⁸ | clusters | rank0 | 40 | 35,995 | 21.2 | 48,086 | 6,946 + 3,938 + 849 = 11,732 | 6.07 | 12,433 | 13,502–13,700 | 10,477 / 464 → 4,940 / 300 |
| 10⁸ | clusters | rank0 | 80 | 37,235 | 13.4 | 48,920 | 6,926 + 3,937 + 1,175 = 12,038 | 3.39 | 12,372 | 13,234–13,349 | 10,518 / 311 → 4,921 / 208 |
| 10⁸ | clusters | rank0 | 160 | 36,921 | 11.7 | 48,934 | 6,909 + 4,427 + 918 = 12,254 | 2.05 | 12,420 | 13,296–13,346 | 10,494 / 163 → 4,904 / 134 |

With rlst 0.10.0 (T1s, jobs 295044 and 295538, the real `Octree::new`): N = 10⁷ cube 197–199 /
69 ms and Plummer 242–254 / 87–92 ms at 40 / 160 ranks; N = 10⁸ cube 2,672–2,700 / 675–679 ms
and Plummer 3,342–3,628 / 820–886 ms. T1's (rlst 0.9.0): 552–634 ms at 40 and 2,607–2,656 ms
at 160 (N = 10⁷), 6,140–7,317 and 9,656–10,300 ms (N = 10⁸); scale-out.md §7.1.

### 3.3 What the numbers say

- **The splitter fix and the sort removal, separated.** At 160 Kathleen ranks the cube's
  `Octree::new` went 2,607 → 69 ms with rlst #40 (T1s) and 69 → 30–31 ms now (N = 10⁷);
  9,656 → 675–679 → 283–289 ms (N = 10⁸). The first is the imbalance (four ranks receiving
  N/4 keys); the second is the sort itself: 24-byte records sorted locally twice, the
  splitter rounds (all-gathers of candidates that grow with P), and the second key move.
  Between the two replays the first sort (1,910 ms at 40 ranks, N = 10⁸) becomes 501 ms of
  routing, and the O1 rounds lose their own sort of the input keys (280 → 39 ms).
- **The second sort** cost 0.2–24 ms; the balance by the bounds that replaces it, with
  the second key move, 0.07–7 ms.
- **Every point on rank 0** is now bounded by rank 0's own work: its one local sort (350 ms
  for 2 × 10⁷ keys on locust, 6.9 s for 2 × 10⁸ on Kathleen) and its count of all keys per
  round (78–152 ms and 2.0–4.4 s), instead of a distributed sort of everything from rank 0
  (1.2–1.4 s and 36–37 s). The before replay takes 3.8–4.3 s at N = 10⁷ and 45–49 s at
  N = 10⁸ on Kathleen; the after replay 0.9–1.2 s and 10–12.7 s.
- **Received keys per rank** (max / mean) after the routing: 1.002–1.08 at the timed runs
  and up to 1.12 at 512–640 ranks, against `parsort`'s exact 1.00; the cut is by block, so
  the share is within O3's bound, and the pairs are distinct keys (half the records here,
  every key being passed twice).
- `Octree::new` differs from the after replay's sum of steps: it also builds the
  neighbour map, and it has no barrier between steps (so a rank's waits overlap); at 40
  Kathleen ranks with the Plummer sphere and the clusters its ranks spread by up to 0.3 s.

## 4. Collectives at 512 and 640 ranks (counted, not timed)

N = 10⁷, oversubscribed: locust 512 ranks (`--map-by :OVERSUBSCRIBE --bind-to none`),
Kathleen 640 ranks in job 415613 (160 per node, `--oversubscribe --map-by ppr:160:node
--bind-to none -x PSM2_SHAREDCONTEXTS=1`). Per rank, max over ranks, ranges over the
cube, the Plummer sphere and the clusters; the random share unless said. Collectives count
every call, rlst's validity all-reduces included (an rlst `all_to_allv` is 4 all-reduces,
the all-to-all of counts and the all-to-all-v); messages are the point-to-point sends to
other ranks with a non-zero count (all-gathers counted as P − 1).

| Step | collectives | messages | sent (MB) | received (MB) | P-long arrays (kB) | all-reduce payload (kB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **512, before:** first sort (`parsort`) | 37 | 3,577 | 0.96 | 1.82–1.83 | 16.4 | 294–299 |
| successor exchange | 5 | 511 | 0 | 0 | 2.56 | 0 |
| key move 1 | 6 | 2 | 0.010–0.019 | 0.013–0.019 | 4.1 | 0 |
| second sort | 27–32 | 2,207–2,939 | 0.046–0.118 | 0.65–0.78 | 12.3–14.3 | 205–234 |
| successor exchange | 5 | 511 | 0 | 0 | 2.56 | 0 |
| key move 2 | 6 | 3 | 0.006–0.013 | 0.005–0.018 | 4.1 | 0 |
| **512, after:** routing rounds | 6–9 | 0 | 0 | 0 | 0 | 175–262 |
| pairs (all-to-all-v) | 6 | 511 | 0.32 | 0.34–0.35 | 4.1 | 0 |
| key move 1 | 6 | 0–1 | 0–0.009 | 0–0.009 | 4.1 | 0 |
| balance by the bounds | 6 | 119–358 | 0.009–0.021 | 0.01–0.40 | 4.1 | 0 |
| **640, before:** first sort | 37 | 4,473 | 0.77 | 1.84–1.85 | 20.5 | 365–367 |
| successor exchange | 5 | 639 | 0 | 0 | 3.2 | 0 |
| key move 1 | 6 | 2 | 0.002–0.012 | 0.002–0.011 | 5.12 | 0 |
| second sort | 27–32 | 2,857–3,378 | 0.043–0.104 | 0.79–0.94 | 15.4–17.9 | 254–291 |
| successor exchange | 5 | 639 | 0 | 0 | 3.2 | 0 |
| key move 2 | 6 | 3 | 0.005–0.011 | 0.003–0.014 | 5.12 | 0 |
| **640, after:** routing rounds | 6–10 | 0 | 0 | 0 | 0 | 300–324 |
| pairs (all-to-all-v) | 6 | 639 | 0.26 | 0.26–0.27 | 5.12 | 0 |
| key move 1 | 6 | 0–2 | 0–0.008 | 0–0.008 | 5.12 | 0 |
| balance by the bounds | 6 | 143–417 | 0.009–0.021 | 0.011–0.45 | 5.12 | 0 |

The O1 rounds (an all-reduce of 2B values per round) and the ghost exchange are the same in
both. With every point on rank 0, rank 0 sends 480 MB of sort records before and 160 MB of
pairs after (2 × 10⁷ keys, 10⁷ distinct); the other columns are as above. What still grows
with P here: the all-to-all of counts in every all-to-all-v (the P-long column) and the
routing rounds' all-reduce of the provisional blocks (O(kP) values, as O1's), both
scale-out.md §7.2's S5 material; `parsort`'s candidate all-gathers, which grew with P too,
are gone.

## 5. Checks

M3 Max (sandboxed unless said otherwise): `cargo fmt --all`; `cargo clippy -p nd-octree
--all-targets --features strict -- -D warnings`; `RUSTDOCFLAGS="-D warnings" cargo doc
--no-deps -p nd-octree`; the root checks (`cargo fmt -- --check`, `cargo clippy -- -D
warnings`, `--examples`, `RUST_MIN_STACK=8388608 cargo test`, `cargo doc --no-deps`) and the
strict workspace checks (`cargo clippy --workspace --all-targets -- -D warnings`,
`RUST_MIN_STACK=8388608 cargo test --workspace`): all pass. MPI runs as in §2.3.

Kathleen (2026-10-12):

| Job | Nodes | What | Waited | Ran | Node-hours | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 415612 | 1 (node-c11b-003) | release and debug builds, the probe | 1 s | 2:29 | 0.04 | passed |
| 415614 | 2 (node-c11b-[003,007]) | `mpi.sbatch 80` (`test`) | 1 s | 5:46 | 0.19 | 9 of 9 pass |
| 415613 | 4 (node-c11a-[073,105,118,126]) | examples at 160 and 640, timed probe at 40, 80, 160, counts at 640 | 66 s | 1:23:12 | 5.55 | 46 of 46 launches pass |
| 422926 | 4 (node-c11a-[008,016,025,059]) | edge cases at 160 and 640 after the generator fix | 2 s | 22:58 | 1.53 | 2 of 2 pass |

7.31 node-hours in all, of T3a's 20 (README, "Node-hours").

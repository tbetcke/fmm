# The partition map and the locally essential key set (Phase 5S T3)

S1 in `nd-octree`, as docs/phase5s/T3-octree-partition-map.md asks and
docs/design/scale-out.md §3 (signed off) specifies: the partition replicated only as keys
(`PartitionMap`), and every rank's key set narrowed to its locally essential set H,
assembled by one point-to-point push instead of all-gathers. Run on 2026-10-12 on the M3
Max, locust (GH200) and Kathleen.

**Measured** means run here and read from the output; timings are reported, never asserted.
locust is a shared node: its load is in the driver's log before every launch. Kathleen
nodes are exclusive; every job is named.

## Summary

- **The held set is H, exactly, and contains everything the lists name.** On every rank,
  the octree's keys equal H = N(S) ∪ C(N(S)) computed by brute force from the gathered tree,
  with the design's kinds; every key a U-, V-, W- or X-row of S, the halo or a parent-child
  step names is held; H is a subset of the Phase 5 set, and on one rank the same map
  (`test_mpi_locally_essential`: M3 Max 1–8, locust 1–8 and 512 oversubscribed, Kathleen 80
  on 2 nodes and 640 oversubscribed on 4). The leaves, blocks, bounds and weights are Phase 5's key for key
  (T3a's reference copy at 1–8 on the final code), and `multi_rank` gives 0 differing values
  at 2, 4 and 8 ranks.
- **Held keys per rank no longer grow with P** (2.5 × 10⁴ points per rank, max over the
  ranks): from 40 to 640 Kathleen ranks H stays at 3,585–7,025 (cube), 6,569–9,177
  (Plummer) and 6,497–7,825 (clusters), at most 1.96 times its 40-rank value (the design's
  bound is 2), while the Phase 5 set grows 3,881 → 42,417, 11,561 → 50,377 and
  16,145 → 59,097. At 512 locust ranks: Phase 5 39,865 / 43,377 / 54,753 keys, H 3,793 /
  9,129 / 8,729, the figures T1 measured (scale-out.md §3.7).
- **`Octree::new`**, timed against `main`: 7–13% faster for the cube (1% on the 160-rank
  plateau of §4.3) and 3–11% for the clusters at 40–160 Kathleen ranks, the Plummer sphere within −4% to +4% (medians of 3–4
  launches; e.g. N = 10⁷ on 160 ranks: cube 29.7 → 27.5 ms, clusters 51.0 → 46.9 ms,
  Plummer 50.5 → 52.3 ms); on locust at 2–72 ranks the same or faster (N = 10⁷ on 8 ranks:
  14–21% faster), at most 0.1 ms slower.
- **Messages and bytes** of the key exchange at 512 (locust) and 640 (Kathleen)
  oversubscribed ranks, per rank: the two all-gathers of Phase 5 (2 (P − 1) messages,
  0.61–0.91 MB received) are gone; what remains is the leaf-bit all-gather-v (one byte per
  block, 15–33 kB) and the push (37–57 messages, 20–73 kB sent, about half the bytes of the
  Phase 5 halo all-to-all-v, which it replaces). 14 collectives become 7.
- **`nd-fmm-plan` until T4: an adapter** rebuilds the Phase 5 key set for `Plan::new`, so
  every plan, list and value stays Phase 5's (§1, item 5).
- **Found on the way:** the first version of the push spent 15 ms per rank on map lookups
  (8 Plummer ranks, N = 4 × 10⁶) and made `Octree::new` 5–20% slower on Kathleen; a fast
  path for cells inside the own range and an early-exit neighbour test fixed it (§4). And a
  launch effect at 160 Kathleen ranks that is not T3's (§4.3).

## 1. The change

`octree/src/octree/partition_map.rs` (new), `octree/src/octree/implementation.rs`
(`gather_leaf_blocks`, `generate_all_keys`), `octree/src/octree.rs` (`Octree::new`,
`Octree::partition_map`, the docs of `KeyType::Global`, `with_ghost_children`,
`with_block_refinement`, `all_keys` and `new`'s collectives).

1. **`PartitionMap`** (public, as scale-out.md §3.5 names it): every coarse block in
   Morton order, the cut (P + 1 offsets), one leaf bit per block, and the rank it
   classifies for. `owner(key)` (the block's rank at or below a block, the rank of the
   first descendant block above the blocks: S2's owner, design §11 T3 (a)),
   `sharers(key)` (the contiguous range of ranks with a block at or below the key),
   `top_kind(key)` (own block `LocalLeaf`/`LocalInterior`, shared ancestor `Global`,
   another rank's block `GhostLeaf`/`GhostInterior` of its owner, an ancestor of other
   ranks' blocks only `GhostInterior` of its owner, `None` below the blocks),
   `rank_blocks`, `is_leaf_block`, `block_owner`. Every lookup is one binary search over
   the blocks (on the key's first finest descendant) and one over the cut,
   O(log B + log P), without communication. Memory: 8 B per block, a bit per block and
   8 (P + 1) B for the cut (about 3 MB at 10⁴ ranks and 36 blocks per rank, *model*,
   §8.2). `coarse_tree_bounds` stays, computed from the same blocks and cut (the design
   keeps both), so `owner_rank`, `local_leaf` and `lookup_leaves` are unchanged; a unit
   test checks `owner` against the bounds.
2. **The leaf bits** are gathered once after the balance (`gather_leaf_blocks`): one
   all-gather-v of one flag per own block, whose counts every rank takes from the cut
   (no count exchange).
3. **`generate_all_keys` over H** = N(S) ∪ C(N(S)) (§3.1): S the own keys and the shared
   ancestors; the top-tree keys among the same-level neighbours of S and, with the
   ghost-children layer, the children of the ancestors among them, from the map; every
   other key of H by one push (all-to-all of counts and all-to-all-v): every rank sends
   each own key k, with its children when interior under the layer, to the owner of each
   same-level neighbour cell of k at or below a block and to the sharers (with blocks) of
   each top-tree neighbour cell (§3.3, step 2). A cell inside another rank's block may
   lie below a leaf there, so the sender cannot tell whether the receiver holds it in S;
   the receiver keeps a pushed key only if it lies in N(S) (it or a same-level
   neighbour in S) or is a child of a key of N(S), so H is exactly the design's set. The
   `gather_to_all` of the `Global` ancestors and the advertisement of every block to
   every rank are gone; the old path is removed, not kept behind an option (§11, T3 (b)).
4. **Kinds.** `Global` keeps its name and now means a shared ancestor; a rank without
   blocks holds no key; on one rank nothing changes (the whole tree, the root `Global`
   unless it is the only leaf, as in Phase 5; `top_kind` returns the same on one rank).
   The closing debug check of the ghost-children guarantee stays, on the returned map.
   Debug builds also check that every received key is owned by its sender
   (`PartitionMap::owner`) and never reclassifies a key.
5. **`nd-fmm-plan` until T4** (the brief: "the plan still built from the old path or an
   adapter if T4 has not landed (state which)"): **an adapter**. The Phase 5 plan
   resolves the lists of every `Global` key, including the ones that are not shared
   ancestors (P2 prunes their rows after resolving them), and requires every block held;
   H does not hold their neighbourhoods. So `Plan::new` rebuilds Phase 5's key set from
   the octree's leaves, blocks and bounds (`fmm-plan/src/phase5_keys.rs`, private,
   Phase 5's `generate_all_keys` without its debug checks) and plans over it: every list,
   exchange and value is Phase 5's. `tests/mpi_regressions.rs` reads the held keys from
   the plan's index instead of `octree.all_keys()`. No other plan code changed; T4 builds
   the plan over H ∪ T (§3.7) and removes the adapter.
6. **Examples.** New `test_mpi_locally_essential` (registered, and in
   `run-mpi-tests.sh` as decision 5 orders); `test_mpi_complete_tree` and
   `test_mpi_weighted_partition` check the new meaning of `Global` and the map's kinds
   instead of the replicated blocks; T3a's private copy of the Phase 5 construction
   (`examples/phase5_reference/`) is removed as scheduled, after the key-for-key runs of
   §2.2. `examples/phase5_keys/` keeps Phase 5's `generate_all_keys` for the comparison
   of the held sets.

## 2. Correctness

### 2.1 Unit tests (`nd-octree` lib, 71 pass)

- `test_partition_map_lookups_against_brute_force`: a 29-block coarse tree on levels 1–3,
  four cuts with ranks without blocks at the start, in the middle and at the end, every rank:
  the placement, owner, sharers and kind of every key of the tree and one level below every
  block against a scan of the blocks; owners are never ranks without blocks.
- `test_partition_map_owner_matches_the_bounds`: `owner` of the first and last finest keys
  of every block equals `owner_rank_for_key` on `tree_bins`, with empty ranks in the cut.
- `test_partition_map_root`: the root as the only block on one rank (`Global` unless a
  leaf) and on several (local to its owner, a ghost elsewhere); the root above the blocks
  (`Global` on the ranks with blocks, a ghost of the first block's rank on the others).
- `test_last_finest_descendant` (both corners, bit arithmetic, against the decoded index on
  every level of 200 random keys), and two `should_panic` tests of `PartitionMap::new`.
- `test_ghost_children_option_is_a_noop_on_one_rank` now runs the new `generate_all_keys`
  on one rank: the whole tree with the Phase 5 kinds, the option a no-op.

### 2.2 The MPI examples

`test_mpi_locally_essential` (new, registered; §1 item 6) builds six workloads (the cube, the
Plummer sphere, the clusters, a graded corner, a heavy block that leaves ranks without
blocks, the root as the only block), each with and without the ghost-children layer, and
checks on every rank against the gathered tree: the map (blocks, cut, leaf flags, and the
owner, sharers and kind of every tree key and of both finest corners of every leaf, the owner
also against `owner_rank`); `all_keys` equal to H with its kinds (a rank without blocks holds
nothing); every key named by S's rows (geometric definitions, with the layer), the halo of
the own keys and the parent-child steps held; against the Phase 5 key set
(`examples/phase5_keys/`, Phase 5's `generate_all_keys` on the same tree): H a subset with
the same kinds except the ancestors of other ranks' blocks only (`Global` before, ghosts of
their owner now), equal on one rank. A mutant that drops the pushed interior keys fails it
(Plummer, every rank).

Key for key against the Phase 5 construction: T3a's private copy (`phase5_reference`)
compared leaves, own blocks, gathered blocks, bounds and weights (no longer `all_keys`) in
`test_mpi_weighted_partition` and `test_mpi_construction_edge_cases` at 1, 2, 3, 4 and 8
ranks on the M3 Max with the final code (temporary copies of both examples, removed again),
and at 1–8 on locust and 80 on Kathleen (job 515095) with the first version (§4.1). The
copy is then removed from the tree, as T3a scheduled.

### 2.3 Runs (final code unless said)

| Where | Ranks | What | Result |
| --- | --- | --- | --- |
| M3 Max | 1, 2, 3, 4, 8 | the seven registered examples, debug (35 launches) | pass |
| M3 Max | 1, 2, 4, 8 | `run-mpi-tests.sh` (`mpi_regressions`, `mpi_exec`, `mpi_threading`, `invariance`, seven examples; 44 launches) | pass; the new example 11.0 / 6.5 / 4.7 / 3.8 s in debug |
| M3 Max | 2, 4, 8 | `multi_rank` (release, ignored) | pass; 0 values differ from the union in rank order in all 10 rows at each count |
| locust | 1, 2, 4, 8 | `run-mpi-tests.sh` (44 launches) | pass |
| locust | 3 | the seven examples, debug | pass |
| locust | 2, 4, 8 | `multi_rank` | pass; 0 values differ in every row |
| locust | 512, oversubscribed | `test_mpi_locally_essential` (release) | pass |
| Kathleen 518418 | 2 nodes | T2's correctness job: the MPI list at 2, 4, 8, 80 ranks; invariance at 80 against 40, 20, 10; the smoke run | 8 of 8 steps pass |
| Kathleen 518414 | 640 (4 × 160), oversubscribed | `test_mpi_locally_essential` (release) | pass (378 s) |
| locust, Kathleen 515095 and 515088 | as above | the same with the first version (before §4.1), plus the Phase 5 construction comparison | all pass |

The smoke run's comparison with the one-rank FMM (1.04e-15 relative L2, a random share, so
not bit for bit) is the figure T2 recorded.

## 3. Held keys per rank

The probe (throwaway, not committed, as T1's and T3a's): for a `scaling` workload
(`Workload::new(dist, n)`, the random share, the domain by `compute_global_bounding_box`,
every point's key twice as `Fmm` passes sources and targets, `max_level` 16, 64 points per
leaf, the ghost-children layer) it builds the octree and counts `all_keys` per rank; on the
same tree it replays Phase 5's `generate_all_keys` (the copy of `examples/phase5_keys`, with
counters) for the Phase 5 set and its messages, checks that H is a subset of it, and replays
the push's targets from the public `PartitionMap` for its messages and bytes. A copy of the
probe built in `git archive main` (06016d2) gives the "before" `Octree::new` and confirms its
`all_keys` equals the replayed Phase 5 set (8 M3 Max ranks, all three workloads).

2.5 × 10⁴ points per rank; held keys max / mean over the ranks, Phase 5 → H:

| Ranks | where | cube | Plummer | clusters |
| ---: | --- | --- | --- | --- |
| 2 | locust | 3,017 / 3,013 → 3,017 / 3,013 | 3,353 / 3,333 → 3,209 / 3,101 | 4,249 / 4,121 → 3,761 / 3,553 |
| 4 | locust | 1,961 / 1,961 → 1,961 / 1,961 | 4,713 / 4,443 → 4,345 / 3,905 | 5,801 / 5,499 → 4,249 / 3,819 |
| 8 | locust | 1,753 / 1,683 → 1,529 / 1,396 | 6,561 / 5,665 → 5,857 / 4,801 | 7,721 / 6,996 → 5,129 / 4,057 |
| 16 | locust | 5,753 / 4,893 → 5,505 / 4,568 | 8,065 / 7,681 → 6,017 / 5,100 | 10,057 / 9,528 → 5,993 / 4,294 |
| 32 | locust | 3,817 / 3,253 → 3,521 / 2,891 | 10,161 / 9,562 → 6,217 / 5,200 | 14,329 / 13,168 → 6,521 / 5,356 |
| 64 | locust | 6,697 / 6,169 → 2,889 / 2,085 | 14,345 / 13,040 → 7,177 / 5,894 | 18,129 / 17,279 → 7,217 / 5,456 |
| 72 | locust | 7,537 / 6,984 → 3,745 / 2,977 | 14,809 / 13,474 → 7,153 / 5,912 | 18,665 / 17,822 → 7,425 / 5,406 |
| 128 | locust, oversubscribed | 11,497 / 9,978 → 7,609 / 5,824 | 18,105 / 16,869 → 7,257 / 5,951 | 26,289 / 25,390 → 7,801 / 5,303 |
| 256 | locust, oversubscribed | 9,009 / 8,055 → 5,017 / 3,836 | 27,897 / 25,833 → 8,369 / 6,016 | 41,185 / 38,785 → 8,593 / 6,218 |
| 512 | locust, oversubscribed | 39,865 / 39,239 → 3,793 / 2,710 | 43,377 / 40,775 → 9,129 / 6,340 | 54,753 / 53,232 → 8,729 / 6,105 |
| 40 | Kathleen, 1 node | 3,881 / 3,317 → 3,585 / 2,972 | 11,561 / 10,700 → 6,569 / 5,548 | 16,145 / 14,712 → 6,497 / 5,324 |
| 80 | Kathleen, 2 nodes | 9,361 / 8,329 → 5,537 / 4,295 | 15,497 / 14,121 → 7,433 / 6,024 | 19,121 / 18,145 → 6,905 / 5,270 |
| 160 | Kathleen, 4 nodes | 10,977 / 9,771 → 7,025 / 5,624 | 20,105 / 18,598 → 7,425 / 5,855 | 31,265 / 29,289 → 7,569 / 5,593 |
| 320 | Kathleen, oversubscribed | 8,737 / 7,888 → 4,713 / 3,664 | 32,561 / 29,922 → 8,889 / 6,148 | 45,305 / 43,223 → 8,409 / 6,322 |
| 640 | Kathleen, oversubscribed | 42,417 / 41,470 → 6,273 / 4,958 | 50,377 / 47,859 → 9,177 / 6,359 | 59,097 / 57,442 → 7,825 / 5,896 |

(Kathleen job 518414; locust 2026-10-12, the final run. The first version of job 515088
gave the same counts.)

- **Against T1's bound** (scale-out.md §3.1: at fixed N/P, H's maximum at most twice its
  40-rank value up to 640 Kathleen ranks): the largest ratio is 1.96 (cube at 160 ranks,
  7,025 against 3,585); Plummer 1.40 and clusters 1.20, both at 640. On locust, which has no
  40-rank run, H's maximum from 8 to 512 ranks grows 2.5× (cube), 1.56× (Plummer) and
  1.70× (clusters), T1's 1.5–2.5×; the cube's H moves with the level of its uniform blocks
  rather than with P (1,529 at 8, 7,609 at 128, 3,793 at 512), and the 128-rank value is
  2.16 times the 32-rank one. The Phase 5 set grows 10.4× (cube), 4.3× and 3.8× from 32 to
  512 locust ranks and 10.9×, 4.4× and 3.7× from 40 to 640 Kathleen ranks.
- **The split** at 640 Kathleen ranks, max per rank: own 1,779 / 2,540 / 2,420, shared
  ancestors 15 / 15 / 64, ghosts 4,746 / 6,909 / 5,941. Phase 5's busiest rank held 20–24
  times the most own keys of any rank.
- **The partition map**, the replicated part that stays: 32,768 / 17,963 / 19,440 blocks at
  640 ranks (9 B per block with its leaf bit: 0.16–0.29 MB per rank, *as measured*;
  T1's model 36 blocks per rank).
- **T4 still holds the top-tree keys** (every block and ancestor of one, design §3.7) until
  T5; T4 reports H ∪ T.

## 4. `Octree::new` before and after

The slowest rank's time, release, the third of three builds in one launch (a barrier before
each), the random share; "before" is the probe built in `git archive main`. Medians over
the launches, with the range where the launches disagree.

### 4.1 The first version was slower, and why

Job 515088 ran the first version: `Octree::new` 5–20% slower than `main` on Kathleen (for
example the Plummer sphere at N = 10⁷ on 40 ranks: 127–140 → 150–164 ms). Timed by step on
8 M3 Max ranks (Plummer, N = 4 × 10⁶): the new key step took 25 ms against the old 21 ms, 15
of them in the loop that finds the push targets (a binary search of the map per neighbour
cell, with the first finest descendant computed by decoding and encoding the key) and 4.6 in
the receiver's filter (26 neighbours built per received key). Two changes, no change of a
key: a neighbour cell deeper than every block whose first finest descendant lies in the
rank's own range is an own cell and skips the map; `first_finest_descendant` and
`last_finest_descendant` became bit operations (the position bits of a key are its
anchor's); the filter tests the parent first and stops at the first neighbour in S. The key
step became 10.6 ms (8 M3 Max ranks) and `Octree::new` 97–99 → 83–85 ms there (`main`:
93–94 ms; the cube 92 → 80 ms, the clusters 104 → 84 ms).

### 4.2 locust (one node, Open MPI 5.0.10, `--map-by core --bind-to core`)

One launch per row. The node had no other user's process; the 1-minute load before each
launch (42–50) came from this sequence's own preceding launches, run one at a time.

| N | ranks | cube | Plummer | clusters |
| ---: | ---: | --- | --- | --- |
| 5 × 10⁴ | 2 | 4.0 → 3.3 | 4.9 → 4.5 | 6.0 → 5.5 |
| 10⁵ | 4 | 3.0 → 2.7 | 5.2 → 4.9 | 6.3 → 5.8 |
| 2 × 10⁵ | 8 | 3.1 → 2.7 | 6.3 → 5.9 | 6.8 → 6.7 |
| 4 × 10⁵ | 16 | 5.9 → 5.0 | 6.6 → 6.2 | 7.8 → 7.9 |
| 8 × 10⁵ | 32 | 4.7 → 4.1 | 7.2 → 7.3 | 8.7 → 8.7 |
| 1.6 × 10⁶ | 64 | 5.1 → 4.6 | 9.2 → 8.6 | 10.9 → 10.3 |
| 1.8 × 10⁶ | 72 | 6.5 → 5.7 | 10.1 → 8.9 | 11.9 → 10.7 |
| 10⁷ | 8 | 174.7 → 150.5 | 294.6 → 236.8 | 311.4 → 244.9 |
| 10⁷ | 32 | 49.4 → 44.2 | 83.3 → 80.5 | 84.1 → 73.3 |
| 10⁷ | 72 | 30.0 → 27.1 | 50.5 → 50.3 | 50.8 → 40.6 |

(ms, slowest rank, before → after.)

### 4.3 Kathleen (jobs 518414 and 519325, 40 ranks per node, `--map-by ppr:20:socket:PE=1 --bind-to core`, Open MPI 4.1.6 over PSM2, `PSM2_TID=0`)

| N | ranks | cube | Plummer | clusters |
| ---: | ---: | --- | --- | --- |
| 10⁶ | 40 | 11.0 → 9.7 | 16.3 → 17.0 | 18.7 → 18.1 |
| 2 × 10⁶ | 80 | 16.8 → 14.6 | 20.0 → 19.2 | 22.6 → 21.6 |
| 4 × 10⁶ | 160 | 178.2 → 175.9 | 185.2 → 184.0 | 30.8 / 187.5–190.9 → 26.5 |
| 10⁷ | 40 | 81.6 → 76.1 | 134.8 → 135.1 | 129.3 → 117.7 |
| 10⁷ | 80 | 50.4 → 47.1 | 78.5 → 78.3 | 78.1 → 69.6 |
| 10⁷ | 160 | 29.7 → 27.5 | 50.5 → 52.3 | 51.0 → 46.9 |

(ms, slowest rank, medians of 3 (fixed N/P) and 4 (N = 10⁷) launches, before → after; the
Plummer sphere and the clusters spread their ranks by up to 13 ms within a launch.)

**An effect at 160 ranks that is not T3's.** At N = 4 × 10⁶ on 160 ranks the cube and the
Plummer sphere take 175–186 ms in every launch of both builds, against 26–31 ms for the
clusters and for N = 10⁷ (which has 2.5 times the keys); single launches of both builds hit
the same plateau elsewhere (the clusters before, 187–191 ms in 2 of 3; the cube after at
N = 10⁷, 185 ms in 1 of 4). A fixed delay of about 150 ms in some collective, independent
of the key set; not investigated here (T8 measures the construction's collectives).

At 320–640 oversubscribed ranks `Octree::new` is not timed.

## 5. Messages and bytes of the key exchange (counted, not timed)

Per rank, max over the ranks, ranges over the cube, the Plummer sphere and the clusters,
2.5 × 10⁴ points per rank: locust 512 ranks (`--map-by :OVERSUBSCRIBE --bind-to none`),
Kathleen 640 (job 518414, 160 per node, `--oversubscribe --map-by ppr:160:node --bind-to
none -x PSM2_SHAREDCONTEXTS=1`). Messages count point-to-point sends to other ranks with a
non-zero count, all-gathers as P − 1; bytes are MPI type sizes (a key 8 B; Phase 5's record
of key, rank and leaf flag 17 B; the pushed key with its leaf flag 9 B; a leaf flag 1 B).
Every all-to-all-v adds its all-to-all of P counts and rlst's four validity all-reduces;
every rlst all-gather an all-gather of P counts and two all-reduces.

| Step | collectives | messages | sent | received |
| --- | ---: | ---: | --- | --- |
| **512, Phase 5:** all-gather of the `Global` ancestors, all-gather of the advertised keys | 8 | 1,022 | own share | 0.62–0.83 MB |
| halo (all-to-all-v) | 6 | 37–50 | 38–100 kB | 38–88 kB |
| **512, T3:** leaf flags (all-gather-v, counts from the cut) | 1 | 511 | own blocks | 14.6–18.4 kB |
| push (all-to-all-v) | 6 | 37–57 | 21–73 kB | 20–47 kB |
| **640, Phase 5:** the two all-gathers | 8 | 1,278 | own share | 0.61–0.91 MB |
| halo (all-to-all-v) | 6 | 40–52 | 60–95 kB | 63–88 kB |
| **640, T3:** leaf flags | 1 | 639 | own blocks | 18.0–32.8 kB |
| push | 6 | 40–55 | 32–66 kB | 33–47 kB |

What still grows with P here: the leaf flags (one byte per block, B ≈ 28–51 per rank at
640) and the push's all-to-all of P counts (S5's material, scale-out.md §7.2). The pushed
keys are the ones H needs below other ranks' blocks plus the cells that turn out to lie
below a receiver's leaf (dropped by its filter): received 2,250–5,203 keys per rank at 512,
against H's 3,071–6,829 ghosts, the rest of which come from the map.

## 6. Checks

M3 Max (sandboxed): `cargo fmt --all`; `cargo clippy -p nd-octree --all-targets --features
strict -- -D warnings`; `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p nd-octree`;
`cargo build -p nd-octree --examples`; the root checks (`cargo fmt -- --check`, `cargo
clippy -- -D warnings`, `cargo clippy --examples -- -D warnings`, `RUST_MIN_STACK=8388608
cargo test`, `cargo doc --no-deps`) and the stricter workspace checks (`cargo clippy
--workspace --all-targets -- -D warnings`, `RUST_MIN_STACK=8388608 cargo test
--workspace`): all pass on the final code (`cargo test` 718 passed, `--workspace` 763
passed, 0 failed). MPI runs as in §2.3.

Kathleen (2026-10-12):

| Job | Nodes | What | Waited | Ran | Node-hours | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 515088 | 4 (node-c11a-[048,073,087,142]) | first version: probe builds, timed 40–160, N = 10⁷, counts at 320 and 640, the example at 640 | 31 s | 11:44 | 0.78 | 5 of 5 steps pass |
| 515095 | 2 (node-c11b-[011-012]) | first version: T2's correctness job (`test`) | 0 s | 22:14 | 0.74 | 8 of 8 pass |
| 518414 | 4 (node-c11a-[008,135,142,144]) | final: as 515088, the timed sweep twice | 2 s | 11:33 | 0.77 | 6 of 6 pass |
| 518418 | 2 (node-c11b-[029,042]) | final: T2's correctness job (`test`) | 0 s | 22:09 | 0.74 | 8 of 8 pass |
| 519325 | 4 (node-c11a-[005,043,068,074]) | final: N = 10⁷ three more times, the timed sweep once more | 2 s | 1:33 | 0.10 | 4 of 4 pass |

3.13 node-hours in all, of T3's 20 (README, "Node-hours").


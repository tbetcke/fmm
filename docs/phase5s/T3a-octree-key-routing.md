# Phase 5S / T3a — keys routed by weight instead of sample sorts in `nd-octree` (C5S.2; S4)

Added on 2026-10-11 at the user's request, from T1's finding (`docs/design/scale-out.md`
§1.4 and §7.1). distributed-fmm.md §14.4 kept S4 in reserve, to be built only if measured
at 1,000 or more ranks to need it, and T8 was to decide. T1 measured that it is needed at
8 ranks already.

`Octree::new` (octree/src/octree.rs; octree/src/octree/implementation.rs) sorts every
input key globally twice with rlst 0.9.0's sample sort `parallel_sort::parsort`:
- in `linearize`, before the coarse tree;
- inside `balance`, on the 2:1 closure of the refined tree.

**The first sort's splitters fail on the random share.** `parsort` takes 4 samples per
rank, the first key of each local quarter. When every rank's keys span the whole domain,
every rank's samples sit at the same four quartiles, and four ranks (those near P/4, P/2,
3P/4 and P) receive a quarter of all keys each. Measured (T1, locust, N = 10⁷): the
heaviest rank received P/4 times its share at 8–512 ranks. The sort stayed at 170–240 ms
from 8 to 72 ranks, 72–79% of `Octree::new`, and the heavy rank's peak was 241 MB against
a mean of 59 MB. This is T0's 2 GB on four ranks at N = 10⁸ (S0.5).

rlst PR #40 (codeberg rlst/rlst, histogram sort with sampling) balances `parsort`
exactly. Measured against rlst 0.9.0 in the same octree build at 72 ranks: the first sort
from 186 to 62 ms, `Octree::new` from 238 to 84 ms, the heavy rank's peak from 241 to
72 MB. But the octree needs no general sort at all: O1 already counts keys per block with
one all-reduce per round, which is a histogram sort specialised to Morton keys. This task
removes both global sorts, so `nd-octree` does not depend on rlst's splitters, and drops
the sort's other costs: 24-byte unique records, a full all-to-all-v of every key, the
`communicate_back` rounds, and a second key move.

This task builds S4 as `docs/design/scale-out.md` §7.1 specifies.

Read first:
- octree/CLAUDE.md (construction contract, MPI examples), root CLAUDE.md ("MPI");
- `docs/design/scale-out.md` §1.4, §7.1 and §8.1 (signed off); distributed-fmm.md §3.2
  (O1–O4; the T0 condition: every block is a node of the one-rank tree) and §14.4 (S4);
- the code: `Octree::new`, `linearize`, `compute_coarse_tree`, `global_block_counts`,
  `refine_heavy_blocks`, `local_coarse_tree_weights`, `partition_blocks`, `tree_bins`,
  `redistribute_with_respect_to_coarse_tree`, `create_local_tree`, `balance`,
  `deepest_level`, the debug checks (`is_linear_tree`, `is_complete_linear_tree`), and the
  tests at the end of implementation.rs (`serial_coarse_tree`,
  `test_coarse_tree_by_weight_keeps_the_one_rank_leaves`); the MPI examples, in particular
  `test_mpi_weighted_partition` and `test_mpi_construction_edge_cases`;
- rlst 0.9.0's `parallel_sort.rs`, `array_tools.rs` (`all_to_allv`, `communicate_back`) from
  the registry.

Do:
1. **One local sort.** Sort each rank's input keys once (duplicates kept) and reuse them
   for every count below. Today `compute_coarse_tree` sorts its own copy, `parsort` sorts
   again, and the received keys are sorted a third time.
2. **Route the keys by weight, without a sort** (replaces `linearize`'s `parsort` and
   `communicate_back`):
   - starting from the root, count each rank's keys per provisional block in one merge
     pass, sum the counts with one all-reduce per round, and split every block heavier
     than W/(k P) and above `max_level`. This is O1's round with the weight rule alone:
     no distinct-key rule, since distinct counts need the keys deduplicated first;
   - cut the provisional blocks at the nearest boundaries (`partition_blocks`);
   - send each rank's keys to the provisional owners, deduplicated locally and sent as
     (key, multiplicity) pairs: one all-to-all of counts and one all-to-all-v;
   - on each rank, merge the received sorted runs and sum the multiplicities. Every copy
     of a key is then on one rank, and the distinct keys are linearized: the input keys
     are all on the deepest level, so linearizing is deduplication.
3. **O1 unchanged on the routed keys.** `compute_coarse_tree` counts distinct keys and
   weights from the routed keys and their multiplicities (O2's weight with duplicates).
   Its blocks, its weights and the cut are what the Phase 5 code computes, bit for bit,
   because they depend only on the global counts. The key move to the O1 owners stays (an
   all-to-all-v), now mostly local, because both partitions are by weight.
4. **Balance without a sort** (replaces `balance`'s `linearize` and the second key move):
   - route each closure key to the rank whose partition range holds its Morton key
     (`sort_to_bins` on `coarse_tree_bounds`, one all-to-all-v);
   - linearize locally.

   Why it is exact: a closure key coarser than the blocks lies in the range of the rank
   that owns its first descendant block, whose leaves descend from it, so the key is
   removed locally as the global linearization removes it. A key at or below a block
   lies in that block's range with all its descendants. Duplicates from several ranks
   land on one rank. The result is already distributed by the bounds, so the second
   `redistribute_with_respect_to_coarse_tree` goes.
5. **The debug checks.** `is_linear_tree` and `is_complete_linear_tree` stay as they are.
   Add a debug check that the routed keys are distinct across ranks: no rank's last key
   equals the next rank's first, through the existing `communicate_back`, in debug only.
6. **Docs.** Update `Octree::new`'s collective list and octree/CLAUDE.md's construction
   contract. The leaves, the coarse tree, O1–O4 and every key are unchanged; only the
   construction's communication changes.

Tests that define done:
- **Key for key**, on every workload of `test_mpi_weighted_partition` (and the clusters):
  the leaves, the coarse blocks with their weights, the partition bounds and `all_keys`
  (with every `KeyType`) equal those of the Phase 5 construction. The Phase 5 path stays
  in the MPI example as a private reference copy (as T1's probe did) until T3 lands, and
  is then removed. Run at 1, 2, 3, 4 and 8 ranks on the M3 Max and locust, at 512
  oversubscribed on locust, and (Kathleen) at 160 ranks and 1,280 oversubscribed.
- Unit tests:
  - the routing rounds on hand-made counts (heavy blocks, `max_level`, a single block);
  - the merge of (key, multiplicity) runs (duplicates across runs, empty runs);
  - the local balance linearization against `morton::balance` on gathered keys.
- The registered `nd-octree` MPI examples pass at 1, 2, 3, 4 and 8 ranks, and
  `test_mpi_construction_edge_cases` at 1–8, including a rank without keys, every key on
  one rank, all keys identical, and a tree whose only block is the root.
- `mpi_regressions`, `mpi_exec` and `mpi_threading` pass unchanged at 1, 2, 4 and 8
  ranks (every `Fmm` output bit for bit, since the tree is the same), and `multi_rank`
  (release, ignored) at 2, 4 and 8.

Measured (release; reported, not asserted), before and after, on the cube, the Plummer
sphere and the clusters with the random share and with every point on rank 0:
- timed: `Octree::new` by step (the steps of T1's probe, which T1 left in the session
  scratch; a timed build of the octree alone is enough), keys per rank after each move
  (max/mean), the peak resident size per rank:
  - at 8–72 ranks on locust (N = 10⁶–10⁷);
  - (Kathleen) at 40, 80 and 160 ranks (N = 10⁷, 10⁸);
- counted, not timed: the messages and bytes of each collective of `Octree::new` at 512
  oversubscribed ranks on locust and (Kathleen) 1,280;
- "before" is rlst 0.10.0 (T1s), whose `parsort` already balances exactly; T1's probe
  numbers with rlst 0.9.0 are quoted beside it, so the report separates the splitter fix
  from the removal of the sorts.

Must pass:
- `cargo fmt --all`;
- `cargo clippy -p nd-octree --all-targets --features strict -- -D warnings`;
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p nd-octree`;
- the root checks and the stricter workspace checks;
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks; the six registered examples at 1–8
  ranks by hand under `timeout`;
- (Kathleen) T2's correctness job at 2 nodes, if T2 has merged; otherwise `mpi.sbatch` at
  80 ranks on 2 nodes.

Do not:
- change the leaves, the coarse tree, O1–O4, the cut or any key (T3 changes the key set,
  T7 the cut);
- add a collective that some ranks skip, or an all-gather whose size grows with P;
- change the rlst pin (T1s moved it to 0.10.0);
- change `nd-fmm-plan` or `nd-fmm-exec`;
- store points in the octree.

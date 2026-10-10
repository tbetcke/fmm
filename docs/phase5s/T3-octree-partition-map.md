# Phase 5S / T3 — the partition map and the locally essential key set in `nd-octree` (C5S.2; S1, part 1)

Today every rank of `nd-octree` learns the top of the tree by all-gathers. In
`generate_all_keys` (octree/src/octree/implementation.rs):
- the ancestors of every rank's coarse blocks are gathered to every rank
  (`gather_to_all`) and inserted as `KeyType::Global`;
- keys next to a `Global` key are advertised to every rank (`send_to_all`);
- with the ghost-children layer (`OctreeOptions::with_ghost_children`, which the FMM
  uses), every rank's coarse blocks are advertised to all ranks, and an advertised
  interior key travels with its eight children.

So every rank holds every `Global` box, every coarse block and their neighbourhoods. At
512 ranks on the N = 10⁶ cube every rank held all 37,449 boxes of the tree (Phase 5 T10,
phase5-gh200.md §10), and at 8,192 ranks the model of distributed-fmm.md §14.2 gives about
1.2 million replicated held boxes per rank.

This task builds the `nd-octree` half of S1 as `docs/design/scale-out.md` §3 (T1, signed
off) specifies:
- the partition replicated only as **keys**: the block list with owners (the partition
  map), from which any rank computes the owner and the kind of any top-tree key locally;
- the **locally essential key set** of each rank, assembled by sparse advertisements to
  the owners the map names instead of all-gathers.

T4 builds the plan on it. The leaves, the partition and every key a list names stay the
same, so T4 can keep every output bit.

Read first:
- octree/CLAUDE.md (MPI examples, collectives, tests), root CLAUDE.md ("MPI");
- docs/phase5s/README.md (requirements 2, 4, 6, 7, 9) and `docs/design/scale-out.md` §3
  and §8 (signed off), and its sign-off decisions on `KeyType::Global` and the
  ghost-children layer;
- distributed-fmm.md §3.2 (O1–O4), §14.2 and §14.4 (S1);
- the code: `generate_all_keys` (the global-key gather, `advertise`, `send_to_all`,
  `advertised_blocks`, the closing ghost-children check), `compute_coarse_tree`,
  `partition_blocks`, `tree_bins`, `owner_rank_for_key`, `coarse_tree_bounds`,
  `compute_neighbours`, `Octree::new` and `OctreeOptions` (octree/src/octree.rs), the
  tests at the end of implementation.rs (for example
  `test_ghost_children_option_is_a_noop_on_one_rank`), and the MPI examples in
  octree/examples (`test_mpi_complete_tree`, `test_mpi_leaf_lookup`,
  `test_mpi_construction_edge_cases`, `test_mpi_weighted_partition`, …);
- the `nd-fmm-plan` code that reads the octree's keys (`Plan::new`, `index.rs`), to see what
  T4 needs from this task.

Do:
- **The partition map.** A public type (name as signed off) holding the coarse blocks in
  Morton order with their owner ranks, replicated on every rank, with:
  - the owner of any key at or below a block, and of any key above the blocks (its first
    descendant block's owner, the S2 owner rule, if T1 places that rule here);
  - the kind of any top-tree key relative to this rank (own, a shared ancestor of an own
    block, another rank's), computed locally without communication;
  - its memory per rank documented (16 B per block, about 2 MB at 8,192 ranks, *model*),
    and a lookup cost of O(log blocks).
  It replaces the per-rank lower bounds (`coarse_tree_bounds`) only if the design says so;
  otherwise both stay, consistent.
- **The locally essential key set.** `generate_all_keys`, or its successor, returns for each
  rank exactly the set the design names (own keys and halo, the ancestors of own blocks,
  the keys their V, W and X rows can name, with the children a list needs), with the
  `KeyType` (or its renamed successor) of each:
  - no `gather_to_all` of `Global` keys and no advertisement of every block to all ranks;
  - the advertisements go point-to-point to the owners the partition map names, through
    the collectives the design lists (a count all-to-all and an all-to-all-v until S5, or
    the sparse exchange if T8 replaces them later);
  - empty ranks take part in every collective;
  - the ghost-children guarantee (or its replacement) still checked on the returned map,
    as the closing check does today.
- **One rank and the old behaviour.** On one rank nothing changes. Keep the old path behind
  an option only if the design asks for it (for example to compare held sets in tests);
  otherwise remove it.
- **Tests** (octree/CLAUDE.md): unit tests of the map's lookups (owners and kinds at
  block boundaries, above the blocks, the root, empty ranks); an MPI example that checks,
  on every rank, that the new set contains every key the old `generate_all_keys` returned
  that a V, W or X row, a halo or a parent-child step from an own key can name (computed
  by brute force from the gathered tree in the example, at small N), and reports the
  held-set sizes old against new. Registered with templated-examples as the other MPI
  examples are.
- **Measured**: held keys per rank old against new on the cube, the Plummer sphere and the
  clusters at fixed N/P from 2 to 512 ranks on locust (oversubscribed above 72, not
  timed) and (Kathleen) from 40 to 160 ranks on 1–4 nodes and to 1,280 oversubscribed on
  the 4 nodes; `Octree::new`'s time with and without the change, timed to 72 ranks on
  locust and 160 on Kathleen (release, reported, not asserted), and the messages and bytes
  of its collectives on the oversubscribed runs. The plan itself does not run on the new
  set until T4.

Tests that define done:
- The `nd-octree` unit tests and MPI examples pass at 1, 2, 3, 4 and 8 ranks (M3 Max and
  locust) and (Kathleen) at 80 ranks on 2 nodes.
- On every rank the new set contains every key the lists can name (the brute-force check)
  and the leaves and the partition are unchanged against the Phase 5 octree (compared key
  by key).
- The held keys per rank no longer grow with P at fixed N/P beyond T1's bound (measured
  table), whereas the Phase 5 set does.
- The Phase 5 suites still pass: `mpi_regressions`, `mpi_exec`, `mpi_threading`,
  `multi_rank` (release, ignored) at 1, 2, 4 and 8 ranks, with the plan still built from
  the old path or an adapter if T4 has not landed (state which).

Must pass:
- `cargo fmt --all`; `cargo clippy -p nd-octree --all-targets -- -D warnings`; the root
  checks and the stricter workspace checks;
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks; the six registered `nd-octree` MPI
  examples at 1–8 ranks by hand under `timeout`;
- (Kathleen) the correctness job of T2 at 2 nodes.

Do not:
- change the leaves, the coarse tree, O1–O4 or the cut (T7 changes the cut);
- add a collective that only some ranks enter, or an all-gather whose size grows with P
  beyond the partition map;
- store points in the octree (octree/CLAUDE.md);
- change `nd-fmm-plan` beyond what keeps it compiling (T4 builds on this).

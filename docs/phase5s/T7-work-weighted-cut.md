# Phase 5S / T7 — the work-weighted cut (C5S.4; S3)

`nd-octree` cuts the coarse blocks among the ranks by points (O2, O3; Phase 5 T4). On one
node that balanced the work at up to 8 ranks. At 72 ranks the points stay within 1.07 of
the mean, but the M2L work does not: the V pairs' max over mean is 1.17 (cube), 1.39
(Plummer) and 1.29 (clusters), and the compute stages follow (1.17–1.26; Phase 5 T10,
phase5-gh200.md §8). distributed-fmm.md §3.6 modelled it within 0.12 (1.174, 1.511,
1.328), and it grows with P at fixed N. The same section's simulation of a 50/50 mix of
points and M2L work reached 1.10–1.20 on both at N = 10⁶ on 72 ranks.

Because O1's leaves do not depend on the cut, a second cut is possible after the local
refinement without changing a bit (distributed-fmm.md §14.4, S3):
- each rank refines its blocks (`create_local_tree`, no communication);
- each block is weighed by a cost model (points for P2P, P2M and L2P; boxes for M2L, M2M
  and L2L);
- the ranks agree the cut again by the nearest-boundary rule;
- the blocks move once more (an all-to-all-v of keys).

This task builds it as `docs/design/scale-out.md` §6 (T1, signed off) specifies, with the
cost model's constants taken from per-kind timings per machine, precision and p.

Read first:
- octree/CLAUDE.md, fmm-exec/CLAUDE.md, root CLAUDE.md;
- docs/phase5s/README.md (requirements 2, 4, 6) and `docs/design/scale-out.md` §6 and its
  decision 3 (the model, the target, the default);
- distributed-fmm.md §3.5 and §3.6 ("The cut by work"), §14.4 (S3), §15.2;
- the code: `compute_coarse_tree`, `local_coarse_tree_weights`, `partition_blocks`,
  `redistribute_with_respect_to_coarse_tree`, `create_local_tree`, `balance`, `tree_bins`
  (octree/src/octree/implementation.rs), `Octree::new` and `OctreeOptions` and the weight
  O2 added; T3's partition map; `nd_fmm_exec::fmm::build` (where the octree is built and
  the redistribution follows); `FmmBuilder::kind_timings`, `KindTiming`,
  `StageTimings::kinds` (Phase 4S T5); `nd_fmm_validate::scaling` (the per-rank work and
  the compute imbalance).

Do:
- **The cost model.** A per-block cost from counts each rank has after the local
  refinement: points and leaves (P2P, P2M, L2P, and the near-pair count if cheap to get),
  boxes per level (M2M, L2L) and V-row lengths (M2L), with constants per machine,
  precision, p and strategy. Derive the constants from per-kind timings (a small
  calibration run, `kind_timings(KindTiming::Synchronous)`), check the model against the
  measured per-rank stage times of Phase 5 T10 (locust, 8 and 72 ranks) and (Kathleen)
  of 40, 80 and 160 ranks, and report the fit. Where the constants live (a table in
  `nd-fmm-exec`, an option, a measured file) as the design decides; `nd-octree` receives
  only weights (it never knows the kernel).
- **The second cut in `nd-octree`.** After the local refinement, weighed blocks, the
  nearest-boundary cut agreed on every rank, the blocks' keys (and so their points, by
  the redistribution that follows in `Fmm::build`) moved once. The leaves and every key
  stay the same; only ownership changes. API: an option or a weight input as signed off;
  one rank unchanged; empty ranks allowed (O4).
- **`Fmm`.** An option (default as decision 3 says) that passes the weights; the partition
  map (T3) and the plan (T4) built after the second cut.
- **Tests.** Unit tests of the weighted cut (heavy blocks, ties, empty ranks); an MPI
  example in `nd-octree` showing the leaves unchanged against the first cut, key by key;
  in `nd-fmm-exec`, the one-rank comparison and T2's invariance check with the option on
  (bit for bit: only the partition changes).
- **Measured** (release; reported, not asserted), first cut against second, on the cube,
  the Plummer sphere and the clusters:
  - timed: the compute stages' max over mean, the build time added and the evaluation
    time at 8 and 72 ranks on locust (N = 10⁶–10⁷) and (Kathleen) at 40, 80 and 160 ranks,
    40 × 1 and 2 × 20 per node (N = 10⁷, 10⁸);
  - counted, not timed: per-rank points, leaves, V pairs and the modelled cost (counts
    times the fitted constants) at 512 ranks on locust and 640 and 1,280 on Kathleen's 4
    nodes, oversubscribed (N = 10⁶–10⁷).

Tests that define done:
- With the second cut on: the output bit for bit the one-rank `Fmm` in rank order
  (`mpi_exec`, `multi_rank`) at 1, 2, 4 and 8 ranks, and T2's invariance check at 1, 2, 4
  ranks, at 512 against 64 on locust and (Kathleen) at 160 against 20.
- The leaves equal the first cut's on every workload (the `nd-octree` example).
- The modelled per-rank cost's max over mean at 512–1,280 oversubscribed ranks within the
  signed-off target (distributed-fmm.md §14.5: ≤ 1.10 at 1,024 ranks) on the three
  workloads, and the measured compute stages at 72 and 160 ranks improved against the
  first cut, or a report of how far it stays and why (then the target is a sign-off
  question again, not silently missed). Timed balance beyond 160 ranks stays open.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks; the `nd-octree` MPI examples at 1–8
  ranks by hand;
- (Kathleen) the correctness job of T2 at 2 nodes.

Do not:
- change the leaves or O1's coarse tree; only the cut and ownership change;
- put kernel knowledge or timings into `nd-octree` (it receives weights);
- make the cut depend on timings taken during the run (the constants are fixed inputs, so
  the partition is deterministic);
- assert timings.

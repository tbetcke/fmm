# Phase 5S / T8 — construction and collectives at scale, conditional (C5S.5; S4, S5)

distributed-fmm.md §14.4 keeps two packages in reserve, to be built only if measured to
need them at 1,000 or more ranks:
- **S4, construction at scale** (`nd-octree`). `Octree::new` sorts keys with the sample
  sort `parallel_sort::parsort` (4 samples per rank) in `linearize` and in `balance`
  (octree/src/octree/implementation.rs). Each sort balances its ranges only roughly, and
  its all-to-all is O(P). The alternative routes the input keys straight to the owners of
  O1's blocks (the cut needs only global counts per block, which local counting and one
  all-reduce per round give), then deduplicates per block; both sample sorts go.
- **S5, communication primitives at scale.** The O(P) count all-to-alls of
  `Redistribution` (fmm-plan/src/redistribute.rs) and of rlst 0.9.0's ghost-communicator
  builds (32 kB of counts per rank at 8,192 ranks, *model*), and T3's advertisements if
  they use a count all-to-all, are replaced by a sparse dynamic exchange (non-blocking
  consensus: synchronous sends, then a non-blocking barrier). rsmpi 0.8.2 has both
  (`immediate_synchronous_send`, `immediate_barrier`), so S5 needs no rsmpi addition;
  confirm it, together with the probe calls a receiver needs.

No job in this phase is larger than 4 nodes, so nothing can be timed at 1,000 ranks. T1
states each trigger on what can be measured (decision 4 of docs/phase5s/README.md):
message counts, bytes and memory on oversubscribed runs (locust to 512 ranks, Kathleen's
4 nodes to 1,280), times to 72 ranks (locust) and 160 (Kathleen), and a model beyond.
This task measures the final structure (after T6 and T7), evaluates the triggers, and
**builds only what the triggers and the sign-off require**. The expected outcome is that
both close on counts, bytes and models, with their timing at scale left open; a report
that says so, with the numbers, is a complete result.

Read first:
- octree/CLAUDE.md, fmm-plan/CLAUDE.md, root CLAUDE.md ("MPI");
- docs/phase5s/README.md (including "Without Kathleen") and `docs/design/scale-out.md` §7
  (the triggers), §8 and §10 (the time model);
- distributed-fmm.md §3.2 (O1, the T0 condition), §4.2 (the redistribution's collectives),
  §14.2 (the "acceptable" rows: O1's rounds, the block keys, the count all-to-alls), §14.4
  (S4, S5) and §14.5 ("A cluster may not be available");
- the code: `Octree::new`, `linearize`, `balance`, `parallel_sort::parsort`,
  `global_block_counts`, `redistribute_with_respect_to_coarse_tree`; `Redistribution::new`;
  rlst's `GhostCommunicator::build` (from the registry); T3's advertisement exchange;
  rsmpi 0.8.2's `immediate_synchronous_send`, `immediate_barrier`, the probe calls,
  `request::scope`.

Do:
1. **Measure** (release; BLAS variables at 1), on the cube and the Plummer sphere:
   - **counted, not timed**, on oversubscribed runs (locust 128, 256, 512 ranks; Kathleen's
     4 nodes 320, 640, 1,280 ranks) at N = 10⁶–10⁷: the messages and bytes per rank of
     each sample sort, of the coarse tree's rounds, of each O(P) count all-to-all (in
     `Redistribution::new`, rlst's graph builds, T3's advertisements), and the
     intermediate imbalance of the sample sorts (keys per rank after each sort, max over
     mean); the memory of each at its peak;
   - **timed**, at 8–72 ranks on locust and (Kathleen) 40, 80 and 160 ranks: `Octree::new`
     by part (each `parsort`, the coarse tree's rounds, the redistribution of keys,
     `generate_all_keys`'s successor) and each count all-to-all, against the whole build
     and an evaluation; the Omni-Path latency and bandwidth the time model uses
     (Kathleen) or the stated assumed values (without Kathleen);
   - **modelled** to 10³ and 10⁴ ranks: time per item from the counted messages and bytes
     at the stated latency and bandwidth (the method of `scale-out.md` §10), marked
     *model*.
2. **Decide** against T1's triggers. Write the result into `docs/design/scale-out.md`
   (an addendum to §7) and the PR description: which package fires, on which counted or
   modelled number, and what stays open (the timing at scale, network effects a model
   misses).
3. **Build what fires**, as the sign-off of this step accepts (ask before building):
   - S4: keys routed to the blocks' owners by local counting and per-round all-reduces,
     deduplication per block, the T0 condition checked after the move; the leaves, the
     partition and every key bit for bit the same (compared key by key at 1–8 ranks, at
     512 oversubscribed on locust and (Kathleen) at 160); both sample sorts removed from
     `Octree::new`;
   - S5: a safe sparse dynamic exchange in `nd-fmm-plan` (or wherever the design places
     it) replacing the count all-to-alls it names; tested against the dense version
     (same counts and payloads on every rank) at 1–8 ranks, at 512 oversubscribed and
     (Kathleen) at 160; the rlst graph builds replaced only if rlst's API allows it
     without a fork (otherwise ask: an upstream change).
   If neither fires: no code; the report closes both, with the counted costs and the rank
   count at which each would fire (*model*).

Tests that define done:
- Step 1's counts, timings and models and step 2's decision reported, with node-hours used
  (budget: at most 20 node-hours unless the sign-off raises it; no job larger than 4
  nodes).
- For a package built: its own tests as above; the Phase 5 suites and T2's invariance check
  pass unchanged (bit for bit) at 1, 2, 4, 8 ranks, at 512 against 64 on locust and
  (Kathleen) at 160 against 20; its counted messages and bytes below the trigger at 1,280
  ranks.
- For a package not built: the report, and nothing else changed.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks (if code changed);
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks; the `nd-octree` MPI examples at 1–8
  ranks (if `nd-octree` changed);
- (Kathleen) the correctness job of T2 at 2 nodes (if code changed).

Do not:
- build S4 or S5 without a trigger firing and the sign-off of step 2;
- time an oversubscribed run, or present a modelled time as measured;
- submit a job larger than 4 nodes;
- change the leaves, O1–O4, the second cut of T7 or any output bit;
- use `unsafe`, raw MPI FFI or an rlst fork.

# Phase 5S / T5 — the owned top tree in `nd-fmm-plan` (C5S.3; S2, part 1)

After T4 a rank holds only its locally essential set, but every evaluation still:
- gathers every coarse block's multipole to every rank (`CoarseExchange::gather`,
  `all_gather_varcount_into` in fmm-plan/src/exchange.rs): 2.6 MB per rank at 72 ranks and
  11.6 MB at 512 on the N = 10⁶ cube (Phase 5 T10), about 170 MB at 8,192 ranks on
  N = 10⁸ (*model*, distributed-fmm.md §14.2);
- runs the global M2M over every `Global` box on every rank (`upward_global`): about
  300,000 M2M per rank at 8,192 ranks, three times the useful work (*model*).

This task builds S2 in `nd-fmm-plan` as `docs/design/scale-out.md` §4 (T1, signed off)
specifies:
- every `Global` box (shared ancestor) gets an **owner**: the rank of its first descendant
  block;
- an **upward reduction**: the owners of a box's children send their exact multipoles to
  the box's owner, which adds them **by octant**, in the one-rank order, so every
  multipole keeps its one-rank bits (one round per top-tree level);
- a **replicated top** above the signed-off level L_r, formed by the reduction and
  replicated by one all-gather of fixed size; below it the multipole exchange carries the
  multipoles a rank's rows read, as ghosts;
- the coarse gather removed; P2 kept; new `HostData` events for the reduction and the
  replicated top; overlap posting the reduction's levels as their children complete.

T6 does the `nd-fmm-exec` side (the device operator's events, `Fmm`, timings).

Read first:
- fmm-plan/CLAUDE.md, root CLAUDE.md ("MPI");
- docs/phase5s/README.md (requirements 2, 5–8) and `docs/design/scale-out.md` §4, §8, §9;
- distributed-fmm.md §3.6 (P2, the global M2M), §6 (the hook's events), §8 (overlap, the
  order-preserving property, the scoped requests, `test` after every level call), §14.2,
  §14.4 (S2: why an all-reduce of partial multipoles is rejected);
- fmm-plan-redesign.md §7 and §7.5 (the accumulation order);
- the code: `CoarseExchange` (`new`, `sent_blocks`, `received_blocks`, `gathered`,
  `gather`), `upward_global` (both the method and the free function in evaluator.rs),
  `evaluate_overlapped`, `far_field`, `MultipoleExchange` (its per-level sends and the
  overlapped variant), `HostData` and its six events in operator.rs, the recording
  operator and the overlap tests of `tests/mpi_regressions.rs`, `tests/overlap_stress.rs`.

Do:
- **Owners and the reduction.** From the partition map (T3): the owner of every shared
  ancestor; for each top-tree level from the blocks upwards, the sends of exact child
  multipoles to the parent's owner and the receives there; the parent's multipole formed
  by M2M of its eight children in octant order, exactly the operations and order of the
  one-rank pass. Prove it in the code's docs and test it: every shared-ancestor
  multipole equal bit for bit to the one-rank `Fmm`'s (by key) on the cube, the Plummer
  sphere and a graded scenario at 2, 4 and 8 ranks.
- **The replicated top.** The boxes at levels ≤ L_r (as signed off): formed by the
  reduction on their owners, then replicated by one all-gather whose size does not depend
  on P (state its bytes at p = 8, f32 and f64). Below L_r, the multipoles a rank's V, W
  and X rows read reach it through the multipole exchange (its neighbour sets now include
  the owners of the shared ancestors the rows name).
- **The coarse gather removed.** `CoarseExchange` deleted or reduced to what the design
  keeps; `Evaluator::coarse_exchange` and its users updated; the global M2M per rank only
  over the boxes it owns plus the replicated top.
- **Events.** New `HostData` events (names and lists as signed off) before each reduction
  send and after each receive, and around the replicated top's all-gather; the six
  existing events keep their meaning. The host operator ignores them, so the host path's
  bits do not change.
- **Overlap.** In `evaluate_overlapped`, each reduction level is posted as soon as its
  children are complete, with scoped requests and `test` after every level call; the
  output stays bit for bit the blocking path's.
- **Tests** in `tests/mpi_regressions.rs` (scenarios unchanged, on 1, 2 and 4 ranks): the
  recording operator sees the same M2M, M2L and L2L pairs once each, in the documented
  order and groupings; the exchange checks cover the reduction's messages (sent equals
  received over the ranks, every needed multipole received exactly once); the overlap
  check covers the reduction; the overlap stress test (ignored) passes at 8 ranks.
- **Measured** (release; reported): per rank, the global M2M count, the reduction's
  messages and bytes per level, the replicated top's bytes, the multipole exchange's
  bytes and neighbours, at 2–512 ranks on locust and (Kathleen) 40–1,280 ranks (above 72
  and 160 respectively oversubscribed: counted, not timed), against Phase 5's coarse
  gather and global pass.

Tests that define done:
- Every shared-ancestor multipole bit for bit the one-rank value at 2, 4 and 8 ranks.
- `mpi_regressions` passes at 1, 2, 4 and 8 ranks; `overlap_stress` at 8.
- `nd-fmm-exec`'s suites pass unchanged with the host operator (`mpi_exec`, `multi_rank`
  with 0 values differing), and T2's invariance check passes at 1, 2, 4 ranks, at 512
  against 64 on locust and (Kathleen) at 160 against 20. If the device operator cannot follow the new events before T6,
  device-backend scenarios may be marked as T6's (say which), never silently skipped.
- No per-evaluation item grows with P at fixed N/P (requirement 5), counted to 512 ranks
  on locust and (Kathleen) 1,280 ranks.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks; `cargo doc
  --no-deps`;
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks on the M3 Max and on locust;
- (Kathleen) the correctness job of T2 at 2 nodes.

Do not:
- reduce partial multipoles with `MPI_Reduce`, an all-reduce or any operation whose order
  depends on the MPI library or the rank count;
- change the accumulation order of fmm-plan-redesign §7.5 or the overlap order of Phase 5
  decision 9;
- change P2's rows or any list entry;
- use `unsafe` or raw MPI FFI.

# The distributed FMM: design for Phase 5

Phase 5, task T1 (docs/phase5/T1-distributed-design.md). Drafted on 2026-10-08 for
sign-off by hand before T4. **Signed off on 2026-10-08: every recommendation of §13
accepted** (docs/phase5/README.md, decisions 1 and 6–13). It answers the requirements of docs/phase5/README.md
("Requirements on the distributed FMM") and settles the decisions that change code in
T2–T10.

Companion documents, cited by section:
- **redesign**: [fmm-plan-redesign.md](fmm-plan-redesign.md) (the plan, the exchanges,
  the evaluator; §9 redistribution, §10 device and overlap);
- **device-path**: [device-path.md](device-path.md) (§4 residency and the C5.1 hook
  sketch, §12 errors, §14 multi-rank);
- **laplace-fmm-plan**: [laplace-fmm-plan.md](laplace-fmm-plan.md) (§7, the Phase 4 and
  Phase 4S "Recommendation for Phase 5");
- **README**: [docs/phase5/README.md](../phase5/README.md).

Every number is either *measured*, with its machine and command (§1.2 says how), or
marked *model*.

## Summary

The pieces of a distributed FMM exist, and the measurements of §1.2 show what has to
change before they make one:

1. **The tree depends on the rank count and on the input distribution.** The octree's
   coarse blocks come from the ranges of a sample sort with 4 samples per rank. Some
   blocks lie inside a leaf of the one-rank tree, and are then split, so the
   distributed tree has leaves the one-rank tree does not: in 15 of 24 measured
   configurations at 4 and 8 ranks. **The partition is also poor**: the busiest rank
   holds 1.25 times the mean number of points at 2 ranks, about 2.0 at 8, and up to 14
   at 72 (§3.1). Both come from `nd-octree`'s coarse tree, not from the FMM.
2. **With the same tree, a multi-rank result differs from the one-rank result only in
   the order of the near and X rows.** Every multipole agrees bit for bit, and so does
   every local without an X row (§5.2). The measured difference is 2–7 u_T relative L2
   in f32 and f64 (u_T the unit roundoff).

The design therefore proposes:

- **`nd-octree` (T4, §3).** A coarse tree built top-down by weight from the root, whose
  blocks are always nodes of the one-rank tree. The leaves are then the one-rank leaves
  on every rank count and for every input distribution. This was replayed on all 48
  measured configurations. Blocks are weighted by points, not distinct keys; a
  partition rule cuts at the nearest block boundary; and ranks without blocks replace
  the "fewer blocks than ranks" panic. In simulation on the real coarse trees, the
  imbalance falls to at most 1.06 at 2–8 ranks and 1.12 at 72. The construction itself
  costs about nothing more; its cost is downstream: finer blocks mean more `Global` boxes,
  whose locals every rank computes. That is negligible up to 16 ranks but up to 6 fair
  shares of M2L work at 72 ranks and N = 10⁵ (simulated, §3.6), so the design pairs O1
  with **P2**, an `nd-fmm-plan` change by which a rank computes only the `Global` locals
  its own blocks descend from (bit for bit the same values). At 16–72 ranks points also
  stop balancing M2L work (up to 1.5× at N = 10⁶); a work weight is left to Phase 6.
- **`nd-fmm-plan` (T5, §4, §5).** `Redistribution` as redesign §9, with (key, position) on
  the wire, the order within a leaf (origin rank, origin position), `_into` variants and no
  cheap path (an all-to-all-v with only local data costs under 1 µs, measured). Near and
  X rows ordered by the entry's (level, key) instead of its leaf index. On one rank that
  is the same order, so one-rank results do not change. With it, a multi-rank `Fmm`
  equals the one-rank `Fmm` over the union of the points, in rank order, **bit for
  bit**; this was measured on a scratch copy of the code (§5.3). Requirement 2's
  tolerance is then needed only between input distributions: 100 u_T relative L2
  (1.1e-14 f64, 6.0e-6 f32).
- **The hook (T7, §6).** Six events, one combined "send" event before the coarse
  gather, so that a device downloads everything it sends in one sync. The shadow check
  drives the `Evaluator` directly; no new public constructor is needed.
- **The device on several ranks (T8, §7).** The ghost source tail is uploaded in one
  call. The multipoles and coarse blocks to send are gathered and downloaded in one
  sync; the received ones are uploaded packed and scattered by one new kernel,
  `movement::scatter_columns`. That is two syncs per evaluation on several ranks, one on
  one rank. Errors are agreed by one all-reduce at the end of `evaluate`. A correction
  to the brief: with every source kind on the device, the host source store does not
  hold the charges (Phase 4S T9), so `Fmm` writes the charges of the sent leaves there.
- **Overlap (T9, §8).** Order-preserving only. The source exchange is hidden behind the
  local upward pass, and the multipole exchange, posted right after it, behind the
  coarse gather, the global pass and the coarser downward levels. The mechanism is
  scoped rsmpi point-to-point on the ghost communicators' graph communicators, with
  `Request::test` between level calls; measured, messages do not progress without MPI
  calls. On one node an exchange is a memory copy by the CPU, which overlap cannot hide,
  so the gain there is small (*model*, §8.2); the criterion is the exposed wait.

**Cluster scale (§14).** The goal beyond Phase 5 is benchmarks on thousands of ranks.
Simulated at 1,024 and 8,192 ranks, O1's partition scales (26–32 blocks per rank, balance
within 1.11), and a rank needs only about a thousand top-tree multipoles at any P. But
the replicated top tree that Phase 5 keeps grows like P. At 8,192 ranks every rank would
do about three times its useful work in redundant M2M, receive 170 MB per evaluation in
the coarse gather, and hold 1.5 GB of replicated coefficients (*model*). §14 plans the
replacement as a scale-out phase after Phase 5: a locally essential top tree in
`nd-octree` (the larger octree change), an owned top tree with an order-preserving
reduction in `nd-fmm-plan`, a work-weighted cut, and validation at scale by rank-count
invariance. It also sets the constraints Phase 5 keeps so that this extends it.

§13 lists the questions for sign-off with a recommendation each.

## 1. Starting point

### 1.1 What exists on several ranks today

**`nd-octree`** (`Octree::new`, octree/src/octree.rs; the private
octree/src/octree/implementation.rs):

| Step | Function | What it does on several ranks |
| --- | --- | --- |
| 1 | `linearize` | sample sort of the input keys (`rlst::parallel_sort::parsort`, 4 weighted samples per rank, `OVERSAMPLING`), then removes duplicates and ancestors: each rank holds a contiguous, *roughly* equal range of distinct keys |
| 2 | `compute_coarse_tree` | each rank contributes the largest boxes of the region spanned by its range (`completed_region`), capped at `max_level`; the union is gathered to every rank, completed and 2:1 balanced serially: the replicated coarse tree. On one rank it is the root |
| 3 | `compute_coarse_tree_weights` | weight of a block = the number of *distinct* linearized keys under it (all-reduce) |
| 4 | `load_balance` → `partition_blocks` | contiguous, non-empty ranges of blocks: rank p takes blocks "while the weight before is below total (p + 1) / P", at least one, and never so many that a later rank is left without one; panics if there are fewer blocks than ranks |
| 5 | `redistribute_with_respect_to_coarse_tree` | moves the keys to their blocks' ranks (all-to-all-v) |
| 6 | `create_local_tree` | refines each block while it holds more than `max_fine_keys` distinct keys, to `max_level` |
| 7 | `balance` | distributed 2:1 balance: a per-key closure (siblings, neighbours of parents), then `linearize` again; partition-independent given its input |
| 8 | `get_tree_bins`, `generate_all_keys` | partition bounds (first block of every rank, gathered); `Global` keys (ancestors of blocks, replicated); ghosts (one-cell halo, and with the ghost-children layer the children of interior neighbours and every rank's coarse blocks, gathered to every rank) |

Lookups: `owner_rank(key)` (O(log P), no communication), `local_leaf(key)`,
`lookup_leaves(keys)` (collective).

**`nd-fmm-plan`**:
- `Plan::new` (fmm-plan/src/plan.rs): two all-reduces (level count, validity). Builds
  `BoxIndex` (every held key, local, `Global` and ghost, Morton order per level) and the
  `LeafNumbering` (local leaves by (level, key), then the ghost leaves a U- or X-row
  names, by key; redesign §3.3).
- `Evaluator::new` (evaluator.rs): an all-reduce of validation, then three exchanges
  (exchange.rs):
  - `SourceExchange`: one rlst `GhostCommunicator<MortonKey>` over the ghost leaves,
    chunk sizes per key (the owner's count × point size), receive buffer = the ghost
    tail of the source store; send buffer gathered by `send_leaves()`;
  - `MultipoleExchange`: one `GhostCommunicator` per level over the ghost boxes named by
    V-rows of level l and W-rows of level l − 1; flat send and receive buffers,
    `send_boxes(l)`, `receive_boxes(l)`;
  - `CoarseExchange`: a duplicate of the communicator; every rank's coarse blocks
    (`keys()`, `block(b)`, `rank_blocks(r)`), one all-gather-v of their multipoles per
    evaluation.
- The evaluation (redesign §7.2): stage 1 `exchange_sources`, 2 `upward_local`,
  3 `upward_global` (gather, then the global M2M), 4 `exchange_multipoles` (levels
  0..=L), 5 `downward`, 6 `evaluate_leaves`.

**`nd-fmm-exec`** (`FmmBuilder::build`, fmm-exec/src/fmm.rs):

| Step | What | Collectives |
| --- | --- | --- |
| 1 | settings, MPI threading, pool, supplied domain, finite points, device opened | all-reduce (agreement) |
| 2 | any points at all? | all-reduce |
| 3 | domain: supplied, or `compute_global_bounding_box`; every point inside | all-reduce pair (extent), the box's reductions, all-reduce (agreement) |
| 4 | `Octree::new` from the keys of the rank's sources and targets; `Plan::new` | §9.1 |
| 5 | every point's local leaf; **`PointsNotOwned`** if any rank has a point another rank owns; then **`DeviceNeedsOneRank`** with a device on several ranks (no collective) | all-reduce (count) |
| 6 | leaf order, stable; radii; output order | — |
| 7 | `max_leaf_points`; tables; operator (with a device, `DeviceOperator::new`); `Evaluator::new` | all-reduce (max); §9.1 |
| 8 | leaf-scaled coordinates into the stores; device upload | — |

`Fmm::evaluate` agrees the charge-vector length (one all-reduce), loads the charges,
`reset`s, runs the six stages and the output pass.

The collectives of redesign §7.3 hold as written, with two details it leaves out:
`CoarseExchange::new` duplicates the communicator (a collective), and every exchange
constructor ends with an all-reduce that agrees the layout (exchange.rs, module docs).
§9.1 has the full table.

### 1.2 Measurements

**How.** A throwaway program in the session's scratch directory (`t1measure`, not
committed), built in release against this workspace (`Cargo.lock` copied). Every rank
draws the same global point set (`nd_fmm_validate::fmm_accuracy::Distribution`, seed
`0xc32`: the cube [−1, 1)³, the Plummer sphere a = 0.1, the five Gaussian clusters; and
a "coincident" set: the cube with 20% of the points on 100 positions, 200 or 2,000 each).
It passes its share to `Octree::new` exactly as `Fmm::build` does (the keys of its
sources, then of its targets, sources = targets), with `max_level` 16, 64 keys per leaf
and the ghost-children layer. Inputs:
- **share**: every P-th point (`tests/mpi_exec.rs`'s `share`);
- **rank 0**: every point on rank 0;
- **owner**: the octree's own partition, found by rebuilding until every rank passes
  exactly the points it owns (the fixed point; 1–3 rebuilds where it exists).

It then builds `Plan::new`, an `Evaluator` with a no-op operator of the Laplace sizes
at p = 8, and the three exchanges, and reports per rank: leaves, points, near pairs
(Σ over targets of the sources in the near row), V, X and W pairs, ghosts, held boxes,
exchange sizes, and times. Exchange times come from the no-op evaluator, each stage
after a barrier: pure communication. With the owner input, it also builds the real
`Fmm` (host path, one thread per rank). Rank 0 builds the one-rank `Fmm` of all points
on `SimpleCommunicator::self_comm()` and compares outputs, trees, expansions and errors
against `direct_sum` at 1,000 sampled targets.

**Machines.**
- **M3 Max**: `mpirun --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 -n P`,
  every run under `timeout`, BLAS variables 1, Open MPI 5.0.10 (Homebrew). Measured on
  2026-10-08 with a background load of about 1–4 cores from other processes (another
  build, an antivirus scanner); its times are indicative.
- **locust**: `tools/gh200/sync.sh`, then inside `tools/gh200/env.sh`,
  `timeout … mpirun --report-bindings -n P` (one rank per core, bound), Open MPI 5.0.10
  (spack). Load before the first runs 0.5–1.1, later only the tail of these runs
  (at most 3.6 before a run); GPU idle; no other user's job before, during or after.

#### Partition today (input: share)

Points of the busiest rank over the mean (max/mean); the leaves, near pairs and V pairs
follow within 0.1 in every case. M3 Max at 2–8 ranks, locust at 16–72. The trees and
partitions are deterministic: both machines gave identical per-rank point counts in all
24 configurations at 2–8 ranks.

| Workload | N | 2 | 4 | 8 | 16 | 32 | 64 | 72 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cube | 10⁵ | 1.25 | 1.50 | 1.99 | 1.25 | 1.51 | 1.94 | 2.04 |
| cube | 10⁶ | 1.25 | 1.50 | 1.99 | 1.24 | 1.50 | 1.99 | 2.09 |
| Plummer | 10⁵ | 1.00 | 1.52 | 1.98 | 2.07 | 3.13 | 1.49 | 2.91 |
| Plummer | 10⁶ | 1.25 | 1.47 | 1.04 | 2.07 | 3.34 | 2.28 | 2.85 |
| clusters | 10⁵ | 1.60 | 1.60 | 1.75 | 3.99 | 6.97 | 12.4 | 13.9 |
| clusters | 10⁶ | 1.60 | 1.62 | 1.73 | 4.00 | 6.99 | 12.3 | 13.8 |
| coincident | 10⁵ | 1.24 | 1.43 | 1.90 | 1.29 | 1.40 | 2.00 | 2.13 |
| coincident | 10⁶ | 1.24 | 1.48 | 1.14 | 1.26 | 1.60 | 2.01 | 2.15 |

Coarse blocks per rank vary from 1 to 36 (cube 10⁵, 8 ranks), 4 to 415 (clusters 10⁶,
8 ranks) and 1 to 559 (clusters 10⁵, 72 ranks). The heaviest single block weighs up to
1.6 fair shares (clusters, 4–8 ranks) and 14 (clusters, 72 ranks). The partition also
depends on the input: with every point on rank 0 the same cube at 8 ranks has 22 blocks
(92 with the share input). The owner input's fixed point does not exist in 3 of the 12
configurations at N = 10⁵ and 2–8 ranks, nor for the cube at N = 10⁶ on 4 ranks: the
partition moves again with each rebuild.

**Measured per-rank time against work** (M3 Max, owner input, N = 10⁵, the real `Fmm`,
one thread per rank; f64 and f32 at p = 3 and 8; 9 configurations at 2–8 ranks, 36
runs): in each run the max/mean of the compute stages (upward, downward, leaves) follows
the max/mean of the points within 0.04 in 32 of 36 runs and within 0.11 in all (the
worst: Plummer at 8 ranks, f64 p = 8, time 1.89, points 2.01). Within a run, points are
an adequate proxy for work (§3.5).

#### The tree on several ranks

Leaves of the P-rank tree that the one-rank tree does not have, and the converse (M3
Max, share input; every 2-rank tree and every rank-0-input tree equals the one-rank
tree):

| Workload | N | 4 ranks | 8 ranks | points in differing leaves (4, 8) |
| --- | --- | --- | --- | --- |
| cube | 10⁵ | 30 / 2 | 15 / 1 | 48, 28 |
| cube | 10⁶ | 15 / 1 | 59 / 3 | 33, 82 |
| Plummer | 10⁵ | 16 / 2 | 16 / 2 | 6, 6 |
| Plummer | 10⁶ | 8 / 1 | 8 / 1 | 9, 9 |
| clusters | 10⁶ | — | 8 / 1 | –, 5 |
| coincident | 10⁵ | 30 / 2 | 23 / 2 | 37, 37 |
| coincident | 10⁶ | 30 / 2 | 59 / 3 | 42, 59 |

(— : equal. Clusters at 10⁵ equal at 4 and 8.) A one-rank leaf is split into its 8
children wherever a coarse block lies strictly inside it. A serial replay of
`create_local_tree` and `balance` on the gathered coarse blocks reproduces both the
one-rank tree and every P-rank tree exactly (48 of 48 configurations). From the blocks
coarsened to nodes of the one-rank tree (§3.2, O1) it gives the one-rank leaves in all
48, with and without 2:1 balancing the coarse tree.

#### Ghosts, exchanges and replicated data at p = 8

Per rank, maximum over ranks (values; bytes are 8 per value in f64 and 4 in f32). The
source exchange carries 4 values per point (CONVENTIONS §3.13):

| | cube 10⁶, 8 ranks | Plummer 10⁶, 8 ranks | cube 10⁶, 72 ranks | Plummer 10⁶, 72 ranks |
| --- | --- | --- | --- | --- |
| local leaves (mean) | 4,103 | 6,668 | 457 | 744 |
| ghost leaves (max) | 1,076 | 2,459 | 785 | 1,071 |
| source exchange sent (max) | 140,052 (1.1 MB f64) | 196,056 (1.6 MB) | 106,268 (0.85 MB) | 84,484 (0.68 MB) |
| multipole exchange sent (max) | 321,570 (2.6 MB) | 609,687 (4.9 MB) | 1,243,350 (9.9 MB) | 1,951,047 (15.6 MB) |
| multipole exchange sent (median; 10th–90th percentile) | 220,887; 132,856–318,168 | 565,623; 517,298–585,986 | 169,533; 127,753–337,422 | 236,115; 131,374–822,215 |
| neighbours (sources / multipoles) | 6–7 / 7 | 7 / 7 | 8–71 / 71 | 2–71 / 71 |
| coarse blocks (all ranks) / `Global` boxes | 218 / 31 | 323 / 46 | 1,359 / 194 | 1,653 / 236 |
| coarse gather per evaluation | 17,658 (141 kB) | 26,163 (209 kB) | 110,079 (0.88 MB) | 133,893 (1.07 MB) |
| held boxes (max / mean) | 12,913 / 7,742 | 15,881 / 14,969 | 8,169 / 6,587 | 12,089 / 8,746 |

Ghost boxes per level, summed over the 8 ranks of the Plummer sphere at N = 10⁶: 0, 0,
273, 3,507, 6,857, 6,784, 8,480, 11,056, 13,856, 7,656 (levels 0–9); multipole values
sent per level, max over ranks: 0, 0, 6,804, 57,024, 66,744, 87,480, 97,848, 129,600,
148,392, 87,480.

At 72 ranks every rank exchanges multipoles with all 71 others: the coarse levels' V
rows reach every rank. Rank 0 sends 7–8 times the median rank (not analysed; it is one
rank of today's imbalanced partition). About 90% of a rank's held boxes are
ghosts or replicated at 72 ranks (cube: 6,587 held on average, about 520 local).

#### Times

Build, max over ranks, median of 3 (release; `Plan::new` dominated by the rank with
the most keys; M3 Max one-rank, cube 10⁵: `Octree::new` 5.9 ms, `Plan::new` 34.5 ms,
`Evaluator::new` 4.1 ms):

| locust | P | `Octree::new` | `Plan::new` | `Evaluator::new` |
| --- | --- | --- | --- | --- |
| cube 10⁶ | 2, 8, 16, 72 | 105, 35, 23, 23 ms | 261, 101, 32, 14 ms | 42, 16, 6.1, 6.7 ms |
| Plummer 10⁶ | 2, 8, 16, 72 | 145, 36, 34, 25 ms | 433, 88, 89, 30 ms | 65, 14, 14, 14 ms |
| cube 10⁵ | 2, 8, 72 | 9.3, 3.9, 4.1 ms | 26, 10, 2.0 ms | 3.8, 1.9, 3.9 ms |

M3 Max at 2–8 ranks: the same shape, between 0.7× (2 ranks) and 2× (8 ranks, under the
background load) locust's.

Pure exchange times (no-op operator, after a barrier, max over ranks, median of 3–5):

| | P | source exchange | coarse gather + global M2M (no-op) | multipole exchange (all levels) |
| --- | --- | --- | --- | --- |
| locust, cube 10⁶ | 2, 8, 16, 72 | 0.35, 0.29, 0.44, 1.32 ms | 0.003, 0.042, 0.086, 1.21 ms | 0.39, 0.51, 0.84, 4.25 ms |
| locust, Plummer 10⁶ | 2, 8, 16, 72 | 0.58, 0.35, 0.83, 1.79 ms | 0.005, 0.045, 0.11, 1.37 ms | 0.95, 1.37, 1.78, 7.95 ms |
| M3 Max, cube 10⁶ | 2, 4, 8 | 0.74, 0.92, 1.43 ms | 0.031, 0.096, 0.42 ms | 1.26, 1.71, 3.40 ms |
| M3 Max, Plummer 10⁶ | 2, 4, 8 | 1.09, 1.40, 1.30 ms | 0.043, 0.089, 0.33 ms | 2.60, 3.96, 4.37 ms |

The evaluations these exchanges sit in are §8.2's input.

#### Multi-rank against one rank (host path, owner input)

Wherever the owner input exists, the `Fmm`'s tree equals the one-rank tree (checked in
every run below), so these differences are the row order alone (§5):

| Workload, N | P | f64 φ / ∇φ, relative L2 | f64 φ, relative max | f32 φ / ∇φ, relative L2 | φ bit-identical |
| --- | --- | --- | --- | --- | --- |
| cube 10⁵ | 2, 4, 8 | 2.5e-16 / 1.4e-16 … 5.1e-16 / 5.3e-16 | ≤ 9.6e-16 | 1.3e-7 / 6.7e-8 … 2.7e-7 / 2.1e-7 | 93%, 87%, 80% |
| Plummer 10⁵ | 2, 4, 8 | 4.7e-16 / 2.6e-16 … 7.2e-16 / 4.1e-16 | ≤ 1.9e-15 | 2.5e-7 / 1.6e-7 … 3.9e-7 / 2.3e-7 | 81%, 66%, 52% |
| clusters 10⁵ | 4, 8 | 1.8e-16 / 3.4e-17 … 4.6e-16 / 1.0e-16 | ≤ 7.8e-16 | 9.4e-8 / 3.1e-8 … 2.5e-7 / 6.4e-8 | 94%, 71% |
| coincident 10⁵ | 2 | 2.2e-16 / 8.0e-17 | 4.3e-16 | 1.2e-7 / 3.8e-8 | 93% |
| cube 10⁶ | 2, 8 | 2.4e-16 / 7.5e-17 … 3.9e-16 / 1.6e-16 | 1.5e-15 | 1.3e-7 / 4.7e-8 … 2.1e-7 / 8.4e-8 | 97%, 89% |

The same at p = 3 and p = 8 to two digits, and on locust (Grace) to two digits for the
cube and the Plummer sphere at N = 10⁵ and 10⁶, which adds 16 ranks: at most 7.5e-16 /
5.4e-16 (f64) and 4.1e-7 / 2.3e-7 (f32). The errors against the direct sum are the
one-rank run's: ratio 1.0000 in f64 and 0.9992–1.0008 in f32. On every Plummer and cube
run checked box by box (multipoles and locals of every non-ghost, non-`Global` box
against the one-rank `Fmm`'s, by key): **no multipole differs**; locals differ only in
boxes with an X row (Plummer 10⁵: 1,359 of 4,834 such boxes at 4 ranks, 1,948 at 8;
none without one).

#### Communication primitives (§8.4, §4.5)

A ring exchange with rsmpi's scoped `immediate_send`/`immediate_receive_into`, then 10
ms of computation, then the wait (median of 11, max over ranks):

| Message | M3 Max, 8 ranks: blocking / no MPI call during the work / `test` every 0.5, 2, 5 ms | locust, 72 ranks: the same |
| --- | --- | --- |
| 4 kB | 0.10 / 0.014 / 0, 0, 0 ms | 0.012 / 0.010 / 0, 0, 0.005 ms |
| 64 kB | 0.50 / 0.25 / 0, 0, 0.14 ms | 0.017 / 0.028 / 0, 0, 0.010 ms |
| 1 MB | 1.34 / 1.43 / 0, 0, 1.31 ms | 0.46 / 0.51 / 0.001, 0.34, 0.44 ms |
| 4 MB | 2.38 / 2.31 / 0.001, 0.001, 5.6 ms | 3.2 / 3.1 / 2.6, 3.5, 3.8 ms |

Without MPI calls a message of 64 kB or more does not progress. With a few `test` calls
in the window it completes, but on locust (single-copy transfers on Linux) a 4 MB
message's copy is done in the wait regardless. A one-byte all-reduce costs 80 µs on the
M3 Max at 8 ranks and 1.2–3.4 µs on locust at 8–72 ranks. An all-to-all-v in which every
rank sends only to itself costs 0.2–0.8 µs on both.

**rsmpi 0.8.2 defects found** (`mpi-0.8.2/src/request.rs`): `RequestCollection::test_some`,
`wait_some` and `test_any` panic ("could not cast c_int to usize") once every request of
the collection has completed, because MPI then returns `MPI_UNDEFINED` as the count or
index. `test_some` also reads the statuses by request index, not by completion slot.
`test_all`, `wait_all` and the plain `Request::test`/`wait` are correct.

### 1.3 What `IndexFmm` checked, and where each check lives after T2

| Check (fmm-plan/tests/mpi_regressions.rs, src/*_tests.rs) | What it proved | After T2 |
| --- | --- | --- |
| counts of one: every leaf receives every leaf index once, on the per-pair, row and grouping paths | the evaluator's data flow covers every (source leaf, target leaf) pair exactly once, through the exchanges and the global levels | pair coverage: the recording operator ("every oracle pair issued exactly once", call order, groupings) on every scenario and rank count; values through the exchanges: the Laplace FMM against the one-rank FMM (§5, bit for bit) and the direct sum in `tests/mpi_exec.rs` (T6) |
| variable counts `hash(key) % 5`, count check, every path | the variable-count layouts of the stores and the source exchange | the exchange checks (sources with `hash(key) % 5` counts, real points), unchanged; the Laplace FMM on every scenario |
| second evaluation bit-identical | `reset` and determinism | a second evaluation of the recording operator (or a minimal value operator in the test file), T2; `tests/mpi_exec.rs` repeatability |
| paths agree (per pair, rows, groupings) | the `PerPair` adapter | `fmm-exec/tests/mpi_exec.rs` "batched against per-pair" (`LaplaceOperator` against `PerPair<LaplaceOperator>`, bit for bit) |
| `examples/test_index_fmm.rs` (weekly, 3 ranks) | the above at 3 ranks | the multi-rank CI job (T3) and the weekly run of `nd-fmm-exec`'s registered examples |
| `src/index_fmm_tests.rs`, `src/evaluator_tests.rs` (serial) | the index operators; the passes on ghost-free trees | evaluator passes: `evaluator_tests.rs` with the recording operator (T2) |

The one gap: inside `nd-fmm-plan` no test then carries values through the exchanges.
§6's events, checked by the recording operator (T7), and the shadow check close it for
the data movements. §10 keeps one value-carrying check in `nd-fmm-plan`.

## 2. Requirements

The ten requirements of the README, and how this design meets each. Three are proposed
to change (§13, question 1).

| # | Requirement | How the design meets it |
| --- | --- | --- |
| 1 | Same interface; any subset of points per rank, empty included; output in the caller's order; `PointsNotOwned` disappears | `Redistribution` (§4) inside `build` and `evaluate`; signatures unchanged; `PointsNotOwned` removed, `FmmError::Redistribution` added for the one error left (an `i32` overflow, agreed) |
| 2 | Equal to one rank within a derived tolerance; errors against the direct sum match | **Proposed change:** with O1 (§3) and P1 (§5), the host path on P ranks equals the one-rank `Fmm` over the union of every rank's points in rank order **bit for bit** (measured on a scratch copy, §5.3). Between input distributions, and if P1 is rejected, relative L2 ≤ 100 u_T for φ and ∇φ: 1.1e-14 (f64), 6.0e-6 (f32), derived in §5.4. The device against the host on the same ranks: Phase 4's bounds (T8) |
| 3 | Deterministic for fixed input, ranks and settings, every thread count; overlap changes no bit | every movement is an exact copy, every sum has a fixed order (§5.1); overlap is order-preserving (§8.1); device tuning decided once (§7.6) |
| 4 | Collective discipline; errors agreed before the next collective; no hang | §9.1 lists every collective with what empty ranks do; input errors ride on existing agreements; one new all-reduce per build (device errors after the device operator is built, every backend) and, on the device path with P > 1, one per evaluation (§7.5) |
| 5 | Kernel-agnostic plan | `Redistribution` moves opaque payloads; the hook names stores and index lists, not Laplace; overlap is in the evaluator; tested with the recording operator in `nd-fmm-plan` and with Laplace in `nd-fmm-exec` (§10) |
| 6 | Safe | scoped rsmpi point-to-point (§8.4); no `unsafe`, no FFI, no rlst change; avoids rsmpi's broken `RequestCollection::test_some` (§1.2) |
| 7 | Minimal device traffic: exactly what each exchange sends and receives, packed, plus the one-rank minimum | **Refined:** the source exchange's sent chunks are written on the host (coordinates at build, the charges by `Fmm`), so nothing is downloaded for it; everything else as stated. Formula in §7.2 |
| 8 | Memory per rank N/P plus the ghost layer; the replicated part stated | §9.2: per rank 2 n_c s K + (7 + o) s N/P + …, with K = local + ghost + replicated boxes; the replicated part grows linearly in P and dominates the coefficient memory beyond P ≈ √(N/4,700) (*model*); measured, 92% of the held boxes at 72 ranks for N = 10⁶ |
| 9 | Tested on what can run: 1, 2, 4 (CI), 1, 2, 4, 8 by hand on both machines | §10; nothing here needs more |
| 10 | Measured, never asserted | §11 |

## 3. Partition and `nd-octree`

### 3.1 What the partition costs today

§1.2 measured it: the busiest rank holds 1.25× the mean points at 2 ranks on the
uniform cube, 1.5× at 4, 2.0× at 8, up to 14× at 72 (clusters). The leaves differ from
the one-rank tree in 15 of 24 configurations at 4 and 8 ranks, and the partition changes
with the input distribution. Six causes, all in octree/src/octree/implementation.rs:

1. **The cut rule overshoots.** `partition_blocks` gives rank p the blocks that *start*
   before total (p + 1)/P. With equal blocks whose prefix weight falls just short of the
   target, rank p takes one block too many: on the uniform cube at 2 ranks, 5 of 8
   octants (1.25). The next ranks then take at least one block each, and the last ranks
   are starved (one block of 10 points at 8 ranks, Plummer N = 10⁵).
2. **Blocks are as coarse as the sample sort's ranges.** `compute_coarse_tree` takes
   the largest boxes of each rank's sorted range. `parsort` balances those ranges with 4
   samples per rank, so the blocks weigh up to 1.6 fair shares at 4–8 ranks and 14 at 72
   (clusters), and no cut rule can balance them.
3. **The blocks depend on the input distribution** (the sample sort's ranges do): with
   every point on rank 0 the cube at 8 ranks has 22 blocks, with the share input 92.
4. **Blocks inside one-rank leaves change the tree.** A gap between two ranks' ranges is
   filled with small blocks by `complete_tree`; a block strictly inside a leaf of the
   one-rank tree is refined from there (`create_local_tree`), and the 2:1 balance
   closure splits the leaf. That is the only cause: replayed serially, the gathered
   blocks reproduce every measured tree (§1.2).
5. **Weights count distinct keys**, so coincident points weigh as one.
6. **The panic with fewer blocks than ranks** is reached through a small `max_level`
   (`single leaf` of `tests/mpi_exec.rs` is one-rank only for that reason), not through
   few points: 3 points on 8 ranks build, because the gap filling makes blocks.
   Globally empty input panics on several ranks ("no rank contributed a key") and gives
   a root leaf on one; `Fmm` returns `NoPoints` before either.

### 3.2 Proposed changes

Each change is in T4, with its reason, API, collectives and tests. They are designed to
go together; §13, question 2, accepts or rejects each.

**O1. A coarse tree by weight, built from the root, whose blocks are nodes of the
one-rank tree** (causes 2, 3, 4). On several ranks, after `linearize`, replace
`compute_coarse_tree` by:

```text
blocks ← [root]
repeat:
    every rank counts, for each block b, its linearized keys under b (distinct keys,
    d(b)) and its input keys under b (with multiplicity, w(b)); one all-reduce (sum) of
    2·len(blocks) u64
    split ← { b : w(b) > W / (k·P) and d(b) > max_fine_keys and level(b) < max_level }
    if split is empty: stop
    replace every b in split by its eight children
2:1 balance the blocks serially (as today), and one more all-reduce for the weights of
the blocks balancing created
```

W is the total weight and k the refinement factor (`OctreeOptions::with_block_refinement`,
default 8). Only boxes with more than `max_fine_keys` distinct keys are split, and the
one-rank tree splits exactly those (`create_local_tree` from the root). So every block
is a node of the one-rank tree before balancing (T0), and a node of its balanced form
after; refining and balancing from such blocks gives exactly the one-rank leaves. Proof
sketch: T0 refines the blocks, so balance(blocks) is refined by balance(T0); the blocks
refined by `create_local_tree` give a tree R that refines T0 and whose boxes are nodes of
balance(T0); the 2:1 balance is the coarsest balanced refinement, so balance(R) =
balance(T0). Replayed on all 48 configurations of §1.2 with a variant of O1 that has the
same property (today's blocks coarsened to nodes of T0), with and without balancing the
blocks: the one-rank leaves every time. The tree then depends on neither P nor the input distribution; the
partition still depends on P, never on the input.

Rounds: one per refined level, about log₈(W/(k P)) + 2, all-reduces of a few thousand
values (*model*). The gather of the largest boxes (`gather_to_all` in
`compute_coarse_tree`) goes. One rank keeps the root as its only block, so one-rank
trees and results do not change.

**O2. Weights by points** (cause 5). `w(b)` counts every key passed to `Octree::new`,
duplicates included; for `Fmm`, which passes one key per source and one per target, that
is points. Measured: per-rank time follows points within 0.11 in max/mean (§1.2).

```rust
/// What the partition weighs (Phase 5; `OctreeOptions::with_partition_weight`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PartitionWeight {
    /// Every key passed to `Octree::new`, duplicates included: for an FMM that passes
    /// one key per source and one per target, the points. The default.
    #[default]
    Keys,
    /// Every distinct finest-level key once (the weight up to Phase 4).
    DistinctKeys,
}

impl OctreeOptions {
    /// Set what the partition of the coarse blocks weighs; see [`PartitionWeight`].
    pub fn with_partition_weight(self, weight: PartitionWeight) -> Self;
    /// Refine the coarse tree until no block weighs more than 1/`factor` of a rank's
    /// fair share, wherever the one-rank tree refines too (more than `max_fine_keys`
    /// distinct keys) and above `max_level`. Default 8; 1 gives the coarsest blocks.
    pub fn with_block_refinement(self, factor: usize) -> Self;
}
```

The refinement does not refine leaves: a block that holds at most `max_fine_keys`
distinct keys (a one-rank leaf) or lies on `max_level` stays whole however heavy it is.
That bounds the imbalance by the heaviest such block (the coincident set at 72 ranks:
1.12, below).

**O3. The cut at the nearest block boundary** (cause 1). Boundary p (1 ≤ p < P) is the
block boundary, at or after boundary p − 1, whose prefix weight is closest to W p/P.
Ranges may be empty. Each boundary is then within half the heaviest block of its target,
so max/mean ≤ 1 + w_max / (W/P), w_max the heaviest block; with O1 that is
1 + max(1/k, the heaviest unsplittable block over a fair share).

**O4. Ranks without blocks** (cause 6). A rank whose range is empty has no leaves; it
enters every collective. With O1 a problem with at most `max_fine_keys` distinct keys has
one block (the root), and fewer blocks than ranks is common for small problems, so the
panic must go, not become an error: requirement 1 asks `Fmm` to run on any rank count.
T4 changes what assumes a non-empty range:
- `get_tree_bins`: an empty rank's bound is the next non-empty rank's first block (after
  the last non-empty rank, a bound above every valid key); `owner_rank` returns the last
  rank whose bound is at most the key, which is never an empty one;
- `redistribute_with_respect_to_coarse_tree`: bins from those bounds;
- `generate_all_keys`: the root as a block (one block in all) is a local key on its
  owner and its ghost elsewhere, never `Global`; today that case is "unreachable inside
  `size > 1`";
- the docs of the construction contract ("every rank owns at least one block" goes).

Globally empty input then gives a root leaf on rank 0 and nothing elsewhere, as on one
rank; no panic remains that `Fmm` can reach.

**O5. No new lookup.** `Redistribution` needs `owner_rank` (O(log P), local) on every
item and `local_leaf` with `BoxIndex::find_leaf` on the owner; both exist. The cost per
item is one binary search over P bounds, which `build` already pays today in step 5
(`local_leaf` on every point); a batched lookup would save nothing that matters.

**The effect, simulated on the real coarse trees** (rank 0 replays the partition on the
blocks the octree built, weighed by points over every point's key; M3 Max at 2–8 ranks,
locust at 16–72; points max/mean):

| | today | O3 alone | O1 (k = 8) + O2 + O3 | O1 (k = 16) + O2 + O3 |
| --- | --- | --- | --- | --- |
| 2–8 ranks, all 24 configurations | 1.00–1.99 | 1.00–1.58 (clusters) | 1.000–1.057 | 1.000–1.030 |
| 16–72 ranks, all 32 configurations | 1.24–13.9 | 1.01–13.9 (clusters) | 1.004–1.115 | 1.003–1.115 |
| blocks in all, at 72 ranks | 897–3,368 | – | 1,660–4,285 | 4,096–5,293 |

(The simulation used a variant of O1: today's blocks coarsened to one-rank nodes, then
the heavy ones refined. It may keep finer blocks than O1 in places; both satisfy O3's
bound, 1 + max(1/k, the heaviest unsplittable block over a fair share).) The largest
residual, 1.115,
is the coincident set at N = 10⁵ on 72 ranks: its 200-point positions sit in level-16
leaves that cannot be split. With every point on rank 0 as input, k = 8 gives 1.004–1.025
at 4 and 8 ranks.

**Rejected candidates:**
- *A caller-supplied cost per key.* The FMM's cost is per leaf (near pairs, V pairs),
  which the octree does not know when it partitions; points track time within 0.11 at
  2–8 ranks (§1.2), and at 16–72 ranks no single weight balances both P2P and M2L work
  (§3.5, §3.6). A second cut by modelled per-leaf work is a Phase 6 candidate (§13,
  question 12).
- *A minimum number of blocks per rank.* The weight threshold of O1 says what matters
  (no heavy blocks) and adds no blocks where the work is light.
- *An agreed error when there are fewer blocks than ranks.* O4 makes the case work.
- *Coarsening today's blocks to one-rank nodes* (what §1.2 replayed). It fixes the
  leaves but keeps the dependence on the input and the coarse blocks; O1 is simpler.

### 3.3 What stays

The keys, levels, neighbour rules and 2:1 balance; `max_fine_keys` and `max_level` as
leaf criteria (distinct keys: changing it would change one-rank trees and results); the
replicated coarse tree; the ghost layer and the ghost-children layer; the lookups. The
octree stores no points: O2 counts the keys it is given.

**Defaults.** On one rank nothing changes. On several ranks O1–O4 are the default: every
multi-rank tree and partition changes, which T4's brief forbade ("the defaults keep
today's behaviour unless the design says otherwise"). This design says otherwise
(§12, T4).

### 3.4 Collectives and tests

Collectives of `Octree::new` on several ranks: `compute_coarse_tree`'s `gather_to_all`
and `compute_coarse_tree_weights`' all-reduce are replaced by one all-reduce per
refinement round and one after balancing (§9.1); every rank enters every round, an empty
rank with zero counts; the round count is decided by the replicated weights, so it is
the same everywhere.

Tests (T4): serial unit tests of the cut rule (empty ranges, the overshoot case, a block
heavier than a share), of the refinement on hand-made counts, and of the bounds with
empty ranks; an MPI example `test_mpi_weighted_partition` (registered) with the four
workloads of §1.2 at 2, 4 and 8 ranks: the leaves equal the one-rank leaves (gathered
and compared on rank 0), the per-rank weight within O3's bound 1 + w_max/(W/P) of the
mean, computed by the test from the replicated weights, a rank with no keys,
every key on one rank, a tiny problem (one block) on 8 ranks; the existing examples
unchanged at 1 rank and passing at 3.

### 3.5 The weight the FMM passes

Per-leaf work, *model*: P2P n_t Σ_near n_s, M2L |V| n_c², P2M and L2P (n_s + n_t) n_c,
M2P and P2L |W| n_t n_c and |X| n_s n_c. A per-key weight would need the tree; points are
what the octree can count before it has one.

- **At 2–8 ranks points suffice.** Measured, per-rank time follows them within 0.11 of
  max/mean at imbalances of 1.0–2.0 (§1.2). Simulated with O1 (k = 8; §3.6), cutting by
  points keeps every rank's own M2L work (the V pairs of the boxes under its blocks)
  within 1.10 of the mean, except the coincident set at N = 10⁶ (1.13 at 8 ranks) and
  the clusters at N = 10⁵ (1.25 at 8 ranks).
- **At 16–72 ranks they do not.** The M2L work per point varies across the domain (a
  sparse region has as many boxes per volume as a dense one, and few points). Cutting by
  points (max/mean ≤ 1.12) leaves the busiest rank's own M2L work at 1.05–1.51 of the
  mean at N = 10⁶ and up to 4.4 at N = 10⁵ (clusters, 72 ranks). Cutting by M2L work
  instead balances it to 1.01–1.13 but unbalances points (P2P, P2M, L2P) by up to 2.1; a
  50/50 mix of the two stays within 1.20 on both at N = 10⁶, and reaches 1.41 (points)
  and 2.0 (M2L) at N = 10⁵ (simulated with O1 at k = 8; table in §3.6).

**The FMM passes no weight beyond its keys** (O2's default: sources and targets, each
once). That meets the phase's gates, which run at 2–8 ranks. A weight by modelled work
needs the leaves, so it would be a second cut after the tree is built (the leaves do not
depend on the cut, O1), with the per-kind costs measured per machine and precision; it
is a Phase 6 candidate, and T10 reports the per-rank work at 16–72 ranks that would
decide it (§13, question 12).

### 3.6 The replicated global pass, and what finer blocks cost

Every rank computes the multipoles **and the locals** of every `Global` box (redesign
§7.4): the global M2M, and the M2L, L2L and P2L rows of the `Global` boxes in the
downward pass. That work is repeated on every rank. The ancestors of the coarse blocks
are the `Global` boxes, so finer blocks (O1) mean more of it. Measured today, summed over
the ranks, the V pairs (the M2L work) are this many times the one-rank amount (the
`v_pairs` of §1.2's tree runs):

| | 8 ranks | 16 | 32 | 64 | 72 |
| --- | --- | --- | --- | --- | --- |
| cube, N = 10⁵ | 1.00 | 1.01 | 1.21 | 1.69 | 1.84 |
| cube, N = 10⁶ | 1.00 | 1.01 | 1.02 | 1.13 | 1.16 |
| Plummer, N = 10⁵ | 1.01 | 1.04 | 1.30 | 2.80 | 2.83 |
| Plummer, N = 10⁶ | 1.00 | 1.01 | 1.05 | 1.18 | 1.24 |

**Simulated** (the scratch program: on rank 0, the one-rank tree's nodes and their V
rows; the blocks of today's octree, or of O1 built top-down from the root as §3.2 states
it, 2:1 balanced; the `Global` boxes as their strict ancestors; the cut at the nearest
boundary by points for every set of blocks, today's included, so the "today's blocks"
column is not today's partition, which §1.2 measured; M3 Max at 2–8 ranks, locust at
16–72). The V counting reproduces the
one-rank totals exactly (cube 10⁵: 640,584) and today's measured sums above within 4%
(1.77 against 1.84, 2.81 against 2.83, 1.15 against 1.16, 1.23 against 1.24 at 72
ranks). M2L work of the busiest rank over a fair share (the one-rank total over P), with
every `Global` box as today, and with only its own blocks' `Global` ancestors (P2,
below):

| Workload, N, ranks | blocks (`Global`) today → O1 k = 8 | today's blocks, every `Global` | O1 k = 1, every | O1 k = 8, every | O1 k = 8, own ancestors (P2) | O1 k = 8, own boxes only |
| --- | --- | --- | --- | --- | --- | --- |
| cube 10⁵, 8 | 92 (13) → 309 (44) | 1.003 | 1.000 | 1.033 | 1.014 | 1.012 |
| Plummer 10⁵, 8 | 162 (23) → 288 (41) | 1.066 | 1.065 | 1.090 | 1.055 | 1.042 |
| clusters 10⁵, 8 | 939 (134) → 1,317 (188) | 1.401 | 1.440 | 1.344 | 1.263 | 1.236 |
| cube 10⁶, 72 | 1,359 (194) → 4,096 (585) | 1.338 | 1.222 | 1.829 | 1.174 | 1.156 |
| Plummer 10⁶, 72 | 1,653 (236) → 2,696 (385) | 2.551 | 1.608 | 1.972 | 1.511 | 1.502 |
| clusters 10⁶, 72 | 3,368 (481) → 4,208 (601) | 14.00 | 1.578 | 1.851 | 1.328 | 1.316 |
| cube 10⁵, 72 | 904 (129) → 3,935 (562) | 2.304 | 1.869 | 7.345 | 1.382 | 1.224 |
| Plummer 10⁵, 72 | 1,170 (167) → 2,717 (388) | 3.691 | 2.215 | 6.830 | 1.471 | 1.285 |
| clusters 10⁵, 72 | 2,885 (412) → 4,117 (588) | 15.05 | 5.291 | 8.967 | 4.488 | 4.093 |

What it shows:
- **Up to 16 ranks the repeated work is small** with today's blocks and with O1: the
  `Global` boxes' M2L work is at most 0.04 of a fair share at k = 8 on 16 ranks at
  N = 10⁶, and 0.08–0.39 at N = 10⁵; at 2–8 ranks at most 0.11.
- **At 72 ranks it is large, and O1's finer blocks make it larger.** At N = 10⁶ it is
  0.47–0.70 of a fair share at k = 8 (0.03–0.13 at k = 1, 0.09–0.41 today); at N = 10⁵,
  1.8–6.1 fair shares (0.26–1.08 at k = 1). Without a change to the global pass, k = 8 at
  72 ranks does worse than today's coarser blocks under the same cut for the cube and
  the coincident set, and better only where today's blocks are too heavy to balance (the
  Plummer sphere and the clusters at N = 10⁶).
- **The construction itself is not the cost.** O1's rounds are a local count over the
  rank's keys and an all-reduce of a few thousand values each (*model*: under 2 ms at
  N = 10⁶ on 8 ranks, against `Octree::new`'s measured 35 ms on locust); it drops a
  `gather_to_all`. The cost is downstream, in the replicated global pass, and also in
  the replicated data of §9.2 (the coarse gather, the held boxes), which grow with the
  blocks.

**P2. Each rank computes only the `Global` locals its own blocks need** (`nd-fmm-plan`).
A rank's locals depend only on the locals of their ancestors: L2L runs parent to child,
and L2P, M2P and P2P act on local leaves only. In `Plan::from_key_types`, which knows the
rank's own coarse blocks, a `Global` box that is not an ancestor of one of them gets
empty V, X and L2L rows; its local stays zero, which the evaluator's docs state
(`Evaluator::locals`). Everything else is unchanged:
- the global M2M stays whole, because the M2L rows that remain read the multipoles of
  any `Global` box or coarse block;
- the coarse gather stays whole;
- every kept row is the same row, so every value is the same: **bit for bit** today's
  results on every rank count, and the one-rank plan does not change (its one `Global`
  box, the root, has empty rows anyway).

With P2 the repeated work falls to what the busiest rank's own ancestors need: the
"own ancestors" column, within 0.02 of a fair share of the rank's own boxes at k = 8 at
N = 10⁶ on 72 ranks, and within 0.4 at N = 10⁵ (the clusters; 0.16–0.19 for the cube and
the Plummer sphere). The global M2M stays repeated: 8 M2M per `Global` box, 0.02–0.06 of
a fair share at N = 10⁶ on 72 ranks and 0.18–0.51 at N = 10⁵ (*model*, an M2M costing
about one M2L pair at the same p). Pruning it as well (only the `Global` multipoles that
kept rows read) is possible but not proposed: it needs the V rows' sources per rank, and
it matters only at a few thousand points per rank.

**The cut by work** (simulated as above, O1 at k = 8, the nearest-boundary cut by
three weights; max/mean of the points and of the rank's own M2L work):

| Workload, N, ranks | by points: points / M2L | by M2L work: points / M2L | 50/50 mix: points / M2L |
| --- | --- | --- | --- |
| clusters 10⁵, 8 | 1.025 / 1.253 | 1.259 / 1.016 | 1.098 / 1.119 |
| cube 10⁶, 72 | 1.017 / 1.167 | 1.205 / 1.014 | 1.100 / 1.085 |
| Plummer 10⁶, 72 | 1.083 / 1.512 | 1.300 / 1.069 | 1.147 / 1.196 |
| clusters 10⁶, 72 | 1.065 / 1.326 | 1.363 / 1.051 | 1.185 / 1.174 |
| coincident 10⁶, 72 | 1.113 / 1.417 | 1.324 / 1.114 | 1.196 / 1.169 |
| cube 10⁵, 72 | 1.015 / 1.338 | 1.430 / 1.076 | 1.174 / 1.152 |
| Plummer 10⁵, 72 | 1.056 / 1.392 | 1.547 / 1.057 | 1.229 / 1.224 |
| clusters 10⁵, 72 | 1.056 / 4.390 | 1.936 / 1.051 | 1.413 / 2.001 |

The other configurations at 2–8 ranks stay within 1.07 in points and 1.13 in M2L work
with the cut by points. Which mix is right depends on how P2P and M2L costs compare (p, precision, machine),
which is why §3.5 leaves the work weight to Phase 6.

**With P2, k = 8 stays the recommendation.** Without P2, the coarser blocks win at 64–72
ranks (k = 1 to 2), at the price of the point balance there (1.13–1.54 at N = 10⁶,
72 ranks), and k = 8 up to 16 ranks; T4 would then make k depend on P. The simulation
ran every workload of §1.2 at k = 1, 2, 4, 8 and 16 on 2–72 ranks; the tables above are
its representative rows, and the raw output is not committed.

### 3.7 The top tree under P2: what each rank needs

With P2 a rank computes the locals of its own blocks' `Global` ancestors only. What it
still receives and computes for the top tree is everything else: the coarse gather
brings every block's multipole, and the global M2M forms every `Global` multipole. What
it **needs** is the multipoles its kept V rows read: the V sources of its own `Global`
ancestors. Those are `Global` boxes, blocks, or boxes below a block (which the multipole
exchange already brings as ghosts). Simulated as in §3.6 (O1 at k = 8):

| | 8 ranks, N = 10⁶ | 72 ranks, N = 10⁶ | 72 ranks, N = 10⁵ |
| --- | --- | --- | --- |
| top-tree multipoles a rank needs (max over ranks; cube, Plummer, clusters) | 56–997 | 284–1,108 | 304–1,015 |
| of which `Global` boxes | 0–53% | 35–69% | 35–70% |
| of which coarse blocks | 6–47% | 0–24% | 0–24% |
| the top tree every rank holds (blocks + `Global` boxes) | 330–1,700 | 3,100–4,800 | 2,900–4,700 |

At 72 ranks a rank therefore uses 5–25% of the block multipoles the gather brings, and
about half the `Global` multipoles it computes. It cannot simply compute fewer locally:
a needed `Global` box near the root (levels 2–3) has much of the domain below it, so
forming it locally takes most of the blocks. The gather is the price of computing the
top tree redundantly, not waste in it.

On one node with P ≤ 72 that price is small after P2 (*model*, from §3.6's counts):
at N = 10⁶ on 72 ranks the global M2M is 0.02–0.06 of a fair share and the gather about
2.8 MB per rank per evaluation (f64, p = 8; about 3–4 ms on locust, today's 1.2 ms scaled
by the block count), together about 5% of an evaluation. **Phase 5 keeps the replicated
top tree.** It does not scale to many ranks: everything in it grows like the number of
blocks, which grows like P. §14 quantifies that at 1,024–8,192 ranks and plans the
replacement.

## 4. Redistribution (C5.1)

### 4.1 API

Redesign §9, refined. In `nd_fmm_plan::redistribute`, kernel-agnostic:

```rust
/// Routing of items (points) to the ranks that own their leaves, and back (C5.1).
///
/// Received items are grouped by local leaf, in leaf order, and within a leaf ordered
/// by (origin rank, position on the origin).
pub struct Redistribution { /* per sent item: destination (permutation); per received
                               item: origin rank and position, leaf slot; counts per leaf
                               and per neighbour; the largest `per_item` that fits */ }

/// The reasons a [`Redistribution`] cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RedistributionError {
    /// The key of item `index` is invalid or not at the finest level.
    InvalidKey { index: usize },
    /// A count or displacement of the exchange does not fit MPI's `i32`.
    Overflow,
    /// Building failed on another rank.
    OtherRank,
}

impl Redistribution {
    /// Route the items with finest-level `keys` to the owners of their leaves.
    ///
    /// # Collective operation
    /// One all-to-all of counts, one all-reduce that agrees the errors and the largest
    /// `per_item`, one all-to-all-v of (key, position): every rank, empty included.
    pub fn new<C: CommunicatorCollectives>(octree: &Octree<'_, C>, plan: &Plan,
                                           keys: &[MortonKey]) -> Result<Self, RedistributionError>;
    /// The items this rank passed to `new`.
    pub fn nsent(&self) -> usize;
    /// The items this rank owns.
    pub fn nreceived(&self) -> usize;
    /// Received items per local leaf, in leaf order: the counts for `Evaluator::new`.
    pub fn counts(&self) -> &[usize];
    /// Origin (rank, position) of every received item, in received order.
    pub fn origins(&self) -> &[(u32, u32)];
    /// The largest `per_item` that `forward` and `backward` accept, the same on every rank.
    pub fn max_per_item(&self) -> usize;
    /// `per_item` values of each sent item to its owner, in received order.
    /// Collective: one all-to-all-v.
    pub fn forward<T: Equivalence + Copy + Default>(&self, payload: &[T], per_item: usize) -> Vec<T>;
    /// `forward` into `received` (`nreceived · per_item` values), without allocating
    /// beyond one send buffer.
    pub fn forward_into<T: Equivalence + Copy + Default>(&self, payload: &[T], per_item: usize,
                                                         received: &mut [T]);
    /// `per_item` values of each received item back to its origin, in the original item
    /// order. Collective: one all-to-all-v.
    pub fn backward<T: Equivalence + Copy + Default>(&self, results: &[T], per_item: usize) -> Vec<T>;
    /// `backward` into `returned` (`nsent · per_item` values).
    pub fn backward_into<T: Equivalence + Copy + Default>(&self, results: &[T], per_item: usize,
                                                          returned: &mut [T]);
}
```

Changes against redesign §9:
- **`_into` variants**, for `Fmm`'s charges and output on every evaluation (the
  allocation-free pattern Phase 4S T9 asked for).
- **`max_per_item`**, agreed in `new`'s all-reduce (the largest `per_item` for which
  every count and displacement, on every rank, fits `i32`). `forward` and `backward` then
  check `per_item` against a value every rank holds, and fail alike on every rank
  without a collective.
- **A payload of the wrong length** follows the exchanges' pattern (exchange.rs): the
  rank takes part with default values and then panics, so its neighbours complete the
  collective; no agreement per call. `Fmm` agrees the charge-vector length already.
- `nsent`, `nreceived`.

### 4.2 Collectives, memory and traffic

| Call | Collectives | Traffic per rank |
| --- | --- | --- |
| `new` | all-to-all (P counts); all-reduce (errors, `max_per_item`); all-to-all-v of (key u64, position u32) | 12 bytes per item moved, sent and received |
| `forward` | one all-to-all-v | `per_item` · s per item moved |
| `backward` | one all-to-all-v | the same, reversed |

Items that stay on the rank are copied, not sent (an all-to-all-v copies the rank's own
block). Memory: 4 bytes per sent item (the permutation), 12 per received item (origin,
slot), P counts and displacements, one send buffer per call. Linear in the items, O(P)
in the counts.

### 4.3 The order within a leaf

Received items of a leaf are ordered by (origin rank, origin position). The one-rank
reference of the tests is the `Fmm` over the union of every rank's points **in rank
order** (rank 0's, then rank 1's, …): its leaf order is stable in that order, which is
exactly (origin rank, origin position). So P2M, P2P and every per-leaf sum see the same
order on P ranks as in the reference.

A rank-count-independent order (by finest key, then the coordinates) was considered and
is **not** recommended:
- the one-rank `Fmm` orders a leaf's points by the caller's order; sorting it by key
  would change one-rank results, which the phase forbids;
- with the rank-order union as the reference, the order already agrees; what it would
  buy is equality *between input distributions*, which §5.4's tolerance covers;
- coincident points tie on key and coordinates, so a total order would still need the
  origin.

### 4.4 How `Fmm` uses it (T6)

`build`:
1. Steps 1–4 as today (the octree from every rank's own keys, with O1–O4's defaults).
2. Two `Redistribution`s, sources and targets, from the keys `build` already computed.
3. `forward` the f64 coordinates (`per_item` 3) of both; the owner converts them to
   leaf-scaled values (CONVENTIONS §3.13) in leaf order, as step 8 does today.
4. The counts per leaf from `counts()`; `max_leaf_points`, tables, operator, evaluator
   as today. Step 5's `PointsNotOwned` and its all-reduce go.
5. The output pass's order becomes the received order of the targets, which is leaf
   order: the owner scales its targets in leaf order and `backward` returns them.

`evaluate`:
1. The charge-length agreement, as today (§7.5 adds the device error to it).
2. `forward_into` the charges (`per_item` 1) to the owners, into leaf order: the host
   source chunks and, on a device, the upload buffer.
3. The six stages.
4. The output pass on the owner, in leaf order (on the device where it runs there,
   Phase 4S T9), then one `backward_into` of o values per target (φ, or φ and ∇φ,
   o = 1 or 4) and an unpack into `Output`, in the caller's order.

Collectives: per build two `new`s and two coordinate `forward`s (eight collectives), and
one all-reduce fewer (step 5); per evaluation two all-to-all-v more.

Timings: `BuildTimings::redistribute` (both `new`s and the coordinate forwards);
`StageTimings::forward_charges` and `StageTimings::backward_output` (the two moves, the
latter beside `output`).

### 4.5 No cheap path

When every point is already on its owner, the all-to-all-v's copy the rank's own block
and send nothing. Measured, such an all-to-all-v costs 0.2–0.8 µs (M3 Max at 8 ranks,
locust at 8 and 72), against an evaluation of milliseconds. A cheap path would need its
own agreement and a second code path; **none is recommended**.

### 4.6 Errors

`FmmError::PointsNotOwned` is removed. `FmmError::Redistribution(RedistributionError)`
covers `Overflow` (2³¹ values in one exchange, agreed on every rank); `InvalidKey` cannot
come from `Fmm`, whose keys come from points inside the domain. The docs of `build` and
of the crate say so.

## 5. Equal to one rank (requirement 2)

### 5.1 Every source of difference

| Source | Changes bits today? | With O1 and P1 |
| --- | --- | --- |
| the leaves (the tree) | yes, in 15 of 24 configurations at 4–8 ranks (§1.2): different near/far splits, a difference at the level of the FMM error | none: the one-rank leaves (O1) |
| the partition: which boxes are local, ghost or `Global` | no by itself: a `Global` box's multipole is the same M2M by octant as a local interior's, from the same children; ghost values are exact copies | none |
| the order of points within a leaf | equal to the reference's (§4.3); differs between input distributions | the same |
| **near rows (P2P) and X rows (P2L), by leaf index**: local leaves before ghost leaves on P ranks | **yes**: the only difference measured with equal trees (§1.2: every multipole equal; locals differ only where an X row exists) | none (P1) |
| V rows by offset index, W rows by box index (Morton order), M2M and L2L by octant | no: the same order on every rank count | none |
| the coarse gather, the exchanges | no: exact copies | none |
| the global pass, on every rank | no: identical inputs, identical order (redesign §7.4) | none |
| the operators, threads | no: per target, fixed order, bit for bit for every thread count | none |
| P2P scratch (`max_leaf_points`) | no: all-reduced | none |
| the output scaling | no: the same division on the owner | none |

**P1. Near and X rows ordered by the entry leaf's (level, key)** (`nd-fmm-plan`,
`PartialLevel::finish` in plan.rs: sort each row by `(leaves.level(j), leaves.key(j))`
instead of `j`). On one rank the local leaves are numbered by (level, key), so the rows
do not change and one-rank results stay bit for bit. On several ranks a ghost leaf sits
in the row where the one-rank row has it. The accumulation rule (operator.rs docs,
redesign §7.5) changes its wording from "by leaf index" to "by the source leaf's (level,
key), which is the leaf index on one rank". What it costs: rows are no longer ascending
in leaf index (plan tests that assume it change), and a split of rows into local and
ghost parts for overlap (redesign §10) would now reorder (§8.3 recommends none).

### 5.2 Measured

With equal trees (§1.2): 2–7 u_T relative L2 in φ and ∇φ (u_T = 2⁻⁵³, 2⁻²⁴), at most 17
u_T relative max, in f64 and f32, rising slowly with P (Plummer, N = 10⁵: 4.7e-16,
5.6e-16, 7.2e-16 at 2, 4, 8 ranks); every multipole equal, locals different only where
an X row exists, outputs 52–97% bit-identical; errors against the direct sum the one-rank
run's (ratio 1.0000 in f64, 0.9992–1.0008 in f32).

### 5.3 With P1: bit for bit

Measured on a scratch copy of the workspace with P1 applied (the one-line sort of
`PartialLevel::finish` for the near and X rows; M3 Max, owner input, N = 10⁵, f64 and
f32 at p = 8):
- **cube, Plummer and clusters at 2, 4 and 8 ranks** (the clusters at 2 ranks have no
  owner input today): every output value equal to the one-rank `Fmm`'s, φ and ∇φ, bit
  for bit (100,000 of 100,000); every multipole and every local equal; the direct-sum
  errors identical;
- **one rank unchanged**: the one-rank outputs (cube, Plummer, clusters; f64 p = 8, f32
  p = 3) hash to the same value with and without P1.

With the order within each rank's points reversed (`--reverse`: every leaf's points in
the opposite order, which is what a different input distribution does to a leaf), the
difference from the one-rank run is the same on 2, 4 and 8 ranks: relative L2 of φ
8.5e-16–1.04e-15 (f64) and 4.4e-7–5.8e-7 (f32), of ∇φ at most 3.1e-16 and 1.6e-7;
relative max of φ at most 1.33e-15 and 7.9e-7 (cube, Plummer, clusters and the
coincident set, p = 3 and 8). That is 7–10 u_T relative L2 and at most 13 u_T relative
max: what two input distributions can differ by.

### 5.4 The tolerance

**Proposed:** the host path on P ranks equals the one-rank `Fmm` over the union of every
rank's points in rank order bit for bit, given O1 and P1. That is the check of T6, on
every scenario and rank count, and it catches any defect of a ghost, an exchange or the
global pass, however small.

Between input distributions (different orders within leaves), and as the fallback if
P1 is rejected while the trees agree: **relative L2 ≤ 100 u_T over every target, for φ
and ∇φ: 1.1e-14 (f64) and 6.0e-6 (f32)**; the relative max is reported. The device on P
ranks is checked against the host on the same P ranks within Phase 4's FMM bounds (T8);
with the host bit for bit one rank, that bounds the device against one rank too.

Derivation:
- the measured differences from reordering are 2–7 u_T relative L2 for the rows (§5.2)
  and 7–10 u_T for the points within leaves (§5.3); a reordered sum of n terms differs by
  at most 2 γ_n κ relative, with κ = Σ|terms| / |Σ terms| (n ≤ 27 × 64 for the near field
  and κ of order 1–10), and typically by √n u κ (*model*);
- 100 u_T leaves a factor of at least 10 over every measured value, for larger P, the
  other workloads and p = 18;
- a defect is far larger: a ghost leaf of 64 sources missing at 64 targets changes the
  relative L2 by about √(64/10⁵) · 10⁻² ≈ 2.5e-4 (*model*, the cube at N = 10⁵), 2 × 10¹⁰
  times the f64 bound and 40 times the f32 one;
- against the README's provisional bounds (1e-12, 1e-5): 100 × tighter in f64, 1.7 × in
  f32.

**If O1 is rejected**, the trees differ in most configurations at 4 and 8 ranks and the
difference is at the FMM error's level: no rounding tolerance applies, and the check
falls back to "within 2 × the one-rank error against the direct sum", which would hide a
ghost defect of that size. That is the strongest reason for O1.

### 5.5 The reference in tests, and its cost

Every rank rebuilds the union locally: the scenarios draw a global point set and take
shares, so each rank knows every rank's share without communication. Each rank builds
the one-rank `Fmm` of the union on `SimpleCommunicator::self_comm()` and compares its
own targets (README, "The oracles"); no collective. At one rank the run is its own
reference and is skipped. The cost, on P > 1 only, is one one-rank build and
evaluation per scenario per rank, in parallel, so about the scenario's one-rank cost in
wall time. The one-rank debug run measured 59.9 s for `tests/mpi_exec.rs` at the start
of this task, at the limit of its one-minute budget, so T6 adds nothing at one rank.

## 6. The data-movement hook (C5.1, device)

### 6.1 Every host read and write of the evaluator, on several ranks

Re-derived from fmm-plan/src/evaluator.rs and exchange.rs (device-path §4.2 did it for
one rank):

| Stage | Function | Reads from the host stores | Writes into the host stores |
| --- | --- | --- | --- |
| construction | `Evaluator::new` → `Data::new` (`LevelBuffers::from_index`, `LeafStore::new`) | — | allocates the five stores zeroed |
| `reset` | `Evaluator::reset` → `Data::reset` → `LevelBuffers::clear` (multipoles, locals), `LeafStore::clear` (target output) | — | every multipole, local and target output (+0.0) |
| 1 | `Evaluator::exchange_sources` → `SourceExchange::forward` | `sources.chunk(j)` for every j of `send_leaves()` (local leaves: coordinates and this evaluation's charges) | `sources.range_mut(ghost_leaves())`: the ghost tail |
| 3 | `Evaluator::upward_global` → `CoarseExchange::gather` | `multipoles.chunk(level, i)` of this rank's blocks (`rank_blocks(rank)`); on one rank the root, before the global M2M has formed it (+0.0, never observed) | every other rank's block (`block(b)`), ghost boxes on this rank |
| 4 | `Evaluator::exchange_multipoles` → `MultipoleExchange::forward_all` → `forward(l)`, l = 0..=L | `multipoles.level(l).chunk(i)` for i in `send_boxes(l)` | `chunk_mut(i)` for i in `receive_boxes(l)` |

A store with the wrong layout makes an exchange take part with default values and then
panic; it writes nothing. Everything else that touches the stores is the caller's
(`sources_mut`, `local_sources_mut`, `target_input_mut`, `local_target_inputs_mut`) or
an operator call, which writes only its batch's output. Nothing else in `nd-fmm-plan`
reads or writes a store.

### 6.2 The events

device-path §4.5 as the starting point, refined: one "send" event covers everything an
evaluation sends of the multipoles, and fires before the coarse gather.

```rust
/// A data movement of the evaluator outside the operator calls, with the index lists it
/// uses (design §6). A "send" event comes before the evaluator reads the host store to
/// send; a "received" event after it wrote the received values into the host store. No
/// communication is pending when an event fires.
pub enum HostData<'a, T> {
    /// `reset` zeroed every multipole, local and target output.
    Reset,
    /// The source exchange is about to read the local chunks of `leaves`
    /// (`SourceExchange::send_leaves`; a leaf sent to two ranks appears twice).
    SendSources { leaves: &'a [u32], sources: &'a mut LeafStore<T> },
    /// The source exchange wrote the ghost tail: the leaves `leaves` of the numbering.
    ReceivedSources { leaves: Range<usize>, sources: &'a LeafStore<T> },
    /// Every multipole the evaluation sends is final: this rank's coarse blocks
    /// (`coarse.sent_blocks()`, empty on one rank) and every level's
    /// `exchange.send_boxes(l)`. Once, before the coarse gather.
    SendMultipoles {
        coarse: &'a CoarseExchange<T>,
        exchange: &'a MultipoleExchange<T>,
        multipoles: &'a mut LevelBuffers<T>,
    },
    /// The coarse gather wrote every other rank's blocks (`coarse.received_blocks()`); the
    /// values, packed, are `coarse.gathered()`.
    ReceivedCoarse { coarse: &'a CoarseExchange<T>, multipoles: &'a LevelBuffers<T> },
    /// The multipole exchange of `level` wrote `exchange.receive_boxes(level)`; the
    /// values, packed, are `exchange.receive_buffer(level)`.
    ReceivedMultipoles { level: usize, exchange: &'a MultipoleExchange<T>, multipoles: &'a LevelBuffers<T> },
}

pub trait FmmOperator: FmmSizes {
    /* the eight level calls */
    /// A data movement of the evaluator; the default does nothing (design §6).
    fn host_data(&mut self, event: HostData<'_, Self::Value>) {}
}
```

`CoarseExchange` gains `sent_blocks() -> Range<usize>` (this rank's blocks, empty on a
one-rank communicator) and `received_blocks() -> impl Iterator<Item = usize>` (every
other rank's); everything else the events need is public already (`send_leaves`,
`ghost_leaves`, `send_boxes`, `receive_boxes`, `receive_buffer`, `block`, `gathered`,
`chunk`).

Order in a blocking evaluation:

| Stage | Events and movements, in order |
| --- | --- |
| `reset` | zeroing; `Reset` |
| 1 | `SendSources`; the exchange; `ReceivedSources` |
| 2 | level calls only |
| 3 | `SendMultipoles`; the gather; `ReceivedCoarse`; the global M2M calls |
| 4 | for l = 0..=L: the exchange of level l; `ReceivedMultipoles(l)` |
| 5, 6 | level calls only |

**Why the multipole "send" moves to stage 3.** Every box any exchange sends is a local
leaf or local interior box: the exchanges skip `Global` boxes, and a ghost is never sent.
Its multipole is final after `upward_local` (P2M, the local M2M); the global M2M writes
only `Global` boxes, and the gather only other ranks' blocks. So all sends are known
before the gather, and one event lets a device download the coarse blocks and every
level's sends in one sync instead of two (§7). The events still come at the stage
boundaries `Fmm` sees.

On one rank every event comes, with empty lists: no ghost leaf, no sent block (the one
block, the root, is read only for the self-gather, whose value nobody uses), no ghost
box. An operator then handles one and several ranks alike, and the Phase 4 transfers do
not change.

### 6.3 The host path stays bit-identical

The default `host_data` does nothing; the evaluator's movements, their order and the
accumulation order do not change; `LaplaceOperator` and `PerPair` keep the default.
Firing an event is a match on an enum and a call per stage: 6 + L events per evaluation.

### 6.4 The shadow check without `IndexFmm`

In `fmm-exec/tests/` (test code, not a public type), `Shadow` wraps `LaplaceOperator`:
- it owns its five stores, laid out as the evaluator's (`LevelBuffers::from_index`,
  `LeafStore::new` with the evaluator's counts, ghost leaves included);
- every level call runs the wrapped operator on a batch rebuilt from its own stores:
  the batch types' fields are all public, and `LeafStore::range`/`range_mut` and
  `LevelBuffers::level`/`level_mut`/`parent_child_mut`/`child_parent_mut` are public. At
  the deepest level M2P's `multipoles` is the batch's own empty slice
  (`LevelSlice::empty` is crate-private, and the slice carries no data). **No new public
  constructor is needed**;
- it learns of the caller's data as a device does: the test hands it the leaf-scaled
  points once (`load_points`) and the charges per evaluation (`begin_evaluation`); of
  the evaluator's movements only through `host_data`:
  - `Reset`: zero its multipoles, locals and target output;
  - `SendSources`: write its local chunks of `leaves` into the host store;
    `ReceivedSources`: copy the host ghost tail into its store;
  - `SendMultipoles`: write its multipoles of the sent blocks and boxes into the host
    store; `ReceivedCoarse`, `ReceivedMultipoles`: copy the received slots from the host
    store (or the packed buffers) into its own.

The test drives `Evaluator` directly (the `Fmm` builds its own operator): the points of
each `tests/mpi_exec.rs` scenario through T5's `Redistribution` and
`nd_fmm_exec::geometry::leaf_coordinates`, one evaluator with `LaplaceOperator` and one
with `Shadow`, the same tables, `max_leaf_points` and P2P kernel. The shadow's target
output equals the plain evaluator's bit for bit at 1, 2 and 4 ranks. With the events
ignored it differs on 2 ranks in a scenario with ghosts, and the test asserts that.

### 6.5 Overlap and the events

The overlapped stages (§8.1) fire the same events with the same lists, at the same
logical points: `SendSources` before the source exchange is posted, `ReceivedSources`
after its wait; `SendMultipoles` before the multipole exchange is posted (now ahead of
the gather as in the blocking path); `ReceivedCoarse` after the gather;
`ReceivedMultipoles(l)` after level l's wait, before the first call of level l that
reads ghosts. No event fires while a request is pending on a buffer the event hands
out: the send buffers are filled after `Send…` returns, and `Received…` comes after the
wait.

## 7. The device on several ranks

### 7.1 What `DeviceOperator` does at each event

Checked against fmm-exec/src/device.rs and fmm.rs; **one correction to the briefs** (T1,
T8): the host source chunks do **not** hold this evaluation's charges when P2M, P2L and
P2P all run on the device (`host_sources` is false, Phase 4S T9: "the charge load …
skips the host's source store"). The device copy is current, the host's charges are not.

| Event | Device action | Transfers |
| --- | --- | --- |
| `Reset` | the zero kernels `begin_evaluation` launches today (moved here); `begin_evaluation` keeps the charge upload and scatter | none |
| `SendSources` | nothing: `Fmm` writes the charges of the sent leaves into the host chunks before the stages, on several ranks (the coordinates are there from build) | none |
| `ReceivedSources` | one upload of the host ghost tail (contiguous, the `LeafStore` layout) into the device source store's tail | 4 G_s s up |
| `SendMultipoles` | one `gather_columns` of the sent blocks' and boxes' columns into a packed device buffer, one download, and the host writes them into the host store's slots | (B_o + Σ_l S_l) n_c s down, 1 sync |
| `ReceivedCoarse` | one upload of the other ranks' blocks from `coarse.gathered()` (two contiguous ranges), one `scatter_columns` into the multipoles | B_r n_c s up |
| `ReceivedMultipoles(l)` | one upload of `exchange.receive_buffer(l)`, one `scatter_columns` into level l | R_l n_c s up |

Notation as device-path §4.1, plus: G_s ghost source points; B_o this rank's blocks
(zero on one rank), B_r the other ranks'; S_l and R_l the boxes sent and received on
level l. `Fmm` writes the sent leaves' charges (at most N_s values) on the host, on the
pool, beside the device upload of all charges.

The construction order changes: the device source store needs the ghost leaves' counts,
which `Evaluator::new` learns from the source exchange. `Fmm::build` builds the
evaluator with the host engine, then builds the device operator with the source store's
counts (every leaf of the numbering) and swaps it in (`Evaluator::operator_mut`; the
sizes are the same), then agrees device build errors with one all-reduce on every
backend (§9.1).

### 7.2 Transfers and syncs per evaluation

Every kind on the device, P ranks:

| Direction | Bytes | Calls |
| --- | --- | --- |
| up | N_s s (charges) + 4 G_s s (ghost tail) + B_r n_c s + Σ_l R_l n_c s | 1 + [G_s > 0] + [B_r > 0] + #{l : R_l > 0} |
| down | o N_t s (output) + (B_o + Σ_l S_l) n_c s | 1 + [anything sent] |
| syncs | | **2** with anything sent; 1 otherwise (one rank: 1, Phase 4's minimum) |
| launches added | | 1 gather if anything is sent, 1 scatter if B_r > 0, one per level with R_l > 0 |

The two syncs are unavoidable while the exchanges run on the host: the values to send
must reach the host before the gather (the sends ride in the same download). Example,
*model* from §1.2's sizes: the Plummer sphere at N = 10⁶ on 8 ranks, p = 8, f32, the
busiest rank: about 2.4 MB down (the sends) and 3.3 MB up (0.8 MB ghost tail, 0.1 MB
coarse blocks, 2.4 MB received multipoles) per evaluation beside the one-rank minimum
(0.5 MB of charges up, 2 MB of output down).

The ghost tail's coordinates do not change between evaluations; uploading only its
charges (G_s s instead of 4 G_s s) needs a charge slot list for the ghosts and changes
nothing else. It is left out: the source exchange is 0.3–1.5 ms of an evaluation of
seconds at this size (§1.2).

### 7.3 Kernels and device memory

- **Gather**: `nd_fmm_kernels::movement::gather_columns`, one launch over the whole
  multipole buffer with global column indices where every level has the same size
  (Laplace: n_c on every level), otherwise one launch per level. It exists.
- **Scatter**: a new `movement::scatter_columns` (assignment, distinct columns, the
  pattern of `scatter_values` with columns of n values). `scatter_add_columns` into
  slots that `Reset` zeroed would give the same values except a −0.0 received becomes
  +0.0, which breaks the bit identity of the host fallback. One kernel and its tests in
  `nd-fmm-kernels` (allowed by the README's scope for T8).
- **Memory added** per rank: the packed send buffer (B_o + Σ S_l) n_c s, the coarse
  receive buffer B_r n_c s, one receive buffer of max_l R_l n_c s reused per level (one
  stream orders an upload after the scatter that read the buffer before it), and the
  column lists (u32, uploaded once at build). The ghost slots exist already: the device
  stores have the plan's layouts, ghosts included (device-path §14). *Model*, Plummer
  N = 10⁶ on 8 ranks, p = 8, f64: about 6 MB.

### 7.4 A device per rank

- `Fmm::build` splits the communicator once (`split_shared`) and records the local rank
  in `DeviceReport`. Both machines have one GPU, and `nd_fmm_kernels::Device::open` takes
  no index: every rank opens the default device. A `Device::open_index` belongs to the
  first multi-GPU node, not to Phase 5 (a correction to T8's brief, which asks for the
  local rank modulo the device count).
- **CPU runtime**: `threads(n)` caps the units per cube per rank; ranks × n stay within
  the cores (device-path §11). GPU-shaped kernels on the CPU runtime keep their own cube
  size and may exceed it; the report says so.
- **Metal**: every rank opens the one GPU; runs share it, by hand, outside the sandbox.
- **CUDA on locust**: every rank opens the H100 in its own process; contexts are
  time-sliced (no MPS); kernel compilation once per process (seconds, cached by the
  driver in `CUDA_CACHE_PATH`).

### 7.5 Errors (decision 12)

A launch or transfer error surfaces at a download (device-path §12, F9). On several
ranks the mid-evaluation download (`SendMultipoles`) can fail; the rank then sends the
host store's values (zeros after `Reset`) and its neighbours compute wrong results
without knowing. The error must reach every rank, without a hang.

| Option | Cost | Covers |
| --- | --- | --- |
| an all-reduce at the end of `evaluate` (after the output download) | one per evaluation: 1.2–3.4 µs on locust at 8–72 ranks, 80 µs on the M3 Max at 8 (§1.2) | every device error of the evaluation, the last download's included |
| a flag in an existing exchange (an extra value in the coarse gather) | none | errors up to the mid-evaluation download, not the output's; puts a non-`Value` into a `Value` buffer |

**Recommendation: the all-reduce at the end of `evaluate`**, on the device path with
P > 1 (every rank has the same backend: step 1 agrees it, §9.1), and the `Fmm`'s kept
error folded into the existing charge-length agreement at the start of the next
`evaluate` (today a rank with `device_error` returns before the stages while the others
enter them, which would hang on several ranks). The failing rank returns
`FmmError::Device`, the others `FmmError::OtherRank`, from that evaluation on.

### 7.6 Tuning on several ranks

Tuned choices change bits (the M2L strategy, and the layouts beyond the bit-for-bit
families), and ranks tuning alone on a shared GPU would time each other's work. With a
tuning cache, rank 0 tunes and broadcasts the decision record (one broadcast per build,
§9.1), and every rank applies it; without a cache every rank takes the static rule, as
on one rank. Two runs with the same cache then give the same bits (requirement 3).

### 7.7 "Device-resident ghost buffers" (C5.2)

After §7.1–§7.3 the ghost data never stage through a host copy of a device store: the
packed buffers are the exchanges' own, the host holds them only between the download
and MPI. On unified memory (the M3 Max; GH200's coherent NVLink-C2C) the copies could
be avoided by mapping, which CubeCL 0.11.0-pre.4 does not offer through `nd-fmm-kernels`;
at 2–5 MB per evaluation (*model*, §7.2) it is not worth a kernel-crate API in Phase 5.
Nothing more is proposed.

## 8. Overlap (C5.2)

### 8.1 The order-preserving overlap

Two overlapped stages, each one evaluator method that posts, works and waits (a scoped
request cannot outlive its method):

| Stage | Replaces | Does |
| --- | --- | --- |
| `exchange_sources_and_upward_local` | stages 1, 2 | `SendSources`; post the source exchange; P2M and the local M2M level by level, `test` between calls; wait; `ReceivedSources` |
| `far_field` | stages 3, 4, 5 | `SendMultipoles`; post the multipole exchange of every level; the coarse gather (blocking, §8.5); `ReceivedCoarse`; the global M2M; for l = 1..=L: wait for level l (the levels with ghost boxes; `ReceivedMultipoles(l)`), then L2L, M2L, P2L of l, `test` between calls; wait for any level left |
| `evaluate_leaves` | stage 6 | unchanged |

Order preservation: no level call moves, so every value receives its contributions in
the order of redesign §7.5 and the output is bit-identical to the blocking path. The
upward pass needs no ghost source (P2M and M2M read local data); level l of the downward
pass needs ghost multipoles of level l only (V), and stage 6 those of l + 1 (W), all
received by the end of `far_field`.

The public stages: `Evaluator::evaluate_overlapped()` runs the three; the blocking six
stay public and unchanged. The debug order check gains the overlapped stages as an
alternative path (`Stage::SourcesAndUpward`, `Stage::FarField`, then
`Stage::EvaluateLeaves`).

### 8.2 Expected gain (*model*)

Measured inputs (§1.2, the real `Fmm`, max over ranks, release, one thread per rank):
the pure exchange times, and the stages they can hide behind:

| f64, p = 8 | P | source exchange / `upward_local` | multipole exchange / work before the deepest level's wait | exchanges, share of the evaluation |
| --- | --- | --- | --- | --- |
| locust, cube 10⁶ | 8 | 0.29 ms / 32 ms | 0.51 ms / ≈ 250 ms (*model*: L2L, M2L, P2L of levels 2–4, about 1/7 of the 1,796 ms downward) | 0.4% |
| locust, Plummer 10⁶ | 8 | 0.35 ms / 63 ms | 1.37 ms / ≥ 400 ms (*model*) | 1.2% |
| M3 Max, cube 10⁶ | 8 | 1.43 ms / 25 ms | 3.40 ms / ≈ 330 ms (*model*) | 0.6% |
| M3 Max, Plummer 10⁶ | 8 | 1.30 ms / 52 ms | 4.37 ms / ≥ 400 ms (*model*) | 1.2% |

Every exchange has at least 17 times its own duration of work to hide behind (the
source exchange 17–180 ×, the multipole exchange 90–490 ×). The exception is the coarse
levels' messages (levels 2 and 3, about 10–15% of the multipole bytes at 8 ranks), whose
waits come after only the gather, the global M2M and the few calls of the coarser
levels; they may stay partly exposed, at most their transfer time, about 0.1–0.5 ms
(*model*). On one node,
though, an exchange is a memory copy by a CPU (shared memory, single-copy on Linux), and
a copy cannot overlap computation on the same core: §1.2's 4 MB messages on locust still
spent their copy time in the wait. What overlap can hide on one node is the
synchronisation and the transfer's latency, at most the exchanges' share of an
evaluation, **0.4–3% (measured share; the gain is less)**. The load imbalance the
blocking path shows as wait at the first collective after the upward pass (40–65 ms of
`upward_global` at N = 10⁶ on 2–8 ranks today) is not communication; T4's partition
removes it, overlap would not.

At a network's bandwidth the exchanges become real transfers (DMA, no CPU copy), and
overlap hides them (*model*, per rank, the largest sends of §1.2):

| | source exchange | multipole exchange | coarse gather (received) |
| --- | --- | --- | --- |
| Plummer 10⁶, 8 ranks, f64: bytes | 1.6 MB | 4.9 MB | 0.21 MB |
| at 10 Gbit/s (1.25 GB/s) | 1.3 ms | 3.9 ms | 0.17 ms |
| at 100 Gbit/s | 0.13 ms | 0.39 ms | 0.02 ms |
| work behind it (measured, locust) | `upward_local` 63 ms | ≥ 400 ms | none (blocking) |
| Plummer 10⁶, 72 ranks, f64 (today's partition): bytes | 0.7 MB | 15.6 MB (rank 0) | 1.07 MB |
| at 10 Gbit/s | 0.5 ms | 12.5 ms | 0.86 ms |
| work behind it (*model*: the 8-rank figures / 9) | 7 ms | ≥ 45 ms | none |

Hidden at both bandwidths, with the order kept: **the coarse levels hold enough downward
work** for the multipole exchange, because it is posted before the gather, not after the
global pass. The coarse gather stays exposed: 0.02–0.9 ms per evaluation at these sizes.

### 8.3 Reordering options (decision 9)

| Option | What changes in the order of redesign §7.5 | Recommendation |
| --- | --- | --- |
| P2P before L2P (P2P overlaps the multipole exchange) | every target's output sums near before far: all output bits change | not offered |
| local before ghost parts of near and X rows | with P1 the rows are in (level, key) order, so a split reorders every row with a ghost; without P1 it keeps the order, but then multi-rank is no longer bit for bit one rank | not offered |
| P2P into its own buffer, added at the end | a different rounding for every target | not offered |

The model of §8.2 leaves no communication exposed that a reordering would hide, so
decision 9's condition for offering one ("only if T1's model shows the order-preserving
overlap leaves communication exposed") is not met. **None in Phase 5.**

### 8.4 The non-blocking mechanism (decision 10)

| | scoped rsmpi point-to-point | rlst `MPI_Ineighbor_alltoallv` | rsmpi non-blocking dense all-to-all-v |
| --- | --- | --- | --- |
| safety | safe: `request::scope` with `immediate_send`/`immediate_receive_into` (rsmpi 0.8.2) | needs `unsafe` in rlst behind a safe API; an upstream change and a new pinned release | safe |
| counts | rebuilt from public data: per neighbour k, the values are `send_offsets()[a_k..b_k]` with a_k, b_k prefix sums of `send_counts()` (likewise receive); rlst's private value counts are not needed | rlst's own | O(P) arrays of mostly zeros |
| messages | one per neighbour and direction | one collective | P per rank (zero-size included, implementation-dependent) |
| progress | `test` calls | the same | the same |

**Recommendation: scoped point-to-point.** Details:
- **Communicator and tags.** Each `GhostCommunicator` has its own graph communicator
  (`forward_comm()`, created with `reorder = 0`, so its ranks are the plan
  communicator's). The source exchange and every level's multipole exchange post on their
  own `forward_comm()`, so messages of different exchanges never match each other; within
  one, one message per neighbour and direction, tag 0, in MPI's non-overtaking order.
- **The rank's own messages**: none (a ghost is never owned by its holder; the builders
  never add one).
- **Empty neighbours**: a neighbour whose value count is zero (a source exchange of
  empty leaves) is skipped on both sides; both know the count from the build.
- **Requests**: plain `Request`s in a `Vec`, `Request::test` (it returns the request back
  if unfinished) and `wait`. Not `RequestCollection::test_some`/`test_any`/`wait_some`,
  which panic once every request has completed (§1.2).
- **Lifetime**: each overlapped stage opens one `request::scope`; the send and receive
  buffers belong to the exchanges (fields of the `Evaluator` disjoint from the stores),
  and the receive buffers are split per neighbour with `split_at_mut`. Every request
  is waited for before the scope ends.
- **Progress**: one `test` pass over the pending requests after every level call of the
  overlapped stages (2L + 1 calls in the upward pass, 3L in the downward), on the calling
  thread (MPI at `Funneled`; no worker calls MPI). Measured (§1.2): without MPI calls a
  message of 64 kB or more does not progress at all. With a `test` every 2 ms a 1 MB
  message is hidden on the M3 Max; on locust it needs one every 0.5 ms, and a 4 MB
  message's copy happens in the wait whatever the interval (a single-copy transfer is one
  copy by the receiving CPU). Level calls at N = 10⁶ last tens to hundreds of
  milliseconds, so on locust part of each large message's copy will land in a wait; that
  copy is CPU time on one node and could not overlap computation anyway (§8.2). Splitting
  level calls for more `test` calls is not proposed: it would need sub-batches of a view.

### 8.5 The coarse gather

`immediate_all_gather_varcount_into` exists, but nothing can hide the gather: its inputs
(this rank's blocks) are complete only at the end of `upward_local`, and its outputs feed
the global M2M, whose outputs feed every local through L2L. What the design does
instead: the multipole exchange is posted before the gather, so the gather's time is
also time in which those messages travel.

### 8.6 "Communication hidden" made measurable (decision 11)

`StageTimings` gains, per exchange (sources, multipoles): `total` (post to completion,
wall), `exposed` (time blocked in the waits) and `progress` (time in the `test` calls).
**Criterion**: on the benchmark case, the cube and the Plummer sphere at N = 10⁶, f64,
p = 8, 8 ranks with one thread each, release, on each machine, the median over at least
10 evaluations of the maximum over ranks of exposed(sources) + exposed(multipoles) is at
most 10% of the median maximum of `exchange_sources` + `exchange_multipoles` in the
blocking path of the same build. Both are free of load-imbalance wait: the evaluation
starts with an agreement, and the gather synchronises the ranks before the multipole
waits. The coarse gather and `progress` are reported, not part of the criterion. On one
node a hidden copy moves into `progress`, and on locust a large message's copy can stay in
the wait (§8.4); so T9 reports, beside the criterion, `progress`, the bytes received with
the copy time they imply at the machine's measured copy bandwidth, and the evaluation time
with and without overlap. If the criterion fails only by that copy time, T9 says so and the
sign-off of T9 decides.

## 9. Collectives and memory

### 9.1 Every collective, in order (after T4–T9)

Every rank enters every row, a rank with no points with empty slices and zero counts;
no collective sits in a branch only some ranks take.

| Call | Collectives, in order | Changed by |
| --- | --- | --- |
| `Octree::new` (P > 1) | `linearize`: the sample sort (all-reduce of sizes; two `gather_to_all` of samples and weights, each an all-gather of counts and an all-gather-v; the global minimum; an all-to-all of counts, an agreement and an all-to-all-v), the successor exchange; **O1: one all-reduce per refinement round, and one more if 2:1 balancing changes the blocks** (replacing the `gather_to_all` of `compute_coarse_tree` and the weights' all-reduce); the key move (all-to-all-v); `balance` (all-reduce of the deepest level, `linearize` again); the key move again; `generate_all_keys` (two `gather_to_all`, one all-to-all-v). The partition bounds come from the replicated coarse tree without communication, so the two bound `gather_to_all`s of the draft (the bins and `get_tree_bins`) are gone (T4, accepted 2026-10-08) | T4 |
| `Plan::new` | all-reduce (level count), all-reduce (validity) | — |
| `Redistribution::new` | all-to-all (counts), all-reduce (errors, `max_per_item`), all-to-all-v (key, position) | T5 |
| `Redistribution::forward`, `backward` | one all-to-all-v | T5 |
| `Evaluator::new` | all-reduce (validity); `SourceExchange::new` (all-reduce; rlst: three validity all-reduces, an all-to-all, two graph creates, two neighbour all-to-alls; all-reduce); `MultipoleExchange::new` (all-reduce; the rlst build per level; all-reduce); `CoarseExchange::new` (all-reduce; communicator duplicate; all-gather; all-gather-v; all-reduce) | — |
| `Fmm::build` | step 1 all-reduce (agreement, which also agrees the backend); step 2 all-reduce; step 3 (extent pair, the bounding box, agreement); `Octree::new`; `Plan::new`; two `Redistribution::new`, two coordinate `forward`s; step 7 all-reduce (max leaf points); with a tuning cache and a device, one broadcast of rank 0's decisions; `Evaluator::new`; **one all-reduce agreeing the device operator's build (every backend)**. Step 5's all-reduce (`PointsNotOwned`) goes | T6, T8 |
| `Fmm::evaluate`, blocking | all-reduce (charge length and a kept device error); `forward` of the charges; neighbour all-to-all (sources); all-gather-v (coarse); L + 1 neighbour all-to-alls (multipoles); `backward` of the output; with a device on P > 1, one all-reduce (device errors) | T6, T8 |
| `Fmm::evaluate`, overlapped | the same, with the source and multipole exchanges as point-to-point requests inside the two overlapped stages (§8.1); the coarse gather stays an all-gather-v, between posting and waiting for the multipole exchange | T9 |

New collectives per build: two `Redistribution`s (six), two forwards, the device
agreement (one), the tuning broadcast (device with a cache); minus one (step 5). Per
evaluation: two all-to-all-v's, and on the device path with P > 1 one all-reduce.

### 9.2 Memory per rank

Notation: N_s, N_t the points the rank owns (≈ N/P after T4); G_s ghost source points;
J_loc, J_gh local and ghost leaves; K = K_loc + K_gh + K_rep the boxes held: local, the
ghost halo, and the replicated part (`Global` boxes, every rank's coarse blocks, the keys
around `Global` boxes that `generate_all_keys` gathers to every rank, with the
ghost-children layer); B the coarse blocks; E the list entries.

| Data | Bytes | Grows like |
| --- | --- | --- |
| multipoles, locals (a slot for every held box, redesign §5.1) | 2 n_c s K | N/P + halo + replicated |
| source store, target input, target output | (4 (N_s + G_s) + 3 N_t + o N_t) s | N/P + halo |
| box index, leaf numbering | 32 K + 13 (J_loc + J_gh) | as K |
| views | about 12 E_V + 4 (E_U + E_W + E_X) + … (device-path §4.1) | N/P + halo |
| `Redistribution` (two) | 4 per sent and 12 per received point, O(P) counts | N/P, P |
| exchange buffers | the values sent and received, s each; the coarse gather B n_c s | halo; **B** |
| coarse exchange keys and blocks | 16 B | **B** |
| partition bounds | 8 P | **P** |

What is replicated on every rank: the coarse blocks (B), the `Global` boxes (measured
B/7: 31 of 218 … 236 of 1,653), the keys around them, the gathered multipoles of every
block per evaluation (B n_c s), and the O(P) partition data. After O1, B grows linearly
in P: 1,660–4,285 blocks at 72 ranks, 23–60 per rank (simulated, §3.2). Measured (today's
partition): K_gh + K_rep ≈ 4.5 B at 72 ranks (cube, N = 10⁶: 6,587 held boxes of which
about 520 local).

*Model* of the crossover: K_gh + K_rep ≈ 4.5 c_B P with c_B ≈ 40 blocks per rank, and
K_loc ≈ (8/7) N/(30 P) (about 30 points per leaf on the cube): the replicated part
dominates the coefficient memory beyond P ≈ √(N/4,700), about 15 at N = 10⁶, 46 at
N = 10⁷, 146 at N = 10⁸. Measured: 40% of the held boxes are not local at 8 ranks and
92% at 72 (cube, N = 10⁶). In bytes it is small at these sizes (8.5 MB of coefficients
per rank at 72 ranks, cube 10⁶, p = 8, f64), but it is what limits strong scaling at
small N/P and on a cluster: T10 reports it, and reducing it (fewer keys gathered around
`Global` boxes, or a reduction tree instead of the replicated global pass) is a Phase 6
question. The replicated *work* of the global pass is §3.6's: P2 removes the downward
part, and the global M2M (8 M2M per `Global` box) stays. **At cluster scale this section's replicated
part is the limit**: at 8,192 ranks the blocks number about 260,000 and the held
replicated boxes about 1.2 million per rank, 1.5 GB of coefficient slots per rank at p = 8
in f64 (*model*, §14.2). §14 plans the change.

## 10. Testing

### 10.1 The layers per task

| Layer | Task | Where | Ranks | Oracle |
| --- | --- | --- | --- | --- |
| partition and leaves | T4 | `nd-octree` unit tests; `examples/test_mpi_weighted_partition.rs` (registered); the existing examples | unit; 1 and 3 (existing), 2, 4, 8 (new) | the one-rank tree (gathered leaves, key for key); per-rank weight within 1 + w_max/(W/P) of the mean (§3.2, O3) on the four workloads |
| lists, exchanges, recording operator | T2, T5 (P1, P2), T7 (events), T9 (non-blocking) | `nd-fmm-plan` `tests/mpi_regressions.rs` | 1, 2, 4 (CI), 8 by hand | the list oracle (with P2, a `Global` box that is not an ancestor of an own block has empty rows; the oracle expects that); the exchange checks; the recording operator (pairs once each, call order, groupings, events) |
| a count FMM (kept value check) | T2 | `tests/mpi_regressions.rs`: a kernel-free operator in which every multipole, local and output counts source points (P2M adds 1 per point, every other kind adds its input); every target's output equals the global number of sources, exactly | 1, 2, 4 | the global source count; a second evaluation bit for bit |
| redistribution round trips | T5 | `src/redistribute_tests.rs`; `tests/mpi_regressions.rs` | unit; 1, 2, 4, 8 | the owner (`local_leaf`), the identity of forward then backward, the order within leaves |
| multi-rank `Fmm` against one rank | T6 | `fmm-exec/tests/mpi_exec.rs`, every scenario; the ignored gate `tests/multi_rank.rs` | 1, 2, 4 (CI), 8 by hand; gate 2, 4, 8 | **bit for bit** the one-rank `Fmm` over the union in rank order (P > 1); two input distributions within 100 u_T; `direct_sum` as before |
| the shadow check | T7 | `fmm-exec/tests/` (Evaluator-driven, §6.4) | 1, 2, 4; 8 by hand | the plain `LaplaceOperator` evaluator, bit for bit; differs with the events ignored |
| the device on several ranks | T8 | `tests/mpi_exec.rs` (CPU runtime), `tests/device_fmm.rs` or a new executable (ignored), `tests/device_metal.rs`, `tests/device_cuda.rs` | 1, 2, 4 | the host fallback bit for bit the host path on the same ranks; the default within Phase 4's FMM bounds of the host on the same ranks; the transfers of §7.2; an injected error agreed on every rank |
| overlap against blocking | T9 | `tests/mpi_regressions.rs` (exchanges), `tests/mpi_exec.rs` (every scenario with `overlap(true)`) | 1, 2, 4, 8; a 100-evaluation stress at 8 | the blocking path, bit for bit |
| ignored gates | T6, T8 | release, own executables | 2, 4, 8 | as their rows |

### 10.2 Input distributions

Every scenario draws its global point set from a seed on every rank. The four inputs of
the README, each derived locally:
- **all on rank 0**: rank 0 takes every point;
- **a seeded random share**: point i goes to rank `hash(seed, i) % P`;
- **the octree's partition**: after T4 the partition does not depend on the input, so one
  `Octree::new` on any input gives the owners (a fixed point at once);
- **one empty rank**: the share, with the last rank's points given to rank 0.

Every rank can rebuild every rank's input, hence the union in rank order for the
reference, without communication.

### 10.3 The CI job and the budgets

T3's job runs the MPI test executables at 2 and 4 ranks. The one-rank debug run stays
under a minute; `tests/mpi_exec.rs` took 59.9 s at the start of this task, so the
references of T6 run only at P > 1, and T3's measurement is repeated after T6 to fix the
job's budget at 2 and 4 ranks.

### 10.4 A rank count a test cannot run

A test that needs a rank count checks `comm.size()` first, on every rank, and returns
with a printed "skipped at P ranks" on every rank, before any collective; never a
panic, never a branch around a collective. With O4 the only such case left is a
scenario that needs at least two ranks (for example "one empty rank").

## 11. Scaling method (C5.3)

For T10, which measures and reports; no number here is a target.
- **Strong scaling**: the cube and the Plummer sphere at N = 10⁶ (and 10⁵, to show where
  the work runs out), f64 p = 3 and 8, f32 p = 8, the default strategy; one thread per
  rank. M3 Max: 1, 2, 4, 8, 12 ranks. locust: 1, 2, 4, 8, 16, 32, 64 and 72 ranks (72
  uses every core; 64 leaves the system's own processes room, both reported). Overlap
  off and on.
- **Weak scaling**: N = 10⁵ per rank, the cube and the Plummer sphere, the same ranks.
- **Ranks × threads at a fixed core count**: the M3 Max at 12 (1 × 12, 2 × 6, 4 × 3,
  6 × 2, 12 × 1), locust at 72 (1 × 72, 4 × 18, 8 × 9, 18 × 4, 72 × 1); N = 10⁶,
  p = 8, f64.
- **Repetitions and statistics**: one warm-up and ten evaluations per build; per rank the
  median per stage; over ranks the max, min and mean, and the compute stages' max/mean;
  the build once per configuration, three builds where its time is reported.
- **Per stage and rank**: forward of the charges, the source exchange (total, exposed),
  `upward_local`, the coarse gather and the global M2M separately (the gather's time
  is today's imbalance wait), the multipole exchange (total, exposed), `downward`,
  `evaluate_leaves`, the output pass, the backward of the output; the build by part
  (domain, octree, plan, redistribution, tables, evaluator with its exchange builds);
  bytes and messages per exchange; memory by §9.2's formula, and the peak resident size
  where the machine reports it without a new dependency (`/proc/self/status` on
  locust).
- **Biases, stated in the report**: one node over shared memory, where an exchange is a
  copy by a CPU and costs no NIC time (§8.2); memory bandwidth shared by the ranks, which
  slows memory-bound stages as ranks grow; on the M3 Max no process binding, efficiency
  cores possible, and background load (state it); on locust `--report-bindings`, one
  NUMA node, and the load checked before and after every run (README, "Ranks on
  locust").
- **The inter-node command**, documented, not run:
  `mpirun -n 64 --map-by ppr:8:node --bind-to core target/release/examples/scaling
  --dist cube --n 1000000 --precision f64 --p 8 --overlap on` (Open MPI; `srun -n 64
  --ntasks-per-node=8 …` under Slurm). What it would settle: the exchanges at network
  bandwidth against §8.2's model, the value of overlap there, and scaling beyond one
  node with the replicated part of §9.2. At thousands of ranks that replicated part is
  the limit, and §14 plans its replacement; the harness must already generate points
  per rank by index and check rank-count invariance (§14.4, S7).

## 12. Task check

| Task | Modules | Tests | Brief change proposed |
| --- | --- | --- | --- |
| T2 | `fmm-plan/src/{lib, index_fmm (removed), evaluator_tests}.rs`, `tests/mpi_regressions.rs`, `Cargo.toml` | as the brief, plus the count FMM of §10.1 | the count FMM as the kept value check (the brief allows "a minimal operator in the test file"; this design asks for it) |
| T3 | `.github/workflows/run-tests.yml` | as the brief | measure again after T6 (the references at P > 1) before fixing the budget |
| T4 | `octree/src/octree.rs` (`OctreeOptions`, `PartitionWeight`), `octree/src/octree/implementation.rs` (coarse tree by weight, the cut, empty ranges, bounds, the root as a block); `examples/test_mpi_weighted_partition.rs` | §3.4 | (a) **the defaults change on several ranks** (O1–O4; one rank unchanged), against "the defaults keep today's behaviour"; (b) the weights need not travel through the parallel sort: each round counts the input keys per block; (c) no agreed error and no `Result` from `Octree::new`: ranks without blocks (O4) replace the panic; (d) acceptance adds: the leaves equal the one-rank leaves |
| T5 | `fmm-plan/src/redistribute.rs` (+ `redistribute_tests.rs`), `plan.rs` (P1, P2), `operator.rs` and `evaluator.rs` docs (the accumulation rule's wording; the zero locals of pruned `Global` boxes) | §10.1 | (a) add P1, with a check that rows are in (level, key) order; (b) add P2, each in its own commit, with the one-rank plan unchanged and every `tests/mpi_exec.rs` output bit for bit at 1 and 2 ranks before and after; (c) a payload of the wrong length takes part and panics (the exchanges' pattern), no agreement per call; (d) the API of §4.1 (`max_per_item`, `_into`, `nsent`, `nreceived`) |
| T6 | `fmm-exec/src/fmm.rs` (§4.4), `tests/mpi_exec.rs`, a new `tests/multi_rank.rs`, `nd-fmm-validate` `fmm_accuracy` | §10.1 | (a) the multi-rank check is bit for bit against the union-in-rank-order reference, and within 100 u_T between two distributions; (b) `FmmError::Redistribution` replaces `PointsNotOwned`; (c) "a tiny problem on more ranks than the coarse tree has blocks" runs (one block, the other ranks empty), and `single leaf` runs on every rank count |
| T7 | `fmm-plan/src/{operator, evaluator, exchange}.rs`; the recording operator; the shadow in `fmm-exec/tests/` | §6.4 | (a) one `SendMultipoles` before the coarse gather covers the coarse blocks and every level's sends (no separate coarse "send"); (b) `CoarseExchange::{sent_blocks, received_blocks}`; (c) the shadow drives `Evaluator` directly, since `Fmm` takes no custom operator |
| T8 | `fmm-exec/src/{device, fmm}.rs`; `fmm-kernels/src/movement.rs` (`scatter_columns`) | §10.1 | (a) correction: the host source chunks do not hold the charges with every source kind on the device; `Fmm` writes the sent leaves' charges; (b) correction: `Device::open` takes no index, every rank opens the one device; (c) the transfer formula of §7.2, two syncs; (d) the device operator is built after `Evaluator::new` and swapped in |
| T9 | `fmm-plan/src/{exchange, evaluator}.rs`; `fmm-exec/src/fmm.rs` (`FmmBuilder::overlap`, `StageTimings`) | §10.1 | (a) the multipole exchange is posted before the coarse gather, not only behind the coarser downward levels; (b) the coarse gather stays blocking; (c) `Request::test`/`wait`, not `RequestCollection::test_some` (rsmpi defect); (d) the criterion of §8.6; (e) no device part |
| T10 | `nd-fmm-validate` (`scaling`), results, design documents | as the brief | report the gather's imbalance wait separately, the replicated part of §9.2 at 64–72 ranks, memory by formula (`/proc/self/status` on locust), and per rank the M2L and P2P work against the time at 16–72 ranks, the input to question 12; the harness generates points per rank by index and checks rank-count invariance; one oversubscribed run on locust at 256 and 512 ranks for correctness and memory (§14.5) |

**Facts in the briefs that this design found to differ** (not changed in the briefs; for
the sign-off):
- T1 and T8: "the host source chunks … already hold the coordinates and this evaluation's
  charges" holds only with P2M, P2L or P2P on the host fallback (Phase 4S T9,
  `host_sources`).
- README "Risks" and T4: "a small `Fmm` on 8 ranks can reach" the fewer-blocks panic:
  measured, 3 points on 8 ranks build; a small `max_level` reaches it. Under O1 small
  problems would reach it routinely, which is why O4 removes it.
- T8: "the index from the rank's position on its node … modulo the device count":
  `nd_fmm_kernels::Device::open` takes no index; both machines have one GPU.
- README "Non-blocking communication": rsmpi 0.8.2's `RequestCollection::test_some`,
  `test_any` and `wait_some` panic once every request has completed; use `Request::test`,
  `wait`, `test_all` or `wait_all`.

## 13. Questions for sign-off

**Signed off on 2026-10-08: every recommendation below is accepted, questions 1–14, and so
are the further choices listed after the table.** docs/phase5/README.md records the
answers as decisions 1 and 6–13.

| # | Question | Recommendation |
| --- | --- | --- |
| 1 | The requirements and tolerances (README decision 1) | Accept 1–10, with: requirement 2 — bit for bit the one-rank `Fmm` over the union in rank order on the host path (with O1 and P1), and 100 u_T relative L2 (1.1e-14 f64, 6.0e-6 f32) between input distributions and if P1 is rejected (the device against the host on the same ranks within Phase 4's bounds); requirement 7 — the source exchange's sends need no download (§7.1); requirement 8 — the replicated part grows linearly in P, and is stated (§9.2) |
| 2 | Each `nd-octree` change (decision 6) | Accept O1 (coarse tree by weight from the root, blocks are nodes of the one-rank tree, k = 8 with P2, question 11), O2 (points as the default weight), O3 (the nearest-boundary cut), O4 (ranks without blocks instead of the panic); O5: no new lookup. Defaults change on several ranks; one rank unchanged (§3) |
| 3 | The tolerance of requirement 2, and the order within leaves (decision 7) | Bit for bit with O1 and P1 (P1: near and X rows by the entry's (level, key), in T5); 100 u_T otherwise; order within a leaf (origin rank, origin position); no rank-count-independent point order (§4.3, §5) |
| 4 | The redistribution API and the cheap path (decision 8) | §4.1: (key, position) on the wire, `_into` variants, `max_per_item` agreed at `new`, a wrong payload length takes part and panics; **no cheap path** (0.2–0.8 µs measured) |
| 5 | The hook's events and the shadow check | Six events (§6.2), the multipole "send" once before the coarse gather; `CoarseExchange::{sent_blocks, received_blocks}`; the shadow drives `Evaluator` directly, no new public constructor (§6.4) |
| 6 | Device errors on several ranks (decision 12) | One all-reduce at the end of `evaluate` on the device path with P > 1, and a kept error folded into the charge-length agreement (§7.5) |
| 7 | The overlap order, and any opt-in reordering (decision 9) | Order-preserving only, with the multipole exchange posted before the coarse gather; no reordering offered in Phase 5 (§8.1–§8.3) |
| 8 | The non-blocking mechanism (decision 10) | Scoped rsmpi point-to-point on each exchange's graph communicator, `Request::test` after every level call; no rlst change (§8.4) |
| 9 | The "communication hidden" criterion (decision 11) | §8.6: exposed waits of the source and multipole exchanges ≤ 10% of their blocking time, cube and Plummer at N = 10⁶, f64 p = 8, 8 ranks × 1 thread, on each machine |
| 10 | A device overlap part in T9 | None in Phase 5: the device runs the overlapped stages with the same events, for correctness; device time is not a scaling figure (decision 3) |
| 11 | (new) The replicated global pass: P2 | Accept P2 in T5: a rank computes only the `Global` locals its own blocks descend from; bit for bit the same values; removes up to 6 fair shares of repeated M2L work at 72 ranks (§3.6). The global M2M stays whole |
| 12 | (new) The partition weight at 16–72 ranks | Points (O2) for Phase 5, which meets the 2–8-rank gates; at 16–72 ranks they leave M2L work 1.05–1.51 of the mean at N = 10⁶ (§3.5). A second cut by modelled work, after the leaves are known, is a Phase 6 candidate, decided by T10's per-rank work |
| 13 | (new) The top tree (§3.7, §14.4 S2) | Phase 5 keeps the replicated top tree with P2: about 5% of an evaluation at N = 10⁶ on 72 ranks (*model*). It does not scale beyond one node; its replacement, an owned top tree with an order-preserving reduction and a fixed-size replicated top, is the scale-out phase's second package |
| 14 | (new) Cluster scale: the scale-out plan (§14) | Accept §14 as the direction: Phase 5 keeps §14.5's constraints (no new P-growing per-rank structure; P2 written for S2; the harness's per-rank point generation and rank-count invariance; an oversubscribed locust run at 256–512 ranks). A scale-out phase follows Phase 5, ahead of the Phase 6 list: S1 (locally essential top tree, `nd-octree` and `nd-fmm-plan`), S2, S7, S3, then S4–S5 as measured. It wants a cluster (README decision 2 revisited); without one it is accepted on oversubscribed one-node runs and estimated scaling, marked as such (§14.5) |

Further choices made here, listed so that they can be overruled:
- the refinement factor k = 8 (`with_block_refinement`), from §3.2's and §3.6's
  simulations, on the condition that P2 is accepted (without it, k should fall to 1–2
  beyond 16 ranks, §3.6);
- `movement::scatter_columns`, a new kernel (§7.3), rather than `scatter_add_columns`;
- every rank opens device 0; with a tuning cache, rank 0 tunes and broadcasts (§7.4,
  §7.6);
- the names: `PartitionWeight`, `with_partition_weight`, `with_block_refinement`,
  `HostData`, `evaluate_overlapped`, `BuildTimings::redistribute`,
  `StageTimings::{forward_charges, backward_output}`, `FmmError::Redistribution`.

## 14. Toward thousands of ranks

### 14.1 Why this section

The goal beyond Phase 5 is benchmark runs over **thousands of MPI ranks**, on clusters.
Phase 5 runs on one node (the M3 Max, locust; README decision 2), and its gates are at
2–8 ranks, so Phase 5's design is judged there. But a design that works at 72 ranks and
fails at 2,000 is not the design wanted. This section therefore asks of every structure:
does its per-rank cost stay bounded as P grows at fixed N/P (weak scaling), and does it
shrink like 1/P at fixed N (strong scaling)? It lists what does not, plans the changes,
larger octree changes included, and says what Phase 5 must already do so that those
changes extend it rather than undo it.

**Method.** Nothing here ran on more than 72 ranks. The counts at 1,024 and 8,192 ranks
come from the scratch program's single-process simulation with a virtual rank count: O1
built from the root on N = 10⁷ and 10⁸ points (the cube, the Plummer sphere, the
clusters; k = 8), the nearest-boundary cut by points, the `Global` boxes as the blocks'
ancestors, and per rank (512 ranks sampled at P > 512) its own `Global` ancestors and
the V sources they read. It reproduces §3.6's full simulation exactly at 8 and 72 ranks
(cube N = 10⁶: 4,096 blocks, 585 `Global` boxes, at most 284 needed multipoles). Times
and bytes at those P are *model*, from costs measured on locust's one-rank cube at
N = 10⁶, p = 8, f64, one thread: an evaluation 9.8 µs per point (9.79 s), an M2L pair
about 1.2 µs (the downward stage's 7.17 s over 6.04 × 10⁶ V pairs), and an M2M taken as
one M2L pair (the same dense n_c × n_c product).

### 14.2 What grows with P

Simulated at k = 8 (blocks and `Global` boxes; needed multipoles and their owners under
P2, max over the sampled ranks):

| | 72 ranks (N = 10⁷; the last two rows N = 10⁸) | 1,024 ranks, N = 10⁸ | 8,192 ranks, N = 10⁸ |
| --- | --- | --- | --- |
| points max/mean (cube, Plummer, clusters) | 1.02–1.07 | 1.03–1.10 | 1.03–1.11 |
| coarse blocks (per rank) | 2,640–4,264 (37–59) | 27,168–32,768 (27–32) | 212,010–262,144 (26–32) |
| `Global` boxes | 377–609 | 3,881–4,681 | 30,287–37,449 |
| own `Global` ancestors of a rank | 13–77 | 11–77 | 14–23 |
| top-tree multipoles a rank needs | 272–1,002 | 568–1,330 | 976–1,440 |
| … sent by how many other ranks (owned top tree) | 28–71 | 216–546 | 621–1,069 (mean 351–745) |

The partition scales: O1 keeps 26–32 blocks per rank and the balance within 1.11 at
8,192 ranks. What a rank needs of the top tree is nearly constant in P (about a
thousand multipoles). Everything the top tree replicates grows like P. Per rank and per
evaluation, cube at N = 10⁸, p = 8, f64 (*model*):

| Item (where) | grows like | 72 ranks | 8,192 ranks | verdict |
| --- | --- | --- | --- | --- |
| useful work per rank | N/P | 1.4 × 10⁶ points, ~14 s | 12,200 points, ~0.12 s | — |
| global M2M on every rank (`upward_global`) | B | 4,700 M2M, 6 ms | 300,000 M2M, **0.36 s, 3 × the useful work** | blocker |
| `Global` M2L on every rank without P2 (`downward`) | B | 56,000 pairs, 0.07 s | 6.0 × 10⁶ pairs, **7 s, about 80 fair shares of M2L** | blocker; P2 removes it |
| coarse gather received (`CoarseExchange`) | B | 2.7 MB | **170 MB per rank, 1.4 TB across the machine** | blocker |
| replicated held boxes (ghost-children layer, keys around `Global` boxes; `generate_all_keys`) | B | ~18,000 | **~1.2 × 10⁶, 1.5 GB of coefficient slots per rank** | blocker |
| `Plan::new`, lists over every held key | B | as held | ~1.2 × 10⁶ keys, seconds and hundreds of MB per rank | blocker |
| `Octree::new`: O1's rounds (all-reduce of 2 B counts) | B | 66 kB per round | 4 MB per round, about 7 rounds, once per build | acceptable |
| replicated block keys and partition bounds | B, P | 33 kB | 2 MB + 64 kB | acceptable |
| all-to-all of counts (`Redistribution`, rlst ghost builds, the sample sort) | P | 288 B | 32 kB per call | acceptable to ~10⁴ ranks |
| multipole-exchange neighbours (today: every `Global` V row) | P | 71 (all) | all P − 1 | with P2: the owners of ~1,000 boxes |

At 8,192 ranks, then, Phase 5's structure even with O1 and P2 spends about three times
the useful work on the replicated global M2M, moves 170 MB per rank per evaluation in the
gather, and holds 1.5 GB of replicated coefficients per rank. On a typical cluster node
with 1–4 GB per core it does not run. The limits come from replicating the top tree and
every coarse block on every rank, not from the partition, the exchanges or the
redistribution.

### 14.3 What already scales

- **The partition (O1–O4)**: blocks per rank stay at 26–32 and the balance within 1.11 at
  8,192 ranks; the leaves are the one-rank leaves at any P.
- **The leaves' independence of P** gives the strongest test available at scale (§14.5,
  S7): runs at different rank counts give the same bits.
- **Redistribution** (§4): linear in the points moved, O(P) count arrays (32 kB at 8,192
  ranks).
- **Rows by (level, key) (P1)** and **P2**: P2 is the first half of the owned top tree
  (S2).
- **Exchanges**: the source exchange and the multipole exchange below the top tree are
  neighbourhood exchanges; the scoped point-to-point mechanism (§8.4) is per neighbour.
- **Overlap** (§8): at a network's bandwidth the exchanges become real transfers, which
  the order-preserving overlap hides (§8.2).

### 14.4 The changes, as work packages

Each package keeps the results of every rank count bit for bit the one-rank results
(requirement 2 as proposed) and the determinism of requirement 3.

**S1. A locally essential top tree** (`nd-octree`, `nd-fmm-plan`; the larger octree
change). Today the ghost-children layer replicates every coarse block on every rank, and
`generate_all_keys` gathers to every rank the `Global` keys and every key adjacent to one,
with children. Instead each rank holds:
- its own boxes and their one-cell halo, as today;
- the ancestors of its own blocks (its `Global` boxes, now "shared ancestors"), and the
  boxes their V, W and X rows name, as ghosts of their owners;
- the replicated partition map only as *keys*: the block list with owners (2 MB at 8,192
  ranks), from which the owner and the kind of any top-tree box are computed locally.

`Plan::new` then builds lists over about a thousand top-tree boxes instead of a million;
the plan's check that every coarse block is held (redesign §3.5, decision 6) is replaced
by the check that every listed key is held. Collectives: the all-gathers of
`generate_all_keys` become point-to-point advertisements to the owners the map names
(one sparse exchange). Effect (*model*, 8,192 ranks): held top-tree boxes from about
1.2 × 10⁶ to about 10³ per rank.

**S2. An owned top tree** (`nd-fmm-plan`). Every `Global` box gets an owner (the rank of
its first descendant block: descendants of a box are a contiguous range of ranks). Upward:
the owners of the children send their exact multipoles to the parent's owner, which adds
them **by octant**, so every `Global` multipole has the one-rank bits; one round per
top-tree level. Downward: P2 (each rank its own ancestors' locals). The needed `Global`
multipoles then reach the ranks whose rows read them through the multipole exchange, as
ghosts do. A plain all-reduce of partial multipoles would be simpler and is rejected: it
changes the order of the additions, so the bits would depend on the rank count and the
MPI library.

§14.2's owner counts show the catch: a rank needs its ~1,000 top-tree multipoles from up
to ~1,000 other ranks at 8,192 ranks, because the boxes near the root are needed by
nearly every rank. So S2 is a hybrid: the top levels down to a fixed size L_r (a few
thousand boxes, chosen once, independent of P) are formed by the owned reduction and then
replicated by one all-gather of fixed size (a few MB). Below L_r the needs are
neighbourhood-local, and the multipole exchange carries them. Effect (*model*, 8,192
ranks): the global M2M from 300,000 per rank to the boxes it owns (about G/P ≈ 5) plus
the replicated top (fixed); the gather from 170 MB to a few MB; bit for bit unchanged. The
hook (§6) gains events for the reduction's sends and receives; overlap posts the
reduction's levels as their children complete.

**S3. A work-weighted cut** (`nd-octree`, `nd-fmm-exec`). §3.5 and §3.6: at 16–72 ranks
points leave M2L work 1.05–1.51 of the mean at N = 10⁶; it grows with P at fixed N.
Because the leaves do not depend on the cut (O1), a second cut is possible after the
local refinement: each rank refines its blocks (`create_local_tree`, no communication),
weighs each block by a cost model (points for P2P, P2M and L2P; boxes for M2L and L2L,
a box's V row being about 189 pairs inside a uniform region), the ranks agree the cut
again by the nearest-boundary rule, and blocks move once more (an all-to-all-v of keys).
The cost constants come from per-kind timings (`StageTimings::kinds`, Phase 4S T5),
per machine, precision and p. The simulation's 50/50 mix of points and M2L work already
reached 1.10–1.20 on both at N = 10⁶ on 72 ranks.

**S4. Construction at scale** (`nd-octree`; only if measured at ≥ 1,024 ranks to need
it). The sample sort (`parsort`, 4 samples per rank) runs twice per `Octree::new` and
balances its intermediate ranges only roughly; its all-to-all is O(P). An alternative
the design keeps in reserve: route the input keys straight to the O1 blocks' owners
(the cut needs only global counts per block, which local counting and one all-reduce per
round give), then deduplicate per block. Distinct counts add up over disjoint blocks, so
they are exact once a block's keys are on one rank, and the T0 condition (O1) can be
checked after the move. That removes both sample sorts from the construction.

**S5. Communication primitives at scale.** The O(P) count all-to-alls of `Redistribution`
and of the rlst ghost-communicator builds are acceptable up to about 10⁴ ranks (32 kB of
counts per rank at 8,192 ranks). Beyond that, or if measured costly, a sparse dynamic
exchange (non-blocking consensus: synchronous sends, then a non-blocking barrier) replaces
them, with safe rsmpi calls. After S1 and S2 every per-evaluation pattern is a
neighbourhood exchange or a fixed-size all-gather.

**S6. Ranks × threads, devices.** Everything replicated scales with the rank count, not
the core count. One rank per socket or NUMA domain with threads (the host path is
threaded and bit-identical for every thread count, C3.5) divides it by the threads per
rank, and is the configuration the benchmarks should favour on clusters. On GPU nodes,
one rank per GPU; `nd_fmm_kernels::Device::open` then needs an index (multi-GPU nodes;
§7.4).

**S7. Validation at scale.** No one-rank reference exists for N = 10⁹. With O1, P1, P2
and S2's order-preserving reduction, results do not depend on the rank count. So:
- **rank-count invariance**: a run on P ranks equals a run on P/8 ranks bit for bit, if
  rank r of the small run holds the concatenation of the inputs of ranks 8r … 8r + 7 of
  the large one (the same union order). It is a regression test at any scale;
- **distributed direct-sum sampling**: each rank sums its own sources' contributions at
  1,000 sampled targets, and one all-reduce adds them (exact enough in f64 with a
  compensated sum), so the error against the direct sum costs O(N/P) per rank;
- **distributed point generation**: point i drawn from a counter-based generator keyed
  by (seed, i), so each rank draws its own index range; no rank ever holds the global set
  (the Phase 5 tests do, which is fine up to N ≈ 10⁷).

### 14.5 Staging

**In Phase 5** (one node, as designed), constraints so that S1–S7 extend it:
- no new per-rank structure that grows with P beyond the partition map (keys and
  bounds): the coarse gather and the replicated top tree are the existing ones, kept
  only until S2;
- P2 in T5 (§3.6), written so that S2's owners and reduction can replace the global M2M
  under it;
- the hook's events (§6) named after movements, not after the all-gather, so that S2
  adds events instead of changing them;
- T10's `scaling` harness generates points per rank by index (S7) and has the
  rank-count-invariance check;
- T10 adds one oversubscribed run on locust (for example 256 and 512 ranks, `mpirun
  --map-by :OVERSUBSCRIBE`, N = 10⁶–10⁷, correctness and memory only, never timed). It
  measures the replicated held boxes, the coarse gather's bytes and `Plan::new`'s memory
  at P beyond 72 for real, and checks §14.2's model.

**After Phase 5, a scale-out phase**, ahead of the current Phase 6 list (laplace-fmm-plan
§7), because none of C6.x matters if the code cannot run on a cluster:

| Order | Package | Depends on | Acceptance (on a cluster) |
| --- | --- | --- | --- |
| 1 | S1 (locally essential top tree) | Phase 5 | held boxes per rank independent of P at fixed N/P, measured at 1,024 ranks; bit for bit Phase 5's results at 2–72 ranks |
| 2 | S2 (owned top tree, hybrid) | S1 | no per-rank cost that grows with P in an evaluation; bit for bit; the gather removed |
| 3 | S7 (validation at scale) | S1, S2 | rank-count invariance from 8 to 1,024 ranks and beyond; distributed direct-sum errors as one rank's |
| 4 | S3 (work-weighted cut) | S1 | compute-stage max/mean ≤ 1.10 on the workloads at 1,024 ranks |
| 5 | S4, S5 | measurements at ≥ 1,024 ranks | as measured |
| — | S6 | — | a guideline and the multi-GPU device index, as needed |

It wants **a cluster**: README decision 2 ("the M3 Max and locust, one node each") is
revisited for that phase. Development and correctness at 256–1,024 ranks (tens of nodes),
benchmarks at several thousand. Until then locust's oversubscribed runs cover
correctness and memory beyond 72 ranks.

**A cluster may not be available.** The scale-out phase is then planned and accepted on
*estimated* scaling behaviour, and says so in every result:
- **What can still be measured, on one node.** Correctness and bit identity at any rank
  count (oversubscribed locust runs up to 512–1,024 ranks, small N); the per-rank data
  structures, memory, message counts and bytes per rank at those P, which do not depend on
  where the ranks run; and the per-rank compute at the per-rank sizes of a large run
  (N/P = 10⁴–10⁶, on one rank or a few).
- **What is estimated.** Times at P beyond one node: a model that adds the measured
  per-rank compute, the counted messages and bytes at an assumed network latency and
  bandwidth (for example 1–2 µs and 10–100 Gbit/s, stated), and the measured load
  imbalance, in the way §8.2 and §14.2 do. Every such figure is marked *model*, with
  its inputs.
- **Acceptance without a cluster.** §14.5's acceptance column then reads: the per-rank
  quantities (held boxes, top-tree traffic, replicated work) measured bounded in P at
  fixed N/P on oversubscribed runs, and rank-count invariance up to the largest P run; the
  timing criteria are replaced by the model, and are checked on a cluster when one
  becomes available, with the scaling command of §11.
- **The risk this leaves.** Network effects that a model misses (contention, the
  latency of many small messages, collectives' algorithms at scale) stay unmeasured;
  the report lists them as open, and S5's sparse exchanges stay conditional on a real
  measurement.

### 14.6 The effect, per rank and per evaluation at 8,192 ranks

Cube, N = 10⁸ (12,200 points per rank), p = 8, f64; *model* from §14.2:

| | Phase 5 without P2 | Phase 5 (O1 + P2) | after S1 + S2 |
| --- | --- | --- | --- |
| replicated M2L | 6.0 × 10⁶ pairs (≈ 80 fair shares) | ~1,100 pairs (≈ 0.015) | ~1,100 pairs |
| replicated M2M | 300,000 | 300,000 (≈ 3 × the useful work) | ≈ 5 owned + the fixed top |
| coarse gather / top-tree traffic | 170 MB | 170 MB | ≈ 0.6 MB of needed multipoles + a fixed top of a few MB |
| replicated coefficient memory | ≈ 1.5 GB | ≈ 1.5 GB | ≈ MBs (≈ 10³ top-tree boxes) |
| bit for bit across rank counts | yes (O1, P1) | yes | yes |

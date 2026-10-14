# Scale-out: a locally essential, owned top tree (design for Phase 5S)

Phase 5S, task T1 (docs/phase5s/T1-scale-out-design.md). Drafted on 2026-10-11 for
sign-off by hand before T3. **Signed off on 2026-10-11: every recommendation of §12
accepted** (questions 1–11; docs/phase5s/README.md, decisions 0–6). It answers "Requirements on the scale-out" of
docs/phase5s/README.md and settles the decisions that change code in T2–T9: what a rank
holds of the top of the tree (S1), who forms each top-tree multipole and how it reaches
the ranks that read it (S2), how results are checked where no one-rank run fits (S7),
how the cut balances work (S3), and when the construction and the O(P) collectives are
replaced (S4, S5). It is the Phase 5S counterpart of
[distributed-fmm.md](distributed-fmm.md), whose §14 planned these packages from a
single-process simulation; this document replaces that simulation by measurements of
the Phase 5 code on Kathleen (1–4 nodes, 40–160 ranks timed, 320–640 oversubscribed;
1,280 did not start, §10.1) and on locust (8–72 ranks timed, 128–512 oversubscribed).

Companion documents, cited by section:
- **distributed**: [distributed-fmm.md](distributed-fmm.md) (§3.2 O1–O4, §3.6 P2, §6
  the hook, §8 overlap, §9 collectives and memory, §14 the scale-out plan, §15 Phase 5's
  outcome);
- **redesign**: [fmm-plan-redesign.md](fmm-plan-redesign.md) (§3.3–§3.5 the box index
  and decision 6, §7 the evaluator);
- **device-path**: [device-path.md](device-path.md) (§14, the device on several ranks);
- **README**: [docs/phase5s/README.md](../phase5s/README.md); **T0**:
  `fmm-validate/results/phase5n-kathleen.md`, "Phase 5S, T0: four nodes" (S0.1–S0.12).

Every number is *measured*, with machine, nodes, ranks × threads, job id and command
(Appendix A), or marked *model*. "Oversubscribed" runs are counted, never timed.

## Summary

Measured with the Phase 5 code, on Kathleen at 40–160 ranks (1–4 nodes) with counts at
320–640, and on locust at 8–72 ranks with counts at 128–512 (§1.3, Appendix A):

1. **The construction has a defect, and it is T0's two open findings.** rlst's sample
   sort takes four local quantiles per rank. On a random share every rank's quantiles
   coincide, and ranks P/4, P/2, 3P/4 and P − 1 receive a quarter of all keys each. At
   160 ranks the first sort is 95% of `Octree::new` (2.5 s at N = 10⁷, 8.5 s at 10⁸), and
   its four heavy ranks are T0's 2 GB ranks (§1.4). rlst PR #40 (histogram sort with
   sampling, opened at the user's request) makes `Octree::new` 11–36 times faster at 160
   ranks and removes the peak. T3a (scheduled at the user's request) removes the octree's
   two sorts altogether (§7.1).
2. **S1: a rank holds H = N(S) ∪ C(N(S))**, S its own keys and shared ancestors, N the
   held same-level neighbours, C their children. Derived from the list rule (§3.1), it
   contains every key S's rows name on every run. It grows 1.5–2.5× from 8 to 512 ranks at
   fixed N/P, where Phase 5's held set grows 6.6–23× (39,865 keys on the cube at 512, its
   whole top tree). It is assembled from a replicated partition map (blocks, cut, leaf
   bits: about 3 MB at 10⁴ ranks) and one point-to-point push, with no all-gather of keys
   (§3.2, §3.3). `KeyType::Global` and `with_ghost_children` keep their names and narrow
   their meaning. T4 keeps the top-tree keys held until T5 removes them (§3.7).
3. **S2: owners and a reduction, bit for bit.** The owner of a top-tree key is the rank of
   its first descendant block. Children's owners send exact multipoles to the parent's
   owner, which forms the M2M by octant, 2–7 rounds of at most 7 messages each (§4.1,
   §4.2). The global M2M per rank falls from 16,608–37,448 to at most 616, and the coarse
   gather (9.4–21.2 MB per rank and evaluation at 512–640 ranks) goes.
4. **The replicated top is chosen by readers, not by level.** A fixed level L_r leaves a
   box below it that 0.65 P ranks read on the Plummer sphere. Replicating the boxes that
   at least P/16 ranks read (at most 4,096, ≤ 2.65 MB at f64 p = 8) holds every owner's
   fan-out to ⌈P/16⌉ − 1. It adds no neighbour to the multipole exchange at 320–512 ranks,
   and at 640 one (Plummer, clusters) or 13 (cube) to the busiest rank's 47–125 (§4.3).
5. **S3: the cut by modelled cost.** Per-kind constants times the plan's counts reproduce
   each rank's compute within 1.0–1.4% and the measured imbalance (1.51 against 1.52 on
   the Plummer sphere at 160 Kathleen ranks). The cut by points degrades to 1.19–2.69
   times the mean modelled cost at 512 ranks; the cut by the model holds 1.02–1.11,
   limited by the heaviest block. The weights must come from the plan, not from the local
   refinement (a proxy reaches only 1.09–1.22), and the constants are per machine
   (M2L/P2P differs 2.3× between locust and Kathleen) (§6). The clusters' measured
   compute imbalance at 160 Kathleen ranks is 1.24–1.57 (§1.5).
6. **S7** checks P against P/k bit for bit inside one job; the M2L strategy must be named,
   since `Auto` resolves by ranks per node (§5). **S5's evaluation part fires on the
   model** (the redistribution's two all-to-all-v's, 1.5–8% of an evaluation at 10⁴
   ranks); its build part does not (§7.2).
7. **The time model** (§10.4, Kathleen's measured α_e = 1.5 µs, 0.27 GB/s per rank when a
   node's ranks all exchange, 1.5 GB/s all-gathers) puts Phase 5's replicated top at
   0.83–1.05 s per evaluation at 8,192 ranks and 2.5 × 10⁴ points per rank, and Phase 5S's
   at about 2 ms: 1.64 s against 0.55 s (cube), 2.71 s against 1.21 s (Plummer), *model*.

§12 lists eleven sign-off questions with recommendations.

## 1. Starting point

### 1.1 What the code does on several ranks

**`Octree::new`** (octree/src/octree.rs; the private module
octree/src/octree/implementation.rs). In order: `linearize` sorts every input key with
rlst's sample sort `parallel_sort::parsort` and removes ancestors across rank
boundaries (`communicate_back`); `compute_coarse_tree` builds the coarse tree by weight
from the root (O1: one all-reduce of two counts per block per round, then a serial 2:1
balance), **replicated on every rank**, and `partition_blocks` cuts it at the nearest
block boundaries (O3); `redistribute_with_respect_to_coarse_tree` moves the keys to the
owners (an all-to-all-v); `create_local_tree` refines each block locally; `balance`
closes the 2:1 condition locally and linearizes the union with a **second `parsort`**;
the keys move again; **`generate_all_keys`** classifies every key the rank will hold.
Only the rank's own range of blocks is kept (`coarse_tree_leafs`), with the per-rank
lower bounds (`coarse_tree_bounds`); the replicated coarse tree itself is dropped.

`generate_all_keys` is where the top of the tree is replicated:
- the ancestors of every rank's blocks are gathered to every rank (`gather_to_all`) and
  inserted as `KeyType::Global`;
- every key with a same-level neighbour that is `Global` is advertised to **all** ranks
  (`send_to_all`, a second `gather_to_all`), with its eight children under the
  ghost-children layer (`advertise`);
- with the ghost-children layer (`OctreeOptions::with_ghost_children`, which `Fmm`
  sets) every rank also advertises **every own block to all ranks** (`advertised_blocks`),
  so that every child of every `Global` key is held everywhere;
- every key also goes point-to-point to the owners of its other same-level neighbour
  cells (one all-to-all-v), as the one-cell halo, with its children under the layer.

**`Plan::new`** (fmm-plan/src/plan.rs) numbers every held key (`BoxIndex`), applies the
list rule (`interaction_manager::key_lists`) to every non-ghost key, `Global` keys
included, and checks that every list entry, every child of a non-ghost interior key and
every own block is held (redesign §3.5, decision 6: "every coarse block held"). P2
(distributed §3.6) empties the V, X and L2L rows of a `Global` box that is not an
ancestor of an own block, but its lists are still resolved, so its neighbourhood must
be held.

**`CoarseExchange`** (fmm-plan/src/exchange.rs) gathers every rank's block multipoles to
every rank in every evaluation (`all_gather_varcount_into`, built with an all-gather of
counts and an all-gather-v of every block key), and **`upward_global`**
(fmm-plan/src/evaluator.rs) runs the M2M of every `Global` box on every rank, deepest
level first. The downward pass forms only the locals of the rank's own ancestors (P2).

### 1.2 Every collective that grows with P

Per rank; B the coarse blocks (about 26–36 per rank, measured, §1.3), G the `Global`
boxes (about B/7), P the ranks.

| Call | Collective | What grows with P |
| --- | --- | --- |
| `Octree::new`, each of two `parsort`s | all-reduce; two `gather_to_all` (each two agreements, an all-gather of P counts, an all-gather-v of 4P samples); `global_min` (all-gather of P flags, broadcast, all-reduce); `all_to_allv` (four agreements, an all-to-all of P counts, all-to-all-v) | O(P) counts and samples; **the splitters (§1.4)** |
| `Octree::new`, `communicate_back` (twice) | all-gather of P flags; `gather_to_all` of one key | O(P) |
| `Octree::new`, O1 rounds | one all-reduce of 2B u64 per round (5–10 rounds measured) | O(B) = O(P) data |
| `Octree::new`, two key moves | all-to-all of P counts, all-to-all-v | O(P) counts |
| `Octree::new`, `generate_all_keys` | `gather_to_all` of every rank's `Global` ancestors; `gather_to_all` of the keys next to `Global` keys (with children) and of every block; all-to-all of P counts, all-to-all-v | **O(B) keys gathered to every rank** |
| `Plan::new` | two all-reduces | lists over every held key: **O(B) keys** |
| `Redistribution::new` (twice) | all-to-all of P counts, all-reduce, communicator duplicate, all-to-all-v | O(P) counts |
| `Evaluator::new`: `SourceExchange::new`, `MultipoleExchange::new` (one per level) | rlst `GhostCommunicatorBuilder::build`: three agreement all-reduces, **an all-to-all of P counts**, two `MPI_Dist_graph_create_adjacent`, two neighbour all-to-all-v | O(P) counts per build, 1 + nlevels builds |
| `Evaluator::new`: `CoarseExchange::new` | all-reduce, communicator duplicate, all-gather of P counts, all-gather-v of B keys, all-reduce | **O(B) keys** |
| `Fmm::evaluate`: forward of the charges, backward of the output | one `MPI_Alltoallv` each | O(P) count arrays; Open MPI's algorithm may visit every peer |
| `Fmm::evaluate`: `upward_global` | all-gather-v of every block's multipole; the global M2M over every `Global` box | **O(B n_c) bytes and O(G) M2M per rank and evaluation** |
| `Fmm::evaluate`: source and multipole exchanges | neighbour all-to-all-v (one, and one per level) | the multipole exchange's neighbours: every owner of a block a `Global` V row names |

### 1.3 Measured: what a rank holds and needs (Phase 5 code)

A throwaway probe (Appendix A) builds the Phase 5 octree and plan from the `scaling`
workloads (every point by index, a random share, `max_level` 16, 64 points per leaf,
the keys `Fmm` passes) and reports per rank, max / mean over the ranks:
- the held keys split into local, `Global`, ghost and other ranks' blocks;
- **S**, the rank's own keys plus its *shared ancestors* (the `Global` keys above its own
  blocks), the keys whose rows it computes under P2;
- the keys the rows of S **name** (V, W, U and X entries, parents, children);
- the **locally essential set** H (§3.1): S, its held same-level neighbours, and the
  children of every interior key among them;
- the **top-tree multipoles** S's V and W rows read that a rank does not form itself
  (`Global` keys whose owner, the rank of the first descendant block, is another rank),
  per level, their owners, and how many ranks need each one;
- what S2's reduction would send and receive per level (§4).

Held keys per rank at a fixed 2.5 × 10⁴ points per rank (weak scaling), max over the
ranks unless stated: Phase 5's held set, the keys S's rows name, H (§3.1), H with every
top-tree key (T4's transitional set, §3.7), and the replicated top R by readers (§4.3):

| workload | ranks | machine | Phase 5 held, max | named by S, max | H, max / mean | H ∪ T (T4), max | R (K = 16) | blocks |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| cube | 8 | locust | 1,753 | 1,523 | 1,529 / 1,396 | 1,633 | 35 | 309 |
| cube | 32 | locust | 3,817 | 3,498 | 3,521 / 2,891 | 3,817 | 64 | 512 |
| cube | 40 | Kathleen | 3,881 | 3,569 | 3,585 / 2,972 | 3,881 | 64 | 512 |
| cube | 72 | locust | 7,537 | 3,728 | 3,745 / 2,977 | 7,537 | 576 | 4,096 |
| cube | 80 | Kathleen | 9,361 | 5,514 | 5,537 / 4,295 | 9,361 | 576 | 4,096 |
| cube | 160 | Kathleen | 10,977 | 7,002 | 7,025 / 5,624 | 10,977 | 576 | 4,096 |
| cube | 256 | locust | 9,009 | 4,976 | 5,017 / 3,836 | 9,009 | 576 | 4,096 |
| cube | 320 | Kathleen | 8,737 | 4,674 | 4,713 / 3,663 | 8,737 | 576 | 4,096 |
| cube | 512 | locust | 39,865 | 3,739 | 3,793 / 2,709 | 23,833 | 1,298 | 18,418 |
| cube | 640 | Kathleen | 42,417 | 6,230 | 6,273 / 4,958 | 42,417 | 2,304 | 32,768 |
| Plummer | 8 | locust | 6,561 | 5,851 | 5,857 / 4,801 | 5,857 | 32 | 288 |
| Plummer | 32 | locust | 10,161 | 6,211 | 6,217 / 5,200 | 6,985 | 220 | 1,604 |
| Plummer | 72 | locust | 14,809 | 7,059 | 7,153 / 5,911 | 8,977 | 377 | 2,703 |
| Plummer | 160 | Kathleen | 20,105 | 7,330 | 7,425 / 5,855 | 11,401 | 651 | 4,621 |
| Plummer | 256 | locust | 27,897 | 8,271 | 8,369 / 6,015 | 15,337 | 1,060 | 7,540 |
| Plummer | 320 | Kathleen | 32,561 | 8,760 | 8,889 / 6,147 | 17,921 | 1,326 | 9,402 |
| Plummer | 512 | locust | 43,377 | 9,002 | 9,129 / 6,339 | 24,057 | 2,036 | 14,589 |
| Plummer | 640 | Kathleen | 50,377 | 9,054 | 9,177 / 6,359 | 27,809 | 2,327 | 17,963 |
| clusters | 8 | locust | 7,721 | 5,123 | 5,129 / 4,057 | 5,937 | 100 | 1,324 |
| clusters | 32 | locust | 14,329 | 6,516 | 6,521 / 5,355 | 8,457 | 336 | 2,717 |
| clusters | 40 | Kathleen | 16,145 | 6,401 | 6,497 / 5,324 | 9,337 | 436 | 3,305 |
| clusters | 72 | locust | 18,665 | 7,421 | 7,425 / 5,405 | 10,689 | 524 | 4,236 |
| clusters | 80 | Kathleen | 19,121 | 6,886 | 6,905 / 5,270 | 10,841 | 528 | 4,425 |
| clusters | 160 | Kathleen | 31,265 | 7,476 | 7,569 / 5,593 | 14,489 | 871 | 7,162 |
| clusters | 256 | locust | 41,185 | 8,485 | 8,593 / 6,218 | 19,241 | 1,360 | 10,662 |
| clusters | 320 | Kathleen | 45,305 | 8,316 | 8,409 / 6,322 | 21,393 | 1,636 | 12,685 |
| clusters | 512 | locust | 54,753 | 8,725 | 8,729 / 6,105 | 26,657 | 2,138 | 17,277 |
| clusters | 640 | Kathleen | 59,097 | 7,806 | 7,825 / 5,896 | 28,465 | 2,263 | 19,440 |

(locust: up to 72 ranks one per core, 256 and 512 oversubscribed; Kathleen: 40 per node
on 1, 2 and 4 nodes, jobs 252223 and 252846; 320 and 640 oversubscribed at 80 and 160 per
node, jobs 252223 and 255188, the two jobs' 320-rank rows equal; 1,280 did not start,
§10.1. The cube at 640 Kathleen ranks and N = 10⁷: Phase 5 holds 39,545, H 3,441, with
32,768 blocks: every rank holds the whole top of the tree.)

What it shows:
- **Phase 5's held set grows with P** at fixed N/P: 6.6×–23× from 8 to 512 ranks, as the
  blocks grow (26–36 per rank) and the top tree with them; at 512 ranks the cube holds
  39,865 keys, every key of its replicated top, and at 640 Kathleen ranks 42,417 (Plummer
  50,377, clusters 59,097).
- **H barely grows**: 1.5–2.5× from 8 to 512 ranks (cube 1,529 → 3,793 with a peak of
  7,025 at 160 where the cube's tree changes level; Plummer 5,857 → 9,129; clusters
  5,129 → 8,729), a logarithmic growth through the shared ancestors' levels. At 640
  Kathleen ranks H is 9,177 (Plummer) and 7,825 (clusters), within the 512-rank range, and
  6,273 on the cube, which at 640 again sits where its tree changes level (32,768
  blocks). H is 9–31% of Phase 5's set at 512 ranks and 13–18% at 640, and every key S's
  rows name is in it (§3.1).
- **T4's transitional set H ∪ T** still grows with the blocks (23,833–26,657 at 512,
  27,809–42,417 at 640); T5
  removes T.
- **R** stays a few thousand boxes at 512 ranks (§4.3).

### 1.4 Found: the first sample sort sends a quarter of all keys to each of four ranks

rlst's `parsort` (rlst 0.9.0, src/distributed_tools/parallel_sort.rs, `get_bins`) takes
**4 samples per rank**, the first key of each quarter of the rank's sorted keys, and
picks bucket bound b as the first sample whose cumulative weight reaches b W/P. With a
random share every rank's keys span the whole domain, so its four samples sit at the
same four global quartiles as every other rank's: each quartile holds P nearly equal
samples, one per rank. The cumulative weight jumps by W/4 across each cluster, the P/4
bounds of each quarter fall on every fourth sample of one cluster, and ranks P/4, P/2,
3P/4 and P − 1 receive **the whole quarter** that follows their cluster. Measured in
`Octree::new`'s first sort (`linearize`), keys received over keys passed, max over the
ranks: 2.00 at 8 ranks, 7.96 at 32, 17.85 at 72, 31.5 at 128, 62.4 at 256, 123.5 at 512
(locust, N = 10⁷, every workload), that is **P/4: four ranks receive N/4 keys each, and
about half the ranks receive almost none** (rank 0 receives 6). Consequences:
- **the first sort does not scale**: its slowest rank takes 170–240 ms at N = 10⁷ from
  8 to 72 ranks (locust), 72–78% of `Octree::new` at 72 ranks; on Kathleen 0.40–0.42 s
  at 40 and 80 ranks and **2.49–2.51 s at 160 ranks on 4 nodes, 94–96% of
  `Octree::new`** (job 252220), which is T0's octree "jump" at 160 ranks (S0.5, S0.10):
  four ranks receive 39.4 times their share each, across the network;
- **the peak memory of four ranks scales with N**: the heavy rank's peak resident size
  during the sort is 241 MB against a mean of 59 MB at 72 ranks (N = 10⁷, locust).
  At N = 10⁸ on Kathleen the heavy ranks peak at 2.22 GB (40 ranks) and 1.94 GB (160)
  during the sort, against a mean of 0.65 and 0.20 GB (job 252220): **T0's unexplained
  2 GB on four ranks** (S0.5), on ranks P/4, P/2, 3P/4 and P − 1. The sort takes 4.75 s
  of `Octree::new`'s 6.1–7.3 s at 40 ranks and 8.53–8.59 s of 9.7–10.3 s at 160;
- the second sort (inside `balance`) sorts keys already distributed by block ranges, so
  its samples spread; its imbalance is 1.67 at 72 ranks (cube), and its time small.

With the owners' input (every rank a contiguous key range) the first sort would be
balanced too; with the random share that every test and the `scaling` harness use, and
that an application's input order is likely to resemble, it is not. This is S4's
trigger, measured (§7).

### 1.5 Measured: the clusters at 1, 2 and 4 nodes

T0 timed the cube and the Plummer sphere; the brief adds the clusters (`scaling`'s five
Gaussian clusters of width 0.02) at N = 10⁷ and 10⁸, f64, p = 3 (`Auto` → Dense) and
p = 8 (Rotation; at N = 10⁸ `Auto` → Rotation), both splits, the random share, one
warm-up and 10 evaluations (2 at N = 10⁸), the median of the max over the ranks
(Kathleen, Phase 5 code with rlst 0.9.0; job 255188, and 252846 for 4 nodes at 2 × 20).
Evaluation wall time, blocking path (the overlapped path within 1%):

| N, p | split | 1 node | 2 nodes | 4 nodes | 1 → 4 nodes | compute max / mean, 1 / 2 / 4 nodes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 10⁷, 3 | 40 × 1 | 2,170 ms | 1,180 ms | 671 ms | 3.23× | 1.26 / 1.37 / 1.57 |
| 10⁷, 3 | 2 × 20 | 2,314 ms | 1,131 ms | 580 ms | 3.99× | 1.03 / 1.03 / 1.09 |
| 10⁷, 8 | 40 × 1 | 13.42 s | 7.12 s | 3.91 s | 3.43× | 1.12 / 1.18 / 1.30 |
| 10⁷, 8 | 2 × 20 | 13.11 s | 6.34 s | 3.23 s | 4.06× | 1.04 / 1.02 / 1.05 |
| 10⁸, 3 | 40 × 1 | 18.83 s | 9.64 s | 5.04 s | 3.74× | 1.27 / 1.29 / 1.36 |
| 10⁸, 3 | 2 × 20 | 21.48 s | 10.66 s | 5.42 s | 3.97× | 1.04 / 1.04 / 1.08 |
| 10⁸, 8 | 40 × 1 | 125.4 s | 63.2 s | 33.4 s | 3.75× | 1.16 / 1.17 / 1.24 |
| 10⁸, 8 | 2 × 20 | 124.4 s | 58.7 s | 30.6 s | 4.06× | 1.07 / 1.03 / 1.08 |

The direct-sum errors at N = 10⁸ are the same at every node count and split (φ, L2:
4.0 × 10⁻³ at p = 3, 2.9 × 10⁻⁵ at p = 8). What it shows:
- **at 40 × 1 the clusters lose to imbalance as the Plummer sphere does** (T0's
  1.31–1.62): the compute's max over mean grows with the ranks to 1.24–1.57 at 160,
  and the 4-node speed-up stays at 3.2–3.75; at 2 × 20 the threads of a rank share its
  work and the speed-up is 4.0 (S3, §6);
- **the build at 40 × 1 jumps at 4 nodes**: 1.5, 1.1 and 3.1 s at N = 10⁷ (`Octree::new`
  0.64, 0.59 and 2.68 s), 18.5, 11.9 and 12.9 s at 10⁸ (`Octree::new` 7.4, 6.6 and
  10.2 s): the first sort (§1.4);
- **the build at 2 × 20 is slow**: 17.8, 8.3 and 4.0 s at N = 10⁷, 237, 106 and 50 s at
  10⁸, of which `Plan::new` and the redistribution are 180, 78 and 37 s, against 10, 4.7
  and 2.3 s at 40 × 1. That is about the factor of 20 fewer workers: these parts of the
  build appear to run on one thread per rank (inferred from the ratio, not profiled).
  It is outside S1–S7; T8 (the construction) and T9 should report it. Peak resident
  size at 2 × 20 and N = 10⁸ on one node: 27.5–30.9 GB per rank, against at most 2.2 GB
  at 40 × 1 (the sort's heavy rank).

## 2. Requirements

The ten requirements of docs/phase5s/README.md ("Requirements on the scale-out"), and how
this design meets each. Proposed changes are sign-off questions (§12, question 1).

| # | Requirement (short) | How the design meets it |
| --- | --- | --- |
| 1 | Same interface | `FmmBuilder::build` and `Fmm::evaluate` unchanged. New options are additive: the second cut (§6) on several ranks, bits unchanged; the replicated top's threshold (§4.3) fixed, not an option |
| 2 | Bit for bit at every rank count | S1 changes what is held, not a list entry or a row order (§3.6). S2 forms every top-tree multipole by the M2M of its exact children by octant (§4.2), the one-rank order. S3 changes only the partition (§6.4). S4 changes neither the leaves nor the partition (§7.1) |
| 3 | Rank-count invariance | Follows from 2: P ranks and P/k ranks both equal the one-rank `Fmm` over the same union in the same order. **The M2L strategy must be named, not `Auto`**: `Auto` resolves by the ranks per node (Phase 5N T4), which differ between the two runs (§5.1) |
| 4 | Bounded per-rank cost at fixed N/P | Held boxes: the locally essential set (§3) plus the replicated top (§4.3); the remaining O(P) items and their sizes at 10²–10⁴ ranks are §8.2's table. **Proposed change:** T4 holds the top-tree keys (every block and every ancestor of one, about 1.15 B) until T5 removes them, so C5S.2's bound is met by T4 on the locally essential part and by T5/T6 on the total (§3.7, §11) |
| 5 | No per-evaluation cost growing with P | The coarse gather and the global M2M go (§4). What remains per evaluation: neighbourhood exchanges, S2's reduction rounds, one all-gather of fixed size (§4.3), and the two `MPI_Alltoallv` of the redistribution, whose O(P) argument arrays S5 replaces by point-to-point on the build's routes (§7.2). **Measured caveat:** the owners' fan-out of top-tree multipoles below the replicated top is bounded by P/K, not by a constant (§4.3); fixed K holds it to the phase's targets |
| 6 | Collective discipline | Every new exchange is built and run on every rank, empty ranks with empty lists; the push exchange of S1, S2's reduction and the replicated top are scoped point-to-point or collectives entered by all; errors agreed with the existing build all-reduces (§8.1) |
| 7 | Kernel-agnostic and safe | S1, S2, S5 live in `nd-octree` and `nd-fmm-plan` and use the recording and shadow operators; S3's constants live in `nd-fmm-exec`, the octree receives weights. Every new exchange uses safe rsmpi calls (scoped requests, `immediate_*`), which rsmpi 0.8.2 has (§7.2) |
| 8 | Hook, device and overlap | Three new `HostData` events (§4.5); the device mirrors them with packed buffers; the sync count grows from at most 2 to at most 2 + r_s, r_s the reduction levels at which the rank sends (§4.5). Overlap order-preserving (§4.6) |
| 9 | Tested on what can run | §9: CI at 1, 2, 4 ranks; by hand at 8 (M3 Max, locust); Kathleen at 2 nodes routinely, 4 at most |
| 10 | Measured, never asserted | §10: every time beyond 160 ranks is a *model* with its inputs |

## 3. S1: the locally essential top tree

### 3.1 What a rank holds, derived from the lists

Under P2 a rank computes rows for exactly the set **S = its own keys (local leaves and
interior keys, at or below its blocks) ∪ its shared ancestors** (the `Global` keys above
its own blocks). Every other key it holds is there because a row of S, or the list rule
that builds those rows, reads it. From `interaction_manager::key_lists` and
`plan::first_pass`, for a box b ∈ S with parent p (p ∈ S as well: the parent of an own
key is own or a shared ancestor, the parent of a shared ancestor is one):
- **V(b)**: the children of the interior same-level neighbours of p that are not adjacent
  to b;
- **X(b)**: the leaf same-level neighbours of p not adjacent to b;
- **U(b)**, **W(b)** (b a leaf): b's leaf neighbours, and the coarser leaf covering a
  missing neighbour cell (a neighbour of p); the children of b's interior neighbours,
  adjacent (U, leaves by 2:1 balance) or not (W);
- **L2L, M2M**: p, and b's children (an own interior key's children are own; a shared
  ancestor's children are top-tree keys: shared ancestors, blocks, or ancestors of other
  ranks' blocks only);
- **the rule itself** reads the kind of every same-level neighbour of b and of p, and
  the children of the interior ones; `validate_tree` reads the same keys.

So the held set is

> **H = N(S) ∪ C(N(S))**, with N(S) = S and every key of the tree that is a same-level
> neighbour of a key of S, and C(·) the children of the interior keys among them.

Every entry of every row of S is in H, and nothing outside H is read. Measured on every
run of §1.3 (locust 8–512 ranks, Kathleen 40–640; the probe computes both from the Phase 5
plan): **every key the rows of S name lies in H** (0 exceptions), and H holds on average
6–104 keys more than the rows name, at most 5% of H (neighbours the rule inspects
without listing them).
The halo of Phase 5 (the one-cell same-level ghosts of the own keys) and the
ghost-children guarantee restricted to S are both part of H; what H drops is everything
Phase 5 holds for the `Global` keys that are not shared ancestors: their neighbourhoods
(the `send_to_all` keys with their children) and every other rank's block.

H is bounded in P at fixed N/P: its own part is N/P; its halo part is the surface of the
own region; its top part is S's shared ancestors (at most 4–21 per rank on the cube and
the Plummer sphere, 54–83 on the clusters, at 8–512 ranks, measured) with their
neighbours and children, at most 27 × 9 keys each. Measured (§1.3): H's maximum grows
1.5–2.5× from 8 to 512 ranks at 2.5 × 10⁴ points per rank, where Phase 5's held set grows
6.6–23×. **Proposed bound for C5S.2** (T4 on H, T5/T6 on the total): at fixed N/P, H's
maximum at most twice its 40-rank value, and H plus R at most twice that plus R_max, up
to 640 ranks on Kathleen (1,280 does not start there, §10.1) and 512 on locust.

### 3.2 The partition map

The coarse tree and its cut are already computed on every rank in `Octree::new`
(`compute_coarse_tree`, `partition_blocks`) and then dropped except for the own range.
S1 keeps them as the **partition map**:

| Field | Bytes per rank | At 10² / 10³ / 10⁴ ranks (*model*, 36 blocks per rank) |
| --- | --- | --- |
| every block, ascending (`MortonKey`) | 8 B | 29 kB / 288 kB / 2.9 MB |
| the cut: P + 1 offsets into the blocks | 8 (P + 1) | 0.8 kB / 8 kB / 80 kB |
| whether each block is a leaf (one bit) | B / 8 | 0.5 kB / 4.5 kB / 45 kB |

(36 blocks per rank is the cube's measured figure at 512 ranks, k = 8; the Plummer
sphere has 28–29, the clusters 33–34; distributed §14.2 simulated 26–32.) From it any
rank computes, without communication, for any key at or above the blocks (a *top-tree
key*: a block or an ancestor of one):
- **its owner**: for a block, its rank; for an ancestor, the rank of its first descendant
  block (S2's owner rule, §4.1); for a key below a block, the block's rank, as
  `owner_rank` does today;
- **its sharers**: the ranks with a block at or below it, a contiguous range of ranks
  (with O4, empty ranks inside the range have no block);
- **its kind** relative to this rank: an own block (`LocalLeaf`/`LocalInterior` by the
  leaf bit), a shared ancestor (`Global`), another rank's block
  (`GhostLeaf`/`GhostInterior(owner)`), or an ancestor of other ranks' blocks only
  (`GhostInterior(owner)`).

The leaf bit is the only part not known before the 2:1 balance: a block that
`create_local_tree` leaves whole can still be split by the balance of its neighbours.
It is gathered once, after the second key move: one all-gather-v of one byte per own
block, whose counts every rank computes from the cut (no count exchange). The
per-rank lower bounds (`coarse_tree_bounds`) stay as they are, computed from the same
map, so `owner_rank`, `local_leaf` and `lookup_leaves` do not change.

### 3.3 Assembling H without all-gathers

1. **The top part of H** (its top-tree keys) is computed locally from the map: the
   top-tree keys among the same-level neighbours of every key of S (in practice of the
   shared ancestors and the own blocks: an own block's neighbour can be an ancestor of
   another rank's finer blocks, a key no rank owns and so none pushes) and their
   top-tree children, each with its kind (§3.2). The children of an interior neighbour
   that is another rank's block are below the blocks; they come with the push.
2. **The part below other ranks' blocks** comes by one push: every rank sends each of
   its own keys k to every other rank that holds a same-level neighbour cell of k in its
   S, with k's children when k is interior (the ghost-children step). The targets come
   from the map: a cell inside a block goes to the block's owner (today's halo rule); a
   cell that is a top-tree key goes to its sharers (a contiguous range of ranks, empty
   ranks inside it skipped), which hold it as a shared ancestor. (By the 2:1 balance of
   the coarse tree a same-level neighbour of a shared ancestor is itself a top-tree key,
   never strictly below a block, so these two cases cover H; the fact-check of this
   draft found no key of H that no rank pushes.) This replaces `send_to_all`, which sent the same keys to every
   rank.
3. **No key list is gathered**: the `gather_to_all` of the `Global` ancestors (step 1
   computes them) and the advertisement of every block to every rank go.

Collectives of the new `generate_all_keys`, in order: the leaf-bit all-gather-v (B
bytes received, §3.2); one all-to-all of P counts and one all-to-all-v for the push
(the existing pair, until S5 replaces the count all-to-all, §7.2). Empty ranks send and
receive nothing and enter both. The closing debug check of the ghost-children guarantee
stays, restricted to S.

`KeyType::Global` **keeps its name and narrows its meaning** to "shared ancestor": an
ancestor of an own block, held by every rank that has a block below it, whose local each
of them forms (P2) and whose multipole its owner forms (S2). A top-tree key that is not
a shared ancestor is a ghost of its owner. **`with_ghost_children` keeps its name**: it
asks for H's closure (children of the interior same-level neighbours of S) and no longer
replicates blocks; the docs of both change, no variant is added. Renaming `Global` to
`Shared` would touch `nd-fmm-plan`, `nd-fmm-exec` and `nd-fmm-validate` for no
behavioural gain (§12, question 2).

### 3.4 `Plan::new` over H

The box index, the lists and the exchanges build from the octree's keys as today: the
index over every held key, the rule over every non-ghost key (now exactly S, so P2's
"pruned" `Global` rows disappear rather than being emptied), the source and multipole
exchanges over the ghosts the rows name. The checks change:
- "every list entry, every child of a non-ghost interior key is held" stays (now
  checked over S);
- "every own coarse block is a non-ghost key" stays (it holds by construction);
- "every coarse block of every rank is held" (redesign §3.5, decision 6), which `Plan`
  enforces through the children of every `Global` key (`MissingChild`) and
  `CoarseExchange::new` through its closing all-reduce (`UnheldKey`), **goes**: only
  shared ancestors are `Global` now, and the coarse gather goes in T5 (§3.7 for T4).

### 3.5 API (`nd-octree`, `nd-fmm-plan`)

```rust
/// The coarse blocks and their owners, replicated on every rank as keys (Phase 5S S1;
/// docs/design/scale-out.md §3.2). 8 bytes per block and per rank, and a bit per block.
pub struct PartitionMap { /* blocks, offsets, leaf bits, this rank */ }

impl PartitionMap {
    /// Every coarse block, ascending.
    pub fn blocks(&self) -> &[MortonKey];
    /// The blocks of `rank`, as a range of indices into `blocks` (empty for a rank
    /// without blocks, O4).
    pub fn rank_blocks(&self, rank: usize) -> Range<usize>;
    /// Whether block `b` is a leaf of the tree.
    pub fn is_leaf_block(&self, b: usize) -> bool;
    /// The owner of `key`: the rank of the block containing it, or for a key above the
    /// blocks the rank of its first descendant block (S2's owner). O(log blocks).
    pub fn owner(&self, key: MortonKey) -> usize;
    /// The ranks with a block at or below `key`, a contiguous range; one rank for a key
    /// at or below a block.
    pub fn sharers(&self, key: MortonKey) -> Range<usize>;
    /// The kind of a top-tree key (a block or an ancestor of one) relative to this rank,
    /// or `None` for a key below the blocks or not in the tree. No communication.
    pub fn top_kind(&self, key: MortonKey) -> Option<KeyType>;
}

impl<C: CommunicatorCollectives> Octree<'_, C> {
    /// The partition map (see [`PartitionMap`]).
    pub fn partition_map(&self) -> &PartitionMap;
}
```

`Plan` gains nothing public for S1; `PlanError::CoarseBlock` stays (own blocks).
`nd_fmm_validate::scaling`'s held-box split (the private `held_boxes`, today local,
ghost and `Global`) reports own, halo, shared ancestors, other top keys and the
replicated top (§4.3).

### 3.6 Why every bit stays the same

The rows of a key of S depend only on the tree and the rule, never on what else is
held; H contains every key they name (§3.1, measured), with the same kind (a ghost keeps
its owner; a shared ancestor stays `Global`). P1 orders near and X rows by the entry
leaf's (level, key), not by its index, so a smaller index changes no order. P2's kept
rows are S's rows. So every list entry, every row order and every value is Phase 5's.
T4 checks it with the list oracle of `mpi_regressions` and `mpi_exec`'s one-rank
comparisons.

### 3.7 Between T4 and T5: the top-tree keys stay held

Until S2 removes the coarse gather and the global M2M (T5), every rank still needs a
slot for every block and the M2M rows of every `Global` key. The map gives both without
communication, so T4's plan holds **H ∪ T**, T the top-tree keys (every block and every
ancestor of one): ancestors that are not shared ancestors are typed `Global` for the
global pass but get no list resolution (P2's pruned rows, now not even resolved), other
ranks' blocks stay ghosts of their owners. T is about 1.15 B keys, O(P); T5 drops
T ∖ (H ∪ R), R the replicated top (§4.3). Measured (§1.3), max over the ranks at 512
ranks and 2.5 × 10⁴ points per rank: Phase 5 holds 39,865 keys (cube), H 3,793, H ∪ T
23,833 (Plummer 43,377, 9,129 and 24,057; clusters 54,753, 8,729 and 26,657). **So T4
alone removes the neighbourhood replication (40–51% of the held keys at 512 ranks) and T5
the rest**; on the uniform cube below 512 ranks, where the top tree is every block's
ancestors only, T4 removes nothing (H ∪ T equals Phase 5's set) and T5 all of it; C5S.2's "held boxes independent of P" is met on H
in T4 and on the total in T5/T6 (requirement 4's proposed change, §12, question 1).

The alternative order, S2 before S1 (T5 and T6 on the Phase 5 key set, then T3 and T4),
makes every step meet its gate directly, but S2 on the Phase 5 key set would have to
treat every non-shared `Global` key as a ghost of its owner for the top exchange while
the octree still classifies it as `Global`, which is the T3 change. It is not
recommended.

## 4. S2: the owned top tree

### 4.1 Owners

**The owner of a top-tree key is the rank of its first descendant block**
(`PartitionMap::owner`). The blocks below a key are a contiguous range of blocks, so its
sharers are a contiguous range of ranks and the owner is the first of them. The owner of
a key's first child is the owner of the key, so a rank owns, for each own block b, every
ancestor whose first descendant block is b. Measured owned `Global` keys per rank, max: 5–19
(cube, Plummer sphere), 71–77 (clusters, whose empty regions give a few ranks long left
edges) at 8–512 ranks (locust); Phase 5 runs the M2M of every `Global` key on every rank
(B/7 keys: 585 at 72 ranks, 2,076–2,641 at 512).

### 4.2 The upward reduction

For each level l from the deepest level with a top-tree key under a `Global` parent (the
deepest block level) up to level 3:
1. the owners of the level-l top-tree keys whose parent another rank owns send their
   exact multipoles to the parent's owner (own blocks are final after `upward_local`;
   owned `Global` keys after step 2 of level l + 1);
2. the owner of each level-(l − 1) `Global` key, holding all eight children (its own and
   the received ones), forms its multipole with the ordinary global-pass `m2m` call, the
   children **by octant**.

Every top-tree multipole is then the M2M of the same eight values in the same order as on
one rank (where the local pass forms it, children by octant: redesign §7.5), so it has
the one-rank bits. A multipole received is a copy, never a partial sum. **An all-reduce
of partial multipoles is rejected** (distributed §14.4): it sums the children in an
order the MPI library and the rank count choose, so the bits would change with both.

The reduction stops when level 2 is formed: no V or W row reads a multipole of level 0
or 1 (the root has no neighbours, so V and X rows of levels 0 and 1 are empty), and the
locals there are zero (L2L into level 2 reads a zero local). The one-rank pass still
forms them; their values are never read, so no output changes. Measured per rank at
8–512 ranks (locust): 2–7 rounds; at most 7 child messages received per level and 33 over
all levels, at most 7 messages sent over all levels, at most 47 child multipoles received
(Kathleen at 640 ranks, cube: 23 messages over all levels). Built once per `Evaluator::new` as one exchange per level (rlst's
`GhostCommunicator`: on the parent's owner the children owned elsewhere are ghosts of
their owners), run every evaluation with that level's neighbour all-to-all-v or scoped
point-to-point (overlap).

### 4.3 The replicated top, by needers

§14.4 of distributed proposed to replicate the top-tree levels down to a fixed level
L_r and to fetch the multipoles below it from their owners. Measured, that works for the
uniform cube (needs only at levels 2–4) but **not for adaptive trees**: on the Plummer
sphere at 512 ranks the top-tree multipoles a rank reads lie at levels 2–7, below L_r = 4
one box is still read by **335 ranks (0.65 P)**, and the clusters reach 186 (0.36 P)
(locust, N = 10⁶ and 10⁷). In a dense core a deep box's V range holds a fixed fraction of
all points, so a fixed fraction of the ranks read it, whatever the level. An owner would
send it to O(P) ranks every evaluation.

What bounds the replicated part is the number of ranks that read a box, not its level:
the sum over boxes of their readers is the sum over ranks of their needs (at most about
900 top-tree multipoles per rank, measured), so **at most K × 900 boxes are read by P/K
ranks or more, whatever P**. The design therefore replicates:

> **R = the top-tree boxes that at least max(2, ⌈P/K⌉) ranks read, at most R_max of
> them (the most read first), with K = 16 and R_max = 4,096** (§12, question 3).

- **Formed** by the reduction on their owners; **replicated** by one all-gather-v of
  their multipoles per evaluation, |R| n_c s bytes, at most 2.65 MB at p = 8 in f64
  (1.33 MB in f32), whose per-rank counts every rank computes from R and the map (no
  count exchange per evaluation).
- **Chosen at build**: the build of the top-tree exchange (the multipole exchange's
  rlst build over the top-tree ghosts) delivers to every owner the readers of each of its boxes; one all-reduce of a histogram of reader counts
  (log₂ P bins) fixes the threshold under the cap, and one all-gather-v of R's keys (at
  most R_max × 8 B, with an all-gather of P counts) gives R to every rank.
- **Below R**, the multipoles a rank's rows read come from their owners through the
  multipole exchange, posted per level once the reduction has formed that level. Most
  are ghosts (`GhostInterior(owner)` in H, §3.2). **Shared ancestors the rank holds but
  does not own** are `Global`, not ghosts, yet its rows read them (two non-adjacent
  level-2 shared ancestors are in each other's V rows), so the exchange also receives
  every non-owned shared ancestor a row of S reads, its owner from the map. Today's
  `multipole_ghosts` keeps only keys with a ghost rank; T5 extends it.

Measured at 512 ranks (locust, oversubscribed, the cube, the Plummer sphere and the
clusters at N = 10⁶, 10⁷ and 2.5 × 10⁴ per rank), against a fixed L_r = 4:

| | R by needers, K = 16 (threshold 32) | K = 64 (threshold 8) | L_r = 4 (levels 2–4) |
| --- | --- | --- | --- |
| replicated boxes | 1,298–2,144 | 2,017–2,632 | 2,104–4,672 |
| replicated bytes, p = 8, f64 | 0.84–1.39 MB | 1.31–1.71 MB | 1.36–3.03 MB |
| most ranks reading one box below it | 31 | 7 | 335 |
| neighbours of the multipole exchanges after S2 (max) | 56–117, **the Phase 5 multipole exchange's own** | the same | 55–323 |

So with K = 16 S2 adds **no neighbour** to the multipole exchange at 512 ranks: every
top-tree multipole a rank reads below R comes from a rank it already exchanges with.
On Kathleen (job 255188, 2.5 × 10⁴ points per rank, the three workloads):

| | 320 ranks, K = 16 (threshold 20) | 640 ranks, K = 16 (threshold 40) | 640 ranks, L_r = 4 |
| --- | --- | --- | --- |
| replicated boxes | 576–1,636 | 2,263–2,327 | 4,240 (Plummer) |
| replicated bytes, p = 8, f64 | 0.37–1.06 MB | 1.47–1.51 MB | 2.75 MB |
| most ranks reading one box below it | 0–19 | 34–39 | 232–414 (cube: 0) |
| neighbours after S2 (max) against Phase 5's (max) | 57 / 57, 104 / 104, 70 / 70 | 60 / 47, 126 / 125, 91 / 90 | 47 / 47, 355 / 125, 136 / 90 |

(cube, Plummer, clusters in that order.) At 640 the fan-out stays below the threshold, and
the busiest rank gains one neighbour on the Plummer sphere and the clusters and 13 on the
cube, whose 32,768 blocks put most of its top tree just below R; at N = 10⁷ on 640 ranks
the cube gains 12 (job 252846). 1,280 ranks did not start on Kathleen (§10.1), so the
larger counts are locust's 512. The cap R_max binds where |R| would exceed it: |R| grew from 556 to
2,023 boxes from 128 to 512 ranks on the Plummer sphere (thresholds 8 to 32), so at
several thousand ranks the cap, not K, will fix R, and the residual fan-out (the readers
of the R_max-th box) is reported, not bounded (§8.2).

### 4.4 What replaces `CoarseExchange` and `upward_global`

| Phase 5 | Phase 5S (T5) |
| --- | --- |
| `CoarseExchange`: all-gather-v of every block's multipole, B n_c s per rank and evaluation | **removed** |
| `upward_global`: the M2M of every `Global` key on every rank | `upward_global`: the reduction (§4.2), the M2M of the owned `Global` keys only, then the all-gather of R (§4.3) |
| the multipole exchange: the ghosts of V and W rows, local senders | the same, plus the top-tree ghosts below R, whose senders are their owners |
| P2: the locals of the shared ancestors, on every sharer | unchanged |

Per rank and evaluation, measured (locust, 512 ranks): the global M2M from 16,608–21,128
(8 × every `Global` key) to 40–616 (8 × owned), the coarse gather's 9.4–12.0 MB received
to R's 0.84–1.39 MB; the reduction's 2–7 rounds add at most 33 small messages. At 640
ranks on Kathleen (cube, N = 10⁷): the coarse gather is 21.2 MB per rank and evaluation
and the global M2M 37,448, against R's 1.49 MB (2,304 boxes) and 88 owned M2M.

The pass order on the blocking path becomes: sources; `upward_local`; the reduction,
level by level, with the owned M2M; the all-gather of R; the multipole exchange, level
by level; `downward`; the leaves.

### 4.5 Events and the device

Three new `HostData` events (`nd-fmm-plan`), names for sign-off (§12, question 4):

| Event | When | Lists |
| --- | --- | --- |
| `SendReduction { level, boxes }` | before the reduction sends level l | the own top-tree keys of level l the rank sends (blocks and owned `Global` keys) |
| `ReceivedReduction { level, boxes }` | after the received children of level l are written | the ghost slots written |
| `ReceivedTop { boxes }` | after the all-gather of R wrote R's slots | R's slots not owned by the rank |

`ReceivedCoarse` goes with the gather. `SendMultipoles` moves after the all-gather of R,
since its sends now include owned `Global` keys final only after the reduction; it still
fires once, for every level's sends. **5 + nlevels + 2r** events per evaluation (r the
reduction levels; Phase 5's 5 + nlevels, minus `ReceivedCoarse`, plus `ReceivedTop` and
two per reduction level), on every rank, empty lists on one rank.

The device operator (device-path §14) at each: `SendReduction`, one gather of the listed
columns into a packed buffer and one download (**one sync**: the values must reach the
host before the MPI send, and the owned keys of the level were formed on the device); `ReceivedReduction` and
`ReceivedTop`, one upload and one `scatter_columns` each. Per evaluation, extending
distributed §7.2 (n_c s bytes per multipole):

| Direction | Bytes added | Calls added |
| --- | --- | --- |
| up | Σ_l R^red_l n_c s + |R ∖ own| n_c s, minus Phase 5's B_r n_c s | one per level with received children, one for R |
| down | Σ_l S^red_l n_c s | one per level the rank sends |
| syncs | | **at most 2 + r_s** (1 + [anything sent] + r_s), r_s the reduction levels at which the rank sends (at most 7 measured) |

The device on several ranks stays a correctness check (Phase 5 decision 3), so the extra
syncs cost nothing that is measured as scaling.

### 4.6 Overlap

`far_field` posts the receives of every reduction level first; then for l from the
deepest block level up: `SendReduction` of level l and post level l's sends (own blocks
and owned keys of level l, all final by then), wait for level l's children,
`ReceivedReduction`, the owned `m2m` of level l − 1, a `test` after each call, so that the
events fire with the blocking path's lists and in its order; then the all-gather of R
(`immediate_all_gather_varcount_into` with the downward levels above R's deepest level
running behind it is possible, but T5 keeps it blocking, as Phase 5 kept the gather,
distributed §8.5); then the multipole exchange as today, each level waited for before its
downward calls. Every level call runs in the blocking path's order, so the output is bit
for bit the blocking path's.

## 5. S7: validation at scale

### 5.1 Rank-count invariance

**The rule.** A run on P ranks equals a run on P/k ranks bit for bit when rank r of the
small run holds the concatenation of the inputs of ranks k r, …, k r + k − 1 of the large
one, in that order. Both are the one-rank `Fmm` over the same union in the same order:
the tree is the one-rank tree (O1), near and X rows are ordered by the entry's (level,
key) (P1), V rows by offset index and W rows by key, every
top-tree multipole is the M2M of its children by octant (S2), and within a leaf the
points are ordered by (origin rank, origin position), which the concatenation preserves.
Two conditions are easy to miss and the check must fix them:
- **the strategy is named** (`Dense`, `Rotation` or `Classes`): `Auto` resolves by the
  ranks per node, 40 at 160 ranks on 4 nodes and 5 at 20, and Rotation and Dense differ
  in the last bits (T4.4 of phase5n-kathleen.md);
- the precision, p, `max_level`, the leaf size and the domain are the same (the domain is
  the bounding box of the same union, so it is).
Threads, overlap, the table cache and the cut (S3) do not change bits.

**Inside one MPI job.** Every rank generates its input by index (`Workload::share` gives
rank j of P the points i with `hash(i) mod P = j`, in index order), so any rank produces
any rank's share. The world (P ranks) runs first. Then `world.split` with colour 0 on
the ranks k r (the first of each group of k) and no colour elsewhere gives the small
communicator; the other ranks skip the small run and wait at the next world collective,
entering nothing of the small communicator. Each small rank r generates the shares of
world ranks k r … k r + k − 1, concatenates them, and runs. The comparison moves outputs
by index, never gathers: small rank r sends each world rank k r + i its slice of the
output (point-to-point, sizes known on both sides), world rank k r + i compares bit for
bit with its own output, and one all-reduce over the world agrees the count of differing
values, so the check fails on every rank or on none.

**In CI (T2):** P = 4 against 2 and 1 (k = 2 and 4) and P = 2 against 1, in
`run-tests-mpi`; the cube, the Plummer sphere and a graded scenario with one empty rank,
N ≈ 10⁴, f64 p = 8 and f32 p = 3, overlap off and on (T2 brief). **At scale:** P against
P/8, 160 against 20 on Kathleen's 4 nodes (the small run's ranks are one in eight, spread
over the nodes), 640 against 80 oversubscribed on the same 4 nodes (1,280 against 160,
the T2 brief's first plan, did not start, §10.1), 512 against 64
oversubscribed on locust, N = 10⁶–10⁷. The small run holds k times the points per rank:
at 160 against 20 and N = 10⁷, 5 × 10⁵ points per small rank (under 2 GB).

### 5.2 The direct-sum sample and generation by index at N ≥ 10⁸

The `scaling` harness already has both (Phase 5 T10): every point by index, and the
errors against the direct sum at 1,000 sampled targets, each rank summing its own
sources and one all-reduce adding the parts (f64; the sum's order depends on P only in
the last bits, far inside the 0.1% (f64) and 1% (f32) acceptance against a run at a
smaller rank count). What changes at N = 10⁸ and beyond:
- **no reference on rank 0**: `--reference` gathers every point to rank 0; it refuses
  (an agreed error with the size in its message) above N = 2 × 10⁷, the largest that ran
  on one rank in Phase 5N (T2 implements it);
- **cost**: the sample costs N/P × 1,000 pair evaluations per rank, 6 × 10⁸ at N = 10⁸
  on 160 ranks (about a second, *model*), and one all-reduce of 4,000 values;
- **memory**: the harness holds O(N/P), but generating the random share filters all N
  indices on every rank: O(N) time per rank, 10⁸ hashes at N = 10⁸ (about a second,
  *model*), 10¹⁰ at N = 10¹⁰. T2 keeps the share for N ≤ 10⁹ and adds an input of
  contiguous index ranges for larger N (O(N/P) time), stated in the report.
T0 ran N = 10⁸ without the reference on 1, 2 and 4 nodes (S0.5): the errors agree in every
printed digit across splits and node counts.

## 6. S3: the work-weighted cut

### 6.1 The cost model, measured

Per rank, the compute of an evaluation is Σ_k α_k c_k, over the eight operator kinds,
with counts c_k from the plan and constants α_k from per-kind timings
(`KindTiming::Synchronous`, the median of three evaluations, one thread per rank):

| Kind | Count c_k | α_k, locust, f64 p = 8, Rotation (ns) | α_k, Kathleen, same (ns) |
| --- | --- | ---: | ---: |
| P2M | source points | 161 | 585–598 |
| M2M | children (both passes) | 1,009 | 2,711–2,743 |
| M2L | V pairs | 1,079–1,128 | 3,076–3,104 |
| P2L | X pairs × source points | 178–186 | 559–593 |
| L2L | boxes with a parent | 1,073 | 2,762–2,801 |
| L2P | target points | 390 | 1,032–1,071 |
| M2P | W pairs × target points | 442–456 | 1,179–1,250 |
| P2P | near interactions (Σ_leaf n_t Σ_near n_s) | 2.44–2.58 | 2.80–3.18 |

(ranges over the cube, the Plummer sphere and the clusters at N = 10⁷, on locust at 8,
32 and 72 ranks and on Kathleen at 40, 80 and 160 ranks, one thread per rank; a constant
is the sum of a kind's times over the ranks over the sum of its counts.) The constants
are stable across workloads and rank counts (within 5% on locust, within 13% on
Kathleen, P2P's spread), and **the model reproduces every rank's measured compute within
1.0–1.4%** at 8, 32 and 72 ranks on locust, and the measured max/mean: 1.31 against 1.32
modelled (Plummer, 72, locust), 1.51 against 1.52 (Plummer, 160, Kathleen), 1.31 against
1.26 (clusters, 160). **The ratios between kinds differ by machine** (M2L over P2P: 440 on
locust, 970–1,100 on Kathleen), so the constants are per machine class, not universal.
With 20 threads per rank (Kathleen 2 × 20) the model is looser (1.13 measured against
1.02 modelled on the Plummer sphere): the kind times of threaded calls do not add per
rank as simply. The
W and X kinds matter on adaptive trees: on the Plummer sphere at 72 ranks M2P and P2L are
22% of the compute on average and up to 35% on the busiest rank. A model of points and
boxes alone (§14.4 of distributed) misses them.

### 6.2 The cut by modelled cost, replayed

Per block, the modelled cost is Σ_k α_k c_k(block), the counts of the boxes and leaves
below it; the replicated work of the shared ancestors stays with its rank. The probe
replays the nearest-boundary cut (O3) on every block with three weights, and reports the
modelled per-rank cost and the points, max over mean (locust; 256 and 512 oversubscribed
with the 72-rank constants):

| Workload, N, ranks | by points (Phase 5): cost / points | by modelled cost: cost / points | by a local proxy: cost / points | heaviest block / fair share |
| --- | --- | --- | --- | ---: |
| cube 10⁷, 72 | 1.067 / 1.017 | 1.017 / 1.071 | 1.066 / 1.007 | 0.022 |
| Plummer 10⁷, 72 | 1.322 / 1.060 | 1.049 / 1.406 | 1.081 / 1.333 | 0.087 |
| clusters 10⁷, 72 | 1.188 / 1.044 | 1.050 / 1.291 | 1.071 / 1.291 | 0.124 |
| cube 10⁷, 512 | 1.191 / 1.121 | 1.113 / 1.215 | 1.186 / 1.110 | 0.143 |
| Plummer 10⁶, 512 | 1.705 / 1.098 | 1.066 / 1.845 | 1.222 / 1.419 | 0.153 |
| Plummer 10⁷, 512 | 1.627 / 1.082 | 1.045 / 1.669 | 1.154 / 1.558 | 0.146 |
| clusters 10⁶, 512 | 2.685 / 1.075 | 1.060 / 1.480 | 1.186 / 1.488 | 0.161 |
| clusters 10⁷, 512 | 1.551 / 1.087 | 1.059 / 1.564 | 1.162 / 1.559 | 0.149 |
| Kathleen: cube 10⁷, 160 | 1.091 / 1.026 | 1.035 / 1.132 | 1.092 / – | 0.050 |
| Kathleen: Plummer 10⁷, 160 | 1.520 / 1.064 | 1.094 / 1.481 | 1.137 / – | 0.136 |
| Kathleen: clusters 10⁷, 160 | 1.259 / 1.073 | 1.087 / 1.546 | 1.162 / – | 0.124 |
| Kathleen: cube 10⁷, 320 | 1.095 / 1.031 | 1.070 / 1.182 | 1.099 / 1.031 | 0.099 |
| Kathleen: Plummer 10⁷, 320 | 1.574 / 1.083 | 1.089 / 1.813 | 1.188 / 1.604 | 0.195 |
| Kathleen: clusters 10⁷, 320 | 1.350 / 1.077 | 1.096 / 1.613 | 1.139 / 1.624 | 0.189 |
| Kathleen: cube 10⁷, 640 | 1.088 / 1.020 | 1.020 / 1.225 | 1.092 / 1.025 | 0.038 |
| Kathleen: Plummer 10⁷, 640 | 1.710 / 1.088 | 1.055 / 1.859 | 1.181 / 1.640 | 0.126 |
| Kathleen: clusters 10⁷, 640 | 1.458 / 1.088 | 1.053 / 1.740 | 1.170 / 1.673 | 0.168 |

(Kathleen rows: 160 ranks timed, job 252220, Kathleen's constants; 320 and 640
oversubscribed with the 160-rank constants, the cube in job 252846, the others in 255188,
the local proxy fitted to the oversubscribed times; 1,280 did not start, §10.1.) What it
shows:
- **the cut by points degrades with P**: 1.19 (cube) to 2.69 (clusters) times the mean
  modelled cost at 512 ranks, 1.09–1.71 at 640 Kathleen ranks; the measured compute at 160 ranks on Kathleen (T0, S0.7)
  is 1.31–1.62 on the Plummer sphere at 40 per node;
- **the cut by the exact model holds 1.02–1.11** at 72–640 ranks, limited by the
  heaviest block (O3: at most half a block above the mean; blocks reach 0.09–0.22 of a
  fair share in cost, because O1 refines them by points);
- **a proxy from the local refinement alone** (keys, Σ keys² over the pre-balance
  leaves, leaves; fitted per run, so optimistic) reaches only 1.15–1.22 at 512;
- the cut by cost unbalances points by up to 1.86: memory per rank follows points
  (§8.2), and at 4.8 GB per core that is affordable.

### 6.3 Design

1. **Weights from the plan, not from the local refinement.** The second cut runs after
   `Plan::new` on the first cut: every rank sums the eight counts per own block (the
   near interactions need the ghost leaves' source counts, which the source exchange's
   build delivers; the octree records the keys per leaf, with multiplicity, so the
   counts exist before the points move), weighs them with the constants, and passes one
   weight per own block to `nd-octree`.
2. **The cut in `nd-octree`.** `Octree::repartition(&self, weights: &[f64]) -> Octree`:
   one all-gather-v of the block weights (B values, counts from the map), the
   nearest-boundary cut by weight (O3), the leaves of the moved blocks sent to their new
   owners (one all-to-all-v), H and the leaf bits rebuilt (§3.3). The coarse tree, the
   leaves and every key stay; only ownership changes. Then `Plan::new` again, and the
   points move once, after the final cut (`Fmm::build`'s redistribution).
3. **Optionally, blocks refined by cost** (§12, question 5): a block heavier than a
   fair share over k (in cost) is replaced by its children before the cut, as O1 refines
   by points. The children are nodes of the tree, so the leaves do not change; the coarse
   tree does (more blocks, more shared ancestors), which T7's brief forbids, hence a
   sign-off question. Without it the cut stays at about 1 + w_max/2, 1.05–1.11 at 512
   ranks.
4. **The constants** live in `nd-fmm-exec`: a table per strategy, precision and p
   (measured on Kathleen by a calibration example with `kind_timings`), and
   `FmmBuilder::cost_model` to override it. Only their ratios move the cut, and only the
   partition depends on them: a wrong constant costs balance, never a bit. The cut does
   not depend on timings taken during the run, so it is deterministic.

Cost of the second cut: one more `generate_all_keys` push, one more `Plan::new`, an
all-gather-v of B weights and an all-to-all-v of leaves: at 160 ranks on N = 10⁸ about
0.5–2.5 s of `Plan::new` (T0, S0.8), against an evaluation that the cut shortens from
1.6 to about 1.1 times the mean compute (Plummer at 40 per node; *model* from §6.2).

### 6.4 Why no bit changes, the target, the default

The leaves are the one-rank leaves for any cut (O1); a block's owner changes, not its
keys; every row is a row of the same tree in (level, key) order. So the outputs are the
one-rank `Fmm`'s bit for bit at every rank count, as with the first cut.

**Target** (decision 3), restated on what can be measured: the modelled per-rank cost's
max over mean **≤ 1.10 at 512 (locust) and 640 (Kathleen) oversubscribed ranks**
on the cube, the Plummer sphere and the clusters at N = 10⁶–10⁷ (distributed §14.5: 1.10
at 1,024 ranks), and the measured compute stages' max over mean at 72 (locust) and 160
(Kathleen) ranks better than the first cut's. Without refinement by cost the replay
gives 1.05–1.11, so the target needs question 5's refinement or accepts 1.12.
**Default:** on, on several ranks, once T7 has measured its build cost at 160 ranks below
a quarter of the build; one rank unchanged.

## 7. S4 and S5: triggers

### 7.1 S4, construction at scale: the trigger has fired

`Octree::new` by step (the probe's replay, a barrier before each step, max over the
ranks; the replay reproduces the octree key for key on every run), N = 10⁷:

| Machine, ranks | first `parsort` (ms) | its keys received, max over passed | its peak resident, max / mean (MB) | O1 rounds (ms, rounds) | key move 1 (ms) | second `parsort` (ms) | `generate_all_keys` (ms) | `Octree::new` (ms) |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| locust, 8 | 237–238 | 2.0 | 407 / 328 | 56–74, 4–8 | 8.4 | 1.5–5.2 | 32–85 | 388–520 |
| locust, 32 | 173–175 | 8.0 | 271 / 100 | 23–36, 4–9 | 11.5 | 0.8–1.8 | 8–21 | 232–275 |
| locust, 72 | 186–188 | 17.9 | 241 / 59 | 20–29, 5–9 | 15.5 | 1.7–3.4 | 6–13 | 236–262 |
| Kathleen, 40 | 413–418 | 9.96 | 243 / 83 | 43–75, 4–9 | 25 | 2.1–3.9 | 15–36 | 552–634 |
| Kathleen, 80 | 404–416 | 19.9 | 240 / 74 | 40–62, 5–9 | 32 | 2.1–3.9 | 11–25 | 533–602 |
| Kathleen, 160 | 2,486–2,512 | 39.4 | 230 / 57 | 36–64, 5–10 | 35 | 1.9–4.8 | 7–20 | 2,607–2,656 |
| Kathleen, 2 × 20 per node, 8 ranks | 452–456 | 2.0 | 419 / 340 | 86–119, 4–8 | 15–17 | 2.4–6.4 | 37–104 | 685–854 |
| Kathleen, 40, N = 10⁸ | 4,750–4,763 | 9.99 | 2,223 / 655 | 510–822, 4–9 | 261–270 | 12–32 | 147–550 | 6,140–7,317 |
| Kathleen, 160, N = 10⁸ | 8,526–8,592 | 39.8 | 1,939 / 200 | 357–617, 5–10 | 355–357 | 5–11 | 35–127 | 9,656–10,300 |

(the cube, the Plummer sphere and the clusters; ranges over the three; Kathleen job
252220, 40 ranks per node unless stated.) The first sort is 72–79% of `Octree::new` at
72 ranks on locust and 88–96% at 160 on Kathleen, and does not fall with P (§1.4). At 128–512
oversubscribed ranks its heavy rank receives N/4 keys (31.5–123.5 times its share) with
a peak of 290–376 MB at N = 10⁷; on Kathleen at N = 10⁸ the heavy rank peaks at 2.22 GB
(40 ranks) and 1.94 GB (160 ranks) during the sort.

**Trigger (decision 4), stated on what is measured:** S4 is built when, in either sample
sort, the most keys any rank receives exceed twice the mean, or the sorts take more than
a quarter of `Octree::new` at the largest timed rank count. **Both hold now**, at every
rank count from 8 up, on the random share, with rlst 0.9.0. (With rlst PR #40, below,
the imbalance clause no longer holds and the time clause still does: a sort remains
70–80% of the construction, now scaling with N/P.) **Scheduled** (at the user's request,
2026-10-11) as **T3a** (docs/phase5s/T3a-octree-key-routing.md), before T3 rather than in
T8, because every later task's 4-node build times and memory carry the defect:
- **the first sort** is replaced by routing the input keys to provisional owners by a
  weight histogram: O1's own rounds (local counts per block over the unsorted keys, one
  all-reduce per round) with the split rule by weight alone, then the cut and one
  all-to-all-v of the keys; every copy of a key lands on one rank, where a local sort
  and deduplication give the linearized keys, whose distinct counts make O1's real rounds
  exact. The rounds cost what O1's do (an all-reduce of 2B u64 each);
- **the second sort** (inside `balance`) is replaced by routing the closure keys to the
  owners by the partition bounds and linearizing locally: a key whose descendants exist
  lies in the range of the rank that owns its first descendant, so the global
  linearization is the union of the local ones;
- the leaves, the partition and every key stay bit for bit (compared key by key, as
  T8's brief asks), so no output changes.
**The splitter fix upstream.** rlst PR #40 (codeberg rlst/rlst, opened 2026-10-11 at the
user's request) replaces `parsort`'s four local quantiles by histogram sort with sampling,
so that every rank receives exactly ⌊N/P⌋ or ⌈N/P⌉ records whatever the input. Measured
in the same `Octree::new` (the probe built against the PR's branch; locust, N = 10⁷;
the octree identical key for key):

| | ranks | rlst 0.9.0 | rlst #40 |
| --- | ---: | ---: | ---: |
| first sort, slowest rank (ms), cube | 8 / 32 / 72 | 237 / 172 / 187 | 180 / 77 / 62 |
| keys received over passed, max / min over ranks | 72 | 17.85 / 2 × 10⁻⁵ | 1.006 / 0.992 |
| peak resident during the sort, max (MB) | 72 | 241 | 72 |
| `Octree::new` (ms), cube / Plummer sphere | 72 | 238 / 263 | 84 / 104 |
| `Octree::new` (ms), cube, oversubscribed (not a timing) | 512 | 2,444 | 388 |
| peak resident during the sort, max (MB), oversubscribed | 512 | 374 | 94 |

On Kathleen (job 252846, 40 ranks per node, the same comparison):

| | ranks | rlst 0.9.0 | rlst #40 |
| --- | ---: | ---: | ---: |
| first sort, slowest rank (ms), N = 10⁷ | 160 | 2,486–2,512 | 59 |
| `Octree::new` (ms), N = 10⁷, cube / Plummer | 160 | 2,607 / 2,655 | 73 / 92 |
| first sort, slowest rank (ms), N = 10⁸ | 40 / 160 | 4,750–4,763 / 8,526–8,592 | 1,944–1,957 / 522–524 |
| `Octree::new` (ms), N = 10⁸, cube / Plummer | 40 | 6,140 / 7,317 | 2,692 / 3,624 |
| `Octree::new` (ms), N = 10⁸, cube / Plummer | 160 | 9,656 / 10,093 | 689 / 900 |
| peak resident during the sort, max / mean (MB), N = 10⁸ | 40 | 2,220 / 655 | 611 / 607 |
| peak resident during the sort, max / mean (MB), N = 10⁸ | 160 | 1,935 / 200 | 192 / 188 |

At 160 ranks `Octree::new` becomes 11–36 times faster, and the four 2 GB ranks of T0
disappear. The PR cures the imbalance for every user of
`parsort`; T3a removes the octree's two sorts altogether (no rlst dependency for the
octree's splitters, one all-to-all-v of keys instead of a sort's, deduplicated keys on
the wire). **rlst 0.10.0 (released 2026-10-11) carries #40, and T1s (scheduled at the
user's request, before T2) moves the pin to it**, with CubeCL 0.11.0; T3a then starts
from balanced sorts.

### 7.2 S5, the O(P) collectives

What remains O(P) after S1–S3 is counts, not data: every `MPI_Alltoall` of P counts (the
key moves, the push of §3.3, `Redistribution::new`, rlst's ghost-communicator builds, S2's
exchange builds), the leaf-bit and R-key all-gathers' count arrays, and two
`MPI_Alltoallv` per evaluation (the redistribution's forward and backward), whose argument
arrays are P long and whose algorithm in Open MPI may visit every peer even when most
counts are zero. Measured: `MPI_Alltoall` of one i32 per peer takes 10.1 µs at 72 ranks
on locust, and 47.6, 52.0 and 59.8 µs at 40, 80 and 160 ranks on Kathleen (§10.3). Per build there are about 2 (key moves) + 1
(push) + 2 (`Redistribution::new`) + 1 + nlevels (source and multipole exchanges) +
nlevels (S2's exchanges) + 2 (the sorts, until S4) ≈ 40 count all-to-alls at
nlevels = 16, and 2 more with S3's second cut (the moved leaves and the second push).

*Model* at P ranks: an all-to-all of counts costs at most about P α_a, with α_a = 0.37 µs
per peer (the 160-rank figure; Open MPI's algorithm for small all-to-alls makes the
measured cost grow slower than P from 40 to 160), so 40 of them cost about 15 ms per
build at 10³ ranks and 0.15 s at 10⁴ (*model*). The per-evaluation `MPI_Alltoallv` pair
costs about 2 P α_v when every peer is visited: with α_v = α_a, 0.76 ms at 1,024 ranks,
3.0 ms at 4,096, 6.1 ms at 8,192 and 7.4 ms at 10⁴ (*model*).

**Triggers (decision 4):**
- **S5 for the evaluation** fires when the two redistribution all-to-all-v's, modelled at
  10⁴ ranks, exceed 1% of a modelled evaluation at 10⁴ points per rank. The remedy is
  cheap and needs no consensus: the routes are fixed at build, so `forward` and `backward`
  become scoped point-to-point between the ranks with non-zero counts (immediate sends
  and receives, as the overlapped exchanges do). **It fires on the model**: 7.4 ms per
  evaluation at 10⁴ ranks is 1.5% of a modelled f64 p = 8 evaluation at 10⁴ points per rank
  (about 0.5 s, from §10.4's per-point compute) and 5–8% at p = 3. T8 builds it.
- **S5 for the build** fires when the count all-to-alls, modelled at 10⁴ ranks, exceed a
  quarter of a modelled build. **It does not fire on the model** (0.15 s of a build of
  several seconds at 10⁴ ranks). The remedy is the sparse dynamic exchange (non-blocking
  consensus): synchronous sends (`Destination::immediate_synchronous_send`), probes
  (`Source::immediate_matched_probe`, `Message::matched_receive_into`) and a
  non-blocking barrier (`CommunicatorCollectives::immediate_barrier`), all in rsmpi
  0.8.2 (src/point_to_point.rs and src/collective.rs), with `any_process()`
  (src/topology/mod.rs) as the probe source. **No rsmpi addition is needed.** rlst's ghost-communicator build keeps its
  all-to-all (an rlst change, asked for upstream, not forked) unless that trigger fires.

### 7.3 T8: the triggers on the final structure (addendum, 2026-10-13)

T8 (docs/phase5s/T8-construction-and-collectives.md; fmm-validate/results/phase5s-t8.md)
measured S4, S5 and S8 (the fan-out below R, added by T5) on main at e788576 (T3a, T3,
T5–T7). It used locust at 8–72 ranks (timed) and 128–512 (counted), and Kathleen at
40–160 ranks (timed) and 320–640 (counted; job 604012, 1.99 node-hours). Counts and
times per call came from an instrumented copy of the tree that was not committed.
Every figure at 10³–10⁴ ranks is a *model*.

| Package | Trigger | Measured or modelled | Outcome |
| --- | --- | --- | --- |
| S4 | most keys received > 2 × mean, or sorts > ¼ of `Octree::new` | no sorts since T3a; keys after routing within 1.002–1.083 of the mean (timed) and 1.02–1.12 (128–640 counted); routing 35–65 % of `Octree::new`, which is 7–12 % of the build | closed (built in T3a) |
| S5, build | the all-to-alls of counts > ¼ of a modelled build at 10⁴ ranks | 21–32 per build, 19–30 of them in rlst (12–23 in `GhostCommunicatorBuilder::build`, 7 in `array_tools::all_to_allv`), 2 in `Redistribution::new`; timed 0.5–3 % of the build at 40–160 ranks; *model* at 10⁴ (2.5 × 10⁴ points per rank): 19–27 % with §7.2's α_a P (0.376 µs per peer), 6–9 % with the measured trend 43.8 µs + 0.102 µs P | on the edge on §7.2's bound; does not fire on the trend |
| S5, evaluation | the all-to-all-v pair > 1 % of an evaluation at 10⁴ points per rank | with every count 0, Open MPI 4.1.6 took 0.3, 167 and 1.2 µs at 40, 80 and 160 ranks: its `tuned` component chose pairwise (P − 1 steps, 2.1 µs per peer) at 80 and basic linear (zero counts skipped) at 40 and 160; *model* at 10⁴: 0 / 7.5 / 42 ms against 0.22–5.7 ms | fires on §7.2's model and on pairwise, not on basic linear |
| S8 | the busiest sender's multipole messages at 10⁴ × 4 µs > 1 % of an evaluation at 2.5 × 10⁴ points per rank; or the multipole exposed wait > 1 % of the evaluation at 160 ranks | messages 41–93 at 40 ranks, 113–486 at 512–640 (fits P^0.71–0.75 on the clusters, P^0.28–0.38 on the cube and the Plummer sphere); *model* at 10⁴: 1.4–28 ms against 5.5–14 ms (f64 p = 8) and 0.56–1.85 ms (f32 p = 3); exposed wait at 160 ranks 0.002–0.013 ms (≤ 0.004 %) | modelled clause fires (f32 p = 3 on every workload from 600–8,000 ranks; f64 p = 8 under the linear fits); measured clause does not |

Still O(P) and covered by no package: the all-reduces of the routing's weight rounds and
of O1's rounds. They move about 0.9 MB per build at 640 ranks, growing like P; *model*
about 14 MB and 57 ms per build at 10⁴ ranks (distributed §14.2 called them
"acceptable"). Open: every timing at scale, which all-to-all-v algorithm Open MPI picks
at 10³–10⁴ ranks, contention on a node's link, collective algorithms at thousands of
ranks, and R's cap setting the threshold above 640 ranks.

**S5 is built upstream** (the user, 2026-10-13): T8r
(docs/phase5s/T8r-rlst-sparse-exchange.md) adds a sparse `all_to_allv`, an NBX
`GhostCommunicatorBuilder::build` and a fixed-route exchange to rlst, every result
unchanged, released as rlst 0.10.1. T8's step 3 adopts them for the build and for the
evaluation's redistribution. **S8 fires on the modelled clause and is built as its own task after T8, T8s**
(docs/phase5s/T8s-fan-out-tree.md; signed off by
the user on 2026-10-13; the report's §6, question 3): the forwarding tree of a box's
readers that T8's brief describes, R, its cap and K unchanged.

**S5, built (T8 step 3, 2026-10-14; the report's §9).** rlst 0.10.1 is pinned (T8r: rlst
PRs #42 and #43, tag `v0.10.1`), and every output is unchanged bit for bit:
- the octree's key move, balance, push and the second cut's leaf move go through
  `sparse_all_to_allv`;
- `Redistribution` routes by rlst's `SparseRoutes` (point to point in `forward` and
  `backward`, no collective, no P-long array);
- rlst's ghost builds find their send lists by consensus underneath.

Counted per build: 21–32 all-to-alls of P counts become 2 (the routing and
`leaf_point_counts`), the P-long arrays fall by 65–80 % and the all-reduces by a quarter,
with 19–30 non-blocking barriers new. *Model* at 10⁴ ranks: 2.1–2.7 % of a build on
§7.2's bound and 0.8–1.1 % on the trend. Timed:
- the redistribution per evaluation is 6–30 % faster at 160 Kathleen ranks;
- the build is 2–6 ms slower on one node, where the harness's random share makes every
  route dense;
- on 4 Kathleen nodes the build trips a 160 ms transport stall (Open MPI 4.1.6 over PSM2)
  more often than before. A micro-benchmark finds that stall on every kind of dense
  burst, the old all-to-all-v included, and never on 2 nodes; it is open for T9.

## 8. Collectives and memory per rank

### 8.1 Every collective, in order, after S1, S2 and S3

Changes against distributed §9.1 in bold. Every rank enters every row, an empty rank
with empty lists and zero counts; no collective sits in a branch only some ranks take.

| Call | Collectives, in order |
| --- | --- |
| `Octree::new` (P > 1) | `linearize` (**S4: the histogram rounds and one all-to-all-v replace the sample sort**); O1: one all-reduce per round and one after balancing; the key move (all-to-all-v; **T8, S5: rlst's `sparse_all_to_allv`, a message per rank with keys, a non-blocking barrier and one all-reduce**); `balance` (all-reduce of the deepest level; **S4: an all-to-all-v by the bounds replaces the second sort; T8: a sparse exchange**); the key move again; **the leaf-bit all-gather-v (B bytes, §3.2); the push (§3.3; T8: a sparse exchange in place of an all-to-all of P counts and an all-to-all-v)**. The two `gather_to_all` of `generate_all_keys` go |
| `Plan::new` | two all-reduces (unchanged) |
| **S3: `Octree::repartition`** (several ranks, default on) | **all-gather-v of the block weights (B values); the moved leaves (T8: a sparse exchange in place of an all-to-all of P counts and an all-to-all-v); the leaf-bit all-gather-v; the push (as `Octree::new`)**; then `Plan::new` again |
| `Redistribution::new` (twice) | **T8, S5: rlst's `SparseRoutes::new` (communicator duplicate, a sparse exchange of the item counts, an all-reduce); the all-reduce of errors; the keys and positions point to point along the routes** (before T8: all-to-all of counts, all-reduce, communicator duplicate, all-to-all-v) |
| `Evaluator::new` | all-reduce; `SourceExchange::new`; `MultipoleExchange::new` (one rlst build per level); **the reduction's exchanges (one rlst build per level, §4.2); R: one all-reduce of a reader histogram, one all-gather of P counts and one all-gather-v of R's keys (§4.3)**. `CoarseExchange::new` goes |
| `Fmm::build` | as distributed §9.1 and node-m2l.md §4 (step agreements, `split_shared`, the table-cache barrier, the device agreement), with the rows above |
| `Fmm::evaluate`, blocking | all-reduce (charge length, kept device error); forward of the charges (**S5, built in T8: point-to-point on the build's routes, rlst's `SparseRoutes`**); the source exchange; **the reduction: one neighbour exchange per level; the all-gather-v of R**; the multipole exchange, one per level; with a device on P > 1, one all-reduce (device errors); backward of the output (**S5: point-to-point**) |
| `Fmm::evaluate`, overlapped | the same, the reduction and the source and multipole exchanges as scoped point-to-point (§4.6); the all-gather of R blocking |

New collectives per build, and why each is needed (README, "Errors": no new agreement):
the leaf-bit all-gather-v (the kind of other ranks' blocks is not known otherwise); the
reduction's builds (one per level, the existing rlst mechanism); R's histogram and key
all-gather (the threshold is global). No new agreement: their errors ride on the
existing validity all-reduces of `Octree::new`, `Evaluator::new` and `Fmm::build`. Per
evaluation, against Phase 5: the coarse gather (all-gather-v of B n_c s) is replaced by
r neighbour exchanges and one all-gather-v of fixed size.

### 8.2 Memory per rank, and what remains O(P)

Notation as distributed §9.2; after S1 and S2 the held boxes are K = K_loc + K_halo +
K_top + K_R: own, halo (the ghosts of H below other ranks' blocks), H's top-tree keys,
and the replicated top not already in H.

| Data | Bytes | Grows like |
| --- | --- | --- |
| multipoles, locals | 2 n_c s K | N/P + surface + shared ancestors (bounded) + |R| (fixed) |
| source store, target input and output | (4 (N_s + G_s) + 3 N_t + o N_t) s | N/P + halo |
| box index, leaf numbering | 32 K + 13 (J_loc + J_gh) | as K |
| views | 12 E_V + 4 (E_U + E_W + E_X) + … | N/P + halo |
| `Redistribution` (two) | 4 per sent, 12 per received point; **32 P of count arrays each** | N/P; **P** |
| exchange buffers | values sent and received, s each; R's buffer |R| n_c s | halo; fixed |
| partition map | 8 B + B/8 + 8 (P + 1) | **B, P** |
| ghost-communicator count arrays (rlst) | 8 P per build, transient (kept per neighbour) | **P**, transient |

Every remaining O(P) item, per rank, *model* (36 blocks per rank, nlevels = 16, f64,
p = 8; at 2.5 × 10⁴ points per rank):

| Item | When | 10² ranks | 10³ ranks | 10⁴ ranks |
| --- | --- | ---: | ---: | ---: |
| partition map (keys, leaf bits, cut) | build, kept | 30 kB | 300 kB | 3.0 MB |
| leaf-bit all-gather-v, received | build | 3.6 kB | 36 kB | 360 kB |
| O1 rounds: all-reduce of 2B u64 (last round) | build, 5–10 rounds | 58 kB | 576 kB | 5.8 MB |
| count arrays of one all-to-all | build, about 40 calls | 0.4 kB | 4 kB | 40 kB |
| rlst count arrays of one build | build, transient | 0.8 kB | 8 kB | 80 kB |
| `Redistribution` count arrays (two) | kept | 6.4 kB | 64 kB | 640 kB |
| S3's block weights, all-gather-v | build | 29 kB | 288 kB | 2.9 MB |
| R's keys and counts | build | 33 kB | 37 kB | 73 kB |
| the redistribution's `MPI_Alltoallv` (until S5) | evaluation | P-long argument arrays; latency below (§10) | | |
| fan-out of the most-read top-tree box below R | evaluation | ≤ max(2, ⌈P/16⌉) − 1 = 6 | ≤ 62 (cap not binding at 640, measured 34–39) | ≤ 624, or the readers of the 4,096th box if the cap binds (not measured) |

Against Phase 5 at 8,192 ranks (distributed §14.2, *model*): about 1.2 million replicated
held boxes (1.5 GB of coefficient slots) and 170 MB of coarse gather per evaluation.
After S1 and S2 the held boxes are H plus R: measured H ≤ 9,177 keys at 2.5 × 10⁴ points
per rank up to 640 ranks and R ≤ 4,096 at every rank count (§1.3, §4.3), at most 17 MB of
coefficient slots at p = 8 in f64.

Measured per rank: the peak resident size (`VmHWM`) of the Phase 5 build by part, max /
mean over the ranks in MB, the ranges over the cube, the Plummer sphere and the clusters
(Kathleen, rlst 0.9.0):

| Ranks, N, job | after input | `Octree::new` | `Plan::new` | after the evaluations |
| --- | --- | --- | --- | --- |
| 40, 10⁷, 252220 | 43–45 / 42–43 | 263–267 / 100–105 | 102–156 / 97–129 | 243–337 / 229–306 |
| 80, 10⁷, 252220 | 52 / 50–51 | 256–258 / 87–89 | 92–120 / 86–104 | 195–244 / 181–225 |
| 160, 10⁷, 252220 | 47–48 / 44–45 | 242–245 / 69–73 | 76–92 / 70–83 | 163–216 / 151–203 |
| 40, 10⁸, 252220 | 263–264 / 261 | 2,256–2,270 / 691–705 | 562–1,100 / 540–868 | – |
| 160, 10⁸, 252220 | 103–104 / 100–101 | 1,968–1,976 / 225–230 | 206–382 / 198–287 | – |
| 320, 8 × 10⁶, 255188 (oversubscribed) | 64–65 / 59 | 222–225 / 87–94 | 104–112 / 89–98 | – |
| 640, 1.6 × 10⁷, 255188 (oversubscribed) | 91–92 / 86–87 | 396–402 / 127–133 | 145–155 / 130–136 | – |

The maximum of `Octree::new` is the first sort's heavy rank in every row (§1.4); without
it the build's peak follows the points and the held set, and `Plan::new`'s maximum the
held set (the Plummer sphere and the clusters above the cube). The resident size after
input grows with the ranks per node (MPI's and PSM2's buffers), not with the points: 42
MB per rank at 40 ranks and 2.5 × 10⁵ points, 86 MB at 640 and 2.5 × 10⁴. The probe also
measures the model's inputs (held keys, buffers) on the same runs.

## 9. Testing

### 9.1 The layers per task

| Layer | Task | Where | Ranks | Oracle |
| --- | --- | --- | --- | --- |
| rank-count invariance | T2 | `nd-fmm-exec`: a new MPI-owning test executable `tests/invariance.rs`, registered in `run-mpi-tests.sh` (keeps `mpi_exec`'s one-minute debug budget); `nd-fmm-validate`: `scaling --invariance k` | 1, 2, 4 (CI); 8 by hand; P/8 at scale | P ranks against P/k, bit for bit, by index (§5.1) |
| S4 key routing | T3a | `nd-octree` unit tests of the histogram rounds and the routing; the MPI examples | unit; 1–8 | the Phase 5 octree, key by key (leaves, blocks, every key, `KeyType`) |
| the partition map and H | T3 | `nd-octree` unit tests (owners and kinds at block boundaries, above the blocks, the root, empty ranks); a new registered MPI example `test_mpi_locally_essential` | unit; 1, 2, 3, 4, 8; CI at 2, 4 | brute force on the gathered tree at small N: every key a row of S, a halo or a parent-child step names is in H, H ⊆ the closure of §3.1; the leaves and the partition unchanged; held-set sizes printed old against new |
| the plan over H | T4 | `nd-fmm-plan` `tests/mpi_regressions.rs` (scenarios unchanged) | 1, 2, 4 (CI), 8 by hand | the list oracle over S (pruned rows no longer exist); every listed key held; the exchange checks; the recording operator: every pair once, the same calls and groupings; a minimality check (nothing held outside H ∪ T) |
| the owned top tree | T5 | `mpi_regressions`; `overlap_stress` | 1, 2, 4 (CI); 8 | every top-tree multipole bit for bit against the one-rank tree's by key (a per-pair test operator whose values mix, as `Mixer` does, so a wrong order shows); the reduction's messages sent equal received over the ranks; every needed multipole received once; R the same on every rank; overlap bit for bit |
| `Fmm` and the device | T6 | `mpi_exec` (every scenario, the shadow check, the device scenarios on the CPU runtime), `multi_rank` (ignored), `device_ranks` | 1, 2, 4 (CI); 8; gate 2, 4, 8 | the one-rank `Fmm` in rank order, 0 values differing; the shadow bit for bit with the new events, different without them; every device transfer equal to the extended formula (§4.5) |
| the second cut | T7 | `nd-octree` example (leaves unchanged against the first cut); `mpi_exec` with the cut on; T2's invariance | 1, 2, 4 (CI); 8 | key by key; bit for bit |
| S5 | T8 | `mpi_regressions` (the redistribution's point-to-point forward and backward against the all-to-all-v, bit for bit) | 1, 2, 4 | the blocking path |

### 9.2 CI and what runs by hand

`run-tests-mpi` (decision 5) keeps 2 and 4 ranks and its budgets (15 minutes per run
step, 40 for the job): it adds T2's
invariance executable (about a minute at 4 ranks in debug at N ≈ 10⁴, to be measured by
T2), T3's example and nothing else; T4–T7 extend existing executables. By hand: 8 ranks
on the M3 Max and locust; the ignored gates (`multi_rank`, `overlap_stress`,
`device_ranks` on CUDA and Metal); oversubscribed counts on locust (to 512) and Kathleen
(to 640; 1,280 did not start, §10.1); Kathleen's correctness job of T2 at 2 nodes in
`test` (the MPI list at 80 ranks and 2 × 2 × 20, the invariance at 80 against 40, 20
and 10).

A test that needs a rank count checks `comm.size()` on every rank before any collective
and prints "skipped at P ranks" on every rank (distributed §10.4); the invariance test
needs no skip (k runs over the divisors of P, k = 1 included).

## 10. Scaling method and the time model

### 10.1 Measured runs (T9, and each task's own)

- **Strong scaling**: N = 10⁷ and 10⁸, the cube, the Plummer sphere and the clusters, f64
  p = 3 (Dense under the rule) and p = 8 (Rotation), on Kathleen at 1, 2 and 4 nodes in
  both splits (40 × 1 and 2 × 20 per node, the rule's strategy; Phase 5N decision on
  splits), and on locust at 8, 16, 32, 64 and 72 ranks.
- **Weak scaling**: 10⁶ points per socket at 2 × 20 per node, and 2.5 × 10⁴ points per
  rank at 40 × 1 per node, 1, 2, 4 nodes; locust 8–72 ranks at 2.5 × 10⁴ per rank.
- **Repetitions and statistics**: as T0 (one warm-up, ten evaluations per path, the
  median per rank of the max over ranks; N = 10⁸ two evaluations); repeats of four
  configurations on disjoint node sets (`--exclude`).
- **Per stage and rank**: the forward of the charges, the source exchange, `upward_local`,
  **the reduction (per level) and the all-gather of R** in place of the coarse gather,
  the multipole exchange, `downward`, the leaves, the output, the backward; exposed waits
  on the overlapped path; the build by part (domain, octree with its steps, the second
  cut, plan, redistribution, tables, evaluator with the exchange builds); held boxes
  (own, halo, shared ancestors, top, R); bytes and messages per exchange; memory by
  formula and the peak resident size.
- **Phase 5 against Phase 5S**: the same launches on the merged Phase 5N code (`7acb6fb`)
  and on each task's merge, stage by stage.
- **Oversubscribed counts**: 128, 256, 512 ranks on locust; 320 and 640 on Kathleen's 4
  nodes, N = 10⁶–10⁷ and 2.5 × 10⁴ per rank: held boxes, traffic, R, the reduction,
  memory. **1,280 ranks (320 per node) did not start on Kathleen** in this task: over TCP
  (`--mca pml ob1 --mca btl self,vader,tcp`) already 640 failed with "bind … Address
  already in use" (the node's ports run out, job 252223); over PSM2 with shared contexts
  (`-x PSM2_SHAREDCONTEXTS=1`) 640 runs, but at 1,280 every launch died within 6 s (exit
  139), and a node whose contexts the crash left busy ("assign_context command failed:
  Device or resource busy") failed every later launch of the job (job 252846; rerun
  without that node in 255188). The briefs' 1,280 on Kathleen (T2–T9) became 640, and the
  P/8 invariance at scale 640 against 80 (amended at the user's request; §12, question 11).

### 10.2 The time model beyond 160 ranks (*model*)

No job is larger than 4 nodes, so every time beyond 160 ranks is a *model*
(distributed §14.5), its inputs stated with every figure:

> T_eval(P) = max over ranks of the modelled compute (Σ_k α_k c_k, §6.1, from the counts
> per rank of an oversubscribed run at that P, or the per-rank size of a large run timed
> on 1–4 nodes) + Σ over exchanges of (messages × α_e + bytes / β_e) on the busiest rank
> + r rounds × α_e (the reduction) + the all-gather of R (log₂ P α_e + |R| n_c s / β_g)
> + the redistribution's two all-to-all-v's (2 P α_v until S5, 2 m α_e after).

Inputs, measured on Kathleen (§10.3) and stated: α_e the per-message cost, β_e the
per-rank bandwidth when every rank of a node exchanges (T0: 0.4 GB/s for the source
exchange, 0.2 GB/s for the multipole exchange at 160 ranks), β_g the all-gather's
effective rate, α_v the all-to-all-v's cost per peer; and the alternatives 1–2 µs and
10–100 Gbit/s (README) beside them. The measured imbalance enters through the modelled
compute's max over the ranks (§6.2's replay at 512–640 ranks). Effects the model
misses are listed as open: contention between the 40 ranks of a node on one Omni-Path
link, the 80 ms stalls T0 measured at 4 × 40 ranks (S0.6, a transport timeout,
inferred), collective algorithms at thousands of ranks, and the many small messages of
the reduction and the multipole exchange.

### 10.3 Network inputs (Kathleen)

The probe's micro-benchmarks (job 252846, node-c11a-[011,056,076,082], Open MPI 4.1.6,
PSM2 with `PSM2_TID=0`, release, the median of 10–20 repetitions, max over ranks). Rank
r is paired with r + P/2, so with 40 ranks per node every pair crosses the network at 80
and 160 ranks and stays on the node at 40:

| test | 4 ranks, 1 per node | 40, one node | 80, 2 nodes | 160, 4 nodes |
| --- | ---: | ---: | ---: | ---: |
| ping-pong, half round trip, 8 B (µs) | 1.67 | 1.47 | 1.43 | 1.48 |
| ping-pong, 1 MB (GB/s) | 9.88 | 3.44 | 9.12 | 9.20 |
| every pair at once, 1 MB, per rank (GB/s) | 6.82 | 3.16 | 0.28 | **0.27** |
| every pair at once, 64 kB, per rank (GB/s) | 3.63 | 3.63 | 0.26 | 0.21 |
| `MPI_Alltoall`, one i32 per peer (µs) | 3.4 | 47.6 | 52.0 | 59.8 |
| `MPI_Allgather`, 0.37 MB in all (ms; GB/s) | 0.05; 6.8 | 0.28; 1.34 | 0.24; 1.56 | 0.25; 1.49 |
| `MPI_Allgather`, 2.58 MB in all (ms; GB/s) | 0.35; 7.4 | 1.79; 1.44 | 1.65; 1.56 | 1.95; 1.32 |
| `MPI_Allgather`, 11.6 MB in all (ms; GB/s) | 1.64; 7.1 | 6.43; 1.80 | 6.82; 1.70 | 6.76; 1.71 |
| `MPI_Barrier` (µs) | 3.1 | 8.8 | 11.5 | 31.3 |

- **Latency α_e ≈ 1.5 µs** between nodes, as the README assumed (1–2 µs).
- **One pair alone moves 9.2 GB/s** (Omni-Path's 12.5 GB/s less overheads); **40 pairs of
  a node at once share the link: 0.27 GB/s per rank** (10.9 GB/s per node). That is the
  rate T0's exchanges measured (0.21–0.83 GB/s, S0.7), so the model uses **β_e = 0.27 GB/s
  per rank** when every rank of a node exchanges across the network, not the link rate.
- **The all-gathers run at β_g ≈ 1.3–1.7 GB/s** of their total size, nearly independent of
  P from 40 to 160 ranks: R's 2.65 MB (the cap) takes about 2 ms.

### 10.4 The model at 10³–10⁴ ranks

Weak scaling at 2.5 × 10⁴ points per rank, f64 p = 8, Rotation, 40 ranks per node, the
blocking path (*model*, inputs below):

| Term (ms per evaluation) | cube, Phase 5: 1,024 / 4,096 / 8,192 | cube, Phase 5S | Plummer, Phase 5: 1,024 / 4,096 / 8,192 | Plummer, Phase 5S |
| --- | --- | --- | --- | --- |
| compute, busiest rank | 576 / 576 / 576 | 537 | 1,863 / 1,863 / 1,863 | 1,194 |
| global M2M (`upward_global`) | 115 / 460 / 921 | ≤ 0.3 | 91 / 363 / 727 | ≤ 0.3 |
| coarse gather / R's all-gather | 16 / 64 / 127 | 2 | 13 / 50 / 100 | 2 |
| source and multipole exchanges | 10 | 10 | 16 | 16 |
| S2's reduction | – | 0.1 | – | 0.1 |
| the redistribution's all-to-all-v pair | 0.8 / 3.0 / 6.1 | 0 (S5) | 0.8 / 3.0 / 6.1 | 0 (S5) |
| **total** | **718 / 1,113 / 1,640** | **549** | **1,984 / 2,295 / 2,712** | **1,212** |
| coefficient slots per rank (MB) | 105 / 420 / 840 | ≤ 17 | 83 / 330 / 660 | ≤ 17 |

Inputs, each stated:
- **compute**: the mean measured compute per point at 62,500 points per rank on 160
  Kathleen ranks (T0, S0.7: 1,209 ms cube, 2,858 ms Plummer; 19.3 and 45.7 µs per point)
  times 2.5 × 10⁴, times the max/mean of the modelled cost per rank: Phase 5's cut by
  points 1.19 (cube) and 1.63 (Plummer), S3's cut by the model 1.11 and 1.045 (§6.2, the
  512-rank replay, held fixed above 512: an assumption, not a measurement);
- **global M2M**: 8 × G M2M per rank at α_M2M = 2.73 µs (§6.1, Kathleen), with B = 36 P
  (cube) and 28.4 P (Plummer) blocks and G = B/7; after S2 at most 88 owned M2M (§4.4);
- **gathers**: B n_c s at β_g = 1.5 GB/s; R at the cap, 2.65 MB;
- **exchanges**: T0's bytes per rank at 160 ranks scaled to 2.5 × 10⁴ points by the
  surface ((0.4)^(2/3)): 2.65 MB (cube) and 3.82 MB (Plummer) at β_e = 0.27 GB/s, plus
  150 and 415 messages at an effective 4 µs each (T0: 342 messages in 1.43 ms);
- **redistribution**: 2 P α_v, α_v = 0.37 µs (§7.2), with the input on the owners;
- **memory**: Phase 5's held keys about 2.2 B at 512 ranks (measured: 39,865 for 18,418
  blocks) at 2 n_c s = 1,296 B each; Phase 5S: H ≤ 9,177 plus R ≤ 4,096 (§1.3, §4.3).

Not in the model: contention beyond one link per node, T0's 80 ms stalls at 4 × 40 ranks,
collective algorithms at thousands of ranks, and the extra neighbours S2 adds above 640
ranks (none at 320–512, 1–13 at 640, §4.3). **At 8,192 ranks the model puts Phase 5's
replicated top at about 0.83–1.05 s of every evaluation (the global M2M and the gather),
and Phase 5S's top-tree terms at about 2 ms.**

### 10.5 Job plan and node-hours (decision 6)

T1 used **10.4 node-hours** on Kathleen of its 60 (seven jobs, Appendix A; 10.3 in four
4-node jobs) and locust by hand. Proposed per task, within 4 nodes per job and T0's
measured waits (median 12 s for 4 nodes in `small` on a Sunday; 27 and 60 minutes when the
machine was full or the fair share low):

| Task | Kathleen runs | Node-hours (budget) |
| --- | --- | ---: |
| T2 | correctness job at 2 nodes; invariance at 160/20 and 640/80; the harness at N = 10⁸ | 10 (20) |
| T3 | held keys 40–160 timed, 320–640 oversubscribed; `Octree::new` before and after; the correctness job | 8 (20) |
| T4 | as T3 for the plan; evaluation at 80 and 160 | 8 (20) |
| T5 | reduction counts to 640; the correctness job | 6 (20) |
| T6 | Phase 5 against 5S at 40, 80, 160; counts to 640 | 12 (20) |
| T7 | first against second cut at 40, 80, 160 (both splits, N = 10⁷, 10⁸); counts to 640 | 15 (20) |
| T8 | counts and the timed build at 160; models | 8 (20) |
| T9 | the scaling report: strong and weak at 1, 2, 4 nodes, Phase 5 against 5S, three workloads, both splits, repeats | 50 (60) |
| **phase** | | **about 120 of 240** |

T9's plan: one 4-node job per sweep (strong N = 10⁷, strong N = 10⁸, weak, checks,
oversubscribed counts), each sized from T0's (14, 21, 7, 6 min) times the three
workloads and both codes, at most two jobs pending at once.

**Without Kathleen** (README): the timed runs are locust's to 72 ranks and the counts to
512; the 4-node runs become "when the queue allows"; acceptance on the counts and the
model (§10.2) with α_e and β_e at the stated alternatives.

## 11. Task check

| Task | Modules | Tests | Brief change proposed |
| --- | --- | --- | --- |
| T2 | `fmm-exec/tests/invariance.rs` (new, registered in `.github/scripts/run-mpi-tests.sh`); `nd_fmm_validate::scaling` (`--invariance k`, `--reference` refused above 2 × 10⁷, an input of index ranges for N > 10⁹); `tools/kathleen/jobs/` (the correctness job) | §9.1 | (a) **the invariance check names the strategy** (`Auto` resolves by ranks per node, §5.1); (b) its own executable rather than an `mpi_exec` scenario, for `mpi_exec`'s one-minute debug budget |
| T1s (scheduled 2026-10-11) | `Cargo.toml` pins (rlst 0.10.0, CubeCL 0.11.0), `octree/Cargo.toml`, the ports, the live docs | the full checks, host bit for bit, device within bounds | **new brief** docs/phase5s/T1s-dependency-upgrade.md, added at the user's request; before T2 |
| T3a (scheduled 2026-10-11) | `octree/src/octree/implementation.rs` (`linearize`, `balance`: S4's routing), the MPI examples | §9.1 | **new brief** docs/phase5s/T3a-octree-key-routing.md, added at the user's request; T8 keeps S5 |
| T3 | `octree/src/octree.rs` (`PartitionMap`, `Octree::partition_map`, `KeyType` and `with_ghost_children` docs), `implementation.rs` (`generate_all_keys` over H, the leaf-bit all-gather, the push); `examples/test_mpi_locally_essential.rs` | §9.1 | (a) the map's owner of a key above the blocks is S2's (first descendant block); (b) the old path is removed, not kept behind an option; the brute-force example is the comparison |
| T4 | `fmm-plan/src/plan.rs` (H ∪ T, no list resolution for non-shared `Global` keys), `exchange.rs` (`MultipoleExchange` over ghosts incl. top-tree ghosts), `nd_fmm_validate::scaling::held_boxes` | §9.1 | **acceptance**: held boxes bounded on H; H ∪ T reported, T removed in T5 (§3.7, question 1) |
| T5 | `fmm-plan/src/exchange.rs` (the reduction's exchanges, R; `CoarseExchange` removed), `evaluator.rs` (`upward_global`, `far_field`), `operator.rs` (three events), `plan.rs` (owned `m2m_global` rows, T ∖ (H ∪ R) dropped) | §9.1 | (a) **R by readers with a cap, not a level L_r** (§4.3, question 3); (b) the reduction stops at level 2; (c) `SendMultipoles` moves after the all-gather of R |
| T6 | `fmm-exec/src/fmm.rs` (`StageTimings`: `reduction`, `replicated_top`; `exchange_traffic`), `device.rs` (three events, the formula, 2 + r_s syncs), `tests/device_common`, `nd_fmm_validate::scaling` | §9.1 | the sync count changes from 2 to 2 + r_s (§4.5) |
| T7 | `nd-octree` (`Octree::repartition`), `fmm-exec/src/fmm.rs` (counts per block, the cost table, `FmmBuilder::cost_model`, the second cut), a calibration example | §9.1 | (a) **the weights come from the plan, not from the local refinement** (§6.3); (b) the target 1.10 needs blocks refined by cost (question 5) or reads 1.12 |
| T8 | `fmm-plan/src/redistribute.rs` (point-to-point forward and backward); NBX only if its trigger fires | §9.1 | S4 moved to T3a; T8 keeps S5 and re-measures S4 |
| T9 | `nd_fmm_validate`, results, design documents, `tools/scaling/run.sh` (the clusters in `nodes-strong`/`nodes-large`) | as the brief | the clusters added to the sweeps (T0 ran the cube and the Plummer sphere) |

**Facts in the briefs that this design found to differ** (not changed in the briefs):
- README, T3, T5: "the replicated top above L_r … a few thousand boxes, fixed": a top by
  level does not bound the owners' fan-out on adaptive trees (335 readers of one box at
  512 ranks on the Plummer sphere, §4.3); the design replicates by readers.
- T7: "a per-block cost from counts each rank has after the local refinement … boxes
  per level (M2M, L2L) and V-row lengths": the W and X kinds are 22–35% of the compute on
  the Plummer sphere (§6.1), and a local proxy reaches only 1.15–1.22 at 512 ranks
  against 1.05–1.11 for the plan's counts (§6.2).
- T8: "S4 … only if measured … to need it at 1,000 or more ranks": it is needed at 8 ranks
  already, on the random share (§1.4, §7.1).
- T0 S0.5 ("not traced here … a transient of the construction is the likely place"): it
  is the first sample sort (§1.4).

## 12. Questions for sign-off

**Signed off on 2026-10-11 by the user: every recommendation below is accepted, questions
1–11.** docs/phase5s/README.md records the answers as decisions 0–6.

| # | Question | Recommendation |
| --- | --- | --- |
| 1 | The requirements (decision 1) | Accept 1–10 with: requirement 3 — the invariance check names the strategy; requirement 4 — T4 meets the bound on H and T5/T6 on the total (the top-tree keys held until T5, §3.7); requirement 5 — "bounded" for the owners' fan-out below R means at most max(2, P/16) readers while |R| ≤ R_max, with the readers of the R_max-th box reported beyond (§4.3) |
| 2 | The held set of S1, `KeyType::Global`, the ghost-children layer | H = N(S) ∪ C(N(S)) (§3.1), assembled from the partition map and one push (§3.3); the map (§3.2: blocks, cut, leaf bits) public as `PartitionMap`; `Global` kept with the meaning "shared ancestor"; `with_ghost_children` kept, without block replication; the old path removed |
| 3 | The replicated top (decision 2) | Not a level L_r: **R = the top-tree boxes read by at least max(2, ⌈P/16⌉) ranks, at most 4,096 of them** (≤ 2.65 MB at p = 8 in f64), chosen at build, replicated by one all-gather-v per evaluation; below R, the multipole exchange (§4.3) |
| 4 | The new events and the device's transfers | `SendReduction`, `ReceivedReduction`, `ReceivedTop`; `ReceivedCoarse` removed; `SendMultipoles` after R; the device's syncs 2 + r_s (§4.5) |
| 5 | The work weight, target, default (decision 3) | The plan's counts per block with per-kind constants per strategy, precision and p (§6.1, §6.3), a second cut by `Octree::repartition` after `Plan::new`; target ≤ 1.10 modelled at 512 (locust) and 640 (Kathleen) oversubscribed ranks **with blocks refined by cost** (a coarse-tree change, bits unchanged) — or, without that refinement, ≤ 1.12; default on, on several ranks |
| 6 | The S4 and S5 triggers (decision 4) | S4 has fired (§7.1): its key routing is scheduled as **T3a** before T3 (brief added at the user's request); rlst PR #40 fixes `parsort`'s splitters upstream (rlst 0.10.0), adopted by T1s before T2; S5's evaluation part (point-to-point redistribution on the build's routes) fires on the model and is built in T8; its build part (non-blocking consensus) only if the count all-to-alls exceed a quarter of a modelled build at 10⁴ ranks; no rsmpi addition needed |
| 7 | CI and what runs by hand (decision 5) | `run-tests-mpi` at 2 and 4 ranks adds T2's invariance executable and T3's example; everything else by hand (§9.2) |
| 8 | The node-hour budget and T9's plan (decision 6) | About 120 node-hours for the phase within 240 budgeted (§10.5); T9 one 4-node job per sweep |
| 9 | The time model beyond 160 ranks | §10.2, with Kathleen's measured α_e, β_e and β_g (§10.3) and the README's 1–2 µs and 10–100 Gbit/s beside them; every figure *model*, the open effects listed |
| 10 | (new) Decision 0, 4 nodes in practice | The phase runs at 4 nodes as planned (T0's S0.12): of T1's four 4-node jobs two waited 2 s, one 33 minutes (Sunday evening), and one, submitted 20 s after that one, started when it ended; 10.4 node-hours in all (Appendix A) |
| 11 | (new) Oversubscription on Kathleen | **640 ranks (160 per node) is Kathleen's ceiling**, over PSM2 with shared contexts; 1,280 did not start (§10.1). The briefs' 1,280 on Kathleen (T2–T9) amended to 640 and the invariance at scale to 640 against 80 (done at the user's request); locust's 512 remains the largest count elsewhere |

## Appendix A. How the numbers were measured

**The probe** (throwaway, in the session's scratch directory, not committed, as the
brief allows): one Rust binary with path dependencies on this tree's `nd-octree`,
`nd-fmm-plan`, `nd-fmm-exec` and `nd-fmm-validate` at `7fa691e` (main after T0, #97),
built in release on each machine. For a `scaling` workload (`Workload::new(dist, n)`,
the random share, the domain by `compute_global_bounding_box`, the keys of sources and
targets as `Fmm` passes them, `max_level` 16, 64 points per leaf, the ghost-children
layer) it:
1. replays `Octree::new` step by step with the private functions of
   octree/src/octree/implementation.rs copied, a barrier before each step, the time, the
   bytes the step's data collectives move, its O(P) count bytes, and the peak resident
   size during the step (`VmHWM`, reset through `/proc/self/clear_refs` before each
   step); the replay's leaves, blocks and key map equal `Octree::new`'s on every run (a
   check the probe prints);
2. builds the real `Octree` and `Plan` (timed) and computes from them, per rank: the held
   keys by kind; S, the keys S's rows name, H (§3.1); the top-tree multipoles S's V and W
   rows read and do not own, per level, with their owners (the partition map from the
   replay); the readers of each (the requests the owners receive through an rlst ghost
   build per level); R by readers for K = 16, 64, 256 and the levels L_r = 1–5; S2's
   reduction per level;
3. with `--fmm`: builds `Fmm` (`KindTiming::Synchronous`, the rule's strategy with
   Kathleen's and locust's budget of 96 MB), evaluates 1 + 3 times, and fits the
   per-kind constants of §6.1 to the per-rank kind times and the plan's counts (the
   ghost leaves' source counts by an rlst ghost exchange), then replays the cut of §6.2;
   on oversubscribed runs with constants passed in (`--alpha`, from the timed runs);
4. with `--net`: the micro-benchmarks of §10.3.
Rank 0 prints the max, mean and min over the ranks of every quantity. Raw outputs are
summarised here and kept only in the machines' log directories.

**Machines and placement.** locust: 72 Neoverse-V2 cores, one node, Open MPI 5.0.10,
`mpirun --map-by core --bind-to core` up to 72 ranks, `--map-by :OVERSUBSCRIBE
--bind-to none` above (counts only); the load checked before and after each launch (0.0
before the timed runs, 2026-10-11). Kathleen: `tools/kathleen/jobs/run.sbatch` steps,
Open MPI 4.1.6, `PSM2_TID=0` (T0, S0.1), `mpirun --map-by ppr:<k>:socket:PE=<t>
--bind-to core` up to 40 ranks per node, every launch under `timeout`; oversubscribed
launches with `--oversubscribe --map-by ppr:<k>:node --bind-to none`, counts only: over
TCP (`--mca pml ob1 --mca btl self,vader,tcp`) in job 252223, which ran 320 ranks, then
over PSM2 with `-x PSM2_SHAREDCONTEXTS=1` in jobs 252846 and 255188, which ran 320 and
640; 1,280 did not start either way (§10.1).

**Jobs** (Kathleen, 2026-10-11, `sacct`; node-hours are nodes × elapsed):

| Job | Nodes | What | Waited | Ran | Node-hours |
| --- | ---: | --- | ---: | ---: | ---: |
| 252208 | 1 | build of the tree and the probe | 1 s | 1:49 | 0.03 |
| 252220 | 4 | timed 40, 80, 160 and 2 × 20 (N = 10⁷), N = 10⁸ at 40 and 160 | 33:21 | 8:46 | 0.58 |
| 252223 | 4 | weak 40–1,280 over TCP: 320 ran (cube, Plummer), 640 and 1,280 failed | 41:47 (behind 252220) | 2:37 | 0.17 |
| 252225, 252226 | 1 | builds (the probe against rlst PR #40) | 1 s | 1:01 | 0.02 |
| 252846 | 4 | rlst #40 against 0.9.0, network, the clusters at 4 nodes (2 × 20), 320–1,280 over PSM2; after the 1,280 crash the later launches failed (§10.1) | 2 s | 38:44 | 2.58 |
| 255188 | 4 | the clusters at 1, 2, 4 nodes in both splits; weak and FMM counts at 320 and 640 (`--exclude=node-c11a-011`) | 2 s | 1:45:09 | 7.01 |
| **total** | | | | | **10.4** |

Nodes of 255188: node-c11a-[082,109,131,134].

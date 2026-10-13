# The owned top tree in `nd-fmm-plan` (Phase 5S T5)

S2 in `nd-fmm-plan`, as docs/phase5s/T5-owned-top-tree.md asks and
docs/design/scale-out.md §4 (signed off) specifies: every `Global` box (shared ancestor)
has an owner, the rank of its first descendant block; an upward reduction sends exact
child multipoles to the parents' owners, which form them by octant as one rank does; a
replicated top R of the most-read top-tree boxes is all-gathered once per evaluation;
below it the multipole exchange carries what the rows read; the coarse gather and the
replicated global pass are gone, and the plan holds H alone. Run on 2026-10-12/13 on the
M3 Max, locust (GH200) and Kathleen.

**Measured** means run here and read from the output; timings are reported, never asserted.
locust is a shared node: its load is in each output file before and after every launch.
Kathleen nodes are exclusive; every job is named.

## Summary

- **Every top-tree multipole is the one-rank value, bit for bit.** A new check of
  `mpi_regressions` evaluates an order-sensitive mixing operator on the plan and on the
  one-rank plan of the same tree and compares, by key, every multipole a rank computes or
  receives (the shared ancestors' among them, the root included), every local it holds and
  every output: 0 differences on the 12 scenarios (the graded one included), the cube and
  the Plummer sphere at 1, 2, 3, 4 and 8 ranks (M3 Max), 1, 2, 4, 8 (locust) and 80
  (Kathleen). A mutant that runs each global M2M before the reduction of its children
  fails it on every rank.
- **The host output is bit for bit Phase 5's**: `multi_rank` 0 values differing in every
  row at 1, 2, 4 and 8 ranks (M3 Max and locust); T2's invariance check 0 values
  differing at 1, 2, 4 and 8 ranks, at 512 against 64 on locust, and on Kathleen at 80
  against 40, 20 and 10, 160 against 80, 40 and 20 (three workloads each) and 640 against
  80; the `mpi_exec` suites, the shadow check included, pass unchanged with the host
  operator at 1, 2, 4 and 8 ranks and at 80 on Kathleen.
- **The global M2M per rank** falls from Phase 5's 8 × (every `Global` box) to 8 × the
  owned ones: at 640 Kathleen ranks 37,448 → 88 (cube), 20,528 → 88 (Plummer), 22,216 →
  456 (clusters), at 512 locust ranks 21,048 → 72, 16,672 → 96 and 19,744 → 592; the
  coarse gather, 21.2, 11.6 and 12.6 MB per rank and evaluation at 640 (11.9, 9.5 and
  11.2 MB at 512), is gone.
- **The reduction** sends at most 3–7 messages per rank and evaluation over every level
  and receives at most 7 per level (13–35 over all levels at 640 ranks), in 5–8 rounds,
  at most 31 kB per rank.
- **The replicated top** is 1.49 MB (cube), 2.32 MB (Plummer) and 1.82 MB (clusters)
  per rank and evaluation at 640 ranks, at most 2.63 MB anywhere (the design's bound at
  4,096 boxes is 2.65 MB); a rank writes the 120–1,167 of its boxes it holds.
- **The held boxes per rank are H**, exactly (checked by brute force in
  `mpi_regressions`): at 640 Kathleen ranks 42,417 → 6,273 (cube), 27,809 → 9,177
  (Plummer), 28,465 → 7,825 (clusters) against T4's H ∪ T, T4's H to the key.
- **At fixed N/P** (2.5 × 10⁴ points per rank, to 640 Kathleen and 512 locust ranks) the
  global pass, the reduction, R, the held boxes and every exchange's bytes stay bounded
  (§4). The multipole exchange's messages and neighbours still grow with P (14 → 106
  neighbours on the clusters from 40 to 640 ranks), much less than Phase 5's messages
  (1,679 → 316 at 640): a top-tree box below R goes to fewer ranks than R's threshold,
  max(2, ⌈P/16⌉) or a higher power of two where the 4,096-box cap binds (it does on the
  clusters and the Plummer sphere), the signed-off reading of requirement 5 (scale-out.md
  §12, question 1), not a constant. The growth is the boxes about a rank's size leaving R
  as the threshold rises (§4, "Where the neighbours come from"); most messages stay on
  the node (§4, "By node"). A trigger for it (S8) is drafted into T8's brief.

## 1. The change

`fmm-plan/src/plan.rs`, `exchange.rs`, `evaluator.rs`, `operator.rs`, `lib.rs`,
`index.rs` and their tests; `fmm-plan/tests/mpi_regressions.rs`; the API users in
`nd-fmm-exec` (`fmm.rs`, `device.rs`, `tests/shadow`, `tests/device_common`,
`tests/mpi_exec.rs`, `tests/device_ranks.rs`, `examples/overlap_timing.rs`) and
`nd-fmm-validate` (`scaling.rs`, `tests/scaling.rs`); the crates' CLAUDE.md files.

1. **The plan holds H** (scale-out.md §3.7: T5 drops T). `Plan::new` takes the octree's
   `all_keys` as they are; T4's top-tree keys outside H are gone, and so a top-tree key
   that is not a shared ancestor is a ghost of its owner, as the octree classifies it.
   R's boxes that no row of the rank names get no slot (§2, reading 3).
2. **Owners** (§4.1). From the replicated `PartitionMap`, locally, the owner of every
   held top-tree key (`Plan::owner`, `computes`, `is_top`, `top_boxes`,
   `deepest_block_level`, `reduction_levels`). A `Global` box keeps its M2M row of the
   global pass on its owner only. `Plan::from_key_types` takes a `PartitionMap` instead of
   the own blocks.
3. **The upward reduction** (§4.2; `ReductionExchange`): one rlst ghost exchange per
   level 1 to D, D the deepest block level (the same on every rank, from the map). On the
   owner of a `Global` box the children other ranks compute are the ghosts of their
   level, from those ranks. `upward_global` runs, for l = L − 1 down to 0, the reduction
   of level l + 1 if it is one and then the global `m2m(l)`: every owned `Global` box is
   the M2M of its eight exact children by octant, the one-rank sum. Never a reduction of
   partial multipoles.
4. **The replicated top** (§4.3; `TopExchange`): R = the top-tree boxes with at least
   max(2, ⌈P/16⌉) readers, at most 4,096 (`TOP_READER_FRACTION`, `TOP_MAX_BOXES`). At
   build one rlst build over the top-tree boxes every rank reads but neither computes nor
   receives through the reduction gives every owner its readers; one all-reduce carries
   the error flag and the boxes at or above each candidate threshold (max(2, ⌈P/16⌉),
   then powers of two); an all-gather of P counts and an all-gather-v give R to every rank
   in rank order. Per evaluation one all-gather-v of R's multipoles, which writes the
   boxes of R a rank holds but neither computes nor receives through the reduction.
5. **The multipole exchange** carries the boxes of V- and W-rows that other ranks
   compute (ghosts and, new, the `Global` boxes other ranks own), without what the
   reduction and R deliver: every multipole a row reads reaches a rank once.
6. **The coarse gather is removed** (`CoarseExchange`, `Evaluator::coarse_exchange`,
   `HostData::ReceivedCoarse`), with T4's held top-tree keys.
7. **Events** (§4.5): `SendReduction`/`ReceivedReduction` per reduction level, deepest
   first, before the global M2M that reads it; `SendMultipoles` (the top's sent keys and
   every level's multipole sends) after the reduction and before the all-gather;
   `ReceivedTop` after it. 5 + nlevels + 2 D events per evaluation, D = 0 on one rank. The
   host operator ignores them, so the host path's bits do not change; the shadow operator
   of `nd-fmm-exec`'s tests mirrors them and stays bit for bit the plain operator.
8. **Overlap** (§4.6): `far_field` posts every reduction level's receives first, then
   for each level, deepest first, `SendReduction`, posts its sends (packed from the store,
   final by then), waits, `ReceivedReduction`, and the global M2M, a `test` after each;
   then `SendMultipoles`, the multipole exchange posted, the all-gather of R (blocking,
   as Phase 5 kept the gather) and `ReceivedTop`, and the downward pass as before.
   `ReductionExchange::forward_all_overlapped` and `ReductionInFlight` (`send`, `wait`,
   `test`) are the mechanism; `OverlapTimes` has `reduction` and `top_gather` (the
   coarse gather's field renamed).
9. **`nd-fmm-exec`** follows the API: `ExchangeTraffic` has `reduction` per level,
   `top_sent`, `top_received` and `multipole_neighbours`; `ExchangeSizes` `sent_top`,
   `received_top`; `device::ExchangeLists` the top's boxes and `DataKind::TopMultipoles`.
   The device does not mirror the reduction before T6: with a box to reduce the device
   operator fails with a device error (agreed on every rank), and the device scenarios on
   several ranks are marked as T6's (§3.3).
10. **`scaling`** reports the new structures per rank (`TopTreeRow`): the global M2M
    pairs of the owned boxes, the reduction summed and per level, the multipole exchange's
    neighbours, the replicated top's bytes and boxes, and Phase 5's global pass
    (8 (B − 1)/7 pairs for B blocks) and coarse gather on the same tree; the summary line
    carries their maxima.

## 2. Readings and departures from the design (for review)

1. **The reduction runs to level 1, forming the root**, not stopping at level 2 (§4.2,
   §11 (b)). So every `Global` box, the root included, is formed on its owner and has
   its one-rank bits, which the brief's test asks for ("every shared-ancestor
   multipole"), and one rank keeps forming its root. It costs at most two more rounds of
   at most 7 messages of one multipole each (levels 1 and 2), counted in §5.
2. **`SendMultipoles` fires before the all-gather of R**, not after it (§4.5, §11 (c)):
   the all-gather sends the rank's own boxes of R, so their "send" event must come first
   for the hook's coverage (an operator writes the values to send at a send event). The
   event count is the design's, 5 + nlevels + 2 r.
3. **R holds no slot outside H**: a box of R no row of a rank names stays in the gathered
   buffer only (the design counted R ∖ H among the held boxes, §8.2). The held boxes are
   then H exactly.
4. **Readers**: the ranks that need a box from its owner through the multipole exchange,
   so neither its owner nor a rank that receives it through the reduction. Threshold
   candidates: max(2, ⌈P/16⌉), then powers of two up to P, the first under the cap (one
   all-reduce of their counts, the design's "histogram of log₂ P bins").
5. **What the reduction delivers is not sent again** by the multipole exchange or written
   by R, so every multipole a rank reads arrives once (the brief's "every needed
   multipole received exactly once").
6. **|R| is larger than T1 measured on the Plummer sphere**, and at 320 Kathleen ranks on
   all three workloads: 3,584 (Plummer) and 2,810 (clusters) boxes at 640 Kathleen ranks
   against T1's 2,263–2,327, 2,319–4,055 at 320 against 576–1,636, 3,759 (Plummer) at 512
   locust ranks against 1,298–2,144 (the cube's 2,026 and the clusters' 1,988 inside it);
   on the cube at 640 it is T1's (≈ 2,301 against 2,304 boxes, 1.49 MB, 88 owned M2M
   pairs, all three as §4.4 states). T1's probe was not committed, so the
   difference is not traced; this R is checked by brute force (`mpi_regressions`,
   `check_top`: every rank's reads gathered and counted) and stays within the cap (at most
   2.63 MB, the bound being 2.65 MB).

## 3. Correctness

### 3.1 Tests

- `mpi_regressions` (scenarios unchanged, plus the cube and the Plummer sphere for the
  one-rank comparison): the held keys are exactly H with the octree's kinds
  (`check_held_keys`, brute force); the global M2M rows are on the owners only
  (`check_plan`); the reduction receives exactly the children of the owned `Global`
  boxes that other ranks compute and sends only what the rank computes, blocking deepest
  first and non-blocking bit for bit (`check_reduction`); R is the same on every rank and
  exactly the top-tree boxes with at least max(2, ⌈P/16⌉) readers counted by brute force,
  in rank order by (level, key), its gather delivering every owner's values and writing
  exactly the boxes it should (`check_top`); the multipole exchange receives exactly the
  V- and W-row boxes other ranks compute minus what the reduction and R deliver, and every
  multipole a list reads is received exactly once over the three
  (`check_multipole_exchange`); `Traffic` sent equals received over the ranks for every
  level of both point-to-point exchanges; the order-sensitive `Mixer` repeats bit for bit,
  blocking and overlapped, and equals the one-rank evaluation of the same tree by key,
  the verdict agreed on every rank (`check_against_one_rank`); the recorder sees every
  oracle pair once, the global M2M pairs on the owners only, the calls and the
  5 + nlevels + 2 D events in the documented order and groupings, blocking and
  overlapped, with the lists of independently built exchanges and of the evaluator's
  accessors. `Mixer` now rotates before it adds, so a contribution in another order
  shows.
- `plan_tests`: the plans of hand-built maps take `PartitionMap`s (`one_rank_map`,
  `map_of`); P2's test checks the global M2M rows on the owner only (the rank of the
  first anchor owns the anchor and its first-child ancestors, the last only the anchor).
- `overlap_stress` (ignored) unchanged: 100 overlapped evaluations bit for bit at 8 ranks.
- `nd-fmm-exec`: the shadow operator follows the new events; `device_runs_here` and
  `device_ranks.rs` mark the device scenarios on several ranks as T6's (§3.3).

### 3.2 Runs

| Where | Ranks | What | Result |
| --- | --- | --- | --- |
| M3 Max | 1, 2, 4, 8 | `run-mpi-tests.sh`: `mpi_regressions`, `mpi_exec`, `mpi_threading`, `invariance`; the seven `nd-octree` examples | pass (44 launches) |
| M3 Max | 1, 2, 3, 4, 8 | `mpi_regressions`, final test | pass |
| M3 Max | 8 | `overlap_stress` (release, ignored) | pass; 0 of 100 evaluations differ |
| M3 Max | 1, 2, 4, 8 | `multi_rank` (release, ignored) | pass; 0 values differ in all 30 rows |
| M3 Max | 1 | `cargo test -p nd-fmm-exec --features cpu --release` | 86 passed |
| locust | 1, 2, 4, 8 | `run-mpi-tests.sh` (44 launches); `mpi_regressions` again with the final test | pass |
| locust | 1, 2, 4, 8 | `multi_rank` | pass; 0 values differ in all 40 rows |
| locust | 8 | `overlap_stress` | pass; 0 of 100 evaluations differ |
| locust | 512 against 64, oversubscribed | `scaling --invariance 8`, the N = 10⁷ cube, f64 p = 8, Rotation | 0 of 4 × 10⁷ values differ |
| Kathleen 530664 | 2 nodes | T2's correctness job | the 2-, 4- and 8-rank launches pass; every 80-rank launch fails in `MPI_Init` on node-c11a-011 (PSM2 cannot open an HFI context, as in T4's job 521270) |
| Kathleen 531957 | 2 nodes, node-c11a-011 excluded | T2's correctness job: the MPI list at 2, 4, 8 and 80 ranks; invariance at 80 against 40, 20, 10 (three workloads); the smoke run | 8 of 8 steps pass; 0 values differ in each invariance check |
| Kathleen 530665 | 160 (4 × 40) | `scaling --invariance 2,4,8` (N = 10⁶): the cube f64 p = 8 and f32 p = 3, the Plummer sphere f64 p = 8 | 0 of 8 × 10⁶ values differ in each of the 9 checks |
| Kathleen 530665 | 640 against 80, oversubscribed | `scaling --invariance 8`, the N = 10⁶ cube, f64 p = 8 | 0 of 8 × 10⁶ values differ |

### 3.3 The device on several ranks: T6's

The brief allows device-backend scenarios to be marked as T6's where the device cannot
follow the new events before T6. It cannot: the device operator would form the owned
`Global` boxes from children it never received. So `DeviceOperator::host_data` fails at
a `SendReduction` or `ReceivedReduction` that moves a box (a device error, agreed on
every rank after the output's download), and these print that they are T6's on every rank
of a run on several ranks instead of running: the device repeat of every
`tests/mpi_exec.rs` scenario (`device_runs_here`), the CPU runtime's threads rule in
**device backends**, **device errors and tuning on several ranks**, and the ignored gate
`tests/device_ranks.rs`. On one rank every one of them runs as before (no reduction
level, R empty): `cargo test -p nd-fmm-exec --features cpu --release` passes on the M3
Max. The device's handling of the new events, its transfers and its formula are T6's.

## 4. Per rank and evaluation, Phase 5 against T5

The `scaling` example, release, 2.5 × 10⁴ points per rank (`--n-per-rank 25000`), the
random share, f64 p = 8 (n_c = 81, 648 B per multipole), Rotation; per rank, max over the
ranks. "Phase 5" is `main` (0ab21a6, T4 merged) for the held boxes, the multipole
exchange and the times, and for the global pass and the coarse gather the T5 run's count
on the same tree (8 (B − 1)/7 pairs; every other rank's blocks), which equals `main`'s
measured coarse-gather bytes in every row. Reduction and R: messages sent / received over
every level; bytes received.

Kathleen (job 530665, node-c11a-[045,092,109,115]; Open MPI 4.1.6 over PSM2,
`PSM2_TID=0`; 40, 80 and 160 ranks as 40 × 1 per node, `--map-by ppr:20:socket:PE=1
--bind-to core`; 320 and 640 oversubscribed on the 4 nodes, `--oversubscribe --map-by
ppr:80:node` / `ppr:160:node --bind-to none -x PSM2_SHAREDCONTEXTS=1`, counted, not
timed):

| Ranks | Workload | held boxes, T4 → T5 | global M2M pairs, P5 → T5 | coarse gather, P5, MB | reduction: messages, kB | R: MB, boxes written | multipole exchange: MB, P5 → T5 | messages sent, P5 → T5 | neighbours |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | cube | 3,881 → 3,585 | 584 → 32 | 0.32 | 2 / 12, 10 | 0.36, 266 | 1.66 → 1.53 | 68 → 44 | 22 |
| 80 | cube | 9,361 → 5,537 | 4,680 → 80 | 2.62 | 3 / 16, 14 | 2.31, 777 | 2.32 → 1.98 | 90 → 57 | 31 |
| 160 | cube | 10,977 → 7,025 | 4,680 → 56 | 2.64 | 3 / 17, 16 | 1.78, 698 | 3.14 → 2.90 | 118 → 72 | 36 |
| 320 | cube | 8,737 → 4,713 | 4,680 → 40 | 2.65 | 3 / 19, 15 | 1.50, 634 | 2.25 → 2.02 | 129 → 84 | 43 |
| 640 | cube | 42,417 → 6,273 | 37,448 → 88 | 21.20 | 3 / 23, 18 | 1.49, 639 | 2.66 → 2.69 | 115 → 139 | 60 |
| 40 | Plummer | 7,521 → 6,569 | 2,200 → 152 | 1.23 | 5 / 10, 21 | 1.40, 1,110 | 2.86 → 2.36 | 198 → 92 | 29 |
| 80 | Plummer | 9,457 → 7,433 | 3,344 → 128 | 1.88 | 5 / 14, 26 | 2.15, 1,285 | 3.34 → 2.75 | 269 → 110 | 38 |
| 160 | Plummer | 11,401 → 7,425 | 5,280 → 112 | 2.99 | 4 / 22, 28 | 2.11, 1,186 | 3.34 → 2.89 | 325 → 105 | 39 |
| 320 | Plummer | 17,921 → 8,889 | 10,744 → 104 | 6.09 | 4 / 31, 31 | 2.63, 1,242 | 3.58 → 3.29 | 495 → 119 | 47 |
| 640 | Plummer | 27,809 → 9,177 | 20,528 → 88 | 11.63 | 4 / 35, 30 | 2.32, 1,167 | 3.85 → 3.72 | 738 → 155 | 64 |
| 40 | clusters | 9,337 → 6,497 | 3,776 → 480 | 2.13 | 7 / 11, 20 | 2.38, 887 | 2.97 → 2.49 | 127 → 39 | 14 |
| 80 | clusters | 10,841 → 6,905 | 5,056 → 536 | 2.86 | 7 / 14, 21 | 2.54, 973 | 3.10 → 2.60 | 241 → 68 | 29 |
| 160 | clusters | 14,489 → 7,569 | 8,184 → 520 | 4.63 | 7 / 14, 25 | 2.24, 889 | 3.30 → 2.98 | 443 → 110 | 40 |
| 320 | clusters | 21,393 → 8,409 | 14,496 → 488 | 8.21 | 7 / 17, 23 | 1.97, 797 | 3.86 → 3.62 | 857 → 196 | 55 |
| 640 | clusters | 28,465 → 7,825 | 22,216 → 456 | 12.59 | 7 / 13, 20 | 1.82, 790 | 3.55 → 3.44 | 1,679 → 316 | 106 |

locust (Open MPI 5.0.10; up to 72 ranks `--map-by slot:PE=1 --bind-to core`, above
`--map-by :OVERSUBSCRIBE --bind-to none`, counted, not timed):

| Ranks | Workload | held boxes, T4 → T5 | global M2M pairs, P5 → T5 | coarse gather, P5, MB | reduction: messages, kB | R: MB, boxes written | multipole exchange: MB, P5 → T5 | messages sent, P5 → T5 | neighbours |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | cube | 3,017 → 3,017 | 72 → 40 | 0.02 | 1 / 1, 3 | 0.00, 0 | 0.44 → 0.44 | 3 → 3 | 1 |
| 4 | cube | 1,961 → 1,961 | 72 → 24 | 0.03 | 1 / 3, 4 | 0.03, 48 | 0.51 → 0.48 | 9 → 6 | 3 |
| 8 | cube | 1,633 → 1,529 | 352 → 72 | 0.18 | 2 / 8, 5 | 0.15, 191 | 0.55 → 0.45 | 26 → 16 | 7 |
| 16 | cube | 5,753 → 5,505 | 584 → 48 | 0.31 | 2 / 8, 7 | 0.32, 279 | 1.96 → 1.81 | 34 → 22 | 11 |
| 32 | cube | 3,817 → 3,521 | 584 → 32 | 0.32 | 1 / 10, 8 | 0.36, 262 | 1.48 → 1.35 | 51 → 34 | 17 |
| 64 | cube | 4,897 → 2,889 | 2,584 → 72 | 1.46 | 3 / 14, 9 | 1.18, 535 | 1.34 → 1.10 | 94 → 57 | 28 |
| 72 | cube | 7,537 → 3,745 | 4,680 → 88 | 2.62 | 3 / 15, 14 | 2.34, 782 | 1.63 → 1.31 | 79 → 46 | 30 |
| 128 | cube | 11,497 → 7,609 | 4,680 → 56 | 2.63 | 3 / 15, 12 | 1.75, 699 | 3.23 → 2.96 | 96 → 62 | 31 |
| 256 | cube | 9,009 → 5,017 | 4,680 → 40 | 2.64 | 3 / 17, 14 | 1.53, 669 | 2.30 → 2.08 | 119 → 78 | 39 |
| 512 | cube | 23,833 → 3,793 | 21,048 → 72 | 11.93 | 3 / 19, 18 | 1.31, 637 | 1.69 → 1.61 | 139 → 123 | 47 |
| 2 | Plummer | 3,209 → 3,209 | 224 → 160 | 0.08 | 4 / 4, 12 | 0.00, 0 | 0.80 → 0.80 | 6 → 6 | 1 |
| 4 | Plummer | 4,345 → 4,345 | 280 → 104 | 0.12 | 4 / 6, 12 | 0.14, 214 | 1.81 → 1.68 | 21 → 18 | 3 |
| 8 | Plummer | 5,857 → 5,857 | 328 → 72 | 0.17 | 6 / 9, 21 | 0.19, 287 | 2.70 → 2.53 | 49 → 42 | 7 |
| 16 | Plummer | 6,129 → 6,017 | 1,048 → 128 | 0.58 | 5 / 9, 21 | 0.66, 821 | 2.78 → 2.34 | 100 → 69 | 15 |
| 32 | Plummer | 6,985 → 6,217 | 1,832 → 152 | 1.03 | 6 / 9, 24 | 1.17, 1,041 | 2.82 → 2.32 | 170 → 83 | 24 |
| 64 | Plummer | 8,937 → 7,177 | 2,944 → 136 | 1.66 | 5 / 12, 22 | 1.89, 1,214 | 3.14 → 2.61 | 227 → 90 | 31 |
| 72 | Plummer | 8,977 → 7,153 | 3,088 → 136 | 1.74 | 5 / 13, 24 | 1.98, 1,235 | 3.24 → 2.68 | 233 → 92 | 32 |
| 128 | Plummer | 10,433 → 7,257 | 4,560 → 120 | 2.58 | 4 / 19, 25 | 1.73, 1,123 | 3.31 → 2.90 | 329 → 119 | 39 |
| 256 | Plummer | 15,337 → 8,369 | 8,616 → 104 | 4.88 | 4 / 28, 29 | 1.39, 1,052 | 3.39 → 3.16 | 496 → 139 | 58 |
| 512 | Plummer | 24,057 → 9,129 | 16,672 → 96 | 9.45 | 4 / 32, 29 | 2.44, 1,204 | 3.87 → 3.65 | 630 → 145 | 56 |
| 2 | clusters | 3,857 → 3,761 | 560 → 344 | 0.19 | 5 / 5, 9 | 0.00, 0 | 0.47 → 0.48 | 7 → 7 | 1 |
| 4 | clusters | 4,729 → 4,249 | 1,056 → 344 | 0.51 | 4 / 5, 10 | 0.18, 231 | 1.02 → 0.92 | 18 → 14 | 3 |
| 8 | clusters | 5,937 → 5,129 | 1,512 → 384 | 0.83 | 6 / 9, 21 | 0.56, 466 | 1.73 → 1.57 | 31 → 18 | 4 |
| 16 | clusters | 7,025 → 5,993 | 2,280 → 536 | 1.27 | 6 / 10, 21 | 1.40, 724 | 2.14 → 1.75 | 58 → 28 | 7 |
| 32 | clusters | 8,457 → 6,521 | 3,104 → 584 | 1.75 | 6 / 11, 19 | 1.95, 887 | 2.55 → 2.25 | 122 → 52 | 15 |
| 64 | clusters | 10,265 → 7,217 | 4,568 → 552 | 2.58 | 7 / 13, 22 | 2.08, 887 | 2.94 → 2.47 | 243 → 104 | 28 |
| 72 | clusters | 10,689 → 7,425 | 4,840 → 568 | 2.74 | 6 / 13, 20 | 2.41, 956 | 2.96 → 2.54 | 267 → 104 | 28 |
| 128 | clusters | 13,257 → 7,801 | 6,888 → 616 | 3.90 | 7 / 14, 25 | 1.72, 765 | 3.38 → 3.04 | 436 → 181 | 37 |
| 256 | clusters | 19,241 → 8,593 | 12,184 → 608 | 6.90 | 7 / 15, 23 | 1.48, 711 | 3.93 → 3.66 | 820 → 285 | 56 |
| 512 | clusters | 26,657 → 8,729 | 19,744 → 592 | 11.19 | 7 / 17, 23 | 1.29, 652 | 3.60 → 3.45 | 1,472 → 510 | 98 |

What grows with P, at fixed N/P (requirement 5):
- **Gone:** the coarse gather (O(B) = O(P) bytes per rank and evaluation; 0.32 → 21.2 MB
  from 40 to 640 ranks on the cube) and the global M2M of every `Global` box (O(P) pairs;
  584 → 37,448).
- **Bounded:** the owned global M2M (32–536 pairs, by the rank's left edge in the block
  order, not by P), the reduction (at most 7 messages per level and rank, 5–8 levels by the
  depth of the blocks, ≤ 31 kB), R (capped at 4,096 boxes, ≤ 2.65 MB; 0.36–2.63 MB here),
  the held boxes (H, T4's bound), the multipole exchange's bytes (1.5–3.7 MB; between 40
  and 640 ranks within 1.9× on the cube, 1.6× on the Plummer sphere and 1.5× on the
  clusters).
- **Growing, less than in Phase 5:** the multipole exchange's messages and neighbours
  (neighbours 14 → 106 on the clusters and 22 → 60 on the cube from 40 to 640 Kathleen
  ranks; 1 → 98 and 1 → 47 from 2 to 512 on locust). A top-tree box below R goes to at
  most the threshold less one ranks, max(2, ⌈P/16⌉) − 1 unless the cap raised it, so a
  rank's fan-out may grow with P: the signed-off reading of requirement 5, not a constant
  bound. The cap does bind: on the locust clusters at 64–512 ranks and the Plummer
  sphere at 128 and 256 the threshold is twice ⌈P/16⌉ (below), and on Kathleen on the
  Plummer sphere and the clusters at 160 (16 for ⌈P/16⌉ = 10) and the clusters at 640 (64
  for 40; job 565143, below). The
  messages fall against Phase 5 at every count but the cube at 640 Kathleen ranks
  (115 → 139; 1,679 → 316 on the clusters there): the boxes many ranks read went into R,
  while the `Global` boxes other ranks own are now fetched from their owners.
  The redistribution's all-to-all-v's stay O(P) in their argument arrays (T8).
- **Where the neighbours come from** (a probe after the sweep, not kept in the code:
  each rank's multipole needs recomputed from its V- and W-lists, every rank's count
  equal to the harness's; locust, 2.5 × 10⁴ points per rank, f64 p = 8, 64–512 ranks):

  | Workload | Ranks | threshold (⌈P/16⌉) | R keys | neighbours: mean, max | from top-tree boxes, mean | from block-interior boxes, mean | from W-lists, mean | largest fan-out of a box |
  | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
  | clusters | 64 | 8 (4) | 3,214 | 12.8, 28 | 8.9 | 12.3 | 8.6 | 10 |
  | clusters | 128 | 16 (8) | 2,650 | 21.2, 37 | 20.1 | 17.9 | 9.8 | 15 |
  | clusters | 256 | 32 (16) | 2,288 | 35.2, 56 | 35.1 | 22.7 | 11.8 | 31 |
  | clusters | 512 | 64 (32) | 1,988 | 64.6, 98 | 64.6 | 25.8 | 8.6 | 63 |
  | cube | 64 | 4 (4) | 1,825 | 15.4, 28 | 3.1 | 15.4 | 4.0 | 12 |
  | cube | 512 | 32 (32) | 2,026 | 26.7, 47 | 24.7 | 21.7 | 5.1 | 31 |
  | Plummer | 64 | 4 (4) | 2,936 | 22.5, 31 | 0.0 | 22.5 | 11.6 | 13 |
  | Plummer | 512 | 32 (32) | 3,760 | 38.3, 56 | 35.8 | 30.3 | 9.7 | 31 |

  The neighbours a rank reaches through block-interior boxes and through W-lists grow
  slowly (12 → 26 and flat on the clusters); the growth is in the top-tree boxes. The
  boxes about a rank's size (levels 6–8 on the clusters) are read by a roughly constant
  number of ranks while the threshold doubles with P, so they leave R (the clusters'
  level 7: 679 → 377 → 174 → 15 boxes of R from 64 to 512 ranks, level 6: 847 → 483)
  and come point to point, each from its own owner: on the clusters at 512 ranks 53 of a
  rank's 65 neighbours (mean) are first needed on levels 6 and 7. R shrinks while the
  neighbours grow. At 512 ranks the ranks with the most neighbours are fine ranks inside
  the clusters (the worst has leaves on level 10 and no W-list neighbour; mean 77.8 for
  the ranks whose coarsest leaf is on level 10), not coarse ranks beside them (at 64 ranks
  the worst was a coarse rank, 28 neighbours). A top-tree box goes to fewer ranks than
  the threshold by construction; the last column also counts block-interior boxes, never
  candidates for R (12 ranks on the cube at 64).
- **By node** (since this report, `scaling` counts the multipole exchange by node from
  `Placement::node`, with the largest fan-out of a box and R's threshold, in the report
  and the summary line). Kathleen job 565143 (node-c11b-[005,025-026,038], 4 nodes, 7 min
  29 s; 160 ranks as 40 per node, one per core; 640 as 160 per node, oversubscribed, so
  more of each rank's neighbours share its node than at 40 per node), one evaluation,
  counted, not timed; every count of the sweep (§4) repeated exactly:

  | Ranks | Workload | threshold (⌈P/16⌉) | most ranks one box goes to | messages sent: all, to other nodes (max; mean) | neighbours: all, on other nodes (max; mean) | other nodes (max) |
  | ---: | --- | --- | ---: | --- | --- | ---: |
  | 160 | cube | 10 (10) | 10 | 72, 36; 53, 14 | 36, 18; 23, 7 | 3 |
  | 160 | Plummer | 16 (10) | 15 | 105, 88; 69, 28 | 39, 27; 28, 12 | 3 |
  | 160 | clusters | 16 (10) | 15 | 110, 73; 64, 13 | 40, 23; 23, 5 | 3 |
  | 640 | cube | 40 (40) | 39 | 139, 70; 95, 16 | 60, 24; 35, 6 | 3 |
  | 640 | Plummer | 40 (40) | 39 | 155, 129; 103, 28 | 64, 42; 41, 12 | 3 |
  | 640 | clusters | 64 (40) | 63 | 316, 203; 130, 20 | 106, 56; 56, 10 | 3 |

  Most of a rank's messages stay on its node: the ranks of a node hold a contiguous
  stretch of the block order, so on average 13–28 of a rank's 53–130 messages leave the
  node. The ranks at the ends of a node's stretch carry the rest (max over mean 2.5–10).
  With 4 nodes every rank can reach at most 3 others, so the count of nodes says
  nothing yet; the 160-rank (cube) case is the only one where a box went to more ranks
  than the threshold less one (10 against 9): a box inside a block, which has no such
  bound.

## 5. The reduction per level

Kathleen, 640 ranks (oversubscribed): per level the ranks that receive, the most messages
a rank receives and their sum, bytes likewise (a child multipole is 648 B; a message
carries one rank's children of one parent's owner):

| Workload | Level: receiving ranks, messages max / sum |
| --- | --- |
| cube | 1: 1, 7 / 7; 2: 8, 7 / 56; 3: 64, 7 / 448; 4: 512, 2 / 563; 5: 554, 1 / 554 |
| Plummer | 1: 1, 7 / 7; 2: 8, 4 / 23; 3: 22, 6 / 56; 4: 54, 7 / 119; 5: 97, 7 / 234; 6: 211, 7 / 403; 7: 321, 4 / 385; 8: 170, 1 / 170 |
| clusters | 1: 1, 4 / 4; 2: 4, 3 / 8; 3: 9, 4 / 15; 4: 14, 7 / 36; 5: 42, 7 / 111; 6: 122, 7 / 321; 7: 317, 6 / 538; 8: 420, 1 / 420 |

Every message is at most 4,536 B (7 children). Levels 1 and 2 (§2, reading 1) add one
message per level-1 octant owned elsewhere and at most 7 per level-2 parent: 7 + 56
messages over all 640 ranks on the cube.

## 6. The evaluation

T5 changes what the evaluation communicates and the global M2M, not the other level
calls. The evaluation wall time (median of 5, max over the ranks; T5's blocking path), and
the stage `upward global` (Phase 5: the coarse gather and the global M2M; T5: the
reduction, the owned global M2M and the all-gather of R), ms, `main` first in every pair:

| Ranks | cube | cube: upward global | Plummer | Plummer: upward global | clusters | clusters: upward global |
| ---: | --- | --- | --- | --- | --- | --- |
| Kathleen 40 | 663.1 → 611.7 | 4.8 → 2.7 | 1,575.4 → 1,471.0 | 10.3 → 4.6 | 1,708.1 → 1,628.9 | 17.9 → 6.6 |
| Kathleen 80 | 2,231.5 → 2,519.6 | 23.2 → 10.4 | 1,604.3 → 1,477.4 | 16.1 → 6.0 | 1,761.0 → 1,661.1 | 26.9 → 12.1 |
| Kathleen 160 | 1,267.1 → 1,149.4 | 20.8 → 8.5 | 1,662.1 → 1,711.6 | 25.8 → 165.6 | 1,996.3 → 1,888.2 | 38.7 → 15.8 |
| locust 2 | 383.8 → 393.5 | 0.1 → 0.2 | 560.6 → 570.9 | 0.6 → 0.4 | 554.0 → 558.2 | 1.3 → 1.2 |
| locust 8 | 215.9 → 217.6 | 0.5 → 0.4 | 512.4 → 517.2 | 0.8 → 0.5 | 563.4 → 567.2 | 2.4 → 1.4 |
| locust 32 | 285.6 → 289.4 | 1.1 → 0.5 | 545.6 → 548.6 | 3.0 → 1.3 | 619.3 → 618.1 | 4.9 → 2.3 |
| locust 64 | 280.5 → 277.4 | 4.2 → 1.2 | 623.3 → 626.2 | 5.7 → 2.7 | 646.7 → 643.4 | 8.2 → 3.3 |
| locust 72 | 446.7 → 444.1 | 8.0 → 1.9 | 621.0 → 627.3 | 6.1 → 2.8 | 695.1 → 696.4 | 9.4 → 4.0 |

The stage `upward global` falls 1.4–4.3× from 8 ranks up (1.8–2.7× on Kathleen; at
2 locust ranks both are under 1.5 ms). The exception, the Plummer sphere at 160 Kathleen
ranks (25.8 → 165.6 ms), is one launch: T4's `main` launch there spent 178 ms in the same
stage against 25 ms (phase5s-t4.md §5); not repeated. The overlapped path gives the
blocking path's output in every launch (`overlap_differing=0` in all 15 of T5's Kathleen
launches and all 30 on locust).

The totals differ by −8% to +13% at the timed Kathleen counts, in both directions and by
far more than the stage `upward global` saves (a few ms): Kathleen 40 ranks 663 → 612 ms
(cube), the cube at 80 ranks 2,231.5 → 2,519.6 ms. T5 leaves every other stage's lists
and work as they were; at that 80-rank cube the slowest rank's `leaves` grew from 1,131
to 1,384 ms and the backward output's wait for it from 511 to 796 ms, while the
exchanges' stages fell (the multipole exchange 4.9 → 3.8 ms). So the totals' differences
are launch-to-launch, as T4 found (1–4% there, the second launch of a pair slower on
locust); on locust, where the global pass was small, they stay within −1.1% and +2.5%.

## 7. Checks

M3 Max (sandboxed): `cargo fmt --all`; the root checks (`cargo fmt -- --check`, `cargo
clippy -- -D warnings`, `cargo clippy --examples -- -D warnings`, `RUST_MIN_STACK=8388608
cargo test`: 719 passed, `cargo doc --no-deps`) and the stricter workspace checks (`cargo
clippy --workspace --all-targets -- -D warnings`, `RUST_MIN_STACK=8388608 cargo test
--workspace`: 764 passed, 0 failed); `cargo clippy -p nd-fmm-plan --all-targets -- -D
warnings`, `cargo doc -p nd-fmm-plan --no-deps` (no warning); `cargo clippy -p
nd-fmm-validate --all-targets -- -D warnings`; `cargo clippy -p nd-fmm-exec --all-targets
--features cpu,metal -- -D warnings`, `cargo check -p nd-fmm-exec --features cuda`. MPI
runs as in §3.2.

Kathleen (2026-10-13):

| Job | Nodes | What | Waited | Ran | Node-hours | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 530629 | 1 (node-c11a-011) | release builds of `scaling`, branch and `main`; the debug MPI list | 1 s | 3:16 | 0.05 | 3 of 3 steps pass |
| 530664 | 2 (node-c11a-[011,013]) | T2's correctness job, first attempt | after 530629 | 13:38 | 0.45 | the single-node launches pass; every 80-rank launch fails in `MPI_Init` on node-c11a-011 (PSM2: no HFI context) |
| 530665 | 4 (node-c11a-[045,092,109,115]), `--exclude=node-c11a-011` | invariance at 160 and 640, the sweep (40–640 ranks, `main` and T5) | after 530629 | 22:22 | 1.49 | 1 of 1 step passes, every launch exit 0 |
| 531957 | 2 (node-c11a-[013,048]), `--exclude=node-c11a-011` | T2's correctness job | 1 s | 22:13 | 0.74 | 8 of 8 steps pass |
| 565143 | 4 (node-c11b-[005,025-026,038]), `--exclude=node-c11a-011` | the multipole exchange by node (§4), 160 and 640 ranks | 18 s | 7:29 | 0.50 | 2 of 2 steps pass, every launch exit 0 |

3.23 node-hours in all, of T5's 20 (README, "Node-hours"). node-c11a-011 still fails
launches across nodes, as in T4; a later job should exclude it.

locust (2026-10-13, Open MPI 5.0.10; one node, no other user's process in any load check;
the load before the timed launches came from the sweep's own preceding launches, run one
at a time, the 1-minute load before them up to 48 after the oversubscribed invariance run):
the correctness runs 00:43–01:30, the sweep 01:30–01:38 (`main` first in every pair),
`mpi_regressions` with the final test at 01:45.

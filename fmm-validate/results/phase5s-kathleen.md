# Scaling at scale (Phase 5S T9, C5S.6)

The scaling report of Phase 5S, as docs/phase5s/T9-scaling-at-scale.md asks: the
scale-out code (main at bb6730c: S1's locally essential set, S2's owned top tree with the
replicated top R, S3's work-weighted cut, S4's key routing, S5's point-to-point
redistribution, S8's forwarding tree) against the Phase 5 code, on Kathleen at 1, 2 and 4
nodes, on locust to 72 ranks and on the M3 Max to 12, run from 2026-10-15.

Every number carries one of four labels:
- **timed across nodes**: Kathleen, 1–4 nodes (40–160 cores, 2 × 20 Cascade Lake cores per
  node), Intel Omni-Path, Open MPI 4.1.6 with the `psm2` MTL, `PSM2_TID=0` and
  `PSM2_ERRCHK_TIMEOUT=10:40:2` (tools/kathleen/README.md, "MPI and the launcher"), ranks
  bound per core by `mpirun --map-by ppr:<k>:socket:PE=<t> --bind-to core`; each run's job,
  nodes, ranks per node and threads per rank are in §12;
- **timed on one node**: locust (GH200, 72 Neoverse-V2 cores, Open MPI 5.0.10) and the M3
  Max (12 performance cores, Open MPI 5.0.10, loopback TCP);
- **counted**: oversubscribed runs (locust to 512 ranks, Kathleen's 4 nodes to 640),
  never timed;
- ***model***: every time beyond 160 ranks, its inputs stated with it.

Release builds, every BLAS thread variable at 1, timings reported and never asserted.

## Summary

- **Strong scaling across nodes** (Kathleen, timed): N = 10⁸ scales from 1 to 4 nodes with an
  efficiency of 0.96–0.99 in all 12 configurations (the cube and the Plummer sphere, f64 p = 3
  and 8, f32 p = 8, 40 × 1 and 2 × 20 per node); N = 10⁷ 0.93–1.11 (§2). At f64 p = 8 the
  N = 10⁸ cube takes 14.4 s and the Plummer sphere 32.6 s on 160 cores.
- **Against Phase 5, same day and launches**: at 4 nodes 0.67–0.74 of its time on the Plummer
  sphere with 40 ranks per node, 0.89–1.01 elsewhere (§2, §4). The largest gain at this scale
  is the work-weighted cut (S3; the compute max/mean 1.52–1.64 → 1.08–1.11, §6); at one node
  with 2 × 20 the point-to-point redistribution (S5) is 1.15–2.35× faster than Phase 5's
  all-to-all-v.
- **Weak scaling**: the Plummer sphere and the clusters keep 0.88–1.21 of their one-node time at
  4 nodes (§3); the cube's tree changes between sizes.
- **Bounded per rank** (counted to 512 on locust and 640 on Kathleen, §5): at fixed N/P the held
  boxes stay at 4,000–8,400, the replicated top at 1.25–2.56 MB, the multipole exchange at
  1.8–3.7 MB, the memory model at 18–32 MB per rank; Phase 5 held every box of the tree
  (37,449) and gathered 11.6 MB at 512 ranks.
- **The exposed collectives** (§1, signed off 2026-10-15): Open MPI's flat all-gather-v of R is
  the fastest variant within 4 nodes (0.22 and 1.74 ms at R's cap on 160 ranks); the brief's
  two-level gather is 3–6 ms (its node broadcast), a hierarchical one 3.5 ms. **No remedy was
  built**; the flat gather grows linearly in P (*model* 6.3 and 12–24 ms at 10⁴ ranks), the
  hierarchical design is recorded. **Overlap stays off by default** (§7): neutral with one
  thread per rank, intermittently up to 4.4 % slower with 20 (one rank's downward pass; cause
  not established).
- **Correctness**: rank-count invariance 0 values differing at 80, 160, 512 and 640 ranks (both
  paths); the direct-sum errors identical in every printed digit (f64) and within 0.34 % (f32)
  across node counts, splits, cuts and against Phase 5; T2's correctness job passes at 2 nodes
  (§8).
- **Regressions** (one node, timed): locust and the M3 Max within −4 % to +2 % of Phase 5's
  reports up to 8 ranks at the same strategy, faster from 16 ranks; T6's locust anomaly does not
  recur (§9).
- **Beyond 160 ranks, *model*** (§10): weak scaling at f64 p = 8 within 2 % of the 160-rank
  time at 10⁴ ranks; at f32 p = 3 +2–16 %, the all-gather of R and the busiest sender's messages
  the growing terms; strong scaling at N = 10⁸ 0.96–1.00 to 10³ ranks, 0.41–0.94 at 10⁴ (the
  per-point cost at 10⁴ points per rank decides).
- **Found**: the cube at 2 × 10⁶ points (80 ranks) spends 12× the uniform trees' leaf time on a
  half-refined level's W-list M2P (767,220 evaluations per rank, each about 500 P2P pairs' cost),
  for Phase 6 (C6.9); node-c11a-011's Omni-Path contexts are busy (§11).
- **Node-hours**: 24.95 on Kathleen of the 60 budgeted (§12).

## 1. The exposed collectives (step 1, signed off)

The brief's addition from T8s: measure the replicated top's all-gather-v, the overlapped
path's waits per level and the blocking tree's receive copy before any build, then decide.
Measured on 2026-10-15 in jobs 604144, 604149 and 604150 (4 nodes each, §12), with the new
`collectives` example of nd-fmm-validate (`MPI_Allgatherv` of a buffer of R's size, split
evenly over the ranks, against two gathers in two levels; every value checked, 0 differ in
every launch) and the `scaling` harness's new overlap details (`PathTimes::overlap`: the
`test` calls of each exchange, the final wait for the sends, the wait per level).

### 1.1 The all-gather of R, micro-benchmarked (timed across nodes)

R at its cap of 4,096 boxes is 0.26 MB at f32 p = 3 and 2.65 MB at f64 p = 8. The median
over 30 calls of the slowest rank's time, ms (each call after a barrier):

| ranks (nodes × per node) | 0.26 MB: default | best forced | two-level | hierarchical | 2.65 MB: default | best forced | two-level | hierarchical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 (1 × 40) | 0.236 | 0.147 (4) | 0.569 | 0.441 | 1.434 | 1.434 (default) | 4.814 | 3.181 |
| 40 (2 × 20) | 0.277 | 0.120 (4) | 0.539 | – | 1.333 | 0.992 (4) | 2.859 | – |
| 40 (4 × 10) | 0.249 | 0.123 (4) | 0.364 | 0.238 | 1.205 | 0.968 (4) | 3.076 | 2.476 |
| 80 (2 × 40) | 0.518 | 0.169 (4) | 0.613 | 0.386 | 1.655 | 1.627 (4) | 5.474 | 3.256 |
| 80 (4 × 20) | 0.434 | 0.144 (4) | 0.555 | 0.265 | 1.333 | 1.134 (4) | 2.847 | 3.132 |
| 160 (4 × 40) | 0.228 | 0.221 (4) | 0.606 | 0.349 | 1.738 | 1.750 (4) | 5.884 | 3.463 |

- **default**: Open MPI's own choice; **best forced**: the fastest of the forced
  algorithms (`OMPI_MCA_coll_tuned_use_dynamic_rules=1`,
  `OMPI_MCA_coll_tuned_allgatherv_algorithm` = 1 gatherv and broadcast, 2 Bruck, 3 ring,
  4 neighbour exchange; in brackets), each a launch of its own; `coll/han` raised to
  priority 100 changes nothing (0.233 and 1.750 ms at 160: it has no all-gather-v).
- **two-level** (the brief's design): a gather-v on each node to its first rank, the
  leaders' all-gather-v, a broadcast of the whole buffer on each node. At 160 ranks its
  parts take 0.41 ms (node gather), 0.77 ms (leaders) and **5.68 ms (the node broadcast)**
  at 2.65 MB. Forcing the broadcast algorithm (job 604149, 160 ranks): the best is the
  scatter–allgather ring (9) at 3.13 ms, the two-level gather then 3.22 ms; the others
  3.3–24 ms.
- **hierarchical** (no broadcast): an all-gather-v on each node, the slices of each node's
  block all-gathered across the nodes by the ranks in the same place on every node, an
  all-gather-v on each node of the slices received, copied into place. At 160 ranks its
  parts take 0.55, 0.29 and 2.81 ms at 2.65 MB: delivering the whole buffer to the 40
  ranks of a node costs at least about 1.4 ms in any form (the flat gather on one node).
- Repeats of the default at 160 ranks in three launches on two node sets: 0.227–0.228 and
  1.738–1.751 ms.

**The flat all-gather-v is the fastest variant at every placement measured at 40 ranks
per node**, and within 0.1 ms of the best forced algorithm wherever Open MPI's default
picks neighbour exchange (160 ranks). At 80 ranks the default picks a slower algorithm for
0.26 MB (0.43–0.52 ms against 0.14–0.17 with neighbour exchange), a few tenths of a
millisecond. No two-level variant beats it within 4 nodes.

### 1.2 The flat all-gather's growth, refitted (*model*)

At 40 ranks per node (1, 2 and 4 nodes) the neighbour-exchange times fit
t(P) = a + b·P: **0.26 MB: a = 0.12 ms, b = 0.62 µs per rank; 2.65 MB: a = 1.43 ms,
b = 2.06 µs per rank** (Open MPI's default gives b = 2.3 µs over 40–160, 1.0 µs over
80–160). That replaces `scale-out.md` §10.2's all-gather term (log₂ P α_e + |R| n_c s /
β_g, which predicted about 2 ms at any P): the measured time grows linearly with P, as a
P/2-step algorithm does, though more slowly than T8s's inference (10–17 ms at 10⁴).

| R at the cap | 160 (measured) | 10³ (*model*) | 10⁴ (*model*) | 1 % of an evaluation at 2.5 × 10⁴ points per rank (T8 §5.2) | crosses it at (*model*) |
| --- | ---: | ---: | ---: | --- | --- |
| 0.26 MB (f32 p = 3) | 0.23 ms | 0.74 ms | 6.3 ms | cube 0.56, Plummer 1.47, clusters 1.85 ms | about 700, 2,200, 2,800 ranks |
| 2.65 MB (f64 p = 8) | 1.74 ms | 2.4–3.7 ms | 12–24 ms | cube 5.49, Plummer 12.1, clusters 14.3 ms | about 1,800–4,000, 4,600–10,600, 5,500–12,800 ranks |

The hierarchical gather's node parts do not grow with the nodes; its part across the nodes
moves 1/k of the buffer among one rank per node (*model* at 250 nodes, 125 steps of
1.5 µs plus 66 kB at 0.27 GB/s): about **0.55 ms (f32 p = 3) and 3.8 ms (f64 p = 8) at
10⁴ ranks**, against 6.3 and 12–24 ms flat; break-even about 700 and 1,000–2,400 ranks.
Open MPI may pick another algorithm at 10⁴ ranks; the model assumes it keeps neighbour
exchange.

### 1.3 In the FMM at 160 ranks (timed across nodes) and 640 (counted)

2.5 × 10⁴ points per rank, 40 ranks per node, both paths from one build, five timed
evaluations (job 604144). R's size in these runs is below the cap: 1.73–2.28 MB at
f64 p = 8 and 0.18–0.22 MB at f32 p = 3. "Replicated top" is the stage's time per rank:
its minimum over the ranks (the rank that arrives last, so the collective's own cost)
and maximum (which also holds the waits for slower ranks):

| precision | workload | evaluation blocking / overlapped (ms) | replicated top, min / max (ms) | exchanges blocking / exposed overlapped (ms) | wait for a level, max (ms) | `test` calls of the multipole exchange, max (ms) | the blocking tree's receive copy, max (ms) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| f64 p = 8 | cube | 1,162.9 / 1,147.8 | 1.71 / 7.23 | 6.36 / 0.004 | 0.001 | 0.061 | 1.34 |
| f64 p = 8 | Plummer | 1,254.7 / 1,244.4 | 3.70 / 21.7 | 12.76 / 0.91 | 0.113 | 0.274 | 1.17 |
| f64 p = 8 | clusters | 1,427.9 / 1,420.5 | 5.10 / 28.1 | 15.17 / 4.58 | 0.001 | 0.237 | 1.29 |
| f32 p = 3 | cube | 55.1 / 54.9 | 0.29 / 0.84 | 1.10 / 0.12 | 0.001 | 0.034 | 0.08 |
| f32 p = 3 | Plummer | 145.3 / 146.2 | 0.39 / 4.29 | 1.84 / 0.005 | 0.002 | 0.085 | 0.07 |
| f32 p = 3 | clusters | 182.5 / 182.4 | 0.73 / 3.82 | 1.89 / 0.23 | 0.096 | 0.064 | 0.09 |

- **The replicated top's own cost is 0.1–0.5 % of an evaluation** (the minimum: 1.7–5.1 ms
  of 1,150–1,430 ms at f64 p = 8, 0.3–0.7 ms of 55–182 ms at f32 p = 3); its maximum is the
  load imbalance arriving at the collective. The uneven owners of R make the gather slower
  than the even split of §1.1 (3.7–5.1 ms against 1.74 ms at a similar size).
- **No forward lags a level**: the overlapped path's wait for any level is at most
  0.11 ms (the tree's depth is 2 at 160 ranks; at 640 counted ranks it is 3 on every
  workload, with 713–2,046 chunks forwarded by the busiest forwarder and at most 8 copies
  of a box per rank, as in T8s).
- **The overlapped path's per-message cost**: its `test` calls take 0.03–0.27 ms per
  evaluation for the multipole exchange (0.02–0.58 ms for the source exchange).
- **The blocking tree's receive copy** (`copy_into` of `TreeExchange::forward`, timed with
  a temporary probe, not committed): 1.2–1.3 ms per evaluation on the slowest rank at
  f64 p = 8 (0.8–0.9 ms on average), 15–20 % of that path's multipole stage; under 0.1 ms
  at f32 p = 3. It explains part of T8s's +0.05–1.8 ms of the blocking tree against the
  direct exchange (T8s §7.3).
- **Overlap**: the overlapped evaluation is 1.3 % faster to 0.6 % slower than the blocking
  one; it hides the exchanges' 0.5–2 % of an evaluation and leaves the replicated top and
  the imbalance as they are.

### 1.4 Decided (signed off by the user on 2026-10-15)

The question put first: does any remedy bring a noticeable improvement? Not at a scale that
can be timed: within 160 ranks every remedy is slower than the flat gather, and the flat
gather is 0.1–1.3 % of an evaluation; only the model at 10³–10⁴ ranks favours one, mainly
at f32 p = 3. So:
1. **The all-gather: no remedy is built.** `TopExchange::gather` stays the one
   all-gather-v. The refitted model (§1.2) and both designs, the two-level gather with its
   measured node broadcast and the hierarchical gather with its modelled break-even
   (about 700 ranks at f32 p = 3, 1,000–2,400 at f64 p = 8), are recorded as open, for
   Phase 6 or a larger cluster (§11).
2. **Overlap on by default if the sweeps confirm it**: the overlapped path at or below the
   blocking path within the noise in every configuration at N = 10⁷ and 10⁸, and T0's two
   slow overlapped runs (a single run each, before the PSM2 timeout fix) not recurring;
   §7 has the outcome. Its expected gains lie where communication is a larger share than
   here (slower networks, thousands of ranks at low order, GPU nodes); the GPU case needs
   more than the default (the exchanges progress only between level calls, which on a
   device return as soon as their launches are queued).
3. **The blocking tree's improvement** (posting every level at once, scattering from the
   staging buffer): not built; the receive copy's cost is recorded as open.

## 2. Strong scaling at 1, 2 and 4 nodes (timed across nodes)

`tools/scaling/run.sh t9-strong` on this branch (jobs 604157 and 604158 at one node, 604160
at two, 604159 at four; §12) and on the Phase 5 tree (604197, 604229, 604193 and 604198):
the cube and the Plummer sphere, f64 p = 3 and 8 and f32 p = 8 under `auto` (the rule's
strategy with the harness's 96 MB budget: Dense at f64 p = 3, Rotation at f64 p = 8, at
f32 p = 8 Rotation at 40 × 1 and Dense at 2 × 20), 40 ranks × 1 thread and 2 ranks × 20
threads per node, the random share, both paths from one build, ten timed evaluations per
path at N = 10⁷ and two at 10⁸ after one warm-up, the errors of one charge vector
(§8). The blocking path's wall time (ms; the median over the evaluations of the slowest
rank), the efficiency from one to four nodes (speed-up over 4), the overlapped path over
the blocking one at 1, 2 and 4 nodes (§7), and the Phase 5 code's blocking time on the same
launches (Phase 5 ran f64 p = 3 and 8 at N = 10⁸ only at four nodes and p = 8; T0's runs
of that code are the rest, `phase5n-kathleen.md` S0.5):

#### N = 10,000,000

| problem | split | strategy | 1 node | 2 nodes | 4 nodes | 1→4 (eff.) | overlapped / blocking at 1, 2, 4 | Phase 5 at 1, 2, 4 | 5S / 5 at 4 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| cube, f64 p = 3 | 40 × 1 | Dense | 1,050.9 | 520.2 | 263.0 | 4.00 (1.00) | 1.001, 1.004, 1.015 | 1,068.3, 529.0, 272.3 | 0.966 |
| cube, f64 p = 3 | 2 × 20 | Dense | 1,537.5 | 720.8 | 347.7 | 4.42 (1.11) | 1.005, 1.019, 1.011 | 3,613.3, 714.3, 360.4 | 0.965 |
| cube, f64 p = 8 | 40 × 1 | Rotation | 5,488.0 | 2,770.8 | 1,396.0 | 3.93 (0.98) | 1.000, 0.998, 0.998 | 5,720.3, 2,846.3, 1,450.4 | 0.962 |
| cube, f64 p = 8 | 2 × 20 | Rotation | 5,791.1 | 2,885.1 | 1,417.6 | 4.09 (1.02) | 1.001, 1.003, 1.008 | 7,962.0, 2,868.7, 1,432.1 | 0.990 |
| cube, f32 p = 8 | 40 × 1 | Rotation | 4,642.2 | 2,298.2 | 1,166.7 | 3.98 (0.99) | 0.993, 1.001, 0.998 | 4,798.8, 2,391.4, 1,223.8 | 0.953 |
| cube, f32 p = 8 | 2 × 20 | Dense | 3,622.7 | 1,792.5 | 914.5 | 3.96 (0.99) | 0.997, 1.007, 1.007 | 5,692.4, 1,767.9, 1,014.8 | 0.901 |
| plummer, f64 p = 3 | 40 × 1 | Dense | 1,810.7 | 931.6 | 483.9 | 3.74 (0.94) | 1.001, 0.999, 1.000 | 2,182.4, 1,225.3, 682.0 | 0.710 |
| plummer, f64 p = 3 | 2 × 20 | Dense | 2,542.6 | 1,212.3 | 612.0 | 4.15 (1.04) | 1.003, 0.987, 1.008 | 4,545.1, 1,157.9, 648.6 | 0.944 |
| plummer, f64 p = 8 | 40 × 1 | Rotation | 12,888.9 | 6,661.5 | 3,453.9 | 3.73 (0.93) | 0.999, 0.998, 0.997 | 16,479.7, 8,384.8, 4,673.7 | 0.739 |
| plummer, f64 p = 8 | 2 × 20 | Rotation | 14,019.8 | 6,896.9 | 3,525.2 | 3.98 (0.99) | 0.986, 0.995, 1.001 | 16,075.3, 6,746.5, 3,625.9 | 0.972 |
| plummer, f32 p = 8 | 40 × 1 | Rotation | 11,841.4 | 6,099.2 | 3,133.8 | 3.78 (0.94) | 0.995, 0.995, 0.998 | 15,033.6, 7,785.8, 4,318.5 | 0.726 |
| plummer, f32 p = 8 | 2 × 20 | Dense | 10,399.0 | 5,036.3 | 2,563.4 | 4.06 (1.01) | 0.999, 1.001, 1.005 | 12,041.3, 4,823.9, 2,815.1 | 0.911 |

#### N = 100,000,000

| problem | split | strategy | 1 node | 2 nodes | 4 nodes | 1→4 (eff.) | overlapped / blocking at 1, 2, 4 | Phase 5 at 1, 2, 4 | 5S / 5 at 4 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| cube, f64 p = 3 | 40 × 1 | Dense | 13,331.6 | 6,684.9 | 3,421.4 | 3.90 (0.97) | 1.008, 1.002, 1.002 | –, –, – | – |
| cube, f64 p = 3 | 2 × 20 | Dense | 18,487.0 | 9,387.3 | 4,660.7 | 3.97 (0.99) | 1.002, 0.996, 1.010 | –, –, – | – |
| cube, f64 p = 8 | 40 × 1 | Rotation | 56,184.0 | 28,346.4 | 14,370.8 | 3.91 (0.98) | 1.000, 1.016, 1.009 | –, –, 14,611.8 | 0.984 |
| cube, f64 p = 8 | 2 × 20 | Rotation | 62,471.8 | 30,461.3 | 15,802.9 | 3.95 (0.99) | 1.000, 1.044, 1.006 | –, –, 15,900.4 | 0.994 |
| cube, f32 p = 8 | 40 × 1 | Rotation | 47,137.4 | 23,793.6 | 12,138.4 | 3.88 (0.97) | 1.005, 1.001, 0.989 | 48,588.1, 23,979.7, 12,064.9 | 1.006 |
| cube, f32 p = 8 | 2 × 20 | Dense | 40,040.2 | 20,629.3 | 10,196.4 | 3.93 (0.98) | 1.036, 1.025, 1.034 | 66,958.7, 21,102.8, 11,522.9 | 0.885 |
| plummer, f64 p = 3 | 40 × 1 | Dense | 18,175.7 | 8,630.6 | 4,624.6 | 3.93 (0.98) | 0.999, 1.001, 0.965 | –, –, – | – |
| plummer, f64 p = 3 | 2 × 20 | Dense | 22,810.8 | 11,730.9 | 5,907.1 | 3.86 (0.97) | 0.998, 1.003, 1.001 | –, –, – | – |
| plummer, f64 p = 8 | 40 × 1 | Rotation | 127,075.7 | 63,222.9 | 32,619.5 | 3.90 (0.97) | 0.994, 0.999, 1.000 | –, –, 47,434.0 | 0.688 |
| plummer, f64 p = 8 | 2 × 20 | Rotation | 134,062.5 | 67,713.5 | 34,810.8 | 3.85 (0.96) | 0.981, 1.003, 0.995 | –, –, 35,212.4 | 0.989 |
| plummer, f32 p = 8 | 40 × 1 | Rotation | 113,474.6 | 56,922.7 | 29,439.3 | 3.85 (0.96) | 1.013, 1.000, 0.989 | 134,900.8, 78,583.0, 43,676.5 | 0.674 |
| plummer, f32 p = 8 | 2 × 20 | Dense | 94,968.8 | 49,323.9 | 24,457.9 | 3.88 (0.97) | 1.002, 0.992, 1.007 | 124,738.5, 48,474.9, 26,427.9 | 0.925 |

- **N = 10⁸ scales from 1 to 4 nodes at 0.96–0.99** in every configuration (the cube
  0.97–0.99, the Plummer sphere 0.96–0.98); at N = 10⁷ 0.93–1.11, the Plummer sphere at
  40 × 1 lowest (0.93–0.94: its compute max/mean grows from 1.05 to 1.11 with the ranks).
- **Which split**: at f64 p = 8, 40 × 1 is now 2–10 % faster than 2 × 20 at every node count
  (T0, before the cut by work, found 2 × 20 ahead by 1–29 %); at f64 p = 3 40 × 1 is
  1.26–1.36× faster at 4 nodes; at f32 p = 8 2 × 20 with Dense is 1.14–1.28× faster than
  40 × 1 with Rotation.
- **Against Phase 5** on the same launches and day (the last column at 4 nodes): 0.67–0.74
  on the Plummer sphere at 40 × 1 (its compute max/mean 1.52–1.64 → 1.08–1.11, §6), 0.89–1.01
  on everything else; at one and two nodes (the Phase 5 columns) the same pattern, where the
  cut matters less. 604159 and 604198 ran on the same four nodes; the 2 × 20 Phase 5 runs of
  604193 on other nodes.
- **At one node with 2 × 20 the Phase 5 code is 1.15–2.35× slower** (the cube at N = 10⁷, f64
  p = 3: 3,613 against 1,538 ms), all of it in the redistribution: each of the 2 ranks
  forwards about 5 × 10⁶ charges and moves the output back, which Phase 5 does with one
  `MPI_Alltoallv` each way (0.97 and 1.74 s per evaluation) and Phase 5S point to point on the
  build's routes (S5, T8: 0.17 and 0.43 s). T0's one-node runs of the Phase 5 code (1,482 ms
  for that launch) predate `PSM2_TID=0`; that this setting slows the all-to-all-v's large
  messages within a node is an inference, not measured. At 40 × 1 (2.5 × 10⁵ points per rank)
  and at 2 and 4 nodes the two codes' redistributions cost about the same.
- **The build** at four nodes and N = 10⁸: 1.9–7.4 s at 40 × 1 and 31–50 s at 2 × 20 (one
  thread builds per rank, as in T0), the second cut included.

## 3. Weak scaling (timed across nodes)

`t9-weak`: 2.5 × 10⁴ points per rank at 40 × 1 (N = 10⁶, 2 × 10⁶, 4 × 10⁶) and 10⁶ points
per socket at 2 × 20 (N = 2 × 10⁶, 4 × 10⁶, 8 × 10⁶), the three workloads, f64 p = 3 and 8,
f32 p = 8 and 3; the same jobs. The blocking wall time (ms), its ratio from one to four
nodes (the one-node time over the four-node time: 1 is perfect), the overlapped path over
the blocking one, and the Phase 5 code at 1, 2 and 4 nodes with this branch's time over it:

| problem | per rank | split | strategy | 1 node | 2 nodes | 4 nodes | 1→4 | overlapped / blocking at 1, 2, 4 | Phase 5 at 1, 2, 4 | 5S / 5 at 1, 2, 4 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| cube, f64 p = 3 | 2.5 × 10⁴ | 40 × 1 | Dense | 95.7 | 431.5 | 106.3 | 0.90 | 1.003, 1.000, 0.999 | 97.3, 450.3, 111.5 | 0.983, 0.958, 0.953 |
| cube, f64 p = 3 | 10⁶ | 2 × 20 | Dense | 964.4 | 259.4 | 262.6 | 3.67 | 0.994, 1.019, 1.029 | 1,175.2, 260.1, 275.1 | 0.821, 0.997, 0.955 |
| cube, f64 p = 8 | 2.5 × 10⁴ | 40 × 1 | Rotation | 614.7 | 1,962.2 | 1,174.7 | 0.52 | 0.999, 1.002, 0.998 | 677.2, 2,165.6, 1,248.4 | 0.908, 0.906, 0.941 |
| cube, f64 p = 8 | 10⁶ | 2 × 20 | Rotation | 3,866.8 | 2,334.7 | 1,320.2 | 2.93 | 1.009, 1.000, 1.009 | 4,268.2, 2,280.2, 1,326.5 | 0.906, 1.024, 0.995 |
| cube, f32 p = 8 | 2.5 × 10⁴ | 40 × 1 | Rotation | 533.2 | 1,898.2 | 1,040.0 | 0.51 | 0.996, 1.001, 0.997 | 566.2, 2,062.8, 1,100.8 | 0.942, 0.920, 0.945 |
| cube, f32 p = 8 | 10⁶ | 2 × 20 | Dense | 3,758.6 | 1,459.9 | 844.4 | 4.45 | 0.940, 1.001, 1.032 | 3,682.5, 1,442.9, 941.4 | 1.021, 1.012, 0.897 |
| cube, f32 p = 3 | 2.5 × 10⁴ | 40 × 1 | Dense | 47.8 | 381.7 | 57.0 | 0.84 | 1.000, 1.001, 1.066 | 50.9, 395.8, 65.8 | 0.939, 0.964, 0.866 |
| cube, f32 p = 3 | 10⁶ | 2 × 20 | Dense | 848.0 | 156.8 | 149.6 | 5.67 | 0.989, 1.006, 0.967 | 956.3, 155.7, 156.6 | 0.887, 1.008, 0.955 |
| plummer, f64 p = 3 | 2.5 × 10⁴ | 40 × 1 | Dense | 218.0 | 207.9 | 191.6 | 1.14 | 1.002, 1.000, 0.998 | 260.2, 244.5, 272.9 | 0.838, 0.850, 0.702 |
| plummer, f64 p = 3 | 10⁶ | 2 × 20 | Dense | 512.8 | 473.1 | 495.6 | 1.03 | 1.031, 1.007, 1.013 | 799.6, 460.7, 529.7 | 0.641, 1.027, 0.936 |
| plummer, f64 p = 8 | 2.5 × 10⁴ | 40 × 1 | Rotation | 1,347.5 | 1,371.7 | 1,278.5 | 1.05 | 0.997, 1.000, 0.997 | 1,592.8, 1,549.7, 1,620.0 | 0.846, 0.885, 0.789 |
| plummer, f64 p = 8 | 10⁶ | 2 × 20 | Rotation | 2,900.2 | 2,655.4 | 2,831.5 | 1.02 | 0.995, 0.992, 0.998 | 3,132.0, 2,630.1, 2,908.6 | 0.926, 1.010, 0.973 |
| plummer, f32 p = 8 | 2.5 × 10⁴ | 40 × 1 | Rotation | 1,266.7 | 1,268.9 | 1,153.6 | 1.10 | 0.999, 1.001, 0.999 | 1,462.4, 1,413.6, 1,474.7 | 0.866, 0.898, 0.782 |
| plummer, f32 p = 8 | 10⁶ | 2 × 20 | Dense | 2,241.0 | 1,940.0 | 2,077.0 | 1.08 | 1.013, 1.022, 0.999 | 2,336.0, 1,912.2, 2,297.5 | 0.959, 1.015, 0.904 |
| plummer, f32 p = 3 | 2.5 × 10⁴ | 40 × 1 | Dense | 180.4 | 174.9 | 148.6 | 1.21 | 0.999, 1.000, 1.010 | 214.2, 197.9, 212.6 | 0.843, 0.884, 0.699 |
| plummer, f32 p = 3 | 10⁶ | 2 × 20 | Dense | 387.5 | 350.8 | 377.0 | 1.03 | 1.007, 1.000, 1.000 | 593.6, 347.8, 406.1 | 0.653, 1.009, 0.928 |
| clusters, f64 p = 3 | 2.5 × 10⁴ | 40 × 1 | Dense | 229.8 | 198.2 | 228.6 | 1.01 | 0.999, 0.998, 1.002 | 278.0, 277.4, 348.1 | 0.827, 0.715, 0.657 |
| clusters, f64 p = 3 | 10⁶ | 2 × 20 | Dense | 476.7 | 498.3 | 484.9 | 0.98 | 1.011, 1.002, 1.019 | 746.9, 486.6, 514.7 | 0.638, 1.024, 0.942 |
| clusters, f64 p = 8 | 2.5 × 10⁴ | 40 × 1 | Rotation | 1,501.0 | 1,311.2 | 1,427.8 | 1.05 | 0.999, 0.997, 0.996 | 1,735.6, 1,730.5, 1,977.3 | 0.865, 0.758, 0.722 |
| clusters, f64 p = 8 | 10⁶ | 2 × 20 | Rotation | 2,562.2 | 2,732.8 | 2,912.2 | 0.88 | 1.004, 0.993, 0.993 | 2,862.7, 2,671.7, 3,007.5 | 0.895, 1.023, 0.968 |
| clusters, f32 p = 8 | 2.5 × 10⁴ | 40 × 1 | Rotation | 1,394.5 | 1,199.1 | 1,293.7 | 1.08 | 0.999, 0.997, 1.000 | 1,603.2, 1,589.4, 1,839.3 | 0.870, 0.754, 0.703 |
| clusters, f32 p = 8 | 10⁶ | 2 × 20 | Dense | 1,904.1 | 2,065.6 | 2,092.6 | 0.91 | 0.998, 1.004, 1.005 | 2,085.5, 2,017.5, 2,340.4 | 0.913, 1.024, 0.894 |
| clusters, f32 p = 3 | 2.5 × 10⁴ | 40 × 1 | Dense | 204.9 | 160.1 | 189.1 | 1.08 | 0.972, 1.000, 1.001 | 230.7, 229.4, 284.1 | 0.888, 0.698, 0.666 |
| clusters, f32 p = 3 | 10⁶ | 2 × 20 | Dense | 344.0 | 376.4 | 369.0 | 0.93 | 1.009, 1.010, 0.991 | 534.0, 367.0, 393.1 | 0.644, 1.026, 0.939 |

- **The Plummer sphere and the clusters scale weakly**: 0.88–1.21 of their one-node time
  at four nodes (above 1 where the 4-node tree is easier per point), 0.88–1.08 at 2 × 20.
- **The cube changes its tree between the sizes**: 6, 7 and 7 levels at 2.5 × 10⁴ points
  per rank, the seventh half refined at 2 × 10⁶ points (80 ranks), with W and X lists
  (12,688 W pairs per rank on average) whose M2P and P2L make its leaves 12× those of the
  uniform trees (§11): its weak figures compare different work, as T0 found.
- **Against Phase 5 at four nodes**: 0.66–1.00 of its time (0.66–0.79 on the Plummer sphere
  and the clusters at 40 × 1, the cut); at two nodes 0.70–1.03; at one node 0.64–1.02, the
  2 × 20 runs there the lowest (the redistribution, §2).

## 4. Phase 5 against Phase 5S, stage by stage (timed across nodes)

At four nodes on the same launches (§2; maxima over the ranks, the blocking path): what S1
and S2 changed (the held boxes, the gather, the global pass), S3 (the compute balance) and
the evaluation:

| problem | split | held boxes, max: 5 → 5S | gather / R per evaluation (MB): 5 → 5S | upward global, max (ms): 5 → 5S | compute max/mean: 5 → 5S | evaluation (ms): 5 → 5S | 5S / 5 |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| cube, N = 1e+07, f64 p = 3 | 40 × 1 | 10,969 → 6,953 | 0.52 → 0.35 | 3.4 → 2.7 | 1.078 → 1.039 | 272.3 → 263.0 | 0.966 |
| cube, N = 1e+07, f64 p = 3 | 2 × 20 | 46,745 → 46,449 | 0.03 → 0.02 | 2.1 → 1.6 | 1.062 → 1.025 | 360.4 → 347.7 | 0.965 |
| cube, N = 1e+07, f64 p = 8 | 40 × 1 | 10,969 → 6,777 | 2.64 → 1.74 | 24.0 → 15.9 | 1.094 → 1.046 | 1,450.4 → 1,396.0 | 0.962 |
| cube, N = 1e+07, f64 p = 8 | 2 × 20 | 46,745 → 46,449 | 0.15 → 0.11 | 9.7 → 5.3 | 1.015 → 1.013 | 1,432.1 → 1,417.6 | 0.990 |
| cube, N = 1e+07, f32 p = 8 | 40 × 1 | 10,969 → 6,777 | 1.32 → 0.86 | 25.4 → 13.1 | 1.095 → 1.051 | 1,223.8 → 1,166.7 | 0.953 |
| cube, N = 1e+07, f32 p = 8 | 2 × 20 | 46,745 → 46,449 | 0.08 → 0.05 | 8.0 → 5.0 | 1.204 → 1.093 | 1,014.8 → 914.5 | 0.901 |
| plummer, N = 1e+07, f64 p = 3 | 40 × 1 | 27,993 → 17,249 | 0.60 → 0.40 | 3.8 → 8.7 | 1.610 → 1.098 | 682.0 → 483.9 | 0.710 |
| plummer, N = 1e+07, f64 p = 3 | 2 × 20 | 107,857 → 106,961 | 0.03 → 0.04 | 2.2 → 2.2 | 1.201 → 1.131 | 648.6 → 612.0 | 0.944 |
| plummer, N = 1e+07, f64 p = 8 | 40 × 1 | 27,993 → 13,409 | 3.04 → 2.04 | 33.5 → 40.7 | 1.522 → 1.110 | 4,673.7 → 3,453.9 | 0.739 |
| plummer, N = 1e+07, f64 p = 8 | 2 × 20 | 107,857 → 106,961 | 0.16 → 0.18 | 13.1 → 11.0 | 1.128 → 1.090 | 3,625.9 → 3,525.2 | 0.972 |
| plummer, N = 1e+07, f32 p = 8 | 40 × 1 | 27,993 → 13,793 | 1.52 → 1.02 | 30.1 → 42.0 | 1.553 → 1.113 | 4,318.5 → 3,133.8 | 0.726 |
| plummer, N = 1e+07, f32 p = 8 | 2 × 20 | 107,857 → 106,961 | 0.08 → 0.09 | 7.9 → 8.4 | 1.209 → 1.111 | 2,815.1 → 2,563.4 | 0.911 |
| cube, N = 1e+08, f64 p = 8 | 40 × 1 | 37,305 → 32,585 | 2.64 → 1.79 | 74.4 → 82.4 | 1.074 → 1.057 | 14,611.8 → 14,370.8 | 0.984 |
| cube, N = 1e+08, f64 p = 8 | 2 × 20 | 356,481 → 356,193 | 0.15 → 0.08 | 75.0 → 38.6 | 1.060 → 1.043 | 15,900.4 → 15,802.9 | 0.994 |
| cube, N = 1e+08, f32 p = 8 | 40 × 1 | 37,305 → 32,585 | 1.32 → 0.89 | 62.8 → 75.7 | 1.069 → 1.058 | 12,064.9 → 12,138.4 | 1.006 |
| cube, N = 1e+08, f32 p = 8 | 2 × 20 | 356,481 → 356,193 | 0.07 → 0.04 | 100.9 → 37.9 | 1.150 → 1.033 | 11,522.9 → 10,196.4 | 0.885 |
| plummer, N = 1e+08, f64 p = 8 | 40 × 1 | 107,905 → 78,241 | 3.06 → 2.05 | 249.2 → 339.2 | 1.597 → 1.082 | 47,434.0 → 32,619.5 | 0.688 |
| plummer, N = 1e+08, f64 p = 8 | 2 × 20 | 902,785 → 901,865 | 0.16 → 0.18 | 85.0 → 59.3 | 1.107 → 1.083 | 35,212.4 → 34,810.8 | 0.989 |
| plummer, N = 1e+08, f32 p = 8 | 40 × 1 | 107,905 → 78,241 | 1.53 → 1.03 | 143.6 → 357.8 | 1.636 → 1.086 | 43,676.5 → 29,439.3 | 0.674 |
| plummer, N = 1e+08, f32 p = 8 | 2 × 20 | 902,785 → 901,865 | 0.08 → 0.09 | 104.2 → 48.8 | 1.163 → 1.094 | 26,427.9 → 24,457.9 | 0.925 |

- **The held boxes** fall 1.0–2.1× at 160 ranks with 40 ranks per node (the cube 10,969 →
  6,777–6,953, the Plummer sphere 27,993 → 13,409–17,249 at N = 10⁷); at 2 × 20 (8 ranks)
  nothing is replicated to speak of. The counts at 128–640 ranks (§5) are where S1 and S2
  show: Phase 5's set grows to every box of the tree, H stays bounded.
- **The gather**: Phase 5's coarse gather 0.03–3.06 MB per evaluation at 160 ranks against
  R's 0.02–2.05 MB here; at 512–640 ranks 9.4–21.2 MB against 1.25–2.56 MB (§5).
- **"Upward global"** (Phase 5: the coarse gather and the global M2M of every `Global` box;
  here the reduction, the owned global M2M and R's all-gather) is 0.2–1.8 % of an
  evaluation in both codes at 160 ranks; its maximum on the Plummer sphere at 40 × 1 grows
  (33 → 41 ms at N = 10⁷, 249 → 339 ms at 10⁸) because the remaining imbalance now
  arrives at the reduction's waits, not because the stage costs more: its own time (the
  fastest rank's) is 0.01–5 ms (§1.3).
- **The evaluation** gains most from S3 at this scale (§6); S1 and S2 remove the terms that
  grow with P (§5, §10), which at 160 ranks are still small.

## 5. Memory, held boxes and traffic per rank (counted)

Oversubscribed, never timed: locust at 128, 256 and 512 ranks (Open MPI 5.0.10, `--map-by
:OVERSUBSCRIBE --bind-to none`, the open-file limit raised; idle but for these launches),
Kathleen's 4 nodes at 320 and 640 (80 and 160 ranks per node, PSM2 with shared contexts,
`--bind-to none`; job 604159). f64 p = 8, Rotation, the warm-up evaluation only; maxima over
the ranks. "R" is what the replicated top's all-gather delivers per evaluation; the
reduction's bytes are those received over every level.

| workload | size | ranks | held boxes | R (MB) | threshold | reduction (kB) | multipole exchange (MB) | multipole messages sent | depth | memory model (MB) | resident (MB) | modelled max/mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | N = 10⁶ | 128 (l) | 2,161 | 1.99 | 8 | 11.0 | 0.79 | 40 | 2 | 9.2 | 128 | 1.039 |
| cube | N = 10⁶ | 512 (l) | 1,457 | 1.49 | 32 | 16.8 | 0.52 | 61 | 2 | 5.7 | 451 | 1.150 |
| cube | N = 10⁶ | 640 (K) | 1,401 | 1.49 | 40 | 18.1 | 0.52 | 73 | 2 | 5.6 | 163 | 1.054 |
| cube | N = 10⁷ | 512 (l) | 3,857 | 1.49 | 32 | 17.5 | 1.70 | 92 | 2 | 16.5 | 733 | 1.111 |
| cube | N = 10⁷ | 640 (K) | 3,481 | 1.49 | 40 | 17.5 | 1.59 | 91 | 2 | 14.8 | 202 | 1.023 |
| cube | 2.5 × 10⁴ per rank | 128 (l) | 7,209 | 1.86 | 8 | 13.6 | 2.82 | 64 | 2 | 26.9 | 167 | 1.032 |
| cube | 2.5 × 10⁴ per rank | 320 (K) | 4,713 | 1.50 | 20 | 14.9 | 2.02 | 87 | 2 | 19.9 | 144 | 1.090 |
| cube | 2.5 × 10⁴ per rank | 512 (l) | 4,025 | 1.49 | 32 | 20.1 | 1.84 | 98 | 2 | 18.4 | 750 | 1.110 |
| cube | 2.5 × 10⁴ per rank | 640 (K) | 6,225 | 1.49 | 40 | 18.1 | 2.71 | 120 | 3 | 23.2 | 214 | 1.020 |
| Plummer | N = 10⁶ | 512 (l) | 2,713 | 2.26 | 32 | 27.9 | 0.86 | 79 | 2 | 9.3 | 465 | 1.071 |
| Plummer | N = 10⁷ | 512 (l) | 7,273 | 2.34 | 32 | 29.2 | 2.97 | 125 | 2 | 26.6 | 758 | 1.056 |
| Plummer | N = 10⁷ | 640 (K) | 6,465 | 2.29 | 40 | 31.1 | 2.67 | 122 | 3 | 23.8 | 216 | 1.064 |
| Plummer | 2.5 × 10⁴ per rank | 128 (l) | 7,489 | 1.65 | 16 | 26.6 | 3.10 | 118 | 2 | 28.5 | 171 | 1.089 |
| Plummer | 2.5 × 10⁴ per rank | 320 (K) | 7,929 | 2.56 | 20 | 28.5 | 3.22 | 116 | 2 | 31.6 | 158 | 1.066 |
| Plummer | 2.5 × 10⁴ per rank | 512 (l) | 8,425 | 2.40 | 32 | 31.1 | 3.44 | 127 | 2 | 31.8 | 772 | 1.073 |
| Plummer | 2.5 × 10⁴ per rank | 640 (K) | 8,425 | 2.32 | 40 | 28.5 | 3.45 | 147 | 3 | 31.6 | 230 | 1.083 |
| clusters | N = 10⁷ | 512 (l) | 7,241 | 1.26 | 64 | 22.0 | 3.24 | 233 | 3 | 26.5 | 717 | 1.058 |
| clusters | N = 10⁷ | 640 (K) | 6,601 | 1.76 | 64 | 22.7 | 2.97 | 217 | 3 | 23.7 | 214 | 1.057 |
| clusters | 2.5 × 10⁴ per rank | 128 (l) | 6,449 | 1.71 | 16 | 22.7 | 2.77 | 124 | 2 | 27.4 | 165 | 1.094 |
| clusters | 2.5 × 10⁴ per rank | 320 (K) | 8,417 | 1.96 | 32 | 21.4 | 3.63 | 164 | 2 | 31.0 | 158 | 1.057 |
| clusters | 2.5 × 10⁴ per rank | 512 (l) | 8,265 | 1.25 | 64 | 22.0 | 3.72 | 241 | 3 | 31.1 | 740 | 1.061 |
| clusters | 2.5 × 10⁴ per rank | 640 (K) | 7,641 | 1.69 | 64 | 20.7 | 3.41 | 244 | 3 | 30.6 | 226 | 1.072 |

(l) locust, (K) Kathleen; "depth" is the forwarding tree's (stages of the multipole
exchange, at most 3: 3 on the clusters from 256–512 locust ranks and on every workload at
640 Kathleen ranks but the N = 10⁶ and 10⁷ cube). Every launch of both machines is in §12.

- **At fixed N/P the per-rank structures stay bounded in P** (requirement 4): at
  2.5 × 10⁴ points per rank from 128 to 640 ranks the held boxes stay at 4,000–8,400, R at
  1.25–2.56 MB (the cap of 4,096 boxes is 2.65 MB), the reduction at 13.6–31 kB, the
  multipole exchange at 1.8–3.7 MB, the memory model at 18–32 MB per rank. **Against Phase
  5 at 512 ranks** (phase5-gh200.md §10, the N = 10⁶ cube): 37,449 held boxes and an
  11.6 MB coarse gather per rank then, 1,457 held boxes and 1.49 MB of R now (the
  harness's own record of Phase 5's coarse gather on the same trees, `phase5_coarse_bytes`:
  9.4–12.0 MB at 512 ranks; Phase 5's global M2M pairs per rank, `phase5_global_m2m`:
  16,608–21,128 against 80–520 owned now).
- **The busiest sender's messages grow slowly**: 40–64 at 128 ranks to 73–244 at 640;
  the clusters most (124 → 244), as T8s measured (its §7.1: 98–244 at 512–640).
- **The resident size grows with the ranks per machine, not with the work**: 128–171 MB at
  128 locust ranks, 412–772 MB at 512, against a memory model of 5.6–32 MB; on Kathleen
  123–230 MB at 80–160 ranks per node. The harness's resident size after the input and
  before any build is 16–20 MB on locust and 62–90 MB on Kathleen (MPI's and PSM2's
  buffers at start-up), and on one oversubscribed machine every rank is a shared-memory
  peer of every other: Open MPI's per-peer buffers, not the FMM's structures (Phase 5 at
  512 locust ranks: 744–819 MB).
- **The modelled cost's balance** (T7's cut on by default, its constants): max over mean
  1.020–1.150, the largest the cube at N = 10⁶ on 512 ranks (about 2,000 points per rank),
  as T7 accepted (1.124 there; README decision 3).

## 6. Load balance with and without the work-weighted cut (timed across nodes)

N = 10⁷, f64 p = 8, Rotation, both cuts in the same job (`t9-strong`'s `-cutpoints`
launches): the evaluation (blocking, ms), the measured compute max/mean and the modelled
cost's max/mean (T7's constants), and whether the second cut moved blocks (`recut`, the 2 %
gain rule of README decision 3):

| nodes | workload | split | cut by points: evaluation, compute, modelled | cut by work: evaluation, compute, modelled | recut | work / points |
| ---: | --- | --- | --- | --- | --- | ---: |
| 1 | cube | 40 × 1 | 5,609.7, 1.070, 1.071 | 5,488.0, 1.042, 1.040 | yes | 0.978 |
| 1 | Plummer | 40 × 1 | 15,973.7, 1.307, 1.306 | 12,888.9, 1.048, 1.058 | yes | 0.807 |
| 2 | cube | 40 × 1 | 2,866.5, 1.078, 1.076 | 2,770.8, 1.039, 1.020 | yes | 0.967 |
| 2 | Plummer | 40 × 1 | 8,384.6, 1.361, 1.371 | 6,661.5, 1.072, 1.047 | yes | 0.794 |
| 4 | cube | 40 × 1 | 1,472.7, 1.107, 1.091 | 1,396.0, 1.046, 1.034 | yes | 0.948 |
| 4 | Plummer | 40 × 1 | 4,661.2, 1.515, 1.522 | 3,453.9, 1.110, 1.087 | yes | 0.741 |
| 1 | cube | 2 × 20 | 5,807.2, 1.002, 1.000 | 5,791.1, 1.006, 1.000 | no | 0.997 |
| 1 | Plummer | 2 × 20 | 13,847.9, 1.040, 1.000 | 14,019.8, 1.050, 1.000 | no | 1.012 |
| 2 | cube | 2 × 20 | 2,898.1, 1.014, 1.000 | 2,885.1, 1.016, 1.000 | no | 0.996 |
| 2 | Plummer | 2 × 20 | 6,813.3, 1.044, 1.002 | 6,896.9, 1.057, 1.002 | no | 1.012 |
| 4 | cube | 2 × 20 | 1,435.3, 1.010, 1.001 | 1,417.6, 1.013, 1.001 | no | 0.988 |
| 4 | Plummer | 2 × 20 | 3,537.2, 1.098, 1.020 | 3,525.2, 1.090, 1.020 | no | 0.997 |

- **At 40 ranks per node the cut by work is the largest single gain of the phase** at
  these sizes: the Plummer sphere's compute max/mean falls from 1.31–1.52 to 1.05–1.11 and
  its evaluation by 19–26 %; the cube's by 2–5 %. The modelled max/mean follows the measured
  one within 0.03 (T7's constants carry over to the final structure).
- **At 2 × 20 the gain rule leaves the cut by points** (8 ranks hold blocks of very
  similar modelled cost: max/mean 1.000–1.020), so both launches have the same partition
  and their evaluation times, 0.3–1.2 % apart, **measure the run-to-run noise** of this
  harness between two launches on the same nodes. The measured compute max/mean at 2 × 20
  (1.04–1.10 on the Plummer sphere) is the threads' own imbalance within a rank and the
  per-rank cost the model does not see (P2P's share, T8s §9), not the cut.
- What the cut costs at build: 92–656 ms at 40 × 1 (12–37 % of the build), 268–1,206 ms at
  2 × 20 (where it changes nothing); in T7's measured range.

## 7. Overlap at network bandwidth (timed across nodes)

Every timed launch ran both paths from one build, alternating per evaluation (`--overlap
both`). Over the 156 timed launches of Phase 5S on Kathleen (strong and weak, 1–4 nodes,
both splits, the three workloads, every precision; §2, §3), the overlapped evaluation over
the blocking one:

| | launches | median | min | max | above 1.02 | below 0.98 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| every launch | 156 | 1.0003 | 0.940 | 1.066 | 9 | 4 |
| 40 ranks × 1 thread per node | 78 | 0.9995 | 0.965 | 1.066 | 1 | 2 |
| 2 ranks × 20 threads per node | 78 | 1.0025 | 0.940 | 1.044 | 8 | 2 |

- **What there is to hide is small**: the blocking source and multipole exchanges are
  0.2–1.1 % of an evaluation at 160 ranks at 2.5 × 10⁴ points per rank (§1.3, §3) and at
  N = 10⁷ (at most 34 ms of 3,454 ms; §2), and overlap hides them (exposed waits
  0.004–4.6 ms).
- **With one thread per rank overlap is neutral**: median 0.9995, one launch above +2 %
  (the 57 ms weak cube at f32 p = 3, +3.8 ms on the slowest rank's downward pass and the
  backward wait).
- **With 20 threads per rank it is slightly and intermittently slower**: median +0.25 %,
  eight launches at +2.2 % to +4.4 %, against a run-to-run noise of 0.3–1.2 % (§6). The
  clearest case is the N = 10⁸ cube at f32 p = 8 with Dense at 2 × 20: +3.6 %, +2.5 % and
  +3.4 % at 1, 2 and 4 nodes. In each, **one rank's downward pass is slower on the
  overlapped path, every other rank's unchanged**: rank 0, 6,365 against 5,477 ms at 4
  nodes and 24,287 against 22,891 ms at one; the others wait for it in the backward of the
  output.
- **The diagnostic** (job 604225, one node, that configuration, three evaluations per path,
  A/B/A/B): with PSM2's receive thread on (the default) the overlapped path was 0.995 and
  1.056 of the blocking one, with `PSM2_RCVTHREAD=0` 1.007 and 0.994; the slowdown (rank 0's
  downward pass 25.7 against 23.4 s) appeared once in four launches, with the thread on. The
  hypothesis (rank 0, which owns the top of the tree, keeps sends pending through the
  downward pass; with requests pending PSM2's receive thread wakes and takes a core from
  that rank's 20 rayon threads) fits but is not shown: two launches each cannot separate a
  one-in-two event from chance. Rank 0's source exchange completed 1.4 s after it was posted
  on the overlapped path in three of the four launches (the work hid it), so that is not the
  cause either.

**Overlap stays off by default** (§1.4, decision 2: on only if the overlapped path is at
or below the blocking path within the noise in every configuration): it is not, at
2 × 20 at N = 10⁸, and the cause is not established. `FmmBuilder::overlap` and
`Fmm::set_overlap` are unchanged; the expected regimes where overlap pays (§1.4) and a
longer A/B of the receive thread (for example ten rounds, and `PSM2_RCVTHREAD=0` in the
sweeps) are open (§11).

## 8. Correctness

- **Rank-count invariance, 0 values differ at every pair run** (both paths, `--overlap
  both`, the cube and the Plummer sphere at f64 p = 8 with Rotation and the cube at f32
  p = 3 with Dense):
  - Kathleen, 4 nodes (job 604231, `correctness.sbatch invariance oversubscribed`): 160
    ranks against 80, 40 and 20 (N = 10⁶; 8 × 10⁶ values per check), and 640 ranks
    oversubscribed against 80 (the cube, f64 p = 8);
  - Kathleen, 2 nodes (job 604230, T2's correctness job): 80 against 40, 20 and 10;
  - locust (oversubscribed, 2026-10-15): 512 against 64 ranks at N = 10⁷ (4 × 10⁷ values
    per check), blocking only and again with both paths.
- **T2's correctness job at 2 nodes passes in full** (job 604230, `test`, 23 min 51 s):
  the MPI test list at 2, 4, 8 and 80 ranks, the invariance above, the smoke run at 4 × 20
  (φ L2 2.913e-3, ∇φ L2 4.673e-3 against the direct sum, as in T1).
- **The direct-sum errors at every timed configuration** (one charge vector at 1,000
  sampled targets, the distributed direct sum; every `t9-strong` and `t9-weak` launch,
  Phase 5S and Phase 5): against the run of the same configuration at the fewest nodes (for
  strong scaling; weak scaling's problems differ per node count, so there against Phase 5 at
  the same count), with the same resolved strategy:
  - **f64: identical in every printed digit** of φ L2 and ∇φ L2 (4 significant digits) in
    all 72 strong comparisons across node counts, splits and both cuts and the 28 against
    Phase 5, and in the 24 weak comparisons against Phase 5;
  - **f32: within 0.34 %** (16 comparisons across node counts, 21 against Phase 5; the
    largest the Plummer sphere at N = 10⁷, p = 8, 2 × 20: φ L2 1.479e-5 at 2 nodes against
    1.484e-5 at 1), within the 1 % of the README's oracle; the weak comparisons against
    Phase 5 identical in every digit.
  - Where the rule resolves a different strategy (f32 p = 8: Dense at 2 × 20, Rotation at
    40 × 1) the errors differ by the strategy (cube N = 10⁷: 1.845e-5 against 1.833e-5), as
    they should.
- **Overlap bit for bit**: every timed launch reports 0 values of the two paths differing.
- The Phase 5S suites: `RUST_MIN_STACK=8388608 cargo test --workspace` on the M3 Max (68
  executables, 773 passed, 0 failed, 57 ignored), the smoke run of `scaling` at 1 and 2
  ranks, and the MPI test list at 2, 4, 8 and 80 ranks inside T2's correctness job (§12).
  The library is unchanged in this task, so the shadow, device and CUDA checks of T8s stand.

## 9. Regressions on locust and the M3 Max (timed on one node)

Phase 5 T10's strong sweep (`tools/scaling/run.sh strong`: the cube and the Plummer
sphere at N = 10⁶, f64 p = 3 and 8, f32 p = 8, one thread per rank, both paths, ten timed
evaluations) on this branch, against `phase5-gh200.md` §1 and `phase5-m3max.md` §1.
Blocking wall time (ms), in brackets the ratio to Phase 5 and the strategy (D Dense,
R Rotation). Phase 5's f64 p = 8 reference is its Rotation run ("The M2L strategy at p = 8"),
since Phase 5N's rule picks Rotation there; at f32 p = 8 the rule picks Rotation above 8
(locust) or 6 (M3 Max) ranks where Phase 5 ran Dense, and those cells compare strategies.
locust's 72-rank f32 reference is its repeat (370.5 ms; the first run, 755 ms, met a
system job).

**locust** (2026-10-15, 09:56–10:37; idle but for these launches, the load in each file):

| problem | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 72 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube, f32 p = 8 | 5,350.6 (1.017, D) | 2,689.5 (1.016, D) | 1,348.3 (1.004, D) | 751.0 (1.020, D) | 473.7 (0.907, R) | 243.1 (0.626, R) | 130.3 (0.360, R) | 111.0 (0.300, R) |
| cube, f64 p = 3 | 2,489.9 (0.994, D) | 1,271.2 (1.006, D) | 638.0 (1.010, D) | 321.0 (1.016, D) | 159.5 (0.992, D) | 82.0 (0.988, D) | 44.0 (0.970, D) | 37.8 (0.922, D) |
| cube, f64 p = 8 | 9,446.9 (1.011, R) | 4,749.1 (1.004, R) | 2,343.8 (0.998, R) | 1,182.8 (1.007, R) | 586.6 (0.969, R) | 297.6 (0.936, R) | 159.4 (0.892, R) | 138.8 (0.849, R) |
| Plummer, f32 p = 8 | 14,027.5 (0.988, D) | 7,008.8 (0.976, D) | 3,586.9 (0.984, D) | 1,866.8 (0.985, D) | 1,114.5 (0.931, R) | 577.8 (0.695, R) | 291.7 (0.417, R) | 267.7 (0.388, R) |
| Plummer, f64 p = 3 | 4,065.5 (1.014, D) | 2,038.9 (1.001, D) | 1,034.7 (0.996, D) | 524.6 (1.007, D) | 261.3 (0.928, D) | 138.1 (0.930, D) | 75.4 (0.836, D) | 75.1 (0.933, D) |
| Plummer, f64 p = 8 | 19,468.1 (0.990, R) | 9,705.4 (0.990, R) | 4,951.5 (0.977, R) | 2,507.0 (0.996, R) | 1,239.1 (0.950, R) | 626.6 (0.900, R) | 327.4 (0.782, R) | 291.8 (0.819, R) |

**M3 Max** (2026-10-15, 10:00–10:38; outside the sandbox; the other processes at
28–194 % CPU at launch, median 65 %, 360 s of waits in all):

| problem | 1 | 2 | 4 | 8 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cube, f32 p = 8 | 4,794.0 (0.997, D) | 2,451.5 (0.997, D) | 1,280.2 (0.999, D) | 775.0 (0.718, R) | 578.1 (0.450, R) |
| cube, f64 p = 3 | 1,154.4 (0.957, D) | 597.8 (1.003, D) | 313.2 (1.010, D) | 168.9 (0.998, D) | 124.3 (0.937, D) |
| cube, f64 p = 8 | 6,546.9 (1.017, R) | 3,335.2 (1.013, R) | 1,719.3 (1.019, R) | 924.5 (1.079, R) | 694.4 (1.009, R) |
| Plummer, f32 p = 8 | 12,049.1 (0.994, D) | 6,185.5 (0.995, D) | 3,257.6 (0.982, D) | 1,973.3 (0.908, R) | 1,284.9 (0.560, R) |
| Plummer, f64 p = 3 | 2,433.3 (0.997, D) | 1,254.2 (1.006, D) | 653.8 (1.001, D) | 348.3 (1.008, D) | 244.5 (0.950, D) |
| Plummer, f64 p = 8 | 14,623.3 (1.000, R) | 7,495.0 (1.006, R) | 3,895.9 (1.014, R) | 2,167.4 (1.063, R) | 1,601.3 (0.991, R) |

- **Within run-to-run noise up to 8 ranks wherever the strategy is the same**: 0.976–1.020
  on locust, 0.957–1.019 on the M3 Max, except f64 p = 8 at 8 M3 Max ranks (1.063–1.079).
- **Faster where many ranks share the tree**: on locust from 16 ranks, 0.78–0.97 at 64–72
  ranks for the same strategy (the work-weighted cut and the owned top tree), and
  0.30–0.42 at f32 p = 8, where Phase 5N's rule picks Rotation over Phase 5's Dense.
- **The 8-rank M3 Max cells, explained in part**: the Phase 5 tree (06016d2) rebuilt and run
  in A/B/A/B order with this branch on the same day (8 ranks, f64 p = 8, Rotation, overlap
  off, three rounds each): the cube 880, 931, 935 ms against 891, 909, 906 ms; the Plummer
  sphere 2,112, 2,126, 2,144 against 2,052, 2,097, 2,092 ms. Today's Phase 5 binary is itself
  about 5 % slower than its report (machine state), and this branch is 1.6–2.7 % slower
  than it, all of it in the downward pass's local compute (mean 748 against 725 ms, 1,352
  against 1,320 ms) on the same level calls and work: not traced (as T6's locust anomaly,
  a binary or layout effect is the candidate).
- **locust's control, A/B/A at 8 and 72 ranks** (this branch, the Phase 5 tree, this branch
  again; `RANKS="8 72"`): at 8 ranks this branch is within −0.3 % to +1.4 % of Phase 5
  (second run), and its local upward pass 6–7 % faster (22.9–27.8 against 24.5–29.6 ms):
  **T6's open locust anomaly (that pass 11 % slower) does not recur.** At 72 ranks this
  branch is 5–16 % faster (one Phase 5 run, the cube at f64 p = 8, met a straggler: 270 ms,
  its report 164 ms).

## 10. The time model beyond 160 ranks (*model*)

`scale-out.md` §10.2's model with this task's inputs; every figure here is a *model*.
40 ranks per node, f64 p = 8 (Rotation) and f32 p = 3 (Dense), the blocking path, the
second cut on. The inputs, each stated:
- **compute**: weak, the measured compute per rank at 160 ranks and 2.5 × 10⁴ points per
  rank (§3, job 604159: the mean and the measured max/mean), held fixed in P because the
  modelled max/mean at 512–640 counted ranks (1.02–1.10, §5) is that at 160 (1.04–1.10);
  strong, the measured mean compute per point at the nearest per-rank sizes on one node
  (§3 and `t9-model`, 40 ranks: the Plummer sphere 45–52 µs at every size from 10⁴ to 10⁶
  points per rank; the cube 21–23 µs at 2.5 × 10⁴–6.25 × 10⁵, but 50 µs on the small tree of
  the 10⁴-per-rank run, whose last level is half refined, with W and X lists and their M2P and
  P2L, §11: a range) times the measured
  max/mean at 160 ranks (1.06 cube, 1.08 Plummer);
- **exchanges**: the measured blocking source and multipole exchanges at 160 ranks (bytes per
  rank bounded in P at fixed N/P, §5; their rate 0.33–0.73 GB/s per rank, the same on one
  node as on four), scaled by the surface ((N/P)^(2/3)) for strong scaling, plus the busiest
  sender's extra messages beyond 160 at 4 µs each from T8s's fits (power law / linear,
  phase5s-t8s.md §7.2);
- **R**: the flat all-gather-v of §1.2 (even owners; the uneven owners measured 1.5–3× that
  at 160, §1.3, so this is a lower bound);
- **reduction**: below 0.2 ms (13–31 kB, 13–35 messages up to 640 ranks, §5);
- **redistribution**: point to point on the build's routes (S5), measured 0.8–1.4 ms forward
  at 160 and bounded by the neighbours.

**Weak scaling, 2.5 × 10⁴ points per rank** (evaluation in ms; 160 measured, the rest
*model*):

| precision | workload | 160 (measured) | 10³: compute + exchanges + messages + R = total | 10⁴: compute + exchanges + messages + R = total | 10⁴ against 160 |
| --- | --- | ---: | --- | --- | ---: |
| f64 p = 8 | cube | 1,174.7 | 1,158.7 + 6.6 + 0.2–0.3 + 2.4–3.7 + 2 = 1,170 | 1,158.7 + 6.6 + 0.8–4.2 + 12–24 + 2 = 1,180–1,196 | 1.00–1.02 |
| f64 p = 8 | Plummer | 1,278.5 | 1,248.2 + 9.7 + 0.2–0.3 + 2.4–3.7 + 2 = 1,263 | 1,248.2 + 9.7 + 0.5–3.3 + 12–24 + 2 = 1,272–1,287 | 1.00–1.01 |
| f64 p = 8 | clusters | 1,427.8 | 1,406.7 + 9.1 + 0.7–1.0 + 2.4–3.7 + 2 = 1,422 | 1,406.7 + 9.1 + 3.2–11.5 + 12–24 + 2 = 1,433–1,453 | 1.00–1.02 |
| f32 p = 3 | cube | 57.0 | 52.9 + 1.1 + 0.2–0.3 + 0.7 + 1.6 = 56.5 | 52.9 + 1.1 + 0.7–4.0 + 6.3 + 1.6 = 62.6–65.9 | 1.10–1.16 |
| f32 p = 3 | Plummer | 148.6 | 141.2 + 1.9 + 0.3–0.4 + 0.7 + 1.7 = 145.8 | 141.2 + 1.9 + 0.7–4.2 + 6.3 + 1.7 = 151.8–155.3 | 1.02–1.05 |
| f32 p = 3 | clusters | 189.1 | 182.3 + 1.7 + 0.7–0.9 + 0.7 + 1.7 = 187.2 | 182.3 + 1.7 + 2.7–10.7 + 6.3 + 1.7 = 194.7–202.7 | 1.03–1.07 |

(The 160 measured includes the waits for the slowest rank that the sum of maxima leaves
out, hence 160 slightly above the 10³ sum.) Against Phase 5's model at 8,192 ranks
(scale-out.md §10.4: cube 1,640 ms, Plummer 2,712 ms at f64 p = 8, the global pass and the
coarse gather alone 0.83–1.05 s), Phase 5S's evaluation stays within 2 % of its 160-rank
time at f64 p = 8. **At f32 p = 3 the flat all-gather of R becomes the largest term that
grows** (6.3 ms at 10⁴, 4–11 % of an evaluation), with the busiest sender's messages
(up to 10.7 ms on the clusters, overlap hides them); the hierarchical gather of §1.2
(*model* 0.55 ms) would remove most of the first.

**Strong scaling, N = 10⁸** (f64 p = 8, 40 × 1; the evaluation in ms; 160 measured, the
rest *model*; efficiency against 160 ranks):

| workload | 160 (measured) | 10³ (10⁵ per rank) | eff. | 10⁴ (10⁴ per rank) | eff. |
| --- | ---: | --- | ---: | --- | ---: |
| cube | 14,370.8 | 2,150 × 1.06 + 10 + 1 + 2.4–3.7 = 2,290 | 1.00 | (215–500) × 1.06 + 2.2 + 1–4.5 + 12–24 + 2 = 245–563 | 0.41–0.94 |
| Plummer | 32,619.5 | 5,000 × 1.08 + 29 + 1 + 2.4–3.7 = 5,430 | 0.96 | 500 × 1.08 + 6.1 + 1–3.7 + 12–24 + 2 = 561–576 | 0.91–0.93 |

- **To 10³ ranks the evaluation scales** at N = 10⁸ (efficiency 0.96–1.00): the per-point
  cost is flat in the per-rank size, communication 0.6 % of an evaluation.
- **At 10⁴ ranks (10⁴ points per rank)** the Plummer sphere keeps 0.91–0.93; the cube
  0.94 if its per-point cost stays at the large trees' 21.5 µs and 0.41 if it rises to the
  small tree's 50 µs (a half-refined level's M2P and P2L, §11): the per-point cost at small
  N/P, not communication, decides it, and the
  cube's must be measured on a large tree at that per-rank size (the 4-node limit allows
  10⁴ points per rank only at N = 1.6 × 10⁶). R's flat all-gather (12–24 ms) is 2–10 % of
  these evaluations: in strong scaling to 10⁴ ranks it is the remedy of §1.2 that would
  matter first, at f64 p = 8 as well.

Not in the model (open, §11): contention beyond the measured per-node rate when hundreds of
nodes exchange, Open MPI's algorithm choice at 10⁴ ranks, the network's topology, the
4-node transport stalls (handled by `PSM2_ERRCHK_TIMEOUT`, unmeasured beyond 4 nodes), the
build (S4/S5 left O(P) terms: O1's and the weight rounds' all-reduces, *model* about 57 ms
per build at 10⁴, T8s §9), and the device.

## 11. Open

Recorded, not built (the brief's "Record, not build" and this task's findings):
- **The replicated top's all-gather at scale** (§1): flat measured fastest within 4 nodes
  but linear in P (*model* 6.3 ms at f32 p = 3 and 12–24 ms at f64 p = 8 at 10⁴ ranks);
  the hierarchical gather (no broadcast; *model* 0.55 and 3.8 ms; break-even 700 and
  1,000–2,400 ranks) and the two-level gather of the brief (measured 3–6 ms at 160 ranks,
  its node broadcast the cost) are designed and micro-benchmarked (`collectives`), to be
  built and measured on a larger cluster. R's uneven owners make the gather 1.5–3× the even
  split's time (§1.3).
- **Overlap's slowdown at 20 threads per rank** (§7): one rank's downward pass, intermittent;
  the PSM2 receive thread a candidate; a longer A/B and the sweeps with
  `PSM2_RCVTHREAD=0` before overlap can become the default; and progress during device
  execution before overlap can pay on GPU nodes.
- **The blocking tree's receive copy** (§1.3): 1.2–1.3 ms per evaluation at f64 p = 8;
  receiving in place where a message's slots are contiguous.
- **Pacing dense bursts in rlst's sparse exchanges** (the 4-node packet losses, T8 §9.3;
  upstream); **the octree's O1 and weight-round all-reduces** (the largest O(P) build term
  left, *model* about 57 ms per build at 10⁴ ranks); **`ReductionExchange::new` on two
  sparse exchanges** as the tree's build; **level aggregation** (it would expose the
  overlapped path's per-level waits; T8s §2.2).
- **The imbalance at f32 p = 3** (§3: compute max/mean 1.06–1.18 at 160 ranks with
  2.5 × 10⁴ points per rank against a modelled 1.04–1.07; T8s's 26–42 % of wall time at its
  configuration): the cost model underweights P2P there.
- **The cube's leaf time at 80 Kathleen ranks, explained** (T8s job 604104: 1,068 ms at
  f64 p = 8 against 88 at 40 and 72 at 160): reproduced here (weak, 40 × 1: 1,083 ms at 80
  ranks against 89 at 40 and 70 at 160; f32 p = 3 291 against 31 and 27 ms). At
  N = 2 × 10⁶ the cube's tree is half refined to a seventh level, so W and X lists appear:
  12,688 W pairs per rank on average, **767,220 M2P evaluations on the busiest rank** in the
  leaves stage (and 787,413 P2L in the downward pass, 932 against 500 ms), none at 10⁶ or
  4 × 10⁶ points, whose trees are uniform. At f64 p = 8 an M2P evaluation costs about
  1.3 µs (1 s for 767,220), about 500 times a P2P pair (24.5 × 10⁶ pairs in about 60 ms),
  while a W-list box on the refined level holds at most 64 sources: the multipole is far
  dearer than the direct sum it replaces. **For Phase 6**: M2P and P2L only where a box
  holds more points than the expansion's cost (P2P for the W and X lists below that), an
  operator-level change outside this phase's scope (the README's "Out of scope").
- **The M3 Max's downward pass at 8 ranks** 1.6–2.7 % slower than the Phase 5 binary's on
  the same work (§9), cause not traced; locust's open T6 anomaly closed (§9).
- **Every timing beyond 160 ranks** (§10): contention across hundreds of nodes, collective
  algorithms at 10⁴ ranks, the cube's per-point cost at 10⁴ points per rank on a large tree.
- **node-c11a-011's Omni-Path contexts** are busy (seen in T2, job 447359, and here, job
  604193: 40 ranks per node fail in `MPI_Init`, 2 per node start); later 4-node jobs used
  `--exclude=node-c11a-011`.

## 12. Jobs, node-hours and checks

**Kathleen** (2026-10-15, `sacct`; node-hours are nodes × elapsed; every launch under
`timeout`; Open MPI 4.1.6 over PSM2 with this branch's `tools/kathleen/env.sh`):

| Job | Nodes, QoS, nodes used | What | Waited | Ran | Node-hours |
| --- | --- | --- | ---: | ---: | ---: |
| 604144 | 4, `small`, node-c11a-[038,080,109,117] | step 1: the all-gather micro-benchmark (40/80/160 ranks, default and forced algorithms, HAN), the FMM at 160 timed (with the copy probe) and 640 counted | 0:15:59 | 8:32 | 0.57 |
| 604149 | 4, `small`, node-c11a-[007,019,024,085] | step 1: the node broadcast's forced algorithms at 160 | 0:02:19 | 0:27 | 0.03 |
| 604150 | 4, `small`, node-c11a-[015,127-128,140] | step 1: the hierarchical gather at 40/80/160 | 0:00:19 | 1:13 | 0.08 |
| 604157 | 1, `singlenode`, node-c11b-010 | Phase 5S: `t9-strong` (N = 10⁷), `t9-weak`, `t9-model` | 0:00:01 | 1:41:51 | 1.70 |
| 604158 | 1, `singlenode`, node-c11b-028 | Phase 5S: `t9-strong` (N = 10⁸) | 0:03:24 | 2:42:39 | 2.71 |
| 604160 | 2, `small`, node-c11a-[121,125] | Phase 5S: `t9-strong`, `t9-weak` | 0:01:40 | 2:02:38 | 4.09 |
| 604159 | 4, `small`, node-c11b-[024-027] | Phase 5S: `t9-strong`, `t9-weak`, `t9-counts` (320, 640) | 0:29:26 | 1:16:05 | 5.07 |
| 604193 | 4, `small`, node-c11a-[010-011,047,056] | Phase 5: `t9-strong` (N = 10⁷; 10⁸ at f32 and f64 p = 8), `t9-weak`; the 40 × 1 launches failed on node-c11a-011 (§11) | 0:03:03 | 38:16 | 2.55 |
| 604198 | 4, `small`, node-c11b-[024-027] | Phase 5: the 40 × 1 launches again, `--exclude=node-c11a-011` (the nodes of 604159) | 0:01:29 | 24:12 | 1.61 |
| 604229 | 2, `small`, node-c11b-[024-025] | Phase 5: `t9-strong` (N = 10⁷; 10⁸ at f32 p = 8), `t9-weak` | 0:01:47 | 1:05:20 | 2.18 |
| 604197 | 1, `singlenode`, node-c11a-115 | Phase 5: `t9-strong` (N = 10⁷; 10⁸ at f32 p = 8), `t9-weak` | 0:00:01 | 2:55:40 | 2.93 |
| 604225 | 1, `singlenode`, node-c11b-023 | the overlap diagnostic: `PSM2_RCVTHREAD` A/B/A/B (§7) | 0:49:02 | 32:15 | 0.54 |
| 604230 | 2, `test`, node-c11a-[025,032] | T2's correctness job (MPI list at 2, 4, 8, 80; invariance at 80; smoke) | 0:15:59 | 23:51 | 0.80 |
| 604231 | 4, `small`, node-c11a-[031,034,056,062] | invariance at 160 and 640 (`correctness.sbatch invariance oversubscribed`) | 0:25:23 | 1:25 | 0.09 |
| | | | | **total** | **24.95** |

Within the task's budget of 60 node-hours (README decision 6; the exposed collectives'
measurements 0.68 of them). The Phase 5 tree is `06016d2` (T2 merged: the evaluation of
Phase 5N, the octree of T3a; memory "Phase 5 against Phase 5S" of T6 and T7), built in
its own directory on Kathleen and run with this branch's `run.sh` and environment. Raw
outputs stay in Kathleen's `/scratch/scratch/ucahtbe/fmm/logs/` (`scaling-<job>`,
`p5-t9-<job>`, `p5s-t9-step1`, `p5s-t9-rcvthread-604225`) and locust's
`/data/ucahtbe/logs/p5s-t9/`.

**locust** (2026-10-15; idle but for these launches, the load before and after each in its
file): the strong sweep at 1–72 ranks (09:56–10:37), the A/B/A control at 8 and 72 against
the Phase 5 tree (10:37–10:44), the counts at 128, 256 and 512 (10:44–10:47), invariance at
512 against 64 (12:05–12:08, twice).

**M3 Max** (2026-10-15, outside the sandbox): the strong sweep at 1–12 ranks (10:00–10:38),
the A/B/A/B control at 8 ranks against the Phase 5 tree (about 10:45–11:05).

**Checks** (this branch; the library unchanged, the changes are `nd-fmm-validate`'s
harness, the `collectives` example, `tools/` and documents): `cargo fmt -- --check`,
`cargo clippy -- -D warnings`, `cargo clippy --examples -- -D warnings`, `cargo doc
--no-deps` (no warning), `cargo clippy --workspace --all-targets -- -D warnings` and
`RUST_MIN_STACK=8388608 cargo test --workspace` on the M3 Max: all pass (the workspace tests: 68 executables, 773 passed, 0 failed, 57 ignored);
`cargo test -p nd-fmm-validate` at one rank and `tests/scaling.rs` at 2 ranks pass; T2's
correctness job at 2 nodes passes (job 604230).

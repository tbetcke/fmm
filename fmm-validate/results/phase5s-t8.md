# Construction and collectives at scale (Phase 5S T8)

S4, S5 and S8 evaluated, as docs/phase5s/T8-construction-and-collectives.md asks, on the
final structure (main at e788576: T3a's routing, T3's partition map, T5/T6's owned top
tree, T7's second cut on by default). Steps 1 (measure) and 2 (decide) of the brief; step
3 (build what fires) waits for the sign-off of §6. No library code changed. Run on
2026-10-13 on locust (GH200) and Kathleen.

**Measured** means run here and read from the output; timings are reported, never
asserted. **Counted** runs are oversubscribed and never timed. ***Model*** marks every
figure beyond 160 ranks; its inputs are stated with it. locust is a shared node: the load
is in each output file before every launch (idle but for these launches). Kathleen nodes
are exclusive; the job is named in §7.

## Summary

- **S4 (built in T3a), re-measured: closed.** No sample sort is left in `Octree::new`.
  Routing the keys (one local sort, the weight rounds, one all-to-all-v of
  deduplicated pairs) takes 35–58 % of `Octree::new` (65 % for the cube at N = 10⁸), and
  `Octree::new` takes 7–12 % of the build (N = 10⁷, 8–72 locust ranks and 40–160
  Kathleen ranks, §2). Keys per rank after routing stay within 1.002–1.083 of the mean
  when timed and 1.02–1.12 at 128–640 counted ranks. The 2× imbalance clause of T1's trigger is far away. The
  weight rounds' and O1's all-reduces (about 0.9 MB per build at 640 ranks, growing
  like P) are the one O(P) term left in the construction (§2.3, *model* §3.3).
- **S5 for the build: on the edge on T1's own bound, does not fire on the measured
  trend.**
  - Every build runs 21–32 all-to-alls of P counts. 19–30 of them are inside rlst: 12–23
    in its ghost-communicator builds and 7 in its `array_tools::all_to_allv`, which the
    octree and the cost model call. Only `Redistribution::new`'s 2 are this repository's
    own. The build also makes 24–46 graph-communicator creations (§3).
  - Timed, the all-to-alls take 0.5–3 % of the build at 8–72 locust and 40–160
    Kathleen ranks.
  - *Model* at 10⁴ ranks, 2.5 × 10⁴ points per rank: 19–27 % of a modelled build with
    T1's linear α_a = 0.376 µs per peer, and 6–9 % with the measured trend 43.8 µs +
    0.102 µs per peer (40–160 ranks). The trigger is "a quarter of a modelled build".
- **S5 for the evaluation (signed off as firing in decision 4): fires, depending on the
  library.** Each evaluation runs two all-to-all-v's (charges forward, output backward)
  with P-long arrays.
  - Open MPI 4.1.6 chooses between basic linear and pairwise by a fixed rule. Measured
    with every count 0: 0.3 µs at 40, 167 µs at 80 (pairwise, about 2.1 µs per peer)
    and 1.2 µs at 160 Kathleen ranks (§4).
  - *Model* at 10⁴ ranks: 0 ms (basic linear), 7.5 ms (T1's α_v) or 42 ms (pairwise)
    per evaluation. The threshold is 1 % of an evaluation at 10⁴ points per rank,
    4.7–5.7 ms at f64 p = 8 and 0.22–0.74 ms at f32 p = 3.
  - The remedy (point to point on the routes fixed at build) is cheap and makes the
    cost independent of that choice.
- **S8 (the fan-out below R): fires on the modelled clause, not on the measured one.**
  - The busiest sender's multipole messages grow with P: 41–93 at 40 ranks, 113–486 at
    512–640 ranks. The clusters grow almost linearly (P^0.71–0.75), the cube and the
    Plummer sphere slowly (P^0.28–0.38) (§5).
  - *Model* at 10⁴ ranks and 4 µs per message: 1.4–15 ms (power-law fits) or 5.7–28 ms
    (linear fits). The thresholds are 1 % of the evaluation at 2.5 × 10⁴ points per
    rank: 5.5–14 ms at f64 p = 8 and 0.56–1.85 ms at f32 p = 3.
  - It fires at f32 p = 3 on all three workloads, from about 600–900 ranks on the
    clusters and the cube. At f64 p = 8 it fires within 10⁴ ranks only under the linear
    fits (the clusters from about 6,000 ranks, the cube from 9,000), or on the clusters'
    power law against the cube's 5.5 ms.
  - Measured at 160 Kathleen ranks, the overlapped path exposes 0.002–0.013 ms of the
    multipole exchange per evaluation, at most 0.004 % of the evaluation. The second
    clause does not fire.
- **Questions for sign-off: §6.** Kathleen used **1.99 node-hours** in one 4-node job
  (§7). The measurement tool was a throwaway, instrumented copy of the tree outside the
  repository (§1).

## 1. Method

**The probe** (not committed; the brief's "for a package not built: nothing else
changed"). A copy of e788576 in the session's scratch directory, built separately, with:
- **Named sites.** A stack of names on the calling thread (`nd_octree::probe` in the copy)
  around every step of `Octree::new` (route keys, O1 rounds, key move, local tree,
  balance, leaf-bit all-gather, push, neighbours), of `Octree::repartition`, of
  `Fmm::build` (settings, `split_shared`, domain, `Octree::new`, `Plan::new`, the second
  cut with `leaf_point_counts` and its `SourceExchange<u8>`, both `Redistribution::new`,
  the coordinates' forward, tables, `Evaluator::new` with each exchange's build) and of
  `Fmm::evaluate` (forward charges, backward output). Each site records its wall time,
  and the steps of `Octree::new` and the build parts their peak resident size (`VmHWM`,
  reset on entry). With `PROBE_BARRIERS=1` each step of `Octree::new` starts with a
  barrier (as T1's and T3a's replays did); the barriers are their own site and are
  excluded from every count.
- **An MPI interposer.** The copied `scaling` example defines the `MPI_*` functions the
  workspace and rlst call (the collectives, graph creation, `Comm_dup`/`split`,
  point-to-point and waits) and forwards each to `PMPI_*`, recording under the current
  site: calls, logical messages to other ranks (an all-to-all or all-gather counts P − 1,
  an all-to-all-v its non-zero counts to others, a neighbour all-to-all-v its non-zero
  neighbours), bytes sent to and received from other ranks, the bytes of the P-long
  argument arrays, all-reduce payloads, and the time inside the call (which includes
  waiting for the other ranks). It is `unsafe` FFI, acceptable in a throwaway
  measurement tool and nowhere in the repository; the
  brief's "no `unsafe`, no raw MPI FFI" binds what this task builds.
- **Micro-benchmarks** (`PROBE_MICRO=1`): the median of 30 repetitions of the slowest
  rank's time, each after a barrier, at one rank per core.

The site and call tables are reduced over the ranks (max and mean) and printed after the
harness's own report; the `scaling` harness is unchanged otherwise. The resets of
`VmHWM` make the harness's own "resident" column the peak since the last step, not since
the start: it is not used here.

**Runs** (release; every BLAS thread variable at 1; the strategy named; the second cut on,
as default):
- *Timed*, one rank per core: locust 8, 32, 72 ranks (`--map-by slot:PE=1 --bind-to
  core`) and Kathleen 40, 80, 160 (40 × 1 per node, `--map-by ppr:20:socket:PE=1
  --bind-to core`), the cube, the Plummer sphere and the clusters at N = 10⁷, f64 p = 8
  Rotation, barriers before the octree steps, three evaluations; Kathleen at 160 also
  N = 10⁸ (cube, Plummer, one evaluation). At 2.5 × 10⁴ points per rank, f64 p = 8
  Rotation and f32 p = 3 Dense, both paths, at locust 8, 16, 32, 64, 72 (three
  evaluations) and Kathleen 40, 80, 160 (five).
- *Counted*, oversubscribed: locust 128, 256, 512 (`--map-by :OVERSUBSCRIBE --bind-to
  none`) and Kathleen 320, 640 (`--oversubscribe --map-by ppr:<P/4>:node --bind-to none
  -x PSM2_SHAREDCONTEXTS=1`), N = 10⁷ (f64 p = 8) and 2.5 × 10⁴ points per rank (both
  precisions), one evaluation.
- *Micro-benchmarks*: locust 8–72, Kathleen 40, 80, 160 ranks.

Every launch exited 0: locust 71 (5 micro-benchmarks, 39 timed, 27 counted), Kathleen 50 (3,
29, 18).

## 2. S4: the construction, re-measured

### 2.1 `Octree::new` by step, timed

The slowest rank's time per step (ms), a barrier before each step; "build" is the whole
`Fmm::build`; keys after routing and the peak during routing over the ranks.

locust, N = 10⁷:

| ranks | workload | route keys | O1 rounds | key move | local tree | balance | leaf bits | push | neighbours | `Octree::new` | build | keys after routing, max / mean | peak during routing, max / mean (MB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | cube | 76.3 | 9.44 | 3.95 | 18.4 | 4.04 | 0.020 | 8.25 | 30.0 | 153 | 1,746 | 1.002 | 200 / 200 |
| 8 | Plummer | 83.7 | 16.4 | 3.87 | 19.3 | 8.85 | 0.021 | 24.1 | 74.2 | 232 | 2,601 | 1.019 | 201 / 201 |
| 8 | clusters | 83.2 | 19.2 | 3.92 | 22.7 | 9.39 | 0.024 | 21.3 | 78.4 | 240 | 2,727 | 1.030 | 201 / 200 |
| 32 | cube | 27.6 | 2.68 | 1.74 | 3.78 | 1.35 | 0.032 | 4.17 | 6.00 | 49.4 | 448 | 1.005 | 93.6 / 90.1 |
| 32 | Plummer | 29.9 | 6.14 | 1.74 | 4.80 | 3.06 | 0.036 | 18.5 | 17.7 | 83.2 | 1,079 | 1.043 | 94.4 / 90.1 |
| 32 | clusters | 29.6 | 6.27 | 1.74 | 5.44 | 2.95 | 0.039 | 11.9 | 16.9 | 75.3 | 1,101 | 1.052 | 93.1 / 90.4 |
| 72 | cube | 16.2 | 2.43 | 1.30 | 1.58 | 0.828 | 0.095 | 2.93 | 2.25 | 28.6 | 295 | 1.017 | 55.5 / 52.9 |
| 72 | Plummer | 18.3 | 3.70 | 1.27 | 2.11 | 1.92 | 0.079 | 12.8 | 8.03 | 48.8 | 575 | 1.060 | 55.8 / 53.3 |
| 72 | clusters | 17.2 | 3.77 | 1.33 | 2.25 | 1.74 | 0.135 | 8.62 | 6.81 | 42.6 | 548 | 1.044 | 55.9 / 53.4 |

Kathleen, N = 10⁷:

| ranks | workload | route keys | O1 rounds | key move | local tree | balance | leaf bits | push | neighbours | `Octree::new` | build | keys after routing, max / mean | peak during routing, max / mean (MB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 40 | cube | 49.9 | 5.20 | 4.74 | 7.56 | 3.13 | 0.352 | 8.11 | 8.06 | 90.7 | 987 | 1.020 | 56.8 / 53.1 |
| 40 | Plummer | 52.8 | 11.0 | 4.73 | 7.63 | 5.54 | 0.406 | 35.5 | 24.9 | 143 | 1,808 | 1.039 | 58.9 / 54.9 |
| 40 | clusters | 53.4 | 10.6 | 4.26 | 8.09 | 4.92 | 0.386 | 19.3 | 22.8 | 127 | 1,836 | 1.050 | 61.1 / 55.1 |
| 80 | cube | 35.4 | 4.03 | 2.22 | 2.67 | 1.98 | 0.393 | 5.79 | 3.91 | 61.4 | 511 | 1.016 | 58.2 / 56.1 |
| 80 | Plummer | 33.6 | 7.87 | 3.07 | 3.96 | 4.07 | 0.406 | 24.4 | 12.6 | 91.7 | 1,046 | 1.043 | 59.7 / 57.2 |
| 80 | clusters | 34.1 | 6.15 | 2.40 | 3.61 | 3.51 | 0.424 | 15.5 | 11.0 | 80.2 | 989 | 1.059 | 61.9 / 57.0 |
| 160 | cube | 21.5 | 3.10 | 1.13 | 1.27 | 1.66 | 0.405 | 4.05 | 1.88 | 36.8 | 321 | 1.025 | 53.1 / 50.2 |
| 160 | Plummer | 27.4 | 6.18 | 1.18 | 1.55 | 2.81 | 0.397 | 15.7 | 6.70 | 64.1 | 615 | 1.064 | 53.0 / 50.8 |
| 160 | clusters | 25.3 | 6.57 | 1.27 | 1.69 | 2.73 | 0.378 | 12.9 | 5.76 | 57.6 | 559 | 1.073 | 54.8 / 50.9 |

Kathleen, N = 10⁸, 160 ranks:

| ranks | workload | route keys | O1 rounds | key move | local tree | balance | leaf bits | push | neighbours | `Octree::new` | build | keys after routing, max / mean | peak during routing, max / mean (MB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 160 | cube | 170 | 17.7 | 12.5 | 15.6 | 5.14 | 0.398 | 15.1 | 19.7 | 262 | 2,492 | 1.018 | 131 / 125 |
| 160 | Plummer | 191 | 25.5 | 13.7 | 20.2 | 15.3 | 0.434 | 117 | 98.8 | 484 | 6,339 | 1.083 | 132 / 126 |

- Routing (`route_keys`: the local sort of the input keys, the weight rounds, one
  all-to-all-v of (key, multiplicity) pairs and the merge) is 35–58 % of `Octree::new` at N = 10⁷ (39–65 % at N = 10⁸),
  and scales with N/P (49.9 → 21.5 ms from 40 to 160 Kathleen ranks on the cube; 170 ms
  at N = 10⁸ on 160). The push of the locally essential keys is the next largest on
  the Plummer sphere and the clusters (8.6–35.5 ms at N = 10⁷; 117 ms for the Plummer sphere at N = 10⁸).
- `Octree::new` is 7–12 % of the build at every timed count; the build is 11–23 % of one
  evaluation at N = 10⁷ (e.g. 559 of 3,502 ms, the clusters at 160 Kathleen ranks).
- The T3a numbers repeat (T3a §3: locust 8 ranks, the cube's routing 75.4 ms; here 76.3).

### 2.2 The intermediate imbalance and memory, counted

Keys per rank after routing (max over mean), the peak resident size during routing (MB,
max / mean), N = 10⁷, f64 p = 8: locust 128 / 256 / 512 ranks 1.03–1.10 / 1.06–1.09 /
1.08–1.12 (peaks 89–92, 57–58, 91–94 MB); Kathleen 320 / 640 ranks 1.03–1.08 / 1.02–1.09
(peaks 73–75, 104 MB, at 80 and 160 ranks per node). The key move after O1 leaves the same
ratio in every row.
**The S4 trigger (scale-out.md §7.1: the most keys a rank receives above twice the mean,
or the sorts above a quarter of `Octree::new`) does not hold: there are no sorts, and the
routing is within 1.12.**

### 2.3 Messages and bytes per step, counted

Per rank, max over the ranks, per build: Kathleen, 640 ranks, the clusters at 2.5 × 10⁴
points per rank, f64 p = 8 (the largest count; the other workloads and locust's 512 are in
the same range). "Collectives" counts every call (rlst's validity all-reduces included),
"messages" the logical point-to-point messages to other ranks (§1), "graph creations"
`MPI_Dist_graph_create_adjacent`.

| site | collectives | messages | sent kB | received kB | P-long arrays kB | all-reduce kB | all-to-alls of counts | graph creations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Evaluator::new | 1.00 | 0 | 0 | 0 | 0 | 0.0010 | 0 | 0 |
| MultipoleExchange::new | 98.0 | 8,167 | 129 | 129 | 73.4 | 0.244 | 12.0 | 24.0 |
| ReductionExchange::new | 66.0 | 5,130 | 20.8 | 20.8 | 41.4 | 0.232 | 8.00 | 16.0 |
| SourceExchange::new | 10.0 | 745 | 26.3 | 26.3 | 7.67 | 0.035 | 1.00 | 2.00 |
| TopExchange::new (R) | 13.0 | 2,677 | 437 | 457 | 31.0 | 0.259 | 1.00 | 2.00 |
| octree: balance (closure by the bounds) | 7.00 | 782 | 23.4 | 461 | 15.4 | 0.012 | 1.00 | 0 |
| octree: coarse tree (O1 rounds) | 10.0 | 0 | 0 | 0 | 0 | 702 | 0 | 0 |
| octree: generate_all_keys (push) | 6.00 | 694 | 68.4 | 44.6 | 15.4 | 0.0040 | 1.00 | 0 |
| octree: key move | 6.00 | 641 | 15.1 | 15.1 | 15.4 | 0.0040 | 1.00 | 0 |
| octree: leaf-bit all-gather | 1.00 | 639 | 0.420 | 19.4 | 5.12 | 0 | 0 | 0 |
| octree: route keys | 15.0 | 1,278 | 410 | 435 | 15.4 | 196 | 1.00 | 0 |
| Plan::new | 2.00 | 0 | 0 | 0 | 0 | 0.0090 | 0 | 0 |
| Redistribution::new (sources) | 4.00 | 1,278 | 310 | 625 | 20.5 | 0.016 | 1.00 | 0 |
| Redistribution::new (targets) | 4.00 | 1,278 | 310 | 625 | 20.5 | 0.016 | 1.00 | 0 |
| forward coordinates | 2.00 | 1,278 | 1,221 | 2,478 | 20.5 | 0 | 0 | 0 |
| cut: gather block weights | 1.00 | 639 | 3.36 | 156 | 5.12 | 0 | 0 | 0 |
| cut: generate_all_keys (push) | 6.00 | 697 | 56.4 | 43.0 | 15.4 | 0.0040 | 1.00 | 0 |
| cut: move leaves | 6.00 | 642 | 17.6 | 14.8 | 15.4 | 0.0040 | 1.00 | 0 |
| cut: Plan::new again | 2.00 | 0 | 0 | 0 | 0 | 0.0090 | 0 | 0 |
| cut: SourceExchange<u8>::new | 10.0 | 743 | 31.3 | 31.3 | 7.61 | 0.035 | 1.00 | 2.00 |
| cut: leaf_point_counts | 6.00 | 1,278 | 614 | 651 | 15.4 | 0.0040 | 1.00 | 0 |
| split_shared | 1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| step 1: settings agreed | 1.00 | 0 | 0 | 0 | 0 | 0.0010 | 0 | 0 |
| step 7: tables | 1.00 | 0 | 0 | 0 | 0 | 0.016 | 0 | 0 |
| steps 2-3: points, domain | 10.0 | 0 | 0 | 0 | 0 | 0.113 | 0 | 0 |
| **build** | 289 | 28,586 | 3,695 | 6,232 | 340 | 899 | 32.0 | 46.0 |

What grows with P in the construction: the P-long arrays of every all-to-all-v (16 P
bytes) with its all-to-all of counts, and the all-reduces of the weight rounds and O1's
rounds, an array over the provisional or coarse blocks (about k P of them): 196 + 702 kB
per build at 640 ranks on the clusters, 175 + 626 kB at 512, growing like P. At 10⁴ ranks
that is about 14 MB of all-reduce per build, about 57 ms at the 0.25 GB/s that an
all-reduce of 92 kB reached at 160 ranks (*model*; distributed-fmm.md §14.2 called O1's
rounds "acceptable": 4 MB per round at 8,192 ranks). Neither S4 nor S5 touches them.

## 3. S5 for the build: the all-to-alls of counts

### 3.1 Per build, against P

f64 p = 8 at 2.5 × 10⁴ points per rank; per rank, max over the ranks, per build; times only
for the timed counts ("in-call" is the time inside the calls, the wait for slower ranks
included).

| workload | machine | ranks | collectives | all-to-alls of counts | of them in rlst builds | graph creations | messages | all-reduce kB | P-long kB | build ms | a2a in-call ms | graphs in-call ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | Kathleen | 40 | 190 | 21.0 | 12.0 | 24.0 | 1,573 | 14.7 | 22.2 | 121 | 3.12 | 3.06 |
| cube | Kathleen | 80 | 208 | 23.0 | 14.0 | 28.0 | 3,155 | 113 | 42.1 | 148 | 3.45 | 3.61 |
| cube | Kathleen | 160 | 208 | 23.0 | 14.0 | 28.0 | 6,001 | 113 | 78.3 | 373 | 3.99 | 4.21 |
| cube | Kathleen | 320 | 194 | 21.0 | 14.0 | 28.0 | 10,969 | 113 | 134 | (not timed) | – | – |
| cube | Kathleen | 640 | 226 | 25.0 | 16.0 | 32.0 | 24,127 | 900 | 302 | (not timed) | – | – |
| cube | locust | 128 | 208 | 23.0 | 14.0 | 28.0 | 4,846 | 113 | 63.7 | (not timed) | – | – |
| cube | locust | 256 | 208 | 23.0 | 14.0 | 28.0 | 9,364 | 113 | 121 | (not timed) | – | – |
| cube | locust | 512 | 226 | 25.0 | 16.0 | 32.0 | 19,416 | 555 | 244 | (not timed) | – | – |
| Plummer | Kathleen | 40 | 263 | 29.0 | 20.0 | 40.0 | 1,998 | 111 | 27.0 | 199 | 3.58 | 3.43 |
| Plummer | Kathleen | 80 | 263 | 29.0 | 20.0 | 40.0 | 3,762 | 168 | 48.6 | 234 | 3.75 | 4.22 |
| Plummer | Kathleen | 160 | 281 | 31.0 | 22.0 | 44.0 | 7,402 | 396 | 90.6 | 242 | 6.56 | 4.72 |
| Plummer | Kathleen | 320 | 281 | 31.0 | 22.0 | 44.0 | 14,420 | 689 | 174 | (not timed) | – | – |
| Plummer | Kathleen | 640 | 281 | 31.0 | 22.0 | 44.0 | 28,303 | 1,212 | 338 | (not timed) | – | – |
| Plummer | locust | 128 | 263 | 29.0 | 20.0 | 40.0 | 5,772 | 239 | 72.6 | (not timed) | – | – |
| Plummer | locust | 256 | 281 | 31.0 | 22.0 | 44.0 | 11,689 | 594 | 142 | (not timed) | – | – |
| Plummer | locust | 512 | 281 | 31.0 | 22.0 | 44.0 | 22,757 | 1,024 | 272 | (not timed) | – | – |
| clusters | Kathleen | 40 | 263 | 29.0 | 20.0 | 40.0 | 1,803 | 123 | 23.5 | 209 | 3.14 | 3.49 |
| clusters | Kathleen | 80 | 271 | 30.0 | 21.0 | 42.0 | 3,599 | 169 | 45.4 | 218 | 6.86 | 4.18 |
| clusters | Kathleen | 160 | 281 | 31.0 | 22.0 | 44.0 | 7,240 | 370 | 89.2 | 241 | 5.97 | 4.83 |
| clusters | Kathleen | 320 | 281 | 31.0 | 22.0 | 44.0 | 14,179 | 607 | 172 | (not timed) | – | – |
| clusters | Kathleen | 640 | 289 | 32.0 | 23.0 | 46.0 | 28,586 | 899 | 340 | (not timed) | – | – |
| clusters | locust | 128 | 281 | 31.0 | 22.0 | 44.0 | 5,900 | 310 | 73.5 | (not timed) | – | – |
| clusters | locust | 256 | 281 | 31.0 | 22.0 | 44.0 | 11,544 | 521 | 142 | (not timed) | – | – |
| clusters | locust | 512 | 281 | 31.0 | 22.0 | 44.0 | 22,693 | 801 | 276 | (not timed) | – | – |

- **21–32 all-to-alls of P counts per build**, nearly independent of P (one more per
  level of the multipole and reduction exchanges as the tree deepens): 12–23 of them in
  rlst's `GhostCommunicatorBuilder::build` (one per build of the source exchange, of the
  top exchange, of the source exchange of the second cut, and one per level of the
  multipole and reduction exchanges), 7 in rlst's `array_tools::all_to_allv` (the
  routing, the key move, the balance and the push of `Octree::new`, `leaf_point_counts`,
  the second cut's leaf move and push; each with 4 validity all-reduces), and 2 in this
  repository's own code (the two `Redistribution::new`). rlst's builds also make two graph communicators each
  (24–46 per build) and run three validity all-reduces each.
- Timed: 3–7 ms per build at 40–160 Kathleen ranks at 2.5 × 10⁴ points per rank (1.1–3.1 %
  of the build; 4–26 ms and 0.5–2.0 % at N = 10⁷ on both machines); graph creations
  3.1–4.8 ms at 2.5 × 10⁴ per rank.

### 3.2 The collectives, micro-benchmarked

µs, the median of 30 of the slowest rank (locust: Open MPI 5.0.10 over shared memory;
Kathleen: Open MPI 4.1.6 over PSM2, 40 ranks per node). "8 peers" is 7 at locust's 8
ranks.

| operation | locust 8 | 72 | Kathleen 40 | 80 | 160 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPI_Alltoall, one i32 per peer | 4.5 | 10.0 | 47.9 | 50.1 | 60.1 |
| MPI_Alltoall, one u64 per peer | 1.9 | 9.8 | 27.0 | 41.7 | 54.0 |
| MPI_Alltoallv, every count 0 | 0.1 | 0.5 | 0.3 | 167.0 | 1.2 |
| MPI_Alltoallv, 2 peers x 8192 B | 5.6 | 8.4 | 17.1 | 111.3 | 17.2 |
| point to point (Isend/Irecv, wait), 2 peers x 8192 B | 5.6 | 6.8 | 24.1 | 23.4 | 26.0 |
| MPI_Alltoallv, 8 peers x 8192 B | 18.4 | 31.1 | 92.0 | 162.1 | 76.8 |
| point to point (Isend/Irecv, wait), 8 peers x 8192 B | 18.6 | 26.1 | 58.8 | 64.2 | 58.8 |
| MPI_Alltoallv, 8 peers x 65536 B | 53.8 | 150.2 | 115.1 | 352.8 | 185.7 |
| point to point (Isend/Irecv, wait), 8 peers x 65536 B | 53.4 | 154.6 | 112.4 | 172.8 | 179.9 |
| MPI_Barrier | 1.3 | 3.8 | 8.0 | 10.1 | 13.3 |
| MPI_Ibarrier, then wait | 1.8 | 5.0 | 9.9 | 17.4 | 20.5 |
| MPI_Allreduce, 8 B | 1.5 | 3.8 | 11.7 | 12.7 | 15.9 |
| MPI_Allreduce, 2B u64 at B = 36 P (16 · 36 P bytes) | 8.1 | 64.6 | 74.6 | 266.9 | 373.3 |
| MPI_Allgather, 36 bools per rank (leaf bits) | 1.7 | 10.2 | 25.3 | 35.0 | 45.2 |

- The all-to-all of one count per peer: 47.9, 50.1, 60.1 µs at 40, 80, 160 Kathleen ranks
  (T1 §10.3: 47.6, 52.0, 59.8). Fitted: **43.8 µs + 0.102 µs × P**; T1's model used
  **0.376 µs × P** (the 160-rank figure per peer), an upper bound if the cost is linear.
- An all-to-all-v with every count 0 costs 0.3 and 1.2 µs at 40 and 160 Kathleen ranks
  and **167 µs at 80**, as do its other rows at 80 (111 µs for 2 peers against 17): Open
  MPI 4.1.6's `tuned` component chooses between basic linear (which skips zero counts) and
  pairwise (P − 1 exchanges, 2.1 µs per peer here) by a fixed rule on the communicator
  size (`coll_tuned_alltoallv_algorithm` is "ignore": not set). Point to point to the
  same peers costs the same or less at every count (23–26 µs for 2 peers).

### 3.3 The model at 10³ and 10⁴ ranks (*model*)

Build(P) = the measured build at 160 Kathleen ranks (2.5 × 10⁴ points per rank) + n (c(P)
− c(160)) for the n all-to-alls of counts per build (640's count) at c(P) per call + the
growth of the rounds' all-reduces (payload ∝ P, at 0.25 GB/s); the share is n c(P) /
Build(P). Two c(P): T1's linear 0.376 µs P, and the measured trend 43.8 µs + 0.102 µs P.

| precision | workload | build at 160 (ms) | n | 10³, T1 | 10³, trend | 10⁴, T1 | 10⁴, trend |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| f64 p = 8 | cube | 373 | 25 | 9 / 383 ms (2 %) | 4 / 378 ms (1 %) | 94 / 494 ms (19 %) | 27 / 426 ms (6 %) |
| f64 p = 8 | Plummer | 242 | 31 | 12 / 261 ms (4 %) | 5 / 254 ms (2 %) | 117 / 456 ms (26 %) | 33 / 372 ms (9 %) |
| f64 p = 8 | clusters | 241 | 32 | 12 / 259 ms (5 %) | 5 / 252 ms (2 %) | 120 / 452 ms (27 %) | 34 / 365 ms (9 %) |
| f32 p = 3 | cube | 374 | 25 | 9 / 384 ms (2 %) | 4 / 378 ms (1 %) | 94 / 494 ms (19 %) | 27 / 427 ms (6 %) |
| f32 p = 3 | Plummer | 276 | 31 | 12 / 294 ms (4 %) | 5 / 287 ms (2 %) | 117 / 489 ms (24 %) | 33 / 406 ms (8 %) |
| f32 p = 3 | clusters | 264 | 32 | 12 / 282 ms (4 %) | 5 / 274 ms (2 %) | 120 / 474 ms (25 %) | 34 / 388 ms (9 %) |

**The trigger (decision 4: the count all-to-alls above a quarter of a modelled build at
10⁴ ranks)**: on T1's linear bound the Plummer sphere and the clusters reach 24–27 %, the
cube 19 %; on the measured trend 6–9 %. On T1's bound it would fire at about 10⁴ ranks,
on the trend at a few times 10⁴ (the share grows like P at fixed N/P either way). And 29–30
of the 31–32 calls are rlst's (`GhostCommunicatorBuilder::build` and
`array_tools::all_to_allv`): a remedy in this repository alone reaches only
`Redistribution::new`'s 2; the rest is an rlst change (the brief: ask first).

## 4. S5 for the evaluation: the redistribution's all-to-all-v pair

Every evaluation calls two `MPI_Alltoallv` with P-long arrays (the charges forward, the
output backward; `Redistribution::{forward_into, backward_into}`), whatever the routes. On
the random share of the harness both are dense (every rank sends to every other); in an
application whose points sit on or near their owners most counts are zero. Their cost
when the counts are sparse is §3.2's: negligible with basic linear, 2.1 µs per peer with
pairwise (Kathleen, 80 ranks). Per evaluation, at 10³ / 10⁴ ranks (*model*): basic linear
about 0 / 0 ms; T1's α_v = α_a, 2 × 0.376 µs P: 0.75 / 7.5 ms; pairwise, 2 × 2.1 µs P: 4.2
/ 42 ms.

The trigger (decision 4: above 1 % of a modelled evaluation at 10⁴ points per rank):
the evaluation there, scaled from the measured one at 160 Kathleen ranks and 2.5 × 10⁴
points per rank (blocking), is 466–570 ms at f64 p = 8 (1 %: 4.7–5.7 ms) and 22–74 ms at
f32 p = 3 (0.22–0.74 ms). **It fires on T1's model (as decision 4 recorded) and on
pairwise, and not on basic linear; which one Open MPI picks at 10⁴ ranks is its fixed
rule, not measured here.** The remedy fixes the routes at build, so `forward` and
`backward` become scoped immediate sends and receives between the ranks with non-zero
counts (as the overlapped exchanges post them), with the same bytes in the same order:
no consensus, no new collective, the P-long arrays gone.

## 5. S8: the fan-out below the replicated top

### 5.1 Counted and timed

From `scaling`'s summary (T5's fields), 2.5 × 10⁴ points per rank, one evaluation; (K)
Kathleen, (l) locust; 40–160 Kathleen and 64–72 locust timed, the others counted.

busiest sender's messages per evaluation:

| precision | workload | 40 (K) | 64 (l) | 72 (l) | 80 (K) | 128 (l) | 160 (K) | 256 (l) | 320 (K) | 512 (l) | 640 (K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 8 | cube | 48 | 59 | 51 | 56 | 64 | 73 | 79 | 84 | 117 | 142 |
| f64 p = 8 | Plummer | 93 | 80 | 95 | 106 | 112 | 106 | 139 | 116 | 155 | 184 |
| f64 p = 8 | clusters | 41 | 94 | 96 | 71 | 150 | 141 | 261 | 212 | 486 | 340 |
| f32 p = 3 | cube | 48 | 60 | 51 | 57 | 66 | 73 | 84 | 84 | 113 | 139 |
| f32 p = 3 | Plummer | 91 | 80 | 90 | 89 | 106 | 106 | 134 | 130 | 157 | 169 |
| f32 p = 3 | clusters | 44 | 90 | 94 | 97 | 154 | 137 | 262 | 236 | 464 | 458 |

largest fan-out of a box:

| precision | workload | 40 (K) | 64 (l) | 72 (l) | 80 (K) | 128 (l) | 160 (K) | 256 (l) | 320 (K) | 512 (l) | 640 (K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 8 | cube | 9 | 14 | 10 | 10 | 9 | 10 | 15 | 19 | 27 | 33 |
| f64 p = 8 | Plummer | 13 | 13 | 14 | 17 | 16 | 17 | 31 | 19 | 31 | 39 |
| f64 p = 8 | clusters | 7 | 10 | 11 | 11 | 15 | 15 | 31 | 31 | 63 | 63 |
| f32 p = 3 | cube | 10 | 13 | 10 | 9 | 9 | 10 | 15 | 19 | 27 | 33 |
| f32 p = 3 | Plummer | 12 | 13 | 16 | 15 | 15 | 16 | 31 | 19 | 31 | 39 |
| f32 p = 3 | clusters | 8 | 11 | 11 | 11 | 15 | 15 | 31 | 31 | 63 | 63 |

R's threshold:

| precision | workload | 40 (K) | 64 (l) | 72 (l) | 80 (K) | 128 (l) | 160 (K) | 256 (l) | 320 (K) | 512 (l) | 640 (K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 8 | cube | 3 | 4 | 5 | 5 | 8 | 10 | 16 | 20 | 32 | 40 |
| f64 p = 8 | Plummer | 3 | 4 | 5 | 5 | 16 | 16 | 32 | 20 | 32 | 40 |
| f64 p = 8 | clusters | 3 | 8 | 8 | 8 | 16 | 16 | 32 | 32 | 64 | 64 |
| f32 p = 3 | cube | 3 | 4 | 5 | 5 | 8 | 10 | 16 | 20 | 32 | 40 |
| f32 p = 3 | Plummer | 3 | 4 | 5 | 5 | 16 | 16 | 32 | 20 | 32 | 40 |
| f32 p = 3 | clusters | 3 | 8 | 8 | 8 | 16 | 16 | 32 | 32 | 64 | 64 |

neighbours (max):

| precision | workload | 40 (K) | 64 (l) | 72 (l) | 80 (K) | 128 (l) | 160 (K) | 256 (l) | 320 (K) | 512 (l) | 640 (K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 8 | cube | 24 | 29 | 31 | 32 | 31 | 34 | 38 | 43 | 53 | 59 |
| f64 p = 8 | Plummer | 29 | 30 | 32 | 38 | 40 | 40 | 58 | 47 | 55 | 62 |
| f64 p = 8 | clusters | 17 | 27 | 30 | 27 | 38 | 38 | 61 | 56 | 102 | 108 |
| f32 p = 3 | cube | 24 | 29 | 33 | 31 | 33 | 35 | 39 | 43 | 53 | 59 |
| f32 p = 3 | Plummer | 28 | 30 | 33 | 29 | 40 | 38 | 54 | 52 | 63 | 62 |
| f32 p = 3 | clusters | 17 | 28 | 29 | 31 | 38 | 39 | 58 | 60 | 100 | 103 |

Kathleen, by node (messages to other nodes; neighbours on other nodes; other nodes):

| precision | workload | 80 | 160 | 320 | 640 |
| --- | --- | --- | --- | --- | --- |
| f64 p = 8 | cube | 21; 12; 1 | 34; 17; 3 | 43; 19; 3 | 62; 24; 3 |
| f64 p = 8 | Plummer | 47; 16; 1 | 89; 27; 3 | 88; 28; 3 | 121; 40; 3 |
| f64 p = 8 | clusters | 37; 9; 1 | 57; 19; 3 | 101; 29; 3 | 181; 61; 3 |
| f32 p = 3 | cube | 21; 12; 1 | 37; 18; 3 | 43; 19; 3 | 64; 24; 3 |
| f32 p = 3 | Plummer | 37; 14; 1 | 78; 26; 3 | 80; 28; 3 | 114; 37; 3 |
| f32 p = 3 | clusters | 37; 12; 1 | 62; 19; 3 | 95; 32; 3 | 208; 58; 3 |

The busiest sender's messages grow with P on every workload, fastest on the clusters (the
boxes about a rank's size are read by a roughly constant number of ranks while R's
threshold rises with P, so they leave R and go point to point: T5 §4). The largest fan-out
of a box is the threshold less one wherever the threshold binds (63 at 512–640 on the
clusters, where the 4,096-box cap doubles the threshold to 64). Counts do not depend on the
machine (the second cut's constants are Kathleen's on both), so the two machines' columns
form one series.

Timed at 160 Kathleen ranks, the multipole exchange on the overlapped path:

| workload | precision | evaluation, blocking (ms) | multipole exchange, blocking (ms) | evaluation, overlapped (ms) | multipoles: exposed wait (ms) | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cube | f64 p = 8 | 1,164 | 4.91 | 1,164 | 0.009 | 0.0008 % |
| Plummer | f64 p = 8 | 1,263 | 8.49 | 1,262 | 0.013 | 0.0010 % |
| clusters | f64 p = 8 | 1,425 | 6.62 | 1,421 | 0.005 | 0.0004 % |
| cube | f32 p = 3 | 55.6 | 0.532 | 55.8 | 0.002 | 0.0036 % |
| Plummer | f32 p = 3 | 147 | 0.859 | 147 | 0.003 | 0.0020 % |
| clusters | f32 p = 3 | 185 | 0.973 | 185 | 0.003 | 0.0016 % |

The overlapped path hides the multipole exchange entirely (the blocking stage is 4.9–8.5 ms
at f64 p = 8 and 0.53–0.97 ms at f32 p = 3, 0.4–1.0 % of the evaluation). **The measured
clause of the trigger (the multipole exposed wait above 1 % of the evaluation at 160
Kathleen ranks) does not fire.**

### 5.2 The model at 10³ and 10⁴ ranks (*model*)

The busiest sender's messages fitted per workload and precision over the nine counts from
64 to 640 ranks, as a power law (least squares in log–log) and as a line; the time at an
effective 4 µs per message (scale-out.md §10.4; T0: 342 messages in 1.43 ms). The
threshold is 1 % of a modelled evaluation at 2.5 × 10⁴ points per rank: at f64 p = 8 the
cube's 549 ms and the Plummer sphere's 1,212 ms of §10.4, the clusters' 1,425 ms measured
at 160 Kathleen ranks (§10.4 has none); at f32 p = 3 the evaluation measured at 160 ranks
(56, 147, 185 ms; §10.4 has none). "Fires at P" solves fit × 4 µs = 1 % for P.

| precision | workload | power fit | 10³ | 10⁴ | linear fit | 10³ | 10⁴ | evaluation (ms) | 1 % (ms) | messages at 1 % | fires at P (power; linear) |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| f64 p = 8 | cube | 10.3 P^0.38 | 147 (0.59 ms) | 358 (1.43 ms) | 45 + 0.145 P | 190 (0.76 ms) | 1,496 (5.98 ms) | 549 | 5.49 | 1,372 | 329,251; 9,152 |
| f64 p = 8 | Plummer | 28.1 P^0.28 | 189 (0.76 ms) | 357 (1.43 ms) | 85 + 0.147 P | 232 (0.93 ms) | 1,554 (6.21 ms) | 1,212 | 12.12 | 3,030 | 23,356,005; 20,051 |
| f64 p = 8 | clusters | 4.2 P^0.71 | 578 (2.31 ms) | 2,972 (11.89 ms) | 60 + 0.589 P | 648 (2.59 ms) | 5,947 (23.79 ms) | 1,425 | 14.25 | 3,563 | 12,905; 5,951 |
| f32 p = 3 | cube | 11.2 P^0.37 | 145 (0.58 ms) | 340 (1.36 ms) | 47 + 0.137 P | 184 (0.74 ms) | 1,417 (5.67 ms) | 56 | 0.56 | 139 | 894; 673 |
| f32 p = 3 | Plummer | 23.5 P^0.30 | 193 (0.77 ms) | 388 (1.55 ms) | 81 + 0.147 P | 228 (0.91 ms) | 1,552 (6.21 ms) | 147 | 1.47 | 367 | 8,329; 1,944 |
| f32 p = 3 | clusters | 3.8 P^0.75 | 655 (2.62 ms) | 3,639 (14.56 ms) | 47 + 0.702 P | 749 (3.00 ms) | 7,062 (28.25 ms) | 185 | 1.85 | 463 | 628; 592 |

**The modelled clause fires**: at f32 p = 3 on all three workloads under both fits (from
about 600 ranks on the clusters, 700–900 on the cube, 2,000–8,000 on the Plummer sphere);
at f64 p = 8 on the clusters at 6,000 (linear) or 13,000 (power law) ranks, beyond 10⁴ on
the power law against the clusters' own 14.3 ms but within it against the cube's 5.5 ms of
the brief, and on the cube only under the linear fit (9,000 ranks). Two qualifiers: the
model charges every message at 4 µs on the critical path, the blocking path's cost, while
the overlapped path measured nothing exposed at 160 ranks; and at 10³–10⁴ ranks R's cap
(4,096 boxes) will set the threshold rather than ⌈P/16⌉ (it already does on the clusters
at 512–640), so the fan-out of the busiest box grows with P/|R| beyond the fitted range
(not measured).

## 6. Decision (step 2), for sign-off

| Package | Trigger (decision 4; T5's for S8) | Counted or modelled number | Fires? |
| --- | --- | --- | --- |
| S4, construction | most keys received > 2 × mean, or sorts > ¼ of `Octree::new` | routing within 1.002–1.12 to 640 ranks; no sorts | no (built in T3a; closed) |
| S5, build | count all-to-alls > ¼ of a modelled build at 10⁴ | 19–27 % (T1's α_a), 6–9 % (trend); 29–30 of 31–32 in rlst | on the edge on T1's bound; no on the trend |
| S5, evaluation | the all-to-all-v pair > 1 % of an evaluation at 10⁴ points per rank | 0 / 7.5 / 42 ms against 0.22–5.7 ms | yes on T1's model and on pairwise; no on basic linear |
| S8, modelled | busiest sender's messages at 10⁴ × 4 µs > 1 % of an evaluation | f32 p = 3: 1.4–28 ms against 0.56–1.85 ms; f64 p = 8: 1.4–24 ms against 5.5–14 ms | yes (f32 p = 3 everywhere; f64 p = 8 clusters on the linear fit or the cube's threshold) |
| S8, measured | multipole exposed wait > 1 % of the evaluation at 160 ranks | 0.002–0.013 ms, ≤ 0.004 % | no |

Open, whatever is decided: the timing at scale of all of these (nothing runs timed beyond
160 ranks); which all-to-all-v algorithm Open MPI picks at 10³–10⁴ ranks; contention of
the 40 ranks of a node on one link, which the 4 µs per message does not model; collective
algorithms at thousands of ranks (the all-to-all of counts' growth beyond 160); R's cap
binding above 640 ranks (§5.2); the rounds' all-reduces (§2.3), which no package covers.

Questions (the recommendations first; the brief: nothing is built without them):
1. **S5 for the evaluation.** (a) Build it in T8 as decision 4 recorded: `forward` and
   `backward` point to point on the routes fixed at build, tested against the all-to-all-v
   version at 1–8 ranks, 512 (locust) and 160 (Kathleen), bit for bit; small, in
   `nd-fmm-plan`'s `redistribute.rs`. **Recommended**: it removes a cost whose size depends
   on a library choice we do not control, and the P-long arrays. (b) Close it on the
   basic-linear measurement.
2. **S5 for the build. Scheduled on 2026-10-13 at the user's request as T8r**
   (docs/phase5s/T8r-rlst-sparse-exchange.md): the sparse exchanges are built in rlst,
   released as 0.10.1, and adopted in this task's step 3, together with S5 for the
   evaluation through rlst's fixed-route exchange. The options as first put: (a) Do not
   build in this repository: it does not fire on the
   measured trend, and 29–30 of the 31–32 calls are rlst's; record the trigger as "on the
   edge at 10⁴ ranks on T1's bound" for T9. (b) The sparse dynamic exchange (non-blocking
   consensus, rsmpi 0.8.2's `immediate_synchronous_send`, probes and `immediate_barrier`)
   in rlst's `GhostCommunicatorBuilder::build` and a sparse `all_to_allv`, released and
   pinned here.
3. **S8. Signed off on 2026-10-13 by the user: option (a), its own task after T8**
   (docs/phase5s/T8s-fan-out-tree.md). It fires on the modelled clause. (a) Build the forwarding tree as its own task
   after T8 (T8b), with the design of the brief (each top-tree box below R sent along a
   tree of its readers fixed at build, a node's first reader forwarding within its node,
   copies never sums, checked slot by slot, a new `HostData` event agreed with the device
   mirror first): a structural change of the multipole exchange, which today rides on
   rlst's per-level ghost communicators, and of the device's events. **Recommended**: the
   overlapped path measures nothing exposed at 160 ranks, so nothing is lost by measuring
   T9 first, and the change deserves its own review. (b) Build it in T8. (c) Close it on the
   measured clause (the overlapped path), with the modelled crossing points of §5.2 as the
   record.

## 7. Kathleen job and node-hours

| Job | Nodes, QoS | What | Waited | Ran | Node-hours |
| --- | --- | --- | ---: | ---: | ---: |
| 604012 | 4, `small`, node-c11b-[002,008,017,032] | build of the probe copy; micro-benchmarks at 40, 80, 160; timed at 40, 80, 160 (N = 10⁷, 10⁸, 2.5 × 10⁴ per rank); counted at 320, 640 | 1:04:03 | 29:47 | 1.99 |

Within the task's budget of 20 node-hours. Raw outputs stay in Kathleen's
`/scratch/scratch/ucahtbe/fmm/logs/p5s-t8/` and locust's `/data/ucahtbe/logs/p5s-t8/`.

## 8. Checks

No code changed in this step: the repository holds this report and the addendum to
docs/design/scale-out.md §7 (§7.3). The checks of the brief that apply to a change of code
run with step 3, if anything is built.

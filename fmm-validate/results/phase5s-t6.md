# The owned top tree in `nd-fmm-exec` (Phase 5S T6)

S2 in `nd-fmm-exec`, as docs/phase5s/T6-owned-top-tree-exec.md asks and
docs/design/scale-out.md §4 (signed off) specifies: the device operator mirrors T5's
upward reduction (its new `HostData` events), so the device runs on several ranks again;
`Fmm`'s stage timings and exchange sizes report the reduction and the replicated top; the
`scaling` harness measures them per rank. Run on 2026-10-13 on the M3 Max, locust (GH200)
and Kathleen.

**Measured** means run here and read from the output; timings are reported, never asserted.
locust is a shared node: its load is in each output file before and after every launch.
Kathleen nodes are exclusive; every job is named.

"Phase 5" below is the code before Phase 5S's structural tasks: `06016d2` (T2 merged, the
octree of T3a; the evaluation as Phase 5N left it), built in a detached worktree beside
this branch and run with its own `scaling` harness. "T6" is this branch.

## Summary

- **The device follows the owned top tree on several ranks.** At each `SendReduction(l)`
  the device gathers the level's sent multipoles into a packed buffer and downloads them;
  at each `ReceivedReduction(l)` it uploads the received children and scatters them into
  the multipoles. The device gate (`tests/device_ranks.rs`) passes at 1, 2 and 4 ranks on
  the CPU runtime (M3 Max, locust) and on CUDA (locust), and at 2 ranks on Metal (M3 Max),
  every transfer of every rank equal to the extended formula, the output within Phase 4's
  bounds of the one-rank host run (relative L2 of φ at most 2.7e-6 in f32 and 1.4e-15 in
  f64, against bounds of 1e-5 and 1e-12). `mpi_exec`'s device repeat of every scenario
  (105 on several ranks) passes at 1, 2, 4 and 8 ranks on both machines.
- **Syncs per evaluation: 1 + [anything else sent] + r_s, at most 2 + r_s**, as signed off
  (scale-out.md §4.5, §12 question 4; Phase 5's two). r_s, the reduction levels at which a
  rank sends, reached 5 in the gate (the Plummer sphere, N = 10⁵: 7 syncs on that rank).
  The reduction's device buffers add at most about 20 kB per rank in the gate (1% of the
  exchange buffers).
- **Bit for bit, as before**: `multi_rank` 0 values differing in every row at 1, 2, 4 and 8
  ranks (M3 Max and locust); the shadow operator bit for bit the plain operator on every
  scenario and rank count, and different with only the owned top tree's three events
  disabled on 2, 4 and 8 ranks (2,312, 6,000 and 6,000 of 6,000 values); overlap bit for
  bit the blocking path in every launch; rank-count invariance 0 values differing at 1, 2,
  4, 8 (M3 Max, locust; CI's list), at 80 against 40, 20, 10 and 160 against 80, 40, 20
  (Kathleen, three workloads each), 640 against 80 (Kathleen) and 512 against 64 (locust;
  both oversubscribed).
- **`StageTimings::{reduction, replicated_top}`** report the two new movements in both
  paths; the harness has them as rows of the stage tables and the summary line, the
  reduction's traffic per rank, and the memory model with the partition map.
- **Phase 5 against T6 (timed):** the stage "upward global" (Phase 5: the coarse gather and
  the global M2M of every `Global` box; T6: the reduction, the owned M2M and the
  all-gather of R) falls 1.4–2.8× at 80 and 160 Kathleen ranks and 1.3–4.3× at 72 locust
  ranks; at 8 locust ranks and at 40 Kathleen ranks (one node) it is about the same
  (0.25–3.7×, within ±2 ms, on locust with the 1 ms of the open point below), where the
  coarse gather was small. The evaluation wall time moves by +0.2% to −2.2% on Kathleen
  and −1.7% to −3.6% at 72 locust ranks; T6 changes nothing else in the evaluation.
- **Open:** at 8 locust ranks T6's binary runs the local upward pass 11% slower than T5's
  and Phase 5's on every rank, in every order, though T6 changes no level call; on the
  M3 Max and on Kathleen it is faster. About 0.3% of the wall time; not traced (§2,
  reading 6).
- **Per rank (counted):** the coarse gather (up to 21.2 MB per rank and evaluation at 640
  Kathleen ranks) is replaced by R's all-gather (≤ 2.63 MB) and the reduction (≤ 35
  messages, ≤ 31 kB received); the memory model falls from 92–132 MB to 24–32 MB per rank
  and the resident size from 335–431 MB to 268–298 MB at 640 ranks; at 160 ranks 41–87 →
  34–56 MB (model) and 175–230 → 168–194 MB (resident).

## 1. The change

`fmm-exec/src/device.rs`, `fmm.rs`, `tests/device_common`, `tests/device_ranks.rs`,
`tests/mpi_exec.rs`, `tests/shadow`; `nd-fmm-validate`'s `scaling.rs` (with
`fmm_accuracy.rs` and `device_fmm.rs`, which aggregate `StageTimings`) and
`tests/scaling.rs`; the crates' CLAUDE.md files. `nd-fmm-plan` is unchanged.

1. **The device mirrors the upward reduction** (scale-out.md §4.5). `ExchangeLists` copies
   the reduction's per-level lists at build (`reduction_send`, `reduction_receive`, from
   `ReductionExchange::send_boxes`/`receive_boxes`); the device operator allocates one
   packed send buffer and one packed receive buffer for them, the columns of every
   reduction level one after the other and the values of the widest level, reused level
   after level. At `SendReduction(l)` it gathers the level's sent boxes (own blocks, and
   the owned `Global` boxes that the global M2M of level l has just formed on the device)
   into the send buffer, downloads it (one sync) and writes each column into its slot of
   the host store, where the exchange reads it; at `ReceivedReduction(l)` it uploads the
   reduction's receive buffer and scatters it into the multipoles
   (`movement::scatter_columns`, an assignment), before the global M2M of level l − 1
   reads it. A level the rank neither sends nor receives costs nothing. T5's refusal (a
   device error at any box to reduce) is gone.
2. **Transfer accounting.** `DataKind::{SentReduction, ReceivedReduction}` (15 kinds),
   `DeviceCounters::reduction_levels` (per level: the downloads sent, the uploads
   received), `ExchangeSizes::{sent_reduction, received_reduction}` (S^red_l and R^red_l
   per level), and the formula of `tests/device_common::expected_evaluation`: per level
   with S^red_l > 0 one gather launch and one download of S^red_l n_c s bytes (one sync),
   per level with R^red_l > 0 one upload of R^red_l n_c s bytes and one scatter launch.
   **The syncs per evaluation are 1 + [anything else sent] + r_s, at most 2 + r_s**, r_s
   the reduction levels at which the rank sends: the multipoles a level sends are formed
   on the device by the global M2M of that level, and they must reach the host before
   that level's exchange, which comes before the next level's M2M, so the downloads cannot
   be merged. On one rank every list is empty and an evaluation is Phase 4's minimum (one
   sync). `check_minimum`, the new `check_reduction_levels` and **device errors** check
   the count.
3. **The device scenarios on several ranks run again**: `mpi_exec`'s device repeat of every
   scenario, the CPU runtime's threads rule, **device errors and tuning on several
   ranks** and the gate `tests/device_ranks.rs`, whose transfer table gains the
   reduction's columns ("reduction down", "reduction up").
4. **The shadow check without the new events.** `tests/shadow`'s `Hook` (`On`, `Off`,
   `WithoutTopTree`): with `WithoutTopTree` the shadow ignores exactly T5's three events
   (`SendReduction`, `ReceivedReduction`, `ReceivedTop`) and follows the others.
   **host-data hook** checks that it is exact on one rank and differs on several, where it
   also asserts that the reduction receives a child.
5. **`StageTimings::{reduction, replicated_top}`**, parts of `upward_global`, not added to
   the total, timed in both paths by the host clock in `ExecOperator::host_data`, the one
   place every event passes (`EventClock`): `reduction`, summed over the levels, from the
   arrival of `SendReduction(l)` to the return of `ReceivedReduction(l)` (the device's
   moves, the packing, the exchange or its posting and wait, the wait for other ranks);
   `replicated_top`, the all-gather of R (from the return of `SendMultipoles` to the
   arrival of `ReceivedTop` blocking; `OverlapTimes::top_gather` overlapped, where the
   multipole exchange is posted in between) plus the `ReceivedTop` event. No new
   `BuildTimings` part: the reduction's and R's builds are inside `Evaluator::new`
   (`BuildTimings::evaluator`, whose documentation says so). `ExchangeTraffic` has had the
   reduction per level (messages, values) since T5.
6. **`scaling`**: `STAGES` (17 rows) has "upward reduction" and "replicated top" as parts of
   "upward global", from the new fields, in both paths; T5's row "top all-gather" (timed
   only overlapped, NaN blocking) is gone, and the summary keys `reduction_max_ms` and
   `replicated_top_max_ms` replace `top_gather_max_ms`. The C5.2 line reports the
   reduction beside the criterion (the criterion stays distributed-fmm.md §8.6's: the
   source and multipole exchanges). `RankTraffic::reduction` (summed over the levels,
   moved from `TopTreeRow`, which keeps the levels) has columns in the per-rank traffic
   table. `MemoryModel::partition` is the partition map of scale-out.md §8.2 (8 B for the
   blocks' keys, their leaf bits, 8 (P + 1) for the cut) in place of the bounds' 8 P; the
   held boxes K are H (T5), and the buffers already held the reduction's and R's values
   (T5). `tools/scaling/run.sh` and the Kathleen jobs are unchanged.

## 2. Readings and departures (for review)

1. **Syncs: 2 + r_s, as signed off**, measured up to r_s = 5 in the gate (§4). They are
   correctness-only on several ranks (Phase 5 decision 3), so no measured scaling pays
   for them.
2. **The reduction's buffers are reused level after level** (the widest level's values),
   as the multipole exchange's receive buffer is: each send level's download waits for its
   gather, and one stream orders each upload after the scatter that read the buffer
   before it. They add (max_l S^red_l + max_l R^red_l) n_c s bytes of values plus 4 bytes
   per column; in the gate at most the reduction's bytes per evaluation, 20.1 kB on the
   busiest rank (the Plummer sphere, f64 p = 8, 4 ranks), against 1.6–2.0 MB of exchange
   buffers in `DeviceReport::exchange_bytes`.
3. **The two stage times come from the events, not from `nd-fmm-plan`**, which T6 does not
   change: the reduction's span in the overlapped path includes the `test` call between
   posting a level and waiting for it, which `OverlapTimes::upward_global` leaves out, so
   there the two parts may exceed "upward global" by that call (the smoke test checks the
   inequality on the blocking path only). Each part's maximum over the ranks can come from
   another rank than the stage's.
4. **The C5.2 criterion is unchanged**: the reduction is reported beside it, as Phase 5
   reported the coarse gather; it is a few small messages per level (at most 7 per level
   and rank).
5. **Where the wait goes.** The first synchronising step after the local upward pass
   absorbs its imbalance: in Phase 5 the coarse gather, in T6 the first reduction level a
   rank receives and R's all-gather. So "upward global" holds a wait for the slowest
   rank's local M2M in both codes (at 40 Kathleen ranks the reduction's maximum is 13.8 ms
   on the cube, almost all of it waiting), and the comparison is of whole stages.
6. **Open: T6's binary on locust.** At 8 locust ranks T6's local upward pass (P2M and the
   local M2M, whose level calls T6 does not change) is 11% slower than T5's (`main`) and
   Phase 5's on every rank, in every order (§5.3), and its "upward global" about 1 ms
   slower; the evaluation's wall time differs by −0.7% to +1.4%. On the M3 Max, also
   aarch64, the same comparison goes the other way (the cube, N = 10⁶, f64 p = 8, 8
   ranks, overlap off, T5, T6, T5, T6: upward local 22.2, 20.8, 21.6, 20.7 ms; upward
   global 2.40, 2.43, 2.21, 2.31 ms), and on Kathleen T6's local pass is 4.7–7.5% faster
   than Phase 5's (§5.1). What T6 adds on the host path is the event clock (two `Instant`
   reads per event, at most about 40 events per evaluation, none in the local pass) and larger
   types; neither is in the local pass's loops. The cause, presumably code placement in
   T6's binary on the Neoverse V2, is not found; it costs about 3 ms of 1,160 ms here.

## 3. Correctness

### 3.1 Tests

- `mpi_exec` (every scenario; `--features cpu` adds the device repeat of each, against the
  host path on the same ranks, with the extended formula, and **device errors and tuning
  on several ranks**): the device repeats ran 16 f32 and 76 f64 scenarios on one rank and
  20 f32 and 85 f64 on 2, 4 and 8, on the host fallback bit for bit and with every kind on
  the device within the FMM bounds; 35 shadow evaluations bit for bit, 231 overlapped
  evaluations bit for bit the blocking path, 105 evaluations bit for bit the one-rank
  `Fmm` over the union (on several ranks). **host-data hook** (the cube, N = 1,500, p = 4):

  | Ranks | ghost leaves / boxes | children of the reduction / boxes of R received | without the hook: values differing | without the top tree's events: values differing |
  | ---: | --- | --- | --- | --- |
  | 1 | 0 / 0 | 0 / 0 | 0 of 6,000 | 0 of 6,000 |
  | 2 | 36 / 71 | 5 / 0 | 6,000 of 6,000 | 2,312 of 6,000 |
  | 4 | 98 / 213 | 9 / 189 | 6,000 of 6,000 | 6,000 of 6,000 |
  | 8 | 198 / 498 | 13 / 442 | 6,000 of 6,000 | 6,000 of 6,000 |

- `device_ranks` (the gate, release, ignored): §4.
- `multi_rank` (ignored): 0 values differ from the union in rank order in all 30 rows at
  2, 4 and 8 ranks, on both machines; passes at one rank.
- `fmm-validate/tests/scaling.rs` at 1 and 2 ranks: the new rows finite, the reduction
  zero on one rank and both parts inside the blocking "upward global", the reduction's
  traffic none on one rank and sent equal to received over the ranks, the partition map
  at least one block's key and the cut.
- `tests/invariance.rs` (CI's list) at 1, 2, 4 and 8, and `scaling --invariance` on
  Kathleen and locust (§3.2).

### 3.2 Runs

| Where | Ranks | What | Result |
| --- | --- | --- | --- |
| M3 Max | 1, 2, 4, 8 | `run-mpi-tests.sh` (debug): `mpi_regressions`, `mpi_exec`, `mpi_threading`, `invariance`, the seven `nd-octree` examples | 44 of 44 pass |
| M3 Max | 1, 2, 4, 8 | `mpi_exec`, `--features cpu`, release | pass |
| M3 Max | 1, 2, 4, 8 | `multi_rank` (release, ignored) | pass; 0 values differ in all 30 rows |
| M3 Max | 1, 2, 4 | `device_ranks`, CPU runtime | pass |
| M3 Max | 2 | `device_ranks`, Metal (outside the sandbox) | pass |
| M3 Max | 1 | `cargo test -p nd-fmm-exec --features cpu --release`; the same `-- --ignored`; `--features metal --release -- --ignored` | 86 passed; 10 passed; 13 passed |
| M3 Max | 1, 2 | `fmm-validate/tests/scaling.rs` | pass |
| locust | 1, 2, 4, 8 | `run-mpi-tests.sh` (debug) | 44 of 44 pass |
| locust | 1, 2, 4, 8 | `mpi_exec` `--features cpu`, `mpi_threading`, `multi_rank` (release) | pass; `multi_rank` 0 values differ in all 30 rows |
| locust | 1, 2, 4 | `device_ranks`, CPU runtime and CUDA | pass |
| locust | 1 | `cargo clippy -p nd-fmm-exec --all-targets --features cpu,cuda -- -D warnings`; `cargo test -p nd-fmm-exec --features cuda --release`, and `-- --ignored` | clippy clean; 80 passed; 13 passed (every CUDA test, the C4.8 gate on CUDA included) |
| locust | 512 against 64, oversubscribed | `scaling --invariance 8`, the N = 10⁷ cube, f64 p = 8, Rotation | 0 of 8 × 10⁷ values differ |
| Kathleen 587503 | 2 nodes | T2's correctness job: the MPI list at 2, 4, 8 and 80 ranks; invariance at 80 against 40, 20, 10 (the cube and the Plummer sphere f64 p = 8, the cube f32 p = 3); the smoke run | 8 of 8 steps pass; 44 of 44 MPI launches; 0 values differ in each of the 9 checks |
| Kathleen 592032 | 4 nodes | invariance at 160 against 80, 40, 20 (the same three workloads), 640 against 80 oversubscribed (the cube) | 6 of 6 steps pass; 0 of 8 × 10⁶ values differ in each of the 10 checks |

## 4. The device on several ranks

The gate (`tests/device_ranks.rs`, release, 1 thread per rank, the ranks sharing one
device): the cube and the Plummer sphere at N = 10⁴ (CPU runtime: f64 p = 8, f32 p = 3) and
N = 10⁵ (Metal: f32 p = 3, 8; CUDA: f32 p = 3, 8 and f64 p = 8), each against the host on
the same ranks (every kind on the fallback bit for bit, the default placement within the
bounds, the formula) and against the one-rank host run; the transfer table at N = 10⁵, f32
p = 8 on the CPU runtime and as a workload point on Metal and CUDA. Largest relative L2
from the one-rank host run (φ / ∇φ):

| Backend | Ranks | f32 | f64 |
| --- | ---: | --- | --- |
| CUDA (locust) | 2 | 2.7e-6 / 1.2e-7 | 1.3e-15 / 2.8e-16 |
| CUDA (locust) | 4 | 2.7e-6 / 1.2e-7 | 1.4e-15 / 2.9e-16 |
| Metal (M3 Max) | 2 | 2.7e-6 / 1.5e-7 | – |

The transfers of one evaluation per rank, every one equal to the formula ("= formula"
yes on every rank of every point and backend). Columns as the gate prints them: calls ×
bytes per kind, then syncs, launches and the device memory of the exchange buffers and
ghost slots (bytes).

The cube, N = 10⁵, f32 p = 8, 4 ranks (CUDA; the CPU runtime's table is the same):

| rank | charges up | ghost sources up | reduction down | reduction up | sent multipoles down | top up | received multipoles up | output down | syncs | launches | exchange buffers B | ghost slots B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 × 100,736 | 1 × 105,920 | 0 | 1 × 1,944 | 1 × 243,648 | 1 × 15,552 | 2 × 238,464 | 1 × 402,944 | 2 | 46 | 453,936 | 617,840 |
| 1 | 1 × 99,972 | 1 × 106,768 | 1 × 648 | 0 | 1 × 243,648 | 1 × 15,552 | 2 × 238,464 | 1 × 399,888 | 3 | 43 | 452,624 | 618,688 |
| 2 | 1 × 99,220 | 1 × 106,720 | 1 × 648 | 0 | 1 × 243,648 | 1 × 15,552 | 2 × 238,464 | 1 × 396,880 | 3 | 43 | 452,624 | 618,640 |
| 3 | 1 × 100,072 | 1 × 106,656 | 1 × 648 | 0 | 1 × 243,648 | 1 × 15,552 | 2 × 238,464 | 1 × 400,288 | 3 | 43 | 452,624 | 618,576 |

The Plummer sphere, N = 10⁵, f64 p = 8, 4 ranks (CUDA), the deepest reduction:

| rank | charges up | ghost sources up | reduction down | reduction up | sent multipoles down | top up | received multipoles up | output down | syncs | launches | exchange buffers B | ghost slots B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 × 198,592 | 1 × 407,648 | 0 | 4 × 10,368 | 1 × 1,283,040 | 1 × 114,048 | 6 × 1,018,656 | 1 × 794,368 | 2 | 123 | 1,727,640 | 2,698,976 |
| 1 | 1 × 203,224 | 1 × 611,200 | 4 × 8,424 | 4 × 11,664 | 1 × 1,360,152 | 1 × 117,288 | 6 × 1,612,224 | 1 × 812,896 | 6 | 124 | 2,028,644 | 4,097,440 |
| 2 | 1 × 194,232 | 1 × 474,816 | 5 × 12,960 | 2 × 8,424 | 1 × 1,262,952 | 1 × 123,120 | 6 × 1,277,856 | 1 × 776,928 | 7 | 126 | 1,789,504 | 3,301,392 |
| 3 | 1 × 203,952 | 1 × 376,352 | 3 × 9,072 | 0 | 1 × 1,163,808 | 1 × 131,544 | 6 × 990,144 | 1 × 815,808 | 5 | 119 | 1,635,460 | 2,626,208 |

At 2 ranks (the Plummer sphere, f64 p = 8, CUDA) rank 1 sends at 5 levels (5 × 13,608
B in all) and syncs 7 times, rank 0 receives at 5 levels and syncs twice; on Metal (f32
p = 8) the same 5 levels and 7 syncs. Rank 0 owns the root and its first-child ancestors,
so it receives and never sends in the reduction; the other ranks send their blocks and
owned boxes up. Every point's transfers equal the formula; so do every scenario's in
`mpi_exec`'s device repeats (§3.1).

## 5. Phase 5 against T6, per rank and evaluation

### 5.1 Kathleen, timed: 40, 80 and 160 ranks

Job 587502 (node-c11a-[065,079,096,098], 4 nodes in `small`, `--exclude=node-c11a-011`;
Open MPI 4.1.6 over PSM2, `PSM2_TID=0`); 40, 80 and 160 ranks as 40 × 1 per node on 1, 2
and 4 of its nodes (`mpirun --map-by ppr:20:socket:PE=1 --bind-to core`); the N = 10⁷
cube, Plummer sphere and clusters, f64 p = 8, Rotation, the random share, overlap both, 5
timed evaluations per path after one warm-up; Phase 5 first in every pair. Times in ms,
the maximum over the ranks of each rank's median; "T6: reduction / top" the blocking
path's parts of "upward global"; C5.2 the overlapped path's exposed waits over the
blocking path's exchange stages (§8.6; at most 10% is the criterion, defined at 8 ranks).

| ranks | workload | wall, blocking | upward global | T6: reduction / top | top all-gather, overlapped (P5: coarse gather) | multipole exchange | upward local | downward | leaves | wall, overlapped | C5.2 % |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | cube | 5,563.0 → 5,575.0 | 23.36 → 22.98 | 13.78 / 10.28 | 28.36 → 14.93 | 8.57 → 8.35 | 183.5 → 171.0 | 4,354.6 → 4,370.0 | 976.9 → 986.2 | 5,560.0 → 5,574.9 | 0.1 → 0.0 |
| 80 | cube | 2,830.2 → 2,814.5 | 22.41 → 9.54 | 3.55 / 9.30 | 13.85 → 13.62 | 6.78 → 6.61 | 92.0 → 85.9 | 2,200.7 → 2,200.0 | 488.4 → 492.6 | 2,827.3 → 2,812.4 | 0.1 → 0.0 |
| 160 | cube | 1,447.7 → 1,432.3 | 21.52 → 7.62 | 3.33 / 7.51 | 11.88 → 10.66 | 4.86 → 4.59 | 46.6 → 43.1 | 1,113.6 → 1,114.1 | 251.5 → 253.5 | 1,445.2 → 1,432.7 | 0.3 → 4.4 |
| 40 | Plummer | 15,928.5 → 15,777.8 | 35.04 → 29.02 | 26.59 / 28.46 | 35.57 → 32.53 | 21.77 → 19.79 | 216.2 → 202.4 | 11,709.2 → 11,796.0 | 3,921.5 → 3,707.6 | 15,902.4 → 15,772.5 | 0.0 → 0.0 |
| 80 | Plummer | 8,405.2 → 8,330.6 | 31.39 → 22.73 | 13.17 / 17.15 | 26.25 → 23.27 | 20.98 → 18.31 | 109.4 → 103.8 | 6,670.3 → 6,706.5 | 2,568.2 → 2,429.6 | 8,395.9 → 8,320.0 | 0.1 → 0.1 |
| 160 | Plummer | 4,607.7 → 4,547.6 | 34.15 → 19.36 | 11.93 / 19.19 | 25.40 → 21.66 | 17.14 → 14.13 | 57.7 → 54.3 | 3,492.0 → 3,511.1 | 1,431.3 → 1,368.6 | 4,603.3 → 4,544.6 | 13.4 → 9.4 |
| 40 | clusters | 14,045.0 → 13,959.8 | 39.30 → 28.94 | 16.36 / 20.47 | 34.67 → 26.24 | 17.06 → 15.12 | 211.2 → 199.2 | 10,423.0 → 10,473.9 | 4,402.1 → 4,183.0 | 14,065.5 → 13,941.9 | 0.1 → 0.1 |
| 80 | clusters | 7,572.5 → 7,543.7 | 41.15 → 29.09 | 12.06 / 28.90 | 33.82 → 33.09 | 16.91 → 16.78 | 108.5 → 103.4 | 5,722.4 → 5,779.4 | 2,553.0 → 2,422.7 | 7,570.8 → 7,543.1 | 0.1 → 0.0 |
| 160 | clusters | 4,085.8 → 3,994.9 | 44.89 → 22.78 | 9.53 / 20.90 | 25.84 → 25.82 | 14.74 → 11.67 | 56.3 → 53.3 | 3,055.7 → 3,080.4 | 1,575.5 → 1,493.6 | 4,082.3 → 3,993.2 | 4.5 → 22.7 |

- "upward global" falls 1.4–2.8× at 80 and 160 ranks (2.8× on the cube at 160), and
  1.0–1.4× at 40 (one node), where Phase 5's coarse gather was 0.3–2.2 MB.
- The wall time moves by +0.2% to −2.2% (blocking) and +0.3% to −2.2% (overlapped). The
  compute stages, whose level calls T6 does not change, move by up to 7.5% in both
  directions between the two codes (upward local −4.7% to −7.5%, downward +0.0% to
  +1.0%, leaves −4.4% to −5.5% on the Plummer sphere and the clusters, +0.8% to +1.0% on
  the cube): the plans differ (T4 and T5 changed the held set, and with it the stores'
  layout), not the work.
- C5.2 at 160 ranks: the clusters' 22.7% is one rank's wait in the source exchange (3.4 ms
  against a mean of 0.03 ms; the multipole exchange's exposed wait is 0.01 ms in both
  codes); T5 and T6 do not touch the source exchange.

Per rank and evaluation, the same launches (maxima over the ranks; the reduction: messages
received over every level and bytes received; R's bytes: what the all-gather delivers;
memory: the model, Phase 5's with its own terms, and the peak resident size `VmHWM`):

| ranks | workload | P5 coarse gather, MB | T6 top, MB | T6 reduction: messages, received | multipole exchange, MB | multipole messages sent | global M2M pairs | held boxes | memory model, MB | resident, MB |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | cube | 0.32 | 0.36 | 12, 10.4 kB | 5.72 → 5.59 | 90 → 66 | 584 → 32 | 16,801 → 16,505 | 92.8 → 92.2 | 247.0 → 238.8 |
| 80 | cube | 2.62 | 2.33 | 16, 13.6 kB | 4.27 → 3.96 | 96 → 64 | 4,680 → 80 | 14,457 → 10,609 | 62.0 → 56.0 | 206.3 → 198.8 |
| 160 | cube | 2.64 | 1.77 | 17, 15.6 kB | 3.13 → 2.90 | 115 → 75 | 4,680 → 56 | 10,969 → 7,017 | 40.7 → 34.2 | 175.0 → 167.9 |
| 40 | Plummer | 1.19 | 1.35 | 10, 25.9 kB | 10.47 → 9.97 | 222 → 120 | 2,120 → 152 | 39,489 → 34,361 | 153.3 → 145.5 | 338.9 → 313.6 |
| 80 | Plummer | 1.85 | 2.11 | 14, 25.9 kB | 7.43 → 6.98 | 289 → 133 | 3,288 → 128 | 30,193 → 22,249 | 100.3 → 88.7 | 259.4 → 235.3 |
| 160 | Plummer | 3.04 | 2.07 | 21, 25.9 kB | 5.45 → 5.00 | 377 → 131 | 5,376 → 128 | 27,993 → 14,849 | 74.9 → 55.9 | 215.5 → 194.1 |
| 40 | clusters | 2.18 | 2.42 | 12, 21.4 kB | 9.68 → 9.20 | 137 → 49 | 3,864 → 480 | 43,313 → 31,905 | 153.3 → 137.9 | 313.5 → 284.1 |
| 80 | clusters | 2.86 | 2.56 | 13, 20.1 kB | 7.40 → 6.88 | 249 → 82 | 5,064 → 520 | 34,345 → 19,993 | 102.4 → 82.4 | 247.9 → 224.7 |
| 160 | clusters | 4.67 | 2.27 | 14, 24.6 kB | 5.58 → 5.21 | 442 → 125 | 8,248 → 496 | 37,745 → 13,609 | 87.1 → 52.3 | 230.2 → 193.3 |

### 5.2 Kathleen, counted: 320 and 640 ranks oversubscribed

The same job; 320 and 640 ranks on the 4 nodes (`--oversubscribe --map-by ppr:80:node` /
`ppr:160:node --bind-to none -x PSM2_SHAREDCONTEXTS=1`), 2.5 × 10⁴ points per rank, f64
p = 8, Rotation, one evaluation per path; counted, never timed:

| ranks | workload | P5 coarse gather, MB | T6 top, MB | T6 reduction: messages, received | multipole exchange, MB | multipole messages sent | global M2M pairs | held boxes | memory model, MB | resident, MB |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 320 | cube | 2.65 | 1.50 | 19, 14.9 kB | 2.25 → 2.02 | 129 → 84 | 4,680 → 40 | 8,737 → 4,713 | 26.9 → 19.9 | 184.2 → 173.6 |
| 640 | cube | 21.20 | 1.49 | 23, 18.1 kB | 2.66 → 2.69 | 115 → 139 | 37,448 → 88 | 42,417 → 6,273 | 91.5 → 23.5 | 335.0 → 267.6 |
| 320 | Plummer | 6.09 | 2.63 | 31, 31.1 kB | 3.58 → 3.29 | 495 → 119 | 10,744 → 104 | 32,561 → 8,889 | 67.1 → 32.0 | 235.7 → 191.1 |
| 640 | Plummer | 11.63 | 2.32 | 35, 29.8 kB | 3.85 → 3.72 | 738 → 155 | 20,528 → 88 | 50,377 → 9,177 | 96.6 → 32.4 | 369.2 → 279.1 |
| 320 | clusters | 8.21 | 1.97 | 17, 23.3 kB | 3.86 → 3.62 | 857 → 196 | 14,496 → 488 | 45,305 → 8,409 | 92.6 → 29.8 | 272.3 → 198.2 |
| 640 | clusters | 12.59 | 1.82 | 13, 20.1 kB | 3.55 → 3.44 | 1,679 → 316 | 22,216 → 456 | 59,097 → 7,825 | 131.9 → 30.5 | 431.0 → 298.0 |

T6's counts equal T5's (job 530665, phase5s-t5.md §4) in every quantity both report: the
device and harness changes moved nothing on the host path. The memory per rank, new here: the
model falls 1.4–4.3× at 320 and 640 ranks and 1.2–1.7× at 160; the resident size, which
holds MPI's and PSM2's buffers (about 86 MB per rank at 160 ranks per node, scale-out.md
§8.2), 1.1–1.5× at 320 and 640. The partition map, the one O(P) term of the model,
is 0.27 MB per rank at 640 ranks on the cube (32,768 blocks), against Phase 5's 8 P = 5 kB
of partition bounds there (Phase 5's model counts the coarse gather's buffer and keys
instead, 21.7 MB there).

### 5.3 locust, timed: 8 and 72 ranks

Open MPI 5.0.10, `--map-by slot:PE=1 --bind-to core`, one rank per core; T10's sweeps
`strong` (the cube and the Plummer sphere at N = 10⁶, f64 p = 3 and 8, f32 p = 8, overlap
both) and `balance` (the cube, the Plummer sphere and the clusters at N = 10⁶, f64 p = 8,
overlap off) at `RANKS="8 72"`, 10 timed evaluations per path, through
`tools/scaling/run.sh` unchanged; each sweep Phase 5 first, then T6 (not interleaved per
launch). No other user's process in any load check (one idle `nvitop`); the 1-minute load
before each launch is the sweep's own previous launch (12–24 before the 8-rank launches,
which follow a 72-rank one).

f64 p = 8, overlap both (columns as in §5.1):

| ranks | workload | wall, blocking | upward global | T6: reduction / top | top all-gather, overlapped (P5: coarse gather) | multipole exchange | upward local | downward | leaves | wall, overlapped | C5.2 % |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | cube | 1,162.0 → 1,162.4 | 1.52 → 1.70 | 0.33 / 0.38 | 0.82 → 0.58 | 0.48 → 0.46 | 26.3 → 28.6 | 832.0 → 835.8 | 294.6 → 290.3 | 1,162.2 → 1,161.8 | 72.9 → 89.9 |
| 72 | cube | 161.2 → 155.3 | 7.60 → 2.01 | 0.66 / 1.84 | 3.91 → 2.54 | 1.75 → 1.47 | 2.9 → 3.2 | 111.5 → 112.3 | 34.0 → 34.1 | 160.0 → 154.6 | 13.2 → 17.4 |
| 8 | Plummer | 2,453.0 → 2,467.4 | 3.21 → 3.59 | 3.37 / 1.11 | 3.14 → 3.87 | 1.61 → 1.55 | 29.5 → 32.4 | 1,587.8 → 1,609.4 | 852.0 → 846.7 | 2,454.1 → 2,465.3 | 0.4 → 18.9 |
| 72 | Plummer | 350.8 → 345.0 | 5.64 → 2.37 | 0.83 / 2.17 | 5.39 → 3.81 | 3.93 → 3.10 | 3.6 → 3.9 | 229.5 → 229.1 | 140.0 → 138.2 | 351.7 → 343.2 | 9.3 → 8.3 |

The other configurations (blocking path):

| ranks | workload | wall, blocking, ms | upward global | T6: reduction / top | upward local | downward | memory model, MB | resident, MB |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | cube, f64 p = 3 | 315.5 → 314.2 | 0.18 → 0.73 | 0.59 / 0.72 | 4.0 → 4.3 | 51.0 → 49.6 | 29.8 → 29.7 | 101.5 → 100.4 |
| 72 | cube, f64 p = 3 | 41.5 → 40.2 | 1.08 → 0.72 | 0.29 / 0.55 | 0.5 → 0.5 | 7.1 → 6.6 | 7.7 → 6.5 | 77.5 → 72.2 |
| 8 | Plummer, f64 p = 3 | 513.9 → 512.6 | 0.41 → 0.47 | 0.35 / 0.35 | 4.3 → 4.8 | 128.3 → 127.8 | 42.9 → 42.6 | 121.2 → 122.1 |
| 72 | Plummer, f64 p = 3 | 81.2 → 78.8 | 0.79 → 0.59 | 0.13 / 0.54 | 0.6 → 0.6 | 19.4 → 19.2 | 10.5 → 8.3 | 107.6 → 86.1 |
| 8 | cube, f32 p = 8 | 743.0 → 735.5 | 1.44 → 0.39 | 0.22 / 0.26 | 24.4 → 27.3 | 580.4 → 570.4 | 27.0 → 26.7 | 96.7 → 95.2 |
| 72 | cube, f32 p = 8 | 131.6 → 127.2 | 5.78 → 1.34 | 0.46 / 1.22 | 2.9 → 3.3 | 105.3 → 104.4 | 10.4 → 7.4 | 85.3 → 71.5 |
| 8 | Plummer, f32 p = 8 | 1,860.7 → 1,855.6 | 2.50 → 4.51 | 2.99 / 2.04 | 27.3 → 30.2 | 1,133.4 → 1,133.9 | 44.2 → 43.5 | 127.9 → 128.3 |
| 72 | Plummer, f32 p = 8 | 319.4 → 311.5 | 4.44 → 1.84 | 0.80 / 1.58 | 3.5 → 3.8 | 216.2 → 216.4 | 15.8 → 10.5 | 128.5 → 96.3 |
| 8 | cube, f64 p = 8, overlap off (balance) | 1,161.1 → 1,162.1 | 1.52 → 1.54 | 0.67 / 0.71 | 25.9 → 28.8 | 836.3 → 833.1 | 40.1 → 39.5 | 101.9 → 102.6 |
| 72 | cube, f64 p = 8, overlap off (balance) | 160.3 → 154.9 | 7.74 → 2.08 | 0.58 / 1.87 | 2.9 → 3.4 | 111.1 → 111.8 | 18.9 → 13.0 | 89.8 → 82.6 |
| 8 | Plummer, f64 p = 8, overlap off (balance) | 2,448.6 → 2,481.3 | 2.68 → 4.40 | 3.73 / 1.15 | 29.5 → 32.4 | 1,614.4 → 1,654.4 | 67.5 → 66.0 | 134.2 → 132.0 |
| 72 | Plummer, f64 p = 8, overlap off (balance) | 346.7 → 340.1 | 5.67 → 2.29 | 0.69 / 2.08 | 3.6 → 3.8 | 227.7 → 230.1 | 28.5 → 17.9 | 115.9 → 93.5 |
| 8 | clusters, f64 p = 8, overlap off (balance) | 2,687.8 → 2,690.9 | 3.66 → 2.56 | 2.02 / 0.95 | 31.2 → 34.0 | 1,690.0 → 1,731.7 | 71.3 → 67.0 | 138.7 → 134.5 |
| 72 | clusters, f64 p = 8, overlap off (balance) | 403.5 → 392.0 | 9.66 → 3.03 | 0.85 / 2.80 | 3.9 → 4.1 | 235.0 → 233.6 | 36.7 → 19.1 | 104.5 → 81.3 |

- **At 72 ranks** "upward global" falls 1.3–4.3× in every configuration and the wall time
  1.7–3.6%; the memory model 1.2–1.9× and the resident size 1.1–1.5×.
- **At 8 ranks** the coarse gather was 0.17–0.90 MB and the replicated M2M 328–1,632 pairs,
  both cheap; T6's reduction makes up to 6 rounds of neighbourhood exchanges, and the
  first of them waits for the slower ranks' local M2M (reading 5, §2). "upward global"
  moves from 0.18 → 0.73 ms to 3.66 → 2.56 ms (0.25–3.7×; at most +2.0 ms, the Plummer
  sphere at f32 p = 8), and the wall time by −1.0% to +1.3%.
- The compute stages differ by up to 12% between the codes at 8 ranks (upward local
  +7.5% to +12%, downward −2.7% to +2.5%), more than on Kathleen and against T6 in most
  rows;
  T5 found the same direction on locust (the second launch of a pair slower,
  phase5s-t5.md §6). It is not the order (reading 6, §2): repeated at 8 ranks after the
  sweeps (10:26–10:45, the machine otherwise idle) in the orders T6, Phase 5, T6 and T5
  (`main`, 67eefab), T6, T5, the cube and the Plummer sphere at f64 p = 8:

  | Order | cube: upward local / upward global / wall, ms | Plummer: upward local / upward global / wall, ms |
  | --- | --- | --- |
  | T6, Phase 5, T6 | 29.1 / 1.98 / 1,182.5; 26.6 / 1.74 / 1,165.8; 29.4 / 2.18 / 1,172.1 | 33.0 / 2.98 / 2,479.6; 29.7 / 2.54 / 2,455.7; 33.0 / 3.96 / 2,470.2 |
  | T5, T6, T5 | 26.1 / 1.07 / 1,171.5; 29.0 / 2.35 / 1,163.5; 26.1 / 1.52 / 1,162.6 | 29.6 / 2.58 / 2,462.5; 32.4 / 3.96 / 2,456.2; 29.4 / 2.24 / 2,447.3 |

  Phase 5 and T5 agree; T6's local upward pass is 11% slower on every one of the 8 ranks
  (28.5–29.0 against 25.5–26.1 ms on the cube) and its "upward global" about 1 ms
  slower on every rank, in both orders.
- **C5.2** (the criterion's case: the cube and the Plummer sphere, N = 10⁶, f64 p = 8, 8
  ranks): 72.9% → 89.9% (cube) and 0.4% → 18.9% (Plummer). Every exposed millisecond is
  in the source exchange's wait (0.58 → 0.66 ms on the cube, 0.006 → 0.39 ms on the
  Plummer sphere), which T5 and T6 do not change; the multipole exchange's exposed wait
  is 0.001–0.004 ms in both codes, and the blocking exchanges it is measured against are
  0.7–2.2 ms (distributed-fmm.md §8.6 expects a copy to stay in the wait on locust,
  §8.4). The reduction beside it: 0.05 ms
  overlapped against 0.33 ms blocking (cube), 2.7 against 3.4 ms (Plummer).

### 5.4 locust, counted: 256 and 512 ranks oversubscribed

`tools/scaling/run.sh oversubscribed` (the N = 10⁶ cube at 256 and 512 ranks, f64 p = 8,
overlap both, one evaluation, 8 error vectors and the one-rank reference) and, launched as
the sweep launches them (`--map-by :OVERSUBSCRIBE --bind-to none`, the open-file limit
raised), the Plummer sphere and the clusters at 512 ranks without the reference; counted,
never timed:

| ranks | workload | P5 coarse gather, MB | T6 top, MB | T6 reduction: messages, received | multipole exchange, MB | multipole messages sent | global M2M pairs | held boxes | memory model, MB | resident, MB |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | cube | 2.64 | 1.54 | 17, 13.0 kB | 0.87 → 0.65 | 83 → 55 | 4,680 → 40 | 5,737 → 1,785 | 13.7 → 6.9 | 719.9 → 696.6 |
| 512 | cube | 11.57 | 1.43 | 20, 18.1 kB | 0.62 → 0.54 | 109 → 85 | 20,416 → 80 | 37,449 → 1,481 | 63.9 → 5.7 | 785.2 → 735.3 |
| 512 | Plummer | 9.41 | 2.38 | 31, 30.5 kB | 1.09 → 0.90 | 639 → 95 | 16,608 → 88 | 35,865 → 2,801 | 62.0 → 9.2 | 267.7 → 186.9 |
| 512 | clusters | 11.18 | 1.29 | 17, 22.7 kB | 1.25 → 1.14 | 1,413 → 472 | 19,736 → 592 | 37,545 → 3,137 | 89.2 → 15.9 | 318.2 → 205.9 |

The cube's resident maximum is rank 0's, which also builds the one-rank reference over
every point; the others are the evaluation's. At 512 ranks (about 2,000 points per rank)
Phase 5 held every box of the one-rank tree on the cube (37,449), T6 1,481; the memory
model falls 6.9–11×. The errors against the direct sum and the one-rank reference are the
same in both codes (the cube at 512: φ 1.889e-5 relative L2, 9.09e-16 from the
reference), and overlap is bit for bit the blocking path in all eight launches.

## 6. Checks

M3 Max (sandboxed unless noted): `cargo fmt --all`; the root checks (`cargo fmt --
--check`, `cargo clippy -- -D warnings`, `cargo clippy --examples -- -D warnings`,
`RUST_MIN_STACK=8388608 cargo test`: 719 passed, 0 failed; `cargo doc --no-deps`, no
warning) and the stricter workspace checks (`cargo clippy --workspace --all-targets -- -D
warnings`; `RUST_MIN_STACK=8388608 cargo test --workspace`: 764 passed, 0 failed);
`cargo clippy -p nd-fmm-exec --all-targets --features cpu -- -D warnings`, the same with
`cpu,metal`, `cargo check -p nd-fmm-exec --features cuda`, `cargo doc -p nd-fmm-exec
--no-deps --features cpu` (no warning); `cargo clippy -p nd-fmm-validate --all-targets --
-D warnings`; `cargo test -p nd-fmm-exec --features cpu --release` (86 passed), its
ignored tests (10 passed: accuracy, adaptive, the C4.8 gate, the device gate and
`multi_rank` at one rank, the operator sweeps) and, outside the sandbox, `--features metal
--release -- --ignored` (13 passed). The first `--features cpu` run failed one unit test
(`device::tests::traffic_by_data_adds_up` still expected 13 data kinds; fixed, 15); the
workspace runs do not build the device module. MPI runs as in §3.2.

The Kathleen runs used the branch as synced at 08:51 (before that unit-test fix and the
last documentation edits; the host and harness code were the same).

| Job | Nodes | What | Waited | Ran | Node-hours | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 587107 | 1 (node-c11a-019) | release `scaling` and the workspace's test executables, branch | 1 s | 3:52 | 0.06 | 2 of 2 steps pass |
| 587108 | 1 (node-c11a-057) | release `scaling`, Phase 5 (`06016d2`) | 1 s | 1:27 | 0.02 | 1 of 1 step passes |
| 587503 | 2 (node-c11a-[019,057]), `--exclude=node-c11a-011` | T2's correctness job | 0 s | 20:47 | 0.69 | 8 of 8 steps pass |
| 587502 | 4 (node-c11a-[065,079,096,098]), `--exclude=node-c11a-011` | the timed comparison at 40–160 ranks and the counts at 320 and 640 (§5.1, §5.2) | 4 s | 44:45 | 2.98 | 1 of 1 step passes, every launch exit 0 |
| 592032 | 4 (node-c11a-[035,042,046,125]), `--exclude=node-c11a-011` | invariance at 160 and 640 | 21 s | 2:12 | 0.15 | 6 of 6 steps pass |

3.91 node-hours in all, of T6's 20 (scale-out.md §10.5 planned 12). The Scratch quota was at
13% (33.6 of 250 GiB) before; 13 trees of earlier tasks and probes (33 GB) are still in
`src/`, more than the README's "at most two", and were left for the user to remove, with
this task's two (the branch and `detached-06016d2`).

locust (2026-10-13, Open MPI 5.0.10; the GPU idle (0%) before and after the device runs,
655 MiB held by another user's idle process): the correctness runs 08:54–09:47, the timed
sweeps 09:47–09:56, the oversubscribed counts and the invariance check 09:56–10:02, the
CUDA suites 10:02–10:25, the reverse-order repeat (§5.3) from 10:26; one run at a time.

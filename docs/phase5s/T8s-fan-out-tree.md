# Phase 5S / T8s — the fan-out below the replicated top (S8)

**Added on 2026-10-13 (T8, at the user's request): S8 is built as its own task after
T8.** T8's step 2 found that S8's modelled clause fires and its measured clause does not
(`fmm-validate/results/phase5s-t8.md` §5, §6; `docs/design/scale-out.md` §7.3). The
user signed off on 2026-10-13 that S8's remedy is built here, after T8 and not inside it.

**The problem.** A top-tree box below R goes point to point from its owner to every rank
that reads it, at most the threshold less one ranks: max(2, ⌈P/16⌉) − 1, or more where
R's cap of 4,096 boxes doubles the threshold (T5 §4). The boxes about a rank's size are
read by a roughly constant number of ranks while the threshold rises with P, so they
leave R. Their owners then send them to up to 63 ranks each at 512–640 ranks.

T8 measured it (2.5 × 10⁴ points per rank, one evaluation; counts at 512–640 ranks
oversubscribed):
- **The busiest sender's multipole messages per evaluation** (all levels) are 41–93 at 40
  ranks and 113–486 at 512–640. They grow like P^0.71–0.75 on the clusters and
  P^0.28–0.38 on the cube and the Plummer sphere.
- **At 640 ranks**, the largest fan-out of a box is 33–63, and the busiest rank sends
  139–458 messages to 59–108 neighbours, 62–208 of them to other nodes.
- **The modelled clause** (*model* at 10⁴ ranks, 4 µs per message): 1.4–28 ms per
  evaluation, against 1 % of an evaluation at 2.5 × 10⁴ points per rank: 5.5–14 ms at
  f64 p = 8 and 0.56–1.85 ms at f32 p = 3. It fires at f32 p = 3 on every workload (the
  clusters from about 600 ranks, the cube from 700–900, the Plummer sphere from
  2,000–8,000) and at f64 p = 8 under the linear fits.
- **The measured clause:** the overlapped path's multipole exposed wait at 160 Kathleen
  ranks is 0.002–0.013 ms, at most 0.004 % of the evaluation. The model charges every
  message on the blocking path's critical path; overlap hides it so far.

**The remedy (signed off on 2026-10-13, T8's brief).** Each top-tree box below R is sent
along a tree of its readers, fixed at build:
- no rank sends one box to more than a stated number F of ranks;
- a node's first reader forwards within its node;
- the forwarding runs in the level's multipole exchange, after the reduction has formed
  the level;
- every slot receives the same multipole as the direct exchange, as copies, never sums;
- R, its cap R_max = 4,096 and K = 16 stay as signed off (scale-out.md §4.3, §12,
  question 3).

Not remedies (T8's brief): a larger `TOP_MAX_BOXES`; a fixed number of readers for R,
since R's all-gather would grow with P again, the O(P) volume S2 removed; a fixed top
level (T1: 335 readers below level 4).

Read first:
- root CLAUDE.md ("MPI", the Kathleen and locust sections), fmm-plan/CLAUDE.md,
  fmm-exec/CLAUDE.md (the host-data hook, the shadow operator, overlap, the device on
  several ranks);
- docs/phase5s/README.md (requirements 2, 5–9), `docs/design/scale-out.md` §4.3 (R by
  needers), §4.5 (events and the device), §4.6 (overlap), §7.3 (T8's addendum), §10.4
  (the time model);
- T5's report §4 (where the neighbours come from), T6's report (the device's events), T8's
  report §5 and §6;
- the code:
  - `fmm-plan/src/exchange.rs`: `MultipoleExchange`, `LevelExchange`,
    `multipole_ghosts`, `TopExchange::new` (its readers' build, which gives every owner
    the readers of each of its boxes), `Posted` and `Flight`;
  - `fmm-plan/src/evaluator.rs`: `upward_global`, `exchange_multipoles`, `far_field`;
  - `HostData::{SendMultipoles, ReceivedMultipoles}`;
  - `fmm-exec`'s device mirror of those events (`device::ExchangeLists`,
    `tests/device_common::expected_evaluation`);
  - `Fmm::exchange_traffic`, and `scaling`'s multipole fields (T5's: messages sent, to
    other nodes, the largest fan-out of a box, R's threshold, neighbours);
- T8r's rlst 0.10.1 (`sparse_all_to_allv`, the fixed-route exchange), if it is merged by
  then: the forwarding instructions are a sparse exchange at build.

Do:
1. **Design note first** (a section in this task's report, reviewed before the code). It
   must settle:
   - **The tree per box.** The owner holds the readers of each of its boxes below R. Order
     them by (node, rank), with the node from the evaluator's placement. `Fmm::build`'s
     one `split_shared` knows each rank's node, and the readers' node ids travel with
     their requests at build: no P-long all-gather. The owner sends to at most F ranks,
     the first reader of each node (a k-ary tree over the nodes when they are more than
     F). Each node's first reader forwards within its node, again at most F per rank.
     State F (for example 8) and the tree's depth, at most ⌈log_F r⌉ + 1 for r readers.
   - **Load across ranks.** A rank's messages are its distinct destinations, summed over
     the levels. A tree that moves the fan-out from owners to forwarders must not build a
     new busiest sender, so the choice of forwarders spreads the load, deterministically
     on every rank. State the rule and its bound on a rank's messages per level.
   - **The exchange.** Today every level's exchange is one rlst ghost communicator, built
     at `MultipoleExchange::new` and posted at once after `upward_global`. A forwarder
     resends a box it received, not one it computes (`LevelExchange::new` rejects that
     today as `UnheldKey`). Two stages per level: from the owners (the tree's first
     hops), then from the forwarders, posted as soon as the forwarded box has arrived.
     That happens in the overlapped path's `test` calls and in a second round of the
     blocking path. Every slot is written exactly once with the owner's bits.
   - **The device and the events.** Forward from the host receive buffer, which every
     received multipole passes through. The device then sees only the existing
     `SendMultipoles` and `ReceivedMultipoles(l)` with unchanged receive lists, and needs
     no new event. If the design needs one (for example a forwarder's send reading a slot
     the device wrote), agree it with T6's device mirror first: its sizes in
     `Fmm::exchange_sizes`, its transfers in `tests/device_common`, and the shadow
     operator's coverage.
   - **The collectives at build.** The forwarding instructions (owner to each reader: from
     whom it receives each box and to whom it forwards it) go by one sparse exchange at
     build, rlst 0.10.1's `sparse_all_to_allv` or a ghost-communicator build. Every rank
     enters every collective; errors are agreed (requirement 6). Count the new
     collectives in scale-out.md §8.1's table.
   - **The cost.** The extra hop adds one latency per level to the blocking path. Model
     it against the messages it saves (4 µs per message, scale-out.md §10.4), at 160
     ranks measured and at 10³–10⁴ modelled.
2. **Count the alternative of aggregating levels, before building.** The busiest sender's
   messages are summed over the levels (the clusters at 640 ranks: 340–458 messages to
   103–108 neighbours). Sending one message per destination for all levels together
   would cut that by about the number of levels, but it removes the per-level waits of
   the overlapped path. Count it from the plan's lists (a probe, as T8's, not committed),
   alone and with the tree, at 512 (locust) and 640 (Kathleen) ranks. Report it beside
   the tree. Build only the tree unless the sign-off at this step adds the aggregation.
3. **Build the tree** in `nd-fmm-plan` (`MultipoleExchange`, both paths), with the
   evaluator's order unchanged:
   - blocking: `exchange_multipoles`;
   - overlapped: `far_field`, the forwards posted from its `test` calls.

   The tree is on by default once the checks pass. Keep the direct exchange as an option
   on the evaluator and `FmmBuilder`, as the test oracle. Report
   the tree in `Fmm::exchange_traffic`: the messages per rank and level, the largest
   fan-out of a box, the depth, and the forwarded slots. Add these to `scaling`'s
   summary.
4. **Tests** (nd-fmm-plan with the recording operator, nd-fmm-exec with the shadow operator
   and the device):
   - **Slot by slot:** every received slot of every level equals the direct exchange's,
     with the bits of the owner's multipole. Check it on the plan trees of
     `tests/mpi_regressions.rs` (graded, coincident points, one empty rank) at 1, 2, 4
     and 8 ranks, at 512 oversubscribed on locust and at 160 on Kathleen.
   - **Bit for bit:** the Phase 5 suites, `tests/mpi_exec.rs` (blocking and overlapped,
     every thread count), `tests/multi_rank.rs`, and T2's invariance at 1, 2, 4, 8 ranks,
     512 against 64 on locust, and on Kathleen 160 against 20 and 640 against 80.
   - **The hook:** the shadow check unchanged, and different with the hook disabled.
   - **The device on several ranks:** `tests/device_ranks.rs` (CUDA on locust, Metal on
     the M3 Max) and `tests/device_common`'s transfer formula, unchanged if no event is
     added.
5. **Measure** (release; locust timed to 72 ranks and counted to 512; Kathleen timed at
   40–160 and counted at 320–640; 2.5 × 10⁴ points per rank, the cube, the Plummer
   sphere and the clusters, f64 p = 8 and f32 p = 3, both paths):
   - T8's S8 table and fits again, with the tree;
   - the busiest sender's messages and the largest fan-out of a box against P;
   - the exposed wait and the blocking stage at 160 ranks;
   - the modelled clause at 10³ and 10⁴ ranks against T8's thresholds.

   Name the job and the nodes, and report the node-hours (budget 20, no job larger than 4
   nodes).
6. **Write it up:** this task's report (`fmm-validate/results/phase5s-t8s.md`),
   scale-out.md §7.3 (S8: built, with its numbers) and §8.1 (the collectives), the
   fmm-plan and fmm-exec CLAUDE.md files, and the README's checklist.

Tests that define done:
- slot-by-slot equality with the direct exchange, and every suite and invariance check
  above bit for bit;
- at 512 (locust) and 640 (Kathleen) ranks: no box sent by one rank to more than F ranks,
  and the busiest sender's messages below T8's measurement on every workload;
- the modelled clause at 10⁴ ranks reported against T8's thresholds per workload and
  precision. Where the tree alone does not bring a workload below its threshold (the
  cube at f32 p = 3 may not: its growth is not mainly fan-out), report it with step 2's
  counts for the sign-off;
- the multipole exposed wait at 160 Kathleen ranks reported, and the blocking path's cost
  of the extra hop measured there.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks;
- fmm-exec's device checks (CLAUDE.md there; CUDA on locust, Metal by hand);
- (Kathleen) the correctness job of T2 at 2 nodes, and invariance at 160 and 640 at 4.

Do not:
- change R, its cap, K, the reduction, O1–O4, the second cut, or any output bit;
- sum on a forwarder (copies only);
- add a per-evaluation collective, or anything per evaluation that grows with P;
- use `unsafe`, raw MPI FFI or a fork of rlst. An rlst addition goes upstream as in T8r,
  asked for first.

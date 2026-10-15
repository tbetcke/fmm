# The fan-out below the replicated top (Phase 5S T8s)

S8, as docs/phase5s/T8s-fan-out-tree.md asks: each multipole that its owner sends to many
ranks goes along a tree of its readers, fixed at build. Run on 2026-10-14 on main at
7292d6a, on the M3 Max, locust (GH200) and Kathleen:
- steps 1 and 2: the design note (§1) and the counts that decide what is built (§2),
  signed off by the user (§3);
- step 3: the tree, built (§5); step 4: the tests and checks (§6); step 5: measured (§7).

**Counted** runs are oversubscribed or untimed. ***Simulated*** marks every figure of §1–§2
for a tree or for level aggregation: the probe (§4) dumps every multipole the direct
exchange sends, by rank, level and destination, and an offline simulation applies a rule
to that dump. §7 measures the built tree. Every figure at 10³–10⁴ ranks is a ***model***,
its inputs stated with it. locust was idle but for these launches (load in each output
file). Kathleen nodes are exclusive; the jobs are in §4 and §8.

## Summary

**Steps 3–5 (built and measured; §5–§7):**
- **Built:** the forwarding tree is the default of the multipole exchange
  (`MultipoleRouting::Tree`, F = 8, `FmmBuilder::multipole_routing`); the direct exchange
  stays as the oracle. No new event; no collective per evaluation; the build has two sparse
  exchanges in place of one rlst build per level.
- **Bit for bit:** every check of the brief passes (§6): slot by slot against the direct
  exchange at 1–8 ranks (with a fan-out of 2 so that it forwards), the Phase 5 suites, the
  shadow and device checks (CPU runtime, Metal, CUDA), invariance at 512 against 64 (locust)
  and at 160 and 640 (Kathleen), T2's correctness job at 2 nodes.
- **Measured, 512 and 640 ranks:** the busiest sender's messages 98–244 against T8's 113–486
  (9–51 % fewer), no box sent by one rank to more than 8 ranks (T8: 27–63), depth 2–3,
  messages to other nodes at 640 cut from 62–208 to 44–82 (§7.1).
- ***Model* at 10⁴ ranks:** f64 p = 8 below its threshold on every workload; f32 p = 3
  still fires on the cube and the clusters (§7.2), as signed off.
- **Timed:** the overlapped path's exposed wait stays negligible (0.002–0.21 ms at 160 Kathleen
  ranks, at most 0.12 %); the blocking stage is 0.05–1.8 ms slower at 160 ranks, which a control without
  forwarding shows to be the point-to-point exchange's receive copy, not the hop (§7.3).
- Kathleen used **3.73 node-hours** in all (§8), within the 20 budgeted.

**Steps 1–2 (design and counts):**

- **The probe reproduces T8 exactly.** At 512 locust ranks the busiest sender's multipole
  messages are 117, 155 and 486 at f64 p = 8 and 113, 157 and 464 at f32 p = 3 (cube,
  Plummer, clusters). At 640 Kathleen ranks they are 142, 184, 340 and 139, 169, 458.
  These are T8's figures and the harness's own (§2.1). Refitted over the nine counts from 64
  to 640 ranks, the direct exchange gives T8's fits again (cube 10.3 P^0.38, clusters
  4.2 P^0.71).
- **The fan-out is not one box's** (§1.1). The busiest rank owns many boxes per level,
  each read by up to 63 ranks, each by a different set: 133–165 distinct readers per level
  on the clusters at 512. A quarter of the boxes with more than 8 readers (5,374 of 21,899) lie inside blocks, not
  in the top tree.
- **Proposed rule** (§1.2): F = 8, any box with more than F readers goes by tree, and an
  owner's boxes of a level share their first hops. Those hops are split across nodes in
  proportion to their readers, at least one per node. *Simulated* (§2.1), the busiest
  sender's messages at 512 and 640 ranks:
  - cube 117 → 97 and 142 → 118;
  - Plummer 155 → 126 and 184 → 150;
  - clusters 486 → 238 and 340 → 240;
  - at f32 p = 3: 113 → 97, 139 → 116, 157 → 139, 169 → 142, 464 → 223, 458 → 247.

  Every case lies below T8's measurement, at most 8 copies of a box per rank and depth 2–3.
  Messages to other nodes fall by 29–66 % (181 → 61 on the clusters).
- **Against the brief's literal rule** (one tree per box, a node's first reader as head):
  the brief's rule gives 103–328 at 512 and 133–327 at 640. A strict once-per-node variant
  of the proposal gives 137–325 at 640, one above T8's figure on the Plummer sphere at f32
  (170 > 169). The proposal is better on every workload (§2.1).
- **Level aggregation** (step 2, §2.2): one message per destination for all levels together.
  - Alone it gives 51–223 messages at 512–640 ranks. It halves the cube and the Plummer
    sphere but grows almost linearly on the clusters (P^0.94–0.96): it merges the levels
    but not the fan-out.
  - With the tree it gives 61–132.
  - It removes the per-level waits of the overlapped path.
- ***Model* at 10⁴ ranks** (4 µs per message, the thresholds of T8 §5.2; §2.3): with the
  tree, every workload at f64 p = 8 is below 1 % of an evaluation under both fits:
  - cube 1.04 or 4.31 ms against 5.49;
  - Plummer 0.89 or 3.72 against 12.12;
  - clusters 3.56 or 11.63 against 14.25.

  At f32 p = 3 the trigger still fires on the cube (1.02 or 4.16 ms against 0.56) and the
  clusters (3.25 or 11.18 against 1.85), and on the Plummer sphere under the linear fit. The
  cube's growth is not fan-out: the boxes with at most 8 readers alone, which no tree
  touches, reach 0.49 or 1.83 ms against 0.56. Only aggregation reaches below that floor
  (cube with the tree 0.56 or 2.44 ms).
- **No new event, no per-evaluation collective.** The build replaces one rlst ghost build
  per level by two sparse exchanges (§1.6). Kathleen used **0.53 node-hours** (§4).

## 1. Design note (step 1)

### 1.1 Where the fan-out comes from

The multipole exchange of level l sends each box of level l that a V-row of level l or a
W-row of level l − 1 names on another rank, without what the reduction and R deliver. It
goes from the rank that computes it to every rank that reads it, through one rlst ghost
communicator per level: one message per (sender, destination, level). A rank's messages
per evaluation are its distinct destinations summed over the levels
(`Traffic::messages_sent`, T5's busiest sender). On the clusters at f64 p = 8 and 512
locust ranks (busiest sender 486):

- **Many reader sets per owner.** The busiest rank owns many top-tree boxes per level, each
  read by up to 63 ranks (R's threshold is 64), each by a different set of ranks. The
  union of the readers of its boxes is 133 ranks at level 4, 165 at level 5 and 106 at
  level 6; 133, 145 and 78 of them get only top-tree boxes from it. A tree per box bounds
  each box's copies, but the owner still messages a different set of first hops for each
  box. The messages to remove are the union, not one box's fan-out.
- **Trees inside blocks too.** 21,899 boxes have more than 8 readers: 16,525 top-tree boxes
  and 5,374 inside blocks (a small rank in a dense cluster is read by many). "No rank sends
  one box to more than F ranks" therefore concerns every box of the exchange.
- **A floor no tree touches.** The boxes with at most 8 readers give the busiest rank 128
  messages, more than half of what any tree with F = 8 leaves (§2.1, "floor").
- **3–7 levels carry tree boxes** at 512 ranks (cube 4–6; Plummer 2 and 4–9; clusters 3–9).

### 1.2 The rule

Fixed at build, on each owner, from the readers of each of its boxes and each reader's
node; F = 8 (`TREE_FAN_OUT`, §3, question 2).

1. **Which boxes.** Every box of the multipole exchange with more than F readers, on any
   level. A box with at most F readers goes direct, as today.
2. **The parts of an owner's level.** For owner o and level l, let U be the union of the
   readers of o's tree boxes of l, ordered by (node, rank), each rank's node from the
   evaluator's placement (§1.5). U is split into at most F parts:
   - **More than F nodes:** F runs of whole nodes, balanced by readers.
   - **2 to F nodes:** each node gets one part, and the parts left over go to the nodes
     with the most readers (largest remainder of their share of F). A node's readers are
     cut into its parts as runs of near-equal size.
   - **One node:** F runs if it is the sender's node, else one part. A node other than the
     sender's is never cut by the sender: one head there splits it.
3. **Priority of a reader:** the number of o's tree boxes of l it reads (most first),
   then its rank.
4. **The tree of box b.** For each part holding readers of b, o sends b to b's reader of
   highest priority in the part, the part's *head* for b (stage 0). The head forwards b to
   b's other readers in the part (stage 1): directly if they are at most F and all on its
   node, otherwise by the same split and priority to sub-heads (stage 2), and so on.

Properties:
- **No rank sends one box to more than F ranks** (at most F parts, at most F per head and
  stage). *Simulated*: largest fan-out 8 everywhere, against 27–63 direct.
- **Depth** at most ⌈log_F r⌉ + 1 stages for r readers, one more where a node's share is
  uneven. *Simulated*: 2–3 at 512 and 640 ranks.
- **The owner never messages more ranks than in the direct exchange**: its first hops are
  readers.
- **Every reader receives each box exactly once**, an exact copy of the owner's chunk,
  never a sum.
- **Heads by priority.** A reader most central to the owner's boxes heads its part for all
  of them, so one first hop carries several boxes in one message. Different owners have
  different central readers, so the heads spread.

**Departures from the brief, for review (§3, question 1).** The brief's tree is per box,
with the first reader (lowest (node, rank)) of each node as head and each node reached
once.
- **Heads by priority, shared by an owner's boxes of a level.** Per box with lowest-rank
  heads, the clusters' busiest sender falls only to 328 (512) and 306 (640). Sharing the
  heads by priority gives 238 and 240.
- **Each node's share of F, not one head per node.** With exactly one head per node
  (strict), every owner on another node sends to the same few heads of a node. Those heads
  then forward for many owners, and at 4 nodes the busiest rank becomes a forwarder: 183 of
  its 312 messages on the clusters at 640. Strict gives 137, 153, 312 (f64) and 136, 170,
  325 (f32) at 640, and is worse than direct at 320 on the cube and the Plummer sphere
  (98 against 84, 126 against 116). Proportional parts give 118, 150, 240 and 116, 142,
  247, with messages to other nodes 44–81 against strict's 39–65 and direct's 62–208.
  Within its node a forwarder still forwards; a node now receives a box from as many heads
  as its share of the F parts (at most 7, *simulated*) instead of once.

### 1.3 Load across ranks

A rank's messages per level are its distinct (destination, stage) pairs. A forward is a
message of its own, one hop after the rank's stage-0 messages. Per rank r and level l,
with D(r, l) the destinations of r's boxes with at most F readers (unchanged by any tree):

- **As owner:** at most D(r, l) plus its heads, which are readers. So the stage-0
  destinations are a subset of the direct exchange's.
- **As head:** for each (owner, part) it heads, at most the part's other readers, and at most
  F copies of a box per stage.
- **Deterministic:** each owner forms its trees alone from its own boxes and the readers'
  nodes. No collective chooses them; every run gets the same trees.

*Simulated*, the busiest rank after the tree, clusters f64 p = 8 at 640 Kathleen ranks
(240 messages):
- 119 messages for its own boxes with at most 8 readers;
- 38 to the heads of its trees;
- 97 forwards.

On the cube at 512 locust ranks (97 messages) the split is 59, 21 and 24. The mean falls
with the maximum: at 640, 96 → 81 (cube), 103 → 90 (Plummer) and 134 → 102 (clusters).
The tree can add a few messages where few boxes exceed F: at 64–320 ranks up to +4
(Plummer 80 → 84 at 64 ranks; cube 84 → 87 at 320).

Not proposed, *simulated* on the same data (locust, 512):
- **Preferring as heads the owner's direct destinations:** mixed (clusters 238 → 254,
  Plummer 126 → 124).
- **Merging a forwarder's leaf copies** (those its receiver does not forward) into its later
  message to the same rank: 2–8 % fewer (clusters 238 → 218, cube 97 → 94). The cost is
  delaying those copies a hop; question 2.
- **A head chosen by a hash of the key:** spreads the forwarding but merges nothing (453).

### 1.4 The exchange

Today each level is one rlst `GhostCommunicator`: the blocking path calls its neighbour
all-to-all, the overlapped path posts a receive and a send per neighbour on its graph
communicator (`Posted`, `Flight`). Two things rule it out for the tree:
- a forwarder resends a box it received but does not compute, which `LevelExchange::new`
  rejects (`UnheldKey`);
- rlst's communicator knows one sender per ghost.

The tree gets its own routes, and the direct exchange stays as it is:

- **`MultipoleRouting`** on `MultipoleExchange::new`: `Direct` (today's rlst communicators,
  unchanged, the oracle) or `Tree { fan_out }` (the default; the node an argument of the build).
  - The public lists keep their meaning in both:
    - `send_boxes(l)`: the boxes sent from the store, a box sent to two ranks twice;
    - `receive_boxes(l)`: every box received, strictly ascending;
    - `receive_buffer(l)`: in that order.
  - The two exchanges' receive lists are equal, and the slot-by-slot test compares their
    receive buffers.
- **Routes.** Per level and stage s a rank has receive messages and send messages.
  - Each message carries its boxes ascending by key, so sender and receiver derive the same
    layout from the instructions (§1.6) alone.
  - Stage-0 sends read the store, gathered by `send_boxes` as today.
  - Stage-s sends read the chunks received at stage s − 1, from the staging buffer the
    receives land in. Each send lists the receive messages it reads.
- **Messages** go point to point on a duplicate of the plan's communicator, tag 32 s + l
  (levels ≤ 16), one per (sender, receiver, level, stage).
  - The stage is in the tag because a rank can receive a stage-2 input before a stage-1
    one and post its forwards out of stage order. MPI's non-overtaking order would then
    pair them wrongly.
  - F ≥ 2 keeps the depth, and so the tag, small.
- **Blocking** (`exchange_multipoles`, `forward(l)`):
  1. Per level, post every receive of every stage and the stage-0 sends.
  2. For each stage s, wait for the stage-s receives, then post the stage-(s + 1) forwards.
  3. Wait for the sends and scatter the receive buffer into the slots.

  No collective. Every rank calls the levels in the same order. Deadlock-free by induction
  over (level, stage): a stage-s message depends only on stage-(s − 1) receives of its
  level, and a rank that has left level l has posted every send of l.
- **Overlapped** (`far_field`, `forward_all_overlapped`):
  - After `SendMultipoles`, post every receive of every level and stage and every stage-0
    send, as today.
  - Each `test` (after every level call) also posts each forward whose input messages have
    completed.
  - `wait(l)` takes level l's stages in order: wait for the stage-s receives, post the
    stage-(s + 1) forwards of l, repeat. It then copies and scatters as today.
  - So a forward goes out as soon as its box has arrived, in a `test` while the rank works,
    or in a wait. The order of level calls and events is unchanged.
  - The blocking all-gather of R inside the work holds forwards back until it returns; it is
    a collective every rank enters, so nobody waits on them meanwhile.
- **Every slot written once.** Each receive-buffer position belongs to exactly one receive
  message, copied with `copy_from_slice`. The stores get the direct exchange's bits whatever
  the arrival order.

### 1.5 The node of a rank

`Fmm::build` splits the communicator by node once (`split_shared`). A rank's node id is the
world rank of the node communicator's rank 0, from rsmpi's `Group::translate_rank`, which
is local and adds no collective. The id travels with the rank's requests at build (§1.6),
so no P-long array is formed. `nd-fmm-plan` knows no placement:
- the node is an argument of `Evaluator::with_routing`, given by the caller;
- `Evaluator::new` takes each rank as its own node, the one-node split (§2.1: on one node
  the split is by runs);
- `Fmm::build` passes the real id;
- the plan's tests pass made-up nodes (rank / 2) to exercise the node split at 4–8 ranks.

### 1.6 The collectives at build

`MultipoleExchange::new` with `Tree`, in order (every rank enters each, an empty rank with
empty data):

| Step | Collective | Carries |
| --- | --- | --- |
| validity, sizes agreed | all-reduce (unchanged) | |
| the tree's communicator | communicator duplicate (new) | |
| requests | rlst 0.10.2 `sparse_all_to_allv` (new; replaces the per-level rlst ghost builds) | to each owner: the requesting rank's node id, then the keys it reads from that owner, all levels |
| instructions | `sparse_all_to_allv` (new) | to each reader, per requested key in request order: its sender and stage, and the ranks it forwards the key to |
| layout agreed | all-reduce (unchanged) | an `UnheldKey` found on any rank (a requested box the owner does not compute) |

- **Fewer collectives.** A build with the tree has a duplicate and two sparse exchanges
  in place of one rlst ghost build per level (up to 17). Each of those is a consensus round,
  two graph creations and two neighbour all-to-alls.
- **No new agreement** (README, "Errors"): the errors ride on the existing layout all-reduce.
- **Per evaluation:** no collective at all; the blocking path's neighbour all-to-alls become
  point to point.

Step 6 adds these rows to scale-out.md §8.1.

### 1.7 The device and the events

Forwards read the host staging buffer that every received multipole passes through, never
a store. The device therefore sees only the events it already handles:
- `SendMultipoles`, with the stage-0 boxes, fewer copies than before;
- `ReceivedMultipoles(l)`, with the same `receive_boxes(l)` and packed `receive_buffer(l)`
  as the direct exchange.

No event is added, and the shadow operator's coverage is unchanged. The transfer formula
of `tests/device_common` is unchanged; its download sizes shrink with the duplicates
`send_boxes` loses (`Fmm::exchange_sizes`).

### 1.8 Options and reports

- **`nd-fmm-plan`:**
  - `MultipoleRouting::{Direct, Tree { fan_out }}` (default the tree) and
    `TREE_FAN_OUT = 8`;
  - `Evaluator::with_routing(…, routing, node)`, with this rank's node a separate argument
    (it is per rank, not a choice of routing; built so in step 3, the note's draft had it
    inside `Tree`); `Evaluator::new` keeps its signature (the tree, F = 8, the rank as its
    node);
  - `MultipoleExchange::{routing, depth, forwarded, fan_out, destinations, sources}`;
  - `communicator(l)` goes: it exists only for `Direct`, and its two users
    (`Fmm::exchange_traffic` and `mpi_regressions`) move to the new accessors.
- **`nd-fmm-exec`:** `FmmBuilder::multipole_routing` (default the tree, F = 8, the node from
  `split_shared`). `ExchangeTraffic` gains per level the messages by stage, the depth, the
  forwarded slots and the largest fan-out of a box over every stage.
- **`scaling`:** `--routing direct|tree`. The summary adds `multipole_depth_max` and
  `multipole_forwarded_max`; `multipole_messages_sent_max` and `multipole_fan_out_max`
  count every stage.

### 1.9 The cost of the extra hop

**Blocking path.** The tree adds (depth − 1) latencies per level that carries tree boxes.
At 512–640 ranks that is at most 2 hops × 1.5 µs (scale-out.md §10.3) × 7 levels =
21 µs per evaluation. Against it, the saving is (direct − tree) messages × 4 µs on the
busiest sender:
- 512 ranks: 0.08 ms (cube), 0.12 ms (Plummer), 0.99 ms (clusters);
- 640 ranks: 0.10, 0.14 and 0.40 ms;
- 10⁴ ranks (*model*, power fits at f64 p = 8): 0.39, 0.54 and 8.3 ms. The depth there is
  at most ⌈log_8 r⌉ + 1 = 5 for r < 4,096, so at most 4 hops × 10 levels × 1.5 µs = 60 µs.

**Overlapped path.** The forwards ride on the `test` calls of the downward pass, hidden
where the exchange is hidden today (T8 §5.1: 0.002–0.013 ms exposed at 160 ranks). Step 5
measures both paths at 160 Kathleen ranks.

### 1.10 Tests (step 4)

- **`mpi_regressions`:** every scenario also builds the tree exchange with F = 2 and the
  made-up nodes, so that forwarding of depth 2–3 runs at 4–8 ranks.
  - Every received slot of every level matches the `Direct` exchange of the same plan, bit
    for bit, blocking and overlapped.
  - No box is sent by a rank to more than F ranks, and each slot arrives once.
  - The recorder and `check_against_one_rank` run on the default, the tree.
- **`nd-fmm-exec`:** `mpi_exec`, `multi_rank`, `invariance`, the shadow check and the device
  checks run on the default, unchanged. One `mpi_exec` scenario compares `Direct` with the
  tree bit for bit.
- **At scale** (`scaling --routing`): 512 locust ranks and 160 and 640 Kathleen ranks
  through the invariance runs; the counts of §2 measured there, no longer simulated.

## 2. Step 2: the tree and level aggregation, counted

2.5 × 10⁴ points per rank, f64 p = 8 `Rotation` and f32 p = 3 `Dense`, the second cut on
(default), one build per run. (l) locust: one node, oversubscribed above 72 ranks.
(K) Kathleen: 4 nodes, 40–160 ranks one per core, 320 and 640 oversubscribed at 80 and
160 per node. Counts depend only on the partition, so the two machines form one series,
as in T8.

### 2.1 The tree

The busiest sender's multipole messages per evaluation. Columns:
- **direct (T8):** the probe's count, equal to the harness's and to T8's;
- **brief's rule:** one tree per box, F = 8, the lowest-ranked reader of each node as head;
- **node once:** the proposal with exactly one head per node;
- **proposed:** §1.2;
- **F = 16:** the proposal with F = 16;
- **depth, forwarded slots, to other nodes:** of the proposal;
- **floor:** the busiest sender's messages for its boxes with at most 8 readers alone.

All *simulated* except the direct column:

| precision | workload | ranks | direct (T8) | brief's rule | node once | **proposed** | F = 16 | depth | forwarded slots | to other nodes, direct → proposed | floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| f64 p = 8 | cube | 512 (l) | 117 | 103 | – | **97** | 108 | 2 | 104,410 | – (one node) | 73 |
| f64 p = 8 | cube | 640 (K) | 142 | 134 | 137 | **118** | 133 | 3 | 174,105 | 62 → 44 | 80 |
| f64 p = 8 | Plummer | 512 (l) | 155 | 137 | – | **126** | 139 | 2 | 155,320 | – (one node) | 89 |
| f64 p = 8 | Plummer | 640 (K) | 184 | 153 | 153 | **150** | 158 | 3 | 220,165 | 121 → 70 | 98 |
| f64 p = 8 | clusters | 512 (l) | 486 | 328 | – | **238** | 244 | 3 | 255,546 | – (one node) | 128 |
| f64 p = 8 | clusters | 640 (K) | 340 | 306 | 312 | **240** | 229 | 3 | 298,362 | 181 → 61 | 124 |
| f32 p = 3 | cube | 512 (l) | 113 | 105 | – | **97** | 105 | 2 | 103,611 | – (one node) | 69 |
| f32 p = 3 | cube | 640 (K) | 139 | 133 | 136 | **116** | 130 | 2 | 174,956 | 64 → 45 | 80 |
| f32 p = 3 | Plummer | 512 (l) | 157 | 145 | – | **139** | 147 | 2 | 156,973 | – (one node) | 93 |
| f32 p = 3 | Plummer | 640 (K) | 169 | 161 | 170 | **142** | 153 | 3 | 221,904 | 114 → 67 | 90 |
| f32 p = 3 | clusters | 512 (l) | 464 | 326 | – | **223** | 219 | 3 | 254,816 | – (one node) | 125 |
| f32 p = 3 | clusters | 640 (K) | 458 | 327 | 325 | **247** | 264 | 3 | 305,543 | 208 → 81 | 149 |

Against P, direct → proposed:

| precision | workload | 40 (K) | 64 (l) | 72 (l) | 80 (K) | 128 (l) | 160 (K) | 256 (l) | 320 (K) | 512 (l) | 640 (K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 8 | cube | 48 → 50 | 59 → 57 | 51 → 51 | 56 → 58 | 64 → 64 | 73 → 74 | 79 → 78 | 84 → 87 | 117 → 97 | 142 → 118 |
| f64 p = 8 | Plummer | 93 → 91 | 80 → 84 | 95 → 95 | 106 → 107 | 112 → 114 | 106 → 94 | 139 → 125 | 116 → 110 | 155 → 126 | 184 → 150 |
| f64 p = 8 | clusters | 41 → 41 | 94 → 94 | 96 → 96 | 71 → 71 | 150 → 127 | 141 → 127 | 261 → 169 | 212 → 161 | 486 → 238 | 340 → 240 |
| f32 p = 3 | cube | 48 → 46 | 60 → 58 | 51 → 52 | 57 → 58 | 66 → 66 | 73 → 75 | 84 → 83 | 84 → 87 | 113 → 97 | 139 → 116 |
| f32 p = 3 | Plummer | 91 → 92 | 80 → 81 | 90 → 90 | 89 → 88 | 106 → 104 | 106 → 100 | 134 → 121 | 130 → 115 | 157 → 139 | 169 → 142 |
| f32 p = 3 | clusters | 44 → 44 | 90 → 90 | 94 → 94 | 97 → 97 | 154 → 131 | 137 → 125 | 262 → 174 | 236 → 171 | 464 → 223 | 458 → 247 |

- **The done criteria hold.** At 512 and 640 ranks the proposal is below T8's measurement
  on every workload and precision, by 11–52 %, with at most 8 copies of a box per rank.
- **Little to gain below 128 ranks**, where R's threshold leaves few boxes with more than
  8 readers. There the tree changes the count by −4 to +4 (the brief's rule likewise; at
  320 the cube gains 3).
- **F = 16** saves a stage but sends more at 512–640 on 10 of the 12 rows.

### 2.2 Level aggregation (the brief's alternative)

One message per destination for all levels together (and, with the tree, per destination
and stage). Busiest sender, *simulated*:

| precision | workload | ranks | direct | aggregated | proposed tree | tree and aggregated |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f64 p = 8 | cube | 512 | 117 | 51 | 97 | 61 |
| f64 p = 8 | cube | 640 | 142 | 63 | 118 | 64 |
| f64 p = 8 | Plummer | 512 | 155 | 61 | 126 | 68 |
| f64 p = 8 | Plummer | 640 | 184 | 69 | 150 | 79 |
| f64 p = 8 | clusters | 512 | 486 | 210 | 238 | 107 |
| f64 p = 8 | clusters | 640 | 340 | 197 | 240 | 127 |
| f32 p = 3 | cube | 512 | 113 | 53 | 97 | 61 |
| f32 p = 3 | cube | 640 | 139 | 61 | 116 | 65 |
| f32 p = 3 | Plummer | 512 | 157 | 62 | 139 | 73 |
| f32 p = 3 | Plummer | 640 | 169 | 69 | 142 | 79 |
| f32 p = 3 | clusters | 512 | 464 | 210 | 223 | 110 |
| f32 p = 3 | clusters | 640 | 458 | 223 | 247 | 132 |

- **Aggregation alone** cuts the messages by about the number of levels (2.1–2.7 times on
  the cube and the Plummer sphere). On the clusters it leaves the fan-out, and it grows
  like P^0.94–0.96 (§2.3).
- **The tree alone** keeps the per-level waits.
- **Both together** are the lowest on the clusters and on average. With the tree, the stages
  stay separate messages, which costs aggregation part of its gain on the cube (61–65
  against 51–63).
- **The overlapped path loses its per-level waits** under aggregation. Every level's
  multipoles are final when the exchange is posted, so the sends lose nothing. But the first
  wait (level 2) then waits for each sender's whole message, every level's boxes, before the
  downward pass starts.

### 2.3 The model at 10³ and 10⁴ ranks (*model*)

Fits over the nine counts from 64 to 640 ranks (T8's ranks: 64, 72, 128, 256, 512 locust;
80, 160, 320, 640 Kathleen), power law and line. The busiest sender's messages at 4 µs per
message. The threshold is 1 % of an evaluation at 2.5 × 10⁴ points per rank, T8 §5.2's.
"Fires at" solves fit × 4 µs = threshold.

| precision | workload | direct at 10⁴, power / linear (T8) | **tree** at 10³ | **tree** at 10⁴, power / linear | tree and aggregated at 10⁴ | floor at 10⁴ | 1 % | tree fires at (power; linear) |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| f64 p = 8 | cube | 1.43 / 5.98 ms | 0.50 / 0.61 ms | 14.3 P^0.32: **1.04** / **4.31** ms | 0.56 / 2.51 | 0.54 / 2.03 | 5.49 ms | 2.0 × 10⁶; 12,857 |
| f64 p = 8 | Plummer | 1.43 / 6.21 | 0.59 / 0.70 | 44.4 P^0.17: **0.89** / **3.72** | 0.73 / 2.95 | 0.41 / 1.05 | 12.12 | never; 34,990 |
| f64 p = 8 | clusters | 11.89 / 23.79 | 1.21 / 1.44 | 12.0 P^0.47: **3.56** / **11.63** | 3.20 / 7.04 | 0.91 / 3.16 | 14.25 | 193,481; 12,314 |
| f32 p = 3 | cube | 1.36 / 5.67 | 0.50 / 0.60 | 15.1 P^0.31: **1.02** / **4.16** | 0.56 / 2.44 | 0.49 / 1.83 | 0.56 | 1,411; 888 |
| f32 p = 3 | Plummer | 1.55 / 6.21 | 0.63 / 0.74 | 32.6 P^0.23: **1.07** / **4.38** | 0.86 / 3.23 | 0.43 / 1.23 | 1.47 | 39,632; 2,803 |
| f32 p = 3 | clusters | 14.56 / 28.25 | 1.19 / 1.42 | 14.6 P^0.44: **3.25** / **11.18** | 3.29 / 7.26 | 0.97 / 3.82 | 1.85 | 2,747; 1,399 |

**f64 p = 8: below the threshold on every workload with the tree, under both fits.** The
clusters fell from 11.9 / 23.8 ms to 3.6 / 11.6 ms against 14.25, and from P^0.71 to P^0.47.
Against the cube's 5.49 ms (the brief's yardstick) the clusters' linear fit still crosses,
at 10⁴ ranks.

**f32 p = 3: the tree does not bring the cube and the clusters below the threshold**, as
the brief expected for the cube. It moves the crossing from 600–900 ranks to
900–2,700, and the Plummer sphere's from 2,000–8,000 to 2,800–40,000.
- **The floor fires on its own** under the linear fit: 1.83 ms on the cube, against a
  threshold of 0.56 ms (140 messages). Those are the boxes with at most 8 readers, which no
  tree changes, so nothing per level reaches that threshold.
- **Aggregation with the tree** brings the cube to the threshold under the power law (0.56
  ms) but not under the line (2.44 ms). It leaves the clusters at 3.3 / 7.3 ms against
  1.85.
- **Caveats** (T8 §5.2's, unchanged):
  - The model charges every message at 4 µs on the blocking path's critical path, while
    the overlapped path exposed 0.002–0.013 ms at 160 ranks.
  - Above 640 ranks R's cap (4,096 boxes) sets the threshold, so the direct fan-out grows
    beyond the fits. The tree's bound of F copies per box does not depend on the threshold.

## 3. Questions for sign-off

**Signed off on 2026-10-14 by the user: every recommendation (1 (a), 2 (a), 3 (a), 4 (a),
5 (a)).** Step 3 builds the rule of §1.2 with F = 8, the tree only.

Recommendations first.

1. **The rule.**
   - (a) **Recommended:** §1.2 as proposed. F = 8; every box with more than F readers;
     heads by priority, shared by an owner's boxes of a level; the first hops split across
     nodes by readers (a node receives a box from as many heads as its share of F, at most
     F − 1).
   - (b) The same with exactly one head per node, so each node receives each box once:
     fewer messages to other nodes (39–65 against 44–81 at 640), but more in all (137–325
     against 116–247; 170 on the Plummer sphere at f32, one above T8's 169).
   - (c) The brief's literal rule, one tree per box with the first reader of each node:
     103–328 at 512 and 133–327 at 640.
2. **F and refinements.**
   - (a) **Recommended:** F = 8, no merging.
   - (b) F = 16: one stage less, more messages on 10 of 12 rows at 512–640.
   - (c) F = 8 with a forwarder's leaf copies merged into its later message to the same rank:
     2–8 % fewer messages, those copies a hop later.
3. **Level aggregation (step 2's alternative).**
   - (a) **Recommended:** build only the tree here, and record aggregation with the tree as
     the remaining lever for f32 p = 3 (§2.3) for T9 and the Phase 6 briefs. T9 can then
     weigh it against what the overlapped path hides at 160 ranks.
   - (b) Build both, the tree and one message per destination and stage for all levels: the
     largest cut (61–132 at 512–640), at the cost of the overlapped path's per-level waits.
4. **The f32 p = 3 trigger.** It stays fired on the cube and the clusters after the tree.
   (a) **Recommended:** record it as fired, with the floor of §2.3 (no per-level scheme
   reaches 140 messages), and leave the decision to T9's measurements of the overlapped
   path. (b) Count aggregation as part of S8 (question 3 (b)).
5. **The API of §1.8.**
   - (a) **Recommended:** `MultipoleRouting` on the evaluator and `FmmBuilder`, the tree the
     default; `MultipoleExchange::communicator` replaced by the accessors.
   - (b) Keep `communicator` as an `Option`.

## 4. Method, runs and node-hours

**The probe** (not committed, as T8's; reverted from the tree after the runs, kept as a patch):
- `Fmm::probe_multipole_sends` (`fmm-exec/src/fmm.rs`) lists every (level, top-tree flag,
  destination, key) the multipole exchange sends, from each level's rlst communicator.
- A hook in `scaling::run`, behind `FANOUT_PROBE=<file>`, gathers them to rank 0 with every
  rank's node (`Placement::node`, from `split_shared`) and appends them to one binary file.
- **Validated:** the busiest sender's messages and the largest fan-out from the dump equal
  the harness's `multipole_messages_sent_max` and `multipole_fan_out_max` at 8 ranks (M3
  Max: 16 and 3), and every count of §2.1's direct column equals T8 §5.1's.
- **The simulation** (Python, outside the repository) groups the dump by (owner, level,
  key) into reader sets and applies each rule. It counts per rank the distinct (destination,
  level, stage) messages, or (destination, stage) when aggregated, the copies of each box
  per sender, the depth, the forwarded copies and the messages to other nodes.

**Runs** (release; every BLAS thread variable at 1; `--evals 0`, a build and one warm-up
evaluation; input a random share):
- locust: 64, 72, 128, 256 and 512 ranks (`--map-by :OVERSUBSCRIBE --bind-to none` above
  72), Open MPI 5.0.10; 30 launches, all exited 0.
- Kathleen job 604096: 40, 80 and 160 ranks at 40 per node (`--map-by ppr:20:socket:PE=1
  --bind-to core`), and 320 and 640 oversubscribed (`--oversubscribe --map-by
  ppr:<P/4>:node --bind-to none -x PSM2_SHAREDCONTEXTS=1`); Open MPI 4.1.6 over PSM2,
  `PSM2_ERRCHK_TIMEOUT=10:40:2`; 30 launches, all exited 0.

| Job | Nodes, QoS | What | Waited | Ran | Node-hours |
| --- | --- | --- | ---: | ---: | ---: |
| 604095 | 1, `singlenode`, node-c11b-003 | build of the probe tree (`scaling`, release) | 0:00:01 | 3:26 | 0.06 |
| 604096 | 4, `small`, node-c11b-[003,039,046,048] | the probe at 40, 80, 160, 320 and 640 ranks, both precisions, three workloads | 0:47:26 | 7:09 | 0.48 |
| | | | | **total** | **0.53** |

Raw dumps stay in locust's `/data/ucahtbe/logs/p5s-t8s/` and Kathleen's
`/scratch/scratch/ucahtbe/fmm/logs/p5s-t8s/` (about 1 GB on each) until the report is
signed off.

## 5. Step 3: the tree, built

Built after the sign-off of §3. The forwarding tree is the default of every multipole
exchange; the direct exchange stays as `MultipoleRouting::Direct`, the oracle.

- **`nd-fmm-plan`:**
  - `exchange_tree.rs`, a private submodule of `exchange`: the rule (`split`,
    `level_trees`; §1.2), the build (`TreeExchange::new`: a communicator duplicate and two
    `sparse_all_to_allv`, §1.6), the routes per level and stage, and both paths;
  - `MultipoleRouting::{Direct, Tree { fan_out }}`, `TREE_FAN_OUT = 8`;
  - `Evaluator::with_routing(…, routing, node)` (`Evaluator::new` unchanged: the tree, the
    rank as its node);
  - `MultipoleExchange::{routing, destinations, sources, depth, fan_out, forwarded}` in
    place of `communicator`;
  - `ExchangeError::FanOut` (a fan-out below 2, agreed with the sizes).
- **`nd-fmm-exec`:** `FmmBuilder::multipole_routing`, `Fmm::multipole_routing`; the node of
  each rank from `build`'s one `split_shared` (`Group::translate_rank`, no
  communication); `ExchangeTraffic::{multipole_depth, multipole_forwarded}`, and
  `multipole_fan_out` over every stage.
- **`nd-fmm-validate`:** `scaling --routing direct|tree|treeF`; the summary's `routing`,
  `multipole_depth_max`, `multipole_forwarded_max`; two report lines.
- **Two changes during the work:**
  - **A defect the unit tests found:** `split` balanced runs of whole nodes against the
    current part only, so every part after the first came out too large and the trees too
    deep (8–9 stages for 110–171 readers on as many nodes at F = 3). It now balances the
    cumulative count; the random test's bound (⌈log_F nodes⌉ + ⌈log_F readers per node⌉ + 2)
    holds on 200 cases. The simulation of §2 did not have the defect.
  - **Eager waits:** the first build waited for a level's messages stage by stage. After
    step 5's first timing (job 604104) the waits post each forward as soon as the messages
    it resends have arrived (polling the level's receives). Every route, every message and
    every bit is the same; on locust it changed no time (§7.3).

No event was added: forwards read the staging buffer every received multipole passes
through. The device mirror, its transfer formula and the shadow operator are unchanged
(the shadow builds its evaluator with the `Fmm`'s routing). `send_boxes` loses the
copies that the forwarders now send.

## 6. Step 4: tests and checks

**New tests:**
- `exchange_tree_tests.rs` (5, serial): `split` on one node and by nodes; `level_trees` on
  200 random reader sets (every reader once, at most F copies per sender, a forward one
  stage after its receipt, the depth bound, deterministic); shared first hops; forwards
  within a node.
- `mpi_regressions`: on every scenario the tree at its default and at its narrowest (fan-out
  2, made-up nodes of two ranks), blocking and overlapped, slot by slot against `Direct`
  (the same received boxes, receive buffers and stores, bit for bit), no box sent to more
  ranks than the fan-out, messages and values sent equal to those received; the `Mixer`
  stores under `Direct` and the narrow tree equal the default's; the recorder runs under the
  default and the narrow tree, blocking and overlapped. On 4–8 ranks the narrow tree
  forwards (16 chunks at 4, 413 at 8).
- `mpi_exec`, **multipole routing**: a Plummer sphere (N = 3,000, p = 5), the narrow tree, the
  default tree and `Direct`, each against its one-rank reference and the others bit for bit,
  blocking and overlapped, the shadow check on the narrow tree in both paths, the device
  repeats under every routing with the `cpu` feature. On 4 and 8 ranks the narrow tree
  forwards 254 and 1,772 chunks at depth 2 and 3.

**Checks run, all passing (the final code unless marked):**

| Check | Where | Result |
| --- | --- | --- |
| `cargo fmt --check`, `cargo clippy -- -D warnings` (default, `--examples`, `--workspace --all-targets`), `cargo doc --no-deps` | M3 Max | clean |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | M3 Max | passed |
| `mpi_regressions` at 1, 2, 4, 8 | M3 Max | passed |
| `mpi_exec`, `mpi_threading`, `invariance` at 1, 2, 4, 8 | M3 Max | passed |
| `.github/scripts/run-mpi-tests.sh` at 2 and 4 (22 launches) | M3 Max | passed |
| `overlap_stress` at 8, `multi_rank` at 1, 2, 4, 8 (release, ignored) | M3 Max | passed |
| `cargo clippy -p nd-fmm-exec --all-targets --features cpu,metal`; `cargo test -p nd-fmm-exec --features cpu --release`; `mpi_exec` and `device_ranks` (`cpu`) at 2 and 4 | M3 Max | passed |
| `device_ranks` on Metal at 2 ranks (outside the sandbox) | M3 Max | passed |
| CUDA: `cargo clippy -p nd-fmm-exec --all-targets --features cpu,cuda`; `cargo test -p nd-fmm-exec --features cuda --release`, and `-- --ignored`; `device_ranks` at 2 and 4 | locust | passed (twice: before and after the eager waits) |
| invariance 512 against 64 (cube and Plummer f64 p = 8, cube f32 p = 3, N = 10⁶, overlap both) | locust | 0 values differ (twice) |
| T2's correctness job at 2 nodes (MPI list at 2, 4, 8, 80; invariance at 80 against 40, 20, 10; smoke) | Kathleen 604115 | passed |
| invariance at 160 against 80, 40, 20 and at 640 against 80 oversubscribed | Kathleen 604103 (before the eager waits) | 0 values differ |
| the same on the final code | Kathleen 604114 | 0 values differ |

The first 2-node correctness job (604105) failed one launch: `mpi_exec` at 80 ranks, the
new scenario's assertion that the narrow tree forwards. At 80 ranks the scenario's 3,000
points leave too few per rank for a box below R to have three readers, so nothing is
forwarded. The assertion now holds for 4–8 ranks only, where it was meant; the rerun
(604115) passed.

## 7. Step 5: measured

2.5 × 10⁴ points per rank, the second cut on, the default tree (F = 8). Counted runs: locust
64–512 (`--map-by :OVERSUBSCRIBE --bind-to none` above 72), Kathleen 80–640 (320 and 640
oversubscribed, 80 and 160 one rank per core); one evaluation. Timed runs: locust 72
ranks, one per core, three evaluations; Kathleen 40, 80 and 160, one per core, five
evaluations; both paths from one build.

### 7.1 The busiest sender, measured

Messages per evaluation, T8's direct count → the tree's (measured, `scaling`'s
`multipole_messages_sent_max`):

| precision | workload | 64 (l) | 72 (l) | 80 (K) | 128 (l) | 160 (K) | 256 (l) | 320 (K) | 512 (l) | 640 (K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 p = 8 | cube | 59 → 58 | 51 → 51 | 56 → 56 | 64 → 64 | 73 → 74 | 79 → 78 | 84 → 87 | 117 → 98 | 142 → 120 |
| f64 p = 8 | Plummer | 80 → 86 | 95 → 92 | 106 → 103 | 112 → 118 | 106 → 94 | 139 → 118 | 116 → 116 | 155 → 127 | 184 → 147 |
| f64 p = 8 | clusters | 94 → 94 | 96 → 96 | 71 → 71 | 150 → 124 | 141 → 127 | 261 → 166 | 212 → 164 | 486 → 241 | 340 → 244 |
| f32 p = 3 | cube | 60 → 57 | 51 → 52 | 57 → 59 | 66 → 67 | 73 → 75 | 84 → 81 | 84 → 87 | 113 → 97 | 139 → 119 |
| f32 p = 3 | Plummer | 80 → 80 | 90 → 87 | 89 → 89 | 106 → 109 | 106 → 101 | 134 → 122 | 130 → 115 | 157 → 143 | 169 → 145 |
| f32 p = 3 | clusters | 90 → 90 | 94 → 94 | 97 → 97 | 154 → 128 | 137 → 122 | 262 → 165 | 236 → 166 | 464 → 228 | 458 → 244 |

- **At 512 and 640 ranks every workload is below T8's measurement**, by 9–51 %: the cube
  117 → 98 and 142 → 120, the Plummer sphere 155 → 127 and 184 → 147, the clusters 486 → 241
  and 340 → 244 (f64 p = 8); 113 → 97, 139 → 119, 157 → 143, 169 → 145, 464 → 228, 458 → 244
  (f32 p = 3). The simulation of §2 gave 97–247; the measurements are within 3 % of it.
- **No box is sent by one rank to more than 8 ranks** at any rank count (`multipole_fan_out_max`
  = 8 from 64 ranks on; T8: 27–63 at 512–640). Depth 2; 3 at 512–640 on the clusters, at
  640 on every workload and at 320 on the Plummer sphere in f32.
- **At 640 ranks**: 713–2,046 chunks forwarded by the busiest forwarder; messages to other
  nodes 44, 70, 61 (f64) and 45, 67, 82 (f32) against T8's 62, 121, 181 and 64, 114, 208;
  ranks received from 54, 58, 78 against T8's 59, 62, 108 (f64).
- **Below 128 ranks** the tree changes little (R holds most boxes with many readers there):
  −5 % to +8 %, the largest increase the Plummer sphere at 64 ranks (80 → 86). From 128
  ranks the clusters gain most (−17 % at 128, −37 % at 256).

### 7.2 The model at 10³ and 10⁴ ranks (*model*)

Fits over the nine measured counts from 64 to 640 ranks, 4 µs per message, T8 §5.2's
thresholds (1 % of an evaluation at 2.5 × 10⁴ points per rank):

| precision | workload | T8, direct, at 10⁴: power / linear | tree: power fit | at 10³ | at 10⁴, power / linear | 1 % | tree fires at (power; linear) |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| f64 p = 8 | cube | 1.43 / 5.98 ms | 13.8 P^0.32 | 0.51 / 0.62 ms | **1.07 / 4.45 ms** | 5.49 ms | 1,582,287; 12,441 |
| f64 p = 8 | Plummer | 1.43 / 6.21 ms | 43.6 P^0.18 | 0.60 / 0.70 ms | **0.90 / 3.70 ms** | 12.12 ms | beyond 10⁷; 35,218 |
| f64 p = 8 | clusters | 11.89 / 23.79 ms | 11.5 P^0.48 | 1.23 / 1.47 ms | **3.68 / 11.96 ms** | 14.25 ms | 171,971; 11,962 |
| f32 p = 3 | cube | 1.36 / 5.67 ms | 14.8 P^0.31 | 0.51 / 0.61 ms | **1.04 / 4.28 ms** | 0.56 ms | 1,368; 866 |
| f32 p = 3 | Plummer | 1.55 / 6.21 ms | 30.7 P^0.24 | 0.65 / 0.76 ms | **1.14 / 4.63 ms** | 1.47 ms | 28,564; 2,646 |
| f32 p = 3 | clusters | 14.56 / 28.25 ms | 14.6 P^0.43 | 1.17 / 1.41 ms | **3.17 / 11.21 ms** | 1.85 ms | 2,885; 1,402 |

The measured fits agree with step 2's simulated ones (§2.3) to within 0.3 ms at 10⁴:
- **f64 p = 8: below the threshold on every workload under both fits.** The clusters fall
  from T8's 11.9 / 23.8 ms to 3.7 / 12.0 ms against 14.25, and grow like P^0.48 instead of
  P^0.71.
- **f32 p = 3: the trigger still fires** on the cube (from 870–1,370 ranks) and the
  clusters (1,400–2,900), and on the Plummer sphere under the linear fit (2,600). As §2.3
  found, the boxes with at most 8 readers alone reach the cube's threshold under the
  linear fit; signed off as fired, for T9 (§3, question 4).

### 7.3 Timed: the exposed wait and the extra hop

**locust, 72 ranks** (one node, one rank per core; idle but for these launches). Three
routings interleaved, each run after the other with the same settings: `direct`, the
default `tree` (F = 8), and `tree1000` (the tree's point-to-point code with a fan-out no
box reaches, so every box goes direct: the code without the forwarding):

| precision | workload | blocking stage, direct (ms) | blocking stage, tree (ms) | blocking stage, tree1000 (ms) | exposed wait overlapped, direct / tree / tree1000 (ms) | evaluation blocking, direct / tree / tree1000 (ms) | evaluation overlapped, direct / tree / tree1000 (ms) |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| f64 p = 8 | cube | 2.08 | 2.57 | 2.50 | 0.005 / 0.007 / 0.006 | 398.4 / 398.6 / 398.5 | 397.3 / 399.4 / 398.6 |
| f64 p = 8 | Plummer | 4.59 | 5.59 | 5.51 | 0.007 / 0.008 / 0.007 | 513.1 / 518.9 / 515.4 | 512.5 / 516.0 / 514.9 |
| f64 p = 8 | clusters | 4.25 | 5.06 | 5.21 | 0.006 / 0.007 / 0.007 | 504.4 / 501.3 / 506.0 | 500.1 / 497.7 / 503.0 |
| f32 p = 3 | cube | 0.21 | 0.23 | 0.24 | 0.002 / 0.002 / 0.002 | 81.9 / 83.1 / 81.8 | 81.9 / 81.8 / 81.9 |
| f32 p = 3 | Plummer | 0.49 | 0.57 | 0.56 | 0.003 / 0.003 / 0.004 | 72.6 / 72.6 / 72.6 | 72.4 / 72.5 / 72.4 |
| f32 p = 3 | clusters | 0.42 | 0.41 | 0.42 | 0.003 / 0.004 / 0.004 | 67.7 / 67.7 / 67.7 | 67.6 / 68.4 / 67.4 |

**Kathleen, 160 ranks** (4 nodes, 40 per node, one per core; job 604113, the final code):

| precision | workload | blocking stage, direct (ms) | blocking stage, tree (ms) | blocking stage, tree1000 (ms) | exposed wait overlapped, direct / tree / tree1000 (ms) | evaluation blocking, direct / tree / tree1000 (ms) | evaluation overlapped, direct / tree / tree1000 (ms) |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| f64 p = 8 | cube | 4.12 | 5.89 | 5.35 | 0.003 / 0.007 / 0.004 | 1041.3 / 1047.1 / 1038.7 | 1041.2 / 1038.6 / 1034.5 |
| f64 p = 8 | Plummer | 6.43 | 7.74 | 10.44 | 0.005 / 0.197 / 0.006 | 1194.0 / 1193.4 / 1198.9 | 1196.1 / 1194.7 / 1194.4 |
| f64 p = 8 | clusters | 7.44 | 8.69 | 8.44 | 0.014 / 0.014 / 0.012 | 1313.5 / 1316.0 / 1312.9 | 1309.5 / 1308.4 / 1312.2 |
| f32 p = 3 | cube | 0.56 | 0.67 | 0.61 | 0.002 / 0.002 / 0.002 | 66.3 / 64.8 / 62.1 | 61.8 / 65.2 / 62.5 |
| f32 p = 3 | Plummer | 0.87 | 1.06 | 1.00 | 0.003 / 0.004 / 0.004 | 145.7 / 150.5 / 148.5 | 142.5 / 144.2 / 149.0 |
| f32 p = 3 | clusters | 1.09 | 1.14 | 1.13 | 0.003 / 0.213 / 0.003 | 182.0 / 182.4 / 182.9 | 184.2 / 182.3 / 183.0 |

- **The overlapped path hides the multipole exchange under every routing**: the exposed wait
  is 0.002–0.008 ms at 72 locust ranks and 0.002–0.21 ms at 160 Kathleen ranks, at most
  0.12 % of the evaluation (0.21 ms of 182 ms, the clusters at f32 p = 3). The measured clause of S8's trigger (above 1 % at 160 ranks)
  does not fire, as in T8 (0.002–0.013 ms).
- **The blocking path's multipole stage is slower under the tree**: at 72 locust ranks by
  −0.01 to +1.0 ms (−2 % to +24 %), at 160 Kathleen ranks by 0.05–1.8 ms (+5 % to +43 %). Most of it is not the
  extra hop:
  - `tree1000`, which forwards nothing, costs as much as the tree (cube f64 p = 8: 2.08 ms
    direct, 2.57 tree, 2.50 tree1000; Plummer 4.59, 5.59, 5.51), so the cost is the
    point-to-point exchange itself against the direct exchange's rlst neighbour all-to-all.
    The same at 160 Kathleen ranks: tree against tree1000 −2.7 to +0.5 ms, tree1000 against
    direct +0.04 to +4.0 ms.
  - It grows with the bytes: +19–24 % at f64 p = 8 (81 values per multipole), −2 % to
    +15 % at f32 p = 3 (16). The likely cause, an inference not measured, is the copy of every received
    chunk from the staging buffer into the receive buffer, which the neighbour all-to-all
    avoids by receiving in place.
  - The forwarding itself is within the noise: tree against tree1000 −0.15 to +0.08 ms
    (step 1's model: at most 21 µs). The eager waits of §5 changed none of these times
    (locust, cube f64 p = 8, tree: 2.36 ms before, 2.47 and 2.57 ms after, within the
    run-to-run spread).
- **The evaluation** moves by at most ±0.6 % at f64 p = 8 and −4 % to +5 % at f32 p = 3 between
  the routings, within the run-to-run spread of
  these runs.

The extra hop is therefore cheap; the blocking path's copy is a cost of the tree's
point-to-point exchange, at most 0.2 % of an evaluation at 160 ranks (1.8 ms of 1,047). Receiving directly into the
receive buffer where a message's slots are contiguous (most stage-0 messages) would remove
most of it; not built here (§9, open).

## 8. Kathleen jobs and node-hours

| Job | Nodes, QoS | What | Waited | Ran | Node-hours |
| --- | --- | --- | ---: | ---: | ---: |
| 604095 | 1, `singlenode`, node-c11b-003 | step 2: build of the probe tree | 0:00:01 | 3:26 | 0.06 |
| 604096 | 4, `small`, node-c11b-[003,039,046,048] | step 2: the probe at 40–640 ranks | 0:47:26 | 7:09 | 0.48 |
| 604103 | 4, `small`, node-c11a-[054,070-072] | step 4: invariance at 160 (k = 2, 4, 8) and 640 (k = 8), before the eager waits | 0:06:51 | 2:07 | 0.14 |
| 604104 | 4, `small`, node-c11a-[054,070-072] | step 5: timed at 40, 80, 160 (both routings), counted at 320, 640; before the eager waits | 0:08:58 | 12:31 | 0.83 |
| 604105 | 2, `test`, node-c11b-[039,044] | step 4: T2's correctness job; one launch failed (§6) | 0:06:50 | 23:36 | 0.79 |
| 604115 | 2, `test`, node-c11a-[054,070] | step 4: T2's correctness job, final code | 0:06:41 | 22:54 | 0.76 |
| 604114 | 4, `small`, node-c11a-[006,030,033,040] | step 4: invariance at 160 and 640, final code | 2:32:23 | 0:59 | 0.07 |
| 604113 | 4, `small`, node-c11a-[006,030,033,040] | step 5: timed at 40, 80, 160, direct, tree and tree1000, final code | 2:23:21 | 9:02 | 0.60 |
| | | | | **total** | **3.73** |

Within the task's budget of 20 node-hours. Raw outputs stay in Kathleen's
`/scratch/scratch/ucahtbe/fmm/logs/p5s-t8s*` and locust's `/data/ucahtbe/logs/p5s-t8s*`.

## 9. Open, and the next steps (folded into T9)

An analysis of these measurements (an advisory pass over the step-5 logs, the probe dumps
and the code, 2026-10-14) ranked what to improve in the MPI communication next. The
user folded its first recommendations into T9 (docs/phase5s/T9-scaling-at-scale.md, "The
exposed collectives"). In brief:

- **At 160 Kathleen ranks the communication is small** (measured, the fastest rank, the
  blocking path): 8–16 ms of a 1,045–1,320 ms evaluation at f64 p = 8 (0.8–1.2 %), 2.9–3.4 ms
  of 70–208 ms at f32 p = 3 (1.7–4.1 %). Overlapped, the source and multipole exchanges
  expose 0.00–0.02 ms. What stays exposed: the replicated top's all-gather-v (1.7–7.0 ms
  at f64 p = 8, 0.3–0.7 ms at f32 p = 3), the redistribution of a randomly placed input
  (1.9–3.2 ms), and the load imbalance (8–42 % of wall time), larger than every
  communication term. 75–90 % of a rank's multipole bytes stay on its node (counted).
- **The replicated top's all-gather is the one per-evaluation term that grows with P.** If
  Open MPI 4.1.6 picks a P/2-step algorithm for R's size (inferred from its decision rule,
  not measured), it takes 10–17 ms at 10⁴ ranks (*model*, 5–8 times `scale-out.md`
  §10.4's), above 1 % of an evaluation at f32 p = 3 from a few hundred ranks. T9 measures
  it first and, if the sign-off agrees, builds a two-level gather (node leaders, then a
  broadcast on each node; *model* 1.5–2 ms at 10⁴).
- **Overlap on by default**, after T9's sweeps confirm it: it hides the exchanges (0.4–1.5 %
  of an evaluation at 160 ranks) and most of S8's modelled per-message cost.
- **The tree's blocking path** (+0.3–0.6 ms at 160 ranks, §7.3): posting every level at once
  and scattering from the staging buffer, only if the blocking path keeps a default.
- **Not recommended**, checked against the data: level aggregation as specified (§2.2; it
  would expose the overlapped path's per-level waits), persistent requests,
  neighbourhood collectives and shared-memory windows (no safe API in rsmpi 0.8.2, not
  called for), merging forwarders' leaf copies, PSM2 eager-limit tuning (the messages are
  below its rendezvous threshold).

Open beyond that:
- Every timing above 160 ranks (the model of §7.2); contention on a node's link, which the
  4 µs per message does not model; R's cap setting the threshold above 640 ranks, where the
  direct fan-out would grow but the tree's stays at 8.
- The f32 p = 3 trigger (§7.2), for T9 with the overlapped path's measurements; level
  aggregation (§2.2) as the lever that reaches below the floor of boxes with at most 8
  readers.
- The blocking path's copy (§7.3): receiving in place where a message's slots are contiguous.

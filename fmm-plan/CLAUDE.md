# nd-fmm-plan

The root `CLAUDE.md` applies as well; it holds every workspace-wide rule. This file
adds only what is specific to this crate.

## Project

`nd-fmm-plan` (Cargo package `nd-fmm-plan`, Rust crate `nd_fmm_plan`, Rust 2024) plans the
**topology and data flow** of a fast multipole method on top of an
`nd_octree::Octree`: a Morton-ordered integer index of the boxes, the interaction lists
as index arrays, the data stores and ghost exchanges, and the distributed evaluation
order with a level-batched operator interface. It does **not** own point coordinates,
kernel evaluations, or translation operators (M2M / M2L / L2L / near-field arithmetic)
— analytical and kernel-independent FMMs are meant to share this layer. Keep
kernel-specific arithmetic out. The design is `docs/design/fmm-plan-redesign.md`
(signed off; the decisions are recorded in its §12).

Real FMM operators belong outside this crate (the Laplace operator in `nd-fmm-exec`)
and plug in through the `FmmOperator` trait, or `PairOperator` with the `PerPair`
adapter. The crate's own tests use test operators whose values do not matter: they
record and check the calls, the batches and the slices they are handed. The values of
the distributed passes are checked by the Laplace FMM in `nd-fmm-exec`.

- Library only, no binaries.
- Version `0.1.0-dev`; the public API is unstable. `redistribute::Redistribution`
  moves points to the ranks that own their leaves and results back (Phase 5 T5,
  `docs/design/distributed-fmm.md` §4); `nd-fmm-exec`'s `Fmm` uses it from T6.

## Code map

| Path | Contents |
| --- | --- |
| `src/lib.rs` | Crate doc (module overview, the compute graph, future extensions) and the `pub mod` lines. |
| `src/index.rs` | `BoxIndex`, `LeafNumbering`: Morton-ordered `u32` box indices per level for every held key (local, `Global`, ghost), and the leaf numbering (local leaves by (level, key), then the ghost leaves named by U or X, by key). |
| `src/interaction_manager.rs` | `V_LIST_DIRECTIONS` (the 316 V-list offsets, CONVENTIONS §3.12) and the private per-key list rule `key_lists`, with its helpers. See the section below. |
| `src/lists.rs` | `Csr`, `GroupedCsr` as `VList`/`Children`/`Parents`, `LevelLists` (the views of one level), `offset_index`, `NOFFSETS`, `NOCTANTS`. |
| `src/near.rs` | The near rule (Phase 6 T12, C6.9; docs/design/optimisation.md §8a): `LeafPair` (a W or X pair between two leaves with the fine leaf's count), `LeafCounts` (source counts of the local leaves and the W rows' leaves, target counts of the local leaves, by key; `exchange`, collective: the W rows' ghost leaves' counts from their owners as the chunk sizes of a ghost communicator, `exchange::ghost_source_counts`), `NearRule` (an operator-supplied predicate over `LeafPair` and the counts), `NearPairs` (the candidates and conversions a plan reports). `Plan::with_near_rule` and `Plan::from_key_types_with_near_rule` apply it in `first_pass`: a converted W entry (a leaf) or X entry of a local leaf joins its near row (in (level, key) order), a converted ghost W entry is numbered as a ghost leaf; a missing count is `PlanError::MissingCount`. Without a rule `Plan::new` is unchanged bit for bit. |
| `src/plan.rs` | `Plan::new` (collective), `Plan::from_key_types` (local, from a key map and a `PartitionMap`), `PlanError`. Since Phase 5S T5 the plan holds H, the octree's locally essential keys (`all_keys`; module docs, "Held keys"; T4 held H ∪ T until the coarse gather went), and from the partition map the owner of every held top-tree key ("Owners": the rank of its first descendant block; `owner`, `computes`, `is_top`, `top_boxes`, `rank`, `deepest_block_level`, `reduction_levels`). A `Global` box has its global M2M row on its owner only. Near and X rows by the entry leaf's (level, key) (P1), and for a `Global` box that is not an ancestor of the rank's own coarse blocks (P2; only in a hand-built map since T5) no V, X or L2L rows and no list resolution at all; distributed-fmm §5.1, §3.6. |
| `src/store.rs` | `LevelBuffers`, `LeafStore` and their slice types: one buffer per level kind, CSR leaf data with variable counts. |
| `src/exchange.rs` | `SourceExchange`, `ReductionExchange`, `TopExchange`, `MultipoleExchange`, `ExchangeError`: variable-size source exchange into the ghost tail; since Phase 5S T5 (scale-out.md §4) the upward reduction (per reduction level, the children of the owned `Global` boxes that other ranks compute, from them), the replicated top R (the top-tree boxes with at least max(2, ⌈P/`TOP_READER_FRACTION`⌉) readers, at most `TOP_MAX_BOXES`, chosen at build from one rlst build of the readers, one all-reduce and an all-gather of the keys; one all-gather-v per evaluation), and the per-level multipole exchange of the V- and W-row boxes other ranks compute, without what the reduction and R deliver (so every multipole a rank reads arrives once). The coarse gather is gone. Each names the slots it reads and writes (`send_leaves`/`ghost_leaves`, `send_boxes`/`receive_boxes`, `sent_keys`/`received_keys` with `slot`). Next to the blocking calls, the non-blocking `SourceExchange::forward_overlapped`, `ReductionExchange::forward_all_overlapped` (receives posted ahead, each level's sends by `ReductionInFlight::send`) and `MultipoleExchange::forward_all_overlapped` (Phase 5 T9): scoped rsmpi point-to-point on each exchange's graph communicator, the caller's work run in between with an in-flight handle (`SourcesInFlight`, `ReductionInFlight`, `MultipolesInFlight`: `test`, `wait(level)`, `exchange`), `ExchangeTimes`; `Traffic` (messages and values per exchange). |
| `src/exchange_tree.rs` | The forwarding tree of the multipole exchange (Phase 5S T8s, S8; `fmm-validate/results/phase5s-t8s.md` §1, signed off on 2026-10-14), a private submodule of `exchange` included by `#[path]`: `split` and `level_trees` (the rule: per owner and level, the readers of its boxes with more than F readers split into at most F parts by node and rank, heads by the number of the owner's boxes they read, forwards within a part), `TreeExchange::new` (a communicator duplicate, two `sparse_all_to_allv`: requests with the requester's node, instructions), the routes per level and stage, the blocking `forward` (polling the level's receives, each forward posted once its inputs are in) and the overlapped `Parts`/`TreeFlight` (forwards posted from `test` and the waits). `MultipoleExchange` holds either it (`MultipoleRouting::Tree`, the default, `TREE_FAN_OUT` = 8) or the per-level rlst communicators (`MultipoleRouting::Direct`, the oracle); its lists, receive buffers and the stores' bits are the same under both. |
| `src/operator.rs` | `FmmSizes`, the level-batched `FmmOperator` with its batch types `P2m` … `P2p` and `UpwardPass`, the per-pair `PairOperator` and its `PerPair` adapter; `HostData` and `FmmOperator::host_data` (default: nothing), the data-movement hook (Phase 5 T7, below). |
| `src/evaluator.rs` | `Evaluator`, `EvaluatorError`: the pass order on the plan, stores and exchanges, public stages, debug-checked stage order; the host-data events around `reset` and the exchanges (the module docs list every host read and write and its event); read-only access to the four exchanges (`source_exchange`, `reduction_exchange`, `top_exchange`, `multipole_exchange`; Phase 5 T8, 5S T5), whose index lists an operator with its own stores sizes its buffers from at build. `upward_global` is the owned top tree since Phase 5S T5 (module docs, "The owned top tree"): per reduction level, deepest first, the reduction, then the global M2M that reads it; then the all-gather of R. The overlapped path (Phase 5 T9): `evaluate_overlapped`, `exchange_sources_and_upward_local`, `far_field`, `overlap_times` (`OverlapTimes`: the reduction, the multipoles, the top's all-gather; `MAX_LEVELS`); the passes are free functions over the stores with a `between` callback (the `test` calls). |
| `src/redistribute.rs` | `Redistribution`, `RedistributionError`: items routed by finest key to the owners of their leaves (`new`: rlst's `SparseRoutes` from the item counts (a communicator duplicate, a sparse exchange of the counts, an all-reduce), an all-reduce of errors and `max_per_item`, key and position along the routes; Phase 5S T8, S5), payloads `forward` into leaf order and `backward` into the caller's order (point to point along the routes, no collective, `_into` variants, in-place cycle permutations). Within a leaf, items are ordered by (origin rank, origin position). |
| `src/near_tests.rs` | 5 serial unit tests of the near rule on the key maps of `plan_tests.rs` (the trees of `interaction_manager_tests.rs`, all local, with a `Global` root and with a ghost half): the converted entries exactly those the predicate names, each candidate offered once with its count, every other view unchanged, the ghost leaves those of U and X plus the converted ghost W entries; W–X duality for the pairs not converted; near and X rows in (level, key) order with every pair converted; a rule that converts nothing gives `Plan::new`'s views; a missing count is an error. No MPI. |
| `src/interaction_manager_tests.rs` | 7 serial unit tests of the per-key rule on synthetic key maps (hand counts, W/X on a refined octant, brute-force oracle, invariants, adjacency, ghost classification, V-list offsets), through the test helper `ListMaps`, included via `#[path]`. No MPI. |
| `src/plan_tests.rs` | 15 serial unit tests of the index and the index-form lists against `ListMaps` on the trees of `interaction_manager_tests.rs` (partition maps built with `PartitionMap::new`: `one_rank_map`, `map_of`), among them the row order of P1, the pruned `Global` rows of P2 with the global M2M rows on the owner only (Phase 5S T5), and (Phase 5S T4) the plan over a hand-built H ∪ T with blocks on levels 1–4, whose rows equal those over the whole map key for key though a pruned `Global` box names keys it does not hold. No MPI. |
| `src/exchange_tree_tests.rs` | 5 serial unit tests of the forwarding tree's rule: `split` on one node and by nodes, `level_trees` on 200 random reader sets (every reader once, at most F copies per sender, a forward one stage after its receipt, the depth within ⌈log_F nodes⌉ + ⌈log_F readers per node⌉ + 2, deterministic), shared first hops, forwards within a node. No MPI. |
| `src/store_tests.rs`, `src/exchange_tests.rs` | 9 serial unit tests of the store layouts and 6 of the ghost bucketing and per-key chunk sizes (on the trees of `plan_tests.rs`). No MPI. |
| `src/redistribute_tests.rs` | 4 serial unit tests of the local parts of `Redistribution`: bucketing, in-place cycle permutations against a naive permutation, `max_per_item`, and the grouping by leaf, the order within a leaf, the counts, the origins and the round trip of `Sender` and `Receiver` on hand-made and random routing data, with the all-to-all-v simulated. No MPI. |
| `src/operator_tests.rs`, `src/evaluator_tests.rs` | 3 serial unit tests of the `PerPair` adapter on hand-made batches, and 4 of the evaluator passes: a marker operator checks that every batch hands the views and slices of its level and leaves (variable counts, on the ghost-free trees of `plan_tests.rs`), reset and repeat, call order, validation errors. No MPI. |
| `tests/mpi_regressions.rs` | One `#[test]` that owns MPI init and runs 12 named scenarios sequentially, then the cube and the Plummer sphere (200 points per rank, by index) for the one-rank comparison. Each scenario builds an `Octree` from the union of a source and a target point set (with the ghost-children layer) and computes brute-force U/V/W/X lists of every non-ghost key over the gathered global tree (`oracle_lists`). It then builds a `Plan` and checks the numbering, the leaves, the index-form lists against the oracle, the view invariants and the global M2M rows on the owners only (`computes`, from the partition map), and (`check_held_keys`, Phase 5S T4, T5) that the held keys are exactly H by brute force over the gathered tree and blocks, each with the octree's kind (a top-tree key above other ranks' blocks only a ghost of its first block's rank); no `Global` box loses a row to P2 on several ranks. The exchanges: sources with counts of one, `hash(key) % 5` and the real source points per leaf; the reduction (`check_reduction`: levels 1 to the deepest block level, receives exactly the children of the owned `Global` boxes that other ranks compute, sends only what the rank computes, blocking deepest first and non-blocking bit for bit); the replicated top (`check_top`: R the same on every rank and exactly the top-tree keys with at least max(2, ⌈P/16⌉) readers counted by brute force over every rank's needs, in rank order by (level, key), its sent and written keys, the gathered values); the multipole exchange (with per-level sizes, exactly the V- and W-row boxes other ranks compute minus what the reduction and R deliver, and every multipole a list reads received exactly once over the three). The `Evaluator`: with `hash(key) % 5` counts a per-pair operator whose values do not matter and whose mixing does not commute (`Mixer`) must give bit-identical stores on a second evaluation, on an evaluator that owns its plan and overlapped; `check_against_one_rank` evaluates it on the plan and on the one-rank plan of the same tree (`self_comm`) and compares by key every multipole the rank computes or receives (zero elsewhere), every non-ghost local and every local leaf's output, bit for bit; a recording operator checks the call order, the groupings of every batch and that every oracle pair is issued exactly once (the global M2M pairs on the owner only); it also records the host-data events (Phase 5 T7, 5S T5: `SendReduction`/`ReceivedReduction` per reduction level, `SendMultipoles` with the top's sent keys, `ReceivedTop`) and checks, with the verdict agreed on every rank, their order among the level calls, that every rank gets all 5 + nlevels + 2 D of them, that their index lists equal those of exchanges built independently of the evaluator's and those its exchange accessors give, and what they name (boxes sent are computed by the rank, boxes received are not, the packed buffers equal the written slots). The oracle knows P2, and `check_plan` checks P1's row order. Finally it redistributes the scenario's global sources and targets (every rank's points, in rank order) from four input distributions (all on rank 0, a seeded share, the owners, the share with the last rank empty): every item arrives once, on its owner, in its leaf, in (origin rank, position) order, with the counts of the plan's leaves; `f64` × 3 and `u32` × 1 payloads that name their origin round-trip; two forwards agree bit for bit; and forward's bits equal the dense all-to-all-v of the same payloads (rlst's `all_to_allv`, ordered by leaf and origin; the oracle of the point-to-point routes, Phase 5S T8). It also checks the agreed errors (invalid keys on one rank, a plan of another tree), the panic on every rank of a `per_item` above `max_per_item`, and that a payload of the wrong length on rank 0 panics there after taking part while the other ranks complete. The multipole exchange (Phase 5S T8s) also runs as the forwarding tree at its default and at its narrowest (`narrow_tree`: fan-out 2, made-up nodes of two ranks), blocking and overlapped: the same received boxes, receive buffers and slots as `MultipoleRouting::Direct`, bit for bit, no box sent by a rank to more than the fan-out, messages and values sent equal to those received over all ranks; on 4 to 8 ranks the narrow tree forwards (asserted over the scenarios). `check_repeated_evaluation` repeats the `Mixer` stores under `Direct` and the narrow tree, and `check_batches` runs the recorder under the default and the narrow tree, blocking and overlapped. The near rule (Phase 6 T12, `check_near_rule`): on every scenario, counts per leaf from the scenario's points and a kernel-agnostic predicate (`near_predicate`) that converts some pairs of each list; the counts the rule reads equal the true ones (ghost W leaves' from their owners), the plan with the rule passes `check_plan` against the oracle with the conversions applied (`near_oracle`) and reports its `NearPairs`, the source exchange delivers every ghost leaf (the converted W entries included) with its count, the reduction, the replicated top and the multipole exchange follow the converted rows, the recorder sees every converted pair as P2P once and no M2P or P2L for it, `check_against_one_rank` with a one-rank plan of the same rule is bit for bit, and a rule that converts nothing gives `Plan::new`'s plan; over the scenarios both lists convert and keep pairs, and on several ranks some converted W entry is a ghost (printed at the end). On the graded and the dense-leaf scenarios rank 0 prints the held keys and the traffic (`--nocapture`), the cube and the Plummer sphere the `Global` multipoles formed and the reduction levels. Overlap (Phase 5 T9): the non-blocking source exchange, reduction and multipole exchange deliver bit for bit what the blocking ones do, receive buffers included; `Traffic` matches the communicator, and messages and values sent equal those received over all ranks; `evaluate_overlapped` gives the `Mixer` stores of the blocking evaluation; and the recorder runs again through `evaluate_overlapped` (`check_batches(…, true)`): the same calls, pairs, groupings, events and lists, in the overlapped order (`expected_overlapped_calls`). |
| `tests/overlap_stress.rs` | One ignored `#[test]` that owns MPI init (Phase 5 T9): the graded scenario with the last rank empty, 100 overlapped evaluations of a mixing operator, each bit for bit the blocking one, agreed on every rank. By hand at 8 ranks in release under a timeout (its module docs give the command). |
| `examples/plan_build_cost.rs` | Release-mode timing and heap use of `Plan::new`, on one rank (nothing asserted). |
| `examples/redistribution_cost.rs` | Release-mode time and traffic of `Redistribution::new`, `forward` and `backward` for N points of the cube, all on rank 0 or a random share, on any number of ranks (nothing asserted; not registered for `run-examples`). |

Read `src/plan.rs`, `src/lists.rs`, `src/evaluator.rs`, `src/redistribute.rs` and
`tests/mpi_regressions.rs` for how the API actually behaves; prose is a summary, they are the contract.

### Navigating the code

Follow the root navigation rules (LSP first; `rlst` sources in the registry cache, not
`../rlst`). `nd-octree` is the workspace sibling at `../octree/src/…`. Within this
crate's small source modules (plus their `*_tests.rs` files), reading a whole
file beats either LSP or `grep`.

## The interaction lists

`Plan::from_key_types` applies the private per-key rule `interaction_manager::key_lists`
to every non-ghost held key but the pruned `Global` boxes of P2 (since Phase 5S T5 the
held keys are H, the octree's `all_keys`, and `Plan::new` takes the owners of the
top-tree keys from its `PartitionMap` locally) and translates the lists to indices. The rule is **local — no
collectives**; `Plan::new` adds two all-reduces (the level count, and validation agreed
on every rank). The module docs of `src/interaction_manager.rs` and `src/lists.rs` are
the contract; the points that matter most when changing them:

- **Rows for every box.** A box view of level l has one row per held box of the level,
  ghosts included (with empty rows); a leaf view has one row per local leaf of the
  level. Inapplicable lists (U and W for interior boxes, V and X at levels 0 and 1)
  are empty rows. Rows never contain the box itself, except the near list (U plus the
  leaf itself, design §12 decision 1a).
- **Adjacency** means the closed cubes share at least a vertex and neither box
  is an ancestor of the other. Boxes may be at different levels.
- **It relies on the octree's guarantees**: complete, 2:1 balanced across all
  26 directions, plus the one-cell same-level ghost halo. Those let the rule decide
  everything from a neighbour's leaf/interior classification and Morton
  arithmetic without looking children up. If `nd-octree` ever relaxes balance
  or shrinks the halo, the `.expect`/`debug_assert`s in the rule fire.
- **Every listed key is a held key**, with its own `KeyType` — and, for a ghost, its
  owner rank via `KeyType::ghost_rank`. This holds only when the octree is built with
  `OctreeOptions::with_ghost_children(true)`; `Plan::new` returns
  `PlanError::MissingGhostChildren` otherwise, and checks every entry, the parent of
  every box with rows, every child of a non-ghost interior box and every own coarse
  block (the held-key check, which since Phase 5S T4 replaces "every coarse block of
  every rank held" of design §3.5, decision 6; since T5 no rank holds another rank's
  block unless H names it). `Plan::from_key_types` is what `src/plan_tests.rs` drives with hand-built maps.
- **Listed boxes are routinely ghosts** — V and W entries are children of remote
  neighbours. `exchange` moves their data; the plan provides topology only.
- **The near rule** (Phase 6 T12, `src/near.rs`) is optional and kernel-agnostic: the
  operator supplies the predicate and the counts, the plan only applies it to W entries
  that are leaves and X entries of leaves, locally, keeping the near rows' (level, key)
  order. Never put cost constants or kernel arithmetic here, and never convert an
  interior W box (its points are in leaves no list names).

## Host data (Phase 5 T7)

`docs/design/distributed-fmm.md` §6. Outside the level calls the evaluator touches its
host stores in exactly five places: `reset`'s zeroing and the four exchanges (the
`evaluator` module docs, "Host data"). Around each it calls `FmmOperator::host_data`
with a `HostData` event and the index lists the movement uses: `Reset`;
`SendSources`/`ReceivedSources`; per reduction level, deepest first,
`SendReduction`/`ReceivedReduction` (Phase 5S T5), each pair before the global M2M that
reads it; `SendMultipoles` (once, after the reduction and before the all-gather of the
replicated top, for the top's sent keys and every level's multipole sends) and
`ReceivedTop`; `ReceivedMultipoles` per level. 5 + nlevels + 2 D events per evaluation
(D the deepest block level, the same on every rank), with empty lists and D = 0 on one
rank. `ReceivedCoarse` went with the coarse gather (T5).

- **Operators that keep their own copies** of the stores (a device operator, the shadow
  of `nd-fmm-exec`'s tests) must: zero their multipoles, locals and target output at
  `Reset`; write the values of the listed leaves, blocks and boxes into the host store
  at a "send" event; copy the listed slots from the host store or the packed buffers at
  a "received" event; and learn the caller's data (points, charges) from the caller.
  Nothing else of the evaluator touches a store.
- **Keep the coverage exact.** A new read or write of a store by the evaluator or an
  exchange needs an event (or a list in an existing one), the recorder's checks in
  `tests/mpi_regressions.rs`, and the shadow check of `nd-fmm-exec`
  (`tests/shadow/`), which must stay bit for bit the plain operator on every rank count
  and must differ with the hook disabled.
- The default `host_data` does nothing, and `PerPair` and the Laplace operator keep it:
  the host path is unchanged bit for bit. Never move data, add a collective or change the
  exchange or accumulation order to fire an event.

## Overlap (Phase 5 T9)

`docs/design/distributed-fmm.md` §8, decisions 9 and 10 of docs/phase5/README.md; the
`evaluator` module docs, "Overlap", and the `exchange` module docs, "Non-blocking
exchanges", are the contract.

- **Order-preserving only.** `evaluate_overlapped` makes the calls of `evaluate` in the
  same order; only the exchanges move (the source exchange behind `upward_local`; the
  reduction's receives posted ahead and each level's sends just after the global M2M
  that forms it; the multipole exchange posted after the reduction, before the all-gather
  of the top, and each level waited for just before the downward calls of that level). The output and every store are bit for bit
  the blocking path's; `tests/mpi_regressions.rs` and `nd-fmm-exec`'s `tests/mpi_exec.rs`
  check it. Never reorder a level call or an accumulation for overlap (decision 9 offers
  no reordering).
- **The forwarding tree** (Phase 5S T8s): its forwards resend chunks from the staging
  buffer, never a store, so the events are those of the direct exchange; each forward is
  posted once its inputs have arrived, in `test` and in `wait(level)` (which polls the
  level's receives).
  Every rank waits for the levels in the same order; never block on a message before
  posting the forwards whose inputs have arrived, or ranks deadlock.
- **Mechanism.** Scoped rsmpi point-to-point (`mpi::request::scope`,
  `immediate_send`/`immediate_receive_into`, `Request::test`/`wait`), never
  `RequestCollection::test_some`/`wait_some`/`test_any` (they panic in rsmpi 0.8.2 once
  every request has completed), never raw MPI or `unsafe`. A request cannot outlive its
  scope, so an overlapped stage is one method that posts, works and waits; the receives
  land in staging buffers of the exchanges, the send buffers are moved out of the
  multipole exchange while in flight so that the work can borrow it for the events. A
  panic inside the work with requests pending aborts the process (rsmpi).
- **Progress.** A `test` after every level call, on the calling thread; no worker thread
  calls MPI. The all-gather of the replicated top stays a blocking all-gather-v, inside
  the far field. In the reduction every rank posts level l's sends before it waits for
  level l (deepest first), or ranks deadlock.
- **Events.** The same kinds with the same lists, in the same order among themselves
  (`operator` module docs, "Host data"); keep `expected_overlapped_calls` and the shadow
  check of `nd-fmm-exec` in step with any change.
- **Times.** The overlapped stages read `Instant` for `OverlapTimes` (per exchange: post
  to completion, exposed wait, `test` calls; the wait per level; the top's all-gather; the
  work parts). Reports only: nothing depends on them, and no test asserts one.

## Dependencies

- All dependencies come from the root `[workspace.dependencies]`, and the package
  fields `edition`, `license`, `homepage` and `repository` from `[workspace.package]`.
  The crate adds the features it needs: `rlst` with `mpi`, `mpi` with `complex` (the
  workspace entry has `derive`).
- `nd-octree` is the workspace path dependency on `../octree`. Keep it that way: do not
  switch to a git or crates.io source, which can diverge from the workspace copy.
- `rlst` 0.12.0 (since Helmholtz CHR.2; 0.10.2 from Phase 5S T8q, 0.10.1 from T8, 0.10.0 from T1s) and `mpi` 0.8.2 (rsmpi) come from crates.io. The crate uses rlst's
  `distributed_tools::{GhostCommunicator, GhostCommunicatorBuilder, ChunkSizes}` in
  `exchange.rs`, `distributed_tools::SparseRoutes` in `redistribute.rs`, `distributed_tools::sparse_all_to_allv` in `exchange_tree.rs`, and
  `distributed_tools::{all_to_allv, array_tools::gather_to_all}` in the tests. Tests and examples also use `rlst_dynamic_array` and `println_mpi`;
  `rand_chacha` is a dev-dependency.

## Crate checks

The root checks cover this crate. For this crate alone, add `-p nd-fmm-plan` (e.g.
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan`).

`cargo test -p nd-fmm-plan` gives 58 unit tests plus the integration test **on one rank
only**. It exercises no ghost exchange and no ghost layer, which is where the
interesting bugs are.

### Multi-rank runs

CI runs `mpi_regressions` in debug at 2 and 4 ranks on every pull request (the
`run-tests-mpi` job, root `CLAUDE.md`, "Checks"). 8 ranks and any ignored test stay by
hand, on the M3 Max and on locust (tools/gh200/README.md, "MPI at n ranks"), and so does
the run before you push.

For any change touching the plan, the lists, the exchanges, the evaluator or the
redistribution (whose
ghost-dependent and global-level paths only exist on more than one rank), build the
test binary and launch it under MPI, under an external timeout:

```sh
cargo test -p nd-fmm-plan --test mpi_regressions --no-run   # note the executable path it prints
RUST_MIN_STACK=8388608 timeout 600 mpiexec -n 2 target/debug/deps/mpi_regressions-<hash> --test-threads=1
RUST_MIN_STACK=8388608 timeout 600 mpiexec -n 4 target/debug/deps/mpi_regressions-<hash> --test-threads=1
```

Use the executable Cargo prints, not the sibling `.d` file, and note the hash
changes on rebuild.

On macOS, add the loopback flags from the root `CLAUDE.md`; with them, 2 and 4 ranks
pass:

```sh
mpiexec --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 -n 4 <exe> --test-threads=1
```

`tests/mpi_regressions.rs` is the crate's multi-rank run. No example of this crate is
registered for the weekly `run-examples` job; to have one run there at 3 ranks,
register it with `[[example]]` and `[package.metadata.example.<name>.templated-examples]`
as the root `CLAUDE.md` describes.

## Octree input and MPI discipline

Octree fine keys must be valid Morton keys at level `DEEPEST_LEVEL` (16) —
`Octree::new` validates this and panics naming the offending key, so build them
with `points_to_morton(&points, DEEPEST_LEVEL as usize, &bbox)`. Point arrays
are `[3, npoints]`, one point per **column**; `PhysicalBox` is
`[xmin, ymin, zmin, xmax, ymax, zmax]`. Build the octree with
`OctreeOptions::with_ghost_children(true)`; `Plan::new` requires it.

**MPI discipline.** The root MPI rules apply; several test scenarios exist precisely
to cover empty and uneven ranks. `tests/mpi_regressions.rs` owns the one MPI
initialisation for `tests/`; do not add a second independently initialising test
there. Globally empty octree construction is not covered by any test and should not
be assumed to work.

## Conventions

- Match the existing module layout and rustdoc style. Cite the design as
  `design §x.y` (`docs/design/fmm-plan-redesign.md`) and conventions as
  `CONVENTIONS §3.x`.
- Keep topology tests deterministic and free of translation operators. Serial
  helper tests go next to the helpers (`src/*_tests.rs`, included with `#[path]`);
  anything needing MPI goes in `tests/` or `examples/`.
- Existing coverage: interaction lists — serial: hand-counted U/V sizes on a uniform
  level-2 tree, W/X on a one-octant-refined tree, a brute-force geometric oracle on
  adaptive trees, list invariants, the adjacency helper, ghost classification, and the
  index-form views against the per-key rule; multi-rank: every
  `tests/mpi_regressions.rs` scenario compares all four index-form lists of every local
  non-ghost box against a brute-force oracle over the gathered global tree. Most
  scenarios only reach leaf level 3; `graded corner blob` adds a dense corner at leaf
  level 6 against coarse remote neighbours, and `dense max-level leaf` puts 1,000
  duplicate points into one leaf at `max_level`, so one source chunk dwarfs the others.
  Every scenario also runs the exchanges, a repeated evaluation and the recording
  operator. Scenarios
  cover refinement caps, duplicate keys, empty populations, and uneven and empty ranks.
- Add new `tests/mpi_regressions.rs` scenarios to the existing `cases` array rather
  than as new `#[test]` functions, so MPI ownership stays with one test. The scenario
  set must not shrink.
- Keep the local `../octree` path dependency unless asked to change it.
- Phase 3 rebuilt this crate (docs/phase3/README.md, T1 and T4–T7): the old map-based
  API (`InteractionManager`, `FmmGhostCommunicator`, `FmmEvaluator`, `LevelData`, the
  per-pair `FmmOperator`) was removed in T7, after T6 showed that the new evaluator
  reproduces it on every scenario. Changes now go to the code as it is; there is no
  parallel reference implementation to keep.
- Phase 5 T2 removed the index FMM (a test FMM that propagated leaf indices) with its
  tests and examples. Do not add a replacement test FMM that propagates values exactly:
  the values of the passes are `nd-fmm-exec`'s to check, against the direct sum and, on
  several ranks, against one rank (docs/phase5/README.md).

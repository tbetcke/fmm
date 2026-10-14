//! A library to generate FMM plans.
//!
//! `nd-fmm-plan` plans the topology and data flow of a distributed fast multipole method
//! on an [`nd_octree::Octree`]: which boxes interact, in which order, and when data
//! crosses ranks. It owns no kernel arithmetic; translation operators plug in through
//! [`operator::FmmOperator`]. The design is `docs/design/fmm-plan-redesign.md`.
//!
//! - [`index`]: every box a rank holds gets a dense, Morton-ordered `u32` index per
//!   level, and every leaf it needs a leaf index ([`index::BoxIndex`]).
//! - [`lists`]: the U-, V-, W- and X-lists and the parent/child relations as index
//!   arrays, held once per target (CSR) and once grouped by V-list offset or child
//!   octant; [`interaction_manager`] holds the 316 V-list offsets
//!   ([`V_LIST_DIRECTIONS`](interaction_manager::V_LIST_DIRECTIONS)) and the per-key
//!   list rule.
//! - [`plan`]: [`Plan`](plan::Plan) builds the index and the lists of every level from
//!   an octree, over the octree's locally essential keys (Phase 5S T4, T5,
//!   [Held keys](plan#held-keys)), with the owner of every top-tree key from its
//!   partition map ([Owners](plan#owners)).
//! - [`store`]: the data of an evaluation, one buffer per level and kind for
//!   multipoles and locals, and CSR leaf stores with variable counts per leaf.
//! - [`exchange`]: the ghost exchanges of sources and multipoles, the upward reduction of
//!   the owned top tree and the replicated top (Phase 5S T5,
//!   `docs/design/scale-out.md` §4).
//! - [`operator`]: the level-batched operator interface, with a per-pair adapter.
//! - [`evaluator`]: [`Evaluator`](evaluator::Evaluator) runs the distributed pass order
//!   on a plan, its stores and its exchanges.
//! - [`redistribute`]: [`Redistribution`](redistribute::Redistribution) moves per-item
//!   values (points) from the ranks that hold them to the ranks that own their leaves,
//!   grouped by leaf, and results back to the caller's order.
//!
//! # Compute graph
//!
//! [`Evaluator::evaluate`](evaluator::Evaluator::evaluate) runs the following stages in
//! order. Stages marked *collective* must be entered by every rank of the communicator
//! in the same order. Every operator call receives one level of one kind, with both
//! views of its lists: the target-centric rows and the groupings by offset or octant.
//!
//! 1. **Source exchange** (collective). The source data of every ghost leaf in a U- or
//!    X-list is fetched from its owner, with variable counts per leaf, in one exchange
//!    for all levels.
//! 2. **Local upward pass.** From the deepest level up: `p2m` on the local leaves of the
//!    level, then `m2m` from the level into the local interior boxes of its parent
//!    level. A local interior box has all its children on the local rank, so no
//!    communication is needed.
//! 3. **Owned top tree** (collective; Phase 5S T5). Every
//!    [`Global`](nd_octree::octree::KeyType::Global) box (an ancestor of this rank's
//!    coarse blocks) has one owner, the rank of its first descendant block. Level by level
//!    from the deepest coarse block up, the upward reduction sends the multipoles each
//!    rank computes to the owners of their parents, and each owner forms its `Global`
//!    boxes with `m2m` of the global pass from their eight exact children, by octant, as
//!    one rank does. Then one all-gather of fixed size replicates the top-tree boxes that
//!    many ranks read (the replicated top).
//! 4. **Multipole exchange** (collective). The multipole of every box in a V- or W-list
//!    that another rank computes, and that neither the reduction nor the replicated top
//!    delivered, is fetched from it, one exchange per level.
//! 5. **Downward pass.** Level by level from level 1: `l2l` from the parents, `m2l` over
//!    the V-list and `p2l` over the X-list. Every rank computes the locals of its
//!    `Global` boxes itself, from identical inputs in identical order, so no
//!    communication is needed (P2, [`plan`]).
//! 6. **Leaf evaluation.** Level by level: `l2p`, `m2p` over the W-list, and `p2p` over
//!    the near list (the U-list and the leaf itself) of the local leaves.
//!
//! The [`evaluator`] module documents the calls, the collectives and the accumulation
//! order of every value.
//!
//! [`Evaluator::evaluate_overlapped`](evaluator::Evaluator::evaluate_overlapped) makes
//! the same calls in the same order with the exchanges non-blocking (Phase 5 T9): the
//! source exchange travels behind stage 2, the reduction posts its receives ahead and
//! each level's sends as soon as it is formed, and the multipole exchange of every level,
//! posted after the reduction, travels behind the all-gather of the top and stage 5 until
//! the level that reads it; the output is bit for bit that of `evaluate`
//! ([Overlap](evaluator#overlap)).
//!
//! Besides the level calls, the evaluator tells the operator of each of its own data
//! movements, `reset`'s zeroing and the four exchanges, with an event of
//! [`FmmOperator::host_data`](operator::FmmOperator::host_data) that carries the index
//! lists the movement uses ([Host data](operator#host-data),
//! `docs/design/distributed-fmm.md` §6). An operator that keeps its own copies of the
//! stores, for example on a device, stays consistent through these events alone; the
//! default does nothing.
//!
//! The evaluator works on the points of the local leaves, grouped by leaf. A caller
//! whose points lie on any rank moves them to their owners with a
//! [`Redistribution`](redistribute::Redistribution) (`forward`, point to point on routes
//! fixed at build), which
//! also gives the counts per leaf, and moves the results back with `backward`
//! (`docs/design/distributed-fmm.md` §4).
//!
//! # Testing
//!
//! The crate checks topology and data flow, not values. The lists are compared with a
//! brute-force geometric oracle, the exchanges with values seeded from the keys, and
//! the evaluator with test operators that record the calls, the batches and the pairs
//! they are handed (`tests/mpi_regressions.rs`, on any number of ranks), and with an
//! order-sensitive mixing operator whose every value on several ranks must be the
//! one-rank value of the same tree, bit for bit (Phase 5S T5). The
//! redistribution is checked by round trips of payloads that name their origin, for
//! several input distributions of every scenario. The values of the passes are checked
//! by the Laplace FMM in `nd-fmm-exec` against the direct sum.
//!
//! # Future extensions
//!
//! - Reordering the accumulation to overlap more (P2P before L2P, local before ghost row
//!   parts; `docs/design/distributed-fmm.md` §8.3): not offered, since the
//!   order-preserving overlap leaves nothing exposed that a reordering would hide.
//! - Device-resident data: the views are flat index arrays and the stores flat buffers,
//!   so both can be uploaded once (design §10).

pub mod evaluator;
pub mod exchange;
pub mod index;
pub mod interaction_manager;
pub mod lists;
pub mod operator;
pub mod plan;
pub mod redistribute;
pub mod store;

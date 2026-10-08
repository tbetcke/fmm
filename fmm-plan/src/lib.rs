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
//!   an octree.
//! - [`store`]: the data of an evaluation, one buffer per level and kind for
//!   multipoles and locals, and CSR leaf stores with variable counts per leaf.
//! - [`exchange`]: the ghost exchanges of sources and multipoles, and the gather of the
//!   coarse blocks for the global upward pass.
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
//! 3. **Global upward pass** (collective). The multipoles of the coarse blocks of all
//!    ranks are gathered on every rank, and every rank computes the multipoles of the
//!    [`Global`](nd_octree::octree::KeyType::Global) boxes itself, with `m2m` of the
//!    global pass, deepest level first.
//! 4. **Multipole exchange** (collective). The multipole of every ghost box in a V- or
//!    W-list is fetched from its owner, one exchange per level.
//! 5. **Downward pass.** Level by level from level 1: `l2l` from the parents, `m2l` over
//!    the V-list and `p2l` over the X-list. Every rank computes the locals of the
//!    `Global` boxes above its own coarse blocks itself, so no communication is needed;
//!    the other `Global` boxes get no downward rows (P2, [`plan`]).
//! 6. **Leaf evaluation.** Level by level: `l2p`, `m2p` over the W-list, and `p2p` over
//!    the near list (the U-list and the leaf itself) of the local leaves.
//!
//! The [`evaluator`] module documents the calls, the collectives and the accumulation
//! order of every value.
//!
//! The evaluator works on the points of the local leaves, grouped by leaf. A caller
//! whose points lie on any rank moves them to their owners with a
//! [`Redistribution`](redistribute::Redistribution) (`forward`, one all-to-all-v), which
//! also gives the counts per leaf, and moves the results back with `backward`
//! (`docs/design/distributed-fmm.md` §4).
//!
//! # Testing
//!
//! The crate checks topology and data flow, not values. The lists are compared with a
//! brute-force geometric oracle, the exchanges with values seeded from the keys, and
//! the evaluator with test operators that record the calls, the batches and the pairs
//! they are handed (`tests/mpi_regressions.rs`, on any number of ranks). The
//! redistribution is checked by round trips of payloads that name their origin, for
//! several input distributions of every scenario. The values of the passes are checked
//! by the Laplace FMM in `nd-fmm-exec` against the direct sum.
//!
//! # Future extensions
//!
//! - Overlapping the exchanges with the computation that does not need ghost data
//!   (design §10).
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

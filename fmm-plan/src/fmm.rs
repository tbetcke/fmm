//! Distributed evaluation of an FMM.
//!
//! The evaluation is split into the *flow* of an FMM — which boxes interact,
//! in which order, and when data crosses ranks — and the *operators* that act
//! on the data of a single pair of boxes. The flow is implemented once in
//! [`evaluator::FmmEvaluator`]; the operators are supplied through the
//! [`operator::FmmOperator`] trait. [`index_fmm`] provides a test operator
//! that propagates leaf indices instead of numbers, which checks the complete
//! distributed compute graph without any numerical error.
//!
//! # Compute graph
//!
//! [`evaluator::FmmEvaluator::evaluate`] runs the following phases in order.
//! Phases marked *collective* must be entered by every rank of the octree's
//! communicator in the same order.
//!
//! 1. **Source exchange** (collective). The source data of every ghost leaf
//!    in a U- or X-list is fetched from its owner.
//! 2. **Local upward pass.** P2M on the local leaves, then M2M level by level
//!    into the local interior keys. A local interior key has all its children
//!    on the local rank, so no communication is needed.
//! 3. **Global upward pass** (collective). The multipoles of the coarse-tree
//!    leaves of all ranks are gathered on every rank, and every rank computes
//!    the multipoles of the [`Global`](nd_octree::octree::KeyType::Global) keys
//!    itself.
//! 4. **Multipole exchange** (collective). The multipole of every ghost box in
//!    a V- or W-list is fetched from its owner.
//! 5. **Downward pass.** Level by level from the root: L2L from the parent,
//!    M2L over the V-list and P2L over the X-list. The local expansions of the
//!    global keys are computed redundantly on every rank, so no communication
//!    is needed.
//! 6. **Leaf evaluation.** For every local leaf L2P, M2P over the W-list, and
//!    P2P over the U-list and the leaf itself.
//!
//! # Future extensions
//!
//! The operators are currently applied to one pair of boxes at a time. Natural
//! next steps are batched operators per level or per V-list direction (see
//! [`InteractionManager::v_list_by_direction`](crate::interaction_manager::InteractionManager::v_list_by_direction)),
//! overlapping the exchanges with the local upward pass, variable-size source
//! data per leaf, and device-resident data.

pub mod evaluator;
pub mod index_fmm;
pub mod operator;

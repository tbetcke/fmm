//! The distributed driver of an FMM on a [`Plan`].
//!
//! [`Evaluator`] owns the stores of one evaluation (design §5, §7,
//! `docs/design/fmm-plan-redesign.md`): the multipoles and locals of every box as
//! [`LevelBuffers`], and the source data, target input and target output as
//! [`LeafStore`]s with variable counts per leaf. It runs the pass order of the
//! [compute graph](crate#compute-graph) on them, one [`FmmOperator`] call per level and
//! operator kind, and fills the ghost parts of the stores with the exchanges of
//! [`exchange`](super::exchange).
//!
//! # Pass order
//!
//! L is the deepest level, `plan.nlevels() − 1`.
//!
//! | Step | Stage | Calls, in order |
//! | --- | --- | --- |
//! | 1 | [`exchange_sources`](Evaluator::exchange_sources) | the source exchange (collective) |
//! | 2 | [`upward_local`](Evaluator::upward_local) | for l = L down to 0: `p2m(l)`, then `m2m(l − 1)` of the local pass (l ≥ 1) |
//! | 3 | [`upward_global`](Evaluator::upward_global) | the coarse gather (collective); for l = L − 1 down to 0: `m2m(l)` of the global pass |
//! | 4 | [`exchange_multipoles`](Evaluator::exchange_multipoles) | the multipole exchange of every level, l = 0 to L (collective) |
//! | 5 | [`downward`](Evaluator::downward) | for l = 1 to L: `l2l(l)`, `m2l(l)`, `p2l(l)` |
//! | 6 | [`evaluate_leaves`](Evaluator::evaluate_leaves) | for l = 0 to L: `l2p(l)`, `m2p(l)`, `p2p(l)` |
//!
//! Every call is made on every rank for every level, also when its view is empty.
//!
//! # Global coarse levels
//!
//! Every rank gathers the multipoles of every rank's coarse blocks into their slots of
//! its level buffers, and then forms the multipoles of the `Global` boxes itself, with
//! the ordinary `m2m` call of the global pass, deepest level first. The downward pass
//! forms the locals of the `Global` boxes that are ancestors of the rank's own coarse
//! blocks (P2, [`plan`](super::plan)) from identical inputs in identical order, so they
//! agree bit for bit across the ranks that form them (design §7.4); the locals of the
//! other `Global` boxes stay zero. The root multipole is formed on every rank count.
//!
//! # Accumulation order
//!
//! Every value is zeroed by [`reset`](Evaluator::reset) and then receives its
//! contributions in this order (requirement 8, design §7.5), provided the operator
//! follows the [accumulation rule](super::operator#accumulation-rule):
//!
//! | Value | Contributions, in this order |
//! | --- | --- |
//! | multipole of a local leaf | `p2m` |
//! | multipole of a `LocalInterior` box | `m2m` (local pass), children by octant |
//! | multipole of a `Global` box | `m2m` (global pass), children by octant |
//! | multipole of a ghost box | overwritten by the coarse gather or the multipole exchange |
//! | local of a non-ghost box on level ≥ 1 | `l2l` from the parent; then the V-list by offset index; then the X-list by the source leaf's (level, key) |
//! | target output of a local leaf | `l2p`; then the W-list by box index; then the near list (U-list and the leaf itself) by the source leaf's (level, key) |
//!
//! With a fixed tree, ranks and counts this fixes every floating-point sum, so two
//! evaluations are bit-identical. None of these orders depends on the rank count: the
//! X and near rows follow the source leaf's (level, key), not its leaf index (P1,
//! `docs/design/distributed-fmm.md` §5.1), so on several ranks every value receives
//! its contributions in the order of one rank over the same tree.
//!
//! # Collectives
//!
//! | Call | Collectives, in order |
//! | --- | --- |
//! | [`Evaluator::new`] | all-reduce (validity); [`SourceExchange::new`]; [`MultipoleExchange::new`]; [`CoarseExchange::new`] |
//! | [`exchange_sources`](Evaluator::exchange_sources) | one neighbour all-to-all |
//! | [`upward_global`](Evaluator::upward_global) | one all-gather-v |
//! | [`exchange_multipoles`](Evaluator::exchange_multipoles) | one neighbour all-to-all per level, in level order |
//! | [`evaluate`](Evaluator::evaluate) | all of the above stages, in order |
//!
//! No collective sits in a branch that only some ranks take: the level loops run over
//! the global level count, and validation is agreed before any exchange is built. Ranks
//! with no points take part with empty slices.
//!
//! The stages are public so that a caller can time each one. Called out of order they
//! give wrong results, but every rank still enters the same collectives as long as all
//! ranks call the same stages; debug builds panic on a stage called out of order.

#[cfg(test)]
#[path = "evaluator_tests.rs"]
pub(crate) mod tests;

use std::{borrow::Borrow, error::Error, fmt};

use mpi::{collective::SystemOperation, traits::CommunicatorCollectives};

use super::exchange::{CoarseExchange, ExchangeError, MultipoleExchange, SourceExchange};
use super::operator::{FmmOperator, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, UpwardPass};
use super::plan::Plan;
use super::store::{LeafSliceMut, LeafStore, LevelBuffers, LevelSlice};

/// The reasons an [`Evaluator`] cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluatorError {
    /// The source counts do not have one entry per local leaf.
    SourceCountsLength {
        /// The number of local leaves.
        expected: usize,
        /// The number of counts given.
        actual: usize,
    },
    /// The target counts do not have one entry per local leaf.
    TargetCountsLength {
        /// The number of local leaves.
        expected: usize,
        /// The number of counts given.
        actual: usize,
    },
    /// An operator size that must be positive is zero.
    ZeroSize {
        /// The [`FmmSizes`](super::operator::FmmSizes) method that returned zero.
        size: &'static str,
    },
    /// The number of values of a store overflows `usize`.
    Overflow,
    /// Building an exchange failed, on this rank or on another.
    Exchange(ExchangeError),
    /// Validation failed on another rank.
    OtherRank,
}

impl fmt::Display for EvaluatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceCountsLength { expected, actual } => {
                write!(
                    f,
                    "{actual} source counts given for {expected} local leaves"
                )
            }
            Self::TargetCountsLength { expected, actual } => {
                write!(
                    f,
                    "{actual} target counts given for {expected} local leaves"
                )
            }
            Self::ZeroSize { size } => write!(f, "the operator's {size} is zero"),
            Self::Overflow => write!(f, "the number of values of a store overflows usize"),
            Self::Exchange(error) => write!(f, "building an exchange failed: {error}"),
            Self::OtherRank => write!(f, "building the evaluator failed on another rank"),
        }
    }
}

impl Error for EvaluatorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Exchange(error) => Some(error),
            _ => None,
        }
    }
}

/// The stages of an evaluation, in order; `Reset` is the state before the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Reset,
    ExchangeSources,
    UpwardLocal,
    UpwardGlobal,
    ExchangeMultipoles,
    Downward,
    EvaluateLeaves,
}

/// The stores of one evaluation, and the level calls that need no communication.
///
/// Kept apart from the exchanges so that the passes can be tested without MPI.
#[derive(Clone, Debug)]
pub(crate) struct Data<V> {
    pub(crate) multipoles: LevelBuffers<V>,
    pub(crate) locals: LevelBuffers<V>,
    /// Every leaf of the numbering: local, then ghost.
    pub(crate) sources: LeafStore<V>,
    /// The local leaves.
    pub(crate) target_input: LeafStore<V>,
    /// The local leaves.
    pub(crate) target_output: LeafStore<V>,
}

impl<V: Copy + Default> Data<V> {
    /// Zeroed stores for `plan`, with `source_counts` for every leaf of the numbering and
    /// `target_counts` for every local leaf; the sizes must have been validated.
    pub(crate) fn new<Op: FmmOperator<Value = V>>(
        plan: &Plan,
        operator: &Op,
        source_counts: &[usize],
        target_counts: &[usize],
    ) -> Self {
        let index = plan.index();
        let (multipole_sizes, local_sizes) = level_sizes(operator, plan.nlevels());
        Self {
            multipoles: LevelBuffers::from_index(index, &multipole_sizes),
            locals: LevelBuffers::from_index(index, &local_sizes),
            sources: LeafStore::new(source_counts, operator.source_point_size()),
            target_input: LeafStore::new(target_counts, operator.target_input_point_size()),
            target_output: LeafStore::new(target_counts, operator.target_output_point_size()),
        }
    }

    /// Zero the multipoles, the locals and the target output.
    pub(crate) fn reset(&mut self) {
        self.multipoles.clear();
        self.locals.clear();
        self.target_output.clear();
    }

    /// Step 2: P2M and the local M2M pass, deepest level first.
    pub(crate) fn upward_local<Op: FmmOperator<Value = V>>(&mut self, plan: &Plan, op: &mut Op) {
        let index = plan.index();
        for level in (0..plan.nlevels()).rev() {
            op.p2m(P2m {
                level,
                index,
                leaves: plan.level(level).p2m(),
                sources: self.sources.range(0..self.sources.nleaves()),
                multipoles: self.multipoles.level_mut(level),
            });
            if level > 0 {
                self.m2m(plan, op, level - 1, UpwardPass::Local);
            }
        }
    }

    /// The M2M pass of step 3, deepest level first, once the coarse blocks are in place.
    pub(crate) fn upward_global<Op: FmmOperator<Value = V>>(&mut self, plan: &Plan, op: &mut Op) {
        for level in (0..plan.nlevels().saturating_sub(1)).rev() {
            self.m2m(plan, op, level, UpwardPass::Global);
        }
    }

    /// One M2M call: the parents of `level` from their children on `level + 1`.
    fn m2m<Op: FmmOperator<Value = V>>(
        &mut self,
        plan: &Plan,
        op: &mut Op,
        level: usize,
        pass: UpwardPass,
    ) {
        let lists = plan.level(level);
        let (multipoles, child_multipoles) = self.multipoles.parent_child_mut(level);
        op.m2m(M2m {
            level,
            index: plan.index(),
            pass,
            children: match pass {
                UpwardPass::Local => lists.m2m_local(),
                UpwardPass::Global => lists.m2m_global(),
            },
            child_multipoles,
            multipoles,
        });
    }

    /// Step 5: L2L, M2L and P2L, from level 1 down.
    pub(crate) fn downward<Op: FmmOperator<Value = V>>(&mut self, plan: &Plan, op: &mut Op) {
        let index = plan.index();
        for level in 1..plan.nlevels() {
            let lists = plan.level(level);
            let (locals, parent_locals) = self.locals.child_parent_mut(level - 1);
            op.l2l(L2l {
                level,
                index,
                parents: lists.l2l(),
                parent_locals,
                locals,
            });
            op.m2l(M2l {
                level,
                index,
                pairs: lists.v(),
                multipoles: self.multipoles.level(level),
                locals: self.locals.level_mut(level),
            });
            op.p2l(P2l {
                level,
                index,
                x: lists.x(),
                sources: self.sources.range(0..self.sources.nleaves()),
                locals: self.locals.level_mut(level),
            });
        }
    }

    /// Step 6: L2P, M2P and P2P for the local leaves of every level.
    pub(crate) fn evaluate_leaves<Op: FmmOperator<Value = V>>(&mut self, plan: &Plan, op: &mut Op) {
        let index = plan.index();
        let nlevels = plan.nlevels();
        for level in 0..nlevels {
            let lists = plan.level(level);
            let leaves = index.leaves().local(level);
            op.l2p(L2p {
                level,
                index,
                leaves: leaves.clone(),
                boxes: lists.l2p(),
                locals: self.locals.level(level),
                target_input: self.target_input.range(leaves.clone()),
                target_output: self.target_output.range_mut(leaves.clone()),
            });
            op.m2p(M2p {
                level,
                index,
                leaves: leaves.clone(),
                w: lists.w(),
                multipoles: if level + 1 < nlevels {
                    self.multipoles.level(level + 1)
                } else {
                    LevelSlice::empty(self.multipoles.size(level))
                },
                target_input: self.target_input.range(leaves.clone()),
                target_output: self.target_output.range_mut(leaves.clone()),
            });
            op.p2p(P2p {
                level,
                index,
                leaves: leaves.clone(),
                near: lists.near(),
                sources: self.sources.range(0..self.sources.nleaves()),
                target_input: self.target_input.range(leaves.clone()),
                target_output: self.target_output.range_mut(leaves),
            });
        }
    }
}

/// The multipole and local sizes of `operator` on the levels `0..nlevels`.
fn level_sizes<Op: FmmOperator>(operator: &Op, nlevels: usize) -> (Vec<usize>, Vec<usize>) {
    (
        (0..nlevels).map(|l| operator.multipole_size(l)).collect(),
        (0..nlevels).map(|l| operator.local_size(l)).collect(),
    )
}

/// Check the counts and the operator sizes for `plan`, locally.
pub(crate) fn validate<Op: FmmOperator>(
    plan: &Plan,
    operator: &Op,
    source_counts: &[usize],
    target_counts: &[usize],
) -> Result<(), EvaluatorError> {
    let index = plan.index();
    let nlocal = index.leaves().nlocal();
    if source_counts.len() != nlocal {
        return Err(EvaluatorError::SourceCountsLength {
            expected: nlocal,
            actual: source_counts.len(),
        });
    }
    if target_counts.len() != nlocal {
        return Err(EvaluatorError::TargetCountsLength {
            expected: nlocal,
            actual: target_counts.len(),
        });
    }
    let (multipole_sizes, local_sizes) = level_sizes(operator, plan.nlevels());
    let zero = |size| Err(EvaluatorError::ZeroSize { size });
    if multipole_sizes.contains(&0) {
        return zero("multipole_size");
    }
    if local_sizes.contains(&0) {
        return zero("local_size");
    }
    if operator.source_point_size() == 0 {
        return zero("source_point_size");
    }
    if operator.target_output_point_size() == 0 {
        return zero("target_output_point_size");
    }

    let level_values = |sizes: &[usize]| {
        sizes
            .iter()
            .enumerate()
            .try_fold(0usize, |total, (l, &size)| {
                index.len(l).checked_mul(size)?.checked_add(total)
            })
    };
    let point_values = |counts: &[usize], point_size: usize| {
        counts
            .iter()
            .try_fold(0usize, |total, &count| total.checked_add(count))?
            .checked_mul(point_size)
    };
    let point_size = operator
        .target_input_point_size()
        .max(operator.target_output_point_size());
    if level_values(&multipole_sizes).is_none()
        || level_values(&local_sizes).is_none()
        || point_values(source_counts, operator.source_point_size()).is_none()
        || point_values(target_counts, point_size).is_none()
    {
        return Err(EvaluatorError::Overflow);
    }
    Ok(())
}

/// Distributed evaluation of an FMM with the batched operators `Op` on a [`Plan`].
///
/// See the [module documentation](self) for the pass order, the accumulation order and
/// the collectives.
///
/// The evaluator holds its plan as `P`: by default it borrows it (`&'p Plan`), and with
/// `P = Plan` it owns it, so that an evaluator can be stored next to the octree it was
/// planned from without borrowing from its own owner. Both behave the same.
pub struct Evaluator<'p, C: CommunicatorCollectives, Op: FmmOperator, P: Borrow<Plan> = &'p Plan> {
    plan: P,
    comm: &'p C,
    operator: Op,
    data: Data<Op::Value>,
    source_exchange: SourceExchange<Op::Value>,
    multipole_exchange: MultipoleExchange<Op::Value>,
    coarse_exchange: CoarseExchange<Op::Value>,
    completed: Stage,
}

impl<'p, C: CommunicatorCollectives, Op: FmmOperator, P: Borrow<Plan>> Evaluator<'p, C, Op, P> {
    /// Build the stores and exchanges of an evaluation on `plan` with `operator`.
    ///
    /// `plan` is a `&Plan` or, for an evaluator that owns its plan, a `Plan`.
    ///
    /// `source_counts[j]` and `target_counts[j]` are the numbers of source and target
    /// points of local leaf j, in the leaf order of
    /// [`LeafNumbering`](super::index::LeafNumbering); zero is allowed. The ghost leaves'
    /// source counts are their owners', delivered by the source exchange. Every store
    /// starts zeroed: fill the local sources and the target input with
    /// [`sources_mut`](Self::sources_mut) and
    /// [`target_input_mut`](Self::target_input_mut) before
    /// [`evaluate`](Self::evaluate). New counts need a new evaluator.
    ///
    /// # Collective operation
    /// On `comm`, the communicator of the octree of `plan`: one all-reduce that agrees
    /// the validation, then the builds of the source, multipole and coarse exchanges, in
    /// that order, on every rank.
    ///
    /// # Errors
    /// Returned on every rank if, on any rank, a count slice does not have one entry per
    /// local leaf, an operator size that must be positive is zero, a store overflows, or
    /// an exchange cannot be built (for example because the ranks' operators disagree on
    /// a size). The rank that found the defect returns it; the other ranks return
    /// [`EvaluatorError::OtherRank`] (or the exchange's
    /// [`ExchangeError::OtherRank`]).
    pub fn new(
        plan: P,
        comm: &'p C,
        operator: Op,
        source_counts: &[usize],
        target_counts: &[usize],
    ) -> Result<Self, EvaluatorError> {
        let planned: &Plan = plan.borrow();
        let local = validate(planned, &operator, source_counts, target_counts);
        let mut valid = false;
        comm.all_reduce_into(&local.is_ok(), &mut valid, SystemOperation::logical_and());
        local?;
        if !valid {
            return Err(EvaluatorError::OtherRank);
        }

        let (multipole_sizes, _) = level_sizes(&operator, planned.nlevels());
        let source_exchange =
            SourceExchange::new(planned, comm, source_counts, operator.source_point_size())
                .map_err(EvaluatorError::Exchange)?;
        let multipole_exchange = MultipoleExchange::new(planned, comm, &multipole_sizes)
            .map_err(EvaluatorError::Exchange)?;
        let coarse_exchange = CoarseExchange::new(planned, comm, &multipole_sizes)
            .map_err(EvaluatorError::Exchange)?;

        let data = Data::new(
            planned,
            &operator,
            source_exchange.leaf_counts(),
            target_counts,
        );
        Ok(Self {
            plan,
            comm,
            operator,
            data,
            source_exchange,
            multipole_exchange,
            coarse_exchange,
            completed: Stage::Reset,
        })
    }

    /// Return the plan.
    pub fn plan(&self) -> &Plan {
        self.plan.borrow()
    }

    /// Return the communicator.
    pub fn comm(&self) -> &'p C {
        self.comm
    }

    /// Return the operator.
    pub fn operator(&self) -> &Op {
        &self.operator
    }

    /// Return the operator mutably, for example to change a parameter between
    /// evaluations. Its sizes must not change.
    pub fn operator_mut(&mut self) -> &mut Op {
        &mut self.operator
    }

    /// Return the source data of the local leaf `leaf` mutably: `count · point_size`
    /// values.
    ///
    /// # Panics
    ///
    /// Panics if `leaf` is not a local leaf index.
    pub fn sources_mut(&mut self, leaf: usize) -> &mut [Op::Value] {
        self.assert_local(leaf);
        self.data.sources.chunk_mut(leaf)
    }

    /// Return the source data of every local leaf mutably, leaf r of the slice being
    /// local leaf r.
    pub fn local_sources_mut(&mut self) -> LeafSliceMut<'_, Op::Value> {
        let nlocal = self.plan().index().leaves().nlocal();
        self.data.sources.range_mut(0..nlocal)
    }

    /// Return the target input of the local leaf `leaf` mutably.
    ///
    /// # Panics
    ///
    /// Panics if `leaf` is not a local leaf index.
    pub fn target_input_mut(&mut self, leaf: usize) -> &mut [Op::Value] {
        self.assert_local(leaf);
        self.data.target_input.chunk_mut(leaf)
    }

    /// Return the target input of every local leaf mutably, leaf r of the slice being
    /// local leaf r.
    pub fn local_target_inputs_mut(&mut self) -> LeafSliceMut<'_, Op::Value> {
        let nlocal = self.data.target_input.nleaves();
        self.data.target_input.range_mut(0..nlocal)
    }

    /// Return the target output of the local leaf `leaf`, as of the last evaluation.
    ///
    /// # Panics
    ///
    /// Panics if `leaf` is not a local leaf index.
    pub fn target_output(&self, leaf: usize) -> &[Op::Value] {
        self.assert_local(leaf);
        self.data.target_output.chunk(leaf)
    }

    /// Return the source store: every leaf of the numbering, the ghost leaves with the
    /// values of the last source exchange.
    pub fn source_store(&self) -> &LeafStore<Op::Value> {
        &self.data.sources
    }

    /// Return the target-input store of the local leaves.
    pub fn target_input_store(&self) -> &LeafStore<Op::Value> {
        &self.data.target_input
    }

    /// Return the target-output store of the local leaves.
    pub fn target_output_store(&self) -> &LeafStore<Op::Value> {
        &self.data.target_output
    }

    /// Return the multipoles of every box.
    pub fn multipoles(&self) -> &LevelBuffers<Op::Value> {
        &self.data.multipoles
    }

    /// Return the locals of every box; those of ghost boxes stay zero, and so do those
    /// of the `Global` boxes that are not an ancestor of one of this rank's coarse blocks
    /// (P2, [`plan`](super::plan)).
    pub fn locals(&self) -> &LevelBuffers<Op::Value> {
        &self.data.locals
    }

    /// Zero the multipoles, the locals and the target output. Sources and target input
    /// are kept.
    pub fn reset(&mut self) {
        self.data.reset();
        self.completed = Stage::Reset;
    }

    /// [`reset`](Self::reset), then run every stage in order.
    ///
    /// # Collective operation
    /// Every rank must call it; see the [module documentation](self).
    pub fn evaluate(&mut self) {
        self.reset();
        self.exchange_sources();
        self.upward_local();
        self.upward_global();
        self.exchange_multipoles();
        self.downward();
        self.evaluate_leaves();
    }

    /// Step 1: fetch the sources of the ghost leaves of the U- and X-lists.
    ///
    /// # Collective operation
    /// One neighbour all-to-all; every rank must call it.
    pub fn exchange_sources(&mut self) {
        self.enter(Stage::ExchangeSources);
        self.source_exchange.forward(&mut self.data.sources);
    }

    /// Step 2: P2M on the local leaves and M2M into the `LocalInterior` boxes, deepest
    /// level first. Local.
    pub fn upward_local(&mut self) {
        self.enter(Stage::UpwardLocal);
        self.data
            .upward_local(self.plan.borrow(), &mut self.operator);
    }

    /// Step 3: gather every rank's coarse-block multipoles and form the multipoles of
    /// the `Global` boxes, deepest level first.
    ///
    /// # Collective operation
    /// One all-gather-v; every rank must call it.
    pub fn upward_global(&mut self) {
        self.enter(Stage::UpwardGlobal);
        self.coarse_exchange.gather(&mut self.data.multipoles);
        self.data
            .upward_global(self.plan.borrow(), &mut self.operator);
    }

    /// Step 4: fetch the multipoles of the ghost boxes of the V- and W-lists, level by
    /// level.
    ///
    /// # Collective operation
    /// One neighbour all-to-all per level, in level order; every rank must call it.
    pub fn exchange_multipoles(&mut self) {
        self.enter(Stage::ExchangeMultipoles);
        self.multipole_exchange
            .forward_all(&mut self.data.multipoles);
    }

    /// Step 5: L2L, M2L and P2L, level by level from level 1. Local.
    pub fn downward(&mut self) {
        self.enter(Stage::Downward);
        self.data.downward(self.plan.borrow(), &mut self.operator);
    }

    /// Step 6: L2P, M2P and P2P on the local leaves of every level. Local.
    pub fn evaluate_leaves(&mut self) {
        self.enter(Stage::EvaluateLeaves);
        self.data
            .evaluate_leaves(self.plan.borrow(), &mut self.operator);
    }

    /// Record that `stage` runs; debug builds check that it follows the last one.
    fn enter(&mut self, stage: Stage) {
        debug_assert!(
            stage as u8 == self.completed as u8 + 1,
            "stage {stage:?} called after {:?}; call the stages in order, after reset",
            self.completed
        );
        self.completed = stage;
    }

    fn assert_local(&self, leaf: usize) {
        let nlocal = self.plan().index().leaves().nlocal();
        assert!(
            leaf < nlocal,
            "leaf {leaf} is not one of the {nlocal} local leaves"
        );
    }
}

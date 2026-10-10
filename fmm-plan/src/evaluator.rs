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
//! [`evaluate_overlapped`](Evaluator::evaluate_overlapped) runs the same calls in the same
//! order with the exchanges non-blocking ([Overlap](#overlap)).
//!
//! # Host data
//!
//! Outside the level calls the evaluator reads and writes its stores in these places
//! only, and tells the operator about each with an event of
//! [`FmmOperator::host_data`] ([`HostData`], [Host data](super::operator#host-data);
//! `docs/design/distributed-fmm.md` §6):
//!
//! | Stage | Reads | Writes | Events, in order with the movement |
//! | --- | --- | --- | --- |
//! | [`new`](Evaluator::new) | — | allocates the five stores zeroed | none |
//! | [`reset`](Evaluator::reset) | — | every multipole, local and target output (zeroed) | the zeroing; `Reset` |
//! | 1 [`exchange_sources`](Evaluator::exchange_sources) | the source chunks of `send_leaves` | the ghost tail of the sources (`ghost_leaves`) | `SendSources`; the exchange; `ReceivedSources` |
//! | 2 [`upward_local`](Evaluator::upward_local) | — | — | none |
//! | 3 [`upward_global`](Evaluator::upward_global) | the multipoles of this rank's coarse blocks (`sent_blocks`) | the multipoles of the other ranks' blocks (`received_blocks`) | `SendMultipoles` (also for step 4's sends); the gather; `ReceivedCoarse`; then the global `m2m` calls |
//! | 4 [`exchange_multipoles`](Evaluator::exchange_multipoles) | per level, the multipoles of `send_boxes(l)` | per level, the multipoles of `receive_boxes(l)` | for l = 0 to L: the exchange of level l; `ReceivedMultipoles` of level l |
//! | 5, 6 | — | — | none |
//!
//! The overlapped stages ([Overlap](#overlap)) fire the same events with the same lists
//! at the same logical points (design §6.5): `SendSources` before the source exchange is
//! posted, `ReceivedSources` after its wait; `SendMultipoles` before the multipole
//! exchange is posted, ahead of the gather; `ReceivedCoarse` after the gather;
//! `ReceivedMultipoles` of level l after level l's wait, before the first call of level l
//! that reads ghosts, so between the downward calls of levels l − 1 and l. No event fires
//! while a request is pending on a buffer it hands out: the send buffers are packed after
//! the "send" event returns, and a "received" event comes after its wait.
//!
//! So an evaluation fires 5 + `plan.nlevels()` events, on every rank, also when an
//! exchange moves nothing; on one rank every index list is empty. On one rank the coarse
//! gather still reads the root's multipole (the one coarse block, before the global M2M
//! forms it) for a gather to the rank itself, whose value nobody uses, so `sent_blocks`
//! is empty there. The caller writes the sources and the target input through
//! [`sources_mut`](Evaluator::sources_mut),
//! [`local_sources_mut`](Evaluator::local_sources_mut),
//! [`target_input_mut`](Evaluator::target_input_mut) and
//! [`local_target_inputs_mut`](Evaluator::local_target_inputs_mut); the level calls
//! write only their batch's output. Nothing else in the crate reads or writes a store.
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
//! | [`exchange_sources_and_upward_local`](Evaluator::exchange_sources_and_upward_local) | none: point-to-point requests |
//! | [`far_field`](Evaluator::far_field) | one all-gather-v, with the multipole exchange's point-to-point requests in flight |
//! | [`evaluate_overlapped`](Evaluator::evaluate_overlapped) | the overlapped stages, in order |
//!
//! No collective sits in a branch that only some ranks take: the level loops run over
//! the global level count, and validation is agreed before any exchange is built. Ranks
//! with no points take part with empty slices.
//!
//! The stages are public so that a caller can time each one. Called out of order they
//! give wrong results, but every rank still enters the same collectives as long as all
//! ranks call the same stages; debug builds panic on a stage called out of order, and on
//! a mix of the two paths in one evaluation. Every rank must take the same path.
//!
//! # Overlap
//!
//! [`evaluate_overlapped`](Evaluator::evaluate_overlapped) hides the exchanges behind
//! work that does not need their data, without moving a level call
//! (`docs/design/distributed-fmm.md` §8.1, decisions 9 and 10 of docs/phase5/README.md):
//!
//! | Step | Stage | Replaces | Does, in order |
//! | --- | --- | --- | --- |
//! | 1, 2 | [`exchange_sources_and_upward_local`](Evaluator::exchange_sources_and_upward_local) | `exchange_sources`, `upward_local` | `SendSources`; post the source exchange; the calls of `upward_local`, a `test` after each; wait; `ReceivedSources` |
//! | 3, 4, 5 | [`far_field`](Evaluator::far_field) | `upward_global`, `exchange_multipoles`, `downward` | `SendMultipoles`; post the multipole exchange of every level; the coarse gather (blocking) and `ReceivedCoarse`; the global M2M calls; for l = 0 to L: wait for level l, `ReceivedMultipoles` of l, and for l ≥ 1 the calls of `downward` on level l; a `test` after every call |
//! | 6 | [`evaluate_leaves`](Evaluator::evaluate_leaves) | — | unchanged |
//!
//! - **Why it is correct.** P2M and the local M2M read local sources only, so the
//!   ghost sources can travel behind them. Every multipole an exchange sends is a local
//!   leaf or local interior box, final after the local upward pass, so the multipole
//!   exchange is posted before the gather. Level l of the downward pass reads ghost
//!   multipoles of level l only (V); stage 6 reads those of every level (W), all received
//!   by the end of the far field. The coarse gather stays blocking: its inputs are final
//!   only after the local upward pass and its outputs feed the global M2M (design §8.5).
//! - **Determinism.** No level call moves, so every value receives its contributions in
//!   the order of [Accumulation order](#accumulation-order), and the output, the stores
//!   and the events are bit for bit those of [`evaluate`](Evaluator::evaluate); messages
//!   are copied, never combined, in whatever order they arrive. No reordering is offered
//!   (decision 9).
//! - **Progress.** MPI moves a message only inside MPI calls, so a `test` of the pending
//!   requests follows every level call (2L + 1 in the upward pass; L global M2M and 3L
//!   downward calls in the far field), on the calling thread; operators that use threads
//!   run them inside the level calls, and no worker calls MPI. A message larger than the
//!   work between two calls can hide is finished in the wait.
//! - **Times.** [`overlap_times`](Evaluator::overlap_times) gives, per exchange, the time
//!   from post to completion, the exposed wait and the time in `test` calls, the wait per
//!   level, the coarse gather, and the work of each part ([`OverlapTimes`]).

#[cfg(test)]
#[path = "evaluator_tests.rs"]
pub(crate) mod tests;

use std::{
    borrow::Borrow,
    error::Error,
    fmt,
    time::{Duration, Instant},
};

use mpi::{collective::SystemOperation, traits::CommunicatorCollectives};
use nd_octree::constants::DEEPEST_LEVEL;

use super::exchange::{
    CoarseExchange, ExchangeError, ExchangeTimes, MultipoleExchange, SourceExchange,
};
use super::operator::{FmmOperator, HostData, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, UpwardPass};
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

/// The stages of an evaluation; `Reset` is the state before the first. The blocking
/// path runs `ExchangeSources` to `Downward`, the overlapped path `SourcesAndUpward` and
/// `FarField` in their place; both end with `EvaluateLeaves`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Reset,
    ExchangeSources,
    UpwardLocal,
    UpwardGlobal,
    ExchangeMultipoles,
    Downward,
    EvaluateLeaves,
    SourcesAndUpward,
    FarField,
}

impl Stage {
    /// The stages `self` may follow.
    fn after(self) -> &'static [Stage] {
        match self {
            Self::Reset => &[],
            Self::ExchangeSources | Self::SourcesAndUpward => &[Self::Reset],
            Self::UpwardLocal => &[Self::ExchangeSources],
            Self::UpwardGlobal => &[Self::UpwardLocal],
            Self::ExchangeMultipoles => &[Self::UpwardGlobal],
            Self::Downward => &[Self::ExchangeMultipoles],
            Self::FarField => &[Self::SourcesAndUpward],
            Self::EvaluateLeaves => &[Self::Downward, Self::FarField],
        }
    }
}

/// The number of levels an octree can have, 0 to `DEEPEST_LEVEL`: the length of
/// [`OverlapTimes::level_waits`].
pub const MAX_LEVELS: usize = DEEPEST_LEVEL as usize + 1;

/// Wall times of the overlapped stages of the last evaluation
/// ([`Evaluator::overlap_times`]; `docs/design/distributed-fmm.md` §8.6), read on the
/// calling thread, for reports; nothing depends on them. Zero after a blocking
/// evaluation.
///
/// The parts of a stage that are not communication are named after the blocking stages
/// they replace, so that the two paths can be compared:
/// [`exchange_sources_and_upward_local`](Evaluator::exchange_sources_and_upward_local)
/// lasts [`upward_local`](Self::upward_local) plus the time of the source exchange outside
/// it (packing, posting, `test` calls, the wait, the copy into the ghost tail and the
/// events), and [`far_field`](Evaluator::far_field) lasts
/// [`upward_global`](Self::upward_global) plus [`downward`](Self::downward) plus that of the
/// multipole exchange.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OverlapTimes {
    /// The source exchange ([`SourceExchange::forward_overlapped`]).
    pub sources: ExchangeTimes,
    /// The multipole exchange of every level
    /// ([`MultipoleExchange::forward_all_overlapped`]).
    pub multipoles: ExchangeTimes,
    /// The part of [`multipoles`](Self::multipoles)`.exposed` spent waiting for each
    /// level, `level_waits[l]` for level l; the final wait for the sends is the rest.
    pub level_waits: [Duration; MAX_LEVELS],
    /// The coarse gather (blocking, inside the far field).
    pub coarse_gather: Duration,
    /// The level calls of the local upward pass (P2M, local M2M), without the `test`
    /// calls between them.
    pub upward_local: Duration,
    /// The `SendMultipoles` event, the coarse gather, the `ReceivedCoarse` event and the
    /// global M2M calls, without the `test` calls: what the blocking `upward_global` does.
    pub upward_global: Duration,
    /// The level calls of the downward pass (L2L, M2L, P2L), without the `test` calls.
    pub downward: Duration,
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
        upward_local(plan, op, &self.sources, &mut self.multipoles, || {});
    }

    /// The M2M pass of step 3, deepest level first, once the coarse blocks are in place.
    pub(crate) fn upward_global<Op: FmmOperator<Value = V>>(&mut self, plan: &Plan, op: &mut Op) {
        upward_global(plan, op, &mut self.multipoles, || {});
    }

    /// Step 5: L2L, M2L and P2L, from level 1 down.
    pub(crate) fn downward<Op: FmmOperator<Value = V>>(&mut self, plan: &Plan, op: &mut Op) {
        for level in 1..plan.nlevels() {
            downward_level(
                plan,
                op,
                level,
                &self.multipoles,
                &mut self.locals,
                &self.sources,
                || {},
            );
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

/// Step 2 on `sources` and `multipoles`: P2M and the local M2M pass, deepest level first,
/// with `between` after every level call.
fn upward_local<V: Copy + Default, Op: FmmOperator<Value = V>>(
    plan: &Plan,
    op: &mut Op,
    sources: &LeafStore<V>,
    multipoles: &mut LevelBuffers<V>,
    mut between: impl FnMut(),
) {
    let index = plan.index();
    for level in (0..plan.nlevels()).rev() {
        op.p2m(P2m {
            level,
            index,
            leaves: plan.level(level).p2m(),
            sources: sources.range(0..sources.nleaves()),
            multipoles: multipoles.level_mut(level),
        });
        between();
        if level > 0 {
            m2m(plan, op, multipoles, level - 1, UpwardPass::Local);
            between();
        }
    }
}

/// The M2M pass of step 3 on `multipoles`, deepest level first, with `between` after
/// every level call.
fn upward_global<V: Copy + Default, Op: FmmOperator<Value = V>>(
    plan: &Plan,
    op: &mut Op,
    multipoles: &mut LevelBuffers<V>,
    mut between: impl FnMut(),
) {
    for level in (0..plan.nlevels().saturating_sub(1)).rev() {
        m2m(plan, op, multipoles, level, UpwardPass::Global);
        between();
    }
}

/// One M2M call: the parents of `level` from their children on `level + 1`.
fn m2m<V: Copy + Default, Op: FmmOperator<Value = V>>(
    plan: &Plan,
    op: &mut Op,
    multipoles: &mut LevelBuffers<V>,
    level: usize,
    pass: UpwardPass,
) {
    let lists = plan.level(level);
    let (multipoles, child_multipoles) = multipoles.parent_child_mut(level);
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

/// Step 5 on one level (≥ 1): L2L, M2L and P2L into `locals`, with `between` after every
/// level call.
fn downward_level<V: Copy + Default, Op: FmmOperator<Value = V>>(
    plan: &Plan,
    op: &mut Op,
    level: usize,
    multipoles: &LevelBuffers<V>,
    all_locals: &mut LevelBuffers<V>,
    sources: &LeafStore<V>,
    mut between: impl FnMut(),
) {
    let index = plan.index();
    let lists = plan.level(level);
    let (locals, parent_locals) = all_locals.child_parent_mut(level - 1);
    op.l2l(L2l {
        level,
        index,
        parents: lists.l2l(),
        parent_locals,
        locals,
    });
    between();
    op.m2l(M2l {
        level,
        index,
        pairs: lists.v(),
        multipoles: multipoles.level(level),
        locals: all_locals.level_mut(level),
    });
    between();
    op.p2l(P2l {
        level,
        index,
        x: lists.x(),
        sources: sources.range(0..sources.nleaves()),
        locals: all_locals.level_mut(level),
    });
    between();
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
    overlap: OverlapTimes,
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
            overlap: OverlapTimes::default(),
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

    /// Return the source exchange, for its index lists: an operator that keeps its own
    /// stores can size its buffers from them before the first evaluation, where the
    /// [`HostData`] events hand it the same lists.
    pub fn source_exchange(&self) -> &SourceExchange<Op::Value> {
        &self.source_exchange
    }

    /// Return the multipole exchange, for its index lists (as
    /// [`source_exchange`](Self::source_exchange)).
    pub fn multipole_exchange(&self) -> &MultipoleExchange<Op::Value> {
        &self.multipole_exchange
    }

    /// Return the coarse gather, for its blocks (as
    /// [`source_exchange`](Self::source_exchange)).
    pub fn coarse_exchange(&self) -> &CoarseExchange<Op::Value> {
        &self.coarse_exchange
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
    ///
    /// The operator then receives [`HostData::Reset`].
    pub fn reset(&mut self) {
        self.data.reset();
        self.completed = Stage::Reset;
        self.overlap = OverlapTimes::default();
        self.operator.host_data(HostData::Reset);
    }

    /// Return the times of the overlapped stages of the last evaluation ([`OverlapTimes`]);
    /// zero after a blocking one.
    pub fn overlap_times(&self) -> OverlapTimes {
        self.overlap
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

    /// [`reset`](Self::reset), then the overlapped stages: the same level calls, events and
    /// output bit for bit as [`evaluate`](Self::evaluate), with the source and multipole
    /// exchanges non-blocking behind local work ([Overlap](self#overlap)).
    ///
    /// # Collective operation
    /// Every rank must call it, or every rank [`evaluate`](Self::evaluate); see the
    /// [module documentation](self).
    pub fn evaluate_overlapped(&mut self) {
        self.reset();
        self.exchange_sources_and_upward_local();
        self.far_field();
        self.evaluate_leaves();
    }

    /// Step 1: fetch the sources of the ghost leaves of the U- and X-lists.
    ///
    /// The operator receives [`HostData::SendSources`] before the exchange and
    /// [`HostData::ReceivedSources`] after it.
    ///
    /// # Collective operation
    /// One neighbour all-to-all; every rank must call it.
    pub fn exchange_sources(&mut self) {
        self.enter(Stage::ExchangeSources);
        self.operator.host_data(HostData::SendSources {
            leaves: self.source_exchange.send_leaves(),
            sources: &mut self.data.sources,
        });
        self.source_exchange.forward(&mut self.data.sources);
        self.operator.host_data(HostData::ReceivedSources {
            leaves: self.source_exchange.ghost_leaves(),
            sources: &self.data.sources,
        });
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
    /// The operator receives [`HostData::SendMultipoles`] before the gather, for every
    /// multipole the evaluation sends (the coarse blocks and every level's multipole
    /// exchange), and [`HostData::ReceivedCoarse`] after it.
    ///
    /// # Collective operation
    /// One all-gather-v; every rank must call it.
    pub fn upward_global(&mut self) {
        self.enter(Stage::UpwardGlobal);
        self.operator.host_data(HostData::SendMultipoles {
            coarse: &self.coarse_exchange,
            exchange: &self.multipole_exchange,
            multipoles: &mut self.data.multipoles,
        });
        self.coarse_exchange.gather(&mut self.data.multipoles);
        self.operator.host_data(HostData::ReceivedCoarse {
            coarse: &self.coarse_exchange,
            multipoles: &self.data.multipoles,
        });
        self.data
            .upward_global(self.plan.borrow(), &mut self.operator);
    }

    /// Step 4: fetch the multipoles of the ghost boxes of the V- and W-lists, level by
    /// level.
    ///
    /// The operator receives [`HostData::ReceivedMultipoles`] after the exchange of each
    /// level.
    ///
    /// # Collective operation
    /// One neighbour all-to-all per level, in level order; every rank must call it.
    pub fn exchange_multipoles(&mut self) {
        self.enter(Stage::ExchangeMultipoles);
        for level in 0..self.multipole_exchange.nlevels() {
            self.multipole_exchange
                .forward(level, &mut self.data.multipoles);
            self.operator.host_data(HostData::ReceivedMultipoles {
                level,
                exchange: &self.multipole_exchange,
                multipoles: &self.data.multipoles,
            });
        }
    }

    /// Step 5: L2L, M2L and P2L, level by level from level 1. Local.
    pub fn downward(&mut self) {
        self.enter(Stage::Downward);
        self.data.downward(self.plan.borrow(), &mut self.operator);
    }

    /// Steps 1 and 2 overlapped: the source exchange in flight behind P2M and the local
    /// M2M ([Overlap](self#overlap)).
    ///
    /// The operator receives [`HostData::SendSources`] before the exchange is posted and
    /// [`HostData::ReceivedSources`] after its wait, as in
    /// [`exchange_sources`](Self::exchange_sources); in between come the level calls of
    /// [`upward_local`](Self::upward_local), with a `test` of the exchange after each.
    ///
    /// # Collective operation
    /// No collective: one point-to-point receive and send per neighbour; every rank must
    /// call it, in place of [`exchange_sources`](Self::exchange_sources) and
    /// [`upward_local`](Self::upward_local).
    pub fn exchange_sources_and_upward_local(&mut self) {
        self.enter(Stage::SourcesAndUpward);
        let Self {
            plan,
            operator,
            data,
            source_exchange,
            overlap,
            ..
        } = self;
        let plan: &Plan = (*plan).borrow();
        operator.host_data(HostData::SendSources {
            leaves: source_exchange.send_leaves(),
            sources: &mut data.sources,
        });
        let multipoles = &mut data.multipoles;
        let (work, times) =
            source_exchange.forward_overlapped(&mut data.sources, |sources, flight| {
                let start = Instant::now();
                upward_local(plan, operator, sources, multipoles, || flight.test());
                start.elapsed().saturating_sub(flight.times().progress)
            });
        overlap.sources = times;
        overlap.upward_local = work;
        operator.host_data(HostData::ReceivedSources {
            leaves: source_exchange.ghost_leaves(),
            sources: &data.sources,
        });
    }

    /// Steps 3, 4 and 5 overlapped: the multipole exchange of every level in flight
    /// behind the coarse gather, the global M2M and the downward pass of the coarser
    /// levels ([Overlap](self#overlap)).
    ///
    /// In order: [`HostData::SendMultipoles`]; the exchange of every level posted; the
    /// coarse gather (blocking) and [`HostData::ReceivedCoarse`]; the global M2M calls;
    /// then for l = 0 to L the wait for level l and [`HostData::ReceivedMultipoles`] of
    /// level l, and for l ≥ 1 the L2L, M2L and P2L calls of level l. A `test` of the
    /// exchange follows every level call. The level calls and the events come in the
    /// order of the blocking stages; only the `ReceivedMultipoles` events move, each to
    /// just before the first call that reads its level's ghosts.
    ///
    /// # Collective operation
    /// One all-gather-v (the coarse gather) between posting and waiting for the multipole
    /// exchange, point-to-point otherwise; every rank must call it, in place of
    /// [`upward_global`](Self::upward_global),
    /// [`exchange_multipoles`](Self::exchange_multipoles) and
    /// [`downward`](Self::downward).
    pub fn far_field(&mut self) {
        self.enter(Stage::FarField);
        let Self {
            plan,
            operator,
            data,
            multipole_exchange,
            coarse_exchange,
            overlap,
            ..
        } = self;
        let plan: &Plan = (*plan).borrow();
        let start = Instant::now();
        operator.host_data(HostData::SendMultipoles {
            coarse: coarse_exchange,
            exchange: multipole_exchange,
            multipoles: &mut data.multipoles,
        });
        let mut upward_global_time = start.elapsed();
        let (mut downward, mut waits, mut gather) =
            (Duration::ZERO, [Duration::ZERO; MAX_LEVELS], Duration::ZERO);
        let Data {
            multipoles,
            locals,
            sources,
            ..
        } = data;
        let ((), times) =
            multipole_exchange.forward_all_overlapped(multipoles, |flight, multipoles| {
                let start = Instant::now();
                coarse_exchange.gather(multipoles);
                gather = start.elapsed();
                operator.host_data(HostData::ReceivedCoarse {
                    coarse: coarse_exchange,
                    multipoles,
                });
                upward_global(plan, operator, multipoles, || flight.test());
                upward_global_time += start.elapsed().saturating_sub(flight.times().progress);
                for level in 0..plan.nlevels() {
                    let exposed = flight.wait(level, multipoles);
                    if let Some(wait) = waits.get_mut(level) {
                        *wait = exposed;
                    }
                    operator.host_data(HostData::ReceivedMultipoles {
                        level,
                        exchange: flight.exchange(),
                        multipoles,
                    });
                    if level > 0 {
                        let (start, progress) = (Instant::now(), flight.times().progress);
                        downward_level(plan, operator, level, multipoles, locals, sources, || {
                            flight.test()
                        });
                        downward += start
                            .elapsed()
                            .saturating_sub(flight.times().progress - progress);
                    }
                }
            });
        *overlap = OverlapTimes {
            multipoles: times,
            level_waits: waits,
            coarse_gather: gather,
            upward_global: upward_global_time,
            downward,
            ..*overlap
        };
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
            stage.after().contains(&self.completed),
            "stage {stage:?} called after {:?}; call the stages of one path in order, after \
             reset",
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

//! The empty-box rule: the work of boxes whose subtree holds no target or no source point
//! left out of the lists (Helmholtz CH4.0d1, `docs/helmholtz/tasks/CH4.0d1.md`).
//!
//! The octree is complete, so its leaves include boxes without points, and the lists come
//! from the keys alone ([`interaction_manager`](crate::interaction_manager)): an empty box
//! has every row a full one has. Where a box's subtree holds no target point, its local is
//! read by no point; where it holds no source point, its multipole is zero. The rule leaves
//! that work out, from the points in each box's subtree ([`LeafCounts`], by key, which the
//! operator supplies: [`LeafCounts::exchange_with_boxes`]), while the plan builds its lists
//! ([`Plan::with_rules`](crate::plan::Plan::with_rules) with
//! [`PlanRules::empty_boxes`](crate::plan::PlanRules::empty_boxes)).
//!
//! # What the rule changes
//!
//! For s(b) and t(b), the source and target points in the whole subtree of the box b:
//!
//! - **No target.** A box b with rows and t(b) = 0 gets empty V, X and W rows, and an empty
//!   L2L row (the L2L from its parent into b). Its local stays zero and no point reads it;
//!   its P2M, L2P and near rows stay (they hold no target, so they write nothing a point
//!   reads, or nothing at all). Its X entries are not numbered as ghost leaves for them.
//! - **No source.** A box b with s(b) = 0 leaves every V and W row that names it, and the
//!   M2M row of its parent, local or global. Its multipole is zero: P2M of no point, M2M of
//!   zeros. The X and near rows that name it (P2L and P2P of no source) stay.
//! - **Every box**, `Global` boxes included: the counts of a top-tree key (a coarse block
//!   or an ancestor of one) are summed from every rank's blocks, so every rank, and one
//!   rank, sees the same counts of every box. The upward reduction and the replicated top
//!   follow the global M2M rows and the V and W rows as they are.
//!
//! The near rule ([`near`](crate::near)) sees only the W and X entries this rule keeps.
//!
//! # Exact zeros only
//!
//! Every row keeps its order and is only shortened, so every remaining sum is the sum
//! without the rule less terms that are exact zeros: a zero multipole, or a local that no
//! point reads. A sum s + (±0) is s, except that −0 + (+0) is +0, so the output equals the
//! output without the rule as values (−0 and +0 equal), not always bit for bit. The
//! exchanges follow the rows: the multipole exchange requests the boxes the V and W rows
//! name, so the multipoles of the boxes left out are not sent. The counts are by key and of
//! whole subtrees, so every rank leaves out of its rows what the one-rank plan leaves out
//! of the same rows, and the output on P ranks is the one-rank output with the rule, bit for
//! bit.
//!
//! Without the rule the plan is the plan of [`Plan::new`](crate::plan::Plan::new), bit for
//! bit.

#[cfg(test)]
#[path = "empty_tests.rs"]
mod tests;

use crate::index::BoxIndex;
use crate::near::LeafCounts;
use crate::plan::PlanError;

/// What the empty-box rule left out of a plan's rows on one rank
/// ([`Plan::empty_boxes`](crate::plan::Plan::empty_boxes)); all zero without the rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmptyBoxes {
    /// Boxes with rows whose subtree holds no target point: their V, X, W and L2L rows
    /// emptied.
    pub target_boxes: usize,
    /// V entries of those boxes.
    pub v_without_targets: usize,
    /// X entries of those boxes.
    pub x_without_targets: usize,
    /// W entries of those boxes.
    pub w_without_targets: usize,
    /// L2L rows of those boxes (each one entry).
    pub l2l_without_targets: usize,
    /// V entries of the other boxes that name a box whose subtree holds no source point.
    pub v_without_sources: usize,
    /// W entries of the other boxes that name such a box.
    pub w_without_sources: usize,
    /// M2M entries that name such a box: the local rows', and the global rows' on the
    /// owner of the parent.
    pub m2m_without_sources: usize,
}

/// What the rule knows of one held box: whether its subtree holds a source point and a
/// target point, if counted.
#[derive(Clone, Copy, Debug)]
struct Occupancy {
    sources: Option<bool>,
    targets: Option<bool>,
}

/// The occupancy of every held box, by (level, box index), read from the counts once.
pub(crate) struct Occupancies {
    levels: Vec<Vec<Occupancy>>,
}

impl Occupancies {
    /// The occupancy of every box of `index` from `counts`.
    pub(crate) fn new(index: &BoxIndex, counts: &LeafCounts) -> Self {
        let levels = (0..index.nlevels())
            .map(|level| {
                index
                    .keys(level)
                    .iter()
                    .map(|&key| Occupancy {
                        sources: counts.box_sources(key).map(|n| n > 0),
                        targets: counts.box_targets(key).map(|n| n > 0),
                    })
                    .collect()
            })
            .collect();
        Self { levels }
    }

    /// Whether the subtree of box `i` of `level` holds a source point.
    ///
    /// # Errors
    /// [`PlanError::MissingCount`] if the counts lack its subtree's source count.
    pub(crate) fn has_sources(
        &self,
        index: &BoxIndex,
        level: usize,
        i: usize,
    ) -> Result<bool, PlanError> {
        self.levels[level][i]
            .sources
            .ok_or(PlanError::MissingCount {
                key: index.key(level, i),
            })
    }

    /// Whether the subtree of box `i` of `level` holds a target point.
    ///
    /// # Errors
    /// [`PlanError::MissingCount`] if the counts lack its subtree's target count.
    pub(crate) fn has_targets(
        &self,
        index: &BoxIndex,
        level: usize,
        i: usize,
    ) -> Result<bool, PlanError> {
        self.levels[level][i]
            .targets
            .ok_or(PlanError::MissingCount {
                key: index.key(level, i),
            })
    }
}

//! The near rule: W and X pairs between two leaves turned into near pairs (Phase 6 T12,
//! C6.9; `docs/design/optimisation.md` §8a).
//!
//! For a leaf t, the W-list holds boxes one level finer that do not touch t, and the
//! X-list of a box s the leaves one level coarser that do not touch s; s ∈ W(t) iff
//! t ∈ X(s) ([`interaction_manager`](crate::interaction_manager)). Between a leaf t and a
//! finer **leaf** s the FMM makes two expansions: the multipole of s at the points of t
//! (the W pair, `m2p`) and a local of s from the points of t (the X pair, `p2l`). Both can
//! be evaluated directly instead, as a near pair (`p2p`), which is exact; for leaves with
//! few points that is also cheaper. Which pairs pay is the operator's knowledge, not the
//! plan's: a [`NearRule`] carries a predicate the operator supplies and the point counts
//! it reads ([`LeafCounts`]), and the plan applies it while it builds the lists
//! ([`Plan::with_near_rule`](crate::plan::Plan::with_near_rule)).
//!
//! # What the rule changes
//!
//! - **W.** For each W entry s of a local leaf t whose box s is a leaf, the predicate
//!   sees [`LeafPair::W`] with the source count of s. If it holds, t's near row gains s
//!   and t's W row loses it; a ghost leaf s is then named in the leaf numbering as a
//!   U-list leaf is, so the source exchange delivers its points.
//! - **X.** For each X entry t of a local leaf s, the predicate sees [`LeafPair::X`] with
//!   the target count of s. If it holds, s's near row gains t and s's X row loses it.
//! - Nothing else: an interior W box keeps its M2P (its points lie in leaves that no
//!   list names), the X entries of an interior box keep their P2L, and the two directions
//!   of a pair are decided separately, each on the rank of its target. A pair the rule
//!   keeps stays in its W and X rows, so their duality holds for every pair not converted.
//! - The near rows keep their (level, key) order (P1), so the sums of P2P are those of
//!   one rank, and the rule is local: [`Plan::with_near_rule`](crate::plan::Plan::with_near_rule)
//!   adds no collective. Given the same counts and predicate, every rank makes the rows
//!   the one-rank plan has for its boxes.
//!
//! Without a rule the plan is the plan of [`Plan::new`](crate::plan::Plan::new), bit for
//! bit.
//!
//! # The counts
//!
//! The rule reads the source count of every leaf a W row names (local or ghost) and the
//! target count of every local leaf. [`LeafCounts::exchange`] gathers them from the
//! local leaves' counts and the plan without the rule: the W-row leaves of another rank
//! are not always among the ghost leaves the source exchange delivers (U and X name
//! those), so they are fetched from their owners. Counts are by key, so the same leaf
//! has the same count on every rank.

#[cfg(test)]
#[path = "near_tests.rs"]
mod tests;

use std::collections::HashMap;

use mpi::traits::CommunicatorCollectives;
use nd_octree::MortonKey;

use super::exchange::{ExchangeError, ghost_source_counts};
use super::plan::Plan;
use crate::interaction_manager::is_leaf;

/// A W or X pair between two leaves that a [`NearRule`] may turn into a near pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafPair {
    /// The W entry `source`, a leaf one level finer, of the leaf `target`: the multipole
    /// of `source` at the target points of `target` (M2P), or, as a near pair, the
    /// sources of `source` at those points (P2P).
    W {
        /// The coarser leaf, whose W row names `source`.
        target: MortonKey,
        /// The finer leaf.
        source: MortonKey,
        /// The source points of `source`.
        sources: usize,
    },
    /// The X entry `source`, a leaf one level coarser, of the leaf `target`: a local of
    /// `target` from the sources of `source` (P2L), or, as a near pair, those sources at
    /// the target points of `target` (P2P).
    X {
        /// The finer leaf, whose X row names `source`.
        target: MortonKey,
        /// The coarser leaf.
        source: MortonKey,
        /// The target points of `target`.
        targets: usize,
    },
}

/// The point counts a [`NearRule`] reads, by leaf key: the source count of every leaf a
/// W row of a local leaf names and of every local leaf, and the target count of every
/// local leaf.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeafCounts {
    sources: HashMap<MortonKey, usize>,
    targets: HashMap<MortonKey, usize>,
}

impl LeafCounts {
    /// No counts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source count of the leaf `key`.
    pub fn set_sources(&mut self, key: MortonKey, count: usize) {
        self.sources.insert(key, count);
    }

    /// Set the target count of the leaf `key`.
    pub fn set_targets(&mut self, key: MortonKey, count: usize) {
        self.targets.insert(key, count);
    }

    /// The source count of the leaf `key`, if known.
    pub fn sources(&self, key: MortonKey) -> Option<usize> {
        self.sources.get(&key).copied()
    }

    /// The target count of the leaf `key`, if known.
    pub fn targets(&self, key: MortonKey) -> Option<usize> {
        self.targets.get(&key).copied()
    }

    /// The counts a near rule reads on the leaves of `plan`, a plan of the same octree
    /// built without a rule: `sources[j]` and `targets[j]` points on local leaf j, and the
    /// source counts of the ghost leaves the W rows name, from their owners.
    ///
    /// # Collective operation
    /// On `comm`, the communicator of the octree of `plan`: the build of a ghost
    /// communicator of the W rows' ghost leaves (rlst's, which moves the counts as chunk
    /// sizes and no value) and two all-reduces.
    ///
    /// # Errors
    /// Returned on every rank if, on any rank, `sources` or `targets` does not have one
    /// entry per local leaf, or a rank is asked for a key that is not one of its local
    /// leaves.
    pub fn exchange<C: CommunicatorCollectives>(
        plan: &Plan,
        comm: &C,
        sources: &[usize],
        targets: &[usize],
    ) -> Result<Self, ExchangeError> {
        let index = plan.index();
        let leaves = index.leaves();
        let nlocal = leaves.nlocal();
        // The exchange checks the source counts; both are agreed there.
        let checked = if targets.len() == nlocal {
            Ok(())
        } else {
            Err(ExchangeError::CountsLength {
                expected: nlocal,
                actual: targets.len(),
            })
        };
        // The ghost leaves of the W rows, each once, with their owners.
        let mut ghosts: Vec<(MortonKey, usize)> = Vec::new();
        for level in 0..index.nlevels().saturating_sub(1) {
            for &s in plan.level(level).w().entries() {
                let kind = index.kind(level + 1, s as usize);
                if let (true, Some(owner)) = (is_leaf(kind), kind.ghost_rank()) {
                    ghosts.push((index.key(level + 1, s as usize), owner));
                }
            }
        }
        ghosts.sort_unstable();
        ghosts.dedup();
        let received = ghost_source_counts(plan, comm, checked, sources, &ghosts)?;

        let mut counts = Self::new();
        for j in 0..nlocal {
            counts.set_sources(leaves.key(j), sources[j]);
            counts.set_targets(leaves.key(j), targets[j]);
        }
        for (key, count) in received {
            counts.set_sources(key, count);
        }
        Ok(counts)
    }
}

/// A rule that turns W and X pairs between two leaves into near pairs: a predicate the
/// operator supplies and the counts it reads (the [module documentation](self)).
///
/// The plan calls the predicate once per W entry of a local leaf whose box is a leaf
/// and once per X entry of a local leaf, with that pair's count from the counts; a pair
/// whose count is missing is an error of the plan
/// ([`PlanError::MissingCount`](crate::plan::PlanError::MissingCount)).
pub struct NearRule<'a> {
    counts: &'a LeafCounts,
    predicate: &'a dyn Fn(LeafPair) -> bool,
}

impl<'a> NearRule<'a> {
    /// The rule of `predicate` over `counts`: a pair becomes a near pair where the
    /// predicate holds.
    pub fn new(counts: &'a LeafCounts, predicate: &'a dyn Fn(LeafPair) -> bool) -> Self {
        Self { counts, predicate }
    }

    /// The counts the rule reads.
    pub fn counts(&self) -> &LeafCounts {
        self.counts
    }

    /// Whether `pair` becomes a near pair.
    pub fn converts(&self, pair: LeafPair) -> bool {
        (self.predicate)(pair)
    }
}

impl std::fmt::Debug for NearRule<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NearRule")
            .field("counts", &self.counts)
            .finish_non_exhaustive()
    }
}

/// The W and X pairs between leaves a plan has, and those its near rule converted, on
/// one rank ([`Plan::near_pairs`](crate::plan::Plan::near_pairs)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NearPairs {
    /// W entries of local leaves whose box is a leaf, before the rule.
    pub w_leaf_pairs: usize,
    /// Of those, the W pairs the rule made near pairs.
    pub w_converted: usize,
    /// X entries of local leaves, before the rule.
    pub x_leaf_pairs: usize,
    /// Of those, the X pairs the rule made near pairs.
    pub x_converted: usize,
}

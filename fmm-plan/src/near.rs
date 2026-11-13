//! The near rule: W and X pairs between two leaves turned into near pairs (Phase 6 T12,
//! C6.9; `docs/design/optimisation.md` §8a), and optionally V pairs (Helmholtz CH4.0d2).
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
//! - **V** (Helmholtz CH4.0d2; only with [`NearRule::with_v_pairs`]). For each V entry s
//!   of a local leaf t whose box s is a leaf (of t's level, not adjacent to t), the V
//!   predicate sees [`VPair`] with the source count of s and the target count of t. If it
//!   holds, t's near row gains s and t's V row loses it (the M2L from s's multipole to t's
//!   local, which only t's L2P reads); a ghost leaf s is named in the leaf numbering, so
//!   the source exchange delivers its points. The source exchange's halo then reaches the
//!   V leaves too, up to three leaf widths from t's boundary.
//! - Nothing else: an interior W or V box keeps its M2P or M2L (its points lie in leaves
//!   that no list names), the X entries of an interior box keep their P2L, and the two directions
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
//! has the same count on every rank. With V pairs the rule also reads the source count of
//! every leaf a V row names: [`LeafCounts::exchange_with_boxes`] holds that of every held
//! ghost leaf, those of the V rows included ([`LeafCounts::exchange`] does not).

#[cfg(test)]
#[path = "near_tests.rs"]
mod tests;

use std::collections::HashMap;

use mpi::traits::CommunicatorCollectives;
use nd_octree::{MortonKey, Octree, morton};

use nd_octree::octree::KeyType;

use super::exchange::{ExchangeError, gather_all, ghost_key_counts, ghost_source_counts};
use super::index::BoxIndex;
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
/// local leaf. And those the empty-box rule reads ([`empty`](crate::empty)), by box key:
/// the source and target points in the subtree of every local box, and the source points
/// in the subtree of every ghost box a V or W row names.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeafCounts {
    sources: HashMap<MortonKey, usize>,
    targets: HashMap<MortonKey, usize>,
    box_sources: HashMap<MortonKey, usize>,
    box_targets: HashMap<MortonKey, usize>,
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

    /// Set the source points in the subtree of the box `key` (for a leaf, its own).
    pub fn set_box_sources(&mut self, key: MortonKey, count: usize) {
        self.box_sources.insert(key, count);
    }

    /// Set the target points in the subtree of the box `key` (for a leaf, its own).
    pub fn set_box_targets(&mut self, key: MortonKey, count: usize) {
        self.box_targets.insert(key, count);
    }

    /// The source points in the subtree of the box `key`, if known.
    pub fn box_sources(&self, key: MortonKey) -> Option<usize> {
        self.box_sources.get(&key).copied()
    }

    /// The target points in the subtree of the box `key`, if known.
    pub fn box_targets(&self, key: MortonKey) -> Option<usize> {
        self.box_targets.get(&key).copied()
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

    /// The counts of [`exchange`](Self::exchange), and those the empty-box rule reads
    /// ([`empty`](crate::empty)), on `index`, the box index of `octree`
    /// ([`BoxIndex::of_octree`]; Helmholtz CH4.0d1b: no plan needed): the source and target
    /// points in the subtree of every local box (its local leaves' counts summed up the tree;
    /// every descendant of a local box is local), of every top-tree key it holds (the coarse
    /// blocks' counts of every rank, summed up the replicated top tree), and the source points
    /// in the subtree of every other held ghost box, and of every held ghost leaf, from its
    /// owner. `sources[j]` and `targets[j]` are the points on local leaf j.
    ///
    /// Every held box thus has the counts of its whole subtree (a ghost its source count),
    /// the same on every rank and on one rank: those of every box a plan's rows name, which
    /// any plan of the octree reads ([`Plan::with_rules_on`](crate::plan::Plan::with_rules_on)
    /// with the empty-box rule, the near rule or both). Every held ghost below another rank's
    /// coarse blocks is a local box of its owner, which counts it; the counts move as the
    /// chunk sizes of one ghost communicator, a count per held ghost box, whether or not a
    /// row names it.
    ///
    /// # Collective operation
    /// On the octree's communicator: an all-gather and an all-gather-v of the coarse blocks'
    /// counts, then the build of a ghost communicator (rlst's, which moves the counts as
    /// chunk sizes and no value) and two all-reduces, as [`exchange`](Self::exchange).
    ///
    /// # Errors
    /// Returned on every rank if, on any rank, `sources` or `targets` does not have one
    /// entry per local leaf of `index` ([`ExchangeError::CountsLength`]), a rank is asked for
    /// a key that is not one of its local boxes ([`ExchangeError::UnheldKey`]), or the
    /// blocks' counts outgrow MPI's counts ([`ExchangeError::Overflow`]).
    pub fn exchange_with_boxes<C: CommunicatorCollectives>(
        octree: &Octree<'_, C>,
        index: &BoxIndex,
        sources: &[usize],
        targets: &[usize],
    ) -> Result<Self, ExchangeError> {
        let comm = octree.comm();
        let map = octree.partition_map();
        let is_top = |key: MortonKey| map.top_kind(key).is_some();
        let leaves = index.leaves();
        let nlocal = leaves.nlocal();
        let checked = [sources.len(), targets.len()]
            .into_iter()
            .find(|&len| len != nlocal)
            .map_or(Ok(()), |actual| {
                Err(ExchangeError::CountsLength {
                    expected: nlocal,
                    actual,
                })
            });
        let is_local = |level: usize, i: usize| {
            matches!(
                index.kind(level, i),
                KeyType::LocalLeaf | KeyType::LocalInterior
            )
        };

        // The subtree counts of every local box, (sources, targets) by (level, box index),
        // deepest level first. A `Global` box sums its held children, which are its whole
        // subtree only where it is a coarse block (the root on one rank): the counts of the
        // other `Global` boxes come from the blocks below.
        let nlevels = index.nlevels();
        let mut subtree: Vec<Vec<(usize, usize)>> = (0..nlevels)
            .map(|level| vec![(0, 0); index.len(level)])
            .collect();
        if checked.is_ok() {
            for level in (0..nlevels).rev() {
                for i in (0..index.len(level)).filter(|&i| !index.kind(level, i).is_ghost()) {
                    subtree[level][i] = if let Some(j) = index.box_leaf(level, i) {
                        (sources[j as usize], targets[j as usize])
                    } else if let Some(first) = index.first_child(level, i) {
                        let first = first as usize;
                        subtree[level + 1][first..first + 8]
                            .iter()
                            .fold((0, 0), |(s, t), &(cs, ct)| (s + cs, t + ct))
                    } else {
                        (0, 0)
                    };
                }
            }
        }

        // The coarse blocks' counts of every rank, summed into every ancestor: the counts
        // of every top-tree key.
        let own: Vec<u64> = map.blocks()[map.rank_blocks(map.rank())]
            .iter()
            .flat_map(|&block| {
                let (s, t) = index
                    .find(block)
                    .map_or((0, 0), |(level, i)| subtree[level][i as usize]);
                [block, s as u64, t as u64]
            })
            .collect();
        let blocks = gather_all(comm, &own)?;
        let mut top: HashMap<MortonKey, (usize, usize)> = HashMap::new();
        for &[block, s, t] in blocks.as_chunks::<3>().0 {
            let mut key = Some(block);
            while let Some(k) = key {
                let entry = top.entry(k).or_default();
                entry.0 += s as usize;
                entry.1 += t as usize;
                key = morton::parent(k);
            }
        }

        // Every held ghost box that is not a top-tree key, and every held ghost leaf (the
        // near rule's), each once, with its owner.
        let mut ghosts: Vec<(MortonKey, usize)> = Vec::new();
        for level in 0..nlevels {
            for (&key, &kind) in index.keys(level).iter().zip(index.kinds(level)) {
                if let Some(owner) = kind.ghost_rank()
                    && (is_leaf(kind) || !is_top(key))
                {
                    ghosts.push((key, owner));
                }
            }
        }
        ghosts.sort_unstable();
        ghosts.dedup();
        let owned = |key: MortonKey| {
            let (level, i) = index.find(key)?;
            is_local(level, i as usize).then(|| subtree[level][i as usize].0)
        };
        let received = ghost_key_counts(comm, checked, &ghosts, &owned)?;

        let mut counts = Self::new();
        for j in 0..nlocal {
            counts.set_sources(leaves.key(j), sources[j]);
            counts.set_targets(leaves.key(j), targets[j]);
        }
        for (level, level_counts) in subtree.iter().enumerate() {
            for (i, &(s, t)) in level_counts.iter().enumerate() {
                if is_local(level, i) {
                    counts.set_box_sources(index.key(level, i), s);
                    counts.set_box_targets(index.key(level, i), t);
                }
            }
        }
        for level in 0..nlevels {
            for &key in index.keys(level).iter().filter(|&&key| is_top(key)) {
                let (s, t) = top.get(&key).copied().unwrap_or_default();
                counts.set_box_sources(key, s);
                counts.set_box_targets(key, t);
            }
        }
        for (key, count) in received {
            if index
                .find(key)
                .is_some_and(|(level, j)| is_leaf(index.kind(level, j as usize)))
            {
                counts.set_sources(key, count);
            }
            counts.set_box_sources(key, count);
        }
        Ok(counts)
    }
}

/// A V pair between two leaves of one level that a [`NearRule`] with a V predicate
/// ([`NearRule::with_v_pairs`]; Helmholtz CH4.0d2) may turn into a near pair: the multipole
/// of `source` translated to a local of `target` (M2L), or, as a near pair, the sources of
/// `source` at the target points of `target` (P2P). The level is the keys'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VPair {
    /// The local leaf whose V row names `source`.
    pub target: MortonKey,
    /// The leaf of the V entry, local or a ghost.
    pub source: MortonKey,
    /// The source points of `source`.
    pub sources: usize,
    /// The target points of `target`.
    pub targets: usize,
}

/// A rule that turns W and X pairs between two leaves into near pairs, and with
/// [`with_v_pairs`](Self::with_v_pairs) V pairs between two leaves too: the predicates the
/// operator supplies and the counts they read (the [module documentation](self)).
///
/// The plan calls the predicate once per W entry of a local leaf whose box is a leaf
/// and once per X entry of a local leaf, with that pair's count from the counts, and the
/// V predicate, if any, once per V entry of a local leaf whose box is a leaf, with both
/// counts; a pair whose count is missing is an error of the plan
/// ([`PlanError::MissingCount`](crate::plan::PlanError::MissingCount)).
pub struct NearRule<'a> {
    counts: &'a LeafCounts,
    predicate: &'a dyn Fn(LeafPair) -> bool,
    v_predicate: Option<&'a dyn Fn(VPair) -> bool>,
}

impl<'a> NearRule<'a> {
    /// The rule of `predicate` over `counts`: a W or X pair becomes a near pair where the
    /// predicate holds. No V pair is offered.
    pub fn new(counts: &'a LeafCounts, predicate: &'a dyn Fn(LeafPair) -> bool) -> Self {
        Self {
            counts,
            predicate,
            v_predicate: None,
        }
    }

    /// The rule that also offers every V pair between two leaves to `predicate` (Helmholtz
    /// CH4.0d2): a V entry s of a local leaf t, s a leaf (local or a ghost), becomes a near
    /// pair where it holds. t's V row loses s and its near row gains it, in (level, key)
    /// order; a ghost s is named in the leaf numbering, so the source exchange delivers its
    /// points. Interior V entries are never offered (their points lie in leaves no list
    /// names).
    pub fn with_v_pairs(self, predicate: &'a dyn Fn(VPair) -> bool) -> Self {
        Self {
            v_predicate: Some(predicate),
            ..self
        }
    }

    /// The counts the rule reads.
    pub fn counts(&self) -> &LeafCounts {
        self.counts
    }

    /// Whether `pair` becomes a near pair.
    pub fn converts(&self, pair: LeafPair) -> bool {
        (self.predicate)(pair)
    }

    /// Whether the rule offers V pairs ([`with_v_pairs`](Self::with_v_pairs)).
    pub fn offers_v_pairs(&self) -> bool {
        self.v_predicate.is_some()
    }

    /// Whether the V pair `pair` becomes a near pair; never without a V predicate.
    pub fn converts_v(&self, pair: VPair) -> bool {
        self.v_predicate.is_some_and(|predicate| predicate(pair))
    }
}

impl std::fmt::Debug for NearRule<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NearRule")
            .field("counts", &self.counts)
            .field("v_pairs", &self.offers_v_pairs())
            .finish_non_exhaustive()
    }
}

/// The W, X and V pairs between leaves a plan has, and those its near rule converted, on
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
    /// V entries of local leaves whose box is a leaf, before the rule, where the rule offers
    /// V pairs ([`NearRule::with_v_pairs`]); zero otherwise.
    pub v_leaf_pairs: usize,
    /// Of those, the V pairs the rule made near pairs.
    pub v_converted: usize,
}

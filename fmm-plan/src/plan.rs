//! The plan of an FMM: the box index and the interaction lists of every level.
//!
//! [`Plan::new`] numbers every box the rank holds ([`BoxIndex`]) and builds the views of
//! [`LevelLists`] for every level, from the octree's key classification alone. The
//! lists come from the per-key rule of
//! [`interaction_manager`](crate::interaction_manager) and are translated to indices at
//! build time (design §4.1, `docs/design/fmm-plan-redesign.md`).
//!
//! # Held keys
//!
//! Since Phase 5S T5 the plan holds **H**, the octree's locally essential set
//! ([`Octree::all_keys`](nd_octree::Octree::all_keys); `docs/design/scale-out.md` §3.1,
//! §3.4): with S the own keys and the shared ancestors (the ancestors of this rank's
//! coarse blocks, `Global`), S, every same-level neighbour of a key of S and the children
//! of the interior ones, each with the octree's kind (a top-tree key that is not a shared
//! ancestor is a ghost of its owner). H does not grow at fixed N/P. Phase 5S T4 also held
//! T, every coarse block and every ancestor of one, for the coarse gather and the global
//! pass that T5 replaced by the owned top tree; the replicated top
//! ([`TopExchange`](super::exchange::TopExchange)) needs no slot outside H either, since
//! no row names a key outside H.
//!
//! # Owners
//!
//! From the replicated [`PartitionMap`] the plan keeps, without communication, the owner
//! of every held *top-tree key* (a coarse block or an ancestor of one): the rank of its
//! first descendant block (`docs/design/scale-out.md` §4.1,
//! [`PartitionMap::owner`]). The owner of a box is the one rank that computes its
//! multipole ([`Plan::owner`]): a local box's is this rank, a ghost's is its
//! [`KeyType::ghost_rank`], and a `Global` box's is the owner of the map, which may be
//! another rank. Only the owner of a `Global` box has its M2M row of the global pass:
//! its children are local boxes, `Global` boxes it owns as well, or ghosts of their
//! owners, which the upward reduction delivers
//! ([`ReductionExchange`](super::exchange::ReductionExchange)). On one rank every box is
//! owned, and the root (`Global` unless it is a leaf) keeps its row.
//!
//! # Guarantees
//!
//! - Every non-ghost box has rows in every view that applies to it; ghost boxes have
//!   empty rows. See the [`lists`](super::lists) module for the guarantees of the views
//!   and [`index`](super::index) for those of the numbering.
//! - **Only the `Global` locals this rank needs** (P2, `docs/design/distributed-fmm.md`
//!   §3.6). A rank's locals descend only from the `Global` boxes above its own coarse
//!   blocks (its shared ancestors), so a `Global` box that is not an ancestor of one of
//!   them has empty V, X and L2L rows, and its local stays zero. Its lists are not even
//!   resolved: the plan does not hold its neighbourhood (Phase 5S T4). Since Phase 5S T5
//!   every `Global` box of [`Plan::new`] is a shared ancestor, so P2 drops no row there;
//!   it still applies to a key map that types other keys `Global`
//!   ([`Plan::from_key_types`]). Every row that stays is the row it would be without P2,
//!   so every value of a needed box is the same, bit for bit.
//! - **Global M2M rows on the owner only.** A `Global` box has its eight children in its
//!   M2M row of the global pass on its [owner](#owners) and an empty row elsewhere. Its
//!   other sharers read its multipole, where a row names it, from the owner
//!   ([`MultipoleExchange`](super::exchange::MultipoleExchange)) or from the replicated
//!   top.
//! - **Every key a list or a parent-child step names is held**: every list entry of a
//!   box with rows, the parent of such a box, every child of a non-ghost interior box
//!   (in particular every child of a `Global` box, a top-tree key) and every own coarse
//!   block. [`Plan::new`] checks this and fails on every rank otherwise. The keys the
//!   exchanges name are among them.
//! - **Phase 5's lists.** Every list entry, as a key, and the order of every row are
//!   those of the Phase 5 plan over the Phase 5 key set: the rows depend only on the
//!   tree and the rule, P1 orders by (level, key), and the kinds of the keys the rows
//!   name are Phase 5's, but for a top-tree key that is not a shared ancestor, which is a
//!   ghost of its owner since Phase 5S T5 (`docs/design/scale-out.md` §3.6, §4). Only the
//!   box indices change, because fewer keys are held.
//! - **The near rule** (Phase 6 T12, C6.9; [`Plan::with_near_rule`], the
//!   [`near`](crate::near) module): optional, applied locally while the rows are built.
//!   The W entries of a leaf that are leaves and the X entries of a leaf that the rule
//!   converts move to the leaf's near row, in its (level, key) order; a converted ghost W
//!   entry is numbered as a ghost leaf. Every other row is the row without the rule, and
//!   [`Plan::new`] is the plan without it, bit for bit.
//! - **The empty-box rule** (Helmholtz CH4.0d1; [`Plan::with_rules`] with
//!   [`PlanRules::empty_boxes`], the [`empty`](crate::empty) module): optional, applied
//!   locally while the rows are built. A box whose subtree holds no target loses its V, X, W
//!   and L2L rows; a box whose subtree holds no source leaves the V, W and M2M rows that
//!   name it; `Global` boxes too, by the counts of whole subtrees. Rows are only shortened,
//!   never reordered; [`Plan::new`] is the plan without the rule, bit for bit. Its counts
//!   need the local leaves' points, so a caller builds the octree's index first
//!   ([`BoxIndex::of_octree`]), routes the points and counts on it
//!   ([`LeafCounts::exchange_with_boxes`]), and builds the plan once on the same index
//!   ([`Plan::with_rules_on`]; Helmholtz CH4.0d1b).
//! - The plan owns its arrays and borrows nothing from the octree. A new tree needs a new
//!   plan: box indices, unlike keys, change when the tree changes.
//!
//! # Cost
//!
//! The lists are built locally, in O(E log K) for E list entries and K held keys, and the
//! owners in O(K_T (log B + log P)) for K_T held top-tree keys, B coarse blocks and P
//! ranks. [`Plan::new`] adds two all-reduces.

#[cfg(test)]
#[path = "plan_tests.rs"]
pub(crate) mod tests;

use std::{collections::HashMap, error::Error, fmt};

use mpi::{collective::SystemOperation, traits::CommunicatorCollectives};
use nd_octree::{
    MortonKey, Octree,
    constants::DEEPEST_LEVEL,
    morton,
    octree::{KeyType, PartitionMap},
};

use super::empty::{EmptyBoxes, Occupancies};
use super::index::BoxIndex;
use super::lists::{
    CsrBuilder, GroupedBuilder, LevelLists, NOCTANTS, NOFFSETS, Overflow, offset, offset_index,
};
use super::near::{LeafCounts, LeafPair, NearPairs, NearRule, VPair};
use crate::interaction_manager::{is_interior, is_leaf, key_lists, valid_neighbours};

/// The box index and the interaction lists of an octree on one rank.
///
/// See the [module documentation](self).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    index: BoxIndex,
    levels: Vec<LevelLists>,
    coarse_blocks: Vec<MortonKey>,
    /// The rank of the partition map.
    rank: usize,
    /// `[level]`: (box index, owner) of every held top-tree key, ascending.
    top: Vec<Vec<(u32, u32)>>,
    /// The deepest level of a coarse block of any rank, capped at the deepest level.
    deepest_block_level: usize,
    /// The W and X pairs between leaves, and those the near rule converted.
    near_pairs: NearPairs,
    /// What the empty-box rule left out.
    empty_boxes: EmptyBoxes,
}

/// The optional rules a plan applies while it builds its rows
/// ([`Plan::with_rules`]); the default applies neither, the plan of [`Plan::new`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanRules<'a> {
    /// The near rule (Phase 6 T12, C6.9; the [`near`](crate::near) module).
    pub near: Option<&'a NearRule<'a>>,
    /// The empty-box rule from these counts (Helmholtz CH4.0d1; the
    /// [`empty`](crate::empty) module): the subtree counts of every box with rows and of
    /// every box a V, W or M2M row names; [`LeafCounts::exchange_with_boxes`] gives those
    /// of every held box.
    pub empty_boxes: Option<&'a LeafCounts>,
}

/// The reasons a [`Plan`] cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The octree was built without the ghost-children layer
    /// ([`OctreeOptions::with_ghost_children`](nd_octree::OctreeOptions::with_ghost_children)).
    MissingGhostChildren,
    /// A key lies on a level at or beyond the level count.
    LevelOutOfRange {
        /// The key.
        key: MortonKey,
        /// The level count.
        nlevels: usize,
    },
    /// A level holds 2^32 boxes or more, which `u32` indices cannot number.
    TooManyBoxes {
        /// The level.
        level: usize,
        /// The number of boxes held on it.
        count: usize,
    },
    /// A view of a level has 2^32 entries or more, which its `u32` offsets cannot hold.
    TooManyEntries {
        /// The level.
        level: usize,
    },
    /// The key map does not describe a complete, 2:1 balanced tree with its one-cell
    /// same-level halo, which the list rule relies on.
    MalformedTree {
        /// The key at which the defect was found.
        key: MortonKey,
        /// What is wrong.
        reason: &'static str,
    },
    /// A list entry of a non-ghost box is not a key of the map; usually the tree lacks the
    /// ghost-children layer.
    UnheldListEntry {
        /// The box whose list names the entry.
        key: MortonKey,
        /// The entry.
        entry: MortonKey,
    },
    /// A non-ghost interior box does not hold all eight children.
    MissingChild {
        /// The interior box.
        key: MortonKey,
    },
    /// A coarse block of this rank is not a non-ghost key of the map.
    CoarseBlock {
        /// The coarse block.
        key: MortonKey,
    },
    /// The counts of a rule lack a count it reads: for the near rule the source count of a
    /// leaf a W row names, or the target count of a local leaf with an X row
    /// ([`near`](crate::near)); for the empty-box rule a subtree count of a box with rows or
    /// of a box a V, W or M2M row names ([`empty`](crate::empty)).
    MissingCount {
        /// The leaf or box.
        key: MortonKey,
    },
    /// The box index given to [`Plan::with_rules_on`] is not the index of the octree's
    /// held keys: built from another octree, or with another level count.
    IndexMismatch,
    /// Construction failed on another rank.
    OtherRank,
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingGhostChildren => {
                write!(f, "the octree was built without the ghost-children layer")
            }
            Self::LevelOutOfRange { key, nlevels } => {
                write!(f, "key {key} lies beyond the {nlevels} levels of the plan")
            }
            Self::TooManyBoxes { level, count } => {
                write!(
                    f,
                    "level {level} holds {count} boxes, more than u32 can index"
                )
            }
            Self::TooManyEntries { level } => {
                write!(
                    f,
                    "a view of level {level} has more entries than u32 can offset"
                )
            }
            Self::MalformedTree { key, reason } => {
                write!(f, "malformed tree at key {key}: {reason}")
            }
            Self::UnheldListEntry { key, entry } => {
                write!(
                    f,
                    "list entry {entry} of key {key} is not a key of the tree"
                )
            }
            Self::MissingChild { key } => {
                write!(f, "interior key {key} does not hold all eight children")
            }
            Self::CoarseBlock { key } => {
                write!(f, "coarse block {key} is not a non-ghost key of the tree")
            }
            Self::MissingCount { key } => {
                write!(f, "the plan's rule has no point count of key {key}")
            }
            Self::IndexMismatch => {
                write!(f, "the box index is not the index of the octree's keys")
            }
            Self::OtherRank => write!(f, "building the plan failed on another rank"),
        }
    }
}

impl Error for PlanError {}

impl Plan {
    /// Build the box index and the interaction lists of `octree`.
    ///
    /// The plan holds the octree's locally essential keys H (see
    /// [Held keys](self#held-keys)), and the owners of its top-tree keys from the
    /// octree's partition map ([Owners](self#owners)).
    ///
    /// # Collective operation
    /// One all-reduce for the global level count and one for the validity check, in
    /// that order, on every rank. The held keys and the lists are built without
    /// communication.
    ///
    /// # Errors
    /// Returned on every rank if, on any rank, the tree lacks the ghost-children layer,
    /// a list entry, a parent or a child of a non-ghost interior box (in particular of a
    /// `Global` box) is not held, or a level or view outgrows `u32` indices. The rank
    /// that found the defect returns it; the other ranks return [`PlanError::OtherRank`].
    pub fn new<C: CommunicatorCollectives>(octree: &Octree<'_, C>) -> Result<Self, PlanError> {
        Self::with_rules(octree, &PlanRules::default())
    }

    /// Build the plan of `octree` as [`new`](Self::new) does, with the W and X pairs
    /// between leaves that `rule` names turned into near pairs (Phase 6 T12, C6.9; the
    /// [`near`](crate::near) module).
    ///
    /// # Collective operation
    /// Those of [`new`](Self::new): the rule is applied locally.
    ///
    /// # Errors
    /// Those of [`new`](Self::new), and [`PlanError::MissingCount`] where the rule's counts
    /// lack a count it reads, on every rank alike.
    pub fn with_near_rule<C: CommunicatorCollectives>(
        octree: &Octree<'_, C>,
        rule: &NearRule<'_>,
    ) -> Result<Self, PlanError> {
        Self::with_rules(
            octree,
            &PlanRules {
                near: Some(rule),
                ..PlanRules::default()
            },
        )
    }

    /// Build the plan of `octree` as [`new`](Self::new) does, with the rules of `rules`:
    /// the near rule ([`near`](crate::near)) and the empty-box rule
    /// ([`empty`](crate::empty)), each optional.
    ///
    /// # Collective operation
    /// Those of [`new`](Self::new): the rules are applied locally.
    ///
    /// # Errors
    /// Those of [`new`](Self::new), and [`PlanError::MissingCount`] where a rule's counts
    /// lack a count it reads: on the rank that found it, [`PlanError::OtherRank`] on the
    /// others.
    pub fn with_rules<C: CommunicatorCollectives>(
        octree: &Octree<'_, C>,
        rules: &PlanRules<'_>,
    ) -> Result<Self, PlanError> {
        let nlevels = octree.global_max_level() + 1;
        let map = octree.partition_map();
        debug_assert_eq!(
            &map.blocks()[map.rank_blocks(map.rank())],
            &octree.coarse_tree_leafs()[..],
            "the partition map names this rank's coarse blocks"
        );
        let local = if octree.options().ghost_children() {
            Self::from_key_types_with_rules(octree.all_keys(), nlevels, map, rules)
        } else {
            Err(PlanError::MissingGhostChildren)
        };
        agree(octree, local)
    }

    /// Build the plan of `octree` as [`with_rules`](Self::with_rules) does, on `index`, the
    /// octree's box index from [`BoxIndex::of_octree`], instead of numbering the keys again
    /// (Helmholtz CH4.0d1b): a caller that needs the index before the plan (to route points
    /// to the local leaves and count them for [`PlanRules::empty_boxes`]) builds the plan
    /// once. The plan is [`with_rules`](Self::with_rules)' bit for bit.
    ///
    /// # Collective operation
    /// Those of [`new`](Self::new): the rules are applied locally.
    ///
    /// # Errors
    /// Those of [`with_rules`](Self::with_rules), and [`PlanError::IndexMismatch`] if
    /// `index` is not the index of the octree's held keys on its global level count (or
    /// has ghost leaves numbered): on the rank that found it, [`PlanError::OtherRank`] on
    /// the others.
    pub fn with_rules_on<C: CommunicatorCollectives>(
        octree: &Octree<'_, C>,
        index: BoxIndex,
        rules: &PlanRules<'_>,
    ) -> Result<Self, PlanError> {
        let nlevels = octree.global_max_level() + 1;
        let map = octree.partition_map();
        debug_assert_eq!(
            &map.blocks()[map.rank_blocks(map.rank())],
            &octree.coarse_tree_leafs()[..],
            "the partition map names this rank's coarse blocks"
        );
        let local = if !octree.options().ghost_children() {
            Err(PlanError::MissingGhostChildren)
        } else if !index.is_index_of(octree.all_keys(), nlevels) {
            Err(PlanError::IndexMismatch)
        } else {
            Self::from_index_with_rules(index, octree.all_keys(), map, rules)
        };
        agree(octree, local)
    }

    /// Build the plan from a key classification and a partition map; local, for tests on
    /// hand-built maps.
    ///
    /// `all_keys` must describe a complete, 2:1 (26-neighbour) balanced tree together with
    /// its one-cell same-level halo, as
    /// [`Octree::all_keys`](nd_octree::Octree::all_keys) does, on the levels
    /// `0..nlevels`. This rank's coarse blocks are those of `map`'s rank
    /// ([`PartitionMap::rank_blocks`]); `map` also gives the owners of the top-tree keys
    /// ([Owners](self#owners)).
    ///
    /// # Errors
    /// Returns the first defect found, scanning levels and keys in ascending order, so the
    /// error does not depend on the iteration order of `all_keys`. Never panics on a
    /// malformed map.
    pub fn from_key_types(
        all_keys: &HashMap<MortonKey, KeyType>,
        nlevels: usize,
        map: &PartitionMap,
    ) -> Result<Self, PlanError> {
        Self::from_key_types_with_rules(all_keys, nlevels, map, &PlanRules::default())
    }

    /// [`from_key_types`](Self::from_key_types) with the W and X pairs between leaves that
    /// `rule` names turned into near pairs (the [`near`](crate::near) module); local, for
    /// tests on hand-built maps.
    ///
    /// # Errors
    /// Those of [`from_key_types`](Self::from_key_types), and
    /// [`PlanError::MissingCount`] where the rule's counts lack a count it reads.
    pub fn from_key_types_with_near_rule(
        all_keys: &HashMap<MortonKey, KeyType>,
        nlevels: usize,
        map: &PartitionMap,
        rule: &NearRule<'_>,
    ) -> Result<Self, PlanError> {
        Self::from_key_types_with_rules(
            all_keys,
            nlevels,
            map,
            &PlanRules {
                near: Some(rule),
                ..PlanRules::default()
            },
        )
    }

    /// [`from_key_types`](Self::from_key_types) with the rules of `rules` (the near rule
    /// and the empty-box rule, each optional; [`with_rules`](Self::with_rules)); local, for
    /// tests on hand-built maps.
    ///
    /// # Errors
    /// Those of [`from_key_types`](Self::from_key_types), and
    /// [`PlanError::MissingCount`] where a rule's counts lack a count it reads.
    pub fn from_key_types_with_rules(
        all_keys: &HashMap<MortonKey, KeyType>,
        nlevels: usize,
        map: &PartitionMap,
        rules: &PlanRules<'_>,
    ) -> Result<Self, PlanError> {
        let index = index_from_keys(all_keys, nlevels)?;
        Self::from_index_with_rules(index, all_keys, map, rules)
    }

    /// The plan of [`from_key_types_with_rules`](Self::from_key_types_with_rules) on
    /// `index`, the index of `all_keys` ([`index_from_keys`]) before its ghost leaves are
    /// numbered; local.
    pub(crate) fn from_index_with_rules(
        mut index: BoxIndex,
        all_keys: &HashMap<MortonKey, KeyType>,
        map: &PartitionMap,
        rules: &PlanRules<'_>,
    ) -> Result<Self, PlanError> {
        let nlevels = index.nlevels();
        let rank = map.rank();
        let coarse_blocks = map.blocks()[map.rank_blocks(rank)].to_vec();
        // P2: the `Global` boxes whose locals this rank needs, the ancestors of its own
        // coarse blocks (its shared ancestors), ascending.
        let mut needed_globals = Vec::new();
        for &block in &coarse_blocks {
            let mut key = block;
            while let Some(parent) = morton::parent(key) {
                needed_globals.push(parent);
                key = parent;
            }
        }
        needed_globals.sort_unstable();
        needed_globals.dedup();

        validate_tree(&index, all_keys, &needed_globals)?;
        for &key in &coarse_blocks {
            match index.find(key) {
                Some((level, i)) if !index.kind(level, i as usize).is_ghost() => {}
                _ => return Err(PlanError::CoarseBlock { key }),
            }
        }

        let top = top_owners(&index, map);
        let owned = |level: usize, i: usize| {
            top[level]
                .binary_search_by_key(&(i as u32), |&(j, _)| j)
                .is_ok_and(|t| top[level][t].1 as usize == rank)
        };

        // First pass: every view that needs only box indices, and the U- and X-entries
        // as boxes, which become leaf indices once the ghost leaves are numbered.
        let occupancies = rules
            .empty_boxes
            .map(|counts| Occupancies::new(&index, counts));
        let mut partial = Vec::with_capacity(nlevels);
        let mut ghost_leaves = Vec::new();
        let mut near_pairs = NearPairs::default();
        let mut empty_boxes = EmptyBoxes::default();
        for level in 0..nlevels {
            partial.push(first_pass(
                &index,
                all_keys,
                &needed_globals,
                &|i| owned(level, i),
                level,
                &mut ghost_leaves,
                (
                    rules.near,
                    &mut near_pairs,
                    occupancies.as_ref(),
                    &mut empty_boxes,
                ),
            )?);
        }
        index.set_ghost_leaves(ghost_leaves);

        // Second pass: the near and X views in leaf indices.
        let levels = partial
            .into_iter()
            .enumerate()
            .map(|(level, partial)| partial.finish(&index, level))
            .collect::<Result<_, _>>()?;

        let deepest_block_level = map
            .blocks()
            .iter()
            .map(|&block| morton::level(block))
            .max()
            .unwrap_or(0)
            .min(nlevels.saturating_sub(1));
        Ok(Self {
            index,
            levels,
            coarse_blocks,
            rank,
            top,
            deepest_block_level,
            near_pairs,
            empty_boxes,
        })
    }

    /// Return the box index.
    pub fn index(&self) -> &BoxIndex {
        &self.index
    }

    /// Return the number of levels, the global maximum level plus one.
    pub fn nlevels(&self) -> usize {
        self.levels.len()
    }

    /// Return the views of `level`.
    pub fn level(&self, level: usize) -> &LevelLists {
        &self.levels[level]
    }

    /// Return the views of every level, by level.
    pub fn levels(&self) -> &[LevelLists] {
        &self.levels
    }

    /// Return this rank's coarse blocks, ascending.
    pub fn coarse_blocks(&self) -> &[MortonKey] {
        &self.coarse_blocks
    }

    /// Return the rank of the partition map the plan was built with: this rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Return the held top-tree keys of `level` (coarse blocks and ancestors of one) as
    /// (box index, owner), ascending in the box index ([Owners](self#owners)).
    pub fn top_boxes(&self, level: usize) -> &[(u32, u32)] {
        &self.top[level]
    }

    /// Whether box `i` of `level` is a top-tree key: a coarse block of any rank or an
    /// ancestor of one.
    pub fn is_top(&self, level: usize, i: usize) -> bool {
        self.top_owner(level, i).is_some()
    }

    /// Return the rank that computes the multipole of box `i` of `level`
    /// ([Owners](self#owners)): this rank for a local box, the owner rank for a ghost,
    /// the owner of the partition map for a `Global` box.
    pub fn owner(&self, level: usize, i: usize) -> usize {
        match self.index.kind(level, i) {
            KeyType::LocalLeaf | KeyType::LocalInterior => self.rank,
            KeyType::GhostLeaf(rank) | KeyType::GhostInterior(rank) => rank,
            KeyType::Global => self
                .top_owner(level, i)
                .expect("every `Global` box is a top-tree key"),
        }
    }

    /// Whether this rank computes the multipole of box `i` of `level`: a local box, or a
    /// `Global` box it owns.
    pub fn computes(&self, level: usize, i: usize) -> bool {
        self.owner(level, i) == self.rank
    }

    /// Return the W and X pairs between leaves of this rank's rows, and how many of each
    /// the near rule converted (none without a rule; the [`near`](crate::near) module).
    pub fn near_pairs(&self) -> NearPairs {
        self.near_pairs
    }

    /// Return what the empty-box rule left out of this rank's rows (nothing without the
    /// rule; the [`empty`](crate::empty) module).
    pub fn empty_boxes(&self) -> EmptyBoxes {
        self.empty_boxes
    }

    /// Return the deepest level of a coarse block of any rank, the same on every rank
    /// (from the replicated partition map); 0 on one rank.
    pub fn deepest_block_level(&self) -> usize {
        self.deepest_block_level
    }

    /// Return the levels of the upward reduction, `1..=deepest_block_level()`: level l
    /// sends the multipoles of level l to the owners of their parents
    /// ([`ReductionExchange`](super::exchange::ReductionExchange)). Empty on one rank.
    pub fn reduction_levels(&self) -> std::ops::Range<usize> {
        1..self.deepest_block_level + 1
    }

    /// The owner of the top-tree key `i` of `level`, if it is one.
    fn top_owner(&self, level: usize, i: usize) -> Option<usize> {
        let top = &self.top[level];
        top.binary_search_by_key(&(i as u32), |&(j, _)| j)
            .ok()
            .map(|t| top[t].1 as usize)
    }
}

/// Number the keys of `all_keys` on the levels `0..nlevels` ([`BoxIndex`]), with their
/// kinds and the local leaves; local. The checks of
/// [`Plan::from_key_types`] that need no rows, in its order.
pub(crate) fn index_from_keys(
    all_keys: &HashMap<MortonKey, KeyType>,
    nlevels: usize,
) -> Result<BoxIndex, PlanError> {
    let mut sorted: Vec<MortonKey> = all_keys.keys().copied().collect();
    sorted.sort_unstable();
    let mut keys = vec![Vec::new(); nlevels];
    for key in sorted {
        if !morton::is_valid(key) {
            return Err(PlanError::MalformedTree {
                key,
                reason: "the key is not a valid Morton key",
            });
        }
        let level = morton::level(key);
        if level >= nlevels {
            return Err(PlanError::LevelOutOfRange { key, nlevels });
        }
        keys[level].push(key);
    }
    for (level, level_keys) in keys.iter().enumerate() {
        if level_keys.len() > u32::MAX as usize {
            return Err(PlanError::TooManyBoxes {
                level,
                count: level_keys.len(),
            });
        }
    }
    let kinds = keys
        .iter()
        .map(|level| level.iter().map(|key| all_keys[key]).collect())
        .collect();
    Ok(BoxIndex::new(keys, kinds))
}

/// Agree the outcome of a local build (a plan or a box index) on every rank of `octree`'s
/// communicator: one
/// all-reduce; a rank whose build succeeded returns [`PlanError::OtherRank`] if another
/// rank's failed.
pub(crate) fn agree<T, C: CommunicatorCollectives>(
    octree: &Octree<'_, C>,
    local: Result<T, PlanError>,
) -> Result<T, PlanError> {
    let mut valid = false;
    octree
        .comm()
        .all_reduce_into(&local.is_ok(), &mut valid, SystemOperation::logical_and());
    match local {
        Ok(_) if !valid => Err(PlanError::OtherRank),
        result => result,
    }
}

/// The held top-tree keys of every level as (box index, owner), ascending, from `map`
/// ([Owners](self#owners)). Local.
fn top_owners(index: &BoxIndex, map: &PartitionMap) -> Vec<Vec<(u32, u32)>> {
    (0..index.nlevels())
        .map(|level| {
            index
                .keys(level)
                .iter()
                .enumerate()
                .filter(|&(_, &key)| map.top_kind(key).is_some())
                .map(|(i, &key)| (i as u32, map.owner(key) as u32))
                .collect()
        })
        .collect()
}

/// Whether `key`, of kind `kind`, is a `Global` box whose local this rank does not need
/// (P2): not an ancestor of one of its coarse blocks, `needed_globals` (ascending).
fn is_pruned(key: MortonKey, kind: KeyType, needed_globals: &[MortonKey]) -> bool {
    kind == KeyType::Global && needed_globals.binary_search(&key).is_err()
}

/// Check the preconditions of the per-key rule, so that it cannot panic: no interior key
/// lies on the deepest level, and for every key the rule runs on (every non-ghost key but
/// the pruned `Global` boxes of P2) every same-level neighbour of its parent is held,
/// and, for a leaf, every same-level neighbour cell is held or covered by a held leaf
/// parent.
fn validate_tree(
    index: &BoxIndex,
    all_keys: &HashMap<MortonKey, KeyType>,
    needed_globals: &[MortonKey],
) -> Result<(), PlanError> {
    let malformed = |key, reason| Err(PlanError::MalformedTree { key, reason });
    for level in 0..index.nlevels() {
        for (&key, &kind) in index.keys(level).iter().zip(index.kinds(level)) {
            if is_interior(kind) && level >= DEEPEST_LEVEL as usize {
                return malformed(key, "an interior key lies on the deepest level");
            }
            if kind.is_ghost() || is_pruned(key, kind, needed_globals) {
                continue;
            }
            if is_leaf(kind) {
                for neighbour in valid_neighbours(key) {
                    if all_keys.contains_key(&neighbour) {
                        continue;
                    }
                    let covered = morton::parent(neighbour)
                        .and_then(|parent| all_keys.get(&parent))
                        .is_some_and(|&parent_kind| is_leaf(parent_kind));
                    if !covered {
                        return malformed(
                            key,
                            "a missing neighbour cell of a leaf is not covered by a held leaf parent",
                        );
                    }
                }
            }
            if let Some(parent) = morton::parent(key)
                && valid_neighbours(parent).any(|neighbour| !all_keys.contains_key(&neighbour))
            {
                return malformed(key, "a same-level neighbour of the parent is not held");
            }
        }
    }
    Ok(())
}

/// The views of one level after the first pass. U- and X-entries are still (level, box).
struct PartialLevel {
    v: GroupedBuilder<u16>,
    m2m_local: GroupedBuilder<u8>,
    m2m_global: GroupedBuilder<u8>,
    l2l: GroupedBuilder<u8>,
    p2m: CsrBuilder,
    w: CsrBuilder,
    l2p: CsrBuilder,
    /// Per local leaf of the level: offsets into `u_entries`.
    u_offsets: Vec<usize>,
    u_entries: Vec<(usize, u32)>,
    /// Per box of the level: offsets into `x_entries`.
    x_offsets: Vec<usize>,
    x_entries: Vec<(usize, u32)>,
}

/// Build the views of `level` that need only box indices, and collect the U- and
/// X-entries, recording every ghost leaf among them in `ghost_leaves`.
///
/// With a near `rule`, the W entries of a local leaf whose box is a leaf, and the X
/// entries of a local leaf, that the rule converts join the leaf's U-entries (its near
/// row) instead, a ghost W entry recorded in `ghost_leaves` as well; `near_pairs` counts
/// the candidates and the conversions.
///
/// With the occupancies of the empty-box rule, a box whose subtree holds no target gets
/// empty V, X, W and L2L rows (its X entries not recorded in `ghost_leaves`), and the V, W
/// and M2M entries naming a box whose subtree holds no source are left out, before the near
/// rule sees a W or X entry; `empty_boxes` counts them ([`empty`](crate::empty)).
///
/// A `Global` box that is not in `needed_globals` (ascending) gets empty V, X and L2L
/// rows (P2). Its lists are not resolved: since Phase 5S T4 the plan does not hold its
/// neighbourhood. A `Global` box has its M2M row of the global pass only if `owned`
/// (by its box index) says this rank owns it ([Owners](self#owners)).
fn first_pass(
    index: &BoxIndex,
    all_keys: &HashMap<MortonKey, KeyType>,
    needed_globals: &[MortonKey],
    owned: &dyn Fn(usize) -> bool,
    level: usize,
    ghost_leaves: &mut Vec<(usize, u32)>,
    (rule, near_pairs, occupancies, empty_boxes): (
        Option<&NearRule<'_>>,
        &mut NearPairs,
        Option<&Occupancies>,
        &mut EmptyBoxes,
    ),
) -> Result<PartialLevel, PlanError> {
    let mut partial = PartialLevel {
        v: GroupedBuilder::new(NOFFSETS),
        m2m_local: GroupedBuilder::new(NOCTANTS),
        m2m_global: GroupedBuilder::new(NOCTANTS),
        l2l: GroupedBuilder::new(NOCTANTS),
        p2m: CsrBuilder::default(),
        w: CsrBuilder::default(),
        l2p: CsrBuilder::default(),
        u_offsets: vec![0],
        u_entries: Vec::new(),
        x_offsets: vec![0],
        x_entries: Vec::new(),
    };
    let malformed = |key, reason| PlanError::MalformedTree { key, reason };
    let mut v_row = Vec::new();

    for (i, (&key, &kind)) in index.keys(level).iter().zip(index.kinds(level)).enumerate() {
        if kind.is_ghost() {
            partial.v.push_row([]);
            partial.m2m_local.push_row([]);
            partial.m2m_global.push_row([]);
            partial.l2l.push_row([]);
            partial.p2m.push_row([]);
            partial.x_offsets.push(partial.x_entries.len());
            continue;
        }

        // P2 (distributed-fmm §3.6): this rank's locals descend only from the `Global`
        // boxes above its own coarse blocks, so the others get no downward rows.
        if is_pruned(key, kind, needed_globals) {
            let first = index
                .first_child(level, i)
                .ok_or(PlanError::MissingChild { key })?;
            partial.v.push_row([]);
            partial.m2m_local.push_row([]);
            if owned(i) {
                let children = (0..NOCTANTS as u8).map(|o| (o, first + o as u32));
                partial.m2m_global.push_row(children_with_sources(
                    children,
                    level,
                    occupancies.map(|o| (index, o)),
                    &mut empty_boxes.m2m_without_sources,
                )?);
            } else {
                partial.m2m_global.push_row([]);
            }
            partial.l2l.push_row([]);
            partial.p2m.push_row([]);
            partial.x_offsets.push(partial.x_entries.len());
            continue;
        }
        let [u, v, w, x] = key_lists(key, kind, all_keys);
        let resolve = |entry| {
            index
                .find(entry)
                .ok_or(PlanError::UnheldListEntry { key, entry })
        };
        let is_leaf_box =
            |(entry_level, j): (usize, u32)| is_leaf(index.kind(entry_level, j as usize));
        // A non-ghost box with a leaf index is a local leaf: the only boxes with near rows.
        let leaf = index.box_leaf(level, i);
        let rule = rule.filter(|_| leaf.is_some());
        let missing = |key| PlanError::MissingCount { key };
        // The empty-box rule: does the subtree of this box hold no target, and does that of
        // the box (entry level, j) hold a source?
        let no_targets = match occupancies {
            Some(occupancies) => !occupancies.has_targets(index, level, i)?,
            None => false,
        };
        if no_targets {
            empty_boxes.target_boxes += 1;
        }
        let has_sources = |(entry_level, j): (usize, u32)| match occupancies {
            Some(occupancies) => occupancies.has_sources(index, entry_level, j as usize),
            None => Ok(true),
        };

        v_row.clear();
        let mut v_near = Vec::new();
        for source in v {
            let (source_level, j) = resolve(source)?;
            if source_level != level {
                return Err(malformed(key, "a V-list entry lies on another level"));
            }
            let d = offset_index(offset(key, source)).ok_or(malformed(
                key,
                "a V-list offset is not one of V_LIST_DIRECTIONS",
            ))?;
            if no_targets {
                empty_boxes.v_without_targets += 1;
                continue;
            }
            if !has_sources((source_level, j))? {
                empty_boxes.v_without_sources += 1;
                continue;
            }
            // The near rule's V pairs (Helmholtz CH4.0d2): a converted leaf joins the
            // U-entries.
            if let Some(rule) = rule.filter(|rule| rule.offers_v_pairs())
                && is_leaf_box((source_level, j))
            {
                near_pairs.v_leaf_pairs += 1;
                let pair = VPair {
                    target: key,
                    source,
                    sources: rule.counts().sources(source).ok_or(missing(source))?,
                    targets: rule.counts().targets(key).ok_or(missing(key))?,
                };
                if rule.converts_v(pair) {
                    near_pairs.v_converted += 1;
                    v_near.push((source_level, j));
                    continue;
                }
            }
            v_row.push((d as u16, j));
        }
        v_row.sort_unstable();
        partial.v.push_row(v_row.iter().copied());

        // W-entries are keys of one level, sorted, so they are ascending in box index.
        // Those the near rule converts join the U-entries.
        let mut w_row = Vec::with_capacity(w.len());
        let mut w_near = Vec::new();
        for source in w {
            let (source_level, j) = resolve(source)?;
            if source_level != level + 1 {
                return Err(malformed(
                    key,
                    "a W-list entry does not lie one level finer",
                ));
            }
            if no_targets {
                empty_boxes.w_without_targets += 1;
                continue;
            }
            if !has_sources((source_level, j))? {
                empty_boxes.w_without_sources += 1;
                continue;
            }
            if let Some(rule) = rule
                && is_leaf_box((source_level, j))
            {
                near_pairs.w_leaf_pairs += 1;
                let sources = rule.counts().sources(source).ok_or(missing(source))?;
                if rule.converts(LeafPair::W {
                    target: key,
                    source,
                    sources,
                }) {
                    near_pairs.w_converted += 1;
                    w_near.push((source_level, j));
                    continue;
                }
            }
            w_row.push(j);
        }

        for source in u {
            let entry = resolve(source)?;
            if !is_leaf_box(entry) || entry.0 + 1 < level || entry.0 > level + 1 {
                return Err(malformed(
                    key,
                    "a U-list entry is not a leaf within one level of the box",
                ));
            }
            partial.u_entries.push(entry);
            if index.kind(entry.0, entry.1 as usize).is_ghost() {
                ghost_leaves.push(entry);
            }
        }
        for entry in w_near.into_iter().chain(v_near) {
            partial.u_entries.push(entry);
            if index.kind(entry.0, entry.1 as usize).is_ghost() {
                ghost_leaves.push(entry);
            }
        }
        for source in x {
            let entry = resolve(source)?;
            if !is_leaf_box(entry) || entry.0 + 1 != level {
                return Err(malformed(
                    key,
                    "an X-list entry is not a leaf one level coarser",
                ));
            }
            if no_targets {
                empty_boxes.x_without_targets += 1;
                continue;
            }
            if index.kind(entry.0, entry.1 as usize).is_ghost() {
                ghost_leaves.push(entry);
            }
            if let Some(rule) = rule {
                near_pairs.x_leaf_pairs += 1;
                let targets = rule.counts().targets(key).ok_or(missing(key))?;
                if rule.converts(LeafPair::X {
                    target: key,
                    source,
                    targets,
                }) {
                    near_pairs.x_converted += 1;
                    partial.u_entries.push(entry);
                    continue;
                }
            }
            partial.x_entries.push(entry);
        }
        partial.x_offsets.push(partial.x_entries.len());

        if is_interior(kind) {
            let first = index
                .first_child(level, i)
                .ok_or(PlanError::MissingChild { key })?;
            let children = (0..NOCTANTS as u8).map(|o| (o, first + o as u32));
            // The empty-box rule leaves out the children whose subtrees hold no source, and
            // counts them on the rows this rank keeps.
            let mut kept = |children| {
                children_with_sources(
                    children,
                    level,
                    occupancies.map(|o| (index, o)),
                    &mut empty_boxes.m2m_without_sources,
                )
            };
            if kind == KeyType::Global {
                partial.m2m_local.push_row([]);
                if owned(i) {
                    partial.m2m_global.push_row(kept(children)?);
                } else {
                    partial.m2m_global.push_row([]);
                }
            } else {
                partial.m2m_local.push_row(kept(children)?);
                partial.m2m_global.push_row([]);
            }
        } else {
            partial.m2m_local.push_row([]);
            partial.m2m_global.push_row([]);
        }

        match morton::parent(key) {
            Some(parent) => {
                let Some((_, j)) = index.find(parent) else {
                    return Err(malformed(key, "the parent of a non-ghost key is not held"));
                };
                if no_targets {
                    empty_boxes.l2l_without_targets += 1;
                    partial.l2l.push_row([]);
                } else {
                    partial.l2l.push_row([(morton::child_index(key) as u8, j)]);
                }
            }
            None => partial.l2l.push_row([]),
        }

        match leaf {
            Some(leaf) => {
                partial.p2m.push_row([leaf]);
                partial.w.push_row(w_row);
                partial.l2p.push_row([i as u32]);
                partial.u_offsets.push(partial.u_entries.len());
            }
            None => partial.p2m.push_row([]),
        }
    }
    Ok(partial)
}

/// The children (octant, box index on `level + 1`) of an M2M row whose subtrees hold a
/// source, under the empty-box rule's occupancies (all of them without the rule), counting
/// those left out in `left_out`.
fn children_with_sources(
    children: impl Iterator<Item = (u8, u32)>,
    level: usize,
    occupancies: Option<(&BoxIndex, &Occupancies)>,
    left_out: &mut usize,
) -> Result<Vec<(u8, u32)>, PlanError> {
    let mut row = Vec::with_capacity(NOCTANTS);
    for (o, j) in children {
        match occupancies {
            Some((index, occupancies))
                if !occupancies.has_sources(index, level + 1, j as usize)? =>
            {
                *left_out += 1;
            }
            _ => row.push((o, j)),
        }
    }
    Ok(row)
}

impl PartialLevel {
    /// Translate the U- and X-entries to leaf indices and finish every view.
    fn finish(self, index: &BoxIndex, level: usize) -> Result<LevelLists, PlanError> {
        let leaf_of = |(entry_level, j): (usize, u32)| {
            index
                .box_leaf(entry_level, j as usize)
                .expect("every leaf named by a U- or X-entry has a leaf index")
        };

        // P1 (distributed-fmm §5.1): rows by the entry leaf's (level, key), which is the
        // leaf index on one rank and places a ghost leaf where the one-rank row has it.
        let leaves = index.leaves();
        let by_level_and_key = |&j: &u32| (leaves.level(j as usize), leaves.key(j as usize));

        let mut near = CsrBuilder::default();
        let mut row = Vec::new();
        let local = leaves.local(level);
        for (r, bounds) in self.u_offsets.windows(2).enumerate() {
            row.clear();
            row.push((local.start + r) as u32);
            row.extend(
                self.u_entries[bounds[0]..bounds[1]]
                    .iter()
                    .map(|&e| leaf_of(e)),
            );
            row.sort_unstable_by_key(by_level_and_key);
            near.push_row(row.iter().copied());
        }

        let mut x = CsrBuilder::default();
        for bounds in self.x_offsets.windows(2) {
            row.clear();
            row.extend(
                self.x_entries[bounds[0]..bounds[1]]
                    .iter()
                    .map(|&e| leaf_of(e)),
            );
            row.sort_unstable_by_key(by_level_and_key);
            x.push_row(row.iter().copied());
        }

        let overflow = |_: Overflow| PlanError::TooManyEntries { level };
        Ok(LevelLists {
            v: self.v.finish().map_err(overflow)?,
            x: x.finish().map_err(overflow)?,
            m2m_local: self.m2m_local.finish().map_err(overflow)?,
            m2m_global: self.m2m_global.finish().map_err(overflow)?,
            l2l: self.l2l.finish().map_err(overflow)?,
            p2m: self.p2m.finish().map_err(overflow)?,
            near: near.finish().map_err(overflow)?,
            w: self.w.finish().map_err(overflow)?,
            l2p: self.l2p.finish().map_err(overflow)?,
        })
    }
}

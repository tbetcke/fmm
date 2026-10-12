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
//! Since Phase 5S T4 the plan holds **H ∪ T** (`docs/design/scale-out.md` §3.4, §3.7):
//! - H, the octree's locally essential set
//!   ([`Octree::all_keys`](nd_octree::Octree::all_keys)): with S the own keys and the
//!   shared ancestors (the ancestors of this rank's coarse blocks), S, every same-level
//!   neighbour of a key of S and the children of the interior ones;
//! - T, the top-tree keys: every coarse block of every rank and every ancestor of one,
//!   from the replicated [`PartitionMap`](nd_octree::octree::PartitionMap), without
//!   communication. Other ranks' blocks are ghosts of their owners; **every** ancestor
//!   of a block is `Global`, as in Phase 5, also where the octree holds it as a ghost
//!   (an ancestor of other ranks' blocks only).
//!
//! T is what the coarse gather and the global upward pass still need until Phase 5S T5
//! replaces them: a slot for every block, and the M2M row of every `Global` box. It
//! grows with P (about 1.15 B keys for B blocks); H does not grow at fixed N/P.
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
//!   resolved: the plan does not hold its neighbourhood (Phase 5S T4). Its M2M row of
//!   the global pass stays, because the V rows that remain read the multipoles of any
//!   `Global` box. Every row that stays is the row it would be without P2, so every
//!   value of a needed box is the same, bit for bit. One rank has no `Global` box with
//!   a row to drop, and a rank without coarse blocks keeps no `Global` downward row.
//! - **Every key a list or a parent-child step names is held**: every list entry of a
//!   box with rows, the parent of such a box, every child of a non-ghost interior box
//!   (in particular every child of a `Global` box, which is a coarse block or `Global`
//!   itself) and every own coarse block. [`Plan::new`] checks this and fails on every
//!   rank otherwise. The keys the exchanges name are the ghosts among them, and every
//!   coarse block (T).
//! - **Phase 5's lists.** Every list entry, as a key, and the order of every row are
//!   those of the Phase 5 plan over the Phase 5 key set: the rows depend only on the
//!   tree and the rule, P1 orders by (level, key), and the kinds of the keys the rows
//!   name are Phase 5's (`docs/design/scale-out.md` §3.6). Only the box indices change,
//!   because fewer keys are held.
//! - The plan owns its arrays and borrows nothing from the octree. A new tree needs a new
//!   plan: box indices, unlike keys, change when the tree changes.
//!
//! # Cost
//!
//! The lists are built locally, in O(E log K) for E list entries and K held keys, and T
//! in O(B) for B coarse blocks. [`Plan::new`] adds two all-reduces.

#[cfg(test)]
#[path = "plan_tests.rs"]
pub(crate) mod tests;

use std::{collections::HashMap, error::Error, fmt};

use mpi::{collective::SystemOperation, traits::CommunicatorCollectives};
use nd_octree::{MortonKey, Octree, constants::DEEPEST_LEVEL, morton, octree::KeyType};

use super::index::BoxIndex;
use super::lists::{
    CsrBuilder, GroupedBuilder, LevelLists, NOCTANTS, NOFFSETS, Overflow, offset, offset_index,
};
use crate::interaction_manager::{is_interior, is_leaf, key_lists, valid_neighbours};

/// The box index and the interaction lists of an octree on one rank.
///
/// See the [module documentation](self).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    index: BoxIndex,
    levels: Vec<LevelLists>,
    coarse_blocks: Vec<MortonKey>,
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
            Self::OtherRank => write!(f, "building the plan failed on another rank"),
        }
    }
}

impl Error for PlanError {}

impl Plan {
    /// Build the box index and the interaction lists of `octree`.
    ///
    /// The plan holds the octree's locally essential keys and the top-tree keys of its
    /// partition map, H ∪ T (see [Held keys](self#held-keys)).
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
        let nlevels = octree.global_max_level() + 1;
        let held = held_keys(octree);
        let local = if octree.options().ghost_children() {
            Self::from_key_types(&held, nlevels, octree.coarse_tree_leafs())
        } else {
            Err(PlanError::MissingGhostChildren)
        };

        let mut valid = false;
        octree
            .comm()
            .all_reduce_into(&local.is_ok(), &mut valid, SystemOperation::logical_and());
        match local {
            Ok(_) if !valid => Err(PlanError::OtherRank),
            result => result,
        }
    }

    /// Build the plan from a key classification; local, for tests on hand-built maps.
    ///
    /// `all_keys` must describe a complete, 2:1 (26-neighbour) balanced tree together with
    /// its one-cell same-level halo, as
    /// [`Octree::all_keys`](nd_octree::Octree::all_keys) does, on the levels
    /// `0..nlevels`. `own_coarse_blocks` are this rank's coarse blocks
    /// ([`Octree::coarse_tree_leafs`](nd_octree::Octree::coarse_tree_leafs)).
    ///
    /// # Errors
    /// Returns the first defect found, scanning levels and keys in ascending order, so the
    /// error does not depend on the iteration order of `all_keys`. Never panics on a
    /// malformed map.
    pub fn from_key_types(
        all_keys: &HashMap<MortonKey, KeyType>,
        nlevels: usize,
        own_coarse_blocks: &[MortonKey],
    ) -> Result<Self, PlanError> {
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
        let mut index = BoxIndex::new(keys, kinds);

        let mut coarse_blocks = own_coarse_blocks.to_vec();
        coarse_blocks.sort_unstable();
        coarse_blocks.dedup();
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

        // First pass: every view that needs only box indices, and the U- and X-entries
        // as boxes, which become leaf indices once the ghost leaves are numbered.
        let mut partial = Vec::with_capacity(nlevels);
        let mut ghost_leaves = Vec::new();
        for level in 0..nlevels {
            partial.push(first_pass(
                &index,
                all_keys,
                &needed_globals,
                level,
                &mut ghost_leaves,
            )?);
        }
        index.set_ghost_leaves(ghost_leaves);

        // Second pass: the near and X views in leaf indices.
        let levels = partial
            .into_iter()
            .enumerate()
            .map(|(level, partial)| partial.finish(&index, level))
            .collect::<Result<_, _>>()?;

        Ok(Self {
            index,
            levels,
            coarse_blocks,
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
}

/// The keys the plan holds, H ∪ T, with their kinds (see [Held keys](self#held-keys)): the
/// octree's locally essential set, with every coarse block and every ancestor of one
/// added from the partition map; a block keeps the kind the map gives it, and every
/// ancestor of a block is `Global`. Local, O(B) for B blocks.
fn held_keys<C: CommunicatorCollectives>(octree: &Octree<'_, C>) -> HashMap<MortonKey, KeyType> {
    let map = octree.partition_map();
    let mut held = octree.all_keys().clone();
    for &block in map.blocks() {
        let kind = map
            .top_kind(block)
            .expect("a coarse block is a top-tree key");
        let previous = held.insert(block, kind);
        debug_assert!(
            previous.is_none_or(|previous| previous == kind),
            "the octree and its partition map classify block {block} alike"
        );
        let mut key = block;
        while let Some(parent) = morton::parent(key) {
            // An ancestor already `Global` was reached from an earlier block, or is a
            // shared ancestor, whose ancestors are shared ancestors too.
            if held.insert(parent, KeyType::Global) == Some(KeyType::Global) {
                break;
            }
            key = parent;
        }
    }
    held
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
/// A `Global` box that is not in `needed_globals` (ascending) gets empty V, X and L2L
/// rows (P2) and only its M2M row of the global pass. Its lists are not resolved: since
/// Phase 5S T4 the plan does not hold its neighbourhood.
fn first_pass(
    index: &BoxIndex,
    all_keys: &HashMap<MortonKey, KeyType>,
    needed_globals: &[MortonKey],
    level: usize,
    ghost_leaves: &mut Vec<(usize, u32)>,
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
            partial
                .m2m_global
                .push_row((0..NOCTANTS as u8).map(|o| (o, first + o as u32)));
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

        v_row.clear();
        for source in v {
            let (source_level, j) = resolve(source)?;
            if source_level != level {
                return Err(malformed(key, "a V-list entry lies on another level"));
            }
            let d = offset_index(offset(key, source)).ok_or(malformed(
                key,
                "a V-list offset is not one of V_LIST_DIRECTIONS",
            ))?;
            v_row.push((d as u16, j));
        }
        v_row.sort_unstable();
        partial.v.push_row(v_row.iter().copied());

        // W-entries are keys of one level, sorted, so they are ascending in box index.
        let mut w_row = Vec::with_capacity(w.len());
        for source in w {
            let (source_level, j) = resolve(source)?;
            if source_level != level + 1 {
                return Err(malformed(
                    key,
                    "a W-list entry does not lie one level finer",
                ));
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
        for source in x {
            let entry = resolve(source)?;
            if !is_leaf_box(entry) || entry.0 + 1 != level {
                return Err(malformed(
                    key,
                    "an X-list entry is not a leaf one level coarser",
                ));
            }
            partial.x_entries.push(entry);
            if index.kind(entry.0, entry.1 as usize).is_ghost() {
                ghost_leaves.push(entry);
            }
        }
        partial.x_offsets.push(partial.x_entries.len());

        if is_interior(kind) {
            let first = index
                .first_child(level, i)
                .ok_or(PlanError::MissingChild { key })?;
            let children = (0..NOCTANTS as u8).map(|o| (o, first + o as u32));
            if kind == KeyType::Global {
                partial.m2m_local.push_row([]);
                partial.m2m_global.push_row(children);
            } else {
                partial.m2m_local.push_row(children);
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
                partial.l2l.push_row([(morton::child_index(key) as u8, j)]);
            }
            None => partial.l2l.push_row([]),
        }

        match index.box_leaf(level, i) {
            // A non-ghost box with a leaf index is a local leaf.
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

impl PartialLevel {
    /// Translate the U- and X-entries to leaf indices and finish every view.
    fn finish(self, index: &BoxIndex, level: usize) -> Result<LevelLists, PlanError> {
        let leaf_of = |(entry_level, j): (usize, u32)| {
            index
                .box_leaf(entry_level, j as usize)
                .expect("every leaf named by a U- or X-list has a leaf index")
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

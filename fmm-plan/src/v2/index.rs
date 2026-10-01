//! The box index: every box a rank holds, numbered densely in Morton order per level.
//!
//! [`BoxIndex`] numbers, on every level l in `0..nlevels`, every key of
//! [`Octree::all_keys`](nd_octree::Octree::all_keys) on that level, whatever its
//! [`KeyType`]: local leaves and interiors, `Global` boxes and ghosts. Box i of level l
//! is the i-th smallest key of the level, so the numbering is Morton order with the
//! positions this rank does not hold left out. Every level buffer, exchange buffer and
//! list view of the plan addresses boxes by these indices (design §3,
//! `docs/design/fmm-plan-redesign.md`).
//!
//! Leaves have a second numbering, because leaf data are not level data:
//! - the local leaves first, as leaf indices `0..nlocal`, ordered by (level, key), so the
//!   local leaves of level l are one range, [`LeafNumbering::local`];
//! - then the ghost leaves that some U- or X-list of a non-ghost box names, as
//!   `nlocal..len`, ordered by key, which is the order of their owners
//!   ([`LeafNumbering::ghosts`]).
//!
//! Other ghost leaves have a box index but no leaf index.
//!
//! # Guarantees
//!
//! - **Dense and Morton-ordered.** [`keys(l)`](BoxIndex::keys) is strictly ascending and
//!   holds every key of the map on level l, so box indices are `0..len(l)`.
//! - **Deterministic.** The index is a pure function of the key classification and the
//!   level count: the iteration order of the input `HashMap` never matters.
//! - **Consecutive children.** If all eight children of a box are held, they are eight
//!   consecutive indices on the next level, in octant order (`morton::child_index`), and
//!   [`first_child`](BoxIndex::first_child) returns the first. That holds for every
//!   non-ghost interior box and every interior neighbour of one when the tree carries the
//!   ghost-children layer, but generally not for the ghost interiors that the layer adds.
//! - **No `HashMap`.** Index → key, kind, leaf and first child are array reads. Key →
//!   index is a binary search, meant for building, loading points, reading results and
//!   tests; hot paths take every index from a list view.
//!
//! # Cost
//!
//! Built in O(K log K) for K held keys (one sort per level). Memory: 32 bytes per box
//! (key, kind, first child, leaf) and 13 bytes per numbered leaf.

use std::ops::Range;

use nd_octree::{MortonKey, morton, octree::KeyType};

/// The marker for "no index" in the `u32` arrays of [`BoxIndex`].
pub const NONE: u32 = u32::MAX;

/// Every box this rank holds, numbered `0..len(level)` in Morton order on each level.
///
/// See the [module documentation](self).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoxIndex {
    /// `[level][box]`, strictly ascending.
    keys: Vec<Vec<MortonKey>>,
    /// `[level][box]`, the classification of every key.
    kinds: Vec<Vec<KeyType>>,
    /// `[level][box]`: index of child 0 on `level + 1` if all eight children are held,
    /// otherwise [`NONE`].
    first_child: Vec<Vec<u32>>,
    /// `[level][box]`: leaf index, or [`NONE`].
    box_leaf: Vec<Vec<u32>>,
    leaves: LeafNumbering,
}

/// The leaf numbering of a [`BoxIndex`]: local leaves by (level, key), then the ghost
/// leaves named by a U- or X-list, by key.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeafNumbering {
    keys: Vec<MortonKey>,
    levels: Vec<u8>,
    boxes: Vec<u32>,
    /// `nlevels + 1` offsets: the local leaves of level l are
    /// `local_offsets[l]..local_offsets[l + 1]`.
    local_offsets: Vec<usize>,
}

impl BoxIndex {
    /// Number the keys of `keys` (strictly ascending per level) with their `kinds`, and
    /// the local leaves. Ghost leaves are added with
    /// [`set_ghost_leaves`](Self::set_ghost_leaves).
    pub(crate) fn new(keys: Vec<Vec<MortonKey>>, kinds: Vec<Vec<KeyType>>) -> Self {
        let nlevels = keys.len();
        debug_assert!(keys.iter().all(|level| level.is_sorted()));

        let mut first_child: Vec<Vec<u32>> =
            keys.iter().map(|level| vec![NONE; level.len()]).collect();
        for level in 0..nlevels.saturating_sub(1) {
            let finer = &keys[level + 1];
            for (i, &key) in keys[level].iter().enumerate() {
                let Some(children) = morton::children(key) else {
                    continue;
                };
                // The children are the keys in [children[0], children[7]] on the next
                // level, so they are consecutive whenever all eight are held.
                if let Ok(first) = finer.binary_search(&children[0])
                    && finer.get(first + 7) == Some(&children[7])
                {
                    first_child[level][i] = first as u32;
                }
            }
        }

        let mut box_leaf: Vec<Vec<u32>> =
            keys.iter().map(|level| vec![NONE; level.len()]).collect();
        let mut leaves = LeafNumbering {
            local_offsets: vec![0],
            ..Default::default()
        };
        for level in 0..nlevels {
            for (i, (&key, &kind)) in keys[level].iter().zip(&kinds[level]).enumerate() {
                if kind == KeyType::LocalLeaf {
                    box_leaf[level][i] = leaves.keys.len() as u32;
                    leaves.keys.push(key);
                    leaves.levels.push(level as u8);
                    leaves.boxes.push(i as u32);
                }
            }
            leaves.local_offsets.push(leaves.keys.len());
        }

        Self {
            keys,
            kinds,
            first_child,
            box_leaf,
            leaves,
        }
    }

    /// Number the ghost leaves `ghosts`, given as (level, box), after the local leaves,
    /// in key order. Duplicates are numbered once.
    pub(crate) fn set_ghost_leaves(&mut self, mut ghosts: Vec<(usize, u32)>) {
        ghosts.sort_unstable_by_key(|&(level, i)| self.keys[level][i as usize]);
        ghosts.dedup();
        for (level, i) in ghosts {
            debug_assert!(matches!(
                self.kinds[level][i as usize],
                KeyType::GhostLeaf(_)
            ));
            self.box_leaf[level][i as usize] = self.leaves.keys.len() as u32;
            self.leaves.keys.push(self.keys[level][i as usize]);
            self.leaves.levels.push(level as u8);
            self.leaves.boxes.push(i);
        }
    }

    /// Return the number of levels, the global maximum level plus one.
    pub fn nlevels(&self) -> usize {
        self.keys.len()
    }

    /// Return the number of boxes held on `level`.
    pub fn len(&self, level: usize) -> usize {
        self.keys[level].len()
    }

    /// Return true if no box is held on any level.
    pub fn is_empty(&self) -> bool {
        self.keys.iter().all(Vec::is_empty)
    }

    /// Return the keys of `level`, strictly ascending; box i is `keys(level)[i]`.
    pub fn keys(&self, level: usize) -> &[MortonKey] {
        &self.keys[level]
    }

    /// Return the classifications of the boxes of `level`, parallel to
    /// [`keys`](Self::keys).
    pub fn kinds(&self, level: usize) -> &[KeyType] {
        &self.kinds[level]
    }

    /// Return the key of box `i` of `level`.
    pub fn key(&self, level: usize, i: usize) -> MortonKey {
        self.keys[level][i]
    }

    /// Return the classification of box `i` of `level`.
    pub fn kind(&self, level: usize, i: usize) -> KeyType {
        self.kinds[level][i]
    }

    /// Return the index on `level + 1` of child 0 of box `i` of `level`, if all eight
    /// children are held; child o is then `first_child + o`.
    pub fn first_child(&self, level: usize, i: usize) -> Option<u32> {
        let first = self.first_child[level][i];
        (first != NONE).then_some(first)
    }

    /// Return [`first_child`](Self::first_child) for every box of `level`, with
    /// [`NONE`] for a box without all its children.
    pub fn first_children(&self, level: usize) -> &[u32] {
        &self.first_child[level]
    }

    /// Return the leaf index of box `i` of `level`, if it has one.
    pub fn box_leaf(&self, level: usize, i: usize) -> Option<u32> {
        let leaf = self.box_leaf[level][i];
        (leaf != NONE).then_some(leaf)
    }

    /// Return [`box_leaf`](Self::box_leaf) for every box of `level`, with [`NONE`] for a
    /// box without a leaf index.
    pub fn box_leaves(&self, level: usize) -> &[u32] {
        &self.box_leaf[level]
    }

    /// Return the leaf numbering.
    pub fn leaves(&self) -> &LeafNumbering {
        &self.leaves
    }

    /// Return the key of leaf `j`.
    pub fn leaf_key(&self, j: usize) -> MortonKey {
        self.leaves.keys[j]
    }

    /// Return the level and box index of `key`, or `None` if `key` is not held.
    ///
    /// A binary search on the level of `key`: O(log n).
    pub fn find(&self, key: MortonKey) -> Option<(usize, u32)> {
        if !morton::is_valid(key) {
            return None;
        }
        let level = morton::level(key);
        let i = self.keys.get(level)?.binary_search(&key).ok()?;
        Some((level, i as u32))
    }

    /// Return the leaf index of `key`, or `None` if `key` has none.
    pub fn find_leaf(&self, key: MortonKey) -> Option<u32> {
        let (level, i) = self.find(key)?;
        self.box_leaf(level, i as usize)
    }

    /// Return the index of the local leaf that contains the finest-level key `fine`, or
    /// `None` if no local leaf contains it.
    ///
    /// Searches the ancestors of `fine` level by level: O(L log n).
    pub fn local_leaf_containing(&self, fine: MortonKey) -> Option<u32> {
        if !morton::is_valid(fine) {
            return None;
        }
        (0..self.nlevels().min(morton::level(fine) + 1)).find_map(|level| {
            let ancestor = morton::ancestor_at_level(fine, level)?;
            let i = self.keys[level].binary_search(&ancestor).ok()?;
            (self.kinds[level][i] == KeyType::LocalLeaf).then(|| self.box_leaf[level][i])
        })
    }
}

impl LeafNumbering {
    /// Return the number of numbered leaves, local and ghost.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Return true if no leaf is numbered.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Return the number of local leaves; they are leaves `0..nlocal()`.
    pub fn nlocal(&self) -> usize {
        *self.local_offsets.last().unwrap()
    }

    /// Return the leaf indices of the local leaves of `level`.
    pub fn local(&self, level: usize) -> Range<usize> {
        self.local_offsets[level]..self.local_offsets[level + 1]
    }

    /// Return the leaf indices of the ghost leaves, `nlocal()..len()`.
    pub fn ghosts(&self) -> Range<usize> {
        self.nlocal()..self.len()
    }

    /// Return the key of leaf `j`.
    pub fn key(&self, j: usize) -> MortonKey {
        self.keys[j]
    }

    /// Return the level of leaf `j`.
    pub fn level(&self, j: usize) -> usize {
        self.levels[j] as usize
    }

    /// Return the box index of leaf `j` on its level.
    pub fn box_index(&self, j: usize) -> u32 {
        self.boxes[j]
    }

    /// Return the keys of all leaves, by leaf index.
    pub fn keys(&self) -> &[MortonKey] {
        &self.keys
    }

    /// Return the levels of all leaves, by leaf index.
    pub fn levels(&self) -> &[u8] {
        &self.levels
    }

    /// Return the box indices of all leaves, by leaf index.
    pub fn boxes(&self) -> &[u32] {
        &self.boxes
    }
}

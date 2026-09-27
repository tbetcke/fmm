//! A manager for interactions of boxes.
//!
//! The interaction manager stores, for every non-ghost key of an [`Octree`], the
//! four classical FMM interaction lists. Two boxes are *adjacent* if their closed
//! cubes intersect, that is if they share at least a vertex, an edge or a face,
//! **and** neither box is an ancestor of the other. In particular a box is never
//! adjacent to itself and the root box is adjacent to nothing. Boxes of different
//! levels may be adjacent.
//!
//! For a box `B` with parent `P` the lists are
//!
//! - The U-list. Empty unless `B` is a leaf. All leaves of the tree that are
//!   adjacent to `B`. The box itself is **not** part of its own U-list.
//! - The V-list. Empty unless `B` has a parent. All boxes on the level of `B`
//!   whose parent is adjacent to `P` and that are not adjacent to `B`. This is
//!   the usual interaction list.
//! - The W-list. Empty unless `B` is a leaf. All boxes that are finer than `B`,
//!   whose parent is adjacent to `B`, and that are themselves not adjacent
//!   to `B`.
//! - The X-list. Empty unless `B` has a parent. All leaves at the level of `P`,
//!   that is the same-level neighbours of `P`, that are not adjacent to `B`. The
//!   X-list is the dual of the W-list, that is `N` is in the X-list of `B` if and
//!   only if `B` is in the W-list of `N`.
//!
//! These lists are stored as hash maps from a Morton key to the sorted vector of
//! the keys of the corresponding list.
//!
//! # Guarantees
//!
//! - Construction is purely **local**. No MPI collective is used, so a rank may
//!   build its interaction lists at any time and independently of other ranks.
//! - There is an entry in **all four** maps for **every** non-ghost key of
//!   [`Octree::all_keys`](nd_octree::Octree::all_keys), that is for every
//!   [`KeyType::LocalLeaf`], [`KeyType::LocalInterior`] and [`KeyType::Global`]
//!   key. A list that does not apply is present as an empty vector, never as a
//!   missing entry. Ghost keys receive no entry.
//! - Every list is sorted in ascending order, contains no duplicates and never
//!   contains the key it belongs to.
//!
//! # What the lists are, and are not
//!
//! When the tree carries the ghost-children layer — as it does when it is built
//! through [`FmmTree`](crate::fmm_tree::FmmTree) — every listed key is a key of
//! [`Octree::all_keys`](nd_octree::Octree::all_keys), carrying its own
//! [`KeyType`]. That layer is what makes the entries of the V- and W-lists,
//! which are children of neighbouring boxes, resolvable locally.
//! [`InteractionManager::new`] accepts any [`Octree`], however, and on a tree
//! built with [`OctreeOptions::default`](nd_octree::OctreeOptions::default) some
//! V- and W-list entries are generally absent from the key map.
//!
//! The listed keys are keys of the *global* tree, so many of them are ghosts of
//! the local rank. The manager provides topology only; moving the multipole or
//! particle data associated with those boxes to this rank remains the
//! responsibility of the caller.
//!
//! To learn which rank owns a listed key — the usual reason for wanting a V- or
//! W-list entry resolved — look the key up in
//! [`Octree::all_keys`](nd_octree::Octree::all_keys) and read
//! [`KeyType::ghost_rank`]. A [`KeyType::GhostLeaf`] or
//! [`KeyType::GhostInterior`] names its owning rank; a [`KeyType::LocalLeaf`] or
//! [`KeyType::LocalInterior`] key belongs to this rank; and a
//! [`KeyType::Global`] key has no single owner, since it exists on every rank.
//! [`Octree::owner_rank`](nd_octree::Octree::owner_rank) does **not** work for
//! these keys: it rejects anything that is not at the finest level, whereas list
//! entries are boxes at arbitrary levels.
//!
//! # Requirements on the tree
//!
//! The reductions used here rely on the tree being complete and 2:1 balanced
//! across all 26 neighbour directions, and on the ghost layer being a one-cell
//! same-level halo, as guaranteed by [`Octree`]. Together these imply that every
//! same-level neighbour cell of an interior box exists in the tree, and that two
//! adjacent leaves differ by at most one level.
//!
//! Resolvability of the list entries additionally requires the ghost-children
//! layer, that is a tree built with
//! [`OctreeOptions::with_ghost_children`](nd_octree::OctreeOptions::with_ghost_children):
//! for every non-ghost key and every same-level neighbour cell of it that the
//! tree holds as an interior box, all eight children of that neighbour must be
//! keys of the tree. [`InteractionManager::new`] checks the resulting lists
//! against [`Octree::all_keys`](nd_octree::Octree::all_keys) whenever the tree
//! reports that layer.

#[cfg(test)]
#[path = "interaction_manager_tests.rs"]
mod tests;

use std::collections::HashMap;

use mpi::traits::CommunicatorCollectives;
use nd_octree::{MortonKey, Octree, morton, octree::KeyType};

/// Manages interaction lists for boxes in an octree structure.
///
/// The interaction manager organizes boxes into different interaction
/// categories. See the [module documentation](self) for the precise definitions
/// and for the guarantees the lists satisfy.
pub struct InteractionManager {
    /// Maps each non-ghost key to the leaves adjacent to it (U-list).
    u_list: HashMap<MortonKey, Vec<MortonKey>>,

    /// Maps each non-ghost key to the boxes on its own level whose parent is
    /// adjacent to its parent and that are not adjacent to it (V-list).
    v_list: HashMap<MortonKey, Vec<MortonKey>>,

    /// Maps each non-ghost key to the leaves on the level of its parent, that is
    /// the same-level neighbours of the parent, that are not adjacent to it
    /// (X-list).
    x_list: HashMap<MortonKey, Vec<MortonKey>>,

    /// Maps each non-ghost key to the finer boxes whose parent is adjacent to it
    /// and that are not adjacent to it (W-list).
    w_list: HashMap<MortonKey, Vec<MortonKey>>,
}

impl InteractionManager {
    /// Create a new `InteractionManager` from an octree.
    ///
    /// Computes the U-, V-, W- and X-lists of every non-ghost key of
    /// `octree`. The computation is local to this rank and uses **no**
    /// collective operation, so it need not be entered by all ranks.
    ///
    /// # Arguments
    ///
    /// * `octree` - The distributed octree whose local keys are classified.
    ///
    /// # Panics
    ///
    /// Panics if `octree` was built with the ghost-children layer but a list
    /// entry is not a key of
    /// [`Octree::all_keys`](nd_octree::Octree::all_keys), which would mean the
    /// layer did not deliver its guarantee.
    pub fn new<C: CommunicatorCollectives>(octree: &Octree<'_, C>) -> Self {
        let all_keys = octree.all_keys();
        let manager = Self::from_key_types(all_keys);

        // The guarantee only holds for a tree that carries the ghost-children
        // layer, so a tree built without it is left unchecked.
        if octree.options().ghost_children() {
            for map in [
                &manager.u_list,
                &manager.v_list,
                &manager.w_list,
                &manager.x_list,
            ] {
                for entry in map.values().flatten() {
                    assert!(
                        all_keys.contains_key(entry),
                        "the ghost-children layer must make every list entry a key of the tree"
                    );
                }
            }
        }

        manager
    }

    /// Return the U-lists, keyed by the non-ghost keys of the tree.
    pub fn u_list(&self) -> &HashMap<MortonKey, Vec<MortonKey>> {
        &self.u_list
    }

    /// Return the V-lists, keyed by the non-ghost keys of the tree.
    pub fn v_list(&self) -> &HashMap<MortonKey, Vec<MortonKey>> {
        &self.v_list
    }

    /// Return the W-lists, keyed by the non-ghost keys of the tree.
    pub fn w_list(&self) -> &HashMap<MortonKey, Vec<MortonKey>> {
        &self.w_list
    }

    /// Return the X-lists, keyed by the non-ghost keys of the tree.
    pub fn x_list(&self) -> &HashMap<MortonKey, Vec<MortonKey>> {
        &self.x_list
    }

    /// Build all four lists from a key classification map.
    ///
    /// Requires the map to describe a complete, 2:1 (26-neighbour) balanced tree
    /// together with its one-cell same-level ghost halo, as
    /// [`Octree::all_keys`](nd_octree::Octree::all_keys) provides.
    fn from_key_types(all_keys: &HashMap<MortonKey, KeyType>) -> Self {
        let mut u_list = HashMap::<MortonKey, Vec<MortonKey>>::new();
        let mut v_list = HashMap::<MortonKey, Vec<MortonKey>>::new();
        let mut w_list = HashMap::<MortonKey, Vec<MortonKey>>::new();
        let mut x_list = HashMap::<MortonKey, Vec<MortonKey>>::new();

        // Each key is treated independently, so the iteration order of the map
        // does not matter.
        for (&key, &key_type) in all_keys.iter() {
            if key_type.is_ghost() {
                continue;
            }

            let mut u = Vec::<MortonKey>::new();
            let mut v = Vec::<MortonKey>::new();
            let mut w = Vec::<MortonKey>::new();
            let mut x = Vec::<MortonKey>::new();

            // The U- and W-lists are derived from the same-level neighbour cells
            // of a leaf. Both are empty for interior keys.
            if is_leaf(key_type) {
                for neighbour in valid_neighbours(key) {
                    match all_keys.get(&neighbour) {
                        // The neighbour cell is a leaf of the tree. It is
                        // adjacent to `key` and hence in the U-list.
                        Some(&neighbour_type) if is_leaf(neighbour_type) => u.push(neighbour),
                        // The neighbour cell is refined. Its children adjacent to
                        // `key` are leaves by 2:1 balance and form part of the
                        // U-list, the remaining children form part of the
                        // W-list. Deeper descendants cannot contribute: their
                        // parents are not adjacent to `key`.
                        Some(&neighbour_type) if is_interior(neighbour_type) => {
                            for child in children_of(neighbour) {
                                if is_adjacent(child, key) {
                                    u.push(child);
                                } else {
                                    w.push(child);
                                }
                            }
                        }
                        Some(_) => {
                            unreachable!("KeyType is partitioned into leaf and interior variants")
                        }
                        // The neighbour cell is not in the tree, so it is covered
                        // by a coarser leaf. By 2:1 balance that leaf is the
                        // parent of the cell. Several cells may share it.
                        None => {
                            let parent = morton::parent(neighbour)
                                .expect("a neighbour cell is never the root");
                            debug_assert!(
                                all_keys.get(&parent).copied().is_some_and(is_leaf),
                                "the parent of a missing neighbour cell must be a local leaf"
                            );
                            u.push(parent);
                        }
                    }
                }
            }

            // The V- and X-lists are derived from the same-level neighbour cells
            // of the parent. Both are empty for the root.
            if let Some(parent) = morton::parent(key) {
                for neighbour in valid_neighbours(parent) {
                    // The parent is interior, so by 2:1 balance all its
                    // same-level neighbour cells exist, and by the halo property
                    // they are all known locally. A miss means that assumption
                    // broke, which must not be papered over.
                    let neighbour_type = *all_keys
                        .get(&neighbour)
                        .expect("a same-level neighbour of an interior key must be known locally");
                    if is_interior(neighbour_type) {
                        // All children of the neighbour exist in the tree. Those
                        // not adjacent to `key` form the V-list.
                        for child in children_of(neighbour) {
                            if !is_adjacent(child, key) {
                                v.push(child);
                            }
                        }
                    } else if !is_adjacent(neighbour, key) {
                        // A leaf neighbour of the parent that does not touch
                        // `key` is an X-list entry and contributes nothing to
                        // the V-list.
                        x.push(neighbour);
                    }
                }
            }

            finish(&mut u);
            finish(&mut v);
            finish(&mut w);
            finish(&mut x);

            u_list.insert(key, u);
            v_list.insert(key, v);
            w_list.insert(key, w);
            x_list.insert(key, x);
        }

        Self {
            u_list,
            v_list,
            x_list,
            w_list,
        }
    }
}

/// Return true if the key type denotes an interior box of the tree.
fn is_interior(key_type: KeyType) -> bool {
    matches!(
        key_type,
        KeyType::LocalInterior | KeyType::GhostInterior(_) | KeyType::Global
    )
}

/// Return true if the key type denotes a leaf of the tree.
fn is_leaf(key_type: KeyType) -> bool {
    matches!(key_type, KeyType::LocalLeaf | KeyType::GhostLeaf(_))
}

/// Iterate over the in-domain same-level neighbour cells of `key`.
fn valid_neighbours(key: MortonKey) -> impl Iterator<Item = MortonKey> {
    morton::neighbours(key)
        .into_iter()
        .filter(|&neighbour| morton::is_valid(neighbour))
}

/// Return the children of an interior key.
fn children_of(key: MortonKey) -> [MortonKey; 8] {
    morton::children(key).expect("an interior key is not on the deepest level")
}

/// Return true if the closed cubes of `a` and `b` touch without one containing
/// the other.
fn is_adjacent(a: MortonKey, b: MortonKey) -> bool {
    if morton::is_ancestor(a, b) || morton::is_ancestor(b, a) {
        return false;
    }

    let (a_level, a_index) = morton::decode(a);
    let (b_level, b_index) = morton::decode(b);

    // Lift both boxes to the finer of the two levels, where each box covers a
    // closed range of cells.
    let level = a_level.max(b_level);
    let a_shift = level - a_level;
    let b_shift = level - b_level;

    for dim in 0..3 {
        let a_min = a_index[dim] << a_shift;
        let a_max = a_min + (1 << a_shift) - 1;
        let b_min = b_index[dim] << b_shift;
        let b_max = b_min + (1 << b_shift) - 1;
        // The boxes touch in this dimension if the closed ranges overlap after
        // growing one of them by a single cell.
        if a_min > b_max + 1 || b_min > a_max + 1 {
            return false;
        }
    }

    true
}

/// Sort a list ascendingly and remove duplicates.
fn finish(list: &mut Vec<MortonKey>) {
    list.sort_unstable();
    list.dedup();
}

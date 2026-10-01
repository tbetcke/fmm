//! The interaction-list rule and the V-list offsets.
//!
//! This module defines the four classical FMM interaction lists of a box, and computes
//! them for one key of a key map (the private `key_lists`). [`Plan`](crate::plan::Plan)
//! applies that rule to every non-ghost key of an [`Octree`](nd_octree::Octree) and stores the result as
//! index arrays ([`lists`](crate::lists)). [`V_LIST_DIRECTIONS`] lists the 316 offsets of
//! a V-list pair, in the order that the offset index of a V-list batch refers to
//! (CONVENTIONS §3.12).
//!
//! Two boxes are *adjacent* if their closed cubes intersect, that is if they share at
//! least a vertex, an edge or a face, **and** neither box is an ancestor of the other.
//! In particular a box is never adjacent to itself and the root box is adjacent to
//! nothing. Boxes of different levels may be adjacent.
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
//! # Guarantees
//!
//! - The rule is purely **local**: it reads only the key map
//!   ([`Octree::all_keys`](nd_octree::Octree::all_keys)) and uses no MPI collective.
//! - Lists are computed for non-ghost keys ([`KeyType::LocalLeaf`],
//!   [`KeyType::LocalInterior`] and [`KeyType::Global`]). A list that does not apply is
//!   empty.
//! - Every list is sorted in ascending key order, contains no duplicates and never
//!   contains the key it belongs to.
//!
//! # What the lists are, and are not
//!
//! When the tree carries the ghost-children layer, that is when it is built
//! with
//! [`OctreeOptions::with_ghost_children`](nd_octree::OctreeOptions::with_ghost_children),
//! every listed key is a key of
//! [`Octree::all_keys`](nd_octree::Octree::all_keys), carrying its own
//! [`KeyType`]. That layer is what makes the entries of the V- and W-lists,
//! which are children of neighbouring boxes, resolvable locally;
//! [`Plan::new`](crate::plan::Plan::new) requires it.
//!
//! The listed keys are keys of the *global* tree, so many of them are ghosts of the
//! local rank. A [`KeyType::GhostLeaf`] or [`KeyType::GhostInterior`] names its owning
//! rank ([`KeyType::ghost_rank`]); a [`KeyType::Global`] key has no single owner, since
//! it exists on every rank. The rule provides topology only; moving the data of those
//! boxes is the job of [`exchange`](crate::exchange).
//!
//! # Requirements on the tree
//!
//! The reductions used here rely on the tree being complete and 2:1 balanced
//! across all 26 neighbour directions, and on the ghost layer being a one-cell
//! same-level halo, as guaranteed by [`Octree`](nd_octree::Octree). Together these imply that every
//! same-level neighbour cell of an interior box exists in the tree, and that two
//! adjacent leaves differ by at most one level.
//!
//! Resolvability of the list entries additionally requires the ghost-children
//! layer: for every non-ghost key and every same-level neighbour cell of it that the
//! tree holds as an interior box, all eight children of that neighbour must be
//! keys of the tree.

#[cfg(test)]
#[path = "interaction_manager_tests.rs"]
pub(crate) mod tests;

use std::collections::HashMap;

use nd_octree::{MortonKey, morton, octree::KeyType};

/// Offsets `target - source` of every possible V-list interaction in three
/// dimensions, in lexicographic `(x, y, z)` order.
///
/// A V-list source and its target lie on the same level, so the offset is
/// measured in index units of that level. The parents of source and target are
/// adjacent, which bounds every component by 3, and source and target are not
/// adjacent, which forces at least one component of magnitude 2 or 3. All
/// `7^3 - 3^3 = 316` such vectors occur.
pub const V_LIST_DIRECTIONS: [[i64; 3]; 316] = [
    // x = -3
    [-3, -3, -3],
    [-3, -3, -2],
    [-3, -3, -1],
    [-3, -3, 0],
    [-3, -3, 1],
    [-3, -3, 2],
    [-3, -3, 3],
    [-3, -2, -3],
    [-3, -2, -2],
    [-3, -2, -1],
    [-3, -2, 0],
    [-3, -2, 1],
    [-3, -2, 2],
    [-3, -2, 3],
    [-3, -1, -3],
    [-3, -1, -2],
    [-3, -1, -1],
    [-3, -1, 0],
    [-3, -1, 1],
    [-3, -1, 2],
    [-3, -1, 3],
    [-3, 0, -3],
    [-3, 0, -2],
    [-3, 0, -1],
    [-3, 0, 0],
    [-3, 0, 1],
    [-3, 0, 2],
    [-3, 0, 3],
    [-3, 1, -3],
    [-3, 1, -2],
    [-3, 1, -1],
    [-3, 1, 0],
    [-3, 1, 1],
    [-3, 1, 2],
    [-3, 1, 3],
    [-3, 2, -3],
    [-3, 2, -2],
    [-3, 2, -1],
    [-3, 2, 0],
    [-3, 2, 1],
    [-3, 2, 2],
    [-3, 2, 3],
    [-3, 3, -3],
    [-3, 3, -2],
    [-3, 3, -1],
    [-3, 3, 0],
    [-3, 3, 1],
    [-3, 3, 2],
    [-3, 3, 3],
    // x = -2
    [-2, -3, -3],
    [-2, -3, -2],
    [-2, -3, -1],
    [-2, -3, 0],
    [-2, -3, 1],
    [-2, -3, 2],
    [-2, -3, 3],
    [-2, -2, -3],
    [-2, -2, -2],
    [-2, -2, -1],
    [-2, -2, 0],
    [-2, -2, 1],
    [-2, -2, 2],
    [-2, -2, 3],
    [-2, -1, -3],
    [-2, -1, -2],
    [-2, -1, -1],
    [-2, -1, 0],
    [-2, -1, 1],
    [-2, -1, 2],
    [-2, -1, 3],
    [-2, 0, -3],
    [-2, 0, -2],
    [-2, 0, -1],
    [-2, 0, 0],
    [-2, 0, 1],
    [-2, 0, 2],
    [-2, 0, 3],
    [-2, 1, -3],
    [-2, 1, -2],
    [-2, 1, -1],
    [-2, 1, 0],
    [-2, 1, 1],
    [-2, 1, 2],
    [-2, 1, 3],
    [-2, 2, -3],
    [-2, 2, -2],
    [-2, 2, -1],
    [-2, 2, 0],
    [-2, 2, 1],
    [-2, 2, 2],
    [-2, 2, 3],
    [-2, 3, -3],
    [-2, 3, -2],
    [-2, 3, -1],
    [-2, 3, 0],
    [-2, 3, 1],
    [-2, 3, 2],
    [-2, 3, 3],
    // x = -1
    [-1, -3, -3],
    [-1, -3, -2],
    [-1, -3, -1],
    [-1, -3, 0],
    [-1, -3, 1],
    [-1, -3, 2],
    [-1, -3, 3],
    [-1, -2, -3],
    [-1, -2, -2],
    [-1, -2, -1],
    [-1, -2, 0],
    [-1, -2, 1],
    [-1, -2, 2],
    [-1, -2, 3],
    [-1, -1, -3],
    [-1, -1, -2],
    [-1, -1, 2],
    [-1, -1, 3],
    [-1, 0, -3],
    [-1, 0, -2],
    [-1, 0, 2],
    [-1, 0, 3],
    [-1, 1, -3],
    [-1, 1, -2],
    [-1, 1, 2],
    [-1, 1, 3],
    [-1, 2, -3],
    [-1, 2, -2],
    [-1, 2, -1],
    [-1, 2, 0],
    [-1, 2, 1],
    [-1, 2, 2],
    [-1, 2, 3],
    [-1, 3, -3],
    [-1, 3, -2],
    [-1, 3, -1],
    [-1, 3, 0],
    [-1, 3, 1],
    [-1, 3, 2],
    [-1, 3, 3],
    // x = 0
    [0, -3, -3],
    [0, -3, -2],
    [0, -3, -1],
    [0, -3, 0],
    [0, -3, 1],
    [0, -3, 2],
    [0, -3, 3],
    [0, -2, -3],
    [0, -2, -2],
    [0, -2, -1],
    [0, -2, 0],
    [0, -2, 1],
    [0, -2, 2],
    [0, -2, 3],
    [0, -1, -3],
    [0, -1, -2],
    [0, -1, 2],
    [0, -1, 3],
    [0, 0, -3],
    [0, 0, -2],
    [0, 0, 2],
    [0, 0, 3],
    [0, 1, -3],
    [0, 1, -2],
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, -3],
    [0, 2, -2],
    [0, 2, -1],
    [0, 2, 0],
    [0, 2, 1],
    [0, 2, 2],
    [0, 2, 3],
    [0, 3, -3],
    [0, 3, -2],
    [0, 3, -1],
    [0, 3, 0],
    [0, 3, 1],
    [0, 3, 2],
    [0, 3, 3],
    // x = 1
    [1, -3, -3],
    [1, -3, -2],
    [1, -3, -1],
    [1, -3, 0],
    [1, -3, 1],
    [1, -3, 2],
    [1, -3, 3],
    [1, -2, -3],
    [1, -2, -2],
    [1, -2, -1],
    [1, -2, 0],
    [1, -2, 1],
    [1, -2, 2],
    [1, -2, 3],
    [1, -1, -3],
    [1, -1, -2],
    [1, -1, 2],
    [1, -1, 3],
    [1, 0, -3],
    [1, 0, -2],
    [1, 0, 2],
    [1, 0, 3],
    [1, 1, -3],
    [1, 1, -2],
    [1, 1, 2],
    [1, 1, 3],
    [1, 2, -3],
    [1, 2, -2],
    [1, 2, -1],
    [1, 2, 0],
    [1, 2, 1],
    [1, 2, 2],
    [1, 2, 3],
    [1, 3, -3],
    [1, 3, -2],
    [1, 3, -1],
    [1, 3, 0],
    [1, 3, 1],
    [1, 3, 2],
    [1, 3, 3],
    // x = 2
    [2, -3, -3],
    [2, -3, -2],
    [2, -3, -1],
    [2, -3, 0],
    [2, -3, 1],
    [2, -3, 2],
    [2, -3, 3],
    [2, -2, -3],
    [2, -2, -2],
    [2, -2, -1],
    [2, -2, 0],
    [2, -2, 1],
    [2, -2, 2],
    [2, -2, 3],
    [2, -1, -3],
    [2, -1, -2],
    [2, -1, -1],
    [2, -1, 0],
    [2, -1, 1],
    [2, -1, 2],
    [2, -1, 3],
    [2, 0, -3],
    [2, 0, -2],
    [2, 0, -1],
    [2, 0, 0],
    [2, 0, 1],
    [2, 0, 2],
    [2, 0, 3],
    [2, 1, -3],
    [2, 1, -2],
    [2, 1, -1],
    [2, 1, 0],
    [2, 1, 1],
    [2, 1, 2],
    [2, 1, 3],
    [2, 2, -3],
    [2, 2, -2],
    [2, 2, -1],
    [2, 2, 0],
    [2, 2, 1],
    [2, 2, 2],
    [2, 2, 3],
    [2, 3, -3],
    [2, 3, -2],
    [2, 3, -1],
    [2, 3, 0],
    [2, 3, 1],
    [2, 3, 2],
    [2, 3, 3],
    // x = 3
    [3, -3, -3],
    [3, -3, -2],
    [3, -3, -1],
    [3, -3, 0],
    [3, -3, 1],
    [3, -3, 2],
    [3, -3, 3],
    [3, -2, -3],
    [3, -2, -2],
    [3, -2, -1],
    [3, -2, 0],
    [3, -2, 1],
    [3, -2, 2],
    [3, -2, 3],
    [3, -1, -3],
    [3, -1, -2],
    [3, -1, -1],
    [3, -1, 0],
    [3, -1, 1],
    [3, -1, 2],
    [3, -1, 3],
    [3, 0, -3],
    [3, 0, -2],
    [3, 0, -1],
    [3, 0, 0],
    [3, 0, 1],
    [3, 0, 2],
    [3, 0, 3],
    [3, 1, -3],
    [3, 1, -2],
    [3, 1, -1],
    [3, 1, 0],
    [3, 1, 1],
    [3, 1, 2],
    [3, 1, 3],
    [3, 2, -3],
    [3, 2, -2],
    [3, 2, -1],
    [3, 2, 0],
    [3, 2, 1],
    [3, 2, 2],
    [3, 2, 3],
    [3, 3, -3],
    [3, 3, -2],
    [3, 3, -1],
    [3, 3, 0],
    [3, 3, 1],
    [3, 3, 2],
    [3, 3, 3],
];

/// Compute the U-, V-, W- and X-lists of the non-ghost `key`, in that order.
///
/// This is the per-key rule that [`Plan`](crate::plan::Plan) applies to every non-ghost
/// key. Each list is sorted ascending and deduplicated.
/// The rule requires `all_keys` to describe a complete, 2:1 (26-neighbour)
/// balanced tree together with its one-cell same-level ghost halo, as
/// [`Octree::all_keys`](nd_octree::Octree::all_keys) provides; it panics if a
/// same-level neighbour of the parent of `key` is not a key of the map.
pub(crate) fn key_lists(
    key: MortonKey,
    key_type: KeyType,
    all_keys: &HashMap<MortonKey, KeyType>,
) -> [Vec<MortonKey>; 4] {
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
                    let parent =
                        morton::parent(neighbour).expect("a neighbour cell is never the root");
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

    [u, v, w, x]
}

/// Return true if the key type denotes an interior box of the tree.
pub(crate) fn is_interior(key_type: KeyType) -> bool {
    matches!(
        key_type,
        KeyType::LocalInterior | KeyType::GhostInterior(_) | KeyType::Global
    )
}

/// Return true if the key type denotes a leaf of the tree.
pub(crate) fn is_leaf(key_type: KeyType) -> bool {
    matches!(key_type, KeyType::LocalLeaf | KeyType::GhostLeaf(_))
}

/// Iterate over the in-domain same-level neighbour cells of `key`.
pub(crate) fn valid_neighbours(key: MortonKey) -> impl Iterator<Item = MortonKey> {
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

//! Test the computation of a complete octree.

use itertools::Itertools;
use mpi::traits::Communicator;
use nd_octree::{
    constants::DEEPEST_LEVEL,
    morton::{self},
    octree::{
        KeyType, Octree, OctreeOptions, compute_global_bounding_box,
        is_complete_linear_and_balanced,
    },
    points_to_morton,
};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rlst::{distributed_tools::array_tools::gather_to_all, rlst_dynamic_array};
use std::collections::HashMap;

pub fn main() {
    // Initialise MPI
    let universe = mpi::initialize().unwrap();

    // Get the world communicator
    let comm = universe.world();

    // Initialise a seeded Rng.
    let mut rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);

    // Create `npoints` per rank.
    let npoints = 1000;

    // Generate random points.

    let mut points = rlst_dynamic_array!(f64, [3, npoints]);

    points.fill_from_standard_normal(&mut rng);

    // Make sure that the points live on the unit sphere.
    for mut point in points.col_iter_mut() {
        let len = point.norm_2().unwrap();
        point /= len;
    }

    let bounding_box = compute_global_bounding_box(&points, &comm);

    let fine_keys = points_to_morton(&points, DEEPEST_LEVEL as usize, &bounding_box);

    let options = OctreeOptions::new()
        .with_max_level(15)
        .with_max_fine_keys(50);
    let tree = Octree::new(&fine_keys, options, &comm);

    // We now check that each node of the tree has all its neighbors available.

    let leaf_tree = tree.leaf_keys();
    let all_keys = tree.all_keys();

    assert!(is_complete_linear_and_balanced(leaf_tree, &comm));
    for &key in leaf_tree {
        // We only check interior keys. Leaf keys may not have a neighbor
        // on the same level.
        let mut parent = morton::parent(key).unwrap();
        while morton::level(parent) > 0 {
            // Check that the key itself is there.
            assert!(all_keys.contains_key(&key));
            // Check that all its neighbours are there.
            for neighbor in morton::neighbours(parent)
                .iter()
                .filter(|&key| morton::is_valid(*key))
            {
                assert!(all_keys.contains_key(neighbor));
            }
            parent = morton::parent(parent).unwrap();
            // Check that the parent is there.
            assert!(all_keys.contains_key(&parent));
        }
    }

    // At the end check that the root of the tree is also contained.
    assert!(all_keys.contains_key(&morton::root()));

    // Count the number of ghosts on each rank
    // Count the number of global keys on each rank.

    // Assert that all ghosts are from a different rank and count them.

    let nghosts = all_keys
        .iter()
        .filter_map(|(_, &value)| value.ghost_rank())
        .count();

    if comm.size() == 1 {
        assert_eq!(nghosts, 0);
    } else {
        assert!(nghosts > 0);
    }

    let nglobal = all_keys
        .iter()
        .filter(|(_, value)| matches!(**value, KeyType::Global))
        .count();

    // Assert that all globals across all ranks have the same count.

    let nglobals = gather_to_all(std::slice::from_ref(&nglobal), &comm);

    assert_eq!(nglobals.iter().unique().count(), 1);

    // Check the neighbour relationships.

    let all_neighbours = tree.neighbour_map();
    let all_keys = tree.all_keys();

    for (key, key_type) in all_keys {
        // Ghost keys should not be in the neighbour map.
        match key_type {
            KeyType::GhostLeaf(_) => assert!(!all_neighbours.contains_key(key)),
            KeyType::GhostInterior(_) => assert!(!all_neighbours.contains_key(key)),
            _ => {
                // If it is not a ghost the key should be in the neighbour map.
                assert!(all_neighbours.contains_key(key));
            }
        }
    }

    // Build the same tree once more, with the ghost-children layer. Construction is
    // deterministic (the sort is seeded with the rank), so the two trees describe
    // the same topology and may be compared key by key.

    let ghost_tree = Octree::new(&fine_keys, options.with_ghost_children(true), &comm);
    let ghost_keys = ghost_tree.all_keys();

    // The guarantee: every child of an interior same-level neighbour of a non-ghost
    // key is itself a key of the tree. This needs no communication.

    for (&key, &key_type) in ghost_keys {
        if key_type.is_ghost() {
            continue;
        }
        for neighbour in interior_neighbours(ghost_keys, key) {
            for child in morton::children(neighbour).unwrap() {
                assert!(ghost_keys.contains_key(&child));
            }
        }
    }

    // The layer only adds keys: every baseline key is present with the same type.

    for (key, key_type) in all_keys {
        assert_eq!(ghost_keys.get(key), Some(key_type));
    }

    // Neighbour maps and the set of global keys are untouched by the layer.

    assert_eq!(tree.neighbour_map(), ghost_tree.neighbour_map());

    let globals = |keys: &HashMap<morton::MortonKey, KeyType>| {
        keys.iter()
            .filter(|&(_, &value)| value == KeyType::Global)
            .map(|(&key, _)| key)
            .sorted()
            .collect_vec()
    };
    assert_eq!(globals(all_keys), globals(ghost_keys));
    let nglobals = gather_to_all(std::slice::from_ref(&globals(ghost_keys).len()), &comm);
    assert_eq!(nglobals.iter().unique().count(), 1);

    // The layer also replicates every coarse block of every rank, with the leaf or
    // interior classification and the owner its own rank assigns. Every rank enters
    // the gathers, also on one rank.

    let own_blocks = ghost_tree.coarse_tree_leafs();
    let own_block_is_leaf = own_blocks
        .iter()
        .map(|block| ghost_keys[block] == KeyType::LocalLeaf)
        .collect_vec();
    let block_counts = gather_to_all(std::slice::from_ref(&own_blocks.len()), &comm);
    let blocks = gather_to_all(own_blocks, &comm);
    let block_is_leaf = gather_to_all(&own_block_is_leaf, &comm);
    let owners = block_counts
        .iter()
        .enumerate()
        .flat_map(|(owner, &count)| std::iter::repeat_n(owner, count))
        .collect_vec();
    let rank = comm.rank() as usize;
    let mut supplied_blocks = 0;
    for ((block, &is_leaf), &owner) in blocks.iter().zip(&block_is_leaf).zip(&owners) {
        let key_type = ghost_keys.get(block).copied();
        if owner == rank {
            // On one rank the only block is the root, which is `Global` unless it
            // is the only leaf.
            assert!(matches!(
                key_type,
                Some(KeyType::LocalLeaf | KeyType::LocalInterior | KeyType::Global)
            ));
        } else if is_leaf {
            assert_eq!(key_type, Some(KeyType::GhostLeaf(owner)));
        } else {
            assert_eq!(key_type, Some(KeyType::GhostInterior(owner)));
        }
        if !all_keys.contains_key(block) {
            supplied_blocks += 1;
        }
    }

    // Hence every child of every `Global` key is a key of the tree.

    for (&key, &key_type) in ghost_keys {
        if key_type != KeyType::Global {
            continue;
        }
        for child in morton::children(key).unwrap() {
            assert!(ghost_keys.contains_key(&child));
        }
    }
    println!(
        "rank {}: {} coarse blocks, {supplied_blocks} of them supplied by the ghost-children layer",
        comm.rank(),
        blocks.len()
    );

    // Non-vacuity: on more than one rank the baseline tree is missing children that
    // the layer supplies. The count is reported so a silently vacuous run is visible.

    let mut missing = 0;
    for (&key, &key_type) in all_keys {
        if key_type.is_ghost() {
            continue;
        }
        for neighbour in interior_neighbours(all_keys, key) {
            missing += morton::children(neighbour)
                .unwrap()
                .iter()
                .filter(|child| !all_keys.contains_key(child))
                .count();
        }
    }
    println!(
        "rank {}: {missing} child slots are unresolved without the ghost-children layer",
        comm.rank()
    );
    if comm.size() == 1 {
        // One rank holds the whole tree, so nothing can be missing and the option
        // is a no-op.
        assert_eq!(missing, 0);
        assert_eq!(all_keys, ghost_keys);
    } else {
        // Both quantities are global: with a different input or partitioner an
        // individual rank may legitimately have nothing unresolved, so the sums
        // over all ranks are what must be non-zero. Every rank enters the gathers.
        let missing_per_rank = gather_to_all(std::slice::from_ref(&missing), &comm);
        let growth = ghost_keys.len() - all_keys.len();
        let growth_per_rank = gather_to_all(std::slice::from_ref(&growth), &comm);
        assert!(missing_per_rank.iter().sum::<usize>() > 0);
        assert!(growth_per_rank.iter().sum::<usize>() > 0);
    }

    if comm.rank() == 0 {
        println!("No errors were found in setting up tree.");
    }
}

/// The same-level neighbour cells of `key` that the tree holds as interior boxes.
fn interior_neighbours(
    all_keys: &HashMap<morton::MortonKey, KeyType>,
    key: morton::MortonKey,
) -> impl Iterator<Item = morton::MortonKey> {
    morton::neighbours(key)
        .into_iter()
        .filter(|&neighbour| morton::is_valid(neighbour))
        .filter(|neighbour| {
            all_keys.get(neighbour).is_some_and(|key_type| {
                !matches!(key_type, KeyType::LocalLeaf | KeyType::GhostLeaf(_))
            })
        })
        .collect_vec()
        .into_iter()
}

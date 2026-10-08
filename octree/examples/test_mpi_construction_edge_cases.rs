//! Construction inputs at the edge of the supported range.
//!
//! Covers a rank without keys, duplicate keys straddling rank boundaries, a
//! heavy cluster that cannot be split by the coarse tree, `max_level` being
//! honoured on every rank count, and distinct neighbour entries.

use itertools::Itertools;
use mpi::traits::Communicator;
use nd_octree::{
    Octree, OctreeOptions,
    constants::DEEPEST_LEVEL,
    morton,
    octree::{KeyType, is_complete_linear_and_balanced},
    tools::{generate_random_keys, seeded_rng},
};
use rand::RngExt;
use rlst::distributed_tools::array_tools::gather_to_all;
use std::collections::HashSet;

/// Build a tree and check the invariants every construction must satisfy.
fn check_tree<'c, C: mpi::traits::CommunicatorCollectives>(
    fine_keys: &[morton::MortonKey],
    options: OctreeOptions,
    comm: &'c C,
) -> Octree<'c, C> {
    let tree = Octree::new(fine_keys, options, comm);
    let max_level = options.max_level();
    let rank = comm.rank() as usize;

    assert!(is_complete_linear_and_balanced(tree.leaf_keys(), comm));
    // A rank owns leaves exactly when it owns coarse blocks, and some rank does. A
    // rank without blocks is allowed (Phase 5 T4): its bound is never an owner's.
    assert_eq!(
        tree.leaf_keys().is_empty(),
        tree.coarse_tree_leafs().is_empty()
    );
    let nblocks = gather_to_all(&[tree.coarse_tree_leafs().len()], comm);
    assert!(nblocks.iter().any(|&n| n > 0));
    for leaf in tree.leaf_keys() {
        let first = morton::from_index_and_level(
            morton::decode(*leaf)
                .1
                .map(|i| i << (DEEPEST_LEVEL as usize - morton::level(*leaf))),
            DEEPEST_LEVEL as usize,
        );
        assert_eq!(tree.owner_rank(first), Ok(rank));
    }
    // No leaf is deeper than requested, whatever the rank count.
    assert!(tree.global_max_level() <= max_level.min(DEEPEST_LEVEL as usize));
    // Neighbour lists carry each key once, and only non-ghost keys have lists.
    for (key, neighbours) in tree.neighbour_map() {
        assert!(!tree.all_keys()[key].is_ghost());
        assert_eq!(neighbours.iter().unique().count(), neighbours.len());
    }
    // Every input key is found on its owner.
    let queries = fine_keys.iter().copied().take(64).collect_vec();
    let found = tree.lookup_leaves(&queries).unwrap();
    for (&query, location) in queries.iter().zip(found) {
        let location = location.unwrap();
        assert!(morton::is_ancestor(location.leaf, query));
        assert_eq!(tree.owner_rank(query), Ok(location.owner_rank));
        if location.owner_rank == rank {
            assert!(tree.leaf_keys().contains(&location.leaf));
        }
    }
    tree
}

/// Check the partition's bound: every rank's weight (the input keys it owns,
/// duplicates included) is at most a fair share plus the heaviest coarse block.
///
/// # Collective operation
/// Gathers the keys and the blocks of every rank, so every rank must call it.
fn check_weight_bound<C: mpi::traits::CommunicatorCollectives>(
    tree: &Octree<'_, C>,
    fine_keys: &[morton::MortonKey],
    comm: &C,
) {
    let size = comm.size() as usize;
    let mut keys = gather_to_all(fine_keys, comm);
    keys.sort_unstable();
    let blocks = gather_to_all(tree.coarse_tree_leafs(), comm);
    let heaviest = blocks
        .iter()
        .map(|&block| {
            keys.iter()
                .filter(|&&key| morton::is_ancestor(block, key))
                .count()
        })
        .max()
        .unwrap();
    let mut per_rank = vec![0; size];
    for &key in &keys {
        per_rank[tree.owner_rank(key).unwrap()] += 1;
    }
    let total = keys.len();
    for &weight in &per_rank {
        assert!(weight * size <= total + heaviest * size);
    }
}

/// Check that every ghost of `tree` is consistent with the rank it names: it
/// names another rank, it is an ancestor-or-equal of a leaf that rank owns, and
/// it is flagged as a leaf exactly when it is one of that rank's leaves.
///
/// # Collective operation
/// Gathers the leaves of every rank, so every rank must call it.
fn check_ghost_bookkeeping<C: mpi::traits::CommunicatorCollectives>(
    tree: &Octree<'_, C>,
    comm: &C,
) {
    let rank = comm.rank() as usize;
    let owned = gather_to_all(tree.leaf_keys(), comm);
    let owner_ranks = gather_to_all(&vec![rank; tree.leaf_keys().len()], comm);
    let owned = owned.into_iter().zip(owner_ranks).collect_vec();
    for (&key, &key_type) in tree.all_keys() {
        if let Some(ghost_rank) = key_type.ghost_rank() {
            assert_ne!(ghost_rank, rank);
            let is_leaf = matches!(key_type, KeyType::GhostLeaf(_));
            let covered = owned
                .iter()
                .any(|&(leaf, owner)| owner == ghost_rank && morton::is_ancestor(key, leaf));
            assert!(covered);
            assert_eq!(is_leaf, owned.contains(&(key, ghost_rank)));
        }
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let deepest = DEEPEST_LEVEL as usize;

    // A rank contributing no keys still takes part and owns part of the tree.
    let mut rng = seeded_rng(rank);
    let keys = if rank == 1 {
        Vec::new()
    } else {
        generate_random_keys(1000, &mut rng)
    };
    check_tree(
        &keys,
        OctreeOptions::new()
            .with_max_level(12)
            .with_max_fine_keys(4),
        &comm,
    );

    // One distinct key per rank, each duplicated: the parallel sort splits runs of
    // equal keys across ranks, and load balancing must still give every rank a block.
    // The spacing comes from the rank count, so that every index stays within the
    // 2^16 cells of level 16 on any number of ranks.
    let spacing = (1 << deepest) / (size + 1);
    let key = morton::from_index_and_level([spacing * (rank + 1); 3], deepest);
    let tree = check_tree(&[key, key, key, key], OctreeOptions::default(), &comm);
    let leaves = gather_to_all(tree.leaf_keys(), &comm);
    assert!(morton::is_complete_linear_octree(&leaves));
    // The duplicates collapse to one key, so exactly `size` leaves hold a key.
    let all_keys = gather_to_all(&[key], &comm);
    assert_eq!(
        leaves
            .iter()
            .filter(|&&leaf| all_keys.iter().any(|&k| morton::is_ancestor(leaf, k)))
            .count(),
        size
    );

    // A heavy cluster on one rank plus a stray key, with duplicates far away on
    // the other ranks: one coarse block holds most keys and cannot be split.
    let mut keys = Vec::new();
    if rank == 0 {
        for i in 0..399usize {
            keys.push(morton::from_index_and_level(
                [i % 32, (i / 32) % 32, 1 + i / 1024],
                deepest,
            ));
        }
        keys.push(morton::from_index_and_level([0, 32, 32], deepest));
    } else {
        let far = morton::from_index_and_level([60000 - 100 * rank; 3], deepest);
        keys.extend(std::iter::repeat_n(far, 600));
    }
    // The heavy block goes whole to one rank, which may leave another without
    // blocks (Phase 5 T4); the cut keeps every rank within the heaviest block of a
    // fair share.
    let tree = check_tree(&keys, OctreeOptions::new().with_max_fine_keys(4), &comm);
    check_weight_bound(&tree, &keys, &comm);

    // Clustered points far from the domain corner make the natural coarse tree deep;
    // `max_level` must still bound the leaves and the result must be usable.
    let mut rng = seeded_rng(rank);
    let base = 20000 + rank * 200;
    let mut keys = (0..1000)
        .map(|_| {
            morton::from_index_and_level(
                [
                    base + rng.random_range(0..64usize),
                    base + rng.random_range(0..64usize),
                    base + rng.random_range(0..64usize),
                ],
                deepest,
            )
        })
        .collect_vec();
    keys.push(morton::from_index_and_level(
        [65000 - 100 * rank; 3],
        deepest,
    ));
    for max_level in [3, 6, 16] {
        let tree = check_tree(
            &keys,
            OctreeOptions::new()
                .with_max_level(max_level)
                .with_max_fine_keys(4),
            &comm,
        );
        if max_level == 3 {
            // With the level cap the coarse tree is small, so the ownership
            // partition is coarse: a level-3 block holding every cluster cannot be
            // split, and ranks may own no block. Every rank with blocks has its own
            // bound, and a rank without blocks shares the bound of a later rank.
            let nblocks = gather_to_all(&[tree.coarse_tree_leafs().len()], &comm);
            let owners = (0..size)
                .filter(|&r| nblocks[r] > 0)
                .map(|r| tree.coarse_tree_bounds()[r])
                .collect::<HashSet<_>>();
            assert_eq!(owners.len(), nblocks.iter().filter(|&&n| n > 0).count());
            for r in (0..size).filter(|&r| nblocks[r] == 0) {
                assert!(
                    r + 1 == size
                        || tree.coarse_tree_bounds()[r] == tree.coarse_tree_bounds()[r + 1]
                );
            }
        }
    }

    // Ghost bookkeeping: ghosts name a different rank, and each rank's ghosts are
    // leaves or interior keys of the rank they name. Both option settings are
    // checked from the same input: the default one-cell halo, and the
    // ghost-children layer, whose extra children and blocks must be classified
    // just as consistently.
    let mut rng = seeded_rng(7 + rank);
    let keys = generate_random_keys(500, &mut rng);
    let base_options = OctreeOptions::new()
        .with_max_level(10)
        .with_max_fine_keys(2);
    for options in [base_options, base_options.with_ghost_children(true)] {
        let tree = check_tree(&keys, options, &comm);
        check_ghost_bookkeeping(&tree, &comm);
    }

    if rank == 0 {
        println!("Construction edge cases passed on {size} rank(s).");
    }
}

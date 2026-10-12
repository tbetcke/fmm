//! Phase 5's `generate_all_keys` (`nd-octree` as of Phase 5S T3a, its debug checks left
//! out), applied to a tree built by today's `Octree::new`: the key set every rank held
//! before Phase 5S T3, for the examples to compare the locally essential set with. The
//! same adapter keeps `nd-fmm-plan` on the Phase 5 set until T4
//! (fmm-plan/src/phase5_keys.rs).

use std::collections::{HashMap, HashSet};

use mpi::traits::{CommunicatorCollectives, Equivalence};
use nd_octree::{MortonKey, Octree, morton, octree::KeyType};
use rlst::distributed_tools::{all_to_allv, array_tools::gather_to_all};

/// Phase 5's classification of every key this rank held: the leaves and their
/// ancestors, every ancestor of every coarse block as `Global`, the ghosts next to the
/// own keys and to the `Global` keys and, with the ghost-children layer, their children
/// and every coarse block of every rank.
///
/// # Collective operation
/// On more than one rank, two all-gathers and an all-to-all-v; every rank must call it.
pub fn phase5_held_keys<C: CommunicatorCollectives>(
    octree: &Octree<'_, C>,
) -> HashMap<MortonKey, KeyType> {
    /// A key with the rank that advertises it and whether it is a leaf there.
    #[derive(Copy, Clone, Default, Equivalence, PartialEq, Eq, Hash)]
    struct KeyWithRank {
        key: MortonKey,
        rank: usize,
        is_leaf: bool,
    }

    let comm = octree.comm();
    let leaf_tree = octree.leaf_keys();
    let coarse_tree = octree.coarse_tree_leafs();
    let coarse_tree_bounds = octree.coarse_tree_bounds();
    let ghost_children = octree.options().ghost_children();
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let mut all_keys = HashMap::<MortonKey, KeyType>::new();

    // Every ancestor of every rank's blocks, below the root, is `Global`.
    if size > 1 {
        let mut global_keys = HashSet::<MortonKey>::new();
        for &key in coarse_tree {
            let Some(mut parent) = morton::parent(key) else {
                continue;
            };
            while morton::level(parent) > 0 && !global_keys.contains(&parent) {
                global_keys.insert(parent);
                parent = morton::parent(parent).unwrap();
            }
        }
        let global_keys = gather_to_all(&global_keys.into_iter().collect::<Vec<_>>(), comm);
        for &key in &global_keys {
            all_keys.entry(key).or_insert(KeyType::Global);
        }
    }

    // The leaves and their ancestors up to the `Global` keys.
    for &leaf in leaf_tree {
        all_keys.insert(leaf, KeyType::LocalLeaf);
        let Some(mut parent) = morton::parent(leaf) else {
            continue;
        };
        while morton::level(parent) > 0 && !all_keys.contains_key(&parent) {
            all_keys.insert(parent, KeyType::LocalInterior);
            parent = morton::parent(parent).unwrap();
        }
    }

    // The root: local to its owner and a ghost elsewhere when it is the only block,
    // `Global` otherwise unless it is a leaf.
    let root_is_block = size > 1 && coarse_tree_bounds.first() == Some(&morton::root());
    let owns_root_block = root_is_block && coarse_tree.first() == Some(&morton::root());
    if owns_root_block {
        all_keys
            .entry(morton::root())
            .or_insert(KeyType::LocalInterior);
    } else if !root_is_block {
        all_keys.entry(morton::root()).or_insert(KeyType::Global);
    }

    if size > 1 {
        let mut rank_send_ghost = vec![HashSet::<KeyWithRank>::new(); size];
        let mut send_to_all = HashSet::<KeyWithRank>::new();
        let advertise = |key: MortonKey, is_leaf: bool, target: &mut HashSet<KeyWithRank>| {
            target.insert(KeyWithRank { key, rank, is_leaf });
            if !ghost_children || is_leaf {
                return;
            }
            let Some(children) = morton::children(key) else {
                return;
            };
            for child in children {
                let child_status = all_keys
                    .get(&child)
                    .expect("a local interior key holds all of its children locally");
                target.insert(KeyWithRank {
                    key: child,
                    rank,
                    is_leaf: *child_status == KeyType::LocalLeaf,
                });
            }
        };
        for (&key, &status) in all_keys.iter() {
            if status == KeyType::Global {
                continue;
            }
            let is_leaf = matches!(status, KeyType::LocalLeaf);
            for &neighbor in morton::neighbours(key)
                .iter()
                .filter(|&&key| morton::is_valid(key))
            {
                if all_keys
                    .get(&neighbor)
                    .is_some_and(|&value| value == KeyType::Global)
                {
                    advertise(key, is_leaf, &mut send_to_all);
                } else {
                    let neighbor_rank = coarse_tree_bounds
                        .partition_point(|&bound| bound <= neighbor)
                        .checked_sub(1)
                        .expect("a neighbour key must lie within the global partition bounds");
                    if neighbor_rank != rank {
                        advertise(key, is_leaf, &mut rank_send_ghost[neighbor_rank]);
                    }
                }
            }
        }
        let advertised_blocks = if ghost_children {
            &coarse_tree[..]
        } else if owns_root_block {
            &coarse_tree[..1]
        } else {
            &[]
        };
        for &block in advertised_blocks {
            let is_leaf = match all_keys[&block] {
                KeyType::LocalLeaf => true,
                KeyType::LocalInterior => false,
                KeyType::Global => continue,
                KeyType::GhostLeaf(_) | KeyType::GhostInterior(_) => {
                    unreachable!("a coarse block of this rank is never a ghost")
                }
            };
            send_to_all.insert(KeyWithRank {
                key: block,
                rank,
                is_leaf,
            });
        }
        let send_ghost_to_all = gather_to_all(&send_to_all.into_iter().collect::<Vec<_>>(), comm);
        let mut arr = Vec::<KeyWithRank>::new();
        let mut counts = Vec::<usize>::new();
        for keys in &rank_send_ghost {
            arr.extend(keys.iter());
            counts.push(keys.len());
        }
        let (_in_counts, mut ghost_keys) = all_to_allv(comm, &counts, &arr);
        ghost_keys.extend(send_ghost_to_all.iter());
        for key in &ghost_keys {
            if key.rank == rank {
                continue;
            }
            let key_type = if key.is_leaf {
                KeyType::GhostLeaf(key.rank)
            } else {
                KeyType::GhostInterior(key.rank)
            };
            all_keys.insert(key.key, key_type);
        }
    }

    all_keys
}

//! The Phase 5 construction of `Octree::new`, kept as a private reference copy for
//! Phase 5S T3a (docs/phase5s/T3a-octree-key-routing.md): the input keys linearized
//! by rlst's sample sort (`parsort`) and the successor exchange, the coarse tree by
//! weight (O1), the cut (O3), the key move, the local refinement, the distributed
//! 2:1 balance with a second sample sort, the key move again and the ghost
//! exchange. The private functions of octree/src/octree/implementation.rs as of
//! Phase 5S T1s are copied without their debug checks.
//!
//! The examples compare `Octree::new` with it key for key. It is removed when T3
//! lands.

use std::collections::{HashMap, HashSet};

use itertools::{Itertools, izip};
use mpi::{
    collective::SystemOperation,
    traits::{CommunicatorCollectives, Equivalence},
};
use nd_octree::{
    MortonKey, Octree, OctreeOptions, PartitionWeight, constants::DEEPEST_LEVEL, morton,
    octree::KeyType,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rlst::distributed_tools::{
    all_to_allv,
    array_tools::{communicate_back, gather_to_all},
    parallel_sort, sort_to_bins,
};

/// What the Phase 5 construction computes on this rank.
pub struct Reference {
    /// The replicated coarse tree.
    pub blocks: Vec<MortonKey>,
    /// The global weight of each block.
    pub weights: Vec<u64>,
    /// This rank's blocks.
    pub own_blocks: Vec<MortonKey>,
    /// The partition bounds of every rank.
    pub bounds: Vec<MortonKey>,
    /// This rank's leaves.
    pub leaves: Vec<MortonKey>,
    /// This rank's keys with their classification.
    pub all_keys: HashMap<MortonKey, KeyType>,
}

/// Build the octree of `fine_keys` as Phase 5's `Octree::new` did.
///
/// # Collective operation
/// The collectives of Phase 5's `Octree::new`; every rank must call it.
pub fn construct<C: CommunicatorCollectives>(
    fine_keys: &[MortonKey],
    options: OctreeOptions,
    comm: &C,
) -> Reference {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let mut rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);

    let linear_keys = linearize(fine_keys, &mut rng, comm);
    let (blocks, weights) = compute_coarse_tree(fine_keys, &linear_keys, options, comm);
    let partition = partition_blocks(&weights, size);
    let own_blocks = blocks[partition[rank]..partition[rank + 1]].to_vec();
    let bounds = tree_bins(&blocks, &partition);
    let local_fine_keys = redistribute(&linear_keys, &bounds, comm);
    let refined = create_local_tree(
        &local_fine_keys,
        &own_blocks,
        options.max_level(),
        options.max_fine_keys(),
    );
    let leaves = redistribute(&balance(&refined, &mut rng, comm), &bounds, comm);
    let all_keys = generate_all_keys(&leaves, &own_blocks, &bounds, options, comm);
    Reference {
        blocks,
        weights,
        own_blocks,
        bounds,
        leaves,
        all_keys,
    }
}

fn compute_coarse_tree<C: CommunicatorCollectives>(
    fine_keys: &[MortonKey],
    linear_keys: &[MortonKey],
    options: OctreeOptions,
    comm: &C,
) -> (Vec<MortonKey>, Vec<u64>) {
    if comm.size() == 1 {
        return (vec![morton::root()], vec![fine_keys.len() as u64]);
    }
    let max_level = options.max_level().min(DEEPEST_LEVEL as usize);
    let shares = options.block_refinement().max(1) as u64 * comm.size() as u64;
    let sorted_fine_keys;
    let weighted_keys = match options.partition_weight() {
        PartitionWeight::Keys => {
            let mut keys = fine_keys.to_vec();
            keys.sort_unstable();
            sorted_fine_keys = keys;
            sorted_fine_keys.as_slice()
        }
        PartitionWeight::DistinctKeys => linear_keys,
    };
    let mut blocks = vec![morton::root()];
    let weights = loop {
        let (distinct, weights) = global_block_counts(linear_keys, weighted_keys, &blocks, comm);
        match refine_heavy_blocks(
            &blocks,
            &distinct,
            &weights,
            shares,
            options.max_fine_keys(),
            max_level,
        ) {
            Some(refined) => blocks = refined,
            None => break weights,
        }
    };
    let balanced = morton::balance(&blocks, morton::root());
    let weights = if balanced == blocks {
        weights
    } else {
        blocks = balanced;
        global_block_counts(&[], weighted_keys, &blocks, comm).1
    };
    (blocks, weights)
}

fn global_block_counts<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    weighted_keys: &[MortonKey],
    blocks: &[MortonKey],
    comm: &C,
) -> (Vec<u64>, Vec<u64>) {
    let nblocks = blocks.len();
    let mut local = Vec::with_capacity(2 * nblocks);
    local.extend(
        local_coarse_tree_weights(linear_keys, blocks)
            .into_iter()
            .map(|count| count as u64),
    );
    local.extend(
        local_coarse_tree_weights(weighted_keys, blocks)
            .into_iter()
            .map(|count| count as u64),
    );
    let mut global = vec![0u64; 2 * nblocks];
    comm.all_reduce_into(&local, &mut global, SystemOperation::sum());
    let weights = global.split_off(nblocks);
    (global, weights)
}

fn refine_heavy_blocks(
    blocks: &[MortonKey],
    distinct: &[u64],
    weights: &[u64],
    shares: u64,
    max_fine_keys: usize,
    max_level: usize,
) -> Option<Vec<MortonKey>> {
    let total: u128 = weights.iter().map(|&weight| weight as u128).sum();
    let splits = |(&block, &distinct, &weight): (&MortonKey, &u64, &u64)| {
        weight as u128 * shares as u128 > total
            && distinct > max_fine_keys as u64
            && morton::level(block) < max_level
    };
    if !izip!(blocks, distinct, weights).any(splits) {
        return None;
    }
    let mut refined = Vec::with_capacity(blocks.len() + 7 * blocks.len());
    for entry in izip!(blocks, distinct, weights) {
        if splits(entry) {
            refined
                .extend(morton::children(*entry.0).expect("a block above max_level has children"));
        } else {
            refined.push(*entry.0);
        }
    }
    Some(refined)
}

fn local_coarse_tree_weights(sorted_keys: &[MortonKey], coarse_tree: &[MortonKey]) -> Vec<usize> {
    let mut weights = vec![0; coarse_tree.len()];
    let mut keys = sorted_keys.iter().copied().peekable();
    for (weight, &block) in izip!(weights.iter_mut(), coarse_tree) {
        while keys
            .next_if(|&key| morton::is_ancestor(block, key))
            .is_some()
        {
            *weight += 1;
        }
    }
    weights
}

fn redistribute<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    coarse_tree_bounds: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    if comm.size() == 1 {
        return linear_keys.to_vec();
    }
    let rank_counts = sort_to_bins(linear_keys, coarse_tree_bounds);
    all_to_allv(comm, &rank_counts, linear_keys).1
}

fn create_local_tree(
    sorted_fine_keys: &[MortonKey],
    coarse_keys: &[MortonKey],
    max_level: usize,
    max_keys: usize,
) -> Vec<MortonKey> {
    let max_level = max_level.min(DEEPEST_LEVEL as usize);
    let counts = sort_to_bins(sorted_fine_keys, coarse_keys);
    let mut remainder = sorted_fine_keys;
    let mut refined_keys = Vec::<MortonKey>::new();
    for (&count, &coarse_key) in izip!(counts.iter(), coarse_keys.iter()) {
        let current;
        (current, remainder) = remainder.split_at(count);
        if morton::level(coarse_key) < max_level && current.len() > max_keys {
            refined_keys.extend(create_local_tree(
                current,
                morton::children(coarse_key).unwrap().as_slice(),
                max_level,
                max_keys,
            ));
        } else {
            refined_keys.push(coarse_key)
        }
    }
    refined_keys
}

fn linearize<R: Rng, C: CommunicatorCollectives>(
    keys: &[MortonKey],
    rng: &mut R,
    comm: &C,
) -> Vec<MortonKey> {
    if comm.size() == 1 {
        return morton::linearize(keys);
    }
    let sorted_keys = parallel_sort::parsort(keys, comm, rng)
        .unwrap_or_else(|_| panic!("Could not sort sequence."));
    let mut result = Vec::<MortonKey>::new();
    let next_key = communicate_back(&sorted_keys, comm);
    for (&m1, &m2) in sorted_keys.iter().tuple_windows() {
        if !morton::is_ancestor(m1, m2) {
            result.push(m1);
        }
    }
    if let Some(&last) = sorted_keys.last() {
        match next_key {
            Some(next) if morton::is_ancestor(last, next) => {}
            _ => result.push(last),
        }
    }
    result
}

fn partition_blocks(weights: &[u64], size: usize) -> Vec<usize> {
    let nblocks = weights.len();
    let total: u128 = weights.iter().map(|&weight| weight as u128).sum();
    let size_wide = size as u128;
    let mut bounds = Vec::with_capacity(size + 1);
    bounds.push(0);
    let mut position = 0;
    let mut prefix = 0u128;
    for p in 1..size {
        let target = total * p as u128;
        let distance = |prefix: u128| (size_wide * prefix).abs_diff(target);
        while position < nblocks && distance(prefix + weights[position] as u128) <= distance(prefix)
        {
            prefix += weights[position] as u128;
            position += 1;
        }
        bounds.push(position);
    }
    bounds.push(nblocks);
    bounds
}

fn tree_bins(coarse_tree: &[MortonKey], partition: &[usize]) -> Vec<MortonKey> {
    let size = partition.len() - 1;
    let mut bins = vec![morton::invalid_key(); size];
    let mut next = morton::invalid_key();
    for p in (0..size).rev() {
        if partition[p] < partition[p + 1] {
            next = coarse_tree[partition[p]];
        }
        bins[p] = next;
    }
    bins
}

fn balance<R: Rng, C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    rng: &mut R,
    comm: &C,
) -> Vec<MortonKey> {
    let deepest_level = deepest_level(linear_keys, comm);
    if deepest_level == 0 {
        return linear_keys.to_vec();
    }
    let mut work_list = linear_keys
        .iter()
        .copied()
        .filter(|&key| morton::level(key) == deepest_level)
        .collect_vec();
    let mut result = Vec::<MortonKey>::new();
    for level in (1..=deepest_level).rev() {
        let mut parents = HashSet::<MortonKey>::new();
        let mut new_work_list = Vec::<MortonKey>::new();
        for key in work_list.iter() {
            let parent = morton::parent(*key).unwrap();
            if !parents.contains(&parent) {
                parents.insert(parent);
                result.extend_from_slice(morton::siblings(*key).unwrap().as_slice());
                new_work_list.extend(
                    morton::neighbours(parent)
                        .iter()
                        .copied()
                        .filter(|&key| morton::is_valid(key)),
                );
            }
        }
        new_work_list.extend(
            linear_keys
                .iter()
                .copied()
                .filter(|&key| morton::level(key) == level - 1),
        );
        work_list = new_work_list;
    }
    linearize(&result, rng, comm)
}

fn deepest_level<C: CommunicatorCollectives>(keys: &[MortonKey], comm: &C) -> usize {
    let local_deepest_level = keys
        .iter()
        .map(|&key| morton::level(key))
        .max()
        .unwrap_or(0);
    if comm.size() == 1 {
        return local_deepest_level;
    }
    let mut global_deepest_level: usize = 0;
    comm.all_reduce_into(
        &local_deepest_level,
        &mut global_deepest_level,
        SystemOperation::max(),
    );
    global_deepest_level
}

fn get_key_index(arr: &[MortonKey], key: MortonKey) -> Option<usize> {
    arr.partition_point(|candidate| *candidate <= key)
        .checked_sub(1)
}

fn generate_all_keys<C: CommunicatorCollectives>(
    leaf_tree: &[MortonKey],
    coarse_tree: &[MortonKey],
    coarse_tree_bounds: &[MortonKey],
    options: OctreeOptions,
    comm: &C,
) -> HashMap<MortonKey, KeyType> {
    #[derive(Copy, Clone, Default, Equivalence, PartialEq, Eq, Hash)]
    struct KeyWithRank {
        key: MortonKey,
        rank: usize,
        is_leaf: bool,
    }

    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let mut all_keys = HashMap::<MortonKey, KeyType>::new();
    let leaf_keys: HashSet<MortonKey> = HashSet::from_iter(leaf_tree.iter().copied());

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
        let global_keys = gather_to_all(&global_keys.iter().copied().collect_vec(), comm);
        for &key in &global_keys {
            all_keys.entry(key).or_insert(KeyType::Global);
        }
    }

    for leaf in leaf_keys {
        all_keys.insert(leaf, KeyType::LocalLeaf);
        let Some(mut parent) = morton::parent(leaf) else {
            continue;
        };
        while morton::level(parent) > 0 && !all_keys.contains_key(&parent) {
            all_keys.insert(parent, KeyType::LocalInterior);
            parent = morton::parent(parent).unwrap();
        }
    }

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
            if !options.ghost_children() || is_leaf {
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
                    let neighbor_rank = get_key_index(coarse_tree_bounds, neighbor)
                        .expect("a neighbour key must lie within the global partition bounds");
                    if neighbor_rank != rank {
                        advertise(key, is_leaf, &mut rank_send_ghost[neighbor_rank]);
                    }
                }
            }
        }
        let advertised_blocks = if options.ghost_children() {
            coarse_tree
        } else if owns_root_block {
            &coarse_tree[..1]
        } else {
            &[]
        };
        for &block in advertised_blocks {
            let status = *all_keys
                .get(&block)
                .expect("a coarse block is a key of its own rank");
            let is_leaf = match status {
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
        let send_ghost_to_all = gather_to_all(&send_to_all.into_iter().collect_vec(), comm);
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

/// Assert on every rank that `tree`, built by `Octree::new` from this rank's
/// `fine_keys` with `options`, is the Phase 5 octree key for key: the same leaves,
/// blocks and partition bounds on every rank, the same keys with the same
/// `KeyType`, and blocks whose weights are those of the Phase 5 construction.
///
/// The weights are not part of the octree, so the reference's weights are checked
/// against the weights of the same blocks counted from the gathered input keys.
///
/// # Collective operation
/// The Phase 5 construction, gathers and one all-reduce; every rank must call it.
pub fn assert_same_as_phase5<C: CommunicatorCollectives>(
    label: &str,
    tree: &Octree<'_, C>,
    fine_keys: &[MortonKey],
    options: OctreeOptions,
    comm: &C,
) {
    let reference = construct(fine_keys, options, comm);
    let mut failed = Vec::new();
    if tree.leaf_keys() != &reference.leaves {
        failed.push("leaves");
    }
    if tree.coarse_tree_leafs() != &reference.own_blocks {
        failed.push("own blocks");
    }
    if gather_to_all(tree.coarse_tree_leafs(), comm) != reference.blocks {
        failed.push("blocks");
    }
    if tree.coarse_tree_bounds() != &reference.bounds {
        failed.push("bounds");
    }
    if tree.all_keys() != &reference.all_keys {
        failed.push("all_keys");
    }

    let mut keys = gather_to_all(fine_keys, comm);
    keys.sort_unstable();
    if comm.size() > 1 && options.partition_weight() == PartitionWeight::DistinctKeys {
        keys.dedup();
    }
    let weights = local_coarse_tree_weights(&keys, &reference.blocks)
        .into_iter()
        .map(|count| count as u64)
        .collect_vec();
    if weights != reference.weights {
        failed.push("weights");
    }

    let mut all_same = false;
    comm.all_reduce_into(
        &failed.is_empty(),
        &mut all_same,
        SystemOperation::logical_and(),
    );
    assert!(
        all_same,
        "rank {}: {label}: not the Phase 5 octree on some rank; here: {failed:?}",
        comm.rank()
    );
}

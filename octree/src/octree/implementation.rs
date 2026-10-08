//! Parallel Octree structure

use std::collections::{HashMap, HashSet};

use crate::morton;
use crate::{constants::DEEPEST_LEVEL, morton::MortonKey};

use mpi::traits::Equivalence;

use itertools::{Itertools, izip};
use mpi::{collective::SystemOperation, traits::CommunicatorCollectives};
use rand::Rng;
use rlst::distributed_tools::parallel_sort::{self};
use rlst::distributed_tools::{
    all_to_allv,
    array_tools::{communicate_back, gather_to_all},
    sort_to_bins,
};

use super::{KeyType, LeafLocation, LookupBatchError, LookupError, OctreeOptions, PartitionWeight};

/// Build the coarse tree by weight from the root, replicated on every rank (design
/// O1, docs/design/distributed-fmm.md §3.2).
///
/// Starting from the root, every round counts for each block `b` the distinct keys
/// under it, `d(b)` (from `linear_keys`), and its weight `w(b)` (every input key
/// under it, duplicates included, or `d(b)` with [`PartitionWeight::DistinctKeys`]),
/// sums both over the ranks in one all-reduce, and splits into its eight children
/// every block with `w(b) > W / (k P)`, `d(b) > max_fine_keys` and a level below
/// `max_level` ([`refine_heavy_blocks`]). It stops when no block splits. The blocks
/// are then 2:1 balanced serially; if that changes them, one more all-reduce weighs
/// the balanced blocks. The one-rank tree splits exactly the boxes with more than
/// `max_fine_keys` distinct keys above `max_level`, so every block is a node of it.
///
/// Every rank holds the same counts after each all-reduce, so the number of rounds
/// is the same on every rank, and a rank without keys enters each round with zero
/// counts. On one rank the coarse tree is the root and nothing is communicated.
/// # Parameters
///
/// - `fine_keys`: The keys this rank passed to [`crate::Octree::new`], in any order,
///   duplicates included.
/// - `linear_keys`: This rank's share of the linearized keys: sorted, distinct, and
///   every distinct key on exactly one rank.
/// - `options`: The construction options: `max_level`, `max_fine_keys`, the
///   partition weight and the block refinement factor.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Returns
///
/// The blocks, a complete, linear and 2:1 balanced tree no deeper than `max_level`,
/// and the global weight of each block, the same on every rank.
///
/// # Collective operation
///
/// On more than one rank, one all-reduce per round and at most one after balancing;
/// every rank must enter.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let keys = vec![morton::deepest_first()];
/// let (blocks, weights) = compute_coarse_tree(&keys, &keys, OctreeOptions::default(), &comm);
/// assert_eq!((blocks, weights), (vec![morton::root()], vec![1]));
/// ```
pub fn compute_coarse_tree<C: CommunicatorCollectives>(
    fine_keys: &[MortonKey],
    linear_keys: &[MortonKey],
    options: OctreeOptions,
    comm: &C,
) -> (Vec<MortonKey>, Vec<u64>) {
    debug_assert!(is_linear_tree(linear_keys, comm));

    // On a single node a complete coarse tree is simply the root.
    if comm.size() == 1 {
        return (vec![morton::root()], vec![fine_keys.len() as u64]);
    }

    let max_level = options.max_level().min(DEEPEST_LEVEL as usize);
    let shares = options.block_refinement().max(1) as u64 * comm.size() as u64;

    // The weighted keys, sorted so that each round counts them in one merge pass.
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

    // Balancing only refines, and the blocks are replicated, so every rank takes the
    // same branch.
    let balanced = morton::balance(&blocks, morton::root());
    let weights = if balanced == blocks {
        weights
    } else {
        blocks = balanced;
        global_block_counts(&[], weighted_keys, &blocks, comm).1
    };
    debug_assert!(morton::is_complete_linear_and_balanced(&blocks));
    (blocks, weights)
}

/// Count the distinct and the weighted keys under each block, summed over the ranks.
///
/// # Parameters
///
/// - `linear_keys`: This rank's sorted distinct keys, counted as `d(b)`.
/// - `weighted_keys`: This rank's sorted weighted keys, duplicates counted, as `w(b)`.
/// - `blocks`: The replicated, sorted, complete blocks.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Returns
///
/// `(d, w)`, one entry per block each, the same on every rank.
///
/// # Collective operation
///
/// One all-reduce (sum) of `2 len(blocks)` values; every rank must enter.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let keys = vec![morton::deepest_first()];
/// let (d, w) = global_block_counts(&keys, &keys, &[morton::root()], &comm);
/// ```
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

/// Split every heavy, splittable block into its eight children (one round of O1).
///
/// A block is split when it weighs more than `W / shares` (`W` the total weight),
/// holds more than `max_fine_keys` distinct keys and lies above `max_level`. The
/// one-rank tree refines every box with more than `max_fine_keys` distinct keys
/// above `max_level`, so the children of a split block are nodes of it too.
/// # Parameters
///
/// - `blocks`: Sorted, complete blocks.
/// - `distinct`: Distinct keys under each block, `d(b)`.
/// - `weights`: Weight of each block, `w(b)`.
/// - `shares`: The refinement factor times the number of ranks, `k P`.
/// - `max_fine_keys`: The leaf criterion of the tree: a block with at most this
///   many distinct keys is a leaf of the one-rank tree and stays whole.
/// - `max_level`: The deepest level a block may have.
///
/// # Returns
///
/// The refined blocks, sorted and complete, or `None` when no block splits.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let refined = refine_heavy_blocks(&[morton::root()], &[2], &[2], 2, 1, 16);
/// assert_eq!(refined.unwrap(), morton::children(morton::root()).unwrap().to_vec());
/// ```
pub(crate) fn refine_heavy_blocks(
    blocks: &[MortonKey],
    distinct: &[u64],
    weights: &[u64],
    shares: u64,
    max_fine_keys: usize,
    max_level: usize,
) -> Option<Vec<MortonKey>> {
    debug_assert!(blocks.len() == distinct.len() && blocks.len() == weights.len());
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

/// Count for each coarse block how many of the sorted keys descend from it.
///
/// Both slices are sorted and the descendants of a block are contiguous in Morton
/// order, so a single merge pass suffices.
/// # Parameters
///
/// - `sorted_keys`: Sorted finest-level keys, each descending from exactly one block.
/// - `coarse_tree`: Sorted, non-overlapping blocks covering every key.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let blocks = morton::children(morton::root()).unwrap();
/// assert_eq!(local_coarse_tree_weights(&[morton::deepest_first()], &blocks)[0], 1);
/// ```
pub(crate) fn local_coarse_tree_weights(
    sorted_keys: &[MortonKey],
    coarse_tree: &[MortonKey],
) -> Vec<usize> {
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
    debug_assert!(
        keys.peek().is_none(),
        "every key must descend from a block of the complete coarse tree"
    );
    weights
}

/// Redistribute sorted keys to the ranks that own them.
///
/// Every key goes to the last rank whose bound is at most the key, the owner of
/// [`crate::Octree::owner_rank`]; a rank without blocks receives nothing.
/// # Parameters
///
/// - `linear_keys`: Locally held sorted, non-overlapping Morton keys, each one a
///   coarse block or a descendant of one.
/// - `coarse_tree_bounds`: The partition bounds of every rank, from [`tree_bins`].
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Collective operation
///
/// One all-to-all-v on more than one rank; every rank must enter.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let keys = vec![morton::deepest_first()];
/// let redistributed = redistribute_with_respect_to_coarse_tree(&keys, &[morton::root()], &comm);
/// ```
pub fn redistribute_with_respect_to_coarse_tree<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    coarse_tree_bounds: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    if comm.size() == 1 {
        return linear_keys.to_vec();
    }

    // The keys and the bounds are both sorted, so one pass counts the keys of each
    // rank. Repeated bounds (ranks without blocks) select the last of them.
    let rank_counts = sort_to_bins(linear_keys, coarse_tree_bounds);
    let (_in_counts, result) = all_to_allv(comm, &rank_counts, linear_keys);

    // A rank whose blocks contain no keys, or that has no blocks, receives nothing.

    #[cfg(debug_assertions)]
    {
        // Check that the result array is sorted, and that every key is owned here.
        use rlst::distributed_tools::array_tools::is_sorted_array;
        debug_assert!(is_sorted_array(&result, comm));
        let rank = comm.rank() as usize;
        debug_assert!(
            sort_to_bins(&result, coarse_tree_bounds)
                .iter()
                .enumerate()
                .all(|(owner, &count)| owner == rank || count == 0)
        );
    }

    result
}

/// Return a complete tree generated from local keys and associated coarse keys.
///
/// The coarse keys are refined until the maximum level is reached or until each coarse key
/// is the ancestor of at most `max_keys` fine keys.
/// It is assumed that the level of the fine keys is at least as large as `max_level`.
/// # Parameters
///
/// - `sorted_fine_keys`: Sorted local finest-level keys to distribute among `coarse_keys`.
/// - `coarse_keys`: Sorted local coarse-tree leaves that bound refinement.
/// - `max_level`: Finest permitted leaf level.
/// - `max_keys`: Refinement target: leaves split while they contain more keys.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let fine = vec![morton::deepest_first(), morton::from_index_and_level([1, 0, 0], 16)];
/// let coarse = vec![morton::root()];
/// let leaves = create_local_tree(&fine, &coarse, 16, 1);
/// assert!(leaves.iter().all(|&key| key != morton::root()));
/// ```
pub fn create_local_tree(
    sorted_fine_keys: &[MortonKey],
    coarse_keys: &[MortonKey],
    mut max_level: usize,
    max_keys: usize,
) -> Vec<MortonKey> {
    if max_level > DEEPEST_LEVEL as usize {
        max_level = DEEPEST_LEVEL as usize;
    }

    // We split the sorted fine keys into subslices so that each subslice
    // is associated with a coarse slice.

    let bins = coarse_keys.to_vec();

    let counts = sort_to_bins(sorted_fine_keys, &bins);

    // We now know how many fine keys are associated with each coarse block. We iterate
    // through and locally refine for each block that requires it.

    let mut remainder = sorted_fine_keys;
    let mut refined_keys = Vec::<MortonKey>::new();

    for (&count, &coarse_key) in izip!(counts.iter(), coarse_keys.iter()) {
        let current;
        (current, remainder) = remainder.split_at(count);
        if morton::level(coarse_key) < max_level && current.len() > max_keys {
            // We need to refine the current split.
            refined_keys.extend_from_slice(
                create_local_tree(
                    current,
                    morton::children(coarse_key).unwrap().as_slice(),
                    max_level,
                    max_keys,
                )
                .as_slice(),
            );
        } else {
            refined_keys.push(coarse_key)
        }
    }

    refined_keys
}

/// Linearize a set of weighted Morton keys.
/// # Parameters
///
/// - `keys`: Locally held Morton keys to sort, validate, or classify, depending on the helper.
/// - `rng`: Random source used for sampling.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let mut rng = crate::tools::seeded_rng(0);
/// let linear = linearize(&[morton::deepest_first()], &mut rng, &comm);
/// ```
pub fn linearize<R: Rng, C: CommunicatorCollectives>(
    keys: &[MortonKey],
    rng: &mut R,
    comm: &C,
) -> Vec<MortonKey> {
    // If we only have one process we use the standard serial linearization.

    if comm.size() == 1 {
        return morton::linearize(keys);
    }

    // We are first sorting the keys. Then in a linear process across all processors we
    // go through the arrays and delete ancestors of nodes.

    let sorted_keys = parallel_sort::parsort(keys, comm, rng)
        .unwrap_or_else(|_| panic!("Could not sort sequence."));

    // Each process needs to send its first element to the previous process. Each process
    // then goes through its own list and retains elements that are not ancestors of the
    // next element.

    let mut result = Vec::<MortonKey>::new();

    let next_key = communicate_back(&sorted_keys, comm);

    // Treat the local keys
    for (&m1, &m2) in sorted_keys.iter().tuple_windows() {
        // m1 is also ancestor of m2 if they are identical.
        if morton::is_ancestor(m1, m2) {
            continue;
        } else {
            result.push(m1);
        }
    }

    // The parallel sort may leave this rank without keys. Otherwise keep the last
    // key unless it is an ancestor of `next_key`, the first key on the next
    // non-empty process; the last non-empty process always keeps its last key.

    if let Some(&last) = sorted_keys.last() {
        match next_key {
            Some(next) if morton::is_ancestor(last, next) => {}
            _ => result.push(last),
        }
    }

    debug_assert!(is_linear_tree(&result, comm));

    result
}

/// Cut weighted blocks into `size` contiguous ranges at the nearest block boundaries
/// (design O3, docs/design/distributed-fmm.md §3.2).
///
/// Returns `size + 1` boundaries; range `p` is `bounds[p]..bounds[p + 1]`. With `W`
/// the total weight, boundary `p` (`0 < p < size`) is the block boundary, at or
/// after boundary `p - 1`, whose prefix weight is closest to `W p / size`; of equally
/// close boundaries it takes the last. Each boundary is then within half the
/// heaviest block of its target, so no range weighs more than `W / size` plus the
/// heaviest block. Ranges may be empty: a block heavier than a fair share goes whole
/// to one range, and with fewer blocks than ranges some ranges are always empty.
/// With no weight at all, the first range takes every block.
/// # Parameters
///
/// - `weights`: Block weights in block order.
/// - `size`: Number of ranges, at least one.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// assert_eq!(partition_blocks(&[1, 1, 1, 1], 2), vec![0, 2, 4]);
/// assert_eq!(partition_blocks(&[5], 3), vec![0, 0, 1, 1]);
/// ```
pub(crate) fn partition_blocks(weights: &[u64], size: usize) -> Vec<usize> {
    assert!(size > 0, "a partition needs at least one range");
    let nblocks = weights.len();
    let total: u128 = weights.iter().map(|&weight| weight as u128).sum();
    let size_wide = size as u128;

    // `prefix` is the weight before block `position`. The distance of a boundary from
    // target p is |size prefix - W p|, in integers. The prefix weights do not
    // decrease, so the distance falls and then rises along the blocks: walking on
    // while the next boundary is at least as close finds the last closest boundary.
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

/// The partition bounds of every rank: the first block of each rank's range.
///
/// A rank with an empty range takes the bound of the next rank with blocks, or,
/// after the last such rank, [`morton::invalid_key`], which sorts above every valid
/// key. The bounds do not decrease, and the last rank whose bound is at most a key
/// owns it, which is never a rank without blocks.
/// # Parameters
///
/// - `coarse_tree`: The replicated complete, linear coarse tree.
/// - `partition`: The `size + 1` range boundaries of [`partition_blocks`].
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let octants = morton::children(morton::root()).unwrap();
/// let bins = tree_bins(&octants, &[0, 0, 4, 8]);
/// assert_eq!(bins, vec![octants[0], octants[0], octants[4]]);
/// ```
pub(crate) fn tree_bins(coarse_tree: &[MortonKey], partition: &[usize]) -> Vec<MortonKey> {
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

/// Balance a distributed tree.
///
/// A rank may hold no keys; it still enters every collective.
/// # Parameters
///
/// - `linear_keys`: Locally held sorted, non-overlapping Morton keys.
/// - `rng`: Random source used for sampling.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let mut rng = crate::tools::seeded_rng(0);
/// let balanced = balance(&[morton::root()], &mut rng, &comm);
/// ```
pub fn balance<R: Rng, C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    rng: &mut R,
    comm: &C,
) -> Vec<MortonKey> {
    // Treat the case that the tree is only the root, on one rank: the loop below
    // would give an empty output, as it only iterates up to level 1. The deepest
    // level is global, so every rank takes the same branch, also a rank without keys.

    let deepest_level = deepest_level(linear_keys, comm);
    if deepest_level == 0 {
        return linear_keys.to_vec();
    }

    // Start with keys at deepest level
    let mut work_list = linear_keys
        .iter()
        .copied()
        .filter(|&key| morton::level(key) == deepest_level)
        .collect_vec();

    let mut result = Vec::<MortonKey>::new();

    // Now go through and make sure that for each key siblings and neighbours of parents are added

    for level in (1..=deepest_level).rev() {
        let mut parents = HashSet::<MortonKey>::new();
        let mut new_work_list = Vec::<MortonKey>::new();
        // We filter the work list by level and also make sure that
        // only one sibling of each of the parents children is added to
        // our current level list.
        for key in work_list.iter() {
            let parent = morton::parent(*key).unwrap();
            if !parents.contains(&parent) {
                parents.insert(parent);
                result.extend_from_slice(morton::siblings(*key).unwrap().as_slice());
                new_work_list.extend_from_slice(
                    morton::neighbours(parent)
                        .iter()
                        .copied()
                        .filter(|&key| morton::is_valid(key))
                        .collect_vec()
                        .as_slice(),
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

    let result = linearize(&result, rng, comm);

    debug_assert!(crate::octree::is_complete_linear_and_balanced(
        &result, comm
    ));
    result
}

/// Return true if the keys are linear.
/// # Parameters
///
/// - `arr`: Locally sorted Morton-key partition being checked for the stated invariant.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert!(is_linear_tree(&[morton::deepest_first()], &comm));
/// ```
pub fn is_linear_tree<C: CommunicatorCollectives>(arr: &[MortonKey], comm: &C) -> bool {
    let mut is_linear = true;

    for (&key1, &key2) in arr.iter().tuple_windows() {
        if key1 >= key2 || morton::is_ancestor(key1, key2) {
            is_linear = false;
            break;
        }
    }

    if comm.size() == 1 {
        return is_linear;
    }

    // Now check the interfaces. A rank without keys has no interface of its own;
    // `communicate_back` skips it when locating the next key.

    if let (Some(next_key), Some(&last)) = (communicate_back(arr, comm), arr.last())
        && (last >= next_key || morton::is_ancestor(last, next_key))
    {
        is_linear = false;
    }

    let mut global_is_linear = false;

    comm.all_reduce_into(
        &is_linear,
        &mut global_is_linear,
        SystemOperation::logical_and(),
    );

    global_is_linear
}

/// Return true on all ranks if distributed tree is complete. Otherwise, return false.
/// # Parameters
///
/// - `arr`: Locally sorted Morton-key partition being checked for the stated invariant.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert!(is_complete_linear_tree(&[morton::root()], &comm));
/// ```
pub fn is_complete_linear_tree<C: CommunicatorCollectives>(arr: &[MortonKey], comm: &C) -> bool {
    // First check that the local tree on each node is complete.

    let mut complete_linear = true;
    for (key1, key2) in arr.iter().tuple_windows() {
        // Make sure that the keys are sorted and not duplicated.
        if key1 >= key2 {
            complete_linear = false;
            break;
        }
        // The next key should be an ancestor of the next non-descendent key.
        if let Some(expected_next) = morton::next_non_descendent_key(*key1) {
            if !morton::is_ancestor(*key2, expected_next) {
                complete_linear = false;
                break;
            }
        } else {
            // Only for the very last key there should not be a next non-descendent key.
            complete_linear = false;
        }
    }

    // We now check the interfaces. A rank without keys has no interface of its own.

    let next_first = communicate_back(arr, comm);
    match (arr.last(), next_first) {
        (Some(&last_key), Some(next_first)) => {
            // There is a later non-empty rank.

            // Check that the keys are sorted and not duplicated.
            if last_key >= next_first {
                complete_linear = false;
            }

            // Check that the next key is an ancestor of the next non-descendent.
            if let Some(expected_next) = morton::next_non_descendent_key(last_key) {
                if !morton::is_ancestor(next_first, expected_next) {
                    complete_linear = false;
                }
            } else {
                complete_linear = false;
            }
        }
        (Some(&last_key), None) => {
            // We are on the last non-empty rank.
            // Check that the last key is ancestor of deepest last.
            if !morton::is_ancestor(last_key, morton::deepest_last()) {
                complete_linear = false;
            }
        }
        (None, _) => {}
    }

    // Now check that the first key of the whole tree is an ancestor of the deepest first.

    let global_first = gather_to_all(&arr[..arr.len().min(1)], comm);
    match global_first.first() {
        Some(&first) if morton::is_ancestor(first, morton::deepest_first()) => {}
        _ => complete_linear = false,
    }

    // Now communicate everything together.

    let mut result = false;
    comm.all_reduce_into(
        &complete_linear,
        &mut result,
        SystemOperation::logical_and(),
    );

    result
}

/// Return the deepest level of a distributed list of Morton keys.
///
/// A rank without keys contributes level 0.
/// # Parameters
///
/// - `keys`: Locally held Morton keys to sort, validate, or classify, depending on the helper.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert_eq!(deepest_level(&[morton::root()], &comm), 0);
/// ```
pub fn deepest_level<C: CommunicatorCollectives>(keys: &[MortonKey], comm: &C) -> usize {
    let local_deepest_level = keys
        .iter()
        .map(|elem| morton::level(*elem))
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

/// For a sorted array return either position of the key or positioin directly before search key.
/// # Parameters
///
/// - `arr`: Locally held, Morton-ordered tree keys.
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert_eq!(get_key_index(&[1, 2], 2), Some(1));
/// assert_eq!(get_key_index(&[5, 7], 3), None);
/// ```
pub fn get_key_index(arr: &[MortonKey], key: MortonKey) -> Option<usize> {
    // Either the position of an exact match or the position of the closest smaller
    // key. `None` when `key` precedes every element, which has no predecessor.
    checked_predecessor(arr, key)
}

// /// Generate a map that associates each leaf with the corresponding point indices.
// pub fn assign_points_to_leaf_keys(
//     point_keys: &[MortonKey],
//     leaf_keys: &[MortonKey],
// ) -> HashMap<MortonKey, Vec<usize>> {
//     let mut point_map = HashMap::<MortonKey, Vec<usize>>::new();
//
//     for (index, point_key) in point_keys.iter().enumerate() {
//         let leaf_key_index = get_key_index(leaf_keys, *point_key);
//
//         let leaf_key = leaf_keys[leaf_key_index];
//         debug_assert!(morton::is_ancestor(leaf_key, *point_key));
//
//         point_map.entry(leaf_key).or_default().push(index);
//     }
//
//     point_map
// }

// /// Check if a key is associated with the current rank.
// ///
// /// Note that the key does not need to exist as leaf. It just needs
// /// to be descendent of a coarse key on the current rank.
// pub fn key_on_current_rank(
//     key: MortonKey,
//     coarse_tree_bounds: &[MortonKey],
//     rank: usize,
//     size: usize,
// ) -> bool {
//     if rank == size - 1 {
//         key >= *coarse_tree_bounds.last().unwrap()
//     } else {
//         coarse_tree_bounds[rank] <= key && key < coarse_tree_bounds[rank + 1]
//     }
// }

/// Validate the keys accepted by [`crate::Octree::new`].
///
/// Every key must be valid and at [`DEEPEST_LEVEL`]. The error names the first
/// offending key and its position.
/// # Parameters
///
/// - `fine_keys`: Locally supplied construction keys.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// assert!(validate_fine_keys(&[morton::deepest_first()]).is_ok());
/// assert!(validate_fine_keys(&[morton::root()]).is_err());
/// ```
pub(crate) fn validate_fine_keys(fine_keys: &[MortonKey]) -> Result<(), String> {
    for (index, &key) in fine_keys.iter().enumerate() {
        if !morton::is_valid(key) {
            return Err(format!(
                "fine key {key:#x} at position {index} is invalid: construction keys must be \
                 valid Morton keys at level {DEEPEST_LEVEL}"
            ));
        }
        let level = morton::level(key);
        if level != DEEPEST_LEVEL as usize {
            return Err(format!(
                "fine key {key:#x} at position {index} is at level {level}: construction \
                 keys must be at level {DEEPEST_LEVEL}, for example from `points_to_morton`"
            ));
        }
    }
    Ok(())
}

/// Validate a key accepted by leaf lookup APIs.
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert!(validate_lookup_key(morton::deepest_first()).is_ok());
/// ```
pub(crate) fn validate_lookup_key(key: MortonKey) -> Result<(), LookupError> {
    if !morton::is_valid(key) {
        return Err(LookupError::InvalidKey);
    }
    let level = morton::level(key);
    if level != DEEPEST_LEVEL as usize {
        return Err(LookupError::NotFinestLevel { level });
    }
    Ok(())
}

/// Return the predecessor position in a sorted sequence, if one exists.
/// # Parameters
///
/// - `arr`: Locally held, Morton-ordered tree keys.
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert_eq!(checked_predecessor(&[1, 3], 2), Some(0));
/// ```
pub(crate) fn checked_predecessor(arr: &[MortonKey], key: MortonKey) -> Option<usize> {
    arr.partition_point(|candidate| *candidate <= key)
        .checked_sub(1)
}

/// Return the owner of a valid finest-level key from rank partition bounds.
/// # Parameters
///
/// - `bounds`: Per-rank lower bounds of the Morton partition.
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert_eq!(owner_rank_for_key(&[morton::deepest_first()], morton::deepest_first()), Some(0));
/// ```
pub(crate) fn owner_rank_for_key(bounds: &[MortonKey], key: MortonKey) -> Option<usize> {
    checked_predecessor(bounds, key)
}

/// Return the local leaf containing a valid finest-level key.
/// # Parameters
///
/// - `leaves`: Sorted locally owned leaf keys.
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert_eq!(local_leaf_for_key(&[morton::root()], morton::deepest_first()), Some(morton::root()));
/// ```
pub(crate) fn local_leaf_for_key(leaves: &[MortonKey], key: MortonKey) -> Option<MortonKey> {
    let candidate = *leaves.get(checked_predecessor(leaves, key)?)?;
    morton::is_ancestor(candidate, key).then_some(candidate)
}

/// Return whether MPI counts and their displacements fit the `i32` MPI interface.
/// # Parameters
///
/// - `counts`: Element counts that will be passed to MPI.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert!(counts_fit_mpi_i32(&[1, 2]));
/// ```
pub(crate) fn counts_fit_mpi_i32(counts: &[usize]) -> bool {
    counts
        .iter()
        .try_fold(0usize, |total, &count| {
            (count <= i32::MAX as usize)
                .then(|| total.checked_add(count))
                .flatten()
                .filter(|&sum| sum <= i32::MAX as usize)
        })
        .is_some()
}

struct PackedLookupQueries {
    counts: Vec<usize>,
    keys: Vec<MortonKey>,
    original_indices: Vec<usize>,
    results: Vec<Option<Result<LeafLocation, LookupError>>>,
}

/// Validate and stably pack valid queries into owner-rank buckets.
///
/// # Parameters
///
/// - `keys`: Queries in caller order; invalid queries are retained as error slots.
/// - `coarse_tree_bounds`: Lower Morton bound for each owner rank.
/// - `size`: Number of owner-rank buckets to create.
///
/// # Examples
///
/// ```ignore
/// // `implementation` is private; collective leaf lookup calls this helper internally.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let packed = pack_lookup_queries(&[morton::deepest_first()], &[morton::deepest_first()], 1);
/// assert_eq!(packed.counts, vec![1]);
/// ```
fn pack_lookup_queries(
    keys: &[MortonKey],
    coarse_tree_bounds: &[MortonKey],
    size: usize,
) -> PackedLookupQueries {
    let mut buckets = vec![Vec::new(); size];
    let mut index_buckets = vec![Vec::new(); size];
    let mut results = vec![None; keys.len()];

    for (index, &key) in keys.iter().enumerate() {
        match validate_lookup_key(key) {
            Err(error) => results[index] = Some(Err(error)),
            Ok(()) => match owner_rank_for_key(coarse_tree_bounds, key) {
                Some(owner) if owner < size => {
                    buckets[owner].push(key);
                    index_buckets[owner].push(index);
                }
                _ => results[index] = Some(Err(LookupError::LeafNotFound)),
            },
        }
    }

    PackedLookupQueries {
        counts: buckets.iter().map(Vec::len).collect_vec(),
        keys: buckets.into_iter().flatten().collect_vec(),
        original_indices: index_buckets.into_iter().flatten().collect_vec(),
        results,
    }
}

/// Restore received replies to their original query positions.
///
/// # Parameters
///
/// - `results`: Output slots in original query order; matching entries are mutated.
/// - `original_indices`: Query positions paired one-for-one with `replies`.
/// - `replies`: Leaf keys returned by the owner ranks in packed-query order.
/// - `keys`: Original query keys indexed by `original_indices`.
/// - `coarse_tree_bounds`: Owner-rank bounds used to annotate successful replies.
///
/// # Examples
///
/// ```ignore
/// // `implementation` is private; collective lookup uses this restoration step.
/// let mut results = vec![None];
/// scatter_lookup_replies(&mut results, [0], [morton::root()], &[morton::deepest_first()], &[morton::deepest_first()]);
/// assert!(results[0].as_ref().unwrap().is_ok());
/// ```
fn scatter_lookup_replies(
    results: &mut [Option<Result<LeafLocation, LookupError>>],
    original_indices: impl IntoIterator<Item = usize>,
    replies: impl IntoIterator<Item = MortonKey>,
    keys: &[MortonKey],
    coarse_tree_bounds: &[MortonKey],
) {
    for (index, leaf) in original_indices.into_iter().zip(replies) {
        let result = if leaf == morton::invalid_key() {
            Err(LookupError::LeafNotFound)
        } else {
            let owner = owner_rank_for_key(coarse_tree_bounds, keys[index])
                .expect("valid packed query must have an owner");
            Ok(LeafLocation {
                leaf,
                owner_rank: owner,
            })
        };
        results[index] = Some(result);
    }
}

/// Execute the collective request/reply portion of batched leaf lookup.
///
/// # Parameters
///
/// - `keys`: Local queries, whose result order matches this slice.
/// - `coarse_tree_bounds`: Lower Morton bound for every owner rank.
/// - `leaf_keys`: Sorted leaves local to `comm.rank()`.
/// - `comm`: Communicator whose ranks all enter this exchange in the same order.
///
/// # Examples
///
/// ```ignore
/// // `implementation` is private; `Octree::lookup_leaves` invokes this collectively.
/// let found = lookup_leaves_collective(&[morton::deepest_first()], &[morton::deepest_first()], &[morton::root()], &comm);
/// assert!(found.unwrap()[0].is_ok());
/// ```
pub(crate) fn lookup_leaves_collective<C: CommunicatorCollectives>(
    keys: &[MortonKey],
    coarse_tree_bounds: &[MortonKey],
    leaf_keys: &[MortonKey],
    comm: &C,
) -> Result<Vec<Result<LeafLocation, LookupError>>, LookupBatchError> {
    let packed = pack_lookup_queries(keys, coarse_tree_bounds, comm.size() as usize);
    let PackedLookupQueries {
        counts,
        keys: packed_keys,
        original_indices,
        mut results,
    } = packed;
    let size = comm.size() as usize;

    // Exchange unconverted counts first so every rank can reject conversion overflow.
    let mut received_counts = vec![0usize; size];
    comm.all_to_all_into(&counts, &mut received_counts);
    let local_overflow = !counts_fit_mpi_i32(&counts) || !counts_fit_mpi_i32(&received_counts);
    let mut any_overflow = false;
    comm.all_reduce_into(
        &local_overflow,
        &mut any_overflow,
        SystemOperation::logical_or(),
    );
    if any_overflow {
        return Err(LookupBatchError::CountOverflow);
    }

    let (_request_counts, received_keys) = all_to_allv(comm, &counts, &packed_keys);
    let replies = received_keys
        .iter()
        .map(|&key| local_leaf_for_key(leaf_keys, key).unwrap_or_else(morton::invalid_key))
        .collect_vec();
    let (_reply_counts, received_replies) = all_to_allv(comm, &received_counts, &replies);

    debug_assert_eq!(received_replies.len(), original_indices.len());
    scatter_lookup_replies(
        &mut results,
        original_indices,
        received_replies,
        keys,
        coarse_tree_bounds,
    );

    Ok(results
        .into_iter()
        .map(|result| result.expect("every lookup result slot must be populated"))
        .collect())
}

/// Generate all leaf and interior keys.
///
/// `leaf_tree` and `coarse_tree` are assumed to be sorted.
///
/// With [`OctreeOptions::ghost_children`] enabled the ghost exchange additionally
/// carries, for every key a rank advertises to another rank, that key's eight
/// children whenever the key is interior, and it advertises every local coarse
/// block to all ranks. Together these guarantee that for every non-ghost key of
/// the result and every same-level neighbour cell of it which the result holds as
/// an interior key, all eight children of that neighbour are in the result as
/// well. This is one step of closure relative to the keys stored without the
/// option, not a fixed point. Both additions travel inside the two collectives the
/// exchange already performs, so no communication round is added.
/// # Parameters
///
/// - `leaf_tree`: Sorted local leaf keys to classify.
/// - `coarse_tree`: Local partition of the complete coarse tree.
/// - `coarse_tree_bounds`: Per-rank lower Morton bounds used to identify remote ownership.
/// - `options`: Optional topology layers requested from [`crate::Octree::new`].
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let leaves = morton::children(morton::root()).unwrap();
/// let options = OctreeOptions::default();
/// let keys = generate_all_keys(&leaves, &leaves, &[leaves[0]], options, &comm);
/// ```
pub fn generate_all_keys<C: CommunicatorCollectives>(
    leaf_tree: &[MortonKey],
    coarse_tree: &[MortonKey],
    coarse_tree_bounds: &[MortonKey],
    options: OctreeOptions,
    comm: &C,
) -> HashMap<MortonKey, KeyType> {
    /// This struct combines rank and key information for sending ghosts to neighbors.
    ///
    /// `Default` is required by the collective helpers, which need a placeholder
    /// element for ranks contributing nothing. Such a value is never read back as
    /// a ghost.
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

    // If size == 1 we simply create locally the keys, so don't need to treat the global keys.

    if size > 1 {
        let mut global_keys = HashSet::<MortonKey>::new();

        // First deal with the parents of the coarse tree. These are different
        // as they may exist on multiple nodes, so receive a different label.

        for &key in coarse_tree {
            // A root-only tree has no ancestors to add.
            let Some(mut parent) = morton::parent(key) else {
                continue;
            };
            // Stop as soon as an ancestor chain that was already walked is reached.
            // `all_keys` is still empty here, so testing it would never short-circuit.
            while morton::level(parent) > 0 && !global_keys.contains(&parent) {
                global_keys.insert(parent);
                parent = morton::parent(parent).unwrap();
            }
        }

        // We now send around the parents of the coarse tree to every node. These will
        // be global keys.

        let global_keys = gather_to_all(&global_keys.iter().copied().collect_vec(), comm);

        // We can now insert the global keys into `all_keys` with the `Global` label.

        for &key in &global_keys {
            all_keys.entry(key).or_insert(KeyType::Global);
        }
    }

    // We now deal with the fine leafs and their ancestors.
    // The leafs of the coarse tree will also be either part
    // of the fine tree leafs or will be interior keys. In either
    // case the following loop catches them.

    for leaf in leaf_keys {
        debug_assert!(!all_keys.contains_key(&leaf));
        all_keys.insert(leaf, KeyType::LocalLeaf);
        // Keep a root leaf local; it has no parent, and the root insertion
        // below must not replace its LocalLeaf classification.
        let Some(mut parent) = morton::parent(leaf) else {
            continue;
        };
        while morton::level(parent) > 0 && !all_keys.contains_key(&parent) {
            all_keys.insert(parent, KeyType::LocalInterior);
            parent = morton::parent(parent).unwrap();
        }
    }

    // Need to explicitly add the root at the end. On several ranks the root may be
    // the only coarse block; the bounds are then the root up to its owner and above
    // every valid key after it, since the first bound is always the first block.
    // That block is a local key of its owner and reaches the other ranks as a ghost
    // through the gather below, never as `Global`.
    let root_is_block = size > 1 && coarse_tree_bounds.first() == Some(&morton::root());
    let owns_root_block = root_is_block && coarse_tree.first() == Some(&morton::root());
    if owns_root_block {
        all_keys
            .entry(morton::root())
            .or_insert(KeyType::LocalInterior);
    } else if !root_is_block {
        all_keys.entry(morton::root()).or_insert(KeyType::Global);
    }

    // We only need to deal with ghosts if the size is larger than 1.

    if size > 1 {
        // For each rank the set of keys that we want to send to it. Sets, because a
        // key is a neighbour of the same rank through several directions but only
        // needs to be sent once.

        let mut rank_send_ghost = vec![HashSet::<KeyWithRank>::new(); size];

        let mut send_to_all = HashSet::<KeyWithRank>::new();

        // Advertise `key` to a receiver, together with its eight children when the
        // ghost-children layer is requested and `key` is interior. An interior key
        // holds all of its children on this rank, so every child's classification
        // is available locally. The target is a set, so duplicates are free.
        let advertise = |key: MortonKey, is_leaf: bool, target: &mut HashSet<KeyWithRank>| {
            target.insert(KeyWithRank { key, rank, is_leaf });
            if !options.ghost_children() || is_leaf {
                return;
            }
            // An interior key is never at the deepest level, but the API is total.
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
            // We need not send around global keys to neighbors.
            if status == KeyType::Global {
                continue;
            }
            let is_leaf = matches!(status, KeyType::LocalLeaf);
            for &neighbor in morton::neighbours(key)
                .iter()
                .filter(|&&key| morton::is_valid(key))
            {
                // If the neighbour is a global key then continue.
                if all_keys
                    .get(&neighbor)
                    .is_some_and(|&value| value == KeyType::Global)
                {
                    // Global keys exist on all nodes, so need to send their neighbors to all nodes.

                    advertise(key, is_leaf, &mut send_to_all);
                } else {
                    // Get rank of the neighbour. Every non-global key is at or above
                    // the first partition bound, so a predecessor must exist. Keys
                    // whose neighbour is local need not be sent anywhere.
                    let neighbor_rank = get_key_index(coarse_tree_bounds, neighbor)
                        .expect("a neighbour key must lie within the global partition bounds");
                    if neighbor_rank != rank {
                        advertise(key, is_leaf, &mut rank_send_ghost[neighbor_rank]);
                    }
                }
            }
        }

        // The children of a `Global` key are either `Global`, and hence already on
        // every rank, or coarse blocks. Replicating the local blocks therefore
        // closes the ghost-children guarantee at the `Global` keys. The blocks go
        // without their children: a block is advertised here as a key of the tree,
        // not as a neighbour of anything. A root that is the only block is
        // advertised with or without the layer: it is a key of every rank.
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
                // On several ranks a coarse block is never `Global`: the root,
                // when it is the only block, is local to its owner (above). The
                // branch is spelled out because advertising a `Global` key would
                // be both pointless — it is already on every rank — and harmful,
                // since the receive loop would overwrite the receivers' `Global`
                // entry with a `GhostInterior`.
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
        // We now know which key needs to be sent to which rank.
        // Turn to array, get the counts and send around.

        let (arr, counts) = {
            let mut arr = Vec::<KeyWithRank>::new();
            let mut counts = Vec::<usize>::new();
            for keys in &rank_send_ghost {
                arr.extend(keys.iter());
                counts.push(keys.len());
            }
            (arr, counts)
        };

        // These are all the keys that are neighbors to our keys. We now go through
        // and store those that do not live on our tree as into `all_keys` with a label
        // of `Ghost`.
        let (_in_counts, mut ghost_keys) = all_to_allv(comm, &counts, &arr);
        // Add the neighbors of any global key.
        ghost_keys.extend(send_ghost_to_all.iter());

        for key in &ghost_keys {
            if key.rank == rank {
                // Keys adjacent to a global key were gathered to every rank,
                // including their own.
                continue;
            }
            let key_type = if key.is_leaf {
                KeyType::GhostLeaf(key.rank)
            } else {
                KeyType::GhostInterior(key.rank)
            };
            // Every non-global key belongs to exactly one rank, so no two ranks
            // advertise the same key and a received key never overwrites a local,
            // global, or differently classified entry.
            #[cfg(debug_assertions)]
            if let Some(&existing) = all_keys.get(&key.key) {
                debug_assert_eq!(
                    existing, key_type,
                    "a received ghost key must not reclassify an existing key"
                );
            }
            all_keys.insert(key.key, key_type);
        }
    }

    // The ghost-children guarantee, checked on the map that is returned.
    #[cfg(debug_assertions)]
    if options.ghost_children() {
        for (&key, &key_type) in all_keys.iter() {
            if key_type.is_ghost() {
                continue;
            }
            for neighbour in morton::neighbours(key)
                .iter()
                .copied()
                .filter(|&neighbour| morton::is_valid(neighbour))
            {
                let Some(&neighbour_type) = all_keys.get(&neighbour) else {
                    continue;
                };
                if matches!(neighbour_type, KeyType::LocalLeaf | KeyType::GhostLeaf(_)) {
                    continue;
                }
                let children = morton::children(neighbour)
                    .expect("an interior key is not at the deepest level");
                for child in children {
                    debug_assert!(
                        all_keys.contains_key(&child),
                        "the ghost-children layer must resolve every child of an interior neighbour"
                    );
                }
            }
        }
    }

    all_keys
}

/// Compute the neighbours of all non-ghost keys.
/// # Parameters
///
/// - `all_keys`: Classified local, global, and ghost keys from which adjacency is derived.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let neighbours = compute_neighbours(&std::collections::HashMap::new());
/// ```
pub fn compute_neighbours(
    all_keys: &HashMap<MortonKey, KeyType>,
) -> HashMap<MortonKey, Vec<MortonKey>> {
    let mut neighbours = HashMap::<MortonKey, Vec<MortonKey>>::new();
    for (key, key_type) in all_keys.iter().filter(|(_, key_type)| !key_type.is_ghost()) {
        if *key_type == KeyType::LocalInterior || (*key_type == KeyType::Global) {
            // For interior keys there always exists neighbours on the same level since the tree is balanced.
            // A global key is always interior since it is a parent of a coarse tree key.
            neighbours.insert(
                *key,
                morton::neighbours(*key)
                    .iter()
                    .copied()
                    .filter(|key| morton::is_valid(*key))
                    .collect_vec(),
            );
            continue;
        }

        // For leaf keys we need to check if the neighbours exist on the same level or above.

        if *key_type == KeyType::LocalLeaf {
            // A root leaf has no neighbours, but still needs a map entry.
            let entry = neighbours.entry(*key).or_default();
            for neighbour in morton::neighbours(*key)
                .iter()
                .copied()
                .filter(|key| morton::is_valid(*key))
            {
                let neighbour = if all_keys.contains_key(&neighbour) {
                    // The easy case. The neighbour itself is in the tree.
                    neighbour
                } else {
                    // The neighbour is on the next level closer to root.
                    // Note, we cannot mistakenly add the parent of a sibling since the tree is complete.
                    let parent = morton::parent(neighbour).unwrap();
                    debug_assert!(all_keys.contains_key(&parent)); // Neighbour parent must be in `all_keys`.
                    parent
                };
                // Several directions can share one coarser neighbour; list it once.
                if !entry.contains(&neighbour) {
                    entry.push(neighbour);
                }
                // Note that the case cannot happen that neither the neighbor nor its parent exists. In a 2:1 tree
                // the only other case could be that the neighbours children are in the tree. But in that case the
                // neighbour itself is also in the tree as its children are.
            }
            continue;
        }
    }
    neighbours
}

#[cfg(test)]
mod test {
    use itertools::{Itertools, izip};
    use rand::RngExt;
    use rlst::distributed_tools::sort_to_bins;

    use std::collections::HashMap;

    use crate::{
        constants::{DEEPEST_LEVEL, NSIBLINGS},
        morton,
        octree::{
            KeyType, LeafLocation, LookupError, OctreeOptions, checked_predecessor,
            compute_neighbours, counts_fit_mpi_i32, create_local_tree, generate_all_keys,
            get_key_index, local_coarse_tree_weights, local_leaf_for_key, owner_rank_for_key,
            partition_blocks, refine_heavy_blocks, tree_bins, validate_fine_keys,
            validate_lookup_key,
        },
        tools::{generate_random_keys, seeded_rng},
    };

    /// On one rank the ghost exchange is skipped entirely, so the ghost-children
    /// option must not change the generated key map. The tree below is adaptive,
    /// so it has interior keys at two levels and leaves at three.
    #[test]
    fn test_ghost_children_option_is_a_noop_on_one_rank() {
        let _universe = mpi::initialize().expect("this test owns MPI initialization");
        let comm = mpi::topology::SimpleCommunicator::self_comm();

        let octants = morton::children(morton::root()).unwrap();
        let grandchildren = morton::children(octants[0]).unwrap();
        let mut leaves = octants[1..].to_vec();
        leaves.extend_from_slice(&grandchildren[1..]);
        leaves.extend_from_slice(&morton::children(grandchildren[0]).unwrap());
        leaves.sort_unstable();
        let coarse_tree = octants.to_vec();
        let bounds = vec![octants[0]];

        let baseline = generate_all_keys(
            &leaves,
            &coarse_tree,
            &bounds,
            OctreeOptions::default(),
            &comm,
        );
        let with_children = generate_all_keys(
            &leaves,
            &coarse_tree,
            &bounds,
            OctreeOptions::new().with_ghost_children(true),
            &comm,
        );
        assert_eq!(baseline, with_children);

        // A single rank holds the whole tree, so the guarantee is met either way.
        assert!(baseline.values().all(|key_type| !key_type.is_ghost()));
        for (&key, &key_type) in &baseline {
            if key_type == KeyType::LocalLeaf {
                continue;
            }
            for child in morton::children(key).unwrap() {
                assert!(baseline.contains_key(&child));
            }
        }
    }

    #[test]
    fn test_lookup_validation_and_checked_searches() {
        let first = morton::deepest_first();
        let last = morton::deepest_last();
        assert_eq!(
            validate_lookup_key(morton::invalid_key()),
            Err(LookupError::InvalidKey)
        );
        assert_eq!(validate_lookup_key(u64::MAX), Err(LookupError::InvalidKey));
        assert_eq!(
            validate_lookup_key(morton::root()),
            Err(LookupError::NotFinestLevel { level: 0 })
        );
        assert_eq!(
            validate_lookup_key(17),
            Err(LookupError::NotFinestLevel { level: 17 })
        );
        assert_eq!(validate_lookup_key(first), Ok(()));
        assert_eq!(morton::level(last), DEEPEST_LEVEL as usize);

        let bounds = [first, morton::from_index_and_level([1, 0, 0], 1)];
        assert_eq!(owner_rank_for_key(&bounds, first), Some(0));
        assert_eq!(owner_rank_for_key(&bounds, last), Some(1));
        assert_eq!(checked_predecessor(&[], first), None);
        assert_eq!(checked_predecessor(&bounds, 0), None);
    }

    #[test]
    fn test_local_leaf_checked_containment_and_count_limits() {
        let root = morton::root();
        let query = morton::from_index_and_level([3, 2, 1], DEEPEST_LEVEL as usize);
        assert_eq!(local_leaf_for_key(&[root], query), Some(root));
        assert_eq!(local_leaf_for_key(&[query], query), Some(query));
        assert_eq!(local_leaf_for_key(&[], query), None);
        let later = morton::from_index_and_level([4, 0, 0], DEEPEST_LEVEL as usize);
        assert_eq!(local_leaf_for_key(&[later], query), None);
        assert!(counts_fit_mpi_i32(&[0, i32::MAX as usize]));
        assert!(!counts_fit_mpi_i32(&[i32::MAX as usize, 1]));
        assert!(!counts_fit_mpi_i32(&[usize::MAX]));
    }

    #[test]
    fn test_lookup_mixed_leaf_intervals_and_partition_boundaries() {
        let root_children = morton::children(morton::root()).unwrap();
        let mut leaves = morton::children(root_children[0]).unwrap().to_vec();
        leaves.extend_from_slice(&root_children[1..]);
        leaves.sort_unstable();
        let finest_start = |key| {
            let (level, index) = morton::decode(key);
            let shift = DEEPEST_LEVEL as usize - level;
            morton::from_index_and_level(
                [index[0] << shift, index[1] << shift, index[2] << shift],
                DEEPEST_LEVEL as usize,
            )
        };

        let refined_start = finest_start(leaves[0]);
        let refined_shift = DEEPEST_LEVEL as usize - morton::level(leaves[7]);
        let refined_end = finest_start(leaves[7]) + ((1u64 << (3 * refined_shift)) - 1) * (1 << 15);
        assert_eq!(local_leaf_for_key(&leaves, refined_start), Some(leaves[0]));
        assert_eq!(local_leaf_for_key(&leaves, refined_end), Some(leaves[7]));
        let coarse_start = finest_start(root_children[1]);
        assert_eq!(
            local_leaf_for_key(&leaves, coarse_start),
            Some(root_children[1])
        );

        let bounds = [morton::deepest_first(), coarse_start];
        assert_eq!(owner_rank_for_key(&bounds, coarse_start), Some(1));
        assert_eq!(
            owner_rank_for_key(&bounds, coarse_start - (1 << 15)),
            Some(0)
        );

        // A predecessor can exist but still not contain the query.
        assert_eq!(
            local_leaf_for_key(&root_children[..2], finest_start(root_children[2])),
            None
        );
    }

    #[test]
    fn test_lookup_packing_and_reply_restoration() {
        let split = morton::from_index_and_level([32768, 0, 0], DEEPEST_LEVEL as usize);
        let keys = [
            split,
            morton::root(),
            morton::deepest_first(),
            split,
            morton::invalid_key(),
        ];
        let bounds = [morton::deepest_first(), split];
        let packed = super::pack_lookup_queries(&keys, &bounds, 2);
        assert_eq!(packed.counts, vec![1, 2]);
        assert_eq!(packed.keys, vec![morton::deepest_first(), split, split]);
        assert_eq!(packed.original_indices, vec![2, 0, 3]);
        assert_eq!(
            packed.results[1],
            Some(Err(LookupError::NotFinestLevel { level: 0 }))
        );
        assert_eq!(packed.results[4], Some(Err(LookupError::InvalidKey)));

        let mut results = packed.results;
        super::scatter_lookup_replies(
            &mut results,
            packed.original_indices,
            [11, 22, 33],
            &keys,
            &bounds,
        );
        assert_eq!(
            results,
            vec![
                Some(Ok(LeafLocation {
                    leaf: 22,
                    owner_rank: 1
                })),
                Some(Err(LookupError::NotFinestLevel { level: 0 })),
                Some(Ok(LeafLocation {
                    leaf: 11,
                    owner_rank: 0
                })),
                Some(Ok(LeafLocation {
                    leaf: 33,
                    owner_rank: 1
                })),
                Some(Err(LookupError::InvalidKey)),
            ]
        );

        let empty =
            super::pack_lookup_queries(&[morton::root(), morton::invalid_key()], &bounds, 2);
        assert_eq!(empty.counts, vec![0, 0]);
        assert!(empty.keys.is_empty());
        assert!(empty.original_indices.is_empty());
    }

    #[test]
    fn test_get_key_rank() {
        let mut rng = seeded_rng(0);

        let mut keys = generate_random_keys(50, &mut rng);

        keys.sort_unstable();

        let mid = keys[25];

        assert_eq!(Some(25), get_key_index(&keys, mid));

        // Now remove the mid index and do the same again.

        keys.remove(25);

        // The result should be 24.

        assert_eq!(Some(24), get_key_index(&keys, mid));

        // A key ordered below every element has no predecessor.

        assert_eq!(None, get_key_index(&keys, morton::root()));
    }

    #[test]
    fn test_create_local_tree_refines_only_crowded_blocks() {
        let deepest = DEEPEST_LEVEL as usize;
        // Two neighbouring finest-level keys that share every ancestor but the last.
        let fine = [
            morton::deepest_first(),
            morton::from_index_and_level([1, 0, 0], deepest),
        ];
        let root = morton::root();

        // A block whose key count is within budget is kept as a leaf.
        assert_eq!(create_local_tree(&fine, &[root], deepest, 2), vec![root]);
        // So is a block that may not be refined any further, whatever its count.
        assert_eq!(create_local_tree(&fine, &[root], 0, 1), vec![root]);

        // A budget of one forces refinement all the way to the deepest level,
        // because the two keys only separate there.
        let leaves = create_local_tree(&fine, &[root], deepest, 1);
        assert!(morton::is_complete_linear_octree(&leaves));
        assert_eq!(
            leaves.iter().map(|&key| morton::level(key)).max(),
            Some(deepest)
        );
        for &key in &fine {
            assert_eq!(
                leaves
                    .iter()
                    .filter(|&&leaf| morton::is_ancestor(leaf, key))
                    .count(),
                1
            );
        }
        // Only the chain of blocks containing the keys is refined: each of the
        // levels above the deepest one contributes its other seven cells, and the
        // final split contributes all eight.
        assert_eq!(leaves.len(), 7 * (deepest - 1) + NSIBLINGS);

        // `max_level` caps the refinement and is clamped to the deepest level.
        let shallow = create_local_tree(&fine, &[root], 2, 1);
        assert!(morton::is_complete_linear_octree(&shallow));
        assert!(shallow.iter().all(|&key| morton::level(key) <= 2));
        assert_eq!(create_local_tree(&fine, &[root], 100, 1), leaves);

        // A coarse tree of several blocks keeps the empty ones untouched and
        // refines each crowded block independently.
        let octants = morton::children(root).unwrap();
        let refined = create_local_tree(&fine, &octants, 3, 1);
        assert!(morton::is_complete_linear_octree(&refined));
        assert_eq!(
            refined
                .iter()
                .filter(|&&key| octants[1..].contains(&key))
                .count(),
            7
        );
        assert!(!refined.contains(&octants[0]));
    }

    #[test]
    fn test_compute_neighbours_uses_coarser_parents_and_skips_ghosts() {
        // One level-one octant is refined; the remaining seven stay leafs, one of
        // them owned by another rank.
        let octants = morton::children(morton::root()).unwrap();
        let refined = morton::children(octants[0]).unwrap();
        let ghost = octants[7];

        let mut all_keys = HashMap::<morton::MortonKey, KeyType>::new();
        all_keys.insert(morton::root(), KeyType::Global);
        all_keys.insert(octants[0], KeyType::LocalInterior);
        for &octant in &octants[1..] {
            all_keys.insert(octant, KeyType::LocalLeaf);
        }
        all_keys.insert(ghost, KeyType::GhostLeaf(1));
        for &key in &refined {
            all_keys.insert(key, KeyType::LocalLeaf);
        }

        let neighbours = compute_neighbours(&all_keys);

        // Ghosts get no entry of their own, every other key does.
        assert!(!neighbours.contains_key(&ghost));
        assert_eq!(neighbours.len(), all_keys.len() - 1);

        // All 26 neighbours of the root are invalid, so its list is empty.
        assert_eq!(neighbours[&morton::root()], Vec::new());

        // An interior key keeps its same-level neighbours.
        assert_eq!(
            neighbours[&octants[0]],
            morton::neighbours(octants[0])
                .iter()
                .copied()
                .filter(|&key| morton::is_valid(key))
                .collect_vec()
        );

        // The refined cell at the centre of the domain touches all 26 neighbouring
        // cells: seven siblings on its own level, the other nineteen only through
        // their parents, the seven coarser level-one leafs, each listed once.
        let centre = morton::from_index_and_level([1, 1, 1], 2);
        assert!(refined.contains(&centre));
        let entry = &neighbours[&centre];
        assert_eq!(entry.len(), 2 * (NSIBLINGS - 1));
        assert_eq!(entry.iter().unique().count(), entry.len());
        let (same_level, coarser): (Vec<_>, Vec<_>) = entry
            .iter()
            .partition(|&&key| morton::level(key) == morton::level(centre));
        assert_eq!(same_level.len(), NSIBLINGS - 1);
        assert!(
            same_level
                .iter()
                .all(|&&key| refined.contains(&key) && key != centre)
        );
        assert_eq!(
            coarser.iter().copied().sorted().collect_vec(),
            octants[1..].iter().collect_vec()
        );
        // A ghost is not a key of the map, but it is still a neighbour value.
        assert!(entry.contains(&ghost));
    }

    #[test]
    fn test_validate_fine_keys_names_the_offending_key() {
        let first = morton::deepest_first();
        assert_eq!(validate_fine_keys(&[]), Ok(()));
        assert_eq!(
            validate_fine_keys(&[first, first, morton::deepest_last()]),
            Ok(())
        );

        let invalid = validate_fine_keys(&[first, morton::invalid_key()]).unwrap_err();
        assert!(invalid.contains("position 1"));
        assert!(invalid.contains("invalid"));

        let coarse = validate_fine_keys(&[morton::from_index_and_level([1, 0, 0], 5)]).unwrap_err();
        assert!(coarse.contains("position 0"));
        assert!(coarse.contains("level 5"));

        // The root is a valid key but not a construction key.
        assert!(validate_fine_keys(&[morton::root()]).is_err());
    }

    /// The cut of `partition_blocks`, by brute force: boundary p is the last boundary
    /// at or after boundary p - 1 whose prefix weight is closest to W p / P.
    fn nearest_boundaries(weights: &[u64], size: usize) -> Vec<usize> {
        let total: u64 = weights.iter().sum();
        let prefix = std::iter::once(0)
            .chain(weights.iter().scan(0, |sum, &weight| {
                *sum += weight;
                Some(*sum)
            }))
            .collect_vec();
        let mut bounds = vec![0];
        for p in 1..size {
            let target = total as i128 * p as i128;
            let distance = |b: usize| (size as i128 * prefix[b] as i128 - target).abs();
            let start = *bounds.last().unwrap();
            let best = (start..=weights.len()).map(distance).min().unwrap();
            let last_best = (start..=weights.len())
                .rev()
                .find(|&b| distance(b) == best)
                .unwrap();
            bounds.push(last_best);
        }
        bounds.push(weights.len());
        bounds
    }

    #[test]
    fn test_partition_blocks_cuts_at_the_nearest_block_boundary() {
        // Unit weights split evenly.
        assert_eq!(partition_blocks(&[1, 1, 1, 1], 2), vec![0, 2, 4]);
        assert_eq!(partition_blocks(&[1, 1, 1, 1, 1, 1], 3), vec![0, 2, 4, 6]);
        assert_eq!(partition_blocks(&[1, 1, 1], 3), vec![0, 1, 2, 3]);
        // The overshoot of the former rule: a prefix just short of the target took
        // one block more (5 of 8 octants at 2 ranks). The nearest boundary is 4.
        assert_eq!(
            partition_blocks(&[100, 100, 100, 98, 100, 100, 100, 100], 2),
            vec![0, 4, 8]
        );
        // Zero-weight blocks between two boundaries at the same distance go to the
        // earlier range: ties take the last boundary.
        assert_eq!(partition_blocks(&[0, 2, 0, 0, 2, 0], 2), vec![0, 4, 6]);
        // A block heavier than a fair share goes whole to one range.
        assert_eq!(partition_blocks(&[1, 10, 1], 3), vec![0, 1, 2, 3]);
        assert_eq!(partition_blocks(&[10, 1, 1], 2), vec![0, 1, 3]);
        // Ranges may be empty: fewer blocks than ranges, a heavy block, no weight.
        assert_eq!(partition_blocks(&[5], 3), vec![0, 0, 1, 1]);
        assert_eq!(partition_blocks(&[5], 8), vec![0, 0, 0, 0, 1, 1, 1, 1, 1]);
        assert_eq!(partition_blocks(&[1, 30, 1], 4), vec![0, 1, 2, 2, 3]);
        assert_eq!(partition_blocks(&[0, 0, 0], 3), vec![0, 3, 3, 3]);
        assert_eq!(partition_blocks(&[], 2), vec![0, 0, 0]);
        // A single range takes everything.
        assert_eq!(partition_blocks(&[3, 0, 2], 1), vec![0, 3]);

        // Randomised, against the brute-force rule, with the bound of design O3:
        // no range weighs more than a fair share plus the heaviest block.
        let mut rng = seeded_rng(11);
        for _ in 0..500 {
            let nblocks = rng.random_range(0..40usize);
            let size = rng.random_range(1..12usize);
            let heavy = rng.random_range(1..200u64);
            let weights = (0..nblocks)
                .map(|_| match rng.random_range(0..4) {
                    0 => 0,
                    1 => heavy,
                    _ => rng.random_range(0..10u64),
                })
                .collect_vec();
            let bounds = partition_blocks(&weights, size);
            assert_eq!(bounds, nearest_boundaries(&weights, size));
            assert_eq!(bounds.len(), size + 1);
            assert_eq!((bounds[0], bounds[size]), (0, nblocks));
            assert!(bounds.iter().tuple_windows().all(|(a, b)| a <= b));
            let total: u64 = weights.iter().sum();
            let heaviest = weights.iter().copied().max().unwrap_or(0);
            for (&start, &end) in bounds.iter().tuple_windows() {
                let weight: u64 = weights[start..end].iter().sum();
                assert!(weight * size as u64 <= total + heaviest * size as u64);
            }
        }
    }

    #[test]
    fn test_tree_bins_give_empty_ranks_the_next_bound() {
        let octants = morton::children(morton::root()).unwrap();
        let after = morton::invalid_key();
        // Every range non-empty: each rank's first block.
        assert_eq!(
            tree_bins(&octants, &[0, 3, 8]),
            vec![octants[0], octants[3]]
        );
        // Empty ranges first, in the middle and last.
        assert_eq!(
            tree_bins(&octants, &[0, 0, 2, 2, 8, 8]),
            vec![octants[0], octants[0], octants[2], octants[2], after]
        );
        // The root as the only block, on the middle rank of three.
        assert_eq!(
            tree_bins(&[morton::root()], &[0, 0, 1, 1]),
            vec![morton::root(), morton::root(), after]
        );

        // The owner of every key is never a rank without blocks, and it is the rank
        // whose range holds the key's block; the bins sort keys the same way.
        let mut rng = seeded_rng(4);
        let mut keys = generate_random_keys(300, &mut rng);
        keys.sort_unstable();
        let mut blocks = morton::children(octants[3]).unwrap().to_vec();
        blocks.extend_from_slice(&octants[..3]);
        blocks.extend_from_slice(&octants[4..]);
        blocks.sort_unstable();
        for partition in [
            vec![0, 0, 5, 5, 9, 15, 15],
            vec![0, 15, 15, 15],
            vec![0, 0, 0, 15],
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ] {
            let bins = tree_bins(&blocks, &partition);
            assert!(bins.iter().tuple_windows().all(|(a, b)| a <= b));
            for &key in &keys {
                let owner = owner_rank_for_key(&bins, key).unwrap();
                assert!(partition[owner] < partition[owner + 1]);
                let block = (partition[owner]..partition[owner + 1])
                    .find(|&b| morton::is_ancestor(blocks[b], key));
                assert!(block.is_some());
            }
            let counts = sort_to_bins(&keys, &bins);
            for (owner, &count) in counts.iter().enumerate() {
                let expected = keys
                    .iter()
                    .filter(|&&key| owner_rank_for_key(&bins, key) == Some(owner))
                    .count();
                assert_eq!(count, expected);
            }
        }
    }

    #[test]
    fn test_refine_heavy_blocks_on_hand_made_counts() {
        let root = morton::root();
        let octants = morton::children(root).unwrap();
        // The root, heavier than W / (k P) with k P = 2, splits.
        assert_eq!(
            refine_heavy_blocks(&[root], &[3], &[3], 2, 1, 16),
            Some(octants.to_vec())
        );
        // One factor alone keeps a block whole: at most `max_fine_keys` distinct
        // keys (a leaf of the one-rank tree, however heavy), the level limit, or a
        // weight of at most W / (k P). With a single block W / (k P) < w for k P > 1.
        assert_eq!(refine_heavy_blocks(&[root], &[1], &[1000], 16, 1, 16), None);
        assert_eq!(refine_heavy_blocks(&[root], &[4], &[1000], 16, 4, 16), None);
        assert_eq!(refine_heavy_blocks(&[root], &[9], &[9], 16, 1, 0), None);
        assert_eq!(refine_heavy_blocks(&[root], &[9], &[9], 1, 1, 16), None);

        // Several blocks: total 100, k P = 10, so blocks above 10 with more than two
        // distinct keys split, in place, keeping Morton order and completeness.
        let distinct = [5, 50, 2, 30, 3, 3, 0, 7];
        let weights = [5, 50, 20, 11, 10, 2, 0, 2];
        let refined = refine_heavy_blocks(&octants, &distinct, &weights, 10, 2, 16).unwrap();
        let mut expected = Vec::new();
        for (index, &octant) in octants.iter().enumerate() {
            if index == 1 || index == 3 {
                expected.extend(morton::children(octant).unwrap());
            } else {
                expected.push(octant);
            }
        }
        assert_eq!(refined, expected);
        assert!(morton::is_complete_linear_octree(&refined));
        // Nothing heavy left: no round.
        assert_eq!(
            refine_heavy_blocks(&octants, &[5; 8], &[1; 8], 8, 1, 16),
            None
        );
    }

    /// The coarse tree of O1, replayed serially as if one rank held every key: the
    /// rounds of `compute_coarse_tree` without the all-reduce.
    fn serial_coarse_tree(
        distinct_keys: &[morton::MortonKey],
        weighted_keys: &[morton::MortonKey],
        shares: u64,
        max_fine_keys: usize,
        max_level: usize,
    ) -> Vec<morton::MortonKey> {
        let count = |keys: &[morton::MortonKey], blocks: &[morton::MortonKey]| {
            local_coarse_tree_weights(keys, blocks)
                .into_iter()
                .map(|n| n as u64)
                .collect_vec()
        };
        let mut blocks = vec![morton::root()];
        while let Some(refined) = refine_heavy_blocks(
            &blocks,
            &count(distinct_keys, &blocks),
            &count(weighted_keys, &blocks),
            shares,
            max_fine_keys,
            max_level,
        ) {
            blocks = refined;
        }
        morton::balance(&blocks, morton::root())
    }

    #[test]
    fn test_coarse_tree_by_weight_keeps_the_one_rank_leaves() {
        // Clustered keys with duplicates: a dense blob, a sparse cloud and one key
        // repeated many times. Every block is a node of the one-rank tree, so the
        // leaves refined and balanced from the blocks are the one-rank leaves, for
        // every refinement factor and rank count.
        let deepest = DEEPEST_LEVEL as usize;
        let mut rng = seeded_rng(21);
        let mut weighted = generate_random_keys(300, &mut rng);
        for _ in 0..700 {
            weighted.push(morton::from_index_and_level(
                [
                    40000 + rng.random_range(0..300usize),
                    1000 + rng.random_range(0..300usize),
                    20000 + rng.random_range(0..300usize),
                ],
                deepest,
            ));
        }
        weighted.extend(std::iter::repeat_n(
            morton::from_index_and_level([5, 60000, 7], deepest),
            500,
        ));
        weighted.sort_unstable();
        let distinct = morton::linearize(&weighted);

        for (max_fine_keys, max_level) in [(1, 16), (8, 16), (8, 6), (64, 16)] {
            let one_rank = morton::balance(
                &create_local_tree(&distinct, &[morton::root()], max_level, max_fine_keys),
                morton::root(),
            );
            for shares in [1, 2, 8, 64, 512] {
                let blocks =
                    serial_coarse_tree(&distinct, &weighted, shares, max_fine_keys, max_level);
                assert!(morton::is_complete_linear_and_balanced(&blocks));
                assert!(blocks.iter().all(|&b| morton::level(b) <= max_level));
                let leaves = morton::balance(
                    &create_local_tree(&distinct, &blocks, max_level, max_fine_keys),
                    morton::root(),
                );
                assert_eq!(leaves, one_rank, "{max_fine_keys} {max_level} {shares}");

                // No block that the rule could still split is heavier than
                // W / shares.
                let distinct_counts = local_coarse_tree_weights(&distinct, &blocks);
                let weights = local_coarse_tree_weights(&weighted, &blocks);
                for (&block, &d, &w) in izip!(&blocks, &distinct_counts, &weights) {
                    if d > max_fine_keys && morton::level(block) < max_level {
                        assert!(w as u64 * shares <= weighted.len() as u64);
                    }
                }
            }
        }
    }

    #[test]
    fn test_local_coarse_tree_weights_counts_descendants_per_block() {
        let octants = morton::children(morton::root()).unwrap();
        let mut rng = seeded_rng(5);
        let mut keys = generate_random_keys(200, &mut rng);
        keys.sort_unstable();

        let weights = local_coarse_tree_weights(&keys, &octants);
        assert_eq!(weights.iter().sum::<usize>(), keys.len());
        for (&block, &weight) in octants.iter().zip(&weights) {
            let expected = keys
                .iter()
                .filter(|&&key| morton::is_ancestor(block, key))
                .count();
            assert_eq!(weight, expected);
        }

        // Duplicates count once each.
        let mut doubled = keys.iter().flat_map(|&key| [key, key]).collect_vec();
        doubled.sort_unstable();
        assert_eq!(
            local_coarse_tree_weights(&doubled, &octants),
            weights.iter().map(|&w| 2 * w).collect_vec()
        );

        // A single block holds everything; no keys means no weight.
        assert_eq!(
            local_coarse_tree_weights(&keys, &[morton::root()]),
            vec![200]
        );
        assert_eq!(local_coarse_tree_weights(&[], &octants), vec![0; 8]);

        // Blocks at mixed levels.
        let mut blocks = morton::children(octants[0]).unwrap().to_vec();
        blocks.extend_from_slice(&octants[1..]);
        let weights = local_coarse_tree_weights(&keys, &blocks);
        assert_eq!(weights.iter().sum::<usize>(), keys.len());
    }
}

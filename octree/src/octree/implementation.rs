//! Parallel Octree structure

use std::collections::{HashMap, HashSet};

use crate::morton;
use crate::{
    constants::{DEEPEST_LEVEL, DIRECTIONS, LEVEL_MASK},
    morton::MortonKey,
};

use mpi::traits::Equivalence;

use itertools::{Itertools, izip};
use mpi::{collective::SystemOperation, datatype::PartitionMut, traits::CommunicatorCollectives};
use rlst::distributed_tools::{
    all_to_allv,
    array_tools::{communicate_back, gather_to_all},
    sort_to_bins,
};

use super::{
    KeyType, LeafLocation, LookupBatchError, LookupError, OctreeOptions, PartitionMap,
    PartitionWeight, Placement, last_finest_descendant,
};

/// A distinct finest-level key and its number of copies, as the key routing sends
/// it (S4, docs/design/scale-out.md §7.1).
#[derive(Copy, Clone, Debug, Default, Equivalence, PartialEq, Eq)]
pub(crate) struct KeyCount {
    /// The key.
    pub(crate) key: MortonKey,
    /// The copies of the key, on the sending rank or, after the merge, on all ranks.
    pub(crate) count: u64,
}

/// Route the input keys to provisional owners by weight, without a sort (S4,
/// docs/design/scale-out.md §7.1).
///
/// Sorts this rank's keys once, duplicates kept. Starting from the root, every
/// round counts the keys under each provisional block in one merge pass, sums the
/// counts over the ranks in one all-reduce, and splits every block heavier than
/// `W / (k P)` and above `max_level` ([`weight_rounds`]): O1's round with the
/// weight rule alone. The blocks are cut at the nearest boundaries
/// ([`partition_blocks`]), and every rank sends its keys, deduplicated, as
/// (key, multiplicity) pairs to the owners of their blocks in one all-to-all-v.
/// Each rank merges the sorted runs it receives and sums the multiplicities
/// ([`merge_key_counts`]). Every copy of a key is then on one rank, and the
/// distinct keys are the linearized keys: the input keys are all on the deepest
/// level, so linearizing is deduplication.
///
/// The provisional blocks only balance the key move; [`compute_coarse_tree`]
/// builds the coarse tree from the routed keys as before.
///
/// # Parameters
///
/// - `fine_keys`: The keys this rank passed to [`crate::Octree::new`], in any order,
///   duplicates included.
/// - `options`: The construction options: `max_level` and the block refinement
///   factor `k`.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Returns
///
/// This rank's share of the linearized keys, sorted, distinct and every distinct
/// key on exactly one rank, and the copies of each over all ranks.
///
/// # Collective operation
///
/// On more than one rank, one all-reduce per round and the all-to-all-v of the
/// pairs (with its all-to-all of counts); every rank must enter. Debug builds add a
/// check that the routed keys are distinct across the ranks. On one rank nothing
/// is communicated.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let key = morton::deepest_first();
/// let (keys, copies) = route_keys(&[key, key], OctreeOptions::default(), &comm);
/// assert_eq!((keys, copies), (vec![key], vec![2]));
/// ```
pub fn route_keys<C: CommunicatorCollectives>(
    fine_keys: &[MortonKey],
    options: OctreeOptions,
    comm: &C,
) -> (Vec<MortonKey>, Vec<u64>) {
    // The one local sort: the rounds count these keys, and their runs of equal keys
    // are the pairs.
    let mut sorted_keys = fine_keys.to_vec();
    sorted_keys.sort_unstable();

    let routed = if comm.size() == 1 {
        count_duplicates(&sorted_keys)
    } else {
        let size = comm.size() as usize;
        let max_level = options.max_level().min(DEEPEST_LEVEL as usize);
        let shares = options.block_refinement().max(1) as u64 * size as u64;
        let (blocks, weights) = weight_rounds(shares, max_level, |blocks| {
            let local = local_coarse_tree_weights(&sorted_keys, blocks)
                .into_iter()
                .map(|count| count as u64)
                .collect_vec();
            let mut global = vec![0u64; blocks.len()];
            comm.all_reduce_into(&local, &mut global, SystemOperation::sum());
            global
        });
        let bounds = tree_bins(&blocks, &partition_blocks(&weights, size));

        let pairs = count_duplicates(&sorted_keys);
        drop(sorted_keys);
        let counts = rank_counts(&pairs, &bounds, |pair| pair.key);
        let (run_lengths, received) = all_to_allv(comm, &counts, &pairs);
        drop(pairs);
        let routed = merge_key_counts(&received, &run_lengths);
        debug_assert!(is_distinct_across_ranks(
            &routed.iter().map(|pair| pair.key).collect_vec(),
            comm
        ));
        routed
    };
    routed
        .into_iter()
        .map(|pair| (pair.key, pair.count))
        .unzip()
}

/// Refine provisional blocks by weight alone, from the root (the rounds of
/// [`route_keys`]).
///
/// Every round asks `global_weights` for the weight of each block and splits every
/// block that weighs more than `W / shares` and lies above `max_level` into its
/// eight children, until no block splits. It is O1's round
/// ([`refine_heavy_blocks`]) without the distinct-key rule, which needs the keys
/// deduplicated first.
///
/// # Parameters
///
/// - `shares`: The refinement factor times the number of ranks, `k P`.
/// - `max_level`: The deepest level a block may have.
/// - `global_weights`: The weight of each of the given sorted, complete blocks,
///   summed over the ranks; called once per round.
///
/// # Returns
///
/// The blocks, sorted and complete, and their weights.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let (blocks, weights) = weight_rounds(1, 16, |blocks| vec![5; blocks.len()]);
/// assert_eq!((blocks, weights), (vec![morton::root()], vec![5]));
/// ```
pub(crate) fn weight_rounds(
    shares: u64,
    max_level: usize,
    mut global_weights: impl FnMut(&[MortonKey]) -> Vec<u64>,
) -> (Vec<MortonKey>, Vec<u64>) {
    let mut blocks = vec![morton::root()];
    loop {
        let weights = global_weights(&blocks);
        // With the weights as the distinct counts and `max_fine_keys` 0 the
        // distinct-key rule always holds for a block heavier than `W / shares`,
        // which weighs at least one key: only the weight and the level decide.
        match refine_heavy_blocks(&blocks, &weights, &weights, shares, 0, max_level) {
            Some(refined) => blocks = refined,
            None => return (blocks, weights),
        }
    }
}

/// Collapse the runs of equal keys of a sorted slice into (key, multiplicity) pairs.
///
/// # Parameters
///
/// - `sorted_keys`: Sorted keys, duplicates allowed.
///
/// # Returns
///
/// One pair per distinct key, in order.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let pairs = count_duplicates(&[1, 1, 2]);
/// assert_eq!(pairs, vec![KeyCount { key: 1, count: 2 }, KeyCount { key: 2, count: 1 }]);
/// ```
pub(crate) fn count_duplicates(sorted_keys: &[MortonKey]) -> Vec<KeyCount> {
    sorted_keys
        .chunk_by(|a, b| a == b)
        .map(|run| KeyCount {
            key: run[0],
            count: run.len() as u64,
        })
        .collect()
}

/// Merge sorted runs of (key, multiplicity) pairs, summing the multiplicities of
/// equal keys.
///
/// # Parameters
///
/// - `runs`: The runs one after another, as an all-to-all-v receives them; each
///   run sorted by key and distinct.
/// - `run_lengths`: The length of each run; runs may be empty.
///
/// # Returns
///
/// The pairs of all runs, sorted by key and distinct, with the summed copies.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let pair = |key, count| KeyCount { key, count };
/// let merged = merge_key_counts(&[pair(1, 2), pair(1, 3), pair(4, 1)], &[1, 0, 2]);
/// assert_eq!(merged, vec![pair(1, 5), pair(4, 1)]);
/// ```
pub(crate) fn merge_key_counts(runs: &[KeyCount], run_lengths: &[usize]) -> Vec<KeyCount> {
    debug_assert_eq!(run_lengths.iter().sum::<usize>(), runs.len());
    let mut rest = runs;
    let mut slices = Vec::with_capacity(run_lengths.len());
    for &length in run_lengths {
        let (run, tail) = rest.split_at(length);
        debug_assert!(run.iter().tuple_windows().all(|(a, b)| a.key < b.key));
        slices.push(run.iter().copied());
        rest = tail;
    }
    slices
        .into_iter()
        .kmerge_by(|a, b| a.key < b.key)
        .coalesce(|a, b| {
            if a.key == b.key {
                Ok(KeyCount {
                    key: a.key,
                    count: a.count + b.count,
                })
            } else {
                Err((a, b))
            }
        })
        .collect()
}

/// Count the items of a sorted slice per rank by the partition bounds.
///
/// An item goes to the last rank whose bound is at most its position, as with
/// [`sort_to_bins`]: repeated bounds (ranks without blocks) select the last of them.
/// One binary search per rank.
///
/// # Parameters
///
/// - `sorted`: Items whose positions do not decrease, none before the first bound.
/// - `bounds`: The partition bounds of every rank, from [`tree_bins`].
/// - `position`: The finest-level key that places an item.
///
/// # Returns
///
/// The number of items for each rank.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// assert_eq!(rank_counts(&[1, 3, 3, 9], &[0, 3, 3, 8], |&key| key), vec![1, 0, 2, 1]);
/// ```
pub(crate) fn rank_counts<T>(
    sorted: &[T],
    bounds: &[MortonKey],
    position: impl Fn(&T) -> MortonKey,
) -> Vec<usize> {
    let starts = bounds
        .iter()
        .map(|&bound| sorted.partition_point(|item| position(item) < bound))
        .chain(std::iter::once(sorted.len()))
        .collect_vec();
    debug_assert!(
        starts[0] == 0,
        "every item must lie at or after the first bound"
    );
    starts.iter().tuple_windows().map(|(a, b)| b - a).collect()
}

/// Return true on every rank if no rank's last key reaches the first key of the
/// next rank with keys: the keys are distinct across the ranks.
///
/// # Parameters
///
/// - `sorted_keys`: This rank's sorted keys.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Collective operation
///
/// The all-gathers of `communicate_back` and one all-reduce; every rank must enter.
/// Used in debug checks only.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert!(is_distinct_across_ranks(&[1, 2], &comm));
/// ```
fn is_distinct_across_ranks<C: CommunicatorCollectives>(
    sorted_keys: &[MortonKey],
    comm: &C,
) -> bool {
    let distinct = match (sorted_keys.last(), communicate_back(sorted_keys, comm)) {
        (Some(&last), Some(next)) => last < next,
        _ => true,
    };
    let mut all_distinct = false;
    comm.all_reduce_into(&distinct, &mut all_distinct, SystemOperation::logical_and());
    all_distinct
}

/// Build the coarse tree by weight from the root, replicated on every rank (design
/// O1, docs/design/distributed-fmm.md §3.2).
///
/// Starting from the root, every round counts for each block `b` the distinct keys
/// under it, `d(b)`, and its weight `w(b)` (every input key under it, duplicates
/// included, from the multiplicities, or `d(b)` with
/// [`PartitionWeight::DistinctKeys`]), sums both over the ranks in one all-reduce,
/// and splits into its eight children every block with `w(b) > W / (k P)`,
/// `d(b) > max_fine_keys` and a level below `max_level` ([`refine_heavy_blocks`]).
/// It stops when no block splits. The blocks are then 2:1 balanced serially; if
/// that changes them, one more all-reduce weighs the balanced blocks. The one-rank
/// tree splits exactly the boxes with more than `max_fine_keys` distinct keys
/// above `max_level`, so every block is a node of it. The blocks and weights
/// depend only on the global counts, so they do not depend on how the keys are
/// distributed over the ranks.
///
/// Every rank holds the same counts after each all-reduce, so the number of rounds
/// is the same on every rank, and a rank without keys enters each round with zero
/// counts. On one rank the coarse tree is the root and nothing is communicated.
/// # Parameters
///
/// - `linear_keys`: This rank's share of the linearized keys, from [`route_keys`]:
///   sorted, distinct, and every distinct key on exactly one rank.
/// - `multiplicities`: The copies of each linearized key passed to
///   [`crate::Octree::new`] over all ranks.
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
/// let (blocks, weights) = compute_coarse_tree(&keys, &[2], OctreeOptions::default(), &comm);
/// assert_eq!((blocks, weights), (vec![morton::root()], vec![2]));
/// ```
pub fn compute_coarse_tree<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    multiplicities: &[u64],
    options: OctreeOptions,
    comm: &C,
) -> (Vec<MortonKey>, Vec<u64>) {
    debug_assert!(is_linear_tree(linear_keys, comm));
    debug_assert_eq!(linear_keys.len(), multiplicities.len());

    // On a single node a complete coarse tree is simply the root.
    if comm.size() == 1 {
        return (vec![morton::root()], vec![multiplicities.iter().sum()]);
    }

    let max_level = options.max_level().min(DEEPEST_LEVEL as usize);
    let shares = options.block_refinement().max(1) as u64 * comm.size() as u64;
    let weight = options.partition_weight();

    let mut blocks = vec![morton::root()];
    let weights = loop {
        let (distinct, weights) =
            global_block_counts(linear_keys, multiplicities, weight, &blocks, comm);
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
        global_block_counts(linear_keys, multiplicities, weight, &blocks, comm).1
    };
    debug_assert!(morton::is_complete_linear_and_balanced(&blocks));
    (blocks, weights)
}

/// Count the distinct keys and the weight under each block, summed over the ranks.
///
/// # Parameters
///
/// - `linear_keys`: This rank's sorted distinct keys, counted as `d(b)`.
/// - `multiplicities`: The copies of each key, summed as `w(b)` with
///   [`PartitionWeight::Keys`].
/// - `weight`: What a block weighs: its keys with their copies, or `d(b)`.
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
/// let (d, w) = global_block_counts(&keys, &[3], PartitionWeight::Keys, &[morton::root()], &comm);
/// ```
fn global_block_counts<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    multiplicities: &[u64],
    weight: PartitionWeight,
    blocks: &[MortonKey],
    comm: &C,
) -> (Vec<u64>, Vec<u64>) {
    let nblocks = blocks.len();
    let (distinct, copies) = local_block_counts(linear_keys, multiplicities, blocks);
    let mut local = distinct;
    match weight {
        PartitionWeight::Keys => local.extend(copies),
        PartitionWeight::DistinctKeys => local.extend_from_within(..),
    }
    let mut global = vec![0u64; 2 * nblocks];
    comm.all_reduce_into(&local, &mut global, SystemOperation::sum());
    let weights = global.split_off(nblocks);
    (global, weights)
}

/// Count for each block the sorted distinct keys under it and their copies.
///
/// Both slices are sorted and the descendants of a block are contiguous in Morton
/// order, so a single merge pass suffices, as in [`local_coarse_tree_weights`].
/// # Parameters
///
/// - `linear_keys`: Sorted distinct finest-level keys, each descending from exactly
///   one block.
/// - `multiplicities`: The copies of each key.
/// - `blocks`: Sorted, non-overlapping blocks covering every key.
///
/// # Returns
///
/// The keys under each block and the sum of their copies.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let (d, w) = local_block_counts(&[morton::deepest_first()], &[3], &[morton::root()]);
/// assert_eq!((d, w), (vec![1], vec![3]));
/// ```
pub(crate) fn local_block_counts(
    linear_keys: &[MortonKey],
    multiplicities: &[u64],
    blocks: &[MortonKey],
) -> (Vec<u64>, Vec<u64>) {
    let mut distinct = vec![0; blocks.len()];
    let mut copies = vec![0; blocks.len()];
    let mut keys = izip!(linear_keys, multiplicities).peekable();
    for (d, w, &block) in izip!(distinct.iter_mut(), copies.iter_mut(), blocks) {
        while let Some((_, &count)) = keys.next_if(|&(&key, _)| morton::is_ancestor(block, key)) {
            *d += 1;
            *w += count;
        }
    }
    debug_assert!(
        keys.peek().is_none(),
        "every key must descend from a block of the complete coarse tree"
    );
    (distinct, copies)
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

/// The total weight that [`quantize_weights`] scales the block weights to: 2⁵², so that
/// each block keeps its weight to about 2⁻⁵² of the total and the sums of
/// [`partition_blocks`] stay exact.
const QUANTIZED_TOTAL: f64 = (1u64 << 52) as f64;

/// Turn real block weights into the integer weights of [`partition_blocks`] (Phase 5S
/// S3, docs/design/scale-out.md §6.3).
///
/// Each weight becomes `round(w 2⁵² / W)`, `W` the sum of the weights in block order, so
/// the cut by the integers is the nearest-boundary cut of the real weights up to about
/// 2⁻⁵² of the total per block. The result depends only on the weights and their order,
/// so every rank that holds the same weights computes the same integers. With no weight
/// at all every block gets zero.
///
/// # Parameters
///
/// - `weights`: Block weights in block order, finite and not negative.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// assert_eq!(quantize_weights(&[1.0, 0.0, 3.0]), vec![1 << 50, 0, 3 << 50]);
/// ```
pub(crate) fn quantize_weights(weights: &[f64]) -> Vec<u64> {
    debug_assert!(weights.iter().all(|w| w.is_finite() && *w >= 0.0));
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return vec![0; weights.len()];
    }
    let scale = QUANTIZED_TOTAL / total;
    weights
        .iter()
        .map(|&weight| (weight * scale).round() as u64)
        .collect()
}

/// The weight of the heaviest range of a cut: the largest sum of `weights` over the
/// ranges `partition[r]..partition[r + 1]` (Phase 5S S3; 0 without ranges).
///
/// # Parameters
///
/// - `weights`: Block weights in block order.
/// - `partition`: The range boundaries of a cut, from 0 to the number of blocks.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// assert_eq!(heaviest_range(&[1, 2, 3, 4], &[0, 3, 4]), 6);
/// ```
pub(crate) fn heaviest_range(weights: &[u64], partition: &[usize]) -> u128 {
    partition
        .windows(2)
        .map(|range| {
            weights[range[0]..range[1]]
                .iter()
                .map(|&weight| weight as u128)
                .sum::<u128>()
        })
        .max()
        .unwrap_or(0)
}

/// Gather every rank's block weights to every rank, in block order (Phase 5S S3).
///
/// # Parameters
///
/// - `own`: The weight of each of this rank's blocks, in block order.
/// - `partition`: The `size + 1` range boundaries of the current cut; rank `r` passes
///   `partition[r + 1] - partition[r]` weights.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Collective operation
///
/// One all-gather-v on more than one rank, its counts from the cut; every rank must
/// enter, a rank without blocks with no weights. On one rank nothing is communicated.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// assert_eq!(gather_block_weights(&[2.0], &[0, 1], &comm), vec![2.0]);
/// ```
pub(crate) fn gather_block_weights<C: CommunicatorCollectives>(
    own: &[f64],
    partition: &[usize],
    comm: &C,
) -> Vec<f64> {
    if comm.size() == 1 {
        return own.to_vec();
    }
    let counts = partition
        .windows(2)
        .map(|range| i32::try_from(range[1] - range[0]).expect("a rank's blocks fit an MPI count"))
        .collect_vec();
    let displacements = partition[..partition.len() - 1]
        .iter()
        .map(|&offset| i32::try_from(offset).expect("the blocks fit an MPI displacement"))
        .collect_vec();
    let mut weights = vec![0.0f64; *partition.last().unwrap()];
    comm.all_gather_varcount_into(
        own,
        &mut PartitionMut::new(&mut weights[..], &counts[..], &displacements[..]),
    );
    weights
}

/// Move this rank's leaves to the ranks that own them under new partition bounds
/// (Phase 5S S3, docs/design/scale-out.md §6.3).
///
/// The leaves of every rank lie in its blocks, and the ranges of the blocks are
/// contiguous, so the leaves of a rank go to consecutive ranks and arrive sorted: the
/// received leaves, in sender order, are this rank's leaves under the new bounds.
///
/// # Parameters
///
/// - `leaves`: This rank's sorted leaves.
/// - `bounds`: The new partition bounds of every rank, from [`tree_bins`].
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Collective operation
///
/// One all-to-all-v (with its all-to-all of counts) on more than one rank; every rank
/// must enter, a rank without leaves sending nothing. On one rank nothing is
/// communicated.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let root = [morton::root()];
/// assert_eq!(move_leaves(&root, &root, &comm), vec![morton::root()]);
/// ```
pub(crate) fn move_leaves<C: CommunicatorCollectives>(
    leaves: &[MortonKey],
    bounds: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    if comm.size() == 1 {
        return leaves.to_vec();
    }
    let counts = rank_counts(leaves, bounds, |&key| first_finest_descendant(key));
    let (_received_counts, received) = all_to_allv(comm, &counts, leaves);
    debug_assert!(
        received.iter().tuple_windows().all(|(a, b)| a < b),
        "the moved leaves arrive sorted"
    );
    debug_assert!(
        received
            .iter()
            .all(|&key| owner_rank_for_key(bounds, key) == Some(comm.rank() as usize)),
        "every moved leaf is owned by the rank that receives it"
    );
    received
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

/// Balance a distributed tree, without a sort (S4, docs/design/scale-out.md §7.1).
///
/// Every rank builds the 2:1 closure of its own keys ([`balance_closure`]) and
/// linearizes it locally. It then routes each remaining key to the rank whose
/// partition range holds the key's first finest-level descendant
/// ([`first_finest_descendant`]), in one all-to-all-v, and every rank linearizes
/// what it receives. The union of the local results is the global linearization
/// of the closure, already distributed by the bounds:
/// - a key coarser than the blocks goes to the rank that owns its first
///   descendant block, whose leaves, or their descendants in the closure, descend
///   from it and go to the same rank, so the key is removed locally as the global
///   linearization removes it;
/// - a key at or below a block lies in that block's range with all its
///   descendants;
/// - duplicates from several ranks land on one rank.
///
/// The result has no key coarser than a block, so every key is owned by this rank
/// as [`crate::Octree::owner_rank`] defines it.
///
/// A rank may hold no keys; it still enters every collective.
/// # Parameters
///
/// - `linear_keys`: Locally held sorted, non-overlapping Morton keys, each a
///   coarse block of this rank or a descendant of one.
/// - `coarse_tree_bounds`: The partition bounds of every rank, from [`tree_bins`].
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Collective operation
///
/// An all-reduce of the deepest level and, on more than one rank and below the
/// root, the all-to-all-v of the closure (with its all-to-all of counts); every
/// rank must enter. Debug builds add the collectives of their invariant checks.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let balanced = balance(&[morton::root()], &[morton::root()], &comm);
/// ```
pub fn balance<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    coarse_tree_bounds: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    // Treat the case that the tree is only the root, on one rank: the closure would
    // be empty, as it only iterates up to level 1. The deepest level is global, so
    // every rank takes the same branch, also a rank without keys.

    let deepest_level = deepest_level(linear_keys, comm);
    if deepest_level == 0 {
        return linear_keys.to_vec();
    }

    let closure = morton::linearize(&balance_closure(linear_keys, deepest_level));
    let result = if comm.size() == 1 {
        closure
    } else {
        let counts = rank_counts(&closure, coarse_tree_bounds, |&key| {
            first_finest_descendant(key)
        });
        let (_in_counts, received) = all_to_allv(comm, &counts, &closure);
        drop(closure);
        morton::linearize(&received)
    };

    #[cfg(debug_assertions)]
    {
        let rank = comm.rank() as usize;
        debug_assert!(
            result
                .iter()
                .all(|&key| owner_rank_for_key(coarse_tree_bounds, key) == Some(rank)),
            "every balanced key must be owned by the rank that holds it"
        );
    }
    debug_assert!(crate::octree::is_complete_linear_and_balanced(
        &result, comm
    ));
    result
}

/// The local 2:1 closure of a rank's keys: for every key, its siblings, and for
/// every parent, the neighbours of the parent, level by level from
/// `deepest_level` up to level 1.
///
/// The closure of the union of the ranks' keys is the union of their closures, so
/// its global linearization is the balanced tree ([`morton::balance`] on one rank).
///
/// # Parameters
///
/// - `linear_keys`: Locally held sorted, non-overlapping Morton keys.
/// - `deepest_level`: The deepest level of the keys over all ranks, at least 1.
///
/// # Returns
///
/// The closure, unsorted, with duplicates.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let octants = morton::children(morton::root()).unwrap();
/// let closure = balance_closure(&octants, 1);
/// assert_eq!(morton::linearize(&closure), octants.to_vec());
/// ```
pub(crate) fn balance_closure(linear_keys: &[MortonKey], deepest_level: usize) -> Vec<MortonKey> {
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

    result
}

/// The first finest-level descendant of a key: the finest key at its anchor.
///
/// It places a key of any level in the partition: the rank whose range holds it
/// owns the key's first descendant block, or the block the key lies in.
///
/// # Parameters
///
/// - `key`: A valid Morton key.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// assert_eq!(first_finest_descendant(morton::root()), morton::deepest_first());
/// ```
pub(crate) fn first_finest_descendant(key: MortonKey) -> MortonKey {
    // The position bits of a key are those of its anchor on the deepest level, so only
    // the level changes.
    (key & !LEVEL_MASK) | DEEPEST_LEVEL
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

/// Gather whether every block is a leaf of the tree: the leaf bits of the partition map
/// (Phase 5S S1; docs/design/scale-out.md §3.2).
///
/// A block that [`create_local_tree`] leaves whole can still be split by the 2:1 balance,
/// so the bits are known only after it. Every rank contributes one flag per own block;
/// the counts follow from the cut, which every rank holds, so no count is exchanged.
/// # Parameters
///
/// - `leaf_tree`: This rank's sorted leaves.
/// - `own_blocks`: This rank's blocks, sorted.
/// - `partition`: The `P + 1` range boundaries of [`partition_blocks`].
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Returns
///
/// For every block of the coarse tree, in order, whether it is a leaf.
///
/// # Collective operation
///
/// One all-gather-v on more than one rank, of one flag per block; every rank must
/// enter, a rank without blocks with no flag. On one rank nothing is communicated.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let root = [morton::root()];
/// assert_eq!(gather_leaf_blocks(&root, &root, &[0, 1], &comm), vec![true]);
/// ```
pub(crate) fn gather_leaf_blocks<C: CommunicatorCollectives>(
    leaf_tree: &[MortonKey],
    own_blocks: &[MortonKey],
    partition: &[usize],
    comm: &C,
) -> Vec<bool> {
    let own = own_blocks
        .iter()
        .map(|block| leaf_tree.binary_search(block).is_ok())
        .collect_vec();
    if comm.size() == 1 {
        return own;
    }
    let counts = partition
        .windows(2)
        .map(|range| i32::try_from(range[1] - range[0]).expect("a rank's blocks fit an MPI count"))
        .collect_vec();
    let displacements = partition[..partition.len() - 1]
        .iter()
        .map(|&offset| i32::try_from(offset).expect("the blocks fit an MPI displacement"))
        .collect_vec();
    let mut leaf_blocks = vec![false; *partition.last().unwrap()];
    comm.all_gather_varcount_into(
        &own[..],
        &mut PartitionMut::new(&mut leaf_blocks[..], &counts[..], &displacements[..]),
    );
    leaf_blocks
}

/// Generate the locally essential key set of this rank and classify every key (Phase 5S
/// S1; docs/design/scale-out.md §3.1, §3.3).
///
/// Let S be this rank's own keys (its leaves and their ancestors at or below its blocks)
/// and its *shared ancestors* (the ancestors of its blocks). The result is
///
/// > H = N(S) ∪ C(N(S)),
///
/// N(S) being S and every key of the tree that is a same-level neighbour of a key of S,
/// and C(N(S)) the children of the interior keys of N(S), which only
/// [`OctreeOptions::ghost_children`] asks for. It holds every key that a V-, W- or X-row
/// of a key of S, the halo of S or a parent-child step from S names, and nothing more. A
/// rank without blocks holds no key.
///
/// The kinds: an own leaf is [`KeyType::LocalLeaf`], an own interior key
/// [`KeyType::LocalInterior`], a shared ancestor [`KeyType::Global`], and every other key
/// a ghost of its owner ([`PartitionMap::owner`]): a key at or below a block of its
/// block's rank, an ancestor of other ranks' blocks of the rank of its first descendant
/// block. On one rank the result is the whole tree with the Phase 5 kinds (the root
/// `Global` unless it is the only leaf).
///
/// Assembled without gathering keys:
/// 1. the top-tree keys of H (blocks and their ancestors) come from the partition map,
///    with their kinds, without communication;
/// 2. every other key of H lies below another rank's block, and that rank pushes it:
///    every rank sends each of its own keys k, with k's children when k is interior and
///    the ghost-children layer is asked for, to every other rank that holds a
///    same-level neighbour cell of k in its S: the owner of the cell's block for a cell
///    at or below a block, the sharers of the cell for a top-tree cell. The receiver
///    keeps what lies in H.
///
/// For every key of S and every same-level neighbour cell of it that the result holds as
/// an interior key, the result holds all eight children of the neighbour when the
/// ghost-children layer is asked for (checked in debug builds).
/// # Parameters
///
/// - `leaf_tree`: This rank's sorted leaves.
/// - `map`: The partition map, with this rank's blocks and the leaf bits.
/// - `options`: Optional topology layers requested from [`crate::Octree::new`].
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Collective operation
///
/// On more than one rank, the push: one all-to-all-v (with its all-to-all of counts),
/// which every rank enters, a rank without blocks sending and receiving nothing. On one
/// rank nothing is communicated.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let leaves = morton::children(morton::root()).unwrap();
/// let map = PartitionMap::new(vec![morton::root()], vec![0, 1], &[false], 0);
/// let keys = generate_all_keys(&leaves, &map, OctreeOptions::default(), &comm);
/// ```
pub fn generate_all_keys<C: CommunicatorCollectives>(
    leaf_tree: &[MortonKey],
    map: &PartitionMap,
    options: OctreeOptions,
    comm: &C,
) -> HashMap<MortonKey, KeyType> {
    /// A key pushed to another rank, and whether it is a leaf there. The sender is the
    /// key's owner, which the receiver learns from the position in the exchange.
    ///
    /// `Default` is required by the collective helpers, which need a placeholder
    /// element. Such a value is never read back.
    #[derive(Copy, Clone, Default, Equivalence, PartialEq, Eq, PartialOrd, Ord)]
    struct PushedKey {
        key: MortonKey,
        is_leaf: bool,
    }

    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    debug_assert_eq!(map.rank(), rank);
    debug_assert_eq!(map.nranks(), size);
    let own_range = map.rank_blocks(rank);
    let own_blocks = &map.blocks()[own_range.clone()];

    // S: the shared ancestors, the own blocks with the kinds the map gives them, and the
    // own keys below the blocks, from the leaves up to their blocks.
    let mut all_keys = HashMap::<MortonKey, KeyType>::new();
    for &block in own_blocks {
        let mut key = block;
        while let Some(parent) = morton::parent(key) {
            if all_keys.insert(parent, KeyType::Global).is_some() {
                break;
            }
            key = parent;
        }
    }
    for (b, &block) in own_range.clone().zip(own_blocks) {
        let kind = map
            .top_kind(block)
            .expect("a block is a top-tree key of the map");
        debug_assert_eq!(
            map.is_leaf_block(b),
            leaf_tree.binary_search(&block).is_ok()
        );
        all_keys.insert(block, kind);
    }
    for &leaf in leaf_tree {
        all_keys.insert(leaf, KeyType::LocalLeaf);
        let mut key = leaf;
        while let Some(parent) = morton::parent(key)
            && !all_keys.contains_key(&parent)
        {
            all_keys.insert(parent, KeyType::LocalInterior);
            key = parent;
        }
    }

    // On one rank S is the whole tree.
    if size == 1 {
        return all_keys;
    }

    // Membership of N(S): a key of S, or a same-level neighbour of one. The keys of S
    // are exactly the non-ghost keys of the map, also once ghosts are added.
    let in_s = |all_keys: &HashMap<MortonKey, KeyType>, key: MortonKey| {
        all_keys.get(&key).is_some_and(|kind| !kind.is_ghost())
    };
    let in_n = |all_keys: &HashMap<MortonKey, KeyType>, key: MortonKey| {
        in_s(all_keys, key) || any_neighbour(key, |neighbour| in_s(all_keys, neighbour))
    };

    // 1. The top part of H, from the map: the top-tree keys among the same-level
    // neighbours of S, and with the ghost-children layer the children of the ancestors
    // among N(S), which are top-tree keys too. No key deeper than the deepest block is a
    // top-tree key. The children of an interior block of another rank lie below it and
    // come with the push.
    let deepest_block = map
        .blocks()
        .iter()
        .map(|&block| morton::level(block))
        .max()
        .unwrap();
    let mut top = Vec::new();
    for (&key, kind) in all_keys.iter() {
        if morton::level(key) > deepest_block {
            continue;
        }
        if *kind == KeyType::Global {
            top.push(key);
        }
        for neighbour in valid_neighbours(key) {
            if let Some(kind) = map.top_kind(neighbour) {
                if !kind.is_ghost() {
                    // An own block or a shared ancestor, already in S.
                    debug_assert_eq!(all_keys.get(&neighbour), Some(&kind));
                    continue;
                }
                top.push(neighbour);
            }
        }
    }
    top.sort_unstable();
    top.dedup();
    let mut top_children = Vec::new();
    for &key in &top {
        if let Some(kind @ (KeyType::GhostLeaf(_) | KeyType::GhostInterior(_))) = map.top_kind(key)
        {
            top_children.push((key, kind));
        }
        if options.ghost_children() && matches!(map.placement(key), Placement::Ancestor(_)) {
            for child in morton::children(key).expect("an ancestor of a block has children") {
                let kind = map
                    .top_kind(child)
                    .expect("a child of an ancestor of the blocks is a top-tree key");
                top_children.push((child, kind));
            }
        }
    }
    for (key, kind) in top_children {
        let existing = *all_keys.entry(key).or_insert(kind);
        debug_assert_eq!(existing, kind, "the map classifies a top-tree key once");
    }

    // 2. The push: every own key to the ranks that hold a same-level neighbour cell of
    // it in their S, with its children when it is interior and the layer is asked for.
    // A cell deeper than every block whose first finest descendant lies in this rank's
    // range is inside an own block: no target, and no search of the map.
    let own_finest = own_blocks
        .first()
        .zip(own_blocks.last())
        .map(|(&first, &last)| first_finest_descendant(first)..=last_finest_descendant(last));
    let is_own_cell = |cell: MortonKey| {
        morton::level(cell) > deepest_block
            && own_finest
                .as_ref()
                .is_some_and(|range| range.contains(&first_finest_descendant(cell)))
    };
    let mut send = vec![Vec::<PushedKey>::new(); size];
    let mut targets = Vec::new();
    for (&key, &kind) in all_keys.iter() {
        let is_leaf = match kind {
            KeyType::LocalLeaf => true,
            KeyType::LocalInterior => false,
            _ => continue,
        };
        targets.clear();
        for neighbour in valid_neighbours(key).filter(|&cell| !is_own_cell(cell)) {
            match map.placement(neighbour) {
                Placement::Block(b) | Placement::Below(b) => targets.push(map.block_owner(b)),
                Placement::Ancestor(_) => targets.extend(
                    map.sharers(neighbour)
                        .filter(|&sharer| !map.rank_blocks(sharer).is_empty()),
                ),
            }
        }
        targets.sort_unstable();
        targets.dedup();
        for &target in targets.iter().filter(|&&target| target != rank) {
            send[target].push(PushedKey { key, is_leaf });
            if !options.ghost_children() || is_leaf {
                continue;
            }
            for child in morton::children(key).expect("an interior key has children") {
                let child_kind = all_keys
                    .get(&child)
                    .expect("a local interior key holds all of its children locally");
                send[target].push(PushedKey {
                    key: child,
                    is_leaf: *child_kind == KeyType::LocalLeaf,
                });
            }
        }
    }
    let mut counts = Vec::with_capacity(size);
    let mut flat = Vec::new();
    for keys in &mut send {
        keys.sort_unstable();
        keys.dedup();
        counts.push(keys.len());
        flat.append(keys);
    }
    drop(send);
    let (received_counts, received) = all_to_allv(comm, &counts, &flat);
    drop(flat);

    // Keep what lies in H: a key of N(S), or a child of an interior key of N(S) with the
    // layer. A cell inside another rank's block that S names may lie below a leaf there,
    // so a sender cannot tell whether the receiver holds it; the receiver can.
    let mut start = 0;
    let mut ghosts = Vec::new();
    for (sender, &count) in received_counts.iter().enumerate() {
        for pushed in &received[start..start + count] {
            debug_assert_ne!(sender, rank, "no rank pushes to itself");
            debug_assert_eq!(map.owner(pushed.key), sender, "the sender owns the key");
            // Most pushed keys are children of a pushed key of N(S): the parent first.
            let keep = (options.ghost_children()
                && morton::parent(pushed.key).is_some_and(|parent| in_n(&all_keys, parent)))
                || in_n(&all_keys, pushed.key);
            if keep {
                let kind = if pushed.is_leaf {
                    KeyType::GhostLeaf(sender)
                } else {
                    KeyType::GhostInterior(sender)
                };
                ghosts.push((pushed.key, kind));
            }
        }
        start += count;
    }
    for (key, kind) in ghosts {
        // Every key below a block belongs to one rank, and a top-tree key is classified
        // by the map as its owner classifies it, so a pushed key never reclassifies one.
        let existing = *all_keys.entry(key).or_insert(kind);
        debug_assert_eq!(
            existing, kind,
            "a received ghost key must not reclassify an existing key"
        );
    }

    // The ghost-children guarantee, checked on the map that is returned.
    #[cfg(debug_assertions)]
    if options.ghost_children() {
        for (&key, &key_type) in all_keys.iter() {
            if key_type.is_ghost() {
                continue;
            }
            for neighbour in valid_neighbours(key) {
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

/// Whether `predicate` holds for a valid same-level neighbour cell of `key`, stopping at
/// the first; the key is decoded once (`morton::neighbours` encodes all 26 cells).
fn any_neighbour(key: MortonKey, mut predicate: impl FnMut(MortonKey) -> bool) -> bool {
    let (level, index) = morton::decode(key);
    let size = 1i64 << level;
    DIRECTIONS.iter().any(|direction| {
        let cell = [0, 1, 2].map(|d| index[d] as i64 + direction[d]);
        cell.iter().all(|&i| (0..size).contains(&i))
            && predicate(morton::from_index_and_level(
                cell.map(|i| i as usize),
                level,
            ))
    })
}

/// The valid same-level neighbour cells of a key.
fn valid_neighbours(key: MortonKey) -> impl Iterator<Item = MortonKey> {
    morton::neighbours(key)
        .into_iter()
        .filter(|&neighbour| morton::is_valid(neighbour))
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

    use std::collections::{BTreeMap, HashMap};

    use crate::{
        constants::{DEEPEST_LEVEL, NSIBLINGS},
        morton,
        octree::{
            KeyCount, KeyType, LeafLocation, LookupError, OctreeOptions, PartitionMap,
            balance_closure, checked_predecessor, compute_neighbours, count_duplicates,
            counts_fit_mpi_i32, create_local_tree, first_finest_descendant, generate_all_keys,
            heaviest_range, local_block_counts, local_coarse_tree_weights, local_leaf_for_key,
            merge_key_counts, owner_rank_for_key, partition_blocks, quantize_weights, rank_counts,
            refine_heavy_blocks, tree_bins, validate_fine_keys, validate_lookup_key, weight_rounds,
        },
        tools::{generate_random_keys, seeded_rng},
    };

    /// On one rank there is no push, so the ghost-children option must not change the
    /// generated key map, which is the whole tree with the Phase 5 kinds: the root
    /// `Global`, every other interior key local. The tree below is adaptive, so it has
    /// interior keys at two levels and leaves at three.
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
        let map = PartitionMap::new(vec![morton::root()], vec![0, 1], &[false], 0);

        let baseline = generate_all_keys(&leaves, &map, OctreeOptions::default(), &comm);
        let with_children = generate_all_keys(
            &leaves,
            &map,
            OctreeOptions::new().with_ghost_children(true),
            &comm,
        );
        assert_eq!(baseline, with_children);

        // A single rank holds the whole tree, so the guarantee is met either way.
        let mut expected: HashMap<_, _> = morton::get_interior_keys(&leaves)
            .into_iter()
            .map(|key| (key, KeyType::LocalInterior))
            .collect();
        expected.insert(morton::root(), KeyType::Global);
        expected.extend(leaves.iter().map(|&leaf| (leaf, KeyType::LocalLeaf)));
        assert_eq!(baseline, expected);
        for (&key, &key_type) in &baseline {
            if key_type == KeyType::LocalLeaf {
                continue;
            }
            for child in morton::children(key).unwrap() {
                assert!(baseline.contains_key(&child));
            }
        }

        // A root that is the only leaf is the only key, a local leaf.
        let map = PartitionMap::new(vec![morton::root()], vec![0, 1], &[true], 0);
        let root_only = generate_all_keys(&[morton::root()], &map, OctreeOptions::default(), &comm);
        assert_eq!(
            root_only,
            HashMap::from([(morton::root(), KeyType::LocalLeaf)])
        );
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

        assert_eq!(Some(25), checked_predecessor(&keys, mid));

        // Now remove the mid index and do the same again.

        keys.remove(25);

        // The result should be 24.

        assert_eq!(Some(24), checked_predecessor(&keys, mid));

        // A key ordered below every element has no predecessor.

        assert_eq!(None, checked_predecessor(&keys, morton::root()));
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

    /// The second cut (Phase 5S S3): real weights scaled to integers and cut by
    /// [`partition_blocks`], with heavy blocks, ties and empty ranges.
    #[test]
    fn test_cut_by_real_weights() {
        let cut = |weights: &[f64], size| partition_blocks(&quantize_weights(weights), size);

        // The scaling: integers proportional to the weights, summing to about 2⁵².
        assert_eq!(
            quantize_weights(&[1.0, 0.0, 3.0]),
            vec![1 << 50, 0, 3 << 50]
        );
        assert_eq!(quantize_weights(&[0.0, 0.0]), vec![0, 0]);
        assert!(quantize_weights(&[]).is_empty());
        // Only the ratios matter.
        assert_eq!(
            quantize_weights(&[2.5e-9, 7.5e-9]),
            quantize_weights(&[2.5e6, 7.5e6])
        );

        // The heaviest range of a cut, which the minimum gain compares.
        assert_eq!(heaviest_range(&[1, 2, 3, 4], &[0, 3, 4]), 6);
        assert_eq!(heaviest_range(&[1, 2, 3, 4], &[0, 2, 2, 4]), 7);
        assert_eq!(heaviest_range(&[], &[0, 0, 0]), 0);

        // Unit weights split evenly, as by keys.
        assert_eq!(cut(&[1.0; 4], 2), vec![0, 2, 4]);
        assert_eq!(cut(&[0.5; 6], 3), vec![0, 2, 4, 6]);
        // Heavy blocks: the cut follows the weight, not the number of blocks.
        assert_eq!(
            cut(&[9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2),
            vec![0, 1, 10]
        );
        assert_eq!(cut(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0], 2), vec![0, 6, 7]);
        // A block heavier than a fair share goes whole to one range, and the
        // neighbouring ranges may be empty.
        assert_eq!(cut(&[1.0, 30.0, 1.0], 4), vec![0, 1, 2, 2, 3]);
        assert_eq!(cut(&[100.0], 3), vec![0, 0, 1, 1]);
        // Ties: zero weights between two equally close boundaries go to the earlier
        // range (the later boundary), also in tiny real weights.
        assert_eq!(cut(&[0.0, 2.0, 0.0, 0.0, 2.0, 0.0], 2), vec![0, 4, 6]);
        assert_eq!(cut(&[0.0, 1e-300, 0.0, 0.0, 1e-300, 0.0], 2), vec![0, 4, 6]);
        // No weight at all: the first range takes every block, the others are empty.
        assert_eq!(cut(&[0.0, 0.0, 0.0], 3), vec![0, 3, 3, 3]);
        assert_eq!(cut(&[], 2), vec![0, 0, 0]);
        // One range takes everything.
        assert_eq!(cut(&[3.0, 0.0, 2.0], 1), vec![0, 3]);

        // Randomised, against the nearest boundaries of the real weights and the bound
        // of design O3: no range weighs more than a fair share plus the heaviest block
        // (up to the scaling's rounding, one unit per block).
        let mut rng = seeded_rng(17);
        for _ in 0..500 {
            let nblocks = rng.random_range(0..40usize);
            let size = rng.random_range(1..12usize);
            let heavy = rng.random_range(1.0..1e6f64);
            let weights = (0..nblocks)
                .map(|_| match rng.random_range(0..4) {
                    0 => 0.0,
                    1 => heavy,
                    _ => rng.random_range(0.0..10.0f64),
                })
                .collect_vec();
            let bounds = cut(&weights, size);
            assert_eq!(bounds.len(), size + 1);
            assert_eq!((bounds[0], bounds[size]), (0, nblocks));
            assert!(bounds.iter().tuple_windows().all(|(a, b)| a <= b));
            let total: f64 = weights.iter().sum();
            let heaviest = weights.iter().copied().fold(0.0f64, f64::max);
            for (&start, &end) in bounds.iter().tuple_windows() {
                let weight: f64 = weights[start..end].iter().sum();
                assert!(weight <= total / size as f64 + heaviest + 1e-9 * total);
            }
            // Each boundary is a nearest boundary of the real prefix weights, within the
            // rounding of the scaling.
            let mut prefix = vec![0.0f64];
            for &weight in &weights {
                prefix.push(prefix.last().unwrap() + weight);
            }
            for p in 1..size {
                let target = total * p as f64 / size as f64;
                let best = (bounds[p - 1]..=nblocks)
                    .map(|b| (prefix[b] - target).abs())
                    .fold(f64::INFINITY, f64::min);
                assert!((prefix[bounds[p]] - target).abs() <= best + 1e-9 * total.max(1.0));
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

    #[test]
    fn test_weight_rounds_on_hand_made_counts() {
        let root = morton::root();
        let octants = morton::children(root).unwrap();

        // A single block: with one share the root never weighs more than W / 1, with
        // no keys nothing weighs anything, and `max_level` 0 keeps the root whole.
        assert_eq!(
            weight_rounds(1, 16, |blocks| vec![7; blocks.len()]),
            (vec![root], vec![7])
        );
        assert_eq!(
            weight_rounds(64, 16, |blocks| vec![0; blocks.len()]),
            (vec![root], vec![0])
        );
        assert_eq!(
            weight_rounds(64, 0, |blocks| vec![9; blocks.len()]),
            (vec![root], vec![9])
        );

        // Heavy blocks, by hand: W = 100 and k P = 8, so a block splits above 12.5.
        // The root splits, then its first octant (60), then the first two of that
        // octant's children (20 each); their children are light.
        let grandchildren = morton::children(octants[0]).unwrap();
        let first = morton::children(grandchildren[0]).unwrap();
        let second = morton::children(grandchildren[1]).unwrap();
        let mut hand_made = HashMap::new();
        hand_made.insert(root, 100);
        for (blocks, weights) in [
            (octants, [60, 10, 10, 5, 5, 5, 5, 0]),
            (grandchildren, [20, 20, 5, 5, 5, 5, 0, 0]),
            (first, [10, 5, 5, 0, 0, 0, 0, 0]),
            (second, [4, 4, 4, 4, 4, 0, 0, 0]),
        ] {
            hand_made.extend(blocks.into_iter().zip(weights));
        }
        let mut rounds = 0;
        let mut weigh = |blocks: &[morton::MortonKey]| {
            rounds += 1;
            blocks.iter().map(|block| hand_made[block]).collect_vec()
        };
        let (blocks, weights) = weight_rounds(8, 16, &mut weigh);
        let mut expected = first.to_vec();
        expected.extend(second);
        expected.extend_from_slice(&grandchildren[2..]);
        expected.extend_from_slice(&octants[1..]);
        assert_eq!(blocks, expected);
        assert_eq!(
            weights,
            expected.iter().map(|block| hand_made[block]).collect_vec()
        );
        assert_eq!(rounds, 4);
        // `max_level` 2 keeps the grandchildren whole.
        let mut rounds = 0;
        let (blocks, _) = weight_rounds(8, 2, |blocks: &[morton::MortonKey]| {
            rounds += 1;
            blocks.iter().map(|block| hand_made[block]).collect_vec()
        });
        let mut expected = grandchildren.to_vec();
        expected.extend_from_slice(&octants[1..]);
        assert_eq!((blocks, rounds), (expected, 3));

        // One key, repeated: no distinct-key rule, so its block splits down to
        // `max_level`, one level per round, seven siblings left on each level.
        let key = morton::from_index_and_level([12345, 678, 9], DEEPEST_LEVEL as usize);
        let keys = vec![key; 40];
        for max_level in [5, 16] {
            let mut rounds = 0;
            let (blocks, weights) =
                weight_rounds(16, max_level, |blocks: &[morton::MortonKey]| {
                    rounds += 1;
                    local_coarse_tree_weights(&keys, blocks)
                        .into_iter()
                        .map(|count| count as u64)
                        .collect_vec()
                });
            assert_eq!(rounds, max_level + 1);
            assert_eq!(blocks.len(), 7 * max_level + 1);
            assert!(morton::is_complete_linear_octree(&blocks));
            let heavy = blocks
                .iter()
                .position(|&b| morton::is_ancestor(b, key))
                .unwrap();
            assert_eq!(morton::level(blocks[heavy]), max_level);
            assert_eq!(weights[heavy], 40);
            assert_eq!(weights.iter().sum::<u64>(), 40);
        }

        // Clustered keys: every block above `max_level` is at most W / (k P), and
        // every block below the root has a parent that was heavier.
        let mut rng = seeded_rng(29);
        let mut keys = generate_random_keys(300, &mut rng);
        for _ in 0..900 {
            keys.push(morton::from_index_and_level(
                [
                    30000 + rng.random_range(0..100usize),
                    500 + rng.random_range(0..100usize),
                    9000 + rng.random_range(0..100usize),
                ],
                DEEPEST_LEVEL as usize,
            ));
        }
        keys.sort_unstable();
        let count = |block| {
            keys.iter()
                .filter(|&&key| morton::is_ancestor(block, key))
                .count() as u64
        };
        for (shares, max_level) in [(2, 16), (16, 16), (64, 16), (512, 4)] {
            let (blocks, weights) = weight_rounds(shares, max_level, |blocks| {
                local_coarse_tree_weights(&keys, blocks)
                    .into_iter()
                    .map(|count| count as u64)
                    .collect_vec()
            });
            assert!(morton::is_complete_linear_octree(&blocks));
            for (&block, &weight) in blocks.iter().zip(&weights) {
                assert_eq!(weight, count(block));
                assert!(morton::level(block) <= max_level);
                if morton::level(block) < max_level {
                    assert!(weight * shares <= keys.len() as u64);
                }
                if let Some(parent) = morton::parent(block) {
                    assert!(count(parent) * shares > keys.len() as u64);
                }
            }
        }
    }

    #[test]
    fn test_merge_key_counts_sums_the_runs() {
        let pair = |key, count| KeyCount { key, count };
        assert_eq!(count_duplicates(&[]), Vec::new());
        assert_eq!(
            count_duplicates(&[3, 3, 5, 9, 9, 9]),
            vec![pair(3, 2), pair(5, 1), pair(9, 3)]
        );

        // Duplicates across runs; empty runs first, in the middle and last.
        let runs = [
            pair(1, 2),
            pair(4, 1),
            pair(1, 3),
            pair(2, 1),
            pair(4, 5),
            pair(9, 1),
        ];
        assert_eq!(
            merge_key_counts(&runs, &[0, 2, 0, 3, 1, 0]),
            vec![pair(1, 5), pair(2, 1), pair(4, 6), pair(9, 1)]
        );
        assert_eq!(merge_key_counts(&runs[..2], &[2]), runs[..2].to_vec());
        assert_eq!(merge_key_counts(&[], &[0, 0, 0]), Vec::new());
        assert_eq!(merge_key_counts(&[], &[]), Vec::new());

        // Randomised against a map: each rank's keys, few distinct and many
        // duplicates, deduplicated by the rank and merged by the receiver.
        let mut rng = seeded_rng(17);
        for _ in 0..100 {
            let mut expected = BTreeMap::<morton::MortonKey, u64>::new();
            let mut received = Vec::new();
            let mut lengths = Vec::new();
            for _ in 0..rng.random_range(0..9usize) {
                let mut keys = (0..rng.random_range(0..40usize))
                    .map(|_| {
                        morton::from_index_and_level(
                            [rng.random_range(0..8usize), 0, 0],
                            DEEPEST_LEVEL as usize,
                        )
                    })
                    .collect_vec();
                keys.sort_unstable();
                for &key in &keys {
                    *expected.entry(key).or_default() += 1;
                }
                let run = count_duplicates(&keys);
                lengths.push(run.len());
                received.extend(run);
            }
            assert_eq!(
                merge_key_counts(&received, &lengths),
                expected
                    .into_iter()
                    .map(|(key, count)| pair(key, count))
                    .collect_vec()
            );
        }
    }

    #[test]
    fn test_rank_counts_and_first_finest_descendants() {
        let deepest = DEEPEST_LEVEL as usize;
        let root = morton::root();
        assert_eq!(first_finest_descendant(root), morton::deepest_first());
        let key = morton::from_index_and_level([3, 5, 7], deepest);
        assert_eq!(first_finest_descendant(key), key);
        assert_eq!(
            first_finest_descendant(morton::from_index_and_level([1, 0, 1], 1)),
            morton::from_index_and_level([1 << 15, 0, 1 << 15], deepest)
        );

        // Keys of every level. The first finest descendant is the first finest key
        // under the key, and it does not decrease along sorted keys.
        let mut rng = seeded_rng(9);
        let fine = generate_random_keys(400, &mut rng);
        let mut coarse = fine
            .iter()
            .map(|&key| morton::ancestor_at_level(key, rng.random_range(0..=deepest)).unwrap())
            .collect_vec();
        coarse.sort_unstable();
        for &key in &coarse {
            let first = first_finest_descendant(key);
            assert!(morton::is_ancestor(key, first));
            assert_eq!(morton::level(first), deepest);
            assert!(
                first == morton::deepest_first() || !morton::is_ancestor(key, first - (1 << 15))
            );
        }
        assert!(
            coarse
                .iter()
                .map(|&key| first_finest_descendant(key))
                .tuple_windows()
                .all(|(a, b)| a <= b)
        );

        let mut keys = fine;
        keys.sort_unstable();
        let octants = morton::children(root).unwrap();
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
            let size = bins.len();
            // Finest keys as `sort_to_bins` sorts them, repeated and invalid bounds
            // included; no keys, no counts.
            assert_eq!(
                rank_counts(&keys, &bins, |&key| key),
                sort_to_bins(&keys, &bins)
            );
            assert_eq!(
                rank_counts(&[] as &[morton::MortonKey], &bins, |&key| key),
                vec![0; size]
            );
            // A key of any level goes to the owner of the block it lies in, or of its
            // first descendant block.
            let counts = rank_counts(&coarse, &bins, |&key| first_finest_descendant(key));
            assert_eq!(counts.iter().sum::<usize>(), coarse.len());
            let mut start = 0;
            for (rank, &count) in counts.iter().enumerate() {
                for &key in &coarse[start..start + count] {
                    let block = blocks
                        .iter()
                        .position(|&block| {
                            morton::is_ancestor(block, key)
                                || morton::is_ancestor(key, block)
                                    && first_finest_descendant(key)
                                        == first_finest_descendant(block)
                        })
                        .unwrap();
                    assert!(partition[rank] <= block && block < partition[rank + 1]);
                }
                start += count;
            }
        }
    }

    #[test]
    fn test_local_block_counts_against_the_expanded_keys() {
        let mut rng = seeded_rng(23);
        let mut weighted = generate_random_keys(200, &mut rng);
        weighted.extend_from_within(..50);
        weighted.extend(std::iter::repeat_n(weighted[7], 30));
        weighted.sort_unstable();
        let (keys, copies): (Vec<_>, Vec<_>) = count_duplicates(&weighted)
            .into_iter()
            .map(|pair| (pair.key, pair.count))
            .unzip();
        assert_eq!(keys, morton::linearize(&weighted));

        let octants = morton::children(morton::root()).unwrap();
        let mut blocks = morton::children(octants[2]).unwrap().to_vec();
        blocks.extend_from_slice(&octants[..2]);
        blocks.extend_from_slice(&octants[3..]);
        blocks.sort_unstable();
        let widen = |counts: Vec<usize>| counts.into_iter().map(|c| c as u64).collect_vec();
        let (distinct, weights) = local_block_counts(&keys, &copies, &blocks);
        assert_eq!(distinct, widen(local_coarse_tree_weights(&keys, &blocks)));
        assert_eq!(
            weights,
            widen(local_coarse_tree_weights(&weighted, &blocks))
        );
        assert_eq!(
            local_block_counts(&[], &[], &blocks),
            (vec![0; blocks.len()], vec![0; blocks.len()])
        );
    }

    /// S4's balance replayed serially on `size` ranks: the blocks of O1, the cut,
    /// every rank's refined tree, its local closure linearized and routed by the first
    /// finest descendants, and every rank's linearization of what it receives. The
    /// union is the balanced tree of all the leaves, distributed by the bounds.
    #[test]
    fn test_routed_balance_against_morton_balance() {
        let deepest = DEEPEST_LEVEL as usize;
        let mut rng = seeded_rng(31);
        let mut weighted = generate_random_keys(200, &mut rng);
        for _ in 0..600 {
            weighted.push(morton::from_index_and_level(
                [
                    50000 + rng.random_range(0..200usize),
                    3000 + rng.random_range(0..200usize),
                    41000 + rng.random_range(0..200usize),
                ],
                deepest,
            ));
        }
        weighted.extend(std::iter::repeat_n(
            morton::from_index_and_level([7, 61000, 300], deepest),
            300,
        ));
        weighted.sort_unstable();
        let distinct = morton::linearize(&weighted);

        let mut coarser_than_first_block = 0;
        for size in [1, 2, 3, 5, 8, 13] {
            for (max_fine_keys, max_level) in [(1, 16), (8, 16), (8, 5), (64, 16), (4, 2)] {
                let label = format!("{size} ranks, {max_fine_keys} keys, level {max_level}");
                let shares = 8 * size as u64;
                let blocks =
                    serial_coarse_tree(&distinct, &weighted, shares, max_fine_keys, max_level);
                let weights = local_coarse_tree_weights(&weighted, &blocks)
                    .into_iter()
                    .map(|w| w as u64)
                    .collect_vec();
                let partition = partition_blocks(&weights, size);
                let bounds = tree_bins(&blocks, &partition);
                let refined = (0..size)
                    .map(|rank| {
                        let keys = distinct
                            .iter()
                            .copied()
                            .filter(|&key| owner_rank_for_key(&bounds, key) == Some(rank))
                            .collect_vec();
                        let own = &blocks[partition[rank]..partition[rank + 1]];
                        create_local_tree(&keys, own, max_level, max_fine_keys)
                    })
                    .collect_vec();
                let leaves = refined.concat();
                let deepest_level = leaves.iter().map(|&key| morton::level(key)).max().unwrap();
                assert!(deepest_level > 0);

                // The Phase 5 balance: the union of the closures, linearized globally.
                let expected = morton::balance(&leaves, morton::root());
                let union = refined
                    .iter()
                    .flat_map(|keys| balance_closure(keys, deepest_level))
                    .collect_vec();
                assert_eq!(morton::linearize(&union), expected, "{label}");

                let mut received = vec![Vec::new(); size];
                for keys in &refined {
                    let closure = morton::linearize(&balance_closure(keys, deepest_level));
                    let counts =
                        rank_counts(&closure, &bounds, |&key| first_finest_descendant(key));
                    let mut start = 0;
                    for (rank, &count) in counts.iter().enumerate() {
                        let part = &closure[start..start + count];
                        coarser_than_first_block += part
                            .iter()
                            .filter(|&&key| owner_rank_for_key(&bounds, key) != Some(rank))
                            .count();
                        received[rank].extend_from_slice(part);
                        start += count;
                    }
                }
                let balanced = received
                    .iter()
                    .map(|keys| morton::linearize(keys))
                    .collect_vec();
                assert_eq!(balanced.concat(), expected, "{label}");
                for (rank, keys) in balanced.iter().enumerate() {
                    assert!(
                        keys.iter()
                            .all(|&key| owner_rank_for_key(&bounds, key) == Some(rank)),
                        "{label}"
                    );
                }
            }
        }
        // A closure key coarser than the first block of a rank sorts before that
        // rank's bound; by its own key it would go to an earlier rank, which holds
        // none of its descendants. The replay routed such keys.
        assert!(coarser_than_first_block > 0);
    }
}

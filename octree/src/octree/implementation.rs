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

use super::{KeyType, LeafLocation, LookupBatchError, LookupError, OctreeOptions};

/// Complete the region spanned by the first and last local key, inclusive.
///
/// The result is sorted and linear. When a rank holds a single key the two bounds
/// coincide and the region is that one key: appending it twice would repeat it.
/// # Parameters
///
/// - `first`: Smallest local key of the range.
/// - `last`: Largest local key of the range, equal to `first` for a single key.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// assert_eq!(completed_region(morton::root(), morton::root()), vec![morton::root()]);
/// ```
pub(crate) fn completed_region(first: MortonKey, last: MortonKey) -> Vec<MortonKey> {
    let mut region = morton::fill_between_keys(first, last);

    region.insert(0, first);
    if last != first {
        region.push(last);
    }

    region
}

/// Compute the global coarse tree, replicated on every rank.
///
/// Each rank contributes the coarsest keys of the region spanned by its linear keys,
/// each coarsened to `max_level` where it is deeper. The union is gathered to every
/// rank and linearized, completed, and 2:1 balanced serially, so every rank returns
/// the same complete, linear, balanced tree with no key deeper than `max_level`.
/// A rank without linear keys contributes nothing.
/// # Parameters
///
/// - `linear_keys`: Locally held sorted, non-overlapping finest-level Morton keys.
/// - `max_level`: Deepest permitted coarse-tree level; deeper values are clamped to
///   [`DEEPEST_LEVEL`].
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Panics
///
/// Panics when no rank holds any key.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let keys = vec![morton::deepest_first()];
/// let coarse = compute_coarse_tree(&keys, 16, &comm);
/// ```
pub fn compute_coarse_tree<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    max_level: usize,
    comm: &C,
) -> Vec<MortonKey> {
    debug_assert!(is_linear_tree(linear_keys, comm));

    // On a single node a complete coarse tree is simply the root.
    if comm.size() == 1 {
        return vec![morton::root()];
    }

    let max_level = max_level.min(DEEPEST_LEVEL as usize);

    // Each process selects the largest boxes of the region spanned by its keys,
    // coarsened to `max_level` so the coarse tree never forces deeper leaves.
    let largest_boxes = match (linear_keys.first(), linear_keys.last()) {
        (Some(&first), Some(&last)) => {
            let region = completed_region(first, last);
            let min_level = region.iter().map(|&key| morton::level(key)).min().unwrap();
            let capped_level = min_level.min(max_level);
            region
                .iter()
                .filter(|&&key| morton::level(key) == min_level)
                .map(|&key| morton::ancestor_at_level(key, capped_level).unwrap())
                .collect_vec()
        }
        _ => Vec::new(),
    };

    // The boxes are few per rank, so the whole coarse tree is built on every rank.
    let global_boxes = gather_to_all(&largest_boxes, comm);
    assert!(
        !global_boxes.is_empty(),
        "cannot construct an octree without any keys: no rank contributed a key"
    );

    let coarse_tree = morton::balance(
        &morton::complete_tree(&morton::linearize(&global_boxes)),
        morton::root(),
    );
    debug_assert!(morton::is_complete_linear_and_balanced(&coarse_tree));
    coarse_tree
}

/// Compute the weight of each coarse tree block as the number of linear keys it contains.
///
/// The coarse tree is the replicated tree from [`compute_coarse_tree`], so the
/// returned weights are global and identical on every rank.
/// # Parameters
///
/// - `linear_keys`: Locally held sorted, non-overlapping finest-level Morton keys.
/// - `coarse_tree`: The replicated complete, linear coarse tree.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let keys = vec![morton::deepest_first()];
/// let coarse = vec![morton::root()];
/// assert_eq!(compute_coarse_tree_weights(&keys, &coarse, &comm), vec![1]);
/// ```
pub fn compute_coarse_tree_weights<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    coarse_tree: &[MortonKey],
    comm: &C,
) -> Vec<usize> {
    let local_weights = local_coarse_tree_weights(linear_keys, coarse_tree);

    let mut global_weights = vec![0; coarse_tree.len()];
    comm.all_reduce_into(&local_weights, &mut global_weights, SystemOperation::sum());
    global_weights
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

/// Redistribute sorted keys with respect to a linear coarse tree.
/// # Parameters
///
/// - `linear_keys`: Locally held sorted, non-overlapping Morton keys.
/// - `coarse_tree`: Local partition of the complete coarse tree.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let keys = vec![morton::deepest_first()];
/// let coarse = vec![morton::root()];
/// let redistributed = redistribute_with_respect_to_coarse_tree(&keys, &coarse, &comm);
/// ```
pub fn redistribute_with_respect_to_coarse_tree<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    coarse_tree: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    let size = comm.size();

    if size == 1 {
        return linear_keys.to_vec();
    }

    // We want to globally redistribute keys so that the keys on each process are descendents
    // of the local coarse tree keys.

    // We are using here the fact that the coarse tree is complete and sorted.
    // We are sending around to each process the first local index. This
    // defines bins in which we sort our keys. The keys are then sent around to the correct
    // processes via an alltoallv operation.

    let my_first = coarse_tree
        .first()
        .expect("load balancing assigns every rank at least one coarse block");

    let global_bins = gather_to_all(std::slice::from_ref(my_first), comm);

    // We now have our bins. We go through our keys and store how
    // many keys are assigned to each rank. We are using here that
    // our keys and the coarse tree are both sorted.

    // This will store for each rank how many keys will be assigned to it.

    let rank_counts = sort_to_bins(linear_keys, &global_bins);

    // We now have the counts for each rank. Let's redistribute accordingly and return.

    let (_in_counts, result) = all_to_allv(comm, &rank_counts, linear_keys);

    // A rank whose coarse blocks contain no keys legitimately receives nothing.

    #[cfg(debug_assertions)]
    {
        // Check that the result array is sorted.

        use rlst::distributed_tools::array_tools::is_sorted_array;
        debug_assert!(is_sorted_array(&result, comm));

        // Check that the first and last result key are within the bounds
        // given by the local coarse tree.

        if let (Some(&first), Some(&last)) = (result.first(), result.last()) {
            debug_assert!(*coarse_tree.first().unwrap() <= first);
            debug_assert!(
                last < *coarse_tree.last().unwrap()
                    || morton::is_ancestor(*coarse_tree.last().unwrap(), last)
            );
        }
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

/// Assign the replicated coarse tree to ranks by weight and return this rank's blocks.
///
/// Every rank receives a contiguous, non-empty range of blocks; see
/// [`partition_blocks`] for the rule.
/// # Parameters
///
/// - `coarse_tree`: The replicated complete, linear coarse tree.
/// - `weights`: Global block weights, one for each entry of `coarse_tree`.
/// - `comm`: Communicator whose ranks share the tree.
///
/// # Panics
///
/// Panics when the tree has fewer blocks than `comm` has ranks.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let coarse = vec![morton::root()];
/// let balanced = load_balance(&coarse, &[1], &comm);
/// ```
pub fn load_balance<C: CommunicatorCollectives>(
    coarse_tree: &[MortonKey],
    weights: &[usize],
    comm: &C,
) -> Vec<MortonKey> {
    assert_eq!(coarse_tree.len(), weights.len());

    let bounds = partition_blocks(weights, comm.size() as usize);
    let rank = comm.rank() as usize;
    coarse_tree[bounds[rank]..bounds[rank + 1]].to_vec()
}

/// Split weighted blocks into `size` contiguous, non-empty ranges of near-equal weight.
///
/// Returns `size + 1` boundaries; range `p` is `bounds[p]..bounds[p + 1]`. Range `p`
/// takes the blocks that start before the weight boundary `total * (p + 1) / size`,
/// but always at least one block, and never so many that a later range would be left
/// empty. A single block heavier than a fair share therefore goes whole to one range.
/// # Parameters
///
/// - `weights`: Block weights in block order.
/// - `size`: Number of ranges, at most the number of blocks.
///
/// # Panics
///
/// Panics when there are fewer blocks than ranges.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// assert_eq!(partition_blocks(&[1, 1, 1, 1], 2), vec![0, 2, 4]);
/// ```
pub(crate) fn partition_blocks(weights: &[usize], size: usize) -> Vec<usize> {
    let nblocks = weights.len();
    assert!(
        nblocks >= size,
        "the coarse tree has {nblocks} blocks but every one of the {size} ranks needs \
         at least one: increase `max_level`, provide more distinct keys, or use fewer ranks"
    );
    let total: usize = weights.iter().sum();

    let mut bounds = Vec::with_capacity(size + 1);
    bounds.push(0);
    let mut position = 0;
    let mut weight_before = 0;
    for p in 0..size {
        let target = total * (p + 1) / size;
        let ranks_after = size - 1 - p;
        // Take one block unconditionally, then keep taking blocks that start before
        // the target while leaving one block for every later rank.
        weight_before += weights[position];
        position += 1;
        while position < nblocks - ranks_after && weight_before < target {
            weight_before += weights[position];
            position += 1;
        }
        bounds.push(position);
    }
    // Trailing blocks, which have zero weight, belong to the last rank.
    *bounds.last_mut().unwrap() = nblocks;
    bounds
}

/// Balance a distributed tree.
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
    // Treat the case that the length of the keys is one and is only the root.
    // This would lead to an empty output below as we only iterate up to level 1.

    if linear_keys.len() == 1 && *linear_keys.first().unwrap() == morton::root() {
        return vec![morton::root()];
    }

    let deepest_level = deepest_level(linear_keys, comm);

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
        .expect("cannot take the deepest level of an empty local key set");

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

/// For a complete linear bin get on each process the first key of all processes.
///
/// This information can be used to query on which process a key is living.
/// # Parameters
///
/// - `complete_linear_tree`: Complete local coarse-tree partition whose first key defines this rank's bin.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```ignore
/// // This call is illustrative; `implementation` is private to the crate.
/// let comm = mpi::topology::SimpleCommunicator::self_comm();
/// let bins = get_tree_bins(&[morton::root()], &comm);
/// ```
pub fn get_tree_bins<C: CommunicatorCollectives>(
    complete_linear_tree: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    gather_to_all(
        std::slice::from_ref(complete_linear_tree.first().expect(
            "empty local partition of the complete tree: there are fewer keys than MPI ranks",
        )),
        comm,
    )
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

    // Need to explicitly add the root at the end.
    all_keys.entry(morton::root()).or_insert(KeyType::Global);

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
        // not as a neighbour of anything.
        if options.ghost_children() {
            for &block in coarse_tree {
                let status = *all_keys
                    .get(&block)
                    .expect("a coarse block is a key of its own rank");
                let is_leaf = match status {
                    KeyType::LocalLeaf => true,
                    KeyType::LocalInterior => false,
                    // A coarse block is classified `Global` when the coarse tree is
                    // the root alone. `load_balance` produces at least one block per
                    // rank, so this branch is unreachable inside `size > 1`; it is
                    // spelled out because advertising such a block would be both
                    // pointless — a `Global` key is already on every rank — and
                    // harmful, since the receive loop would overwrite the receivers'
                    // `Global` entry with a `GhostInterior`.
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
    use itertools::Itertools;
    use rand::RngExt;

    use std::collections::HashMap;

    use crate::{
        constants::{DEEPEST_LEVEL, NSIBLINGS},
        morton,
        octree::{
            KeyType, LeafLocation, LookupError, OctreeOptions, checked_predecessor,
            completed_region, compute_neighbours, counts_fit_mpi_i32, create_local_tree,
            generate_all_keys, get_key_index, local_coarse_tree_weights, local_leaf_for_key,
            owner_rank_for_key, partition_blocks, validate_fine_keys, validate_lookup_key,
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
    fn test_completed_region_is_linear_for_single_and_spanning_ranges() {
        // A rank holding one key spans a region of exactly that key. Repeating it
        // here would make the coarse tree non-linear.
        let only = morton::from_index_and_level([1000, 0, 0], DEEPEST_LEVEL as usize);
        assert_eq!(completed_region(only, only), vec![only]);
        assert_eq!(
            completed_region(morton::root(), morton::root()),
            vec![morton::root()]
        );

        // A spanning range keeps both bounds and stays sorted and non-overlapping.
        let mut rng = seeded_rng(3);
        let mut keys = generate_random_keys(32, &mut rng);
        keys.sort_unstable();
        let (first, last) = (keys[0], keys[31]);

        let region = completed_region(first, last);

        assert_eq!(region.first(), Some(&first));
        assert_eq!(region.last(), Some(&last));
        assert!(region.iter().tuple_windows().all(|(a, b)| a < b));
        assert!(
            region
                .iter()
                .tuple_windows()
                .all(|(&a, &b)| !morton::is_ancestor(a, b))
        );
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

    #[test]
    fn test_partition_blocks_is_contiguous_non_empty_and_weight_balanced() {
        // Unit weights split evenly.
        assert_eq!(partition_blocks(&[1, 1, 1, 1], 2), vec![0, 2, 4]);
        assert_eq!(partition_blocks(&[1, 1, 1, 1, 1, 1], 3), vec![0, 2, 4, 6]);
        // Exactly one unit block per rank gives one block per rank, including rank 0.
        assert_eq!(partition_blocks(&[1, 1, 1], 3), vec![0, 1, 2, 3]);
        // Zero-weight blocks travel with the rank whose range they start in, and
        // trailing zero-weight blocks go to the last rank.
        assert_eq!(
            partition_blocks(&[0, 0, 1, 0, 1, 0, 1, 0], 3),
            vec![0, 3, 5, 8]
        );
        // A block heavier than a fair share goes whole to one rank and the others
        // still receive a block each.
        assert_eq!(partition_blocks(&[399, 0, 0, 1, 0], 2), vec![0, 1, 5]);
        assert_eq!(partition_blocks(&[10, 0, 0], 3), vec![0, 1, 2, 3]);
        assert_eq!(partition_blocks(&[0, 10, 0], 3), vec![0, 1, 2, 3]);
        // A single rank takes everything.
        assert_eq!(partition_blocks(&[3, 0, 2], 1), vec![0, 3]);

        // Randomised: contiguous, every range non-empty, and no range starts past the
        // weight boundary unless forced to take a block.
        let mut rng = seeded_rng(11);
        for _ in 0..200 {
            let nblocks = rng.random_range(1..40usize);
            let size = rng.random_range(1..=nblocks);
            let weights = (0..nblocks)
                .map(|_| rng.random_range(0..5usize))
                .collect_vec();
            let bounds = partition_blocks(&weights, size);
            assert_eq!(bounds.len(), size + 1);
            assert_eq!(bounds[0], 0);
            assert_eq!(bounds[size], nblocks);
            assert!(bounds.iter().tuple_windows().all(|(a, b)| a < b));
        }
    }

    #[test]
    #[should_panic(expected = "fewer")]
    fn test_partition_blocks_rejects_fewer_blocks_than_ranks() {
        let _ = partition_blocks(&[1, 1], 3);
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

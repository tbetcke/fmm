//! An FMM tree holds the source points, target points, and the underlying Octree

#[cfg(test)]
#[path = "fmm_tree_tests.rs"]
mod tests;

use std::collections::HashMap;

use itertools::{Itertools, izip};
use mpi::traits::Communicator;
use nd_octree::{MortonKey, Octree, OctreeOptions, morton::is_ancestor};
use rlst::{
    RawAccess, Stride, ValueArrayImpl,
    distributed_tools::{DataPermutation, all_to_allv, sort_to_bins},
    sparse::distributed_array::DistributedArray,
};

pub struct FmmTree<'a, C>
where
    C: Communicator,
{
    /// The permutation of sources to the octree layout
    source_permutation: DataPermutation<'a, C>,
    /// The permutation of targets to the octree layout
    target_permutation: DataPermutation<'a, C>,
    /// The original global source indices in tree order
    source_indices: Vec<usize>,
    /// The original global target indices in tree order
    target_indices: Vec<usize>,
    /// The indptr mapping source index positions to the corresponding leaf keys
    source_indptr: Vec<usize>,
    /// The indptr mapping target index positions to the corresponding leaf keys
    target_indptr: Vec<usize>,
    /// The underlying octree structure
    octree: Octree<'a, C>,
}

impl<'a, C> FmmTree<'a, C>
where
    C: Communicator,
{
    pub fn new<
        SourceArrayImpl: ValueArrayImpl<u64, 1> + RawAccess<Item = u64> + Stride<1>,
        TargetArrayImpl: ValueArrayImpl<u64, 1> + RawAccess<Item = u64> + Stride<1>,
    >(
        source_keys: &DistributedArray<'a, C, SourceArrayImpl, 1>,
        target_keys: &DistributedArray<'a, C, TargetArrayImpl, 1>,
        options: OctreeOptions,
        comm: &'a C,
    ) -> Self {
        let nsources = source_keys.len();
        let ntargets = target_keys.len();

        assert!(source_keys.local.is_contiguous());
        assert!(target_keys.local.is_contiguous());

        let source_layout = source_keys.index_layout.clone();
        let target_layout = target_keys.index_layout.clone();

        let source_keys = source_keys.local.data().unwrap();
        let target_keys = target_keys.local.data().unwrap();

        // fine_keys contains the union of all keys
        let mut fine_keys = Vec::with_capacity(nsources + ntargets);
        fine_keys.extend_from_slice(source_keys);
        fine_keys.extend_from_slice(target_keys);

        // First create the octree. The ghost-children layer makes every entry of
        // the interaction lists a key of `Octree::all_keys` (see
        // [`crate::interaction_manager`]), so it is enabled whatever the caller
        // requested.
        let octree = Octree::new(&fine_keys, options.with_ghost_children(true), comm);

        // We need to sort the source and target keys and their indices. To do this
        // we push keys and indices together in a struct and sort them together.

        let (sorted_source_indices, sorted_source_keys): (Vec<_>, Vec<_>) = source_keys
            .iter()
            .enumerate()
            .map(|(ind, &key)| (source_layout.local2global(ind).unwrap(), key))
            .sorted_by_key(|&elem| elem.1)
            .unzip();

        let (sorted_target_indices, sorted_target_keys): (Vec<_>, Vec<_>) = target_keys
            .iter()
            .enumerate()
            .map(|(ind, &key)| (target_layout.local2global(ind).unwrap(), key))
            .sorted_by_key(|&elem| elem.1)
            .unzip();

        let bins = octree.coarse_tree_bounds();

        // We compute the bins from the sorted keys
        let source_bins = sort_to_bins(&sorted_source_keys, bins);
        let target_bins = sort_to_bins(&sorted_target_keys, bins);

        // We communicate keys and their global indices around
        let (_, communicated_source_indices) =
            all_to_allv(comm, &source_bins, &sorted_source_indices);
        let (_, communicated_target_indices) =
            all_to_allv(comm, &target_bins, &sorted_target_indices);

        let (_, communicated_source_keys) = all_to_allv(comm, &source_bins, &sorted_source_keys);
        let (_, communicated_target_keys) = all_to_allv(comm, &target_bins, &sorted_target_keys);

        let (sorted_source_indices, _sorted_source_keys, source_indptr) = sort_by_leafs(
            &communicated_source_indices,
            &communicated_source_keys,
            octree.leaf_keys(),
        );

        let (sorted_target_indices, _sorted_target_keys, target_indptr) = sort_by_leafs(
            &communicated_target_indices,
            &communicated_target_keys,
            octree.leaf_keys(),
        );

        let source_permutation =
            DataPermutation::new(source_layout.clone(), &sorted_source_indices, 1);
        let target_permutation =
            DataPermutation::new(target_layout.clone(), &sorted_target_indices, 1);

        Self {
            source_permutation,
            target_permutation,
            source_indices: sorted_source_indices,
            target_indices: sorted_target_indices,
            source_indptr,
            target_indptr,
            octree,
        }
    }

    /// Return the source permutation
    ///
    /// The permutation has a chunk size of one, i.e. one value per source. For records
    /// with several values build a permutation from [`Self::source_indices`].
    pub fn source_permutation(&self) -> &DataPermutation<'a, C> {
        &self.source_permutation
    }

    /// Return the target permutation
    ///
    /// The permutation has a chunk size of one, i.e. one value per target. For records
    /// with several values build a permutation from [`Self::target_indices`].
    pub fn target_permutation(&self) -> &DataPermutation<'a, C> {
        &self.target_permutation
    }

    /// Return the original global source indices in tree order
    ///
    /// Together with the source index layout this gives a permutation of any chunk size
    /// via `DataPermutation::new(layout, tree.source_indices(), chunk_size)`.
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Return the original global target indices in tree order
    ///
    /// Together with the target index layout this gives a permutation of any chunk size
    /// via `DataPermutation::new(layout, tree.target_indices(), chunk_size)`.
    pub fn target_indices(&self) -> &[usize] {
        &self.target_indices
    }

    /// Return the source map
    pub fn source_indptr(&self) -> &[usize] {
        &self.source_indptr
    }

    /// Return the target map
    pub fn target_indptr(&self) -> &[usize] {
        &self.target_indptr
    }

    /// Return the underlying octree
    pub fn octree(&self) -> &Octree<'a, C> {
        &self.octree
    }
}

/// For a sorted array of keys return the ancestor of `key`
fn get_ancestor_key(arr: &[MortonKey], key: MortonKey) -> MortonKey {
    // Does a binary search of the key. If the key is found with Ok(..)
    // the exact index is returned of the found key. If the key is not found
    // the closest larger index is returned. So we subtract one to get the closest
    // smaller index.

    let ancestor = unsafe {
        *arr.get_unchecked(match arr.binary_search(&key) {
            Ok(index) => index,
            Err(index) => index - 1,
        })
    };

    debug_assert!(is_ancestor(ancestor, key));

    ancestor
}

/// Sort keys and indices according to the leafs of the octree
///
/// The function returns a triple `(sorted_indices, sorted_keys, indptr)`.
/// The keys associated with the ith leaf are contained in
/// `sorted_keys[indptr[i]..indptr[i+1]]`, similary with `sorted_indices`.
fn sort_by_leafs(
    indices: &[usize],
    keys: &[u64],
    leafs: &[u64],
) -> (Vec<usize>, Vec<u64>, Vec<usize>) {
    // Compute for each key the ancestor key from the leafs.

    let ancestor_keys = keys
        .iter()
        .map(|&key| get_ancestor_key(leafs, key))
        .collect_vec();

    // Now resort indices and keys by ancestor_keys

    let (sorted_indices, sorted_keys, sorted_ancestor_keys): (Vec<_>, Vec<_>, Vec<_>) = izip!(
        indices.iter().copied(),
        keys.iter().copied(),
        ancestor_keys.iter().copied()
    )
    .sorted_by_key(|&elem| elem.2)
    .multiunzip();

    // We are now computing the counts of all leaf keys with respect to the sorted ancestor keys

    // We initialize an empty hash map which will map each leaf key to the number of contained
    // keys.
    let mut counts = HashMap::<u64, usize>::from_iter(leafs.iter().map(|&elem| (elem, 0)));

    // We now get the count of each existing ancestor key
    let key_counts = sorted_ancestor_keys.iter().copied().counts();

    // We now update in the map of all leaf keys the counts of those keys
    // that appear in the `ancestor_keys` map.
    for (key, count) in key_counts.iter() {
        *counts.get_mut(key).unwrap() = *count
    }

    // Finally, convert counts into an indptr vector

    let mut indptr = Vec::<usize>::with_capacity(1 + leafs.len());

    let mut acc = 0;
    for key in leafs.iter() {
        indptr.push(acc);
        acc += *counts.get(key).unwrap();
    }
    indptr.push(acc);

    // And now we return everything
    (sorted_indices, sorted_keys, indptr)
}

//! A test FMM that propagates leaf indices.
//!
//! Every leaf of the global tree is assigned an index in `0..N`, where `N` is
//! the global number of leaves. All FMM data is a vector of `N` counts, and
//! every operator merges its input into its output by adding the counts. The
//! sources of a leaf are its index; P2M, P2L and P2P add one at that index,
//! and all other operators add the input vector to the output vector.
//!
//! If the FMM visits every pair of leaves exactly once, then each leaf ends up
//! with a count of one for every index. A missing interaction leaves a zero,
//! an interaction counted twice a two. [`run_index_fmm`] runs the complete
//! distributed FMM and checks this on every rank.

#[cfg(test)]
#[path = "index_fmm_tests.rs"]
mod tests;

use mpi::{collective::SystemOperation, traits::CommunicatorCollectives};
use nd_octree::{MortonKey, Octree};
use rlst::distributed_tools::array_tools::gather_to_all;

use super::evaluator::FmmEvaluator;
use super::operator::FmmOperator;
use crate::interaction_manager::InteractionManager;

/// The operators of the index-propagating test FMM.
#[derive(Clone, Copy, Debug)]
pub struct IndexFmm {
    nleaves: usize,
}

impl IndexFmm {
    /// Create the operators for a tree with `nleaves` leaves in total.
    ///
    /// # Panics
    ///
    /// Panics if `nleaves` is zero.
    pub fn new(nleaves: usize) -> Self {
        assert!(nleaves > 0, "the tree has at least one leaf");
        Self { nleaves }
    }

    /// Return the global number of leaves.
    pub fn nleaves(&self) -> usize {
        self.nleaves
    }
}

fn add(input: &[u32], output: &mut [u32]) {
    for (out, &value) in output.iter_mut().zip(input) {
        *out += value;
    }
}

fn add_index(sources: &[u32], output: &mut [u32]) {
    output[sources[0] as usize] += 1;
}

impl FmmOperator for IndexFmm {
    type Value = u32;

    fn multipole_size(&self, _level: usize) -> usize {
        self.nleaves
    }

    fn local_size(&self, _level: usize) -> usize {
        self.nleaves
    }

    fn source_size(&self) -> usize {
        1
    }

    fn target_size(&self) -> usize {
        self.nleaves
    }

    fn p2m(&self, _leaf: MortonKey, sources: &[u32], multipole: &mut [u32]) {
        add_index(sources, multipole);
    }

    fn m2m(&self, _child: MortonKey, _parent: MortonKey, child: &[u32], parent: &mut [u32]) {
        add(child, parent);
    }

    fn m2l(&self, _source: MortonKey, _target: MortonKey, multipole: &[u32], local: &mut [u32]) {
        add(multipole, local);
    }

    fn p2l(&self, _source: MortonKey, _target: MortonKey, sources: &[u32], local: &mut [u32]) {
        add_index(sources, local);
    }

    fn l2l(&self, _parent: MortonKey, _child: MortonKey, parent: &[u32], child: &mut [u32]) {
        add(parent, child);
    }

    fn l2p(&self, _leaf: MortonKey, local: &[u32], targets: &mut [u32]) {
        add(local, targets);
    }

    fn m2p(&self, _source: MortonKey, _target: MortonKey, multipole: &[u32], targets: &mut [u32]) {
        add(multipole, targets);
    }

    fn p2p(&self, _source: MortonKey, _target: MortonKey, sources: &[u32], targets: &mut [u32]) {
        add_index(sources, targets);
    }
}

/// Return the global number of leaves and the global index of every local
/// leaf, in the order of [`Octree::leaf_keys`].
///
/// The ranks hold consecutive ranges of the Morton order, so the index of a
/// leaf is its position in the global Morton order of all leaves. This is a
/// collective operation.
pub fn global_leaf_indices<C: CommunicatorCollectives>(
    octree: &Octree<'_, C>,
) -> (usize, Vec<u32>) {
    let comm = octree.comm();
    let counts = gather_to_all(&[octree.leaf_keys().len()], comm);
    let offset: usize = counts[..comm.rank() as usize].iter().sum();
    let nleaves = counts.iter().sum();
    let nlocal = octree.leaf_keys().len();
    let indices = (offset..offset + nlocal)
        .map(|index| u32::try_from(index).expect("the number of leaves fits into u32"))
        .collect();
    (nleaves, indices)
}

/// Check that every local leaf holds exactly one count of every index.
///
/// On failure the message names the first failing leaf and the indices with a
/// wrong count.
pub fn check_targets<C: CommunicatorCollectives>(
    evaluator: &FmmEvaluator<'_, '_, C, IndexFmm>,
) -> Result<(), String> {
    for &leaf in evaluator.leaves() {
        let targets = evaluator.targets(leaf).unwrap();
        let wrong: Vec<(usize, u32)> = targets
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, count)| count != 1)
            .collect();
        if !wrong.is_empty() {
            let shown = wrong.iter().take(8).collect::<Vec<_>>();
            return Err(format!(
                "leaf {leaf}: {} of {} indices have a wrong (index, count): {shown:?}",
                wrong.len(),
                targets.len()
            ));
        }
    }
    Ok(())
}

/// Run the index FMM on `octree` and check the result on every rank.
///
/// Returns `Ok(())` on every rank if all ranks pass. Otherwise every rank
/// returns an error, carrying its own failure message if it failed. This is a
/// collective operation.
pub fn run_index_fmm<C: CommunicatorCollectives>(
    octree: &Octree<'_, C>,
    lists: &InteractionManager,
) -> Result<(), String> {
    let (nleaves, indices) = global_leaf_indices(octree);
    let mut evaluator = FmmEvaluator::new(octree, lists, IndexFmm::new(nleaves));
    for (&leaf, &index) in octree.leaf_keys().iter().zip(&indices) {
        evaluator.sources_mut(leaf)[0] = index;
    }
    evaluator.evaluate();

    let local = check_targets(&evaluator);
    let mut all_passed = false;
    octree.comm().all_reduce_into(
        &local.is_ok(),
        &mut all_passed,
        SystemOperation::logical_and(),
    );
    match local {
        Err(message) => Err(message),
        Ok(()) if all_passed => Ok(()),
        Ok(()) => Err("the index FMM failed on another rank".to_string()),
    }
}

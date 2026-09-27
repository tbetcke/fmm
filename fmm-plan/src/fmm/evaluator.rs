//! The distributed driver of an FMM.
//!
//! See the [module documentation](super) for the compute graph that
//! [`FmmEvaluator::evaluate`] runs.

#[cfg(test)]
#[path = "evaluator_tests.rs"]
mod tests;

use std::collections::HashMap;

use itertools::Itertools;
use mpi::traits::CommunicatorCollectives;
use nd_octree::{MortonKey, Octree, morton, octree::KeyType};
use rlst::distributed_tools::array_tools::gather_to_all;

use super::operator::FmmOperator;
use crate::ghost_communicator::{FmmGhostCommunicator, LevelChunkSizes};
use crate::interaction_manager::InteractionManager;

/// Data of a fixed number of values per key, stored contiguously per level.
///
/// The chunk of a key on level `l` has `chunk_size(l)` values. Keys of
/// different levels live in different buffers, so the chunks of a parent and a
/// child can be borrowed at the same time with [`LevelData::coarse_fine_mut`].
pub struct LevelData<T> {
    chunk_sizes: Vec<usize>,
    data: Vec<Vec<T>>,
    positions: HashMap<MortonKey, usize>,
}

impl<T: Copy + Default> LevelData<T> {
    /// Create zero-initialised storage for `keys` on the levels
    /// `0..chunk_sizes.len()`. Duplicate keys are stored once.
    ///
    /// # Panics
    ///
    /// Panics if a key lies on a level without a chunk size.
    pub fn new(keys: impl IntoIterator<Item = MortonKey>, chunk_sizes: Vec<usize>) -> Self {
        let mut counts = vec![0; chunk_sizes.len()];
        let mut positions = HashMap::new();
        for key in keys {
            let level = morton::level(key);
            assert!(level < chunk_sizes.len(), "no chunk size for level {level}");
            positions.entry(key).or_insert_with(|| {
                counts[level] += 1;
                counts[level] - 1
            });
        }
        let data = counts
            .iter()
            .zip(&chunk_sizes)
            .map(|(&count, &size)| vec![T::default(); count * size])
            .collect();
        Self {
            chunk_sizes,
            data,
            positions,
        }
    }

    /// Return true if `key` has a chunk.
    pub fn contains(&self, key: MortonKey) -> bool {
        self.positions.contains_key(&key)
    }

    /// Return the chunk of `key`, or `None` if `key` is not stored.
    pub fn get(&self, key: MortonKey) -> Option<&[T]> {
        let range = self.range(key)?;
        Some(&self.data[morton::level(key)][range])
    }

    /// Return the chunk of `key` mutably, or `None` if `key` is not stored.
    pub fn get_mut(&mut self, key: MortonKey) -> Option<&mut [T]> {
        let range = self.range(key)?;
        Some(&mut self.data[morton::level(key)][range])
    }

    /// Return the chunk of `coarse` and the chunk of `fine` mutably.
    ///
    /// # Panics
    ///
    /// Panics if `coarse` does not lie on a coarser level than `fine`, or if
    /// either key is not stored.
    pub fn coarse_fine_mut(&mut self, coarse: MortonKey, fine: MortonKey) -> (&mut [T], &mut [T]) {
        let (coarse_level, fine_level) = (morton::level(coarse), morton::level(fine));
        assert!(
            coarse_level < fine_level,
            "keys must lie on different levels"
        );
        let coarse_range = self.range(coarse).expect("coarse key is stored");
        let fine_range = self.range(fine).expect("fine key is stored");
        let (low, high) = self.data.split_at_mut(fine_level);
        (
            &mut low[coarse_level][coarse_range],
            &mut high[0][fine_range],
        )
    }

    /// Reset every value to `T::default()`.
    pub fn clear(&mut self) {
        for level in &mut self.data {
            level.fill(T::default());
        }
    }

    fn range(&self, key: MortonKey) -> Option<std::ops::Range<usize>> {
        let position = *self.positions.get(&key)?;
        let size = self.chunk_sizes[morton::level(key)];
        Some(position * size..(position + 1) * size)
    }
}

/// Distributed evaluation of an FMM with the operators `Op`.
///
/// See the [module documentation](super) for the compute graph.
pub struct FmmEvaluator<'o, 'c, C: CommunicatorCollectives, Op: FmmOperator> {
    octree: &'o Octree<'c, C>,
    lists: &'o InteractionManager,
    operator: Op,
    /// The number of levels, that is the global maximum level plus one.
    nlevels: usize,
    /// The local leaves, in ascending order.
    leaves: Vec<MortonKey>,
    /// The non-ghost keys of every level, in ascending order.
    keys_by_level: Vec<Vec<MortonKey>>,
    /// The `(child, parent)` pairs of the local upward pass, by child level.
    local_m2m: Vec<Vec<(MortonKey, MortonKey)>>,
    /// The global keys of every level, in ascending order.
    global_by_level: Vec<Vec<MortonKey>>,
    /// Source data of the local leaves and of the ghost leaves in U- and X-lists.
    sources: LevelData<Op::Value>,
    /// Target data of the local leaves.
    targets: LevelData<Op::Value>,
    /// Multipoles of every key of the tree.
    multipoles: LevelData<Op::Value>,
    /// Local expansions of every non-ghost key.
    locals: LevelData<Op::Value>,
    source_ghosts: FmmGhostCommunicator<Op::Value>,
    multipole_ghosts: FmmGhostCommunicator<Op::Value>,
}

impl<'o, 'c, C: CommunicatorCollectives, Op: FmmOperator> FmmEvaluator<'o, 'c, C, Op> {
    /// Create an evaluator for `octree` with the interaction lists `lists`.
    ///
    /// All data starts zeroed; set the sources with
    /// [`sources_mut`](Self::sources_mut) before calling
    /// [`evaluate`](Self::evaluate).
    ///
    /// This is a collective operation on the communicator of `octree`.
    ///
    /// # Panics
    ///
    /// Panics if an operator size is zero, or if `lists` names a key that is
    /// not a key of `octree`.
    pub fn new(octree: &'o Octree<'c, C>, lists: &'o InteractionManager, operator: Op) -> Self {
        let all_keys = octree.all_keys();
        let nlevels = octree.global_max_level() + 1;

        let leaves = octree.leaf_keys().clone();

        let mut keys_by_level = vec![Vec::new(); nlevels];
        let mut global_by_level = vec![Vec::new(); nlevels];
        let mut local_m2m = vec![Vec::new(); nlevels];
        for (&key, &key_type) in all_keys {
            if key_type.is_ghost() {
                continue;
            }
            let level = morton::level(key);
            keys_by_level[level].push(key);
            if key_type == KeyType::Global {
                global_by_level[level].push(key);
            } else if let Some(parent) = morton::parent(key)
                && all_keys[&parent] == KeyType::LocalInterior
            {
                local_m2m[level].push((key, parent));
            }
        }
        for keys in keys_by_level.iter_mut().chain(&mut global_by_level) {
            keys.sort_unstable();
        }
        for pairs in &mut local_m2m {
            pairs.sort_unstable();
        }

        let multipole_sizes = (0..nlevels)
            .map(|level| operator.multipole_size(level))
            .collect_vec();
        let local_sizes = (0..nlevels)
            .map(|level| operator.local_size(level))
            .collect_vec();

        let source_entries = [lists.u_list(), lists.x_list()]
            .into_iter()
            .flat_map(|map| map.values().flatten().copied());
        let source_ghosts = FmmGhostCommunicator::new(
            octree,
            source_entries,
            LevelChunkSizes::Uniform(operator.source_size()),
        );
        let multipole_entries = [lists.v_list(), lists.w_list()]
            .into_iter()
            .flat_map(|map| map.values().flatten().copied());
        let multipole_ghosts = FmmGhostCommunicator::new(
            octree,
            multipole_entries,
            LevelChunkSizes::PerLevel(multipole_sizes.clone()),
        );

        let received_sources = (0..nlevels)
            .flat_map(|level| source_ghosts.receive_keys(level).iter().copied())
            .collect_vec();
        let sources = LevelData::new(
            leaves.iter().copied().chain(received_sources),
            vec![operator.source_size(); nlevels],
        );
        let targets = LevelData::new(
            leaves.iter().copied(),
            vec![operator.target_size(); nlevels],
        );
        let multipoles = LevelData::new(all_keys.keys().copied(), multipole_sizes);
        let locals = LevelData::new(keys_by_level.iter().flatten().copied(), local_sizes);

        Self {
            octree,
            lists,
            operator,
            nlevels,
            leaves,
            keys_by_level,
            local_m2m,
            global_by_level,
            sources,
            targets,
            multipoles,
            locals,
            source_ghosts,
            multipole_ghosts,
        }
    }

    /// Return the operator.
    pub fn operator(&self) -> &Op {
        &self.operator
    }

    /// Return the octree.
    pub fn octree(&self) -> &'o Octree<'c, C> {
        self.octree
    }

    /// Return the local leaves in ascending order.
    pub fn leaves(&self) -> &[MortonKey] {
        &self.leaves
    }

    /// Return the source data of the local leaf `leaf` mutably.
    ///
    /// # Panics
    ///
    /// Panics if `leaf` is not a local leaf.
    pub fn sources_mut(&mut self, leaf: MortonKey) -> &mut [Op::Value] {
        assert!(self.leaves.binary_search(&leaf).is_ok(), "not a local leaf");
        self.sources.get_mut(leaf).unwrap()
    }

    /// Return the target data of the local leaf `leaf`, or `None` if `leaf` is
    /// not a local leaf.
    pub fn targets(&self, leaf: MortonKey) -> Option<&[Op::Value]> {
        self.targets.get(leaf)
    }

    /// Return the multipole of `key`, or `None` if `key` is not a key of the
    /// tree. A ghost multipole is only set if it is required by this rank.
    pub fn multipole(&self, key: MortonKey) -> Option<&[Op::Value]> {
        self.multipoles.get(key)
    }

    /// Return the local expansion of the non-ghost `key`, or `None` if `key`
    /// is not a non-ghost key of the tree.
    pub fn local(&self, key: MortonKey) -> Option<&[Op::Value]> {
        self.locals.get(key)
    }

    /// Run the complete FMM.
    ///
    /// Multipoles, local expansions and targets are reset first, so the
    /// evaluation may be repeated with new sources. This is a collective
    /// operation on the communicator of the octree.
    pub fn evaluate(&mut self) {
        self.multipoles.clear();
        self.locals.clear();
        self.targets.clear();
        self.exchange_sources();
        self.upward_local();
        self.upward_global();
        self.exchange_multipoles();
        self.downward();
        self.evaluate_leaves();
    }

    /// Fetch the sources of the ghost leaves in U- and X-lists (collective).
    pub fn exchange_sources(&mut self) {
        exchange(&mut self.source_ghosts, &mut self.sources);
    }

    /// P2M on the local leaves and M2M into the local interior keys.
    pub fn upward_local(&mut self) {
        let op = &self.operator;
        for &leaf in &self.leaves {
            op.p2m(
                leaf,
                self.sources.get(leaf).unwrap(),
                self.multipoles.get_mut(leaf).unwrap(),
            );
        }
        for level in (1..self.nlevels).rev() {
            for &(child, parent) in &self.local_m2m[level] {
                let (parent_m, child_m) = self.multipoles.coarse_fine_mut(parent, child);
                op.m2m(child, parent, child_m, parent_m);
            }
        }
    }

    /// Compute the multipoles of the global keys on every rank (collective).
    ///
    /// The multipoles of the coarse-tree leaves of all ranks are gathered on
    /// every rank and translated upwards through the global keys.
    pub fn upward_global(&mut self) {
        let comm = self.octree.comm();
        let own_coarse = self.octree.coarse_tree_leafs();
        let own_values = own_coarse
            .iter()
            .flat_map(|&key| self.multipoles.get(key).unwrap().iter().copied())
            .collect_vec();
        let coarse = gather_to_all(own_coarse, comm);
        let values = gather_to_all(&own_values, comm);

        // Group the gathered coarse leaves by level, each with its multipole.
        let mut coarse_by_level = vec![Vec::new(); self.nlevels];
        let mut offset = 0;
        for &key in &coarse {
            let level = morton::level(key);
            let size = self.operator.multipole_size(level);
            coarse_by_level[level].push((key, offset..offset + size));
            offset += size;
        }
        assert_eq!(offset, values.len());

        let op = &self.operator;
        for level in (1..self.nlevels).rev() {
            for (key, range) in &coarse_by_level[level] {
                let parent = morton::parent(*key).unwrap();
                op.m2m(
                    *key,
                    parent,
                    &values[range.clone()],
                    self.multipoles
                        .get_mut(parent)
                        .expect("the parent of a coarse-tree leaf is a global key"),
                );
            }
            for &key in &self.global_by_level[level] {
                let parent = morton::parent(key).unwrap();
                let (parent_m, child_m) = self.multipoles.coarse_fine_mut(parent, key);
                op.m2m(key, parent, child_m, parent_m);
            }
        }
    }

    /// Fetch the multipoles of the ghost boxes in V- and W-lists (collective).
    pub fn exchange_multipoles(&mut self) {
        exchange(&mut self.multipole_ghosts, &mut self.multipoles);
    }

    /// L2L, M2L and P2L level by level from the root.
    pub fn downward(&mut self) {
        let op = &self.operator;
        for level in 1..self.nlevels {
            for &key in &self.keys_by_level[level] {
                let parent = morton::parent(key).unwrap();
                let (parent_l, local) = self.locals.coarse_fine_mut(parent, key);
                op.l2l(parent, key, parent_l, local);
                for &source in &self.lists.v_list()[&key] {
                    op.m2l(source, key, self.multipoles.get(source).unwrap(), local);
                }
                for &source in &self.lists.x_list()[&key] {
                    op.p2l(source, key, self.sources.get(source).unwrap(), local);
                }
            }
        }
    }

    /// L2P, M2P and P2P on every local leaf.
    pub fn evaluate_leaves(&mut self) {
        let op = &self.operator;
        for &leaf in &self.leaves {
            let targets = self.targets.get_mut(leaf).unwrap();
            op.l2p(leaf, self.locals.get(leaf).unwrap(), targets);
            for &source in &self.lists.w_list()[&leaf] {
                op.m2p(source, leaf, self.multipoles.get(source).unwrap(), targets);
            }
            for &source in self.lists.u_list()[&leaf].iter().chain([&leaf]) {
                op.p2p(source, leaf, self.sources.get(source).unwrap(), targets);
            }
        }
    }
}

/// Send the owned chunks of `data` to the ranks that hold them as ghosts and
/// store the received ghost chunks in `data`.
fn exchange<T: mpi::traits::Equivalence + Copy + Default>(
    ghosts: &mut FmmGhostCommunicator<T>,
    data: &mut LevelData<T>,
) {
    for level in 0..ghosts.nlevels() {
        for (key, chunk) in ghosts.send_chunks_mut(level) {
            chunk.copy_from_slice(data.get(key).expect("a sent key is stored"));
        }
    }
    ghosts.forward_all();
    for level in 0..ghosts.nlevels() {
        for (key, chunk) in ghosts.receive_chunks(level) {
            data.get_mut(key)
                .expect("a received key is stored")
                .copy_from_slice(chunk);
        }
    }
}

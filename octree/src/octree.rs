//! Definition of Octree.
mod implementation;
use std::{collections::HashMap, error::Error, fmt};

pub(crate) use implementation::*;
use itertools::Itertools;
use mpi::{
    collective::SystemOperation,
    traits::{CommunicatorCollectives, Root},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlst::{Array, ValueArrayImpl, distributed_tools::array_tools::gather_to_rank};

use crate::{constants::DEEPEST_LEVEL, geometry::PhysicalBox, morton, morton::MortonKey};

/// Stores the type of the key relative to the octree.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum KeyType {
    /// A local leaf.
    LocalLeaf,
    /// A local interior key.
    LocalInterior,
    /// A global key.
    Global,
    /// A ghost leaf key from a specific process.
    GhostLeaf(usize),
    /// A ghost interior key from a specific process.
    GhostInterior(usize),
}

/// The leaf containing a finest-level Morton key and its owning MPI rank.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct LeafLocation {
    /// The containing leaf key.
    pub leaf: MortonKey,
    /// The zero-based MPI rank that owns `leaf`.
    pub owner_rank: usize,
}

/// An error returned for an invalid leaf-lookup query.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum LookupError {
    /// The key has the Morton invalid marker set.
    InvalidKey,
    /// The key is valid but is not at the finest supported level.
    NotFinestLevel {
        /// The key level.
        level: usize,
    },
    /// No containing leaf was found, indicating a tree invariant failure.
    LeafNotFound,
}

impl fmt::Display for LookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKey => write!(f, "Morton key is invalid"),
            Self::NotFinestLevel { level } => {
                write!(f, "Morton key level {level} is not the finest level")
            }
            Self::LeafNotFound => write!(f, "no containing leaf was found"),
        }
    }
}

impl Error for LookupError {}

/// An error that prevents a collective batch lookup from exchanging its requests.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum LookupBatchError {
    /// An MPI count or displacement cannot be represented by a signed 32-bit integer.
    CountOverflow,
}

impl fmt::Display for LookupBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountOverflow => write!(f, "MPI lookup count or displacement overflow"),
        }
    }
}

impl Error for LookupBatchError {}

impl KeyType {
    /// Return true if the key is a ghost.
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::KeyType;
    /// assert!(KeyType::GhostLeaf(2).is_ghost());
    /// ```
    pub fn is_ghost(&self) -> bool {
        matches!(self, KeyType::GhostLeaf(_)) || matches!(self, KeyType::GhostInterior(_))
    }

    /// Return `Some(rank)` for ghost keys, otherwise return `None`.
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::KeyType;
    /// assert_eq!(KeyType::GhostInterior(2).ghost_rank(), Some(2));
    /// ```
    pub fn ghost_rank(&self) -> Option<usize> {
        match self {
            Self::GhostLeaf(rank) => Some(*rank),
            Self::GhostInterior(rank) => Some(*rank),
            _ => None,
        }
    }
}

/// What the partition of the coarse blocks over the ranks weighs (Phase 5;
/// [`OctreeOptions::with_partition_weight`]).
///
/// The weight decides which coarse blocks are refined and where the ranks' ranges
/// are cut (see [`Octree::new`]). It never changes the leaves, which depend only on
/// the distinct keys, `max_level` and `max_fine_keys`. On one rank it has no effect.
///
/// # Examples
///
/// ```
/// use nd_octree::octree::{OctreeOptions, PartitionWeight};
/// assert_eq!(PartitionWeight::default(), PartitionWeight::Keys);
/// let options = OctreeOptions::new().with_partition_weight(PartitionWeight::DistinctKeys);
/// assert_eq!(options.partition_weight(), PartitionWeight::DistinctKeys);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PartitionWeight {
    /// Every key passed to [`Octree::new`], duplicates included: for an FMM that
    /// passes one key per source and one per target, the points. The default.
    #[default]
    Keys,
    /// Every distinct finest-level key once (the weight up to Phase 4).
    DistinctKeys,
}

/// Options that select the shape of an [`Octree`] and the optional topology it
/// stores.
///
/// The value carries every construction setting other than the input keys and
/// the communicator, so it is the only settings argument of [`Octree::new`].
/// The default refines to [`DEEPEST_LEVEL`] with a target of one finest-level
/// key per leaf and requests no optional layer, which is the smallest tree
/// satisfying the documented invariants. Every enabled layer widens the stored
/// key map and the payload exchanged during construction. On several ranks the
/// default partition weighs every key ([`PartitionWeight::Keys`]) and refines the
/// coarse tree until no splittable block weighs more than an eighth of a rank's
/// fair share ([`OctreeOptions::with_block_refinement`]).
///
/// # Examples
///
/// ```
/// use nd_octree::octree::{OctreeOptions, PartitionWeight};
/// let options = OctreeOptions::new()
///     .with_max_level(12)
///     .with_max_fine_keys(4)
///     .with_ghost_children(true)
///     .with_partition_weight(PartitionWeight::Keys)
///     .with_block_refinement(8);
/// assert_eq!(options.max_level(), 12);
/// assert_eq!(options.max_fine_keys(), 4);
/// assert!(options.ghost_children());
/// assert_eq!(options.partition_weight(), PartitionWeight::Keys);
/// assert_eq!(options.block_refinement(), 8);
/// ```
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct OctreeOptions {
    max_level: usize,
    max_fine_keys: usize,
    ghost_children: bool,
    partition_weight: PartitionWeight,
    block_refinement: usize,
}

impl Default for OctreeOptions {
    /// Return the deepest tree with no optional layer.
    /// # Returns
    /// Options with `max_level` at [`DEEPEST_LEVEL`], `max_fine_keys` at one, the
    /// ghost-children layer disabled, the partition weighing every key
    /// ([`PartitionWeight::Keys`]) and a block refinement factor of 8.
    fn default() -> Self {
        Self {
            max_level: DEEPEST_LEVEL as usize,
            max_fine_keys: 1,
            ghost_children: false,
            partition_weight: PartitionWeight::Keys,
            block_refinement: 8,
        }
    }
}

impl OctreeOptions {
    /// Return the default options.
    /// # Returns
    /// The same value as [`OctreeOptions::default`].
    /// # Examples
    ///
    /// ```
    /// use nd_octree::{constants::DEEPEST_LEVEL, octree::OctreeOptions};
    /// let options = OctreeOptions::new();
    /// assert_eq!(options.max_level(), DEEPEST_LEVEL as usize);
    /// assert_eq!(options.max_fine_keys(), 1);
    /// assert!(!options.ghost_children());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum leaf level.
    ///
    /// No leaf of the constructed tree is deeper than `max_level`, on any number
    /// of ranks. Values larger than [`DEEPEST_LEVEL`] are clamped during
    /// construction.
    /// # Arguments
    /// - `max_level`: Maximum leaf level, in `0..=16`.
    ///
    /// # Returns
    /// The options with the maximum leaf level set to `max_level`.
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::OctreeOptions;
    /// assert_eq!(OctreeOptions::new().with_max_level(12).max_level(), 12);
    /// ```
    pub fn with_max_level(self, max_level: usize) -> Self {
        Self { max_level, ..self }
    }

    /// Return the maximum leaf level.
    /// # Returns
    /// The requested maximum leaf level, before clamping to [`DEEPEST_LEVEL`].
    /// # Examples
    ///
    /// ```
    /// use nd_octree::{constants::DEEPEST_LEVEL, octree::OctreeOptions};
    /// assert_eq!(OctreeOptions::default().max_level(), DEEPEST_LEVEL as usize);
    /// ```
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Set the target maximum number of finest-level keys in a leaf.
    ///
    /// The target bounds refinement: a box holding more than `max_fine_keys`
    /// finest-level keys is refined further unless it already sits at
    /// [`OctreeOptions::max_level`], so a leaf at the level limit may hold more.
    /// # Arguments
    /// - `max_fine_keys`: Target maximum number of finest-level keys in a leaf.
    ///
    /// # Returns
    /// The options with the target key count set to `max_fine_keys`.
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::OctreeOptions;
    /// assert_eq!(OctreeOptions::new().with_max_fine_keys(4).max_fine_keys(), 4);
    /// ```
    pub fn with_max_fine_keys(self, max_fine_keys: usize) -> Self {
        Self {
            max_fine_keys,
            ..self
        }
    }

    /// Return the target maximum number of finest-level keys in a leaf.
    /// # Returns
    /// The requested target key count per leaf.
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::OctreeOptions;
    /// assert_eq!(OctreeOptions::default().max_fine_keys(), 1);
    /// ```
    pub fn max_fine_keys(&self) -> usize {
        self.max_fine_keys
    }

    /// Request, or drop, the ghost-children layer.
    ///
    /// With the layer enabled, construction also replicates the children of the
    /// interior boxes bordering this rank: for every non-ghost key of
    /// [`Octree::all_keys`] and every same-level neighbour cell of that key which
    /// the tree holds as an interior box, all eight children of the neighbour are
    /// keys of [`Octree::all_keys`] too, each with the classification its owner
    /// assigns. This is one step of closure relative to the keys stored without
    /// the option, not a fixed point: the children of a ghost child that the
    /// layer itself added are not added in turn.
    ///
    /// The layer is what a fast multipole method needs to resolve its V- and
    /// W-list entries locally, since those are children of neighbouring boxes.
    ///
    /// On more than one rank the layer also replicates every coarse block, that
    /// is every key of [`Octree::coarse_tree_leafs`] on every rank, without its
    /// children. A block of another rank is a key of [`Octree::all_keys`],
    /// classified as [`KeyType::GhostLeaf`] or [`KeyType::GhostInterior`] with its
    /// owner. The children of a [`KeyType::Global`] key are `Global` keys or
    /// coarse blocks, so every child of every `Global` key is a key of the map as
    /// well. This lets a distributed fast multipole method form the multipoles of
    /// the `Global` keys on every rank from the multipoles of all blocks.
    /// # Arguments
    /// - `enabled`: Whether the ghost-children layer is stored.
    ///
    /// # Returns
    /// The options with the ghost-children layer set to `enabled`.
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::OctreeOptions;
    /// let options = OctreeOptions::new().with_ghost_children(true);
    /// assert!(options.ghost_children());
    /// assert!(!options.with_ghost_children(false).ghost_children());
    /// ```
    pub fn with_ghost_children(self, enabled: bool) -> Self {
        Self {
            ghost_children: enabled,
            ..self
        }
    }

    /// Return true if the ghost-children layer is requested.
    /// # Returns
    /// Whether the ghost-children layer is enabled.
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::OctreeOptions;
    /// assert!(!OctreeOptions::default().ghost_children());
    /// ```
    pub fn ghost_children(&self) -> bool {
        self.ghost_children
    }

    /// Set what the partition of the coarse blocks weighs; see [`PartitionWeight`].
    ///
    /// The weight selects the blocks that [`OctreeOptions::with_block_refinement`]
    /// refines and the cut of the blocks into the ranks' ranges. It does not change
    /// the leaves, and on one rank it has no effect.
    /// # Arguments
    /// - `weight`: What a coarse block weighs.
    ///
    /// # Returns
    /// The options with the partition weight set to `weight`.
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::{OctreeOptions, PartitionWeight};
    /// let options = OctreeOptions::new().with_partition_weight(PartitionWeight::DistinctKeys);
    /// assert_eq!(options.partition_weight(), PartitionWeight::DistinctKeys);
    /// ```
    pub fn with_partition_weight(self, weight: PartitionWeight) -> Self {
        Self {
            partition_weight: weight,
            ..self
        }
    }

    /// Return what the partition of the coarse blocks weighs.
    /// # Returns
    /// The requested [`PartitionWeight`].
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::{OctreeOptions, PartitionWeight};
    /// assert_eq!(OctreeOptions::default().partition_weight(), PartitionWeight::Keys);
    /// ```
    pub fn partition_weight(&self) -> PartitionWeight {
        self.partition_weight
    }

    /// Refine the coarse tree until no block weighs more than 1/`factor` of a rank's
    /// fair share, wherever the one-rank tree refines too (more than `max_fine_keys`
    /// distinct keys) and above `max_level`. Default 8; 1 gives the coarsest blocks.
    ///
    /// On `P` ranks with total weight `W`, a block is split into its eight children
    /// while it weighs more than `W / (factor P)`, holds more than
    /// [`OctreeOptions::max_fine_keys`] distinct keys and lies above
    /// [`OctreeOptions::max_level`]. A block that cannot be split stays whole however
    /// heavy it is, so the busiest rank's weight is at most the mean times
    /// `1 + w_max / (W / P)`, `w_max` the heaviest block. Finer blocks mean more
    /// coarse blocks and `Global` keys, which every rank holds (the replicated
    /// coarse tree, and with [`OctreeOptions::with_ghost_children`] every block).
    /// The factor never changes the leaves, and on one rank it has no effect.
    /// A factor of 0 acts as 1.
    /// # Arguments
    /// - `factor`: How many blocks a rank's fair share should at least be split into.
    ///
    /// # Returns
    /// The options with the block refinement factor set to `factor`.
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::OctreeOptions;
    /// assert_eq!(OctreeOptions::new().with_block_refinement(16).block_refinement(), 16);
    /// ```
    pub fn with_block_refinement(self, factor: usize) -> Self {
        Self {
            block_refinement: factor,
            ..self
        }
    }

    /// Return the block refinement factor.
    /// # Returns
    /// The requested factor, see [`OctreeOptions::with_block_refinement`].
    /// # Examples
    ///
    /// ```
    /// use nd_octree::octree::OctreeOptions;
    /// assert_eq!(OctreeOptions::default().block_refinement(), 8);
    /// ```
    pub fn block_refinement(&self) -> usize {
        self.block_refinement
    }
}

/// A general structure for octrees.
pub struct Octree<'o, C> {
    coarse_tree_leafs: Vec<MortonKey>,
    leaf_keys: Vec<MortonKey>,
    coarse_tree_bounds: Vec<MortonKey>,
    all_keys: HashMap<MortonKey, KeyType>,
    neighbours: HashMap<MortonKey, Vec<MortonKey>>,
    options: OctreeOptions,
    comm: &'o C,
}

impl<'o, C: CommunicatorCollectives> Octree<'o, C> {
    /// Create a new distributed Octree.
    ///
    /// # Arguments
    /// - `fine_keys`: Local Morton keys at [`DEEPEST_LEVEL`] (level 16), usually
    ///   produced from local points with [`points_to_morton`]. Duplicates are
    ///   allowed, and a rank may pass an empty slice.
    /// - `options`: Every other construction setting, see [`OctreeOptions`]: the
    ///   maximum leaf level ([`OctreeOptions::with_max_level`]), the target number
    ///   of finest-level keys per leaf ([`OctreeOptions::with_max_fine_keys`]),
    ///   the optional topology layers, and on several ranks the partition weight
    ///   ([`OctreeOptions::with_partition_weight`]) and block refinement
    ///   ([`OctreeOptions::with_block_refinement`]). Pass [`OctreeOptions::default`]
    ///   for a tree refined to [`DEEPEST_LEVEL`] with one key per leaf and no
    ///   optional layer.
    /// - `comm`: The communicator that participates in construction and must
    ///   outlive the returned tree.
    ///
    /// # Collective operation
    /// Every rank in `comm` must call this method, also a rank without keys.
    /// Construction linearizes the keys, builds a coarse tree no deeper than
    /// `max_level` that is replicated on every rank, assigns each rank a
    /// contiguous range of its blocks, refines, and 2:1-balances the result. Keys
    /// are redistributed internally while building the distributed topology. The
    /// resulting tree stores Morton-key topology and ownership metadata, not input
    /// points or application data.
    ///
    /// On several ranks the coarse tree is built by weight from the root: every
    /// block that weighs more than `W / (k P)` (total weight `W`, `P` ranks,
    /// `k` = [`OctreeOptions::block_refinement`]), holds more than `max_fine_keys`
    /// distinct keys and lies above `max_level` is split into its children, round
    /// by round, and the blocks are then 2:1 balanced. A block is therefore a node of
    /// the one-rank tree, and the leaves are the one-rank leaves on every rank count
    /// and for every distribution of the keys over the ranks. The weight is set by
    /// [`OctreeOptions::with_partition_weight`]: every key, duplicates included, by
    /// default. Boundary `p` of the ranges is the block boundary, at or after
    /// boundary `p - 1`, whose prefix weight is closest to `W p / P` (ties go to the
    /// later boundary), so the busiest rank holds at most the mean times
    /// `1 + w_max / (W / P)`, `w_max` the heaviest block. A range may be empty: a
    /// rank without blocks has no leaves, no local keys and no ghosts of its own,
    /// and it still enters every collective. With fewer blocks than ranks, for
    /// example a tree whose only block is the root, some ranks are therefore empty.
    /// The partition depends on the rank count and the weights, never on which
    /// rank passed which key.
    ///
    /// The collectives on several ranks, in order: those of the parallel sort and
    /// the successor exchange of the linearization; one all-reduce per refinement
    /// round of the coarse tree, and one more if 2:1 balancing the blocks changes
    /// them; the key move to the owners (an all-to-all-v); those of the distributed
    /// 2:1 balance (an all-reduce of the deepest level and a linearization); the key
    /// move again; and the ghost exchange (two all-gathers and an all-to-all-v).
    /// Debug builds add the collectives of their invariant checks. The partition
    /// bounds come from the replicated coarse tree and need no communication.
    ///
    /// Enabling [`OctreeOptions::with_ghost_children`] does not add a communication
    /// round: the additional keys travel inside the two collectives that the ghost
    /// exchange already performs.
    ///
    /// On one rank the coarse tree is the root, and the options that shape the
    /// partition have no effect.
    ///
    /// # Panics
    /// Panics when a key is invalid or not at [`DEEPEST_LEVEL`]. The message names
    /// the key and its position.
    ///
    /// # Returns
    /// A new distributed `Octree`. When no rank passes a key, its only leaf is the
    /// root, on rank 0.
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// assert!(!tree.leaf_keys().is_empty());
    /// ```
    pub fn new(fine_keys: &[MortonKey], options: OctreeOptions, comm: &'o C) -> Self {
        // Reject malformed input before any communication; every rank checks its
        // own keys, so a bad key aborts the run with a message naming it.
        validate_fine_keys(fine_keys).unwrap_or_else(|message| panic!("{message}"));

        let max_level = options.max_level();
        let max_fine_keys = options.max_fine_keys();
        let rank = comm.rank() as usize;
        let size = comm.size() as usize;

        // We need a random number generator for sorting. For simplicity we use a ChaCha8 random number generator
        // seeded with the rank of the process.
        let mut rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);

        // Linearize the keys.
        let linear_keys = linearize(fine_keys, &mut rng, comm);

        // Build the complete, 2:1 balanced coarse tree by weight, no deeper than
        // `max_level`, with the weight of each block. Both are replicated on every
        // rank, so the partition and its bounds are computed locally and agree.
        let (global_coarse_tree, weights) =
            compute_coarse_tree(fine_keys, &linear_keys, options, comm);
        let partition = partition_blocks(&weights, size);
        let coarse_tree = global_coarse_tree[partition[rank]..partition[rank + 1]].to_vec();
        let coarse_tree_bounds = tree_bins(&global_coarse_tree, &partition);
        debug_assert!(is_complete_linear_tree(&coarse_tree, comm));

        // Redistribute the fine keys with respect to the partitioned coarse tree.
        let local_fine_keys =
            redistribute_with_respect_to_coarse_tree(&linear_keys, &coarse_tree_bounds, comm);

        // Refine the blocks until we are at max level or the fine keys per box are
        // few enough, then 2:1 balance the refined tree and redistribute it again
        // with respect to the coarse tree.
        let refined_tree =
            create_local_tree(&local_fine_keys, &coarse_tree, max_level, max_fine_keys);
        let leaf_tree = redistribute_with_respect_to_coarse_tree(
            &balance(&refined_tree, &mut rng, comm),
            &coarse_tree_bounds,
            comm,
        );

        let all_keys =
            generate_all_keys(&leaf_tree, &coarse_tree, &coarse_tree_bounds, options, comm);
        let neighbours = compute_neighbours(&all_keys);

        Self {
            coarse_tree_leafs: coarse_tree,
            leaf_keys: leaf_tree,
            coarse_tree_bounds,
            all_keys,
            neighbours,
            options,
            comm,
        }
    }

    /// Return the options the tree was constructed with.
    /// # Returns
    /// The [`OctreeOptions`] passed to [`Octree::new`].
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let options = OctreeOptions::new().with_ghost_children(true);
    /// let tree = Octree::new(&fine_keys, options, &comm);
    /// assert!(tree.options().ghost_children());
    /// ```
    pub fn options(&self) -> OctreeOptions {
        self.options
    }

    /// Return the coarse tree leafs.
    ///
    /// These are this rank's coarse blocks, a contiguous range of the replicated
    /// coarse tree; every local leaf is one of them or descends from one. On
    /// several ranks the range may be empty, and the rank then owns no leaves. On
    /// one rank the only block is the root.
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// let _ = tree.coarse_tree_leafs();
    /// ```
    pub fn coarse_tree_leafs(&self) -> &Vec<MortonKey> {
        &self.coarse_tree_leafs
    }

    /// Return the leaf nodes.
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// let _ = tree.leaf_keys();
    /// ```
    pub fn leaf_keys(&self) -> &Vec<MortonKey> {
        &self.leaf_keys
    }

    /// Get the coarse tree bounds.
    ///
    /// This returns an array of size the number of ranks, where each element is
    /// the first coarse-tree block of the corresponding rank. Every key owned by
    /// a rank is one of its blocks or descends from one, so the bounds partition
    /// the Morton range. The bounds are non-decreasing. A rank without blocks has
    /// the bound of the next rank that has blocks, or, after the last such rank,
    /// [`morton::invalid_key`], which sorts above every valid key.
    ///
    /// A key is owned by the last rank whose bound is at most the key, which is
    /// never a rank without blocks. If rank i owns a key and is not the last rank
    /// then
    /// ```text
    /// coarse_tree_bounds[i] <= key < coarse_tree_bounds[i+1]
    /// ```
    /// where as if i is the last rank then
    /// ```text
    /// coarse_tree_bounds[i] <= key
    /// ```
    /// This allows to find the rank of a given Morton key.
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// let _ = tree.coarse_tree_bounds();
    /// ```
    pub fn coarse_tree_bounds(&self) -> &Vec<MortonKey> {
        &self.coarse_tree_bounds
    }

    /// Return the communicator.
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// assert_eq!(tree.comm().size(), comm.size());
    /// ```
    pub fn comm(&self) -> &C {
        self.comm
    }

    /// Return a map of all leaf and interior keys.
    ///
    /// The map assigns each key a [KeyType] identifier. It is one of:
    /// - [KeyType::LocalLeaf] for leaf keys
    /// - [KeyType::LocalInterior] for interior keys
    /// - [KeyType::Global] for global keys
    /// - [KeyType::GhostLeaf] and [KeyType::GhostInterior], keys adjacent to keys
    ///   on the current rank but living on a different rank.
    ///
    /// Leaf keys have no children. A local interior key has all of its children on
    /// the local rank; a ghost interior key generally does not. The ghost-children
    /// layer ([`OctreeOptions::with_ghost_children`]) adds one step of closure, not
    /// a fixed point: with it enabled, for every non-ghost key of this map and every
    /// same-level neighbour cell of that key which the tree holds as an interior
    /// box, all eight children of that neighbour are keys of this map too. A ghost
    /// interior that the layer itself added does **not** have its own children here.
    /// The layer also makes every coarse block of every rank, and hence every child
    /// of every global key, a key of this map; see
    /// [`OctreeOptions::with_ghost_children`].
    /// Global keys are keys that are not uniquely assigned to a rank but exist on all ranks.
    /// The global keys are those that are close to the root of the tree. By construction these
    /// are the ancestors of the coarse tree leafs, where as the coarse tree leafs themselves are
    /// the first level of keys distributed across ranks. When the root is the only coarse
    /// block on several ranks, there is no global key: the root is local to the rank that
    /// owns it and a ghost of that rank on every other rank. A rank without coarse blocks
    /// holds only global keys and ghosts. Ghost keys are keys that are not local to
    /// the current rank but lie along the interface to the current rank. Their identifiers store the value
    /// of the rank that they originate from.
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// let _ = tree.all_keys();
    /// ```
    pub fn all_keys(&self) -> &HashMap<MortonKey, KeyType> {
        &self.all_keys
    }

    /// Get the neighbour map.
    ///
    /// Returns a hash map that contains as keys all the keys obtained from [Octree::all_keys] except
    /// those of type [KeyType::GhostLeaf] or [KeyType::GhostInterior].
    /// The values are the distinct neighbours of the key. For an interior or global
    /// key these are its valid same-level neighbours. For a leaf, each of the up to
    /// 26 neighbouring cells contributes either the same-level key, when the tree
    /// holds it, or that cell's parent, one level coarser; a coarser neighbour that
    /// covers several cells is listed once.
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// let _ = tree.neighbour_map();
    /// ```
    pub fn neighbour_map(&self) -> &HashMap<MortonKey, Vec<MortonKey>> {
        &self.neighbours
    }

    /// Return the rank owning a valid finest-level Morton key without communication.
    ///
    /// Partition intervals are lower-inclusive and upper-exclusive, except that the
    /// final rank with blocks includes all remaining finest-level keys; a rank
    /// without blocks owns no key ([`Octree::coarse_tree_bounds`]). Invalid keys are
    /// reported before level validation. This lookup costs `O(log P)` for `P` ranks.
    /// # Parameters
    ///
    /// - `key`: Morton key to inspect or transform.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// assert!(tree.owner_rank(morton::deepest_first()).is_ok());
    /// ```
    pub fn owner_rank(&self, key: MortonKey) -> Result<usize, LookupError> {
        validate_lookup_key(key)?;
        owner_rank_for_key(&self.coarse_tree_bounds, key).ok_or(LookupError::LeafNotFound)
    }

    /// Return the local leaf containing a valid finest-level Morton key.
    ///
    /// A valid key owned by another rank returns `Ok(None)`, even when a matching
    /// ghost is stored locally. Invalid keys are reported before level validation.
    /// A missing local containing leaf is a defensive tree-invariant error. This
    /// lookup costs `O(log P + log L)` for `L` local leaves.
    /// # Parameters
    ///
    /// - `key`: Morton key to inspect or transform.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// assert!(tree.local_leaf(morton::deepest_first()).unwrap().is_some());
    /// ```
    pub fn local_leaf(&self, key: MortonKey) -> Result<Option<MortonKey>, LookupError> {
        let owner = self.owner_rank(key)?;
        if owner != self.comm.rank() as usize {
            return Ok(None);
        }
        local_leaf_for_key(&self.leaf_keys, key)
            .ok_or(LookupError::LeafNotFound)
            .map(Some)
    }

    /// Collectively find leaves for finest-level Morton keys on their owning ranks.
    ///
    /// Every rank in the communicator must call this method in the same collective
    /// order, including ranks with empty or entirely invalid input. Results preserve
    /// input order and return invalid inputs per query, so valid queries are retained.
    /// The outer error is returned consistently on all ranks only when MPI count or
    /// displacement limits prevent the exchange. Physical-point lookup is deferred.
    /// The communication-free routing and local searches cost `O(log P + log L)` per
    /// valid key; communication and temporary storage are linear in exchanged queries.
    /// # Parameters
    ///
    /// - `keys`: Morton keys that define the input tree or query batch.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// assert!(tree.lookup_leaves(&[morton::deepest_first()]).unwrap()[0].is_ok());
    /// ```
    pub fn lookup_leaves(
        &self,
        keys: &[MortonKey],
    ) -> Result<Vec<Result<LeafLocation, LookupError>>, LookupBatchError> {
        lookup_leaves_collective(keys, &self.coarse_tree_bounds, &self.leaf_keys, self.comm)
    }

    /// Return the local maximum level
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// assert!(tree.local_max_level() <= 16);
    /// ```
    pub fn local_max_level(&self) -> usize {
        self.leaf_keys
            .iter()
            .map(|key| morton::level(*key))
            .max()
            .unwrap_or(0)
    }

    /// Return the global maximum level
    /// # Examples
    ///
    /// ```no_run
    /// use mpi::traits::Communicator;
    /// let universe = mpi::initialize().expect("MPI must be initialized once");
    /// // Use a self communicator so this complete input is not replicated across ranks.
    /// use mpi::topology::SimpleCommunicator;
    /// use nd_octree::{morton, Octree, octree::OctreeOptions};
    /// let comm = SimpleCommunicator::self_comm();
    /// let fine_keys = [
    ///     morton::deepest_first(),
    ///     morton::from_index_and_level([1, 0, 0], 16),
    /// ];
    /// let tree = Octree::new(&fine_keys, OctreeOptions::default(), &comm);
    /// assert!(tree.global_max_level() <= 16);
    /// ```
    pub fn global_max_level(&self) -> usize {
        let mut global_max_level = 0;
        self.comm.all_reduce_into(
            &self.local_max_level(),
            &mut global_max_level,
            SystemOperation::max(),
        );
        global_max_level
    }
}

/// Test if an array of keys are the leafs of a complete linear and balanced tree.
/// # Parameters
///
/// - `arr`: This rank's sorted leaf-key partition; concatenating all ranks' partitions must form one tree.
/// - `comm`: Communicator whose ranks enter this collective phase.
///
/// # Examples
///
/// ```no_run
/// use mpi::traits::Communicator;
/// let universe = mpi::initialize().expect("MPI must be initialized once");
/// // This complete leaf partition belongs to one rank; use `self_comm` even under mpirun.
/// use mpi::topology::SimpleCommunicator;
/// use nd_octree::{morton, octree::is_complete_linear_and_balanced};
/// let comm = SimpleCommunicator::self_comm();
/// let leaves = morton::children(morton::root()).unwrap();
/// assert!(is_complete_linear_and_balanced(&leaves, &comm));
/// ```
pub fn is_complete_linear_and_balanced<C: CommunicatorCollectives>(
    arr: &[MortonKey],
    comm: &C,
) -> bool {
    // Send the tree to the root node and check there that it is balanced.

    let mut balanced = false;

    if let Some(arr) = gather_to_rank(arr, 0, comm) {
        balanced = morton::is_complete_linear_and_balanced(&arr);
    }

    comm.process_at_rank(0).broadcast_into(&mut balanced);

    balanced
}

/// Compute the padded cubic bounding box across all ranks' points.
///
/// # Parameters
///
/// - `points`: Local `[3, n]` point array, with one point per column.
/// - `comm`: Communicator whose ranks all contribute their local extrema.
///
/// # Collective operation
/// Every rank in `comm` must call this function in the same order. Inputs must
/// contain at least one finite point globally; empty or degenerate inputs do
/// not produce a meaningful mapping domain.
///
/// # Examples
///
/// ```no_run
/// use mpi::traits::Communicator;
/// use nd_octree::octree::compute_global_bounding_box;
/// use rlst::SliceArray;
///
/// let universe = mpi::initialize().expect("MPI must be initialized once");
/// let world = universe.world();
/// // Two distinct finite points, stored as columns of a `[3, 2]` array.
/// let data = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
/// let points = SliceArray::from_shape(&data, [3, 2]);
/// let domain = compute_global_bounding_box(&points, &world);
/// assert!(domain.coordinates()[3] > domain.coordinates()[0]);
/// ```
pub fn compute_global_bounding_box<
    ArrayImpl: ValueArrayImpl<f64, 2>,
    C: CommunicatorCollectives,
>(
    points: &Array<ArrayImpl, 2>,
    comm: &C,
) -> PhysicalBox {
    // Make sure that the points array is a multiple of 3.

    assert_eq!(points.shape()[0], 3);

    // Now compute the minimum and maximum across each dimension.

    let mut xmin = f64::MAX;
    let mut xmax = f64::MIN;

    let mut ymin = f64::MAX;
    let mut ymax = f64::MIN;

    let mut zmin = f64::MAX;
    let mut zmax = f64::MIN;

    for point in points.col_iter() {
        let x = point.get_value([0]).unwrap();
        let y = point.get_value([1]).unwrap();
        let z = point.get_value([2]).unwrap();

        xmin = f64::min(xmin, x);
        xmax = f64::max(xmax, x);

        ymin = f64::min(ymin, y);
        ymax = f64::max(ymax, y);

        zmin = f64::min(zmin, z);
        zmax = f64::max(zmax, z);
    }

    let mut global_xmin = 0.0;
    let mut global_xmax = 0.0;

    let mut global_ymin = 0.0;
    let mut global_ymax = 0.0;

    let mut global_zmin = 0.0;
    let mut global_zmax = 0.0;

    comm.all_reduce_into(&xmin, &mut global_xmin, SystemOperation::min());
    comm.all_reduce_into(&xmax, &mut global_xmax, SystemOperation::max());

    comm.all_reduce_into(&ymin, &mut global_ymin, SystemOperation::min());
    comm.all_reduce_into(&ymax, &mut global_ymax, SystemOperation::max());

    comm.all_reduce_into(&zmin, &mut global_zmin, SystemOperation::min());
    comm.all_reduce_into(&zmax, &mut global_zmax, SystemOperation::max());

    let xdiam = global_xmax - global_xmin;
    let ydiam = global_ymax - global_ymin;
    let zdiam = global_zmax - global_zmin;

    let xmean = global_xmin + 0.5 * xdiam;
    let ymean = global_ymin + 0.5 * ydiam;
    let zmean = global_zmin + 0.5 * zdiam;

    // We increase diameters by box size on deepest level
    // and use the maximum diameter to compute a
    // cubic bounding box.

    let deepest_box_diam = 1.0 / (1 << DEEPEST_LEVEL) as f64;

    let max_diam = [xdiam, ydiam, zdiam].into_iter().reduce(f64::max).unwrap();

    // A non-positive or non-finite diameter means no rank contributed a point, or
    // every point coincides. Both make the reference transform divide by zero and
    // map every point to the same key, so fail here instead of silently building a
    // meaningless tree.
    assert!(
        max_diam > 0.0 && max_diam.is_finite(),
        "cannot compute a bounding box with maximum extent {max_diam}: \
         the points must not all coincide and at least one rank must contribute a point"
    );

    let max_diam = max_diam * (1.0 + deepest_box_diam);

    PhysicalBox::new([
        xmean - 0.5 * max_diam,
        ymean - 0.5 * max_diam,
        zmean - 0.5 * max_diam,
        xmean + 0.5 * max_diam,
        ymean + 0.5 * max_diam,
        zmean + 0.5 * max_diam,
    ])
}

/// Convert each local point to a Morton key at the requested level.
///
/// # Parameters
///
/// - `points`: Local `[3, n]` point array, with one point per column.
/// - `max_level`: Mapping level; values above 16 are clamped to 16.
/// - `bounding_box`: Physical domain that strictly contains every point.
///
/// # Panics
/// Panics when `points` does not have shape `[3, n]`.
///
/// # Examples
///
/// ```
/// use nd_octree::{points_to_morton, PhysicalBox};
/// use rlst::rlst_dynamic_array;
///
/// let points = rlst_dynamic_array!(f64, [3, 0]);
/// let domain = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
/// assert!(points_to_morton(&points, 16, &domain).is_empty());
/// ```
pub fn points_to_morton<ArrayImpl: ValueArrayImpl<f64, 2>>(
    points: &Array<ArrayImpl, 2>,
    max_level: usize,
    bounding_box: &PhysicalBox,
) -> Vec<MortonKey> {
    // Make sure that the points array has the right dimension
    assert_eq!(points.shape()[0], 3);
    // Make sure that max level never exceeds DEEPEST_LEVEL
    let max_level = if max_level > DEEPEST_LEVEL as usize {
        DEEPEST_LEVEL as usize
    } else {
        max_level
    };

    // Bunch the points in arrays of 3.

    points
        .col_iter()
        .map(|point| {
            let point = unsafe {
                [
                    point.get_value_unchecked([0]),
                    point.get_value_unchecked([1]),
                    point.get_value_unchecked([2]),
                ]
            };
            morton::from_physical_point(point, bounding_box, max_level)
        })
        .collect_vec()
}

#[cfg(test)]
mod test {
    use rlst::SliceArray;

    use crate::{
        constants::DEEPEST_LEVEL,
        geometry::PhysicalBox,
        morton,
        octree::{KeyType, LookupBatchError, LookupError, OctreeOptions, points_to_morton},
    };

    #[test]
    fn test_octree_options_default_and_builder() {
        assert_eq!(OctreeOptions::new(), OctreeOptions::default());
        assert!(!OctreeOptions::default().ghost_children());
        let enabled = OctreeOptions::new().with_ghost_children(true);
        assert!(enabled.ghost_children());
        // The builder consumes and returns, so it is idempotent and reversible.
        assert_eq!(enabled.with_ghost_children(true), enabled);
        assert_eq!(enabled.with_ghost_children(false), OctreeOptions::default());
    }

    #[test]
    fn test_points_to_morton_maps_columns_and_clamps() {
        let domain = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        // One point per column: the first cell, an interior point, and a point on
        // the upper face of the domain.
        let data = [0.0, 0.0, 0.0, 0.6, 0.1, 0.9, 1.0, 1.0, 1.0];
        let points = SliceArray::from_shape(&data, [3, 3]);

        assert_eq!(
            points_to_morton(&points, 2, &domain),
            vec![
                morton::from_index_and_level([0, 0, 0], 2),
                morton::from_index_and_level([2, 0, 3], 2),
                // A point on the face is clamped into the last cell, not reported.
                morton::from_index_and_level([3, 3, 3], 2),
            ]
        );

        // A level beyond the deepest supported one is clamped, not rejected.
        let deepest = points_to_morton(&points, DEEPEST_LEVEL as usize, &domain);
        assert_eq!(points_to_morton(&points, 100, &domain), deepest);
        assert!(
            deepest
                .iter()
                .all(|&key| morton::level(key) == DEEPEST_LEVEL as usize)
        );

        // Level zero maps everything to the root.
        assert_eq!(
            points_to_morton(&points, 0, &domain),
            vec![morton::root(); 3]
        );
    }

    #[test]
    #[should_panic]
    fn test_points_to_morton_rejects_a_non_three_dimensional_array() {
        let data = [0.0, 0.0, 0.5, 0.5];
        let points = SliceArray::from_shape(&data, [2, 2]);
        let _ = points_to_morton(
            &points,
            2,
            &PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        );
    }

    #[test]
    fn test_key_type_ghost_reporting() {
        for key_type in [KeyType::LocalLeaf, KeyType::LocalInterior, KeyType::Global] {
            assert!(!key_type.is_ghost());
            assert_eq!(key_type.ghost_rank(), None);
        }
        for key_type in [KeyType::GhostLeaf(0), KeyType::GhostInterior(3)] {
            assert!(key_type.is_ghost());
        }
        // Ghost ranks are zero based, so rank zero must not read as "no rank".
        assert_eq!(KeyType::GhostLeaf(0).ghost_rank(), Some(0));
        assert_eq!(KeyType::GhostInterior(3).ghost_rank(), Some(3));
    }

    #[test]
    fn test_lookup_errors_describe_themselves() {
        assert_eq!(LookupError::InvalidKey.to_string(), "Morton key is invalid");
        assert_eq!(
            LookupError::NotFinestLevel { level: 4 }.to_string(),
            "Morton key level 4 is not the finest level"
        );
        assert_eq!(
            LookupError::LeafNotFound.to_string(),
            "no containing leaf was found"
        );
        assert_eq!(
            LookupBatchError::CountOverflow.to_string(),
            "MPI lookup count or displacement overflow"
        );
    }
}

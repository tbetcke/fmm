//! Exchange of ghost data between the ranks of an FMM.
//!
//! The interaction lists of a rank routinely name boxes that are owned by other
//! ranks (see [`crate::interaction_manager`]). An [`FmmGhostCommunicator`] moves
//! the data associated with those boxes — multipole coefficients, local
//! expansions, particle data, … — from their owners to the ranks that hold them
//! as ghosts. It is built on top of the index-based
//! [`GhostCommunicator`] of rlst, with one such communicator per level of the
//! tree.
//!
//! # Workflow
//!
//! The communicator owns a send and a receive buffer on every level. It only
//! performs the communication; moving data in and out of the buffers is the
//! responsibility of the FMM implementation:
//!
//! 1. Fill the send buffer of a level, for instance with
//!    [`FmmGhostCommunicator::send_chunks_mut`], which yields each key that this
//!    rank owns and another rank requires, together with its chunk of values.
//! 2. Call [`FmmGhostCommunicator::forward`] (or
//!    [`FmmGhostCommunicator::forward_all`]).
//! 3. Copy the ghost values out of the receive buffer, for instance with
//!    [`FmmGhostCommunicator::receive_chunks`] or
//!    [`FmmGhostCommunicator::receive_chunk`].
//!
//! [`FmmGhostCommunicator::backward`] sends values in the opposite direction,
//! from the receive buffers of the ghost holders into the send buffers of the
//! owners, for example to return ghost contributions.
//!
//! # Device data
//!
//! Both buffers live in host memory. FMM data that resides on a GPU is staged by
//! the FMM implementation: it copies its device data into the send buffer before
//! the exchange and the receive buffer back to the device afterwards. The flat
//! buffers ([`FmmGhostCommunicator::send_buffer_mut`],
//! [`FmmGhostCommunicator::receive_buffer`]) are contiguous, with the chunk of the
//! `i`-th key at `i * chunk_size..(i + 1) * chunk_size`, so a single bulk copy
//! suffices once the device data is arranged in the order of
//! [`FmmGhostCommunicator::send_keys`] or [`FmmGhostCommunicator::receive_keys`].
//!
//! # Which keys are exchanged
//!
//! The communicator is initialised with the Morton keys whose data this rank
//! requires. Only ghost keys ([`KeyType::GhostLeaf`] and
//! [`KeyType::GhostInterior`]) are exchanged; their owner is read from
//! [`Octree::all_keys`]. Local keys need no communication and are skipped.
//! [`KeyType::Global`] keys are skipped as well: every rank is assumed to compute
//! the data of the global keys itself, so no global data is synchronised. Keys
//! may therefore be passed straight from the interaction lists, and duplicates
//! are ignored.
//!
//! # Collective operations
//!
//! [`FmmGhostCommunicator::new`] is a collective operation on the communicator of
//! the octree. [`FmmGhostCommunicator::forward`] and
//! [`FmmGhostCommunicator::backward`] are neighbourhood collectives on the
//! communicator of their level: every rank must call them for the same levels in
//! the same order, including ranks without any ghosts on that level.

#[cfg(test)]
#[path = "ghost_communicator_tests.rs"]
mod tests;

use std::collections::HashMap;

use mpi::{
    collective::SystemOperation,
    traits::{CommunicatorCollectives, Equivalence},
};
use nd_octree::{MortonKey, Octree, morton, octree::KeyType};
use rlst::distributed_tools::{ChunkSizes, GhostCommunicator, GhostCommunicatorBuilder};

/// Number of values that each ghost key carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LevelChunkSizes {
    /// The same chunk size on every level.
    Uniform(usize),
    /// `sizes[level]` values for every key on `level`. There must be an entry
    /// for every level up to the global maximum level of the tree.
    PerLevel(Vec<usize>),
}

impl LevelChunkSizes {
    /// Return the chunk size on `level`, or `None` if no size is given for it.
    pub fn chunk_size(&self, level: usize) -> Option<usize> {
        match self {
            Self::Uniform(size) => Some(*size),
            Self::PerLevel(sizes) => sizes.get(level).copied(),
        }
    }
}

/// The ghost communicator and the buffers of a single level.
struct LevelGhosts<T> {
    communicator: GhostCommunicator<MortonKey>,
    chunk_size: usize,
    send_buffer: Vec<T>,
    receive_buffer: Vec<T>,
}

/// Ghost communicator for FMM data with values of type `T`.
///
/// See the [module documentation](self) for the workflow and the guarantees.
pub struct FmmGhostCommunicator<T: Equivalence + Copy + Default> {
    /// The communicators and buffers, indexed by level.
    levels: Vec<LevelGhosts<T>>,
    /// The position of every ghost key in the receive keys of its level.
    receive_positions: HashMap<MortonKey, usize>,
}

impl<T: Equivalence + Copy + Default> FmmGhostCommunicator<T> {
    /// Create a ghost communicator for the keys in `ghost_keys`.
    ///
    /// One level communicator is built for every level from 0 up to the global
    /// maximum level of `octree`. Keys that are not ghosts of this rank are
    /// skipped, see the [module documentation](self).
    ///
    /// This is a collective operation on the communicator of `octree`.
    ///
    /// # Panics
    ///
    /// Panics on every rank if, on any rank, a key is not a key of
    /// [`Octree::all_keys`], `chunk_sizes` does not give a size for every level,
    /// or a chunk size is zero. Panics if an owner and a receiver of a ghost
    /// disagree about the chunk size of its level.
    pub fn new<C: CommunicatorCollectives>(
        octree: &Octree<'_, C>,
        ghost_keys: impl IntoIterator<Item = MortonKey>,
        chunk_sizes: LevelChunkSizes,
    ) -> Self {
        let comm = octree.comm();
        let nlevels = octree.global_max_level() + 1;

        let buckets = bucket_ghosts(octree.all_keys(), ghost_keys, nlevels);
        let sizes: Option<Vec<usize>> = (0..nlevels)
            .map(|level| chunk_sizes.chunk_size(level).filter(|&size| size > 0))
            .collect();

        // Validate on every rank before the level communicators are built, so
        // that invalid input on one rank cannot strand the others in a
        // collective.
        let local_valid = buckets.is_some() && sizes.is_some();
        let mut valid = false;
        comm.all_reduce_into(&local_valid, &mut valid, SystemOperation::logical_and());
        assert!(
            valid,
            "every ghost key must be a key of the tree and every level needs a positive chunk size"
        );
        let (buckets, sizes) = (buckets.unwrap(), sizes.unwrap());

        let mut receive_positions = HashMap::new();
        let levels = buckets
            .into_iter()
            .zip(sizes)
            .map(|(bucket, chunk_size)| {
                let communicator = GhostCommunicatorBuilder::<MortonKey>::new()
                    .add_ghosts(bucket)
                    .chunk_sizes(ChunkSizes::Uniform(chunk_size))
                    .build(comm);
                receive_positions.extend(
                    communicator
                        .receive_indices()
                        .iter()
                        .enumerate()
                        .map(|(position, &key)| (key, position)),
                );
                LevelGhosts {
                    send_buffer: vec![T::default(); communicator.send_buffer_len()],
                    receive_buffer: vec![T::default(); communicator.receive_buffer_len()],
                    communicator,
                    chunk_size,
                }
            })
            .collect();

        Self {
            levels,
            receive_positions,
        }
    }

    /// Return the number of levels, that is the global maximum level plus one.
    pub fn nlevels(&self) -> usize {
        self.levels.len()
    }

    /// Return the number of values per key on `level`.
    pub fn chunk_size(&self, level: usize) -> usize {
        self.levels[level].chunk_size
    }

    /// Return the rlst ghost communicator of `level`.
    pub fn communicator(&self, level: usize) -> &GhostCommunicator<MortonKey> {
        &self.levels[level].communicator
    }

    /// Return the keys on `level` that this rank owns and sends to other ranks.
    ///
    /// This is the order of the chunks in the send buffer. A key appears once
    /// for every rank that requires it.
    pub fn send_keys(&self, level: usize) -> &[MortonKey] {
        self.levels[level].communicator.send_indices()
    }

    /// Return the ghost keys on `level` that this rank receives.
    ///
    /// This is the order of the chunks in the receive buffer: the keys are
    /// sorted by owning rank and then by key. Every key appears once.
    pub fn receive_keys(&self, level: usize) -> &[MortonKey] {
        self.levels[level].communicator.receive_indices()
    }

    /// Return the position of the ghost `key` in the receive keys of its level,
    /// or `None` if this rank does not receive it.
    pub fn receive_position(&self, key: MortonKey) -> Option<usize> {
        self.receive_positions.get(&key).copied()
    }

    /// Return the send buffer of `level`.
    pub fn send_buffer(&self, level: usize) -> &[T] {
        &self.levels[level].send_buffer
    }

    /// Return the send buffer of `level` mutably.
    pub fn send_buffer_mut(&mut self, level: usize) -> &mut [T] {
        &mut self.levels[level].send_buffer
    }

    /// Return the receive buffer of `level`.
    pub fn receive_buffer(&self, level: usize) -> &[T] {
        &self.levels[level].receive_buffer
    }

    /// Return the receive buffer of `level` mutably.
    pub fn receive_buffer_mut(&mut self, level: usize) -> &mut [T] {
        &mut self.levels[level].receive_buffer
    }

    /// Iterate over the send keys of `level` together with their chunks in the
    /// send buffer, mutably.
    pub fn send_chunks_mut(&mut self, level: usize) -> impl Iterator<Item = (MortonKey, &mut [T])> {
        let level = &mut self.levels[level];
        level
            .communicator
            .send_indices()
            .iter()
            .copied()
            .zip(level.send_buffer.chunks_exact_mut(level.chunk_size))
    }

    /// Iterate over the receive keys of `level` together with their chunks in
    /// the receive buffer.
    pub fn receive_chunks(&self, level: usize) -> impl Iterator<Item = (MortonKey, &[T])> {
        let level = &self.levels[level];
        level
            .communicator
            .receive_indices()
            .iter()
            .copied()
            .zip(level.receive_buffer.chunks_exact(level.chunk_size))
    }

    /// Return the chunk of the ghost `key` in the receive buffer of its level,
    /// or `None` if this rank does not receive it.
    pub fn receive_chunk(&self, key: MortonKey) -> Option<&[T]> {
        let position = self.receive_position(key)?;
        let level = &self.levels[morton::level(key)];
        let start = position * level.chunk_size;
        Some(&level.receive_buffer[start..start + level.chunk_size])
    }

    /// Send the send buffer of `level` to the receive buffers of the ranks that
    /// hold its keys as ghosts.
    ///
    /// This is a neighbourhood collective: every rank must call it.
    pub fn forward(&mut self, level: usize) {
        let level = &mut self.levels[level];
        level
            .communicator
            .forward_send_values(&level.send_buffer, &mut level.receive_buffer);
    }

    /// Call [`forward`](Self::forward) for every level in ascending order.
    pub fn forward_all(&mut self) {
        for level in 0..self.nlevels() {
            self.forward(level);
        }
    }

    /// Send the receive buffer of `level` back to the send buffers of the
    /// owning ranks, overwriting them.
    ///
    /// An owner that sends a key to several ranks receives one chunk from each
    /// of them. This is a neighbourhood collective: every rank must call it.
    pub fn backward(&mut self, level: usize) {
        let level = &mut self.levels[level];
        level
            .communicator
            .backward_send_values(&level.receive_buffer, &mut level.send_buffer);
    }
}

/// Group the ghost keys among `keys` by level, each paired with its owner.
///
/// Local and global keys are skipped. Every bucket is sorted by key and free of
/// duplicates. Returns `None` if a key is not in `all_keys` or lies on a level
/// of `nlevels` or deeper.
fn bucket_ghosts(
    all_keys: &HashMap<MortonKey, KeyType>,
    keys: impl IntoIterator<Item = MortonKey>,
    nlevels: usize,
) -> Option<Vec<Vec<(MortonKey, usize)>>> {
    let mut buckets = vec![Vec::new(); nlevels];
    for key in keys {
        if let Some(owner) = all_keys.get(&key)?.ghost_rank() {
            buckets.get_mut(morton::level(key))?.push((key, owner));
        }
    }
    for bucket in &mut buckets {
        bucket.sort_unstable();
        bucket.dedup();
    }
    Some(buckets)
}

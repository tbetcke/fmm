//! Ghost exchange and coarse gather in index form.
//!
//! Three exchanges fill the stores of [`store`](super::store) with data other ranks own
//! (design §8, `docs/design/fmm-plan-redesign.md`). Each is built once, from a
//! [`Plan`], and then moves values between flat buffers in index order on every
//! evaluation, without looking a key up:
//!
//! | Exchange | Ghosts | Chunk of a ghost | Fills |
//! | --- | --- | --- | --- |
//! | [`SourceExchange`] | the ghost leaves of the leaf numbering (named by a U- or X-row), in leaf order | the owner's source points, `count · point_size` values, zero allowed | the ghost tail of the source [`LeafStore`], in place |
//! | [`MultipoleExchange`] | per level l, the ghost boxes named by a V-row of level l or a W-row of level l − 1, in box order | `sizes[l]` values | their slots of level l of the multipole [`LevelBuffers`] |
//! | [`CoarseExchange`] | every rank's coarse blocks, in rank order | `sizes[level]` values | the slots of the other ranks' blocks of the multipole [`LevelBuffers`] |
//!
//! The first two are neighbourhood exchanges on rlst's
//! [`GhostCommunicator`] with Morton keys as global indices; local and `Global` boxes are
//! never exchanged. The third is one all-gather on a duplicate of the plan's
//! communicator, for the global upward pass (design §7.4).
//!
//! # Layouts
//!
//! - **Sources.** The ghost leaves are numbered by key, every non-`Global` key lies in
//!   its owner's Morton range, and rlst orders the receive buffer by owner, keeping the
//!   order of each owner's ghosts. So the receive buffer has the order and the layout of
//!   the ghost tail of the source store, and the exchange writes straight into it:
//!   `forward_send_values(send, sources.range_mut(ghosts).as_mut_slice())`. The owners
//!   size each chunk ([`ChunkSizes::PerIndex`]); the build delivers the sizes, and with
//!   them the ghost counts ([`SourceExchange::leaf_counts`]). The send buffer is flat, in
//!   the order of [`SourceExchange::send_leaves`] (a leaf sent to two ranks appears
//!   twice), and is gathered from the store before each exchange.
//! - **Multipoles.** Per level, a flat send buffer gathered by
//!   [`MultipoleExchange::send_boxes`] and a flat receive buffer in ascending box order
//!   ([`MultipoleExchange::receive_boxes`]), scattered into the level buffer after the
//!   exchange. Ghost boxes are interleaved with local and `Global` boxes, so the receive
//!   buffer cannot be the level buffer itself.
//! - **Coarse blocks.** A flat buffer of this rank's blocks in key order, all-gathered
//!   into a flat buffer of every rank's blocks in rank order, which is global key order
//!   because ranks own ascending key ranges ([`CoarseExchange::keys`]). The blocks of the
//!   other ranks are then scattered into their slots; every coarse block has a box index
//!   on every rank (design §3.5).
//!
//! # Collectives
//!
//! Every constructor validates its input, agrees the outcome on every rank with one
//! all-reduce before it builds any communicator, and returns the same error kind on
//! every rank: the rank that found a defect returns it, the others
//! [`ExchangeError::OtherRank`]. After the build, one more all-reduce agrees that every
//! rank found the layout the plan promises. Ranks with no points take part with empty
//! slices.
//!
//! | Call | Collectives, in order |
//! | --- | --- |
//! | [`SourceExchange::new`] | all-reduce (validity, point size agreed); rlst build (three validity all-reduces, an all-to-all, two graph creates, two neighbour all-to-alls); all-reduce (layout) |
//! | [`MultipoleExchange::new`] | all-reduce (validity, sizes agreed); one rlst build per level, l = 0..nlevels; all-reduce (layout) |
//! | [`CoarseExchange::new`] | all-reduce (validity, sizes agreed); communicator duplicate; all-gather (counts); all-gather-v (keys); all-reduce (every block held) |
//! | [`SourceExchange::forward`] | one neighbour all-to-all |
//! | [`MultipoleExchange::forward`] | one neighbour all-to-all on the level's communicator |
//! | [`MultipoleExchange::forward_all`] | [`forward`](MultipoleExchange::forward) for l = 0..nlevels, in order |
//! | [`CoarseExchange::gather`] | one all-gather-v, counts fixed at construction |
//!
//! Every rank must call the exchange methods in the same order. A rank that passes a
//! store of the wrong layout still takes part in the communication, with default values
//! in place of its data, and then panics, so its neighbours are not stranded in the
//! collective. rlst panics on every rank if a buffer outgrows MPI's `i32` counts.
//!
//! # Guarantees
//!
//! - **Index order.** Every send and receive buffer is flat and contiguous, in leaf or
//!   box order (requirement 7), and every index array is built once.
//! - **Exact copies.** Every received chunk is the owner's chunk, value for value; no
//!   exchange combines values, so results are bit-identical from run to run.
//! - **Variable chunks.** Source chunks have the owner's count, zero included.
//! - **Named slots.** Each exchange names what it reads and writes in the stores:
//!   [`SourceExchange::send_leaves`] and [`ghost_leaves`](SourceExchange::ghost_leaves),
//!   [`MultipoleExchange::send_boxes`] and [`receive_boxes`](MultipoleExchange::receive_boxes),
//!   [`CoarseExchange::sent_blocks`] and [`received_blocks`](CoarseExchange::received_blocks).
//!   It reads and writes nothing else, which is what the evaluator's
//!   [host-data events](super::operator#host-data) pass on to the operator.

#[cfg(test)]
#[path = "exchange_tests.rs"]
mod tests;

use std::{cell::Cell, error::Error, fmt, ops::Range};

use mpi::{
    collective::SystemOperation,
    datatype::PartitionMut,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Equivalence},
};
use nd_octree::{MortonKey, morton};
use rlst::distributed_tools::{ChunkSizes, GhostCommunicator, GhostCommunicatorBuilder};

use super::index::BoxIndex;
use super::plan::Plan;
use super::store::{LeafStore, LevelBuffers};

/// The reasons an exchange cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExchangeError {
    /// The counts do not have one entry per local leaf.
    CountsLength {
        /// The number of local leaves.
        expected: usize,
        /// The number of counts given.
        actual: usize,
    },
    /// The sizes do not have one entry per level.
    SizesLength {
        /// The number of levels.
        expected: usize,
        /// The number of sizes given.
        actual: usize,
    },
    /// A size that must be positive is zero.
    ZeroSize,
    /// The ranks pass different sizes.
    SizesDisagree,
    /// A number of values overflows `usize` or MPI's `i32` counts.
    Overflow,
    /// A key that another rank requests from this rank, or a coarse block, is not held
    /// in the form the exchange needs (a local leaf for sources, a non-ghost box for
    /// multipoles, any box for coarse blocks).
    UnheldKey {
        /// The key.
        key: MortonKey,
    },
    /// The ghosts do not arrive in index order, so an octree invariant (each rank owns
    /// one ascending key range) is broken.
    ReceiveOrder,
    /// Building the exchange failed on another rank.
    OtherRank,
}

impl fmt::Display for ExchangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountsLength { expected, actual } => {
                write!(f, "{actual} counts given for {expected} local leaves")
            }
            Self::SizesLength { expected, actual } => {
                write!(f, "{actual} sizes given for {expected} levels")
            }
            Self::ZeroSize => write!(f, "a size that must be positive is zero"),
            Self::SizesDisagree => write!(f, "the ranks pass different sizes"),
            Self::Overflow => write!(f, "a number of values overflows the exchange counts"),
            Self::UnheldKey { key } => {
                write!(f, "key {key} is not held in the form the exchange needs")
            }
            Self::ReceiveOrder => write!(f, "the ghosts do not arrive in index order"),
            Self::OtherRank => write!(f, "building the exchange failed on another rank"),
        }
    }
}

impl Error for ExchangeError {}

/// Agree a local outcome on every rank with one all-reduce, and check that `values` is
/// the same on every rank.
///
/// Returns the local error if there is one, [`ExchangeError::OtherRank`] if another rank
/// failed, and [`ExchangeError::SizesDisagree`] on every rank if all succeeded but the
/// values differ. `values` must have the same length on every rank.
fn agree<C: CommunicatorCollectives>(
    comm: &C,
    local: Result<(), ExchangeError>,
    values: &[usize],
) -> Result<(), ExchangeError> {
    // The maximum of [failed, v, −v] gives "any rank failed", max v and −min v at once.
    let as_i64 = |value: usize| i64::try_from(value).unwrap_or(i64::MAX);
    let mut mine = Vec::with_capacity(1 + 2 * values.len());
    mine.push(i64::from(local.is_err()));
    mine.extend(values.iter().map(|&value| as_i64(value)));
    mine.extend(values.iter().map(|&value| -as_i64(value)));
    let mut all = vec![0i64; mine.len()];
    comm.all_reduce_into(&mine[..], &mut all[..], SystemOperation::max());
    local?;
    if all[0] != 0 {
        return Err(ExchangeError::OtherRank);
    }
    let n = values.len();
    if (0..n).any(|i| all[1 + i] != -all[1 + n + i]) {
        return Err(ExchangeError::SizesDisagree);
    }
    Ok(())
}

/// Agree a local outcome on every rank with one all-reduce.
fn agree_valid<C: CommunicatorCollectives>(
    comm: &C,
    local: Result<(), ExchangeError>,
) -> Result<(), ExchangeError> {
    agree(comm, local, &[])
}

/// Check that `sizes` has one positive entry per level, and return it padded or cut to
/// `nlevels` entries, so that every rank can take part in [`agree`] with the same length.
fn level_sizes(sizes: &[usize], nlevels: usize) -> (Result<(), ExchangeError>, Vec<usize>) {
    let padded = (0..nlevels)
        .map(|level| sizes.get(level).copied().unwrap_or(0))
        .collect();
    let check = if sizes.len() != nlevels {
        Err(ExchangeError::SizesLength {
            expected: nlevels,
            actual: sizes.len(),
        })
    } else if sizes.contains(&0) {
        Err(ExchangeError::ZeroSize)
    } else {
        Ok(())
    };
    (check, padded)
}

/// The ghost leaves of `index`, in leaf order, each with its owner.
fn source_ghosts(index: &BoxIndex) -> Vec<(MortonKey, usize)> {
    let leaves = index.leaves();
    leaves
        .ghosts()
        .map(|j| {
            let kind = index.kind(leaves.level(j), leaves.box_index(j) as usize);
            let owner = kind
                .ghost_rank()
                .expect("every ghost leaf of the numbering is a ghost");
            (leaves.key(j), owner)
        })
        .collect()
}

/// The number of source values that the owner sends for `key`: `count · point_size` if
/// `key` is a local leaf, `None` otherwise.
fn owned_chunk_size(
    index: &BoxIndex,
    local_counts: &[usize],
    point_size: usize,
    key: MortonKey,
) -> Option<usize> {
    let leaf = index.find_leaf(key)? as usize;
    local_counts.get(leaf)?.checked_mul(point_size)
}

/// The ghost counts that the chunk sizes `sizes` of the source exchange describe, or
/// `None` if a size is not a multiple of `point_size`.
fn ghost_counts(sizes: &[usize], point_size: usize) -> Option<Vec<usize>> {
    sizes
        .iter()
        .map(|&size| (size % point_size == 0).then_some(size / point_size))
        .collect()
}

/// The ghost boxes of `level` named by a V-row of `level` or a W-row of `level − 1`,
/// ascending, each once, with its owner.
fn multipole_ghosts(plan: &Plan, level: usize) -> Vec<(u32, usize)> {
    let index = plan.index();
    let mut boxes: Vec<u32> = plan.level(level).v().sources().to_vec();
    if level > 0 {
        boxes.extend_from_slice(plan.level(level - 1).w().entries());
    }
    boxes.sort_unstable();
    boxes.dedup();
    boxes
        .into_iter()
        .filter_map(|i| {
            let owner = index.kind(level, i as usize).ghost_rank()?;
            Some((i, owner))
        })
        .collect()
}

/// The exchange of source data for the ghost leaves of the U- and X-lists.
///
/// See the [module documentation](self).
pub struct SourceExchange<T> {
    communicator: GhostCommunicator<MortonKey>,
    point_size: usize,
    nlocal: usize,
    /// Counts of every leaf of the numbering: the caller's, then the owners'.
    leaf_counts: Vec<usize>,
    /// Leaf index of every chunk of the send buffer.
    send_leaves: Vec<u32>,
    send_buffer: Vec<T>,
}

impl<T: Equivalence + Copy + Default> SourceExchange<T> {
    /// Build the exchange for the ghost leaves of `plan`, with `local_counts[j]` source
    /// points of `point_size` values on local leaf j.
    ///
    /// # Collective operation
    /// On `comm`, the communicator of the octree of `plan`; see the
    /// [module documentation](self) for the collectives.
    ///
    /// # Errors
    /// Returned on every rank if, on any rank, `local_counts` does not have one entry per
    /// local leaf, `point_size` is zero or differs between ranks, the values overflow, a
    /// rank is asked for a key that is not one of its local leaves, or the ghosts do not
    /// arrive in leaf order.
    pub fn new<C: CommunicatorCollectives>(
        plan: &Plan,
        comm: &C,
        local_counts: &[usize],
        point_size: usize,
    ) -> Result<Self, ExchangeError> {
        let index = plan.index();
        let nlocal = index.leaves().nlocal();
        let local = if local_counts.len() != nlocal {
            Err(ExchangeError::CountsLength {
                expected: nlocal,
                actual: local_counts.len(),
            })
        } else if point_size == 0 {
            Err(ExchangeError::ZeroSize)
        } else if local_counts
            .iter()
            .try_fold(0usize, |total, &count| total.checked_add(count))
            .and_then(|total| total.checked_mul(point_size))
            .is_none()
        {
            Err(ExchangeError::Overflow)
        } else {
            Ok(())
        };
        agree(comm, local, &[point_size])?;

        let ghosts = source_ghosts(index);
        let unheld = Cell::new(None);
        let communicator = GhostCommunicatorBuilder::<MortonKey>::new()
            .add_ghosts(ghosts.iter().copied())
            .chunk_sizes(ChunkSizes::PerIndex(Box::new(|key| {
                owned_chunk_size(index, local_counts, point_size, key).unwrap_or_else(|| {
                    unheld.set(Some(key));
                    0
                })
            })))
            .build(comm);

        let send_leaves: Result<Vec<u32>, ExchangeError> = communicator
            .send_indices()
            .iter()
            .map(|&key| {
                index
                    .find_leaf(key)
                    .filter(|&j| (j as usize) < nlocal)
                    .ok_or(ExchangeError::UnheldKey { key })
            })
            .collect();
        let counts = ghost_counts(communicator.receive_chunk_sizes(), point_size);
        let in_order = communicator
            .receive_indices()
            .iter()
            .eq(ghosts.iter().map(|(key, _)| key));
        let layout = match (unheld.get(), &send_leaves, &counts) {
            (Some(key), _, _) => Err(ExchangeError::UnheldKey { key }),
            (None, Err(error), _) => Err(*error),
            (None, Ok(_), Some(_)) if in_order => Ok(()),
            _ => Err(ExchangeError::ReceiveOrder),
        };
        agree_valid(comm, layout)?;

        let mut leaf_counts = local_counts.to_vec();
        leaf_counts.extend(counts.unwrap());
        Ok(Self {
            send_buffer: vec![T::default(); communicator.send_buffer_len()],
            communicator,
            point_size,
            nlocal,
            leaf_counts,
            send_leaves: send_leaves?,
        })
    }

    /// Return the number of values per source point.
    pub fn point_size(&self) -> usize {
        self.point_size
    }

    /// Return the source counts of every leaf of the numbering: the local counts given
    /// to [`new`](Self::new), then the owners' counts of the ghost leaves.
    pub fn leaf_counts(&self) -> &[usize] {
        &self.leaf_counts
    }

    /// Return the owners' source counts of the ghost leaves, in leaf order.
    pub fn ghost_counts(&self) -> &[usize] {
        &self.leaf_counts[self.nlocal..]
    }

    /// Return the leaf indices of the ghost leaves, the leaves this exchange fills.
    pub fn ghost_leaves(&self) -> Range<usize> {
        self.nlocal..self.leaf_counts.len()
    }

    /// Return a zeroed source store with the layout of this exchange: every leaf of the
    /// numbering, with [`leaf_counts`](Self::leaf_counts).
    pub fn new_store(&self) -> LeafStore<T> {
        LeafStore::new(&self.leaf_counts, self.point_size)
    }

    /// Return the local leaf of every chunk of the send buffer, in send order.
    pub fn send_leaves(&self) -> &[u32] {
        &self.send_leaves
    }

    /// Return the rlst communicator, for its counts and neighbours.
    pub fn communicator(&self) -> &GhostCommunicator<MortonKey> {
        &self.communicator
    }

    /// Send the local chunks that other ranks hold as ghosts, and receive the ghost
    /// chunks straight into the ghost tail of `sources`.
    ///
    /// This is a neighbourhood collective: every rank must call it.
    ///
    /// # Panics
    ///
    /// Panics, after taking part in the exchange, if `sources` does not have the layout
    /// of [`new_store`](Self::new_store).
    pub fn forward(&mut self, sources: &mut LeafStore<T>) {
        let layout =
            sources.point_size() == self.point_size && sources.has_counts(&self.leaf_counts);
        if layout {
            let offsets = self.communicator.send_offsets();
            for (k, &leaf) in self.send_leaves.iter().enumerate() {
                self.send_buffer[offsets[k]..offsets[k + 1]]
                    .copy_from_slice(sources.chunk(leaf as usize));
            }
            let mut tail = sources.range_mut(self.ghost_leaves());
            self.communicator
                .forward_send_values(&self.send_buffer, tail.as_mut_slice());
        } else {
            let mut scratch = vec![T::default(); self.communicator.receive_buffer_len()];
            self.communicator
                .forward_send_values(&self.send_buffer, &mut scratch);
        }
        assert!(layout, "the source store does not have the exchange layout");
    }
}

/// The exchange of one level's multipoles.
struct LevelExchange<T> {
    communicator: GhostCommunicator<MortonKey>,
    size: usize,
    len: usize,
    send_boxes: Vec<u32>,
    receive_boxes: Vec<u32>,
    send_buffer: Vec<T>,
    receive_buffer: Vec<T>,
}

/// The exchange of multipoles for the ghost boxes of the V- and W-lists, one per level.
///
/// See the [module documentation](self).
pub struct MultipoleExchange<T> {
    levels: Vec<LevelExchange<T>>,
}

impl<T: Equivalence + Copy + Default> MultipoleExchange<T> {
    /// Build the exchange for the ghost boxes of `plan`, with `sizes[l]` values per
    /// multipole of level l.
    ///
    /// # Collective operation
    /// On `comm`, the communicator of the octree of `plan`; see the
    /// [module documentation](self) for the collectives.
    ///
    /// # Errors
    /// Returned on every rank if, on any rank, `sizes` does not have one positive entry
    /// per level, the sizes differ between ranks, a rank is asked for a key that is not
    /// one of its non-ghost boxes, or the ghosts do not arrive in box order.
    pub fn new<C: CommunicatorCollectives>(
        plan: &Plan,
        comm: &C,
        sizes: &[usize],
    ) -> Result<Self, ExchangeError> {
        let index = plan.index();
        let nlevels = index.nlevels();
        let (local, sizes) = level_sizes(sizes, nlevels);
        agree(comm, local, &sizes)?;

        let mut layout = Ok(());
        let mut levels = Vec::with_capacity(nlevels);
        for (level, &size) in sizes.iter().enumerate() {
            let ghosts = multipole_ghosts(plan, level);
            let keys = index.keys(level);
            let communicator = GhostCommunicatorBuilder::<MortonKey>::new()
                .add_ghosts(ghosts.iter().map(|&(i, owner)| (keys[i as usize], owner)))
                .chunk_sizes(ChunkSizes::Uniform(size))
                .build(comm);

            let mut send_boxes = Vec::with_capacity(communicator.total_send_count());
            for &key in communicator.send_indices() {
                match index.find(key) {
                    Some((l, i)) if l == level && !index.kind(l, i as usize).is_ghost() => {
                        send_boxes.push(i)
                    }
                    _ if layout.is_ok() => layout = Err(ExchangeError::UnheldKey { key }),
                    _ => {}
                }
            }
            let receive_boxes: Vec<u32> = ghosts.iter().map(|&(i, _)| i).collect();
            let in_order = communicator
                .receive_indices()
                .iter()
                .eq(receive_boxes.iter().map(|&i| &keys[i as usize]));
            if !in_order && layout.is_ok() {
                layout = Err(ExchangeError::ReceiveOrder);
            }

            levels.push(LevelExchange {
                send_buffer: vec![T::default(); communicator.send_buffer_len()],
                receive_buffer: vec![T::default(); communicator.receive_buffer_len()],
                communicator,
                size,
                len: index.len(level),
                send_boxes,
                receive_boxes,
            });
        }
        agree_valid(comm, layout)?;
        Ok(Self { levels })
    }

    /// Return the number of levels.
    pub fn nlevels(&self) -> usize {
        self.levels.len()
    }

    /// Return the number of values per multipole of `level`.
    pub fn size(&self, level: usize) -> usize {
        self.levels[level].size
    }

    /// Return the box index on `level` of every chunk of the send buffer, in send order;
    /// a box sent to two ranks appears twice.
    pub fn send_boxes(&self, level: usize) -> &[u32] {
        &self.levels[level].send_boxes
    }

    /// Return the box index on `level` of every chunk of the receive buffer, strictly
    /// ascending.
    pub fn receive_boxes(&self, level: usize) -> &[u32] {
        &self.levels[level].receive_boxes
    }

    /// Return the receive buffer of `level`, as of the last exchange.
    pub fn receive_buffer(&self, level: usize) -> &[T] {
        &self.levels[level].receive_buffer
    }

    /// Return the rlst communicator of `level`, for its counts and neighbours.
    pub fn communicator(&self, level: usize) -> &GhostCommunicator<MortonKey> {
        &self.levels[level].communicator
    }

    /// Send the multipoles of `level` that other ranks hold as ghosts, and write the
    /// received ghost multipoles into their slots of `multipoles`.
    ///
    /// This is a neighbourhood collective: every rank must call it for the same levels in
    /// the same order.
    ///
    /// # Panics
    ///
    /// Panics, after taking part in the exchange, if `multipoles` does not have the
    /// plan's number of levels and boxes and this exchange's sizes.
    pub fn forward(&mut self, level: usize, multipoles: &mut LevelBuffers<T>) {
        let nlevels = self.levels.len();
        let exchange = &mut self.levels[level];
        let layout = multipoles.nlevels() == nlevels
            && multipoles.len(level) == exchange.len
            && multipoles.size(level) == exchange.size;
        let size = exchange.size;
        if layout {
            let source = multipoles.level(level);
            for (chunk, &i) in exchange
                .send_buffer
                .chunks_exact_mut(size)
                .zip(&exchange.send_boxes)
            {
                chunk.copy_from_slice(source.chunk(i as usize));
            }
        }
        exchange
            .communicator
            .forward_send_values(&exchange.send_buffer, &mut exchange.receive_buffer);
        assert!(
            layout,
            "the multipole buffers do not have the exchange layout"
        );
        let mut target = multipoles.level_mut(level);
        for (chunk, &i) in exchange
            .receive_buffer
            .chunks_exact(size)
            .zip(&exchange.receive_boxes)
        {
            target.chunk_mut(i as usize).copy_from_slice(chunk);
        }
    }

    /// Call [`forward`](Self::forward) for every level, in ascending order.
    pub fn forward_all(&mut self, multipoles: &mut LevelBuffers<T>) {
        for level in 0..self.nlevels() {
            self.forward(level, multipoles);
        }
    }
}

/// The gather of every rank's coarse-block multipoles to every rank, for the global
/// upward pass.
///
/// See the [module documentation](self).
pub struct CoarseExchange<T> {
    comm: SimpleCommunicator,
    rank: usize,
    /// Every coarse block, in rank order, which is key order.
    keys: Vec<MortonKey>,
    /// Level and box index on this rank of every coarse block.
    blocks: Vec<(usize, u32)>,
    /// `nblocks + 1` value offsets into the gathered buffer.
    block_offsets: Vec<usize>,
    /// `size + 1` block offsets: the blocks of rank r are `rank_blocks[r]..rank_blocks[r + 1]`.
    rank_blocks: Vec<usize>,
    value_counts: Vec<i32>,
    value_displacements: Vec<i32>,
    sizes: Vec<usize>,
    lens: Vec<usize>,
    send_buffer: Vec<T>,
    gathered: Vec<T>,
}

impl<T: Equivalence + Copy + Default> CoarseExchange<T> {
    /// Build the gather of the coarse blocks of every rank, with `sizes[l]` values per
    /// multipole of level l.
    ///
    /// # Collective operation
    /// On `comm`, the communicator of the octree of `plan`; see the
    /// [module documentation](self) for the collectives. The exchange keeps a duplicate
    /// of `comm`.
    ///
    /// # Errors
    /// Returned on every rank if, on any rank, `sizes` does not have one positive entry
    /// per level, the sizes differ between ranks, a coarse block of any rank has no box
    /// index on this rank, or the values overflow MPI's `i32` counts.
    pub fn new<C: CommunicatorCollectives>(
        plan: &Plan,
        comm: &C,
        sizes: &[usize],
    ) -> Result<Self, ExchangeError> {
        let index = plan.index();
        let nlevels = index.nlevels();
        let own = plan.coarse_blocks();
        let (mut local, sizes) = level_sizes(sizes, nlevels);
        if local.is_ok() && i32::try_from(own.len()).is_err() {
            local = Err(ExchangeError::Overflow);
        }
        agree(comm, local, &sizes)?;

        let comm = comm.duplicate();
        let nranks = comm.size() as usize;
        let rank = comm.rank() as usize;
        let mut key_counts = vec![0i32; nranks];
        comm.all_gather_into(&(own.len() as i32), &mut key_counts[..]);
        let key_displacements = displacements(&key_counts).ok_or(ExchangeError::Overflow)?;
        let nkeys = key_counts.iter().map(|&count| count as usize).sum();
        let mut keys = vec![MortonKey::default(); nkeys];
        comm.all_gather_varcount_into(
            own,
            &mut PartitionMut::new(&mut keys[..], &key_counts[..], &key_displacements[..]),
        );

        // Every rank maps the same keys with the same sizes, so the offsets and counts
        // agree everywhere; only the box indices are rank-specific.
        let mut rank_blocks = vec![0usize];
        for &count in &key_counts {
            rank_blocks.push(rank_blocks.last().unwrap() + count as usize);
        }
        let mut blocks = Vec::with_capacity(nkeys);
        let mut block_offsets = vec![0usize];
        let mut held = Ok(());
        for &key in &keys {
            let block = index.find(key);
            if block.is_none() && held.is_ok() {
                held = Err(ExchangeError::UnheldKey { key });
            }
            blocks.push(block.unwrap_or((0, 0)));
            // The size comes from the key, not from this rank's index, so that every rank
            // computes the same offsets, and fails in the same way, even for an unheld key.
            let size = sizes.get(morton::level(key)).copied().unwrap_or(0);
            let end = block_offsets
                .last()
                .unwrap()
                .checked_add(size)
                .ok_or(ExchangeError::Overflow)?;
            block_offsets.push(end);
        }
        let value_counts: Option<Vec<i32>> = rank_blocks
            .windows(2)
            .map(|r| i32::try_from(block_offsets[r[1]] - block_offsets[r[0]]).ok())
            .collect();
        let value_counts = value_counts.ok_or(ExchangeError::Overflow)?;
        let value_displacements = displacements(&value_counts).ok_or(ExchangeError::Overflow)?;
        agree_valid(&comm, held)?;

        let own_values = block_offsets[rank_blocks[rank + 1]] - block_offsets[rank_blocks[rank]];
        Ok(Self {
            send_buffer: vec![T::default(); own_values],
            gathered: vec![T::default(); *block_offsets.last().unwrap()],
            comm,
            rank,
            keys,
            blocks,
            block_offsets,
            rank_blocks,
            value_counts,
            value_displacements,
            sizes,
            lens: (0..nlevels).map(|level| index.len(level)).collect(),
        })
    }

    /// Return every rank's coarse blocks, in rank order, which is ascending key order.
    pub fn keys(&self) -> &[MortonKey] {
        &self.keys
    }

    /// Return the level and the box index on this rank of coarse block `b` (an index
    /// into [`keys`](Self::keys)).
    pub fn block(&self, b: usize) -> (usize, u32) {
        self.blocks[b]
    }

    /// Return the coarse blocks of `rank`, as a range of indices into
    /// [`keys`](Self::keys).
    pub fn rank_blocks(&self, rank: usize) -> Range<usize> {
        self.rank_blocks[rank]..self.rank_blocks[rank + 1]
    }

    /// Return the blocks whose multipoles [`gather`](Self::gather) reads from this rank's
    /// store and sends, as a range of indices into [`keys`](Self::keys): this rank's
    /// blocks, or none on a one-rank communicator, where the gather goes to the rank
    /// itself and nobody uses the value it reads.
    pub fn sent_blocks(&self) -> Range<usize> {
        if self.rank_blocks.len() == 2 {
            0..0
        } else {
            self.rank_blocks(self.rank)
        }
    }

    /// Return the blocks whose multipoles [`gather`](Self::gather) writes into this rank's
    /// store, every other rank's, as indices into [`keys`](Self::keys), ascending; none on
    /// a one-rank communicator.
    pub fn received_blocks(&self) -> impl Iterator<Item = usize> + use<T> {
        let own = self.rank_blocks(self.rank);
        (0..own.start).chain(own.end..self.keys.len())
    }

    /// Return the multipole of coarse block `b` in the gathered buffer, as of the last
    /// [`gather`](Self::gather).
    pub fn chunk(&self, b: usize) -> &[T] {
        &self.gathered[self.block_offsets[b]..self.block_offsets[b + 1]]
    }

    /// Return the gathered multipoles of every coarse block, block after block, as of
    /// the last [`gather`](Self::gather).
    pub fn gathered(&self) -> &[T] {
        &self.gathered
    }

    /// Gather the multipoles of every rank's coarse blocks and write those of the other
    /// ranks' blocks into their slots of `multipoles`; this rank's blocks are left as
    /// they are.
    ///
    /// This is a collective operation: every rank must call it.
    ///
    /// # Panics
    ///
    /// Panics, after taking part in the gather, if `multipoles` does not have the plan's
    /// number of levels and boxes and this exchange's sizes.
    pub fn gather(&mut self, multipoles: &mut LevelBuffers<T>) {
        let layout = multipoles.nlevels() == self.sizes.len()
            && (0..self.sizes.len())
                .all(|l| multipoles.len(l) == self.lens[l] && multipoles.size(l) == self.sizes[l]);
        let own = self.rank_blocks(self.rank);
        if layout {
            let base = self.block_offsets[own.start];
            for b in own.clone() {
                let (level, i) = self.blocks[b];
                self.send_buffer[self.block_offsets[b] - base..self.block_offsets[b + 1] - base]
                    .copy_from_slice(multipoles.chunk(level, i as usize));
            }
        }
        self.comm.all_gather_varcount_into(
            &self.send_buffer[..],
            &mut PartitionMut::new(
                &mut self.gathered[..],
                &self.value_counts[..],
                &self.value_displacements[..],
            ),
        );
        assert!(
            layout,
            "the multipole buffers do not have the exchange layout"
        );
        for b in self.received_blocks() {
            let (level, i) = self.blocks[b];
            multipoles
                .chunk_mut(level, i as usize)
                .copy_from_slice(&self.gathered[self.block_offsets[b]..self.block_offsets[b + 1]]);
        }
    }
}

/// Exclusive prefix sums of `counts`, or `None` if they overflow `i32`.
fn displacements(counts: &[i32]) -> Option<Vec<i32>> {
    let mut total = 0i32;
    counts
        .iter()
        .map(|&count| {
            let start = total;
            total = total.checked_add(count)?;
            Some(start)
        })
        .collect()
}

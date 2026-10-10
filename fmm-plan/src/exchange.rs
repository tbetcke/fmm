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
//! | [`SourceExchange::forward_overlapped`] | none: point-to-point requests on the exchange's graph communicator |
//! | [`MultipoleExchange::forward_all_overlapped`] | none: point-to-point requests on every level's graph communicator |
//!
//! Every rank must call the exchange methods in the same order. A rank that passes a
//! store of the wrong layout still takes part in the communication, with default values
//! in place of its data, and then panics, so its neighbours are not stranded in the
//! collective. rlst panics on every rank if a buffer outgrows MPI's `i32` counts.
//!
//! # Non-blocking exchanges
//!
//! [`SourceExchange::forward_overlapped`] and [`MultipoleExchange::forward_all_overlapped`]
//! move the same values as [`SourceExchange::forward`] and
//! [`MultipoleExchange::forward_all`], bit for bit, while the caller works
//! (`docs/design/distributed-fmm.md` §8.4, decision 10 of docs/phase5/README.md). rsmpi
//! has no non-blocking neighbourhood collective, so they post safe scoped point-to-point
//! requests instead:
//!
//! - **Messages.** On the exchange's own graph communicator (rlst's `forward_comm`,
//!   created without reordering, so its ranks are the plan communicator's): per
//!   neighbour with values, one receive and one send, tag 0, the value ranges rebuilt
//!   from the public counts and offsets of the [`GhostCommunicator`] (neighbour k's
//!   indices are the k-th block of `send_counts`, its values the range of `send_offsets`
//!   over them; likewise for receiving). Each exchange, and each level of the multipole
//!   exchange, has its own communicator, so messages of different exchanges never match
//!   each other, and within one MPI's non-overtaking order and one message per
//!   neighbour and direction pair them. A neighbour whose value count is zero (a source
//!   exchange of empty leaves) is skipped on both sides, which both know from the build.
//!   No rank messages itself: a ghost is never owned by its holder.
//! - **Scope.** A scoped request cannot outlive its scope, so each method posts, runs
//!   the caller's work and waits, in one `mpi::request::scope`. The receives land in a
//!   staging buffer of the exchange (allocated at the first non-blocking call), not in
//!   the store, so that the work can read the stores while messages are pending; the
//!   send buffers are packed before posting and not touched until every request has
//!   completed. A panic inside the work, with requests pending, aborts the process
//!   (rsmpi).
//! - **Progress.** MPI moves a message only inside MPI calls (design §1.2: 64 kB and
//!   more do not progress at all without them). The work calls `test` on its in-flight
//!   handle ([`SourcesInFlight`], [`MultipolesInFlight`]) between its own calls: one pass
//!   of `MPI_Test` over the pending requests, on the calling thread
//!   (`Request::test`, not `RequestCollection::test_some`, which panics once every request
//!   has completed in rsmpi 0.8.2). The waits block in `MPI_Wait`.
//! - **Times.** Each returns its [`ExchangeTimes`]: from the first post to the
//!   completion of the last request as the test or wait that found it saw it, the time
//!   blocked in waits, and the time in `test` calls, read on the calling thread for
//!   reports; nothing depends on them.
//! - **Determinism.** The values are copied, never combined, so the stores receive the
//!   same bits as from the blocking exchanges, whatever the order in which messages
//!   arrive.
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

use std::{
    cell::Cell,
    error::Error,
    fmt,
    ops::Range,
    time::{Duration, Instant},
};

use mpi::{
    collective::SystemOperation,
    datatype::PartitionMut,
    request::{self, LocalScope, Request},
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Destination, Equivalence, Source},
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

/// Wall times of one non-blocking exchange in an evaluation
/// (`docs/design/distributed-fmm.md` §8.6), read on the calling thread, for reports.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExchangeTimes {
    /// From the first post to the completion of the last request, as seen by the `test`
    /// or wait that found it complete.
    pub total: Duration,
    /// Blocked in waits: the communication the work did not hide.
    pub exposed: Duration,
    /// The part of [`exposed`](Self::exposed) in the final wait for the sends, once the
    /// work is done: a send completes when its receiver has taken the message, so this
    /// also holds the time a faster rank waits for a slower neighbour to reach an MPI
    /// call (load imbalance, not transfer).
    pub sends: Duration,
    /// In the `test` calls between the work's calls.
    pub progress: Duration,
}

/// What one exchange moves on this rank per call: the messages of the non-blocking
/// exchange (one per neighbour with values) and the values, sent and received.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Traffic {
    /// Messages sent: neighbours this rank sends values to.
    pub messages_sent: usize,
    /// Messages received: neighbours this rank receives values from.
    pub messages_received: usize,
    /// Values sent, over every neighbour (a chunk sent to two ranks counts twice).
    pub values_sent: usize,
    /// Values received.
    pub values_received: usize,
}

/// The neighbours `ranks`, each with the range of its values in a buffer laid out by
/// `offsets`; the indices of neighbour k are the k-th block of `counts`.
fn neighbour_values<'g>(
    ranks: &'g [i32],
    counts: &'g [i32],
    offsets: &'g [usize],
) -> impl Iterator<Item = (i32, Range<usize>)> + 'g {
    let mut first = 0usize;
    ranks.iter().zip(counts).map(move |(&rank, &count)| {
        let last = first + count as usize;
        let values = offsets[first]..offsets[last];
        first = last;
        (rank, values)
    })
}

/// The out-neighbours of `communicator` with their ranges in the send buffer.
fn sends(
    communicator: &GhostCommunicator<MortonKey>,
) -> impl Iterator<Item = (i32, Range<usize>)> + '_ {
    neighbour_values(
        communicator.out_ranks(),
        communicator.send_counts(),
        communicator.send_offsets(),
    )
}

/// The in-neighbours of `communicator` with their ranges in the receive buffer.
fn receives(
    communicator: &GhostCommunicator<MortonKey>,
) -> impl Iterator<Item = (i32, Range<usize>)> + '_ {
    neighbour_values(
        communicator.in_ranks(),
        communicator.receive_counts(),
        communicator.receive_offsets(),
    )
}

/// The [`Traffic`] of one forward exchange on `communicator`.
fn traffic(communicator: &GhostCommunicator<MortonKey>) -> Traffic {
    Traffic {
        messages_sent: sends(communicator).filter(|(_, v)| !v.is_empty()).count(),
        messages_received: receives(communicator)
            .filter(|(_, v)| !v.is_empty())
            .count(),
        values_sent: communicator.send_buffer_len(),
        values_received: communicator.receive_buffer_len(),
    }
}

/// A request of a non-blocking exchange, registered with the scope of its stage.
type Pending<'a, 's, T> = Request<'a, [T], &'s LocalScope<'a>>;

/// One message of a non-blocking exchange.
enum Message<'a, 's, T> {
    /// A send in flight.
    Sending(Pending<'a, 's, T>),
    /// A receive in flight, with the offset of its values in the receive buffer.
    Receiving(usize, Pending<'a, 's, T>),
    /// A completed receive whose values are not yet copied out.
    Received(usize, &'a [T]),
    /// Completed, and copied out if a receive.
    Done,
}

/// The messages of one forward exchange (the sources, or one level of multipoles),
/// posted in one scope: a receive and a send per neighbour with values.
struct Posted<'a, 's, T> {
    messages: Vec<Message<'a, 's, T>>,
    /// Messages not yet complete.
    pending: usize,
}

impl<'a, 's, T: Equivalence + Copy> Posted<'a, 's, T> {
    /// Post the receives of `communicator` into `receive` and then its sends from `send`,
    /// both laid out as its buffers, skipping the neighbours without values.
    fn new(
        scope: &'s LocalScope<'a>,
        communicator: &GhostCommunicator<MortonKey>,
        send: &'a [T],
        receive: &'a mut [T],
    ) -> Self {
        let comm = communicator.forward_comm();
        let mut messages =
            Vec::with_capacity(communicator.in_ranks().len() + communicator.out_ranks().len());
        let mut rest = receive;
        for (rank, values) in receives(communicator) {
            debug_assert_ne!(rank, comm.rank(), "a rank holds no ghost of its own");
            let (piece, tail) = std::mem::take(&mut rest).split_at_mut(values.len());
            rest = tail;
            if !piece.is_empty() {
                let request = comm
                    .process_at_rank(rank)
                    .immediate_receive_into(scope, piece);
                messages.push(Message::Receiving(values.start, request));
            }
        }
        for (rank, values) in sends(communicator) {
            if !values.is_empty() {
                let request = comm
                    .process_at_rank(rank)
                    .immediate_send(scope, &send[values]);
                messages.push(Message::Sending(request));
            }
        }
        Self {
            pending: messages.len(),
            messages,
        }
    }

    /// One `MPI_Test` per message in flight.
    fn test(&mut self) {
        for message in &mut self.messages {
            *message = match std::mem::replace(message, Message::Done) {
                Message::Sending(request) => match request.test() {
                    Ok(_) => {
                        self.pending -= 1;
                        Message::Done
                    }
                    Err(request) => Message::Sending(request),
                },
                Message::Receiving(offset, request) => match request.test_with_data() {
                    Ok((_, values)) => {
                        self.pending -= 1;
                        Message::Received(offset, values)
                    }
                    Err(request) => Message::Receiving(offset, request),
                },
                other => other,
            };
        }
    }

    /// Wait for every receive.
    fn wait_receives(&mut self) {
        for message in &mut self.messages {
            if matches!(message, Message::Receiving(..)) {
                let Message::Receiving(offset, request) = std::mem::replace(message, Message::Done)
                else {
                    unreachable!()
                };
                *message = Message::Received(offset, request.wait_for_data());
                self.pending -= 1;
            }
        }
    }

    /// Copy the values received, and not yet copied, into `into`, laid out as the receive
    /// buffer.
    fn copy_received(&mut self, into: &mut [T]) {
        for message in &mut self.messages {
            if let Message::Received(offset, values) = *message {
                into[offset..offset + values.len()].copy_from_slice(values);
                *message = Message::Done;
            }
        }
    }

    /// Wait for every send.
    fn wait_sends(&mut self) {
        for message in &mut self.messages {
            if matches!(message, Message::Sending(_)) {
                let Message::Sending(request) = std::mem::replace(message, Message::Done) else {
                    unreachable!()
                };
                request.wait_without_status();
                self.pending -= 1;
            }
        }
    }
}

/// The posted exchanges of one non-blocking call, with their times.
struct Flight<'a, 's, T> {
    /// One per exchange: the sources, or every level of multipoles.
    groups: Vec<Posted<'a, 's, T>>,
    /// When posting began.
    posted: Instant,
    /// When the last message was seen complete.
    completed: Option<Instant>,
    times: ExchangeTimes,
}

impl<'a, 's, T: Equivalence + Copy> Flight<'a, 's, T> {
    fn new(groups: usize) -> Self {
        Self {
            groups: Vec::with_capacity(groups),
            posted: Instant::now(),
            completed: None,
            times: ExchangeTimes::default(),
        }
    }

    /// Record `now` as the completion if no message is left in flight.
    fn note(&mut self, now: Instant) {
        if self.completed.is_none() && self.groups.iter().all(|g| g.pending == 0) {
            self.completed = Some(now);
        }
    }

    /// Wait for every send, and return the times.
    fn finish(&mut self) -> ExchangeTimes {
        let start = Instant::now();
        for group in &mut self.groups {
            group.wait_sends();
        }
        let end = Instant::now();
        self.times.exposed += end - start;
        self.times.sends = end - start;
        self.note(end);
        debug_assert!(self.groups.iter().all(|g| g.pending == 0));
        self.times.total = self.completed.unwrap_or(end) - self.posted;
        self.times
    }
}

/// A [`Flight`] as the work of a non-blocking call sees it, without the lifetimes of its
/// scope.
trait InFlight<T> {
    /// One pass of `MPI_Test` over every message in flight.
    fn test(&mut self);
    /// Wait for the receives of `group` and copy them into `into`; returns the time
    /// blocked in the wait (the copy not included).
    fn receive(&mut self, group: usize, into: &mut [T]) -> Duration;
    /// The times so far.
    fn times(&self) -> ExchangeTimes;
}

impl<T: Equivalence + Copy> InFlight<T> for Flight<'_, '_, T> {
    fn test(&mut self) {
        if self.completed.is_some() {
            return;
        }
        let start = Instant::now();
        for group in &mut self.groups {
            group.test();
        }
        let end = Instant::now();
        self.times.progress += end - start;
        self.note(end);
    }

    fn receive(&mut self, group: usize, into: &mut [T]) -> Duration {
        let start = Instant::now();
        self.groups[group].wait_receives();
        let end = Instant::now();
        self.times.exposed += end - start;
        self.note(end);
        self.groups[group].copy_received(into);
        end - start
    }

    fn times(&self) -> ExchangeTimes {
        self.times
    }
}

/// A source exchange in flight, as the work of
/// [`SourceExchange::forward_overlapped`] sees it.
pub struct SourcesInFlight<'f, T> {
    flight: &'f mut (dyn InFlight<T> + 'f),
}

impl<T> SourcesInFlight<'_, T> {
    /// Let MPI progress the exchange: one `MPI_Test` per message in flight, on the
    /// calling thread. Call it between the work's own calls.
    pub fn test(&mut self) {
        self.flight.test();
    }

    /// The times of the exchange so far ([`ExchangeTimes::total`] is set at the end).
    pub fn times(&self) -> ExchangeTimes {
        self.flight.times()
    }
}

/// A multipole exchange in flight, as the work of
/// [`MultipoleExchange::forward_all_overlapped`] sees it.
pub struct MultipolesInFlight<'f, T> {
    exchange: &'f mut MultipoleExchange<T>,
    flight: &'f mut (dyn InFlight<T> + 'f),
    /// The levels whose ghost slots [`wait`](Self::wait) wrote.
    written: Vec<bool>,
    /// Whether every store handed to [`wait`](Self::wait) had the exchange's layout.
    layout: bool,
}

impl<T: Equivalence + Copy + Default> MultipolesInFlight<'_, T> {
    /// Let MPI progress the exchange of every level: one `MPI_Test` per message in
    /// flight, on the calling thread. Call it between the work's own calls.
    pub fn test(&mut self) {
        self.flight.test();
    }

    /// Wait for the multipoles of `level`, write them into
    /// [`receive_buffer`](MultipoleExchange::receive_buffer)`(level)` and into their
    /// slots of `multipoles`, as [`MultipoleExchange::forward`] does. Returns the time
    /// blocked. A second call for a level writes the slots again, with the same values.
    ///
    /// A store without the exchange's layout is not written; the exchange then panics at
    /// the end of [`MultipoleExchange::forward_all_overlapped`], once nothing is in
    /// flight.
    pub fn wait(&mut self, level: usize, multipoles: &mut LevelBuffers<T>) -> Duration {
        let exposed = self
            .flight
            .receive(level, &mut self.exchange.levels[level].receive_buffer);
        if self.exchange.has_layout(level, multipoles) {
            self.exchange.scatter(level, multipoles);
        } else {
            self.layout = false;
        }
        self.written[level] = true;
        exposed
    }

    /// Return the exchange, for its index lists and the receive buffers of the levels
    /// waited for.
    pub fn exchange(&self) -> &MultipoleExchange<T> {
        self.exchange
    }

    /// The times of the exchange so far ([`ExchangeTimes::total`] is set at the end).
    pub fn times(&self) -> ExchangeTimes {
        self.flight.times()
    }
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
    /// The receives of [`forward_overlapped`](Self::forward_overlapped), allocated at its
    /// first call.
    staging: Vec<T>,
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
            staging: Vec::new(),
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

    /// Return the messages and values of one exchange on this rank.
    pub fn traffic(&self) -> Traffic {
        traffic(&self.communicator)
    }

    /// Whether `sources` has the layout of [`new_store`](Self::new_store).
    fn has_layout(&self, sources: &LeafStore<T>) -> bool {
        sources.point_size() == self.point_size && sources.has_counts(&self.leaf_counts)
    }

    /// Gather the chunks of [`send_leaves`](Self::send_leaves) into the send buffer.
    fn pack(&mut self, sources: &LeafStore<T>) {
        let offsets = self.communicator.send_offsets();
        for (k, &leaf) in self.send_leaves.iter().enumerate() {
            self.send_buffer[offsets[k]..offsets[k + 1]]
                .copy_from_slice(sources.chunk(leaf as usize));
        }
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
        let layout = self.has_layout(sources);
        if layout {
            self.pack(sources);
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

    /// [`forward`](Self::forward) without blocking: pack the send buffer from `sources`,
    /// post the exchange, run `work` while the messages travel, wait, and write the ghost
    /// chunks into the ghost tail of `sources`, bit for bit as `forward` does (see
    /// [Non-blocking exchanges](self#non-blocking-exchanges)).
    ///
    /// `work` gets `sources` to read, with the ghost tail as it was before the call, and
    /// the exchange in flight, on which it calls [`test`](SourcesInFlight::test) between
    /// its own calls so that MPI progresses the messages. It must not call MPI on the
    /// plan's communicator in a way that waits for another rank's part of this exchange.
    /// Returns its result and the [`ExchangeTimes`].
    ///
    /// Every rank must call it, as [`forward`](Self::forward) (no collective: one
    /// point-to-point receive and send per neighbour with values).
    ///
    /// # Panics
    ///
    /// Panics, after the exchange has completed, if `sources` does not have the layout of
    /// [`new_store`](Self::new_store); the rank sends what its send buffer held before.
    pub fn forward_overlapped<R>(
        &mut self,
        sources: &mut LeafStore<T>,
        work: impl FnOnce(&LeafStore<T>, &mut SourcesInFlight<'_, T>) -> R,
    ) -> (R, ExchangeTimes) {
        let layout = self.has_layout(sources);
        if layout {
            self.pack(sources);
        }
        let mut staging = std::mem::take(&mut self.staging);
        staging.resize(self.communicator.receive_buffer_len(), T::default());
        let ghosts = self.ghost_leaves();
        let (result, times) = request::scope(|scope| {
            let mut flight = Flight::new(1);
            flight.groups.push(Posted::new(
                scope,
                &self.communicator,
                &self.send_buffer,
                &mut staging,
            ));
            let result = work(
                sources,
                &mut SourcesInFlight {
                    flight: &mut flight,
                },
            );
            if layout {
                let mut tail = sources.range_mut(ghosts);
                flight.receive(0, tail.as_mut_slice());
            } else {
                let mut scratch = vec![T::default(); self.communicator.receive_buffer_len()];
                flight.receive(0, &mut scratch);
            }
            (result, flight.finish())
        });
        self.staging = staging;
        assert!(layout, "the source store does not have the exchange layout");
        (result, times)
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
    /// The receives of
    /// [`forward_all_overlapped`](MultipoleExchange::forward_all_overlapped), allocated at
    /// its first call.
    staging: Vec<T>,
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
                staging: Vec::new(),
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

    /// Return the messages and values of the exchange of `level` on this rank.
    pub fn traffic(&self, level: usize) -> Traffic {
        traffic(&self.levels[level].communicator)
    }

    /// Whether level `level` of `multipoles` has the plan's number of levels and boxes and
    /// this exchange's size.
    fn has_layout(&self, level: usize, multipoles: &LevelBuffers<T>) -> bool {
        let exchange = &self.levels[level];
        multipoles.nlevels() == self.levels.len()
            && multipoles.len(level) == exchange.len
            && multipoles.size(level) == exchange.size
    }

    /// Gather the multipoles of [`send_boxes`](Self::send_boxes)`(level)` into the send
    /// buffer of `level`.
    fn pack(&mut self, level: usize, multipoles: &LevelBuffers<T>) {
        let exchange = &mut self.levels[level];
        let source = multipoles.level(level);
        for (chunk, &i) in exchange
            .send_buffer
            .chunks_exact_mut(exchange.size)
            .zip(&exchange.send_boxes)
        {
            chunk.copy_from_slice(source.chunk(i as usize));
        }
    }

    /// Write the receive buffer of `level` into the slots of
    /// [`receive_boxes`](Self::receive_boxes)`(level)`.
    fn scatter(&self, level: usize, multipoles: &mut LevelBuffers<T>) {
        let exchange = &self.levels[level];
        let mut target = multipoles.level_mut(level);
        for (chunk, &i) in exchange
            .receive_buffer
            .chunks_exact(exchange.size)
            .zip(&exchange.receive_boxes)
        {
            target.chunk_mut(i as usize).copy_from_slice(chunk);
        }
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
        let layout = self.has_layout(level, multipoles);
        if layout {
            self.pack(level, multipoles);
        }
        let exchange = &mut self.levels[level];
        exchange
            .communicator
            .forward_send_values(&exchange.send_buffer, &mut exchange.receive_buffer);
        assert!(
            layout,
            "the multipole buffers do not have the exchange layout"
        );
        self.scatter(level, multipoles);
    }

    /// Call [`forward`](Self::forward) for every level, in ascending order.
    pub fn forward_all(&mut self, multipoles: &mut LevelBuffers<T>) {
        for level in 0..self.nlevels() {
            self.forward(level, multipoles);
        }
    }

    /// [`forward_all`](Self::forward_all) without blocking: pack every level's send
    /// buffer from `multipoles`, post the exchange of every level, and run `work` while
    /// the messages travel (see [Non-blocking exchanges](self#non-blocking-exchanges)).
    ///
    /// `work` gets `multipoles` back and the exchange in flight. It waits for each level
    /// with [`MultipolesInFlight::wait`], which writes the level's receive buffer and
    /// ghost slots bit for bit as [`forward`](Self::forward) does, in any order, and calls
    /// [`test`](MultipolesInFlight::test) between its own calls so that MPI progresses the
    /// messages. It must not call MPI on the plan's communicator in a way that waits for
    /// another rank's part of this exchange; a collective that every rank enters (the
    /// coarse gather) is fine. When `work` returns, every send is waited for. Returns its
    /// result and the [`ExchangeTimes`].
    ///
    /// Every rank must call it, as [`forward_all`](Self::forward_all) (no collective: one
    /// point-to-point receive and send per level and neighbour with values).
    ///
    /// # Panics
    ///
    /// Panics, after the exchange has completed, if `work` did not wait for every level
    /// (whose values then reach only the receive buffer), or if a store passed here or to
    /// `wait` does not have the plan's number of levels and boxes and this exchange's
    /// sizes; the rank sends what its send buffers held before.
    pub fn forward_all_overlapped<R>(
        &mut self,
        multipoles: &mut LevelBuffers<T>,
        work: impl FnOnce(&mut MultipolesInFlight<'_, T>, &mut LevelBuffers<T>) -> R,
    ) -> (R, ExchangeTimes) {
        let nlevels = self.levels.len();
        let layout = (0..nlevels).all(|level| self.has_layout(level, multipoles));
        if layout {
            for level in 0..nlevels {
                self.pack(level, multipoles);
            }
        }
        // The buffers the requests borrow live outside the exchange while they are in
        // flight, so that the work can read the exchange.
        let sends: Vec<Vec<T>> = self
            .levels
            .iter_mut()
            .map(|level| std::mem::take(&mut level.send_buffer))
            .collect();
        let mut staging: Vec<Vec<T>> = self
            .levels
            .iter_mut()
            .map(|level| {
                let mut staging = std::mem::take(&mut level.staging);
                staging.resize(level.receive_buffer.len(), T::default());
                staging
            })
            .collect();
        let (result, times, written) = request::scope(|scope| {
            let mut flight = Flight::new(nlevels);
            for ((level, send), receive) in self.levels.iter().zip(&sends).zip(&mut staging) {
                flight
                    .groups
                    .push(Posted::new(scope, &level.communicator, send, receive));
            }
            let mut in_flight = MultipolesInFlight {
                exchange: &mut *self,
                flight: &mut flight,
                written: vec![false; nlevels],
                layout,
            };
            let result = work(&mut in_flight, multipoles);
            let MultipolesInFlight {
                written, layout, ..
            } = in_flight;
            // What the work did not wait for reaches the receive buffers only.
            for (level, _) in written.iter().enumerate().filter(|(_, w)| !**w) {
                flight.receive(level, &mut self.levels[level].receive_buffer);
            }
            let written = layout && written.iter().all(|&w| w);
            (result, flight.finish(), written)
        });
        for ((level, send), staging) in self.levels.iter_mut().zip(sends).zip(staging) {
            level.send_buffer = send;
            level.staging = staging;
        }
        assert!(
            layout,
            "the multipole buffers do not have the exchange layout"
        );
        assert!(
            written,
            "every level must be waited for, with buffers of the exchange layout"
        );
        (result, times)
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

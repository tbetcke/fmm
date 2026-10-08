//! Redistribution of items to the ranks that own their leaves, and of values back.
//!
//! An FMM's caller holds its points wherever it likes, but the [`Evaluator`] needs each
//! point on the rank that owns its leaf, grouped by leaf. A [`Redistribution`] routes
//! items (points) by their finest-level Morton keys to those ranks once, and then moves
//! opaque per-item values along that route: [`forward`](Redistribution::forward) from
//! the caller's order to the owners, grouped by leaf, and
//! [`backward`](Redistribution::backward) from the owners to the caller's order
//! (`docs/design/distributed-fmm.md` §4, redesign §9). It knows nothing of kernels or
//! charges: a payload is `per_item` values of any MPI type per item.
//!
//! [`Evaluator`]: crate::evaluator::Evaluator
//!
//! # What moves, and when
//!
//! - [`Redistribution::new`] finds the owner of every item's key with
//!   [`Octree::owner_rank`], without communication, and sends each owner the key and
//!   the position of every item it owns. The owner finds the item's local leaf with
//!   [`Octree::local_leaf`] and the leaf's index in the [`Plan`].
//! - [`forward`](Redistribution::forward) moves `per_item` values of each item to its
//!   owner. They arrive grouped by local leaf, in leaf order, and within a leaf by
//!   (origin rank, origin position): the **received order**, in which
//!   [`counts`](Redistribution::counts) and [`origins`](Redistribution::origins) are
//!   given.
//! - [`backward`](Redistribution::backward) moves `per_item` values of each received
//!   item, given in received order, back to its origin, and returns them in the order of
//!   the keys passed to `new`.
//!
//! Items that stay on their rank are copied by the all-to-all-v, not sent.
//!
//! # Collectives
//!
//! Every rank enters every collective, a rank with no items with empty slices.
//!
//! | Call | Collectives, in order |
//! | --- | --- |
//! | [`Redistribution::new`] | all-to-all (item counts); all-reduce (errors and [`max_per_item`](Redistribution::max_per_item)); communicator duplicate; all-to-all-v (key and position of every item) |
//! | [`forward`](Redistribution::forward), [`forward_into`](Redistribution::forward_into) | one all-to-all-v |
//! | [`backward`](Redistribution::backward), [`backward_into`](Redistribution::backward_into) | one all-to-all-v |
//!
//! `new` agrees every error before the first collective that depends on the input, so
//! it fails on every rank or on none. `forward` and `backward` check `per_item` against
//! [`max_per_item`](Redistribution::max_per_item), which every rank holds, and panic
//! alike on every rank before any communication if it is too large. A rank that passes
//! a slice of the wrong length takes part in the exchange with default values and then
//! panics, so the other ranks are not stranded (the pattern of
//! [`exchange`](super::exchange)).
//!
//! # Determinism
//!
//! The received order, and with it every value `forward` returns, is a function of the
//! octree and of every rank's keys in their order; it does not depend on timing. For
//! fixed keys on fixed ranks, two calls of `forward` or `backward` return the same
//! values, bit for bit. A different distribution of the same items over the ranks
//! changes the order within a leaf (it is the order of the origins), and with the octree
//! the leaves themselves may change. On one rank the received order is the caller's
//! order, stably grouped by leaf.
//!
//! # Memory and traffic
//!
//! Per item, `new` keeps at most 6 bytes on its origin (the cycles of the permutation
//! into the send buffer) and at most 14 bytes on its owner (origin rank and position,
//! and the cycles of the permutation into the received order), plus O(P) counts for P
//! ranks; it sends 12 bytes per item (key and position). `forward` and `backward` send
//! `per_item` values per item, allocate one send buffer of the values they send (and
//! `forward` and `backward` the returned vector, the `_into` variants nothing else), and
//! permute in place.
//!
//! # Example
//!
//! ```no_run
//! use mpi::traits::Communicator;
//! use nd_fmm_plan::{plan::Plan, redistribute::Redistribution};
//! use nd_octree::{Octree, OctreeOptions, constants::DEEPEST_LEVEL, morton};
//!
//! let universe = mpi::initialize().expect("MPI is initialised once");
//! let comm = universe.world();
//! let rank = comm.rank() as usize;
//! // Finest-level keys of this rank's points, anywhere in the domain.
//! let keys: Vec<u64> = (0..100)
//!     .map(|i| {
//!         let cell = |d: usize| (37 * i + 11 * d + 5 * rank) % (1 << DEEPEST_LEVEL);
//!         morton::from_index_and_level([cell(0), cell(1), cell(2)], DEEPEST_LEVEL as usize)
//!     })
//!     .collect();
//! let options = OctreeOptions::new().with_ghost_children(true);
//! let octree = Octree::new(&keys, options, &comm);
//! let plan = Plan::new(&octree).expect("a plan of the octree");
//! let redistribution = Redistribution::new(&octree, &plan, &keys).expect("valid keys");
//!
//! // Three coordinates per point to the owners, grouped by leaf; `counts()` gives the
//! // points per local leaf, for `Evaluator::new`.
//! let coordinates = vec![0.5f64; 3 * keys.len()];
//! let owned = redistribution.forward(&coordinates, 3);
//! assert_eq!(owned.len(), 3 * redistribution.nreceived());
//! assert_eq!(redistribution.counts().iter().sum::<usize>(), redistribution.nreceived());
//!
//! // One result per owned point back to its origin, in the order of `keys`.
//! let results = vec![1.0f64; redistribution.nreceived()];
//! let returned = redistribution.backward(&results, 1);
//! assert_eq!(returned.len(), keys.len());
//! ```

#[cfg(test)]
#[path = "redistribute_tests.rs"]
mod tests;

use std::{cell::RefCell, error::Error, fmt};

use mpi::{
    collective::SystemOperation,
    datatype::{Partition, PartitionMut},
    topology::SimpleCommunicator,
    traits::{CommunicatorCollectives, Equivalence},
};
use nd_octree::{MortonKey, Octree};

use super::plan::Plan;

/// The reasons a [`Redistribution`] cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RedistributionError {
    /// The key of item `index` is not a valid Morton key on the finest level.
    InvalidKey {
        /// The position of the item in the keys of this rank.
        index: usize,
    },
    /// The local leaves of the plan are not the octree's: the plan was built from
    /// another octree.
    PlanMismatch,
    /// The items sent or received by some rank do not fit MPI's `i32` counts.
    Overflow,
    /// Building failed on another rank.
    OtherRank,
}

impl fmt::Display for RedistributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKey { index } => {
                write!(
                    f,
                    "the key of item {index} is not a finest-level Morton key"
                )
            }
            Self::PlanMismatch => {
                write!(
                    f,
                    "the plan was not built from the octree of the redistribution"
                )
            }
            Self::Overflow => write!(f, "the items of some rank overflow MPI's i32 counts"),
            Self::OtherRank => write!(f, "building the redistribution failed on another rank"),
        }
    }
}

impl Error for RedistributionError {}

/// Routing of items (points) to the ranks that own their leaves, and back (C5.1).
///
/// Received items are grouped by local leaf, in leaf order, and within a leaf ordered
/// by (origin rank, position on the origin). See the [module documentation](self).
///
/// The redistribution keeps a duplicate of the octree's communicator, and is not
/// `Sync`: its calls are collective, from one thread per rank.
pub struct Redistribution {
    comm: SimpleCommunicator,
    sender: Sender,
    receiver: Receiver,
    max_per_item: usize,
    /// Scratch for the value counts and displacements of one all-to-all-v: four blocks
    /// of P, so that `forward` and `backward` allocate no counts.
    scratch: RefCell<Vec<i32>>,
}

impl fmt::Debug for Redistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Redistribution")
            .field("nsent", &self.nsent())
            .field("nreceived", &self.nreceived())
            .field("max_per_item", &self.max_per_item)
            .finish_non_exhaustive()
    }
}

/// The key and position of an item, as `new` sends them to the owner.
#[derive(Clone, Copy, Debug, Default, Equivalence)]
struct Routed {
    key: MortonKey,
    position: u32,
}

impl Redistribution {
    /// Route the items with the finest-level `keys` to the owners of their leaves.
    ///
    /// `plan` must be the plan of `octree` ([`Plan::new`]); the received items are
    /// grouped by its local leaves.
    ///
    /// # Collective operation
    /// On the octree's communicator, every rank: one all-to-all of the item counts, one
    /// all-reduce that agrees the errors and [`max_per_item`](Self::max_per_item), a
    /// communicator duplicate, and one all-to-all-v of the key and position of every item
    /// (12 bytes per item). A rank with no items passes an empty slice.
    ///
    /// # Errors
    /// Returned on every rank if, on any rank, a key is not a valid finest-level key
    /// ([`RedistributionError::InvalidKey`], the first such item), the plan's local leaves
    /// are not the octree's ([`RedistributionError::PlanMismatch`]), or a rank sends or
    /// receives 2³¹ items or more ([`RedistributionError::Overflow`]). The rank that found
    /// a defect returns it; the other ranks return [`RedistributionError::OtherRank`].
    /// The agreement comes before the exchange of the keys, so nothing that depends on
    /// invalid input is communicated.
    pub fn new<C: CommunicatorCollectives>(
        octree: &Octree<'_, C>,
        plan: &Plan,
        keys: &[MortonKey],
    ) -> Result<Self, RedistributionError> {
        let comm = octree.comm();
        let nranks = comm.size() as usize;
        let index = plan.index();
        let nlocal = index.leaves().nlocal();

        // The owners, without communication; a rank with an invalid key sends nothing.
        let mut local = Ok(());
        let mut destinations = Vec::with_capacity(keys.len());
        for (position, &key) in keys.iter().enumerate() {
            match octree.owner_rank(key) {
                Ok(owner) => destinations.push(owner as u32),
                Err(_) => {
                    local = Err(RedistributionError::InvalidKey { index: position });
                    break;
                }
            }
        }
        let leaves = octree.leaf_keys();
        let mismatch = leaves.len() != nlocal
            || leaves
                .iter()
                .any(|&leaf| index.find_leaf(leaf).is_none_or(|j| j as usize >= nlocal));
        if local.is_ok() && mismatch {
            local = Err(RedistributionError::PlanMismatch);
        }
        let sender = if local.is_ok() {
            Sender::new(&destinations, nranks)
        } else {
            Sender::new(&[], nranks)
        };

        let mut receive_counts = vec![0usize; nranks];
        comm.all_to_all_into(&sender.rank_counts[..], &mut receive_counts[..]);
        let nreceived = receive_counts
            .iter()
            .try_fold(0usize, |total, &count| total.checked_add(count));
        let local_max = nreceived.map_or(0, |nreceived| max_per_item(keys.len(), nreceived));
        // The maximum of [failed, −max_per_item] gives "any rank failed" and the minimum.
        let mine = [i64::from(local.is_err()), -(local_max as i64)];
        let mut all = [0i64; 2];
        comm.all_reduce_into(&mine[..], &mut all[..], SystemOperation::max());
        local?;
        if all[0] != 0 {
            return Err(RedistributionError::OtherRank);
        }
        let max_per_item = (-all[1]) as usize;
        if max_per_item == 0 {
            return Err(RedistributionError::Overflow);
        }

        // Every count and every position now fits `i32`.
        let comm = comm.duplicate();
        let mut scratch = vec![0i32; 4 * nranks];
        let mut routed: Vec<Routed> = keys
            .iter()
            .enumerate()
            .map(|(position, &key)| Routed {
                key,
                position: position as u32,
            })
            .collect();
        sender.pack(&mut routed, 1);
        let mut arrived = vec![Routed::default(); nreceived.unwrap_or(0)];
        all_to_all_v(
            &comm,
            &mut scratch,
            (&routed, &sender.rank_counts),
            (&mut arrived, &receive_counts),
            1,
        );
        drop(routed);

        let positions: Vec<u32> = arrived.iter().map(|item| item.position).collect();
        let arrival_leaves: Vec<u32> = arrived
            .iter()
            .map(|item| {
                let leaf = octree
                    .local_leaf(item.key)
                    .expect("a received key is a valid finest-level key")
                    .expect("every rank routes by the same partition bounds");
                index
                    .find_leaf(leaf)
                    .expect("every leaf of the octree is a local leaf of the plan")
            })
            .collect();
        drop(arrived);
        let receiver = Receiver::new(receive_counts, &positions, &arrival_leaves, nlocal);

        Ok(Self {
            comm,
            sender,
            receiver,
            max_per_item,
            scratch: RefCell::new(scratch),
        })
    }

    /// Return the number of items this rank passed to [`new`](Self::new).
    pub fn nsent(&self) -> usize {
        self.sender.len()
    }

    /// Return the number of items this rank owns, the items it receives.
    pub fn nreceived(&self) -> usize {
        self.receiver.origins.len()
    }

    /// Return the number of received items of every local leaf, in leaf order: the
    /// counts for [`Evaluator::new`](crate::evaluator::Evaluator::new). The items of leaf
    /// j are the received items `counts[..j].sum()..counts[..=j].sum()`.
    pub fn counts(&self) -> &[usize] {
        &self.receiver.leaf_counts
    }

    /// Return the origin (rank, position in the keys passed to [`new`](Self::new) on
    /// that rank) of every received item, in received order.
    pub fn origins(&self) -> &[(u32, u32)] {
        &self.receiver.origins
    }

    /// Return the largest `per_item` that [`forward`](Self::forward) and
    /// [`backward`](Self::backward) accept: the largest for which every count and
    /// displacement of their exchanges, on every rank, fits MPI's `i32`. The same on
    /// every rank.
    pub fn max_per_item(&self) -> usize {
        self.max_per_item
    }

    /// Move `per_item` values of each item to its owner, and return them in received
    /// order.
    ///
    /// `payload` holds `nsent() · per_item` values, those of item i at
    /// `i · per_item..(i + 1) · per_item`, in the order of the keys passed to
    /// [`new`](Self::new); the result holds `nreceived() · per_item` values, those of
    /// received item r at `r · per_item`.
    ///
    /// # Collective operation
    /// One all-to-all-v, on every rank, with the same `per_item` on every rank.
    ///
    /// # Panics
    /// Panics on every rank, before communicating, if `per_item` exceeds
    /// [`max_per_item`](Self::max_per_item); and after taking part in the exchange if
    /// `payload` has the wrong length.
    pub fn forward<T: Equivalence + Copy + Default>(
        &self,
        payload: &[T],
        per_item: usize,
    ) -> Vec<T> {
        self.check_per_item(per_item);
        let mut received = vec![T::default(); self.receiver.len() * per_item];
        self.forward_into(payload, per_item, &mut received);
        received
    }

    /// [`forward`](Self::forward) into `received`, which must hold
    /// `nreceived() · per_item` values. Allocates only the send buffer.
    ///
    /// # Collective operation
    /// As [`forward`](Self::forward).
    ///
    /// # Panics
    /// As [`forward`](Self::forward), and after taking part in the exchange if
    /// `received` has the wrong length.
    pub fn forward_into<T: Equivalence + Copy + Default>(
        &self,
        payload: &[T],
        per_item: usize,
        received: &mut [T],
    ) {
        self.check_per_item(per_item);
        let nsend = self.sender.len() * per_item;
        let nreceive = self.receiver.len() * per_item;
        let lengths = payload.len() == nsend && received.len() == nreceive;
        let mut send = if lengths {
            payload.to_vec()
        } else {
            vec![T::default(); nsend]
        };
        self.sender.pack(&mut send, per_item);
        let mut scratch = self.scratch.borrow_mut();
        let counts = (&self.sender.rank_counts, &self.receiver.rank_counts);
        if lengths {
            all_to_all_v(
                &self.comm,
                &mut scratch,
                (&send, counts.0),
                (received, counts.1),
                per_item,
            );
            self.receiver.unpack(received, per_item);
        } else {
            let mut ignored = vec![T::default(); nreceive];
            all_to_all_v(
                &self.comm,
                &mut scratch,
                (&send, counts.0),
                (&mut ignored, counts.1),
                per_item,
            );
        }
        assert!(
            lengths,
            "forward: {} payload values for {nsend} and {} received values for {nreceive}",
            payload.len(),
            received.len()
        );
    }

    /// Move `per_item` values of each received item back to its origin, and return them
    /// in the order of the keys passed to [`new`](Self::new).
    ///
    /// `results` holds `nreceived() · per_item` values in received order; the result
    /// holds `nsent() · per_item` values, those of item i at `i · per_item`.
    ///
    /// # Collective operation
    /// One all-to-all-v, on every rank, with the same `per_item` on every rank.
    ///
    /// # Panics
    /// Panics on every rank, before communicating, if `per_item` exceeds
    /// [`max_per_item`](Self::max_per_item); and after taking part in the exchange if
    /// `results` has the wrong length.
    pub fn backward<T: Equivalence + Copy + Default>(
        &self,
        results: &[T],
        per_item: usize,
    ) -> Vec<T> {
        self.check_per_item(per_item);
        let mut returned = vec![T::default(); self.sender.len() * per_item];
        self.backward_into(results, per_item, &mut returned);
        returned
    }

    /// [`backward`](Self::backward) into `returned`, which must hold `nsent() · per_item`
    /// values. Allocates only the send buffer.
    ///
    /// # Collective operation
    /// As [`backward`](Self::backward).
    ///
    /// # Panics
    /// As [`backward`](Self::backward), and after taking part in the exchange if
    /// `returned` has the wrong length.
    pub fn backward_into<T: Equivalence + Copy + Default>(
        &self,
        results: &[T],
        per_item: usize,
        returned: &mut [T],
    ) {
        self.check_per_item(per_item);
        let nsend = self.receiver.len() * per_item;
        let nreceive = self.sender.len() * per_item;
        let lengths = results.len() == nsend && returned.len() == nreceive;
        let mut send = if lengths {
            results.to_vec()
        } else {
            vec![T::default(); nsend]
        };
        self.receiver.pack(&mut send, per_item);
        let mut scratch = self.scratch.borrow_mut();
        let counts = (&self.receiver.rank_counts, &self.sender.rank_counts);
        if lengths {
            all_to_all_v(
                &self.comm,
                &mut scratch,
                (&send, counts.0),
                (returned, counts.1),
                per_item,
            );
            self.sender.unpack(returned, per_item);
        } else {
            let mut ignored = vec![T::default(); nreceive];
            all_to_all_v(
                &self.comm,
                &mut scratch,
                (&send, counts.0),
                (&mut ignored, counts.1),
                per_item,
            );
        }
        assert!(
            lengths,
            "backward: {} result values for {nsend} and {} returned values for {nreceive}",
            results.len(),
            returned.len()
        );
    }

    /// Panic, alike on every rank, if `per_item` exceeds the agreed maximum.
    fn check_per_item(&self, per_item: usize) {
        assert!(
            per_item <= self.max_per_item,
            "{per_item} values per item exceed the {} that fit MPI's i32 counts",
            self.max_per_item
        );
    }
}

/// The largest number of values per item for which `nsent` items sent and `nreceived`
/// items received fit MPI's `i32` counts and displacements; zero if even one does not.
fn max_per_item(nsent: usize, nreceived: usize) -> usize {
    i32::MAX as usize / nsent.max(nreceived).max(1)
}

/// Write the value counts and displacements of `per_item` values per item, for the item
/// `counts`, into `values` and `displacements`.
///
/// The caller guarantees that the total fits `i32` (`per_item` at most
/// [`max_per_item`]).
fn value_counts(counts: &[usize], per_item: usize, values: &mut [i32], displacements: &mut [i32]) {
    let mut total = 0usize;
    for ((&count, value), displacement) in counts.iter().zip(values).zip(displacements) {
        *displacement = total as i32;
        *value = (count * per_item) as i32;
        total += count * per_item;
    }
}

/// One all-to-all-v of `per_item` values per item, with the item counts of every rank
/// on both sides; `scratch` holds 4 P counts.
fn all_to_all_v<T: Equivalence>(
    comm: &SimpleCommunicator,
    scratch: &mut [i32],
    (send, send_counts): (&[T], &[usize]),
    (receive, receive_counts): (&mut [T], &[usize]),
    per_item: usize,
) {
    let nranks = send_counts.len();
    let (send_values, rest) = scratch.split_at_mut(nranks);
    let (send_displacements, rest) = rest.split_at_mut(nranks);
    let (receive_values, receive_displacements) = rest.split_at_mut(nranks);
    value_counts(send_counts, per_item, send_values, send_displacements);
    value_counts(
        receive_counts,
        per_item,
        receive_values,
        receive_displacements,
    );
    comm.all_to_all_varcount_into(
        &Partition::new(send, &send_values[..], &send_displacements[..]),
        &mut PartitionMut::new(receive, &receive_values[..], &receive_displacements[..]),
    );
}

/// Stably sort elements into buckets: return the number of elements of each of the
/// `nbuckets` buckets, and the slot of every element when the buckets are laid out one
/// after the other, each in element order.
fn bucket(of: &[u32], nbuckets: usize) -> (Vec<usize>, Vec<u32>) {
    let mut counts = vec![0usize; nbuckets];
    for &b in of {
        counts[b as usize] += 1;
    }
    let mut next: Vec<usize> = counts
        .iter()
        .scan(0, |start, &count| {
            let first = *start;
            *start += count;
            Some(first)
        })
        .collect();
    let slots = of
        .iter()
        .map(|&b| {
            let slot = next[b as usize];
            next[b as usize] += 1;
            slot as u32
        })
        .collect();
    (counts, slots)
}

/// A permutation σ of chunks, as its cycles, to apply in place: element a moves to σ(a).
///
/// Fixed points are left out, so a permutation that moves nothing costs nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Cycles {
    /// The cycles one after the other; a cycle (c₀, c₁, …) has σ(cⱼ) = cⱼ₊₁, and σ of its
    /// last element is c₀.
    elements: Vec<u32>,
    /// `ncycles + 1` offsets into `elements`.
    starts: Vec<u32>,
}

impl Cycles {
    /// The cycles of σ(a) = `slots[a]`, a permutation of `0..slots.len()`.
    fn new(slots: &[u32]) -> Self {
        let mut visited = vec![false; slots.len()];
        let mut cycles = Self {
            elements: Vec::new(),
            starts: vec![0],
        };
        for first in 0..slots.len() {
            if visited[first] || slots[first] as usize == first {
                continue;
            }
            let mut a = first;
            while !visited[a] {
                visited[a] = true;
                cycles.elements.push(a as u32);
                a = slots[a] as usize;
            }
            debug_assert_eq!(a, first, "slots must be a permutation");
            cycles.starts.push(cycles.elements.len() as u32);
        }
        cycles
    }

    /// Move chunk a of `values` (chunks of `size` values) to chunk σ(a), for every a.
    fn apply<T>(&self, values: &mut [T], size: usize) {
        for cycle in self.starts.windows(2) {
            let cycle = &self.elements[cycle[0] as usize..cycle[1] as usize];
            for &c in &cycle[1..] {
                swap_chunks(values, size, cycle[0] as usize, c as usize);
            }
        }
    }

    /// Move chunk σ(a) of `values` to chunk a, for every a: the inverse of
    /// [`apply`](Self::apply).
    fn apply_inverse<T>(&self, values: &mut [T], size: usize) {
        for cycle in self.starts.windows(2) {
            let cycle = &self.elements[cycle[0] as usize..cycle[1] as usize];
            for &c in cycle[1..].iter().rev() {
                swap_chunks(values, size, cycle[0] as usize, c as usize);
            }
        }
    }
}

/// Swap chunks `a` and `b` (a ≠ b) of `values`, chunks of `size` values.
fn swap_chunks<T>(values: &mut [T], size: usize, a: usize, b: usize) {
    let (low, high) = (a.min(b), a.max(b));
    let (head, tail) = values.split_at_mut(high * size);
    head[low * size..(low + 1) * size].swap_with_slice(&mut tail[..size]);
}

/// The origin's side of a redistribution: items in caller order to the send buffer,
/// grouped by destination rank, each group in caller order.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Sender {
    /// Items sent to each rank, this rank included.
    rank_counts: Vec<usize>,
    /// Item i moves to its slot in the send buffer.
    cycles: Cycles,
}

impl Sender {
    /// The sender of items with the destination ranks `destinations`, on `nranks` ranks.
    fn new(destinations: &[u32], nranks: usize) -> Self {
        let (rank_counts, slots) = bucket(destinations, nranks);
        Self {
            rank_counts,
            cycles: Cycles::new(&slots),
        }
    }

    /// The number of items.
    fn len(&self) -> usize {
        self.rank_counts.iter().sum()
    }

    /// Reorder per-item chunks from caller order to send order, in place.
    fn pack<T>(&self, values: &mut [T], per_item: usize) {
        self.cycles.apply(values, per_item);
    }

    /// Reorder per-item chunks from send order to caller order, in place.
    fn unpack<T>(&self, values: &mut [T], per_item: usize) {
        self.cycles.apply_inverse(values, per_item);
    }
}

/// The owner's side of a redistribution: items in arrival order (by origin rank, each
/// origin's items in their positions' order) to the received order (by local leaf, then
/// in arrival order).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Receiver {
    /// Items received from each rank, this rank included.
    rank_counts: Vec<usize>,
    /// Received items per local leaf.
    leaf_counts: Vec<usize>,
    /// Origin (rank, position) of every item, in received order.
    origins: Vec<(u32, u32)>,
    /// Arrival a moves to its slot in received order.
    cycles: Cycles,
}

impl Receiver {
    /// The receiver of `rank_counts[r]` items from each rank r, arriving in rank order,
    /// with the origin `positions` and local `leaves` (of `nleaves`) of the arrivals.
    fn new(rank_counts: Vec<usize>, positions: &[u32], leaves: &[u32], nleaves: usize) -> Self {
        debug_assert_eq!(rank_counts.iter().sum::<usize>(), positions.len());
        debug_assert_eq!(positions.len(), leaves.len());
        let (leaf_counts, slots) = bucket(leaves, nleaves);
        let mut origins = vec![(0, 0); positions.len()];
        let ranks = rank_counts
            .iter()
            .enumerate()
            .flat_map(|(rank, &count)| std::iter::repeat_n(rank as u32, count));
        for ((rank, &position), &slot) in ranks.zip(positions).zip(&slots) {
            origins[slot as usize] = (rank, position);
        }
        Self {
            rank_counts,
            leaf_counts,
            origins,
            cycles: Cycles::new(&slots),
        }
    }

    /// The number of items.
    fn len(&self) -> usize {
        self.origins.len()
    }

    /// Reorder per-item chunks from arrival order to received order, in place.
    fn unpack<T>(&self, values: &mut [T], per_item: usize) {
        self.cycles.apply(values, per_item);
    }

    /// Reorder per-item chunks from received order to arrival order, in place.
    fn pack<T>(&self, values: &mut [T], per_item: usize) {
        self.cycles.apply_inverse(values, per_item);
    }
}

//! The forwarding tree of the multipole exchange (Phase 5S T8s, S8;
//! `fmm-validate/results/phase5s-t8s.md` §1, signed off on 2026-10-14).
//!
//! The direct exchange sends each multipole from the rank that computes it to every rank
//! that reads it. With [`MultipoleRouting::Tree`](super::MultipoleRouting::Tree) a box
//! with more than `fan_out` readers goes along a tree of its readers instead, fixed at
//! build by its owner ([`level_trees`]): the owner sends it to at most `fan_out` *heads*,
//! and each head forwards the copy it received to at most `fan_out` further readers, stage
//! after stage. A box with at most `fan_out` readers goes direct, at stage 0. Every reader
//! receives each box once, a copy of the owner's chunk; nothing is summed.
//!
//! **The build** (collective on the plan's communicator; [`TreeExchange::new`]): a
//! duplicate of the communicator; one sparse exchange of requests (every rank sends each
//! owner its node id and the keys it reads from it, every level); the owners form the
//! trees; one sparse exchange of instructions (to every reader, per requested key in
//! request order: its sender, its stage, and the ranks it forwards the box to). Each rank
//! then lays out its messages per level and stage. No P-long array is formed beyond the
//! counts that rlst's `sparse_all_to_allv` takes.
//!
//! **The messages** go point to point on the duplicate, one per (sender, receiver, level,
//! stage), with the tag `32 stage + level` (a rank may post a stage-2 forward before a
//! stage-1 one, so the stage is in the tag rather than left to MPI's ordering). Each
//! message carries its boxes in ascending key order, which sender and receiver both know
//! from the instructions. Stage-0 messages read the store (gathered by `send_boxes`, as
//! the direct exchange does); a forward reads the chunks its rank received at the stage
//! before, from the staging buffer every receive lands in, never a store.
//!
//! **Progress.** A forward is posted as soon as every message it reads has arrived: in
//! the waits, which poll every receive of the level until all have arrived, and in each
//! `test` of the overlapped path. A stage-s message
//! depends only on stage-(s − 1) messages of its level, and every rank waits for the
//! levels in the same order and for the stages of a level in order, so no rank waits for a
//! forward that another rank holds back.

#[cfg(test)]
#[path = "exchange_tree_tests.rs"]
mod tests;

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use mpi::{
    request::LocalScope,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Destination, Equivalence, Source},
};
use nd_octree::MortonKey;
use rlst::distributed_tools::{GhostCommunicator, sparse_all_to_allv};

use super::{ExchangeError, ExchangeTimes, InFlight, Pending, Traffic};
use crate::plan::Plan;
use crate::store::LevelBuffers;

/// The most stages a tree may have: the stage is part of the tag (`32 stage + level`), and
/// MPI guarantees tags up to 32,767.
const MAX_STAGES: usize = 1024;

/// The tag of the messages of `level` at `stage`.
fn tag(level: usize, stage: usize) -> i32 {
    (32 * stage + level) as i32
}

/// One edge of a forwarding tree: (sender, receiver, stage).
pub(super) type Edge = (usize, usize, usize);

/// The parts of `list`, readers ordered by (node, rank), that a sender on `sender_node`
/// sends to, one head each: at most `fan_out` of them.
///
/// - Readers on one node: `fan_out` runs of near-equal size if it is the sender's node,
///   else one part (the node's head splits it).
/// - Readers on 2 to `fan_out` nodes: one part per node, and the parts left over to the
///   nodes with the most readers (a node gets one more while its share of `fan_out`,
///   readers × `fan_out` / total, reaches its parts); each node's readers cut into its
///   parts as runs.
/// - Readers on more than `fan_out` nodes: `fan_out` runs of whole nodes, balanced by
///   readers.
pub(super) fn split(
    list: &[usize],
    fan_out: usize,
    sender_node: usize,
    node: &dyn Fn(usize) -> usize,
) -> Vec<Vec<usize>> {
    let mut groups: Vec<&[usize]> = Vec::new();
    let mut start = 0;
    for i in 1..=list.len() {
        if i == list.len() || node(list[i]) != node(list[start]) {
            groups.push(&list[start..i]);
            start = i;
        }
    }
    let runs = |group: &[usize], k: usize| -> Vec<Vec<usize>> {
        let n = group.len();
        (0..k)
            .map(|i| group[i * n / k..(i + 1) * n / k].to_vec())
            .collect()
    };
    match groups.len() {
        0 => Vec::new(),
        1 if node(groups[0][0]) != sender_node => vec![groups[0].to_vec()],
        1 => runs(groups[0], fan_out.min(groups[0].len())),
        g if g <= fan_out => {
            let total = list.len();
            let mut slots = vec![1usize; g];
            let mut extra = fan_out - g;
            let mut order: Vec<usize> = (0..g).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(groups[i].len()));
            while extra > 0 {
                let mut moved = false;
                for &i in &order {
                    let n = groups[i].len();
                    if extra > 0 && slots[i] < n && n * fan_out >= slots[i] * total {
                        slots[i] += 1;
                        extra -= 1;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
            groups
                .iter()
                .zip(&slots)
                .flat_map(|(group, &k)| runs(group, k))
                .collect()
        }
        g => {
            let total = list.len();
            let mut parts: Vec<Vec<usize>> = Vec::with_capacity(fan_out);
            let mut current = Vec::new();
            let mut done = 0;
            for (i, group) in groups.iter().enumerate() {
                current.extend_from_slice(group);
                done += group.len();
                let groups_left = g - i - 1;
                let parts_left = fan_out - parts.len() - 1;
                if parts_left > 0
                    && (done * fan_out >= total * (parts.len() + 1) || groups_left == parts_left)
                {
                    parts.push(std::mem::take(&mut current));
                }
            }
            if !current.is_empty() {
                parts.push(current);
            }
            debug_assert!(parts.len() <= fan_out);
            parts
        }
    }
}

/// The forwarding trees of `owner`'s boxes of one level that have more than `fan_out`
/// readers (`readers[b]`, any order): for each box, its edges.
///
/// The readers of all these boxes, ordered by (node, rank), are split into the parts of
/// [`split`] as the owner sees them. A reader's priority is the number of these boxes it
/// reads (most first), then its rank. For each part holding readers of a box, the owner
/// sends the box to the box's reader of highest priority in the part (stage 0), which
/// forwards it to the box's other readers in the part: directly if they are at most
/// `fan_out` and all on its node, otherwise by the same split to the heads of the
/// sub-parts (stage 1), and so on. No rank sends a box to more than `fan_out` ranks.
pub(super) fn level_trees(
    owner: usize,
    node: &dyn Fn(usize) -> usize,
    readers: &[Vec<usize>],
    fan_out: usize,
) -> Vec<Vec<Edge>> {
    let mut count: HashMap<usize, usize> = HashMap::new();
    for box_readers in readers {
        for &r in box_readers {
            *count.entry(r).or_default() += 1;
        }
    }
    let by_place = |r: &usize| (node(*r), *r);
    let mut union: Vec<usize> = count.keys().copied().collect();
    union.sort_unstable_by_key(by_place);
    let parts = split(&union, fan_out, node(owner), node);
    let mut part_of: HashMap<usize, usize> = HashMap::new();
    for (j, part) in parts.iter().enumerate() {
        for &r in part {
            part_of.insert(r, j);
        }
    }
    let priority = |r: &usize| (std::cmp::Reverse(count[r]), *r);
    let head = |group: &[usize]| *group.iter().min_by_key(|r| priority(r)).unwrap();

    fn serve(
        sender: usize,
        rest: Vec<usize>,
        stage: usize,
        fan_out: usize,
        node: &dyn Fn(usize) -> usize,
        head: &dyn Fn(&[usize]) -> usize,
        edges: &mut Vec<Edge>,
    ) {
        if rest.len() <= fan_out && rest.iter().all(|&r| node(r) == node(sender)) {
            edges.extend(rest.iter().map(|&r| (sender, r, stage)));
            return;
        }
        for part in split(&rest, fan_out, node(sender), node) {
            let h = head(&part);
            edges.push((sender, h, stage));
            let others: Vec<usize> = part.into_iter().filter(|&r| r != h).collect();
            if !others.is_empty() {
                serve(h, others, stage + 1, fan_out, node, head, edges);
            }
        }
    }

    readers
        .iter()
        .map(|box_readers| {
            let mut sorted = box_readers.clone();
            sorted.sort_unstable_by_key(by_place);
            let mut by_part: Vec<Vec<usize>> = vec![Vec::new(); parts.len()];
            for r in sorted {
                by_part[part_of[&r]].push(r);
            }
            let mut edges = Vec::new();
            for group in by_part.into_iter().filter(|g| !g.is_empty()) {
                let h = head(&group);
                edges.push((owner, h, 0));
                let others: Vec<usize> = group.into_iter().filter(|&r| r != h).collect();
                if !others.is_empty() {
                    serve(h, others, 1, fan_out, node, &head, &mut edges);
                }
            }
            edges
        })
        .collect()
}

/// A receive message of one level.
#[derive(Clone, Debug, Default)]
struct Receive {
    peer: usize,
    stage: usize,
    /// The positions in the receive buffer (in boxes) of its chunks, in message order.
    slots: Vec<u32>,
}

/// A send message of one level.
#[derive(Clone, Debug, Default)]
struct Send {
    peer: usize,
    stage: usize,
    /// Stage 0: the first chunk in the send buffer (`send_boxes`). Later stages: the
    /// first chunk in the forward buffer.
    first: usize,
    chunks: usize,
    /// A forward's chunks: (receive message, chunk in it).
    sources: Vec<(usize, usize)>,
    /// A forward's receive messages, ascending.
    depends: Vec<usize>,
}

/// The messages of one level on this rank.
#[derive(Clone, Debug, Default)]
struct Routes {
    level: usize,
    size: usize,
    /// By (stage, peer).
    receives: Vec<Receive>,
    /// Stage 0 first, by peer; then the forwards by (stage, peer).
    sends: Vec<Send>,
    /// 1 + the last stage of a message of this rank, 0 without messages.
    stages: usize,
}

/// One level of the [`TreeExchange`].
pub(super) struct TreeLevel<T> {
    /// Moved out while an overlapped exchange is in flight.
    routes: Routes,
    level: usize,
    size: usize,
    pub(super) nlevels: usize,
    pub(super) len: usize,
    /// The boxes the stage-0 messages send from the store, message after message.
    pub(super) send_boxes: Vec<u32>,
    /// Every box received, strictly ascending.
    pub(super) receive_boxes: Vec<u32>,
    send_buffer: Vec<T>,
    forward_buffer: Vec<T>,
    staging: Vec<T>,
    pub(super) receive_buffer: Vec<T>,
    /// What the routes give, kept here since the routes are moved out in flight.
    traffic: Traffic,
    destinations: Vec<usize>,
    sources: Vec<usize>,
    depth: usize,
    fan_out: usize,
    forwarded: usize,
}

impl<T: Equivalence + Copy + Default> TreeLevel<T> {
    pub(super) fn size(&self) -> usize {
        self.size
    }

    /// Whether `multipoles` has the plan's number of levels, and on this level its number
    /// of boxes and this exchange's size.
    pub(super) fn has_layout(&self, multipoles: &LevelBuffers<T>) -> bool {
        multipoles.nlevels() == self.nlevels
            && multipoles.len(self.level) == self.len
            && multipoles.size(self.level) == self.size
    }

    fn pack(&mut self, multipoles: &LevelBuffers<T>) {
        let size = self.size;
        let source = multipoles.level(self.level);
        for (chunk, &i) in self
            .send_buffer
            .chunks_exact_mut(size)
            .zip(&self.send_boxes)
        {
            chunk.copy_from_slice(source.chunk(i as usize));
        }
    }

    pub(super) fn scatter(&self, multipoles: &mut LevelBuffers<T>) {
        let mut target = multipoles.level_mut(self.level);
        for (chunk, &i) in self
            .receive_buffer
            .chunks_exact(self.size)
            .zip(&self.receive_boxes)
        {
            target.chunk_mut(i as usize).copy_from_slice(chunk);
        }
    }

    pub(super) fn traffic(&self) -> Traffic {
        self.traffic
    }

    pub(super) fn destinations(&self) -> Vec<usize> {
        self.destinations.clone()
    }

    pub(super) fn sources(&self) -> Vec<usize> {
        self.sources.clone()
    }

    pub(super) fn depth(&self) -> usize {
        self.depth
    }

    pub(super) fn fan_out(&self) -> usize {
        self.fan_out
    }

    pub(super) fn forwarded(&self) -> usize {
        self.forwarded
    }
}

/// The multipole exchange along forwarding trees: every level's routes, on a duplicate of
/// the plan's communicator.
pub(super) struct TreeExchange<T> {
    /// The duplicate; moved out while an overlapped exchange is in flight.
    comm: Option<SimpleCommunicator>,
    pub(super) levels: Vec<TreeLevel<T>>,
}

/// A box of the plan: (level, box index).
type Slot = (usize, u32);

/// What a rank learns about one key it requested: from whom and at which stage it arrives,
/// and to whom it forwards it.
struct Instruction {
    sender: usize,
    stage: usize,
    forwards: Vec<usize>,
}

impl<T: Equivalence + Copy + Default> TreeExchange<T> {
    /// Build the routes for `ghosts[l]` (box index, owner), ascending per level, with
    /// `sizes[l]` values per multipole: the collectives of the module documentation. A
    /// key requested from this rank that is not a box whose multipole it computes, or a
    /// tree deeper than the tags allow, sets `layout` to the first error, to be agreed by
    /// the caller. Every rank enters every collective whatever it found.
    pub(super) fn new<C: CommunicatorCollectives>(
        plan: &Plan,
        comm: &C,
        sizes: &[usize],
        ghosts: &[Vec<(u32, usize)>],
        fan_out: usize,
        node: usize,
        layout: &mut Result<(), ExchangeError>,
    ) -> Self {
        let comm = comm.duplicate();
        let nranks = comm.size() as usize;
        let rank = comm.rank() as usize;
        let index = plan.index();
        let nlevels = index.nlevels();

        // Requests: to each owner, this rank's node and then the keys it reads from it,
        // level by level, ascending.
        let mut requested: Vec<(usize, usize, u32)> = ghosts
            .iter()
            .enumerate()
            .flat_map(|(level, g)| g.iter().map(move |&(i, owner)| (owner, level, i)))
            .collect();
        requested.sort_by_key(|&(owner, _, _)| owner);
        let mut counts = vec![0usize; nranks];
        let mut data = Vec::with_capacity(requested.len() + nranks.min(requested.len()));
        for owner_requests in requested.chunk_by(|a, b| a.0 == b.0) {
            let owner = owner_requests[0].0;
            counts[owner] = owner_requests.len() + 1;
            data.push(node as u64);
            data.extend(
                owner_requests
                    .iter()
                    .map(|&(_, level, i)| index.key(level, i as usize)),
            );
        }
        let (request_counts, request_data) = sparse_all_to_allv(&comm, &counts, &data);

        // The owner's side: the readers of each requested box, with their nodes.
        let mut nodes: HashMap<usize, usize> = HashMap::new();
        nodes.insert(rank, node);
        let mut readers: Vec<HashMap<u32, Vec<usize>>> = vec![HashMap::new(); nlevels];
        // Per requesting rank, its keys in request order as (level, box), or None if not
        // valid here.
        let mut asked: Vec<(usize, Vec<Option<Slot>>)> = Vec::new();
        let mut offset = 0;
        for (reader, &count) in request_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let chunk = &request_data[offset..offset + count];
            offset += count;
            nodes.insert(reader, chunk[0] as usize);
            let keys = chunk[1..]
                .iter()
                .map(|&key: &MortonKey| match index.find(key) {
                    Some((l, i)) if plan.computes(l, i as usize) => {
                        readers[l].entry(i).or_default().push(reader);
                        Some((l, i))
                    }
                    _ => {
                        if layout.is_ok() {
                            *layout = Err(ExchangeError::UnheldKey { key });
                        }
                        None
                    }
                })
                .collect();
            asked.push((reader, keys));
        }

        // The trees: per level, the boxes with more than `fan_out` readers by tree, the
        // others direct.
        let node_of = |r: usize| nodes.get(&r).copied().unwrap_or(r);
        let mut instructions: HashMap<(usize, u32, usize), Instruction> = HashMap::new();
        let mut own_sends: Vec<(usize, usize, u32)> = Vec::new(); // (level, receiver, box)
        for (level, level_readers) in readers.iter().enumerate() {
            let mut boxes: Vec<u32> = level_readers.keys().copied().collect();
            boxes.sort_unstable();
            let (tree_boxes, direct): (Vec<u32>, Vec<u32>) = boxes
                .into_iter()
                .partition(|i| level_readers[i].len() > fan_out);
            for i in direct {
                for &r in &level_readers[&i] {
                    own_sends.push((level, r, i));
                    instructions.insert(
                        (level, i, r),
                        Instruction {
                            sender: rank,
                            stage: 0,
                            forwards: Vec::new(),
                        },
                    );
                }
            }
            let tree_readers: Vec<Vec<usize>> = tree_boxes
                .iter()
                .map(|i| level_readers[i].clone())
                .collect();
            let trees = level_trees(rank, &node_of, &tree_readers, fan_out);
            for (&i, edges) in tree_boxes.iter().zip(trees) {
                for &(sender, receiver, stage) in &edges {
                    instructions.insert(
                        (level, i, receiver),
                        Instruction {
                            sender,
                            stage,
                            forwards: Vec::new(),
                        },
                    );
                    if sender == rank {
                        own_sends.push((level, receiver, i));
                    }
                }
                for &(sender, receiver, _) in &edges {
                    if sender != rank {
                        instructions
                            .get_mut(&(level, i, sender))
                            .expect("a forwarder is a reader")
                            .forwards
                            .push(receiver);
                    }
                }
            }
        }

        // Instructions: to each requesting rank, per key in request order, its sender,
        // stage, the number of ranks it forwards to and those ranks.
        let mut counts = vec![0usize; nranks];
        let mut data: Vec<u64> = Vec::new();
        for (reader, keys) in &asked {
            let start = data.len();
            for key in keys {
                match key.and_then(|(l, i)| instructions.get(&(l, i, *reader))) {
                    Some(instruction) => {
                        data.extend([
                            instruction.sender as u64,
                            instruction.stage as u64,
                            instruction.forwards.len() as u64,
                        ]);
                        data.extend(instruction.forwards.iter().map(|&r| r as u64));
                    }
                    // A key this rank does not compute: direct, never sent (the build
                    // fails on every rank once the layout is agreed).
                    None => data.extend([rank as u64, 0, 0]),
                }
            }
            counts[*reader] = data.len() - start;
        }
        let (instruction_counts, instruction_data) = sparse_all_to_allv(&comm, &counts, &data);

        // The reader's side: per level, the receives (stage, sender, box) and the forwards
        // (stage, receiver, box).
        let mut receives: Vec<Vec<(usize, usize, u32)>> = vec![Vec::new(); nlevels];
        let mut forwards: Vec<Vec<(usize, usize, u32)>> = vec![Vec::new(); nlevels];
        let mut starts = Vec::with_capacity(nranks);
        let mut total = 0;
        for &count in &instruction_counts {
            starts.push(total);
            total += count;
        }
        for owner_requests in requested.chunk_by(|a, b| a.0 == b.0) {
            let mut cursor = starts[owner_requests[0].0];
            for &(_, level, i) in owner_requests {
                let sender = instruction_data[cursor] as usize;
                let stage = instruction_data[cursor + 1] as usize;
                let nforwards = instruction_data[cursor + 2] as usize;
                receives[level].push((stage, sender, i));
                for &receiver in &instruction_data[cursor + 3..cursor + 3 + nforwards] {
                    forwards[level].push((stage + 1, receiver as usize, i));
                }
                cursor += 3 + nforwards;
            }
        }

        let mut levels = Vec::with_capacity(nlevels);
        for level in 0..nlevels {
            let size = sizes[level];
            let receive_boxes: Vec<u32> = ghosts[level].iter().map(|&(i, _)| i).collect();
            let slot = |i: u32| receive_boxes.binary_search(&i).unwrap() as u32;

            let mut routes = Routes {
                level,
                size,
                ..Routes::default()
            };
            let mut level_receives = std::mem::take(&mut receives[level]);
            level_receives.sort_unstable();
            // Where each received box lands: (message, chunk).
            let mut arrives: HashMap<u32, (usize, usize)> = HashMap::new();
            for message in level_receives.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
                let (stage, sender, _) = message[0];
                for (c, &(_, _, i)) in message.iter().enumerate() {
                    arrives.insert(i, (routes.receives.len(), c));
                }
                routes.receives.push(Receive {
                    peer: sender,
                    stage,
                    slots: message.iter().map(|&(_, _, i)| slot(i)).collect(),
                });
            }

            let mut send_boxes = Vec::new();
            let mut sent = own_sends
                .iter()
                .filter(|&&(l, _, _)| l == level)
                .map(|&(_, receiver, i)| (receiver, i))
                .collect::<Vec<_>>();
            sent.sort_unstable();
            for message in sent.chunk_by(|a, b| a.0 == b.0) {
                routes.sends.push(Send {
                    peer: message[0].0,
                    stage: 0,
                    first: send_boxes.len(),
                    chunks: message.len(),
                    ..Send::default()
                });
                send_boxes.extend(message.iter().map(|&(_, i)| i));
            }
            let mut level_forwards = std::mem::take(&mut forwards[level]);
            level_forwards.sort_unstable();
            let mut forwarded = 0;
            for message in level_forwards.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
                let (stage, receiver, _) = message[0];
                let sources: Vec<(usize, usize)> =
                    message.iter().map(|&(_, _, i)| arrives[&i]).collect();
                let mut depends: Vec<usize> = sources.iter().map(|&(m, _)| m).collect();
                depends.sort_unstable();
                depends.dedup();
                routes.sends.push(Send {
                    peer: receiver,
                    stage,
                    first: forwarded,
                    chunks: message.len(),
                    sources,
                    depends,
                });
                forwarded += message.len();
            }
            let last_stage = routes
                .receives
                .iter()
                .map(|r| r.stage)
                .chain(routes.sends.iter().map(|s| s.stage))
                .max();
            routes.stages = last_stage.map_or(0, |s| s + 1);
            if routes.stages > MAX_STAGES && layout.is_ok() {
                *layout = Err(ExchangeError::Overflow);
            }
            let mut copies: HashMap<u32, usize> = HashMap::new();
            for &i in &send_boxes {
                *copies.entry(i).or_default() += 1;
            }
            for &(_, _, i) in &level_forwards {
                *copies.entry(i).or_default() += 1;
            }
            let ranks = |peers: &mut dyn Iterator<Item = usize>| {
                let mut ranks: Vec<usize> = peers.collect();
                ranks.sort_unstable();
                ranks.dedup();
                ranks
            };
            let traffic = Traffic {
                messages_sent: routes.sends.len(),
                messages_received: routes.receives.len(),
                values_sent: routes.sends.iter().map(|m| m.chunks).sum::<usize>() * size,
                values_received: receive_boxes.len() * size,
            };
            levels.push(TreeLevel {
                traffic,
                destinations: ranks(&mut routes.sends.iter().map(|m| m.peer)),
                sources: ranks(&mut routes.receives.iter().map(|m| m.peer)),
                depth: routes.stages,
                level,
                size,
                nlevels,
                len: index.len(level),
                send_buffer: vec![T::default(); send_boxes.len() * size],
                forward_buffer: vec![T::default(); forwarded * size],
                staging: vec![T::default(); receive_boxes.len() * size],
                receive_buffer: vec![T::default(); receive_boxes.len() * size],
                fan_out: copies.values().copied().max().unwrap_or(0),
                forwarded,
                routes,
                send_boxes,
                receive_boxes,
            });
        }
        Self {
            comm: Some(comm),
            levels,
        }
    }

    /// The blocking exchange of `level`: pack, post every message of the level, wait for
    /// the receives, forwarding as the copies arrive, then write the receive buffer and
    /// the slots. Panics, after taking part, if `multipoles` does not have the layout.
    pub(super) fn forward(&mut self, level: usize, multipoles: &mut LevelBuffers<T>) {
        let exchange = &mut self.levels[level];
        let layout = exchange.has_layout(multipoles);
        if layout {
            exchange.pack(multipoles);
        }
        let routes = std::mem::take(&mut exchange.routes);
        let mut staging = std::mem::take(&mut exchange.staging);
        let mut forward = std::mem::take(&mut exchange.forward_buffer);
        let receive_buffer = &mut exchange.receive_buffer;
        let send_buffer = &exchange.send_buffer;
        let comm = self.comm.as_ref().expect("no exchange in flight");
        mpi::request::scope(|scope| {
            let mut flight = LevelFlight::post(
                scope,
                comm,
                &routes,
                send_buffer,
                &mut staging,
                &mut forward,
            );
            flight.wait_receives(scope, comm);
            flight.copy_into(receive_buffer);
            flight.wait_sends();
        });
        exchange.routes = routes;
        exchange.staging = staging;
        exchange.forward_buffer = forward;
        assert!(
            layout,
            "the multipole buffers do not have the exchange layout"
        );
        exchange.scatter(multipoles);
    }

    /// Pack every level's stage-0 sends from `multipoles` (if it has the layout) and move
    /// out what an overlapped exchange borrows while in flight: the communicator, the
    /// routes, the send, staging and forward buffers. [`restore`](Self::restore) puts them
    /// back.
    pub(super) fn take_parts(&mut self, multipoles: &LevelBuffers<T>, layout: bool) -> Parts<T> {
        if layout {
            for level in &mut self.levels {
                level.pack(multipoles);
            }
        }
        let mut parts = Parts {
            comm: self.comm.take().expect("no exchange in flight"),
            routes: Vec::with_capacity(self.levels.len()),
            sends: Vec::with_capacity(self.levels.len()),
            staging: Vec::with_capacity(self.levels.len()),
            forward: Vec::with_capacity(self.levels.len()),
        };
        for level in &mut self.levels {
            parts.routes.push(std::mem::take(&mut level.routes));
            parts.sends.push(std::mem::take(&mut level.send_buffer));
            parts.staging.push(std::mem::take(&mut level.staging));
            parts
                .forward
                .push(std::mem::take(&mut level.forward_buffer));
        }
        parts
    }

    /// Put back what [`take_parts`](Self::take_parts) moved out.
    pub(super) fn restore(&mut self, parts: Parts<T>) {
        let Parts {
            comm,
            routes,
            sends,
            staging,
            forward,
        } = parts;
        self.comm = Some(comm);
        for (level, (((routes, send), staging), forward)) in self
            .levels
            .iter_mut()
            .zip(routes.into_iter().zip(sends).zip(staging).zip(forward))
        {
            level.routes = routes;
            level.send_buffer = send;
            level.staging = staging;
            level.forward_buffer = forward;
        }
    }
}

/// What an overlapped tree exchange borrows while in flight
/// ([`TreeExchange::take_parts`]).
pub(super) struct Parts<T> {
    comm: SimpleCommunicator,
    routes: Vec<Routes>,
    sends: Vec<Vec<T>>,
    staging: Vec<Vec<T>>,
    forward: Vec<Vec<T>>,
}

impl<T: Equivalence + Copy> Parts<T> {
    /// Post every level: its receives and its stage-0 sends.
    pub(super) fn post<'a, 's>(&'a mut self, scope: &'s LocalScope<'a>) -> TreeFlight<'a, 's, T> {
        let posted = Instant::now();
        let Self {
            comm,
            routes,
            sends,
            staging,
            forward,
        } = self;
        let comm: &'a SimpleCommunicator = comm;
        let levels = routes
            .iter()
            .zip(sends.iter())
            .zip(staging.iter_mut().zip(forward.iter_mut()))
            .map(|((routes, send), (staging, forward))| {
                LevelFlight::post(scope, comm, routes, send, staging, forward)
            })
            .collect();
        TreeFlight {
            scope,
            comm,
            levels,
            posted,
            completed: None,
            times: ExchangeTimes::default(),
        }
    }
}

/// A receive in flight, or arrived.
enum Incoming<'a, 's, T> {
    Pending(Pending<'a, 's, T>),
    Arrived(&'a [T]),
    /// Only while being replaced.
    Taken,
}

/// A send not yet posted (a forward, with its buffer), in flight, or complete.
enum Outgoing<'a, 's, T> {
    Unposted(&'a mut [T]),
    Sending(Pending<'a, 's, T>),
    Done,
}

/// The messages of one level of a [`TreeExchange`], posted in one scope.
struct LevelFlight<'a, 's, T> {
    routes: &'a Routes,
    incoming: Vec<Incoming<'a, 's, T>>,
    outgoing: Vec<Outgoing<'a, 's, T>>,
}

impl<'a, 's, T: Equivalence + Copy> LevelFlight<'a, 's, T> {
    /// Post every receive of the level into `staging` and every stage-0 send from `send`;
    /// the forwards wait in their pieces of `forward`.
    fn post(
        scope: &'s LocalScope<'a>,
        comm: &SimpleCommunicator,
        routes: &'a Routes,
        send: &'a [T],
        staging: &'a mut [T],
        forward: &'a mut [T],
    ) -> Self {
        let size = routes.size;
        let mut rest = staging;
        let incoming = routes
            .receives
            .iter()
            .map(|receive| {
                let (piece, tail) =
                    std::mem::take(&mut rest).split_at_mut(receive.slots.len() * size);
                rest = tail;
                Incoming::Pending(
                    comm.process_at_rank(receive.peer as i32)
                        .immediate_receive_into_with_tag(
                            scope,
                            piece,
                            tag(routes.level, receive.stage),
                        ),
                )
            })
            .collect();
        let mut rest = forward;
        let outgoing = routes
            .sends
            .iter()
            .map(|message| {
                let len = message.chunks * size;
                if message.stage == 0 {
                    let values = &send[message.first * size..message.first * size + len];
                    Outgoing::Sending(
                        comm.process_at_rank(message.peer as i32)
                            .immediate_send_with_tag(scope, values, tag(routes.level, 0)),
                    )
                } else {
                    let (piece, tail) = std::mem::take(&mut rest).split_at_mut(len);
                    rest = tail;
                    Outgoing::Unposted(piece)
                }
            })
            .collect();
        Self {
            routes,
            incoming,
            outgoing,
        }
    }

    /// Post every forward whose receive messages have all arrived.
    fn post_ready(&mut self, scope: &'s LocalScope<'a>, comm: &SimpleCommunicator) {
        let size = self.routes.size;
        for (message, outgoing) in self.routes.sends.iter().zip(&mut self.outgoing) {
            if !matches!(outgoing, Outgoing::Unposted(_))
                || !message
                    .depends
                    .iter()
                    .all(|&m| matches!(self.incoming[m], Incoming::Arrived(_)))
            {
                continue;
            }
            let Outgoing::Unposted(piece) = std::mem::replace(outgoing, Outgoing::Done) else {
                unreachable!()
            };
            for (chunk, &(m, c)) in piece.chunks_exact_mut(size).zip(&message.sources) {
                let Incoming::Arrived(values) = self.incoming[m] else {
                    unreachable!()
                };
                chunk.copy_from_slice(&values[c * size..(c + 1) * size]);
            }
            let piece: &'a [T] = piece;
            *outgoing = Outgoing::Sending(
                comm.process_at_rank(message.peer as i32)
                    .immediate_send_with_tag(scope, piece, tag(self.routes.level, message.stage)),
            );
        }
    }

    /// One `MPI_Test` per message in flight, then post the forwards that became ready.
    fn test(&mut self, scope: &'s LocalScope<'a>, comm: &SimpleCommunicator) {
        for incoming in &mut self.incoming {
            if let Incoming::Pending(_) = incoming {
                let Incoming::Pending(request) = std::mem::replace(incoming, Incoming::Taken)
                else {
                    unreachable!()
                };
                *incoming = match request.test_with_data() {
                    Ok((_, values)) => Incoming::Arrived(values),
                    Err(request) => Incoming::Pending(request),
                };
            }
        }
        self.post_ready(scope, comm);
        for outgoing in &mut self.outgoing {
            if let Outgoing::Sending(_) = outgoing {
                let Outgoing::Sending(request) = std::mem::replace(outgoing, Outgoing::Done) else {
                    unreachable!()
                };
                *outgoing = match request.test() {
                    Ok(_) => Outgoing::Done,
                    Err(request) => Outgoing::Sending(request),
                };
            }
        }
    }

    /// Wait for every receive of the level, posting each forward as soon as the messages
    /// it resends have arrived: one `MPI_Test` per receive in flight per pass, until none
    /// is left. A forward waits for its own inputs only, never for the level's other
    /// messages (Phase 5S T8s, step 5: waiting stage by stage made the blocking stage up
    /// to 3.8 ms slower at 160 Kathleen ranks). Posting a forward once its inputs are in
    /// cannot deadlock: a stage-s message depends only on stage-(s − 1) messages.
    fn wait_receives(&mut self, scope: &'s LocalScope<'a>, comm: &SimpleCommunicator) {
        loop {
            let mut pending = false;
            for incoming in &mut self.incoming {
                if let Incoming::Pending(_) = incoming {
                    let Incoming::Pending(request) = std::mem::replace(incoming, Incoming::Taken)
                    else {
                        unreachable!()
                    };
                    *incoming = match request.test_with_data() {
                        Ok((_, values)) => Incoming::Arrived(values),
                        Err(request) => {
                            pending = true;
                            Incoming::Pending(request)
                        }
                    };
                }
            }
            self.post_ready(scope, comm);
            if !pending {
                break;
            }
        }
        debug_assert!(
            self.outgoing
                .iter()
                .all(|o| !matches!(o, Outgoing::Unposted(_)))
        );
    }

    /// Copy every received chunk into its slot of `into`, the receive buffer in box order.
    fn copy_into(&self, into: &mut [T]) {
        let size = self.routes.size;
        for (receive, incoming) in self.routes.receives.iter().zip(&self.incoming) {
            let Incoming::Arrived(values) = incoming else {
                panic!("copy before every receive arrived");
            };
            for (chunk, &slot) in values.chunks_exact(size).zip(&receive.slots) {
                let slot = slot as usize;
                into[slot * size..(slot + 1) * size].copy_from_slice(chunk);
            }
        }
    }

    /// Wait for every send; every forward is posted by then.
    fn wait_sends(&mut self) {
        for outgoing in &mut self.outgoing {
            match std::mem::replace(outgoing, Outgoing::Done) {
                Outgoing::Sending(request) => request.wait_without_status(),
                Outgoing::Unposted(_) => panic!("a forward was never posted"),
                Outgoing::Done => {}
            }
        }
    }

    /// Whether a message is still in flight or to be posted.
    fn pending(&self) -> bool {
        self.incoming
            .iter()
            .any(|i| matches!(i, Incoming::Pending(_)))
            || self.outgoing.iter().any(|o| !matches!(o, Outgoing::Done))
    }
}

/// Every level of a [`TreeExchange`] in flight, as [`InFlight`] for the overlapped path.
pub(super) struct TreeFlight<'a, 's, T> {
    scope: &'s LocalScope<'a>,
    comm: &'a SimpleCommunicator,
    levels: Vec<LevelFlight<'a, 's, T>>,
    posted: Instant,
    completed: Option<Instant>,
    times: ExchangeTimes,
}

impl<T: Equivalence + Copy> TreeFlight<'_, '_, T> {
    fn note(&mut self, now: Instant) {
        if self.completed.is_none() && self.levels.iter().all(|l| !l.pending()) {
            self.completed = Some(now);
        }
    }

    /// Wait for every send, and return the times.
    pub(super) fn finish(&mut self) -> ExchangeTimes {
        let start = Instant::now();
        for level in &mut self.levels {
            level.wait_sends();
        }
        let end = Instant::now();
        self.times.exposed += end - start;
        self.times.sends = end - start;
        self.note(end);
        self.times.total = self.completed.unwrap_or(end) - self.posted;
        self.times
    }
}

impl<T: Equivalence + Copy> InFlight<T> for TreeFlight<'_, '_, T> {
    fn test(&mut self) {
        if self.completed.is_some() {
            return;
        }
        let start = Instant::now();
        for level in &mut self.levels {
            level.test(self.scope, self.comm);
        }
        let end = Instant::now();
        self.times.progress += end - start;
        self.note(end);
    }

    fn receive(&mut self, group: usize, into: &mut [T]) -> Duration {
        let start = Instant::now();
        self.levels[group].wait_receives(self.scope, self.comm);
        let end = Instant::now();
        self.times.exposed += end - start;
        self.note(end);
        self.levels[group].copy_into(into);
        end - start
    }

    fn send(
        &mut self,
        _group: usize,
        _communicator: &GhostCommunicator<MortonKey>,
        _pack: &mut dyn FnMut(&mut [T]),
    ) -> bool {
        false
    }

    fn times(&self) -> ExchangeTimes {
        self.times
    }
}

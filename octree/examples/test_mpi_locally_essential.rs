//! The locally essential key set and the partition map (Phase 5S T3;
//! docs/design/scale-out.md §3.1–§3.3, §9.1).
//!
//! Every rank draws the same global point set from a seed and passes every P-th point.
//! For each workload (the uniform cube, the Plummer sphere, five Gaussian clusters, a
//! graded corner, a heavy block that leaves ranks without blocks, and the root as the
//! only block), with and without the ghost-children layer, it gathers the tree (every
//! leaf and every block with its owner) and checks on every rank, by brute force:
//! - the partition map: the blocks, the cut and the leaf flags are the gathered ones,
//!   and the owner, the sharers and the kind of every key of the tree (and of the first
//!   and last finest keys of every leaf) are those of the gathered tree;
//! - the held keys are exactly H = N(S) ∪ C(N(S)) (C only with the layer), S the own
//!   keys and the shared ancestors, with the kinds of the design: own leaves and
//!   interior keys local, shared ancestors `Global`, every other key a ghost of its
//!   owner; a rank without blocks holds nothing;
//! - every key that a U-, V-, W- or X-row of a key of S names (the geometric
//!   definitions over the gathered tree, with the layer), the halo of the own keys and
//!   every parent-child step from S is held;
//! - against the Phase 5 key set (`phase5_keys`, Phase 5's `generate_all_keys` on the
//!   same tree): H is a subset of it, every key both hold has the same kind except the
//!   ancestors of other ranks' blocks only, `Global` in Phase 5 and ghosts of their
//!   owner now, and on one rank both are the same map.
//!
//! Rank 0 prints the held keys per rank, Phase 5 against now (max and mean over the
//! ranks). Every check is agreed over all ranks before it is asserted, so a failure
//! stops every rank, not one.

#[path = "phase5_keys/mod.rs"]
mod phase5_keys;

use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use mpi::{
    collective::SystemOperation,
    traits::{Communicator, CommunicatorCollectives},
};
use nd_octree::{
    MortonKey, Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, morton,
    octree::KeyType, tools::seeded_rng,
};
use rand::RngExt;
use rlst::distributed_tools::array_tools::gather_to_all;

/// The number of points of each workload.
const NPOINTS: usize = 3000;

/// Assert `ok` on every rank once every rank has evaluated it.
///
/// # Collective operation
/// One all-reduce; every rank must call it.
fn agree<C: CommunicatorCollectives>(ok: bool, what: &str, comm: &C) {
    let mut all = false;
    comm.all_reduce_into(&ok, &mut all, SystemOperation::logical_and());
    assert!(
        all,
        "rank {}: check failed on some rank: {what}",
        comm.rank()
    );
}

/// A standard normal number by the Box–Muller transform.
fn normal(rng: &mut impl RngExt) -> f64 {
    let u: f64 = 1.0 - rng.random::<f64>();
    let v: f64 = rng.random();
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

/// The points of a workload in [-1, 1]³, the same on every rank.
fn workload(name: &str) -> Vec<[f64; 3]> {
    let mut rng = seeded_rng(0x7e3);
    let uniform =
        |rng: &mut rand_chacha::ChaCha8Rng| [0, 1, 2].map(|_| 2.0 * rng.random::<f64>() - 1.0);
    match name {
        "cube" => (0..NPOINTS).map(|_| uniform(&mut rng)).collect(),
        "plummer" => (0..NPOINTS)
            .map(|_| {
                loop {
                    // Radius from the inverse of the Plummer mass, a = 0.1, truncated
                    // at radius 1; an isotropic direction.
                    let mass: f64 = rng.random::<f64>().max(1e-12);
                    let radius = 0.1 / (mass.powf(-2.0 / 3.0) - 1.0).sqrt();
                    if radius < 1.0 {
                        let direction = [0, 1, 2].map(|_| normal(&mut rng));
                        let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
                        break direction.map(|x| radius * x / norm);
                    }
                }
            })
            .collect(),
        "clusters" => {
            const CENTRES: [[f64; 3]; 5] = [
                [-0.5, -0.5, -0.5],
                [0.5, -0.4, 0.3],
                [-0.3, 0.6, 0.2],
                [0.4, 0.5, -0.6],
                [0.6, -0.3, 0.45],
            ];
            (0..NPOINTS)
                .map(|i| {
                    loop {
                        let offset = [0, 1, 2].map(|_| 0.02 * normal(&mut rng));
                        if offset.iter().all(|x| x.abs() < 0.08) {
                            break [0, 1, 2].map(|d| CENTRES[i % 5][d] + offset[d]);
                        }
                    }
                })
                .collect()
        }
        // Half the points uniform, half in a corner cube of side 0.02: deep leaves
        // next to coarse ones, so blocks and their ancestors meet across levels.
        "graded corner" | "heavy block" => (0..NPOINTS)
            .map(|i| {
                if i % 2 == 0 {
                    uniform(&mut rng)
                } else {
                    [0, 1, 2].map(|_| -1.0 + 0.02 * rng.random::<f64>())
                }
            })
            .collect(),
        _ => unreachable!("no workload {name}"),
    }
}

/// The finest-level keys of points in the padded domain [-1.01, 1.01]³.
fn keys_of(points: &[[f64; 3]]) -> Vec<MortonKey> {
    let domain = PhysicalBox::new([-1.01, -1.01, -1.01, 1.01, 1.01, 1.01]);
    points
        .iter()
        .map(|&point| morton::from_physical_point(point, &domain, DEEPEST_LEVEL as usize))
        .collect()
}

/// Where a key lies relative to the blocks, found by scanning them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// The key is block `b`.
    Block(usize),
    /// The key is an ancestor of blocks; `b` is the first block below it.
    Ancestor(usize),
    /// The key lies strictly inside block `b`.
    Below(usize),
}

/// The tree gathered from every rank.
struct GatheredTree {
    /// Every leaf, ascending.
    leaves: Vec<MortonKey>,
    /// Every key of the tree, by level, each level ascending.
    by_level: Vec<Vec<MortonKey>>,
    /// Every key of the tree.
    keys: HashSet<MortonKey>,
    /// Every block, ascending, with its owner.
    blocks: Vec<MortonKey>,
    block_owners: Vec<usize>,
    /// The number of blocks of every rank.
    nblocks: Vec<usize>,
}

impl GatheredTree {
    /// Gather the leaves and the blocks of every rank.
    ///
    /// # Collective operation
    /// Three gathers; every rank must call it.
    fn new<C: CommunicatorCollectives>(tree: &Octree<'_, C>, comm: &C) -> Self {
        let leaves = gather_to_all(tree.leaf_keys(), comm);
        let blocks = gather_to_all(tree.coarse_tree_leafs(), comm);
        let nblocks = gather_to_all(&[tree.coarse_tree_leafs().len()], comm);
        let block_owners = nblocks
            .iter()
            .enumerate()
            .flat_map(|(owner, &n)| std::iter::repeat_n(owner, n))
            .collect_vec();
        let mut keys: HashSet<MortonKey> = morton::get_interior_keys(&leaves);
        keys.extend(leaves.iter().copied());
        let mut by_level = vec![Vec::new(); DEEPEST_LEVEL as usize + 1];
        for &key in &keys {
            by_level[morton::level(key)].push(key);
        }
        for level in &mut by_level {
            level.sort_unstable();
        }
        Self {
            leaves,
            by_level,
            keys,
            blocks,
            block_owners,
            nblocks,
        }
    }

    fn is_leaf(&self, key: MortonKey) -> bool {
        self.leaves.binary_search(&key).is_ok()
    }

    fn place(&self, key: MortonKey) -> Place {
        if let Some(b) = self.blocks.iter().position(|&block| block == key) {
            Place::Block(b)
        } else if let Some(b) = self
            .blocks
            .iter()
            .position(|&block| morton::is_ancestor(block, key))
        {
            Place::Below(b)
        } else {
            Place::Ancestor(
                self.blocks
                    .iter()
                    .position(|&block| morton::is_ancestor(key, block))
                    .expect("the blocks cover the domain"),
            )
        }
    }

    /// The owner of a key: the rank of its block, or of its first descendant block.
    fn owner(&self, key: MortonKey) -> usize {
        match self.place(key) {
            Place::Block(b) | Place::Below(b) | Place::Ancestor(b) => self.block_owners[b],
        }
    }

    /// The ranks with a block at or below `key`.
    fn sharers(&self, key: MortonKey) -> std::ops::Range<usize> {
        let below = self
            .blocks
            .iter()
            .positions(|&block| morton::is_ancestor(key, block))
            .collect_vec();
        match below.as_slice() {
            [] => {
                let owner = self.owner(key);
                owner..owner + 1
            }
            [first, .., last] | [first @ last] => {
                self.block_owners[*first]..self.block_owners[*last] + 1
            }
        }
    }

    /// The kind of a key of the tree relative to `rank`, by the design (§3.2): own keys
    /// local, shared ancestors `Global`, every other key a ghost of its owner. On one
    /// rank the root is `Global` unless it is a leaf, as in Phase 5.
    fn kind(&self, key: MortonKey, rank: usize) -> KeyType {
        let leaf = self.is_leaf(key);
        let nranks = self.nblocks.len();
        if nranks == 1 && key == morton::root() && !leaf {
            return KeyType::Global;
        }
        let owner = self.owner(key);
        let shared = matches!(self.place(key), Place::Ancestor(_))
            && self
                .blocks
                .iter()
                .zip(&self.block_owners)
                .any(|(&block, &r)| r == rank && morton::is_ancestor(key, block));
        match (owner == rank, leaf) {
            _ if shared => KeyType::Global,
            (true, true) => KeyType::LocalLeaf,
            (true, false) => KeyType::LocalInterior,
            (false, true) => KeyType::GhostLeaf(owner),
            (false, false) => KeyType::GhostInterior(owner),
        }
    }

    /// The tree keys among the same-level neighbour cells of `key`.
    fn neighbours(&self, key: MortonKey) -> impl Iterator<Item = MortonKey> + '_ {
        morton::neighbours(key)
            .into_iter()
            .filter(|&neighbour| morton::is_valid(neighbour) && self.keys.contains(&neighbour))
    }
}

/// Inclusive index bounds of a box, in cells of the deepest level.
fn deepest_bounds(key: MortonKey) -> [[u64; 2]; 3] {
    let (level, index) = morton::decode(key);
    let size = 1u64 << (DEEPEST_LEVEL as usize - level);
    [0usize, 1, 2].map(|dim| {
        let low = index[dim] as u64 * size;
        [low, low + size - 1]
    })
}

/// Two boxes touch when their closed cubes intersect and neither contains the other.
/// Written independently of the implementation under test.
fn touching(a: MortonKey, b: MortonKey) -> bool {
    if morton::is_ancestor(a, b) || morton::is_ancestor(b, a) {
        return false;
    }
    let a_bounds = deepest_bounds(a);
    let b_bounds = deepest_bounds(b);
    (0..3).all(|dim| {
        a_bounds[dim][0] <= b_bounds[dim][1] + 1 && b_bounds[dim][0] <= a_bounds[dim][1] + 1
    })
}

/// Every key that the U-, V-, W- and X-rows of `key` name, by the geometric
/// definitions over the gathered tree (as `nd-fmm-plan`'s regression oracle): U the
/// touching leaves of a leaf, V the same-level keys whose parents touch the parent but
/// which do not touch the key, W (a leaf) the finer keys whose parents touch it but
/// which do not, X the coarser leaves that touch the parent but not the key. By the 2:1
/// balance every entry lies within two levels of the key.
fn row_entries(tree: &GatheredTree, key: MortonKey) -> Vec<MortonKey> {
    let key_is_leaf = tree.is_leaf(key);
    let level = morton::level(key);
    let parent = morton::parent(key);
    let mut entries = Vec::new();
    let levels = level.saturating_sub(2)..(level + 3).min(tree.by_level.len());
    for &other in levels.flat_map(|l| &tree.by_level[l]) {
        let other_is_leaf = tree.is_leaf(other);
        let other_level = morton::level(other);
        let u = key_is_leaf && other_is_leaf && touching(other, key);
        let v = parent.is_some_and(|parent| {
            other_level == level
                && morton::parent(other)
                    .is_some_and(|op| op != parent && touching(op, parent) && !touching(other, key))
        });
        let w = key_is_leaf
            && other_level > level
            && morton::parent(other).is_some_and(|op| touching(op, key) && !touching(other, key));
        let x = parent.is_some_and(|parent| {
            other_is_leaf && other_level < level && touching(parent, other) && !touching(key, other)
        });
        if u || v || w || x {
            entries.push(other);
        }
    }
    entries
}

/// Held keys per rank, Phase 5 against now.
struct Sizes {
    old: Vec<usize>,
    new: Vec<usize>,
}

/// Check the partition map and the held keys of `tree` against the gathered tree and
/// the Phase 5 key set.
///
/// # Collective operation
/// Gathers, Phase 5's key exchange and agreements; every rank must call it.
fn check<C: CommunicatorCollectives>(name: &str, tree: &Octree<'_, C>, comm: &C) -> Sizes {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let options = tree.options();
    let gathered = GatheredTree::new(tree, comm);
    let map = tree.partition_map();
    let all_keys = tree.all_keys();

    // The partition map against the gathered blocks, leaves and owners.
    let mut ok = map.blocks() == gathered.blocks.as_slice() && map.nranks() == size;
    ok &= map.rank() == rank;
    for r in 0..size {
        ok &= map.rank_blocks(r).len() == gathered.nblocks[r];
    }
    for (b, &block) in gathered.blocks.iter().enumerate() {
        ok &= map.is_leaf_block(b) == gathered.is_leaf(block);
        ok &= map.block_owner(b) == gathered.block_owners[b];
    }
    for &key in &gathered.keys {
        ok &= map.owner(key) == gathered.owner(key);
        ok &= map.sharers(key) == gathered.sharers(key);
        let top = !matches!(gathered.place(key), Place::Below(_));
        ok &= map.top_kind(key) == top.then(|| gathered.kind(key, rank));
    }
    for &leaf in &gathered.leaves {
        let (level, index) = morton::decode(leaf);
        let shift = DEEPEST_LEVEL as usize - level;
        for corner in [0, (1 << shift) - 1] {
            let finest = morton::from_index_and_level(index.map(|i| (i << shift) | corner), 16);
            ok &= map.owner(finest) == gathered.owner(leaf);
            ok &= tree.owner_rank(finest) == Ok(map.owner(finest));
            ok &= map.top_kind(finest).is_none() || finest == leaf;
        }
    }
    agree(ok, &format!("{name}: the partition map"), comm);

    // S: the own keys and the shared ancestors; then N(S) and C(N(S)).
    let own_blocks = map.blocks()[map.rank_blocks(rank)].to_vec();
    let own = |key: MortonKey| own_blocks.iter().any(|&b| morton::is_ancestor(b, key));
    let shared = |key: MortonKey| {
        own_blocks
            .iter()
            .any(|&b| b != key && morton::is_ancestor(key, b))
    };
    let s = gathered
        .keys
        .iter()
        .copied()
        .filter(|&key| own(key) || shared(key))
        .collect::<HashSet<_>>();
    let mut n = s.clone();
    for &key in &s {
        n.extend(gathered.neighbours(key));
    }
    let mut h = n.clone();
    if options.ghost_children() {
        for &key in &n {
            if !gathered.is_leaf(key) {
                h.extend(morton::children(key).unwrap());
            }
        }
    }
    let expected: HashMap<MortonKey, KeyType> = h
        .iter()
        .map(|&key| (key, gathered.kind(key, rank)))
        .collect();
    let mut ok = *all_keys == expected;
    ok &= !own_blocks.is_empty() || all_keys.is_empty();
    agree(
        ok,
        &format!("{name}: the held keys are N(S) ∪ C(N(S))"),
        comm,
    );

    // Every key a row of S (with the layer), the halo of the own keys or a parent-child
    // step from S names is held.
    let mut named = HashSet::new();
    for &key in &s {
        if options.ghost_children() {
            named.extend(row_entries(&gathered, key));
        }
        if own(key) {
            named.extend(gathered.neighbours(key));
        }
        named.extend(morton::parent(key));
        if !gathered.is_leaf(key) {
            named.extend(morton::children(key).unwrap());
        }
    }
    let ok = named.iter().all(|key| all_keys.contains_key(key));
    agree(ok, &format!("{name}: every named key is held"), comm);

    // Against Phase 5: H is a subset of its keys, with the same kinds but for the
    // ancestors of other ranks' blocks only (`Global` then, ghosts now); on one rank
    // the same map.
    let old = phase5_keys::phase5_held_keys(tree);
    let mut ok = true;
    for (key, &kind) in all_keys {
        ok &= match old.get(key) {
            Some(&old_kind) if old_kind == kind => true,
            Some(KeyType::Global) => {
                matches!(gathered.place(*key), Place::Ancestor(_))
                    && kind == KeyType::GhostInterior(gathered.owner(*key))
            }
            _ => false,
        };
    }
    ok &= named
        .iter()
        .filter(|key| old.contains_key(key))
        .all(|key| all_keys.contains_key(key));
    if size == 1 {
        ok &= old == *all_keys;
    }
    agree(ok, &format!("{name}: against the Phase 5 key set"), comm);

    Sizes {
        old: gather_to_all(&[old.len()], comm),
        new: gather_to_all(&[all_keys.len()], comm),
    }
}

/// The maximum and the mean of per-rank counts.
fn max_mean(counts: &[usize]) -> (usize, f64) {
    let max = counts.iter().copied().max().unwrap_or(0);
    let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    (max, mean)
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;

    let base = OctreeOptions::new()
        .with_max_level(16)
        .with_max_fine_keys(16);
    for name in [
        "cube",
        "plummer",
        "clusters",
        "graded corner",
        "heavy block",
        "root only",
    ] {
        let options = match name {
            // The corner cluster cannot be split below level 2, so its block takes
            // most of the weight and leaves ranks without blocks (O4).
            "heavy block" => base.with_max_level(2),
            "root only" => base.with_max_level(0),
            _ => base,
        };
        let keys = keys_of(&workload(if name == "root only" { "cube" } else { name }));
        let local = keys.iter().copied().skip(rank).step_by(size).collect_vec();
        for ghost_children in [true, false] {
            let options = options.with_ghost_children(ghost_children);
            let label = format!("{name}, ghost children {ghost_children}");
            let tree = Octree::new(&local, options, &comm);
            let sizes = check(&label, &tree, &comm);
            let empty = gather_to_all(&[tree.coarse_tree_leafs().is_empty()], &comm);
            if rank == 0 {
                let (old_max, old_mean) = max_mean(&sizes.old);
                let (new_max, new_mean) = max_mean(&sizes.new);
                println!(
                    "{label} on {size} rank(s): {} blocks, {} rank(s) without blocks; held \
                     keys per rank max/mean: Phase 5 {old_max}/{old_mean:.0}, now \
                     {new_max}/{new_mean:.0}",
                    tree.partition_map().blocks().len(),
                    empty.iter().filter(|&&e| e).count(),
                );
            }
        }
    }
    if rank == 0 {
        println!("Locally essential key set checks passed on {size} rank(s).");
    }
}

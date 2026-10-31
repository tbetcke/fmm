//! The partition by weight (Phase 5 T4; docs/design/distributed-fmm.md §3.2, §3.4).
//!
//! Every rank draws the same global point set from a seed, so it can rebuild every
//! rank's input and the one-rank tree of all points without communication. For the
//! four workloads of the design (the uniform cube, the Plummer sphere, five Gaussian
//! clusters, and a duplicate-heavy cube with a fifth of its points on 100 positions),
//! each with the options `Fmm::build` uses (`max_level` 16, 64 keys per leaf, the
//! ghost-children layer), it checks:
//! - the leaves, gathered from every rank, equal the leaves of the one-rank tree of
//!   all keys (O1: every coarse block is a node of the one-rank tree);
//! - every rank's weight, the input keys it owns with duplicates, is at most a fair
//!   share plus the heaviest coarse block (O3's bound 1 + w_max / (W / P) of the
//!   mean), computed from the gathered blocks and keys;
//! - the partition does not depend on which rank passed which key: every point's key
//!   on its share rank (every P-th point), every key on rank 0, and the share with the
//!   last rank's keys moved to rank 0 (a rank with no keys) give the same blocks;
//! - every invariant of a tree: completeness, linearity and 2:1 balance, the ghosts
//!   name the rank that owns them with its classification, the neighbour map has
//!   entries for exactly the non-ghost keys, each neighbour once, every held block or
//!   ancestor of a block has the partition map's kind, every own block is held, and
//!   so is every child of every `Global` key (the ghost-children layer).
//!
//! The duplicate-heavy cube is also built with `PartitionWeight::DistinctKeys` and
//! with a block refinement of 1: the same leaves. The former panic (fewer blocks
//! than ranks) is a tree whose only block is the root, built with `max_level` 0 and
//! with fewer distinct keys than `max_fine_keys`, and the globally empty input: one
//! rank owns the root, the others own nothing and hold no key (Phase 5S T3), on every
//! rank count.
//!
//! The second cut (Phase 5S T7, S3; docs/design/scale-out.md §6.3,
//! `Octree::repartition`): every workload's tree (the share) is cut again by several
//! weights per block, the square of its leaves (strongly uneven), the levels of its
//! leaves cubed, one heavy block (so that ranks are left without blocks), no weight at
//! all (rank 0 takes every block) and unit weights, and each recut tree is checked:
//! - **the leaves are the first cut's, key for key** (gathered in rank order), and so
//!   are the blocks with their leaf flags;
//! - the cut: contiguous ranges, each boundary the nearest to its target of the
//!   gathered weights (up to the scaling's rounding), so every rank weighs at most a
//!   fair share plus the heaviest block; leaves exactly on the ranks with blocks, each
//!   in a block of its rank and owned by it;
//! - complete, linear and 2:1 balanced; the held keys with the partition map's kinds,
//!   the own leaves exactly the local leaves, every own block and every child of a
//!   `Global` key held, the neighbour map over exactly the non-ghost keys; on one rank
//!   the tree unchanged;
//! - a second repartition by the same weights keeps the cut;
//! - with a minimum gain (−1, 0, 0.02, 0.5, 1) the tree is the recut tree where the new
//!   cut lowers the heaviest rank's weight by at least that fraction, and the first cut
//!   otherwise, by brute force over the gathered weights.
//!
//! `test_mpi_locally_essential` checks the held keys of a recut tree by brute force.
//!
//! The minimum leaf level (Helmholtz CH3.0a; `OctreeOptions::with_min_leaf_level`, the
//! maximum leaf width of docs/design/helmholtz-fmm-proposal.md §3.7): the uniform cube
//! and the Plummer sphere, the first 2,000 points of each, at levels 4 and 5, each
//! checked as above (every invariant and the one-rank leaves, from the share and from
//! rank 0), and further:
//! - the one-rank leaves are the brute-force refinement of the rule (split a box above
//!   `max_level` that holds more than `max_fine_keys` distinct keys, or a key and lies
//!   above the level), 2:1 balanced;
//! - every leaf that holds a key lies at the level or below;
//! - the level zero builds the tree without the option, key for key, and a level only
//!   refines.
//!
//! Phase 5S T3a also compared every tree key for key with a private copy of the Phase 5
//! construction (the sample sorts), removed with T3, whose `test_mpi_locally_essential`
//! checks the held keys.
//!
//! Every check is agreed over all ranks before it is asserted, so a failure stops
//! every rank, not one.

use itertools::Itertools;
use mpi::{
    collective::SystemOperation,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives},
};
use nd_octree::{
    MortonKey, Octree, OctreeOptions, PartitionWeight, PhysicalBox,
    constants::DEEPEST_LEVEL,
    morton,
    octree::{KeyType, is_complete_linear_and_balanced},
    tools::seeded_rng,
};
use rand::RngExt;
use rlst::distributed_tools::array_tools::gather_to_all;
use std::collections::HashSet;

/// The number of points of each workload.
const NPOINTS: usize = 10_000;

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

/// The points of a workload, the same on every rank.
fn workload(name: &str) -> Vec<[f64; 3]> {
    let mut rng = seeded_rng(0xc32);
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
        "duplicates" => {
            // A fifth of the points on 100 positions, 20 copies each at N = 10,000.
            let positions = (0..100).map(|_| uniform(&mut rng)).collect_vec();
            (0..NPOINTS)
                .map(|i| {
                    if i % 5 == 0 {
                        positions[(i / 5) % 100]
                    } else {
                        uniform(&mut rng)
                    }
                })
                .collect()
        }
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

/// How the global keys are passed to the ranks.
#[derive(Clone, Copy, Debug)]
enum Input {
    /// Key i on rank i mod P.
    Share,
    /// Every key on rank 0.
    RankZero,
    /// The share, with the last rank's keys on rank 0, so the last rank passes none.
    EmptyLast,
}

/// This rank's input keys.
fn input(keys: &[MortonKey], input: Input, rank: usize, size: usize) -> Vec<MortonKey> {
    let owner = |i: usize| match input {
        Input::Share => i % size,
        Input::RankZero => 0,
        Input::EmptyLast => {
            if i % size == size - 1 {
                0
            } else {
                i % size
            }
        }
    };
    (0..keys.len())
        .filter(|&i| owner(i) == rank)
        .map(|i| keys[i])
        .collect()
}

/// The leaves of the one-rank tree of `keys`, built on this rank alone.
fn one_rank_leaves(keys: &[MortonKey], options: OctreeOptions) -> Vec<MortonKey> {
    let comm = SimpleCommunicator::self_comm();
    Octree::new(keys, options, &comm).leaf_keys().clone()
}

/// The leaves of the minimum-leaf-level rule by plain recursion from `key`, before
/// the 2:1 balance: a box above `max_level` is split while it holds more than
/// `max_keys` of the sorted distinct `keys`, or one of them and lies above `min_level`.
fn brute_force_leaves(
    key: MortonKey,
    keys: &[MortonKey],
    (max_level, max_keys, min_level): (usize, usize, usize),
    leaves: &mut Vec<MortonKey>,
) {
    let level = morton::level(key);
    let split = keys.len() > max_keys || (!keys.is_empty() && level < min_level);
    if level >= max_level || !split {
        leaves.push(key);
        return;
    }
    let mut rest = keys;
    for child in morton::children(key).unwrap() {
        let n = rest.partition_point(|&k| morton::is_ancestor(child, k));
        let (mine, others) = rest.split_at(n);
        brute_force_leaves(child, mine, (max_level, max_keys, min_level), leaves);
        rest = others;
    }
}

/// Check the trees of the minimum leaf level on the first 2,000 points of `name`.
///
/// # Collective operation
/// Tree constructions, gathers and agreements; every rank must call it.
fn check_min_leaf_level<C: CommunicatorCollectives>(name: &str, options: OctreeOptions, comm: &C) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let keys = keys_of(&workload(name)[..2000]);
    let distinct = morton::linearize(&keys);
    let unruled = one_rank_leaves(&keys, options);
    for min_level in [0, 4, 5] {
        let label = format!("{name}, minimum leaf level {min_level}");
        let ruled = options.with_min_leaf_level(min_level);
        let one_rank = one_rank_leaves(&keys, ruled);

        let mut expected = Vec::new();
        brute_force_leaves(
            morton::root(),
            &distinct,
            (ruled.max_level(), ruled.max_fine_keys(), min_level),
            &mut expected,
        );
        let mut ok = one_rank == morton::balance(&expected, morton::root());
        ok &= one_rank.iter().all(|&leaf| {
            morton::level(leaf) >= min_level
                || !distinct.iter().any(|&key| morton::is_ancestor(leaf, key))
        });
        // The level only refines.
        if min_level == 0 {
            ok &= one_rank == unruled;
        } else {
            ok &= one_rank.len() >= unruled.len();
        }
        agree(ok, &format!("{label}: the rule's one-rank leaves"), comm);

        for kind in [Input::Share, Input::RankZero] {
            let local = input(&keys, kind, rank, size);
            let tree = Octree::new(&local, ruled, comm);
            let balance = check_tree(&format!("{label}, {kind:?}"), &tree, &keys, &one_rank, comm);
            if rank == 0 {
                println!(
                    "{label}, {kind:?} on {size} rank(s): {} blocks, {} leaves ({} without \
                     the level), weight max/mean {:.3} within the bound {:.3}",
                    balance.blocks,
                    one_rank.len(),
                    unruled.len(),
                    balance.max_over_mean,
                    balance.bound
                );
            }
        }
    }
}

/// What a partition weighs, as the test reports it.
struct Balance {
    blocks: usize,
    max_over_mean: f64,
    bound: f64,
}

/// Check the invariants of `tree`, built from `keys` (the global input keys, every
/// rank's included), and return the balance of the key weight.
///
/// # Collective operation
/// Gathers and agreements; every rank must call it.
fn check_tree<C: CommunicatorCollectives>(
    name: &str,
    tree: &Octree<'_, C>,
    keys: &[MortonKey],
    one_rank: &[MortonKey],
    comm: &C,
) -> Balance {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let all_keys = tree.all_keys();

    // Complete, linear and 2:1 balanced, and the one-rank leaves.
    agree(
        is_complete_linear_and_balanced(tree.leaf_keys(), comm),
        &format!("{name}: complete, linear, balanced"),
        comm,
    );
    let leaves = gather_to_all(tree.leaf_keys(), comm);
    let owners = gather_to_all(&vec![rank; tree.leaf_keys().len()], comm);
    agree(
        leaves == one_rank,
        &format!("{name}: one-rank leaves"),
        comm,
    );

    // Ownership: a rank has leaves exactly when it has blocks; every leaf's keys are
    // owned by the leaf's rank; the bounds do not decrease.
    let nblocks = gather_to_all(&[tree.coarse_tree_leafs().len()], comm);
    let mut ok = tree.leaf_keys().is_empty() == tree.coarse_tree_leafs().is_empty();
    ok &= tree
        .coarse_tree_bounds()
        .iter()
        .tuple_windows()
        .all(|(a, b)| a <= b);
    for (&leaf, &owner) in leaves.iter().zip(&owners) {
        ok &= nblocks[owner] > 0;
        ok &= keys
            .iter()
            .filter(|&&key| morton::is_ancestor(leaf, key))
            .all(|&key| tree.owner_rank(key) == Ok(owner));
    }
    agree(ok, &format!("{name}: ownership"), comm);

    // Ghosts name another rank and that rank's classification; the neighbour map has
    // exactly the non-ghost keys, each neighbour once.
    let owned = leaves
        .iter()
        .copied()
        .zip(owners.iter().copied())
        .collect_vec();
    let mut ok = true;
    for (&key, &key_type) in all_keys {
        if let Some(ghost_rank) = key_type.ghost_rank() {
            ok &= ghost_rank != rank && nblocks[ghost_rank] > 0;
            let is_leaf = matches!(key_type, KeyType::GhostLeaf(_));
            ok &= is_leaf == owned.contains(&(key, ghost_rank));
            ok &= owned
                .iter()
                .any(|&(leaf, owner)| owner == ghost_rank && morton::is_ancestor(key, leaf));
        }
        ok &= tree.neighbour_map().contains_key(&key) != key_type.is_ghost();
    }
    for neighbours in tree.neighbour_map().values() {
        ok &= neighbours.iter().unique().count() == neighbours.len();
    }
    agree(ok, &format!("{name}: ghosts and neighbours"), comm);

    // Every held block or ancestor of a block has the partition map's kind (Phase 5S
    // T3: other ranks' blocks only where the locally essential set needs them), every
    // own block is held, and so is every child of every `Global` key.
    let blocks = gather_to_all(tree.coarse_tree_leafs(), comm);
    let map = tree.partition_map();
    let mut ok = map.blocks() == blocks.as_slice();
    for (&key, &key_type) in all_keys {
        if let Some(top_kind) = map.top_kind(key) {
            ok &= key_type == top_kind;
        }
        if key_type == KeyType::Global {
            ok &= morton::children(key)
                .unwrap()
                .iter()
                .all(|child| all_keys.contains_key(child));
        }
    }
    for block in tree.coarse_tree_leafs() {
        ok &= all_keys
            .get(block)
            .is_some_and(|key_type| !key_type.is_ghost());
    }
    agree(ok, &format!("{name}: blocks and the partition map"), comm);

    // The weight bound of O3: every rank within a fair share plus the heaviest block.
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();
    let heaviest = blocks
        .iter()
        .map(|&block| {
            let start = sorted.partition_point(|&key| key < block);
            sorted[start..]
                .iter()
                .take_while(|&&key| morton::is_ancestor(block, key))
                .count()
        })
        .max()
        .unwrap();
    let mut per_rank = vec![0usize; size];
    for &key in keys {
        per_rank[tree.owner_rank(key).unwrap()] += 1;
    }
    let total = keys.len();
    agree(
        per_rank
            .iter()
            .all(|&weight| weight * size <= total + heaviest * size),
        &format!("{name}: weight within a fair share plus the heaviest block"),
        comm,
    );
    let mean = total as f64 / size as f64;
    Balance {
        blocks: blocks.len(),
        max_over_mean: *per_rank.iter().max().unwrap() as f64 / mean,
        bound: 1.0 + heaviest as f64 / mean,
    }
}

/// Check a tree whose only coarse block is the root: one rank owns it, a leaf, and
/// every other rank owns nothing and holds no key (its locally essential set is empty).
///
/// # Collective operation
/// Gathers and agreements; every rank must call it.
fn check_root_only<C: CommunicatorCollectives>(name: &str, tree: &Octree<'_, C>, comm: &C) {
    let rank = comm.rank() as usize;
    let nblocks = gather_to_all(&[tree.coarse_tree_leafs().len()], comm);
    let owner = nblocks.iter().position(|&n| n > 0).unwrap();
    let root = morton::root();
    let mut ok = nblocks.iter().sum::<usize>() == 1;
    ok &= tree.owner_rank(morton::deepest_first()) == Ok(owner);
    ok &= tree.owner_rank(morton::deepest_last()) == Ok(owner);
    if rank == owner {
        ok &= tree.coarse_tree_leafs() == &vec![root];
        ok &= tree.leaf_keys() == &vec![root];
        ok &= tree.all_keys().get(&root) == Some(&KeyType::LocalLeaf);
        ok &= tree.neighbour_map().get(&root) == Some(&Vec::new());
    } else {
        ok &= tree.leaf_keys().is_empty();
        ok &= tree.local_leaf(morton::deepest_first()) == Ok(None);
        if comm.size() > 1 {
            ok &= tree.all_keys().is_empty();
            ok &= tree.neighbour_map().is_empty();
            ok &= tree.partition_map().top_kind(root) == Some(KeyType::GhostLeaf(owner));
        }
    }
    let found = tree
        .lookup_leaves(&[morton::deepest_first(), morton::deepest_last()])
        .unwrap();
    ok &= found.iter().all(|location| {
        location.is_ok_and(|location| location.leaf == root && location.owner_rank == owner)
    });
    agree(ok, &format!("{name}: the root as the only block"), comm);
    agree(
        is_complete_linear_and_balanced(tree.leaf_keys(), comm),
        &format!("{name}: complete, linear, balanced"),
        comm,
    );
}

/// The weights of the second cut.
#[derive(Clone, Copy, Debug)]
enum Weights {
    /// The square of the number of leaves of a block.
    LeavesSquared,
    /// The sum over the leaves of a block of the leaf's level cubed.
    LevelsCubed,
    /// Unit weights, but the last block of the tree weighs a million.
    OneHeavy,
    /// Every weight zero.
    Zero,
    /// Every weight one.
    Unit,
}

/// The weight of each of this rank's blocks.
fn block_weights<C: CommunicatorCollectives>(tree: &Octree<'_, C>, weights: Weights) -> Vec<f64> {
    let leaves = tree.leaf_keys();
    let nblocks = tree.partition_map().blocks().len();
    let last = tree.partition_map().blocks()[nblocks - 1];
    tree.coarse_tree_leafs()
        .iter()
        .map(|&block| {
            let below = leaves
                .iter()
                .filter(|&&leaf| leaf == block || morton::is_ancestor(block, leaf));
            match weights {
                Weights::LeavesSquared => (below.count() as f64).powi(2),
                Weights::LevelsCubed => below
                    .map(|&leaf| (morton::level(leaf) as f64).powi(3))
                    .sum(),
                Weights::OneHeavy if block == last => 1e6,
                Weights::OneHeavy | Weights::Unit => 1.0,
                Weights::Zero => 0.0,
            }
        })
        .collect()
}

/// The max over mean of the ranks' weights, given every block's weight and the number
/// of blocks of every rank.
fn max_over_mean(weights: &[f64], nblocks: &[usize]) -> f64 {
    let total: f64 = weights.iter().sum();
    if total == 0.0 {
        return 1.0;
    }
    let mut start = 0;
    let mut max = 0.0f64;
    for &n in nblocks {
        max = max.max(weights[start..start + n].iter().sum());
        start += n;
    }
    max * nblocks.len() as f64 / total
}

/// Check `recut`, the second cut of `tree` by `own_weights`, and return its weight's
/// max over mean with that of the first cut by the same weights.
///
/// # Collective operation
/// Gathers and agreements; every rank must call it.
fn check_recut<C: CommunicatorCollectives>(
    name: &str,
    tree: &Octree<'_, C>,
    recut: &Octree<'_, C>,
    own_weights: &[f64],
    comm: &C,
) -> (f64, f64) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let map = recut.partition_map();

    // The leaves, key for key, and the blocks with their leaf flags.
    let first = gather_to_all(tree.leaf_keys(), comm);
    let second = gather_to_all(recut.leaf_keys(), comm);
    agree(
        first == second,
        &format!("{name}: the leaves of the first cut, key for key"),
        comm,
    );
    let first_map = tree.partition_map();
    let mut ok = map.blocks() == first_map.blocks();
    ok &= (0..map.blocks().len()).all(|b| map.is_leaf_block(b) == first_map.is_leaf_block(b));
    ok &= recut.options() == tree.options();
    agree(
        ok,
        &format!("{name}: the blocks and their leaf flags"),
        comm,
    );

    // The cut: contiguous ranges, each boundary nearest to its target.
    let weights = gather_to_all(own_weights, comm);
    let old_blocks = gather_to_all(&[tree.coarse_tree_leafs().len()], comm);
    let new_blocks = gather_to_all(&[recut.coarse_tree_leafs().len()], comm);
    let blocks = gather_to_all(recut.coarse_tree_leafs(), comm);
    let total: f64 = weights.iter().sum();
    let heaviest = weights.iter().copied().fold(0.0f64, f64::max);
    let mut ok = blocks == map.blocks() && weights.len() == blocks.len();
    let mut prefix = vec![0.0f64];
    for &weight in &weights {
        prefix.push(prefix.last().unwrap() + weight);
    }
    let mut start = 0;
    for (r, &n) in new_blocks.iter().enumerate() {
        ok &= map.rank_blocks(r) == (start..start + n);
        let weight: f64 = weights[start..start + n].iter().sum();
        ok &= weight <= total / size as f64 + heaviest + 1e-9 * total;
        if r > 0 {
            let target = total * r as f64 / size as f64;
            let best = (map.rank_blocks(r - 1).start..=blocks.len())
                .map(|b| (prefix[b] - target).abs())
                .fold(f64::INFINITY, f64::min);
            ok &= (prefix[start] - target).abs() <= best + 1e-9 * total;
        }
        start += n;
    }
    agree(ok, &format!("{name}: the nearest-boundary cut"), comm);

    // Ownership: leaves exactly with blocks, each in one of its rank's blocks and owned by
    // it; the bounds do not decrease.
    let own_blocks = recut.coarse_tree_leafs();
    let mut ok = recut.leaf_keys().is_empty() == own_blocks.is_empty();
    ok &= recut
        .coarse_tree_bounds()
        .iter()
        .tuple_windows()
        .all(|(a, b)| a <= b);
    for &leaf in recut.leaf_keys() {
        ok &= own_blocks
            .iter()
            .any(|&block| block == leaf || morton::is_ancestor(block, leaf));
        let (level, index) = morton::decode(leaf);
        let shift = DEEPEST_LEVEL as usize - level;
        for corner in [0, (1 << shift) - 1] {
            let finest = morton::from_index_and_level(index.map(|i| (i << shift) | corner), 16);
            ok &= recut.owner_rank(finest) == Ok(rank);
            ok &= recut.local_leaf(finest) == Ok(Some(leaf));
        }
    }
    agree(ok, &format!("{name}: ownership"), comm);
    agree(
        is_complete_linear_and_balanced(recut.leaf_keys(), comm),
        &format!("{name}: complete, linear, balanced"),
        comm,
    );

    // The held keys: the map's kinds, the own leaves, the own blocks, the children of the
    // `Global` keys, the neighbour map.
    let all_keys = recut.all_keys();
    let mut ok = true;
    let mut local_leaves = Vec::new();
    for (&key, &kind) in all_keys {
        if let Some(top_kind) = map.top_kind(key) {
            ok &= kind == top_kind;
        }
        match kind {
            KeyType::LocalLeaf => local_leaves.push(key),
            KeyType::LocalInterior => ok &= map.owner(key) == rank,
            KeyType::Global => {
                ok &= own_blocks
                    .iter()
                    .any(|&block| morton::is_ancestor(key, block));
                ok &= morton::children(key)
                    .unwrap()
                    .iter()
                    .all(|child| all_keys.contains_key(child));
            }
            KeyType::GhostLeaf(owner) | KeyType::GhostInterior(owner) => {
                ok &= owner != rank && map.owner(key) == owner;
            }
        }
        ok &= recut.neighbour_map().contains_key(&key) != kind.is_ghost();
    }
    local_leaves.sort_unstable();
    ok &= local_leaves == *recut.leaf_keys();
    ok &= own_blocks
        .iter()
        .all(|block| all_keys.get(block).is_some_and(|kind| !kind.is_ghost()));
    ok &= recut.neighbour_map().len() == all_keys.values().filter(|k| !k.is_ghost()).count();
    agree(ok, &format!("{name}: the held keys"), comm);

    // One rank: unchanged.
    if size == 1 {
        let mut ok = recut.all_keys() == tree.all_keys();
        ok &= recut.coarse_tree_bounds() == tree.coarse_tree_bounds();
        ok &= recut.neighbour_map() == tree.neighbour_map();
        agree(ok, &format!("{name}: unchanged on one rank"), comm);
    }

    (
        max_over_mean(&weights, &old_blocks),
        max_over_mean(&weights, &new_blocks),
    )
}

/// The minimum gain of `Octree::repartition`: with `min_gain` the tree is `recut` (the cut
/// without it) where that lowers the heaviest rank's weight by at least `min_gain` of it,
/// and `tree` unchanged otherwise, decided here by brute force over the gathered weights
/// (a decision within 10⁻⁹ of the boundary, where the scaled integers may round either
/// way, is not checked). -1 always takes the new cut (the mean plus the heaviest block is
/// at most twice the current heaviest rank), 1 only a cut whose heaviest rank weighs
/// nothing.
///
/// # Collective operation
/// Repartitions, gathers and agreements; every rank must call it.
fn check_min_gain<C: CommunicatorCollectives>(
    name: &str,
    tree: &Octree<'_, C>,
    recut: &Octree<'_, C>,
    own_weights: &[f64],
    comm: &C,
) {
    let weights = gather_to_all(own_weights, comm);
    let heaviest = |nblocks: &[usize]| {
        let mut start = 0;
        let mut max = 0.0f64;
        for &n in nblocks {
            max = max.max(weights[start..start + n].iter().sum());
            start += n;
        }
        max
    };
    let current = heaviest(&gather_to_all(&[tree.coarse_tree_leafs().len()], comm));
    let new = heaviest(&gather_to_all(&[recut.coarse_tree_leafs().len()], comm));
    let total: f64 = weights.iter().sum();
    for gain in [-1.0, 0.0, 0.02, 0.5, 1.0] {
        let decided = tree.repartition(own_weights, Some(gain));
        let margin = new - (1.0 - gain) * current;
        let expected = if margin <= -1e-9 * total.max(1.0) || (new == 0.0 && gain <= 1.0) {
            Some(recut)
        } else if margin >= 1e-9 * total.max(1.0) {
            Some(tree)
        } else {
            None
        };
        let ok = expected.is_none_or(|expected| {
            decided.coarse_tree_bounds() == expected.coarse_tree_bounds()
                && decided.partition_map() == expected.partition_map()
                && decided.leaf_keys() == expected.leaf_keys()
                && decided.all_keys() == expected.all_keys()
        });
        agree(ok, &format!("{name}: the minimum gain {gain}"), comm);
    }
}

/// Cut `tree` again with every [`Weights`] and check each result ([`check_recut`]); a
/// second repartition by the same weights keeps the cut, and the minimum gain keeps or
/// takes it as the gathered weights say ([`check_min_gain`]).
///
/// # Collective operation
/// Repartitions, gathers and agreements; every rank must call it.
fn check_second_cut<C: CommunicatorCollectives>(name: &str, tree: &Octree<'_, C>, comm: &C) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    for weights in [
        Weights::LeavesSquared,
        Weights::LevelsCubed,
        Weights::OneHeavy,
        Weights::Zero,
        Weights::Unit,
    ] {
        let label = format!("{name}, second cut by {weights:?}");
        let own_weights = block_weights(tree, weights);
        let recut = tree.repartition(&own_weights, None);
        let (first, second) = check_recut(&label, tree, &recut, &own_weights, comm);
        let empty = gather_to_all(&[recut.coarse_tree_leafs().is_empty()], comm);
        // The weights of a block do not depend on its owner, so a second repartition
        // keeps the cut.
        let again = recut.repartition(&block_weights(&recut, weights), None);
        agree(
            again.coarse_tree_bounds() == recut.coarse_tree_bounds()
                && again.all_keys() == recut.all_keys(),
            &format!("{label}: the cut is a fixed point"),
            comm,
        );
        check_min_gain(&label, tree, &recut, &own_weights, comm);
        if rank == 0 {
            println!(
                "{label} on {size} rank(s): {} rank(s) without blocks; weight max/mean \
                 {first:.3} by keys, {second:.3} by weight",
                empty.iter().filter(|&&e| e).count(),
            );
        }
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;

    let options = OctreeOptions::new()
        .with_max_level(16)
        .with_max_fine_keys(64)
        .with_ghost_children(true);

    for name in ["cube", "plummer", "clusters", "duplicates"] {
        let keys = keys_of(&workload(name));
        let one_rank = one_rank_leaves(&keys, options);
        let mut partitions = Vec::new();
        for kind in [Input::Share, Input::RankZero, Input::EmptyLast] {
            if size == 1 && matches!(kind, Input::EmptyLast) {
                continue;
            }
            let label = format!("{name}, {kind:?}");
            let local = input(&keys, kind, rank, size);
            let tree = Octree::new(&local, options, &comm);
            let balance = check_tree(&label, &tree, &keys, &one_rank, &comm);
            partitions.push(gather_to_all(tree.coarse_tree_leafs(), &comm));
            partitions.push(tree.coarse_tree_bounds().clone());
            if matches!(kind, Input::Share) {
                check_second_cut(name, &tree, &comm);
            }
            if rank == 0 {
                println!(
                    "{label} on {size} rank(s): {} blocks, {} leaves, weight max/mean {:.3} \
                     within the bound {:.3}",
                    balance.blocks,
                    one_rank.len(),
                    balance.max_over_mean,
                    balance.bound
                );
            }
        }
        // Blocks and bounds, the same for every input distribution.
        agree(
            partitions.chunks(2).all_equal(),
            &format!("{name}: the partition depends on the input distribution"),
            &comm,
        );

        if name == "duplicates" {
            // Other weights and refinement factors change the partition, never the
            // leaves.
            for (label, other) in [
                (
                    "distinct keys",
                    options.with_partition_weight(PartitionWeight::DistinctKeys),
                ),
                ("block refinement 1", options.with_block_refinement(1)),
                ("block refinement 0", options.with_block_refinement(0)),
            ] {
                let local = input(&keys, Input::Share, rank, size);
                let tree = Octree::new(&local, other, &comm);
                let leaves = gather_to_all(tree.leaf_keys(), &comm);
                agree(
                    leaves == one_rank,
                    &format!("{name}, {label}: leaves"),
                    &comm,
                );
                let blocks = gather_to_all(tree.coarse_tree_leafs(), &comm);
                if rank == 0 {
                    println!("{name}, {label} on {size} rank(s): {} blocks", blocks.len());
                }
            }
        }
    }

    // The minimum leaf level, from a uniform input and one with a dense core and a
    // sparse halo.
    for name in ["cube", "plummer"] {
        check_min_leaf_level(name, options, &comm);
    }

    // The former panic case: a tree with one block, the root, on every rank count.
    // With `max_level` 0, and with fewer distinct keys than `max_fine_keys` (three
    // points, each on one rank, or all on rank 0), each with and without the
    // ghost-children layer.
    let mut rng = seeded_rng(5);
    let few = (0..3)
        .map(|_| {
            morton::from_index_and_level(
                [0, 1, 2].map(|_| rng.random_range(0..1usize << DEEPEST_LEVEL)),
                DEEPEST_LEVEL as usize,
            )
        })
        .collect_vec();
    for ghost_children in [false, true] {
        let base = OctreeOptions::new().with_ghost_children(ghost_children);
        for (label, local, options) in [
            (
                "max_level 0",
                input(&few, Input::Share, rank, size),
                base.with_max_level(0),
            ),
            (
                "three keys",
                input(&few, Input::Share, rank, size),
                base.with_max_fine_keys(4),
            ),
            (
                "three keys on rank 0",
                input(&few, Input::RankZero, rank, size),
                base.with_max_fine_keys(4),
            ),
            ("no keys anywhere", Vec::new(), base),
        ] {
            let tree = Octree::new(&local, options, &comm);
            let label = format!("{label}, ghost children {ghost_children}");
            check_root_only(&label, &tree, &comm);
            if label.starts_with("no keys") {
                // As on one rank: the root is a leaf of rank 0.
                agree(
                    (rank == 0) == !tree.leaf_keys().is_empty(),
                    "no keys anywhere: the root on rank 0",
                    &comm,
                );
            }
        }
    }
    // Few keys on many ranks still split where the one-rank tree splits.
    let local = input(&few, Input::Share, rank, size);
    let tree = Octree::new(&local, OctreeOptions::default(), &comm);
    let leaves = gather_to_all(tree.leaf_keys(), &comm);
    agree(
        leaves == one_rank_leaves(&few, OctreeOptions::default()),
        "three keys, one per leaf: one-rank leaves",
        &comm,
    );
    let owners = few
        .iter()
        .map(|&key| tree.owner_rank(key).unwrap())
        .collect::<HashSet<_>>();
    if rank == 0 {
        println!(
            "three keys, one per leaf, on {size} rank(s): {} leaves on {} rank(s)",
            leaves.len(),
            owners.len()
        );
        println!("Weighted partition checks passed on {size} rank(s).");
    }
}

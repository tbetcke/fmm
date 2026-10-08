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
//!   entries for exactly the non-ghost keys, each neighbour once, every coarse block
//!   of every rank is a key of every rank (the ghost-children layer), and so is every
//!   child of every `Global` key.
//!
//! The duplicate-heavy cube is also built with `PartitionWeight::DistinctKeys` and
//! with a block refinement of 1: the same leaves. The former panic (fewer blocks
//! than ranks) is a tree whose only block is the root, built with `max_level` 0 and
//! with fewer distinct keys than `max_fine_keys`, and the globally empty input: one
//! rank owns the root, the others own nothing and hold it as a ghost, on every rank
//! count.
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

    // Every block of every rank, and every child of every `Global` key, is a key of
    // every rank, with its owner's classification.
    let blocks = gather_to_all(tree.coarse_tree_leafs(), comm);
    let block_owners = nblocks
        .iter()
        .enumerate()
        .flat_map(|(owner, &n)| std::iter::repeat_n(owner, n))
        .collect_vec();
    let mut ok = true;
    for (&block, &owner) in blocks.iter().zip(&block_owners) {
        let is_leaf = owned.contains(&(block, owner));
        ok &= match all_keys.get(&block) {
            Some(KeyType::LocalLeaf) => owner == rank && is_leaf,
            Some(KeyType::LocalInterior) => owner == rank && !is_leaf,
            Some(KeyType::GhostLeaf(r)) => *r == owner && is_leaf,
            Some(KeyType::GhostInterior(r)) => *r == owner && !is_leaf,
            // On one rank the only block is the root, `Global` unless it is a leaf.
            Some(KeyType::Global) => size == 1 && block == morton::root() && !is_leaf,
            None => false,
        };
    }
    for (&key, &key_type) in all_keys {
        if key_type == KeyType::Global {
            ok &= morton::children(key)
                .unwrap()
                .iter()
                .all(|child| all_keys.contains_key(child));
        }
    }
    agree(ok, &format!("{name}: replicated blocks"), comm);

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
/// every other rank owns nothing and holds the root as its ghost.
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
            ok &= tree.all_keys().len() == 1;
            ok &= tree.all_keys().get(&root) == Some(&KeyType::GhostLeaf(owner));
            ok &= tree.neighbour_map().is_empty();
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

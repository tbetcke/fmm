//! The trees: nd-octree's points-per-leaf rule with a prototype maximum leaf width
//! (proposal §3.7).
//!
//! nd-octree has no maximum leaf width yet (CH3.0 adds one). The prototype refines in
//! the spike, by nd-octree's own rule (a box with more than n distinct level-16 keys is
//! refined, `create_local_tree`) plus the width rule (a box that holds a point and is
//! wider than c_leaf λ is refined). The leaves of that complete tree go to `Octree::new`
//! as one level-16 key each, with `max_fine_keys = 1`: every refined box then holds at
//! least eight keys and every leaf one, so nd-octree rebuilds the same tree and 2:1
//! balances it as it balances the Laplace trees.

use mpi::traits::CommunicatorCollectives;
use nd_octree::constants::{DEEPEST_LEVEL, LEVEL_MASK};
use nd_octree::{MortonKey, Octree, OctreeOptions, PhysicalBox, morton};

/// The deepest level, as a `usize`.
pub const DEEPEST: usize = DEEPEST_LEVEL as usize;

/// The root cube [0, 1]³.
pub fn root_box() -> PhysicalBox {
    PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
}

/// The level-16 key of every point, as `points_to_morton` computes it.
pub fn fine_keys(points: &[[f64; 3]]) -> Vec<MortonKey> {
    let domain = root_box();
    points
        .iter()
        .map(|&p| morton::from_physical_point(p, &domain, DEEPEST))
        .collect()
}

/// The shallowest level whose boxes are at most c_leaf λ wide, λ = 2π/k (root side 1);
/// `None` for no width limit.
pub fn width_level(c_leaf: Option<f64>, k: f64) -> Option<usize> {
    let c = c_leaf?;
    let lambda = 2.0 * std::f64::consts::PI / k;
    (0..=DEEPEST).find(|&l| 0.5f64.powi(l as i32) <= c * lambda)
}

/// The leaves of the complete tree refined by more than `max_keys` distinct keys, or by a
/// key in a box shallower than `min_level`, never below `max_level`. `sorted` holds
/// distinct level-16 keys in ascending order.
pub fn refine(
    sorted: &[MortonKey],
    max_keys: usize,
    min_level: Option<usize>,
    max_level: usize,
) -> Vec<MortonKey> {
    let mut leaves = Vec::new();
    refine_box(
        morton::root(),
        sorted,
        max_keys,
        min_level.unwrap_or(0),
        max_level,
        &mut leaves,
    );
    leaves
}

fn refine_box(
    key: MortonKey,
    keys: &[MortonKey],
    max_keys: usize,
    min_level: usize,
    max_level: usize,
    leaves: &mut Vec<MortonKey>,
) {
    let level = morton::level(key);
    let split = keys.len() > max_keys || (!keys.is_empty() && level < min_level);
    if level >= max_level || !split {
        leaves.push(key);
        return;
    }
    // Sorted level-16 keys have ascending ancestors on every level.
    let mut rest = keys;
    for child in morton::children(key).unwrap() {
        let n =
            rest.partition_point(|&k| morton::ancestor_at_level(k, level + 1).unwrap() == child);
        let (mine, others) = rest.split_at(n);
        refine_box(child, mine, max_keys, min_level, max_level, leaves);
        rest = others;
    }
}

/// The first level-16 descendant of `key`.
pub fn first_descendant(key: MortonKey) -> MortonKey {
    (key & !LEVEL_MASK) | DEEPEST_LEVEL
}

/// The options of the Laplace FMM's tree (`nd-fmm-exec`'s `Fmm`, step 4), with `max_keys`.
fn options(max_keys: usize) -> OctreeOptions {
    OctreeOptions::new()
        .with_max_level(DEEPEST)
        .with_max_fine_keys(max_keys)
        .with_ghost_children(true)
}

/// nd-octree's own tree of the points: `max_keys` points per leaf, no width limit.
pub fn octree_of_points<'c, C: CommunicatorCollectives>(
    keys: &[MortonKey],
    max_keys: usize,
    comm: &'c C,
) -> Octree<'c, C> {
    Octree::new(keys, options(max_keys), comm)
}

/// The prototype: the tree of `refine` with the width rule, balanced by nd-octree.
pub fn octree_with_width<'c, C: CommunicatorCollectives>(
    keys: &[MortonKey],
    max_keys: usize,
    min_level: Option<usize>,
    comm: &'c C,
) -> Octree<'c, C> {
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let leaves = refine(&sorted, max_keys, min_level, DEEPEST);
    let seeds: Vec<MortonKey> = leaves.into_iter().map(first_descendant).collect();
    Octree::new(&seeds, options(1), comm)
}

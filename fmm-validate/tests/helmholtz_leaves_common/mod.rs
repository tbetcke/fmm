//! The check of the Helmholtz leaf rule on one tree (Helmholtz CH3.0b), shared by
//! `tests/helmholtz_leaves.rs` and `tests/helmholtz_leaves_sweep.rs`.
//!
//! The tree is the one `Fmm`'s step 4 builds (`max_level` 16, the host's default leaf
//! size at p ≤ 11, the ghost-children layer) with
//! `OctreeOptions::with_min_leaf_level(min_leaf_level(k, domain, C_LEAF))`, on
//! `SimpleCommunicator::self_comm()`: each rank checks whole trees alone, so the check
//! takes no collective and runs the same on any number of ranks. The leaves on several
//! ranks are the one-rank leaves (CH3.0a, `nd-octree`'s `test_mpi_weighted_partition`).

use std::collections::BTreeSet;

use mpi::topology::SimpleCommunicator;
use nd_fmm_exec::fmm::{Backend, default_max_points_per_leaf};
use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::{C_LEAF, Wavenumber, min_leaf_level};
use nd_fmm_validate::helmholtz_workloads::DOMAIN;
use nd_octree::constants::DEEPEST_LEVEL;
use nd_octree::{MortonKey, Octree, OctreeOptions, PhysicalBox, morton};

/// The deepest level, 16.
pub const DEEPEST: usize = DEEPEST_LEVEL as usize;

/// The workloads' domain, the root cube (D = 1).
pub fn domain() -> Domain {
    Domain::new(&PhysicalBox::new(DOMAIN)).expect("the root cube is a cubic domain")
}

/// The level-16 key of every point, as `points_to_morton` computes it in `Fmm`.
pub fn fine_keys(points: &[[f64; 3]]) -> Vec<MortonKey> {
    let domain = domain().physical_box();
    points
        .iter()
        .map(|&p| morton::from_physical_point(p, &domain, DEEPEST))
        .collect()
}

/// `Fmm`'s tree options (step 4), without the leaf rule.
pub fn fmm_options() -> OctreeOptions {
    OctreeOptions::new()
        .with_max_level(DEEPEST)
        .with_max_fine_keys(default_max_points_per_leaf(8, Backend::Host))
        .with_ghost_children(true)
}

/// The tree of `keys` with the leaf rule of `k`.
pub fn tree<'c>(
    keys: &[MortonKey],
    k: &Wavenumber,
    comm: &'c SimpleCommunicator,
) -> Octree<'c, SimpleCommunicator> {
    let level = min_leaf_level(k, &domain(), C_LEAF);
    Octree::new(keys, fmm_options().with_min_leaf_level(level), comm)
}

/// What one tree under the rule shows.
#[derive(Clone, Debug)]
pub struct Leaves {
    /// The rule's level, `min_leaf_level`.
    pub rule: usize,
    /// The leaves of the tree, every one.
    pub leaves: usize,
    /// The leaves that hold a point.
    pub occupied: usize,
    /// The shallowest and deepest level of an occupied leaf.
    pub levels: (usize, usize),
    /// The widest occupied leaf in wavelengths, w 2⁻ˡ / λ; `None` for Yukawa.
    pub widest: Option<f64>,
}

/// Builds the tree of `keys` under the rule of `k` and checks proposal §3.7: every leaf
/// that holds a point is at most c_leaf λ wide, w 2⁻ˡ ≤ c_leaf λ (the arithmetic of
/// `min_leaf_level`). Panics, naming `name`, if one is wider.
pub fn check(name: &str, keys: &[MortonKey], k: &Wavenumber) -> Leaves {
    let comm = SimpleCommunicator::self_comm();
    let tree = tree(keys, k, &comm);
    let rule = min_leaf_level(k, &domain(), C_LEAF);
    let occupied: BTreeSet<MortonKey> = keys
        .iter()
        .map(|&key| {
            tree.local_leaf(key)
                .expect("a valid key")
                .expect("one rank holds every leaf")
        })
        .collect();
    let levels = occupied.iter().map(|&leaf| morton::level(leaf));
    let shallowest = levels.clone().min().expect("some point");
    let deepest = levels.max().expect("some point");
    assert!(
        shallowest >= rule,
        "{name}, k = {} + {}i: an occupied leaf on level {shallowest}, above the rule's {rule}",
        k.re(),
        k.im()
    );
    let width = domain().side() * 0.5f64.powi(shallowest as i32);
    if let Some(lambda) = k.wavelength() {
        assert!(
            width <= C_LEAF * lambda,
            "{name}, k = {} + {}i: an occupied leaf {} λ wide",
            k.re(),
            k.im(),
            width / lambda
        );
    }
    Leaves {
        rule,
        leaves: tree.leaf_keys().len(),
        occupied: occupied.len(),
        levels: (shallowest, deepest),
        widest: k.wavelength().map(|lambda| width / lambda),
    }
}

/// The leaves of the tree of `keys` under the rule of `k`, sorted.
#[allow(
    dead_code,
    reason = "tests/helmholtz_leaves_sweep.rs checks the rule only"
)]
pub fn leaf_keys(keys: &[MortonKey], k: &Wavenumber) -> Vec<MortonKey> {
    let comm = SimpleCommunicator::self_comm();
    let mut leaves = tree(keys, k, &comm).leaf_keys().clone();
    leaves.sort_unstable();
    leaves
}

/// The leaves of `Fmm`'s tree of `keys` without the option, sorted.
#[allow(
    dead_code,
    reason = "tests/helmholtz_leaves_sweep.rs checks the rule only"
)]
pub fn leaf_keys_without_rule(keys: &[MortonKey]) -> Vec<MortonKey> {
    let comm = SimpleCommunicator::self_comm();
    let mut leaves = Octree::new(keys, fmm_options(), &comm).leaf_keys().clone();
    leaves.sort_unstable();
    leaves
}

//! Run the index-propagating test FMM on random points, with one point per leaf and with
//! random counts per leaf.
//!
//! With one point per leaf every leaf must end up with every leaf index exactly once.
//! With random counts every target point of every leaf must hold, at the index of each
//! leaf, the number of source points of that leaf. Both are checked through the per-pair
//! adapter and through both walks of the batched operator. Run with, e.g.,
//! `mpirun -n 4 target/debug/examples/test_index_fmm`.

use mpi::traits::Communicator;
use nd_fmm_plan::{
    index_fmm::{IndexPath, Walk, run_index_fmm},
    plan::Plan,
};
use nd_octree::{Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, points_to_morton};
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{Rng, SeedableRng},
};
use rlst::{println_mpi, rlst_dynamic_array};

/// A random number of points: zero for about one leaf in eight, otherwise 1 to 4, so that
/// zero counts occur but rarely hide an interaction.
fn random_count(rng: &mut ChaCha8Rng) -> usize {
    match rng.next_u32() % 8 {
        0 => 0,
        r => 1 + r as usize % 4,
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let mut rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);

    let npoints = 200;
    let mut sources = rlst_dynamic_array!(f64, [3, npoints]);
    let mut targets = rlst_dynamic_array!(f64, [3, npoints]);
    sources.fill_from_equally_distributed(&mut rng);
    targets.fill_from_equally_distributed(&mut rng);

    // One tree serves both populations. The ghost-children layer makes every
    // interaction-list entry a key of the tree.
    let bounding_box = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let mut fine_keys = points_to_morton(&sources, DEEPEST_LEVEL as usize, &bounding_box);
    fine_keys.extend(points_to_morton(
        &targets,
        DEEPEST_LEVEL as usize,
        &bounding_box,
    ));
    let options = OctreeOptions::new()
        .with_max_level(DEEPEST_LEVEL as usize)
        .with_max_fine_keys(10)
        .with_ghost_children(true);
    let octree = Octree::new(&fine_keys, options, &comm);

    let plan = Plan::new(&octree).unwrap_or_else(|error| panic!("plan: {error}"));
    let nlocal = plan.index().leaves().nlocal();
    let ones = vec![1; nlocal];
    let source_counts: Vec<usize> = (0..nlocal).map(|_| random_count(&mut rng)).collect();
    let target_counts: Vec<usize> = (0..nlocal).map(|_| random_count(&mut rng)).collect();
    for (label, sources, targets) in [
        ("one point per leaf", &ones, &ones),
        ("random counts", &source_counts, &target_counts),
    ] {
        for path in [
            IndexPath::PerPair,
            IndexPath::Batched(Walk::Rows),
            IndexPath::Batched(Walk::Groupings),
        ] {
            match run_index_fmm(&plan, &comm, path, sources, targets) {
                Ok(()) => println_mpi!(
                    comm.rank(),
                    "Index FMM ({label}, {path:?}) correct on {nlocal} local leaves, {} source and {} target points",
                    sources.iter().sum::<usize>(),
                    targets.iter().sum::<usize>()
                ),
                Err(message) => panic!(
                    "rank {}: index FMM ({label}, {path:?}) failed: {message}",
                    comm.rank()
                ),
            }
        }
    }
}

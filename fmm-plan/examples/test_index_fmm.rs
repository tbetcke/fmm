//! Run the index-propagating test FMM on random points, with the old evaluator and with
//! the new one with random counts per leaf.
//!
//! With the old evaluator every leaf must end up with every leaf index exactly once.
//! With the new one every target point of every leaf must hold, at the index of each
//! leaf, the number of source points of that leaf, through the per-pair adapter and
//! through both walks of the batched operator. Run with, e.g.,
//! `mpirun -n 4 target/debug/examples/test_index_fmm`.

use mpi::traits::Communicator;
use nd_fmm_plan::fmm::index_fmm::run_index_fmm;
use nd_fmm_plan::interaction_manager::InteractionManager;
use nd_fmm_plan::v2::{
    index_fmm::{self, IndexPath, Walk},
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
    let lists = InteractionManager::new(&octree);

    match run_index_fmm(&octree, &lists) {
        Ok(()) => println_mpi!(
            comm.rank(),
            "Old index FMM correct on {} local leaves",
            octree.leaf_keys().len()
        ),
        Err(message) => panic!("rank {}: old index FMM failed: {message}", comm.rank()),
    }

    // The new evaluator, with random source and target counts per local leaf.
    let plan = Plan::new(&octree).unwrap_or_else(|error| panic!("plan: {error}"));
    let nlocal = plan.index().leaves().nlocal();
    let source_counts: Vec<usize> = (0..nlocal).map(|_| random_count(&mut rng)).collect();
    let target_counts: Vec<usize> = (0..nlocal).map(|_| random_count(&mut rng)).collect();
    for path in [
        IndexPath::PerPair,
        IndexPath::Batched(Walk::Rows),
        IndexPath::Batched(Walk::Groupings),
    ] {
        match index_fmm::run_index_fmm(&plan, &comm, path, &source_counts, &target_counts) {
            Ok(()) => println_mpi!(
                comm.rank(),
                "New index FMM ({path:?}) correct on {nlocal} local leaves, {} source and {} target points",
                source_counts.iter().sum::<usize>(),
                target_counts.iter().sum::<usize>()
            ),
            Err(message) => panic!(
                "rank {}: new index FMM ({path:?}) failed: {message}",
                comm.rank()
            ),
        }
    }
}

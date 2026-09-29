//! Run the index-propagating test FMM on random points.
//!
//! Every leaf must end up with every leaf index exactly once. Run with, e.g.,
//! `mpirun -n 4 target/debug/examples/test_index_fmm`.

use nd_fmm_plan::fmm::index_fmm::run_index_fmm;
use nd_fmm_plan::interaction_manager::InteractionManager;
use mpi::traits::Communicator;
use nd_octree::{Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, points_to_morton};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rlst::{println_mpi, rlst_dynamic_array};

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
            "Index FMM correct on {} local leaves",
            octree.leaf_keys().len()
        ),
        Err(message) => panic!("rank {}: index FMM failed: {message}", comm.rank()),
    }
}

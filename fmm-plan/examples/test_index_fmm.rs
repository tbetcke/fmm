//! Run the index-propagating test FMM on random points.
//!
//! Every leaf must end up with every leaf index exactly once. Run with, e.g.,
//! `mpirun -n 4 target/debug/examples/test_index_fmm`.

use std::rc::Rc;

use fmm_plan::fmm::index_fmm::run_index_fmm;
use fmm_plan::fmm_tree::FmmTree;
use fmm_plan::interaction_manager::InteractionManager;
use mpi::traits::Communicator;
use nd_octree::{OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, points_to_morton};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rlst::{
    Array, distributed_tools::IndexLayout, println_mpi, rlst_dynamic_array,
    sparse::distributed_array::DistributedArray,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let mut rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);

    let npoints = 200;
    let mut sources = rlst_dynamic_array!(f64, [3, npoints]);
    let mut targets = rlst_dynamic_array!(f64, [3, npoints]);
    sources.fill_from_equally_distributed(&mut rng);
    targets.fill_from_equally_distributed(&mut rng);

    let bounding_box = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let distribute = |points| {
        let keys: Array<_, _> =
            points_to_morton(points, DEEPEST_LEVEL as usize, &bounding_box).into();
        let layout = Rc::new(IndexLayout::from_local_counts(keys.len(), &comm));
        DistributedArray::new(layout, keys)
    };
    let sources = distribute(&sources);
    let targets = distribute(&targets);

    let fmm_tree = FmmTree::new(
        &sources,
        &targets,
        OctreeOptions::new()
            .with_max_level(DEEPEST_LEVEL as usize)
            .with_max_fine_keys(10),
        &comm,
    );
    let lists = InteractionManager::new(fmm_tree.octree());

    match run_index_fmm(fmm_tree.octree(), &lists) {
        Ok(()) => println_mpi!(
            comm.rank(),
            "Index FMM correct on {} local leaves",
            fmm_tree.octree().leaf_keys().len()
        ),
        Err(message) => panic!("rank {}: index FMM failed: {message}", comm.rank()),
    }
}

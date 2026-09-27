//! Bounded MPI regressions for small construction inputs.

use mpi::traits::Communicator;
use nd_octree::{
    morton::{deepest_first, root},
    octree::{KeyType, Octree, OctreeOptions},
};

#[test]
fn regression_single_rank_root_only_tree_has_empty_neighbour_list() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    assert_eq!(comm.size(), 1, "run this regression with one MPI rank");

    let tree = Octree::new(&[deepest_first()], OctreeOptions::default(), &comm);

    assert_eq!(tree.leaf_keys(), &vec![root()]);
    assert_eq!(tree.all_keys().get(&root()), Some(&KeyType::LocalLeaf));
    assert_eq!(tree.neighbour_map().get(&root()), Some(&Vec::new()));
}

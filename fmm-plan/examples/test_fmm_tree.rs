//! Test the creation of an FMM tree

use std::rc::Rc;

use fmm_plan::fmm_tree::FmmTree;
use fmm_plan::interaction_manager::InteractionManager;
use itertools::{Itertools, izip};
use mpi::collective::SystemOperation;
use mpi::traits::{Communicator, CommunicatorCollectives};
use nd_octree::PhysicalBox;
use nd_octree::morton::is_ancestor;
use nd_octree::{OctreeOptions, constants::DEEPEST_LEVEL, points_to_morton};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rlst::{
    Array, distributed_tools::IndexLayout, println_mpi, rlst_dynamic_array,
    sparse::distributed_array::DistributedArray,
};

fn main() {
    // Initialise MPI
    let universe = mpi::initialize().unwrap();

    // Get the world communicator
    let comm = universe.world();

    // Initialise a seeded Rng.
    let mut rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);

    let npoints_sources = 100;
    let npoints_targets = 200;

    // Generate random points.

    let mut points_sources = rlst_dynamic_array!(f64, [3, npoints_sources]);
    let mut points_targets = rlst_dynamic_array!(f64, [3, npoints_targets]);

    points_sources.fill_from_equally_distributed(&mut rng);
    points_targets.fill_from_equally_distributed(&mut rng);

    let bounding_box = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let source_fine_keys: Array<_, _> =
        points_to_morton(&points_sources, DEEPEST_LEVEL as usize, &bounding_box).into();

    let target_fine_keys: Array<_, _> =
        points_to_morton(&points_targets, DEEPEST_LEVEL as usize, &bounding_box).into();

    let index_layout_sources = Rc::new(IndexLayout::from_local_counts(
        source_fine_keys.len(),
        &comm,
    ));

    let index_layout_targets = Rc::new(IndexLayout::from_local_counts(
        target_fine_keys.len(),
        &comm,
    ));

    let distributed_source_fine_keys =
        DistributedArray::new(index_layout_sources.clone(), source_fine_keys);

    let distributed_target_fine_keys =
        DistributedArray::new(index_layout_targets.clone(), target_fine_keys);

    let fmm_tree = FmmTree::new(
        &distributed_source_fine_keys,
        &distributed_target_fine_keys,
        OctreeOptions::new()
            .with_max_level(DEEPEST_LEVEL as usize)
            .with_max_fine_keys(10),
        &comm,
    );

    let n_actual_sources = *fmm_tree.source_indptr().last().unwrap();
    let n_actual_targets = *fmm_tree.target_indptr().last().unwrap();

    let mut n_actual_all_sources = 0;
    let mut n_actual_all_targets = 0;

    comm.all_reduce_into(
        &n_actual_sources,
        &mut n_actual_all_sources,
        SystemOperation::sum(),
    );

    comm.all_reduce_into(
        &n_actual_targets,
        &mut n_actual_all_targets,
        SystemOperation::sum(),
    );

    assert_eq!(n_actual_all_sources, npoints_sources * comm.size() as usize);
    assert_eq!(n_actual_all_targets, npoints_targets * comm.size() as usize);

    // Now check that the source index pointers are correct.
    // To do this we map the assigned leafs of the sources and targets back to the processes of
    // the original arrays and check that the leafs are ancestors of the fine keys.

    let mut source_leafs_permuted = Vec::<u64>::default();

    // Go through the source leafs and push each leaf the number of times
    // into the `source_leafs` array, identical to how many sources are in the box.
    for (&leaf, (s, e)) in izip!(
        fmm_tree.octree().leaf_keys().iter(),
        fmm_tree.source_indptr().iter().tuple_windows()
    ) {
        for _ in 0..e - s {
            source_leafs_permuted.push(leaf);
        }
    }

    // The source leafs correspond to a permuted ordering across nodes. Reverse the permutation.

    let mut source_leafs = vec![0.0 as u64; npoints_sources];

    fmm_tree
        .source_permutation()
        .backward_permute(&source_leafs_permuted, &mut source_leafs, 1);

    // Check that the back permuted array has the same length as the input fine keys

    assert_eq!(
        source_leafs.len(),
        distributed_source_fine_keys.local.data().unwrap().len()
    );

    // We have everything on our node. We can now check whether each source leaf is an ancestor of
    // the corresponding source fine leaf.

    for (&source_leaf, &fine_source_key) in izip!(
        source_leafs.iter(),
        distributed_source_fine_keys.local.data().unwrap().iter()
    ) {
        assert!(is_ancestor(source_leaf, fine_source_key));
    }

    // Now do the same with the targets

    let mut target_leafs_permuted = Vec::<u64>::default();

    // Go through the target leafs and push each leaf the number of times
    // into the `target_leafs` array, identical to how many targets are in the box.
    for (&leaf, (s, e)) in izip!(
        fmm_tree.octree().leaf_keys().iter(),
        fmm_tree.target_indptr().iter().tuple_windows()
    ) {
        for _ in 0..e - s {
            target_leafs_permuted.push(leaf);
        }
    }

    // The target leafs correspond to a permuted ordering across nodes. Reverse the permutation.

    let mut target_leafs = vec![0.0 as u64; npoints_targets];

    fmm_tree
        .target_permutation()
        .backward_permute(&target_leafs_permuted, &mut target_leafs, 1);

    // Check that the back permuted array has the same length as the input fine keys

    assert_eq!(
        target_leafs.len(),
        distributed_target_fine_keys.local.data().unwrap().len()
    );

    // We have everything on our node. We can now check whether each target leaf is an ancestor of
    // the corresponding target fine leaf.

    for (&target_leaf, &fine_target_key) in izip!(
        target_leafs.iter(),
        distributed_target_fine_keys.local.data().unwrap().iter()
    ) {
        assert!(is_ancestor(target_leaf, fine_target_key));
    }

    println_mpi!(comm.rank(), "Correctly created FMM tree");

    // Build the interaction lists and check their structural guarantees. The
    // full correctness oracle lives in `tests/fmm_tree.rs`.

    let interaction_lists = InteractionManager::new(fmm_tree.octree());

    let mut expected_entries = fmm_tree
        .octree()
        .all_keys()
        .iter()
        .filter(|(_, key_type)| !key_type.is_ghost())
        .map(|(&key, _)| key)
        .collect_vec();
    expected_entries.sort_unstable();

    for list in [
        interaction_lists.u_list(),
        interaction_lists.v_list(),
        interaction_lists.w_list(),
        interaction_lists.x_list(),
    ] {
        let mut entries = list.keys().copied().collect_vec();
        entries.sort_unstable();
        assert_eq!(entries, expected_entries);

        for (&key, entry) in list.iter() {
            assert!(entry.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(!entry.contains(&key));
        }
    }

    println_mpi!(comm.rank(), "Correctly created interaction lists");
}

//! Exercise collective finest-level leaf lookup.

use itertools::Itertools;
use mpi::traits::Communicator;
use nd_octree::{
    LeafLocation, LookupError, Octree, OctreeOptions,
    constants::DEEPEST_LEVEL,
    morton::{self, MortonKey},
};
use rlst::distributed_tools::array_tools::gather_to_all;

fn representative(key: MortonKey) -> MortonKey {
    let (level, index) = morton::decode(key);
    let shift = DEEPEST_LEVEL as usize - level;
    morton::from_index_and_level(
        [index[0] << shift, index[1] << shift, index[2] << shift],
        16,
    )
}

fn oracle(key: MortonKey, leaves: &[(MortonKey, usize)]) -> LeafLocation {
    let (leaf, owner_rank) = leaves
        .iter()
        .find(|(leaf, _)| morton::is_ancestor(*leaf, key))
        .copied()
        .expect("complete fixture must contain every finest key");
    LeafLocation { leaf, owner_rank }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let rank = comm.rank() as usize;

    // A dense region plus sparse points creates deterministic mixed refinement.
    let mut fine_keys = Vec::new();
    for i in 0..96 {
        fine_keys.push(morton::from_index_and_level(
            [1000 + (i % 4), 2000 + ((i / 4) % 4), 3000 + ((i / 16) % 4)],
            16,
        ));
    }
    for i in 0..16 {
        let offset = rank * 4096 + i * 197;
        fine_keys.push(morton::from_index_and_level(
            [offset % 65536, (offset * 3) % 65536, (offset * 7) % 65536],
            16,
        ));
    }

    let options = OctreeOptions::new()
        .with_max_level(12)
        .with_max_fine_keys(4);
    let tree = Octree::new(&fine_keys, options, &comm);
    let leaves = gather_to_all(tree.leaf_keys(), &comm);
    let leaf_ranks = gather_to_all(&vec![rank; tree.leaf_keys().len()], &comm);
    let owned_leaves = leaves.into_iter().zip(leaf_ranks).collect_vec();
    assert!(
        owned_leaves
            .iter()
            .map(|(key, _)| morton::level(*key))
            .unique()
            .count()
            > 1
    );

    let representatives = owned_leaves
        .iter()
        .map(|(leaf, _)| representative(*leaf))
        .collect_vec();
    let mut queries = vec![morton::deepest_first(), morton::deepest_last()];
    // Test both sides of each rank partition boundary at finest level.
    for &bound in tree.coarse_tree_bounds().iter().skip(1) {
        let boundary = representative(bound);
        queries.push(boundary - (1 << 15));
        queries.push(boundary);
    }
    queries.extend(representatives.iter().copied().take(24));
    queries.push(representatives[0]);
    queries.insert(2, morton::root());
    queries.insert(5, morton::invalid_key());

    let results = tree.lookup_leaves(&queries).unwrap();
    assert_eq!(results.len(), queries.len());
    for (&query, result) in queries.iter().zip(&results) {
        match tree.owner_rank(query) {
            Ok(owner) => {
                assert_eq!(
                    tree.local_leaf(query).unwrap(),
                    (owner == rank).then(|| oracle(query, &owned_leaves).leaf)
                );
                assert_eq!(*result, Ok(oracle(query, &owned_leaves)));
            }
            Err(error) => assert_eq!(*result, Err(error)),
        }
    }

    // All ranks participate in empty calls, and only one rank may issue work.
    assert!(tree.lookup_leaves(&[]).unwrap().is_empty());
    let one_rank_queries = if rank == 0 {
        representatives.as_slice()
    } else {
        &[]
    };
    let one_rank_results = tree.lookup_leaves(one_rank_queries).unwrap();
    for (&query, result) in one_rank_queries.iter().zip(one_rank_results) {
        assert_eq!(result, Ok(oracle(query, &owned_leaves)));
    }
    if comm.size() > 1 {
        // The first rank after rank 0 that owns leaves has only invalid input but must
        // still serve rank 0's requests. A rank may own no leaves (an empty range), so
        // it is not always rank 1; every rank picks the same one from the gathered
        // leaves.
        let server = owned_leaves
            .iter()
            .map(|&(_, owner)| owner)
            .find(|&owner| owner != 0)
            .expect("a rank other than 0 owns leaves");
        let server_queries = representatives
            .iter()
            .copied()
            .filter(|&query| tree.owner_rank(query).unwrap() == server)
            .take(4)
            .collect_vec();
        assert!(!server_queries.is_empty());
        let invalid_only = if rank == server {
            &[morton::root()][..]
        } else if rank == 0 {
            server_queries.as_slice()
        } else {
            &[]
        };
        let invalid_only_results = tree.lookup_leaves(invalid_only).unwrap();
        assert_eq!(invalid_only_results.len(), invalid_only.len());
        if rank == server {
            assert_eq!(
                invalid_only_results,
                vec![Err(LookupError::NotFinestLevel { level: 0 })]
            );
        } else if rank == 0 {
            for (&query, result) in invalid_only.iter().zip(invalid_only_results) {
                assert_eq!(tree.owner_rank(query), Ok(server));
                assert_eq!(result, Ok(oracle(query, &owned_leaves)));
            }
        }
    }

    if comm.size() > 1 {
        assert!(
            representatives
                .iter()
                .any(|&query| tree.owner_rank(query).unwrap() != rank)
        );
    }
}

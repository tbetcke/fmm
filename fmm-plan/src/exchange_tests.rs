//! Pure local tests of the ghost bucketing and the per-key chunk sizes of the exchanges,
//! on the synthetic key maps of `plan_tests.rs`. No MPI.
use nd_octree::{MortonKey, morton, octree::KeyType};

use super::{
    ExchangeError, displacements, ghost_counts, level_sizes, multipole_ghosts, owned_chunk_size,
    source_ghosts,
};
use crate::interaction_manager::tests::ListMaps;
use crate::plan::tests::{cases, plan};

/// A source count per key that every rank can recompute; zero for about one key in five.
fn count_of(key: MortonKey) -> usize {
    (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize % 5
}

fn is_ghost_case(name: &str) -> bool {
    name.ends_with("ghost half")
}

#[test]
fn source_ghosts_are_the_named_ghost_leaves_in_leaf_order() {
    let mut ghosts_seen = 0;
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        let leaves = index.leaves();
        let ghosts = source_ghosts(index);
        let expected: Vec<MortonKey> = leaves.ghosts().map(|j| leaves.key(j)).collect();
        let keys: Vec<MortonKey> = ghosts.iter().map(|&(key, _)| key).collect();
        assert_eq!(keys, expected, "{name}");
        assert!(keys.windows(2).all(|pair| pair[0] < pair[1]), "{name}");
        for &(key, owner) in &ghosts {
            assert_eq!(
                map[&key],
                KeyType::GhostLeaf(owner),
                "{name}: owner of {key}"
            );
        }
        if is_ghost_case(&name) {
            ghosts_seen += ghosts.len();
        } else {
            assert!(ghosts.is_empty(), "{name}");
        }
    }
    assert!(ghosts_seen > 0, "no case has a ghost leaf");
}

#[test]
fn owners_size_each_chunk_by_their_own_count() {
    let mut zeros = 0;
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        let leaves = index.leaves();
        let counts: Vec<usize> = (0..leaves.nlocal())
            .map(|j| count_of(leaves.key(j)))
            .collect();
        zeros += counts.iter().filter(|&&count| count == 0).count();
        for point_size in [1, 4] {
            let size = |key| owned_chunk_size(index, &counts, point_size, key);
            for j in 0..leaves.nlocal() {
                let key = leaves.key(j);
                assert_eq!(size(key), Some(point_size * count_of(key)), "{name}: {key}");
            }
            // Only local leaves are sized: ghosts, interiors and unheld keys are not.
            for (&key, &kind) in &map {
                if kind != KeyType::LocalLeaf {
                    assert_eq!(size(key), None, "{name}: {key} is not a local leaf");
                }
            }
            assert_eq!(size(morton::from_index_and_level([0, 0, 0], 9)), None);
            assert_eq!(size(morton::invalid_key()), None);
        }
    }
    assert!(zeros > 0, "no zero count occurs");
}

#[test]
fn ghost_counts_divide_the_received_chunk_sizes() {
    assert_eq!(ghost_counts(&[8, 0, 4], 4), Some(vec![2, 0, 1]));
    assert_eq!(ghost_counts(&[3], 1), Some(vec![3]));
    assert_eq!(ghost_counts(&[8, 3], 4), None);
    assert_eq!(ghost_counts(&[], 4), Some(vec![]));
}

#[test]
fn multipole_ghosts_are_the_ghost_entries_of_v_and_w_rows() {
    let mut ghosts_seen = 0;
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        let lists = ListMaps::new(&map);
        for level in 0..plan.nlevels() {
            let mut expected: Vec<MortonKey> = Vec::new();
            for (&key, &kind) in &map {
                if kind.is_ghost() {
                    continue;
                }
                if morton::level(key) == level {
                    expected.extend(&lists.v_list()[&key]);
                }
                if morton::level(key) + 1 == level {
                    expected.extend(&lists.w_list()[&key]);
                }
            }
            expected.retain(|entry| map[entry].is_ghost());
            expected.sort_unstable();
            expected.dedup();

            let ghosts = multipole_ghosts(&plan, level);
            assert!(
                ghosts.windows(2).all(|pair| pair[0].0 < pair[1].0),
                "{name}"
            );
            let keys: Vec<MortonKey> = ghosts
                .iter()
                .map(|&(i, _)| index.key(level, i as usize))
                .collect();
            assert_eq!(keys, expected, "{name}: level {level}");
            for (&key, &(_, owner)) in keys.iter().zip(&ghosts) {
                assert_eq!(
                    map[&key].ghost_rank(),
                    Some(owner),
                    "{name}: owner of {key}"
                );
            }
            ghosts_seen += ghosts.len();
        }
    }
    assert!(ghosts_seen > 0, "no case has a ghost multipole");
}

#[test]
fn level_sizes_need_one_positive_entry_per_level() {
    assert_eq!(level_sizes(&[1, 4, 9], 3), (Ok(()), vec![1, 4, 9]));
    assert_eq!(
        level_sizes(&[1, 4], 3),
        (
            Err(ExchangeError::SizesLength {
                expected: 3,
                actual: 2
            }),
            vec![1, 4, 0]
        )
    );
    assert_eq!(level_sizes(&[1, 4, 9, 16], 3).1, vec![1, 4, 9]);
    assert_eq!(level_sizes(&[1, 0, 9], 3).0, Err(ExchangeError::ZeroSize));
    assert_eq!(level_sizes(&[], 0), (Ok(()), vec![]));
}

#[test]
fn displacements_are_exclusive_prefix_sums_within_i32() {
    assert_eq!(displacements(&[2, 0, 3]), Some(vec![0, 2, 2]));
    assert_eq!(displacements(&[]), Some(vec![]));
    assert_eq!(displacements(&[i32::MAX, 0]), Some(vec![0, i32::MAX]));
    assert_eq!(displacements(&[i32::MAX, 1, 1]), None);
}

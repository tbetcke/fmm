//! Pure local tests of the ghost bucketing; no MPI.
use super::{LevelChunkSizes, bucket_ghosts};
use nd_octree::{MortonKey, morton, octree::KeyType};
use std::collections::HashMap;

fn key(index: [usize; 3], level: usize) -> MortonKey {
    morton::from_index_and_level(index, level)
}

/// A small classification map with every kind of key on levels 0 to 2.
fn key_map() -> HashMap<MortonKey, KeyType> {
    HashMap::from([
        (key([0, 0, 0], 0), KeyType::Global),
        (key([0, 0, 0], 1), KeyType::Global),
        (key([1, 0, 0], 1), KeyType::GhostInterior(3)),
        (key([0, 1, 0], 1), KeyType::GhostLeaf(1)),
        (key([0, 0, 1], 1), KeyType::LocalLeaf),
        (key([0, 0, 0], 2), KeyType::LocalLeaf),
        (key([1, 1, 1], 2), KeyType::LocalInterior),
        (key([2, 0, 0], 2), KeyType::GhostLeaf(3)),
        (key([3, 0, 0], 2), KeyType::GhostLeaf(2)),
    ])
}

#[test]
fn ghosts_are_bucketed_by_level_with_their_owner() {
    let all_keys = key_map();
    let keys = all_keys.keys().copied();
    let buckets = bucket_ghosts(&all_keys, keys, 3).unwrap();
    assert_eq!(buckets.len(), 3);
    assert!(buckets[0].is_empty());
    let mut level_one = vec![(key([1, 0, 0], 1), 3), (key([0, 1, 0], 1), 1)];
    level_one.sort_unstable();
    assert_eq!(buckets[1], level_one);
    let mut level_two = vec![(key([2, 0, 0], 2), 3), (key([3, 0, 0], 2), 2)];
    level_two.sort_unstable();
    assert_eq!(buckets[2], level_two);
}

#[test]
fn local_and_global_keys_are_skipped() {
    let all_keys = key_map();
    let keys = [
        key([0, 0, 0], 0),
        key([0, 0, 0], 1),
        key([0, 0, 1], 1),
        key([0, 0, 0], 2),
        key([1, 1, 1], 2),
    ];
    let buckets = bucket_ghosts(&all_keys, keys, 3).unwrap();
    assert!(buckets.iter().all(Vec::is_empty));
}

#[test]
fn duplicates_are_removed_and_buckets_sorted() {
    let all_keys = key_map();
    let a = key([3, 0, 0], 2);
    let b = key([2, 0, 0], 2);
    let buckets = bucket_ghosts(&all_keys, [a, b, a, a, b], 3).unwrap();
    let mut expected = vec![(a, 2), (b, 3)];
    expected.sort_unstable();
    assert_eq!(buckets[2], expected);
}

#[test]
fn unknown_or_too_deep_keys_are_rejected() {
    let all_keys = key_map();
    assert!(bucket_ghosts(&all_keys, [key([3, 3, 3], 2)], 3).is_none());
    // A known ghost on a level beyond the requested number of levels.
    assert!(bucket_ghosts(&all_keys, [key([2, 0, 0], 2)], 2).is_none());
    assert!(bucket_ghosts(&all_keys, [], 0).unwrap().is_empty());
}

#[test]
fn level_chunk_sizes() {
    assert_eq!(LevelChunkSizes::Uniform(4).chunk_size(0), Some(4));
    assert_eq!(LevelChunkSizes::Uniform(4).chunk_size(16), Some(4));
    let per_level = LevelChunkSizes::PerLevel(vec![1, 2, 3]);
    assert_eq!(per_level.chunk_size(0), Some(1));
    assert_eq!(per_level.chunk_size(2), Some(3));
    assert_eq!(per_level.chunk_size(3), None);
}

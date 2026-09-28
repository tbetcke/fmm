//! Pure local tests of the level storage; no MPI.
use super::LevelData;
use nd_octree::morton;

fn key(index: [usize; 3], level: usize) -> u64 {
    morton::from_index_and_level(index, level)
}

#[test]
fn chunks_have_the_size_of_their_level_and_start_zeroed() {
    let keys = [key([0, 0, 0], 0), key([1, 0, 0], 1), key([0, 1, 0], 1)];
    let data = LevelData::<u32>::new(keys, vec![2, 3]);
    assert_eq!(data.get(keys[0]), Some(&[0, 0][..]));
    assert_eq!(data.get(keys[1]), Some(&[0, 0, 0][..]));
    assert!(data.contains(keys[2]));
    assert!(data.get(key([1, 1, 0], 1)).is_none());
}

#[test]
fn chunks_are_disjoint_and_duplicates_are_stored_once() {
    let a = key([1, 0, 0], 1);
    let b = key([0, 1, 0], 1);
    let mut data = LevelData::<u32>::new([a, b, a], vec![1, 2]);
    data.get_mut(a).unwrap().copy_from_slice(&[1, 2]);
    data.get_mut(b).unwrap().copy_from_slice(&[3, 4]);
    assert_eq!(data.get(a), Some(&[1, 2][..]));
    assert_eq!(data.get(b), Some(&[3, 4][..]));
    data.clear();
    assert_eq!(data.get(b), Some(&[0, 0][..]));
}

#[test]
fn coarse_and_fine_chunks_are_borrowed_together() {
    let parent = key([0, 0, 0], 1);
    let child = key([1, 1, 1], 2);
    let mut data = LevelData::<u32>::new([parent, child], vec![1, 2, 3]);
    data.get_mut(child).unwrap().copy_from_slice(&[1, 2, 3]);
    let (coarse, fine) = data.coarse_fine_mut(parent, child);
    coarse.copy_from_slice(&fine[..2]);
    fine[0] = 7;
    assert_eq!(data.get(parent), Some(&[1, 2][..]));
    assert_eq!(data.get(child), Some(&[7, 2, 3][..]));
}

#[test]
#[should_panic(expected = "different levels")]
fn coarse_and_fine_chunks_on_one_level_are_rejected() {
    let a = key([1, 0, 0], 1);
    let b = key([0, 1, 0], 1);
    let mut data = LevelData::<u32>::new([a, b], vec![1, 1]);
    data.coarse_fine_mut(a, b);
}

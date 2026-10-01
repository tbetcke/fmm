//! Pure local tests of the level buffers and leaf stores. No MPI.
use super::{LeafStore, LevelBuffers};

#[test]
fn level_chunks_lie_at_box_index_times_size() {
    let lens = [1, 8, 5];
    let sizes = [4, 2, 3];
    let mut buffers = LevelBuffers::<u64>::new(&lens, &sizes);
    assert_eq!(buffers.nlevels(), 3);
    assert_eq!(buffers.offsets(), [0, 4, 20, 35]);
    assert_eq!(buffers.as_slice().len(), 35);
    for level in 0..3 {
        assert_eq!(buffers.len(level), lens[level]);
        assert_eq!(buffers.size(level), sizes[level]);
        let mut slice = buffers.level_mut(level);
        assert_eq!(slice.len(), lens[level]);
        assert_eq!(slice.size(), sizes[level]);
        for (i, chunk) in slice.chunks_mut().enumerate() {
            for (k, value) in chunk.iter_mut().enumerate() {
                *value = (100 * level + 10 * i + k) as u64;
            }
        }
    }
    // Box i of level l starts at offset(l) + i * size(l) of the one buffer.
    for level in 0..3 {
        let slice = buffers.level(level);
        for i in 0..lens[level] {
            let start = buffers.offsets()[level] + i * sizes[level];
            assert_eq!(
                slice.chunk(i),
                &buffers.as_slice()[start..start + sizes[level]]
            );
            assert_eq!(buffers.chunk(level, i), slice.chunk(i));
            assert_eq!(slice.chunk(i)[0], (100 * level + 10 * i) as u64);
            assert_eq!(
                &slice.as_slice()[i * sizes[level]..][..1],
                [slice.chunk(i)[0]]
            );
        }
    }
}

#[test]
fn levels_are_independent() {
    let mut buffers = LevelBuffers::<f64>::new(&[1, 8, 64], &[3, 3, 5]);
    buffers.level_mut(1).as_mut_slice().fill(1.0);
    assert!(buffers.level(0).as_slice().iter().all(|&v| v == 0.0));
    assert!(buffers.level(1).as_slice().iter().all(|&v| v == 1.0));
    assert!(buffers.level(2).as_slice().iter().all(|&v| v == 0.0));

    // M2M: the parent level mutably, the child level shared, from one buffer.
    let (mut parents, children) = buffers.parent_child_mut(0);
    assert_eq!((parents.len(), parents.size()), (1, 3));
    assert_eq!((children.len(), children.size()), (8, 3));
    for c in 0..children.len() {
        for (p, &v) in parents.chunk_mut(0).iter_mut().zip(children.chunk(c)) {
            *p += v;
        }
    }
    assert_eq!(buffers.chunk(0, 0), [8.0; 3]);

    // L2L: the child level mutably, the parent level shared.
    let (mut children, parents) = buffers.child_parent_mut(1);
    assert_eq!((children.len(), children.size()), (64, 5));
    assert_eq!((parents.len(), parents.size()), (8, 3));
    children.chunk_mut(63)[4] = parents.chunk(7)[2] + 1.0;
    assert_eq!(buffers.chunk(2, 63), [0.0, 0.0, 0.0, 0.0, 2.0]);
    assert!(buffers.level(1).as_slice().iter().all(|&v| v == 1.0));

    buffers.clear();
    assert!(buffers.as_slice().iter().all(|&v| v == 0.0));
}

#[test]
fn empty_levels_have_no_chunks() {
    let mut buffers = LevelBuffers::<u32>::new(&[0, 2, 0], &[1, 2, 3]);
    assert_eq!(buffers.offsets(), [0, 0, 4, 4]);
    assert!(buffers.level(0).is_empty());
    assert_eq!(buffers.level_mut(2).chunks_mut().count(), 0);
    let empty = LevelBuffers::<u32>::new(&[], &[]);
    assert_eq!(empty.nlevels(), 0);
    assert!(empty.as_slice().is_empty());
}

#[test]
#[should_panic(expected = "positive")]
fn level_sizes_must_be_positive() {
    LevelBuffers::<u32>::new(&[1, 8], &[1, 0]);
}

#[test]
fn leaf_offsets_and_lengths_follow_the_counts() {
    let counts = [3, 0, 1, 0, 0, 4];
    let store = LeafStore::<u32>::new(&counts, 4);
    assert_eq!(store.nleaves(), 6);
    assert_eq!(store.point_size(), 4);
    assert_eq!(store.point_offsets(), [0, 3, 3, 4, 4, 4, 8]);
    assert_eq!(store.as_slice().len(), 32);
    for (j, &count) in counts.iter().enumerate() {
        assert_eq!(store.count(j), count);
        assert_eq!(store.chunk(j).len(), 4 * count);
    }
    assert!(store.has_counts(&counts));
    assert!(!store.has_counts(&[3, 0, 1, 0, 0]));
    assert!(!store.has_counts(&[3, 1, 0, 0, 0, 4]));
}

#[test]
fn leaf_chunks_are_disjoint_and_clear_resets_them() {
    let counts = [2, 0, 3, 1];
    let mut store = LeafStore::<u64>::new(&counts, 3);
    for j in 0..4 {
        for (k, value) in store.chunk_mut(j).iter_mut().enumerate() {
            *value = (10 * j + k) as u64 + 1;
        }
    }
    // Every value belongs to exactly one chunk, in leaf order.
    let mut expected = Vec::new();
    for (j, &count) in counts.iter().enumerate() {
        expected.extend((0..3 * count).map(|k| (10 * j + k) as u64 + 1));
    }
    assert_eq!(store.as_slice(), expected);

    store.clear();
    assert!(store.as_slice().iter().all(|&v| v == 0));
    assert_eq!(store.point_offsets(), [0, 2, 2, 5, 6]);
}

#[test]
fn a_leaf_without_points_or_values_has_an_empty_chunk() {
    // Zero counts, and a store with zero values per point (target input of an operator
    // that needs none): the counts survive, every chunk is empty.
    let mut store = LeafStore::<f32>::new(&[0, 5, 0], 0);
    assert_eq!(store.count(1), 5);
    assert!(store.chunk(1).is_empty() && store.chunk_mut(0).is_empty());
    assert!(store.as_slice().is_empty());
    let empty = LeafStore::<f32>::new(&[], 3);
    assert_eq!(empty.nleaves(), 0);
    assert!(empty.range(0..0).is_empty());
}

#[test]
fn leaf_ranges_give_contiguous_slices_with_their_offsets() {
    let counts = [1, 2, 0, 3, 1];
    let mut store = LeafStore::<u32>::new(&counts, 2);
    for j in 0..5 {
        store.chunk_mut(j).fill(j as u32 + 1);
    }

    let range = store.range(1..4);
    assert_eq!(range.nleaves(), 3);
    assert_eq!(range.point_size(), 2);
    assert_eq!(range.as_slice(), [2, 2, 2, 2, 4, 4, 4, 4, 4, 4]);
    for r in 0..3 {
        assert_eq!(range.count(r), counts[1 + r]);
        assert_eq!(range.chunk(r), store.chunk(1 + r));
    }

    let mut range = store.range_mut(1..5);
    assert_eq!(range.count(3), 1);
    range.chunk_mut(2)[5] = 40;
    // Per-row slices without allocation, as a level call splits its targets.
    let lengths: Vec<usize> = range.chunks_mut().map(|chunk| chunk.len()).collect();
    assert_eq!(lengths, [4, 0, 6, 2]);
    for (r, chunk) in range.chunks_mut().enumerate() {
        chunk[..].iter_mut().for_each(|v| *v += 10 * r as u32);
    }
    // Halves for a recursive split under `rayon::join`.
    let (mut left, right) = range.split_at_mut(2);
    assert_eq!((left.nleaves(), right.nleaves()), (2, 2));
    assert_eq!(right.chunk(1), [35, 35]);
    left.chunk_mut(0)[0] = 0;
    assert_eq!(store.chunk(1), [0, 2, 2, 2]);
    assert_eq!(store.chunk(3), [24, 24, 24, 24, 24, 60]);
    assert_eq!(store.chunk(0), [1, 1]);
}

#[test]
fn leaf_store_tail_is_the_ghost_range() {
    // Local leaves 0..3, ghost leaves 3..5: the ghost tail is one contiguous slice.
    let mut store = LeafStore::<u32>::new(&[2, 1, 0, 3, 0], 1);
    store
        .range_mut(3..5)
        .as_mut_slice()
        .copy_from_slice(&[7, 8, 9]);
    assert_eq!(store.chunk(3), [7, 8, 9]);
    assert!(store.chunk(4).is_empty());
    assert_eq!(&store.as_slice()[3..], [7, 8, 9]);
}

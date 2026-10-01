//! Panics: an M2L pair that is not a V-list pair, a non-parent pair (debug builds),
//! chunks whose length is not a multiple of the size per point, and a leaf with more
//! sources than the operator was built for.

use nd_fmm_exec::tables::M2lStrategy;
use nd_octree::morton;

use crate::common::{len, operator};

fn key(index: [usize; 3], level: usize) -> u64 {
    morton::from_index_and_level(index, level)
}

#[test]
#[should_panic(expected = "is not a V-list offset")]
fn m2l_of_neighbours_panics() {
    let mut op = operator(M2lStrategy::Dense, 1, 0);
    let mut local = vec![0.0; len(1)];
    op.m2l_pair(key([3, 3, 3], 4), key([4, 3, 2], 4), &[0.0; 4], &mut local);
}

#[test]
#[should_panic(expected = "is not a V-list offset")]
fn m2l_of_distant_boxes_panics() {
    let mut op = operator(M2lStrategy::Dense, 1, 0);
    let mut local = vec![0.0; len(1)];
    op.m2l_pair(key([0, 0, 0], 4), key([4, 0, 0], 4), &[0.0; 4], &mut local);
}

#[test]
#[should_panic(expected = "different levels")]
fn m2l_across_levels_panics() {
    let mut op = operator(M2lStrategy::Dense, 1, 0);
    let mut local = vec![0.0; len(1)];
    op.m2l_pair(key([0, 0, 0], 3), key([4, 0, 0], 4), &[0.0; 4], &mut local);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "is not the parent of")]
fn m2m_of_a_non_parent_panics_in_debug_builds() {
    let mut op = operator(M2lStrategy::Dense, 1, 0);
    let mut parent = vec![0.0; len(1)];
    op.m2m_pair(key([4, 5, 6], 5), key([1, 2, 3], 4), &[0.0; 4], &mut parent);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "is not the parent of")]
fn l2l_to_a_non_child_panics_in_debug_builds() {
    let mut op = operator(M2lStrategy::Rotation, 1, 0);
    let mut child = vec![0.0; len(1)];
    op.l2l_pair(key([1, 2, 3], 4), key([2, 4, 8], 5), &[0.0; 4], &mut child);
}

#[test]
#[should_panic(expected = "a source chunk holds 4 values per point, got 7 values")]
fn a_source_chunk_of_the_wrong_length_panics() {
    let mut op = operator(M2lStrategy::Dense, 1, 0);
    let mut multipole = vec![0.0; len(1)];
    op.p2m_leaf(&[0.0; 7], &mut multipole);
}

#[test]
#[should_panic(expected = "a source chunk holds 4 values per point")]
fn a_p2p_source_chunk_of_the_wrong_length_panics() {
    let mut op = operator(M2lStrategy::Dense, 1, 4);
    let t = key([1, 1, 1], 3);
    let mut output = vec![0.0; 4];
    op.p2p_pair(key([2, 1, 1], 3), t, &[0.0; 5], &[0.1; 3], &mut output);
}

#[test]
#[should_panic(expected = "a target input chunk holds 3 values per point, got 4 values")]
fn a_target_input_chunk_of_the_wrong_length_panics() {
    let mut op = operator(M2lStrategy::Dense, 1, 0);
    let mut output = vec![0.0; 4];
    op.l2p_leaf(&[0.0; 4], &[0.0; 4], &mut output);
}

#[test]
#[should_panic(expected = "a target output chunk holds 4 values per point, for 2 points")]
fn a_target_output_chunk_of_the_wrong_length_panics() {
    let mut op = operator(M2lStrategy::Dense, 1, 0);
    let mut output = vec![0.0; 2];
    op.l2p_leaf(&[0.0; 4], &[0.0; 6], &mut output);
}

#[test]
#[should_panic(expected = "more than max_leaf_points = 1")]
fn a_source_leaf_larger_than_the_scratch_panics() {
    let mut op = operator(M2lStrategy::Dense, 1, 1);
    let sources = [0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 1.0, 1.0];
    let mut output = vec![0.0; 4];
    let t = key([1, 1, 1], 3);
    op.p2p_pair(key([2, 1, 1], 3), t, &sources, &[0.1; 3], &mut output);
}

//! Pure local tests of the index operators; no MPI.
use super::IndexFmm;
use crate::fmm::operator::FmmOperator;
use nd_octree::morton;

#[test]
fn sizes_follow_the_number_of_leaves() {
    let op = IndexFmm::new(5);
    assert_eq!(op.multipole_size(3), 5);
    assert_eq!(op.local_size(0), 5);
    assert_eq!(op.target_size(), 5);
    assert_eq!(op.source_size(), 1);
}

#[test]
fn particle_operators_add_the_source_index() {
    let op = IndexFmm::new(4);
    let (a, b) = (morton::root(), morton::root());
    let mut out = [0, 1, 0, 0];
    op.p2m(a, &[2], &mut out);
    op.p2l(a, b, &[1], &mut out);
    op.p2p(a, b, &[2], &mut out);
    assert_eq!(out, [0, 2, 2, 0]);
}

#[test]
fn translation_operators_add_their_input() {
    let op = IndexFmm::new(3);
    let key = morton::root();
    let input = [1, 0, 2];
    let mut out = [0, 1, 0];
    op.m2m(key, key, &input, &mut out);
    op.m2l(key, key, &input, &mut out);
    op.l2l(key, key, &input, &mut out);
    op.l2p(key, &input, &mut out);
    op.m2p(key, key, &input, &mut out);
    assert_eq!(out, [5, 1, 10]);
}

#[test]
#[should_panic(expected = "at least one leaf")]
fn an_empty_tree_is_rejected() {
    IndexFmm::new(0);
}

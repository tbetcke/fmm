//! Pure local tests; no MPI initialization is needed for the grouping helpers.
use super::{get_ancestor_key, sort_by_leafs};
use nd_octree::{PhysicalBox, morton::is_ancestor, points_to_morton};
use rlst::rlst_dynamic_array;

fn octant_keys(level: usize) -> Vec<u64> {
    let mut points = rlst_dynamic_array!(f64, [3, 8]);
    for j in 0..8 {
        for i in 0..3 {
            points[[i, j]] = if j & (1 << i) == 0 { 0.25 } else { 0.75 };
        }
    }
    let mut keys = points_to_morton(
        &points,
        level,
        &PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    );
    keys.sort_unstable();
    keys
}

#[test]
fn ancestor_lookup_handles_exact_matches_and_descendants_at_both_ends() {
    let leaves = octant_keys(1);
    let fine = octant_keys(4);
    for (&leaf, &key) in leaves.iter().zip(&fine) {
        assert_eq!(get_ancestor_key(&leaves, leaf), leaf);
        assert!(is_ancestor(leaf, key));
        assert_eq!(get_ancestor_key(&leaves, key), leaf);
    }
}

#[test]
fn ancestor_lookup_handles_mixed_depth_leaves() {
    let mut leaves = octant_keys(1);
    // Refine one entry along the path of its sole test point. The queried
    // points are covered, which is the helper's required precondition.
    let fine = octant_keys(4);
    leaves[3] = octant_keys(2)[3];
    assert!(leaves.windows(2).all(|pair| pair[0] < pair[1]));
    for (&leaf, &key) in leaves.iter().zip(&fine) {
        assert_eq!(get_ancestor_key(&leaves, key), leaf);
    }
}

#[test]
fn grouping_preserves_index_key_pairs_duplicates_and_empty_leaf_ranges() {
    let leaves = octant_keys(1);
    let fine = octant_keys(4);
    let keys = [fine[6], fine[1], fine[6], leaves[1], fine[3]];
    let indices = [42, 7, 99, 13, 2];
    let (actual_indices, actual_keys, indptr) = sort_by_leafs(&indices, &keys, &leaves);
    assert_eq!(actual_indices, [7, 13, 2, 42, 99]);
    assert_eq!(actual_keys, [fine[1], leaves[1], fine[3], fine[6], fine[6]]);
    assert_eq!(indptr, [0, 0, 2, 2, 3, 3, 3, 5, 5]);
}

#[test]
fn grouping_empty_population_retains_all_leaf_offsets() {
    let leaves = octant_keys(1);
    let (indices, keys, indptr) = sort_by_leafs(&[], &[], &leaves);
    assert!(indices.is_empty());
    assert!(keys.is_empty());
    assert_eq!(indptr, vec![0; leaves.len() + 1]);
}

#[test]
fn grouping_empty_population_and_no_leaves_returns_sentinel() {
    assert_eq!(sort_by_leafs(&[], &[], &[]), (vec![], vec![], vec![0]));
}

#[test]
fn grouping_one_leaf_preserves_input_order_not_fine_key_order() {
    let leaves = octant_keys(1);
    let fine = octant_keys(4);
    let keys = [fine[0], leaves[0], fine[0]];
    let indices = [8, 3, 5];
    let (actual_indices, actual_keys, indptr) = sort_by_leafs(&indices, &keys, &leaves[..1]);
    assert_eq!(actual_indices, indices);
    assert_eq!(actual_keys, keys);
    assert_eq!(indptr, [0, 3]);
}

#[test]
fn grouping_matches_linear_ancestor_oracle_for_many_input_orders() {
    let leaves = octant_keys(1);
    let fine = octant_keys(4);
    for rotation in 0..32 {
        let indices: Vec<_> = (0..32).map(|i| 100 + (i + rotation) % 32).collect();
        let keys: Vec<_> = indices.iter().map(|i| fine[(i * 5 + i / 3) % 8]).collect();
        let mut expected_indices = Vec::new();
        let mut expected_keys = Vec::new();
        let mut expected_indptr = vec![0];
        for &leaf in &leaves {
            for (&index, &key) in indices.iter().zip(&keys) {
                if is_ancestor(leaf, key) {
                    expected_indices.push(index);
                    expected_keys.push(key);
                }
            }
            expected_indptr.push(expected_indices.len());
        }
        assert_eq!(expected_indices.len(), indices.len());
        assert_eq!(
            sort_by_leafs(&indices, &keys, &leaves),
            (expected_indices, expected_keys, expected_indptr),
            "rotation {rotation}"
        );
    }
}

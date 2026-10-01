//! Pure local tests; the interaction lists are built without MPI from synthetic
//! key classification maps.
use super::{InteractionManager, V_LIST_DIRECTIONS, is_adjacent};
use nd_octree::{MortonKey, constants::DEEPEST_LEVEL, morton, octree::KeyType};
use std::collections::HashMap;

/// All cells of a uniformly refined tree at the given level.
pub(crate) fn uniform_leaves(level: usize) -> Vec<MortonKey> {
    let n = 1usize << level;
    let mut leaves = Vec::with_capacity(n * n * n);
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                leaves.push(morton::from_index_and_level([x, y, z], level));
            }
        }
    }
    leaves.sort_unstable();
    leaves
}

/// The level-one tree with the octant `[0, 0, 0]` refined once.
fn refined_octant_leaves() -> Vec<MortonKey> {
    let refined = morton::from_index_and_level([0, 0, 0], 1);
    let mut leaves: Vec<MortonKey> = uniform_leaves(1)
        .into_iter()
        .filter(|&key| key != refined)
        .collect();
    leaves.extend(morton::children(refined).unwrap());
    leaves.sort_unstable();
    leaves
}

/// Deterministic adaptive trees, balanced from a handful of seed keys.
fn adaptive_leaves() -> Vec<Vec<MortonKey>> {
    let seeds: [Vec<MortonKey>; 3] = [
        // Two non-adjacent octants refined to level two.
        vec![
            morton::from_index_and_level([0, 0, 0], 2),
            morton::from_index_and_level([3, 3, 3], 2),
        ],
        // The same, with one level-two box refined further.
        vec![
            morton::from_index_and_level([0, 0, 0], 3),
            morton::from_index_and_level([3, 3, 3], 2),
        ],
        // A deep interior box and a deep corner box.
        vec![
            morton::from_index_and_level([3, 4, 2], 3),
            morton::from_index_and_level([7, 7, 7], 3),
            morton::from_index_and_level([1, 0, 0], 1),
        ],
    ];
    seeds
        .into_iter()
        .map(|seed| {
            let leaves = morton::balance(&seed, morton::root());
            assert!(morton::is_complete_linear_and_balanced(&leaves));
            leaves
        })
        .collect()
}

/// Every tree used by these tests.
pub(crate) fn all_trees() -> Vec<Vec<MortonKey>> {
    let mut trees = vec![
        uniform_leaves(1),
        uniform_leaves(2),
        refined_octant_leaves(),
    ];
    trees.extend(adaptive_leaves());
    trees
}

/// Classify a complete, balanced leaf list as a purely local tree.
pub(crate) fn key_types(leaves: &[MortonKey]) -> HashMap<MortonKey, KeyType> {
    assert!(morton::is_complete_linear_and_balanced(leaves));
    let mut all_keys = HashMap::<MortonKey, KeyType>::new();
    for key in morton::get_interior_keys(leaves) {
        all_keys.insert(key, KeyType::LocalInterior);
    }
    for &leaf in leaves {
        all_keys.insert(leaf, KeyType::LocalLeaf);
    }
    all_keys
}

/// Inclusive index bounds of a box, expressed in cells of the deepest level.
fn deepest_bounds(key: MortonKey) -> [[u64; 2]; 3] {
    let (level, index) = morton::decode(key);
    let size = 1u64 << (DEEPEST_LEVEL as usize - level);
    [0usize, 1, 2].map(|dim| {
        let low = index[dim] as u64 * size;
        [low, low + size - 1]
    })
}

/// Adjacency written independently of the implementation: two boxes touch when
/// their closed cubes intersect and neither contains the other.
fn touching(a: MortonKey, b: MortonKey) -> bool {
    if morton::is_ancestor(a, b) || morton::is_ancestor(b, a) {
        return false;
    }
    let a_bounds = deepest_bounds(a);
    let b_bounds = deepest_bounds(b);
    (0..3).all(|dim| {
        a_bounds[dim][0] <= b_bounds[dim][1] + 1 && b_bounds[dim][0] <= a_bounds[dim][1] + 1
    })
}

/// A brute-force oracle over the global key set, using only the geometric
/// definitions of the four lists.
pub(crate) struct Oracle {
    pub(crate) keys: Vec<MortonKey>,
    leaves: Vec<MortonKey>,
}

impl Oracle {
    pub(crate) fn new(leaves: &[MortonKey]) -> Self {
        let mut keys: Vec<MortonKey> = morton::get_interior_keys(leaves).into_iter().collect();
        keys.extend_from_slice(leaves);
        keys.sort_unstable();
        keys.dedup();
        let mut leaves = leaves.to_vec();
        leaves.sort_unstable();
        Self { keys, leaves }
    }

    fn is_leaf(&self, key: MortonKey) -> bool {
        self.leaves.binary_search(&key).is_ok()
    }

    /// All leaves adjacent to a leaf `b`.
    pub(crate) fn u(&self, b: MortonKey) -> Vec<MortonKey> {
        if !self.is_leaf(b) {
            return Vec::new();
        }
        self.leaves
            .iter()
            .copied()
            .filter(|&leaf| leaf != b && touching(leaf, b))
            .collect()
    }

    /// All boxes on the level of `b` whose parent touches the parent of `b` and
    /// that do not touch `b`.
    pub(crate) fn v(&self, b: MortonKey) -> Vec<MortonKey> {
        let Some(b_parent) = morton::parent(b) else {
            return Vec::new();
        };
        self.keys
            .iter()
            .copied()
            .filter(|&key| {
                morton::level(key) == morton::level(b)
                    && morton::parent(key)
                        .is_some_and(|parent| parent != b_parent && touching(parent, b_parent))
                    && !touching(key, b)
            })
            .collect()
    }

    /// All boxes finer than the leaf `b` whose parent touches `b` and that do
    /// not touch `b` themselves.
    pub(crate) fn w(&self, b: MortonKey) -> Vec<MortonKey> {
        if !self.is_leaf(b) {
            return Vec::new();
        }
        self.keys
            .iter()
            .copied()
            .filter(|&key| {
                morton::level(key) > morton::level(b)
                    && morton::parent(key).is_some_and(|parent| touching(parent, b))
                    && !touching(key, b)
            })
            .collect()
    }

    /// The dual of the W-list: all leaves `l` with `b` in `w(l)`, written out.
    pub(crate) fn x(&self, b: MortonKey) -> Vec<MortonKey> {
        let Some(b_parent) = morton::parent(b) else {
            return Vec::new();
        };
        self.leaves
            .iter()
            .copied()
            .filter(|&leaf| {
                morton::level(leaf) < morton::level(b)
                    && touching(b_parent, leaf)
                    && !touching(b, leaf)
            })
            .collect()
    }
}

/// Names and keys of the eight level-one octants.
fn level_one_keys() -> Vec<MortonKey> {
    uniform_leaves(1)
}

#[test]
fn uniform_level_two_tree_matches_hand_counts() {
    let leaves = uniform_leaves(2);
    let lists = InteractionManager::from_key_types(&key_types(&leaves));

    // A corner, a face, an edge and an interior box of the 4x4x4 grid. The
    // U-count is the in-domain closed 3x3x3 neighbourhood minus the box itself,
    // and every one of the 64 boxes outside that neighbourhood is a V-entry,
    // since all eight level-one boxes are mutually adjacent.
    for (index, u_count, v_count) in [
        ([0usize, 0, 0], 7, 56),
        ([1, 0, 0], 11, 52),
        ([1, 1, 0], 17, 46),
        ([1, 1, 1], 26, 37),
    ] {
        let key = morton::from_index_and_level(index, 2);
        assert_eq!(lists.u_list()[&key].len(), u_count, "U of {index:?}");
        assert_eq!(lists.v_list()[&key].len(), v_count, "V of {index:?}");
    }

    // A uniform tree has no finer or coarser neighbours at all.
    for &leaf in &leaves {
        assert!(lists.w_list()[&leaf].is_empty());
        assert!(lists.x_list()[&leaf].is_empty());
    }

    // Level-one boxes are interior, so U and W are empty, and their parent is
    // the root, which has no neighbours, so V and X are empty as well.
    for key in level_one_keys().into_iter().chain([morton::root()]) {
        assert!(lists.u_list()[&key].is_empty());
        assert!(lists.v_list()[&key].is_empty());
        assert!(lists.w_list()[&key].is_empty());
        assert!(lists.x_list()[&key].is_empty());
    }
}

#[test]
fn one_refined_octant_gives_expected_w_and_x() {
    let leaves = refined_octant_leaves();
    let lists = InteractionManager::from_key_types(&key_types(&leaves));
    let child = |index: [usize; 3]| morton::from_index_and_level(index, 2);
    let octant = |index: [usize; 3]| morton::from_index_and_level(index, 1);

    // The refined octant spans the level-two cells with indices in {0, 1}. The
    // level-one leaf [1, 0, 0] spans the cells with x-index 2 and 3, so only the
    // children with x-index 1 touch it; the other four are its W-list.
    let mut expected: Vec<MortonKey> = (0..2)
        .flat_map(|y| (0..2).map(move |z| child([0, y, z])))
        .collect();
    expected.sort_unstable();
    assert_eq!(lists.w_list()[&octant([1, 0, 0])], expected);

    // Only the child [1, 1, 1] touches the level-one leaf [1, 1, 1], through the
    // centre vertex of the domain.
    let mut expected: Vec<MortonKey> = morton::children(octant([0, 0, 0]))
        .unwrap()
        .into_iter()
        .filter(|&key| key != child([1, 1, 1]))
        .collect();
    expected.sort_unstable();
    assert_eq!(lists.w_list()[&octant([1, 1, 1])], expected);

    // The child [0, 0, 0] sits in the far corner of the refined octant and
    // touches none of the seven level-one leaves.
    let mut expected: Vec<MortonKey> = level_one_keys()
        .into_iter()
        .filter(|&key| key != octant([0, 0, 0]))
        .collect();
    expected.sort_unstable();
    assert_eq!(lists.x_list()[&child([0, 0, 0])], expected);

    // The child [1, 1, 1] touches every level-one leaf, so its X-list is empty.
    assert!(lists.x_list()[&child([1, 1, 1])].is_empty());

    // The child [1, 0, 0] has y- and z-index 0, so a level-one leaf touches it
    // exactly when its y- and z-index are 0 as well; that is only [1, 0, 0].
    let mut expected: Vec<MortonKey> = level_one_keys()
        .into_iter()
        .filter(|&key| key != octant([0, 0, 0]) && key != octant([1, 0, 0]))
        .collect();
    expected.sort_unstable();
    assert_eq!(lists.x_list()[&child([1, 0, 0])], expected);

    // Its U-list are its seven siblings and the coarser leaf [1, 0, 0], which
    // covers the four missing level-two neighbour cells.
    let mut expected: Vec<MortonKey> = morton::children(octant([0, 0, 0]))
        .unwrap()
        .into_iter()
        .filter(|&key| key != child([1, 0, 0]))
        .collect();
    expected.push(octant([1, 0, 0]));
    expected.sort_unstable();
    assert_eq!(lists.u_list()[&child([1, 0, 0])], expected);

    // The X-list is the dual of the W-list.
    for (&key, x) in lists.x_list().iter() {
        for &leaf in x {
            assert!(lists.w_list()[&leaf].contains(&key));
        }
    }
    for (&leaf, w) in lists.w_list().iter() {
        for &key in w {
            assert!(lists.x_list()[&key].contains(&leaf));
        }
    }
}

#[test]
fn lists_match_brute_force_oracle_on_adaptive_trees() {
    // Count non-empty lists so that the comparison cannot pass vacuously.
    let mut populated = [0usize; 4];
    for leaves in all_trees() {
        let lists = InteractionManager::from_key_types(&key_types(&leaves));
        let oracle = Oracle::new(&leaves);
        for &key in &oracle.keys {
            let expected = [oracle.u(key), oracle.v(key), oracle.w(key), oracle.x(key)];
            let actual = [
                lists.u_list().get(&key).unwrap(),
                lists.v_list().get(&key).unwrap(),
                lists.w_list().get(&key).unwrap(),
                lists.x_list().get(&key).unwrap(),
            ];
            for (index, list_name) in ["U", "V", "W", "X"].into_iter().enumerate() {
                assert_eq!(*actual[index], expected[index], "{list_name}-list of {key}");
                populated[index] += usize::from(!expected[index].is_empty());
            }
        }
    }
    assert!(populated.iter().all(|&count| count > 0), "{populated:?}");
}

#[test]
fn invariants_hold() {
    for leaves in all_trees() {
        let all_keys = key_types(&leaves);
        let lists = InteractionManager::from_key_types(&all_keys);
        for map in [
            lists.u_list(),
            lists.v_list(),
            lists.w_list(),
            lists.x_list(),
        ] {
            assert_eq!(map.len(), all_keys.len());
            for (&key, list) in map.iter() {
                assert!(all_keys.contains_key(&key));
                assert!(list.windows(2).all(|pair| pair[0] < pair[1]));
                assert!(!list.contains(&key));
                assert!(list.iter().all(|entry| all_keys.contains_key(entry)));
            }
        }
        for (&key, &key_type) in all_keys.iter() {
            if key_type == KeyType::LocalInterior {
                assert!(lists.u_list()[&key].is_empty());
                assert!(lists.w_list()[&key].is_empty());
            }
            if morton::level(key) <= 1 {
                assert!(lists.v_list()[&key].is_empty());
                assert!(lists.x_list()[&key].is_empty());
            }
        }
    }
}

#[test]
fn adjacency_helper() {
    let at = |index: [usize; 3], level: usize| morton::from_index_and_level(index, level);

    // Face, edge and vertex contact on a common level.
    assert!(is_adjacent(at([0, 0, 0], 2), at([1, 0, 0], 2)));
    assert!(is_adjacent(at([0, 0, 0], 2), at([1, 1, 0], 2)));
    assert!(is_adjacent(at([0, 0, 0], 2), at([1, 1, 1], 2)));
    // Boxes one cell apart do not touch.
    assert!(!is_adjacent(at([0, 0, 0], 2), at([2, 0, 0], 2)));
    assert!(!is_adjacent(at([0, 0, 0], 2), at([0, 2, 1], 2)));

    // Mixed levels: the level-one box [1, 0, 0] covers the level-two cells with
    // x-index 2 and 3 and y-, z-index 0 and 1.
    assert!(is_adjacent(at([1, 0, 0], 1), at([1, 0, 0], 2)));
    assert!(is_adjacent(at([1, 0, 0], 1), at([1, 1, 1], 2)));
    assert!(!is_adjacent(at([1, 0, 0], 1), at([0, 0, 0], 2)));
    // Vertex contact across two levels.
    assert!(is_adjacent(at([1, 1, 1], 1), at([1, 1, 1], 2)));
    assert!(is_adjacent(at([1, 1, 1], 1), at([3, 3, 3], 3)));
    assert!(!is_adjacent(at([1, 1, 1], 1), at([2, 3, 3], 3)));

    // Nesting and identity are not adjacency.
    assert!(!is_adjacent(at([1, 0, 0], 1), at([2, 0, 0], 2)));
    assert!(!is_adjacent(at([2, 0, 0], 2), at([1, 0, 0], 1)));
    assert!(!is_adjacent(at([1, 0, 0], 2), at([1, 0, 0], 2)));

    // The root contains everything, so it touches nothing.
    assert!(!is_adjacent(morton::root(), morton::root()));
    for &key in &uniform_leaves(2) {
        assert!(!is_adjacent(morton::root(), key));
        assert!(!is_adjacent(key, morton::root()));
    }
}

#[test]
fn ghost_classification_is_respected() {
    let leaves = uniform_leaves(2);
    let local = key_types(&leaves);
    let reference = InteractionManager::from_key_types(&local);

    // Pretend the slab with the largest x-index belongs to another rank. The
    // lists depend only on the leaf/interior classification, not on ownership.
    let mut mixed = local.clone();
    for (&key, key_type) in local.iter() {
        let (_, index) = morton::decode(key);
        if morton::level(key) == 2 && index[0] == 3 || morton::level(key) == 1 && index[0] == 1 {
            let ghost = match key_type {
                KeyType::LocalLeaf => KeyType::GhostLeaf(1),
                _ => KeyType::GhostInterior(1),
            };
            *mixed.get_mut(&key).unwrap() = ghost;
        }
    }
    let ghost_count = mixed.values().filter(|value| value.is_ghost()).count();
    assert_eq!(ghost_count, 16 + 4);

    let lists = InteractionManager::from_key_types(&mixed);
    for (&key, &key_type) in mixed.iter() {
        for (map, map_reference) in [
            (lists.u_list(), reference.u_list()),
            (lists.v_list(), reference.v_list()),
            (lists.w_list(), reference.w_list()),
            (lists.x_list(), reference.x_list()),
        ] {
            if key_type.is_ghost() {
                assert!(!map.contains_key(&key));
            } else {
                assert_eq!(map[&key], map_reference[&key]);
            }
        }
    }
    assert_eq!(lists.u_list().len(), mixed.len() - ghost_count);
}

#[test]
fn v_list_directions_are_complete() {
    let mut expected = Vec::new();
    for x in -3i64..=3 {
        for y in -3i64..=3 {
            for z in -3i64..=3 {
                if x.abs().max(y.abs()).max(z.abs()) >= 2 {
                    expected.push([x, y, z]);
                }
            }
        }
    }
    // The expected list is generated in lexicographic order, so equality also
    // checks that the constant is sorted and free of duplicates.
    assert_eq!(V_LIST_DIRECTIONS.to_vec(), expected);
}

#[test]
fn v_list_by_direction_matches_v_list() {
    for leaves in all_trees() {
        let lists = InteractionManager::from_key_types(&key_types(&leaves));
        let depth = leaves
            .iter()
            .map(|&leaf| morton::level(leaf))
            .max()
            .unwrap();
        for level in 0..=depth + 1 {
            let by_direction = lists.v_list_by_direction(level);
            assert_eq!(by_direction.len(), V_LIST_DIRECTIONS.len());

            let mut pair_count = 0;
            for direction in V_LIST_DIRECTIONS {
                let pairs = &by_direction[&direction];
                assert!(pairs.is_sorted());
                for &(target, source) in pairs {
                    assert_eq!(morton::level(target), level);
                    assert!(lists.v_list()[&target].contains(&source));
                    let (_, target_index) = morton::decode(target);
                    let (_, source_index) = morton::decode(source);
                    let offset =
                        [0, 1, 2].map(|dim| target_index[dim] as i64 - source_index[dim] as i64);
                    assert_eq!(offset, direction);
                }
                pair_count += pairs.len();
            }

            let expected_count: usize = lists
                .v_list()
                .iter()
                .filter(|&(&target, _)| morton::level(target) == level)
                .map(|(_, sources)| sources.len())
                .sum();
            assert_eq!(pair_count, expected_count);
        }
    }
}

#[test]
fn every_v_list_direction_occurs() {
    let lists = InteractionManager::from_key_types(&key_types(&uniform_leaves(3)));
    let by_direction = lists.v_list_by_direction(3);
    for direction in V_LIST_DIRECTIONS {
        assert!(
            !by_direction[&direction].is_empty(),
            "direction {direction:?} has no interaction"
        );
    }
}

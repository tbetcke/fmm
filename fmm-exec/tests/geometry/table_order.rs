//! Table order: the octant and offset order of `nd-fmm-tables` against `nd-octree` and
//! `V_LIST_DIRECTIONS` of `nd-fmm-plan` (CONVENTIONS §3.12, "Child octants" and
//! "V-list offsets"). Error measure: exact equality (integers).

use nd_fmm_plan::interaction_manager::V_LIST_DIRECTIONS;
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offset_index, m2l_offsets, octant_direction};
use nd_octree::morton;

use crate::common::{SplitMix64, sample_keys};

#[test]
fn m2l_offsets_equal_v_list_directions() {
    assert_eq!(V_LIST_DIRECTIONS.len(), M2L_OFFSET_COUNT);
    assert_eq!(m2l_offsets(), V_LIST_DIRECTIONS);
    for (position, d) in V_LIST_DIRECTIONS.into_iter().enumerate() {
        assert_eq!(m2l_offset_index(d), Some(position), "d = {d:?}");
    }
    println!("m2l_offsets() = V_LIST_DIRECTIONS: 316 offsets, same content and order");
}

#[test]
fn children_follow_the_child_index_and_octant_direction() {
    let mut rng = SplitMix64::new(0x7301);
    let mut checked = 0;
    for level in [0, 1, 7, 15] {
        let parents = match level {
            0 => vec![morton::root()],
            1 => (0..8)
                .map(|o| morton::from_index_and_level([o >> 2, (o >> 1) & 1, o & 1], 1))
                .collect(),
            _ => sample_keys(&mut rng, level, 40),
        };
        for parent in parents {
            let (parent_level, parent_index) = morton::decode(parent);
            assert_eq!(parent_level, level);
            let children = morton::children(parent).unwrap();
            for (o, &child) in children.iter().enumerate() {
                let bits = [(o >> 2) & 1, (o >> 1) & 1, o & 1];
                assert_eq!(morton::child_index(child), o, "parent {parent}, o = {o}");
                assert_eq!(morton::parent(child), Some(parent));
                let want: [usize; 3] = core::array::from_fn(|k| 2 * parent_index[k] + bits[k]);
                assert_eq!(morton::decode(child), (level + 1, want), "o = {o}");
                assert_eq!(
                    octant_direction(o),
                    bits.map(|b| 2 * b as i64 - 1),
                    "o = {o}"
                );
                checked += 1;
            }
        }
    }
    println!("children on levels 0, 1, 7, 15: {checked} (parent, octant) pairs agree");
}

//! The cube group O_h and the symmetry classes of the V-list offsets (CONVENTIONS
//! §3.12, "The cube symmetry group" and "Symmetry classes").

use std::collections::{BTreeMap, BTreeSet};

use nd_fmm_tables::geometry::octant_direction;
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, OCTANT_COUNT, m2l_offset_index, m2l_offsets};
use nd_fmm_tables::symmetry::{
    GROUP_ORDER, M2L_CLASS_COUNT, SignedPermutation, class_of, class_representatives,
};

/// The six permutations of (0, 1, 2) in lexicographic order.
const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// The class sizes of CONVENTIONS §3.12, in class order.
const CLASS_SIZES: [usize; 16] = [6, 6, 24, 24, 12, 24, 12, 24, 24, 24, 48, 24, 8, 24, 24, 8];

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    core::array::from_fn(|i| core::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn mat_vec(a: &[[f64; 3]; 3], v: [i64; 3]) -> [i64; 3] {
    core::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k] as f64).sum::<f64>() as i64)
}

#[test]
fn enumeration_order_is_that_of_section_3_12() {
    // g = 8k + 4 [s_x < 0] + 2 [s_y < 0] + [s_z < 0], k the position of π in the
    // lexicographic list. Error measure: exact equality (integers).
    let all = SignedPermutation::all();
    for (g, e) in all.iter().enumerate() {
        assert_eq!(e.index(), g);
        assert_eq!(SignedPermutation::from_index(g), *e);
        assert_eq!(e.permutation(), PERMUTATIONS[g / 8], "g = {g}");
        let s = e.signs();
        let bits = 4 * (s[0] < 0) as usize + 2 * (s[1] < 0) as usize + (s[2] < 0) as usize;
        assert_eq!(bits, g % 8, "g = {g}");
        // (P v)_a = s_a v_π(a)
        let v = [5, -7, 11];
        let want: [i64; 3] = core::array::from_fn(|a| s[a] * v[e.permutation()[a]]);
        assert_eq!(e.apply(v), want, "g = {g}");
    }
    assert_eq!(all[0], SignedPermutation::IDENTITY);
    assert_eq!(all[0].apply([1, 2, 3]), [1, 2, 3]);
    assert_eq!(all[7].apply([1, 2, 3]), [-1, -2, -3]);
    for e in &all[..8] {
        let m = e.matrix::<f64>();
        assert!((0..3).all(|i| (0..3).all(|j| i == j || m[i][j] == 0.0)));
    }
}

#[test]
fn group_has_48_distinct_elements_24_proper_closed_with_transpose_inverse() {
    // Error measure: exact equality (the matrices have entries 0 and ±1).
    let all = SignedPermutation::all();
    assert_eq!(all.len(), GROUP_ORDER);
    let distinct: BTreeSet<[[i64; 3]; 3]> = all
        .iter()
        .map(|e| e.matrix::<f64>().map(|row| row.map(|v| v as i64)))
        .collect();
    assert_eq!(distinct.len(), 48);
    assert_eq!(all.iter().filter(|e| e.is_proper()).count(), 24);
    for e in &all {
        let m = e.matrix::<f64>();
        assert_eq!(e.det() as f64, det3(&m), "{e:?}");
        assert_eq!(e.is_proper(), e.det() == 1);
        // matrix() and apply() agree.
        for v in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, -3, 5]] {
            assert_eq!(mat_vec(&m, v), e.apply(v));
        }
        // The transpose is the matrix transpose and the inverse.
        let t = e.transpose().matrix::<f64>();
        assert!((0..3).all(|i| (0..3).all(|j| t[i][j] == m[j][i])), "{e:?}");
        assert_eq!(e.compose(&e.transpose()), SignedPermutation::IDENTITY);
        assert_eq!(e.transpose().compose(e), SignedPermutation::IDENTITY);
        assert_eq!(e.transpose().transpose(), *e);
        for f in &all {
            // Closure: the product is an element, and matches the matrix product and
            // the action.
            let c = e.compose(f);
            assert!(all.contains(&c));
            assert_eq!(c.matrix::<f64>(), matmul(&m, &f.matrix()), "{e:?} {f:?}");
            let v = [2, -3, 5];
            assert_eq!(c.apply(v), e.apply(f.apply(v)));
            assert_eq!(c.det(), e.det() * f.det());
        }
    }
}

#[test]
fn octants_form_one_orbit_and_element_o_maps_octant_0_to_o() {
    // CONVENTIONS §3.12: P s_o = s_o′ permutes the octants, and element o,
    // −diag(s_o), is the first with P s₀ = s_o. Error measure: exact equality.
    let signs: Vec<[i64; 3]> = (0..OCTANT_COUNT).map(octant_direction).collect();
    let all = SignedPermutation::all();
    for e in &all {
        let image: BTreeSet<[i64; 3]> = signs.iter().map(|&s| e.apply(s)).collect();
        assert_eq!(image, signs.iter().copied().collect());
    }
    for (o, &s) in signs.iter().enumerate() {
        let first = all.iter().position(|e| e.apply(signs[0]) == s).unwrap();
        assert_eq!(first, o);
        assert_eq!(all[o].apply(signs[0]), s);
    }
}

#[test]
fn offsets_form_the_16_classes_of_section_3_12() {
    // The orbits of the 316 offsets under O_h are the 16 classes, their sizes are those
    // of the table in §3.12 and sum to 316, and each orbit contains exactly one
    // representative, with 0 ≤ d_x ≤ d_y ≤ d_z. Error measure: exact equality.
    let all = SignedPermutation::all();
    let offsets = m2l_offsets();
    let mut orbits: Vec<BTreeSet<[i64; 3]>> = Vec::new();
    for d in offsets {
        if orbits.iter().any(|o| o.contains(&d)) {
            continue;
        }
        let orbit: BTreeSet<[i64; 3]> = all.iter().map(|e| e.apply(d)).collect();
        assert!(orbit.iter().all(|&d| m2l_offset_index(d).is_some()));
        orbits.push(orbit);
    }
    assert_eq!(orbits.len(), M2L_CLASS_COUNT);
    assert_eq!(
        orbits.iter().map(BTreeSet::len).sum::<usize>(),
        M2L_OFFSET_COUNT
    );
    let representatives = class_representatives();
    for (class, rep) in representatives.iter().enumerate() {
        assert!(
            0 <= rep[0] && rep[0] <= rep[1] && rep[1] <= rep[2],
            "{rep:?}"
        );
        let orbit = orbits.iter().find(|o| o.contains(rep)).unwrap();
        assert_eq!(orbit.len(), CLASS_SIZES[class], "class {class}");
        let reps: Vec<_> = orbit
            .iter()
            .filter(|d| 0 <= d[0] && d[0] <= d[1] && d[1] <= d[2])
            .collect();
        assert_eq!(reps, [rep], "class {class}");
    }
    // Class order is the lexicographic order of the representatives.
    assert!(representatives.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(CLASS_SIZES.iter().sum::<usize>(), 316);
}

#[test]
fn class_of_gives_the_first_element_that_maps_the_representative_to_d() {
    // For every offset, class_of gives P with P · representative = d, and P is the
    // first such element in the enumeration order. Error measure: exact equality.
    let all = SignedPermutation::all();
    let representatives = class_representatives();
    let mut sizes = BTreeMap::new();
    let mut used = BTreeSet::new();
    for (index, d) in m2l_offsets().into_iter().enumerate() {
        let (class, element) = class_of(index);
        let rep = representatives[class];
        assert_eq!(element.apply(rep), d, "index {index}");
        let first = all.iter().position(|e| e.apply(rep) == d).unwrap();
        assert_eq!(element.index(), first, "index {index}");
        let mut sorted = d.map(i64::abs);
        sorted.sort_unstable();
        assert_eq!(sorted, rep);
        *sizes.entry(class).or_insert(0) += 1;
        used.insert(element.index());
    }
    assert_eq!(sizes.values().copied().collect::<Vec<_>>(), CLASS_SIZES);
    // Every representative maps to itself by I, and all 48 elements occur.
    for rep in representatives {
        let (_, element) = class_of(m2l_offset_index(rep).unwrap());
        assert_eq!(element, SignedPermutation::IDENTITY);
    }
    assert_eq!(used.len(), 48);
    eprintln!("class_of uses {} distinct group elements", used.len());
}

#[test]
#[should_panic(expected = "M2L offset index 316 out of range for 316 offsets")]
fn class_of_rejects_index_316() {
    let _ = class_of(M2L_OFFSET_COUNT);
}

#[test]
#[should_panic(expected = "group element index must be below 48")]
fn from_index_rejects_48() {
    let _ = SignedPermutation::from_index(48);
}

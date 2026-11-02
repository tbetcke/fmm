//! The symmetry-reduced M2L tables of the Helmholtz family (Helmholtz CH3.2b,
//! `nd_fmm_tables::helmholtz::classes`): the cube-group transforms against the orthonormal
//! harmonics of CONVENTIONS §3.15 (absolute error, values of order one; 2.2e-15 measured on
//! the M6) and their structure
//! (exact); every offset against Dense in the FMM-weighted measure of this test executable
//! within [`BOUND`] times the conditioning (3.7e-14 measured on the M6: a margin of about
//! five, as Dense's against Rotation, which CI has run since CH3.2a; the matrices are
//! Dense's bit for bit, so only the transforms' rounding differs); the class matrices and the representatives bit
//! for bit Dense; the vectors, the f32 tables and the level selection by the budget.

use nd_fmm_math::orthonormal::harmonics;
use nd_fmm_math::{Complex, Layout};
use nd_fmm_ref::helmholtz::Level;
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offset_index};
use nd_fmm_tables::helmholtz::classes::{ClassM2l, ClassScratch, CubeTransforms};
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use nd_fmm_tables::symmetry::{SignedPermutation, class_representatives};

use super::{BOUND, K_ROOTS, assert_same, compare, conditioning, input, weights, widened};

const DEGREES: [usize; 5] = [0, 0, 7, 6, 5];
const LEVELS: [bool; 5] = [false, false, true, true, true];

/// Seeded unit-scale points, not on an axis or a symmetry plane.
fn points() -> Vec<[f64; 3]> {
    (0..12)
        .map(|i| {
            let t = 0.7 + 0.53 * i as f64;
            [
                t.cos() * (1.3 * t).sin(),
                (0.9 * t).sin(),
                0.4 + (1.7 * t).cos(),
            ]
        })
        .collect()
}

/// O(P) applied to a whole expansion: a fresh vector.
fn transformed(t: &CubeTransforms<f64>, element: SignedPermutation, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    t.apply(element, x, &mut out);
    out
}

/// **Y**_n(P x̂) = Oⁿ(P) **Y**_n(x̂) for all 48 elements, the defining identity.
#[test]
fn transforms_map_the_harmonics() {
    let p = 12;
    let len = Layout::new(p).len();
    let t = CubeTransforms::<f64>::new(p);
    let mut worst = 0.0f64;
    for x in points() {
        let mut y = vec![0.0; len];
        harmonics(p, x, &mut y);
        for element in SignedPermutation::all() {
            let px = element
                .matrix::<f64>()
                .map(|row| row[0] * x[0] + row[1] * x[1] + row[2] * x[2]);
            let mut expected = vec![0.0; len];
            harmonics(p, px, &mut expected);
            let got = transformed(&t, element, &y);
            for (a, b) in got.iter().zip(&expected) {
                worst = worst.max((a - b).abs());
            }
        }
    }
    println!("Y(P x) against O(P) Y(x), p = {p}: worst {worst:.2e}");
    assert!(worst <= 3e-14, "{worst:e}");
}

/// O Oᵀ = I and O(P₁ P₂) = O(P₁) O(P₂) on all pairs, applied to a seeded expansion whose
/// degree-n coefficients are of size 2⁻ⁿ (1.1e-16 measured on the M6); and every block's
/// Gram matrix the identity entry by entry, which weighs every degree alike.
#[test]
fn transforms_are_orthogonal_and_a_homomorphism() {
    let p = 8;
    let t = CubeTransforms::<f64>::new(p);
    let all = SignedPermutation::all();
    let mut worst = 0.0f64;
    let x = input(p, 1, 5);
    for a in all {
        // Oᵀ = O(Pᵀ): the round trip is the identity.
        let back = transformed(&t, a.transpose(), &transformed(&t, a, &x));
        for (u, v) in back.iter().zip(&x) {
            worst = worst.max((u - v).abs());
        }
        for b in all {
            let product = transformed(&t, a.compose(&b), &x);
            let chained = transformed(&t, a, &transformed(&t, b, &x));
            for (u, v) in product.iter().zip(&chained) {
                worst = worst.max((u - v).abs());
            }
        }
    }
    println!("orthogonality and homomorphism, p = {p}: worst {worst:.2e}");
    assert!(worst <= 1e-14, "{worst:e}");
    for n in 0..=p {
        let mut gram = vec![0.0; (2 * n + 1) * (2 * n + 1)];
        for a in all {
            let block = t.block(a, n);
            let w = 2 * n + 1;
            for i in 0..w {
                for j in 0..w {
                    gram[i * w + j] = (0..w).map(|k| block[i * w + k] * block[j * w + k]).sum();
                }
            }
            for i in 0..w {
                for j in 0..w {
                    let e = if i == j { 1.0 } else { 0.0 };
                    assert!((gram[i * w + j] - e).abs() <= 1e-14, "O Oᵀ, degree {n}");
                }
            }
        }
    }
}

/// The 16 elements that map the z-axis to ±itself are signed permutations with entries 0
/// and ±1 that map each slot pair (±m) to itself, diagonal for g < 8; those that exchange x
/// and y (16 ≤ g ≤ 23) exchange the two slots of every pair of odd m and are diagonal on
/// the even m (CONVENTIONS §3.12, "The z-axis elements"); −I is (−1)ⁿ.
#[test]
fn z_axis_elements_are_signed_permutations() {
    let p = 9;
    let t = CubeTransforms::<f64>::new(p);
    for element in SignedPermutation::all() {
        let g = element.index();
        let z_axis = element.permutation()[2] == 2;
        assert_eq!(t.is_permutation(element), z_axis, "element {g}");
        if !z_axis {
            continue;
        }
        for n in 0..=p {
            let w = 2 * n + 1;
            let block = t.block(element, n);
            for i in 0..w {
                let row = &block[i * w..(i + 1) * w];
                let nonzero: Vec<usize> = (0..w).filter(|&j| row[j] != 0.0).collect();
                assert_eq!(nonzero.len(), 1, "element {g}, degree {n}, row {i}");
                let j = nonzero[0];
                assert_eq!(row[j].abs(), 1.0);
                let (mi, mj) = (i as i64 - n as i64, j as i64 - n as i64);
                assert_eq!(mi.abs(), mj.abs(), "element {g}: slot pairs");
                if g < 8 || mi % 2 == 0 {
                    assert_eq!(i, j, "element {g}, order {mi}: diagonal");
                } else {
                    assert_eq!(mi, -mj, "element {g}, order {mi}: the pair exchanged");
                }
            }
        }
    }
    let minus = SignedPermutation::from_index(7);
    for n in 0..=p {
        let w = 2 * n + 1;
        let block = t.block(minus, n);
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        for i in 0..w {
            for j in 0..w {
                assert!(block[i * w + j] == if i == j { sign } else { 0.0 });
            }
        }
    }
}

/// A transform of degree p applied to a plane of a lower degree uses the leading blocks.
#[test]
fn a_lower_degree_plane_uses_the_leading_blocks() {
    let (high, low) = (CubeTransforms::<f64>::new(9), CubeTransforms::<f64>::new(5));
    let x = input(5, 1, 11);
    for element in SignedPermutation::all() {
        assert_same(
            &transformed(&high, element, &x),
            &transformed(&low, element, &x),
            "leading blocks",
        );
    }
}

/// Classes against Dense on every offset of `level`, two planes and, for real k, one: the
/// largest FMM-weighted error over the conditioning.
fn check_against_dense(
    tables: &HelmholtzTables<f64>,
    dense: &DenseM2l<f64>,
    classes: &ClassM2l<f64>,
    level: usize,
) -> f64 {
    let p = tables.degree(level);
    let len = Layout::new(p).len();
    let w = weights(&Level::new(tables.k_hat(level), 1.0), false, p);
    let mut scratch = classes.scratch();
    let plane_counts: &[usize] = if tables.real_shifts() { &[2, 1] } else { &[2] };
    let mut worst = 0.0f64;
    for d in 0..M2L_OFFSET_COUNT {
        for &planes in plane_counts {
            let x = input(p, planes, 7 * d as u64 + 1);
            let mut got = vec![0.5; 2 * len];
            classes.m2l(level, d, &x, planes, &mut got, 1, &mut scratch);
            let mut expected = vec![0.5; 2 * len];
            dense.m2l(level, d, &x, planes, &mut expected, 1);
            worst = worst.max(compare(&got, &expected, 0.5, &w));
        }
    }
    worst / conditioning(tables, level)
}

#[test]
fn every_offset_matches_dense() {
    for (re, im) in K_ROOTS {
        let k = Complex::new(re, im);
        let tables = HelmholtzTables::<f64>::build(k, &DEGREES, DEFAULT_BLOCK_BUDGET);
        let dense = DenseM2l::<f64>::build(k, &DEGREES, &LEVELS);
        let classes = ClassM2l::<f64>::build(k, &DEGREES, &LEVELS);
        let worst = (2..5)
            .map(|level| check_against_dense(&tables, &dense, &classes, level))
            .fold(0.0, f64::max);
        println!("k̂₀ = {re} + {im}i: Classes against Dense, worst {worst:.2e}");
        assert!(worst <= BOUND, "k̂₀ = {re} + {im}i: {worst:e}");
    }
}

/// The class matrices are Dense's of the representatives bit for bit, and at a
/// representative (element I) into a zero local Classes is Dense bit for bit.
#[test]
fn representatives_are_dense_bit_for_bit() {
    for (re, im) in [(2.0, 0.0), (8.0, 1.5)] {
        let k = Complex::new(re, im);
        let degrees = [0, 0, 5];
        let levels = [false, false, true];
        let dense = DenseM2l::<f64>::build(k, &degrees, &levels);
        let classes = ClassM2l::<f64>::build(k, &degrees, &levels);
        let len = Layout::new(5).len();
        let mut scratch = classes.scratch();
        for (class, d0) in class_representatives().into_iter().enumerate() {
            let d = m2l_offset_index(d0).unwrap();
            assert_same(classes.matrix(2, class), dense.matrix(2, d), "class matrix");
            assert_eq!(classes.class_of(d), (class, SignedPermutation::IDENTITY));
            let planes: &[usize] = if im == 0.0 { &[1, 2] } else { &[2] };
            for &planes in planes {
                let x = input(5, planes, d as u64);
                let (mut got, mut expected) = (vec![0.0; 2 * len], vec![0.0; 2 * len]);
                classes.m2l(2, d, &x, planes, &mut got, 1, &mut scratch);
                dense.m2l(2, d, &x, planes, &mut expected, 1);
                assert_same(&got, &expected, "representative");
            }
        }
    }
}

#[test]
fn each_vector_is_the_one_vector_application() {
    for (re, im) in [(3.0, 0.0), (3.0, 0.7)] {
        let k = Complex::new(re, im);
        let degrees = [0, 0, 6, 8];
        let classes = ClassM2l::<f64>::build(k, &degrees, &[false, false, true, true]);
        let mut scratch = classes.scratch();
        let planes: &[usize] = if im == 0.0 { &[1, 2] } else { &[2] };
        for &planes in planes {
            for (level, d) in [(3, 200), (2, 17)] {
                let (vectors, len) = (3, Layout::new(degrees[level]).len());
                let x = input(degrees[level], planes * vectors, 3);
                let mut all = vec![0.25; vectors * 2 * len];
                classes.m2l(level, d, &x, planes, &mut all, vectors, &mut scratch);
                for v in 0..vectors {
                    let mut one = vec![0.25; 2 * len];
                    let xv = &x[v * planes * len..(v + 1) * planes * len];
                    classes.m2l(level, d, xv, planes, &mut one, 1, &mut scratch);
                    assert_same(&all[v * 2 * len..(v + 1) * 2 * len], &one, "M2L vector");
                }
            }
        }
    }
}

/// For real k a real multipole as one plane and as two planes with Im = 0 give the same
/// local: the zero plane's transforms and products are exactly zero.
#[test]
fn one_plane_is_two_planes_with_zero_imaginary_part() {
    let k = Complex::new(2.0, 0.0);
    let degrees = [0, 0, 6];
    let classes = ClassM2l::<f64>::build(k, &degrees, &[false, false, true]);
    let mut scratch = classes.scratch();
    let len = Layout::new(6).len();
    for d in (0..M2L_OFFSET_COUNT).step_by(7) {
        let x = input(6, 1, d as u64);
        let (mut one, mut two) = (vec![0.0; 2 * len], vec![0.0; 2 * len]);
        classes.m2l(2, d, &x, 1, &mut one, 1, &mut scratch);
        classes.m2l(2, d, &widened(&x), 2, &mut two, 1, &mut scratch);
        assert_same(&one, &two, "one plane against two");
    }
}

/// f32 tables are the f64 tables rounded entry by entry, and so are the transforms.
#[test]
fn f32_tables_are_the_f64_tables_rounded() {
    let k = Complex::new(5.0, 0.3);
    let degrees = [0, 0, 6];
    let levels = [false, false, true];
    let (c64, c32) = (
        ClassM2l::<f64>::build(k, &degrees, &levels),
        ClassM2l::<f32>::build(k, &degrees, &levels),
    );
    for class in 0..16 {
        for (a, b) in c64.matrix(2, class).iter().zip(c32.matrix(2, class)) {
            assert!(*b == *a as f32, "class {class}: {b:e} is not {a:e} rounded");
        }
    }
    let (t64, t32) = (c64.transforms().unwrap(), c32.transforms().unwrap());
    for element in SignedPermutation::all() {
        for n in 0..=6 {
            for (a, b) in t64.block(element, n).iter().zip(t32.block(element, n)) {
                assert!(b == *a as f32);
            }
        }
    }
}

#[test]
fn levels_are_selected_from_the_finest_within_the_budget() {
    let bytes = |p: usize| ClassM2l::<f64>::level_bytes(p);
    let transforms = |p: usize| CubeTransforms::<f64>::bytes_for(p);
    assert_eq!(bytes(3), 16 * 2 * 16 * 16 * 8);
    // 16 permutations of 16 slots (u32 and f64) and 32 sets of blocks_len(3) = 84 values.
    assert_eq!(transforms(3), 16 * 16 * 12 + 32 * 84 * 8);
    let degrees = [0, 0, 4, 6, 5, 3];
    let all = ClassM2l::<f64>::select(&degrees, usize::MAX);
    assert_eq!(all, [false, false, true, true, true, true]);
    // Finest first: level 5 (p = 3) and 4 (p = 5, the transforms grow to 5); level 3
    // (p = 6) does not fit, level 2 (p = 4) still does, the transforms staying at 5.
    let budget = bytes(3) + bytes(5) + transforms(5) + bytes(4);
    assert_eq!(
        ClassM2l::<f64>::select(&degrees, budget),
        [false, false, true, false, true, true]
    );
    // The transforms count: one byte short of level 5 alone chooses nothing there.
    let tight = bytes(3) + transforms(3) - 1;
    assert_eq!(ClassM2l::<f64>::select(&[0, 0, 3], tight), [false; 3]);
    assert_eq!(ClassM2l::<f64>::select(&degrees, 0), [false; 6]);

    let k = Complex::new(1.0, 0.0);
    let levels = ClassM2l::<f64>::select(&degrees, budget);
    let classes = ClassM2l::<f64>::build(k, &degrees, &levels);
    assert_eq!(classes.nlevels(), 6);
    for (l, &on) in levels.iter().enumerate() {
        assert_eq!(classes.is_class(l), on, "level {l}");
    }
    assert_eq!(classes.bytes(), budget);
    assert_eq!(classes.transforms().unwrap().p(), 5);
    assert_eq!(classes.degree(3), 6);
}

#[test]
#[should_panic(expected = "no class M2L on level 2")]
fn a_level_without_tables_is_refused() {
    let classes = ClassM2l::<f64>::build(
        Complex::new(1.0, 0.0),
        &[0, 0, 3, 3],
        &[false, false, false, true],
    );
    let len = Layout::new(3).len();
    let mut scratch = classes.scratch();
    classes.m2l(
        2,
        0,
        &vec![0.0; len],
        1,
        &mut vec![0.0; 2 * len],
        1,
        &mut scratch,
    );
}

#[test]
#[should_panic(expected = "multipole length")]
fn a_short_multipole_is_refused() {
    let classes = ClassM2l::<f64>::build(Complex::new(1.0, 0.0), &[0, 0, 3], &[false, false, true]);
    let len = Layout::new(3).len();
    let mut scratch = classes.scratch();
    classes.m2l(
        2,
        0,
        &vec![0.0; len - 1],
        1,
        &mut vec![0.0; 2 * len],
        1,
        &mut scratch,
    );
}

#[test]
#[should_panic(expected = "scratch for a degree below")]
fn a_small_scratch_is_refused() {
    let classes = ClassM2l::<f64>::build(Complex::new(1.0, 0.0), &[0, 0, 3], &[false, false, true]);
    let len = Layout::new(3).len();
    let mut scratch = ClassScratch::new(2);
    classes.m2l(
        2,
        0,
        &vec![0.0; len],
        1,
        &mut vec![0.0; 2 * len],
        1,
        &mut scratch,
    );
}

#[test]
#[should_panic(expected = "levels 0 and 1")]
fn class_tables_on_levels_0_and_1_are_refused() {
    ClassM2l::<f64>::build(Complex::new(1.0, 0.0), &[0, 3, 3], &[false, true, true]);
}

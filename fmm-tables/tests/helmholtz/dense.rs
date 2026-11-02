//! The dense M2L tables of the Helmholtz family (Helmholtz CH3.2a,
//! `nd_fmm_tables::helmholtz::dense`): every offset against the RCR tables of the same
//! level (Rotation) and a sample against `rcr`, in the FMM-weighted measure of this test
//! executable within [`BOUND`] times the conditioning; the columns, the vectors and the
//! f32 tables; the level selection by the budget.

use nd_fmm_math::{Complex, Layout};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, rcr};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};

use super::{BOUND, K_ROOTS, assert_same, compare, conditioning, input, weights, widened};

const DEGREES: [usize; 5] = [0, 0, 7, 6, 5];

/// Dense against Rotation on every offset of `level`, two planes and, for real k, one:
/// the largest FMM-weighted error over the conditioning.
fn check_against_rotation(
    tables: &HelmholtzTables<f64>,
    dense: &DenseM2l<f64>,
    level: usize,
) -> f64 {
    let p = tables.degree(level);
    let len = Layout::new(p).len();
    let w = weights(&Level::new(tables.k_hat(level), 1.0), false, p);
    let mut scratch = tables.scratch();
    let plane_counts: &[usize] = if tables.real_shifts() { &[2, 1] } else { &[2] };
    let mut worst = 0.0f64;
    for d in 0..M2L_OFFSET_COUNT {
        for &planes in plane_counts {
            let x = input(p, planes, 7 * d as u64 + 1);
            let mut got = vec![0.5; 2 * len];
            dense.m2l(level, d, &x, planes, &mut got, 1);
            let mut expected = vec![0.5; 2 * len];
            tables.m2l(level, d, &x, planes, &mut expected, 1, &mut scratch);
            worst = worst.max(compare(&got, &expected, 0.5, &w));
        }
    }
    worst / conditioning(tables, level)
}

#[test]
fn every_offset_matches_rotation() {
    for (re, im) in K_ROOTS {
        let k = Complex::new(re, im);
        let tables = HelmholtzTables::<f64>::build(k, &DEGREES, DEFAULT_BLOCK_BUDGET);
        let dense = DenseM2l::<f64>::build(k, &DEGREES, &[false, false, true, true, true]);
        let worst = (2..5)
            .map(|level| check_against_rotation(&tables, &dense, level))
            .fold(0.0, f64::max);
        println!("k̂₀ = {re} + {im}i: Dense against Rotation, worst {worst:.2e}");
        assert!(worst <= BOUND, "k̂₀ = {re} + {im}i: {worst:e}");
    }
}

#[test]
fn a_sample_matches_rcr() {
    let k = Complex::new(8.0, 1.5);
    let degrees = [0, 0, 0, 9];
    let dense = DenseM2l::<f64>::build(k, &degrees, &[false, false, false, true]);
    let tables = HelmholtzTables::<f64>::build(k, &degrees, DEFAULT_BLOCK_BUDGET);
    let (level, p) = (3, 9);
    let len = Layout::new(p).len();
    let k_level = Level::new(tables.k_hat(level), 1.0);
    let w = weights(&k_level, false, p);
    let from = Frame::new([0.0; 3], 1.0);
    let mut ws = Workspace::new(p);
    let mut worst = 0.0f64;
    for d in (0..M2L_OFFSET_COUNT).step_by(13) {
        let to = Frame::new(m2l_offsets()[d].map(|x| 2.0 * x as f64), 1.0);
        let table = rcr::m2l_table(p, p, &k_level, &from, &to);
        let x = input(p, 2, d as u64);
        let mut got = vec![0.0; 2 * len];
        dense.m2l(level, d, &x, 2, &mut got, 1);
        let mut expected = vec![0.0; 2 * len];
        rcr::m2l(
            p,
            p,
            &k_level,
            &from,
            &to,
            &table,
            &mut ws,
            &x,
            &mut expected,
        );
        worst = worst.max(compare(&got, &expected, 0.0, &w));
    }
    let worst = worst / conditioning(&tables, level);
    println!("Dense against rcr, worst {worst:.2e}");
    assert!(worst <= BOUND, "{worst:e}");
}

/// Column j of a matrix is Rotation's M2L of the unit multipole e_j in f64, bit for bit:
/// the definition of the build.
#[test]
fn columns_are_rotation_of_unit_multipoles() {
    for (re, im) in [(2.0, 0.0), (8.0, 1.5)] {
        let k = Complex::new(re, im);
        let degrees = [0, 0, 5];
        let tables = HelmholtzTables::<f64>::build(k, &degrees, DEFAULT_BLOCK_BUDGET);
        let dense = DenseM2l::<f64>::build(k, &degrees, &[false, false, true]);
        let n = Layout::new(5).len();
        let mut scratch = tables.scratch();
        for d in [0, 57, 158, 315] {
            let a = dense.matrix(2, d);
            assert_eq!(a.len(), 2 * n * n);
            for j in 0..n {
                let mut e = vec![0.0; n];
                e[j] = 1.0;
                let mut column = vec![0.0; 2 * n];
                tables.m2l(2, d, &e, 1, &mut column, 1, &mut scratch);
                assert_same(&a[j * 2 * n..(j + 1) * 2 * n], &column, "column");
            }
        }
    }
}

#[test]
fn each_vector_is_the_one_vector_application() {
    for (re, im) in [(3.0, 0.0), (3.0, 0.7)] {
        let k = Complex::new(re, im);
        let degrees = [0, 0, 6, 8];
        let dense = DenseM2l::<f64>::build(k, &degrees, &[false, false, true, true]);
        let planes: &[usize] = if im == 0.0 { &[1, 2] } else { &[2] };
        for &planes in planes {
            let (vectors, len) = (3, Layout::new(8).len());
            let x = input(8, planes * vectors, 3);
            let mut all = vec![0.25; vectors * 2 * len];
            dense.m2l(3, 200, &x, planes, &mut all, vectors);
            for v in 0..vectors {
                let mut one = vec![0.25; 2 * len];
                let xv = &x[v * planes * len..(v + 1) * planes * len];
                dense.m2l(3, 200, xv, planes, &mut one, 1);
                assert_same(&all[v * 2 * len..(v + 1) * 2 * len], &one, "M2L vector");
            }
        }
    }
}

/// For real k a real multipole as one plane and as two planes with Im = 0 give the same
/// local to rounding (the second adds the products of the zero plane, exactly zero, so
/// bit for bit up to the sign of a zero).
#[test]
fn one_plane_is_two_planes_with_zero_imaginary_part() {
    let k = Complex::new(2.0, 0.0);
    let degrees = [0, 0, 6];
    let dense = DenseM2l::<f64>::build(k, &degrees, &[false, false, true]);
    let len = Layout::new(6).len();
    for d in (0..M2L_OFFSET_COUNT).step_by(7) {
        let x = input(6, 1, d as u64);
        let (mut one, mut two) = (vec![0.0; 2 * len], vec![0.0; 2 * len]);
        dense.m2l(2, d, &x, 1, &mut one, 1);
        dense.m2l(2, d, &widened(&x), 2, &mut two, 1);
        assert_same(&one, &two, "one plane against two");
    }
}

/// f32 tables are the f64 tables rounded entry by entry.
#[test]
fn f32_tables_are_the_f64_tables_rounded() {
    let k = Complex::new(5.0, 0.3);
    let degrees = [0, 0, 6];
    let levels = [false, false, true];
    let (d64, d32) = (
        DenseM2l::<f64>::build(k, &degrees, &levels),
        DenseM2l::<f32>::build(k, &degrees, &levels),
    );
    for d in 0..M2L_OFFSET_COUNT {
        for (a, b) in d64.matrix(2, d).iter().zip(d32.matrix(2, d)) {
            assert!(*b == *a as f32, "offset {d}: {b:e} is not {a:e} rounded");
        }
    }
    assert_eq!(d32.bytes() * 2, d64.bytes());
}

#[test]
fn levels_are_selected_from_the_finest_within_the_budget() {
    let bytes = |p: usize| DenseM2l::<f64>::level_bytes(p);
    assert_eq!(bytes(3), M2L_OFFSET_COUNT * 2 * 16 * 16 * 8);
    let degrees = [0, 0, 4, 6, 5, 3];
    // Everything fits.
    let all = DenseM2l::<f64>::select(&degrees, usize::MAX);
    assert_eq!(all, [false, false, true, true, true, true]);
    // Finest first: levels 5 and 4 fit, level 3 (p = 6) does not, level 2 (p = 4) still does.
    let budget = bytes(3) + bytes(5) + bytes(4);
    assert_eq!(
        DenseM2l::<f64>::select(&degrees, budget),
        [false, false, true, false, true, true]
    );
    assert_eq!(DenseM2l::<f64>::select(&degrees, 0), [false; 6]);
    // f32 halves the bytes.
    assert_eq!(DenseM2l::<f32>::level_bytes(3) * 2, bytes(3));

    let k = Complex::new(1.0, 0.0);
    let levels = DenseM2l::<f64>::select(&degrees, budget);
    let dense = DenseM2l::<f64>::build(k, &degrees, &levels);
    assert_eq!(dense.nlevels(), 6);
    for (l, &on) in levels.iter().enumerate() {
        assert_eq!(dense.is_dense(l), on, "level {l}");
    }
    assert_eq!(dense.bytes(), budget);
    assert_eq!(dense.degree(3), 6);
}

#[test]
#[should_panic(expected = "no dense M2L on level 2")]
fn a_level_without_tables_is_refused() {
    let dense = DenseM2l::<f64>::build(Complex::new(1.0, 0.0), &[0, 0, 3], &[false; 3]);
    let len = Layout::new(3).len();
    dense.m2l(2, 0, &vec![0.0; len], 1, &mut vec![0.0; 2 * len], 1);
}

#[test]
#[should_panic(expected = "multipole length")]
fn a_short_multipole_is_refused() {
    let dense = DenseM2l::<f64>::build(Complex::new(1.0, 0.0), &[0, 0, 3], &[false, false, true]);
    let len = Layout::new(3).len();
    dense.m2l(2, 0, &vec![0.0; len - 1], 1, &mut vec![0.0; 2 * len], 1);
}

#[test]
#[should_panic(expected = "local length")]
fn a_one_plane_local_is_refused() {
    let dense = DenseM2l::<f64>::build(Complex::new(1.0, 0.0), &[0, 0, 3], &[false, false, true]);
    let len = Layout::new(3).len();
    dense.m2l(2, 0, &vec![0.0; len], 1, &mut vec![0.0; len], 1);
}

#[test]
#[should_panic(expected = "levels 0 and 1")]
fn dense_tables_on_levels_0_and_1_are_refused() {
    DenseM2l::<f64>::build(Complex::new(1.0, 0.0), &[0, 3, 3], &[false, true, true]);
}

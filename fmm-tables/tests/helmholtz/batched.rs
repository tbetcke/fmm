//! The batched M2L's tables of the Helmholtz family (Helmholtz CH3.2c,
//! `nd_fmm_tables::helmholtz::batched`): Dense's matrices in split planes, bit for bit on
//! every offset of levels 2–4; Dense's bytes and level selection; the f32 tables; the
//! refusals.

use nd_fmm_math::{Complex, Layout};
use nd_fmm_tables::geometry::M2L_OFFSET_COUNT;
use nd_fmm_tables::helmholtz::batched::BatchedM2l;
use nd_fmm_tables::helmholtz::dense::DenseM2l;

use super::K_ROOTS;

const DEGREES: [usize; 5] = [0, 0, 7, 6, 5];
const LEVELS: [bool; 5] = [false, false, true, true, true];

/// The planes of every offset of every chosen level against Dense's 2n × n matrix: A_r
/// from rows 0..n and A_i from rows n..2n of each column, bit for bit.
#[test]
fn planes_are_dense_matrices_split() {
    for (re, im) in K_ROOTS {
        let k = Complex::new(re, im);
        let dense = DenseM2l::<f64>::build(k, &DEGREES, &LEVELS);
        let batched = BatchedM2l::<f64>::build(k, &DEGREES, &LEVELS);
        assert_eq!(batched.nlevels(), DEGREES.len());
        assert_eq!(batched.bytes(), dense.bytes());
        for level in 2..DEGREES.len() {
            assert!(batched.is_batched(level));
            assert_eq!(batched.degree(level), DEGREES[level]);
            let n = Layout::new(DEGREES[level]).len();
            for d in 0..M2L_OFFSET_COUNT {
                let a = dense.matrix(level, d);
                let (a_r, a_i) = batched.planes(level, d);
                assert_eq!((a_r.len(), a_i.len()), (n * n, n * n));
                for j in 0..n {
                    let column = &a[j * 2 * n..(j + 1) * 2 * n];
                    let same = |x: &[f64], y: &[f64]| {
                        x.iter().zip(y).all(|(u, v)| u.to_bits() == v.to_bits())
                    };
                    assert!(
                        same(&a_r[j * n..(j + 1) * n], &column[..n]),
                        "k̂₀ = {re} + {im}i, level {level}, offset {d}, A_r column {j}"
                    );
                    assert!(
                        same(&a_i[j * n..(j + 1) * n], &column[n..]),
                        "k̂₀ = {re} + {im}i, level {level}, offset {d}, A_i column {j}"
                    );
                }
            }
        }
    }
}

#[test]
fn levels_and_bytes_are_dense_ones() {
    for p in [0, 3, 7, 12] {
        assert_eq!(
            BatchedM2l::<f64>::level_bytes(p),
            DenseM2l::<f64>::level_bytes(p)
        );
        assert_eq!(
            BatchedM2l::<f32>::level_bytes(p),
            DenseM2l::<f32>::level_bytes(p)
        );
    }
    let degrees = [0, 0, 9, 7, 5, 4];
    for budget in [0, 1 << 20, 30 << 20, 60 << 20, usize::MAX] {
        assert_eq!(
            BatchedM2l::<f64>::select(&degrees, budget),
            DenseM2l::<f64>::select(&degrees, budget),
            "budget {budget}"
        );
    }
    // No level chosen: nothing built.
    let none = BatchedM2l::<f64>::build(Complex::new(2.0, 0.0), &degrees, &[false; 6]);
    assert_eq!(none.bytes(), 0);
    assert!((0..6).all(|l| !none.is_batched(l)));
}

#[test]
fn f32_tables_are_the_f64_tables_rounded() {
    let k = Complex::new(2.0, 0.5);
    let levels = [false, false, true, false, false];
    let wide = BatchedM2l::<f64>::build(k, &DEGREES, &levels);
    let narrow = BatchedM2l::<f32>::build(k, &DEGREES, &levels);
    assert!(!narrow.is_batched(3));
    for d in [0, 100, 315] {
        let (w_r, w_i) = wide.planes(2, d);
        let (n_r, n_i) = narrow.planes(2, d);
        for (a, b) in w_r.iter().chain(w_i).zip(n_r.iter().chain(n_i)) {
            assert_eq!((*a as f32).to_bits(), b.to_bits());
        }
    }
}

#[test]
#[should_panic(expected = "no batched M2L on level 3")]
fn a_level_without_tables_is_refused() {
    let k = Complex::new(2.0, 0.0);
    let batched = BatchedM2l::<f64>::build(k, &DEGREES, &[false, false, true, false, false]);
    batched.planes(3, 0);
}

#[test]
#[should_panic(expected = "offset index 316")]
fn an_offset_beyond_316_is_refused() {
    let k = Complex::new(2.0, 0.0);
    let batched = BatchedM2l::<f64>::build(k, &DEGREES, &[false, false, true, false, false]);
    batched.planes(2, 316);
}

#[test]
#[should_panic(expected = "levels 0 and 1")]
fn tables_on_levels_0_and_1_are_refused() {
    let k = Complex::new(2.0, 0.0);
    BatchedM2l::<f64>::build(k, &DEGREES, &[false, true, false, false, false]);
}

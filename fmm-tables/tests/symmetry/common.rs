//! Helpers of the symmetry tests: their degrees and tolerances, the cached dense and
//! class tables, and the product L A R of a transform, a matrix and a transform with
//! its term magnitudes |L| |A| |R|. The shared helpers and the error measure are in
//! `support`, re-exported here.

use std::sync::OnceLock;

use nd_fmm_math::Layout;
use nd_fmm_tables::symmetry::CoefficientTransform;
use nd_fmm_tables::{M2lClasses, M2lTables, MatrixSet};

pub use crate::support::*;

/// Largest degree of the full tables in the debug run.
pub const P_DEBUG: usize = 8;

/// Degrees of the full tables in the debug run.
pub const DEBUG_DEGREES: [usize; 7] = [0, 1, 2, 3, 4, 5, P_DEBUG];

/// Largest degree tested for M2L in f64 (CONVENTIONS §3.9).
pub const P_MAX_M2L: usize = 20;

/// Largest degree tested in f32 (CONVENTIONS §3.9).
pub const P_MAX_F32: usize = 8;

/// Tolerance of the class form against the dense tables, and of the octant tables from
/// octant 0, per degree relative to the terms (C2.2).
pub const CLASS_TOL: f64 = 1e-13;

/// The dense M2L tables of degree p ≤ [`P_DEBUG`] in f64, built once per test binary.
pub fn dense(p: usize) -> &'static M2lTables<f64> {
    static TABLES: [OnceLock<M2lTables<f64>>; P_DEBUG + 1] =
        [const { OnceLock::new() }; P_DEBUG + 1];
    TABLES[p].get_or_init(|| M2lTables::build(p))
}

/// The class form of degree p ≤ [`P_DEBUG`] in f64, built once per test binary.
pub fn classes(p: usize) -> &'static M2lClasses<f64> {
    static TABLES: [OnceLock<M2lClasses<f64>>; P_DEBUG + 1] =
        [const { OnceLock::new() }; P_DEBUG + 1];
    TABLES[p].get_or_init(|| M2lClasses::build(p))
}

/// Adds T x to a zero vector and returns it.
pub fn transform(t: &CoefficientTransform<f64>, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    t.apply(x, &mut out);
    out
}

/// The term magnitudes Σⱼ |Tᵢⱼ| xⱼ of a transform applied to the nonnegative `x`.
pub fn transform_terms(t: &CoefficientTransform<f64>, x: &[f64]) -> Vec<f64> {
    let p = t.p();
    let mut out = vec![0.0; x.len()];
    for (n, range) in Layout::new(p).degrees() {
        let block = t.block(n);
        let x = &x[range.clone()];
        for (o, row) in out[range].iter_mut().zip(block.chunks_exact(2 * n + 1)) {
            *o = row.iter().zip(x).map(|(a, v)| a.abs() * v.abs()).sum();
        }
    }
    out
}

/// The product L A R of the transforms `left` and `right` and matrix `i` of `set`, and
/// its term magnitudes |L| |A| |R|, both n × n column-major. Formed in f64 with the
/// block structure of the transforms, independently of `M2lClasses::expand`.
pub fn product(
    left: &CoefficientTransform<f64>,
    set: &MatrixSet<f64>,
    i: usize,
    right: &CoefficientTransform<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let p = left.p();
    let n = set.n();
    let a = set.matrix(i);
    let (mut ar, mut ar_terms) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (degree, range) in Layout::new(p).degrees() {
        let block = right.block(degree);
        let width = 2 * degree + 1;
        for (j, c) in range.clone().enumerate() {
            for (k, row) in range.clone().enumerate() {
                let t = block[k * width + j];
                for r in 0..n {
                    ar[r + c * n] += a[r + row * n] * t;
                    ar_terms[r + c * n] += (a[r + row * n] * t).abs();
                }
            }
        }
    }
    let (mut lar, mut lar_terms) = (vec![0.0; n * n], vec![0.0; n * n]);
    for c in 0..n {
        let column = &ar[c * n..(c + 1) * n];
        lar[c * n..(c + 1) * n].copy_from_slice(&transform(left, column));
        let column = &ar_terms[c * n..(c + 1) * n];
        lar_terms[c * n..(c + 1) * n].copy_from_slice(&transform_terms(left, column));
    }
    (lar, lar_terms)
}

/// The worst block error of the n × n column-major matrix `got` against `want`,
/// relative to the terms matrix `terms` (the matrix analogue of the "terms, per degree"
/// measure).
///
/// Block (j, n) holds the rows of output degree j and the columns of input degree n.
/// In the orthonormal weighting of both sides, W_out M W_in⁻¹ with W of `output` and
/// `input`, its error is ‖W_out (got − want) W_in⁻¹‖_F over ‖W_out τ W_in⁻¹‖_F, the
/// Frobenius norms of the block. Single columns are not compared: for a unit input some
/// output degrees vanish exactly (L2L output degree 0 from input slot (4, −4) at
/// t = −½ (1, 1, 1) is Im R₄⁴(t) = 0), and both sides then hold rounding residuals whose
/// terms are residuals as well. A whole block cannot vanish unless both sides are
/// exactly zero, as the blocks above the diagonal of L2L and below it of M2M are.
pub fn block_error(
    output: Kind,
    input: Kind,
    p: usize,
    got: &[f64],
    want: &[f64],
    terms: &[f64],
) -> f64 {
    let layout = Layout::new(p);
    let n = layout.len();
    let mut worst: f64 = 0.0;
    for (_, cols) in layout.degrees() {
        for (_, rows) in layout.degrees() {
            let (mut e2, mut s2) = (0.0, 0.0);
            for c in cols.clone() {
                let (nc, mc) = layout.nm(c);
                let wc = weight(input, nc, mc);
                for r in rows.clone() {
                    let (nr, mr) = layout.nm(r);
                    let w = weight(output, nr, mr) / wc;
                    e2 += (w * (got[r + c * n] - want[r + c * n])).powi(2);
                    s2 += (w * terms[r + c * n]).powi(2);
                }
            }
            assert!(
                s2 > 0.0 || e2 == 0.0,
                "block with zero terms but error {e2}"
            );
            if s2 > 0.0 {
                worst = worst.max((e2 / s2).sqrt());
            }
        }
    }
    worst
}

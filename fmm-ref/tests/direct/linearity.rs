//! Linearity and accumulation: each translation of a linear combination is the same
//! combination of the translations, and every operator adds onto nonzero initial
//! values.

use nd_fmm_ref::Workspace;

use crate::common::{
    Op, P_MAX, SplitMix64, Worst, apply, degree_error, degree_norms, len, random_coefficients,
};

/// Tolerance of the linearity comparisons, per degree relative to the term magnitudes
/// of both inputs, scaled by their coefficients.
pub const LINEARITY_TOL: f64 = 1e-14;

/// Tolerance of the accumulation comparisons, per degree relative to the term
/// magnitudes plus the weighted norm of the initial values: one rounding per slot.
pub const ACCUMULATION_TOL: f64 = 1e-15;

#[test]
fn translations_are_linear() {
    // Error measure: coefficients per degree in the orthonormal weighting of the
    // output kind, relative to |α| s(x) + |β| s(y), with s the term magnitudes of the
    // translation (`common`).
    let mut rng = SplitMix64::new(0x750b);
    let mut ws = Workspace::new(P_MAX);
    let worst = Worst::new("linearity, per degree");
    for op in Op::ALL {
        for (from, to) in op.pairs() {
            for p in [0, 1, 5, 12, op.p_max()] {
                let x = random_coefficients(op.input(), p, &mut rng);
                let y = random_coefficients(op.input(), p, &mut rng);
                let (alpha, beta) = (rng.range(-2.0, 2.0), rng.range(-2.0, 2.0));
                let combined: Vec<f64> = x
                    .iter()
                    .zip(&y)
                    .map(|(a, b)| alpha * a + beta * b)
                    .collect();
                let got = apply(op, p, &from, &to, &mut ws, &combined);
                let (tx, ty) = (
                    apply(op, p, &from, &to, &mut ws, &x),
                    apply(op, p, &from, &to, &mut ws, &y),
                );
                let want: Vec<f64> = tx
                    .iter()
                    .zip(&ty)
                    .map(|(a, b)| alpha * a + beta * b)
                    .collect();
                let scale: Vec<f64> = op
                    .scale(p, &from, &to, &x)
                    .iter()
                    .zip(op.scale(p, &from, &to, &y))
                    .map(|(sx, sy)| alpha.abs() * sx + beta.abs() * sy)
                    .collect();
                let e = worst.update(degree_error(op.output(), p, &got, &want, &scale));
                assert!(
                    e <= LINEARITY_TOL,
                    "{op:?} {from:?} -> {to:?}, p = {p}: {e:.3e}"
                );
            }
        }
    }
}

#[test]
fn translations_accumulate_onto_initial_values() {
    // Error measure: coefficients per degree in the orthonormal weighting of the
    // output kind, relative to the term magnitudes plus the weighted norm of the
    // initial values. The result must equal the initial values plus a fresh
    // translation.
    let mut rng = SplitMix64::new(0x750c);
    let mut ws = Workspace::new(P_MAX);
    let worst = Worst::new("accumulation, per degree");
    for op in Op::ALL {
        for (from, to) in op.pairs() {
            for p in [0, 1, 5, 12, op.p_max()] {
                let input = random_coefficients(op.input(), p, &mut rng);
                let initial = random_coefficients(op.output(), p, &mut rng);
                let fresh = apply(op, p, &from, &to, &mut ws, &input);
                let mut accumulated = initial.clone();
                op.function()(p, &from, &to, &mut ws, &input, &mut accumulated);
                let want: Vec<f64> = initial.iter().zip(&fresh).map(|(a, b)| a + b).collect();
                let scale: Vec<f64> = op
                    .scale(p, &from, &to, &input)
                    .iter()
                    .zip(degree_norms(op.output(), p, &initial))
                    .map(|(s, i)| s + i)
                    .collect();
                let e = worst.update(degree_error(op.output(), p, &accumulated, &want, &scale));
                assert!(
                    e <= ACCUMULATION_TOL,
                    "{op:?} {from:?} -> {to:?}, p = {p}: {e:.3e}"
                );

                // A zero input leaves the output unchanged.
                let mut unchanged = initial.clone();
                op.function()(p, &from, &to, &mut ws, &vec![0.0; len(p)], &mut unchanged);
                assert_eq!(unchanged, initial, "{op:?}, p = {p}: zero input");
            }
        }
    }
}

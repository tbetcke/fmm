//! f32 against f64, p ≤ 8 (CONVENTIONS §3.9).

use nd_fmm_ref::{Frame, Workspace};

use crate::common::{
    Op, P_MAX_F32, SplitMix64, Worst, apply, degree_error, len, random_coefficients,
};

/// Tolerance of f32 against f64, per degree relative to the term magnitudes (brief T5).
const F32_TOL: f64 = 1e-5;

/// `frame` rounded to f32, and the rounded frame in f64.
fn rounded(frame: &Frame<f64>) -> (Frame<f32>, Frame<f64>) {
    let single = Frame::new(frame.centre.map(|c| c as f32), frame.radius as f32);
    let double = Frame::new(single.centre.map(f64::from), f64::from(single.radius));
    (single, double)
}

#[test]
fn f32_matches_f64() {
    // Error measure: coefficients per degree in the orthonormal weighting of the
    // output kind, relative to the term magnitudes of the translation (`common`), of
    // the f32 result against the f64 result on the same f32-representable frames and
    // coefficients.
    let mut rng = SplitMix64::new(0x750d);
    let mut ws32 = Workspace::<f32>::new(P_MAX_F32);
    let mut ws64 = Workspace::<f64>::new(P_MAX_F32);
    let worst = Worst::new("f32 vs f64, per degree");
    for op in Op::ALL {
        for (from, to) in op.pairs() {
            let (from32, from64) = rounded(&from);
            let (to32, to64) = rounded(&to);
            for p in 0..=P_MAX_F32 {
                let input32: Vec<f32> = random_coefficients(op.input(), p, &mut rng)
                    .iter()
                    .map(|&c| c as f32)
                    .collect();
                let input64: Vec<f64> = input32.iter().map(|&c| f64::from(c)).collect();
                let mut out32 = vec![0.0f32; len(p)];
                op.function()(p, &from32, &to32, &mut ws32, &input32, &mut out32);
                let got: Vec<f64> = out32.iter().map(|&c| f64::from(c)).collect();
                let want = apply(op, p, &from64, &to64, &mut ws64, &input64);
                let scale = op.scale(p, &from64, &to64, &input64);
                let e = worst.update(degree_error(op.output(), p, &got, &want, &scale));
                assert!(e <= F32_TOL, "{op:?} {from:?} -> {to:?}, p = {p}: {e:.3e}");
            }
        }
    }
}

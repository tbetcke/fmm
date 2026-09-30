//! f32 against f64, p ≤ 8 (CONVENTIONS §3.9).

use nd_fmm_ref::{Frame, Workspace};

use crate::common::{
    Op, P_MAX_F32, SplitMix64, Worst, apply, degree_error, len, place, random_coefficients,
};
use crate::precision::{F32_TOL, rounded};

#[test]
fn f32_matches_f64() {
    // The rotation operators in f32 against the direct operators in f64, on the same
    // f32-representable frames and coefficients: the typical pairs of `Op::pairs`,
    // which rotate, and a shift along −z, which does not.
    //
    // Error measure: per degree, weighted, relative to the term magnitudes of the
    // translation (`common`), 1e-5, as for `direct` (`precision`).
    let mut rng = SplitMix64::new(0x7608);
    let mut ws32 = Workspace::<f32>::new(P_MAX_F32);
    let mut ws64 = Workspace::<f64>::new(P_MAX_F32);
    let worst = Worst::new("rotation f32 vs direct f64, per degree");
    for op in Op::ALL {
        let mut pairs = op.pairs();
        let (from, _) = pairs[0];
        let shift = if let Op::M2l = op { -4.0 } else { -0.5 };
        pairs.push((
            from,
            Frame::new(place(&from, [0.0, 0.0, shift]), from.radius),
        ));
        for (from, to) in pairs {
            let (from32, from64) = rounded(&from);
            let (to32, to64) = rounded(&to);
            for p in 0..=P_MAX_F32 {
                let input32: Vec<f32> = random_coefficients(op.input(), p, &mut rng)
                    .iter()
                    .map(|&c| c as f32)
                    .collect();
                let input64: Vec<f64> = input32.iter().map(|&c| f64::from(c)).collect();
                let mut out32 = vec![0.0f32; len(p)];
                op.rotation()(p, &from32, &to32, &mut ws32, &input32, &mut out32);
                let got: Vec<f64> = out32.iter().map(|&c| f64::from(c)).collect();
                let want = apply(op, p, &from64, &to64, &mut ws64, &input64);
                let scale = op.scale(p, &from64, &to64, &input64);
                let e = worst.update(degree_error(op.output(), p, &got, &want, &scale));
                assert!(e <= F32_TOL, "{op:?} {from:?} -> {to:?}, p = {p}: {e:.3e}");
            }
        }
    }
}

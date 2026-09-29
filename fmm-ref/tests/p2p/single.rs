//! f32: `p2p` in single precision matches `direct_sum` on the same (f32-representable)
//! inputs to 1e-6 relative to the sum of term magnitudes.

use nd_fmm_ref::p2p::{direct_sum, p2p};

use crate::common::{CUBES, SplitMix64, Worst, gradient_error, magnitudes, potential_error};

/// Tolerance of `p2p::<f32>` against `direct_sum`, relative to the sum of term
/// magnitudes.
const TOL: f64 = 1e-6;

fn narrow(points: &[[f64; 3]]) -> Vec<[f32; 3]> {
    points.iter().map(|p| p.map(|c| c as f32)).collect()
}

fn widen(points: &[[f32; 3]]) -> Vec<[f64; 3]> {
    points.iter().map(|p| p.map(f64::from)).collect()
}

#[test]
fn f32_p2p_matches_direct_sum() {
    // Error measure: per target, |p2p::<f32> − direct_sum| / Σⱼ |qⱼ| / rᵢⱼ for the
    // potential and per component / Σⱼ |qⱼ| / rᵢⱼ² for the gradient. `direct_sum` runs
    // on the f32 inputs widened exactly to f64, so the difference is the f32 rounding
    // of `p2p` alone.
    let mut worst = [
        Worst::new("p2p f32 vs direct_sum, potential"),
        Worst::new("p2p f32 vs direct_sum, gradient"),
    ];
    let mut rng = SplitMix64::new(0xf32);
    for (centre, half) in CUBES {
        for (ns, nt) in [(1, 16), (16, 16), (64, 64), (256, 16)] {
            let (sources, charges) = rng.charged(ns, centre, half);
            let targets: Vec<[f64; 3]> = (0..nt).map(|_| rng.in_cube(centre, half)).collect();
            let (sources32, targets32) = (narrow(&sources), narrow(&targets));
            let charges32: Vec<f32> = charges.iter().map(|&q| q as f32).collect();
            let (sources, targets) = (widen(&sources32), widen(&targets32));
            let charges: Vec<f64> = charges32.iter().map(|&q| f64::from(q)).collect();

            let n = targets.len();
            let (mut phi32, mut grad32) = (vec![0.0f32; n], vec![[0.0f32; 3]; n]);
            p2p(
                &sources32,
                &charges32,
                &targets32,
                &mut phi32,
                Some(&mut grad32),
            );
            let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
            direct_sum(&sources, &charges, &targets, &mut phi, Some(&mut grad));
            let (scale, gradient_scale) = magnitudes(&sources, &charges, &targets);

            let phi32: Vec<f64> = phi32.iter().map(|&v| f64::from(v)).collect();
            let grad32: Vec<[f64; 3]> = grad32.iter().map(|g| g.map(f64::from)).collect();
            let e = worst[0].update(potential_error(&phi32, &phi, &scale));
            assert!(e <= TOL, "{ns} sources: potential error {e:e}");
            let e = worst[1].update(gradient_error(&grad32, &grad, &gradient_scale));
            assert!(e <= TOL, "{ns} sources: gradient error {e:e}");
        }
    }
}

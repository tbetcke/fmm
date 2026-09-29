//! Gradients: both functions equal the analytic −Σⱼ qⱼ (xᵢ − yⱼ) / |xᵢ − yⱼ|³ to 1e-15
//! relative, and agree with central differences of the potential to 1e-7.

use nd_fmm_ref::p2p::{direct_sum, p2p};

use crate::common::{
    CUBES, SplitMix64, Worst, dd_reference, gradient_error, magnitudes, potential_error,
};

/// Tolerance against the analytic gradient, relative to Σⱼ |qⱼ| / rᵢⱼ².
const ANALYTIC_TOL: f64 = 1e-15;

/// Tolerance against central differences, relative to Σⱼ |qⱼ| / rᵢⱼ².
const DIFFERENCE_TOL: f64 = 1e-7;

#[test]
fn gradient_equals_analytic() {
    // Error measure: per target and component, |gradient − analytic| / Σⱼ |qⱼ| / rᵢⱼ².
    // The analytic gradient is evaluated in double-double arithmetic
    // (`common::dd_reference`). The potential of `p2p` against the same reference, per
    // target / Σⱼ |qⱼ| / rᵢⱼ, is only reported.
    let mut worst = [
        Worst::new("p2p gradient vs analytic"),
        Worst::new("direct_sum gradient vs analytic"),
        Worst::new("p2p potential vs double-double (reported)"),
    ];
    let mut rng = SplitMix64::new(0x62ad);
    for (centre, half) in CUBES {
        for (ns, nt) in [(1, 16), (8, 16), (64, 64)] {
            let (sources, charges) = rng.charged(ns, centre, half);
            let targets: Vec<[f64; 3]> = (0..nt).map(|_| rng.in_cube(centre, half)).collect();
            let (reference, reference_gradient) = dd_reference(&sources, &charges, &targets);
            let (scale, gradient_scale) = magnitudes(&sources, &charges, &targets);

            let n = targets.len();
            let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
            p2p(
                &sources,
                &charges,
                &targets,
                &mut potential,
                Some(&mut gradient),
            );
            let e = worst[0].update(gradient_error(
                &gradient,
                &reference_gradient,
                &gradient_scale,
            ));
            assert!(e <= ANALYTIC_TOL, "p2p, {ns} sources: gradient error {e:e}");
            worst[2].update(potential_error(&potential, &reference, &scale));

            let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
            direct_sum(
                &sources,
                &charges,
                &targets,
                &mut potential,
                Some(&mut gradient),
            );
            let e = worst[1].update(gradient_error(
                &gradient,
                &reference_gradient,
                &gradient_scale,
            ));
            assert!(
                e <= ANALYTIC_TOL,
                "direct_sum, {ns} sources: gradient error {e:e}"
            );
        }
    }
}

#[test]
fn gradient_matches_central_differences() {
    // Error measure: per target and component, |gradient − difference quotient| /
    // Σⱼ |qⱼ| / rᵢⱼ². The difference quotient is (φ(x + h eₖ) − φ(x − h eₖ)) / (2h)
    // with the potential from `direct_sum`, h = 10⁻⁵ times the cube half-width, and the
    // actually representable step in the denominator. Targets keep a distance of at
    // least a tenth of the half-width from every source, so the O(h²) truncation error
    // stays near 10⁻⁸ relative.
    let mut worst = [
        Worst::new("p2p gradient vs central differences"),
        Worst::new("direct_sum gradient vs central differences"),
    ];
    let mut rng = SplitMix64::new(0xd1ff);
    for (centre, half) in CUBES {
        let (sources, charges) = rng.charged(32, centre, half);
        let targets: Vec<[f64; 3]> = std::iter::repeat_with(|| rng.in_cube(centre, half))
            .filter(|x| {
                sources.iter().all(|y| {
                    let r2: f64 = (0..3).map(|k| (x[k] - y[k]).powi(2)).sum();
                    r2.sqrt() >= 0.1 * half
                })
            })
            .take(16)
            .collect();
        let n = targets.len();
        let (_, gradient_scale) = magnitudes(&sources, &charges, &targets);

        let h = 1e-5 * half;
        let mut quotient = vec![[0.0; 3]; n];
        for (x, dq) in targets.iter().zip(&mut quotient) {
            for (k, dqk) in dq.iter_mut().enumerate() {
                let (mut plus, mut minus) = (*x, *x);
                plus[k] += h;
                minus[k] -= h;
                let mut phi = [0.0; 2];
                direct_sum(&sources, &charges, &[plus, minus], &mut phi, None);
                *dqk = (phi[0] - phi[1]) / (plus[k] - minus[k]);
            }
        }

        let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
        p2p(
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        let e = worst[0].update(gradient_error(&gradient, &quotient, &gradient_scale));
        assert!(e <= DIFFERENCE_TOL, "p2p: error {e:e}");

        let (mut potential, mut gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
        direct_sum(
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        let e = worst[1].update(gradient_error(&gradient, &quotient, &gradient_scale));
        assert!(e <= DIFFERENCE_TOL, "direct_sum: error {e:e}");
    }
}

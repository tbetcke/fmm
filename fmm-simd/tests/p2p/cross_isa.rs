//! Cross-ISA agreement: on a machine with several ISAs, every ISA agrees with every
//! other within twice the tolerances of `terms` and `sums` (design §5.5: results
//! differ between ISAs, within the accuracy contract).

use nd_fmm_simd::P2pKernel;

use crate::common::{Oracle, Outputs, Precision, Problem, Rng, kernels, run, uniform};

/// The largest difference between `a` and `b` relative to the oracle's scales of
/// `problem`, for the potential and the gradient.
fn difference<T: Precision>(oracle: &Oracle, a: &Outputs<T>, b: &Outputs<T>) -> (f64, f64) {
    let relative = |x: T, y: T, scale: f64| (x.to_f64() - y.to_f64()).abs() / scale;
    let mut ep = 0.0f64;
    for (i, (&x, &y)) in a.0.iter().zip(&b.0).enumerate() {
        ep = ep.max(relative(x, y, oracle.potential_scale[i]));
    }
    let mut eg = 0.0f64;
    if let (Some(ga), Some(gb)) = (&a.1, &b.1) {
        for (i, (x, y)) in ga.iter().zip(gb).enumerate() {
            for k in 0..3 {
                eg = eg.max(relative(x[k], y[k], oracle.gradient_scale[i]));
            }
        }
    }
    (ep, eg)
}

/// Every pair of available ISAs, on seeded sums (twice the sum tolerance) and on
/// single pairs (twice the term tolerances).
fn cross_isa<T: Precision>(test: &str) {
    let kernels: Vec<P2pKernel<T>> = kernels::<T>(test);
    let mut problems: Vec<Problem<T>> = [(7, 9), (64, 25), (1000, 17), (4096, 8)]
        .into_iter()
        .enumerate()
        .map(|(seed, (n_s, n_t))| uniform(n_s, n_t, 0xc0_55 + seed as u64))
        .collect();
    // Single pairs at random separations within [2⁻⁴, 2³].
    let mut rng = Rng(0xc0_56);
    let pairs: Vec<Problem<T>> = (0..500)
        .map(|_| {
            let mut p = uniform(1, 1, rng.next_u64());
            let s = T::from_f64(rng.uniform(-4.0, 3.0).exp2());
            p.targets[0] = p.targets[0].map(|c| c * s);
            p
        })
        .collect();
    problems.extend(pairs);
    let mut worst = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (i, a) in kernels.iter().enumerate() {
        for b in &kernels[i + 1..] {
            for problem in &problems {
                let oracle = Oracle::new(problem);
                let single = problem.sources.len() == 1;
                let (tol_p, tol_g) = if single {
                    (2.0 * 8.0 * T::U, 2.0 * 16.0 * T::U)
                } else {
                    (2.0 * T::SUM_TOL, 2.0 * T::SUM_TOL)
                };
                for with_gradient in [false, true] {
                    let (ep, eg) = difference(
                        &oracle,
                        &run(*a, problem, with_gradient),
                        &run(*b, problem, with_gradient),
                    );
                    assert!(
                        ep <= tol_p && eg <= tol_g,
                        "{} against {}: {ep:e}, {eg:e} (n_s = {}, n_t = {})",
                        a.isa(),
                        b.isa(),
                        problem.sources.len(),
                        problem.targets.len()
                    );
                    if single {
                        worst.0 = worst.0.max(ep / T::U);
                        worst.1 = worst.1.max(eg / T::U);
                    } else {
                        worst.2 = worst.2.max(ep);
                        worst.3 = worst.3.max(eg);
                    }
                }
            }
            println!(
                "{test}: {} against {}: largest difference: terms {:.3} u (potential), {:.3} u \
                 (gradient); sums {:.3e} (potential), {:.3e} (gradient)",
                a.isa(),
                b.isa(),
                worst.0,
                worst.1,
                worst.2,
                worst.3
            );
        }
    }
    if kernels.len() < 2 {
        println!("{test}: only one ISA available; nothing to compare");
    }
}

#[test]
fn isas_agree_f32() {
    cross_isa::<f32>("isas_agree_f32");
}

#[test]
fn isas_agree_f64() {
    cross_isa::<f64>("isas_agree_f64");
}

//! The ends of the kernel domain (CONVENTIONS §3.13): pairs at the smallest and the
//! largest r² give finite terms within the term tolerances of the reference (8 u_T for
//! the potential, 16 u_T relative to |q| / r² for the gradient), and a sum over them
//! within the sum tolerance of `direct_sum`. In f32 the gradient contract starts at
//! r² = 2⁻⁸⁴ (§3.13, "Range of the terms in f32"), so the gradient is checked there
//! and the potential alone at 2⁻¹⁰⁸.

use crate::common::{
    Oracle, Precision, Problem, Worst, assert_within, kernels, run, run_reference,
};

/// Single pairs (target, source) with r² at the ends: a separation h = √`r2_min`
/// along one axis (r² = `r2_min`) or along all three (3 `r2_min`), with the points at
/// the origin and at leaf-scale coordinates, and separations (±8, ±8, 0) (r² = 2⁷).
fn end_pairs<T: Precision>(r2_min: f64) -> Vec<([T; 3], [T; 3])> {
    let h = T::from_f64(r2_min.sqrt());
    let z = T::zero();
    let one = T::one();
    let half = T::from_f64(0.5);
    vec![
        ([h, z, z], [z, z, z]),
        ([z, z, z], [z, -h, z]),
        ([h, h, h], [z, z, z]),
        ([z, half, half], [-h, half, half]),
        ([one, one, z], [T::from_f64(-7.0), T::from_f64(-7.0), z]),
        (
            [T::from_f64(-3.0), one, z],
            [T::from_f64(5.0), T::from_f64(-7.0), z],
        ),
    ]
}

/// Every end pair as a one-pair problem, and all of them as sources of one target.
fn domain_ends<T: Precision>(test: &str) {
    let ends = [
        (false, crate::common::DOMAIN_MIN),
        (true, T::GRADIENT_R2_MIN),
    ];
    for (with_gradient, r2_min) in ends {
        let mut worst = Worst::new(format!(
            "{test} ({}, r² from {r2_min:e}, gradient: {with_gradient})",
            std::any::type_name::<T>()
        ));
        let pairs = end_pairs::<T>(r2_min);
        for kernel in kernels::<T>(test) {
            for (i, &(x, y)) in pairs.iter().enumerate() {
                for q in [T::one(), -T::one(), T::from_f64(0.3)] {
                    let p = Problem {
                        sources: vec![y],
                        charges: vec![q],
                        targets: vec![x],
                        potential: vec![T::zero()],
                        gradient: vec![[T::zero(); 3]],
                    };
                    let r2: f64 = (0..3)
                        .map(|k| (x[k].to_f64() - y[k].to_f64()).powi(2))
                        .sum();
                    assert!((r2_min..=crate::common::DOMAIN_MAX).contains(&r2));
                    let (kp, kg) = run(kernel, &p, with_gradient);
                    let (rp, rg) = run_reference(&p, with_gradient);
                    let ep = ((kp[0].to_f64() - rp[0].to_f64()) / rp[0].to_f64()).abs();
                    let mut eg = 0.0f64;
                    if let (Some(kg), Some(rg)) = (kg, rg) {
                        for k in 0..3 {
                            let e = (kg[0][k].to_f64() - rg[0][k].to_f64()).abs();
                            eg = eg.max(e * r2 / q.to_f64().abs());
                        }
                    }
                    assert!(
                        ep <= 8.0 * T::U && eg <= 16.0 * T::U,
                        "{} pair {i} (r² = {r2:e}): {:.3} u, {:.3} u",
                        kernel.isa(),
                        ep / T::U,
                        eg / T::U
                    );
                    worst.record(kernel.isa(), (ep, eg));
                }
            }
            // All end pairs' sources at one target: the sum.
            let x = pairs[0].0;
            let sources: Vec<[T; 3]> = pairs
                .iter()
                .map(|&(xi, y)| [0, 1, 2].map(|k| x[k] - (xi[k] - y[k])))
                .collect();
            let n = sources.len();
            let p = Problem {
                charges: (0..n)
                    .map(|j| T::from_f64(if j % 2 == 0 { 1.0 } else { -0.5 }))
                    .collect(),
                sources,
                targets: vec![x],
                potential: vec![T::zero()],
                gradient: vec![[T::zero(); 3]],
            };
            let oracle = Oracle::new(&p);
            let errors = oracle.errors(&run(kernel, &p, with_gradient));
            assert_within(errors, oracle.tolerance::<T>(), &kernel.isa());
        }
        worst.print(T::U, "u");
    }
}

#[test]
fn domain_ends_f32() {
    domain_ends::<f32>("domain_ends_f32");
}

#[test]
fn domain_ends_f64() {
    domain_ends::<f64>("domain_ends_f64");
}

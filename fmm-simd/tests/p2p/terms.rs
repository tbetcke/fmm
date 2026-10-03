//! Terms: one source and one target over seeded separations across the kernel domain
//! (design §3, requirement 2). The potential is within 8 u_T of the reference's term,
//! relative; each gradient component within 16 u_T relative to |q| / r².

use nd_fmm_simd::P2pKernel;

use crate::common::{DOMAIN_MAX, DOMAIN_MIN, Precision, Problem, Rng, Worst, kernels};

/// One pair with r² log-uniform in [`r2_min`, 2⁷] and a charge uniform in [−1, 1).
///
/// The source lies in a cube of side about 32 r around the origin, so that the target
/// at a separation r from it stays distinct from it when rounded to `T`; the
/// separation in `T` is then taken as it is, and the pair redrawn if its r² left the
/// interval.
fn pair<T: Precision>(rng: &mut Rng, r2_min: f64) -> Problem<T> {
    loop {
        let r = rng.uniform(r2_min.log2(), DOMAIN_MAX.log2()).exp2().sqrt();
        let direction = [0; 3].map(|_| rng.uniform(-1.0, 1.0));
        let norm = direction.iter().map(|c| c * c).sum::<f64>().sqrt();
        if norm < 0.1 {
            continue;
        }
        let side = (16.0 * r).min(1.0);
        let y: [T; 3] = [0; 3].map(|_| T::from_f64(side * rng.uniform(-1.0, 1.0)));
        let x: [T; 3] = [0, 1, 2].map(|k| T::from_f64(y[k].to_f64() + r * direction[k] / norm));
        let r2: f64 = (0..3)
            .map(|k| (x[k].to_f64() - y[k].to_f64()).powi(2))
            .sum();
        let q = T::from_f64(rng.uniform(-1.0, 1.0));
        if !(r2_min..=DOMAIN_MAX).contains(&r2) || q == T::zero() {
            continue;
        }
        return Problem {
            sources: vec![y],
            charges: vec![q],
            targets: vec![x],
            potential: vec![T::zero()],
            gradient: vec![[T::zero(); 3]],
        };
    }
}

/// The relative errors of one pair: the potential against the reference's term, the
/// gradient components against |q| / r² (r² in f64 from the inputs).
fn pair_errors<T: Precision>(
    kernel: P2pKernel<T>,
    p: &Problem<T>,
    with_gradient: bool,
) -> (f64, f64) {
    let (kp, kg) = crate::common::run(kernel, p, with_gradient);
    let (rp, rg) = crate::common::run_reference(p, with_gradient);
    let finite = |v: T| v.to_f64().is_finite();
    assert!(
        kp.iter().copied().all(finite) && kg.iter().flatten().flatten().copied().all(finite),
        "{} non-finite term at {p:?}",
        kernel.isa()
    );
    let ep = ((kp[0].to_f64() - rp[0].to_f64()) / rp[0].to_f64()).abs();
    let mut eg = 0.0f64;
    if let (Some(kg), Some(rg)) = (kg, rg) {
        let r2: f64 = (0..3)
            .map(|k| (p.targets[0][k].to_f64() - p.sources[0][k].to_f64()).powi(2))
            .sum();
        let scale = p.charges[0].to_f64().abs() / r2;
        for k in 0..3 {
            eg = eg.max((kg[0][k].to_f64() - rg[0][k].to_f64()).abs() / scale);
        }
    }
    (ep, eg)
}

/// `n` pairs per ISA, potential only over the whole domain and with gradients over the
/// gradient's range ([`Precision::GRADIENT_R2_MIN`] to 2⁷).
fn terms<T: Precision>(test: &str, n: usize) {
    let (potential_tol, gradient_tol) = (8.0 * T::U, 16.0 * T::U);
    for (with_gradient, r2_min) in [(false, DOMAIN_MIN), (true, T::GRADIENT_R2_MIN)] {
        let mut worst = Worst::new(format!(
            "{test} ({}, {n} pairs, r² in [{r2_min:e}, 2⁷], gradient: {with_gradient})",
            std::any::type_name::<T>()
        ));
        for kernel in kernels::<T>(test) {
            let mut rng = Rng(0x7e57_0001 + u64::from(with_gradient));
            for _ in 0..n {
                let p = pair::<T>(&mut rng, r2_min);
                let (ep, eg) = pair_errors(kernel, &p, with_gradient);
                assert!(
                    ep <= potential_tol && eg <= gradient_tol,
                    "{} pair {p:?}: errors {:.3} u (potential), {:.3} u (gradient)",
                    kernel.isa(),
                    ep / T::U,
                    eg / T::U
                );
                worst.record(kernel.isa(), (ep, eg));
            }
        }
        worst.print(T::U, "u");
    }
}

#[test]
fn terms_f32() {
    terms::<f32>("terms_f32", 4096);
}

#[test]
fn terms_f64() {
    terms::<f64>("terms_f64", 4096);
}

#[test]
#[ignore = "10⁶ pairs; run in release with --ignored"]
fn terms_f32_large() {
    terms::<f32>("terms_f32_large", 1_000_000);
}

#[test]
#[ignore = "10⁶ pairs; run in release with --ignored"]
fn terms_f64_large() {
    terms::<f64>("terms_f64_large", 1_000_000);
}

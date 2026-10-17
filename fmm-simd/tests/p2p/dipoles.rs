//! `P2pKernel::evaluate_dipoles` (Phase 6 T8, C6.4; CONVENTIONS §3.7, "Dipole sources")
//! on every ISA this machine offers, against `nd_fmm_ref::p2p::{p2p_dipoles,
//! direct_sum_dipoles}`.
//!
//! Error measures:
//! - *Terms*: one source and one target with r² log-uniform over the kernel domain (in
//!   f32 over the dipole range [2⁻⁸⁰, 2⁷], CONVENTIONS §3.13, "Range of the terms in
//!   f32") and moments in [−1, 1)³: the potential against the reference's term in the same
//!   precision relative to |d| / r², each gradient component relative to |d| / r³, in
//!   units of u_T; tolerances [`POTENTIAL_TERM_U`] and [`GRADIENT_TERM_U`] u_T
//!   (docs/design/simd-p2p.md §4.8).
//! - *Sums*: against `direct_sum_dipoles` on the exactly widened inputs, relative per
//!   target to Σⱼ |dⱼ| / rⱼ² (potential) and Σⱼ |dⱼ| / rⱼ³ (each gradient component):
//!   1e-5 (f32) and 1e-13 (f64).
//! - *Bits*: the scalar path against `p2p_dipoles` bit for bit; chunk and target-position
//!   invariance bit for bit on every ISA; coincident pairs (sources equal to the targets)
//!   add nothing.

use nd_fmm_ref::p2p::{direct_sum_dipoles, p2p_dipoles};
use nd_fmm_simd::{Isa, P2pKernel};

use crate::common::{DOMAIN_MAX, DOMAIN_MIN, Precision, Rng, Worst, kernels, widen};

/// The potential term's tolerance, in u_T relative to |d| / r².
const POTENTIAL_TERM_U: f64 = 32.0;
/// Each gradient component's tolerance, in u_T relative to |d| / r³.
const GRADIENT_TERM_U: f64 = 176.0;
/// The smallest r² of the f32 dipole contract with |dₖ| ≤ 1 (2⁻⁸⁰).
const F32_DIPOLE_R2_MIN: f64 = 8.271806125530277e-25;

/// A dipole problem: sources, moments, targets and the initial outputs.
#[derive(Clone, Debug)]
struct Problem<T> {
    sources: Vec<[T; 3]>,
    dipoles: Vec<[T; 3]>,
    targets: Vec<[T; 3]>,
    potential: Vec<T>,
    gradient: Vec<[T; 3]>,
}

type Outputs<T> = (Vec<T>, Vec<[T; 3]>);

fn run<T: Precision>(kernel: P2pKernel<T>, p: &Problem<T>, with_gradient: bool) -> Outputs<T> {
    let (mut phi, mut grad) = (p.potential.clone(), p.gradient.clone());
    let g = with_gradient.then_some(&mut grad[..]);
    kernel.evaluate_dipoles(&p.sources, &p.dipoles, &p.targets, &mut phi, g);
    (phi, grad)
}

fn run_reference<T: Precision>(p: &Problem<T>, with_gradient: bool) -> Outputs<T> {
    let (mut phi, mut grad) = (p.potential.clone(), p.gradient.clone());
    let g = with_gradient.then_some(&mut grad[..]);
    p2p_dipoles(&p.sources, &p.dipoles, &p.targets, &mut phi, g);
    (phi, grad)
}

fn bits<T: Precision>((phi, grad): &Outputs<T>) -> Vec<u64> {
    phi.iter()
        .chain(grad.iter().flatten())
        .map(|v| (*v).to_f64().to_bits())
        .collect()
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// One pair with r² log-uniform in [`r2_min`, 2⁷], as `terms::pair`, and a moment in
/// [−1, 1)³.
fn pair<T: Precision>(rng: &mut Rng, r2_min: f64) -> Problem<T> {
    loop {
        let r = rng.uniform(r2_min.log2(), DOMAIN_MAX.log2()).exp2().sqrt();
        let direction = [0; 3].map(|_| rng.uniform(-1.0, 1.0));
        let n = norm(direction);
        if n < 0.1 {
            continue;
        }
        let side = (16.0 * r).min(1.0);
        let y: [T; 3] = [0; 3].map(|_| T::from_f64(side * rng.uniform(-1.0, 1.0)));
        let x: [T; 3] = [0, 1, 2].map(|k| T::from_f64(y[k].to_f64() + r * direction[k] / n));
        let r2: f64 = (0..3)
            .map(|k| (x[k].to_f64() - y[k].to_f64()).powi(2))
            .sum();
        let d: [T; 3] = [0; 3].map(|_| T::from_f64(rng.uniform(-1.0, 1.0)));
        if !(r2_min..=DOMAIN_MAX).contains(&r2) || d.iter().all(|c| *c == T::zero()) {
            continue;
        }
        return Problem {
            sources: vec![y],
            dipoles: vec![d],
            targets: vec![x],
            potential: vec![T::zero()],
            gradient: vec![[T::zero(); 3]],
        };
    }
}

fn terms<T: Precision>(test: &str, n: usize, r2_min: f64) {
    let mut worst = Worst::new(format!(
        "{test} ({}, {n} pairs, r² in [{r2_min:e}, 2⁷])",
        std::any::type_name::<T>()
    ));
    for kernel in kernels::<T>(test) {
        let mut rng = Rng(0xd1b0_1e00);
        for _ in 0..n {
            let p = pair::<T>(&mut rng, r2_min);
            let (kp, kg) = run(kernel, &p, true);
            let (only, _) = run(kernel, &p, false);
            let (rp, rg) = run_reference(&p, true);
            let w: [f64; 3] =
                core::array::from_fn(|k| p.targets[0][k].to_f64() - p.sources[0][k].to_f64());
            let r2 = w.iter().map(|c| c * c).sum::<f64>();
            let d = norm(p.dipoles[0].map(|c| c.to_f64()));
            let finite =
                kp[0].to_f64().is_finite() && kg[0].iter().all(|c| (*c).to_f64().is_finite());
            assert!(finite, "{} non-finite term at {p:?}", kernel.isa());
            let scale = d / r2;
            let ep = ((kp[0].to_f64() - rp[0].to_f64()) / scale)
                .abs()
                .max(((only[0].to_f64() - rp[0].to_f64()) / scale).abs());
            let scale = d / (r2 * r2.sqrt());
            let eg = (0..3)
                .map(|k| ((kg[0][k].to_f64() - rg[0][k].to_f64()) / scale).abs())
                .fold(0.0, f64::max);
            assert!(
                ep <= POTENTIAL_TERM_U * T::U && eg <= GRADIENT_TERM_U * T::U,
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

#[test]
fn dipole_terms_f32() {
    terms::<f32>("dipole_terms_f32", 4096, F32_DIPOLE_R2_MIN);
}

#[test]
fn dipole_terms_f64() {
    terms::<f64>("dipole_terms_f64", 4096, DOMAIN_MIN);
}

#[test]
#[ignore = "10⁶ pairs; run in release with --ignored"]
fn dipole_terms_f32_large() {
    terms::<f32>("dipole_terms_f32_large", 1_000_000, F32_DIPOLE_R2_MIN);
}

#[test]
#[ignore = "10⁶ pairs; run in release with --ignored"]
fn dipole_terms_f64_large() {
    terms::<f64>("dipole_terms_f64_large", 1_000_000, DOMAIN_MIN);
}

/// `n_sources` dipoles and `n_targets` targets uniform in [−1, 1)³, moments in [−1, 1)³,
/// initial outputs uniform in [−1, 1).
fn uniform<T: Precision>(n_sources: usize, n_targets: usize, seed: u64) -> Problem<T> {
    let mut rng = Rng(seed);
    Problem {
        sources: (0..n_sources).map(|_| rng.point()).collect(),
        dipoles: (0..n_sources).map(|_| rng.point()).collect(),
        targets: (0..n_targets).map(|_| rng.point()).collect(),
        potential: (0..n_targets)
            .map(|_| T::from_f64(rng.uniform(-1.0, 1.0)))
            .collect(),
        gradient: (0..n_targets).map(|_| rng.point()).collect(),
    }
}

fn sums<T: Precision>(test: &str, tol: f64) {
    let mut worst = Worst::new(format!("{test} ({})", std::any::type_name::<T>()));
    for kernel in kernels::<T>(test) {
        for (ns, nt, seed) in [(1, 1, 1), (7, 13, 2), (64, 33, 3), (300, 71, 4)] {
            let p = uniform::<T>(ns, nt, seed);
            let (sources, dipoles, targets) =
                (widen(&p.sources), widen(&p.dipoles), widen(&p.targets));
            let mut phi: Vec<f64> = p.potential.iter().map(|v| (*v).to_f64()).collect();
            let mut grad: Vec<[f64; 3]> = widen(&p.gradient);
            direct_sum_dipoles(&sources, &dipoles, &targets, &mut phi, Some(&mut grad));
            let (kp, kg) = run(kernel, &p, true);
            let (mut ep, mut eg) = (0.0f64, 0.0f64);
            for i in 0..nt {
                let (mut sp, mut sg) = (p.potential[i].to_f64().abs(), 0.0);
                for (y, d) in sources.iter().zip(&dipoles) {
                    let r2: f64 = (0..3).map(|k| (targets[i][k] - y[k]).powi(2)).sum();
                    if r2 > 0.0 {
                        sp += norm(*d) / r2;
                        sg += norm(*d) / (r2 * r2.sqrt());
                    }
                }
                let sg = sg + norm(widen(&p.gradient[i..=i])[0]);
                ep = ep.max((kp[i].to_f64() - phi[i]).abs() / sp);
                for k in 0..3 {
                    eg = eg.max((kg[i][k].to_f64() - grad[i][k]).abs() / sg);
                }
            }
            assert!(
                ep <= tol && eg <= tol,
                "{} ({ns} sources, {nt} targets): errors {ep:e}, {eg:e}",
                kernel.isa()
            );
            worst.record(kernel.isa(), (ep, eg));
        }
    }
    worst.print(1.0, "(relative)");
}

#[test]
fn dipole_sums_f32() {
    sums::<f32>("dipole_sums_f32", 1e-5);
}

#[test]
fn dipole_sums_f64() {
    sums::<f64>("dipole_sums_f64", 1e-13);
}

fn scalar_is_reference<T: Precision>() {
    let kernel = P2pKernel::<T>::new(Isa::Scalar).unwrap();
    for (ns, nt, seed) in [(5, 9, 11), (40, 17, 12)] {
        let p = uniform::<T>(ns, nt, seed);
        for g in [false, true] {
            assert_eq!(bits(&run(kernel, &p, g)), bits(&run_reference(&p, g)));
        }
    }
}

#[test]
fn scalar_dipoles_equal_the_reference() {
    // Error measure: exact equality of the bits.
    scalar_is_reference::<f32>();
    scalar_is_reference::<f64>();
}

fn invariance<T: Precision>(test: &str) {
    for kernel in kernels::<T>(test) {
        let p = uniform::<T>(37, 45, 21);
        for g in [false, true] {
            let whole = run(kernel, &p, g);
            // Sources in two calls: the second starts from the first's outputs.
            for split in [1, 16, 36] {
                let mut first = p.clone();
                first.sources.truncate(split);
                first.dipoles.truncate(split);
                let (phi, grad) = run(kernel, &first, g);
                let second = Problem {
                    sources: p.sources[split..].to_vec(),
                    dipoles: p.dipoles[split..].to_vec(),
                    targets: p.targets.clone(),
                    potential: phi,
                    gradient: grad,
                };
                assert_eq!(
                    bits(&run(kernel, &second, g)),
                    bits(&whole),
                    "{} split {split}",
                    kernel.isa()
                );
            }
            // Each target alone gives its bits in the whole call.
            for i in [0, 7, 44] {
                let one = Problem {
                    sources: p.sources.clone(),
                    dipoles: p.dipoles.clone(),
                    targets: vec![p.targets[i]],
                    potential: vec![p.potential[i]],
                    gradient: vec![p.gradient[i]],
                };
                let (phi, grad) = run(kernel, &one, g);
                assert_eq!(phi[0].to_f64().to_bits(), whole.0[i].to_f64().to_bits());
                if g {
                    assert_eq!(
                        bits(&(phi, grad)),
                        bits(&(vec![whole.0[i]], vec![whole.1[i]]))
                    );
                }
            }
        }
    }
}

#[test]
fn dipole_invariance_f32() {
    // Error measure: exact equality of the bits.
    invariance::<f32>("dipole_invariance_f32");
}

#[test]
fn dipole_invariance_f64() {
    // Error measure: exact equality of the bits.
    invariance::<f64>("dipole_invariance_f64");
}

fn coincident<T: Precision>(test: &str) {
    for kernel in kernels::<T>(test) {
        let mut p = uniform::<T>(23, 23, 31);
        p.targets = p.sources.clone();
        p.potential.iter_mut().for_each(|v| *v = T::zero());
        p.gradient.iter_mut().for_each(|v| *v = [T::zero(); 3]);
        let (phi, grad) = run(kernel, &p, true);
        let (rp, rg) = run_reference(&p, true);
        assert!(
            phi.iter()
                .chain(grad.iter().flatten())
                .all(|v| (*v).to_f64().is_finite())
        );
        for i in 0..23 {
            let scale: f64 = p
                .sources
                .iter()
                .zip(&p.dipoles)
                .filter(|(y, _)| **y != p.targets[i])
                .map(|(y, d)| {
                    let r2: f64 = (0..3)
                        .map(|k| (p.targets[i][k].to_f64() - y[k].to_f64()).powi(2))
                        .sum();
                    norm(d.map(|c| c.to_f64())) / (r2 * r2.sqrt())
                })
                .sum();
            assert!((phi[i].to_f64() - rp[i].to_f64()).abs() <= 1e3 * T::U * scale);
            for k in 0..3 {
                assert!((grad[i][k].to_f64() - rg[i][k].to_f64()).abs() <= 1e3 * T::U * scale);
            }
        }
    }
}

#[test]
fn dipole_coincident_pairs_f32() {
    // Error measure: finite outputs, and within 1000 u_T of the reference, which skips
    // x == y, relative to Σⱼ |dⱼ| / rⱼ³ over the other pairs (a loose sum check: the skip
    // is what is tested).
    coincident::<f32>("dipole_coincident_pairs_f32");
}

#[test]
fn dipole_coincident_pairs_f64() {
    // Error measure: as for f32.
    coincident::<f64>("dipole_coincident_pairs_f64");
}

#[test]
#[should_panic(expected = "`sources` and `dipoles` must have the same length, got 2 and 1")]
fn evaluate_dipoles_rejects_mismatched_moments() {
    let kernel = P2pKernel::<f64>::detect();
    let mut phi = [0.0];
    kernel.evaluate_dipoles(&[[0.0; 3]; 2], &[[1.0; 3]], &[[1.0; 3]], &mut phi, None);
}

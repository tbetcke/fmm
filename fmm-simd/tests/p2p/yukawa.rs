//! `P2pKernel::evaluate_yukawa` (Helmholtz CH2.4b; CONVENTIONS §3.15, k = iκ) on every ISA
//! this machine offers, against `nd_fmm_ref::helmholtz::p2p::{p2p_yukawa,
//! direct_sum_yukawa}`.
//!
//! Error measures:
//! - *Terms*: one source and one target with r² log-uniform over the kernel domain (f32
//!   gradients from 2⁻⁸⁴, as Laplace's), κr log-uniform in [10⁻⁶, [`Contract::KR_MAX`]] (and
//!   κ = 0 for every sixteenth pair), |q| log-uniform in [2⁻²⁰, 1] with a random sign: the
//!   potential against the reference's term in the same precision relative to
//!   |q| e^{−κr}/r, each gradient component relative to |q| (1 + κr) e^{−κr}/r², in units of
//!   u_T; tolerances (a + b κr) u_T with a = [`POTENTIAL_TERM_U`] or [`GRADIENT_TERM_U`] and
//!   b = [`KR_TERM_U`] (docs/design/simd-p2p.md §4.10).
//! - *Sums*: against `direct_sum_yukawa` on the exactly widened inputs, from the same
//!   initial outputs, relative per target to |φ₀| + Σⱼ |qⱼ| e^{−κrⱼ}/rⱼ (potential) and
//!   max |(∇φ₀)ₖ| + Σⱼ |qⱼ| (1 + κrⱼ) e^{−κrⱼ}/rⱼ² (each gradient component): 1e-6 (f32) and
//!   1e-14 (f64), or twice the error of `p2p_yukawa` on the same inputs where that is
//!   larger (C3S.4's rule, as `common::Oracle::tolerance`).
//! - *Bits*: at κ = 0 the kernel equals `evaluate` bit for bit on every vector ISA (the
//!   scalar ISA in the potential, its gradient being the Yukawa reference's); the scalar
//!   path equals `p2p_yukawa` bit for bit on grid-valued inputs (where r² = 0 exactly when
//!   the points coincide); chunk and target-position invariance bit for bit on every ISA.

use nd_fmm_ref::helmholtz::p2p::{direct_sum_yukawa, p2p_yukawa};
use nd_fmm_simd::{Isa, P2pKernel};

use crate::common::{
    DOMAIN_MAX, DOMAIN_MIN, Precision, Problem, Rng, Worst, fmm_shaped, gathered, kernels, run,
    seeded, uniform, widen,
};

/// The potential term's tolerance at κr = 0, in u_T relative to |q| e^{−κr}/r.
const POTENTIAL_TERM_U: f64 = 20.0;
/// Each gradient component's tolerance at κr = 0, in u_T relative to
/// |q| (1 + κr) e^{−κr}/r².
const GRADIENT_TERM_U: f64 = 52.0;
/// The growth of both tolerances with κr, in u_T per unit of κr: the conditioning of
/// e^{−κr} in r, which the reference shares.
const KR_TERM_U: f64 = 12.0;

/// The range of κr of the contract per precision.
trait Contract: Precision {
    /// The largest κr of the contract: 64 in f32 and 512 in f64, where e^{−κr} and the
    /// term's products stay normal for |q| ≥ 2⁻²⁰ on the kernel domain.
    const KR_MAX: f64;
}

impl Contract for f32 {
    const KR_MAX: f64 = 64.0;
}

impl Contract for f64 {
    const KR_MAX: f64 = 512.0;
}

/// The outputs of one evaluation, potential and gradient (zero and untouched without one).
type Outputs<T> = (Vec<T>, Vec<[T; 3]>);

fn run_yukawa<T: Precision>(
    kernel: P2pKernel<T>,
    kappa: T,
    p: &Problem<T>,
    with_gradient: bool,
) -> Outputs<T> {
    let (mut phi, mut grad) = (p.potential.clone(), p.gradient.clone());
    let g = with_gradient.then_some(&mut grad[..]);
    kernel.evaluate_yukawa(kappa, &p.sources, &p.charges, &p.targets, &mut phi, g);
    (phi, grad)
}

fn run_reference<T: Precision>(kappa: T, p: &Problem<T>, with_gradient: bool) -> Outputs<T> {
    let (mut phi, mut grad) = (p.potential.clone(), p.gradient.clone());
    let g = with_gradient.then_some(&mut grad[..]);
    p2p_yukawa(kappa, &p.sources, &p.charges, &p.targets, &mut phi, g);
    (phi, grad)
}

fn bits<T: Precision>((phi, grad): &Outputs<T>) -> Vec<u64> {
    phi.iter()
        .chain(grad.iter().flatten())
        .map(|v| (*v).to_f64().to_bits())
        .collect()
}

/// One pair with r² log-uniform in [`r2_min`, 2⁷] (as `dipoles::pair`), κ with κr
/// log-uniform in [10⁻⁶, `T::KR_MAX`] or 0, and |q| log-uniform in [2⁻²⁰, 1]. Returns the
/// problem, κ and κr in f64 from the rounded κ and the exact distance.
fn pair<T: Contract>(rng: &mut Rng, r2_min: f64, zero_kappa: bool) -> (Problem<T>, T, f64) {
    loop {
        let r = rng.uniform(r2_min.log2(), DOMAIN_MAX.log2()).exp2().sqrt();
        let direction = [0; 3].map(|_| rng.uniform(-1.0, 1.0));
        let n = direction.iter().map(|c| c * c).sum::<f64>().sqrt();
        if n < 0.1 {
            continue;
        }
        let side = (16.0 * r).min(1.0);
        let y: [T; 3] = [0; 3].map(|_| T::from_f64(side * rng.uniform(-1.0, 1.0)));
        let x: [T; 3] = [0, 1, 2].map(|k| T::from_f64(y[k].to_f64() + r * direction[k] / n));
        let r2: f64 = (0..3)
            .map(|k| (x[k].to_f64() - y[k].to_f64()).powi(2))
            .sum();
        if !(r2_min..=DOMAIN_MAX).contains(&r2) {
            continue;
        }
        let r = r2.sqrt();
        let kappa = if zero_kappa {
            T::zero()
        } else {
            let kr = rng.uniform(1e-6f64.ln(), T::KR_MAX.ln()).exp();
            T::from_f64(kr / r)
        };
        let kr = kappa.to_f64() * r;
        if kr > T::KR_MAX || !kappa.to_f64().is_finite() {
            continue;
        }
        let magnitude = rng.uniform(-20.0, 0.0).exp2();
        let q = if rng.below(2) == 0 {
            magnitude
        } else {
            -magnitude
        };
        let problem = Problem {
            sources: vec![y],
            charges: vec![T::from_f64(q)],
            targets: vec![x],
            potential: vec![T::zero()],
            gradient: vec![[T::zero(); 3]],
        };
        return (problem, kappa, kr);
    }
}

/// The largest error per κr decade, in u_T, for the report.
#[derive(Default)]
struct Decades {
    /// Per bin [0, 10⁻²), [10⁻², 1), [1, 10), [10, 100), [100, ∞): the largest potential
    /// and gradient errors in u_T.
    bins: [(f64, f64); 5],
    /// The largest potential and gradient errors per unit of κr, over κr ≥ 1, in u_T.
    per_kr: (f64, f64),
}

impl Decades {
    fn record(&mut self, kr: f64, ep: f64, eg: f64) {
        let bin = match kr {
            kr if kr < 1e-2 => 0,
            kr if kr < 1.0 => 1,
            kr if kr < 10.0 => 2,
            kr if kr < 100.0 => 3,
            _ => 4,
        };
        let b = &mut self.bins[bin];
        *b = (b.0.max(ep), b.1.max(eg));
        if kr >= 1.0 {
            self.per_kr = (self.per_kr.0.max(ep / kr), self.per_kr.1.max(eg / kr));
        }
    }

    fn print(&self, what: &str, isa: Isa) {
        let names = [
            "[0, 0.01)",
            "[0.01, 1)",
            "[1, 10)",
            "[10, 100)",
            "[100, max]",
        ];
        for (name, (ep, eg)) in names.iter().zip(self.bins) {
            println!(
                "{what}: {isa} κr in {name}: max error potential {ep:.3} u, gradient {eg:.3} u"
            );
        }
        let (ep, eg) = self.per_kr;
        println!(
            "{what}: {isa} κr ≥ 1: max error per unit κr potential {ep:.3} u, gradient {eg:.3} u"
        );
    }
}

fn terms<T: Contract>(test: &str, n: usize) {
    let mut worst = Worst::new(format!(
        "{test} ({}, {n} pairs, κr up to {})",
        std::any::type_name::<T>(),
        T::KR_MAX
    ));
    for kernel in kernels::<T>(test) {
        let mut rng = Rng(0x7e4a_3a00);
        let mut decades = Decades::default();
        for i in 0..n {
            // The potential over the whole domain, the gradient from its f32 range.
            let r2_min = if i % 2 == 0 {
                DOMAIN_MIN
            } else {
                T::GRADIENT_R2_MIN
            };
            let with_gradient = i % 2 == 1;
            let (p, kappa, kr) = pair::<T>(&mut rng, r2_min, i % 16 == 0);
            let (kp, kg) = run_yukawa(kernel, kappa, &p, with_gradient);
            let (only, _) = run_yukawa(kernel, kappa, &p, false);
            let (rp, rg) = run_reference(kappa, &p, with_gradient);
            let r2: f64 = (0..3)
                .map(|k| (p.targets[0][k].to_f64() - p.sources[0][k].to_f64()).powi(2))
                .sum();
            let r = r2.sqrt();
            let q = p.charges[0].to_f64().abs();
            let decay = (-kr).exp();
            let finite =
                kp[0].to_f64().is_finite() && kg[0].iter().all(|c| (*c).to_f64().is_finite());
            assert!(
                finite,
                "{} non-finite term at {p:?}, κ = {kappa:?}",
                kernel.isa()
            );
            let scale = q * decay / r;
            let ep = ((kp[0].to_f64() - rp[0].to_f64()) / scale)
                .abs()
                .max(((only[0].to_f64() - rp[0].to_f64()) / scale).abs());
            let eg = if with_gradient {
                let scale = q * (1.0 + kr) * decay / r2;
                (0..3)
                    .map(|k| ((kg[0][k].to_f64() - rg[0][k].to_f64()) / scale).abs())
                    .fold(0.0, f64::max)
            } else {
                0.0
            };
            let tp = (POTENTIAL_TERM_U + KR_TERM_U * kr) * T::U;
            let tg = (GRADIENT_TERM_U + KR_TERM_U * kr) * T::U;
            assert!(
                ep <= tp && eg <= tg,
                "{} pair {p:?}, κ = {kappa:?} (κr = {kr:e}): errors {:.3} u (potential), \
                 {:.3} u (gradient), tolerances {:.3} u and {:.3} u",
                kernel.isa(),
                ep / T::U,
                eg / T::U,
                tp / T::U,
                tg / T::U
            );
            worst.record(kernel.isa(), (ep, eg));
            decades.record(kr, ep / T::U, eg / T::U);
        }
        decades.print(test, kernel.isa());
    }
    worst.print(T::U, "u");
}

#[test]
fn yukawa_terms_f32() {
    terms::<f32>("yukawa_terms_f32", 4096);
}

#[test]
fn yukawa_terms_f64() {
    terms::<f64>("yukawa_terms_f64", 4096);
}

#[test]
#[ignore = "10⁶ pairs; run in release with --ignored"]
fn yukawa_terms_f32_large() {
    terms::<f32>("yukawa_terms_f32_large", 1_000_000);
}

#[test]
#[ignore = "10⁶ pairs; run in release with --ignored"]
fn yukawa_terms_f64_large() {
    terms::<f64>("yukawa_terms_f64_large", 1_000_000);
}

/// The oracle of a Yukawa problem: `direct_sum_yukawa` from the initial outputs, the
/// scales per target, and the reference's own errors.
struct Oracle {
    potential: Vec<f64>,
    gradient: Vec<[f64; 3]>,
    potential_scale: Vec<f64>,
    gradient_scale: Vec<f64>,
    reference: (f64, f64),
}

impl Oracle {
    fn new<T: Precision>(kappa: T, p: &Problem<T>) -> Self {
        let (sources, targets) = (widen(&p.sources), widen(&p.targets));
        let charges: Vec<f64> = p.charges.iter().map(|q| (*q).to_f64()).collect();
        let k = kappa.to_f64();
        let mut potential: Vec<f64> = p.potential.iter().map(|v| (*v).to_f64()).collect();
        let mut gradient = widen(&p.gradient);
        direct_sum_yukawa(
            k,
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        let mut potential_scale = Vec::new();
        let mut gradient_scale = Vec::new();
        for (i, x) in targets.iter().enumerate() {
            let mut sp = p.potential[i].to_f64().abs();
            let mut sg = p.gradient[i]
                .iter()
                .map(|g| (*g).to_f64().abs())
                .fold(0.0, f64::max);
            for (y, q) in sources.iter().zip(&charges) {
                if x == y {
                    continue;
                }
                let r2: f64 = (0..3).map(|c| (x[c] - y[c]).powi(2)).sum();
                let r = r2.sqrt();
                let decay = (-k * r).exp();
                sp += q.abs() * decay / r;
                sg += q.abs() * (1.0 + k * r) * decay / r2;
            }
            potential_scale.push(sp);
            gradient_scale.push(sg);
        }
        let mut oracle = Self {
            potential,
            gradient,
            potential_scale,
            gradient_scale,
            reference: (0.0, 0.0),
        };
        oracle.reference = oracle.errors(&run_reference(kappa, p, true));
        oracle
    }

    fn errors<T: Precision>(&self, (phi, grad): &Outputs<T>) -> (f64, f64) {
        let relative = |got: T, want: f64, scale: f64| {
            let got = got.to_f64();
            if !got.is_finite() {
                return f64::INFINITY;
            }
            let e = (got - want).abs();
            if e == 0.0 { 0.0 } else { e / scale }
        };
        let mut ep = 0.0f64;
        let mut eg = 0.0f64;
        for i in 0..phi.len() {
            ep = ep.max(relative(phi[i], self.potential[i], self.potential_scale[i]));
            for (&gk, &want) in grad[i].iter().zip(&self.gradient[i]) {
                eg = eg.max(relative(gk, want, self.gradient_scale[i]));
            }
        }
        (ep, eg)
    }

    fn tolerance<T: Precision>(&self) -> (f64, f64) {
        (
            T::SUM_TOL.max(2.0 * self.reference.0),
            T::SUM_TOL.max(2.0 * self.reference.1),
        )
    }
}

/// `p` with initial outputs uniform in [−1, 1), seeded.
fn with_initial<T: Precision>(mut p: Problem<T>, seed: u64) -> Problem<T> {
    let mut rng = Rng(seed);
    p.potential = p
        .potential
        .iter()
        .map(|_| T::from_f64(rng.uniform(-1.0, 1.0)))
        .collect();
    p.gradient = p.gradient.iter().map(|_| rng.point()).collect();
    p
}

/// The sum problems: FMM-shaped gathered sets (same level and mixed) and uniform sets,
/// the largest with `max_sources` sources.
fn sum_problems<T: Precision>(max_sources: usize) -> Vec<(String, Problem<T>)> {
    let mut problems = Vec::new();
    for (n, mixed, seed) in [(16, false, 41), (24, true, 42), (64, false, 43)] {
        let (leaves, targets) = fmm_shaped::<T>(n, mixed, seed);
        let (sources, charges) = gathered(&leaves);
        let nt = targets.len();
        let p = Problem {
            sources,
            charges,
            targets,
            potential: vec![T::zero(); nt],
            gradient: vec![[T::zero(); 3]; nt],
        };
        problems.push((format!("W1 n_t={n} mixed={mixed}"), p));
    }
    for (ns, nt, seed) in [
        (1, 1, 51),
        (7, 13, 52),
        (300, 71, 53),
        (max_sources, 37, 54),
    ] {
        let p = with_initial(uniform::<T>(ns, nt, seed), seed);
        problems.push((format!("uniform {ns} × {nt}"), p));
    }
    problems
}

fn sums<T: Precision>(test: &str, max_sources: usize) {
    let mut worst = Worst::new(format!("{test} ({})", std::any::type_name::<T>()));
    let problems = sum_problems::<T>(max_sources);
    for kernel in kernels::<T>(test) {
        for (name, p) in &problems {
            for kappa in [0.1, 1.0, 10.0].map(T::from_f64) {
                let oracle = Oracle::new(kappa, p);
                let tol = oracle.tolerance::<T>();
                let errors = oracle.errors(&run_yukawa(kernel, kappa, p, true));
                assert!(
                    errors.0 <= tol.0 && errors.1 <= tol.1,
                    "{} {name}, κ = {kappa:?}: errors {:e}, {:e} exceed {:e}, {:e}",
                    kernel.isa(),
                    errors.0,
                    errors.1,
                    tol.0,
                    tol.1
                );
                let only = run_yukawa(kernel, kappa, p, false);
                let ep = oracle.errors(&(only.0, p.gradient.clone())).0;
                assert!(
                    ep <= tol.0,
                    "{} {name}, κ = {kappa:?}: {ep:e}",
                    kernel.isa()
                );
                worst.record(kernel.isa(), errors);
            }
        }
    }
    worst.print(1.0, "(relative)");
}

#[test]
fn yukawa_sums_f32() {
    sums::<f32>("yukawa_sums_f32", 512);
}

#[test]
fn yukawa_sums_f64() {
    sums::<f64>("yukawa_sums_f64", 512);
}

#[test]
#[ignore = "4,096 sources; run in release with --ignored"]
fn yukawa_sums_f32_large() {
    sums::<f32>("yukawa_sums_f32_large", 4096);
}

#[test]
#[ignore = "4,096 sources; run in release with --ignored"]
fn yukawa_sums_f64_large() {
    sums::<f64>("yukawa_sums_f64_large", 4096);
}

fn kappa_zero_is_laplace<T: Precision>(test: &str) {
    for kernel in kernels::<T>(test) {
        for (ns, nt, seed) in [(0, 5, 61), (9, 0, 62), (13, 29, 63), (64, 41, 64)] {
            let mut p = seeded(ns, nt, seed).cast::<T>();
            // Some targets on the sources, so that coincident pairs occur.
            for i in (0..nt.min(ns)).step_by(3) {
                p.targets[i] = p.sources[i];
            }
            for g in [false, true] {
                let (lp, lg) = run(kernel, &p, g);
                let mut laplace = (lp, lg.unwrap_or_else(|| p.gradient.clone()));
                let mut yukawa = run_yukawa(kernel, T::zero(), &p, g);
                if kernel.isa() == Isa::Scalar {
                    // The scalar paths are the references' loops: Laplace's gradient term
                    // −(q dₖ / r³) and Yukawa's (q (−1 / r³)) dₖ round differently; the
                    // potentials q / r and (q · 1) / r agree.
                    laplace.1.clear();
                    yukawa.1.clear();
                }
                assert_eq!(
                    bits(&yukawa),
                    bits(&laplace),
                    "{} ({ns} sources, {nt} targets, gradient {g})",
                    kernel.isa()
                );
            }
        }
    }
}

#[test]
fn yukawa_at_kappa_zero_equals_laplace_f32() {
    // Error measure: exact equality of the bits.
    kappa_zero_is_laplace::<f32>("yukawa_at_kappa_zero_equals_laplace_f32");
}

#[test]
fn yukawa_at_kappa_zero_equals_laplace_f64() {
    // Error measure: exact equality of the bits.
    kappa_zero_is_laplace::<f64>("yukawa_at_kappa_zero_equals_laplace_f64");
}

fn scalar_is_reference<T: Precision>() {
    let kernel = P2pKernel::<T>::new(Isa::Scalar).unwrap();
    for (ns, nt, seed) in [(5, 9, 71), (40, 17, 72), (0, 3, 73)] {
        let mut p = seeded(ns, nt, seed).cast::<T>();
        for i in (0..nt.min(ns)).step_by(2) {
            p.targets[i] = p.sources[i];
        }
        for kappa in [0.0, 0.75, 3.5].map(T::from_f64) {
            for g in [false, true] {
                assert_eq!(
                    bits(&run_yukawa(kernel, kappa, &p, g)),
                    bits(&run_reference(kappa, &p, g)),
                    "({ns} sources, {nt} targets, κ = {kappa:?}, gradient {g})"
                );
            }
        }
    }
}

#[test]
fn scalar_yukawa_equals_the_reference() {
    // Error measure: exact equality of the bits (grid-valued inputs: r² = 0 exactly
    // where the points coincide).
    scalar_is_reference::<f32>();
    scalar_is_reference::<f64>();
}

fn invariance<T: Precision>(test: &str) {
    let kappa = T::from_f64(1.25);
    for kernel in kernels::<T>(test) {
        let p = with_initial(uniform::<T>(37, 45, 81), 81);
        for g in [false, true] {
            let whole = run_yukawa(kernel, kappa, &p, g);
            // Sources in two calls: the second starts from the first's outputs.
            for split in [1, 16, 36] {
                let mut first = p.clone();
                first.sources.truncate(split);
                first.charges.truncate(split);
                let (phi, grad) = run_yukawa(kernel, kappa, &first, g);
                let second = Problem {
                    sources: p.sources[split..].to_vec(),
                    charges: p.charges[split..].to_vec(),
                    targets: p.targets.clone(),
                    potential: phi,
                    gradient: grad,
                };
                assert_eq!(
                    bits(&run_yukawa(kernel, kappa, &second, g)),
                    bits(&whole),
                    "{} split {split}",
                    kernel.isa()
                );
            }
            // Each target alone, and every prefix length, gives its bits in the whole call.
            for len in [1, 7, 8, 17, 33, 45] {
                let part = Problem {
                    sources: p.sources.clone(),
                    charges: p.charges.clone(),
                    targets: p.targets[45 - len..].to_vec(),
                    potential: p.potential[45 - len..].to_vec(),
                    gradient: p.gradient[45 - len..].to_vec(),
                };
                let out = run_yukawa(kernel, kappa, &part, g);
                let tail = (whole.0[45 - len..].to_vec(), whole.1[45 - len..].to_vec());
                assert_eq!(bits(&out), bits(&tail), "{} last {len}", kernel.isa());
            }
        }
    }
}

#[test]
fn yukawa_invariance_f32() {
    // Error measure: exact equality of the bits.
    invariance::<f32>("yukawa_invariance_f32");
}

#[test]
fn yukawa_invariance_f64() {
    // Error measure: exact equality of the bits.
    invariance::<f64>("yukawa_invariance_f64");
}

fn coincident_and_empty<T: Precision>(test: &str) {
    let kappa = T::from_f64(0.5);
    for kernel in kernels::<T>(test) {
        // Targets on the sources: the self pairs add nothing, the outputs stay finite and
        // within the sum tolerance of the direct sum, which skips x == y.
        let mut p = with_initial(uniform::<T>(23, 23, 91), 91);
        p.targets = p.sources.clone();
        let oracle = Oracle::new(kappa, &p);
        let out = run_yukawa(kernel, kappa, &p, true);
        assert!(
            out.0
                .iter()
                .chain(out.1.iter().flatten())
                .all(|v| (*v).to_f64().is_finite())
        );
        let tol = oracle.tolerance::<T>();
        let errors = oracle.errors(&out);
        assert!(errors.0 <= tol.0 && errors.1 <= tol.1, "{}", kernel.isa());
        // No sources: the outputs keep their bits.
        let mut empty = p.clone();
        empty.sources.clear();
        empty.charges.clear();
        for g in [false, true] {
            let out = run_yukawa(kernel, kappa, &empty, g);
            assert_eq!(bits(&out), bits(&(p.potential.clone(), p.gradient.clone())));
        }
        // No targets.
        let mut none = p.clone();
        none.targets.clear();
        none.potential.clear();
        none.gradient.clear();
        let out = run_yukawa(kernel, kappa, &none, true);
        assert!(out.0.is_empty() && out.1.is_empty());
    }
}

#[test]
fn yukawa_coincident_and_empty_inputs_f32() {
    // Error measure: finite outputs within the sum tolerance; the bits of untouched
    // outputs.
    coincident_and_empty::<f32>("yukawa_coincident_and_empty_inputs_f32");
}

#[test]
fn yukawa_coincident_and_empty_inputs_f64() {
    // Error measure: as for f32.
    coincident_and_empty::<f64>("yukawa_coincident_and_empty_inputs_f64");
}

#[test]
#[should_panic(expected = "`sources` and `charges` must have the same length, got 2 and 1")]
fn evaluate_yukawa_rejects_mismatched_charges() {
    let kernel = P2pKernel::<f64>::detect();
    let mut phi = [0.0];
    kernel.evaluate_yukawa(1.0, &[[0.0; 3]; 2], &[1.0], &[[1.0; 3]], &mut phi, None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient` must have the same length, got 1 and 2")]
fn evaluate_yukawa_rejects_mismatched_gradient() {
    let kernel = P2pKernel::<f32>::detect();
    let (mut phi, mut grad) = ([0.0], [[0.0; 3]; 2]);
    kernel.evaluate_yukawa(
        1.0,
        &[[0.0; 3]],
        &[1.0],
        &[[1.0; 3]],
        &mut phi,
        Some(&mut grad),
    );
}

fn rejects_kappa(kappa: f64) {
    let kernel = P2pKernel::<f64>::detect();
    let mut phi = [0.0];
    kernel.evaluate_yukawa(kappa, &[[0.0; 3]], &[1.0], &[[1.0; 3]], &mut phi, None);
}

#[test]
#[should_panic(expected = "Yukawa P2P needs a finite wavenumber κ ≥ 0")]
fn evaluate_yukawa_rejects_negative_kappa() {
    rejects_kappa(-0.5);
}

#[test]
#[should_panic(expected = "Yukawa P2P needs a finite wavenumber κ ≥ 0")]
fn evaluate_yukawa_rejects_nan_kappa() {
    rejects_kappa(f64::NAN);
}

#[test]
#[should_panic(expected = "Yukawa P2P needs a finite wavenumber κ ≥ 0")]
fn evaluate_yukawa_rejects_infinite_kappa() {
    rejects_kappa(f64::INFINITY);
}

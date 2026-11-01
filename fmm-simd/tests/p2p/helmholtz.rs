//! `P2pKernel::evaluate_helmholtz` (Helmholtz CH3.4b; CONVENTIONS §3.15) on every ISA this
//! machine offers, against `nd_fmm_ref::helmholtz::p2p::{p2p, direct_sum}`, with real and
//! complex charges and real and complex k.
//!
//! Error measures:
//! - *Terms*: one source and one target with r² log-uniform over the kernel domain (f32
//!   gradients from 2⁻⁸⁴ with (1 + |k| r)/r³ ≤ 2¹²⁶, Laplace's range at k = 0), Re(k) r log-uniform in [10⁻⁶, [`RE_KR_MAX`]] (and
//!   Re k = 0 for every sixteenth pair), Im k = 0 for a third of the pairs and Im(k) r
//!   log-uniform in [10⁻⁶, [`Contract::IM_KR_MAX`]] otherwise, |q| log-uniform in [2⁻²⁰, 1]
//!   with a random sign (real charges) or argument (complex): each part of the potential
//!   against the reference's term in the same precision relative to |q| e^{−Im(k) r}/r, each
//!   part of each gradient component relative to |q| (1 + |k| r) e^{−Im(k) r}/r², in units of
//!   u_T; tolerances (a + b |k| r) u_T with a = [`POTENTIAL_TERM_U`] or [`GRADIENT_TERM_U`]
//!   and b = [`KR_TERM_U`] (docs/design/simd-p2p.md §4.12).
//! - *Sums*: against `direct_sum` on the exactly widened inputs, from the same initial
//!   outputs, relative per target to max(|Re φ₀|, |Im φ₀|) + Σⱼ |qⱼ| e^{−Im(k) rⱼ}/rⱼ
//!   (potential) and the largest initial gradient part + Σⱼ |qⱼ| (1 + |k| rⱼ) e^{−Im(k) rⱼ}/rⱼ²
//!   (each part of each gradient component): 1e-6 (f32) and 1e-14 (f64), or twice the error
//!   of the reference `p2p` on the same inputs where that is larger (C3S.4's rule, as
//!   `common::Oracle::tolerance`).
//! - *Bits*: with real charges the real plane at k = 0 equals `evaluate` and at k = iκ
//!   `evaluate_yukawa` bit for bit on every vector ISA (the scalar ISA in what its reference
//!   loops share); complex charges with Im q = 0 give the real charges' values (compared
//!   as numbers, so a zero's sign may differ); the scalar path equals `p2p` bit for bit on
//!   grid-valued inputs (where r² = 0 exactly when the points coincide); chunk and
//!   target-position invariance bit for bit on every ISA.

use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::p2p::{direct_sum, p2p};
use nd_fmm_simd::{HelmholtzCharges, Isa, P2pKernel};

use crate::common::{
    DOMAIN_MAX, DOMAIN_MIN, Precision, Problem, Rng, Worst, fmm_shaped, gathered, kernels, run,
    seeded, uniform, widen,
};

/// Each part of the potential term's tolerance at k = 0, in u_T relative to
/// |q| e^{−Im(k) r}/r.
const POTENTIAL_TERM_U: f64 = 32.0;
/// Each part of each gradient component's tolerance at k = 0, in u_T relative to
/// |q| (1 + |k| r) e^{−Im(k) r}/r².
const GRADIENT_TERM_U: f64 = 66.0;
/// The growth of both tolerances with |k| r, in u_T per unit of |k| r: the conditioning of
/// e^{ikr} in r, which the reference shares.
const KR_TERM_U: f64 = 12.0;
/// The largest Re(k) r of the contract: the sine and cosine's domain, 2¹⁰.
const RE_KR_MAX: f64 = 1024.0;

/// The range of Im(k) r and of the gradient of the contract per precision.
trait Contract: Precision {
    /// The largest Im(k) r of the contract: 64 in f32 and 512 in f64, as the Yukawa
    /// kernel's κr.
    const IM_KR_MAX: f64;
    /// The largest (1 + |k| r) / r³ of the gradient contract: 2¹²⁶ in f32, where the
    /// gradient factor stays in range (Laplace's ρ³ ≤ 2¹²⁶ at k = 0, so r² ≥ 2⁻⁸⁴); no
    /// limit on the kernel domain in f64.
    const GRADIENT_FACTOR_MAX: f64;
}

impl Contract for f32 {
    const IM_KR_MAX: f64 = 64.0;
    const GRADIENT_FACTOR_MAX: f64 = 8.507059173023462e37;
}

impl Contract for f64 {
    const IM_KR_MAX: f64 = 512.0;
    const GRADIENT_FACTOR_MAX: f64 = f64::INFINITY;
}

/// A Helmholtz problem: real (one value per source) or complex (Re q, Im q per source)
/// charges, and two output planes per target, the real one and then the imaginary one.
#[derive(Clone, Debug)]
struct HProblem<T> {
    sources: Vec<[T; 3]>,
    charges: Vec<T>,
    complex: bool,
    targets: Vec<[T; 3]>,
    potential: Vec<T>,
    gradient: Vec<[T; 3]>,
}

impl<T: Precision> HProblem<T> {
    /// `p` with real charges, or complex ones whose real parts are `p`'s and imaginary parts
    /// uniform in [−1, 1); the real output plane `p`'s, the imaginary one uniform in [−1, 1)
    /// if `p`'s outputs are not all zero, else zero.
    fn from(p: &Problem<T>, complex: bool, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let charges = if complex {
            p.charges
                .iter()
                .flat_map(|&q| [q, T::from_f64(rng.uniform(-1.0, 1.0))])
                .collect()
        } else {
            p.charges.clone()
        };
        let zero = p.potential.iter().all(|v| *v == T::zero());
        let im = |rng: &mut Rng| {
            if zero {
                T::zero()
            } else {
                T::from_f64(rng.uniform(-1.0, 1.0))
            }
        };
        let potential = p
            .potential
            .iter()
            .copied()
            .chain(p.potential.iter().map(|_| im(&mut rng)))
            .collect();
        let gradient = p
            .gradient
            .iter()
            .copied()
            .chain(p.gradient.iter().map(|_| [0; 3].map(|_| im(&mut rng))))
            .collect();
        Self {
            sources: p.sources.clone(),
            charges,
            complex,
            targets: p.targets.clone(),
            potential,
            gradient,
        }
    }

    fn kernel_charges(&self) -> HelmholtzCharges<'_, T> {
        if self.complex {
            HelmholtzCharges::Complex(&self.charges)
        } else {
            HelmholtzCharges::Real(&self.charges)
        }
    }

    /// The charges as complex numbers.
    fn complex_charges(&self) -> Vec<Complex<T>> {
        if self.complex {
            self.charges
                .as_chunks::<2>()
                .0
                .iter()
                .map(|q| Complex::new(q[0], q[1]))
                .collect()
        } else {
            self.charges
                .iter()
                .map(|&q| Complex::new(q, T::zero()))
                .collect()
        }
    }
}

/// The outputs of one evaluation in two planes, potential and gradient (the initial
/// gradient, untouched, without one).
type Outputs<T> = (Vec<T>, Vec<[T; 3]>);

fn run_helmholtz<T: Precision>(
    kernel: P2pKernel<T>,
    k: Complex<T>,
    p: &HProblem<T>,
    with_gradient: bool,
) -> Outputs<T> {
    let (mut phi, mut grad) = (p.potential.clone(), p.gradient.clone());
    let g = with_gradient.then_some(&mut grad[..]);
    kernel.evaluate_helmholtz(k, &p.sources, p.kernel_charges(), &p.targets, &mut phi, g);
    (phi, grad)
}

/// The reference `p2p` on `p` in the same precision, from its initial outputs, split into
/// planes.
fn run_reference<T: Precision>(k: Complex<T>, p: &HProblem<T>, with_gradient: bool) -> Outputs<T> {
    let n = p.targets.len();
    let mut phi: Vec<Complex<T>> = (0..n)
        .map(|i| Complex::new(p.potential[i], p.potential[n + i]))
        .collect();
    let mut grad: Vec<[Complex<T>; 3]> = (0..n)
        .map(|i| core::array::from_fn(|c| Complex::new(p.gradient[i][c], p.gradient[n + i][c])))
        .collect();
    let g = with_gradient.then_some(&mut grad[..]);
    p2p(k, &p.sources, &p.complex_charges(), &p.targets, &mut phi, g);
    let potential = phi
        .iter()
        .map(|v| v.re)
        .chain(phi.iter().map(|v| v.im))
        .collect();
    let gradient = grad
        .iter()
        .map(|g| g.map(|c| c.re))
        .chain(grad.iter().map(|g| g.map(|c| c.im)))
        .collect();
    (potential, gradient)
}

fn bits<T: Precision>((phi, grad): &Outputs<T>) -> Vec<u64> {
    phi.iter()
        .chain(grad.iter().flatten())
        .map(|v| (*v).to_f64().to_bits())
        .collect()
}

/// The real plane of `out` for `n` targets.
fn real_plane<T: Precision>((phi, grad): &Outputs<T>, n: usize) -> Outputs<T> {
    (phi[..n].to_vec(), grad[..n].to_vec())
}

/// The values of `out`, as numbers (so that −0 equals +0).
fn values<T: Precision>((phi, grad): &Outputs<T>) -> Vec<f64> {
    phi.iter()
        .chain(grad.iter().flatten())
        .map(|v| (*v).to_f64())
        .collect()
}

/// One pair with r² log-uniform in [`r2_min`, 2⁷] (as `yukawa::pair`), k as the module
/// documentation says for pair `i`, and the charge; with `gradient`, within the gradient's
/// range ([`Contract::GRADIENT_FACTOR_MAX`]). Returns the problem, k, |q| and, in f64
/// from the rounded k and the exact distance, r², Re(k) r and Im(k) r.
fn pair<T: Contract>(
    rng: &mut Rng,
    r2_min: f64,
    i: usize,
    complex: bool,
    gradient: bool,
) -> (HProblem<T>, Complex<T>, f64, [f64; 3]) {
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
        let re_k = if i.is_multiple_of(16) {
            T::zero()
        } else {
            T::from_f64(rng.uniform(1e-6f64.ln(), RE_KR_MAX.ln()).exp() / r)
        };
        let im_k = if i.is_multiple_of(3) {
            T::zero()
        } else {
            T::from_f64(rng.uniform(1e-6f64.ln(), T::IM_KR_MAX.ln()).exp() / r)
        };
        let (re_kr, im_kr) = (re_k.to_f64() * r, im_k.to_f64() * r);
        if re_kr > RE_KR_MAX
            || im_kr > T::IM_KR_MAX
            || (gradient && (1.0 + re_kr.hypot(im_kr)) / (r2 * r) > T::GRADIENT_FACTOR_MAX)
            || !re_k.to_f64().is_finite()
            || !im_k.to_f64().is_finite()
        {
            continue;
        }
        let magnitude = rng.uniform(-20.0, 0.0).exp2();
        let charges = if complex {
            let angle = rng.uniform(-std::f64::consts::PI, std::f64::consts::PI);
            vec![
                T::from_f64(magnitude * angle.cos()),
                T::from_f64(magnitude * angle.sin()),
            ]
        } else if rng.below(2) == 0 {
            vec![T::from_f64(magnitude)]
        } else {
            vec![T::from_f64(-magnitude)]
        };
        let q = charges
            .iter()
            .map(|c| (*c).to_f64().powi(2))
            .sum::<f64>()
            .sqrt();
        let problem = HProblem {
            sources: vec![y],
            charges,
            complex,
            targets: vec![x],
            potential: vec![T::zero(); 2],
            gradient: vec![[T::zero(); 3]; 2],
        };
        return (problem, Complex::new(re_k, im_k), q, [r2, re_kr, im_kr]);
    }
}

/// The largest error per |k| r decade, in u_T, for the report.
#[derive(Default)]
struct Decades {
    /// Per bin [0, 10⁻²), [10⁻², 1), [1, 10), [10, 100), [100, ∞): the largest potential
    /// and gradient errors in u_T.
    bins: [(f64, f64); 5],
    /// The largest potential and gradient errors per unit of |k| r, over |k| r ≥ 1, in u_T.
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
                "{what}: {isa} |k|r in {name}: max error potential {ep:.3} u, gradient {eg:.3} u"
            );
        }
        let (ep, eg) = self.per_kr;
        println!(
            "{what}: {isa} |k|r ≥ 1: max error per unit |k|r potential {ep:.3} u, gradient {eg:.3} u"
        );
    }
}

fn terms<T: Contract>(test: &str, n: usize) {
    let mut worst = Worst::new(format!(
        "{test} ({}, {n} pairs, Re(k) r up to {RE_KR_MAX}, Im(k) r up to {})",
        std::any::type_name::<T>(),
        T::IM_KR_MAX
    ));
    for kernel in kernels::<T>(test) {
        let mut rng = Rng(0x7e4a_3a01);
        let mut decades = [Decades::default(), Decades::default()];
        for i in 0..n {
            // The potential over the whole domain, the gradient from its f32 range.
            let with_gradient = i % 2 == 1;
            let r2_min = if with_gradient {
                T::GRADIENT_R2_MIN
            } else {
                DOMAIN_MIN
            };
            let complex = (i / 2) % 2 == 1;
            let (p, k, q, [r2, re_kr, im_kr]) =
                pair::<T>(&mut rng, r2_min, i, complex, with_gradient);
            let (kp, kg) = run_helmholtz(kernel, k, &p, with_gradient);
            let (only, _) = run_helmholtz(kernel, k, &p, false);
            let (rp, rg) = run_reference(k, &p, with_gradient);
            let finite = kp
                .iter()
                .chain(kg.iter().flatten())
                .all(|c| (*c).to_f64().is_finite());
            assert!(
                finite,
                "{} non-finite term at {p:?}, k = {k:?}",
                kernel.isa()
            );
            let r = r2.sqrt();
            let kr = re_kr.hypot(im_kr);
            let decay = (-im_kr).exp();
            let scale = q * decay / r;
            let ep = (0..2)
                .map(|o| {
                    let want = rp[o].to_f64();
                    ((kp[o].to_f64() - want) / scale)
                        .abs()
                        .max(((only[o].to_f64() - want) / scale).abs())
                })
                .fold(0.0, f64::max);
            let eg = if with_gradient {
                let scale = q * (1.0 + kr) * decay / r2;
                (0..2)
                    .flat_map(|o| (0..3).map(move |c| (o, c)))
                    .map(|(o, c)| ((kg[o][c].to_f64() - rg[o][c].to_f64()) / scale).abs())
                    .fold(0.0, f64::max)
            } else {
                0.0
            };
            let tp = (POTENTIAL_TERM_U + KR_TERM_U * kr) * T::U;
            let tg = (GRADIENT_TERM_U + KR_TERM_U * kr) * T::U;
            assert!(
                ep <= tp && eg <= tg,
                "{} pair {p:?}, k = {k:?} (Re(k) r = {re_kr:e}, Im(k) r = {im_kr:e}): errors \
                 {:.3} u (potential), {:.3} u (gradient), tolerances {:.3} u and {:.3} u",
                kernel.isa(),
                ep / T::U,
                eg / T::U,
                tp / T::U,
                tg / T::U
            );
            worst.record(kernel.isa(), (ep, eg));
            decades[usize::from(complex)].record(kr, ep / T::U, eg / T::U);
        }
        decades[0].print(&format!("{test}, real charges"), kernel.isa());
        decades[1].print(&format!("{test}, complex charges"), kernel.isa());
    }
    worst.print(T::U, "u");
}

#[test]
fn helmholtz_terms_f32() {
    terms::<f32>("helmholtz_terms_f32", 4096);
}

#[test]
fn helmholtz_terms_f64() {
    terms::<f64>("helmholtz_terms_f64", 4096);
}

#[test]
#[ignore = "10⁶ pairs; run in release with --ignored"]
fn helmholtz_terms_f32_large() {
    terms::<f32>("helmholtz_terms_f32_large", 1_000_000);
}

#[test]
#[ignore = "10⁶ pairs; run in release with --ignored"]
fn helmholtz_terms_f64_large() {
    terms::<f64>("helmholtz_terms_f64_large", 1_000_000);
}

/// The oracle of a Helmholtz problem: `direct_sum` from the initial outputs, the scales
/// per target, and the reference's own errors.
struct Oracle {
    potential: Vec<Complex<f64>>,
    gradient: Vec<[Complex<f64>; 3]>,
    potential_scale: Vec<f64>,
    gradient_scale: Vec<f64>,
    reference: (f64, f64),
}

impl Oracle {
    fn new<T: Precision>(k: Complex<T>, p: &HProblem<T>) -> Self {
        let (sources, targets) = (widen(&p.sources), widen(&p.targets));
        let charges: Vec<Complex<f64>> = p
            .complex_charges()
            .iter()
            .map(|q| Complex::new(q.re.to_f64(), q.im.to_f64()))
            .collect();
        let k64 = Complex::new(k.re.to_f64(), k.im.to_f64());
        let n = targets.len();
        let w = |v: T| v.to_f64();
        let mut potential: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(w(p.potential[i]), w(p.potential[n + i])))
            .collect();
        let mut gradient: Vec<[Complex<f64>; 3]> = (0..n)
            .map(|i| {
                core::array::from_fn(|c| Complex::new(w(p.gradient[i][c]), w(p.gradient[n + i][c])))
            })
            .collect();
        direct_sum(
            k64,
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        let mut potential_scale = Vec::new();
        let mut gradient_scale = Vec::new();
        for (i, x) in targets.iter().enumerate() {
            let mut sp = w(p.potential[i]).abs().max(w(p.potential[n + i]).abs());
            let mut sg = p.gradient[i]
                .iter()
                .chain(&p.gradient[n + i])
                .map(|g| w(*g).abs())
                .fold(0.0, f64::max);
            for (y, q) in sources.iter().zip(&charges) {
                if x == y {
                    continue;
                }
                let r2: f64 = (0..3).map(|c| (x[c] - y[c]).powi(2)).sum();
                let r = r2.sqrt();
                let decay = (-k64.im * r).exp();
                sp += q.norm() * decay / r;
                sg += q.norm() * (1.0 + k64.norm() * r) * decay / r2;
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
        oracle.reference = oracle.errors(&run_reference(k, p, true));
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
        let n = self.potential.len();
        let mut ep = 0.0f64;
        let mut eg = 0.0f64;
        for i in 0..n {
            let (want, scale) = (self.potential[i], self.potential_scale[i]);
            ep = ep
                .max(relative(phi[i], want.re, scale))
                .max(relative(phi[n + i], want.im, scale));
            let scale = self.gradient_scale[i];
            for ((want, &re), &im) in self.gradient[i].iter().zip(&grad[i]).zip(&grad[n + i]) {
                eg = eg
                    .max(relative(re, want.re, scale))
                    .max(relative(im, want.im, scale));
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
/// the largest with `max_sources` sources, with real and with complex charges.
fn sum_problems<T: Precision>(max_sources: usize) -> Vec<(String, HProblem<T>)> {
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
        for complex in [false, true] {
            let name = format!("W1 n_t={n} mixed={mixed} complex={complex}");
            problems.push((name, HProblem::from(&p, complex, seed + 100)));
        }
    }
    for (ns, nt, seed) in [
        (1, 1, 51),
        (7, 13, 52),
        (300, 71, 53),
        (max_sources, 37, 54),
    ] {
        let p = with_initial(uniform::<T>(ns, nt, seed), seed);
        for complex in [false, true] {
            let name = format!("uniform {ns} × {nt} complex={complex}");
            problems.push((name, HProblem::from(&p, complex, seed + 100)));
        }
    }
    problems
}

/// The wavenumbers of the sum tests, in the units of the points ([−1, 1)³, a leaf of
/// width 2): real, and complex with Im k > 0.
fn sum_wavenumbers<T: Precision>() -> [Complex<T>; 5] {
    [(0.1, 0.0), (1.0, 0.0), (10.0, 0.0), (1.0, 0.5), (10.0, 1.0)]
        .map(|(re, im)| Complex::new(T::from_f64(re), T::from_f64(im)))
}

fn sums<T: Precision>(test: &str, max_sources: usize) {
    let mut worst = Worst::new(format!("{test} ({})", std::any::type_name::<T>()));
    let problems = sum_problems::<T>(max_sources);
    for kernel in kernels::<T>(test) {
        for (name, p) in &problems {
            for k in sum_wavenumbers::<T>() {
                let oracle = Oracle::new(k, p);
                let tol = oracle.tolerance::<T>();
                let errors = oracle.errors(&run_helmholtz(kernel, k, p, true));
                assert!(
                    errors.0 <= tol.0 && errors.1 <= tol.1,
                    "{} {name}, k = {k:?}: errors {:e}, {:e} exceed {:e}, {:e}",
                    kernel.isa(),
                    errors.0,
                    errors.1,
                    tol.0,
                    tol.1
                );
                let only = run_helmholtz(kernel, k, p, false);
                let ep = oracle.errors(&(only.0, p.gradient.clone())).0;
                assert!(ep <= tol.0, "{} {name}, k = {k:?}: {ep:e}", kernel.isa());
                worst.record(kernel.isa(), errors);
            }
        }
    }
    worst.print(1.0, "(relative)");
}

#[test]
fn helmholtz_sums_f32() {
    sums::<f32>("helmholtz_sums_f32", 512);
}

#[test]
fn helmholtz_sums_f64() {
    sums::<f64>("helmholtz_sums_f64", 512);
}

#[test]
#[ignore = "4,096 sources; run in release with --ignored"]
fn helmholtz_sums_f32_large() {
    sums::<f32>("helmholtz_sums_f32_large", 4096);
}

#[test]
#[ignore = "4,096 sources; run in release with --ignored"]
fn helmholtz_sums_f64_large() {
    sums::<f64>("helmholtz_sums_f64_large", 4096);
}

/// Grid-valued problems with some targets on the sources, so that coincident pairs occur.
fn grid_problems<T: Precision>(step: usize) -> Vec<(usize, usize, Problem<T>)> {
    [(0, 5, 61), (9, 0, 62), (13, 29, 63), (64, 41, 64)]
        .into_iter()
        .map(|(ns, nt, seed)| {
            let mut p = seeded(ns, nt, seed).cast::<T>();
            for i in (0..nt.min(ns)).step_by(step) {
                p.targets[i] = p.sources[i];
            }
            (ns, nt, p)
        })
        .collect()
}

fn k_zero_is_laplace<T: Precision>(test: &str) {
    for kernel in kernels::<T>(test) {
        for (ns, nt, p) in grid_problems::<T>(3) {
            let hp = HProblem::from(&p, false, 65);
            for g in [false, true] {
                let (lp, lg) = run(kernel, &p, g);
                let mut laplace = (lp, lg.unwrap_or_else(|| p.gradient.clone()));
                let out = run_helmholtz(kernel, Complex::new(T::zero(), T::zero()), &hp, g);
                let mut real = real_plane(&out, nt);
                if kernel.isa() == Isa::Scalar {
                    // The scalar paths are the references' loops: Laplace's gradient term
                    // −(q dₖ / r³) and Helmholtz's (q ((−1 + 0i) e / r³)) dₖ round
                    // differently; the potentials q / r and ((q + 0i) (1 + 0i)) / r agree.
                    laplace.1.clear();
                    real.1.clear();
                }
                assert_eq!(
                    bits(&real),
                    bits(&laplace),
                    "{} ({ns} sources, {nt} targets, gradient {g})",
                    kernel.isa()
                );
                // The imaginary plane adds zeros.
                let initial = (hp.potential[nt..].to_vec(), hp.gradient[nt..].to_vec());
                let imaginary = (out.0[nt..].to_vec(), out.1[nt..].to_vec());
                assert_eq!(values(&imaginary), values(&initial), "{}", kernel.isa());
            }
        }
    }
}

#[test]
fn helmholtz_at_k_zero_equals_laplace_f32() {
    // Error measure: exact equality of the bits (the real plane), of the values (the
    // imaginary plane).
    k_zero_is_laplace::<f32>("helmholtz_at_k_zero_equals_laplace_f32");
}

#[test]
fn helmholtz_at_k_zero_equals_laplace_f64() {
    // Error measure: as for f32.
    k_zero_is_laplace::<f64>("helmholtz_at_k_zero_equals_laplace_f64");
}

fn imaginary_k_is_yukawa<T: Precision>(test: &str) {
    for kernel in kernels::<T>(test) {
        for (ns, nt, p) in grid_problems::<T>(3) {
            let hp = HProblem::from(&p, false, 66);
            for kappa in [0.5, 3.0].map(T::from_f64) {
                for g in [false, true] {
                    let (mut yp, mut yg) = (p.potential.clone(), p.gradient.clone());
                    let gy = g.then_some(&mut yg[..]);
                    kernel.evaluate_yukawa(kappa, &p.sources, &p.charges, &p.targets, &mut yp, gy);
                    let out = run_helmholtz(kernel, Complex::new(T::zero(), kappa), &hp, g);
                    assert_eq!(
                        bits(&real_plane(&out, nt)),
                        bits(&(yp, yg)),
                        "{} ({ns} sources, {nt} targets, κ = {kappa:?}, gradient {g})",
                        kernel.isa()
                    );
                }
            }
        }
    }
}

#[test]
fn helmholtz_at_imaginary_k_equals_yukawa_f32() {
    // Error measure: exact equality of the bits of the real plane.
    imaginary_k_is_yukawa::<f32>("helmholtz_at_imaginary_k_equals_yukawa_f32");
}

#[test]
fn helmholtz_at_imaginary_k_equals_yukawa_f64() {
    // Error measure: as for f32.
    imaginary_k_is_yukawa::<f64>("helmholtz_at_imaginary_k_equals_yukawa_f64");
}

fn complex_zero_imaginary_is_real<T: Precision>(test: &str) {
    for kernel in kernels::<T>(test) {
        let p = with_initial(uniform::<T>(37, 45, 67), 67);
        let real = HProblem::from(&p, false, 68);
        let mut complex = HProblem::from(&p, true, 68);
        complex.potential.clone_from(&real.potential);
        complex.gradient.clone_from(&real.gradient);
        for q in complex.charges.iter_mut().skip(1).step_by(2) {
            *q = T::zero();
        }
        for k in
            [(1.5, 0.0), (1.5, 0.75)].map(|(a, b)| Complex::new(T::from_f64(a), T::from_f64(b)))
        {
            for g in [false, true] {
                assert_eq!(
                    values(&run_helmholtz(kernel, k, &complex, g)),
                    values(&run_helmholtz(kernel, k, &real, g)),
                    "{} k = {k:?}, gradient {g}",
                    kernel.isa()
                );
            }
        }
    }
}

#[test]
fn helmholtz_complex_charges_with_zero_imaginary_part_equal_real_ones_f32() {
    // Error measure: exact equality of the values.
    complex_zero_imaginary_is_real::<f32>(
        "helmholtz_complex_charges_with_zero_imaginary_part_equal_real_ones_f32",
    );
}

#[test]
fn helmholtz_complex_charges_with_zero_imaginary_part_equal_real_ones_f64() {
    // Error measure: as for f32.
    complex_zero_imaginary_is_real::<f64>(
        "helmholtz_complex_charges_with_zero_imaginary_part_equal_real_ones_f64",
    );
}

fn scalar_is_reference<T: Precision>() {
    let kernel = P2pKernel::<T>::new(Isa::Scalar).unwrap();
    for (_, _, p) in grid_problems::<T>(2) {
        for complex in [false, true] {
            let hp = HProblem::from(&p, complex, 72);
            for (re, im) in [(0.0, 0.0), (0.75, 0.0), (3.5, 0.25), (0.0, 0.5)] {
                let k = Complex::new(T::from_f64(re), T::from_f64(im));
                for g in [false, true] {
                    assert_eq!(
                        bits(&run_helmholtz(kernel, k, &hp, g)),
                        bits(&run_reference(k, &hp, g)),
                        "({} sources, {} targets, complex {complex}, k = {k:?}, gradient {g})",
                        p.sources.len(),
                        p.targets.len()
                    );
                }
            }
        }
    }
}

#[test]
fn scalar_helmholtz_equals_the_reference() {
    // Error measure: exact equality of the bits (grid-valued inputs: r² = 0 exactly
    // where the points coincide).
    scalar_is_reference::<f32>();
    scalar_is_reference::<f64>();
}

fn invariance<T: Precision>(test: &str) {
    for kernel in kernels::<T>(test) {
        for complex in [false, true] {
            let p = HProblem::from(&with_initial(uniform::<T>(37, 45, 81), 81), complex, 82);
            let qp = if complex { 2 } else { 1 };
            for k in [(1.25, 0.0), (1.25, 0.5)]
                .map(|(a, b)| Complex::new(T::from_f64(a), T::from_f64(b)))
            {
                for g in [false, true] {
                    let whole = run_helmholtz(kernel, k, &p, g);
                    // Sources in two calls: the second starts from the first's outputs.
                    for split in [1, 16, 36] {
                        let mut first = p.clone();
                        first.sources.truncate(split);
                        first.charges.truncate(qp * split);
                        let (phi, grad) = run_helmholtz(kernel, k, &first, g);
                        let second = HProblem {
                            sources: p.sources[split..].to_vec(),
                            charges: p.charges[qp * split..].to_vec(),
                            potential: phi,
                            gradient: grad,
                            ..p.clone()
                        };
                        assert_eq!(
                            bits(&run_helmholtz(kernel, k, &second, g)),
                            bits(&whole),
                            "{} complex {complex}, k = {k:?}, split {split}",
                            kernel.isa()
                        );
                    }
                    // The last `len` targets alone, for lengths from one target to all, give
                    // their bits in the whole call.
                    let n = 45;
                    let planes = |v: &[T], len: usize| [&v[n - len..n], &v[2 * n - len..]].concat();
                    let gplanes =
                        |v: &[[T; 3]], len: usize| [&v[n - len..n], &v[2 * n - len..]].concat();
                    for len in [1, 7, 8, 17, 33, 45] {
                        let part = HProblem {
                            targets: p.targets[n - len..].to_vec(),
                            potential: planes(&p.potential, len),
                            gradient: gplanes(&p.gradient, len),
                            ..p.clone()
                        };
                        let out = run_helmholtz(kernel, k, &part, g);
                        let tail = (planes(&whole.0, len), gplanes(&whole.1, len));
                        assert_eq!(
                            bits(&out),
                            bits(&tail),
                            "{} complex {complex}, k = {k:?}, last {len}",
                            kernel.isa()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn helmholtz_invariance_f32() {
    // Error measure: exact equality of the bits.
    invariance::<f32>("helmholtz_invariance_f32");
}

#[test]
fn helmholtz_invariance_f64() {
    // Error measure: exact equality of the bits.
    invariance::<f64>("helmholtz_invariance_f64");
}

fn coincident_and_empty<T: Precision>(test: &str) {
    let k = Complex::new(T::from_f64(2.0), T::from_f64(0.5));
    for kernel in kernels::<T>(test) {
        for complex in [false, true] {
            // Targets on the sources: the self pairs add nothing, the outputs stay finite
            // and within the sum tolerance of the direct sum, which skips x == y.
            let mut base = with_initial(uniform::<T>(23, 23, 91), 91);
            base.targets = base.sources.clone();
            let p = HProblem::from(&base, complex, 92);
            let oracle = Oracle::new(k, &p);
            let out = run_helmholtz(kernel, k, &p, true);
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
            let empty = HProblem {
                sources: Vec::new(),
                charges: Vec::new(),
                ..p.clone()
            };
            for g in [false, true] {
                let out = run_helmholtz(kernel, k, &empty, g);
                assert_eq!(bits(&out), bits(&(p.potential.clone(), p.gradient.clone())));
            }
            // No targets.
            let none = HProblem {
                targets: Vec::new(),
                potential: Vec::new(),
                gradient: Vec::new(),
                ..p.clone()
            };
            let out = run_helmholtz(kernel, k, &none, true);
            assert!(out.0.is_empty() && out.1.is_empty());
        }
    }
}

#[test]
fn helmholtz_coincident_and_empty_inputs_f32() {
    // Error measure: finite outputs within the sum tolerance; the bits of untouched
    // outputs.
    coincident_and_empty::<f32>("helmholtz_coincident_and_empty_inputs_f32");
}

#[test]
fn helmholtz_coincident_and_empty_inputs_f64() {
    // Error measure: as for f32.
    coincident_and_empty::<f64>("helmholtz_coincident_and_empty_inputs_f64");
}

#[test]
#[should_panic(expected = "`sources` and `charges` must have the same length, got 2 and 1")]
fn evaluate_helmholtz_rejects_mismatched_real_charges() {
    let kernel = P2pKernel::<f64>::detect();
    let mut phi = [0.0; 2];
    let charges = HelmholtzCharges::Real(&[1.0]);
    let k = Complex::new(1.0, 0.0);
    kernel.evaluate_helmholtz(k, &[[0.0; 3]; 2], charges, &[[1.0; 3]], &mut phi, None);
}

#[test]
#[should_panic(expected = "`sources × 2` and `charges` must have the same length, got 2 and 1")]
fn evaluate_helmholtz_rejects_mismatched_complex_charges() {
    let kernel = P2pKernel::<f64>::detect();
    let mut phi = [0.0; 2];
    let charges = HelmholtzCharges::Complex(&[1.0]);
    let k = Complex::new(1.0, 0.0);
    kernel.evaluate_helmholtz(k, &[[0.0; 3]], charges, &[[1.0; 3]], &mut phi, None);
}

#[test]
#[should_panic(expected = "`targets × 2` and `potential` must have the same length, got 2 and 1")]
fn evaluate_helmholtz_rejects_one_potential_plane() {
    let kernel = P2pKernel::<f32>::detect();
    let mut phi = [0.0];
    let charges = HelmholtzCharges::Real(&[1.0]);
    let k = Complex::new(1.0, 0.0);
    kernel.evaluate_helmholtz(k, &[[0.0; 3]], charges, &[[1.0; 3]], &mut phi, None);
}

#[test]
#[should_panic(expected = "`targets × 2` and `gradient` must have the same length, got 2 and 1")]
fn evaluate_helmholtz_rejects_one_gradient_plane() {
    let kernel = P2pKernel::<f32>::detect();
    let (mut phi, mut grad) = ([0.0; 2], [[0.0; 3]; 1]);
    let charges = HelmholtzCharges::Real(&[1.0]);
    let k = Complex::new(1.0, 0.0);
    kernel.evaluate_helmholtz(
        k,
        &[[0.0; 3]],
        charges,
        &[[1.0; 3]],
        &mut phi,
        Some(&mut grad),
    );
}

fn rejects_k(re: f64, im: f64) {
    let kernel = P2pKernel::<f64>::detect();
    let mut phi = [0.0; 2];
    let charges = HelmholtzCharges::Real(&[1.0]);
    let k = Complex::new(re, im);
    kernel.evaluate_helmholtz(k, &[[0.0; 3]], charges, &[[1.0; 3]], &mut phi, None);
}

#[test]
#[should_panic(expected = "Helmholtz P2P needs a finite wavenumber k with Re k ≥ 0 and Im k ≥ 0")]
fn evaluate_helmholtz_rejects_negative_real_part() {
    rejects_k(-0.5, 0.0);
}

#[test]
#[should_panic(expected = "Helmholtz P2P needs a finite wavenumber k with Re k ≥ 0 and Im k ≥ 0")]
fn evaluate_helmholtz_rejects_negative_imaginary_part() {
    rejects_k(1.0, -0.5);
}

#[test]
#[should_panic(expected = "Helmholtz P2P needs a finite wavenumber k with Re k ≥ 0 and Im k ≥ 0")]
fn evaluate_helmholtz_rejects_nan() {
    rejects_k(f64::NAN, 0.0);
}

#[test]
#[should_panic(expected = "Helmholtz P2P needs a finite wavenumber k with Re k ≥ 0 and Im k ≥ 0")]
fn evaluate_helmholtz_rejects_infinity() {
    rejects_k(1.0, f64::INFINITY);
}

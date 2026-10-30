//! Particle-to-particle interaction of the Helmholtz family: the near-field operator
//! [`p2p`] and the f64 direct-sum oracle [`direct_sum`] for complex k, and their real
//! Yukawa counterparts [`p2p_yukawa`] and [`direct_sum_yukawa`] (CONVENTIONS §3.15,
//! "Kernel and factors").
//!
//! Each adds, at each target xᵢ, the potential and optionally the gradient with respect
//! to xᵢ of point charges qⱼ at sources yⱼ, with rⱼ = |xᵢ − yⱼ|,
//!
//! φ̃(xᵢ) += Σⱼ qⱼ e^{ikrⱼ}/rⱼ,  ∇φ̃(xᵢ) += Σⱼ qⱼ (ikrⱼ − 1) e^{ikrⱼ}/rⱼ³ (xᵢ − yⱼ),
//!
//! the potential φ̃ = 4πφ of §3.15, without the factor 1/(4π) (which `nd-fmm-exec`
//! applies once). For Yukawa, k = iκ, the kernel is e^{−κr}/r and the gradient factor
//! −(κr + 1) e^{−κr}/r³, all real. Every function accumulates (+=) into its outputs;
//! `gradient`, if given, holds ∂x, ∂y, ∂z per target.
//!
//! # The term
//!
//! With d = xᵢ − yⱼ, r² = (d₀² + d₁²) + d₂² and r = √r² as in Laplace's [`crate::p2p`],
//!
//! ```text
//! e  = e^{−Im(k) r} (cos(Re(k) r) + i sin(Re(k) r))       (= e^{ikr})
//! φ̃ ← φ̃ + (q e) / r
//! g  = ((−Im(k) r − 1) + i Re(k) r) e / (r² r)             (= (ikr − 1) e^{ikr}/r³)
//! ∇φ̃ₖ ← ∇φ̃ₖ + (q g) dₖ
//! ```
//!
//! and the Yukawa term is the same with Re k = 0 and Im k = κ in real arithmetic. At
//! k = iκ the complex term performs exactly the operations of the Yukawa term on each
//! part of the charge (its imaginary parts are exact zeros), so the two agree bit for
//! bit; at k = 0 the potential is Laplace's q / r bit for bit. The phase Re(k) r is
//! rounded once, so a term carries a relative error of order |k| r times the unit
//! roundoff.
//!
//! # Wavenumbers
//!
//! The functions compute the formula for any finite k; the range Re k ≥ 0, Im k ≥ 0,
//! k ≠ 0 of §3.15 is the builder's to enforce. k = 0 gives Laplace's kernel, which the
//! limit tests use.
//!
//! # Coincident points
//!
//! As in [`crate::p2p`]: a pair with xᵢ == yⱼ exactly contributes nothing, which excludes
//! self-interaction when targets and sources are the same points (§3.13, "Coincident
//! pairs"); nearly coincident pairs are not skipped.
//!
//! # The two functions of each pair
//!
//! - [`p2p`] and [`p2p_yukawa`], generic over [`RealScalar`], sum over the sources in input
//!   order with plain additions into the outputs, so their results are reproducible
//!   operation by operation. The fast Helmholtz P2P kernels are checked against them.
//! - [`direct_sum`] and [`direct_sum_yukawa`] are the f64 oracles: the same terms, summed
//!   per target with Neumaier's compensated summation, separately for each real component,
//!   and added to the output in one final addition. Serial, O(N_s · N_t).
//!
//! # Panics
//!
//! Every function panics if `sources` and `charges` differ in length, if `potential` and
//! `targets` differ in length, if a given `gradient` has a length other than
//! `targets.len()`, or if the wavenumber is not finite.
//!
//! # Example
//!
//! ```
//! use nd_fmm_math::Complex;
//! use nd_fmm_ref::helmholtz::p2p::{direct_sum, direct_sum_yukawa};
//!
//! let sources = [[0.0, 0.0, 0.0]];
//! let targets = [[0.0, 0.0, 2.0]];
//! let k = Complex::new(0.5, 0.0);
//! let mut potential = [Complex::new(0.0, 0.0)];
//! direct_sum(k, &sources, &[Complex::new(1.0, 0.0)], &targets, &mut potential, None);
//! // e^{ikr}/r at k r = 1.
//! assert!((potential[0].re - 1.0_f64.cos() / 2.0).abs() < 1e-16);
//! assert!((potential[0].im - 1.0_f64.sin() / 2.0).abs() < 1e-16);
//!
//! // Yukawa: the real kernel equals the complex one at k = iκ, bit for bit.
//! let mut real = [0.0];
//! direct_sum_yukawa(0.5, &sources, &[1.0], &targets, &mut real, None);
//! let mut complex = [Complex::new(0.0, 0.0)];
//! let k = Complex::new(0.0, 0.5);
//! direct_sum(k, &sources, &[Complex::new(1.0, 0.0)], &targets, &mut complex, None);
//! assert_eq!(complex[0], Complex::new(real[0], 0.0));
//! ```

use nd_fmm_math::{Complex, RealScalar};

use crate::p2p::{Neumaier, check_pair};

/// P2P: adds the potential e^{ikr}/r of point charges, and optionally its gradient, at
/// each target (CONVENTIONS §3.15).
///
/// The sources are visited in input order and each term of the
/// [module documentation](self) is added directly into `potential` and `gradient`, with
/// no reordering and no compensation. A pair with xᵢ == yⱼ exactly is skipped.
///
/// # Panics
///
/// If `sources.len() != charges.len()`, `potential.len() != targets.len()`, a given
/// `gradient` has a length other than `targets.len()`, or k is not finite.
pub fn p2p<T: RealScalar>(
    k: Complex<T>,
    sources: &[[T; 3]],
    charges: &[Complex<T>],
    targets: &[[T; 3]],
    potential: &mut [Complex<T>],
    gradient: Option<&mut [[Complex<T>; 3]]>,
) {
    check_k(k);
    check_lengths(
        sources,
        charges.len(),
        targets,
        potential.len(),
        gradient.as_deref().map(<[_]>::len),
    );
    match gradient {
        None => {
            for (&x, phi) in targets.iter().zip(potential.iter_mut()) {
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = Term::new(k, x, y) {
                        *phi = *phi + term.potential(q);
                    }
                }
            }
        }
        Some(gradient) => {
            for ((&x, phi), g) in targets.iter().zip(potential.iter_mut()).zip(gradient) {
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = Term::new(k, x, y) {
                        *phi = *phi + term.potential(q);
                        for (gk, dgk) in g.iter_mut().zip(term.gradient(q)) {
                            *gk = *gk + dgk;
                        }
                    }
                }
            }
        }
    }
}

/// Direct sum: the f64 oracle for the potential e^{ikr}/r of point charges, and
/// optionally its gradient, at each target (CONVENTIONS §3.15).
///
/// Each term is computed exactly as in [`p2p`]. The terms of each target are summed in
/// input order with Neumaier's compensated summation, separately for the real and the
/// imaginary part of the potential and of each gradient component, as Laplace's
/// [`crate::p2p::direct_sum`] does; the compensated sums are then added to the outputs in
/// one final addition each. A pair with xᵢ == yⱼ exactly is skipped.
///
/// # Panics
///
/// As [`p2p`].
pub fn direct_sum(
    k: Complex<f64>,
    sources: &[[f64; 3]],
    charges: &[Complex<f64>],
    targets: &[[f64; 3]],
    potential: &mut [Complex<f64>],
    gradient: Option<&mut [[Complex<f64>; 3]]>,
) {
    check_k(k);
    check_lengths(
        sources,
        charges.len(),
        targets,
        potential.len(),
        gradient.as_deref().map(<[_]>::len),
    );
    match gradient {
        None => {
            for (&x, phi) in targets.iter().zip(potential.iter_mut()) {
                let mut sum = ComplexSum::default();
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = Term::new(k, x, y) {
                        sum.add(term.potential(q));
                    }
                }
                *phi += sum.value();
            }
        }
        Some(gradient) => {
            for ((&x, phi), g) in targets.iter().zip(potential.iter_mut()).zip(gradient) {
                let mut sum = ComplexSum::default();
                let mut grad = [ComplexSum::default(); 3];
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = Term::new(k, x, y) {
                        sum.add(term.potential(q));
                        for (gk, dgk) in grad.iter_mut().zip(term.gradient(q)) {
                            gk.add(dgk);
                        }
                    }
                }
                *phi += sum.value();
                for (gk, sk) in g.iter_mut().zip(&grad) {
                    *gk += sk.value();
                }
            }
        }
    }
}

/// Yukawa P2P: adds the potential e^{−κr}/r of real point charges, and optionally its
/// gradient −(κr + 1) e^{−κr}/r³ (x − y), at each target (CONVENTIONS §3.15, k = iκ).
///
/// The real counterpart of [`p2p`]: the same order and operations with Re k = 0, so that
/// [`p2p`] at k = iκ gives this result in its real part, bit for bit, for each real
/// charge vector. A pair with xᵢ == yⱼ exactly is skipped.
///
/// # Panics
///
/// As [`p2p`], with κ for k.
pub fn p2p_yukawa<T: RealScalar>(
    kappa: T,
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: Option<&mut [[T; 3]]>,
) {
    check_kappa(kappa);
    check_lengths(
        sources,
        charges.len(),
        targets,
        potential.len(),
        gradient.as_deref().map(<[_]>::len),
    );
    match gradient {
        None => {
            for (&x, phi) in targets.iter().zip(potential.iter_mut()) {
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = YukawaTerm::new(kappa, x, y) {
                        *phi = *phi + term.potential(q);
                    }
                }
            }
        }
        Some(gradient) => {
            for ((&x, phi), g) in targets.iter().zip(potential.iter_mut()).zip(gradient) {
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = YukawaTerm::new(kappa, x, y) {
                        *phi = *phi + term.potential(q);
                        for (gk, dgk) in g.iter_mut().zip(term.gradient(q)) {
                            *gk = *gk + dgk;
                        }
                    }
                }
            }
        }
    }
}

/// Yukawa direct sum: the f64 oracle for the potential e^{−κr}/r of real point charges,
/// and optionally its gradient, at each target (CONVENTIONS §3.15, k = iκ).
///
/// The terms of [`p2p_yukawa`], summed as in [`direct_sum`]; [`direct_sum`] at k = iκ
/// gives this result in its real part, bit for bit.
///
/// # Panics
///
/// As [`p2p`], with κ for k.
pub fn direct_sum_yukawa(
    kappa: f64,
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
    potential: &mut [f64],
    gradient: Option<&mut [[f64; 3]]>,
) {
    check_kappa(kappa);
    check_lengths(
        sources,
        charges.len(),
        targets,
        potential.len(),
        gradient.as_deref().map(<[_]>::len),
    );
    match gradient {
        None => {
            for (&x, phi) in targets.iter().zip(potential.iter_mut()) {
                let mut sum = Neumaier::default();
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = YukawaTerm::new(kappa, x, y) {
                        sum.add(term.potential(q));
                    }
                }
                *phi += sum.value();
            }
        }
        Some(gradient) => {
            for ((&x, phi), g) in targets.iter().zip(potential.iter_mut()).zip(gradient) {
                let mut sum = Neumaier::default();
                let mut grad = [Neumaier::default(); 3];
                for (&y, &q) in sources.iter().zip(charges) {
                    if let Some(term) = YukawaTerm::new(kappa, x, y) {
                        sum.add(term.potential(q));
                        for (gk, dgk) in grad.iter_mut().zip(term.gradient(q)) {
                            gk.add(dgk);
                        }
                    }
                }
                *phi += sum.value();
                for (gk, sk) in g.iter_mut().zip(&grad) {
                    *gk += sk.value();
                }
            }
        }
    }
}

/// The pair quantities of one target and source: d = x − y, r, r² r, and the complex
/// e = e^{ikr} and ikr − 1.
struct Term<T> {
    d: [T; 3],
    r: T,
    r3: T,
    e: Complex<T>,
    factor: Complex<T>,
}

impl<T: RealScalar> Term<T> {
    #[inline]
    fn new(k: Complex<T>, x: [T; 3], y: [T; 3]) -> Option<Self> {
        if x == y {
            return None;
        }
        let d = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let r = r2.sqrt();
        let damping = -(k.im * r);
        let phase = k.re * r;
        let decay = damping.exp();
        Some(Self {
            d,
            r,
            r3: r2 * r,
            e: Complex::new(decay * phase.cos(), decay * phase.sin()),
            factor: Complex::new(damping - T::one(), phase),
        })
    }

    /// (q e) / r.
    #[inline]
    fn potential(&self, q: Complex<T>) -> Complex<T> {
        q * self.e / self.r
    }

    /// (q g) dₖ with g = ((ikr − 1) e) / (r² r).
    #[inline]
    fn gradient(&self, q: Complex<T>) -> [Complex<T>; 3] {
        let qg = q * (self.factor * self.e / self.r3);
        self.d.map(|dk| qg * dk)
    }
}

/// The real pair quantities of the Yukawa kernel: [`Term`] at Re k = 0, Im k = κ.
struct YukawaTerm<T> {
    d: [T; 3],
    r: T,
    r3: T,
    e: T,
    factor: T,
}

impl<T: RealScalar> YukawaTerm<T> {
    #[inline]
    fn new(kappa: T, x: [T; 3], y: [T; 3]) -> Option<Self> {
        if x == y {
            return None;
        }
        let d = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let r = r2.sqrt();
        let damping = -(kappa * r);
        Some(Self {
            d,
            r,
            r3: r2 * r,
            e: damping.exp(),
            factor: damping - T::one(),
        })
    }

    /// (q e) / r.
    #[inline]
    fn potential(&self, q: T) -> T {
        q * self.e / self.r
    }

    /// (q g) dₖ with g = (−(κr) − 1) e / (r² r).
    #[inline]
    fn gradient(&self, q: T) -> [T; 3] {
        let qg = q * (self.factor * self.e / self.r3);
        self.d.map(|dk| qg * dk)
    }
}

/// Neumaier's compensated sum of the real and the imaginary parts.
#[derive(Clone, Copy, Default)]
struct ComplexSum {
    re: Neumaier,
    im: Neumaier,
}

impl ComplexSum {
    #[inline]
    fn add(&mut self, term: Complex<f64>) {
        self.re.add(term.re);
        self.im.add(term.im);
    }

    #[inline]
    fn value(&self) -> Complex<f64> {
        Complex::new(self.re.value(), self.im.value())
    }
}

fn check_k<T: RealScalar>(k: Complex<T>) {
    assert!(
        k.re.is_finite() && k.im.is_finite(),
        "Helmholtz P2P needs a finite wavenumber k"
    );
}

fn check_kappa<T: RealScalar>(kappa: T) {
    assert!(kappa.is_finite(), "Yukawa P2P needs a finite wavenumber κ");
}

fn check_lengths<T>(
    sources: &[[T; 3]],
    charges: usize,
    targets: &[[T; 3]],
    potential: usize,
    gradient: Option<usize>,
) {
    check_pair(sources.len(), charges, "sources", "charges");
    check_pair(targets.len(), potential, "targets", "potential");
    if let Some(gradient) = gradient {
        check_pair(targets.len(), gradient, "targets", "gradient");
    }
}

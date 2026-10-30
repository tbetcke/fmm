//! Leaf operators of the Helmholtz family: P2M and P2L form a box's expansions from point
//! charges, L2P and M2P evaluate them at targets (CONVENTIONS §3.15, "Radial functions
//! and scaling" and "Storage").
//!
//! # Expansions
//!
//! About a box centre c, with v = x − c and ρⱼ = yⱼ − c in the units of the points, a
//! multipole represents φ̃(x) = 4π ik Σₙ,ₘ Mₙᵐ hₙ(k|v|) Yₙᵐ(v̂) outside the box's sphere
//! and a local φ̃(x) = Σₙ,ₘ Lₙᵐ jₙ(k|v|) Yₙᵐ(v̂) inside it, n ≤ p, with the real
//! orthonormal harmonics Yₙᵐ of §3.15 ([`nd_fmm_math::orthonormal::harmonics`]) and
//! hₙ = hₙ⁽¹⁾. The coefficients are stored with the [`Level`]'s power scale s and damping
//! exponent β, M̂ = e^{−β} M / sⁿ and L̂ = e^β L sⁿ, and each operator applies only the
//! scaled radial functions of [`nd_fmm_math::radial`] and a non-positive exponent:
//!
//! ```text
//! P2M   M̂ₙᵐ += Σⱼ qⱼ ĵₙ(kρⱼ) e^{Im(k)ρⱼ − β} Yₙᵐ(ρ̂ⱼ)            ρⱼ ≤ a
//! L2P   φ̃   += Σ L̂ₙᵐ ĵₙ(k|v|) e^{Im(k)|v| − β} Yₙᵐ(v̂)          |v| ≤ a
//! P2L   L̂ₙᵐ += 4π ik Σⱼ qⱼ ĥₙ(kρⱼ) e^{β − Im(k)ρⱼ} Yₙᵐ(ρ̂ⱼ)       ρⱼ > a
//! M2P   φ̃   += 4π ik Σ M̂ₙᵐ ĥₙ(k|v|) e^{β − Im(k)|v|} Yₙᵐ(v̂)     |v| > a
//! ```
//!
//! with a the level's circumradius. The factor 4π ik is applied by P2L and M2P, the two
//! leaf operators where a contribution crosses from a singular to a regular
//! representation (§3.15, "Where the factors go"); none applies 1/(4π). A complex charge
//! enters as it is: the harmonics are real and never conjugated.
//!
//! The gradient with respect to x is that of the sum: for f(|v|) Yₙᵐ(v̂),
//! f′ Yₙᵐ r̂ + (f/|v|)(∂θ Yₙᵐ θ̂ + (1/sin θ) ∂φ Yₙᵐ φ̂), with f′ = k ĵₙ′ (the radial
//! derivatives of [`nd_fmm_math::radial`]) and the angular derivatives of
//! [`nd_fmm_math::orthonormal::harmonics_derivatives`]. At the centre of a local
//! (|v| = 0) the tangential factor takes its limit, k/(3s) for n = 1 and 0 otherwise.
//!
//! # Storage
//!
//! The complex path ([`p2m`], [`p2l`], [`l2p`], [`m2p`]) serves any k of the [`Level`] and
//! complex charges. A complex expansion is two real planes of (p + 1)² in the layout
//! idx(n, m) of §3.6, the real parts then the imaginary parts. For real k and real charges
//! the multipole is real (§3.15's storage table): [`p2m`] leaves its imaginary plane at
//! exactly zero.
//!
//! Yukawa's real path ([`p2m_yukawa`], [`p2l_yukawa`], [`l2p_yukawa`], [`m2p_yukawa`])
//! needs k = iκ and real charges. It stores M̂^Y = i⁻ⁿ M̂ and L̂^Y = iⁿ L̂, (p + 1)² reals,
//! with î, k̂ in place of ĵ, ĥ and 4πκ in place of 4π ik:
//!
//! ```text
//! P2M   M̂ᵞₙᵐ += Σⱼ qⱼ îₙ(κρⱼ) e^{κρⱼ − β} Yₙᵐ(ρ̂ⱼ)
//! L2P   φ̃    += Σ L̂ᵞₙᵐ îₙ(κ|v|) e^{κ|v| − β} Yₙᵐ(v̂)
//! P2L   L̂ᵞₙᵐ += 4πκ Σⱼ qⱼ k̂ₙ(κρⱼ) e^{β − κρⱼ} Yₙᵐ(ρ̂ⱼ)
//! M2P   φ̃    += 4πκ Σ M̂ᵞₙᵐ k̂ₙ(κ|v|) e^{β − κ|v|} Yₙᵐ(v̂)
//! ```
//!
//! A complex charge vector on the Yukawa path is two real vectors.
//!
//! # Tested range
//!
//! `tests/helmholtz/` checks the operators in f64 for P2M sources and L2P targets in the
//! box (its corners and centre included, ρ ≤ a) and P2L sources and M2P targets at
//! |ρ| = 2.5 a, for |k| a from 10⁻⁶ to 100 (Im(k) a to 100 in the series), Yukawa against
//! the complex path to κ a = 300, and finiteness at Im(k) a = 1000 with P2L sources and
//! M2P targets at 1.5 a. Outside the sphere's side of each operator the
//! applied exponent is positive and the result is not guaranteed. f32 is checked at
//! p = 8 and s = 1 only (CONVENTIONS §3.15 reserves it).
//!
//! # Panics
//!
//! Every operator panics if the workspace was built for a smaller degree than `p`, if the
//! frame's radius is not the level's half-width, if a coefficient slice does not have
//! length 2(p + 1)² (complex path) or (p + 1)² (Yukawa path), if paired slices differ in
//! length, and the Yukawa operators if the level is not Yukawa. [`p2l`], [`m2p`] and their
//! Yukawa forms panic for a point at the centre.
//!
//! # Example
//!
//! ```
//! use nd_fmm_math::Complex;
//! use nd_fmm_ref::Frame;
//! use nd_fmm_ref::helmholtz::{Level, Workspace, leaf, p2p};
//!
//! let p = 20;
//! let k = Complex::new(2.0, 0.5);
//! let level = Level::new(k, 0.5);
//! let frame = Frame::new([0.0; 3], 0.5);
//! let mut ws = Workspace::new(p);
//!
//! let sources = [[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]];
//! let charges = [Complex::new(1.0, 0.5), Complex::new(-0.5, 0.0)];
//! let mut multipole = vec![0.0; 2 * (p + 1) * (p + 1)];
//! leaf::p2m(p, &level, &frame, &sources, &charges, &mut ws, &mut multipole);
//!
//! let targets = [[3.0, 0.5, -1.0]];
//! let mut potential = [Complex::new(0.0, 0.0)];
//! leaf::m2p(p, &level, &frame, &multipole, &targets, &mut ws, &mut potential, None);
//!
//! let mut exact = [Complex::new(0.0, 0.0)];
//! p2p::direct_sum(k, &sources, &charges, &targets, &mut exact, None);
//! assert!((potential[0] - exact[0]).norm() < 1e-10 * exact[0].norm());
//! ```

use core::ops::{Add, Mul};

use nd_fmm_math::orthonormal::{DerivativeScratch, harmonics, harmonics_derivatives};
use nd_fmm_math::radial::{bessel_j, hankel_h, modified_i, modified_k};
use nd_fmm_math::{Complex, Layout, RealScalar};
use num_traits::Zero;

use super::workspace::Buffers;
use super::{Level, Workspace};
use crate::Frame;
use crate::p2p::check_pair;

/// P2M: adds the stored multipole coefficients of complex point charges to `multipole`,
/// M̂ₙᵐ += Σⱼ qⱼ ĵₙ(kρⱼ) e^{Im(k)ρⱼ − β} Yₙᵐ(ρ̂ⱼ) (module documentation).
///
/// `multipole` holds 2(p + 1)² reals, the real plane then the imaginary plane. Each
/// source is added in turn, degree by degree.
///
/// # Panics
///
/// If `ws.p() < p`, `frame.radius != level.half_width()`, `multipole.len() != 2(p + 1)²`
/// or `sources.len() != charges.len()`.
pub fn p2m<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    frame: &Frame<T>,
    sources: &[[T; 3]],
    charges: &[Complex<T>],
    ws: &mut Workspace<T>,
    multipole: &mut [T],
) {
    check_frame(level, frame);
    check_coefficients(p, multipole, 2, "multipole");
    check_pair(sources.len(), charges.len(), "sources", "charges");
    let buffers = ws.buffers(p);
    let k = level.k();
    let (re, im) = multipole.split_at_mut(multipole.len() / 2);
    for (&y, &q) in sources.iter().zip(charges) {
        let rho = offset(frame, y);
        let r = Polar::of(rho).r;
        bessel_j(k * r, level.scale(), buffers.radial, None);
        harmonics(p, rho, buffers.harmonics);
        let e = (k.im * r - level.damping()).exp();
        form(p, q * e, buffers.radial, buffers.harmonics, |i, w| {
            re[i] = re[i] + w.re;
            im[i] = im[i] + w.im;
        });
    }
}

/// P2L: adds the stored local coefficients of complex point charges to `local`,
/// L̂ₙᵐ += 4π ik Σⱼ qⱼ ĥₙ(kρⱼ) e^{β − Im(k)ρⱼ} Yₙᵐ(ρ̂ⱼ) (module documentation).
///
/// `local` holds 2(p + 1)² reals, the real plane then the imaginary plane. Each source is
/// added in turn, degree by degree.
///
/// # Panics
///
/// As [`p2m`], with `local` for `multipole`, and for a source at the centre.
pub fn p2l<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    frame: &Frame<T>,
    sources: &[[T; 3]],
    charges: &[Complex<T>],
    ws: &mut Workspace<T>,
    local: &mut [T],
) {
    check_frame(level, frame);
    check_coefficients(p, local, 2, "local");
    check_pair(sources.len(), charges.len(), "sources", "charges");
    let buffers = ws.buffers(p);
    let k = level.k();
    let factor = four_pi_ik(k);
    let (re, im) = local.split_at_mut(local.len() / 2);
    for (&y, &q) in sources.iter().zip(charges) {
        let rho = offset(frame, y);
        let r = Polar::of(rho).r;
        hankel_h(k * r, level.scale(), buffers.radial, None);
        harmonics(p, rho, buffers.harmonics);
        let e = (level.damping() - k.im * r).exp();
        form(
            p,
            factor * q * e,
            buffers.radial,
            buffers.harmonics,
            |i, w| {
                re[i] = re[i] + w.re;
                im[i] = im[i] + w.im;
            },
        );
    }
}

/// L2P: adds the potential of a stored complex local expansion, and optionally its
/// gradient, at each target, φ̃ += Σ L̂ₙᵐ ĵₙ(k|v|) e^{Im(k)|v| − β} Yₙᵐ(v̂) (module
/// documentation). `gradient`, if given, holds ∂x, ∂y, ∂z per target.
///
/// # Panics
///
/// If `ws.p() < p`, `frame.radius != level.half_width()`, `local.len() != 2(p + 1)²`,
/// `potential.len() != targets.len()` or a given `gradient` has another length.
#[allow(clippy::too_many_arguments)]
pub fn l2p<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    frame: &Frame<T>,
    local: &[T],
    targets: &[[T; 3]],
    ws: &mut Workspace<T>,
    potential: &mut [Complex<T>],
    gradient: Option<&mut [[Complex<T>; 3]]>,
) {
    check_frame(level, frame);
    check_coefficients(p, local, 2, "local");
    check_targets(targets.len(), potential.len(), gradient.as_deref());
    let mut buffers = ws.buffers(p);
    let k = level.k();
    let (re, im) = local.split_at(local.len() / 2);
    let coefficient = |i: usize| Complex::new(re[i], im[i]);
    // The tangential factor's limit at the centre, k ĵ₁(z)/z → k/(3s).
    let at_centre = k * (T::one() / (T::from_f64(3.0) * level.scale()));
    for_each_target(targets, potential, gradient, |request, x| {
        let v = offset(frame, x);
        let polar = Polar::of(v);
        let z = k * polar.r;
        let e = (k.im * polar.r - level.damping()).exp();
        let with_gradient = request.gradient;
        bessel_j(
            z,
            level.scale(),
            buffers.radial,
            with_gradient.then_some(&mut *buffers.radial_derivative),
        );
        let (phi, grad) = evaluate(
            p,
            v,
            &polar,
            &coefficient,
            &mut buffers.as_complex(),
            k,
            at_centre,
            with_gradient,
        );
        (phi * e, grad.map(|g| g.map(|c| c * e)))
    });
}

/// M2P: adds the potential of a stored complex multipole expansion, and optionally its
/// gradient, at each target, φ̃ += 4π ik Σ M̂ₙᵐ ĥₙ(k|v|) e^{β − Im(k)|v|} Yₙᵐ(v̂) (module
/// documentation). `gradient`, if given, holds ∂x, ∂y, ∂z per target.
///
/// # Panics
///
/// As [`l2p`], with `multipole` for `local`, and for a target at the centre.
#[allow(clippy::too_many_arguments)]
pub fn m2p<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    frame: &Frame<T>,
    multipole: &[T],
    targets: &[[T; 3]],
    ws: &mut Workspace<T>,
    potential: &mut [Complex<T>],
    gradient: Option<&mut [[Complex<T>; 3]]>,
) {
    check_frame(level, frame);
    check_coefficients(p, multipole, 2, "multipole");
    check_targets(targets.len(), potential.len(), gradient.as_deref());
    let mut buffers = ws.buffers(p);
    let k = level.k();
    let factor = four_pi_ik(k);
    let (re, im) = multipole.split_at(multipole.len() / 2);
    let coefficient = |i: usize| Complex::new(re[i], im[i]);
    for_each_target(targets, potential, gradient, |request, x| {
        let v = offset(frame, x);
        let polar = Polar::of(v);
        let z = k * polar.r;
        let e = (level.damping() - k.im * polar.r).exp();
        let with_gradient = request.gradient;
        hankel_h(
            z,
            level.scale(),
            buffers.radial,
            with_gradient.then_some(&mut *buffers.radial_derivative),
        );
        let zero = Complex::zero();
        let (phi, grad) = evaluate(
            p,
            v,
            &polar,
            &coefficient,
            &mut buffers.as_complex(),
            k,
            zero,
            with_gradient,
        );
        let scale = factor * e;
        (phi * scale, grad.map(|g| g.map(|c| c * scale)))
    });
}

/// Yukawa P2M: adds M̂ᵞₙᵐ += Σⱼ qⱼ îₙ(κρⱼ) e^{κρⱼ − β} Yₙᵐ(ρ̂ⱼ) for real charges at a Yukawa
/// level, k = iκ (module documentation, "Storage"). `multipole` holds (p + 1)² reals.
///
/// # Panics
///
/// If the level is not Yukawa, `ws.p() < p`, `frame.radius != level.half_width()`,
/// `multipole.len() != (p + 1)²` or `sources.len() != charges.len()`.
pub fn p2m_yukawa<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    frame: &Frame<T>,
    sources: &[[T; 3]],
    charges: &[T],
    ws: &mut Workspace<T>,
    multipole: &mut [T],
) {
    let kappa = check_yukawa(level);
    check_frame(level, frame);
    check_coefficients(p, multipole, 1, "multipole");
    check_pair(sources.len(), charges.len(), "sources", "charges");
    let buffers = ws.buffers(p);
    for (&y, &q) in sources.iter().zip(charges) {
        let rho = offset(frame, y);
        let r = Polar::of(rho).r;
        modified_i(kappa * r, level.scale(), buffers.real, None);
        harmonics(p, rho, buffers.harmonics);
        let e = (kappa * r - level.damping()).exp();
        form(p, q * e, buffers.real, buffers.harmonics, |i, w| {
            multipole[i] = multipole[i] + w;
        });
    }
}

/// Yukawa P2L: adds L̂ᵞₙᵐ += 4πκ Σⱼ qⱼ k̂ₙ(κρⱼ) e^{β − κρⱼ} Yₙᵐ(ρ̂ⱼ) for real charges at a
/// Yukawa level (module documentation, "Storage"). `local` holds (p + 1)² reals.
///
/// # Panics
///
/// As [`p2m_yukawa`], with `local` for `multipole`, and for a source at the centre.
pub fn p2l_yukawa<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    frame: &Frame<T>,
    sources: &[[T; 3]],
    charges: &[T],
    ws: &mut Workspace<T>,
    local: &mut [T],
) {
    let kappa = check_yukawa(level);
    check_frame(level, frame);
    check_coefficients(p, local, 1, "local");
    check_pair(sources.len(), charges.len(), "sources", "charges");
    let buffers = ws.buffers(p);
    let factor = four_pi::<T>() * kappa;
    for (&y, &q) in sources.iter().zip(charges) {
        let rho = offset(frame, y);
        let r = Polar::of(rho).r;
        modified_k(kappa * r, level.scale(), buffers.real, None);
        harmonics(p, rho, buffers.harmonics);
        let e = (level.damping() - kappa * r).exp();
        form(
            p,
            factor * q * e,
            buffers.real,
            buffers.harmonics,
            |i, w| {
                local[i] = local[i] + w;
            },
        );
    }
}

/// Yukawa L2P: adds φ̃ += Σ L̂ᵞₙᵐ îₙ(κ|v|) e^{κ|v| − β} Yₙᵐ(v̂), and optionally its
/// gradient, at each target (module documentation, "Storage"). `local` holds (p + 1)²
/// reals.
///
/// # Panics
///
/// If the level is not Yukawa, `ws.p() < p`, `frame.radius != level.half_width()`,
/// `local.len() != (p + 1)²`, `potential.len() != targets.len()` or a given `gradient`
/// has another length.
#[allow(clippy::too_many_arguments)]
pub fn l2p_yukawa<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    frame: &Frame<T>,
    local: &[T],
    targets: &[[T; 3]],
    ws: &mut Workspace<T>,
    potential: &mut [T],
    gradient: Option<&mut [[T; 3]]>,
) {
    let kappa = check_yukawa(level);
    check_frame(level, frame);
    check_coefficients(p, local, 1, "local");
    check_targets(targets.len(), potential.len(), gradient.as_deref());
    let mut buffers = ws.buffers(p);
    let coefficient = |i: usize| local[i];
    let at_centre = kappa / (T::from_f64(3.0) * level.scale());
    for_each_target(targets, potential, gradient, |request, x| {
        let v = offset(frame, x);
        let polar = Polar::of(v);
        let e = (kappa * polar.r - level.damping()).exp();
        let with_gradient = request.gradient;
        modified_i(
            kappa * polar.r,
            level.scale(),
            buffers.real,
            with_gradient.then_some(&mut *buffers.real_derivative),
        );
        let (phi, grad) = evaluate(
            p,
            v,
            &polar,
            &coefficient,
            &mut buffers.as_real(),
            kappa,
            at_centre,
            with_gradient,
        );
        (phi * e, grad.map(|g| g.map(|c| c * e)))
    });
}

/// Yukawa M2P: adds φ̃ += 4πκ Σ M̂ᵞₙᵐ k̂ₙ(κ|v|) e^{β − κ|v|} Yₙᵐ(v̂), and optionally its
/// gradient, at each target (module documentation, "Storage"). `multipole` holds
/// (p + 1)² reals.
///
/// # Panics
///
/// As [`l2p_yukawa`], with `multipole` for `local`, and for a target at the centre.
#[allow(clippy::too_many_arguments)]
pub fn m2p_yukawa<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    frame: &Frame<T>,
    multipole: &[T],
    targets: &[[T; 3]],
    ws: &mut Workspace<T>,
    potential: &mut [T],
    gradient: Option<&mut [[T; 3]]>,
) {
    let kappa = check_yukawa(level);
    check_frame(level, frame);
    check_coefficients(p, multipole, 1, "multipole");
    check_targets(targets.len(), potential.len(), gradient.as_deref());
    let mut buffers = ws.buffers(p);
    let factor = four_pi::<T>() * kappa;
    let coefficient = |i: usize| multipole[i];
    for_each_target(targets, potential, gradient, |request, x| {
        let v = offset(frame, x);
        let polar = Polar::of(v);
        let e = (level.damping() - kappa * polar.r).exp();
        let with_gradient = request.gradient;
        modified_k(
            kappa * polar.r,
            level.scale(),
            buffers.real,
            with_gradient.then_some(&mut *buffers.real_derivative),
        );
        let (phi, grad) = evaluate(
            p,
            v,
            &polar,
            &coefficient,
            &mut buffers.as_real(),
            kappa,
            T::zero(),
            with_gradient,
        );
        let scale = factor * e;
        (phi * scale, grad.map(|g| g.map(|c| c * scale)))
    });
}

/// The arithmetic the two paths share: a complex number or a real one.
trait Field<T>: Copy + Zero + Add<Output = Self> + Mul<Output = Self> + Mul<T, Output = Self> {}

impl<T: RealScalar> Field<T> for T {}

impl<T: RealScalar> Field<T> for Complex<T> {}

/// The radial values and derivatives of one evaluation, and the angular buffers.
struct Evaluation<'a, T, F> {
    harmonics: &'a mut [T],
    d_theta: &'a mut [T],
    d_phi: &'a mut [T],
    scratch: &'a mut DerivativeScratch<T>,
    values: &'a [F],
    derivatives: &'a [F],
}

impl<T: RealScalar> Buffers<'_, T> {
    fn as_complex(&mut self) -> Evaluation<'_, T, Complex<T>> {
        Evaluation {
            harmonics: self.harmonics,
            d_theta: self.d_theta,
            d_phi: self.d_phi,
            scratch: self.scratch,
            values: self.radial,
            derivatives: self.radial_derivative,
        }
    }

    fn as_real(&mut self) -> Evaluation<'_, T, T> {
        Evaluation {
            harmonics: self.harmonics,
            d_theta: self.d_theta,
            d_phi: self.d_phi,
            scratch: self.scratch,
            values: self.real,
            derivatives: self.real_derivative,
        }
    }
}

/// Σₙ fₙ Σₘ cₙᵐ Yₙᵐ(v̂), and with `with_gradient` its gradient
/// Σₙ Σₘ cₙᵐ (kf fₙ′ Yₙᵐ r̂ + tₙ (∂θ Yₙᵐ θ̂ + (1/sin θ) ∂φ Yₙᵐ φ̂)), with tₙ = fₙ/|v|, or
/// `at_centre` for n = 1 and zero otherwise at |v| = 0. The radial values fₙ and their
/// derivatives are in `buffers`; the exponential and operator factors are the caller's.
#[allow(clippy::too_many_arguments)]
fn evaluate<T: RealScalar, F: Field<T>>(
    p: usize,
    v: [T; 3],
    polar: &Polar<T>,
    coefficient: &impl Fn(usize) -> F,
    buffers: &mut Evaluation<'_, T, F>,
    kf: F,
    at_centre: F,
    with_gradient: bool,
) -> (F, Option<[F; 3]>) {
    let mut phi = F::zero();
    if !with_gradient {
        harmonics(p, v, buffers.harmonics);
        for n in 0..=p {
            let mut sum = F::zero();
            for i in n * n..(n + 1) * (n + 1) {
                sum = sum + coefficient(i) * buffers.harmonics[i];
            }
            phi = phi + buffers.values[n] * sum;
        }
        return (phi, None);
    }
    harmonics_derivatives(
        p,
        v,
        buffers.harmonics,
        buffers.d_theta,
        buffers.d_phi,
        buffers.scratch,
    );
    let (mut radial, mut polar_part, mut azimuthal) = (F::zero(), F::zero(), F::zero());
    for n in 0..=p {
        let (mut value, mut theta, mut phi_sum) = (F::zero(), F::zero(), F::zero());
        for i in n * n..(n + 1) * (n + 1) {
            let c = coefficient(i);
            value = value + c * buffers.harmonics[i];
            theta = theta + c * buffers.d_theta[i];
            phi_sum = phi_sum + c * buffers.d_phi[i];
        }
        let tangential = if polar.r > T::zero() {
            buffers.values[n] * (T::one() / polar.r)
        } else if n == 1 {
            at_centre
        } else {
            F::zero()
        };
        phi = phi + buffers.values[n] * value;
        radial = radial + buffers.derivatives[n] * value;
        polar_part = polar_part + tangential * theta;
        azimuthal = azimuthal + tangential * phi_sum;
    }
    (
        phi,
        Some(polar.cartesian(kf * radial, polar_part, azimuthal)),
    )
}

/// Adds w fₙ Yₙᵐ for every slot through `add(i, value)`, degree by degree.
fn form<T: RealScalar, F: Field<T>>(
    p: usize,
    w: F,
    values: &[F],
    harmonics: &[T],
    mut add: impl FnMut(usize, F),
) {
    for (n, &value) in values[..=p].iter().enumerate() {
        let wn = w * value;
        let slots = n * n..(n + 1) * (n + 1);
        for (i, &y) in slots.clone().zip(&harmonics[slots]) {
            add(i, wn * y);
        }
    }
}

/// What [`for_each_target`] asks of its evaluation.
#[derive(Clone, Copy)]
struct Request {
    gradient: bool,
}

/// Evaluates each target through `value(request, x) -> (φ̃, ∇φ̃)` and adds the result into
/// `potential` and, if given, `gradient`.
fn for_each_target<T: RealScalar, F: Field<T>>(
    targets: &[[T; 3]],
    potential: &mut [F],
    gradient: Option<&mut [[F; 3]]>,
    mut value: impl FnMut(Request, [T; 3]) -> (F, Option<[F; 3]>),
) {
    match gradient {
        None => {
            for (&x, phi) in targets.iter().zip(potential.iter_mut()) {
                let (v, _) = value(Request { gradient: false }, x);
                *phi = *phi + v;
            }
        }
        Some(gradient) => {
            for ((&x, phi), g) in targets.iter().zip(potential.iter_mut()).zip(gradient) {
                let (v, dv) = value(Request { gradient: true }, x);
                *phi = *phi + v;
                if let Some(dv) = dv {
                    for (gk, dk) in g.iter_mut().zip(dv) {
                        *gk = *gk + dk;
                    }
                }
            }
        }
    }
}

/// |v| and the angles of v in the frame of
/// [`harmonics_derivatives`]: θ = 0 at v = 0, φ = 0 on the z-axis.
struct Polar<T> {
    r: T,
    cos_theta: T,
    sin_theta: T,
    cos_phi: T,
    sin_phi: T,
}

impl<T: RealScalar> Polar<T> {
    fn of([x, y, z]: [T; 3]) -> Self {
        let rho = x.hypot(y);
        let r = rho.hypot(z);
        if r == T::zero() {
            return Self {
                r,
                cos_theta: T::one(),
                sin_theta: T::zero(),
                cos_phi: T::one(),
                sin_phi: T::zero(),
            };
        }
        let (cos_phi, sin_phi) = if rho > T::zero() {
            (x / rho, y / rho)
        } else {
            (T::one(), T::zero())
        };
        Self {
            r,
            cos_theta: z / r,
            sin_theta: rho / r,
            cos_phi,
            sin_phi,
        }
    }

    /// a_r r̂ + a_θ θ̂ + a_φ φ̂ in Cartesian components.
    fn cartesian<F: Field<T>>(&self, radial: F, polar: F, azimuthal: F) -> [F; 3] {
        let (ct, st, cp, sp) = (self.cos_theta, self.sin_theta, self.cos_phi, self.sin_phi);
        [
            radial * (st * cp) + polar * (ct * cp) + azimuthal * (-sp),
            radial * (st * sp) + polar * (ct * sp) + azimuthal * cp,
            radial * ct + polar * (-st),
        ]
    }
}

/// x − c.
pub(super) fn offset<T: RealScalar>(frame: &Frame<T>, x: [T; 3]) -> [T; 3] {
    core::array::from_fn(|i| x[i] - frame.centre[i])
}

pub(super) fn four_pi<T: RealScalar>() -> T {
    T::from_f64(4.0) * T::PI()
}

/// 4π ik = 4π (−Im k + i Re k).
pub(super) fn four_pi_ik<T: RealScalar>(k: Complex<T>) -> Complex<T> {
    let c = four_pi::<T>();
    Complex::new(-(c * k.im), c * k.re)
}

pub(super) fn check_frame<T: RealScalar>(level: &Level<T>, frame: &Frame<T>) {
    assert!(
        frame.radius == level.half_width(),
        "the frame's radius {} is not the level's half-width {}",
        frame.radius.to_f64(),
        level.half_width().to_f64()
    );
}

/// κ = Im k of a Yukawa level.
pub(super) fn check_yukawa<T: RealScalar>(level: &Level<T>) -> T {
    assert!(
        level.is_yukawa(),
        "Yukawa's real path needs k = iκ (Re k = 0)"
    );
    level.k().im
}

pub(super) fn check_coefficients<T>(p: usize, coefficients: &[T], planes: usize, name: &str) {
    let expected = planes * Layout::new(p).len();
    assert_eq!(
        coefficients.len(),
        expected,
        "{name} has length {}, expected {expected} for p = {p}",
        coefficients.len()
    );
}

fn check_targets<F>(targets: usize, potential: usize, gradient: Option<&[[F; 3]]>) {
    check_pair(targets, potential, "targets", "potential");
    if let Some(gradient) = gradient {
        check_pair(targets, gradient.len(), "targets", "gradient");
    }
}

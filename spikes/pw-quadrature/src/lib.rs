//! The quadratures of the Helmholtz exponential (plane-wave) M2L per k w_l, a spike
//! (Helmholtz CH0.4, S4; proposal §5.3, §8 R7).
//!
//! For a target above a source by u = (x, y, z) in box widths, with 1 ≤ z ≤ 4 and
//! |x|, |y| ≤ 4 (a V pair whose offset along its direction is 2 or 3 boxes, as for
//! Laplace in CONVENTIONS §3.14), and κ = k w the wavenumber in box widths,
//!
//! e^{iκ|u|}/|u| = ∫_Γ e^{−λ z} J₀(t ρ) dλ,  t = √(λ² + κ²),  ρ = √(x² + y²),
//!
//! along a contour Γ from λ = −iκ (t = 0) to +∞ (the Sommerfeld integral in the decay
//! rate λ; the integrand is entire in λ, so Γ is free up to growth). With
//! J₀(t ρ) = (1/2π) ∫ e^{i t (x cos α + y sin α)} dα, a quadrature is
//!
//! e^{iκ|u|}/|u| ≈ Σₖ (wₖ / Nₖ) Σ_{j<Nₖ} e^{−λₖ z + i tₖ (x cos αⱼ + y sin αⱼ)},  αⱼ = 2πj/Nₖ,
//!
//! with complex nodes λₖ, weights wₖ and α counts Nₖ, plus Fourier counts Fₖ (the
//! orders |m| ≤ Fₖ of a source's waves kept at λₖ).
//!
//! - [`fmm3d`]: FMM3D's tables (`src/Helmholtz/hwts3e.f` and its data files), parsed;
//! - [`quadrature`]: a quadrature and its evaluation, the kernel and the
//!   point-to-point chain;
//! - [`region`]: the sample points and the error norms;
//! - [`generator`]: an independent generator, for any κ;
//! - [`bessel`] and [`gauss`]: Jₙ of real argument and Gauss–Legendre rules.
//!
//! SPIKE_REPORT.md has the findings.

pub mod bessel;
pub mod fmm3d;
pub mod gauss;
pub mod generator;
pub mod quadrature;
pub mod region;

/// The complex scalar of the spike.
pub type C = num_complex::Complex64;

/// The kernel e^{iκ|u|}/|u| in box widths: κ = k w, u the offset in box widths.
pub fn kernel(kappa: C, u: [f64; 3]) -> C {
    let r = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    (C::i() * kappa * r).exp() / r
}

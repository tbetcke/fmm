//! A generator of the quadratures of the Helmholtz exponential (plane-wave) M2L, for any
//! κ = k w with Re κ, Im κ ≥ 0 (Helmholtz CH6.1b; S4's generator,
//! `spikes/pw-quadrature/src/generator.rs` and SPIKE_REPORT.md §4): the trusted path that
//! the exponential M2L of `nd_fmm_tables::helmholtz::plane_wave`, built on FMM3D's tables,
//! is tested against with a second, independently made quadrature.
//!
//! # The representation
//!
//! For a target above a source by u = (x, y, z) in box widths, 1 ≤ z ≤ 4 and |x|, |y| ≤ 4
//! (a V pair whose offset along its direction is 2 or 3 boxes, CONVENTIONS §3.14), and
//! κ = k w the wavenumber in box widths,
//!
//! e^{iκ|u|}/|u| = ∫_Γ e^{−λ z} J₀(t ρ) dλ,  t = √(λ² + κ²),  ρ = √(x² + y²),
//!
//! on a contour Γ from λ = −iκ (t = 0) to +∞, and with the trapezoidal rule in the angle α
//! of J₀(t ρ) = (1/2π) ∫ e^{i t (x cos α + y sin α)} dα a quadrature is
//!
//! e^{iκ|u|}/|u| ≈ Σₖ (wₖ / Nₖ) Σ_{j<Nₖ} e^{−λₖ z + i tₖ (x cos αⱼ + y sin αⱼ)},  αⱼ = 2πj/Nₖ,
//!
//! with nodes λₖ, weights wₖ, α counts Nₖ, and Fourier counts Fₖ: a source's waves at λₖ
//! keep the orders |m| ≤ Fₖ.
//!
//! # The generator
//!
//! [`generate`] runs S4's four steps, in the spike's order of operations, so that its
//! values are the spike's bit for bit:
//!
//! 1. **The contour of real t.** λ = −i q with q = √(κ² − t²), Im q ≥ 0 (the outgoing
//!    branch), and dλ = t dt / λ, for t from 0 to t_max, where Re λ reaches
//!    ln(1/τ) + 4 for the λ rule's target τ ≤ ε/4 (the tail beyond is below τ/50 at
//!    z ≥ 1). On it J₀(t ρ) has a real
//!    argument, so the waves never grow and |e^{−λ z}| ≤ 1. For Re κ > 0 the variable is
//!    θ on [0, π/2] (t = Re κ sin θ) and φ on [0, φ_max] (t = Re κ cosh φ), which removes
//!    the 1/√ singularity of t/λ at t = κ for real κ; for 0 < Im κ ≪ Re κ the panels are
//!    graded geometrically towards θ = π/2 and φ = 0, where the near-singularity sits at
//!    a width √(Im κ / Re κ). For Re κ = 0 the variable is t itself, graded towards 0 at a
//!    width Im κ.
//! 2. **Panels.** Composite Gauss–Legendre of [`ORDER`] points, the panel count doubled
//!    (to at most [`MAX_PANELS`] per segment) until the λ sum meets a quarter of its
//!    target against the kernel at 25 × 65 samples of (z, ρ) on the region and at the
//!    24 × 64 midpoints between them.
//! 3. **Compression.** A column-pivoted Gram–Schmidt (orthogonalised twice) of the matrix
//!    of the integrand at the samples selects nodes until the residual is below a
//!    tolerance; the weights of the selected nodes are refitted by least squares to the
//!    full rule's values (the first step of a generalised Gaussian quadrature, without
//!    its Gauss–Newton elimination). The tolerance is divided by 4 until the target holds
//!    again on both sample sets, down to 10⁻⁴ of the target.
//! 4. **α and Fourier counts.** With the node's size cₖ = 2 |wₖ| e^{−Re λₖ} and the share
//!    ε/(4s) per node (s the nodes): Fₖ the fewest orders with
//!    cₖ max_{m > Fₖ, x ≤ tₖ r_s} |J_m(x)| ≤ ε/(4s), r_s = √2/2 ([`SOURCE_REACH`]) the
//!    lateral reach of a source from its box's centre; Nₖ the fewest with the same bound
//!    on m ≥ Nₖ for x ≤ tₖ ρ_max, ρ_max = 4√2 ([`RHO_MAX`]), and at least the count for
//!    ρ_max − r_s plus Fₖ, so that the chain's truncated orders do not alias.
//!
//! The result is checked against the kernel on the region's grid, z ∈ {1, 1.25, …, 4},
//! x, y ∈ {−4, −3.5, …, 4} ([`Quadrature::kernel_error`]). If it misses ε, the λ rule's
//! target is divided by 4 when the λ sum alone misses ε/2 (from ε/4 down to ε/1024), and otherwise
//! the counts' share is halved (down to 1/1024 of its start). ε below [`FLOOR`] is raised
//! to it.
//!
//! # Where it stops short
//!
//! The spike returns its best quadrature at these bounds without saying so (S4 §4). Here
//! [`Quadrature::shortfall`] says so: it is `None` when the quadrature meets the requested
//! ε on the grid, and otherwise names the error reached and the bounds hit ([`Bound`]).
//! A caller that relies on ε checks it.
//!
//! The errors are absolute, in box widths (FMM3D's norm, S4 §1, as CONVENTIONS §3.14's
//! Laplace tables). Everything is f64 and a function of κ and ε only (proposal §2,
//! constraint 9). The generator is slow beside a table lookup (its tests' 35 κ and ε take
//! 14 s together in release on the M6), as a trusted path may be.
//!
//! # Matched to a degree
//!
//! [`generate_for_degree`] (Helmholtz CH6.1f) fits the λ rule to the chain at a degree p
//! instead of the kernel, so that the chain reproduces RCR at degree p where a quadrature
//! fitted to the kernel errs on the truncated multipole's tail (its module
//! documentation).

mod degree;
mod rule;
mod special;

pub use degree::generate_for_degree;

use core::f64::consts::TAU;

use nd_fmm_math::Complex;

use rule::{lambda_rule, tail_count};

/// The complex scalar of the generator.
type C = Complex<f64>;

/// Gauss–Legendre points per panel.
pub const ORDER: usize = 20;

/// The smallest ε the generator aims for: f64 round-off of sums of 10³ terms of size 1.
pub const FLOOR: f64 = 1e-13;

/// The most panels per segment before the λ rule stops refining.
pub const MAX_PANELS: usize = 64;

/// ρ_max = 4√2, the region's largest lateral offset.
pub const RHO_MAX: f64 = 4.0 * core::f64::consts::SQRT_2;

/// r_s = √2/2, the largest lateral offset of a point from its box's centre.
pub const SOURCE_REACH: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// The kernel e^{iκ|u|}/|u| in box widths: κ = k w, u the offset in box widths.
pub fn kernel(kappa: Complex<f64>, u: [f64; 3]) -> Complex<f64> {
    let r = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    (C::i() * kappa * r).exp() / r
}

/// A bound at which [`generate`] stopped refining.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// ε was below [`FLOOR`] and was raised to it.
    Floor,
    /// The λ rule reached [`MAX_PANELS`] per segment with its error above its target.
    Panels,
    /// The compression reached its tolerance floor with its error above its target.
    Compression,
    /// The λ rule's target or the counts' share reached its last tightening.
    Tightening,
}

/// How a generated quadrature misses the requested ε: on the region's grid for
/// [`generate`], on its samples against RCR at its degree for [`generate_for_degree`].
#[derive(Clone, Debug, PartialEq)]
pub struct Shortfall {
    /// The largest error reached, in box widths, above the requested ε: |Q − G| on the
    /// grid ([`generate`]), or |chain − RCR| per unit charge on the samples
    /// ([`generate_for_degree`]).
    pub error: f64,
    /// The bounds hit by the final attempt, in the order of [`Bound`]'s variants.
    pub bounds: Vec<Bound>,
}

/// A generated quadrature at one κ = k w (the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct Quadrature {
    kappa: C,
    eps: f64,
    lambda: Vec<C>,
    t: Vec<C>,
    weight: Vec<C>,
    alpha: Vec<usize>,
    fourier: Vec<usize>,
    shortfall: Option<Shortfall>,
    degree: Option<usize>,
}

/// The region's grid: z ∈ {1, 1.25, …, 4}, x, y ∈ {−4, −3.5, …, 4}.
struct Grid {
    xy: Vec<f64>,
    z: Vec<f64>,
}

impl Grid {
    fn region() -> Self {
        Self {
            xy: (0..17).map(|i| -4.0 + 0.5 * i as f64).collect(),
            z: (0..13).map(|i| 1.0 + 0.25 * i as f64).collect(),
        }
    }

    /// The points, z slowest, then y, then x.
    fn points(&self) -> impl Iterator<Item = [f64; 3]> + '_ {
        self.z.iter().flat_map(move |&z| {
            self.xy
                .iter()
                .flat_map(move |&y| self.xy.iter().map(move |&x| [x, y, z]))
        })
    }
}

impl Quadrature {
    /// A quadrature from its nodes, weights and counts; tₖ = √(λₖ² + κ²).
    fn new(
        kappa: C,
        lambda: Vec<C>,
        weight: Vec<C>,
        alpha: Vec<usize>,
        fourier: Vec<usize>,
    ) -> Self {
        let t = lambda
            .iter()
            .map(|&l| (l * l + kappa * kappa).sqrt())
            .collect();
        Self {
            kappa,
            eps: 0.0,
            lambda,
            t,
            weight,
            alpha,
            fourier,
            shortfall: None,
            degree: None,
        }
    }

    /// κ = k w.
    pub fn kappa(&self) -> Complex<f64> {
        self.kappa
    }

    /// The requested ε.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// The degree p the quadrature was matched to ([`generate_for_degree`]), `None` for
    /// [`generate`]'s, which are fitted to the kernel.
    pub fn degree(&self) -> Option<usize> {
        self.degree
    }

    /// The number of nodes, s.
    pub fn nodes(&self) -> usize {
        self.lambda.len()
    }

    /// The decay rates λₖ.
    pub fn lambda(&self) -> &[Complex<f64>] {
        &self.lambda
    }

    /// The transverse wavenumbers tₖ = √(λₖ² + κ²), formed from λₖ (so real to rounding:
    /// the nodes lie on the contour of real t).
    pub fn t(&self) -> &[Complex<f64>] {
        &self.t
    }

    /// The weights wₖ (for dλ).
    pub fn weights(&self) -> &[Complex<f64>] {
        &self.weight
    }

    /// The α counts Nₖ.
    pub fn alpha_counts(&self) -> &[usize] {
        &self.alpha
    }

    /// The Fourier counts Fₖ: a source's waves at λₖ keep the orders |m| ≤ Fₖ.
    pub fn fourier_counts(&self) -> &[usize] {
        &self.fourier
    }

    /// The complex wave values per direction, Σₖ Nₖ.
    pub fn wave_count(&self) -> usize {
        self.alpha.iter().sum()
    }

    /// The Fourier coefficients per direction, Σₖ (2Fₖ + 1).
    pub fn mode_count(&self) -> usize {
        self.fourier.iter().map(|&f| 2 * f + 1).sum()
    }

    /// `None` when the quadrature meets the requested ε (on the region's grid, or for a
    /// degree-matched one against RCR on its samples); otherwise the error reached and the
    /// bounds that stopped the generator.
    pub fn shortfall(&self) -> Option<&Shortfall> {
        self.shortfall.as_ref()
    }

    /// The quadrature with its α counts at one offset u (box widths), term by term.
    pub fn evaluate(&self, u: [f64; 3]) -> Complex<f64> {
        (0..self.nodes())
            .map(|k| {
                let n = self.alpha[k];
                (0..n)
                    .map(|j| {
                        let (sa, ca) = (TAU * j as f64 / n as f64).sin_cos();
                        (-self.lambda[k] * u[2] + C::i() * self.t[k] * (u[0] * ca + u[1] * sa))
                            .exp()
                    })
                    .sum::<C>()
                    * self.weight[k]
                    / n as f64
            })
            .sum()
    }

    /// The largest |Q − G| over the region's grid (the [module documentation](self)), Q
    /// the quadrature with its α counts and G the kernel, in box widths.
    pub fn kernel_error(&self) -> f64 {
        self.kernel_error_with(&Grid::region(), &self.alpha)
    }

    /// The largest |Q − G| over `grid` with the α counts `counts`, by sums over the grid's
    /// separable factors.
    fn kernel_error_with(&self, grid: &Grid, counts: &[usize]) -> f64 {
        let (nx, nz) = (grid.xy.len(), grid.z.len());
        let mut sum = vec![C::new(0.0, 0.0); nz * nx * nx];
        let mut ex = vec![C::new(0.0, 0.0); nx];
        let mut ey = vec![C::new(0.0, 0.0); nx];
        for (k, &n) in counts.iter().enumerate().take(self.nodes()) {
            let c = self.weight[k] / n as f64;
            let ez: Vec<C> = grid
                .z
                .iter()
                .map(|&z| (-self.lambda[k] * z).exp())
                .collect();
            let it = C::i() * self.t[k];
            for j in 0..n {
                let (sa, ca) = (TAU * j as f64 / n as f64).sin_cos();
                for (i, &x) in grid.xy.iter().enumerate() {
                    ex[i] = (it * (x * ca)).exp();
                    ey[i] = (it * (x * sa)).exp();
                }
                for (iz, &z) in ez.iter().enumerate() {
                    let cz = c * z;
                    for iy in 0..nx {
                        let czy = cz * ey[iy];
                        let row = &mut sum[(iz * nx + iy) * nx..][..nx];
                        for (s, &e) in row.iter_mut().zip(&ex) {
                            *s += czy * e;
                        }
                    }
                }
            }
        }
        sum.iter()
            .zip(grid.points())
            .map(|(q, u)| (q - kernel(self.kappa, u)).norm())
            .fold(0.0, f64::max)
    }

    /// The λ sum alone (α resolved to round-off: Nₖ ≥ |tₖ| ρ + 60 for the grid's largest
    /// lateral offset ρ) against the kernel on `grid`.
    fn lambda_error(&self, grid: &Grid) -> f64 {
        let rho = grid.xy.iter().fold(0.0f64, |m, x| m.max(x.abs())) * 2f64.sqrt();
        let counts: Vec<usize> = self
            .t
            .iter()
            .map(|t| (t.norm() * rho).ceil() as usize + 60)
            .collect();
        self.kernel_error_with(grid, &counts)
    }
}

/// The generated quadrature for κ at ε (the [module documentation](self)), with its α
/// and Fourier counts and, if it misses ε on the region's grid, its shortfall.
///
/// # Panics
///
/// If Re κ or Im κ is negative or not finite, or ε is not positive and finite.
pub fn generate(kappa: Complex<f64>, eps: f64) -> Quadrature {
    generate_with(kappa, eps, MAX_PANELS)
}

/// [`generate`] with the λ rule's panel cap as a parameter (tests force the bound).
fn generate_with(kappa: C, requested: f64, max_panels: usize) -> Quadrature {
    assert!(
        kappa.re.is_finite() && kappa.im.is_finite() && kappa.re >= 0.0 && kappa.im >= 0.0,
        "κ = {kappa} outside the closed first quadrant"
    );
    assert!(
        requested.is_finite() && requested > 0.0,
        "ε = {requested:e} is not positive and finite"
    );
    let eps = requested.max(FLOOR);
    let grid = Grid::region();
    let mut target = eps / 4.0;
    'rule: loop {
        let fitted = lambda_rule(kappa, eps, target, max_panels);
        let rule = &fitted.rule;
        let s = rule.t.len();
        let mut share = eps / (4.0 * s as f64);
        loop {
            let mut physical = Vec::with_capacity(s);
            let mut fourier = Vec::with_capacity(s);
            for k in 0..s {
                let scale = 2.0 * rule.weight[k].norm() * (-rule.lambda[k].re).exp();
                let t = rule.t[k];
                let f = tail_count(t * SOURCE_REACH, scale, share) - 1;
                let n = tail_count(t * RHO_MAX, scale, share)
                    .max(tail_count(t * (RHO_MAX - SOURCE_REACH), scale, share) + f);
                physical.push(n);
                fourier.push(f);
            }
            let mut q = Quadrature::new(
                kappa,
                rule.lambda.clone(),
                rule.weight.clone(),
                physical,
                fourier,
            );
            let error = q.kernel_error_with(&grid, &q.alpha);
            let gave_up = if error <= eps {
                false
            } else if target < eps / 256.0 {
                true
            } else if q.lambda_error(&grid) > eps / 2.0 {
                target /= 4.0;
                continue 'rule;
            } else {
                share /= 2.0;
                if share < eps / (4096.0 * s as f64) {
                    true
                } else {
                    continue;
                }
            };
            q.eps = requested;
            if error > requested {
                let mut bounds = Vec::new();
                if requested < FLOOR {
                    bounds.push(Bound::Floor);
                }
                if fitted.panels_capped {
                    bounds.push(Bound::Panels);
                }
                if fitted.compression_capped {
                    bounds.push(Bound::Compression);
                }
                if gave_up {
                    bounds.push(Bound::Tightening);
                }
                q.shortfall = Some(Shortfall { error, bounds });
            }
            return q;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bounds are reported. One panel per segment cannot meet ε at κ = 2, and
    /// ε = 10⁻¹⁵ lies below [`FLOOR`], which is named too: the result has a shortfall whose
    /// error is its largest absolute |Q − G| on the grid, above ε, and whose bounds hold
    /// the floor, the panels and the last tightening. (One panel keeps the test fast: the
    /// generator near 10⁻¹³ takes minutes.)
    #[test]
    fn a_forced_bound_is_reported() {
        let q = generate_with(C::new(2.0, 0.0), 1e-15, 1);
        assert_eq!(q.eps(), 1e-15);
        let shortfall = q.shortfall().expect("one panel per segment misses ε");
        assert!(shortfall.error > 1e-15);
        assert_eq!(shortfall.error, q.kernel_error());
        for bound in [Bound::Floor, Bound::Panels, Bound::Tightening] {
            assert!(shortfall.bounds.contains(&bound), "{shortfall:?}");
        }
    }

    /// A quadrature that meets ε, the largest absolute |Q − G| on the grid in box widths,
    /// has no shortfall; its wave and mode counts are the sums of its counts.
    #[test]
    fn no_shortfall_when_eps_is_met() {
        let q = generate(C::new(5.0, 0.0), 1e-3);
        assert!(q.shortfall().is_none());
        assert!(q.kernel_error() <= 1e-3);
        assert_eq!(q.wave_count(), q.alpha_counts().iter().sum::<usize>());
        assert_eq!(
            q.mode_count(),
            q.fourier_counts().iter().map(|f| 2 * f + 1).sum::<usize>()
        );
    }
}

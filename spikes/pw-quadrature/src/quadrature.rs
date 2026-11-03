//! A plane-wave quadrature and its evaluation (the crate documentation's formula).

use crate::fmm3d::Domain;
use crate::region::{Accumulator, Errors, Grid, Pairs};
use crate::{C, kernel};
use core::f64::consts::TAU;

/// A quadrature at one κ = k w: per node the decay rate λₖ, the transverse wavenumber
/// tₖ = √(λₖ² + κ²), the weight wₖ, the α count Nₖ and the Fourier count Fₖ.
#[derive(Clone, Debug)]
pub struct Quadrature {
    /// κ = k w.
    pub kappa: C,
    /// λₖ.
    pub lambda: Vec<C>,
    /// tₖ.
    pub t: Vec<C>,
    /// wₖ.
    pub weight: Vec<C>,
    /// Nₖ.
    pub physical: Vec<usize>,
    /// Fₖ: orders |m| ≤ Fₖ.
    pub fourier: Vec<usize>,
}

impl Quadrature {
    /// A quadrature from its nodes, weights and counts; tₖ = √(λₖ² + κ²).
    pub fn new(
        kappa: C,
        lambda: Vec<C>,
        weight: Vec<C>,
        physical: Vec<usize>,
        fourier: Vec<usize>,
    ) -> Self {
        let t = lambda
            .iter()
            .map(|&l| (l * l + kappa * kappa).sqrt())
            .collect();
        Self {
            kappa,
            lambda,
            t,
            weight,
            physical,
            fourier,
        }
    }

    /// FMM3D's quadrature of a domain, at κ (`hwts3e`, `hnumfour`, `hnumphys`).
    pub fn from_fmm3d(domain: &Domain, kappa: C) -> Self {
        let (lambda, weight) = domain.nodes(kappa);
        Self::new(
            kappa,
            lambda,
            weight,
            domain.physical.clone(),
            domain.fourier.clone(),
        )
    }

    /// The number of nodes, s.
    pub fn nodes(&self) -> usize {
        self.lambda.len()
    }

    /// The complex wave values per direction, Σₖ Nₖ (FMM3D's `nexptotp`).
    pub fn physical_terms(&self) -> usize {
        self.physical.iter().sum()
    }

    /// The Fourier coefficients per direction, Σₖ (2Fₖ + 1) (FMM3D's `nexptot`).
    pub fn fourier_terms(&self) -> usize {
        self.fourier.iter().map(|&f| 2 * f + 1).sum()
    }

    /// α counts that resolve J₀(tₖ ρ) to round-off for ρ ≤ ρ_max: Nₖ ≥ |tₖ| ρ_max + 60.
    pub fn converged_alpha(&self, rho_max: f64) -> Vec<usize> {
        self.t
            .iter()
            .map(|t| (t.norm() * rho_max).ceil() as usize + 60)
            .collect()
    }

    /// The quadrature with the α counts `counts` at one offset u, term by term (the
    /// slow path the grid sums are tested against).
    pub fn evaluate(&self, u: [f64; 3], counts: &[usize]) -> C {
        (0..self.nodes())
            .map(|k| {
                let n = counts[k];
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

    /// The quadrature with the α counts `counts` against the kernel on the grid.
    pub fn kernel_errors_with(&self, grid: &Grid, counts: &[usize]) -> Errors {
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
        let mut acc = Accumulator::default();
        for (q, u) in sum.iter().zip(grid.points()) {
            acc.add(*q, kernel(self.kappa, u));
        }
        acc.errors()
    }

    /// The quadrature as used (its α counts) against the kernel on the grid.
    pub fn kernel_errors(&self, grid: &Grid) -> Errors {
        self.kernel_errors_with(grid, &self.physical)
    }

    /// The λ quadrature alone (α resolved to round-off) against the kernel on the grid.
    pub fn lambda_errors(&self, grid: &Grid) -> Errors {
        let rho = grid.xy.iter().fold(0.0f64, |m, x| m.max(x.abs())) * 2f64.sqrt();
        self.kernel_errors_with(grid, &self.converged_alpha(rho))
    }

    /// The point-to-point chain against the kernel: each source's waves at λₖ, truncated
    /// to the orders |m| ≤ Fₖ (`truncate`) or kept whole, sampled at the Nₖ α nodes and
    /// evaluated at every target (in box widths from the source box's centre).
    pub fn chain_errors(&self, pairs: &Pairs, truncate: bool) -> Errors {
        let mut acc = Accumulator::default();
        for &s in &pairs.sources {
            let sums = self.chain_sums(s, pairs, truncate);
            let mut i = 0;
            for &z in &pairs.target_z {
                for &y in &pairs.target_xy {
                    for &x in &pairs.target_xy {
                        let u = [x - s[0], y - s[1], z - s[2]];
                        acc.add(sums[i], kernel(self.kappa, u));
                        i += 1;
                    }
                }
            }
        }
        acc.errors()
    }

    /// The chain from one source s to every target of `pairs` (z slowest, then y, then
    /// x): the waves of s per node, e^{λₖ s_z − i tₖ (s_x cos α + s_y sin α)}, kept whole
    /// at the Nₖ α nodes or truncated to the orders |m| ≤ Fₖ (their coefficients from a
    /// DFT fine enough to be exact to round-off), then the targets' factors.
    pub fn chain_sums(&self, s: [f64; 3], pairs: &Pairs, truncate: bool) -> Vec<C> {
        let (nx, nz) = (pairs.target_xy.len(), pairs.target_z.len());
        let mut sum = vec![C::new(0.0, 0.0); nz * nx * nx];
        let (mut ex, mut ey) = (vec![C::new(0.0, 0.0); nx], vec![C::new(0.0, 0.0); nx]);
        for k in 0..self.nodes() {
            let n = self.physical[k];
            let c = self.weight[k] / n as f64;
            let (lambda, t) = (self.lambda[k], self.t[k]);
            let wave = |alpha: f64| {
                let (sa, ca) = alpha.sin_cos();
                (lambda * s[2] - C::i() * t * (s[0] * ca + s[1] * sa)).exp()
            };
            let values: Vec<C> = if truncate {
                let f = self.fourier[k] as i64;
                let m = 2 * (f as usize + (t.norm() * 0.75).ceil() as usize + 32);
                let samples: Vec<C> = (0..m).map(|l| wave(TAU * l as f64 / m as f64)).collect();
                let coeffs: Vec<C> = (-f..=f)
                    .map(|order| {
                        samples
                            .iter()
                            .enumerate()
                            .map(|(l, v)| {
                                v * C::from_polar(1.0, -TAU * (order * l as i64) as f64 / m as f64)
                            })
                            .sum::<C>()
                            / m as f64
                    })
                    .collect();
                (0..n)
                    .map(|j| {
                        let a = TAU * j as f64 / n as f64;
                        (-f..=f)
                            .zip(&coeffs)
                            .map(|(order, c)| c * C::from_polar(1.0, order as f64 * a))
                            .sum()
                    })
                    .collect()
            } else {
                (0..n).map(|j| wave(TAU * j as f64 / n as f64)).collect()
            };
            let ez: Vec<C> = pairs
                .target_z
                .iter()
                .map(|&z| (-lambda * z).exp())
                .collect();
            let it = C::i() * t;
            for (j, v) in values.iter().enumerate() {
                let (sa, ca) = (TAU * j as f64 / n as f64).sin_cos();
                for (i, &x) in pairs.target_xy.iter().enumerate() {
                    ex[i] = (it * (x * ca)).exp();
                    ey[i] = (it * (x * sa)).exp();
                }
                let cv = c * v;
                for (iz, &z) in ez.iter().enumerate() {
                    let cz = cv * z;
                    for iy in 0..nx {
                        let czy = cz * ey[iy];
                        let row = &mut sum[(iz * nx + iy) * nx..][..nx];
                        for (q, &e) in row.iter_mut().zip(&ex) {
                            *q += czy * e;
                        }
                    }
                }
            }
        }
        sum
    }
}

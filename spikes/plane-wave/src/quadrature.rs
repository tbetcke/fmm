//! The plane-wave quadratures: FMM3D's tables for separation 1 (`quadrature_data`), and
//! the choice of a quadrature for a degree p.
//!
//! For a target above a source by z in [1, 4] box widths and laterally within
//! ρ ≤ 4√2 (a V-list pair whose offset along the direction is 2 or 3 boxes),
//!
//! 1/|u| = (1/2π) ∫₀^∞ e^{−λ z} ∫₀^{2π} e^{iλ(x cos α + y sin α)} dα dλ
//!       ≈ Σₖ (wₖ / Nₖ) Σⱼ e^{−λₖ z + iλₖ(x cos αⱼ + y sin αⱼ)},  αⱼ = 2π j / Nₖ,
//!
//! with s nodes λₖ and weights wₖ (a generalised Gaussian quadrature) and Nₖ trapezoidal
//! nodes in α for λₖ (FMM3D's `numthetafour`). The multipole's orders above Fₖ − 1
//! (`numthetahalf`) are dropped for λₖ. FMM3D states a maximum error of the s-node
//! quadrature over that region (box width 1); with its α nodes the error measured over
//! the region is 6–23× larger, largest at z = 1 and ρ = 4√2
//! ([`Quadrature::measured_error`]; the λ sum alone, with many α nodes, meets the
//! stated error). Choices here go by the measured error.

use crate::quadrature_data::TABLES;

/// The smallest and largest numbers of λ nodes in the tables.
pub const NODES: core::ops::RangeInclusive<usize> = 2..=39;

/// One quadrature of FMM3D for separation 1 (box width 1).
#[derive(Clone, Debug)]
pub struct Quadrature {
    /// The λ nodes.
    pub lambdas: Vec<f64>,
    /// The λ weights.
    pub weights: Vec<f64>,
    /// The α nodes per λ as FMM3D uses them, Nₖ.
    pub alpha_nodes: Vec<usize>,
    /// The Fourier modes per λ, Fₖ: orders 0 ≤ m < Fₖ are kept.
    pub fourier: Vec<usize>,
    /// The maximum error FMM3D states for the quadrature (absolute, for 1/|u| in box
    /// widths over the region of the module documentation).
    pub max_error: f64,
}

impl Quadrature {
    /// The quadrature with `s` λ nodes.
    ///
    /// # Panics
    ///
    /// If `s` is outside [`NODES`].
    pub fn with_nodes(s: usize) -> Self {
        assert!(NODES.contains(&s), "FMM3D's tables have 2 to 39 λ nodes");
        let (lambdas, weights, alpha_nodes, fourier, max_error) = TABLES[s - 2];
        Self {
            lambdas: lambdas.to_vec(),
            weights: weights.to_vec(),
            alpha_nodes: alpha_nodes.to_vec(),
            fourier: fourier.to_vec(),
            max_error,
        }
    }

    /// The quadrature with the fewest λ nodes whose measured error
    /// ([`measured_error`](Self::measured_error)) is at most `error`.
    ///
    /// # Panics
    ///
    /// If no table reaches `error`.
    pub fn for_error(error: f64) -> Self {
        NODES
            .clone()
            .map(Self::with_nodes)
            .find(|q| q.measured_error() <= error)
            .expect("no FMM3D quadrature reaches the error")
    }

    /// The largest error of the quadrature as used here (the even α counts) against
    /// 1/|u| on a grid of the region: z ∈ {1, 1.25, …, 4}, x, y ∈ {−4, −3.5, …, 4}.
    pub fn measured_error(&self) -> f64 {
        let even = self.even_alpha_nodes();
        let mut worst: f64 = 0.0;
        for iz in 0..13 {
            let z = 1.0 + 0.25 * iz as f64;
            for ix in 0..17 {
                let x = -4.0 + 0.5 * ix as f64;
                for iy in 0..17 {
                    let y = -4.0 + 0.5 * iy as f64;
                    let mut sum = 0.0;
                    for (k, &n) in even.iter().enumerate() {
                        let lambda = self.lambdas[k];
                        let mut inner = 0.0;
                        for j in 0..n {
                            let alpha = core::f64::consts::TAU * j as f64 / n as f64;
                            inner += (lambda * (x * alpha.cos() + y * alpha.sin())).cos();
                        }
                        sum += self.weights[k] / n as f64 * (-lambda * z).exp() * inner;
                    }
                    let exact = 1.0 / (x * x + y * y + z * z).sqrt();
                    worst = worst.max((sum - exact).abs());
                }
            }
        }
        worst
    }

    /// The number of λ nodes, s.
    pub fn nodes(&self) -> usize {
        self.lambdas.len()
    }

    /// The α nodes per λ used here: Nₖ rounded up to an even number, so that α and
    /// α + π are both nodes and half of them carry the expansion (the conjugate
    /// symmetry of a real potential, `planewave` module documentation).
    pub fn even_alpha_nodes(&self) -> Vec<usize> {
        self.alpha_nodes.iter().map(|&n| n + n % 2).collect()
    }

    /// The complex values per direction stored here: Σₖ Nₖ' / 2 with Nₖ' the even counts.
    pub fn half_terms(&self) -> usize {
        self.even_alpha_nodes().iter().map(|&n| n / 2).sum()
    }

    /// The complex values per direction FMM3D stores, Σₖ Nₖ (all α nodes).
    pub fn fmm3d_terms(&self) -> usize {
        self.alpha_nodes.iter().sum()
    }

    /// The Fourier modes per direction, Σₖ Fₖ.
    pub fn fourier_terms(&self) -> usize {
        self.fourier.iter().sum()
    }

    /// The Fourier modes kept for λₖ at degree p: min(Fₖ, p + 1).
    pub fn modes(&self, k: usize, p: usize) -> usize {
        self.fourier[k].min(p + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_consistent() {
        for s in NODES {
            let q = Quadrature::with_nodes(s);
            assert_eq!(q.nodes(), s);
            assert_eq!(q.weights.len(), s);
            assert_eq!(q.alpha_nodes.len(), s);
            assert_eq!(q.fourier.len(), s);
            assert!(q.lambdas.windows(2).all(|w| w[0] < w[1]));
            assert!(q.max_error > 0.0 && q.max_error < 0.2);
        }
        // FMM3D's sums for the sizes its lfmm3d chooses (checked by hand against
        // lwtsexp_sep1.f): s = 12, 20, 29, 37.
        let sums: Vec<usize> = [12, 20, 29, 37]
            .iter()
            .map(|&s| Quadrature::with_nodes(s).fmm3d_terms())
            .collect();
        assert_eq!(sums, [217, 705, 1627, 2770]);
    }

    /// The λ sum meets the stated error (many α nodes); FMM3D's α counts exceed it.
    #[test]
    fn stated_and_measured_errors() {
        for s in [8, 12] {
            let q = Quadrature::with_nodes(s);
            let mut worst: f64 = 0.0;
            for &z in &[1.0, 1.3, 2.0, 3.1, 4.0] {
                for &(x, y) in &[(0.0, 0.0), (1.0, -2.0), (4.0, 4.0), (-3.5, 2.5)] {
                    let mut sum = 0.0;
                    for k in 0..s {
                        let (lambda, n) = (q.lambdas[k], 400);
                        let mut inner = 0.0;
                        for j in 0..n {
                            let alpha = core::f64::consts::TAU * j as f64 / n as f64;
                            inner += (lambda * (x * alpha.cos() + y * alpha.sin())).cos();
                        }
                        sum += q.weights[k] / n as f64 * (-lambda * z).exp() * inner;
                    }
                    let exact = 1.0 / (x * x + y * y + z * z).sqrt();
                    worst = worst.max((sum - exact).abs());
                }
            }
            assert!(worst <= 1.05 * q.max_error, "s = {s}: {worst:.3e}");
            let measured = q.measured_error();
            assert!(measured > 5.0 * q.max_error && measured < 25.0 * q.max_error);
        }
    }
}

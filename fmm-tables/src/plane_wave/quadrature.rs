//! The plane-wave quadratures (CONVENTIONS §3.14, "The quadrature"): FMM3D's tables for
//! separation 1 (`quadrature_data`), and the choice of a quadrature for a degree p.
//!
//! For a target above a source by z ∈ [1, 4] box widths and laterally within ρ ≤ 4√2
//! (a V-list pair whose offset along its direction is 2 or 3 boxes),
//!
//! 1/|u| = (1/2π) ∫₀^∞ e^{−λ z} ∫₀^{2π} e^{iλ(x cos α + y sin α)} dα dλ
//!       ≈ Σₖ (wₖ / Nₖ) Σⱼ e^{−λₖ z + iλₖ(x cos αⱼ + y sin αⱼ)},  αⱼ = 2π j / Nₖ,
//!
//! with s nodes λₖ and weights wₖ (a generalised Gaussian quadrature) and Nₖ trapezoidal
//! nodes in α for λₖ (FMM3D's `numthetafour`). The multipole's orders from Fₖ on
//! (`numthetahalf`) are dropped for λₖ. FMM3D states a maximum error of the s-node
//! quadrature over that region (box width 1); with its α nodes the error measured over
//! the region is 6–23× larger, largest at z = 1 and ρ = 4√2
//! ([`Quadrature::measured_error`]; Phase 6 T10, `spikes/plane-wave/SPIKE_REPORT.md` §2).
//! The degree's quadrature, [`nodes_for_degree`], is chosen by the measured error of the
//! FMM instead (Phase 6 T13).
//!
//! The tables are NIST's, extracted from FMM3D v2.1.0 (`src/Laplace/lwtsexp_sep1.f`) by
//! `tools/fixtures/plane_wave_quadrature.sh`, with the file's notice kept in
//! `quadrature_data.rs`.

use super::quadrature_data::TABLES;

/// The smallest and largest numbers of λ nodes in the tables.
pub const NODES: core::ops::RangeInclusive<usize> = 2..=39;

/// The largest degree with a plane-wave quadrature, [`nodes_for_degree`].
pub const MAX_DEGREE: usize = 20;

/// The number of λ nodes s of the quadrature of degree p, for p ≤ [`MAX_DEGREE`]
/// (CONVENTIONS §3.14, "The quadrature").
///
/// *Source* (Phase 6 T13; the "matched" rule of Phase 6 T10's spike,
/// `spikes/plane-wave/SPIKE_REPORT.md` §5): the fewest nodes whose plane-wave M2L over the
/// whole of a uniform level 3 (512 boxes, four random charges each, 53,352 V pairs) errs
/// at most 0.1× the M2L truncation error of degree p, both the largest over four random
/// points of every box relative to the box's Σ_V |q| (the ignored test
/// `level_every_degree` prints the study), made non-decreasing in p. It found s = 17
/// at p = 17 but 30 at p = 18, near the rounding floor of the waves (about 5e-12 Σ|q|),
/// so p = 18 takes 32 as p = 17 does; Phase 6 T10 found s = 10, 15, 18, 21 and 32 at
/// p = 3, 6, 8, 12 and 18 the same way. So the FMM's errors stay within about 1.1× those
/// of the exact M2L, which the gates of `nd-fmm-exec` check (C3.2, C3.3).
///
/// # Panics
///
/// If `p` > [`MAX_DEGREE`].
pub fn nodes_for_degree(p: usize) -> usize {
    assert!(
        p <= MAX_DEGREE,
        "the plane-wave quadratures reach degree {MAX_DEGREE}, not {p}"
    );
    DEGREE_NODES[p]
}

/// [`nodes_for_degree`] by degree, p = 0 … [`MAX_DEGREE`].
const DEGREE_NODES: [usize; MAX_DEGREE + 1] = [
    3, 7, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 21, 23, 26, 27, 27, 32, 32, 36, 36,
];

/// One quadrature of FMM3D for separation 1 (box width 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Quadrature {
    /// The λ nodes, ascending.
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
        assert!(
            NODES.contains(&s),
            "FMM3D's tables have 2 to 39 λ nodes, not {s}"
        );
        let (lambdas, weights, alpha_nodes, fourier, max_error) = TABLES[s - 2];
        Self {
            lambdas: lambdas.to_vec(),
            weights: weights.to_vec(),
            alpha_nodes: alpha_nodes.to_vec(),
            fourier: fourier.to_vec(),
            max_error,
        }
    }

    /// The quadrature of degree `p`: [`with_nodes`](Self::with_nodes) of
    /// [`nodes_for_degree`]`(p)`.
    ///
    /// # Panics
    ///
    /// If `p` > [`MAX_DEGREE`].
    pub fn for_degree(p: usize) -> Self {
        Self::with_nodes(nodes_for_degree(p))
    }

    /// The largest error of the quadrature as used here (the even α counts of
    /// [`even_alpha_nodes`](Self::even_alpha_nodes)) against 1/|u| on a grid of the
    /// region: z ∈ {1, 1.25, …, 4}, x, y ∈ {−4, −3.5, …, 4}.
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

    /// The α nodes per λ used here: Nₖ rounded up to an even number Nₖ', so that α and
    /// α + π are both nodes and half of them carry the waves (CONVENTIONS §3.14, "Half
    /// the α nodes").
    pub fn even_alpha_nodes(&self) -> Vec<usize> {
        self.alpha_nodes.iter().map(|&n| n + n % 2).collect()
    }

    /// The complex values per direction stored here, H = Σₖ Nₖ' / 2.
    pub fn half_terms(&self) -> usize {
        self.even_alpha_nodes().iter().map(|&n| n / 2).sum()
    }

    /// The complex values per direction FMM3D stores, Σₖ Nₖ (all α nodes).
    pub fn fmm3d_terms(&self) -> usize {
        self.alpha_nodes.iter().sum()
    }

    /// The Fourier modes kept for λₖ at degree p: Fₖ' = min(Fₖ, p + 1).
    pub fn modes(&self, k: usize, p: usize) -> usize {
        self.fourier[k].min(p + 1)
    }
}

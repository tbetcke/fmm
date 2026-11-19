//! The signature grid of a high-frequency level (proposal §5.4; CONVENTIONS §3.15
//! "Signatures", S6's proposal).

use nd_fmm_math::RealScalar;
use nd_fmm_math::orthonormal::legendre;

use super::dd::{Dd, gauss_legendre};

/// The (θ, φ) grid of the diagonal form for expansions of degree p (P = p + 1, proposal
/// §5.4; CONVENTIONS §3.15 "Signatures"):
///
/// - 2P − 1 = 2p + 1 **rings** at the Gauss–Legendre nodes uᵢ = cos θᵢ in ascending order
///   (ring 0 nearest the south pole), with weights wᵢ;
/// - N_φ = 4P − 3 **azimuths** rounded up to a multiple of 4 (so N_φ = 4P = 4p + 4),
///   φⱼ = 2πj/N_φ;
/// - **sample** (i, j) at index i N_φ + j, in the direction
///   ŝ = (sin θᵢ cos φⱼ, sin θᵢ sin φⱼ, cos θᵢ), with the quadrature weight wᵢ 2π/N_φ.
///
/// The quadrature is exact for every spherical polynomial of degree up to 4P − 4, the
/// degree of Yₙᵐ T F with n ≤ p, a signature F of bandwidth p and a transfer function T of
/// bandwidth 2p. The φ count, a multiple of 4, maps the azimuths onto themselves under
/// x → −x, y → −y and x ↔ y, and the nodes are symmetric, so z → −z maps ring i onto ring
/// 2p − i: the reflection classes ([`Symmetry`](super::Symmetry)).
///
/// Nodes, weights and azimuths are computed in double-double and rounded once to `T`; each
/// ring's normalised Legendre values P̄ₙᵐ(uᵢ) ([`legendre::normalised`]) are computed in f64
/// from the rounded node and rounded to `T`. The directions stay in double-double for the
/// transfer functions ([`Transfer`](super::Transfer)).
///
/// ```
/// use nd_fmm_ref::helmholtz::diagonal::Grid;
///
/// let grid = Grid::<f64>::new(7);
/// assert_eq!((grid.rings(), grid.azimuths(), grid.samples()), (15, 32, 480));
/// let total: f64 = (0..grid.rings()).map(|i| grid.weight(i)).sum::<f64>() * 32.0;
/// assert!((total - 4.0 * std::f64::consts::PI).abs() < 1e-14);
/// ```
#[derive(Clone, Debug)]
pub struct Grid<T> {
    p: usize,
    azimuths: usize,
    /// cos θᵢ and sin θᵢ in double-double.
    u: Vec<Dd>,
    s: Vec<Dd>,
    /// cos φⱼ and sin φⱼ in double-double.
    cos_dd: Vec<Dd>,
    sin_dd: Vec<Dd>,
    /// cos θᵢ rounded.
    nodes: Vec<T>,
    /// wᵢ 2π/N_φ rounded.
    weights: Vec<T>,
    /// cos φⱼ, sin φⱼ rounded.
    cos: Vec<T>,
    sin: Vec<T>,
    /// Per ring, √((2 − δ_{m0})(2n + 1)/(4π)) P̄ₙᵐ(uᵢ) at `legendre::index(n, m)`: the
    /// θ-factor of Yₙ^{±m} at the ring.
    theta: Vec<T>,
}

/// N_φ for degree p: 4P − 3 with P = p + 1, rounded up to a multiple of 4.
pub const fn azimuth_count(p: usize) -> usize {
    (4 * (p + 1) - 3).div_ceil(4) * 4
}

impl<T: RealScalar> Grid<T> {
    /// The grid for degree p.
    pub fn new(p: usize) -> Self {
        let rings = 2 * p + 1;
        let azimuths = azimuth_count(p);
        let (u, w) = gauss_legendre(rings);
        let dphi = Dd::PI * Dd::from_f64(2.0) / Dd::from_usize(azimuths);
        let s: Vec<Dd> = u.iter().map(|&x| (Dd::ONE - x * x).sqrt()).collect();
        let (mut cos_dd, mut sin_dd) = (Vec::with_capacity(azimuths), Vec::with_capacity(azimuths));
        for j in 0..azimuths {
            let (sn, cs) = (dphi * Dd::from_usize(j)).sin_cos();
            cos_dd.push(cs);
            sin_dd.push(sn);
        }
        let round = |x: Dd| T::from_f64(x.to_f64());
        let len = legendre::len(p);
        let mut theta = vec![T::zero(); rings * len];
        let mut pbar = vec![0.0f64; len];
        for i in 0..rings {
            legendre::normalised(p, u[i].to_f64(), s[i].to_f64(), &mut pbar);
            for n in 0..=p {
                for m in 0..=n {
                    let c = (if m == 0 { 1.0 } else { 2.0 }) * (2 * n + 1) as f64
                        / (4.0 * core::f64::consts::PI);
                    theta[i * len + legendre::index(n, m)] =
                        T::from_f64(c.sqrt() * pbar[legendre::index(n, m)]);
                }
            }
        }
        Self {
            p,
            azimuths,
            nodes: u.iter().map(|&x| round(x)).collect(),
            weights: w.iter().map(|&x| round(x * dphi)).collect(),
            cos: cos_dd.iter().map(|&x| round(x)).collect(),
            sin: sin_dd.iter().map(|&x| round(x)).collect(),
            u,
            s,
            cos_dd,
            sin_dd,
            theta,
        }
    }

    /// The degree p of the expansions.
    pub fn degree(&self) -> usize {
        self.p
    }

    /// The number of rings, 2p + 1.
    pub fn rings(&self) -> usize {
        self.nodes.len()
    }

    /// The number of azimuths N_φ ([`azimuth_count`]).
    pub fn azimuths(&self) -> usize {
        self.azimuths
    }

    /// The number of samples, (2p + 1) N_φ, the length of a signature.
    pub fn samples(&self) -> usize {
        self.rings() * self.azimuths
    }

    /// cos θᵢ of ring i.
    pub fn node(&self, i: usize) -> T {
        self.nodes[i]
    }

    /// The quadrature weight wᵢ 2π/N_φ of a sample on ring i.
    pub fn weight(&self, i: usize) -> T {
        self.weights[i]
    }

    /// The direction ŝ of sample (i, j), rounded to `T`.
    pub fn direction(&self, i: usize, j: usize) -> [T; 3] {
        self.direction_dd(i, j).map(|x| T::from_f64(x.to_f64()))
    }

    /// The direction of sample (i, j) in double-double.
    pub(super) fn direction_dd(&self, i: usize, j: usize) -> [Dd; 3] {
        [
            self.s[i] * self.cos_dd[j],
            self.s[i] * self.sin_dd[j],
            self.u[i],
        ]
    }

    /// cos φⱼ and sin φⱼ for j < N_φ, each from double-double rounded once to `T`
    /// (CH4.1a: the tables of `nd_fmm_tables::helmholtz::diagonal` are built from them).
    ///
    /// ```
    /// use nd_fmm_ref::helmholtz::diagonal::Grid;
    ///
    /// let grid = Grid::<f64>::new(3);
    /// let (cos, sin) = grid.azimuth_tables();
    /// assert_eq!((cos.len(), cos[0], sin[0]), (16, 1.0, 0.0));
    /// assert_eq!(cos[8], -1.0);
    /// ```
    pub fn azimuth_tables(&self) -> (&[T], &[T]) {
        (&self.cos, &self.sin)
    }

    /// Ring i's θ-factors of Yₙ^{±m}, √((2 − δ_{m0})(2n + 1)/(4π)) P̄ₙᵐ(uᵢ), at
    /// [`legendre::index`]`(n, m)` for 0 ≤ m ≤ n ≤ p: the real factor that multiplies
    /// cos mφ (coefficient +m) and sin mφ (coefficient −m) in Yₙ^{±m} on the ring (CH4.1a:
    /// the tables of `nd_fmm_tables::helmholtz::diagonal` are built from them).
    ///
    /// # Panics
    ///
    /// If i is not a ring (i ≥ 2p + 1).
    ///
    /// ```
    /// use nd_fmm_math::orthonormal::legendre;
    /// use nd_fmm_ref::helmholtz::diagonal::Grid;
    ///
    /// let grid = Grid::<f64>::new(4);
    /// let theta = grid.theta(2);
    /// assert_eq!(theta.len(), legendre::len(4));
    /// // Y₀⁰ = 1/√(4π) on every ring.
    /// assert!((theta[legendre::index(0, 0)] - 0.5 / std::f64::consts::PI.sqrt()).abs() < 1e-16);
    /// ```
    pub fn theta(&self, i: usize) -> &[T] {
        let len = legendre::len(self.p);
        &self.theta[i * len..(i + 1) * len]
    }
}

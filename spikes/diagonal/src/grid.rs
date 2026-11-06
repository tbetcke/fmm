//! The signature grid of proposal §5.4 and its tables.
//!
//! For P = p + 1 (degrees 0..p), 2P − 1 Gauss–Legendre nodes uᵢ = cos θᵢ and N_φ = 4P − 3
//! azimuths padded to a multiple of 4, φⱼ = 2πj/N_φ (CONVENTIONS §3.15 "Signatures",
//! proposed): exact for every product of degree up to 4P − 4, the degree of Yₙᵐ T F with
//! n < P, F of bandwidth P − 1 and T of bandwidth 2P − 2. The padding makes the φ grid
//! map onto itself under x → −x, y → −y and x ↔ y ([`crate::classes`]); the nodes in cos θ
//! are symmetric, so z → −z maps the rings onto each other.
//!
//! With the basis of S7 (`nd_fmm_spike_truncation::harmonics`),
//! Yₙ⁰ = Θₙ⁰(θ), Yₙ^{+m} = Θₙᵐ(θ) cos mφ, Yₙ^{−m} = Θₙᵐ(θ) sin mφ, where Θₙᵐ = p̄ₙᵐ(cos θ)/√π
//! for m ≥ 1 and p̄ₙ⁰(cos θ)/√(2π) for m = 0; a ring's Θ is S7's Yₙᵐ at φ = 0.
//!
//! Everything is computed in double-double and rounded to the working precision, as the
//! tables of §6.2 are rounded when stored.

use nd_fmm_spike_truncation::coaxial::gauss_legendre;
use nd_fmm_spike_truncation::harmonics::{Harmonics, Polar, idx};
use nd_fmm_spike_truncation::real::{Dd, Real};

/// Index of (n, m), 0 ≤ m ≤ n, in a triangle.
#[inline(always)]
pub const fn tri(n: usize, m: usize) -> usize {
    n * (n + 1) / 2 + m
}

/// Number of (n, m), 0 ≤ m ≤ n ≤ p.
pub const fn tri_len(p: usize) -> usize {
    (p + 1) * (p + 2) / 2
}

/// N_φ for degree p: 4P − 3 with P = p + 1, rounded up to a multiple of 4.
pub const fn phi_count(p: usize) -> usize {
    (4 * (p + 1) - 3).div_ceil(4) * 4
}

/// The grid for degree p, in double-double.
#[derive(Clone, Debug)]
pub struct Grid {
    /// The degree p (P = p + 1).
    pub p: usize,
    /// cos θᵢ, the 2p + 1 Gauss–Legendre nodes, ascending.
    pub u: Vec<Dd>,
    /// sin θᵢ.
    pub s: Vec<Dd>,
    /// The quadrature weight of a sample on ring i, wᵢ 2π/N_φ.
    pub w: Vec<Dd>,
    /// N_φ.
    pub nphi: usize,
    /// cos φⱼ, j < N_φ.
    pub cos: Vec<Dd>,
    /// sin φⱼ.
    pub sin: Vec<Dd>,
}

impl Grid {
    /// The grid for degree p.
    pub fn new(p: usize) -> Self {
        let nt = 2 * p + 1;
        let (u, w) = gauss_legendre::<Dd>(nt);
        let nphi = phi_count(p);
        let two_pi = Dd::pi() * Dd::from_f64(2.0);
        let dphi = two_pi / Dd::from_usize(nphi);
        let s = u.iter().map(|&x| (Dd::ONE - x * x).sqrt()).collect();
        let w = w.into_iter().map(|x| x * dphi).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(nphi), Vec::with_capacity(nphi));
        for j in 0..nphi {
            let (sn, cs) = (dphi * Dd::from_usize(j)).sin_cos();
            cos.push(cs);
            sin.push(sn);
        }
        Grid {
            p,
            u,
            s,
            w,
            nphi,
            cos,
            sin,
        }
    }

    /// Number of rings, 2p + 1.
    pub fn rings(&self) -> usize {
        self.u.len()
    }

    /// Number of samples, (2p + 1) N_φ.
    pub fn samples(&self) -> usize {
        self.rings() * self.nphi
    }

    /// Θₙᵐ(θᵢ) at [`tri`], n ≤ p, rounded to T; `harm` is S7's work space of degree p.
    pub fn theta<T: Real>(&self, i: usize, harm: &mut Harmonics<Dd>, out: &mut Vec<T>) {
        self.eval_ring(i, harm);
        self.theta_of(harm, out);
    }

    /// Evaluates ring i's harmonics into `harm` (degree p), for [`Grid::theta_of`].
    pub fn eval_ring(&self, i: usize, harm: &mut Harmonics<Dd>) {
        let polar = Polar {
            r: Dd::ONE,
            u: self.u[i],
            s: self.s[i],
            cp: Dd::ONE,
            sp: Dd::ZERO,
        };
        harm.eval(&polar, false);
    }

    /// The ring's Θ from `harm` after [`Grid::eval_ring`], rounded to T.
    pub fn theta_of<T: Real>(&self, harm: &Harmonics<Dd>, out: &mut Vec<T>) {
        out.clear();
        for n in 0..=self.p {
            for m in 0..=n {
                out.push(round(harm.y[idx(n, m as isize)]));
            }
        }
    }

    /// The direction ŝ of sample (i, j).
    pub fn direction(&self, i: usize, j: usize) -> [Dd; 3] {
        [self.s[i] * self.cos[j], self.s[i] * self.sin[j], self.u[i]]
    }

    /// The tables of working precision T.
    pub fn tables<T: Real>(&self) -> Tables<T> {
        Tables {
            w: self.w.iter().map(|&x| round(x)).collect(),
            cos: self.cos.iter().map(|&x| round(x)).collect(),
            sin: self.sin.iter().map(|&x| round(x)).collect(),
        }
    }
}

/// A double-double value rounded to T: through f64 for f32 and f64 (for f32 a double
/// rounding, at most one f32 ulp of difference from the nearest), exactly for [`Dd`].
#[inline]
pub fn round<T: Real>(x: Dd) -> T {
    if T::EPSILON < 1e-20 {
        // T is Dd: rebuild it from its two parts exactly.
        T::from_f64(x.hi) + T::from_f64(x.lo)
    } else {
        T::from_f64(x.hi + x.lo)
    }
}

/// The grid's weights and azimuth tables in working precision.
#[derive(Clone, Debug)]
pub struct Tables<T> {
    /// wᵢ 2π/N_φ per ring.
    pub w: Vec<T>,
    /// cos φⱼ.
    pub cos: Vec<T>,
    /// sin φⱼ.
    pub sin: Vec<T>,
}

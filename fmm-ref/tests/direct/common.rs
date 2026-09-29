//! Helpers shared by the translation tests: a seeded random generator, test frames,
//! vector arithmetic, random coefficients, the per-degree weighted coefficient error
//! and the per-degree term magnitudes of each translation.
//!
//! # Weighted norms and term magnitudes
//!
//! Coefficients are compared per degree in the orthonormal weighting of
//! CONVENTIONS §3.8 (docs/phase1/README.md, "Error measures"): slot m of degree n is
//! weighted by Nₘ = √((n + |m|)! (n − |m|)!) cₘ for multipoles and by Nₘ/Sₘ =
//! cₘ / √((n + |m|)! (n − |m|)!) for locals, with c₀ = 1 and cₘ = √2 otherwise. With
//! the addition theorem Σₘ (n − |m|)!/(n + |m|)! (Pₙᵐ)² = 1 (§3.3 Legendre functions),
//! these norms are
//!
//! ‖Rₙ(u)‖_N = |u|ⁿ,  ‖Iₙ(u)‖_{N/S} = 1/|u|ⁿ⁺¹,
//!
//! so a charge q at scaled position u contributes |q| |u|ⁿ to the degree-n norm of a
//! multipole (P2M) and |q| / |u|ⁿ⁺¹ to that of a local (P2L) expansion.
//!
//! A translation adds many terms per output coefficient, and they can cancel: a
//! source near the output centre has a small output coefficient of high degree, made
//! of large terms. The error is therefore measured relative to the magnitude of the
//! terms, not of the result. In the weighted norms, the terms of each translation of
//! CONVENTIONS §3.11 with input degree magnitudes aₙ are bounded, up to a factor
//! O(√p) from the Cauchy–Schwarz inequality over the orders, by
//!
//! - M2M: sⱼ = Σₖ≤ⱼ C(j, k) ρᵏ aₖ |b|ʲ⁻ᵏ, which for a charge at u is |q| (ρ|u| + |b|)ʲ,
//!   the triangle-inequality bound of |ρu + b|ʲ;
//! - L2L: sⱼ = σʲ⁺¹ Σₙ≥ⱼ C(n, j) aₙ |t|ⁿ⁻ʲ, the coefficient of sʲ in
//!   σ Σₙ aₙ (|t| + σs)ⁿ;
//! - M2L: sⱼ = σʲ⁺¹ Σₙ C(n + j, n) aₙ / |b|ⁿ⁺ʲ⁺¹, the coefficient of sʲ in
//!   σ Σₙ aₙ / (|b| − σs)ⁿ⁺¹.
//!
//! These are [`m2m_scale`], [`l2l_scale`] and [`m2l_scale`]. They compose: the scale of
//! a chain of two translations bounds the scale of the single translation it equals.

use core::cell::{Cell, RefCell};

use nd_fmm_math::{Layout, RealScalar};
use nd_fmm_ref::{Frame, Workspace, direct};

/// Largest degree tested in f64 (CONVENTIONS §3.9).
pub const P_MAX: usize = 30;

/// Largest degree of M2L tested in f64 (CONVENTIONS §3.9: irregular harmonics up to
/// degree 40).
pub const P_MAX_M2L: usize = 20;

/// Largest degree tested in f32 (CONVENTIONS §3.9).
pub const P_MAX_F32: usize = 8;

/// √3: scaled radius of the sphere around a box (CONVENTIONS §3.9).
pub const SQRT3: f64 = 1.732_050_807_568_877_2;

/// Tolerance of the exact translations (M2M, L2L, compositions, M2L structure), per
/// degree relative to the term magnitudes (brief T5).
pub const EXACT_TOL: f64 = 1e-13;

/// SplitMix64: a small, deterministic generator, so the tests need no extra dependency.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniformly distributed direction, by rejection sampling in the unit ball.
    pub fn unit_vector(&mut self) -> [f64; 3] {
        loop {
            let v = [0; 3].map(|_| self.range(-1.0, 1.0));
            let r = norm(v);
            if (1e-3..=1.0).contains(&r) {
                return v.map(|c| c / r);
            }
        }
    }

    /// Random vector with length uniform in [rmin, rmax] and a uniform direction.
    pub fn in_shell(&mut self, rmin: f64, rmax: f64) -> [f64; 3] {
        let r = self.range(rmin, rmax);
        self.unit_vector().map(|c| r * c)
    }

    /// Uniform in the cube [−1, 1]³.
    pub fn in_cube(&mut self) -> [f64; 3] {
        [0; 3].map(|_| self.range(-1.0, 1.0))
    }

    /// Random charge in [−1, 1], bounded away from zero.
    pub fn charge(&mut self) -> f64 {
        let q = self.range(0.1, 1.0);
        if self.uniform() < 0.5 { -q } else { q }
    }
}

/// Parent frames for the exactness tests: radii {1, 0.37, 2⁻¹⁶}, each at two centres
/// away from the origin.
pub fn frames() -> Vec<Frame<f64>> {
    let centres = [[0.3, -1.1, 0.7], [-2.5, 1.75, 3.25]];
    let radii = [1.0, 0.37, 2f64.powi(-16)];
    radii
        .iter()
        .flat_map(|&r| centres.iter().map(move |&c| Frame::new(c, r)))
        .collect()
}

/// The eight octant children of `parent`: radius r/2, centre c + (r/2) (±1, ±1, ±1).
pub fn octants(parent: &Frame<f64>) -> Vec<Frame<f64>> {
    let h = parent.radius / 2.0;
    (0..8)
        .map(|octant| {
            let sign = |bit: usize| if octant >> bit & 1 == 1 { 1.0 } else { -1.0 };
            let centre = core::array::from_fn(|i| parent.centre[i] + sign(i) * h);
            Frame::new(centre, h)
        })
        .collect()
}

/// A random frame nested in `outer`: its radius is `outer.radius` times a factor in
/// [0.1, 0.9], and its centre lies so that its sphere of radius √3 r stays inside the
/// sphere of radius √3 r of `outer`.
pub fn nested(outer: &Frame<f64>, rng: &mut SplitMix64) -> Frame<f64> {
    let s = rng.range(0.1, 0.9);
    let offset = rng.in_shell(0.0, SQRT3 * (1.0 - s));
    Frame::new(place(outer, offset), s * outer.radius)
}

/// The point c + r u of `frame` at scaled coordinates `u`, rounded to f64. Tests use
/// the rounded point as the true one; its scaled coordinates are `frame.scaled(..)`.
pub fn place(frame: &Frame<f64>, u: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| frame.centre[i] + frame.radius * u[i])
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| a[i] - b[i])
}

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

pub fn len(p: usize) -> usize {
    Layout::new(p).len()
}

/// Which weight of the orthonormal basis of CONVENTIONS §3.8 a coefficient vector
/// takes (docs/phase1/README.md, "Error measures").
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    /// Conjugates of regular harmonics (multipoles): weight Nₘ.
    Multipole,
    /// Conjugates of irregular harmonics (locals): weight Nₘ/Sₘ.
    Local,
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|j| j as f64).product()
}

/// Binomial coefficient C(n, k).
pub fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// Weight of slot m of degree n: Nₘ = √((n + |m|)! (n − |m|)!) cₘ, or Nₘ/Sₘ =
/// cₘ / √((n + |m|)! (n − |m|)!), with c₀ = 1 and cₘ = √2 otherwise (§3.8).
pub fn weight(kind: Kind, n: usize, m: isize) -> f64 {
    let k = m.unsigned_abs();
    let c = if k == 0 { 1.0 } else { 2f64.sqrt() };
    let f = (factorial(n + k) * factorial(n - k)).sqrt();
    match kind {
        Kind::Multipole => f * c,
        Kind::Local => c / f,
    }
}

/// Per-degree weighted norms ‖W xₙ‖₂, n ≤ p.
pub fn degree_norms(kind: Kind, p: usize, x: &[f64]) -> Vec<f64> {
    let layout = Layout::new(p);
    layout
        .degrees()
        .map(|(n, range)| {
            range
                .map(|i| (weight(kind, n, layout.nm(i).1) * x[i]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

/// Worst per-degree error of `got` against `want`: the largest over n of
/// ‖W (got − want)ₙ‖₂ / scaleₙ, weighted as in [`weight`].
pub fn degree_error(kind: Kind, p: usize, got: &[f64], want: &[f64], scale: &[f64]) -> f64 {
    let diff: Vec<f64> = got.iter().zip(want).map(|(a, b)| a - b).collect();
    degree_norms(kind, p, &diff)
        .iter()
        .zip(scale)
        .map(|(e, s)| {
            assert!(
                *s > 0.0 || *e == 0.0,
                "degree with zero scale but error {e}"
            );
            if *s > 0.0 { e / s } else { 0.0 }
        })
        .fold(0.0, f64::max)
}

/// Random coefficients of degree ≤ p whose weighted entries are uniform in [−1, 1]
/// (slot m of degree n is a uniform number divided by its weight).
pub fn random_coefficients(kind: Kind, p: usize, rng: &mut SplitMix64) -> Vec<f64> {
    let layout = Layout::new(p);
    (0..layout.len())
        .map(|i| {
            let (n, m) = layout.nm(i);
            rng.range(-1.0, 1.0) / weight(kind, n, m)
        })
        .collect()
}

/// Degree magnitudes Σⱼ |qⱼ| |uⱼ|ⁿ of the P2M expansion of charges `q` at scaled
/// positions `u` (module documentation).
pub fn p2m_magnitudes(p: usize, u: &[[f64; 3]], q: &[f64]) -> Vec<f64> {
    (0..=p)
        .map(|n| {
            u.iter()
                .zip(q)
                .map(|(&u, q)| q.abs() * norm(u).powi(n as i32))
                .sum()
        })
        .collect()
}

/// M2M term magnitudes sⱼ = Σₖ≤ⱼ C(j, k) ρᵏ aₖ |b|ʲ⁻ᵏ (module documentation).
pub fn m2m_scale(a: &[f64], rho: f64, b: f64) -> Vec<f64> {
    (0..a.len())
        .map(|j| {
            (0..=j)
                .map(|k| binomial(j, k) * rho.powi(k as i32) * a[k] * b.powi((j - k) as i32))
                .sum()
        })
        .collect()
}

/// L2L term magnitudes sⱼ = σʲ⁺¹ Σₙ≥ⱼ C(n, j) aₙ |t|ⁿ⁻ʲ (module documentation).
pub fn l2l_scale(a: &[f64], sigma: f64, t: f64) -> Vec<f64> {
    (0..a.len())
        .map(|j| {
            let sum: f64 = (j..a.len())
                .map(|n| binomial(n, j) * a[n] * t.powi((n - j) as i32))
                .sum();
            sigma.powi(j as i32 + 1) * sum
        })
        .collect()
}

/// M2L term magnitudes sⱼ = σʲ⁺¹ Σₙ C(n + j, n) aₙ / |b|ⁿ⁺ʲ⁺¹ (module documentation).
pub fn m2l_scale(a: &[f64], sigma: f64, b: f64) -> Vec<f64> {
    (0..a.len())
        .map(|j| {
            let sum: f64 = (0..a.len())
                .map(|n| binomial(n + j, n) * a[n] / b.powi((n + j + 1) as i32))
                .sum();
            sigma.powi(j as i32 + 1) * sum
        })
        .collect()
}

/// The scaled M2M shift b = (c − c′)/r′ and radius factor ρ = r/r′ (§3.11).
pub fn m2m_shift(from: &Frame<f64>, to: &Frame<f64>) -> ([f64; 3], f64) {
    (to.scaled(from.centre), from.radius / to.radius)
}

/// The scaled L2L and M2L shift (c′ − c)/r and radius factor σ = r′/r (§3.11).
pub fn l2l_shift(from: &Frame<f64>, to: &Frame<f64>) -> ([f64; 3], f64) {
    (from.scaled(to.centre), to.radius / from.radius)
}

/// The signature shared by `direct::m2m`, `direct::l2l` and `direct::m2l`.
pub type Translate<T> = fn(usize, &Frame<T>, &Frame<T>, &mut Workspace<T>, &[T], &mut [T]);

/// One of the three translations, with the kinds of its input and output and its term
/// magnitudes.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    M2m,
    L2l,
    M2l,
}

impl Op {
    pub const ALL: [Op; 3] = [Op::M2m, Op::L2l, Op::M2l];

    /// The operator of `nd_fmm_ref::direct`.
    pub fn function<T: RealScalar>(self) -> Translate<T> {
        match self {
            Op::M2m => direct::m2m::<T>,
            Op::L2l => direct::l2l::<T>,
            Op::M2l => direct::m2l::<T>,
        }
    }

    /// Kind of the input coefficients.
    pub fn input(self) -> Kind {
        match self {
            Op::M2m | Op::M2l => Kind::Multipole,
            Op::L2l => Kind::Local,
        }
    }

    /// Kind of the output coefficients.
    pub fn output(self) -> Kind {
        match self {
            Op::M2m => Kind::Multipole,
            Op::L2l | Op::M2l => Kind::Local,
        }
    }

    /// Per-degree term magnitudes of the translation of `input` from `from` to `to`
    /// (module documentation).
    pub fn scale(self, p: usize, from: &Frame<f64>, to: &Frame<f64>, input: &[f64]) -> Vec<f64> {
        let magnitudes = degree_norms(self.input(), p, input);
        match self {
            Op::M2m => {
                let (b, rho) = m2m_shift(from, to);
                m2m_scale(&magnitudes, rho, norm(b))
            }
            Op::L2l => {
                let (t, sigma) = l2l_shift(from, to);
                l2l_scale(&magnitudes, sigma, norm(t))
            }
            Op::M2l => {
                let (b, sigma) = l2l_shift(from, to);
                m2l_scale(&magnitudes, sigma, norm(b))
            }
        }
    }

    /// Largest degree tested in f64: 30, and 20 for M2L (CONVENTIONS §3.9).
    pub fn p_max(self) -> usize {
        match self {
            Op::M2l => P_MAX_M2L,
            Op::M2m | Op::L2l => P_MAX,
        }
    }

    /// Typical frame pairs (from, to), as on an octree: child to parent (M2M), parent to
    /// child (L2L), and V-list neighbours on one level plus a pair with σ = 0.6 (M2L).
    pub fn pairs(self) -> Vec<(Frame<f64>, Frame<f64>)> {
        let parent = Frame::new([0.3, -1.1, 0.7], 0.37);
        let children = octants(&parent);
        let child = Frame::new([0.45, -1.0, 0.55], 0.15);
        match self {
            Op::M2m => vec![(children[3], parent), (child, parent)],
            Op::L2l => vec![(parent, children[5]), (parent, child)],
            Op::M2l => {
                let shifted =
                    |d: [f64; 3], sigma: f64| Frame::new(place(&parent, d), sigma * parent.radius);
                vec![
                    (parent, shifted([4.0, 0.0, -2.0], 1.0)),
                    (parent, shifted([-6.0, 6.0, 6.0], 1.0)),
                    (parent, shifted([2.5, -2.0, 1.5], 0.6)),
                ]
            }
        }
    }
}

/// Runs `op` in f64 from zero output.
pub fn apply(
    op: Op,
    p: usize,
    from: &Frame<f64>,
    to: &Frame<f64>,
    ws: &mut Workspace<f64>,
    input: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; len(p)];
    op.function()(p, from, to, ws, input, &mut out);
    out
}

/// Tracks and prints the worst value of an error measure. It updates through a shared
/// reference, so that proptest bodies (which are `Fn` closures) can use it.
pub struct Worst {
    name: &'static str,
    value: Cell<f64>,
}

impl Worst {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            value: Cell::new(0.0),
        }
    }

    pub fn update(&self, value: f64) -> f64 {
        self.value.set(self.value.get().max(value));
        value
    }
}

impl Drop for Worst {
    fn drop(&mut self) {
        eprintln!("worst {}: {:.3e}", self.name, self.value.get());
    }
}

/// Degrees at which [`Strict`] prints its errors.
pub const STRICT_DEGREES: [usize; 5] = [12, 16, 20, 25, P_MAX];

/// Tracks, per degree p, the worst error relative to the reference itself (not to the
/// term magnitudes), and prints it at the degrees [`STRICT_DEGREES`]. Reported for
/// transparency only; no test asserts on it.
pub struct Strict {
    name: &'static str,
    values: RefCell<[f64; P_MAX + 1]>,
}

impl Strict {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            values: RefCell::new([0.0; P_MAX + 1]),
        }
    }

    pub fn update(&self, p: usize, value: f64) {
        let mut values = self.values.borrow_mut();
        values[p] = values[p].max(value);
    }
}

impl Drop for Strict {
    fn drop(&mut self) {
        let values = self.values.borrow();
        let row: Vec<String> = STRICT_DEGREES
            .iter()
            .map(|&p| format!("p = {p}: {:.1e}", values[p]))
            .collect();
        eprintln!("strict {}: {}", self.name, row.join(", "));
    }
}

//! The degrees of the octree's levels (proposal §3.5) and the candidates for the power
//! scale's radius (SPIKE_REPORT.md §5). Lengths are in units of a box's circumradius
//! a = √3 w/2, as in proposal §3.5's k a.

/// p of proposal §3.5 (Gumerov–Duraiswami 2009 eq. 18, δ = 2): p = ⌈P⌉ − 1. The same f64
/// arithmetic as `tools/reference.py`.
pub fn degree(ka: f64, eps: f64) -> usize {
    let p_hi = ka + (1.5 * (1.0 / eps).ln()).powf(2.0 / 3.0) * ka.powf(1.0 / 3.0);
    let p_lo = 1.0 - (eps * 0.5f64.powf(1.5)).ln() / 2.0f64.ln();
    ((p_lo.powi(4) + p_hi.powi(4)).powf(0.25)).ceil() as usize - 1
}

/// k a of a leaf c_leaf wavelengths wide: 2π c_leaf · √3/2.
pub fn leaf_ka(c_leaf: f64) -> f64 {
    std::f64::consts::PI * c_leaf * 3.0f64.sqrt()
}

/// The radius r of the power scale s = min(|k| r, 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Radius {
    /// The box width w = 2a/√3: FMM3D 2.1.0 (`hfmm3d.f`, `scales(ilev)`).
    Width,
    /// The half-width w/2 = a/√3: proposal §3.4.
    HalfWidth,
    /// The circumradius a: spike S2 (CH0.2) and this spike's fixture.
    Circumradius,
}

impl Radius {
    /// Every candidate.
    pub const ALL: [Radius; 3] = [Radius::Width, Radius::HalfWidth, Radius::Circumradius];

    /// s = min(|k| r, 1) on a level with |k| a = `ka`.
    pub fn scale(self, ka: f64) -> f64 {
        let r = match self {
            Radius::Width => 2.0 / 3.0f64.sqrt(),
            Radius::HalfWidth => 1.0 / 3.0f64.sqrt(),
            Radius::Circumradius => 1.0,
        };
        (ka * r).min(1.0)
    }

    /// A short name for tables.
    pub fn name(self) -> &'static str {
        match self {
            Radius::Width => "w",
            Radius::HalfWidth => "w/2",
            Radius::Circumradius => "a",
        }
    }
}

/// The tolerance whose degrees bound the ranges: the smallest of proposal §3.5's.
pub const EPS: f64 = 1e-10;

/// The largest k a of a level the ranges cover: level 2 at kD = 3000 (proposal §3.5).
pub const KA_MAX: f64 = 650.0;

/// The highest order of the regular functions (j, i) any operator needs at argument
/// modulus r: the M2M and L2L start values jₗ(k a/2) to p_l + p_{l+1}, so k a = 2r. It
/// covers P2M and L2P (k ρ ≤ k a_leaf) and the projection's jₙ(k a) to p + 1.
pub fn regular_top(r: f64) -> usize {
    let ka = (2.0 * r).min(KA_MAX);
    degree(ka, EPS) + degree(ka / 2.0, EPS)
}

/// The highest order of the singular functions (h, k) any operator needs at argument
/// modulus r: the M2L start values and the transfer function hₗ(k t) to 2p, with
/// t ≥ 2w = 4a/√3, so k a ≤ √3 r/4. It covers M2P and P2L (k ρ ≥ √3 k a_b, to p + 1) and
/// the projection's outer sphere (k ρ = 4 k a/√3, to p + 1).
pub fn singular_top(r: f64) -> usize {
    2 * degree((r * 3.0f64.sqrt() / 4.0).min(KA_MAX), EPS)
}

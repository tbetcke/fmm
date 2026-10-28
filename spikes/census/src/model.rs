//! The first-pass model: the formula p of proposal §3.5, the bound and the rule of §3.6,
//! the grid sizes of §6.7 and the memory and operation counts of §7, applied to a
//! [`Census`]. The passing classes are a function of k, the level and ε only; the regime
//! also of the global tree's counts, never of charges or a rank's share (constraint 9).
//!
//! Operation counts are complex multiply-adds with §7's conventions: an RCR translation
//! 2P³ (P1³ + P2³ when rectangular, from degree p1 to p2), a diagonal pair (2P − 1) N_φ, a
//! conversion (S → F or F → R) 2P³ for the Legendre step plus (2P − 1)² N_φ for the φ step
//! as a product (§6.7), and (p + 1)² per point and expansion for P2M, L2P, M2P and P2L.
//! P2P is counted in point pairs.

use std::f64::consts::PI;

use crate::census::{Census, DISTANCES, LevelCensus, NDIST};

/// C_T of the bound, proposal §3.6 (first pass; S6 sets it).
pub const C_T: f64 = 2.0;

/// θ of the rule, proposal §3.6 (first pass; S6 sets it).
pub const THETA: f64 = 1.0;

/// The unit roundoff of f64.
pub const U_F64: f64 = f64::EPSILON / 2.0;

/// Bytes of a complex f64.
const COMPLEX: f64 = 16.0;

/// Phase 5S's cap on the replicated top (`nd_fmm_plan::exchange::TOP_MAX_BOXES`).
pub const TOP_MAX_BOXES: u64 = 4096;

/// The polar angles of the M2L family's rotations (`nd_fmm_tables::rotation`, §5.3).
pub const M2L_POLAR_ANGLES: f64 = 47.0;

/// p of proposal §3.5 (Gumerov–Duraiswami 2009 eq. 18, δ = 2): p = ⌈P⌉ − 1. Copied from
/// `spikes/radial/src/ranges.rs`.
pub fn degree(ka: f64, eps: f64) -> usize {
    let p_hi = ka + (1.5 * (1.0 / eps).ln()).powf(2.0 / 3.0) * ka.powf(1.0 / 3.0);
    let p_lo = 1.0 - (eps * 0.5f64.powf(1.5)).ln() / 2.0f64.ln();
    ((p_lo.powi(4) + p_hi.powi(4)).powf(0.25)).ceil() as usize - 1
}

/// The width of a level-l box (root side 1).
pub fn width(level: usize) -> f64 {
    0.5f64.powi(level as i32)
}

/// k a_l, a_l the circumradius √3 w_l / 2.
pub fn level_ka(k: f64, level: usize) -> f64 {
    k * 3f64.sqrt() / 2.0 * width(level)
}

/// |h_n(x)| for n ≤ `top`, h_n = j_n + i y_n, by the upward recurrence (stable for h;
/// f64 overflow gives infinity).
pub fn hankel_abs(x: f64, top: usize) -> Vec<f64> {
    let (s, c) = x.sin_cos();
    let mut prev = (s / x, -c / x);
    let mut out = vec![prev.0.hypot(prev.1)];
    if top == 0 {
        return out;
    }
    let mut cur = (s / (x * x) - c / x, -c / (x * x) - s / x);
    out.push(cur.0.hypot(cur.1));
    for n in 1..top {
        let f = (2 * n + 1) as f64 / x;
        let next = (f * cur.0 - prev.0, f * cur.1 - prev.1);
        prev = cur;
        cur = next;
        out.push(cur.0.hypot(cur.1));
    }
    out
}

/// max_{j ≤ top} (2j + 1)|h_j(x)|.
pub fn max_hankel_term(x: f64, top: usize) -> f64 {
    hankel_abs(x, top)
        .iter()
        .enumerate()
        .map(|(j, h)| (2 * j + 1) as f64 * h)
        .fold(0.0, f64::max)
}

/// The bound E(l, c) of proposal §3.6 for a class at |t|² = `d2` w², degree p, box width
/// w: C_T k|t| u max_{j ≤ 2P − 2} (2j + 1)|h_j(k|t|)|, P = p + 1.
pub fn bound(k: f64, w: f64, p: usize, d2: i64) -> f64 {
    let x = k * (d2 as f64).sqrt() * w;
    C_T * x * U_F64 * max_hankel_term(x, 2 * p)
}

/// The mask of distance classes that pass the rule E ≤ θ ε on a level of width w.
pub fn passing(k: f64, w: f64, p: usize, eps: f64) -> u16 {
    DISTANCES
        .iter()
        .enumerate()
        .filter(|&(_, &d2)| bound(k, w, p, d2) <= THETA * eps)
        .fold(0, |m, (i, _)| m | 1 << i)
}

/// N_φ on the host (§6.7): the smallest multiple of 4 at least 4P − 3 whose other prime
/// factors are at most 7.
pub fn n_phi_host(p: usize) -> usize {
    let smooth = |mut n: usize| {
        for f in [2, 3, 5, 7] {
            while n.is_multiple_of(f) {
                n /= f;
            }
        }
        n == 1
    };
    let mut n = (4 * (p + 1) - 3).next_multiple_of(4);
    while !smooth(n / 4) {
        n += 4;
    }
    n
}

/// N_φ on a device level with a power-of-two FFT (§6.7).
pub fn n_phi_pow2(p: usize) -> usize {
    (4 * (p + 1) - 3).next_power_of_two()
}

/// Reals of one rotation block set (one polar angle), B = (p + 1)(2p + 1)(2p + 3)/3.
pub fn rotation_block_reals(p: usize) -> f64 {
    ((p + 1) * (2 * p + 1) * (2 * p + 3)) as f64 / 3.0
}

/// Complex entries of one coaxial (S|R) table: Σ_{m=0}^{p} (p + 1 − m)².
pub fn coaxial_entries(p: usize) -> f64 {
    (1..=p + 1).map(|j| (j * j) as f64).sum()
}

/// An RCR translation from degree `p1` to `p2`: P1³ + P2³.
pub fn rcr_cost(p1: usize, p2: usize) -> f64 {
    ((p1 + 1) as f64).powi(3) + ((p2 + 1) as f64).powi(3)
}

/// One conversion S → F or F → R of one box: 2P³ + (2P − 1)² N_φ.
pub fn conversion_cost(p: usize, n_phi: usize) -> f64 {
    let pp = (p + 1) as f64;
    2.0 * pp.powi(3) + (2.0 * pp - 1.0).powi(2) * n_phi as f64
}

/// Signature samples of one box: (2P − 1) N_φ.
pub fn signature_len(p: usize, n_phi: usize) -> f64 {
    ((2 * p + 1) * n_phi) as f64
}

/// The regime of a level (proposal §3.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    /// Levels 0 and 1: no V list.
    Top,
    /// Low frequency: every V pair by Rotation.
    L,
    /// High frequency: the passing classes diagonal, the rest RCR.
    H,
}

/// Operation counts of one level, complex multiply-adds (P2P in point pairs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ops {
    /// P2P point pairs of the level's leaves.
    pub p2p_pairs: f64,
    /// P2M of the level's leaves.
    pub p2m: f64,
    /// L2P of the level's leaves.
    pub l2p: f64,
    /// M2M from this level to the one above (when that one has V lists).
    pub m2m: f64,
    /// L2L into this level from the one above (when that one has V lists).
    pub l2l: f64,
    /// M2L into the level's boxes.
    pub m2l: f64,
    /// M2P of the level's leaves (W list, multipoles of level + 1).
    pub m2p: f64,
    /// P2L into the level's boxes (X list).
    pub p2l: f64,
}

impl Ops {
    /// The leaf costs of D11: P2P (weighted, `pair_weight` per pair), P2M, L2P, M2P, P2L.
    pub fn leaf_cost(&self, pair_weight: f64) -> f64 {
        pair_weight * self.p2p_pairs + self.p2m + self.l2p + self.m2p + self.p2l
    }

    /// The far field: M2M, L2L, M2L.
    pub fn far(&self) -> f64 {
        self.m2m + self.l2l + self.m2l
    }

    fn add(&mut self, o: &Ops) {
        self.p2p_pairs += o.p2p_pairs;
        self.p2m += o.p2m;
        self.l2p += o.l2p;
        self.m2m += o.m2m;
        self.l2l += o.l2l;
        self.m2l += o.m2l;
        self.m2p += o.m2p;
        self.p2l += o.p2l;
    }
}

/// The model of one level.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelModel {
    /// The level.
    pub level: usize,
    /// The box width in wavelengths.
    pub width_lambda: f64,
    /// k a_l.
    pub ka: f64,
    /// p_l by the formula.
    pub p: usize,
    /// The regime.
    pub regime: Regime,
    /// The distance classes that pass the bound (on an H level, the diagonal ones).
    pub passing: u16,
    /// N_φ on the host.
    pub n_phi: usize,
    /// N_φ as a power of two.
    pub n_phi_pow2: usize,
    /// V pairs between occupied boxes.
    pub v_pairs: u64,
    /// V pairs on RCR (all of them on an L level).
    pub v_rcr: u64,
    /// Boxes converted S → F (sources of a diagonal pair) on an H level.
    pub sig_sources: u64,
    /// Boxes converted F → R (targets of a diagonal pair) on an H level.
    pub sig_targets: u64,
    /// The M2L count with every class on RCR, for comparison.
    pub m2l_all_rcr: f64,
    /// The M2L count with every class diagonal (unstable where the bound fails).
    pub m2l_all_diagonal: f64,
    /// Bytes of the multipoles of the occupied boxes (complex; real multipoles half).
    pub multipole_bytes: f64,
    /// Bytes of the locals of the occupied boxes.
    pub local_bytes: f64,
    /// Bytes of the signature scratch (sources and targets).
    pub signature_bytes: f64,
    /// Bytes of the transfer functions of the passing reflection classes present.
    pub transfer_bytes: f64,
    /// Bytes of the (S|R) tables of the RCR distance classes present.
    pub coaxial_bytes: f64,
    /// Bytes of the rotation block sets: one for M2M and L2L, 47 for RCR M2L.
    pub rotation_bytes: f64,
    /// The operations.
    pub ops: Ops,
}

/// The model of a whole tree.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeModel {
    /// k (kD, root side 1).
    pub k: f64,
    /// ε.
    pub eps: f64,
    /// Per level, root first.
    pub levels: Vec<LevelModel>,
}

impl TreeModel {
    /// Applies the model to `census` at wavenumber `k` and tolerance `eps`.
    pub fn new(census: &Census, k: f64, eps: f64) -> Self {
        let n = census.levels.len();
        let p: Vec<usize> = (0..n).map(|l| degree(level_ka(k, l), eps)).collect();
        let levels = census
            .levels
            .iter()
            .map(|c| level_model(c, &p, k, eps))
            .collect();
        TreeModel { k, eps, levels }
    }

    /// The sum of the operations over the levels.
    pub fn ops(&self) -> Ops {
        let mut total = Ops::default();
        for l in &self.levels {
            total.add(&l.ops);
        }
        total
    }

    /// The H levels.
    pub fn high_frequency_levels(&self) -> Vec<usize> {
        self.levels
            .iter()
            .filter(|l| l.regime == Regime::H)
            .map(|l| l.level)
            .collect()
    }

    /// Bytes of the replicated top per rank: the multipoles of levels 2 to the deepest
    /// level whose occupied boxes from level 2 down stay within [`TOP_MAX_BOXES`], and
    /// the deepest such level.
    pub fn replicated_top(&self, census: &Census) -> (f64, usize) {
        let mut boxes = 0;
        let mut bytes = 0.0;
        let mut deepest = 1;
        for (m, c) in self.levels.iter().zip(&census.levels).skip(2) {
            boxes += c.occupied;
            if boxes > TOP_MAX_BOXES {
                break;
            }
            bytes += m.multipole_bytes;
            deepest = m.level;
        }
        (bytes, deepest)
    }

    /// Bytes of the multipoles of levels 2–4 (proposal §6.6's measure).
    pub fn top_levels_2_to_4(&self) -> f64 {
        self.levels
            .iter()
            .filter(|l| (2..=4).contains(&l.level))
            .map(|l| l.multipole_bytes)
            // Not `sum`: an empty f64 sum is −0.0.
            .fold(0.0, |a, b| a + b)
    }
}

fn level_model(c: &LevelCensus, p: &[usize], k: f64, eps: f64) -> LevelModel {
    let l = c.level;
    let w = width(l);
    let pl = p[l];
    let pp = (pl + 1) as f64;
    let n_phi = n_phi_host(pl);
    let siglen = signature_len(pl, n_phi);
    let v_pairs = c.v_pairs();
    let rcr = 2.0 * pp.powi(3);
    let m2l_all_rcr = v_pairs as f64 * rcr;
    let all = (1u16 << NDIST) - 1;
    let m2l_all_diagonal = (c.sources_in(all) + c.targets_in(all)) as f64
        * conversion_cost(pl, n_phi)
        + v_pairs as f64 * siglen;

    // The rule of §3.6: H if some class passes and the hybrid costs less than RCR.
    let present: u16 = c
        .v_by_distance
        .iter()
        .enumerate()
        .filter(|&(_, &n)| n > 0)
        .fold(0, |m, (i, _)| m | 1 << i);
    let pass = if l >= 2 {
        passing(k, w, pl, eps) & present
    } else {
        0
    };
    let diag: u64 = (0..NDIST)
        .filter(|i| pass & (1 << i) != 0)
        .map(|i| c.v_by_distance[i])
        .sum();
    let sig_sources = c.sources_in(pass);
    let sig_targets = c.targets_in(pass);
    let hybrid = (sig_sources + sig_targets) as f64 * conversion_cost(pl, n_phi)
        + diag as f64 * siglen
        + (v_pairs - diag) as f64 * rcr;
    let regime = if l < 2 {
        Regime::Top
    } else if pass != 0 && hybrid < m2l_all_rcr {
        Regime::H
    } else {
        Regime::L
    };
    let (pass, diag, sig_sources, sig_targets, m2l) = match regime {
        Regime::H => (pass, diag, sig_sources, sig_targets, hybrid),
        Regime::L => (pass, 0, 0, 0, m2l_all_rcr),
        Regime::Top => (0, 0, 0, 0, 0.0),
    };
    let v_rcr = v_pairs - diag;

    // Memory.
    let expansion = c.occupied as f64 * pp * pp * COMPLEX;
    let signature_bytes = (sig_sources + sig_targets) as f64 * siglen * COMPLEX;
    let passing_classes = if regime == Regime::H {
        let classes = crate::census::reflection_classes();
        classes
            .iter()
            .zip(&c.v_by_class)
            .filter(|&(r, &n)| {
                let d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                let i = DISTANCES.iter().position(|&x| x == d2).unwrap();
                n > 0 && pass & (1 << i) != 0
            })
            .count()
    } else {
        0
    };
    let transfer_bytes = passing_classes as f64 * siglen * COMPLEX;
    let rcr_distances = (0..NDIST)
        .filter(|&i| c.v_by_distance[i] > 0 && pass & (1 << i) == 0)
        .count();
    let coaxial_bytes = rcr_distances as f64 * coaxial_entries(pl) * COMPLEX;
    let block_sets = if l >= 2 {
        1.0 + if v_rcr > 0 { M2L_POLAR_ANGLES } else { 0.0 }
    } else {
        0.0
    };
    let rotation_bytes = block_sets * rotation_block_reals(pl) * 8.0;

    // Operations.
    let sq = |q: usize| ((q + 1) * (q + 1)) as f64;
    let parent_has_v = l >= 3;
    let translations = if parent_has_v {
        c.occupied as f64 * rcr_cost(pl, p[l - 1])
    } else {
        0.0
    };
    let next = if l + 1 < p.len() { sq(p[l + 1]) } else { 0.0 };
    let ops = Ops {
        p2p_pairs: c.p2p_pairs as f64,
        p2m: c.leaf_points as f64 * sq(pl),
        l2p: c.leaf_points as f64 * sq(pl),
        m2m: translations,
        l2l: translations,
        m2l,
        m2p: c.w_points as f64 * next,
        p2l: c.x_points as f64 * sq(pl),
    };
    LevelModel {
        level: l,
        width_lambda: w * k / (2.0 * PI),
        ka: level_ka(k, l),
        p: pl,
        regime,
        passing: pass,
        n_phi,
        n_phi_pow2: n_phi_pow2(pl),
        v_pairs,
        v_rcr,
        sig_sources,
        sig_targets,
        m2l_all_rcr,
        m2l_all_diagonal,
        multipole_bytes: expansion,
        local_bytes: expansion,
        signature_bytes,
        transfer_bytes,
        coaxial_bytes,
        rotation_bytes,
        ops,
    }
}

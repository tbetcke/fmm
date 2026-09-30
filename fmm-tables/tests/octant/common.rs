//! Helpers shared by the octant-table tests: a seeded random generator, the test
//! domains and their boxes, cached tables, random coefficients and the per-degree
//! weighted error relative to the terms of a table application.
//!
//! # Error measure
//!
//! Coefficients are compared per degree in the orthonormal weighting of
//! CONVENTIONS §3.8 (docs/phase1/README.md and docs/phase2/README.md, "Error
//! measures"): slot m of degree n is weighted by Nₘ = √((n + |m|)! (n − |m|)!) cₘ for
//! multipoles and by Nₘ/Sₘ = cₘ / √((n + |m|)! (n − |m|)!) for locals, with c₀ = 1 and
//! cₘ = √2 otherwise. The error of degree n is ‖W (got − want)ₙ‖₂ divided by
//! ‖W τₙ‖₂, where τᵢ = Σₖ |Aᵢₖ xₖ| are the term magnitudes of the table application
//! y = A x ("terms", [`terms`]). For a chain of two applications the terms are
//! |A₂| (|A₁| |x|), which bound those of each step.

use core::cell::Cell;
use std::sync::OnceLock;

use nd_fmm_math::Layout;
use nd_fmm_ref::Frame;
use nd_fmm_tables::{L2lTables, M2mTables, MatrixSet};

/// Largest degree tested in f64 (CONVENTIONS §3.9).
pub const P_MAX: usize = 30;

/// Largest degree of the debug run.
pub const P_DEBUG: usize = 12;

/// Largest degree tested in f32 (CONVENTIONS §3.9).
pub const P_MAX_F32: usize = 8;

/// Degrees of the debug run of the level and chain tests.
pub const DEBUG_DEGREES: [usize; 7] = [0, 1, 2, 3, 5, 8, P_DEBUG];

/// Tolerance of a table against `direct` in a dyadic domain, per degree relative to the
/// terms (C2.1).
pub const LEVEL_TOL: f64 = 1e-14;

/// Tolerance of the physical chains and compositions, per degree relative to the terms.
pub const CHAIN_TOL: f64 = 1e-13;

/// Parent levels of the level-independence test.
pub const LEVELS: [u32; 6] = [0, 1, 4, 8, 12, 15];

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

    /// Uniform in 0..n.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
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

/// A cubic domain with lower corner `corner` and side `side` (CONVENTIONS §3.12,
/// "Domain and levels").
#[derive(Clone, Copy, Debug)]
pub struct Domain {
    pub corner: [f64; 3],
    pub side: f64,
}

/// The dyadic domain a = (−1.25, 0.5, 2), w = 3: corner and side need few bits, so on
/// every level l ≤ 16 each centre, each centre difference and each scaled shift is
/// exact in f64.
pub const DYADIC: Domain = Domain {
    corner: [-1.25, 0.5, 2.0],
    side: 3.0,
};

/// The generic domain a = (0.1, −2.3, 7.9), w = 0.37: neither is exactly representable,
/// so centres are rounded.
pub const GENERIC: Domain = Domain {
    corner: [0.1, -2.3, 7.9],
    side: 0.37,
};

impl Domain {
    /// The frame (c, r_l) of the box on `level` with index `index`:
    /// c = a + (index + ½) w / 2^l and r_l = w / 2^(l+1), each formed in f64 in this
    /// order (CONVENTIONS §3.12).
    pub fn frame(&self, level: u32, index: [u64; 3]) -> Frame<f64> {
        let h = self.side / (1u64 << level) as f64;
        let centre = core::array::from_fn(|a| self.corner[a] + (index[a] as f64 + 0.5) * h);
        Frame::new(centre, self.side / (1u64 << (level + 1)) as f64)
    }

    /// A random box index on `level`.
    pub fn random_index(level: u32, rng: &mut SplitMix64) -> [u64; 3] {
        [0; 3].map(|_| rng.below(1 << level))
    }
}

/// The index on level l + 1 of child `o` of the box `index` on level l:
/// (2i + x, 2j + y, 2k + z) with o = 4x + 2y + z (CONVENTIONS §3.12).
pub fn child_index(index: [u64; 3], o: usize) -> [u64; 3] {
    let bits = [(o >> 2) & 1, (o >> 1) & 1, o & 1];
    core::array::from_fn(|a| 2 * index[a] + bits[a] as u64)
}

/// The M2M tables of degree p in f64, built once per test binary.
pub fn m2m(p: usize) -> &'static M2mTables<f64> {
    static TABLES: [OnceLock<M2mTables<f64>>; P_MAX + 1] = [const { OnceLock::new() }; P_MAX + 1];
    TABLES[p].get_or_init(|| M2mTables::build(p))
}

/// The L2L tables of degree p in f64, built once per test binary.
pub fn l2l(p: usize) -> &'static L2lTables<f64> {
    static TABLES: [OnceLock<L2lTables<f64>>; P_MAX + 1] = [const { OnceLock::new() }; P_MAX + 1];
    TABLES[p].get_or_init(|| L2lTables::build(p))
}

pub fn len(p: usize) -> usize {
    Layout::new(p).len()
}

/// Which weight of the orthonormal basis of CONVENTIONS §3.8 a coefficient vector
/// takes.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    /// Multipoles: weight Nₘ.
    Multipole,
    /// Locals: weight Nₘ/Sₘ.
    Local,
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|j| j as f64).product()
}

/// Weight of slot m of degree n: Nₘ or Nₘ/Sₘ (module documentation).
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
/// ‖W (got − want)ₙ‖₂ / ‖W τₙ‖₂ with τ the term magnitudes `terms`.
pub fn degree_error(kind: Kind, p: usize, got: &[f64], want: &[f64], terms: &[f64]) -> f64 {
    let diff: Vec<f64> = got.iter().zip(want).map(|(a, b)| a - b).collect();
    let scale = degree_norms(kind, p, terms);
    degree_norms(kind, p, &diff)
        .iter()
        .zip(&scale)
        .map(|(e, s)| {
            assert!(
                *s > 0.0 || *e == 0.0,
                "degree with zero terms but error {e}"
            );
            if *s > 0.0 { e / s } else { 0.0 }
        })
        .fold(0.0, f64::max)
}

/// The term magnitudes τᵢ = Σₖ |Aᵢₖ| xₖ of applying matrix `i` of `set` to the
/// nonnegative magnitudes `x` (module documentation). For a coefficient vector pass
/// its absolute values; for a chain pass the terms of the previous step.
pub fn terms(set: &MatrixSet<f64>, i: usize, x: &[f64]) -> Vec<f64> {
    let n = set.n();
    let mut out = vec![0.0; n];
    for (&xk, column) in x.iter().zip(set.matrix(i).chunks_exact(n)) {
        for (o, a) in out.iter_mut().zip(column) {
            *o += a.abs() * xk.abs();
        }
    }
    out
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

/// Applies matrix `i` of `set` to `x` from a zero output.
pub fn apply(set: &MatrixSet<f64>, i: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; set.n()];
    set.apply(i, x, &mut y);
    y
}

/// The point c + r u of `frame` at scaled coordinates `u`, rounded to f64.
pub fn place(frame: &Frame<f64>, u: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| frame.centre[i] + frame.radius * u[i])
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

    pub fn get(&self) -> f64 {
        self.value.get()
    }
}

impl Drop for Worst {
    fn drop(&mut self) {
        eprintln!("worst {}: {:.3e}", self.name, self.value.get());
    }
}

//! The workloads of design simd-p2p.md §8.2, the f64 oracle and the accuracy measures.
//!
//! - W1, FMM-shaped: a target leaf of n_t points uniform in [−1, 1]³ and the 27 leaves
//!   of its 3 × 3 × 3 block, n_t sources each, in [−3, 3]³. The centre leaf's sources
//!   are the targets themselves, so the self pairs are coincident. A pool of 64 sets
//!   is cycled. Per-pair form: 27 calls of n_t sources; gathered form: one call of
//!   27 n_t sources.
//! - W2, all-pairs: N sources uniform in [0, 1]³; targets distinct from them (another N
//!   uniform points) or equal to them.
//!
//! Charges are uniform in [−1, 1]. Accuracy (design §8.3): against `direct_sum` on the
//! same rounded inputs, the largest error relative to the sum of term magnitudes of its
//! target (Σ|q|/r for φ, Σ|q|/r² for each gradient component), and the relative L2
//! error over all targets.

use nd_fmm_ref::p2p::direct_sum;

use crate::rng::Rng;
use crate::simd::Elem;

/// Pool size of W1.
pub const POOL: usize = 64;

/// One set of inputs, with its outputs.
pub struct Set<E> {
    /// Targets.
    pub targets: Vec<[E; 3]>,
    /// The source leaves of the per-pair form (one leaf for W2).
    pub leaves: Vec<(Vec<[E; 3]>, Vec<E>)>,
    /// All sources in one chunk, leaf after leaf.
    pub sources: Vec<[E; 3]>,
    /// Their charges.
    pub charges: Vec<E>,
    /// Potential output, n_t.
    pub pot: Vec<E>,
    /// Gradient output, n_t.
    pub grad: Vec<[E; 3]>,
    /// green-kernels' interleaved output, 4 n_t.
    pub gk: Vec<E>,
}

impl<E: Elem> Set<E> {
    fn new(targets: Vec<[E; 3]>, leaves: Vec<(Vec<[E; 3]>, Vec<E>)>) -> Self {
        let sources = leaves.iter().flat_map(|l| l.0.iter().copied()).collect();
        let charges = leaves.iter().flat_map(|l| l.1.iter().copied()).collect();
        let n = targets.len();
        Self {
            targets,
            leaves,
            sources,
            charges,
            pot: vec![E::ZERO; n],
            grad: vec![[E::ZERO; 3]; n],
            gk: vec![E::ZERO; 4 * n],
        }
    }

    /// Source–target pairs of one evaluation, coincident ones included.
    pub fn pairs(&self) -> usize {
        self.targets.len() * self.sources.len()
    }

    /// Zeroes the outputs.
    pub fn clear(&mut self) {
        self.pot.fill(E::ZERO);
        self.grad.fill([E::ZERO; 3]);
        self.gk.fill(E::ZERO);
    }
}

/// How the sources of a set are passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// One call per source leaf.
    PerPair,
    /// One call with all sources.
    Gathered,
}

impl Form {
    /// Name for the tables.
    pub fn name(self) -> &'static str {
        match self {
            Form::PerPair => "per-pair",
            Form::Gathered => "gathered",
        }
    }
}

/// One workload cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    /// W1 with n_t and a form.
    W1(usize, Form),
    /// W2 with N; `true` if targets equal sources.
    W2(usize, bool),
}

impl Cell {
    /// Label for the tables.
    pub fn label(self) -> String {
        match self {
            Cell::W1(n, f) => format!("W1 n_t={n} {}", f.name()),
            Cell::W2(n, eq) => format!("W2 N={n} {}", if eq { "t=s" } else { "t≠s" }),
        }
    }

    /// The form the kernel is called in.
    pub fn form(self) -> Form {
        match self {
            Cell::W1(_, f) => f,
            Cell::W2(..) => Form::Gathered,
        }
    }

    /// Whether this cell is FMM-shaped (W1).
    pub fn is_w1(self) -> bool {
        matches!(self, Cell::W1(..))
    }
}

fn point<E: Elem>(rng: &mut Rng, c: [f64; 3], h: f64) -> [E; 3] {
    c.map(|ck| E::from_f64(ck + rng.range(-h, h)))
}

/// The W1 pool for n_t.
pub fn w1<E: Elem>(n_t: usize, seed: u64) -> Vec<Set<E>> {
    let mut rng = Rng::new(seed ^ (n_t as u64) << 32);
    (0..POOL)
        .map(|_| {
            let targets: Vec<[E; 3]> = (0..n_t).map(|_| point(&mut rng, [0.0; 3], 1.0)).collect();
            let mut leaves = Vec::with_capacity(27);
            for a in -1..=1 {
                for b in -1..=1 {
                    for c in -1..=1 {
                        let centre = [2.0 * a as f64, 2.0 * b as f64, 2.0 * c as f64];
                        let src: Vec<[E; 3]> = if (a, b, c) == (0, 0, 0) {
                            targets.clone()
                        } else {
                            (0..n_t).map(|_| point(&mut rng, centre, 1.0)).collect()
                        };
                        let q: Vec<E> = (0..n_t)
                            .map(|_| E::from_f64(rng.range(-1.0, 1.0)))
                            .collect();
                        leaves.push((src, q));
                    }
                }
            }
            Set::new(targets, leaves)
        })
        .collect()
}

/// The W2 set for N.
pub fn w2<E: Elem>(n: usize, equal: bool, seed: u64) -> Set<E> {
    let mut rng = Rng::new(seed ^ (n as u64) << 32);
    let src: Vec<[E; 3]> = (0..n).map(|_| point(&mut rng, [0.5; 3], 0.5)).collect();
    let q: Vec<E> = (0..n).map(|_| E::from_f64(rng.range(-1.0, 1.0))).collect();
    let targets = if equal {
        src.clone()
    } else {
        (0..n).map(|_| point(&mut rng, [0.5; 3], 0.5)).collect()
    };
    Set::new(targets, vec![(src, q)])
}

/// The f64 oracle of one set: `direct_sum` and the term magnitudes.
pub struct Oracle {
    pot: Vec<f64>,
    grad: Vec<[f64; 3]>,
    mag_pot: Vec<f64>,
    mag_grad: Vec<f64>,
}

impl Oracle {
    /// The oracle on the set's (rounded) inputs.
    pub fn new<E: Elem>(set: &Set<E>) -> Self {
        let up =
            |v: &[[E; 3]]| -> Vec<[f64; 3]> { v.iter().map(|p| p.map(|c| c.to_f64())).collect() };
        let s = up(&set.sources);
        let t = up(&set.targets);
        let q: Vec<f64> = set.charges.iter().map(|&c| c.to_f64()).collect();
        let n = t.len();
        let mut pot = vec![0.0; n];
        let mut grad = vec![[0.0; 3]; n];
        direct_sum(&s, &q, &t, &mut pot, Some(&mut grad));
        let mut mag_pot = vec![0.0; n];
        let mut mag_grad = vec![0.0; n];
        for (i, x) in t.iter().enumerate() {
            for (y, &qj) in s.iter().zip(&q) {
                if x == y {
                    continue;
                }
                let r2 = (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2);
                mag_pot[i] += qj.abs() / r2.sqrt();
                mag_grad[i] += qj.abs() / r2;
            }
        }
        Self {
            pot,
            grad,
            mag_pot,
            mag_grad,
        }
    }
}

/// Accuracy of one output against the oracle.
#[derive(Clone, Copy, Debug, Default)]
pub struct Accuracy {
    /// Largest error relative to the term magnitudes, potential.
    pub max_pot: f64,
    /// Relative L2 error, potential.
    pub l2_pot: f64,
    /// Largest error relative to the term magnitudes, gradient components (0 without).
    pub max_grad: f64,
    /// Relative L2 error, gradient (0 without).
    pub l2_grad: f64,
    /// Any non-finite output.
    pub non_finite: bool,
}

impl Accuracy {
    /// Combines two measurements (worst of each).
    pub fn worst(self, o: Accuracy) -> Accuracy {
        Accuracy {
            max_pot: self.max_pot.max(o.max_pot),
            l2_pot: self.l2_pot.max(o.l2_pot),
            max_grad: self.max_grad.max(o.max_grad),
            l2_grad: self.l2_grad.max(o.l2_grad),
            non_finite: self.non_finite || o.non_finite,
        }
    }

    /// The largest error relative to the term magnitudes, φ and ∇φ.
    pub fn max(self) -> f64 {
        self.max_pot.max(self.max_grad)
    }

    /// Whether this passes the sum tolerance of design §3, requirement 2: 1e-14 (f64)
    /// or 1e-6 (f32) relative to the term magnitudes, for up to 4,096 sources; scaled
    /// by n_s / 4,096 beyond (W2 at N = 10⁴), where plain summation may grow.
    ///
    /// `relaxed` rows (f64 below full precision) are checked against 1e-8 instead, the
    /// floor of the FMM's own error at p ≤ 20 (design §4.3).
    pub fn passes<E: Elem>(self, n_s: usize, relaxed: bool) -> bool {
        let tol = if relaxed {
            1e-8
        } else if E::NAME == "f32" {
            1e-6
        } else {
            1e-14
        };
        let tol = tol * (n_s as f64 / 4096.0).max(1.0);
        !self.non_finite && self.max() <= tol
    }
}

/// The accuracy of potentials and optional gradients against the oracle.
pub fn accuracy<E: Elem>(o: &Oracle, pot: &[E], grad: Option<&[[E; 3]]>) -> Accuracy {
    let mut a = Accuracy::default();
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &p) in pot.iter().enumerate() {
        let v = p.to_f64();
        a.non_finite |= !v.is_finite();
        let e = (v - o.pot[i]).abs();
        if o.mag_pot[i] > 0.0 {
            a.max_pot = a.max_pot.max(e / o.mag_pot[i]);
        }
        num += e * e;
        den += o.pot[i] * o.pot[i];
    }
    a.l2_pot = (num / den).sqrt();
    if let Some(grad) = grad {
        let (mut num, mut den) = (0.0, 0.0);
        for (i, gi) in grad.iter().enumerate() {
            for (c, &gc) in gi.iter().enumerate() {
                let v = gc.to_f64();
                a.non_finite |= !v.is_finite();
                let e = (v - o.grad[i][c]).abs();
                if o.mag_grad[i] > 0.0 {
                    a.max_grad = a.max_grad.max(e / o.mag_grad[i]);
                }
                num += e * e;
                den += o.grad[i][c] * o.grad[i][c];
            }
        }
        a.l2_grad = (num / den).sqrt();
    }
    if a.max_pot.is_nan() || a.max_grad.is_nan() {
        a.non_finite = true;
    }
    a
}

//! Helpers shared by the operator tests: a seeded generator, the test domains, keys and
//! their neighbours, points on a dyadic grid, chunks in the layout of CONVENTIONS
//! §3.13, the absolute frames of §3.12, the "terms" error measure, cached tables and a
//! recorder of worst errors.
//!
//! # Error measure ("terms")
//!
//! As in the Phase 1 and Phase 2 tests (docs/phase1/README.md, docs/phase2/README.md,
//! "Error measures"; mirrored from fmm-tables/tests/support): coefficients are compared
//! per degree in the orthonormal weighting of CONVENTIONS §3.8, slot m of degree n
//! weighted by Nₘ = √((n + |m|)! (n − |m|)!) cₘ for multipoles and by
//! cₘ / √((n + |m|)! (n − |m|)!) for locals (c₀ = 1, cₘ = √2 otherwise). The error of
//! degree n is ‖W (got − want)ₙ‖₂ / ‖W τₙ‖₂, with τ the magnitudes of the terms that
//! form each output slot: τᵢ = Σₖ |Aᵢₖ xₖ| for a table application y = A x (A the dense
//! f64 table, for every strategy), and τᵢ = Σⱼ |qⱼ Xᵢ(uⱼ)| for P2M and P2L. Values at
//! points (L2P, M2P, P2P) are compared relative to the sum of the magnitudes of their
//! terms: Σᵢ wᵢ |Cᵢ Xᵢ(v)| (wᵢ = 1 for m = 0, 2 otherwise, the doubling rule of §3.6)
//! for an expansion and Σ |q| / |x − y| for P2P; gradients likewise with ∇X and
//! |q| / |x − y|², as Euclidean norms over the three components.
//!
//! # Points
//!
//! For the tight tolerances the points of a box b are generated as x = c_b + r_b u
//! with every component of u on the grid 2⁻²⁰ ℤ ∩ [−1, 1] (`grid_points`). In the
//! dyadic domain x, x − a and the leaf-scaled u of `leaf_coordinates` are then exact
//! in f64 on every level, and so are the mapped P2P sources ĉ + r̂ u: the operator in
//! leaf-scaled coordinates and the reference at absolute frames see the same geometry
//! bit for bit, and differ only by rounding in their arithmetic. (For random doubles
//! the leaf-scaled u carries the loading error ε₆₄ |x − a| / r_l of §3.13, up to 2.9e-11
//! at level 16, which is the input's precision and not the operator's.)

use core::cell::Cell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use nd_fmm_exec::geometry::{Domain, centre, leaf_coordinates, radius};
use nd_fmm_exec::operator::{Isa, LaplaceOperator, P2pChoice};
use nd_fmm_exec::tables::{M2lStrategy, Tables};
use nd_fmm_math::{Layout, harmonics};
use nd_fmm_ref::Frame;
use nd_fmm_tables::MatrixSet;
use nd_octree::{MortonKey, PhysicalBox, morton};

/// The strategies with tables of their own.
pub const STRATEGIES: [M2lStrategy; 3] = [
    M2lStrategy::Dense,
    M2lStrategy::Classes,
    M2lStrategy::Rotation,
];

/// The levels of the C3.1 criterion.
pub const LEVELS: [usize; 3] = [2, 9, 16];

/// The tolerance of a translation with `strategy` against `nd_fmm_ref::direct`
/// (terms, per degree): 1e-14 dense, 1e-13 classes and rotation (C3.1).
pub fn translation_tolerance(strategy: M2lStrategy) -> f64 {
    match strategy {
        M2lStrategy::Dense => 1e-14,
        _ => 1e-13,
    }
}

/// The tolerance of the leaf operators and P2P against `nd_fmm_ref` (terms; C3.1).
pub const LEAF_TOL: f64 = 1e-13;

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
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A random charge in [−1, 1], bounded away from zero.
    pub fn charge(&mut self) -> f64 {
        let q = self.range(0.1, 1.0);
        if self.uniform() < 0.5 { -q } else { q }
    }

    /// A uniformly random key on `level`.
    pub fn key(&mut self, level: usize) -> MortonKey {
        let n = 1 << level;
        morton::from_index_and_level([0; 3].map(|_| self.below(n)), level)
    }

    /// A random key on `level` whose 26 neighbours all lie in the domain (level ≥ 2).
    pub fn interior_key(&mut self, level: usize) -> MortonKey {
        let n = (1 << level) - 2;
        morton::from_index_and_level([0; 3].map(|_| 1 + self.below(n)), level)
    }

    /// A random grid coordinate in [−1, 1]: k / 2²⁰ for an integer |k| ≤ 2²⁰.
    pub fn grid(&mut self) -> f64 {
        let steps = 1usize << 20;
        (self.below(2 * steps + 1) as f64 - steps as f64) / steps as f64
    }

    /// `count` random charges.
    pub fn charges(&mut self, count: usize) -> Vec<f64> {
        (0..count).map(|_| self.charge()).collect()
    }
}

/// The dyadic domain of Phase 2, a = (−1.25, 0.5, 2) and w = 3: every centre, centre
/// difference and half-width on levels 0–16 is exact in f64.
pub fn dyadic_domain() -> Domain {
    Domain::new(&PhysicalBox::new([-1.25, 0.5, 2.0, 1.75, 3.5, 5.0])).unwrap()
}

/// The generic domain, a = (0.1, −2.3, 7.9) and w = 0.37, built by hand as
/// [a, fl(a + w)].
pub fn generic_domain() -> Domain {
    let (a, w) = ([0.1, -2.3, 7.9], 0.37);
    Domain::new(&PhysicalBox::new([
        a[0],
        a[1],
        a[2],
        a[0] + w,
        a[1] + w,
        a[2] + w,
    ]))
    .unwrap()
}

/// The absolute frame (c, r_l) of `key` in `domain` (CONVENTIONS §3.12).
pub fn absolute_frame(key: MortonKey, domain: &Domain) -> Frame<f64> {
    Frame::new(centre(key, domain), radius(morton::level(key), domain))
}

/// The radius r_l of the box `key`.
pub fn key_radius(key: MortonKey, domain: &Domain) -> f64 {
    radius(morton::level(key), domain)
}

/// The keys of the boxes on the level of `key` that share at least a corner with it
/// and lie in the domain.
pub fn neighbours(key: MortonKey) -> Vec<MortonKey> {
    morton::neighbours(key)
        .into_iter()
        .filter(|&k| morton::is_valid(k))
        .collect()
}

/// Half-open bounds of `key` along each axis, in cells of the deepest level.
fn bounds(key: MortonKey) -> [[u64; 2]; 3] {
    let (level, index) = morton::decode(key);
    let size = 1u64 << (16 - level);
    index.map(|i| [i as u64 * size, (i as u64 + 1) * size])
}

/// Whether the closed boxes `a` and `b` share at least a point.
pub fn touches(a: MortonKey, b: MortonKey) -> bool {
    let (a, b) = (bounds(a), bounds(b));
    (0..3).all(|k| a[k][0] <= b[k][1] && b[k][0] <= a[k][1])
}

/// The V-list pair (source, target) on `level` with target − source = d, the target
/// uniform among the boxes whose source lies in the domain.
pub fn random_pair(level: usize, d: [i64; 3], rng: &mut SplitMix64) -> (MortonKey, MortonKey) {
    let n = 1i64 << level;
    let mut source = [0; 3];
    let mut target = [0; 3];
    for k in 0..3 {
        let (lo, hi) = (d[k].max(0), (n - 1).min(n - 1 + d[k]));
        assert!(lo <= hi, "level {level} has no pair at offset {d:?}");
        let t = lo + rng.below((hi - lo + 1) as usize) as i64;
        target[k] = t as usize;
        source[k] = (t - d[k]) as usize;
    }
    (
        morton::from_index_and_level(source, level),
        morton::from_index_and_level(target, level),
    )
}

/// Points of the box `key` on the grid of the module documentation: the absolute
/// coordinates x = c + r u and the leaf-scaled u. In the dyadic domain the function
/// asserts that `leaf_coordinates` reproduces u exactly.
pub fn grid_points(
    key: MortonKey,
    domain: &Domain,
    count: usize,
    rng: &mut SplitMix64,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let frame = absolute_frame(key, domain);
    let u: Vec<[f64; 3]> = (0..count).map(|_| [0; 3].map(|_| rng.grid())).collect();
    let x: Vec<[f64; 3]> = u
        .iter()
        .map(|u| core::array::from_fn(|k| frame.centre[k] + frame.radius * u[k]))
        .collect();
    if *domain == dyadic_domain() {
        for (x, u) in x.iter().zip(&u) {
            assert_eq!(leaf_coordinates::<f64>(*x, key, domain), *u, "x = {x:?}");
        }
    }
    (x, u)
}

/// Random points of the box `key` (any doubles, not on a grid): the absolute
/// coordinates and the leaf-scaled ones of `leaf_coordinates`.
pub fn random_points(
    key: MortonKey,
    domain: &Domain,
    count: usize,
    rng: &mut SplitMix64,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let [x0, y0, z0, x1, y1, z1] = morton::physical_box(key, &domain.physical_box()).coordinates();
    let x: Vec<[f64; 3]> = (0..count)
        .map(|_| [rng.range(x0, x1), rng.range(y0, y1), rng.range(z0, z1)])
        .collect();
    let u = x
        .iter()
        .map(|&x| leaf_coordinates(x, key, domain))
        .collect();
    (x, u)
}

/// A source chunk (CONVENTIONS §3.13): the coordinate triples, then the charges.
pub fn source_chunk<T: Copy>(u: &[[T; 3]], charges: &[T]) -> Vec<T> {
    assert_eq!(u.len(), charges.len());
    u.iter()
        .flatten()
        .copied()
        .chain(charges.iter().copied())
        .collect()
}

/// A target input chunk: the coordinate triples.
pub fn target_chunk<T: Copy>(u: &[[T; 3]]) -> Vec<T> {
    u.iter().flatten().copied().collect()
}

/// Splits a target output chunk with gradients into φ̂ and the triples ĝ.
pub fn split_output(output: &[f64]) -> (&[f64], Vec<[f64; 3]>) {
    let n = output.len() / 4;
    let (potential, gradient) = output.split_at(n);
    (potential, gradient.as_chunks::<3>().0.to_vec())
}

/// The number of coefficients of degree ≤ p.
pub fn len(p: usize) -> usize {
    Layout::new(p).len()
}

/// Which weight of the orthonormal basis a coefficient vector takes.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Multipole,
    Local,
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|j| j as f64).product()
}

/// The weight of slot m of degree n (module documentation).
pub fn weight(kind: Kind, n: usize, m: isize) -> f64 {
    let k = m.unsigned_abs();
    let c = if k == 0 { 1.0 } else { 2f64.sqrt() };
    let f = (factorial(n + k) * factorial(n - k)).sqrt();
    match kind {
        Kind::Multipole => f * c,
        Kind::Local => c / f,
    }
}

fn degree_norms(kind: Kind, p: usize, x: &[f64]) -> Vec<f64> {
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

/// The worst per-degree error of `got` against `want` relative to the term
/// magnitudes `terms` (module documentation).
pub fn degree_error(kind: Kind, p: usize, got: &[f64], want: &[f64], terms: &[f64]) -> f64 {
    assert_eq!(got.len(), len(p));
    assert_eq!(want.len(), len(p));
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

/// The term magnitudes τᵢ = Σₖ |Aᵢₖ xₖ| of applying matrix `i` of `set` to `x`.
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

/// Random coefficients of degree ≤ p whose weighted entries are uniform in [−1, 1].
pub fn random_coefficients(kind: Kind, p: usize, rng: &mut SplitMix64) -> Vec<f64> {
    let layout = Layout::new(p);
    (0..layout.len())
        .map(|i| {
            let (n, m) = layout.nm(i);
            rng.range(-1.0, 1.0) / weight(kind, n, m)
        })
        .collect()
}

/// Which solid harmonics a term measure uses.
#[derive(Clone, Copy, Debug)]
pub enum Basis {
    Regular,
    Irregular,
}

/// The values and the three gradient components of the harmonics of `basis` at `v`.
fn harmonics_at(basis: Basis, p: usize, v: [f64; 3]) -> (Vec<f64>, [Vec<f64>; 3]) {
    let mut values = vec![0.0; len(p)];
    let mut grad = [0; 3].map(|_| vec![0.0; len(p)]);
    let [gx, gy, gz] = &mut grad;
    let components = [&mut gx[..], &mut gy[..], &mut gz[..]];
    match basis {
        Basis::Regular => harmonics::regular_grad(p, v, &mut values, components),
        Basis::Irregular => harmonics::irregular_grad(p, v, &mut values, components),
    }
    (values, grad)
}

/// The coefficient terms Σⱼ |qⱼ Xᵢ(frame.scaled(yⱼ))| of P2M (regular) or P2L
/// (irregular), per slot.
pub fn point_terms(
    basis: Basis,
    p: usize,
    frame: &Frame<f64>,
    points: &[[f64; 3]],
    charges: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; len(p)];
    for (&y, &q) in points.iter().zip(charges) {
        let (values, _) = harmonics_at(basis, p, frame.scaled(y));
        for (o, v) in out.iter_mut().zip(&values) {
            *o += (q * v).abs();
        }
    }
    out
}

/// The evaluation terms of the expansion `coefficients` of `basis` in `frame` at `x`,
/// with the factors 1/R and 1/R² that `leaf::l2p` and `leaf::m2p` apply: the potential
/// terms and the Euclidean norm of the three gradient terms.
pub fn evaluation_terms(
    basis: Basis,
    p: usize,
    frame: &Frame<f64>,
    coefficients: &[f64],
    x: [f64; 3],
) -> (f64, f64) {
    let layout = Layout::new(p);
    let (values, grad) = harmonics_at(basis, p, frame.scaled(x));
    let sum = |values: &[f64]| -> f64 {
        (0..layout.len())
            .map(|i| {
                let w = if layout.nm(i).1 == 0 { 1.0 } else { 2.0 };
                w * (coefficients[i] * values[i]).abs()
            })
            .sum()
    };
    let r = frame.radius;
    let gradient = grad.iter().map(|g| sum(g).powi(2)).sum::<f64>().sqrt();
    (sum(&values) / r, gradient / (r * r))
}

/// The P2P terms Σ |q| / |x − y| and Σ |q| / |x − y|² at `x`, coincident points
/// excluded.
pub fn p2p_terms(sources: &[[f64; 3]], charges: &[f64], x: [f64; 3]) -> (f64, f64) {
    let mut terms = (0.0, 0.0);
    for (y, q) in sources.iter().zip(charges) {
        if *y == x {
            continue;
        }
        let r = norm(sub(x, *y));
        terms.0 += q.abs() / r;
        terms.1 += q.abs() / (r * r);
    }
    terms
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|k| a[k] - b[k])
}

pub fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The tables built so far, by resolved strategy and degree.
type Built = Mutex<HashMap<(M2lStrategy, usize), &'static Tables<f64>>>;

/// The f64 tables of `strategy` at degree `p`, built once per test binary.
pub fn tables(strategy: M2lStrategy, p: usize) -> &'static Tables<f64> {
    static CACHE: OnceLock<Built> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mut cache = cache.lock().unwrap();
    cache
        .entry((strategy.resolve(p), p))
        .or_insert_with(|| Box::leak(Box::new(Tables::build(p, strategy))))
}

/// A potentials-and-gradients operator with the f64 tables of `strategy` at `p`, and the
/// default P2P kernel (`P2pChoice::Auto`).
pub fn operator(strategy: M2lStrategy, p: usize, max_leaf_points: usize) -> LaplaceOperator<f64> {
    LaplaceOperator::new(tables(strategy, p).clone(), true, max_leaf_points)
}

/// The P2P kernels a test runs: `Reference` and every ISA this machine offers, which it
/// prints (`--nocapture`), so a test report can list the ISAs that ran.
pub fn p2p_choices(test: &str) -> Vec<P2pChoice> {
    let choices: Vec<P2pChoice> = std::iter::once(P2pChoice::Reference)
        .chain(Isa::available().map(P2pChoice::Isa))
        .collect();
    let names: Vec<String> = choices.iter().map(ToString::to_string).collect();
    eprintln!("{test}: P2P kernels {}", names.join(", "));
    choices
}

/// The dense f64 matrices of the three translations at degree p, for the terms.
pub struct Dense {
    pub m2m: MatrixSet<f64>,
    pub l2l: MatrixSet<f64>,
    pub m2l: MatrixSet<f64>,
}

/// The dense matrices at `p`, read off the cached dense [`Tables`] column by column
/// (each column is the table applied to a unit vector), once per test binary.
pub fn dense(p: usize) -> &'static Dense {
    static CACHE: OnceLock<Mutex<HashMap<usize, &'static Dense>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mut cache = cache.lock().unwrap();
    cache.entry(p).or_insert_with(|| {
        let tables = tables(M2lStrategy::Dense, p);
        let n = len(p);
        let mut scratch = tables.scratch();
        let m2m = read_matrices(n, 8, |o, x, y| tables.m2m(o, x, y, &mut scratch));
        let l2l = read_matrices(n, 8, |o, x, y| tables.l2l(o, x, y, &mut scratch));
        let m2l = read_matrices(n, 316, |d, x, y| tables.m2l(d, x, y, &mut scratch));
        Box::leak(Box::new(Dense { m2m, l2l, m2l }))
    })
}

/// The `count` matrices of size `n` of a family, column k of matrix i being
/// `apply(i, e_k, column)` into a zeroed column.
fn read_matrices(
    n: usize,
    count: usize,
    mut apply: impl FnMut(usize, &[f64], &mut [f64]),
) -> MatrixSet<f64> {
    let mut set = MatrixSet::zeros(n, count);
    let mut unit = vec![0.0; n];
    for i in 0..count {
        for k in 0..n {
            unit[k] = 1.0;
            apply(i, &unit, set.column_mut(i, k));
            unit[k] = 0.0;
        }
    }
    set
}

/// Tracks and prints the worst value of an error measure. It updates through a shared
/// reference, so that proptest bodies (`Fn` closures) can use it.
pub struct Worst {
    name: String,
    value: Cell<f64>,
}

impl Worst {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: Cell::new(0.0),
        }
    }

    /// Records `value` and returns it.
    pub fn update(&self, value: f64) -> f64 {
        assert!(!value.is_nan(), "{}: NaN", self.name);
        self.value.set(self.value.get().max(value));
        value
    }

    /// Records `value` and asserts that it is at most `tolerance`.
    pub fn check(&self, value: f64, tolerance: f64, context: impl FnOnce() -> String) {
        self.update(value);
        assert!(
            value <= tolerance,
            "{}: {value:.3e} exceeds {tolerance:.1e} ({})",
            self.name,
            context()
        );
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

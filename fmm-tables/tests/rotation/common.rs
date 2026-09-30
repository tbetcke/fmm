//! Helpers of the rotation-table tests: their degrees, levels and tolerances, the cached
//! rotation and dense tables, the reference operators, random V-list pairs and child
//! boxes in the dyadic domain, and the comparison of one table application with a
//! reference result. The shared helpers and the error measure are in `support`,
//! re-exported here.

use std::sync::OnceLock;

use nd_fmm_ref::{Frame, Workspace, direct, rotation};
use nd_fmm_tables::rotation::{Operator, RotationScratch, ShiftTables};
use nd_fmm_tables::symmetry::Expansion;
use nd_fmm_tables::{L2lTables, M2lTables, M2mTables, MatrixSet, RotationTables};

pub use crate::support::*;

/// Largest degree of the debug run.
pub const P_DEBUG: usize = 8;

/// Degrees of the debug run.
pub const DEBUG_DEGREES: [usize; 6] = [0, 1, 2, 3, 5, P_DEBUG];

/// Largest degree tested for M2L in f64 (CONVENTIONS §3.9).
pub const P_MAX_M2L: usize = 20;

/// Largest degree tested for M2M and L2L in f64 (CONVENTIONS §3.9).
pub const P_MAX: usize = 30;

/// Largest degree tested in f32 (CONVENTIONS §3.9).
pub const P_MAX_F32: usize = 8;

/// Tolerance against `nd_fmm_ref::rotation`, per degree relative to the terms (brief T6).
pub const REFERENCE_TOL: f64 = 1e-14;

/// Tolerance against `direct` for p ≤ 20, per degree relative to the terms (C2.3, the
/// Phase 1 gate).
pub const DIRECT_TOL: f64 = 1e-13;

/// Tolerance of M2M and L2L for 20 < p ≤ 30, per degree relative to the terms (brief T6,
/// as in Phase 1 T6: the accuracy of the rotation blocks at n ≤ 30).
pub const HIGH_DEGREE_TOL: f64 = 1e-11;

/// Levels of the level-independence test: the level of the V-list pair for M2L, and of
/// the child for M2M and L2L (the parent is one level up).
pub const LEVELS: [u32; 3] = [2, 9, 16];

/// The three families.
pub const OPERATORS: [Operator; 3] = [Operator::M2l, Operator::M2m, Operator::L2l];

/// The signature of the `nd-fmm-ref` translations in f64.
pub type Translate = fn(usize, &Frame<f64>, &Frame<f64>, &mut Workspace<f64>, &[f64], &mut [f64]);

/// The rotation tables of degree p ≤ [`P_DEBUG`] in f64, built once per test binary.
pub fn tables(p: usize) -> &'static RotationTables<f64> {
    static TABLES: [OnceLock<RotationTables<f64>>; P_DEBUG + 1] =
        [const { OnceLock::new() }; P_DEBUG + 1];
    TABLES[p].get_or_init(|| RotationTables::build(p))
}

/// The dense tables of T3 and T4 of degree p, one per family.
pub struct Dense {
    pub m2m: MatrixSet<f64>,
    pub l2l: MatrixSet<f64>,
    pub m2l: MatrixSet<f64>,
}

impl Dense {
    /// Builds the dense tables of degree `p`.
    pub fn build(p: usize) -> Self {
        Self {
            m2m: M2mTables::<f64>::build(p).matrices().clone(),
            l2l: L2lTables::<f64>::build(p).matrices().clone(),
            m2l: M2lTables::<f64>::build(p).matrices().clone(),
        }
    }

    /// The dense matrices of `op`.
    pub fn of(&self, op: Operator) -> &MatrixSet<f64> {
        match op {
            Operator::M2m => &self.m2m,
            Operator::L2l => &self.l2l,
            Operator::M2l => &self.m2l,
        }
    }
}

/// The dense tables of degree p ≤ [`P_DEBUG`], built once per test binary.
pub fn dense(p: usize) -> &'static Dense {
    static TABLES: [OnceLock<Dense>; P_DEBUG + 1] = [const { OnceLock::new() }; P_DEBUG + 1];
    TABLES[p].get_or_init(|| Dense::build(p))
}

/// The weighting of the output of `op`.
pub fn output_kind(op: Operator) -> Kind {
    kind(op.output())
}

/// The weighting of the input of `op`.
pub fn input_kind(op: Operator) -> Kind {
    kind(op.input())
}

/// The weighting of coefficients of `expansion`.
pub fn kind(expansion: Expansion) -> Kind {
    match expansion {
        Expansion::Multipole => Kind::Multipole,
        Expansion::Local => Kind::Local,
    }
}

/// The `nd_fmm_ref::rotation` operator of `op`.
pub fn reference(op: Operator) -> Translate {
    match op {
        Operator::M2m => rotation::m2m::<f64>,
        Operator::L2l => rotation::l2l::<f64>,
        Operator::M2l => rotation::m2l::<f64>,
    }
}

/// The `nd_fmm_ref::direct` operator of `op`.
pub fn direct_operator(op: Operator) -> Translate {
    match op {
        Operator::M2m => direct::m2m::<f64>,
        Operator::L2l => direct::l2l::<f64>,
        Operator::M2l => direct::m2l::<f64>,
    }
}

/// Applies entry `t` of `tables` to `x` from a zero output.
pub fn run(tables: &ShiftTables<f64>, t: usize, x: &[f64]) -> Vec<f64> {
    let mut scratch = RotationScratch::new(tables.p());
    let mut y = vec![0.0; x.len()];
    tables.apply(t, x, &mut y, &mut scratch);
    y
}

/// Applies `translate` from `from` to `to` to `x` from a zero output.
pub fn translate(
    translate: Translate,
    p: usize,
    (from, to): (&Frame<f64>, &Frame<f64>),
    ws: &mut Workspace<f64>,
    x: &[f64],
) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    translate(p, from, to, ws, x, &mut y);
    y
}

/// Per-degree error of `got` against `want` for entry `t` of `op` with input `x`,
/// relative to the terms of the dense matrix `t` of `dense` (the measure of `main`).
pub fn error(
    op: Operator,
    dense: &MatrixSet<f64>,
    t: usize,
    x: &[f64],
    got: &[f64],
    want: &[f64],
) -> f64 {
    let p = dense.n().isqrt() - 1;
    let scale = terms(dense, t, x);
    degree_error(output_kind(op), p, got, want, &scale)
}

/// A random V-list pair on `level` at offset `d`: the (source, target) box indices with
/// target − source = d. Along each axis the target is uniform among the boxes whose
/// source lies in the domain.
pub fn random_pair(level: u32, d: [i64; 3], rng: &mut SplitMix64) -> ([u64; 3], [u64; 3]) {
    let n = 1i64 << level;
    let mut source = [0; 3];
    let mut target = [0; 3];
    for a in 0..3 {
        let (lo, hi) = (d[a].max(0), (n - 1).min(n - 1 + d[a]));
        assert!(lo <= hi, "level {level} has no pair at offset {d:?}");
        let t = lo + rng.below((hi - lo + 1) as u64) as i64;
        target[a] = t as u64;
        source[a] = (t - d[a]) as u64;
    }
    (source, target)
}

/// The index on level l + 1 of child `o` of the box `index` on level l:
/// (2i + x, 2j + y, 2k + z) with o = 4x + 2y + z (CONVENTIONS §3.12).
pub fn child_index(index: [u64; 3], o: usize) -> [u64; 3] {
    let bits = [(o >> 2) & 1, (o >> 1) & 1, o & 1];
    core::array::from_fn(|a| 2 * index[a] + bits[a] as u64)
}

/// The (input, output) frames of entry `t` of `op` in the dyadic domain: a random
/// V-list pair on `level` for M2L, and for M2M and L2L a random parent on level
/// `level − 1` and its child `t` on `level`. Asserts that the §3.12 parameters are
/// those of the canonical frames, bit for bit.
pub fn level_frames(
    op: Operator,
    level: u32,
    t: usize,
    rng: &mut SplitMix64,
) -> (Frame<f64>, Frame<f64>) {
    let (from, to) = match op {
        Operator::M2l => {
            let d = op.direction(t);
            let (source, target) = random_pair(level, d, rng);
            (DYADIC.frame(level, source), DYADIC.frame(level, target))
        }
        Operator::M2m | Operator::L2l => {
            let parent_level = level - 1;
            let index = [0; 3].map(|_| rng.below(1 << parent_level));
            let parent = DYADIC.frame(parent_level, index);
            let child = DYADIC.frame(level, child_index(index, t));
            if op == Operator::M2m {
                (child, parent)
            } else {
                (parent, child)
            }
        }
    };
    let (canonical_from, canonical_to) = op.frames(t);
    let shift = |a: &Frame<f64>, b: &Frame<f64>, r: f64| -> [f64; 3] {
        core::array::from_fn(|i| (b.centre[i] - a.centre[i]) / r)
    };
    assert_eq!(
        shift(&from, &to, to.radius),
        shift(&canonical_from, &canonical_to, canonical_to.radius),
        "{op:?} level {level}, entry {t}"
    );
    assert_eq!(
        to.radius / from.radius,
        canonical_to.radius / canonical_from.radius
    );
    (from, to)
}

//! Helpers of the M2L-table tests: their degrees, levels and tolerances, the class
//! representatives of CONVENTIONS §3.12, random V-list pairs in the dyadic domain, and
//! the cached tables. The shared helpers and the error measure are in `support`,
//! re-exported here.

use std::sync::OnceLock;

use nd_fmm_tables::geometry::m2l_offset_index;
use nd_fmm_tables::m2l::build_matrices;
use nd_fmm_tables::{M2lTables, MatrixSet};

pub use crate::support::*;

/// Largest degree of the full tables in the debug run.
pub const P_DEBUG: usize = 8;

/// Degrees of the full tables in the debug run.
pub const DEBUG_DEGREES: [usize; 7] = [0, 1, 2, 3, 4, 5, P_DEBUG];

/// Degree of the partial table of the debug run, [`subset`]: a full table at p = 12
/// takes about 45 s to build in debug mode.
pub const P_SUBSET: usize = 12;

/// Largest degree tested for M2L in f64 (CONVENTIONS §3.9).
pub const P_MAX_M2L: usize = 20;

/// Largest degree tested in f32 (CONVENTIONS §3.9).
pub const P_MAX_F32: usize = 8;

/// Tolerance of a table against `direct` in a dyadic domain, per degree relative to the
/// terms (C2.2).
pub const LEVEL_TOL: f64 = 1e-14;

/// Tolerance of the physical chains and of the monopole column, per degree (or per
/// potential) relative to the terms.
pub const CHAIN_TOL: f64 = 1e-13;

/// Levels of the level-independence test (C2.2).
pub const LEVELS: [u32; 3] = [2, 9, 16];

/// The representatives of the 16 symmetry classes, in class order (CONVENTIONS §3.12,
/// "Symmetry classes"): the offsets with 0 ≤ d_x ≤ d_y ≤ d_z.
pub const CLASS_REPRESENTATIVES: [[i64; 3]; 16] = [
    [0, 0, 2],
    [0, 0, 3],
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 2],
    [0, 2, 3],
    [0, 3, 3],
    [1, 1, 2],
    [1, 1, 3],
    [1, 2, 2],
    [1, 2, 3],
    [1, 3, 3],
    [2, 2, 2],
    [2, 2, 3],
    [2, 3, 3],
    [3, 3, 3],
];

/// The offsets of the chain against the direct sum (brief T4).
pub const DIRECT_SUM_OFFSETS: [[i64; 3]; 3] = [[2, 0, 0], [2, 2, 2], [3, 3, 3]];

/// The M2L tables of degree p ≤ [`P_DEBUG`] in f64, built once per test binary.
pub fn m2l(p: usize) -> &'static M2lTables<f64> {
    static TABLES: [OnceLock<M2lTables<f64>>; P_DEBUG + 1] =
        [const { OnceLock::new() }; P_DEBUG + 1];
    TABLES[p].get_or_init(|| M2lTables::build(p))
}

/// Some matrices of the M2L table of one degree, built with [`build_matrices`].
pub struct Subset {
    /// The table indices of the matrices, in order.
    pub indices: Vec<usize>,
    /// Matrix t is that of table index `indices[t]`.
    pub matrices: MatrixSet<f64>,
}

impl Subset {
    /// Builds the matrices of degree `p` of the offsets `offsets`.
    pub fn build(p: usize, offsets: &[[i64; 3]]) -> Self {
        let indices: Vec<usize> = offsets
            .iter()
            .map(|&d| m2l_offset_index(d).expect("a V-list offset"))
            .collect();
        let matrices = build_matrices(p, &indices);
        Self { indices, matrices }
    }

    /// The position in [`Subset::matrices`] of table index `index`.
    pub fn position(&self, index: usize) -> usize {
        self.indices
            .iter()
            .position(|&i| i == index)
            .expect("an index of the subset")
    }
}

/// The offsets of [`subset`]: the 16 class representatives, then the offsets of
/// [`DIRECT_SUM_OFFSETS`] that are not representatives.
pub fn subset_offsets() -> Vec<[i64; 3]> {
    let mut offsets = CLASS_REPRESENTATIVES.to_vec();
    offsets.extend(
        DIRECT_SUM_OFFSETS
            .iter()
            .filter(|d| !CLASS_REPRESENTATIVES.contains(d)),
    );
    offsets
}

/// The matrices at p = [`P_SUBSET`] of [`subset_offsets`], built once per test binary.
pub fn subset() -> &'static Subset {
    static SUBSET: OnceLock<Subset> = OnceLock::new();
    SUBSET.get_or_init(|| Subset::build(P_SUBSET, &subset_offsets()))
}

/// A random V-list pair on `level` at offset `d`: the (source, target) box indices with
/// target − source = d. Along each axis the target is uniform among the boxes whose
/// source lies in the domain.
///
/// # Panics
///
/// If the level has too few boxes for d (level < 2).
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

/// The degree p of a set of (p + 1)² × (p + 1)² matrices.
pub fn degree(set: &MatrixSet<f64>) -> usize {
    set.n().isqrt() - 1
}

//! Helpers of the octant-table tests: their degrees, levels and tolerances, the generic
//! domain, child box indices and the cached tables. The shared helpers and the error
//! measure are in `support`, re-exported here.

use std::sync::OnceLock;

use nd_fmm_tables::{L2lTables, M2mTables};

pub use crate::support::*;

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

/// The generic domain a = (0.1, −2.3, 7.9), w = 0.37: neither is exactly representable,
/// so centres are rounded.
pub const GENERIC: Domain = Domain {
    corner: [0.1, -2.3, 7.9],
    side: 0.37,
};

impl Domain {
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

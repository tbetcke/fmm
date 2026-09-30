//! Dense M2L tables for the 316 V-list offsets (CONVENTIONS §3.11, §3.12; C2.2).
//!
//! The table holds one real (p + 1)² × (p + 1)² matrix per V-list offset d, in the
//! lexicographic offset order of §3.12 (table index [`m2l_offset_index`]), column-major
//! and contiguous in a [`MatrixSet`] (CONVENTIONS §3.12, "Matrix layout"). Column k of
//! matrix index(d) is `nd_fmm_ref::direct::m2l` applied to the k-th unit vector at the
//! canonical frames of [`m2l_frames`]: the source (0, 1) and the target (2d, 1), so
//! the M2L of §3.11 has b = 2d and σ = 1. It is built in f64 and rounded entry by entry
//! to the stored precision. By the scaling of §3.7 the same matrix is the M2L from a
//! box to the box at offset d on the same level, on every level of every cubic domain.
//!
//! Convergence: every V-list offset has 4 ≤ |b| = 2|d| ≤ 6√3 ≈ 10.4, beyond the
//! convergence limit |b| > √3 (1 + σ) = 2√3 ≈ 3.46 of §3.11, so the double expansion
//! converges for sources and targets anywhere in their boxes; the truncation error at
//! degree p is bounded in §3.11 ("M2L", Truncation).
//!
//! Storage: 316 (p + 1)⁴ reals. In f64 that is about 17 MB at p = 8, 211 MB at p = 16
//! and 492 MB at p = 20. The symmetry-reduced form, 16 class matrices and per-offset
//! coefficient transforms (§3.12, "Symmetry classes"), comes in Phase 2 / T5.
//! [`build_matrices`] builds the matrices of chosen offsets only, bit-identical to
//! those of the full table.
//!
//! ```
//! use nd_fmm_ref::{Workspace, direct};
//! use nd_fmm_tables::geometry::m2l_frames;
//! use nd_fmm_tables::m2l::M2lTables;
//!
//! let p = 3;
//! let tables = M2lTables::<f64>::build(p);
//! let index = tables.index([2, -1, 0]).unwrap();
//! let multipole: Vec<f64> = (0..16).map(|i| 1.0 / (1.0 + i as f64)).collect();
//! let mut from_table = vec![0.0; 16];
//! tables.apply(index, &multipole, &mut from_table);
//!
//! let (source, target) = m2l_frames(index);
//! let mut from_direct = vec![0.0; 16];
//! direct::m2l(p, &source, &target, &mut Workspace::new(p), &multipole, &mut from_direct);
//! for (a, b) in from_table.iter().zip(&from_direct) {
//!     assert!((a - b).abs() < 1e-15);
//! }
//! ```

use nd_fmm_math::{Layout, RealScalar};
use nd_fmm_ref::{Workspace, direct};

use crate::MatrixSet;
use crate::geometry::{M2L_OFFSET_COUNT, m2l_frames, m2l_offset_index};

/// Builds the M2L matrices of degree `p` for the given table indices, in the given
/// order: matrix t of the result is matrix `indices[t]` of [`M2lTables::build`]`(p)`,
/// bit for bit (CONVENTIONS §3.12, "Canonical frames and level independence").
///
/// Column k of each matrix is `direct::m2l` of the k-th unit vector at
/// [`m2l_frames`]`(index)`, into a zeroed f64 column, rounded entry by entry to `T`.
/// Building a subset, such as the class representatives of §3.12, costs
/// `indices.len()` (p + 1)² calls of `direct::m2l` instead of 316 (p + 1)². Uses one
/// [`Workspace`]; allocates.
///
/// # Panics
///
/// If an index is 316 or more.
pub fn build_matrices<T: RealScalar>(p: usize, indices: &[usize]) -> MatrixSet<T> {
    let n = Layout::new(p).len();
    let mut ws = Workspace::new(p);
    let mut unit = vec![0.0; n];
    let mut column = vec![0.0; n];
    let mut matrices = MatrixSet::<T>::zeros(n, indices.len());
    for (t, &index) in indices.iter().enumerate() {
        let (source, target) = m2l_frames(index);
        for k in 0..n {
            unit[k] = 1.0;
            column.fill(0.0);
            direct::m2l(p, &source, &target, &mut ws, &unit, &mut column);
            unit[k] = 0.0;
            // The rounding of `MatrixSet::cast`, one column at a time.
            for (a, &v) in matrices.column_mut(t, k).iter_mut().zip(&column) {
                *a = T::from_f64(v);
            }
        }
    }
    matrices
}

/// The dense M2L tables: for each of the 316 V-list offsets d, the matrix of M2L from a
/// source box to the target box at offset d on the same level (CONVENTIONS §3.11,
/// "M2L"; §3.12).
///
/// Matrix index(d) is built at the canonical frames of [`m2l_frames`]: the source
/// (0, 1) to the target (2d, 1), so b = 2d and σ = 1, with 4 ≤ |b| ≤ 6√3 for every
/// offset (module documentation). No block is zero: output degree j uses every input
/// degree n ≤ p, through the irregular harmonics of b of degree n + j ≤ 2p.
///
/// Storage: 316 (p + 1)⁴ reals, in the offset order of §3.12, each matrix column-major
/// with output = A · input ([`MatrixSet`]); in f64 about 17 MB at p = 8, 211 MB at
/// p = 16 and 492 MB at p = 20. Building costs 316 (p + 1)² calls of `direct::m2l`,
/// each O(p⁴), so O(p⁶) in all. Phase 2 / T5 provides the 16-class form.
#[derive(Clone, Debug, PartialEq)]
pub struct M2lTables<T: RealScalar> {
    p: usize,
    matrices: MatrixSet<T>,
}

impl<T: RealScalar> M2lTables<T> {
    /// Builds the M2L tables of degree `p`: column k of matrix `index` is `direct::m2l`
    /// of the k-th unit vector at [`m2l_frames`]`(index)`, into a zeroed column,
    /// computed in f64 and then rounded to `T`. This is [`build_matrices`] for every
    /// table index in order. Uses one [`Workspace`]; allocates the table and O((p + 1)²)
    /// scratch, so the peak memory is the table itself.
    pub fn build(p: usize) -> Self {
        let indices: Vec<usize> = (0..M2L_OFFSET_COUNT).collect();
        Self {
            p,
            matrices: build_matrices(p, &indices),
        }
    }

    /// Returns the degree p of input and output.
    #[inline]
    pub fn p(&self) -> usize {
        self.p
    }

    /// Returns the 316 matrices, in the offset order of CONVENTIONS §3.12.
    #[inline]
    pub fn matrices(&self) -> &MatrixSet<T> {
        &self.matrices
    }

    /// Returns the table index of the V-list offset `offset` = d, or `None` if d is not
    /// in {−3..3}³ \ {−1..1}³. Delegates to [`m2l_offset_index`] (CONVENTIONS §3.12,
    /// "V-list offsets"); d = index(target) − index(source) in the box-index units of
    /// the level.
    #[inline]
    pub fn index(&self, offset: [i64; 3]) -> Option<usize> {
        m2l_offset_index(offset)
    }

    /// Adds the M2L of the source multipole expansion `multipole` to the local
    /// expansion `local` of the target box at the offset with table index `index`
    /// (CONVENTIONS §3.11). Both are scaled coefficients of degree p, (p + 1)² reals in
    /// the storage of §3.6. Accumulates and allocates nothing; the summation order is
    /// that of [`MatrixSet::apply`].
    ///
    /// # Panics
    ///
    /// If `index >= 316`, or `multipole` or `local` does not have length (p + 1)².
    #[inline]
    pub fn apply(&self, index: usize, multipole: &[T], local: &mut [T]) {
        self.matrices.apply(index, multipole, local);
    }
}

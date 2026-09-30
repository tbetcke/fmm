//! M2M and L2L tables for the 8 child octants (CONVENTIONS §3.11, §3.12; C2.1).
//!
//! Each table holds one real (p + 1)² × (p + 1)² matrix per child index o, in
//! child-index order, column-major and contiguous in a [`MatrixSet`]: 8 (p + 1)⁴ reals
//! per family (CONVENTIONS §3.12, "Matrix layout"). Column k of matrix o is the
//! `nd_fmm_ref::direct` operator applied to the k-th unit vector at the canonical
//! frames of [`geometry`](crate::geometry), built in f64 and rounded entry by entry to
//! the stored precision. By the scaling of §3.7 the same matrix is the operator from a
//! child to its parent (M2M) or from a parent to its child (L2L) on every level of every
//! cubic domain.
//!
//! ```
//! use nd_fmm_ref::{Workspace, direct};
//! use nd_fmm_tables::geometry::m2m_frames;
//! use nd_fmm_tables::octant::M2mTables;
//!
//! let p = 4;
//! let tables = M2mTables::<f64>::build(p);
//! let input: Vec<f64> = (0..25).map(|i| 1.0 / (1.0 + i as f64)).collect();
//! let mut from_table = vec![0.0; 25];
//! tables.apply(3, &input, &mut from_table);
//!
//! let (child, parent) = m2m_frames(3);
//! let mut from_direct = vec![0.0; 25];
//! direct::m2m(p, &child, &parent, &mut Workspace::new(p), &input, &mut from_direct);
//! for (a, b) in from_table.iter().zip(&from_direct) {
//!     assert!((a - b).abs() < 1e-14);
//! }
//! ```

use nd_fmm_math::{Layout, RealScalar};
use nd_fmm_ref::{Frame, Workspace, direct};

use crate::MatrixSet;
use crate::geometry::{OCTANT_COUNT, l2l_frames, m2m_frames};

/// The signature of `direct::m2m` and `direct::l2l` in f64.
type Translate = fn(usize, &Frame<f64>, &Frame<f64>, &mut Workspace<f64>, &[f64], &mut [f64]);

/// The M2M tables: for each child index o, the matrix of M2M from child o to its parent
/// (CONVENTIONS §3.11, "M2M"; §3.12).
///
/// Matrix o is built at the canonical frames of [`m2m_frames`]: the child (½ s_o, ½) to
/// the parent (0, 1), so b = ½ s_o and ρ = ½. Output degree j uses input degrees k ≤ j
/// only, so every block above the block diagonal is exactly zero.
///
/// Storage: 8 (p + 1)⁴ reals, in child-index order, each matrix column-major with
/// output = A · input ([`MatrixSet`]). Building costs 8 (p + 1)² calls of
/// `direct::m2m`, each O(p⁴), so O(p⁶) in all.
#[derive(Clone, Debug, PartialEq)]
pub struct M2mTables<T: RealScalar> {
    pub(crate) inner: OctantTables<T>,
}

impl<T: RealScalar> M2mTables<T> {
    /// Builds the M2M tables of degree `p`: column k of matrix o is `direct::m2m` of the
    /// k-th unit vector at [`m2m_frames`]`(o)`, into a zeroed column, computed in f64
    /// and then rounded to `T`. Uses one [`Workspace`]; allocates.
    pub fn build(p: usize) -> Self {
        Self {
            inner: OctantTables::build(p, direct::m2m::<f64>, m2m_frames),
        }
    }

    /// Returns the degree p of input and output.
    #[inline]
    pub fn p(&self) -> usize {
        self.inner.p
    }

    /// Returns the 8 matrices, in child-index order.
    #[inline]
    pub fn matrices(&self) -> &MatrixSet<T> {
        &self.inner.matrices
    }

    /// Adds the M2M of the child multipole expansion `input`, in child octant `o`, to
    /// the parent multipole expansion `output` (CONVENTIONS §3.11). Both are scaled
    /// coefficients of degree p, (p + 1)² reals in the storage of §3.6. Accumulates and
    /// allocates nothing; the summation order is that of [`MatrixSet::apply`].
    ///
    /// # Panics
    ///
    /// If `o >= 8`, or `input` or `output` does not have length (p + 1)².
    #[inline]
    pub fn apply(&self, o: usize, input: &[T], output: &mut [T]) {
        self.inner.matrices.apply(o, input, output);
    }
}

/// The L2L tables: for each child index o, the matrix of L2L from a parent to its child
/// o (CONVENTIONS §3.11, "L2L"; §3.12).
///
/// Matrix o is built at the canonical frames of [`l2l_frames`]: the parent (0, 1) to
/// the child (½ s_o, ½), so t = ½ s_o and σ = ½. Output degree j uses input degrees
/// n ≥ j only, so every block below the block diagonal is exactly zero.
///
/// Storage: 8 (p + 1)⁴ reals, in child-index order, each matrix column-major with
/// output = A · input ([`MatrixSet`]). Building costs 8 (p + 1)² calls of
/// `direct::l2l`, each O(p⁴), so O(p⁶) in all.
#[derive(Clone, Debug, PartialEq)]
pub struct L2lTables<T: RealScalar> {
    pub(crate) inner: OctantTables<T>,
}

impl<T: RealScalar> L2lTables<T> {
    /// Builds the L2L tables of degree `p`: column k of matrix o is `direct::l2l` of the
    /// k-th unit vector at [`l2l_frames`]`(o)`, into a zeroed column, computed in f64
    /// and then rounded to `T`. Uses one [`Workspace`]; allocates.
    pub fn build(p: usize) -> Self {
        Self {
            inner: OctantTables::build(p, direct::l2l::<f64>, l2l_frames),
        }
    }

    /// Returns the degree p of input and output.
    #[inline]
    pub fn p(&self) -> usize {
        self.inner.p
    }

    /// Returns the 8 matrices, in child-index order.
    #[inline]
    pub fn matrices(&self) -> &MatrixSet<T> {
        &self.inner.matrices
    }

    /// Adds the L2L of the parent local expansion `input` to the local expansion
    /// `output` of child octant `o` (CONVENTIONS §3.11). Both are scaled coefficients of
    /// degree p, (p + 1)² reals in the storage of §3.6. Accumulates and allocates
    /// nothing; the summation order is that of [`MatrixSet::apply`].
    ///
    /// # Panics
    ///
    /// If `o >= 8`, or `input` or `output` does not have length (p + 1)².
    #[inline]
    pub fn apply(&self, o: usize, input: &[T], output: &mut [T]) {
        self.inner.matrices.apply(o, input, output);
    }
}

/// The implementation shared by [`M2mTables`] and [`L2lTables`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OctantTables<T: RealScalar> {
    pub(crate) p: usize,
    pub(crate) matrices: MatrixSet<T>,
}

impl<T: RealScalar> OctantTables<T> {
    /// Applies `translate` at `frames(o)` to each unit vector, column by column and
    /// octant by octant, in f64, and rounds the result to `T`.
    fn build(
        p: usize,
        translate: Translate,
        frames: fn(usize) -> (Frame<f64>, Frame<f64>),
    ) -> Self {
        let n = Layout::new(p).len();
        let mut ws = Workspace::new(p);
        let mut unit = vec![0.0; n];
        let mut matrices = MatrixSet::<f64>::zeros(n, OCTANT_COUNT);
        for o in 0..OCTANT_COUNT {
            let (from, to) = frames(o);
            for k in 0..n {
                unit[k] = 1.0;
                translate(p, &from, &to, &mut ws, &unit, matrices.column_mut(o, k));
                unit[k] = 0.0;
            }
        }
        Self {
            p,
            matrices: matrices.cast(),
        }
    }
}

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
//! and 492 MB at p = 20. [`build_matrices`] builds the matrices of chosen offsets only,
//! bit-identical to those of the full table.
//!
//! [`M2lClasses`] is the symmetry-reduced form (§3.12, "Symmetry classes"): the 16
//! class matrices and the coefficient transforms of the cube group
//! ([`crate::symmetry`]), M2L(d) = T_L(P) M2L(representative) T_M(Pᵀ). It needs
//! 16 (p + 1)⁴ reals plus the transforms, about 34 MB in f64 at p = 20, and applies
//! each offset without expanding, or [`M2lClasses::expand`]s to the dense tables.
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
use crate::symmetry::{CoefficientTransform, SignedPermutation, class_of, class_representatives};

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
/// each O(p⁴), so O(p⁶) in all. [`M2lClasses`] is the 16-class form.
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

/// The symmetry-reduced M2L tables: the matrices of the 16 class representatives and,
/// for each of the 316 V-list offsets, its class and group element (CONVENTIONS §3.12,
/// "Symmetry classes").
///
/// For the offset d with class representative d₀ and group element P, d = P d₀ and
/// M2L(d) = T_L(P) M2L(d₀) T_M(Pᵀ) (§3.12, "Operator identities"), with the coefficient
/// transforms T_M and T_L of [`CoefficientTransform`]. The class of d and P follow the
/// rule of §3.12 ([`class_of`]), so the class form is reproducible. The class matrices
/// are built as in [`M2lTables`] and are bit-identical to its matrices of the
/// representatives.
///
/// Storage: 16 (p + 1)⁴ reals for the class matrices, in class order in a
/// [`MatrixSet`], plus T_M(P) and T_L(P) for all 48 elements, 96 · `blocks_len(p)` =
/// 32 (p + 1)(2p + 1)(2p + 3) reals, and 316 (class, element) pairs. In f64 that is
/// about 1.6 MB at p = 8, 15.7 MB at p = 16 and 34.4 MB at p = 20, against 17 MB,
/// 211 MB and 492 MB for [`M2lTables`].
///
/// [`M2lClasses::apply`] costs (p + 1)⁴ multiply–adds for the class matrix plus two
/// transforms of `blocks_len(p)` = (p + 1)(2p + 1)(2p + 3)/3 multiply–adds each, O(p³).
/// Building costs 16 (p + 1)² calls of `direct::m2l` instead of 316 (p + 1)².
#[derive(Clone, Debug, PartialEq)]
pub struct M2lClasses<T: RealScalar> {
    p: usize,
    matrices: MatrixSet<T>,
    /// T_M(P) of element g at position g.
    multipole: Vec<CoefficientTransform<T>>,
    /// T_L(P) of element g at position g.
    local: Vec<CoefficientTransform<T>>,
    /// The class and the group element of each offset, in table-index order.
    offsets: Vec<(usize, SignedPermutation)>,
}

impl<T: RealScalar> M2lClasses<T> {
    /// Builds the class form of degree `p`: the matrices of the 16 representatives with
    /// [`build_matrices`], T_M(P) and T_L(P) of all 48 elements with
    /// [`CoefficientTransform::new`], and the class and element of every offset with
    /// [`class_of`]. Everything is computed in f64 and then rounded to `T`, so the class
    /// form in f32 is that of f64 rounded entry by entry. Allocates.
    pub fn build(p: usize) -> Self {
        let indices = class_representatives()
            .map(|d| m2l_offset_index(d).expect("a representative is a V-list offset"));
        let all = SignedPermutation::all();
        Self {
            p,
            matrices: build_matrices(p, &indices),
            multipole: all
                .iter()
                .map(|&e| CoefficientTransform::multipole(p, e))
                .collect(),
            local: all
                .iter()
                .map(|&e| CoefficientTransform::local(p, e))
                .collect(),
            offsets: (0..M2L_OFFSET_COUNT).map(class_of).collect(),
        }
    }

    /// Returns the degree p of input and output.
    #[inline]
    pub fn p(&self) -> usize {
        self.p
    }

    /// Returns the 16 class matrices M2L(d₀), in class order (CONVENTIONS §3.12).
    #[inline]
    pub fn matrices(&self) -> &MatrixSet<T> {
        &self.matrices
    }

    /// Returns the table index of the V-list offset `offset` = d, or `None` if d is not
    /// in {−3..3}³ \ {−1..1}³ ([`m2l_offset_index`]).
    #[inline]
    pub fn index(&self, offset: [i64; 3]) -> Option<usize> {
        m2l_offset_index(offset)
    }

    /// Returns the class of the offset with table index `index`.
    ///
    /// # Panics
    ///
    /// If `index >= 316`.
    #[inline]
    pub fn class(&self, index: usize) -> usize {
        self.offsets[index].0
    }

    /// Returns the group element P of the offset with table index `index`, with
    /// d = P d₀ for the representative d₀ of its class.
    ///
    /// # Panics
    ///
    /// If `index >= 316`.
    #[inline]
    pub fn element(&self, index: usize) -> SignedPermutation {
        self.offsets[index].1
    }

    /// Returns T_M(P) of degree ≤ p for `element` = P.
    #[inline]
    pub fn multipole_transform(&self, element: SignedPermutation) -> &CoefficientTransform<T> {
        &self.multipole[element.index()]
    }

    /// Returns T_L(P) of degree ≤ p for `element` = P.
    #[inline]
    pub fn local_transform(&self, element: SignedPermutation) -> &CoefficientTransform<T> {
        &self.local[element.index()]
    }

    /// Adds the M2L of the source multipole expansion `multipole` to the local
    /// expansion `local` of the target box at the offset with table index `index`,
    /// without expanding: T_M(Pᵀ), the class matrix and T_L(P) are applied in turn,
    /// the first two into `scratch` (CONVENTIONS §3.12, "Symmetry classes").
    ///
    /// Both expansions are scaled coefficients of degree p, (p + 1)² reals in the
    /// storage of §3.6. Accumulates into `local` and allocates nothing. Costs
    /// (p + 1)⁴ + 2 `blocks_len(p)` multiply–adds, O(p⁴) plus two O(p³) transforms.
    ///
    /// # Panics
    ///
    /// If `index >= 316`, `scratch` is not of degree p, or `multipole` or `local` does
    /// not have length (p + 1)².
    pub fn apply(
        &self,
        index: usize,
        multipole: &[T],
        local: &mut [T],
        scratch: &mut M2lScratch<T>,
    ) {
        let (class, element) = self.offsets[index];
        let n = self.matrices.n();
        assert_eq!(
            scratch.rotated.len(),
            n,
            "`scratch` must be of degree p = {}",
            self.p
        );
        scratch.rotated.fill(T::zero());
        self.multipole[element.transpose().index()].apply(multipole, &mut scratch.rotated);
        scratch.translated.fill(T::zero());
        self.matrices
            .apply(class, &scratch.rotated, &mut scratch.translated);
        self.local[element.index()].apply(&scratch.translated, local);
    }

    /// Returns the dense tables: matrix index(d) is T_L(P) M2L(d₀) T_M(Pᵀ), formed in
    /// `T` (CONVENTIONS §3.12, "Symmetry classes").
    ///
    /// Each matrix is formed block by block: column (n, j) of A T_M(Pᵀ) is the sum over
    /// the slots i of degree n of T_M(Pᵀ)ᵢⱼ times column (n, i) of A, and T_L(P) is then
    /// applied to each column with [`CoefficientTransform::apply`]. That costs
    /// 2 (p + 1)² `blocks_len(p)` multiply–adds per offset, O(p⁵), against O(p⁶) for
    /// [`M2lTables::build`]. The result agrees with [`M2lTables::build`] up to rounding;
    /// the matrices of the representatives, whose element is I, are bit-identical.
    /// Allocates the tables and one (p + 1)⁴ buffer.
    pub fn expand(&self) -> M2lTables<T> {
        let layout = Layout::new(self.p);
        let n = layout.len();
        let mut dense = MatrixSet::<T>::zeros(n, M2L_OFFSET_COUNT);
        let mut right = vec![T::zero(); n * n];
        for (index, &(class, element)) in self.offsets.iter().enumerate() {
            let a = self.matrices.matrix(class);
            let t_m = &self.multipole[element.transpose().index()];
            right.fill(T::zero());
            for (degree, range) in layout.degrees() {
                let block = t_m.block(degree);
                let width = 2 * degree + 1;
                for (j, c) in range.clone().enumerate() {
                    let column = &mut right[c * n..(c + 1) * n];
                    for (i, k) in range.clone().enumerate() {
                        let t = block[i * width + j];
                        for (y, &x) in column.iter_mut().zip(&a[k * n..(k + 1) * n]) {
                            *y = *y + t * x;
                        }
                    }
                }
            }
            let t_l = &self.local[element.index()];
            for (c, column) in right.chunks_exact(n).enumerate() {
                t_l.apply(column, dense.column_mut(index, c));
            }
        }
        M2lTables {
            p: self.p,
            matrices: dense,
        }
    }

    /// Returns the same class form in precision `U`, every entry of the class matrices
    /// and the transforms rounded to nearest.
    pub fn cast<U: RealScalar>(&self) -> M2lClasses<U> {
        M2lClasses {
            p: self.p,
            matrices: self.matrices.cast(),
            multipole: self
                .multipole
                .iter()
                .map(CoefficientTransform::cast)
                .collect(),
            local: self.local.iter().map(CoefficientTransform::cast).collect(),
            offsets: self.offsets.clone(),
        }
    }
}

/// Caller-owned scratch of [`M2lClasses::apply`]: two coefficient vectors of degree p.
#[derive(Clone, Debug)]
pub struct M2lScratch<T: RealScalar> {
    /// T_M(Pᵀ) times the multipole.
    rotated: Vec<T>,
    /// The class matrix times `rotated`.
    translated: Vec<T>,
}

impl<T: RealScalar> M2lScratch<T> {
    /// Allocates the scratch for degree `p`, 2 (p + 1)² reals.
    pub fn new(p: usize) -> Self {
        let n = Layout::new(p).len();
        Self {
            rotated: vec![T::zero(); n],
            translated: vec![T::zero(); n],
        }
    }
}

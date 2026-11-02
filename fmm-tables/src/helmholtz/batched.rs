//! The batched M2L's tables of the Helmholtz family (Helmholtz CH3.2c;
//! docs/design/helmholtz-fmm-proposal.md §5.3, Batched, and §6.5, the split planes): Dense's
//! matrices in split planes, for a column kernel that takes square matrices.
//!
//! # Layout
//!
//! The M2L of offset d on level l, of degree p = p_l and n = (p + 1)² coefficients, is the
//! complex n × n matrix A = A_r + i A_i of [`DenseM2l`]. [`BatchedM2l`] stores it as two
//! real n × n matrices, A_r then A_i, each column-major: column j of A_r is rows 0..n of
//! Dense's column j, column j of A_i its rows n..2n, bit for bit. The 316 pairs of a level
//! lie one after the other in offset order, Dense's bytes: 316 · 2n² values a level.
//!
//! # Build and levels
//!
//! [`BatchedM2l::build`] builds Dense's matrices of each chosen level (`dense::matrices`,
//! Rotation's M2L of the unit multipoles in f64, rounded to T) and re-lays them offset by
//! offset. [`BatchedM2l::select`] and [`BatchedM2l::level_bytes`] are Dense's: the levels
//! within a byte budget from the finest up, a function of the degrees and the budget only
//! (proposal §2, constraint 9).
//!
//! # Application
//!
//! The tables are applied by the batched M2L of `nd-fmm-exec` (its Helmholtz operator's
//! module docs, "M2L"), which gives each column the chain of `MatrixSet::apply` with one
//! square matrix at a time: two products per pair for a real multipole (A_r into the local's
//! real plane, A_i into its imaginary one, bit for bit Dense's chain), four for a complex
//! one (Dense's to rounding: the real plane's products of a column come before the
//! imaginary plane's, where Dense's chain alternates them per k).

use nd_fmm_math::{Complex, Layout, RealScalar};

use super::dense::{DenseM2l, matrices};
use super::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use crate::geometry::M2L_OFFSET_COUNT;

/// The batched M2L's tables of the chosen levels of an octree (the
/// [module documentation](self)).
#[derive(Clone, Debug)]
pub struct BatchedM2l<T: RealScalar> {
    /// Per level its degree and, if chosen, its 316 pairs of planes.
    levels: Vec<(usize, Option<Vec<T>>)>,
}

impl<T: RealScalar> BatchedM2l<T> {
    /// The bytes of one level's tables at degree p, Dense's: 316 · 2 (p + 1)⁴ values of T.
    pub fn level_bytes(p: usize) -> usize {
        DenseM2l::<T>::level_bytes(p)
    }

    /// The levels whose tables fit in `budget` bytes, from the finest level up; never levels
    /// 0 and 1 ([`DenseM2l::select`]).
    pub fn select(degrees: &[usize], budget: usize) -> Vec<bool> {
        DenseM2l::<T>::select(degrees, budget)
    }

    /// Builds the tables of the levels with `levels[l]` set, for the root's k̂₀ = k r₀ and
    /// the degrees p_l (module documentation, "Build and levels"); with no level set, builds
    /// nothing.
    ///
    /// # Panics
    ///
    /// If `levels` and `degrees` differ in length, level 0 or 1 is chosen, or as
    /// [`HelmholtzTables::build`] panics.
    pub fn build(k_root: Complex<f64>, degrees: &[usize], levels: &[bool]) -> Self {
        assert_eq!(
            levels.len(),
            degrees.len(),
            "one choice per level: {} choices, {} degrees",
            levels.len(),
            degrees.len()
        );
        assert!(
            levels.iter().take(2).all(|&on| !on),
            "no batched M2L on levels 0 and 1: they have no V list"
        );
        if !levels.contains(&true) {
            return Self {
                levels: degrees.iter().map(|&p| (p, None)).collect(),
            };
        }
        let tables = HelmholtzTables::<f64>::build(k_root, degrees, DEFAULT_BLOCK_BUDGET);
        let levels = degrees
            .iter()
            .zip(levels)
            .enumerate()
            .map(|(l, (&p, &on))| (p, on.then(|| split(&tables, l))))
            .collect();
        Self { levels }
    }

    /// The number of levels.
    pub fn nlevels(&self) -> usize {
        self.levels.len()
    }

    /// The degree p_l of `level`.
    pub fn degree(&self, level: usize) -> usize {
        self.levels[level].0
    }

    /// Whether `level` has batched tables.
    pub fn is_batched(&self, level: usize) -> bool {
        self.levels[level].1.is_some()
    }

    /// The bytes of every level's tables.
    pub fn bytes(&self) -> usize {
        self.levels
            .iter()
            .filter_map(|(_, m)| m.as_ref())
            .map(|m| m.len() * size_of::<T>())
            .sum()
    }

    /// The planes (A_r, A_i) of offset index `d` on `level`, each n × n column-major
    /// (module documentation, "Layout").
    ///
    /// # Panics
    ///
    /// If `level` has no batched tables or d ≥ 316.
    pub fn planes(&self, level: usize, d: usize) -> (&[T], &[T]) {
        assert!(d < M2L_OFFSET_COUNT, "offset index {d}");
        let (p, matrices) = &self.levels[level];
        let matrices = matrices
            .as_ref()
            .unwrap_or_else(|| panic!("no batched M2L on level {level}"));
        let size = Layout::new(*p).len().pow(2);
        matrices[2 * d * size..2 * (d + 1) * size].split_at(size)
    }
}

/// Dense's 316 matrices of `level`, each re-laid from [A_r; A_i] (2n × n) to A_r then A_i
/// (n × n each), column-major.
fn split<T: RealScalar>(tables: &HelmholtzTables<f64>, level: usize) -> Vec<T> {
    let n = Layout::new(tables.degree(level)).len();
    let mut values: Vec<T> = matrices(tables, level, 0..M2L_OFFSET_COUNT);
    let mut column = vec![T::zero(); 2 * n * n];
    for matrix in values.chunks_exact_mut(2 * n * n) {
        column.copy_from_slice(matrix);
        let (a_r, a_i) = matrix.split_at_mut(n * n);
        for (j, dense) in column.chunks_exact(2 * n).enumerate() {
            a_r[j * n..(j + 1) * n].copy_from_slice(&dense[..n]);
            a_i[j * n..(j + 1) * n].copy_from_slice(&dense[n..]);
        }
    }
    values
}

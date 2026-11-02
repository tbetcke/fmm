//! The dense M2L tables of the Helmholtz family (Helmholtz CH3.2a;
//! docs/design/helmholtz-fmm-proposal.md §5.3, Dense): per level and V-list offset the
//! complex M2L as one real matrix, built in f64 from the level's RCR tables and rounded to T.
//!
//! # Layout
//!
//! The M2L of offset d on level l, of degree p = p_l and n = (p + 1)² coefficients, is a
//! complex n × n matrix A = A_r + i A_i from the multipole's coefficients to the local's
//! (CONVENTIONS §3.6 order in both). [`DenseM2l`] stores it as one 2n × n real matrix,
//! column-major, A_r in rows 0..n over A_i in rows n..2n: column j is the local of the unit
//! multipole e_j in two planes, real part then imaginary part, the layout of a box's
//! complex local (CONVENTIONS §3.15, "Storage"). The 316 matrices of a level lie one after
//! the other in offset order, 316 · 2n² values a level (21 MB at p = 7 in f64, 331 MB at
//! p = 15, 983 MB at p = 20).
//!
//! # Build
//!
//! [`DenseM2l::build`] builds [`HelmholtzTables`] in f64 for the same k̂₀ and degrees, and
//! takes column j of offset d on level l as their M2L of e_j (one real plane) into zeros:
//! the matrices hold the stored form of `HelmholtzTables` (its "The stored form, folded
//! in"), Rotation's coefficients applied to unit vectors, in f64, rounded to T entry by
//! entry. So Dense agrees with Rotation to the rounding of the two orders of operations,
//! and with `nd_fmm_ref::helmholtz::rcr` as Rotation does. The build is serial and
//! deterministic; it costs n Rotation applications per offset, O(p⁵).
//!
//! # Levels and the budget
//!
//! Only levels ≥ 2 carry a V list. [`DenseM2l::select`] picks the levels whose tables fit a
//! byte budget, from the finest level up, as `HelmholtzTables::build` stores its y-blocks:
//! a level that does not fit what is left is skipped, and a coarser one may still fit. The
//! choice is a function of the degrees and the budget only (proposal §2, constraint 9).
//!
//! # Application
//!
//! [`DenseM2l::m2l`] adds the M2L of `vectors` multipoles to their locals, per vector the
//! one-vector operation, each product and sum rounded apart (as `MatrixSet::apply`):
//!
//! - one plane (the real multipole of real k and real charges):
//!   L\[r\] ← L\[r\] + A\[r, k\] M\[k\] for k = 0, 1, …, n − 1 in turn, over the 2n rows;
//! - two planes: in one pass over the table, for k = 0, 1, …, n − 1 in turn,
//!   L_r\[r\] ← (L_r\[r\] + A_r\[r, k\] M_r\[k\]) − A_i\[r, k\] M_i\[k\] and
//!   L_i\[r\] ← (L_i\[r\] + A_i\[r, k\] M_r\[k\]) + A_r\[r, k\] M_i\[k\], over the n rows.
//!
//! So the four real products of a complex M2L are combined pair by pair, with no per-target
//! epilogue (Helmholtz CH3.5, question 2), and each table is read once per pair: per pair
//! the application is bound by reading the table, 2n² values.

use nd_fmm_math::{Complex, Layout, RealScalar};

use super::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use crate::geometry::M2L_OFFSET_COUNT;

/// The dense M2L tables of the chosen levels of an octree (the
/// [module documentation](self)).
#[derive(Clone, Debug)]
pub struct DenseM2l<T: RealScalar> {
    /// Per level its degree and, if dense, its 316 matrices.
    levels: Vec<(usize, Option<Vec<T>>)>,
}

impl<T: RealScalar> DenseM2l<T> {
    /// The bytes of one level's 316 matrices at degree p: 316 · 2 (p + 1)⁴ values of T.
    pub fn level_bytes(p: usize) -> usize {
        let n = Layout::new(p).len();
        M2L_OFFSET_COUNT * 2 * n * n * size_of::<T>()
    }

    /// The levels whose tables fit in `budget` bytes, from the finest level up; never levels
    /// 0 and 1 (module documentation, "Levels and the budget").
    pub fn select(degrees: &[usize], budget: usize) -> Vec<bool> {
        let mut chosen = vec![false; degrees.len()];
        let mut left = budget;
        for (l, &p) in degrees.iter().enumerate().skip(2).rev() {
            let bytes = Self::level_bytes(p);
            if bytes <= left {
                left -= bytes;
                chosen[l] = true;
            }
        }
        chosen
    }

    /// Builds the tables of the levels with `levels[l]` set, for the root's k̂₀ = k r₀ and
    /// the degrees p_l (module documentation, "Build"); with no level set, builds nothing.
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
            "no dense M2L on levels 0 and 1: they have no V list"
        );
        if !levels.contains(&true) {
            return Self {
                levels: degrees.iter().map(|&p| (p, None)).collect(),
            };
        }
        let tables = HelmholtzTables::<f64>::build(k_root, degrees, DEFAULT_BLOCK_BUDGET);
        let mut scratch = tables.scratch();
        let levels = degrees
            .iter()
            .zip(levels)
            .enumerate()
            .map(|(l, (&p, &on))| {
                let matrices = on.then(|| {
                    let n = Layout::new(p).len();
                    let mut identity = vec![0.0; n * n];
                    for j in 0..n {
                        identity[j * n + j] = 1.0;
                    }
                    let mut columns = vec![0.0; 2 * n * n];
                    let mut values = Vec::with_capacity(M2L_OFFSET_COUNT * 2 * n * n);
                    for d in 0..M2L_OFFSET_COUNT {
                        // Vector j of n is e_j; its local is column j, two planes.
                        columns.fill(0.0);
                        tables.m2l(l, d, &identity, 1, &mut columns, n, &mut scratch);
                        values.extend(columns.iter().map(|&v| T::from_f64(v)));
                    }
                    values
                });
                (p, matrices)
            })
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

    /// Whether `level` has dense tables.
    pub fn is_dense(&self, level: usize) -> bool {
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

    /// The 2n × n matrix of offset index `d` on `level`, column-major (module
    /// documentation, "Layout").
    ///
    /// # Panics
    ///
    /// If `level` has no dense tables or d ≥ 316.
    pub fn matrix(&self, level: usize, d: usize) -> &[T] {
        assert!(d < M2L_OFFSET_COUNT, "offset index {d}");
        let (p, matrices) = &self.levels[level];
        let matrices = matrices
            .as_ref()
            .unwrap_or_else(|| panic!("no dense M2L on level {level}"));
        let size = 2 * Layout::new(*p).len().pow(2);
        &matrices[d * size..(d + 1) * size]
    }

    /// M2L on `level` at the offset index `d`: adds the translations of `vectors`
    /// multipoles of `planes` planes (2, or 1 for the real multipoles of real k and real
    /// charges) to the target's `vectors` complex locals (module documentation,
    /// "Application").
    ///
    /// # Panics
    ///
    /// If `level` has no dense tables, d ≥ 316, `planes` is not 1 or 2, or a slice is not
    /// `vectors` expansions at the level's degree.
    pub fn m2l(
        &self,
        level: usize,
        d: usize,
        multipole: &[T],
        planes: usize,
        local: &mut [T],
        vectors: usize,
    ) {
        let a = self.matrix(level, d);
        assert!(matches!(planes, 1 | 2), "{planes} multipole planes");
        let n = Layout::new(self.levels[level].0).len();
        assert_eq!(multipole.len(), vectors * planes * n, "multipole length");
        assert_eq!(local.len(), vectors * 2 * n, "local length");
        for (m, l) in multipole
            .chunks_exact(planes * n)
            .zip(local.chunks_exact_mut(2 * n))
        {
            if planes == 1 {
                for (column, &x) in a.chunks_exact(2 * n).zip(m) {
                    for (y, &e) in l.iter_mut().zip(column) {
                        *y = *y + e * x;
                    }
                }
                continue;
            }
            let (m_r, m_i) = m.split_at(n);
            let (l_r, l_i) = l.split_at_mut(n);
            for ((column, &x_r), &x_i) in a.chunks_exact(2 * n).zip(m_r).zip(m_i) {
                let (a_r, a_i) = column.split_at(n);
                for ((y_r, y_i), (&e_r, &e_i)) in
                    l_r.iter_mut().zip(l_i.iter_mut()).zip(a_r.iter().zip(a_i))
                {
                    *y_r = *y_r + e_r * x_r - e_i * x_i;
                    *y_i = *y_i + e_i * x_r + e_r * x_i;
                }
            }
        }
    }
}

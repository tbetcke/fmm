//! The batched M2L of the Helmholtz operator (Helmholtz CH3.2c; the operator module docs,
//! "M2L"): its level call, block by block, and its pair body, both through one chain of
//! column-kernel calls per offset ([`apply`]).
//!
//! The level call is Laplace's batched M2L (`driver::m2l_block`) with Helmholtz's layout: a
//! box holds m expansions, the multipoles of `planes` planes of n values and the locals of
//! two, and each offset's split planes (A_r, A_i) are applied by two or four calls of the
//! column kernel. The imaginary planes are reached by the same column offsets on the slices
//! shifted by n, so one pair of column lists serves every call. `driver.rs` and Laplace's
//! paths are not changed (the user, 2026-11-02; docs/helmholtz/tasks/CH3.2c.md).

use nd_fmm_math::Layout;
use nd_fmm_plan::operator::M2l;
use nd_fmm_simd::{MatrixKernel, SimdScalar};
use nd_fmm_tables::helmholtz::batched::BatchedM2l;

use crate::driver::{Columns, Execution, offset_index};
use crate::operator::M2L_BLOCK_TARGETS;

/// What the batched M2L reads besides a call's data: the tables, the column kernel, the
/// multipoles' planes (1 or 2) and the charge vectors m.
#[derive(Clone, Copy)]
pub(super) struct Batched<'a, T: SimdScalar> {
    pub(super) tables: &'a BatchedM2l<T>,
    pub(super) matrix: &'a MatrixKernel<T>,
    pub(super) planes: usize,
    pub(super) vectors: usize,
}

impl<T: SimdScalar> Batched<'_, T> {
    /// n = (p_l + 1)² of `level`.
    fn n(&self, level: usize) -> usize {
        Layout::new(self.tables.degree(level)).len()
    }

    /// The M2L level call on a level with tables: blocks of [`M2L_BLOCK_TARGETS`] targets,
    /// and in each, for every offset index d in order, the pairs of `batch(d)` whose target
    /// lies in the block (two binary searches), m columns each, through [`apply`]. Each
    /// local receives the chain of [`pair`](Self::pair) for each of its sources in the
    /// order of its row, so the call is the pair-by-pair operator bit for bit.
    pub(super) fn m2l<S: Send>(
        &self,
        execution: &mut Execution<S>,
        batch: M2l<'_, T>,
        columns: fn(&mut S) -> &mut Columns<T>,
    ) {
        let M2l {
            level,
            index,
            pairs,
            multipoles,
            locals,
        } = batch;
        if pairs.is_empty() {
            return;
        }
        let (m, n) = (self.vectors, self.n(level));
        let (multipole_size, local_size) = (self.planes * n, 2 * n);
        let multipoles = multipoles.as_slice();
        execution.blocks(locals, M2L_BLOCK_TARGETS, |first, block, scratch| {
            let end = first + block.len() / (m * local_size);
            let Columns {
                pack,
                sources,
                targets,
            } = columns(scratch);
            for d in 0..pairs.ngroups() {
                let (batch_targets, batch_sources) = pairs.batch(d);
                let lo = batch_targets.partition_point(|&t| (t as usize) < first);
                let hi = lo + batch_targets[lo..].partition_point(|&t| (t as usize) < end);
                if lo == hi {
                    continue;
                }
                sources.clear();
                targets.clear();
                for (&t, &s) in batch_targets[lo..hi].iter().zip(&batch_sources[lo..hi]) {
                    debug_assert_eq!(
                        offset_index(index.key(level, s as usize), index.key(level, t as usize)),
                        d,
                        "the offset index of a V-list batch disagrees with its keys"
                    );
                    // The m columns of the pair, vector after vector: the real planes.
                    for v in 0..m {
                        sources.push((s as usize * m + v) * multipole_size);
                        targets.push(((t as usize - first) * m + v) * local_size);
                    }
                }
                self.apply(level, d, multipoles, sources, block, targets, pack);
            }
        });
    }

    /// The pair body: the M2L of one source's m multipoles at offset index `d` into one
    /// target's m locals, the chain of [`apply`](Self::apply) with one column per vector.
    pub(super) fn pair(
        &self,
        level: usize,
        d: usize,
        multipole: &[T],
        local: &mut [T],
        columns: &mut Columns<T>,
    ) {
        let n = self.n(level);
        let m = self.vectors;
        assert_eq!(multipole.len(), m * self.planes * n, "multipole length");
        assert_eq!(local.len(), m * 2 * n, "local length");
        columns.sources.clear();
        columns.targets.clear();
        for v in 0..m {
            columns.sources.push(v * self.planes * n);
            columns.targets.push(v * 2 * n);
        }
        let Columns {
            pack,
            sources,
            targets,
        } = columns;
        self.apply(level, d, multipole, sources, local, targets, pack);
    }

    /// Adds the M2L of offset index `d` on `level` from the multipoles whose real planes
    /// start at `sources` to the locals whose real planes start at `targets`, each column of
    /// the column kernel's chain (k ascending, product and sum rounded apart):
    ///
    /// - one plane: L_r += A_r M, then L_i += A_i M (Dense's chain, bit for bit);
    /// - two planes: L_r += A_r M_r; L_r −= A_i M_i, as L_r negated, A_i M_i added and L_r
    ///   negated back (exact: the bits of adding −A_i M_i, but for the sign of an exact zero);
    ///   L_i += A_i M_r; L_i += A_r M_i (Dense's sums to rounding).
    #[expect(
        clippy::too_many_arguments,
        reason = "the offset, two column lists over two slices, and the packing buffer"
    )]
    fn apply(
        &self,
        level: usize,
        d: usize,
        multipoles: &[T],
        sources: &[usize],
        locals: &mut [T],
        targets: &[usize],
        pack: &mut [T],
    ) {
        let n = self.n(level);
        let (a_r, a_i) = self.tables.planes(level, d);
        let matrix = self.matrix;
        matrix.accumulate_columns(n, a_r, multipoles, sources, locals, targets, pack);
        if self.planes == 1 {
            matrix.accumulate_columns(n, a_i, multipoles, sources, &mut locals[n..], targets, pack);
            return;
        }
        let imaginary = &multipoles[n..];
        negate(locals, targets, n);
        matrix.accumulate_columns(n, a_i, imaginary, sources, locals, targets, pack);
        negate(locals, targets, n);
        let locals_i = &mut locals[n..];
        matrix.accumulate_columns(n, a_i, multipoles, sources, locals_i, targets, pack);
        matrix.accumulate_columns(n, a_r, imaginary, sources, locals_i, targets, pack);
    }
}

/// Negates the n values at each of `targets` in `values` (exact).
#[inline]
fn negate<T: SimdScalar>(values: &mut [T], targets: &[usize], n: usize) {
    for &o in targets {
        for x in &mut values[o..o + n] {
            *x = -*x;
        }
    }
}

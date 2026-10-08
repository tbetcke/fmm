//! The operator interface: level-batched calls, and a per-pair adapter.
//!
//! An FMM supplies its arithmetic through [`FmmOperator`]: one call per level and
//! operator kind, which receives everything the level needs as one batch (design §6,
//! `docs/design/fmm-plan-redesign.md`). A simple operator implements [`PairOperator`]
//! instead, one method per pair of boxes, and runs through the [`PerPair`] adapter.
//!
//! # Batches
//!
//! Every batch carries the target `level`, the [`BoxIndex`] (keys of every box and leaf,
//! for geometry), the views of the plan that the call executes, its inputs (shared) and
//! its output (exclusive):
//!
//! | Call (target level l) | View | Inputs | Output |
//! | --- | --- | --- | --- |
//! | [`p2m`](FmmOperator::p2m) | [`p2m`](super::lists::LevelLists::p2m): box → its local leaf | sources of every leaf | multipoles of l |
//! | [`m2m`](FmmOperator::m2m) | [`m2m_local`](super::lists::LevelLists::m2m_local) or [`m2m_global`](super::lists::LevelLists::m2m_global): parent → children, rows and octant batches | multipoles of l + 1 | multipoles of l |
//! | [`m2l`](FmmOperator::m2l) | [`v`](super::lists::LevelLists::v): rows and offset batches | multipoles of l | locals of l |
//! | [`p2l`](FmmOperator::p2l) | [`x`](super::lists::LevelLists::x) | sources of every leaf | locals of l |
//! | [`l2l`](FmmOperator::l2l) | [`l2l`](super::lists::LevelLists::l2l): child → parent, rows and octant batches | locals of l − 1 | locals of l |
//! | [`l2p`](FmmOperator::l2p) | [`l2p`](super::lists::LevelLists::l2p): local leaf → its box | locals of l; target input of the level's local leaves | target output of the level's local leaves |
//! | [`m2p`](FmmOperator::m2p) | [`w`](super::lists::LevelLists::w) | multipoles of l + 1; target input | target output |
//! | [`p2p`](FmmOperator::p2p) | [`near`](super::lists::LevelLists::near): U-list and the leaf itself | sources of every leaf; target input | target output |
//!
//! Box views have one row per box of the level, ghosts included with empty rows, so row
//! t is chunk t of the level's output buffer. Leaf views have one row per local leaf of
//! the level: row r is leaf `leaves.start + r` of the leaf numbering and leaf r of the
//! batch's target slices. Source slices cover every leaf, local and ghost, so a source
//! leaf index from a row is a chunk index of `sources`. A grouped view
//! ([`VList`], [`Children`], [`Parents`]) holds the same pairs twice, by target and by
//! group, so a host operator can walk the rows and a GEMM operator the batches (design
//! §6.4, §6.5).
//!
//! # Accumulation rule
//!
//! Every call **adds** into its output, which holds the contributions of earlier calls.
//! An operator may execute the targets of a call in any order and on any number of
//! threads, provided it adds the contributions of each target in the order of that
//! target's row in the target-centric view: by offset index (V), by octant (M2M), by
//! the source leaf's (level, key) (X, near; the leaf index on one rank) or by box index
//! (W). A walk over the batches of a grouped view in
//! group order meets each target in that order too. With every operator following the
//! rule, the evaluator's fixed call order fixes every sum, and results are bit-identical
//! from run to run ([`Evaluator`](super::evaluator::Evaluator), design §7.5). Operators
//! document the order they use.
//!
//! # Points
//!
//! A leaf's chunk of a leaf store holds `count · point_size` values: the points of the
//! leaf one after the other, or in any other layout the operator chooses (CONVENTIONS
//! §3.13 for Laplace). Counts may be zero, and a leaf with no points has an empty chunk;
//! operators treat such a pair as a no-op. The number of target points of a leaf is
//! `target_output.count(r)`, or the length of its output chunk divided by
//! [`target_output_point_size`](FmmSizes::target_output_point_size).

#[cfg(test)]
#[path = "operator_tests.rs"]
pub(crate) mod tests;

use std::ops::Range;

use mpi::traits::Equivalence;
use nd_octree::MortonKey;

use super::index::BoxIndex;
use super::lists::{Children, Csr, Parents, VList};
use super::store::{LeafSlice, LeafSliceMut, LevelSlice, LevelSliceMut};

/// The data sizes of an FMM, shared by the batched and the per-pair interface.
///
/// The evaluator reads every size once, at construction, and rejects a zero where the
/// size must be positive.
pub trait FmmSizes {
    /// The scalar type of every buffer: coefficients, source data, target input and
    /// target output.
    type Value: Equivalence + Copy + Default + Send + Sync;

    /// Return the number of values of a multipole expansion on `level`; positive.
    fn multipole_size(&self, level: usize) -> usize;

    /// Return the number of values of a local expansion on `level`; positive.
    fn local_size(&self, level: usize) -> usize;

    /// Return the number of values per source point; positive.
    fn source_point_size(&self) -> usize;

    /// Return the number of values per target point in the target input; may be zero.
    fn target_input_point_size(&self) -> usize;

    /// Return the number of values per target point in the target output; positive.
    fn target_output_point_size(&self) -> usize;
}

/// The upward pass an M2M call belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UpwardPass {
    /// The local pass: `LocalInterior` parents, whose children are all on this rank.
    Local,
    /// The global pass: `Global` parents, formed on every rank from the gathered
    /// coarse blocks.
    Global,
}

/// P2M on one level: the multipole of every local leaf of the level from its sources.
#[derive(Debug)]
pub struct P2m<'a, T> {
    /// The level of the boxes.
    pub level: usize,
    /// The box index, for keys.
    pub index: &'a BoxIndex,
    /// Row t: the leaf index of box t, if box t is a local leaf, else empty.
    pub leaves: &'a Csr,
    /// Source data of every leaf, by leaf index.
    pub sources: LeafSlice<'a, T>,
    /// Multipoles of the level, box t at t · size; the call adds into them.
    pub multipoles: LevelSliceMut<'a, T>,
}

/// M2M on one level and pass: the multipoles of the level's parents from their
/// children on `level + 1`.
#[derive(Debug)]
pub struct M2m<'a, T> {
    /// The level of the parents.
    pub level: usize,
    /// The box index, for keys.
    pub index: &'a BoxIndex,
    /// The pass: which parents have rows.
    pub pass: UpwardPass,
    /// Row t: the eight children of parent t (box indices on `level + 1`) with their
    /// octants, by octant; empty for a box of the other pass, a leaf or a ghost. Batch o:
    /// (parents, children) of octant o.
    pub children: &'a Children,
    /// Multipoles of `level + 1`, by box index.
    pub child_multipoles: LevelSlice<'a, T>,
    /// Multipoles of the level, box t at t · size; the call adds into them.
    pub multipoles: LevelSliceMut<'a, T>,
}

/// M2L on one level: every V-list pair whose target lies on the level.
#[derive(Debug)]
pub struct M2l<'a, T> {
    /// The level of targets and sources.
    pub level: usize,
    /// The box index, for keys.
    pub index: &'a BoxIndex,
    /// Row t: the sources of target t (box indices on the level) with their offset
    /// indices in `V_LIST_DIRECTIONS`, by offset index. Batch d: (targets, sources) of
    /// offset index d.
    pub pairs: &'a VList,
    /// Multipoles of the level, by box index.
    pub multipoles: LevelSlice<'a, T>,
    /// Locals of the level, box t at t · size; the call adds into them.
    pub locals: LevelSliceMut<'a, T>,
}

/// P2L on one level: every X-list pair whose target lies on the level.
#[derive(Debug)]
pub struct P2l<'a, T> {
    /// The level of the targets.
    pub level: usize,
    /// The box index, for keys.
    pub index: &'a BoxIndex,
    /// Row t: the X-list of box t, as leaf indices (leaves on `level − 1`), ascending in
    /// the leaf's key.
    pub x: &'a Csr,
    /// Source data of every leaf, by leaf index.
    pub sources: LeafSlice<'a, T>,
    /// Locals of the level, box t at t · size; the call adds into them.
    pub locals: LevelSliceMut<'a, T>,
}

/// L2L on one level: the locals of the level's boxes from their parents on
/// `level − 1`.
#[derive(Debug)]
pub struct L2l<'a, T> {
    /// The level of the children; at least 1.
    pub level: usize,
    /// The box index, for keys.
    pub index: &'a BoxIndex,
    /// Row t: the parent of box t (a box index on `level − 1`) with the octant of box t;
    /// empty for a ghost. Batch o: (children, parents) of octant o.
    pub parents: &'a Parents,
    /// Locals of `level − 1`, by box index.
    pub parent_locals: LevelSlice<'a, T>,
    /// Locals of the level, box t at t · size; the call adds into them.
    pub locals: LevelSliceMut<'a, T>,
}

/// L2P for the local leaves of one level.
#[derive(Debug)]
pub struct L2p<'a, T> {
    /// The level of the leaves.
    pub level: usize,
    /// The box index, for keys.
    pub index: &'a BoxIndex,
    /// The leaf indices of the rows: row r is leaf `leaves.start + r`.
    pub leaves: Range<usize>,
    /// Row r: the box index of leaf `leaves.start + r` on the level.
    pub boxes: &'a Csr,
    /// Locals of the level, by box index.
    pub locals: LevelSlice<'a, T>,
    /// Target input of the rows' leaves, leaf r for row r.
    pub target_input: LeafSlice<'a, T>,
    /// Target output of the rows' leaves, leaf r for row r; the call adds into it.
    pub target_output: LeafSliceMut<'a, T>,
}

/// M2P for the local leaves of one level: every W-list pair whose target lies on the
/// level.
#[derive(Debug)]
pub struct M2p<'a, T> {
    /// The level of the target leaves.
    pub level: usize,
    /// The box index, for keys.
    pub index: &'a BoxIndex,
    /// The leaf indices of the rows: row r is leaf `leaves.start + r`.
    pub leaves: Range<usize>,
    /// Row r: the W-list of leaf `leaves.start + r`, as box indices on `level + 1`,
    /// ascending.
    pub w: &'a Csr,
    /// Multipoles of `level + 1`, by box index; empty on the deepest level, whose rows
    /// are all empty.
    pub multipoles: LevelSlice<'a, T>,
    /// Target input of the rows' leaves, leaf r for row r.
    pub target_input: LeafSlice<'a, T>,
    /// Target output of the rows' leaves, leaf r for row r; the call adds into it.
    pub target_output: LeafSliceMut<'a, T>,
}

/// P2P for the local leaves of one level: every near pair (U-list and self) whose
/// target lies on the level.
#[derive(Debug)]
pub struct P2p<'a, T> {
    /// The level of the target leaves.
    pub level: usize,
    /// The box index, for keys.
    pub index: &'a BoxIndex,
    /// The leaf indices of the rows: row r is leaf `leaves.start + r`.
    pub leaves: Range<usize>,
    /// Row r: the near list of leaf `leaves.start + r`, as leaf indices, ascending in the
    /// leaf's (level, key); it holds the leaf itself, which an operator recognises by
    /// source = target.
    pub near: &'a Csr,
    /// Source data of every leaf, by leaf index.
    pub sources: LeafSlice<'a, T>,
    /// Target input of the rows' leaves, leaf r for row r.
    pub target_input: LeafSlice<'a, T>,
    /// Target output of the rows' leaves, leaf r for row r; the call adds into it.
    pub target_output: LeafSliceMut<'a, T>,
}

/// Level-batched operators.
///
/// The [`Evaluator`](super::evaluator::Evaluator) calls each method once per level of
/// the plan, on every rank, also when the level's view is empty (the operator then
/// returns at once), in this order (design §7.2, L the deepest level):
///
/// | Stage | Calls, in order |
/// | --- | --- |
/// | `upward_local` | for l = L down to 0: `p2m(l)`, then `m2m(l − 1)` of the [local pass](UpwardPass::Local) (l ≥ 1) |
/// | `upward_global` | for l = L − 1 down to 0: `m2m(l)` of the [global pass](UpwardPass::Global) |
/// | `downward` | for l = 1 to L: `l2l(l)`, `m2l(l)`, `p2l(l)` |
/// | `evaluate_leaves` | for l = 0 to L: `l2p(l)`, `m2p(l)`, `p2p(l)` |
///
/// So `m2m` is called for the levels 0..L once per pass, `l2l`, `m2l` and `p2l` for
/// 1..=L, and the others for 0..=L.
///
/// Every method follows the [accumulation rule](self#accumulation-rule): it adds into
/// its output, and adds the contributions of each target in the order of the target's
/// row. It may read every input of its batch and the keys of the index, and write only
/// its output. `&mut self` lets the operator own scratch space; it must not keep state
/// that changes its results from one evaluation to the next.
pub trait FmmOperator: FmmSizes {
    /// Add the multipole of every local leaf of the level from its source points.
    ///
    /// For every box t with a leaf j in its row: `multipoles[t] += P2M(sources[j])`.
    fn p2m(&mut self, batch: P2m<'_, Self::Value>);

    /// Add the multipoles of the children to their parents, for one pass.
    ///
    /// For every parent t and every (child c, octant o) of its row, by octant:
    /// `multipoles[t] += M2M_o(child_multipoles[c])`.
    fn m2m(&mut self, batch: M2m<'_, Self::Value>);

    /// Add the V-list translations of the level.
    ///
    /// For every target t and every (source s, offset index d) of its row, by offset
    /// index: `locals[t] += M2L_d(multipoles[s])`.
    fn m2l(&mut self, batch: M2l<'_, Self::Value>);

    /// Add the X-list sources of the level's boxes to their locals.
    ///
    /// For every target t and every leaf j of its row, in row order (by the leaf's
    /// (level, key)): `locals[t] += P2L(sources[j])`.
    fn p2l(&mut self, batch: P2l<'_, Self::Value>);

    /// Add the locals of the parents to their children.
    ///
    /// For every box t with a (parent p, octant o) in its row:
    /// `locals[t] += L2L_o(parent_locals[p])`.
    fn l2l(&mut self, batch: L2l<'_, Self::Value>);

    /// Add the local expansion of every local leaf of the level at its target points.
    ///
    /// For every row r with box i: `target_output[r] += L2P(locals[i], target_input[r])`.
    fn l2p(&mut self, batch: L2p<'_, Self::Value>);

    /// Add the W-list multipoles at the target points of the level's local leaves.
    ///
    /// For every row r and every box s of its row, by box index:
    /// `target_output[r] += M2P(multipoles[s], target_input[r])`.
    fn m2p(&mut self, batch: M2p<'_, Self::Value>);

    /// Add the direct interactions of the near list at the target points of the level's
    /// local leaves.
    ///
    /// For every row r and every leaf j of its row, in row order (by the leaf's
    /// (level, key)): `target_output[r] += P2P(sources[j], target_input[r])`.
    fn p2p(&mut self, batch: P2p<'_, Self::Value>);
}

/// Operators with one method per pair of boxes; [`PerPair`] turns them into an
/// [`FmmOperator`].
///
/// Every method adds into its output (the last argument). Boxes are given by their
/// keys, for geometry, and M2M, M2L and L2L also by the octant or offset index of the
/// pair, for a table lookup. A leaf with no points has an empty chunk.
pub trait PairOperator: FmmSizes {
    /// Add the multipole of the source points of `leaf`.
    fn p2m(&mut self, leaf: MortonKey, sources: &[Self::Value], multipole: &mut [Self::Value]);

    /// Add the multipole of `child`, of octant `octant` of `parent`, to the multipole of
    /// `parent`.
    fn m2m(
        &mut self,
        child: MortonKey,
        parent: MortonKey,
        octant: usize,
        child_multipole: &[Self::Value],
        parent_multipole: &mut [Self::Value],
    );

    /// Add the multipole of `source` to the local of `target`, a V-list pair of offset
    /// index `offset_index` in `V_LIST_DIRECTIONS`.
    fn m2l(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        offset_index: usize,
        source_multipole: &[Self::Value],
        target_local: &mut [Self::Value],
    );

    /// Add the source points of the leaf `source` to the local of `target` (X-list).
    fn p2l(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        sources: &[Self::Value],
        target_local: &mut [Self::Value],
    );

    /// Add the local of `parent` to the local of `child`, of octant `octant` of
    /// `parent`.
    fn l2l(
        &mut self,
        parent: MortonKey,
        child: MortonKey,
        octant: usize,
        parent_local: &[Self::Value],
        child_local: &mut [Self::Value],
    );

    /// Add the local of `leaf` at its target points.
    fn l2p(
        &mut self,
        leaf: MortonKey,
        local: &[Self::Value],
        target_input: &[Self::Value],
        target_output: &mut [Self::Value],
    );

    /// Add the multipole of `source` at the target points of the leaf `target`
    /// (W-list).
    fn m2p(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        source_multipole: &[Self::Value],
        target_input: &[Self::Value],
        target_output: &mut [Self::Value],
    );

    /// Add the source points of the leaf `source` directly at the target points of the
    /// leaf `target` (U-list, and the self pair with `source == target`).
    fn p2p(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        sources: &[Self::Value],
        target_input: &[Self::Value],
        target_output: &mut [Self::Value],
    );
}

/// Runs a [`PairOperator`] as an [`FmmOperator`]: serially, target by target in index
/// order, each target's row in order.
///
/// Every pair of every row is issued exactly once, also when a chunk is empty. The
/// order of each target's contributions is that of its row, as the
/// [accumulation rule](self#accumulation-rule) requires.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PerPair<P>(pub P);

impl<P: PairOperator> FmmSizes for PerPair<P> {
    type Value = P::Value;

    fn multipole_size(&self, level: usize) -> usize {
        self.0.multipole_size(level)
    }

    fn local_size(&self, level: usize) -> usize {
        self.0.local_size(level)
    }

    fn source_point_size(&self) -> usize {
        self.0.source_point_size()
    }

    fn target_input_point_size(&self) -> usize {
        self.0.target_input_point_size()
    }

    fn target_output_point_size(&self) -> usize {
        self.0.target_output_point_size()
    }
}

impl<P: PairOperator> FmmOperator for PerPair<P> {
    fn p2m(&mut self, batch: P2m<'_, P::Value>) {
        let P2m {
            index,
            leaves,
            sources,
            mut multipoles,
            ..
        } = batch;
        for t in 0..leaves.nrows() {
            for &j in leaves.row(t) {
                let j = j as usize;
                self.0
                    .p2m(index.leaf_key(j), sources.chunk(j), multipoles.chunk_mut(t));
            }
        }
    }

    fn m2m(&mut self, batch: M2m<'_, P::Value>) {
        let M2m {
            level,
            index,
            children,
            child_multipoles,
            mut multipoles,
            ..
        } = batch;
        for t in 0..children.nrows() {
            let (sources, octants) = children.row(t);
            for (&c, &o) in sources.iter().zip(octants) {
                self.0.m2m(
                    index.key(level + 1, c as usize),
                    index.key(level, t),
                    o as usize,
                    child_multipoles.chunk(c as usize),
                    multipoles.chunk_mut(t),
                );
            }
        }
    }

    fn m2l(&mut self, batch: M2l<'_, P::Value>) {
        let M2l {
            level,
            index,
            pairs,
            multipoles,
            mut locals,
        } = batch;
        for t in 0..pairs.nrows() {
            let (sources, offsets) = pairs.row(t);
            for (&s, &d) in sources.iter().zip(offsets) {
                self.0.m2l(
                    index.key(level, s as usize),
                    index.key(level, t),
                    d as usize,
                    multipoles.chunk(s as usize),
                    locals.chunk_mut(t),
                );
            }
        }
    }

    fn p2l(&mut self, batch: P2l<'_, P::Value>) {
        let P2l {
            level,
            index,
            x,
            sources,
            mut locals,
        } = batch;
        for t in 0..x.nrows() {
            for &j in x.row(t) {
                let j = j as usize;
                self.0.p2l(
                    index.leaf_key(j),
                    index.key(level, t),
                    sources.chunk(j),
                    locals.chunk_mut(t),
                );
            }
        }
    }

    fn l2l(&mut self, batch: L2l<'_, P::Value>) {
        let L2l {
            level,
            index,
            parents,
            parent_locals,
            mut locals,
        } = batch;
        for t in 0..parents.nrows() {
            let (sources, octants) = parents.row(t);
            for (&p, &o) in sources.iter().zip(octants) {
                self.0.l2l(
                    index.key(level - 1, p as usize),
                    index.key(level, t),
                    o as usize,
                    parent_locals.chunk(p as usize),
                    locals.chunk_mut(t),
                );
            }
        }
    }

    fn l2p(&mut self, batch: L2p<'_, P::Value>) {
        let L2p {
            index,
            leaves,
            boxes,
            locals,
            target_input,
            mut target_output,
            ..
        } = batch;
        for r in 0..boxes.nrows() {
            for &i in boxes.row(r) {
                self.0.l2p(
                    index.leaf_key(leaves.start + r),
                    locals.chunk(i as usize),
                    target_input.chunk(r),
                    target_output.chunk_mut(r),
                );
            }
        }
    }

    fn m2p(&mut self, batch: M2p<'_, P::Value>) {
        let M2p {
            level,
            index,
            leaves,
            w,
            multipoles,
            target_input,
            mut target_output,
        } = batch;
        for r in 0..w.nrows() {
            for &s in w.row(r) {
                self.0.m2p(
                    index.key(level + 1, s as usize),
                    index.leaf_key(leaves.start + r),
                    multipoles.chunk(s as usize),
                    target_input.chunk(r),
                    target_output.chunk_mut(r),
                );
            }
        }
    }

    fn p2p(&mut self, batch: P2p<'_, P::Value>) {
        let P2p {
            index,
            leaves,
            near,
            sources,
            target_input,
            mut target_output,
            ..
        } = batch;
        for r in 0..near.nrows() {
            for &j in near.row(r) {
                let j = j as usize;
                self.0.p2p(
                    index.leaf_key(j),
                    index.leaf_key(leaves.start + r),
                    sources.chunk(j),
                    target_input.chunk(r),
                    target_output.chunk_mut(r),
                );
            }
        }
    }
}

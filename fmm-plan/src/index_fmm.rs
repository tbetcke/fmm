//! A test FMM that propagates leaf indices, with variable counts per leaf.
//!
//! Every leaf of the global tree has a global index in `0..N`, its position in the
//! global Morton order of all leaves, where N is the global number of leaves
//! ([`GlobalLeaves`]). Every coefficient and every target point holds a vector of N
//! counts:
//! - every source point of leaf j carries the value j (one value per point);
//! - P2M, P2L and P2P add one at index j for every source point of leaf j;
//! - M2M, M2L, L2L add their input vector to their output vector, and L2P and M2P add
//!   it to the vector of every target point;
//! - the target input has no values; the target output has N values per point.
//!
//! If the FMM visits every pair of leaves exactly once, every target point of every leaf
//! holds, at index j, exactly the number of source points of leaf j ([`check_counts`]).
//! With one point per leaf the check is a count of one at every index. A missed
//! interaction leaves too small a count, an interaction counted twice too large a one.
//!
//! **Zero counts hide defects.** A leaf without source points contributes zero
//! everywhere, so a missed or doubled interaction with it cannot be seen, and a leaf
//! without target points checks nothing. Tests should keep most counts nonzero.
//!
//! The FMM comes in two implementations of the same arithmetic, which must give the same
//! output:
//! - [`IndexFmm`], a [`PairOperator`], run through the [`PerPair`] adapter;
//! - [`BatchedIndexFmm`], an [`FmmOperator`] implemented directly, which walks either
//!   the target-centric rows or the groupings of each view ([`Walk`]).
//!
//! Every operation is an exact integer addition, so the output does not depend on the
//! order of the additions. [`run_index_fmm`] runs the complete distributed FMM and
//! checks it on every rank.
//!
//! Memory grows as boxes × N: at 10⁴ leaves a level buffer holds about 10⁸ `u32`
//! (design §11.4).

#[cfg(test)]
#[path = "index_fmm_tests.rs"]
mod tests;

use mpi::{collective::SystemOperation, traits::CommunicatorCollectives};
use nd_octree::MortonKey;
use rlst::distributed_tools::array_tools::gather_to_all;

use super::evaluator::Evaluator;
use super::index::BoxIndex;
use super::lists::{Csr, GroupedCsr};
use super::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PairOperator, PerPair,
};
use super::plan::Plan;
use super::store::{LeafSliceMut, LeafStore, LevelSlice, LevelSliceMut};

/// Add `input` to `output`, value by value.
fn add(input: &[u32], output: &mut [u32]) {
    for (out, &value) in output.iter_mut().zip(input) {
        *out += value;
    }
}

/// Add one at the index carried by every source point in `sources`.
fn add_indices(sources: &[u32], output: &mut [u32]) {
    for &j in sources {
        output[j as usize] += 1;
    }
}

/// Add `input` to the vector of every target point of `target_output`.
fn add_at_points(input: &[u32], target_output: &mut [u32], nleaves: usize) {
    for point in target_output.chunks_exact_mut(nleaves) {
        add(input, point);
    }
}

/// Add one at the index of every source point of `sources` to the vector of every
/// target point of `target_output`.
fn add_indices_at_points(sources: &[u32], target_output: &mut [u32], nleaves: usize) {
    for point in target_output.chunks_exact_mut(nleaves) {
        add_indices(sources, point);
    }
}

/// The sizes of the index FMM for `nleaves` leaves.
macro_rules! index_sizes {
    () => {
        type Value = u32;

        fn multipole_size(&self, _level: usize) -> usize {
            self.nleaves
        }

        fn local_size(&self, _level: usize) -> usize {
            self.nleaves
        }

        fn source_point_size(&self) -> usize {
            1
        }

        fn target_input_point_size(&self) -> usize {
            0
        }

        fn target_output_point_size(&self) -> usize {
            self.nleaves
        }
    };
}

/// The index FMM as a [`PairOperator`]; see the [module documentation](self).
///
/// Run it through [`PerPair`]: `PerPair(IndexFmm::new(nleaves))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexFmm {
    nleaves: usize,
}

impl IndexFmm {
    /// Create the operators for a tree with `nleaves` leaves in total.
    ///
    /// # Panics
    ///
    /// Panics if `nleaves` is zero.
    pub fn new(nleaves: usize) -> Self {
        assert!(nleaves > 0, "the tree has at least one leaf");
        Self { nleaves }
    }

    /// Return the global number of leaves.
    pub fn nleaves(&self) -> usize {
        self.nleaves
    }
}

impl FmmSizes for IndexFmm {
    index_sizes!();
}

impl PairOperator for IndexFmm {
    fn p2m(&mut self, _leaf: MortonKey, sources: &[u32], multipole: &mut [u32]) {
        add_indices(sources, multipole);
    }

    fn m2m(&mut self, _: MortonKey, _: MortonKey, _: usize, child: &[u32], parent: &mut [u32]) {
        add(child, parent);
    }

    fn m2l(&mut self, _: MortonKey, _: MortonKey, _: usize, source: &[u32], target: &mut [u32]) {
        add(source, target);
    }

    fn p2l(&mut self, _: MortonKey, _: MortonKey, sources: &[u32], local: &mut [u32]) {
        add_indices(sources, local);
    }

    fn l2l(&mut self, _: MortonKey, _: MortonKey, _: usize, parent: &[u32], child: &mut [u32]) {
        add(parent, child);
    }

    fn l2p(&mut self, _leaf: MortonKey, local: &[u32], _: &[u32], target_output: &mut [u32]) {
        add_at_points(local, target_output, self.nleaves);
    }

    fn m2p(
        &mut self,
        _source: MortonKey,
        _target: MortonKey,
        multipole: &[u32],
        _target_input: &[u32],
        target_output: &mut [u32],
    ) {
        add_at_points(multipole, target_output, self.nleaves);
    }

    fn p2p(
        &mut self,
        _source: MortonKey,
        _target: MortonKey,
        sources: &[u32],
        _target_input: &[u32],
        target_output: &mut [u32],
    ) {
        add_indices_at_points(sources, target_output, self.nleaves);
    }
}

/// How [`BatchedIndexFmm`] walks the views of a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Walk {
    /// Target by target through the target-centric rows (`row(t)`), as a host operator
    /// does (design §6.5). Leaf outputs are split into per-row chunks first.
    Rows,
    /// Through the groupings, as a GEMM operator does (design §6.4): every octant or
    /// offset batch in group order, gathering its sources into scratch and
    /// scatter-adding into its targets. Views without a grouping are walked through
    /// their raw arrays (`row_offsets()`, `entries()`), as a device would.
    Groupings,
}

/// The index FMM as a batched [`FmmOperator`], without the per-pair adapter; see the
/// [module documentation](self).
///
/// Both walks add each target's contributions in the order of its row, as the
/// [accumulation rule](super::operator#accumulation-rule) requires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedIndexFmm {
    nleaves: usize,
    walk: Walk,
    /// The gathered sources of one batch ([`Walk::Groupings`]).
    scratch: Vec<u32>,
}

impl BatchedIndexFmm {
    /// Create the operators for a tree with `nleaves` leaves in total, walking `walk`.
    ///
    /// # Panics
    ///
    /// Panics if `nleaves` is zero.
    pub fn new(nleaves: usize, walk: Walk) -> Self {
        assert!(nleaves > 0, "the tree has at least one leaf");
        Self {
            nleaves,
            walk,
            scratch: Vec::new(),
        }
    }

    /// Return the global number of leaves.
    pub fn nleaves(&self) -> usize {
        self.nleaves
    }

    /// Return the walk.
    pub fn walk(&self) -> Walk {
        self.walk
    }

    /// Add `input[source]` to `output[target]` for every pair of a grouped view.
    fn translate<G: Copy + Into<usize>>(
        &mut self,
        view: &GroupedCsr<G>,
        input: LevelSlice<'_, u32>,
        mut output: LevelSliceMut<'_, u32>,
    ) {
        match self.walk {
            Walk::Rows => {
                for (t, out) in output.chunks_mut().enumerate() {
                    for &s in view.row(t).0 {
                        add(input.chunk(s as usize), out);
                    }
                }
            }
            Walk::Groupings => {
                let size = input.size();
                for g in 0..view.ngroups() {
                    let (targets, sources) = view.batch(g);
                    // Gather the sources into one column-major block, then scatter-add
                    // column k into target k (the "translation" is the identity).
                    self.scratch.clear();
                    for &s in sources {
                        self.scratch.extend_from_slice(input.chunk(s as usize));
                    }
                    for (&t, column) in targets.iter().zip(self.scratch.chunks_exact(size)) {
                        add(column, output.chunk_mut(t as usize));
                    }
                }
            }
        }
    }

    /// Call `pair(t, entry)` for every entry of every row of `view`, row by row.
    fn walk_csr(&self, view: &Csr, mut pair: impl FnMut(usize, usize)) {
        match self.walk {
            Walk::Rows => {
                for t in 0..view.nrows() {
                    for &entry in view.row(t) {
                        pair(t, entry as usize);
                    }
                }
            }
            Walk::Groupings => {
                let (offsets, entries) = (view.row_offsets(), view.entries());
                for t in 0..offsets.len() - 1 {
                    for &entry in &entries[offsets[t] as usize..offsets[t + 1] as usize] {
                        pair(t, entry as usize);
                    }
                }
            }
        }
    }

    /// Call `leaf(r, entry, output)` for every entry of every row of the leaf view
    /// `view`, with the output chunk of row r.
    fn walk_leaves(
        &self,
        view: &Csr,
        mut target_output: LeafSliceMut<'_, u32>,
        mut leaf: impl FnMut(usize, &mut [u32]),
    ) {
        match self.walk {
            Walk::Rows => {
                for (r, out) in target_output.chunks_mut().enumerate() {
                    for &entry in view.row(r) {
                        leaf(entry as usize, out);
                    }
                }
            }
            Walk::Groupings => {
                self.walk_csr(view, |r, entry| leaf(entry, target_output.chunk_mut(r)));
            }
        }
    }
}

impl FmmSizes for BatchedIndexFmm {
    index_sizes!();
}

impl FmmOperator for BatchedIndexFmm {
    fn p2m(&mut self, batch: P2m<'_, u32>) {
        let P2m {
            leaves,
            sources,
            mut multipoles,
            ..
        } = batch;
        self.walk_csr(leaves, |t, j| {
            add_indices(sources.chunk(j), multipoles.chunk_mut(t));
        });
    }

    fn m2m(&mut self, batch: M2m<'_, u32>) {
        self.translate(batch.children, batch.child_multipoles, batch.multipoles);
    }

    fn m2l(&mut self, batch: M2l<'_, u32>) {
        self.translate(batch.pairs, batch.multipoles, batch.locals);
    }

    fn p2l(&mut self, batch: P2l<'_, u32>) {
        let P2l {
            x,
            sources,
            mut locals,
            ..
        } = batch;
        self.walk_csr(x, |t, j| add_indices(sources.chunk(j), locals.chunk_mut(t)));
    }

    fn l2l(&mut self, batch: L2l<'_, u32>) {
        self.translate(batch.parents, batch.parent_locals, batch.locals);
    }

    fn l2p(&mut self, batch: L2p<'_, u32>) {
        let nleaves = self.nleaves;
        let L2p {
            boxes,
            locals,
            target_output,
            ..
        } = batch;
        self.walk_leaves(boxes, target_output, |i, out| {
            add_at_points(locals.chunk(i), out, nleaves);
        });
    }

    fn m2p(&mut self, batch: M2p<'_, u32>) {
        let nleaves = self.nleaves;
        let M2p {
            w,
            multipoles,
            target_output,
            ..
        } = batch;
        self.walk_leaves(w, target_output, |s, out| {
            add_at_points(multipoles.chunk(s), out, nleaves);
        });
    }

    fn p2p(&mut self, batch: P2p<'_, u32>) {
        let nleaves = self.nleaves;
        let P2p {
            near,
            sources,
            target_output,
            ..
        } = batch;
        self.walk_leaves(near, target_output, |j, out| {
            add_indices_at_points(sources.chunk(j), out, nleaves);
        });
    }
}

/// The global numbering of the leaves: every leaf's position in the global Morton order
/// of all leaves.
///
/// The ranks hold consecutive ranges of the Morton order, so the global index of a local
/// leaf is the number of leaves on lower ranks plus its position in this rank's Morton
/// order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalLeaves {
    nleaves: usize,
    offset: usize,
    /// Global index of every local leaf, in leaf order.
    indices: Vec<u32>,
}

impl GlobalLeaves {
    /// Number the leaves of every rank.
    ///
    /// # Collective operation
    /// One all-gather of the local leaf counts.
    ///
    /// # Panics
    ///
    /// Panics if the global number of leaves does not fit into `u32`.
    pub fn new<C: CommunicatorCollectives>(plan: &Plan, comm: &C) -> Self {
        let counts = gather_to_all(&[plan.index().leaves().nlocal()], comm);
        let offset = counts[..comm.rank() as usize].iter().sum();
        Self::with_offset(plan, offset, counts.iter().sum())
    }

    /// Number the local leaves of `plan` from `offset`, in a tree of `nleaves` leaves;
    /// local. `GlobalLeaves::with_offset(plan, 0, nlocal)` numbers a one-rank tree.
    ///
    /// # Panics
    ///
    /// Panics if `nleaves` does not fit into `u32` or the local leaves do not fit
    /// between `offset` and `nleaves`.
    pub fn with_offset(plan: &Plan, offset: usize, nleaves: usize) -> Self {
        let leaves = plan.index().leaves();
        let nlocal = leaves.nlocal();
        assert!(
            u32::try_from(nleaves).is_ok(),
            "the number of leaves fits into u32"
        );
        assert!(offset + nlocal <= nleaves, "the local leaves fit the tree");
        let mut by_key: Vec<usize> = (0..nlocal).collect();
        by_key.sort_unstable_by_key(|&j| leaves.key(j));
        let mut indices = vec![0u32; nlocal];
        for (position, j) in by_key.into_iter().enumerate() {
            indices[j] = (offset + position) as u32;
        }
        Self {
            nleaves,
            offset,
            indices,
        }
    }

    /// Return the global number of leaves.
    pub fn nleaves(&self) -> usize {
        self.nleaves
    }

    /// Return the global index of the first local leaf in Morton order.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Return the global index of every local leaf, in leaf order.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Return `local_counts` (one per local leaf, in leaf order) in global-index order,
    /// starting at [`offset`](Self::offset).
    pub fn by_global_index(&self, local_counts: &[usize]) -> Vec<usize> {
        assert_eq!(
            local_counts.len(),
            self.indices.len(),
            "one count per local leaf"
        );
        let mut sorted = vec![0usize; local_counts.len()];
        for (&count, &index) in local_counts.iter().zip(&self.indices) {
            sorted[index as usize - self.offset] = count;
        }
        sorted
    }

    /// Return the counts of every leaf of the tree, by global index, from each rank's
    /// `local_counts` (one per local leaf, in leaf order).
    ///
    /// # Collective operation
    /// One gather to all ranks.
    pub fn gather_counts<C: CommunicatorCollectives>(
        &self,
        comm: &C,
        local_counts: &[usize],
    ) -> Vec<usize> {
        gather_to_all(&self.by_global_index(local_counts), comm)
    }

    /// Set every source value of every local leaf to the leaf's global index.
    ///
    /// # Panics
    ///
    /// Panics if `sources` does not hold the local leaves.
    pub fn fill_sources(&self, mut sources: LeafSliceMut<'_, u32>) {
        assert_eq!(sources.nleaves(), self.indices.len(), "the local leaves");
        for (chunk, &index) in sources.chunks_mut().zip(&self.indices) {
            chunk.fill(index);
        }
    }
}

/// Check that every target point of every local leaf holds, at index j, the number of
/// source points of leaf j, `global_source_counts[j]`.
///
/// `target_output` holds the local leaves of `index`, with `global_source_counts.len()`
/// values per point. On failure the message names the first failing leaf and point, and
/// the first indices with a wrong count.
pub fn check_counts(
    index: &BoxIndex,
    target_output: &LeafStore<u32>,
    global_source_counts: &[usize],
) -> Result<(), String> {
    let nleaves = global_source_counts.len();
    for j in 0..target_output.nleaves() {
        let chunk = target_output.chunk(j);
        if chunk.len() != target_output.count(j) * nleaves {
            return Err(format!(
                "leaf {}: {} values for {} points of {nleaves} indices",
                index.leaf_key(j),
                chunk.len(),
                target_output.count(j)
            ));
        }
        for (point, values) in chunk.chunks_exact(nleaves).enumerate() {
            let wrong: Vec<(usize, u32, usize)> = values
                .iter()
                .zip(global_source_counts)
                .enumerate()
                .filter(|&(_, (&count, &expected))| count as usize != expected)
                .map(|(i, (&count, &expected))| (i, count, expected))
                .collect();
            if !wrong.is_empty() {
                let shown = &wrong[..wrong.len().min(8)];
                return Err(format!(
                    "leaf {}, point {point}: {} of {nleaves} indices have a wrong count, \
                     (index, count, expected): {shown:?}",
                    index.leaf_key(j),
                    wrong.len()
                ));
            }
        }
    }
    Ok(())
}

/// The implementation of the index FMM that [`run_index_fmm`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexPath {
    /// [`IndexFmm`] through the [`PerPair`] adapter.
    PerPair,
    /// [`BatchedIndexFmm`] with the given walk.
    Batched(Walk),
}

/// Run the index FMM on `plan` with the given counts per local leaf, and check the
/// result on every rank.
///
/// Returns `Ok(())` on every rank if all ranks pass. Otherwise every rank returns an
/// error, carrying its own failure message if it failed.
///
/// # Collective operation
/// On `comm`, the communicator of the octree of `plan`: [`GlobalLeaves::new`],
/// [`GlobalLeaves::gather_counts`], [`Evaluator::new`], [`Evaluator::evaluate`] and one
/// all-reduce, in that order, on every rank.
pub fn run_index_fmm<C: CommunicatorCollectives>(
    plan: &Plan,
    comm: &C,
    path: IndexPath,
    source_counts: &[usize],
    target_counts: &[usize],
) -> Result<(), String> {
    let leaves = GlobalLeaves::new(plan, comm);
    let global_counts = leaves.gather_counts(comm, source_counts);
    let nleaves = leaves.nleaves();
    let local = match path {
        IndexPath::PerPair => evaluate_and_check(
            plan,
            comm,
            PerPair(IndexFmm::new(nleaves)),
            &leaves,
            &global_counts,
            source_counts,
            target_counts,
        ),
        IndexPath::Batched(walk) => evaluate_and_check(
            plan,
            comm,
            BatchedIndexFmm::new(nleaves, walk),
            &leaves,
            &global_counts,
            source_counts,
            target_counts,
        ),
    };

    let mut all_passed = false;
    comm.all_reduce_into(
        &local.is_ok(),
        &mut all_passed,
        SystemOperation::logical_and(),
    );
    match local {
        Err(message) => Err(message),
        Ok(()) if all_passed => Ok(()),
        Ok(()) => Err("the index FMM failed on another rank".to_string()),
    }
}

/// Build an evaluator with `operator`, fill its sources with the global leaf indices,
/// evaluate and check. Collective; an evaluator error is returned on every rank.
fn evaluate_and_check<C: CommunicatorCollectives, Op: FmmOperator<Value = u32>>(
    plan: &Plan,
    comm: &C,
    operator: Op,
    leaves: &GlobalLeaves,
    global_counts: &[usize],
    source_counts: &[usize],
    target_counts: &[usize],
) -> Result<(), String> {
    let mut evaluator = Evaluator::new(plan, comm, operator, source_counts, target_counts)
        .map_err(|error| format!("building the evaluator: {error}"))?;
    leaves.fill_sources(evaluator.local_sources_mut());
    evaluator.evaluate();
    check_counts(plan.index(), evaluator.target_output_store(), global_counts)
}

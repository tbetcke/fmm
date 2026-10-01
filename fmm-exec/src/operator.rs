//! The Laplace operator on the level-batched interface of `nd-fmm-plan`, host path
//! (C3.1).
//!
//! [`LaplaceOperator`] implements [`FmmOperator`] for the kernel 1/|x − y|
//! (CONVENTIONS §3.1): the translations apply the tables of [`Tables`], looked up by
//! child index and offset index, and the leaf operators and P2P call `nd_fmm_ref` in the
//! leaf-scaled coordinates of CONVENTIONS §3.13. No operator formula is derived here.
//! It also implements [`PairOperator`], so that `PerPair<LaplaceOperator<T>>` runs the
//! same kernels pair by pair, for comparison.
//!
//! # Data
//!
//! - Multipoles and locals: the scaled coefficients of CONVENTIONS §3.7 in the real
//!   storage of §3.6, (p + 1)² values on every level.
//! - Source chunk of a leaf with n points, 4n values (§3.13, "Source chunks"): the n
//!   leaf-scaled coordinate triples u, then the n charges.
//! - Target input, 3n values: the leaf-scaled positions (§3.13, "Target input and
//!   output").
//! - Target output, n values φ̂, or 4n with gradients: φ̂ for every point, then the
//!   triples ĝ. φ̂ = r_t Σ q / |x − y| and ĝ = r_t² ∇ₓ Σ q / |x − y|, in units of the
//!   target leaf; 1/(4π) and r_t are applied once, by the FMM, when output leaves it
//!   (§3.1, §3.13, "Output").
//!
//! Chunks are read in place: the coordinates with `as_chunks::<3>()`, never copied.
//!
//! # Operators
//!
//! With (ĉ(s|t), r̂(s|t)) the exact frame of box s seen from box t
//! ([`relative_frame`]), n_s source and n_t target points, and the costs of one pair:
//!
//! | Operator | Computes | Frame or table | Cost |
//! | --- | --- | --- | --- |
//! | P2M at leaf s | `leaf::p2m` | ((0, 0, 0), 1) on u_s (§3.13) | O(n_s p²) |
//! | M2M child c → parent | table of o = `child_index(c)` (§3.12) | (½ s_o, ½) → (0, 1) | (p + 1)⁴, or ≈ 3p³ by rotation |
//! | M2L source s → target t | table of `m2l_offset_index(index(t) − index(s))` (§3.12) | (0, 1) → (2d, 1) | (p + 1)⁴ dense, (p + 1)⁴ + O(p³) classes, ≈ (10/3) p³ rotation |
//! | P2L leaf s → box t | `leaf::p2l` | (ĉ(t\|s), r̂(t\|s)) on u_s | O(n_s p²) |
//! | L2L parent → child c | table of o = `child_index(c)` | (0, 1) → (½ s_o, ½) | (p + 1)⁴, or ≈ 3p³ by rotation |
//! | L2P at leaf t | `leaf::l2p` | ((0, 0, 0), 1) on u_t | O(n_t p²) |
//! | M2P box s → leaf t | `leaf::m2p` | (ĉ(s\|t), r̂(s\|t)) on u_t | O(n_t p²) |
//! | P2P leaf s → leaf t | `p2p::p2p` | ŷ = ĉ(s\|t) + r̂(s\|t) u_s, u_s itself for s = t | O(n_s n_t), plus O(n_s) to map |
//!
//! (§3.11, "L2P and M2P", and §3.13, "Operators in scaled coordinates", derive each
//! line.) The gradient, when wanted, roughly quadruples the cost of L2P, M2P and P2P.
//! Every operator adds into its output; a leaf with no points is a no-op.
//!
//! # Batched execution and order
//!
//! Each level call runs serially, target by target in index order, through the plan's
//! target-centric view: box t of the level for P2M, M2M, M2L, P2L and L2L (row t lines
//! up with chunk t of the output level), row r, local leaf `leaves.start + r`, for
//! L2P, M2P and P2P. Each target's contributions are added in the order of its row, as
//! the plan's accumulation rule requires (`nd_fmm_plan::operator`): children by octant
//! (M2M), sources by offset index (M2L), leaves by leaf index (P2L, P2P), boxes by box
//! index (M2P), and the single parent (L2L) or leaf (P2M, L2P). With the evaluator's
//! fixed call order this fixes every floating-point sum. The tables are chosen by the
//! entry's octant or offset index; debug builds check that it agrees with the keys.
//!
//! The body of one target is a function of that target's output slice, the call's
//! shared inputs and one scratch set (`Kernels::*_target`), so the target loop can be
//! run by a parallel iterator unchanged (T10).
//!
//! # Scratch
//!
//! The interface passes `&mut self`, so the operator owns its scratch outright, with
//! no `RefCell`: an `nd_fmm_ref::Workspace` for p, the [`TableScratch`] of its tables,
//! and one buffer of `max_leaf_points` points into which P2P maps the sources of
//! another leaf. All three are sized at construction; no operator allocates.
//!
//! # Example
//!
//! One charge, P2M at its leaf, M2L to a leaf three boxes away, L2P at a target there:
//!
//! ```
//! use nd_fmm_exec::geometry::{Domain, leaf_coordinates, radius};
//! use nd_fmm_exec::operator::LaplaceOperator;
//! use nd_fmm_exec::tables::{M2lStrategy, Tables};
//! use nd_octree::{PhysicalBox, morton};
//!
//! let domain = Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap();
//! let p = 8;
//! let tables = Tables::<f64>::build(p, M2lStrategy::Rotation);
//! let mut op = LaplaceOperator::new(tables, false, 1);
//!
//! // A unit charge at y in box s and a target at x in box t, offset (3, 0, 0) on level 3.
//! let s = morton::from_index_and_level([1, 2, 2], 3);
//! let t = morton::from_index_and_level([4, 2, 2], 3);
//! let (y, x) = ([0.2, 0.3, 0.3], [0.55, 0.3, 0.3]);
//! let [u0, u1, u2] = leaf_coordinates::<f64>(y, s, &domain);
//! let sources = [u0, u1, u2, 1.0]; // the coordinates, then the charge
//! let target = leaf_coordinates::<f64>(x, t, &domain);
//!
//! let n = (p + 1) * (p + 1);
//! let (mut multipole, mut local, mut phi_hat) = (vec![0.0; n], vec![0.0; n], [0.0]);
//! op.p2m_leaf(&sources, &mut multipole);
//! op.m2l_pair(s, t, &multipole, &mut local);
//! op.l2p_leaf(&local, &target, &mut phi_hat);
//!
//! // φ̂ is in units of the target leaf: φ = φ̂ / r_t, without 4π (CONVENTIONS §3.13).
//! let phi = phi_hat[0] / radius(3, &domain);
//! assert!((phi - 1.0 / 0.35).abs() < 1e-6 * phi);
//! ```

use core::fmt;

use mpi::traits::Equivalence;
use nd_fmm_math::RealScalar;
use nd_fmm_plan::index::BoxIndex;
use nd_fmm_plan::lists::{Children, Csr, Parents, VList};
use nd_fmm_plan::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PairOperator,
};
use nd_fmm_plan::store::{LeafSlice, LevelSlice};
use nd_fmm_ref::{Frame, Workspace, leaf, p2p};
use nd_fmm_tables::geometry::m2l_offset_index;
use nd_octree::{MortonKey, morton};

use crate::geometry::relative_frame;
use crate::tables::{TableScratch, Tables};

/// Values per source point: the leaf-scaled coordinates and the charge (CONVENTIONS
/// §3.13, "Source chunks").
pub const SOURCE_POINT_SIZE: usize = 4;

/// Values per target point in the target input: the leaf-scaled position (CONVENTIONS
/// §3.13, "Target input and output").
pub const TARGET_INPUT_POINT_SIZE: usize = 3;

/// The Laplace kernel on the batched interface of `nd-fmm-plan`; see the
/// [module documentation](self).
#[derive(Clone, Debug)]
pub struct LaplaceOperator<T: RealScalar> {
    kernels: Kernels<T>,
    scratch: Scratch<T>,
}

/// The read-only part of the operator: what every target body shares.
#[derive(Clone, Debug)]
pub(crate) struct Kernels<T: RealScalar> {
    p: usize,
    tables: Tables<T>,
    gradients: bool,
    max_leaf_points: usize,
}

/// One scratch set: what a target body writes besides its output.
#[derive(Clone, Debug)]
pub(crate) struct Scratch<T: RealScalar> {
    workspace: Workspace<T>,
    tables: TableScratch<T>,
    /// The sources of another leaf in the target's coordinates (P2P); fixed length.
    mapped: Vec<[T; 3]>,
}

/// The shared inputs of one box call (P2M, M2M, M2L, P2L, L2L): its level, the index,
/// its view and its input.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BoxCall<'a, V, I> {
    level: usize,
    index: &'a BoxIndex,
    view: &'a V,
    input: I,
}

/// The shared inputs of one leaf call (L2P, M2P, P2P): its level, the index, the leaf
/// index of row 0, its view, its input and the target input of its rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LeafCall<'a, T, I> {
    level: usize,
    index: &'a BoxIndex,
    first_leaf: usize,
    view: &'a Csr,
    input: I,
    target_input: LeafSlice<'a, T>,
}

impl<T: RealScalar> LaplaceOperator<T> {
    /// Creates the operator for the degree of `tables`, with gradients in the target
    /// output if `gradients`, for leaves with at most `max_leaf_points` source points.
    ///
    /// `max_leaf_points` must bound the source count of every leaf that P2P reads,
    /// local and ghost: on several ranks a global maximum, one all-reduce, taken by the
    /// caller. Allocates the scratch: a `Workspace` for p, the tables' scratch and
    /// 3 `max_leaf_points` values for P2P.
    pub fn new(tables: Tables<T>, gradients: bool, max_leaf_points: usize) -> Self {
        let p = tables.p();
        let scratch = Scratch {
            workspace: Workspace::new(p),
            tables: tables.scratch(),
            mapped: vec![[T::zero(); 3]; max_leaf_points],
        };
        Self {
            kernels: Kernels {
                p,
                tables,
                gradients,
                max_leaf_points,
            },
            scratch,
        }
    }

    /// Returns the expansion degree p.
    pub fn p(&self) -> usize {
        self.kernels.p
    }

    /// Returns the tables.
    pub fn tables(&self) -> &Tables<T> {
        &self.kernels.tables
    }

    /// Returns whether the target output holds gradients.
    pub fn gradients(&self) -> bool {
        self.kernels.gradients
    }

    /// Returns the largest number of source points per leaf the operator accepts.
    pub fn max_leaf_points(&self) -> usize {
        self.kernels.max_leaf_points
    }

    /// Returns the capacity, in points, of the buffer into which P2P maps the sources
    /// of another leaf. It is `max_leaf_points` from construction on and never changes:
    /// no operator allocates (tests check it).
    pub fn scratch_capacity(&self) -> usize {
        self.scratch.mapped.capacity()
    }

    /// P2M: adds the multipole of the source chunk `sources` of a leaf, in the leaf's
    /// frame (CONVENTIONS §3.13: `leaf::p2m` at the unit frame on u). O(n p²).
    ///
    /// # Panics
    ///
    /// If `sources.len()` is not a multiple of 4, or `multipole` does not have length
    /// (p + 1)².
    pub fn p2m_leaf(&mut self, sources: &[T], multipole: &mut [T]) {
        self.kernels.p2m(sources, multipole, &mut self.scratch);
    }

    /// M2M: adds the multipole of `child` to that of `parent`, with the table of
    /// `morton::child_index(child)` (CONVENTIONS §3.12). Debug builds check that
    /// `parent` is the parent of `child`.
    ///
    /// # Panics
    ///
    /// If a slice does not have length (p + 1)².
    pub fn m2m_pair(
        &mut self,
        child: MortonKey,
        parent: MortonKey,
        child_multipole: &[T],
        parent_multipole: &mut [T],
    ) {
        let o = morton::child_index(child);
        debug_check_octant(child, parent, o);
        self.kernels
            .m2m(o, child_multipole, parent_multipole, &mut self.scratch);
    }

    /// M2L: adds the multipole of `source` to the local of `target`, a V-list pair, with
    /// the table of `m2l_offset_index(index(target) − index(source))` (CONVENTIONS
    /// §3.12).
    ///
    /// # Panics
    ///
    /// With both keys if the boxes lie on different levels or their offset is not a
    /// V-list offset; if a slice does not have length (p + 1)².
    pub fn m2l_pair(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        source_multipole: &[T],
        target_local: &mut [T],
    ) {
        let d = offset_index(source, target);
        self.kernels
            .m2l(d, source_multipole, target_local, &mut self.scratch);
    }

    /// P2L: adds the sources of the leaf `source` to the local of the box `target`, at
    /// the frame (ĉ(t|s), r̂(t|s)) (CONVENTIONS §3.13). O(n p²).
    ///
    /// # Panics
    ///
    /// If `sources.len()` is not a multiple of 4, or `target_local` does not have
    /// length (p + 1)².
    pub fn p2l_pair(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        sources: &[T],
        target_local: &mut [T],
    ) {
        let frame = relative_frame(target, source);
        self.kernels
            .p2l(&frame, sources, target_local, &mut self.scratch);
    }

    /// L2L: adds the local of `parent` to that of `child`, with the table of
    /// `morton::child_index(child)` (CONVENTIONS §3.12). Debug builds check that
    /// `parent` is the parent of `child`.
    ///
    /// # Panics
    ///
    /// If a slice does not have length (p + 1)².
    pub fn l2l_pair(
        &mut self,
        parent: MortonKey,
        child: MortonKey,
        parent_local: &[T],
        child_local: &mut [T],
    ) {
        let o = morton::child_index(child);
        debug_check_octant(child, parent, o);
        self.kernels
            .l2l(o, parent_local, child_local, &mut self.scratch);
    }

    /// L2P: adds the local `local` of a leaf at its target points, in the leaf's frame
    /// (CONVENTIONS §3.13: `leaf::l2p` at the unit frame on u). O(n p²).
    ///
    /// # Panics
    ///
    /// If `target_input.len()` is not a multiple of 3, `target_output` does not hold
    /// 1 (or with gradients 4) values per target point, or `local` does not have length
    /// (p + 1)².
    pub fn l2p_leaf(&mut self, local: &[T], target_input: &[T], target_output: &mut [T]) {
        self.kernels
            .l2p(local, target_input, target_output, &mut self.scratch);
    }

    /// M2P: adds the multipole of the box `source` at the target points of the leaf
    /// `target`, at the frame (ĉ(s|t), r̂(s|t)) (CONVENTIONS §3.13). O(n p²).
    ///
    /// # Panics
    ///
    /// As [`l2p_leaf`](Self::l2p_leaf).
    pub fn m2p_pair(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        source_multipole: &[T],
        target_input: &[T],
        target_output: &mut [T],
    ) {
        let frame = relative_frame(source, target);
        self.kernels.m2p(
            &frame,
            source_multipole,
            target_input,
            target_output,
            &mut self.scratch,
        );
    }

    /// P2P: adds the sources of the leaf `source` directly at the target points of the
    /// leaf `target` (CONVENTIONS §3.13). For `source == target` the source chunk is
    /// used as it is, so coincident points are skipped (§3.13, "Coincident pairs");
    /// otherwise the sources are first mapped to ŷ = ĉ(s|t) + r̂(s|t) u_s in the
    /// scratch buffer. O(n_s n_t).
    ///
    /// # Panics
    ///
    /// As [`l2p_leaf`](Self::l2p_leaf), if `sources.len()` is not a multiple of 4, or if
    /// a different leaf has more than `max_leaf_points` sources.
    pub fn p2p_pair(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        sources: &[T],
        target_input: &[T],
        target_output: &mut [T],
    ) {
        let frame = (source != target).then(|| relative_frame(source, target));
        self.kernels.p2p(
            frame.as_ref(),
            sources,
            target_input,
            target_output,
            &mut self.scratch,
        );
    }
}

/// The frame ((0, 0, 0), 1) of a leaf in its own coordinates.
#[inline]
fn unit_frame<T: RealScalar>() -> Frame<T> {
    Frame {
        centre: [T::zero(); 3],
        radius: T::one(),
    }
}

/// Splits a source chunk into its coordinate triples and its charges, in place.
#[inline]
fn source_points<T>(sources: &[T]) -> (&[[T; 3]], &[T]) {
    assert!(
        sources.len().is_multiple_of(SOURCE_POINT_SIZE),
        "a source chunk holds {SOURCE_POINT_SIZE} values per point, got {} values",
        sources.len()
    );
    let n = sources.len() / SOURCE_POINT_SIZE;
    let (coordinates, charges) = sources.split_at(3 * n);
    (coordinates.as_chunks::<3>().0, charges)
}

/// The potentials and, if wanted, the gradients of a target output chunk.
type TargetOutput<'a, T> = (&'a mut [T], Option<&'a mut [[T; 3]]>);

impl<T: RealScalar> Kernels<T> {
    /// Values per target point in the target output.
    fn output_point_size(&self) -> usize {
        if self.gradients { 4 } else { 1 }
    }

    /// Splits the target input into its positions and the target output into its
    /// potentials and gradients, in place.
    #[inline]
    fn targets<'a>(
        &self,
        target_input: &'a [T],
        target_output: &'a mut [T],
    ) -> (&'a [[T; 3]], TargetOutput<'a, T>) {
        assert!(
            target_input.len().is_multiple_of(TARGET_INPUT_POINT_SIZE),
            "a target input chunk holds {TARGET_INPUT_POINT_SIZE} values per point, got {} \
             values",
            target_input.len()
        );
        let n = target_input.len() / TARGET_INPUT_POINT_SIZE;
        let size = self.output_point_size();
        assert_eq!(
            target_output.len(),
            size * n,
            "a target output chunk holds {size} values per point, for {n} points"
        );
        let (potential, gradient) = target_output.split_at_mut(n);
        let gradient = self.gradients.then_some(gradient.as_chunks_mut::<3>().0);
        (target_input.as_chunks::<3>().0, (potential, gradient))
    }

    /// P2M at the unit frame.
    #[inline]
    fn p2m(&self, sources: &[T], multipole: &mut [T], scratch: &mut Scratch<T>) {
        let (points, charges) = source_points(sources);
        if points.is_empty() {
            return;
        }
        leaf::p2m(
            self.p,
            &unit_frame(),
            points,
            charges,
            &mut scratch.workspace,
            multipole,
        );
    }

    /// M2M with the table of child index `o`.
    #[inline]
    fn m2m(&self, o: usize, child: &[T], parent: &mut [T], scratch: &mut Scratch<T>) {
        self.tables.m2m(o, child, parent, &mut scratch.tables);
    }

    /// M2L with the table of offset index `d`.
    #[inline]
    fn m2l(&self, d: usize, multipole: &[T], local: &mut [T], scratch: &mut Scratch<T>) {
        self.tables.m2l(d, multipole, local, &mut scratch.tables);
    }

    /// L2L with the table of child index `o`.
    #[inline]
    fn l2l(&self, o: usize, parent: &[T], child: &mut [T], scratch: &mut Scratch<T>) {
        self.tables.l2l(o, parent, child, &mut scratch.tables);
    }

    /// P2L at `frame` = (ĉ(t|s), r̂(t|s)).
    #[inline]
    fn p2l(&self, frame: &Frame<T>, sources: &[T], local: &mut [T], scratch: &mut Scratch<T>) {
        let (points, charges) = source_points(sources);
        if points.is_empty() {
            return;
        }
        leaf::p2l(
            self.p,
            frame,
            points,
            charges,
            &mut scratch.workspace,
            local,
        );
    }

    /// L2P at the unit frame.
    #[inline]
    fn l2p(&self, local: &[T], target_input: &[T], target_output: &mut [T], s: &mut Scratch<T>) {
        let (targets, (potential, gradient)) = self.targets(target_input, target_output);
        if targets.is_empty() {
            return;
        }
        leaf::l2p(
            self.p,
            &unit_frame(),
            local,
            targets,
            &mut s.workspace,
            potential,
            gradient,
        );
    }

    /// M2P at `frame` = (ĉ(s|t), r̂(s|t)).
    #[inline]
    fn m2p(
        &self,
        frame: &Frame<T>,
        multipole: &[T],
        target_input: &[T],
        target_output: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let (targets, (potential, gradient)) = self.targets(target_input, target_output);
        if targets.is_empty() {
            return;
        }
        leaf::m2p(
            self.p,
            frame,
            multipole,
            targets,
            &mut scratch.workspace,
            potential,
            gradient,
        );
    }

    /// P2P: from the leaf's own chunk if `frame` is `None` (the self pair), otherwise
    /// from its sources mapped through `frame` = (ĉ(s|t), r̂(s|t)).
    #[inline]
    fn p2p(
        &self,
        frame: Option<&Frame<T>>,
        sources: &[T],
        target_input: &[T],
        target_output: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let (points, charges) = source_points(sources);
        let (targets, (potential, gradient)) = self.targets(target_input, target_output);
        if points.is_empty() || targets.is_empty() {
            return;
        }
        let points = match frame {
            None => points,
            Some(frame) => {
                assert!(
                    points.len() <= scratch.mapped.len(),
                    "a leaf has {} source points, more than max_leaf_points = {}",
                    points.len(),
                    scratch.mapped.len()
                );
                let mapped = &mut scratch.mapped[..points.len()];
                for (y, u) in mapped.iter_mut().zip(points) {
                    // r̂ u is exact (r̂ is a power of two); the sum rounds once (§3.13).
                    *y = core::array::from_fn(|k| frame.centre[k] + frame.radius * u[k]);
                }
                &*mapped
            }
        };
        p2p::p2p(points, charges, targets, potential, gradient);
    }

    /// The P2M body of box t: the source chunk of its leaf, if it has one.
    #[inline]
    pub(crate) fn p2m_target(
        &self,
        call: &BoxCall<'_, Csr, LeafSlice<'_, T>>,
        t: usize,
        multipole: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        for &j in call.view.row(t) {
            debug_assert_eq!(
                call.index.leaf_key(j as usize),
                call.index.key(call.level, t)
            );
            self.p2m(call.input.chunk(j as usize), multipole, scratch);
        }
    }

    /// The M2M body of parent t: its children, by octant.
    #[inline]
    pub(crate) fn m2m_target(
        &self,
        call: &BoxCall<'_, Children, LevelSlice<'_, T>>,
        t: usize,
        multipole: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let (children, octants) = call.view.row(t);
        for (&c, &o) in children.iter().zip(octants) {
            debug_check_octant(
                call.index.key(call.level + 1, c as usize),
                call.index.key(call.level, t),
                o as usize,
            );
            self.m2m(o as usize, call.input.chunk(c as usize), multipole, scratch);
        }
    }

    /// The M2L body of target t: its V-list sources, by offset index.
    #[inline]
    pub(crate) fn m2l_target(
        &self,
        call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
        t: usize,
        local: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let (sources, offsets) = call.view.row(t);
        for (&s, &d) in sources.iter().zip(offsets) {
            debug_assert_eq!(
                offset_index(
                    call.index.key(call.level, s as usize),
                    call.index.key(call.level, t)
                ),
                d as usize,
                "the offset index of a V-list entry disagrees with its keys"
            );
            self.m2l(d as usize, call.input.chunk(s as usize), local, scratch);
        }
    }

    /// The P2L body of box t: its X-list leaves, by leaf index.
    #[inline]
    pub(crate) fn p2l_target(
        &self,
        call: &BoxCall<'_, Csr, LeafSlice<'_, T>>,
        t: usize,
        local: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let target = call.index.key(call.level, t);
        for &j in call.view.row(t) {
            let frame = relative_frame(target, call.index.leaf_key(j as usize));
            self.p2l(&frame, call.input.chunk(j as usize), local, scratch);
        }
    }

    /// The L2L body of box t: its parent, if it has a row.
    #[inline]
    pub(crate) fn l2l_target(
        &self,
        call: &BoxCall<'_, Parents, LevelSlice<'_, T>>,
        t: usize,
        local: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let (parents, octants) = call.view.row(t);
        for (&parent, &o) in parents.iter().zip(octants) {
            debug_check_octant(
                call.index.key(call.level, t),
                call.index.key(call.level - 1, parent as usize),
                o as usize,
            );
            self.l2l(
                o as usize,
                call.input.chunk(parent as usize),
                local,
                scratch,
            );
        }
    }

    /// The L2P body of row r: the local of the leaf's box.
    #[inline]
    pub(crate) fn l2p_target(
        &self,
        call: &LeafCall<'_, T, LevelSlice<'_, T>>,
        r: usize,
        output: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        for &i in call.view.row(r) {
            let local = call.input.chunk(i as usize);
            self.l2p(local, call.target_input.chunk(r), output, scratch);
        }
    }

    /// The M2P body of row r: its W-list boxes, by box index.
    #[inline]
    pub(crate) fn m2p_target(
        &self,
        call: &LeafCall<'_, T, LevelSlice<'_, T>>,
        r: usize,
        output: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let target = call.index.leaf_key(call.first_leaf + r);
        for &s in call.view.row(r) {
            let frame = relative_frame(call.index.key(call.level + 1, s as usize), target);
            let multipole = call.input.chunk(s as usize);
            self.m2p(
                &frame,
                multipole,
                call.target_input.chunk(r),
                output,
                scratch,
            );
        }
    }

    /// The P2P body of row r: its near list, the leaf itself included, by leaf index.
    #[inline]
    pub(crate) fn p2p_target(
        &self,
        call: &LeafCall<'_, T, LeafSlice<'_, T>>,
        r: usize,
        output: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let leaf = call.first_leaf + r;
        let target = call.index.leaf_key(leaf);
        for &j in call.view.row(r) {
            let j = j as usize;
            let frame = (j != leaf).then(|| relative_frame(call.index.leaf_key(j), target));
            self.p2p(
                frame.as_ref(),
                call.input.chunk(j),
                call.target_input.chunk(r),
                output,
                scratch,
            );
        }
    }
}

/// A key with its level and index, for panic messages.
struct Key(MortonKey);

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (level, index) = morton::decode(self.0);
        write!(f, "{} (level {level}, index {index:?})", self.0)
    }
}

/// The position of index(target) − index(source) in `V_LIST_DIRECTIONS`.
///
/// # Panics
///
/// With both keys if the levels differ or the offset is not a V-list offset.
fn offset_index(source: MortonKey, target: MortonKey) -> usize {
    let (source_level, source_index) = morton::decode(source);
    let (target_level, target_index) = morton::decode(target);
    assert_eq!(
        source_level,
        target_level,
        "M2L between boxes on different levels: source {}, target {}",
        Key(source),
        Key(target)
    );
    let d = [0, 1, 2].map(|k| target_index[k] as i64 - source_index[k] as i64);
    m2l_offset_index(d).unwrap_or_else(|| {
        panic!(
            "M2L from source {} to target {}: the offset {d:?} is not a V-list offset",
            Key(source),
            Key(target)
        )
    })
}

/// Debug builds: checks that `parent` is the parent of `child` and `o` its child index.
#[inline]
fn debug_check_octant(child: MortonKey, parent: MortonKey, o: usize) {
    debug_assert!(
        morton::parent(child) == Some(parent),
        "{} is not the parent of {}",
        Key(parent),
        Key(child)
    );
    debug_assert_eq!(morton::child_index(child), o, "octant of {}", Key(child));
}

impl<T: RealScalar + Equivalence + Default> FmmSizes for LaplaceOperator<T> {
    type Value = T;

    /// (p + 1)² on every level.
    fn multipole_size(&self, _level: usize) -> usize {
        (self.kernels.p + 1).pow(2)
    }

    /// (p + 1)² on every level.
    fn local_size(&self, _level: usize) -> usize {
        (self.kernels.p + 1).pow(2)
    }

    /// [`SOURCE_POINT_SIZE`], 4.
    fn source_point_size(&self) -> usize {
        SOURCE_POINT_SIZE
    }

    /// [`TARGET_INPUT_POINT_SIZE`], 3.
    fn target_input_point_size(&self) -> usize {
        TARGET_INPUT_POINT_SIZE
    }

    /// 1, or 4 with gradients.
    fn target_output_point_size(&self) -> usize {
        self.kernels.output_point_size()
    }
}

/// Serial, target by target in index order, each target's row in order; see the
/// [module documentation](self#batched-execution-and-order).
impl<T: RealScalar + Equivalence + Default> FmmOperator for LaplaceOperator<T> {
    fn p2m(&mut self, batch: P2m<'_, T>) {
        let P2m {
            level,
            index,
            leaves,
            sources,
            mut multipoles,
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: leaves,
            input: sources,
        };
        for (t, multipole) in multipoles.chunks_mut().enumerate() {
            self.kernels
                .p2m_target(&call, t, multipole, &mut self.scratch);
        }
    }

    fn m2m(&mut self, batch: M2m<'_, T>) {
        let M2m {
            level,
            index,
            children,
            child_multipoles,
            mut multipoles,
            ..
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: children,
            input: child_multipoles,
        };
        for (t, multipole) in multipoles.chunks_mut().enumerate() {
            self.kernels
                .m2m_target(&call, t, multipole, &mut self.scratch);
        }
    }

    fn m2l(&mut self, batch: M2l<'_, T>) {
        let M2l {
            level,
            index,
            pairs,
            multipoles,
            mut locals,
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: pairs,
            input: multipoles,
        };
        for (t, local) in locals.chunks_mut().enumerate() {
            self.kernels.m2l_target(&call, t, local, &mut self.scratch);
        }
    }

    fn p2l(&mut self, batch: P2l<'_, T>) {
        let P2l {
            level,
            index,
            x,
            sources,
            mut locals,
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: x,
            input: sources,
        };
        for (t, local) in locals.chunks_mut().enumerate() {
            self.kernels.p2l_target(&call, t, local, &mut self.scratch);
        }
    }

    fn l2l(&mut self, batch: L2l<'_, T>) {
        let L2l {
            level,
            index,
            parents,
            parent_locals,
            mut locals,
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: parents,
            input: parent_locals,
        };
        for (t, local) in locals.chunks_mut().enumerate() {
            self.kernels.l2l_target(&call, t, local, &mut self.scratch);
        }
    }

    fn l2p(&mut self, batch: L2p<'_, T>) {
        let L2p {
            level,
            index,
            leaves,
            boxes,
            locals,
            target_input,
            mut target_output,
        } = batch;
        let call = LeafCall {
            level,
            index,
            first_leaf: leaves.start,
            view: boxes,
            input: locals,
            target_input,
        };
        for (r, output) in target_output.chunks_mut().enumerate() {
            self.kernels.l2p_target(&call, r, output, &mut self.scratch);
        }
    }

    fn m2p(&mut self, batch: M2p<'_, T>) {
        let M2p {
            level,
            index,
            leaves,
            w,
            multipoles,
            target_input,
            mut target_output,
        } = batch;
        let call = LeafCall {
            level,
            index,
            first_leaf: leaves.start,
            view: w,
            input: multipoles,
            target_input,
        };
        for (r, output) in target_output.chunks_mut().enumerate() {
            self.kernels.m2p_target(&call, r, output, &mut self.scratch);
        }
    }

    fn p2p(&mut self, batch: P2p<'_, T>) {
        let P2p {
            level,
            index,
            leaves,
            near,
            sources,
            target_input,
            mut target_output,
        } = batch;
        let call = LeafCall {
            level,
            index,
            first_leaf: leaves.start,
            view: near,
            input: sources,
            target_input,
        };
        for (r, output) in target_output.chunks_mut().enumerate() {
            self.kernels.p2p_target(&call, r, output, &mut self.scratch);
        }
    }
}

/// The same kernels pair by pair, for `nd_fmm_plan::operator::PerPair`: tables by the
/// given octant or offset index (debug builds check it against the keys), frames from
/// the keys, the self pair of P2P by `source == target`.
impl<T: RealScalar + Equivalence + Default> PairOperator for LaplaceOperator<T> {
    fn p2m(&mut self, _leaf: MortonKey, sources: &[T], multipole: &mut [T]) {
        self.p2m_leaf(sources, multipole);
    }

    fn m2m(
        &mut self,
        child: MortonKey,
        parent: MortonKey,
        octant: usize,
        child_multipole: &[T],
        parent_multipole: &mut [T],
    ) {
        debug_check_octant(child, parent, octant);
        self.kernels
            .m2m(octant, child_multipole, parent_multipole, &mut self.scratch);
    }

    fn m2l(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        offset_index: usize,
        source_multipole: &[T],
        target_local: &mut [T],
    ) {
        debug_assert_eq!(
            self::offset_index(source, target),
            offset_index,
            "the offset index of a V-list pair disagrees with its keys"
        );
        self.kernels.m2l(
            offset_index,
            source_multipole,
            target_local,
            &mut self.scratch,
        );
    }

    fn p2l(&mut self, source: MortonKey, target: MortonKey, sources: &[T], local: &mut [T]) {
        self.p2l_pair(source, target, sources, local);
    }

    fn l2l(
        &mut self,
        parent: MortonKey,
        child: MortonKey,
        octant: usize,
        parent_local: &[T],
        child_local: &mut [T],
    ) {
        debug_check_octant(child, parent, octant);
        self.kernels
            .l2l(octant, parent_local, child_local, &mut self.scratch);
    }

    fn l2p(&mut self, _leaf: MortonKey, local: &[T], target_input: &[T], output: &mut [T]) {
        self.l2p_leaf(local, target_input, output);
    }

    fn m2p(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        source_multipole: &[T],
        target_input: &[T],
        target_output: &mut [T],
    ) {
        self.m2p_pair(
            source,
            target,
            source_multipole,
            target_input,
            target_output,
        );
    }

    fn p2p(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        sources: &[T],
        target_input: &[T],
        target_output: &mut [T],
    ) {
        self.p2p_pair(source, target, sources, target_input, target_output);
    }
}

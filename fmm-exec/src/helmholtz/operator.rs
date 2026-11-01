//! The Helmholtz family's operator of the plan (Helmholtz CH3.1a;
//! docs/design/helmholtz-fmm-proposal.md §5.1–§5.3, §5.7, §6.2, §6.3; CONVENTIONS §3.15):
//! [`HelmholtzOperator`], for real and complex k with real and complex charges, RCR on
//! every level.
//!
//! # Levels and units
//!
//! Each level l works in the units of its boxes (half-width r_l = 1), with the wavenumber
//! k̂_l = k r_l formed in f64 and rounded to T once, as the leaf operators of
//! [`Densities`] form it; the expansions of level l carry the stored form of
//! `nd_fmm_ref::helmholtz::Level::new(k̂_l, 1)` and have degree p_l (the
//! [`degree`](mod@super::degree) module). The chunks and box blocks are those of [`Densities`]
//! at p_l: m expansions per box, one after the other, each of
//! [`Densities::multipole_planes`] or [`Densities::local_planes`] planes.
//!
//! # Operators
//!
//! | Kind | Body |
//! | --- | --- |
//! | P2M, L2P | [`Densities::p2m`], [`Densities::l2p`] in the leaf's own frame |
//! | M2M, L2L, M2L | `nd_fmm_tables::helmholtz::HelmholtzTables` (RCR with the level's tables; real shifts on each plane for real k) |
//! | P2L | `nd_fmm_ref::helmholtz::leaf::p2l` in the source leaf's units, at (ĉ(t\|s), r̂(t\|s)) |
//! | M2P | `nd_fmm_ref::helmholtz::leaf::m2p` in the target leaf's units, at (ĉ(s\|t), r̂(s\|t)) |
//! | P2P | `nd_fmm_ref::helmholtz::p2p::p2p` in the target leaf's units, the sources mapped as ĉ(s\|t) + r̂(s\|t) u |
//!
//! A box of level l holds r_l φ̃ in its own units, as a leaf's chunk does (CONVENTIONS
//! §3.13; `nd_fmm_tables::helmholtz`, "Levels and units"). P2L forms the local in the
//! source leaf's units, so it scales the charges by r̂(t|s) = r_t / r_s (exact, a power of
//! two); M2P needs nothing, as a multipole's factor 4π ik̂ carries the change of units.
//! P2L and M2P pass the reference a `Level` in the points' units: the wavenumber of the
//! leaf whose points they read, and the frame's radius as the other box's half-width, so
//! its power scale and damping are the other box's level's (the half-widths differ by
//! powers of two, which scale exactly). A W box lies on a finer level than the leaf, an X
//! leaf on a coarser one than the box; the operator finds their levels from the frame's
//! radius. M2P widens a real multipole (real k, real charges) to two planes in scratch.
//!
//! P2P, M2P and L2P accumulate into the target output chunk: P2P and M2P load each target's
//! two output columns of a vector as complex values, let the reference operator add its
//! terms to them, and store them back; L2P is [`Densities::l2p`]'s. Every operator gives
//! each charge vector exactly the one-vector operator, and adds (+=); none applies 1/(4π)
//! or the leaf radius (the output pass does, CONVENTIONS §3.13).
//!
//! P2P is the reference kernel, the trusted path, until CH3.4 adds the vectorised one.
//!
//! # Execution
//!
//! The level calls run on the shared drivers of `nd-fmm-exec` (Helmholtz CH2.0, CH3.5),
//! target by target in the plan's order, serially or on the operator's pool
//! ([`HelmholtzOperator::with_pool`]), each thread with its own scratch; the output is the
//! same at every thread count. The batched M2L does not apply (`driver::m2l_block` assumes
//! Laplace's layout; Helmholtz CH3.2). [`PairOperator`] runs the same bodies pair by pair.
//!
//! Yukawa (Re k = 0) is not served: its real path is CH2.1's.

use std::sync::{Arc, Mutex};

use mpi::traits::Equivalence;
use nd_fmm_math::{Complex, Layout};
use nd_fmm_plan::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PairOperator,
};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, leaf, p2p};
use nd_fmm_simd::{MatrixKernel, SimdScalar};
use nd_fmm_tables::MatrixSet;
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzScratch, HelmholtzTables};
use nd_octree::{MortonKey, morton};
use rayon::ThreadPool;
use thiserror::Error;

use super::{Densities, LeafScratch, Wavenumber, WavenumberKind};
use crate::driver::{self, Columns, Execution, HostKernel, debug_check_octant, offset_index};
use crate::geometry::{Domain, radius, relative_frame};

/// Why [`HelmholtzOperator::new`] refused its inputs.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum OperatorError {
    /// k is Yukawa's (Re k = 0), whose real path is Helmholtz CH2.1's.
    #[error("Yukawa (Re k = 0) is not served by this operator")]
    Yukawa,
    /// The densities are of another kind of k.
    #[error("the densities are for {densities:?} k, the wavenumber is {k:?}")]
    KindMismatch {
        /// The kind of the densities.
        densities: WavenumberKind,
        /// The kind of the wavenumber.
        k: WavenumberKind,
    },
    /// No degrees, so no levels.
    #[error("no levels")]
    NoLevels,
}

/// The Helmholtz-family operator (module documentation).
pub struct HelmholtzOperator<T: SimdScalar> {
    kernel: Kernel<T>,
    execution: Execution<Scratch<T>>,
}

/// What the bodies read: the path, the levels and the tables.
struct Kernel<T: SimdScalar> {
    k: Wavenumber,
    densities: Densities,
    gradients: bool,
    max_leaf_points: usize,
    /// r_l of each level, in the units of the points.
    radii: Vec<f64>,
    tables: HelmholtzTables<T>,
}

/// One thread's scratch.
struct Scratch<T: SimdScalar> {
    leaf: LeafScratch<T>,
    workspace: Workspace<T>,
    tables: HelmholtzScratch<T>,
    /// Another leaf's sources in the target's units (P2P).
    mapped: Vec<[T; 3]>,
    /// One vector's charges.
    charges: Vec<Complex<T>>,
    /// One multipole of two planes (M2P of a real multipole).
    widened: Vec<T>,
    /// Never used: no batched M2L.
    columns: Columns<T>,
}

impl<T: SimdScalar> HelmholtzOperator<T> {
    /// The operator of an octree on `domain` with `degrees.len()` levels of degrees p_l,
    /// for `k` and the path `densities` (its kind must be k's), with gradients in the
    /// target output if `gradients`, for leaves with at most `max_leaf_points` source
    /// points (every leaf P2P reads, local and ghost: on several ranks a global maximum).
    /// Builds the tables of every level with the default block budget
    /// ([`DEFAULT_BLOCK_BUDGET`]) and one scratch set; runs serially until
    /// [`with_pool`](Self::with_pool).
    ///
    /// # Errors
    ///
    /// [`OperatorError`]: Yukawa, a kind mismatch, no degrees.
    pub fn new(
        k: Wavenumber,
        densities: Densities,
        domain: &Domain,
        degrees: &[usize],
        gradients: bool,
        max_leaf_points: usize,
    ) -> Result<Self, OperatorError> {
        Self::with_block_budget(
            k,
            densities,
            domain,
            degrees,
            gradients,
            max_leaf_points,
            DEFAULT_BLOCK_BUDGET,
        )
    }

    /// As [`new`](Self::new), with `block_budget` bytes for the stored M2L y-blocks
    /// (`HelmholtzTables::build`); the output does not depend on it.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn with_block_budget(
        k: Wavenumber,
        densities: Densities,
        domain: &Domain,
        degrees: &[usize],
        gradients: bool,
        max_leaf_points: usize,
        block_budget: usize,
    ) -> Result<Self, OperatorError> {
        if k.kind() == WavenumberKind::Yukawa {
            return Err(OperatorError::Yukawa);
        }
        if densities.kind() != k.kind() {
            return Err(OperatorError::KindMismatch {
                densities: densities.kind(),
                k: k.kind(),
            });
        }
        if degrees.is_empty() {
            return Err(OperatorError::NoLevels);
        }
        let radii: Vec<f64> = (0..degrees.len()).map(|l| radius(l, domain)).collect();
        let k_root = Complex::new(k.re() * radii[0], k.im() * radii[0]);
        let tables = HelmholtzTables::build(k_root, degrees, block_budget);
        let kernel = Kernel {
            k,
            densities,
            gradients,
            max_leaf_points,
            radii,
            tables,
        };
        let execution = Execution {
            scratch: vec![Mutex::new(Scratch::new(&kernel))],
            pool: None,
            serial: false,
        };
        Ok(Self { kernel, execution })
    }

    /// Runs every level call in `pool`, each target on one of its threads, with one
    /// scratch set per thread. The output is bit for bit the serial operator's.
    pub fn with_pool(mut self, pool: Arc<ThreadPool>) -> Self {
        let n = pool.current_num_threads().max(1);
        self.execution.scratch = (0..n)
            .map(|_| Mutex::new(Scratch::new(&self.kernel)))
            .collect();
        self.execution.pool = Some(pool);
        self
    }

    /// The degree p_l of `level`.
    pub fn degree(&self, level: usize) -> usize {
        self.kernel.tables.degree(level)
    }

    /// The number of levels.
    pub fn nlevels(&self) -> usize {
        self.kernel.tables.nlevels()
    }

    /// The path.
    pub fn densities(&self) -> &Densities {
        &self.kernel.densities
    }

    /// The wavenumber.
    pub fn wavenumber(&self) -> Wavenumber {
        self.kernel.k
    }

    /// Whether the target output holds gradients.
    pub fn gradients(&self) -> bool {
        self.kernel.gradients
    }

    /// The RCR tables of the levels.
    pub fn tables(&self) -> &HelmholtzTables<T> {
        &self.kernel.tables
    }
}

impl<T: SimdScalar> Scratch<T> {
    fn new(kernel: &Kernel<T>) -> Self {
        let p_max = (0..kernel.tables.nlevels())
            .map(|l| kernel.tables.degree(l))
            .max()
            .unwrap_or(0);
        Self {
            leaf: LeafScratch::new(p_max),
            workspace: Workspace::new(p_max),
            tables: kernel.tables.scratch(),
            mapped: vec![[T::zero(); 3]; kernel.max_leaf_points],
            charges: vec![Complex::new(T::zero(), T::zero()); kernel.max_leaf_points],
            widened: vec![T::zero(); 2 * Layout::new(p_max).len()],
            columns: Columns {
                pack: Vec::new(),
                sources: Vec::new(),
                targets: Vec::new(),
            },
        }
    }
}

impl<T: SimdScalar> Kernel<T> {
    fn p(&self, level: usize) -> usize {
        self.tables.degree(level)
    }

    /// k̂ of `level` rounded to T.
    fn k_hat(&self, level: usize) -> Complex<T> {
        let k = self.tables.k_hat(level);
        Complex::new(T::from_f64(k.re), T::from_f64(k.im))
    }

    /// log₂ of a frame's radius, a power of two: the difference of two boxes' levels.
    fn levels_apart(frame: &Frame<T>) -> isize {
        frame.radius.to_f64().log2().round() as isize
    }

    /// The level `level` + `j`.
    fn shifted(level: usize, j: isize) -> usize {
        level
            .checked_add_signed(j)
            .expect("a frame's box lies on a level ≥ 0")
    }

    /// The points of a source chunk.
    fn source_points<'a>(&self, sources: &'a [T]) -> &'a [[T; 3]] {
        let n = sources.len() / self.densities.source_point_size();
        &sources.as_chunks::<3>().0[..n]
    }

    /// Vector v's charges of the n sources of `sources` into `out`, as complex values.
    fn charges_of(&self, sources: &[T], n: usize, v: usize, out: &mut [Complex<T>]) {
        let complex = self.densities.columns() == 2 * self.densities.vectors();
        for (j, q) in out[..n].iter_mut().enumerate() {
            let re = sources[self.densities.charge_index(n, j, v, 0)];
            let im = if complex {
                sources[self.densities.charge_index(n, j, v, 1)]
            } else {
                T::zero()
            };
            *q = Complex::new(re, im);
        }
    }

    fn p2m(&self, level: usize, sources: &[T], multipole: &mut [T], s: &mut Scratch<T>) {
        let n = self.source_points(sources).len();
        if n == 0 {
            return;
        }
        let p = self.p(level);
        self.densities.p2m(
            &self.k,
            self.radii[level],
            p,
            sources,
            n,
            &mut s.leaf,
            multipole,
        );
    }

    fn l2p(&self, level: usize, local: &[T], targets: &[T], output: &mut [T], s: &mut Scratch<T>) {
        if targets.is_empty() {
            return;
        }
        let p = self.p(level);
        self.densities.l2p(
            &self.k,
            self.radii[level],
            p,
            local,
            targets,
            &mut s.leaf,
            output,
            self.gradients,
        );
    }

    fn m2m(&self, level: usize, o: usize, child: &[T], parent: &mut [T], s: &mut Scratch<T>) {
        let planes = self.densities.multipole_planes();
        let m = self.densities.vectors();
        self.tables
            .m2m(level, o, child, parent, planes, m, &mut s.tables);
    }

    fn l2l(&self, level: usize, o: usize, parent: &[T], child: &mut [T], s: &mut Scratch<T>) {
        let planes = self.densities.local_planes();
        let m = self.densities.vectors();
        self.tables
            .l2l(level, o, parent, child, planes, m, &mut s.tables);
    }

    fn m2l(&self, level: usize, d: usize, multipole: &[T], local: &mut [T], s: &mut Scratch<T>) {
        let planes = self.densities.multipole_planes();
        let m = self.densities.vectors();
        self.tables
            .m2l(level, d, multipole, planes, local, m, &mut s.tables);
    }

    /// P2L into a box of `level` from a leaf at `frame` = (ĉ(t|s), r̂(t|s)), in the leaf's
    /// units.
    fn p2l(
        &self,
        level: usize,
        frame: &Frame<T>,
        sources: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        let points = self.source_points(sources);
        let n = points.len();
        if n == 0 {
            return;
        }
        let p = self.p(level);
        // r̂(t|s) = r_t / r_s = 2^(l_s − l_t).
        let source_level = Self::shifted(level, Self::levels_apart(frame));
        let units = Level::new(self.k_hat(source_level), frame.radius);
        let size = self.densities.local_size(p) / self.densities.vectors();
        for (v, local) in local.chunks_exact_mut(size).enumerate() {
            self.charges_of(sources, n, v, &mut s.charges);
            // In the leaf's units the reference forms r_s φ̃; the box holds r_t φ̃:
            // the charges times r̂(t|s), exactly (a power of two).
            for q in &mut s.charges[..n] {
                *q = *q * frame.radius;
            }
            leaf::p2l(
                p,
                &units,
                frame,
                points,
                &s.charges[..n],
                &mut s.workspace,
                local,
            );
        }
    }

    /// M2P at the targets of a leaf of `level` from a box at `frame` = (ĉ(s|t), r̂(s|t)), in
    /// the leaf's units.
    fn m2p(
        &self,
        level: usize,
        frame: &Frame<T>,
        multipole: &[T],
        targets: &[T],
        output: &mut [T],
        s: &mut Scratch<T>,
    ) {
        let targets = targets.as_chunks::<3>().0;
        if targets.is_empty() {
            return;
        }
        // r̂(s|t) = r_s / r_t = 2^(l_t − l_s).
        let source_level = Self::shifted(level, -Self::levels_apart(frame));
        let p = self.p(source_level);
        let units = Level::new(self.k_hat(level), frame.radius);
        let planes = self.densities.multipole_planes();
        let len = Layout::new(p).len();
        let Scratch {
            workspace, widened, ..
        } = s;
        for (v, multipole) in multipole.chunks_exact(planes * len).enumerate() {
            let multipole = if planes == 2 {
                multipole
            } else {
                let w = &mut widened[..2 * len];
                w[..len].copy_from_slice(multipole);
                w[len..].fill(T::zero());
                &*w
            };
            self.each_target(v, targets.len(), output, |i, phi, grad| {
                leaf::m2p(
                    p,
                    &units,
                    frame,
                    multipole,
                    &targets[i..=i],
                    workspace,
                    phi,
                    grad,
                );
            });
        }
    }

    /// P2P at the targets of a leaf of `level` from its own sources (`frame` `None`) or
    /// another leaf's at `frame` = (ĉ(s|t), r̂(s|t)).
    fn p2p(
        &self,
        level: usize,
        frame: Option<&Frame<T>>,
        sources: &[T],
        targets: &[T],
        output: &mut [T],
        s: &mut Scratch<T>,
    ) {
        let points = self.source_points(sources);
        let targets = targets.as_chunks::<3>().0;
        let n = points.len();
        if n == 0 || targets.is_empty() {
            return;
        }
        let k = self.k_hat(level);
        let Scratch {
            mapped, charges, ..
        } = s;
        let points = match frame {
            None => points,
            Some(frame) => {
                assert!(
                    n <= mapped.len(),
                    "a leaf has {n} source points, more than max_leaf_points = {}",
                    mapped.len()
                );
                let mapped = &mut mapped[..n];
                for (y, u) in mapped.iter_mut().zip(points) {
                    // r̂ u is exact (r̂ is a power of two); the sum rounds once (§3.13).
                    *y = core::array::from_fn(|i| frame.centre[i] + frame.radius * u[i]);
                }
                &*mapped
            }
        };
        for v in 0..self.densities.vectors() {
            self.charges_of(sources, n, v, charges);
            let charges = &charges[..n];
            self.each_target(v, targets.len(), output, |i, phi, grad| {
                p2p::p2p(k, points, charges, &targets[i..=i], phi, grad);
            });
        }
    }

    /// For each of the `nt` targets of vector v: loads its two output columns as complex
    /// values, lets `add` accumulate into them (a potential, and a gradient with
    /// gradients), and stores them back.
    fn each_target(
        &self,
        v: usize,
        nt: usize,
        output: &mut [T],
        mut add: impl FnMut(usize, &mut [Complex<T>], Option<&mut [[Complex<T>; 3]]>),
    ) {
        let d = &self.densities;
        let (re, im) = (d.output_column(v, 0), d.output_column(v, 1));
        for i in 0..nt {
            let (pr, pi) = (d.potential_index(nt, re, i), d.potential_index(nt, im, i));
            let mut phi = [Complex::new(output[pr], output[pi])];
            if self.gradients {
                let mut g = [core::array::from_fn(|axis| {
                    Complex::new(
                        output[d.gradient_index(nt, re, i, axis)],
                        output[d.gradient_index(nt, im, i, axis)],
                    )
                })];
                add(i, &mut phi, Some(&mut g));
                for (axis, c) in g[0].iter().enumerate() {
                    output[d.gradient_index(nt, re, i, axis)] = c.re;
                    output[d.gradient_index(nt, im, i, axis)] = c.im;
                }
            } else {
                add(i, &mut phi, None);
            }
            output[pr] = phi[0].re;
            output[pi] = phi[0].im;
        }
    }
}

/// The bodies above on the shared drivers.
impl<T: SimdScalar> HostKernel<T> for Kernel<T> {
    type Scratch = Scratch<T>;

    #[inline]
    fn vectors(&self) -> usize {
        self.densities.vectors()
    }

    #[inline]
    fn p2m(&self, level: usize, sources: &[T], multipole: &mut [T], s: &mut Scratch<T>) {
        Kernel::p2m(self, level, sources, multipole, s);
    }

    #[inline]
    fn m2m(&self, level: usize, o: usize, child: &[T], parent: &mut [T], s: &mut Scratch<T>) {
        Kernel::m2m(self, level, o, child, parent, s);
    }

    #[inline]
    fn m2l(
        &self,
        level: usize,
        d: usize,
        _child: usize,
        multipole: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        Kernel::m2l(self, level, d, multipole, local, s);
    }

    #[inline]
    fn l2l(&self, level: usize, o: usize, parent: &[T], child: &mut [T], s: &mut Scratch<T>) {
        Kernel::l2l(self, level, o, parent, child, s);
    }

    #[inline]
    fn p2l(
        &self,
        level: usize,
        frame: &Frame<T>,
        sources: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        Kernel::p2l(self, level, frame, sources, local, s);
    }

    #[inline]
    fn l2p(&self, level: usize, local: &[T], targets: &[T], output: &mut [T], s: &mut Scratch<T>) {
        Kernel::l2p(self, level, local, targets, output, s);
    }

    #[inline]
    fn m2p(
        &self,
        level: usize,
        frame: &Frame<T>,
        multipole: &[T],
        targets: &[T],
        output: &mut [T],
        s: &mut Scratch<T>,
    ) {
        Kernel::m2p(self, level, frame, multipole, targets, output, s);
    }

    #[inline]
    fn p2p(
        &self,
        level: usize,
        frame: Option<&Frame<T>>,
        sources: &[T],
        targets: &[T],
        output: &mut [T],
        s: &mut Scratch<T>,
    ) {
        Kernel::p2p(self, level, frame, sources, targets, output, s);
    }

    #[inline]
    fn batched_m2l(&self, _level: usize) -> Option<(&MatrixSet<T>, &MatrixKernel<T>)> {
        None
    }

    #[inline]
    fn columns(scratch: &mut Scratch<T>) -> &mut Columns<T> {
        &mut scratch.columns
    }
}

impl<T: SimdScalar + Equivalence + Default> FmmSizes for HelmholtzOperator<T> {
    type Value = T;

    /// [`Densities::multipole_size`] at p_l: m (p_l + 1)² or 2 m (p_l + 1)².
    fn multipole_size(&self, level: usize) -> usize {
        self.kernel.densities.multipole_size(self.degree(level))
    }

    /// [`Densities::local_size`] at p_l: 2 m (p_l + 1)².
    fn local_size(&self, level: usize) -> usize {
        self.kernel.densities.local_size(self.degree(level))
    }

    /// [`Densities::source_point_size`]: 3 + c m.
    fn source_point_size(&self) -> usize {
        self.kernel.densities.source_point_size()
    }

    /// 3, the leaf-scaled position.
    fn target_input_point_size(&self) -> usize {
        3
    }

    /// [`Densities::target_output_point_size`]: 2 m, or 8 m with gradients.
    fn target_output_point_size(&self) -> usize {
        self.kernel
            .densities
            .target_output_point_size(self.kernel.gradients)
    }
}

/// Target by target through the shared drivers (module documentation, "Execution").
impl<T: SimdScalar + Equivalence + Default> FmmOperator for HelmholtzOperator<T> {
    fn p2m(&mut self, batch: P2m<'_, T>) {
        driver::p2m(&self.kernel, &mut self.execution, batch);
    }

    fn m2m(&mut self, batch: M2m<'_, T>) {
        driver::m2m(&self.kernel, &mut self.execution, batch);
    }

    fn m2l(&mut self, batch: M2l<'_, T>) {
        driver::m2l(&self.kernel, &mut self.execution, batch);
    }

    fn p2l(&mut self, batch: P2l<'_, T>) {
        driver::p2l(&self.kernel, &mut self.execution, batch);
    }

    fn l2l(&mut self, batch: L2l<'_, T>) {
        driver::l2l(&self.kernel, &mut self.execution, batch);
    }

    fn l2p(&mut self, batch: L2p<'_, T>) {
        driver::l2p(&self.kernel, &mut self.execution, batch);
    }

    fn m2p(&mut self, batch: M2p<'_, T>) {
        driver::m2p(&self.kernel, &mut self.execution, batch);
    }

    fn p2p(&mut self, batch: P2p<'_, T>) {
        driver::p2p(&self.kernel, &mut self.execution, batch);
    }
}

/// The same bodies pair by pair, for `nd_fmm_plan::operator::PerPair` and the tests: the
/// level from the keys, the octant or offset index as given (debug builds check it
/// against the keys), the frames from the keys, the self pair of P2P by
/// `source == target`. Serial, on the first scratch set.
impl<T: SimdScalar + Equivalence + Default> PairOperator for HelmholtzOperator<T> {
    fn p2m(&mut self, leaf: MortonKey, sources: &[T], multipole: &mut [T]) {
        let level = morton::level(leaf) as usize;
        self.kernel
            .p2m(level, sources, multipole, self.execution.first());
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
        let level = morton::level(parent) as usize;
        self.kernel.m2m(
            level,
            octant,
            child_multipole,
            parent_multipole,
            self.execution.first(),
        );
    }

    fn m2l(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        offset: usize,
        source_multipole: &[T],
        target_local: &mut [T],
    ) {
        debug_assert_eq!(
            offset_index(source, target),
            offset,
            "the offset index of a V-list pair disagrees with its keys"
        );
        let level = morton::level(target) as usize;
        self.kernel.m2l(
            level,
            offset,
            source_multipole,
            target_local,
            self.execution.first(),
        );
    }

    fn p2l(&mut self, source: MortonKey, target: MortonKey, sources: &[T], local: &mut [T]) {
        let frame = relative_frame(target, source);
        let level = morton::level(target) as usize;
        self.kernel
            .p2l(level, &frame, sources, local, self.execution.first());
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
        let level = morton::level(child) as usize;
        self.kernel.l2l(
            level,
            octant,
            parent_local,
            child_local,
            self.execution.first(),
        );
    }

    fn l2p(&mut self, leaf: MortonKey, local: &[T], target_input: &[T], output: &mut [T]) {
        let level = morton::level(leaf) as usize;
        self.kernel
            .l2p(level, local, target_input, output, self.execution.first());
    }

    fn m2p(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        source_multipole: &[T],
        target_input: &[T],
        target_output: &mut [T],
    ) {
        let frame = relative_frame(source, target);
        let level = morton::level(target) as usize;
        self.kernel.m2p(
            level,
            &frame,
            source_multipole,
            target_input,
            target_output,
            self.execution.first(),
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
        let frame = (source != target).then(|| relative_frame(source, target));
        let level = morton::level(target) as usize;
        self.kernel.p2p(
            level,
            frame.as_ref(),
            sources,
            target_input,
            target_output,
            self.execution.first(),
        );
    }
}

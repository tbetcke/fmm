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
//! | M2L under [`HelmholtzM2l::Dense`] | `nd_fmm_tables::helmholtz::dense::DenseM2l` on the levels whose tables fit the budget, RCR on the others |
//! | M2L under [`HelmholtzM2l::Classes`] | `nd_fmm_tables::helmholtz::classes::ClassM2l` on the levels whose tables fit the budget, RCR on the others |
//! | M2L under [`HelmholtzM2l::Batched`] | `nd_fmm_tables::helmholtz::batched::BatchedM2l` by `nd_fmm_simd::MatrixKernel`, block by block, on the levels whose tables fit the budget, RCR on the others |
//! | P2L | `nd_fmm_ref::helmholtz::leaf::p2l` in the source leaf's units, at (ĉ(t\|s), r̂(t\|s)) |
//! | M2P | `nd_fmm_ref::helmholtz::leaf::m2p` in the target leaf's units, at (ĉ(s\|t), r̂(s\|t)) |
//! | P2P | `nd_fmm_simd::P2pKernel::evaluate_helmholtz` on the ISA of the [`P2pChoice`], or `nd_fmm_ref::helmholtz::p2p::p2p` for [`P2pChoice::Reference`], in the target leaf's units, the sources mapped as ĉ(s\|t) + r̂(s\|t) u |
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
//! P2P, M2P and L2P accumulate into the target output chunk: M2P, and P2P with the
//! reference, load each target's two output columns of a vector as complex values, let the
//! reference operator add its terms to them, and store them back; L2P is
//! [`Densities::l2p`]'s. Every operator gives each charge vector exactly the one-vector
//! operator, and adds (+=); none applies 1/(4π) or the leaf radius (the output pass does,
//! CONVENTIONS §3.13).
//!
//! # P2P kernel
//!
//! P2P runs the kernel of a [`P2pChoice`] (Helmholtz CH3.4c), set by
//! [`with_p2p`](HelmholtzOperator::with_p2p) and by default [`P2pChoice::Auto`], as
//! Laplace's operator does: `nd_fmm_simd::P2pKernel::evaluate_helmholtz` (Helmholtz CH3.4b)
//! on the default ISA of the machine (`Isa::detect`) or on a named one, and
//! [`P2pChoice::Reference`] keeps `nd_fmm_ref::helmholtz::p2p::p2p`, the trusted slower
//! path. The kernel takes one vector of all the leaf's targets per call: the charges of
//! vector v, Re q and Im q of a source together (the chunk's own strengths for one vector,
//! gathered into scratch for several), and the output chunk's planes as they lie, since the
//! potentials of output columns 2v and 2v + 1 are 2 n_t adjacent values and their gradients
//! 2 n_t adjacent triples, the kernel's layout. Each ISA is within the kernel's accuracy
//! contract of the reference (docs/design/simd-p2p.md §4.12), and `Isa::Scalar` is the
//! reference bit for bit. For one machine, ISA and build the output is deterministic, the
//! same at every thread count and on P ranks (constraint 2);
//! [`p2p_kernel`](HelmholtzOperator::p2p_kernel) reports the kernel that runs.
//!
//! # M2L
//!
//! M2L runs by the level's RCR tables ([`HelmholtzM2l::Rotation`], the default), or under
//! [`HelmholtzM2l::Dense`] (Helmholtz CH3.2a; proposal §5.3) by one complex matrix per
//! offset on each level whose tables fit a byte budget, chosen from the finest level up
//! (`DenseM2l::select`, a function of the degrees and the budget only, so every rank
//! chooses alike), and by RCR on the others: [`with_m2l`](HelmholtzOperator::with_m2l).
//! The dense tables are Rotation's coefficients applied to unit multipoles in f64 and
//! rounded to T, so Dense agrees with Rotation to rounding, not bit for bit. Dense combines
//! the four real products of a complex M2L pair by pair, in the target's row order as every
//! body, so its output is the same at every thread count and on P ranks the one-rank
//! output; it adds no per-target epilogue and no driver change.
//!
//! Under [`HelmholtzM2l::Classes`] (Helmholtz CH3.2b; proposal §5.3) a level whose tables
//! fit the same budget (`ClassM2l::select`, from the finest level up) holds the 16 class
//! matrices of CONVENTIONS §3.12 in Dense's layout, bit for bit Dense's matrices of the
//! representatives, and the cube-group transforms of the orthonormal coefficients: each pair
//! takes its multipole through O(Pᵀ), the class matrix by Dense's chain and O(P) into the
//! local, signed permutations for the elements that fix the z-axis and dense per-degree
//! blocks for the others. So Classes agrees with Dense, and with Rotation, to rounding, and
//! is deterministic and rank-invariant as Dense is.
//!
//! Under [`HelmholtzM2l::Batched`] (Helmholtz CH3.2c; proposal §5.3) a level whose tables
//! fit the same budget (Dense's `select`) holds Dense's matrices in split planes, A_r and
//! A_i each n × n (`BatchedM2l`), and its M2L level call runs as Laplace's batched M2L
//! does, with its own block (not `driver::m2l_block`, whose layout is Laplace's; the user,
//! 2026-11-02): blocks of `M2L_BLOCK_TARGETS` consecutive targets, and in each the offset
//! indices d in order, the pairs of the grouped view's `batch(d)` whose target lies in the
//! block, m columns a pair, through `nd_fmm_simd::MatrixKernel::accumulate_columns` (on the
//! ISA of the P2P kernel, the scalar path for [`P2pChoice::Reference`]), each table read
//! once per block. Per offset, on each column, in the default [`MulAdd::Separate`] mode
//! (`MatrixSet::apply`'s steps, product and sum rounded apart):
//!
//! - real multipoles (real k, real charges): L_r += A_r M, then L_i += A_i M, Dense's chain
//!   bit for bit;
//! - complex multipoles: L_r += A_r M_r; L_r −= A_i M_i (L_r negated, A_i M_i added and L_r
//!   negated back, exact); L_i += A_i M_r; L_i += A_r M_i. Each column sums the real plane's
//!   products before the imaginary plane's where Dense alternates them per k, so Batched is
//!   Dense to rounding here (CH3.2a's sign-off 2), with no per-target epilogue.
//!
//! In the opt-in [`MulAdd::Fused`] mode ([`with_m2l_mul_add`](HelmholtzOperator::with_m2l_mul_add))
//! each step is one fma: Dense to rounding also with real multipoles (per element within
//! about n u of the separate form, relative to the sum of its terms' magnitudes), and about
//! 1.8 times the separate form's rate on the M6, where the kernel is bound by its arithmetic
//! (fmm-validate/results/helmholtz-ch3-2c.md).
//!
//! An element depends on its own column only, and a target meets the offsets in the order
//! of its row, so the level call is the pair body (the same calls with one column per
//! vector) bit for bit: the same at every thread count, block split and ISA, and on P ranks
//! the one-rank output. Levels without tables run target by target by RCR.
//!
//! # Execution
//!
//! The level calls run on the shared drivers of `nd-fmm-exec` (Helmholtz CH2.0, CH3.5),
//! target by target in the plan's order, serially or on the operator's pool
//! ([`HelmholtzOperator::with_pool`]), each thread with its own scratch; the output is the
//! same at every thread count. The shared batched M2L does not apply (`driver::m2l_block`
//! assumes Laplace's layout); under [`HelmholtzM2l::Batched`] the M2L level call of a level
//! with tables runs the operator's own block instead ("M2L"). [`PairOperator`] runs the same
//! bodies pair by pair.
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
use nd_fmm_simd::{
    HelmholtzCharges, Isa, IsaUnavailable, MatrixKernel, MulAdd, P2pKernel, SimdScalar,
};
use nd_fmm_tables::MatrixSet;
use nd_fmm_tables::helmholtz::batched::BatchedM2l;
use nd_fmm_tables::helmholtz::classes::{ClassM2l, ClassScratch};
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzScratch, HelmholtzTables};
use nd_octree::{MortonKey, morton};
use rayon::ThreadPool;
use thiserror::Error;

use super::batched::Batched;
use super::{Densities, LeafScratch, Wavenumber, WavenumberKind};
use crate::driver::{self, Columns, Execution, HostKernel, debug_check_octant, offset_index};
use crate::geometry::{Domain, radius, relative_frame};
use crate::operator::{M2L_BLOCK_TARGETS, P2pChoice};

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

/// The M2L of the Helmholtz operator's levels (module documentation, "M2L"; proposal §5.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HelmholtzM2l {
    /// RCR on every level (the default): rotation, coaxial translation, rotation back.
    #[default]
    Rotation,
    /// One complex matrix per offset on the levels whose tables fit the dense budget, from
    /// the finest level up; RCR on the others.
    Dense,
    /// The 16 class matrices per level and the cube-group transforms, on the levels whose
    /// tables fit the dense budget, from the finest level up; RCR on the others.
    Classes,
    /// Dense's matrices in split planes, applied block by block by the column kernel, on the
    /// levels whose tables fit the dense budget, from the finest level up; RCR on the others.
    Batched,
}

/// The default byte budget of the dense or class M2L tables of one rank, 1 GiB.
pub const DEFAULT_DENSE_BUDGET: usize = 1 << 30;

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
    /// The M2L setting.
    m2l: HelmholtzM2l,
    /// The dense M2L tables of the levels that have them, under [`HelmholtzM2l::Dense`].
    dense: Option<DenseM2l<T>>,
    /// The class M2L tables of the levels that have them, under [`HelmholtzM2l::Classes`].
    classes: Option<ClassM2l<T>>,
    /// The batched M2L tables of the levels that have them, under [`HelmholtzM2l::Batched`].
    batched: Option<BatchedM2l<T>>,
    /// The batched M2L's column kernel, on the P2P kernel's ISA (scalar for the reference).
    matrix: MatrixKernel<T>,
    /// The P2P kernel of `nd-fmm-simd`, or `None` for the reference.
    p2p: Option<P2pKernel<T>>,
}

/// One thread's scratch.
struct Scratch<T: SimdScalar> {
    leaf: LeafScratch<T>,
    workspace: Workspace<T>,
    tables: HelmholtzScratch<T>,
    /// The class M2L's rotated multipole and translated local.
    classes: ClassScratch<T>,
    /// Another leaf's sources in the target's units (P2P).
    mapped: Vec<[T; 3]>,
    /// One vector's charges.
    charges: Vec<Complex<T>>,
    /// One vector's charges for the P2P kernel, Re q and Im q per source.
    strengths: Vec<T>,
    /// One multipole of two planes (M2P of a real multipole).
    widened: Vec<T>,
    /// The batched M2L's packing buffer and column lists (sized for the batched levels).
    columns: Columns<T>,
}

impl<T: SimdScalar> HelmholtzOperator<T> {
    /// The operator of an octree on `domain` with `degrees.len()` levels of degrees p_l,
    /// for `k` and the path `densities` (its kind must be k's), with gradients in the
    /// target output if `gradients`, for leaves with at most `max_leaf_points` source
    /// points (every leaf P2P reads, local and ghost: on several ranks a global maximum).
    /// Builds the tables of every level with the default block budget
    /// ([`DEFAULT_BLOCK_BUDGET`]) and one scratch set; runs serially until
    /// [`with_pool`](Self::with_pool), with the P2P kernel of [`P2pChoice::Auto`] until
    /// [`with_p2p`](Self::with_p2p).
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
            m2l: HelmholtzM2l::Rotation,
            dense: None,
            classes: None,
            batched: None,
            matrix: MatrixKernel::detect(MulAdd::Separate),
            p2p: Some(P2pKernel::detect()),
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

    /// Runs M2L by `choice` (module documentation, "M2L"): under [`HelmholtzM2l::Dense`]
    /// builds the dense tables of the levels that fit in `dense_budget` bytes from the
    /// finest level up (`DenseM2l::select`; [`DEFAULT_DENSE_BUDGET`] is a default), under
    /// [`HelmholtzM2l::Classes`] the class tables within the same budget
    /// (`ClassM2l::select`, the transforms included), under [`HelmholtzM2l::Batched`]
    /// Dense's tables in split planes (`BatchedM2l::select`, Dense's), RCR on the others; under
    /// [`HelmholtzM2l::Rotation`] ignores the budget. The tables are built in f64, serially,
    /// O(p⁵) per offset and level (16 offsets a level for Classes, 316 for Dense).
    pub fn with_m2l(mut self, choice: HelmholtzM2l, dense_budget: usize) -> Self {
        let kernel = &mut self.kernel;
        kernel.m2l = choice;
        let degrees: Vec<usize> = (0..kernel.tables.nlevels())
            .map(|l| kernel.tables.degree(l))
            .collect();
        let k_root = kernel.tables.k_hat(0);
        kernel.dense = (choice == HelmholtzM2l::Dense).then(|| {
            let levels = DenseM2l::<T>::select(&degrees, dense_budget);
            DenseM2l::build(k_root, &degrees, &levels)
        });
        kernel.classes = (choice == HelmholtzM2l::Classes).then(|| {
            let levels = ClassM2l::<T>::select(&degrees, dense_budget);
            ClassM2l::build(k_root, &degrees, &levels)
        });
        kernel.batched = (choice == HelmholtzM2l::Batched).then(|| {
            let levels = BatchedM2l::<T>::select(&degrees, dense_budget);
            BatchedM2l::build(k_root, &degrees, &levels)
        });
        // The class scratch and the batched columns are sized by their tables' largest
        // degree.
        for scratch in &mut self.execution.scratch {
            *scratch.get_mut().expect("a scratch set is never poisoned") = Scratch::new(kernel);
        }
        self
    }

    /// The M2L setting ([`with_m2l`](Self::with_m2l)).
    pub fn m2l_choice(&self) -> HelmholtzM2l {
        self.kernel.m2l
    }

    /// The M2L that runs on `level`: [`HelmholtzM2l::Dense`], [`HelmholtzM2l::Classes`] or
    /// [`HelmholtzM2l::Batched`] where the level has dense, class or batched tables,
    /// [`HelmholtzM2l::Rotation`] everywhere else (levels 0 and 1 have no M2L).
    pub fn m2l_on(&self, level: usize) -> HelmholtzM2l {
        let kernel = &self.kernel;
        match (&kernel.dense, &kernel.classes, &kernel.batched) {
            (Some(dense), _, _) if dense.is_dense(level) => HelmholtzM2l::Dense,
            (_, Some(classes), _) if classes.is_class(level) => HelmholtzM2l::Classes,
            (_, _, Some(batched)) if batched.is_batched(level) => HelmholtzM2l::Batched,
            _ => HelmholtzM2l::Rotation,
        }
    }

    /// The bytes of the dense M2L tables (0 unless [`HelmholtzM2l::Dense`]).
    pub fn dense_bytes(&self) -> usize {
        self.kernel.dense.as_ref().map_or(0, DenseM2l::bytes)
    }

    /// The bytes of the class M2L tables, the transforms included (0 unless
    /// [`HelmholtzM2l::Classes`]).
    pub fn class_bytes(&self) -> usize {
        self.kernel.classes.as_ref().map_or(0, ClassM2l::bytes)
    }

    /// The bytes of the batched M2L tables (0 unless [`HelmholtzM2l::Batched`]).
    pub fn batched_bytes(&self) -> usize {
        self.kernel.batched.as_ref().map_or(0, BatchedM2l::bytes)
    }

    /// Runs P2P with the kernel of `choice` (module documentation, "P2P kernel"), and the
    /// batched M2L's column kernel on its ISA (the scalar path for
    /// [`P2pChoice::Reference`]).
    ///
    /// # Errors
    ///
    /// [`IsaUnavailable`] for `P2pChoice::Isa(isa)` if this machine cannot run `isa`.
    pub fn with_p2p(mut self, choice: P2pChoice) -> Result<Self, IsaUnavailable> {
        self.kernel.p2p = match choice {
            P2pChoice::Auto => Some(P2pKernel::detect()),
            P2pChoice::Reference => None,
            P2pChoice::Isa(isa) => Some(P2pKernel::new(isa)?),
        };
        let isa = self.kernel.p2p.as_ref().map_or(Isa::Scalar, P2pKernel::isa);
        self.kernel.matrix = MatrixKernel::new(isa, self.kernel.matrix.mul_add())?;
        Ok(self)
    }

    /// Runs the batched M2L's column kernel in the mode `mul_add` (module documentation,
    /// "M2L"; [`MulAdd::Separate`] by default): [`MulAdd::Fused`] forms each step as one
    /// fma, within n u of the separate form per element, so the output is Dense's to
    /// rounding also with real multipoles; deterministic, the same at every thread count,
    /// ISA and rank count. Only [`HelmholtzM2l::Batched`] reads it.
    pub fn with_m2l_mul_add(mut self, mul_add: MulAdd) -> Self {
        self.kernel.matrix = MatrixKernel::new(self.kernel.matrix.isa(), mul_add)
            .expect("the kernel's own ISA is available");
        self
    }

    /// The batched M2L's multiply–add mode ([`with_m2l_mul_add`](Self::with_m2l_mul_add)).
    pub fn m2l_mul_add(&self) -> MulAdd {
        self.kernel.matrix.mul_add()
    }

    /// The P2P kernel as it runs: [`P2pChoice::Reference`] or [`P2pChoice::Isa`] with the
    /// ISA of the kernel, never [`P2pChoice::Auto`].
    pub fn p2p_kernel(&self) -> P2pChoice {
        match &self.kernel.p2p {
            None => P2pChoice::Reference,
            Some(kernel) => P2pChoice::Isa(kernel.isa()),
        }
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
        // The batched M2L's largest n, and m columns per pair of a block.
        let n_batched = kernel.batched.as_ref().map_or(0, |b| {
            (0..b.nlevels())
                .filter(|&l| b.is_batched(l))
                .map(|l| Layout::new(b.degree(l)).len())
                .max()
                .unwrap_or(0)
        });
        let pairs = match n_batched {
            0 => 0,
            _ => kernel.densities.vectors() * M2L_BLOCK_TARGETS,
        };
        Self {
            leaf: LeafScratch::new(p_max),
            workspace: Workspace::new(p_max),
            tables: kernel.tables.scratch(),
            classes: kernel
                .classes
                .as_ref()
                .map_or_else(|| ClassScratch::new(0), ClassM2l::scratch),
            mapped: vec![[T::zero(); 3]; kernel.max_leaf_points],
            charges: vec![Complex::new(T::zero(), T::zero()); kernel.max_leaf_points],
            strengths: vec![T::zero(); 2 * kernel.max_leaf_points],
            widened: vec![T::zero(); 2 * Layout::new(p_max).len()],
            columns: Columns {
                pack: vec![T::zero(); MatrixKernel::<T>::pack_len(n_batched)],
                sources: Vec::with_capacity(pairs),
                targets: Vec::with_capacity(pairs),
            },
        }
    }
}

impl<T: SimdScalar> Kernel<T> {
    fn p(&self, level: usize) -> usize {
        self.tables.degree(level)
    }

    /// The batched M2L of `level`, if it has tables.
    fn batched(&self, level: usize) -> Option<Batched<'_, T>> {
        let tables = self.batched.as_ref().filter(|b| b.is_batched(level))?;
        Some(Batched {
            tables,
            matrix: &self.matrix,
            planes: self.densities.multipole_planes(),
            vectors: self.densities.vectors(),
        })
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
        assert!(
            n <= out.len(),
            "a leaf has {n} source points, more than max_leaf_points = {}",
            out.len()
        );
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
        match (&self.dense, &self.classes, self.batched(level)) {
            (Some(dense), _, _) if dense.is_dense(level) => {
                dense.m2l(level, d, multipole, planes, local, m);
            }
            (_, Some(classes), _) if classes.is_class(level) => {
                classes.m2l(level, d, multipole, planes, local, m, &mut s.classes);
            }
            (_, _, Some(batched)) => batched.pair(level, d, multipole, local, &mut s.columns),
            _ => self
                .tables
                .m2l(level, d, multipole, planes, local, m, &mut s.tables),
        }
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
            mapped,
            charges,
            strengths,
            ..
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
        let Some(kernel) = &self.p2p else {
            for v in 0..self.densities.vectors() {
                self.charges_of(sources, n, v, charges);
                let charges = &charges[..n];
                self.each_target(v, targets.len(), output, |i, phi, grad| {
                    p2p::p2p(k, points, charges, &targets[i..=i], phi, grad);
                });
            }
            return;
        };
        let d = &self.densities;
        let (m, nt) = (d.vectors(), targets.len());
        // The real columns of a vector's charges: 1 (real) or 2 (complex).
        let c = d.columns() / m;
        // The potentials of every output column, then their gradients (CONVENTIONS §3.13).
        let (potentials, gradients) = output.split_at_mut(d.output_columns() * nt);
        for v in 0..m {
            let charges = if m == 1 {
                &sources[3 * n..3 * n + c * n]
            } else {
                assert!(
                    c * n <= strengths.len(),
                    "a leaf has {n} source points, more than max_leaf_points = {}",
                    strengths.len() / 2
                );
                let w = &mut strengths[..c * n];
                for (j, q) in w.chunks_exact_mut(c).enumerate() {
                    q.copy_from_slice(&sources[d.charge_index(n, j, v, 0)..][..c]);
                }
                &*w
            };
            let charges = if c == 1 {
                HelmholtzCharges::Real(charges)
            } else {
                HelmholtzCharges::Complex(charges)
            };
            let re = d.output_column(v, 0);
            let potential = &mut potentials[re * nt..(re + 2) * nt];
            let gradient = self
                .gradients
                .then(|| &mut gradients.as_chunks_mut::<3>().0[re * nt..(re + 2) * nt]);
            kernel.evaluate_helmholtz(k, points, charges, targets, potential, gradient);
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

/// Target by target through the shared drivers, but for the M2L of a level with batched
/// tables (module documentation, "Execution" and "M2L").
impl<T: SimdScalar + Equivalence + Default> FmmOperator for HelmholtzOperator<T> {
    fn p2m(&mut self, batch: P2m<'_, T>) {
        driver::p2m(&self.kernel, &mut self.execution, batch);
    }

    fn m2m(&mut self, batch: M2m<'_, T>) {
        driver::m2m(&self.kernel, &mut self.execution, batch);
    }

    /// Under [`HelmholtzM2l::Batched`] block by block on the levels with tables (module
    /// documentation, "M2L"), target by target elsewhere.
    fn m2l(&mut self, batch: M2l<'_, T>) {
        match self.kernel.batched(batch.level) {
            Some(batched) => batched.m2l(&mut self.execution, batch, |s| &mut s.columns),
            None => driver::m2l(&self.kernel, &mut self.execution, batch),
        }
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

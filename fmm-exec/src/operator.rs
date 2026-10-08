//! The Laplace operator on the level-batched interface of `nd-fmm-plan`, host path
//! (C3.1).
//!
//! [`LaplaceOperator`] implements [`FmmOperator`] for the kernel 1/|x − y|
//! (CONVENTIONS §3.1): the translations apply the tables of [`Tables`], looked up by
//! child index and offset index, the leaf operators call `nd_fmm_ref`, and P2P calls the
//! SIMD kernel of `nd-fmm-simd` or `nd_fmm_ref::p2p` ([`P2pChoice`]), all in the
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
//! | P2P leaf s → leaf t | `P2pKernel::evaluate` of `nd-fmm-simd`, or `p2p::p2p` ([`P2pChoice`]) | ŷ = ĉ(s\|t) + r̂(s\|t) u_s, u_s itself for s = t | O(n_s n_t), plus O(n_s) to map |
//!
//! (§3.11, "L2P and M2P", and §3.13, "Operators in scaled coordinates", derive each
//! line.) The gradient, when wanted, roughly quadruples the cost of L2P, M2P and P2P.
//! Every operator adds into its output; a leaf with no points is a no-op.
//!
//! # Batched execution and order
//!
//! Each level call runs target by target through the plan's target-centric view: box t
//! of the level for P2M, M2M, M2L, P2L and L2L (row t lines up with chunk t of the
//! output level), row r, local leaf `leaves.start + r`, for L2P, M2P and P2P. Each
//! target's contributions are added in the order of its row, as
//! the plan's accumulation rule requires (`nd_fmm_plan::operator`): children by octant
//! (M2M), sources by offset index (M2L), leaves by leaf index (P2L, P2P), boxes by box
//! index (M2P), and the single parent (L2L) or leaf (P2M, L2P). With the evaluator's
//! fixed call order this fixes every floating-point sum. The tables are chosen by the
//! entry's octant or offset index; debug builds check that it agrees with the keys.
//!
//! The body of one target is a function of that target's output slice, the call's
//! shared inputs and one scratch set (`Kernels::*_target`), and nothing else.
//!
//! # P2P kernel
//!
//! P2P runs the kernel of [`P2pChoice`], set by [`with_p2p`](LaplaceOperator::with_p2p)
//! and by default [`P2pChoice::Auto`]: `nd_fmm_simd::P2pKernel` on the widest ISA of the
//! machine (NEON on aarch64, AVX2 + FMA on x86_64 if the CPU has it, else its scalar
//! path). [`P2pChoice::Reference`] keeps `nd_fmm_ref::p2p::p2p`, the P2P of Phase 3, as
//! the trusted slower path; [`P2pChoice::Isa`] picks an ISA, for tests and benchmarks.
//! [`p2p_kernel`](LaplaceOperator::p2p_kernel) reports the kernel that runs.
//!
//! Only the kernel changes: the mapping of the sources of another leaf into the
//! scratch buffer, the order of the near list and one kernel call per (target leaf,
//! source leaf) pair stay those of Phase 3. Every kernel adds each target's sources in
//! input order into the output (chunk invariance), so the per-pair and batched paths
//! agree bit for bit for every choice. One call per target leaf over its whole gathered
//! near field gives the same bits too, but was measured in Phase 3S T6 at 1–6% faster on
//! the leaf stage (NEON, the uniform level-4 tree of C3.2 at p = 3 and 8, f32 and f64,
//! with and without gradients; above 5% only for f32 with gradients at p = 3, in one of
//! two runs), short of the 5% that would justify its buffer and copy, so calls stay per
//! source leaf.
//!
//! Reproducibility (docs/design/simd-p2p.md §5.5): for one machine, ISA and build the
//! output is bit-identical from run to run, for every thread count and between the
//! per-pair and batched paths. Between ISAs, and between a kernel and the reference, it
//! differs in the last bits, within the accuracy of [`P2pChoice`]. The scalar ISA
//! equals the reference bit for bit wherever r² ≠ 0, and every aarch64 CPU gives the
//! same NEON results; on AVX2 the estimate `vrsqrtps` is not architecturally defined,
//! so Intel and AMD CPUs may differ in the last bits. Without gradients the vector
//! kernels add each potential term with one fma, with gradients with a product and a
//! sum, so their potentials with and without gradients differ in the last bits; the
//! reference's do not.
//!
//! # Threads (C3.5)
//!
//! Without a pool (the default, [`new`](LaplaceOperator::new)), every level call runs
//! its targets serially, in index order, on the calling thread. With a pool
//! ([`with_pool`](LaplaceOperator::with_pool)), every level call runs inside
//! `ThreadPool::install` and hands each target to one pool thread:
//!
//! - box calls (P2M, M2M, M2L, P2L, L2L) split the output level with
//!   `par_chunks_mut(size)`, one chunk per box;
//! - leaf calls (L2P, M2P, P2P) split the target output into one mutable slice per
//!   leaf by its CSR offsets, with `LeafSliceMut::split_at_mut` halving the rows
//!   recursively under `rayon::join`. Nothing is allocated per call or per pair.
//!
//! Each thread writes only its own targets' slices, and each target runs the same body
//! as in the serial loop, so it sees the same contributions in the same order: the
//! output is bit-identical to the serial path for every number of threads. The order in
//! which *targets* are processed varies from run to run, but no target reads another
//! target's output. Load balance is rayon's work stealing, down to single targets.
//!
//! The pool's threads only compute: no level call communicates, so worker threads never
//! call MPI ([`Fmm`](crate::fmm::Fmm) keeps every collective on the calling thread). Nor
//! do they call BLAS or LAPACK: the tables are applied by hand-written loops of
//! `nd-fmm-tables`, the leaf operators by the plain Rust of `nd-fmm-ref`, and P2P by the
//! intrinsics of `nd-fmm-simd` or the plain Rust of `nd-fmm-ref`
//! ([`threading`](crate::threading)). The P2P kernel is stateless and shared by every
//! thread, one call at a time per thread, so it adds no per-thread state.
//!
//! # Scratch
//!
//! The interface passes `&mut self`, so the operator owns its scratch outright, with
//! no `RefCell`. A scratch set is an `nd_fmm_ref::Workspace` for p, the
//! [`TableScratch`] of its tables, and one buffer of `max_leaf_points` points into
//! which P2P maps the sources of another leaf. The operator holds one set, or with a
//! pool one per pool thread, each in its own `Mutex`, created when the pool is given and
//! selected by `rayon::current_thread_index()`. Only thread i ever locks set i, and only
//! around one target body, which never yields to rayon, so the lock is uncontended by
//! construction (a contended lock panics rather than waits). The serial path and the
//! per-pair methods use set 0 through `Mutex::get_mut`, without locking. Every set is
//! sized at construction; no operator allocates. The P2P kernels of `nd-fmm-simd` need
//! no scratch: they read the mapped sources and write only the target output.
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
use core::str::FromStr;
use std::sync::{Arc, Mutex, PoisonError, TryLockError};

use mpi::traits::Equivalence;
use nd_fmm_math::RealScalar;
use nd_fmm_plan::index::BoxIndex;
use nd_fmm_plan::lists::{Children, Csr, Parents, VList};
use nd_fmm_plan::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PairOperator,
};
use nd_fmm_plan::store::{LeafSlice, LeafSliceMut, LevelSlice, LevelSliceMut};
use nd_fmm_ref::{Frame, Workspace, leaf, p2p};
use nd_fmm_simd::P2pKernel;
pub use nd_fmm_simd::{Isa, IsaUnavailable, SimdScalar};
use nd_fmm_tables::geometry::m2l_offset_index;
use nd_octree::{MortonKey, morton};
use rayon::ThreadPool;
use rayon::prelude::*;
use thiserror::Error;

use crate::geometry::relative_frame;
use crate::tables::{TableScratch, Tables};

/// Values per source point: the leaf-scaled coordinates and the charge (CONVENTIONS
/// §3.13, "Source chunks").
pub const SOURCE_POINT_SIZE: usize = 4;

/// Values per target point in the target input: the leaf-scaled position (CONVENTIONS
/// §3.13, "Target input and output").
pub const TARGET_INPUT_POINT_SIZE: usize = 3;

/// The P2P kernel of a [`LaplaceOperator`] (docs/design/simd-p2p.md §6).
///
/// | Choice | Kernel |
/// | --- | --- |
/// | [`Auto`](Self::Auto), the default | `nd_fmm_simd::P2pKernel` on [`Isa::detect`], the widest ISA of this machine |
/// | [`Reference`](Self::Reference) | `nd_fmm_ref::p2p::p2p`, the scalar loop of Phase 3: the trusted slower path |
/// | [`Isa`](Self::Isa)`(isa)` | `nd_fmm_simd::P2pKernel` on `isa`, if this machine can run it |
///
/// Every choice computes the same operator (CONVENTIONS §3.13): the kernels of
/// `nd-fmm-simd` exclude a pair by r² = 0 and the reference by exact coincidence, which
/// agree on leaf-scaled data ("Fast kernels"). Their results differ in the last bits:
/// each potential term within 8 u_T of the reference's, each gradient component within
/// 16 u_T relative to |q| / r² (u_T = 2⁻²⁴, 2⁻⁵³). [`Isa::Scalar`] equals the reference
/// bit for bit wherever r² ≠ 0.
///
/// [`resolve`](Self::resolve) replaces `Auto` by the ISA it picks, and is what
/// [`LaplaceOperator::p2p_kernel`] and `Fmm::p2p_kernel` report. The text form, for
/// command lines, is `auto`, `reference` or an ISA name (`scalar`, `neon`, `avx2`):
///
/// ```
/// use nd_fmm_exec::operator::{Isa, P2pChoice};
///
/// assert_eq!(P2pChoice::default(), P2pChoice::Auto);
/// assert_eq!("neon".parse(), Ok(P2pChoice::Isa(Isa::Neon)));
/// assert_eq!(P2pChoice::Reference.to_string(), "reference");
/// assert_eq!(P2pChoice::Auto.resolve(), Ok(P2pChoice::Isa(Isa::detect())));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum P2pChoice {
    /// The kernel of `nd-fmm-simd` on the widest ISA of this machine ([`Isa::detect`]):
    /// NEON on aarch64, AVX2 + FMA on x86_64 if the CPU has it, else the scalar path.
    #[default]
    Auto,
    /// `nd_fmm_ref::p2p::p2p`, the P2P of Phase 3.
    Reference,
    /// The kernel of `nd-fmm-simd` on this ISA.
    Isa(Isa),
}

impl P2pChoice {
    /// The choice as it runs on this machine: `Auto` becomes `Isa(Isa::detect())`; the
    /// others stay as they are.
    ///
    /// # Errors
    ///
    /// [`IsaUnavailable`] for `Isa(isa)` if this machine cannot run `isa`
    /// ([`Isa::is_available`]).
    pub fn resolve(self) -> Result<Self, IsaUnavailable> {
        match self {
            Self::Auto => Ok(Self::Isa(Isa::detect())),
            Self::Reference => Ok(Self::Reference),
            Self::Isa(isa) if isa.is_available() => Ok(Self::Isa(isa)),
            Self::Isa(isa) => Err(IsaUnavailable { isa }),
        }
    }

    /// The kernel of the choice, `None` for the reference.
    fn kernel<T: SimdScalar>(self) -> Result<Option<P2pKernel<T>>, IsaUnavailable> {
        match self {
            Self::Auto => Ok(Some(P2pKernel::detect())),
            Self::Reference => Ok(None),
            Self::Isa(isa) => P2pKernel::new(isa).map(Some),
        }
    }
}

/// Prints `auto`, `reference` or the ISA's name.
impl fmt::Display for P2pChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Reference => f.write_str("reference"),
            Self::Isa(isa) => isa.fmt(f),
        }
    }
}

/// The error of parsing a [`P2pChoice`] from text it does not name.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error(
    "`{text}` is not a P2P kernel; expected `auto`, `reference` or an instruction set: {isas}",
    isas = Isa::all().map(|isa| format!("`{isa}`")).collect::<Vec<_>>().join(", ")
)]
pub struct UnknownP2pChoice {
    /// The text that was parsed.
    pub text: String,
}

/// Parses the [`Display`](fmt::Display) form: `auto`, `reference` or an ISA name
/// (`scalar`, `neon`, `avx2`), whether or not this machine can run the ISA.
impl FromStr for P2pChoice {
    type Err = UnknownP2pChoice;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "reference" => Ok(Self::Reference),
            _ => Isa::all()
                .find(|isa| isa.to_string() == s)
                .map(Self::Isa)
                .ok_or_else(|| UnknownP2pChoice { text: s.to_owned() }),
        }
    }
}

/// The Laplace kernel on the batched interface of `nd-fmm-plan`; see the
/// [module documentation](self).
#[derive(Clone, Debug)]
pub struct LaplaceOperator<T: SimdScalar> {
    kernels: Kernels<T>,
    execution: Execution<T>,
}

/// The read-only part of the operator: what every target body shares.
#[derive(Clone, Debug)]
pub(crate) struct Kernels<T: SimdScalar> {
    p: usize,
    tables: Tables<T>,
    gradients: bool,
    max_leaf_points: usize,
    /// The P2P kernel of `nd-fmm-simd`, or `None` for `nd_fmm_ref::p2p::p2p`.
    p2p: Option<P2pKernel<T>>,
}

/// One scratch set: what a target body writes besides its output.
#[derive(Clone, Debug)]
pub(crate) struct Scratch<T: RealScalar> {
    workspace: Workspace<T>,
    tables: TableScratch<T>,
    /// The sources of another leaf in the target's coordinates (P2P); fixed length.
    mapped: Vec<[T; 3]>,
}

impl<T: SimdScalar> Scratch<T> {
    /// A scratch set for `kernels`.
    fn new(kernels: &Kernels<T>) -> Self {
        Self {
            workspace: Workspace::new(kernels.p),
            tables: kernels.tables.scratch(),
            mapped: vec![[T::zero(); 3]; kernels.max_leaf_points],
        }
    }
}

/// Where the level calls run, and the scratch sets they use (module documentation,
/// "Threads" and "Scratch").
#[derive(Debug)]
struct Execution<T: RealScalar> {
    /// One set, or one per thread of `pool`, indexed by `rayon::current_thread_index()`.
    /// Set 0 also serves the serial path and the per-pair methods.
    scratch: Vec<Mutex<Scratch<T>>>,
    /// The pool of the level calls, if any.
    pool: Option<Arc<ThreadPool>>,
    /// Runs the level calls serially even with a pool.
    serial: bool,
}

/// A copy shares the pool and gets its own scratch sets.
impl<T: RealScalar> Clone for Execution<T> {
    fn clone(&self) -> Self {
        Self {
            scratch: self
                .scratch
                .iter()
                .map(|set| Mutex::new(set.lock().unwrap_or_else(PoisonError::into_inner).clone()))
                .collect(),
            pool: self.pool.clone(),
            serial: self.serial,
        }
    }
}

impl<T: RealScalar> Execution<T> {
    /// Scratch set 0, without locking.
    fn first(&mut self) -> &mut Scratch<T> {
        self.scratch[0]
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs `body(t, chunk, scratch)` for every chunk t of `output`: in index order on
    /// the calling thread, or in the pool with `par_chunks_mut`.
    fn boxes<F>(&mut self, mut output: LevelSliceMut<'_, T>, body: F)
    where
        F: Fn(usize, &mut [T], &mut Scratch<T>) + Sync,
    {
        match (&self.pool, self.serial) {
            (Some(pool), false) => {
                let (sets, size) = (&self.scratch, output.size());
                pool.install(|| {
                    output
                        .as_mut_slice()
                        .par_chunks_mut(size)
                        .enumerate()
                        .for_each(|(t, chunk)| with_scratch(sets, |s| body(t, chunk, s)));
                });
            }
            _ => {
                let scratch = self.first();
                for (t, chunk) in output.chunks_mut().enumerate() {
                    body(t, chunk, scratch);
                }
            }
        }
    }

    /// Runs `body(r, chunk, scratch)` for every leaf r of `output`: in index order on the
    /// calling thread, or in the pool, split by [`split_leaves`].
    fn leaves<F>(&mut self, mut output: LeafSliceMut<'_, T>, body: F)
    where
        F: Fn(usize, &mut [T], &mut Scratch<T>) + Sync,
    {
        match (&self.pool, self.serial) {
            (Some(pool), false) => {
                let sets = &self.scratch;
                pool.install(|| split_leaves(output, 0, sets, &body));
            }
            _ => {
                let scratch = self.first();
                for (r, chunk) in output.chunks_mut().enumerate() {
                    body(r, chunk, scratch);
                }
            }
        }
    }
}

/// Runs `body(first + r, chunk r, scratch)` for every leaf r of `output`, halving the
/// leaves recursively under `rayon::join`: one disjoint mutable slice per leaf, by the
/// CSR offsets, without allocation.
fn split_leaves<T: RealScalar, F>(
    mut output: LeafSliceMut<'_, T>,
    first: usize,
    sets: &[Mutex<Scratch<T>>],
    body: &F,
) where
    F: Fn(usize, &mut [T], &mut Scratch<T>) + Sync,
{
    match output.nleaves() {
        0 => {}
        1 => with_scratch(sets, |s| body(first, output.chunk_mut(0), s)),
        n => {
            let mid = n / 2;
            let (left, right) = output.split_at_mut(mid);
            rayon::join(
                || split_leaves(left, first, sets, body),
                || split_leaves(right, first + mid, sets, body),
            );
        }
    }
}

/// Runs `f` with the scratch set of the current pool thread.
///
/// # Panics
///
/// Outside a pool thread, or if the set is locked: only this thread locks it, and only
/// around a target body that never yields to rayon, so that would be a defect.
#[inline]
fn with_scratch<T: RealScalar>(sets: &[Mutex<Scratch<T>>], f: impl FnOnce(&mut Scratch<T>)) {
    let i = rayon::current_thread_index().expect("a level call runs on the pool's threads");
    let mut set = match sets[i].try_lock() {
        Ok(set) => set,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => panic!("the scratch set of pool thread {i} is in use"),
    };
    f(&mut set);
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

impl<T: SimdScalar> LaplaceOperator<T> {
    /// Creates the operator for the degree of `tables`, with gradients in the target
    /// output if `gradients`, for leaves with at most `max_leaf_points` source points.
    ///
    /// `max_leaf_points` must bound the source count of every leaf that P2P reads,
    /// local and ghost: on several ranks a global maximum, one all-reduce, taken by the
    /// caller. Allocates the scratch: a `Workspace` for p, the tables' scratch and
    /// 3 `max_leaf_points` values for P2P.
    ///
    /// The operator runs serially; [`with_pool`](Self::with_pool) gives it threads. Its
    /// P2P kernel is [`P2pChoice::Auto`]; [`with_p2p`](Self::with_p2p) chooses another.
    pub fn new(tables: Tables<T>, gradients: bool, max_leaf_points: usize) -> Self {
        let kernels = Kernels {
            p: tables.p(),
            tables,
            gradients,
            max_leaf_points,
            p2p: Some(P2pKernel::detect()),
        };
        let execution = Execution {
            scratch: vec![Mutex::new(Scratch::new(&kernels))],
            pool: None,
            serial: false,
        };
        Self { kernels, execution }
    }

    /// Runs every level call in `pool`, each target on one of its threads ([module
    /// documentation](self#threads-c35)). Allocates one scratch set per pool thread,
    /// replacing the operator's sets.
    ///
    /// The output is bit-identical to that of the serial operator. Only the threads of
    /// `pool` run targets; the global rayon pool is never used.
    pub fn with_pool(mut self, pool: Arc<ThreadPool>) -> Self {
        let n = pool.current_num_threads().max(1);
        self.execution.scratch = (0..n)
            .map(|_| Mutex::new(Scratch::new(&self.kernels)))
            .collect();
        self.execution.pool = Some(pool);
        self
    }

    /// Runs P2P with the kernel of `choice` ([`P2pChoice`]). Every other operator, the
    /// order of every sum and the scratch stay as they are.
    ///
    /// # Errors
    ///
    /// [`IsaUnavailable`] for `P2pChoice::Isa(isa)` if this machine cannot run `isa`.
    pub fn with_p2p(mut self, choice: P2pChoice) -> Result<Self, IsaUnavailable> {
        self.kernels.p2p = choice.kernel()?;
        Ok(self)
    }

    /// Returns the P2P kernel as it runs: [`P2pChoice::Reference`] or
    /// [`P2pChoice::Isa`] with the ISA of the kernel, never [`P2pChoice::Auto`]
    /// ([`P2pChoice::resolve`]).
    pub fn p2p_kernel(&self) -> P2pChoice {
        match &self.kernels.p2p {
            None => P2pChoice::Reference,
            Some(kernel) => P2pChoice::Isa(kernel.isa()),
        }
    }

    /// Returns the number of threads the level calls run on: the pool's, or 1 without a
    /// pool or when [`set_serial`](Self::set_serial) is on.
    pub fn threads(&self) -> usize {
        match (&self.execution.pool, self.execution.serial) {
            (Some(pool), false) => pool.current_num_threads(),
            _ => 1,
        }
    }

    /// The pool the level calls run on: `None` without a pool or when
    /// [`set_serial`](Self::set_serial) is on. `Fmm` runs its charge load and output pass
    /// on it too (Phase 4S T9).
    pub(crate) fn pool(&self) -> Option<&Arc<ThreadPool>> {
        self.execution
            .pool
            .as_ref()
            .filter(|_| !self.execution.serial)
    }

    /// With `serial`, runs the level calls on the calling thread even if the operator
    /// has a pool, with scratch set 0: the serial path of the same operator, for
    /// comparisons and timings. Without a pool it changes nothing.
    pub fn set_serial(&mut self, serial: bool) {
        self.execution.serial = serial;
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
    /// of another leaf, in scratch set 0. It is `max_leaf_points` from construction on
    /// and never changes: no operator allocates (tests check it).
    pub fn scratch_capacity(&self) -> usize {
        self.scratch_capacities()[0]
    }

    /// Returns [`scratch_capacity`](Self::scratch_capacity) for every scratch set: one,
    /// or one per pool thread. Each is `max_leaf_points` and never changes.
    pub fn scratch_capacities(&self) -> Vec<usize> {
        self.execution
            .scratch
            .iter()
            .map(|set| {
                set.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .mapped
                    .capacity()
            })
            .collect()
    }

    /// P2M: adds the multipole of the source chunk `sources` of a leaf, in the leaf's
    /// frame (CONVENTIONS §3.13: `leaf::p2m` at the unit frame on u). O(n p²).
    ///
    /// # Panics
    ///
    /// If `sources.len()` is not a multiple of 4, or `multipole` does not have length
    /// (p + 1)².
    pub fn p2m_leaf(&mut self, sources: &[T], multipole: &mut [T]) {
        self.kernels.p2m(sources, multipole, self.execution.first());
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
            .m2m(o, child_multipole, parent_multipole, self.execution.first());
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
            .m2l(d, source_multipole, target_local, self.execution.first());
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
            .p2l(&frame, sources, target_local, self.execution.first());
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
            .l2l(o, parent_local, child_local, self.execution.first());
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
            .l2p(local, target_input, target_output, self.execution.first());
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
            self.execution.first(),
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
            self.execution.first(),
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

impl<T: SimdScalar> Kernels<T> {
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
        self.p2p_points(points, charges, targets, potential, gradient);
    }

    /// P2P of `sources` with `charges` at `targets`, by the operator's kernel.
    #[inline]
    fn p2p_points(
        &self,
        sources: &[[T; 3]],
        charges: &[T],
        targets: &[[T; 3]],
        potential: &mut [T],
        gradient: Option<&mut [[T; 3]]>,
    ) {
        match &self.p2p {
            None => p2p::p2p(sources, charges, targets, potential, gradient),
            Some(kernel) => kernel.evaluate(sources, charges, targets, potential, gradient),
        }
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

impl<T: SimdScalar + Equivalence + Default> FmmSizes for LaplaceOperator<T> {
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

/// Target by target, each target's row in order: serially in index order, or in the
/// operator's pool; see the [module documentation](self#batched-execution-and-order).
impl<T: SimdScalar + Equivalence + Default> FmmOperator for LaplaceOperator<T> {
    fn p2m(&mut self, batch: P2m<'_, T>) {
        let P2m {
            level,
            index,
            leaves,
            sources,
            multipoles,
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: leaves,
            input: sources,
        };
        let kernels = &self.kernels;
        self.execution.boxes(multipoles, |t, multipole, scratch| {
            kernels.p2m_target(&call, t, multipole, scratch);
        });
    }

    fn m2m(&mut self, batch: M2m<'_, T>) {
        let M2m {
            level,
            index,
            children,
            child_multipoles,
            multipoles,
            ..
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: children,
            input: child_multipoles,
        };
        let kernels = &self.kernels;
        self.execution.boxes(multipoles, |t, multipole, scratch| {
            kernels.m2m_target(&call, t, multipole, scratch);
        });
    }

    fn m2l(&mut self, batch: M2l<'_, T>) {
        let M2l {
            level,
            index,
            pairs,
            multipoles,
            locals,
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: pairs,
            input: multipoles,
        };
        let kernels = &self.kernels;
        self.execution.boxes(locals, |t, local, scratch| {
            kernels.m2l_target(&call, t, local, scratch);
        });
    }

    fn p2l(&mut self, batch: P2l<'_, T>) {
        let P2l {
            level,
            index,
            x,
            sources,
            locals,
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: x,
            input: sources,
        };
        let kernels = &self.kernels;
        self.execution.boxes(locals, |t, local, scratch| {
            kernels.p2l_target(&call, t, local, scratch);
        });
    }

    fn l2l(&mut self, batch: L2l<'_, T>) {
        let L2l {
            level,
            index,
            parents,
            parent_locals,
            locals,
        } = batch;
        let call = BoxCall {
            level,
            index,
            view: parents,
            input: parent_locals,
        };
        let kernels = &self.kernels;
        self.execution.boxes(locals, |t, local, scratch| {
            kernels.l2l_target(&call, t, local, scratch);
        });
    }

    fn l2p(&mut self, batch: L2p<'_, T>) {
        let L2p {
            level,
            index,
            leaves,
            boxes,
            locals,
            target_input,
            target_output,
        } = batch;
        let call = LeafCall {
            level,
            index,
            first_leaf: leaves.start,
            view: boxes,
            input: locals,
            target_input,
        };
        let kernels = &self.kernels;
        self.execution.leaves(target_output, |r, output, scratch| {
            kernels.l2p_target(&call, r, output, scratch);
        });
    }

    fn m2p(&mut self, batch: M2p<'_, T>) {
        let M2p {
            level,
            index,
            leaves,
            w,
            multipoles,
            target_input,
            target_output,
        } = batch;
        let call = LeafCall {
            level,
            index,
            first_leaf: leaves.start,
            view: w,
            input: multipoles,
            target_input,
        };
        let kernels = &self.kernels;
        self.execution.leaves(target_output, |r, output, scratch| {
            kernels.m2p_target(&call, r, output, scratch);
        });
    }

    fn p2p(&mut self, batch: P2p<'_, T>) {
        let P2p {
            level,
            index,
            leaves,
            near,
            sources,
            target_input,
            target_output,
        } = batch;
        let call = LeafCall {
            level,
            index,
            first_leaf: leaves.start,
            view: near,
            input: sources,
            target_input,
        };
        let kernels = &self.kernels;
        self.execution.leaves(target_output, |r, output, scratch| {
            kernels.p2p_target(&call, r, output, scratch);
        });
    }
}

/// The same kernels pair by pair, for `nd_fmm_plan::operator::PerPair`: tables by the
/// given octant or offset index (debug builds check it against the keys), frames from
/// the keys, the self pair of P2P by `source == target`.
impl<T: SimdScalar + Equivalence + Default> PairOperator for LaplaceOperator<T> {
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
        self.kernels.m2m(
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
            self.execution.first(),
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
            .l2l(octant, parent_local, child_local, self.execution.first());
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

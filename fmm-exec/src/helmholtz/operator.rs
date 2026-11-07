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
//! | M2L under [`HelmholtzM2l::PlaneWave`] | `nd_fmm_tables::helmholtz::plane_wave::merge::PlaneWaveM2l`, merge and shift on the levels with a quadrature, RCR on the others |
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
//! M2L runs by the level's RCR tables ([`HelmholtzM2l::Rotation`], the operator's default
//! until [`with_m2l`](HelmholtzOperator::with_m2l); `HelmholtzFmmBuilder`'s is
//! [`HelmholtzM2l::Auto`]), or under
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
//! [`HelmholtzM2l::Auto`] (Helmholtz CH3.2d) is the rule per level: Batched in the fused
//! mode on every vector ISA ([`auto_m2l_mul_add`]; unless
//! [`with_m2l_mul_add`](HelmholtzOperator::with_m2l_mul_add) names a mode) on the levels of degree p_l at most a crossover that fit the budget, from
//! the finest level up (`BatchedM2l::select_within`), RCR on the others. The crossover is
//! [`auto_batched_max_degree`] of the column kernel's ISA and the multipole planes, or the
//! degree named by [`with_m2l_max_degree`](HelmholtzOperator::with_m2l_max_degree); so the
//! choice is a function of the degrees, the budget and the machine class only (proposal
//! §2, constraint 9), and each level runs Batched's or Rotation's level call as those
//! settings do, deterministic and rank-invariant alike. `HelmholtzFmmBuilder` agrees the
//! crossover over the ranks.
//!
//! Under [`HelmholtzM2l::PlaneWave`] (Helmholtz CH6.1d; opt-in; CONVENTIONS §3.14, "Merge
//! and shift", with the conversions and shifts of §3.15, "Exponential M2L") a level runs the
//! exponential M2L where its [`PlaneWavePrecision`] gives it a quadrature: by the rule
//! (`nd_fmm_tables::helmholtz::plane_wave::merge::select`, measured in the FMM in Helmholtz
//! CH6.1e: iprec 2 at the FMM's tolerance ε ≥ 10⁻⁶ and none below,
//! [`with_plane_wave_tolerance`](HelmholtzOperator::with_plane_wave_tolerance), or without
//! one by the level's degree; a function of k, the level's width, its degree and ε) or
//! named per level, and only where κ_l = k w_l lies in the quadrature's tables; RCR on the
//! others. Under the rule the FMM's errors stay within about 1.1× Rotation's on CH3.1c's
//! cases (`fmm-validate/results/helmholtz-ch6-1e.md`). Its level call restates Laplace's
//! (`crate::operator`, "The plane-wave M2L") without its vector kernels, by chunks of
//! parents (Helmholtz CH6.2a): the level is cut, in index order, into chunks of consecutive
//! whole parents whose computed children's distinct V sources fit the waves buffer (each
//! parent joins the chunk while they fit; one parent's sources, at most its 216 slots, always
//! do). For each chunk, each charge vector and each frame f = 0, 1, 2,
//!
//! 1. **outgoing**: the up and down waves of frame f of the chunk's sources, own and ghost
//!    boxes (the multipole rotated into the frame by Oⁿ(Q), its modes and the waves, 2W
//!    reals each, every value complex), the pool splitting the sources;
//! 2. **merge and shift**, parent by parent (consecutive boxes with one parent key), the
//!    pool splitting the chunk at parent boundaries: the parent's computed children (with
//!    a V row) and their sources by slot; for the up and then the down direction of f,
//!    every accumulator whose (geometric) mask holds a computed child sums its slots
//!    present in ascending order, each source's waves times its source shift, from zero;
//!    every computed child adds its accumulators in ascending mask order and multiplies the
//!    sum by its child shift; then each computed child with waves converts its up and then
//!    its down waves into a local in the frame and adds it, rotated back by Oⁿ(Q)ᵀ.
//!
//! Every local receives frame 0, then 1, then 2, each a fixed chain of its parent's
//! accumulators, which depend on the geometry of the slots alone, and a box's waves are a
//! function of its multipole alone, so the output is the same at every thread count and
//! every bound of the buffer and, as Laplace's argument goes, on P ranks the one-rank output
//! bit for bit. A source read by two chunks has its waves computed in each. The pair body ([`PairOperator`]) translates each pair by
//! `PlaneWaveLevel::m2l` along the direction merge and shift gives it
//! (`PlaneWaveTables::pair_direction` of the offset and the target's child index), which
//! the level call equals to rounding (its two shift factors are the pair's to rounding).
//! It approximates M2L within its quadrature's error, so it is not Rotation's output. The
//! waves buffer holds one frame of a chunk's sources, 2W reals per box, at most a byte bound
//! ([`with_plane_wave_buffer`](HelmholtzOperator::with_plane_wave_buffer); none: a whole
//! level; at least [`MIN_BUFFER_BOXES`] boxes of each level), with a position per box of the
//! largest level and the chunk's source list, sized from the plan by `HelmholtzFmmBuilder`
//! ([`with_plane_wave_boxes`](HelmholtzOperator::with_plane_wave_boxes)), which bounds it
//! by its node budget over the ranks per node (proposal §2, constraint 7); each scratch set
//! holds a level's buffers and the pair scratch of every level with waves;
//! [`plane_wave_bytes`](HelmholtzOperator::plane_wave_bytes) reports them with the tables,
//! and [`plane_wave_counts`](HelmholtzOperator::plane_wave_counts) the chunks and the waves
//! computed.
//! f64 only (§3.15, "Precision and range"): f32's range is untested (the factors reach
//! about 10³⁴ at iprec 3).
//!
//! Every M2L level call, under every setting, is timed on the calling thread's clock and
//! summed per level ([`m2l_times`](HelmholtzOperator::m2l_times); Helmholtz CH6.2b): two
//! clock reads per call, no value changed. The exponential M2L's level call also times its
//! two passes per chunk, vector and frame, the waves and merge and shift
//! ([`plane_wave_times`](HelmholtzOperator::plane_wave_times); Helmholtz CH6.2d): four clock
//! reads more, no value changed.
//!
//! # Execution
//!
//! The level calls run on the shared drivers of `nd-fmm-exec` (Helmholtz CH2.0, CH3.5),
//! target by target in the plan's order, serially or on the operator's pool
//! ([`HelmholtzOperator::with_pool`]), each thread with its own scratch; the output is the
//! same at every thread count. The shared batched M2L does not apply (`driver::m2l_block`
//! assumes Laplace's layout); under [`HelmholtzM2l::Batched`] the M2L level call of a level
//! with tables runs the operator's own block instead, and under [`HelmholtzM2l::PlaneWave`]
//! a level with waves merge and shift ("M2L"). [`PairOperator`] runs the same
//! bodies pair by pair.
//!
//! Yukawa (Re k = 0) is not served: its real path is CH2.1's.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
use nd_fmm_tables::helmholtz::plane_wave::Precision;
use nd_fmm_tables::helmholtz::plane_wave::merge::{PlaneWaveM2l, select};
use nd_fmm_tables::helmholtz::plane_wave::quadrature::RHO_MAX;
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzScratch, HelmholtzTables};
use nd_octree::{MortonKey, morton};
use rayon::ThreadPool;
use thiserror::Error;

use super::batched::Batched;
use super::{Densities, LeafScratch, Wavenumber, WavenumberKind};
use crate::driver::{self, Columns, Execution, HostKernel, debug_check_octant, offset_index};
use crate::geometry::{Domain, radius, relative_frame};
use crate::operator::{M2L_BLOCK_TARGETS, P2pChoice};

#[path = "operator_plane_wave.rs"]
mod plane_wave;

pub use plane_wave::{MIN_BUFFER_BOXES, PlaneWaveCounts, PlaneWaveTimes};
use plane_wave::{PlaneBuffer, PlaneScratch};

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
    /// RCR on every level (the operator's default before
    /// [`with_m2l`](HelmholtzOperator::with_m2l)): rotation, coaxial translation, rotation
    /// back.
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
    /// The exponential M2L by merge and shift (Helmholtz CH6.1d; opt-in) on the levels with
    /// a quadrature of the [`PlaneWavePrecision`] setting
    /// ([`with_plane_wave_precision`](HelmholtzOperator::with_plane_wave_precision)), RCR on
    /// the others. f64 only (CONVENTIONS §3.15, "Precision and range"): in f32 the factors,
    /// up to about 10³⁴ at iprec 3, are untested and the setting is not refused.
    PlaneWave,
    /// The rule per level (Helmholtz CH3.2d): Batched, in [`auto_m2l_mul_add`]'s mode (fused
    /// on every vector ISA) unless one is named,
    /// on the levels of degree at most the crossover of the machine class
    /// ([`auto_batched_max_degree`]) whose tables fit the dense budget, from the finest level
    /// up; RCR on the others. The default of `HelmholtzFmmBuilder` (and of this type) since
    /// Helmholtz CH3.2d.
    #[default]
    Auto,
}

/// The precision of each level's quadrature under [`HelmholtzM2l::PlaneWave`] (Helmholtz
/// CH6.1d; module documentation, "M2L").
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PlaneWavePrecision {
    /// `nd_fmm_tables::helmholtz::plane_wave::merge::select`: on each level with a V list
    /// whose κ_l = k w_l lies in the quadrature's tables, the precision
    /// `precision_for_tolerance` gives the FMM's tolerance
    /// ([`HelmholtzOperator::with_plane_wave_tolerance`]), or without one the precision
    /// `precision_for_degree` gives the level's degree (measured in the FMM, Helmholtz
    /// CH6.1e), none elsewhere. A function of k, the level's width, its degree and the
    /// tolerance only (proposal §2, constraint 9).
    #[default]
    Rule,
    /// The precision of each level by index, `None` for RCR, for the measurements and the
    /// tests. Entries for levels 0 and 1 (no V list) and beyond the tree are ignored, and
    /// levels beyond the list take none; a level whose κ_l lies outside the tables runs RCR
    /// whatever it names.
    PerLevel(Vec<Option<Precision>>),
}

/// The crossover degree of [`HelmholtzM2l::Auto`] (Helmholtz CH3.2d; the operator module
/// documentation, "M2L"): the largest p_l at which the rule runs the fused Batched M2L, for
/// the column kernel's ISA (the machine class) and the multipole's planes (1 for real k with
/// real charges, 2 otherwise). A function of these two alone, so every rank of one machine
/// class chooses alike (proposal §2, constraint 9). Set from the end-to-end crossover sweeps
/// of fmm-validate/results/helmholtz-ch3-2d.md: NEON on the M6 and locust, AVX-512 and AVX2
/// on Kathleen; the scalar kernel from its cost per pair on the M6 and Kathleen.
pub fn auto_batched_max_degree(isa: Isa, planes: usize) -> usize {
    let one_plane = planes == 1;
    match (isa, one_plane) {
        (Isa::Avx512, _) => 15,
        (Isa::Avx2, false) => 11,
        (Isa::Scalar, false) => 6,
        (Isa::Scalar, true) => 8,
        // NEON (and any later ISA until measured); AVX2 with one plane.
        (_, false) => 10,
        (_, true) => 12,
    }
}

/// The column kernel's mode of [`HelmholtzM2l::Auto`] on `isa` unless one is named
/// (Helmholtz CH3.2d): [`MulAdd::Fused`], where each step is one hardware fma, but
/// [`MulAdd::Separate`] for the scalar kernel in a build whose target has no fma
/// instruction (x86_64 without the `fma` target feature), where `mul_add` is a library
/// call: about 10× Rotation's cost per pair on Kathleen at p = 6
/// (fmm-validate/results/helmholtz-ch3-2d.md, §3). A function of the ISA and the build only,
/// so every rank of one machine class and build chooses alike.
pub fn auto_m2l_mul_add(isa: Isa) -> MulAdd {
    let hardware_fma = cfg!(any(target_arch = "aarch64", target_feature = "fma"));
    match isa {
        Isa::Scalar if !hardware_fma => MulAdd::Separate,
        _ => MulAdd::Fused,
    }
}

/// The default byte budget of the dense or class M2L tables of one rank, 1 GiB.
pub const DEFAULT_DENSE_BUDGET: usize = 1 << 30;

/// The Helmholtz-family operator (module documentation).
pub struct HelmholtzOperator<T: SimdScalar> {
    kernel: Kernel<T>,
    execution: Execution<Scratch<T>>,
    /// Under [`HelmholtzM2l::PlaneWave`], the level call's waves of one frame of a chunk's
    /// sources, up and down, and their positions.
    plane_buffer: PlaneBuffer<T>,
    /// The M2L level calls' wall time per level since the tables were last chosen
    /// ([`m2l_times`](HelmholtzOperator::m2l_times)).
    m2l_times: Vec<Duration>,
    /// The exponential M2L's passes per level since the tables were last chosen
    /// ([`plane_wave_times`](HelmholtzOperator::plane_wave_times)).
    plane_wave_times: Vec<PlaneWaveTimes>,
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
    /// The byte budget of the M2L tables ([`HelmholtzOperator::with_m2l`]).
    dense_budget: usize,
    /// The crossover degree named for [`HelmholtzM2l::Auto`], or `None` for the rule's.
    max_degree: Option<usize>,
    /// The batched M2L's multiply–add mode if named, else `None` (the setting's default).
    mul_add: Option<MulAdd>,
    /// The dense M2L tables of the levels that have them, under [`HelmholtzM2l::Dense`].
    dense: Option<DenseM2l<T>>,
    /// The class M2L tables of the levels that have them, under [`HelmholtzM2l::Classes`].
    classes: Option<ClassM2l<T>>,
    /// The batched M2L tables of the levels that have them, under [`HelmholtzM2l::Batched`].
    batched: Option<BatchedM2l<T>>,
    /// The quadratures' precisions under [`HelmholtzM2l::PlaneWave`].
    plane_wave_precision: PlaneWavePrecision,
    /// The FMM's tolerance, which [`PlaneWavePrecision::Rule`] reads, if it has one.
    plane_wave_tolerance: Option<f64>,
    /// The lateral reach of the quadratures' α counts (`RHO_MAX`, the region's corner,
    /// unless named; Helmholtz CH6.2e).
    plane_wave_reach: f64,
    /// The exponential M2L's tables of the levels that have them, under
    /// [`HelmholtzM2l::PlaneWave`].
    plane_waves: Option<PlaneWaveM2l<T>>,
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
    /// The exponential M2L's buffers, under [`HelmholtzM2l::PlaneWave`].
    plane: Option<PlaneScratch<T>>,
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
            dense_budget: DEFAULT_DENSE_BUDGET,
            max_degree: None,
            mul_add: None,
            dense: None,
            classes: None,
            batched: None,
            plane_wave_precision: PlaneWavePrecision::Rule,
            plane_wave_tolerance: None,
            plane_wave_reach: RHO_MAX,
            plane_waves: None,
            matrix: MatrixKernel::detect(MulAdd::Separate),
            p2p: Some(P2pKernel::detect()),
        };
        let execution = Execution {
            scratch: vec![Mutex::new(Scratch::new(&kernel))],
            pool: None,
            serial: false,
        };
        let m2l_times = vec![Duration::ZERO; kernel.tables.nlevels()];
        let plane_wave_times = vec![PlaneWaveTimes::default(); kernel.tables.nlevels()];
        Ok(Self {
            kernel,
            execution,
            plane_buffer: PlaneBuffer::default(),
            m2l_times,
            plane_wave_times,
        })
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
    /// Dense's tables in split planes (`BatchedM2l::select`, Dense's), RCR on the others;
    /// under [`HelmholtzM2l::Auto`] Dense's tables in split planes on the levels of degree at
    /// most the crossover ([`max_degree`](Self::max_degree)) that fit the budget
    /// (`BatchedM2l::select_within`), RCR on the others, with the column kernel in
    /// [`auto_m2l_mul_add`]'s mode unless [`with_m2l_mul_add`](Self::with_m2l_mul_add) named
    /// one (before or after; the other settings' default is [`MulAdd::Separate`]);
    /// under [`HelmholtzM2l::Rotation`] ignores the budget. The tables are built in f64,
    /// serially, O(p⁵) per offset and level (16 offsets a level for Classes, 316 for Dense).
    pub fn with_m2l(mut self, choice: HelmholtzM2l, dense_budget: usize) -> Self {
        self.kernel.m2l = choice;
        self.kernel.dense_budget = dense_budget;
        let mul_add = self.kernel.mul_add_for(self.kernel.matrix.isa());
        self.kernel.matrix = MatrixKernel::new(self.kernel.matrix.isa(), mul_add)
            .expect("the kernel's own ISA is available");
        self.select_tables();
        self
    }

    /// Names the crossover degree of [`HelmholtzM2l::Auto`] (`None`: the rule's,
    /// [`auto_batched_max_degree`] of the column kernel's ISA and the multipole planes), for
    /// the measurements, the tests and an agreement over ranks; the other settings ignore
    /// it. Under Auto the levels are chosen again if the degree changes.
    pub fn with_m2l_max_degree(mut self, max_degree: Option<usize>) -> Self {
        let before = self.kernel.auto_max_degree();
        self.kernel.max_degree = max_degree;
        if self.kernel.m2l == HelmholtzM2l::Auto && self.kernel.auto_max_degree() != before {
            self.select_tables();
        }
        self
    }

    /// The crossover degree under [`HelmholtzM2l::Auto`]: the named one, else the rule's for
    /// the column kernel's ISA and the multipole planes; `None` under the other settings.
    pub fn max_degree(&self) -> Option<usize> {
        (self.kernel.m2l == HelmholtzM2l::Auto).then(|| self.kernel.auto_max_degree())
    }

    /// Takes each level's quadrature under [`HelmholtzM2l::PlaneWave`] by `precision`
    /// ([`PlaneWavePrecision::Rule`] by default), before or after
    /// [`with_m2l`](Self::with_m2l); the other settings ignore it. Under PlaneWave the levels'
    /// tables are built again.
    pub fn with_plane_wave_precision(mut self, precision: PlaneWavePrecision) -> Self {
        self.kernel.plane_wave_precision = precision;
        if self.kernel.m2l == HelmholtzM2l::PlaneWave {
            self.select_tables();
        }
        self
    }

    /// Gives [`PlaneWavePrecision::Rule`] the FMM's tolerance ε (`Degrees::Tolerance`'s;
    /// `None`, the default, for named degrees, when the rule reads each level's degree),
    /// before or after [`with_m2l`](Self::with_m2l). Under PlaneWave the levels' tables are
    /// built again.
    pub fn with_plane_wave_tolerance(mut self, tolerance: Option<f64>) -> Self {
        self.kernel.plane_wave_tolerance = tolerance;
        if self.kernel.m2l == HelmholtzM2l::PlaneWave {
            self.select_tables();
        }
        self
    }

    /// Takes the α counts of every level's quadrature under [`HelmholtzM2l::PlaneWave`] for
    /// the lateral offsets ρ ≤ `reach` (box widths; `Quadrature::with_reach`) in place of the
    /// region's corner 4√2 (`RHO_MAX`, the default), before or after
    /// [`with_m2l`](Self::with_m2l); fewer waves, which bound the kernel on ρ ≤ `reach` only.
    /// For the measurements (Helmholtz CH6.2e). Under PlaneWave the levels' tables are built
    /// again.
    ///
    /// # Panics
    ///
    /// Under PlaneWave, if `reach` is not finite or not above the source's reach √2/2.
    pub fn with_plane_wave_reach(mut self, reach: f64) -> Self {
        self.kernel.plane_wave_reach = reach;
        if self.kernel.m2l == HelmholtzM2l::PlaneWave {
            self.select_tables();
        }
        self
    }

    /// The lateral reach of the quadratures' α counts
    /// ([`with_plane_wave_reach`](Self::with_plane_wave_reach)).
    pub fn plane_wave_reach(&self) -> f64 {
        self.kernel.plane_wave_reach
    }

    /// The waves per direction of `level`, Σₖ Nₖ of its quadrature, under
    /// [`HelmholtzM2l::PlaneWave`]; 0 where the level runs RCR (and under the other
    /// settings).
    pub fn plane_wave_waves(&self, level: usize) -> usize {
        self.kernel
            .plane_wave_level(level)
            .map_or(0, |l| l.tables().wave_len() / 2)
    }

    /// The precision of `level`'s quadrature under [`HelmholtzM2l::PlaneWave`], `None` where
    /// the level runs RCR (and under the other settings).
    pub fn plane_wave_precision(&self, level: usize) -> Option<Precision> {
        self.kernel.plane_wave_level(level).map(|l| l.precision())
    }

    /// Sizes the buffers of the exponential M2L's level call for levels of `boxes[l]` boxes
    /// (each level's own and ghost boxes), so that no level call grows them; nothing unless
    /// [`HelmholtzM2l::PlaneWave`]. `HelmholtzFmmBuilder` sizes them from the plan.
    pub fn with_plane_wave_boxes(mut self, boxes: &[usize]) -> Self {
        self.plane_buffer.level_boxes = boxes.to_vec();
        self.plane_buffer.size(self.kernel.plane_waves.as_ref());
        self
    }

    /// Bounds the waves buffer of the exponential M2L's level call at `bytes` (`None`: every
    /// box of a level, the default). A level call then runs by chunks of whole parents whose
    /// sources' waves fit the bound, at least [`MIN_BUFFER_BOXES`] boxes of each level
    /// ("M2L" in the module documentation); the output is the same at every bound.
    /// `HelmholtzFmmBuilder` sets it from its node budget
    /// ([`plane_wave_budget`](super::HelmholtzFmmBuilder::plane_wave_budget)).
    pub fn with_plane_wave_buffer(mut self, bytes: Option<usize>) -> Self {
        self.plane_buffer.bound = bytes;
        self.plane_buffer.size(self.kernel.plane_waves.as_ref());
        self
    }

    /// The waves buffer's byte bound ([`with_plane_wave_buffer`](Self::with_plane_wave_buffer)).
    pub fn plane_wave_buffer(&self) -> Option<usize> {
        self.plane_buffer.bound
    }

    /// The bytes of the waves buffer as sized (part of
    /// [`plane_wave_bytes`](Self::plane_wave_bytes)).
    pub fn plane_wave_buffer_bytes(&self) -> usize {
        self.plane_buffer.wave_bytes()
    }

    /// The counts of the exponential M2L's level calls since its tables were last chosen:
    /// level calls, chunks and the waves computed ([`PlaneWaveCounts`]).
    pub fn plane_wave_counts(&self) -> PlaneWaveCounts {
        self.plane_buffer.counts
    }

    /// The wall time of the M2L level calls of each level since the tables were last chosen
    /// (Helmholtz CH6.2b): one entry per level, the time of every [`FmmOperator::m2l`] call
    /// on it summed, whichever M2L ran there ([`m2l_on`](Self::m2l_on)); zero on the levels
    /// without a call. This rank's, on the calling thread's clock; the pair body is not timed.
    pub fn m2l_times(&self) -> &[Duration] {
        &self.m2l_times
    }

    /// The wall time of the exponential M2L's two passes on each level since the tables were
    /// last chosen (Helmholtz CH6.2d): one entry per level, zero where no level call ran merge
    /// and shift; within the level's [`m2l_times`](Self::m2l_times) entry. This rank's, on
    /// the calling thread's clock.
    pub fn plane_wave_times(&self) -> &[PlaneWaveTimes] {
        &self.plane_wave_times
    }

    /// The bytes the exponential M2L holds: its tables, the level call's buffers (the waves,
    /// a position per box and the chunk's sources) and every scratch set's buffers (0 unless
    /// [`HelmholtzM2l::PlaneWave`] gives a level waves).
    pub fn plane_wave_bytes(&self) -> usize {
        let tables = self
            .kernel
            .plane_waves
            .as_ref()
            .map_or(0, PlaneWaveM2l::bytes);
        let scratch: usize = self
            .execution
            .scratch
            .iter()
            .filter_map(|s| s.lock().ok()?.plane.as_ref().map(PlaneScratch::bytes))
            .sum();
        tables + self.plane_buffer.bytes() + scratch
    }

    /// Builds the M2L tables of the setting, the budget and (under Auto) the crossover, and
    /// the scratch sized for them.
    fn select_tables(&mut self) {
        let kernel = &mut self.kernel;
        let choice = kernel.m2l;
        let dense_budget = kernel.dense_budget;
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
        let max_degree = match choice {
            HelmholtzM2l::Batched => Some(usize::MAX),
            HelmholtzM2l::Auto => Some(kernel.auto_max_degree()),
            _ => None,
        };
        kernel.batched = max_degree.map(|max_degree| {
            let levels = BatchedM2l::<T>::select_within(&degrees, dense_budget, max_degree);
            BatchedM2l::build(k_root, &degrees, &levels)
        });
        kernel.plane_waves = (choice == HelmholtzM2l::PlaneWave).then(|| {
            let precisions: Vec<Option<Precision>> = match &kernel.plane_wave_precision {
                PlaneWavePrecision::Rule => select(k_root, &degrees, kernel.plane_wave_tolerance),
                PlaneWavePrecision::PerLevel(named) => (0..degrees.len())
                    .map(|l| named.get(l).copied().flatten().filter(|_| l >= 2))
                    .collect(),
            };
            PlaneWaveM2l::build_with_reach(k_root, &degrees, &precisions, kernel.plane_wave_reach)
        });
        // No level with waves: nothing to hold.
        kernel.plane_waves = kernel
            .plane_waves
            .take()
            .filter(|m2l| (0..m2l.nlevels()).any(|l| m2l.level(l).is_some()));
        self.plane_buffer.size(kernel.plane_waves.as_ref());
        self.plane_buffer.counts = PlaneWaveCounts::default();
        self.m2l_times.fill(Duration::ZERO);
        self.plane_wave_times.fill(PlaneWaveTimes::default());
        // The class scratch, the batched columns and the exponential M2L's buffers are sized
        // by their tables' largest level.
        for scratch in &mut self.execution.scratch {
            *scratch.get_mut().expect("a scratch set is never poisoned") = Scratch::new(kernel);
        }
    }

    /// The M2L setting ([`with_m2l`](Self::with_m2l)).
    pub fn m2l_choice(&self) -> HelmholtzM2l {
        self.kernel.m2l
    }

    /// The M2L that runs on `level`: [`HelmholtzM2l::Dense`], [`HelmholtzM2l::Classes`] or
    /// [`HelmholtzM2l::Batched`] where the level has dense, class or batched tables (under
    /// [`HelmholtzM2l::Auto`] the levels the rule gives Batched), [`HelmholtzM2l::PlaneWave`]
    /// where it has a quadrature, [`HelmholtzM2l::Rotation`] everywhere else (levels 0 and 1
    /// have no M2L).
    pub fn m2l_on(&self, level: usize) -> HelmholtzM2l {
        let kernel = &self.kernel;
        if kernel.plane_wave_level(level).is_some() {
            return HelmholtzM2l::PlaneWave;
        }
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

    /// The bytes of the batched M2L tables (0 unless [`HelmholtzM2l::Batched`] or
    /// [`HelmholtzM2l::Auto`]).
    pub fn batched_bytes(&self) -> usize {
        self.kernel.batched.as_ref().map_or(0, BatchedM2l::bytes)
    }

    /// Runs P2P with the kernel of `choice` (module documentation, "P2P kernel"), and the
    /// batched M2L's column kernel on its ISA (the scalar path for
    /// [`P2pChoice::Reference`]). Under [`HelmholtzM2l::Auto`] without a named crossover
    /// the levels are chosen again if the ISA's crossover differs.
    ///
    /// # Errors
    ///
    /// [`IsaUnavailable`] for `P2pChoice::Isa(isa)` if this machine cannot run `isa`.
    pub fn with_p2p(mut self, choice: P2pChoice) -> Result<Self, IsaUnavailable> {
        let before = self.kernel.auto_max_degree();
        self.kernel.p2p = match choice {
            P2pChoice::Auto => Some(P2pKernel::detect()),
            P2pChoice::Reference => None,
            P2pChoice::Isa(isa) => Some(P2pKernel::new(isa)?),
        };
        let isa = self.kernel.p2p.as_ref().map_or(Isa::Scalar, P2pKernel::isa);
        self.kernel.matrix = MatrixKernel::new(isa, self.kernel.mul_add_for(isa))?;
        if self.kernel.m2l == HelmholtzM2l::Auto && self.kernel.auto_max_degree() != before {
            self.select_tables();
        }
        Ok(self)
    }

    /// Runs the batched M2L's column kernel in the mode `mul_add` (module documentation,
    /// "M2L"; [`MulAdd::Separate`] by default): [`MulAdd::Fused`] forms each step as one
    /// fma, within n u of the separate form per element, so the output is Dense's to
    /// rounding also with real multipoles; deterministic, the same at every thread count,
    /// ISA and rank count. Only [`HelmholtzM2l::Batched`] and [`HelmholtzM2l::Auto`] read
    /// it. A named mode holds for every later [`with_m2l`](Self::with_m2l) and
    /// [`with_p2p`](Self::with_p2p); unnamed, Auto's is [`auto_m2l_mul_add`] of the ISA and
    /// the others' [`MulAdd::Separate`].
    pub fn with_m2l_mul_add(mut self, mul_add: MulAdd) -> Self {
        self.kernel.mul_add = Some(mul_add);
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
            plane: kernel.plane_waves.as_ref().map(PlaneScratch::new),
        }
    }
}

impl<T: SimdScalar> Kernel<T> {
    /// The column kernel's mode on `isa`: the named one, else the setting's default (under
    /// [`HelmholtzM2l::Auto`] [`auto_m2l_mul_add`], else [`MulAdd::Separate`]).
    fn mul_add_for(&self, isa: Isa) -> MulAdd {
        self.mul_add.unwrap_or(match self.m2l {
            HelmholtzM2l::Auto => auto_m2l_mul_add(isa),
            _ => MulAdd::Separate,
        })
    }

    /// The crossover of [`HelmholtzM2l::Auto`]: the named one, else the rule's.
    fn auto_max_degree(&self) -> usize {
        self.max_degree.unwrap_or_else(|| {
            auto_batched_max_degree(self.matrix.isa(), self.densities.multipole_planes())
        })
    }

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

    fn m2l(
        &self,
        (level, d, child): (usize, usize, usize),
        multipole: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        let planes = self.densities.multipole_planes();
        let m = self.densities.vectors();
        if let Some(merge) = self.plane_wave_level(level) {
            return self.plane_wave_pair(merge, (level, d, child), multipole, local, s);
        }
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
        child: usize,
        multipole: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        Kernel::m2l(self, (level, d, child), multipole, local, s);
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

    /// Under [`HelmholtzM2l::Batched`] block by block on the levels with tables, under
    /// [`HelmholtzM2l::PlaneWave`] by merge and shift on the levels with waves (module
    /// documentation, "M2L"), target by target elsewhere.
    ///
    /// Each call's wall time is added to its level's [`m2l_times`](Self::m2l_times).
    fn m2l(&mut self, batch: M2l<'_, T>) {
        let level = batch.level;
        let start = Instant::now();
        if self.kernel.plane_wave_level(level).is_some() {
            self.plane_wave_m2l(batch);
        } else {
            match self.kernel.batched(level) {
                Some(batched) => batched.m2l(&mut self.execution, batch, |s| &mut s.columns),
                None => driver::m2l(&self.kernel, &mut self.execution, batch),
            }
        }
        self.m2l_times[level] += start.elapsed();
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
        let child = morton::child_index(target);
        self.kernel.m2l(
            (level, offset, child),
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

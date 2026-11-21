//! The Helmholtz FMM (Helmholtz CH3.1b; docs/design/helmholtz-fmm-proposal.md §6.1–§6.3;
//! CONVENTIONS §3.15): [`HelmholtzFmmBuilder`] and [`HelmholtzFmm`], for real and complex
//! k with real and complex charges, on any number of ranks, RCR on every level.
//!
//! # Build
//!
//! [`HelmholtzFmmBuilder::build`] runs the point pipeline that [`Fmm`](crate::fmm::Fmm)
//! runs (the crate-private `pipeline` module; the `fmm` module documentation, "Build" and
//! "Several ranks"), with the Helmholtz operator in place of Laplace's:
//!
//! 1. the settings, the thread pool, finite points and the supplied domain, agreed on
//!    every rank; Yukawa (Re k = 0) is refused here, as its path is Helmholtz CH2.1's;
//! 2. whether any rank has a point;
//! 3. the domain, the supplied one or the bounding box of every rank's points, with every
//!    point strictly inside it;
//! 4. the octree of every point, with the leaf rule (proposal §3.7):
//!    [`min_leaf_level`] of k, the domain and c_leaf passed to
//!    `OctreeOptions::with_min_leaf_level`, so that no leaf holding a point is wider than
//!    c_leaf λ; and its box index (`nd_fmm_plan::index::BoxIndex::of_octree`). There is no
//!    near rule and no work cut: both weigh the work with Laplace's cost model;
//! 5. and 6. every point to the rank that owns its leaf, routed on the index's local
//!    leaves, the leaf ranges, radii and the output order; then the plan, built once on the
//!    index (`nd_fmm_plan::plan::Plan::with_rules_on`; Helmholtz CH4.0d1b), by default with
//!    the empty-box rule (Helmholtz CH4.0d1; [`HelmholtzFmmBuilder::empty_boxes`]): every
//!    box's subtree counts from the routes' points per local leaf
//!    (`nd_fmm_plan::near::LeafCounts::exchange_with_boxes`), so that the plan leaves out the
//!    work of the boxes whose subtree holds no target or no source point
//!    (`nd_fmm_plan::empty`). Every plan of the octree has the index's local leaves, so the
//!    routes hold. Only exact zeros go: the output is the rule off's as values (−0 and +0
//!    equal);
//! 7. the degree of every level ([`Degrees::resolve`]), the [`HelmholtzOperator`] with its
//!    tables, on the pool, and the evaluator; the largest leaf is agreed on every rank.
//!    Under [`HelmholtzM2l::PlaneWave`]'s rule below ε = 10⁻⁶, and under the default
//!    [`HelmholtzM2l::Auto`] at 10⁻⁸ ≤ ε < 10⁻⁷, the operator raises the
//!    degree of each level with waves to the resolving degree (Helmholtz CH6.1g), and
//!    [`HelmholtzFmm::degrees`] reports the degrees it runs at;
//! 8. the leaf-scaled coordinates (CONVENTIONS §3.13).
//!
//! The degrees, the leaf level and the tables depend on k, the domain, ε and c_leaf only
//! (proposal §2, constraint 9), so every rank finds the same.
//!
//! # Evaluation
//!
//! [`HelmholtzFmm::evaluate`] takes the m charge vectors of the build
//! ([`Charges`], all real or all complex as [`HelmholtzFmmBuilder::charges`] says) at this
//! rank's sources, in the caller's order, and returns the [`Output`] at this rank's
//! targets, in the caller's order, in §3.15's split planes, 1/(4π) included. The leaf
//! chunks of [`Densities`] are §3.13 chunks of c m source and o_m output vectors (the
//! `densities` module, "The leaf chunks"), so the evaluation is `Fmm`'s with those as the
//! vectors: the c m columns (Re q⁰, Im q⁰, …) packed point-major and forwarded to their
//! owners, the charge load, the six blocking stages of the evaluator, the output pass with
//! Laplace's scales 4π r and 4π r², the backward move, and the columns unpacked into the
//! output's planes.
//!
//! # Determinism
//!
//! The level calls run target by target on the shared drivers, each target's
//! contributions in the order of its row, so the output is the same at every thread count
//! and, on P ranks, the one-rank FMM's over the union of the points in rank order, bit for
//! bit (proposal §2, constraint 2), as for `Fmm`.
//!
//! # P2P kernel
//!
//! [`HelmholtzFmmBuilder::p2p_kernel`] chooses the kernel of P2P ([`P2pChoice`], Helmholtz
//! CH3.4c): by default `nd_fmm_simd::P2pKernel::evaluate_helmholtz` on the default ISA of
//! this machine, [`P2pChoice::Reference`] the trusted slower path
//! (`nd_fmm_ref::helmholtz::p2p::p2p`). The choice changes the output only in the last bits
//! (the kernel's accuracy contract, the [`operator`](super::operator) module documentation,
//! "P2P kernel"); [`HelmholtzFmm::p2p_kernel`] reports the kernel that runs.
//!
//! Not yet: overlap, the device path, the near rule and the work cut, and table caches.
//!
//! ```no_run
//! use mpi::traits::Communicator;
//! use nd_fmm_exec::helmholtz::{Charges, Degrees, HelmholtzFmmBuilder, Wavenumber};
//!
//! let universe = mpi::initialize().unwrap();
//! let world = universe.world();
//! let points: Vec<[f64; 3]> = (0..1000)
//!     .map(|i| {
//!         let t = i as f64 / 1000.0;
//!         [t, (7.0 * t).fract(), (13.0 * t).fract()]
//!     })
//!     .collect();
//! let k = Wavenumber::new(20.0, 0.0).unwrap();
//! let mut fmm = HelmholtzFmmBuilder::new(k, Degrees::Tolerance(1e-6))
//!     .gradients(true)
//!     .build::<f64, _>(&points, &points, &world)
//!     .unwrap();
//! let (re, im) = (vec![1.0; points.len()], vec![0.5; points.len()]);
//! let output = fmm.evaluate(&[Charges::Complex { re: &re, im: &im }]).unwrap();
//! let phi = output.potential_of(0);
//! println!("Re φ = {}, Im φ = {:?}", phi.re[0], phi.im.map(|im| im[0]));
//! ```

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use nd_fmm_math::Complex;
use nd_fmm_plan::evaluator::Evaluator;
use nd_fmm_plan::evaluator::EvaluatorError;
use nd_fmm_plan::index::BoxIndex;
use nd_fmm_plan::near::{LeafCounts, LeafPair, NearRule, VPair};
use nd_fmm_plan::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PerPair,
};
use nd_fmm_plan::plan::{Plan, PlanRules};
use nd_fmm_tables::helmholtz::DEFAULT_BLOCK_BUDGET;
use nd_fmm_tables::helmholtz::diagonal::{CLASS_COUNT, DEFAULT_SIGNATURE_NODE_BUDGET};
use nd_fmm_tables::helmholtz::plane_wave::merge::plane_wave_degree;
use nd_fmm_tables::helmholtz::plane_wave::quadrature::{RHO_MAX, assert_reach};
use nd_octree::constants::DEEPEST_LEVEL;
use nd_octree::octree::KeyType;
use nd_octree::{MortonKey, Octree, OctreeOptions, PhysicalBox, morton};
use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

use super::{
    ChargeKind, Charges, DEFAULT_DENSE_BUDGET, DEFAULT_DIAGONAL_BUILD, DIAGONAL_COST_CONSTANTS,
    DegreeError, Degrees, Densities, DensityError, DiagonalCosts, DiagonalShape, HelmholtzDiagonal,
    HelmholtzM2l, HelmholtzOperator, LevelShape, OperatorError, Output, PlaneWavePrecision,
    PlaneWaveRaise, PlaneWaveSplit, RcrSplit, TableTimings, TransferBuild, Wavenumber,
    WavenumberKind, auto_batched_max_degree, auto_batched_min_columns, auto_plane_wave_min_degree,
    diagonal_costs, min_leaf_level, offset_class,
};
use crate::fmm::{FmmError, PointSet, SettingsError};
use crate::geometry::{Domain, radius};
use crate::helmholtz::operator::{WaveSettings, wave_levels};
use crate::operator::{Isa, MulAdd, P2pChoice, SimdScalar};
use crate::pipeline::{
    self, Points, check_finite, gather_output, load_source_charges, pack_charges,
    unpack_output_many,
};
use crate::threading::REQUIRED_MPI_THREADING;

/// The deepest level of a Morton key, 16 (CONVENTIONS §3.12).
const DEEPEST: usize = DEEPEST_LEVEL as usize;

/// The default maximum leaf width in wavelengths of [`HelmholtzFmmBuilder::c_leaf`], 2
/// (Helmholtz CH4.0d3, signed off 2026-11-14, D11 amended; fmm-validate/results/helmholtz-ch4-0d3.md §4):
/// the leaves one level coarser than D11's [`C_LEAF`](super::C_LEAF) where the leaf rule
/// binds, which pays with the default leaf sizes (W1 at kD = 500, both near rules: build and
/// one evaluation 0.62, 0.71 and 0.78× c_leaf 1's at 256 points on Kathleen, 0.84, 0.84 and
/// 0.70× at NEON's 128, 128 and 256 points on locust, at 10⁻³, 10⁻⁶ and 10⁻⁸), the errors
/// within ε. The P2P's phase reaches Re(k) r ≤ 22 c_leaf, 44 c_leaf with converted V pairs.
/// D11 chose 1 as the largest c_leaf with no leaf on a high-frequency level in the census
/// (spikes/census/SPIKE_REPORT.md: at 2, 9 trees at 10⁻⁶); no high-frequency M2L runs yet
/// (CH4.2, CH4.3).
pub const DEFAULT_C_LEAF: f64 = 2.0;

/// The points per leaf of [`HelmholtzFmmBuilder`] when it names none (Helmholtz CH4.0d3;
/// fmm-validate/results/helmholtz-ch4-0d3.md), from the build plus one evaluation under the
/// W and X rule and the V rule of the cost model ([`WxNear::CostModel`],
/// [`VNear::CostModel`]) on CH3.6's matrix and the large cases at 10⁻³, 10⁻⁶ and 10⁻⁸: 256
/// on x86 (`isa` AVX-512, AVX2 or scalar; Kathleen: 0.64, 0.54 and 0.42× the 64 points with
/// both rules off, the geometric means); on NEON 128 at a tolerance of at least 10⁻⁷ or
/// named degrees, 256 below (locust: 0.84, 0.75 and 0.58×; 256 at 10⁻³ and 10⁻⁶ made the
/// evaluation alone slower than 128's, 1.10 and 0.66× against 0.89 and 0.56×). The rule does
/// not read the multipole planes: with one plane only NEON at 10⁻⁶ crossed the 5 % line (256
/// 5.7 % faster on build plus one evaluation over two cases, the evaluation 1 % slower). It
/// reads the tolerance and the ISA only (constraint 9); [`P2pChoice::Reference`] takes the
/// scalar kernel's class, x86's. Laplace's rule is
/// [`crate::fmm::default_max_points_per_leaf`].
///
/// ```
/// use nd_fmm_exec::helmholtz::default_max_points_per_leaf;
/// use nd_fmm_exec::operator::Isa;
///
/// assert_eq!(default_max_points_per_leaf(Some(1e-6), Isa::Avx512), 256);
/// assert_eq!(default_max_points_per_leaf(Some(1e-6), Isa::Neon), 128);
/// assert_eq!(default_max_points_per_leaf(Some(1e-8), Isa::Neon), 256);
/// assert_eq!(default_max_points_per_leaf(None, Isa::Neon), 128);
/// ```
pub fn default_max_points_per_leaf(tolerance: Option<f64>, isa: Isa) -> usize {
    match (isa, tolerance) {
        (Isa::Neon, Some(epsilon)) if epsilon < 1e-7 => 256,
        (Isa::Neon, _) => 128,
        _ => 256,
    }
}

/// The default bytes per node of the exponential M2L's waves buffer, 8 GiB
/// ([`HelmholtzFmmBuilder::plane_wave_budget`]; Helmholtz CH6.2a, signed off 2026-11-04).
pub const DEFAULT_PLANE_WAVE_NODE_BUDGET: usize = 8 << 30;

/// The default bytes per node of the Rotation M2L's contributions by pairs, 1 GiB
/// ([`HelmholtzFmmBuilder::rcr_budget`]; Helmholtz CH4.0i): on one rank per node level 2 of
/// W1 at kD = 500 (about 3,000 pairs of 279 KB at p = 131) in one chunk.
pub const DEFAULT_RCR_NODE_BUDGET: usize = 1 << 30;

/// The default bytes per node of the dense, class and batched M2L tables, 8 GiB
/// ([`HelmholtzFmmBuilder::dense_node_budget`]; Helmholtz CH4.0, proposal §2, constraint 7):
/// each rank's budget is the smaller of [`HelmholtzFmmBuilder::dense_budget`] and this over
/// the ranks per node.
pub const DEFAULT_DENSE_NODE_BUDGET: usize = 8 << 30;

/// The degrees of the m* table that `build` agrees over the ranks (Helmholtz CH4.0): 0 to
/// 64; a degree above it takes 64 columns, which no level reaches.
const MAX_TERM_DEGREE: usize = 64;

/// Why a Helmholtz FMM was not built or not evaluated.
#[derive(Debug, Error)]
pub enum HelmholtzFmmError {
    /// A step that the Helmholtz FMM shares with `Fmm` failed: the settings, the points,
    /// the domain, the plan, the redistribution, the evaluator or the thread pool.
    #[error(transparent)]
    Build(FmmError),
    /// c_leaf is not positive and finite.
    #[error("c_leaf must be positive and finite, got {0}")]
    LeafWidth(f64),
    /// The degrees could not be resolved.
    #[error("the degrees: {0}")]
    Degree(#[from] DegreeError),
    /// The operator refused k (Yukawa) or its inputs.
    #[error("the operator: {0}")]
    Operator(#[from] OperatorError),
    /// The charges of an evaluation do not fit the build.
    #[error("the charges: {0}")]
    Charges(#[from] DensityError),
    /// A named V crossover ([`VNear::Crossover`]) is negative or not a number.
    #[error("the V crossover must be at least 0, got {0}")]
    VCrossover(f64),
    /// A named W or X crossover ([`WxNear::Crossovers`]) is negative or not a number.
    #[error("the W and X crossovers must be at least 0, got {0} and {1}")]
    WxCrossover(f64, f64),
    /// The input is invalid on another rank.
    #[error("the input is invalid on another rank")]
    OtherRank,
}

/// Which V pairs between two leaves the plan makes near pairs (Helmholtz CH4.0d2;
/// [`HelmholtzFmmBuilder::near_v`]; `nd_fmm_plan::near::NearRule::with_v_pairs`): a V pair
/// (t, s) of two leaves converts where n_s n_t, the source points of s times the target
/// points of t, is below the crossover; its M2L is replaced by the exact P2P. On a level
/// whose M2L is the exponential one the decision is per parent P and source s instead
/// (merge and shift serves s to every child of P whose V list holds it, so a pair alone
/// would be counted twice): every such child converts if all of them are leaves and
/// n_s n_P is below the crossover times their number (n_P the target points under P;
/// the user's choice on 2026-11-13, "Group per (parent, source)").
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum VNear {
    /// No V pair converts (the default until CH4.0d2's sign-off).
    #[default]
    Off,
    /// The pairs with n_s n_t below this crossover on every level, for the measurements;
    /// `f64::INFINITY` converts every V pair between leaves.
    Crossover(f64),
    /// The crossover of each level from the cost model ([`v_crossover`]): the thread time a
    /// V pair's M2L takes on the level, under the cheapest M2L the setting may run there,
    /// over the P2P's per interaction.
    CostModel,
}

impl std::fmt::Display for VNear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Crossover(n) => write!(f, "{n}"),
            Self::CostModel => write!(f, "cost"),
        }
    }
}

/// Whether the V rule of [`VNear::CostModel`] is decided again after the plan (Helmholtz
/// CH4.3b2b; [`HelmholtzFmmBuilder::v_after_plan`]). Before the plan the builder knows the
/// levels whose M2L may be the exponential one ([`HelmholtzFmmBuilder::near_v`]), and sets
/// their crossover from the waves' per-pair constant, grouped per (parent, source); the time
/// rule ([`HelmholtzDiagonal::Auto`]) decides from the plan's V lists which of them keep the
/// waves. After the operator's choice, each level named below gets the crossover of the M2L
/// it runs ([`v_after_plan_crossover`]) where that is higher (and where the first plan still
/// holds a V pair between two leaves, on some rank), and the plan is built once more if any
/// crossover rose: it converts the first plan's pairs and those the higher crossover
/// adds (per pair on a level without waves, per (parent, source) on a level with them), so
/// its V lists are within the first plan's and the operator's M2L per level and buffers
/// hold. Opt-in, for CH4.3b2b's measurements and CH4.3b2d's trusted measure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VAfterPlan {
    /// The plan of the crossovers before it (the default).
    #[default]
    Off,
    /// The levels the time rule took from the waves
    /// ([`HelmholtzOperator::taken_from_waves`]): the diagonal form's pair, or RCR's where
    /// no class takes the form, at the level's degree.
    TakenLevels,
    /// Those, and the levels that keep the waves: the waves' level cost by the time rule's
    /// model over the level's pairs.
    AllLevels,
    /// The levels that keep the waves at a raised degree (p_w above the degree given;
    /// Helmholtz CH4.3b2d), alone: the waves' level cost over the level's pairs, as
    /// `AllLevels` prices them. The trusted measure of [`VRaisedPrice::PerBox`].
    RaisedLevels,
}

impl std::fmt::Display for VAfterPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::TakenLevels => write!(f, "taken"),
            Self::AllLevels => write!(f, "all"),
            Self::RaisedLevels => write!(f, "raised"),
        }
    }
}

/// How [`VNear::CostModel`] prices a V pair before the plan on a level whose exponential M2L
/// runs at a raised degree p_w above the degree given (ε < 10⁻⁶, CH6.1g's resolving degree;
/// Helmholtz CH4.3b2d; [`HelmholtzFmmBuilder::v_raised_price`]). Every other level keeps
/// CH4.0d2's crossover.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VRaisedPrice {
    /// The waves' per-pair constant at p_w, [`VCosts::plane_wave`] b (p_w + 1)² (CH4.0d2's).
    #[default]
    PerPair,
    /// The higher of that and the waves' cost per box with a V row at p_w
    /// ([`DiagonalCosts::plane_wave`]) over the V entries per leaf that the per-pair price
    /// leaves ([`raised_crossover`]), the leaves and entries counted from the octree's keys
    /// and the point counts before the plan ([`HelmholtzFmm::raised_v_shapes`]): on a sparse
    /// raised level the waves cost per box, so a pair saves more than the constant says
    /// (CH4.3b2b's report §6). Where the level's boxes with a V row are leaves this is the
    /// crossover that [`VAfterPlan::RaisedLevels`] finds after a first plan, in one plan.
    PerBox,
}

impl std::fmt::Display for VRaisedPrice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PerPair => write!(f, "pair"),
            Self::PerBox => write!(f, "box"),
        }
    }
}

/// The V lists of a level's leaves as counted before the plan (Helmholtz CH4.3b2d,
/// [`VRaisedPrice::PerBox`]), summed over the ranks: the leaves (each counted by the rank
/// that holds it as a local leaf) with a V entry left, and those entries, by the per-key
/// rule of `nd_fmm_plan::interaction_manager` less what the empty-box rule leaves out and
/// the pairs between leaves that the V rule converts at the per-pair crossover: the leaves'
/// V rows of that rule's plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeafVShape {
    /// The leaves with a V entry.
    pub leaves: u64,
    /// Their V entries.
    pub pairs: u64,
}

/// The M2L a level runs, as [`v_after_plan_crossover`] prices a V pair of it (Helmholtz
/// CH4.3b2b).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VLevelM2l {
    /// The exponential M2L: its cost per box with a V row, over the level's pairs.
    Waves,
    /// The diagonal form: its pair (on a level whose other classes run RCR too, the form's
    /// pair prices them as well: lower than RCR's, so fewer of them convert).
    Form,
    /// RCR (Rotation): its pair.
    Rcr,
}

/// Which W and X pairs between two leaves the plan makes near pairs (Helmholtz CH4.0d3;
/// [`HelmholtzFmmBuilder::near_wx`]; `nd_fmm_plan::near::NearRule`, Laplace's mechanism of
/// Phase 6 T12): a W pair (a leaf t and a finer leaf s of its W list) converts where s holds
/// fewer source points than the crossover of s's level, its M2P at t's points replaced by
/// the exact P2P; an X pair (a leaf t and a coarser leaf s of its X list) where t holds fewer
/// target points than the crossover of t's level, its P2L from s's points replaced by the
/// P2P. Interior W entries never convert. Each target point's work is then n_s P2P
/// interactions in place of one M2P (or each source point's n_t in place of one P2L).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum WxNear {
    /// No W or X pair converts.
    #[default]
    Off,
    /// The W pairs whose finer leaf holds fewer than `w` source points and the X pairs whose
    /// finer leaf holds fewer than `x` target points, on every level, for the measurements;
    /// `f64::INFINITY` converts every one.
    Crossovers {
        /// The crossover of the W pairs, in source points.
        w: f64,
        /// The crossover of the X pairs, in target points.
        x: f64,
    },
    /// The crossovers of each level from the cost model ([`wx_crossovers`]): the thread time
    /// of an M2P (P2L) per point at the finer leaf's degree over the P2P's per interaction.
    CostModel,
}

impl std::fmt::Display for WxNear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Crossovers { w, x } => write!(f, "w{w}x{x}"),
            Self::CostModel => write!(f, "cost"),
        }
    }
}

impl From<FmmError> for HelmholtzFmmError {
    /// [`FmmError::OtherRank`] as [`HelmholtzFmmError::OtherRank`], the rest as
    /// [`HelmholtzFmmError::Build`].
    fn from(error: FmmError) -> Self {
        match error {
            FmmError::OtherRank => Self::OtherRank,
            error => Self::Build(error),
        }
    }
}

/// The settings of a Helmholtz FMM ([module documentation](self)).
#[derive(Clone, Debug)]
pub struct HelmholtzFmmBuilder {
    k: Wavenumber,
    degrees: Degrees,
    charges: ChargeKind,
    vectors: usize,
    gradients: bool,
    threads: usize,
    domain: Option<[f64; 6]>,
    max_points_per_leaf: Option<usize>,
    max_level: usize,
    block_budget: usize,
    m2l: HelmholtzM2l,
    dense_budget: usize,
    dense_node_budget: usize,
    m2l_mul_add: Option<MulAdd>,
    m2l_max_degree: Option<usize>,
    batched_term: bool,
    m2l_block_targets: Option<usize>,
    auto_plane_wave: bool,
    m2l_plane_wave_min_degree: Option<usize>,
    plane_wave_precision: PlaneWavePrecision,
    plane_wave_reach: f64,
    plane_wave_budget: usize,
    plane_wave_split: PlaneWaveSplit,
    plane_wave_raise: PlaneWaveRaise,
    rcr_split: RcrSplit,
    rcr_budget: usize,
    diagonal: HelmholtzDiagonal,
    diagonal_build: TransferBuild,
    diagonal_costs: Option<DiagonalCosts>,
    v_costs: Option<VCosts>,
    diagonal_budget: usize,
    diagonal_chunk: Option<usize>,
    c_leaf: f64,
    p2p: P2pChoice,
    empty_boxes: bool,
    near_v: VNear,
    near_wx: WxNear,
    v_after_plan: VAfterPlan,
    v_raised_price: VRaisedPrice,
    pair_calls: bool,
}

impl HelmholtzFmmBuilder {
    /// The settings for wavenumber `k` with the degrees `degrees`: complex charges, one
    /// vector, potentials only, one thread, the domain from the points,
    /// [`default_max_points_per_leaf`] points per leaf, leaves down to level 16, the
    /// default block budget of the M2L y-blocks, M2L by the rule per level
    /// [`HelmholtzM2l::Auto`] within [`DEFAULT_DENSE_BUDGET`] per rank and
    /// [`DEFAULT_DENSE_NODE_BUDGET`] per node, the exponential M2L's waves
    /// within [`DEFAULT_PLANE_WAVE_NODE_BUDGET`] per node, the Rotation M2L by pairs on levels
    /// of high degree ([`RcrSplit::Auto`]) within [`DEFAULT_RCR_NODE_BUDGET`] per node, no
    /// diagonal M2L ([`HelmholtzDiagonal::Off`]; its scratch within
    /// [`DEFAULT_SIGNATURE_NODE_BUDGET`] per node and its transfer functions by
    /// [`DEFAULT_DIAGONAL_BUILD`] when named), the leaf width [`DEFAULT_C_LEAF`],
    /// the P2P kernel of [`P2pChoice::Auto`], the empty-box rule and the V, W and X pairs
    /// near pairs by the cost model.
    pub fn new(k: Wavenumber, degrees: Degrees) -> Self {
        Self {
            k,
            degrees,
            charges: ChargeKind::Complex,
            vectors: 1,
            gradients: false,
            threads: 1,
            domain: None,
            max_points_per_leaf: None,
            max_level: DEEPEST,
            block_budget: DEFAULT_BLOCK_BUDGET,
            m2l: HelmholtzM2l::Auto,
            dense_budget: DEFAULT_DENSE_BUDGET,
            dense_node_budget: DEFAULT_DENSE_NODE_BUDGET,
            m2l_mul_add: None,
            m2l_max_degree: None,
            batched_term: true,
            m2l_block_targets: None,
            auto_plane_wave: true,
            m2l_plane_wave_min_degree: None,
            plane_wave_precision: PlaneWavePrecision::Rule,
            plane_wave_reach: RHO_MAX,
            plane_wave_budget: DEFAULT_PLANE_WAVE_NODE_BUDGET,
            plane_wave_split: PlaneWaveSplit::Auto,
            plane_wave_raise: PlaneWaveRaise::Rule,
            rcr_split: RcrSplit::Auto,
            rcr_budget: DEFAULT_RCR_NODE_BUDGET,
            diagonal: HelmholtzDiagonal::Off,
            diagonal_build: DEFAULT_DIAGONAL_BUILD,
            diagonal_costs: None,
            v_costs: None,
            diagonal_budget: DEFAULT_SIGNATURE_NODE_BUDGET,
            diagonal_chunk: None,
            c_leaf: DEFAULT_C_LEAF,
            p2p: P2pChoice::Auto,
            empty_boxes: true,
            near_v: VNear::CostModel,
            near_wx: WxNear::CostModel,
            v_after_plan: VAfterPlan::Off,
            v_raised_price: VRaisedPrice::PerPair,
            pair_calls: false,
        }
    }

    /// Real or complex charges (CONVENTIONS §3.15's real-charge flag; default complex).
    pub fn charges(mut self, charges: ChargeKind) -> Self {
        self.charges = charges;
        self
    }

    /// The number m ≥ 1 of charge vectors of every evaluation (default 1). Vector v of the
    /// output is the one-vector FMM's of vector v, bit for bit.
    pub fn vectors(mut self, vectors: usize) -> Self {
        self.vectors = vectors;
        self
    }

    /// Whether the output holds the gradients (default no).
    pub fn gradients(mut self, gradients: bool) -> Self {
        self.gradients = gradients;
        self
    }

    /// The threads of the level calls, the charge load and the output pass (default 1).
    /// The output is the same at every thread count. With more than one, MPI must provide
    /// at least `Threading::Funneled`.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// The domain `[x_min, y_min, z_min, x_max, y_max, z_max]`, a cube that holds every
    /// point strictly inside, the same on every rank (default: the bounding box of every
    /// rank's points).
    pub fn domain(mut self, domain: PhysicalBox) -> Self {
        self.domain = Some(domain.coordinates());
        self
    }

    /// The octree's refinement target, points per leaf (default
    /// [`default_max_points_per_leaf`] of the tolerance and the P2P kernel's ISA, agreed over
    /// the ranks; [`DEFAULT_MAX_POINTS_PER_LEAF`](crate::fmm::DEFAULT_MAX_POINTS_PER_LEAF) before Helmholtz CH4.0d3). The leaf rule
    /// refines further where k needs it. [`HelmholtzFmm::max_points_per_leaf`] reports the
    /// target used.
    pub fn max_points_per_leaf(mut self, max_points_per_leaf: usize) -> Self {
        self.max_points_per_leaf = Some(max_points_per_leaf);
        self
    }

    /// The deepest leaf level, at most 16 (default 16).
    pub fn max_level(mut self, max_level: usize) -> Self {
        self.max_level = max_level;
        self
    }

    /// The bytes of the stored M2L y-blocks (`nd_fmm_tables::helmholtz::HelmholtzTables`;
    /// default [`DEFAULT_BLOCK_BUDGET`]); the output does not depend on it.
    pub fn block_budget(mut self, bytes: usize) -> Self {
        self.block_budget = bytes;
        self
    }

    /// The M2L of the levels (default [`HelmholtzM2l::Auto`] since Helmholtz CH3.2d, signed
    /// off on 2026-11-02; the operator module documentation, "M2L"). Under
    /// [`HelmholtzM2l::Dense`], [`HelmholtzM2l::Classes`], [`HelmholtzM2l::Batched`] and
    /// [`HelmholtzM2l::Auto`] the output agrees with Rotation's to rounding, not bit for bit
    /// (Auto's is Rotation's bit for bit on the levels it leaves to Rotation, and its levels
    /// with waves, by default, approximate M2L as PlaneWave's do); Batched's is
    /// Dense's bit for bit with real multipoles (real k, real charges). Under
    /// [`HelmholtzM2l::PlaneWave`] (opt-in) the levels with waves approximate M2L within their
    /// quadrature's error ([`plane_wave_precision`](Self::plane_wave_precision)); f64 only
    /// (untested in f32, not refused).
    pub fn m2l(mut self, choice: HelmholtzM2l) -> Self {
        self.m2l = choice;
        self
    }

    /// The bytes of the dense M2L tables of each rank under [`HelmholtzM2l::Dense`] and
    /// [`HelmholtzM2l::Batched`] (the same tables in split planes), and of the class tables
    /// with their transforms under [`HelmholtzM2l::Classes`] (default
    /// [`DEFAULT_DENSE_BUDGET`]), at most the node budget's share
    /// ([`dense_node_budget`](Self::dense_node_budget)): the levels that fit, from the finest
    /// level up, run the setting, the others RCR.
    pub fn dense_budget(mut self, bytes: usize) -> Self {
        self.dense_budget = bytes;
        self
    }

    /// The bytes of the tables of [`dense_budget`](Self::dense_budget) on one node (default
    /// [`DEFAULT_DENSE_NODE_BUDGET`]; Helmholtz CH4.0, proposal §2, constraint 7): each rank
    /// takes at most this over the largest number of ranks on one node, agreed at
    /// [`build`](Self::build), so the levels a rank's tables cover may depend on the ranks
    /// per node (as Laplace's node rule, Phase 5N T4); [`HelmholtzFmm::dense_budget`]
    /// reports the budget per rank. `usize::MAX`: the per-rank budget alone. Every rank holds
    /// its own copy of the tables, which its threads share: on a node of many cores, run a
    /// few ranks with threads each (on a 40-core node 4 × 10 keeps 1 GiB per rank under the
    /// default, where 40 × 1 leaves 205 MB a rank).
    pub fn dense_node_budget(mut self, bytes: usize) -> Self {
        self.dense_node_budget = bytes;
        self
    }

    /// The batched M2L's multiply–add mode (default `auto_m2l_mul_add` of the P2P kernel's
    /// ISA under [`HelmholtzM2l::Auto`], fused on every vector ISA, and [`MulAdd::Separate`]
    /// under [`HelmholtzM2l::Batched`]; read
    /// under these two only): [`MulAdd::Fused`] gives Dense's output to rounding also with
    /// real multipoles (`HelmholtzOperator::with_m2l_mul_add`).
    pub fn m2l_mul_add(mut self, mul_add: MulAdd) -> Self {
        self.m2l_mul_add = Some(mul_add);
        self
    }

    /// The crossover degree of [`HelmholtzM2l::Auto`]: Batched on the levels of degree at
    /// most `max_degree` that fit the dense budget, Rotation above (default the rule's,
    /// `auto_batched_max_degree` of the P2P kernel's ISA and the multipole planes). Either
    /// way [`build`](Self::build) agrees it over the ranks (the smallest), so every rank
    /// chooses the same levels; [`HelmholtzFmm::m2l_max_degree`] reports it.
    pub fn m2l_max_degree(mut self, max_degree: usize) -> Self {
        self.m2l_max_degree = Some(max_degree);
        self
    }

    /// Whether [`HelmholtzM2l::Auto`] reads the levels' V lists over the whole tree
    /// (Helmholtz CH4.0; default on): Batched only on the levels whose tables its level call
    /// would read for enough columns (`LevelShape::batched_columns` at least
    /// `auto_batched_min_columns`, the operator module documentation, "M2L"). `false` gives
    /// CH3.2d's rule, for the measurements. The counts are summed over the ranks at
    /// [`build`](Self::build) (one all-reduce), so every rank chooses alike;
    /// [`HelmholtzFmm::level_shapes`] reports them.
    pub fn batched_term(mut self, on: bool) -> Self {
        self.batched_term = on;
        self
    }

    /// Whether the plan leaves out the work of the boxes whose subtree holds no target or
    /// no source point (Helmholtz CH4.0d1, the empty-box rule of `nd_fmm_plan::empty`;
    /// default on). Only exact zeros go, so the output is the rule off's as values (−0 and
    /// +0 equal), and on P ranks the one-rank FMM's of the same setting bit for bit. The
    /// build counts every box's subtree from the points the routes deliver to each local
    /// leaf (an all-gather of the coarse blocks' counts and a ghost exchange of counts)
    /// before it builds the plan, once (Helmholtz CH4.0d1b); [`HelmholtzFmm::plan`]'s
    /// `empty_boxes` reports
    /// what it left out. Under [`HelmholtzM2l::Auto`] the levels' shapes
    /// ([`HelmholtzFmm::level_shapes`]) count the rule's V pairs, so a level's M2L may
    /// differ from the rule off's.
    pub fn empty_boxes(mut self, on: bool) -> Self {
        self.empty_boxes = on;
        self
    }

    /// Which V pairs between two leaves become near pairs (Helmholtz CH4.0d2, [`VNear`];
    /// default [`VNear::CostModel`] since Helmholtz CH4.0d3, [`VNear::Off`] before). The plan applies `nd_fmm_plan`'s near rule with V pairs on
    /// the counts the empty-box rule reads (the routes' points per local leaf, every held
    /// ghost leaf's from its owner; counted even with the empty-box rule off): each pair is
    /// decided on its target's rank from counts by key, so every rank count makes the
    /// one-rank rows, and the output on P ranks is the one-rank FMM's of the same setting
    /// bit for bit. A converted pair is exact (its M2L's truncation error gone), so the
    /// output is not the rule off's; the source exchange then delivers the converted ghost
    /// leaves' points too, and a multipole no row reads is no longer sent.
    /// [`HelmholtzFmm::plan`]'s `near_pairs` reports the pairs offered and converted. W and
    /// X pairs are [`near_wx`](Self::near_wx)'s.
    pub fn near_v(mut self, choice: VNear) -> Self {
        self.near_v = choice;
        self
    }

    /// Whether [`VNear::CostModel`]'s crossovers are decided again after the plan, on the
    /// levels [`VAfterPlan`] names (Helmholtz CH4.3b2b; default [`VAfterPlan::Off`]): the
    /// crossover of the M2L the operator runs there where it is higher, and a second plan if
    /// any rose ([`HelmholtzFmm::replanned`]; its time is
    /// [`HelmholtzBuildTimings::replan`]). The raised crossovers come from the agreed costs
    /// and V lists, so every rank decides alike, and the output on P ranks is the one-rank
    /// FMM's of the same setting bit for bit. Nothing changes under another [`VNear`].
    pub fn v_after_plan(mut self, choice: VAfterPlan) -> Self {
        self.v_after_plan = choice;
        self
    }

    /// How [`VNear::CostModel`] prices a V pair before the plan on a level whose waves run at
    /// a raised degree (Helmholtz CH4.3b2d, [`VRaisedPrice`]; default
    /// [`VRaisedPrice::PerPair`]). `PerBox` counts the raised levels' leaves and V entries
    /// from the octree's keys before the plan (one pass over the local leaves of those
    /// levels and one all-reduce; in [`HelmholtzBuildTimings::counts`]), so every rank
    /// decides alike and the output on P ranks is the one-rank FMM's of the same setting bit
    /// for bit. One plan; nothing changes under another [`VNear`] or without a raised level.
    pub fn v_raised_price(mut self, price: VRaisedPrice) -> Self {
        self.v_raised_price = price;
        self
    }

    /// The cost model of the near rules' crossovers ([`VNear::CostModel`], the planned and the
    /// after-plan crossovers, and [`WxNear::CostModel`]'s P2P per interaction; Helmholtz
    /// CH4.3b2b; default `None`: [`v_costs`] of the P2P kernel's ISA), for the tests, whose
    /// plans then do not depend on the machine class. The crossovers it gives are agreed over
    /// the ranks as the default's.
    pub fn v_costs(mut self, costs: Option<VCosts>) -> Self {
        self.v_costs = costs;
        self
    }

    /// Which W and X pairs between two leaves become near pairs (Helmholtz CH4.0d3,
    /// [`WxNear`]; default [`WxNear::CostModel`]), by the same near rule and counts as
    /// [`near_v`](Self::near_v), in the same plan: each pair decided on its target's rank
    /// from counts by key and crossovers agreed over the ranks, so the output on P ranks is
    /// the one-rank FMM's of the same setting bit for bit. A converted pair is exact, so the
    /// output is not the rule off's. [`HelmholtzFmm::plan`]'s `near_pairs` reports the W and
    /// X pairs offered and converted.
    pub fn near_wx(mut self, choice: WxNear) -> Self {
        self.near_wx = choice;
        self
    }

    /// The width of Batched's blocks in its level call (default: by the level's boxes and
    /// the threads, `batched_block_targets`; Helmholtz CH4.0), for the measurements and the
    /// tests (0 is taken as 1). The output does not depend on it.
    pub fn m2l_block_targets(mut self, targets: usize) -> Self {
        self.m2l_block_targets = Some(targets.max(1));
        self
    }

    /// Lets [`HelmholtzM2l::Auto`] take the exponential M2L (Helmholtz CH6.2h, decision D8;
    /// default on since its sign-off, 2026-11-08; `false` gives CH3.2d's rule): on the levels
    /// of degree at least
    /// [`m2l_plane_wave_min_degree`](Self::m2l_plane_wave_min_degree) to which the precision
    /// rule gives a quadrature (from the tolerance of [`Degrees::Tolerance`]: iprec 2 at
    /// ε ≥ 10⁻⁶, and since Helmholtz CH6.1g iprec 3 at the raised degree for 10⁻⁸ ≤ ε < 10⁻⁷,
    /// since Helmholtz CH6.1h's sign-off for 10⁻⁷ ≤ ε < 10⁻⁶ too), with Im κ_l below
    /// [`AUTO_PLANE_WAVE_MAX_IM_KAPPA`](super::AUTO_PLANE_WAVE_MAX_IM_KAPPA), in f64 only;
    /// Batched and Rotation on the others as before
    /// (`HelmholtzOperator::with_auto_plane_wave`). With [`Degrees::PerLevel`] the quadrature
    /// is each degree's, as under [`HelmholtzM2l::PlaneWave`] (iprec 2 to degree 15, none
    /// above). Those levels approximate M2L within their quadrature's error, as under
    /// [`HelmholtzM2l::PlaneWave`].
    pub fn auto_plane_wave(mut self, on: bool) -> Self {
        self.auto_plane_wave = on;
        self
    }

    /// The smallest degree at which [`HelmholtzM2l::Auto`] takes the exponential M2L under
    /// [`auto_plane_wave`](Self::auto_plane_wave) (default the rule's,
    /// `auto_plane_wave_min_degree` of the P2P kernel's ISA and the multipole planes). Either
    /// way [`build`](Self::build) agrees it over the ranks (the smallest);
    /// [`HelmholtzFmm::m2l_plane_wave_min_degree`] reports it.
    pub fn m2l_plane_wave_min_degree(mut self, min_degree: usize) -> Self {
        self.m2l_plane_wave_min_degree = Some(min_degree);
        self
    }

    /// The precision of each level's quadrature under [`HelmholtzM2l::PlaneWave`] (default
    /// [`PlaneWavePrecision::Rule`]; `HelmholtzOperator::with_plane_wave_precision`). The
    /// rule reads the tolerance of [`Degrees::Tolerance`], or with named degrees each
    /// level's degree (`HelmholtzOperator::with_plane_wave_tolerance`); it is a function of
    /// k, the domain, the degrees and the tolerance, so every rank chooses alike. Below
    /// ε = 10⁻⁶ the rule raises the degree of each level with a quadrature
    /// (`nd_fmm_tables::helmholtz::plane_wave::merge::plane_wave_degree`; Helmholtz CH6.1g);
    /// named precisions keep the degrees. A list named here must be the same on every rank.
    pub fn plane_wave_precision(mut self, precision: PlaneWavePrecision) -> Self {
        self.plane_wave_precision = precision;
        self
    }

    /// The lateral reach of the quadratures' α counts under [`HelmholtzM2l::PlaneWave`]
    /// (default the region's corner 4√2; `HelmholtzOperator::with_plane_wave_reach`), for
    /// the measurements (Helmholtz CH6.2e). The same on every rank.
    ///
    /// # Panics
    ///
    /// If `reach` is not finite or not above the source's reach √2/2.
    pub fn plane_wave_reach(mut self, reach: f64) -> Self {
        assert_reach(reach);
        self.plane_wave_reach = reach;
        self
    }

    /// How the exponential M2L's merge and shift splits over the threads (default
    /// [`PlaneWaveSplit::Auto`]; `HelmholtzOperator::with_plane_wave_split`), for the
    /// measurements (Helmholtz CH6.2f). Every setting gives the same output bit for bit.
    pub fn plane_wave_split(mut self, split: PlaneWaveSplit) -> Self {
        self.plane_wave_split = split;
        self
    }

    /// Which operators of a level the rule raises to the resolving degree run at it (default
    /// [`PlaneWaveRaise::Rule`]: the M2L alone at 10⁻⁷ ≤ ε < 10⁻⁶, also the leaves at
    /// 10⁻⁸ ≤ ε < 10⁻⁷; `HelmholtzOperator::with_plane_wave_raise`; Helmholtz CH6.1h, signed
    /// off 2026-11-11). [`PlaneWaveRaise::Level`] raises every operator (CH6.1g's form),
    /// [`PlaneWaveRaise::Leaves`] all but the translations to and from the parent,
    /// [`PlaneWaveRaise::M2l`] the M2L alone. The same on every rank.
    pub fn plane_wave_raise(mut self, raise: PlaneWaveRaise) -> Self {
        self.plane_wave_raise = raise;
        self
    }

    /// The bytes per node of the exponential M2L's waves buffer under
    /// [`HelmholtzM2l::PlaneWave`] (default [`DEFAULT_PLANE_WAVE_NODE_BUDGET`]; proposal §2,
    /// constraint 7): each rank's bound is the budget over the ranks per node, agreed in
    /// [`build`](Self::build) (the largest node's), at least [`MIN_BUFFER_BOXES`](super::MIN_BUFFER_BOXES) boxes of
    /// a level (`HelmholtzOperator::with_plane_wave_buffer`); `usize::MAX` holds every box
    /// of a level. The level call then runs by chunks of parents, and the output is the same
    /// at every budget; [`HelmholtzFmm::ranks_per_node`] and the operator's
    /// `plane_wave_buffer` report it.
    pub fn plane_wave_budget(mut self, bytes: usize) -> Self {
        self.plane_wave_budget = bytes;
        self
    }

    /// How the Rotation (RCR) M2L's level calls split over the threads (default
    /// [`RcrSplit::Auto`]: by pairs on levels of high degree, and of middle degree with few
    /// targets per thread, [`rcr_by_pairs`](super::rcr_by_pairs);
    /// `HelmholtzOperator::with_rcr_split`; Helmholtz CH4.0i). Every setting gives the same
    /// output bit for bit.
    pub fn rcr_split(mut self, split: RcrSplit) -> Self {
        self.rcr_split = split;
        self
    }

    /// The bytes per node of the Rotation M2L's contributions by pairs (default
    /// [`DEFAULT_RCR_NODE_BUDGET`]; proposal §2, constraint 7; Helmholtz CH4.0i): each rank's
    /// bound is the budget over the ranks per node, agreed in [`build`](Self::build), at
    /// least one pair per thread (`HelmholtzOperator::with_rcr_buffer`); `usize::MAX` holds
    /// every pair of a level. A level call by pairs then runs by chunks of pairs, and the
    /// output is the same at every budget.
    pub fn rcr_budget(mut self, bytes: usize) -> Self {
        self.rcr_budget = bytes;
        self
    }

    /// The diagonal M2L of the high-frequency levels (default [`HelmholtzDiagonal::Off`];
    /// Helmholtz CH4.2b1, opt-in until CH4.3; `HelmholtzOperator::with_diagonal`): on the
    /// levels whose M2L is RCR's, each offset class with E(l, c) ≤ θ ε on the diagonal form,
    /// RCR on the others. [`HelmholtzDiagonal::Rule`] reads the tolerance of
    /// [`Degrees::Tolerance`] (no level with named degrees); [`HelmholtzDiagonal::Auto`] (Helmholtz
    /// CH4.3b1) too, and takes a level only where the time rule's model puts it below the M2L
    /// the level would run otherwise, RCR's or, under [`HelmholtzM2l::Auto`], the exponential
    /// M2L's ([`diagonal_costs`](Self::diagonal_costs)). The output is deterministic and on
    /// P ranks the one-rank FMM's bit for bit.
    ///
    /// # Panics
    ///
    /// If [`HelmholtzDiagonal::Tolerance`] names ε ≤ 0 or NaN.
    pub fn diagonal(mut self, choice: HelmholtzDiagonal) -> Self {
        if let HelmholtzDiagonal::Tolerance(epsilon) = choice {
            assert!(
                epsilon > 0.0,
                "the diagonal form's tolerance {epsilon} is not positive"
            );
        }
        self.diagonal = choice;
        self
    }

    /// The build of the diagonal M2L's transfer functions (default [`DEFAULT_DIAGONAL_BUILD`],
    /// [`TransferBuild::Interpolated`]; Helmholtz CH4.2c2; `HelmholtzOperator::with_diagonal_build`):
    /// interpolated in ŝ · t̂ from a double-double table, faster to build (4.0–6.4× per W1 run on
    /// Kathleen) with the bound's C_T = 0.23, or [`TransferBuild::Direct`] per sample in
    /// double-double with C_T = 0.15, so a class may pass at a tighter tolerance. The rule reads
    /// the build's constants.
    pub fn diagonal_build(mut self, build: TransferBuild) -> Self {
        self.diagonal_build = build;
        self
    }

    /// The costs of the diagonal form's time rule ([`HelmholtzDiagonal::Auto`]; Helmholtz
    /// CH4.3b1; default `None`: [`diagonal_costs`] of the P2P kernel's
    /// ISA), agreed over the ranks (the largest of each constant), for the measurements and
    /// the tests.
    pub fn diagonal_costs(mut self, costs: Option<DiagonalCosts>) -> Self {
        self.diagonal_costs = costs;
        self
    }

    /// The bytes per node of a diagonal level call's scratch (default
    /// [`DEFAULT_SIGNATURE_NODE_BUDGET`]; proposal §2, constraint 7; Helmholtz CH4.2b1): each
    /// rank's share is the budget over the ranks per node, and a level takes the diagonal form
    /// only where its scratch at the largest count of its V sources over the ranks fits that
    /// share, else RCR, the same on every rank (`HelmholtzOperator::with_diagonal_budget`);
    /// `usize::MAX` for no bound.
    pub fn diagonal_budget(mut self, bytes: usize) -> Self {
        self.diagonal_budget = bytes;
        self
    }

    /// The width of the diagonal level call's chunks of targets (`None`, the default: by the
    /// level's boxes and the threads; `HelmholtzOperator::with_diagonal_chunk`), for the
    /// tests: no bit depends on it.
    ///
    /// # Panics
    ///
    /// If the width is 0.
    #[doc(hidden)]
    pub fn diagonal_chunk(mut self, chunk: Option<usize>) -> Self {
        assert_ne!(chunk, Some(0), "a chunk holds at least one target");
        self.diagonal_chunk = chunk;
        self
    }

    /// The maximum leaf width in wavelengths (default [`DEFAULT_C_LEAF`] since Helmholtz
    /// CH4.0d3; D11's [`C_LEAF`](super::C_LEAF) before).
    pub fn c_leaf(mut self, c_leaf: f64) -> Self {
        self.c_leaf = c_leaf;
        self
    }

    /// The P2P kernel (default [`P2pChoice::Auto`]; the module documentation, "P2P
    /// kernel"). [`build`](Self::build) refuses [`P2pChoice::Isa`] with an ISA this machine
    /// cannot run, [`SettingsError::P2pIsaUnavailable`]; the check is local, and agreed by
    /// every rank, the others returning [`HelmholtzFmmError::OtherRank`].
    pub fn p2p_kernel(mut self, choice: P2pChoice) -> Self {
        self.p2p = choice;
        self
    }

    /// For tests: run the operator pair by pair (`nd_fmm_plan::operator::PerPair`) in place
    /// of its level calls. The output is the same bit for bit.
    #[doc(hidden)]
    pub fn pair_calls(mut self, on: bool) -> Self {
        self.pair_calls = on;
        self
    }

    /// The settings' own errors, checked on every rank before any collective.
    fn check_settings(&self) -> Result<(), HelmholtzFmmError> {
        if self.k.kind() == WavenumberKind::Yukawa {
            return Err(OperatorError::Yukawa.into());
        }
        let settings = if self.max_level > DEEPEST {
            Some(SettingsError::MaxLevelTooDeep {
                max_level: self.max_level,
            })
        } else if self.max_points_per_leaf == Some(0) {
            Some(SettingsError::ZeroPointsPerLeaf)
        } else if self.threads == 0 {
            Some(SettingsError::ZeroThreads)
        } else if self.vectors == 0 {
            Some(SettingsError::ZeroVectors)
        } else if let Err(error) = self.p2p.resolve() {
            Some(SettingsError::P2pIsaUnavailable { isa: error.isa })
        } else {
            None
        };
        if let Some(error) = settings {
            return Err(FmmError::InvalidSettings(error).into());
        }
        if !(self.c_leaf > 0.0 && self.c_leaf.is_finite()) {
            return Err(HelmholtzFmmError::LeafWidth(self.c_leaf));
        }
        if let VNear::Crossover(n) = self.near_v
            && (n.is_nan() || n < 0.0)
        {
            return Err(HelmholtzFmmError::VCrossover(n));
        }
        if let WxNear::Crossovers { w, x } = self.near_wx
            && (w.is_nan() || w < 0.0 || x.is_nan() || x < 0.0)
        {
            return Err(HelmholtzFmmError::WxCrossover(w, x));
        }
        if let Degrees::Tolerance(_) = self.degrees {
            // The tolerance's own check; the levels come with the plan.
            self.degrees.resolve(&self.k, &unit_domain(), 0)?;
        }
        Ok(())
    }

    /// With more than one thread: checks that MPI provides `provided` ≥
    /// [`REQUIRED_MPI_THREADING`] and builds the pool.
    fn pool(&self, provided: Threading) -> Result<Option<Arc<ThreadPool>>, FmmError> {
        if self.threads == 1 {
            return Ok(None);
        }
        if provided < REQUIRED_MPI_THREADING {
            return Err(FmmError::MpiThreading {
                required: REQUIRED_MPI_THREADING,
                provided,
            });
        }
        ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .thread_name(|i| format!("nd-fmm-exec-helmholtz-{i}"))
            .build()
            .map(|pool| Some(Arc::new(pool)))
            .map_err(|error| FmmError::ThreadPool(error.to_string()))
    }

    /// Builds the FMM of `sources` and `targets`, in the caller's order ([module
    /// documentation](self#build)). The two sets may overlap or coincide; a rank may pass
    /// no points.
    ///
    /// # Collective operation
    ///
    /// Every rank of `comm` must call it, with the same settings.
    ///
    /// # Errors
    ///
    /// The settings' errors ([`HelmholtzFmmError::Operator`] for Yukawa,
    /// [`HelmholtzFmmError::LeafWidth`], [`HelmholtzFmmError::Degree`],
    /// [`HelmholtzFmmError::Build`] with a `SettingsError`), a point that is not finite or
    /// outside the supplied domain, no point on any rank, and the errors of the shared
    /// steps; the rank that found an input error returns it, the others
    /// [`HelmholtzFmmError::OtherRank`].
    pub fn build<'o, T, C>(
        &self,
        sources: &[[f64; 3]],
        targets: &[[f64; 3]],
        comm: &'o C,
    ) -> Result<HelmholtzFmm<'o, T, C>, HelmholtzFmmError>
    where
        T: SimdScalar + Equivalence + Default,
        C: CommunicatorCollectives,
    {
        let mut timings = HelmholtzBuildTimings::default();
        let mut clock = Instant::now();
        // Step 1: settings, MPI threading and the pool, finite points, the supplied
        // domain; one agreement.
        let provided = mpi::environment::threading_support();
        let local = self.check_settings().and_then(|()| {
            let pool = self.pool(provided)?;
            check_finite(sources, PointSet::Sources)?;
            check_finite(targets, PointSet::Targets)?;
            let supplied = self.domain.map(|c| Domain::new(&PhysicalBox::new(c)));
            let supplied = supplied.transpose().map_err(FmmError::from)?;
            Ok((pool, supplied))
        });
        let (pool, supplied) = agree(comm, local)?;
        // The node (proposal §2, constraint 7): one split of the communicator by node, on
        // every rank, whose size divides the waves budget of the exponential M2L (step 7).
        let node = comm.split_shared(comm.rank());

        // Step 2: are there points at all? The same all-reduce agrees the largest number of
        // ranks per node and the points per leaf (the rule's differ only where the ranks'
        // ISAs do; constraints 2 and 9).
        let tolerance = match self.degrees {
            Degrees::Tolerance(epsilon) => Some(epsilon),
            Degrees::PerLevel(_) => None,
        };
        let isa = match self.p2p.resolve() {
            Ok(P2pChoice::Isa(isa)) => isa,
            _ => Isa::Scalar,
        };
        let max_points_per_leaf = self
            .max_points_per_leaf
            .unwrap_or_else(|| default_max_points_per_leaf(tolerance, isa));
        let local = [
            (sources.len() + targets.len()) as u64,
            node.size() as u64,
            max_points_per_leaf as u64,
        ];
        let mut agreed = [0u64; 3];
        comm.all_reduce_into(&local[..], &mut agreed[..], SystemOperation::max());
        if agreed[0] == 0 {
            return Err(FmmError::NoPoints.into());
        }
        let ranks_per_node = agreed[1].max(1) as usize;
        let max_points_per_leaf = agreed[2] as usize;
        drop(node);

        timings.settings = lap(&mut clock);

        // Step 3: the domain, and every point strictly inside it.
        let (coordinates, domain) = pipeline::domain(sources, targets, supplied, comm)?;
        timings.domain = lap(&mut clock);

        // Step 4: the octree with the leaf rule, and its plan.
        let leaf_level = min_leaf_level(&self.k, &domain, self.c_leaf);
        let options = OctreeOptions::new()
            .with_max_level(self.max_level)
            .with_max_fine_keys(max_points_per_leaf)
            .with_ghost_children(true)
            .with_min_leaf_level(leaf_level);
        let (keys, octree) = pipeline::octree(&coordinates, &domain, options, comm);
        let index = BoxIndex::of_octree(&octree).map_err(FmmError::Plan)?;
        timings.octree = lap(&mut clock);

        // Steps 5 and 6: the points to their owners, routed on the index's local leaves.
        // Three coordinates per point move now, and on every evaluation c m strengths per
        // source and the output values per target.
        let densities = Densities::new(self.k.kind(), self.charges, self.vectors);
        let output_values = densities.target_output_point_size(self.gradients);
        let per_item = output_values.max(3).max(densities.columns());
        let (points, owned) = Points::route(
            &octree,
            &index,
            (&keys, sources.len()),
            &coordinates,
            &domain,
            per_item,
        )?;
        drop(keys);
        drop(coordinates);
        timings.routes = lap(&mut clock);
        // Step 7 (its first part before the plan, Helmholtz CH4.0d2): the degrees and the
        // M2L's agreed settings, which the V rule reads. The largest number of sources in a
        // leaf, over every rank, sizes P2P's scratch.
        let local = points
            .source_leaves
            .counts
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let mut max_leaf_points = 0usize;
        comm.all_reduce_into(&local, &mut max_leaf_points, SystemOperation::max());
        let degrees = agree(
            comm,
            self.degrees
                .resolve(&self.k, &domain, index.nlevels())
                .map_err(HelmholtzFmmError::from),
        )?;
        // Auto's crossover, agreed over the ranks (constraints 2 and 9): the smallest of
        // every rank's, which differ only where the ranks' ISAs do.
        let local = self
            .m2l_max_degree
            .unwrap_or_else(|| auto_batched_max_degree(isa, densities.multipole_planes()))
            as u64;
        let mut max_degree = 0u64;
        comm.all_reduce_into(&local, &mut max_degree, SystemOperation::min());
        // And the smallest degree of Auto's exponential M2L, agreed alike (Helmholtz CH6.2h).
        let local = self
            .m2l_plane_wave_min_degree
            .unwrap_or_else(|| auto_plane_wave_min_degree(isa, densities.multipole_planes()))
            as u64;
        let mut plane_wave_min_degree = 0u64;
        comm.all_reduce_into(&local, &mut plane_wave_min_degree, SystemOperation::min());
        // The levels whose M2L is the exponential one: the operator's choice from the same
        // settings and degrees (`wave_levels`, Helmholtz CHA.11), less, under the diagonal
        // form's time rule, those it takes after the plan (Helmholtz CH4.3b1). There merge and
        // shift serves a parent's sources to its children by geometric masks, so the V rule
        // decides per parent and source (`v_group`), which converts exact pairs under every
        // M2L, so a level the form takes keeps an exact plan.
        let k_root = Complex::new(
            self.k.re() * radius(0, &domain),
            self.k.im() * radius(0, &domain),
        );
        // The diagonal form's time rule's costs (Helmholtz CH4.3b1), agreed alike (the
        // largest of each constant).
        let local = self
            .diagonal_costs
            .unwrap_or_else(|| diagonal_costs(isa))
            .to_array();
        let mut agreed = [0.0f64; DIAGONAL_COST_CONSTANTS];
        comm.all_reduce_into(&local[..], &mut agreed[..], SystemOperation::max());
        let costs = DiagonalCosts::from_array(agreed);
        let waves: Vec<bool> = match wave_levels::<T>(
            &WaveSettings {
                m2l: self.m2l,
                precision: &self.plane_wave_precision,
                auto_plane_wave: self.auto_plane_wave,
                min_degree: usize::try_from(plane_wave_min_degree).unwrap_or(usize::MAX),
                tolerance,
            },
            k_root,
            &degrees,
        ) {
            Some(levels) => levels.iter().map(Option::is_some).collect(),
            None => vec![false; degrees.len()],
        };
        // The plan, once, on the index: with the empty-box rule and the V pairs' near rule
        // (Helmholtz CH4.0d2) from the routes' points per local leaf. Every plan of the
        // octree has the index's local leaves, so the routes and the leaf ranges hold for it.
        let counts = if self.empty_boxes || self.near_v != VNear::Off || self.near_wx != WxNear::Off
        {
            Some(
                LeafCounts::exchange_with_boxes(
                    &octree,
                    &index,
                    &points.source_leaves.counts,
                    &points.target_leaves.counts,
                )
                .map_err(|error| FmmError::Evaluator(EvaluatorError::Exchange(error)))?,
            )
        } else {
            None
        };
        // The W and X crossovers of each level (Helmholtz CH4.0d3), by the finer leaf's
        // level: named, or the cost model's at the level's degree; agreed over the ranks.
        // The near rules' cost model: named, or the machine class's.
        let v_cost_model = self.v_costs.unwrap_or_else(|| v_costs(isa));
        // The degrees are those given, before the exponential M2L's raise: where a raised
        // level runs M2P and P2L at p_w (`PlaneWaveRaise::Level`, `Leaves`), the crossovers
        // are low, which converts fewer pairs, never a wrong one.
        let local: Vec<f64> = (0..degrees.len())
            .flat_map(|l| match self.near_wx {
                WxNear::Off => [0.0, 0.0],
                WxNear::Crossovers { w, x } => [w, x],
                WxNear::CostModel => wx_crossovers(&wx_costs(isa), &v_cost_model, degrees[l]),
            })
            .collect();
        let mut wx = vec![0.0f64; local.len()];
        comm.all_reduce_into(&local[..], &mut wx[..], SystemOperation::min());
        let leaf_pairs = |pair: LeafPair| match pair {
            LeafPair::W {
                source, sources, ..
            } => (sources as f64) < wx[2 * morton::level(source)],
            LeafPair::X {
                target, targets, ..
            } => (targets as f64) < wx[2 * morton::level(target) + 1],
        };
        // The degree of each level's waves where predicted (p_w, CH6.1g; the degree given
        // elsewhere).
        let wave_degree = |l: usize| match tolerance {
            Some(epsilon) if waves[l] => plane_wave_degree(degrees[l], epsilon),
            _ => degrees[l],
        };
        // The levels whose waves are raised (p_w above the degree given), and those that
        // `VRaisedPrice::PerBox` prices per box (Helmholtz CH4.3b2d).
        let raised_waves: Vec<bool> = (0..degrees.len())
            .map(|l| waves[l] && wave_degree(l) > degrees[l])
            .collect();
        let raised_levels: Vec<bool> = raised_waves
            .iter()
            .map(|&raised| {
                raised
                    && self.near_v == VNear::CostModel
                    && self.v_raised_price == VRaisedPrice::PerBox
            })
            .collect();
        // The crossover of each level: named, or the cost model's for the cheapest M2L the
        // setting may run there (Auto's choice after the plan reads the pairs left, so a
        // dearer one only means that more pairs would have paid).
        let crossovers: Vec<f64> = (0..degrees.len())
            .map(|l| match self.near_v {
                VNear::Off => 0.0,
                VNear::Crossover(n) => n,
                VNear::CostModel => {
                    let p = degrees[l];
                    let (m2l, p) = if waves[l] {
                        (HelmholtzM2l::PlaneWave, wave_degree(l))
                    } else {
                        let cheapest = match self.m2l {
                            HelmholtzM2l::Auto if p <= max_degree as usize => HelmholtzM2l::Batched,
                            HelmholtzM2l::Auto | HelmholtzM2l::PlaneWave => HelmholtzM2l::Rotation,
                            m2l => m2l,
                        };
                        (cheapest, p)
                    };
                    v_crossover(&v_cost_model, m2l, p, self.gradients)
                }
            })
            .collect();
        // Agreed over the ranks (the smallest), so that ranks of different ISAs decide alike
        // (constraints 2 and 9).
        let mut agreed = vec![0.0f64; crossovers.len()];
        comm.all_reduce_into(&crossovers[..], &mut agreed[..], SystemOperation::min());
        let crossovers = agreed;
        // Under `PerBox`, each raised level's leaves and V entries that the per-pair crossover
        // leaves (its V rule's decision per (parent, source), as the plan would take it),
        // counted from the keys and counts and summed over the ranks; then the level's
        // crossover the higher of the per-pair one and the waves' cost per box over those
        // entries per leaf, agreed alike. The settings and the waves are every rank's, so
        // every rank takes part (constraints 2 and 9).
        let raised_v_shapes = if raised_levels.contains(&true) {
            let parent = Cell::new(None);
            let first = |pair: VPair| {
                counts.as_ref().is_some_and(|counts| {
                    v_group(
                        pair,
                        crossovers[morton::level(pair.target)],
                        (octree.all_keys(), counts),
                        &parent,
                    )
                })
            };
            let local = leaf_v_shapes(
                octree.all_keys(),
                counts.as_ref(),
                self.empty_boxes,
                &raised_levels,
                &first,
            );
            let local: Vec<u64> = local.iter().flat_map(|s| [s.leaves, s.pairs]).collect();
            let mut summed = vec![0u64; local.len()];
            comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
            Some(
                summed
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| LeafVShape {
                        leaves: c[0],
                        pairs: c[1],
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let crossovers = match &raised_v_shapes {
            Some(shapes) => {
                let local: Vec<f64> = (0..crossovers.len())
                    .map(|l| {
                        if raised_levels[l] {
                            crossovers[l].max(raised_crossover(
                                &v_cost_model,
                                &costs,
                                wave_degree(l),
                                shapes[l],
                                self.gradients,
                            ))
                        } else {
                            crossovers[l]
                        }
                    })
                    .collect();
                let mut agreed = vec![0.0f64; local.len()];
                comm.all_reduce_into(&local[..], &mut agreed[..], SystemOperation::min());
                agreed
            }
            None => crossovers,
        };
        let all_keys = octree.all_keys();
        let parent = Cell::new(None);
        let v_pairs = |pair: VPair| {
            let level = morton::level(pair.target);
            if waves[level] {
                counts.as_ref().is_some_and(|counts| {
                    v_group(pair, crossovers[level], (all_keys, counts), &parent)
                })
            } else {
                ((pair.sources * pair.targets) as f64) < crossovers[level]
            }
        };
        let near = match (&counts, self.near_v, self.near_wx) {
            (Some(counts), VNear::Crossover(_) | VNear::CostModel, _) => {
                Some(NearRule::new(counts, &leaf_pairs).with_v_pairs(&v_pairs))
            }
            (Some(counts), VNear::Off, WxNear::Crossovers { .. } | WxNear::CostModel) => {
                Some(NearRule::new(counts, &leaf_pairs))
            }
            _ => None,
        };
        let rules = PlanRules {
            near: near.as_ref(),
            empty_boxes: counts.as_ref().filter(|_| self.empty_boxes),
        };
        timings.counts = lap(&mut clock);
        let plan = Plan::with_rules_on(&octree, index, &rules).map_err(FmmError::Plan)?;
        timings.plan = lap(&mut clock);

        // Auto's term for Batched (Helmholtz CH4.0): each level's V pairs and boxes, each
        // box counted by the rank that computes it, summed over the ranks.
        let shapes = if self.batched_term {
            let nlevels = plan.nlevels();
            let mut local = vec![0u64; 2 * nlevels];
            for (l, counts) in local.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let v = plan.level(l).v();
                for t in (0..v.nrows()).filter(|&t| plan.computes(l, t)) {
                    counts[0] += v.row(t).0.len() as u64;
                    counts[1] += 1;
                }
            }
            let mut summed = vec![0u64; 2 * nlevels];
            comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
            Some(
                summed
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| LevelShape {
                        pairs: c[0],
                        targets: c[1],
                    })
                    .collect(),
            )
        } else {
            None
        };
        // And m* by degree, agreed alike (the largest), so that ranks of different ISAs
        // choose the same levels (constraints 2 and 9).
        let local: Vec<f64> = (0..=MAX_TERM_DEGREE)
            .map(|p| auto_batched_min_columns(isa, densities.multipole_planes(), p))
            .collect();
        let mut min_columns = vec![0.0f64; local.len()];
        comm.all_reduce_into(&local[..], &mut min_columns[..], SystemOperation::max());
        // The dense budget of this rank: its own, at most the node's over the ranks per node
        // (constraint 7).
        let dense_budget = self
            .dense_budget
            .min(self.dense_node_budget / ranks_per_node);
        // Under `PlaneWave` the level call's buffers, for each level's boxes, own and ghost,
        // and the waves within the node budget's share of this rank.
        let level_boxes: Vec<usize> = (0..plan.index().nlevels())
            .map(|level| plan.index().len(level))
            .collect();
        // The Rotation M2L's buffer by pairs, for each level's V pairs and targets with a V
        // row on this rank (Helmholtz CH4.0i).
        let rcr_levels: Vec<(usize, usize)> = (0..plan.nlevels())
            .map(|l| {
                let v = plan.level(l).v();
                let targets = v.row_offsets().windows(2).filter(|w| w[1] > w[0]).count();
                (v.len(), targets)
            })
            .collect();
        // The diagonal M2L's fallback (Helmholtz CH4.2b1): each level's distinct V sources on
        // this rank, own and ghost, the largest count over the ranks, so that every rank
        // takes the same levels (constraints 2 and 9). Counted on every rank, whatever the
        // setting, so that every rank reaches the all-reduce.
        let local: Vec<u64> = (0..plan.nlevels())
            .map(|l| {
                let v = plan.level(l).v();
                let mut seen = vec![false; plan.index().len(l)];
                let mut count = 0u64;
                for &s in v.sources() {
                    let seen = &mut seen[s as usize];
                    count += u64::from(!*seen);
                    *seen = true;
                }
                count
            })
            .collect();
        let mut diagonal_sources = vec![0u64; local.len()];
        comm.all_reduce_into(
            &local[..],
            &mut diagonal_sources[..],
            SystemOperation::max(),
        );
        let diagonal_sources: Vec<usize> = diagonal_sources
            .iter()
            .map(|&c| usize::try_from(c).unwrap_or(usize::MAX))
            .collect();
        // The time rule's V lists (Helmholtz CH4.3b1): per level the boxes with a V row and the
        // pairs per offset class, each box counted by the rank that computes it, summed over
        // the ranks. Counted on every rank, whatever the setting.
        let width = 1 + CLASS_COUNT;
        let mut local = vec![0u64; width * plan.nlevels()];
        for (l, counts) in local.chunks_exact_mut(width).enumerate() {
            let v = plan.level(l).v();
            for t in (0..v.nrows()).filter(|&t| plan.computes(l, t)) {
                let (sources, offsets) = v.row(t);
                if sources.is_empty() {
                    continue;
                }
                counts[0] += 1;
                for &d in offsets {
                    counts[1 + offset_class(d as usize)] += 1;
                }
            }
        }
        let mut summed = vec![0u64; local.len()];
        comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
        let diagonal_shapes: Vec<DiagonalShape> = summed
            .chunks_exact(width)
            .map(|c| DiagonalShape {
                targets: c[0],
                class_pairs: core::array::from_fn(|k| c[1 + k]),
            })
            .collect();
        timings.shapes = lap(&mut clock);
        // The operator's settings first, its tables once after them, on the pool (Helmholtz
        // CH4.0h).
        let mut operator = HelmholtzOperator::unbuilt(
            self.k,
            densities,
            &domain,
            &degrees,
            self.gradients,
            max_leaf_points,
            self.block_budget,
        )?
        .with_p2p(self.p2p)
        .expect("the P2P kernel's ISA is available (checked in step 1)")
        .with_m2l_max_degree(Some(usize::try_from(max_degree).unwrap_or(usize::MAX)))
        .with_m2l_plane_wave_min_degree(Some(
            usize::try_from(plane_wave_min_degree).unwrap_or(usize::MAX),
        ))
        .with_auto_plane_wave(self.auto_plane_wave)
        .with_plane_wave_precision(self.plane_wave_precision.clone())
        .with_plane_wave_tolerance(match self.degrees {
            Degrees::Tolerance(epsilon) => Some(epsilon),
            Degrees::PerLevel(_) => None,
        })
        .with_plane_wave_reach(self.plane_wave_reach)
        .with_plane_wave_split(self.plane_wave_split)
        .with_plane_wave_raise(self.plane_wave_raise)
        .with_m2l_min_columns(Some(min_columns))
        .with_level_shapes(shapes)
        .with_m2l_block_targets(self.m2l_block_targets)
        .with_m2l(self.m2l, dense_budget)
        // The bound first: sized for the boxes without it, the buffer would hold every box of
        // the largest level until the bound came (218 GB on W3 at kD = 500; Helmholtz CH4.0c).
        .with_plane_wave_buffer(
            (self.plane_wave_budget != usize::MAX).then(|| self.plane_wave_budget / ranks_per_node),
        )
        .with_plane_wave_boxes(&level_boxes)
        .with_rcr_split(self.rcr_split)
        .with_rcr_buffer((self.rcr_budget != usize::MAX).then(|| self.rcr_budget / ranks_per_node))
        .with_rcr_levels(&rcr_levels)
        .with_diagonal_budget(
            (self.diagonal_budget != usize::MAX).then(|| self.diagonal_budget / ranks_per_node),
        )
        .with_diagonal_sources(Some(diagonal_sources))
        .with_diagonal_costs(Some(costs))
        .with_diagonal_shapes(Some(diagonal_shapes))
        .with_diagonal_chunk(self.diagonal_chunk)
        .with_diagonal_build(self.diagonal_build)
        .with_diagonal(self.diagonal);
        if let Some(mul_add) = self.m2l_mul_add {
            operator = operator.with_m2l_mul_add(mul_add);
        }
        if let Some(pool) = &pool {
            operator = operator.with_pool(Arc::clone(pool));
        }
        let operator = operator.build_tables();
        // The degrees the operator runs at (raised on the levels with waves below 10⁻⁶).
        let degrees: Vec<usize> = (0..degrees.len()).map(|l| operator.degree(l)).collect();
        timings.tables = operator.table_timings();
        timings.operator = lap(&mut clock);
        // The V rule after the plan (Helmholtz CH4.3b2b): each level `v_after_plan` names gets
        // the crossover of the M2L the operator runs there where that is higher, from the agreed
        // costs and V lists alone, so every rank raises the same levels and builds the second
        // plan or none (constraints 2 and 9). A level whose first plan holds no V pair between
        // two leaves (over every rank) has nothing more to convert and keeps its crossover, so
        // that no second plan is built for nothing.
        let on = self.near_v == VNear::CostModel && self.v_after_plan != VAfterPlan::Off;
        let v_leaf_pairs: Vec<u64> = if on {
            let index = plan.index();
            let leaf = |l: usize, i: usize| {
                matches!(index.kind(l, i), KeyType::LocalLeaf | KeyType::GhostLeaf(_))
            };
            let local: Vec<u64> = (0..plan.nlevels())
                .map(|l| {
                    let v = plan.level(l).v();
                    (0..v.nrows())
                        .filter(|&t| plan.computes(l, t) && leaf(l, t))
                        .map(|t| {
                            let sources = v.row(t).0;
                            sources.iter().filter(|&&s| leaf(l, s as usize)).count() as u64
                        })
                        .sum()
                })
                .collect();
            let mut summed = vec![0u64; local.len()];
            comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
            summed
        } else {
            Vec::new()
        };
        let after: Option<Vec<f64>> = on.then(|| {
            let shapes = operator.diagonal_shapes();
            (0..crossovers.len())
                .map(|l| {
                    if v_leaf_pairs.get(l).is_none_or(|&n| n == 0) {
                        return crossovers[l];
                    }
                    let run = if operator.taken_from_waves(l)
                        && self.v_after_plan != VAfterPlan::RaisedLevels
                    {
                        Some(if operator.diagonal_classes(l).is_some() {
                            VLevelM2l::Form
                        } else {
                            VLevelM2l::Rcr
                        })
                    } else if waves[l]
                        && operator.m2l_on(l) == HelmholtzM2l::PlaneWave
                        && (self.v_after_plan == VAfterPlan::AllLevels
                            || (self.v_after_plan == VAfterPlan::RaisedLevels && raised_waves[l]))
                    {
                        Some(VLevelM2l::Waves)
                    } else {
                        None
                    };
                    let shape = shapes.and_then(|s| s.get(l)).copied();
                    match (run, shape) {
                        (Some(run), Some(shape)) => crossovers[l].max(v_after_plan_crossover(
                            &v_cost_model,
                            &costs,
                            run,
                            operator.degree(l),
                            &shape,
                            self.gradients,
                        )),
                        _ => crossovers[l],
                    }
                })
                .collect()
        });
        // Agreed over the ranks (the smallest), as the planned crossovers: the P2P's constant
        // is the rank's ISA's.
        let after = after.map(|local| {
            let mut agreed = vec![0.0f64; local.len()];
            comm.all_reduce_into(&local[..], &mut agreed[..], SystemOperation::min());
            agreed
        });
        let raised: Vec<bool> = match &after {
            Some(after) => after
                .iter()
                .zip(&crossovers)
                .map(|(after, planned)| after > planned)
                .collect(),
            None => vec![false; crossovers.len()],
        };
        let replanned = raised.contains(&true);
        let plan = if replanned {
            // The first plan goes before the second is built (constraint 7). Its index has the
            // ghost leaves numbered, so the second plan takes a fresh index of the octree.
            drop(plan);
            let index = BoxIndex::of_octree(&octree).map_err(FmmError::Plan)?;
            let after = after.as_deref().unwrap_or(&crossovers);
            let runs_waves: Vec<bool> = (0..crossovers.len())
                .map(|l| operator.m2l_on(l) == HelmholtzM2l::PlaneWave)
                .collect();
            // A raised level converts the first plan's pairs and those its new crossover adds:
            // per (parent, source) where the level keeps its waves, per pair elsewhere. So the V
            // lists lie within the first plan's (every buffer sized from it holds), and every
            // plan stays exact under the M2L the level runs.
            let v_pairs_after = |pair: VPair| {
                let level = morton::level(pair.target);
                if !raised[level] {
                    v_pairs(pair)
                } else if runs_waves[level] {
                    counts.as_ref().is_some_and(|counts| {
                        v_group(pair, after[level], (all_keys, counts), &parent)
                    })
                } else {
                    v_pairs(pair) || ((pair.sources * pair.targets) as f64) < after[level]
                }
            };
            let near_after = counts
                .as_ref()
                .map(|counts| NearRule::new(counts, &leaf_pairs).with_v_pairs(&v_pairs_after));
            let rules = PlanRules {
                near: near_after.as_ref(),
                empty_boxes: counts.as_ref().filter(|_| self.empty_boxes),
            };
            Plan::with_rules_on(&octree, index, &rules).map_err(FmmError::Plan)?
        } else {
            plan
        };
        timings.replan = lap(&mut clock);
        let calls = if self.pair_calls {
            Calls::Pairs(PerPair(operator))
        } else {
            Calls::Levels(operator)
        };
        let mut evaluator = Evaluator::new(
            plan,
            comm,
            calls,
            &points.source_leaves.counts,
            &points.target_leaves.counts,
        )
        .map_err(FmmError::Evaluator)?;
        timings.evaluator = lap(&mut clock);

        // Step 8: the leaf-scaled coordinates.
        points.load_coordinates(&owned, &mut evaluator, &domain);
        drop(owned);
        timings.load = lap(&mut clock);

        let strengths = densities.columns();
        Ok(HelmholtzFmm {
            octree,
            evaluator,
            domain,
            densities,
            gradients: self.gradients,
            degrees,
            min_leaf_level: leaf_level,
            max_leaf_points,
            max_points_per_leaf,
            c_leaf: self.c_leaf,
            pool,
            packed: Vec::new(),
            charges: vec![T::zero(); strengths * points.sources.nreceived()],
            received_output: vec![T::zero(); output_values * points.targets.nreceived()],
            returned_output: vec![T::zero(); output_values * points.targets.nsent()],
            points,
            ranks_per_node,
            empty_boxes: self.empty_boxes,
            near_v: self.near_v,
            near_wx: self.near_wx,
            v_after_plan: self.v_after_plan,
            v_raised_price: self.v_raised_price,
            raised_v_shapes,
            v_crossovers: crossovers,
            v_crossovers_after: after,
            replanned,
            build_timings: timings,
        })
    }
}

/// The time since `clock`, which then restarts.
fn lap(clock: &mut Instant) -> Duration {
    let now = Instant::now();
    let elapsed = now - *clock;
    *clock = now;
    elapsed
}

/// The wall time of [`HelmholtzFmmBuilder::build`] by step on this rank (Helmholtz CH4.0h;
/// [`HelmholtzFmm::build_timings`]), on the calling thread's clock: the collective steps
/// include the wait for the other ranks. No value of the FMM depends on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HelmholtzBuildTimings {
    /// Steps 1 and 2: the settings, the pool, the points' checks, the node's split and the
    /// agreements.
    pub settings: Duration,
    /// Step 3: the domain, and every point inside it.
    pub domain: Duration,
    /// Step 4: the octree with the leaf rule, and its box index.
    pub octree: Duration,
    /// Steps 5 and 6: the points to their owners.
    pub routes: Duration,
    /// Step 7 up to the plan: the largest leaf, the degrees and the M2L's agreed settings,
    /// the points per box (an exchange) and the near rules' crossovers.
    pub counts: Duration,
    /// The plan with its rules (`Plan::with_rules_on`; the near rules' decisions inside).
    pub plan: Duration,
    /// Auto's level shapes and m* (two all-reduces), the budgets and the levels' boxes.
    pub shapes: Duration,
    /// The operator with its settings, its tables included ([`tables`](Self::tables)).
    pub operator: Duration,
    /// The operator's tables by part (within [`operator`](Self::operator)).
    pub tables: TableTimings,
    /// The V rule after the plan (Helmholtz CH4.3b2b): its crossovers (an all-reduce) and
    /// the second plan, where one is built.
    pub replan: Duration,
    /// The evaluator: the stores and the exchanges.
    pub evaluator: Duration,
    /// Step 8: the leaf-scaled coordinates.
    pub load: Duration,
}

impl HelmholtzBuildTimings {
    /// The sum of the steps (the tables are part of the operator).
    pub fn total(&self) -> Duration {
        self.settings
            + self.domain
            + self.octree
            + self.routes
            + self.counts
            + self.plan
            + self.shapes
            + self.operator
            + self.replan
            + self.evaluator
            + self.load
    }
}

/// The constants of [`VNear::CostModel`] for one machine class (Helmholtz CH4.0d2), in
/// thread-nanoseconds: the P2P's per interaction (complex charges with gradients, f64), and
/// per V pair Rotation's a + b (p + 1)³, Batched's (also Dense's and Classes') b (p + 1)⁴ and
/// the exponential M2L's b (p_w + 1)² at the degree its level runs. Fitted from the P2P and
/// M2L times per level that the converted pairs added and saved
/// (fmm-validate/results/helmholtz-ch4-0d2.md); the waves' constant from levels whose pairs
/// all converted, where the level's waves go too.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VCosts {
    /// The P2P per interaction, with gradients.
    pub p2p: f64,
    /// Rotation per pair, [a, b] of a + b (p + 1)³.
    pub rotation: [f64; 2],
    /// Batched per pair, b of b (p + 1)⁴.
    pub batched: f64,
    /// The exponential M2L per pair, b of b (p_w + 1)².
    pub plane_wave: f64,
}

/// [`VCosts`] of x86 (AVX-512 and AVX2; measured on Kathleen, Xeon Gold 6248, one rank ×
/// 40 threads).
pub const V_COSTS_X86: VCosts = VCosts {
    p2p: 6.5,
    rotation: [800.0, 6.0],
    batched: 0.25,
    plane_wave: 30.0,
};

/// [`VCosts`] of NEON (measured on locust, Neoverse-V2, one rank × 72 threads; the M6 takes
/// them unmeasured).
pub const V_COSTS_NEON: VCosts = VCosts {
    p2p: 14.0,
    rotation: [600.0, 2.5],
    batched: 0.07,
    plane_wave: 6.0,
};

/// The cost model of the W and X pairs' near rule for one machine class (Helmholtz CH4.0d3,
/// [`WxNear::CostModel`]), in thread-nanoseconds: an M2P per target point and a P2L per
/// source point, each a + b (p + 1)² at the degree p of the finer leaf's level, complex
/// charges with gradients in f64 (the P2P per interaction is [`VCosts::p2p`], CH4.0d2's: the
/// calibration of these constants measured 6.9 and 16.6 ns, so the crossovers lie 6 % and 18 %
/// above its own P2P's on x86 and NEON; the rule's gain was measured with them). The fits
/// overestimate M2P at p = 11, a level of few rows (about 1.9×; the report's §2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WxCosts {
    /// M2P per target point, [a, b].
    pub m2p: [f64; 2],
    /// P2L per source point, [a, b].
    pub p2l: [f64; 2],
}

/// [`WxCosts`] of x86 (AVX-512 and AVX2; measured on Kathleen, Xeon Gold 6248, one rank ×
/// 40 threads, P2L at p = 7, 15 and 32, M2P also at 11: fmm-validate/results/helmholtz-ch4-0d3.md).
pub const WX_COSTS_X86: WxCosts = WxCosts {
    m2p: [274.0, 26.4],
    p2l: [339.0, 7.85],
};

/// [`WxCosts`] of NEON (measured on locust, Neoverse-V2, one rank × 72 threads, P2L at p = 7,
/// 15 and 32, M2P also at 11: fmm-validate/results/helmholtz-ch4-0d3.md; the fitted M2P
/// constant term, −87 ns, set to 0; the M6 takes them unmeasured).
pub const WX_COSTS_NEON: WxCosts = WxCosts {
    m2p: [0.0, 20.2],
    p2l: [129.0, 5.67],
};

/// The [`WxCosts`] of the machine class of `isa` (NEON's for NEON, x86's otherwise).
pub fn wx_costs(isa: Isa) -> WxCosts {
    match isa {
        Isa::Neon => WX_COSTS_NEON,
        _ => WX_COSTS_X86,
    }
}

/// The crossovers [W, X] of a level of degree `p` (Helmholtz CH4.0d3, [`WxNear::CostModel`]):
/// an M2P per target point over the P2P per interaction, in the W leaf's source points, and a
/// P2L per source point over it, in the X leaf's target points. Every constant was measured
/// with gradients; without them the crossovers are not corrected (unmeasured).
pub fn wx_crossovers(costs: &WxCosts, v_costs: &VCosts, p: usize) -> [f64; 2] {
    let q2 = ((p + 1) * (p + 1)) as f64;
    [
        (costs.m2p[0] + costs.m2p[1] * q2) / v_costs.p2p,
        (costs.p2l[0] + costs.p2l[1] * q2) / v_costs.p2p,
    ]
}

/// The share of the P2P's time per interaction without gradients: the P2P kernel's
/// potential over its potential and gradient (Helmholtz CH3.4b, Laplace's AVX-512 f64 0.40
/// against 0.60 ns).
pub const V_P2P_POTENTIAL_SHARE: f64 = 0.67;

/// The [`VCosts`] of the machine class of `isa` (NEON's for NEON, x86's otherwise).
pub fn v_costs(isa: Isa) -> VCosts {
    match isa {
        Isa::Neon => V_COSTS_NEON,
        _ => V_COSTS_X86,
    }
}

/// The crossover of n_s n_t on a level of degree `p` whose cheapest M2L is `m2l` (the
/// waves' degree for the exponential M2L), with or without gradients, under `costs`
/// (Helmholtz CH4.0d2, [`VNear::CostModel`]): the M2L's thread time per pair over the
/// P2P's per interaction.
pub fn v_crossover(costs: &VCosts, m2l: HelmholtzM2l, p: usize, gradients: bool) -> f64 {
    let q = (p + 1) as f64;
    let m2l = match m2l {
        HelmholtzM2l::Rotation => costs.rotation[0] + costs.rotation[1] * q.powi(3),
        HelmholtzM2l::PlaneWave => costs.plane_wave * q * q,
        _ => costs.batched * q.powi(4),
    };
    let p2p = if gradients {
        costs.p2p
    } else {
        costs.p2p * V_P2P_POTENTIAL_SHARE
    };
    m2l / p2p
}

/// The crossover of n_s n_t after the plan on a level that runs `m2l` at degree `p` with V
/// lists `shape` (Helmholtz CH4.3b2b, [`VAfterPlan`]): the M2L's thread time per pair over the
/// P2P's per interaction, with or without gradients. The diagonal form's pair and the waves'
/// cost per box with a V row from the time rule's `diagonal` costs (the waves' over the
/// level's pairs, their mean per pair; 0 on a level without pairs), RCR's pair from `v`'s
/// (as [`v_crossover`]). A function of its arguments alone (constraint 9).
pub fn v_after_plan_crossover(
    v: &VCosts,
    diagonal: &DiagonalCosts,
    m2l: VLevelM2l,
    p: usize,
    shape: &DiagonalShape,
    gradients: bool,
) -> f64 {
    let pair = match m2l {
        VLevelM2l::Waves => {
            let pairs: u64 = shape.class_pairs.iter().sum();
            if pairs == 0 {
                return 0.0;
            }
            DiagonalCosts::unit(diagonal.plane_wave, p) * shape.targets as f64 / pairs as f64
        }
        VLevelM2l::Form => DiagonalCosts::unit(diagonal.pair, p),
        VLevelM2l::Rcr => return v_crossover(v, HelmholtzM2l::Rotation, p, gradients),
    };
    let p2p = if gradients {
        v.p2p
    } else {
        v.p2p * V_P2P_POTENTIAL_SHARE
    };
    pair / p2p
}

/// The leaves and V entries of each level `levels` names (Helmholtz CH4.3b2d,
/// [`VRaisedPrice::PerBox`]), on this rank, as [`LeafVShape`] counts them: per local leaf t,
/// its V entries by the per-key rule of `nd_fmm_plan::interaction_manager`, the children of
/// the interior same-level neighbours of t's parent P that are not adjacent to t; under the
/// empty-box rule none for a t without target points, and none for a source box without
/// source points; less the entries between two leaves that `first` converts (the V rule at
/// the per-pair crossover), so the entries a plan of that rule leaves in t's row, and t
/// counted only with one left. From keys and counts by key, so summed over the ranks it is
/// the one-rank count. The candidates are found once per parent and filtered per child, in
/// key order, as the plan offers its pairs (`first` may cache per parent).
fn leaf_v_shapes(
    keys: &HashMap<MortonKey, KeyType>,
    counts: Option<&LeafCounts>,
    empty_boxes: bool,
    levels: &[bool],
    first: &dyn Fn(VPair) -> bool,
) -> Vec<LeafVShape> {
    let interior = |key: &MortonKey| {
        matches!(
            keys.get(key),
            Some(KeyType::LocalInterior | KeyType::GhostInterior(_) | KeyType::Global)
        )
    };
    let empty = counts.filter(|_| empty_boxes);
    let has_sources =
        |key: MortonKey| empty.is_none_or(|counts| counts.box_sources(key).is_none_or(|n| n > 0));
    let has_targets =
        |key: MortonKey| empty.is_none_or(|counts| counts.box_targets(key).is_none_or(|n| n > 0));
    let is_leaf = |key: &MortonKey| {
        matches!(
            keys.get(key),
            Some(KeyType::LocalLeaf | KeyType::GhostLeaf(_))
        )
    };
    let mut leaves: Vec<MortonKey> = keys
        .iter()
        .filter(|(key, kind)| {
            **kind == KeyType::LocalLeaf && levels.get(morton::level(**key)) == Some(&true)
        })
        .map(|(key, _)| *key)
        .collect();
    // Key order: a level's siblings come together, so each parent's candidates once.
    leaves.sort_unstable();
    let mut shapes = vec![LeafVShape::default(); levels.len()];
    let mut candidates: Vec<(MortonKey, [usize; 3])> = Vec::new();
    let mut current = None;
    for leaf in leaves {
        let Some(parent) = morton::parent(leaf) else {
            continue;
        };
        if current != Some(parent) {
            current = Some(parent);
            candidates.clear();
            for neighbour in morton::neighbours(parent)
                .into_iter()
                .filter(|&n| morton::is_valid(n) && interior(&n))
            {
                let Some(children) = morton::children(neighbour) else {
                    continue;
                };
                candidates.extend(
                    children
                        .into_iter()
                        .filter(|&c| has_sources(c))
                        .map(|c| (c, morton::decode(c).1)),
                );
            }
        }
        if !has_targets(leaf) {
            continue;
        }
        let (level, t) = morton::decode(leaf);
        let targets = counts.and_then(|counts| counts.targets(leaf));
        let pairs = candidates
            .iter()
            .filter(|(_, c)| (0..3).any(|i| c[i].abs_diff(t[i]) > 1))
            .filter(|(source, _)| {
                // A pair between two leaves that the first rule converts leaves the row.
                let sources = counts.and_then(|counts| counts.sources(*source));
                match (is_leaf(source), sources, targets) {
                    (true, Some(sources), Some(targets)) => !first(VPair {
                        target: leaf,
                        source: *source,
                        sources,
                        targets,
                    }),
                    _ => true,
                }
            })
            .count() as u64;
        if pairs > 0 {
            shapes[level].leaves += 1;
            shapes[level].pairs += pairs;
        }
    }
    shapes
}

/// The crossover of n_s n_t before the plan on a level whose waves run at the raised degree
/// `p_w` (Helmholtz CH4.3b2d, [`VRaisedPrice::PerBox`]): the waves' cost per box with a V row
/// at p_w ([`DiagonalCosts::plane_wave`]) times the level's leaves with a V entry over their
/// entries (the mean per pair, as [`v_after_plan_crossover`] prices a level that keeps its
/// waves after the plan), over the P2P's per interaction, with or without gradients; 0 on a
/// level without entries. A function of its arguments alone (constraint 9).
pub fn raised_crossover(
    v: &VCosts,
    diagonal: &DiagonalCosts,
    p_w: usize,
    shape: LeafVShape,
    gradients: bool,
) -> f64 {
    if shape.pairs == 0 {
        return 0.0;
    }
    let pair =
        DiagonalCosts::unit(diagonal.plane_wave, p_w) * shape.leaves as f64 / shape.pairs as f64;
    let p2p = if gradients {
        v.p2p
    } else {
        v.p2p * V_P2P_POTENTIAL_SHARE
    };
    pair / p2p
}

/// The V rule's decision for `pair` on a level with waves (Helmholtz CH4.0d2): the
/// exponential M2L's merge and shift serves each source s of a parent P's slots to every
/// child of P whose V list holds it (geometric masks), so a pair cannot convert alone. The
/// decision is once per (P, s), the same for each child t of P that reads s: they convert
/// if every such child is a leaf and n_s n_P < `crossover` m_s, with n_P the target points
/// of P's subtree and m_s the children reading s (n_P / m_s bounds the mean target count
/// of those children). From keys and counts by key, so every rank decides as one rank.
/// `parent` caches P's leaf mask and n_P for the next pair: the plan offers a target's V
/// entries, and the targets of a level, in key order, so siblings come together.
fn v_group(
    pair: VPair,
    crossover: f64,
    (keys, counts): (&HashMap<MortonKey, KeyType>, &LeafCounts),
    parent: &Cell<Option<(MortonKey, u8, Option<usize>)>>,
) -> bool {
    let Some(p) = morton::parent(pair.target) else {
        return false;
    };
    let (leaves, targets) =
        match parent.get() {
            Some((key, leaves, targets)) if key == p => (leaves, targets),
            _ => {
                let Some(children) = morton::children(p) else {
                    return false;
                };
                // Every sibling of a local leaf is held (it touches the leaf), and so are P's
                // counts (P is local or a top-tree key): a gap would decide unlike one rank.
                debug_assert!(
                    children.iter().all(|child| keys.contains_key(child)),
                    "the siblings of a local leaf are held"
                );
                let leaves = children.iter().enumerate().fold(0u8, |mask, (o, child)| {
                    match keys.get(child) {
                        Some(KeyType::LocalLeaf | KeyType::GhostLeaf(_)) => mask | (1 << o),
                        _ => mask,
                    }
                });
                let targets = counts.box_targets(p);
                debug_assert!(
                    targets.is_some(),
                    "the target count of a local leaf's parent"
                );
                parent.set(Some((p, leaves, targets)));
                (leaves, targets)
            }
        };
    let Some(targets) = targets else {
        return false;
    };
    // The cheap bound first: at most eight children read s.
    let n = (pair.sources * targets) as f64;
    if n >= 8.0 * crossover {
        return false;
    }
    // s's parent is a neighbour of P, so s is in the V list of every child of P it does
    // not touch.
    let (_, s) = morton::decode(pair.source);
    let Some(children) = morton::children(p) else {
        return false;
    };
    let mut readers = 0u8;
    for (o, &child) in children.iter().enumerate() {
        let (_, c) = morton::decode(child);
        if (0..3).any(|i| c[i].abs_diff(s[i]) > 1) {
            readers |= 1 << o;
        }
    }
    readers & !leaves == 0 && n < crossover * f64::from(readers.count_ones())
}

/// The unit cube, for the tolerance's own check.
fn unit_domain() -> Domain {
    Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).expect("the unit cube")
}

/// Agrees `local` on every rank: the rank that failed returns its error, the others
/// [`HelmholtzFmmError::OtherRank`] (one all-reduce).
fn agree<C: CommunicatorCollectives, V>(
    comm: &C,
    local: Result<V, HelmholtzFmmError>,
) -> Result<V, HelmholtzFmmError> {
    let mut valid = false;
    comm.all_reduce_into(&local.is_ok(), &mut valid, SystemOperation::logical_and());
    match local {
        Ok(_) if !valid => Err(HelmholtzFmmError::OtherRank),
        result => result,
    }
}

/// The operator of the evaluator: the level calls, or (for tests) the same bodies pair by
/// pair.
enum Calls<T: SimdScalar + Equivalence + Default> {
    Levels(HelmholtzOperator<T>),
    Pairs(PerPair<HelmholtzOperator<T>>),
}

impl<T: SimdScalar + Equivalence + Default> Calls<T> {
    fn operator(&self) -> &HelmholtzOperator<T> {
        match self {
            Self::Levels(operator) | Self::Pairs(PerPair(operator)) => operator,
        }
    }
}

/// Every method delegates to the operator.
macro_rules! delegate {
    ($($kind:ident: $batch:ident),* $(,)?) => {
        $(
            fn $kind(&mut self, batch: $batch<'_, T>) {
                match self {
                    Self::Levels(operator) => FmmOperator::$kind(operator, batch),
                    Self::Pairs(pairs) => pairs.$kind(batch),
                }
            }
        )*
    };
}

impl<T: SimdScalar + Equivalence + Default> FmmSizes for Calls<T> {
    type Value = T;

    fn multipole_size(&self, level: usize) -> usize {
        self.operator().multipole_size(level)
    }

    fn local_size(&self, level: usize) -> usize {
        self.operator().local_size(level)
    }

    fn source_point_size(&self) -> usize {
        self.operator().source_point_size()
    }

    fn target_input_point_size(&self) -> usize {
        self.operator().target_input_point_size()
    }

    fn target_output_point_size(&self) -> usize {
        self.operator().target_output_point_size()
    }
}

impl<T: SimdScalar + Equivalence + Default> FmmOperator for Calls<T> {
    delegate!(p2m: P2m, m2m: M2m, m2l: M2l, p2l: P2l, l2l: L2l, l2p: L2p, m2p: M2p, p2p: P2p);
}

/// A Helmholtz FMM built on the points of every rank ([module documentation](self)).
pub struct HelmholtzFmm<'o, T, C = SimpleCommunicator>
where
    T: SimdScalar + Equivalence + Default,
    C: CommunicatorCollectives,
{
    octree: Octree<'o, C>,
    evaluator: Evaluator<'o, C, Calls<T>, Plan>,
    domain: Domain,
    densities: Densities,
    gradients: bool,
    degrees: Vec<usize>,
    min_leaf_level: usize,
    max_leaf_points: usize,
    /// The octree's refinement target as built (Helmholtz CH4.0d3).
    max_points_per_leaf: usize,
    /// The maximum leaf width in wavelengths as built (Helmholtz CH4.0d3).
    c_leaf: f64,
    pool: Option<Arc<ThreadPool>>,
    /// The caller's columns packed point-major (c m per source) for the forward move, when
    /// there is more than one column; allocated by the first evaluation that needs it.
    packed: Vec<T>,
    /// The strengths of the owned sources in leaf order, c m per source.
    charges: Vec<T>,
    /// The output pass's values, o per owned target in leaf order.
    received_output: Vec<T>,
    /// The backward move's values, o per target in the caller's order.
    returned_output: Vec<T>,
    /// The routes, the leaf ranges, the radii and the output order.
    points: Points,
    /// The largest number of ranks on one node, agreed at build.
    ranks_per_node: usize,
    /// Whether the plan has the empty-box rule.
    empty_boxes: bool,
    /// The V pairs' near rule as built (Helmholtz CH4.0d2).
    near_v: VNear,
    /// The W and X pairs' near rule as built (Helmholtz CH4.0d3).
    near_wx: WxNear,
    /// The V rule after the plan as built (Helmholtz CH4.3b2b).
    v_after_plan: VAfterPlan,
    /// The V rule's price on raised levels as built (Helmholtz CH4.3b2d).
    v_raised_price: VRaisedPrice,
    /// The raised levels' leaves and V entries counted before the plan, summed over the
    /// ranks, under [`VRaisedPrice::PerBox`] where a level is raised.
    raised_v_shapes: Option<Vec<LeafVShape>>,
    /// The V rule's crossover of each level before the plan, agreed.
    v_crossovers: Vec<f64>,
    /// The crossovers after the plan, agreed, under [`VAfterPlan`] with the cost model.
    v_crossovers_after: Option<Vec<f64>>,
    /// Whether a crossover rose after the plan, so the plan was built again.
    replanned: bool,
    /// The build's wall time by step (Helmholtz CH4.0h).
    build_timings: HelmholtzBuildTimings,
}

impl<'o, T, C> HelmholtzFmm<'o, T, C>
where
    T: SimdScalar + Equivalence + Default,
    C: CommunicatorCollectives,
{
    /// Evaluates the potentials, and the gradients if built with them, of the m charge
    /// vectors `charges` at every target of this rank ([module
    /// documentation](self#evaluation)): `charges[v]` holds vector v at this rank's
    /// sources of [`HelmholtzFmmBuilder::build`], in their order. Allocates the output:
    /// [`evaluate_into`](Self::evaluate_into) a fresh [`Output`], with the same bits.
    ///
    /// # Collective operation
    ///
    /// Every rank must call it: one all-reduce, then the collectives of the evaluator's
    /// stages and of the two moves.
    ///
    /// # Errors
    ///
    /// [`HelmholtzFmmError::Charges`] on a rank whose charges do not fit the build
    /// ([`Densities::check_charges`]), [`HelmholtzFmmError::OtherRank`] on the others;
    /// nothing is evaluated.
    pub fn evaluate(&mut self, charges: &[Charges<'_, T>]) -> Result<Output<T>, HelmholtzFmmError> {
        let mut output = Output::new(&self.densities, 0, self.gradients);
        self.evaluate_into(charges, &mut output)?;
        Ok(output)
    }

    /// [`evaluate`](Self::evaluate) into `output`, whose buffers are reused: the same bits.
    /// An `output` of another path (vectors, or real and complex planes) is replaced by
    /// one of this path first; its buffers are resized as needed, every element written
    /// once. On an error `output` is left as it was.
    ///
    /// # Collective operation
    ///
    /// As [`evaluate`](Self::evaluate).
    ///
    /// # Errors
    ///
    /// As [`evaluate`](Self::evaluate).
    pub fn evaluate_into(
        &mut self,
        charges: &[Charges<'_, T>],
        output: &mut Output<T>,
    ) -> Result<(), HelmholtzFmmError> {
        let local = self
            .densities
            .check_charges(charges, self.points.sources.nsent())
            .map_err(HelmholtzFmmError::from);
        agree(self.evaluator.comm(), local)?;
        let pool = self.pool.as_deref();

        // The c m columns, Re q then Im q per complex vector, packed point-major and moved
        // to the ranks that own their sources, in leaf order.
        let columns: Vec<&[T]> = charges
            .iter()
            .flat_map(|vector| match *vector {
                Charges::Real(q) => [Some(q), None],
                Charges::Complex { re, im } => [Some(re), Some(im)],
            })
            .flatten()
            .collect();
        let w = columns.len();
        if w > 1 {
            pack_charges(&columns, pool, &mut self.packed);
        }
        let input = if w > 1 { &self.packed[..] } else { columns[0] };
        self.points
            .sources
            .forward_into(input, w, &mut self.charges);
        load_source_charges(
            self.evaluator.local_sources_mut(),
            &self.points.source_leaves,
            &self.charges,
            (w, None),
            pool,
        );

        self.evaluator.evaluate();

        // The output pass with the output columns as the vectors, the move back, and the
        // columns into the output's planes.
        let o = self.densities.output_columns();
        let store = self.evaluator.target_output_store();
        gather_output(
            store.as_slice(),
            store.point_offsets(),
            &self.points.outputs,
            (self.gradients, o),
            pool,
            &mut self.received_output,
        );
        let per_target = self.densities.target_output_point_size(self.gradients);
        self.points.targets.backward_into(
            &self.received_output,
            per_target,
            &mut self.returned_output,
        );
        if output.vectors() != self.densities.vectors()
            || output.is_complex() != (o == 2 * self.densities.vectors())
        {
            *output = Output::new(&self.densities, 0, self.gradients);
        }
        let buffers = (&mut output.potential, &mut output.gradient);
        unpack_output_many(&self.returned_output, (self.gradients, o), pool, buffers);
        Ok(())
    }

    /// The output of the last evaluation by the output pass of
    /// [`Densities::write_leaf_output`] (Helmholtz CH3.0c) leaf by leaf, moved back column
    /// by column, for tests only: [`evaluate`](Self::evaluate) returns it bit for bit.
    ///
    /// # Collective operation
    ///
    /// Every rank must call it: the backward moves of every column.
    #[doc(hidden)]
    pub fn reference_output(&self) -> Output<T> {
        let ntargets = self.points.targets.nreceived();
        let mut received = Output::new(&self.densities, ntargets, self.gradients);
        let store = self.evaluator.target_output_store();
        for (j, &radius) in self.points.radii.iter().enumerate() {
            let targets: Vec<usize> = self.points.target_leaves.range(j).collect();
            self.densities.write_leaf_output(
                store.chunk(j),
                radius,
                self.gradients,
                &targets,
                &mut received,
            );
        }
        let mut output = Output::new(&self.densities, 0, self.gradients);
        for column in 0..self.densities.output_columns() {
            let at = column * ntargets..(column + 1) * ntargets;
            let moved = self
                .points
                .targets
                .backward(&received.potential[at.clone()], 1);
            output.potential.extend(moved);
            if let (Some(all), Some(gradient)) = (output.gradient.as_mut(), &received.gradient) {
                let flat = self.points.targets.backward(gradient[at].as_flattened(), 3);
                all.extend_from_slice(flat.as_chunks::<3>().0);
            }
        }
        output
    }

    /// The octree.
    pub fn octree(&self) -> &Octree<'o, C> {
        &self.octree
    }

    /// The plan.
    pub fn plan(&self) -> &Plan {
        self.evaluator.plan()
    }

    /// The operator.
    pub fn operator(&self) -> &HelmholtzOperator<T> {
        self.evaluator.operator().operator()
    }

    /// The P2P kernel as it runs: [`P2pChoice::Reference`] or [`P2pChoice::Isa`], never
    /// [`P2pChoice::Auto`].
    pub fn p2p_kernel(&self) -> P2pChoice {
        self.operator().p2p_kernel()
    }

    /// The levels' V lists over the whole tree that [`HelmholtzM2l::Auto`]'s term reads
    /// ([`HelmholtzFmmBuilder::batched_term`]; Helmholtz CH4.0), one per level, the same on
    /// every rank; `None` with the term off.
    pub fn level_shapes(&self) -> Option<&[LevelShape]> {
        self.operator().level_shapes()
    }

    /// The bytes of the dense, class or batched M2L tables this rank may hold: the builder's
    /// [`dense_budget`](HelmholtzFmmBuilder::dense_budget), at most the
    /// [`dense_node_budget`](HelmholtzFmmBuilder::dense_node_budget) over the ranks per node
    /// (Helmholtz CH4.0).
    pub fn dense_budget(&self) -> usize {
        self.operator().dense_budget()
    }

    /// The agreed crossover degree under [`HelmholtzM2l::Auto`]
    /// ([`HelmholtzFmmBuilder::m2l_max_degree`]), `None` under the other settings; the
    /// operator's `m2l_on` gives each level's M2L.
    pub fn m2l_max_degree(&self) -> Option<usize> {
        self.operator().max_degree()
    }

    /// The agreed smallest degree at which [`HelmholtzM2l::Auto`] takes the exponential M2L
    /// ([`HelmholtzFmmBuilder::m2l_plane_wave_min_degree`]), `None` unless Auto with
    /// [`HelmholtzFmmBuilder::auto_plane_wave`] in f64.
    pub fn m2l_plane_wave_min_degree(&self) -> Option<usize> {
        self.operator().plane_wave_min_degree()
    }

    /// The largest number of ranks on one node, agreed at build: the divisor of the
    /// exponential M2L's waves budget ([`HelmholtzFmmBuilder::plane_wave_budget`]).
    pub fn ranks_per_node(&self) -> usize {
        self.ranks_per_node
    }

    /// The build's wall time on this rank by step (Helmholtz CH4.0h).
    pub fn build_timings(&self) -> HelmholtzBuildTimings {
        self.build_timings
    }

    /// Whether the plan has the empty-box rule ([`HelmholtzFmmBuilder::empty_boxes`]);
    /// what it left out on this rank is [`plan`](Self::plan)'s `empty_boxes`.
    pub fn empty_boxes(&self) -> bool {
        self.empty_boxes
    }

    /// The source points the source exchange moves on this rank per evaluation, (sent,
    /// received): every ghost leaf's points (those of the U, X and converted W and V
    /// entries, [`HelmholtzFmmBuilder::near_v`]), a point sent to two ranks counted twice.
    pub fn source_points_exchanged(&self) -> (usize, usize) {
        let exchange = self.evaluator.source_exchange();
        let sent = exchange.traffic().values_sent / exchange.point_size().max(1);
        (sent, exchange.ghost_counts().iter().sum())
    }

    /// The near interactions of each level's P2P on this rank (Helmholtz CH4.0d2): over the
    /// local leaves of the level, the leaf's target points times the source points of every
    /// leaf of its near row (itself included), Σ_t n_t Σ_s n_s.
    pub fn near_interactions(&self) -> Vec<u64> {
        let plan = self.evaluator.plan();
        let sources = self.evaluator.source_exchange().leaf_counts();
        let targets = &self.points.target_leaves.counts;
        let leaves = plan.index().leaves();
        (0..plan.nlevels())
            .map(|level| {
                let near = plan.level(level).near();
                let first = leaves.local(level).start;
                (0..near.nrows())
                    .map(|r| {
                        let n_s: usize = near.row(r).iter().map(|&j| sources[j as usize]).sum();
                        (targets[first + r] * n_s) as u64
                    })
                    .sum()
            })
            .collect()
    }

    /// The point interactions of each level's P2L and M2P on this rank (Helmholtz CH4.0d3):
    /// `[Σ_X n_s, Σ_W n_t]`, over the boxes of the level the source points of every leaf of
    /// their X rows, and over the local leaves of the level their target points times the
    /// entries of their W rows (the work of P2L and M2P, each per point and expansion).
    pub fn far_interactions(&self) -> Vec<[u64; 2]> {
        let plan = self.evaluator.plan();
        let sources = self.evaluator.source_exchange().leaf_counts();
        let targets = &self.points.target_leaves.counts;
        let leaves = plan.index().leaves();
        (0..plan.nlevels())
            .map(|level| {
                let lists = plan.level(level);
                let x = lists.x();
                let p2l: usize = (0..x.nrows())
                    .map(|t| x.row(t).iter().map(|&j| sources[j as usize]).sum::<usize>())
                    .sum();
                let w = lists.w();
                let first = leaves.local(level).start;
                let m2p: usize = (0..w.nrows())
                    .map(|r| targets[first + r] * w.row(r).len())
                    .sum();
                [p2l as u64, m2p as u64]
            })
            .collect()
    }

    /// The V pairs' near rule as built ([`HelmholtzFmmBuilder::near_v`]); the pairs offered
    /// and converted on this rank are [`plan`](Self::plan)'s `near_pairs`.
    pub fn near_v(&self) -> VNear {
        self.near_v
    }

    /// The W and X pairs' near rule as built ([`HelmholtzFmmBuilder::near_wx`]); the pairs
    /// offered and converted on this rank are [`plan`](Self::plan)'s `near_pairs`.
    pub fn near_wx(&self) -> WxNear {
        self.near_wx
    }

    /// The V rule after the plan as built ([`HelmholtzFmmBuilder::v_after_plan`]).
    pub fn v_after_plan(&self) -> VAfterPlan {
        self.v_after_plan
    }

    /// The V rule's price on raised levels as built ([`HelmholtzFmmBuilder::v_raised_price`]).
    pub fn v_raised_price(&self) -> VRaisedPrice {
        self.v_raised_price
    }

    /// The leaves and V entries of each level as counted before the plan under
    /// [`VRaisedPrice::PerBox`] (those the per-pair crossover leaves; [`LeafVShape`]), summed
    /// over the ranks: on the levels whose waves run at a raised degree, zero on the others; `None` where nothing was counted (another price,
    /// another [`VNear`], no raised level).
    pub fn raised_v_shapes(&self) -> Option<&[LeafVShape]> {
        self.raised_v_shapes.as_deref()
    }

    /// The V rule's crossover of each level as set before the plan, agreed over the ranks:
    /// n_s n_t (per (parent, source) on a level with predicted waves) below it converts; 0
    /// with the rule off, the named one under [`VNear::Crossover`].
    pub fn v_crossovers(&self) -> &[f64] {
        &self.v_crossovers
    }

    /// The crossovers after the plan ([`HelmholtzFmmBuilder::v_after_plan`]; `None` with it
    /// off or under a [`VNear`] other than the cost model): each at least
    /// [`v_crossovers`](Self::v_crossovers)'.
    pub fn v_crossovers_after_plan(&self) -> Option<&[f64]> {
        self.v_crossovers_after.as_deref()
    }

    /// Whether a crossover rose after the plan, so that the plan was built again
    /// ([`HelmholtzBuildTimings::replan`]). The operator's choices, and the level and diagonal
    /// shapes it reports, are then those of the first plan, which decided them.
    pub fn replanned(&self) -> bool {
        self.replanned
    }

    /// The domain.
    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// The path: k's kind, the charges' kind and the vectors.
    pub fn densities(&self) -> &Densities {
        &self.densities
    }

    /// Whether the output holds gradients.
    pub fn gradients(&self) -> bool {
        self.gradients
    }

    /// The degree p_l at which every level runs: [`Degrees::resolve`]'s, or under
    /// [`HelmholtzM2l::PlaneWave`]'s rule below ε = 10⁻⁶ and under [`HelmholtzM2l::Auto`]
    /// at 10⁻⁸ ≤ ε < 10⁻⁷ the raised degree of a level with waves
    /// (`HelmholtzOperator::degree`; Helmholtz CH6.1g).
    pub fn degrees(&self) -> &[usize] {
        &self.degrees
    }

    /// The number of levels of the plan.
    pub fn nlevels(&self) -> usize {
        self.degrees.len()
    }

    /// The number of local leaves.
    pub fn nleaves(&self) -> usize {
        self.points.radii.len()
    }

    /// The leaf level of the leaf rule: no leaf holding a point lies above it (unless
    /// the maximum level does).
    pub fn min_leaf_level(&self) -> usize {
        self.min_leaf_level
    }

    /// The largest number of sources in a leaf, over every rank.
    pub fn max_leaf_points(&self) -> usize {
        self.max_leaf_points
    }

    /// The octree's refinement target, points per leaf, as built
    /// ([`HelmholtzFmmBuilder::max_points_per_leaf`]; Helmholtz CH4.0d3).
    pub fn max_points_per_leaf(&self) -> usize {
        self.max_points_per_leaf
    }

    /// The maximum leaf width in wavelengths as built ([`HelmholtzFmmBuilder::c_leaf`];
    /// Helmholtz CH4.0d3).
    pub fn c_leaf(&self) -> f64 {
        self.c_leaf
    }

    /// This rank's local leaves per level of the plan (Helmholtz CH4.0d3): `[leaves,
    /// leaves holding a source or a target]`. Summed over the ranks, the same on every rank
    /// count.
    pub fn leaves_by_level(&self) -> Vec<[usize; 2]> {
        let leaves = self.evaluator.plan().index().leaves();
        let mut counts = vec![[0usize; 2]; self.nlevels()];
        for j in 0..self.nleaves() {
            let count = &mut counts[leaves.level(j)];
            count[0] += 1;
            let points = self.points.source_leaves.counts[j] + self.points.target_leaves.counts[j];
            count[1] += usize::from(points > 0);
        }
        counts
    }

    /// This rank's sources and targets, as passed to the build.
    pub fn npoints(&self) -> (usize, usize) {
        (self.points.sources.nsent(), self.points.targets.nsent())
    }
}

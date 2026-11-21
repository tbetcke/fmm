//! The Helmholtz FMM against the sampled direct sum on the reference workloads
//! (Helmholtz CH3.1c; docs/design/helmholtz-fmm-proposal.md §9.1 and §10, CH3.1), on any
//! number of ranks.
//!
//! A [`Case`] names a workload of [`helmholtz_workloads`](crate::helmholtz_workloads), k,
//! the degrees and the charges; [`Problem`] draws the points, the sampled targets and the
//! charge vectors; [`Oracle`] is the direct sum in f64 at the sampled targets
//! (`nd_fmm_ref::helmholtz::p2p::direct_sum`, scaled by 1/(4π) as the FMM's output is); [`run`] builds the
//! [`HelmholtzFmm`] at each thread count, evaluates it and measures it:
//!
//! - **Points.** Sources equal targets. Rank r of P passes the contiguous block
//!   [`Problem::block`] of the workload's points; the FMM moves them to the ranks that own
//!   their leaves.
//! - **Charges.** §9.1: per vector, parts uniform in [−1, 1), drawn from the generator of
//!   `seed + 1 + v`, Re q at every point, then Im q (complex charges only). Or a smooth
//!   density ([`Case::density`], Helmholtz CH4.0b): the trace of a plane wave or a constant.
//! - **Errors.** §9.1's norms over the sampled targets: the relative ℓ2 error of the
//!   complex potential, ‖φ − φ_ref‖₂ / ‖φ_ref‖₂, and the same over the 3-vector gradient
//!   field, per vector and over all vectors together (the sums of squares added), reduced
//!   over the ranks.
//! - **Checksum.** A rank-invariant checksum of the whole output ([`checksum`]): the
//!   wrapping sum over every output value of a mix of its global point index, its output
//!   slot and its bits, summed over the ranks. Equal outputs give equal checksums on any
//!   number of ranks. One changed value always changes it (the mix is a bijection);
//!   several changed values leave it equal with probability about 2⁻⁶⁴.
//!   Determinism (proposal §2, constraint 2) is shown by equal checksums across thread
//!   counts and rank counts.
//! - **Times.** The build and the evaluation, wall time of the slowest rank.
//! - **P2P kernel.** [`Case::p2p`] (Helmholtz CH3.4c): `Auto` by default, `Reference` for
//!   the times before the vectorised kernel; [`Row::p2p`] the kernel that ran.
//! - **M2L.** [`Case::m2l`] and [`Case::dense_budget`] (Helmholtz CH3.2a, CH3.2b, CH3.2c):
//!   Rotation by default, or Dense, Classes or Batched on the levels whose tables fit the
//!   budget, Batched in [`Case::m2l_mul_add`]'s mode (separate by default);
//!   [`Row::m2l`], [`Row::dense_levels`] and [`Row::dense_bytes`] what ran. Auto
//!   (Helmholtz CH3.2d), the rule per level, at the rule's crossover or at
//!   [`Case::m2l_max_degree`] (fused by default); [`Row::m2l_max_degree`] the agreed one.
//!   The exponential M2L (Helmholtz CH6.1e), each level's quadrature by
//!   [`Case::plane_wave`] (the rule by default, or named per level); [`Row::plane_wave`] the
//!   precision each level ran. [`Case::plane_wave_reach`] (Helmholtz CH6.2e) the lateral
//!   reach of the quadratures' α counts (the corner's by default); [`Row::plane_wave_reach`]
//!   the one built and [`Row::plane_wave_waves`] each level's waves per direction.
//!   [`Case::rcr_split`] (Helmholtz CH4.0i) how the Rotation M2L's level calls split over
//!   the threads (the library's `Auto` by default); [`Row::rcr_split`] the setting and
//!   [`Row::rcr_by_pairs`] per level whether a level call ran by pairs;
//!   [`Case::plane_wave_split`] (Helmholtz CH6.2f) how merge and shift splits over the
//!   threads (the library's `Auto` by default); [`Row::plane_wave_split`] the setting and
//!   [`Row::plane_wave_parts`] its thread-seconds by part per level. Auto with the
//!   exponential M2L (Helmholtz CH6.2h, D8): [`Case::auto_plane_wave`], at the rule's
//!   smallest degree or [`Case::m2l_plane_wave_min_degree`]; [`Row::m2l_plane_wave_min_degree`]
//!   the agreed one and [`Row::plane_wave`] the levels with waves.
//!   [`Case::plane_wave_raise`] (Helmholtz CH6.1h) which operators of a level the rule raises
//!   below 10⁻⁶ run at the raised degree (the library's `Rule` by default);
//!   [`Row::plane_wave_raise`] the setting and [`Row::translation_degrees`] each level's
//!   translation degree.
//!   [`Case::diagonal`] (Helmholtz CH4.2b2) the diagonal M2L on the RCR levels
//!   (`HelmholtzFmmBuilder::diagonal`, off by default, the library's until CH4.3);
//!   [`Row::diagonal_classes`] per level the classes it took, [`Row::diagonal_pairs`] the
//!   pairs on each form, [`Row::diagonal_seconds`] the level call's passes and parts,
//!   [`Row::diagonal_conversions`] the conversions computed, [`Row::diagonal_build`] the
//!   tables' build and [`Row::diagonal_multiply_adds`] a conversion's multiply-adds;
//!   [`Case::diagonal_build`] (Helmholtz CH4.2c2) the transfer functions' build
//!   (`HelmholtzFmmBuilder::diagonal_build`, direct or interpolated), [`Row::diagonal_build_kind`]
//!   as built.
//!   [`Case::empty_boxes`] (Helmholtz CH4.0d1) whether the plan leaves out the work of boxes
//!   whose subtree holds no target or no source point (on by default, the library's);
//!   [`Row::left_out`] what it left out and [`Row::remote_multipoles`] the multipoles the
//!   ranks read from other ranks; [`Row::v_pairs`] counts the pairs that ran.
//!   [`Case::near_v`] (Helmholtz CH4.0d2) which V pairs between leaves become near pairs
//!   (off by default, the library's); [`Row::v_near`] the pairs offered and converted,
//!   [`Row::source_points`] the points the source exchange moves, and per level
//!   [`Row::p2p_seconds`] and [`Row::near_interactions`], for the P2P's cost per
//!   interaction.
//!   [`Case::max_points_per_leaf`] and [`Case::c_leaf`] (Helmholtz CH4.0d3) the tree;
//!   [`Row::leaves_by_level`] its leaves, [`Row::kind_seconds`] the other kinds' level calls
//!   per level and [`Row::far_leaf_entries`] the X and W entries, for the leaf size's cost;
//!   [`Case::near_wx`] which W and X pairs between leaves become near pairs (off by default,
//!   the library's), [`Row::wx_near`] the pairs offered and converted and
//!   [`Row::far_interactions`] the P2L and M2P point interactions per level, for the rule's
//!   cost model.
//!   [`Case::v_after_plan`] (Helmholtz CH4.3b2b) whether the V rule's crossovers are decided
//!   again after the plan; [`Row::v_crossovers`] the crossovers before and after it and
//!   [`Row::replanned`] whether the plan was built again (the build part "replan").

use std::ops::Range;
use std::thread;
use std::time::{Duration, Instant};

use mpi::collective::SystemOperation;
use mpi::traits::CommunicatorCollectives;
use nd_fmm_exec::geometry::{Domain, radius};
use nd_fmm_exec::helmholtz::{
    ChargeKind, Charges, DEFAULT_DENSE_BUDGET, Degrees, DiagonalCosts, DiagonalCounts,
    DiagonalTimes, HelmholtzDiagonal, HelmholtzFmm, HelmholtzFmmBuilder, HelmholtzFmmError,
    HelmholtzM2l, LevelShape, Output, PlaneWaveCounts, PlaneWavePrecision, PlaneWaveRaise,
    PlaneWaveSplit, PlaneWaveTimes, RcrSplit, TransferBuild, VAfterPlan, VNear, Wavenumber, WxNear,
    diagonal_rule_classes,
};
use nd_fmm_exec::operator::{M2L_BLOCK_TARGETS, MulAdd, P2pChoice};
use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::diagonal::azimuth_count;
use nd_fmm_ref::helmholtz::p2p::direct_sum;
use nd_fmm_tables::helmholtz::diagonal::{
    DEFAULT_SIGNATURE_NODE_BUDGET, Signatures, scratch_shape,
};
use nd_fmm_tables::helmholtz::plane_wave::Precision;
use nd_fmm_tables::helmholtz::plane_wave::merge::plane_wave_degree;
use nd_octree::PhysicalBox;

use crate::SplitMix64;
use crate::helmholtz_workloads::{SEED, Workload};
use crate::points;

/// One accuracy run: a workload at one k with the degrees, charges and thread counts.
#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    /// The workload (D = 1, so kD is k).
    pub workload: Workload,
    /// k as (Re k, Im k).
    pub k: (f64, f64),
    /// The degrees of the FMM's levels.
    pub degrees: Degrees,
    /// Real or complex charges.
    pub charges: ChargeKind,
    /// The charges' values (Helmholtz CH4.0b; [`Density::Random`], §9.1's, by default).
    pub density: Density,
    /// The number of charge vectors.
    pub vectors: usize,
    /// Whether the FMM computes gradients (and the errors include ∇φ).
    pub gradients: bool,
    /// The number of sampled targets (at most N).
    pub samples: usize,
    /// The seed of the samples and charges.
    pub seed: u64,
    /// The FMM's P2P kernel.
    pub p2p: P2pChoice,
    /// The FMM's M2L.
    pub m2l: HelmholtzM2l,
    /// The bytes of the dense, class or batched M2L tables per rank under
    /// [`HelmholtzM2l::Dense`], [`HelmholtzM2l::Classes`] or [`HelmholtzM2l::Batched`].
    pub dense_budget: usize,
    /// The batched M2L's multiply–add mode (read under [`HelmholtzM2l::Batched`] and
    /// [`HelmholtzM2l::Auto`]; `None` the library's default for the setting).
    pub m2l_mul_add: Option<MulAdd>,
    /// The crossover degree of [`HelmholtzM2l::Auto`] (`None` the rule's).
    pub m2l_max_degree: Option<usize>,
    /// Whether [`HelmholtzM2l::Auto`] takes the exponential M2L
    /// (`HelmholtzFmmBuilder::auto_plane_wave`, Helmholtz CH6.2h; `true` in [`Case::new`], the
    /// library's default since D8); named always, so that the earlier reports' `auto` runs
    /// repeat whatever the library's default.
    pub auto_plane_wave: bool,
    /// The smallest degree at which Auto takes the exponential M2L (`None` the rule's).
    pub m2l_plane_wave_min_degree: Option<usize>,
    /// Whether [`HelmholtzM2l::Auto`] reads the levels' V lists
    /// (`HelmholtzFmmBuilder::batched_term`, Helmholtz CH4.0; `true` in [`Case::new`], the
    /// library's default); named always, as `auto_plane_wave`.
    pub batched_term: bool,
    /// The width of Batched's blocks (`HelmholtzFmmBuilder::m2l_block_targets`, Helmholtz
    /// CH4.0; `None` the library's split). No bit depends on it.
    pub m2l_block_targets: Option<usize>,
    /// The precision of each level's quadrature under [`HelmholtzM2l::PlaneWave`]
    /// (Helmholtz CH6.1e).
    pub plane_wave: PlaneWavePrecision,
    /// The bytes per node of its waves buffer (`HelmholtzFmmBuilder::plane_wave_budget`,
    /// Helmholtz CH6.2a; `None` the library's default, `usize::MAX` a whole level).
    pub plane_wave_budget: Option<usize>,
    /// The lateral reach of the quadratures' α counts under [`HelmholtzM2l::PlaneWave`]
    /// (`HelmholtzFmmBuilder::plane_wave_reach`, Helmholtz CH6.2e; `None` the library's, the
    /// region's corner 4√2).
    pub plane_wave_reach: Option<f64>,
    /// How merge and shift splits over the threads under [`HelmholtzM2l::PlaneWave`]
    /// (`HelmholtzFmmBuilder::plane_wave_split`, Helmholtz CH6.2f; `None` the library's,
    /// `Auto`). No bit depends on it.
    pub plane_wave_split: Option<PlaneWaveSplit>,
    /// Which operators of a level the rule raises below 10⁻⁶ run at the raised degree
    /// (`HelmholtzFmmBuilder::plane_wave_raise`, Helmholtz CH6.1h; `None` the library's,
    /// `Rule`).
    pub plane_wave_raise: Option<PlaneWaveRaise>,
    /// The evaluations of one build (at least 1; Helmholtz CH6.2b): the row's times and
    /// errors are the last one's.
    pub evaluations: usize,
    /// The FMM's domain `[x_min, y_min, z_min, x_max, y_max, z_max]`
    /// (`HelmholtzFmmBuilder::domain`, Helmholtz CH4.0b; `None` the library's, the points'
    /// bounding cube).
    pub domain: Option<[f64; 6]>,
    /// The points per leaf (`HelmholtzFmmBuilder::max_points_per_leaf`, Helmholtz CH4.0b;
    /// `None` the library's); above N, a tree of leaves on the leaf rule's level alone.
    pub max_points_per_leaf: Option<usize>,
    /// The maximum leaf width in wavelengths (`HelmholtzFmmBuilder::c_leaf`, Helmholtz
    /// CH4.0d3; `None` the library's, `DEFAULT_C_LEAF`).
    pub c_leaf: Option<f64>,
    /// Whether the plan leaves out the work of empty boxes
    /// (`HelmholtzFmmBuilder::empty_boxes`, Helmholtz CH4.0d1; `true` in [`Case::new`], the
    /// library's default); named always, as `auto_plane_wave`.
    pub empty_boxes: bool,
    /// Which V pairs between leaves become near pairs (`HelmholtzFmmBuilder::near_v`,
    /// Helmholtz CH4.0d2; [`VNear::Off`] in [`Case::new`], the library's default before
    /// Helmholtz CH4.0d3, `VNear::CostModel` since; named always, so a case's pairs do not
    /// move with the library's default).
    pub near_v: VNear,
    /// Which W and X pairs between leaves become near pairs (`HelmholtzFmmBuilder::near_wx`,
    /// Helmholtz CH4.0d3; [`WxNear::Off`] in [`Case::new`]; the library's default is
    /// `WxNear::CostModel`; named always, as `near_v`).
    pub near_wx: WxNear,
    /// The bytes of the M2L's stored rotation blocks per rank
    /// (`HelmholtzFmmBuilder::block_budget`, Helmholtz CH4.0e; `None` the library's,
    /// `DEFAULT_BLOCK_BUDGET`). A level whose blocks do not fit forms them per V pair; no bit
    /// depends on it.
    pub block_budget: Option<usize>,
    /// How the Rotation M2L's level calls split over the threads
    /// (`HelmholtzFmmBuilder::rcr_split`, Helmholtz CH4.0i; `None` the library's, `Auto`). No
    /// bit depends on it.
    pub rcr_split: Option<RcrSplit>,
    /// The diagonal M2L on the RCR levels (`HelmholtzFmmBuilder::diagonal`, Helmholtz
    /// CH4.2b2; [`HelmholtzDiagonal::Off`] in [`Case::new`], the library's default until
    /// CH4.3; named always, as `near_v`).
    pub diagonal: HelmholtzDiagonal,
    /// The diagonal M2L's transfer functions' build (`HelmholtzFmmBuilder::diagonal_build`,
    /// Helmholtz CH4.2c2; [`TransferBuild::Direct`] in [`Case::new`], named always, so that
    /// CH4.2b2's runs repeat; the library's default is interpolated since CH4.2c2's sign-off).
    pub diagonal_build: TransferBuild,
    /// The diagonal form's time rule's costs (`HelmholtzFmmBuilder::diagonal_costs`, Helmholtz
    /// CH4.3b1; `None` in [`Case::new`], the machine class's; [`FORM_COSTS`] and [`RCR_COSTS`]
    /// force one option, Helmholtz CH4.3b2a).
    pub diagonal_costs: Option<DiagonalCosts>,
    /// Whether the V rule's crossovers are decided again after the plan
    /// (`HelmholtzFmmBuilder::v_after_plan`, Helmholtz CH4.3b2b; [`VAfterPlan::Off`] in
    /// [`Case::new`], the library's default).
    pub v_after_plan: VAfterPlan,
}

/// Time-rule costs under which the diagonal form beats every other M2L wherever a class passes
/// the bound (Helmholtz CH4.3b2a): the form free, RCR 1 per (p + 1)² + (p + 1)³, the waves 10¹²
/// per box. Under `HelmholtzDiagonal::Auto` each level the rule reaches with a class that passes
/// then runs the form at its degree (a level with waves at its resolving degree), for timing it.
pub const FORM_COSTS: DiagonalCosts = DiagonalCosts {
    s2f: [0.0, 0.0],
    f2r: [0.0, 0.0],
    pair: [0.0, 0.0],
    rotation: [1.0, 1.0],
    plane_wave: [1e12, 1e12],
};

/// Time-rule costs under which RCR beats the form and the form the waves (Helmholtz CH4.3b2a):
/// the diagonal pair dear, the waves dearer, RCR free. Under `HelmholtzDiagonal::Auto` each level
/// the rule reaches with a class that passes then runs RCR at its degree (the rule never puts RCR
/// against the waves alone).
pub const RCR_COSTS: DiagonalCosts = DiagonalCosts {
    pair: [1e9, 1e9],
    rotation: [0.0, 0.0],
    ..FORM_COSTS
};

impl Case {
    /// A case with complex charges, one vector, gradients and 1,000 samples (§9.1), seeded
    /// with the workloads' [`SEED`], the P2P kernel of [`P2pChoice::Auto`] and M2L by
    /// [`HelmholtzM2l::Rotation`] (the dense budget [`DEFAULT_DENSE_BUDGET`]; under
    /// [`HelmholtzM2l::PlaneWave`] the precisions of [`PlaneWavePrecision::Rule`]).
    pub fn new(workload: Workload, k: (f64, f64), degrees: Degrees) -> Self {
        Self {
            workload,
            k,
            degrees,
            charges: ChargeKind::Complex,
            density: Density::Random,
            vectors: 1,
            gradients: true,
            samples: 1000,
            seed: SEED,
            p2p: P2pChoice::Auto,
            m2l: HelmholtzM2l::Rotation,
            dense_budget: DEFAULT_DENSE_BUDGET,
            m2l_mul_add: None,
            m2l_max_degree: None,
            auto_plane_wave: true,
            m2l_plane_wave_min_degree: None,
            batched_term: true,
            m2l_block_targets: None,
            plane_wave: PlaneWavePrecision::Rule,
            plane_wave_budget: None,
            plane_wave_reach: None,
            plane_wave_split: None,
            plane_wave_raise: None,
            evaluations: 1,
            domain: None,
            max_points_per_leaf: None,
            c_leaf: None,
            empty_boxes: true,
            near_v: VNear::Off,
            near_wx: WxNear::Off,
            block_budget: None,
            rcr_split: None,
            diagonal: HelmholtzDiagonal::Off,
            diagonal_build: TransferBuild::Direct,
            diagonal_costs: None,
            v_after_plan: VAfterPlan::Off,
        }
    }
}

/// The values of the charges (Helmholtz CH4.0b; review 2026-11-12, M5): §9.1's random
/// charges, or a smooth density, whose far field is coherent and dominates the norms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Density {
    /// §9.1's: per vector Re q and Im q uniform in [−1, 1).
    Random,
    /// The trace of a plane wave, q_j = e^{i Re(k) d·x_j} with d = [`PLANE_WAVE_DIRECTION`],
    /// the same for every vector; complex charges only.
    PlaneWave,
    /// q_j = 1 for every point and vector.
    Constant,
}

impl Density {
    /// A short name: `random`, `planewave`, `constant`.
    pub fn name(&self) -> &'static str {
        match self {
            Density::Random => "random",
            Density::PlaneWave => "planewave",
            Density::Constant => "constant",
        }
    }

    /// The density of a [`name`](Self::name).
    pub fn from_name(name: &str) -> Option<Self> {
        [Density::Random, Density::PlaneWave, Density::Constant]
            .into_iter()
            .find(|d| d.name() == name)
    }
}

/// The direction d of [`Density::PlaneWave`], a unit vector off the axes and the diagonals.
pub const PLANE_WAVE_DIRECTION: [f64; 3] = [0.36, 0.48, 0.8];

/// The points, sampled targets and charge vectors of a [`Case`].
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    /// The workload's points.
    pub points: Vec<[f64; 3]>,
    /// The positions in `points` of the sampled targets, distinct.
    pub sample: Vec<usize>,
    /// Per vector, Re q at every point.
    pub re: Vec<Vec<f64>>,
    /// Per vector, Im q at every point (empty vectors for real charges).
    pub im: Vec<Vec<f64>>,
}

impl Problem {
    /// Draws the problem: the workload's points, the sample (a partial Fisher–Yates
    /// shuffle from the generator of `case.seed`, at most N targets) and charge vector v
    /// from the generator of `case.seed + 1 + v`, or by [`Case::density`].
    ///
    /// # Panics
    ///
    /// For [`Density::PlaneWave`] with real charges.
    pub fn new(case: &Case) -> Self {
        let points = case.workload.points(case.k.0);
        let n = points.len();
        let mut rng = SplitMix64::new(case.seed);
        let mut order: Vec<usize> = (0..n).collect();
        let samples = case.samples.min(n);
        for i in 0..samples {
            let j = i + ((((rng.next_u64() >> 32) * (n - i) as u64) >> 32) as usize);
            order.swap(i, j);
        }
        order.truncate(samples);
        let (mut re, mut im) = (Vec::new(), Vec::new());
        let complex = case.charges == ChargeKind::Complex;
        for v in 0..case.vectors as u64 {
            match case.density {
                Density::Random => {
                    let mut rng = SplitMix64::new(case.seed + 1 + v);
                    re.push(points::charges(&mut rng, n));
                    im.push(if complex {
                        points::charges(&mut rng, n)
                    } else {
                        Vec::new()
                    });
                }
                Density::PlaneWave => {
                    assert!(complex, "a plane wave's trace needs complex charges");
                    let phase = |x: &[f64; 3]| {
                        case.k.0 * (0..3).map(|a| PLANE_WAVE_DIRECTION[a] * x[a]).sum::<f64>()
                    };
                    re.push(points.iter().map(|x| phase(x).cos()).collect());
                    im.push(points.iter().map(|x| phase(x).sin()).collect());
                }
                Density::Constant => {
                    re.push(vec![1.0; n]);
                    im.push(if complex { vec![0.0; n] } else { Vec::new() });
                }
            }
        }
        Self {
            points,
            sample: order,
            re,
            im,
        }
    }

    /// The number of points N.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the problem has no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The positions of the points that rank `rank` of `size` passes: a contiguous block,
    /// the first N mod P ranks one point more.
    pub fn block(&self, rank: usize, size: usize) -> Range<usize> {
        let (n, size) = (self.len(), size.max(1));
        let (base, extra) = (n / size, n % size);
        let start = rank * base + rank.min(extra);
        start..start + base + usize::from(rank < extra)
    }

    /// Charge vector v as complex numbers in f64 (Im q = 0 for real charges).
    pub fn complex_charges(&self, v: usize) -> Vec<Complex<f64>> {
        let im = &self.im[v];
        self.re[v]
            .iter()
            .enumerate()
            .map(|(i, &re)| Complex::new(re, im.get(i).copied().unwrap_or(0.0)))
            .collect()
    }
}

/// The exact potentials and gradients at the sampled targets, per vector, in f64, with
/// 1/(4π).
#[derive(Clone, Debug, PartialEq)]
pub struct Oracle {
    /// φ per vector and sampled target.
    pub potential: Vec<Vec<Complex<f64>>>,
    /// ∇φ per vector and sampled target (empty without gradients).
    pub gradient: Vec<Vec<[Complex<f64>; 3]>>,
}

impl Oracle {
    /// The oracle of `problem` at k, at the sampled targets with positions `targets` in
    /// `problem.sample`, on `threads` threads (each a contiguous share of the targets, so
    /// the values do not depend on the thread count). O(m N n_targets).
    pub fn of_samples(
        k: Complex<f64>,
        problem: &Problem,
        targets: &[usize],
        gradients: bool,
        threads: usize,
    ) -> Self {
        let points: Vec<[f64; 3]> = targets
            .iter()
            .map(|&s| problem.points[problem.sample[s]])
            .collect();
        let chunk = points.len().div_ceil(threads.max(1)).max(1);
        let (mut potential, mut gradient) = (Vec::new(), Vec::new());
        for v in 0..problem.re.len() {
            let charges = problem.complex_charges(v);
            let mut phi = vec![Complex::new(0.0, 0.0); points.len()];
            let mut grad =
                vec![[Complex::new(0.0, 0.0); 3]; if gradients { points.len() } else { 0 }];
            thread::scope(|scope| {
                let mut grad_chunks = grad.chunks_mut(chunk);
                for (x, phi) in points.chunks(chunk).zip(phi.chunks_mut(chunk)) {
                    let grad = grad_chunks.next();
                    let (sources, charges) = (&problem.points, &charges);
                    scope.spawn(move || direct_sum(k, sources, charges, x, phi, grad));
                }
            });
            // The direct sum leaves out 1/(4π); the FMM's output includes it (§3.15).
            let scale = 1.0 / (4.0 * std::f64::consts::PI);
            potential.push(phi.into_iter().map(|v| v * scale).collect());
            gradient.push(grad.into_iter().map(|g| g.map(|v| v * scale)).collect());
        }
        Self {
            potential,
            gradient,
        }
    }

    /// [`of_samples`](Self::of_samples) of every sampled target, computed once over the
    /// ranks of `comm`: each rank every P-th sample from its rank, and all-reduces (sums,
    /// zeros elsewhere) give every rank the whole oracle, the values of one rank bit for
    /// bit.
    ///
    /// # Collective operation
    ///
    /// On `comm`, every rank with the same arguments.
    pub fn sharded<C: CommunicatorCollectives>(
        k: Complex<f64>,
        problem: &Problem,
        gradients: bool,
        threads: usize,
        comm: &C,
    ) -> Self {
        let (rank, size) = (comm.rank() as usize, comm.size() as usize);
        let n = problem.sample.len();
        let mine: Vec<usize> = (rank..n).step_by(size).collect();
        let part = Self::of_samples(k, problem, &mine, gradients, threads);
        let (mut potential, mut gradient) = (Vec::new(), Vec::new());
        for v in 0..part.potential.len() {
            let width = if gradients { 8 } else { 2 };
            let mut local = vec![0.0f64; width * n];
            for (j, &s) in mine.iter().enumerate() {
                let phi = part.potential[v][j];
                local[width * s] = phi.re;
                local[width * s + 1] = phi.im;
                if gradients {
                    for (a, g) in part.gradient[v][j].iter().enumerate() {
                        local[width * s + 2 + 2 * a] = g.re;
                        local[width * s + 3 + 2 * a] = g.im;
                    }
                }
            }
            let mut all = vec![0.0f64; width * n];
            comm.all_reduce_into(&local[..], &mut all[..], SystemOperation::sum());
            let values = all.chunks_exact(width);
            potential.push(values.clone().map(|c| Complex::new(c[0], c[1])).collect());
            gradient.push(if gradients {
                values
                    .map(|c| std::array::from_fn(|a| Complex::new(c[2 + 2 * a], c[3 + 2 * a])))
                    .collect()
            } else {
                Vec::new()
            });
        }
        Self {
            potential,
            gradient,
        }
    }
}

/// A 64-bit mix (SplitMix64's finaliser).
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// This rank's part of the rank-invariant checksum of `output` (module documentation),
/// for targets whose global positions start at `first`: the wrapping sum over every value
/// of mix(bits + mix(slot + mix(i))), i the global position and slot the value's place in
/// the target's output (φ columns first, then ∇φ by column and axis). The ranks' parts
/// add up, wrapping, to the checksum of the whole output.
pub fn checksum(output: &Output<f64>, first: usize) -> u64 {
    let n = output.ntargets();
    let columns = output.potential.len() / n.max(1);
    let mut sum = 0u64;
    for i in 0..n {
        let point = mix(SEED ^ (first + i) as u64);
        for o in 0..columns {
            let value = output.potential[o * n + i];
            sum = sum.wrapping_add(mix(value.to_bits().wrapping_add(mix(point ^ o as u64))));
            if let Some(gradient) = &output.gradient {
                for (a, g) in gradient[o * n + i].iter().enumerate() {
                    let slot = (columns + 3 * o + a) as u64;
                    sum = sum.wrapping_add(mix(g.to_bits().wrapping_add(mix(point ^ slot))));
                }
            }
        }
    }
    sum
}

/// The wrapping sum of every rank's `part`.
///
/// # Collective operation
///
/// On `comm`.
pub fn reduce_checksum<C: CommunicatorCollectives>(part: u64, comm: &C) -> u64 {
    let mut parts = vec![0u64; comm.size() as usize];
    comm.all_gather_into(&part, &mut parts[..]);
    parts.into_iter().fold(0, u64::wrapping_add)
}

/// The relative ℓ2 errors of `output` at the sampled targets this rank holds, per vector,
/// [potential, gradient]: the sums of squares of error and reference, summed over the
/// ranks, and their ratios. The last entry is over all vectors together.
///
/// # Collective operation
///
/// On `comm`.
pub fn errors<C: CommunicatorCollectives>(
    output: &Output<f64>,
    problem: &Problem,
    block: &Range<usize>,
    oracle: &Oracle,
    comm: &C,
) -> Vec<[f64; 2]> {
    let vectors = output.vectors();
    // Per vector [φ error², φ reference², ∇φ error², ∇φ reference²].
    let mut sums = vec![0.0f64; 4 * vectors];
    for (s, &position) in problem.sample.iter().enumerate() {
        if !block.contains(&position) {
            continue;
        }
        let i = position - block.start;
        for v in 0..vectors {
            let phi = output.potential_of(v);
            let approx = Complex::new(phi.re[i], phi.im.map_or(0.0, |im| im[i]));
            let exact = oracle.potential[v][s];
            sums[4 * v] += (approx - exact).norm_sqr();
            sums[4 * v + 1] += exact.norm_sqr();
            if let Some(grad) = output.gradient_of(v) {
                for a in 0..3 {
                    let approx = Complex::new(grad.re[i][a], grad.im.map_or(0.0, |im| im[i][a]));
                    let exact = oracle.gradient[v][s][a];
                    sums[4 * v + 2] += (approx - exact).norm_sqr();
                    sums[4 * v + 3] += exact.norm_sqr();
                }
            }
        }
    }
    let mut all = vec![0.0f64; 4 * vectors];
    comm.all_reduce_into(&sums[..], &mut all[..], SystemOperation::sum());
    let ratio = |e: f64, r: f64| if r > 0.0 { (e / r).sqrt() } else { 0.0 };
    let mut rows: Vec<[f64; 2]> = all
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| [ratio(c[0], c[1]), ratio(c[2], c[3])])
        .collect();
    let total = |j: usize| all.iter().skip(j).step_by(4).sum::<f64>();
    rows.push([ratio(total(0), total(1)), ratio(total(2), total(3))]);
    rows
}

/// The crossovers of [`HelmholtzM2l::Auto`] that choose differently on a tree of `degrees`
/// (Helmholtz CH3.2d's sweep): `usize::MAX` (every level with tables), then each distinct
/// p_l of the levels with M2L (≥ 2) but the largest, descending, then one below the
/// smallest (Rotation on every level). Empty without a level of M2L.
pub fn sweep_crossovers(degrees: &[usize]) -> Vec<usize> {
    let mut distinct: Vec<usize> = degrees.iter().skip(2).copied().collect();
    distinct.sort_unstable();
    distinct.dedup();
    let Some(&smallest) = distinct.first() else {
        return Vec::new();
    };
    distinct.pop();
    std::iter::once(usize::MAX)
        .chain(distinct.into_iter().rev())
        .chain(smallest.checked_sub(1))
        .collect()
}

/// The outcome of one [`Case`] at one thread count.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// The rayon threads per rank.
    pub threads: usize,
    /// The P2P kernel that ran (never `Auto`).
    pub p2p: P2pChoice,
    /// The M2L setting.
    pub m2l: HelmholtzM2l,
    /// The batched M2L's multiply–add mode.
    pub m2l_mul_add: MulAdd,
    /// The agreed crossover degree under [`HelmholtzM2l::Auto`], else `None`.
    pub m2l_max_degree: Option<usize>,
    /// The agreed smallest degree of Auto's exponential M2L, `None` unless
    /// [`Case::auto_plane_wave`] (Helmholtz CH6.2h).
    pub m2l_plane_wave_min_degree: Option<usize>,
    /// Per level, whether its M2L ran the setting's tables, Dense, Classes, Batched or the
    /// exponential M2L (else RCR).
    pub dense_levels: Vec<bool>,
    /// The bytes of the dense, class or batched M2L tables per rank, plus under
    /// [`HelmholtzM2l::PlaneWave`] (or Auto with [`Case::auto_plane_wave`]) the exponential
    /// M2L's tables and buffers (`plane_wave_bytes`).
    pub dense_bytes: usize,
    /// Per level, the precision of its quadrature under [`HelmholtzM2l::PlaneWave`] (and
    /// under Auto with [`Case::auto_plane_wave`]), `None` where it ran another M2L.
    pub plane_wave: Vec<Option<Precision>>,
    /// The bytes of the exponential M2L's waves buffer per rank (part of `dense_bytes`;
    /// Helmholtz CH6.2a).
    pub plane_wave_buffer: usize,
    /// The exponential M2L's level calls, chunks and waves computed in the evaluation, this
    /// rank's (Helmholtz CH6.2a).
    pub plane_wave_counts: PlaneWaveCounts,
    /// The lateral reach of the quadratures' α counts as built (Helmholtz CH6.2e).
    pub plane_wave_reach: f64,
    /// How merge and shift split over the threads (Helmholtz CH6.2f).
    pub plane_wave_split: PlaneWaveSplit,
    /// Which operators of a raised level ran at the raised degree (Helmholtz CH6.1h).
    pub plane_wave_raise: PlaneWaveRaise,
    /// Per level, the degree of its translations to and from its parent level
    /// (`HelmholtzOperator::translation_degree`): `degrees`' but on the levels
    /// [`PlaneWaveRaise::Leaves`] and [`PlaneWaveRaise::M2l`] keep at the degree given
    /// (Helmholtz CH6.1h).
    pub translation_degrees: Vec<usize>,
    /// Per level, its waves per direction Σₖ Nₖ under [`HelmholtzM2l::PlaneWave`], 0 where
    /// it ran RCR (and under every other setting; Helmholtz CH6.2e).
    pub plane_wave_waves: Vec<usize>,
    /// The ranks.
    pub ranks: usize,
    /// The levels of the tree.
    pub levels: usize,
    /// The leaf rule's minimum leaf level.
    pub min_leaf_level: usize,
    /// The degree of every level.
    pub degrees: Vec<usize>,
    /// The circumradius √3 r_l of every level's boxes in the domain of the FMM (the
    /// points' bounding cube unless named), for x = Re(k) a_l (Helmholtz CH3.3).
    pub circumradii: Vec<f64>,
    /// The leaves of the tree, summed over the ranks.
    pub leaves: usize,
    /// The most points in a leaf.
    pub max_leaf_points: usize,
    /// The octree's refinement target as built (`HelmholtzFmm::max_points_per_leaf`;
    /// Helmholtz CH4.0d3).
    pub max_points_per_leaf: usize,
    /// The maximum leaf width in wavelengths as built (`HelmholtzFmm::c_leaf`; Helmholtz
    /// CH4.0d3).
    pub c_leaf: f64,
    /// Per level, its leaves and the leaves holding a source or a target, summed over the
    /// ranks (`HelmholtzFmm::leaves_by_level`; Helmholtz CH4.0d3).
    pub leaves_by_level: Vec<[usize; 2]>,
    /// Per level, the level calls' seconds of the kinds of
    /// `nd_fmm_exec::helmholtz::TIMED_KINDS` (P2M, M2M, L2L, L2P, P2L, M2P) in the last
    /// evaluation, each the slowest rank's (`HelmholtzOperator::kind_times`; Helmholtz
    /// CH4.0d3).
    pub kind_seconds: Vec<[f64; 6]>,
    /// Per level, the plan's X entries (P2L, leaves of the next coarser level) and W entries
    /// (M2P, boxes of the next finer level), summed over the ranks (Helmholtz CH4.0d3).
    pub far_leaf_entries: Vec<[usize; 2]>,
    /// Per level, the point interactions of P2L and M2P, [Σ_X n_s, Σ_W n_t], summed over the
    /// ranks (`HelmholtzFmm::far_interactions`; Helmholtz CH4.0d3).
    pub far_interactions: Vec<[u64; 2]>,
    /// The W and X pairs' near rule as built (Helmholtz CH4.0d3).
    pub near_wx: WxNear,
    /// The W pairs between leaves the rule offered and converted, and the X pairs, summed
    /// over the ranks (`NearPairs`' W and X fields; Helmholtz CH4.0d3).
    pub wx_near: [usize; 4],
    /// The build, seconds (slowest rank).
    pub build: f64,
    /// The build by part, seconds, each the slowest rank's, in the order of [`BUILD_PARTS`]
    /// (`HelmholtzFmm::build_timings`; Helmholtz CH4.0h). The operator includes its tables.
    pub build_parts: [f64; BUILD_PARTS.len()],
    /// The evaluation, seconds (slowest rank); the last of [`Case::evaluations`].
    pub evaluate: f64,
    /// The first evaluation, seconds (slowest rank); `evaluate` with one evaluation.
    pub first_evaluate: f64,
    /// Per level, the M2L level calls' seconds in the last evaluation, the slowest rank's
    /// (`HelmholtzOperator::m2l_times`; Helmholtz CH6.2b).
    pub m2l_seconds: Vec<f64>,
    /// Per level, whether the M2L's rotation blocks are stored
    /// (`HelmholtzTables::m2l_blocks_stored`, Helmholtz CH4.0e); else they are formed per V
    /// pair. The same on every rank (the tables follow the global degrees and the budget).
    pub m2l_blocks_stored: Vec<bool>,
    /// How the Rotation M2L's level calls split, as built (`HelmholtzOperator::rcr_split`;
    /// Helmholtz CH4.0i).
    pub rcr_split: RcrSplit,
    /// Per level, whether the M2L's level calls of the last evaluation ran by pairs on some
    /// rank (`HelmholtzOperator::rcr_counts`; Helmholtz CH4.0i).
    pub rcr_by_pairs: Vec<bool>,
    /// Per level, the exponential M2L's passes in the last evaluation, [waves, merge and
    /// shift] seconds, each the slowest rank's (`HelmholtzOperator::plane_wave_times`;
    /// Helmholtz CH6.2d); zero where no level call ran merge and shift.
    pub plane_wave_seconds: Vec<[f64; 2]>,
    /// Per level, merge and shift's thread-seconds in the last evaluation, [busy, shifts,
    /// children, conversions], each the slowest rank's (`PlaneWaveTimes`' parts; Helmholtz
    /// CH6.2f): busy over merge and shift's seconds is the threads it kept busy.
    pub plane_wave_parts: Vec<[f64; 4]>,
    /// Per level, the V-list pairs of the plan, each target counted on the rank that owns it
    /// and summed over the ranks: the same on every rank count (Helmholtz CH6.2b).
    pub v_pairs: Vec<usize>,
    /// Per level, its boxes, each counted on the rank that owns it and summed over the ranks
    /// (Helmholtz CH4.0): with `v_pairs` the level's `LevelShape`.
    pub v_targets: Vec<usize>,
    /// Per level, the estimate m̂ of Batched's columns per table read from the level's
    /// shape (`LevelShape::batched_columns`, Helmholtz CH4.0), which Auto's term reads.
    pub batched_columns: Vec<f64>,
    /// Per level, the (block, offset) visits of Batched's level call at blocks of 64 rows,
    /// counted on each rank over the targets it owns and summed over the ranks (Helmholtz
    /// CH4.0): on one rank the level call's own count, so `v_pairs` over it is the exact
    /// columns per table read that m̂ estimates.
    pub block_visits: Vec<usize>,
    /// Whether Auto read the levels' V lists (Helmholtz CH4.0).
    pub batched_term: bool,
    /// The width of Batched's blocks if named (Helmholtz CH4.0).
    pub m2l_block_targets: Option<usize>,
    /// Whether the plan had the empty-box rule (Helmholtz CH4.0d1).
    pub empty_boxes: bool,
    /// What the empty-box rule left out, summed over the ranks, in the order of
    /// `nd_fmm_plan::empty::EmptyBoxes`' fields: boxes without targets, their V, X and W
    /// entries and L2L rows, then the V, W and M2M entries naming a box without sources (all
    /// zero with the rule off). A box with rows on several ranks (a `Global` box) counts on
    /// each.
    pub left_out: [usize; 8],
    /// The multipoles each rank reads from other ranks (the boxes its V and W rows name that
    /// it does not compute, each once), summed over the ranks: what the reduction, the
    /// replicated top and the multipole exchange deliver (Helmholtz CH4.0d1).
    pub remote_multipoles: usize,
    /// Per level, the diagonal form's scratch if every V pair of the level ran it (an upper
    /// bound: CH4.3's rule takes fewer pairs; Helmholtz CH4.1b), each rank's under its share
    /// of `DEFAULT_SIGNATURE_NODE_BUDGET` (over `HelmholtzFmm::ranks_per_node`) at its degree
    /// and threads, the largest over the ranks.
    pub signature_scratch: Vec<SignatureScratch>,
    /// The V pairs' near rule as built (Helmholtz CH4.0d2).
    pub near_v: VNear,
    /// The V pairs between leaves the rule offered and those it converted, summed over the
    /// ranks (zero offered with the rule off; `NearPairs`' V fields).
    pub v_near: [usize; 2],
    /// The source points the source exchange moves per evaluation, [sent, received], summed
    /// over the ranks (`HelmholtzFmm::source_points_exchanged`; Helmholtz CH4.0d2).
    pub source_points: [usize; 2],
    /// Per level, the P2P level calls' seconds in the last evaluation, the slowest rank's
    /// (`HelmholtzOperator::p2p_times`; Helmholtz CH4.0d2).
    pub p2p_seconds: Vec<f64>,
    /// Per level, the near interactions Σ_t n_t Σ_s n_s of its P2P, summed over the ranks
    /// (`HelmholtzFmm::near_interactions`; Helmholtz CH4.0d2).
    pub near_interactions: Vec<u64>,
    /// The diagonal M2L's setting as built (Helmholtz CH4.2b2).
    pub diagonal: HelmholtzDiagonal,
    /// The diagonal M2L's transfer functions' build as built (Helmholtz CH4.2c2;
    /// `HelmholtzOperator::diagonal_build`).
    pub diagonal_build_kind: TransferBuild,
    /// Per level, the offset classes that took the diagonal form (0 where the level has none;
    /// `HelmholtzOperator::diagonal_classes`), the same on every rank.
    pub diagonal_classes: Vec<usize>,
    /// Per level, the V pairs of the last evaluation's diagonal level calls, [diagonal, RCR],
    /// summed over the ranks (`DiagonalCounts`).
    pub diagonal_pairs: Vec<[usize; 2]>,
    /// Per level, the conversions of the last evaluation's diagonal level calls, [S → F
    /// (signatures), F → R (accumulators converted back)], summed over the ranks.
    pub diagonal_conversions: Vec<[usize; 2]>,
    /// Per level, the last evaluation's diagonal level calls' times, [S → F pass, targets'
    /// pass, S → F, pairs, F → R, RCR pairs] seconds (the passes wall, the parts
    /// thread-seconds), each the slowest rank's (`HelmholtzOperator::diagonal_times`).
    pub diagonal_seconds: Vec<[f64; 6]>,
    /// Per level, the build of its transfer functions and of its conversions, seconds, the
    /// slowest rank's (`HelmholtzOperator::diagonal_build_time`; zero without the form).
    pub diagonal_build: Vec<[f64; 2]>,
    /// Per level, one conversion's real multiply-adds (`HelmholtzOperator::
    /// diagonal_multiply_adds`; 0 without the form).
    pub diagonal_multiply_adds: Vec<u64>,
    /// Per level, the diagonal form's time rule as the operator sees it (Helmholtz CH4.3b1):
    /// [k a = Re(k) √3 r_l, the bound's classes at the FMM's tolerance and the build (at the
    /// level's degree as run, the resolving degree on a level the waves raised, which the time
    /// rule keeps since Helmholtz CH4.3b2a; 0 without a tolerance), and the model's seconds of
    /// the level call (thread-ns of `HelmholtzOperator::diagonal_costs` over the ranks ×
    /// threads) with those classes diagonal and by RCR alone, both at that degree, and by the
    /// exponential M2L at its resolving degree (the degree as run where raised, else
    /// `plane_wave_degree(p_l, ε)`)], each on the level's agreed shape
    /// (`HelmholtzOperator::diagonal_shapes`).
    pub diagonal_model: Vec<[f64; 5]>,
    /// The time rule's costs as built, agreed over the ranks (`HelmholtzOperator::diagonal_costs`;
    /// Helmholtz CH4.3b2a), whatever the setting.
    pub diagonal_costs: DiagonalCosts,
    /// Per level, whether the time rule took it from the exponential M2L
    /// (`HelmholtzOperator::taken_from_waves`; Helmholtz CH4.3b2a), the same on every rank.
    pub taken_from_waves: Vec<bool>,
    /// The V rule after the plan as built (`HelmholtzFmm::v_after_plan`; Helmholtz CH4.3b2b).
    pub v_after_plan: VAfterPlan,
    /// Per level, the V rule's crossover [before the plan, after it] (`HelmholtzFmm::
    /// v_crossovers` and `v_crossovers_after_plan`, the second the first where nothing was
    /// decided after the plan), agreed over the ranks.
    pub v_crossovers: Vec<[f64; 2]>,
    /// Whether a crossover rose after the plan, so that the plan was built again
    /// (`HelmholtzFmm::replanned`).
    pub replanned: bool,
    /// The relative ℓ2 errors [φ, ∇φ] per vector, then over all vectors.
    pub errors: Vec<[f64; 2]>,
    /// The checksum of the whole output.
    pub checksum: u64,
}

impl Row {
    /// The width of the widest leaf holding a point, in the FMM's domain (Helmholtz
    /// CH4.0d3; from the level's circumradius √3 r_l, w = 2 r_l).
    pub fn widest_occupied_leaf(&self) -> f64 {
        self.leaves_by_level
            .iter()
            .position(|&[_, occupied]| occupied > 0)
            .map_or(0.0, |l| 2.0 * self.circumradii[l] / 3f64.sqrt())
    }

    /// A bound on the phase Re(k) r of the P2P's pairs (Helmholtz CH4.0d3; proposal §5.7,
    /// CH3.4b's contract range): a near pair's leaves are adjacent, or V leaves of one level
    /// made near (`near_v`, at most three boxes apart per axis), and no leaf holding a point
    /// is wider than [`widest_occupied_leaf`](Self::widest_occupied_leaf) w, so r ≤ 2√3 w,
    /// or 4√3 w with V pairs converted. A converted W or X pair (`near_wx`) joins a leaf to
    /// a finer one inside a neighbour of its width, so its points lie within 2√3 times the
    /// coarser leaf's width and it adds nothing to the bound.
    pub fn phase_bound(&self, re_k: f64) -> f64 {
        let span = if self.v_near[1] > 0 { 4.0 } else { 2.0 };
        re_k * span * 3f64.sqrt() * self.widest_occupied_leaf()
    }
}

/// The parts of [`Row::build_parts`]: the builder's steps (`HelmholtzBuildTimings`), then the
/// operator's tables by part (`TableTimings`, within "operator").
pub const BUILD_PARTS: [&str; 16] = [
    "settings",
    "domain",
    "octree",
    "routes",
    "counts",
    "plan",
    "shapes",
    "operator",
    "RCR tables",
    "matrices",
    "plane-wave tables",
    "diagonal tables",
    "scratch",
    "replan",
    "evaluator",
    "load",
];

/// The slowest rank's `seconds`.
fn slowest<C: CommunicatorCollectives>(seconds: f64, comm: &C) -> f64 {
    let mut max = 0.0f64;
    comm.all_reduce_into(&seconds, &mut max, SystemOperation::max());
    max
}

/// Builds and evaluates the FMM of `case` on `problem` at `threads` threads per rank and
/// measures it against `oracle` (module documentation).
///
/// # Collective operation
///
/// On `comm`, every rank with the same arguments.
///
/// # Errors
///
/// The FMM's build and evaluation errors.
///
/// # Panics
///
/// If `case.k` is not a valid wavenumber.
pub fn run<C: CommunicatorCollectives>(
    case: &Case,
    problem: &Problem,
    oracle: &Oracle,
    threads: usize,
    comm: &C,
) -> Result<Row, HelmholtzFmmError> {
    run_output(case, problem, oracle, threads, comm).map(|run| run.row)
}

/// A [`Row`] with what it was measured on (Helmholtz CH3.6, the full-target crosscheck).
#[derive(Clone, Debug)]
pub struct Run {
    /// The row.
    pub row: Row,
    /// This rank's output, at the points of [`Problem::block`] in their order.
    pub output: Output<f64>,
    /// The FMM's domain.
    pub domain: Domain,
}

/// One level's diagonal-form scratch over the ranks ([`Row::signature_scratch`]; Helmholtz
/// CH4.1b): per rank the boxes with a V row it holds (those of the replicated top on every
/// rank that holds them, as each runs their M2L) and the distinct sources of their rows
/// (ghosts included), and `nd_fmm_tables::helmholtz::diagonal::scratch_shape` of them at the
/// level's degree in f64 under the rank's share of the node budget; the largest over the
/// ranks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignatureScratch {
    /// The most boxes with a V row held on one rank.
    pub targets: usize,
    /// The most distinct V sources on one rank.
    pub sources: usize,
    /// The rank's budget, the node budget over the ranks per node.
    pub budget: usize,
    /// The most bytes of a rank's shape, among the ranks whose shape fits.
    pub bytes: usize,
    /// The most bytes of a rank's source signatures, over every rank (also those over the
    /// budget).
    pub source_bytes: usize,
    /// The fewest targets per chunk among the ranks with targets whose shape fits.
    pub chunk: usize,
    /// The most chunks of one rank.
    pub chunks: usize,
    /// The ranks whose shape does not fit (the level would run RCR there).
    pub over_budget: usize,
}

/// [`Row::signature_scratch`] of a built FMM: one all-reduce (max) and one (sum).
fn signature_scratch<C: CommunicatorCollectives>(
    comm: &C,
    fmm: &HelmholtzFmm<'_, f64, C>,
    threads: usize,
) -> Vec<SignatureScratch> {
    let plan = fmm.plan();
    let nlevels = fmm.nlevels();
    let budget = DEFAULT_SIGNATURE_NODE_BUDGET / fmm.ranks_per_node().max(1);
    let mut most = vec![0u64; 6 * nlevels];
    let mut over = vec![0u64; nlevels];
    let mut sources = Vec::new();
    for l in 0..nlevels {
        let v = plan.level(l).v();
        sources.clear();
        let mut targets = 0;
        // Every row the rank holds, not only the targets it computes: a box of the
        // replicated top has its V row on every rank that holds it, and each runs it.
        for t in 0..v.nrows() {
            let row = v.row(t).0;
            if !row.is_empty() {
                targets += 1;
                sources.extend_from_slice(row);
            }
        }
        sources.sort_unstable();
        sources.dedup();
        let at = &mut most[6 * l..6 * l + 6];
        at[0] = targets as u64;
        at[1] = sources.len() as u64;
        if targets == 0 {
            continue;
        }
        let p = fmm.degrees()[l];
        let samples = (2 * p + 1) * azimuth_count(p);
        at[3] = Signatures::<f64>::bytes_for(samples, sources.len()) as u64;
        match scratch_shape::<f64>(p, sources.len(), targets, threads, budget) {
            Some(shape) => {
                at[2] = shape.bytes() as u64;
                // The fewest per chunk as the most of its negation's complement.
                at[4] = u64::MAX - shape.chunk as u64;
                at[5] = shape.chunks(targets) as u64;
            }
            None => over[l] = 1,
        }
    }
    let mut largest = vec![0u64; 6 * nlevels];
    comm.all_reduce_into(&most[..], &mut largest[..], SystemOperation::max());
    let mut overs = vec![0u64; nlevels];
    comm.all_reduce_into(&over[..], &mut overs[..], SystemOperation::sum());
    largest
        .as_chunks::<6>()
        .0
        .iter()
        .zip(&overs)
        .map(|(m, &over)| SignatureScratch {
            targets: m[0] as usize,
            sources: m[1] as usize,
            budget,
            bytes: m[2] as usize,
            source_bytes: m[3] as usize,
            chunk: if m[4] == 0 {
                0
            } else {
                (u64::MAX - m[4]) as usize
            },
            chunks: m[5] as usize,
            over_budget: over as usize,
        })
        .collect()
}

/// [`run`], also returning this rank's output and the FMM's domain.
///
/// # Collective operation
///
/// On `comm`, every rank with the same arguments.
///
/// # Errors
///
/// The FMM's build and evaluation errors.
///
/// # Panics
///
/// If `case.k` is not a valid wavenumber.
pub fn run_output<C: CommunicatorCollectives>(
    case: &Case,
    problem: &Problem,
    oracle: &Oracle,
    threads: usize,
    comm: &C,
) -> Result<Run, HelmholtzFmmError> {
    let k = Wavenumber::new(case.k.0, case.k.1).expect("a valid wavenumber");
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let block = problem.block(rank, size);
    let points = &problem.points[block.clone()];
    comm.barrier();
    let start = Instant::now();
    let mut builder = HelmholtzFmmBuilder::new(k, case.degrees.clone())
        .charges(case.charges)
        .vectors(case.vectors)
        .gradients(case.gradients)
        .threads(threads)
        .p2p_kernel(case.p2p)
        .m2l(case.m2l)
        .dense_budget(case.dense_budget)
        .auto_plane_wave(case.auto_plane_wave)
        .batched_term(case.batched_term)
        .empty_boxes(case.empty_boxes)
        .near_v(case.near_v)
        .near_wx(case.near_wx)
        .plane_wave_precision(case.plane_wave.clone());
    if let Some(width) = case.m2l_block_targets {
        builder = builder.m2l_block_targets(width);
    }
    if let Some(mul_add) = case.m2l_mul_add {
        builder = builder.m2l_mul_add(mul_add);
    }
    if let Some(max_degree) = case.m2l_max_degree {
        builder = builder.m2l_max_degree(max_degree);
    }
    if let Some(min_degree) = case.m2l_plane_wave_min_degree {
        builder = builder.m2l_plane_wave_min_degree(min_degree);
    }
    if let Some(budget) = case.plane_wave_budget {
        builder = builder.plane_wave_budget(budget);
    }
    if let Some(budget) = case.block_budget {
        builder = builder.block_budget(budget);
    }
    if let Some(reach) = case.plane_wave_reach {
        builder = builder.plane_wave_reach(reach);
    }
    if let Some(split) = case.plane_wave_split {
        builder = builder.plane_wave_split(split);
    }
    if let Some(split) = case.rcr_split {
        builder = builder.rcr_split(split);
    }
    builder = builder
        .diagonal(case.diagonal)
        .diagonal_build(case.diagonal_build)
        .diagonal_costs(case.diagonal_costs)
        .v_after_plan(case.v_after_plan);
    if let Some(raise) = case.plane_wave_raise {
        builder = builder.plane_wave_raise(raise);
    }
    if let Some(domain) = case.domain {
        builder = builder.domain(PhysicalBox::new(domain));
    }
    if let Some(points) = case.max_points_per_leaf {
        builder = builder.max_points_per_leaf(points);
    }
    if let Some(c_leaf) = case.c_leaf {
        builder = builder.c_leaf(c_leaf);
    }
    let mut fmm: HelmholtzFmm<'_, f64, C> = builder.build(points, points, comm)?;
    let build = slowest(start.elapsed().as_secs_f64(), comm);
    let timings = fmm.build_timings();
    let local = [
        timings.settings,
        timings.domain,
        timings.octree,
        timings.routes,
        timings.counts,
        timings.plan,
        timings.shapes,
        timings.operator,
        timings.tables.rcr,
        timings.tables.matrices,
        timings.tables.plane_waves,
        timings.tables.diagonal,
        timings.tables.scratch,
        timings.replan,
        timings.evaluator,
        timings.load,
    ]
    .map(|d| d.as_secs_f64());
    let mut build_parts = [0.0f64; BUILD_PARTS.len()];
    comm.all_reduce_into(&local[..], &mut build_parts[..], SystemOperation::max());
    let charges: Vec<Charges<'_, f64>> = (0..case.vectors)
        .map(|v| match case.charges {
            ChargeKind::Real => Charges::Real(&problem.re[v][block.clone()]),
            ChargeKind::Complex => Charges::Complex {
                re: &problem.re[v][block.clone()],
                im: &problem.im[v][block.clone()],
            },
        })
        .collect();
    // The M2L times before the last evaluation, so that the row has the last one's.
    let (mut output, mut evaluate, mut first_evaluate) = (None, 0.0, 0.0);
    let mut before: Vec<Duration> = Vec::new();
    let mut p2p_before: Vec<Duration> = Vec::new();
    let mut kinds_before: Vec<[Duration; 6]> = Vec::new();
    let mut passes_before: Vec<PlaneWaveTimes> = Vec::new();
    let mut rcr_before: Vec<usize> = Vec::new();
    let mut diagonal_before: Vec<DiagonalTimes> = Vec::new();
    let mut diagonal_counts_before = Default::default();
    for evaluation in 0..case.evaluations.max(1) {
        before = fmm.operator().m2l_times().to_vec();
        p2p_before = fmm.operator().p2p_times().to_vec();
        kinds_before = fmm.operator().kind_times().to_vec();
        passes_before = fmm.operator().plane_wave_times().to_vec();
        rcr_before = fmm.operator().rcr_counts().level_calls.clone();
        diagonal_before = fmm.operator().diagonal_times().to_vec();
        diagonal_counts_before = fmm.operator().diagonal_counts().clone();
        comm.barrier();
        let start = Instant::now();
        output = Some(fmm.evaluate(&charges)?);
        evaluate = slowest(start.elapsed().as_secs_f64(), comm);
        if evaluation == 0 {
            first_evaluate = evaluate;
        }
    }
    let output = output.expect("at least one evaluation");
    let errors = errors(&output, problem, &block, oracle, comm);
    let checksum = reduce_checksum(checksum(&output, block.start), comm);
    let mut leaves = 0usize;
    comm.all_reduce_into(&fmm.nleaves(), &mut leaves, SystemOperation::sum());
    let local: Vec<usize> = fmm.leaves_by_level().into_iter().flatten().collect();
    let mut summed = vec![0usize; local.len()];
    comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
    let leaves_by_level: Vec<[usize; 2]> = summed.chunks(2).map(|c| [c[0], c[1]]).collect();
    let nlevels = fmm.nlevels();
    let seconds: Vec<f64> = fmm
        .operator()
        .m2l_times()
        .iter()
        .zip(&before)
        .map(|(after, before)| (*after - *before).as_secs_f64())
        .collect();
    let mut m2l_seconds = vec![0.0f64; nlevels];
    comm.all_reduce_into(&seconds[..], &mut m2l_seconds[..], SystemOperation::max());
    let m2l_blocks_stored: Vec<bool> = (0..nlevels)
        .map(|l| fmm.operator().tables().m2l_blocks_stored(l))
        .collect();
    // The level calls by pairs of the last evaluation, per level, summed over the ranks.
    let calls: Vec<u64> = (0..nlevels)
        .map(|l| {
            let after = fmm
                .operator()
                .rcr_counts()
                .level_calls
                .get(l)
                .copied()
                .unwrap_or(0);
            let before = rcr_before.get(l).copied().unwrap_or(0);
            (after - before) as u64
        })
        .collect();
    let mut summed = vec![0u64; nlevels];
    comm.all_reduce_into(&calls[..], &mut summed[..], SystemOperation::sum());
    let rcr_by_pairs: Vec<bool> = summed.iter().map(|&c| c > 0).collect();
    let seconds: Vec<f64> = fmm
        .operator()
        .p2p_times()
        .iter()
        .zip(&p2p_before)
        .map(|(after, before)| (*after - *before).as_secs_f64())
        .collect();
    let mut p2p_seconds = vec![0.0f64; nlevels];
    comm.all_reduce_into(&seconds[..], &mut p2p_seconds[..], SystemOperation::max());
    let seconds: Vec<f64> = fmm
        .operator()
        .kind_times()
        .iter()
        .zip(&kinds_before)
        .flat_map(|(after, before)| (0..6).map(|j| (after[j] - before[j]).as_secs_f64()))
        .collect();
    let mut slowest_kinds = vec![0.0f64; 6 * nlevels];
    comm.all_reduce_into(&seconds[..], &mut slowest_kinds[..], SystemOperation::max());
    let kind_seconds: Vec<[f64; 6]> = slowest_kinds.as_chunks::<6>().0.to_vec();
    let local: Vec<usize> = (0..nlevels)
        .flat_map(|l| {
            let lists = fmm.plan().level(l);
            [lists.x().len(), lists.w().len()]
        })
        .collect();
    let mut summed = vec![0usize; local.len()];
    comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
    let far_leaf_entries: Vec<[usize; 2]> = summed.chunks(2).map(|c| [c[0], c[1]]).collect();
    let mut near_interactions = vec![0u64; nlevels];
    comm.all_reduce_into(
        &fmm.near_interactions()[..],
        &mut near_interactions[..],
        SystemOperation::sum(),
    );
    let local: Vec<u64> = fmm.far_interactions().into_iter().flatten().collect();
    let mut summed = vec![0u64; local.len()];
    comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
    let far_interactions: Vec<[u64; 2]> = summed.as_chunks::<2>().0.to_vec();
    let pairs = fmm.plan().near_pairs();
    let local = [
        pairs.w_leaf_pairs,
        pairs.w_converted,
        pairs.x_leaf_pairs,
        pairs.x_converted,
    ]
    .map(|n| n as u64);
    let mut wx_totals = [0u64; 4];
    comm.all_reduce_into(&local[..], &mut wx_totals[..], SystemOperation::sum());
    let (sent, received) = fmm.source_points_exchanged();
    let local = [pairs.v_leaf_pairs, pairs.v_converted, sent, received].map(|n| n as u64);
    let mut v_totals = [0u64; 4];
    comm.all_reduce_into(&local[..], &mut v_totals[..], SystemOperation::sum());
    let passes: Vec<f64> = fmm
        .operator()
        .plane_wave_times()
        .iter()
        .zip(&passes_before)
        .flat_map(|(after, before)| {
            [
                (after.waves - before.waves).as_secs_f64(),
                (after.parents - before.parents).as_secs_f64(),
            ]
        })
        .collect();
    let mut slowest_passes = vec![0.0f64; 2 * nlevels];
    comm.all_reduce_into(&passes[..], &mut slowest_passes[..], SystemOperation::max());
    let plane_wave_seconds: Vec<[f64; 2]> = slowest_passes.as_chunks::<2>().0.to_vec();
    let parts: Vec<f64> = fmm
        .operator()
        .plane_wave_times()
        .iter()
        .zip(&passes_before)
        .flat_map(|(after, before)| {
            [
                (after.busy - before.busy).as_secs_f64(),
                (after.shifts - before.shifts).as_secs_f64(),
                (after.children - before.children).as_secs_f64(),
                (after.conversions - before.conversions).as_secs_f64(),
            ]
        })
        .collect();
    let mut slowest_parts = vec![0.0f64; 4 * nlevels];
    comm.all_reduce_into(&parts[..], &mut slowest_parts[..], SystemOperation::max());
    let plane_wave_parts: Vec<[f64; 4]> = slowest_parts.as_chunks::<4>().0.to_vec();
    // Each target once, on the rank that owns it (a replicated top box has rows on every
    // rank).
    let plan = fmm.plan();
    let pairs: Vec<u64> = (0..nlevels)
        .map(|l| {
            let v = plan.level(l).v();
            (0..v.nrows())
                .filter(|&t| plan.computes(l, t))
                .map(|t| v.row(t).0.len() as u64)
                .sum()
        })
        .collect();
    let mut v_pairs = vec![0u64; nlevels];
    comm.all_reduce_into(&pairs[..], &mut v_pairs[..], SystemOperation::sum());
    // The boxes, and the (block, offset) visits of Batched's level call at blocks of 64 rows
    // (each offset's batch lists its targets ascending), per rank over its own targets.
    let mut local = vec![0u64; 2 * nlevels];
    for (l, counts) in local.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let v = plan.level(l).v();
        counts[0] = (0..v.nrows()).filter(|&t| plan.computes(l, t)).count() as u64;
        for d in 0..v.ngroups() {
            let mut last = None;
            for &t in v.batch(d).0 {
                let block = t as usize / M2L_BLOCK_TARGETS;
                if plan.computes(l, t as usize) && last != Some(block) {
                    counts[1] += 1;
                    last = Some(block);
                }
            }
        }
    }
    let mut summed = vec![0u64; 2 * nlevels];
    comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
    let v_targets: Vec<usize> = summed.iter().step_by(2).map(|&n| n as usize).collect();
    // What the empty-box rule left out, and the multipoles read from other ranks.
    let e = plan.empty_boxes();
    let mut remote: Vec<(usize, u32)> = (0..nlevels)
        .flat_map(|l| {
            let lists = plan.level(l);
            let v = lists.v().sources().iter().map(move |&j| (l, j));
            let w = lists.w().entries().iter().map(move |&j| (l + 1, j));
            v.chain(w)
        })
        .filter(|&(l, j)| !plan.computes(l, j as usize))
        .collect();
    remote.sort_unstable();
    remote.dedup();
    let local = [
        e.target_boxes,
        e.v_without_targets,
        e.x_without_targets,
        e.w_without_targets,
        e.l2l_without_targets,
        e.v_without_sources,
        e.w_without_sources,
        e.m2m_without_sources,
        remote.len(),
    ]
    .map(|n| n as u64);
    let mut totals = [0u64; 9];
    comm.all_reduce_into(&local[..], &mut totals[..], SystemOperation::sum());
    let signature_scratch = signature_scratch(comm, &fmm, threads);
    let diagonal = diagonal_columns(comm, &fmm, &diagonal_before, &diagonal_counts_before);
    let block_visits: Vec<usize> = summed
        .iter()
        .skip(1)
        .step_by(2)
        .map(|&n| n as usize)
        .collect();
    let row = Row {
        threads,
        p2p: fmm.p2p_kernel(),
        m2l: fmm.operator().m2l_choice(),
        m2l_mul_add: fmm.operator().m2l_mul_add(),
        m2l_max_degree: fmm.m2l_max_degree(),
        m2l_plane_wave_min_degree: fmm.m2l_plane_wave_min_degree(),
        dense_levels: (0..fmm.nlevels())
            .map(|l| fmm.operator().m2l_on(l) != HelmholtzM2l::Rotation)
            .collect(),
        dense_bytes: fmm.operator().dense_bytes()
            + fmm.operator().class_bytes()
            + fmm.operator().batched_bytes()
            + fmm.operator().plane_wave_bytes(),
        plane_wave: (0..fmm.nlevels())
            .map(|l| fmm.operator().plane_wave_precision(l))
            .collect(),
        plane_wave_buffer: fmm.operator().plane_wave_buffer_bytes(),
        plane_wave_counts: fmm.operator().plane_wave_counts(),
        plane_wave_reach: fmm.operator().plane_wave_reach(),
        plane_wave_split: fmm.operator().plane_wave_split(),
        plane_wave_raise: fmm.operator().plane_wave_raise(),
        translation_degrees: (0..fmm.nlevels())
            .map(|l| fmm.operator().translation_degree(l))
            .collect(),
        plane_wave_waves: (0..fmm.nlevels())
            .map(|l| fmm.operator().plane_wave_waves(l))
            .collect(),
        ranks: size,
        levels: fmm.nlevels(),
        min_leaf_level: fmm.min_leaf_level(),
        degrees: fmm.degrees().to_vec(),
        circumradii: (0..fmm.nlevels())
            .map(|l| 3f64.sqrt() * radius(l, fmm.domain()))
            .collect(),
        leaves,
        max_leaf_points: fmm.max_leaf_points(),
        max_points_per_leaf: fmm.max_points_per_leaf(),
        c_leaf: fmm.c_leaf(),
        leaves_by_level,
        kind_seconds,
        far_leaf_entries,
        build,
        build_parts,
        evaluate,
        first_evaluate,
        m2l_seconds,
        m2l_blocks_stored,
        rcr_split: fmm.operator().rcr_split(),
        rcr_by_pairs,
        plane_wave_seconds,
        plane_wave_parts,
        batched_columns: v_pairs
            .iter()
            .zip(&v_targets)
            .map(|(&pairs, &targets)| {
                LevelShape {
                    pairs,
                    targets: targets as u64,
                }
                .batched_columns()
            })
            .collect(),
        v_pairs: v_pairs.into_iter().map(|n| n as usize).collect(),
        v_targets,
        block_visits,
        batched_term: fmm.level_shapes().is_some(),
        m2l_block_targets: fmm.operator().m2l_block_targets(),
        empty_boxes: fmm.empty_boxes(),
        left_out: core::array::from_fn(|i| totals[i] as usize),
        remote_multipoles: totals[8] as usize,
        signature_scratch,
        near_v: fmm.near_v(),
        far_interactions,
        near_wx: fmm.near_wx(),
        wx_near: wx_totals.map(|n| n as usize),
        v_near: [v_totals[0] as usize, v_totals[1] as usize],
        source_points: [v_totals[2] as usize, v_totals[3] as usize],
        p2p_seconds,
        near_interactions,
        diagonal: fmm.operator().diagonal(),
        diagonal_build_kind: fmm.operator().diagonal_build(),
        diagonal_classes: diagonal.classes,
        diagonal_pairs: diagonal.pairs,
        diagonal_conversions: diagonal.conversions,
        diagonal_seconds: diagonal.seconds,
        diagonal_build: diagonal.build,
        diagonal_multiply_adds: diagonal.multiply_adds,
        diagonal_model: diagonal_model(case, &fmm, size * threads),
        diagonal_costs: fmm.operator().diagonal_costs(),
        taken_from_waves: (0..fmm.nlevels())
            .map(|l| fmm.operator().taken_from_waves(l))
            .collect(),
        v_after_plan: fmm.v_after_plan(),
        v_crossovers: {
            let before = fmm.v_crossovers();
            let after = fmm.v_crossovers_after_plan().unwrap_or(before);
            before.iter().zip(after).map(|(&b, &a)| [b, a]).collect()
        },
        replanned: fmm.replanned(),
        errors,
        checksum,
    };
    Ok(Run {
        row,
        output,
        domain: *fmm.domain(),
    })
}

/// [`Row::diagonal_model`] (Helmholtz CH4.3b1): per level k a, the bound's classes and the
/// time rule's modelled seconds of the level by the form, by RCR and by the waves, the
/// thread-ns over `cores` (ranks × threads).
fn diagonal_model<C: CommunicatorCollectives>(
    case: &Case,
    fmm: &HelmholtzFmm<f64, C>,
    cores: usize,
) -> Vec<[f64; 5]> {
    let op = fmm.operator();
    let costs = op.diagonal_costs();
    let shapes = op.diagonal_shapes().unwrap_or(&[]);
    let epsilon = match case.degrees {
        Degrees::Tolerance(epsilon) => Some(epsilon),
        Degrees::PerLevel(_) => None,
    };
    let seconds = |ns: f64| ns / 1e9 / cores.max(1) as f64;
    (0..fmm.nlevels())
        .map(|l| {
            let r = radius(l, fmm.domain());
            let k_hat = Complex::new(case.k.0 * r, case.k.1 * r);
            let ka = k_hat.re * 3f64.sqrt();
            // The degree as run: on a level the waves raised the resolving degree, which the
            // time rule keeps whatever M2L runs (Helmholtz CH4.3b2a).
            let p = op.degree(l);
            let base = op.base_degree(l);
            let shape = shapes.get(l).copied().unwrap_or_default();
            let classes = match epsilon {
                Some(epsilon) if l >= 2 => {
                    diagonal_rule_classes::<f64>(p, k_hat, epsilon, op.diagonal_build())
                }
                _ => Vec::new(),
            };
            let p_w = match epsilon {
                Some(epsilon) if p == base => plane_wave_degree(base, epsilon),
                _ => p,
            };
            [
                ka,
                classes.len() as f64,
                seconds(costs.diagonal_level(p, &shape, &classes)),
                seconds(costs.rcr_level(p, &shape)),
                seconds(costs.plane_wave_level(p_w, &shape)),
            ]
        })
        .collect()
}

/// The diagonal M2L's columns of a [`Row`] (Helmholtz CH4.2b2).
struct DiagonalColumns {
    classes: Vec<usize>,
    pairs: Vec<[usize; 2]>,
    conversions: Vec<[usize; 2]>,
    seconds: Vec<[f64; 6]>,
    build: Vec<[f64; 2]>,
    multiply_adds: Vec<u64>,
}

/// [`DiagonalColumns`] of the last evaluation of `fmm`, its times and counts before it given:
/// one all-reduce (sum) of the counts and one (max) of the times.
fn diagonal_columns<C: CommunicatorCollectives>(
    comm: &C,
    fmm: &HelmholtzFmm<'_, f64, C>,
    before: &[DiagonalTimes],
    counts_before: &DiagonalCounts,
) -> DiagonalColumns {
    let (op, nlevels) = (fmm.operator(), fmm.nlevels());
    let counts = op.diagonal_counts();
    let last = |after: &[usize], before: &[usize], l: usize| {
        (after.get(l).copied().unwrap_or(0) - before.get(l).copied().unwrap_or(0)) as u64
    };
    let local: Vec<u64> = (0..nlevels)
        .flat_map(|l| {
            [
                last(&counts.diagonal_pairs, &counts_before.diagonal_pairs, l),
                last(&counts.rcr_pairs, &counts_before.rcr_pairs, l),
                last(&counts.signatures, &counts_before.signatures, l),
                last(&counts.back, &counts_before.back, l),
            ]
        })
        .collect();
    let mut summed = vec![0u64; local.len()];
    comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
    let zero = DiagonalTimes::default();
    let local: Vec<f64> = (0..nlevels)
        .flat_map(|l| {
            let after = op.diagonal_times().get(l).copied().unwrap_or_default();
            let before = before.get(l).copied().unwrap_or(zero);
            let built = op.diagonal_build_time(l).unwrap_or_default();
            [
                after.signatures - before.signatures,
                after.targets - before.targets,
                after.s2f - before.s2f,
                after.pairs - before.pairs,
                after.f2r - before.f2r,
                after.rcr - before.rcr,
                built.0,
                built.1,
            ]
            .map(|d| d.as_secs_f64())
        })
        .collect();
    let mut slowest = vec![0.0f64; local.len()];
    comm.all_reduce_into(&local[..], &mut slowest[..], SystemOperation::max());
    let counts = summed.as_chunks::<4>().0;
    let times = slowest.as_chunks::<8>().0;
    DiagonalColumns {
        classes: (0..nlevels)
            .map(|l| op.diagonal_classes(l).map_or(0, |c| c.len()))
            .collect(),
        pairs: counts
            .iter()
            .map(|c| [c[0] as usize, c[1] as usize])
            .collect(),
        conversions: counts
            .iter()
            .map(|c| [c[2] as usize, c[3] as usize])
            .collect(),
        seconds: times
            .iter()
            .map(|t| core::array::from_fn(|i| t[i]))
            .collect(),
        build: times.iter().map(|t| [t[6], t[7]]).collect(),
        multiply_adds: (0..nlevels)
            .map(|l| op.diagonal_multiply_adds(l).unwrap_or(0))
            .collect(),
    }
}

/// k as a complex number.
pub fn complex_k(case: &Case) -> Complex<f64> {
    Complex::new(case.k.0, case.k.1)
}

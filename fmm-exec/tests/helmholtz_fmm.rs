//! The Helmholtz FMM (Helmholtz CH3.1b; docs/helmholtz/tasks/CH3.1b.md) on any number of
//! ranks: its own executable, whose one test initialises MPI at `Threading::Funneled`.
//!
//! Every scenario generates its points and charges by index from a seed (the same on every
//! rank), and each rank passes every P-th of them (`share`). Per scenario:
//!
//! - against the direct sum (`nd_fmm_ref::helmholtz::p2p::direct_sum` in f64, 1/(4π)
//!   included, over every rank's sources): the relative L2 error of φ and ∇φ over every
//!   rank's targets within `ERROR_FACTOR` times ε for the default degree rule, or within
//!   `P8_BOUND` for degree 8 on every level;
//! - the output pass bit for bit the reference pass of `Densities::write_leaf_output`;
//! - 4 threads bit for bit one (2 and 4 in release builds);
//! - the operator pair by pair (`PerPair`) bit for bit its level calls;
//! - on several ranks, the one-rank FMM over the union of every rank's points and charges
//!   in rank order (`SimpleCommunicator::self_comm()`, on every rank) bit for bit.
//!
//! Every scenario runs the default P2P kernel (`P2pChoice::Auto`, the vectorised kernel of
//! `nd-fmm-simd` on the default ISA). On two scenarios the P2P kernels (Helmholtz CH3.4c):
//! `Reference` and every ISA this machine offers, `Isa::Scalar` bit for bit `Reference`,
//! `Auto` bit for bit `Isa(Isa::detect())`, and every ISA within `KERNEL_TOLERANCE` of
//! `Reference` (relative L2 over every rank's targets) and within the scenario's bound of
//! the direct sum.
//!
//! On the same two scenarios the Dense M2L (Helmholtz CH3.2a): within `DENSE_TOLERANCE` of
//! Rotation's output (relative L2 over every rank's targets) and the scenario's bound of the
//! direct sum, 2 threads, pair by pair and the one-rank FMM bit for bit, and on the deeper
//! tree with the finest level's tables alone (RCR above) within `DENSE_TOLERANCE` of
//! Rotation's (in debug builds every Dense check of the deeper tree runs that way: the dense
//! builds of p = 8 on four levels took about 20 s there). The Classes M2L (Helmholtz
//! CH3.2b) on the same two scenarios with the same checks within `DENSE_TOLERANCE`, every
//! level and the finest level alone in every build (16 offsets a level to build). The
//! Batched M2L (Helmholtz CH3.2c) on the same two scenarios with Dense's checks and its
//! debug-build reduction (it builds Dense's tables), and against Dense's output of the same
//! levels: bit for bit with real multipoles (real k, real charges), within
//! `DENSE_TOLERANCE` with complex ones; with two vectors the level call against the pair
//! body bit for bit; and in the fused mode within `DENSE_TOLERANCE` of Rotation and Dense
//! and the one-rank FMM bit for bit. The rule per level (Helmholtz CH3.2d), `Auto`, on the
//! same two scenarios with one degree more on every level but the finest and the finest
//! level's degree as the crossover (fused Batched there, Rotation above, a split by degree on
//! the deeper tree): each level's M2L as the rule says, the fused mode, within
//! `DENSE_TOLERANCE` of Rotation of the same degrees and the scenario's bound of the direct
//! sum, 2 threads, pair by pair and the one-rank FMM bit for bit; the rule's
//! own crossover agreed alike on every rank, a named mode kept, and Auto the builder's
//! default (every other check names Rotation, its reference); these with the term of the
//! levels' V lists off (CH3.2d's rule). The term (Helmholtz CH4.0, `check_batched_term`):
//! the levels' shapes the same on every rank and the one-rank FMM's, m* the table agreed over
//! the ranks, each level Batched exactly where its estimate of the columns per read reaches it
//! (some level Rotation by the term alone), within `DENSE_TOLERANCE` of Rotation, 2 threads
//! and the one-rank FMM bit for bit; the library's default against the direct sum and the
//! one-rank FMM; Batched's blocks of 1, 7 and 64 targets at 1 and 2 threads (in debug builds
//! 1 at 1 and 7 at 2) the default's bits; the dense budget per rank the node budget over the
//! ranks per node. The exponential M2L
//! (Helmholtz CH6.1d), `PlaneWave`, on the same two scenarios: each level's precision as
//! the rule gives it (`nd_fmm_tables::helmholtz::plane_wave::merge::select`, at the
//! scenario's tolerance or its named degrees), the scenario's bound of the direct sum and
//! the errors within `PLANE_WAVE_RATIO` of Rotation's (Helmholtz CH6.1e), the level call
//! by merge and shift within `PLANE_PAIR_TOLERANCE` of its pair path (CH6.1c's
//! translation along the merge's direction), 2 threads and the one-rank FMM bit
//! for bit, vector v of two the one-vector FMM bit for bit, precisions named per level
//! (Rotation's level call bit for bit where every level names none). In debug builds the
//! pair path of a tree with waves on several levels runs on the finest level's alone.
//! Merge and shift split by child and by parents (Helmholtz CH6.2f): each the default bit
//! for bit at 1 and 2 threads, by child at the buffer's floor and on P ranks the one-rank
//! FMM; the default split by child at 2 threads and never at 1; its parts timed.
//!
//! Below ε = 10⁻⁶ (Helmholtz CH6.1g, `check_resolving_degree`): the clusters at kD = 1 and
//! the cube at kD = 20 + 2i at 10⁻⁷ and 10⁻⁸ under PlaneWave's rule, each level with waves
//! at the resolving degree, within `PLANE_WAVE_RATIO` of Rotation's errors at the degrees
//! given, 2 threads and the one-rank FMM bit for bit, and the default (Auto) PlaneWave's
//! output at 10⁻⁸ and Rotation's at 10⁻⁷, bit for bit; and the same under
//! `PlaneWaveRaise::Leaves` and `PlaneWaveRaise::M2l` (Helmholtz CH6.1h: the raised levels'
//! translations to and from the parent, and under `M2l` their leaves' operators, at the
//! degrees given), each with its own errors within `PLANE_WAVE_RATIO` of Rotation's.
//!
//! The Rotation M2L by pairs (Helmholtz CH4.0i, `check_rcr_split`): on two scenarios
//! (complex k with complex charges and rank 0 empty; real k with real charges and separate
//! targets) `RcrSplit::Pairs` serially, on 2 threads with the default buffer and on 3 at the
//! budget's floor (one pair per thread, so rows span chunks), and `RcrSplit::Auto` at 20
//! threads (no level by pairs below the rule's degree), each bit for bit `RcrSplit::Boxes` at
//! one thread, with the counts of the level calls by pairs; two vectors; the library's M2L (Auto: waves and
//! Batched on some levels, Rotation on the rest); on P ranks the one-rank FMM.
//!
//! Then vectors (vector v of an FMM of two bit for bit the one-vector FMM, of complex and
//! of real charges), real charges against complex ones with Im q = 0, and the errors (an
//! ISA this machine lacks among them), each on every rank or on none. One scenario leaves
//! rank 0 without points.
//!
//! The diagonal M2L (Helmholtz CH4.2b1, `check_diagonal`): on the cube at kD = 40 + 2i and the
//! clusters at kD = 20, ε = 10⁻⁴ (the clusters' level 4 mixes the diagonal form and RCR),
//! under Rotation and the library's Auto, the errors against the direct sum within
//! `DIAGONAL_RATIO` (1.1) of the setting's without it, the classes the same on every rank,
//! 2 threads (4 in release) and chunk widths bit for bit, the one-rank FMM, two vectors each
//! the one-vector FMM, and the budget's floor Rotation's bits; where Auto leaves no RCR level,
//! the setting's output unchanged. Its debug cost: about 110 s of the one-rank run on a loaded
//! M6 (load about 23), 8 s at CI's opt-level 1. Under Rotation also with interpolated transfer
//! functions, the library's default build since CH4.2c2's sign-off (Helmholtz CH4.2c2,
//! `check_interpolated`; the other checks name the direct build): each level's classes within
//! the direct build's, the errors within `DIAGONAL_RATIO` of Rotation's, 2 threads and the one-rank FMM bit
//! for bit. Under both settings the time rule (Helmholtz CH4.3b1, `check_diagonal_auto`), at
//! the machine's costs and at free ones (every level the bound reaches, Auto's levels with waves
//! included): the agreed shapes, the classes the same on every rank, the errors within
//! `DIAGONAL_RATIO` of the setting's without the form, the one-rank FMM bit for bit with its
//! shapes, and at free costs 2 threads and chunk widths bit for bit.
//!
//! The V rule after the plan (Helmholtz CH4.3b2b, `check_v_after_plan`): on the sphere at
//! kD = 30 under the cost model's V rule, Auto with waves from degree 0 and the time rule at
//! named costs (the form taking the levels with waves with a dear pair; RCR taking them; the
//! form dearer than the waves), `VAfterPlan::TakenLevels` and `AllLevels`: the M2L per level the setting off's,
//! the crossovers raised on the levels named and nowhere else, the V pairs per level at most
//! and the converted pairs and near interactions at least the setting off's, the errors within
//! `NEAR_ERROR_BOUND` of them, 2 threads and the one-rank FMM bit for bit; under Rotation
//! nothing raised, no second plan, the setting off's output bit for bit.
//!
//! Every comparison is agreed by an all-reduce, so a failure fails every rank. Run by hand
//! at several ranks under a timeout, e.g. on macOS
//! `cargo test -p nd-fmm-exec --test helmholtz_fmm --no-run`, then
//! `timeout 600 mpirun -n 4 --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0
//! <binary> --test-threads=1 --nocapture` (CI: `.github/scripts/run-mpi-tests.sh`).

use std::f64::consts::PI;
use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use nd_fmm_exec::fmm::{FmmError, SettingsError};
use nd_fmm_exec::helmholtz::{
    C_LEAF, ChargeKind, Charges, DEFAULT_C_LEAF, DEFAULT_DENSE_BUDGET, DEFAULT_DENSE_NODE_BUDGET,
    DEFAULT_DIAGONAL_BUILD, DEFAULT_PLANE_WAVE_NODE_BUDGET, DIAGONAL_COSTS_NEON, Degrees,
    DensityError, DiagonalCosts, DiagonalTimes, HelmholtzDiagonal, HelmholtzFmm,
    HelmholtzFmmBuilder, HelmholtzFmmError, HelmholtzM2l, LeafVShape, MIN_BUFFER_BOXES,
    OperatorError, Output, PlaneWaveCounts, PlaneWavePrecision, PlaneWaveRaise, PlaneWaveSplit,
    PlaneWaveTimes, RCR_PAIR_FEW_DEGREE, RcrSplit, TransferBuild, V_COSTS_NEON, VAfterPlan, VNear,
    VRaisedPrice, Wavenumber, WxNear, auto_batched_max_degree, auto_batched_min_columns,
    auto_m2l_mul_add, auto_plane_wave_min_degree, default_max_points_per_leaf, raised_crossover,
};
use nd_fmm_exec::operator::{Isa, MulAdd, P2pChoice};
use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::p2p;
use nd_fmm_tables::helmholtz::batched::BatchedM2l;
use nd_fmm_tables::helmholtz::classes::{ClassM2l, CubeTransforms};
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_fmm_tables::helmholtz::plane_wave::Precision;
use nd_fmm_tables::helmholtz::plane_wave::merge::{plane_wave_degree, select};
use nd_fmm_tables::helmholtz::plane_wave::quadrature::RHO_MAX;
use nd_octree::PhysicalBox;
use nd_octree::octree::KeyType;

type C = Complex<f64>;

/// The error against the direct sum may be this many times ε: the default rule meets ε/2
/// on the reference workloads' sampled targets (Helmholtz CH3.3), and these small trees
/// and one rank's targets are not those workloads.
const ERROR_FACTOR: f64 = 10.0;

/// The error bound of the scenarios with degree 8 on every level, where truncation
/// dominates: about 5 to 13 times the errors measured on the M6 (7.5e-5 and 2.2e-4).
const P8_BOUND: f64 = 1e-3;

/// Every ISA's output may differ from `Reference`'s by this much (relative L2 of φ and of
/// ∇φ over every rank's targets): the P2P kernel's contract (docs/design/simd-p2p.md §4.12)
/// is a few tens of u_T per term at these k r; measured at most 6.9e-16 on the M6
/// (NEON). As Laplace's tolerance for its kernels in f64 (`tests/mpi_exec.rs`).
const KERNEL_TOLERANCE: f64 = 1e-13;

/// The Dense or Classes M2L's output may differ from Rotation's by this much (relative L2
/// of φ and of ∇φ over every rank's targets): the same operator in another order of
/// operations (Classes also through the cube-group transforms), so rounding (at most
/// 7.9e-16 for Dense and 2.4e-16 for Classes on the M6), with a margin of about ten.
const DENSE_TOLERANCE: f64 = 1e-14;

/// The exponential M2L's level call (merge and shift) may differ from its pair path (one
/// shift per pair, CH6.1c's translation) by this much (relative L2 of φ and of ∇φ over
/// every rank's targets): the same waves, shifted in two factors and summed in another
/// order, so rounding (at most 2.0e-16 on the M6), with a margin for other libms.
const PLANE_PAIR_TOLERANCE: f64 = 1e-14;

/// The largest ratio of `PlaneWave`'s errors against the direct sum to Rotation's under
/// the precision rule (CH6.1's acceptance, "within about 1.1× Rotation's", Helmholtz
/// CH6.1e).
const PLANE_WAVE_RATIO: f64 = 1.1;

/// The diagonal M2L's errors against the direct sum within this factor of Rotation's (or of
/// the setting's without it), φ and ∇φ: the acceptance of Helmholtz CH4.2b
/// (docs/helmholtz/roadmap.md).
const DIAGONAL_RATIO: f64 = 1.1;

/// SplitMix64, uniform in [0, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// How the points lie in the unit cube (the domain [0, 1]³ is supplied).
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// Uniform.
    Cube,
    /// Four Gaussian clusters of width 0.05, so the leaves lie on several levels.
    Clusters,
    /// The sphere of radius 0.4 about the cube's centre, so most leaves the leaf rule makes
    /// hold no point (Helmholtz CH4.0d1).
    Sphere,
}

/// `n` points of `layout` strictly inside the unit cube, from `seed`.
fn points(layout: Layout, n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut rng = Rng(seed);
    let centres = [
        [0.2, 0.3, 0.25],
        [0.7, 0.75, 0.3],
        [0.35, 0.7, 0.8],
        [0.8, 0.2, 0.7],
    ];
    (0..n)
        .map(|i| match layout {
            Layout::Cube => core::array::from_fn(|_| 0.001 + 0.998 * rng.next()),
            Layout::Clusters => {
                let c = centres[i % 4];
                core::array::from_fn(|a| {
                    // Box–Muller, clamped inside the cube.
                    let (u, v) = (rng.next().max(1e-12), rng.next());
                    let g = (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos();
                    (c[a] + 0.05 * g).clamp(0.001, 0.999)
                })
            }
            Layout::Sphere => {
                let z = 2.0 * rng.next() - 1.0;
                let phi = 2.0 * PI * rng.next();
                let r = (1.0 - z * z).sqrt();
                [
                    0.5 + 0.4 * r * phi.cos(),
                    0.5 + 0.4 * r * phi.sin(),
                    0.5 + 0.4 * z,
                ]
            }
        })
        .collect()
}

/// `n` values uniform in [−1, 1), from `seed`.
fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    (0..n).map(|_| 2.0 * rng.next() - 1.0).collect()
}

/// This rank's share of `items`: every `size`-th item from `rank`.
fn share<V: Copy>(items: &[V], comm: &SimpleCommunicator) -> Vec<V> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    items.iter().skip(rank).step_by(size).copied().collect()
}

/// This rank's share of `items`, with `empty`: on several ranks, rank 0 has none and the
/// other ranks share every item; otherwise [`share`].
fn share_of<V: Copy>(items: &[V], empty: bool, comm: &SimpleCommunicator) -> Vec<V> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    if !empty || size == 1 {
        share(items, comm)
    } else if rank == 0 {
        Vec::new()
    } else {
        items
            .iter()
            .skip(rank - 1)
            .step_by(size - 1)
            .copied()
            .collect()
    }
}

/// The values of every rank's `local`, in rank order, and the position of this rank's
/// first value among them.
fn union<V: Equivalence + Copy + Default>(
    local: &[V],
    comm: &SimpleCommunicator,
) -> (Vec<V>, usize) {
    let size = comm.size() as usize;
    let mut counts = vec![0i32; size];
    comm.all_gather_into(&(local.len() as i32), &mut counts[..]);
    let displacements: Vec<i32> = counts
        .iter()
        .scan(0, |start, &count| {
            let first = *start;
            *start += count;
            Some(first)
        })
        .collect();
    let total = counts.iter().map(|&n| n as usize).sum();
    let mut all = vec![V::default(); total];
    let mut partition = PartitionMut::new(&mut all[..], &counts[..], &displacements[..]);
    comm.all_gather_varcount_into(local, &mut partition);
    (all, displacements[comm.rank() as usize] as usize)
}

/// The points of every rank in rank order, and the position of this rank's first point.
fn union_points(local: &[[f64; 3]], comm: &SimpleCommunicator) -> (Vec<[f64; 3]>, usize) {
    let (flat, first) = union(local.as_flattened(), comm);
    (flat.as_chunks::<3>().0.to_vec(), first / 3)
}

/// The sum of `x` over every rank.
fn global_sum(x: f64, comm: &SimpleCommunicator) -> f64 {
    let mut sum = 0.0;
    comm.all_reduce_into(&x, &mut sum, SystemOperation::sum());
    sum
}

/// Whether `ok` holds on every rank.
fn everywhere(ok: bool, comm: &SimpleCommunicator) -> bool {
    let mut all = false;
    comm.all_reduce_into(&ok, &mut all, SystemOperation::logical_and());
    all
}

/// Checks the M2L times per level (Helmholtz CH6.2b): one entry per level, zero on level 0
/// (the evaluator's downward pass calls M2L from level 1), positive on every level with V
/// pairs on this rank, on every rank or none.
fn check_m2l_times(name: &str, fmm: &HelmholtzFmm<'_, f64>, comm: &SimpleCommunicator) {
    let times = fmm.operator().m2l_times();
    let ok = times.len() == fmm.nlevels()
        && times[0] == Duration::ZERO
        && (0..fmm.nlevels())
            .all(|level| fmm.plan().level(level).v().is_empty() || times[level] > Duration::ZERO);
    assert!(
        everywhere(ok, comm),
        "{name}: an M2L time on every level with V pairs (rank {}: {times:?})",
        comm.rank()
    );
}

/// Checks the exponential M2L's passes per level (Helmholtz CH6.2d): one entry per level;
/// on a level with waves and V pairs on this rank both passes timed and within the level's
/// M2L time; zero elsewhere; on every rank or none.
fn check_plane_wave_times(name: &str, fmm: &HelmholtzFmm<'_, f64>, comm: &SimpleCommunicator) {
    let op = fmm.operator();
    let (times, passes) = (op.m2l_times(), op.plane_wave_times());
    let ok = passes.len() == fmm.nlevels()
        && (0..fmm.nlevels()).all(|level| {
            let pass = passes[level];
            let waves =
                op.plane_wave_precision(level).is_some() && !fmm.plan().level(level).v().is_empty();
            if waves {
                pass.waves > Duration::ZERO
                    && pass.parents > Duration::ZERO
                    && pass.waves + pass.parents <= times[level]
                    // The parts (CH6.2f): thread-seconds, within the busy time.
                    && pass.busy > Duration::ZERO
                    && pass.conversions > Duration::ZERO
                    && pass.shifts + pass.children + pass.conversions <= pass.busy
            } else {
                pass == PlaneWaveTimes::default()
            }
        });
    assert!(
        everywhere(ok, comm),
        "{name}: both passes and the parts timed on every level with waves, within its M2L \
         time (rank {}: \
         {passes:?} of {times:?})",
        comm.rank()
    );
}

/// Every value of an output as bits: per column φ, then the gradient triples.
fn bits(output: &Output<f64>) -> Vec<u64> {
    let mut bits: Vec<u64> = output.potential.iter().map(|x| x.to_bits()).collect();
    if let Some(gradient) = &output.gradient {
        bits.extend(gradient.as_flattened().iter().map(|x| x.to_bits()));
    }
    bits
}

/// Asserts on every rank that `got` equals `want` bit for bit on every rank.
fn assert_same(what: &str, got: &Output<f64>, want: &Output<f64>, comm: &SimpleCommunicator) {
    let same = bits(got) == bits(want);
    if !same {
        eprintln!("rank {}: {what}: the outputs differ", comm.rank());
    }
    assert!(
        everywhere(same, comm),
        "{what}: not bit for bit on every rank"
    );
}

/// A scenario.
#[derive(Clone, Debug)]
struct Scenario {
    name: &'static str,
    layout: Layout,
    /// Sources, and targets (the same set if `None`).
    n: (usize, Option<usize>),
    k: (f64, f64),
    degrees: Degrees,
    /// The error bound against the direct sum (`ERROR_FACTOR` ε, or `P8_BOUND`).
    bound: f64,
    charges: ChargeKind,
    gradients: bool,
    /// Whether the scenario runs in debug builds (each takes a few seconds there).
    debug: bool,
    /// Whether rank 0 passes no point on several ranks.
    empty_rank0: bool,
}

/// The inputs of a scenario on this rank.
struct Inputs {
    sources: Vec<[f64; 3]>,
    targets: Vec<[f64; 3]>,
    /// Per vector: Re q, Im q.
    charges: Vec<(Vec<f64>, Vec<f64>)>,
}

impl Inputs {
    /// The scenario's inputs, m vectors, this rank's share.
    fn new(scenario: &Scenario, m: usize, comm: &SimpleCommunicator) -> Self {
        let (ns, nt) = scenario.n;
        let all_sources = points(scenario.layout, ns, 11);
        let all_targets = nt.map(|nt| points(scenario.layout, nt, 12));
        let empty = scenario.empty_rank0;
        let sources = share_of(&all_sources, empty, comm);
        let targets = share_of(all_targets.as_ref().unwrap_or(&all_sources), empty, comm);
        let charges = (0..m as u64)
            .map(|v| {
                let re = share_of(&values(ns, 100 + 2 * v), empty, comm);
                let im = match scenario.charges {
                    ChargeKind::Real => vec![0.0; re.len()],
                    ChargeKind::Complex => share_of(&values(ns, 101 + 2 * v), empty, comm),
                };
                (re, im)
            })
            .collect();
        Self {
            sources,
            targets,
            charges,
        }
    }

    /// The charges as the FMM takes them, for the path `kind`.
    fn charges(&self, kind: ChargeKind) -> Vec<Charges<'_, f64>> {
        self.charges
            .iter()
            .map(|(re, im)| match kind {
                ChargeKind::Real => Charges::Real(re),
                ChargeKind::Complex => Charges::Complex { re, im },
            })
            .collect()
    }
}

/// The builder of a scenario, M2L by Rotation (named: the library's default is Auto).
fn builder(scenario: &Scenario) -> HelmholtzFmmBuilder {
    default_builder(scenario).m2l(HelmholtzM2l::Rotation)
}

/// The scenario's settings with the library's default M2L ([`HelmholtzM2l::Auto`] since
/// Helmholtz CH3.2d); `builder` names Rotation, the reference of every check. The tree and
/// the near pairs stay those before Helmholtz CH4.0d3's defaults (32 points per leaf, D11's
/// c_leaf, the V, W and X rules off), which the checks name where they vary them;
/// `check_defaults` checks the library's.
fn default_builder(scenario: &Scenario) -> HelmholtzFmmBuilder {
    let k = Wavenumber::new(scenario.k.0, scenario.k.1).unwrap();
    HelmholtzFmmBuilder::new(k, scenario.degrees.clone())
        .charges(scenario.charges)
        .gradients(scenario.gradients)
        .domain(PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
        .max_points_per_leaf(32)
        .c_leaf(C_LEAF)
        .near_v(VNear::Off)
        .near_wx(WxNear::Off)
}

/// Builds and evaluates; panics on an error.
fn run<'o>(
    builder: &HelmholtzFmmBuilder,
    inputs: &Inputs,
    kind: ChargeKind,
    comm: &'o SimpleCommunicator,
) -> (HelmholtzFmm<'o, f64>, Output<f64>) {
    let mut fmm = builder
        .build(&inputs.sources, &inputs.targets, comm)
        .unwrap_or_else(|error| panic!("rank {}: the build: {error}", comm.rank()));
    let output = fmm
        .evaluate(&inputs.charges(kind))
        .unwrap_or_else(|error| panic!("rank {}: the evaluation: {error}", comm.rank()));
    (fmm, output)
}

/// The relative L2 errors of φ and ∇φ of vector v of `output` against the direct sum over
/// every rank's sources, at this rank's targets, summed over every rank.
fn errors(
    scenario: &Scenario,
    inputs: &Inputs,
    output: &Output<f64>,
    v: usize,
    comm: &SimpleCommunicator,
) -> (f64, f64) {
    let k = C::new(scenario.k.0, scenario.k.1);
    let (all_sources, _) = union_points(&inputs.sources, comm);
    let (re, im) = &inputs.charges[v];
    let (re, _) = union(re, comm);
    let (im, _) = union(im, comm);
    let charges: Vec<C> = re.iter().zip(&im).map(|(&r, &i)| C::new(r, i)).collect();
    let n = inputs.targets.len();
    let zero = C::new(0.0, 0.0);
    let mut phi = vec![zero; n];
    let mut grad = vec![[zero; 3]; n];
    p2p::direct_sum(
        k,
        &all_sources,
        &charges,
        &inputs.targets,
        &mut phi,
        Some(&mut grad),
    );
    let scale = 1.0 / (4.0 * PI);
    let potential = output.potential_of(v);
    let at = |planes: &[f64], i| planes[i];
    let mut sums = [0.0f64; 4];
    for i in 0..n {
        let got = C::new(potential.re[i], potential.im.map_or(0.0, |im| at(im, i)));
        let want = phi[i] * scale;
        sums[0] += (got - want).norm_sqr();
        sums[1] += want.norm_sqr();
        if let Some(g) = output.gradient_of(v) {
            for a in 0..3 {
                let got = C::new(g.re[i][a], g.im.map_or(0.0, |im| im[i][a]));
                let want = grad[i][a] * scale;
                sums[2] += (got - want).norm_sqr();
                sums[3] += want.norm_sqr();
            }
        }
    }
    let sums = sums.map(|s| global_sum(s, comm));
    let gradient = if scenario.gradients {
        (sums[2] / sums[3]).sqrt()
    } else {
        0.0
    };
    ((sums[0] / sums[1]).sqrt(), gradient)
}

/// On several ranks, the one-rank FMM of `builder` over the union of every rank's inputs
/// in rank order, built by every rank on `self_comm` (so that a failure there fails every
/// rank), whose output at each rank's targets must equal `output` bit for bit.
fn check_one_rank(
    what: &str,
    builder: &HelmholtzFmmBuilder,
    inputs: &Inputs,
    kind: ChargeKind,
    output: &Output<f64>,
    comm: &SimpleCommunicator,
) {
    if comm.size() == 1 {
        return;
    }
    let (sources, _) = union_points(&inputs.sources, comm);
    let (targets, first) = union_points(&inputs.targets, comm);
    let charges: Vec<(Vec<f64>, Vec<f64>)> = inputs
        .charges
        .iter()
        .map(|(re, im)| (union(re, comm).0, union(im, comm).0))
        .collect();
    let union_inputs = Inputs {
        sources,
        targets,
        charges,
    };
    // Every rank builds the reference, so that a failure there fails every rank.
    let one = SimpleCommunicator::self_comm();
    let (fmm, all) = run(&builder.clone().threads(1), &union_inputs, kind, &one);
    let reference = bits(&all);
    let n = union_inputs.targets.len();
    let columns = fmm.densities().output_columns();
    let gradients = output.gradient.is_some();
    // This rank's targets of every column, φ then ∇φ.
    let local = inputs.targets.len();
    let (phi, grad) = reference.split_at(columns * n);
    let mut want = Vec::new();
    for column in 0..columns {
        want.extend_from_slice(&phi[column * n + first..column * n + first + local]);
    }
    if gradients {
        for column in 0..columns {
            let at = 3 * (column * n + first);
            want.extend_from_slice(&grad[at..at + 3 * local]);
        }
    }
    let same = bits(output) == want;
    if !same {
        eprintln!(
            "rank {}: {what}: differs from the one-rank FMM",
            comm.rank()
        );
    }
    assert!(
        everywhere(same, comm),
        "{what}: not the one-rank FMM over the union in rank order, bit for bit"
    );
}

/// The scenario's checks.
fn check(scenario: &Scenario, comm: &SimpleCommunicator) {
    let start = Instant::now();
    let rank = comm.rank();
    let inputs = Inputs::new(scenario, 1, comm);
    let builder = builder(scenario);
    let kind = scenario.charges;
    let (fmm, output) = run(&builder, &inputs, kind, comm);

    let (phi, grad) = errors(scenario, &inputs, &output, 0, comm);
    if rank == 0 {
        println!(
            "{}: {} levels, degrees {:?}, leaf level {}, {} leaves on rank 0, largest leaf {}: \
             relative L2 φ {phi:.2e}, ∇φ {grad:.2e} (bound {:.1e})",
            scenario.name,
            fmm.nlevels(),
            fmm.degrees(),
            fmm.min_leaf_level(),
            fmm.nleaves(),
            fmm.max_leaf_points(),
            scenario.bound,
        );
    }
    assert!(
        phi < scenario.bound && grad < scenario.bound,
        "{}: φ {phi:.2e}, ∇φ {grad:.2e} against the direct sum, bound {:.1e}",
        scenario.name,
        scenario.bound
    );

    assert_same(
        &format!(
            "{}: the output pass against write_leaf_output",
            scenario.name
        ),
        &output,
        &fmm.reference_output(),
        comm,
    );
    drop(fmm);

    // 2 threads in release builds only (the debug budget).
    let threads: &[usize] = if cfg!(debug_assertions) {
        &[4]
    } else {
        &[2, 4]
    };
    for &threads in threads {
        let (_, threaded) = run(&builder.clone().threads(threads), &inputs, kind, comm);
        assert_same(
            &format!("{}: {threads} threads", scenario.name),
            &threaded,
            &output,
            comm,
        );
    }

    let (_, pairs) = run(&builder.clone().pair_calls(true), &inputs, kind, comm);
    assert_same(
        &format!("{}: pair by pair", scenario.name),
        &pairs,
        &output,
        comm,
    );

    check_one_rank(scenario.name, &builder, &inputs, kind, &output, comm);
    if rank == 0 {
        println!(
            "{}: checks passed in {:.1?}",
            scenario.name,
            start.elapsed()
        );
    }
}

/// The relative L2 differences of φ and of ∇φ of `a` from `b` over every rank's targets.
fn difference(a: &Output<f64>, b: &Output<f64>, comm: &SimpleCommunicator) -> (f64, f64) {
    let sums = |x: &[f64], y: &[f64]| -> [f64; 2] {
        let d = x.iter().zip(y).map(|(u, w)| (u - w) * (u - w)).sum();
        [d, y.iter().map(|w| w * w).sum()]
    };
    let relative = |[d, n]: [f64; 2]| {
        let [d, n] = [d, n].map(|s| global_sum(s, comm));
        if n == 0.0 { d.sqrt() } else { (d / n).sqrt() }
    };
    let phi = relative(sums(&a.potential, &b.potential));
    let grad = match (&a.gradient, &b.gradient) {
        (Some(x), Some(y)) => relative(sums(x.as_flattened(), y.as_flattened())),
        _ => 0.0,
    };
    (phi, grad)
}

/// The P2P kernels (module documentation): `Reference` and every available ISA.
fn check_kernels(scenario: &Scenario, comm: &SimpleCommunicator) {
    let rank = comm.rank();
    let inputs = Inputs::new(scenario, 1, comm);
    let builder = builder(scenario);
    let kind = scenario.charges;
    let (fmm, reference) = run(
        &builder.clone().p2p_kernel(P2pChoice::Reference),
        &inputs,
        kind,
        comm,
    );
    assert_eq!(fmm.p2p_kernel(), P2pChoice::Reference);
    drop(fmm);
    for isa in Isa::all().filter(|isa| isa.is_available()) {
        let (fmm, output) = run(
            &builder.clone().p2p_kernel(P2pChoice::Isa(isa)),
            &inputs,
            kind,
            comm,
        );
        assert_eq!(fmm.p2p_kernel(), P2pChoice::Isa(isa));
        drop(fmm);
        if isa == Isa::Scalar {
            assert_same(
                &format!("{}: Isa::Scalar against Reference", scenario.name),
                &output,
                &reference,
                comm,
            );
        }
        if isa == Isa::detect() {
            let (fmm, auto) = run(&builder, &inputs, kind, comm);
            assert_eq!(fmm.p2p_kernel(), P2pChoice::Isa(isa));
            assert_same(
                &format!("{}: Auto against Isa({isa})", scenario.name),
                &auto,
                &output,
                comm,
            );
        }
        let (phi, grad) = difference(&output, &reference, comm);
        let (error_phi, error_grad) = errors(scenario, &inputs, &output, 0, comm);
        if rank == 0 {
            println!(
                "{}: {isa} against Reference: relative L2 φ {phi:.2e}, ∇φ {grad:.2e}; \
                 against the direct sum φ {error_phi:.2e}, ∇φ {error_grad:.2e}",
                scenario.name
            );
        }
        assert!(
            phi <= KERNEL_TOLERANCE && grad <= KERNEL_TOLERANCE,
            "{}: {isa} against Reference: φ {phi:e}, ∇φ {grad:e}, tolerance {KERNEL_TOLERANCE:e}",
            scenario.name
        );
        assert!(
            error_phi < scenario.bound && error_grad < scenario.bound,
            "{}: {isa}: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against the direct sum",
            scenario.name
        );
    }
    if rank == 0 {
        println!("{}: P2P kernels passed", scenario.name);
    }
}

/// The Dense M2L (Helmholtz CH3.2a), the Classes M2L (Helmholtz CH3.2b) or the Batched M2L
/// (Helmholtz CH3.2c; module documentation): within the scenario's bound of the direct sum
/// and within `DENSE_TOLERANCE` of Rotation's output; 2 threads, pair by pair and the
/// one-rank FMM bit for bit; where more than one level has V lists, a budget that leaves
/// the coarser levels to RCR within the tolerance too (in debug builds Dense's and
/// Batched's only run of such a tree, for time; Classes builds 16 offsets a level and runs
/// both). Batched also against `dense`, Dense's output of this check on the same levels: bit
/// for bit with real multipoles, within the tolerance with complex ones. Returns the output.
fn check_tables(
    scenario: &Scenario,
    m2l: HelmholtzM2l,
    dense: Option<&Output<f64>>,
    comm: &SimpleCommunicator,
) -> Output<f64> {
    let start = Instant::now();
    let rank = comm.rank();
    let name = format!("{}: {m2l:?}", scenario.name);
    let tolerance = DENSE_TOLERANCE;
    // The budget of the finest level's tables alone.
    let finest_budget = |p: usize| match m2l {
        HelmholtzM2l::Classes => {
            ClassM2l::<f64>::level_bytes(p) + CubeTransforms::<f64>::bytes_for(p)
        }
        _ => DenseM2l::<f64>::level_bytes(p),
    };
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let (fmm, rotation) = run(&builder(scenario), &inputs, kind, comm);
    let degrees = fmm.degrees().to_vec();
    drop(fmm);
    // Debug builds give a tree with more than one level of V lists the finest level's
    // dense tables alone (RCR on the coarser levels): the dense builds dominate there.
    let finest = degrees.len() - 1;
    let finest_only = cfg!(debug_assertions)
        && finest > 2
        && matches!(m2l, HelmholtzM2l::Dense | HelmholtzM2l::Batched);
    let mut tables = builder(scenario).m2l(m2l);
    if finest_only {
        tables = tables.dense_budget(finest_budget(degrees[finest]));
    }
    let (fmm, output) = run(&tables, &inputs, kind, comm);
    let op = fmm.operator();
    assert_eq!(op.m2l_choice(), m2l);
    for level in 2..fmm.nlevels() {
        let want = match finest_only && level < finest {
            true => HelmholtzM2l::Rotation,
            false => m2l,
        };
        assert_eq!(op.m2l_on(level), want, "level {level}");
    }
    let bytes = op.dense_bytes() + op.class_bytes() + op.batched_bytes();
    drop(fmm);
    let (phi, grad) = difference(&output, &rotation, comm);
    let (error_phi, error_grad) = errors(scenario, &inputs, &output, 0, comm);
    if rank == 0 {
        println!(
            "{name} ({:.1} MB) against Rotation: relative L2 φ {phi:.2e}, ∇φ {grad:.2e}; \
             against the direct sum φ {error_phi:.2e}, ∇φ {error_grad:.2e}",
            bytes as f64 / 1e6
        );
    }
    assert!(
        phi <= tolerance && grad <= tolerance,
        "{name} against Rotation: φ {phi:e}, ∇φ {grad:e}"
    );
    assert!(
        error_phi < scenario.bound && error_grad < scenario.bound,
        "{name}: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against the direct sum"
    );
    let (_, threaded) = run(&tables.clone().threads(2), &inputs, kind, comm);
    assert_same(&format!("{name}, 2 threads"), &threaded, &output, comm);
    let (_, pairs) = run(&tables.clone().pair_calls(true), &inputs, kind, comm);
    assert_same(&format!("{name}, pair by pair"), &pairs, &output, comm);
    check_one_rank(&name, &tables, &inputs, kind, &output, comm);
    if let Some(dense) = dense {
        // Dense on the same levels: one multipole plane (real k, real charges) gives Dense's
        // chain exactly, two give its sums to rounding.
        let one_plane = scenario.k.1 == 0.0 && kind == ChargeKind::Real;
        let (phi, grad) = difference(&output, dense, comm);
        if rank == 0 {
            println!("{name} against Dense: relative L2 φ {phi:.2e}, ∇φ {grad:.2e}");
        }
        if one_plane {
            assert_same(&format!("{name} against Dense"), &output, dense, comm);
        } else {
            assert!(
                phi <= tolerance && grad <= tolerance,
                "{name} against Dense: φ {phi:e}, ∇φ {grad:e}"
            );
        }
        // Two vectors: the block's m > 1 columns, the level call against the pair body bit
        // for bit.
        let two = Inputs::new(scenario, 2, comm);
        let vectors = tables.clone().vectors(2);
        let (_, level_calls) = run(&vectors, &two, kind, comm);
        let (_, pairs) = run(&vectors.clone().pair_calls(true), &two, kind, comm);
        assert_same(
            &format!("{name}, two vectors, pair by pair"),
            &pairs,
            &level_calls,
            comm,
        );
        // The fused mode: Rotation's and Dense's output to rounding; the one-rank FMM bit for
        // bit (threads and pair by pair as the separate mode's runs above show: the level call
        // and the pair body share their calls, and the mode only changes a step).
        let fused = tables.clone().m2l_mul_add(MulAdd::Fused);
        let name = format!("{name}, fused");
        let (_, output) = run(&fused, &inputs, kind, comm);
        let (phi, grad) = difference(&output, &rotation, comm);
        let (dense_phi, dense_grad) = difference(&output, dense, comm);
        if rank == 0 {
            println!(
                "{name} against Rotation: relative L2 φ {phi:.2e}, ∇φ {grad:.2e}; against \
                 Dense φ {dense_phi:.2e}, ∇φ {dense_grad:.2e}"
            );
        }
        assert!(
            phi.max(grad).max(dense_phi).max(dense_grad) <= tolerance,
            "{name}: φ {phi:e}, ∇φ {grad:e} against Rotation, φ {dense_phi:e}, ∇φ \
             {dense_grad:e} against Dense"
        );
        check_one_rank(&name, &fused, &inputs, kind, &output, comm);
    }

    // Also the finest level's tables alone, RCR on the coarser levels (with more than one
    // level of V lists), unless that was the run above.
    if finest > 2 && !finest_only {
        let budget = finest_budget(degrees[finest]);
        let (fmm, mixed) = run(&tables.clone().dense_budget(budget), &inputs, kind, comm);
        assert_eq!(fmm.operator().m2l_on(finest), m2l);
        assert_eq!(fmm.operator().m2l_on(2), HelmholtzM2l::Rotation);
        drop(fmm);
        let (phi, grad) = difference(&mixed, &rotation, comm);
        assert!(
            phi <= tolerance && grad <= tolerance,
            "{name} on the finest level against Rotation: φ {phi:e}, ∇φ {grad:e}"
        );
    }
    if rank == 0 {
        println!("{name} checks passed in {:.1?}", start.elapsed());
    }
    output
}

/// The rule per level, `Auto`, on the scenario's tree with one degree more on every level
/// but the finest and the finest level's degree named as the crossover: the finest level
/// runs fused Batched, the coarser levels Rotation (a split by degree wherever the tree has
/// more than one level with M2L), against Rotation of the same degrees and the direct sum,
/// threads, pair by pair and the one-rank FMM; the rule's own crossover (agreed over the
/// ranks) and a named mode.
fn check_auto(scenario: &Scenario, comm: &SimpleCommunicator) {
    let start = Instant::now();
    let rank = comm.rank();
    let name = format!("{}: Auto", scenario.name);
    let tolerance = DENSE_TOLERANCE;
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let (fmm, _) = run(&builder(scenario), &inputs, kind, comm);
    let mut degrees = fmm.degrees().to_vec();
    drop(fmm);
    let finest = degrees.len() - 1;
    let crossover = degrees[finest];
    for p in &mut degrees[2..finest] {
        *p += 1;
    }
    let split = Scenario {
        degrees: Degrees::PerLevel(degrees.clone()),
        ..scenario.clone()
    };
    let (_, rotation) = run(&builder(&split), &inputs, kind, comm);
    // CH3.2d's rule: the exponential M2L (D8's default) off; with it in check_auto_plane_wave.
    let auto = builder(&split)
        .m2l(HelmholtzM2l::Auto)
        .auto_plane_wave(false)
        .batched_term(false)
        .m2l_max_degree(crossover);
    let (fmm, output) = run(&auto, &inputs, kind, comm);
    let op = fmm.operator();
    assert_eq!(op.m2l_choice(), HelmholtzM2l::Auto);
    assert_eq!(fmm.m2l_max_degree(), Some(crossover));
    assert_eq!(op.m2l_mul_add(), auto_m2l_mul_add(Isa::detect()));
    assert_eq!(fmm.degrees(), degrees);
    assert_eq!(
        op.m2l_on(finest),
        HelmholtzM2l::Batched,
        "{name}, the finest level"
    );
    for level in 2..finest {
        assert_eq!(
            op.m2l_on(level),
            HelmholtzM2l::Rotation,
            "{name}, level {level}"
        );
    }
    check_m2l_times(&name, &fmm, comm);
    drop(fmm);
    let (phi, grad) = difference(&output, &rotation, comm);
    let (error_phi, error_grad) = errors(scenario, &inputs, &output, 0, comm);
    if rank == 0 {
        println!(
            "{name} (crossover {crossover}, degrees {degrees:?}) against Rotation: relative L2 \
             φ {phi:.2e}, ∇φ {grad:.2e}; against the direct sum φ {error_phi:.2e}, ∇φ \
             {error_grad:.2e}"
        );
    }
    assert!(
        phi <= tolerance && grad <= tolerance,
        "{name} against Rotation: φ {phi:e}, ∇φ {grad:e}"
    );
    assert!(
        error_phi < scenario.bound && error_grad < scenario.bound,
        "{name}: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against the direct sum"
    );
    let (_, threaded) = run(&auto.clone().threads(2), &inputs, kind, comm);
    assert_same(&format!("{name}, 2 threads"), &threaded, &output, comm);
    let (_, pairs) = run(&auto.clone().pair_calls(true), &inputs, kind, comm);
    assert_same(&format!("{name}, pair by pair"), &pairs, &output, comm);
    check_one_rank(&name, &auto, &inputs, kind, &output, comm);

    // The library's default M2L is Auto, at the rule's own crossover (no tables: a budget of
    // zero), the same on every rank; a named mode kept.
    let planes = match (scenario.k.1 == 0.0, kind) {
        (true, ChargeKind::Real) => 1,
        _ => 2,
    };
    let rule = default_builder(scenario)
        .dense_budget(0)
        .m2l_mul_add(MulAdd::Separate);
    let fmm = rule
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .unwrap_or_else(|error| panic!("rank {rank}: the build: {error}"));
    let own = auto_batched_max_degree(Isa::detect(), planes);
    let agreed = fmm.m2l_max_degree().expect("Auto has a crossover");
    let mut smallest = 0u64;
    comm.all_reduce_into(&(own as u64), &mut smallest, SystemOperation::min());
    assert_eq!(agreed as u64, smallest, "{name}: the agreed crossover");
    assert_eq!(
        fmm.operator().m2l_choice(),
        HelmholtzM2l::Auto,
        "the default"
    );
    assert_eq!(fmm.operator().m2l_mul_add(), MulAdd::Separate);
    assert_eq!(fmm.operator().batched_bytes(), 0);
    if rank == 0 {
        println!(
            "{name}: the rule's crossover {agreed} ({planes} planes); checks passed in {:.1?}",
            start.elapsed()
        );
    }
}

/// Auto's term of the levels' V lists (Helmholtz CH4.0), on the scenario's degrees with one
/// added on levels 2 to the finest but one (as `check_auto`), the crossover above every
/// degree and the exponential M2L off: the shapes the same on every rank and the one-rank
/// FMM's; each level Batched exactly where its estimate of the columns per read reaches m*
/// of its degree, m* the agreed table (every rank's largest); within `DENSE_TOLERANCE` of
/// Rotation, 2 threads (in release builds; in debug builds the width checks' runs at 2
/// threads) and the one-rank FMM bit for bit. The library's default (the term and
/// the exponential M2L on) against the direct sum and the one-rank FMM. Batched on every level
/// at named block widths, each the default width's bits (blocks of 1, 7 and 64 at 1 and 2
/// threads; in debug builds 1 at 1 thread and 7 at 2). The dense budget per rank: the node
/// budget over the ranks per node. Returns whether a level ran Rotation by the term alone.
fn check_batched_term(scenario: &Scenario, comm: &SimpleCommunicator) -> bool {
    let start = Instant::now();
    let rank = comm.rank();
    let name = format!("{}: Batched's term", scenario.name);
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let (fmm, _) = run(&builder(scenario), &inputs, kind, comm);
    let mut degrees = fmm.degrees().to_vec();
    drop(fmm);
    let finest = degrees.len() - 1;
    for p in &mut degrees[2..finest] {
        *p += 1;
    }
    let scenario = &Scenario {
        degrees: Degrees::PerLevel(degrees.clone()),
        ..scenario.clone()
    };
    let (_, rotation) = run(&builder(scenario), &inputs, kind, comm);
    let auto = builder(scenario)
        .m2l(HelmholtzM2l::Auto)
        .auto_plane_wave(false)
        .m2l_max_degree(usize::MAX);
    let (fmm, output) = run(&auto, &inputs, kind, comm);
    let shapes = fmm
        .level_shapes()
        .expect("the term is on by default")
        .to_vec();
    assert_eq!(shapes.len(), degrees.len());
    let planes = fmm.densities().multipole_planes();
    // The column kernel runs on the P2P kernel's ISA, the scalar one for the reference.
    let isa = match fmm.operator().p2p_kernel() {
        P2pChoice::Isa(isa) => isa,
        _ => Isa::Scalar,
    };
    // m*: the largest of every rank's table, which one machine class makes its own.
    let local: Vec<f64> = (0..=20)
        .map(|p| auto_batched_min_columns(isa, planes, p))
        .collect();
    let mut agreed = vec![0.0f64; local.len()];
    comm.all_reduce_into(&local[..], &mut agreed[..], SystemOperation::max());
    for (p, &m) in agreed.iter().enumerate() {
        assert_eq!(
            fmm.operator().m2l_min_columns(p),
            m,
            "{name}: m* at p = {p}"
        );
    }
    let mut on = Vec::new();
    for (level, (shape, &p)) in shapes.iter().zip(&degrees).enumerate().skip(2) {
        let want = if shape.batched_columns() >= agreed[p] {
            HelmholtzM2l::Batched
        } else {
            HelmholtzM2l::Rotation
        };
        assert_eq!(fmm.operator().m2l_on(level), want, "{name}, level {level}");
        on.push(want);
    }
    // The shapes: every rank's the same, and the one-rank tree's.
    let flat: Vec<u64> = shapes.iter().flat_map(|s| [s.pairs, s.targets]).collect();
    let mut low = vec![0u64; flat.len()];
    let mut high = vec![0u64; flat.len()];
    comm.all_reduce_into(&flat[..], &mut low[..], SystemOperation::min());
    comm.all_reduce_into(&flat[..], &mut high[..], SystemOperation::max());
    assert_eq!(low, high, "{name}: the shapes differ between ranks");
    if comm.size() > 1 {
        let (sources, _) = union_points(&inputs.sources, comm);
        let (targets, _) = union_points(&inputs.targets, comm);
        let one = SimpleCommunicator::self_comm();
        let single = auto
            .clone()
            .build::<f64, _>(&sources, &targets, &one)
            .unwrap_or_else(|error| panic!("rank {rank}: the one-rank build: {error}"));
        assert_eq!(
            single.level_shapes(),
            Some(&shapes[..]),
            "{name}: not the one-rank tree's shapes"
        );
    }
    check_m2l_times(&name, &fmm, comm);
    drop(fmm);
    let (phi, grad) = difference(&output, &rotation, comm);
    if rank == 0 {
        let columns: Vec<String> = shapes[2..]
            .iter()
            .map(|s| format!("{:.1}", s.batched_columns()))
            .collect();
        println!(
            "{name} (degrees {degrees:?}; columns per read {columns:?}; levels {on:?}) against \
             Rotation: relative L2 φ {phi:.2e}, ∇φ {grad:.2e}"
        );
    }
    assert!(
        phi <= DENSE_TOLERANCE && grad <= DENSE_TOLERANCE,
        "{name} against Rotation: φ {phi:e}, ∇φ {grad:e}"
    );
    // At 2 threads (in debug builds the width checks below run Batched at 2 threads).
    if !cfg!(debug_assertions) {
        let (_, threaded) = run(&auto.clone().threads(2), &inputs, kind, comm);
        assert_same(&format!("{name}, 2 threads"), &threaded, &output, comm);
    }
    check_one_rank(&name, &auto, &inputs, kind, &output, comm);

    // The library's default, the term and the exponential M2L on, at the rule's crossover.
    let default = default_builder(scenario);
    let (fmm, output) = run(&default, &inputs, kind, comm);
    assert!(fmm.level_shapes().is_some(), "{name}: the term by default");
    drop(fmm);
    let (error_phi, error_grad) = errors(scenario, &inputs, &output, 0, comm);
    assert!(
        error_phi < scenario.bound && error_grad < scenario.bound,
        "{name}, the default: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against the direct sum"
    );
    check_one_rank(
        &format!("{name}, the default"),
        &default,
        &inputs,
        kind,
        &output,
        comm,
    );

    // The split: Batched on every level (no term) at the default width and at named ones.
    let every = auto.clone().batched_term(false);
    let (fmm, wide) = run(&every, &inputs, kind, comm);
    assert!((2..=finest).all(|l| fmm.operator().m2l_on(l) == HelmholtzM2l::Batched));
    assert_eq!(fmm.level_shapes(), None);
    drop(fmm);
    let widths: &[(usize, usize)] = if cfg!(debug_assertions) {
        &[(1, 1), (7, 2)]
    } else {
        &[(1, 1), (1, 2), (7, 1), (7, 2), (64, 1), (64, 2)]
    };
    for &(width, threads) in widths {
        let (fmm, narrow) = run(
            &every.clone().m2l_block_targets(width).threads(threads),
            &inputs,
            kind,
            comm,
        );
        assert_eq!(fmm.operator().m2l_block_targets(), Some(width));
        drop(fmm);
        assert_same(
            &format!("{name}, blocks of {width} at {threads} threads"),
            &narrow,
            &wide,
            comm,
        );
    }

    // The node budget: the default per rank (read from a build without tables), and a node
    // budget of the finest level's tables per rank, which batches the finest level alone.
    let fmm = builder(scenario)
        .dense_budget(0)
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .unwrap_or_else(|error| panic!("rank {rank}: the build: {error}"));
    let ranks = fmm.ranks_per_node();
    assert_eq!(fmm.dense_budget(), 0);
    drop(fmm);
    let fmm = builder(scenario)
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .unwrap_or_else(|error| panic!("rank {rank}: the build: {error}"));
    assert_eq!(
        fmm.dense_budget(),
        DEFAULT_DENSE_BUDGET.min(DEFAULT_DENSE_NODE_BUDGET / ranks),
        "{name}: the default budget per rank"
    );
    drop(fmm);
    let finest_bytes = BatchedM2l::<f64>::level_bytes(degrees[finest]);
    let fmm = auto
        .clone()
        .batched_term(false)
        .dense_node_budget(finest_bytes * ranks)
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .unwrap_or_else(|error| panic!("rank {rank}: the build: {error}"));
    assert_eq!(fmm.dense_budget(), finest_bytes, "{name}: the node's share");
    assert_eq!(fmm.operator().m2l_on(finest), HelmholtzM2l::Batched);
    assert!((2..finest).all(|l| fmm.operator().m2l_on(l) == HelmholtzM2l::Rotation));
    drop(fmm);
    if rank == 0 {
        println!(
            "{name}: {ranks} rank(s) per node; checks passed in {:.1?}",
            start.elapsed()
        );
    }
    on.contains(&HelmholtzM2l::Rotation)
}

/// Auto with the exponential M2L (Helmholtz CH6.2h, D8), on the scenario's degrees with one
/// added on levels 2 to the finest but one (as `check_auto`), so that levels differ: with
/// the crossover and the smallest degree named 0 it is PlaneWave bit for bit; with the
/// crossover at the finest level's degree and the smallest degree one above it, each level's
/// method by the rule, against the direct sum within `PLANE_WAVE_RATIO` of Rotation's error,
/// 2 threads and the one-rank FMM bit for bit; the rule's smallest degree agreed over the
/// ranks, on by default (D8) and none while off.
fn check_auto_plane_wave(scenario: &Scenario, comm: &SimpleCommunicator) {
    let start = Instant::now();
    let rank = comm.rank();
    let name = format!("{}: Auto with waves", scenario.name);
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let built = |builder: &HelmholtzFmmBuilder| {
        builder
            .clone()
            .dense_budget(0)
            .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
            .unwrap_or_else(|error| panic!("rank {rank}: the build: {error}"))
    };
    let mut degrees = built(&builder(scenario)).degrees().to_vec();
    let finest = degrees.len() - 1;
    let crossover = degrees[finest];
    for p in &mut degrees[2..finest] {
        *p += 1;
    }
    let scenario = &Scenario {
        degrees: Degrees::PerLevel(degrees.clone()),
        ..scenario.clone()
    };
    let k_root = C::new(scenario.k.0, scenario.k.1) * 0.5;
    let rule = select(k_root, &degrees, None);
    // The levels split by degree alone: the term of the levels' V lists off (its check is
    // `check_batched_term`).
    let auto = builder(scenario)
        .m2l(HelmholtzM2l::Auto)
        .auto_plane_wave(true)
        .batched_term(false);

    // Every level the rule gives waves: PlaneWave's output.
    let (_, plane_wave) = run(
        &builder(scenario).m2l(HelmholtzM2l::PlaneWave),
        &inputs,
        kind,
        comm,
    );
    let every = auto.clone().m2l_max_degree(0).m2l_plane_wave_min_degree(0);
    let (fmm, output) = run(&every, &inputs, kind, comm);
    assert_eq!(fmm.m2l_plane_wave_min_degree(), Some(0));
    for (level, &precision) in rule.iter().enumerate().skip(2) {
        assert_eq!(fmm.operator().plane_wave_precision(level), precision);
    }
    drop(fmm);
    assert_same(&format!("{name}, PlaneWave"), &output, &plane_wave, comm);

    // The rule: Batched to the finest level's degree, waves above it.
    let (_, rotation) = run(&builder(scenario), &inputs, kind, comm);
    let split = auto
        .clone()
        .m2l_max_degree(crossover)
        .m2l_plane_wave_min_degree(crossover + 1);
    let (fmm, output) = run(&split, &inputs, kind, comm);
    let op = fmm.operator();
    let mut methods = Vec::new();
    for level in 2..degrees.len() {
        let want = if degrees[level] > crossover && rule[level].is_some() {
            HelmholtzM2l::PlaneWave
        } else if degrees[level] <= crossover {
            HelmholtzM2l::Batched
        } else {
            HelmholtzM2l::Rotation
        };
        assert_eq!(op.m2l_on(level), want, "{name}, level {level}");
        methods.push(want);
    }
    check_m2l_times(&name, &fmm, comm);
    drop(fmm);
    let (error_phi, error_grad) = errors(scenario, &inputs, &output, 0, comm);
    let (rotation_phi, rotation_grad) = errors(scenario, &inputs, &rotation, 0, comm);
    if rank == 0 {
        println!(
            "{name} (degrees {degrees:?}, levels 2– {methods:?}) against the direct sum: φ \
             {error_phi:.2e}, ∇φ {error_grad:.2e}; Rotation's φ {rotation_phi:.2e}, ∇φ \
             {rotation_grad:.2e}"
        );
    }
    assert!(
        error_phi <= PLANE_WAVE_RATIO * rotation_phi
            && error_grad <= PLANE_WAVE_RATIO * rotation_grad,
        "{name}: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against Rotation's {rotation_phi:.2e}, \
         {rotation_grad:.2e}"
    );
    let (_, threaded) = run(&split.clone().threads(2), &inputs, kind, comm);
    assert_same(&format!("{name}, 2 threads"), &threaded, &output, comm);
    check_one_rank(&name, &split, &inputs, kind, &output, comm);

    // The smallest degree above every level's: today's Auto bit for bit.
    let above = degrees.iter().max().unwrap() + 1;
    let (_, none) = run(
        &split.clone().m2l_plane_wave_min_degree(above),
        &inputs,
        kind,
        comm,
    );
    let today = builder(scenario)
        .m2l(HelmholtzM2l::Auto)
        .auto_plane_wave(false)
        .batched_term(false)
        .m2l_max_degree(crossover);
    let (_, today) = run(&today, &inputs, kind, comm);
    assert_same(&format!("{name}, above every degree"), &none, &today, comm);

    // The rule's smallest degree, agreed over the ranks; on by default (D8), none while off.
    let planes = match (scenario.k.1 == 0.0, kind) {
        (true, ChargeKind::Real) => 1,
        _ => 2,
    };
    let own = auto_plane_wave_min_degree(Isa::detect(), planes);
    let mut smallest = 0u64;
    comm.all_reduce_into(&(own as u64), &mut smallest, SystemOperation::min());
    assert_eq!(
        built(&auto).m2l_plane_wave_min_degree().map(|d| d as u64),
        Some(smallest),
        "{name}: the agreed smallest degree"
    );
    assert_eq!(
        built(&default_builder(scenario)).m2l_plane_wave_min_degree(),
        Some(smallest as usize),
        "{name}: on by default"
    );
    assert_eq!(
        built(&default_builder(scenario).auto_plane_wave(false)).m2l_plane_wave_min_degree(),
        None,
        "{name}: off"
    );
    if rank == 0 {
        println!(
            "{name}: the rule's smallest degree {smallest} ({planes} planes); checks passed in \
             {:.1?}",
            start.elapsed()
        );
    }
}

/// The exponential M2L (module documentation): the rule's precisions, the direct sum and
/// Rotation's error, the pair path, threads, the one-rank FMM, two vectors and named
/// precisions.
/// With `bounds`, also the bounded waves buffer (`check_plane_wave_buffer`); returns
/// whether its floor took more chunks than the default budget on some rank. On the plan
/// without the empty-box rule (Helmholtz CH4.0d1), so that the clusters' finest level keeps
/// more V sources than the floor holds; `check_empty_boxes` runs PlaneWave with the rule.
fn check_plane_wave(scenario: &Scenario, bounds: bool, comm: &SimpleCommunicator) -> bool {
    let start = Instant::now();
    let rank = comm.rank();
    let name = format!("{}: PlaneWave", scenario.name);
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let full = builder(scenario).empty_boxes(false);
    let (fmm, rotation) = run(&full, &inputs, kind, comm);
    let degrees = fmm.degrees().to_vec();
    drop(fmm);
    // The domain is the unit cube, r₀ = ½.
    let k_root = C::new(scenario.k.0, scenario.k.1) * 0.5;
    let tolerance = match scenario.degrees {
        Degrees::Tolerance(epsilon) => Some(epsilon),
        Degrees::PerLevel(_) => None,
    };
    let rule = select(k_root, &degrees, tolerance);
    let tables = full.clone().m2l(HelmholtzM2l::PlaneWave);
    let (fmm, output) = run(&tables, &inputs, kind, comm);
    let op = fmm.operator();
    assert_eq!(op.m2l_choice(), HelmholtzM2l::PlaneWave);
    for (level, &precision) in rule.iter().enumerate().skip(2) {
        assert_eq!(op.plane_wave_precision(level), precision, "level {level}");
        let want = match precision {
            Some(_) => HelmholtzM2l::PlaneWave,
            None => HelmholtzM2l::Rotation,
        };
        assert_eq!(op.m2l_on(level), want, "{name}, level {level}");
    }
    assert!(
        rule.iter().any(Option::is_some),
        "{name}: a level with waves"
    );
    let bytes = op.plane_wave_bytes();
    let (whole_buffer, whole_counts) = (op.plane_wave_buffer_bytes(), op.plane_wave_counts());
    let ranks_per_node = fmm.ranks_per_node();
    assert!((1..=comm.size() as usize).contains(&ranks_per_node));
    assert_eq!(
        op.plane_wave_buffer(),
        Some(DEFAULT_PLANE_WAVE_NODE_BUDGET / ranks_per_node),
        "{name}: the node budget over the ranks per node"
    );
    let mut calls = 0usize;
    comm.all_reduce_into(
        &whole_counts.level_calls,
        &mut calls,
        SystemOperation::sum(),
    );
    assert!(calls > 0, "{name}: level calls with waves");
    assert!(
        everywhere(whole_counts.chunks <= whole_counts.level_calls, comm),
        "{name}: one chunk per level call within the default budget"
    );
    check_m2l_times(&name, &fmm, comm);
    check_plane_wave_times(&name, &fmm, comm);
    assert_eq!(op.plane_wave_reach(), RHO_MAX);
    assert_eq!(op.plane_wave_split(), PlaneWaveSplit::Auto);
    let corner_waves: Vec<usize> = (0..degrees.len()).map(|l| op.plane_wave_waves(l)).collect();
    drop(fmm);
    let (phi, grad) = difference(&output, &rotation, comm);
    let (error_phi, error_grad) = errors(scenario, &inputs, &output, 0, comm);
    let (rotation_phi, rotation_grad) = errors(scenario, &inputs, &rotation, 0, comm);
    if rank == 0 {
        let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
        println!(
            "{name} (precisions {rule:?}, {:.1} MB) against the direct sum: φ {error_phi:.2e}, \
             ∇φ {error_grad:.2e}, {:.3}× and {:.3}× Rotation's; against Rotation φ {phi:.2e}, \
             ∇φ {grad:.2e}",
            bytes as f64 / 1e6,
            ratio(error_phi, rotation_phi),
            ratio(error_grad, rotation_grad),
        );
    }
    assert!(
        error_phi < scenario.bound && error_grad < scenario.bound,
        "{name}: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against the direct sum"
    );
    assert!(
        error_phi <= PLANE_WAVE_RATIO * rotation_phi
            && error_grad <= PLANE_WAVE_RATIO * rotation_grad,
        "{name}: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against Rotation's {rotation_phi:.2e}, \
         {rotation_grad:.2e}"
    );
    let (fmm, threaded) = run(&tables.clone().threads(2), &inputs, kind, comm);
    let threaded_counts = fmm.operator().plane_wave_counts();
    drop(fmm);
    assert_same(&format!("{name}, 2 threads"), &threaded, &output, comm);
    check_one_rank(&name, &tables, &inputs, kind, &output, comm);
    check_plane_wave_split(
        &name,
        &tables,
        (&inputs, kind, &output),
        (whole_counts, threaded_counts),
        comm,
    );

    // The bounded buffer, on the scenarios the caller names (in debug builds the one whose
    // levels chunk).
    let floor_chunked = bounds
        && check_plane_wave_buffer(
            &name,
            &tables,
            (&inputs, kind, &output),
            (bytes, whole_buffer, whole_counts),
            ranks_per_node,
            comm,
        );

    // Two vectors: each the one-vector FMM bit for bit.
    let two = Inputs::new(scenario, 2, comm);
    let (_, both) = run(&tables.clone().vectors(2), &two, kind, comm);
    for v in 0..2 {
        let one = Inputs {
            sources: two.sources.clone(),
            targets: two.targets.clone(),
            charges: vec![two.charges[v].clone()],
        };
        let (_, single) = run(&tables, &one, kind, comm);
        let column = |output: &Output<f64>, v: usize| {
            let phi = output.potential_of(v);
            let mut bits: Vec<u64> = phi
                .re
                .iter()
                .chain(phi.im.unwrap_or(&[]))
                .map(|x| x.to_bits())
                .collect();
            if let Some(g) = output.gradient_of(v) {
                let im = g.im.unwrap_or(&[]);
                bits.extend(g.re.iter().chain(im).flatten().map(|x| x.to_bits()));
            }
            bits
        };
        assert!(
            everywhere(column(&both, v) == column(&single, 0), comm),
            "{name}: vector {v} of two is not the one-vector FMM bit for bit"
        );
    }

    // The counts' reach (Helmholtz CH6.2e): the corner's named is the default bit for bit;
    // FMM3D's reach 4 has at most the corner's waves on every level, fewer on one, within
    // PLANE_WAVE_RATIO of Rotation's error, the same at 2 threads and on P ranks the
    // one-rank FMM.
    let (_, corner) = run(
        &tables.clone().plane_wave_reach(RHO_MAX),
        &inputs,
        kind,
        comm,
    );
    assert_same(
        &format!("{name}, the corner's reach"),
        &corner,
        &output,
        comm,
    );
    let reach = tables.clone().plane_wave_reach(4.0);
    let (fmm, fewer) = run(&reach, &inputs, kind, comm);
    let waves: Vec<usize> = (0..degrees.len())
        .map(|l| fmm.operator().plane_wave_waves(l))
        .collect();
    drop(fmm);
    assert!(
        waves.iter().zip(&corner_waves).all(|(a, b)| a <= b) && waves != corner_waves,
        "{name}: waves at reach 4 {waves:?}, at the corner {corner_waves:?}"
    );
    assert!(
        waves
            .iter()
            .zip(&corner_waves)
            .all(|(&a, &b)| (a == 0) == (b == 0)),
        "{name}: the same levels with waves at reach 4"
    );
    let (reach_phi, reach_grad) = errors(scenario, &inputs, &fewer, 0, comm);
    if rank == 0 {
        println!(
            "{name} at reach 4 (waves {waves:?} against {corner_waves:?}): φ {reach_phi:.2e}, \
             ∇φ {reach_grad:.2e}, {:.3}× and {:.3}× Rotation's",
            reach_phi / rotation_phi,
            reach_grad / rotation_grad,
        );
    }
    assert!(
        reach_phi <= PLANE_WAVE_RATIO * rotation_phi
            && reach_grad <= PLANE_WAVE_RATIO * rotation_grad,
        "{name} at reach 4: φ {reach_phi:.2e}, ∇φ {reach_grad:.2e} against Rotation's \
         {rotation_phi:.2e}, {rotation_grad:.2e}"
    );
    let (_, threaded) = run(&reach.clone().threads(2), &inputs, kind, comm);
    assert_same(
        &format!("{name}, reach 4, 2 threads"),
        &threaded,
        &fewer,
        comm,
    );
    check_one_rank(
        &format!("{name}, reach 4"),
        &reach,
        &inputs,
        kind,
        &fewer,
        comm,
    );

    // Named precisions: none on every level is Rotation's level call bit for bit; iprec 1
    // on the finest level and none above, the finest level alone.
    let none = tables
        .clone()
        .plane_wave_precision(PlaneWavePrecision::PerLevel(vec![None; degrees.len()]));
    let (fmm, all_rotation) = run(&none, &inputs, kind, comm);
    assert!((2..degrees.len()).all(|l| fmm.operator().m2l_on(l) == HelmholtzM2l::Rotation));
    assert_eq!(fmm.operator().plane_wave_bytes(), 0);
    drop(fmm);
    assert_same(
        &format!("{name}, no level named"),
        &all_rotation,
        &rotation,
        comm,
    );
    let finest = degrees.len() - 1;
    let mut named = vec![None; degrees.len()];
    named[finest] = Some(Precision::Iprec1);
    let one_level = tables
        .clone()
        .plane_wave_precision(PlaneWavePrecision::PerLevel(named));
    let (fmm, finest_only) = run(&one_level, &inputs, kind, comm);
    let op = fmm.operator();
    assert_eq!(op.plane_wave_precision(finest), Some(Precision::Iprec1));
    assert_eq!(op.m2l_on(finest), HelmholtzM2l::PlaneWave);
    assert!((2..finest).all(|l| op.m2l_on(l) == HelmholtzM2l::Rotation));
    drop(fmm);
    let (error_phi, error_grad) = errors(scenario, &inputs, &finest_only, 0, comm);
    assert!(
        error_phi < scenario.bound && error_grad < scenario.bound,
        "{name}, iprec 1 on the finest level: φ {error_phi:.2e}, ∇φ {error_grad:.2e}"
    );

    // The pair path: CH6.1c's translation of each pair along the merge's direction, one
    // conversion per pair. In debug builds of a tree with waves on several levels, on the
    // finest level's waves alone (its sparse parents; every level took 25 s on the M6).
    let several = rule.iter().filter(|p| p.is_some()).count() > 1;
    let (merged, merged_output) = match cfg!(debug_assertions) && several {
        true => (&one_level, &finest_only),
        false => (&tables, &output),
    };
    let (_, pairs) = run(&merged.clone().pair_calls(true), &inputs, kind, comm);
    let (phi, grad) = difference(merged_output, &pairs, comm);
    if rank == 0 {
        println!("{name}: merge and shift against the pair path φ {phi:.2e}, ∇φ {grad:.2e}");
    }
    assert!(
        phi <= PLANE_PAIR_TOLERANCE && grad <= PLANE_PAIR_TOLERANCE,
        "{name} against the pair path: φ {phi:e}, ∇φ {grad:e}"
    );
    if rank == 0 {
        println!("{name} checks passed in {:.1?}", start.elapsed());
    }
    let mut chunked = false;
    comm.all_reduce_into(&floor_chunked, &mut chunked, SystemOperation::logical_or());
    chunked
}

/// The exponential M2L below ε = 10⁻⁶ (Helmholtz CH6.1g; CH6.1f's resolving degree): at
/// ε = 10⁻⁷ and 10⁻⁸ under PlaneWave's rule every level with a quadrature runs iprec 3, at
/// max(p_l, 28) and at max(p_l, 32), the others at the degrees
/// given; the errors against the direct sum within [`PLANE_WAVE_RATIO`] of Rotation's at
/// the degrees given (CH6.1's acceptance) and within the scenario's bound; 2 threads and,
/// on several ranks, the one-rank FMM bit for bit. The clusters at kD = 1 have levels with
/// waves at κ_l ≤ 0.08, the clustered trees' deep levels where CH6.1e's iprec 3 at p_l
/// missed 1.1×; the smallest |κ_l| of a level with waves and M2L pairs is agreed over the
/// ranks. PlaneWave runs the default `PlaneWaveRaise::Rule` (CH6.1h: the M2L alone at 10⁻⁷,
/// the leaves too at 10⁻⁸), its translations at the degrees given; [`check_m2l_raise`] the
/// named forms. The default M2L (Auto, D8's bound lifted at 10⁻⁸ ≤ ε < 10⁻⁷ by CH6.1g's
/// sign-off and at 10⁻⁷ ≤ ε < 10⁻⁶ by CH6.1h's, 2026-11-11): PlaneWave's output bit for bit
/// (every level's degree above Batched's crossover). Returns the smallest |κ_l| of a level
/// with waves and M2L pairs.
fn check_resolving_degree(scenario: &Scenario, comm: &SimpleCommunicator) -> f64 {
    let start = Instant::now();
    let rank = comm.rank();
    let Degrees::Tolerance(epsilon) = scenario.degrees else {
        panic!("{}: the rule reads a tolerance", scenario.name);
    };
    let name = format!("{}: PlaneWave at the resolving degree", scenario.name);
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let (fmm, rotation) = run(&builder(scenario), &inputs, kind, comm);
    let given = fmm.degrees().to_vec();
    drop(fmm);
    // The domain is the unit cube, r₀ = ½, so κ_l = 2 k r_l = k 2⁻ˡ.
    let k = C::new(scenario.k.0, scenario.k.1);
    let rule = select(k * 0.5, &given, Some(epsilon));
    let tables = builder(scenario).m2l(HelmholtzM2l::PlaneWave);
    let (fmm, output) = run(&tables, &inputs, kind, comm);
    let op = fmm.operator();
    let mut deep = f64::INFINITY;
    for (level, &precision) in rule.iter().enumerate() {
        assert_eq!(
            op.plane_wave_precision(level),
            precision,
            "{name}, level {level}"
        );
        let want = match precision {
            Some(_) => plane_wave_degree(given[level], epsilon),
            None => given[level],
        };
        assert_eq!(fmm.degrees()[level], want, "{name}, level {level}");
        assert_eq!(op.base_degree(level), given[level], "{name}, level {level}");
        assert_eq!(
            op.translation_degree(level),
            given[level],
            "{name}, level {level}"
        );
        if precision.is_some() && op.m2l_times()[level] > Duration::ZERO {
            deep = deep.min((k * 0.5f64.powi(level as i32)).norm());
        }
    }
    assert!(
        rule.iter().any(|&q| q.is_some()) && fmm.degrees() != given,
        "{name}: a raised level with waves"
    );
    let raised = fmm.degrees().to_vec();
    drop(fmm);
    // The deepest level with waves and M2L pairs on any rank, so that every rank asserts on
    // the same value.
    let mut deepest = f64::INFINITY;
    comm.all_reduce_into(&deep, &mut deepest, SystemOperation::min());
    let deep = deepest;
    let (error_phi, error_grad) = errors(scenario, &inputs, &output, 0, comm);
    let (rotation_phi, rotation_grad) = errors(scenario, &inputs, &rotation, 0, comm);
    let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
    if rank == 0 {
        println!(
            "{name} (degrees {given:?} → {raised:?}, precisions {rule:?}, smallest |κ_l| with \
             M2L pairs {deep:.3}) against the direct sum: φ {error_phi:.2e}, ∇φ \
             {error_grad:.2e}, {:.3}× and {:.3}× Rotation's",
            ratio(error_phi, rotation_phi),
            ratio(error_grad, rotation_grad),
        );
    }
    assert!(
        error_phi < scenario.bound && error_grad < scenario.bound,
        "{name}: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against the direct sum"
    );
    assert!(
        error_phi <= PLANE_WAVE_RATIO * rotation_phi
            && error_grad <= PLANE_WAVE_RATIO * rotation_grad,
        "{name}: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against Rotation's {rotation_phi:.2e}, \
         {rotation_grad:.2e}"
    );
    let (_, threaded) = run(&tables.clone().threads(2), &inputs, kind, comm);
    assert_same(&format!("{name}, 2 threads"), &threaded, &output, comm);
    check_one_rank(&name, &tables, &inputs, kind, &output, comm);
    check_m2l_raise(
        scenario,
        &tables,
        (&inputs, &given, &raised, &output),
        (rotation_phi, rotation_grad),
        comm,
    );
    let (fmm, auto) = run(&default_builder(scenario), &inputs, kind, comm);
    assert_eq!(fmm.operator().m2l_choice(), HelmholtzM2l::Auto);
    let max_degree = fmm.m2l_max_degree().expect("Auto's crossover");
    assert!(
        given.iter().skip(2).all(|&p| p > max_degree),
        "{name}: above the crossover"
    );
    assert_eq!(fmm.degrees(), raised, "{name}: Auto's degrees");
    assert_same(&format!("{name}, Auto"), &auto, &output, comm);
    drop(fmm);
    // The cubes at kD = 20 + 2i pass no class at p_w, so their levels keep the waves: in
    // release builds only.
    if matches!(scenario.layout, Layout::Cube) && !cfg!(debug_assertions) {
        check_time_rule_raised(scenario, comm);
    }
    if rank == 0 {
        println!("{name}: {:.1?}", start.elapsed());
    }
    deep
}

/// Costs under which RCR beats the form and the form the waves (the diagonal pair dear, the
/// waves dearer).
const FREE_RCR: DiagonalCosts = DiagonalCosts {
    pair: [1e9, 1e9],
    rotation: [0.0, 0.0],
    ..FREE_DIAGONAL
};

/// The diagonal form's time rule at matched accuracy (Helmholtz CH4.3b2a) on a scenario whose
/// levels raised by the defaults' waves lie in the rule's range (the cubes of
/// [`check_resolving_degree`] at kD = 20 + 2i, level 2 at k a 4.3, where no class passes at p_w,
/// in release builds;
/// [`time_rule_scenario`] at kD = 40 + 2i, level 2 at k a 8.7, where some do), under the
/// defaults: at the machine's costs, at free form costs (the form where a class passes at p_w)
/// and at free RCR costs (RCR on the same levels, no other), the degrees and translation degrees
/// the defaults', the errors within [`DIAGONAL_RATIO`] of the defaults', 2 threads, chunks of 3
/// at free form costs and on P ranks the one-rank FMM bit for bit; on the cubes at kD = 20 + 2i
/// no level taken and the defaults' output bit for bit. Returns the most classes a raised level
/// took from the waves at free form costs.
fn check_time_rule_raised(scenario: &Scenario, comm: &SimpleCommunicator) -> usize {
    let inputs = &Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let defaults = default_builder(scenario);
    let (fmm, auto) = run(&defaults, inputs, kind, comm);
    let nlevels = fmm.nlevels();
    let op = fmm.operator();
    let waves: Vec<bool> = (0..nlevels)
        .map(|l| op.m2l_on(l) == HelmholtzM2l::PlaneWave)
        .collect();
    let translations: Vec<usize> = (0..nlevels).map(|l| op.translation_degree(l)).collect();
    let given: Vec<usize> = (0..nlevels).map(|l| op.base_degree(l)).collect();
    let raised = fmm.degrees().to_vec();
    drop(fmm);
    let plain = errors(scenario, inputs, &auto, 0, comm);
    let mut most = 0;
    let mut form_levels: Option<Vec<usize>> = None;
    // In debug builds the forced costs alone, and 2 threads at free form costs alone (each build
    // at p_w = 28 costs seconds at opt-level 0).
    let debug = cfg!(debug_assertions);
    let all = [
        ("its machine's costs", None),
        ("free form costs", Some(FREE_DIAGONAL)),
        ("free RCR costs", Some(FREE_RCR)),
    ];
    for &(costs_name, costs) in &all[usize::from(debug)..] {
        let what = format!(
            "{}: the time rule at the resolving degree, {costs_name}",
            scenario.name
        );
        let rule = defaults
            .clone()
            .diagonal(HelmholtzDiagonal::Auto)
            .diagonal_costs(costs);
        let (fmm, want) = run(&rule, inputs, kind, comm);
        let op = fmm.operator();
        assert_eq!(fmm.degrees(), &raised[..], "{what}: the degrees");
        let got: Vec<usize> = (0..nlevels).map(|l| op.translation_degree(l)).collect();
        assert_eq!(got, translations, "{what}: the translation degrees");
        let taken: Vec<(usize, usize, usize)> = (0..nlevels)
            .filter(|&l| waves[l] && op.m2l_on(l) != HelmholtzM2l::PlaneWave)
            .map(|l| {
                let classes = op.diagonal_classes(l).map_or(0, |c| c.len());
                (l, raised[l], classes)
            })
            .collect();
        let any_form = (0..nlevels).any(|l| op.diagonal_classes(l).is_some());
        drop(fmm);
        // These scenarios run at 10⁻⁷ and 10⁻⁸ with p_l below 28: every level with waves raised.
        assert!(
            taken.iter().all(|&(l, p_w, _)| p_w > given[l]),
            "{what}: a level taken from the waves not raised: {taken:?}"
        );
        let levels: Vec<usize> = taken.iter().map(|&(l, ..)| l).collect();
        if costs == Some(FREE_DIAGONAL) {
            // The form where a class passes at p_w, the waves where none does.
            assert!(
                taken.iter().all(|&(_, _, c)| c > 0),
                "{what}: a level taken without a class: {taken:?}"
            );
            most = taken.iter().map(|&(_, _, c)| c).max().unwrap_or(0);
            form_levels = Some(levels.clone());
        }
        if costs == Some(FREE_RCR) {
            // RCR on the levels the form would take, never against the waves alone.
            assert!(
                taken.iter().all(|&(_, _, c)| c == 0),
                "{what}: the form at free RCR costs: {taken:?}"
            );
            if let Some(form) = &form_levels {
                assert_eq!(&levels, form, "{what}: not the form's levels");
            }
        }
        if taken.is_empty() && !any_form {
            assert_same(&format!("{what}: no level changed"), &want, &auto, comm);
        }
        let errors = errors(scenario, inputs, &want, 0, comm);
        if comm.rank() == 0 {
            println!(
                "{what}: (level, p_w, classes) taken from the waves {taken:?}; φ {:.3e} against \
                 the defaults' {:.3e} ({:.3}×), ∇φ {:.3e} against {:.3e}",
                errors.0,
                plain.0,
                errors.0 / plain.0,
                errors.1,
                plain.1
            );
        }
        assert!(
            errors.0 <= DIAGONAL_RATIO * plain.0 && errors.1 <= DIAGONAL_RATIO * plain.1,
            "{what}: errors {errors:?} against the defaults' {plain:?}"
        );
        if costs == Some(FREE_DIAGONAL) || (costs.is_some() && !debug) {
            let (_, threaded) = run(&rule.clone().threads(2), inputs, kind, comm);
            assert_same(&format!("{what}, 2 threads"), &threaded, &want, comm);
        }
        if costs == Some(FREE_DIAGONAL) {
            let (_, chunked) = run(&rule.clone().diagonal_chunk(Some(3)), inputs, kind, comm);
            assert_same(&format!("{what}, chunks of 3"), &chunked, &want, comm);
        }
        if costs.is_some() {
            check_one_rank(&what, &rule, inputs, kind, &want, comm);
        }
    }
    most
}

/// Costs under which the form takes every level with waves where a class passes (the waves and
/// RCR far dearer) with a pair far dearer than the V rule's per-pair constant of the waves, so
/// that the crossover after the plan rises on those levels.
const DEAR_FORM_PAIR: DiagonalCosts = DiagonalCosts {
    s2f: [0.0, 0.0],
    f2r: [0.0, 0.0],
    pair: [1e4, 0.0],
    rotation: [1e12, 1e12],
    plane_wave: [1e12, 1e12],
};

/// Costs under which every level keeps its waves (the form dearer still), so that only
/// `VAfterPlan::AllLevels` raises a crossover: the waves' level cost over its pairs.
const DEARER_FORM: DiagonalCosts = DiagonalCosts {
    pair: [1e14, 1e14],
    ..DEAR_FORM_PAIR
};

/// The V rule after the plan (Helmholtz CH4.3b2b; `HelmholtzFmmBuilder::v_after_plan`) on
/// `scenario` under the cost model's V rule: Auto with waves from degree 0 and the time rule at
/// [`DEAR_FORM_PAIR`] (the form takes levels with waves) under `TakenLevels` and `AllLevels`, at
/// [`FREE_RCR`] (RCR takes them) under `TakenLevels`, and at [`DEARER_FORM`] (the waves kept) under `AllLevels`: the M2L per level the setting
/// off's; the crossovers after the plan at least those before, above them exactly on the levels
/// the setting names whose first plan holds V pairs between leaves (the levels taken, and under `AllLevels` those that keep
/// their waves), a second plan exactly then; the V pairs of every level at most, the converted V
/// pairs and the near interactions at least the setting off's; the errors against the direct sum
/// within [`NEAR_ERROR_BOUND`] of the setting off's; 2 threads and on P ranks the one-rank FMM
/// bit for bit. Under Rotation nothing rises: no second plan, the setting off's output bit for
/// bit. `RaisedLevels` (Helmholtz CH4.3b2d) at both [`DEAR_FORM_PAIR`] and [`DEARER_FORM`] names
/// neither the levels taken nor the levels that keep their waves at the degree given, so on a
/// scenario without raised levels (ε ≥ 10⁻⁶) nothing rises where `TakenLevels` and `AllLevels`
/// do; and `VRaisedPrice::PerBox` counts nothing there, the per-pair price's output bit for bit.
/// The V costs are named (NEON's), so the plans do not depend on the machine class.
fn check_v_after_plan(scenario: &Scenario, comm: &SimpleCommunicator) {
    let inputs = &Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    // NEON's V costs named, so that the plans (and the levels left with leaf pairs to raise)
    // are the same on every machine class: x86's waves' crossover is about 11 times NEON's,
    // and its first plan converts every leaf pair of the sphere's level 3.
    let base = default_builder(scenario)
        .near_v(VNear::CostModel)
        .v_costs(Some(V_COSTS_NEON));
    let auto = base
        .clone()
        .m2l_plane_wave_min_degree(0)
        .diagonal(HelmholtzDiagonal::Auto);
    let sum = |x: u64| {
        let mut total = 0u64;
        comm.all_reduce_into(&x, &mut total, SystemOperation::sum());
        total
    };
    let mut raised_any = [false; 2];
    for (setting, off, choice) in [
        (
            "the form taking the waves",
            auto.clone().diagonal_costs(Some(DEAR_FORM_PAIR)),
            VAfterPlan::TakenLevels,
        ),
        (
            "the form taking the waves",
            auto.clone().diagonal_costs(Some(DEAR_FORM_PAIR)),
            VAfterPlan::AllLevels,
        ),
        // RCR taking the levels the form would (`FREE_RCR`): RCR's pair after the plan.
        (
            "RCR taking the waves",
            auto.clone().diagonal_costs(Some(FREE_RCR)),
            VAfterPlan::TakenLevels,
        ),
        (
            "the waves kept",
            auto.clone().diagonal_costs(Some(DEARER_FORM)),
            VAfterPlan::AllLevels,
        ),
        // `RaisedLevels` (CH4.3b2d): neither the levels taken nor those that keep their waves
        // unraised.
        (
            "the form taking the waves",
            auto.clone().diagonal_costs(Some(DEAR_FORM_PAIR)),
            VAfterPlan::RaisedLevels,
        ),
        (
            "the waves kept",
            auto.clone().diagonal_costs(Some(DEARER_FORM)),
            VAfterPlan::RaisedLevels,
        ),
        (
            "Rotation",
            base.clone().m2l(HelmholtzM2l::Rotation),
            VAfterPlan::AllLevels,
        ),
    ] {
        let what = format!(
            "{}, {setting}, the V rule after the plan {choice}",
            scenario.name
        );
        let (fmm_off, output_off) = run(&off, inputs, kind, comm);
        assert_eq!(fmm_off.v_after_plan(), VAfterPlan::Off, "{what}");
        assert!(
            !fmm_off.replanned() && fmm_off.v_crossovers_after_plan().is_none(),
            "{what}"
        );
        let nlevels = fmm_off.nlevels();
        let op = fmm_off.operator();
        let m2l_off: Vec<(HelmholtzM2l, Option<Vec<usize>>, bool)> = (0..nlevels)
            .map(|l| (op.m2l_on(l), op.diagonal_classes(l), op.taken_from_waves(l)))
            .collect();
        let pairs_off: Vec<u64> = (0..nlevels)
            .map(|l| sum(fmm_off.plan().level(l).v().len() as u64))
            .collect();
        let converted_off = sum(fmm_off.plan().near_pairs().v_converted as u64);
        let interactions_off = sum(fmm_off.near_interactions().iter().sum());
        let planned = fmm_off.v_crossovers().to_vec();
        // The V pairs between two leaves per level, over every rank: only those can convert.
        let plan = fmm_off.plan();
        let index = plan.index();
        let leaf = |l: usize, i: usize| {
            matches!(index.kind(l, i), KeyType::LocalLeaf | KeyType::GhostLeaf(_))
        };
        let leaf_pairs: Vec<u64> = (0..nlevels)
            .map(|l| {
                let v = plan.level(l).v();
                let local = (0..v.nrows())
                    .filter(|&t| plan.computes(l, t) && leaf(l, t))
                    .map(|t| v.row(t).0.iter().filter(|&&s| leaf(l, s as usize)).count())
                    .sum::<usize>();
                sum(local as u64)
            })
            .collect();
        drop(fmm_off);
        let errors_off = errors(scenario, inputs, &output_off, 0, comm);

        let on = off.clone().v_after_plan(choice);
        let (fmm, output) = run(&on, inputs, kind, comm);
        assert_eq!(fmm.v_after_plan(), choice, "{what}");
        let op = fmm.operator();
        let m2l: Vec<(HelmholtzM2l, Option<Vec<usize>>, bool)> = (0..nlevels)
            .map(|l| (op.m2l_on(l), op.diagonal_classes(l), op.taken_from_waves(l)))
            .collect();
        assert_eq!(m2l, m2l_off, "{what}: the M2L per level");
        assert_eq!(
            fmm.v_crossovers(),
            &planned[..],
            "{what}: the crossovers before the plan"
        );
        let after = fmm
            .v_crossovers_after_plan()
            .unwrap_or_else(|| panic!("{what}: no crossovers after the plan"))
            .to_vec();
        // The levels the setting names: taken from the waves, and under `AllLevels` those that
        // keep them; a level without V pairs between leaves has nothing more to convert.
        let named: Vec<bool> = (0..nlevels)
            .map(|l| {
                let (m2l, _, taken) = &m2l[l];
                let keeps = choice == VAfterPlan::AllLevels && *m2l == HelmholtzM2l::PlaneWave;
                // The scenario raises no level (ε ≥ 10⁻⁶), so `RaisedLevels` names none.
                let taken = *taken && choice != VAfterPlan::RaisedLevels;
                (taken || keeps) && leaf_pairs[l] > 0
            })
            .collect();
        let raised: Vec<bool> = after.iter().zip(&planned).map(|(a, p)| a > p).collect();
        assert!(
            after.iter().zip(&planned).all(|(a, p)| a >= p),
            "{what}: a crossover fell: {planned:?} → {after:?}"
        );
        assert_eq!(
            raised, named,
            "{what}: the raised levels ({planned:?} → {after:?})"
        );
        let replanned = raised.contains(&true);
        assert_eq!(fmm.replanned(), replanned, "{what}: the second plan");
        if setting == "Rotation" {
            assert!(!replanned, "{what}: nothing to raise under Rotation");
        } else if choice == VAfterPlan::RaisedLevels {
            assert!(!replanned, "{what}: no raised level");
        } else {
            raised_any[usize::from(setting == "the waves kept")] |= replanned;
        }
        let pairs: Vec<u64> = (0..nlevels)
            .map(|l| sum(fmm.plan().level(l).v().len() as u64))
            .collect();
        assert!(
            pairs.iter().zip(&pairs_off).all(|(a, b)| a <= b),
            "{what}: V pairs per level {pairs:?} against {pairs_off:?}"
        );
        let converted = sum(fmm.plan().near_pairs().v_converted as u64);
        let interactions = sum(fmm.near_interactions().iter().sum());
        assert!(
            converted >= converted_off && interactions >= interactions_off,
            "{what}: {converted} converted and {interactions} near interactions against \
             {converted_off} and {interactions_off}"
        );
        drop(fmm);
        if !replanned {
            assert_same(
                &format!("{what}: nothing raised"),
                &output,
                &output_off,
                comm,
            );
        }
        let errors = errors(scenario, inputs, &output, 0, comm);
        if comm.rank() == 0 {
            println!(
                "{what}: crossovers {planned:.0?} → {after:.0?}; V pairs per level {pairs_off:?} \
                 → {pairs:?}; converted {converted_off} → {converted}; φ {:.3e} against {:.3e}, \
                 ∇φ {:.3e} against {:.3e}",
                errors.0, errors_off.0, errors.1, errors_off.1
            );
        }
        assert!(
            errors.0 <= NEAR_ERROR_BOUND * errors_off.0
                && errors.1 <= NEAR_ERROR_BOUND * errors_off.1,
            "{what}: errors {errors:?} against the setting off's {errors_off:?}"
        );
        if replanned {
            let (_, threaded) = run(&on.clone().threads(2), inputs, kind, comm);
            assert_same(&format!("{what}, 2 threads"), &threaded, &output, comm);
            check_one_rank(&what, &on, inputs, kind, &output, comm);
        }
    }
    assert!(
        raised_any[0] && raised_any[1],
        "{}: no second plan for the levels taken ({}) or kept ({})",
        scenario.name,
        raised_any[0],
        raised_any[1]
    );
    // The per-box price (CH4.3b2d) without a raised level: nothing counted, one plan, the
    // per-pair price's output bit for bit.
    let what = format!("{}, the V rule's price per box", scenario.name);
    let kept = auto.clone().diagonal_costs(Some(DEARER_FORM));
    let (_, output_pair) = run(&kept, inputs, kind, comm);
    let (fmm, output) = run(
        &kept.v_raised_price(VRaisedPrice::PerBox),
        inputs,
        kind,
        comm,
    );
    assert!(
        fmm.raised_v_shapes().is_none() && !fmm.replanned(),
        "{what}: counted or planned again"
    );
    drop(fmm);
    assert_same(&what, &output, &output_pair, comm);
}

/// The scenario of [`check_v_raised`]: 3,000 points on the sphere at kD = 30 under the leaf
/// rule at ε = 10⁻⁷ (level 3's leaves hold about 18 points, so that NEON's per-pair crossover
/// keeps some of their V pairs),
/// whose levels with waves run at p_w = 28 above the degrees given, with V pairs between
/// leaves.
fn raised_v_scenario() -> Scenario {
    Scenario {
        name: "sphere, kD = 30, ε = 1e-7, complex charges, gradients",
        n: (3000, None),
        degrees: Degrees::Tolerance(1e-7),
        bound: ERROR_FACTOR * 1e-7,
        ..sphere_scenario()
    }
}

/// The V rule's price on raised levels (Helmholtz CH4.3b2d, `VRaisedPrice`) on `scenario`, under
/// Auto with waves from degree 0, the empty-box rule and NEON's V and time-rule costs named (so
/// that the plans are the same on every machine class):
/// - the counts before the plan against the plan: the computed local leaves with a V row and
///   their entries on each raised level of `PerPair`'s plan (what the per-pair crossover
///   leaves), summed over the ranks;
/// - `PerBox` against `PerPair`: the M2L per level unchanged; the crossovers the higher of the
///   per-pair one and `raised_crossover` of the counts exactly on the raised levels with entries,
///   unchanged elsewhere, raised on some level; V pairs per level at most and conversions at
///   least `PerPair`'s; one plan; the errors within `NEAR_ERROR_BOUND`; 2 threads and on P
///   ranks the one-rank FMM bit for bit;
/// - under Rotation (no waves) `PerBox` counts nothing and is `PerPair`'s output bit for bit;
/// - `VAfterPlan::RaisedLevels` (CH4.3b2d's trusted measure) under `PerPair`: raised exactly the
///   raised levels with waves and V pairs between leaves, a second plan; 2 threads and the
///   one-rank FMM bit for bit.
fn check_v_raised(scenario: &Scenario, comm: &SimpleCommunicator) {
    let inputs = &Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let sum = |x: u64| {
        let mut total = 0u64;
        comm.all_reduce_into(&x, &mut total, SystemOperation::sum());
        total
    };
    // Leaves of up to 128 points, so that the per-pair crossover leaves leaf pairs to convert.
    let base = default_builder(scenario)
        .max_points_per_leaf(128)
        .near_v(VNear::CostModel)
        .v_costs(Some(V_COSTS_NEON))
        .diagonal_costs(Some(DIAGONAL_COSTS_NEON))
        .m2l_plane_wave_min_degree(0);
    // The degrees given: Rotation's (no waves, nothing raised).
    let (rotation, rotation_output) = run(
        &base.clone().m2l(HelmholtzM2l::Rotation),
        inputs,
        kind,
        comm,
    );
    let nlevels = rotation.nlevels();
    let given: Vec<usize> = (0..nlevels)
        .map(|l| rotation.operator().degree(l))
        .collect();
    drop(rotation);
    let (rotation_box, output) = run(
        &base
            .clone()
            .m2l(HelmholtzM2l::Rotation)
            .v_raised_price(VRaisedPrice::PerBox),
        inputs,
        kind,
        comm,
    );
    assert!(
        rotation_box.raised_v_shapes().is_none(),
        "{}: Rotation counted",
        scenario.name
    );
    drop(rotation_box);
    assert_same(
        &format!("{}: Rotation, PerBox", scenario.name),
        &output,
        &rotation_output,
        comm,
    );

    let what = format!("{}, the V rule's price per box", scenario.name);
    let (fmm_pair, output_pair) = run(&base, inputs, kind, comm);
    assert_eq!(fmm_pair.v_raised_price(), VRaisedPrice::PerPair, "{what}");
    assert!(fmm_pair.raised_v_shapes().is_none(), "{what}");
    let op = fmm_pair.operator();
    let m2l_pair: Vec<(HelmholtzM2l, usize)> =
        (0..nlevels).map(|l| (op.m2l_on(l), op.degree(l))).collect();
    let raised: Vec<bool> = (0..nlevels)
        .map(|l| m2l_pair[l].0 == HelmholtzM2l::PlaneWave && m2l_pair[l].1 > given[l])
        .collect();
    assert!(raised.contains(&true), "{what}: no raised level");
    let pairs_pair: Vec<u64> = (0..nlevels)
        .map(|l| sum(fmm_pair.plan().level(l).v().len() as u64))
        .collect();
    let converted_pair = sum(fmm_pair.plan().near_pairs().v_converted as u64);
    let crossovers_pair = fmm_pair.v_crossovers().to_vec();
    // The counts against the per-pair plan's own V rows: per level its computed local leaves
    // with a V entry left, and those entries.
    let plan = fmm_pair.plan();
    let index = plan.index();
    let counted: Vec<LeafVShape> = (0..nlevels)
        .map(|l| {
            let v = plan.level(l).v();
            let rows: Vec<u64> = (0..v.nrows())
                .filter(|&t| plan.computes(l, t) && index.kind(l, t) == KeyType::LocalLeaf)
                .map(|t| v.row(t).0.len() as u64)
                .filter(|&n| n > 0)
                .collect();
            LeafVShape {
                leaves: sum(rows.len() as u64),
                pairs: sum(rows.iter().sum()),
            }
        })
        .collect();
    drop(fmm_pair);
    let errors_pair = errors(scenario, inputs, &output_pair, 0, comm);

    let on = base.clone().v_raised_price(VRaisedPrice::PerBox);
    let (fmm, output) = run(&on, inputs, kind, comm);
    assert_eq!(fmm.v_raised_price(), VRaisedPrice::PerBox, "{what}");
    assert!(!fmm.replanned(), "{what}: a second plan");
    let op = fmm.operator();
    let m2l: Vec<(HelmholtzM2l, usize)> =
        (0..nlevels).map(|l| (op.m2l_on(l), op.degree(l))).collect();
    assert_eq!(m2l, m2l_pair, "{what}: the M2L and degree per level");
    let shapes = fmm
        .raised_v_shapes()
        .unwrap_or_else(|| panic!("{what}: nothing counted"))
        .to_vec();
    for l in 0..nlevels {
        let want = if raised[l] {
            counted[l]
        } else {
            LeafVShape::default()
        };
        assert_eq!(
            shapes[l], want,
            "{what}: level {l}'s counts against the plan"
        );
    }
    let crossovers = fmm.v_crossovers().to_vec();
    for l in 0..nlevels {
        let want = if raised[l] && shapes[l].pairs > 0 {
            crossovers_pair[l].max(raised_crossover(
                &V_COSTS_NEON,
                &DIAGONAL_COSTS_NEON,
                m2l[l].1,
                shapes[l],
                scenario.gradients,
            ))
        } else {
            crossovers_pair[l]
        };
        assert!(
            (crossovers[l] - want).abs() <= 1e-12 * want.max(1.0),
            "{what}: level {l}'s crossover {} against {want}",
            crossovers[l]
        );
    }
    assert!(
        crossovers.iter().zip(&crossovers_pair).any(|(a, b)| a > b),
        "{what}: no crossover rose ({crossovers_pair:?} → {crossovers:?})"
    );
    let pairs: Vec<u64> = (0..nlevels)
        .map(|l| sum(fmm.plan().level(l).v().len() as u64))
        .collect();
    assert!(
        pairs.iter().zip(&pairs_pair).all(|(a, b)| a <= b),
        "{what}: V pairs per level {pairs:?} against {pairs_pair:?}"
    );
    let converted = sum(fmm.plan().near_pairs().v_converted as u64);
    assert!(
        converted > converted_pair,
        "{what}: {converted} converted against {converted_pair}; counts {shapes:?}; \
         crossovers {crossovers_pair:.0?} → {crossovers:.0?}; V pairs {pairs_pair:?} → {pairs:?}; \
         raised {raised:?}"
    );
    drop(fmm);
    let errors = errors(scenario, inputs, &output, 0, comm);
    if comm.rank() == 0 {
        println!(
            "{what}: counts {shapes:?}; crossovers {crossovers_pair:.0?} → {crossovers:.0?}; V \
             pairs per level {pairs_pair:?} → {pairs:?}; converted {converted_pair} → \
             {converted}; φ {:.3e} against {:.3e}, ∇φ {:.3e} against {:.3e}",
            errors.0, errors_pair.0, errors.1, errors_pair.1
        );
    }
    assert!(
        errors.0 <= NEAR_ERROR_BOUND * errors_pair.0
            && errors.1 <= NEAR_ERROR_BOUND * errors_pair.1,
        "{what}: errors {errors:?} against PerPair's {errors_pair:?}"
    );
    // The counts on the pool (chunks of leaves) the serial counts.
    let (fmm, threaded) = run(&on.clone().threads(2), inputs, kind, comm);
    assert_eq!(
        fmm.raised_v_shapes(),
        Some(&shapes[..]),
        "{what}, 2 threads: the counts"
    );
    drop(fmm);
    assert_same(&format!("{what}, 2 threads"), &threaded, &output, comm);
    check_one_rank(&what, &on, inputs, kind, &output, comm);

    // The trusted measure: the second plan on the raised levels that keep their waves.
    let what = format!(
        "{}, the V rule after the plan on raised levels",
        scenario.name
    );
    let trusted = base.clone().v_after_plan(VAfterPlan::RaisedLevels);
    let (fmm, output) = run(&trusted, inputs, kind, comm);
    let after = fmm
        .v_crossovers_after_plan()
        .unwrap_or_else(|| panic!("{what}: no crossovers after the plan"))
        .to_vec();
    let leaf_pairs_named: Vec<bool> = (0..nlevels)
        .map(|l| raised[l] && counted[l].pairs > 0)
        .collect();
    let rose: Vec<bool> = after
        .iter()
        .zip(&crossovers_pair)
        .map(|(a, p)| a > p)
        .collect();
    // A raised level rises where its first plan kept a V pair between leaves: on this tree
    // every raised level with entries left.
    assert_eq!(
        rose, leaf_pairs_named,
        "{what}: {crossovers_pair:?} → {after:?}"
    );
    assert!(fmm.replanned(), "{what}: no second plan");
    // The per-box price before the plan counts what the per-pair plan leaves, so where a
    // raised level's boxes with a V row are leaves (on this tree), it is the trusted measure's
    // crossover after the plan.
    for l in (0..nlevels).filter(|&l| leaf_pairs_named[l]) {
        assert!(
            (after[l] - crossovers[l]).abs() <= 1e-9 * after[l],
            "{what}: level {l}: after the plan {} against {} per box before it",
            after[l],
            crossovers[l]
        );
    }
    drop(fmm);
    if comm.rank() == 0 {
        println!("{what}: crossovers {crossovers_pair:.0?} → {after:.0?}");
    }
    let (_, threaded) = run(&trusted.clone().threads(2), inputs, kind, comm);
    assert_same(&format!("{what}, 2 threads"), &threaded, &output, comm);
    check_one_rank(&what, &trusted, inputs, kind, &output, comm);
}

/// The scenario of [`check_time_rule_raised`] whose raised level passes classes at p_w: the
/// cube at kD = 40 + 2i, ε = 10⁻⁷ (level 2 at k a 8.7, p_l 20 raised to 28).
fn time_rule_scenario() -> Scenario {
    Scenario {
        name: "cube, kD = 40 + 2i, ε = 1e-7, complex charges",
        layout: Layout::Cube,
        n: (600, None),
        k: (40.0, 2.0),
        degrees: Degrees::Tolerance(1e-7),
        bound: ERROR_FACTOR * 1e-7,
        charges: ChargeKind::Complex,
        gradients: false,
        debug: true,
        empty_rank0: false,
    }
}

/// The named `PlaneWaveRaise` forms (CH6.1h) on a scenario of [`check_resolving_degree`],
/// against its default output (`rule`, `PlaneWaveRaise::Rule`): each form's stores and M2L
/// at the raised degrees, its translations at the degrees given (`Level`: raised); the form
/// the rule resolves to the default's output bit for bit; each other form within
/// `PLANE_WAVE_RATIO` of Rotation's errors (`M2l` at 10⁻⁸, outside the rule's band, printed
/// only: it erred 1.53× on one of the report's cases), 2 threads and on P ranks the one-rank
/// FMM bit for bit; `Leaves` `Level`'s output bit for bit where no level below level 2 is
/// raised (the translations between raised levels and level 1 or 0 carry nothing that
/// reaches the output), else not; `M2l` never `Level`'s (the raised levels' leaves at the
/// degrees given).
fn check_m2l_raise(
    scenario: &Scenario,
    tables: &HelmholtzFmmBuilder,
    (inputs, given, raised, rule): (&Inputs, &[usize], &[usize], &Output<f64>),
    rotation: (f64, f64),
    comm: &SimpleCommunicator,
) {
    let Degrees::Tolerance(epsilon) = scenario.degrees else {
        panic!("{}: the rule reads a tolerance", scenario.name);
    };
    let resolved = PlaneWaveRaise::Rule.resolved(Some(epsilon));
    let mut outputs = Vec::new();
    for raise in [
        PlaneWaveRaise::Level,
        PlaneWaveRaise::Leaves,
        PlaneWaveRaise::M2l,
    ] {
        let name = format!("{}: PlaneWave raised by {raise:?}", scenario.name);
        let kind = scenario.charges;
        let named = tables.clone().plane_wave_raise(raise);
        let (fmm, output) = run(&named, inputs, kind, comm);
        let op = fmm.operator();
        assert_eq!(op.plane_wave_raise(), raise, "{name}");
        assert_eq!(fmm.degrees(), raised, "{name}: the degrees");
        let translations: Vec<usize> = (0..given.len()).map(|l| op.translation_degree(l)).collect();
        let want = if raise == PlaneWaveRaise::Level {
            raised
        } else {
            given
        };
        assert_eq!(translations, want, "{name}: the translation degrees");
        drop(fmm);
        if raise == resolved {
            assert_same(&format!("{name}, the rule's"), &output, rule, comm);
        } else {
            let within = check_raise_errors(&name, scenario, inputs, &output, rotation, comm);
            if !(raise == PlaneWaveRaise::M2l && epsilon < 1e-7) {
                within.unwrap_or_else(|message| panic!("{message}"));
            }
            let (_, threaded) = run(&named.clone().threads(2), inputs, kind, comm);
            assert_same(&format!("{name}, 2 threads"), &threaded, &output, comm);
            check_one_rank(&name, &named, inputs, kind, &output, comm);
        }
        outputs.push(output);
    }
    let [level, leaves, m2l] = &outputs[..] else {
        unreachable!("three forms")
    };
    let name = scenario.name;
    if (3..given.len()).any(|l| given[l] < raised[l]) {
        let same = everywhere(bits(leaves) == bits(level), comm);
        assert!(!same, "{name}: Leaves' output not Level's");
    } else {
        assert_same(
            &format!("{name}, Leaves' output Level's"),
            leaves,
            level,
            comm,
        );
    }
    let same = everywhere(bits(m2l) == bits(level), comm);
    assert!(!same, "{name}: M2l's output not Level's");
}

/// A form's errors against the direct sum over Rotation's, printed on rank 0; `Err` with the
/// message if either is above `PLANE_WAVE_RATIO` (the same on every rank: the errors are
/// reduced over the ranks).
fn check_raise_errors(
    name: &str,
    scenario: &Scenario,
    inputs: &Inputs,
    output: &Output<f64>,
    (rotation_phi, rotation_grad): (f64, f64),
    comm: &SimpleCommunicator,
) -> Result<(), String> {
    let (error_phi, error_grad) = errors(scenario, inputs, output, 0, comm);
    let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
    if comm.rank() == 0 {
        println!(
            "{name} against the direct sum: φ {error_phi:.2e}, ∇φ {error_grad:.2e}, {:.3}× and \
             {:.3}× Rotation's",
            ratio(error_phi, rotation_phi),
            ratio(error_grad, rotation_grad),
        );
    }
    if error_phi <= PLANE_WAVE_RATIO * rotation_phi
        && error_grad <= PLANE_WAVE_RATIO * rotation_grad
    {
        Ok(())
    } else {
        Err(format!(
            "{name}: φ {error_phi:.2e}, ∇φ {error_grad:.2e} against Rotation's \
             {rotation_phi:.2e}, {rotation_grad:.2e}"
        ))
    }
}

/// Merge and shift split by child and by parents (CH6.2f): each named the default's output
/// bit for bit, by parents at 2 threads (at 1 it is the default's path), by child at 1 and 2
/// threads and on several ranks against the one-rank FMM, each chunk split as named; the
/// default (`counts` at 1 and at 2 threads) never by child at one thread and by child on some
/// rank at 2 (level 2 has at most 64 boxes, within 64 per thread).
fn check_plane_wave_split(
    name: &str,
    tables: &HelmholtzFmmBuilder,
    (inputs, kind, output): (&Inputs, ChargeKind, &Output<f64>),
    (one, two): (PlaneWaveCounts, PlaneWaveCounts),
    comm: &SimpleCommunicator,
) {
    let mut by_child = 0usize;
    comm.all_reduce_into(&two.by_child, &mut by_child, SystemOperation::sum());
    assert!(
        everywhere(one.by_child == 0, comm) && by_child > 0,
        "{name}: the default split by child at 2 threads only ({one:?}, {two:?})"
    );
    for (split, threads) in [
        (PlaneWaveSplit::Parents, 2),
        (PlaneWaveSplit::Children, 1),
        (PlaneWaveSplit::Children, 2),
    ] {
        let named = tables.clone().plane_wave_split(split).threads(threads);
        let (fmm, got) = run(&named, inputs, kind, comm);
        let op = fmm.operator();
        let counts = op.plane_wave_counts();
        let want = match split {
            PlaneWaveSplit::Children => counts.chunks,
            _ => 0,
        };
        assert!(
            everywhere(
                op.plane_wave_split() == split && counts.by_child == want,
                comm
            ),
            "{name}: {split:?} at {threads} threads split {counts:?}"
        );
        drop(fmm);
        assert_same(
            &format!("{name}, {split:?}, {threads} threads"),
            &got,
            output,
            comm,
        );
    }
    let children = tables.clone().plane_wave_split(PlaneWaveSplit::Children);
    check_one_rank(
        &format!("{name}, {:?}", PlaneWaveSplit::Children),
        &children,
        inputs,
        kind,
        output,
        comm,
    );
}

/// The waves buffer bounded (CH6.2a): at the floor (a budget of 0: `MIN_BUFFER_BOXES` boxes
/// of a level), at twice the floor's buffer and (in release builds) unbounded, the output
/// of the default budget, `output` (a whole level here), bit for bit; the floor at 2 threads
/// and, on several ranks, against the one-rank FMM (which chunks the union's levels). The
/// greedy cut takes the fewest chunks of each capacity, so a larger bound never takes more.
/// `whole` is the default build's `plane_wave_bytes`, buffer bytes and counts. Returns
/// whether the floor took more chunks than the default on this rank.
fn check_plane_wave_buffer(
    name: &str,
    tables: &HelmholtzFmmBuilder,
    (inputs, kind, output): (&Inputs, ChargeKind, &Output<f64>),
    (bytes, whole_buffer, whole_counts): (usize, usize, PlaneWaveCounts),
    ranks_per_node: usize,
    comm: &SimpleCommunicator,
) -> bool {
    let rank = comm.rank();
    let floor = tables.clone().plane_wave_budget(0);
    let (fmm, bounded) = run(&floor, inputs, kind, comm);
    let op = fmm.operator();
    let floor_counts = op.plane_wave_counts();
    let floor_buffer = op.plane_wave_buffer_bytes();
    // The build never held more than the bound's buffer, not even before the bound was set
    // (Helmholtz CH4.0c: the whole level's, 218 GB on W3 at kD = 500).
    let ok = op.plane_wave_buffer() == Some(0)
        && op.plane_wave_buffer_peak_bytes() == floor_buffer
        && floor_buffer <= whole_buffer
        && op.plane_wave_bytes() <= bytes
        && floor_counts.level_calls == whole_counts.level_calls
        && floor_counts.chunks >= whole_counts.chunks
        && floor_counts.waves >= whole_counts.waves
        // A level call with more sources than the floor holds runs in several chunks.
        && (whole_counts.waves <= 3 * MIN_BUFFER_BOXES * whole_counts.level_calls
            || floor_counts.chunks > whole_counts.chunks);
    assert!(
        everywhere(ok, comm),
        "{name}: the floor's buffer and counts ({floor_counts:?} against {whole_counts:?}; \
         the buffer {floor_buffer} bytes, its peak {})",
        op.plane_wave_buffer_peak_bytes()
    );
    let floor_chunked = floor_counts.chunks > whole_counts.chunks;
    drop(fmm);
    assert_same(&format!("{name}, the floor bound"), &bounded, output, comm);
    let twice = (2 * floor_buffer).saturating_mul(ranks_per_node);
    let (fmm, bounded) = run(&tables.clone().plane_wave_budget(twice), inputs, kind, comm);
    let middle_counts = fmm.operator().plane_wave_counts();
    drop(fmm);
    assert!(
        everywhere(
            (whole_counts.chunks..=floor_counts.chunks).contains(&middle_counts.chunks),
            comm
        ),
        "{name}: twice the floor's chunks {middle_counts:?}"
    );
    if rank == 0 {
        println!(
            "{name}: the whole level's buffer {:.2} MB, {} chunks, {} waves; at the floor \
             {} chunks, {} waves; at twice the floor {} chunks, {} waves (rank 0)",
            whole_buffer as f64 / 1e6,
            whole_counts.chunks,
            whole_counts.waves,
            floor_counts.chunks,
            floor_counts.waves,
            middle_counts.chunks,
            middle_counts.waves,
        );
    }
    assert_same(&format!("{name}, twice the floor"), &bounded, output, comm);
    // Unbounded in release builds only (the default budget holds a whole level here).
    if !cfg!(debug_assertions) {
        let (fmm, bounded) = run(
            &tables.clone().plane_wave_budget(usize::MAX),
            inputs,
            kind,
            comm,
        );
        let ok = fmm.operator().plane_wave_buffer().is_none()
            && fmm.operator().plane_wave_counts() == whole_counts;
        assert!(everywhere(ok, comm), "{name}: unbounded");
        drop(fmm);
        assert_same(&format!("{name}, unbounded"), &bounded, output, comm);
    }
    // The floor at 2 threads, split by child and by parents (CH6.2f).
    for (split, label) in [
        (PlaneWaveSplit::Children, "by child"),
        (PlaneWaveSplit::Parents, "by parents"),
    ] {
        let (_, bounded) = run(
            &floor.clone().plane_wave_split(split).threads(2),
            inputs,
            kind,
            comm,
        );
        assert_same(
            &format!("{name}, the floor, 2 threads, {label}"),
            &bounded,
            output,
            comm,
        );
    }
    check_one_rank(
        &format!("{name}, the floor"),
        &floor,
        inputs,
        kind,
        output,
        comm,
    );
    floor_chunked
}

/// The Rotation M2L by pairs (Helmholtz CH4.0i): every split, thread count and buffer the
/// output of target by target at one thread, bit for bit, with the level calls by pairs
/// counted; two vectors; the library's M2L; on P ranks the one-rank FMM at the floor.
fn check_rcr_split(scenario: &Scenario, comm: &SimpleCommunicator) {
    let name = scenario.name;
    let kind = scenario.charges;
    // The sum over the ranks of the level calls by pairs, per level.
    let calls = |fmm: &HelmholtzFmm<'_, f64>| -> Vec<usize> {
        let mut local = fmm.operator().rcr_counts().level_calls.clone();
        local.resize(fmm.nlevels(), 0);
        let mut summed = vec![0usize; local.len()];
        comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
        summed
    };
    for (m, base) in [(1, builder(scenario)), (2, builder(scenario).vectors(2))] {
        let inputs = Inputs::new(scenario, m, comm);
        let boxes = base.clone().rcr_split(RcrSplit::Boxes);
        let (fmm, want) = run(&boxes, &inputs, kind, comm);
        assert!(
            calls(&fmm).iter().all(|&c| c == 0),
            "{name}: by boxes, a level call by pairs"
        );
        drop(fmm);
        // The pool with the default buffer (a whole level a chunk) and at the floor (one
        // pair per thread, so rows span chunks), serially, and Auto.
        let runs = [
            (RcrSplit::Pairs, 1, None),
            (RcrSplit::Pairs, 2, None),
            (RcrSplit::Pairs, 3, Some(0)),
            (RcrSplit::Auto, 20, None),
        ];
        for (split, threads, budget) in runs {
            let mut named = base.clone().rcr_split(split).threads(threads);
            if let Some(bytes) = budget {
                named = named.rcr_budget(bytes);
            }
            let what =
                format!("{name}, m = {m}: {split:?} at {threads} threads, budget {budget:?}");
            let (fmm, got) = run(&named, &inputs, kind, comm);
            let by_level = calls(&fmm);
            let counts = fmm.operator().rcr_counts().clone();
            let calls_here: usize = counts.level_calls.iter().sum();
            assert!(
                everywhere(fmm.operator().rcr_split() == split, comm),
                "{what}: the split as built"
            );
            match split {
                // Every Rotation level with pairs: level 2 on some rank at least.
                RcrSplit::Pairs => assert!(by_level[2] > 0, "{what}: level 2 not by pairs"),
                // Auto's rule reads the degree: by pairs only on levels of degree at least
                // RCR_PAIR_FEW_DEGREE (none here, p ≤ 15), whatever the threads.
                _ => {
                    let degrees = fmm.degrees();
                    assert!(
                        (0..by_level.len())
                            .all(|l| by_level[l] == 0 || degrees[l] >= RCR_PAIR_FEW_DEGREE),
                        "{what}: by pairs below the rule's degree ({by_level:?}, {degrees:?})"
                    );
                }
            }
            // At the floor (one pair per thread) a level call takes a chunk per `threads`
            // pairs, so more chunks than calls wherever a call has more pairs.
            if budget == Some(0) && split == RcrSplit::Pairs {
                assert!(
                    everywhere(calls_here == 0 || counts.chunks > calls_here, comm),
                    "{what}: the floor did not chunk ({counts:?})"
                );
            }
            drop(fmm);
            assert_same(&what, &got, &want, comm);
        }
        let floor = base.clone().rcr_split(RcrSplit::Pairs).rcr_budget(0);
        check_one_rank(
            &format!("{name}, m = {m}: by pairs at the floor"),
            &floor,
            &inputs,
            kind,
            &want,
            comm,
        );
    }
    // The library's M2L (Auto: waves or Batched on some levels, Rotation on the others).
    let inputs = Inputs::new(scenario, 1, comm);
    let auto = default_builder(scenario);
    let (_, want) = run(
        &auto.clone().rcr_split(RcrSplit::Boxes),
        &inputs,
        kind,
        comm,
    );
    let (fmm, got) = run(
        &auto.rcr_split(RcrSplit::Pairs).rcr_budget(0).threads(3),
        &inputs,
        kind,
        comm,
    );
    let rotation = (2..fmm.nlevels())
        .filter(|&l| fmm.operator().m2l_on(l) == HelmholtzM2l::Rotation)
        .count();
    let by_level = calls(&fmm);
    let pairs_calls: usize = by_level.iter().sum();
    assert!(
        rotation == 0 || pairs_calls > 0,
        "{name}: Auto's M2L with Rotation levels, none by pairs"
    );
    let ok = (2..fmm.nlevels()).all(|l| {
        fmm.operator().m2l_on(l) == HelmholtzM2l::Rotation
            || fmm
                .operator()
                .rcr_counts()
                .level_calls
                .get(l)
                .copied()
                .unwrap_or(0)
                == 0
    });
    assert!(
        everywhere(ok, comm),
        "{name}: by pairs on a level without Rotation"
    );
    if comm.rank() == 0 {
        println!(
            "{name}: by pairs bit for bit by boxes; under Auto {rotation} Rotation levels, \
             level calls by pairs per level {by_level:?}"
        );
    }
    drop(fmm);
    assert_same(&format!("{name}: Auto's M2L by pairs"), &got, &want, comm);
}

/// The diagonal M2L (Helmholtz CH4.2b1) under Rotation and under the library's Auto, against
/// the setting without it: some level diagonal, and (summed over the scenarios, returned)
/// whether a level mixed classes on both forms; the errors against the direct sum within
/// [`DIAGONAL_RATIO`] of the setting's without it; 2 threads (and 4 in release) bit for bit
/// one; chunks of 1, 2, 3, 5 and 7 targets (1 and 3 in debug builds) and 3 at 2 threads bit
/// for bit the default; on P ranks the one-rank
/// FMM bit for bit; two vectors each the one-vector FMM; the budget's floor (every level RCR)
/// Rotation's bits; a second evaluation the same bits, the signature buffer not grown. Returns the levels that mixed the two forms on some rank.
fn check_diagonal(scenario: &Scenario, comm: &SimpleCommunicator) -> usize {
    let name = scenario.name;
    let kind = scenario.charges;
    let inputs = Inputs::new(scenario, 1, comm);
    let sum = |local: &[usize], nlevels: usize| -> Vec<usize> {
        let mut local = local.to_vec();
        local.resize(nlevels, 0);
        let mut summed = vec![0usize; nlevels];
        comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
        summed
    };
    let mut mixed = 0;
    let mut taken = 0;
    for (setting, base) in [
        ("Rotation", builder(scenario)),
        ("Auto", default_builder(scenario).near_v(VNear::Off)),
    ] {
        let (fmm, plain) = run(&base, &inputs, kind, comm);
        let plain_errors = errors(scenario, &inputs, &plain, 0, comm);
        drop(fmm);
        // The direct build named: the trusted path (the library's default build, interpolated
        // since Helmholtz CH4.2c2's sign-off, is `check_interpolated`'s).
        let ruled = base.clone().diagonal(HelmholtzDiagonal::Rule);
        let diagonal = ruled.clone().diagonal_build(TransferBuild::Direct);
        let (mut fmm, want) = run(&diagonal, &inputs, kind, comm);
        let what = format!("{name}, {setting} with the diagonal M2L");
        // A second evaluation: the same bits, and the buffer sized by the build holds every
        // level (no growth between the levels' layouts).
        let peak = fmm.operator().diagonal_buffer_peak_bytes();
        let again = fmm
            .evaluate(&inputs.charges(kind))
            .unwrap_or_else(|error| panic!("rank {}: the evaluation: {error}", comm.rank()));
        assert_same(&format!("{what}: evaluated again"), &again, &want, comm);
        assert!(
            everywhere(fmm.operator().diagonal_buffer_peak_bytes() == peak, comm),
            "{what}: the buffer grew in an evaluation"
        );
        check_diagonal_times(&what, &fmm, 1, 1, comm);
        let op = fmm.operator();
        let nlevels = fmm.nlevels();
        let direct_classes: Vec<Vec<usize>> = (0..nlevels)
            .map(|l| op.diagonal_classes(l).unwrap_or_default())
            .collect();
        let counts = op.diagonal_counts().clone();
        let (diagonal_pairs, rcr_pairs) = (
            sum(&counts.diagonal_pairs, nlevels),
            sum(&counts.rcr_pairs, nlevels),
        );
        let levels: Vec<(usize, usize)> = (2..nlevels)
            .filter_map(|l| op.diagonal_classes(l).map(|c| (l, c.len())))
            .collect();
        // The classes are the same on every rank (constraint 9).
        let mut flat: Vec<u64> = (0..nlevels)
            .flat_map(|l| {
                let mut row = [0u64; 34];
                for c in op.diagonal_classes(l).unwrap_or_default() {
                    row[c] = 1;
                }
                row
            })
            .collect();
        let mut agreed = flat.clone();
        comm.all_reduce_into(&flat[..], &mut agreed[..], SystemOperation::max());
        let same = agreed == flat;
        flat.clear();
        assert!(
            everywhere(same, comm),
            "{what}: the classes differ between ranks"
        );
        // The time rule (Helmholtz CH4.3b1).
        taken += check_diagonal_auto(&what, scenario, &base, &inputs, plain_errors, comm);
        // Under Auto a level with waves or batched tables takes no diagonal form; with no RCR
        // level the setting changes nothing.
        if levels.is_empty() {
            assert!(setting == "Auto", "{what}: no level diagonal");
            drop(fmm);
            assert_same(&format!("{what}: no RCR level"), &want, &plain, comm);
            if comm.rank() == 0 {
                println!("{what}: no RCR level, the output unchanged");
            }
            continue;
        }
        for &(l, _) in &levels {
            assert!(
                diagonal_pairs[l] > 0,
                "{what}: level {l} took no diagonal pair"
            );
            assert_eq!(op.m2l_on(l), HelmholtzM2l::Rotation, "{what}: level {l}");
        }
        mixed += levels
            .iter()
            .filter(|&&(l, classes)| classes < 34 && rcr_pairs[l] > 0)
            .count();
        let diagonal_errors = errors(scenario, &inputs, &want, 0, comm);
        if comm.rank() == 0 {
            println!(
                "{what}: levels (level, classes) {levels:?}, diagonal pairs {diagonal_pairs:?}, \
                 RCR pairs {rcr_pairs:?}; φ {:.3e} against {:.3e} ({:.3}×), ∇φ {:.3e} against \
                 {:.3e}",
                diagonal_errors.0,
                plain_errors.0,
                diagonal_errors.0 / plain_errors.0,
                diagonal_errors.1,
                plain_errors.1,
            );
        }
        assert!(
            diagonal_errors.0 <= DIAGONAL_RATIO * plain_errors.0
                && diagonal_errors.1 <= DIAGONAL_RATIO * plain_errors.1,
            "{what}: errors {diagonal_errors:?} against {plain_errors:?} without it"
        );
        drop(fmm);
        // Threads and chunks.
        let mut runs: Vec<(usize, Option<usize>)> = vec![(2, None)];
        if !cfg!(debug_assertions) {
            runs.push((4, None));
        }
        // Every width in release builds; 1 and 3 in debug builds (each run there takes up to
        // 5 s on the cube on the M6).
        let widths: &[usize] = if cfg!(debug_assertions) {
            &[1, 3]
        } else {
            &[1, 2, 3, 5, 7]
        };
        for &chunk in widths {
            runs.push((1, Some(chunk)));
        }
        runs.push((2, Some(3)));
        for (threads, chunk) in runs {
            let named = diagonal.clone().threads(threads).diagonal_chunk(chunk);
            let (fmm, got) = run(&named, &inputs, kind, comm);
            check_diagonal_times(&what, &fmm, threads, 1, comm);
            drop(fmm);
            assert_same(
                &format!("{what} at {threads} threads, chunks of {chunk:?}"),
                &got,
                &want,
                comm,
            );
        }
        check_one_rank(&what, &diagonal, &inputs, kind, &want, comm);
        if setting == "Rotation" {
            check_interpolated(
                &what,
                scenario,
                &ruled,
                &inputs,
                &direct_classes,
                plain_errors,
                comm,
            );
            // The budget's floor: every level falls back to RCR, Rotation's bits.
            let (fmm, floor) = run(&diagonal.clone().diagonal_budget(0), &inputs, kind, comm);
            assert!(
                (0..fmm.nlevels()).all(|l| fmm.operator().diagonal_classes(l).is_none()),
                "{what}: a level diagonal at the budget's floor"
            );
            drop(fmm);
            assert_same(
                &format!("{what} at the budget's floor"),
                &floor,
                &plain,
                comm,
            );
            // Two vectors, each the one-vector FMM.
            let two = Inputs::new(scenario, 2, comm);
            let (fmm, both) = run(&diagonal.clone().vectors(2), &two, kind, comm);
            check_diagonal_times(&format!("{what}, two vectors"), &fmm, 1, 2, comm);
            drop(fmm);
            for v in 0..2 {
                let one = Inputs {
                    sources: two.sources.clone(),
                    targets: two.targets.clone(),
                    charges: vec![two.charges[v].clone()],
                };
                let (_, single) = run(&diagonal, &one, kind, comm);
                let column = |output: &Output<f64>, v: usize| {
                    let phi = output.potential_of(v);
                    let mut bits: Vec<u64> = phi
                        .re
                        .iter()
                        .chain(phi.im.unwrap_or(&[]))
                        .map(|x| x.to_bits())
                        .collect();
                    if let Some(g) = output.gradient_of(v) {
                        let im = g.im.unwrap_or(&[]);
                        bits.extend(g.re.iter().chain(im).flatten().map(|x| x.to_bits()));
                    }
                    bits
                };
                assert!(
                    everywhere(column(&both, v) == column(&single, 0), comm),
                    "{what}: vector {v} of two is not the one-vector FMM bit for bit"
                );
            }
        }
    }
    assert!(
        everywhere(taken > 0, comm),
        "{name}: the time rule took no level at free costs"
    );
    let mut mixed_anywhere = 0;
    comm.all_reduce_into(&mixed, &mut mixed_anywhere, SystemOperation::max());
    mixed_anywhere
}

/// Costs under which the diagonal form beats every other M2L wherever a class passes.
const FREE_DIAGONAL: DiagonalCosts = DiagonalCosts {
    s2f: [0.0, 0.0],
    f2r: [0.0, 0.0],
    pair: [0.0, 0.0],
    rotation: [1.0, 1.0],
    plane_wave: [1e12, 1e12],
};

/// The diagonal form's time rule (Helmholtz CH4.3b1, `HelmholtzDiagonal::Auto`) on `base`'s
/// settings, at the machine's costs and at free ones (where the form takes every level the
/// bound's classes reach, the levels with waves under Auto included): the builder's shapes on
/// every rank the same, their pairs the plan's V pairs (`LevelShape`), the classes the same on
/// every rank, the errors against the direct sum within [`DIAGONAL_RATIO`] of `plain` (the
/// setting's without the form), on P ranks the one-rank FMM bit for bit and its shapes (the
/// decision a function of rank-invariant counts); at free costs also 2 threads and chunk widths
/// bit for bit. Returns the levels taken at free costs.
fn check_diagonal_auto(
    what: &str,
    scenario: &Scenario,
    base: &HelmholtzFmmBuilder,
    inputs: &Inputs,
    plain: (f64, f64),
    comm: &SimpleCommunicator,
) -> usize {
    let kind = scenario.charges;
    let mut taken_free = 0;
    for (costs_name, costs) in [
        ("its machine's costs", None),
        ("free costs", Some(FREE_DIAGONAL)),
    ] {
        let what = format!("{what}, the time rule at {costs_name}");
        let auto = base
            .clone()
            .diagonal(HelmholtzDiagonal::Auto)
            .diagonal_costs(costs);
        let (fmm, want) = run(&auto, inputs, kind, comm);
        let op = fmm.operator();
        let nlevels = fmm.nlevels();
        if let Some(costs) = costs {
            assert_eq!(op.diagonal_costs(), costs, "{what}");
        }
        let shapes = op
            .diagonal_shapes()
            .unwrap_or_else(|| panic!("{what}: the builder gave no shapes"))
            .to_vec();
        let flat: Vec<u64> = shapes
            .iter()
            .flat_map(|s| std::iter::once(s.targets).chain(s.class_pairs))
            .collect();
        let mut agreed = flat.clone();
        comm.all_reduce_into(&flat[..], &mut agreed[..], SystemOperation::max());
        assert!(
            everywhere(agreed == flat, comm),
            "{what}: the shapes differ between ranks"
        );
        if let Some(levels) = fmm.level_shapes() {
            for (l, (shape, level)) in shapes.iter().zip(levels).enumerate() {
                assert_eq!(shape.pairs(), level.pairs, "{what}: level {l}'s V pairs");
                assert!(
                    shape.targets <= level.targets,
                    "{what}: level {l}'s targets"
                );
            }
        }
        let classes: Vec<Vec<usize>> = (0..nlevels)
            .map(|l| op.diagonal_classes(l).unwrap_or_default())
            .collect();
        let flat: Vec<u64> = classes
            .iter()
            .flat_map(|level| {
                let mut row = [0u64; 34];
                for &c in level {
                    row[c] = 1;
                }
                row
            })
            .collect();
        let mut agreed = flat.clone();
        comm.all_reduce_into(&flat[..], &mut agreed[..], SystemOperation::max());
        assert!(
            everywhere(agreed == flat, comm),
            "{what}: the classes differ between ranks"
        );
        let levels: Vec<(usize, usize, HelmholtzM2l)> = (0..nlevels)
            .filter(|&l| !classes[l].is_empty())
            .map(|l| (l, classes[l].len(), op.m2l_on(l)))
            .collect();
        drop(fmm);
        let errors = errors(scenario, inputs, &want, 0, comm);
        if comm.rank() == 0 {
            println!(
                "{what}: (level, classes, M2L) {levels:?}; φ {:.3e} against {:.3e} ({:.3}×), ∇φ \
                 {:.3e} against {:.3e}",
                errors.0,
                plain.0,
                errors.0 / plain.0,
                errors.1,
                plain.1,
            );
        }
        assert!(
            errors.0 <= DIAGONAL_RATIO * plain.0 && errors.1 <= DIAGONAL_RATIO * plain.1,
            "{what}: errors {errors:?} against {plain:?} without the diagonal form"
        );
        // On P ranks the one-rank FMM's bits (the choice a function of rank-invariant counts),
        // and its shapes.
        check_one_rank(&what, &auto, inputs, kind, &want, comm);
        if comm.size() > 1 {
            let one = SimpleCommunicator::self_comm();
            let union_inputs = Inputs {
                sources: union_points(&inputs.sources, comm).0,
                targets: union_points(&inputs.targets, comm).0,
                charges: inputs
                    .charges
                    .iter()
                    .map(|(re, im)| (union(re, comm).0, union(im, comm).0))
                    .collect(),
            };
            let (fmm, _) = run(&auto.clone().threads(1), &union_inputs, kind, &one);
            let same = fmm.operator().diagonal_shapes() == Some(&shapes[..]);
            drop(fmm);
            assert!(
                everywhere(same, comm),
                "{what}: the shapes differ from the one-rank FMM's"
            );
        }
        if costs.is_some() {
            taken_free = levels.len();
            for (threads, chunk) in [(2, None), (1, Some(3)), (2, Some(1))] {
                let named = auto.clone().threads(threads).diagonal_chunk(chunk);
                let (fmm, got) = run(&named, inputs, kind, comm);
                drop(fmm);
                assert_same(
                    &format!("{what} at {threads} threads, chunks of {chunk:?}"),
                    &got,
                    &want,
                    comm,
                );
            }
        }
    }
    taken_free
}

/// The diagonal M2L with interpolated transfer functions (Helmholtz CH4.2c2) on `diagonal`'s
/// settings, the build not named: the library's default ([`DEFAULT_DIAGONAL_BUILD`],
/// interpolated since CH4.2c2's sign-off) reported, each level's classes within the direct build's
/// (`direct_classes`; C_T = 0.23 against 0.15) and the same on every rank, the errors against
/// the direct sum within [`DIAGONAL_RATIO`] of Rotation's (`plain`), 2 threads bit for bit one,
/// on P ranks the one-rank FMM bit for bit.
fn check_interpolated(
    what: &str,
    scenario: &Scenario,
    diagonal: &HelmholtzFmmBuilder,
    inputs: &Inputs,
    direct_classes: &[Vec<usize>],
    plain: (f64, f64),
    comm: &SimpleCommunicator,
) {
    let kind = scenario.charges;
    let what = format!("{what}, interpolated");
    let interpolated = diagonal.clone();
    let (fmm, want) = run(&interpolated, inputs, kind, comm);
    let op = fmm.operator();
    assert_eq!(DEFAULT_DIAGONAL_BUILD, TransferBuild::Interpolated);
    assert_eq!(
        op.diagonal_build(),
        TransferBuild::Interpolated,
        "{what}: the default"
    );
    let classes: Vec<Vec<usize>> = (0..fmm.nlevels())
        .map(|l| op.diagonal_classes(l).unwrap_or_default())
        .collect();
    drop(fmm);
    let within = classes
        .iter()
        .zip(direct_classes)
        .all(|(i, d)| i.iter().all(|c| d.contains(c)));
    assert!(
        everywhere(within, comm),
        "{what}: classes {classes:?} not within the direct build's {direct_classes:?}"
    );
    let flat: Vec<u64> = classes
        .iter()
        .flat_map(|level| {
            let mut row = [0u64; 34];
            for &c in level {
                row[c] = 1;
            }
            row
        })
        .collect();
    let mut agreed = flat.clone();
    comm.all_reduce_into(&flat[..], &mut agreed[..], SystemOperation::max());
    assert!(
        everywhere(agreed == flat, comm),
        "{what}: the classes differ between ranks"
    );
    assert!(
        classes.iter().any(|c| !c.is_empty()),
        "{what}: no level diagonal"
    );
    let errors = errors(scenario, inputs, &want, 0, comm);
    if comm.rank() == 0 {
        let counts: Vec<(usize, usize, usize)> = classes
            .iter()
            .zip(direct_classes)
            .enumerate()
            .filter(|(_, (i, d))| !i.is_empty() || !d.is_empty())
            .map(|(l, (i, d))| (l, i.len(), d.len()))
            .collect();
        println!(
            "{what}: (level, classes, direct's) {counts:?}; φ {:.3e} against {:.3e} ({:.3}×), ∇φ \
             {:.3e} against {:.3e}",
            errors.0,
            plain.0,
            errors.0 / plain.0,
            errors.1,
            plain.1,
        );
    }
    assert!(
        errors.0 <= DIAGONAL_RATIO * plain.0 && errors.1 <= DIAGONAL_RATIO * plain.1,
        "{what}: errors {errors:?} against {plain:?} without the diagonal form"
    );
    let (fmm, two) = run(&interpolated.clone().threads(2), inputs, kind, comm);
    drop(fmm);
    assert_same(&format!("{what} at 2 threads"), &two, &want, comm);
    check_one_rank(&what, &interpolated, inputs, kind, &want, comm);
}

/// Checks the diagonal level call's times and counts (Helmholtz CH4.2b2) after the evaluations
/// of `fmm` on `threads` threads with `vectors` vectors: on every diagonal level where this rank
/// has V pairs both passes timed and within the level's M2L time, S → F, the pairs and F → R
/// timed, the RCR pairs exactly where some pair stayed RCR, the threads' parts within the
/// threads × the passes; the parts zero on every other level; each level call converts back
/// one accumulator per target with a diagonal pair and vector (all its V rows where the level
/// takes every class, at most those elsewhere); the build times and the multiply-adds on the
/// diagonal levels alone; on every rank or none.
fn check_diagonal_times(
    what: &str,
    fmm: &HelmholtzFmm<'_, f64>,
    threads: usize,
    vectors: usize,
    comm: &SimpleCommunicator,
) {
    let op = fmm.operator();
    let (times, parts, counts) = (op.m2l_times(), op.diagonal_times(), op.diagonal_counts());
    let mut problems: Vec<String> = Vec::new();
    if parts.len() != fmm.nlevels() || counts.back.len() != fmm.nlevels() {
        problems.push(format!("{} levels of times", parts.len()));
    }
    for level in 0..fmm.nlevels().min(parts.len()) {
        let t = parts[level];
        let view = fmm.plan().level(level).v();
        let rows = (0..view.nrows())
            .filter(|&r| !view.row(r).0.is_empty())
            .count();
        let calls = counts.level_calls[level];
        let built = op.diagonal_build_time(level);
        let adds = op.diagonal_multiply_adds(level);
        match op.diagonal_classes(level) {
            Some(classes) => {
                let thread_time = t.s2f + t.pairs + t.f2r + t.rcr;
                let wall = threads as u32 * (t.signatures + t.targets);
                let (diagonal, rcr) = (
                    counts.diagonal_pairs[level] > 0,
                    counts.rcr_pairs[level] > 0,
                );
                let ok = if rows == 0 {
                    thread_time.is_zero() && counts.back[level] == 0
                } else {
                    !t.targets.is_zero()
                        && t.signatures + t.targets <= times[level]
                        && t.signatures.is_zero() != diagonal
                        && t.s2f.is_zero() != diagonal
                        && t.pairs.is_zero() != diagonal
                        && t.f2r.is_zero() != diagonal
                        && (counts.back[level] > 0) == diagonal
                        && t.rcr.is_zero() != rcr
                        && thread_time <= wall
                        && match classes.len() {
                            34 => counts.back[level] == calls * rows * vectors,
                            _ => counts.back[level] <= calls * rows * vectors,
                        }
                };
                if !ok || built.is_none_or(|(w, c)| w.is_zero() || c.is_zero()) || adds.is_none() {
                    problems.push(format!(
                        "level {level}: {t:?}, M2L {:?}, {rows} rows, {calls} calls, back {}, \
                         RCR pairs {}, built {built:?}, multiply-adds {adds:?}",
                        times[level], counts.back[level], counts.rcr_pairs[level]
                    ));
                }
            }
            None => {
                if t != DiagonalTimes::default() || built.is_some() || adds.is_some() {
                    problems.push(format!("level {level} not diagonal: {t:?}, {built:?}"));
                }
            }
        }
    }
    assert!(
        everywhere(problems.is_empty(), comm),
        "{what} at {threads} threads: the diagonal times (rank {}: {problems:?})",
        comm.rank()
    );
}

/// The scenarios of [`check_diagonal`], at ε = 10⁻⁴: the cube at complex k, whose levels 2 and
/// 3 (k a 8.7 and 4.3) take every class, and the clusters at kD = 20, whose level 4 (k a 1.1)
/// leaves the nearest classes on RCR.
fn diagonal_scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "cube, kD = 40 + 2i, ε = 1e-4, complex charges, gradients, rank 0 empty",
            layout: Layout::Cube,
            n: (1500, None),
            k: (40.0, 2.0),
            degrees: Degrees::Tolerance(1e-4),
            bound: ERROR_FACTOR * 1e-4,
            charges: ChargeKind::Complex,
            gradients: true,
            debug: true,
            empty_rank0: true,
        },
        Scenario {
            name: "clusters, kD = 20, ε = 1e-4, real charges",
            layout: Layout::Clusters,
            n: (1200, None),
            k: (20.0, 0.0),
            degrees: Degrees::Tolerance(1e-4),
            bound: ERROR_FACTOR * 1e-4,
            charges: ChargeKind::Real,
            gradients: false,
            debug: true,
            empty_rank0: false,
        },
    ]
}

/// Two vectors: each bit for bit its one-vector FMM, the second against its direct sum;
/// and a reused output.
fn check_vectors(scenario: &Scenario, comm: &SimpleCommunicator) {
    let inputs = Inputs::new(scenario, 2, comm);
    let builder = builder(scenario);
    let kind = scenario.charges;
    let (mut fmm, both) = run(&builder.clone().vectors(2), &inputs, kind, comm);
    for v in 0..2 {
        let one = Inputs {
            sources: inputs.sources.clone(),
            targets: inputs.targets.clone(),
            charges: vec![inputs.charges[v].clone()],
        };
        let (_, single) = run(&builder, &one, kind, comm);
        let column = |output: &Output<f64>, v: usize| {
            let phi = output.potential_of(v);
            let mut bits: Vec<u64> = phi
                .re
                .iter()
                .chain(phi.im.unwrap_or(&[]))
                .map(|x| x.to_bits())
                .collect();
            if let Some(g) = output.gradient_of(v) {
                let im = g.im.unwrap_or(&[]);
                bits.extend(g.re.iter().chain(im).flatten().map(|x| x.to_bits()));
            }
            bits
        };
        let same = column(&both, v) == column(&single, 0);
        assert!(
            everywhere(same, comm),
            "{}: vector {v} of two is not the one-vector FMM bit for bit",
            scenario.name
        );
    }
    // Vector 1 against its own direct sum.
    let (phi, grad) = errors(scenario, &inputs, &both, 1, comm);
    assert!(
        phi < scenario.bound && grad < scenario.bound,
        "{}: vector 1 of two: φ {phi:.2e}, ∇φ {grad:.2e} against the direct sum",
        scenario.name
    );
    // A reused output of another path is replaced; one of this path is rewritten.
    let mut reused = Output::<f64>::new(
        &nd_fmm_exec::helmholtz::Densities::new(
            nd_fmm_exec::helmholtz::WavenumberKind::Real,
            ChargeKind::Complex,
            1,
        ),
        3,
        false,
    );
    for _ in 0..2 {
        fmm.evaluate_into(&inputs.charges(kind), &mut reused)
            .expect("evaluate_into");
        assert_same(
            &format!("{}: evaluate_into", scenario.name),
            &reused,
            &both,
            comm,
        );
    }
    if comm.rank() == 0 {
        println!("{}: two vectors passed", scenario.name);
    }
}

/// Real charges against complex ones with Im q = 0.
fn check_real_charges(scenario: &Scenario, comm: &SimpleCommunicator) {
    let real = Scenario {
        charges: ChargeKind::Real,
        ..scenario.clone()
    };
    let inputs = Inputs::new(&real, 1, comm);
    let (_, as_real) = run(&builder(&real), &inputs, ChargeKind::Real, comm);
    let complex = builder(&Scenario {
        charges: ChargeKind::Complex,
        ..scenario.clone()
    });
    let (_, as_complex) = run(&complex, &inputs, ChargeKind::Complex, comm);
    let diff = |a: &[f64], b: &[f64]| -> [f64; 2] {
        let d = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        [d, b.iter().map(|y| y * y).sum()]
    };
    let [d, n] = diff(&as_real.potential, &as_complex.potential).map(|s| global_sum(s, comm));
    let relative = (d / n).sqrt();
    if comm.rank() == 0 {
        println!(
            "{}: real charges against complex ones with Im q = 0: relative L2 {relative:.1e}",
            scenario.name
        );
    }
    assert!(
        relative < 1e-13,
        "{}: real charges differ from complex ones with Im q = 0 by {relative:.1e}",
        scenario.name
    );
}

/// The errors, each on every rank alike (or, for one rank's input, on that rank and
/// `OtherRank` elsewhere).
fn check_errors(comm: &SimpleCommunicator) {
    let scenario = &scenarios()[1];
    let inputs = Inputs::new(scenario, 1, comm);
    let ok = builder(scenario);
    let rank = comm.rank();

    let yukawa =
        HelmholtzFmmBuilder::new(Wavenumber::new(0.0, 5.0).unwrap(), Degrees::Tolerance(1e-3));
    let error = yukawa
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .err();
    assert!(
        matches!(
            error,
            Some(HelmholtzFmmError::Operator(OperatorError::Yukawa))
        ),
        "rank {rank}: Yukawa: {error:?}"
    );

    let k = Wavenumber::new(1.0, 0.0).unwrap();
    let error = HelmholtzFmmBuilder::new(k, Degrees::Tolerance(2.0))
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .err();
    assert!(
        matches!(error, Some(HelmholtzFmmError::Degree(_))),
        "rank {rank}: a tolerance of 2: {error:?}"
    );

    let error = builder(scenario)
        .c_leaf(0.0)
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .err();
    assert!(
        matches!(error, Some(HelmholtzFmmError::LeafWidth(_))),
        "rank {rank}: c_leaf = 0: {error:?}"
    );

    // A V crossover below 0 or not a number (CH4.0d2).
    for n in [-1.0, f64::NAN] {
        let error = builder(scenario)
            .near_v(VNear::Crossover(n))
            .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
            .err();
        assert!(
            matches!(error, Some(HelmholtzFmmError::VCrossover(_))),
            "rank {rank}: a V crossover of {n}: {error:?}"
        );
    }

    // W and X crossovers below 0 or not a number (CH4.0d3).
    for (w, x) in [(-1.0, 1.0), (1.0, f64::NAN)] {
        let error = builder(scenario)
            .near_wx(WxNear::Crossovers { w, x })
            .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
            .err();
        assert!(
            matches!(error, Some(HelmholtzFmmError::WxCrossover(..))),
            "rank {rank}: W and X crossovers of {w} and {x}: {error:?}"
        );
    }

    // An ISA this machine lacks, on every rank: its own error everywhere.
    if let Some(isa) = Isa::all().find(|isa| !isa.is_available()) {
        let error = builder(scenario)
            .p2p_kernel(P2pChoice::Isa(isa))
            .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
            .err();
        assert!(
            matches!(
                error,
                Some(HelmholtzFmmError::Build(FmmError::InvalidSettings(
                    SettingsError::P2pIsaUnavailable { isa: refused }
                ))) if refused == isa
            ),
            "rank {rank}: the unavailable ISA {isa}: {error:?}"
        );
    }

    // A point that is not finite on rank 0 only: its error there, `OtherRank` elsewhere.
    let mut bad = inputs.sources.clone();
    if rank == 0 {
        bad[0][1] = f64::NAN;
    }
    let error = ok.build::<f64, _>(&bad, &inputs.targets, comm).err();
    if rank == 0 {
        assert!(
            matches!(
                error,
                Some(HelmholtzFmmError::Build(
                    nd_fmm_exec::fmm::FmmError::NonFinitePoint { .. }
                ))
            ),
            "rank 0: a point that is not finite: {error:?}"
        );
    } else {
        assert!(
            matches!(error, Some(HelmholtzFmmError::OtherRank)),
            "rank {rank}: another rank's point that is not finite: {error:?}"
        );
    }

    let error = ok.build::<f64, _>(&[], &[], comm).err();
    assert!(
        matches!(
            error,
            Some(HelmholtzFmmError::Build(
                nd_fmm_exec::fmm::FmmError::NoPoints
            ))
        ),
        "rank {rank}: no points: {error:?}"
    );

    let mut fmm = ok
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .unwrap();
    let error = fmm.evaluate(&[]).err();
    assert!(
        matches!(
            error,
            Some(HelmholtzFmmError::Charges(DensityError::VectorCount { .. }))
        ),
        "rank {rank}: no charge vector: {error:?}"
    );
    let (re, _) = &inputs.charges[0];
    let error = fmm.evaluate(&[Charges::Real(re)]).err();
    assert!(
        matches!(
            error,
            Some(HelmholtzFmmError::Charges(DensityError::ChargeKind { .. }))
        ),
        "rank {rank}: real charges to a complex FMM: {error:?}"
    );
    if comm.size() > 1 {
        // Rank 0 alone passes a short vector.
        let (re, im) = &inputs.charges[0];
        let n = if rank == 0 { re.len() - 1 } else { re.len() };
        let error = fmm
            .evaluate(&[Charges::Complex {
                re: &re[..n],
                im: &im[..n],
            }])
            .err();
        if rank == 0 {
            assert!(
                matches!(
                    error,
                    Some(HelmholtzFmmError::Charges(DensityError::Length { .. }))
                ),
                "rank 0: a short vector: {error:?}"
            );
        } else {
            assert!(
                matches!(error, Some(HelmholtzFmmError::OtherRank)),
                "rank {rank}: another rank's short vector: {error:?}"
            );
        }
    }
    // The FMM still evaluates after refused inputs.
    fmm.evaluate(&inputs.charges(ChargeKind::Complex))
        .expect("an evaluation after the errors");
    if rank == 0 {
        println!("errors passed");
    }
}

/// Asserts on every rank that `got` equals `want` as values (−0 and +0 equal) on every
/// rank: the empty-box rule leaves out exact zeros only.
fn assert_same_values(
    what: &str,
    got: &Output<f64>,
    want: &Output<f64>,
    comm: &SimpleCommunicator,
) {
    let values = |output: &Output<f64>| {
        let mut values = output.potential.clone();
        if let Some(gradient) = &output.gradient {
            values.extend(gradient.as_flattened());
        }
        values
    };
    let same = values(got) == values(want);
    if !same {
        eprintln!("rank {}: {what}: the values differ", comm.rank());
    }
    assert!(
        everywhere(same, comm),
        "{what}: not the same values on every rank"
    );
}

/// The empty-box rule (Helmholtz CH4.0d1; `HelmholtzFmmBuilder::empty_boxes`, on by
/// default): under Rotation, Batched, PlaneWave and Auto with the term off (whose levels'
/// M2L do not depend on the rule) the output equals the rule off's as values (−0 and +0
/// equal), at 2 threads the one-thread output bit for bit and on P ranks the one-rank FMM's;
/// under the default the same where every level runs the M2L it runs with the rule off.
/// Returns what the rule left out over every rank, `[boxes without targets, entries
/// without sources]`.
fn check_empty_boxes(scenario: &Scenario, comm: &SimpleCommunicator) -> [usize; 2] {
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let mut left_out = [0usize; 2];
    let settings: [(&str, HelmholtzFmmBuilder); 5] = [
        ("Rotation", builder(scenario)),
        (
            "Batched",
            default_builder(scenario).m2l(HelmholtzM2l::Batched),
        ),
        (
            "PlaneWave",
            default_builder(scenario).m2l(HelmholtzM2l::PlaneWave),
        ),
        (
            "Auto without the term",
            default_builder(scenario).batched_term(false),
        ),
        ("the default", default_builder(scenario)),
    ];
    for (setting, on) in settings {
        let name = &format!("{}, empty boxes, {setting}", scenario.name);
        let (fmm, output) = run(&on, &inputs, kind, comm);
        assert!(fmm.empty_boxes(), "{name}: the rule is on by default");
        let off = on.clone().empty_boxes(false);
        let (fmm_off, output_off) = run(&off, &inputs, kind, comm);
        assert!(!fmm_off.empty_boxes(), "{name}");
        assert_eq!(fmm_off.plan().empty_boxes(), Default::default(), "{name}");
        let same_m2l = (0..fmm.nlevels())
            .all(|level| fmm.operator().m2l_on(level) == fmm_off.operator().m2l_on(level));
        if setting != "the default" || everywhere(same_m2l, comm) {
            assert_same_values(
                &format!("{name}: against the rule off"),
                &output,
                &output_off,
                comm,
            );
        } else if comm.rank() == 0 {
            println!("{name}: a level's M2L differs from the rule off's, values not compared");
        }
        let (_, threaded) = run(&on.clone().threads(2), &inputs, kind, comm);
        assert_same(&format!("{name}: 2 threads"), &threaded, &output, comm);
        check_one_rank(name, &on, &inputs, kind, &output, comm);
        if setting == "Rotation" {
            let e = fmm.plan().empty_boxes();
            let local = [
                e.target_boxes,
                e.v_without_sources + e.w_without_sources + e.m2m_without_sources,
            ];
            comm.all_reduce_into(&local[..], &mut left_out[..], SystemOperation::sum());
        }
    }
    if comm.rank() == 0 {
        println!(
            "{}, empty boxes: {} boxes without targets, {} entries without sources",
            scenario.name, left_out[0], left_out[1]
        );
    }
    left_out
}

/// The leaf size and c_leaf (Helmholtz CH4.0d3): under the default M2L, at 8 and 512 points
/// per leaf and c_leaf 1 and 2 (on the sphere the leaf rule binds at 1, and 2 lets the
/// leaves be one level coarser), the errors against the direct sum within the scenario's
/// bound, the settings reported as built, the leaves by level summing to the leaves, 2
/// threads the one-thread output bit for bit and on P ranks the one-rank FMM's bit for bit:
/// deterministic and rank-invariant at each leaf size; and the other kinds' level calls timed
/// (`HelmholtzOperator::kind_times`).
fn check_leaf_sizes(scenario: &Scenario, comm: &SimpleCommunicator) {
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let mut levels = Vec::new();
    for (points, c_leaf) in [(8, 1.0), (512, 1.0), (512, 2.0)] {
        let name = &format!(
            "{}, {points} points per leaf, c_leaf {c_leaf}",
            scenario.name
        );
        let built = default_builder(scenario)
            .max_points_per_leaf(points)
            .c_leaf(c_leaf);
        let (fmm, output) = run(&built, &inputs, kind, comm);
        assert_eq!(fmm.max_points_per_leaf(), points, "{name}");
        assert_eq!(fmm.c_leaf(), c_leaf, "{name}");
        let by_level = fmm.leaves_by_level();
        assert_eq!(by_level.len(), fmm.nlevels(), "{name}");
        assert_eq!(
            by_level.iter().map(|c| c[0]).sum::<usize>(),
            fmm.nleaves(),
            "{name}"
        );
        assert!(
            by_level.iter().all(|c| c[1] <= c[0]),
            "{name}: {by_level:?}"
        );
        // The other kinds timed (CH4.0d3), on this rank: L2P on every level with a leaf
        // holding a point (the sphere's targets are its sources, so every such leaf holds a
        // target; with separate targets a leaf of sources alone runs no L2P under the
        // empty-box rule), P2L on every level whose plan has X entries; agreed.
        let times = fmm.operator().kind_times();
        let timed = times.len() == fmm.nlevels()
            && by_level.iter().enumerate().all(|(level, count)| {
                (count[1] == 0 || !times[level][3].is_zero())
                    && (fmm.plan().level(level).x().is_empty() || !times[level][4].is_zero())
            });
        if !timed {
            eprintln!("rank {}: {name}: kind times {times:?}", comm.rank());
        }
        assert!(everywhere(timed, comm), "{name}: a level call not timed");
        let (phi, grad) = errors(scenario, &inputs, &output, 0, comm);
        if comm.rank() == 0 {
            println!(
                "{name}: {} levels, leaf level {}, largest leaf {}: φ {phi:.2e}, ∇φ {grad:.2e}",
                fmm.nlevels(),
                fmm.min_leaf_level(),
                fmm.max_leaf_points(),
            );
        }
        assert!(
            phi < scenario.bound && grad < scenario.bound,
            "{name}: φ {phi:.2e}, ∇φ {grad:.2e}, bound {:.1e}",
            scenario.bound
        );
        levels.push(fmm.min_leaf_level());
        drop(fmm);
        let (_, threaded) = run(&built.clone().threads(2), &inputs, kind, comm);
        assert_same(&format!("{name}: 2 threads"), &threaded, &output, comm);
        check_one_rank(name, &built, &inputs, kind, &output, comm);
    }
    assert_eq!(
        levels[2] + 1,
        levels[1],
        "c_leaf 2: the leaf rule's level one coarser"
    );
}

/// The library's defaults of Helmholtz CH4.0d3: the builder naming nothing but the scenario
/// builds the leaf size of `default_max_points_per_leaf` (the scenario's tolerance, or
/// none for named degrees, and this machine's ISA), `DEFAULT_C_LEAF` and the V, W and X
/// rules of the cost model, and its output is that of the same settings named bit for bit;
/// the errors against the direct sum within the scenario's bound, 2 threads and on P ranks
/// the one-rank FMM bit for bit. Since Helmholtz CH4.0h also the build by part
/// ([`check_build_timings`]); the 2 threads build the tables on the pool, the one thread
/// serially, so the two outputs' bits also hold the parallel tables to the serial ones.
fn check_defaults(scenario: &Scenario, comm: &SimpleCommunicator) {
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let name = &format!("{}, the defaults", scenario.name);
    let k = Wavenumber::new(scenario.k.0, scenario.k.1).unwrap();
    let defaults = HelmholtzFmmBuilder::new(k, scenario.degrees.clone())
        .charges(scenario.charges)
        .gradients(scenario.gradients)
        .domain(PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
    let (fmm, output) = run(&defaults, &inputs, kind, comm);
    let tolerance = match scenario.degrees {
        Degrees::Tolerance(epsilon) => Some(epsilon),
        Degrees::PerLevel(_) => None,
    };
    let points = default_max_points_per_leaf(tolerance, Isa::detect());
    assert_eq!(fmm.max_points_per_leaf(), points, "{name}");
    assert_eq!(fmm.c_leaf(), DEFAULT_C_LEAF, "{name}");
    assert_eq!(fmm.near_v(), VNear::CostModel, "{name}");
    assert_eq!(fmm.near_wx(), WxNear::CostModel, "{name}");
    check_build_timings(name, &defaults, &inputs, comm);
    let (phi, grad) = errors(scenario, &inputs, &output, 0, comm);
    if comm.rank() == 0 {
        println!(
            "{name}: {points} points per leaf, {} levels, leaf level {}: φ {phi:.2e}, ∇φ \
             {grad:.2e}",
            fmm.nlevels(),
            fmm.min_leaf_level(),
        );
    }
    assert!(
        phi < scenario.bound && grad < scenario.bound,
        "{name}: φ {phi:.2e}, ∇φ {grad:.2e}, bound {:.1e}",
        scenario.bound
    );
    drop(fmm);
    let named = default_builder(scenario)
        .max_points_per_leaf(points)
        .c_leaf(DEFAULT_C_LEAF)
        .near_v(VNear::CostModel)
        .near_wx(WxNear::CostModel);
    let (_, output_named) = run(&named, &inputs, kind, comm);
    assert_same(&format!("{name}: named"), &output_named, &output, comm);
    let (_, threaded) = run(&defaults.clone().threads(2), &inputs, kind, comm);
    assert_same(&format!("{name}: 2 threads"), &threaded, &output, comm);
    check_one_rank(name, &defaults, &inputs, kind, &output, comm);
}

/// The build by part (Helmholtz CH4.0h; `HelmholtzFmm::build_timings`) of `builder`, built
/// once more under a wall clock: the octree, the routes, the plan, the operator and its RCR
/// tables timed, the tables within the operator, the steps' sum within the wall time of
/// `build`; and the operator's RCR tables store M2L blocks only on the levels whose M2L is
/// RCR's. On every rank or none.
fn check_build_timings(
    name: &str,
    builder: &HelmholtzFmmBuilder,
    inputs: &Inputs,
    comm: &SimpleCommunicator,
) {
    let start = Instant::now();
    let fmm: HelmholtzFmm<'_, f64> = builder
        .build(&inputs.sources, &inputs.targets, comm)
        .unwrap_or_else(|error| panic!("rank {}: the build: {error}", comm.rank()));
    let wall = start.elapsed();
    let timings = fmm.build_timings();
    let timed = [
        timings.octree,
        timings.routes,
        timings.plan,
        timings.operator,
        timings.tables.rcr,
        timings.evaluator,
    ]
    .iter()
    .all(|&t| t > Duration::ZERO);
    let within = timings.tables.total() <= timings.operator && timings.total() <= wall;
    let op = fmm.operator();
    let blocks = (0..fmm.nlevels())
        .all(|l| !op.tables().m2l_blocks_stored(l) || op.m2l_on(l) == HelmholtzM2l::Rotation);
    assert!(
        everywhere(timed && within && blocks, comm),
        "{name}: the build by part (rank {}: timed {timed}, within {within}, blocks {blocks}: \
         {timings:?})",
        comm.rank()
    );
    if comm.rank() == 0 {
        println!(
            "{name}: build {:.3} s, the operator {:.3} s (RCR tables {:.3} s)",
            timings.total().as_secs_f64(),
            timings.operator.as_secs_f64(),
            timings.tables.rcr.as_secs_f64()
        );
    }
}

/// The errors of the V pairs' near rule may be this many times the rule off's (Helmholtz
/// CH4.0d2; the near rule's bound of Phase 6 T12, `NEAR_ERROR_BOUND` of `tests/mpi_exec.rs`):
/// a converted pair is exact, but the truncation errors of pairs may cancel.
const NEAR_ERROR_BOUND: f64 = 1.1;

/// The V pairs between leaves as near pairs (Helmholtz CH4.0d2;
/// `HelmholtzFmmBuilder::near_v`), under Rotation, PlaneWave (whose levels with waves decide
/// per parent and source: converted alone, a pair's source would still reach its target
/// through the parent's waves), the default and Auto with waves from degree 0, at a
/// crossover of n_s n_t
/// that converts some pairs and keeps others and at one that converts every V pair between
/// leaves, and with the cost model's crossovers: the errors against the direct sum within
/// [`NEAR_ERROR_BOUND`] of the rule off's,
/// 2 threads the one-thread output bit for bit, on P ranks the one-rank FMM's of the same
/// rule bit for bit, and the ghost leaves' points received at least the rule off's. A
/// crossover of 0 converts nothing: the rule off's output bit for bit. Returns the V pairs
/// between leaves offered and converted at the middle crossover under Rotation, the source
/// points received beyond the rule off's, and the V pairs offered and converted under
/// PlaneWave, over every rank.
fn check_near_v(scenario: &Scenario, crossover: f64, comm: &SimpleCommunicator) -> [usize; 5] {
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let mut totals = [0usize; 5];
    for (setting, off) in [
        ("Rotation", builder(scenario)),
        (
            "PlaneWave",
            default_builder(scenario).m2l(HelmholtzM2l::PlaneWave),
        ),
        ("the default", default_builder(scenario)),
        // Auto with waves from degree 0, so that Auto's levels with waves (predicted before
        // the plan, asserted against the operator's) take part at ε = 10⁻³.
        (
            "Auto with waves",
            default_builder(scenario).m2l_plane_wave_min_degree(0),
        ),
    ] {
        let (fmm_off, output_off) = run(&off, &inputs, kind, comm);
        assert_eq!(fmm_off.near_v(), VNear::Off, "the rule off as named");
        if setting == "Auto with waves" || setting == "PlaneWave" {
            let waves = (0..fmm_off.nlevels())
                .any(|l| fmm_off.operator().plane_wave_precision(l).is_some());
            assert!(waves, "{}, {setting}: a level with waves", scenario.name);
        }
        let (phi_off, grad_off) = errors(scenario, &inputs, &output_off, 0, comm);
        let received_off = fmm_off.source_points_exchanged().1;
        let interactions = |fmm: &HelmholtzFmm<'_, f64>| {
            let local: u64 = fmm.near_interactions().iter().sum();
            let mut total = 0u64;
            comm.all_reduce_into(&local, &mut total, SystemOperation::sum());
            total
        };
        let interactions_off = interactions(&fmm_off);
        drop(fmm_off);

        let name = &format!("{}, V pairs at 0, {setting}", scenario.name);
        let (fmm, output) = run(
            &off.clone().near_v(VNear::Crossover(0.0)),
            &inputs,
            kind,
            comm,
        );
        assert_eq!(fmm.plan().near_pairs().v_converted, 0, "{name}");
        assert_same(name, &output, &output_off, comm);
        drop(fmm);

        let choices = [
            VNear::Crossover(crossover),
            VNear::Crossover(f64::INFINITY),
            VNear::CostModel,
        ];
        for choice in choices {
            let name = &format!("{}, V pairs {choice}, {setting}", scenario.name);
            let on = off.clone().near_v(choice);
            let (fmm, output) = run(&on, &inputs, kind, comm);
            assert_eq!(fmm.near_v(), choice, "{name}");
            let pairs = fmm.plan().near_pairs();
            let (offered, converted) = (pairs.v_leaf_pairs, pairs.v_converted);
            if choice == VNear::Crossover(f64::INFINITY) {
                assert_eq!(converted, offered, "{name}: every V pair between leaves");
            }
            assert_eq!(
                pairs.w_converted + pairs.x_converted,
                0,
                "{name}: V pairs only"
            );
            let (phi, grad) = errors(scenario, &inputs, &output, 0, comm);
            if comm.rank() == 0 {
                println!(
                    "{name}: relative L2 φ {phi:.3e} (off {phi_off:.3e}), ∇φ {grad:.3e} \
                     (off {grad_off:.3e})"
                );
            }
            assert!(
                phi <= NEAR_ERROR_BOUND * phi_off && grad <= NEAR_ERROR_BOUND * grad_off,
                "{name}: φ {phi:.3e}, ∇φ {grad:.3e} against the rule off's {phi_off:.3e}, \
                 {grad_off:.3e}"
            );
            // The P2P level calls timed per level (CH4.0d2): an entry on every level with
            // near rows.
            let times = fmm.operator().p2p_times();
            let ok = times.len() == fmm.nlevels()
                && (0..fmm.nlevels()).all(|level| {
                    fmm.plan().level(level).near().nrows() == 0 || times[level] > Duration::ZERO
                });
            assert!(everywhere(ok, comm), "{name}: P2P times {times:?}");
            // The near interactions grow by n_s n_t per converted pair (summed over the
            // ranks, so the verdict is the same on every rank).
            let more = interactions(&fmm);
            assert!(
                more >= interactions_off,
                "{name}: {more} near interactions, the rule off {interactions_off}"
            );
            let received = fmm.source_points_exchanged().1;
            assert!(
                everywhere(received >= received_off, comm),
                "{name}: fewer ghost points than the rule off's"
            );
            let (_, threaded) = run(&on.clone().threads(2), &inputs, kind, comm);
            assert_same(&format!("{name}: 2 threads"), &threaded, &output, comm);
            check_one_rank(name, &on, &inputs, kind, &output, comm);
            if setting == "Rotation" && choice == VNear::Crossover(crossover) {
                let local = [offered, converted, received - received_off];
                comm.all_reduce_into(&local[..], &mut totals[..3], SystemOperation::sum());
            }
            if setting == "PlaneWave" && choice == VNear::Crossover(crossover) {
                let local = [offered, converted];
                comm.all_reduce_into(&local[..], &mut totals[3..], SystemOperation::sum());
            }
        }
    }
    if comm.rank() == 0 {
        println!(
            "{}, V pairs below {crossover}: {} of {} converted, {} more source points received; \
             under PlaneWave (by parent and source on the levels with waves) {} of {}",
            scenario.name, totals[1], totals[0], totals[2], totals[4], totals[3]
        );
    }
    totals
}

/// The W and X pairs between leaves as near pairs (Helmholtz CH4.0d3;
/// `HelmholtzFmmBuilder::near_wx`), under Rotation and the default, at crossovers that
/// convert some pairs and keep others, at ∞ (every W and X pair between leaves), by the cost
/// model, and by the cost model with the V rule's: the errors against the direct sum within
/// [`NEAR_ERROR_BOUND`] of the rule off's, 2 threads the one-thread output bit for bit, on P
/// ranks the one-rank FMM's of the same rule bit for bit, and the P2L and M2P interactions at
/// most the rule off's. Crossovers of 0 convert nothing: the rule off's output bit for bit.
/// Returns the W pairs between leaves offered and converted and the X pairs offered and
/// converted at the middle crossover under Rotation, over every rank.
fn check_near_wx(scenario: &Scenario, crossover: f64, comm: &SimpleCommunicator) -> [usize; 4] {
    let inputs = Inputs::new(scenario, 1, comm);
    let kind = scenario.charges;
    let mut totals = [0usize; 4];
    let far = |fmm: &HelmholtzFmm<'_, f64>| {
        let local: Vec<u64> = fmm.far_interactions().iter().flatten().copied().collect();
        let mut summed = vec![0u64; local.len()];
        comm.all_reduce_into(&local[..], &mut summed[..], SystemOperation::sum());
        summed.iter().sum::<u64>()
    };
    for (setting, off) in [
        ("Rotation", builder(scenario)),
        ("the default", default_builder(scenario)),
    ] {
        let (fmm_off, output_off) = run(&off, &inputs, kind, comm);
        assert_eq!(fmm_off.near_wx(), WxNear::Off, "the rule off as named");
        let (phi_off, grad_off) = errors(scenario, &inputs, &output_off, 0, comm);
        let far_off = far(&fmm_off);
        drop(fmm_off);

        let name = &format!("{}, W and X pairs at 0, {setting}", scenario.name);
        let zero = WxNear::Crossovers { w: 0.0, x: 0.0 };
        let (fmm, output) = run(&off.clone().near_wx(zero), &inputs, kind, comm);
        let pairs = fmm.plan().near_pairs();
        assert_eq!(pairs.w_converted + pairs.x_converted, 0, "{name}");
        assert_same(name, &output, &output_off, comm);
        drop(fmm);

        let choices = [
            (
                WxNear::Crossovers {
                    w: crossover,
                    x: crossover,
                },
                VNear::Off,
            ),
            (
                WxNear::Crossovers {
                    w: f64::INFINITY,
                    x: f64::INFINITY,
                },
                VNear::Off,
            ),
            (WxNear::CostModel, VNear::Off),
            (WxNear::CostModel, VNear::CostModel),
        ];
        for (choice, near_v) in choices {
            let name = &format!(
                "{}, W and X pairs {choice}, V {near_v}, {setting}",
                scenario.name
            );
            let on = off.clone().near_wx(choice).near_v(near_v);
            let (fmm, output) = run(&on, &inputs, kind, comm);
            assert_eq!(fmm.near_wx(), choice, "{name}");
            let pairs = fmm.plan().near_pairs();
            if matches!(choice, WxNear::Crossovers { w, .. } if w == f64::INFINITY) {
                assert_eq!(
                    pairs.w_converted, pairs.w_leaf_pairs,
                    "{name}: every W pair"
                );
                assert_eq!(
                    pairs.x_converted, pairs.x_leaf_pairs,
                    "{name}: every X pair"
                );
            }
            if near_v == VNear::Off {
                assert_eq!(pairs.v_converted, 0, "{name}: W and X pairs only");
            }
            let (phi, grad) = errors(scenario, &inputs, &output, 0, comm);
            if comm.rank() == 0 {
                println!(
                    "{name}: relative L2 φ {phi:.3e} (off {phi_off:.3e}), ∇φ {grad:.3e} \
                     (off {grad_off:.3e})"
                );
            }
            assert!(
                phi <= NEAR_ERROR_BOUND * phi_off && grad <= NEAR_ERROR_BOUND * grad_off,
                "{name}: φ {phi:.3e}, ∇φ {grad:.3e} against the rule off's {phi_off:.3e}, \
                 {grad_off:.3e}"
            );
            let far_on = far(&fmm);
            assert!(
                far_on <= far_off,
                "{name}: {far_on} P2L and M2P interactions, off {far_off}"
            );
            let (_, threaded) = run(&on.clone().threads(2), &inputs, kind, comm);
            assert_same(&format!("{name}: 2 threads"), &threaded, &output, comm);
            check_one_rank(name, &on, &inputs, kind, &output, comm);
            if setting == "Rotation"
                && choice
                    == (WxNear::Crossovers {
                        w: crossover,
                        x: crossover,
                    })
            {
                let local = [
                    pairs.w_leaf_pairs,
                    pairs.w_converted,
                    pairs.x_leaf_pairs,
                    pairs.x_converted,
                ];
                comm.all_reduce_into(&local[..], &mut totals[..], SystemOperation::sum());
            }
        }
    }
    if comm.rank() == 0 {
        println!(
            "{}, W and X pairs below {crossover}: W {} of {}, X {} of {} converted",
            scenario.name, totals[1], totals[0], totals[3], totals[2]
        );
    }
    totals
}

/// The scenario of [`check_empty_boxes`] whose leaves the leaf rule fills with empty boxes:
/// points on a sphere at kD = 30, leaves on level 3 or finer, about three quarters of
/// level 3 empty.
fn sphere_scenario() -> Scenario {
    Scenario {
        name: "sphere, kD = 30, ε = 1e-3, complex charges, gradients",
        layout: Layout::Sphere,
        n: (1000, None),
        k: (30.0, 0.0),
        degrees: Degrees::Tolerance(1e-3),
        bound: ERROR_FACTOR * 1e-3,
        charges: ChargeKind::Complex,
        gradients: true,
        debug: true,
        empty_rank0: false,
    }
}

/// The scenarios, small trees (a few thousand points, at most 32 per leaf).
fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "cube, kD = 20 + 2i, ε = 1e-3, complex charges, gradients, rank 0 empty",
            layout: Layout::Cube,
            n: (1000, None),
            k: (20.0, 2.0),
            degrees: Degrees::Tolerance(1e-3),
            bound: ERROR_FACTOR * 1e-3,
            charges: ChargeKind::Complex,
            gradients: true,
            debug: true,
            empty_rank0: true,
        },
        Scenario {
            name: "cube, kD = 1, ε = 1e-3, complex charges, gradients",
            layout: Layout::Cube,
            n: (1500, None),
            k: (1.0, 0.0),
            degrees: Degrees::Tolerance(1e-3),
            bound: ERROR_FACTOR * 1e-3,
            charges: ChargeKind::Complex,
            gradients: true,
            debug: true,
            empty_rank0: false,
        },
        Scenario {
            name: "clusters, kD = 20, p = 8, real charges, separate targets",
            layout: Layout::Clusters,
            n: (1000, Some(700)),
            k: (20.0, 0.0),
            degrees: Degrees::PerLevel(vec![8; 17]),
            bound: P8_BOUND,
            charges: ChargeKind::Real,
            gradients: false,
            debug: true,
            empty_rank0: false,
        },
        Scenario {
            name: "cube, kD = 20 + 2i, p = 8, complex charges, gradients",
            layout: Layout::Cube,
            n: (1500, None),
            k: (20.0, 2.0),
            degrees: Degrees::PerLevel(vec![8; 17]),
            bound: P8_BOUND,
            charges: ChargeKind::Complex,
            gradients: true,
            debug: true,
            empty_rank0: false,
        },
        Scenario {
            name: "cube, kD = 20, ε = 1e-6, complex charges, gradients",
            layout: Layout::Cube,
            n: (2000, None),
            k: (20.0, 0.0),
            degrees: Degrees::Tolerance(1e-6),
            bound: ERROR_FACTOR * 1e-6,
            charges: ChargeKind::Complex,
            gradients: true,
            debug: false,
            empty_rank0: false,
        },
        Scenario {
            name: "clusters, kD = 100, ε = 1e-6, complex charges",
            layout: Layout::Clusters,
            n: (3000, None),
            k: (100.0, 0.0),
            degrees: Degrees::Tolerance(1e-6),
            bound: ERROR_FACTOR * 1e-6,
            charges: ChargeKind::Complex,
            gradients: false,
            debug: false,
            empty_rank0: false,
        },
    ]
}

/// The scenarios of [`check_resolving_degree`], at ε = 10⁻⁷ and 10⁻⁸: the clusters at
/// kD = 1 (their deep levels at κ_l ≤ 0.08) and the cube at complex kD = 20 + 2i.
fn resolving_scenarios() -> Vec<Scenario> {
    let mut list = Vec::new();
    for epsilon in [1e-7, 1e-8] {
        list.push(Scenario {
            name: if epsilon == 1e-7 {
                "clusters, kD = 1, ε = 1e-7, complex charges"
            } else {
                "clusters, kD = 1, ε = 1e-8, complex charges"
            },
            layout: Layout::Clusters,
            n: (1000, None),
            k: (1.0, 0.0),
            degrees: Degrees::Tolerance(epsilon),
            bound: ERROR_FACTOR * epsilon,
            charges: ChargeKind::Complex,
            gradients: false,
            debug: true,
            empty_rank0: false,
        });
        list.push(Scenario {
            name: if epsilon == 1e-7 {
                "cube, kD = 20 + 2i, ε = 1e-7, real charges, gradients"
            } else {
                "cube, kD = 20 + 2i, ε = 1e-8, real charges, gradients"
            },
            layout: Layout::Cube,
            n: (800, None),
            k: (20.0, 2.0),
            degrees: Degrees::Tolerance(epsilon),
            bound: ERROR_FACTOR * epsilon,
            charges: ChargeKind::Real,
            gradients: true,
            debug: true,
            empty_rank0: false,
        });
    }
    list
}

#[test]
fn helmholtz_fmm() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("MPI initialises once per test executable");
    assert!(
        provided >= Threading::Funneled,
        "MPI provides {provided:?}, below Funneled"
    );
    let comm = universe.world();
    let start = Instant::now();
    let debug = cfg!(debug_assertions);
    for scenario in scenarios() {
        if debug && !scenario.debug {
            if comm.rank() == 0 {
                println!("{}: release builds only", scenario.name);
            }
            continue;
        }
        check(&scenario, &comm);
    }
    // The P2P kernels: complex k with complex charges and gradients, real k with real
    // charges and separate targets.
    check_kernels(&scenarios()[0], &comm);
    check_kernels(&scenarios()[2], &comm);
    // The Dense, Classes and Batched M2L: complex k with complex charges, real k with real
    // charges (real multipoles, one plane).
    let mut chunked = false;
    let mut by_term = false;
    for i in [0, 2] {
        let scenario = &scenarios()[i];
        let dense = check_tables(scenario, HelmholtzM2l::Dense, None, &comm);
        check_tables(scenario, HelmholtzM2l::Classes, None, &comm);
        check_tables(scenario, HelmholtzM2l::Batched, Some(&dense), &comm);
        check_auto(scenario, &comm);
        by_term |= check_batched_term(scenario, &comm);
        check_auto_plane_wave(scenario, &comm);
        // The bounded buffer in debug builds on the clusters alone (scenario 2), whose
        // finest level with waves chunks at the floor.
        chunked |= check_plane_wave(scenario, !debug || i == 2, &comm);
    }
    // The term sent a level to Rotation (the clusters' finest level, about one column per
    // read, below m* of every class).
    assert!(by_term, "the term chose Rotation on no level");
    // The exponential M2L below 10⁻⁶ at the resolving degree (CH6.1g), the clusters' deep
    // levels included.
    for scenario in resolving_scenarios() {
        let deep = check_resolving_degree(&scenario, &comm);
        if matches!(scenario.layout, Layout::Clusters) {
            assert!(
                deep <= 0.08,
                "{}: a deep level, |κ_l| {deep}",
                scenario.name
            );
        }
    }
    // The exponential M2L's bounded buffer chunked a level call (CH6.2a): on one rank the
    // clusters' finest level with waves has more sources than the floor holds; on several
    // ranks the one-rank reference of the floor chunks it.
    assert!(
        comm.size() > 1 || chunked,
        "the floor bound chunks a level call"
    );
    // The empty-box rule (CH4.0d1): the sphere under the leaf rule and the clusters with
    // separate targets (boxes with sources and no target, and the reverse).
    let sphere = check_empty_boxes(&sphere_scenario(), &comm);
    check(&sphere_scenario(), &comm);
    let clusters = check_empty_boxes(&scenarios()[2], &comm);
    assert!(
        sphere[0] > 0 && sphere[1] > 0 && clusters[0] > 0 && clusters[1] > 0,
        "the empty-box rule left out nothing: sphere {sphere:?}, clusters {clusters:?}"
    );
    // The V pairs between leaves as near pairs (CH4.0d2): the cube with rank 0 empty and
    // the sphere under the leaf rule.
    for scenario in [&scenarios()[0], &sphere_scenario()] {
        let [offered, converted, received, pw_offered, pw_converted] =
            check_near_v(scenario, 100.0, &comm);
        // On the sphere the levels with waves convert some groups and keep others.
        if matches!(scenario.layout, Layout::Sphere) {
            assert!(
                0 < pw_converted && pw_converted < pw_offered,
                "{}: under PlaneWave {pw_converted} of {pw_offered} converted",
                scenario.name
            );
        }
        assert!(
            0 < converted && converted < offered,
            "{}: the V rule converted nothing or everything ({converted} of {offered})",
            scenario.name
        );
        assert!(
            comm.size() == 1 || received > 0,
            "{}: no ghost leaf's points for a converted V pair",
            scenario.name
        );
    }
    // The leaf size and c_leaf (CH4.0d3): the sphere under the leaf rule.
    check_leaf_sizes(&sphere_scenario(), &comm);
    // The library's defaults (CH4.0d3): a tolerance and named degrees.
    check_defaults(&sphere_scenario(), &comm);
    check_defaults(&scenarios()[2], &comm);
    // The W and X pairs as near pairs (CH4.0d3): the clusters with separate targets and the
    // sphere under the leaf rule, adaptive trees with W and X lists.
    let mut wx = [0usize; 4];
    for scenario in [&scenarios()[2], &sphere_scenario()] {
        let totals = check_near_wx(scenario, 12.0, &comm);
        for (sum, total) in wx.iter_mut().zip(totals) {
            *sum += total;
        }
    }
    assert!(
        0 < wx[1] && wx[1] < wx[0] && 0 < wx[3] && wx[3] < wx[2],
        "the W and X rule converted none or all: W {} of {}, X {} of {}",
        wx[1],
        wx[0],
        wx[3],
        wx[2]
    );
    // The Rotation M2L by pairs (CH4.0i): complex k and charges with rank 0 empty, real k
    // and charges with separate targets.
    check_rcr_split(&scenarios()[0], &comm);
    check_rcr_split(&scenarios()[2], &comm);
    // Two vectors of complex and of real charges.
    check_vectors(&scenarios()[1], &comm);
    check_vectors(&scenarios()[2], &comm);
    check_real_charges(&scenarios()[2], &comm);
    check_real_charges(&scenarios()[3], &comm);
    // The diagonal M2L (CH4.2b1): some level mixes the diagonal form and RCR.
    let mixed: usize = diagonal_scenarios()
        .iter()
        .map(|scenario| check_diagonal(scenario, &comm))
        .sum();
    assert!(mixed > 0, "no level mixed the diagonal form and RCR");
    // The time rule at matched accuracy (CH4.3b2a): the form on a raised level.
    let classes = check_time_rule_raised(&time_rule_scenario(), &comm);
    assert!(classes > 0, "no raised level took the form");
    // The V rule after the plan (CH4.3b2b): the sphere under the leaf rule.
    check_v_after_plan(&sphere_scenario(), &comm);
    // The V rule's price per box on raised levels (CH4.3b2d): the sphere at 10⁻⁷.
    check_v_raised(&raised_v_scenario(), &comm);
    check_errors(&comm);
    if comm.rank() == 0 {
        println!(
            "helmholtz_fmm on {} ranks: {:.1?}",
            comm.size(),
            start.elapsed()
        );
    }
}
